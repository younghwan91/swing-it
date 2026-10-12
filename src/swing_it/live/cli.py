"""``sw-live`` — 스윙 라이브 작업 CLI.

    sw-live <job> [--paper | --dry-run | --execute --confirm-real] [--data-dir data/live_swing] [--asof YYYY-MM-DD]

작업: ``signal`` · ``preopen`` · ``open-check`` · ``monitor`` · ``close`` · ``eod`` · ``status`` · ``reconcile`` ·
``release-order --oid <oid>``(미해결 주문의 예약·종목 차단을 사람이 푼다).

모드(기본 페이퍼 — 이 전략은 아직 표본 밖 판정이 없다, 설계 §0):

* ``--paper``(기본): 실시세(읽기 전용) + 모의 체결. 주문은 프로세스 밖으로 나가지 않는다.
* ``--dry-run``: 실계좌 조회(잔고) + 주문 POST 안 함(``KiwoomBroker(dry_run=True)``).
* ``--execute --confirm-real``: **실주문.** 두 플래그가 다 있어야 한다. 하나만 주면 거부한다(daytrade-it 패턴).
  ``--paper``/``--dry-run`` 과 함께 줄 수 없다. 환경변수로는 켤 수 없다.

종료 코드: 0 정상 · 1 작업 오류 · 2 시간창 밖/잠금 경합/인자 오류 · 3 상태 손상 ·
4 signal 신규 진입 중단(입력 신선도·DB·최신일·애프터마켓 보정 실패 — 보유 종목 결정·자산은 수행됨).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from swing_it.live.config import DEFAULT_RISK, DEFAULT_STRATEGY

__all__ = ["build_parser", "main", "resolve_mode"]

CLI_JOBS = ("signal", "preopen", "open-check", "monitor", "close", "eod", "status", "reconcile", "release-order")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sw-live", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", choices=CLI_JOBS)
    p.add_argument("--paper", action="store_true", help="페이퍼(기본)")
    p.add_argument("--dry-run", action="store_true", help="실계좌 조회만, 주문 POST 안 함")
    p.add_argument("--execute", action="store_true", help="실주문 경로(--confirm-real 과 함께만)")
    p.add_argument("--confirm-real", action="store_true", help="실계좌 실주문 확인(--execute 와 함께만)")
    p.add_argument("--data-dir", default="data/live_swing", help="상태 루트(모드별 하위 디렉터리)")
    p.add_argument("--stop-file", default=DEFAULT_RISK.stop_file, help="전역 정지 파일(신규 진입 정지)")
    p.add_argument("--holidays-file", default=None, help="휴장일 목록(기본 <data-dir>/holidays.txt)")
    p.add_argument("--asof", default=None, help="거래일 YYYY-MM-DD(기본 오늘 KST)")
    p.add_argument("--capital", type=float, default=None, help="전략 자본(원) — 첫 실행에만 쓴다. live 는 필수")
    p.add_argument("--db", default=None, help="signal 작업 DB(기본 KR_QUANT_DB)")
    p.add_argument("--force", action="store_true", help="signal·eod 를 같은 날 다시 돌린다")
    p.add_argument("--once", action="store_true", help="monitor 한 번만 돌고 끝낸다")
    p.add_argument("--ignore-clock", action="store_true", help="작업 시간창 무시(페이퍼 리허설 전용)")
    p.add_argument("--oid", default=None, help="release-order: 풀 주문의 원장 키(L000012)")
    p.add_argument("--strategy-config", default=None,
                   help="버전 있는 전략 설정 JSON(없으면 코드 기본값). 신규 계획에만 적용, 열린 포지션은 계획 시점 설정 고정")
    p.add_argument("--quotes", choices=("kiwoom", "db"), default="kiwoom",
                   help="시세 원천. db = --asof 과거일 재생(minute_bars·daily_bars, 키움 호출 없음, 페이퍼 전용)")
    p.add_argument("--clock", default=None,
                   help="재생 시계 HH:MM(--quotes db 전용). 없으면 작업별 예정 시각(signal 07:35 · preopen 08:35 · "
                        "open-check 09:02 · monitor 09:05~ · close 15:21 · eod 15:40)")
    p.add_argument("--allow-code-mismatch", action="store_true",
                   help="인계 파일의 code_hash 가 지금 코드와 달라도 읽는다(사람이 확인한 경우만)")
    return p


def resolve_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[str, bool]:
    """``(모드, confirm_real)``. 위험한 조합은 ``parser.error`` (종료 코드 2)."""
    if args.confirm_real and not args.execute:
        parser.error("--confirm-real 은 --execute 와 함께만 쓸 수 있다")
    if args.execute and not args.confirm_real:
        parser.error("--execute 는 --confirm-real 과 함께만 쓸 수 있다(실주문은 두 플래그 모두 필요)")
    if args.execute and (args.paper or args.dry_run):
        parser.error("--execute --confirm-real 은 --paper/--dry-run 과 함께 쓸 수 없다")
    if args.paper and args.dry_run:
        parser.error("--paper 와 --dry-run 은 함께 쓸 수 없다")
    if args.ignore_clock and (args.execute or args.dry_run):
        parser.error("--ignore-clock 은 페이퍼 전용이다")
    if args.quotes == "db" and (args.execute or args.dry_run):
        parser.error("--quotes db 는 페이퍼 전용이다(재생)")
    if args.quotes == "db" and not args.asof:
        parser.error("--quotes db 에는 --asof YYYY-MM-DD 가 필요하다")
    if args.clock and args.quotes != "db":
        parser.error("--clock 은 --quotes db 재생 전용이다(실시간 작업의 시간창을 우회하지 않는다)")
    if args.clock:
        hh, _, mm = args.clock.partition(":")
        if not (hh.isdigit() and mm.isdigit() and len(mm) == 2 and 0 <= int(hh) < 24 and 0 <= int(mm) < 60):
            parser.error("--clock 은 HH:MM")
    if args.execute and args.confirm_real:
        return "live", True
    if args.dry_run:
        return "dry-run", False
    return "paper", False


def banner(mode: str, data_dir: Path, stop_file: Path) -> str:
    text = {
        "paper": "PAPER — 실시세 + 모의 체결, 주문은 밖으로 나가지 않는다",
        "dry-run": "DRY-RUN — 실계좌 조회만, 주문 POST 없음",
        "live": "*** LIVE — 실계좌 실주문 (--execute --confirm-real) ***",
    }[mode]
    return f"[sw-live] {text}\n[sw-live] 상태 {data_dir / mode} · 정지 파일 {stop_file}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode, confirm_real = resolve_mode(parser, args)
    # 상대경로는 실행 디렉터리 기준이라 조용히 다른 파일을 볼 수 있다 — 절대경로로 풀어 배너에 찍는다.
    data_dir = Path(args.data_dir).expanduser().resolve()
    stop_file = Path(args.stop_file).expanduser().resolve()
    holidays = Path(args.holidays_file).expanduser().resolve() if args.holidays_file else data_dir / "holidays.txt"
    asof = date.fromisoformat(args.asof) if args.asof else None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    print(banner(mode, data_dir, stop_file), file=sys.stderr, flush=True)

    from swing_it.live.brokers import (  # noqa: PLC0415 — CLI 인자 검증 뒤에만 무거운 import
        REPLAY_JOB_TIMES,
        DBQuoteProvider,
        SimClock,
        build_bundle,
    )
    from swing_it.live.runtime import (  # noqa: PLC0415
        FeaturesSignalSource,
        RuntimeConfig,
        SwingRuntime,
        load_strategy_config,
    )

    strategy, meta = DEFAULT_STRATEGY, {}
    if args.strategy_config:
        strategy, meta = load_strategy_config(args.strategy_config, allow_code_mismatch=bool(args.allow_code_mismatch))
        print(f"[sw-live] 전략 설정 {meta.get('cfg_hash', '')[:8]} ← {args.strategy_config} (valid_from {meta.get('valid_from')})",
              file=sys.stderr, flush=True)
    else:
        print("[sw-live] 전략 설정: DEFAULT(코드 기본값)", file=sys.stderr, flush=True)
    cfg = RuntimeConfig(data_dir=data_dir, mode=mode, capital=args.capital, stop_file=stop_file,
                        holidays_file=holidays, ignore_clock=bool(args.ignore_clock), strategy=strategy,
                        strategy_file=Path(args.strategy_config) if args.strategy_config else None, strategy_meta=meta,
                        quotes_source=args.quotes)

    job = args.job.replace("-", "_")
    clock = None
    if args.quotes == "db":
        clock = SimClock(asof, args.clock or REPLAY_JOB_TIMES.get(job, "16:00"))

    def bundle_factory():
        quotes = None
        if args.quotes == "db":
            from swing_it.storage import connect, db_default  # noqa: PLC0415

            quotes = DBQuoteProvider(connect(args.db or db_default()), asof, clock=clock)
        kw = {"clock": clock} if clock is not None else {}
        return build_bundle(mode, confirm_real=confirm_real, paper_path=cfg.mode_dir / "paper_broker.json", quotes=quotes,
                            **kw)

    if args.job == "release-order" and not args.oid:
        parser.error("release-order 에는 --oid 가 필요하다")
    rt_kw = {"clock": clock, "sleeper": clock.sleep} if clock is not None else {}
    rt = SwingRuntime(cfg, bundle_factory, signal_source=FeaturesSignalSource(cfg.strategy, db=args.db), **rt_kw)
    return rt.run(job, asof, force=args.force, once=args.once, oid=args.oid)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
