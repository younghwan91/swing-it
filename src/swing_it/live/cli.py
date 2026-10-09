"""``sw-live`` — 스윙 라이브 작업 CLI.

    sw-live <job> [--paper | --dry-run | --execute --confirm-real] [--data-dir data/live_swing] [--asof YYYY-MM-DD]

작업: ``signal`` · ``preopen`` · ``open-check`` · ``monitor`` · ``close`` · ``eod`` · ``status`` · ``reconcile`` ·
``release-order --oid <oid>``(미해결 주문의 예약·종목 차단을 사람이 푼다).

모드(기본 페이퍼 — 이 전략은 아직 표본 밖 판정이 없다, 설계 §0):

* ``--paper``(기본): 실시세(읽기 전용) + 모의 체결. 주문은 프로세스 밖으로 나가지 않는다.
* ``--dry-run``: 실계좌 조회(잔고) + 주문 POST 안 함(``KiwoomBroker(dry_run=True)``).
* ``--execute --confirm-real``: **실주문.** 두 플래그가 다 있어야 한다. 하나만 주면 거부한다(daytrade-it 패턴).
  ``--paper``/``--dry-run`` 과 함께 줄 수 없다. 환경변수로는 켤 수 없다.

종료 코드: 0 정상 · 1 작업 오류 · 2 시간창 밖/잠금 경합/인자 오류 · 3 상태 손상.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from swing_it.live.config import DEFAULT_RISK

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

    from swing_it.live.brokers import build_bundle  # noqa: PLC0415 — CLI 인자 검증 뒤에만 무거운 import
    from swing_it.live.runtime import FeaturesSignalSource, RuntimeConfig, SwingRuntime  # noqa: PLC0415

    cfg = RuntimeConfig(data_dir=data_dir, mode=mode, capital=args.capital, stop_file=stop_file,
                        holidays_file=holidays, ignore_clock=bool(args.ignore_clock))

    def bundle_factory():
        return build_bundle(mode, confirm_real=confirm_real, paper_path=cfg.mode_dir / "paper_broker.json")

    if args.job == "release-order" and not args.oid:
        parser.error("release-order 에는 --oid 가 필요하다")
    rt = SwingRuntime(cfg, bundle_factory, signal_source=FeaturesSignalSource(cfg.strategy, db=args.db))
    return rt.run(args.job.replace("-", "_"), asof, force=args.force, once=args.once, oid=args.oid)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
