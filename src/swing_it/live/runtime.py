"""스윙 라이브 런타임 — 크론 작업 8개(signal·preopen·open_check·monitor·close·eod·status·reconcile).

작업마다 새 프로세스다. 공통 골격(:meth:`SwingRuntime.run`): 모드별 단일 잠금(``InstanceLock``) → 스냅샷·장부 복원 →
작업 → 체결 반영 → 스냅샷·페이퍼 브로커 저장. 예외가 나도 저장은 한다(낸 주문을 잊으면 안 된다).

**판단 = 전략 모듈 하나.** 진입 후보·주문 계획·포지션 상태 전이는 전부 :mod:`swing_it.live.strategy`
(``scan``·``plan_entry``·``PositionState.on_fill_open/on_open/on_quote/on_close``) 가 낸다. 여기는 그 결정을 **주문으로
옮기고 체결을 장부에 되돌리는** 배관만 한다.

하루(KST, 크론 예시는 ``deploy/crontab.swing-live.example``):

=============  =========  ====================================================================================
작업            시각        하는 일
=============  =========  ====================================================================================
signal         07:35      DB(NXT 보정 일봉)로 신호일 d 후보 → 계획(``planned``). 최신일 = 직전 거래일·KRX 보정 확인.
                          보유 종목은 d 의 KRX 종가로 돌파 실패(종가 < 피벗, 익절 전)·MA50 추적·최대 보유 상한(→ 오늘 시가
                          매도)·d 장 마감 자산·다일 낙폭 킬. 시간 손절은 없다(설계 §6).
preopen        08:35      휴장·킬 확인 → 계좌 기준선 → 전일 결정 매도(장전 동시호가) → 지정가 매수(장전 동시호가).
open_check     09:02      휴장 판정(ka10081 오늘 봉) → 시가 체결 반영(페이퍼는 시가 주입) → 미체결 매수 취소(추격 없음)
                          → 체결가로 손절·목표 확정, 보유 포지션 시가 판단(갭 손절·갭 익절·추적 매도 결과).
monitor        09:05~     60초 REST 폴링(10:00~10:15 은 180초). 손절·익절·본전 → 지정가 매도, 미체결이면 1분 뒤 더
               15:19      싸게(0.5%씩, 상한 3%) 다시 낸다(daytrade-it 패턴).
close          15:21      오늘 못 판 매도 의도를 종가 동시호가에(마지막 기회). 종가 기준 결정은 하지 않는다.
eod            15:40      종가 동시호가 체결 반영·남은 주문 만료·잠정 보고만. (MA50 추적·자산·낙폭 킬은 다음날 signal.)
status         아무 때     스냅샷 요약(브로커 호출 없음).
reconcile      아무 때     장부 vs 계좌(기준선 감안) — **보고만** 한다.
=============  =========  ====================================================================================

안전 규칙(바꾸지 않는다):

* **진입만 막고 청산은 막지 않는다.** 정지 파일(``data/swing_STOP`` 전역 + 모드별), 다일 낙폭 킬, 일 손실 킬은 매수만
  막는다. 매도는 코어 OMS 규약대로 킬·가드를 거치지 않는다.
* **사용자 주식은 절대 팔지 않는다.** 매도 수량은 이 전략 장부(``PositionBook``) 보유 − 걸린 자기 매도 잔량까지만이고,
  장부에 없는 종목은 팔지 않는다. 아침 기준선에서 사용자 몫이 있는 종목은 사지 않는다.
* **시간창 밖이면 주문하지 않는다.** 장전 매수를 10시에 내면 추격 매수가 된다 — dry-run·live 는 작업별 시간창을 강제하고,
  페이퍼만 ``ignore_clock`` 으로 리허설할 수 있다.
* **KRX 정규장만 본다.** 시세는 6자리 코드(``ka10001``) = KRX, 주문 거래소 ``KRX``(코어 ``OrderIntent`` 기본).
  2026-09-14 부터 키움 일봉에 NXT 야간 체결이 섞여, 일봉(``ka10081``)은 **날짜(휴장 판정)에만** 쓰고 가격에는 쓰지 않는다.
  MA50 은 signal 이 DB 보정본(``swing_it.krx_session``)에서 가져온 KRX 종가 + eod 의 KRX 종가로만 잰다.
* 체결 판정: 페이퍼는 모의 브로커 체결 그대로, 키움은 ``ka10076`` 미검증이라 **잔고 차이**로 판정한다(가격은 추정 —
  동시호가는 단일가, 장중 매도는 max(지정가, 현재가)). 설명 안 되는 잔고 차이는 고치지 않고 경보만 한다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time as _time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.events import Fill, normalize_ord_no
from krx_quant_core.execution.guards import OrderGuard
from krx_quant_core.execution.oms import AlreadyRunning, InstanceLock, OrderManager
from krx_quant_core.execution.events import OpenOrder
from krx_quant_core.execution.orders import OrderIntent
from krx_quant_core.market.limits import limit_down_price
from krx_quant_core.market.session import KST, now_kst
from krx_quant_core.market.ticks import round_to_tick, round_to_tick_up, shift_ticks
from krx_quant_core.risk.killswitch import KillSwitch, KillSwitchConfig

from swing_it.live.brokers import (
    BrokerBundle,
    QuoteSnap,
    auction_ts,
    entry_guard_config,
    feed_auction,
    feed_quote,
    session_date,
)
from swing_it.live.config import DEFAULT_RISK, DEFAULT_STRATEGY, RiskConfig, StrategyConfig
from swing_it.live.sizing import allocate
from swing_it.live.state import (
    LIVE_ORDER_STATUSES,
    LiveState,
    strategy_hash,
    trade_stats,
    OrderRec,
    PositionRec,
    StateCorrupt,
    StateStore,
    compute_baseline_user,
    drawdown_check,
    strategy_holdings,
)
from swing_it.live.strategy import Action, Candidate, EntryPlan, PositionState, entry_limit, plan_entry

__all__ = [
    "JOBS",
    "JobWindowError",
    "RuntimeConfig",
    "SignalBatch",
    "SignalSource",
    "SwingRuntime",
    "notify",
]

log = logging.getLogger("swing_it.live")

JOBS = ("signal", "preopen", "open_check", "monitor", "close", "eod", "status", "reconcile", "release_order")

#: 작업별 허용 시각(KST, [시작, 끝)). dry-run·live 는 강제, 페이퍼는 ``ignore_clock`` 으로 끌 수 있다.
JOB_WINDOWS: dict[str, tuple[time, time]] = {
    "signal": (time(7, 31), time(8, 29)),
    "preopen": (time(8, 30), time(8, 59, 30)),
    "open_check": (time(9, 0, 30), time(9, 30)),
    "monitor": (time(9, 0, 30), time(15, 20)),
    "close": (time(15, 20), time(15, 29)),
    "eod": (time(15, 31), time(16, 0)),
}

#: 주문을 내는 작업 — 달력(주말·휴장 파일) 휴장이면 아무것도 안 한다. ka10081 기반 "휴장 의심" 은 진입만 막고 감시는 계속한다.
TRADING_JOBS = ("preopen", "open_check", "monitor", "close", "eod")
#: 장전 주문의 시가 대용(현재가)을 허용하는 시각 — ka10001 ``open_pric`` 이 끝내 없으면 이 시각 이후 현재가로 단일가를 대신한다.
OPEN_PROXY_FROM = time(9, 5)


class _LedgerOMS(OrderManager):
    """코어 ``OrderManager`` + 매도 예약 수량을 **자기 원장**으로 센다(검토 M2).

    코어 ``_pending_sell_qty`` 는 브로커 미체결 전부를 합산해, 같은 종목에 사용자가 걸어 둔 매도 주문이 전략 손절을
    "보유 초과" 로 막는다. 공유 계좌에서 남의 주문은 우리 예약이 아니다 — 런타임 원장의 살아 있는 자기 매도 잔량만 센다.
    """

    pending_sell_of: Callable[[str], int] = staticmethod(lambda code: 0)

    def _pending_sell_qty(self, code: str) -> int:
        return int(self.pending_sell_of(code))


class JobWindowError(RuntimeError):
    """작업 시간창 밖이다."""


# --------------------------------------------------------------------------- 알림


def notify(level: str, title: str, body: str = "") -> bool:
    """Discord 웹훅 알림(quant-airflow ``collectors/alert.py`` 규약: **절대 예외를 올리지 않는다**).

    ``SWING_ALERT_WEBHOOK_URL``(없으면 ``ALERT_WEBHOOK_URL``)이 없으면 로그만 남긴다.
    """
    try:
        log.log(logging.ERROR if level == "error" else logging.WARNING if level == "warn" else logging.INFO,
                "[alert %s] %s %s", level, title, body)
        url = (os.environ.get("SWING_ALERT_WEBHOOK_URL") or os.environ.get("ALERT_WEBHOOK_URL") or "").strip()
        if not url:
            return False
        import json  # noqa: PLC0415
        import socket  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        emoji = {"warn": "⚠️", "error": "🔴", "info": "ℹ️"}.get(level, "")
        host = os.environ.get("ALERT_HOST") or socket.gethostname()
        text = f"[{host}] {emoji} swing-live: {title}"[:200]
        if body:
            text += "\n```\n" + body[:1700] + "\n```"
        req = urllib.request.Request(url, data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "swing-live-alert/1"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
        return True
    except Exception:  # noqa: BLE001 — 알림이 본작업을 죽이면 알림이 사고가 된다
        return False


# --------------------------------------------------------------------------- 신호 원천


@dataclass
class SignalBatch:
    """신호 작업 입력 한 묶음(신호일 d 종가까지).

    - ``candidates``: :func:`swing_it.live.strategy.scan` 결과(순위 순).
    - ``trail_closes``: 코드 → KRX 정규장 종가(오래된 것 → 최신, d 포함, 최소 ``trail_ma`` 개) — 후보와 **보유 중** 종목 둘 다.
      보유 종목의 MA50 추적 결정·장 마감 자산은 **여기 d 종가**로 한다(감사 L5 — 장후 키움 시세는 KRX 종가로 검증되지 않았다).
    - ``krx_patched``: 신호일 ``d`` 의 NXT 보정(``swing_it.krx_session.patch_prices``)이 돌았나. ``patched_codes`` 는 d 행이
      실제로 보정된 종목(보정 대상 기간 전이면 ``None`` = 검사 안 함).
    """

    data_date: date
    candidates: list[Candidate]
    trail_closes: dict[str, list[float]] = field(default_factory=dict)
    krx_patched: bool = True
    patched_codes: set[str] | None = None
    names: dict[str, str] = field(default_factory=dict)
    #: 코드 → ``trail_closes`` 와 같은 길이의 날짜(ISO). 있으면 signal 이 **마지막 평가일 이후 모든 거래일**을 차례로 평가한다
    #: (검토 M5 — signal 을 하루 놓쳐도 추적 청산·자산 행이 빠지지 않는다). 없으면 d 하나만 평가한다.
    trail_dates: dict[str, list[str]] = field(default_factory=dict)
    #: 뉴스 거부권(설계 §7 · ``NEWS_VETO.md``)에 걸린 후보 — 주문 없이 **섀도 포지션**으로 추적한다(``trail_closes`` 에도 넣을 것).
    vetoed: list[Candidate] = field(default_factory=list)
    #: 코드 → 거부 사유 {"neg5": 직전 5거래일 악재 수, "neg20": 20거래일 악재 수, ...}. 섀도 원장에 그대로 적는다.
    veto_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 신호 원천이 전하는 경고(거부권 층의 모름 처리·판정 상한·실패 등) — signal 작업이 warn 경보로 보낸다.
    alerts: list[str] = field(default_factory=list)
    #: 코드 → ``trail_dates`` 와 같은 길이의 [시가, 고가, 저가, 종가](DB 보정본). 섀도 포지션의 일봉 상태 전이(백테스트와 같은
    #: ``on_bar``)와 가상 진입 대용(09:02 시세가 없던 날)에 쓴다(통합 검토 M5).
    bars: dict[str, list[list[float]]] = field(default_factory=dict)
    #: 입력 신선도(통합 검토 M1): {"ok": bool, "problems": [...], "supply_max_date", "f_coverage", "pead_coverage", "n_universe"}.
    #: ok=False 면 signal 은 보유 결정·자산은 수행하고 신규 진입만 중단한다.
    freshness: dict[str, Any] = field(default_factory=lambda: {"ok": True, "problems": []})


#: signal 결손 따라잡기용으로 MA 창 위에 더 받는 거래일 수의 **하한**(검토 M5·m10). 실제 배치 크기는 마지막 평가일 이후
#: 거래일 수로 늘어난다 — 이 값을 넘는 결손은 경보한다.
CATCHUP_DAYS = 15


def _ord_num(ord_no: str | None) -> int | None:
    """주문번호의 숫자값(당일 계좌 순번 비교용). 숫자가 아니면 ``None`` — 순서를 알 수 없다."""
    text = str(ord_no or "").strip()
    return int(text) if text.isdigit() else None


class SignalSource(Protocol):
    def load(self, asof: date, *, held: Iterable[str], since: date | None = None) -> SignalBatch:
        """``asof`` (오늘 거래일) 직전까지의 DB 로 신호 묶음을 만든다. ``held`` 는 MA50 종가가 필요한 보유 종목,
        ``since`` 는 보유 종목 중 가장 이른 마지막 평가일(그 뒤 거래일 전부의 종가가 필요하다 — m10)."""
        ...


class FeaturesSignalSource:
    """기본 신호 원천 — :func:`swing_it.live.features.signal_panels_for` (NXT 보정 포함) + :func:`strategy.scan`.

    DB 의존은 이 작업에만 둔다(지연 import). ``patch_info`` 로 신호일 ``d`` 의 KRX 정규장 보정 범위를 확인해
    :class:`SignalBatch` 에 싣는다 — 보정이 안 돈 날짜로는 신호를 내지 않는다(:meth:`SwingRuntime._job_signal`).
    """

    #: DB 읽기 재시도(통합 검토 §1.4: 아침 적재 중 연결 끊김 1회 실관측) — 횟수와 대기(초).
    RETRIES = 3
    BACKOFF = (5.0, 15.0)
    #: 신선도 하한(통합 검토 M1): 유니버스 종목 중 수급 F·PEAD 순위가 있는 비율.
    MIN_COVERAGE = 0.90

    def __init__(self, cfg: StrategyConfig = DEFAULT_STRATEGY, db: str | None = None, *,
                 sleeper: Callable[[float], None] = _time.sleep) -> None:
        self.cfg = cfg
        self.db = db
        self.sleeper = sleeper

    def load(self, asof: date, *, held: Iterable[str], since: date | None = None) -> SignalBatch:
        """``asof`` 직전 DB 거래일 ``d`` 를 신호일로 패널을 만든다 — ``--asof`` 과거일이면 그날 기준으로 **재생**된다(통합 검토 M2).

        DB 읽기(연결·패널·거부권)는 ``RETRIES`` 번까지 재시도한다(끊김 1회로 "오늘 신규 진입 없음" 이 되지 않게). 그래도 실패하면
        마지막 예외를 올린다 → signal 은 abort + error 경보.
        """
        from swing_it.live import features  # noqa: PLC0415 — DB 의존을 신호 작업에만 둔다

        if getattr(features, "signal_panels_for", None) is None:
            raise RuntimeError("swing_it.live.features.signal_panels_for 가 없다(전략 모듈 미완) — 신호를 내지 않는다")
        last: Exception | None = None
        for attempt in range(self.RETRIES):
            if attempt:
                wait = self.BACKOFF[min(attempt - 1, len(self.BACKOFF) - 1)]
                log.warning("signal DB 읽기 재시도 %d/%d (%.0fs 뒤): %s", attempt + 1, self.RETRIES, wait, last)
                self.sleeper(wait)
            try:
                return self._load_once(asof, held=held, since=since)
            except Exception as e:  # noqa: BLE001 — 재시도 대상(연결 끊김 등). 마지막엔 올린다.
                last = e
        assert last is not None
        raise last

    def _prev_trading_day(self, con: Any, asof: date) -> date | None:
        """``asof`` 직전 DB 거래일(가격 테이블 날짜 기준, storage 정문). 없으면 None(= DB 최신일 사용)."""
        from swing_it.storage import last_trading_date_before  # noqa: PLC0415

        try:
            return last_trading_date_before(con, asof.isoformat())
        except Exception as e:  # noqa: BLE001
            log.warning("직전 DB 거래일 조회 실패(DB 최신일로 대신): %s", e)
            return None

    def _freshness(self, con: Any, panels: dict, d: pd.Timestamp) -> dict[str, Any]:
        """입력 신선도(통합 검토 M1): supply_demand 최신일 == d, 유니버스의 F·PEAD 결측 ≤ 10%, 실적 패널 비어 있지 않음."""
        out: dict[str, Any] = {"ok": True, "problems": [], "date": str(d.date())}
        try:
            uni = panels["universe"].loc[d]
            codes = [str(c) for c in uni.index[uni.astype("float64").fillna(0) >= 1]]
            out["n_universe"] = len(codes)
            if codes:
                f = panels["F"].loc[d].reindex(codes)
                pr = panels["pead_rank"].loc[d].reindex(codes)
                out["f_coverage"] = float(f.notna().mean())
                out["pead_coverage"] = float(pr.notna().mean())
                if out["f_coverage"] < self.MIN_COVERAGE:
                    out["problems"].append(f"수급 F 결측 {1 - out['f_coverage']:.0%} > 10% (supply_demand 지연?)")
                if out["pead_coverage"] <= 0.0:
                    out["problems"].append("실적 패널 비어 있음(PEAD 순위 전부 결측 — earnings 적재?)")
                elif out["pead_coverage"] < self.MIN_COVERAGE:
                    out["problems"].append(f"PEAD 순위 결측 {1 - out['pead_coverage']:.0%} > 10%")
        except Exception as e:  # noqa: BLE001 — 검사 자체가 실패하면 문제로 기록(조용히 통과시키지 않는다)
            out["problems"].append(f"신선도 검사 실패: {type(e).__name__}: {e}")
        try:
            pg = type(con).__module__.split(".")[0] != "sqlite3"
            cur = con.cursor()
            cur.execute("SELECT MAX(date) FROM supply_demand")
            row = cur.fetchone()
            v = row[0] if row else None
            if v is not None:
                sd = v if isinstance(v, date) else pd.Timestamp(str(v)).date()
                out["supply_max_date"] = sd.isoformat()
                if sd != d.date():
                    out["problems"].append(f"supply_demand 최신일 {sd} ≠ 신호일 {d.date()}")
            else:
                out["problems"].append("supply_demand 비어 있음")
            del pg
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"supply_demand 최신일 조회 실패: {type(e).__name__}")
        out["ok"] = not out["problems"]
        return out

    def _load_once(self, asof: date, *, held: Iterable[str], since: date | None = None) -> SignalBatch:
        from swing_it.krx_session import NXT_START  # noqa: PLC0415
        from swing_it.live import features  # noqa: PLC0415
        from swing_it.live.strategy import scan  # noqa: PLC0415
        from swing_it.storage import connect, db_default  # noqa: PLC0415

        signal_panels_for = features.signal_panels_for
        con = connect(self.db or db_default())
        vetoed: list[Candidate] = []
        veto_info: dict = {}
        alerts: list[str] = []
        try:
            prev = self._prev_trading_day(con, asof)
            panels, d = signal_panels_for(con, None if prev is None else str(prev), self.cfg)
            d = pd.Timestamp(d)
            if d.date() >= asof:
                raise RuntimeError(f"DB 최신일 {d.date()} ≥ 거래일 {asof} — 오늘 봉(잠정치)이 들어 있다")
            freshness = self._freshness(con, panels, d)
            cands = scan(panels, d, self.cfg)
            if getattr(self.cfg, "news_veto", True) and cands:
                cands, vetoed, veto_info, alerts = self._apply_veto(con, cands, d, panels["close"].index)
        finally:
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass
        close = panels["close"]
        row = close.loc[d]
        have = set(map(str, row.index[row.notna()]))
        krx_patched, patched_codes = True, None
        if d >= pd.Timestamp(NXT_START):
            info = panels.get("patch_info") or {}
            unp = info.get("nxt_unpatched")
            unp_d = set()
            if isinstance(unp, pd.DataFrame) and not unp.empty:
                unp_d = set(map(str, unp.loc[pd.to_datetime(unp["date"]) == d, "code"]))
            patched_codes = have - unp_d
            krx_patched = bool(info.get("nxt_patched", 0)) and bool(patched_codes)
        gap = int(np.busday_count(since, asof)) if since is not None else 0
        n = self.cfg.trail_ma + max(CATCHUP_DAYS, gap + 5)      # m10: 결손 길이만큼 늘린다(상한 없음)
        trail, dates, bars = {}, {}, {}
        ohl = [panels.get(k) for k in ("open", "high", "low")]
        for code in {c.code for c in cands} | {c.code for c in vetoed} | set(held):
            if code in close.columns:
                ser = close[code].loc[:d].dropna().iloc[-n:]
                trail[code] = [float(x) for x in ser]
                dates[code] = [str(pd.Timestamp(i).date()) for i in ser.index]
                if all(x is not None and code in x.columns for x in ohl):
                    o, h, lo_ = (x[code].reindex(ser.index) for x in ohl)
                    bars[code] = [[float(a), float(b), float(c_), float(cl)]
                                  for a, b, c_, cl in zip(o, h, lo_, ser, strict=True)]
        return SignalBatch(data_date=d.date(), candidates=cands, trail_closes=trail, trail_dates=dates,
                           krx_patched=krx_patched, patched_codes=patched_codes, vetoed=vetoed, veto_info=veto_info,
                           alerts=alerts, bars=bars, freshness=freshness)

    def _apply_veto(self, con: Any, cands: list[Candidate], d: pd.Timestamp, trading_days: Any
                    ) -> tuple[list[Candidate], list[Candidate], dict[str, dict[str, Any]], list[str]]:
        """뉴스 거부권(설계 §7, :mod:`swing_it.live.news_veto`) → (진입 후보, 거부 후보, 코드→사유, 경고).

        거부권 층의 어떤 실패도 신호를 막지 않는다: 모듈이 없거나 예외면 **전 후보 진입(모름)** + 경고 한 줄. 사유는 코드별
        ``{"neg5", "neg20", "judged", "unknown"}`` 로 펴서 섀도 원장에 적는다.
        """
        try:
            from swing_it.live import news_veto  # noqa: PLC0415 — 전략 에이전트 모듈(지연 import)
        except ImportError as e:
            return cands, [], {}, [f"뉴스 거부권 모듈 없음({e}) — 전 후보 모름(거부 없음)"]
        try:
            days = [pd.Timestamp(x).date() for x in trading_days]
            res = news_veto.apply_live_veto(con, cands, d.date(), days)
        except Exception as e:  # noqa: BLE001 — 거부권이 죽으면 모름으로 접는다
            return cands, [], {}, [f"뉴스 거부권 실패({type(e).__name__}: {e}) — 전 후보 모름(거부 없음)"]
        info: dict[str, dict[str, Any]] = {k: dict(v) for k, v in ((res.info or {}).get("vetoed") or {}).items()}
        for c in res.vetoed:
            info.setdefault(c.code, {})
            n = res.news.get(c.code)
            if n is not None:
                info[c.code].update({"neg5": n.neg5, "neg20": n.neg20, "judged": n.judged, "unknown": n.unknown})
        return list(res.kept), list(res.vetoed), info, list(res.alerts)


# --------------------------------------------------------------------------- 설정


def _strategy_from_dict(d: dict[str, Any]) -> StrategyConfig:
    """dict → :class:`StrategyConfig`. 알 수 없는 키는 ``ValueError``(조용히 무시 금지), 빠진 키는 코드 기본값. ``"inf"`` 문자열은 inf."""
    names = {f.name for f in fields(StrategyConfig)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ValueError(f"StrategyConfig 에 없는 키: {unknown}")
    kw = {}
    for k, v in d.items():
        if isinstance(v, str) and v.strip().lower() in ("inf", "infinity", "+inf"):
            v = math.inf
        kw[k] = v
    return StrategyConfig(**kw)


def load_strategy_config(path: Path | str) -> tuple[StrategyConfig, dict[str, Any]]:
    """버전 있는 전략 설정 파일 → (StrategyConfig, 메타). ``--strategy-config`` (통합 검토 §3).

    우선 전략 모듈의 로더(``swing_it.live.config.load_candidate``)가 있으면 그것을 쓴다(GA 유전자 → 설정 대응은 그쪽 정본).
    없으면 JSON 을 직접 읽는다: ``{"strategy": {...}}`` 블록, 또는 ``{"genes": {...}}``(``StrategyConfig.from_genes`` 가 있을 때만),
    또는 필드 dict 그대로. 메타엔 파일 sha256·설정 해시·``schema``/``generated``/``valid_from``/``source`` 를 담는다.
    """
    import hashlib  # noqa: PLC0415

    from swing_it.live import config as config_mod  # noqa: PLC0415

    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {"path": str(path), "file_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()}
    loader = getattr(config_mod, "load_candidate", None)
    cfg: StrategyConfig | None = None
    if loader is not None:
        got = loader(path)
        if isinstance(got, StrategyConfig):
            cfg = got
        elif isinstance(got, tuple) and got and isinstance(got[0], StrategyConfig):
            cfg = got[0]
            if len(got) > 1 and isinstance(got[1], dict):
                meta.update(got[1])
    if cfg is None:
        raw = json.loads(raw_text)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: 설정 파일은 JSON 객체여야 한다")
        for k in ("schema", "generated", "valid_from", "source", "procedure"):
            if k in raw:
                meta[k] = raw[k]
        if isinstance(raw.get("strategy"), dict):
            cfg = _strategy_from_dict(raw["strategy"])
        elif isinstance(raw.get("genes"), dict):
            from_genes = getattr(StrategyConfig, "from_genes", None)
            if from_genes is None:
                raise ValueError(f"{path}: genes 블록을 읽을 StrategyConfig.from_genes 가 아직 없다")
            cfg = from_genes(raw["genes"])
        else:
            cfg = _strategy_from_dict({k: v for k, v in raw.items() if k not in meta})
    meta["cfg_hash"] = strategy_hash(cfg.to_dict())
    return cfg, meta


@dataclass
class RuntimeConfig:
    """런타임 설정. 전략·위험 수치는 :mod:`swing_it.live.config` 가 정본이다."""

    data_dir: Path
    mode: str = "paper"
    capital: float | None = None              # 첫 실행에만 쓰는 전략 자본(원). live 는 명시 필수
    strategy: StrategyConfig = DEFAULT_STRATEGY
    risk: RiskConfig = DEFAULT_RISK
    stop_file: Path | None = None             # 전역 정지 파일(기본 risk.stop_file)
    holidays_file: Path | None = None         # 사람이 적는 휴장일(YYYY-MM-DD 한 줄씩)
    ignore_clock: bool = False                # 페이퍼 리허설 전용
    poll_sec: float = 60.0
    quiet_windows: tuple[tuple[str, str], ...] = (("10:00", "10:15"),)
    quiet_poll_sec: float = 180.0
    monitor_until: time = time(15, 19)
    exit_retry_sec: float = 60.0
    exit_discount_step: float = 0.005
    exit_discount_cap: float = 0.03
    auction_sell_floor: float = 0.10          # 동시호가 매도 지정가 = 기준가 × (1 − 0.10)
    prev_close_tolerance: float = 0.01        # 계획 기준가 vs KRX 기준가(ka10001) 허용 차
    lock_wait_sec: float = 120.0
    max_exit_rejects: int = 3                 # 연속 거부·차단 이만큼이면 그 매도 의도를 멈추고 사람을 부른다(검토 M1)
    cancel_confirm_reads: int = 3             # 취소 거부(=이미 체결?) 뒤 잔고 재조회 횟수(검토 C2)
    cancel_confirm_wait_sec: float = 5.0      # 그 재조회 간격 — 3회 × 5초 ≥ 10초
    strategy_file: Path | None = None         # --strategy-config: 버전 있는 후보 JSON(없으면 코드 기본값)
    strategy_meta: dict[str, Any] = field(default_factory=dict)   # 그 파일의 메타(경로·sha256·valid_from 등)
    quotes_source: str = "kiwoom"             # "kiwoom" | "db"(--asof 재생, 키움 호출 없음)

    @property
    def mode_dir(self) -> Path:
        return Path(self.data_dir) / self.mode

    @property
    def global_stop(self) -> Path:
        return Path(self.stop_file) if self.stop_file else Path(self.risk.stop_file)

    @property
    def mode_stop(self) -> Path:
        """모드별 정지 파일 — 낙폭 킬이 쓰는 곳. live 는 전역과 같다(``swing_STOP``), 페이퍼는 ``swing_paper_STOP``."""
        g = self.global_stop
        if self.mode == "live":
            return g
        tag = self.mode.replace("-", "")
        stem = g.name[:-len("_STOP")] if g.name.endswith("_STOP") else g.name
        return g.with_name(f"{stem}_{tag}_STOP")


# --------------------------------------------------------------------------- 작업 문맥


@dataclass
class _Ctx:
    job: str
    asof: date
    state: LiveState
    book: PositionBook
    oms: OrderManager
    kill: KillSwitch
    shadow_book: PositionBook | None = None
    last_px: dict[str, float] = field(default_factory=dict)
    auction_px: dict[str, float] = field(default_factory=dict)
    quotes: dict[str, QuoteSnap] = field(default_factory=dict)

    @property
    def day(self) -> str:
        return self.asof.isoformat()


def _plan_to_dict(p: EntryPlan) -> dict[str, Any]:
    d = asdict(p)
    d["signal_day"] = str(pd.Timestamp(p.signal_day).date())
    return d


def _plan_from_dict(d: dict[str, Any]) -> EntryPlan:
    names = EntryPlan.__dataclass_fields__
    kw = {k: v for k, v in d.items() if k in names}
    kw["signal_day"] = pd.Timestamp(kw["signal_day"])
    return EntryPlan(**kw)


def shadow_vs_real_line(state: LiveState) -> str:
    """``sw-live status`` 한 줄: 거부군(섀도) vs 진입군(실제) — n·평균 순수익·손절 비율(``NEWS_VETO.md`` 합격 바).

    합격 바: 거부군 평균 순수익 < 진입군 **그리고** 거부군 손절 비율 ≥ 진입군(전진 6개월 또는 거부 30건 이상).
    """
    sh = trade_stats(list(state.shadow_history))
    # m10: 진입군은 거부권이 돌기 시작한 뒤(첫 섀도 신호일 이후)의 포지션만 — 같은 운용 창끼리 비교한다.
    first = min((p.signal_date for p in state.shadow_history + list(state.shadow.values())), default=None)
    real_recs = [p for p in state.history if first is None or p.signal_date >= first]
    re_ = trade_stats(real_recs)

    def fmt(d: dict[str, Any]) -> str:
        if d["n"] == 0:
            return "n=0"
        return f"n={d['n']} 평균순수익={d['mean_net']:+.2%} 손절률={d['stop_rate']:.0%}"

    verdict = ""
    if sh["n"] and re_["n"]:
        ok = sh["mean_net"] < re_["mean_net"] and sh["stop_rate"] >= re_["stop_rate"]
        verdict = f" → 합격 바 {'충족' if ok else '미충족'}(거부 {sh['n']}/30건)"
    return (f"뉴스 거부권[순수익 기준 — 사전등록 지표(정합 초과) 아님: 라이브에 정합 대조 없음]: "
            f"거부군(섀도) {fmt(sh)} | 진입군(실제, 거부권 시작 이후) {fmt(re_)}{verdict}")


def _tick_down(px: float) -> int:
    return int(round_to_tick(px))


def _tick_up(px: float) -> int:
    return int(round_to_tick_up(px))


# --------------------------------------------------------------------------- 런타임


class SwingRuntime:
    """작업 실행기. 테스트는 ``bundle``(가짜 브로커·시세)·``clock``·``signal_source``·``sleeper`` 를 주입한다."""

    def __init__(
        self,
        cfg: RuntimeConfig,
        bundle: BrokerBundle | Callable[[], BrokerBundle],
        *,
        clock: Callable[[], datetime] = now_kst,
        signal_source: SignalSource | None = None,
        sleeper: Callable[[float], None] = _time.sleep,
        alert: Callable[[str, str, str], Any] = notify,
    ) -> None:
        if bundle is not None and not callable(bundle) and bundle.mode != cfg.mode:
            raise ValueError(f"브로커 모드 {bundle.mode!r} ≠ 런타임 모드 {cfg.mode!r}")
        if cfg.ignore_clock and cfg.mode != "paper":
            raise ValueError("ignore_clock 은 페이퍼 전용이다")
        self.cfg = cfg
        self._bundle_src = bundle
        self._bundle: BrokerBundle | None = None
        self.clock = clock
        self.signal_source = signal_source
        self.sleeper = sleeper
        self.alert = alert
        self.store = StateStore(cfg.mode_dir, clock=clock)
        self._alerted: set[str] = set()
        #: 현재(신규 계획용) 전략 설정 해시. 원장 모든 사건에 ``cfg=<8자리>`` 로 붙는다. 열린 포지션은 각자 계획 시점 설정에 고정.
        self.cfg_hash = strategy_hash(cfg.strategy.to_dict())
        self.store.default_fields = {"cfg": self.cfg_hash[:8]}
        self._cfg_cache: dict[str, StrategyConfig] = {self.cfg_hash: cfg.strategy}

    # ----- 공통 ------------------------------------------------------------------

    @property
    def bundle(self) -> BrokerBundle:
        if self._bundle is None:
            src = self._bundle_src
            self._bundle = src() if callable(src) and not isinstance(src, BrokerBundle) else src
            if self._bundle.mode != self.cfg.mode:
                raise ValueError(f"브로커 모드 {self._bundle.mode!r} ≠ 런타임 모드 {self.cfg.mode!r}")
        return self._bundle

    def _now(self) -> datetime:
        n = self.clock()
        return n.astimezone(KST) if n.tzinfo else n.replace(tzinfo=KST)

    def _check_window(self, job: str, asof: date) -> None:
        if job not in JOB_WINDOWS or self.cfg.ignore_clock:
            return
        now = self._now()
        lo, hi = JOB_WINDOWS[job]
        if now.date() != asof or not (lo <= now.time() < hi):
            raise JobWindowError(
                f"{job}: 시간창 밖({now:%Y-%m-%d %H:%M:%S}, 허용 {asof} {lo:%H:%M}~{hi:%H:%M}) — 주문하지 않는다"
            )

    def _check_holidays_file(self, ctx: _Ctx) -> None:
        """``holidays.txt`` 가 올해 휴장일을 담고 있나(검토 F4 — 달력이 휴장의 첫 번째 방어선). 없으면 경보(한 번)."""
        hf = self.cfg.holidays_file
        year = str(ctx.asof.year)
        ok = False
        if hf and Path(hf).exists():
            ok = any(ln.strip().startswith(year) for ln in Path(hf).read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#"))
        if not ok and f"holidays:{year}" not in self._alerted:
            self._alerted.add(f"holidays:{year}")
            self.alert("warn", f"holidays.txt 에 {year}년 휴장일이 없다",
                       f"{hf}: 달력 휴장이 첫 방어선이다 — KRX {year} 휴장일을 적을 것(시세·일봉만으로는 휴장을 못 가린다)")

    def _weekday_open(self, asof: date) -> bool:
        if asof.weekday() >= 5:
            return False
        hf = self.cfg.holidays_file
        if hf and Path(hf).exists():
            days = {ln.strip()[:10] for ln in Path(hf).read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")}
            if asof.isoformat() in days:
                return False
        return True

    def run(self, job: str, asof: date | None = None, *, force: bool = False, once: bool = False,
            oid: str | None = None) -> int:
        """작업 하나. 종료 코드: 0 정상(건너뜀 포함), 1 오류, 2 시간창 밖·잠금 경합, 3 상태 손상."""
        if job not in JOBS:
            raise ValueError(f"알 수 없는 작업: {job!r}")
        asof = asof or self._now().date()
        self._oid = oid
        self._setup_logging(asof)
        if job == "status":
            return self._status(asof)
        try:
            self._check_window(job, asof)
        except JobWindowError as e:
            log.error("%s", e)
            self.alert("warn", f"{job}: 시간창 밖 — 건너뜀", str(e))     # 검토 M7: 크론 지연을 조용히 넘기지 않는다
            return 2
        lock = InstanceLock(self.store.lock_path)
        if not self._acquire(lock, wait=0.0 if job == "monitor" else self.cfg.lock_wait_sec):
            log.warning("%s: 다른 작업이 잠금을 잡고 있다(%s) — 종료", job, self.store.lock_path)
            if job != "monitor":
                self.alert("warn", f"{job}: 잠금 경합 — 건너뜀", str(self.store.lock_path))
            return 2 if job != "monitor" else 0
        try:
            return self._run_locked(job, asof, force=force, once=once)
        finally:
            lock.release()

    def _acquire(self, lock: InstanceLock, *, wait: float) -> bool:
        deadline = _time.monotonic() + wait
        while True:
            try:
                lock.acquire()
                return True
            except AlreadyRunning:
                if _time.monotonic() >= deadline:
                    return False
                _time.sleep(min(2.0, max(0.05, wait / 20)))

    def _setup_logging(self, asof: date) -> None:
        self.store.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.store.log_dir / f"{asof.isoformat()}.log"
        for h in log.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve():
                return
        for h in list(log.handlers):
            if isinstance(h, logging.FileHandler):
                log.removeHandler(h)
                h.close()
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
        log.setLevel(logging.INFO)

    def _load(self) -> tuple[LiveState, PositionBook]:
        capital = self.cfg.capital
        if capital is None:
            if self.cfg.mode == "live" and not self.store.state_path.exists():
                raise StateCorrupt("live 첫 실행에는 --capital(전략 자본)을 명시해야 한다")
            capital = 10_000_000.0
        state = self.store.load(mode=self.cfg.mode, capital=capital)
        return state, self.store.load_book(state)

    def _make_ctx(self, job: str, asof: date, state: LiveState, book: PositionBook) -> _Ctx:
        kill = self._killswitch(state, asof)
        oms = _LedgerOMS(self.bundle.broker, guard=OrderGuard(clock=self.clock), book=book, kill=kill,
                         clock=self.clock, order_log=self.store.order_log_path, own_orders_only=True)
        oms.own_orders = set(state.own_ord_nos)
        ctx = _Ctx(job=job, asof=asof, state=state, book=book, oms=oms, kill=kill,
                   shadow_book=self.store.load_shadow_book(state))
        oms.pending_sell_of = lambda code: self._pending_sell(ctx, code)
        return ctx

    def _run_locked(self, job: str, asof: date, *, force: bool, once: bool) -> int:
        try:
            state, book = self._load()
        except StateCorrupt as e:
            log.error("상태 손상: %s", e)
            self.alert("error", "상태 손상 — 작업 중단", str(e))
            return 3
        try:
            ctx = self._make_ctx(job, asof, state, book)
        except Exception as e:  # noqa: BLE001 — 브로커·시세 연결 실패 등
            log.exception("%s: 브로커 준비 실패", job)
            self.alert("error", f"{job}: 브로커 준비 실패", f"{type(e).__name__}: {e}")
            return 1
        if state.config_hash != self.cfg_hash:
            old_pos = [p.code for p in state.open_positions() if p.cfg_hash and p.cfg_hash != self.cfg_hash]
            self.store.event("config_changed", before=(state.config_hash or "")[:8], after=self.cfg_hash[:8],
                             pinned_positions=old_pos, meta=self.cfg.strategy_meta)
            if state.config_hash is not None:
                self.alert("info", "전략 설정 교체", f"{(state.config_hash or '')[:8]} → {self.cfg_hash[:8]} "
                           f"(신규 계획에만 적용; 옛 설정으로 열린 포지션 {len(old_pos)}개는 그대로)")
            state.config_hash = self.cfg_hash
        rc = 0
        try:
            if job in TRADING_JOBS and not self._session_open(ctx):
                log.info("%s: %s 휴장(달력) — 아무것도 안 한다", job, asof)
                return 0
            if job in TRADING_JOBS or job == "signal":
                self._check_holidays_file(ctx)
            if job in TRADING_JOBS:
                self._resume_orders(ctx)      # 검토 C1: 새 주문 전에 저장된 주문 의도·장부를 브로커와 대조
            handler = getattr(self, f"_job_{job}")
            if job == "monitor":
                handler(ctx, once=once)
            elif job in ("signal", "eod"):
                handler(ctx, force=force)
            else:
                handler(ctx)
        except Exception as e:  # noqa: BLE001 — 기록·알림 뒤 비정상 종료 코드(저장은 finally)
            log.exception("%s 실패", job)
            self.store.event("error", job=job, error=f"{type(e).__name__}: {e}")
            self.alert("error", f"{job} 실패", f"{type(e).__name__}: {e}")
            rc = 1
        finally:
            state.own_ord_nos = sorted(ctx.oms.own_orders)
            self.store.save(state)
            self.bundle.persist()
        return rc

    # ----- 휴장·킬 ---------------------------------------------------------------

    def _session_open(self, ctx: _Ctx) -> bool:
        """오늘 장이 열리나. **달력(주말·휴장 파일)만 닫힘으로 확정한다.**

        ka10081 기준 종목 오늘 봉이 아직 없으면 "휴장 의심"(``sessions[day]="suspect"``)이다 — 09:02 에 봉이 없을 뿐일 수 있어
        (검토 M3) 주문을 취소하거나 감시를 멈추지 않는다: 장전 주문은 그대로 두고, 시가 대용·보유일 전진은 세션이 확정될 때까지
        미룬다. 판정은 고정이 아니다: 부를 때마다 다시 보고, 시세가 시가와 함께 오면(:meth:`_quote`) 그 자체로 거래일로 본다.
        """
        st = ctx.state
        known = st.sessions.get(ctx.day)
        if known == "closed":
            return False
        if known == "open":
            return True
        if not self._weekday_open(ctx.asof):
            st.sessions[ctx.day] = "closed"
            self.store.event("holiday", date=ctx.day, source="calendar")
            return False
        if ctx.job == "preopen":
            return True          # 장 전에는 확인할 수단이 없다
        sd = session_date(self.bundle.quotes, ctx.asof)
        if sd == ctx.asof:
            st.sessions[ctx.day] = "open"
            return True
        if known != "suspect":
            st.sessions[ctx.day] = "suspect"
            why = "ka10081 조회 실패" if sd is None else f"기준 종목 최신 봉 {sd}"
            self.store.event("holiday_suspect", date=ctx.day, source="ka10081", latest=None if sd is None else sd.isoformat())
            self.alert("warn", "휴장 의심 — 주문은 유지, 체결 판정·보유일 전진 보류",
                       f"{ctx.day}: {why}. 달력엔 거래일. 오늘 날짜가 찍힌 증거(ka10081 오늘 봉·시세 일자)가 오면 계속한다; "
                       "아니면 eod 가 주문을 거둔다. holidays.txt 를 확인할 것")
        return True

    def _confirm_session(self, ctx: _Ctx) -> bool:
        """세션이 거래일로 확정됐나(ka10081 또는 시가 있는 시세). 의심 상태면 다시 확인한다."""
        if ctx.state.sessions.get(ctx.day) == "open":
            return True
        return self._session_open(ctx) and ctx.state.sessions.get(ctx.day) == "open"

    def _killswitch(self, state: LiveState, asof: date) -> KillSwitch:
        """코어 일 손실 킬 + 전역 정지 파일(비고정 — 파일이 있는 동안만).

        스윙은 장전에만 진입하므로 "일 손실" 은 **직전 거래일 자산 변동**으로 잰다(그 손실이 한도를 넘었으면 오늘 신규
        진입을 쉰다).
        """
        hist = state.equity_history
        equity = float(hist[-1]["equity"]) if hist else float(state.capital)
        ks = KillSwitch(KillSwitchConfig(daily_loss_limit_pct=self.cfg.risk.daily_loss_limit_pct,
                                         daily_capital=equity, stop_file=self.cfg.global_stop,
                                         sticky_stop_file=False))
        ks.observe(asof)
        if len(hist) >= 2 and hist[-1]["date"] < asof.isoformat():
            ks.record_realized(float(hist[-1]["equity"]) - float(hist[-2]["equity"]), on=asof)
        return ks

    def entry_block_reason(self, ctx: _Ctx) -> str | None:
        """신규 진입을 막는 사유(없으면 None). 청산에는 쓰지 않는다."""
        if self.cfg.global_stop.exists():
            return f"정지 파일 {self.cfg.global_stop}"
        if self.cfg.mode_stop.exists():
            return f"정지 파일 {self.cfg.mode_stop}"
        if ctx.state.drawdown_kill.get("active"):
            return f"다일 낙폭 킬({ctx.state.drawdown_kill.get('date')})"
        reason = ctx.kill.check(ctx.asof)
        if reason:
            return f"킬스위치:{reason}"
        return None

    # ----- 자산 ------------------------------------------------------------------

    def _equity(self, ctx: _Ctx, prices: dict[str, float] | None = None) -> tuple[float, float, float]:
        """``(자산, 현금, 평가액)`` — 전략 자본 기준(공유 계좌 전체가 아니다). 가격 없으면 평단."""
        prices = prices or {}
        cost = mv = 0.0
        for code, h in ctx.book.positions().items():
            cost += h.qty * h.avg_price
            mv += h.qty * float(prices.get(code, h.avg_price))
        cash = float(ctx.state.capital) + float(ctx.book.realized_krw) - cost
        return cash + mv, cash, mv

    def _sizing_equity(self, ctx: _Ctx) -> float:
        hist = ctx.state.equity_history
        return float(hist[-1]["equity"]) if hist else float(ctx.state.capital)

    # ----- 주문 ------------------------------------------------------------------

    def _begin_order(self, ctx: _Ctx, *, side: str, code: str, qty: int, price: int, purpose: str, session: str,
                     attempt: int = 0, ref_price: float | None = None) -> OrderRec:
        """주문 **의도**를 POST 전에 원장·스냅샷에 남긴다(검토 C1: POST 와 저장 사이 크래시 → 같은 주문 두 번 방지).

        ``oid`` 가 클라이언트 주문 키다. 다음 실행의 :meth:`_resume_orders` 가 브로커 미체결에서 이 의도를 찾아 주문번호를
        붙이고, 못 찾으면 ``unknown`` 으로 두고 사람을 부른다 — 어느 쪽이든 **다시 POST 하지 않는다**.
        """
        rec = OrderRec(
            oid=ctx.state.next_oid(), trade_date=ctx.day, ts=self._now().isoformat(timespec="seconds"),
            code=code, side=side, qty=int(qty), price=int(price), purpose=purpose, session=session,
            status="intent", attempt=attempt, ref_price=ref_price,
        )
        ctx.state.orders[rec.oid] = rec
        self.store.event("order_intent", **asdict(rec))
        self.store.save(ctx.state)
        return rec

    def _finish_order(self, ctx: _Ctx, rec: OrderRec, managed: Any) -> OrderRec:
        res = managed.result
        rec.status = "dry_run" if res.dry_run and not res.blocked else str(managed.status)
        rec.ord_no = res.ord_no
        rec.reason = res.blocked_reason or res.return_msg
        self.store.event("order", **asdict(rec))
        self.store.save(ctx.state)
        if rec.status == "unknown":
            self.alert("error", "주문 결과 모름(UNKNOWN)", f"{rec.side} {rec.code} {rec.qty}@{rec.price} — 재시도 안 함, "
                       "미체결·잔고로 사람이 확인")
        elif rec.status in ("blocked", "rejected"):
            self.alert("error" if rec.side == "sell" else "warn", f"{'매도' if rec.side == 'sell' else '매수'} 주문 {rec.status}",
                       f"{rec.code} {rec.qty}@{rec.price} [{rec.purpose}]: {rec.reason}")
        return rec

    def _resume_orders(self, ctx: _Ctx) -> None:
        """작업 시작 시 저장된 주문 의도·결과 없는 주문·장부를 브로커와 대조한다(검토 C1·F1·F2·m4). 새 주문은 그 뒤에만 낸다.

        대상: ``intent``(POST 전에 저장, 결과 못 적음) 와 **주문번호 없는 ``unknown``**(접수 뒤 응답 유실). 브로커 미체결에서
        같은 종목·방향 중 **우리 원장에 없는 번호가 정확히 하나**이고, 그 번호가 오늘 우리가 아는 최대 번호보다 **크며**(당일
        계좌 순번), 수량·가격이 같을 때만 이어받는다(error 경보 — 사람이 확인). 후보가 둘 이상이거나 순서를 못 가리면
        ``needs_human``, 없으면 ``unknown`` 그대로 — 두 경우 모두 그 종목의 **신규 진입을 막고**(``blocked_codes``) 경보한다.
        어느 쪽이든 다시 POST 하지 않는다. 풀기: ``sw-live release-order <oid>``.
        """
        st = ctx.state
        pending = [o for o in st.orders.values()
                   if o.status == "intent" or (o.status == "unknown" and not o.ord_no and o.trade_date == ctx.day)]
        if pending:
            try:
                broker_open: list[OpenOrder] | None = list(self.bundle.broker.open_orders())
            except Exception as e:  # noqa: BLE001
                broker_open = None
                log.warning("미체결 조회 실패 — 주문 의도 대조 불가: %s", e)
            known = {normalize_ord_no(o.ord_no) for o in st.orders.values() if o.ord_no}
            max_known = max((n for n in (_ord_num(o.ord_no) for o in st.orders.values()
                                         if o.ord_no and o.trade_date == ctx.day) if n is not None), default=0)
            for o in pending:
                cands = [] if broker_open is None else [
                    bo for bo in broker_open
                    if normalize_ord_no(bo.ord_no) not in known and bo.code == o.code and bo.side == o.side]
                one = cands[0] if len(cands) == 1 else None
                num = _ord_num(one.ord_no) if one is not None else None
                if (one is not None and num is not None and num > max_known
                        and one.qty == o.qty and one.price == o.price):
                    o.ord_no, o.status = one.ord_no, "submitted"
                    o.filled_qty = max(0, one.qty - one.remaining)
                    known.add(normalize_ord_no(one.ord_no))
                    ctx.oms.own_orders.add(normalize_ord_no(one.ord_no))
                    self.store.event("order_resumed", oid=o.oid, ord_no=o.ord_no, code=o.code)
                    self.alert("error", "주문 이어받음 — 사람이 확인",
                               f"{o.side} {o.code} {o.qty}@{o.price} ({o.purpose}) → 브로커 주문 {o.ord_no}. 우리 주문이 맞는지 HTS 로 확인")
                    continue
                if cands or broker_open is None:
                    o.status = "needs_human"
                    o.reason = "ambiguous_resume" if cands else "open_orders_unavailable"
                    why = (f"같은 종목·방향 미체결 {len(cands)}건(번호 {[c.ord_no for c in cands]}) — 우리 것인지 순서로 못 가린다"
                           if cands else "미체결 조회 실패")
                else:
                    o.status = "unknown"
                    o.reason = "intent_unresolved"
                    why = "POST 뒤 결과를 못 적었고 미체결에도 없다 — 체결·거부 여부 모름"
                st.blocked_codes[o.code] = f"{o.oid}:{o.reason}"
                self.store.event("order_unresolved", oid=o.oid, code=o.code, status=o.status, reason=o.reason)
                if f"unresolved:{o.oid}" not in self._alerted:
                    self._alerted.add(f"unresolved:{o.oid}")
                    self.alert("error", f"주문 미해결({o.status}) — {o.code} 신규 진입 차단",
                               f"{o.side} {o.code} {o.qty}@{o.price} ({o.purpose}, {o.oid}): {why}. "
                               f"재발주 없음. 확인 뒤 `sw-live release-order {o.oid}`")
        orphans = [c for c in strategy_holdings(ctx.book) if c not in st.positions]
        if orphans and not ctx.job == "reconcile":
            key = "orphans:" + ",".join(sorted(orphans))
            if key not in self._alerted:
                self._alerted.add(key)
                self.alert("error", "장부에 있는데 원장 포지션이 없는 종목", ", ".join(sorted(orphans)) + " — reconcile 로 확인")

    def _broker_has_order(self, o: OrderRec) -> bool | None:
        """브로커 미체결에 이 주문이 아직 있나. 조회 실패면 ``None``."""
        if not o.ord_no:
            return False
        try:
            key = normalize_ord_no(o.ord_no)
            return any(normalize_ord_no(x.ord_no) == key for x in self.bundle.broker.open_orders())
        except Exception as e:  # noqa: BLE001
            log.warning("미체결 조회 실패: %s", e)
            return None

    def _cancel(self, ctx: _Ctx, o: OrderRec, *, reason: str) -> None:
        if o.status in ("intent", "fill_pending"):
            log.info("취소 보류 %s %s: 상태 %s", o.code, o.oid, o.status)
            return
        if o.status == "dry_run":
            return
        if not o.ord_no:
            # 번호 없는 unknown: 브로커 취소가 확인되기 전엔 expired 로 치지 않는다(검토 F2 — 주문이 살아 있을 수 있다).
            st_ = ctx.state
            st_.blocked_codes.setdefault(o.code, f"{o.oid}:no_ord_no")
            if f"no_ord_no:{o.oid}" not in self._alerted:
                self._alerted.add(f"no_ord_no:{o.oid}")
                self.alert("error", f"번호 없는 주문 — 취소 불가, 사람이 확인 ({o.code})",
                           f"{o.side} {o.qty}@{o.price} ({o.oid}, {reason}): 브로커에 살아 있을 수 있다. HTS 에서 확인·취소 뒤 "
                           f"`sw-live release-order {o.oid}`")
            return
        intent = OrderIntent(side=o.side, code=o.code, qty=o.remaining or o.qty, price=o.price)
        m = ctx.oms.cancel(o.ord_no, intent)
        if str(m.status) in ("canceled",) or m.result.dry_run:
            o.status = "canceled"
        elif str(m.status) == "unknown":
            o.status = "unknown"
            self.alert("warn", "취소 결과 모름", f"{o.code} {o.ord_no} — 미체결 조회로 확인 필요")
        else:
            self._cancel_rejected(ctx, o, m.result.return_msg)
        o.reason = (o.reason + f" | {reason}").strip(" |")
        self.store.event("cancel", oid=o.oid, ord_no=o.ord_no, code=o.code, status=o.status, reason=reason)

    def _cancel_rejected(self, ctx: _Ctx, o: OrderRec, msg: str) -> None:
        """취소 거부 = 대개 이미 체결(검토 C2). 잔고 한 번으로 단정하지 않는다.

        주문을 ``fill_pending`` 으로 두고 잔고를 ``cancel_confirm_reads`` 번(간격 ``cancel_confirm_wait_sec``) 다시 읽어 체결이
        잔고에 나타나길 기다린다. 그동안 이 수량은 예약된 채라 같은 수량을 다시 팔지 않는다. 미체결에 아직 있으면 취소만 실패한
        것이니 ``submitted`` 로 되돌린다. 끝까지 설명이 안 되면 ``released``(검토 F3) — 예약을 풀고 error 경보. 예약이 청산을
        막는 시간은 이 창(≈10초)뿐이다.
        """
        o.status = "fill_pending"
        o.reason = f"cancel_rejected: {msg}"
        reads = max(1, self.cfg.cancel_confirm_reads) if self.bundle.fills_from_holdings else 1
        for i in range(reads):
            if i:
                self.sleeper(self.cfg.cancel_confirm_wait_sec)
            self._sync_fills(ctx)
            if o.remaining == 0 or o.status != "fill_pending":
                return
            if self._broker_has_order(o):
                o.status = "submitted"      # 취소만 거부됐고 주문은 살아 있다
                return
        # 검토 F3: 체결도 미체결도 설명하지 못하면(사용자 HTS 취소·휴장 소멸·취소 API 오류) 예약을 풀어 청산이 다시 나가게 한다.
        # 예약을 영영 쥐고 있으면 손절이 영영 안 나간다. 뒤늦게 체결이 잔고에 나타나면 "설명 안 되는 차이" 경보로 드러난다.
        o.status = "released"
        o.reason = (o.reason + " | released_unexplained").strip(" |")
        self.store.event("order_released", oid=o.oid, ord_no=o.ord_no, code=o.code, remaining=o.remaining)
        self.alert("error", f"취소 거부 뒤 설명 안 됨 → 예약 해제 {o.code}",
                   f"{o.side} 잔량 {o.remaining} ({o.ord_no}): 잔고 {reads}회 재조회·미체결 확인 모두 설명 못 함 — "
                   "주문이 사라진 것으로 보고 예약을 푼다(청산 재시도 허용). 체결이 뒤늦게 보이면 잔고 차이 경보")

    def _pending_sell(self, ctx: _Ctx, code: str) -> int:
        """우리 원장의 걸린 매도 잔량. ``intent`` 는 뺀다 — 지금 이 프로세스가 내는 중인 주문 하나뿐이다(작업 시작의
        :meth:`_resume_orders` 가 이전 실행의 의도를 submitted/unknown 으로 바꿔 두므로)."""
        return sum(o.remaining for o in ctx.state.live_orders(code, "sell") if o.status != "intent")

    def _own_qty(self, ctx: _Ctx, code: str) -> int:
        h = ctx.book.position(code)
        return h.qty if h is not None else 0

    def _submit_sell(self, ctx: _Ctx, pos: PositionRec, qty: int, price: int, *, purpose: str, session: str,
                     attempt: int = 0, ref_price: float | None = None) -> OrderRec | None:
        """전략 몫 매도. 장부 보유 − 걸린 자기 매도 잔량(``fill_pending`` 포함)을 넘으면 줄이고, 장부에 없으면 내지 않는다.

        거부·차단이 ``max_exit_rejects`` 번 이어지면 그 매도 의도를 ``halted`` 로 멈추고 사람을 부른다(검토 M1) — 매 틱 재발주로
        하루 수백 건 거부를 쌓지 않는다. 다음 날 preopen 이 한 번 풀어 준다.
        """
        own = self._own_qty(ctx, pos.code)
        avail = own - self._pending_sell(ctx, pos.code)
        qty = min(int(qty), avail)
        if own <= 0 or qty < 1 or price <= 0:
            log.info("매도 생략 %s: 수량 %s(장부 %s, 가용 %s) 가격 %s", pos.code, qty, own, avail, price)
            return None
        rec = self._begin_order(ctx, side="sell", code=pos.code, qty=qty, price=int(price), purpose=purpose,
                                session=session, attempt=attempt, ref_price=ref_price)
        m = ctx.oms.sell(pos.code, qty, int(price))
        self._finish_order(ctx, rec, m)
        log.info("매도 %s %s주 @%s [%s/%s] → %s %s", pos.code, qty, price, purpose, session, rec.status, rec.reason)
        if rec.status in ("blocked", "rejected"):
            pos.exit_rejects += 1
            if pos.exit_intent and pos.exit_rejects >= self.cfg.max_exit_rejects:
                pos.exit_intent["halted"] = True
                pos.notes.append(f"{ctx.day} 매도 {pos.exit_rejects}회 거부 → halted: {rec.reason}")
                self.alert("error", f"매도 중단(needs_human) {pos.code}",
                           f"{pos.exit_rejects}회 연속 거부·차단 — 재발주 멈춤. 마지막 사유: {rec.reason}. "
                           "원인 해결 뒤 다음 preopen 이 다시 시도한다(또는 HTS 수동 청산)")
        else:
            pos.exit_rejects = 0
        return rec

    # ----- 체결 반영 -------------------------------------------------------------

    def _order_by_ord_no(self, ctx: _Ctx, ord_no: str) -> OrderRec | None:
        key = normalize_ord_no(ord_no)
        for o in ctx.state.orders.values():
            if o.ord_no and normalize_ord_no(o.ord_no) == key:
                return o
        return None

    def _sync_fills(self, ctx: _Ctx) -> list[Fill]:
        """브로커 체결(자기 주문만) → 장부·원장. 키움이면 잔고 차이로 보충한다."""
        applied: list[Fill] = []
        try:
            polled = self.bundle.broker.poll_fills()
        except Exception as e:  # noqa: BLE001
            log.warning("체결 조회 실패: %s", e)
            polled = []
        for f in polled:
            o = self._order_by_ord_no(ctx, f.ord_no)
            if o is None:
                self.store.event("foreign_fill", ord_no=f.ord_no, code=f.code, side=f.side, qty=f.qty)
                continue
            if self._apply_fill(ctx, o, f, est=False):
                applied.append(f)
        if self.bundle.fills_from_holdings:
            applied += self._fills_from_holdings(ctx)
        return applied

    def _fills_from_holdings(self, ctx: _Ctx) -> list[Fill]:
        """잔고 차이 → 체결 추정(``ka10076`` 미검증). **살아 있는 자기 주문의 잔량까지만** 체결로 본다.

        차이 = 계좌 − 기준선 사용자 몫 − 전략 장부. 설명 후보는 오늘 낸 주문 중 상태가 submitted/unknown/partial
        (브로커에 걸려 있는 것)뿐이다 — 취소·만료된 주문에는 **배정하지 않는다**(감사 L4: 09:02 에 취소된 진입 종목을
        사용자가 장중에 사면 그 주식이 전략 몫으로 흡수돼 나중에 팔린다). 장전 주문은 09:00 단일가 하나로만 체결되므로
        단일가가 지정가 밖이면(매수: 시가 > 지정가) 후보에서 뺀다.

        설명되지 않는 차이는 장부에 넣지 않는다: 늘어난 쪽은 **사용자 몫으로 기준선에 더하고**(이 전략은 그 주식을 절대
        팔지 않는다), 줄어든 쪽은 기준선을 같은 만큼 빼서(음수 가능 = 계좌에 전략 몫이 모자람) 다음 조회에서 같은 차이를
        다시 세지 않게 한 뒤 경보한다 — 어느 쪽도 사람이 확인한다(``reconcile``).
        """
        try:
            account = {c: h for c, h in self.bundle.broker.holdings().items() if h.qty > 0}
        except Exception as e:  # noqa: BLE001
            log.warning("잔고 조회 실패 — 체결 판정 건너뜀: %s", e)
            return []
        user = ctx.state.baseline_user()
        codes = {o.code for o in ctx.state.orders.values() if o.trade_date == ctx.day} | set(
            strategy_holdings(ctx.book))
        broker_open: dict[str, OpenOrder] | None = None
        if any(o.live and o.ord_no for o in ctx.state.orders.values() if o.trade_date == ctx.day):
            try:
                broker_open = {normalize_ord_no(x.ord_no): x for x in self.bundle.broker.open_orders()}
            except Exception as e:  # noqa: BLE001 — 잔량을 모르면 주문 잔량 전체를 상한으로 쓴다
                log.warning("미체결 조회 실패(부분 체결 상한 없이 배정): %s", e)
        out: list[Fill] = []
        now = self._now()
        for code in sorted(codes):
            acct = account.get(code)
            own_now = (acct.qty if acct else 0) - user.get(code, 0)
            delta = own_now - self._own_qty(ctx, code)
            if delta == 0:
                continue
            side = "buy" if delta > 0 else "sell"
            left = abs(delta)
            cands = [o for o in ctx.state.orders.values()
                     if o.code == code and o.side == side and o.trade_date == ctx.day and o.ord_no
                     and o.live and o.remaining > 0 and self._auction_consistent(ctx, o)]
            cands.sort(key=lambda o: (0 if o.status == "fill_pending" else 1, o.oid))
            for o in cands:
                if left <= 0:
                    break
                cap = o.remaining
                if broker_open is not None:
                    # 검토 M6: 브로커 미체결이 그 주문의 잔량을 알려 주면 "지금까지 체결된 양" 까지만 배정한다 —
                    # 부분 체결 + 같은 날 사용자 매수가 전략 몫으로 과잉 귀속되는 폭을 줄인다.
                    bo = broker_open.get(normalize_ord_no(o.ord_no))
                    if bo is not None:
                        cap = min(cap, max(0, (bo.qty - bo.remaining) - o.filled_qty))
                q = min(left, cap)
                if q <= 0:
                    continue
                f = Fill(ord_no=o.ord_no or "", code=code, side=side, qty=q,
                         price=self._estimate_price(ctx, o, acct), ts=now)
                if self._apply_fill(ctx, o, f, est=True):
                    out.append(f)
                    left -= q
            if left > 0:
                signed = left if side == "buy" else -left
                users = dict(ctx.state.baseline.get("user") or {})
                users[code] = int(users.get(code, 0)) + signed
                ctx.state.baseline["user"] = users
                self.store.event("mismatch", code=code, side=side, unexplained=left, own_now=own_now,
                                 book=self._own_qty(ctx, code), user=users[code])
                if f"mismatch:{code}" not in self._alerted:
                    self._alerted.add(f"mismatch:{code}")
                    what = ("사용자 매수로 보고 기준선에 더함(전략은 팔지 않음)" if side == "buy"
                            else "계좌 보유가 장부보다 적다(전략 매도 체결이 잔고에 늦게 반영됐거나 사용자가 팔았다) — "
                                 "장부 그대로, 재발주 없음, 사람이 확인")
                    self.alert("error" if side == "sell" else "warn", "잔고 차이 설명 안 됨",
                               f"{code}: {side} {left}주 — {what}, reconcile 확인")
        return out

    def _auction_consistent(self, ctx: _Ctx, o: OrderRec) -> bool:
        """장전 주문은 09:00 단일가 하나로만 체결된다 — 단일가가 지정가 밖이면 이 주문의 체결일 수 없다."""
        if o.session != "open_auction":
            return True
        px = ctx.auction_px.get(o.code)
        if px is None:
            return True     # 시가를 모르면 가격으로 거르지 못한다(상태 조건만)
        return px <= o.price if o.side == "buy" else px >= o.price

    def _estimate_price(self, ctx: _Ctx, o: OrderRec, acct: Any) -> int:
        if o.session in ("open_auction", "close_auction") and o.code in ctx.auction_px:
            return int(ctx.auction_px[o.code])
        if o.side == "buy":
            if acct is not None and self._own_qty(ctx, o.code) == 0 and acct.avg_price > 0:
                return int(round(acct.avg_price))
            return int(min(o.price, ctx.last_px.get(o.code, o.price)))
        return int(max(o.price, ctx.last_px.get(o.code, o.price)))

    def _apply_fill(self, ctx: _Ctx, o: OrderRec, f: Fill, *, est: bool) -> bool:
        try:
            realized = ctx.book.apply(f)
        except ValueError as e:
            self.store.event("unmatched_fill", ord_no=f.ord_no, code=f.code, side=f.side, qty=f.qty, error=str(e))
            self.alert("error", "장부가 체결을 못 받음", f"{f.code} {f.side} {f.qty}: {e}")
            return False
        o.filled_qty += f.qty
        if o.status in ("submitted", "unknown", "partial", "fill_pending"):
            o.status = "filled" if o.remaining == 0 else ("fill_pending" if o.status == "fill_pending" else "partial")
        self.store.event("fill", oid=o.oid, ord_no=f.ord_no, code=f.code, side=f.side, qty=f.qty, price=f.price,
                         purpose=o.purpose, realized=realized, price_estimated=est)
        pos = ctx.state.positions.get(f.code)
        if pos is None:
            self.alert("error", "포지션 없는 체결", f"{f.code} {f.side} {f.qty}@{f.price}")
            return True
        if f.side == "buy":
            self._on_entry_fill(ctx, pos, f)
        else:
            self._on_exit_fill(ctx, pos, f, realized)
        return True

    def _on_entry_fill(self, ctx: _Ctx, pos: PositionRec, f: Fill) -> None:
        pos.qty += f.qty
        pos.entered_qty += f.qty
        if pos.status != "open":
            pos.status = "open"
            pos.entry_date = ctx.day
            pos.entry_price = float(f.price)
            pos.last_day = ctx.day
            plan = _plan_from_dict(pos.plan)
            st = PositionState.from_plan(plan, self._cfg_of(pos), qty=float(f.qty))
            acts = st.on_fill_open(float(f.price), day_no=1, entry_day=ctx.asof)
            self._store_strat(pos, st)
            self.store.event("position_open", code=pos.code, qty=f.qty, price=f.price, stop=st.stop_px,
                             target=st.target_px, d=st.d, cfg_hash=pos.cfg_hash)
            self._apply_actions(ctx, pos, st, acts, session="continuous")
        else:   # 같은 주문의 늦은 추가 체결 — 상태 수량을 늘린다
            st = self._strat(pos)
            st.qty += f.qty
            st.qty_open += f.qty
            self._store_strat(pos, st)

    def _on_exit_fill(self, ctx: _Ctx, pos: PositionRec, f: Fill, realized: float) -> None:
        pos.qty = self._own_qty(ctx, pos.code)
        pos.realized += float(realized)
        if pos.exit_intent:
            left = int(pos.exit_intent.get("qty", 0)) - f.qty
            pos.exit_intent["qty"] = max(0, left)
            if left <= 0:
                pos.exit_reason = pos.exit_intent.get("kind", pos.exit_reason)
                pos.exit_intent = None
                pos.exit_attempt = 0
        if pos.qty <= 0:
            for o in ctx.state.live_orders(pos.code, "sell"):
                self._cancel(ctx, o, reason="position_closed")
            reason = pos.exit_reason or "exit"
            ctx.state.close_position(pos.code, date=ctx.day, reason=reason)
            self.store.event("position_close", code=pos.code, reason=reason, realized=pos.realized)
            self.alert("info", f"청산 {pos.code}", f"{reason} 실현 {pos.realized:,.0f}원")

    # ----- 전략 상태 ---------------------------------------------------------------

    def _cfg_of(self, pos: PositionRec) -> StrategyConfig:
        """이 포지션에 고정된 전략 설정(통합 검토 M3). 계획 시점 dict 에서 복원하고 해시로 캐시한다. 없으면(옛 스냅샷) 현재 설정."""
        if not pos.cfg:
            return self.cfg.strategy
        key = pos.cfg_hash or strategy_hash(pos.cfg)
        cfg = self._cfg_cache.get(key)
        if cfg is None:
            cfg = _strategy_from_dict(pos.cfg)
            self._cfg_cache[key] = cfg
        return cfg

    def _pin_cfg(self, pos: PositionRec) -> None:
        pos.cfg = self.cfg.strategy.to_dict()
        pos.cfg_hash = self.cfg_hash

    def _strat(self, pos: PositionRec) -> PositionState:
        if not pos.strat:
            raise StateCorrupt(f"{pos.code}: 전략 상태 없음")
        return PositionState.from_dict(pos.strat, self._cfg_of(pos))

    def _store_strat(self, pos: PositionRec, st: PositionState) -> None:
        pos.strat = st.to_dict()
        pos.stop = st.stop_px
        pos.target = st.target_px
        pos.partial_done = st.partial_hit
        pos.breakeven = st.breakeven_from is not None

    def _apply_actions(self, ctx: _Ctx, pos: PositionRec, st: PositionState, acts: list[Action], *,
                       session: str) -> None:
        """전략 Action → 매도 의도(``exit_intent``). 실제 주문은 :meth:`_work_exits`·작업별 동시호가 주문이 낸다."""
        for a in acts:
            self.store.event("action", code=pos.code, kind=a.kind, qty=a.qty, qty_frac=a.qty_frac, price=a.price,
                             at=a.at, day_no=a.day_no)
            if a.kind == "partial":
                q = min(int(math.floor(a.qty + 1e-9)), pos.qty)
                if q < 1:
                    log.info("%s 절반 익절 0주(보유 %s) — 상태만 익절 뒤로", pos.code, pos.qty)
                    continue
                if pos.exit_intent and pos.exit_intent.get("kind") != "partial":
                    continue  # 이미 전량 청산 중
                pos.exit_intent = {"kind": "partial", "qty": q, "session": session, "created": ctx.day}
            elif a.at == "next_open" or a.kind.endswith("_next_open"):   # 돌파 실패·MA50 추적 등 "다음날 시가" 결정
                pos.exit_intent = {"kind": a.kind, "qty": pos.qty, "session": "open_auction", "created": ctx.day}
            else:
                upgrade = pos.exit_intent is not None and pos.exit_intent.get("kind") == "partial"
                pos.exit_intent = {"kind": a.kind, "qty": pos.qty, "session": session, "created": ctx.day}
                if upgrade:  # 익절 주문이 걸려 있는데 전량 청산이 왔다 → 다시 낸다
                    for o in ctx.state.live_orders(pos.code, "sell"):
                        self._cancel(ctx, o, reason="upgrade_to_full_exit")
            pos.exit_attempt = 0
            self.alert("info", f"청산 결정 {pos.code}", f"{a.kind} {pos.exit_intent['qty']}주 ({pos.exit_intent['session']})")
        self._store_strat(pos, st)

    def _start_day(self, ctx: _Ctx, pos: PositionRec, open_px: float | None) -> None:
        """보유 포지션의 거래일 시작(하루 한 번): day_no + 1, 시가 판단(:meth:`PositionState.on_open`)."""
        if pos.status != "open" or pos.last_day == ctx.day or not pos.strat:
            return
        st = self._strat(pos)
        if pos.last_day and (ctx.asof - date.fromisoformat(pos.last_day)).days > 4:
            self.alert("warn", "보유일 공백", f"{pos.code}: 마지막 {pos.last_day} → {ctx.day} (day_no 가 밀릴 수 있다)")
        day_no = st.day_no + 1
        pos.last_day = ctx.day
        if open_px is None or not (open_px > 0):
            st.day_no = day_no
            self._store_strat(pos, st)
            return
        acts = st.on_open(float(open_px), day_no)
        if pos.exit_intent and pos.exit_intent.get("session") == "open_auction":
            # 장전 동시호가 매도가 이미 나가 있다(추적 결정) — 의도는 그대로 두고 상태만 갱신한다.
            acts = [a for a in acts if not a.kind.endswith("_next_open")]
            if acts and acts[0].qty_frac >= 1.0:
                pos.exit_intent["session"] = "continuous"
        self._apply_actions(ctx, pos, st, acts, session="continuous")

    # ----- 매도 집행 -------------------------------------------------------------

    def _work_exits(self, ctx: _Ctx, codes: Iterable[str] | None = None) -> None:
        """연속매매 매도 의도 처리: 걸린 주문이 ``exit_retry_sec`` 넘게 안 채워지면 취소하고 더 싸게 다시 낸다."""
        now = self._now()
        for pos in list(ctx.state.open_positions()):
            if codes is not None and pos.code not in codes:
                continue
            it = pos.exit_intent
            if not it or it.get("session") != "continuous" or int(it.get("qty", 0)) < 1 or it.get("halted"):
                continue
            last = ctx.last_px.get(pos.code)
            if not last:
                continue
            live = ctx.state.live_orders(pos.code, "sell")
            if any(o.status in ("fill_pending", "intent") for o in live):
                continue            # 체결 확인 대기 중 — 같은 수량을 다시 내지 않는다(검토 C2)
            if live:
                age = min((now - datetime.fromisoformat(o.ts)).total_seconds() for o in live)
                if age < self.cfg.exit_retry_sec:
                    continue
                for o in live:
                    self._cancel(ctx, o, reason="exit_reprice")
                if self.bundle.fills_from_holdings:
                    self._sync_fills(ctx)   # 취소 직전 체결분 반영
                if ctx.state.positions.get(pos.code) is not pos or not pos.exit_intent:
                    continue
            pos.exit_attempt += 1
            px = self._exit_limit(last, pos.exit_attempt)
            q = ctx.quotes.get(pos.code)
            if q is not None and q.prev_close:
                px = max(px, int(limit_down_price(q.prev_close)))
            self._submit_sell(ctx, pos, int(pos.exit_intent["qty"]), px, purpose=pos.exit_intent["kind"],
                              session="continuous", attempt=pos.exit_attempt, ref_price=last)

    def _exit_limit(self, last: float, attempt: int) -> int:
        """연속매매 매도 지정가. 1차 = 현재가 − 1틱(터치 즉시, 최우선 매수호가 근처), 이후 60초마다 0.5%씩 더 싸게(상한 3%).

        감사 L1: 백테스트는 손절을 손절가 그대로 체결하므로 라이브는 터치 즉시 가장 가까운 호가로 내는 것이 차이를 가장
        작게 한다. 미끄러짐 추정은 런북 "라이브-백테스트 차이".
        """
        base = _tick_down(last)
        if attempt <= 1:
            try:
                return int(shift_ticks(base, -1))
            except ValueError:
                return base
        disc = min(self.cfg.exit_discount_step * (attempt - 1), self.cfg.exit_discount_cap)
        return _tick_down(last * (1.0 - disc))

    def _auction_sell(self, ctx: _Ctx, pos: PositionRec, ref: float, *, session: str,
                      prev_close: float | None) -> None:
        """동시호가 매도: 지정가 = 기준가 × (1 − floor) 호가 올림 — 단일가가 그 위면 단일가 체결(사실상 시장가, 하한 있음)."""
        it = pos.exit_intent
        if not it or it.get("halted") or ctx.state.live_orders(pos.code, "sell"):
            return
        px = _tick_up(ref * (1.0 - self.cfg.auction_sell_floor))
        if prev_close:
            px = max(px, int(limit_down_price(prev_close)))
        it["session"] = session
        self._submit_sell(ctx, pos, int(it["qty"]), px, purpose=it["kind"], session=session, ref_price=ref)

    def _quote(self, ctx: _Ctx, code: str) -> QuoteSnap | None:
        try:
            q = self.bundle.quotes.quote(code)
        except Exception as e:  # noqa: BLE001 — 그 종목만 건너뛴다
            log.warning("시세 실패 %s: %s", code, e)
            return None
        ctx.quotes[code] = q
        ctx.last_px[code] = float(q.last)
        if q.date == ctx.asof and ctx.state.sessions.get(ctx.day) != "open":
            # 시세 자체가 오늘 날짜를 달고 왔다 = 신선한 증거(검토 F4). ``open_pric`` 이 있다는 것만으로는 확정하지 않는다 —
            # 휴장일에 직전 거래일 값을 캐시로 돌려줄 수 있다.
            ctx.state.sessions[ctx.day] = "open"
        return q

    # ----- signal ----------------------------------------------------------------

    def _job_signal(self, ctx: _Ctx, *, force: bool = False) -> None:
        st = ctx.state
        if st.jobs_done.get("signal") == ctx.day and not force:
            log.info("signal: %s 이미 완료", ctx.day)
            return
        if not self._weekday_open(ctx.asof):
            log.info("signal: %s 휴장(달력) — 건너뜀", ctx.day)
            return
        if self.signal_source is None:
            self.signal_source = FeaturesSignalSource(self.cfg.strategy)
        self._expire_stale(ctx)
        held = [p.code for p in st.open_positions()] + [p.code for p in st.shadow.values()
                                                        if p.status in ("open", "planned")]
        since_s = [p.last_eod or p.entry_date for p in st.open_positions() if (p.last_eod or p.entry_date)]
        since = date.fromisoformat(min(since_s)) if since_s else None
        if since is not None and int(np.busday_count(since, ctx.asof)) > CATCHUP_DAYS + 1:
            self.alert("warn", "장기 결손 따라잡기", f"마지막 평가 {since} → {ctx.day}: {CATCHUP_DAYS}거래일 넘는 결손을 한 번에 평가한다")
        try:
            batch = self.signal_source.load(ctx.asof, held=held, since=since)
        except Exception as e:  # noqa: BLE001 — 신호 원천 실패 = 오늘 신규 진입 없음(보유 청산은 계속)
            self.store.event("signal_abort", asof=ctx.day, reason=f"{type(e).__name__}: {e}")
            log.exception("signal 중단: 신호 원천 실패")
            self.alert("error", "신호 중단 — 오늘 신규 진입 없음", f"신호 원천 실패: {type(e).__name__}: {e}")
            return
        # 1) 최신일 = 직전 거래일(키움 기준 종목 일봉 날짜 — 휴장일도 정확히 건너뛴다)
        prev = session_date(self.bundle.quotes, ctx.asof - timedelta(days=1))
        if prev is None:
            self._abort_signal(ctx, "직전 거래일 확인 실패(ka10081)", batch)
            return
        if batch.data_date != prev:
            if batch.krx_patched and batch.data_date < prev:
                self._settle_equity(ctx, batch)      # 검토 M5: 있는 날짜까지의 보유 종목 결정·자산은 수행한다
            self._abort_signal(ctx, f"DB 최신일 {batch.data_date} ≠ 직전 거래일 {prev} (daily_finalize 미완?)", batch)
            return
        # 2) NXT 보정 확인(2026-09-14 이후 일봉은 보정본이 아니면 쓰지 않는다)
        for msg in batch.alerts:
            self.alert("warn", "신호 원천 경고", msg)
        if not batch.krx_patched:
            self._abort_signal(ctx, f"{batch.data_date} KRX 정규장 보정이 돌지 않았다(krx_patched=False)", batch)
            return
        # 2b) 입력 신선도(통합 검토 M1): 수급·실적이 하루 늦으면 후보가 조용히 0 이 된다 — 보유 결정·자산은 하고 진입만 중단
        if not batch.freshness.get("ok", True):
            self._settle_equity(ctx, batch)
            self._shadow_settle(ctx, batch)
            self._abort_signal(ctx, "입력 신선도 실패: " + "; ".join(batch.freshness.get("problems", [])), batch)
            return
        cands = list(batch.candidates)
        if batch.patched_codes is not None:
            unpatched = [c.code for c in cands if c.code not in batch.patched_codes]
            if unpatched:
                self.alert("warn", "보정 안 된 후보 제외", ", ".join(unpatched))
            cands = [c for c in cands if c.code in batch.patched_codes]
        # 3) 보유 종목: 마지막 평가일 이후 거래일마다 KRX 종가(DB 보정본)로 MA50 추적·놓친 종가 결정 → 오늘 시가 매도 의도
        # 4) 그 날짜들의 장 마감 자산(확정) → 다일 낙폭 킬(정지 파일)·일 손실 킬 기준
        row = self._settle_equity(ctx, batch)
        # 5) 계획(수량은 preopen 이 기준선·현금·슬롯으로 다시 자른다)
        n_keep = self.cfg.strategy.trail_ma
        equity = self._sizing_equity(ctx)
        n_new = 0
        for c in cands:
            if c.code in st.positions:      # 같은 종목 보유·계획 중이면 재진입 없음(연구 no_overlap)
                continue
            plan = plan_entry(c, equity, c.close, self.cfg.strategy)
            st.positions[c.code] = PositionRec(
                code=c.code, market=c.market, signal_date=batch.data_date.isoformat(), trade_date=ctx.day,
                status="planned", name=batch.names.get(c.code, ""), plan=_plan_to_dict(plan),
                trail_closes=[float(x) for x in batch.trail_closes.get(c.code, [])][-n_keep:],
            )
            self._pin_cfg(st.positions[c.code])
            st.market_of[c.code] = c.market
            n_new += 1
        n_shadow = self._shadow_plan(ctx, batch, equity, n_keep)
        st.last_signal = {"asof": ctx.day, "data_date": batch.data_date.isoformat(), "n_candidates": len(cands),
                          "codes": [c.code for c in cands], "n_vetoed": len(batch.vetoed), "shadow_planned": n_shadow}
        st.jobs_done["signal"] = ctx.day
        self.store.event("signal", asof=ctx.day, data_date=batch.data_date.isoformat(),
                         candidates=[c.code for c in cands], planned=n_new, cfg_hash=self.cfg_hash,
                         freshness=batch.freshness)
        log.info("signal: %s(d=%s) 후보 %d · 새 계획 %d", ctx.day, batch.data_date, len(cands), n_new)
        self.alert("info", f"신호 {ctx.day}", self._report(ctx, row) + (
            f"\n후보 {len(cands)}: {', '.join(c.code for c in cands[:20])}" if cands else "\n후보 없음") + (
            f"\n거부(섀도) {len(batch.vetoed)}: {', '.join(c.code for c in batch.vetoed[:20])}" if batch.vetoed else ""))

    def _settle_equity(self, ctx: _Ctx, batch: SignalBatch) -> dict[str, Any]:
        """보유 종목 종가 결정(따라잡기 포함) + 날짜별 자산 행·낙폭 킬. 마지막 행을 돌려준다."""
        st = ctx.state
        by_date = self._decide_on_db_close(ctx, batch)
        self._shadow_settle(ctx, batch)
        done = {r["date"] for r in st.equity_history}
        last_prices: dict[str, float] = {}
        row: dict[str, Any] = {}
        d = batch.data_date.isoformat()
        for dt in sorted(set(by_date) | {d}):
            last_prices.update(by_date.get(dt, {}))
            if dt in done and dt != d:
                continue
            equity, cash, mv = self._equity(ctx, last_prices)
            row = drawdown_check(st, date=dt, equity=equity, cash=cash, market_value=mv,
                                 kill_pct=self.cfg.risk.drawdown_kill, stop_file=self.cfg.mode_stop,
                                 on_trigger=lambda msg: self.alert("error", "다일 낙폭 킬 — 신규 진입 정지", msg))
            self.store.event("equity", **row, positions=len(st.open_positions()))
        if st.drawdown_kill.get("active"):
            self.store.event("kill", kind="drawdown", **{k: v for k, v in st.drawdown_kill.items() if k != "active"})
        ctx.kill = self._killswitch(st, ctx.asof)        # 새 자산 행으로 일 손실 킬 다시 계산
        ctx.oms.kill = ctx.kill
        return row

    def _abort_signal(self, ctx: _Ctx, why: str, batch: SignalBatch) -> None:
        self.store.event("signal_abort", asof=ctx.day, reason=why, data_date=batch.data_date.isoformat())
        log.error("signal 중단: %s", why)
        self.alert("error", "신호 중단 — 오늘 신규 진입 없음", why)

    # ----- preopen ---------------------------------------------------------------

    def _job_preopen(self, ctx: _Ctx) -> None:
        st = ctx.state
        self._expire_stale(ctx)
        for pos in st.open_positions():          # 어제 멈춘 매도 의도(M1 halted)는 하루 한 번 다시 시도한다
            if pos.exit_intent and pos.exit_intent.get("halted"):
                pos.exit_intent.pop("halted", None)
                pos.exit_rejects = 0
                self.alert("warn", f"매도 재시도 {pos.code}", "어제 halted 된 매도 의도를 오늘 다시 낸다")
        # 1) 계좌 기준선(하루 한 번, 첫 주문 전). 잔고 차이 체결 판정은 기준선을 잡은 **뒤에만** 한다 — 전날 기준선으로
        #    보면 사용자의 전날 장중 매매가 전부 "설명 안 되는 차이"로 읽힌다.
        if st.baseline.get("date") != ctx.day:
            try:
                account = {c: h.qty for c, h in self.bundle.broker.holdings().items() if h.qty > 0}
            except Exception as e:  # noqa: BLE001
                self.alert("error", "기준선 조회 실패 — 오늘 신규 진입 없음", str(e))
                account = None
            if account is not None:
                user, short = compute_baseline_user(account, strategy_holdings(ctx.book))
                st.baseline = {"date": ctx.day, "user": user, "account": account}
                self.store.event("baseline", date=ctx.day, user=user, account=account, short=short)
                if short:
                    self.alert("error", "장부 > 계좌 — 전략 몫이 계좌에 없다", str(short))
        self._sync_fills(ctx)
        # 2) 전일 결정 매도(장전 동시호가) — 킬과 무관
        for pos in st.open_positions():
            it = pos.exit_intent
            if not it:
                continue
            if it.get("session") != "open_auction":
                it["session"] = "open_auction"   # 어제 못 판 연속매매 의도도 오늘 시가에 판다
            q = self._quote(ctx, pos.code)
            ref = (q.prev_close if q and q.prev_close else None) or (pos.trail_closes[-1] if pos.trail_closes else None)
            if not ref:
                self.alert("warn", "장전 매도 기준가 없음", f"{pos.code}: open_check 이후 연속매매로 판다")
                it["session"] = "continuous"
                continue
            self._auction_sell(ctx, pos, ref, session="open_auction", prev_close=ref)
        # 3) 진입
        planned = [p for p in st.positions.values() if p.status == "planned" and p.trade_date == ctx.day]
        if not planned:
            return
        block = self.entry_block_reason(ctx)
        if st.baseline.get("date") != ctx.day:
            block = block or "기준선 없음"
        if block:
            for p in planned:
                self._skip(ctx, p, f"entry_blocked: {block}")
            self.alert("warn", "신규 진입 정지", block)
            return
        user = st.baseline_user()
        held = {p.code for p in st.open_positions()} | set(user)
        for p in planned:
            if p.code in user:
                self._skip(ctx, p, "user_holds")       # 사용자 보유 종목은 사지 않는다
            elif p.code in st.blocked_codes:            # 미해결 주문이 있는 종목(검토 F1·F2) — 사람이 풀 때까지
                self._skip(ctx, p, f"blocked: {st.blocked_codes[p.code]}")
        plans = []
        for p in [p for p in planned if p.status == "planned"]:
            plan = _plan_from_dict(p.plan)
            q = self._quote(ctx, p.code)
            krx_prev = q.prev_close if q else None
            if krx_prev and abs(krx_prev / plan.ref_px - 1.0) > self.cfg.prev_close_tolerance:
                self.store.event("prev_close_mismatch", code=p.code, plan=plan.ref_px, krx=krx_prev)
                log.warning("%s 기준가 차이 계획 %s vs KRX %s — KRX 로 지정가 재계산", p.code, plan.ref_px, krx_prev)
            if krx_prev:
                limit_px, lu = entry_limit(plan.pivot, krx_prev, self.cfg.strategy)
                plan = EntryPlan(**{**asdict(plan), "limit_px": limit_px, "limit_up_px": lu, "ref_px": krx_prev})
                p.plan = _plan_to_dict(plan)
            plans.append(plan)
        equity, cash, _ = self._equity(ctx)
        cash -= sum(o.remaining * o.price for o in st.live_orders(side="buy"))
        slots = self.cfg.strategy.max_positions - len(st.open_positions()) - len(
            [p for p in st.positions.values() if p.status == "entry_pending"])
        chosen = {pl.code: pl for pl in allocate(plans, cash=cash, open_slots=slots, held=held)}
        sz_equity = self._sizing_equity(ctx)
        for p in [p for p in planned if p.status == "planned"]:
            pl = chosen.get(p.code)
            if pl is None:
                self._skip(ctx, p, "slots_or_cash")
                continue
            self._submit_entry(ctx, p, pl, sz_equity)

    def _expire_stale(self, ctx: _Ctx) -> None:
        """지난 거래일의 진입 계획(그날 preopen·open_check 가 안 돌았다)은 만료 — 남겨 두면 그 종목 재진입을 영영 막는다."""
        for p in list(ctx.state.positions.values()):
            if p.status in ("planned", "entry_pending") and p.trade_date < ctx.day:
                for o in ctx.state.live_orders(p.code, "buy"):
                    self._cancel(ctx, o, reason="stale_plan")
                p.status = "expired"
                p.exit_reason = "stale_plan"
                ctx.state.close_position(p.code, date=ctx.day, reason="stale_plan")
                self.store.event("entry_expired", code=p.code, trade_date=p.trade_date)

    def _skip(self, ctx: _Ctx, p: PositionRec, reason: str) -> None:
        p.status = "skipped"
        p.exit_reason = reason
        ctx.state.close_position(p.code, date=ctx.day, reason=reason)
        self.store.event("entry_skip", code=p.code, reason=reason)

    def _submit_entry(self, ctx: _Ctx, p: PositionRec, plan: EntryPlan, equity: float) -> None:
        qty, limit = int(plan.qty), int(plan.limit_px)
        # 사이징은 기준가(전일종가)로 10% — 지정가는 피벗×1.05 까지라 그만큼 + 2% 여유를 둔다.
        cap_notional = equity * self.cfg.strategy.max_position_frac * (1.0 + self.cfg.strategy.chase + 0.02)
        if qty < 1 or limit <= 0 or qty * limit > cap_notional:
            self._skip(ctx, p, f"sanity: qty={qty} limit={limit} notional>{cap_notional:,.0f}")
            return
        ctx.oms.guard.config = entry_guard_config(
            equity=equity, price=limit, hard_cap_frac=self.cfg.risk.max_qty_hard_cap_frac,
            price_band_pct=self.cfg.risk.price_band_pct,
            max_orders_total=2 * self.cfg.strategy.max_positions)
        # POST 전에 entry_pending + 주문 의도를 저장한다(검토 C1) — 재실행은 이 계획을 다시 사지 않는다.
        p.status = "entry_pending"
        p.plan = {**p.plan, "qty": float(qty)}
        rec = self._begin_order(ctx, side="buy", code=p.code, qty=qty, price=limit, purpose="entry",
                                session="open_auction", ref_price=plan.ref_px)
        m = ctx.oms.buy(p.code, qty, limit, ref_price=plan.ref_px)
        self._finish_order(ctx, rec, m)
        log.info("매수 %s %s주 @%s → %s %s", p.code, qty, limit, rec.status, rec.reason)
        if rec.status not in ("submitted", "unknown", "dry_run"):
            p.status = "planned"
            self._skip(ctx, p, f"order_{rec.status}: {rec.reason}")

    # ----- open_check --------------------------------------------------------------

    def _resolve_open(self, ctx: _Ctx, *, allow_proxy: bool) -> None:
        """장전 주문의 09:00 단일가 체결 판정. 세션이 거래일로 확정된 종목만, 시가가 있는 종목만 처리한다.

        검토 M4: ``open_pric`` 이 없으면 ``missed`` 로 치지 않는다 — 경보하고 주문을 그대로 둔다. ``allow_proxy``(09:05 이후
        monitor 틱)면 현재가를 단일가 대용으로 쓴다(경보 1회, 원장에 ``open_proxy``). 세션이 의심 상태(검토 M3)면 아무것도 안 한다.
        """
        st = ctx.state
        if st.sessions.get(ctx.day) != "open":
            return
        pending = {p.code for p in st.positions.values() if p.status == "entry_pending" and p.trade_date == ctx.day}
        auction_sells = {o.code for o in st.live_orders(side="sell") if o.session == "open_auction"}
        for code in sorted(pending | auction_sells):
            if code in ctx.auction_px:
                continue
            q = ctx.quotes.get(code) or self._quote(ctx, code)
            if q is None:
                continue
            px, proxy = (float(q.open), False) if q.open else ((float(q.last), True) if allow_proxy and q.last else (None, False))
            if px is None:
                if f"open_missing:{code}" not in self._alerted:
                    self._alerted.add(f"open_missing:{code}")
                    self.alert("error", "시가(open_pric) 없음 — 체결 판정 보류", f"{code}: 주문은 그대로, {OPEN_PROXY_FROM:%H:%M} 이후 현재가로 대용")
                continue
            if proxy:
                self.store.event("open_proxy", code=code, last=px)
                if f"open_proxy:{code}" not in self._alerted:
                    self._alerted.add(f"open_proxy:{code}")
                    self.alert("warn", "시가 대용(현재가) 사용", f"{code}: open_pric 없음 → 단일가 = 현재가 {px:,.0f} 로 판정")
            ctx.auction_px[code] = px
            if self.cfg.mode == "paper":
                feed_auction(self.bundle.broker, code, auction_ts(ctx.asof, "0900"), px)
        self._sync_fills(ctx)
        # 단일가를 아는 종목의 미체결 장전 주문: 매수는 취소(추격 없음), 매도는 취소 후 연속매매로
        for o in list(st.live_orders()):
            if o.session != "open_auction" or o.trade_date != ctx.day or o.code not in ctx.auction_px:
                continue
            self._cancel(ctx, o, reason="open_unfilled")
            pos = st.positions.get(o.code)
            if pos and o.side == "sell" and pos.exit_intent:
                pos.exit_intent["session"] = "continuous"
        if self.bundle.fills_from_holdings:
            self._sync_fills(ctx)
        for p in list(st.positions.values()):
            if (p.status == "entry_pending" and p.trade_date == ctx.day and p.code in ctx.auction_px
                    and not st.live_orders(p.code, "buy")):
                p.status = "missed"
                op = ctx.auction_px.get(p.code)
                p.exit_reason = "dry_run" if self.bundle.dry_run else f"open {op} > limit {p.plan.get('limit_px')}"
                st.close_position(p.code, date=ctx.day, reason=p.exit_reason)
                self.store.event("entry_missed", code=p.code, open=op, limit=p.plan.get("limit_px"))

    def _job_open_check(self, ctx: _Ctx) -> None:
        st = ctx.state
        codes = sorted({o.code for o in st.live_orders()} | {p.code for p in st.open_positions()}
                       | {p.code for p in st.positions.values() if p.status == "entry_pending"})
        for code in codes:
            self._quote(ctx, code)
        self._confirm_session(ctx)
        self._resolve_open(ctx, allow_proxy=False)
        self._sync_fills(ctx)
        self._shadow_open(ctx)
        if st.sessions.get(ctx.day) == "open":
            for pos in list(st.open_positions()):
                self._start_day(ctx, pos, ctx.auction_px.get(pos.code))
        self._work_exits(ctx)
        self._paper_refeed(ctx)

    # ----- monitor -----------------------------------------------------------------

    def _interval(self) -> float:
        hhmm = self._now().strftime("%H:%M")
        for lo, hi in self.cfg.quiet_windows:
            if lo <= hhmm < hi:
                return self.cfg.quiet_poll_sec
        return self.cfg.poll_sec

    def _job_monitor(self, ctx: _Ctx, *, once: bool = False) -> None:
        while True:
            self._monitor_tick(ctx)
            self.store.save(ctx.state)
            self.bundle.persist()
            busy = ctx.state.open_positions() or [p for p in ctx.state.positions.values()
                                                  if p.status == "entry_pending" and p.trade_date == ctx.day]
            if once or self._now().time() >= self.cfg.monitor_until or not busy:
                if not once and not busy:
                    log.info("monitor: 보유·대기 없음 — 종료")
                return
            self.sleeper(self._interval())
            if self._now().time() >= self.cfg.monitor_until:
                return

    def _monitor_tick(self, ctx: _Ctx) -> None:
        st = ctx.state
        codes = {p.code for p in st.open_positions()} | {
            p.code for p in st.positions.values() if p.status == "entry_pending" and p.trade_date == ctx.day}
        for code in sorted(codes):
            self._quote(ctx, code)
        self._confirm_session(ctx)                      # 검토 M3: 휴장 의심은 틱마다 다시 본다
        if self.cfg.mode == "paper" and st.sessions.get(ctx.day) == "open":
            for pos in list(st.open_positions()):       # 검토 F4: 세션 확정 전엔 (캐시일 수 있는) 시세로 체결시키지 않는다
                q = ctx.quotes.get(pos.code)
                if q is not None:
                    feed_quote(self.bundle.broker, q)
        if any(p.status == "entry_pending" and p.trade_date == ctx.day for p in st.positions.values()):
            self._resolve_open(ctx, allow_proxy=self._now().time() >= OPEN_PROXY_FROM)   # 검토 M4
        self._sync_fills(ctx)
        if st.sessions.get(ctx.day) != "open":
            return                                      # 세션 미확정: 보유일을 올리거나 판단하지 않는다(체결 반영은 했다)
        for pos in list(st.open_positions()):
            q = ctx.quotes.get(pos.code)
            if q is None:
                continue
            self._start_day(ctx, pos, q.open or q.last)
            if pos.code not in st.positions:
                continue
            st_ = self._strat(pos)
            acts = st_.on_quote(float(q.last))
            if acts:
                self._apply_actions(ctx, pos, st_, acts, session="continuous")
        self._work_exits(ctx)
        self._paper_refeed(ctx)

    def _paper_refeed(self, ctx: _Ctx) -> None:
        """페이퍼: 방금 낸 매도 지정가가 시장가 쪽이면 같은 시세에서 바로 체결되게 한 번 더 먹인다(연속매매 즉시 체결 근사)."""
        if self.cfg.mode != "paper":
            return
        fresh = {o.code for o in ctx.state.live_orders(side="sell") if o.session == "continuous"}
        for code in fresh:
            q = ctx.quotes.get(code)
            if q is not None:
                feed_quote(self.bundle.broker, q)
        if fresh:
            self._sync_fills(ctx)

    # ----- close -------------------------------------------------------------------

    def _job_close(self, ctx: _Ctx) -> None:
        st = ctx.state
        for pos in list(st.open_positions()):
            q = self._quote(ctx, pos.code)
            if q is None:
                self.alert("warn", "종가 판단 시세 없음", pos.code)
        self._confirm_session(ctx)
        if self.cfg.mode == "paper" and st.sessions.get(ctx.day) == "open":
            for pos in list(st.open_positions()):
                q = ctx.quotes.get(pos.code)
                if q is not None:
                    feed_quote(self.bundle.broker, q)
        if any(p.status == "entry_pending" and p.trade_date == ctx.day for p in st.positions.values()):
            self._resolve_open(ctx, allow_proxy=True)
        self._sync_fills(ctx)
        if st.sessions.get(ctx.day) != "open":
            self.alert("warn", "종가 동시호가 매도 보류", f"{ctx.day}: 세션 미확정(휴장 의심)")
            return
        # 설계 §6(2026-10-10): 시간 손절 폐지. 종가 기준 결정(돌파 실패·MA50 추적·최대 보유 안전 상한)은 전부 다음날 07:35
        # signal 이 DB 보정 종가로 한다. 이 작업은 **오늘 아직 못 판 매도 의도**를 종가 동시호가에 내는 마지막 기회일 뿐이다.
        for pos in list(st.open_positions()):
            q = ctx.quotes.get(pos.code)
            if q is None:
                continue
            self._start_day(ctx, pos, q.open or q.last)
            if pos.code not in st.positions:
                continue
            it = pos.exit_intent
            if it and not st.live_orders(pos.code, "sell"):
                self._auction_sell(ctx, pos, float(q.last), session="close_auction", prev_close=q.prev_close)

    # ----- eod ---------------------------------------------------------------------

    def _job_eod(self, ctx: _Ctx, *, force: bool = False) -> None:
        """장 마감 **장부 정리·보고만**. MA50 추적·자산·낙폭 킬 결정은 다음날 07:35 ``signal`` 이 DB 보정(KRX) 종가로 한다.

        감사 L5: 15:31 이후 ``ka10001 cur_prc`` 가 KRX 15:30 단일가라는 근거가 없다(같은 6자리 코드의 일봉이 NXT 통합값을
        준 것이 C1). 그래서 여기서 읽는 가격은 종가 동시호가 체결가 **추정**(``price_estimated``)과 보고용 잠정 평가에만 쓰고,
        결정에는 쓰지 않는다. 자산 이력(``equity_history``)도 여기서 쓰지 않는다.
        """
        st = ctx.state
        codes = sorted({o.code for o in st.live_orders()} | {p.code for p in st.open_positions()})
        for code in codes:
            self._quote(ctx, code)
        self._confirm_session(ctx)
        for code in codes:
            q = ctx.quotes.get(code)
            if q is None:
                continue
            if st.sessions.get(ctx.day) != "open":
                continue                                  # 검토 F4: 세션 미확정이면 체결시키지 않는다(주문은 아래서 거둔다)
            ctx.auction_px[code] = float(q.last)       # 종가 동시호가 체결가 추정(검증 안 된 값 — 결정에 안 쓴다)
            if self.cfg.mode == "paper":
                feed_auction(self.bundle.broker, code, auction_ts(ctx.asof, "1530"), float(q.last))
        self._sync_fills(ctx)
        # 장이 끝났으니 남은 주문은 전부 만료(KRX 당일 주문). 페이퍼는 대기열에서 뺀다.
        for o in list(st.live_orders()):
            self._cancel(ctx, o, reason="session_end")
        if self.bundle.fills_from_holdings:
            self._sync_fills(ctx)
        for pos in list(st.open_positions()):
            q = ctx.quotes.get(pos.code)
            if q is not None:
                self._start_day(ctx, pos, q.open or q.last)
            if pos.code in st.positions and pos.exit_intent:
                pos.exit_intent["session"] = "open_auction"   # 못 판 의도는 내일 시가에
        st.jobs_done["eod"] = ctx.day
        equity, cash, mv = self._equity(ctx, ctx.last_px)
        row = {"date": ctx.day, "equity": round(equity, 2), "cash": round(cash, 2), "mv": round(mv, 2),
               "provisional": True}
        report = self._report(ctx, row)
        self.store.event("eod", **row, positions=len(st.open_positions()))
        log.info("eod 보고(잠정)\n%s", report)
        self.alert("info", f"장 마감 {ctx.day} (잠정 평가)", report)

    def _decide_on_db_close(self, ctx: _Ctx, batch: SignalBatch) -> dict[str, dict[str, float]]:
        """signal 작업: 보유 포지션을 **마지막 평가일 이후 모든 거래일**의 KRX 종가(DB 보정본)로 차례로 평가한다(검토 M5).

        연구 규칙 "종가 < MA50 → 다음 거래일 시가 매도" 그대로다. signal 을 하루 놓쳐도(크론 미실행·daily_finalize 지연) 그날
        종가 결정이 빠지지 않는다 — 배치의 ``trail_dates`` 로 어느 종가가 어느 날인지 안다. 날짜가 없으면 d 하나만 본다.
        돌려주는 값은 날짜 → {code: 종가}(자산 행 따라잡기용).
        """
        st = ctx.state
        by_date: dict[str, dict[str, float]] = {}
        d = batch.data_date.isoformat()
        for pos in list(st.open_positions()):
            n = self._cfg_of(pos).trail_ma
            if batch.patched_codes is not None and pos.code not in batch.patched_codes:
                # 통합 검토 M4: 보유 종목의 d 행이 NXT 미보정(분봉 없음)이면 그 종가로 결정하지 않는다 — 보류 + 경보.
                self.alert("warn", "보유 종목 종가 미보정 — 결정 보류", f"{pos.code}: {d} KRX 보정 없음(분봉 결손). 직전 종가로만 평가")
                if pos.trail_closes:
                    by_date.setdefault(d, {})[pos.code] = pos.trail_closes[-1]
                continue
            hist = [float(x) for x in batch.trail_closes.get(pos.code) or []]
            dates = list(batch.trail_dates.get(pos.code) or [])
            if not hist:
                if pos.trail_closes:
                    self.alert("warn", "보유 종목 DB 종가 없음", f"{pos.code}: {d} 보정 종가 없음 — 직전 값으로 평가, 추적 판단 보류")
                    by_date.setdefault(d, {})[pos.code] = pos.trail_closes[-1]
                else:
                    self.alert("warn", "보유 종목 종가 없음", f"{pos.code}: 추적 판단·평가 불가")
                continue
            pos.trail_closes = hist[-n:]
            if dates and len(dates) == len(hist):
                since = pos.last_eod or pos.entry_date or d
                idxs = [i for i, dt in enumerate(dates)
                        if len(dt) == 10 and (dt > since if pos.last_eod else dt >= since) and dt <= d]
                if not idxs and pos.last_eod is None and dates[-1] >= (pos.entry_date or ""):
                    idxs = [len(hist) - 1]
                if pos.last_eod and len(idxs) > 1:
                    self.alert("warn", "종가 평가 따라잡기", f"{pos.code}: {pos.last_eod} 이후 {len(idxs)}거래일을 한 번에 평가")
            else:
                idxs = [] if pos.last_eod == d else [len(hist) - 1]
                dates = [d if i == len(hist) - 1 else "" for i in range(len(hist))]
            for i in idxs:
                close = hist[i]
                by_date.setdefault(dates[i], {})[pos.code] = close
                if pos.code not in st.positions or pos.status != "open":
                    continue
                ma = sum(hist[i - n + 1:i + 1]) / n if i + 1 >= n else math.nan
                if not math.isfinite(ma):
                    self.alert("warn", "MA50 계산 불가", f"{pos.code}: {dates[i]} KRX 종가 {i + 1}개 < {n}")
                s = self._strat(pos)
                acts = s.on_close(close, ma, s.day_no)
                # 종가 기준 결정은 전부 "다음(=오늘) 시가 매도" 다 — 돌파 실패·추적은 원래 그렇고, 최대 보유 상한도 같은 경로.
                acts = [Action(**{**asdict(a), "at": "next_open"}) if a.at == "close" else a for a in acts]
                self._apply_actions(ctx, pos, s, acts, session="open_auction")
                pos.last_eod = dates[i]
            if pos.code in st.positions:
                by_date.setdefault(d, {}).setdefault(pos.code, hist[-1])
        return by_date

    # ----- 뉴스 거부권 섀도(설계 §7) ----------------------------------------------------
    #
    # 거부된 후보는 **주문 없이** 같은 전략 상태기계로 추적한다. 가상 진입 = 09:02 시가(실포지션과 같은 규칙: 시가 ≤ 지정가).
    # 청산 판단 = 15:40 eod 의 ka10001 일중 시가·고가·저가·현재가로 ``PositionState.on_bar``(손절·절반 익절·본전·돌파 실패·MA50
    # 추적·최대 보유 — 백테스트와 같은 비관적 일봉 순서), MA50 은 DB 보정 종가(trail_closes) + 그날 현재가. "다음날 시가" 결정은
    # 다음 09:02 시가에 가상 체결. 하루 종목당 시세 1건이라 싸다. 가격은 ka10001 일중 필드(미검증·KRX 종가 아님)라 **근사**다 —
    # 거부군 vs 진입군 비교(NEWS_VETO 합격 바)에 쓰는 용도이고, 실포지션 판단에는 쓰지 않는다. 비용은 실장부와 같은 모델
    # (섀도 ``PositionBook``). 실포지션과 코드가 겹치면 섀도를 만들지 않는다.

    def _shadow_plan(self, ctx: _Ctx, batch: SignalBatch, equity: float, n_keep: int) -> int:
        st = ctx.state
        n = 0
        for c in batch.vetoed:
            if c.code in st.shadow or c.code in st.positions:
                continue
            plan = plan_entry(c, equity, c.close, self.cfg.strategy)
            info = dict(batch.veto_info.get(c.code) or {})
            st.shadow[c.code] = PositionRec(
                code=c.code, market=c.market, signal_date=batch.data_date.isoformat(), trade_date=ctx.day,
                status="planned", name=batch.names.get(c.code, ""), plan={**_plan_to_dict(plan), "veto": info},
                trail_closes=[float(x) for x in batch.trail_closes.get(c.code, [])][-n_keep:],
            )
            self._pin_cfg(st.shadow[c.code])
            st.market_of[c.code] = c.market
            self.store.shadow_event("shadow_planned", code=c.code, signal_date=batch.data_date.isoformat(),
                                    trade_date=ctx.day, limit_px=plan.limit_px, qty=plan.qty, veto=info)
            n += 1
        for p in st.shadow.values():            # 보유 중 섀도의 MA50 종가 갱신
            closes = batch.trail_closes.get(p.code)
            if closes:
                p.trail_closes = [float(x) for x in closes][-n_keep:]
        return n

    def _shadow_fill(self, ctx: _Ctx, pos: PositionRec, side: str, qty: int, price: float, *, kind: str) -> None:
        assert ctx.shadow_book is not None
        f = Fill(ord_no=f"S{pos.code}{ctx.day}{side[0]}{len(ctx.state.shadow_history)}", code=pos.code, side=side,
                 qty=int(qty), price=int(round(price)), ts=self._now())
        realized = ctx.shadow_book.apply(f)
        self.store.shadow_event("shadow_fill", code=pos.code, side=side, qty=int(qty), price=int(round(price)),
                                kind=kind, realized=realized, veto=pos.plan.get("veto"))
        if side == "buy":
            pos.qty += int(qty)
            pos.entered_qty += int(qty)
        else:
            pos.qty -= int(qty)
            pos.realized += float(realized)

    def _shadow_close(self, ctx: _Ctx, pos: PositionRec, reason: str) -> None:
        st = ctx.state
        st.shadow.pop(pos.code, None)
        if pos.status == "open" or pos.entered_qty > 0:
            pos.status = "closed"
        pos.closed_date = ctx.day
        pos.exit_reason = pos.exit_reason or reason
        pos.exit_intent = None
        st.shadow_history.append(pos)
        self.store.shadow_event("shadow_close", code=pos.code, reason=pos.exit_reason, realized=pos.realized,
                                entered_qty=pos.entered_qty, entry_price=pos.entry_price, veto=pos.plan.get("veto"))

    def _shadow_open(self, ctx: _Ctx) -> None:
        """09:02: 오늘 계획된 섀도의 가상 진입(시가 ≤ 지정가 → 시가) + 전일 '다음날 시가' 결정의 가상 매도. 세션 확정 뒤에만."""
        st = ctx.state
        if st.sessions.get(ctx.day) != "open":
            return
        for pos in list(st.shadow.values()):
            if pos.status == "planned" and pos.trade_date < ctx.day:
                pos.status = "expired"
                self._shadow_close(ctx, pos, "stale_plan")
                continue
            if pos.status not in ("planned", "open"):
                continue
            q = ctx.quotes.get(pos.code) or self._quote(ctx, pos.code)
            if q is None or not q.open:
                continue
            open_px = float(q.open)
            if pos.status == "planned":
                plan = _plan_from_dict(pos.plan)
                if plan.fills_at(open_px) and int(plan.qty) >= 1:
                    pos.status, pos.entry_date, pos.entry_price, pos.last_day = "open", ctx.day, open_px, ctx.day
                    s = PositionState.from_plan(plan, self._cfg_of(pos), qty=float(int(plan.qty)))
                    acts = s.on_fill_open(open_px, day_no=1, entry_day=ctx.asof)
                    self._shadow_fill(ctx, pos, "buy", int(plan.qty), open_px, kind="entry")
                    self._store_strat(pos, s)
                    for a in acts:                      # below_low: 체결 직후 청산(실포지션과 같은 규칙)
                        if a.price is not None:
                            self._shadow_fill(ctx, pos, "sell", pos.qty, a.price, kind=a.kind)
                            pos.exit_reason = a.kind
                    if pos.qty <= 0:
                        self._shadow_close(ctx, pos, pos.exit_reason or "exit")
                else:
                    pos.status = "missed"
                    pos.exit_reason = f"open {open_px} > limit {plan.limit_px}"
                    self._shadow_close(ctx, pos, pos.exit_reason)
                continue
            if pos.last_day == ctx.day:
                continue
            s = self._strat(pos)
            acts = s.on_open(open_px, s.day_no + 1)
            pos.last_day = ctx.day
            self._shadow_apply(ctx, pos, s, acts, open_px)

    def _shadow_apply(self, ctx: _Ctx, pos: PositionRec, s: PositionState, acts: list[Action], px_next_open: float | None) -> None:
        for a in acts:
            if a.price is None:                         # 다음날 시가 결정 → 의도로 남긴다
                pos.exit_intent = {"kind": a.kind, "qty": pos.qty, "session": "open_auction", "created": ctx.day}
                continue
            q = min(int(math.floor(a.qty + 1e-9)), pos.qty) if a.kind == "partial" else pos.qty
            if q >= 1:
                self._shadow_fill(ctx, pos, "sell", q, a.price, kind=a.kind)
            if a.kind != "partial":
                pos.exit_reason = a.kind
                pos.exit_intent = None
        self._store_strat(pos, s)
        if pos.qty <= 0 or not s.is_open:
            self._shadow_close(ctx, pos, pos.exit_reason or "exit")

    def _shadow_settle(self, ctx: _Ctx, batch: SignalBatch) -> None:
        """signal 작업: 섀도 포지션을 **실포지션과 같은 다음날 DB 보정 일봉**으로 전이한다(통합 검토 M5 — 측정 대칭).

        마지막 평가일 이후 거래일마다 ``on_bar(시가·고가·저가·종가, MA, day_no)`` (백테스트와 같은 비관적 일봉 순서: 손절·절반
        익절·본전·돌파 실패·MA 추적·최대 보유). "다음날 시가" 결정은 다음 봉의 시가에 가상 체결(봉이 아직 없으면 09:02 시세).
        09:02 시세가 없어 ``planned`` 로 남은 섀도는 그 거래일의 DB 시가로 가상 진입한다(m4). 미보정 종목은 그날 보류(M4).
        """
        st = ctx.state
        d = batch.data_date.isoformat()
        for pos in list(st.shadow.values()):
            if pos.status not in ("open", "planned"):
                continue
            bars = batch.bars.get(pos.code) or []
            dates = batch.trail_dates.get(pos.code) or []
            if not bars or len(bars) != len(dates):
                continue
            if batch.patched_codes is not None and pos.code not in batch.patched_codes:
                continue
            closes = [b[3] for b in bars]
            if closes:
                pos.trail_closes = closes[-self._cfg_of(pos).trail_ma:]
            n = self._cfg_of(pos).trail_ma
            if pos.status == "planned":
                idx = [i for i, dt in enumerate(dates) if dt == pos.trade_date]
                if not idx:
                    continue
                i = idx[0]
                plan = _plan_from_dict(pos.plan)
                open_px = bars[i][0]
                if plan.fills_at(open_px) and int(plan.qty) >= 1:
                    pos.status, pos.entry_date, pos.entry_price, pos.last_day = "open", dates[i], open_px, dates[i]
                    s = PositionState.from_plan(plan, self._cfg_of(pos), qty=float(int(plan.qty)))
                    acts = s.on_fill_open(open_px, day_no=1, entry_day=date.fromisoformat(dates[i]))
                    self._shadow_fill(ctx, pos, "buy", int(plan.qty), open_px, kind="entry_db_open")
                    self._store_strat(pos, s)
                    self._shadow_apply(ctx, pos, s, acts, None)
                    if pos.code not in st.shadow:
                        continue
                    o, h, lo_, c = bars[i]
                    ma = sum(closes[i - n + 1:i + 1]) / n if i + 1 >= n else math.nan
                    acts = s.on_bar(o, h, lo_, c, ma, 1)
                    pos.last_eod = dates[i]
                    self._shadow_apply(ctx, pos, s, acts, None)
                else:
                    pos.status = "missed"
                    pos.exit_reason = f"open {open_px} > limit {plan.limit_px}"
                    self._shadow_close(ctx, pos, pos.exit_reason)
                continue
            since = pos.last_eod or pos.entry_date or d
            idxs = [i for i, dt in enumerate(dates)
                    if len(dt) == 10 and (dt > since if pos.last_eod else dt >= since) and dt <= d]
            for i in idxs:
                if pos.code not in st.shadow or pos.status != "open":
                    break
                s = self._strat(pos)
                o, h, lo_, c = bars[i]
                if pos.exit_intent and pos.exit_intent.get("session") == "open_auction":
                    # 전일 "다음날 시가" 결정 → 이 봉의 시가에 가상 체결
                    acts = s.on_open(o, s.day_no + 1)
                    pos.last_day = dates[i]
                    self._shadow_apply(ctx, pos, s, acts, None)
                    pos.last_eod = dates[i]
                    continue
                day_no = s.day_no + (0 if pos.last_day == dates[i] else 1)
                pos.last_day = dates[i]
                ma = sum(closes[i - n + 1:i + 1]) / n if i + 1 >= n else math.nan
                acts = s.on_bar(o, h, lo_, c, ma, day_no)
                pos.last_eod = dates[i]
                self._shadow_apply(ctx, pos, s, acts, None)

    def _report(self, ctx: _Ctx, row: dict[str, Any]) -> str:
        head = f"자산 {row['equity']:,.0f} (현금 {row['cash']:,.0f} · 평가 {row['mv']:,.0f})"
        if "peak" in row:
            head += f" 고점 {row['peak']:,.0f} 낙폭 {row['dd']:.2%}"
        lines = [head + (" [잠정 — 확정은 다음날 signal]" if row.get("provisional") else "")]
        for p in ctx.state.open_positions():
            it = p.exit_intent
            lines.append(f"  {p.code} {p.qty}주 진입 {p.entry_price:,.0f} 손절 {p.stop or 0:,.0f} "
                         f"목표 {p.target or 0:,.0f}{' 익절후' if p.partial_done else ''}"
                         f"{' → 매도예정 ' + it['kind'] if it else ''}")
        today_closed = [p for p in ctx.state.history if p.closed_date == ctx.day and p.entered_qty > 0]
        for p in today_closed:
            lines.append(f"  청산 {p.code} {p.exit_reason} 실현 {p.realized:,.0f}")
        return "\n".join(lines)

    # ----- status / reconcile ---------------------------------------------------------

    def _status(self, asof: date) -> int:
        try:
            state = self.store.load(mode=self.cfg.mode, capital=self.cfg.capital or 10_000_000.0)
        except StateCorrupt as e:
            print(f"상태 손상: {e}")
            return 3
        hist = state.equity_history
        print(f"[{self.cfg.mode}] {self.store.root}  자본 {state.capital:,.0f}")
        if hist:
            r = hist[-1]
            print(f"  마지막 장 마감 {r['date']}: 자산 {r['equity']:,.0f} 고점 {r['peak']:,.0f} 낙폭 {r['dd']:.2%}")
        stops = [p for p in (self.cfg.global_stop, self.cfg.mode_stop) if p.exists()]
        print(f"  정지 파일: {', '.join(map(str, stops)) or '없음'} · 낙폭 킬: {bool(state.drawdown_kill.get('active'))}")
        print(f"  작업: {state.jobs_done} · 신호: {state.last_signal.get('asof')} 후보 "
              f"{state.last_signal.get('n_candidates')}")
        old = [p.code for p in state.open_positions() if p.cfg_hash and p.cfg_hash != self.cfg_hash]
        meta = self.cfg.strategy_meta
        print(f"  설정 {self.cfg_hash[:8]}{' (' + str(meta.get('path')) + ', valid_from ' + str(meta.get('valid_from')) + ')' if meta else ' (DEFAULT 코드 기본값)'}"
              f" · 열린 포지션 중 옛 설정 {len(old)}개{(': ' + ', '.join(old)) if old else ''}")
        for p in state.positions.values():
            print(f"  {p.code} {p.status:13s} qty={p.qty} entry={p.entry_price} stop={p.stop} target={p.target} "
                  f"partial={p.partial_done} intent={p.exit_intent}")
        for o in state.live_orders():
            print(f"  주문 {o.oid} {o.side} {o.code} {o.qty}@{o.price} {o.purpose}/{o.session} {o.status} {o.ord_no}")
        print("  " + shadow_vs_real_line(state))
        for p in state.shadow.values():
            print(f"  섀도 {p.code} {p.status:13s} qty={p.qty} entry={p.entry_price} veto={p.plan.get('veto')} "
                  f"intent={p.exit_intent}")
        return 0

    def _job_status(self, ctx: _Ctx) -> None:  # pragma: no cover — run() 이 잠금 없이 _status 로 보낸다
        self._status(ctx.asof)

    def _job_release_order(self, ctx: _Ctx) -> None:
        """사람의 수동 해제: ``sw-live release-order <oid>``. 주문 예약·종목 차단을 푼다(브로커엔 아무것도 보내지 않는다)."""
        oid = getattr(self, "_oid", None)
        o = ctx.state.orders.get(oid or "")
        if o is None:
            raise ValueError(f"주문 {oid!r} 이 원장에 없다")
        before = o.status
        if o.status in LIVE_ORDER_STATUSES or o.status == "needs_human":
            o.status = "released"
            o.reason = (o.reason + " | released_by_human").strip(" |")
        ctx.state.blocked_codes = {c: r for c, r in ctx.state.blocked_codes.items() if not r.startswith(f"{oid}:")}
        self.store.event("order_released", oid=oid, ord_no=o.ord_no, code=o.code, by="human", before=before)
        print(f"{oid} {o.side} {o.code} {o.qty}@{o.price}: {before} → {o.status}; 차단 해제 {o.code not in ctx.state.blocked_codes}")

    def _job_reconcile(self, ctx: _Ctx) -> None:
        """장부(전략 몫) + 기준선(사용자 몫) vs 계좌. **고치지 않는다.**"""
        account = {c: h.qty for c, h in self.bundle.broker.holdings().items() if h.qty > 0}
        own = strategy_holdings(ctx.book)
        user = ctx.state.baseline_user()
        rows = []
        for code in sorted(set(account) | set(own) | set(user)):
            exp = own.get(code, 0) + user.get(code, 0)
            if exp != account.get(code, 0):
                rows.append(f"{code}: 계좌 {account.get(code, 0)} ≠ 전략 {own.get(code, 0)} + 사용자 {user.get(code, 0)}")
        ledger = {p.code: p.qty for p in ctx.state.open_positions()}
        for code in sorted(set(ledger) | set(own)):
            if ledger.get(code, 0) != own.get(code, 0):
                rows.append(f"{code}: 원장 {ledger.get(code, 0)} ≠ 장부 {own.get(code, 0)}")
        self.store.event("reconcile", ok=not rows, diffs=rows, baseline_date=ctx.state.baseline.get("date"))
        if rows:
            print("대사 차이(고치지 않음):\n  " + "\n  ".join(rows))
            self.alert("warn", "대사 차이", "\n".join(rows))
        else:
            print("대사 일치")
