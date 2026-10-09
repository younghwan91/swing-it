"""스윙 라이브 런타임 리허설 — 가짜 시세·가짜 브로커로 하루/여러 날을 크론처럼(작업마다 새 런타임 = 새 프로세스) 돌린다.

검증하는 것: signal → preopen → open_check → monitor → close → eod 한 바퀴, 손절, 절반 익절 + 본전, 다음날 시가 추적 매도,
시간 손절(종가 동시호가), 장중 재시작 복원(걸린 매도 재가격), 정지 파일(진입만 막고 청산은 계속), 다일 낙폭 킬,
기준선 보호(사용자 주식은 사지도 팔지도 않는다 — 잔고 차이 체결 경로), 휴장, 시간창, 신호 중단, CLI 모드 플래그.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from krx_quant_core.execution.events import Fill, Holding, OpenOrder
from krx_quant_core.execution.orders import OrderIntent, OrderResult
from krx_quant_core.market.session import KST

from swing_it.live import cli
from swing_it.live.brokers import REF_CODE, BrokerBundle, DailyBar, FakeQuoteProvider, build_bundle
from swing_it.live.config import DEFAULT_STRATEGY
from swing_it.live.runtime import RuntimeConfig, SignalBatch, SwingRuntime
from swing_it.live.state import StateStore
from swing_it.live.strategy import Candidate

SIGNAL_D = date(2026, 10, 9)                      # 금
D1, D2, D3, D4 = date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14), date(2026, 10, 15)
SESSIONS = [date(2026, 10, 8), SIGNAL_D, D1, D2, D3, D4, date(2026, 10, 16)]


# --------------------------------------------------------------------------- 하네스


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 12, 7, 35, tzinfo=KST)

    def __call__(self) -> datetime:
        return self.now

    def set(self, day: date, hhmm: str, sec: int = 0) -> None:
        self.now = datetime(day.year, day.month, day.day, int(hhmm[:2]), int(hhmm[3:]), sec, tzinfo=KST)

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def cand(code: str, *, close=10_000.0, pivot=9_800.0, low=9_300.0, rel=0.5, rs=90.0, d=SIGNAL_D,
         market="KOSPI") -> Candidate:
    return Candidate(code=code, signal_day=pd.Timestamp(d), market=market, close=close, pivot=pivot,
                     contraction_low=low, rel=rel, rs=rs, adv20=10_000.0, pead_rank=0.9, flow_f=1.0)


class FakeSignals:
    """거래일 → SignalBatch. 없으면 직전 거래일·후보 없음(보유 종목 MA 종가는 trail 로)."""

    def __init__(self) -> None:
        self.batches: dict[date, SignalBatch] = {}
        self.trail: dict[str, list[float]] = {}
        self.calls: list[date] = []

    def load(self, asof, *, held, since=None):
        self.calls.append(asof)
        self.since = since
        if asof in self.batches:
            b = self.batches[asof]
            for c in held:                      # 보유 종목 MA50 종가는 늘 같이 온다(실제 원천과 같게)
                if c not in b.trail_closes and c in self.trail:
                    b.trail_closes[c] = list(self.trail[c])
        else:
            prev = max(d for d in SESSIONS if d < asof)
            b = SignalBatch(data_date=prev, candidates=[],
                            trail_closes={c: list(self.trail[c]) for c in held if c in self.trail})
        for c, closes in b.trail_closes.items():   # 날짜: data_date 로 끝나는 연속 거래일(SESSIONS 와 같은 평일)
            if c not in b.trail_dates:
                b.trail_dates[c] = [str(x.date()) for x in pd.bdate_range(end=b.data_date, periods=len(closes))]
        return b


@dataclass
class Env:
    tmp: Path
    clock: Clock = field(default_factory=Clock)
    signals: FakeSignals = field(default_factory=FakeSignals)
    alerts: list = field(default_factory=list)
    mode: str = "paper"
    broker: object = None
    strategy: object = DEFAULT_STRATEGY
    capital: float = 10_000_000.0

    def __post_init__(self) -> None:
        self.quotes = FakeQuoteProvider(clock=self.clock)
        self.quotes.set_bars(REF_CODE, [DailyBar(d, 1, 1, 1, 1) for d in SESSIONS])
        hf = self.tmp / "holidays.txt"                      # 올해 휴장일 파일(없으면 런타임이 경보한다 — 검토 F4)
        if not hf.exists():
            hf.write_text("# KRX 2026\n2026-12-25\n", encoding="utf-8")

    @property
    def cfg(self) -> RuntimeConfig:
        return RuntimeConfig(data_dir=self.tmp / "live_swing", mode=self.mode, capital=self.capital,
                             strategy=self.strategy, stop_file=self.tmp / "swing_STOP",
                             holidays_file=self.tmp / "holidays.txt", lock_wait_sec=1.0)

    def bundle(self) -> BrokerBundle:
        if self.mode == "paper":
            return build_bundle("paper", confirm_real=False,
                                paper_path=self.cfg.mode_dir / "paper_broker.json", quotes=self.quotes)
        return BrokerBundle(mode=self.mode, broker=self.broker, quotes=self.quotes, fills_from_holdings=True)

    def run(self, job: str, day: date, hhmm: str, **kw) -> int:
        """크론 한 번 = 새 런타임(새 프로세스와 같다 — 메모리 상태 없음)."""
        self.clock.set(day, hhmm)
        rt = SwingRuntime(self.cfg, self.bundle, clock=self.clock, signal_source=self.signals,
                          sleeper=self.clock.sleep, alert=lambda lv, t, b="": self.alerts.append((lv, t, b)))
        return rt.run(job, day, **kw)

    def state(self):
        return StateStore(self.cfg.mode_dir, clock=self.clock).load(mode=self.mode, capital=self.capital)

    def events(self, kind: str | None = None) -> list[dict]:
        ev = StateStore(self.cfg.mode_dir, clock=self.clock).events()
        return [e for e in ev if kind is None or e["event"] == kind]

    def book(self):
        st = self.state()
        return StateStore(self.cfg.mode_dir, clock=self.clock).load_book(st)

    def q(self, code: str, last: float, **kw) -> None:
        self.quotes.set_quote(code, last, **kw)


def _closed(env: Env, code: str):
    return [p for p in env.state().history if p.code == code]


# --------------------------------------------------------------------------- 하루 리허설


def test_full_day_rehearsal_paper(tmp_path):
    env = Env(tmp_path)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("111111")],
                                          trail_closes={"111111": [9_000.0] * 49})
    assert env.run("signal", D1, "07:35") == 0
    st = env.state()
    p = st.positions["111111"]
    assert p.status == "planned" and p.plan["limit_px"] == 10_290 and p.plan["qty"] == 100
    assert env.run("signal", D1, "07:36") == 0 and len(env.signals.calls) == 1   # 멱등

    env.q("111111", 10_000, prev_close=10_000)
    assert env.run("preopen", D1, "08:35") == 0
    st = env.state()
    assert st.positions["111111"].status == "entry_pending"
    [o] = st.live_orders()
    assert (o.side, o.qty, o.price, o.session) == ("buy", 100, 10_290, "open_auction")
    assert st.baseline["date"] == D1.isoformat()
    assert env.run("preopen", D1, "08:40") == 0 and len(env.state().orders) == 1      # 재실행해도 중복 주문 없음

    env.q("111111", 10_150, open=10_100, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    st = env.state()
    p = st.positions["111111"]
    assert p.status == "open" and p.qty == 100 and p.entry_price == 10_100
    assert p.stop == pytest.approx(10_100 * (1 - (1 - 9_300 * 0.99 / 10_100)))
    assert st.sessions[D1.isoformat()] == "open"
    assert env.book().position("111111").qty == 100

    env.q("111111", 10_500, open=10_100, prev_close=10_000)
    assert env.run("monitor", D1, "10:30", once=True) == 0
    assert env.state().positions["111111"].exit_intent is None
    assert env.run("close", D1, "15:21") == 0
    env.q("111111", 10_400, open=10_100, prev_close=10_000)
    assert env.run("eod", D1, "15:40") == 0
    st = env.state()
    p = st.positions["111111"]
    assert p.exit_intent is None and p.last_eod is None                       # eod 는 결정·자산 확정을 안 한다
    assert [r["date"] for r in st.equity_history] == [SIGNAL_D.isoformat()]      # D1 행은 아직 없다
    assert st.jobs_done["eod"] == D1.isoformat()
    [eod] = env.events("eod")
    assert eod["provisional"] and eod["mv"] == 100 * 10_400
    # 다음날 07:35 signal: DB 보정(KRX) 종가로 MA50 추적·D1 장 마감 자산 확정
    env.signals.trail["111111"] = [9_000.0] * 49 + [10_350.0]
    assert env.run("signal", D2, "07:35") == 0
    st = env.state()
    p = st.positions["111111"]
    assert p.exit_intent is None and p.last_eod == D1.isoformat() and len(p.trail_closes) == 50
    assert p.trail_closes[-1] == 10_350
    row = st.equity_history[-1]
    assert row["date"] == D1.isoformat() and row["mv"] == 100 * 10_350
    assert row["equity"] == pytest.approx(10_000_000 + 100 * 250)
    ev = [e["event"] for e in env.events()]
    for k in ("signal", "baseline", "order", "fill", "position_open", "eod", "equity"):
        assert k in ev
    # 로그 파일이 모드 디렉터리 아래에 있다
    assert (env.cfg.mode_dir / "logs" / f"{D1.isoformat()}.log").exists()


# --------------------------------------------------------------------------- 여러 날


def _enter(env: Env, codes: dict[str, dict], day=D1) -> None:
    """D1 에 codes 진입(시가 10,100 체결)."""
    env.signals.batches[day] = SignalBatch(
        data_date=max(d for d in SESSIONS if d < day),
        candidates=[cand(c, **kw.get("cand", {})) for c, kw in codes.items()],
        trail_closes={c: kw.get("trail", [9_000.0] * 49) for c, kw in codes.items()})
    for c, kw in codes.items():
        env.signals.trail[c] = kw.get("trail", [9_000.0] * 49)
        env.q(c, 10_000, prev_close=10_000)
    assert env.run("signal", day, "07:35") == 0
    assert env.run("preopen", day, "08:35") == 0
    for c in codes:
        env.q(c, 10_100, open=10_100, prev_close=10_000)
    assert env.run("open_check", day, "09:02") == 0
    for c in codes:
        assert env.state().positions[c].status == "open", c


def _eod(env: Env, day: date, closes: dict[str, float]) -> None:
    """15:21 close + 15:40 eod. 그날 KRX 종가는 다음날 signal 이 DB 에서 읽는다 → 가짜 원천의 trail 에 덧붙인다."""
    for c, px in closes.items():
        env.q(c, px, open=px, prev_close=px)
        env.signals.trail.setdefault(c, [9_000.0] * 49).append(float(px))
    assert env.run("close", day, "15:21") == 0
    assert env.run("eod", day, "15:40") == 0


def test_multi_day_stop_partial_breakeven_trail_time(tmp_path):
    env = Env(tmp_path, strategy=replace(DEFAULT_STRATEGY, max_hold=3))    # 최대 보유 상한을 3일로 당겨 그 경로도 본다
    up = [11_800.0] * 49
    _enter(env, {"AAAAAA": {}, "BBBBBB": {}, "CCCCCC": {"trail": up}, "DDDDDD": {}})
    _eod(env, D1, {"AAAAAA": 10_000, "BBBBBB": 10_200, "CCCCCC": 10_300, "DDDDDD": 10_000})
    stop_a = env.state().positions["AAAAAA"].stop
    target = env.state().positions["BBBBBB"].target
    assert 11_880 < target < 11_890

    # D2: 전날 결정 없음 → preopen 매도 없음. 시가 확인 후 장중.
    for c in ("AAAAAA", "BBBBBB", "CCCCCC", "DDDDDD"):
        env.q(c, 10_000, prev_close=10_000)
    env.run("signal", D2, "07:35")
    assert env.run("preopen", D2, "08:35") == 0
    assert not env.state().live_orders()
    for c, op in {"AAAAAA": 9_900, "BBBBBB": 10_000, "CCCCCC": 10_300, "DDDDDD": 10_000}.items():
        env.q(c, op, open=op, prev_close=10_000)
    assert env.run("open_check", D2, "09:02") == 0
    # A: 손절가 아래 → 손절 매도(같은 시세에서 페이퍼 즉시 체결)
    env.q("AAAAAA", 9_150, open=9_900, prev_close=10_000)
    # B·C: 목표가 위 → 절반 익절
    env.q("BBBBBB", 11_900, open=10_000, prev_close=10_000)
    env.q("CCCCCC", 11_900, open=10_300, prev_close=10_000)
    assert env.run("monitor", D2, "10:30", once=True) == 0
    st = env.state()
    assert "AAAAAA" not in st.positions
    [a] = _closed(env, "AAAAAA")
    assert a.exit_reason == "stop" and a.qty == 0 and a.realized < 0 and 9_150 < stop_a
    assert st.positions["BBBBBB"].qty == 50 and st.positions["BBBBBB"].partial_done
    assert st.positions["CCCCCC"].qty == 50
    sells = [o for o in st.orders.values() if o.side == "sell"]
    assert {o.code for o in sells} == {"AAAAAA", "BBBBBB", "CCCCCC"}
    assert all(o.price < o.ref_price for o in sells)                 # 할인 지정가
    # B: 같은 날 진입가 아래로 와도 본전은 다음 거래일부터
    env.q("BBBBBB", 10_050, open=10_000, prev_close=10_000)
    assert env.run("monitor", D2, "11:00", once=True) == 0
    assert env.state().positions["BBBBBB"].qty == 50
    # D2 장 마감: C 는 종가 < MA50(≈11,794) → 다음날 시가 매도 결정. D 는 시간 손절 전(day 2).
    _eod(env, D2, {"BBBBBB": 10_800, "CCCCCC": 11_500, "DDDDDD": 10_000})
    assert env.state().positions["CCCCCC"].exit_intent is None        # eod 는 결정하지 않는다

    # D3 07:35 signal: D2 KRX 종가(DB) < MA50 → 오늘 시가 매도 결정. 08:35 preopen 이 장전 동시호가로 낸다.
    for c, pc in {"BBBBBB": 10_800, "CCCCCC": 11_500, "DDDDDD": 10_000}.items():
        env.q(c, pc, prev_close=pc)
    assert env.run("signal", D3, "07:35") == 0
    st = env.state()
    assert st.positions["CCCCCC"].exit_intent["kind"] == "trail_next_open"
    assert st.positions["CCCCCC"].exit_intent["session"] == "open_auction"
    assert st.positions["BBBBBB"].exit_intent is None
    assert st.equity_history[-1]["date"] == D2.isoformat()
    assert env.run("preopen", D3, "08:35") == 0
    [o] = env.state().live_orders()
    assert (o.code, o.side, o.qty, o.session, o.purpose) == ("CCCCCC", "sell", 50, "open_auction", "trail_next_open")
    assert o.price == 10_350                                          # 11,500 × 0.9 호가 올림
    for c, op in {"BBBBBB": 10_700, "CCCCCC": 11_300, "DDDDDD": 10_000}.items():
        env.q(c, op, open=op, prev_close=10_000)
    assert env.run("open_check", D3, "09:02") == 0
    [c_] = _closed(env, "CCCCCC")
    assert c_.exit_reason == "trail_next_open"
    fills_c = [e for e in env.events("fill") if e["code"] == "CCCCCC" and e["side"] == "sell"]
    assert [f["price"] for f in fills_c] == [11_900, 11_300]          # 익절가, 다음날 시가
    # B: 다음 거래일 본전 → 진입가 이하에서 나머지 전량
    env.q("BBBBBB", 10_050, open=10_700, prev_close=10_800)
    assert env.run("monitor", D3, "10:30", once=True) == 0
    [b] = _closed(env, "BBBBBB")
    assert b.exit_reason == "breakeven" and b.realized > 0
    # D: 15:20 close 작업은 종가 기준 결정을 하지 않는다(시간 손절 폐지, 설계 §6) — 못 판 의도가 없으면 주문도 없다
    env.q("DDDDDD", 10_000, open=10_000, prev_close=10_000)
    assert env.run("close", D3, "15:21") == 0
    assert not env.state().live_orders()
    _eod(env, D3, {"DDDDDD": 9_980})
    assert env.state().positions["DDDDDD"].status == "open"
    # D4 signal: D3 종가로 최대 보유 상한(day 3) → 오늘 시가 매도 의도 → preopen 장전 동시호가 → open_check 시가 체결
    env.q("DDDDDD", 9_980, prev_close=9_980)
    assert env.run("signal", D4, "07:35") == 0
    assert env.state().positions["DDDDDD"].exit_intent["kind"] == "max_hold"
    assert env.run("preopen", D4, "08:35") == 0
    [o] = env.state().live_orders()
    assert (o.code, o.session, o.purpose) == ("DDDDDD", "open_auction", "max_hold")
    env.q("DDDDDD", 9_950, open=9_950, prev_close=9_980)
    assert env.run("open_check", D4, "09:02") == 0
    [d_] = _closed(env, "DDDDDD")
    assert d_.exit_reason == "max_hold"
    assert [e["price"] for e in env.events("fill") if e["code"] == "DDDDDD" and e["side"] == "sell"] == [9_950]
    st = env.state()
    assert not st.positions and not st.live_orders()
    assert env.book().positions() == {}
    # 원장 손익 = 장부 실현손익 = 다음날 signal 이 확정하는 D4 장 마감 자산
    assert sum(p.realized for p in st.history) == pytest.approx(env.book().realized_krw)
    _eod(env, D4, {})
    assert env.run("signal", date(2026, 10, 16), "07:35") == 0
    row = env.state().equity_history[-1]
    assert row["date"] == D4.isoformat() and row["equity"] == pytest.approx(env.capital + env.book().realized_krw)


def test_restart_mid_day_restores_and_reprices(tmp_path):
    env = Env(tmp_path)
    _enter(env, {"AAAAAA": {}})
    _eod(env, D1, {"AAAAAA": 10_000})
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    env.q("AAAAAA", 9_900, open=9_900, prev_close=10_000)
    env.run("open_check", D2, "09:02")
    # 손절 신호 → 터치 즉시 현재가 − 1틱(9,140) 지정가. 매수호가가 그 아래라 안 채워진다.
    env.q("AAAAAA", 9_150, open=9_900, prev_close=10_000, bid=9_050)
    assert env.run("monitor", D2, "10:30", once=True) == 0
    st = env.state()
    [o] = st.live_orders()
    assert (o.price, o.attempt, o.purpose) == (9_140, 1, "stop")
    # 프로세스가 죽었다가 새로 뜬다(새 런타임 = 메모리 없음). 61초 뒤 → 취소 후 현재가 − 0.5%(9,050).
    env.q("AAAAAA", 9_100, open=9_900, prev_close=10_000, bid=9_080)
    env.clock.set(D2, "10:31", 1)
    rt = SwingRuntime(env.cfg, env.bundle, clock=env.clock, signal_source=env.signals, sleeper=env.clock.sleep,
                      alert=lambda *a: None)
    assert rt.run("monitor", D2, once=True) == 0
    st = env.state()
    assert "AAAAAA" not in st.positions
    sells = sorted((o for o in st.orders.values() if o.side == "sell"), key=lambda o: o.oid)
    assert [(o.status, o.attempt) for o in sells] == [("canceled", 1), ("filled", 2)]
    assert sells[1].price == 9_050
    fill = [e for e in env.events("fill") if e["side"] == "sell"][-1]
    assert fill["price"] == 9_080
    assert env.book().positions() == {}


def test_monitor_loop_runs_until_cutoff_with_quiet_window(tmp_path):
    env = Env(tmp_path)
    _enter(env, {"AAAAAA": {}})
    env.q("AAAAAA", 10_200, open=10_100, prev_close=10_000)
    env.clock.set(D1, "15:16")
    n = {"ticks": 0}
    rt = SwingRuntime(env.cfg, env.bundle, clock=env.clock, signal_source=env.signals,
                      sleeper=lambda s: (n.__setitem__("ticks", n["ticks"] + 1), env.clock.sleep(s)),
                      alert=lambda *a: None)
    assert rt.run("monitor", D1) == 0
    assert n["ticks"] == 3 and env.clock().strftime("%H:%M") == "15:19"
    assert rt._interval() == 60
    env.clock.set(D1, "10:05")
    assert rt._interval() == 180


# --------------------------------------------------------------------------- 킬


def test_stop_file_blocks_entries_but_not_exits(tmp_path):
    env = Env(tmp_path)
    _enter(env, {"CCCCCC": {"trail": [11_800.0] * 49}})
    env.q("CCCCCC", 11_900, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)                 # 절반 익절
    _eod(env, D1, {"CCCCCC": 11_500})                          # 종가 < MA50 → 다음날 시가 매도
    (tmp_path / "swing_STOP").write_text("사람이 멈춤")
    env.signals.batches[D2] = SignalBatch(data_date=D1, candidates=[cand("EEEEEE", d=D1)],
                                          trail_closes={"EEEEEE": [9_000.0] * 49})
    env.q("EEEEEE", 10_000, prev_close=10_000)
    env.q("CCCCCC", 11_500, prev_close=11_500)
    env.run("signal", D2, "07:35")
    assert env.run("preopen", D2, "08:35") == 0
    st = env.state()
    [o] = st.live_orders()
    assert (o.code, o.side) == ("CCCCCC", "sell")              # 청산은 나간다
    [e] = _closed(env, "EEEEEE")
    assert e.status == "skipped" and "정지 파일" in e.exit_reason      # 진입은 막힌다
    assert any("신규 진입 정지" in a[1] for a in env.alerts)


def test_drawdown_kill_creates_stop_and_blocks_next_entries(tmp_path):
    env = Env(tmp_path)
    _enter(env, {"AAAAAA": {}})
    st = env.state()
    st.equity_peak = 12_000_000                                 # 고점 대비 −16% 를 만든다
    StateStore(env.cfg.mode_dir, clock=env.clock).save(st)
    _eod(env, D1, {"AAAAAA": 10_000})
    mode_stop = tmp_path / "swing_paper_STOP"
    assert not mode_stop.exists()                               # eod 는 결정하지 않는다
    env.signals.batches[D2] = SignalBatch(data_date=D1, candidates=[cand("EEEEEE", d=D1)])
    env.q("EEEEEE", 10_000, prev_close=10_000)
    env.q("AAAAAA", 10_000, prev_close=10_000)
    assert env.run("signal", D2, "07:35") == 0                  # D1 KRX 종가 기준 자산 9.99M / 고점 12M → 킬
    assert mode_stop.exists() and env.state().drawdown_kill["active"]
    assert not (tmp_path / "swing_STOP").exists()               # 페이퍼 낙폭은 실계좌 정지 파일을 건드리지 않는다
    assert env.events("kill")
    env.run("preopen", D2, "08:35")
    [e] = _closed(env, "EEEEEE")
    assert "낙폭" in e.exit_reason or "정지 파일" in e.exit_reason
    # 사람이 파일을 지우면 다음 signal 에서 고점 재설정
    mode_stop.unlink()
    env.q("AAAAAA", 10_000, open=10_000, prev_close=10_000)
    env.run("open_check", D2, "09:02")
    _eod(env, D2, {"AAAAAA": 10_000})
    assert env.run("signal", D3, "07:35") == 0
    st = env.state()
    assert not st.drawdown_kill["active"] and st.equity_peak < 12_000_000 and not mode_stop.exists()


def test_daily_loss_kill_blocks_next_morning(tmp_path):
    env = Env(tmp_path)
    st = env.state()
    st.capital = 9_600_000.0                                        # signal 이 확정할 D(10-09) 자산
    st.equity_history = [{"date": "2026-10-08", "equity": 10_000_000.0}]   # 전일 −4% > 일 손실 한도 3%
    StateStore(env.cfg.mode_dir, clock=env.clock).save(st)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    [a] = _closed(env, "AAAAAA")
    assert "daily_loss" in a.exit_reason


# --------------------------------------------------------------------------- 공유 계좌(잔고 차이 체결 경로)


class FakeKiwoom:
    """키움 모양의 가짜(ka10076 미검증 → poll_fills 는 늘 []). 체결은 ``match``/``partial_fill`` 이 잔고를 바꿔서 낸다.

    검토(REVIEW_RUNTIME §6)가 요구한 공격 능력:
    - ``lag``: 잔고 조회(kt00018)가 체결·사용자 매매보다 늦다 — 변화 뒤 ``lag`` 번의 ``holdings()`` 는 옛 잔고를 준다.
      ``publish()`` 는 즉시 반영.
    - ``reject_sells``: 매도 주문을 rc 8 로 거부한다.
    - ``foreign``: 남(사용자)의 미체결 주문이 ``open_orders()`` 에 섞여 온다.
    - ``partial_fill``: 주문 일부만 체결(``open_orders()`` 의 ``remaining`` 이 줄어든다).
    """

    dry_run = False

    def __init__(self, holdings: dict[str, int], *, lag: int = 0, reject_sells: bool = False,
                 foreign: list[OpenOrder] | None = None) -> None:
        self._h = {c: Holding(c, q, 10_000.0) for c, q in holdings.items()}
        self.orders: dict[str, OrderIntent] = {}
        self.remaining: dict[str, int] = {}
        self.submitted: list[OrderIntent] = []
        self.n = 0
        self.lag = lag
        self.reject_sells = reject_sells
        self.foreign = list(foreign or [])
        self._visible = dict(self._h)
        self._stale_reads = 0
        self.holdings_calls = 0

    def submit(self, intent):
        assert intent.exchange == "KRX"
        self.submitted.append(intent)
        if intent.side == "sell" and self.reject_sells:
            return OrderResult(intent=intent, dry_run=False, submitted=True, return_code=8, return_msg="매도가능수량 부족")
        if intent.side == "sell":
            assert intent.qty <= self._h.get(intent.code, Holding(intent.code, 0, 0)).qty
        self.n += 1
        ord_no = f"{self.n:07d}"
        self.orders[ord_no] = intent
        self.remaining[ord_no] = intent.qty
        return OrderResult(intent=intent, dry_run=False, submitted=True, ord_no=ord_no, return_code=0)

    def cancel(self, ord_no, intent):
        ok = self.orders.pop(ord_no, None) is not None
        self.remaining.pop(ord_no, None)
        return OrderResult(intent=intent, dry_run=False, submitted=True, return_code=0 if ok else 1)

    def open_orders(self):
        return [OpenOrder(o, i.code, i.side, i.qty, self.remaining[o], i.price) for o, i in self.orders.items()] \
            + list(self.foreign)

    def holdings(self):
        self.holdings_calls += 1
        if self._stale_reads > 0:
            self._stale_reads -= 1
            return dict(self._visible)
        self._visible = dict(self._h)
        return dict(self._visible)

    def publish(self) -> None:
        self._stale_reads = 0

    def _touch(self) -> None:
        if self.lag:
            self._stale_reads = self.lag

    def poll_fills(self) -> list[Fill]:
        return []

    def _fill(self, ord_no: str, qty: int, px: float) -> None:
        i = self.orders[ord_no]
        h = self._h.get(i.code, Holding(i.code, 0, 0.0))
        q = h.qty + (qty if i.side == "buy" else -qty)
        avg = px if i.side == "buy" and h.qty == 0 else h.avg_price
        if q > 0:
            self._h[i.code] = Holding(i.code, q, avg)
        else:
            self._h.pop(i.code, None)
        self.remaining[ord_no] -= qty
        if self.remaining[ord_no] <= 0:
            del self.orders[ord_no]
            del self.remaining[ord_no]

    def match(self, code: str, px: float) -> None:
        for o, i in list(self.orders.items()):
            if i.code == code and ((i.side == "buy" and px <= i.price) or (i.side == "sell" and px >= i.price)):
                self._fill(o, self.remaining[o], px)
        self._touch()

    def partial_fill(self, code: str, qty: int, px: float) -> None:
        for o, i in list(self.orders.items()):
            if i.code == code:
                self._fill(o, min(qty, self.remaining[o]), px)
                break
        self._touch()

    def user_trade(self, code: str, delta: int) -> None:
        h = self._h.get(code, Holding(code, 0, 10_000.0))
        self._h[code] = Holding(code, h.qty + delta, h.avg_price)
        self._touch()


def test_baseline_protection_never_buys_or_sells_user_shares(tmp_path):
    fk = FakeKiwoom({"XXXXXX": 40})                     # 사용자가 원래 들고 있던 종목
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA"), cand("XXXXXX", rel=0.1)],
                                          trail_closes={"AAAAAA": [9_000.0] * 49})
    for c in ("AAAAAA", "XXXXXX"):
        env.q(c, 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    assert env.run("preopen", D1, "08:35") == 0
    assert [i.code for i in fk.submitted] == ["AAAAAA"]          # 사용자 보유 종목은 사지 않는다
    [x] = _closed(env, "XXXXXX")
    assert x.exit_reason == "user_holds"
    assert env.state().baseline["user"] == {"XXXXXX": 40}
    # 09:00 단일가 10,100 → 체결은 잔고 차이로 판정, 가격은 단일가
    fk.match("AAAAAA", 10_100)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    p = env.state().positions["AAAAAA"]
    assert p.status == "open" and p.qty == 100 and p.entry_price == 10_100
    assert [e["price_estimated"] for e in env.events("fill")] == [True]
    # 장중에 사용자가 같은 종목 30주를 산다 → 설명 안 되는 잔고 차이: 장부를 고치지 않고 경보만
    fk.user_trade("AAAAAA", 30)
    env.q("AAAAAA", 10_200, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    assert env.book().position("AAAAAA").qty == 100
    assert env.events("mismatch") and any("잔고 차이" in a[1] for a in env.alerts)
    _eod(env, D1, {"AAAAAA": 10_000})
    # D2: 기준선 = 계좌 130 − 장부 100 → 사용자 30. 손절은 전략 몫 100 주만 판다.
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    assert env.state().baseline["user"] == {"XXXXXX": 40, "AAAAAA": 30}
    env.q("AAAAAA", 9_900, open=9_900, prev_close=10_000)
    env.run("open_check", D2, "09:02")
    env.q("AAAAAA", 9_150, open=9_900, prev_close=10_000)
    env.run("monitor", D2, "10:30", once=True)
    sell = [i for i in fk.submitted if i.side == "sell"]
    assert [(i.code, i.qty) for i in sell] == [("AAAAAA", 100)]
    fk.match("AAAAAA", 9_150)
    env.run("monitor", D2, "10:31", once=True)
    assert fk.holdings()["AAAAAA"].qty == 30 and fk.holdings()["XXXXXX"].qty == 40
    [a] = _closed(env, "AAAAAA")
    assert a.exit_reason == "stop"
    # 이후 어떤 작업도 사용자 몫을 팔지 않는다
    env.run("close", D2, "15:21")
    env.run("eod", D2, "15:40")
    assert [i for i in fk.submitted if i.side == "sell"] == sell


def test_dry_run_rehearsal_places_nothing(tmp_path):
    """dry-run: 실계좌 잔고는 읽지만 주문은 POST 하지 않는다 → 진입은 미체결로 끝난다."""
    from types import SimpleNamespace

    from krx_quant_core.execution.kiwoom_broker import KiwoomBroker

    def boom(**kw):
        raise AssertionError("POST")

    api = SimpleNamespace(order=SimpleNamespace(buy_order=boom, sell_order=boom, cancel_order=boom),
                          account=SimpleNamespace(evaluation_balance_detail=lambda **kw: {
                              "acnt_evlt_remn_indv_tot": [{"stk_cd": "A005930", "rmnd_qty": "3", "pur_pric": "1"}]}))
    env = Env(tmp_path, mode="dry-run", broker=KiwoomBroker(api, dry_run=True))
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    assert env.run("preopen", D1, "08:35") == 0
    [o] = env.state().orders.values()
    assert o.status == "dry_run" and o.ord_no is None
    env.q("AAAAAA", 10_100, open=10_100, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    [a] = _closed(env, "AAAAAA")
    assert a.status == "missed" and a.exit_reason == "dry_run"
    assert env.state().baseline["user"] == {"005930": 3}


# --------------------------------------------------------------------------- 휴장·시간창·신호 중단


def test_suspected_holiday_keeps_orders_and_eod_collects_them(tmp_path):
    """ka10081 에 오늘 봉이 없고 시세도 없다(진짜 휴장 또는 API 장애) → 주문은 유지·경보만, eod 가 거둔다(검토 M3)."""
    env = Env(tmp_path)
    env.quotes.set_bars(REF_CODE, [DailyBar(d, 1, 1, 1, 1) for d in (date(2026, 10, 8), SIGNAL_D)])  # D1 봉 없음
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    assert env.state().live_orders()
    env.quotes.fail.add("AAAAAA")
    assert env.run("open_check", D1, "09:02") == 0
    st = env.state()
    assert st.sessions[D1.isoformat()] == "suspect" and st.live_orders() and st.positions["AAAAAA"].status == "entry_pending"
    assert env.events("holiday_suspect") and any("휴장 의심" in a[1] for a in env.alerts)
    assert env.run("monitor", D1, "10:00", once=True) == 0     # 여전히 의심 — 아무 판정도 안 한다
    assert env.state().live_orders()
    assert env.run("eod", D1, "15:40") == 0                     # 장 끝: 주문 회수
    assert not env.state().live_orders()
    assert build_bundle("paper", confirm_real=False, paper_path=env.cfg.mode_dir / "paper_broker.json",
                        quotes=env.quotes).broker.open_orders() == []
    env.quotes.fail.clear()
    env.run("signal", D2, "07:35")                              # 다음날: 지난 계획 만료
    assert [p.exit_reason for p in env.state().history] == ["stale_plan"]


def test_holidays_file_and_weekend(tmp_path):
    env = Env(tmp_path)
    (tmp_path / "holidays.txt").write_text("# 휴장\n2026-10-12\n")
    env.alerts.clear()
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    assert env.run("signal", D1, "07:35") == 0 and env.signals.calls == []
    assert env.run("preopen", D1, "08:35") == 0
    assert env.state().sessions[D1.isoformat()] == "closed"
    sat = date(2026, 10, 17)
    assert env.run("preopen", sat, "08:35") == 0 and env.state().sessions[sat.isoformat()] == "closed"


def test_job_window_enforced_for_dry_run(tmp_path):
    env = Env(tmp_path, mode="dry-run", broker=FakeKiwoom({}))
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    assert env.run("signal", D1, "07:35") == 0
    assert env.run("preopen", D1, "10:00") == 2               # 장전 매수를 10시에 내면 추격 매수
    assert env.broker.submitted == []
    with pytest.raises(ValueError):
        SwingRuntime(replace(env.cfg, ignore_clock=True), env.bundle, clock=env.clock)


def test_signal_aborts_on_stale_or_unpatched_data(tmp_path):
    env = Env(tmp_path)
    env.signals.batches[D1] = SignalBatch(data_date=date(2026, 10, 8), candidates=[cand("AAAAAA")])
    env.run("signal", D1, "07:35")
    assert not env.state().positions and env.events("signal_abort")
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")], krx_patched=False)
    env.run("signal", D1, "07:36", force=True)
    assert not env.state().positions and len(env.events("signal_abort")) == 2
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA"), cand("BBBBBB")],
                                          patched_codes={"BBBBBB"})
    env.run("signal", D1, "07:37", force=True)
    assert set(env.state().positions) == {"BBBBBB"}
    assert any("보정 안 된 후보" in a[1] for a in env.alerts)


def test_corrupt_state_blocks_jobs(tmp_path):
    env = Env(tmp_path)
    env.cfg.mode_dir.mkdir(parents=True)
    (env.cfg.mode_dir / "state.json").write_text("{", encoding="utf-8")
    assert env.run("preopen", D1, "08:35") == 3
    assert any("상태 손상" in a[1] for a in env.alerts)


def test_live_first_run_requires_capital(tmp_path):
    env = Env(tmp_path, mode="live", broker=FakeKiwoom({}), capital=None)
    assert env.run("preopen", D1, "08:35") == 3


def test_status_runs_without_broker(tmp_path, capsys):
    env = Env(tmp_path)
    rt = SwingRuntime(env.cfg, lambda: (_ for _ in ()).throw(AssertionError("브로커를 만들면 안 된다")),
                      clock=env.clock)
    assert rt.run("status", D1) == 0
    assert "[paper]" in capsys.readouterr().out


# --------------------------------------------------------------------------- CLI 모드 플래그


@pytest.mark.parametrize("argv", [
    ["preopen", "--confirm-real"],
    ["preopen", "--execute"],
    ["preopen", "--execute", "--confirm-real", "--paper"],
    ["preopen", "--execute", "--confirm-real", "--dry-run"],
    ["preopen", "--paper", "--dry-run"],
    ["preopen", "--dry-run", "--ignore-clock"],
    ["preopen", "--execute", "--confirm-real", "--ignore-clock"],
])
def test_cli_refuses_unsafe_flag_combinations(argv):
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as ei:
        cli.resolve_mode(parser, parser.parse_args(argv))
    assert ei.value.code == 2


@pytest.mark.parametrize("argv,mode,real", [
    (["status"], "paper", False),
    (["status", "--paper"], "paper", False),
    (["status", "--dry-run"], "dry-run", False),
    (["status", "--execute", "--confirm-real"], "live", True),
])
def test_cli_mode_resolution(argv, mode, real):
    parser = cli.build_parser()
    assert cli.resolve_mode(parser, parser.parse_args(argv)) == (mode, real)


def test_cli_status_paper_end_to_end(tmp_path, capsys):
    rc = cli.main(["status", "--data-dir", str(tmp_path / "ls"), "--stop-file", str(tmp_path / "swing_STOP")])
    assert rc == 0
    err = capsys.readouterr().err
    assert "PAPER" in err and str(tmp_path / "ls") in err


def test_cli_job_names_map_to_runtime():
    from swing_it.live.runtime import JOBS

    assert {j.replace("-", "_") for j in cli.CLI_JOBS} == set(JOBS)


def test_features_signal_source_patch_coverage(monkeypatch):
    """기본 신호 원천: 신호일 d 의 NXT 보정 범위를 SignalBatch 에 싣는다(보정 안 된 종목 → patched_codes 밖)."""
    import swing_it.live.features as features
    import swing_it.live.strategy as strategy
    import swing_it.storage as storage
    from swing_it.live.runtime import FeaturesSignalSource

    days = pd.bdate_range("2026-08-03", "2026-10-09")
    close = pd.DataFrame({"AAAAAA": range(len(days)), "BBBBBB": 1.0, "CCCCCC": float("nan")}, index=days,
                         dtype=float)
    unp = pd.DataFrame({"code": ["BBBBBB"], "date": [pd.Timestamp("2026-10-09")]})
    panels = {"close": close, "patch_info": {"nxt_rows": 10, "nxt_patched": 9, "nxt_unpatched": unp}}

    class _Con:
        def close(self):
            pass

    monkeypatch.setattr(storage, "connect", lambda db=None: _Con())
    monkeypatch.setattr(storage, "db_default", lambda: "x")
    monkeypatch.setattr(features, "signal_panels_for", lambda con, d, cfg: (panels, days[-1]))
    monkeypatch.setattr(strategy, "scan", lambda p, d, cfg: [cand("AAAAAA"), cand("BBBBBB")])
    b = FeaturesSignalSource().load(D1, held=["CCCCCC"], since=date(2026, 9, 1))
    assert b.data_date == SIGNAL_D and b.krx_patched
    assert b.patched_codes == {"AAAAAA"}                       # B 는 d 행이 보정 안 됨, C 는 d 행이 없음
    assert len(b.trail_closes["AAAAAA"]) == 50 and b.trail_closes["AAAAAA"][-1] == len(days) - 1
    assert b.trail_closes["CCCCCC"] == []
    # 보정이 전혀 안 돌았으면 krx_patched=False → signal 작업이 중단한다
    panels["patch_info"] = {"nxt_rows": 10, "nxt_patched": 0, "nxt_unpatched": unp}
    assert not FeaturesSignalSource().load(D1, held=[]).krx_patched
    # DB 에 오늘 봉이 있으면 거부
    monkeypatch.setattr(features, "signal_panels_for", lambda con, d, cfg: (panels, pd.Timestamp(D1)))
    with pytest.raises(RuntimeError):
        FeaturesSignalSource().load(D1, held=[])


def test_stale_plan_expires_and_does_not_block_reentry(tmp_path):
    env = Env(tmp_path)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.run("signal", D1, "07:35")                         # D1 preopen 이 안 돌았다
    env.signals.batches[D2] = SignalBatch(data_date=D1, candidates=[cand("AAAAAA", d=D1)])
    env.run("signal", D2, "07:35")
    st = env.state()
    assert st.positions["AAAAAA"].trade_date == D2.isoformat()
    assert [p.exit_reason for p in st.history] == ["stale_plan"]


def test_preopen_baseline_absorbs_overnight_user_trades(tmp_path):
    """전날 장중 사용자 매매는 다음 날 기준선이 흡수한다 — 아침마다 가짜 잔고 차이 경보를 내지 않는다."""
    fk = FakeKiwoom({"XXXXXX": 40})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[])
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    fk.user_trade("XXXXXX", 10)                           # D1 장중 사용자 매수
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    assert env.state().baseline["user"] == {"XXXXXX": 50}
    assert not env.events("mismatch")


def test_open_below_contraction_low_exits_same_day(tmp_path):
    """시가 ≤ 수축 저점 체결(연구는 skip_below_low) → 체결 직후 그날 전량 청산."""
    env = Env(tmp_path)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    env.q("AAAAAA", 9_250, open=9_200, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    [a] = _closed(env, "AAAAAA")
    assert a.entered_qty == 100 and a.exit_reason == "below_low" and a.qty == 0
    assert env.book().positions() == {}


# --------------------------------------------------------------------------- 감사 L4 — 잔고 차이 체결 추정의 귀속 규칙


def test_user_manual_buy_after_cancelled_entry_is_not_booked(tmp_path):
    """L4 시나리오: 전략 진입 지정가가 09:02 에 미체결 취소된 종목을 사용자가 같은 날 장중에 산다 → 전략 몫이 아니다."""
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    [buy] = fk.submitted
    assert (buy.side, buy.qty, buy.price) == ("buy", 100, 10_290)
    env.q("AAAAAA", 10_350, open=10_300, prev_close=10_000)           # 시가 > 지정가 → 미체결
    assert env.run("open_check", D1, "09:02") == 0
    [a] = _closed(env, "AAAAAA")
    assert a.status == "missed" and fk.orders == {}
    fk.user_trade("AAAAAA", 50)                                       # 사용자 수동 매수(같은 날·같은 종목)
    env.q("AAAAAA", 10_400, open=10_300, prev_close=10_000)
    assert env.run("monitor", D1, "10:30", once=True) == 0
    st = env.state()
    assert "AAAAAA" not in st.positions and env.book().positions() == {}
    assert st.baseline["user"] == {"AAAAAA": 50}                       # 사용자 몫으로 기준선에 더했다
    [m] = env.events("mismatch")
    assert (m["side"], m["unexplained"], m["user"]) == ("buy", 50, 50)
    assert not [e for e in env.events("fill")]
    env.run("close", D1, "15:21")
    env.run("eod", D1, "15:40")
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    assert [i for i in fk.submitted if i.side == "sell"] == []       # 어느 작업도 그 주식을 팔지 않는다
    assert env.state().baseline["user"] == {"AAAAAA": 50} and fk.holdings()["AAAAAA"].qty == 50


def test_user_buy_while_strategy_order_live_but_auction_price_excludes_it(tmp_path):
    """장전 주문은 09:00 단일가로만 체결된다 — 단일가 > 지정가면 그 주문의 체결일 수 없으니 잔고 증가는 사용자 몫이다."""
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    fk.user_trade("AAAAAA", 70)                                       # 09:00~09:02 사이 사용자 매수(주문은 아직 살아 있다)
    env.q("AAAAAA", 10_350, open=10_300, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    assert env.book().positions() == {} and env.state().baseline["user"] == {"AAAAAA": 70}
    [a] = _closed(env, "AAAAAA")
    assert a.status == "missed"


def test_cancel_rejected_because_filled_still_books_our_fill(tmp_path):
    """매도 취소가 '이미 체결' 로 거부되면 취소 표시 **전에** 잔고로 체결을 확인한다 — 아니면 전략이 판 주식이 사용자 매도로 분류된다."""
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    fk.match("AAAAAA", 10_100)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    assert env.state().positions["AAAAAA"].qty == 100
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)                        # 손절 매도 9,140 걸림
    [o] = env.state().live_orders()
    assert o.price == 9_140
    fk.match("AAAAAA", 9_140)                                         # 브로커에서 체결(우리는 아직 모른다)
    env.clock.set(D1, "10:31", 1)
    env.q("AAAAAA", 9_100, open=10_100, prev_close=10_000)
    assert env.run("monitor", D1, "10:31", once=True) == 0            # 재가격 시도 → 취소 거부(이미 체결)
    [a] = _closed(env, "AAAAAA")
    assert a.exit_reason == "stop" and env.book().positions() == {}
    assert env.state().baseline["user"] == {} and not env.events("mismatch")
    sells = [o for o in env.state().orders.values() if o.side == "sell"]
    assert [(o.status, o.filled_qty) for o in sells] == [("filled", 100)]


def test_user_sell_of_strategy_shares_is_not_absorbed(tmp_path):
    """사용자가 전략 보유 종목을 일부 팔아도(전략 매도 주문 없음) 장부에서 전략 몫이 사라지지 않는다 — 경보만."""
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    fk.match("AAAAAA", 10_100)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    fk.user_trade("AAAAAA", -40)                                      # 사용자가 40주를 판다
    env.q("AAAAAA", 10_200, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    st = env.state()
    assert env.book().position("AAAAAA").qty == 100 and st.positions["AAAAAA"].qty == 100
    assert st.baseline["user"] == {"AAAAAA": -40}                      # 계좌에 전략 몫이 모자람(음수 = 사람 확인)
    assert any(a[0] == "error" and "잔고 차이" in a[1] for a in env.alerts)
    assert not env.events("fill") or all(e["side"] == "buy" for e in env.events("fill"))


def test_holidays_file_missing_year_alerts_once(tmp_path):
    env = Env(tmp_path)
    (tmp_path / "holidays.txt").write_text("# 작년 것만\n2025-12-25\n")
    env.run("signal", D1, "07:35")
    assert [a for a in env.alerts if "holidays.txt" in a[1] and "2026" in a[1]]


def test_quote_without_date_never_confirms_session(tmp_path):
    """검토 F4: 시가가 있어도 시세 자체의 일자가 오늘이 아니면(캐시일 수 있다) 세션을 확정하지 않는다."""
    env = Env(tmp_path)
    env.quotes.set_bars(REF_CODE, [DailyBar(d, 1, 1, 1, 1) for d in SESSIONS if d < D1])
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    env.q("AAAAAA", 10_000, open=9_950, prev_close=10_000)            # 어제 시가가 그대로(캐시) — 일자 없음
    assert env.run("open_check", D1, "09:02") == 0
    st = env.state()
    assert st.sessions[D1.isoformat()] == "suspect" and st.positions["AAAAAA"].status == "entry_pending"
    assert not env.events("fill")
    env.q("AAAAAA", 10_000, open=9_950, prev_close=10_000, date=D1)   # 오늘 일자가 찍힌 시세 → 증거
    assert env.run("monitor", D1, "09:06", once=True) == 0
    assert env.state().sessions[D1.isoformat()] == "open" and env.state().positions["AAAAAA"].status == "open"


def test_any_next_open_action_kind_becomes_preopen_auction_sell(tmp_path):
    """설계 §6: 돌파 실패 등 ``*_next_open`` 종류의 Action 은 전부 장전 동시호가 매도 의도로 옮긴다(전략 모듈이 종류를 더해도 그대로)."""
    from swing_it.live.runtime import SwingRuntime, _Ctx
    from swing_it.live.strategy import Action, PositionState

    env = Env(tmp_path)
    _enter(env, {"AAAAAA": {}})
    rt = SwingRuntime(env.cfg, env.bundle, clock=env.clock, alert=lambda *a: None)
    st = env.state()
    ctx = _Ctx(job="signal", asof=D2, state=st, book=StateStore(env.cfg.mode_dir, clock=env.clock).load_book(st),
               oms=None, kill=None)
    pos = st.positions["AAAAAA"]
    s = PositionState.from_dict(pos.strat, DEFAULT_STRATEGY)
    rt._apply_actions(ctx, pos, s, [Action(kind="failed_breakout_next_open", qty_frac=1.0, price=None,
                                           at="next_open", qty=100, day_no=1)], session="open_auction")
    assert pos.exit_intent == {"kind": "failed_breakout_next_open", "qty": 100, "session": "open_auction",
                               "created": D2.isoformat()}
