"""스윙 3호 뼈대 S — 미너비니 피처의 시점 안전·경계, SEPA 시뮬레이터의 진입/청산 규칙 (합성 데이터만)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.signals.sepa import (  # noqa: E402
    SepaPanels,
    build_skeleton_events,
    control_trades,
    simulate_sepa,
    skeleton_features,
)
from swing_it.features.minervini import (  # noqa: E402
    breadth,
    market_breadth,
    pivot_breakout,
    rs_rating,
    trend_template,
)

nan = np.nan


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _frame(cols: dict[str, list | np.ndarray], n: "int | None" = None) -> pd.DataFrame:
    n = n or len(next(iter(cols.values())))
    return pd.DataFrame({k: np.asarray(v, float) for k, v in cols.items()}, index=_days(n))


def _random_panels(n: int = 420, k: int = 6, seed: int = 0) -> SepaPanels:
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(k)]
    drift = np.linspace(-0.001, 0.003, k)
    close = 100 * np.exp(np.cumsum(drift + 0.02 * rng.standard_normal((n, k)), axis=0))
    high = close * (1 + 0.01 * rng.random((n, k)))
    low = close * (1 - 0.01 * rng.random((n, k)))
    opn = (high + low) / 2
    vol = rng.integers(50, 300, (n, k)).astype(float)
    idx = _days(n)

    def f(a):
        return pd.DataFrame(a, index=idx, columns=codes)

    market = pd.Series(["KOSPI", "KOSDAQ"] * (k // 2), index=codes)
    return SepaPanels(open=f(opn), high=f(high), low=f(low), close=f(close), volume=f(vol),
                      trade_value=f(vol * close), market=market)


# --- 시점 안전 ---

def test_features_do_not_look_ahead():
    p = _random_panels()
    cut = 330
    base = skeleton_features(p)
    rng = np.random.default_rng(1)

    def bump(x: pd.DataFrame) -> pd.DataFrame:
        y = x.copy()
        y.iloc[cut + 1:] *= rng.uniform(0.5, 2.0, size=y.iloc[cut + 1:].shape)
        return y

    q = SepaPanels(open=bump(p.open), high=bump(p.high), low=bump(p.low), close=bump(p.close),
                   volume=bump(p.volume), trade_value=p.trade_value, market=p.market)
    moved = skeleton_features(q)
    for k, v in base.items():
        pd.testing.assert_frame_equal(v.iloc[: cut + 1], moved[k].iloc[: cut + 1], obj=k)
    assert not base["rs"].iloc[cut + 1:].equals(moved["rs"].iloc[cut + 1:])   # 미래는 실제로 바뀌었다


# --- 추세 템플릿 ---

def test_trend_template_requires_full_windows():
    c = _frame({"A": 100 + np.arange(300.0)})
    tt = trend_template(c, high=c, low=c)["A"]
    assert not tt.iloc[:251].any()      # 252일 고저 창이 덜 참
    assert tt.iloc[251]                 # 꾸준한 상승 → 첫 꽉 찬 날부터 True


def test_trend_template_near_high_boundary():
    c = _frame({"A": 100 + 0.5 * np.arange(420.0)})      # row 400 종가 = 300
    h = c.copy()
    h.iloc[395, 0] = 400.0                                # 0.75 × 400 = 300 → 경계 포함
    assert trend_template(c, high=h, low=c)["A"].iloc[400]
    h.iloc[395, 0] = 400.5
    assert not trend_template(c, high=h, low=c)["A"].iloc[400]
    # 고가를 안 넘기면 종가 252일 최고로 본다 → 스파이크는 무시
    assert trend_template(c)["A"].iloc[400]


def test_trend_template_needs_30pct_above_low():
    c = _frame({"A": 100 + 0.01 * np.arange(320.0)})     # 이평 정렬·기울기는 맞지만 저점 대비 +3%
    assert not trend_template(c)["A"].iloc[300:].any()


def test_trend_template_needs_rising_ma200():
    # 300일 횡보 → 100일 급락 → 200일 반등: 반등 끝무렵 MA50>MA150>MA200·고저 조건은 맞는데 MA200 이 아직 내려가는 날이 있다
    x = np.concatenate([np.full(300, 300.0), 300 - 2.0 * np.arange(100.0), 100 + np.arange(1, 201.0)])
    c = _frame({"A": x})
    s = c["A"]
    m50, m150, m200 = (s.rolling(w).mean() for w in (50, 150, 200))
    other = ((s > m50) & (m50 > m150) & (m150 > m200)
             & (s >= 0.75 * s.rolling(252).max()) & (s >= 1.30 * s.rolling(252).min()))
    falling = m200 <= m200.shift(20)
    tt = trend_template(c)["A"]
    assert (other & falling).any() and (other & ~falling).any()
    assert not tt[other & falling].any()
    assert tt[other & ~falling].all()


# --- RS ---

def test_rs_rating_ranks_and_population():
    n = 300
    t = np.arange(n, dtype=float)
    c = _frame({"A": 100 * 1.001 ** t, "B": 100 * 1.002 ** t, "C": 100 * 1.003 ** t, "D": 100 * 1.004 ** t})
    rs = rs_rating(c)
    assert rs.iloc[:252].isna().all().all()             # r252 가 필요
    assert list(rs.iloc[-1]) == [25.0, 50.0, 75.0, 100.0]
    pop = pd.DataFrame(True, index=c.index, columns=c.columns)
    pop["D"] = False
    rs2 = rs_rating(c, population=pop)
    assert np.isnan(rs2["D"].iloc[-1])
    assert rs2["C"].iloc[-1] == pytest.approx(100.0)


# --- 피벗 돌파 ---

def _flat(n: int = 60, last_close: float = 10.5, last_vol: float = 150.0, last_high: float = 11.0):
    close = np.full(n + 1, 10.0)
    high = np.full(n + 1, 10.0)
    vol = np.full(n + 1, 100.0)
    close[-1], high[-1], vol[-1] = last_close, last_high, last_vol
    return _frame({"A": high}), _frame({"A": close}), _frame({"A": vol})


def test_pivot_breakout_close_and_volume():
    h, c, v = _flat()
    brk, piv = pivot_breakout(h, c, v)
    assert brk["A"].iloc[-1]
    assert piv["A"].iloc[-1] == 10.0                    # 오늘 고가(11)는 피벗에 안 들어간다
    assert not brk["A"].iloc[:-1].any()
    assert not pivot_breakout(*_flat(last_vol=149.0))[0]["A"].iloc[-1]   # 거래량 1.5배 미달
    assert not pivot_breakout(*_flat(last_close=10.0))[0]["A"].iloc[-1]  # 피벗과 같으면 돌파 아님
    h, c, v = _flat(n=49)                                # 직전 50일이 안 참
    assert not pivot_breakout(h, c, v)[0]["A"].iloc[-1]


# --- breadth ---

def test_breadth_per_market():
    n = 60
    up = 100 + np.arange(n, dtype=float)
    dn = 200 - np.arange(n, dtype=float)
    c = _frame({"A": up, "B": dn, "C": up, "D": up})
    c.loc[c.index[:20], "D"] = nan                       # D 는 MA50 이 늦게 정의 → 분모에서 빠짐
    mk = pd.Series({"A": "KOSPI", "B": "KOSPI", "C": "KOSDAQ", "D": "KOSDAQ"})
    mb = market_breadth(c, mk)
    assert mb["KOSPI"].iloc[-1] == pytest.approx(0.5)
    assert mb["KOSDAQ"].iloc[-1] == pytest.approx(1.0)
    assert mb.iloc[:49].isna().all().all()
    b = breadth(c, mk)
    assert list(b.columns) == ["A", "B", "C", "D"]
    assert b["B"].iloc[-1] == pytest.approx(0.5) and b["D"].iloc[-1] == pytest.approx(1.0)


# --- 사건 표 ---

def test_build_skeleton_events_universe_and_columns():
    p = _random_panels(n=500, k=8, seed=3)
    allev = build_skeleton_events(p, keep_all=True)
    assert {"code", "event_day", "market", "pivot", "tt", "rs", "breadth", "skeleton"} <= set(allev.columns)
    s = build_skeleton_events(p)
    assert s["skeleton"].all() and len(s) == int(allev["skeleton"].sum())
    assert (s["rs"] >= 70).all() and (s["breadth"] > 0.5).all() and s["tt"].all()
    none = pd.DataFrame(False, index=p.close.index, columns=p.close.columns)
    assert build_skeleton_events(p, universe=none).empty


# --- 시뮬레이터 ---

def _sim_panels(o, h, lo, c, ma=None, vol=None):
    Op, H, L, C = (_frame({"A": x}) for x in (o, h, lo, c))
    MA = _frame({"A": ma if ma is not None else np.zeros(len(c))})
    V = _frame({"A": vol}) if vol is not None else None
    return Op, H, L, C, MA, V


def _ev(day_rows, C, pivot=np.inf, code="A"):
    return pd.DataFrame({"code": code, "event_day": C.index[list(day_rows)], "pivot": pivot})


def test_stop_gap_down_fills_at_open():
    Op, H, L, C, MA, V = _sim_panels(o=[99, 100, 90, 90], h=[100, 101, 91, 91], lo=[98, 99, 85, 85],
                                    c=[100, 100, 88, 88])
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), max_hold=3)
    assert t["exit_reason"].iloc[0] == "stop"
    assert t["exit_px"].iloc[0] == 90 and t["ret"].iloc[0] == pytest.approx(-0.10)
    assert t["hold_days"].iloc[0] == 2


def test_stop_intraday_fills_at_stop_price():
    Op, H, L, C, MA, V = _sim_panels(o=[99, 100, 99], h=[100, 101, 100], lo=[98, 91, 95], c=[100, 97, 99])
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), max_hold=5)
    assert t["exit_reason"].iloc[0] == "stop" and t["ret"].iloc[0] == pytest.approx(-0.08)
    assert t["exit_day"].iloc[0] == C.index[1]          # 진입일 장중 손절


def test_ma50_exit_next_open():
    ma = [0, 0, 0, 105, 0, 0]
    Op, H, L, C, MA, V = _sim_panels(o=[100, 100, 101, 102, 103, 104], h=[101] * 2 + [110] * 4,
                                    lo=[99] * 6, c=[100, 101, 102, 104, 106, 107], ma=ma)
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "ma50" and r["exit_day"] == C.index[4] and r["exit_px"] == 103
    assert r["hold_days"] == 4                           # 진입 1일째(row1) ~ 청산(row4)


def test_ma50_exit_deferred_over_halt():
    ma = [0, 0, 105, 0, 0]
    vol = [100, 100, 100, 0, 100]                        # row3 정지(OHLC 이월)
    Op, H, L, C, MA, V = _sim_panels(o=[100, 100, 101, 104, 99], h=[110] * 5, lo=[99, 99, 99, 104, 98],
                                    c=[100, 101, 104, 104, 100], ma=ma, vol=vol)
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), volume=V, max_hold=10)
    assert t["exit_day"].iloc[0] == C.index[4] and t["exit_px"].iloc[0] == 99


def test_time_exit_at_close_of_max_hold():
    Op, H, L, C, MA, V = _sim_panels(o=[100, 100, 101, 102, 103], h=[110] * 5, lo=[99] * 5,
                                    c=[100, 101, 102, 110, 120])
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), max_hold=3)
    r = t.iloc[0]
    assert r["exit_reason"] == "time" and r["exit_day"] == C.index[3] and r["ret"] == pytest.approx(0.10)
    assert r["hold_days"] == 3


def test_time_beats_ma50_on_last_day():
    ma = [0, 0, 0, 200, 0]                               # 3일째(=max_hold) 종가 < MA50 → 시간 청산이 먼저
    Op, H, L, C, MA, V = _sim_panels(o=[100, 100, 101, 102, 50], h=[110] * 5, lo=[99] * 5,
                                    c=[100, 101, 102, 110, 50], ma=ma)
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), max_hold=3)
    assert t["exit_reason"].iloc[0] == "time" and t["exit_px"].iloc[0] == 110


def test_chase_rule_skips_far_above_pivot():
    Op, H, L, C, MA, V = _sim_panels(o=[100, 95, 96], h=[110] * 3, lo=[94] * 3, c=[100, 96, 97])
    t, s = simulate_sepa(Op, H, L, C, MA, _ev([0], C, pivot=90.0), max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_chase"          # 95 > 90 × 1.05
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C, pivot=91.0), max_hold=2)   # 95 ≤ 95.55
    assert len(t) == 1 and t["entry_px"].iloc[0] == 95


def test_limit_up_open_is_skipped():
    Op, H, L, C, MA, V = _sim_panels(o=[100, 130, 131], h=[110, 130, 135], lo=[99, 125, 128], c=[100, 130, 133])
    t, s = simulate_sepa(Op, H, L, C, MA, _ev([0], C), chase=None, max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_limit_up"
    Op.iloc[1, 0] = 129.0                                  # 상한가 아래 시가는 산다
    assert len(simulate_sepa(Op, H, L, C, MA, _ev([0], C), chase=None, max_hold=2)) == 1
    assert len(simulate_sepa(Op.assign(A=[100, 130, 131]), H, L, C, MA, _ev([0], C), chase=None,
                             limit_up_skip=False, max_hold=2)) == 1


def test_halted_entry_day_is_skipped():
    Op, H, L, C, MA, V = _sim_panels(o=[100, 100, 101], h=[110] * 3, lo=[99] * 3, c=[100, 100, 101],
                                    vol=[100, 0, 100])
    t, s = simulate_sepa(Op, H, L, C, MA, _ev([0], C), volume=V, max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_no_open"


def test_no_overlapping_positions_same_code():
    n = 8
    Op, H, L, C, MA, V = _sim_panels(o=[100] * n, h=[110] * n, lo=[99] * n, c=[100 + i for i in range(n)])
    # 사건: row0(진입 row1, 3일 보유 → row3 종가 청산), row2(보유 중 → 버림), row3(청산일 신호 → 허용)
    t, s = simulate_sepa(Op, H, L, C, MA, _ev([0, 2, 3], C), max_hold=3, return_skipped=True)
    assert list(t["event_day"]) == [C.index[0], C.index[3]]
    assert list(s["skip_reason"]) == ["skip_overlap"]
    assert t["exit_day"].iloc[0] == C.index[3] and t["entry_day"].iloc[1] == C.index[4]
    # 대조는 독립 표본 — 겹침 규칙을 끈다
    assert len(control_trades(Op, H, L, C, MA, _ev([0, 2, 3], C).drop(columns="pivot"), max_hold=3)) == 3


def test_delisting_exits_at_last_close_and_flags():
    Op, H, L, C, MA, V = _sim_panels(o=[100, 100, 101, nan, nan], h=[110, 110, 110, nan, nan],
                                    lo=[99, 99, 99, nan, nan], c=[100, 101, 97, nan, nan])
    t = simulate_sepa(Op, H, L, C, MA, _ev([0], C), max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "delisted" and r["exit_px"] == 97 and r["exit_day"] == C.index[2]
    O2, H2, L2, C2, MA2, _ = _sim_panels(o=[100, 100, 101], h=[110] * 3, lo=[99] * 3, c=[100, 101, 103])
    r2 = simulate_sepa(O2, H2, L2, C2, MA2, _ev([0], C2), max_hold=10).iloc[0]
    assert r2["exit_reason"] == "data_end" and r2["exit_px"] == 103


def test_control_trades_pivot_is_prior_50d_high():
    n = 55
    hi = np.full(n, 100.0)
    hi[10] = 120.0
    o = np.full(n, 100.0)
    o[52] = 127.0                                         # 120 × 1.05 = 126 초과 → 추격 금지
    Op, H, L, C, MA, V = _sim_panels(o=o, h=np.maximum(hi, o), lo=np.full(n, 99.0), c=np.full(n, 100.0))
    pairs = pd.DataFrame({"code": ["A"], "event_day": [C.index[51]]})
    t, s = control_trades(Op, H, L, C, MA, pairs, max_hold=2, chase=0.05, limit_up_skip=False,
                          return_skipped=True)
    assert t.empty and s["pivot"].iloc[0] == 120.0 and s["skip_reason"].iloc[0] == "skip_chase"


def test_build_skeleton_events_trend_entry_drops_breakout():
    """``require_breakout=False`` (탐색 격자 진입 "trend") = 유니버스 ∧ 추세 템플릿 ∧ RS ∧ breadth, 돌파 무관.

    기본값(키 없음)은 원래 뼈대 S 와 같아야 한다 — 판정 러너의 동작이 바뀌지 않는다."""
    p = _random_panels(n=500, k=8, seed=3)
    f = skeleton_features(p)
    s_default = build_skeleton_events(p, feats=f)
    s_explicit = build_skeleton_events(p, {"require_breakout": True}, feats=f)
    pd.testing.assert_frame_equal(s_default, s_explicit)

    tr = build_skeleton_events(p, {"require_breakout": False}, feats=f)
    want = f["tt"] & (f["rs"] >= 70) & (f["breadth"] > 0.5)
    assert len(tr) == int(want.to_numpy().sum()) > len(s_default)
    got = set(zip(tr["code"], tr["event_day"], strict=True))
    di, ci = np.nonzero(want.to_numpy())
    assert got == set(zip(p.close.columns[ci], p.days[di], strict=True))
    # 돌파 사건은 전부 trend 사건 안에 있다(돌파 조건만 빠졌다)
    assert set(zip(s_default["code"], s_default["event_day"], strict=True)) <= got
    # 피벗 열은 그대로 직전 50일 최고 고가(추격 금지 규칙의 기준)
    pv = f["pivot"].to_numpy()[tr["day_idx"].to_numpy(), tr["col_idx"].to_numpy()]
    np.testing.assert_allclose(tr["pivot"].to_numpy(), pv)
    # keep_all 은 추세 템플릿 행 전부
    allev = build_skeleton_events(p, {"require_breakout": False}, feats=f, keep_all=True)
    assert len(allev) == int(f["tt"].to_numpy().sum()) and allev["tt"].all()
