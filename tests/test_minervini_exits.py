"""미너비니 원전 청산 프로토콜 — VCP 피처 시점 안전·시뮬레이터 분기 전부 (합성 데이터만, 사전등록 MINERVINI_EXITS.md)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.signals.minervini_exits import (  # noqa: E402
    contraction_low_level,
    control_minervini,
    simulate_minervini,
    vcp_contraction,
    vcp_ok,
)

nan = np.nan


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _frame(cols: dict[str, list | np.ndarray]) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    return pd.DataFrame({k: np.asarray(v, float) for k, v in cols.items()}, index=_days(n))


def _panels(o, h, lo, c, ma=None, vol=None):
    Op, H, L, C = (_frame({"A": x}) for x in (o, h, lo, c))
    MA = _frame({"A": ma if ma is not None else np.zeros(len(c))})
    V = _frame({"A": vol}) if vol is not None else None
    return Op, H, L, C, MA, V


def _ev(rows, C, *, low, pivot=np.inf):
    """사건 = 신호 행 ``rows``, 수축 저점 ``low`` (스칼라 또는 행별 리스트)."""
    return pd.DataFrame({"code": "A", "event_day": C.index[list(rows)], "pivot": pivot, "contraction_low": low})


def _sim(o, h, lo, c, *, low, ma=None, vol=None, rows=(0,), pivot=np.inf, **kw):
    Op, H, L, C, MA, V = _panels(o, h, lo, c, ma, vol)
    kw.setdefault("time_stop_days", None)
    return simulate_minervini(Op, H, L, C, MA, _ev(rows, C, low=low, pivot=pivot), volume=V, **kw), C


# 기본 장면: 신호 행 0, 진입 행 1 시가 100, 수축 저점 95 → 손절가 94.05 → d = 5.95%, 목표 111.9.
D0 = 1 - 95 * 0.99 / 100


# --- VCP 피처 ---

def test_vcp_depth_and_volume_ratio_windows():
    n = 80
    hi = np.full(n, 101.0)
    low = np.full(n, 99.0)
    vol = np.full(n, 100.0)
    t = 70
    hi[t - 10:t] = 104.0                                # 수축 창 t−10..t−1
    low[t - 10:t] = 100.0
    hi[t - 11], low[t - 11] = 130.0, 80.0               # 창 밖(t−11) — 깊이에 안 들어간다
    vol[t - 10:t] = 50.0                                # 수축 창 평균 50
    vol[t - 60:t - 10] = 200.0                          # 직전 50일(t−60..t−11) 평균 200
    vol[t] = 10_000.0                                   # 돌파일 자신 — 안 들어간다
    hi[t], low[t] = 200.0, 1.0
    H, L, V = _frame({"A": hi}), _frame({"A": low}), _frame({"A": vol})
    depth, ratio = vcp_contraction(H, L, V)
    assert depth["A"].iloc[t] == pytest.approx(0.04)
    assert ratio["A"].iloc[t] == pytest.approx(0.25)
    ok = vcp_ok(depth, ratio)
    assert ok["A"].iloc[t]
    # 경계: 깊이 12% 는 통과, 거래량 비 0.8 은 탈락(엄격 부등호)
    assert vcp_ok(depth * 0 + 0.12, ratio)["A"].iloc[t]
    assert not vcp_ok(depth, ratio * 0 + 0.8)["A"].iloc[t]
    # 창이 덜 차면 NaN → 거짓
    assert depth["A"].iloc[:10].isna().all() and ratio["A"].iloc[:60].isna().all() and not ok["A"].iloc[:60].any()
    assert ratio["A"].iloc[60:].notna().all()


def test_vcp_features_do_not_look_ahead():
    rng = np.random.default_rng(0)
    n, k = 200, 4
    c = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal((n, k)), axis=0))
    idx, cols = _days(n), list("ABCD")
    H = pd.DataFrame(c * 1.01, index=idx, columns=cols)
    L = pd.DataFrame(c * 0.99, index=idx, columns=cols)
    V = pd.DataFrame(rng.integers(50, 300, (n, k)).astype(float), index=idx, columns=cols)
    cut = 150
    d0, r0 = vcp_contraction(H, L, V)
    c0 = contraction_low_level(L)

    def bump(x):
        y = x.copy()
        y.iloc[cut + 1:] *= rng.uniform(0.3, 3.0, size=y.iloc[cut + 1:].shape)
        return y

    d1, r1 = vcp_contraction(bump(H), bump(L), bump(V))
    c1 = contraction_low_level(bump(L))
    pd.testing.assert_frame_equal(d0.iloc[: cut + 1], d1.iloc[: cut + 1])
    pd.testing.assert_frame_equal(r0.iloc[: cut + 1], r1.iloc[: cut + 1])
    pd.testing.assert_frame_equal(c0.iloc[: cut + 1], c1.iloc[: cut + 1])
    # 깊이·거래량 비는 t 행 자체도 안 본다: t 행만 흔들어도 t 행 값이 같다
    H2, L2, V2 = H.copy(), L.copy(), V.copy()
    H2.iloc[cut] *= 5
    L2.iloc[cut] *= 0.2
    V2.iloc[cut] *= 50
    d2, r2 = vcp_contraction(H2, L2, V2)
    pd.testing.assert_series_equal(d0.iloc[cut], d2.iloc[cut])
    pd.testing.assert_series_equal(r0.iloc[cut], r2.iloc[cut])


def test_contraction_low_includes_event_day():
    low = np.full(30, 100.0)
    low[20] = 90.0                                       # t
    low[9] = 80.0                                        # t−11: 창 밖
    cl = contraction_low_level(_frame({"A": low}))["A"]
    assert cl.iloc[20] == 90.0                           # 돌파일 포함 t−10..t
    low[10] = 85.0                                       # t−10: 창 안
    assert contraction_low_level(_frame({"A": low}))["A"].iloc[20] == 85.0
    assert cl.iloc[:10].isna().all() and cl.iloc[10] == 80.0


# --- 손절폭 자르기·버림 ---

def test_stop_from_contraction_low_and_clip_floor_cap():
    o, h, lo, c = [100, 100, 100], [101, 101, 101], [99, 99, 99], [100, 100, 100]
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=2)
    assert t["d"].iloc[0] == pytest.approx(D0) and t["stop_px"].iloc[0] == pytest.approx(95 * 0.99)
    assert t["target_px"].iloc[0] == pytest.approx(100 * (1 + 2 * D0))
    t, _ = _sim(o, h, lo, c, low=99.5, max_hold=2)       # 원 손절폭 1.5% → 하한 3%
    assert t["d"].iloc[0] == pytest.approx(0.03) and t["stop_px"].iloc[0] == pytest.approx(97.0)
    assert t["exit_reason"].iloc[0] == "time"            # 저가 99 > 97 — 자른 뒤 손절가로 본다
    t, _ = _sim(o, h, lo, c, low=80.0, max_hold=2)       # 원 손절폭 20.8% → 상한 10%
    assert t["d"].iloc[0] == pytest.approx(0.10) and t["stop_px"].iloc[0] == pytest.approx(90.0)
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=2, fixed_stop=0.08)   # 민감도: 고정 8%
    assert t["d"].iloc[0] == pytest.approx(0.08) and t["target_px"].iloc[0] == pytest.approx(116.0)


def test_skip_when_open_at_or_below_contraction_low():
    o, h, lo, c = [100, 100, 100], [101] * 3, [99] * 3, [100] * 3
    Op, H, L, C, MA, V = _panels(o, h, lo, c)
    for low in (100.0, 100.5):
        t, s = simulate_minervini(Op, H, L, C, MA, _ev([0], C, low=low), max_hold=2, return_skipped=True)
        assert t.empty and s["skip_reason"].iloc[0] == "skip_below_low"
    t, s = simulate_minervini(Op, H, L, C, MA, _ev([0], C, low=nan), max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_no_contraction"


# --- 전량 손절 ---

def test_full_stop_intraday_and_gap_down():
    t, C = _sim([100, 100, 100], [101, 101, 101], [99, 93, 99], [100, 96, 100], low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "stop" and r["exit_px"] == pytest.approx(94.05) and r["exit_day"] == C.index[1]
    assert not r["partial_hit"] and r["ret"] == pytest.approx(-D0) and r["R"] == pytest.approx(-1.0)
    assert r["ret_half1"] == r["ret_half2"] == pytest.approx(-D0)
    t, C = _sim([100, 100, 90, 90], [101, 101, 91, 91], [99, 99, 85, 85], [100, 100, 88, 88], low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "stop" and r["exit_px"] == 90 and r["ret"] == pytest.approx(-0.10)
    assert r["hold_days"] == 2 and r["R"] == pytest.approx(-0.10 / D0)


def test_same_day_stop_and_target_stop_first():
    t, _ = _sim([100, 100, 100], [101, 120, 101], [99, 90, 99], [100, 100, 100], low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "stop" and not r["partial_hit"] and r["ret"] == pytest.approx(-D0)
    # 갭 상승 시가로 목표를 넘긴 날도 저가가 손절가에 닿았으면 손절 먼저(사전등록 문구 그대로)
    t, _ = _sim([100, 100, 115, 115], [101, 101, 116, 116], [99, 99, 90, 110], [100, 100, 112, 112],
                low=95.0, max_hold=10)
    assert t["exit_reason"].iloc[0] == "stop" and not t["partial_hit"].iloc[0]


# --- 절반 익절 → 본전 ---

def test_partial_at_target_then_breakeven():
    tgt = 100 * (1 + 2 * D0)
    # 진입 행1, 행2 고가 120 ≥ 111.9 → 절반 익절, 행3 저가 99 ≤ 100 → 나머지 본전
    t, C = _sim([100, 100, 105, 110, 110], [101, 103, 120, 111, 111], [99, 99, 104, 99, 99],
                [100, 102, 110, 105, 105], low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["partial_hit"] and r["partial_day"] == C.index[2] and r["partial_px"] == pytest.approx(tgt)
    assert r["exit_reason"] == "breakeven" and r["exit_day"] == C.index[3] and r["exit_px"] == 100
    assert r["ret_half1"] == pytest.approx(2 * D0) and r["ret_half2"] == pytest.approx(0.0)
    assert r["ret"] == pytest.approx(D0) and r["R"] == pytest.approx(1.0)


def test_partial_on_entry_day_breakeven_starts_next_day():
    # 진입일 시가 = 진입가 → 저가 ≤ 진입가가 항상 참. 같은 날 본전 검사는 하지 않는다(다음 날부터).
    t, C = _sim([100, 100, 112, 112], [101, 120, 113, 113], [99, 98, 101, 98], [100, 112, 112, 112],
                low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["partial_day"] == C.index[1] and r["exit_reason"] == "breakeven" and r["exit_day"] == C.index[3]


def test_breakeven_gap_down_fills_at_open():
    t, _ = _sim([100, 100, 105, 97, 97], [101, 103, 120, 98, 98], [99, 99, 104, 96, 96], [100, 102, 110, 97, 97],
                low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "breakeven" and r["exit_px"] == 97
    assert r["ret"] == pytest.approx(0.5 * (2 * D0 - 0.03))


def test_partial_via_gap_up_open():
    t, C = _sim([100, 100, 115, 116, 116], [101, 103, 118, 117, 117], [99, 99, 114, 101, 101],
                [100, 102, 116, 116, 116], low=95.0, max_hold=4)
    r = t.iloc[0]
    assert r["partial_px"] == 115 and r["ret_half1"] == pytest.approx(0.15)
    assert r["exit_reason"] == "time" and r["exit_px"] == 116 and r["ret"] == pytest.approx(0.155)
    # 갭 상승 익절일엔 시가가 먼저라 그날 저가 ≤ 진입가면 그날 본전
    t, C = _sim([100, 100, 115, 116], [101, 103, 118, 117], [99, 99, 99, 101], [100, 102, 105, 116],
                low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "breakeven" and r["exit_day"] == C.index[2] and r["exit_px"] == 100
    assert r["ret"] == pytest.approx(0.075)


# --- 추적 MA ---

def test_trail_exit_next_open_after_partial_only():
    ma = [0, 0, 0, 0, 200, 0, 0]
    t, C = _sim([100, 100, 105, 112, 113, 108, 109], [101, 103, 120, 115, 115, 110, 110],
                [99, 99, 104, 105, 105, 104, 104], [100, 102, 113, 114, 114, 108, 108], ma=ma, low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "trail" and r["exit_day"] == C.index[5] and r["exit_px"] == 108
    assert r["ret"] == pytest.approx(0.5 * (2 * D0 + 0.08))
    # 익절 전에는 MA 이탈로 나가지 않는다(전량은 손절·시간 손절·최대 보유만)
    ma = [0, 200, 200, 200, 200]
    t, C = _sim([100, 100, 101, 102, 103], [101, 103, 103, 104, 104], [99, 99, 99, 99, 99],
                [100, 102, 102, 103, 104], ma=ma, low=95.0, max_hold=4)
    assert t["exit_reason"].iloc[0] == "time" and t["exit_px"].iloc[0] == 104


def test_trail_beats_breakeven_on_same_open():
    ma = [0, 0, 200, 0, 0]
    t, C = _sim([100, 100, 105, 97, 97], [101, 103, 120, 98, 98], [99, 99, 104, 96, 96], [100, 102, 110, 97, 97],
                ma=ma, low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "trail" and r["exit_px"] == 97 and r["exit_day"] == C.index[3]


def test_trail_deferred_over_halt():
    ma = [0, 0, 200, 0, 0, 0]
    vol = [100, 100, 100, 0, 100, 100]
    t, C = _sim([100, 100, 105, 110, 109, 109], [101, 103, 120, 110, 112, 112], [99, 99, 104, 110, 104, 104],
                [100, 102, 110, 110, 110, 110], ma=ma, vol=vol, low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "trail" and r["exit_day"] == C.index[4] and r["exit_px"] == 109


# --- 시간 손절·최대 보유 ---

def _drift(n, start=100.0, step=0.0):
    c = start + step * np.arange(n)
    return list(c), list(c + 1), list(c - 1), list(c)


def test_time_stop_at_day20_close():
    n = 30
    o, h, lo, c = _drift(n)
    c[20] = 99.5                                         # 진입 행1 → 20일째 = 행20
    t, C = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=20)
    r = t.iloc[0]
    assert r["exit_reason"] == "time_stop" and r["exit_day"] == C.index[20] and r["exit_px"] == 99.5
    assert r["hold_days"] == 20
    c[20] = 100.0                                         # 종가 = 진입가 → 경계 포함(≤)
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=20)
    assert t["exit_reason"].iloc[0] == "time_stop"
    c[20] = 100.5                                         # 진입가 위 → 시간 손절 없음, 끝까지
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=20)
    assert t["exit_reason"].iloc[0] == "data_end"
    c[20] = 99.5                                          # 민감도: 시간 손절 끔
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=None)
    assert t["exit_reason"].iloc[0] == "data_end"


def test_time_stop_deferred_over_halt_and_only_once():
    n = 30
    o, h, lo, c = _drift(n)
    vol = [100.0] * n
    vol[20] = 0.0                                        # 20일째 정지 → 다음 거래 가능일(행21) 종가에서 본다
    c[21] = 99.0
    c[25] = 98.0
    t, C = _sim(o, h, lo, c, vol=vol, low=95.0, max_hold=120, time_stop_days=20)
    assert t["exit_reason"].iloc[0] == "time_stop" and t["exit_day"].iloc[0] == C.index[21]
    c[21] = 101.0                                        # 그날 진입가 위면 이후(행25)엔 다시 보지 않는다
    t, _ = _sim(o, h, lo, c, vol=vol, low=95.0, max_hold=120, time_stop_days=20)
    assert t["exit_reason"].iloc[0] == "data_end"


def test_no_time_stop_if_partial_hit():
    n = 30
    o, h, lo, c = _drift(n)
    h[5] = 115.0                                          # 5일째 익절
    for i in range(6, n):                                 # 이후 본전(100)에 안 닿게, 20일째만 진입가 아래 종가
        o[i], h[i], lo[i], c[i] = 101.0, 102.0, 100.5, 101.0
    lo[20], c[20] = 99.4, 99.5
    t, C = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=20)
    r = t.iloc[0]
    # 시간 손절(종가 99.5 전량)이 아니라 나머지 절반 본전(100)
    assert r["partial_hit"] and r["exit_reason"] == "breakeven" and r["exit_day"] == C.index[20]
    assert r["exit_px"] == 100 and r["ret_half1"] == pytest.approx(2 * D0)
    lo[20] = 100.5                                        # 20일째에도 본전 안 닿으면 끝까지
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=20)
    assert t["exit_reason"].iloc[0] == "data_end"
    # 20일째 당일 익절도 "그날까지 익절" — 시간 손절 없음
    o, h, lo, c = _drift(n)
    h[20], c[20] = 115.0, 99.5
    t, _ = _sim(o, h, lo, c, low=95.0, max_hold=120, time_stop_days=20)
    assert t["partial_hit"].iloc[0] and t["exit_reason"].iloc[0] != "time_stop"


def test_max_hold_120_close():
    n = 130
    o, h, lo, c = _drift(n, step=0.05)                   # 서서히 오름, 목표·손절 안 닿음
    t, C = _sim(o, h, lo, c, low=95.0, time_stop_days=20)
    r = t.iloc[0]
    assert r["exit_reason"] == "time" and r["exit_day"] == C.index[120] and r["hold_days"] == 120
    assert r["exit_px"] == pytest.approx(c[120])
    # 익절 뒤 나머지 절반도 120일째 종가
    h2 = list(h)
    h2[3] = 200.0
    lo2 = [x if i < 4 else c[i] - 0.01 for i, x in enumerate(lo)]   # 익절 뒤 본전(진입가)에 안 닿게
    t, C = _sim(o, h2, lo2, c, low=95.0, time_stop_days=20)
    r = t.iloc[0]
    assert r["partial_hit"] and r["exit_reason"] == "time" and r["exit_day"] == C.index[120]
    assert r["ret_half2"] == pytest.approx(c[120] / o[1] - 1)


# --- 체결 제약(sepa 와 같은 규칙) ---

def test_halt_entry_skipped_and_halt_blocks_stop():
    (t, s), _ = _sim([100, 100, 101], [110] * 3, [99] * 3, [100, 100, 101], vol=[100, 0, 100], low=95.0,
                max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_no_open"
    # 정지일(행2)은 이월 저가가 손절가 아래여도 체결 안 됨 → 재개일 갭 하락 시가
    t, C = _sim([100, 100, 93, 92], [101, 101, 93, 93], [99, 99, 93, 91], [100, 100, 93, 92],
                vol=[100, 100, 0, 100], low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["exit_reason"] == "stop" and r["exit_day"] == C.index[3] and r["exit_px"] == 92


def test_chase_and_limit_up_skip():
    o, h, lo, c = [100, 95, 96], [110] * 3, [94] * 3, [100, 96, 97]
    (t, s), _ = _sim(o, h, lo, c, low=90.0, pivot=90.0, max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_chase"
    t, _ = _sim(o, h, lo, c, low=90.0, pivot=91.0, max_hold=2)
    assert len(t) == 1
    o, h, lo, c = [100, 130, 131], [110, 130, 135], [99, 125, 128], [100, 130, 133]
    (t, s), _ = _sim(o, h, lo, c, low=95.0, chase=None, max_hold=2, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_limit_up"
    t, _ = _sim(o, h, lo, c, low=95.0, chase=None, limit_up_skip=False, max_hold=2)
    assert len(t) == 1


def test_delisting_after_partial():
    t, C = _sim([100, 100, 105, 108, nan], [101, 103, 120, 109, nan], [99, 99, 104, 105, nan],
                [100, 102, 110, 107, nan], low=95.0, max_hold=10)
    r = t.iloc[0]
    assert r["partial_hit"] and r["exit_reason"] == "delisted" and r["exit_px"] == 107
    assert r["ret"] == pytest.approx(0.5 * (2 * D0 + 0.07))
    t, _ = _sim([100, 100, 101], [101] * 3, [99] * 3, [100, 100, 103], low=95.0, max_hold=10)
    assert t["exit_reason"].iloc[0] == "data_end" and t["ret"].iloc[0] == pytest.approx(0.03)


def test_no_overlap_until_final_half_exits():
    n = 10
    ma = [0.0] * n
    o = [100.0, 100.0] + [105.0] * (n - 2)               # 익절 뒤 시가는 본전(100) 위
    h = [101.0, 101.0, 120.0] + [110.0] * (n - 3)        # 행2 절반 익절
    lo = [99.0, 99.0, 104.0] + [104.0] * (n - 3)
    c = [100.0, 100.0, 110.0] + [108.0] * (n - 3)
    ma[5] = 200.0                                        # 행5 종가 이탈 → 행6 시가 나머지 청산
    # 사건: 행0(진입), 행3(절반만 나간 보유 중 → 버림), 행6(청산일 신호 → 허용)
    (t, s), _ = _sim(o, h, lo, c, ma=ma, low=95.0, rows=(0, 3, 6), max_hold=3 + 100, return_skipped=True)
    assert list(t["event_day"]) == [_days(n)[0], _days(n)[6]]
    assert list(s["skip_reason"]) == ["skip_overlap"]
    assert t["exit_day"].iloc[0] == _days(n)[6]
    Op, H, L, C, MA, V = _panels(o, h, lo, c, ma)
    pairs = pd.DataFrame({"code": "A", "event_day": C.index[[0, 3, 6]], "contraction_low": 95.0})
    assert len(control_minervini(Op, H, L, C, MA, pairs, chase=None)) == 3     # 대조는 겹침 허용


def test_control_uses_own_contraction_low_and_pivot():
    n = 70
    hi = np.full(n, 101.0)
    lo = np.full(n, 99.0)
    lo[55] = 96.0                                         # 신호 행 60 의 t−10..t 창 안
    lo[48] = 50.0                                         # 창 밖
    hi[30] = 104.0                                        # 피벗 = 직전 50일 최고 고가 = 104
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    Op, H, L, C, MA, V = _panels(o, hi, lo, c)
    pairs = pd.DataFrame({"code": ["A"], "event_day": [C.index[60]]})
    t = control_minervini(Op, H, L, C, MA, pairs, max_hold=3, time_stop_days=None)
    r = t.iloc[0]
    assert r["contraction_low"] == 96.0 and r["pivot"] == 104.0
    assert r["d"] == pytest.approx(1 - 96 * 0.99 / 100)
    o2 = o.copy()
    o2[61] = 110.0                                        # 104 × 1.05 = 109.2 초과 → 추격 금지
    t, s = control_minervini(_frame({"A": o2}), H, L, C, MA, pairs, max_hold=3, return_skipped=True)
    assert t.empty and s["skip_reason"].iloc[0] == "skip_chase"


def test_ret_is_average_of_halves_and_r():
    t, _ = _sim([100, 100, 105, 110, 110], [101, 103, 120, 111, 111], [99, 99, 104, 101, 101],
                [100, 102, 110, 105, 106], low=95.0, max_hold=4)
    r = t.iloc[0]
    assert r["ret"] == pytest.approx(0.5 * (r["ret_half1"] + r["ret_half2"]))
    assert r["ret_half2"] == pytest.approx(0.06) and r["R"] == pytest.approx(r["ret"] / r["d"])
