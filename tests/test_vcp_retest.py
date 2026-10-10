"""VCP 재시험(``research/signals/vcp.py``) — 손으로 확인한 합성 경로.

사전등록 ``research/logs/sepa_explore/VCP_RETEST.md`` §2~§7: VCP 검출(연속 축소 수축·마지막 ≤ 10%·룩어헤드 없음),
피벗 매수정지(갭 3% 버림·체결가 = max(시가, 주문가)·주문 전 손절 9.5% 거르기), 진입일 손절 보수/낙관, 돌파 실패 청산,
분기 단독 Q4(연간 − 9M)·이중 시간축, 기반 수급 F_base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.signals import vcp as V

BAND = 0.004          # 합성 봉: 고가 = 종가 × (1 + 0.4%), 저가 = 종가 × (1 − 0.4%)


def _path(points: list[tuple[int, float]], n: int) -> np.ndarray:
    """(행, 가격) 꼭짓점을 직선으로 이은 종가 경로(마지막 꼭짓점 뒤는 평평)."""
    xs, ys = zip(*points, strict=True)
    return np.interp(np.arange(n), xs, ys)


def _panel(close: np.ndarray, volume: np.ndarray, code: str = "A00001"):
    days = pd.bdate_range("2024-01-01", periods=len(close))
    c = pd.DataFrame({code: close}, index=days)
    h = c * (1 + BAND)
    lo = c * (1 - BAND)
    v = pd.DataFrame({code: volume}, index=days)
    return h, lo, c, v


# 깨끗한 3수축 VCP — 앞 140봉은 오름(창 260 을 채운다), 신호 행 280, 주문일 281.
#   H0 = 100 (행 220, 기반 60일) → 76 (c1 ≈ 24.6%) → 96 → 86 (c2 ≈ 11.1%) → 95 (행 264, 진행 중 고점 = 피벗) → 91.5
#   (c3 ≈ 4.5%, 미확정) → 93.
OFF = 140
CLEAN = [(0, 40.0)] + [(x + OFF, y) for x, y in [(0, 50.0), (60, 90.0), (80, 100.0), (92, 76.0), (104, 96.0),
                                                   (114, 86.0), (124, 95.0), (132, 91.5), (139, 93.0), (140, 93.0)]]
N = 141 + OFF
S = 139 + OFF
DRY_FROM = 125 + OFF


def _shift(pts):
    return [(0, 40.0)] + [(x + OFF, y) for x, y in pts]


def _vol(n: int = N, dry_from: int = DRY_FROM) -> np.ndarray:
    v = np.full(n, 1000.0)
    v[dry_from:] = 500.0
    return v


def _scan(points, vol=None, s: int = S, **params):
    h, lo, c, v = _panel(_path(points, N), _vol() if vol is None else vol)
    H, L, C, Vv = (np.ascontiguousarray(x.to_numpy(float).T) for x in (h, lo, c, v))
    r = V.vcp_scan_arrays(H, L, C, Vv, np.array([0]), np.array([s]), params or None)
    return {k: v_[0] for k, v_ in r.items()}


def test_clean_three_contraction_vcp_passes():
    r = _scan(CLEAN)
    assert r["status"] == V.V_OK, V.VCP_STATUS[r["status"]]
    assert r["n_contr"] == 3
    assert r["i0"] == 80 + OFF and r["ip"] == 124 + OFF
    assert r["h0"] == pytest.approx(100 * (1 + BAND))
    assert r["pivot"] == pytest.approx(95 * (1 + BAND))
    assert r["c1"] == pytest.approx(1 - 76 * (1 - BAND) / (100 * (1 + BAND)))
    assert r["l_last"] == pytest.approx(91.5 * (1 - BAND))
    assert r["c_last"] == pytest.approx(1 - 91.5 * (1 - BAND) / (95 * (1 + BAND)))
    assert r["c_last"] < 0.05                      # 진행 중 수축 — 5% 미만이어도 된다(피벗 = 진행 중 고점)
    # 마름(정정 M7): 마지막 수축(265..280) 500 / 수축 시작 전 50일(215..264) 1000
    assert r["vol_ratio"] == pytest.approx(0.5)


def test_literal_confirmed_only_reading_differs():
    """문자 해석(마지막 확정 고점만)이면 피벗 = 96 고점(행 244) — 한 수축 앞의 낡은 고점(리뷰 C2)."""
    r = _scan(CLEAN, confirmed_only=True)
    assert r["n_contr"] == 2
    assert r["ip"] == 104 + OFF
    assert r["pivot"] == pytest.approx(96 * (1 + BAND))


def test_non_decreasing_contractions_fail():
    pts = _shift([(0, 50.0), (60, 90.0), (80, 100.0), (92, 76.0), (104, 96.0), (114, 71.0), (124, 95.0), (132, 91.5),
                  (139, 93.0), (140, 93.0)])            # c2 ≈ 26% > c1 ≈ 24.6%
    assert _scan(pts)["status"] == V.V_NOT_DECR


def test_last_contraction_over_10pct_fails():
    pts = _shift([(0, 50.0), (60, 90.0), (80, 100.0), (92, 76.0), (104, 96.0), (114, 79.0), (124, 95.0), (132, 84.0),
                  (139, 86.0), (140, 86.0)])            # c2 ≈ 18%, c3 ≈ 12.3% (확정 — 내림 상태, 피벗 = 그 고점)
    r = _scan(pts)
    assert r["ip"] == 124 + OFF
    assert r["c_last"] > 0.10
    assert r["status"] == V.V_LAST


def test_pivot_too_recent_fails():
    """진행 중 고점이 신호 행 2일 전이면(피벗 뒤 < 3 거래일) 셋업 없음."""
    pts = _shift([(0, 50.0), (60, 90.0), (80, 100.0), (92, 76.0), (104, 96.0), (114, 86.0), (137, 95.0), (139, 94.0),
                  (140, 94.0)])
    r = _scan(pts)
    assert r["ip"] == 137 + OFF
    assert r["status"] == V.V_RECENT


def test_base_length_bounds():
    r = _scan([(0, 40.0), (270, 100.0), (280, 95.0), (281, 95.0)])   # H0 가 10일 앞 → 기반 < 15
    assert r["status"] == V.V_BASE and r["i0"] == 270
    pts = [(0, 50.0), (25, 100.0), (100, 80.0), (280, 90.0), (281, 90.0)]
    r = _scan(pts)                                     # 창(21..280) 최고 고가 행 25 → 기반 255 > 250
    assert r["status"] == V.V_BASE and r["i0"] == 25


def test_h0_tie_takes_later_bar():
    c = np.full(N, 50.0)
    c[100] = 100.0
    c[150] = 100.0
    h, lo, cc, v = _panel(c, np.full(N, 1000.0))
    H, L, C, Vv = (np.ascontiguousarray(x.to_numpy(float).T) for x in (h, lo, cc, v))
    r = V.vcp_scan_arrays(H, L, C, Vv, np.array([0]), np.array([S]))
    assert r["i0"][0] == 150


def test_volume_not_dry_fails():
    assert _scan(CLEAN, vol=np.full(N, 1000.0))["status"] == V.V_DRY


def test_no_lookahead_setup_on_d_ignores_bar_d():
    """주문일 d 행의 셋업은 d−1 까지만 본다: d 봉을 바꿔도 셋업이 같고, d−1 봉을 바꾸면 달라진다."""
    h, lo, c, v = _panel(_path(CLEAN, N), _vol())
    mask = pd.DataFrame(False, index=c.index, columns=c.columns)
    mask.iloc[S] = True
    base = V.vcp_setups(h, lo, c, v, mask)
    assert len(base) == 1
    assert base["order_day"].iloc[0] == c.index[S + 1] and base["event_day"].iloc[0] == c.index[S]
    assert bool(base["valid"].iloc[0])
    h2, lo2, c2, v2 = h.copy(), lo.copy(), c.copy(), v.copy()
    h2.iloc[S + 1] = 500.0
    lo2.iloc[S + 1] = 1.0
    c2.iloc[S + 1] = 300.0
    v2.iloc[S + 1] = 1e9
    same = V.vcp_setups(h2, lo2, c2, v2, mask)
    cols = ["valid", "pivot", "l_last", "i0", "ip", "n_contr", "c1", "c_last", "vol_ratio"]
    pd.testing.assert_frame_equal(base[cols], same[cols])
    c3 = c.copy()
    c3.iloc[S] = 99.0                                 # d−1 종가가 피벗 위 → 셋업 깨짐
    assert not bool(V.vcp_setups(h, lo, c3, v, mask)["valid"].iloc[0])


# --- 매수정지 ---

def _book(rows: list[tuple[float, float, float, float]], ma=None, vol=None):
    """(시가, 고가, 저가, 종가) 행 → :class:`vcp.Arrays` (종목 1개)."""
    days = pd.bdate_range("2024-01-01", periods=len(rows))
    a = np.array(rows, float)
    df = {k: pd.DataFrame({"X": a[:, i]}, index=days) for i, k in enumerate("ohlc")}
    m = pd.DataFrame({"X": np.full(len(rows), np.nan) if ma is None else ma}, index=days)
    v = pd.DataFrame({"X": np.full(len(rows), 1000.0) if vol is None else vol}, index=days)
    return V.Arrays(df["o"], df["h"], df["l"], df["c"], m, v)


FLAT = (100.0, 100.5, 99.5, 99.8)          # 셋업 전 봉(종가 < 피벗 100)
UP = (103.0, 104.0, 102.5, 103.5)          # 진입 뒤 평온(손절·피벗 아래로 안 감)
SL = 1 - V.BUYSTOP_PARAMS["sell_slip"]


def _one(A, d: int, pivot: float = 100.0, l_last: float = 95.0, **kw):
    ev = pd.DataFrame({"code": ["X"], "day_idx": [d], "pivot": [pivot], "l_last": [l_last], "ip": [0]})
    if "old_low" in kw:
        ev["old_low"] = kw.pop("old_low")
        kw["old_stop"] = True
    return V.simulate_buystop(A, ev, **kw)


def test_buystop_fill_at_order_price_with_slippage():
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    t, sk = _one(A, 3)
    assert len(t) == 1 and sk.empty
    r = t.iloc[0]
    assert r["order_px"] == 101.0                 # 피벗 100 + 1호가(2,000원 미만 = 1원)
    assert r["fill_px"] == 101.0                  # 시가 99 < 주문가 → 주문가
    assert r["entry_px"] == pytest.approx(101.0 * 1.001)
    assert r["stop_px"] == 94.0                   # L_last 95 − 1호가
    assert r["d"] == pytest.approx((101.101 - 94.0) / 101.101)
    assert not r["gap_fill"] and not r["amb"]
    assert r["exit_reason"] == "data_end"


def test_buystop_gap_fill_at_open():
    A = _book([FLAT, FLAT, FLAT, (102.0, 104.0, 101.5, 103.0)] + [UP] * 6)
    t, _ = _one(A, 3)
    assert t.iloc[0]["fill_px"] == 102.0          # max(시가 102, 주문가 101)
    assert bool(t.iloc[0]["gap_fill"])


def test_buystop_gap_over_3pct_skipped():
    A = _book([FLAT, FLAT, FLAT, (103.5, 106.0, 103.0, 105.0)] + [UP] * 6)
    t, sk = _one(A, 3)
    assert t.empty and sk["skip_reason"].tolist() == ["skip_gap>3%"]


def test_buystop_no_fill_below_order():
    A = _book([FLAT, FLAT, FLAT, (99.0, 100.9, 98.5, 100.0)] + [UP] * 6)
    t, sk = _one(A, 3)
    assert t.empty and sk["skip_reason"].tolist() == ["no_fill"]


def test_buystop_stop_over_9_5pct_pre_skipped():
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    t, sk = _one(A, 3, l_last=91.0)               # 손절가 90 → (100 − 90)/100 = 10% > 9.5%
    assert t.empty and sk["skip_reason"].tolist() == ["skip_risk>9.5%"]
    t2, _ = _one(A, 3, l_last=91.6)               # 손절가 90.6 → 9.4% ≤ 9.5% → 체결
    assert len(t2) == 1


def test_buystop_d_floor_2pct():
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 99.6, 102.0)] + [UP] * 6)
    t, _ = _one(A, 3, l_last=100.5)               # 손절가 99.5 → d ≈ 1.6% < 2% → 진입가 × 0.98
    r = t.iloc[0]
    assert r["d"] == pytest.approx(0.02)
    assert r["stop_px"] == pytest.approx(r["entry_px"] * 0.98)


def test_entry_day_stop_ambiguity_two_paths():
    """시가 ≤ 피벗 ∧ 진입일 저가 ≤ 손절가 = 모호: 기본 열 = 낙관 경로(손절 없음), ``ret_stop0`` = 보수(손절가 × 0.999)."""
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 93.5, 102.0)] + [UP] * 6)
    t, _ = _one(A, 3)
    r = t.iloc[0]
    assert bool(r["amb"])
    assert r["exit_reason"] == "data_end"
    assert r["ret_stop0"] == pytest.approx(94.0 * SL / r["entry_px"] - 1)
    # 갭 체결(시가 > 피벗)이면 체결이 시가라 저가는 체결 뒤 — 모호하지 않고 늘 진입일 손절.
    B = _book([FLAT, FLAT, FLAT, (102.0, 104.0, 93.5, 103.0)] + [UP] * 6)
    tb, _ = _one(B, 3)
    assert not bool(tb.iloc[0]["amb"])
    assert tb.iloc[0]["exit_reason"] == "stop_entry"
    assert tb.iloc[0]["exit_px"] == pytest.approx(94.0 * SL)
    assert tb.iloc[0]["ret_stop0"] == pytest.approx(tb.iloc[0]["ret"])


def test_failed_breakout_on_entry_day_exits_next_open():
    rows = [FLAT, FLAT, FLAT, (99.0, 102.0, 98.0, 99.5), (99.2, 100.0, 98.5, 99.0)] + [UP] * 5
    A = _book(rows)
    t, _ = _one(A, 3)
    r = t.iloc[0]
    assert r["exit_reason"] == "failed_breakout"   # 진입일 종가 99.5 < 피벗 100 → 다음날 시가(정정 M8)
    assert r["exit_day"] == A.days[4] and r["exit_px"] == 99.2
    assert r["hold_days"] == 4 - 2                 # 청산 행 − 신호 행


def test_entry_day_2r_not_recognized_then_partial_and_breakeven():
    # 진입 101.101, 손절 94 → d ≈ 7.02%, 목표 ≈ 115.3. 진입일 고가 120(인정 안 함), 행 4 고가 116 → 절반 익절,
    # 행 5 저가 101 ≤ 진입가 → 본전(× 0.999).
    rows = [FLAT, FLAT, FLAT, (99.0, 120.0, 98.0, 110.0), (110.0, 116.0, 109.0, 114.0), (104.0, 105.0, 101.0, 102.0)]
    rows += [UP] * 4
    t, _ = _one(_book(rows), 3)
    r = t.iloc[0]
    e = 101.0 * 1.001
    tgt = e * (1 + 2 * (e - 94.0) / e)
    assert bool(r["partial_hit"]) and r["partial_day"] == pd.bdate_range("2024-01-01", periods=10)[4]
    assert r["partial_px"] == pytest.approx(tgt)
    assert r["exit_reason"] == "breakeven" and r["exit_px"] == pytest.approx(e * SL)
    assert r["ret"] == pytest.approx(0.5 * (tgt / e - 1) + 0.5 * (SL - 1))


def test_x2_old_stop_rule():
    """X2: 같은 매수정지, 손절만 옛 규칙 — d = clip(1 − 10일 저점 × 0.99 / 진입가, 3%, 10%)."""
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    t, _ = _one(A, 3, old_low=98.0)
    e = 101.0 * 1.001
    assert t.iloc[0]["d"] == pytest.approx(1 - 98.0 * 0.99 / e)
    t2, _ = _one(A, 3, old_low=100.5)             # 1 − 99.495/101.101 ≈ 1.6% → 하한 3%
    assert t2.iloc[0]["d"] == pytest.approx(0.03)
    assert t2.iloc[0]["stop_px"] == pytest.approx(e * 0.97)


def test_x1_old_timing_new_stop():
    """X1: 신호 행 종가 돌파 → 다음날 시가, 추격 5% 버림, 새 손절 d > 10% 면 안 산다."""
    rows = [FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0), (102.5, 104.0, 102.0, 103.5)] + [UP] * 5
    A = _book(rows)
    ev = pd.DataFrame({"code": ["X"], "sig_row": [3], "pivot": [100.0], "l_last": [95.0], "ip": [0]})
    t, sk = V.simulate_x1(A, ev)
    assert len(t) == 1 and t.iloc[0]["entry_px"] == 102.5 and t.iloc[0]["entry_day"] == A.days[4]
    assert t.iloc[0]["d"] == pytest.approx((102.5 - 94.0) / 102.5)
    t2, sk2 = V.simulate_x1(A, ev.assign(l_last=92.0))     # (102.5 − 91)/102.5 ≈ 11.2% > 10%
    assert t2.empty and sk2["skip_reason"].tolist() == ["skip_d>10%"]
    B = _book(rows[:4] + [(105.5, 106.0, 105.0, 105.5)] + [UP] * 5)
    t3, sk3 = V.simulate_x1(B, ev)
    assert t3.empty and sk3["skip_reason"].tolist() == ["skip_chase>5%"]


def test_one_fill_per_pivot_and_no_overlap():
    rows = [FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6
    A = _book(rows)
    ev = pd.DataFrame({"code": ["X", "X"], "day_idx": [3, 4], "pivot": [100.0, 100.0], "l_last": [95.0, 95.0],
                       "ip": [0, 0]})
    t, sk = V.simulate_buystop(A, ev)
    assert len(t) == 1 and sk["skip_reason"].tolist() == ["skip_overlap"]
    t2, sk2 = V.simulate_buystop(A, ev, no_overlap=False)
    assert len(t2) == 1 and sk2["skip_reason"].tolist() == ["skip_pivot_done"]


def test_halted_order_day_skipped():
    rows = [FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6
    vol = np.full(len(rows), 1000.0)
    vol[3] = 0.0
    t, sk = _one(_book(rows, vol=vol), 3)
    assert t.empty and sk["skip_reason"].tolist() == ["skip_halt"]


def test_entry_check_matches_simulator():
    rows = [FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6
    A = _book(rows)
    c = V.buystop_entry_check(A, [0, 0, 0], [3, 3, 3], [100.0, 100.0, 90.0], [95.0, 91.0, 85.0])
    assert c["status"].tolist() == [V.S_OK, V.S_RISK, V.S_GAP]
    t, _ = _one(A, 3)
    assert c["entry"][0] == pytest.approx(t.iloc[0]["entry_px"]) and c["dd"][0] == pytest.approx(t.iloc[0]["d"])


def test_open_entry_control_uses_entry_as_pivot():
    rows = [FLAT, FLAT, FLAT, (100.0, 101.0, 99.0, 99.5), (99.0, 99.5, 98.8, 99.0)] + [UP] * 5
    A = _book(rows)
    r = V.run_open_entry(A, np.array([0]), np.array([3]), np.array([0.05]))
    assert r["status"][0] == V.S_OK
    assert V.REASONS[r["reason"][0]] == "failed_breakout"     # 종가 99.5 < 진입가 100 → 다음날 시가
    assert r["fin"][0] == 99.0


# --- 실적: 분기 단독 Q4 · 이중 시간축 ---

def _ea_rows():
    rows = []
    for y, base in ((2023, 10.0), (2024, 20.0)):
        for q in (1, 2, 3):
            rows.append({"code": "X", "period": f"{y}Q{q}", "avail_date": f"{y}-{[0, 5, 8, 11][q]:02d}-15",
                         "knowledge_date": f"{y}-{[0, 5, 8, 11][q]:02d}-15", "netinc": base + q,
                         "netinc_prior": base / 2 + q, "revenue": 100.0 * (y - 2022) + q,
                         "revenue_prior": 100.0 * (y - 2023) + q + 50})
        rows.append({"code": "X", "period": f"{y}Q4", "avail_date": f"{y + 1}-03-31",
                     "knowledge_date": f"{y + 1}-03-31", "netinc": 4 * base + 6 + 30.0,
                     "netinc_prior": 4 * base / 2 + 6 + 8.0, "revenue": 1000.0 * (y - 2022),
                     "revenue_prior": 1000.0 * (y - 2023) + 400})
    return pd.DataFrame(rows)


def test_q4_standalone_is_annual_minus_9m():
    st = V.standalone_quarter_states(_ea_rows())
    r = V.e_accel_asof(st, ["X"], ["2025-04-10"]).iloc[0]
    assert r["e_period"] == "2024Q4"
    # 2024 연간 = 4×20 + 6 + 30 = 116, 9M = 21+22+23 = 66 → Q4 단독 50.
    assert r["e_netinc"] == pytest.approx(116 - 66)
    # 전기: 연간 전기 = 40 + 6 + 8 = 54, 9M 전기 = 11+12+13 = 36 → 18 (당해 보고서 비교 수치끼리).
    assert r["e_netinc_prior"] == pytest.approx(54 - 36)
    assert r["e_yoy"] == pytest.approx((50 - 18) / 18)
    assert r["e_yoy_prev"] == pytest.approx((23 - 13) / 13)   # 직전 분기(2024Q3)
    rq, rp = 2000 - (201 + 202 + 203), 1400 - (151 + 152 + 153)
    assert r["e_rev_yoy"] == pytest.approx((rq - rp) / rp)
    assert bool(r["e_accel"]) == bool((rq - rp) / rp >= 0.10)


def test_e_pit_before_q4_and_restatement():
    ea = _ea_rows()
    st = V.standalone_quarter_states(ea)
    assert V.e_accel_asof(st, ["X"], ["2025-03-30"]).iloc[0]["e_period"] == "2024Q3"   # 사업보고서 전날
    # 정정: 2024Q2 순이익을 2025-06-01 에 고쳐 알게 된다 → 그 전 날짜의 Q4 단독은 옛 값, 그 뒤는 새 값.
    fix = ea[ea["period"] == "2024Q2"].copy()
    fix["knowledge_date"] = "2025-06-01"
    fix["netinc"] = 32.0
    st2 = V.standalone_quarter_states(pd.concat([ea, fix], ignore_index=True))
    q = V.e_accel_asof(st2, ["X", "X"], ["2025-04-10", "2025-06-02"])
    assert q.loc[0, "e_period"] == "2024Q4" and q.loc[0, "e_netinc"] == pytest.approx(50.0)
    assert q.loc[1, "e_period"] == "2024Q4" and q.loc[1, "e_netinc"] == pytest.approx(116 - (21 + 32 + 23))


def test_q4_unknown_without_all_three_quarters_no_rollback():
    ea = _ea_rows()
    ea = ea[ea["period"] != "2024Q2"]
    r = V.e_accel_asof(V.standalone_quarter_states(ea), ["X"], ["2025-04-10"]).iloc[0]
    assert r["e_period"] == "2024Q4"                # 앞 분기(2024Q3)로 밀리지 않는다
    assert not bool(r["e_known"]) and not bool(r["e_accel"])


def test_yoy_prev_requires_profitable_prior():
    ea = _ea_rows()
    ea.loc[ea["period"] == "2024Q3", "netinc_prior"] = -5.0     # 직전 분기 전년 동기 적자 → yoy_prev 무효
    r = V.e_accel_asof(V.standalone_quarter_states(ea), ["X"], ["2025-04-10"]).iloc[0]
    assert np.isnan(r["e_yoy_prev"]) and not bool(r["e_accel"])


def test_no_age_gate():
    """정정 M2: 140일 문 삭제 — 다음 분기가 안 보이면 오래된 최신 분기도 그대로 쓴다."""
    st = V.standalone_quarter_states(_ea_rows())
    a = V.e_accel_asof(st, ["X", "X"], ["2025-04-10", "2025-12-30"])
    assert a.loc[1, "e_period"] == "2024Q4" and a.loc[1, "e_accel"] == a.loc[0, "e_accel"]


# --- 수급 ---

def test_flow_base_sum_ratio_and_coverage():
    days = pd.bdate_range("2024-01-01", periods=10)
    f = pd.DataFrame({"X": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=days)
    i = pd.DataFrame({"X": [-1.0] * 10}, index=days)
    v = pd.DataFrame({"X": [10.0] * 10}, index=days)
    fc = V.FlowCum(f, i, v)
    # 기반 i0 = 2 → 행 3..6 (외국인 4,5,6,7 · 기관 −1×4): (22 − 4) / 40
    assert fc.f_base([0], [2], [6])[0] == pytest.approx((4 + 5 + 6 + 7 - 4) / 40)
    f2 = f.copy()
    f2.iloc[3:5] = np.nan                            # 4일 중 2일 모름 → 50% < 80% → NaN
    assert np.isnan(V.FlowCum(f2, i, v).f_base([0], [2], [6])[0])


def test_flow_base_trade_value_weighted():
    days = pd.bdate_range("2024-01-01", periods=6)
    f = pd.DataFrame({"X": [0.0, 0, 10, 0, 0, 0]}, index=days)
    i = pd.DataFrame({"X": [0.0, 0, 0, -1, 0, 0]}, index=days)
    v = pd.DataFrame({"X": [10.0, 10, 100, 10, 0, 10]}, index=days)   # 행 4 = 거래량 0 ∧ 순매수 0 → 알려진 0
    w = pd.DataFrame({"X": [1.0, 1, 1, 3, 0, 1]}, index=days)
    fc = V.FlowCum(f, i, v, weight=w)
    # 행 1..5: 하루 비율 0, 0.1, −0.1, 0, 0 · 가중 1,1,3,0,1 → (0.1 − 0.3)/6
    assert fc.f_base([0], [0], [5])[0] == pytest.approx((0.1 - 0.3) / 6)


def test_open_entry_exit_parity_with_minervini_v2():
    """시가 진입 경로(대조)는 기존 ``simulate_minervini`` 청산 v2(라이브: 돌파 실패·시간 손절 끔·고정 손절폭)와 트레이드 단위로 같다
    (매수·매도 미끄러짐 0 일 때 — 기존 시뮬레이터에는 미끄러짐이 없다)."""
    from research.signals.minervini_exits import simulate_minervini

    rng = np.random.default_rng(7)
    n, k = 400, 30
    days = pd.bdate_range("2020-01-01", periods=n)
    codes = [f"S{i}" for i in range(k)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.004, 0.025, (n, k)), axis=0)), index=days, columns=codes)
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, 0.01, (n, k)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.012, (n, k))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.012, (n, k))))
    vol = pd.DataFrame(1000.0, index=days, columns=codes)
    ma = c.rolling(50, min_periods=50).mean()
    A = V.Arrays(o, h, lo, c, ma, vol)
    sig = sorted({(int(rng.integers(55, 390)), j) for j in range(k) for _ in range(8)})
    dd = 0.06
    ev = pd.DataFrame({"code": [codes[j] for _, j in sig], "event_day": [days[s] for s, _ in sig]})
    ev["pivot"] = [o.iloc[s + 1, j] for s, j in sig]           # 돌파 실패 피벗 = 진입가(다음날 시가)
    ev["contraction_low"] = 1e-6
    ref = simulate_minervini(o, h, lo, c, ma, ev, volume=vol, fixed_stop=dd, chase=None, limit_up_skip=True,
                             no_overlap=False, failed_breakout=True, time_stop_days=None, exits_v2=True)
    r = V.run_open_entry(A, np.array([j for _, j in sig]), np.array([s + 1 for s, _ in sig]), np.full(len(sig), dd),
                         params={"sell_slip": 0.0, "slip": 0.0})
    ok = r["status"] == V.S_OK
    assert ok.sum() == len(ref)
    got = pd.DataFrame({"code": [codes[j] for _, j in sig], "event_day": [days[s] for s, _ in sig], "ret": r["ret"],
                        "exit_day": [days[x] if x >= 0 else pd.NaT for x in r["x"]],
                        "reason": [V.REASONS[q] if q >= 0 else "" for q in r["reason"]]})[ok]
    m = ref.merge(got, on=["code", "event_day"], suffixes=("", "_v"))
    assert len(m) == len(ref)
    np.testing.assert_allclose(m["ret"], m["ret_v"], rtol=1e-12, atol=1e-12)
    assert (m["exit_day"] == m["exit_day_v"]).all()
    rmap = {"stop_entry": "stop"}
    assert (m["exit_reason"] == m["reason"].map(lambda x: rmap.get(x, x))).all()
    assert {"stop", "breakeven", "trail", "failed_breakout", "data_end"} <= set(m["exit_reason"])   # 경로가 고루 나온다


# ---------------------------------------------------------------------------
# 정정 2 — 코드 리뷰 A·B 재현 테스트(scratch review_vcp_code/test_review_a.py·alt_fills.py, review_vcp_runner/ 에서 옮김)
# ---------------------------------------------------------------------------

def _rows_to_arr(H, L, C, Vv):
    return [np.ascontiguousarray(np.array(x, float)[None, :]) for x in (H, L, C, Vv)]


def _review_bars(n=262):
    """리뷰 A: 앞 240봉 완만한 오름(고가 < 100), i0 = 240 에 H0 = 100, 100 → 90 하락 뒤 바깥 봉(고가 95 · 저가 75), 그 뒤 박스."""
    c = np.linspace(40, 98, 240)
    H, L, C = list(c * 1.004), list(c * 0.996), list(c)

    def add(h, lo, cl):
        H.append(h)
        L.append(lo)
        C.append(cl)
    add(100.0, 98.0, 99.0)                       # 240 = i0 (H0)
    for k in range(5):                            # 241~245: 저가 94 → 90
        add(97.0 - k, 94.0 - k, 95.0 - k)
    add(95.0, 75.0, 92.0)                         # 246: 내림 상태 바깥 봉 — 고가 95(≥ 90×1.05) · 저가 75
    while len(C) < n:
        add(94.0, 91.5, 93.0)
    v = np.full(n, 1000.0)
    v[247:] = 400.0
    return _rows_to_arr(H, L, C, v)


def test_review_outside_bar_down_state_keeps_extreme():
    """A-C1: 내림 상태에서 한 봉이 +θ 반등과 새 저점을 함께 만들면 **극값 먼저** — 저가 75 가 수축에 들어가 c1 = 25%
    (옛 코드는 그 저가를 버려 c1 = 10% 로 VCP 통과)."""
    H, L, C, Vv = _review_bars()
    r = {k: v[0] for k, v in V.vcp_scan_arrays(H, L, C, Vv, np.array([0]), np.array([261])).items()}
    assert not (r["status"] == V.V_OK and abs(r["c1"] - 0.10) < 1e-9)
    assert r["c1"] == pytest.approx(0.25)


def test_review_outside_bar_up_state_extends_high_first():
    """오름 상태 바깥 봉(새 고점 ∧ 저가 ≤ 새 고점 × 0.95): 고점 갱신 먼저, 반전은 다음 봉부터 — 그 봉에서 고점 확정 안 함."""
    c = np.linspace(40, 98, 260)
    H, L, C = list(c * 1.004), list(c * 0.996), list(c)
    rows = [(100.0, 98.0, 99.0),                 # 260 고점 후보
            (101.0, 94.0, 95.0),                 # 261 바깥 봉: 새 고점 101 · 저가 94 ≤ 101×0.95 → 고점만 갱신(반전 안 봄)
            (96.0, 92.0, 93.0),                  # 262 저가 92 ≤ 95.95 → 고점 101(행 261) 확정, 저점 후보 92
            (97.0, 91.0, 96.0)]                  # 263 저점 후보 91
    for h, lo, cl in rows:
        H.append(h)
        L.append(lo)
        C.append(cl)
    Hh, Ll, Cc, Vv = _rows_to_arr(H, L, C, np.full(len(C), 1000.0))
    r = {k: v[0] for k, v in V.vcp_scan_arrays(Hh, Ll, Cc, Vv, np.array([0]), np.array([263]),
                                                 {"base_min": 1}).items()}
    # H0 = 바깥 봉 고점(행 261). 그 봉에서 바로 확정했다면(같은 봉 반전) 저점 후보가 94 였겠지만, 다음 봉에서 확정 → 확정 고점 1개
    # + 내림 상태 → T = 1 → contractions<2.
    assert r["i0"] == 261 and r["h0"] == 101.0
    assert r["status"] == V.V_FEW and r["n_contr"] == 1
    hv, hi_, lv = _ref_zz(Hh[0], Ll[0], 263, 261)
    assert hv == [101.0] and hi_ == [261] and lv == []


def test_review_pivot_bar_wide_range():
    """피벗 봉 고저폭 ≥ θ: 극값 먼저 가정에서 피벗 봉 자신의 저가는 L_last(피벗 다음날부터)에 들어가지 않는다."""
    OFFP = 140
    pts = [(0, 40.0)] + [(x + OFFP, y) for x, y in [(0, 50.0), (60, 90.0), (80, 100.0), (92, 76.0), (104, 96.0),
                                                     (114, 86.0), (124, 95.0), (132, 91.5), (139, 93.0), (140, 93.0)]]
    close = _path(pts, N)
    h, lo, c, v = _panel(close, _vol())
    lo.iloc[124 + OFFP] = 89.0                       # 피벗 봉 저가 89 — 고점 95.38 대비 6.7% 고저폭
    H_, L_, C_, V_ = (np.ascontiguousarray(x.to_numpy(float).T) for x in (h, lo, c, v))
    r = {k: v_[0] for k, v_ in V.vcp_scan_arrays(H_, L_, C_, V_, np.array([0]), np.array([S])).items()}
    assert r["ip"] == 124 + OFFP
    assert r["l_last"] == pytest.approx(91.5 * (1 - BAND))   # 피벗 봉 저가 89 는 L_last 밖
    assert r["status"] == V.V_OK


def test_lower_low_staircase_excluded():
    """정정 2 A-M3: 수축 저점이 1% 넘게 낮아지면(하락 계단) 셋업 없음. low_rise = 0 이면 통과."""
    pts = _shift([(0, 50.0), (60, 90.0), (80, 100.0), (92, 80.0), (104, 96.0), (114, 78.0), (124, 95.0), (132, 91.5),
                  (139, 93.0), (140, 93.0)])            # c1 20% → c2 ≈ 19% (저점 80 → 78: 2.5% 낮아짐)
    assert _scan(pts)["status"] == V.V_LOWER_LOW
    assert _scan(pts, low_rise=0.0)["status"] == V.V_OK


def test_review_x2_main_old_low_excludes_entry_day():
    """B-M2/A-M2: 주 X2 옛 저점 = min(low[d−10..d−1]) — 진입일 저가를 모른다. 문구판(d 포함)은 진입일 손절을 구조적으로 피한다(기술)."""
    from research.signals.minervini_exits import contraction_low_level

    rows = [FLAT] * 12 + [(99.0, 103.0, 93.0, 102.0)] + [UP] * 6   # 진입일 저가 93(장중 돌파 뒤 급락)
    A = _book(rows)
    low = pd.DataFrame({"X": [r[2] for r in rows]}, index=A.days)
    literal = float(contraction_low_level(low, 10).iloc[12, 0])      # min(low[d−10..d]) = 93
    main = float(low["X"].rolling(10).min().iloc[11])               # min(low[d−10..d−1]) = 99.5
    assert literal == 93.0 and main == 99.5
    ev = pd.DataFrame({"code": ["X"], "day_idx": [12], "pivot": [100.0], "l_last": [99.0], "ip": [0]})
    t_main, _ = V.simulate_buystop(A, ev.assign(old_low=main), old_stop=True)
    t_lit, _ = V.simulate_buystop(A, ev.assign(old_low=literal), old_stop=True)
    t0, _ = V.simulate_buystop(A, ev)
    assert bool(t0.iloc[0]["amb"]) and bool(t_main.iloc[0]["amb"])   # 주 X2 도 V0 와 같이 모호
    assert not bool(t_lit.iloc[0]["amb"]) and t_lit.iloc[0]["stop_px"] < 93.0


def test_review_conservative_path_inherits_optimistic_busy():
    """A MINOR 1: 보유 중 거르기는 낙관 경로 청산일 — 보수 경로라면 비었을 종목의 다음 셋업(다른 피벗)이 빠진다(기록)."""
    rows = [FLAT] * 3 + [(99.0, 103.0, 94.0, 102.0)] + [UP] * 3 + [(103.0, 110.0, 102.5, 109.0)] + [UP] * 4
    A = _book(rows)
    ev = pd.DataFrame({"code": ["X", "X"], "day_idx": [3, 7], "pivot": [100.0, 105.0], "l_last": [96.0, 101.0],
                       "ip": [0, 5]})
    t, sk = V.simulate_buystop(A, ev)
    assert bool(t.iloc[0]["amb"]) and len(t) == 1 and sk["skip_reason"].tolist() == ["skip_overlap"]


def test_entry_day_close_below_stop_is_confirmed_stop():
    """정정 2 B-M1: 진입일 종가 ≤ 손절가 = 확정 손절(모호 아님) — 두 경로 모두 손절가 × 0.999."""
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 93.0, 93.8)] + [UP] * 6)
    t, _ = _one(A, 3)
    r = t.iloc[0]
    assert not bool(r["amb"])
    assert r["exit_reason"] == "stop_entry" and r["exit_px"] == pytest.approx(94.0 * SL)
    assert r["ret_stop0"] == pytest.approx(r["ret"])
    c = V.buystop_entry_check(A, [0], [3], [100.0], [95.0])
    assert bool(c["stop0_confirmed"][0]) and not bool(c["amb"][0])
    B = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 93.5, 102.0)] + [UP] * 6)
    cb = V.buystop_entry_check(B, [0], [3], [100.0], [95.0])
    assert bool(cb["amb"][0]) and not bool(cb["stop0_confirmed"][0])


def test_nxt_cut_truncates_to_data_end():
    """정정 2 B-M3: 미보정 종목은 절단일 전 마지막 거래일 종가에서 data_end, 절단일 이후 주문은 skip_nxt_cut."""
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    assert A.set_cut(["X"], A.days[7]) == 1
    t, _ = _one(A, 3)
    r = t.iloc[0]
    assert r["exit_reason"] == "data_end" and r["exit_day"] == A.days[6] and r["exit_px"] == UP[3]
    t2, sk2 = _one(A, 7)
    assert t2.empty and sk2["skip_reason"].tolist() == ["skip_nxt_cut"]
    c = V.run_open_entry(A, np.array([0]), np.array([4]), np.array([0.05]))
    assert V.REASONS[c["reason"][0]] == "data_end" and c["x"][0] == 6


def test_control_buy_slippage_symmetric():
    """정정 2: 대조 매수에도 +0.1%."""
    A = _book([FLAT, FLAT, FLAT, (100.0, 101.0, 99.0, 100.5)] + [UP] * 6)
    r = V.run_open_entry(A, np.array([0]), np.array([3]), np.array([0.05]))
    assert r["entry"][0] == pytest.approx(100.0 * 1.001)


# 리뷰 A alt_fills.py / analyze2.py 의 독립 참조 구현(순수 파이썬, 극값 먼저 "ef") — 커널과 셋업 판정이 같아야 한다.
def _ref_zz(H, L, t, i0, theta=0.05):
    hv, hi_, lv = [], [], []
    up, hc, ihc, lc = True, H[i0], i0, 0.0
    for j in range(i0 + 1, t + 1):
        hj, lj = H[j], L[j]
        if up:
            if hj >= hc:
                hc, ihc = hj, j
            elif lj <= hc * (1 - theta):
                hv.append(hc)
                hi_.append(ihc)
                up, lc = False, lj
        else:
            if lj <= lc:
                lc = lj
            elif hj >= lc * (1 + theta):
                lv.append(lc)
                up, hc, ihc = True, hj, j
    if up and len(lv) == len(hv) and hv:
        hv.append(hc)
        hi_.append(ihc)
    return hv, hi_, lv


def _ref_ev(H, L, C, Vv, t, P=V.VCP_PARAMS):
    s0 = t - P["win"] + 1
    w = H[s0:t + 1]
    if s0 < 0 or not np.isfinite(w).all():
        return "no_data", None
    i0 = s0 + (len(w) - 1 - int(np.argmax(w[::-1])))
    if not (P["base_min"] <= t - i0 <= P["base_max"]):
        return "base_len", None
    hv, hi_, lv = _ref_zz(H, L, t, i0)
    T = len(hv)
    if T < 2:
        return "contractions<2", None
    if T > 6:
        return "contractions>6", None
    ip, piv = hi_[-1], hv[-1]
    if t - ip < 3:
        return "pivot<3d", None
    ll = L[ip + 1:t + 1].min()
    cs = [(hv[k] - lv[k]) / hv[k] for k in range(T - 1)] + [(piv - ll) / piv]
    if not (0.10 <= cs[0] <= 0.40):
        return "c1_range", None
    if any(not cs[k] < cs[k - 1] for k in range(1, T)):
        return "not_decreasing", None
    lows = list(lv[:T - 1]) + [ll]
    if any(lows[k] < lows[k - 1] * 0.99 for k in range(1, T)):
        return "lower_low", None
    if cs[-1] > 0.10:
        return "last>10%", None
    if piv < 0.85 * H[i0]:
        return "pivot<0.85H0", None
    if not (C[t] < piv and C[t] >= 0.9 * piv):
        return "close_range", None
    if ip - 49 < 0:
        return "volume_dryup", None
    vb, vs = Vv[ip - 49:ip + 1], Vv[ip + 1:t + 1]
    if not np.isfinite(vb).all() or vb.sum() <= 0 or not np.isfinite(vs).all() or vs.mean() / vb.mean() > 0.8:
        return "volume_dryup", None
    return "ok", (piv, ll, ip)


def test_review_reference_zigzag_parity_random_paths():
    """리뷰 A 의 독립 참조 구현(ef)과 커널의 셋업 판정·피벗·L_last 가 랜덤 경로 전 행에서 같다."""
    rng = np.random.default_rng(11)
    n, k = 420, 12
    statuses = set()
    for _ in range(3):
        c = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.022, (n, k)), axis=0))
        h = c * (1 + np.abs(rng.normal(0, 0.012, (n, k))))
        lo = c * (1 - np.abs(rng.normal(0, 0.012, (n, k))))
        vol = rng.lognormal(7, 0.6, (n, k))
        H, L, C, Vv = (np.ascontiguousarray(x.T) for x in (h, lo, c, vol))
        ts = np.arange(259, n)
        for ci in range(k):
            r = V.vcp_scan_arrays(H, L, C, Vv, np.full(len(ts), ci), ts)
            for i, t in enumerate(ts):
                st, info = _ref_ev(H[ci], L[ci], C[ci], Vv[ci], int(t))
                got = V.VCP_STATUS[r["status"][i]]
                statuses.add(st)
                assert got == st, (ci, t, got, st)
                if st == "ok":
                    assert r["pivot"][i] == pytest.approx(info[0]) and r["l_last"][i] == pytest.approx(info[1])
                    assert r["ip"][i] == info[2]
    assert len(statuses) >= 6                        # 여러 탈락 경로를 실제로 지난다


# --- 러너 순수 헬퍼(정정 2) ---

def _runner():
    import sys
    sys.path.insert(0, "research/experiments")
    import vcp_retest_run as R
    return R


def test_split_flags_excludes_batch_days():
    """리뷰 B MINOR 1: 같은 날 ≥ 5건 조정계수 변화는 수집 배치 경계 — 분할 제외에서 뺀다. 개별 변화는 60 거래일 제외."""
    R = _runner()
    days = pd.bdate_range("2024-01-01", periods=100)
    fac = pd.DataFrame(1.0, index=days, columns=[f"S{i}" for i in range(6)])
    fac.iloc[10:, :] = 2.0                         # 행 10: 6종목 동시 변화 = 배치
    fac.iloc[40:, 0] = 4.0                         # 행 40: S0 개별 변화
    f = R.split_flags(fac, excl=60)
    assert f["n_events"] == 7 and f["n_real"] == 1
    assert list(f["batch_days"]) == [str(days[10].date())]
    assert not f["recent"][10:40].any()
    assert f["recent"][40:100, 0].all() and not f["recent"][40:, 1:].any()
    assert f["cum"][0, 41] - f["cum"][0, 40] == 1   # 행 40 까지 누적 = cum[41]


def test_floor_rule_and_funnel_line(tmp_path):
    R = _runner()
    counts = {5000.0: {"V0": 55, "V0_sub": 40}, 3000.0: {"V0": 70, "V0_sub": 31}, 2000.0: {"V0": 90, "V0_sub": 60},
              1000.0: {"V0": 120, "V0_sub": 80}}
    assert R.floor_rule(counts) == (3000.0, False)            # 조건을 만족하는 가장 높은 하한
    assert R.floor_rule({5000.0: {"V0": 10, "V0_sub": 5}}) == (None, True)
    pth = tmp_path / "FUNNEL.md"
    pth.write_text("x\n<!-- adv_floor_rule: 3000 · v0_void: false -->\n", encoding="utf-8")
    assert R.read_floor_rule(str(pth)) == (3000.0, False)
    pth.write_text("<!-- adv_floor_rule: none · v0_void: true -->", encoding="utf-8")
    assert R.read_floor_rule(str(pth)) == (None, True)
    assert R.read_floor_rule(str(tmp_path / "없음.md")) == (None, None)


def test_expand_mixture_weights_and_complete_rows():
    """모호 거래는 p·1−p 두 행(가중 합 1), 보수 행은 완결 — '완결 거래만' 은 행 단위로 낙관 data_end 만 뺀다(리뷰 B MINOR 8)."""
    R = _runner()
    t = pd.DataFrame({"net": [0.05, 0.02], "net2": [0.04, 0.01], "ret": [0.06, 0.03], "d": [0.05, 0.05],
                      "exit_reason": ["trail", "data_end"], "hold_days": [10, 5],
                      "order_day": pd.to_datetime(["2020-01-02", "2026-08-03"]),
                      "entry_day": pd.to_datetime(["2020-01-02", "2026-08-03"]), "amb": [False, True],
                      "net_stop0": [0.05, -0.06], "net2_stop0": [0.04, -0.07], "ret_stop0": [0.06, -0.05]})
    x = R.expand(t, 0.4)
    assert len(x) == 3 and x["w"].sum() == pytest.approx(2.0)
    s = R.wstats(x)
    assert s["net_mean"] == pytest.approx((0.05 + 0.6 * 0.02 + 0.4 * -0.06) / 2)
    xc = x[x["complete"].to_numpy(bool)]
    assert set(xc["exit_reason"]) == {"trail", "stop_entry"}
