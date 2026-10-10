"""천장 아래 하락 쐐기(``research/signals/wedge.py``) — 손으로 확인한 합성 경로.

사전등록 ``research/logs/sepa_explore/WEDGE_TEST.md`` (7196cad + 정정 0·0b·1): 사용자 그림 H0 → A → B → C → D → E(상단선 B–D, 하단선
C–E, A 닿음 ±3%), 진행 중 E, 첫 돌파만, 진짜 돌파 전(종가 < U(d)), 무결성, 반전점 간격, W-nc, 상승 쐐기·대칭 삼각·평평한 천장 실패,
룩어헤드 없음, 상단선 연장 값, 매수정지 체결(주문가 기준 3% 갭·9.5%), 돌파 실패 이틀 vs 하루, 본전 이동, 하락 기준선 널.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.signals import vcp as V
from research.signals import wedge as W

BAND = 0.004
N = 289
T = 287                                     # 신호 행 s, 주문일 d = 288
# 그림: 0~220 오름 → H0 100 (행 220) → A 82 (232) → B 93 (244) → C 80.5 (256) → D 88 (268) → E 79.5 (280) → 반등 83.5 (287)
FIG = [(0, 40.0), (220, 100.0), (232, 82.0), (244, 93.0), (256, 80.5), (268, 88.0), (280, 79.5), (287, 83.5), (288, 83.5)]


def _path(points, n=N):
    xs, ys = zip(*points, strict=True)
    return np.interp(np.arange(n), xs, ys)


def _arrs(points, vol=None, n=N):
    c = _path(points, n)
    v = np.full(n, 1000.0) if vol is None else vol
    if vol is None:
        v[269:] = 500.0
    H, L, C, Vv = (np.ascontiguousarray(x[None, :]) for x in (c * (1 + BAND), c * (1 - BAND), c, v))
    return H, L, C, Vv


def _scan(points, t=T, **kw):
    H, L, C, Vv = _arrs(points)
    r = W.wedge_scan_arrays(H, L, C, Vv, np.array([0]), np.array([t]), kw or None)
    return {k: v[0] for k, v in r.items()}


def test_user_figure_falling_wedge_passes():
    r = _scan(FIG)
    assert r["status"] == W.W_OK, W.WEDGE_STATUS[r["status"]]
    assert r["i0"] == 220 and r["h0"] == pytest.approx(100 * (1 + BAND))
    assert r["n_highs"] == 3 and r["n_contr"] == 2                  # H0·B·D, 쐐기 안 수축 B→C·D→E
    B, C_, D, E = 93 * (1 + BAND), 80.5 * (1 - BAND), 88 * (1 + BAND), 79.5 * (1 - BAND)
    assert r["c1"] == pytest.approx((B - C_) / B)                   # 쐐기 안 첫 수축(H0 → A 아님)
    assert r["c_last"] == pytest.approx((D - E) / D)
    su = math.log(D / B) / (268 - 244)
    sl = math.log(E / C_) / (280 - 256)
    assert r["slope_u"] == pytest.approx(su) and r["slope_l"] == pytest.approx(sl)
    assert r["u_s"] == pytest.approx(D * math.exp(su * (T - 268)))
    assert r["u_d"] == pytest.approx(D * math.exp(su * (T + 1 - 268)))   # 상단선을 d 로 연장
    assert r["i_low"] == 280 and r["i_high"] == 268
    assert r["floor_low"] == pytest.approx(E)                        # 손절 기준 = E 이후 최저 저가
    assert r["depth"] == pytest.approx((100.4 - E) / 100.4)


def test_pre_wedge_reversals_unconstrained():
    """정정 2: 쐐기는 마지막 다섯 점 A~E 만 — H0 → A 사이의 반전점(여기선 B 보다 낮은 고점 90 과 A 보다 높은 저점 84)은 제약이 없다.
    옛 정의(H0 뒤 모든 고점 단조 하락)라면 90 → B 93 상승으로 실패했을 경로."""
    pts = [(0, 40.0), (190, 100.0), (202, 84.0), (214, 90.0), (232, 82.0), (244, 93.0), (256, 80.5), (268, 88.0),
           (280, 79.5), (287, 83.5), (288, 83.5)]
    r = _scan(pts)
    assert r["status"] == W.W_OK, W.WEDGE_STATUS[r["status"]]
    assert r["i0"] == 190 and r["n_highs"] == 4 and r["i_a"] == 232 and r["i_high"] == 268 and r["n_contr"] == 2
    H, L, C, Vv = _arrs(pts)
    kinds = [p["kind"] for p in W.wedge_points(H, L, 0, T)["points"]]
    assert kinds == ["H0", "l", "h", "A", "B", "C", "D", "E"]


def test_rising_wedge_mirror_fails():
    """사용자 그림의 상승 쐐기(고점·저점이 함께 오르며 좁아짐) — 천장 아래여도 하락 쐐기가 아니다."""
    pts = [(0, 40.0), (220, 100.0), (232, 80.0), (244, 86.0), (256, 82.0), (268, 89.0), (280, 85.0), (287, 89.5),
           (288, 89.5)]
    r = _scan(pts)
    assert r["status"] in (W.W_HIGHS, W.W_LOWS) and r["status"] != W.W_OK


def test_symmetric_triangle_fails():
    pts = [(0, 40.0), (220, 100.0), (232, 80.0), (244, 93.0), (256, 82.0), (268, 88.0), (280, 84.0), (287, 87.5),
           (288, 87.5)]                                              # 고점 하락 · 저점 상승
    assert _scan(pts)["status"] == W.W_LOWS


def test_flat_top_fails():
    pts = [(0, 40.0), (220, 100.0), (232, 82.0), (244, 92.0), (256, 80.5), (268, 91.8), (280, 79.5), (287, 84.0),
           (288, 84.0)]                                              # D 91.8 > B 92 × 0.99
    assert _scan(pts)["status"] == W.W_HIGHS


def test_rising_lows_fail():
    pts = [(0, 40.0), (220, 100.0), (232, 80.0), (244, 93.0), (256, 80.5), (268, 88.0), (280, 81.0), (287, 85.5),
           (288, 85.5)]                                              # C 80.5 > A 80 (저점 상승)
    assert _scan(pts)["status"] == W.W_LOWS


def test_first_low_a_must_touch_lower_line():
    """정정 1 M2: |A / 하단선(A) − 1| ≤ 3% — 하단선 C–E 연장선에서 3% 넘게 떨어진 A 는 실패(하단선 3번 닿음 아님)."""
    base = [(0, 40.0), (220, 100.0), None, (244, 95.0), (256, 84.0), (268, 89.0), (280, 80.0), (287, 84.0), (288, 84.0)]
    ok = _scan([p if p is not None else (232, 87.5) for p in base])
    bad = _scan([p if p is not None else (232, 85.0) for p in base])
    assert ok["status"] == W.W_OK, W.WEDGE_STATUS[ok["status"]]
    assert bad["status"] == W.W_A_TOUCH
    far_pts = [(0, 40.0), (220, 100.0), (232, 91.0), (244, 95.5), (256, 84.0), (268, 89.0), (280, 80.0), (287, 84.0),
               (288, 84.0)]                                          # A 가 하단선 연장 위 ≈ +3.2% — 양쪽 조건이라 실패
    far = _scan(far_pts)
    assert far["status"] == W.W_A_TOUCH and far["a_ratio"] > 1.03


def test_too_few_lows_fails():
    pts = [(0, 40.0), (220, 100.0), (232, 82.0), (244, 93.0), (256, 80.5), (263, 84.5), (264, 84.5)]   # A, C 만
    r = _scan(pts, t=263)
    assert r["status"] == W.W_FEW                                    # 확정 고점 H0·B 뿐


def test_close_at_or_above_u_d_fails_m1():
    """정정 1 M1: 종가(s) < U(d) — 선이 하루 내려와 종가 아래로 들어온 '가짜 돌파'(주문가 ≤ 전일 종가)는 셋업이 아니다."""
    pts = FIG[:-2] + [(287, 84.45), (288, 84.45)]                    # U(d) ≈ 84.38 ≤ 종가 84.45 < U(s) ≈ 84.57
    assert _scan(pts)["status"] == W.W_CLOSE


def test_in_progress_e_allowed():
    """정정 1 C2: E 는 D 뒤 최저 저가 — 5% 반등으로 확정되기 전에도 셋업(확정 여부는 기술)."""
    pts = FIG[:-2] + [(287, 81.5), (288, 81.5)]                      # 79.18 × 1.05 = 83.1 > 고가 81.8 → E 미확정
    r = _scan(pts)
    assert r["status"] == W.W_OK, W.WEDGE_STATUS[r["status"]]
    assert r["e_confirmed"] == 0 and r["i_low"] == 280
    assert _scan(FIG)["e_confirmed"] == 1


def test_first_breakout_only():
    """정정 1 C2: E 뒤 고가가 상단선 × 1.005 를 이미 넘은 날이 있으면 그 쐐기는 끝(재돌파 안 삼)."""
    H, L, C, Vv = _arrs(FIG)
    H[0, 284] = 86.0                                                 # U(284) × 1.005 ≈ 85.6 < 86
    r = W.wedge_one(H, L, C, Vv, 0, T)
    assert r["status"] == W.W_BROKEN


def test_integrity_close_below_lower_line():
    H, L, C, Vv = _arrs(FIG)
    C[0, 250] = 79.0                                                 # B → C 구간 종가가 하단선 × 0.995 아래
    assert W.wedge_one(H, L, C, Vv, 0, T)["status"] == W.W_INTEGRITY


def test_spacing_rule():
    assert _scan(FIG, gap_min=13)["status"] == W.W_SPACING          # 반전점 간격 12 < 13
    assert _scan(FIG, a_span=56)["status"] == W.W_SPACING           # A → s = 55 < 56


def test_wnc_turns_off_convergence_only():
    """W-nc: c_k 감소·기울기 순서를 끈 하락 계단 — 수렴하지 않는 모양은 W 에서 실패, W-nc 에서 통과."""
    pts = [(0, 40.0), (220, 100.0), (232, 88.5), (244, 93.0), (256, 82.0), (268, 88.0), (280, 75.5), (287, 80.0),
           (288, 80.0)]                                              # c1 ≈ 12.5% < c2 ≈ 14.9%, 하단선이 더 가파름
    assert _scan(pts)["status"] == W.W_NOT_DECR
    r = _scan(pts, conv=False)
    assert r["status"] == W.W_OK, W.WEDGE_STATUS[r["status"]]


def test_null_descending_line():
    """정정 1 M5 널: 직전 k 일 창 최고 고가에서 지그재그 → 마지막 두 확정 고점 직선(하락) — 그림 경로에선 B–D 상단선과 같다."""
    H, L, C, Vv = _arrs(FIG)
    days = pd.bdate_range("2020-01-01", periods=N)
    mk = lambda a: pd.DataFrame({"X": a[0]}, index=days)  # noqa: E731
    A = V.Arrays(mk(C), mk(H), mk(L), mk(C), mk(np.full((1, N), np.nan)), mk(Vv))
    o = W.null_lines(A, [0], [T], [60], [5], parallel=False)
    assert bool(o["ok"][0]) and o["u_d"][0] == pytest.approx(_scan(FIG)["u_d"]) and o["pid"][0] == 268
    assert o["l_last"][0] == pytest.approx(L[0, T - 4:T + 1].min())


def test_no_lookahead_bar_d():
    H, L, C, Vv = _arrs(FIG)
    a = W.wedge_scan_arrays(H, L, C, Vv, np.array([0]), np.array([T]))
    H2, L2, C2, V2 = (x.copy() for x in (H, L, C, Vv))
    H2[0, T + 1], L2[0, T + 1], C2[0, T + 1], V2[0, T + 1] = 500.0, 1.0, 300.0, 1e9
    b = W.wedge_scan_arrays(H2, L2, C2, V2, np.array([0]), np.array([T]))
    for k in W.OUT_KEYS:
        np.testing.assert_array_equal(a[k], b[k])
    C3 = C.copy()
    C3[0, T] = 99.0                                                  # d−1 종가가 상단선 위 → 셋업 깨짐
    assert W.wedge_scan_arrays(H, L, C3, Vv, np.array([0]), np.array([T]))["status"][0] != W.W_OK


def test_wedge_points_for_chart():
    H, L, C, Vv = _arrs(FIG)
    pt = W.wedge_points(H, L, 0, T)
    kinds = [p["kind"] for p in pt["points"]]
    assert kinds == ["H0", "A", "B", "C", "D", "E"] and pt["i0"] == 220
    assert [p["row"] for p in pt["points"]] == [220, 232, 244, 256, 268, 280]
    r = _scan(FIG)
    assert pt["upper"]["from"]["row"] == 244 and pt["upper"]["to"]["price"] == pytest.approx(r["u_d"])
    assert pt["lower"]["from"]["row"] == 232 and pt["lower"]["to"]["price"] == pytest.approx(r["low_d"])
    assert pt["u_d"] == pytest.approx(r["u_d"])


# --- 매수정지 · 청산 ---

def _book(rows):
    days = pd.bdate_range("2024-01-01", periods=len(rows))
    a = np.array(rows, float)
    df = {k: pd.DataFrame({"X": a[:, i]}, index=days) for i, k in enumerate("ohlc")}
    ma = pd.DataFrame({"X": np.full(len(rows), np.nan)}, index=days)
    v = pd.DataFrame({"X": np.full(len(rows), 1000.0)}, index=days)
    return V.Arrays(df["o"], df["h"], df["l"], df["c"], ma, v)


FLAT = (95.0, 96.0, 94.5, 95.5)
UP = (103.0, 104.0, 102.5, 103.5)


def _sim(A, u_d=100.0, floor=94.0, fb_days=2, d=3):
    S = pd.DataFrame({"code": ["X"], "day_idx": [d], "u_d": [u_d], "floor_low": [floor], "i0": [0], "i_high": [1]})
    return W.simulate_wedge(A, S, fb_days=fb_days)


def test_buystop_fill_order_and_gap_reference():
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    t, _ = _sim(A)
    r = t.iloc[0]
    assert r["order_px"] == 101.0 and r["fill_px"] == 101.0          # U(d) 100 + 1호가, 시가 99 < 주문가
    assert r["entry_px"] == pytest.approx(101.101)
    assert r["stop_px"] == 93.0                                      # 바닥 94 − 1호가
    B = _book([FLAT, FLAT, FLAT, (104.2, 106.0, 104.0, 105.0)] + [UP] * 6)
    tb, sb = _sim(B)                                                 # 시가 104.2 > 주문가 101 × 1.03 = 104.03 → 버림
    assert tb.empty and sb["skip_reason"].tolist() == ["skip_gap>3%"]
    Cc = _book([FLAT, FLAT, FLAT, (103.9, 106.0, 103.5, 105.0)] + [UP] * 6)
    tc, _ = _sim(Cc)                                                 # 104.03 이하 → 시가 체결
    assert tc.iloc[0]["fill_px"] == 103.9


def test_risk_pre_filter_uses_order_price():
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    t, sk = _sim(A, floor=92.0)                                      # (101 − 91)/101 = 9.9% > 9.5%
    assert t.empty and sk["skip_reason"].tolist() == ["skip_risk>9.5%"]
    t2, _ = _sim(A, floor=92.5)                                      # (101 − 91.5)/101 = 9.4%
    assert len(t2) == 1


def test_two_day_vs_one_day_failed_breakout():
    """진입일 종가 < U(d) 하루 뒤 회복: 하루 규칙은 다음날 시가에 팔고, 이틀 규칙(주)은 버틴다. 이틀 연속이면 이틀 규칙도 판다."""
    rows = [FLAT, FLAT, FLAT, (99.0, 102.0, 98.0, 99.5), (100.0, 104.0, 99.8, 103.0)] + [UP] * 5
    A = _book(rows)
    t1, _ = _sim(A, fb_days=1)
    t2, _ = _sim(A, fb_days=2)
    assert t1.iloc[0]["exit_reason"] == "failed_breakout" and t1.iloc[0]["exit_day"] == A.days[4]
    assert t2.iloc[0]["exit_reason"] == "data_end"
    rows2 = [FLAT, FLAT, FLAT, (99.0, 102.0, 98.0, 99.5), (99.6, 100.5, 99.0, 99.2), (98.5, 99.0, 98.0, 98.8)]
    rows2 += [UP] * 4
    B = _book(rows2)
    t3, _ = _sim(B, fb_days=2)
    assert t3.iloc[0]["exit_reason"] == "failed_breakout" and t3.iloc[0]["exit_day"] == B.days[5]
    assert t3.iloc[0]["exit_px"] == 98.5


def test_same_wedge_one_fill():
    A = _book([FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0)] + [UP] * 6)
    S = pd.DataFrame({"code": ["X", "X"], "day_idx": [3, 4], "u_d": [100.0, 99.8], "floor_low": [94.0, 94.0],
                      "i0": [0, 0], "i_high": [1, 1]})
    t, sk = W.simulate_wedge(A, S, no_overlap=False)
    assert len(t) == 1 and sk["skip_reason"].tolist() == ["skip_pivot_done"]
    assert W.wedge_id([0], [1])[0] == 1


def test_numba_python_same_on_random_paths():
    """커널은 파이썬 경로와 같은 결과(numba 가 없으면 같은 함수 — 이때는 자명). 랜덤 경로 전 행 셋업 판정이 참조와 같다."""
    rng = np.random.default_rng(5)
    n, k = 420, 8
    c = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n, k)), axis=0))
    h = c * (1 + np.abs(rng.normal(0, 0.012, (n, k))))
    lo = c * (1 - np.abs(rng.normal(0, 0.012, (n, k))))
    vol = rng.lognormal(7, 0.6, (n, k))
    H, L, C, Vv = (np.ascontiguousarray(x.T) for x in (h, lo, c, vol))
    ts = np.arange(259, n)
    seen = set()
    for ci in range(k):
        r = W.wedge_scan_arrays(H, L, C, Vv, np.full(len(ts), ci), ts)
        for i, t in enumerate(ts):
            ref = W.wedge_one(H, L, C, Vv, ci, int(t))
            for kk in W.OUT_KEYS:
                a, b = r[kk][i], ref[kk]
                assert (a == b) or (np.isnan(a) and np.isnan(b)), (kk, a, b)
            seen.add(int(ref["status"]))
    assert len(seen) >= 4


def test_be1_moves_stop_to_entry_after_plus_1r():
    """정정 0b W-be1: 고가 ≥ 진입가 + 1R 인 날 다음 거래일부터 손절선 = 진입가 × 1.005 (청산 v2 be_r = 1)."""
    rows = [FLAT, FLAT, FLAT, (99.0, 103.0, 98.0, 102.0), (104.0, 110.0, 103.5, 108.0), (104.0, 105.0, 101.0, 102.0)]
    rows += [UP] * 4
    A = _book(rows)
    S = pd.DataFrame({"code": ["X"], "day_idx": [3], "u_d": [100.0], "floor_low": [94.0], "i0": [0], "i_high": [1]})
    t0, _ = W.simulate_wedge(A, S)
    t1, _ = W.simulate_wedge(A, S, params={"be_r": 1.0})
    e = 101.101                                                       # +1R = 101.101 + (101.101 − 93) = 109.2 ≤ 고가 110
    assert t0.iloc[0]["exit_reason"] == "data_end"
    r = t1.iloc[0]
    assert r["exit_reason"] == "breakeven" and r["exit_day"] == A.days[5]
    assert r["exit_px"] == pytest.approx(e * 1.005 * 0.999)


def test_be_r_parity_with_minervini_v2():
    """be_r 경로는 기존 ``simulate_minervini`` 청산 v2(be_r = 1, 돌파 실패, 시간 손절 끔)와 트레이드 단위로 같다(미끄러짐 0)."""
    from research.signals.minervini_exits import simulate_minervini

    rng = np.random.default_rng(3)
    n, k = 400, 25
    days = pd.bdate_range("2020-01-01", periods=n)
    codes = [f"S{i}" for i in range(k)]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.003, 0.025, (n, k)), axis=0)), index=days, columns=codes)
    o = c.shift(1).fillna(c) * (1 + rng.normal(0, 0.01, (n, k)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.012, (n, k))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.012, (n, k))))
    vol = pd.DataFrame(1000.0, index=days, columns=codes)
    ma = c.rolling(50, min_periods=50).mean()
    A = V.Arrays(o, h, lo, c, ma, vol)
    sig = sorted({(int(rng.integers(55, 390)), j) for j in range(k) for _ in range(8)})
    dd = 0.06
    ev = pd.DataFrame({"code": [codes[j] for _, j in sig], "event_day": [days[s] for s, _ in sig]})
    ev["pivot"] = [o.iloc[s + 1, j] for s, j in sig]
    ev["contraction_low"] = 1e-6
    ref = simulate_minervini(o, h, lo, c, ma, ev, volume=vol, fixed_stop=dd, chase=None, no_overlap=False,
                             failed_breakout=True, time_stop_days=None, be_r=1.0)
    r = V.run_open_entry(A, np.array([j for _, j in sig]), np.array([s + 1 for s, _ in sig]), np.full(len(sig), dd),
                         params={"sell_slip": 0.0, "slip": 0.0, "be_r": 1.0})
    ok = r["status"] == V.S_OK
    got = pd.DataFrame({"code": [codes[j] for _, j in sig], "event_day": [days[s] for s, _ in sig], "ret": r["ret"],
                        "reason": [V.REASONS[q] if q >= 0 else "" for q in r["reason"]]})[ok]
    m = ref.merge(got, on=["code", "event_day"], suffixes=("", "_v"))
    assert len(m) == len(ref) == ok.sum()
    np.testing.assert_allclose(m["ret"], m["ret_v"], rtol=1e-12, atol=1e-12)
    assert (m["reason"] == "breakeven").sum() > 0


# --- 러너 순수 헬퍼 ---

def _runner():
    import sys
    sys.path.insert(0, "research/experiments")
    import wedge_test_run as WR
    return WR


def test_wedge_bar5_breakeven_win_margin():
    """바 5: 손익비 2.5 면 손익분기 승률 1/3.5 = 28.6% + 5%p = 33.6% — 본전 제외 승률 33% 는 미달, 34% 는 통과(다른 조건 충족 시)."""
    WR = _runner()
    # 승률(본전 제외) = 승 / (승 + 패): 0.30 / (0.30 + 0.61) ≈ 33.0% 미달, 0.31 / 0.91 ≈ 34.1% 통과
    s = {"n": 100, "n_sub": 50, "gl_ratio": 2.5, "avg_loss_R": -0.6, "r_p5": -1.0, "big_loss": 0.01, "win": 0.30,
         "lossc": 0.61, "be": 0.09, "net2_mean": 0.01, "net2_sub": 0.01, "net2_ex2021": 0.01}
    b = WR.wedge_bars(s, 0.0, 0.001, np.nan)
    assert b["be_need"] == pytest.approx(1 / 3.5 + 0.05) and b["win_ex_be"] == pytest.approx(0.30 / 0.91)
    assert not b["b5"]
    assert WR.wedge_bars({**s, "win": 0.31, "lossc": 0.60}, 0.0, 0.001, np.nan)["b5"]
    assert not WR.wedge_bars({**s, "net2_ex2021": -0.001}, 0.0, 0.001, np.nan)["b2"]   # 2020~21 제외도 > 0
    assert WR.wedge_bars(s, 0.0, 0.001, np.nan)["b1"] and not WR.wedge_bars({**s, "n_sub": 39}, 0.0, 0.0, np.nan)["b1"]


def test_breakout_volume_buckets():
    WR = _runner()
    assert WR.vol_bucket([0.5, 1.0, 1.2, 1.5, 3.0, np.nan]).tolist() == [0, 1, 1, 2, 2, -1]
