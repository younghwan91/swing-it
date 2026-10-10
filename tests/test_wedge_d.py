"""쐐기 D(``research/signals/wedge.py`` WD 부분) — 손으로 확인한 합성 경로.

사전등록 ``research/logs/sepa_explore/WEDGE_D_TEST.md`` (2154f77 + 정정 1 b008fce, 탐색): A = H0, 네 점 A·B·C·D 연속, 상단선 A–C·하단선
B–D, D ≤ B × 0.99, 바닥 진입 = D 봉 고가 + 1호가 매수정지(d − 행(D) ≤ 5, 종가(s) < 주문가, 두 번째 관통 차단), 손절 = D − 1호가(체결 =
손절가 − 1호가), 2R 전 돌파 실패 규칙 없음, WD-nc·WD-up, 되돌림 새 저점 널.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from research.signals import vcp as V
from research.signals import wedge as Wd

BAND = 0.004
N = 270
T = 264                                       # 신호 행 s, 주문일 d = 265 → d − 행(D) = 3
# 0~220 오름 → A = H0 100 (220) → B 85 (235) → C 93 (250) → D 82 (262) → 82.8 (264)
FIG = [(0, 40.0), (220, 100.0), (235, 85.0), (250, 93.0), (262, 82.0), (264, 82.8), (269, 82.8)]


def _arrs(points, n=N, vol=None):
    xs, ys = zip(*points, strict=True)
    c = np.interp(np.arange(n), xs, ys)
    v = np.full(n, 1000.0) if vol is None else vol
    if vol is None:
        v[251:] = 500.0
    return [np.ascontiguousarray(x[None, :]) for x in (c * (1 + BAND), c * (1 - BAND), c, v)]


def _scan(points, t=T, **kw):
    H, L, C, Vv = _arrs(points)
    r = Wd.wd_scan_arrays(H, L, C, Vv, np.array([0]), np.array([t]), kw or None)
    return {k: v[0] for k, v in r.items()}


def test_figure_passes_with_bottom_order():
    r = _scan(FIG)
    assert r["status"] == Wd.D_OK, Wd.WD_STATUS[r["status"]]
    assert (r["i0"], r["i_b"], r["i_c"], r["i_d"]) == (220, 235, 250, 262)
    A, B, Cc, D = 100 * (1 + BAND), 85 * (1 - BAND), 93 * (1 + BAND), 82 * (1 - BAND)
    assert r["c1"] == pytest.approx((A - B) / A) and r["c2"] == pytest.approx((Cc - D) / Cc)
    assert r["slope_u"] == pytest.approx(math.log(Cc / A) / 30) and r["slope_l"] == pytest.approx(math.log(D / B) / 27)
    assert r["d_high"] == pytest.approx(82 * (1 + BAND))
    assert r["order_px"] == pytest.approx(82 * (1 + BAND) + 1.0)      # D 봉 고가 + 1호가(2,000원 미만 = 1원)
    assert r["dlow"] == pytest.approx(D) and r["d_confirmed"] == 0       # D 진행 중(반등 5% 전)
    assert r["u_d"] == pytest.approx(A * math.exp(math.log(Cc / A) / 30 * (T + 1 - 220)))


def test_extra_reversal_between_a_and_b_fails():
    """네 점 연속: A 와 B 사이에 다른 반전점(90 → 95)이 있으면 확정 고점이 셋 → 셋업 없음."""
    pts = [(0, 40.0), (220, 100.0), (226, 90.0), (230, 95.0), (235, 85.0), (250, 93.0), (262, 82.0), (264, 82.8),
           (269, 82.8)]
    assert _scan(pts)["status"] == Wd.D_SHAPE


def test_far_from_d_fails():
    pts = FIG[:-2] + [(263, 82.5), (269, 82.5)]
    assert _scan(pts, t=267)["status"] == Wd.D_FAR                      # d − 행(D) = 268 − 262 = 6 > 5


def test_close_at_or_above_order_fails():
    pts = FIG[:-2] + [(264, 83.5), (269, 83.5)]                         # 종가 83.5 ≥ 주문가 83.33
    # 종가 ≥ 주문가면 그날 고가도 ≥ 주문가 → 두 번째 관통 차단이 먼저 건다(정정 1 이후 close>=order 는 사실상 그 부분집합).
    assert _scan(pts)["status"] in (Wd.D_CONSUMED, Wd.D_CLOSE)


def test_higher_c_or_low_d_fails():
    flat = [(0, 40.0), (220, 100.0), (235, 85.0), (250, 99.8), (262, 82.0), (264, 82.8), (269, 82.8)]
    assert _scan(flat)["status"] == Wd.D_HIGHS                           # C 99.8 > A × 0.99
    rising = [(0, 40.0), (220, 100.0), (235, 85.0), (250, 93.0), (262, 86.0), (264, 86.8), (269, 86.8)]
    assert _scan(rising)["status"] == Wd.D_LOWS                          # D ≥ B
    flat_low = [(0, 40.0), (220, 100.0), (235, 85.0), (250, 93.0), (262, 84.6), (264, 85.4), (269, 85.4)]
    assert _scan(flat_low)["status"] == Wd.D_LOWS                        # D < B 지만 B × 0.99 위(정정 1 M1)


def test_second_pass_blocked():
    """정정 1 M3: D 봉 뒤 고가가 이미 주문가(D 봉 고가 + 1호가)를 넘은 날이 있으면 그 D 로는 사지 않는다."""
    H, L, C, Vv = _arrs(FIG)
    H[0, 263] = 83.5                                                     # 주문가 83.33 이상 — 관통 뒤 종가는 아래
    r = Wd.wd_one(H, L, C, Vv, 0, T)
    assert r["status"] == Wd.D_CONSUMED


def test_wd_nc_turns_off_convergence():
    pts = [(0, 40.0), (220, 100.0), (235, 90.0), (250, 96.0), (262, 84.5), (264, 85.2), (269, 85.2)]   # c1 ≈ 10.4% < c2 ≈ 12.4%
    assert _scan(pts)["status"] == Wd.D_NOT_DECR
    r = _scan(pts, conv=False)
    assert r["status"] == Wd.D_OK, Wd.WD_STATUS[r["status"]]


def test_no_lookahead_bar_d():
    H, L, C, Vv = _arrs(FIG)
    a = Wd.wd_scan_arrays(H, L, C, Vv, np.array([0]), np.array([T]))
    H2, L2, C2, V2 = (x.copy() for x in (H, L, C, Vv))
    H2[0, T + 1], L2[0, T + 1], C2[0, T + 1], V2[0, T + 1] = 500.0, 1.0, 300.0, 1e9
    b = Wd.wd_scan_arrays(H2, L2, C2, V2, np.array([0]), np.array([T]))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_python_path_matches_batch_random():
    rng = np.random.default_rng(9)
    n, k = 420, 8
    c = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n, k)), axis=0))
    h = c * (1 + np.abs(rng.normal(0, 0.012, (n, k))))
    lo = c * (1 - np.abs(rng.normal(0, 0.012, (n, k))))
    vol = rng.lognormal(7, 0.6, (n, k))
    H, L, C, Vv = (np.ascontiguousarray(x.T) for x in (h, lo, c, vol))
    ts = np.arange(259, n)
    seen = set()
    for ci in range(k):
        r = Wd.wd_scan_arrays(H, L, C, Vv, np.full(len(ts), ci), ts)
        for i, t in enumerate(ts):
            ref = Wd.wd_one(H, L, C, Vv, ci, int(t))
            for kk in r:
                a, b = r[kk][i], ref[kk]
                assert (a == b) or (np.isnan(a) and np.isnan(b)), (kk, a, b)
            seen.add(int(ref["status"]))
    assert len(seen) >= 4


def test_wd_points():
    H, L, C, Vv = _arrs(FIG)
    pt = Wd.wd_points(H, L, 0, T, 235, 250, 262)
    assert [p["kind"] for p in pt["points"]] == ["A", "B", "C", "D"]
    assert pt["upper"]["to"]["price"] == pytest.approx(_scan(FIG)["u_d"])


# --- 바닥 매수정지 · 청산 ---

def _book(rows):
    days = pd.bdate_range("2024-01-01", periods=len(rows))
    a = np.array(rows, float)
    df = {k: pd.DataFrame({"X": a[:, i]}, index=days) for i, k in enumerate("ohlc")}
    ma = pd.DataFrame({"X": np.full(len(rows), np.nan)}, index=days)
    v = pd.DataFrame({"X": np.full(len(rows), 1000.0)}, index=days)
    return V.Arrays(df["o"], df["h"], df["l"], df["c"], ma, v)


BASE = (96.0, 97.0, 95.0, 96.5)


def _sim(A, d_high=100.0, dlow=95.0, d=3, up=False, **params):
    S = pd.DataFrame({"code": ["X"], "day_idx": [d], "d_high": [d_high], "u_d": [d_high], "dlow": [dlow], "i0": [0],
                      "i_c": [1]})
    return V.simulate_buystop(A, Wd.wd_events(S, up=up), params={**Wd.WD_BUYSTOP, **params})


def test_bottom_buystop_fill_and_stop():
    A = _book([BASE, BASE, BASE, (99.5, 102.0, 99.0, 101.5)] + [(101.0, 102.0, 100.5, 101.5)] * 6)
    t, _ = _sim(A)
    r = t.iloc[0]
    assert r["order_px"] == 101.0 and r["fill_px"] == 101.0           # D 봉 고가 100 + 1호가
    assert r["stop_px"] == 94.0                                        # D 95 − 1호가
    assert r["entry_px"] == pytest.approx(101.101)


def test_no_failed_breakout_before_2r():
    """2R 전 돌파 실패 규칙 없음: 체결 뒤 종가가 주문가 아래로 며칠 있어도 손절가 위면 보유(이틀·하루 규칙 모두 없음)."""
    rows = [BASE, BASE, BASE, (99.5, 102.0, 99.0, 99.6), (99.5, 100.0, 98.0, 98.5), (98.4, 99.0, 97.5, 98.0)]
    rows += [(98.0, 99.0, 97.6, 98.2)] * 4
    t, _ = _sim(_book(rows))
    assert t.iloc[0]["exit_reason"] == "data_end"


def test_stop_fill_is_stop_minus_one_tick():
    """정정 1 M5: 손절 체결 = min(시가, 손절가) − 1호가(−0.1% 대신)."""
    rows = [BASE, BASE, BASE, (99.5, 102.0, 99.0, 101.5), (100.0, 100.5, 93.0, 93.5)] + [BASE] * 5
    t, _ = _sim(_book(rows))
    r = t.iloc[0]
    assert r["exit_reason"] == "stop" and r["exit_px"] == pytest.approx(94.0 - 1.0)
    rows2 = [BASE, BASE, BASE, (99.5, 102.0, 99.0, 101.5), (92.0, 93.0, 91.0, 92.5)] + [BASE] * 5
    t2, _ = _sim(_book(rows2))                                           # 시가 92 < 손절가 94 → 92 − 1
    assert t2.iloc[0]["exit_px"] == pytest.approx(91.0)


def test_gap_and_risk_reference_order():
    A = _book([BASE, BASE, BASE, (104.2, 106.0, 104.0, 105.0)] + [BASE] * 6)
    t, sk = _sim(A)                                                    # 시가 104.2 > 101 × 1.03
    assert t.empty and sk["skip_reason"].tolist() == ["skip_gap>3%"]
    B = _book([BASE, BASE, BASE, (99.5, 102.0, 99.0, 101.5)] + [BASE] * 6)
    t2, sk2 = _sim(B, dlow=92.0)                                       # (101 − 91)/101 = 9.9% > 9.5%
    assert t2.empty and sk2["skip_reason"].tolist() == ["skip_risk>9.5%"]


def test_null_bottom_kernel():
    """널: 직전 k 일 고가 대비 ≥ 8% 눌린 직전 j 일 최저 저가 봉 — 그 봉 고가가 기준, 손절 기준 = 그 저가."""
    H, L, C, Vv = _arrs(FIG)
    days = pd.bdate_range("2020-01-01", periods=N)
    mk = lambda a: pd.DataFrame({"X": a[0]}, index=days)  # noqa: E731
    A = V.Arrays(mk(C), mk(H), mk(L), mk(C), mk(np.full((1, N), np.nan)), mk(Vv))
    o = Wd.null_bottoms(A, [0], [T], [20], parallel=False)            # 창 245..264 최고 = C(250) → 이후 최저 = D(262)
    assert bool(o["ok"][0]) and o["pid"][0] == 262 and not bool(o["yday"][0])
    assert o["pivot"][0] == pytest.approx(H[0, 262]) and o["l_last"][0] == pytest.approx(L[0, 262])
    far = Wd.null_bottoms(A, [0], [T + 3], [20], parallel=False)      # 같은 바닥이 d − 행 = 6 → 실패
    assert not bool(far["ok"][0])
    y = Wd.null_bottoms(A, [0], [262], [20], parallel=False)          # 어제(s)가 되돌림 최저 → 우선 표시
    assert bool(y["ok"][0]) and bool(y["yday"][0])
    shallow = Wd.null_bottoms(A, [0], [T], [20], pull=0.20, parallel=False)   # 눌림 12.5% < 20%
    assert not bool(shallow["ok"][0])


# ---------------------------------------------------------------------------
# 정정 1 — 러너 보조(열람 제외 · 총 R)
# ---------------------------------------------------------------------------

def _runner():
    pytest.importorskip("psycopg2", reason="러너 import 에 DB 드라이버 필요")
    from research.experiments import wedge_d_run as WDR
    return WDR


def test_viewed_window_overlap(tmp_path):
    WDR = _runner()
    days = pd.bdate_range("2020-01-01", periods=400)
    f = tmp_path / "v.csv"
    f.write_text("code,first_bar,last_bar,source\n005930,%s,%s,x\n" % (days[200].date(), days[220].date()))
    win = WDR.load_viewed(days, str(f), pad=60)
    assert win == {"005930": [(140, 280)]}
    t = pd.DataFrame({"code": ["005930", "005930", "005930", "000660"], "entry_row": [100, 100, 281, 200],
                      "exit_row": [139, 140, 300, 210]})
    assert WDR.viewed_flag(t, win, with_exit=True).tolist() == [False, True, False, False]
    # 깔때기 모드 — 진입일만
    assert WDR.viewed_flag(t, win, with_exit=False).tolist() == [False, False, False, False]


def test_gross_r_definition():
    WDR = _runner()
    # 체결 100, 손절 95 → d = 5%; 청산 90 → 총 R = −2, 청산 110 → +2
    x = pd.DataFrame({"w": [1.0, 1.0], "ret": [-0.10, 0.10], "d": [0.05, 0.05], "net": [-0.11, 0.09]})
    g = WDR.gross_r(x)
    assert math.isclose(g["rg_loss_mean"], -2.0)
    assert math.isclose(g["rg_mean"], 0.0, abs_tol=1e-12)
    b = WDR.bar_input({"avg_loss_R": -9.0, "r_p5": -9.0, **g})
    assert b["avg_loss_R"] == g["rg_loss_mean"] and b["r_p5"] == g["rg_p5"]


def test_viewed_csv_has_twelve_codes():
    v = pd.read_csv("research/logs/sepa_explore/wedge_d/viewed_windows.csv", dtype={"code": str})
    assert len(v) == 12 and v["code"].str.len().eq(6).all()
    assert (pd.to_datetime(v["first_bar"]) <= pd.to_datetime(v["last_bar"])).all()
