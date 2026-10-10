"""research/signals/meta_fast — 메타 세대 1 커널(합성 데이터, DB 없음).

- 정합 초과 시계열 = ``ga_fast.portfolio`` (슬롯 상한 10 고정·조기 해제 끔·절단 섞음) — 주문·진입·짝 없음까지 같다.
- 절대수익: 한 포지션 손익이 손으로 푼 복리 값과 같고, 명목은 진입 신호일 자산의 10%.
- 시가평가 절단(매도분 여러 개)은 ``ga_fast._mark_py`` 와 같은 총수익.
- 슬롯 상한(노출 조절)은 신규 진입만 막는다, 조기 해제는 리밸런스 날 빈 슬롯을 센다.
- P2 경로: 손절·리밸런스 청산(시가)·계속 보유·국면 게이트(N = 0)·절단 시가평가·폐지; 대조는 같은 청산일.
- 월 블록 셔플·성적·등가중 보유 도우미, numba 경로 = 순수 파이썬 경로.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from research.signals import exits_v2_fast as xv
from research.signals import ga_fast as gf
from research.signals import meta_fast as mf
from tests.test_exits_v2_fast import _old_to_v2, _port_inputs


@contextlib.contextmanager
def _pure():
    saved = [(m, n, getattr(m, n)) for m, names in ((mf, mf.HELPERS), (gf, ("_h32", "_sim_one", "_first_tr", "_mark")),
                                                    (xv, xv.HELPERS)) for n in names]
    try:
        for m, n, f in saved:
            setattr(m, n, getattr(f, "py_func", f))
        yield
    finally:
        for m, n, f in saved:
            setattr(m, n, f)


def _ga_world(seed: int = 4):
    """옛 청산(돌파 실패 켬) — ga_fast 와 v2 두 형식 + 대조 번호·절단."""
    b, lp, ev, pc, pd_ = _port_inputs(seed=seed)
    clow, piv = ev["contraction_low"].to_numpy(float), ev["pivot"].to_numpy(float)
    p_old = {**gf.EXIT_PREREG, "ts_days": None, "fb": True}
    old = gf.sim_book(b.O, b.H, b.L, b.C, b.MA, b.tradable, lp, pc, pd_, clow, p_old, pivot=piv)
    new = xv.sim_book_v2(b.O, b.H, b.L, b.C, b.MA, np.full(b.C.shape, np.nan), b.tradable, lp, pc, pd_, clow,
                         _old_to_v2(p_old), pivot=piv)
    P, n_days = len(pc), len(b.days)
    rng = np.random.default_rng(seed)
    st = old[0]
    fill = st == gf.ST_OK
    cidx = np.full((P, 5), -1, np.int32)
    for i in range(P):
        same = np.flatnonzero((pd_ == pd_[i]) & fill & (np.arange(P) != i))
        if len(same) and rng.random() < 0.8:
            cidx[i] = rng.choice(same, 5)
    cut_day = int(pd_[len(pd_) // 2]) + 3
    cutv = np.where(pd_ + 1 < cut_day, cut_day, n_days + 1).astype(np.int32)      # 앞 절반은 절단, 뒤는 없음
    return {"b": b, "pc": pc, "pd": pd_, "old": old, "new": new, "fill": fill, "cidx": cidx, "cutv": cutv,
            "orderable": st != gf.ST_NO_NEXT, "rel": rng.normal(1, 0.3, P), "rs": rng.uniform(50, 99, P),
            "sig": rng.random(P) < 0.5, "cost_tab": np.full((n_days, 2), 0.004), "market": np.zeros(P, np.int8),
            "n_days": n_days, "P": P}


def _meta_inputs(w: dict, cost_mult: float = 1.0, py: bool = False):
    st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = w["new"]
    b = w["b"]
    keep = w["sig"] & w["orderable"] & (w["pd"] + 1 < w["cutv"])
    idx = np.flatnonzero(keep).astype(np.int64)
    f = mf._cands_py if py else mf.cands
    cx, cea, cef, cer, cff, cfr, ctot, cctl, ccost = f(idx, w["cutv"][idx].astype(np.int64), w["pc"], w["pd"], x, ea, ef, er,
                                                       ff, fr, b.O, b.C, w["cidx"], w["cost_tab"], w["market"], cost_mult)
    lo_, hi_ = mf.day_bounds(w["pd"][idx], w["n_days"])
    return idx, (w["pd"][idx], w["pc"][idx], w["fill"][idx], w["rel"][idx], w["rs"][idx], cx, cea, cef, cer, cff, cfr,
                 cctl, ccost, np.zeros(len(idx), np.bool_), lo_, hi_), ctot


def test_excess_series_equals_ga_fast_portfolio_with_cut():
    w = _ga_world()
    b = w["b"]
    st, x, ret, r1, r2, dd, pab, rs = w["old"]
    day_start = np.searchsorted(w["pd"], np.arange(w["n_days"] + 1)).astype(np.int64)
    t0, t1 = int(w["pd"][0]), w["n_days"]
    sig = w["sig"] & (w["pd"] + 1 < w["cutv"])
    d_old, op_old, nord_old, nun_old = gf.portfolio(sig, w["fill"], w["orderable"], w["pc"], w["pd"], day_start, t0, t1,
                                                    w["rel"], w["rs"], x, pab, ret, r2, b.O, b.C, w["cidx"], w["cost_tab"],
                                                    w["market"], w["cutv"], gf.SLOTS, gf.SLOT_W)
    idx, args, _ = _meta_inputs(w)
    cap = np.full(w["n_days"], gf.SLOTS, np.int64)
    for f in (mf.portfolio_meta, mf._portfolio_meta_py):
        raw, exc, gross, op, nord, nun = f(*args, cap, b.O, b.C, t0, t1, gf.SLOT_W, 2 * gf.SLOTS)
        assert len(op_old) > 30 and (idx[op] == op_old).all() and nord == nord_old and nun == nun_old
        np.testing.assert_allclose(exc, d_old, rtol=0, atol=1e-7)          # 옛 엔진 r2 는 float32
        assert np.isfinite(raw).all() and np.nanmax(gross) < 1.5


def test_mark_ev_total_equals_ga_fast_mark():
    w = _ga_world(seed=7)
    b = w["b"]
    st, x, ret, r1, r2, dd, pab, rs = w["old"]
    nst, nx, nret, ndd, nrs, ea, ef, er, ff, fr, fl = w["new"]
    ok = np.flatnonzero(st == gf.ST_OK)
    n_cut = 0
    for i in ok:
        for cut in (int(w["pd"][i]) + 2, int(w["pd"][i]) + 6, int(x[i]), int(x[i]) + 1, w["n_days"] + 1):
            want = gf._mark_py(i, cut, w["pc"], w["pd"], x, pab, ret, r2, b.O, b.C)
            with _pure():
                got = mf._mark_ev_py(i, cut, w["pc"], w["pd"], nx, ea, ef, er, ff, fr, b.O, b.C)
            assert got[0] == want[1]
            assert mf._total_py(*got[1:]) == pytest.approx(want[0], abs=1e-6)
            n_cut += got[0] != x[i]
    assert n_cut > 50


def test_raw_compounding_one_position_by_hand():
    # 종목 1개 · 6일: 신호일 0, 진입 1 시가 100, 종가 110 · 121, 3일 시가 청산 133.1, 비용 1%
    O = np.array([[100.0, 100.0, 115.0, 133.1, 130.0, 130.0]])  # noqa: E741
    C = np.array([[100.0, 110.0, 121.0, 125.0, 130.0, 130.0]])
    one = lambda v, dt=np.float64: np.array([v], dt)  # noqa: E731
    args = (one(0, np.int64), one(0, np.int64), one(True, np.bool_), one(0.0), one(0.0), one(3, np.int64),
            np.full((1, 2), -1, np.int64), np.zeros((1, 2)), np.full((1, 2), np.nan), one(1.0), one(0.331),
            one(np.nan), one(0.01), one(False, np.bool_), *mf.day_bounds(one(0, np.int64), 6))
    raw, exc, gross, op, nord, nun = mf._portfolio_meta_py(*args, np.full(6, 10, np.int64), O, C, 0, 6, 0.1, 20)
    assert list(op) == [0] and nord == 1 and nun == 1
    np.testing.assert_allclose(raw[:4], [0.0, 0.1 * 0.10, 0.1 * 0.11 / 1.01, (0.1 * (0.331 - 0.21) - 0.001) / 1.021])
    assert (exc == 0).all()                                           # 짝 없음 → 정합 초과 0
    assert np.prod(1 + raw) == pytest.approx(1 + 0.1 * 0.331 - 0.001)


def test_cap_blocks_new_entries_only_and_early_frees_slot():
    # 종목 4개, 신호일 0 에 넷 다 신호 — 상한 2 → 둘만, 신호일 2 에 나머지 둘(조기 해제: 보유 둘이 3일 시가 청산)
    n_days = 8
    O = np.full((4, n_days), 100.0)  # noqa: E741
    C = np.full((4, n_days), 100.0)
    pd_ = np.array([0, 0, 0, 0, 2, 2], np.int64)
    pc = np.array([0, 1, 2, 3, 2, 3], np.int64)
    x = np.array([3, 3, 5, 5, 6, 6], np.int64)
    n = len(pd_)
    base = (pd_, pc, np.ones(n, np.bool_), np.arange(n, dtype=float), np.zeros(n), x, np.full((n, 2), -1, np.int64),
            np.zeros((n, 2)), np.full((n, 2), np.nan), np.ones(n), np.zeros(n), np.full(n, 0.0), np.zeros(n))
    lo_, hi_ = mf.day_bounds(pd_, n_days)
    cap = np.array([2, 2, 2, 2, 2, 2, 2, 2], np.int64)
    early = np.array([True, True, False, False, False, False])
    _, _, _, op, nord, _ = mf._portfolio_meta_py(*base, early, lo_, hi_, cap, O, C, 0, n_days, 0.1, 8)
    assert list(op) == [0, 1, 4, 5] and nord == 4                       # 조기 해제로 신호일 2 에 두 자리
    _, _, _, op2, _, _ = mf._portfolio_meta_py(*base, np.zeros(n, np.bool_), lo_, hi_, cap, O, C, 0, n_days, 0.1, 8)
    assert list(op2) == [0, 1]                                          # 조기 해제 없으면 신호일 2 에 자리 없음
    cap0 = cap.copy()
    cap0[0] = 0
    _, _, _, op3, nord3, _ = mf._portfolio_meta_py(*base, early, lo_, hi_, cap0, O, C, 0, n_days, 0.1, 8)
    assert list(op3) == [4, 5] and nord3 == 2                           # 상한 0 인 날은 주문 없음


# ---------------------------------------------------------------------------
# P2 경로
# ---------------------------------------------------------------------------

def _p2_book():
    """종목 3개 × 12일. 0: 꾸준히 오름, 1: 3일째 −20% 갭(손절), 2: 9일 뒤 시세 없음(폐지)."""
    n = 12
    C = np.vstack([100 * 1.01 ** np.arange(n), np.r_[100, 100, 100, 70, 70, 70, 70, 70, 70, 70, 70, 70],
                   np.r_[100 * np.ones(9), np.nan, np.nan, np.nan]]).astype(float)
    O = C.copy()  # noqa: E741
    O[1, 3] = 75.0
    H, L = C * 1.01, C * 0.99
    L[1, 3] = 69.0
    TR = np.isfinite(C)
    last_tr = np.array([n - 1, n - 1, 8], np.int64)
    return O, H, L, C, TR, last_tr


def test_p2_paths_stop_rebalance_continue_gate_mark_delist():
    O, H, L, C, TR, last_tr = _p2_book()  # noqa: E741
    LD = np.zeros_like(TR)
    # 쌍: 리밸런스 0(신호일 0) 종목 0·1·2, 리밸런스 1(신호일 5) 종목 0
    pc = np.array([0, 1, 2, 0], np.int64)
    pd_ = np.array([0, 0, 0, 5], np.int64)
    rb_d = np.array([0, 5, 9], np.int64)
    BIG = 1 << 30
    rank = np.array([[0, 1, 2], [0, BIG, BIG], [BIG, BIG, BIG]], np.int32)   # 리밸런스 2 에서 종목 0 탈락
    fill = np.ones(4, np.bool_)
    sel = np.array([0, 1, 2], np.int64)
    sel_k = np.array([0, 0, 0], np.int64)
    N = np.array([3, 3, 3], np.int64)
    lp = np.array([True, True, False])
    cut = np.full(3, 99, np.int64)
    x, ret, reason, early, xopen = mf._p2_trades_py(sel, sel_k, pc, pd_, rb_d, N, rank, O, H, L, C, TR, LD, last_tr, lp,
                                                    fill, cut, 0.15)
    # 종목 0: 리밸런스 1 에 상위(계속), 리밸런스 2 에 탈락 → 10일 시가
    assert x[0] == 10 and reason[0] == mf.R_REBAL and early[0] and xopen[0]
    assert ret[0] == pytest.approx(O[0, 10] / O[0, 1] - 1)
    # 종목 1: 3일째 시가 75 ≤ 손절 85 → 시가 75
    assert x[1] == 3 and reason[1] == gf.R_STOP and ret[1] == pytest.approx(75 / 100 - 1)
    # 종목 2: 리밸런스 1 에 탈락 → 6일 시가(조기 해제)
    assert x[2] == 6 and reason[2] == mf.R_REBAL and early[2]
    # 국면 게이트(리밸런스 1 의 N = 0) → 종목 0 도 6일 시가
    x2, *_ = mf._p2_trades_py(sel, sel_k, pc, pd_, rb_d, np.array([3, 0, 3], np.int64), rank, O, H, L, C, TR, LD, last_tr, lp,
                              fill, cut, 0.15)
    assert x2[0] == 6
    # 절단 5(→ 4일 종가 시가평가), 종목 2 는 계획 청산 전 폐지가 오면 마지막 거래일 종가
    x3, ret3, reason3, early3, _ = mf._p2_trades_py(sel, sel_k, pc, pd_, rb_d, N, rank, O, H, L, C, TR, LD, last_tr, lp, fill,
                                                    np.full(3, 5, np.int64), 0.15)
    assert x3[0] == 4 and reason3[0] == mf.R_MARK and ret3[0] == pytest.approx(C[0, 4] / O[0, 1] - 1) and not early3[0]
    rank_d = rank.copy()
    rank_d[1, 2] = 1                                                   # 종목 2 를 계속 보유 → 9일부터 시세 없음
    x4, ret4, reason4, *_ = mf._p2_trades_py(sel, sel_k, pc, pd_, rb_d, N, rank_d, O, H, L, C, TR, LD, last_tr, lp, fill, cut,
                                             0.15)
    assert x4[2] == 8 and reason4[2] == gf.R_DELISTED
    # 하한가로 잠긴 3일째(종목 1) → 손절은 4일 시가(70)로 밀린다
    LD2 = LD.copy()
    LD2[1, 3] = True
    x5, ret5, reason5, *_ = mf._p2_trades_py(sel, sel_k, pc, pd_, rb_d, N, rank, O, H, L, C, TR, LD2, last_tr, lp, fill, cut,
                                             0.15)
    assert x5[1] == 4 and reason5[1] == gf.R_STOP and ret5[1] == pytest.approx(70 / 100 - 1)


def test_p2_controls_same_exit_day_and_own_stop():
    O, H, L, C, TR, last_tr = _p2_book()  # noqa: E741
    LD = np.zeros_like(TR)
    pc = np.array([0, 1, 2], np.int64)
    pd_ = np.zeros(3, np.int64)
    sel = np.array([0], np.int64)
    cidx = np.array([[1, 1, 2, 2, 2], [-1] * 5, [-1] * 5], np.int32)
    # 트레이드 0 이 6일 시가 청산 → 대조 1 은 3일 손절(75), 대조 2 는 6일 시가 100
    out = mf._p2_ctl_py(sel, np.array([6], np.int64), np.array([True]), cidx, pc, pd_, O, L, C, TR, LD, 0.15)
    assert out[0] == pytest.approx((2 * (75 / 100 - 1) + 3 * 0.0) / 5)
    # 종가 청산(절단) 이면 그날 종가
    out2 = mf._p2_ctl_py(sel, np.array([2], np.int64), np.array([False]), cidx, pc, pd_, O, L, C, TR, LD, 0.15)
    assert out2[0] == pytest.approx(0.0)
    assert np.isnan(mf._p2_ctl_py(np.array([1], np.int64), np.array([6], np.int64), np.array([True]), cidx, pc, pd_, O, L,
                                  C, TR, LD, 0.15)[0])


# ---------------------------------------------------------------------------
# 도우미
# ---------------------------------------------------------------------------

def test_month_block_perm_keeps_days_and_within_month_order():
    month = np.repeat(np.arange(6), [3, 5, 2, 4, 4, 1])
    rng = np.random.default_rng(0)
    perm = mf.month_block_perm(month, 3, len(month), rng)
    assert sorted(perm) == list(range(3, len(month)))
    seg = month[perm]
    starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
    assert len(starts) == 5                                             # 달이 섞여도 한 달은 한 덩어리
    for s, e in zip(starts, np.r_[starts[1:], len(seg)], strict=True):
        assert (np.diff(perm[s:e]) == 1).all()


def test_equity_stats_matches_hand_values():
    r = np.r_[[0.1] * 126, [-0.5], [0.0] * 125]
    s = mf.equity_stats(r)
    eq = 1.1 ** 126 * 0.5
    assert s["cagr"] == pytest.approx(eq - 1.0)                         # 252 일 = 1년
    assert s["mdd"] == pytest.approx(-0.5) and s["calmar"] == pytest.approx((eq - 1.0) / 0.5)
    assert mf.equity_stats(np.r_[-0.2, 0.05])["mdd"] == pytest.approx(-0.2)   # 첫날 손실도 낙폭


def test_ew_daily_and_monthly_hand_case():
    C = np.array([[10.0, 11.0, 12.1, 13.31, 6.0], [20.0, 20.0, np.nan, 22.0, 22.0]])
    md = np.array([0, 0, 1, 2, 3], np.int64)
    mc = np.array([0, 1, 0, 1, 0], np.int64)
    d = mf.ew_daily(C, md, mc, 1, 5)
    # t=1: 두 종목 (0.1 + 0), t=2: 종목 0 만(+0.1), t=3: 종목 1 NaN→뺌… 종목 1 은 t−1 NaN 이라 빠짐 → 0 개 = 0, t=4: 종목 0 −55% 는 자료 사건
    np.testing.assert_allclose(d, [0.05, 0.1, 0.0, 0.0])
    m = mf.ew_monthly(C, md, mc, np.array([0], np.int64), 1, 5)
    # 신호일 0 에 두 종목 같은 금액, 종목 1 의 NaN 은 이월(수익 0), 재개일 22/20
    v0 = np.cumprod([1.1, 1.1, 1.1, 1.0])                             # 마지막 날 −55% → 0
    v1 = np.array([1.0, 1.0, 1.1, 1.1])
    port = (v0 + v1) / 2
    np.testing.assert_allclose(np.cumprod(1 + m), port)


def test_numba_paths_equal_python_paths():
    pytest.importorskip("numba")
    w = _ga_world(seed=11)
    idx, args, tot = _meta_inputs(w)
    with _pure():
        idx2, args2, tot2 = _meta_inputs(w, py=True)
    for a, z in zip(args, args2, strict=True):
        np.testing.assert_array_equal(a, z)
    cap = np.full(w["n_days"], 7, np.int64)
    b = w["b"]
    t0 = int(w["pd"][0])
    nb = mf.portfolio_meta(*args, cap, b.O, b.C, t0, w["n_days"], 0.1, 20)
    with _pure():
        py_ = mf._portfolio_meta_py(*args, cap, b.O, b.C, t0, w["n_days"], 0.1, 20)
    for a, z in zip(nb, py_, strict=True):
        np.testing.assert_allclose(a, z, rtol=0, atol=1e-12)
    O, H, L, C, TR, last_tr = _p2_book()  # noqa: E741
    p2 = (np.array([0, 1, 2], np.int64), np.zeros(3, np.int64), np.array([0, 1, 2, 0], np.int64),
          np.array([0, 0, 0, 5], np.int64), np.array([0, 5, 9], np.int64), np.array([3, 3, 1], np.int64),
          np.array([[0, 1, 2], [0, 5, 5], [5, 0, 5]], np.int32), O, H, L, C, TR, np.zeros_like(TR), last_tr, np.array([True, True, False]),
          np.ones(4, np.bool_), np.full(3, 99, np.int64), 0.15)
    for a, z in zip(mf.p2_trades(*p2), mf._p2_trades_py(*p2), strict=True):
        np.testing.assert_array_equal(a, z)


def test_locked_down_and_clean_jumps():
    C = np.array([[100.0, 70.5, 70.5, 140.0, 141.0], [100.0, 90.0, np.nan, 89.0, 88.0]])
    O = C.copy()  # noqa: E741
    H, L = C.copy(), C.copy()
    H[1] = C[1] * 1.01
    TR = np.isfinite(C)
    ld = mf.locked_down(O, H, L, C, TR)
    assert ld.tolist() == [[False, True, False, False, False], [False] * 5]
    O2, H2, L2, C2, bad = mf.clean_jumps(O, H, L, C)
    assert bad.tolist() == [[False, False, False, True, False], [False] * 5]     # 70.5 → 140 (+99%) 만 자료 사건
    np.testing.assert_allclose(C2[0], [100, 70.5, 70.5, 70.5, 70.5 * 141 / 140])
    np.testing.assert_allclose(C2[1], C[1], equal_nan=True)


def test_calmar_boot_lower_and_paired_boot():
    m = np.r_[[0.02] * 20, [-0.1], [0.02] * 15]
    lo = mf.calmar_boot_lower(m)
    full = mf.calmar_paths(m[None, :], 12)[0]
    assert np.isfinite(lo) and lo < full
    assert mf.calmar_boot_lower(np.zeros(36)) == 0.0                      # 전부 현금 → Calmar 0
    assert mf.calmar_boot_lower(m[:3]) == -np.inf
    rng = np.random.default_rng(0)
    ix = mf.stationary_idx(50, 5, 3, rng)
    assert ix.shape == (3, 50) and ix.min() >= 0 and ix.max() < 50
    steps = np.diff(ix, axis=1) % 50
    assert (steps == 1).mean() > 0.7                                    # 대부분 블록 안에서 이어진다
    a = rng.normal(0.001, 0.01, 500)
    r = mf.paired_calmar_boot(a, a * 0.5, n=500)
    assert r["dcalmar_lo"] <= r["dcalmar_med"] <= r["dcalmar_hi"] and 0 <= r["p_calmar"] <= 1
    same = mf.paired_calmar_boot(a, a.copy(), n=250)
    assert same["p_calmar"] == 0.0 and same["dcalmar_med"] == 0.0
