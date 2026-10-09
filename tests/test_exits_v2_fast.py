"""research/signals/exits_v2_fast — 미너비니 청산 v2 커널(합성 데이터, DB 없음).

- v2 유전자 전부 끔 = 옛 커널(``ga_fast.sim_pairs``) — 상태·청산 행·사유·수익 **비트 단위**(청산 조합 여러 개: 시간 손절·익절 없음·
  추적 없음·돌파 실패).
- 손으로 만든 한 종목 경로: 본전 이동은 발동 **다음날부터**·진입가 × 1.005, 강세 매도는 다음날 시가 절반·한 번, 연속 이탈일·피벗
  허용폭, 같은 종가 우선순위(강세 매도 > 돌파 실패), 같은 날 손절 > 익절.
- numba 경로 = 순수 파이썬 경로(보조 함수까지 되감아서 — 리뷰 A-M2).
- 포트폴리오 v2 = ``ga_fast.portfolio``(옛 청산, 절단 없음), 날별 합 = Σ 0.1 × (수익 − 대조 − 비용).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from research.signals import exits_v2_fast as xv
from research.signals import ga_fast as gf
from research.signals.minervini_exits import simulate_minervini
from research.signals.sepa import LIMIT_UP_RATIO, _Book
from tests.test_ga_fast import EXIT_COMBOS, _events, _market


@contextlib.contextmanager
def _pure():
    """두 모듈의 JIT 보조 함수를 순수 파이썬으로 되감는다."""
    saved = [(m, n, getattr(m, n)) for m, names in ((xv, xv.HELPERS), (gf, ("_h32", "_sim_one", "_first_tr", "_mark")))
             for n in names]
    try:
        for m, n, f in saved:
            setattr(m, n, getattr(f, "py_func", f))
        yield
    finally:
        for m, n, f in saved:
            setattr(m, n, f)


def _old_to_v2(p: dict) -> dict:
    return {"stop_floor": p["stop_floor"], "stop_cap": p["stop_cap"], "stop_mult": p["stop_mult"], "take_r": p["take_r"],
            "fb": bool(p.get("fb", False)), "ts_days": p["ts_days"] or 0, "max_hold": p["max_hold"],
            "has_trail": p["has_trail"], "lu_skip": p["lu_skip"]}


def _arrays(o, h, lo, c, v, ma):
    b = _Book(o, h, lo, c, ma, v)
    return b, np.ascontiguousarray(b.present[:, -1])


@pytest.mark.parametrize("combo", range(len(EXIT_COMBOS)))
@pytest.mark.parametrize("seed", [0, 3])
def test_v2_all_off_equals_old_kernel_bitwise(combo, seed):
    o, h, lo, c, v = _market(seed=seed)
    p = {**gf.EXIT_PREREG, **EXIT_COMBOS[combo]}
    ma = c.rolling(10 if combo % 2 else 50, min_periods=5).mean()
    b, lp = _arrays(o, h, lo, c, v, ma if p["has_trail"] else ma * np.nan)
    ev = _events(o, h, lo, c, seed=seed + 1)
    pc = b.codes.get_indexer(ev["code"])
    pd_ = b.days.get_indexer(ev["event_day"])
    with _pure():
        old = gf._sim_pairs_py(b.O, b.H, b.L, b.C, b.MA, b.tradable, lp, pc.astype(np.int64), pd_.astype(np.int64),
                               ev["contraction_low"].to_numpy(float), p["stop_floor"], p["stop_cap"], p["stop_mult"],
                               np.nan, p["take_r"], p["ts_days"] or 0, p["max_hold"], p["has_trail"], LIMIT_UP_RATIO,
                               p["lu_skip"], ev["pivot"].to_numpy(float), bool(p.get("fb", False)))
        new = xv.sim_book_v2(b.O, b.H, b.L, b.C, b.MA, np.full(b.C.shape, np.nan), b.tradable, lp, pc, pd_,
                             ev["contraction_low"].to_numpy(float), _old_to_v2(p), pivot=ev["pivot"].to_numpy(float), py=True)
    st, x, ret, r1, r2, dd, pab, rs = old
    nst, nx, nret, ndd, nrs, ea, ef, er, ff, fr, fl = new
    assert (st == nst).all() and (x == nx).all() and (rs == nrs).all()
    ok = st == gf.ST_OK
    assert ok.sum() > 50
    assert (ret[ok] == nret[ok]).all()                                  # 비트 단위
    assert (dd[ok] == ndd[ok].astype(np.float32)).all()
    assert ((pab[ok] >= 0) == ((fl[ok] & xv.F_PARTIAL) > 0)).all()
    assert (pab[ok][pab[ok] >= 0] == ea[ok][pab[ok] >= 0, 0]).all()
    assert (r2[ok] == fr[ok]).all()
    assert ((fl[ok] & (xv.F_STRENGTH | xv.F_BE_TRIG)) == 0).all()


def test_simulate_minervini_v2_path_equals_old_path_at_defaults():
    o, h, lo, c, v = _market(seed=5)
    ma = c.rolling(50, min_periods=5).mean()
    ev = _events(o, h, lo, c, seed=6)
    for fb in (False, True):
        a = simulate_minervini(o, h, lo, c, ma, ev, volume=v, failed_breakout=fb, time_stop_days=None)
        b = simulate_minervini(o, h, lo, c, ma, ev, volume=v, failed_breakout=fb, time_stop_days=None, exits_v2=True)
        assert len(a) > 50 and len(a) == len(b)
        for col in ("code", "event_day", "exit_day", "exit_reason", "hold_days", "ret", "R", "ret_half1", "ret_half2"):
            assert (a[col].to_numpy() == b[col].to_numpy()).all(), col


# ---------------------------------------------------------------------------
# 손으로 만든 경로 — 한 종목
# ---------------------------------------------------------------------------

def _one(path: list[tuple[float, float, float, float]], *, prev: float = 100.0, c_low: float = 95.0,
         pivot: float = 100.0, ma50=None, ma=None, **params):
    """행 0 = 신호일(종가 ``prev``), 행 1.. = 진입일부터 (시가, 고가, 저가, 종가). 커널 순수 파이썬 → 결과 dict."""
    rows = np.array([(prev, prev, prev, prev)] + path + [(np.nan, np.nan, np.nan, np.nan)], float)
    Op, H, L, C = (np.ascontiguousarray(rows[:, k][None, :]) for k in range(4))     # (1 종목, 날짜) — 마지막 행 = 시세 없음
    n = Op.shape[1]
    TR = np.isfinite(Op)
    MA = np.full((1, n), np.nan) if ma is None else np.array([ma + [np.nan] * (n - len(ma))], float)
    M50 = np.full((1, n), np.nan) if ma50 is None else np.array([ma50 + [np.nan] * (n - len(ma50))], float)
    p = {**xv.EXIT_V2_DEFAULT, "stop_floor": 0.05, "stop_cap": 0.05, "lu_skip": False, **params}
    with _pure():
        out = xv.sim_book_v2(Op, H, L, C, MA, M50, TR, np.array([True]), [0], [0], [c_low], p, pivot=[pivot], py=True)
    st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = out
    return {"st": int(st[0]), "x": int(x[0]), "ret": float(ret[0]), "reason": gf.REASON_NAMES[rs[0]] if rs[0] >= 0 else None,
            "ev": [(int(ea[0, k]), float(ef[0, k]), float(er[0, k])) for k in range(2) if ea[0, k] >= 0],
            "fin": (float(ff[0]), float(fr[0])), "flags": int(fl[0])}


def test_be_r_moves_stop_to_entry_plus_cost_from_next_day_only():
    # 진입 100, d = 5% → 손절 95, be_r 1.0 → 고가 ≥ 105 발동. 발동일(행 2) 저가 99 는 아직 손절선 95 → 생존.
    path = [(100, 101, 99, 100), (100, 106, 99, 104), (104, 104, 100.2, 101), (101, 102, 100, 101)]
    r = _one(path, be_r=1.0, fb=False, take_r=np.inf, has_trail=False)
    assert r["reason"] == "breakeven" and r["x"] == 3 and r["ret"] == pytest.approx(0.005)   # 행 3 저가 100.2 ≤ 100.5
    assert r["flags"] & xv.F_BE_TRIG
    off = _one(path, be_r=np.inf, fb=False, take_r=np.inf, has_trail=False)
    assert off["reason"] == "data_end"


def test_be_r_gap_down_fills_at_open_and_trigger_day_keeps_tech_stop():
    path = [(100, 106, 94, 100), (99, 99, 98, 98.5)]                   # 발동일 저가 94 ≤ 95 → 그날은 기술적 손절이 먼저
    r = _one(path, be_r=1.0, fb=False, take_r=np.inf, has_trail=False)
    assert r["reason"] == "stop" and r["x"] == 1 and r["ret"] == pytest.approx(-0.05)
    path = [(100, 106, 99, 104), (98, 99, 97, 98)]                      # 다음날 갭 하락 시가 98 < 100.5 → 시가
    r = _one(path, be_r=1.0, fb=False, take_r=np.inf, has_trail=False)
    assert r["reason"] == "breakeven" and r["ret"] == pytest.approx(-0.02)


def test_strength_sell_half_next_open_once_then_rest():
    # 행 1 종가 100 → 행 2 종가 116 (+16% ≥ 15%) → 행 3 시가 117 에 절반, 행 4 다시 +16% 지만 한 번뿐.
    path = [(100, 101, 99, 100), (101, 117, 100, 116), (117, 118, 115, 117), (117, 136, 117, 135.8), (136, 137, 135, 136)]
    r = _one(path, ss_x=0.15, fb=False, take_r=np.inf, has_trail=False)
    assert r["ev"] == [(3, 0.5, pytest.approx(0.17))] and r["fin"][0] == 0.5
    assert r["reason"] == "data_end" and r["ret"] == pytest.approx(0.5 * 0.17 + 0.5 * 0.36)
    # MA50 이격 조건(y 30%): MA50 100 → 종가 ≥ 130 인 날 다음 시가
    r = _one(path, ss_y=0.30, ma50=[100.0] * 7, fb=False, take_r=np.inf, has_trail=False)
    assert r["ev"][0][0] == 5 and r["ev"][0][2] == pytest.approx(0.36)


def test_strength_beats_failed_breakout_on_same_close_and_fb_waits_next_close():
    # 손절 20%(80), 피벗 99, 2일 연속 이탈. 행 3 종가 98 = +15.3% 이면서 2일째 < 99 → 강세 매도(절반)가 이긴다(행 4 시가 98).
    # 행 4 종가 97 < 99 (3일째) → 돌파 실패 → 행 5 시가 96 에 나머지.
    path = [(100, 101, 99, 100), (100, 100, 84, 85), (85, 98, 85, 98), (98, 99, 97, 97), (96, 97, 95, 96)]
    r = _one(path, ss_x=0.15, pivot=99.0, fb_days=2, stop_floor=0.20, stop_cap=0.20, take_r=np.inf, has_trail=False)
    assert r["ev"] == [(4, 0.5, pytest.approx(-0.02))] and r["reason"] == "failed_breakout" and r["x"] == 5
    assert r["ret"] == pytest.approx(0.5 * -0.02 + 0.5 * -0.04)
    r = _one(path, pivot=99.0, fb_days=2, stop_floor=0.20, stop_cap=0.20, take_r=np.inf, has_trail=False)
    assert r["ev"] == [] and r["reason"] == "failed_breakout" and r["x"] == 4             # 강세 매도 끔 → 행 3 결정


def test_failed_breakout_days_and_tolerance():
    path = [(101, 102, 100, 99.5), (100, 101, 99, 100.5), (100, 101, 99, 99.0), (99, 100, 98, 98.5), (98, 99, 97, 98)]
    one = _one(path, pivot=100.0, take_r=np.inf, has_trail=False)
    assert one["reason"] == "failed_breakout" and one["x"] == 2                      # 1일: 행 1 종가 99.5 < 100
    two = _one(path, pivot=100.0, fb_days=2, take_r=np.inf, has_trail=False)
    assert two["reason"] == "failed_breakout" and two["x"] == 5                      # 2일 연속: 행 3·4 → 행 5 시가
    tol = _one(path, pivot=100.0, fb_tol=0.02, take_r=np.inf, has_trail=False)      # 98 기준: 행 4 98.5, 행 5 98 은 안 걸림
    assert tol["reason"] == "data_end"


def test_partial_fraction_third_and_stop_before_partial_same_day():
    path = [(100, 111, 99, 108), (108, 109, 101, 102), (102, 103, 99, 100)]
    r = _one(path, take_r=2.0, take_frac=1 / 3, fb=False, has_trail=False)        # 목표 110 — 1/3 익절, 나머지 본전(100)
    assert r["ev"] == [(1, pytest.approx(1 / 3), pytest.approx(0.10))] and r["reason"] == "breakeven" and r["x"] == 3
    assert r["ret"] == pytest.approx(0.10 / 3)
    r = _one([(100, 111, 94, 108)], take_r=2.0, fb=False, has_trail=False)        # 같은 날 손절·목표 → 손절
    assert r["reason"] == "stop" and r["ev"] == []


# ---------------------------------------------------------------------------
# numba = 파이썬
# ---------------------------------------------------------------------------

V2_COMBOS = [
    {"be_r": 1.0, "take_r": 2.0, "take_frac": 0.5, "ss_x": 0.15, "ss_y": 0.30, "fb": True, "fb_tol": 0.02, "fb_days": 2,
     "stop_cap": 0.08},
    {"be_r": 0.5, "take_r": np.inf, "ss_x": 0.15, "fb": True, "fb_tol": 0.05, "stop_cap": 0.06},
    {"be_r": 2.0, "take_r": 1.0, "take_frac": 1 / 3, "ss_y": 0.30, "fb": False},
    {"be_r": 1.5, "take_r": 3.0, "take_frac": 1 / 3, "fb": True, "fb_days": 2, "has_trail": False, "ts_days": 20},
]


def test_v2_numba_path_equals_python_path():
    pytest.importorskip("numba")
    o, h, lo, c, v = _market(seed=2)
    ma = c.rolling(20, min_periods=5).mean()
    m50 = np.ascontiguousarray(c.rolling(50, min_periods=5).mean().to_numpy().T)
    b, lp = _arrays(o, h, lo, c, v, ma)
    ev = _events(o, h, lo, c, n_ev=600, seed=9)
    pc, pd_ = b.codes.get_indexer(ev["code"]), b.days.get_indexer(ev["event_day"])
    for p in V2_COMBOS:
        args = (b.O, b.H, b.L, b.C, b.MA, m50, b.tradable, lp, pc, pd_, ev["contraction_low"].to_numpy(float), p)
        nb = xv.sim_book_v2(*args, pivot=ev["pivot"].to_numpy(float))
        with _pure():
            py_ = xv.sim_book_v2(*args, pivot=ev["pivot"].to_numpy(float), py=True)
        for a, z in zip(nb, py_, strict=True):
            np.testing.assert_array_equal(a, z)


# ---------------------------------------------------------------------------
# 포트폴리오 v2
# ---------------------------------------------------------------------------

def _port_inputs(seed: int = 4):
    o, h, lo, c, v = _market(n_codes=12, n_days=400, seed=seed)
    ma = c.rolling(50, min_periods=5).mean()
    b, lp = _arrays(o, h, lo, c, v, ma)
    ev = _events(o, h, lo, c, n_ev=900, seed=seed + 10).sort_values(["event_day", "code"]).reset_index(drop=True)
    pc = b.codes.get_indexer(ev["code"]).astype(np.int64)
    pd_ = b.days.get_indexer(ev["event_day"]).astype(np.int64)
    order = np.lexsort((pc, pd_))
    pc, pd_, ev = pc[order], pd_[order], ev.iloc[order].reset_index(drop=True)
    return b, lp, ev, pc, pd_


def test_portfolio_v2_equals_old_portfolio_on_old_exits_and_sum_identity():
    b, lp, ev, pc, pd_ = _port_inputs()
    clow, piv = ev["contraction_low"].to_numpy(float), ev["pivot"].to_numpy(float)
    p_old = {**gf.EXIT_PREREG, "ts_days": None, "fb": True}
    st, x, ret, r1, r2, dd, pab, rs = gf.sim_book(b.O, b.H, b.L, b.C, b.MA, b.tradable, lp, pc, pd_, clow, p_old, pivot=piv)
    nst, nx, nret, ndd, nrs, ea, ef, er, ff, fr, fl = xv.sim_book_v2(
        b.O, b.H, b.L, b.C, b.MA, np.full(b.C.shape, np.nan), b.tradable, lp, pc, pd_, clow, _old_to_v2(p_old), pivot=piv)
    P, n_days = len(pc), len(b.days)
    rng = np.random.default_rng(0)
    fill = st == gf.ST_OK
    sig = rng.random(P) < 0.5
    orderable = st != gf.ST_NO_NEXT
    rel, rsv = rng.normal(1, 0.3, P), rng.uniform(50, 99, P)
    rel[rng.random(P) < 0.1] = np.nan
    cidx = np.full((P, 5), -1, np.int32)                    # 대조 = 같은 날 아무 체결 쌍(테스트용 — 짝 없음 섞기)
    for i in range(P):
        same = np.flatnonzero((pd_ == pd_[i]) & fill & (np.arange(P) != i))
        if len(same) and rng.random() < 0.8:
            cidx[i] = rng.choice(same, 5)
    market = np.zeros(P, np.int8)
    cost_tab = np.full((n_days, 2), 0.004)
    day_start = np.searchsorted(pd_, np.arange(n_days + 1)).astype(np.int64)
    cutv = np.full(P, n_days + 1, np.int32)
    t0, t1 = int(pd_[0]), n_days
    d_old, op_old, nord_old, nun_old = gf.portfolio(sig, fill, orderable, pc, pd_, day_start, t0, t1, rel, rsv, x, pab, ret,
                                                    r2, b.O, b.C, cidx, cost_tab, market, cutv, gf.SLOTS, gf.SLOT_W)
    # v2: 후보 = sig 쌍(정렬 유지)
    cand = np.flatnonzero(sig)
    ctl = np.array([np.mean(nret[cidx[i]]) if cidx[i, 0] >= 0 else np.nan for i in cand])
    cost = np.where(nx[cand] >= 0, cost_tab[np.maximum(nx[cand], 0), 0], 0.0)
    lo_ = np.searchsorted(pd_[cand], np.arange(n_days), "left").astype(np.int64)
    hi_ = np.searchsorted(pd_[cand], np.arange(n_days), "right").astype(np.int64)
    for f in (xv.portfolio_v2, xv._portfolio_v2_py):
        d_new, op_new, nord, nun = f(pd_[cand], pc[cand], fill[cand], orderable[cand], rel[cand], rsv[cand], nx[cand],
                                     ea[cand], ef[cand], er[cand], ff[cand], fr[cand], ctl, cost, lo_, hi_, b.O, b.C, t0, t1,
                                     gf.SLOTS, gf.SLOT_W)
        assert (cand[op_new] == op_old).all() and nord == nord_old and nun == nun_old and len(op_old) > 30
        np.testing.assert_allclose(d_new, d_old, rtol=0, atol=1e-7)          # 옛 엔진 r2 는 float32
        m = np.isfinite(ctl[op_new])
        oi = cand[op_new]
        want = gf.SLOT_W * (nret[oi] - ctl[op_new] - cost[op_new])[m].sum()
        assert d_new.sum() == pytest.approx(want, abs=1e-10)


def test_portfolio_v2_multiple_lots_sum_identity():
    b, lp, ev, pc, pd_ = _port_inputs(seed=7)
    clow, piv = ev["contraction_low"].to_numpy(float), ev["pivot"].to_numpy(float)
    m50 = np.ascontiguousarray(pd.DataFrame(b.C.T).rolling(50, min_periods=5).mean().to_numpy().T)
    p = {**V2_COMBOS[0], "ss_x": 0.03}                                  # 강세 매도가 자주 걸리게
    st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = xv.sim_book_v2(b.O, b.H, b.L, b.C, b.MA, m50, b.tradable, lp, pc, pd_,
                                                               clow, p, pivot=piv)
    cand = np.flatnonzero(st == gf.ST_OK)
    assert ((fl[cand] & xv.F_STRENGTH) > 0).sum() > 20 and ((ea[cand, 1] >= 0)).sum() > 3
    n_days = len(b.days)
    ctl = np.where(np.arange(len(cand)) % 4 == 0, np.nan, 0.01)
    cost = np.full(len(cand), 0.003)
    lo_ = np.searchsorted(pd_[cand], np.arange(n_days), "left").astype(np.int64)
    hi_ = np.searchsorted(pd_[cand], np.arange(n_days), "right").astype(np.int64)
    d, op, nord, nun = xv.portfolio_v2(pd_[cand], pc[cand], np.ones(len(cand), np.bool_), np.ones(len(cand), np.bool_),
                                       np.zeros(len(cand)), np.zeros(len(cand)), x[cand], ea[cand], ef[cand], er[cand],
                                       ff[cand], fr[cand], ctl, cost, lo_, hi_, b.O, b.C, int(pd_[cand[0]]), n_days,
                                       gf.SLOTS, gf.SLOT_W)
    m = np.isfinite(ctl[op])
    assert d.sum() == pytest.approx(gf.SLOT_W * (ret[cand][op] - ctl[op] - cost[op])[m].sum(), abs=1e-10)
    assert nun == int((~m).sum()) and nord >= len(op)
