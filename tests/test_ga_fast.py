"""research/signals/ga_fast 커널 — 합성 데이터(DB 없음).

- 패리티: 엔진(사전계산 시뮬 + 추격 비트 + 겹침 제거) = ``minervini_exits.simulate_minervini`` (트레이드 단위, 수익 1e−9),
  청산 조합 여러 개(손절 상한·익절 배수·추적 MA·시간 손절·최대 보유). 익절 없음 ∧ 추적 = ``sepa.simulate_sepa`` (고정 손절,
  수축 저점이 정의되고 시가가 그 위인 사건 — 비교 가능한 곳).
- 순수 파이썬 경로는 항상, numba 경로는 numba 가 있을 때(``importorskip``) 같은 결과인지.
- 부트스트랩 하단: 결정적·``_boot_lower`` 와 통계적으로 가깝다. 대조 추출: 같은 층·자기 제외·고정.
- 포트폴리오: 날별 초과의 합 = Σ(수익 − 대조 − 비용) × 10% (purge·절단 없을 때), 슬롯 상한·같은 종목 보유 중 신호 버림.
- GA2 유전자 부호화/복호화(연속 [0,1]·단계 인덱스·끔 끝).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from research.signals import ga_fast as gf
from research.signals.minervini_exits import contraction_low_level, simulate_minervini
from research.signals.sepa import _Book, simulate_sepa
from swing_it.validation.optimization import _boot_lower


# 순수 파이썬 경로: ``_py`` 함수가 부르는 보조 함수(``_h32``·``_sim_one``·``_first_tr``·``_mark``·``_pct_sorted``)도 numba 가 있으면
# JIT 판이라, 그대로 두면 "numba = 파이썬" 비교가 동어반복이 된다(리뷰 A-M2). 비교의 파이썬 쪽은 이 안에서 ``py_func`` 로 바꿔 부른다.
_HELPERS = ("_h32", "_pct_sorted", "_sim_one", "_first_tr", "_mark")


@contextlib.contextmanager
def _pure():
    saved = {n: getattr(gf, n) for n in _HELPERS}
    try:
        for n, f in saved.items():
            setattr(gf, n, getattr(f, "py_func", f))
        yield
    finally:
        for n, f in saved.items():
            setattr(gf, n, f)


def py(fn, *a, **k):
    with _pure():
        return fn(*a, **k)


def test_pure_context_really_unjits_helpers():
    with _pure():
        assert not hasattr(gf._h32, "py_func") and not hasattr(gf._sim_one, "py_func")


def _market(n_codes: int = 8, n_days: int = 320, seed: int = 0):
    """갭·정지일(거래량 0, OHLC 이월)·폐지(뒤 NaN)·데이터 끝이 섞인 합성 패널."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_codes)]
    Op, H, L, C, V = (np.empty((n_days, n_codes)) for _ in range(5))
    for j in range(n_codes):
        px = 100.0
        for t in range(n_days):
            gap = rng.normal(0, 0.02) + (rng.normal(0, 0.08) if rng.random() < 0.05 else 0.0)
            o = px * np.exp(gap)
            c = o * np.exp(rng.normal(0.001, 0.025))
            h = max(o, c) * np.exp(abs(rng.normal(0, 0.015)))
            lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.015)))
            Op[t, j], H[t, j], L[t, j], C[t, j], V[t, j] = o, h, lo, c, rng.uniform(1e3, 1e4)
            px = c
        for t in rng.choice(n_days, size=8, replace=False):     # 정지일: 거래량 0, OHLC = 직전 종가
            if t > 0:
                Op[t, j] = H[t, j] = L[t, j] = C[t, j] = C[t - 1, j]
                V[t, j] = 0.0
    Op[250:, 1] = H[250:, 1] = L[250:, 1] = C[250:, 1] = V[250:, 1] = np.nan     # 폐지
    mk = lambda a: pd.DataFrame(a, index=days, columns=codes)  # noqa: E731
    return mk(Op), mk(H), mk(L), mk(C), mk(V)


def _events(o, h, lo, c, n_ev: int = 400, seed: int = 1, pivot_window: int = 20):
    rng = np.random.default_rng(seed)
    days, codes = c.index, c.columns
    di = rng.integers(15, len(days), size=n_ev)
    ci = rng.integers(0, len(codes), size=n_ev)
    piv = h.shift(1).rolling(pivot_window, min_periods=pivot_window).max().to_numpy()[di, ci]
    clow = contraction_low_level(lo, 10).to_numpy()[di, ci]
    ev = pd.DataFrame({"code": codes[ci], "event_day": days[di], "pivot": piv, "contraction_low": clow})
    return ev.drop_duplicates(["code", "event_day"]).reset_index(drop=True)


EXIT_COMBOS = [
    {},                                                                   # 원전 프로토콜
    {"stop_cap": 0.06, "take_r": 1.5, "ts_days": 10, "max_hold": 20},
    {"stop_cap": 0.15, "take_r": 4.0, "ts_days": None, "max_hold": 60},
    {"stop_cap": 0.08, "take_r": 3.0, "ts_days": 30, "max_hold": 120, "has_trail": False},
    {"stop_cap": 0.12, "take_r": 1.5, "ts_days": None, "max_hold": 20, "has_trail": False},
    {"fb": True},                                                         # 돌파 실패 + 원전(시간 손절 함께)
    {"stop_cap": 0.08, "take_r": 3.0, "ts_days": None, "max_hold": 60, "fb": True},            # GA2 모양(시간 손절 없음)
    {"stop_cap": 0.15, "take_r": 1.5, "ts_days": None, "max_hold": 120, "has_trail": False, "fb": True},
]


def _ref_minervini(o, h, lo, c, v, ma, ev, p: dict, chase=0.05):
    return simulate_minervini(o, h, lo, c, ma, ev, volume=v, stop_floor=p["stop_floor"], stop_cap=p["stop_cap"],
                              stop_mult=p["stop_mult"], take_r=p["take_r"], time_stop_days=p["ts_days"],
                              max_hold=p["max_hold"], chase=chase, limit_up_skip=True,
                              failed_breakout=p.get("fb", False))


def _compare(eng: pd.DataFrame, ref: pd.DataFrame) -> None:
    a = eng.sort_values(["event_day", "code"]).reset_index(drop=True)
    b = ref.sort_values(["event_day", "code"]).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 20
    assert (a["code"].to_numpy() == b["code"].to_numpy()).all()
    assert (pd.to_datetime(a["event_day"]).to_numpy() == pd.to_datetime(b["event_day"]).to_numpy()).all()
    assert (pd.to_datetime(a["entry_day"]).to_numpy() == pd.to_datetime(b["entry_day"]).to_numpy()).all()
    assert (pd.to_datetime(a["exit_day"]).to_numpy() == pd.to_datetime(b["exit_day"]).to_numpy()).all()
    assert np.abs(a["ret"].to_numpy() - b["ret"].to_numpy(float)).max() < 1e-9
    assert (a["exit_reason"].to_numpy() == b["exit_reason"].to_numpy()).all()
    assert (a["hold_days"].to_numpy() == b["hold_days"].to_numpy(int)).all()


@pytest.mark.parametrize("combo", range(len(EXIT_COMBOS)))
@pytest.mark.parametrize("seed", [0, 3])
def test_engine_matches_simulate_minervini_python_path(combo, seed):
    o, h, lo, c, v = _market(seed=seed)
    p = {**gf.EXIT_PREREG, **EXIT_COMBOS[combo]}
    ma = c.rolling(10 if combo % 2 else 50, min_periods=5).mean()
    ma_used = ma if p["has_trail"] else ma * np.nan
    ev = _events(o, h, lo, c, seed=seed + 1)
    book = _Book(o, h, lo, c, ma_used, v)
    eng = py(gf.engine_events, book, ev, p, py=True)
    ref = _ref_minervini(o, h, lo, c, v, ma_used, ev, p)
    _compare(eng, ref)
    reasons = set(eng["exit_reason"])
    assert {"stop"} <= reasons
    if p.get("fb"):
        assert "failed_breakout" in reasons


def test_engine_numba_path_equals_python_path():
    pytest.importorskip("numba")
    assert gf.HAVE_NUMBA
    o, h, lo, c, v = _market(seed=5, n_codes=10)
    ev = _events(o, h, lo, c, n_ev=800, seed=6)
    for combo in EXIT_COMBOS:
        p = {**gf.EXIT_PREREG, **combo}
        ma = c.rolling(20, min_periods=5).mean()
        book = _Book(o, h, lo, c, ma if p["has_trail"] else ma * np.nan, v)
        a = py(gf.engine_events, book, ev, p, py=True)
        b = gf.engine_events(book, ev, p, py=False)
        pd.testing.assert_frame_equal(a, b)
        _compare(b, _ref_minervini(o, h, lo, c, v, book_ma(book, c), ev, p))


def book_ma(book, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(book.MA.T, index=like.index, columns=like.columns)


def test_all_exit_reasons_covered_somewhere():
    seen = set()
    for seed in range(4):
        o, h, lo, c, v = _market(seed=seed)
        ev = _events(o, h, lo, c, seed=seed + 10)
        for combo in EXIT_COMBOS:
            p = {**gf.EXIT_PREREG, **combo}
            book = _Book(o, h, lo, c, c.rolling(10, min_periods=5).mean(), v)
            seen |= set(py(gf.engine_events, book, ev, p, py=True)["exit_reason"])
    assert {"stop", "breakeven", "trail", "time_stop", "time", "failed_breakout"} <= seen


@pytest.mark.parametrize("seed", [0, 2])
def test_full_trail_without_take_profit_matches_simulate_sepa(seed):
    """익절 없음 ∧ 추적 MA ∧ 고정 손절 8% ∧ 시간 손절 끔 = ``simulate_sepa`` (수축 저점이 정의되고 시가가 그 위인 사건)."""
    o, h, lo, c, v = _market(seed=seed)
    ma = c.rolling(20, min_periods=5).mean()
    ev = _events(o, h, lo, c, seed=seed + 4)
    nxt = o.shift(-1).to_numpy()[c.index.get_indexer(ev["event_day"]), c.columns.get_indexer(ev["code"])]
    ev = ev[np.isfinite(ev["contraction_low"]) & ~(nxt <= ev["contraction_low"].to_numpy())].reset_index(drop=True)
    p = {**gf.EXIT_PREREG, "fixed_stop": 0.08, "take_r": np.inf, "ts_days": None, "max_hold": 60, "has_trail": True}
    book = _Book(o, h, lo, c, ma, v)
    eng = py(gf.engine_events, book, ev, p, py=True)
    ref = simulate_sepa(o, h, lo, c, ma, ev, volume=v, stop=0.08, max_hold=60, chase=0.05, limit_up_skip=True)
    ref = ref.assign(exit_reason=ref["exit_reason"].replace({"ma50": "trail"}))
    _compare(eng, ref)


def test_greedy_keeps_exit_day_signal_and_drops_holding_signal():
    ok = np.array([True, True, True, True])
    pc = np.array([0, 0, 0, 1])
    pdd = np.array([1, 3, 5, 3])
    x = np.array([5, 4, 9, 4])
    sel = py(gf._greedy_py, ok, pc, pdd, x, 2, 0, 4)
    assert list(sel) == [0, 2, 3]          # 3일 신호는 보유 중(청산 5) → 버림, 5일(청산일 당일) 신호는 허용


def test_hash_rng_is_bitwise_same_in_python_and_numba():
    x = np.random.default_rng(0).normal(0.01, 0.05, 300)
    a = py(gf._boot_means_py, x, 50, 7)
    if gf.HAVE_NUMBA:
        np.testing.assert_array_equal(a, gf.boot_means(x, 50, 7))
    assert np.array_equal(a, py(gf._boot_means_py, x, 50, 7))
    assert not np.array_equal(a, py(gf._boot_means_py, x, 50, 8))


def test_boot_lower_shape_close_to_optimization_boot_lower():
    x = np.random.default_rng(1).normal(0.004, 0.06, 400)
    ours = py(gf.boot_lower, x, n_boot=500, seed=0, py=True)
    ref = _boot_lower(x, n_boot=500, seed=0)
    se = x.std(ddof=1) / np.sqrt(len(x))
    assert abs(ours - ref) < 0.4 * se
    assert ours < x.mean()
    assert gf.boot_lower(x[:4]) == -1.0


def test_boot_ratio_and_block_boot_basic():
    num = np.full(50, 0.02)
    den = np.full(50, 10.0)
    assert py(gf.boot_lower_ratio, num, den, py=True) == pytest.approx(0.002)
    m = np.full(24, 0.01)
    v = py(gf._block_boot_py, m, 3, 100, 0)
    assert np.allclose(v, 0.12)


def test_draw_controls_same_stratum_exclude_self_fixed():
    strat = np.array([0, 0, 0, 1, 1, -1, 2], np.int64)
    ret = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 99.0, 5.0])
    ok = np.array([True, True, True, True, True, True, True])
    members, ms = gf.stratum_index(strat, ok)
    out = py(gf._draw_controls_py, strat, ret, members, ms, 5, 0)
    assert out[3] == 20.0 and out[4] == 10.0           # 층 1: 자기 제외하면 상대 하나뿐
    assert np.isnan(out[5]) and np.isnan(out[6])       # 층 없음 / 자기뿐
    assert 2.0 <= out[0] <= 3.0 and 1.0 <= out[1] <= 3.0
    if gf.HAVE_NUMBA:
        np.testing.assert_array_equal(out, gf.draw_controls(strat, ret, members, ms, 5, 0))


def _port_inputs(seed=0, o=None):
    if o is None:
        o = _market(seed=seed, n_codes=12)
    op, h, lo, c, v = o
    ev = _events(op, h, lo, c, n_ev=600, seed=seed + 2)
    book = _Book(op, h, lo, c, c.rolling(10, min_periods=5).mean(), v)
    col_of = pd.Series(np.arange(len(c.columns)), index=c.columns)
    ev = ev.assign(d=c.index.get_indexer(ev["event_day"]), ci=col_of.reindex(ev["code"]).to_numpy())
    ev = ev.sort_values(["d", "code"]).reset_index(drop=True)
    pc, pdd = ev["ci"].to_numpy(np.int64), ev["d"].to_numpy(np.int64)
    st, x, ret, r1, r2, dd, pab, rs = py(gf.sim_book, book.O, book.H, book.L, book.C, book.MA, book.tradable,
                                         np.ascontiguousarray(book.present[:, -1]), pc, pdd,
                                         ev["contraction_low"].to_numpy(float), gf.EXIT_PREREG, py=True)
    n_days = len(c.index)
    day_start = np.searchsorted(pdd, np.arange(n_days + 1)).astype(np.int64)
    rng = np.random.default_rng(seed)
    cidx = np.full((len(ev), 5), -1, np.int32)        # 대조 = 같은 신호일의 체결 쌍(실제 층도 날짜를 포함한다)
    for i in range(len(ev)):
        same = np.flatnonzero((pdd == pdd[i]) & (st == gf.ST_OK))
        if len(same):
            cidx[i] = rng.choice(same, size=5)
    cost_tab = np.column_stack([np.full(n_days, 0.004), np.full(n_days, 0.003)])
    return dict(book=book, pc=pc, pdd=pdd, st=st, x=x.astype(np.int64), ret=ret, r1=r1.astype(float),
                r2=r2.astype(float), pab=pab.astype(np.int64), day_start=day_start, n_days=n_days,
                rel=rng.uniform(0.5, 2, len(ev)), rs=rng.uniform(0, 100, len(ev)), cidx=cidx, cost_tab=cost_tab,
                market=rng.integers(0, 2, len(ev)).astype(np.int8))


def _run_port(inp, slots=10, cut=None, t1=None, py=True):
    fill = inp["st"] == gf.ST_OK
    sig = np.ones(len(fill), bool)
    cutv = np.full(len(fill), inp["n_days"] + 1 if cut is None else cut, np.int32)
    f = (lambda *a: globals()["py"](gf._portfolio_py, *a)) if py else gf.portfolio
    return f(sig, fill, sig.copy(), inp["pc"], inp["pdd"], inp["day_start"], 0, inp["n_days"] if t1 is None else t1,
             inp["rel"], inp["rs"], inp["x"], inp["pab"], inp["ret"], inp["r2"], inp["book"].O, inp["book"].C,
             inp["cidx"], inp["cost_tab"], inp["market"], cutv, slots, 0.1)


def _excess(inp, sel, cut=None, py=True):
    cutv = np.full(len(inp["x"]), inp["n_days"] + 1 if cut is None else cut, np.int32)
    f = (lambda *a: globals()["py"](gf._trade_excess_py, *a)) if py else gf.trade_excess
    return f(np.asarray(sel, np.int64), cutv, inp["pc"], inp["pdd"], inp["x"], inp["pab"], inp["ret"], inp["r2"],
             inp["book"].O, inp["book"].C, inp["cidx"], inp["cost_tab"], inp["market"])


def test_portfolio_daily_sum_equals_trade_excess_and_respects_slots():
    inp = _port_inputs()
    daily, opened, n_ord, n_unm = _run_port(inp, slots=3)
    ex, hold, cost, rr, xx = _excess(inp, opened)
    assert n_unm == 0 and n_ord >= len(opened) > 10
    assert daily.sum() == pytest.approx(0.1 * (ex - cost).sum(), abs=1e-9)
    ctl = inp["ret"][inp["cidx"][opened]].mean(axis=1)
    np.testing.assert_allclose(ex, inp["ret"][opened] - ctl, atol=1e-12)
    for t in range(inp["n_days"]):
        live = opened[(inp["pdd"][opened] + 1 <= t) & (inp["x"][opened] >= t)]
        assert len(live) <= 3                         # 청산일 당일 신호의 진입은 다음날부터
        assert len(set(inp["pc"][live])) == len(live)   # 같은 종목 보유 중 신호 버림
    if gf.HAVE_NUMBA:
        d2, o2, n2, u2 = _run_port(inp, slots=3, py=False)
        np.testing.assert_allclose(d2, daily, atol=1e-12)
        np.testing.assert_array_equal(o2, opened)


def test_portfolio_priority_rel_then_rs():
    inp = _port_inputs(seed=1)
    _, opened, _, _ = _run_port(inp, slots=1)
    d0 = inp["pdd"][0]
    same = np.flatnonzero(inp["pdd"] == d0)
    best = same[np.argmin(inp["rel"][same])]
    if inp["st"][best] == gf.ST_OK:
        assert opened[0] == best
    else:
        assert opened[0] != best


def test_portfolio_unmatched_excluded_from_pnl_and_cost():
    inp = _port_inputs(seed=3)
    inp["cidx"][::3] = -1
    daily, opened, _, n_unm = _run_port(inp, slots=4)
    ex, hold, cost, rr, xx = _excess(inp, opened)
    m = np.isfinite(ex)
    assert n_unm == int((~m).sum()) > 0
    assert daily.sum() == pytest.approx(0.1 * (ex[m] - cost[m]).sum(), abs=1e-9)


def test_mark_to_market_cut_uses_no_later_prices_and_matches_portfolio():
    """학습창 끝 절단: cut 이후 가격을 바꿔(다시 시뮬) 도 cut 전 진입 트레이드의 끊은 수익·대조·포트폴리오가 같다."""
    o = _market(seed=4, n_codes=10)
    cut = 200
    inp = _port_inputs(seed=4, o=o)
    op, h, lo, c, v = (x.copy() for x in o)
    rng = np.random.default_rng(9)
    for x in (op, h, lo, c):
        x.iloc[cut:] *= rng.uniform(0.5, 1.5, size=x.iloc[cut:].shape)
    h.iloc[cut:] = np.maximum(h.iloc[cut:], np.maximum(op.iloc[cut:], c.iloc[cut:]))
    lo.iloc[cut:] = np.minimum(lo.iloc[cut:], np.minimum(op.iloc[cut:], c.iloc[cut:]))
    inp2 = _port_inputs(seed=4, o=(op, h, lo, c, v))
    for k in ("cidx", "market", "rel", "rs"):
        inp2[k] = inp[k]
    pre = np.flatnonzero((inp["pdd"] + 1 < cut) & (inp["st"] == gf.ST_OK))
    a = _excess(inp, pre, cut=cut)
    b = _excess(inp2, pre, cut=cut)
    for x1, x2 in zip(a, b, strict=True):
        np.testing.assert_allclose(x1, x2, atol=1e-12)
    assert (a[4] <= cut - 1).all() and (a[4] < inp["x"][pre]).any()
    d1, o1, _, _ = _run_port(inp, cut=cut, t1=cut)
    d2, o2, _, _ = _run_port(inp2, cut=cut, t1=cut)
    np.testing.assert_allclose(d1, d2, atol=1e-12)
    o1 = o1[inp["pdd"][o1] + 1 < cut]                 # cut 행 진입(마지막 신호일)은 창 밖 — 실제 창은 신호에서 뺀다
    ex, _, cost, _, _ = _excess(inp, o1, cut=cut)
    assert d1.sum() == pytest.approx(0.1 * (ex - cost).sum(), abs=1e-9)


def test_mark_partial_before_cut_keeps_half():
    Op = np.array([[0.0, 100.0, 100.0, 100.0, 100.0]])
    C = np.array([[0.0, 101.0, 110.0, 120.0, 90.0]])
    x = np.array([4])
    pab = np.array([2])
    ret = np.array([0.5 * (0.15 + (-0.10))])     # 절반 +15% (2행), 절반 −10% (4행)
    r2 = np.array([-0.10])
    pc, pdd = np.array([0]), np.array([0])
    r, xe, m, pa = py(gf._mark_py, 0, 4, pc, pdd, x, pab, ret, r2, Op, C)     # 끊는 날 = 3행 종가 120
    assert xe == 3 and pa == 2 and m == pytest.approx(0.20) and r == pytest.approx(0.5 * (0.15 + 0.20))
    r, xe, m, pa = py(gf._mark_py, 0, 2, pc, pdd, x, pab, ret, r2, Op, C)     # 익절(2행) 전 = 1행 종가
    assert xe == 1 and pa == -1 and r == pytest.approx(0.01)
    r, xe, m, pa = py(gf._mark_py, 0, 9, pc, pdd, x, pab, ret, r2, Op, C)     # 끊지 않음
    assert xe == 4 and r == ret[0]


def test_draw_idx_matches_draw_controls_mean():
    rng = np.random.default_rng(0)
    strat = rng.integers(-1, 6, 300).astype(np.int64)
    ret = rng.normal(0, 0.05, 300)
    ok = rng.random(300) < 0.8
    members, ms = gf.stratum_index(strat, ok)
    mean = py(gf._draw_controls_py, strat, ret, members, ms, 5, 0)
    idx = py(gf._draw_idx_py, strat, members, ms, 5, 0)
    has = idx[:, 0] >= 0
    np.testing.assert_array_equal(has, np.isfinite(mean))
    np.testing.assert_allclose(ret[idx[has]].mean(axis=1), mean[has], atol=1e-12)
    assert all(strat[j] == strat[i] and j != i for i in np.flatnonzero(has) for j in idx[i])
    if gf.HAVE_NUMBA:
        np.testing.assert_array_equal(idx, gf.draw_idx(strat, members, ms, 5, 0))


def test_mask_ga2_off_ends_and_eps_genes():
    n = 6
    nh = np.array([0.9, 0.9, 0.9, 0.9, 0.9, np.nan], np.float32)
    al = np.full(n, 1.5, np.float32)
    stack = np.full(n, 3, np.int8)
    slope = np.full(n, 7, np.int8)
    rs = np.full(n, 80.0)
    brk = np.full(n, 31, np.int8)
    vol = np.full(n, 2.0)
    vavg = np.ones((n, 5))
    pead = np.array([0.8, 0.8, 0.8, 0.8, np.nan, 0.8])
    rel = np.full(n, 1.0)
    br = np.full(n, 0.6)
    band = np.zeros(n, np.int8)
    ok = np.ones(n, bool)
    eps = np.array([0.5, 0.1, np.nan, 0.5, 0.5, 0.5])
    acc = np.array([1, 1, 1, 0, 1, 1], np.int8)
    rev = np.array([0.2, 0.2, 0.2, 0.2, 0.2, -0.3])
    off = np.full(13, np.nan)
    off[[2, 3, 5, 11]] = [0, 0, -1, 0]
    assert py(gf._mask_ga2_py, 0, n, nh, al, stack, slope, rs, brk, vol, vavg, pead, rel, br, band, off, -1, ok,
                           eps, acc, rev).all()                              # 전부 끔 → NaN 행도 통과
    g = off.copy()
    g[10], g[11], g[12] = 0.3, 1, 0.0                                         # EPS ≥ 30% ∧ 가속 ∧ 매출 ≥ 0
    m = py(gf._mask_ga2_py, 0, n, nh, al, stack, slope, rs, brk, vol, vavg, pead, rel, br, band, g, -1, ok, eps, acc, rev)
    assert list(m) == [True, False, False, False, True, False]
    g2 = off.copy()
    g2[8], g2[0] = 0.5, 0.8                                                   # 켠 조건의 NaN 은 탈락
    m2 = py(gf._mask_ga2_py, 0, n, nh, al, stack, slope, rs, brk, vol, vavg, pead, rel, br, band, g2, -1, ok, eps, acc, rev)
    assert list(m2) == [True, True, True, True, False, False]
