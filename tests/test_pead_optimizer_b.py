"""PEAD 최적화 단계 B 하네스 — 합성 패리티·마스크·λ 맞춤·비용 회계(DB 불요, portfolio-layer 필요)."""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("krx_quant_core.portfolio")
pytest.importorskip("cvxpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import pead_optimizer_b as B  # noqa: E402
from krx_quant_core.costs.impact import trade_cost_bp  # noqa: E402

SECTORS = ("G10", "G20", "G30")
NAMES = ["market", *SECTORS, "size", "momentum"]


def _ctx(seed=0, nN=160, nD=330, gaps=True, delist=True, u_share=0.92):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=nD)
    r = rng.normal(0.0003, 0.02, (nN, nD))
    C = 1000.0 * np.exp(np.cumsum(r, axis=1))
    tv = rng.uniform(1.5e10, 2e11, (nN, nD))  # 원
    tv[: nN // 8] = rng.uniform(1e9, 1.5e10, (nN // 8, nD))  # ADV 미달 종목
    delist_after = np.full(nN, 10**9, dtype=np.int64)
    if gaps:
        for i in rng.choice(nN, 12, replace=False):  # 거래정지(가격 없음)
            s = rng.integers(130, nD - 30)
            C[i, s: s + rng.integers(1, 15)] = np.nan
    if delist:
        for i in rng.choice(nN, 6, replace=False):  # 폐지
            s = int(rng.integers(160, nD - 30))
            C[i, s + 1:] = np.nan
            delist_after[i] = s
    tv[~np.isfinite(C)] = np.nan
    cap = np.exp(rng.normal(27, 1.0, (nN, 1))) * np.exp(np.cumsum(rng.normal(0, 0.01, (nN, nD)), axis=1))
    sig = np.full((nN, nD), np.nan)
    for i in range(nN):  # 분기마다 바뀌는 YoY, 일부 결측
        cuts = np.sort(rng.choice(np.arange(1, nD), 5, replace=False))
        start = 0
        for c in list(cuts) + [nD]:
            if rng.random() > 0.1:
                sig[i, start:c] = rng.normal(0.1, 0.5)
            start = c
    ret = np.full_like(C, np.nan)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    lab = np.array([SECTORS[i % 3] for i in range(nN)], dtype=object)[:, None].repeat(nD, axis=1)
    adv = B._trailing_adv(tv / 1e6, nD, 20, C)
    t0 = int(np.searchsorted(dates, np.datetime64("2019-06-03")))
    grid = list(range(t0, nD, B.STEP))
    models = {}
    for t in grid:
        idx = np.flatnonzero((rng.random(nN) < u_share) & np.isfinite(C[:, t]))
        X = np.zeros((idx.size, len(NAMES)))
        X[:, 0] = 1.0
        for g, s in enumerate(SECTORS):
            X[:, 1 + g] = lab[idx, t] == s
        X[:, 4:] = rng.normal(0, 1, (idx.size, 2))
        A = rng.normal(0, 0.004, (len(NAMES), len(NAMES)))
        F = A @ A.T + np.diag(np.full(len(NAMES), 1e-5))
        D = rng.uniform(1e-4, 9e-4, idx.size)
        models[t] = (idx, X, F, D, NAMES)
    market = np.array(["KOSPI" if i % 2 else "KOSDAQ" for i in range(nN)], dtype=object)
    return B.Ctx(dates=dates, C=C, tv_won=tv, adv=adv, cap=cap, sig=sig, ret=ret, lab=lab, market=market, grid=grid,
                 models=models, delist_after=delist_after)


# ── 마스크 ──────────────────────────────────────────────────────────────────────


def test_eligible_is_cap_rank_21_100_inside_m0():
    ctx = _ctx(1)
    t = ctx.grid[3]
    in_u = ctx.in_u(t)
    m0 = B.m0_mask(t, ctx.sig, ctx.adv, in_u) & np.isfinite(ctx.cap[:, t])
    order = np.flatnonzero(m0)[np.argsort(-ctx.cap[m0, t], kind="stable")]
    el = B.eligible(t, ctx.sig, ctx.adv, ctx.cap, in_u)
    assert list(el) == list(order[20:100])
    assert not np.any(~in_u[el]) and np.all(ctx.adv[el, t] >= B.ADV_FLOOR) and np.all(np.isfinite(ctx.sig[el, t]))
    base = np.isfinite(ctx.sig[:, t]) & (ctx.adv[:, t] >= B.ADV_FLOOR)
    assert B.u_exclusion(t, ctx.sig, ctx.adv, in_u) == pytest.approx((base & ~in_u).sum() / base.sum())


def test_parity_signal_finite_only_on_grid_m0():
    ctx = _ctx(2)
    sp = B.parity_signal(ctx.sig, ctx.adv, ctx.grid, ctx.in_u)
    off = np.setdiff1d(np.arange(ctx.C.shape[1]), ctx.grid)
    assert not np.isfinite(sp[:, off]).any()
    for t in ctx.grid:
        assert np.array_equal(np.isfinite(sp[:, t]), B.m0_mask(t, ctx.sig, ctx.adv, ctx.in_u(t)))


# ── 패리티(PO-1.4) ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_parity_harness_equals_core_1e10(seed):
    ctx = _ctx(seed)
    pc = B.parity_check(ctx)
    assert pc["steps_harness"] >= 6 and pc["same_steps"]
    assert pc["max_abs_diff_book"] <= 1e-10 and pc["max_abs_diff_bench"] <= 1e-10
    assert pc["pass"]


def test_parity_catches_injected_error(monkeypatch):
    """통과만 보는 시험은 규칙이 죽어도 초록이다 — 트랜치를 9종목으로 바꾸면 패리티가 실패해야 한다."""
    ctx = _ctx(0)
    orig = B.tranche_names

    def nine(*a, **k):
        out = orig(*a, **k)
        return None if out is None else out[:9]

    monkeypatch.setattr(B, "tranche_names", nine)
    assert not B.parity_check(ctx)["pass"]


def test_real_engine_reduces_to_parity_without_gaps():
    """가격 결측·폐지가 없으면 실제 엔진(지연 0·나머지 규칙 끔)의 R 이 패리티 북과 같다 — 트랜치 재동일가중·벤치 회계 확인."""
    ctx = _ctx(4, gaps=False, delist=False)
    par = B.parity_book(ctx, 0).set_index("t")
    fl = B.Flags(lag=0, drift=True, forced_sale=False, delist_cash=False, no_trade=False, costs=False)
    df = B.step_returns(B.simulate(ctx, "R", flags=fl), 1.0).set_index("t")
    common = par.index.intersection(df.index)
    assert len(common) >= 6
    np.testing.assert_allclose(df.loc[common, "net"], par.loc[common, "book"], atol=1e-12)
    np.testing.assert_allclose(df.loc[common, "bench"], par.loc[common, "bench"], atol=1e-12)


# ── α·IC ───────────────────────────────────────────────────────────────────────


def test_rank_z_and_alpha():
    z = B.rank_z(np.array([3.0, np.nan, 1.0, 2.0, 2.0]))
    assert z[1] == 0.0 and z[2] < z[3] == z[4] < z[0]
    assert z[0] == pytest.approx(-z[2])
    a = B.alpha_vec(0.05, np.array([4e-4]), np.array([1.0]), 60)
    assert a[0] == pytest.approx(0.05 * math.sqrt(60 * 4e-4))


def test_fold_ic_purge_and_clip():
    days = np.arange(100)
    ics = np.where(days < 50, 0.04, 0.5)
    assert B.fold_ic(days, ics, fold_start=49 + 62, h=60) == pytest.approx(0.04)  # 0..49 만(62일 퍼지)
    assert B.fold_ic(days, ics, fold_start=200, h=60) == pytest.approx(0.10)  # 자르기
    assert B.fold_ic(days, -ics, fold_start=200, h=60) == 0.0


# ── λ 맞춤 ─────────────────────────────────────────────────────────────────────


def test_lambda_monotone_and_match():
    from krx_quant_core.portfolio.optimize import Constraints, optimize

    ctx = _ctx(5)
    t = ctx.grid[4]
    el = B.eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t))
    sub = B.sub_model(ctx.models[t], el, 60)
    b = np.full(el.size, 1.0 / el.size)
    alpha = B.alpha_vec(0.05, sub.D / 60, B.rank_z(ctx.sig[el, t]), 60)
    Sb = sub.X @ (sub.F @ (sub.X.T @ b)) + sub.D * b
    con = Constraints(budget=1.0, long_only=True, max_weight=0.10, sector=ctx.lab[el, t], benchmark=b, sector_band=0.10)

    def solve(lam):
        return optimize(alpha + lam * Sb, sub, None, constraints=con, risk_aversion=lam)

    risks = [B.active_risk(sub, solve(lam).weights - b) for lam in (0.1, 1.0, 10.0, 100.0, 1000.0)]
    assert all(r1 >= r2 - 1e-9 for r1, r2 in zip(risks, risks[1:]))
    target = 0.5 * (risks[1] + risks[3])
    res, lam, hit, n, cause = B.match_lambda(solve, lambda w: B.active_risk(sub, w - b), target)
    assert hit and cause == "" and abs(B.active_risk(sub, res.weights - b) / target - 1) <= B.LAM_TOL
    res, lam, hit, n, cause = B.match_lambda(solve, lambda w: B.active_risk(sub, w - b), 10 * risks[0])  # 못 닿는 목표
    assert not hit and res is not None and n == B.LAM_ITERS and cause == "up"


# ── 비용 회계 ──────────────────────────────────────────────────────────────────


def test_trade_costs_match_manual_formula():
    dw = np.array([0.05, -0.03, 0.0])
    adv = np.array([5e10, 3e10, 1e10])
    vol = np.array([0.02, 0.03, 0.01])
    mk = np.array(["KOSPI", "KOSDAQ", "KOSPI"], dtype=object)
    hs = np.array([5.0, 15.0, 5.0])
    day = date(2022, 6, 2)
    c = B.trade_costs(dw, adv, vol, mk, day, hs, 3e8)
    buy = (1.5 + 5.0 + 1e4 * 0.02 * math.sqrt(0.05 * 3e8 / 5e10)) / 1e4 * 0.05
    sell = (1.5 + 23.0 + 15.0 + 1e4 * 0.03 * math.sqrt(0.03 * 3e8 / 3e10)) / 1e4 * 0.03  # KOSDAQ 2022 0.23%
    assert c[0] == pytest.approx(buy) and c[1] == pytest.approx(sell) and c[2] == 0.0
    # 매도세는 체결일로 고른다(2023 KOSDAQ 0.20%)
    c23 = B.trade_costs(dw, adv, vol, mk, date(2023, 6, 1), hs, 3e8)
    assert c[1] - c23[1] == pytest.approx(3.0 / 1e4 * 0.03)
    # 선형 부분 = 충격 없는 trade_cost_bp
    lin = B.linear_bp(np.array(["SELL", "BUY"]), mk[:2], day, hs[:2])
    ref = trade_cost_bp(np.array(["SELL", "BUY"]), 0.0, adv[:2], vol[:2], mk[:2], day, half_spread_bp=hs[:2])
    np.testing.assert_allclose(lin, ref)


def test_engine_costs_first_build_and_2x():
    ctx = _ctx(6)
    recs = B.simulate(ctx, "R")
    r0 = recs[0]
    f = r0.t + 1
    w = np.zeros(ctx.C.shape[0])
    for i, wi, _, _ in r0.contrib:
        w[i] = wi
    vol = B.trailing_vol(ctx.ret, r0.t)
    manual = B.trade_costs(w, ctx.adv[:, r0.t] * 1e6, np.nan_to_num(vol), ctx.market, ctx.dates[f].date(), ctx.hs_bp, 3e8)
    assert r0.cost1 == pytest.approx(manual.sum())  # 현금에서 구축: Δw = 목표 비중
    assert r0.turnover == pytest.approx(1.0)
    df1, df2 = B.step_returns(recs, 1.0), B.step_returns(recs, 2.0)
    g = np.array([r.gross for r in recs])
    c1 = np.array([r.cost1 for r in recs])
    np.testing.assert_allclose(df2["net"], (1 - 2 * c1) * (1 + g) - 1)
    assert np.all(df2["net"] <= df1["net"] + 1e-15)
    # 스텝마다 비중 합 1(현금 포함) — 총액 경로
    for r in recs:
        assert sum(wi for _, wi, _, _ in r.contrib) <= 1.0 + 1e-12


def test_delisting_goes_to_cash_with_sell_cost():
    ctx = _ctx(8, gaps=False, delist=False)
    recs = B.simulate(ctx, "R")
    k = 3
    held = [i for i, wi, _, _ in recs[k].contrib if wi > 0]
    i = held[0]
    f_next = ctx.grid[k + 1] + 1
    ctx.C[i, f_next - 2:] = np.nan  # 다음 체결 전에 폐지
    ctx.delist_after[i] = f_next - 3
    ctx.Cff = B._ffill_axis1_nb(ctx.C)
    recs2 = B.simulate(ctx, "R")
    assert recs2[k + 1].n_delist == 1
    assert all(j != i or wi == 0 for j, wi, _, _ in recs2[k + 1].contrib)


# ── O 통합(작은 합성) ──────────────────────────────────────────────────────────


def test_optimized_book_constraints_and_lambda_match():
    ctx = _ctx(9, nN=140, nD=290, delist=False, u_share=1.0)  # 강제 매도 없음 → 회전율 상한이 그대로 보인다
    sigma_r = B.r_sigma_series(ctx)
    ic = {y: 0.10 for y in range(2018, 2025)}
    recs = B.simulate(ctx, "O", ic_by_year=ic, sigma_r=sigma_r)
    assert len(recs) >= 5
    ok = [r for r in recs if r.status == "optimal"]
    assert len(ok) == len(recs)
    assert np.mean([r.lam_hit for r in recs]) >= 0.5  # 못 닿는 목표(제약·비용)는 실패로 센다
    for r in recs:
        w = np.array([wi for _, wi, _, _ in r.contrib])
        assert w.max() <= B.MAX_W + 1e-6 and w.sum() <= 1 + 1e-9
        if r.lam_hit:
            assert abs(r.exante_20 / sigma_r[r.t] - 1) <= B.LAM_TOL + 1e-9
    assert all(r.turnover <= B.MAX_TURNOVER + 0.05 for r in recs[1:])  # 체결일 표류 차이만큼 여유
    assert all(np.isfinite(r.factor_share) for r in recs)


# ── 지표·판정 ──────────────────────────────────────────────────────────────────


def _summ(ir2, ir1, sd, ev, opt=1.0, lamf=0.0, n=48):
    return {"IR_2x": ir2, "IR_1x": ir1, "sd_a_2x": sd, "EV_2x": ev, "optimal_share": opt, "lam_fail_share": lamf,
            "oos_steps": n}


def test_verdict_rules():
    null = np.linspace(-1, 0.5, 300)
    v = B.verdict(_summ(0.8, 0.9, 0.02, 0.002), _summ(0.9, 1.0, 0.02, 0.002), null, null, True)
    assert v["x"] == pytest.approx(0.30) and v["verdict"] == "비열등 통과"
    v = B.verdict(_summ(0.5, 0.9, 0.02, 0.002), _summ(0.9, 1.0, 0.02, 0.002), null, null, True)
    assert v["verdict"] == "채택 불가" and v["failed"] == ["B-NI-1"]
    v = B.verdict(_summ(0.8, 0.9, 0.03, 0.002), _summ(0.9, 1.0, 0.02, 0.002), null, null, True)
    assert "B-NI-3" in v["failed"]
    v = B.verdict(_summ(0.8, 0.9, 0.02, 0.002), _summ(0.3, 0.4, 0.02, 0.002), null, null, True)
    assert v["verdict"] == "판단 불가(신호 약함)"  # IR(R) 0.3 < 널 95% 0.425
    v = B.verdict(_summ(0.8, 0.9, 0.02, 0.002, opt=0.9), _summ(0.9, 1.0, 0.02, 0.002), null, null, True)
    assert v["verdict"] == "VOID"
    v = B.verdict(_summ(0.8, 0.9, 0.02, 0.002), _summ(0.9, 1.0, 0.02, 0.002), null, null, False)
    assert v["verdict"] == "VOID"


def test_ir_and_intervals():
    a = np.array([0.01, -0.005, 0.02, 0.0])
    assert B.ir(a) == pytest.approx(a.mean() / a.std(ddof=1) * math.sqrt(252 / 20))
    recs = [B.StepRec(k=k, t=k, date="2021-01-01", gross=0, cost1=0, bench=0, turnover=0, exante_20=0, factor_share=0)
            for k in range(4)]
    recs[0].contrib = [(1, 0.1, 0.10, 0.001)]
    recs[1].contrib = [(1, 0.1, -0.05, 0.0)]
    recs[2].contrib = [(1, 0.0, 0.0, 0.002)]  # 다 팔았다
    recs[3].contrib = [(1, 0.1, 0.02, 0.001)]  # 새 구간
    iv = B.holding_intervals(recs, 1.0)
    assert sorted(iv) == pytest.approx(sorted([0.1 * 0.10 - 0.001 + 0.1 * -0.05 - 0.002, 0.1 * 0.02 - 0.001]))


# ── PO-7: λ 실패 원인·먼지·불변식·관문·캐시 이름 ────────────────────────────────


class _Res:
    def __init__(self, ok, w):
        self.ok, self.weights, self.est_cost = ok, np.array([w]), 0.0


def _fake(risk_of_lam, fail=lambda lam: False):
    calls = []

    def solve(lam):
        calls.append(lam)
        return _Res(not fail(lam), risk_of_lam(lam))

    return solve, calls


def test_match_lambda_causes():
    risk = lambda w: float(w[0])  # noqa: E731
    solve, _ = _fake(lambda lam: 1.0 / lam)
    assert B.match_lambda(solve, risk, 1.0 / B.LAM_LO * 2)[4] == "up"  # 가장 작은 λ 에서도 위험이 모자란다
    assert B.match_lambda(solve, risk, 1.0 / B.LAM_HI / 2)[4] == "down"
    solve, calls = _fake(lambda lam: 1.0 / lam, fail=lambda lam: True)
    out = B.match_lambda(solve, risk, 1.0)
    assert out[0] is None and out[4] == "solver" and len(calls) == B.LAM_ITERS  # 실패에서 멈추지 않는다
    solve, _ = _fake(lambda lam: 2.0 if lam < 1.0 else 0.5)  # 계단 — 목표 1.0 을 사이에 두지만 창은 없다
    out = B.match_lambda(solve, risk, 1.0)
    assert not out[2] and out[4] == "gap"
    # 중간중간 실패해도 계속해서 맞춘다
    n_fail = [0]

    def flaky(lam):
        n_fail[0] += 1
        return n_fail[0] % 3 == 0

    solve, _ = _fake(lambda lam: 1.0 / lam, fail=flaky)
    out = B.match_lambda(solve, risk, 0.37)
    assert out[2] and abs(out[0].weights[0] / 0.37 - 1) <= B.LAM_TOL


def test_clean_dust_zeroes_tiny_keeps_sum_and_fixed():
    w = np.array([0.30, 0.2999995, 1e-9, -3.5e-10, 0.4, 0.0005])
    price = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 5e5])  # 마지막: 1주 = 5e5/3e8 = 0.00167 > 0.0005 → 먼지
    fixed = np.array([False, False, False, False, True, False])
    out, n = B.clean_dust(w, price, 3e8, fixed, cap=None)  # 먼지 규칙만(상한은 아래 시험)
    assert n == 3 and out[2] == 0 and out[3] == 0 and out[5] == 0
    assert out[4] == 0.4 and out.sum() == pytest.approx(w.sum())
    assert (out >= 0).all()


def test_capped_rescale_respects_cap_and_leaves_cash_when_full():
    x = np.array([0.10, 0.10, 0.05, 0.05])
    out = B.capped_rescale(x, 0.40, 0.10)  # 0.30 → 0.40: 0.10 두 칸은 못 늘고, 나머지가 0.10 까지
    assert (out <= 0.10 + 1e-15).all() and out.sum() == pytest.approx(0.40)
    full = B.capped_rescale(np.array([0.10, 0.09]), 0.5, 0.10)  # 다 차면 남는다(현금)
    assert np.allclose(full, [0.10, 0.10])
    down = B.capped_rescale(np.array([0.10, 0.06]), 0.08, 0.10)  # 줄이는 방향은 비례
    assert np.allclose(down, [0.05, 0.03])
    assert np.allclose(B.capped_rescale(np.array([0.2, 0.1]), 0.6, None), [0.4, 0.2])  # R: 상한 없음


def test_clean_dust_redistribution_never_breaks_cap():
    # 실측(B 1차 실행 2026-11-02): 먼지 재분배가 상한 칸을 0.100134 까지 밀었다 — PO-9
    w = np.array([0.10, 0.10, 0.10, 0.0005, 0.0995])
    price = np.array([1e4, 1e4, 1e4, 5e5, 1e4])
    out, n = B.clean_dust(w, price, 3e8, np.zeros(5, bool))
    assert n == 1 and (out <= B.MAX_W + 1e-15).all() and out.sum() == pytest.approx(w.sum())


def test_model_cache_name_binds_fingerprint_and_core():
    a = B.model_cache_name([1, "2023-12-28", 2, 3], "13c56a7")
    assert a != B.model_cache_name([1, "2023-12-28", 2, 4], "13c56a7")
    assert a != B.model_cache_name([1, "2023-12-28", 2, 3], "deadbee")
    assert a.startswith("models_v4_grid_fp") and a.endswith("_core13c56a7.pkl")


def test_invariants_clean_on_synthetic_books_and_catch_injection():
    ctx = _ctx(9, nN=140, nD=290, delist=False, u_share=1.0)
    recs_r = B.simulate(ctx, "R")
    assert B.check_invariants(recs_r, "R") == []
    sigma_r = B.r_sigma_series(ctx)
    recs_o = B.simulate(ctx, "O", ic_by_year={y: 0.10 for y in range(2018, 2025)}, sigma_r=sigma_r)
    assert B.check_invariants(recs_o, "O") == []
    # 먼지 제거 뒤 O 보유 종목 수는 회전율·제약이 허용하는 만큼 — 1e-9 비중이 '보유'로 세지지 않는다
    for r in recs_o:
        assert all(w == 0 or w >= B.DUST_EPS for _, w, _, _ in r.contrib)
    assert sum(r.n_dust for r in recs_o) > 0  # 합성에서도 솔버 먼지가 실제로 나온다
    # 주입: 비중 상한·현금·트랜치·비용
    import copy

    bad = copy.deepcopy(recs_o)
    bad[2].max_w = 0.2
    bad[3].cash += 0.01
    bad[4].cost1 = bad[4].turnover * 0.5e-4  # 0.5bp/회전 < 수수료
    v = B.check_invariants(bad, "O")
    assert any("max w" in x for x in v) and any("Σw+현금" in x for x in v) and any("비용/회전" in x for x in v)
    bad_r = copy.deepcopy(recs_r)
    bad_r[5].n_tranches = 2
    assert any("트랜치" in x for x in B.check_invariants(bad_r, "R"))
    bad[5].turnover = bad[5].turn_bound + 0.01
    assert any("회전율" in x for x in B.check_invariants(bad, "O"))


def test_gate_b_refuses():
    import pead_optimizer_b_report as BR

    fp = [1, "2026-09-11", 2, 3]
    ok = {"verdict": "통과", "fingerprint_before": {"daily_bars_adjusted": fp},
          "repaired": {"label": None, "B_opens": True, "fingerprint_differs_from_pre": True}}
    B.gate_b(ok, fp, BR.IMPLEMENTED)
    with pytest.raises(SystemExit, match="미구현"):
        B.gate_b(ok, fp, {**BR.IMPLEMENTED, "f_boot": False})
    with pytest.raises(SystemExit, match="A1_repaired"):
        B.gate_b(None, fp, BR.IMPLEMENTED)
    with pytest.raises(SystemExit, match="취약"):
        B.gate_b({**ok, "verdict": "실패"}, fp, BR.IMPLEMENTED)
    with pytest.raises(SystemExit, match="지문"):
        B.gate_b(ok, [1, "2026-09-11", 2, 4], BR.IMPLEMENTED)
    with pytest.raises(SystemExit, match="강건성 분기"):
        B.gate_b({**ok, "repaired": {"label": "취약", "B_opens": False, "fingerprint_differs_from_pre": True}}, fp, BR.IMPLEMENTED)
    with pytest.raises(SystemExit, match="강건성 분기"):
        B.gate_b({k: v for k, v in ok.items() if k != "repaired"}, fp, BR.IMPLEMENTED)


def test_daily_rank_ic_keeps_names_delisted_inside_window():
    rng = np.random.default_rng(3)
    nN, nD, h = 150, 140, 20
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, (nN, nD)), axis=1))
    adv = np.full((nN, nD), 1e5)
    cap = np.exp(rng.normal(27, 1, (nN, nD)))
    sig = rng.normal(0, 1, (nN, nD))
    days, ics = B.daily_rank_ic(close, adv, cap, sig, 0, 5, h)
    t = int(days[0])
    el = B.eligible_idx(t, sig, adv, cap, np.ones(nN, bool))
    gone = el[:5]
    close2 = close.copy()
    close2[gone, t + 10:] = np.nan  # 창 안에서 폐지
    _, ics2 = B.daily_rank_ic(close2, adv, cap, sig, t, t, h)
    cff = B._ffill_axis1_nb(close2)
    ret = cff[el, t + 1 + h] / close2[el, t + 1] - 1
    ref = np.corrcoef(stats_rank(sig[el, t]), stats_rank(ret))[0, 1]
    assert ics2[0] == pytest.approx(ref)  # 폐지 종목도 마지막 종가로 들어간다(80종목 전부)


def stats_rank(v):
    from scipy import stats

    return stats.rankdata(v)


# ── PO-8: λ 실패 원인별 VOID·IC 0 폴드 ─────────────────────────────────────────


def test_verdict_po8_up_failures_reported_not_voided():
    null = np.linspace(-1, 0.5, 300)
    o = {**_summ(0.8, 0.9, 0.02, 0.002, lamf=0.30), "lam_up_share": 0.30, "lam_fail_nonup_share": 0.0,
         "exante_20_mean": 0.03}
    r = {**_summ(0.9, 1.0, 0.025, 0.002), "exante_20_mean": 0.04}
    v = B.verdict(o, r, null, null, True)
    assert v["verdict"] == "비열등 통과" and v["risk_label"] == B.LABEL_RISK_UP
    assert v["realized_active_risk_ratio_2x"] == pytest.approx(0.8) and v["exante_risk_ratio"] == pytest.approx(0.75)
    v = B.verdict({**o, "lam_fail_nonup_share": 0.11}, r, null, null, True)
    assert v["verdict"] == "VOID" and v["void"]["lam_fail(solver·gap·down)>0.10"]
    v = B.verdict({**o, "lam_up_share": 0.0, "lam_fail_share": 0.0}, r, null, null, True)
    assert v["risk_label"] is None
    # 옛 요약(원인 없음)은 보수적으로 전체 실패율로 VOID 를 본다
    assert B.verdict(_summ(0.8, 0.9, 0.02, 0.002, lamf=0.2), r, null, null, True)["verdict"] == "VOID"


def test_ic_zero_fold_label_and_mask():
    null = np.linspace(-1, 0.5, 300)
    v = B.verdict(_summ(0.8, 0.9, 0.02, 0.002), _summ(0.9, 1.0, 0.02, 0.002), null, null, True, [2021])
    assert v["ic_zero_folds"] == [2021] and B.LABEL_IC0 in v["ic_zero_label"]
    s = B.mask_ic_zero_folds({"folds_IR_2x": {"2020": 0.1, "2021": 0.5}, "folds_IR_1x": {"2021": 0.4}, "IR_2x": 0.3}, [2021])
    assert s["folds_IR_2x"] == {"2020": 0.1, "2021": B.LABEL_IC0} and s["folds_IR_1x"]["2021"] == B.LABEL_IC0
    assert s["IR_2x"] == 0.3  # 통합 항목은 그대로


def test_book_summary_splits_lambda_causes():
    recs = [B.StepRec(k=k, t=k, date="2021-01-05", gross=0.0, cost1=0.0, bench=0.0, turnover=0.0, exante_20=0.03,
                      factor_share=0.05) for k in range(10)]
    for r, (hit, cause) in zip(recs, [(False, "up")] * 3 + [(False, "gap")] + [(True, "")] * 6):
        r.lam_hit, r.lam_cause = hit, cause
    s = B.book_summary(recs)
    assert s["lam_fail_share"] == pytest.approx(0.4) and s["lam_up_share"] == pytest.approx(0.3)
    assert s["lam_fail_nonup_share"] == pytest.approx(0.1) and s["lam_causes"] == {"up": 3, "down": 0, "gap": 1, "solver": 0}
