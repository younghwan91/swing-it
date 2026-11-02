"""PEAD 최적화 단계 B 보고만 항목(PO-7) — 합성 데이터로만(DB 불요, portfolio-layer 필요)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("krx_quant_core.portfolio")
pytest.importorskip("cvxpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pead_optimizer_b as B  # noqa: E402
import pead_optimizer_b_report as BR  # noqa: E402
from krx_quant_core.risk.factor_model import factor_returns  # noqa: E402
from krx_quant_core.stats.metrics import newey_west_t  # noqa: E402
from test_pead_optimizer_b import _ctx  # noqa: E402

# ── HAC 회귀 ────────────────────────────────────────────────────────────────


def test_ols_hac_intercept_only_equals_core_newey_west():
    x = np.random.default_rng(0).normal(0.01, 0.05, 48)
    r = BR.ols_hac(x, np.empty((48, 0)))
    mu, t = newey_west_t(x, 2)
    assert r["alpha"] == pytest.approx(mu) and r["t_alpha"] == pytest.approx(t)


def test_ols_hac_recovers_alpha_beta():
    rng = np.random.default_rng(1)
    F = rng.normal(0, 0.05, (4000, 3))
    y = 0.002 + F @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.01, 4000)
    r = BR.ols_hac(y, F)
    assert r["alpha"] == pytest.approx(0.002, abs=5e-4)
    np.testing.assert_allclose(r["beta"], [1.0, -0.5, 0.3], atol=0.02)
    assert r["t_alpha"] > 5 and r["n"] == 4000
    # 결측 행은 뺀다
    y2 = y.copy()
    y2[:10] = np.nan
    assert BR.ols_hac(y2, F)["n"] == 3990


def test_hurdle_and_qualification():
    assert BR.alpha_hurdle(1)["hurdle"] == 2.0  # 1.96 < 2.0 바닥
    h = BR.alpha_hurdle(49)
    assert h["hurdle"] == pytest.approx(h["bonferroni_t"]) and 3.2 < h["hurdle"] < 3.4
    assert BR.pead_qualification(1.0, 2.5, 3.0)["label"] == BR.LABEL_QUAL
    assert BR.pead_qualification(3.1, 0.0, 3.0)["label"] is None
    assert BR.pead_qualification(float("nan"), float("nan"), 3.0)["label"] == BR.LABEL_QUAL


def test_ledger_n_sums_ledgers_and_meta_gen3():
    counts = {"pead_optimizer": 7, "pead_concentrated": 29}
    led = BR.ledger_n(lambda lab: counts[lab])
    assert led["N"] == 7 + 29 + 13 and led["parts"]["META_GEN3(상수)"] == 13


# ── 스텝 창·팩터 ────────────────────────────────────────────────────────────


def test_compound_and_long_short_sign():
    r = np.array([np.nan, 0.01, 0.02, np.nan, -0.01])
    v, miss = BR.compound(r, 0, 4)
    assert v == pytest.approx(1.01 * 1.02 * 0.99 - 1) and miss == 1
    nN = 40
    C = np.ones((nN, 3))
    C[:, 2] = 1.0 + np.linspace(-0.1, 0.1, nN)  # 수익이 점수와 같은 순서
    score = np.linspace(0, 1, nN)
    ls = BR.long_short(score, np.arange(nN), C, C, 0, 2)
    assert ls > 0
    assert BR.long_short(-score, np.arange(nN), C, C, 0, 2) == pytest.approx(-ls)


def _recs_and_factors(seed=4):
    ctx = _ctx(seed, nN=160, nD=330)
    recs = B.simulate(ctx, "R")
    kospi = np.random.default_rng(seed).normal(0.0003, 0.01, ctx.C.shape[1])
    kosdaq = np.random.default_rng(seed + 1).normal(0.0003, 0.012, ctx.C.shape[1])
    fac = BR.factor_step_returns(ctx, recs, kospi, kosdaq, B.trailing_vol, B.ADV_FLOOR)
    return ctx, recs, fac, kospi


def test_factor_step_returns_windows_and_alpha_report():
    ctx, recs, fac, kospi = _recs_and_factors()
    assert list(fac.columns[:2]) == ["t", "date"] and len(fac) == len(recs)
    r = recs[3]
    assert fac.loc[3, "KOSPI"] == pytest.approx(np.prod(1 + kospi[r.f + 1:r.f_next + 1]) - 1)
    # 북 = KOSPI + 상수 이면 알파 = 상수, β_KOSPI = 1
    net = pd.DataFrame({"t": fac["t"], "date": fac["date"], "net": fac["KOSPI"] + 0.001})
    fac2 = fac.copy()
    for c in ("KOSDAQ", "UMD", "SMB", "LOWVOL"):
        fac2[c] = np.random.default_rng(len(c)).normal(0, 0.02, len(fac2))
    rep = BR.factor_alpha_report(net, fac2)
    assert rep["alpha"] == pytest.approx(0.001, abs=1e-12) and rep["beta"][0] == pytest.approx(1.0)
    assert rep["alpha_ann"] == pytest.approx(0.001 * 252 / 20, rel=1e-9)
    vk = BR.vs_kospi(net, fac)
    assert vk["excess_mean"] == pytest.approx(0.001)
    d = BR.deflation(np.random.default_rng(2).normal(0.01, 0.05, 48), 49)
    assert d["n_trials"] == 49 and d["expected_max_sharpe_h0"] > 0


def test_attribution_reconciles_with_step_active_return():
    ctx, recs, _, _ = _recs_and_factors(7)
    nN, nD = ctx.C.shape
    rng = np.random.default_rng(0)
    Xs = {"size": rng.normal(0, 1, (nN, nD)), "momentum": rng.normal(0, 1, (nN, nD))}
    cap = np.repeat(ctx.cap[:, :1], nD, axis=1) if ctx.cap.shape[1] == 1 else ctx.cap
    msk = np.isfinite(ctx.C)
    prep = {"R": ctx.ret, "Xs": Xs, "lab": ctx.lab, "cap": cap, "msk": msk}
    fr = factor_returns(prep["R"], Xs, ctx.lab, cap, msk)
    rep = BR.attribution_report(ctx, recs[3:], fr, prep, lambda t: B.eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t)))
    assert rep["n_steps"] == len(recs[3:])
    assert 0.0 <= rep["specific_share_var"] <= 1.0
    # 단리 일 귀속 합 대 복리 스텝 능동수익: 일 2% 변동성·20일이면 차이는 작다
    act = np.array([r.gross - r.bench for r in recs[3:]])
    assert rep["check_mean_abs_gap"] < 0.25 * np.mean(np.abs(act)) + 2e-3
    assert set(rep["mean_by_group"]) == {"market", "sector", "style"}


# ── PO-1.3 f_boot ───────────────────────────────────────────────────────────


def test_nw_t_rows_equals_core():
    X = np.random.default_rng(5).normal(0.002, 0.03, (7, 24))
    np.testing.assert_allclose(BR.nw_t_rows(X), [newey_west_t(r, 2)[1] for r in X])


def test_max_ratio_is_sequential_rejection_threshold():
    rng = np.random.default_rng(6)
    P = rng.normal(0.0, 0.03, (300, 36))
    ratio = BR.max_ratio(P)
    for f in (1.0, 1.19, 1.5):
        brute = np.zeros(300, bool)
        for K, c in zip(BR.OBF_LOOKS, BR.OBF_SHAPE):
            t = np.array([newey_west_t(p[:K], 2)[1] for p in P])
            brute |= np.isfinite(t) & (t > f * c)
        np.testing.assert_array_equal(brute, ratio > f)


def test_f_boot_grid_and_floor():
    d = np.random.default_rng(8).normal(0.003, 0.02, 48)
    fb = BR.f_boot(d, n=20_000)
    f = fb["f_boot"]
    assert fb["alpha_at_f_boot"] <= 0.10
    assert abs(f / BR.F_GRID - round(f / BR.F_GRID)) < 1e-9
    assert fb["multiplier"] == max(1.19, f)
    assert fb["boundaries"] == [round(c * fb["multiplier"], 4) for c in BR.OBF_SHAPE]
    assert fb["T_in_sample"] == 48
    # 최소성: 같은 난수로 한 칸 아래 배수의 α 를 직접 재면 0.10 을 넘는다
    from pead_optimizer_c0 import keyed_rng

    x = d - d.mean()
    rng = keyed_rng("fboot")
    st = rng.integers(0, 48, size=(20_000, 12))
    idx = ((st[:, :, None] + np.arange(3)[None, None, :]) % 48).reshape(20_000, -1)
    ratio = BR.max_ratio(x[idx])
    assert np.mean(ratio > f) <= 0.10 < np.mean(ratio > f - BR.F_GRID)


# ── B.md ───────────────────────────────────────────────────────────────────


def _fake_res():
    summ = {"IR_2x": 0.5, "IR_1x": 0.6, "sd_a_2x": 0.02, "EV_2x": 0.001, "EV_1x": 0.0012, "step_payoff_2x": 1.1,
            "interval_payoff_2x": 1.3, "interval_EV_drop_top5_2x": 0.0, "interval_skew_2x": 0.4, "turnover_mean": 0.3,
            "cost1_mean": 0.001, "names_mean": 28.0, "realized_over_exante": 1.05, "folds_IR_2x": {"2020": 0.1},
            "factor_share_q10_50_90": [0.02, 0.05, 0.1], "factor_alarm_n": 1, "factor_alarm_dates": ["2021-03-02"],
            "factor_alarm_top3": [["2021-03-02", [["G45", 0.08]]]], "lam_causes": {"up": 1, "down": 0, "gap": 0, "solver": 0},
            "optimal_share": 1.0, "lam_fail_share": 0.02, "n_buy_limit_up": 0, "n_dust": 10, "oos_steps": 48}
    fa = {"alpha": 0.001, "alpha_ann": 0.0126, "t_alpha": 1.2, "beta": [1, 0.1, 0, 0, 0], "r2": 0.8, "n": 48}
    v = B.verdict(summ, summ, np.zeros(300), np.zeros(300), True)
    v["ic_zero_folds"] = []
    rep = {"ledger": {"N": 49, "parts": {}}, "hurdle": BR.alpha_hurdle(49),
           "factor_alpha": {"O": {"2x": fa}, "R": {"2x": fa}}, "qualification": BR.pead_qualification(1.2, 1.2, 3.3),
           "vs_kospi": {b: {"excess_mean": 0.0, "excess_IR": 0.0, "cum_book": 0.1, "cum_kospi": 0.1} for b in "OR"},
           "deflation": {b: BR.deflation(np.random.default_rng(0).normal(0, 0.05, 48), 49) for b in "OR"},
           "attribution": {b: {"specific_share_var": 0.9, "mean_factor": 0.0, "mean_specific": 0.0, "mean_by_group": {},
                               "zeroed_name_days": 0, "check_mean_abs_gap": 0.0} for b in "OR"},
           "f_boot": {"T_in_sample": 48, "block": 3, "n": 10, "f_boot": 1.2, "alpha_at_f_boot": 0.1, "alpha_at_1.19": 0.11,
                      "multiplier": 1.2, "boundaries": [2.832, 2.004, 1.632], "looks": [12, 24, 36]}}
    return {"code": {"core_ref": "13c56a7", "swing_it_sha": "abc"}, "fingerprints": {"daily_bars_adjusted_b_range": [1],
            "same_as_A1_repaired": True}, "invariants": {"steps_checked": 100},
            "primary": {"O": summ, "R": summ, "verdict": v, "rho": 0.5, "boot_dIR_q05_50_95": [-0.5, 0, 0.5], "report": rep,
                        "toggles": [{"rule": "① t+1 체결", "steps": 54, "diff_sum": 0.0, "diff_mean": 0.0, "diff_max_abs": 0.0,
                                     "events": {}}]},
            "cells": {"H20_band0.05_aum3e+08": {"O": summ}}, "elapsed_s": 1.0}


def test_render_b_has_registered_labels():
    md = BR.render_b(_fake_res())
    first = [x for x in md.splitlines() if x.strip()][1]
    assert BR.LABEL_FIRST in first  # 제목 다음 첫 줄(PO-3.6·PO-5.3)
    for s in (BR.LABEL_T1, BR.LABEL_MARKET, BR.LABEL_QUAL, "표본 안에서 유리", "열등 아님 선별", "GUARDRAILS §2", "f_boot 1.200",
              "solver·gap·down"):
        assert s in md
    assert "전제가 깨졌다" in md  # 팩터 몫 경보 문장


def test_render_b_po8_labels():
    res = _fake_res()
    v = res["primary"]["verdict"]
    v.update({"lam_up_share": 0.2, "risk_label": B.LABEL_RISK_UP, "realized_active_risk_ratio_2x": 0.8, "exante_risk_ratio": 0.75,
              "ic_zero_label": f"{B.LABEL_IC0} 폴드 [2021]"})
    md = BR.render_b(res)
    assert B.LABEL_RISK_UP in md and "0.800" in md and B.LABEL_IC0 in md


def test_render_b_stop():
    md = BR.render_b({"STOP": "x"})
    assert "멈춤: x" in md and BR.LABEL_FIRST in md


def test_implemented_flags_complete():
    assert set(BR.REQUIRED) == {"factor_alpha", "pead_qualification", "vs_kospi", "attribution", "deflation", "f_boot",
                                "render"}
    assert all(BR.IMPLEMENTED.values())
    assert math.isfinite(BR.alpha_hurdle(7)["hurdle"])
