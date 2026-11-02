"""ML 예측 2단계 — 바·판정·구축 비용·취약성·관문·패리티 대조·원장 N(DB 불요)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("krx_quant_core.portfolio")
pytest.importorskip("cvxpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ml_forecast_book as BK  # noqa: E402
import pead_optimizer_b as B  # noqa: E402
import pead_optimizer_b_report as BR  # noqa: E402
from test_pead_optimizer_b import _ctx  # noqa: E402


def _fr(**kw):
    d = {"EV": 0.01, "interval_payoff": 1.5, "interval_mean_drop_top5": 0.001, "interval_mean_drop_top20": 0.001,
         "step_EV_drop_top5": 0.005, "step_EV_drop_top20": 0.002, "EV_drop_top_name": 0.008}
    d.update(kw)
    return d


def test_bars_pass_and_each_failure_is_named():
    assert all(BK.bars_s2(1.0, 0.5, 0.6, _fr(), 0.5, 0.5).values())
    assert not BK.bars_s2(0.55, 0.5, 0.6, _fr(), 0.5, 0.5)["P-1"]  # N2 는 넘고 N2s 는 못 넘는다 — max 다
    assert not BK.bars_s2(0.55, 0.6, 0.5, _fr(), 0.5, 0.5)["P-1"]
    assert not BK.bars_s2(1.0, 0.5, 0.6, _fr(EV=-0.01), 0.5, 0.5)["P-2"]
    assert not BK.bars_s2(1.0, 0.5, 0.6, _fr(interval_payoff=0.99), 0.5, 0.5)["P-3"]
    for k in ("interval_mean_drop_top5", "interval_mean_drop_top20", "step_EV_drop_top5", "step_EV_drop_top20"):
        assert not BK.bars_s2(1.0, 0.5, 0.6, _fr(**{k: -0.001}), 0.5, 0.5)["P-4"], k
    assert not BK.bars_s2(1.0, 0.5, 0.6, _fr(EV_drop_top_name=0.004), 0.5, 0.5)["P-5"]  # > 0 이지만 0.5·EV 미만
    assert not BK.bars_s2(1.0, 0.5, 0.6, _fr(), 1.1, 0.5)["P-6"]
    assert not BK.bars_s2(1.0, 0.5, 0.6, _fr(), 0.5, 1.1)["P-7"]


def test_bars_nan_fails():
    b = BK.bars_s2(float("nan"), 0.5, 0.6, _fr(EV=float("nan"), interval_payoff=float("nan")), 0.5, 0.5)
    assert not any(b[k] for k in ("P-1", "P-2", "P-3", "P-5", "P-6", "P-7"))


GOOD = {f"P-{i}": True for i in range(1, 8)}


def test_verdict_table_order():
    assert BK.verdict_s2(["x"], GOOD)["verdict"] == "VOID"
    only_p1 = {**GOOD, "P-1": False}
    v = BK.verdict_s2([], only_p1)
    assert v["verdict"] == "ML 이득 없음" and v["failed"] == ["P-1"] and not v["forward_arm"]  # 옛 "판단 불가(검정력)" 없음
    assert BK.verdict_s2([], {**GOOD, "P-7": False})["verdict"] == "ML 이득 없음"
    w = BK.verdict_s2([], GOOD)
    assert w["verdict"] == "전진 후보" and w["forward_arm"]


def test_alpha_is_label_not_gate():
    """정정 ML-4a — 판정 함수가 알파를 받지 않고, 라벨만 바뀌며 '알파 주장 불가' 는 어느 쪽에도 남는다."""
    import inspect

    assert "alpha_t" not in inspect.signature(BK.verdict_s2).parameters
    for t in (1.99, float("nan")):
        assert BK.forward_label(t).startswith("스타일 미분리")
    assert not BK.forward_label(2.0).startswith("스타일 미분리")
    for t in (1.0, 2.0, 5.0):
        assert "알파 주장 불가" in BK.forward_label(t)


def test_max_drawdown_and_absolute_report():
    r = np.array([0.10, -0.20, 0.05, -0.10, 0.30])
    curve = np.r_[1.0, np.cumprod(1 + r)]
    assert BK.max_drawdown(r) == pytest.approx((curve / np.maximum.accumulate(curve) - 1).min())
    assert BK.max_drawdown(np.array([0.01, 0.02])) == 0.0
    k, q = np.full(5, 0.01), np.array([0.02, np.nan, 0.0, 0.0, 0.0])
    a = BK.absolute_report(r, k, q)
    assert a["net_step_mean"] == pytest.approx(r.mean())
    assert a["net_ann_simple"] == pytest.approx(r.mean() * 12.6)
    assert a["net_ann_geometric"] == pytest.approx(np.prod(1 + r) ** (12.6 / 5) - 1)
    assert a["vs_kospi_step_mean"] == pytest.approx(r.mean() - 0.01)
    assert a["cum_kosdaq"] == pytest.approx(0.02)


def test_fragility_math():
    a = np.array([0.02, -0.01, 0.03, 0.0])
    iv = np.array([0.04, 0.01, -0.02, 0.005, 0.003, 0.002, 0.001])
    fr = BK.fragility(a, iv, {1: 0.03, 2: 0.005, 3: -0.01})
    assert fr["EV"] == pytest.approx(0.01)
    assert fr["step_EV_drop_top5"] == pytest.approx((a.sum() - np.sort(iv)[::-1][:5].sum()) / 4)
    assert fr["interval_mean_drop_top5"] == pytest.approx(np.sort(iv)[::-1][5:].mean())
    assert np.isnan(fr["interval_mean_drop_top20"])  # 구간이 모자라면 NaN → 바 미달
    assert fr["top_name_row"] == 1 and fr["EV_drop_top_name"] == pytest.approx((0.04 - 0.03) / 4)
    assert fr["top_name_share_of_sum_a"] == pytest.approx(0.75)
    assert fr["interval_payoff"] == pytest.approx(B.payoff_ratio(iv))


def _book():
    ctx = _ctx()
    U = {t: np.flatnonzero(ctx.in_u(t) & np.isfinite(ctx.C[:, t]))[:60] for t in ctx.grid}
    rng = np.random.default_rng(0)
    sig = np.full(ctx.C.shape, np.nan)
    for t, idx in U.items():
        sig[idx, t] = rng.normal(0, 1, idx.size)
    return ctx, U, sig


def test_run_book_strips_only_first_build_cost_and_keeps_gross():
    ctx, U, sig = _book()
    recs, viol = BK.run_book(ctx, U, sig)
    raw = B.simulate(ctx, "O", B.Cell(h_alpha=BK.H_ALPHA, band=BK.BAND, aum=BK.AUM), rank_sig=sig,
                     ic_by_year={y: BK.IC0 for y in range(2018, 2025)}, sigma_r={t: BK.SIGMA_TARGET for t in ctx.grid},
                     universe=lambda t: U[t], max_weight=BK.MAX_W)
    assert viol == []
    k0 = next(k for k, r in enumerate(raw) if r.n_names > 0)
    assert raw[k0].cost1 > 0 and recs[k0].cost1 == 0.0 and recs[k0].gross == raw[k0].gross
    assert all(c == 0.0 for *_, c in recs[k0].contrib)
    assert recs[k0 + 1].cost1 == raw[k0 + 1].cost1  # 다음 스텝은 그대로
    assert raw[k0].cost1 != 0.0  # 입력을 고치지 않았다


def test_book_respects_prereg_constraints_and_ic0_reaches_accounting():
    """IC₀ 를 바꾸면 거래가 달라진다(GUARDRAILS §7.1 — 파라미터가 회계에 닿는지)."""
    ctx, U, sig = _book()
    recs, _ = BK.run_book(ctx, U, sig)
    assert max(r.max_w for r in recs) <= BK.MAX_W + B.INV_EPS
    other, _ = BK.run_book(ctx, U, sig, ic0=0.06)
    assert BK._step_bytes(recs) != BK._step_bytes(other)
    big, _ = BK.run_book(ctx, U, sig, aum=1e10)
    assert BK._step_bytes(big) != BK._step_bytes(recs)
    # AUM 이 크면 최적화기는 덜 사고팔 수 있다 — 총비용이 아니라 회전당 비용(충격)이 올라야 한다
    per_turn = lambda rr: sum(r.cost1 for r in rr[1:]) / sum(r.turnover for r in rr[1:])  # noqa: E731
    assert per_turn(big) > per_turn(recs)


def test_book_is_deterministic():
    ctx, U, sig = _book()
    assert BK._step_bytes(BK.run_book(ctx, U, sig)[0]) == BK._step_bytes(BK.run_book(ctx, U, sig)[0])


def test_panel_of_uses_risk_coordinates():
    U = {12: np.array([1, 3])}
    p = BK.panel_of({12: np.array([0.5, -0.5])}, U, [12], 10, (5, 8))
    assert p[1, 2] == 0.5 and p[3, 2] == -0.5 and np.isfinite(p).sum() == 2


def test_book_days_are_ic_grid_subset_at_20_steps():
    dates = pd.bdate_range("2019-01-02", "2023-12-28")
    import ml_forecast_common as C

    oos = C.oos_days(dates, C.ic_grid(dates))
    b = BK.book_days(dates, oos)
    assert set(b) <= set(oos) and all(y - x == 20 for x, y in zip(b, b[1:]))
    assert dates[b[0]].year == 2020 and b[-1] + C.LABEL_LAST <= len(dates) - 1


def test_persistence_gate_tolerance():
    g = BK.persistence_gate({"a": 0.6, "b": 0.47}, {"a": 0.47, "b": 0.41})
    assert g["pass"]
    assert not BK.persistence_gate({"a": 0.6}, {"a": 0.16})["pass"]  # 기존 N2 는 떨어진다


def test_parity_compare_catches_difference_and_nested():
    old = {"IR": 0.7, "f": {"2020": 1.0}, "n": float("nan")}
    assert BK.parity_compare({"IR": 0.7 + 1e-11, "f": {"2020": 1.0}, "n": float("nan")}, old)["pass"]
    r = BK.parity_compare({"IR": 0.7 + 1e-9, "f": {"2020": 1.0}, "n": float("nan")}, old)
    assert not r["pass"] and r["mismatch"][0]["path"] == ".IR"
    assert not BK.parity_compare({"IR": 0.7, "f": {}, "n": float("nan")}, old)["pass"]  # 빠진 값도 실패


def test_liq_factor_is_low_turnover_minus_high():
    ctx = _ctx()
    recs = B.simulate(ctx, "R")[3:5]
    U = {r.t: np.flatnonzero(ctx.in_u(r.t) & np.isfinite(ctx.C[:, r.f]))[:60] for r in recs}
    liq = np.random.default_rng(1).normal(0, 1, ctx.C.shape)
    n = len(ctx.dates)
    fac = BK.factor_rows(ctx, recs, U, liq, np.zeros(n), np.zeros(n))
    r = recs[0]
    assert fac["LIQ"].iloc[0] == pytest.approx(BR.long_short(-liq[:, r.t], U[r.t], ctx.Cff, ctx.C, r.f, r.f_next))
    assert set(BK.FACTORS) <= set(fac.columns)


def test_ledger_n_ml_formula():
    counts = {"ml_forecast": 15, "pead_optimizer": 7, "pead_concentrated": 29, "inst_flow_accel": 10, "catalyst_flow_swing": 7, "pullback_prop": 19}
    led = BR.ledger_n_ml(lambda lab: counts[lab])
    assert led["narrow"] == 15 + 7 + 29 + 13 + 3 + 2
    assert led["wide"] == led["narrow"] + 10 + 7 + 19 + 20 + 31


def test_s2_no_win_rate_in_outputs():
    src = (Path(BK.__file__)).read_text(encoding="utf-8")
    assert "win_rate" not in src and "hit_rate" not in src


def test_parity_compare_needs_same_key_type():
    """정수 키와 문자열 키는 빠진 값으로 잡힌다 — 러너는 JSON 왕복으로 맞춘다."""
    assert not BK.parity_compare({20: {"2020": 0.1}}, {"20": {"2020": 0.1}})["pass"]
    import json

    assert BK.parity_compare(json.loads(json.dumps({20: {"2020": 0.1}})), {"20": {"2020": 0.1}})["pass"]
