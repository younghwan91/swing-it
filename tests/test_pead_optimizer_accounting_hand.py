"""비용까지 넣은 회계 손계산·최적화 실패 경로 — core 충격 레이어가 필요하다(``research`` 잡, 건너뛰면 실패)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("krx_quant_core.costs.impact")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_fixtures import hand_ctx, synthetic_ctx  # noqa: E402

import swing_it.book.accounting as A  # noqa: E402
from swing_it.book.metrics import step_returns  # noqa: E402

AUM, TAX_2021_KOSPI_KOSDAQ, COMMISSION = 3e8, 23.0, 1.5
HS = {"KOSPI": 5.0, "KOSDAQ": 15.0}


def test_costs_match_the_hand_calculation():
    """수수료 1.5 + 매도세 23(2021) + 반스프레드 + 일변동성·√(|Δw|·AUM/ADV) (bp). 폐지 청산은 충격이 없다."""
    ctx, spec, steps, ix = hand_ctx()

    def vol(c, t):
        w = ctx.ret[ix[c], max(0, t - 62): t + 1]
        return float(np.std(w[np.isfinite(w)], ddof=1))

    def linear(c, sell):
        return COMMISSION + (TAX_2021_KOSPI_KOSDAQ if sell else 0.0) + HS[ctx.market[ix[c]]]

    recs = A.simulate(ctx, "R", spec=spec)
    for r, want in zip(recs, steps):
        cost = sum((linear(c, dw < 0) + vol(c, t) * math.sqrt(abs(dw) * AUM / 1e11) * 1e4) / 1e4 * abs(dw) for c, dw, t in want["trades"])
        cost += sum(linear(c, True) / 1e4 * w for c, w in want["delist"])
        assert r.cost1 == pytest.approx(cost, abs=1e-12), f"k={r.k}"
        assert r.gross == pytest.approx(want["gross"], abs=1e-12), "비중 경로는 비용과 무관하다"
    assert A.check_invariants(recs, "R", spec=spec) == []
    net = step_returns(recs, 2.0)["net"].to_numpy()
    np.testing.assert_allclose(net, [(1 - 2 * r.cost1) * (1 + r.gross) - 1 for r in recs], atol=1e-14)


def test_after_a_solver_failure_only_forced_sales_trade_and_drift_stays_consistent(monkeypatch):
    c = synthetic_ctx(0)
    n = [0]

    def fake(ctx, cell, t, f, el, w_t, held, in_u, tradeable, costable, frozen, vol, rank_sig, ic, sr, built, so, spec):
        n[0] += 1
        want = np.zeros(ctx.C.shape[0])
        if n[0] == 1:
            want[el[:12]] = 1 / 12
            return want, {"status": "optimal", "lam": 1.0, "lam_hit": True, "est_cost": 0.0}
        return want, {"status": "solver_fail", "lam": float("nan"), "lam_hit": False, "est_cost": float("nan"), "lam_cause": "solver"}

    monkeypatch.setattr(A, "o_target", fake)
    recs = A.simulate(c, "O", ic_by_year={y: 0.05 for y in range(2017, 2026)}, sigma_r={t: 0.05 for t in c.grid})
    assert recs[0].status == "optimal" and all(r.status == "solver_fail" for r in recs[1:])
    for a, b in zip(recs, recs[1:]):
        held = {i: w for i, w, _, _ in a.contrib if w > 0}
        den = a.cash + sum(w * c.Cff[i, b.f] / c.Cff[i, a.f] for i, w in held.items())
        drifted = {i: w * c.Cff[i, b.f] / c.Cff[i, a.f] / den for i, w in held.items()}
        for i, w, _, _ in b.contrib:
            if w > 0:
                assert w == pytest.approx(drifted[i], abs=1e-12), "팔지 않은 종목은 표류 비중 그대로다"
    assert A.check_invariants(recs[1:], "O") == [], "가짜 k=0 해는 밴드를 안 지켰으니 뺀다"
