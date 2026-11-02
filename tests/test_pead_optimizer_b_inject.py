"""B 하네스 주입 인자 — 기본값이면 옛 경로와 비트 동일, 주입하면 실제로 회계까지 닿는다(DB 불요)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("krx_quant_core.portfolio")
pytest.importorskip("cvxpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pead_optimizer_b as B  # noqa: E402
from test_pead_optimizer_b import _ctx  # noqa: E402

IC = {y: 0.10 for y in range(2018, 2025)}


def _bytes(recs):
    return np.array([[r.gross, r.cost1, r.bench, r.turnover] for r in recs]).tobytes()


def _sigma(ctx):
    return B.r_sigma_series(ctx)


def test_none_equals_explicit_b_constants_bitwise():
    ctx = _ctx()
    sr = _sigma(ctx)
    a = B.simulate(ctx, "O", ic_by_year=IC, sigma_r=sr)
    b = B.simulate(ctx, "O", ic_by_year=IC, sigma_r=sr, max_weight=B.MAX_W,
                   universe=lambda t: B.eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t)))
    assert _bytes(a) == _bytes(b)


def test_r_book_ignores_max_weight_and_stays_bitwise():
    ctx = _ctx()
    assert _bytes(B.simulate(ctx, "R")) == _bytes(B.simulate(ctx, "R", max_weight=0.05))


def test_max_weight_reaches_the_book():
    """위반 주입 — 상한을 바꾸면 비중이 실제로 묶이고, 불변식이 옛 상한 위반을 잡는다(민감도 불변성 = 버그 신호, GUARDRAILS §7.1)."""
    ctx = _ctx()
    sr = _sigma(ctx)
    tight = B.simulate(ctx, "O", ic_by_year=IC, sigma_r=sr, max_weight=0.03)
    loose = B.simulate(ctx, "O", ic_by_year=IC, sigma_r=sr)
    assert max(r.max_w for r in tight) <= 0.03 + B.INV_EPS
    assert max(r.max_w for r in loose) > 0.03 + B.INV_EPS
    assert B.check_invariants(tight, "O", max_weight=0.03) == []
    assert any("max w" in v for v in B.check_invariants(loose, "O", max_weight=0.03))


def test_universe_reaches_benchmark_and_book():
    ctx = _ctx()
    sr = _sigma(ctx)
    fixed = {t: np.flatnonzero(ctx.in_u(t))[:40] for t in ctx.grid}
    recs = B.simulate(ctx, "O", ic_by_year=IC, sigma_r=sr, universe=lambda t: fixed[t])
    t = recs[3].t
    f = recs[3].f
    fn = recs[3].f_next
    el = fixed[t]
    bel = el[np.isfinite(ctx.C[el, f])]
    assert recs[3].bench == pytest.approx(np.mean(ctx.Cff[bel, fn] / ctx.C[bel, f] - 1.0))
    assert np.isnan(recs[3].u_excl)
