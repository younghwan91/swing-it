"""ML 예측 널 N2·N3 — 채움 규칙·다중집합·재현(DB 불요, U(t) 가 날마다 바뀌는 합성 패널)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("krx_quant_core.risk.factor_model")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import ml_forecast_common as C  # noqa: E402
import ml_forecast_nulls as NL  # noqa: E402

N = 200


def _world(seed=0, days=12, nan_share=0.0):
    rng = np.random.default_rng(seed)
    U, pred = {}, {}
    for t in range(days):
        idx = np.sort(rng.choice(N - 40, 60, replace=False))  # U(t) 가 날마다 바뀐다
        U[t] = idx
        p = rng.normal(0, 1, idx.size)
        p[rng.random(idx.size) < nan_share] = np.nan
        pred[t] = p
    u_star = np.unique(np.concatenate(list(U.values())))
    return U, pred, u_star


@pytest.mark.parametrize("nan_share", [0.0, 0.3])
def test_n2_keeps_finite_count_and_value_multiset_every_day(nan_share):
    U, pred, u_star = _world(nan_share=nan_share)
    pi = NL.n2_perm(u_star, 7, N)
    for t, idx in U.items():
        out, nf = NL.n2_assign(pred[t], idx, pi, 7, t, N)
        assert np.isfinite(out).sum() == np.isfinite(pred[t]).sum()
        assert np.array_equal(np.sort(out[np.isfinite(out)]), np.sort(pred[t][np.isfinite(pred[t])]))
        assert 0 <= nf <= idx.size


def test_n2_unfilled_name_keeps_source_series():
    """π(i) ∈ U(t) 이고 유한이면 i 는 그날 p_{π(i),t} 를 그대로 받는다(같은 π 를 전 날짜에)."""
    U, pred, u_star = _world()
    pi = NL.n2_perm(u_star, 3, N)
    kept = 0
    for t, idx in U.items():
        out, _ = NL.n2_assign(pred[t], idx, pi, 3, t, N)
        full = np.full(N, np.nan)
        full[idx] = pred[t]
        for j, i in enumerate(idx):
            if np.isfinite(full[pi[i]]):
                assert out[j] == full[pi[i]]
                kept += 1
    assert kept > 0


def test_n2_perm_is_permutation_of_u_star_and_keyed():
    _, _, u_star = _world()
    p = NL.n2_perm(u_star, 1, N)
    assert np.array_equal(np.sort(p[u_star]), u_star)
    assert (np.delete(p, u_star) == -1).all()
    assert np.array_equal(p, NL.n2_perm(u_star, 1, N))
    assert not np.array_equal(p, NL.n2_perm(u_star, 2, N))


def test_n2_same_key_same_result_different_key_different():
    U, pred, u_star = _world(nan_share=0.1)
    pi = NL.n2_perm(u_star, 5, N)
    a, _ = NL.n2_assign(pred[3], U[3], pi, 5, 3, N)
    b, _ = NL.n2_assign(pred[3], U[3], pi, 5, 3, N)
    assert a.tobytes() == b.tobytes()
    c, _ = NL.n2_assign(pred[3], U[3], NL.n2_perm(u_star, 6, N), 6, 3, N)
    assert a.tobytes() != c.tobytes()


def test_n2_fill_rng_key_matters():
    """채움은 (r, t) 키다 — 같은 π 여도 r 이 다르면 채운 자리가 달라진다."""
    U, pred, u_star = _world()
    pi = NL.n2_perm(u_star, 5, N)
    a, n = NL.n2_assign(pred[3], U[3], pi, 5, 3, N)
    b, _ = NL.n2_assign(pred[3], U[3], pi, 6, 3, N)
    assert n > 0 and a.tobytes() != b.tobytes()


def test_n3_weights_and_signs_and_keys():
    w, s = NL.n3_draw(0)
    assert w.size == len(C.FEATURES) and w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert set(np.unique(s)) <= {-1.0, 1.0}
    w2, s2 = NL.n3_draw(0)
    assert np.array_equal(w, w2) and np.array_equal(s, s2)
    assert not np.array_equal(w, NL.n3_draw(1)[0])


def test_n3_scores_shape_and_missing():
    Z = {0: np.full((3, len(C.FEATURES)), np.nan)}
    Z[0][0, 1] = 1.0
    out = NL.n3_scores(Z, 0)[0]
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_q95_definition():
    assert NL.q95(np.arange(101.0)) == pytest.approx(95.0)
    assert NL.q95(np.array([np.nan, 0.0, 1.0])) == pytest.approx(0.95)
