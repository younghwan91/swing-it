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


@pytest.mark.parametrize("nan_share", [0.0, 0.3])
def test_n2s_keeps_finite_count_and_value_multiset_every_day(nan_share):
    U, pred, _ = _world(nan_share=nan_share)
    days = sorted(U)
    out = NL.n2s_walk(pred, U, days, 4, N)
    for t in days:
        assert np.isfinite(out[t]).sum() == np.isfinite(pred[t]).sum()
        assert np.array_equal(np.sort(out[t][np.isfinite(out[t])]), np.sort(pred[t][np.isfinite(pred[t])]))


def test_n2s_keeps_pairs_while_slot_and_source_stay():
    """U(t) 가 안 바뀌면 첫날 짝이 끝까지 이어진다 — 값이 바뀌어도 원천을 그대로 따라간다(N2 와 다른 점)."""
    idx = np.arange(0, 60)
    rng = np.random.default_rng(1)
    days = list(range(10))
    U = {t: idx for t in days}
    pred = {t: rng.normal(0, 1, idx.size) for t in days}
    out = NL.n2s_walk(pred, U, days, 2, N)
    # 첫날 값으로 짝을 복원한다
    order = {v: j for j, v in enumerate(pred[0])}
    src0 = np.array([order[v] for v in out[0]])
    assert not np.array_equal(src0, np.arange(idx.size))
    for t in days[1:]:
        assert np.array_equal(out[t], pred[t][src0])


def test_n2s_drops_pair_when_source_or_slot_leaves_and_repairs_stored():
    idx = np.arange(0, 40)
    rng = np.random.default_rng(2)
    days = [0, 1, 2]
    pred = {t: rng.normal(0, 1, idx.size) for t in days}
    pred[1][5] = np.nan  # 원천 5 가 하루 결측
    U = {0: idx, 1: idx, 2: idx}
    out = NL.n2s_walk(pred, U, days, 3, N)
    assert np.isfinite(out[1]).sum() == idx.size - 1
    # 그날 끊긴 자리 말고는 이튿날도 첫날 짝 그대로다
    src0 = np.array([{v: j for j, v in enumerate(pred[0])}[v] for v in out[0]])
    kept = src0 != 5
    assert np.array_equal(out[1][kept], pred[1][src0[kept]])
    assert np.array_equal(out[2][kept], pred[2][src0[kept]])


def test_n2s_same_key_same_result_and_grid_separate():
    U, pred, _ = _world()
    days = sorted(U)
    a = NL.n2s_walk(pred, U, days, 9, N)
    b = NL.n2s_walk(pred, U, days, 9, N)
    assert all(a[t].tobytes() == b[t].tobytes() for t in days)
    assert any(a[t].tobytes() != NL.n2s_walk(pred, U, days, 10, N)[t].tobytes() for t in days)
    # 북 격자는 따로 걷는다 — 부분열의 첫날은 처음부터 다시 짝짓는다
    sub = days[3:]
    assert NL.n2s_walk(pred, U, sub, 9, N)[sub[0]].tobytes() != a[sub[0]].tobytes()


def test_n2s_is_more_persistent_than_n2():
    """지속성 주입 — 예측이 지속되면 N2s 는 그 지속성을 N2 보다 많이 남긴다(이 널을 더한 이유)."""
    rng = np.random.default_rng(5)
    days = list(range(30))
    pop = np.arange(150)
    base = rng.normal(0, 1, pop.size)
    U, pred = {}, {}
    for t in days:
        idx = np.sort(rng.choice(pop, 70, replace=False))
        U[t] = idx
        pred[t] = base[idx] + rng.normal(0, 0.1, idx.size)
    u_star = np.unique(np.concatenate(list(U.values())))

    def rank_ac(series):
        vals = []
        for t0, t1 in zip(days[:-1], days[1:]):
            c, i0, i1 = np.intersect1d(U[t0], U[t1], return_indices=True)
            a, b = series[t0][i0], series[t1][i1]
            ok = np.isfinite(a) & np.isfinite(b)
            vals.append(C.spearman(a[ok], b[ok]))
        return float(np.mean(vals))

    s = NL.n2s_walk(pred, U, days, 1, N)
    pi = NL.n2_perm(u_star, 1, N)
    n2 = {t: NL.n2_assign(pred[t], U[t], pi, 1, t, N)[0] for t in days}
    assert rank_ac(s) > rank_ac(n2) + 0.1
