"""PBO(CSCV) — 합성 데이터: 잡음뿐이면 ≈ 0.5, 한 후보가 지배하면 ≈ 0, 조합 수·입력 검사."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from swing_it.diagnostics.pbo import _rank_avg, pbo_cscv


def test_pure_noise_pbo_near_half():
    vals = []
    for seed in range(4):
        M = np.random.default_rng(seed).normal(size=(160, 40))
        vals.append(pbo_cscv(M, n_blocks=16)["pbo"])
    assert 0.35 < np.mean(vals) < 0.65


def test_dominant_column_pbo_near_zero():
    M = np.random.default_rng(1).normal(size=(160, 40))
    M[:, 7] += 1.5                       # 한 칸만 진짜 엣지
    for metric in ("sharpe", "mean"):
        r = pbo_cscv(M, n_blocks=16, metric=metric)
        assert r["pbo"] < 0.02
        assert r["best_freq"][7] > 0.95


def test_overfit_column_pbo_high():
    """IS 에서만 좋고 OOS 에서 최악인 칸이 IS 최고로 뽑히면 PBO 가 1 쪽으로."""
    rng = np.random.default_rng(2)
    M = rng.normal(scale=0.1, size=(16, 10))
    M[:, 0] = np.where(np.arange(16) % 2 == 0, 5.0, -5.0)   # 조각마다 크게 흔들리는 칸
    r = pbo_cscv(M, n_blocks=16, metric="mean")
    # 칸 0 이 IS 최고로 뽑힌 조합에서 OOS 는 대부분 최하
    picked = r["is_best"] == 0
    assert picked.any() and np.mean(r["oos_rank"][picked] <= 1.0) > 0.9
    assert r["pbo"] > 0.5


def test_combination_count_and_shapes():
    M = np.random.default_rng(0).normal(size=(40, 5))
    r = pbo_cscv(M, n_blocks=8)
    assert r["n_combos"] == comb(8, 4) == len(r["logits"]) == len(r["oos_rank"])
    assert r["n_candidates"] == 5 and r["best_freq"].sum() == pytest.approx(1.0)
    assert np.all((r["oos_rank"] >= 1) & (r["oos_rank"] <= 5))


def test_constant_zero_columns_do_not_blow_up():
    M = np.zeros((32, 4))
    M[:, 1] = np.random.default_rng(0).normal(size=32)
    r = pbo_cscv(M, n_blocks=4, metric="sharpe")
    assert np.all(np.isfinite(r["logits"]))


def test_nan_treated_as_zero():
    M = np.random.default_rng(3).normal(size=(32, 6))
    M2 = M.copy()
    M2[::5, 2] = np.nan
    M3 = M.copy()
    M3[::5, 2] = 0.0
    np.testing.assert_array_equal(pbo_cscv(M2, n_blocks=8)["logits"], pbo_cscv(M3, n_blocks=8)["logits"])


def test_bad_inputs():
    M = np.zeros((10, 3))
    with pytest.raises(ValueError):
        pbo_cscv(M, n_blocks=3)
    with pytest.raises(ValueError):
        pbo_cscv(M, n_blocks=12)
    with pytest.raises(ValueError):
        pbo_cscv(M, metric="median")


def test_rank_avg_ties_and_nan():
    np.testing.assert_array_equal(_rank_avg(np.array([3.0, 1.0, 3.0, np.nan])), [3.5, 2.0, 3.5, 1.0])
