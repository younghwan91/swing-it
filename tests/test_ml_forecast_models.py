"""ML 예측 모델 — 결정성·누출 주입·라벨 섞기·릿지 식(DB 불요, 합성 패널)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("krx_quant_core.risk.factor_model")
lgb = pytest.importorskip("lightgbm")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import ml_forecast_common as C  # noqa: E402
import ml_forecast_models as MD  # noqa: E402

K = len(C.FEATURES)


def _xy(seed=0, n_days=60, n=300, leak=False):
    rng = np.random.default_rng(seed)
    days = list(range(0, n_days * 5, 5))
    U, Z, y = {}, {}, {}
    N = n + 50
    vol = np.ones((N, n_days * 5 + 2))
    for t in days:
        idx = np.sort(rng.choice(N, n, replace=False))
        U[t] = idx
        yt = np.full(N, np.nan)
        yt[idx] = rng.normal(0, 0.1, n)
        z = rng.normal(0, 1, (n, K))
        if leak:
            z[:, 0] = C.rank_gauss(yt[idx] + rng.normal(0, 0.02, n))
        z[rng.random(z.shape) < 0.05] = np.nan
        y[t] = yt
        Z[t] = z
    return days, U, Z, y, vol


def _fit(days, U, Z, y, vol, seed=1, cols=MD.M_COLS):
    X, Y, w, ids = MD.train_matrix(days, Z, U, y, vol)
    return MD.fit_lgb(X, Y, w, seed, cols), (X, Y, w, ids)


def test_recipe_is_prereg():
    assert C.LGB_ROUNDS == 300
    p = C.LGB_PARAMS
    assert (p["learning_rate"], p["num_leaves"], p["max_depth"], p["min_data_in_leaf"]) == (0.05, 15, 4, 500)
    assert (p["feature_fraction"], p["bagging_fraction"], p["bagging_freq"], p["lambda_l2"]) == (0.7, 0.7, 1, 10.0)
    assert p["deterministic"] and p["force_row_wise"] and p["num_threads"] == 1
    assert C.SEEDS == (1, 2, 3, 4, 5) and C.RIDGE_ALPHA == 0.1
    assert lgb.__version__ == C.LGB_VERSION


def test_same_seed_same_prediction_bytes():
    days, U, Z, y, vol = _xy()
    a, _ = _fit(days[:40], U, Z, y, vol)
    b, _ = _fit(days[:40], U, Z, y, vol)
    t = days[50]
    assert MD.predict_lgb(a, Z[t]).tobytes() == MD.predict_lgb(b, Z[t]).tobytes()


def test_different_seed_differs():
    days, U, Z, y, vol = _xy()
    a, _ = _fit(days[:40], U, Z, y, vol, seed=1)
    b, _ = _fit(days[:40], U, Z, y, vol, seed=2)
    t = days[50]
    assert MD.predict_lgb(a, Z[t]).tobytes() != MD.predict_lgb(b, Z[t]).tobytes()


def test_leak_injection_gives_high_oos_ic_and_shuffling_kills_it():
    """피처 = y + 잡음이면 OOS IC > 0.5(배선 확인). 라벨을 날짜 안에서 섞으면 |IC| 가 작다(GUARDRAILS §4.5).

    **맹점을 고정한 시험이다.** 누출이 있어도 N1 은 0 근처다 — 그래서 N1 은 누출을 잡지 못한다. 학습 경로 스모크로만 읽는다(docs/PIPELINE.md 5단계).
    """
    days, U, Z, y, vol = _xy(leak=True)
    train, test = days[:40], days[44:]
    bst, (X, Y, w, ids) = _fit(train, U, Z, y, vol)
    ics = [C.spearman(MD.predict_lgb(bst, Z[t]), y[t][U[t]]) for t in test]
    assert np.mean(ics) > 0.5
    Ys = MD.shuffle_within_dates(Y, ids, lambda t: C.keyed_rng("test-N1", 0, t))
    bs = MD.fit_lgb(X, Ys, w, 1)
    ics0 = [C.spearman(MD.predict_lgb(bs, Z[t]), y[t][U[t]]) for t in test]
    assert abs(np.mean(ics0)) < 0.05


def test_shuffle_within_dates_preserves_each_date_multiset():
    days, U, Z, y, vol = _xy()
    X, Y, w, ids = MD.train_matrix(days[:10], Z, U, y, vol)
    Ys = MD.shuffle_within_dates(Y, ids, lambda t: C.keyed_rng("x", 1, t))
    assert not np.array_equal(Y, Ys)
    for t in np.unique(ids):
        assert np.array_equal(np.sort(Y[ids == t]), np.sort(Ys[ids == t]))


def test_train_matrix_filters_and_weights_dates_equally():
    days, U, Z, y, vol = _xy()
    d0, d1 = days[0], days[1]
    y[d0][U[d0][:50]] = np.nan  # 라벨 없음
    vol[U[d1][:30], d1 + 1] = 0.0  # 진입 불가
    X, Y, w, ids = MD.train_matrix([d0, d1], Z, U, y, vol)
    n0, n1 = (ids == d0).sum(), (ids == d1).sum()
    assert n0 == 250 and n1 == 270
    assert w[ids == d0].sum() == pytest.approx(w[ids == d1].sum())
    assert w.mean() == pytest.approx(1.0)
    # 날짜마다 rank-gauss
    assert abs(Y[ids == d0].mean()) < 1e-9 and Y[ids == d0].std() == pytest.approx(Y[ids == d1].std(), rel=0.02)


def test_ridge_closed_form():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (500, 4))
    X[0, 0] = np.nan
    y = X[:, 1] * 0.5 + rng.normal(0, 0.1, 500)
    w = rng.uniform(0.5, 1.5, 500)
    b = MD.fit_ridge(X, y, w)
    A = np.where(np.isfinite(X), X, 0.0)
    W = np.diag(w)
    expect = np.linalg.solve(A.T @ W @ A + C.RIDGE_ALPHA * 500 * np.eye(4), A.T @ W @ y)
    assert np.allclose(b, expect)
    assert MD.predict_ridge(b, X)[0] == pytest.approx(A[0] @ b)


def test_scores_E_R_P_columns():
    Z = np.zeros((2, K))
    Z[0, C.FEATURES.index("rev21")] = 2.0
    Z[1, C.FEATURES.index("eps_yoy")] = 3.0
    assert MD.score_R(Z)[0] == -2.0
    assert MD.score_P(Z)[1] == 3.0
    assert MD.score_E(Z)[1] == pytest.approx(3.0 / 6)


def test_average_requires_all_seeds():
    with pytest.raises(AssertionError):
        MD.average_seed_preds({1: {0: np.zeros(1)}})
