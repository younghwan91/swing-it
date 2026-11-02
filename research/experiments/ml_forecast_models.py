"""ML 수익률 예측 — 학습 행·워크포워드·모델 M·S·L, 무학습 E·R·P (PREREG §5·§6, 정정 ML-1 ⑦).

학습: 확장창, 해마다 다시 맞춘다. T0(Y) = Y 의 첫 OOS IC 격자일, 학습 = t + 21 ≤ T0(Y) 인 위상 날짜(2019-01-02~).
조기 종료·격자·내부 CV 없음. LightGBM 은 ``num_threads=1``·시드 1~5 예측 평균.
"""

from __future__ import annotations

import numpy as np
from ml_forecast_common import (
    FEATURES,
    LABEL_LAST,
    LGB_PARAMS,
    LGB_ROUNDS,
    RIDGE_ALPHA,
    S_FEATURES,
    SEEDS,
    rank_gauss,
)
from ml_forecast_features import ew_score
from ml_forecast_target import tradable

M_COLS = tuple(range(len(FEATURES)))
S_COLS = tuple(FEATURES.index(k) for k in S_FEATURES)


def t0_of(dates, grid: list[int], year: int) -> int:
    """T0(Y) = Y 의 첫 OOS IC 격자일."""
    return next(t for t in grid if dates[t].year == year)


def train_days(rows: list[int], t0: int) -> list[int]:
    """학습 날짜 — 퍼지: t + 21 ≤ T0."""
    return [t for t in rows if t + LABEL_LAST <= t0]


def train_matrix(days, Z: dict, U: dict, y: dict, vol) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X, y rank-gauss, w, 날짜 번호). 행 = U(t) ∧ y 유한 ∧ t+1 거래량 > 0. w = n̄ / n_t(정정 ML-1 ⑦)."""
    Xs, ys, ns, ids = [], [], [], []
    for t in days:
        idx = U[t]
        ok = np.isfinite(y[t][idx]) & tradable(vol, t)[idx]
        if not ok.any():
            continue
        Xs.append(Z[t][ok])
        ys.append(rank_gauss(y[t][idx][ok]))
        ns.append(int(ok.sum()))
        ids.append(np.full(int(ok.sum()), t, dtype=np.int64))
    X = np.vstack(Xs)
    Y = np.concatenate(ys)
    n = np.asarray(ns, dtype=np.float64)
    w = np.concatenate([np.full(int(k), n.mean() / k) for k in n])
    return X, Y, w, np.concatenate(ids)


def shuffle_within_dates(Y: np.ndarray, ids: np.ndarray, rng_of) -> np.ndarray:
    """N1 — 날짜 안에서 라벨을 섞는다. ``rng_of(t)`` 는 날짜별 생성기.

    **학습 경로 스모크이지 누출 검사가 아니다.** 피처가 미래와 연결되면 섞인 라벨에는 배울 것이 없어 IC 가 0 으로 나온다. 주입한 누출 6종을 하나도 못 잡았다(검증 5단계).
    """
    out = Y.copy()
    for t in np.unique(ids):
        pos = np.flatnonzero(ids == t)
        out[pos] = Y[pos][rng_of(int(t)).permutation(pos.size)]
    return out


def fit_lgb(X: np.ndarray, Y: np.ndarray, w: np.ndarray, seed: int, cols=M_COLS):
    """고정 레시피(§5). 조기 종료 없음."""
    import lightgbm as lgb

    params = dict(LGB_PARAMS)
    params["seed"] = int(seed)
    ds = lgb.Dataset(X[:, list(cols)], label=Y, weight=w, params={"verbosity": -1}, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=LGB_ROUNDS)


def predict_lgb(booster, Z_t: np.ndarray, cols=M_COLS) -> np.ndarray:
    return booster.predict(Z_t[:, list(cols)], num_threads=1)


def fit_ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    """진단 L — β = (XᵀWX + α·n·I)⁻¹XᵀWy, NaN → 0."""
    A = np.where(np.isfinite(X), X, 0.0)
    W = w[:, None]
    n = A.shape[0]
    return np.linalg.solve((A * W).T @ A + alpha * n * np.eye(A.shape[1]), (A * W).T @ Y)


def predict_ridge(beta: np.ndarray, Z_t: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(Z_t), Z_t, 0.0) @ beta


def score_E(Z_t: np.ndarray) -> np.ndarray:
    return ew_score(Z_t)


def score_R(Z_t: np.ndarray) -> np.ndarray:
    """진단 R — F3 순위(부호 −)."""
    return -Z_t[:, FEATURES.index("rev21")]


def score_P(Z_t: np.ndarray) -> np.ndarray:
    """진단 P — F1 순위."""
    return Z_t[:, FEATURES.index("eps_yoy")]


def seeds_task(args) -> tuple:
    """작업자 단위 — (모델 이름, 시험 연도, 시드, 섞기 r) 를 받아 그 연도 격자일 예측을 낸다.

    공유 입력은 fork 전 전역 :data:`CTX` 에 둔다. r 이 None 이면 실제 라벨, 아니면 N1 섞기.
    """
    name, year, seed, r, salt = args
    c = CTX
    days = c["train"][year]
    X, Y, w, ids = train_matrix(days, c["Z"], c["U"], c["y"], c["vol"])
    if r is not None:
        Y = shuffle_within_dates(Y, ids, lambda t: c["rng"]("ML-N1", r, t))
    cols = M_COLS if name.startswith("M") else S_COLS
    booster = fit_lgb(X, Y, w, seed, cols)
    pred = {t: predict_lgb(booster, c["Z"][t], cols) for t in c["test"][year]}
    return name, year, seed, r, salt, pred


CTX: dict = {}


def average_seed_preds(by_seed: dict[int, dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """시드별 예측의 평균(날짜별). 시드 집합은 §5 그대로여야 한다."""
    assert set(by_seed) == set(SEEDS), f"시드 {sorted(by_seed)} ≠ {SEEDS}"
    days = next(iter(by_seed.values())).keys()
    return {t: np.mean([by_seed[s][t] for s in SEEDS], axis=0) for t in days}
