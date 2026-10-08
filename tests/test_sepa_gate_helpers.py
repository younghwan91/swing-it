"""research/experiments/sepa_confluence_gate 의 순수 헬퍼 — 합성 데이터(DB 없음).

- L2 로지스틱 회귀(뉴턴법)가 분리 가능한 신호를 찾고, 벌점이 계수를 줄이며, 정규방정식(기울기 0)을 만족한다.
- purge 분할: 청산일 < 시험연도 1/1 만 학습.
- walk-forward: 학습 표본이 시험 연도를 넘지 않고, 선택 문턱 = 학습 예측확률 중앙값.
- 결측 대체·지시자·표준화가 학습 통계만 쓴다.
- 날짜 분위·군집 차이 CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.sepa_confluence_gate import (
    _sigmoid,
    cluster_diff_ci,
    day_quartiles,
    fit_logistic_l2,
    ml_design,
    ml_walk_forward,
    predict_logistic,
    purged_train_mask,
)


def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    z = 0.3 + 2.0 * X[:, 0] - 1.0 * X[:, 1]
    y = (rng.random(n) < _sigmoid(z)).astype(float)
    return X, y


def test_logistic_recovers_signs_and_gradient_is_zero():
    X, y = _synthetic()
    w = fit_logistic_l2(X, y, lam=1.0)
    assert w[1] > 1.0 and w[2] < -0.5 and abs(w[3]) < 0.2
    # 정규방정식: Σ(p−y)x + λw = 0 (절편은 벌점 없음)
    p = predict_logistic(w, X)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    reg = np.array([0.0, 1.0, 1.0, 1.0])
    g = Xb.T @ (p - y) + reg * w
    assert np.max(np.abs(g)) < 1e-6


def test_logistic_penalty_shrinks_coefficients():
    X, y = _synthetic(n=300, seed=1)
    w1 = fit_logistic_l2(X, y, lam=1.0)
    w100 = fit_logistic_l2(X, y, lam=100.0)
    assert np.linalg.norm(w100[1:]) < np.linalg.norm(w1[1:])


def test_logistic_constant_column_gets_zero_weight():
    X, y = _synthetic(n=400, seed=2)
    X = np.hstack([X, np.zeros((len(X), 1))])
    w = fit_logistic_l2(X, y, lam=1.0)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)


def test_purged_train_mask_uses_exit_before_test_year():
    ex = pd.Series(pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02", "2020-03-01"]))
    assert purged_train_mask(ex, 2020).tolist() == [True, True, False, False]


def test_ml_design_uses_train_stats_and_flow_indicator():
    tr = pd.DataFrame({"a": [1.0, 2.0, np.nan, 3.0], "f": [np.nan, np.nan, np.nan, np.nan]})
    te = pd.DataFrame({"a": [np.nan, 10.0], "f": [0.5, np.nan]})
    Xtr, Xte, cols = ml_design(tr, te, ["a", "f"], ["f"])
    assert cols == ["a", "f", "flow_known"]
    # a: 학습 중앙값 2 로 대체 → [1,2,2,3], 평균 2, 표준편차 sqrt(0.5)
    sd = np.sqrt(0.5)
    assert Xtr[:, 0] == pytest.approx((np.array([1, 2, 2, 3]) - 2) / sd)
    assert Xte[0, 0] == pytest.approx(0.0) and Xte[1, 0] == pytest.approx(8 / sd)
    # 학습에 수급이 전부 없으면 중앙값 0 대체 → 상수 열(표준편차 1 로 나눔), 지시자도 상수 0
    assert np.all(Xtr[:, 1] == 0) and np.all(Xtr[:, 2] == 0)
    assert Xte[0, 1] == pytest.approx(0.5) and Xte[0, 2] == pytest.approx(1.0) and Xte[1, 2] == pytest.approx(0.0)


def test_walk_forward_purges_and_selects_at_train_median():
    rng = np.random.default_rng(3)
    n = 600
    ev = pd.to_datetime("2018-01-01") + pd.to_timedelta(rng.integers(0, 365 * 4, n), unit="D")
    ex = ev + pd.to_timedelta(rng.integers(1, 90, n), unit="D")
    x = rng.normal(size=n)
    df = pd.DataFrame({"event_day": ev, "exit_day": ex, "x": x, "f": np.nan,
                       "label": (x + 0.3 * rng.normal(size=n) > 0).astype(float)})
    wf = ml_walk_forward(df, features=("x", "f"), flow_features=("f",), first_test_year=2019, last_test_year=2021)
    assert set(wf["test_year"]) == {2019, 2020, 2021}
    for y, g in wf.groupby("test_year"):
        assert (df.loc[g.index, "event_day"].dt.year == y).all()
        assert g["n_train"].iloc[0] == int((df["exit_day"] < pd.Timestamp(f"{y}-01-01")).sum())
    # 신호가 강하면 선택 집합의 라벨 평균이 전체보다 높다
    assert df.loc[wf.index[wf["selected"]], "label"].mean() > df.loc[wf.index, "label"].mean()


def test_day_quartiles_within_mask_only():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    v = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 100.0], [4.0, 3.0, np.nan, 1.0, 2.0]], index=idx, columns=list("abcde"))
    m = pd.DataFrame([[True, True, True, True, False], [True, True, True, True, False]], index=idx, columns=list("abcde"))
    q = day_quartiles(v, m)
    assert q.iloc[0, :4].tolist() == [0, 1, 2, 3] and np.isnan(q.iloc[0, 4])
    # 둘째 날은 유한 값 3개 < 4 → 전부 NaN
    assert q.iloc[1].isna().all()


def test_cluster_diff_ci_point_and_brackets():
    rng = np.random.default_rng(4)
    days = np.repeat(np.arange(200), 3)
    x1 = 0.02 + 0.01 * rng.normal(size=len(days))
    x2 = 0.00 + 0.01 * rng.normal(size=len(days))
    pt, lo, hi = cluster_diff_ci(x1, days, x2, days, n_boot=500)
    assert pt == pytest.approx(x1.mean() - x2.mean())
    assert lo < pt < hi and lo > 0.015
    assert np.isnan(cluster_diff_ci([], [], x2, days)[0])
