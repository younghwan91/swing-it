"""OLS + Newey-West — 팩터 회귀 알파의 HAC t, 스텝 수익의 HAC t. numpy 만 쓴다."""

from __future__ import annotations

import numpy as np

NW_LAG = 2


def ols_hac(y, X, lag: int = NW_LAG) -> dict:
    """OLS + Newey-West(Bartlett, ``lag``) 샌드위치. 자유도 보정 없음 — 상수만이면 core ``newey_west_t`` 와 같다.

    ``X`` 는 상수 열을 포함하지 않는다(여기서 붙인다). 유한하지 않은 행은 뺀다.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    Z = np.column_stack([np.ones(len(y)), X])
    n, p = Z.shape
    if n < p + lag + 2:
        return {"n": int(n), "alpha": float("nan"), "t_alpha": float("nan"), "beta": [float("nan")] * (p - 1),
                "t_beta": [float("nan")] * (p - 1), "r2": float("nan")}
    ZtZi = np.linalg.pinv(Z.T @ Z)
    b = ZtZi @ Z.T @ y
    e = y - Z @ b
    u = Z * e[:, None]
    S = u.T @ u
    for k in range(1, lag + 1):
        G = u[k:].T @ u[:-k]
        S += (1.0 - k / (lag + 1.0)) * (G + G.T)
    V = ZtZi @ S @ ZtZi
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, b / se, np.nan)
    ss = float(((y - y.mean()) ** 2).sum())
    return {"n": int(n), "alpha": float(b[0]), "t_alpha": float(t[0]), "beta": [float(v) for v in b[1:]],
            "t_beta": [float(v) for v in t[1:]], "r2": float(1.0 - (e @ e) / ss) if ss > 0 else float("nan")}


def nw_t_rows(X: np.ndarray, lag: int = NW_LAG) -> np.ndarray:
    """행마다 core ``newey_west_t`` 의 t(평균/HAC se). se=0 이면 NaN."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    mu = X.mean(axis=1)
    d = X - mu[:, None]
    var = (d * d).sum(axis=1) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * (d[:, k:] * d[:, :-k]).sum(axis=1) / n
    se = np.sqrt(np.clip(var, 0.0, None) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, mu / se, np.nan)
