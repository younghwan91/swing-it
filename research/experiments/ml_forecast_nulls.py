"""ML 수익률 예측 — 널 N2(같은 메뉴의 무작위 신호)·N3(같은 메뉴의 무작위 결합) (PREREG §7, 정정 ML-1 ⑨⑩).

N2: 반복 r 마다 U* 위 순열 π 하나(``keyed_rng("ML-N2", r)``). 날짜 t 에서 i ∈ U(t) 는 p_{π(i),t} 를 받는다. π(i) ∉ U(t) 이거나 그 값이
NaN 이면, 그날 아직 배정되지 않은 실제 예측에서 비복원으로 채운다(``keyed_rng("ML-N2-fill", r, t)``). 채움이 필요한 자리가 남은 값보다 많으면
자리를 무작위로 골라 넣고 나머지는 NaN — 그래서 널의 유한 개수와 값 다중집합이 실제와 같다.
같은 함수를 IC 격자·북 격자에 쓴다.
"""

from __future__ import annotations

import numpy as np
from ml_forecast_common import FEATURES, keyed_rng
from ml_forecast_features import weighted_score


def n2_perm(u_star: np.ndarray, r: int, N: int) -> np.ndarray:
    """π — (N,) 배열. U* 안은 U* 안으로 섞이고 밖은 −1."""
    pi = np.full(N, -1, dtype=np.int64)
    pi[u_star] = keyed_rng("ML-N2", r).permutation(u_star)
    return pi


def n2_assign(pred_t: np.ndarray, idx: np.ndarray, pi: np.ndarray, r: int, t: int, N: int) -> tuple[np.ndarray, int]:
    """날짜 t 의 널 예측((|U(t)|,), idx 순서)과 채운 자리 수. ``pred_t`` 는 idx 순서의 실제 예측."""
    full = np.full(N, np.nan)
    full[idx] = pred_t
    src = pi[idx]
    direct = np.zeros(idx.size, dtype=bool)
    ok_src = src >= 0
    direct[ok_src] = np.isfinite(full[src[ok_src]])
    out = np.full(idx.size, np.nan)
    out[direct] = full[src[direct]]
    used = np.zeros(N, dtype=bool)
    used[src[direct]] = True
    left_rows = idx[np.isfinite(pred_t) & ~used[idx]]
    need = np.flatnonzero(~direct)
    n_fill = left_rows.size
    if n_fill:
        rng = keyed_rng("ML-N2-fill", r, t)
        slots = rng.permutation(need)[:n_fill]
        out[slots] = full[rng.permutation(left_rows)]
    return out, int(n_fill)


def n3_draw(r: int, k: int = len(FEATURES)) -> tuple[np.ndarray, np.ndarray]:
    """부호 ±1 과 Dirichlet(1,…,1) 가중."""
    rng = keyed_rng("ML-N3", r)
    w = rng.dirichlet(np.ones(k))
    sign = rng.choice([-1.0, 1.0], size=k)
    return w, sign


def n3_scores(Z: dict[int, np.ndarray], r: int) -> dict[int, np.ndarray]:
    w, sign = n3_draw(r)
    return {t: weighted_score(z, w, sign) for t, z in Z.items()}


def q95(values) -> float:
    """95 백분위(정정 ML-1 ⑨). NaN 은 뺀다."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.quantile(v[np.isfinite(v)], 0.95))
