"""백테스트 과최적 확률 PBO — CSCV(조합 대칭 교차검증, Bailey·Borwein·López de Prado·Zhu 2017).

"격자에서 표본 안 최고를 고르면, 그 칸이 표본 밖에서 중간 아래일 확률" 을 잰다.

절차(논문 그대로):

1. 성과 행렬 ``M`` (시간 T × 후보 N) 을 시간 순서로 **연속한** ``S`` 조각으로 자른다(짝수 S).
2. ``S`` 조각 중 ``S/2`` 를 고르는 모든 조합 ``C(S, S/2)`` 마다: 고른 조각 = 표본 안(IS), 나머지 = 표본 밖(OOS).
3. IS 성과 최고 후보 ``n*`` 를 고르고, OOS 성과에서 그 후보의 상대 순위 ``ω = rank / (N+1)``
   (rank 1 = 최하, 동점은 평균 순위) → 로짓 ``λ = ln(ω / (1−ω))``.
4. ``PBO = P(λ ≤ 0)`` — IS 최고가 OOS 에서 중앙값 이하로 떨어진 조합의 비율.

조각-집합 성과는 두 가지를 지원한다: ``"sharpe"`` = 그 조각들의 행(월) 값 평균 ÷ 표준편차(ddof=1, 아래 바닥
``std_floor``) 와 ``"mean"`` = 평균. 잡음뿐이면 PBO ≈ 0.5, 한 후보가 진짜로 지배하면 PBO ≈ 0.

기술(description)이지 판정이 아니다 — 문턱은 사전등록 문서가 쓴다. numpy 만 쓴다(leaf 패키지 경계).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = ["pbo_cscv"]


def _rank_avg(x: np.ndarray) -> np.ndarray:
    """1..N 오름차순 순위(동점 평균). NaN 은 최하(가장 작은 값)로 본다."""
    v = np.where(np.isfinite(x), x, -np.inf)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pbo_cscv(M: np.ndarray, *, n_blocks: int = 16, metric: str = "sharpe",
             std_floor: float = 1e-6) -> dict:
    """CSCV 로 PBO 를 계산한다.

    Args:
        M: (T, N) 성과 행렬 — 행 = 시간(시간 순서), 열 = 후보(격자 칸). NaN 은 0 으로 본다(호출부가 "거래 없는
            달 = 0" 같은 규약을 미리 채우는 것을 권한다).
        n_blocks: 조각 수 S(짝수, 2 이상, T 이하). 조각은 :func:`numpy.array_split` 로 연속하게 자른다.
        metric: ``"sharpe"`` (평균/표준편차, ddof=1, 표준편차 하한 ``std_floor``) 또는 ``"mean"``.
        std_floor: ``"sharpe"`` 의 표준편차 바닥 — 조각 안 값이 전부 같은(예: 거래 없는 0) 후보가 0 으로
            나뉘지 않게 한다.

    Returns:
        ``pbo`` (λ ≤ 0 비율), ``n_combos``, ``logits`` (조합별 λ), ``is_best`` (조합별 IS 최고 후보 열 번호),
        ``oos_rank`` (그 후보의 OOS 순위, 1 = 최하), ``n_candidates``, ``n_blocks``, ``metric``,
        ``best_freq`` (후보별 IS 최고로 뽑힌 비율), ``is_perf``·``oos_perf`` (조합별 IS 최고 후보의 IS/OOS 성과 —
        성과 저하 기울기를 보려는 용도).
    """
    M = np.asarray(M, float)
    if M.ndim != 2:
        raise ValueError("M 은 (T, N) 2차원이어야 한다")
    T, N = M.shape
    if n_blocks < 2 or n_blocks % 2 or n_blocks > T:
        raise ValueError(f"n_blocks 는 2 이상 짝수, T({T}) 이하여야 한다: {n_blocks}")
    if metric not in ("sharpe", "mean"):
        raise ValueError(f"metric 은 'sharpe' 또는 'mean': {metric!r}")
    X = np.where(np.isfinite(M), M, 0.0)
    blocks = np.array_split(np.arange(T), n_blocks)
    cnt = np.array([len(b) for b in blocks], float)              # (S,)
    s1 = np.stack([X[b].sum(axis=0) for b in blocks])            # (S, N)
    s2 = np.stack([(X[b] ** 2).sum(axis=0) for b in blocks])     # (S, N)
    tot1, tot2, totn = s1.sum(axis=0), s2.sum(axis=0), cnt.sum()

    def perf(a1: np.ndarray, a2: np.ndarray, n: float) -> np.ndarray:
        mu = a1 / n
        if metric == "mean":
            return mu
        var = (a2 - n * mu ** 2) / max(n - 1.0, 1.0)
        sd = np.sqrt(np.clip(var, 0.0, None))
        return mu / np.maximum(sd, std_floor)

    combos = list(combinations(range(n_blocks), n_blocks // 2))
    logits = np.empty(len(combos))
    is_best = np.empty(len(combos), int)
    oos_rank = np.empty(len(combos))
    is_perf = np.empty(len(combos))
    oos_perf = np.empty(len(combos))
    for k, J in enumerate(combos):
        J = list(J)
        a1, a2, n = s1[J].sum(axis=0), s2[J].sum(axis=0), cnt[J].sum()
        p_is = perf(a1, a2, n)
        p_oos = perf(tot1 - a1, tot2 - a2, totn - n)
        best = int(np.nanargmax(np.where(np.isfinite(p_is), p_is, -np.inf)))
        r = _rank_avg(p_oos)[best]
        w = r / (N + 1.0)
        logits[k] = np.log(w / (1.0 - w))
        is_best[k] = best
        oos_rank[k] = r
        is_perf[k] = p_is[best]
        oos_perf[k] = p_oos[best]
    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "n_combos": len(combos),
        "n_candidates": N,
        "n_blocks": n_blocks,
        "metric": metric,
        "logits": logits,
        "is_best": is_best,
        "oos_rank": oos_rank,
        "best_freq": np.bincount(is_best, minlength=N) / len(combos),
        "is_perf": is_perf,
        "oos_perf": oos_perf,
    }
