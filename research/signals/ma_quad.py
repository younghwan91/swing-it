"""q차식 롤링 외삽 — 로그 종가 패널의 마지막 N개에 최소제곱 q차식을 맞춰 t+1..t+k 를 외삽한다.

s = −N+1..0 (0 = t) 기준 설계행렬의 정규방정식은 (N, q) 로만 정해지므로 외삽 가중치 W(k×N)를 한 번 구하고
종목별 슬라이딩 창에 점곱한다. 창에 NaN 이 하나라도 있으면 그 칸은 NaN(0 으로 채우지 않는다).
numba 는 선택이다(:mod:`research.signals.vcp` 와 같은 규약) — 없으면 같은 소스가 파이썬으로 돈다.
"""

from __future__ import annotations

import numpy as np

try:
    import numba

    def _jit(fn):
        return numba.njit(cache=False, nogil=True)(fn)

except ImportError:  # pragma: no cover

    def _jit(fn):
        return fn


def extrapolation_weights(N: int, k: int, q: int) -> np.ndarray:
    """(k, N) — 창 값 y(s=−N+1..0) 에 곱하면 s=1..k 의 q차식 외삽값."""
    if q < 0 or N < q + 1 or k < 1:
        raise ValueError(f"잘못된 (N, k, q) = ({N}, {k}, {q})")
    s = np.arange(-N + 1, 1, dtype=np.float64) / N  # 조건수 완화(공간 불변)
    h = np.arange(1, k + 1, dtype=np.float64) / N
    A = np.vander(s, q + 1, increasing=True)  # (N, q+1)
    V = np.vander(h, q + 1, increasing=True)  # (k, q+1)
    pinv = np.linalg.lstsq(A, np.eye(N), rcond=None)[0]  # (q+1, N)
    return V @ pinv


@_jit
def _roll_dot(y, W, N, out):
    n_codes, n_days = y.shape
    k = W.shape[0]
    for i in range(n_codes):
        for t in range(N - 1, n_days):
            ok = True
            for j in range(N):
                if not np.isfinite(y[i, t - N + 1 + j]):
                    ok = False
                    break
            if not ok:
                continue
            for h in range(k):
                acc = 0.0
                for j in range(N):
                    acc += W[h, j] * y[i, t - N + 1 + j]
                out[i, t, h] = acc


def poly_forecast(y: np.ndarray, N: int, k: int, q: int) -> np.ndarray:
    """(n_codes, n_days, k) float32 — ``[i, t, h-1]`` = t 까지 N개로 맞춘 q차식의 t+h 값.

    창에 NaN 이 있거나 t < N−1 이면 NaN. q=0 은 y[i, t] 반복(N 무관, y[i, t] 가 유한하면 값).
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_codes, n_days = y.shape
    if q == 0:
        out = np.repeat(y[:, :, None], k, axis=2)
        return out.astype(np.float32)
    out = np.full((n_codes, n_days, k), np.nan, dtype=np.float64)
    _roll_dot(y, np.ascontiguousarray(extrapolation_weights(N, k, q)), N, out)
    return out.astype(np.float32)


def rolling_sigma(y: np.ndarray, w: int = 60) -> np.ndarray:
    """(n_codes, n_days) float64 — u 까지(포함) w 개 로그 수익(diff)의 표본 표준편차(ddof=1), 결측 있으면 NaN."""
    y = np.asarray(y, dtype=np.float64)
    r = np.full_like(y, np.nan)
    r[:, 1:] = np.diff(y, axis=1)
    n_codes, n_days = y.shape
    out = np.full((n_codes, n_days), np.nan)
    if n_days < w + 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    for a in range(0, n_codes, 64):  # 종목 덩어리 — (덩어리, 일, w) 임시 배열 메모리 제한
        win = sliding_window_view(r[a : a + 64], w, axis=1)
        out[a : a + 64, w - 1 :] = win.std(axis=2, ddof=1)  # NaN 이 끼면 NaN 전파
    return out
