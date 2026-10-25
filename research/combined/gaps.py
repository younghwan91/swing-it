"""야간 갭 특성 — 수정가 갭, 원주가 기준 호가 비율, 제한폭 절반 사건 갭.

``gap`` 은 같은 수정가 계열끼리 나누므로 분할 불변이다. ``tickp`` 는 반드시 원주가로 계산한다
(수정가 60,905 vs 원주가 3,160 이면 호가 5원/3,160 = 15.8bp 이지 1.6bp 가 아니다 — scalp VERDICT §5).
호가표는 날짜별: 2023-01-25 이후 ``swing_it.features.korea.tick_size_vec``(통합 표), 그 전은 시장별 옛 표
(``research.signals.pattern_events.tick_dated`` 와 같은 값 — 테스트가 대조한다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_it.features.korea import tick_size_vec

TICK_REFORM_DAY = pd.Timestamp("2023-01-25")

_OLD_UPPER = np.array([1_000, 5_000, 10_000, 50_000, 100_000, 500_000], dtype=float)
_OLD_SIZE_KOSPI = np.array([1, 5, 10, 50, 100, 500, 1_000], dtype=float)
_OLD_SIZE_KOSDAQ = np.array([1, 5, 10, 50, 100, 100, 100], dtype=float)


def tick_dated_vec(price: np.ndarray, days: pd.DatetimeIndex, mkt: np.ndarray) -> np.ndarray:
    """price[N,T] → 호가단위[N,T]. days[T] 가 개편일 이상이면 통합 표, 아니면 옛 표(mkt[N]: 0 코스피·1 코스닥)."""
    p = np.asarray(price, dtype=float)
    new = tick_size_vec(p)
    idx = np.searchsorted(_OLD_UPPER, np.where(np.isfinite(p), p, 0.0), side="right")
    kosdaq = (np.asarray(mkt) == 1)[:, None]
    old = np.where(kosdaq, _OLD_SIZE_KOSDAQ[idx], _OLD_SIZE_KOSPI[idx])
    old = np.where(np.isfinite(p) & (p > 0), old, np.nan)
    is_new = (pd.DatetimeIndex(days) >= TICK_REFORM_DAY)[None, :]
    return np.where(is_new, new, old)


def gap_arrays(O: np.ndarray, C: np.ndarray, raw_close: np.ndarray, reopen: np.ndarray,  # noqa: E741
               lu: np.ndarray, days: pd.DatetimeIndex, mkt: np.ndarray) -> dict:
    """야간 갭 배열. 모두 [N,T], t 는 시가 날(전일 종가 = t-1).

    - ``gap[t] = O[t]/C[t-1] - 1`` (수정가). ``reopen`` 칸·t=0 은 NaN.
    - ``tickp[t] = tick(raw_close[t-1]) / raw_close[t-1]`` (원주가, 그날 t-1 의 호가표). t=0 은 NaN.
    - ``gap_event[t] = |gap| >= (lu[t]-1)/2`` — 제한폭 절반 이상, 상승·하락 대칭. NaN 갭은 False.
    """
    O = np.asarray(O, dtype=float)  # noqa: E741
    C = np.asarray(C, dtype=float)
    raw = np.asarray(raw_close, dtype=float)
    n, t = O.shape
    gap = np.full((n, t), np.nan)
    tickp = np.full((n, t), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap[:, 1:] = O[:, 1:] / C[:, :-1] - 1.0
        gap[~np.isfinite(gap)] = np.nan
        gap[np.asarray(reopen, dtype=bool)] = np.nan
        tk = tick_dated_vec(raw, days, mkt)
        tickp[:, 1:] = tk[:, :-1] / raw[:, :-1]
    tickp[~np.isfinite(tickp)] = np.nan
    with np.errstate(invalid="ignore"):
        event = np.abs(gap) >= (np.asarray(lu, dtype=float) - 1.0) / 2.0 - 1e-9  # 부동소수 경계 허용
    return {"gap": gap.astype(np.float32), "tickp": tickp.astype(np.float32),
            "gap_event": event & np.isfinite(gap)}
