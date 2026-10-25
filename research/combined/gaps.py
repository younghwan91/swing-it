"""야간 갭 특성 — 수정가 갭, 수정가 기준 호가 비율, 제한폭 절반 사건 갭.

``gap`` 은 같은 수정가 계열끼리 나누므로 분할 불변이다. ``tickp`` 는 **수정 종가**로 계산한다(정정 0, 판정 26).
원주가로 계산하면 1/tickp 로 원주가가 되살아나고, 원주가/수정 종가 = t 이후 액면분할·무상증자의 **미래 수정 계수**가 train 에 샌다
(절단 검사로 못 잡는다). 수정 종가로 계산한 tickp 는 V 의 C 의 결정적 함수라 정보를 더하지 않는다.
대가: 수정 사건 전 구간에서는 호가 비율이 근사다(수정가 60,905 vs 원주가 3,160 이면 실제 15.8bp 를 1.6bp 로 본다 — scalp VERDICT §5 의 교훈과 맞바꿈).
엔진의 체결 비용은 이와 무관하게 정가 ``cost_rt`` × 2 다.
tickp 는 **유효숫자 2자리**로 거칠게 낸다(정정 0-2 K1) — 정밀한 tick/C 로는 수정 종가 C 를 되살려 V 의 가격 배수를 벗겨 낼 수 있다.
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


def coarse2(x: np.ndarray) -> np.ndarray:
    """유효숫자 2자리 반올림(0·NaN 은 그대로)."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 10.0 ** (np.floor(np.log10(np.abs(x))) - 1)
        out = np.round(x / q) * q
    return np.where(np.isfinite(x) & (x != 0), out, x)


def gap_arrays(O: np.ndarray, C: np.ndarray, reopen: np.ndarray,  # noqa: E741
               lu: np.ndarray, days: pd.DatetimeIndex, mkt: np.ndarray) -> dict:
    """야간 갭 배열. 모두 [N,T], t 는 시가 날(전일 종가 = t-1).

    - ``gap[t] = O[t]/C[t-1] - 1`` (수정가). ``reopen`` 칸·t=0 은 NaN.
    - ``tickp[t] = coarse2(tick(C[t-1]) / C[t-1])`` (수정 종가, 주문일 t 의 호가표, 유효숫자 2자리). t=0 은 NaN.
      원주가는 받지 않는다(미래 수정 계수 누수).
    - ``gap_event[t] = |gap| >= (lu[t]-1)/2`` — 제한폭 절반 이상, 상승·하락 대칭. NaN 갭은 False.
    """
    O = np.asarray(O, dtype=float)  # noqa: E741
    C = np.asarray(C, dtype=float)
    n, t = O.shape
    mkt = np.asarray(mkt)
    days = pd.DatetimeIndex(days)
    assert mkt.shape == (n,) and set(np.unique(mkt)) <= {0, 1}, "mkt: int[N], 0=코스피 1=코스닥"
    assert len(days) == t, "days 길이 != T"
    assert C.shape == (n, t) and np.asarray(reopen).shape == (n, t)
    assert np.asarray(lu).shape[-1] == t, "lu 마지막 축 != T"
    gap = np.full((n, t), np.nan)
    tickp = np.full((n, t), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap[:, 1:] = O[:, 1:] / C[:, :-1] - 1.0
        gap[~np.isfinite(gap)] = np.nan
        gap[np.asarray(reopen, dtype=bool)] = np.nan
        # 가격은 전일(t-1) 수정 종가, 호가표는 주문일(t)에 시행 중인 표
        tickp[:, 1:] = tick_dated_vec(C[:, :-1], days[1:], mkt) / C[:, :-1]
    tickp[~np.isfinite(tickp)] = np.nan
    tickp = coarse2(tickp)
    with np.errstate(invalid="ignore"):
        event = np.abs(gap) >= (np.asarray(lu, dtype=float) - 1.0) / 2.0 - 1e-9  # 부동소수 경계 허용
    return {"gap": gap.astype(np.float32), "tickp": tickp.astype(np.float32),
            "gap_event": event & np.isfinite(gap)}
