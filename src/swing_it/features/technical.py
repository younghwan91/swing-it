"""테크니컬 패널 — 스윙 신호의 가격 축 (이동평균·수익률·유동성).

지금까지 이 저장소엔 공용 테크니컬 모듈이 없었다(MA·RSI 가 ``research/signals/pullback_swing.py``
안에만 있었다). 스윙 연구가 늘어날 자리라 ``src`` 로 올린다.

**규약(전부 시점 안전):** 날짜 ``d`` 행의 값은 ``d`` 종가까지로 계산된다. 전략은 ``d+1`` 에 진입한다.
입력은 날짜×종목 와이드 패널(:func:`swing_it.engine.panels.panel_pivot` 결과 모양), 출력도 같은 모양.
가격은 반드시 분할조정 패널(``daily_bars_adjusted`` — ``storage.read_prices`` 기본)이어야 한다.
"""

from __future__ import annotations

import pandas as pd


def moving_average(close: pd.DataFrame, window: int, *, min_periods: "int | None" = None) -> pd.DataFrame:
    """단순 이동평균. 기본은 창이 꽉 찬 날부터(``min_periods=window``) — 상장 직후 짧은 이력으로
    "추세 위" 가 켜지지 않게."""
    return close.rolling(window, min_periods=min_periods or window).mean()


def simple_return(close: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """``periods`` 거래일 단순수익률. 직전 값이 0·결측이면 NaN."""
    prev = close.shift(periods)
    return (close / prev.where(prev > 0)) - 1.0


def average_trade_value(trade_value: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """창 평균 거래대금(입력 단위 그대로 — daily_bars 는 백만원). 창이 다 차야 값이 있다."""
    return trade_value.rolling(window, min_periods=window).mean()
