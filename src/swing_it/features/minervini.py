"""미너비니 뼈대 패널 — 추세 템플릿·RS 등급·피벗 돌파·시장 breadth (스윙 3호 SEPA 합류의 뼈대 S).

설계서: ``docs/superpowers/specs/2026-10-09-sepa-confluence.md`` §3 "뼈대 S". 수치는 원전(미너비니·오닐)
그대로 고정하고 여기서 탐색하지 않는다 — 함수 인자에 기본값으로 박아 두는 것은 테스트·민감도 기술용이다.

옛 SEPA 파이프라인(``ea9a032`` 에서 삭제)은 종목×날짜 numpy 로 돌았다. 여기는 :mod:`.technical` 과 같은
날짜×종목 와이드 패널 in/out 이다.

**규약(전부 시점 안전):** 날짜 ``d`` 행의 값은 ``d`` 종가까지로 계산된다. 전략은 ``d+1`` 시가에 진입한다.
가격은 분할조정 패널(``storage.read_prices`` 기본)이어야 한다. 창은 **꽉 찬 날부터만** 값이 있다
(``min_periods = window``) — 상장 직후 짧은 이력으로 템플릿이 켜지지 않게. 창 안에 결측(상장 전·폐지 후)이
하나라도 있으면 그날은 NaN → 조건 거짓이다.

거래정지일(거래량 0, OHLC = 직전 종가 이월)은 이 모듈에서 따로 빼지 않는다 — 이월 종가는 그날 알 수 있는
"마지막 가격" 이라 이동평균·수익률에 넣어도 미래를 보지 않는다. 체결 불가 처리는 시뮬레이터 몫이다.
"""

from __future__ import annotations

import pandas as pd

from swing_it.features.technical import moving_average, simple_return

#: RS 가중 — 3/6/9/12개월(63/126/189/252 거래일) 수익률에 2:1:1:1 (IBD 식, 최근 분기 2배).
RS_WEIGHTS: dict[int, float] = {63: 2.0, 126: 1.0, 189: 1.0, 252: 1.0}


def trend_template(close: pd.DataFrame, *, high: "pd.DataFrame | None" = None,
                   low: "pd.DataFrame | None" = None, slope_lag: int = 20,
                   near_high: float = 0.75, above_low: float = 1.30, year: int = 252) -> pd.DataFrame:
    """미너비니 추세 템플릿(원전 수치) → bool 패널.

    ``close > MA50 > MA150 > MA200`` ∧ ``MA200 > MA200[d−20]`` ∧ ``close ≥ 0.75 × 252일 고가``
    ∧ ``close ≥ 1.30 × 252일 저가``.

    52주 고·저는 ``high``/``low`` 를 넘기면 **장중 고·저**의 252일 최대·최소(원전 "52-week high" 는 장중가 —
    뼈대 S 는 이쪽을 쓴다), 안 넘기면 종가의 252일 최대·최소다. 장중가 쪽이 약간 더 엄격하다(고가 ≥ 종가).
    252일 창은 ``d`` 자신을 포함한다 — 오늘 신고가를 찍은 종목은 ``close ≥ 0.75 × 오늘 고가`` 로 통과.

    ``MA200[d−20]`` 은 220 거래일 이력이 필요하고 252일 고저는 252일이 필요하므로, 실제로 값이 켜지는
    첫날은 상장(또는 패널 시작) 후 252번째 거래일이다.
    """
    hi = (close if high is None else high).rolling(year, min_periods=year).max()
    lo = (close if low is None else low).rolling(year, min_periods=year).min()
    ma50, ma150, ma200 = (moving_average(close, w) for w in (50, 150, 200))
    tt = ((close > ma50) & (ma50 > ma150) & (ma150 > ma200)
          & (ma200 > ma200.shift(slope_lag))
          & (close >= near_high * hi) & (close >= above_low * lo))
    # NaN 비교는 이미 False 지만, 창이 덜 찬 날을 명시적으로 막아 둔다(입력이 바뀌어도 의도가 남게).
    full = hi.notna() & lo.notna() & ma200.shift(slope_lag).notna()
    return tt & full


def rs_weighted_return(close: pd.DataFrame, weights: "dict[int, float] | None" = None) -> pd.DataFrame:
    """RS 원점수 — ``2·r63 + r126 + r189 + r252``. 어느 한 기간이라도 NaN 이면 NaN(252일 이력 필요)."""
    w = RS_WEIGHTS if weights is None else weights
    out = None
    for periods, wt in w.items():
        term = wt * simple_return(close, periods)
        out = term if out is None else out + term
    return out


def rs_rating(close: pd.DataFrame, *, population: "pd.DataFrame | None" = None,
              weights: "dict[int, float] | None" = None) -> pd.DataFrame:
    """IBD 식 RS 등급 — 그날 횡단면 백분위(0~100], 클수록 강하다.

    값 = 100 × (그날 유효 종목 중 내 가중수익 이하인 종목 비율, 동점은 평균 순위). 최고 종목이 100,
    ``n`` 종목 중 최하위가 ``100/n``. ``RS ≥ 70`` 은 대략 상위 30% 다.

    모집단은 기본 **그날 가중수익이 정의된 전 종목**이다(IBD 는 자기 DB 전 종목과 비교한다 — 유동성
    유니버스로 좁히면 "시장 대비 강도" 가 "유동주 대비 강도" 로 바뀐다). ``population`` (bool 날짜×종목)을
    주면 그 안에서만 순위를 매기고 밖은 NaN.
    """
    raw = rs_weighted_return(close, weights)
    if population is not None:
        raw = raw.where(population.reindex(index=raw.index, columns=raw.columns, fill_value=False).astype(bool))
    return raw.rank(axis=1, pct=True, method="average") * 100.0


def pivot_level(high: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """피벗 = 직전 ``window`` 거래일(``[d−50, d−1]``) 최고 고가. ``d`` 자신은 넣지 않는다."""
    return high.shift(1).rolling(window, min_periods=window).max()


def pivot_breakout(high: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, *,
                   window: int = 50, vol_mult: float = 1.5) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """피벗 돌파 → ``(breakout bool 패널, pivot 패널)``.

    ``close_d > max(high[d−50..d−1])`` ∧ ``volume_d ≥ 1.5 × mean(volume[d−50..d−1])``.
    거래량 평균에는 정지일(거래량 0)도 0 으로 들어간다 — 그날 알려진 거래량 그대로다. 정지일 당일은
    거래량 0 이라 돌파가 될 수 없다.

    피벗 패널을 함께 돌려주는 이유: 진입 규칙(원전 5% 추격 금지)이 ``d+1 시가 > 피벗 × 1.05`` 를 본다.
    """
    pivot = pivot_level(high, window)
    vol_avg = volume.shift(1).rolling(window, min_periods=window).mean()
    breakout = (close > pivot) & (volume >= vol_mult * vol_avg) & (vol_avg > 0)
    return breakout, pivot


def market_breadth(close: pd.DataFrame, market: pd.Series, *, window: int = 50) -> pd.DataFrame:
    """시장별 breadth = 그날 ``close > MA50`` 인 종목 비율 → 날짜×시장 프레임.

    분모는 그날 MA50 이 **정의된** 종목(상장 50일 미만·폐지 후는 빠진다). 시장 구분은 ``market``
    (code → "KOSPI"/"KOSDAQ"/기타) 그대로 묶는다. 종목 모집단은 패널 전 종목이다(유니버스로 좁히지
    않는다 — 국면은 시장 전체의 상태다).
    """
    ma = moving_average(close, window)
    valid = ma.notna() & close.notna()
    above = (close > ma) & valid
    mk = market.reindex(close.columns)
    num = above.T.groupby(mk.to_numpy()).sum().T
    den = valid.T.groupby(mk.to_numpy()).sum().T
    return num / den.where(den > 0)


def breadth(close: pd.DataFrame, market: pd.Series, *, window: int = 50) -> pd.DataFrame:
    """:func:`market_breadth` 를 종목 축으로 펼친다 → 날짜×종목(각 종목 자리에 그 시장의 breadth)."""
    mb = market_breadth(close, market, window=window)
    mk = market.reindex(close.columns)
    out = mb.reindex(columns=mk.to_numpy())
    out.columns = close.columns
    return out
