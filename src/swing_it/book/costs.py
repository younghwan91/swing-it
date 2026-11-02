"""8단계 거래비용 — 정가 수수료·일자별 매도세·반스프레드·제곱근 충격. 할인·면제 인자는 두지 않는다."""

from __future__ import annotations

from datetime import date

import numpy as np
from krx_quant_core.costs.model import DEFAULT_COMMISSION_RATE, DEFAULT_KOSDAQ_SLIPPAGE_PCT, DEFAULT_KOSPI_SLIPPAGE_PCT
from krx_quant_core.costs.tax import sell_tax_rate

from swing_it.core_layers import core_layer

VOL_WIN, VOL_MIN_OBS = 63, 20
HALF_SPREAD_BP = {"KOSPI": float(DEFAULT_KOSPI_SLIPPAGE_PCT) * 1e4, "KOSDAQ": float(DEFAULT_KOSDAQ_SLIPPAGE_PCT) * 1e4}
COMMISSION_BP = float(DEFAULT_COMMISSION_RATE) * 1e4


def linear_bp(side: np.ndarray, market: np.ndarray, day: date, hs_bp: np.ndarray) -> np.ndarray:
    """수수료 + 매도세(매도만) + 반스프레드(bp) — 충격 없는 부분(폐지 청산에 쓴다)."""
    side = np.asarray(side, dtype=str)
    tax = np.array([sell_tax_rate(day, m) * 1e4 for m in market], dtype=np.float64)
    return COMMISSION_BP + np.where(side == "SELL", tax, 0.0) + np.asarray(hs_bp, dtype=np.float64)


def trade_costs(dw, adv_won, vol, market, day: date, hs_bp, aum: float) -> np.ndarray:
    """종목별 1배 비용(NAV 대비 분수). core ``trade_cost_bp`` 그대로(정가 수수료·일자별 매도세·반스프레드·제곱근 충격 y=1).

    2배는 이 값의 2배다 — 네 항(수수료·매도세·반스프레드·y)이 모두 배수에 선형이다. core ``costs.impact`` 는 기본 핀 0.7.0 에 없어 부를 때 import 한다.
    """
    dw = np.asarray(dw, dtype=np.float64)
    out = np.zeros(dw.shape)
    tr = dw != 0
    if tr.any():
        side = np.where(dw[tr] > 0, "BUY", "SELL")
        trade_cost_bp = core_layer("krx_quant_core.costs.impact").trade_cost_bp
        bp = trade_cost_bp(side, np.abs(dw[tr]) * aum, np.asarray(adv_won)[tr], np.asarray(vol)[tr],
                           np.asarray(market, dtype=object)[tr], day, half_spread_bp=np.asarray(hs_bp)[tr], y=1.0)
        out[tr] = bp / 1e4 * np.abs(dw[tr])
    return out


def trailing_vol(ret: np.ndarray, t: int, *, window: int = VOL_WIN, min_obs: int = VOL_MIN_OBS) -> np.ndarray:
    """t 포함 직전 ``window`` 거래일 일수익 표준편차(ddof=1). 유한 관측 < ``min_obs`` 이면 NaN(→ 그 날 거래 불가)."""
    w = ret[:, max(0, t - window + 1): t + 1]
    n = np.isfinite(w).sum(axis=1)
    with np.errstate(invalid="ignore"):
        v = np.nanstd(w, axis=1, ddof=1)
    v[n < min_obs] = np.nan
    return v
