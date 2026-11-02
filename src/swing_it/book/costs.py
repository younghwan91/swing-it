"""8단계 거래비용 — 정가 수수료·일자별 매도세·반스프레드·제곱근 충격. 할인·면제 인자는 두지 않는다.

**이 가정의 한계.** KOSPI 반스프레드 5bp 는 2017~22 호가 틱 하한보다 낮다. 1배는 그 구간에서 낙관이고 2배가 덮는다(B 영향 상한 4.4bp). 2배는 세금까지 두 배로 친다 —
세금은 실제로 늘지 않으니 비용 2배는 시장 비용만 2배인 것보다 보수적이다.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
from krx_quant_core.backtest.crosssectional import _trailing_adv
from krx_quant_core.costs.model import DEFAULT_COMMISSION_RATE, DEFAULT_KOSDAQ_SLIPPAGE_PCT, DEFAULT_KOSPI_SLIPPAGE_PCT
from krx_quant_core.costs.tax import sell_tax_rate

from swing_it.core_layers import core_layer

VOL_WIN, VOL_MIN_OBS = 63, 20
ADV_WIN, ADV_MIN_OBS = 20, 10   # 창의 절반 — core load_risk_inputs 의 adv 와 같은 하한
HALF_SPREAD_BP = {"KOSPI": float(DEFAULT_KOSPI_SLIPPAGE_PCT) * 1e4, "KOSDAQ": float(DEFAULT_KOSDAQ_SLIPPAGE_PCT) * 1e4}
COMMISSION_BP = float(DEFAULT_COMMISSION_RATE) * 1e4


def linear_bp(side: np.ndarray, market: np.ndarray, day: date, hs_bp: np.ndarray) -> np.ndarray:
    """수수료 + 매도세(매도만) + 반스프레드(bp) — 충격 없는 부분(폐지 청산에 쓴다). ``side`` 는 "BUY"·"SELL" 뿐이고 반스프레드는 0 이상 유한수다."""
    side = np.asarray(side, dtype=str)
    hs = np.asarray(hs_bp, dtype=np.float64)
    if not np.isin(side, ("BUY", "SELL")).all():
        raise ValueError(f"side 는 BUY·SELL 이어야 한다 — 그 밖의 값은 세금 없는 매수로 값이 매겨진다: {sorted(set(side.tolist()))}")
    if side.size != hs.size or side.size != len(market):
        raise ValueError(f"길이가 다르다: side {side.size}, market {len(market)}, hs_bp {hs.size}")
    if not (np.isfinite(hs).all() and (hs >= 0).all()):
        raise ValueError("반스프레드는 0 이상 유한수여야 한다")
    tax = np.array([sell_tax_rate(day, m) * 1e4 for m in market], dtype=np.float64)
    return COMMISSION_BP + np.where(side == "SELL", tax, 0.0) + hs


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


def trailing_adv(tv_m: np.ndarray, like: np.ndarray, *, window: int = ADV_WIN, min_obs: "int | None" = None) -> np.ndarray:
    """t 제외 직전 ``window`` 거래일 평균 거래대금(백만원, core ``_trailing_adv``). ``min_obs`` 를 주면 관측이 모자란 칸은 NaN 이다.

    core 는 관측 1개로도 평균을 낸다 — 재개 첫날 거래대금 하나가 20일 평균으로 읽힌다. ``None`` 은 core 그대로(B 재현). 새 연구는 :data:`ADV_MIN_OBS` 를 넘긴다.
    """
    nD = tv_m.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # 관측 0 인 창의 Mean of empty slice — NaN 이 답이다
        adv = _trailing_adv(tv_m, nD, window, like)
    if min_obs is not None and nD > window:
        n = np.lib.stride_tricks.sliding_window_view(np.isfinite(tv_m[:, : nD - 1]), window, axis=1).sum(axis=2)
        adv[:, window:nD][n < min_obs] = np.nan
    return adv


def trailing_vol(ret: np.ndarray, t: int, *, window: int = VOL_WIN, min_obs: int = VOL_MIN_OBS) -> np.ndarray:
    """t 포함 직전 ``window`` 거래일 일수익 표준편차(ddof=1). 유한 관측 < ``min_obs`` 이면 NaN(→ 그 날 거래 불가)."""
    w = ret[:, max(0, t - window + 1): t + 1]
    n = np.isfinite(w).sum(axis=1)
    with np.errstate(invalid="ignore"):
        v = np.nanstd(w, axis=1, ddof=1)
    v[n < min_obs] = np.nan
    return v
