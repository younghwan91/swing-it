"""시장 국면 패널 — 날짜 × 시장(KOSPI·KOSDAQ·ALL) 의 계량 국면 신호 (적응 층 A·C, ``research/logs/sepa_explore/ADAPTIVE.md`` §1).

**규약(전부 시점 안전):** 날짜 ``d`` 행의 값은 ``d`` 종가까지의 자료로만 계산한다(창은 전부 뒤쪽만 본다 — 중심 창·양방향
필터 없음). 전략은 ``d+1`` 에 적용한다. 입력은 :mod:`swing_it.features.technical` 과 같은 날짜×종목 와이드 패널이고
가격은 분할조정 패널(``storage.read_prices`` 기본 + 2026-09-14~ 는 :mod:`swing_it.krx_session` 정규장 보정)이어야 한다.

A. 계량 국면(시장별):

- ``breadth_ma50``·``breadth_ma200``: 그날 MA 가 정의된 종목 중 종가 > MA 비율(:func:`swing_it.features.minervini.market_breadth`).
- ``index_level``·``index_ret``·``index_n``·``index_maxw``: **거래대금 가중** 유니버스 지수 대용(2026-10-10 리뷰 C1 뒤 재구축 —
  주식수를 쓰지 않는다). 종목 가중 = **직전 20거래일**(``d−20..d−1``) 평균 거래대금(``d−1`` 종가 때 이미 아는 값), 시장마다 단일 종목
  ≤ :data:`INDEX_CAP` (10%) 로 깎고 나머지에 비례 재분배(:func:`capped_weights`). 그날 |r| > :data:`RET_LIMIT` 인 종목은 수익과 가중
  **둘 다**에서 뺀다. ``index_n`` = 가중을 가진 종목 수, ``index_maxw`` = 그날 최대 단일 종목 가중. ``index_vs_ma200`` = 레벨 / MA200 − 1,
  ``rv20`` = 20일 로그수익률 표준편차 × √252, ``idx_sign_ema20`` = sign(``index_ret``) 의 20일 EMA(리뷰 M3 기준선).
  ``official_ret``·``official_vs_ma200`` 은 ``sector_index`` 001/101(2024-01-26~) 의 같은 값(대조용, ALL 은 NaN).
  위생 바(:func:`index_sanity`, 테스트 고정): 매일 ``index_maxw`` ≤ 10%, |``index_ret``| ≤ 31%, 2024-01-26+ 공식 지수 일수익률과 상관 ≥ 0.95.
- ``dispersion``: 그날 유효 일수익률의 횡단면 표준편차.
- ``new_highs``·``new_lows``·``nh_nl_ratio``: 장중 고가 > 직전 252일 최고 고가 / 저가 < 직전 252일 최저 저가 종목 수,
  비율 = nh / (nh + nl) (둘 다 0 이면 NaN — "비율" 이지만 0~1 로 묶었다; 원래 nh/nl 은 nl=0 에서 발산한다).
- ``flow20``: 유니버스 전체 (외국인+기관) 20일 순매수 수량 합 / 20일 거래량(``acc_trde_qty``) 합. 수급 수집이 전 종목으로
  넓어진 2024-02 부터만(그 전은 연 ~650 종목 — :mod:`swing_it.features.flow_leg` 설명) → 그 전은 NaN.
- ``breakouts20``·``breakout_success20``: "돌파가 먹히는 장인가". 피벗 돌파(:func:`~swing_it.features.minervini.pivot_breakout`,
  50일 피벗·거래량 1.5배) 종목이 돌파 ``h=5`` 거래일 뒤 종가에도 피벗 위인 비율. **PIT:** ``d`` 행은 결과가 ``d`` 까지
  확정된 돌파만 센다 — 돌파일 ``b ∈ [d−19, d−5]`` (직전 20일 창 ∩ ``b+5 ≤ d``), 즉 성공 여부 열을 ``h`` 만큼 뒤로 민 뒤
  ``20−h`` 창으로 센다. ``b+5`` 종가가 없으면(폐지) 분모에서도 뺀다.

C. 스펙트럼 척도 신호(시장별 지수 대용 로그 레벨과 ``breadth_ma50`` 두 열에만 — 종목별이 아니다):

- :func:`rolling_periodogram`: 롤링 256거래일 창 → 선형 추세 제거 → 한(Hann) 창 → ``rfft``(제로패딩 1024) → 주기 5~120일
  범위에서 파워 최대 주파수의 역수 = ``*_period``(일), 그 봉우리 주엽(±2/256 cycles/day) 파워 / 전체(DC 제외) = ``*_share``
  (순수 사인이면 ≈ 1, 백색잡음이면 ≈ 주엽 폭 / 전체 폭 ≈ 0.03).
  **유효성(리뷰 M4):** ``*_share`` 가 같은 길이 무작위 보행 1,000벌의 95 백분위(:func:`rw_share_threshold`, 창 길이별 표
  :func:`rw_threshold_table`) **초과**인 날만 ``*_period`` 를 두고, 아니면 NaN — 붉은 잡음이 만드는 대역 끝 봉우리를 "주기" 로 읽지 않게.
- :func:`haar_variance_share`: 인과 Haar 세부계수 4단계(척도 2·4·8·16일: 직전 블록 평균 − 그 앞 블록 평균, 둘 다 과거) 의
  롤링 256일 분산 비율 ``*_haar1..4`` (합 = 1). 저척도 우세 = 잡음, 고척도 우세 = 추세.

출력은 긴 형식 ``(date, market, 피처…)`` 한 프레임(:func:`build_regime_panel`). 수익·트레이드 결과는 여기 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from swing_it.features.minervini import pivot_breakout
from swing_it.features.technical import simple_return

#: 시장 축. ``ALL`` = KOSPI + KOSDAQ (KONEX·미분류는 어느 쪽에도 안 들어간다).
MARKETS: tuple[str, ...] = ("KOSPI", "KOSDAQ", "ALL")
#: 수급 수집이 전 종목으로 넓어진 날 — ``flow20`` 은 이 날부터.
FLOW_SINCE = "2024-02-01"
YEAR = 252
SPEC_WINDOW = 256
SPEC_NFFT = 1024
SPEC_MIN_PERIOD = 5
SPEC_MAX_PERIOD = 120
HAAR_LEVELS = 4
#: 공식 지수 코드(``sector_index``) — 시장별 대조.
OFFICIAL_INDEX = {"KOSPI": "001", "KOSDAQ": "101"}
#: 일수익률 유효 한계 — KRX 가격제한폭 ±30%(2015-06~). 그 밖은 조정 누락·감자 같은 자료 사건이지 수익이 아니다.
RET_LIMIT = 0.31
#: 지수 대용이 뜻을 가지려면 그날 가중치를 가진 종목이 이만큼은 있어야 한다.
MIN_INDEX_N = 100
#: 지수 대용 가중 = 직전 이 거래일 수의 평균 거래대금.
INDEX_ATV_WINDOW = 20
#: 단일 종목 가중 상한(시장마다). 1/cap 종목 미만인 날은 상한을 지킬 수 없어 NaN.
INDEX_CAP = 0.10
#: 위생 바 — 공식 지수(sector_index 001/101) 일수익률과의 최소 상관, 비교 시작일.
INDEX_MIN_CORR = 0.95
OFFICIAL_SINCE = "2024-01-26"
#: 스펙트럼 유효성 귀무 — 무작위 보행 벌 수·분위·시드.
RW_SIMS = 1000
RW_QUANTILE = 0.95
RW_SEED = 20261010


# ---------------------------------------------------------------------------
# 시장별 집계 도우미
# ---------------------------------------------------------------------------

def _market_masks(columns: pd.Index, market: pd.Series) -> dict[str, np.ndarray]:
    mk = market.reindex(columns)
    kospi = (mk == "KOSPI").to_numpy()
    kosdaq = (mk == "KOSDAQ").to_numpy()
    return {"KOSPI": kospi, "KOSDAQ": kosdaq, "ALL": kospi | kosdaq}


def group_sum(panel: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """날짜×종목 → 날짜×시장 합(NaN 은 0 으로 센다 — 개수는 :func:`group_count` 로 따로)."""
    masks = _market_masks(panel.columns, market)
    v = panel.to_numpy(dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    return pd.DataFrame({m: v[:, mask].sum(axis=1) for m, mask in masks.items()}, index=panel.index)


def group_count(panel: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """날짜×종목 → 날짜×시장 유효(finite) 개수."""
    masks = _market_masks(panel.columns, market)
    ok = np.isfinite(panel.to_numpy(dtype=float))
    return pd.DataFrame({m: ok[:, mask].sum(axis=1) for m, mask in masks.items()}, index=panel.index)


def group_mean(panel: pd.DataFrame, market: pd.Series, *, min_count: int = 1) -> pd.DataFrame:
    s, n = group_sum(panel, market), group_count(panel, market)
    return s / n.where(n >= min_count)


def group_std(panel: pd.DataFrame, market: pd.Series, *, min_count: int = 2) -> pd.DataFrame:
    """횡단면 표준편차(ddof=1)."""
    s, n = group_sum(panel, market), group_count(panel, market)
    ss = group_sum(panel * panel, market)
    n = n.where(n >= min_count)
    var = (ss - s * s / n) / (n - 1)
    return np.sqrt(var.clip(lower=0.0))


# ---------------------------------------------------------------------------
# A. 계량 국면
# ---------------------------------------------------------------------------

def valid_returns(close: pd.DataFrame, limit: float = RET_LIMIT) -> pd.DataFrame:
    """일수익률(:func:`simple_return`) 중 |r| ≤ ``limit`` 만 — 가격제한폭 밖은 NaN(조정 누락·감자 등 자료 사건)."""
    r = simple_return(close)
    return r.where(r.abs() <= limit)


# 각 피처는 (종목별 계산 → 시장별 합) 의 "parts" 와 그 합만으로 끝내는 "finalize" 로 나뉜다. 시장별 합은 종목 덩어리끼리
# 더할 수 있으므로 :func:`build_regime_panel` 이 종목을 ``chunk`` 개씩 잘라 돌리며 합을 누적한다(메모리 ≤ 4GB).
# 공개 함수(``dispersion`` 등)는 전 종목 한 번에 parts → finalize 한 것과 같다. 지수 대용(:func:`tv_weighted_index`)만은 상한·재분배가
# 시장 전체에 걸린 비선형이라 덩어리로 나누지 않고 전 종목을 한 번에 다룬다.


def capped_weights(w: np.ndarray, cap: float = INDEX_CAP, *, max_iter: int = 100) -> np.ndarray:
    """행(날짜)마다 음 아닌 가중 → 합 1 로 정규화하고 단일 원소 ≤ ``cap`` 이 되게 깎아 남는 몫을 안 깎인 원소에 **원 가중 비례**로 재분배
    (water-filling — 깎인 집합은 커지기만 하므로 수렴한다). NaN·0 이하는 0. 양수 원소가 ``ceil(1/cap)`` 개 미만인 행은 상한을 지킬 수 없어 NaN."""
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    npos = (w > 0).sum(axis=1)
    tot = w.sum(axis=1, keepdims=True)
    out = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
    capped = np.zeros(w.shape, dtype=bool)
    for _ in range(max_iter):
        over = out > cap * (1 + 1e-12)
        if not over.any():
            break
        capped |= over
        free = np.where(capped, 0.0, w)
        free_tot = free.sum(axis=1, keepdims=True)
        rem = 1.0 - cap * capped.sum(axis=1, keepdims=True)
        out = np.where(capped, cap, np.divide(free * rem, free_tot, out=np.zeros_like(w), where=free_tot > 0))
    out[npos < int(np.ceil(1.0 / cap - 1e-9))] = np.nan
    return out


def tv_weighted_index(close: pd.DataFrame, trade_value: pd.DataFrame, market: pd.Series, *,
                      window: int = INDEX_ATV_WINDOW, cap: float = INDEX_CAP) -> dict[str, pd.DataFrame]:
    """거래대금 가중 지수 대용(시장별) → ``{"index_ret", "index_n", "index_maxw"}`` (각 날짜×시장).

    가중 = ``trade_value`` 의 직전 ``window`` 일(``d−window..d−1``) 평균 — ``d`` 수익을 재는 가중이 ``d−1`` 종가 때 이미 정해진다(PIT).
    그날 수익률이 없거나 |r| > :data:`RET_LIMIT` 이면 그 종목을 **수익과 가중 둘 다**에서 뺀 뒤 시장마다 :func:`capped_weights`.
    전 종목을 한 번에 다룬다(상한·재분배가 시장 전체에 걸린 비선형이라 종목 덩어리로 나눌 수 없다 — 패널 하나 ≈ 2,469×3,108 float)."""
    r = valid_returns(close)
    atv = trade_value.rolling(window, min_periods=window).mean().shift(1)
    rv = r.to_numpy(dtype=float)
    av = atv.to_numpy(dtype=float)
    ok = np.isfinite(rv) & np.isfinite(av) & (av > 0)
    out: dict[str, dict[str, np.ndarray]] = {"index_ret": {}, "index_n": {}, "index_maxw": {}}
    for m, mask in _market_masks(close.columns, market).items():
        if not mask.any():
            for k in out:
                out[k][m] = np.full(len(close.index), np.nan)
            out["index_n"][m] = np.zeros(len(close.index))
            continue
        wm = np.where(ok[:, mask], av[:, mask], 0.0)
        wn = capped_weights(wm, cap)
        rr = np.where(ok[:, mask], rv[:, mask], 0.0)
        out["index_ret"][m] = np.nansum(wn * rr, axis=1) + np.where(np.isnan(wn).all(axis=1), np.nan, 0.0)
        out["index_n"][m] = (wm > 0).sum(axis=1).astype(float)
        out["index_maxw"][m] = np.nanmax(np.where(np.isnan(wn), -np.inf, wn), axis=1)
        out["index_maxw"][m][~np.isfinite(out["index_maxw"][m])] = np.nan
    return {k: pd.DataFrame(v, index=close.index)[list(MARKETS)] for k, v in out.items()}


def index_level(ret: pd.DataFrame) -> pd.DataFrame:
    """일수익률 → 누적 레벨(첫 유효일 전날 = 1). 결측일은 0 수익(레벨 유지)."""
    return (1.0 + ret.fillna(0.0)).cumprod().where(ret.notna().cummax())


def realized_vol(ret: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return np.log1p(ret).rolling(window, min_periods=window).std() * np.sqrt(YEAR)


def versus_ma(level: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    ma = level.rolling(window, min_periods=window).mean()
    return level / ma.where(ma > 0) - 1.0


def breadth_parts(close: pd.DataFrame, market: pd.Series, window: int) -> dict[str, pd.DataFrame]:
    ma = close.rolling(window, min_periods=window).mean()
    valid = ma.notna() & close.notna()
    above = ((close > ma) & valid).astype(float)
    return {f"br{window}_num": group_sum(above, market), f"br{window}_den": group_sum(valid.astype(float), market)}


def _breadth_all(close: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    """시장별 breadth — KOSPI·KOSDAQ 는 :func:`market_breadth` 와 같은 정의(분모 = 그날 MA 가 정의된 종목), ALL 은 두 시장의
    분자·분모 합. (``market_breadth`` 를 부르지 않고 같은 식을 시장 마스크로 낸다 — 테스트가 일치를 지킨다.)"""
    p = breadth_parts(close, market, window)
    den = p[f"br{window}_den"]
    return p[f"br{window}_num"] / den.where(den > 0)


def dispersion_parts(close: pd.DataFrame, market: pd.Series) -> dict[str, pd.DataFrame]:
    r = valid_returns(close)
    return {"disp_s": group_sum(r, market), "disp_ss": group_sum(r * r, market), "disp_n": group_count(r, market)}


def _std_from_parts(s: pd.DataFrame, ss: pd.DataFrame, n: pd.DataFrame, *, min_count: int = 2) -> pd.DataFrame:
    n = n.where(n >= min_count)
    return np.sqrt(((ss - s * s / n) / (n - 1)).clip(lower=0.0))


def dispersion(close: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """그날 유효 일수익률(|r| ≤ :data:`RET_LIMIT`)의 횡단면 표준편차(ddof=1, 시장별)."""
    p = dispersion_parts(close, market)
    return _std_from_parts(p["disp_s"], p["disp_ss"], p["disp_n"])


def nhnl_parts(high: pd.DataFrame, low: pd.DataFrame, market: pd.Series, *, year: int = YEAR) -> dict[str, pd.DataFrame]:
    prev_hi = high.shift(1).rolling(year, min_periods=year).max()
    prev_lo = low.shift(1).rolling(year, min_periods=year).min()
    return {"new_highs": group_sum((high > prev_hi).astype(float), market),
            "new_lows": group_sum((low < prev_lo).astype(float), market)}


def _nhnl_finalize(nh: pd.DataFrame, nl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tot = nh + nl
    return {"new_highs": nh, "new_lows": nl, "nh_nl_ratio": nh / tot.where(tot > 0)}


def new_high_low(high: pd.DataFrame, low: pd.DataFrame, market: pd.Series, *, year: int = YEAR) -> dict[str, pd.DataFrame]:
    """52주 신고가·신저가 종목 수와 비율 → ``{"new_highs", "new_lows", "nh_nl_ratio"}`` (각 날짜×시장).

    신고가 = 고가 > 직전 ``year`` 일 최고 고가(``d`` 자신 제외, 창이 꽉 찬 날만). 신저가는 저가 대칭."""
    p = nhnl_parts(high, low, market, year=year)
    return _nhnl_finalize(p["new_highs"], p["new_lows"])


def flow_parts(net: pd.DataFrame, volume: pd.DataFrame, market: pd.Series) -> dict[str, pd.DataFrame]:
    ok = np.isfinite(net.to_numpy(dtype=float)) & np.isfinite(volume.to_numpy(dtype=float))
    return {"flow_n": group_sum(net.where(ok), market), "flow_v": group_sum(volume.where(ok), market)}


def _flow_finalize(n: pd.DataFrame, v: pd.DataFrame, *, window: int = 20, since: "str | None" = FLOW_SINCE) -> pd.DataFrame:
    n, v = n.copy(), v.copy()
    if since is not None:
        before = n.index < pd.Timestamp(since)
        n.loc[before] = np.nan
        v.loc[before] = np.nan
    n20 = n.rolling(window, min_periods=window).sum()
    v20 = v.rolling(window, min_periods=window).sum()
    return n20 / v20.where(v20 > 0)


def universe_flow(net: pd.DataFrame, volume: pd.DataFrame, market: pd.Series, *, window: int = 20,
                  since: "str | None" = FLOW_SINCE) -> pd.DataFrame:
    """유니버스 (외국인+기관) ``window`` 일 순매수 합 / 거래량 합 (시장별). ``since`` 전 날은 NaN 이고 창도 ``since`` 뒤에서만
    시작한다(창이 단절을 넘지 않게). 순매수·거래량이 둘 다 있는 (날, 종목) 만 센다."""
    p = flow_parts(net, volume, market)
    return _flow_finalize(p["flow_n"], p["flow_v"], window=window, since=since)


def breakout_parts(high: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, market: pd.Series, *,
                   window: int = 50, vol_mult: float = 1.5, horizon: int = 5) -> dict[str, pd.DataFrame]:
    brk, pivot = pivot_breakout(high, close, volume, window=window, vol_mult=vol_mult)
    judged = brk.shift(horizon).fillna(False).astype(bool) & close.notna()
    hit = judged & (close > pivot.shift(horizon))
    return {"brk_judged": group_sum(judged.astype(float), market), "brk_hit": group_sum(hit.astype(float), market)}


def _breakout_finalize(judged: pd.DataFrame, hit: pd.DataFrame, *, horizon: int = 5, lookback: int = 20) -> dict:
    span = lookback - horizon
    n = judged.rolling(span, min_periods=span).sum()
    k = hit.rolling(span, min_periods=span).sum()
    return {"breakouts20": n, "breakout_success20": k / n.where(n > 0)}


def breakout_success(high: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, market: pd.Series, *,
                     window: int = 50, vol_mult: float = 1.5, horizon: int = 5, lookback: int = 20) -> dict:
    """돌파 성공률(시장별) — ``d`` 행 = 돌파일 ``b ∈ [d−lookback+1, d−horizon]`` 인 돌파 중 ``close[b+horizon] > pivot[b]`` 비율.

    성공 여부는 ``b+horizon`` 에야 알 수 있으므로 그 날짜에 기록하고(``shift(horizon)``), ``lookback−horizon`` 창으로 센다 —
    ``d`` 행은 ``d`` 종가까지로 확정된 돌파만 담는다. 반환 ``breakouts20`` = 분모(결과 확정된 돌파 수)."""
    p = breakout_parts(high, close, volume, market, window=window, vol_mult=vol_mult, horizon=horizon)
    return _breakout_finalize(p["brk_judged"], p["brk_hit"], horizon=horizon, lookback=lookback)



# ---------------------------------------------------------------------------
# C. 스펙트럼 — 전부 인과(뒤쪽 창만)
# ---------------------------------------------------------------------------

def _windows(x: np.ndarray, window: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(x, window)


def _spectrum_grid(window: int, nfft: int, min_period: float, max_period: float):
    t = np.arange(window, dtype=float)
    X = np.column_stack([np.ones(window), t - t.mean()])
    proj = X @ np.linalg.inv(X.T @ X)  # (window, 2): coef = Y @ proj
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window) / window)
    freqs = np.fft.rfftfreq(nfft, d=1.0)  # cycles/day
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.where(freqs > 0, freqs, 1.0), np.inf)
    band = (periods >= min_period) & (periods <= max_period)
    return X, proj, hann, periods, band


def window_spectrum(Y: np.ndarray, *, nfft: int = SPEC_NFFT, min_period: float = SPEC_MIN_PERIOD,
                    max_period: float = SPEC_MAX_PERIOD) -> tuple[np.ndarray, np.ndarray]:
    """창 묶음(행 = 창, 열 = 시간, 결측 없음) → (지배 주기, 주엽 파워 비율). 창마다 선형 추세 제거 → 한 창 → 제로패딩 ``rfft``."""
    window = Y.shape[1]
    X, proj, hann, periods, band = _spectrum_grid(window, nfft, min_period, max_period)
    coef = Y @ proj
    detr = Y - coef @ X.T
    spec = np.abs(np.fft.rfft(detr * hann, n=nfft, axis=1)) ** 2
    spec[:, 0] = 0.0
    total = spec.sum(axis=1)
    k = np.where(band[None, :], spec, -1.0).argmax(axis=1)
    # 주엽(main lobe) 파워: 한 창의 주엽 반폭 2/window → 패딩 격자에서 ±2·nfft/window 빈
    cs = np.cumsum(spec, axis=1)
    half = 2 * nfft // window
    r = np.arange(len(k))
    hi_i = np.minimum(k + half, spec.shape[1] - 1)
    lo_i = k - half - 1
    lobe = cs[r, hi_i] - np.where(lo_i >= 0, cs[r, np.maximum(lo_i, 0)], 0.0)
    has_power = total > 0
    period = np.where(has_power, periods[k], np.nan)
    share = np.where(has_power, lobe / np.where(has_power, total, 1.0), np.nan)
    return period, share


def rolling_periodogram(x: "pd.Series | np.ndarray", *, window: int = SPEC_WINDOW, nfft: int = SPEC_NFFT,
                        min_period: int = SPEC_MIN_PERIOD, max_period: int = SPEC_MAX_PERIOD,
                        chunk: int = 512) -> pd.DataFrame:
    """롤링 주기도 → ``period``(지배 주기, 일)·``share``(그 봉우리 주엽 파워 / 전체 파워, DC 제외).

    행 ``d`` 는 ``x[d−window+1..d]`` 만 쓴다. 창 안에 NaN 이 하나라도 있으면 NaN. 패딩은 주파수 격자를 촘촘히 보간할 뿐(새 정보 없음).
    지배 주기는 ``[min_period, max_period]`` 안의 빈에서만 고른다. (유효성 거름은 :func:`spectral_block` 에서.)"""
    s = pd.Series(x) if not isinstance(x, pd.Series) else x
    v = s.to_numpy(dtype=float)
    n = len(v)
    period = np.full(n, np.nan)
    share = np.full(n, np.nan)
    if n < window:
        return pd.DataFrame({"period": period, "share": share}, index=s.index)
    wins = _windows(v, window)  # (n-window+1, window) view
    for lo in range(0, wins.shape[0], chunk):
        Y = wins[lo:lo + chunk]
        ok = np.isfinite(Y).all(axis=1)
        if not ok.any():
            continue
        pr, sh = window_spectrum(Y[ok], nfft=nfft, min_period=min_period, max_period=max_period)
        rows = np.arange(lo, lo + len(Y))[ok] + window - 1
        period[rows], share[rows] = pr, sh
    return pd.DataFrame({"period": period, "share": share}, index=s.index)


@lru_cache(maxsize=32)
def rw_share_threshold(window: int = SPEC_WINDOW, *, n_sim: int = RW_SIMS, q: float = RW_QUANTILE, seed: int = RW_SEED,
                       nfft: int = SPEC_NFFT, min_period: float = SPEC_MIN_PERIOD,
                       max_period: float = SPEC_MAX_PERIOD) -> float:
    """같은 길이(``window``) 무작위 보행 ``n_sim`` 벌의 ``share`` ``q`` 분위 — 지배 주기 유효성 문턱(리뷰 M4). 결정적(시드 고정)."""
    rng = np.random.default_rng(seed)
    walks = np.cumsum(rng.standard_normal((n_sim, window)), axis=1)
    _, share = window_spectrum(walks, nfft=nfft, min_period=min_period, max_period=max_period)
    return float(np.quantile(share, q))


def rw_threshold_table(windows: "tuple[int, ...]" = (64, 128, 256, 512), **kw) -> pd.Series:
    """창 길이별 무작위 보행 문턱 표(:func:`rw_share_threshold`) — 요약 MD 용."""
    return pd.Series({w: rw_share_threshold(w, **kw) for w in windows}, name="rw_share_q95").rename_axis("window")


def haar_details(x: "pd.Series", levels: int = HAAR_LEVELS) -> pd.DataFrame:
    """인과 Haar 세부계수 — 단계 ``j`` (척도 ``2**j``): 직전 ``2**(j−1)`` 일 평균 − 그 앞 ``2**(j−1)`` 일 평균.
    두 블록 다 ``d`` 이전·포함이라 미래를 보지 않는다. 블록이 꽉 찬 날부터 값이 있다."""
    out = {}
    for j in range(1, levels + 1):
        half = 2 ** (j - 1)
        recent = x.rolling(half, min_periods=half).mean()
        before = recent.shift(half)
        out[j] = recent - before
    return pd.DataFrame(out, index=x.index)


def haar_variance_share(x: "pd.Series", *, window: int = SPEC_WINDOW, levels: int = HAAR_LEVELS) -> pd.DataFrame:
    """척도별 세부계수의 롤링 ``window`` 분산 비율(열 ``1..levels``, 행 합 = 1). 창 안 결측이 있으면 NaN."""
    d = haar_details(x, levels)
    var = d.rolling(window, min_periods=window).var()
    tot = var.sum(axis=1, min_count=levels)
    return var.div(tot.where(tot > 0), axis=0)


def spectral_block(series: pd.Series, prefix: str, *, window: int = SPEC_WINDOW) -> pd.DataFrame:
    """한 시계열의 C 블록 — ``{prefix}_period``(무작위 보행 95 분위 초과일 때만, 아니면 NaN)·``{prefix}_share``·``{prefix}_haar1..4``."""
    pg = rolling_periodogram(series, window=window)
    hz = haar_variance_share(series, window=window)
    thr = rw_share_threshold(window)
    period = pg["period"].where(pg["share"] > thr)
    out = pd.DataFrame({f"{prefix}_period": period, f"{prefix}_share": pg["share"]}, index=series.index)
    for j in hz.columns:
        out[f"{prefix}_haar{j}"] = hz[j]
    return out


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------

@dataclass
class RegimeInputs:
    """날짜×종목 와이드 패널(분할조정·정규장 보정 완료, 거래대금 포함) + 종목→시장 + 수급(2024-02~, 거래일 달력으로 reindex)."""

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    market: pd.Series
    trade_value: pd.DataFrame                  # 날짜×종목 거래대금(백만원) — 지수 대용 가중
    flow_net: "pd.DataFrame | None" = None     # 외국인+기관 순매수 수량(날짜×종목, 결측 = 모름)
    flow_volume: "pd.DataFrame | None" = None  # supply_demand.acc_trde_qty
    official: "pd.DataFrame | None" = None     # 날짜×시장 공식 지수 종가(KOSPI·KOSDAQ, 있을 때만)
    patch_info: "dict | None" = None


FEATURES: tuple[str, ...] = (
    "n_codes", "breadth_ma50", "breadth_ma200", "index_n", "index_maxw", "index_level", "index_ret", "index_vs_ma200", "rv20",
    "idx_sign_ema20", "official_ret", "official_vs_ma200",
    "dispersion", "new_highs", "new_lows", "nh_nl_ratio", "flow20", "breakouts20", "breakout_success20",
    "idx_period", "idx_share", "idx_haar1", "idx_haar2", "idx_haar3", "idx_haar4",
    "br_period", "br_share", "br_haar1", "br_haar2", "br_haar3", "br_haar4",
)


def _chunks(codes: pd.Index, size: int) -> list[pd.Index]:
    return [codes[i:i + size] for i in range(0, len(codes), size)]


def accumulate_parts(inp: RegimeInputs, *, chunk: int = 400) -> dict[str, pd.DataFrame]:
    """종목을 ``chunk`` 개씩 잘라 시장별 합(parts)을 누적한다 — 메모리는 덩어리 하나치 임시 패널만 산다."""
    total: dict[str, pd.DataFrame] = {}
    for cols in _chunks(inp.close.columns, chunk):
        close, mk = inp.close[cols], inp.market
        parts: dict[str, pd.DataFrame] = {"n_codes": group_count(close, mk)}
        parts.update(breadth_parts(close, mk, 50))
        parts.update(breadth_parts(close, mk, 200))
        parts.update(dispersion_parts(close, mk))
        parts.update(nhnl_parts(inp.high[cols], inp.low[cols], mk))
        parts.update(breakout_parts(inp.high[cols], close, inp.volume[cols], mk))
        if inp.flow_net is not None and inp.flow_volume is not None:
            parts.update(flow_parts(inp.flow_net[cols], inp.flow_volume[cols], mk))
        for k, v in parts.items():
            total[k] = v if k not in total else total[k] + v
    return total


def build_regime_panel(inp: RegimeInputs, *, chunk: int = 400) -> pd.DataFrame:
    """:class:`RegimeInputs` → 긴 프레임 ``(date, market, *FEATURES)`` (시장 ∈ :data:`MARKETS`). 종목 덩어리별 누적(``chunk``)."""
    idx, markets = inp.close.index, list(MARKETS)
    p = accumulate_parts(inp, chunk=chunk)
    wide: dict[str, pd.DataFrame] = {"n_codes": p["n_codes"]}
    for w in (50, 200):
        den = p[f"br{w}_den"]
        wide[f"breadth_ma{w}"] = p[f"br{w}_num"] / den.where(den > 0)
    tv = tv_weighted_index(inp.close, inp.trade_value, inp.market)
    ret = tv["index_ret"].where(tv["index_n"] >= MIN_INDEX_N)
    wide["index_n"], wide["index_maxw"] = tv["index_n"], tv["index_maxw"].where(ret.notna())
    lvl = index_level(ret)
    wide["index_level"], wide["index_ret"] = lvl, ret
    wide["index_vs_ma200"] = versus_ma(lvl, 200)
    wide["rv20"] = realized_vol(ret, 20)
    wide["idx_sign_ema20"] = np.sign(ret).ewm(span=20, adjust=False, ignore_na=True).mean().where(ret.notna().cummax())
    if inp.official is not None and not inp.official.empty:
        off = inp.official.reindex(index=idx).astype(float).reindex(columns=markets)
        wide["official_ret"] = off.pct_change(fill_method=None)
        wide["official_vs_ma200"] = versus_ma(off, 200)
    else:
        wide["official_ret"] = pd.DataFrame(np.nan, index=idx, columns=markets)
        wide["official_vs_ma200"] = pd.DataFrame(np.nan, index=idx, columns=markets)
    wide["dispersion"] = _std_from_parts(p["disp_s"], p["disp_ss"], p["disp_n"])
    wide.update(_nhnl_finalize(p["new_highs"], p["new_lows"]))
    if "flow_n" in p:
        wide["flow20"] = _flow_finalize(p["flow_n"], p["flow_v"])
    else:
        wide["flow20"] = pd.DataFrame(np.nan, index=idx, columns=markets)
    wide.update(_breakout_finalize(p["brk_judged"], p["brk_hit"]))
    # C 블록 — 시장마다 지수 대용 로그 레벨·breadth_ma50 두 열
    spec = {m: pd.concat([spectral_block(np.log(lvl[m]), "idx"), spectral_block(wide["breadth_ma50"][m], "br")], axis=1)
            for m in MARKETS}
    frames = []
    for m in MARKETS:
        f = pd.DataFrame({k: v[m] for k, v in wide.items()}, index=idx)
        f = pd.concat([f, spec[m]], axis=1)
        f.insert(0, "market", m)
        frames.append(f)
    out = pd.concat(frames).rename_axis("date").reset_index()
    return out.sort_values(["date", "market"]).reset_index(drop=True)[["date", "market", *FEATURES]]


# ---------------------------------------------------------------------------
# DB 적재(읽기 전용) — 메모리 ≤ 4GB: 가격은 서버측 커서로 와이드 패널에 바로 채우고, 수급은 연 단위로 나눠 읽는다
# ---------------------------------------------------------------------------

PRICE_COLS = ("open", "high", "low", "close", "volume", "trade_value")


def _wide(df: pd.DataFrame, col: str, days: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    return df.pivot(index="date", columns="code", values=col).astype(float).reindex(index=days, columns=codes)


def patch_wide_nxt(panels: dict[str, pd.DataFrame], sess: pd.DataFrame, *, since: str) -> dict:
    """와이드 패널의 ``since`` 이후 행만 긴 모양으로 떼어 :func:`swing_it.krx_session.patch_prices` (조정계수 = 행 open 비율)
    로 정규장 값으로 바꾸고 다시 꽂는다. 반환 = 보정 통계."""
    from swing_it.krx_session import patch_prices  # noqa: PLC0415

    days = panels["close"].index
    late = days >= pd.Timestamp(since)
    if not late.any():
        return {"nxt_rows": 0, "nxt_patched": 0}
    long = pd.concat({c: panels[c].loc[late].stack(future_stack=True) for c in PRICE_COLS}, axis=1)
    long = long.dropna(how="all").rename_axis(["date", "code"]).reset_index()
    patched = patch_prices(long, sess, since=since, adjusted=True)
    for c in PRICE_COLS:
        w = patched.pivot(index="date", columns="code", values=c)
        panels[c].loc[w.index, w.columns] = w.to_numpy()
    return {"nxt_rows": int(len(patched)), "nxt_patched": int(patched["krx_patched"].sum())}


def load_regime_inputs(con: Any, *, start: "str | None" = None, end: "str | None" = None,
                       patch_nxt: bool = True, flow_since: str = FLOW_SINCE) -> RegimeInputs:
    """DB → :class:`RegimeInputs`. 가격·거래대금은 정문의 와이드판(:func:`swing_it.storage.read_prices_wide`, 폐지 포함 검사, 서버측 커서)
    → 2026-09-14~ 정규장 보정(:mod:`swing_it.krx_session`). 수급은 :func:`swing_it.storage.read_supply_demand` 정문을
    ``flow_since`` 부터 **연 단위**로 나눠 읽는다(긴 프레임 한 번에 안 만든다)."""
    from swing_it.features.korea import load_listing_meta, market_of_code  # noqa: PLC0415
    from swing_it.krx_session import NXT_START, read_session_bars_agg  # noqa: PLC0415
    from swing_it.storage import read_prices_wide, read_supply_demand  # noqa: PLC0415

    panels = read_prices_wide(con, session="raw", cols=PRICE_COLS)
    all_days = panels["close"].index
    lo = pd.Timestamp(start) if start else all_days.min()
    hi = pd.Timestamp(end) if end else all_days.max()
    keep = (all_days >= lo) & (all_days <= hi)
    panels = {c: p.loc[keep] for c, p in panels.items()}
    days, codes = panels["close"].index, panels["close"].columns
    info: dict = {"nxt_rows": 0, "nxt_patched": 0}
    if patch_nxt and hi >= pd.Timestamp(NXT_START):
        sess = read_session_bars_agg(con, start=max(lo, pd.Timestamp(NXT_START)).strftime("%Y-%m-%d"),
                                     end=hi.strftime("%Y-%m-%d"))
        info = patch_wide_nxt(panels, sess, since=NXT_START)
        del sess

    meta = load_listing_meta(con)
    market = market_of_code(meta, codes)

    flow_net = flow_vol = None
    if hi >= pd.Timestamp(flow_since):
        nets, vols = [], []
        for y in range(pd.Timestamp(flow_since).year, hi.year + 1):
            y_lo = max(pd.Timestamp(flow_since), pd.Timestamp(f"{y}-01-01"))
            y_hi = min(hi, pd.Timestamp(f"{y}-12-31"))
            if y_lo > y_hi:
                continue
            sd = read_supply_demand(con, session="raw", cols=("code", "date", "foreign_", "institution", "acc_trde_qty"),
                                    start=y_lo.strftime("%Y-%m-%d"), end=y_hi.strftime("%Y-%m-%d"))
            sd["date"] = pd.to_datetime(sd["date"])
            sd["code"] = sd["code"].astype(str)
            sd = sd[sd["code"].isin(codes)]
            sd["net"] = sd["foreign_"].astype(float) + sd["institution"].astype(float)
            y_days = days[(days >= y_lo) & (days <= y_hi)]
            nets.append(_wide(sd, "net", y_days, codes))
            vols.append(_wide(sd, "acc_trde_qty", y_days, codes))
            del sd
        if nets:
            flow_net = pd.concat(nets).reindex(index=days)
            flow_vol = pd.concat(vols).reindex(index=days)

    from swing_it.storage import read_index  # noqa: PLC0415

    official = read_index(con, ["001", "101"], cols=("code", "date", "close"), with_level=False,   # ×100 정수 그대로(옛 값과 같음)
                          proxy="없음 — 2024-01-26 전은 official=NaN, 대용 국면은 별도 계산(RegimeInputs.official)")
    if official.empty:
        off = None
    else:
        official["date"] = pd.to_datetime(official["date"])
        off = official.pivot(index="date", columns="code", values="close").astype(float)
        off = off.rename(columns={v: k for k, v in OFFICIAL_INDEX.items()}).reindex(index=days)
    return RegimeInputs(close=panels["close"], high=panels["high"], low=panels["low"], volume=panels["volume"],
                        market=market, trade_value=panels["trade_value"], flow_net=flow_net, flow_volume=flow_vol, official=off,
                        patch_info=info)


def coverage_by_year(panel: pd.DataFrame, features: "tuple[str, ...]" = FEATURES, market: str = "ALL") -> pd.DataFrame:
    """연도×피처 유효 비율(``market`` 행 기준, 기본 ALL — ``official_vs_ma200`` 은 KOSPI/KOSDAQ 에만 있다) — 요약 MD 용."""
    p = panel[panel["market"] == market].copy()
    p["year"] = pd.to_datetime(p["date"]).dt.year
    return p.groupby("year")[list(features)].apply(lambda g: g.notna().mean()).round(3)


def index_sanity(panel: pd.DataFrame, *, since: str = OFFICIAL_SINCE, cap: float = INDEX_CAP) -> pd.DataFrame:
    """지수 대용 위생 수치(시장별) — 최대 단일 종목 가중, 최대 |일수익률|, ``since`` 이후 공식 지수 일수익률과의 상관·표본 수·
    누적 수익(대용 / 공식). 바(:data:`INDEX_CAP`·:data:`RET_LIMIT`·:data:`INDEX_MIN_CORR`) 통과 여부 열 ``ok_*`` 포함."""
    rows = {}
    for m in MARKETS:
        p = panel[panel["market"] == m].set_index("date")
        late = p.loc[p.index >= pd.Timestamp(since), ["index_ret", "official_ret"]].dropna()
        corr = float(late["index_ret"].corr(late["official_ret"])) if len(late) > 2 else np.nan
        rows[m] = {
            "max_weight": float(p["index_maxw"].max()),
            "max_abs_ret": float(p["index_ret"].abs().max()),
            "corr_official": corr,
            "n_official": len(late),
            "cum_proxy": float((1 + late["index_ret"]).prod() - 1) if len(late) else np.nan,
            "cum_official": float((1 + late["official_ret"]).prod() - 1) if len(late) else np.nan,
            "ok_weight": bool(p["index_maxw"].max() <= cap + 1e-9),
            "ok_ret": bool(p["index_ret"].abs().max() <= RET_LIMIT),
            "ok_corr": bool(np.isnan(corr) or corr >= INDEX_MIN_CORR) if m == "ALL" else bool(corr >= INDEX_MIN_CORR),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
