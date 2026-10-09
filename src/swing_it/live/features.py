"""스윙 라이브 신호 피처 — DB → :func:`swing_it.live.strategy.scan` 이 읽는 패널 한 벌 (설계 §3 ``features.py``).

연구 경로(``research/experiments/sepa_confluence_gate.Data`` + ``convergence_run``)와 **같은 함수**(``swing_it.features.*``)
를 같은 순서로 부른다 — 경계 규칙상 연구 코드를 import 할 수 없어 조립만 여기서 다시 한다. 같은 날짜에서 같은 플래그가
나오는지는 ``tests/test_live_strategy.py`` (합성)와 ``research/experiments/live_portfolio_backtest.py --check-features``
(실데이터, 연구 경로와 대조)가 확인한다.

패널(전부 날짜×종목, ``d`` 행 = ``d`` 종가까지):

- ``universe``: ``korea.universe_mask(raw_close=daily_bars.close, adv_floor=cfg.adv_floor)`` — 위생 ∧ ADV20 ≥ 50억.
  연구 CONVERGENCE 의 H5(PEAD 순위 모집단 = U_s ∪ main)와 같은 마스크다.
- ``tt``·``rs``·``breakout``·``pivot``·``breadth``·``ma50``: 기본 cfg 에서 ``sepa.skeleton_features`` 와 같은 값(테스트가 못박는다);
  cfg 가 바뀌면 GA2 유전자 의미(:func:`trend_mask`·:func:`breakout_mask`, ``trail_ma=0`` → ``ma50`` NaN)를 따른다.
  RS·breadth 모집단은 가격 패널 전 종목(시장 구분 = ``stocks``+``delisted_stocks``, ``sepa._market_of`` 와 같은 매핑).
- ``pead_rank``: ``earnings_leg_panel`` → ``pead_rank(eligible = 그날 universe)`` (``convergence_run.repoint_pead``).
- ``rel``: (t−10..t−1 고가 최대 / 저가 최소 − 1) ÷ 그 값의 t−260..t−11 이동 중앙값(≥120개) — ``convergence_run.
  relative_contraction`` 과 같은 식.
- ``F``: ``flow_leg(weight=거래대금)`` 의 F(1/0/NaN).
- ``contraction_low``: ``min(low[t−10..t])`` (``minervini_exits.contraction_low_level``).
- ``adv20``: 20일 평균 거래대금(백만원).
- ``eps_yoy``·``accel``·``rev_yoy``·``eps_from_netinc``: 실적 문(GA2 유전자, 기본 끔) — :func:`earnings_panels_ga` (GA 커널 정의:
  분기 말 as-of 분할 조정 주식수, :func:`eps_adjusted_earnings`).
- ``nh``·``al``·``stack``·``slope10/20/40``: 추세 성분(:func:`trend_components`) — ``tt`` 는 cfg 의 4성분(:func:`trend_mask`).
- ``smart``·``foreign``·``persist``: 수급 유전자용(``flow_leg`` 의 smart_ratio_20·foreign_ratio_20·smart_pos_days_20).

**그날치(라이브):** :func:`signal_panels_for` 는 신호일까지 :data:`LOOKBACK_ROWS` 거래일만 쓴다. 필요한 최장 이력은 rel
분모(t−270 행)·RS(252일 수익률)·52주 고저(252행)·MA200 기울기(220행)다 — 300행이면 그날 값이 전 기간 패널과 같다.
상장일 수(``listing_age``)는 창 시작에서 좌측 절단되지만 창이 252행보다 길어 "≥ 252" 판정은 같다.

**NXT 보정(2026-09-14~):** 일봉 OHLCV 에 NXT 야간 체결이 섞였다(:mod:`swing_it.krx_session`). ``patch_nxt=True``
(기본)면 분봉으로 만든 정규장 값으로 바꾼다 — 조정 패널은 시가 비율, 원자료 종가(1,000원 규칙)는 그대로 덮어쓴다.
분봉이 없는 (종목, 날짜) 는 원값이 남고 ``patch_info`` 에 센다. 연구 경로는 보정하지 않으므로 대조는 보정 전 날짜에서 한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from swing_it.features.earnings_leg import earnings_leg_panel, pead_rank
from swing_it.features.flow_leg import flow_leg
from swing_it.features.korea import load_listing_meta, universe_mask
from swing_it.features.minervini import breadth, pivot_level, rs_rating
from swing_it.features.technical import average_trade_value, moving_average
from swing_it.live.config import DEFAULT_STRATEGY, StrategyConfig

#: 그날치 계산에 쓰는 거래일 수(신호일 포함). 하한은 :func:`min_lookback_rows` (기본 설정 271) — 여유 29행.
LOOKBACK_ROWS = 300
#: 백테스트 패널 시작(연구 ``sepa_confluence_gate.PANEL_START`` 와 같다 — 데이터는 2016-09-09 부터).
PANEL_START = "2016-09-01"
PRICE_COLS = ("open", "high", "low", "close", "volume", "trade_value")
SUPPLY_COLS = ("code", "date", "foreign_", "institution", "acc_trde_qty")
EARN_COLS = ("code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior", "revenue", "revenue_prior",
             "op_income", "op_income_prior")


@dataclass
class RawInputs:
    """DB 에서 읽은 원재료(가격은 날짜×종목 넓은 패널, 수급·실적은 롱)."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    trade_value: pd.DataFrame
    raw_close: pd.DataFrame          # daily_bars.close (1,000원 규칙)
    market: pd.Series                # code → KOSPI/KOSDAQ/X
    meta: pd.DataFrame               # korea.load_listing_meta
    supply: pd.DataFrame             # read_supply_demand(SUPPLY_COLS)
    earnings: pd.DataFrame           # read_earnings(all_versions=True)
    patch_info: dict = field(default_factory=dict)
    shares: "pd.DataFrame | None" = None   # shares_outstanding_history (code, date, shares_outstanding, knowledge_date)

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.close.index


# ---------------------------------------------------------------------------
# 순수 조립
# ---------------------------------------------------------------------------

def market_map(stocks: pd.DataFrame, delisted: pd.DataFrame, codes: pd.Index) -> pd.Series:
    """code → KOSPI/KOSDAQ/X. 폐지 마스터를 먼저, 현재 마스터를 뒤에(재상장은 현재 값) — ``sepa._market_of`` 와 같다."""
    m = pd.concat([delisted[["code", "market"]], stocks[["code", "market"]]]).drop_duplicates("code", keep="last")
    m = m.set_index("code")["market"]
    m = m.map(lambda x: "KOSPI" if x in ("거래소", "KOSPI", "유가증권")
              else ("KOSDAQ" if x in ("코스닥", "KOSDAQ") else "X"))
    return m.reindex(codes).fillna("X")


def min_lookback_rows(cfg: StrategyConfig = DEFAULT_STRATEGY) -> int:
    """그날 값이 전 기간 패널과 같으려면 필요한 최소 행 수 = ``rel_med_window + rel_gap + rel_window`` (기본 271).

    rel 분모는 t−(gap+med_window−1)…t−gap 행의 rng 를 쓰고 rng 는 그 행의 w 행 전(t−gap−med_window+1−w = t−270)까지
    본다 — t 포함 271 행. 창이 이보다 짧으면
    ``rel_min_periods``(120) 때문에 오류 없이 **다른 중앙값**이 나온다(REVIEW_BACKTEST m1 — 262행에서 2,109 종목이 달라졌다).
    RS·52주 고저(252)·MA200 기울기(220)·상장일 수(252)는 이보다 짧다.
    """
    return cfg.rel_med_window + cfg.rel_gap + cfg.rel_window


def relative_contraction(high: pd.DataFrame, low: pd.DataFrame, cfg: StrategyConfig = DEFAULT_STRATEGY
                         ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """상대 수렴 → ``(rng, rel)``. ``rng[t] = max(high[t−w..t−1]) / min(low[t−w..t−1]) − 1``,
    ``rel[t] = rng[t] / median(rng[t−gap−m+1..t−gap])`` (유효 ≥ ``min_periods``). CONVERGENCE.md 정의 그대로."""
    w = cfg.rel_window
    hi = high.shift(1).rolling(w, min_periods=w).max()
    lo = low.shift(1).rolling(w, min_periods=w).min()
    rng = hi / lo.where(lo > 0) - 1.0
    med = rng.rolling(cfg.rel_med_window, min_periods=cfg.rel_min_periods).median().shift(cfg.rel_gap)
    return rng, rng / med.where(med > 0)


def contraction_low(low: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """수축 저점 = ``min(low[t−w..t])`` (돌파일 포함 ``w+1`` 거래일)."""
    return low.rolling(window + 1, min_periods=window + 1).min()


def _long_to_wide(earn_long: pd.DataFrame, col: str, days: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    di = days.get_indexer(pd.to_datetime(earn_long["date"]))
    ci = codes.get_indexer(earn_long["code"])
    v = earn_long[col].to_numpy(dtype=float)
    ok = (di >= 0) & (ci >= 0) & np.isfinite(v)
    out = np.full((len(days), len(codes)), np.nan)
    out[di[ok], ci[ok]] = v[ok]
    return pd.DataFrame(out, index=days, columns=codes)


def trend_components(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> dict:
    """GA2 추세 성분(``ga_run`` 사전계산과 같은 식): ``nh`` = close / 252일 장중 고가(당일 포함, 창 꽉 찬 날만), ``al`` = close /
    252일 저가, ``stack`` = (close>MA50)+(MA50>MA150)+(MA150>MA200) 참 개수(NaN 비교 = 거짓), ``slope{10,20,40}`` = MA200 >
    MA200[t−k] (bool). 원전 템플릿(:func:`trend_template`) = nh ≥ 0.75 ∧ al ≥ 1.30 ∧ stack ≥ 3 ∧ slope20."""
    hi = high.rolling(252, min_periods=252).max()
    lo = low.rolling(252, min_periods=252).min()
    ma50, ma150, ma200 = (moving_average(close, w) for w in (50, 150, 200))
    out = {"nh": close / hi, "al": close / lo,
           "stack": ((close > ma50).astype(int) + (ma50 > ma150).astype(int) + (ma150 > ma200).astype(int))}
    for k in (10, 20, 40):
        out[f"slope{k}"] = ma200 > ma200.shift(k)
    return out


def trend_mask(tc: dict, cfg: StrategyConfig = DEFAULT_STRATEGY) -> pd.DataFrame:
    """cfg 의 4성분으로 만든 추세 템플릿 bool 패널(끔 = 조건 없음, NaN 통과). 기본 cfg 면 :func:`trend_template` 과 같다."""
    like = tc["stack"]
    ok = pd.DataFrame(True, index=like.index, columns=like.columns)
    if cfg.near_high is not None:
        ok &= (tc["nh"] >= cfg.near_high).fillna(False)
    if cfg.above_low is not None:
        ok &= (tc["al"] >= cfg.above_low).fillna(False)
    if cfg.ma_stack > 0:
        ok &= tc["stack"] >= cfg.ma_stack
    if cfg.ma200_slope_days > 0:
        if cfg.ma200_slope_days not in (10, 20, 40):
            raise ValueError(f"ma200_slope_days 는 10/20/40 중 하나: {cfg.ma200_slope_days}")
        ok &= tc[f"slope{cfg.ma200_slope_days}"].fillna(False)
    return ok


def breakout_mask(high: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, cfg: StrategyConfig = DEFAULT_STRATEGY
                  ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """cfg 의 돌파·거래량 조건 → ``(breakout bool, pivot)``. 피벗·거래량 평균 창 = ``cfg.effective_pivot_window``(돌파 요구 없음이면
    50일 — GA 커널과 같다). ``pivot_window=0`` 이면 돌파 조건 없음, ``vol_mult=None`` 이면 거래량 조건 없음(둘 다 끔 = 전부 참).
    거래량 조건은 ``vavg > 0 ∧ vol ≥ x·vavg`` (:func:`pivot_breakout`·GA 마스크 모두 같다)."""
    w = cfg.effective_pivot_window
    pivot = pivot_level(high, w)
    ok = pd.DataFrame(True, index=close.index, columns=close.columns)
    if cfg.pivot_window > 0:
        ok &= (close > pivot).fillna(False)
    if cfg.vol_mult is not None:
        vavg = volume.shift(1).rolling(w, min_periods=w).mean()
        ok &= ((vavg > 0) & (volume >= cfg.vol_mult * vavg)).fillna(False)
    return ok, pivot


def eps_adjusted_earnings(ea: pd.DataFrame, shares: "pd.DataFrame | None", adj_close: pd.DataFrame,
                          raw_close: pd.DataFrame) -> pd.DataFrame:
    """GA 커널(``ga_run.eps_features``)과 **같은 식**으로 순이익 자리에 EPS 를 넣은 실적 프레임.

    주식수 = ``shares_outstanding_history`` 를 그 날짜 직전 거래일의 조정계수(조정 종가 / 원 종가)로 나눈 값(분할 조정)의
    **분기 말 as-of**(이번 분기 말·전년 같은 분기 말). 둘 다 있으면 ``netinc/주식수``·``netinc_prior/주식수_전년``, 아니면 순이익
    그대로. ``op_income`` 자리에 표시(2.0/1.0 → ``op_yoy`` 1 = EPS, 0 = 순이익 대체)를 넣는 트릭까지 같다 — 그래서
    ``earnings_leg_panel(결과)`` 의 ``yoy``·``yoy_prev``·``rev_yoy``·``op_yoy`` 가 GA 의 ``eps_yoy``·가속·``rev_yoy``·``eps_ni`` 다.
    (분기 말 주식수는 그 분기 보고서와 함께 알려지므로 실적 가용일 기준 시점 안전 — GA 의 가정 그대로.)
    """
    ea = ea.copy()
    ea["code"] = ea["code"].astype(str)
    if shares is None or shares.empty:
        now = prev = np.full(len(ea), np.nan)
    else:
        sh = shares[["code", "date", "shares_outstanding"]].copy()
        sh["code"] = sh["code"].astype(str)
        sh["date"] = pd.to_datetime(sh["date"]).astype("datetime64[ns]")
        sh = sh[sh["shares_outstanding"] > 0]
        fac = (adj_close / raw_close.reindex_like(adj_close)).ffill()
        ci_s = fac.columns.get_indexer(sh["code"])
        di_s = fac.index.searchsorted(sh["date"], "right") - 1
        ok = (ci_s >= 0) & (di_s >= 0)
        f = np.full(len(sh), np.nan)
        f[ok] = fac.to_numpy()[di_s[ok], ci_s[ok]]
        sh["sh_adj"] = sh["shares_outstanding"] / f
        sh = sh[np.isfinite(sh["sh_adj"]) & (sh["sh_adj"] > 0)].sort_values(["date", "code"])
        q = ea["period"].astype(str)
        pe = (pd.to_datetime(q.str[:4] + "-" + (q.str[5].astype(int) * 3).astype(str).str.zfill(2) + "-01")
              + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
        tmp = ea.assign(_pe=pe, _pe0=(pe - pd.DateOffset(years=1)).astype("datetime64[ns]"), _row=np.arange(len(ea)))
        got = {}
        for col, key in (("_pe", "now"), ("_pe0", "prev")):
            m = pd.merge_asof(tmp[["code", col, "_row"]].sort_values(col).rename(columns={col: "date"}),
                              sh[["code", "date", "sh_adj"]], on="date", by="code", direction="backward")
            got[key] = m.sort_values("_row")["sh_adj"].to_numpy()
        now, prev = got["now"], got["prev"]
    have = np.isfinite(now) & np.isfinite(prev)
    out = ea[["code", "period", "avail_date", "knowledge_date", "revenue", "revenue_prior"]].copy()
    out["netinc"] = np.where(have, ea["netinc"].to_numpy(float) / np.where(have, now, 1.0), ea["netinc"].to_numpy(float))
    out["netinc_prior"] = np.where(have, ea["netinc_prior"].to_numpy(float) / np.where(have, prev, 1.0),
                                   ea["netinc_prior"].to_numpy(float))
    out["op_income"] = np.where(have, 2.0, 1.0)
    out["op_income_prior"] = 1.0
    return out


def earnings_panels_ga(ea: pd.DataFrame, shares: "pd.DataFrame | None", adj_close: pd.DataFrame, raw_close: pd.DataFrame,
                       days: pd.DatetimeIndex, codes: pd.Index) -> dict:
    """실적 문 패널(GA 커널 정의) — ``eps_yoy`` · ``accel`` · ``rev_yoy`` · ``eps_from_netinc``(날짜×종목).
    :func:`eps_adjusted_earnings` → ``earnings_leg_panel`` → 넓은 패널. 가속 = ``accel_flag``(NaN → 0, GA 와 같다), 실적 행 없음 = NaN."""
    from swing_it.features.earnings_leg import accel_flag  # noqa: PLC0415

    ep = earnings_leg_panel(eps_adjusted_earnings(ea, shares, adj_close, raw_close),
                            [d.strftime("%Y-%m-%d") for d in days])
    if ep.empty:
        nan = pd.DataFrame(np.nan, index=days, columns=codes)
        return {"eps_yoy": nan, "accel": nan.copy(), "rev_yoy": nan.copy(), "eps_from_netinc": nan.copy()}
    ep = ep.assign(accel=accel_flag(ep).astype(float), eps_from_netinc=(ep["op_yoy"] == 0).astype(float))
    ep = ep.rename(columns={"yoy": "eps_yoy"})
    return {k: _long_to_wide(ep, k, days, codes) for k in ("eps_yoy", "accel", "rev_yoy", "eps_from_netinc")}


def pead_rank_panel(earn_long: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """실적 롱 패널 → 그날 ``universe`` 안 YoY 횡단면 백분위(날짜×종목). 밖·모름은 NaN."""
    days, codes = universe.index, universe.columns
    if earn_long.empty:
        return pd.DataFrame(np.nan, index=days, columns=codes)
    di = days.get_indexer(pd.to_datetime(earn_long["date"]))
    ci = codes.get_indexer(earn_long["code"])
    ok = (di >= 0) & (ci >= 0)
    un = universe.to_numpy(dtype=bool)
    elig = np.zeros(len(earn_long), dtype=bool)
    elig[ok] = un[di[ok], ci[ok]]
    rk = pead_rank(earn_long, eligible=pd.Series(elig, index=earn_long.index)).to_numpy(dtype=float)
    out = np.full((len(days), len(codes)), np.nan)
    sel = ok & np.isfinite(rk)
    out[di[sel], ci[sel]] = rk[sel]
    return pd.DataFrame(out, index=days, columns=codes)


def build_signal_panels(inp: RawInputs, cfg: StrategyConfig = DEFAULT_STRATEGY, *, verbose: bool = False
                        ) -> dict:
    """원재료 → :data:`swing_it.live.strategy.SCAN_KEYS` 패널 + ``market``·``ma50``·``open``·``high``·``low``·``volume``.

    연구와 같은 함수·같은 인자다(모듈 설명). ``verbose`` 면 단계별 소요 시간을 찍는다.
    """
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"  [features {time.time() - t0:6.1f}s] {msg}", flush=True)

    c, h, lo, v, tv = inp.close, inp.high, inp.low, inp.volume, inp.trade_value
    uni = universe_mask(c, tv, inp.meta, raw_close=inp.raw_close, adv_floor=cfg.adv_floor,
                        adv_window=cfg.adv_window, min_price=cfg.min_price, min_history=cfg.min_history)
    log("universe")
    brk, pivot = breakout_mask(h, c, v, cfg)
    tc = trend_components(c, h, lo)
    nan_ma = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    out = {
        "open": inp.open, "high": h, "low": lo, "close": c, "volume": v,
        "market": inp.market,
        "universe": uni,
        "ma50": moving_average(c, cfg.trail_ma) if cfg.has_trail else nan_ma,   # trail_ma = 0 → 추적 없음(NaN)
        "tt": trend_mask(tc, cfg),
        "nh": tc["nh"], "al": tc["al"], "stack": tc["stack"],
        **{f"slope{k}": tc[f"slope{k}"] for k in (10, 20, 40)},
        "rs": rs_rating(c),
        "breakout": brk,
        "pivot": pivot,
        "breadth": breadth(c, inp.market),
        "adv20": average_trade_value(tv, cfg.adv_window),
        "contraction_low": contraction_low(lo, cfg.contraction_window),
    }
    log("skeleton")
    out["rel"] = relative_contraction(h, lo, cfg)[1]
    log("rel")
    fl = flow_leg(inp.supply, index=inp.days, columns=c.columns, weight=tv)
    out["F"] = fl["F"]
    out["smart"], out["foreign"] = fl["smart_ratio_20"], fl["foreign_ratio_20"]
    out["persist"] = fl["smart_pos_days_20"]
    log("flow")
    ep = earnings_leg_panel(inp.earnings, [d.strftime("%Y-%m-%d") for d in inp.days])
    out["pead_rank"] = pead_rank_panel(ep, uni)
    out.update(earnings_panels_ga(inp.earnings, inp.shares, c, inp.raw_close, inp.days, c.columns))
    log("pead")
    return out


# ---------------------------------------------------------------------------
# DB 적재
# ---------------------------------------------------------------------------

def _wide(df: pd.DataFrame, col: str, days: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    return df.pivot(index="date", columns="code", values=col).astype(float).reindex(index=days, columns=codes)


def load_inputs(con: Any, *, start: "str | None" = PANEL_START, end: "str | None" = None,
                last_rows: "int | None" = None, patch_nxt: bool = True) -> RawInputs:
    """DB → :class:`RawInputs`. 가격은 정문(:func:`swing_it.storage.read_prices` — 폐지 포함 검사) 뒤 날짜로 자른다.

    Args:
        start, end: 날짜 구간(포함). ``end=None`` = 끝까지.
        last_rows: 주면 ``end`` 까지의 마지막 ``last_rows`` 거래일만(``start`` 무시) — 그날치 라이브.
        patch_nxt: 2026-09-14~ 행을 정규장(분봉) 값으로 바꾼다(:mod:`swing_it.krx_session`).
    """
    from swing_it.krx_session import NXT_START, patch_prices, read_session_bars  # noqa: PLC0415
    from swing_it.storage import read_earnings, read_prices, read_supply_demand  # noqa: PLC0415

    prices = read_prices(con)
    prices["date"] = pd.to_datetime(prices["date"])
    hi = pd.Timestamp(end) if end is not None else prices["date"].max()
    prices = prices[prices["date"] <= hi]
    if last_rows is not None:
        all_days = np.sort(prices["date"].unique())
        lo = pd.Timestamp(all_days[max(len(all_days) - last_rows, 0)])
    else:
        lo = pd.Timestamp(start) if start is not None else prices["date"].min()
    prices = prices[prices["date"] >= lo]
    raw = read_prices(con, cols=("code", "date", "open", "close"), table="daily_bars")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= lo) & (raw["date"] <= hi)]

    # 정수 열(거래량·거래대금)에 정규장 float 를 덮어쓰면 pandas 가 거부한다 — 먼저 float 로.
    prices = prices.astype({c: float for c in PRICE_COLS})
    raw = raw.astype({"open": float, "close": float})
    info: dict = {"nxt_rows": 0, "nxt_patched": 0}
    if patch_nxt and hi >= pd.Timestamp(NXT_START):
        sess = read_session_bars(con, start=max(lo, pd.Timestamp(NXT_START)).strftime("%Y-%m-%d"),
                                 end=hi.strftime("%Y-%m-%d"))
        prices = patch_prices(prices, sess, adjusted=True)
        raw = patch_prices(raw, sess, adjusted=False)
        late = prices["date"] >= pd.Timestamp(NXT_START)
        info = {"nxt_rows": int(late.sum()), "nxt_patched": int(prices.loc[late, "krx_patched"].sum()),
                "nxt_unpatched": prices.loc[late & ~prices["krx_patched"], ["code", "date"]].reset_index(drop=True)}

    days = pd.DatetimeIndex(np.sort(prices["date"].unique()))
    codes = pd.Index(np.sort(prices["code"].astype(str).unique()))
    piv = {col: _wide(prices, col, days, codes) for col in PRICE_COLS}
    raw_close = _wide(raw, "close", days, codes)

    stocks = pd.read_sql_query("SELECT code, market FROM stocks", con)
    delisted = pd.read_sql_query("SELECT code, market FROM delisted_stocks", con)
    meta = load_listing_meta(con)
    sd = read_supply_demand(con, cols=SUPPLY_COLS, start=lo.strftime("%Y-%m-%d"), end=hi.strftime("%Y-%m-%d"))
    ea = read_earnings(con, all_versions=True, cols=EARN_COLS)
    shares = pd.read_sql_query("SELECT code, date, shares_outstanding, knowledge_date FROM shares_outstanding_history",
                               con)
    return RawInputs(raw_close=raw_close, market=market_map(stocks, delisted, codes), meta=meta, supply=sd,
                     earnings=ea, patch_info=info, shares=shares, **piv)


def signal_panels_for(con: Any, d: "str | pd.Timestamp | None" = None, cfg: StrategyConfig = DEFAULT_STRATEGY, *,
                      lookback: int = LOOKBACK_ROWS, patch_nxt: bool = True) -> "tuple[dict, pd.Timestamp]":
    """그날치(라이브 ``signal`` 작업): 신호일 ``d`` (None = DB 최신 거래일)까지 ``lookback`` 거래일 패널.

    Returns: ``(panels, d)`` — ``scan(panels, d, cfg)`` 로 후보를 얻는다. ``panels["patch_info"]`` 에 NXT 보정 건수.

    Raises:
        ValueError: ``lookback`` < :func:`min_lookback_rows` (rel 이 조용히 달라진다) 또는 신호일이 가격 테이블에 없음.
    """
    need = min_lookback_rows(cfg)
    if lookback < need:
        raise ValueError(f"lookback {lookback} < 최소 {need} 행(rel 분모 창) — 그날 값이 전 기간 패널과 달라진다")
    inp = load_inputs(con, end=None if d is None else str(pd.Timestamp(d).date()), last_rows=lookback,
                      patch_nxt=patch_nxt)
    day = inp.days[-1] if d is None else pd.Timestamp(d)
    if day not in inp.days:
        raise ValueError(f"신호일 {day.date()} 가 가격 테이블에 없다(휴장 또는 아직 적재 전)")
    panels = build_signal_panels(inp, cfg)
    panels["patch_info"] = inp.patch_info
    return panels, day


#: 감사 문서(AUDIT.md)·옛 호출부가 쓰던 이름 — :func:`signal_panels_for` 와 같다.
load_live_signal = signal_panels_for
