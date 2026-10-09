"""스윙 라이브 신호 피처 — DB → :func:`swing_it.live.strategy.scan` 이 읽는 패널 한 벌 (설계 §3 ``features.py``).

연구 경로(``research/experiments/sepa_confluence_gate.Data`` + ``convergence_run``)와 **같은 함수**(``swing_it.features.*``)
를 같은 순서로 부른다 — 경계 규칙상 연구 코드를 import 할 수 없어 조립만 여기서 다시 한다. 같은 날짜에서 같은 플래그가
나오는지는 ``tests/test_live_strategy.py`` (합성)와 ``research/experiments/live_portfolio_backtest.py --check-features``
(실데이터, 연구 경로와 대조)가 확인한다.

패널(전부 날짜×종목, ``d`` 행 = ``d`` 종가까지):

- ``universe``: ``korea.universe_mask(raw_close=daily_bars.close, adv_floor=cfg.adv_floor)`` — 위생 ∧ ADV20 ≥ 50억.
  연구 CONVERGENCE 의 H5(PEAD 순위 모집단 = U_s ∪ main)와 같은 마스크다.
- ``tt``·``rs``·``breakout``·``pivot``·``breadth``·``ma50``: ``sepa.skeleton_features`` 와 같은 호출(minervini 모듈).
  RS·breadth 모집단은 가격 패널 전 종목(시장 구분 = ``stocks``+``delisted_stocks``, ``sepa._market_of`` 와 같은 매핑).
- ``pead_rank``: ``earnings_leg_panel`` → ``pead_rank(eligible = 그날 universe)`` (``convergence_run.repoint_pead``).
- ``rel``: (t−10..t−1 고가 최대 / 저가 최소 − 1) ÷ 그 값의 t−260..t−11 이동 중앙값(≥120개) — ``convergence_run.
  relative_contraction`` 과 같은 식.
- ``F``: ``flow_leg(weight=거래대금)`` 의 F(1/0/NaN).
- ``contraction_low``: ``min(low[t−10..t])`` (``minervini_exits.contraction_low_level``).
- ``adv20``: 20일 평균 거래대금(백만원).

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
from swing_it.features.minervini import breadth, pivot_breakout, rs_rating, trend_template
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
    brk, pivot = pivot_breakout(h, c, v, window=cfg.pivot_window, vol_mult=cfg.vol_mult)
    out = {
        "open": inp.open, "high": h, "low": lo, "close": c, "volume": v,
        "market": inp.market,
        "universe": uni,
        "ma50": moving_average(c, cfg.trail_ma),
        "tt": trend_template(c, high=h, low=lo),
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
    log("flow")
    ep = earnings_leg_panel(inp.earnings, [d.strftime("%Y-%m-%d") for d in inp.days])
    out["pead_rank"] = pead_rank_panel(ep, uni)
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
    return RawInputs(raw_close=raw_close, market=market_map(stocks, delisted, codes), meta=meta, supply=sd,
                     earnings=ea, patch_info=info, **piv)


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
