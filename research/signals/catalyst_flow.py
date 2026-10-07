"""공시 촉매 × 수급 확인 × 추세 — 신호·시뮬레이터 (사전등록: research/logs/catalyst_flow_swing/VERDICT.md).

신호는 :mod:`swing_it.features` 의 순수 함수 위에 얹는다. 이 파일이 하는 일은 세 가지다.

1. 패널 적재 — 가격(정문, 폐지 포함)·수급(정문, 외국인+기관)·시장 구분
2. 사건 표 — DART 호재 사건을 거래일에 맞추고, 사건일 종가 기준 조건들을 계산
3. 트레이드 시뮬레이션 — 사건일 다음 거래일 시가 진입, H 거래일째 종가 청산, 하드손절

시뮬레이터는 종목 축으로 벡터화돼 있어(:func:`simulate_block`) 정합 대조 풀(같은 날 전 종목)도 같은
규칙으로 계산한다 — 트레이드와 대조가 **같은 체결 규칙**을 쓰는 것이 대조의 전제다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swing_it.engine.panels import cached_panel_pivot
from swing_it.features.catalysts import align_to_trading_days, catalyst_events, read_dart
from swing_it.features.flows import net_to_volume, smart_money_net
from swing_it.features.technical import average_trade_value, moving_average, simple_return
from swing_it.storage import read_prices, read_supply_demand

# --- 사전등록 config (이게 테스트) ---
PREREG = {
    "hold": 10,             # 보유 거래일(진입일 포함, H 번째 날 종가 청산)
    "stop": 0.08,           # 하드손절
    "max_event_ret": 0.08,  # 사건일 수익률 상한
    "ma_window": 60,        # 추세 MA
    "adv_floor": 5000.0,    # 20일 평균 거래대금(백만원) 하한 = 50억
    "min_price": 1000.0,
    "require_flow": True,   # 외국인+기관 순매수 > 0
    "require_trend": True,  # 종가 ≥ MA60
    "require_calm": True,   # 사건일 수익률 ≤ max_event_ret
    "families": ("buyback", "capex", "supply_contract"),
    "event_lo": "2024-01-22",
    "event_hi": "2026-09-23",
}


@dataclass
class Panels:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    trade_value: pd.DataFrame
    net: pd.DataFrame          # 외국인+기관 순매수 수량
    market: pd.Series          # code → "KOSPI"/"KOSDAQ"/"X"

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.close.index


def load_panels(con, *, start: str = "2023-06-01") -> Panels:
    """가격·수급 패널. 가격은 전 기간을 읽되(정문의 폐지 검사) ``start`` 이후로 자른다."""
    prices = read_prices(con)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["date"] >= pd.Timestamp(start)]
    piv = {c: cached_panel_pivot(prices, c) for c in ("open", "high", "low", "close", "volume", "trade_value")}
    for k, v in piv.items():
        v.index = pd.to_datetime(v.index)
        piv[k] = v.sort_index().astype(float)
    sd = read_supply_demand(con, cols=("code", "date", "foreign_", "institution"), start=start)
    net = smart_money_net(sd).reindex(index=piv["close"].index, columns=piv["close"].columns)
    mk = pd.read_sql_query("SELECT code, market FROM stocks", con)
    dk = pd.read_sql_query("SELECT code, market FROM delisted_stocks", con)
    m = pd.concat([dk, mk]).drop_duplicates("code", keep="last").set_index("code")["market"]
    m = m.map(lambda x: "KOSPI" if x in ("거래소", "KOSPI", "유가증권") else ("KOSDAQ" if x in ("코스닥", "KOSDAQ") else "X"))
    market = m.reindex(piv["close"].columns).fillna("X")
    return Panels(net=net, market=market, **piv)


def feature_frame(p: Panels, *, ma_window: int) -> dict[str, pd.DataFrame]:
    """사건일 판단에 쓰는 패널들(전부 그날 종가까지)."""
    vol = p.volume.where(p.volume > 0)  # 거래정지(거래량 0, 종가 이월) 는 수익률·비율 계산에서 빼낸다
    close = p.close.where(vol.notna())
    return {
        "ret1": simple_return(close, 1),
        "ret5": simple_return(close, 5),
        "ma": moving_average(p.close, ma_window),
        "adv20": average_trade_value(p.trade_value, 20),
        "flow_ratio": net_to_volume(p.net, vol),
    }


def build_events(p: Panels, cfg: dict, *, dart: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """사전등록 조건을 계산한 사건 표. 필터는 ``passes`` 열로 표시만 하고 행은 남긴다(민감도용)."""
    dart = read_dart() if dart is None else dart
    ev = align_to_trading_days(catalyst_events(dart), p.days)
    ev = ev[(ev["event_day"] >= pd.Timestamp(cfg["event_lo"])) & (ev["event_day"] <= pd.Timestamp(cfg["event_hi"]))]
    ev = ev[ev["code"].isin(p.close.columns)].reset_index(drop=True)
    fam_ok = ev["family"].map(lambda f: any(x in cfg["families"] for x in f.split("+")))
    ev = ev[fam_ok].reset_index(drop=True)

    f = feature_frame(p, ma_window=cfg["ma_window"])
    di = p.days.get_indexer(ev["event_day"])
    ci = p.close.columns.get_indexer(ev["code"])

    def pick(panel: pd.DataFrame) -> np.ndarray:
        return panel.to_numpy()[di, ci]

    ev["close"] = pick(p.close)
    ev["net"] = pick(p.net)
    for k in ("ret1", "ret5", "ma", "adv20", "flow_ratio"):
        ev[k] = pick(f[k])
    ev["market"] = p.market.reindex(ev["code"]).to_numpy()
    ev["day_idx"] = di
    ev["col_idx"] = ci

    liq = (ev["adv20"] >= cfg["adv_floor"]) & (ev["close"] >= cfg["min_price"])
    flow = ev["net"] > 0 if cfg["require_flow"] else pd.Series(True, index=ev.index)
    trend = ev["close"] >= ev["ma"] if cfg["require_trend"] else pd.Series(True, index=ev.index)
    calm = ev["ret1"] <= cfg["max_event_ret"] if cfg["require_calm"] else pd.Series(True, index=ev.index)
    ev["passes"] = liq & flow.fillna(False) & trend.fillna(False) & calm.fillna(False)
    return ev


def simulate_block(o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
                   entry_row: int, cols: np.ndarray, *, hold: int, stop: float) -> np.ndarray:
    """``entry_row`` 시가 진입 → ``hold`` 번째 거래일 종가 청산, 하드손절. 종목 축 벡터화.

    - 진입 시가가 NaN(정지·상장 전·폐지 후)이면 결과 NaN — 살 수 없었던 자리다.
    - 손절: 그날 시가가 이미 손절가 아래면 시가(갭), 아니면 저가가 닿을 때 손절가.
      진입일은 시가에 샀으므로 저가만 본다.
    - 중간에 시세가 끊기면(폐지·장기정지) 마지막 유효 종가로 청산 — 폐지 손실을 지우지 않는다.
    """
    n_rows = o.shape[0]
    entry = o[entry_row, cols]
    out = np.full(len(cols), np.nan)
    alive = np.isfinite(entry) & (entry > 0)
    stop_px = entry * (1.0 - stop)
    last_close = np.full(len(cols), np.nan)
    done = ~alive
    for k in range(hold):
        r = entry_row + k
        if r >= n_rows:
            break
        ok = ~done
        op, lw, cl = o[r, cols], lo[r, cols], c[r, cols]
        has = np.isfinite(cl) & np.isfinite(lw)
        if k > 0:
            gap = ok & has & np.isfinite(op) & (op <= stop_px)
            out[gap] = op[gap] / entry[gap] - 1.0
            done |= gap
            ok = ~done
        hit = ok & has & (lw <= stop_px)
        out[hit] = -stop
        done |= hit
        ok = ~done
        last_close = np.where(ok & has, cl, last_close)
        if k == hold - 1:
            fin = ok & np.isfinite(last_close)
            out[fin] = last_close[fin] / entry[fin] - 1.0
            done |= fin
    # 보유 끝까지 못 간 생존 행(데이터 끝·폐지): 마지막 유효 종가로 청산
    rest = ~done & alive & np.isfinite(last_close)
    out[rest] = last_close[rest] / entry[rest] - 1.0
    return out


def simulate_events(p: Panels, ev: pd.DataFrame, *, hold: int, stop: float, min_gap: int) -> pd.DataFrame:
    """통과한 사건 → 트레이드. 같은 종목은 직전 진입 후 ``min_gap`` 거래일 안에 재진입하지 않는다."""
    o, h, lo, c = (x.to_numpy() for x in (p.open, p.high, p.low, p.close))
    rows = []
    last_entry: dict[str, int] = {}
    for r in ev[ev["passes"]].sort_values(["day_idx", "code"]).itertuples():
        entry_row = r.day_idx + 1
        if entry_row >= len(p.days):
            continue
        prev = last_entry.get(r.code)
        if prev is not None and entry_row - prev < min_gap:
            continue
        ret = simulate_block(o, h, lo, c, entry_row, np.array([r.col_idx]), hold=hold, stop=stop)[0]
        if not np.isfinite(ret):
            continue
        last_entry[r.code] = entry_row
        rows.append({"code": r.code, "event_day": r.event_day, "entry_day": p.days[entry_row],
                     "family": r.family, "market": r.market, "ret": ret, "ret1": r.ret1,
                     "ret5": r.ret5, "adv20": r.adv20, "flow_ratio": r.flow_ratio})
    return pd.DataFrame(rows)


def control_pool(p: Panels, event_days: "pd.DatetimeIndex", cfg: dict, *, hold: int, stop: float) -> pd.DataFrame:
    """정합 대조 풀 — 사건이 있던 날마다 **같은 유동성 조건을 통과한 전 종목**을 같은 규칙으로 시뮬레이션."""
    f = feature_frame(p, ma_window=cfg["ma_window"])
    o, h, lo, c = (x.to_numpy() for x in (p.open, p.high, p.low, p.close))
    adv, close, ret5 = f["adv20"].to_numpy(), p.close.to_numpy(), f["ret5"].to_numpy()
    out = []
    for d in pd.DatetimeIndex(sorted(set(event_days))):
        di = p.days.get_loc(d)
        if di + 1 >= len(p.days):
            continue
        elig = np.where((adv[di] >= cfg["adv_floor"]) & (close[di] >= cfg["min_price"]))[0]
        if len(elig) == 0:
            continue
        ret = simulate_block(o, h, lo, c, di + 1, elig, hold=hold, stop=stop)
        ok = np.isfinite(ret)
        out.append(pd.DataFrame({
            "date": d, "code": p.close.columns[elig[ok]], "ret": ret[ok],
            "adv20": adv[di, elig[ok]], "ret5": ret5[di, elig[ok]],
        }))
    pool = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "code", "ret", "adv20", "ret5"])
    pool["market"] = p.market.reindex(pool["code"]).to_numpy()
    return pool


def add_strata(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """정합 층 — 같은 날 안에서 20일 거래대금 4분위 × 5일 수익률 4분위 (+ 시장).

    분위 경계는 **풀(그날 전 종목)** 에서 정해야 트레이드와 대조가 같은 자로 잘린다 —
    :func:`strata_from_pool` 이 그걸 한다. 이 함수는 풀 자신에 층을 붙인다.
    """
    df = df.copy()
    g = df.groupby(date_col)
    df["q_adv"] = g["adv20"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    df["q_ret5"] = g["ret5"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if s.notna().sum() >= 4 else np.nan)
    return df


def strata_from_pool(trades: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """트레이드에 풀과 같은 층을 붙인다 — 같은 (날짜, 종목) 행의 층을 그대로 가져온다."""
    key = pool.set_index(["date", "code"])[["q_adv", "q_ret5"]]
    t = trades.copy()
    idx = pd.MultiIndex.from_arrays([t["event_day"], t["code"]])
    got = key.reindex(idx)
    t["q_adv"] = got["q_adv"].to_numpy()
    t["q_ret5"] = got["q_ret5"].to_numpy()
    t["date"] = t["event_day"]
    return t
