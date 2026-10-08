"""스윙 3호 SEPA 합류 — 뼈대 S 사건표 + 한국식 진입/청산 시뮬레이터 (설계: docs/superpowers/specs/2026-10-09-sepa-confluence.md).

판정 정본은 ``research/logs/sepa_confluence/VERDICT.md`` (사전등록). 이 파일은 1단계(피처·시점 안전·커버리지)의
뼈대 몫이다. 하는 일은 세 가지다.

1. 패널 적재 — 가격(정문 :func:`swing_it.storage.read_prices`, 폐지 포함·분할조정)·시장 구분
2. 사건 표 — 뼈대 S = 유니버스 ∧ 추세 템플릿 ∧ RS≥70 ∧ 피벗 돌파 ∧ 그 시장 breadth > 50% (전부 ``d`` 종가까지)
3. 트레이드 시뮬레이션 — ``d+1`` 시가 진입(추격 5%·상한가 시가는 버림), 하드손절 −8%, 종가 < MA50 → 다음날 시가,
   최대 60거래일. 같은 종목은 보유 중 재진입 없음.

:func:`control_trades` 는 같은 체결 규칙을 임의 (종목, 날짜) 에 적용한다 — 정합 대조가 트레이드와 **같은 체결 규칙**을
쓰는 것이 대조의 전제다(:mod:`catalyst_flow` 와 같은 원칙).

**유니버스(한국 위생)는 여기서 만들지 않는다.** 병렬로 만드는 ``swing_it.features.korea`` (``universe_mask`` 예정)의
bool 패널을 ``universe=`` 로 받는다. ``None`` 이면 전 종목 True — 커버리지·테스트용이고, 판정 러너는 반드시 넘긴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from swing_it.features.korea import LIMIT_UP_REL_TOL
from swing_it.features.minervini import breadth, pivot_breakout, pivot_level, rs_rating, trend_template
from swing_it.features.technical import moving_average
from swing_it.storage import read_prices

# --- 뼈대 S·체결 규칙(원전 수치, 탐색하지 않는다) ---
PREREG_SKELETON = {
    "rs_min": 70.0,          # RS 등급 하한(횡단면 백분위)
    "breadth_min": 0.5,      # 그 종목 시장의 MA50 위 종목 비율 > 50%
    "pivot_window": 50,      # 피벗 = 직전 50일 최고 고가
    "vol_mult": 1.5,         # 돌파일 거래량 ≥ 1.5 × 직전 50일 평균
    "stop": 0.08,            # 하드손절(장중 저가 터치 → 손절가, 갭이면 시가)
    "max_hold": 60,          # 최대 보유 거래일(진입일 = 1일째, 60일째 종가 청산)
    "chase": 0.05,           # 진입 시가 > 피벗 × 1.05 면 추격 금지
    "limit_up_skip": True,   # 상한가 시가는 체결 불가로 버림
    "event_lo": None,        # 사건일 하한(포함). None = 패널 처음
    "event_hi": None,        # 사건일 상한(포함). None = 패널 끝
}

#: 상한가 시가 판정 비율 — 정의는 :func:`swing_it.features.korea.is_limit_up_open` 하나다(``1.30 − LIMIT_UP_REL_TOL``).
#: ``limit_up_price`` 를 직접 못 쓰는 이유: 분할조정 패널의 과거 가격은 호가 격자 위에 있지 않아 틱 내림이 틀린 값을
#: 낸다. 비율은 조정 후에도 보존된다. 그 시가는 이미 추격 금지(피벗×1.05) 한참 위라 진입 규칙상 결과는 같다.
LIMIT_UP_RATIO = 1.30 - LIMIT_UP_REL_TOL

TRADE_COLS = ["code", "event_day", "entry_day", "exit_day", "entry_px", "exit_px", "ret",
              "exit_reason", "hold_days", "market", "pivot"]
SKIP_COLS = ["code", "event_day", "skip_reason", "market", "pivot"]


@dataclass
class SepaPanels:
    """날짜×종목 가격 패널 + 시장 구분. :class:`catalyst_flow.Panels` 와 같은 속성 이름(덕 타이핑 호환)."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    trade_value: pd.DataFrame
    market: pd.Series          # code → "KOSPI"/"KOSDAQ"/"X"(코넥스·미상)

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.close.index


def _market_of(con, codes: pd.Index) -> pd.Series:
    """code → KOSPI/KOSDAQ/X. 폐지 마스터를 먼저, 현재 마스터를 뒤에 둬서 재상장은 현재 값이 이긴다."""
    mk = pd.read_sql_query("SELECT code, market FROM stocks", con)
    dk = pd.read_sql_query("SELECT code, market FROM delisted_stocks", con)
    m = pd.concat([dk, mk]).drop_duplicates("code", keep="last").set_index("code")["market"]
    m = m.map(lambda x: "KOSPI" if x in ("거래소", "KOSPI", "유가증권") else ("KOSDAQ" if x in ("코스닥", "KOSDAQ") else "X"))
    return m.reindex(codes).fillna("X")


def load_panels(con, *, start: "str | None" = None) -> SepaPanels:
    """가격 패널(정문 — 폐지 포함 검사). ``start`` 는 **창 워밍업을 포함한** 자르기 시점이다 —
    뼈대 S 는 252일 이력이 필요하므로 판정 구간 시작보다 1년 이상 앞이어야 한다."""
    prices = read_prices(con)
    prices["date"] = pd.to_datetime(prices["date"])
    if start is not None:
        prices = prices[prices["date"] >= pd.Timestamp(start)]
    piv = {c: prices.pivot(index="date", columns="code", values=c).sort_index().astype(float)
           for c in ("open", "high", "low", "close", "volume", "trade_value")}
    return SepaPanels(market=_market_of(con, piv["close"].columns), **piv)


# --- 사건 표 ---

def skeleton_features(p, cfg: "dict | None" = None) -> dict[str, pd.DataFrame]:
    """뼈대 S 의 재료 패널(전부 날짜×종목, ``d`` 종가까지). ``p`` 는 :class:`SepaPanels` 모양이면 된다."""
    cfg = {**PREREG_SKELETON, **(cfg or {})}
    brk, pivot = pivot_breakout(p.high, p.close, p.volume, window=cfg["pivot_window"], vol_mult=cfg["vol_mult"])
    return {
        "ma50": moving_average(p.close, 50),
        "tt": trend_template(p.close, high=p.high, low=p.low),
        "rs": rs_rating(p.close),
        "breakout": brk,
        "pivot": pivot,
        "breadth": breadth(p.close, p.market),
    }


def _universe_panel(universe: "pd.DataFrame | None", like: pd.DataFrame) -> pd.DataFrame:
    """유니버스 bool 패널을 가격 패널 모양으로. ``None`` = 전부 True. 패널 밖(날짜·종목)은 False."""
    if universe is None:
        return pd.DataFrame(True, index=like.index, columns=like.columns)
    u = universe.reindex(index=like.index, columns=like.columns)
    return u.where(u.notna(), False).astype(bool)


def build_skeleton_events(p, cfg: "dict | None" = None, *, universe: "pd.DataFrame | None" = None,
                          feats: "dict[str, pd.DataFrame] | None" = None, keep_all: bool = False) -> pd.DataFrame:
    """뼈대 S 사건 표 — 한 행 = (종목, 신호일 ``d``). 진입은 시뮬레이터가 ``d+1`` 시가로 한다.

    S = 유니버스 ∧ 추세 템플릿 ∧ RS ≥ 70 ∧ 피벗 돌파(거래량 1.5배) ∧ 그 시장 breadth > 0.5.

    Args:
        p: :class:`SepaPanels` 모양(open/high/low/close/volume/market).
        cfg: :data:`PREREG_SKELETON` 위에 덮을 값(민감도 기술용). 기본 = 원전 수치.
        universe: 유니버스 bool 패널(날짜×종목). ``None`` = 전부 True — **판정 러너는 반드시 넘긴다**.
        feats: :func:`skeleton_features` 결과를 재사용할 때.
        keep_all: True 면 피벗 돌파 행 전부를 남기고 ``skeleton`` 열로 S 여부만 표시한다
            (catalyst_flow 의 ``passes`` 와 같은 쓰임 — 어느 조건이 얼마나 거르는지 세기용).

    ``cfg["require_breakout"]`` (기본 True — :data:`PREREG_SKELETON` 에 **넣지 않는다**: 넣으면 스윙 3호 판정
    config 의 원장 지문이 바뀐다) 가 False 면 피벗 돌파·거래량 조건을 빼고 **추세 템플릿 행**을 바탕으로 쓴다 —
    S' = 유니버스 ∧ 추세 템플릿 ∧ RS ∧ breadth (탐색 격자 ``sepa_explore`` 의 진입 "trend"). ``pivot`` 열은 그대로
    직전 50일 최고 고가라 시뮬레이터의 추격 금지(시가 > 피벗 × 1.05)도 같은 식으로 걸린다. ``keep_all`` 이면
    추세 템플릿 행 전부가 남는다.

    Returns:
        ``code, event_day, market, pivot, close, tt, rs, breadth, universe, skeleton, day_idx, col_idx``.
    """
    cfg = {**PREREG_SKELETON, **(cfg or {})}
    f = skeleton_features(p, cfg) if feats is None else feats
    uni = _universe_panel(universe, p.close)
    base = f["breakout"] if cfg.get("require_breakout", True) else f["tt"]
    di, ci = np.nonzero(base.to_numpy(dtype=bool))
    ev = pd.DataFrame({"code": p.close.columns[ci], "event_day": p.days[di], "day_idx": di, "col_idx": ci})

    def pick(panel: pd.DataFrame) -> np.ndarray:
        return panel.to_numpy()[di, ci]

    ev["market"] = p.market.reindex(ev["code"]).to_numpy()
    ev["pivot"] = pick(f["pivot"])
    ev["close"] = pick(p.close)
    ev["tt"] = pick(f["tt"]).astype(bool)
    ev["rs"] = pick(f["rs"])
    ev["breadth"] = pick(f["breadth"])
    ev["universe"] = pick(uni).astype(bool)
    ev["skeleton"] = (ev["universe"] & ev["tt"] & (ev["rs"] >= cfg["rs_min"]).fillna(False)
                      & (ev["breadth"] > cfg["breadth_min"]).fillna(False))
    if cfg["event_lo"] is not None:
        ev = ev[ev["event_day"] >= pd.Timestamp(cfg["event_lo"])]
    if cfg["event_hi"] is not None:
        ev = ev[ev["event_day"] <= pd.Timestamp(cfg["event_hi"])]
    if not keep_all:
        ev = ev[ev["skeleton"]]
    return ev.sort_values(["event_day", "code"]).reset_index(drop=True)


# --- 시뮬레이터 ---

class _Book:
    """종목×날짜(전치, 행 연속) 배열 묶음. 한 종목의 시계열 슬라이스가 연속 메모리라 사건 루프가 싸다.

    ``tradable`` = 시세가 있고(OHLC 유한) 거래량 > 0 인 날. 정지일(거래량 0, OHLC 이월)은 진입도 청산도
    못 한다. ``present`` = 종가가 있는 날(정지일 포함) — 데이터 끝이 "폐지" 인지 "패널 끝" 인지 가른다.
    """

    def __init__(self, o, h, lo, c, ma50, volume):
        self.days = c.index
        self.codes = c.columns

        def arr(x: pd.DataFrame) -> np.ndarray:
            return np.ascontiguousarray(x.reindex(index=c.index, columns=c.columns).to_numpy(float).T)

        self.O, self.H, self.L, self.C, self.MA = arr(o), arr(h), arr(lo), arr(c), arr(ma50)
        self.present = np.isfinite(self.C)
        tr = self.present & np.isfinite(self.O) & np.isfinite(self.L) & (self.O > 0)
        if volume is not None:
            tr &= arr(volume) > 0
        self.tradable = tr


def _simulate_one(b: _Book, ci: int, d: int, pivot: float, *, stop: float, max_hold: int, chase: float,
                  limit_up_skip: bool) -> "tuple[str, int, float, float]":
    """한 사건(종목 ``ci``, 신호일 행 ``d``) → ``(사유, 청산 행, 진입가, 청산가)``. 건너뛰면 ``("skip_…", -1, nan, nan)``.

    규칙(진입일 = 1일째, 오프셋 0):
    - 진입: ``d+1`` 시가. 그날 정지·시세 없음이면 버림(다음 날로 미루지 않는다 — 살 수 없던 자리).
      시가 > 피벗 × (1+chase) 면 추격 금지, 시가 ≥ 전일종가 × :data:`LIMIT_UP_RATIO` 면 상한가로 버림.
    - 손절: 거래 가능일에 저가 ≤ 손절가 → ``min(시가, 손절가)`` (갭 하락이면 시가). 진입일엔 시가 > 손절가라 손절가.
    - 추세: 거래 가능일 ``k`` (``k ≤ max_hold−1`` 일째) 종가 < MA50 → **다음 거래 가능일 시가** 청산. 정지일이 끼면
      재개일 시가로 밀린다. 같은 날 장중 손절이 먼저 났으면 손절.
    - 시간: ``max_hold`` 일째 종가. 그날이 정지면 다음 거래 가능일 종가로 밀린다.
    - 그 전에 시세가 끊기면(폐지·패널 끝) 마지막 거래 가능일 종가로 청산 — 사유 ``delisted``(패널 마지막 날에도
      종가가 없다) / ``data_end`` (패널이 먼저 끝났다). 폐지 손실을 지우지 않는다.
    """
    n = len(b.days)
    e = d + 1
    if e >= n:
        return "skip_no_next_day", -1, np.nan, np.nan
    tr = b.tradable[ci, e:]
    if not tr[0]:
        return "skip_no_open", -1, np.nan, np.nan
    op, lw, cl, ma = b.O[ci, e:], b.L[ci, e:], b.C[ci, e:], b.MA[ci, e:]
    entry = op[0]
    if chase is not None and np.isfinite(pivot) and entry > pivot * (1.0 + chase):
        return "skip_chase", -1, np.nan, np.nan
    prev = b.C[ci, d]
    if limit_up_skip and np.isfinite(prev) and prev > 0 and entry >= prev * LIMIT_UP_RATIO:
        return "skip_limit_up", -1, np.nan, np.nan
    stop_px = entry * (1.0 - stop)

    # 시간 청산 행: max_hold 일째(오프셋 max_hold−1) 이후 첫 거래 가능일.
    k_last = max_hold - 1
    j_time = None
    if k_last < len(tr):
        rest = tr[k_last:]
        a = int(rest.argmax())
        if rest[a]:
            j_time = k_last + a
    horizon = j_time if j_time is not None else len(tr) - 1
    w = slice(0, horizon + 1)
    w_tr = tr[w]

    hit = w_tr & ((lw[w] <= stop_px) | (op[w] <= stop_px))
    j_stop = int(hit.argmax()) if hit.any() else None

    j_trend = None
    sig_n = min(k_last, horizon + 1)        # 신호는 1 ~ max_hold−1 일째 종가에서만(max_hold 일째는 시간 청산)
    with np.errstate(invalid="ignore"):
        sig = w_tr[:sig_n] & (cl[:sig_n] < ma[:sig_n])
    if sig.any():
        js = int(sig.argmax())
        after = w_tr[js + 1:]
        if after.any():
            j_trend = js + 1 + int(after.argmax())

    if j_stop is not None and (j_trend is None or j_stop < j_trend):
        return "stop", e + j_stop, entry, min(op[j_stop], stop_px)
    if j_trend is not None:
        return "ma50", e + j_trend, entry, op[j_trend]
    if j_time is not None:
        return "time", e + j_time, entry, cl[j_time]
    last = int(np.flatnonzero(tr)[-1])
    reason = "data_end" if b.present[ci, n - 1] else "delisted"
    return reason, e + last, entry, cl[last]


def _run(b: _Book, ev: pd.DataFrame, *, stop: float, max_hold: int, chase: "float | None",
         limit_up_skip: bool, no_overlap: bool) -> "tuple[pd.DataFrame, pd.DataFrame]":
    col_of = pd.Series(np.arange(len(b.codes)), index=b.codes)
    ev = ev[ev["code"].isin(b.codes) & ev["event_day"].isin(b.days)].copy()
    ev["sim_d"] = b.days.get_indexer(ev["event_day"])
    ev["sim_c"] = col_of.reindex(ev["code"]).to_numpy()
    ev = ev.drop_duplicates(["code", "event_day"]).sort_values(["sim_d", "code"])
    busy_until: dict[int, int] = {}          # 종목 → 보유 중인 트레이드의 청산 행
    trades, skips = [], []
    extra = [x for x in ev.columns if x not in TRADE_COLS and x not in ("sim_d", "sim_c", "day_idx", "col_idx")]
    for row in ev.to_dict("records"):
        ci, d = int(row["sim_c"]), int(row["sim_d"])
        base = {"code": row["code"], "event_day": row["event_day"], "market": row.get("market", np.nan),
                "pivot": row.get("pivot", np.nan)}
        # 보유 중(신호일 종가에 포지션이 열려 있음) 이면 버린다. 청산 행 당일 신호는 이미 청산된 뒤라 허용.
        if no_overlap and d < busy_until.get(ci, -1):
            skips.append({**base, "skip_reason": "skip_overlap"})
            continue
        reason, x, ent, ext = _simulate_one(b, ci, d, float(base["pivot"]), stop=stop, max_hold=max_hold,
                                            chase=chase, limit_up_skip=limit_up_skip)
        if reason.startswith("skip_"):
            skips.append({**base, "skip_reason": reason})
            continue
        busy_until[ci] = x
        trades.append({**base, "entry_day": b.days[d + 1], "exit_day": b.days[x], "entry_px": ent,
                       "exit_px": ext, "ret": ext / ent - 1.0, "exit_reason": reason, "hold_days": x - d,
                       **{k: row[k] for k in extra}})
    t = pd.DataFrame(trades, columns=TRADE_COLS + extra) if trades else pd.DataFrame(columns=TRADE_COLS + extra)
    s = pd.DataFrame(skips, columns=SKIP_COLS) if skips else pd.DataFrame(columns=SKIP_COLS)
    return t, s


def simulate_sepa(o: pd.DataFrame, h: pd.DataFrame, lo: pd.DataFrame, c: pd.DataFrame, ma50: pd.DataFrame,
                  events: pd.DataFrame, *, volume: "pd.DataFrame | None" = None, stop: float = 0.08,
                  max_hold: int = 60, chase: "float | None" = 0.05, limit_up_skip: bool = True,
                  no_overlap: bool = True, pivot_window: int = 50, return_skipped: bool = False):
    """뼈대 사건 → 트레이드(총수익, 비용 전). 규칙은 :func:`_simulate_one` 참조.

    Args:
        o, h, lo, c: 날짜×종목 분할조정 OHLC. ``h`` 는 ``events`` 에 ``pivot`` 열이 없을 때 피벗
            (직전 ``pivot_window`` 일 최고 고가)을 계산하는 데만 쓴다.
        ma50: 추세 청산용 MA50 패널(``d`` 종가까지 — :func:`skeleton_features` 의 ``ma50``).
        events: ``code, event_day`` (+ ``pivot``, ``market``, 그 밖의 열은 트레이드 표로 그대로 따라간다).
        volume: 거래량 패널. 주면 거래량 0 인 날(정지)을 체결 불가로 본다 — **실데이터에선 반드시 넘긴다**
            (정지일은 OHLC 가 이월돼 있어 그대로 두면 정지 중 체결이 일어난다).
        no_overlap: 같은 종목 보유 중(신호일 종가 기준) 사건을 버린다. 대조에선 끈다(:func:`control_trades`).
        return_skipped: True 면 ``(trades, skipped)`` — 버린 사건과 사유(추격·상한가·정지·보유 중·다음 날 없음).

    Returns:
        ``code, event_day, entry_day, exit_day, entry_px, exit_px, ret, exit_reason, hold_days, market, pivot`` (+ 사건 열).
        ``exit_reason`` ∈ {stop, ma50, time, delisted, data_end}. ``hold_days`` 는 진입일을 1 로 센 거래일 수.
    """
    ev = events.copy()
    if "pivot" not in ev.columns:
        pv = pivot_level(h.reindex_like(c), pivot_window)
        di = c.index.get_indexer(ev["event_day"])
        ci = c.columns.get_indexer(ev["code"])
        ok = (di >= 0) & (ci >= 0)
        ev["pivot"] = np.where(ok, pv.to_numpy()[np.where(ok, di, 0), np.where(ok, ci, 0)], np.nan)
    b = _Book(o, h, lo, c, ma50, volume)
    t, s = _run(b, ev, stop=stop, max_hold=max_hold, chase=chase, limit_up_skip=limit_up_skip, no_overlap=no_overlap)
    return (t, s) if return_skipped else t


def control_trades(o: pd.DataFrame, h: pd.DataFrame, lo: pd.DataFrame, c: pd.DataFrame, ma50: pd.DataFrame,
                   pairs: pd.DataFrame, *, volume: "pd.DataFrame | None" = None,
                   market: "pd.Series | None" = None, pivot_window: int = 50, no_overlap: bool = False,
                   **rules):
    """임의 (종목, 신호일) 쌍에 **같은 진입/청산 규칙**을 적용 — 정합 대조용.

    대조 종목은 돌파를 안 했으니 "피벗" 이 없다. 그래서 피벗을 사건과 같은 식(직전 50일 최고 고가)으로
    정의한다 — 추격 금지 규칙이 "직전 50일 고점 대비 5% 넘게 뜬 시가" 를 똑같이 거른다. 돌파 안 한 종목은
    대개 시가가 피벗 아래라 이 규칙에 거의 안 걸린다(대조를 덜 거르는 쪽 — 사건 쪽에만 걸리는 체결 제약이
    대조엔 없다는 비대칭은 판정 문서에 기술한다).

    ``pairs`` 에 ``pivot`` 이 있으면 그 값을 쓴다. 대조끼리는 독립 표본이므로 기본 ``no_overlap=False``.
    ``rules`` 는 :func:`simulate_sepa` 의 ``stop/max_hold/chase/limit_up_skip/return_skipped``.
    """
    ev = pairs.copy()
    if market is not None and "market" not in ev.columns:
        ev["market"] = market.reindex(ev["code"]).to_numpy()
    return simulate_sepa(o, h, lo, c, ma50, ev, volume=volume, no_overlap=no_overlap,
                         pivot_window=pivot_window, **rules)


# --- 커버리지(건수만 — 수익은 보지 않는다) ---

def skeleton_coverage(p, feats: dict[str, pd.DataFrame], *, universe: "pd.DataFrame | None" = None,
                      cfg: "dict | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """연도 × 시장 건수 표와 breadth>임계 날 비율 표. **결과(수익)는 계산하지 않는다** — 사전등록 전 단계.

    건수 칸(전부 종목-일): ``tt`` · ``tt_rs`` (∧ RS≥70) · ``breakout`` (피벗 돌파 전체) ·
    ``tt_rs_brk`` (∧ 돌파) · ``S`` (∧ breadth > 0.5 ∧ 유니버스).
    """
    cfg = {**PREREG_SKELETON, **(cfg or {})}
    uni = _universe_panel(universe, p.close)
    tt = feats["tt"]
    rs_ok = (feats["rs"] >= cfg["rs_min"])
    brk = feats["breakout"]
    br_ok = (feats["breadth"] > cfg["breadth_min"])
    layers = {"tt": tt, "tt_rs": tt & rs_ok, "breakout": brk, "tt_rs_brk": tt & rs_ok & brk,
              "S": tt & rs_ok & brk & br_ok & uni}
    year = p.days.year
    mk = p.market.reindex(p.close.columns).to_numpy()
    rows = {}
    for name, panel in layers.items():
        by_year = panel.groupby(year).sum()                  # 연도 × 종목
        rows[name] = by_year.T.groupby(mk).sum().T.stack()   # (연도, 시장)
    counts = pd.DataFrame(rows)
    counts.index.names = ["year", "market"]
    mb = feats["breadth"].T.groupby(mk).first().T             # 날짜 × 시장(종목 자리 값은 시장별로 같다)
    frac = (mb > cfg["breadth_min"]).astype(float).where(mb.notna()).groupby(year).mean()
    return counts, frac
