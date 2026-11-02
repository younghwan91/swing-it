"""smallcap 공용 부품 — 패널·유니버스·사건·체결·비용·벤치·널. PREREG.md 가 정본이고 숫자는 여기 상수로 고정한다.

새 코드라 기존 엔진과의 패리티가 없다. ``tests/test_smallcap_lib.py`` 의 손계산 벡터와 ``smallcap_recheck.py`` 의 거래 재계산이 검증이다.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from krx_quant_core.costs.tax import sell_tax_rate

PIPELINE_LIBRARY = "smallcap 러너의 공용 함수(진입점 없음)"

V2 = "daily_bars_adj_v2"
DATA_START, SIG_START, DATA_END = "2008-01-01", "2009-01-02", "2016-08-31"
BAND_U = (300.0, 2000.0)        # ADV20 3억~20억(백만원)
BAND_U2 = (2000.0, 10000.0)     # ADV20 20억~100억(지도 대조 대역)
ORDER_WON = 1.5e7
AUM_WON = 3e8
SLOTS = 20
PARTICIPATION = 0.05
COMMISSION_BP = 1.5
FALLBACK_HS = ((750.0, 50.0), (2000.0, 30.0), (float("inf"), 15.0))   # (ADV20 상한 백만원, 반스프레드 bp)
IMPACT_Y = 1.0                  # 자리표시값(core 설명과 같다). 보정 안 됨
ADV_FLOOR_M = 100.0             # 매도일 ADV 가 비면 1억으로 본다
VOL_FALLBACK = 0.05             # 매도일 σ 가 비면 5%
TV_WIN, LOW_WIN, WIN_MIN = 50, 50, 40
SIGMA_WIN = 63
TAIL_ROWS = 10
LIMIT_CHANGE = np.datetime64("2015-06-15")
MOM_STEP = 20
BAD_ADJ_RATIO = 0.02           # 정정 SC-2: 수정가/원가 < 2% 칸은 자료 오류(008080 수정가 1원)

# 이름 거르개. '리츠' 는 메리츠를 잡지 않게 앞 글자를 본다.
NAME_FILTER = re.compile(
    r"스팩|기업인수|(?<!메)리츠|REIT|SPAC|부동산투자|맥쿼리인프라|선박투자|바다로|하이골드|동북아\d|호텔투자"
    r"|KODEX|TIGER|KOSEF|KINDEX|KBSTAR|ARIRANG|TREX|HANARO|TIMEFOLIO|ETF|ETN|레버리지|인버스|투융자",
    re.I,
)
# KIND 사유: 비부실(사건·자발) 폐지. 그 밖은 부실로 분류한다(라벨 −100% 에만 쓴다).
NONDISTRESS = re.compile(r"합병|이전상장|유가증권시장 상장|코스닥시장 상장|완전자회사|지주회사|주식교환|주식이전|신청에 의한|존속기간|해산|SPAC|스팩|상장예비심사")


def half_of(code: str) -> int:
    """종목 절반 — 코드 sha256 첫 바이트 짝홀. 0 = 탐색, 1 = 잠금(fcs ``research/autoresearch/prepare.py`` 와 같은 무솔트)."""
    return hashlib.sha256(str(code).encode()).digest()[0] & 1


def name_excluded(name: "str | None") -> bool:
    return bool(name) and bool(NAME_FILTER.search(name))


def is_common(code: str) -> bool:
    return len(code) == 6 and code.isdigit() and code.endswith("0")


def market_group(label: "str | None") -> str:
    """현재 스냅샷 시장 라벨 → KOSPI/KOSDAQ. 모르는 라벨은 거부한다(모름·코넥스 NULL 은 KOSDAQ)."""
    if label is None or label != label:     # NULL·NaN
        return "KOSDAQ"
    if label in ("거래소", "유가증권", "KOSPI"):
        return "KOSPI"
    if label in ("코스닥", "코넥스", "KOSDAQ"):
        return "KOSDAQ"
    raise ValueError(f"모르는 시장 라벨: {label!r}")


def tick_size(raw: float, market: str) -> float:
    """2023-01-25 개편 전 호가표(표본은 전부 그 전). 코스닥 5만원 이상은 100원."""
    for hi, t in ((1000, 1), (5000, 5), (10000, 10), (50000, 50)):
        if raw < hi:
            return float(t)
    if market == "KOSDAQ":
        return 100.0
    for hi, t in ((100000, 100), (500000, 500)):
        if raw < hi:
            return float(t)
    return 1000.0


def half_spread_bp(adv_m: float, raw: float, market: str) -> float:
    """반스프레드 = max(ADV 구간 폴백, 호가 하한). 원 가격이 없으면 호가 하한 25bp."""
    fb = next(hs for hi, hs in FALLBACK_HS if adv_m < hi)
    floor = tick_size(raw, market) / (2.0 * raw) * 1e4 if np.isfinite(raw) and raw > 0 else 25.0
    return max(fb, floor)


def impact_bp(order_won: float, adv_m: float, sigma: float) -> float:
    """제곱근 법칙 y·σ·√(Q/ADV)·1e4 — core ``sqrt_impact_bp`` 와 같은 식. 기본 핀 0.7.0 에 없어 옮겼다."""
    return IMPACT_Y * sigma * np.sqrt(order_won / (adv_m * 1e6)) * 1e4


def cost_one_way(side: str, day: date, market: str, adv_m: float, sigma: float, raw: float,
                 order_won: float = ORDER_WON) -> float:
    """편도 1배 비용(분수). 수수료 + 매도세(매도만) + 반스프레드 + 충격."""
    tax = sell_tax_rate(day, market) * 1e4 if side == "SELL" else 0.0
    return (COMMISSION_BP + tax + half_spread_bp(adv_m, raw, market) + impact_bp(order_won, adv_m, sigma)) * 1e-4


def _roll(a: np.ndarray, win: int, min_obs: int, how: str) -> np.ndarray:
    """t 제외 직전 win 행 집계(행 = 날짜, 열 = 종목 전치 없이 축 0 = 종목)."""
    r = pd.DataFrame(a.T).shift(1).rolling(win, min_periods=min_obs)
    return getattr(r, how)().to_numpy().T


@dataclass
class Panel:
    dates: np.ndarray
    codes: list
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    TV: np.ndarray                 # 원 종가 × 원 거래량(백만원). 없으면 NaN → U 밖
    RAW: np.ndarray
    market: np.ndarray
    excluded: np.ndarray
    distress: np.ndarray
    r: np.ndarray = field(init=False)
    Cb: np.ndarray = field(init=False)
    Cp: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        N, T = self.C.shape
        self.N, self.T = N, T
        self.day_py = [d.astype("datetime64[D]").item() for d in self.dates]
        Cff = pd.DataFrame(self.C.T).ffill().to_numpy().T
        self.Cff = Cff
        r = np.full((N, T), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[:, 1:] = np.where(np.isfinite(self.C[:, 1:]), Cff[:, 1:] / Cff[:, :-1] - 1.0, np.nan)
        self.r = r
        traded = np.nan_to_num(self.V, nan=0.0) > 0
        self.traded = traded
        self.last = np.array([np.flatnonzero(traded[i]).max() if traded[i].any() else -1 for i in range(N)])
        self.first = np.array([np.flatnonzero(traded[i]).min() if traded[i].any() else T for i in range(N)])
        tail = np.zeros((N, T), dtype=bool)
        for i in range(N):
            if 0 <= self.last[i] < T - 1:
                tail[i, np.flatnonzero(traded[i])[-TAIL_ROWS:]] = True
        self.tail = tail
        self.is_new_limit = self.dates >= LIMIT_CHANGE
        lim = np.where(self.is_new_limit, 0.30, 0.15)
        self.thr = np.where(self.is_new_limit, 0.29, 0.14)
        reopen = np.zeros((N, T), dtype=bool)
        reopen[:, 1:] = traded[:, 1:] & ~traded[:, :-1]
        for i in range(N):
            if self.first[i] < T:
                reopen[i, self.first[i]] = False
        with np.errstate(invalid="ignore"):
            over = traded & (np.abs(r) > lim + 0.005)
        self.seam = (reopen | over) & ~tail
        with np.errstate(invalid="ignore"):
            tail_up = tail & traded & (r > lim + 0.005)      # 정정 SC-2: 꼬리의 제한폭 초과 상승도 이익 0
        clip = self.seam | tail_up
        r0 = np.nan_to_num(r, nan=0.0)
        rb = np.where(clip, 0.0, r0)
        rp = np.where(clip, np.minimum(r0, 0.0), r0)
        rows = np.isfinite(Cff)
        self.Cb = np.where(rows, np.cumprod(1.0 + rb, axis=1), np.nan)
        self.Cp = np.where(rows, np.cumprod(1.0 + rp, axis=1), np.nan)
        # 수정 사건 칸 = 수정가/원가 비의 1% 초과 계단(상장·폐지 공통, factor 는 폐지에서 NULL)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = pd.DataFrame((self.C / np.where(self.RAW > 0, self.RAW, np.nan)).T).ffill().to_numpy().T
            step = np.zeros((N, T), dtype=bool)
            step[:, 1:] = np.abs(ratio[:, 1:] / ratio[:, :-1] - 1.0) > 0.01
        self.action = step
        with np.errstate(invalid="ignore", divide="ignore"):
            self.bad_adj = np.nan_to_num(self.C / np.where(self.RAW > 0, self.RAW, np.nan), nan=1.0) < BAD_ADJ_RATIO
        tv = np.where(traded, self.TV, np.nan)
        self.ADV = _roll(tv, 20, 15, "mean")
        self.TVMAX = _roll(tv, TV_WIN, WIN_MIN, "max")
        rs = np.where(traded, r, np.nan)
        self.SIG63 = _roll(rs, SIGMA_WIN, SIGMA_WIN, "std")
        self.SIG20 = _roll(rs, SIGMA_WIN, 20, "std")
        c_tr = np.where(traded, self.C, np.nan)
        self.LOW = _roll(c_tr, LOW_WIN, WIN_MIN, "min")
        with np.errstate(invalid="ignore"):
            self.X = traded & (self.C < self.LOW)
            ok = np.isfinite(self.H) & np.isfinite(self.L) & (self.H > 0) & (self.L > 0)
        self.TP = np.where(ok, (self.H + self.L + self.C) / 3.0, self.C)
        self.LOWPX = np.where(ok, self.L, self.C)
        self.exit_days = [np.flatnonzero(self.X[i]) for i in range(N)]
        self.sig_t0 = int(np.searchsorted(self.dates, np.datetime64(SIG_START)))
        self.cache: dict = {}

    def universe(self, band: tuple) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            u = self.traded & np.isfinite(self.TV) & (self.ADV >= band[0]) & (self.ADV < band[1])
        u[self.excluded, :] = False
        u &= ~self.bad_adj
        u[:, : self.sig_t0] = False
        return u


def hvnm_events(p: Panel, u: np.ndarray) -> pd.DataFrame:
    """거래대금 충격·가격 무변동 사건. rank = TV(t)/ADV20(t)."""
    prev = np.zeros_like(u)
    prev[:, 1:] = p.traded[:, :-1]
    with np.errstate(invalid="ignore"):
        m = u & (p.TV > p.TVMAX) & (np.abs(p.r) <= p.SIG63) & prev & ~p.action & ~p.seam
    ci, ti = np.nonzero(m)
    return pd.DataFrame({"c": ci, "t": ti, "rank": p.TV[ci, ti] / p.ADV[ci, ti]}).sort_values(["t", "c"]).reset_index(drop=True)


def mom_events(p: Panel, u: np.ndarray) -> pd.DataFrame:
    """12-1 모멘텀 — sig_t0 부터 20행마다 U(t) 상위 10%. rank = 신호값."""
    rows = []
    for t in range(p.sig_t0, p.T, MOM_STEP):
        idx = np.flatnonzero(u[:, t] & (p.first + 252 <= t))
        if len(idx) < 10:
            continue
        s = p.Cb[idx, t - 21] / p.Cb[idx, t - 252] - 1.0
        fin = np.isfinite(s)
        idx, s = idx[fin], s[fin]
        cut = np.quantile(s, 0.9)
        rows += [(int(i), t, float(v)) for i, v in zip(idx, s) if v >= cut]
    return pd.DataFrame(rows, columns=["c", "t", "rank"])


def outcome(p: Panel, c: int, t: int, *, close_fill: bool = False, order: float = ORDER_WON):
    """신호일 t 의 체결·청산 한 건. 미체결이면 (사유 문자열). 체결이면 dict. 종목·날짜로 캐시한다.

    진입 t+1 (H+L+C)/3 · 거래대금 5% 한도. 청산 = 종가 < 직전 50행 최저 종가인 다음 날부터 날마다 5% 분할.
    ``close_fill`` 은 종가 대용 라벨(체결 불가능, §5).
    """
    key = (c, t, close_fill)
    hit = p.cache.get(key)
    if hit is not None:
        return hit
    res = _outcome(p, c, t, close_fill, order)
    p.cache[key] = res
    return res


def _outcome(p: Panel, c: int, t: int, close_fill: bool, order: float):
    e = t + 1
    if e >= p.T or not p.traded[c, e]:
        return "halt"
    if np.isfinite(p.r[c, e]) and p.r[c, e] >= p.thr[e] - 1e-9:
        return "limit"
    tv_e = p.TV[c, e]
    if not np.isfinite(tv_e) or tv_e * 1e6 * PARTICIPATION < order:
        return "cap"
    market = p.market[c]
    adv_b = p.ADV[c, t] if np.isfinite(p.ADV[c, t]) else ADV_FLOOR_M
    sig_b = p.SIG20[c, t] if np.isfinite(p.SIG20[c, t]) else VOL_FALLBACK
    cb = cost_one_way("BUY", p.day_py[e], market, max(adv_b, ADV_FLOOR_M), min(sig_b, 1.0), p.RAW[c, e], order)
    px_e = p.C[c, e] if close_fill else p.TP[c, e]
    base = p.C[c, e] / px_e / p.Cp[c, e]

    days = p.exit_days[c]
    j = int(np.searchsorted(days, e + 1))
    s = int(days[j]) if j < len(days) else None
    last = int(p.last[c])
    frac, gross, exit_day, cs, reason, locked = 1.0, 0.0, None, None, None, False
    d = (s + 1) if s is not None else p.T
    while d <= last and d < p.T and frac > 1e-12:
        if not p.traded[c, d]:
            d += 1
            continue
        if not p.tail[c, d] and np.isfinite(p.r[c, d]) and p.r[c, d] <= -p.thr[d] + 1e-9:
            locked = True
            d += 1
            continue
        val = base * p.Cp[c, d]               # 그날 종가 기준 값 배수
        cap = PARTICIPATION * p.TV[c, d] * 1e6 if np.isfinite(p.TV[c, d]) else 0.0
        if cap <= 0:
            d += 1
            continue
        if cs is None:
            adv_s = p.ADV[c, d] if np.isfinite(p.ADV[c, d]) else ADV_FLOOR_M
            sig_s = p.SIG20[c, d] if np.isfinite(p.SIG20[c, d]) else VOL_FALLBACK
            cs = cost_one_way("SELL", p.day_py[d], market, max(adv_s, ADV_FLOOR_M), min(sig_s, 1.0), p.RAW[c, d], order)
        if close_fill:
            px_ratio = 1.0
        elif locked:
            px_ratio = p.LOWPX[c, d] / p.C[c, d]
        else:
            px_ratio = p.TP[c, d] / p.C[c, d]
        locked = False
        take = min(frac, cap / (order * max(val, 1e-9)))
        gross += take * val * px_ratio
        frac -= take
        exit_day = d
        reason = "signal" if s is not None else reason
        d += 1
    if frac > 1e-12:                           # 표본 끝 또는 시리즈 끝: 남은 몫은 마지막 종가
        end = min(last, p.T - 1)
        gross += frac * base * p.Cp[c, end]
        exit_day = end
        reason = "series_end" if last < p.T - 1 else "censored"
        if cs is None:
            adv_s = p.ADV[c, end] if np.isfinite(p.ADV[c, end]) else ADV_FLOOR_M
            sig_s = p.SIG20[c, end] if np.isfinite(p.SIG20[c, end]) else VOL_FALLBACK
            cs = cost_one_way("SELL", p.day_py[end], market, max(adv_s, ADV_FLOOR_M), min(sig_s, 1.0), p.RAW[c, end], order)
    return {"e": e, "x": int(exit_day), "gross": gross - 1.0, "cb": cb, "cs": cs, "reason": reason,
            "c": c, "t": t, "dist": bool(p.distress[c]) and reason == "series_end"}


def net(o: dict, m: float, *, dist100: bool = False) -> float:
    g = -1.0 if (dist100 and o["dist"]) else o["gross"]
    return (1.0 + g) * (1.0 - m * o["cs"]) / (1.0 + m * o["cb"]) - 1.0


class Bench:
    """같은 구간 U(t) 동일가중 매수보유. 폐지는 Cb 끝값으로 남고 현금처럼 평평하다(§4)."""

    def __init__(self, p: Panel, u: np.ndarray, *, dist100: bool = False) -> None:
        self.p, self.u, self.rows = p, u, {}
        self.Cb = p.Cb
        if dist100:
            Cb = p.Cb.copy()
            for i in np.flatnonzero(p.distress):
                if 0 <= p.last[i] < p.T - 1:
                    Cb[i, p.last[i]:] = 0.0
            self.Cb = Cb

    def row(self, t: int, e: int) -> np.ndarray:
        key = (t, e)
        got = self.rows.get(key)
        if got is None:
            idx = np.flatnonzero(self.u[:, t] & np.isfinite(self.Cb[:, e]) & (self.Cb[:, e] > 0))
            got = (self.Cb[idx] / self.Cb[idx, e][:, None]).mean(axis=0) - 1.0 if len(idx) else np.zeros(self.p.T)
            self.rows[key] = got
        return got

    def ret(self, t: int, e: int, x: int) -> float:
        return float(self.row(t, e)[x])


def event_study(p: Panel, ev: pd.DataFrame, bench: Bench, *, m: float = 2.0, close_fill: bool = False,
                dist100: bool = False) -> tuple:
    """슬롯 없는 사건 전수. 가상 보유 중인 종목의 재사건은 뺀다. (체결 표, 미체결 사유 수)."""
    held: dict = {}
    fails = {"halt": 0, "limit": 0, "cap": 0, "held": 0}
    rows = []
    for c, t in zip(ev.c.to_numpy(), ev.t.to_numpy()):
        c, t = int(c), int(t)
        if held.get(c, -1) >= t:
            fails["held"] += 1
            continue
        o = outcome(p, c, t, close_fill=close_fill)
        if isinstance(o, str):
            fails[o] += 1
            continue
        held[c] = o["x"]
        rows.append(_row(p, o, bench, m, dist100))
    return pd.DataFrame(rows), fails


def _row(p: Panel, o: dict, bench: Bench, m: float, dist100: bool) -> dict:
    n = net(o, m, dist100=dist100)
    b = bench.ret(o["t"], o["e"], o["x"])
    return {"c": o["c"], "t": o["t"], "e": o["e"], "x": o["x"], "gross": o["gross"], "net": n, "bench": b,
            "active": n - b, "gross_active": o["gross"] - b, "cb": o["cb"], "cs": o["cs"], "reason": o["reason"],
            "year": int(str(p.dates[o["e"]])[:4])}


def run_book(p: Panel, ev: pd.DataFrame, bench: Bench, *, m: float = 2.0, close_fill: bool = False,
             dist100: bool = False, slots: int = SLOTS) -> tuple:
    """고정 1,500만원·동시 20슬롯 북. 같은 날은 rank 큰 순. 현금 < 1,500만원이면 체결하지 않는다. 대기 없음."""
    ev = ev.sort_values(["t", "rank", "c"], ascending=[True, False, True])
    open_x: dict = {}                  # c → (x, pnl)
    realized, fails, rows = 0.0, {"halt": 0, "limit": 0, "cap": 0, "held": 0, "slots": 0, "cash": 0}, []
    for c, t in zip(ev.c.to_numpy(), ev.t.to_numpy()):
        c, t = int(c), int(t)
        e = t + 1
        for k in [k for k, v in open_x.items() if v[0] < e]:   # 그 전날까지 다 판 자리는 비었다
            realized += open_x.pop(k)[1]
        if c in open_x:
            fails["held"] += 1
            continue
        if len(open_x) >= slots:
            fails["slots"] += 1
            continue
        if AUM_WON + realized - len(open_x) * ORDER_WON < ORDER_WON:
            fails["cash"] += 1
            continue
        o = outcome(p, c, t, close_fill=close_fill)
        if isinstance(o, str):
            fails[o] += 1
            continue
        row = _row(p, o, bench, m, dist100)
        open_x[c] = (o["x"], ORDER_WON * row["net"])
        rows.append(row)
    return pd.DataFrame(rows), fails


def summarize(df: pd.DataFrame, *, years: float) -> dict:
    """기대값·손익비. 승률은 일부러 내지 않는다."""
    if df.empty:
        return {"n": 0}

    def pay(s: pd.Series) -> float:
        win, loss = s[s > 0], s[s < 0]
        return float(win.mean() / -loss.mean()) if len(win) and len(loss) else float("nan")

    def wins(s: pd.Series, q: float = 0.01) -> float:
        g = s.groupby(df["year"])
        return float(s.clip(g.transform(lambda x: x.quantile(q)), g.transform(lambda x: x.quantile(1 - q))).mean())

    cost = (df["cb"] + (1 + df["gross"]) * df["cs"]).sum() * 2.0 * ORDER_WON
    return {"n": len(df), "ev_gross": float(df.gross.mean()), "ev_gross_active": float(df.gross_active.mean()),
            "ev_net": float(df.net.mean()), "ev_active": float(df.active.mean()),
            "ev_active_wins": wins(df.active), "payoff_abs": pay(df.net), "payoff_active": pay(df.active),
            "hold_med": float((df.x - df.e).median()), "cost_rt_mean": float((df.cb + df.cs).mean()) * 2.0,
            "cost_drag_aum_yr": float(cost / AUM_WON / years)}


def fragility(df: pd.DataFrame) -> dict:
    """상위 5·20 체결, 최대 기여 1종목을 뺀 절대·능동 기대값(1위 종목은 합계 순손익 기준)."""
    srt = df.sort_values("net", ascending=False)
    top = df.groupby("c").net.sum().idxmax()
    out = {}
    for name, d in (("top5", srt.iloc[5:]), ("top20", srt.iloc[20:]), ("topname", df[df.c != top])):
        out[name] = {"ev_net": float(d.net.mean()), "ev_active": float(d.active.mean())}
    return out


def random_events(p: Panel, u: np.ndarray, ev: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """B5 — U(t) 안 무작위 칸, 같은 달 사건 수. rank 도 무작위."""
    months = p.dates.astype("datetime64[M]")
    want = pd.Series(months[ev.t.to_numpy()]).value_counts()
    rows = []
    for mo, n in want.items():
        days = np.flatnonzero(months == np.datetime64(mo, "M"))
        ci, ti = np.nonzero(u[:, days])
        pick = rng.integers(0, len(ci), size=min(int(n), len(ci)))
        rows.append(pd.DataFrame({"c": ci[pick], "t": days[ti[pick]]}))
    out = pd.concat(rows, ignore_index=True).drop_duplicates(["c", "t"])
    out["rank"] = rng.random(len(out))
    return out


def shifted_events(p: Panel, u: np.ndarray, ev: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """B6 — 종목별 사건 달력 원형 이동. 이동한 날이 U(t) 밖이면 버린다."""
    span = p.T - p.sig_t0
    rows = []
    for c, g in ev.groupby("c"):
        sh = int(rng.integers(1, span))
        t = p.sig_t0 + (g.t.to_numpy() - p.sig_t0 + sh) % span
        keep = u[int(c), t]
        rows.append(pd.DataFrame({"c": int(c), "t": t[keep], "rank": g["rank"].to_numpy()[keep]}))
    return pd.concat(rows, ignore_index=True)


def book_daily(p: Panel, fills: pd.DataFrame, m: float = 2.0) -> pd.DataFrame:
    """체결 표 → 일별 손익(원)·시장별 노출(원). 시가평가는 Cp, 매수·매도 비용은 그날 부과."""
    pnl = np.zeros(p.T)
    exp_kospi, exp_kosdaq = np.zeros(p.T), np.zeros(p.T)
    for f in fills.itertuples():
        c, e, x = f.c, f.e, f.x
        path = ORDER_WON * p.Cp[c, e:x + 1] / p.Cp[c, e]
        pnl[e + 1: x + 1] += np.diff(path)
        pnl[e] -= ORDER_WON * m * f.cb
        # 진입 슬리피지·청산 체결가 차이는 x 일에 맞춘다(총합 = 체결 순손익)
        total = ORDER_WON * f.net
        pnl[x] += total - (path[-1] - ORDER_WON - ORDER_WON * m * f.cb)
        exp = exp_kospi if p.market[c] == "KOSPI" else exp_kosdaq
        exp[e: x] += path[:-1]
    return pd.DataFrame({"pnl": pnl, "exp_kospi": exp_kospi, "exp_kosdaq": exp_kosdaq}, index=pd.DatetimeIndex(p.dates))


def ols_hac(y, X, lag: int = 3) -> dict:
    """OLS + Newey-West(Bartlett) — pipeline-modules ``diagnostics/hac.py`` 와 같은 식(그 브랜치는 미머지라 옮겼다)."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    Z = np.column_stack([np.ones(len(y)), X])
    n, k_ = Z.shape
    if n < k_ + lag + 2:
        return {"n": int(n), "alpha": float("nan"), "t_alpha": float("nan")}
    zi = np.linalg.pinv(Z.T @ Z)
    b = zi @ Z.T @ y
    e = y - Z @ b
    u = Z * e[:, None]
    S = u.T @ u
    for k in range(1, lag + 1):
        G = u[k:].T @ u[:-k]
        S += (1.0 - k / (lag + 1.0)) * (G + G.T)
    V = zi @ S @ zi
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return {"n": int(n), "alpha": float(b[0]), "t_alpha": float(b[0] / se[0]) if se[0] > 0 else float("nan"),
            "beta": [float(v) for v in b[1:]]}


def load_names(con) -> pd.DataFrame:
    """이름·시장·폐지 사유. 시장은 현재 스냅샷이다."""
    s = pd.read_sql_query("select code,name,market from stocks", con)
    d = pd.read_sql_query("select code,name,market from delisted_stocks", con)
    k = pd.read_sql_query("select code,name,reason from kind_delistings", con)
    nm = pd.concat([s, d, k[["code", "name"]]]).drop_duplicates("code")
    m = pd.concat([s, d]).dropna(subset=["market"]).drop_duplicates("code").set_index("code").market
    nm["market"] = nm.code.map(m)
    nm["group"] = nm.market.map(market_group)
    r = k.dropna(subset=["reason"]).drop_duplicates("code", keep="last").set_index("code").reason
    nm["reason"] = nm.code.map(r)
    return nm


def explore_codes(codes) -> list:
    return [c for c in codes if is_common(c) and half_of(c) == 0]


def wide(df: pd.DataFrame, col: str, dates: np.ndarray, codes: list) -> np.ndarray:
    return df.pivot(index="code", columns="date", values=col).reindex(index=codes, columns=dates).to_numpy(dtype=np.float64)


def load_explore_panel(con) -> tuple:
    """탐색 절반 패널. 읽자마자 잠금 절반을 버리고 ``half==0`` 을 단언한다."""
    from swing_it.storage import read_prices

    df = read_prices(con, cols=("code", "date", "high", "low", "close", "volume", "trade_value", "raw_close", "factor", "source", "close_src"), table=V2,
                     require_delisted=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df.date >= DATA_START) & (df.date <= DATA_END) & df.code.map(is_common)]
    df = df[df.code.map(half_of) == 0].copy()
    assert_explore(df.code.unique())
    assert (df.source == "naver_adj").any(), "폐지 종목이 안 읽혔다(read_prices 가드는 naver_adj 를 못 본다)"
    df = df[df.close_src != "pre_listing"]
    codes = sorted(df.code.unique())
    assert_explore(codes)
    nv = pd.read_sql_query(
        "select code,date,close as ncl,volume as nvol from naver_investor_trend where date between %s and %s and code = any(%s)",
        con, params=(DATA_START, DATA_END, codes))
    nv["date"] = pd.to_datetime(nv["date"])
    assert_explore(nv.code.unique())
    df = df.merge(nv, how="left", on=["code", "date"])
    df.loc[df.close <= 0, "close"] = np.nan
    df.loc[df.high <= 0, "high"] = np.nan
    df.loc[df.low <= 0, "low"] = np.nan
    dates = np.array(sorted(df.date.unique()), dtype="datetime64[ns]")
    names = load_names(con).set_index("code").reindex(codes)
    assert names.group.notna().sum() > 0.9 * len(codes)
    market = names.group.fillna("KOSDAQ").to_numpy()
    assert set(market) <= {"KOSPI", "KOSDAQ"}
    excluded = names.name.fillna("").map(name_excluded).to_numpy(dtype=bool)
    reason = names.reason.fillna("")
    distress = ((reason != "") & ~reason.str.contains(NONDISTRESS)).to_numpy(dtype=bool)
    ncl = wide(df, "ncl", dates, codes)
    nvol = wide(df, "nvol", dates, codes)
    tv = ncl * nvol / 1e6
    p = Panel(dates=dates, codes=codes, H=wide(df, "high", dates, codes), L=wide(df, "low", dates, codes),
              C=wide(df, "close", dates, codes), V=wide(df, "volume", dates, codes), TV=tv, RAW=ncl,
              market=market, excluded=excluded, distress=distress)
    return p, df


def assert_explore(codes) -> None:
    bad = [c for c in codes if half_of(c) != 0]
    assert not bad, f"잠금 절반 종목이 들어왔다: {len(bad)}"
