"""thinband 공용 부품 — 패널·격자·무작위 20종목 슬롯 북·지표·바. 정본은 research/logs/thinband/PREREG.md.

smallcap 사건 엔진과 다른 새 코드다. 손계산 시험은 ``tests/test_thinband_lib.py``, 원 DB 거래 재계산은 9단계 전에 둔다.
연구 단위는 현금 없는 슬롯 금액이다. 비용 배수는 경로를 바꾸지 않는다(비용은 따로 쌓아 나중에 곱한다).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE_LIBRARY = "thinband 러너의 공용 함수(진입점 없음)"

WARM_START, SIG_START, DATA_END = "2016-05-01", "2016-09-01", "2023-12-31"
STEP, SLOTS, N_RUNS, N_LABEL = 60, 20, 1000, 200
SEED = 20261102
BUY_ROWS = 10
SUB_BOUNDARY = np.datetime64("2020-01-01")
VOID_MIN_PERIODS, VOID_MIN_EXPOSURE = 28, 0.60
FACTOR_START = "2017-09-01"
SEGMENTS = {"2018": ("2018-01-01", "2018-12-31"), "2018Q4": ("2018-09-28", "2019-01-03"),
            "2020-03": ("2020-02-17", "2020-03-19"), "2022": ("2022-01-01", "2022-12-31")}
# K4: 사용자 선택 "중간"(2026-11-02, 결과 전). PREREG §5.
K4_MDD_MULT = 1.3
K4_Q5_PNL_WON = -1e8
_JUMP = True                           # 시험용: False 면 날마다 평가해서 한 번에 평가한 값과 대조한다


@dataclass(frozen=True)
class Cfg:
    band: tuple = L.BAND_U
    stay: "tuple | None" = None        # 잔류 대역(버퍼 라벨). None = band 와 같다
    entry: str = "split"               # split = 최대 10행 분할, old = a+1 전량(라벨)
    close_fill: bool = False           # 종가 대용(라벨, 체결 불가능)
    order: float = L.ORDER_WON
    step: int = STEP
    offset: int = 0


def periods_of(t0: int, T: int, step: int = STEP, offset: int = 0) -> list:
    """(a, b) 격자. 끝 자투리가 step 절반 미만이면 앞 기간에 합친다."""
    g = list(range(t0 + offset, T - 1, step)) + [T - 1]
    if len(g) >= 3 and g[-1] - g[-2] < step // 2:
        del g[-2]
    return list(zip(g[:-1], g[1:]))


class Ctx:
    """패널과 대역 마스크·지수를 묶는다. 런마다 다시 만들지 않는다."""

    def __init__(self, p: L.Panel, idx: dict, cfg: Cfg, doom: "np.ndarray | None" = None) -> None:
        self.p, self.cfg = p, cfg
        self.idx = idx
        self.u = p.universe(cfg.band)
        self.hold = p.universe(cfg.stay) if cfg.stay else self.u
        self.periods = periods_of(p.sig_t0, p.T, cfg.step, cfg.offset)
        self.doom = np.zeros(p.N, dtype=bool) if doom is None else doom
        self.I = [idx[p.market[i]] for i in range(p.N)]
        self.ids = {a: np.flatnonzero(self.u[:, a] & np.isfinite(p.Cb[:, a])) for a, _ in self.periods}


@dataclass
class Run:
    gross: np.ndarray
    cost: np.ndarray
    idx: np.ndarray
    expo: np.ndarray
    halted: np.ndarray
    recs: list          # (기간, 종목, 손익, 비용, 지수손익)
    holds: list         # (종목, 진입행, 끝행, 값 배수, 부실 여부)
    locks: int
    liq_rows: list
    cfg: Cfg
    doom_writeoff: int = 0
    censored_halted_nondistress: int = 0


def _rate(p: L.Panel, side: str, c: int, d: int, q: float, adv_day: int) -> float:
    adv = p.ADV[c, adv_day]
    adv = max(adv if np.isfinite(adv) else L.ADV_FLOOR_M, L.ADV_FLOOR_M)
    sig = p.SIG20[c, adv_day]
    sig = min(sig if np.isfinite(sig) else L.VOL_FALLBACK, 1.0)
    return L.cost_one_way(side, p.day_py[d], p.market[c], adv, sig, p.RAW[c, d], max(q, 1.0))


def simulate(ctx: Ctx, rng: np.random.Generator) -> Run:
    """무작위 20슬롯 한 회차. 격자일 a 마감에 정하고 a+1 부터 체결한다."""
    p, cfg = ctx.p, ctx.cfg
    T = p.T
    gross, cost, idxp, expo, halted = (np.zeros(T) for _ in range(5))
    pos: dict = {}      # c → [값, 진입행, 진입 Cp, 잠김]
    recs, holds, liq = [], [], []
    locks = 0

    for k, (a, b) in enumerate(ctx.periods):
        orders: dict = {}
        occupied = 0
        for c in pos:
            if not p.traded[c, a]:      # 정지: 복원·교체 없음, 슬롯을 차지한다
                orders[c] = None
                occupied += 1
            elif ctx.hold[c, a]:
                orders[c] = cfg.order
                occupied += 1
            else:
                orders[c] = 0.0
        pool = ctx.ids[a]
        if len(pos):
            pool = pool[~np.isin(pool, np.fromiter(pos, dtype=np.int64))]
        need = max(SLOTS - occupied, 0)
        if need and len(pool):
            for c in rng.choice(pool, size=min(need, len(pool)), replace=False).tolist():
                pos[c] = [0.0, a + 1, None, False]
                orders[c] = cfg.order
        rows = []
        for c in orders:
            adv = p.ADV[c, a]
            if orders[c] and np.isfinite(adv) and adv > 0:
                rows.append(cfg.order / (L.PARTICIPATION * adv * 1e6))
        liq.append(max(rows) if rows else 0.0)
        for c in list(orders):
            locks += _step(ctx, c, a, b, k, orders[c], pos, gross, cost, idxp, expo, halted, recs, holds)

    end = T - 1
    run = Run(gross, cost, idxp, expo, halted, recs, holds, locks, liq, cfg)
    for c, st in pos.items():
        h = st[0]
        if h <= 0:
            continue
        if ctx.doom[c]:                 # 정지로 표본을 나가는 KIND 부실: −100%
            gross[end] -= h
            recs.append((len(ctx.periods) - 1, c, -h, 0.0, 0.0))
            run.doom_writeoff += 1
            holds.append((c, st[1], end, 0.0, True))
            continue
        if not p.traded[c, end]:
            run.censored_halted_nondistress += 1
        q_rate = _rate(p, "SELL", c, end, h, end)
        cost[end] += h * q_rate
        recs.append((len(ctx.periods) - 1, c, 0.0, h * q_rate, 0.0))
        holds.append((c, st[1], end, _mult(st, p.Cp[c], end), False))
    return run


def _mult(st: list, cp: np.ndarray, d: int) -> float:
    """진입 뒤 값 배수(복원과 무관한 종목 수익). 못 샀으면 1."""
    return float(cp[d] / st[2]) if st[2] else 1.0


def _step(ctx, c, a, b, k, target, pos, gross, cost, idxp, expo, halted, recs, holds) -> int:
    """종목 하나의 기간 (a, b]. 체결이 끝나면 남은 구간은 한 번에 평가한다."""
    p, cfg = ctx.p, ctx.cfg
    st = pos[c]
    h = st[0]
    cp, ind = p.Cp[c], ctx.I[c]
    last = int(p.last[c])
    pnl0, cost0, idx0 = 0.0, 0.0, 0.0
    locks = 0
    side, rem, stop = None, 0.0, b
    if target is not None:
        diff = target - h
        if target == 0.0:
            side, rem = "SELL", float("inf")
        elif diff > 1.0:
            side, rem, stop = "BUY", diff, min(a + (1 if cfg.entry == "old" else BUY_ROWS), b)
        elif diff < -1.0:
            side, rem = "SELL", -diff
    rate = None
    d = a
    while d < b and d < last:
        if (side is None or rem <= 1e-6 or d >= stop) and _JUMP:
            nxt = min(b, last)            # 남은 구간 한 번에
            if h > 0 and nxt > d:
                base = h / cp[d]
                seg = base * cp[d:nxt + 1]
                gross[d + 1:nxt + 1] += np.diff(seg)
                pnl0 += float(seg[-1] - h)
                ii = seg[:-1] * (ind[d + 1:nxt + 1] / ind[d:nxt] - 1.0)
                idxp[d + 1:nxt + 1] += ii
                idx0 += float(ii.sum())
                expo[d + 1:nxt + 1] += seg[:-1]
                halted[d + 1:nxt + 1] += seg[:-1] * ~p.traded[c, d + 1:nxt + 1]
                h = float(seg[-1])
            d = nxt
            side = None
            continue
        d += 1
        if h > 0:
            dl = h * (cp[d] / cp[d - 1] - 1.0)
            gross[d] += dl
            pnl0 += dl
            ii = h * (ind[d] / ind[d - 1] - 1.0)
            idxp[d] += ii
            idx0 += ii
            expo[d] += h
            if not p.traded[c, d]:
                halted[d] += h
            h += dl
        if not p.traded[c, d] or side is None or rem <= 1e-6 or d > stop:
            continue
        r = p.r[c, d]
        tv = p.TV[c, d]
        cap = L.PARTICIPATION * tv * 1e6 if np.isfinite(tv) else 0.0
        if side == "BUY":
            if np.isfinite(r) and r >= p.thr[d] - 1e-9:
                continue
            q = min(rem, cap)
            if cfg.entry == "old" and st[2] is None and cap < rem:
                q = 0.0
            if q <= 0:
                continue
            if rate is None:
                rate = _rate(p, "BUY", c, d, rem, a)
            ratio = 1.0 if cfg.close_fill else p.C[c, d] / p.TP[c, d]
            gain = q * (ratio - 1.0)
            gross[d] += gain
            pnl0 += gain
            cost[d] += q * rate
            cost0 += q * rate
            h += q + gain
            rem -= q
            if st[2] is None:
                st[1], st[2] = d, cp[d] / ratio
        else:
            if not p.tail[c, d] and np.isfinite(r) and r <= -p.thr[d] + 1e-9:
                st[3] = True
                locks += 1
                continue
            q = min(rem, h, cap)
            if q <= 0:
                continue
            if rate is None:
                rate = _rate(p, "SELL", c, d, min(rem, h), d)
            if cfg.close_fill:
                ratio = 1.0
            elif st[3]:
                ratio = p.LOWPX[c, d] / p.C[c, d]
            else:
                ratio = p.TP[c, d] / p.C[c, d]
            st[3] = False
            gain = q * (ratio - 1.0)
            gross[d] += gain
            pnl0 += gain
            cost[d] += q * rate
            cost0 += q * rate
            h -= q
            rem -= q
    st[0] = h
    if last <= b and not ctx.doom[c] and last < p.T - 1:   # 표본 안에서 끝난 시리즈: 마지막 종가로 현금화
        if h > 0:
            sr = _rate(p, "SELL", c, last, h, last)
            cost[last] += h * sr
            cost0 += h * sr
            holds.append((c, st[1], last, _mult(st, cp, last), bool(p.distress[c])))
        pos.pop(c)
    elif target == 0.0 and h <= 1e-6:
        holds.append((c, st[1], d, _mult(st, cp, d), False))
        pos.pop(c)
    elif ctx.doom[c] and d < b:         # 부실 정지: 슬롯에 묶인 채 값 유지
        halted[d + 1:b + 1] += h
        expo[d + 1:b + 1] += h
    recs.append((k, c, pnl0, cost0, idx0))
    return locks


def period_series(ctx: Ctx, run: Run, m: float) -> dict:
    """기간별 슬롯 기준 순수익·순 능동(분모 = 20 × 주문)."""
    den = SLOTS * ctx.cfg.order
    net, act, exp = [], [], []
    for a, b in ctx.periods:
        g = run.gross[a + 1:b + 1].sum()
        c = run.cost[a + 1:b + 1].sum()
        ii = run.idx[a + 1:b + 1].sum()
        net.append((g - m * c) / den)
        act.append((g - m * c - ii) / den)
        exp.append(run.expo[a + 1:b + 1].mean() / den)
    return {"net": np.array(net), "active": np.array(act), "exposure": np.array(exp)}


def curve_metrics(ctx: Ctx, run: Run, m: float, aum: float = L.AUM_WON) -> dict:
    """고정 3억 분모 곡선(현금 없는 슬롯 손익을 그대로 더한다). 연구 단위와 따로 보고한다."""
    p = ctx.p
    t0 = p.sig_t0
    daily = (run.gross - m * run.cost)[t0:] / aum
    ctrl = run.idx[t0:] / aum
    eq = 1.0 + np.cumsum(daily)
    top = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    dd = eq / top - 1.0
    under = run_len = 0
    for x in dd < 0:
        run_len = run_len + 1 if x else 0
        under = max(under, run_len)
    w = 250
    roll = np.convolve(daily, np.ones(w), "valid") if len(daily) >= w else np.array([daily.sum()])
    ceq = 1.0 + np.cumsum(ctrl)
    cdd = ceq / np.maximum.accumulate(np.r_[1.0, ceq])[1:] - 1.0
    seg = {}
    dates = p.dates[t0:]
    for name, (s, e) in SEGMENTS.items():
        mask = (dates >= np.datetime64(s)) & (dates <= np.datetime64(e))
        seg[name] = {"book": float(daily[mask].sum()), "ctrl": float(ctrl[mask].sum())}
    ha = run.halted[t0:] / aum
    durs = [x[2] - x[1] for x in run.holds]
    longest = 0
    for c, e, x, _, _ in run.holds:
        tr = p.traded[c, e:x + 1]
        if not tr.all():
            ed = np.flatnonzero(np.diff(np.r_[1, tr.astype(int), 1]))
            longest = max(longest, int((ed[1::2] - ed[0::2]).max()))
    return {"mdd": float(dd.min()), "underwater": int(under), "worst250": float(roll.min()), "ctrl_mdd": float(cdd.min()),
            "segments": seg, "halted_mean": float(ha.mean()), "halted_max": float(ha.max()), "liq_rows_max": float(max(run.liq_rows or [0.0])),
            "halt_longest": longest, "locks": run.locks, "total": float(daily.sum()), "total_ctrl": float(ctrl.sum()), "hold_med": float(np.median(durs)) if durs else float("nan")}


def tail_metrics(run: Run, order: float) -> dict:
    """보유 단위 꼬리와 상위 제거. 기간-슬롯 단위는 recs."""
    vals = np.array([x[3] for x in run.holds]) if run.holds else np.array([])
    dist = sum(1 for x in run.holds if x[4])
    r = np.array([(x[2] - 2.0 * x[3] - x[4]) / order for x in run.recs]) if run.recs else np.zeros(1)
    srt = np.sort(r)[::-1]
    return {"n_minus50": int((vals <= 0.5).sum()), "n_minus90": int((vals <= 0.1).sum()), "distressed": dist,
            "slot_mean": float(r.mean()), "slot_median": float(np.median(r)), "slot_mean_ex5": float(srt[5:].mean()) if len(srt) > 5 else float("nan"),
            "slot_mean_ex20": float(srt[20:].mean()) if len(srt) > 20 else float("nan")}


def year_block_interval(diff: np.ndarray, years: np.ndarray, seed: int = SEED, n: int = 10000) -> tuple:
    """연도 블록 부트스트랩 95% 구간(보고용, 바 아님)."""
    rng = np.random.default_rng(seed)
    uy = np.unique(years)
    by = [diff[years == y] for y in uy]
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(uy), len(uy))
        out[i] = np.concatenate([by[j] for j in pick]).mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def period_block_interval(x: np.ndarray, block: int = 4, seed: int = SEED, n: int = 10000) -> tuple:
    """기간 블록 부트스트랩 평균 95% 구간. NW lag 3 이 낙관적이어서 같이 낸다."""
    rng = np.random.default_rng(seed)
    N = len(x)
    k = int(np.ceil(N / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, N, k)
        idx = (st[:, None] + np.arange(block)) % N
        out[i] = x[idx.ravel()[:N]].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def kill_bars(active_u: np.ndarray, active_u2: np.ndarray, starts: np.ndarray, exposure: float,
              mdd_median: float, ctrl_mdd_median: float, q5_final_pnl_won: float) -> dict:
    """K1~K4·VOID. 입력은 회차 평균 기간 순 능동(비용 2배). 숫자와 참거짓을 내고 판정은 사람이 읽는다.

    K4(사용자 선택): 중앙 회차 MDD 크기 ≤ 노출 맞춘 지수 MDD 크기 × 1.3, 그리고 q5 회차 최종 손익 ≥ −1억(고정 3억 분모).
    """
    first = starts < SUB_BOUNDARY
    diff = active_u - active_u2
    years = starts.astype("datetime64[Y]").astype(int) + 1970
    k4a = abs(mdd_median) <= abs(ctrl_mdd_median) * K4_MDD_MULT
    k4b = q5_final_pnl_won >= K4_Q5_PNL_WON
    return {"K1": float(active_u.mean()), "K1_ok": bool(active_u.mean() > 0),
            "K2": [float(active_u[first].mean()), float(active_u[~first].mean())],
            "K2_ok": bool(active_u[first].mean() > 0 and active_u[~first].mean() > 0),
            "K3": float(diff.mean()), "K3_ok": bool(diff.mean() > 0), "K3_year_block": year_block_interval(diff, years),
            "K4_mdd_median": mdd_median, "K4_ctrl_mdd_median": ctrl_mdd_median, "K4_mdd_limit": abs(ctrl_mdd_median) * K4_MDD_MULT,
            "K4_q5_final_pnl_won": q5_final_pnl_won, "K4a_ok": bool(k4a), "K4b_ok": bool(k4b), "K4_ok": bool(k4a and k4b),
            "periods": len(active_u), "exposure": exposure,
            "VOID": bool(len(active_u) < VOID_MIN_PERIODS or exposure < VOID_MIN_EXPOSURE)}


def load_panel(con) -> tuple:
    """2016-09~2023 전 종목. 2016-05~08 은 거래량·거래대금(ADV 준비)만 읽고 가격은 읽지 않는다. 2024 이후는 SQL 에서 자른다."""
    want = pd.read_sql_query(f"select distinct code from {L.V2} where date between %s and %s", con, params=(WARM_START, DATA_END))  # noqa: S608
    codes = sorted(c for c in want.code if L.is_common(c))
    px = pd.read_sql_query(f"select code,date,high,low,close,volume,trade_value,raw_close,source,close_src from {L.V2} "  # noqa: S608
                           "where date between %s and %s and code = any(%s)", con, params=(SIG_START, DATA_END, codes))
    wv = pd.read_sql_query(f"select code,date,volume from {L.V2} where date >= %s and date < %s and code = any(%s)", con,  # noqa: S608
                           params=(WARM_START, SIG_START, codes))
    df = pd.concat([px, wv], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    assert df.date.max() <= pd.Timestamp(DATA_END), "표본 끝을 넘었다"
    df = df[df.close_src != "pre_listing"]
    nv = pd.read_sql_query("select code,date,close as ncl,volume as nvol from naver_investor_trend where date between %s and %s and code = any(%s)",
                           con, params=(SIG_START, DATA_END, codes))
    tvw = pd.read_sql_query("select code,date,close*volume/1e6 as tvw from naver_investor_trend where date >= %s and date < %s and code = any(%s)",
                            con, params=(WARM_START, SIG_START, codes))
    for x in (nv, tvw):
        x["date"] = pd.to_datetime(x["date"])
    df = df.merge(nv, how="left", on=["code", "date"]).merge(tvw, how="left", on=["code", "date"])
    for col in ("close", "high", "low"):
        df.loc[df[col] <= 0, col] = np.nan
    df["tv"] = (df.ncl * df.nvol / 1e6).fillna(df.tvw)
    codes = sorted(df.code.unique())
    dates = np.array(sorted(df.date.unique()), dtype="datetime64[ns]")
    kind = pd.read_sql_query("select code,delist_date,reason from kind_delistings", con)
    kind["delist_date"] = pd.to_datetime(kind.delist_date)
    names = L.load_names(con).set_index("code").reindex(codes)
    market = names.group.fillna("KOSDAQ").to_numpy()
    excluded = names.name.fillna("").map(L.name_excluded).to_numpy(dtype=bool)
    reason = names.reason.fillna("")
    distress = ((reason != "") & ~reason.str.contains(L.NONDISTRESS)).to_numpy(dtype=bool)
    p = L.Panel(dates=dates, codes=codes, H=L.wide(df, "high", dates, codes), L=L.wide(df, "low", dates, codes),
                C=L.wide(df, "close", dates, codes), V=L.wide(df, "volume", dates, codes), TV=L.wide(df, "tv", dates, codes),
                RAW=L.wide(df, "ncl", dates, codes), market=market, excluded=excluded, distress=distress, sig_start=SIG_START)
    return p, df, kind


def doom_mask(p: L.Panel, kind: pd.DataFrame) -> np.ndarray:
    """정지 상태로 표본을 나가고(마지막 거래 < 표본 끝) KIND 폐지일이 표본 뒤인 부실 종목. PREREG §3 M1."""
    k = kind.dropna(subset=["delist_date"]).drop_duplicates("code", keep="last").set_index("code").delist_date
    d = pd.Series(p.codes).map(k).to_numpy(dtype="datetime64[ns]")
    with np.errstate(invalid="ignore"):
        late = ~np.isnat(d) & (d > p.dates[-1])
    return p.distress & late & (p.last < p.T - 1)


def label_cfgs(base: Cfg) -> dict:
    """라벨 설정. 승자를 고르지 않는다."""
    return {"entry_old": replace(base, entry="old"), "close_proxy": replace(base, close_fill=True),
            "buffer_1.5_40": replace(base, stay=(150.0, 4000.0)), "aum_1eok": replace(base, order=5e6), "no_doom": replace(base),
            **{f"offset_{o}": replace(base, offset=o) for o in (10, 20, 30, 40, 50)}}
