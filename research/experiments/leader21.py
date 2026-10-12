"""21세대 주도주 보유형 포트폴리오(META_EVOLUTION.md 21세대 — 결과 전 고정). 일별 시뮬레이터 + 격자 72 + 확장 워크포워드 + 기준선.

정의는 META_EVOLUTION.md 21세대 사전등록 그대로다. 캐시 배열은 종목×날짜(수정 주가), 거래대금(백만원)과 EPS as-of 는 DB.

사용: ``uv run --with numba --extra pg python research/experiments/leader21.py --cache <v13> --out <dir> --workers 10`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

SEED = 20261021
START, END = "2017-10-02", "2026-10-08"
FOLDS = (("2017-10-02", "2020-01-01", "2022-01-01"), ("2017-10-02", "2022-01-01", "2024-01-01"))
GRID = {"theta": (70, 80, 90), "N": (20, 50, 120), "ma_exit": (50, 100), "hard": (0.15, 0.25), "pyramid": (0, 1),
        "univ": ("all", "top200"), "regime": (0, 1)}            # 보강(검증 23차 권고, 결과 전): 시가총액 상위 200 · 시장 국면(시총 가중 200일선)
MAX_POS, RISK, POS_CAP, ADV_MIN, LU = 10, 0.005, 0.10, 1000.0, 1.29
SG = {"004690", "017390", "016710", "003100", "004360", "032190", "003380", "030210"}
_D: dict = {}


def _roll(df: pd.DataFrame, w: int, fn: str) -> np.ndarray:
    return getattr(df.rolling(w, min_periods=w), fn)().to_numpy().T


def prepare(cache: str) -> dict:
    import psycopg2

    from research.signals import pattern_events as PE
    from research.signals.eps_asof import eps_ok
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    codes = [str(c) for c in np.load(f"{cache}/codes.npy", allow_pickle=True)]
    O, H, L, C = (np.load(f"{cache}/{k}.npy").astype(float) for k in ("O", "H", "L", "C"))
    TR = np.load(f"{cache}/TR.npy").astype(bool)
    cost = np.load(f"{cache}/cost_tab.npy")
    F = np.load(f"{cache}/F.npy", mmap_mode="r")
    I_ = np.load(f"{cache}/I.npy", mmap_mode="r")
    mk_of = dict(zip(np.asarray(I_[:, PE.II["ci"]]).astype(int), np.asarray(F[:, PE.FI["market"]]).astype(int), strict=True))
    mkt = np.array([mk_of.get(i, 0) for i in range(len(codes))], np.int64)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    tv = pd.read_sql("select code, date, trade_value from daily_bars_adjusted where date >= '2016-08-01'", con, parse_dates=["date"])
    tvw = tv.pivot_table(index="date", columns="code", values="trade_value", aggfunc="last").reindex(index=days, columns=codes)
    adv = tvw.rolling(20, min_periods=15).mean().to_numpy().T
    Cd = pd.DataFrame(C.T)
    ma = {w: _roll(Cd, w, "mean") for w in (50, 100, 150, 200)}
    ma200_prev = np.concatenate([np.full((C.shape[0], 20), np.nan), ma[200][:, :-20]], axis=1)
    hi252, lo252 = _roll(pd.DataFrame(H.T), 252, "max"), _roll(pd.DataFrame(L.T), 252, "min")
    def ret(k):
        return C / np.concatenate([np.full((C.shape[0], k), np.nan), C[:, :-k]], axis=1) - 1.0
    raw = 2 * ret(63) + ret(126) + ret(189) + ret(252)
    uni = np.nan_to_num(adv) >= ADV_MIN
    rs = pd.DataFrame(np.where(uni & np.isfinite(raw), raw, np.nan).T).rank(axis=1, pct=True).to_numpy().T * 100
    with np.errstate(invalid="ignore"):
        tt = ((C > ma[50]) & (ma[50] > ma[150]) & (ma[150] > ma[200]) & (ma[200] > ma200_prev) & (C >= 0.75 * hi252) & (C >= 1.30 * lo252))
    brk = {n: C >= _roll(Cd, n, "max") - 1e-9 for n in GRID["N"]}
    # 시가총액(원 종가 × 상장주식수 as-of) → 그날 순위 · 시총 가중 시장(KOSPI 대용) 200일선 국면
    px = pd.read_sql("select code, date, close from daily_bars where date >= '2016-08-01'", con, parse_dates=["date"])
    pxw = px.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days, columns=codes)
    sh = pd.read_sql("select code, date, shares_outstanding from shares_outstanding_history", con, parse_dates=["date"])
    shw = sh.pivot_table(index="date", columns="code", values="shares_outstanding", aggfunc="last")
    shw = shw.reindex(index=shw.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
    cap = (pxw * shw).to_numpy().T
    crank = pd.DataFrame(cap.T).rank(axis=1, ascending=False).to_numpy().T
    r1 = C[:, 1:] / C[:, :-1] - 1
    r1 = np.concatenate([np.full((C.shape[0], 1), np.nan), r1], axis=1)
    cprev = np.concatenate([np.full((C.shape[0], 1), np.nan), cap[:, :-1]], axis=1)
    w = np.where(np.isfinite(r1) & np.isfinite(cprev) & (cprev > 0), cprev, 0.0)
    cw = np.where(w.sum(0) > 0, (np.nan_to_num(r1) * w).sum(0) / np.maximum(w.sum(0), 1e-12), 0.0)
    cw = np.clip(cw, -0.2, 0.2)
    cwi = np.cumprod(1 + cw)
    regime_on = cwi >= pd.Series(cwi).rolling(200, min_periods=200).mean().to_numpy()
    hi20 = C >= _roll(Cd, 20, "max") - 1e-9
    base = uni & tt & TR & (rs >= min(GRID["theta"]))
    ci, s = np.nonzero(base & brk[min(GRID["N"])] | base & brk[50] | base & brk[120])
    ok = eps_ok(codes, days, ci, s, con)
    eps = np.zeros_like(base)
    eps[ci[ok], s[ok]] = True
    return {"days": days, "codes": codes, "O": O, "H": H, "L": L, "C": C, "TR": TR, "cost": cost, "mkt": mkt, "rs": rs,
            "cand": base & eps, "brk": brk, "hi20": hi20, "ma": ma, "adv": adv, "uni": uni, "crank": crank, "cw": cw,
            "regime_on": regime_on}


def simulate(D: dict, cfg: dict, cost_mult: float) -> "tuple[pd.Series, pd.DataFrame]":
    days, O, H, L, C = D["days"], D["O"], D["H"], D["L"], D["C"]
    t0, t1 = int(days.searchsorted(pd.Timestamp(START))), int(days.searchsorted(pd.Timestamp(END), side="right"))
    unit = min(POS_CAP, RISK / cfg["hard"])
    max_units = max(1, int(round(POS_CAP / unit)))
    mae = D["ma"][cfg["ma_exit"]]
    cand = D["cand"] & D["brk"][cfg["N"]] & (D["rs"] >= cfg["theta"])
    if cfg.get("univ") == "top200":
        cand = cand & (np.nan_to_num(D["crank"], nan=1e9) <= 200)
    reg = cfg.get("regime", 0)
    cash, pos = 1.0, {}                        # ci → {qty, cost_basis, first_px, units, entry_t, last_px}
    p_buy, p_sell, p_add = [], set(), set()
    eq, trades = np.zeros(t1 - t0), []
    eq_prev = 1.0
    def close_pos(ci, px, t, why):
        nonlocal cash
        p = pos.pop(ci)
        c_rt = D["cost"][D["mkt"][ci], min(t, D["cost"].shape[1] - 1)] * cost_mult
        cash += p["qty"] * px * (1 - c_rt)
        trades.append({"code": D["codes"][ci], "entry": str(days[p["entry_t"]].date()), "exit": str(days[t].date()), "why": why,
                       "ret": px * (1 - c_rt) * p["qty"] / p["cost_basis"] - 1, "units": p["units"], "hold": t - p["entry_t"],
                       "w_peak": p["w_peak"]})
    for k, t in enumerate(range(t0, t1)):
        # 1) 시가 체결: 매도 → 추가 → 매수
        for ci in list(p_sell):
            if ci in pos:
                px = O[ci, t] if np.isfinite(O[ci, t]) else pos[ci]["last_px"]
                close_pos(ci, px, t, "ma_exit")
        p_sell.clear()
        for ci in list(p_add):
            if ci in pos and np.isfinite(O[ci, t]) and D["TR"][ci, t]:
                amt = min(unit * eq_prev, cash)
                if amt > 0:
                    pos[ci]["qty"] += amt / O[ci, t]
                    pos[ci]["cost_basis"] += amt
                    pos[ci]["units"] += 1
                    cash -= amt
        p_add.clear()
        for ci in p_buy:
            if ci in pos or len(pos) >= MAX_POS:
                continue
            op, pc = O[ci, t], C[ci, t - 1]
            if not (np.isfinite(op) and op > 0 and D["TR"][ci, t]) or (np.isfinite(pc) and op >= pc * LU):
                continue
            amt = min(unit * eq_prev, cash)
            if amt <= 0:
                continue
            pos[ci] = {"qty": amt / op, "cost_basis": amt, "first_px": op, "units": 1, "entry_t": t, "last_px": op, "w_peak": 0.0}
            cash -= amt
        p_buy = []
        # 2) 장중 하드 손절(평단 기준)
        for ci in list(pos):
            p = pos[ci]
            avg = p["cost_basis"] / p["qty"]
            stop = avg * (1 - cfg["hard"])
            if np.isfinite(L[ci, t]) and L[ci, t] <= stop:
                px = min(O[ci, t], stop) if np.isfinite(O[ci, t]) else stop
                close_pos(ci, px, t, "hard_stop")
        # 3) 종가 평가 · 신호
        mtm = 0.0
        for ci, p in pos.items():
            if np.isfinite(C[ci, t]):
                p["last_px"] = C[ci, t]
            mtm += p["qty"] * p["last_px"]
        e = cash + mtm
        eq[k] = e
        for ci, p in list(pos.items()):
            p["w_peak"] = max(p["w_peak"], p["qty"] * p["last_px"] / e)
            c_ = C[ci, t]
            if not np.isfinite(c_):
                p["nan"] = p.get("nan", 0) + 1
                if p["nan"] >= 5:
                    close_pos(ci, p["last_px"], t, "no_data")
                continue
            p["nan"] = 0
            if np.isfinite(mae[ci, t]) and c_ < mae[ci, t]:
                p_sell.add(ci)
            elif cfg["pyramid"] and p["units"] < max_units and c_ >= p["first_px"] * (1 + 0.10 * p["units"]) and D["hi20"][ci, t]:
                p_add.add(ci)
        if reg and not D["regime_on"][t]:                # 국면 꺼짐: 전부 다음 시가에 팔고 새로 안 산다
            p_sell.update(pos.keys())
            p_add.clear()
            p_buy = []
            eq_prev = e
            continue
        free = MAX_POS - len(pos) + len(p_sell)
        if free > 0:
            cs = np.flatnonzero(cand[:, t])
            cs = [c for c in cs if c not in pos]
            cs.sort(key=lambda c: -D["rs"][c, t])
            p_buy = cs[:free]
        eq_prev = e
    for ci in list(pos):
        close_pos(ci, pos[ci]["last_px"], t1 - 1, "end")
    return pd.Series(eq, index=days[t0:t1]), pd.DataFrame(trades)


def metrics(eq: pd.Series, lo=None, hi=None) -> dict:
    x = eq[(eq.index >= (lo or eq.index[0])) & (eq.index < (hi or eq.index[-1] + pd.Timedelta(days=1)))]
    x = x / x.iloc[0]
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
    cagr = x.iloc[-1] ** (1 / yrs) - 1
    mdd = float((x / x.cummax() - 1).min())
    r = x.resample("ME").last().pct_change().dropna()
    by, prev = {}, 1.0
    for y, g in x.groupby(x.index.year):
        by[int(y)] = float(g.iloc[-1] / prev - 1)
        prev = float(g.iloc[-1])
    return {"cagr": float(cagr), "mdd": mdd, "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan, "total": float(x.iloc[-1] - 1),
            "sharpe_m": float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan, "by_year": by}


def _init(cache):
    _D.update(prepare(cache))


def _task(a):
    i, cfg = a
    out = {"i": i}
    for cm in (1.0, 2.0):
        eq, tr = simulate(_D, cfg, cm)
        out[f"eq{int(cm)}"] = eq.to_numpy()
        out[f"tr{int(cm)}"] = tr
    out["idx"] = eq.index
    return out


def benchmarks(D: dict) -> pd.DataFrame:
    days, C = D["days"], D["C"]
    t0, t1 = int(days.searchsorted(pd.Timestamp(START))), int(days.searchsorted(pd.Timestamp(END), side="right"))
    r = C[:, 1:] / C[:, :-1] - 1
    r = np.concatenate([np.full((C.shape[0], 1), np.nan), r], axis=1)
    r = np.where(np.isfinite(r), np.clip(r, -0.3, 0.3), np.nan)
    uni = D["uni"]
    ew = np.nanmean(np.where(np.concatenate([np.zeros((C.shape[0], 1), bool), uni[:, :-1]], axis=1), r, np.nan), axis=0)
    # 대형주 대용: 직전 달 말 ADV20 상위 30 동일 비중(월 재조정)
    month = days.to_period("M")
    lc = np.full(len(days), np.nan)
    sel = None
    for t in range(t0, t1):
        if t == t0 or month[t] != month[t - 1]:
            a = np.nan_to_num(D["adv"][:, t - 1])
            sel = np.argsort(-a)[:30]
        lc[t] = np.nanmean(r[sel, t])
    idx = days[t0:t1]
    return pd.DataFrame({"market_cw": (1 + pd.Series(D["cw"][t0:t1], index=idx)).cumprod(),
                         "market_ew": (1 + pd.Series(np.nan_to_num(ew[t0:t1]), index=idx)).cumprod(),
                         "largecap30": (1 + pd.Series(np.nan_to_num(lc[t0:t1]), index=idx)).cumprod()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    keys = list(GRID)
    cfgs = [dict(zip(keys, v, strict=True)) for v in itertools.product(*GRID.values())]
    if a.limit:
        cfgs = cfgs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    with cf.ProcessPoolExecutor(min(a.workers, len(cfgs)), initializer=_init, initargs=(a.cache,)) as ex:
        res = sorted(ex.map(_task, list(enumerate(cfgs))), key=lambda r: r["i"])
    idx = res[0]["idx"]
    E1 = pd.DataFrame({i: r["eq1"] for i, r in enumerate(res)}, index=idx)
    E2 = pd.DataFrame({i: r["eq2"] for i, r in enumerate(res)}, index=idx)
    E1.to_pickle(os.path.join(a.out, "eq1.pkl"))
    E2.to_pickle(os.path.join(a.out, "eq2.pkl"))
    for cm in (1, 2):
        pd.concat([r[f"tr{cm}"].assign(cfg=r["i"]) for r in res]).to_csv(os.path.join(a.out, f"trades{cm}.csv.gz"), index=False)
    _init(a.cache)
    B = benchmarks(_D)
    B.to_csv(os.path.join(a.out, "bench.csv"))
    rng = np.random.default_rng(SEED)
    sel, parts1, parts2 = [], [], []
    for lo, te0, te1 in FOLDS:
        sc = {i: metrics(E1[i], lo, te0)["calmar"] for i in E1.columns}
        best = max(sc, key=lambda i: -np.inf if not np.isfinite(sc[i]) else sc[i])
        sel.append(best)
        for E, parts in ((E1, parts1), (E2, parts2)):
            seg = E[best][(E.index >= te0) & (E.index < te1)]
            parts.append(seg / seg.iloc[0])
    def stitch(parts):
        out, base = [], 1.0
        for p in parts:
            out.append(p * base)
            base = float(out[-1].iloc[-1])
        return pd.concat(out)
    oos1, oos2 = stitch(parts1), stitch(parts2)
    def oos_of(i, E):
        ps = []
        for (lo, te0, te1) in FOLDS:
            seg = E[i][(E.index >= te0) & (E.index < te1)]
            ps.append(seg / seg.iloc[0])
        return ps
    null = []
    for _ in range(3000):
        ps = [oos_of(int(rng.integers(0, len(cfgs))), E2)[k] for k in range(len(FOLDS))]
        null.append(metrics(stitch(ps))["cagr"])
    m2 = metrics(oos2)
    Bo = B[(B.index >= "2020-01-01") & (B.index < "2024-01-01")]
    bm = {k: metrics(Bo[k]) for k in Bo.columns}
    summ = {"selected": [cfgs[i] for i in sel], "oos_1x": metrics(oos1), "oos_2x": m2, "bench_oos": bm,
            "null_cagr_median": float(np.median(null)), "null_cagr_p90": float(np.quantile(null, 0.9)), "p_null": float(np.mean(np.array(null) >= m2["cagr"])),
            "grid_oos_2x": {str(i): metrics(stitch(oos_of(i, E2)))["cagr"] for i in range(len(cfgs))},
            "full_2017_2026_2x": {str(i): metrics(E2[i])["cagr"] for i in range(len(cfgs))},
            "contaminated_2024_2026": {"selected_last": metrics(E2[sel[-1]], "2024-01-01", None), "bench": {k: metrics(B[k], "2024-01-01", None) for k in B.columns}}}
    summ["pass"] = bool(m2["cagr"] > max(bm["market_ew"]["cagr"], bm["largecap30"]["cagr"], bm["market_cw"]["cagr"]) and m2["mdd"] > bm["market_ew"]["mdd"]
                        and summ["p_null"] <= 0.10 and sum(v > 0 for v in m2["by_year"].values()) >= 3)
    pd.DataFrame({"oos_1x": oos1, "oos_2x": oos2}).to_csv(os.path.join(a.out, "oos.csv"))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("grid_oos_2x", "full_2017_2026_2x")}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
