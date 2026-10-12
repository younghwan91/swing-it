"""20세대 후처리: 3층이 해마다 고른 설정의 사건을 다시 돌려 봉인 구간(2024-01~2026-10) 판정(사전등록 바)·포트폴리오·벤치마크를 낸다.

사용: ``... uv run --with numba --extra pg python research/experiments/meta20_post.py --cache <v13> --run <meta20 out>`` (KR_QUANT_DB 필요)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

import research.experiments.adapt18 as A  # noqa: E402
from research.experiments.meta20 import END  # noqa: E402

SEED = 20261021
SG = A.SG


def mt(x, cl, rng):
    u, inv = np.unique(cl, return_inverse=True)
    s_, n_ = np.bincount(inv, weights=x), np.bincount(inv)
    i = rng.integers(0, len(u), (3000, len(u)))
    return float(x.mean() / (s_[i].sum(1) / n_[i].sum(1)).std(ddof=1))


def portfolio(t: pd.DataFrame, days, lo: str, hi: str, slots: int = 10) -> pd.Series:
    t = t.sort_values(["d", "code"])
    open_x, take = [], []
    for i, r in t.iterrows():
        open_x = [x for x in open_x if x > r.d]
        if len(open_x) < slots:
            take.append(i)
            open_x.append(r.x)
    p = t.loc[take]
    idx = days[(days >= lo) & (days <= hi)]
    ev = p.groupby(days[p["x"].to_numpy()])["n2s"].sum() * (1.0 / slots)
    return (1 + ev.reindex(idx).fillna(0)).cumprod(), p


def stats(eq: pd.Series) -> dict:
    m = eq.resample("ME").last()
    r = m.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    by, prev = {}, 1.0
    for y, g in eq.groupby(eq.index.year):
        by[int(y)] = float(g.iloc[-1] / prev - 1)
        prev = float(g.iloc[-1])
    return {"total": float(eq.iloc[-1] / eq.iloc[0] - 1), "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1),
            "mdd": float((eq / eq.cummax() - 1).min()), "sharpe_m": float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan, "by_year": by}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    rng = np.random.default_rng(SEED)
    summ = json.load(open(os.path.join(a.run, "summary.json"), encoding="utf-8"))
    A._init(a.cache, END, True)
    days = A._W["days"]
    parts = []
    for y, v in summ["by_year"].items():
        if v["cfg"] is None:
            continue
        t = A.run_cfg(v["cfg"], detail=True)
        dd = days[t["d"].to_numpy()]
        t = t[dd.year == int(y)].copy()
        t["year"], t["proc"] = int(y), "|".join(map(str, v["proc"]))
        parts.append(t)
    t = pd.concat(parts)
    t["date"] = days[t["d"].to_numpy()]
    t["month"] = t["date"].dt.strftime("%Y-%m")
    t.to_csv(os.path.join(a.run, "meta_trades.csv.gz"), index=False)
    out = {}
    for name, yrs in (("sealed_2024_2026", (2024, 2025, 2026)), ("design_2020_2023", (2020, 2021, 2022, 2023))):
        x = t[t["year"].isin(yrs)]
        if not len(x):
            out[name] = {"n": 0}
            continue
        yr = x.groupby("year")["n2s"].mean()
        out[name] = {"n": int(len(x)), "mean_pct": float(x["n2s"].mean() * 100), "t": mt(x["n2s"].to_numpy(), x["month"].to_numpy(), rng),
                     "years_pos": int((yr > 0).sum()), "by_year_pct": {int(k): float(v * 100) for k, v in yr.items()},
                     "exSG_mean_pct": float(x[~x["code"].isin(SG)]["n2s"].mean() * 100),
                     "drop_top5_pct": float(x.sort_values("n2s").iloc[:-5]["n2s"].mean() * 100) if len(x) > 5 else None,
                     "win": float((x["n2s"] > 0).mean()), "median_R": float(x["R"].median()), "p_R4": float((x["R"] >= 4).mean())}
    s = out["sealed_2024_2026"]
    out["pass"] = bool(s.get("n", 0) and s["mean_pct"] >= 0.5 and s["t"] >= 2 and s["years_pos"] >= 2)
    # 포트폴리오 · 벤치마크
    eqS, pS = portfolio(t[t["year"] >= 2024], days, "2024-01-02", "2026-10-08")
    eqD, pD = portfolio(t[(t["year"] >= 2020) & (t["year"] <= 2023)], days, "2020-01-02", "2023-12-28")
    import psycopg2  # noqa: PLC0415
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    ix = pd.read_sql("select code, date, close from sector_index where code in ('001','101') and date >= '2024-01-01' order by date", con,
                     parse_dates=["date"])
    C = np.load(f"{a.cache}/C.npy", mmap_mode="r")
    def ew(lo, hi):
        idx = days[(days >= lo) & (days <= hi)]
        i0, i1 = days.get_loc(idx[0]), days.get_loc(idx[-1])
        cs = np.asarray(C[:, i0 - 1:i1 + 1], float)
        r = cs[:, 1:] / cs[:, :-1] - 1
        r[~np.isfinite(r)] = np.nan
        return (1 + pd.Series(np.nanmean(np.clip(r, -0.3, 0.3), axis=0), index=idx).fillna(0)).cumprod()
    curves = pd.DataFrame({"strategy": eqS, "market_ew": ew("2024-01-02", "2026-10-08")})
    for code, nm in (("001", "kospi"), ("101", "kosdaq")):
        z = ix[ix["code"] == code].set_index("date")["close"].sort_index()
        if len(z):
            curves[nm] = (z / z.iloc[0]).reindex(curves.index, method="ffill")
    curves.to_csv(os.path.join(a.run, "sealed_curves.csv"))
    pd.DataFrame({"strategy": eqD, "market_ew": ew("2020-01-02", "2023-12-28")}).to_csv(os.path.join(a.run, "design_curves.csv"))
    out["portfolio_sealed"] = {k: stats(curves[k].dropna()) for k in curves.columns}
    out["portfolio_sealed"]["trades_taken"] = int(len(pS))
    out["portfolio_design"] = {"strategy": stats(eqD), "trades_taken": int(len(pD))}
    out["chosen"] = summ["by_year"]
    json.dump(out, open(os.path.join(a.run, "post.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
