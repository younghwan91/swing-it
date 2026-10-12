"""19세대 실적 이벤트(PEAD) 사건 연구(META_EVOLUTION.md 19세대 — 결과 전 고정).

사건 = (종목, 분기)의 첫 실적 공시(earnings_disclosures 의 잠정·손익구조변동·정기보고서 중 최초 rcept_dt = r). 날짜만 있어 장 마감 뒤 공시일 수
있으므로 **반응일 s = r 다음 거래일**(r 당일은 정보 전일 수 있다). 반응 EAR = C(s) / C(r 전 거래일) − 1(공시 전 종가 → 반응일 종가, 2일).
진입 = s+1 에 H(s) + 1호가 매수정지(시가 > 주문가 × 1.05 면 안 삼), 손절 = L(s) − 1호가, 10% 상한·2% 하한(정정 G18-1 관례).
청산(관례 고정): 3R(장중 고가)에 본전, 3R 에 절반, 나머지 MA20 종가 이탈, 급등 매도 끔, 보유 상한 250일. 순 2배 비용 × min(1, 5%/손절폭).

거르기(모집단 P0): 반응일 s 에 ADV20 ≥ 10억(거래대금, daily_bars_adjusted) ∧ 거래 가능. EPS 사용자 규칙은 **새 as-of**(값 = max(avail_date,
정기보고서 rcept_dt) 다음 거래일부터)로 s 시점 최신 분기 순이익 > 전년 같은 분기 ∧ > 0 — 변형 V2 에서만(주 시험은 정보원 그 자체).
변수(방향 고정): EAR 클수록 좋다. 규칙: EAR ≥ +5%.

사용: ``uv run --with numba --extra pg python research/experiments/pead19.py --cache <v13> --out <dir>`` (KR_QUANT_DB 필요)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

SEED = 20261019
N_BOOT, N_PERM = 3000, 1000
LO, SEAL = "2019-03-01", "2024-01-01"
ADV_MIN = 1e9
EAR_MIN = 0.05
SG = {"004690", "017390", "016710", "003100", "004360", "032190", "003380", "030210"}


def load(cache: str):
    import psycopg2
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    codes = [str(c) for c in np.load(f"{cache}/codes.npy", allow_pickle=True)]
    disc = pd.read_sql("select code, period, kind, rcept_dt from earnings_disclosures", con, parse_dates=["rcept_dt"])
    earn = pd.read_sql("select code, period, avail_date, netinc, netinc_prior from earnings where avail_date >= '2018-01-01'", con,
                       parse_dates=["avail_date"])
    tv = pd.read_sql("select code, date, trade_value from daily_bars_adjusted where date >= '2018-10-01' and date < '2024-02-01'", con,
                     parse_dates=["date"])
    return days, codes, disc, earn, tv


def build_events(cache: str) -> "tuple[pd.DataFrame, dict]":
    from research.signals import pattern_events as PE
    days, codes, disc, earn, tv = load(cache)
    cmap = {c: i for i, c in enumerate(codes)}
    nd = len(days)
    first = disc.groupby(["code", "period"], as_index=False)["rcept_dt"].min()
    first = first[first["code"].isin(cmap)]
    r_idx = days.searchsorted(first["rcept_dt"].to_numpy(), side="left")          # r 이상 첫 거래일
    r_is_day = (r_idx < nd) & (days[np.minimum(r_idx, nd - 1)] == first["rcept_dt"].to_numpy())
    s = np.where(r_is_day, r_idx + 1, r_idx)                                         # 반응일 = r 다음 거래일
    pre = np.where(r_is_day, r_idx - 1, r_idx - 1)                                   # 공시 전 마지막 거래일
    ev = first.assign(ci=first["code"].map(cmap).astype(int), s=s, pre=pre)
    ev = ev[(ev["s"] < nd - 1) & (ev["pre"] >= 0)]
    ev = ev[(days[ev["s"].to_numpy()] >= pd.Timestamp(LO)) & (days[ev["s"].to_numpy()] < pd.Timestamp(SEAL))]
    C = np.load(f"{cache}/C.npy", mmap_mode="r")
    TR = np.load(f"{cache}/TR.npy", mmap_mode="r")
    ci, s_, pr = ev["ci"].to_numpy(), ev["s"].to_numpy(), ev["pre"].to_numpy()
    ev["ear"] = np.asarray(C[ci, s_], float) / np.asarray(C[ci, pr], float) - 1.0
    ev["tr"] = np.asarray(TR[ci, s_]).astype(bool) & np.asarray(TR[ci, np.minimum(s_ + 1, nd - 1)]).astype(bool)
    # ADV20(거래대금, s 까지)
    w = tv.pivot_table(index="date", columns="code", values="trade_value", aggfunc="last").reindex(index=days)
    adv = w.rolling(20, min_periods=15).mean()
    ev["adv20"] = [adv.iat[si, adv.columns.get_loc(c)] if c in adv.columns else np.nan for c, si in zip(ev["code"], s_, strict=True)]
    # EPS 새 as-of: 값 = max(avail_date, 정기보고서 rcept_dt) 다음 거래일부터
    per = disc[disc["kind"] == "periodic"].groupby(["code", "period"], as_index=False)["rcept_dt"].min().rename(columns={"rcept_dt": "per_dt"})
    e2 = earn.merge(per, on=["code", "period"], how="left")
    e2["known"] = e2[["avail_date", "per_dt"]].max(axis=1)
    kd = days.searchsorted(e2["known"].to_numpy(), side="right")                    # known 날 다음 거래일 행
    e2 = e2.assign(krow=kd).sort_values("krow")
    ev = ev.sort_values("s")
    m = pd.merge_asof(ev, e2[["code", "krow", "netinc", "netinc_prior", "period"]].rename(columns={"period": "eps_period"}),
                      left_on="s", right_on="krow", by="code", direction="backward")
    m["eps_ok"] = (m["netinc"] > 0) & (m["netinc"] > m["netinc_prior"])
    m["date"] = days[m["s"].to_numpy()]
    m = m[m["tr"] & (m["adv20"] >= ADV_MIN) & np.isfinite(m["ear"])]
    return m.reset_index(drop=True), {"days": days, "codes": codes}


def simulate(cache: str, ev: pd.DataFrame) -> pd.DataFrame:
    import research.experiments.pattern_ga_run as G
    from research.experiments.source_scan import MINERVINI
    from research.signals import event_families as EF
    from research.signals import pattern_events as PE
    from research.signals import pattern_fast as PF
    A = {k: np.ascontiguousarray(np.load(f"{cache}/{k}.npy")) for k in ("O", "H", "L", "C", "MA20", "MA50", "ATR", "TR", "LIM", "cost_tab")}
    meta = json.load(open(f"{cache}/meta.json", encoding="utf-8"))
    pmix = float(meta.get("p", 0.5))
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    seal = int(days.searchsorted(pd.Timestamp(SEAL)))
    LIM = np.minimum(A["LIM"], seal).astype(np.int64)
    reform = int(days.searchsorted(pd.Timestamp(PE.TICK_REFORM_DAY)))
    F = np.load(f"{cache}/F.npy", mmap_mode="r")
    I0 = np.load(f"{cache}/I.npy", mmap_mode="r")
    mk_of = dict(zip(np.asarray(I0[:, PE.II["ci"]]).astype(int), np.asarray(F[:, PE.FI["market"]]).astype(int), strict=True))
    g = G.canonical({**G.decode(G.random_x(np.random.default_rng(0), 1)[0]), **MINERVINI, "fam_s": 1})
    g.update(stop_max=0.10, gap_max=0.05, be_r=3.0, take_r=3.0, take_frac=0.5, trail="ma20", trail_gate=0, climax=0)
    px = G.exit_vec(g)
    for k_ in (PF.X_FB_DAYS, PF.X_FAIL_EXIT, PF.X_REENTRY, PF.X_DYN):
        px[k_] = 0
    px[PF.X_MAX_HOLD] = 250
    slip = float(px[PF.X_SLIP])
    n = len(ev)
    I_ = np.zeros((n, len(PE.I_COLS)), np.int64)
    I_[:, PE.II["ci"]] = ev["ci"].to_numpy()
    I_[:, PE.II["i0"]] = ev["s"].to_numpy()
    mk = np.array([mk_of.get(int(c), 0) for c in ev["ci"]], np.int64)
    ci, s = ev["ci"].to_numpy(), ev["s"].to_numpy()
    fx = EF.fills(np.arange(n), I_, PE.II, A["O"], A["H"], A["L"], A["C"], A["TR"], "stop", g, slip, 1.3, mk, reform, A["H"][ci, s], A["L"][ci, s])
    out = np.empty(PF.N_O)
    recs, busy = [], {}
    for j in np.lexsort((fx["ci"], fx["d"])):
        c, d, e, sp = int(fx["ci"][j]), int(fx["d"][j]), float(fx["e"][j]), float(fx["sp"][j])
        if d >= seal or busy.get(c, -1) >= d:
            continue
        r = int(fx["rows"][j])
        m_ = int(mk[r])
        PF.exit_path(A["O"], A["H"], A["L"], A["C"], A["MA20"], A["MA50"], A["ATR"], A["TR"], LIM, c, d, e, sp, -np.inf, A["H"][c, d - 1],
                     A["H"][c, d - 1], 0.0, A["L"][c, d - 1], d - 1, 0.0, 0.0, bool(fx["skip0"][j]), m_, reform, px, out)
        x = int(out[PF.O_X])
        busy[c] = x
        net2 = out[PF.O_PNL] - 2 * A["cost_tab"][m_, x]
        if fx["amb"][j]:
            s0 = sp * 0.999 - PE.tick_dated(sp, d, m_, reform)
            net2 = pmix * (s0 / e - 1 - 2 * A["cost_tab"][m_, d]) + (1 - pmix) * net2
        dd = (e - sp) / e
        row = ev.iloc[r]
        recs.append({"code": row["code"], "period": row["period"], "d": d, "date": str(days[d].date()), "month": str(days[d])[:7],
                     "year": int(days[d].year), "ear": float(row["ear"]), "eps_ok": bool(row["eps_ok"]), "n2s": net2 * min(1.0, 0.05 / dd),
                     "R": out[PF.O_PNL] / dd, "reason": PF.REASONS[int(out[PF.O_REASON])], "stop": dd, "hold": x - d, "e": e, "sp": sp,
                     "exit_day": str(days[x].date()), "exit_px": float(out[PF.O_FIN])})
    return pd.DataFrame(recs)


def mboot(x, cl, rng):
    u, inv = np.unique(cl, return_inverse=True)
    s_, n_ = np.bincount(inv, weights=x), np.bincount(inv)
    i = rng.integers(0, len(u), (N_BOOT, len(u)))
    return float(x.mean()), float((s_[i].sum(1) / n_[i].sum(1)).std(ddof=1))


def rule_stats(f: pd.DataFrame, base: pd.DataFrame, rng) -> dict:
    m, se = mboot(f["n2s"].to_numpy(), f["month"].to_numpy(), rng)
    dx = (f.groupby("month")["n2s"].mean() - base.groupby("month")["n2s"].mean()).dropna().to_numpy()
    bx = rng.integers(0, len(dx), (N_BOOT, len(dx)))
    yf = f.groupby("year")["n2s"].mean()
    return {"n": int(len(f)), "per_year": len(f) / 4.83, "mean_pct": m * 100, "t": m / se if se else np.nan, "vs_base_pp": float(dx.mean() * 100),
            "t_vs_base": float(dx.mean() / dx[bx].mean(1).std(ddof=1)), "years_pos": int((yf > 0).sum()),
            "by_year_pct": {int(k): float(v * 100) for k, v in yf.items()}, "median_R": float(f["R"].median()), "p_R4": float((f["R"] >= 4).mean()),
            "exSG_mean_pct": float(f[~f["code"].isin(SG)]["n2s"].mean() * 100),
            "drop_top5_pct": float(f.sort_values("n2s").iloc[:-5]["n2s"].mean() * 100) if len(f) > 5 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    rng = np.random.default_rng(SEED)
    ev, _ = build_events(a.cache)
    t = simulate(a.cache, ev)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "events.csv.gz"), index=False)
    m, se = mboot(t["n2s"].to_numpy(), t["month"].to_numpy(), rng)
    res = {"n_disclosure_events": int(len(ev)), "base": {"n": int(len(t)), "mean_pct": m * 100, "t": m / se,
                                                         "by_year_pct": {int(k): float(v * 100) for k, v in t.groupby("year")["n2s"].mean().items()}}}
    t["q"] = t.groupby("year")["ear"].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
    res["quintiles"] = {int(k): {"n": int(len(g_)), "ear_med": float(g_["ear"].median()), "mean_pct": float(g_["n2s"].mean() * 100),
                                 "p_R4": float((g_["R"] >= 4).mean())} for k, g_ in t.groupby("q")}
    hi, lo = t["q"] == 4, t["q"] == 0
    mm = t[hi | lo].assign(side=np.where(hi[hi | lo], 1, 0)).groupby(["month", "side"])["n2s"].mean().unstack().dropna()
    dd = (mm[1] - mm[0]).to_numpy()
    bi = rng.integers(0, len(dd), (N_BOOT, len(dd)))
    res["spread_pct"], res["t_spread"] = float(dd.mean() * 100), float(dd.mean() / dd[bi].mean(1).std(ddof=1))
    obs = float(t.loc[hi, "n2s"].mean() - t.loc[lo, "n2s"].mean())
    v, yr, grp = t["ear"].to_numpy(), t["year"].to_numpy(), t.groupby("month").indices
    perm = np.empty(N_PERM)
    for k in range(N_PERM):
        vp = v.copy()
        for ix in grp.values():
            vp[ix] = vp[rng.permutation(ix)]
        qq = pd.Series(vp).groupby(yr).transform(lambda z: pd.qcut(z.rank(method="first"), 5, labels=False)).to_numpy()
        perm[k] = t["n2s"].to_numpy()[qq == 4].mean() - t["n2s"].to_numpy()[qq == 0].mean()
    res["perm_p"] = float((perm >= obs).mean())
    res["rule_ear5"] = rule_stats(t[t["ear"] >= EAR_MIN], t, rng)
    res["rule_ear5_eps"] = rule_stats(t[(t["ear"] >= EAR_MIN) & t["eps_ok"]], t, rng)
    rp = res["rule_ear5"]
    res["pass"] = bool(res["t_spread"] >= 2 and res["perm_p"] <= 0.05 and rp["n"] >= 300 and rp["mean_pct"] >= 0.5 and rp["t"] >= 2
                       and rp["vs_base_pp"] >= 0.5 and rp["t_vs_base"] >= 1.5 and rp["years_pos"] >= 3)
    res["abstain"] = bool(rp["mean_pct"] <= 0)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
