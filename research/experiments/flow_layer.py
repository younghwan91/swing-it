"""17세대 수급 층(META_EVOLUTION.md 17세대 — 결과 전 고정): 미너비니 뼈대(기하 맞춘 모집단) 위에 기관+외국인 수급 변수 하나.

모집단 M0 = S 표 ctx 행(맥락인 날의 1/10 결정적 표본) ∧ EPS 사용자 규칙 ∧ 추세 템플릿 ∧ RS ≥ 70, 진입 = 10일 고가 + 1호가 매수정지
(시가 > 주문가 × 1.03 이면 안 삼), 손절 = 10일 저가 − 1호가(7.5% 상한, 2% 하한), 청산 = 16b 미너비니 고정(``source_scan.MINERVINI``).
슬롯 제약 없는 사건 연구(같은 종목 보유 중 새 사건 건너뜀 — 검증 19차 geomnull 과 같은 방식). 지표 = 순 2배 비용 × min(1, 5%/손절폭).

변수(방향 사전 고정, 클수록 좋다): fl20 = s 의 (기관+외국인) 20일 순매수 수량 / 20일 거래량. 보조 fl5(다중성에 센다).

사용: ``PATTERN_GA_VARIANT=fwm PATTERN_GA_MENU="$(cat menu_g16s.json)" uv run --with numba --extra pg python
research/experiments/flow_layer.py --cache <v13> --out <dir> [--max-hold 120]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

SEED = 20261017
N_BOOT = 3000
N_PERM = 1000
LO, SEAL = "2017-01-01", "2024-01-01"
SG = {"004690", "017390", "016710", "003100", "004360", "032190", "003380", "030210"}   # 2023-04 SG증권발 폭락 연루(검증 18·19차)


def events(cache: str, max_hold: int) -> pd.DataFrame:
    import research.experiments.pattern_ga_run as G
    from research.experiments.source_scan import MINERVINI
    from research.signals import event_families as EF
    from research.signals import event_sources as ES
    from research.signals import pattern_events as PE
    from research.signals import pattern_fast as PF
    A = {k: np.ascontiguousarray(np.load(f"{cache}/{k}.npy")) for k in ("O", "H", "L", "C", "MA20", "MA50", "ATR", "TR", "LIM", "cost_tab")}
    meta = json.load(open(f"{cache}/meta.json", encoding="utf-8"))
    pmix = float(meta.get("p", 0.5))
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    codes = np.load(f"{cache}/codes.npy", allow_pickle=True)
    seal, lo_d = int(days.searchsorted(pd.Timestamp(SEAL))), int(days.searchsorted(pd.Timestamp(LO)))
    LIM = np.minimum(A["LIM"], seal).astype(np.int64)
    reform = int(days.searchsorted(pd.Timestamp(PE.TICK_REFORM_DAY)))
    F = np.load(f"{cache}/F.npy", mmap_mode="r")
    I_ = np.asarray(np.load(f"{cache}/I.npy", mmap_mode="r"))
    XF = np.load(f"{cache}/XF.npy")
    off = np.load(f"{cache}/off.npy")
    a, b = PE.table_slice(off, ES.TID_S)
    fam_lo = int(off[EF.TID_P])
    g = dict(G.canonical({**G.decode(G.random_x(np.random.default_rng(0), 1)[0]), **MINERVINI, "fam_s": 1}), s_kind=ES.K["ctx"])
    px = G.exit_vec(g)
    for k_ in (PF.X_FB_DAYS, PF.X_FAIL_EXIT, PF.X_REENTRY, PF.X_DYN):
        px[k_] = 0
    px[PF.X_MAX_HOLD] = max_hold
    slip = float(px[PF.X_SLIP])
    hi10 = pd.DataFrame(A["H"].T).rolling(10, min_periods=10).max().to_numpy().T
    lo10 = pd.DataFrame(A["L"].T).rolling(10, min_periods=10).min().to_numpy().T
    dcol = I_[a:b, PE.II["d"]]
    rows = np.arange(a, b)[(dcol >= lo_d) & (dcol < seal)]
    Ff = np.asarray(F[rows])
    Xr = XF[rows - fam_lo]
    rows = rows[EF.gene_mask(Xr, Ff, g, "S", PE.FI)]
    ci = I_[rows, PE.II["ci"]].astype(np.int64)
    s = I_[rows, PE.II["i0"]].astype(np.int64)
    mk = np.asarray(F[rows, PE.FI["market"]]).astype(np.int64)
    fx = EF.fills(rows, I_, PE.II, A["O"], A["H"], A["L"], A["C"], A["TR"], "stop", g, slip, float(G.LIMIT_UP_RATIO), mk, reform,
                  hi10[ci, s], lo10[ci, s])
    fl = {r: (XF[r - fam_lo, EF.XI["fl20_s"]], XF[r - fam_lo, EF.XI["fl5_s"]]) for r in fx["rows"]}
    mkt = dict(zip(rows, mk, strict=True))
    out = np.empty(PF.N_O)
    recs, busy = [], {}
    for j in np.lexsort((fx["ci"], fx["d"])):
        c, d, e, sp = int(fx["ci"][j]), int(fx["d"][j]), float(fx["e"][j]), float(fx["sp"][j])
        if d >= seal or busy.get(c, -1) >= d:
            continue
        r = int(fx["rows"][j])
        m_ = int(mkt[r])
        PF.exit_path(A["O"], A["H"], A["L"], A["C"], A["MA20"], A["MA50"], A["ATR"], A["TR"], LIM, c, d, e, sp, -np.inf,
                     A["H"][c, d - 1], A["H"][c, d - 1], 0.0, A["L"][c, d - 1], d - 1, 0.0, 0.0, bool(fx["skip0"][j]), m_, reform, px, out)
        x = int(out[PF.O_X])
        busy[c] = x
        net2 = out[PF.O_PNL] - 2 * A["cost_tab"][m_, x]
        if fx["amb"][j]:
            s0 = sp * 0.999 - PE.tick_dated(sp, d, m_, reform)
            net2 = pmix * (s0 / e - 1 - 2 * A["cost_tab"][m_, d]) + (1 - pmix) * net2
        dd = (e - sp) / e
        recs.append({"code": str(codes[c]), "d": d, "order_day": str(days[d].date()), "month": str(days[d])[:7], "year": int(days[d].year),
                     "n2s": net2 * min(1.0, 0.05 / dd), "R": out[PF.O_PNL] / dd, "reason": PF.REASONS[int(out[PF.O_REASON])],
                     "stop": dd, "hold": x - d, "fl20": float(fl[r][0]), "fl5": float(fl[r][1])})
    return pd.DataFrame(recs)


def mboot(x: np.ndarray, cl: np.ndarray, rng) -> "tuple[float, float]":
    u, inv = np.unique(cl, return_inverse=True)
    s_, n_ = np.bincount(inv, weights=x), np.bincount(inv)
    i = rng.integers(0, len(u), (N_BOOT, len(u)))
    return float(x.mean()), float((s_[i].sum(1) / n_[i].sum(1)).std(ddof=1))


def analyze(t: pd.DataFrame, var: str, rng) -> dict:
    t = t[np.isfinite(t[var])].copy()
    t["q"] = t.groupby("year")[var].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))   # 연도 안 5분위
    big = (t["R"] >= 4).astype(float)
    q = {int(k): {"n": int(len(g_)), "mean_pct": float(g_["n2s"].mean() * 100), "p_R4": float((g_["R"] >= 4).mean())}
         for k, g_ in t.groupby("q")}
    hi, lo = t["q"] == 4, t["q"] == 0
    # Q5 − Q1: 월 평균 차의 월 부트스트랩
    mm = t[hi | lo].assign(side=np.where(hi[hi | lo], 1, 0)).groupby(["month", "side"])["n2s"].mean().unstack().dropna()
    dd = (mm[1] - mm[0]).to_numpy()
    bi = rng.integers(0, len(dd), (N_BOOT, len(dd)))
    spread, se = float(dd.mean()), float(dd[bi].mean(1).std(ddof=1))
    # 큰 승자 빈도 차 P(R ≥ 4)
    pm = t[hi | lo].assign(big=big[hi | lo], side=np.where(hi[hi | lo], 1, 0)).groupby(["month", "side"])["big"].mean().unstack().dropna()
    pd_ = (pm[1] - pm[0]).to_numpy()
    pb_ = rng.integers(0, len(pd_), (N_BOOT, len(pd_)))
    # 순열 널: 달 안에서 변수 섞기 → Q5 − Q1 사건 평균 차
    obs = float(t.loc[hi, "n2s"].mean() - t.loc[lo, "n2s"].mean())
    perm = np.empty(N_PERM)
    v = t[var].to_numpy()
    grp = t.groupby("month").indices
    yr = t["year"].to_numpy()
    for k in range(N_PERM):
        vp = v.copy()
        for ix in grp.values():
            vp[ix] = vp[rng.permutation(ix)]
        qq = pd.Series(vp).groupby(yr).transform(lambda z: pd.qcut(z.rank(method="first"), 5, labels=False)).to_numpy()
        perm[k] = t["n2s"].to_numpy()[qq == 4].mean() - t["n2s"].to_numpy()[qq == 0].mean()
    # 거래 가능한 규칙(부호, 맞춤 없음): var > 0
    f = t[t[var] > 0]
    m, se_f = mboot(f["n2s"].to_numpy(), f["month"].to_numpy(), rng)
    bm = t.groupby("month")["n2s"].mean()
    dx = (f.groupby("month")["n2s"].mean() - bm).dropna().to_numpy()
    bx = rng.integers(0, len(dx), (N_BOOT, len(dx)))
    yf = f.groupby("year")["n2s"].mean()
    return {"n": int(len(t)), "quintiles": q, "spread_pct": spread * 100, "t_spread": spread / se if se else np.nan,
            "pR4_diff": float(pd_.mean()), "t_pR4": float(pd_.mean() / pd_[pb_].mean(1).std(ddof=1)),
            "perm_obs_pct": obs * 100, "perm_p": float((perm >= obs).mean()),
            "rule_pos": {"n": int(len(f)), "per_year": len(f) / 7.0, "mean_pct": m * 100, "t": m / se_f if se_f else np.nan,
                         "vs_base_pp": float(dx.mean() * 100), "t_vs_base": float(dx.mean() / dx[bx].mean(1).std(ddof=1)),
                         "years_pos": int((yf > 0).sum()), "by_year_pct": {int(k): float(x * 100) for k, x in yf.items()},
                         "exSG_mean_pct": float(f[~f["code"].isin(SG)]["n2s"].mean() * 100),
                         "median_R": float(f["R"].median()), "p_R4": float((f["R"] >= 4).mean())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hold", type=int, default=120)
    a = ap.parse_args()
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    rng = np.random.default_rng(SEED)
    t = events(a.cache, a.max_hold)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "events.csv.gz"), index=False)
    m, se = mboot(t["n2s"].to_numpy(), t["month"].to_numpy(), rng)
    base = {"n": int(len(t)), "mean_pct": m * 100, "t": m / se, "median_R": float(t["R"].median()), "p_R4": float((t["R"] >= 4).mean()),
            "by_year_pct": {int(k): float(x * 100) for k, x in t.groupby("year")["n2s"].mean().items()},
            "flow_cover": float(np.isfinite(t["fl20"]).mean())}
    res = {"base": base, "fl20": analyze(t, "fl20", rng), "fl5": analyze(t, "fl5", rng), "max_hold": a.max_hold}
    r = res["fl20"]
    rp = r["rule_pos"]
    res["pass_fl20"] = bool(r["t_spread"] >= 2.0 and r["perm_p"] <= 0.05 and rp["n"] >= 420 and rp["mean_pct"] >= 0.5 and rp["t"] >= 2.0
                           and rp["vs_base_pp"] >= 0.5 and rp["t_vs_base"] >= 1.5 and rp["years_pos"] >= 5)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
