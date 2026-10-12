"""18세대 국장 적응 스웜(META_EVOLUTION.md 18세대 — 결과 전 고정): 미너비니 책 범위 안의 진입·청산 변형 전부(격자 = 입자)를
슬롯 제약 없는 사건 연구로 재고, 확장 워크포워드로 재선택한 표본 밖 성과를 같은 메뉴 무작위 설정 널과 비교한다.

모집단(유전자 pop): M0 = 17세대와 같다(ctx 1/10 표본 ∧ EPS ∧ 추세 템플릿 ∧ RS ≥ 70, 10일 고가 매수정지·10일 저가 손절) ·
tight = S 표 tight 행(같은 거르기, 원천 lvl/stp = 10일 고가/저가).

메뉴(책 범위 — MINERVINI_RULES.md):
- stop_max {5, 6, 7.5, 8, 10%}(TTC p56·TLW p314·p291) · gap_max {3, 5%}(MM p94·p96)
- be_r {2, 3}(TTC p60·61) · take_r {2, 3, 4, 6} × take_frac {1/3, 1/2}(TTC p163·MM p122·p126)
- trail {ma20, ma50}(MM p126 잰저 21일·TTC p162) · trail_gate {0, 1}(TTC p162) · climax {0, 1}(TTC p156) · max_hold {120, 250}
= 2 × 5 × 2 × 2 × 4 × 2 × 2 × 2 × 2 × 2 = 5,120 설정.

워크포워드(확장, 결과 전 고정): 학습 2017-01~2019-12 → 시험 2020-01~2021-12, 학습 2017-01~2021-12 → 시험 2022-01~2023-12.
학습 선택 기준 = 사건 평균(순 2배 비용 × min(1, 5%/손절폭)), 학습 사건 ≥ 150. 표본 밖 = 두 시험창을 이은 것(2020~2023).
널 = 같은 메뉴에서 무작위로 고른 설정의 표본 밖 평균(설정 전부의 표본 밖 분포 = 무작위 설정 널).

사용: ``PATTERN_GA_VARIANT=fwm PATTERN_GA_MENU="$(cat menu_g16s.json)" uv run --with numba --extra pg python
research/experiments/adapt18.py --cache <v13> --out <dir> --workers 10``
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

SEED = 20261018
LO, SEAL = "2017-01-01", "2024-01-01"
FOLDS = (("2017-01-01", "2020-01-01", "2022-01-01"), ("2017-01-01", "2022-01-01", "2024-01-01"))   # (학습 시작, 시험 시작, 시험 끝)
MIN_TRAIN = 150
MENU = {"pop": ("M0", "tight"), "stop_max": (0.05, 0.06, 0.075, 0.08, 0.10), "gap_max": (0.03, 0.05), "be_r": (2.0, 3.0),
        "take_r": (2.0, 3.0, 4.0, 6.0), "take_frac": (1 / 3, 1 / 2), "trail": ("ma20", "ma50"), "trail_gate": (0, 1),
        "climax": (0, 1), "max_hold": (120, 250)}
SG = {"004690", "017390", "016710", "003100", "004360", "032190", "003380", "030210"}
_W: dict = {}


def _init(cache: str) -> None:
    import research.experiments.pattern_ga_run as G
    from research.experiments.source_scan import MINERVINI
    from research.signals import event_families as EF
    from research.signals import event_sources as ES
    from research.signals import pattern_events as PE
    from research.signals import pattern_fast as PF
    A = {k: np.ascontiguousarray(np.load(f"{cache}/{k}.npy")) for k in ("O", "H", "L", "C", "MA20", "MA50", "ATR", "TR", "LIM", "cost_tab")}
    meta = json.load(open(f"{cache}/meta.json", encoding="utf-8"))
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    codes = np.load(f"{cache}/codes.npy", allow_pickle=True)
    seal, lo_d = int(days.searchsorted(pd.Timestamp(SEAL))), int(days.searchsorted(pd.Timestamp(LO)))
    F = np.load(f"{cache}/F.npy", mmap_mode="r")
    I_ = np.asarray(np.load(f"{cache}/I.npy", mmap_mode="r"))
    XF = np.load(f"{cache}/XF.npy")
    off = np.load(f"{cache}/off.npy")
    a, b = PE.table_slice(off, ES.TID_S)
    fam_lo = int(off[EF.TID_P])
    g0 = G.canonical({**G.decode(G.random_x(np.random.default_rng(0), 1)[0]), **MINERVINI, "fam_s": 1})
    hi10 = pd.DataFrame(A["H"].T).rolling(10, min_periods=10).max().to_numpy().T
    lo10 = pd.DataFrame(A["L"].T).rolling(10, min_periods=10).min().to_numpy().T
    dcol = I_[a:b, PE.II["d"]]
    rows0 = np.arange(a, b)[(dcol >= lo_d) & (dcol < seal)]
    pops = {}
    for pop, kind in (("M0", "ctx"), ("tight", "tight")):
        g = dict(g0, s_kind=ES.K[kind])
        r = rows0[EF.gene_mask(XF[rows0 - fam_lo], np.asarray(F[rows0]), g, "S", PE.FI)]
        pops[pop] = r
    _W.update(G=G, PF=PF, PE=PE, EF=EF, A=A, days=days, codes=codes, seal=seal, reform=int(days.searchsorted(pd.Timestamp(PE.TICK_REFORM_DAY))),
              LIM=np.minimum(A["LIM"], seal).astype(np.int64), I=I_, F=F, pops=pops, g0=g0, hi10=hi10, lo10=lo10,
              pmix=float(meta.get("p", 0.5)), fills={})


def _fills(pop: str, stop_max: float, gap_max: float) -> dict:
    key = (pop, stop_max, gap_max)
    if key not in _W["fills"]:
        W = _W
        rows = W["pops"][pop]
        ci = W["I"][rows, W["PE"].II["ci"]].astype(np.int64)
        s = W["I"][rows, W["PE"].II["i0"]].astype(np.int64)
        mk = np.asarray(W["F"][rows, W["PE"].FI["market"]]).astype(np.int64)
        g = dict(W["g0"], stop_max=stop_max, gap_max=gap_max)
        slip = float(W["G"].exit_vec(g)[W["PF"].X_SLIP])
        fx = W["EF"].fills(rows, W["I"], W["PE"].II, W["A"]["O"], W["A"]["H"], W["A"]["L"], W["A"]["C"], W["A"]["TR"], "stop", g, slip,
                           float(W["G"].LIMIT_UP_RATIO), mk, W["reform"], W["hi10"][ci, s], W["lo10"][ci, s])
        fx["mk"] = dict(zip(rows, mk, strict=True))
        fx["order"] = np.lexsort((fx["ci"], fx["d"]))
        W["fills"][key] = fx
    return _W["fills"][key]


def run_cfg(cfg: dict, detail: bool = False) -> pd.DataFrame:
    W = _W
    PF, PE, A = W["PF"], W["PE"], W["A"]
    fx = _fills(cfg["pop"], cfg["stop_max"], cfg["gap_max"])
    g = dict(W["g0"], stop_max=cfg["stop_max"], gap_max=cfg["gap_max"], be_r=cfg["be_r"], take_r=cfg["take_r"], take_frac=cfg["take_frac"],
             trail=cfg["trail"], trail_gate=cfg["trail_gate"], climax=cfg["climax"])
    px = W["G"].exit_vec(g)
    for k_ in (PF.X_FB_DAYS, PF.X_FAIL_EXIT, PF.X_REENTRY, PF.X_DYN):
        px[k_] = 0
    px[PF.X_MAX_HOLD] = cfg["max_hold"]
    out = np.empty(PF.N_O)
    d_, n_, R_, code_, det = [], [], [], [], []
    busy = {}
    for j in fx["order"]:
        c, d, e, sp = int(fx["ci"][j]), int(fx["d"][j]), float(fx["e"][j]), float(fx["sp"][j])
        if d >= W["seal"] or busy.get(c, -1) >= d:
            continue
        m_ = int(fx["mk"][fx["rows"][j]])
        PF.exit_path(A["O"], A["H"], A["L"], A["C"], A["MA20"], A["MA50"], A["ATR"], A["TR"], W["LIM"], c, d, e, sp, -np.inf,
                     A["H"][c, d - 1], A["H"][c, d - 1], 0.0, A["L"][c, d - 1], d - 1, 0.0, 0.0, bool(fx["skip0"][j]), m_, W["reform"], px, out)
        x = int(out[PF.O_X])
        busy[c] = x
        net2 = out[PF.O_PNL] - 2 * A["cost_tab"][m_, x]
        if fx["amb"][j]:
            s0 = sp * 0.999 - PE.tick_dated(sp, d, m_, W["reform"])
            net2 = W["pmix"] * (s0 / e - 1 - 2 * A["cost_tab"][m_, d]) + (1 - W["pmix"]) * net2
        dd = (e - sp) / e
        d_.append(d)
        n_.append(net2 * min(1.0, 0.05 / dd))
        R_.append(out[PF.O_PNL] / dd)
        code_.append(str(W["codes"][c]))
        det.append((x, e, sp, float(out[PF.O_FIN]), PF.REASONS[int(out[PF.O_REASON])], bool(out[PF.O_TP] >= 0), float(out[PF.O_PX1])))
    t = pd.DataFrame({"d": d_, "n2s": n_, "R": R_, "code": code_})
    if detail:                                   # 기술(스웜 검증용) — 판정 숫자는 같다
        dd = pd.DataFrame(det, columns=["x", "e", "sp", "exit_px", "reason", "partial", "partial_px"])
        t = pd.concat([t, dd], axis=1)
    return t


def _task(a: tuple) -> dict:
    i, cfg = a
    t = run_cfg(cfg)
    days = _W["days"]
    t["date"] = days[t["d"].to_numpy()]
    res = {"i": i}
    for k, (tr0, te0, te1) in enumerate(FOLDS):
        tr = t[(t["date"] >= tr0) & (t["date"] < te0)]
        te = t[(t["date"] >= te0) & (t["date"] < te1)]
        res[f"tr{k}_n"], res[f"tr{k}_m"] = len(tr), float(tr["n2s"].mean()) if len(tr) else np.nan
        res[f"te{k}_n"], res[f"te{k}_m"] = len(te), float(te["n2s"].mean()) if len(te) else np.nan
    oos = t[t["date"] >= FOLDS[0][1]]
    res["oos_n"], res["oos_m"] = len(oos), float(oos["n2s"].mean()) if len(oos) else np.nan
    res["all_m"] = float(t["n2s"].mean()) if len(t) else np.nan
    res["pR4"] = float((t["R"] >= 4).mean()) if len(t) else np.nan
    return res


def mt(x: np.ndarray, cl: np.ndarray, rng) -> float:
    u, inv = np.unique(cl, return_inverse=True)
    s_, n_ = np.bincount(inv, weights=x), np.bincount(inv)
    i = rng.integers(0, len(u), (3000, len(u)))
    return float(x.mean() / (s_[i].sum(1) / n_[i].sum(1)).std(ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="배선 확인용(앞 n 설정만)")
    a = ap.parse_args()
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    keys = list(MENU)
    cfgs = [dict(zip(keys, v, strict=True)) for v in itertools.product(*MENU.values())]
    if a.limit:
        cfgs = cfgs[:a.limit]
    with cf.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache,)) as ex:
        res = list(ex.map(_task, list(enumerate(cfgs)), chunksize=8))
    r = pd.DataFrame(res).set_index("i")
    r = pd.concat([pd.DataFrame(cfgs), r], axis=1)
    os.makedirs(a.out, exist_ok=True)
    r.to_csv(os.path.join(a.out, "grid.csv.gz"), index=False)
    # 워크포워드 선택
    sel, oos_parts = [], []
    _init(a.cache)
    for k in range(len(FOLDS)):
        ok = r[r[f"tr{k}_n"] >= MIN_TRAIN]
        best = ok[f"tr{k}_m"].idxmax()
        sel.append(int(best))
        t = run_cfg(cfgs[best], detail=True)
        dd = _W["days"][t["d"].to_numpy()]
        oos_parts.append(t[(dd >= FOLDS[k][1]) & (dd < FOLDS[k][2])].assign(date=dd[(dd >= FOLDS[k][1]) & (dd < FOLDS[k][2])]))
    oos = pd.concat(oos_parts)
    oos["month"] = oos["date"].dt.strftime("%Y-%m")
    rng = np.random.default_rng(SEED)
    m = float(oos["n2s"].mean())
    t_ = mt(oos["n2s"].to_numpy(), oos["month"].to_numpy(), rng)
    # 무작위 설정 널: 각 폴드에서 무작위 설정의 시험창 평균을 사건 수로 가중해 이은 값의 분포
    null = []
    for _ in range(5000):
        tot, n = 0.0, 0
        for k in range(len(FOLDS)):
            row = r.iloc[rng.integers(0, len(r))]
            tot += row[f"te{k}_m"] * row[f"te{k}_n"]
            n += row[f"te{k}_n"]
        null.append(tot / n)
    null = np.asarray(null)
    p_null = float((null >= m).mean())
    yr = oos.groupby(oos["date"].dt.year)["n2s"].mean()
    summ = {"selected": [cfgs[i] for i in sel], "oos_n": int(len(oos)), "oos_mean_pct": m * 100, "oos_t": t_,
            "oos_median_R": float(oos["R"].median()), "oos_pR4": float((oos["R"] >= 4).mean()),
            "oos_exSG_mean_pct": float(oos[~oos["code"].isin(SG)]["n2s"].mean() * 100),
            "by_year_pct": {int(k): float(v * 100) for k, v in yr.items()},
            "null_median_pct": float(np.median(null) * 100), "null_p95_pct": float(np.quantile(null, 0.95) * 100), "p_null": p_null,
            "grid_oos_frac_pos": float((r["oos_m"] > 0).mean()), "grid_oos_best_pct": float(r["oos_m"].max() * 100),
            "n_cfgs": len(r)}
    summ["pass"] = bool(summ["oos_n"] >= 240 and m >= 0.005 and t_ >= 2.0 and p_null <= 0.05 and (yr > 0).sum() >= 3)
    oos.to_csv(os.path.join(a.out, "oos_events.csv.gz"), index=False)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
