"""24세대(META_EVOLUTION.md 24세대 — 결과 전 고정): 종목 교차 홀드아웃 — 종목을 해시로 둘로 나눠 반쪽 A 에서만 고르고 반쪽 B 에서 평가.

골격 = 23세대 C(신호일 종가 진입, leader_C). 격자 = 청산 MA {100, 150, 200} × θ {80, 90} × 하드 {15, 25%} = 12.
선택 = 반쪽 A(2017-10 ~ 2023) 2배 비용 칼마 최대. 평가 = 반쪽 B 같은 기간: 선택 설정 vs 기준(C = MA100·θ90·하드 15%).

사용: ``uv run --with numba --extra pg python research/experiments/leader24.py --cache <v13> --out <dir> --workers 6`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

import research.experiments.leader21 as L  # noqa: E402

BASE = {"N": 20, "pyramid": 1, "univ": "top200", "regime": 1, "entry": "close"}
GRID = {"ma_exit": (100, 150, 200), "theta": (80, 90), "hard": (0.15, 0.25)}
REF = {"ma_exit": 100, "theta": 90, "hard": 0.15}
WIN = ("2017-10-02", "2024-01-01")
_D: dict = {}


def half(code: str) -> str:
    return "A" if int(hashlib.sha1(code.encode()).hexdigest(), 16) % 2 == 0 else "B"


def _init(cache):
    _D.update(L.prepare(cache))
    codes = _D["codes"]
    _D["half"] = {h: [c for c in codes if half(c) != h] for h in ("A", "B")}    # 그 반쪽 밖 = 제외 목록


def _task(a):
    i, cfg, h = a
    eq, tr = L.simulate(_D, {**BASE, **cfg, "exclude": _D["half"][h]}, 2.0)
    x = eq[(eq.index >= WIN[0]) & (eq.index < WIN[1])]
    tw = tr[(tr["entry"] >= WIN[0]) & (tr["entry"] < WIN[1])]
    contrib = (tw["ret"] * tw["w_peak"]).groupby(tw["code"]).sum().sort_values(ascending=False) if len(tw) else pd.Series(dtype=float)
    m = L.metrics(x)
    return {"i": i, "half": h, **{k: v for k, v in m.items() if k != "by_year"}, "by_year": m["by_year"], "n": int(len(tw)),
            "top5": list(contrib.index[:5]), "eq": x}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    keys = list(GRID)
    cfgs = [dict(zip(keys, v, strict=True)) for v in itertools.product(*GRID.values())]
    tasks = [(i, c, h) for i, c in enumerate(cfgs) for h in ("A", "B")]
    with cf.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache,)) as ex:
        res = list(ex.map(_task, tasks))
    os.makedirs(a.out, exist_ok=True)
    R = pd.DataFrame([{k: v for k, v in r.items() if k != "eq"} for r in res])
    R["cfg"] = [json.dumps(cfgs[i]) for i in R["i"]]
    R.to_csv(os.path.join(a.out, "grid.csv"), index=False)
    pd.DataFrame({f"{r['i']}_{r['half']}": r["eq"] for r in res}).to_csv(os.path.join(a.out, "curves.csv"))
    A_ = R[R["half"] == "A"].set_index("i")
    B_ = R[R["half"] == "B"].set_index("i")
    sel = int(A_["calmar"].idxmax())
    ref = cfgs.index(REF)
    # 반사실 상위 5 제거(반쪽 B)
    _init(a.cache)
    def drop5(i):
        top = B_.loc[i, "top5"]
        eq, _ = L.simulate(_D, {**BASE, **cfgs[i], "exclude": _D["half"]["B"] + list(top)}, 2.0)
        return L.metrics(eq[(eq.index >= WIN[0]) & (eq.index < WIN[1])])["cagr"]
    summ = {"selected": cfgs[sel], "ref": REF, "A_sel": A_.loc[sel, ["cagr", "mdd", "calmar"]].to_dict(),
            "B_sel": B_.loc[sel, ["cagr", "mdd", "calmar"]].to_dict(), "B_ref": B_.loc[ref, ["cagr", "mdd", "calmar"]].to_dict(),
            "B_sel_by_year": B_.loc[sel, "by_year"], "B_ref_by_year": B_.loc[ref, "by_year"],
            "B_sel_drop5": drop5(sel), "B_ref_drop5": drop5(ref),
            "rank_corr_A_B": float(A_["calmar"].rank().corr(B_["calmar"].rank())),
            "B_grid_cagr": B_["cagr"].round(4).tolist()}
    d = summ["B_sel"]["cagr"] - summ["B_ref"]["cagr"]
    yrs = sum((summ["B_sel_by_year"][y] - summ["B_ref_by_year"][y]) > 0 for y in summ["B_ref_by_year"])
    summ["pass"] = bool(sel != ref and d >= 0.02 and summ["B_sel_drop5"] >= summ["B_ref_drop5"] and yrs >= 4
                        and summ["B_sel"]["mdd"] >= summ["B_ref"]["mdd"] - 0.05)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
