"""정정 G28-3: 원 종가 시총(read_raw_close)으로 21~26세대 핵심 팔(격자 215 A_open·C_close) 데이터 수정 재실행 — 기술만.

사용: ``uv run --with numba --extra pg python research/experiments/rawcap_rerun.py --cache <v13> --out <dir>`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]
warnings.filterwarnings("ignore")

import research.experiments.leader21 as L  # noqa: E402
from research.experiments.leader23 import ARMS, BASE  # noqa: E402

PERIODS = (("2017-10-02", "2024-01-01"), ("2024-01-01", None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D0 = L.prepare(a.cache)
    D1 = L.prepare(a.cache, raw_close=True)
    out = {}
    u0, u1 = D0["crank"] <= 200, D1["crank"] <= 200
    inter, union = (u0 & u1).sum(0), (u0 | u1).sum(0)
    jac = pd.Series(np.where(union > 0, inter / np.maximum(union, 1), np.nan), index=D0["days"])
    out["top200_jaccard"] = {str(y): float(v) for y, v in jac.groupby(jac.index.year).mean().items()}
    reg = pd.Series(D0["regime_on"] == D1["regime_on"], index=D0["days"])
    out["regime_agree"] = {str(y): float(v) for y, v in reg.groupby(reg.index.year).mean().items()}
    for nm in ("A_open", "C_close"):
        for cm in (1, 2):
            for tag, D in (("defect", D0), ("raw", D1)):
                eq, tr = L.simulate(D, {**BASE, **ARMS[nm]}, cm)
                for lo, hi in PERIODS:
                    m = L.metrics(eq, lo, hi)
                    out[f"{nm}|{cm}x|{tag}|{lo[:4]}"] = {"cagr": float(m["cagr"]), "mdd": float(m["mdd"]), "n_trades": int(len(tr))}
                eq.to_pickle(os.path.join(a.out, f"eq_{nm}_{cm}x_{tag}.pkl"))
    for tag, D in (("defect", D0), ("raw", D1)):
        B = L.benchmarks(D)
        out[f"bench|{tag}"] = {k: {lo[:4]: float(L.metrics(B[k], lo, hi)["cagr"]) for lo, hi in PERIODS} for k in B.columns}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
