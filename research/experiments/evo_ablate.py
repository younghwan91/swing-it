"""진화 어블레이션(검증 12차 — 측정 방법): 리더 후보의 폴드 출력 설정(시드 집합별)을 **고정**하고 조건 하나만 바꿔 폴드 1~4 시험을 다시 계산.

GA 를 다시 돌리지 않으므로 다른 유전자의 궤적 변화가 섞이지 않는다. 결과는 ``pattern_ga_evo/ABL_<tag>_s<off>.json``·``_trades.csv``
(원장 ``pattern_ga_evo`` 에 줄을 남긴다 — 다중성에 센다). 판정은 ``evo_judge.py`` 로 기준 후보와 짝으로 본다.

사용: ``PATTERN_GA_VARIANT=fwm uv run --with numba --extra pg python research/experiments/evo_ablate.py --base G7A --tag pilot
--set '{"first_frac": 0.5, "pyr_close": 1, "pyramid": "1r"}' --cache <캐시>``
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path[:0] = [".", "src"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", required=True, help="모든 폴드 설정에 덮어쓸 유전자 dict(JSON)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="research/logs/sepa_explore")
    a = ap.parse_args()
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    import numpy as np

    import research.experiments.pattern_ga_run as G
    from swing_it.validation.walkforward import FOLDS

    t0 = time.time()
    upd = json.loads(a.set)
    E = G.Engine(a.cache)
    wins = G.fold_windows(E, [FOLDS[k] for k in G.EVO_FOLDS])
    f_last = FOLDS[G.EVO_FOLDS[-1]]
    seal = E.row(f_last.test_hi)
    E.LIM = np.minimum(np.asarray(E.LIM), seal).astype(np.int64)
    E.end_row = seal - 1
    tests = [te for _, te in wins]
    tests[-1] = G.Win(E, f_last.test_lo, f_last.test_hi, f_last.test_hi)
    out = os.path.join(a.out, G.EVO_SUB)
    head = G.run_provenance(a.cache).get("git_head", "")
    for f in sorted(glob.glob(os.path.join(out, f"{a.base}_s*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        off = f.rsplit("_s", 1)[1].split(".")[0]
        cfgs = [dict(c, **upd) for c in d["metrics"]["cfgs"]]
        G.MENU_OVERRIDE = {"ablation_of": a.base, "set": upd, "menu": d.get("menu")}
        G._evolve_write(E, cfgs, tests, seal, out, a, f"ABL_{a.tag}_s{off}", d["seeds"], head, t0)


if __name__ == "__main__":
    main()
