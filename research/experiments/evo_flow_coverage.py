"""9세대 수급 피복률 점검(검증 11차 MUST-FIX 10) — 캐시 사건 중 fl_smart20 이 있는 비율(연도별), 2020~2023 피복 < 90% 면 '자료 불완전'.

사용: ``PATTERN_GA_VARIANT=fwm uv run --with numba python research/experiments/evo_flow_coverage.py <캐시>``
"""
import json
import os
import sys

sys.path[:0] = [".", "src"]
import numpy as np  # noqa: E402

import research.experiments.pattern_ga_run as G  # noqa: E402
from research.signals import pattern_events as PE  # noqa: E402


def main() -> None:
    E = G.Engine(sys.argv[1])
    F, I_ = np.asarray(E.F_orig), np.asarray(E.I)
    yr = E.days[I_[:, PE.II["d"]]].year
    ok = np.isfinite(F[:, PE.FI["fl_smart20"]])
    cov = {int(y): round(float(ok[yr == y].mean()), 3) for y in range(2017, 2027) if (yr == y).any()}
    design = float(ok[(yr >= 2020) & (yr <= 2023)].mean())
    meta = json.load(open(os.path.join(sys.argv[1], "meta.json"), encoding="utf-8")).get("flows")
    print(json.dumps({"coverage_by_year": cov, "design_2020_2023": round(design, 3), "complete": design >= 0.90, "cache_flows_meta": meta},
                     ensure_ascii=False))
    print("수급 피복 2020~2023 = %.1f%% → %s" % (design * 100, "판정 가능" if design >= 0.90 else "자료 불완전 — G9A·G9C 판정 보류"))


if __name__ == "__main__":
    main()
