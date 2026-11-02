"""``pead_b_golden.json`` 을 만든 스크립트 — 기록용이다. **다시 만들지 않는다.**

단계 B 하네스를 ``src/swing_it/book/`` 로 옮기기 **전** 코드(``research/experiments/pead_optimizer_b.py`` 안의 구현)로 합성 데이터를
돌린 값이다. 옮긴 뒤 코드로 다시 만들면 그 코드끼리 비교하게 되어 아무것도 증명하지 못한다.

    python tests/golden/make_pead_b_golden.py <옮기기 전 트리> <출력>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STEP_FIELDS = ("k", "t", "date", "gross", "cost1", "bench", "turnover", "exante_20", "factor_share", "status", "lam",
               "lam_hit", "est_cost", "n_names", "n_forced", "n_delist", "n_frozen", "n_untradeable", "u_excl",
               "perm_nan_share", "f", "f_next", "lam_cause", "n_tranches", "cash", "min_w", "max_w", "max_w_frozen",
               "sector_dev", "sector_tol", "turn_bound", "n_dust", "dust_mass", "delist_mass", "n_buy_limit_up")


def _clean(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return None if v != v else v
    return v


def steps(recs, *, contrib: bool = False) -> list[dict]:
    out = []
    for r in recs:
        row = {k: _clean(getattr(r, k)) for k in STEP_FIELDS}
        row["alarm_top3"] = [[str(n), float(e)] for n, e in r.alarm_top3]
        if contrib:
            row["contrib"] = [[int(i), float(w), float(g), float(c)] for i, w, g, c in r.contrib]
        out.append(row)
    return out


def main() -> int:
    tree, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(tree / "research" / "experiments"))
    sys.path.insert(0, str(tree / "tests"))
    import pead_optimizer_b as B
    import pead_optimizer_b_report as BR
    from test_pead_optimizer_b import _ctx

    out: dict = {"made_from": "옮기기 전 pead_optimizer_b.py — 다시 만들지 않는다", "seeds": {}}
    ic = {y: 0.05 for y in range(2017, 2026)}
    for seed in (0, 3):
        ctx = _ctx(seed)
        s: dict = {}
        s["r"] = steps(B.simulate(ctx, "R"), contrib=seed == 0)
        s["r_no_costs"] = steps(B.simulate(ctx, "R", flags=B.Flags(costs=False)), contrib=seed == 0)
        sigma = B.r_sigma_series(ctx)
        recs_o = B.simulate(ctx, "O", ic_by_year=ic, sigma_r=sigma)
        recs_r = B.simulate(ctx, "R")
        s["o"] = steps(recs_o, contrib=seed == 0)
        s["sigma"] = {str(k): _clean(v) for k, v in sigma.items()}
        s["summary_r"] = json.loads(json.dumps(B.book_summary(recs_r), default=float))
        s["summary_o"] = json.loads(json.dumps(B.book_summary(recs_o), default=float))
        s["invariants_r"] = B.check_invariants(recs_r, "R")
        s["invariants_o"] = B.check_invariants(recs_o, "O")
        s["parity_lag0"] = json.loads(B.parity_book(ctx, 0).to_json(orient="records"))
        s["parity_lag1"] = json.loads(B.parity_book(ctx, 1).to_json(orient="records"))
        s["parity_check"] = json.loads(json.dumps(B.parity_check(ctx), default=float))
        s["full_parity"] = json.loads(json.dumps(B.full_parity(ctx), default=float))
        s["toggles"] = json.loads(json.dumps(B.toggle_report(ctx), default=float))
        days, ics = B.daily_rank_ic(ctx.C, ctx.adv, ctx.cap, ctx.sig, 40, ctx.C.shape[1] - 1, 20)
        s["daily_ic_days"] = [int(d) for d in days]
        s["daily_ic"] = [float(v) for v in ics]
        s["fold_ic"] = B.fold_ic(days, ics, int(days[len(days) // 2]), 20) if len(days) else None
        nulls = B.permutation_nulls(ctx, B.PRIMARY, ic, sigma, workers=2, n=2)
        s["nulls"] = {"R": [_clean(v) for v in nulls["R"]], "O": [_clean(v) for v in nulls["O"]],
                      "nan": [_clean(v) for v in nulls["O_perm_nan_share"]], "violations": nulls["violations"]}
        a_o = B.oos(B.step_returns(recs_o, 2.0))["a"].to_numpy()
        a_r = B.oos(B.step_returns(recs_r, 2.0))["a"].to_numpy()
        s["boot"] = [float(v) for v in B.stationary_boot_dir(a_o, a_r, n=50)] if len(a_o) else []
        rng = np.random.default_rng(seed)
        y, x = rng.normal(0, 1, 60), rng.normal(0, 1, (60, 3))
        s["ols_hac"] = BR.ols_hac(y, x)
        s["nw_t_rows"] = [float(v) for v in BR.nw_t_rows(rng.normal(0, 1, (4, 30)))]
        s["keyed_rng"] = [float(v) for v in B.keyed_rng("golden", seed).random(3)]
        out["seeds"][str(seed)] = s
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=0, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
