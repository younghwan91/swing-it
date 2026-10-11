"""13세대 청산 규칙 어블레이션(META_EVOLUTION.md 13a — 하이쿠 8 에이전트 차트 분석 exitlab 후보).

규칙 후보는 진화 후보(G7A·G11T·G12A~C)의 고유 거래 127건 차트에서 나왔다 → 그 거래로 재는 것은 순환이다. 그래서 **본 증거**는
같은 메뉴 무작위 설정(``random_x`` → ``decode``)의 폴드 1~4 시험창 거래에서 열람 거래(code, order_day)를 뺀 것이다.

설정 하나 = 네 폴드에 같은 설정(이어 붙인 시험창 2020~2023, 봉인 2024-01). 체결 ≥ ``--min-n`` 인 설정만. 규칙마다 같은 설정에 규칙
유전자를 덮어써 다시 돌리고, 설정 단위 짝 차이(거래당 슬롯 대비 순 2배 비용·연환산)의 평균·SE·t 를 낸다(거래 단위가 아니라 설정 단위 —
설정 안 거래가 같은 날 묶여 독립이 아니다).

사용: ``PATTERN_GA_VARIANT=fwm PATTERN_GA_MENU='<G7A 메뉴>' uv run --with numba --extra pg python research/experiments/exit_ablate.py
--cache <v11> --n 400 --out <json>`` (``--fam P`` 면 P 가족 무작위 설정 — W 끔)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path[:0] = [".", "src"]

RULES = {
    "be1h": {"be_r": 1.0, "be_basis": "high"},
    "be1c": {"be_r": 1.0, "be_basis": "close"},
    "lock05": {"be_r": 1.0, "be_basis": "high", "lock_to": 0.5},
    "take1": {"take_r": 1.0, "take_frac": 1 / 3},
    "rev1c": {"rev_r": 1.0, "rev_next": 0},
    "rev15c": {"rev_r": 1.5, "rev_next": 0},
    "rev1n": {"rev_r": 1.0, "rev_next": 1},
    "plow15": {"plow_r": 1.5},
    "ccut05": {"close_cut": 0.5},
    "ccut075": {"close_cut": 0.75},
}
VIEWED = "/tmp/claude-1000/-home-young-git-swing-it-feature-combined-strategy/c0fa40b6-d4a8-41de-bd4c-b160264df3d9/scratchpad/exitlab/facts.json"
SEED = 20261012
_W: dict = {}


def _init(cache: str) -> None:
    import numpy as np

    import research.experiments.pattern_ga_run as G
    from swing_it.validation.walkforward import FOLDS
    E = G.Engine(cache)
    wins = G.fold_windows(E, [FOLDS[k] for k in G.EVO_FOLDS])
    f_last = FOLDS[G.EVO_FOLDS[-1]]
    seal = E.row(f_last.test_hi)
    E.LIM = np.minimum(np.asarray(E.LIM), seal).astype(np.int64)
    E.end_row = seal - 1
    tests = [te for _, te in wins]
    tests[-1] = G.Win(E, f_last.test_lo, f_last.test_hi, f_last.test_hi)
    viewed = {(f["code"], f["entry_day"]) for f in json.load(open(VIEWED, encoding="utf-8"))}
    _W.update(G=G, E=E, tests=tests, viewed=viewed)


def _stats(cfg: dict) -> dict:
    import numpy as np
    G, E, tests = _W["G"], _W["E"], _W["tests"]
    st = E.stitched([cfg] * len(tests), tests)
    t = G.trade_frame(E, st, [cfg] * len(tests))
    if not len(t):
        return {"n": 0, "n_u": 0, "ann": st["ann"]}
    amb = t["amb"].astype(bool)
    n2 = t["net2"].where(~amb, E.pmix * t["net2_stop0"] + (1 - E.pmix) * t["net2"])
    n2s = n2 * np.minimum(1.0, t["alloc"] / 0.10)
    keys = list(zip(t["code"], t["entry_day"].dt.strftime("%Y-%m-%d"), strict=True))
    unv = np.array([k not in _W["viewed"] for k in keys])
    win = n2s > 0
    return {"n": int(len(t)), "n_u": int(unv.sum()), "ann": float(st["ann"]), "m": float(n2s.mean()),
            "m_u": float(n2s[unv].mean()) if unv.any() else float("nan"),
            "wsum_u": float(n2s[unv & win].sum()), "lsum_u": float(n2s[unv & ~win].sum())}


def _task(a: tuple) -> dict:
    i, cfg = a
    out = {"i": i, "base": _stats(cfg)}
    if out["base"]["n"] < _W.get("min_n", 5):
        return out
    for k, upd in RULES.items():
        out[k] = _stats({**cfg, **upd})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--fam", default="W", choices=("W", "P"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    import research.experiments.pattern_ga_run as G
    t0 = time.time()
    X = G.random_x(np.random.default_rng([SEED, 0 if a.fam == "W" else 1]), a.n)
    cfgs = [G.decode(x) for x in X]
    tasks = list(enumerate(cfgs))
    res = []
    with cf.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache,)) as ex:
        for r in ex.map(_task, tasks, chunksize=4):
            res.append(r)
    ok = [r for r in res if r["base"]["n"] >= a.min_n and len(r) > 2]
    summ = {}
    for k in RULES:
        rows = []
        for r in ok:
            b, x = r["base"], r[k]
            if b["n_u"] and x.get("n_u"):
                rows.append((x["m_u"] - b["m_u"], x["ann"] - b["ann"], x["wsum_u"] - b["wsum_u"], x["lsum_u"] - b["lsum_u"]))
        d = np.array(rows) if rows else np.zeros((0, 4))
        def ms(v):
            return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))
        if len(d):
            (m1, s1), (m2, s2) = ms(d[:, 0]), ms(d[:, 1])
            summ[k] = {"configs": len(d), "trade_slot_diff_pct": m1 * 100, "t_trade": m1 / s1 if s1 else float("nan"),
                       "ann_diff_pct": m2 * 100, "t_ann": m2 / s2 if s2 else float("nan"),
                       "frac_better_trade": float((d[:, 0] > 0).mean()),
                       "win_sum_diff_pct": float(d[:, 2].mean() * 100), "loss_sum_diff_pct": float(d[:, 3].mean() * 100)}
    json.dump({"args": vars(a), "rules": RULES, "n_configs": len(ok), "summary": summ, "elapsed": time.time() - t0},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import pandas as pd
    print(f"설정 {len(ok)}/{len(res)} (체결 ≥ {a.min_n})")
    print(pd.DataFrame(summ).T.round(3).to_string())


if __name__ == "__main__":
    main()
