"""research/experiments/exits_v2_run — 칸 번호 ↔ 유전자, 대표·현 칸, 셔플 순열 = ga2_run.set_noise, 이웃·주효과."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "experiments"))

import exits_v2_run as xr  # noqa: E402


def test_grid_is_design_grid_and_cells_roundtrip():
    assert xr.N_CELLS == 5 * 5 * 2 * 4 * 4 * 2 * 3 * 3 == 14_400
    for k in (0, 1, 777, 9_999, xr.N_CELLS - 1):
        assert xr.cell_index(xr.cell_genes(k)) == k
    rep, cur = xr.cell_genes(xr.cell_index(xr.REP)), xr.cell_genes(xr.cell_index(xr.CUR))
    assert rep == xr.REP and cur == xr.CUR


def test_cell_params_current_equals_live_default_exits():
    from research.signals import exits_v2_fast as xv
    from swing_it.live.config import DEFAULT_STRATEGY as D

    p = xr.cell_params(xr.CUR)
    assert p["be_r"] == math.inf and p["take_frac"] == 0.5 and np.isnan(p["ss_x"]) and np.isnan(p["ss_y"])
    assert p["fb"] and p["fb_tol"] == 0.0 and p["fb_days"] == 1 and p["ts_days"] == 0
    assert (p["stop_cap"], p["take_r"], p["max_hold"], p["stop_floor"], p["stop_mult"]) == (
        D.stop_cap, D.take_r, D.max_hold, D.stop_floor, D.stop_mult)
    assert xr.CUR["trail_ma"] == D.trail_ma and D.failed_breakout and D.time_stop_days is None
    for k, v in xv.EXIT_V2_DEFAULT.items():                 # 현 칸 = v2 커널 기본값
        assert (np.isnan(v) and np.isnan(p[k])) if isinstance(v, float) and np.isnan(v) else p[k] == v, k
    q = xr.cell_params(xr.REP)
    assert (q["be_r"], q["ss_x"], q["ss_y"], q["fb_tol"], q["fb_days"], q["stop_cap"]) == (1.0, 0.15, 0.30, 0.02, 2, 0.08)
    off = xr.cell_params({**xr.REP, "fb_tol": None})
    assert off["fb"] is False


def test_world_perm_equals_ga2_set_noise():
    rng = np.random.default_rng(3)
    month = np.sort(rng.integers(0, 12, 500))
    for rep in (0, 7):
        perm = xr.world_perm(month, rep)
        # ga2_run.set_noise 와 같은 식(시드 1000 + rep)
        r2 = np.random.default_rng(1000 + rep)
        order = np.argsort(month, kind="stable")
        b = np.searchsorted(month[order], np.arange(month.max() + 2))
        want = np.arange(len(month))
        for j in range(len(b) - 1):
            idx = order[b[j]:b[j + 1]]
            want[idx] = r2.permutation(idx)
        assert (perm == want).all()
        assert (month[perm] == month).all()                 # 같은 달 안에서만


def test_neighbors_and_landscape_shapes():
    nb = xr.neighbors(0)
    assert len(nb) == sum(s - 1 for s in xr.SHAPE) and len(set(nb)) == len(nb) and 0 not in nb
    res = np.zeros((xr.N_CELLS, len(xr.METRICS)))
    res[:, xr.MI["win"]] = np.arange(xr.N_CELLS)
    ls = xr.landscape(res, "win")
    assert set(ls) == set(xr.GENES) and all(len(ls[g]) == len(xr.LEVELS[g]) for g in xr.GENES)
    # 첫 축(be_r) 단계 중앙값은 단조 증가(칸 번호 = C 순서)
    meds = [m for m, _, _ in ls["be_r"]]
    assert meds == sorted(meds)
