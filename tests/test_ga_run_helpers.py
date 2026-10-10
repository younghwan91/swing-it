"""research/experiments/ga_run · ga2_run 의 순수 헬퍼 — 유전자 부호화/복호화·GA 연산자·워크포워드 효율 (DB 없음)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from research.experiments.ga2_run import (
    DISC_SIZES,
    G2_CONT,
    G2_DISC,
    GENE_ORDER,
    OFF_ZONE,
    flow_decode,
    g2_decode,
    g2_encode,
    g2_median,
    g2_spec,
    is_off,
    wfe,
)
from research.experiments.ga_run import (
    GA1_GENES,
    WINDOWS,
    blx,
    ga1_decode,
    ga1_from_unit,
    gene_spread,
    mutate,
    simple_ga,
    tournament,
)


def test_ga1_decode_clips_rounds_and_maps_pivot_steps():
    c = ga1_decode([39.0, 1.234567, 0.95, 0.5, 0.3, 2.6])
    assert c == {"rs_min": 40.0, "vol_mult": 1.2346, "pead_min": 0.9, "rel_max": 0.6, "breadth_min": 0.3,
                 "pivot_window": 50}
    assert ga1_decode([50, 1, 0, 1, 0, 0.49])["pivot_window"] == 20
    assert ga1_from_unit(np.array([0, 1, 0, 1, 0, 4.0])) == {"rs_min": 40.0, "vol_mult": 2.0, "pead_min": 0.0,
                                                            "rel_max": 3.0, "breadth_min": 0.0, "pivot_window": 60}


def test_ga_operators_stay_in_bounds_and_are_seeded():
    rng = np.random.default_rng(0)
    lo = np.array([g[1] for g in GA1_GENES])
    hi = np.array([g[2] for g in GA1_GENES])
    for _ in range(200):
        a, b = rng.uniform(lo, hi), rng.uniform(lo, hi)
        c = mutate(blx(a, b, rng, lo, hi), rng, lo, hi)
        assert np.all(c >= lo) and np.all(c <= hi)
    fit = np.array([0.0, 5.0, 1.0])
    picks = {tournament(fit, np.random.default_rng(s), k=3) for s in range(30)}
    assert 1 in picks
    # 같은 시드 = 같은 이력
    f = lambda v: -float(np.sum((v - lo) ** 2))  # noqa: E731
    h1, _ = simple_ga(f, GA1_GENES, seed=3, pop=10, gens=3)
    h2, _ = simple_ga(f, GA1_GENES, seed=3, pop=10, gens=3)
    assert [x[2] for x in h1] == [x[2] for x in h2] and len(h1) == 30


def test_ga_elitism_keeps_best():
    lo = np.array([g[1] for g in GA1_GENES])
    f = lambda v: -float(np.sum((v - lo) ** 2))  # noqa: E731
    _, gens = simple_ga(f, GA1_GENES, seed=1, pop=12, gens=6)
    best = [g[1].max() for g in gens]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))


def test_g2_roundtrip_and_off_zones():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = np.r_[rng.uniform(0, 1, len(G2_CONT)), [rng.integers(0, k) for k in DISC_SIZES]]
        g = g2_decode(x)
        assert set(g) == set(GENE_ORDER)
        g2 = g2_decode(g2_encode(g))
        for k in GENE_ORDER:
            if g[k] is None:
                assert g2[k] is None
            elif isinstance(g[k], float) and k in dict((n, 0) for n, *_ in G2_CONT):
                assert g2[k] == pytest.approx(g[k], abs=1e-4)
            else:
                assert g2[k] == g[k]
    x = np.r_[np.full(len(G2_CONT), 0.5), np.zeros(len(DISC_SIZES))]
    x[0] = OFF_ZONE * 0.5            # near_high 끔 끝(lo)
    x[6] = 1.0 - OFF_ZONE * 0.5      # rel_max 끔 끝(hi)
    g = g2_decode(x)
    assert g["near_high"] is None and g["rel_max"] is None and g["rs_min"] == pytest.approx(47.5)
    x[0], x[6] = OFF_ZONE * 1.5, 1.0 - OFF_ZONE * 1.5
    g = g2_decode(x)
    assert g["near_high"] is not None and g["rel_max"] is not None
    # 이산 끝 단계 = 없음/∞, 범위 밖 인덱스는 잘린다
    xd = np.r_[np.full(len(G2_CONT), 0.5), [k - 1 + 3 for k in DISC_SIZES]]
    g = g2_decode(xd)
    assert g["chase"] == math.inf and g["take_r"] == math.inf and g["trail_ma"] == 0 and g["failed_breakout"] == 1
    assert g["max_hold"] == 120 and g["pivot_window"] == 60
    assert is_off("chase", g["chase"]) and is_off("trail_ma", 0) and not is_off("max_hold", 120)


def test_g2_spec_vector_and_exit_key():
    g = {"near_high": None, "above_low": 1.3, "rs_min": 70.0, "vol_mult": None, "breadth_min": 0.5, "pead_min": None,
         "rel_max": 1.0, "ma_stack": 3, "ma200_slope_days": 20, "pivot_window": 0, "chase": 0.05, "stop_cap": 0.10,
         "take_r": math.inf, "trail_ma": 50, "failed_breakout": 1, "max_hold": 60,
         "eps_yoy_min": 0.25, "eps_accel": 1, "rev_yoy_min": None}
    s = g2_spec(g)
    assert np.isnan(s["g"][0]) and s["g"][1] == 1.3 and s["g"][2] == 3 and s["g"][3] == 2 and s["g"][5] == -1
    assert np.isnan(s["g"][6]) and np.isnan(s["g"][8]) and s["g"][9] == 1.0
    assert s["g"][10] == 0.25 and s["g"][11] == 1.0 and np.isnan(s["g"][12])
    assert s["ek"] == (0.10, math.inf, 50, 0, 60, 1, WINDOWS.index(50))     # 돌파 실패 피벗 = 돌파 없음이면 50일
    assert s["w_idx"] == WINDOWS.index(50) and s["c_idx"] == 1          # 돌파 없음 → 추격 피벗 50일
    s2 = g2_spec({**g, "pivot_window": 30, "chase": math.inf})
    assert s2["w_idx"] == WINDOWS.index(30) and s2["g"][5] == WINDOWS.index(30) and s2["c_idx"] == 3
    assert s2["ek"][5:] == (1, WINDOWS.index(30))
    assert g2_spec({**g, "failed_breakout": 0})["ek"][5:] == (0, -1)


def test_g2_median_continuous_median_and_discrete_middle_step():
    base = g2_decode(np.r_[np.full(len(G2_CONT), 0.5), np.zeros(len(DISC_SIZES))])
    a = {**base, "rs_min": 10.0, "pead_min": None, "take_r": 1.5}
    b = {**base, "rs_min": 50.0, "pead_min": None, "take_r": math.inf}
    c = {**base, "rs_min": 90.0, "pead_min": 0.5, "take_r": 3.0}
    m = g2_median([a, b, c])
    assert m["rs_min"] == pytest.approx(50.0) and m["pead_min"] is None and m["take_r"] == 3.0
    assert dict(G2_DISC)["take_r"].index(m["take_r"]) == 2


def test_wfe_and_flow_decode_and_spread():
    assert wfe([0.02, 0.04], [0.1, 0.1]) == pytest.approx(0.3)
    assert np.isnan(wfe([0.1], [-0.2]))
    fl = flow_decode([0.0, 1.0, 3.0])
    assert fl == {"smart_min": -0.1, "foreign_pos": 1, "persist_min": 12}
    sp = gene_spread([{"a": 1.0}, {"a": 3.0}], [("a", 0.0, 4.0)])
    assert sp["a"] == (1.0, 3.0, 0.5)


def test_cma_bipop_budget_and_restarts():
    pytest.importorskip("cmaes")
    from research.experiments.ga_run import cma_bipop

    def f(x):  # 최대 = 연속 0.7, 이산 2
        return -float((x[0] - 0.7) ** 2 + (x[1] - 2) ** 2)

    hist, rst = cma_bipop(f, 1, (5,), budget=400, seed=0)
    assert len(hist) == 400 and sum(r["evals"] for r in rst) == 400
    best = max(hist, key=lambda h: h[2])[1]
    assert abs(best[0] - 0.7) < 0.1 and best[1] == 2.0
    assert all(float(h[1][1]).is_integer() for h in hist)          # 이산 축은 단계 값으로 평가된다


def test_cma_bipop_stall_restart_fires_on_flat_landscape():
    pytest.importorskip("cmaes")
    from research.experiments.ga_run import cma_bipop

    import itertools

    cnt = itertools.count()
    hist, rst = cma_bipop(lambda x: -next(cnt) * 1e-3, 2, (4,), budget=1600, seed=1, stall=True)   # 결코 나아지지 않음
    assert len(rst) >= 2 and rst[0]["why"] == "stall" and {"first", "small", "large"} >= {r["regime"] for r in rst}
    hist2, rst2 = cma_bipop(lambda x: 0.0, 2, (4,), budget=300, seed=1, stall=False)
    assert all(r["why"] in ("budget", "should_stop") for r in rst2)
