"""research/experiments/convergence_run 의 순수 헬퍼 — 합성 데이터(DB 없음).

- 상대 수렴 rel: 분자 창(t−10..t−1)·분모 창(t−260..t−11, 250개, 최소 120)·시점 안전.
- 팔 5개(A·B·C·D·R)가 사전등록 정의대로 한 축씩만 다르다.
- 팔 진입 마스크(NaN rel·F 는 통과 안 함), 손절 비율, 합격 바 1~5 + VOID 산술(문턱 경계 포함).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.convergence_run import (
    ARMS,
    PRIMARY,
    arm_config,
    arm_mask,
    convergence_bars,
    per_year_counts,
    relative_contraction,
    stop_rate,
)
from research.experiments.minervini_exits_run import sim_kwargs
from swing_it.diagnostics.trials import config_fingerprint


def _hl(n: int, seed: int = 0) -> "tuple[pd.DataFrame, pd.DataFrame]":
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2015-01-01", periods=n)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    w = rng.uniform(0.01, 0.06, n)
    return pd.DataFrame({"A": mid * (1 + w)}, index=d), pd.DataFrame({"A": mid * (1 - w)}, index=d)


def test_rel_numerator_and_denominator_windows_match_prereg_text():
    high, low = _hl(400)
    rng, rel = relative_contraction(high, low)
    h, lo = high["A"].to_numpy(), low["A"].to_numpy()
    t = 350
    num = h[t - 10:t].max() / lo[t - 10:t].min() - 1.0               # t−10..t−1 (t 제외)
    assert rng["A"].iloc[t] == pytest.approx(num)
    r = rng["A"].to_numpy()
    den = np.median(r[t - 260:t - 10])                                 # s ∈ t−260..t−11 → 250개
    assert len(r[t - 260:t - 10]) == 250
    assert rel["A"].iloc[t] == pytest.approx(num / den)


def test_rel_min_periods_and_point_in_time():
    high, low = _hl(400, seed=1)
    _, rel = relative_contraction(high, low)
    # rng 첫 유효 행 = 10 (t−10..t−1 이 꽉 참). 분모 유효 개수 ≥ 120 → s ≤ t−11 에 rng 120개: t ≥ 10 + 119 + 11 = 140
    assert rel["A"].iloc[:140].isna().all() and np.isfinite(rel["A"].iloc[140])
    # 미래(t 이후)·당일 고저를 바꿔도 t 값은 같다
    t = 300
    h2, l2 = high.copy(), low.copy()
    h2.iloc[t:] *= 3.0
    l2.iloc[t:] *= 0.3
    _, rel2 = relative_contraction(h2, l2)
    pd.testing.assert_series_equal(rel["A"].iloc[:t + 1], rel2["A"].iloc[:t + 1])


def test_rel_flat_history_is_one_and_tight_window_below_one():
    d = pd.bdate_range("2015-01-01", periods=300)
    high = pd.DataFrame({"A": 110.0}, index=d)
    low = pd.DataFrame({"A": 100.0}, index=d)
    high.iloc[-11:] = 104.0                                            # 마지막 11일 폭 4% (평소 10%)
    _, rel = relative_contraction(high, low)
    assert rel["A"].iloc[-11] == pytest.approx(1.0)
    assert rel["A"].iloc[-1] == pytest.approx(0.04 / 0.10, rel=1e-9)  # 마지막 행: 창 t−10..t−1 = 바뀐 날만


def test_arms_are_prereg_definitions():
    assert list(ARMS) == ["A", "B", "C", "D", "R"] and PRIMARY == "A"
    cfg = {n: arm_config(n) for n in ARMS}
    a = cfg["A"]
    assert a["event_lo"] == "2024-02-19" and a["event_hi"] == "2026-04-01"
    assert a["rel_max"] == 1.0 and a["flow_F"] and a["pead_rank_min"] == 0.70 and a["e_mode"] == "pead_rank"
    assert not a["vcp"] and a["trail"] == "ma50" and a["fixed_stop"] is None
    assert sim_kwargs(a) == {"stop_floor": 0.03, "stop_cap": 0.10, "stop_mult": 0.99, "fixed_stop": None,
                             "take_r": 2.0, "time_stop_days": 20, "max_hold": 120, "chase": 0.05,
                             "limit_up_skip": True}
    assert a["rs_min"] == 70.0 and a["breadth_min"] == 0.5 and a["require_breakout"]
    assert "5,000 ≤ ADV20 < 20,000" in a["universe"] and "ADV20≥200억" in cfg["R"]["universe"]
    assert len({config_fingerprint(c) for c in cfg.values()}) == 5
    ignore = {"cell", "arm", "matched"}
    expect = {"B": {"flow_F", "event_lo"}, "C": {"rel_max"}, "D": {"rel_max"}, "R": {"universe"}}
    for n, keys in expect.items():
        diff = {k for k in set(cfg[n]) | set(a) if cfg[n].get(k) != a.get(k)} - ignore
        assert diff == keys, n
    assert cfg["B"]["event_lo"] == "2017-09-15" and cfg["B"]["flow_F"] is None
    assert cfg["C"]["rel_max"] == 1.33 and cfg["D"]["rel_max"] is None
    # 디버그 구간 축소: 팔 하한보다 늦을 때만 바뀐다
    assert arm_config("A", "2020-01-01", "2025-01-31")["event_lo"] == "2024-02-19"
    assert arm_config("B", "2020-01-01", "2025-01-31")["event_lo"] == "2020-01-01"


def test_arm_mask_nan_rel_and_flow_do_not_pass():
    ev = pd.DataFrame({
        "event_day": pd.to_datetime(["2024-03-01"] * 6 + ["2023-03-01"]),
        "rel": [0.8, 1.0, 1.2, np.nan, 0.5, 0.9, 0.5],
        "F": [1.0, 1.0, 1.0, 1.0, np.nan, 0.0, 1.0],
        "pead_ok": [True] * 7,
    })
    a = arm_mask(ev, ARMS["A"], "2024-02-19", "2026-04-01")
    assert a.tolist() == [True, True, False, False, False, False, False]
    c = arm_mask(ev, ARMS["C"], "2024-02-19", "2026-04-01")
    assert c.tolist() == [True, True, True, False, False, False, False]
    d = arm_mask(ev, ARMS["D"], "2024-02-19", "2026-04-01")
    assert d.tolist() == [True, True, True, True, False, False, False]      # rel 무시(NaN 포함)
    b = arm_mask(ev, ARMS["B"], "2017-09-15", "2026-04-01")
    assert b.tolist() == [True, True, False, False, True, True, True]       # F 무시
    ev2 = ev.assign(pead_ok=[False] * 7)
    assert not arm_mask(ev2, ARMS["B"], "2017-09-15", "2026-04-01").any()


def test_stop_rate_counts_only_stop():
    assert stop_rate(pd.Series(["stop", "breakeven", "trail", "time_stop"])) == 0.25
    assert np.isnan(stop_rate(pd.Series([], dtype=object)))


def _s(**kw):
    s = {"n": 50, "mom_lo": 0.001, "net2_mean": 0.01, "drop_top1": 0.005, "gl_ratio": 2.0, "stop_rate": 0.49,
         "worst_decile": -0.11, "months_pos": 6, "months2_pos": 5, "months_n": 10}
    s.update(kw)
    return s


def test_bars_thresholds_and_edges():
    b = convergence_bars(_s())
    assert not b["void"] and b["b1"] and b["b2"] and b["b3"] and b["b4"] and b["b5"] and not b["b5_2x"]
    assert b["n_met"] == 5
    assert convergence_bars(_s(n=39))["void"] and not convergence_bars(_s(n=40))["void"]
    assert not convergence_bars(_s(mom_lo=0.0))["b1"] and not convergence_bars(_s(mom_lo=np.nan))["b1"]
    assert not convergence_bars(_s(net2_mean=-0.001))["b2"] and not convergence_bars(_s(drop_top1=0.0))["b2"]
    assert not convergence_bars(_s(stop_rate=0.5))["b3"] and not convergence_bars(_s(gl_ratio=1.99))["b3"]
    assert not convergence_bars(_s(worst_decile=-0.1101))["b4"]
    assert not convergence_bars(_s(months_pos=5))["b5"]                     # 정확히 50% 는 미달
    assert convergence_bars({"n": 0})["n_met"] == 0 and convergence_bars({"n": 0})["void"]


def test_per_year_counts():
    d = pd.Series(pd.to_datetime(["2024-03-01", "2024-05-01", "2025-01-02"]))
    assert per_year_counts(d) == {2024: 2, 2025: 1}
