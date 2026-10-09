"""research/experiments/minervini_exits_run 의 순수 헬퍼 — 합성 데이터(DB 없음).

- 칸 7개가 사전등록 민감도(시도 1 + 6) 그대로이고, 한 칸이 정확히 한 값만 바꾼다.
- 연패·이익/손실·하위 10%·연도 양수·합격 바 산술.
- 손절 없음 비교가 공통 키에서만 나란히 놓인다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.minervini_exits_run import (
    CELLS,
    PRIMARY,
    bar_numbers,
    cell_config,
    gain_loss,
    longest_losing_streak,
    nostop_compare,
    rules_key,
    sim_kwargs,
    worst_decile_mean,
    years_positive,
)
from swing_it.diagnostics.trials import config_fingerprint


def test_cells_are_prereg_protocol_plus_six_single_changes():
    assert len(CELLS) == 7
    base = cell_config(PRIMARY, CELLS[PRIMARY])
    assert base["gate"] == "pead≥0.70" and base["e_mode"] == "pead_rank" and base["pead_rank_min"] == 0.70
    assert base["vcp"] and base["trail"] == "ma50" and base["fixed_stop"] is None
    assert sim_kwargs(base) == {"stop_floor": 0.03, "stop_cap": 0.10, "stop_mult": 0.99, "fixed_stop": None,
                                "take_r": 2.0, "time_stop_days": 20, "max_hold": 120, "chase": 0.05,
                                "limit_up_skip": True}
    assert base["breadth_min"] == 0.5 and base["rs_min"] == 70.0 and base["require_breakout"]
    assert "stop" not in base and "ma50_exit" not in base          # 스윙 3호 청산 값은 빠졌다
    fps = {config_fingerprint(cell_config(n, o)) for n, o in CELLS.items()}
    assert len(fps) == 7
    ignore = {"cell", "e_mode", "pead_rank_min", "e_min_yoy", "e_max_age_days"}
    for name, over in CELLS.items():
        if name == PRIMARY:
            continue
        c = cell_config(name, over)
        diff = {k for k in set(c) | set(base) if c.get(k) != base.get(k)} - ignore
        assert diff == set(over), name
    e = cell_config("실적 문 E", CELLS["실적 문 E"])
    assert e["e_mode"] == "e_flag" and e["e_min_yoy"] == 0.25 and e["e_max_age_days"] == 140 and e["vcp"]
    # 대조 풀 키: 주·VCP 끔·E 는 같은 시뮬 규칙 → 같은 풀
    keys = {n: rules_key(cell_config(n, o)) for n, o in CELLS.items()}
    assert keys[PRIMARY] == keys["VCP 끔"] == keys["실적 문 E"]
    assert len(set(keys.values())) == 5


def test_longest_losing_streak():
    assert longest_losing_streak([]) == 0
    assert longest_losing_streak([0.1, -0.1, -0.2, 0.0, -0.1, -0.1, -0.1, 0.2]) == 3
    assert longest_losing_streak([-1, -1]) == 2


def test_gain_loss_ratio_and_zero():
    g = gain_loss([0.10, 0.20, -0.05, -0.05, 0.0, np.nan])
    assert g["win"] == pytest.approx(2 / 5)
    assert g["avg_gain"] == pytest.approx(0.15) and g["avg_loss"] == pytest.approx(-0.05)
    assert g["gl_ratio"] == pytest.approx(3.0) and g["n_zero"] == 1
    assert np.isnan(gain_loss([0.1, 0.2])["gl_ratio"])


def test_worst_decile_and_years_positive():
    x = np.arange(20, dtype=float)
    assert worst_decile_mean(x) == pytest.approx(0.5)                # 하위 2개
    assert worst_decile_mean([3.0]) == 3.0 and np.isnan(worst_decile_mean([]))
    days = pd.Series(pd.to_datetime(["2020-01-02", "2020-05-01", "2021-03-01", "2022-03-01"]))
    vals = pd.Series([0.1, -0.05, -0.01, 0.02])
    assert years_positive(vals, days) == (2, 3)


def _stats(**kw):
    s = {"n": 150, "mom_lo": 0.01, "net_mean": 0.02, "net2_mean": 0.01, "gl_ratio": 2.0, "r_net": 0.1,
         "drop_top1": 0.01, "mom_h1_alpha": 0.01, "mom_h2_alpha": 0.02, "years_pos": 5, "years_n": 9,
         "dsr_prob": 0.6}
    s.update(kw)
    return s


def test_bar_numbers_arithmetic():
    b = bar_numbers(_stats())
    assert not b["void"] and b["n_met"] == 6
    assert bar_numbers(_stats(n=99))["void"]
    assert not bar_numbers(_stats(mom_lo=0.0))["b1"]
    assert not bar_numbers(_stats(net2_mean=-0.001))["b2"]
    assert not bar_numbers(_stats(gl_ratio=1.99))["b3"] and not bar_numbers(_stats(r_net=-0.01))["b3"]
    assert not bar_numbers(_stats(gl_ratio=np.nan))["b3"]
    assert not bar_numbers(_stats(years_pos=4, years_n=8))["b5"]            # 과반 = 절반 초과
    assert not bar_numbers(_stats(mom_h1_alpha=np.nan))["b5"]
    assert not bar_numbers(_stats(dsr_prob=0.5))["b6"]
    assert bar_numbers({"n": 0})["n_met"] == 0


def test_nostop_compare_uses_common_keys():
    idx = pd.Index(["a", "b", "c"])
    day = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    a = pd.DataFrame({"net": [0.1, -0.05, 0.02], "entry_day": day, "code": ["x", "y", "z"], "hold_days": [5, 3, 20]},
                     index=idx)
    b = pd.DataFrame({"net": [0.3, -0.2, 9.9], "entry_day": [day[0], day[1], day[2]], "code": ["x", "y", "w"],
                      "hold_days": [120, 120, 120]}, index=pd.Index(["a", "b", "d"]))
    r = nostop_compare(a, b)
    assert r["n_common"] == 2 and r["n_a"] == 3 and r["n_b"] == 3
    assert r["a"]["mean"] == pytest.approx(0.025) and r["b"]["mean"] == pytest.approx(0.05)
    assert r["b"]["min"] == pytest.approx(-0.2) and r["b"]["streak"] == 1
