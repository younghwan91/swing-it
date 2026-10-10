"""research/experiments/meta_gen1_run 의 순수 헬퍼 — 절차 목록·점수·문턱·슬롯·판단 규칙 산술·페이퍼 추천 규칙(DB 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.meta_gen1_run import (
    GATE_Q,
    SCALE_GRID,
    decide,
    first_argmax,
    gate_open,
    gate_thresholds,
    month_ends,
    monthly_compound,
    n_params,
    null_pct,
    paper_rule,
    procedures,
    scaler_slots,
    simplest,
    static_slots,
    tier2_core,
    window_stats,
    zscore_mean,
)


def test_procedures_are_the_design_table():
    ps = procedures()
    assert len(ps) == 18 and len({p["name"] for p in ps}) == 18
    assert [p["name"] for p in ps if p["primary"]] == ["P1", "P1h", "P2", "P1·G[RA]", "P1h·G[RA]", "P2·G[RA]", "P1·S[RA]", "P2·S[RA]"]
    assert not any(p["mode"] == "S" and p["fam"] == "P1h" for p in ps)          # 표에 P1h·S 는 없다
    by = {p["name"]: p for p in ps}
    assert n_params(by["P2"]) == 9 and n_params(by["P1·G[RA]"]) == 20 and n_params(by["P2·S[RA]"]) == 11
    assert len(GATE_Q) == 4 and len(SCALE_GRID) == 9


def test_zscore_mean_uses_train_rows_and_skips_missing_features():
    raw = np.array([[1.0, np.nan], [3.0, 10.0], [5.0, 20.0], [7.0, 30.0]])
    s = zscore_mean(raw, np.array([1, 2]))
    # 열 0: 학습 평균 4 · 표준편차 1 / 열 1: 15 · 5
    np.testing.assert_allclose(s, [-3.0, -1.0, 1.0, (3.0 + 3.0) / 2])
    const = zscore_mean(np.array([[1.0, 2.0], [1.0, 4.0]]), np.array([0, 1]))      # 표준편차 0 인 열은 뺀다
    np.testing.assert_allclose(const, [-1.0, 1.0])


def test_gate_thresholds_open_and_scaler_slots():
    s = np.arange(10, dtype=float)
    thr = gate_thresholds(s, np.arange(10))
    assert thr[0] == -np.inf and thr[1] == pytest.approx(np.percentile(s, 20))
    assert gate_thresholds(np.full(5, np.nan), np.arange(5)) == [-np.inf] * 4
    op = gate_open(np.array([1.0, 2.0, np.nan]), 1.5)
    assert op.tolist() == [False, True, True]                                     # NaN 은 열림
    np.testing.assert_array_equal(scaler_slots(np.array([0.0, 1.0, -5.0, np.nan, 0.05]), 0.5, 0.5),
                                  [5, 10, 0, 10, 5])                              # 0.525 × 10 = 5.25 → 5
    assert scaler_slots(np.array([0.0]), 0.25, 0.25)[0] == 3                      # 2.5 → 0.5 올림 3


def test_month_ends_argmax_monthly_static_and_pct():
    mid = np.array([0, 0, 1, 1, 1, 2])
    assert month_ends(mid).tolist() == [1, 4]
    assert first_argmax([0.1, 0.3, 0.3, np.nan]) == 1
    np.testing.assert_allclose(monthly_compound(np.array([0.1, 0.1, -0.5]), np.array([3, 3, 4])), [0.21, -0.5])
    assert static_slots(0.4, 0.8) == 5 and static_slots(0.01, 0.8) == 1 and static_slots(0.9, 0.8) == 10
    assert null_pct(2.0, np.array([1.0, 2.0, 3.0, np.nan])) == pytest.approx(0.5)


def test_window_stats_splits():
    dates = pd.bdate_range("2020-01-01", "2026-03-31")
    r = np.where(dates.year == 2025, 0.002, -0.0001)
    s = window_stats(r, dates)
    assert s["cagr_ex25"] < 0 < s["cagr"] and s["cagr_2024"] < 0 and s["cagr_2124"] < 0


def _row(**kw):
    base = {"name": "X", "fam": "P1", "mode": "base", "score": None, "primary": True, "n_params": 19, "alpha_t": 2.5,
            "cagr": 0.10, "calmar": 0.8, "mdd": -0.2, "null_cagr_p95": 0.05, "null_calmar_p95": 0.5, "bar_calmar": 0.1,
            "cagr_x2": 0.05, "cagr_ex25": 0.02, "cagr_2024": 0.01, "dcalmar": np.nan, "null_dcal_p95": np.nan,
            "rot_dcal_p95": np.nan, "p_static": np.nan, "p_expo": 0.9, "null_pass": 0.1, "null_calmar_med": 0.2}
    return {**base, **kw}


def test_decide_two_tiers_and_regime_condition():
    d = decide([_row(), _row(name="Y", mdd=-0.3), _row(name="Z", fam="P2", n_params=9),
                _row(name="G", mode="G", score="RA", dcalmar=0.3, null_dcal_p95=0.2, rot_dcal_p95=0.4, p_static=0.9),
                _row(name="G2", mode="G", score="RA", dcalmar=0.5, null_dcal_p95=0.2, rot_dcal_p95=0.4, p_static=0.79)])
    x, y, z, g, g2 = d
    assert x["t1"] and x["t2"] and x["t2_c3"] is None
    assert not y["t2_c1"] and not y["t2"]
    assert z["t1"] is None and z["t2"]                                            # P2 가족 층 1 해당 없음
    assert not g["t2_c3a"] and not g["t2"]                                        # 순환 널 p95 못 넘음
    assert g2["t2_c3a"] and not g2["t2_c3b"] and not g2["t2"]                     # 노출 맞춤 P < 0.8
    assert tier2_core(0.1, -0.2, 0.5, 0.01, 0.01, -0.01, 0.1)["core"] is False    # 2020~24 CAGR ≤ 0


def test_paper_rule_and_simplest():
    rows = decide([_row(), _row(name="P2", fam="P2", n_params=9, null_pass=0.5, null_calmar_med=0.9),
                   _row(name="P1h", fam="P1h", n_params=24, cagr_2024=0.005), _row(name="R", primary=False, n_params=1)])
    out = []
    for d in rows:
        d.update(paper_rule(d, p1_cagr_2024=0.01))
        out.append(d)
    x, p2, h, r = out
    assert x["paper"] and not p2["r3"] and not p2["paper"]                         # 널 통과율 50% ∧ 널 중앙 위 아님
    assert h["r4"] is False and not h["paper"]                                     # P1h 는 2020~24 에서 P1 을 이겨야
    assert not r["paper"]                                                          # 주 절차가 아니면 추천 대상 아님
    assert simplest(out, "paper")["name"] == "X" and simplest(out, "t2")["name"] == "P2"
    assert simplest([], "t2") is None
