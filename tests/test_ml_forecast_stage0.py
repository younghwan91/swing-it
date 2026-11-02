"""ML 예측 0단계 관문 — 커버리지·결측률 바, 비트 비교(DB 불요)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("krx_quant_core.risk.factor_model")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import ml_forecast_common as C  # noqa: E402
import ml_forecast_stage0 as S0  # noqa: E402


def test_coverage_gate_bars_and_f1_window():
    dates = pd.DatetimeIndex(["2019-09-02", "2019-10-01", "2020-01-02", "2020-02-03", "2021-01-04"])
    days = list(range(5))
    cov = {k: [0.9] * 5 for k in C.FEATURES}
    cov["tp_rev"] = [0.26] * 5
    cov["eps_yoy"] = [0.0, 0.0, 0.8, 0.8, 0.8]  # 2019 는 F1 바에 들어가지 않는다
    g = S0.coverage_gate(cov, days, dates)
    assert all(v["pass"] for v in g.values())
    cov["tp_rev"] = [0.24] * 5
    assert not S0.coverage_gate(cov, days, dates)["tp_rev"]["pass"]
    cov["tp_rev"] = [0.26] * 5
    cov["eps_yoy"] = [1.0, 1.0, 0.6, 0.6, 0.6]
    assert not S0.coverage_gate(cov, days, dates)["eps_yoy"]["pass"]
    cov["eps_yoy"] = [0.8] * 5
    cov["hi52"] = [0.79] * 5
    assert not S0.coverage_gate(cov, days, dates)["hi52"]["pass"]


def test_delist_gate_gap():
    N, T = 100, 3
    raw = {k: np.ones((N, T)) for k in C.FEATURES}
    is_del = np.zeros(N, bool)
    is_del[:10] = True
    U = {t: np.arange(N) for t in range(T)}
    raw["frgn_own"][:3] = np.nan  # 폐지 결측 30%, 상장 0%
    raw["tp_rev"][:1] = np.nan  # 폐지 결측 10%
    g = S0.delist_gate(raw, U, list(range(T)), is_del)
    assert not g["frgn_own"]["pass"] and g["frgn_own"]["gap"] == pytest.approx(0.3)
    assert g["tp_rev"]["pass"] and g["hi52"]["pass"]


def test_same_is_bitwise_and_nan_aware():
    a = np.array([1.0, np.nan, 2.0])
    assert S0._same(a, a.copy())
    assert not S0._same(a, np.array([1.0, np.nan, 2.0 + 1e-15]))
    assert not S0._same(a, np.array([1.0, 0.0, 2.0]))
