"""ML 예측 1단계 — 바·판정·IC 규칙(DB 불요)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("krx_quant_core.risk.factor_model")
pytest.importorskip("krx_quant_core.stats.metrics")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import ml_forecast_common as C  # noqa: E402
import ml_forecast_ic as IC  # noqa: E402


def _stat(mean=0.03, icir=0.3, yrs=3):
    return {"mean": mean, "icir": icir, "years_positive": yrs}


NULL = {"q95_mean": 0.02, "q95_icir": 0.2}
GOOD_D = {"mean": 0.01, "t_nw6": 2.5}
BAD_D = {"mean": 0.01, "t_nw6": 1.5}


def test_bars_E_needs_both_nulls():
    assert IC.bars_for("E", _stat(), NULL, 0.025, None, None)["I-1"]
    assert not IC.bars_for("E", _stat(), NULL, 0.05, None, None)["I-1"]
    assert not IC.bars_for("E", _stat(mean=0.01), NULL, 0.0, None, None)["I-1"]


def test_bars_nan_fails():
    b = IC.bars_for("M", _stat(mean=float("nan"), icir=float("nan")), NULL, None, None, None)
    assert not b["I-1"] and not b["I-2"] and not b["I-4"] and not b["I-5"]


def test_entry_rules():
    m = IC.bars_for("M", _stat(), NULL, None, BAD_D, GOOD_D)
    assert IC.enters_stage2("M", m)  # I-4 는 진입 조건이 아니다
    m2 = IC.bars_for("M", _stat(), NULL, None, GOOD_D, BAD_D)
    assert not IC.enters_stage2("M", m2)  # I-5 실패
    e = IC.bars_for("E", _stat(yrs=2), NULL, 0.0, None, None)
    assert not IC.enters_stage2("E", e)


def test_verdict_order_void_first():
    assert IC.verdict_stage1(["x"], {"M": True, "E": True}) == "VOID"
    assert IC.verdict_stage1([], {"M": False, "E": False}) == "채택 불가(예측력 없음)"
    assert IC.verdict_stage1([], {"M": False, "E": True}).endswith("E")


def test_day_ic_filters_and_minimum():
    n = 150
    rng = np.random.default_rng(0)
    p = rng.normal(0, 1, n)
    y = p + rng.normal(0, 0.1, n)
    tr = np.ones(n, bool)
    ic, k = IC.day_ic(p, y, tr)
    assert k == n and ic > 0.9
    tr2 = tr.copy()
    tr2[:60] = False  # 유효 90 < 100 → 그날은 뺀다
    assert np.isnan(IC.day_ic(p, y, tr2)[0])
    y2 = y.copy()
    y2[:60] = np.nan
    assert IC.day_ic(p, y2, tr)[1] == 90
    # 유효 행은 값이 같은 Spearman
    ok = np.isfinite(y2) & tr
    assert IC.day_ic(p, y2, tr, 10)[0] == pytest.approx(C.spearman(p[ok], y2[ok]))


def test_stat_of_years_and_icir():
    dates = pd.DatetimeIndex(["2020-01-02", "2021-01-04", "2022-01-03", "2023-01-02"])
    s = IC.stat_of({0: 0.01, 1: 0.03, 2: -0.01, 3: 0.05}, dates)
    assert s["years_positive"] == 3 and s["n_days"] == 4
    assert s["icir"] == pytest.approx(np.mean([0.01, 0.03, -0.01, 0.05]) / np.std([0.01, 0.03, -0.01, 0.05], ddof=1))


def test_delta_nw_uses_common_days_and_lag6():
    from krx_quant_core.stats.metrics import newey_west_t

    rng = np.random.default_rng(1)
    a = {t: float(rng.normal(0.01, 0.02)) for t in range(60)}
    b = {t: float(rng.normal(0.0, 0.02)) for t in range(5, 65)}
    d = IC.delta_nw(a, b)
    x = np.array([a[t] - b[t] for t in range(5, 60)])
    assert d["n_days"] == 55
    assert d["t_nw6"] == pytest.approx(newey_west_t(x, 6)[1])


def test_seam_in_tails_counts():
    N, T = 200, 40
    seam = np.zeros((N, T), dtype=bool)
    t = 5
    yt = np.arange(N, dtype=np.float64)
    seam[N - 1, t + 3] = True  # 상위 꼬리
    seam[50, t + 3] = True  # 가운데
    seam[0, t + 1] = True  # 창 밖(t+1)
    r = IC.seam_in_tails({t: yt}, {t: np.arange(N)}, np.ones((N, T)), seam, [t])
    assert r["rows"] == 2 and r["in_top_bottom_1pct"] == 1
