"""ML 예측 — 퍼지와 봉인 창 가드(코드가 막는지 위반을 주입해 본다)."""

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
import ml_forecast_models as MD  # noqa: E402


def _dates():
    return pd.bdate_range("2019-01-02", "2023-12-28")


def test_purge_every_train_row_label_ends_by_t0():
    dates = _dates()
    grid, rows = C.ic_grid(dates), C.phase_rows(dates)
    for yr in C.TEST_YEARS:
        t0 = MD.t0_of(dates, grid, yr)
        tr = MD.train_days(rows, t0)
        assert tr, yr
        assert all(t + C.LABEL_LAST <= t0 for t in tr)
        assert dates[t0].year == yr and dates[tr[-1]].year <= yr
        # 다음 위상 날짜는 퍼지로 빠진다(경계)
        nxt = rows[rows.index(tr[-1]) + 1]
        assert nxt + C.LABEL_LAST > t0


def test_grid_phase_and_last_decision():
    dates = _dates()
    grid, rows = C.ic_grid(dates), C.phase_rows(dates)
    assert dates[grid[0]] == pd.Timestamp("2019-08-01")
    assert set(grid) <= set(rows)
    assert dates[rows[0]] < pd.Timestamp("2019-01-10")
    assert all((t - grid[0]) % C.IC_STEP == 0 for t in rows)
    assert grid[-1] + C.LABEL_LAST <= len(dates) - 1
    assert grid[-1] + C.IC_STEP + C.LABEL_LAST > len(dates) - 1


def test_ic_grid_never_reaches_sealed_window_even_if_panel_does():
    """패널이 2024 까지 있어도 격자·학습 위상은 END 안에서 끝난다."""
    dates = pd.bdate_range("2019-01-02", "2024-06-28")
    last = dates.searchsorted(pd.Timestamp(C.END), side="right") - 1
    for t in C.ic_grid(dates) + C.phase_rows(dates):
        assert t + C.LABEL_LAST <= last


@pytest.mark.parametrize("bad", ["2023-12-29", "2024-01-02", "2026-09-11"])
def test_assert_window_eval_rejects_after_end(bad):
    with pytest.raises(ValueError):
        C.assert_window(bad, "eval")


def test_assert_window_eval_accepts_end():
    assert C.assert_window("2023-12-28", "eval").mode == "eval"


def test_fit_only_window_blocks_ic_function():
    """fit_only 창으로 IC 를 부르면 예외(봉인 구간 지표 계산을 코드로 막는다, §1·§11)."""
    w = C.assert_window("2026-09-11", "fit_only")
    assert w.mode == "fit_only"
    with pytest.raises(PermissionError):
        IC.ic_series({}, {}, {}, np.zeros((1, 1)), [], w)
    with pytest.raises(PermissionError):
        C.require_eval(w, "북")
    with pytest.raises(ValueError):
        C.assert_window("2026-09-14", "fit_only")


def test_require_eval_rejects_forged_window():
    with pytest.raises((PermissionError, ValueError)):
        C.require_eval(C.Window(end="2024-06-28", mode="eval"), "IC")
    with pytest.raises(PermissionError):
        C.require_eval("2023-01-01", "IC")
