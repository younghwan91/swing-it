"""research/signals/catalyst_flow.simulate_block — 진입·손절·만기·폐지 처리."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.signals.catalyst_flow import simulate_block  # noqa: E402


def _arr(rows):
    return np.array(rows, dtype=float)


def test_time_exit_at_close_of_hold_day():
    o = _arr([[100], [101], [102]])
    h = _arr([[105], [106], [107]])
    lo = _arr([[99], [98], [100]])
    c = _arr([[104], [103], [110]])
    r = simulate_block(o, h, lo, c, 0, np.array([0]), hold=3, stop=0.08)
    assert r[0] == pytest.approx(0.10)


def test_stop_on_entry_day_low_and_gap_later():
    o = _arr([[100, 100], [100, 90]])
    h = _arr([[100, 100], [100, 91]])
    lo = _arr([[91, 99], [99, 85]])
    c = _arr([[95, 99], [99, 88]])
    r = simulate_block(o, h, lo, c, 0, np.array([0, 1]), hold=2, stop=0.08)
    assert r[0] == pytest.approx(-0.08)           # 진입일 저가가 손절가(92) 아래
    assert r[1] == pytest.approx(90 / 100 - 1)    # 다음 날 시가가 이미 손절가 아래 → 시가 체결


def test_cannot_enter_without_open_and_delisting_exits_last_close():
    nan = np.nan
    o = _arr([[nan, 100], [nan, 101], [nan, nan]])
    h = _arr([[nan, 102], [nan, 103], [nan, nan]])
    lo = _arr([[nan, 99], [nan, 100], [nan, nan]])
    c = _arr([[nan, 101], [nan, 97], [nan, nan]])
    r = simulate_block(o, h, lo, c, 0, np.array([0, 1]), hold=3, stop=0.08)
    assert np.isnan(r[0])                          # 살 수 없던 자리
    assert r[1] == pytest.approx(97 / 100 - 1)     # 시세 끊김 → 마지막 유효 종가
