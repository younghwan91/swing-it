"""research.combined.gaps — 야간 갭·원주가 틱·사건 갭."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.combined.gaps import gap_arrays, tick_dated_vec
from research.signals.pattern_events import tick_dated

POST = pd.bdate_range("2024-03-04", periods=8)  # 2023-01-25 이후
PRE = pd.bdate_range("2022-03-02", periods=8)  # 개편 전


def _args(n, t, days=POST, mkt=None):
    o = np.full((n, t), 100.0)
    return dict(O=o, C=o.copy(), raw_close=o.copy(), reopen=np.zeros((n, t), bool),
                lu=np.full((n, t), 1.30), days=days, mkt=np.zeros(n, int) if mkt is None else mkt)


def test_split_day_gap_and_reopen_nan():
    a = _args(1, 8)
    # 2:1 분할을 수정가로 소급: 분할일(t=5) 전 가격이 이미 절반으로 조정돼 있다
    a["C"][0] = 50.0
    a["O"][0] = 50.0
    a["raw_close"][0, :5] = 100.0
    a["raw_close"][0, 5:] = 50.0
    r = gap_arrays(**a)
    assert abs(r["gap"][0, 5]) < 1e-6
    assert np.isnan(r["gap"][0, 0])
    a["reopen"][0, 5] = True
    assert np.isnan(gap_arrays(**a)["gap"][0, 5])


def test_tickp_uses_raw_price():
    a = _args(1, 8)
    a["C"][0] = 60905.0  # 수정가
    a["O"][0] = 60905.0
    a["raw_close"][0] = 3160.0  # 원주가 -> 호가 5원(2024 표)
    r = gap_arrays(**a)
    assert np.isclose(r["tickp"][0, 3], 5 / 3160, rtol=1e-5)  # ≈15.8bp
    assert r["tickp"][0, 3] > 1e-3  # 1.6bp 가 아니다


def test_gap_event_symmetric():
    a = _args(3, 8)
    a["O"][0, 4] = 115.0
    a["O"][1, 4] = 85.0
    a["O"][2, 4] = 114.9
    ev = gap_arrays(**a)["gap_event"]
    assert ev[0, 4] and ev[1, 4] and not ev[2, 4]


def test_pre_reform_tick_by_market():
    p = np.array([[60_000.0, 600_000.0]])
    pre = tick_dated_vec(p, PRE[:1], np.array([0]))  # KOSPI
    assert pre.tolist() == [[100.0, 1000.0]]
    pre = tick_dated_vec(p, PRE[:1], np.array([1]))  # KOSDAQ
    assert pre.tolist() == [[100.0, 100.0]]
    post = tick_dated_vec(p, POST[:1], np.array([1]))  # 개편 후 통합표
    assert post.tolist() == [[100.0, 1000.0]]


def test_tick_dated_vec_matches_repo_scalar():
    prices = np.array([500, 999, 1000, 1999, 2000, 4999, 5000, 9999, 10000, 19999, 20000,
                       49999, 50000, 99999, 100000, 199999, 200000, 499999, 500000, 800000.0])
    for days, row in ((PRE[:1], 0), (POST[:1], 1)):
        for m in (0, 1):
            got = tick_dated_vec(prices[None, :], days, np.array([m]))[0]
            ref = [tick_dated(float(p), row, m, 1) for p in prices]
            assert got.tolist() == ref
