"""research.combined.gaps — 야간 갭·수정가 틱 비율·사건 갭."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.combined.gaps import gap_arrays, tick_dated_vec
from research.signals.pattern_events import tick_dated

POST = pd.bdate_range("2024-03-04", periods=8)  # 2023-01-25 이후
PRE = pd.bdate_range("2022-03-02", periods=8)  # 개편 전


def _args(n, t, days=POST, mkt=None):
    o = np.full((n, t), 100.0)
    return dict(O=o, C=o.copy(), reopen=np.zeros((n, t), bool),
                lu=np.full((n, t), 1.30), days=days, mkt=np.zeros(n, int) if mkt is None else mkt)


def test_split_day_gap_and_reopen_nan():
    a = _args(1, 8)
    # 2:1 분할을 수정가로 소급: 분할일(t=5) 전 가격이 이미 절반으로 조정돼 있다
    a["C"][0] = 50.0
    a["O"][0] = 50.0
    r = gap_arrays(**a)
    assert abs(r["gap"][0, 5]) < 1e-6
    assert np.isnan(r["gap"][0, 0])
    a["reopen"][0, 5] = True
    assert np.isnan(gap_arrays(**a)["gap"][0, 5])


def test_tickp_is_adjusted_close_function_no_raw_input():
    """판정 26: tickp 는 수정 종가의 결정적 함수다 — 원주가 입력이 없어 1/tickp 로 원주가(= 미래 수정 계수)를 되살릴 수 없다."""
    import inspect
    assert "raw" not in " ".join(inspect.signature(gap_arrays).parameters)
    a = _args(2, 8, mkt=np.array([0, 1]))
    a["C"][0] = 60905.0                                       # 수정가(원주가 3,160 이었어도 입력이 없다)
    a["C"][1] = np.linspace(1500, 2600, 8)
    a["O"] = a["C"] * 1.01
    r = gap_arrays(**a)["tickp"]
    ref = tick_dated_vec(a["C"][:, :-1], a["days"][1:], a["mkt"]) / a["C"][:, :-1]
    assert np.allclose(r[:, 1:], ref, rtol=1e-6) and np.isnan(r[:, 0]).all()
    assert np.isclose(r[0, 3], 100 / 60905, rtol=1e-5)       # ≈1.6bp — 수정 사건 전 구간은 근사(공개된 맞바꿈)
    b = {**a, "O": a["O"] * 1.05}                             # 시가가 달라도 tickp 는 C 만 본다
    assert np.array_equal(gap_arrays(**b)["tickp"], gap_arrays(**a)["tickp"], equal_nan=True)


def test_data_module_reads_no_raw_close():
    import inspect

    import research.combined.data as cd
    src = inspect.getsource(cd)
    assert "read_raw_close" not in src and "raw_close" not in src


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


def test_tickp_shifts_previous_close():
    a = _args(1, 8)
    a["C"][0, :5] = 3160.0
    a["C"][0, 5:] = 61000.0
    r = gap_arrays(**a)["tickp"]
    assert np.isnan(r[0, 0])
    assert np.isclose(r[0, 5], 5 / 3160, rtol=1e-5)  # 전일(t=4) 종가 3160
    assert np.isclose(r[0, 6], 100 / 61000, rtol=1e-5)  # 전일(t=5) 종가 61000 -> 100원


def test_gap_arrays_pre_reform_kosdaq():
    a = _args(2, 8, days=PRE, mkt=np.array([1, 0]))
    a["C"][:] = 60_000.0
    r = gap_arrays(**a)["tickp"]
    assert np.isclose(r[0, 3], 100 / 60000, rtol=1e-5)  # 코스닥 6만원 -> 100
    assert np.isclose(r[1, 3], 100 / 60000, rtol=1e-5)  # 코스피 6만원 -> 100(옛 표 10만 미만)
    a["C"][:] = 600_000.0
    r = gap_arrays(**a)["tickp"]
    assert np.isclose(r[0, 3], 100 / 600000, rtol=1e-5)  # 코스닥 60만 -> 100
    assert np.isclose(r[1, 3], 1000 / 600000, rtol=1e-5)  # 코스피 60만 -> 1000


def test_tickp_reform_boundary_uses_order_day_table():
    days = pd.DatetimeIndex(["2023-01-20", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26"])
    a = _args(1, 5, days=days, mkt=np.array([1]))
    a["C"][:] = 1500.0  # 옛 표(<5000) 5원, 새 표(<2000) 1원
    r = gap_arrays(**a)["tickp"]
    # t=2 (2023-01-24 주문): 옛 표 5원. t=3 (2023-01-25 주문, 가격은 01-24 종가): 새 표 1원
    assert np.isclose(r[0, 2], 5 / 1500, rtol=1e-5)
    assert np.isclose(r[0, 3], 1 / 1500, rtol=1e-5)
