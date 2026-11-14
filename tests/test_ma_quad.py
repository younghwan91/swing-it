"""q차식 롤링 외삽 테스트 — research.signals.ma_quad."""

import numpy as np
import pandas as pd

import pytest

from research.signals.ma_quad import (
    draw_index,
    forecast_z,
    month_pool,
    month_pools,
    month_starts,
    poly_forecast,
    rolling_sigma,
)


def test_quadratic_exact():
    N, k = 10, 3
    t = np.arange(40, dtype=float)
    # s = t' - t0 기준이 아니라 임의 절대 시간에서도 같은 2차식이어야 하므로 t 자체의 2차식
    y = (1 + 0.1 * t + 0.01 * t**2)[None, :]
    out = poly_forecast(y, N, k, 2)
    assert out.shape == (1, 40, k) and out.dtype == np.float32
    for h in range(1, k + 1):
        exp = 1 + 0.1 * (t + h) + 0.01 * (t + h) ** 2
        assert np.allclose(out[0, N - 1 : 40 - 0, h - 1], exp[N - 1 :], atol=1e-4, rtol=1e-6)
    assert np.isnan(out[0, : N - 1]).all()


def test_line_exact_q1():
    N, k = 8, 4
    t = np.arange(30, dtype=float)
    y = (2 - 0.3 * t)[None, :]
    for q in (1, 2):
        out = poly_forecast(y, N, k, q)
        for h in range(1, k + 1):
            assert np.allclose(out[0, N - 1 :, h - 1], (2 - 0.3 * (t + h))[N - 1 :], atol=1e-5)


def test_q0_is_last_value():
    y = np.random.default_rng(0).normal(size=(3, 25))
    out = poly_forecast(y, 12, 5, 0)
    for h in range(5):
        assert np.allclose(out[:, :, h], y.astype(np.float32))
    y2 = y.copy()
    y2[1, 3] = np.nan
    out2 = poly_forecast(y2, 12, 5, 0)
    assert np.isnan(out2[1, 3]).all()
    assert np.isfinite(out2[1, 4:]).all() and np.isfinite(out2[1, :3]).all()


def test_nan_window_gives_nan():
    N = 6
    y = np.random.default_rng(1).normal(size=(2, 30)).cumsum(axis=1)
    y[0, 12] = np.nan
    out = poly_forecast(y, N, 2, 2)
    assert np.isnan(out[0, 12 : 12 + N]).all()
    assert np.isfinite(out[0, 12 + N :]).all()
    assert np.isfinite(out[0, N - 1 : 12]).all()
    assert np.isfinite(out[1, N - 1 :]).all()


def test_matches_polyfit():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(2, 50)).cumsum(axis=1)
    N, k = 15, 3
    for q in (1, 2, 3):
        out = poly_forecast(y, N, k, q)
        for i in range(2):
            for t in (N - 1, 20, 49):
                s = np.arange(-N + 1, 1)
                c = np.polyfit(s, y[i, t - N + 1 : t + 1], q)
                for h in range(1, k + 1):
                    assert abs(out[i, t, h - 1] - np.polyval(c, h)) < 3e-6 + 3e-6 * abs(out[i, t, h - 1])


def test_rolling_sigma_ddof1():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(2, 200)).cumsum(axis=1) * 0.1
    y[1, 100] = np.nan
    got = rolling_sigma(y, 60)
    assert got.shape == y.shape and got.dtype == np.float64
    for i in range(2):
        ref = pd.Series(y[i]).diff().rolling(60).std().to_numpy()
        assert np.allclose(got[i], ref, equal_nan=True, atol=1e-12)


def test_invalid_args_raise():
    y = np.zeros((1, 30))
    for N, k, q in ((10, 0, 2), (10, 3, -1), (2, 3, 2), (10, 3, 11)):
        with pytest.raises(ValueError):
            poly_forecast(y, N, k, q)
    with pytest.raises(ValueError):
        poly_forecast(y, 10, 0, 0)  # q=0 도 검증
    with pytest.raises(ValueError):
        poly_forecast(y, 0, 3, 0)


# ---------- 오차 라이브러리 ----------


def _setup(n_days=1000, n_codes=3, k=5, seed=0):
    days = pd.bdate_range("2015-01-01", periods=n_days)
    y = np.random.default_rng(seed).normal(size=(n_codes, n_days)).cumsum(axis=1) * 0.01
    fc = poly_forecast(y, 20, k, 1)
    z = forecast_z(y, fc, rolling_sigma(y, 60))
    member = np.ones((n_codes, n_days), dtype=bool)
    return days, y, z, member, k


def test_z_nan_tail():
    days, y, z, member, k = _setup()
    assert z.dtype == np.float32 and z.shape == (3, 1000, k)
    assert np.isnan(z[:, -1, :]).all()  # 마지막 날은 어느 h 도 미래가 없다
    for h in range(1, k + 1):
        assert np.isnan(z[:, 1000 - h :, h - 1]).all()  # t+h ≥ n_days 칸은 전부 NaN
    # 꼬리 직전: h 가 범위 안인 칸만 유한
    t = 1000 - k - 1
    assert np.isfinite(z[:, t, :]).all()
    t = 1000 - 3
    assert np.isfinite(z[:, t, :2]).all() and np.isnan(z[:, t, 2:]).all()


def test_z_value_and_nan_rules():
    y = np.random.default_rng(1).normal(size=(1, 200)).cumsum(axis=1) * 0.01
    fc = poly_forecast(y, 10, 3, 1)
    sg = rolling_sigma(y, 60)
    z = forecast_z(y, fc, sg)
    t = 100
    for h in (1, 2, 3):
        assert np.isclose(z[0, t, h - 1], (y[0, t + h] - fc[0, t, h - 1]) / sg[0, t], rtol=1e-4)
    sg2 = sg.copy()
    sg2[0, 100] = 0.0
    assert np.isnan(forecast_z(y, fc, sg2)[0, 100]).all()
    y2 = y.copy()
    y2[0, 102] = np.nan
    z2 = forecast_z(y2, fc, sg)
    assert np.isnan(z2[0, 100, 1]) and np.isfinite(z2[0, 100, 0])


def test_pool_no_future():
    days, y, z, member, k = _setup()
    starts = month_starts(days)
    assert starts[0] == 0 and days[starts[1]].month != days[starts[1] - 1].month
    n_checked = 0
    for t_m in starts[::6]:
        out = month_pool(days, z, member, int(t_m), k, return_origins=True)
        if out is None:
            continue
        pool, origins = out
        assert pool.shape == (len(origins), k)
        assert (origins[:, 1] + k <= t_m).all()
        n_checked += 1
    assert n_checked > 5


def test_pool_window_3y():
    days, y, z, member, k = _setup(n_days=1500)
    t_m = int(month_starts(days)[-3])
    pool, origins = month_pool(days, z, member, t_m, k, years=3, return_origins=True)
    lo = days[t_m] - pd.DateOffset(years=3)
    assert (days[origins[:, 1]] >= lo).all()
    # 경계 안쪽은 실제로 포함됨(가장 이른 원점이 창 시작 근처)
    assert days[origins[:, 1]].min() <= lo + pd.Timedelta(days=7)


def test_pool_member_and_finite_filter():
    days, y, z, member, k = _setup()
    member = member.copy()
    member[1] = False
    z = z.copy()
    z[2, :, 3] = np.nan  # 코드 2 는 h=4 가 늘 NaN → 제외
    t_m = int(month_starts(days)[-5])
    pool, origins = month_pool(days, z, member, t_m, k, return_origins=True)
    assert set(origins[:, 0]) == {0}
    assert np.isfinite(pool).all() and pool.dtype == np.float32


def test_pool_min_dates():
    days, y, z, member, k = _setup(n_days=1000)
    t_m = int(month_starts(days)[-8])
    pool, origins = month_pool(days, z, member, t_m, k, return_origins=True)
    n = len(np.unique(origins[:, 1]))
    assert n >= 120
    # 원점 날짜를 정확히 120 / 119 개만 남김
    keep = np.unique(origins[:, 1])
    for count, expect in ((120, True), (119, False)):
        m2 = np.zeros_like(member)
        m2[:, keep[:count]] = True
        got = month_pool(days, z, m2, t_m, k)
        assert (got is not None) is expect


def test_month_pools_generator_matches():
    days, y, z, member, k = _setup()
    got = dict(month_pools(days, z, member, k))
    assert got
    for t_m, pool in got.items():
        assert np.array_equal(pool, month_pool(days, z, member, t_m, k))
    # 이력 부족한 초기 달은 키 없음
    assert 0 not in got


def test_draw_index_deterministic():
    d = pd.Timestamp("2020-03-02")
    a = draw_index("005930", d, 2, 20, 5, 1000)
    b = draw_index("005930", d, 2, 20, 5, 1000)
    assert a.shape == (200,) and a.dtype == np.int64
    assert np.array_equal(a, b) and a.min() >= 0 and a.max() < 1000
    assert np.array_equal(a, draw_index("005930", pd.Timestamp("2020-03-17"), 2, 20, 5, 1000))  # 같은 달 = 같은 색인
    assert not np.array_equal(a, draw_index("005930", pd.Timestamp("2020-04-01"), 2, 20, 5, 1000))
    assert not np.array_equal(a, draw_index("000660", d, 2, 20, 5, 1000))
    assert not np.array_equal(a, draw_index("005930", d, 1, 20, 5, 1000))
