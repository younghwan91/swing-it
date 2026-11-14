"""q차식 롤링 외삽 테스트 — research.signals.ma_quad."""

import numpy as np
import pandas as pd

from research.signals.ma_quad import poly_forecast, rolling_sigma


def _quad(n_days=40, n=10, k=3):
    s_all = np.arange(n_days, dtype=float)
    return s_all, n, k


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
                    assert abs(out[i, t, h - 1] - np.polyval(c, h)) < 1e-6 + 1e-6 * abs(out[i, t, h - 1])


def test_rolling_sigma_ddof1():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(2, 200)).cumsum(axis=1) * 0.1
    y[1, 100] = np.nan
    got = rolling_sigma(y, 60)
    assert got.shape == y.shape and got.dtype == np.float64
    for i in range(2):
        ref = pd.Series(y[i]).diff().rolling(60).std().to_numpy()
        assert np.allclose(got[i], ref, equal_nan=True, atol=1e-12)
