"""q차식 롤링 외삽 테스트 — research.signals.ma_quad."""

import numpy as np
import pandas as pd

import pytest

from research.signals.ma_quad import (
    draw_index,
    event_probs,
    ma_state,
    realized,
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


# ---------------------------------------------------------------- Task 3

def _panel(n_codes=3, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n_codes, n_days)), axis=1))


def _events(n_days, k, n_codes=3):
    ci = np.array([0, 1, 2, 1])
    s = np.array([150, 200, 250, 300])
    assert (s + k < n_days).all()
    return ci, s


def test_zero_sigma_matches_realized():
    k = 7
    C = _panel()
    ci, s = _events(C.shape[1], k)
    fc = np.full(C.shape + (k,), np.nan)
    for e, (c, t) in enumerate(zip(ci, s)):
        fc[c, t] = np.log(C[c, t + 1 : t + k + 1])
    sigma = np.zeros(C.shape)
    pool = np.zeros((10, k))
    idx = np.zeros((len(ci), 200), dtype=np.int64)
    p = event_probs(C, fc, sigma, pool, idx, ci, s, k)
    r = realized(C, ci, s, k)
    for key in p:
        assert np.array_equal(p[key], r[key]) and set(np.unique(p[key])) <= {0.0, 1.0}


def test_known_part_arithmetic():
    k = 5
    C = _panel()
    c, t = 1, 220
    rng = np.random.default_rng(1)
    pool = rng.normal(size=(50, k))
    sig = np.full(C.shape, 0.01)
    fc = np.full(C.shape + (k,), np.nan)
    fc[c, t] = np.log(C[c, t]) + 0.001 * np.arange(1, k + 1)
    idx = rng.integers(0, 50, (1, 200))
    # 경로 하나씩 직접 만들어 pandas 로 MA 정배열 비율을 센다
    cnt = 0
    for m in range(200):
        fut = np.exp(fc[c, t] + sig[c, t] * pool[idx[0, m]])
        ser = pd.Series(np.concatenate([C[c, : t + 1], fut]))
        a5, a20, a60, a120 = (ser.rolling(w).mean().iloc[-1] for w in (5, 20, 60, 120))
        cnt += a5 > a20 > a60 > a120
    p = event_probs(C, fc, sig, pool, idx, np.array([c]), np.array([t]), k)
    assert abs(p["p_align"][0] - cnt / 200) < 1e-9


def test_tie_is_not_aligned():
    C = np.full((1, 300), 10.0)
    st = ma_state(C)
    assert not st["aligned"][0, 250]
    k = 3
    fc = np.full((1, 300, k), np.log(10.0))
    p = event_probs(C, fc, np.zeros((1, 300)), np.zeros((4, k)), np.zeros((1, 200), dtype=np.int64),
                    np.array([0]), np.array([200]), k)
    assert p["p_align"][0] == 0.0 and p["p_up_5_20"][0] == 0.0


def test_nan_history_skips_event():
    k = 4
    C = _panel(1, 300)
    C[0, 150] = np.nan  # t=200, k=4 → MA120 창 [85..204] 안
    fc = np.zeros((1, 300, k))
    kw = dict(pool=np.zeros((3, k)), idx=np.zeros((1, 200), dtype=np.int64), ci=np.array([0]), k=k)
    p = event_probs(C, fc, np.full((1, 300), 0.01), s=np.array([200]), **kw)
    assert all(np.isnan(v[0]) for v in p.values())
    # sigma / fc 가 비유한이어도 NaN
    C2 = _panel(1, 300)
    p = event_probs(C2, fc, np.full((1, 300), np.nan), s=np.array([200]), **kw)
    assert np.isnan(p["p_align"][0])
    fc2 = fc.copy()
    fc2[0, 200, 2] = np.nan
    p = event_probs(C2, fc2, np.full((1, 300), 0.01), s=np.array([200]), **kw)
    assert np.isnan(p["p_align"][0])
    # realized 는 창 NaN / 범위 밖에서 NaN
    r = realized(C, np.array([0, 0]), np.array([200, 298]), k)
    assert np.isnan(r["p_align"]).all()


def test_p_in_unit_interval():
    k = 20
    C = _panel(2, 400, seed=3)
    rng = np.random.default_rng(4)
    fc = rng.normal(np.log(100), 0.1, C.shape + (k,))
    sg = rng.uniform(0.005, 0.05, C.shape)
    pool = rng.normal(size=(100, k))
    ci = np.array([0, 1, 0])
    s = np.array([200, 250, 300])
    idx = rng.integers(0, 100, (3, 200))
    p = event_probs(C, fc, sg, pool, idx, ci, s, k)
    for v in p.values():
        assert ((v >= 0) & (v <= 1)).all()
        assert np.allclose(v * 200, np.round(v * 200))


def test_k_ge_window_uses_path_only():
    # k=130 > 120: 알려진 구간 없이 경로만으로 MA 를 만든다. 단조 상승·sigma=0 → 정배열 100%
    k = 130
    C = _panel(1, 500, seed=5)
    fc = np.full((1, 500, k), np.nan)
    fc[0, 200] = np.log(C[0, 200]) + 0.002 * np.arange(1, k + 1)
    p = event_probs(C, fc, np.zeros((1, 500)), np.zeros((2, k)), np.zeros((1, 200), dtype=np.int64),
                    np.array([0]), np.array([200]), k)
    assert p["p_align"][0] == 1.0


def test_f200_and_tmpl_small_panel():
    n = 400
    up = 10 * np.exp(0.004 * np.arange(n))  # 꾸준한 상승
    down = 10 * np.exp(-0.004 * np.arange(n))
    C = np.vstack([up, down])
    st = ma_state(C)
    assert st["f200"][0, 300] and not st["f200"][1, 300]
    assert not st["f200"][0, 100]  # MA200 없음 → False
    assert st["tmpl"][0, 399] and not st["tmpl"][1, 399]
    assert not st["tmpl"][0, 100]
    # 20일 전보다 MA200 이 낮지 않아야 한다: 하락 후 반등 한 점
    C2 = np.concatenate([np.linspace(100, 50, 300), np.linspace(50, 120, 100)])[None]
    st2 = ma_state(C2)
    assert st2["f200"][0, 399] == bool(C2[0, 399] > st2["ma"][200][0, 399] and st2["ma"][200][0, 399] > st2["ma"][200][0, 379])
    # 30% 저점 위 조건: 평평한 가격은 C >= 1.3*min 실패
    flat = np.full((1, 400), 10.0)
    assert not ma_state(flat)["tmpl"][0, 399]
    # d = 1 − 묶음하단/종가
    s3 = ma_state(up[None])
    low = np.minimum(np.minimum(s3["ma"][20], s3["ma"][60]), s3["ma"][120])
    assert np.allclose(s3["bundle_low"][0, 200:], low[0, 200:])
    assert np.allclose(s3["d"][0, 200:], (1 - low / up[None])[0, 200:])


def test_all_keys_pinned_to_pandas_zero_sigma():
    """sigma=0 → 경로 하나뿐. 사건마다 pandas rolling 으로 만든 네 키와 정확히 같아야 한다(MA 창 한 칸 어긋남 검출)."""
    rng = np.random.default_rng(11)
    for k in (3, 5, 30, 119, 130):
        C = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, (1, 600)), axis=1))
        s = np.arange(150, 440, 2)
        ci = np.zeros(len(s), dtype=np.int64)
        fc = np.zeros((1, 600, k))
        for t in s:
            fc[0, t] = np.log(C[0, t + 1 : t + k + 1])
        p = event_probs(C, fc, np.zeros((1, 600)), np.zeros((2, k)), np.zeros((len(s), 200), dtype=np.int64), ci, s, k)
        ma = {w: pd.Series(C[0]).rolling(w).mean().to_numpy() for w in (5, 20, 60, 120)}
        e = s + k
        exp = {
            "p_align": (ma[5][e] > ma[20][e]) & (ma[20][e] > ma[60][e]) & (ma[60][e] > ma[120][e]),
            "p_up_5_20": ma[5][e] > ma[20][e],
            "p_up_20_60": ma[20][e] > ma[60][e],
            "p_up_60_120": ma[60][e] > ma[120][e],
        }
        for key, v in exp.items():
            assert np.array_equal(p[key], v.astype(float)), (k, key)
        # C[t] 를 키워 알려진 구간 끝점 포함 여부까지 확인
        C2 = C.copy()
        C2[0, s] *= 1.5
        fc2 = fc.copy()
        p2 = event_probs(C2, fc2, np.zeros((1, 600)), np.zeros((2, k)), np.zeros((len(s), 200), dtype=np.int64), ci, s, k)
        assert p2["p_up_5_20"].shape == s.shape
