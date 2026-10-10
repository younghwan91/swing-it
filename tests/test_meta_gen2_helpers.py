"""research/experiments/meta_gen2_run 의 순수 헬퍼 — 현금 금리·짝지은 월 블록 부트스트랩·바 산술·B 층 키·뉴스 창(DB 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.meta_gen2_run import (
    CD91,
    a_bar,
    b_bar,
    block_index,
    cash_daily,
    month_ids,
    neg_flag,
    paired_month_boot,
    stratum_b,
    window_counts,
    with_cash,
    years_positive,
)


def test_cash_daily_and_with_cash():
    d = pd.DatetimeIndex(["2020-01-02", "2025-06-30", "2027-01-04", "2019-12-30"])
    r = cash_daily(d)
    np.testing.assert_allclose(r, [CD91[2020] / 252, CD91[2025] / 252, CD91[2026] / 252, CD91[2020] / 252])
    out = with_cash(np.array([0.01, 0.0, -0.01]), np.array([0.2, np.nan, 1.3]), np.array([0.001, 0.001, 0.001]))
    np.testing.assert_allclose(out, [0.01 + 0.8 * 0.001, 0.001, -0.01])          # 노출 NaN = 전부 현금, 노출 > 1 이면 0


def test_month_ids_and_block_index():
    d = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-04-01"])
    assert month_ids(d).tolist() == [0, 0, 1, 3]
    idx = block_index(10, 3, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10) and idx.min() >= 0 and idx.max() <= 9
    assert (np.diff(idx[:, :3], axis=1) == 1).all()                                # 블록 안은 이어진 달


def test_paired_month_boot_zero_and_positive_shift():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2020-01-01", "2022-12-31")
    mon = month_ids(days)
    a = rng.normal(0.0005, 0.01, len(days))
    tm = rng.integers(0, mon[-1] + 1, 300)
    net = rng.normal(0.01, 0.05, 300)
    z = paired_month_boot(a, a, mon, tm, net, tm, net, n=300, seed=0)
    for k in ("cagr", "calmar", "net"):
        assert z[k][0] == pytest.approx(0.0, abs=1e-12) and z[k][3] == pytest.approx(0.0, abs=1e-12)
    up = paired_month_boot(a + 0.0004, a, mon, tm, net + 0.01, tm, net, n=300, seed=0)
    assert up["cagr"][0] > 0 and up["net"][0] == pytest.approx(0.01) and up["cagr"][4] == 1.0
    assert up["months"] == mon[-1] + 1


def test_a_bar_and_b_bar_arithmetic():
    good = {k: {"pt": 0.01, "lo": 0.001, "ex25": 0.002} for k in ("net", "cagr", "calmar")}
    assert a_bar(good)["all"]
    bad = {**good, "calmar": {"pt": 0.01, "lo": -0.001, "ex25": 0.002}}
    r = a_bar(bad)
    assert r["c1"] and not r["c2"] and r["c3"] and not r["all"]
    nan = {**good, "net": {"pt": np.nan, "lo": 0.1, "ex25": 0.1}}
    assert not a_bar(nan)["c1"]
    s = {"exc_lo": 0.001, "net1": 0.002, "exc_h1": 0.01, "exc_h2": 0.02, "exc_ex25": 0.01, "years_pos": 6, "n_years": 10}
    assert b_bar(s)["all"]
    assert not b_bar({**s, "years_pos": 5})["b5"]                                  # 과반 = 절반 초과
    assert not b_bar({**s, "exc_h2": -0.01})["b3"]
    assert not b_bar({**s, "exc_lo": np.nan})["b1"]


def test_stratum_b_keys():
    sid = stratum_b(np.array([3, 3, 3, 4]), np.array([0, 1, -1, 0]), np.array([4, 4, 4, np.nan]), np.array([1, 1, 1, 1]),
                    np.array([2, 2, 2, 2]))
    assert sid[0] != sid[1] and sid[0] >= 0 and sid[2] == -1 and sid[3] == -1
    assert sid[0] == (((3 * 2 + 0) * 5 + 4) * 2 + 1) * 5 + 2


def test_window_counts_and_neg_flag():
    daily = pd.DataFrame({"code": ["A", "A", "A", "B"],
                          "date": pd.to_datetime(["2025-09-01", "2025-10-15", "2025-10-31", "2025-10-01"]),
                          "n_neg": [1, 0, 0, 0], "n_articles": [2, 1, 0, 0], "n_feed": [2, 4, 1, 3]})
    keys = pd.DataFrame({"code": ["A", "A", "B", "C"], "date": pd.to_datetime(["2025-10-31", "2025-10-30", "2025-10-02",
                                                                              "2025-10-02"])})
    w = window_counts(daily, keys, window_days=60)
    # A 10-31: (09-01, 10-31] → 10-15, 10-31 (09-01 은 창 밖 — 정확히 60일 전은 빠진다)
    assert w[["n_neg_60", "n_articles_60", "n_feed_60"]].to_numpy().tolist() == [[0, 1, 5], [1, 3, 6], [0, 0, 3], [0, 0, 0]]
    f = neg_flag(w["n_neg_60"], w["n_articles_60"], w["n_feed_60"])
    assert np.isnan(f[0]) and f[1] == 1.0 and np.isnan(f[2]) and f[3] == 0.0       # 판정률 0.2 → 모름, 피드 0 → 앎


def test_years_positive():
    yp, ny, d = years_positive(np.array([0.1, -0.2, 0.3, np.nan, 0.05]), np.array([2020, 2020, 2021, 2022, 2022]))
    assert (yp, ny) == (2, 3) and d[2020] == (pytest.approx(-0.05), 2) and d[2022] == (pytest.approx(0.05), 1)
