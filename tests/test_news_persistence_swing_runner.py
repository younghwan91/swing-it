"""research/experiments/news_persistence_swing 의 순수 헬퍼 — 보유 수익·정합 대조·월 블록 부트스트랩·합격 바 숫자 (DB 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.news_persistence_swing import (
    COST_RT,
    Px,
    bar_numbers,
    build,
    excess,
    in_range,
    month_boot_diff,
    month_boot_mean,
    month_signs,
    ordering_holds,
)


def _px(close: np.ndarray, open_: np.ndarray | None = None, tv: np.ndarray | None = None) -> Px:
    """DB 없이 Px 를 만든다 — 행 = 거래일, 열 = 종목."""
    n, k = close.shape
    px = object.__new__(Px)
    px.days = pd.bdate_range("2024-01-01", periods=n)
    px.codes = pd.Index([f"{i:06d}" for i in range(k)])
    px.C = close.astype(float)
    px.O = (close if open_ is None else open_).astype(float)
    px.TV = (np.ones_like(close) if tv is None else tv).astype(float)
    c = pd.DataFrame(px.C)
    px.ret5 = (c / c.shift(5) - 1).to_numpy()
    px.market = np.array(["KOSPI"] * k)
    return px


def test_fwd_next_open_exits_at_d_plus_h_close():
    close = np.arange(1, 21, dtype=float).reshape(-1, 1) * 10  # 10, 20, ..., 200
    open_ = close - 5
    px = _px(close, open_)
    # d = 행 3 → 진입 행 4 시가(45), h=5 → 청산 행 8 종가(90)
    assert px.fwd(3, np.array([0]), 5, "next_open")[0] == pytest.approx(90 / 45 - 1)
    # d 종가 진입(40) → 같은 청산(행 8 종가 90)
    assert px.fwd(3, np.array([0]), 5, "close")[0] == pytest.approx(90 / 40 - 1)
    # 보유창이 데이터 밖이면 NaN
    assert np.isnan(px.fwd(15, np.array([0]), 5, "next_open")[0])


def test_fwd_delisting_mid_hold_uses_last_close_and_missing_entry_is_nan():
    close = np.full((12, 2), 100.0)
    close[6:, 0] = np.nan  # 0번 종목은 행 5 종가(100→ 아래서 120 으로) 뒤 폐지
    close[5, 0] = 120.0
    open_ = close.copy()
    open_[1, 1] = np.nan  # 1번 종목은 진입일 시가 없음
    px = _px(close, open_)
    r = px.fwd(0, np.array([0, 1]), 8, "next_open")
    assert r[0] == pytest.approx(120 / 100 - 1)
    assert np.isnan(r[1])


def test_ordering_holds_is_strict_and_nan_false():
    assert ordering_holds(-0.01, 0.0, 0.02)
    assert not ordering_holds(0.0, 0.0, 0.02)
    assert not ordering_holds(0.03, 0.01, 0.02)
    assert not ordering_holds(np.nan, 0.0, 0.02)


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    x = pd.DataFrame(rows, columns=["date", "grp", "ex"])
    x["date"] = pd.to_datetime(x["date"])
    x["month"] = x["date"].dt.to_period("M")
    x["persistence"] = np.where(x["grp"] == "lo", "one_off", "structural")
    return x


def test_month_boot_diff_point_and_ci_bracket():
    rng = np.random.default_rng(1)
    rows = []
    for m in range(1, 13):
        for _ in range(20):
            rows.append((f"2024-{m:02d}-10", "hi", 0.02 + rng.normal(0, 0.01)))
            rows.append((f"2024-{m:02d}-10", "lo", rng.normal(0, 0.01)))
    x = _frame(rows)
    d, lo, hi = month_boot_diff(x, "hi", "lo", col="grp")
    assert d == pytest.approx(x.loc[x.grp == "hi", "ex"].mean() - x.loc[x.grp == "lo", "ex"].mean())
    assert lo < d < hi
    assert lo > 0.01
    # persistence 열로도 같은 값
    assert month_boot_diff(x, "structural", "one_off")[0] == pytest.approx(d)


def test_month_boot_diff_missing_group_is_nan():
    x = _frame([("2024-01-10", "hi", 0.01), ("2024-02-10", "hi", 0.02)])
    d, lo, hi = month_boot_diff(x, "hi", "lo", col="grp")
    assert np.isnan(d) and np.isnan(lo) and np.isnan(hi)
    assert all(np.isnan(v) for v in month_boot_diff(x.iloc[:0], "hi", "lo", col="grp"))


def test_month_boot_diff_resamples_whole_months():
    # 달 하나만 있으면 모든 재추출이 같은 달 → CI 폭 0
    x = _frame([("2024-01-10", "hi", 0.03), ("2024-01-11", "lo", 0.01), ("2024-01-12", "hi", 0.01)])
    d, lo, hi = month_boot_diff(x, "hi", "lo", col="grp")
    assert d == pytest.approx(0.01) and lo == pytest.approx(0.01) and hi == pytest.approx(0.01)


def test_month_boot_mean_brackets_mean():
    x = _frame([(f"2024-{m:02d}-10", "hi", 0.01 * m) for m in range(1, 13)])
    lo, hi = month_boot_mean(x["ex"], x["month"])
    assert lo < x["ex"].mean() < hi


def test_month_signs_counts_months_with_both_groups():
    x = _frame([("2024-01-10", "hi", 0.02), ("2024-01-10", "lo", 0.01),   # +
                ("2024-02-10", "hi", 0.00), ("2024-02-10", "lo", 0.01),   # −
                ("2024-03-10", "hi", 0.05)])                              # lo 없음 → 제외
    assert month_signs(x) == (1, 2)


def test_in_range_inclusive():
    x = _frame([("2024-01-01", "hi", 0.0), ("2024-01-31", "hi", 0.0), ("2024-02-01", "hi", 0.0)])
    assert len(in_range(x, "2024-01-01", "2024-01-31")) == 2


def test_bar_numbers_arithmetic():
    def hv(cls, diff, halves, hi_mean, hi_ci):
        return {"cls": {k: (v, 1) for k, v in zip(("one_off", "multi_quarter", "structural"), cls, strict=True)},
                "ordered": ordering_holds(*cls), "diff": diff, "halves": halves, "hi_mean": hi_mean, "hi_ci": hi_ci}
    res = {"h": {5: hv((0, 1, 2), (0, 0, 0), (0, 0), 0, (0, 0)), 10: hv((0, 1, 2), (0, 0, 0), (0, 0), 0, (0, 0)),
                 20: hv((0, 1, 2), (0.01, 0.002, 0.02), (0.01, -0.001), 0.005, (0.001, 0.009)),
                 40: hv((2, 1, 0), (0, 0, 0), (0, 0), 0, (0, 0))}}
    b = bar_numbers(res)
    assert b[1]["met"] and b[1]["value"][1] == 0.002
    assert b[2]["value"][0] == 3 and b[2]["met"]
    assert not b[3]["met"]
    assert b[4]["value"][0] == pytest.approx(0.005 - COST_RT) and b[4]["met"]
    assert b[4]["value"][1] == pytest.approx(0.001 - COST_RT)


def test_build_and_excess_matched_and_universe_hedge():
    # 48 종목 × 30 일, 0번 종목만 매일 +1% (사건 종목) — 나머지 평평
    n, k = 30, 48
    close = np.full((n, k), 100.0)
    close[:, 0] = 100 * 1.01 ** np.arange(n)
    tv = np.tile(np.arange(k, 0, -1, dtype=float), (n, 1))  # 0번이 거래대금 1위
    px = _px(close, close.copy(), tv)
    ev = pd.DataFrame({"code": [px.codes[0]], "day": [px.days[10]], "rank": [2], "persistence": ["structural"]})
    tr, pl, flow = build(px, ev, 5, "next_open")
    assert flow["trades"] == 1 and flow["not_in_universe"] == 0
    assert tr.loc[0, "exit_day"] == px.days[15]
    assert tr.loc[0, "ret"] == pytest.approx(1.01 ** 4 - 1)  # 행 11 시가 → 행 15 종가
    x, info = excess(tr, pl, 5)
    assert info["unmatched"] == 0
    assert 0.0 <= info["self_share"] <= 1.0
    # 대조는 같은 층(0번 포함 가능)의 평균 — ex = ret − ctrl
    assert x.loc[0, "ex"] == pytest.approx(x.loc[0, "ret"] - x.loc[0, "ctrl"])
    # 유니버스 평균 차감: 0번만 움직이므로 평균 = ret / k
    assert x.loc[0, "uhedge"] == pytest.approx(x.loc[0, "ret"] * (1 - 1 / k))
    assert x.loc[0, "grp"] == "hi"
