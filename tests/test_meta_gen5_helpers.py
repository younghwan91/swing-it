"""research/experiments/meta_gen5_run 의 순수 헬퍼 — 합성 자료, DB·파일 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments import meta_gen4_run as m4
from research.experiments import meta_gen5_run as m5


def _mon(d: pd.DatetimeIndex) -> np.ndarray:
    return (d.year * 12 + d.month - (d[0].year * 12 + d[0].month)).to_numpy()


def test_ma_month_signal_equals_daily_signal_at_month_end_and_holds():
    d = pd.bdate_range("2015-01-01", "2017-12-31")
    mon = _mon(d)
    rng = np.random.default_rng(0)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.015, len(d)))
    x = m5.ma_month_signal(c, 50, mon)
    daily = m4.ma_signal(c, 50)
    me = np.flatnonzero(np.diff(mon) != 0)
    assert not x[:me[0]].any()                                        # 첫 월말 전 = 현금
    for k in range(len(me)):                                          # 월말 값 = 그날 일 신호, 다음 월말 전날까지 유지
        nxt = me[k + 1] if k + 1 < len(me) else len(c)
        assert (x[me[k]:nxt] == daily[me[k]]).all()
    assert not x[:49].any()                                           # 이동평균 없는 월말 = 거짓
    held = m4.held(x, 1, len(c))                                      # 보유는 다음 달 첫 거래일부터만 바뀐다
    ch = np.flatnonzero(held[1:] != held[:-1]) + 2                    # 바뀐 날(전체 행)
    assert np.isin(ch, me + 1).all()


def test_ma_month_signal_ignores_future():
    d = pd.bdate_range("2015-01-01", "2016-12-31")
    mon = _mon(d)
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(d)))
    x = m5.ma_month_signal(c, 40, mon)
    me = np.flatnonzero(np.diff(mon) != 0)
    t = me[10]
    c2 = c.copy()
    c2[t + 1:] *= rng.uniform(0.5, 1.5, len(c) - t - 1)
    x2 = m5.ma_month_signal(c2, 40, mon)
    np.testing.assert_array_equal(x2[:me[11]], x[:me[11]])           # 다음 월말 전까지 같다


def test_top_drawdowns_order_and_recovery():
    eq = np.array([1.0, 1.1, 0.8, 0.9, 1.2, 1.15, 1.25, 0.9, 1.0, 1.3, 1.2])
    ev = m5.top_drawdowns(eq, 5)
    assert [(e["peak"], e["trough"], e["recover"]) for e in ev] == [(6, 7, 9), (1, 2, 4), (9, 10, None), (4, 5, 6)]
    assert ev[0]["dd"] == pytest.approx(0.9 / 1.25 - 1)
    assert len(m5.top_drawdowns(eq, 2)) == 2
    assert m5.top_drawdowns(np.array([1.0, 1.1, 1.2]), 5) == []          # 낙폭 없음


def test_span_path():
    r = np.array([0.1, -0.2, 0.05, -0.1, 0.3])
    p = np.array([1, 1, 0, 1, 1], float)
    s = m5.span_path(r, p, 1, 4)                                       # 날 수익 행 [1, 4)
    e = np.cumprod(1 + r[1:4])
    assert s["ret"] == pytest.approx(e[-1] - 1)
    assert s["dd"] == pytest.approx((e / np.maximum.accumulate(np.r_[1.0, e])[1:] - 1).min())
    assert s["tim"] == pytest.approx(2 / 3)
    z = m5.span_path(r, p, 2, 2)                                       # 빈 구간
    assert z["ret"] == 0.0 and z["dd"] == 0.0 and np.isnan(z["tim"])


def test_calendar_cagr():
    assert m5.calendar_cagr(1.0, pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")) == pytest.approx(2 ** (365.25 / 366) - 1)
    assert np.isnan(m5.calendar_cagr(-1.0, pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31")))


def test_dual_momentum_zero_cash_is_positive_12m_return():
    """현금 0% 판정 세계: C4 문턱 = 0 → 12개월 가격 수익 > 0 과 같다."""
    d = pd.bdate_range("2014-01-01", "2017-12-31")
    mon = _mon(d)
    rng = np.random.default_rng(2)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.012, len(d)))
    x = m4.dual_momentum_signal(c, np.zeros(len(d)), mon)
    me = np.flatnonzero(np.diff(mon) != 0)
    for k in range(12, len(me)):
        assert x[me[k]] == (c[me[k]] > c[me[k - 12]])
