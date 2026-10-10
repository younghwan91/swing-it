"""research/experiments/meta_gen6_run 의 순수 헬퍼 — 합성 자료, DB·파일 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments import meta_gen4_run as m4
from research.experiments import meta_gen6_run as m6


def _mkt(T: int, seed: int = 0, sd: float = 0.01):
    rng = np.random.default_rng(seed)
    ov = rng.normal(0, sd / 2, T)
    iday = rng.normal(0, sd, T)
    rate = np.full(T, 0.03 / 252)
    return ov, iday, rate


def test_calendar_flags_first_trading_day_of_period_and_row0_false():
    d = pd.DatetimeIndex(["2015-12-30", "2015-12-31", "2016-01-04", "2016-01-05", "2016-02-01", "2016-03-31", "2016-04-01", "2017-01-02"])
    np.testing.assert_array_equal(m6.calendar_flags(d, "Y"), [0, 0, 1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(m6.calendar_flags(d, "Q"), [0, 0, 1, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(m6.calendar_flags(d, "M"), [0, 0, 1, 0, 1, 1, 1, 1])
    d2 = pd.DatetimeIndex(["2016-01-04", "2016-01-05"])                 # 첫날이 기간 첫날이어도 진입이라 거짓
    assert not m6.calendar_flags(d2, "Y").any()


def test_band_limits():
    assert m6.band_limits(0.6, "abs", 0.10) == pytest.approx((0.5, 0.7))
    assert m6.band_limits(0.6, "rel", 0.20) == pytest.approx((0.48, 0.72))
    assert m6.band_limits(0.8, "rel", 0.20) == pytest.approx((0.64, 0.96))
    with pytest.raises(ValueError):
        m6.band_limits(0.6, "x", 0.1)


def test_full_equity_drift_equals_gen4_buy_and_hold():
    """w = 1 표류 = 세대 4 엔진의 매수보유(p = 1, 창 첫날 직전에 이미 보유 — 정정 '목표 비중으로 시작', 진입 비용 없음)."""
    ov, iday, rate = _mkt(300, 1)
    s = m6.simulate(ov, iday, rate, 1.0)
    np.testing.assert_allclose(s["r"], m4.core_returns(ov, iday, rate, np.ones(300), 1.0), atol=1e-13)
    assert not s["reb"].any() and s["turn"].sum() == 0.0


def test_drift_weight_follows_prices_and_no_trades():
    ov, iday, rate = _mkt(200, 2)
    w = 0.6
    s = m6.simulate(ov, iday, rate, w, cost=0.0, fee=0.0)
    E = w * np.cumprod((1 + ov) * (1 + iday))
    K = (1 - w) * np.cumprod(1 + rate)
    np.testing.assert_allclose(s["wc"], E / (E + K), rtol=1e-12)
    np.testing.assert_allclose(np.cumprod(1 + s["r"]), E + K, rtol=1e-12)
    assert s["turn"].sum() == 0.0


def test_scheduled_rebalance_at_open_with_cost():
    """예정 날 t 시가: 밤 수익 반영 뒤 NAV 에서 정확히 w 로, 비용 = 0.07% × |w·N − E|."""
    T, w, cost = 6, 0.5, 0.0007
    ov = np.array([0.0, 0.10, 0.0, 0.0, 0.0, 0.0])
    iday = np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0])
    rate = np.zeros(T)
    reb = np.array([0, 0, 0, 1, 0, 0], bool)
    s = m6.simulate(ov, iday, rate, w, reb=reb, cost=cost, fee=0.0)
    E = w * 1.1 * 1.2                                                    # 창 첫날 직전 w 보유, 진입 비용 없음
    K = 1 - w
    assert s["wc"][2] == pytest.approx(E / (E + K))
    N = E + K
    tr = abs(w * N - E)
    assert s["turn"][3] == pytest.approx(tr / N)
    assert s["wc"][3] == pytest.approx(w)
    assert np.cumprod(1 + s["r"])[3] == pytest.approx(N - cost * tr)
    np.testing.assert_array_equal(s["reb"], reb)


def test_band_triggers_at_close_executes_next_open():
    """종가 비중이 밴드 밖인 날 d → d+1 시가 리밸런스(그날 d 에는 매매 없음, 장중 수익은 표류 비중으로)."""
    T, w = 8, 0.6
    ov = np.zeros(T)
    iday = np.zeros(T)
    iday[3] = 0.60                                                       # 날 3 장중 급등 → 종가 비중 > 0.7
    s = m6.simulate(ov, iday, np.zeros(T), w, band=m6.band_limits(w, "abs", 0.10), cost=0.0, fee=0.0)
    assert s["wc"][3] > 0.7
    assert not s["reb"][3] and s["reb"][4] and s["reb"].sum() == 1
    assert s["wc"][4] == pytest.approx(w)
    assert s["r"][3] == pytest.approx(w * 0.60)                          # 결정 날 수익 = 표류 비중 그대로
    # 밴드 안 움직임은 매매 없음
    iday2 = np.zeros(T)
    iday2[3] = 0.10
    s2 = m6.simulate(ov, iday2, np.zeros(T), w, band=m6.band_limits(w, "abs", 0.10), cost=0.0, fee=0.0)
    assert 0.6 < s2["wc"][3] < 0.7 and not s2["reb"].any()


def test_band_does_not_use_future():
    ov, iday, rate = _mkt(400, 3, 0.03)
    b = m6.band_limits(0.6, "abs", 0.05)
    s = m6.simulate(ov, iday, rate, 0.6, band=b)
    t = 250
    iday2 = iday.copy()
    iday2[t + 1:] = -iday2[t + 1:]
    s2 = m6.simulate(ov, iday2, rate, 0.6, band=b)
    np.testing.assert_array_equal(s["reb"][:t + 2], s2["reb"][:t + 2])  # t 종가까지 같으면 t+1 시가 결정도 같다
    np.testing.assert_allclose(s["r"][:t + 1], s2["r"][:t + 1])
    assert s["reb"].sum() > 3


def test_batch_rows_match_single_runs():
    ov, iday, rate = _mkt(120, 4)
    rng = np.random.default_rng(0)
    P = m6.random_reb_dates(120, 5, 4, rng)
    S = m6.simulate(ov, iday, rate, 0.4, reb=P)
    for i in range(4):
        s = m6.simulate(ov, iday, rate, 0.4, reb=P[i])
        np.testing.assert_allclose(S["r"][i], s["r"])
        np.testing.assert_array_equal(S["reb"][i], P[i])


def test_random_reb_dates_exact_count_uniform_excludes_row0():
    rng = np.random.default_rng(1)
    P = m6.random_reb_dates(50, 7, 4000, rng)
    assert P.shape == (4000, 50)
    assert (P.sum(axis=1) == 7).all() and not P[:, 0].any()
    freq = P[:, 1:].mean(axis=0)
    assert freq == pytest.approx(np.full(49, 7 / 49), abs=0.02)        # 날마다 같은 확률
    assert not m6.random_reb_dates(50, 0, 3, rng).any()
    with pytest.raises(ValueError):
        m6.random_reb_dates(5, 5, 1, rng)


def test_null_rebalance_count_matches_rule():
    """널 경로가 실제로 실행한 리밸런스 수 = 규칙의 횟수(달력 규칙)."""
    d = pd.bdate_range("2010-01-01", "2012-12-31")
    ov, iday, rate = _mkt(len(d), 5)
    s = m6.rule_sim("R2", 0.6, d, ov, iday, rate)
    k = int(s["reb"].sum())
    assert k == 11                                                       # 2010-04 ~ 2012-10 분기 첫날
    N = m6.simulate(ov, iday, rate, 0.6, reb=m6.random_reb_dates(len(d), k, 20, np.random.default_rng(2)))
    assert (N["reb"].sum(axis=1) == k).all()


def test_harvest_bonferroni_and_pick():
    assert m6.harvest_term(0.6, 0.09) == pytest.approx(0.5 * 0.24 * 0.09)
    assert m6.harvest_term(1.0, 0.09) == 0.0
    assert m6.bonferroni_p(0.99) == pytest.approx(1 - 24 * 0.01)          # 정정 뒤 24 칸
    assert m6.bonferroni_p(0.99, 21) == pytest.approx(1 - 21 * 0.01)
    assert m6.bonferroni_p(0.5) == 0.0
    assert m6.pick_fewest(["B10", "R2", "B20"], {"B10": 0.5, "R2": 4.0, "B20": 0.5}) == ["B10", "B20", "R2"]
    assert m6.pick_fewest([], {}) == []


def test_cell_criteria_both_worlds_for_1_to_3():
    ok = {(f"c{i}", c): True for i in range(1, 5) for c in m6.CASHES}
    assert m6.cell_criteria(ok)["all"]
    ok2 = dict(ok)
    ok2[("c2", "zero")] = False
    assert not m6.cell_criteria(ok2)["all"]
    ok3 = dict(ok)
    other = next(c for c in m6.CASHES if c != m6.C4_WORLD)
    ok3[("c4", other)] = False                                          # 조건 4 는 정식 세계만 — 엄격한 읽기는 참고
    c = m6.cell_criteria(ok3)
    assert c["all"] and not c["c4_both"]


def test_year_ordinal_and_anchor_flags():
    d = pd.DatetimeIndex(["2015-12-30", "2015-12-31", "2016-01-04", "2016-01-05", "2016-01-06", "2017-01-02", "2017-01-03"])
    o = m6.year_ordinal(d)
    np.testing.assert_array_equal(o, [0, 1, 0, 1, 2, 0, 1])
    np.testing.assert_array_equal(m6.anchor_flags(o, 0), [0, 0, 1, 0, 0, 1, 0])          # 행 0 은 창 첫날이라 거짓
    np.testing.assert_array_equal(m6.anchor_flags(o, 1), [0, 1, 0, 1, 0, 0, 1])
    F = m6.anchor_flags(o, np.arange(3))
    assert F.shape == (3, 7) and F[2].sum() == 1


def test_anchor_zero_equals_r1_calendar():
    d = pd.bdate_range("2009-11-02", "2013-03-29")
    np.testing.assert_array_equal(m6.anchor_flags(m6.year_ordinal(d), 0), m6.calendar_flags(d, "Y"))


def test_upper_cap_only_triggers_above():
    """R1+C10: 상한만 — 비중이 w + 10%p 를 넘으면 다음 시가, 아래로는 아무리 빠져도 1월까지 그대로."""
    T, w = 8, 0.4
    lo, hi = m6.band_limits(w, "up", 0.10)
    assert lo == -np.inf and hi == pytest.approx(0.5)
    iday = np.zeros(T)
    iday[2] = -0.6                                                       # 급락 → 비중 약 0.21, 하한 없음
    s = m6.simulate(np.zeros(T), iday, np.zeros(T), w, band=(lo, hi), cost=0.0, fee=0.0)
    assert s["wc"][2] < 0.3 and not s["reb"].any()
    iday2 = np.zeros(T)
    iday2[2] = 0.6                                                       # 급등 → 비중 > 0.5
    s2 = m6.simulate(np.zeros(T), iday2, np.zeros(T), w, band=(lo, hi), cost=0.0, fee=0.0)
    assert s2["wc"][2] > 0.5 and s2["reb"][3] and s2["reb"].sum() == 1


def test_calendar_and_band_combined():
    """reb 와 band 를 함께 주면 둘 중 하나로 리밸런스(R1+C10)."""
    d = pd.bdate_range("2010-06-01", "2012-06-29")
    ov, iday, rate = _mkt(len(d), 6, 0.03)
    ra = m6.rule_args(m6.CUR, 0.4, d)
    s = m6.simulate(ov, iday, rate, 0.4, **ra)
    cal = m6.calendar_flags(d, "Y")
    assert s["reb"][cal].all()                                          # 1월 첫 거래일은 늘
    extra = s["reb"] & ~cal
    trig = np.flatnonzero(extra)
    assert len(trig) and (s["wc"][trig - 1] > 0.5).all()                 # 그 밖은 전날 종가 비중이 상한 위였던 날만


def test_random_reb_dates_keys_per_row_k_and_nesting():
    rng = np.random.default_rng(3)
    keys = rng.random((50, 99))
    a = m6.random_reb_dates(100, 3, 50, keys=keys)
    b = m6.random_reb_dates(100, 8, 50, keys=keys)
    assert (a.sum(axis=1) == 3).all() and (b.sum(axis=1) == 8).all()
    assert not (a & ~b).any()                                            # 같은 키 → 작은 k 의 날 ⊂ 큰 k 의 날
    k = np.arange(50) % 5
    c = m6.random_reb_dates(100, k, 50, keys=keys)
    np.testing.assert_array_equal(c.sum(axis=1), k)


def test_simulate_market_rows_match_single():
    """경로마다 다른 시장(섞은 수익열)을 (n, T) 로 넘겨도 행마다 따로 돌린 것과 같다."""
    T = 150
    ov, iday, rate = _mkt(T, 7)
    rng = np.random.default_rng(1)
    P = np.stack([rng.permutation(T) for _ in range(3)])
    b = m6.band_limits(0.6, "abs", 0.05)
    S = m6.simulate(ov[P], iday[P], rate, 0.6, band=b)
    for i in range(3):
        s = m6.simulate(ov[P[i]], iday[P[i]], rate, 0.6, band=b)
        np.testing.assert_allclose(S["r"][i], s["r"])
        np.testing.assert_array_equal(S["reb"][i], s["reb"])


def test_family_max_pct():
    rng = np.random.default_rng(4)
    S = rng.normal(size=(1, 1000))
    fm = m6.family_max_pct(S)
    assert np.percentile(fm, 95) == pytest.approx(0.95, abs=0.002)        # 칸 하나면 명목 95%
    S2 = rng.normal(size=(6, 1000))                                     # 독립 6 칸 → 문턱이 올라간다(0.95^(1/6) 근처)
    assert np.percentile(m6.family_max_pct(S2), 95) == pytest.approx(0.95 ** (1 / 6), abs=0.005)
    S3 = np.repeat(S, 6, axis=0)                                        # 완전 상관 6 칸 → 칸 하나와 같다
    assert np.percentile(m6.family_max_pct(S3), 95) == pytest.approx(0.95, abs=0.002)


def test_rule_verdict_drops_degenerate():
    cells = {(nm, w): {"all": True} for nm in ("B20",) for w in m6.W_GRID}
    cells[("B20", 0.8)] = {"all": False}
    assert m6.rule_verdict(cells, {("B20", 0.8)}, "B20") == (True, [0.4, 0.6])
    assert m6.rule_verdict(cells, set(), "B20") == (False, [0.4, 0.6, 0.8])
    assert m6.rule_verdict(cells, {("B20", w) for w in m6.W_GRID}, "B20") == (False, [])
