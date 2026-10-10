"""research/experiments/meta_gen4_run 의 순수 헬퍼 — 합성 자료, DB·파일 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments import meta_gen4_run as m4


def test_ma_signal_uses_only_past_and_today():
    rng = np.random.default_rng(0)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
    x = m4.ma_signal(c, 50)
    assert not x[:49].any()                                   # 50일 미만 = 이동평균 없음 = 현금
    d = 120
    assert x[d] == (c[d] > c[d - 49:d + 1].mean())            # 그날 종가 포함 뒤쪽 50일
    c2 = c.copy()
    c2[d + 1:] *= rng.uniform(0.5, 1.5, len(c) - d - 1)       # 미래를 바꿔도 과거 신호는 그대로
    np.testing.assert_array_equal(m4.ma_signal(c2, 50)[:d + 1], x[:d + 1])


def test_held_is_signal_shifted_one_day():
    x = np.array([0, 1, 1, 0, 1], bool)
    np.testing.assert_array_equal(m4.held(x, 1, 5), [0, 1, 1, 0])     # p_t = x_{t−1}
    with pytest.raises(ValueError):
        m4.held(x, 0, 3)


def test_core_returns_hand_computed():
    ov = np.array([0.01, 0.02, -0.01])
    iday = np.array([0.03, -0.01, 0.02])
    rate = np.full(3, 0.0001)
    p = np.array([1.0, 1.0, 0.0])
    r = m4.core_returns(ov, iday, rate, p, 0.0, cost=0.001, fee=0.0252)
    fee_d = 0.0252 / 252
    e0 = (1 + 0.0001) * (1 - 0.001) * (1 + 0.03 - fee_d) - 1         # 밤 = 현금, 시가 매수(비용), 장중 보유
    e1 = (1 + 0.02) * 1.0 * (1 + (-0.01 - fee_d)) - 1                # 계속 보유
    e2 = (1 - 0.01) * (1 - 0.001) * 1.0 - 1                          # 밤 보유, 시가 매도(비용), 장중 현금
    np.testing.assert_allclose(r, [e0, e1, e2])
    # 경로 묶음(n × T) 도 줄마다 같은 답
    R = m4.core_returns(ov, iday, rate, np.vstack([p, p]), 0.0, cost=0.001, fee=0.0252)
    np.testing.assert_allclose(R, np.vstack([r, r]))


def test_held_positions_ignore_same_day_close():
    """보유 p_t 는 전날 신호 — 날 t 종가 충격은 t 의 보유를 바꾸지 못하고 t+1 부터만 바꿀 수 있다."""
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    x = m4.ma_signal(c, 20)
    p = m4.held(x, 100, 300)
    c2 = c.copy()
    c2[250] *= 1.2                                                   # 250일 종가 충격
    p2 = m4.held(m4.ma_signal(c2, 20), 100, 300)
    np.testing.assert_array_equal(p[:151], p2[:151])                 # 보유 행 0..150 = 날 100..250 → 250 일 보유는 249 신호


def test_dual_momentum_month_end_hold():
    d = pd.bdate_range("2015-01-01", "2017-12-31")
    mon = (d.year * 12 + d.month - (d[0].year * 12 + d[0].month)).to_numpy()
    c = np.linspace(100, 200, len(d))                                 # 계속 오름 → 12개월 뒤부터 참
    x = m4.dual_momentum_signal(c, np.full(len(d), 0.0001), mon)
    me = np.flatnonzero(np.diff(mon) != 0)
    assert not x[:me[12]].any() and x[me[12]]
    for k in range(12, len(me) - 1):                                  # 월말부터 다음 월말 전날까지 같은 값
        assert len(set(x[me[k]:me[k + 1]])) == 1
    c_dn = np.linspace(200, 100, len(d))
    assert not m4.dual_momentum_signal(c_dn, np.full(len(d), 0.0001), mon).any()


def test_markov_probs_and_degenerate():
    p01, p10 = m4.markov_probs(0.6, 0.02)
    assert 0.6 * p10 == pytest.approx(0.01) and 0.4 * p01 == pytest.approx(0.01)
    assert m4.markov_probs(1.0, 0.0) == (0.0, 0.0)
    P = m4.markov_paths(5, 50, 1.0, 0.0, np.random.default_rng(0))
    assert P.all()
    P = m4.markov_paths(5, 50, 0.0, 0.0, np.random.default_rng(0))
    assert not P.any()


def test_markov_null_matches_in_market_and_switch_rate():
    pi, s = 0.62, 0.03
    P = m4.markov_paths(2000, 2000, pi, s, np.random.default_rng(7)).astype(float)
    assert P.mean() == pytest.approx(pi, abs=0.01)
    assert (P[:, 1:] != P[:, :-1]).mean() == pytest.approx(s, rel=0.03)
    pi_r, s_r = m4.timing_stats(np.array([0, 1, 1, 1, 0, 0, 1, 1.0]))
    assert pi_r == pytest.approx(5 / 8) and s_r == pytest.approx(3 / 7)


def test_div_adjust_only_out_of_market():
    r = np.array([0.01, 0.02, -0.01])
    p = np.array([1.0, 0.0, 0.5])
    np.testing.assert_allclose(m4.div_adjust(r, p, 0.0252), r - 0.0001 * np.array([0.0, 1.0, 0.5]))
    # 1년(252일) 내내 밖이면 CAGR 이 연 배당만큼 줄어든다
    z = np.zeros(252)
    out = m4.div_adjust(z, np.zeros(252), 0.018)
    assert np.prod(1 + out) - 1 == pytest.approx(-0.018, abs=2e-4)


def test_path_metrics_matches_equity_stats():
    from research.signals import meta_fast as mf

    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, 500)
    mon = np.repeat(np.arange(25), 20)
    m = m4.path_metrics(r[None, :], mon, np.full(500, 0.0001))
    e = mf.equity_stats(r)
    assert m["cagr"][0] == pytest.approx(e["cagr"]) and m["mdd"][0] == pytest.approx(e["mdd"])
    assert m["calmar"][0] == pytest.approx(e["calmar"])


def test_paired_boot_identical_and_dominant():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0005, 0.01, 600)
    mon = np.repeat(np.arange(30), 20)
    rate = np.full(600, 0.0001)
    B = m4.MonthBoot(mon, 3, 200, 0)
    b = m4.paired_boot(r, r, rate, B)
    assert b["p_sharpe"] == 0.0 and b["p_calmar"] == 0.0 and b["p_both"] == 0.0   # 같으면 '더 큼' 확률 0
    b2 = m4.paired_boot(r + 0.002, r, rate, B)
    assert b2["p_sharpe"] == 1.0 and b2["p_calmar"] > 0.9 and b2["p_both"] <= min(b2["p_sharpe"], b2["p_calmar"])


def test_month_boot_resamples_daily_path_in_month_blocks():
    """일별 경로를 달 블록으로 재표본 — 한 달 안 급락·회복이 월말 값으로 사라지지 않는다."""
    mon = np.repeat(np.arange(6), [20, 21, 19, 22, 20, 21])
    r = np.zeros(len(mon))
    r[45], r[50] = -0.30, 0.30 / 0.70                         # 2번째 달 안에서 −30% 뒤 회복(월말 수익 0)
    B = m4.MonthBoot(mon, 2, 300, 1)
    assert (B.L == np.diff(np.r_[0, np.cumsum(np.bincount(mon))])[B.J].sum(axis=1)).all()
    has = (B.J == 2).any(axis=1)
    assert has.any() and (~has).any()
    eq = np.cumprod(1 + np.append(r, 0.0)[B.IDX], axis=1)
    mdd = (eq / np.maximum.accumulate(np.maximum(eq, 1.0), axis=1) - 1).min(axis=1)
    assert np.allclose(mdd[has], -0.30) and np.allclose(mdd[~has], 0.0)   # 그 달이 뽑힌 표본만 일별 MDD −30% 를 본다
    r2 = r + 0.001
    cal = B.calmar(r2)
    for i in np.flatnonzero(has)[:5]:                         # 표본 경로 그대로 잰 Calmar 와 같다(채운 0 은 영향 없음)
        x = r2[B.IDX[i, :B.L[i]]]
        m = m4.path_metrics(x[None, :], np.zeros(len(x), int), np.zeros(len(x)))
        assert cal[i] == pytest.approx(m["calmar"][0])
    assert np.allclose(m4.monthly_rows(r, mon)[2], 0.0)       # 월말 값으로는 0


def test_excess_sharpe_subtracts_cash():
    mon = np.repeat(np.arange(24), 21)
    rate = np.full(len(mon), 0.0002)
    rng = np.random.default_rng(5)
    r = rng.normal(0.0003, 0.01, len(mon))
    ex = m4.excess_sharpe(r, rate, mon)[0]
    M = m4.monthly_rows(r, mon) - m4.monthly_rows(rate, mon)
    assert ex == pytest.approx(M.mean() / M.std() * np.sqrt(12))
    assert np.isnan(m4.excess_sharpe(rate, rate, mon)[0])     # 현금 그 자체 = 초과 0 → 정의 안 됨


def test_div_accrue_and_after_tax_direction():
    r = np.zeros(252)
    assert np.prod(1 + m4.div_accrue(r, np.ones(252), 0.018)) - 1 == pytest.approx(0.018, abs=2e-4)
    np.testing.assert_allclose(m4.div_accrue(r, np.zeros(252)), r)          # 밖이면 배당 없음


def test_markov_month_paths_switch_only_at_month_starts():
    mon = np.repeat(np.arange(60), 21)
    p = np.repeat(np.random.default_rng(2).random(60) < 0.6, 21).astype(float)
    P = m4.markov_month_paths(300, p, mon, np.random.default_rng(3))
    ms = m4.month_starts(mon)
    sw = np.flatnonzero((P[:, 1:] != P[:, :-1]).any(axis=0)) + 1
    assert set(sw) <= set(ms)                                 # 전환은 달 첫날에만


def test_circular_paths_preserve_spells():
    p = np.r_[np.ones(100), np.zeros(37), np.ones(5), np.zeros(58)]
    C = m4.circular_paths(p, 50, np.random.default_rng(0), min_shift=10)
    assert np.allclose(C.mean(axis=1), p.mean())
    assert not (C == p).all(axis=1).any()
    for row in C:
        assert any(np.array_equal(row, np.roll(p, k)) for k in range(10, len(p) - 10))


def test_pick_option_tie_fewer_switches():
    assert m4.pick_option([0.1, 0.3, 0.3], [5, 9, 2]) == 2
    assert m4.pick_option([0.1, 0.3, 0.2], [5, 9, 2]) == 1
    assert m4.pick_option([np.nan, -0.1], [0, 4]) == 1
    assert m4.pick_option([0.2, 0.2], [3, 3]) == 0


def test_drawdown_events():
    eq = np.array([1.0, 1.1, 0.8, 0.9, 1.2, 1.15, 1.25, 0.9, 1.0])
    ev = m4.drawdown_events(eq, 0.15)
    assert [(a, t) for a, t, _ in ev] == [(1, 2), (6, 7)]
    assert ev[0][2] == pytest.approx(0.8 / 1.1 - 1)


def test_core_sat_mix_rebalance_and_cost():
    mon = np.array([0, 0, 1])
    rc = np.array([0.10, 0.0, 0.0])
    rs = np.zeros(3)
    out = m4.core_sat_mix(rc, np.ones(3), rs, mon, w=0.8, cost=0.01)
    assert out[0] == pytest.approx(0.08)
    wc = 0.88 / 1.08                                                             # 표류한 코어 비중
    assert out[2] == pytest.approx(-(0.01 * abs(wc - 0.8)))
    out0 = m4.core_sat_mix(rc, np.zeros(3), rs, mon, w=0.8, cost=0.01)           # 코어가 현금이면 되돌림 비용 없음
    assert out0[2] == pytest.approx(0.0)


def test_p1_with_cash_reproduces_reviewer_construction():
    d = pd.to_datetime(["2020-12-30", "2021-01-04", "2021-01-05"])
    nav = pd.Series([1.01, 1.0201, 1.0201], index=d)
    rate = np.array([0.001, 0.002, 0.002])
    out = m4.p1_with_cash(nav, {2020: 0.25, 2021: 0.5}, rate)
    np.testing.assert_allclose(out, [0.01 + 0.75 * 0.001, 0.01 + 0.5 * 0.002, 0.0 + 0.5 * 0.002])


def test_criteria_and_simplest():
    st0 = {"cagr": 0.10, "mdd": -0.40, "calmar": 0.25}
    st = {"cagr": 0.09, "cagr_div": 0.085, "mdd": -0.28, "calmar": 0.32}
    boot = {"p_sharpe": 0.85, "p_calmar": 0.81, "p_both": 0.80}
    c = m4.criteria(st, st0, boot, 0.30, (0.3, 1.0), (0.2, 1.0))
    assert c["all"]
    assert not m4.criteria(st, st0, {**boot, "p_both": 0.79}, 0.30, (0.3, 1.0), (0.2, 1.0))["c1"]       # 동시 읽기
    assert m4.criteria(st, st0, {**boot, "p_both": 0.79}, 0.30, (0.3, 1.0), (0.2, 1.0), joint=False)["c1"]
    assert not m4.criteria(st, st0, {**boot, "p_calmar": 0.79}, 0.30, (0.3, 1.0), (0.2, 1.0), joint=False)["c1"]
    assert not m4.criteria({**st, "mdd": -0.31}, st0, boot, 0.30, (0.3, 1.0), (0.2, 1.0))["c2"]
    assert not m4.criteria({**st, "cagr_div": 0.079}, st0, boot, 0.30, (0.3, 1.0), (0.2, 1.0))["c3"]
    assert not m4.criteria(st, st0, boot, 0.33, (0.3, 1.0), (0.2, 1.0))["c4"]
    assert not m4.criteria(st, st0, boot, 0.30, (0.3, 0.9), (0.2, 1.0))["c5"]
    rows = [{"proc": "C1", "n_params": 1, "sw_py": 1.0, "crit": {"all": True}}, {"proc": "C4", "n_params": 0, "sw_py": 1.3, "crit": {"all": True}},
            {"proc": "C1f", "n_params": 0, "sw_py": 6.8, "crit": {"all": False}}]
    assert [r["proc"] for r in m4.simplest_pass(rows)] == ["C4"]
    rows[2]["crit"] = {"all": True}
    assert [r["proc"] for r in m4.simplest_pass(rows)] == ["C4", "C1f"]                    # 동률 → 전환 적은 쪽
    assert m4.simplest_pass([{**r, "crit": {"all": False}} for r in rows]) == []
