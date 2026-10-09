"""스윙 라이브 포트폴리오 백테스트 — 현금·슬롯·비용(분할 청산)·데이터 끝·낙폭 킬·성과 지표."""

import math

import numpy as np
import pandas as pd
import pytest

from swing_it.live.backtest import drawdown, perf_stats, run_backtest
from swing_it.live.config import StrategyConfig

CFG = StrategyConfig()


def _flat(n_days=40, codes=("000010", "000020", "000030")):
    days = pd.bdate_range("2025-01-01", periods=n_days)
    one = pd.DataFrame(100.0, index=days, columns=list(codes))
    frame = {"open": one.copy(), "high": one * 1.005, "low": one * 0.995, "close": one.copy(),
             "volume": one * 0 + 1_000.0}
    const = lambda v: pd.DataFrame(v, index=days, columns=list(codes))  # noqa: E731
    pn = {**frame, "market": pd.Series("KOSPI", index=list(codes)), "ma50": const(90.0),
          "universe": const(False), "tt": const(False), "breakout": const(False), "rs": const(80.0),
          "breadth": const(0.6), "pead_rank": const(0.8), "rel": const(0.5), "F": const(1.0),
          "adv20": const(10_000.0), "pivot": const(99.0), "contraction_low": const(95.0)}
    return days, pn


def _event(pn, day, code, rel=0.5):
    for k in ("universe", "tt", "breakout"):
        pn[k].loc[day, code] = True
    pn["rel"].loc[day, code] = rel


def test_single_trade_cash_accounting_with_split_exit_costs():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    # d = 1 − 95×0.99/100 = 0.0595, target = 111.9. 진입 days[3] 시가 100, days[5] 고가 112 → 절반 익절,
    # days[7] 종가 < MA → days[8] 시가 105 에 나머지.
    pn["high"].loc[days[5], "000010"] = 112.0
    pn["close"].loc[days[5]:, "000010"] = 106.0
    pn["open"].loc[days[6]:, "000010"] = 106.0
    pn["high"].loc[days[6]:, "000010"] = 107.0
    pn["low"].loc[days[6]:, "000010"] = 105.5
    pn["ma50"].loc[days[7], "000010"] = 110.0
    pn["open"].loc[days[8], "000010"] = 105.0
    rate = 0.004
    res = run_backtest(pn, CFG, initial_equity=1e8, cost_fn=lambda d, m: rate)
    t = res.trades
    assert len(t) == 1
    r = t.iloc[0]
    qty = 100_000.0                                        # 10% 상한 1,000만/100원(위험 기준 168,067주보다 작다)
    tgt = 100 * (1 + 2 * (1 - 95 * 0.99 / 100))
    assert r["qty"] == qty and r["entry_day"] == days[3] and r["exit_day"] == days[8]
    assert r["exit_reason"] == "trail" and r["partial_hit"]
    assert r["ret"] == pytest.approx(0.5 * (tgt / 100 - 1) + 0.5 * (105 / 100 - 1))
    cost = rate * (50_000 * tgt + 50_000 * 105)                   # 매도한 부분마다
    assert r["cost_krw"] == pytest.approx(cost)
    pnl = 50_000 * tgt + 50_000 * 105 - cost - qty * 100
    assert r["pnl_krw"] == pytest.approx(pnl)
    assert res.equity.iloc[-1] == pytest.approx(1e8 + pnl)
    # 보유 중 자산 = 현금 + 수량 × 종가
    assert res.daily.loc[days[4], "equity"] == pytest.approx(1e8 - qty * 100 + qty * 100)
    assert res.daily.loc[days[4], "invested"] == pytest.approx(qty * 100)


def test_slot_limit_takes_strongest_contraction():
    days, pn = _flat()
    _event(pn, days[2], "000010", rel=0.9)
    _event(pn, days[2], "000020", rel=0.3)
    res = run_backtest(pn, StrategyConfig(max_positions=1), cost_fn=None)
    assert list(res.trades["code"]) == ["000020"]
    assert res.skips["skip_slots_cash"] == 1


def test_cash_constraint_shrinks_quantity():
    days, pn = _flat()
    for c in ("000010", "000020", "000030"):
        _event(pn, days[2], c)
    # 종목당 상한 45%·위험 100% 면 종목당 45,000주(450만) — 셋째는 현금(1,000만 − 2×45,000×지정가 103.95)만큼만.
    res = run_backtest(pn, StrategyConfig(max_position_frac=0.45, risk_per_trade=1.0), initial_equity=1e7,
                       cost_fn=None)
    q = res.trades.set_index("code")["qty"]
    assert q.sum() * 103.95 <= 1e7 + 1e-6
    assert len(q) == 3 and q.iloc[-1] < q.iloc[0]


def test_held_code_not_reentered_and_counted():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    _event(pn, days[5], "000010")
    res = run_backtest(pn, CFG, cost_fn=None)
    assert len(res.trades) == 1 and res.skips["skip_held"] == 1


def test_chase_and_halt_skip_counts():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    _event(pn, days[2], "000020")
    pn["open"].loc[days[3], "000010"] = 99 * 1.05 + 0.01       # 추격 금지
    pn["volume"].loc[days[3], "000020"] = 0.0                   # 정지
    res = run_backtest(pn, CFG, cost_fn=None)
    assert res.trades.empty
    assert res.skips["skip_chase"] == 1 and res.skips["skip_no_open"] == 1


def test_below_low_round_trip_pays_cost_only():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    pn["contraction_low"].loc[days[2], "000010"] = 100.0       # 시가 100 ≤ 수축 저점
    res = run_backtest(pn, CFG, initial_equity=1e8, cost_fn=lambda d, m: 0.004)
    r = res.trades.iloc[0]
    assert r["exit_reason"] == "below_low" and r["ret"] == 0.0 and r["exit_day"] == r["entry_day"]
    assert r["pnl_krw"] == pytest.approx(-0.004 * r["qty"] * 100)


def test_data_end_closes_at_last_tradable_close():
    days, pn = _flat(n_days=12)
    _event(pn, days[2], "000010")
    pn["close"].loc[days[-1], "000010"] = 101.0
    pn["volume"].loc[days[-1], "000010"] = 0.0                  # 마지막 날 정지 → 그 전날 종가
    res = run_backtest(pn, CFG, cost_fn=None)
    r = res.trades.iloc[0]
    assert r["exit_reason"] == "data_end" and r["exit_day"] == days[-2] and r["exit_px"] == 100.0
    assert res.daily["n_pos"].iloc[-1] == 0


def test_delisted_exit_on_last_tradable_day():
    days, pn = _flat(n_days=20)
    _event(pn, days[2], "000010")
    for k in ("open", "high", "low", "close", "volume"):
        pn[k].loc[days[10]:, "000010"] = np.nan
    res = run_backtest(pn, CFG, cost_fn=None)
    r = res.trades.iloc[0]
    assert r["exit_reason"] == "delisted" and r["exit_day"] == days[9]


def test_drawdown_kill_stops_new_entries():
    days, pn = _flat(n_days=30)
    _event(pn, days[2], "000010")
    pn["low"].loc[days[4], "000010"] = 50.0                      # 손절
    pn["open"].loc[days[4], "000010"] = 50.0                     # 갭 → 시가 50 = −50% 손실
    _event(pn, days[10], "000020")
    killed = run_backtest(pn, StrategyConfig(max_position_frac=1.0, risk_per_trade=1.0), cost_fn=None,
                          dd_kill=0.15)
    assert killed.params["kills"] == [[str(days[4].date()), None]]       # 60일 안이라 재개 없음
    assert list(killed.trades["code"]) == ["000010"]
    free = run_backtest(pn, StrategyConfig(max_position_frac=1.0, risk_per_trade=1.0), cost_fn=None)
    assert list(free.trades["code"]) == ["000010", "000020"]


def test_signal_window_limits_only_signals():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    _event(pn, days[20], "000020")
    res = run_backtest(pn, CFG, start=days[10], cost_fn=None)
    assert list(res.trades["code"]) == ["000020"]
    assert res.daily.index[0] == days[10]


def test_perf_stats_basic():
    idx = pd.bdate_range("2020-01-01", periods=253)
    eq = pd.Series(np.linspace(100.0, 121.0, 253), index=idx)
    eq.iloc[100] = 90.0
    daily = pd.DataFrame({"equity": eq, "invested": eq * 0.5, "n_pos": 3, "buys": 0.0, "sells": 0.0})
    s = perf_stats(daily, bench=pd.Series(np.linspace(1.0, 1.1, 253), index=idx))
    assert s["cagr"] == pytest.approx(0.21, rel=1e-9)
    assert s["mdd"] == pytest.approx(drawdown(eq).min())
    assert s["exposure"] == pytest.approx(0.5)
    assert s["bench"]["total_return"] == pytest.approx(0.1)
    assert set(s["years"]) == {2020, 2020 + (idx[-1].year - 2020)} or 2020 in s["years"]
    assert math.isfinite(s["sharpe"])


def test_stop_slippage_applies_to_stop_fills_only_in_cash():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    pn["low"].loc[days[5], "000010"] = 90.0                      # 손절 터치(손절가 94.05)
    res0 = run_backtest(pn, StrategyConfig(stop_slippage_bps=0.0), cost_fn=None)
    res50 = run_backtest(pn, StrategyConfig(stop_slippage_bps=50.0), cost_fn=None)
    a, b = res0.trades.iloc[0], res50.trades.iloc[0]
    assert a["exit_reason"] == b["exit_reason"] == "stop"
    assert a["ret"] == b["ret"]                                  # 총수익(연구 정의)은 미끄러짐 전
    stop_px = a["stop_px"]
    assert b["slip_krw"] == pytest.approx(b["qty"] * stop_px * 0.005)
    assert a["pnl_krw"] - b["pnl_krw"] == pytest.approx(b["slip_krw"])
    assert res0.equity.iloc[-1] - res50.equity.iloc[-1] == pytest.approx(b["slip_krw"])
    # 패리티 모드는 미끄러짐 0
    u = run_backtest(pn, StrategyConfig(stop_slippage_bps=50.0), unlimited=True)
    assert u.trades.iloc[0]["slip_krw"] == 0.0


def test_drawdown_kill_resumes_after_days_or_recovery():
    days, pn = _flat(n_days=120)
    _event(pn, days[2], "000010")
    pn["low"].loc[days[4], "000010"] = 50.0
    pn["open"].loc[days[4], "000010"] = 50.0                     # −50% → 킬 days[4]
    _event(pn, days[30], "000020")                               # 킬 중 → 무시
    _event(pn, days[70], "000030")                               # days[4]+60 = days[64] 재개 뒤 → 진입
    cfg = StrategyConfig(max_position_frac=1.0, risk_per_trade=1.0)
    res = run_backtest(pn, cfg, cost_fn=None, dd_kill=0.15, kill_resume_days=60)
    assert res.params["kills"] == [[str(days[4].date()), str(days[64].date())]]
    assert list(res.trades["code"]) == ["000010", "000030"]
    # 회복 재개: 보유 포지션이 올라 고점 −5% 안으로 돌아오면 60일 전에도 재개
    days, pn = _flat(n_days=60)
    _event(pn, days[2], "000010")
    pn["close"].loc[days[4], "000010"] = 80.0                    # 평가 −20% → 킬(손절가 94.05 아래지만 저가는 안 건드림)
    pn["close"].loc[days[5]:, "000010"] = 97.0                   # 회복 → 고점 −3% → 재개 days[5]
    pn["low"].loc[days[4]:, "000010"] = 95.0                     # 손절 터치 없음
    _event(pn, days[8], "000020")
    res = run_backtest(pn, cfg, cost_fn=None, dd_kill=0.15, kill_resume_days=60, kill_resume_recovery=0.05)
    assert res.params["kills"][0] == [str(days[4].date()), str(days[5].date())]
    assert "000020" in list(res.trades["code"])


def test_ret_half_is_research_definition_regardless_of_integer_partial():
    days, pn = _flat()
    _event(pn, days[2], "000010")
    pn["high"].loc[days[5], "000010"] = 112.0
    pn["close"].loc[days[5]:, "000010"] = 106.0
    pn["open"].loc[days[6]:, "000010"] = 106.0
    pn["high"].loc[days[6]:, "000010"] = 107.0
    pn["low"].loc[days[6]:, "000010"] = 105.5
    pn["ma50"].loc[days[7], "000010"] = 110.0
    pn["open"].loc[days[8], "000010"] = 105.0
    # 3주 포지션 → 익절 1주(1/3): ret ≠ ret_half
    res = run_backtest(pn, StrategyConfig(max_position_frac=0.03, risk_per_trade=1.0), initial_equity=10_000.0,
                       cost_fn=None)
    r = res.trades.iloc[0]
    tgt = 100 * (1 + 2 * (1 - 95 * 0.99 / 100))
    assert r["qty"] == 3
    assert r["ret"] == pytest.approx((1 / 3) * (tgt / 100 - 1) + (2 / 3) * 0.05)
    assert r["ret_half"] == pytest.approx(0.5 * (tgt / 100 - 1) + 0.5 * 0.05)


def test_perf_stats_monthly_sharpe():
    idx = pd.bdate_range("2020-01-01", periods=400)
    eq = pd.Series(np.linspace(100.0, 130.0, 400), index=idx)
    daily = pd.DataFrame({"equity": eq, "invested": 0.0, "n_pos": 0, "buys": 0.0, "sells": 0.0})
    s = perf_stats(daily)
    assert math.isfinite(s["sharpe_monthly"]) and s["sharpe_monthly"] > 0


def test_failed_breakout_exit_reason_and_next_open_fill():
    days, pn = _flat()
    _event(pn, days[2], "000010")                                # pivot 99, 진입 days[3] 시가 100
    pn["close"].loc[days[5], "000010"] = 98.0                    # 종가 < 피벗 → days[6] 시가
    pn["open"].loc[days[6], "000010"] = 97.0
    res = run_backtest(pn, CFG, cost_fn=None)
    r = res.trades.iloc[0]
    assert r["exit_reason"] == "failed_breakout" and r["exit_day"] == days[6] and r["exit_px"] == 97.0
    assert r["ret"] == pytest.approx(-0.03)
    old = run_backtest(pn, StrategyConfig(failed_breakout=False, time_stop_days=20), cost_fn=None)
    assert old.trades.iloc[0]["exit_reason"] == "time_stop"     # 옛 프로토콜: 피벗 아래 종가는 신호가 아니다(20일 시간 손절)
