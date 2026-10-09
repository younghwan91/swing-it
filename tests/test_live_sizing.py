"""스윙 라이브 사이징 — 위험 1%·10% 상한·정수 주·슬롯·현금·순위(설계 §2)."""

import math

import pandas as pd
import pytest

from swing_it.live.config import StrategyConfig
from swing_it.live.sizing import allocate, position_size, rank_plans, stop_width
from swing_it.live.strategy import EntryPlan

CFG = StrategyConfig()


def _plan(code, rel, rs, qty=100, limit_px=10_000.0):
    return EntryPlan(code=code, signal_day=pd.Timestamp("2025-01-02"), market="KOSPI", limit_px=limit_px,
                     limit_up_px=13_000.0, qty=qty, ref_px=10_000.0, d_est=0.05, pivot=9_500.0,
                     contraction_low=9_400.0, rel=rel, rs=rs)


def test_stop_width_clip_and_bad_input():
    assert stop_width(100.0, 95.0, CFG) == pytest.approx(1 - 95 * 0.99 / 100)
    assert stop_width(100.0, 99.5, CFG) == 0.03
    assert stop_width(100.0, 10.0, CFG) == 0.10
    assert stop_width(float("nan"), 95.0, CFG) == 0.10


def test_position_size_risk_bound():
    # 위험 1% × 1억 = 100만 / (10,000 × 0.10) = 1,000주 = 1,000만(10%) — 두 한도가 같다
    assert position_size(1e8, 10_000.0, 0.10, CFG) == 1_000
    # d 3% → 위험 기준 3,333주 > 상한 1,000주
    assert position_size(1e8, 10_000.0, 0.03, CFG) == 1_000
    # 자산 작으면 내림: 1,234,567 × 1% / (10,000 × 0.1) = 12.3 → 12
    assert position_size(1_234_567.0, 10_000.0, 0.10, CFG) == 12
    assert position_size(1e8, 10_000.0, 0.10, StrategyConfig(risk_per_trade=0.005)) == 500
    assert position_size(-1.0, 10_000.0, 0.1, CFG) == 0
    assert position_size(1e8, 3_000.0, 0.10, CFG, fractional=True) == pytest.approx(1e6 / (3_000 * 0.10))
    assert position_size(1e8, 3_000.0, 0.07, CFG, fractional=True) == pytest.approx(1e7 / 3_000)   # 상한


def test_rank_rel_asc_then_rs_desc_nan_last():
    ps = [_plan("A", 0.9, 99), _plan("B", 0.5, 70), _plan("C", 0.5, 90), _plan("D", math.nan, 100)]
    assert [p.code for p in rank_plans(ps)] == ["C", "B", "A", "D"]


def test_allocate_slots_cash_and_held():
    ps = [_plan("A", 0.9, 99), _plan("B", 0.5, 70), _plan("C", 0.4, 90)]
    got = allocate(ps, cash=1e9, open_slots=2)
    assert [p.code for p in got] == ["C", "B"]
    got = allocate(ps, cash=1e9, open_slots=5, held={"C"})
    assert [p.code for p in got] == ["B", "A"]
    # 현금 1.5종목분 → 첫 종목 전량, 둘째는 줄여서(50주), 셋째는 0주 → 빠짐
    got = allocate(ps, cash=1_500_000.0, open_slots=5)
    assert [(p.code, p.qty) for p in got] == [("C", 100), ("B", 50)]
    assert allocate(ps, cash=1e9, open_slots=0) == []
    assert allocate(ps, cash=5_000.0, open_slots=5) == []           # 1주도 못 산다


def test_allocate_unlimited_fractional():
    ps = [_plan(c, 0.5, 80, qty=1.0) for c in "ABCDEFGHIJKL"]
    got = allocate(ps, cash=math.inf, open_slots=math.inf, fractional=True)
    assert len(got) == 12 and all(p.qty == 1.0 for p in got)
