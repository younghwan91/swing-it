"""스윙 라이브 전략 순수 로직 — 주문 계획·포지션 상태 전이 전 분기·후보 판정·피처 조립(설계 §5 단위 테스트)."""

import math

import numpy as np
import pandas as pd
import pytest

from swing_it.live.config import LIMIT_UP_REL_TOL, StrategyConfig
from swing_it.live.strategy import (
    Candidate,
    EntryPlan,
    PositionState,
    entry_limit,
    floor_tick,
    plan_entry,
    scan,
)

CFG = StrategyConfig()
#: 옛 프로토콜(시간 손절 20일, 돌파 실패 끔) — 시간 손절 분기 테스트용.
TS = StrategyConfig(time_stop_days=20, failed_breakout=False)


def _pos(c_low=95.0, qty=10, pivot=98.0, **kw) -> PositionState:
    return PositionState(code="000010", contraction_low=c_low, qty=qty, pivot=pivot, **kw)


def _filled(entry=100.0, c_low=95.0, qty=10, **kw) -> PositionState:
    p = _pos(c_low, qty, **kw)
    assert p.on_fill_open(entry) == []
    return p


def _kinds(acts):
    return [(a.kind, a.price) for a in acts]


# ---------------------------------------------------------------------------
# 설정·주문 계획
# ---------------------------------------------------------------------------

def test_limit_up_constants_match_research_and_korea():
    from research.signals.sepa import LIMIT_UP_RATIO
    from swing_it.features import korea

    assert LIMIT_UP_REL_TOL == korea.LIMIT_UP_REL_TOL
    assert CFG.limit_up_ratio == LIMIT_UP_RATIO


def test_prereg_exit_constants_match_research():
    from research.signals.minervini_exits import PREREG_MINERVINI as M
    from research.signals.sepa import PREREG_SKELETON as S

    assert (CFG.stop_mult, CFG.stop_floor, CFG.stop_cap, CFG.take_r) == (M["stop_mult"], M["stop_floor"],
                                                                         M["stop_cap"], M["take_r"])
    assert (TS.time_stop_days, CFG.max_hold, CFG.chase, CFG.contraction_window) == (
        M["time_stop_days"], M["max_hold"], M["chase"], M["vcp_window"])
    assert CFG.time_stop_days is None and CFG.failed_breakout          # 설계 §6: 시간 손절 폐지 → 돌파 실패
    assert (CFG.rs_min, CFG.breadth_min, CFG.pivot_window, CFG.vol_mult) == (
        S["rs_min"], S["breadth_min"], S["pivot_window"], S["vol_mult"])


def test_floor_tick_rounds_down_on_krx_grid():
    assert floor_tick(10_512.0) == 10_510.0
    assert floor_tick(2_003.0) == 2_000.0          # 5원 틱 구간 → 내림
    assert floor_tick(1_999.7) == 1_999.0
    assert floor_tick(55_049.0) == 55_000.0        # 100원 틱
    assert math.isnan(floor_tick(float("nan")))


def test_entry_limit_live_caps_at_chase_and_one_tick_below_limit_up():
    # 피벗×1.05 = 10,500 < 상한가−1틱 → 피벗 쪽, 호가 내림
    lim, lu = entry_limit(10_003.0, 9_800.0, CFG, tick_grid=True)
    assert lim == floor_tick(10_003.0 * 1.05) == 10_500.0
    assert lu == 12_740.0                           # 9,800 × 1.3 = 12,740 (10원 틱)
    # 피벗이 높아 상한가가 막는 경우: 상한가 15,000×1.3=19,500 → 한 틱 아래 19,490
    lim, lu = entry_limit(20_000.0, 15_000.0, CFG, tick_grid=True)
    assert (lim, lu) == (19_490.0, 19_500.0)


def test_entry_limit_backtest_uses_research_expressions():
    from research.signals.sepa import LIMIT_UP_RATIO

    lim, lu = entry_limit(123.456, 120.0, CFG, tick_grid=False)
    assert lim == 123.456 * (1.0 + 0.05)
    assert lu == 120.0 * LIMIT_UP_RATIO


def test_fills_at_matches_chase_and_limit_up_rules():
    plan = EntryPlan(code="A", signal_day=pd.Timestamp("2025-01-02"), market="KOSPI", limit_px=105.0,
                     limit_up_px=126.0, qty=10, ref_px=100.0, d_est=0.05, pivot=100.0, contraction_low=95.0,
                     rel=0.8, rs=90.0)
    assert plan.fills_at(105.0)            # 시가 = 지정가 → 체결
    assert not plan.fills_at(105.01)       # 추격 금지
    assert plan.fills_at(90.0)
    assert not plan.fills_at(float("nan"))
    hi = EntryPlan(**{**plan.__dict__, "limit_px": 200.0})
    assert not hi.fills_at(126.0)          # 상한가 기준 이상은 체결 불가(연구 skip_limit_up 은 ≥)


def test_plan_entry_sizes_by_risk_and_cap():
    c = Candidate(code="A", signal_day=pd.Timestamp("2025-01-02"), market="KOSPI", close=10_000.0, pivot=9_800.0,
                  contraction_low=9_500.0, rel=0.7, rs=88.0, adv20=8_000.0, pead_rank=0.9, flow_f=1.0)
    p = plan_entry(c, 100_000_000.0, 10_000.0, CFG, tick_grid=True)
    d = 1 - 9_500 * 0.99 / 10_000            # 0.0595
    assert p.d_est == pytest.approx(d)
    # 위험 1% = 1,000,000 / (10,000 × 0.0595) = 1680.6 > 상한 10% = 1,000 주
    assert p.qty == 1_000
    assert p.limit_px == floor_tick(9_800 * 1.05)


# ---------------------------------------------------------------------------
# 포지션 상태 — 체결
# ---------------------------------------------------------------------------

def test_fill_sets_stop_and_target_with_clipping():
    p = _filled(100.0, c_low=95.0)
    assert p.d == pytest.approx(1 - 95 * 0.99 / 100)
    assert p.stop_px == pytest.approx(100 * (1 - p.d))
    assert p.target_px == pytest.approx(100 * (1 + 2 * p.d))
    assert _filled(100.0, c_low=99.0).d == 0.03          # 하한
    assert _filled(100.0, c_low=50.0).d == 0.10          # 상한


def test_fill_at_or_below_contraction_low_exits_same_day():
    p = _pos(c_low=100.0)
    acts = p.on_fill_open(100.0)
    assert _kinds(acts) == [("below_low", 100.0)]
    assert p.phase == "closed" and p.gross_return() == 0.0


def test_fill_requires_contraction_low():
    with pytest.raises(ValueError):
        _pos(c_low=float("nan")).on_fill_open(100.0)


def test_fill_requires_pivot_when_failed_breakout_on():
    with pytest.raises(ValueError, match="피벗"):
        _pos(pivot=float("nan")).on_fill_open(100.0)
    assert _pos(pivot=float("nan"), cfg=TS).on_fill_open(100.0) == []


def test_default_protocol_is_failed_breakout_without_time_stop():
    assert CFG.failed_breakout and CFG.time_stop_days is None
    p = _filled(100.0, c_low=95.0)
    for k in range(1, 30):                                       # 종가 ≤ 진입가여도 시간 손절 없음(피벗 위)
        assert p.on_bar(100, 101, 99, 99.0, 90, k) == []
    assert p.is_open


def test_failed_breakout_close_below_pivot_sells_next_open():
    p = _filled(100.0, c_low=95.0, pivot=98.0)
    assert p.on_bar(100, 101, 99, 98.0, 90, 1) == []             # 종가 = 피벗 → 아직
    acts = p.on_bar(99, 100, 97.5, 97.9, 90, 2)                  # 종가 < 피벗 → 다음날 시가 결정
    assert [(a.kind, a.price, a.at) for a in acts] == [("failed_breakout_next_open", None, "next_open")]
    assert p.is_open and p.trail_pending and p.pending_kind == "failed_breakout_next_open"
    acts = p.on_bar(93.0, 112.0, 90.0, 95.0, 90, 3)              # 다음날 시가가 손절·목표를 건드려도 시가 청산이 먼저
    assert _kinds(acts) == [("failed_breakout_next_open", 93.0)]
    assert p.phase == "closed" and p.gross_return() == pytest.approx(-0.07)


def test_failed_breakout_stop_and_partial_take_precedence_same_day():
    p = _filled(100.0, c_low=95.0, pivot=98.0)
    assert _kinds(p.on_bar(100, 101, 94.0, 97.0, 90, 1)) == [("stop", p.stop_px)]       # 장중 손절 먼저
    q = _filled(100.0, c_low=95.0, pivot=98.0)
    acts = q.on_bar(100, 112, 99, 97.0, 90, 1)                   # 익절 뒤 종가 < 피벗: 익절 후엔 돌파 실패 안 봄(MA50 추적만)
    assert [a.kind for a in acts] == ["partial"]
    assert q.phase == "half" and not q.trail_pending
    assert [a.kind for a in q.on_bar(101, 102, 100.5, 97.0, 110, 2)] == ["trail_next_open"]  # 종가 < MA50 → 추적


def test_failed_breakout_before_max_hold_on_last_day():
    p = _filled(100.0, c_low=95.0, pivot=98.0)
    for k in range(1, 119):
        p.on_bar(100, 101, 99, 100, 90, k)
    assert [a.kind for a in p.on_bar(100, 101, 99, 97.0, 90, 119)] == ["failed_breakout_next_open"]
    assert _kinds(p.on_bar(99.0, 101, 98, 100, 90, 120)) == [("failed_breakout_next_open", 99.0)]


def test_failed_breakout_live_path_close_then_open():
    p = _filled(100.0, c_low=95.0, pivot=98.0)
    acts = p.on_close(97.0, float("nan"), day_no=1)
    assert [(a.kind, a.at) for a in acts] == [("failed_breakout_next_open", "next_open")]
    assert _kinds(p.on_open(96.0, 2)) == [("failed_breakout_next_open", 96.0)]
    assert p.phase == "closed"


def test_failed_breakout_roundtrip_serialization():
    p = _filled(100.0, c_low=95.0, pivot=98.0)
    p.on_close(97.0, float("nan"), day_no=1)
    q = PositionState.from_dict(p.to_dict(), CFG)
    assert q.pivot == 98.0 and q.pending_kind == "failed_breakout_next_open"
    assert _kinds(q.on_open(96.0, 2)) == [("failed_breakout_next_open", 96.0)]


def test_time_stop_and_failed_breakout_both_on_same_day_time_stop_first():
    both = StrategyConfig(time_stop_days=20, failed_breakout=True)
    p = _filled(100.0, c_low=95.0, pivot=98.0, cfg=both)
    for k in range(1, 20):
        p.on_bar(100, 101, 99, 100, 90, k)
    assert _kinds(p.on_bar(100, 101, 99, 97.0, 90, 20)) == [("time_close", 97.0)]   # 종가 ≤ 진입가 ∧ < 피벗 → 시간 손절
    q = _filled(100.0, c_low=95.0, pivot=101.0, cfg=both)
    for k in range(1, 20):
        q.on_bar(100, 101, 99, 100.5, 90, k)                      # 종가 < 피벗 101 → 1일째에 이미 돌파 실패 결정
    assert q.exits[0]["kind"] == "failed_breakout_next_open" and q.phase == "closed"


# ---------------------------------------------------------------------------
# 포지션 상태 — 일봉(백테스트)
# ---------------------------------------------------------------------------

def test_stop_on_low_touch_and_gap_down_at_open():
    p = _filled(100.0, c_low=95.0)                      # stop ≈ 94.05
    assert _kinds(p.on_bar(100, 101, 94.0, 96, 90, 1)) == [("stop", p.stop_px)]
    q = _filled(100.0, c_low=95.0)
    assert q.on_bar(100, 101, 96, 99, 90, 1) == []
    assert _kinds(q.on_bar(90, 92, 89, 91, 90, 2)) == [("stop", 90)]    # 갭 하락 → 시가


def test_same_day_stop_and_target_stop_first_even_on_gap_up():
    p = _filled(100.0, c_low=95.0)
    assert _kinds(p.on_bar(100, 130, 90, 100, 90, 1)) == [("stop", p.stop_px)]
    q = _filled(100.0, c_low=95.0)
    q.on_bar(100, 101, 99, 100, 90, 1)
    acts = q.on_bar(115, 116, 93, 100, 90, 2)          # 갭 상승 시가가 목표 위지만 저가가 손절에 닿음
    assert _kinds(acts) == [("stop", q.stop_px)]


def test_partial_then_breakeven_from_next_day_only():
    p = _filled(100.0, c_low=95.0, qty=10)              # target ≈ 111.9
    acts = p.on_bar(100, 112, 99.0, 110, 90, 1)          # 익절일 저가 ≤ 진입가지만 같은 날 본전 없음
    assert _kinds(acts) == [("partial", p.target_px)]
    assert acts[0].qty == 5 and p.qty_open == 5 and p.phase == "half"
    acts = p.on_bar(101, 102, 99.5, 101, 90, 2)
    assert _kinds(acts) == [("breakeven", 100.0)]
    assert p.gross_return() == pytest.approx(0.5 * (p.target_px / 100 - 1) + 0.5 * 0.0)


def test_breakeven_gap_down_fills_at_open():
    p = _filled(100.0, c_low=95.0)
    p.on_bar(100, 112, 101, 110, 90, 1)
    assert _kinds(p.on_bar(97, 99, 96, 98, 90, 2)) == [("breakeven", 97)]


def test_gap_up_partial_with_same_day_breakeven():
    p = _filled(100.0, c_low=95.0)
    p.on_bar(100, 101, 99, 100, 90, 1)
    acts = p.on_bar(115, 116, 99.0, 100, 90, 2)          # 시가 ≥ 목표(갭), 저가 ≤ 진입가(손절 위)
    assert _kinds(acts) == [("partial", 115), ("breakeven", 100.0)]
    assert p.phase == "closed"


def test_trail_close_below_ma_sells_next_open():
    p = _filled(100.0, c_low=95.0)
    p.on_bar(100, 112, 101, 111, 90, 1)                  # 익절
    acts = p.on_bar(111, 113, 108, 109, 110, 2)          # 종가 < MA → 다음날 시가 결정
    assert [(a.kind, a.price, a.at) for a in acts] == [("trail_next_open", None, "next_open")]
    assert p.is_open and p.trail_pending
    acts = p.on_bar(99, 100, 95, 96, 110, 3)             # 시가가 본전 아래여도 사유는 trail
    assert _kinds(acts) == [("trail_next_open", 99)]


def test_trail_signal_on_partial_day_itself():
    p = _filled(100.0, c_low=95.0)
    acts = p.on_bar(100, 112, 101, 105, 106, 1)          # 익절일 종가 < MA
    assert [a.kind for a in acts] == ["partial", "trail_next_open"]
    assert _kinds(p.on_bar(104, 105, 103, 104, 106, 2)) == [("trail_next_open", 104)]


def test_no_trail_before_partial():
    p = _filled(100.0, c_low=95.0)
    assert p.on_bar(100, 101, 99, 100.5, 120, 1) == []   # 전량 단계에선 MA 아래여도 아무것도 없다


def test_time_stop_day_20_close_at_or_below_entry():
    p = _filled(100.0, c_low=95.0, cfg=TS)
    for k in range(1, 20):
        assert p.on_bar(100, 101, 99, 100.5, 90, k) == []
    assert _kinds(p.on_bar(100, 101, 99, 100.0, 90, 20)) == [("time_close", 100.0)]


def test_time_stop_checked_once():
    p = _filled(100.0, c_low=95.0, cfg=TS)
    for k in range(1, 20):
        p.on_bar(100, 101, 99, 100.5, 90, k)
    assert p.on_bar(100, 101, 99, 100.5, 90, 20) == []   # 종가 > 진입가 → 시간 손절 안 함
    assert p.on_bar(100, 101, 99, 99.0, 90, 21) == []    # 다시 보지 않는다
    assert p.is_open


def test_time_stop_halt_shifts_to_next_tradable_day():
    p = _filled(100.0, c_low=95.0, cfg=TS)
    for k in range(1, 20):
        p.on_bar(100, 101, 99, 100.5, 90, k)
    assert p.on_bar(100, 100, 100, 100, 90, 20, tradable=False) == []
    assert _kinds(p.on_bar(100, 101, 99, 99.5, 90, 21)) == [("time_close", 99.5)]


def test_time_stop_not_after_partial():
    p = _filled(100.0, c_low=95.0, cfg=TS)
    p.on_bar(100, 112, 101, 111, 90, 1)
    for k in range(2, 25):
        assert p.on_bar(105, 106, 101, 101, 90, k) == []
    assert p.is_open


def test_max_hold_closes_at_close_day_120():
    p = _filled(100.0, c_low=95.0, cfg=TS)
    for k in range(1, 20):
        p.on_bar(100, 101, 99, 101, 90, k)
    p.on_bar(100, 101, 99, 101, 90, 20)                  # 종가 > 진입가
    for k in range(21, 120):
        assert p.on_bar(100, 101, 99, 101, 90, k) == []
    assert _kinds(p.on_bar(100, 101, 99, 101, 90, 120)) == [("max_hold", 101)]


def test_max_hold_after_partial_no_trail_on_last_day():
    p = _filled(100.0, c_low=95.0)
    p.on_bar(100, 112, 101, 111, 90, 1)
    for k in range(2, 120):
        assert p.on_bar(111, 112, 110, 111, 100, k) == []
    assert _kinds(p.on_bar(111, 112, 110, 99.5 + 10, 200, 120)) == [("max_hold", 109.5)]


def test_halt_day_does_nothing():
    p = _filled(100.0, c_low=95.0)
    assert p.on_bar(50, 50, 50, 50, 90, 2, tradable=False) == []
    assert p.is_open and p.day_no == 2


def test_integer_partial_floors_and_single_share_keeps_state():
    p = _filled(100.0, c_low=95.0, qty=7)
    acts = p.on_bar(100, 112, 101, 111, 90, 1)
    assert acts[0].qty == 3 and p.qty_open == 4
    one = _filled(100.0, c_low=95.0, qty=1)
    acts = one.on_bar(100, 112, 101, 111, 90, 1)
    assert acts[0].qty == 0 and one.qty_open == 1 and one.phase == "half"   # 본전·추적 단계로는 넘어간다
    assert _kinds(one.on_bar(100.5, 101, 99, 100, 90, 2)) == [("breakeven", 100.0)]
    assert one.gross_return() == 0.0


def test_fractional_partial_half():
    p = _filled(100.0, c_low=95.0, qty=1.0, fractional=True)
    acts = p.on_bar(100, 112, 101, 111, 90, 1)
    assert acts[0].qty == 0.5 and p.qty_open == 0.5


def test_roundtrip_serialization_mid_trade():
    p = _filled(100.0, c_low=95.0)
    p.on_bar(100, 112, 101, 105, 106, 1)                 # 익절 + 추적 결정
    q = PositionState.from_dict(p.to_dict(), CFG)
    assert q.to_dict() == p.to_dict()
    assert _kinds(q.on_bar(104, 105, 103, 104, 106, 2)) == [("trail_next_open", 104)]


# ---------------------------------------------------------------------------
# 포지션 상태 — 라이브 이벤트
# ---------------------------------------------------------------------------

def test_live_quote_partial_breakeven_next_day():
    p = _filled(100.0, c_low=95.0)
    assert _kinds(p.on_quote(112.0)) == [("partial", 112.0)]
    assert p.on_quote(99.0) == []                         # 본전은 다음 거래일부터
    assert p.on_open(101.0, 2) == []
    assert _kinds(p.on_quote(99.9)) == [("breakeven", 99.9)]


def test_live_partial_day_keeps_original_stop_for_remaining_half():
    """REVIEW_BACKTEST M1: 장중 익절 뒤 같은 날 손절가 → 나머지 절반 손절(본전은 내일부터)."""
    p = _filled(100.0, c_low=95.0, qty=10)                      # stop 94.05, target 111.9
    assert _kinds(p.on_quote(112.0)) == [("partial", 112.0)]
    assert p.on_quote(99.0) == []                                 # 본전은 아직
    assert _kinds(p.on_quote(94.0)) == [("stop", 94.0)]
    assert p.phase == "closed" and p.exits[-1]["qty"] == 5
    # 갭 상승 익절 뒤 같은 날 손절가(본전 위) → 본전이 먼저 걸린다(그날 활성). 손절가는 본전 아래라 도달 전 본전.
    q = _filled(100.0, c_low=95.0, qty=10)
    q.on_open(115.0, 2)
    assert _kinds(q.on_quote(99.0)) == [("breakeven", 99.0)]
    # 익절 다음날 시가가 손절가 아래 → 본전(시가)이 먼저(본전 활성) — 손절 분기는 본전 미활성일 때만
    r = _filled(100.0, c_low=95.0, qty=10)
    r.on_quote(112.0)
    assert _kinds(r.on_open(90.0, 2)) == [("breakeven", 90.0)]


def test_live_open_gap_events():
    p = _filled(100.0, c_low=95.0)
    assert _kinds(p.on_open(90.0, 2)) == [("stop", 90.0)]
    q = _filled(100.0, c_low=95.0)
    assert _kinds(q.on_open(115.0, 2)) == [("partial", 115.0)]
    assert _kinds(q.on_quote(99.0)) == [("breakeven", 99.0)]   # 갭 익절일은 그날부터 본전


def test_live_close_time_stop_and_trail():
    p = _filled(100.0, c_low=95.0, cfg=TS)
    assert not p.time_stop_due(19)
    assert p.time_stop_due(20)
    assert _kinds(p.on_close(99.0, 90.0, day_no=20)) == [("time_close", 99.0)]
    q = _filled(100.0, c_low=95.0)
    q.on_quote(112.0)
    acts = q.on_close(105.0, 106.0, day_no=1)
    assert [(a.kind, a.at) for a in acts] == [("trail_next_open", "next_open")]
    assert _kinds(q.on_open(104.0, 2)) == [("trail_next_open", 104.0)]


# ---------------------------------------------------------------------------
# 후보 판정
# ---------------------------------------------------------------------------

def _scan_panels():
    days = pd.bdate_range("2025-01-01", periods=3)
    codes = ["000010", "000020", "000030", "000040", "000050"]

    def P(vals):
        return pd.DataFrame([vals] * 3, index=days, columns=codes, dtype=float)

    return days, {
        "close": P([100, 100, 100, 100, 100]), "pivot": P([95, 95, 95, 95, 95]),
        "tt": P([1, 1, 1, 1, 0]).astype(bool), "breakout": P([1, 1, 1, 1, 1]).astype(bool),
        "universe": P([1, 1, 1, 1, 1]).astype(bool),
        "rs": P([80, 90, 95, 99, 99]), "breadth": P([0.6] * 5), "pead_rank": P([0.8, 0.9, 0.75, np.nan, 0.9]),
        "rel": P([0.9, 0.5, 0.5, 0.2, 0.2]), "F": P([1, 1, 1, 1, 1]),
        "contraction_low": P([90] * 5), "adv20": P([6000] * 5),
        "market": pd.Series("KOSPI", index=codes),
    }


def test_scan_filters_and_ranks():
    days, pn = _scan_panels()
    got = scan(pn, days[1], CFG)
    # 000040 PEAD NaN, 000050 추세 템플릿 거짓 → 빠짐. 순위 rel 오름차순 → RS 내림차순
    assert [c.code for c in got] == ["000030", "000020", "000010"]
    pn["F"].loc[days[1], "000030"] = np.nan                      # 수급 모름은 통과 못 함
    assert [c.code for c in scan(pn, days[1], CFG)] == ["000020", "000010"]
    no_f = StrategyConfig(require_flow=False)
    assert [c.code for c in scan(pn, days[1], no_f)] == ["000030", "000020", "000010"]
    pn["rel"].loc[days[1], "000010"] = 1.5
    assert "000010" not in [c.code for c in scan(pn, days[1], CFG)]
    assert "000010" in [c.code for c in scan(pn, days[1], StrategyConfig(rel_max=None))]
    assert scan(pn, pd.Timestamp("2030-01-01"), CFG) == []


def test_scan_breadth_strict_and_rs_inclusive():
    days, pn = _scan_panels()
    pn["breadth"].loc[days[0], "000010"] = 0.5
    pn["rs"].loc[days[0], "000020"] = 70.0
    codes = [c.code for c in scan(pn, days[0], CFG)]
    assert "000010" not in codes and "000020" in codes


# ---------------------------------------------------------------------------
# 피처 — 연구 정의와 같은 식, 그날치 창 = 전 기간
# ---------------------------------------------------------------------------

def test_relative_contraction_matches_research():
    from research.experiments.convergence_run import relative_contraction as research_rel
    from swing_it.live.features import relative_contraction

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    c = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.02, (400, 3)), axis=0)) * 100, index=idx)
    h, lo = c * 1.01, c * 0.99
    a, b = relative_contraction(h, lo, CFG), research_rel(h, lo)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_contraction_low_matches_research():
    from research.signals.minervini_exits import contraction_low_level
    from swing_it.live.features import contraction_low

    lo = pd.DataFrame(np.random.default_rng(1).uniform(90, 110, (50, 2)))
    pd.testing.assert_frame_equal(contraction_low(lo, 10), contraction_low_level(lo, 10))


def _raw_inputs(n_days=420, n_codes=30, seed=3):
    from swing_it.live.features import RawInputs

    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=n_days)
    codes = pd.Index([f"{i:05d}0" for i in range(n_codes)])
    c = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.001, 0.02, (n_days, n_codes)), axis=0)) * 5_000,
                     index=days, columns=codes)
    c.iloc[:150, 0] = np.nan                                     # 늦게 상장
    c.iloc[380:, 1] = np.nan                                     # 폐지
    o = c.shift(1).fillna(c) * np.exp(rng.normal(0, 0.005, c.shape))
    h = np.maximum(o, c) * 1.01
    lo = np.minimum(o, c) * 0.99
    v = pd.DataFrame(rng.integers(10_000, 200_000, c.shape), index=days, columns=codes).astype(float).where(c.notna())
    tv = v * c / 1e6 * 10
    market = pd.Series(["KOSPI" if i % 2 else "KOSDAQ" for i in range(n_codes)], index=codes)
    meta = pd.DataFrame({"name": [f"종목{i}" for i in range(n_codes)], "market": market.to_numpy(),
                         "sector": np.nan, "delisted": False, "excluded": None, "src": "stocks"}, index=codes)
    sd = []
    for code in codes:
        for d in days:
            sd.append((code, d.strftime("%Y-%m-%d"), rng.normal(0, 1000), rng.normal(0, 1000), 50_000.0))
    supply = pd.DataFrame(sd, columns=["code", "date", "foreign_", "institution", "acc_trde_qty"])
    er = []
    for code in codes:
        for q, (per, av) in enumerate([("2023Q1", "20230515"), ("2023Q2", "20230814"), ("2023Q3", "20231114"),
                                       ("2023Q4", "20240331"), ("2024Q1", "20240515"), ("2024Q2", "20240814")]):
            ni = rng.normal(100, 50)
            er.append((code, per, av, av, ni, rng.normal(80, 40), 1000.0, 900.0, 50.0, 40.0))
    earnings = pd.DataFrame(er, columns=["code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior",
                                         "revenue", "revenue_prior", "op_income", "op_income_prior"])
    return RawInputs(open=o, high=h, low=lo, close=c, volume=v, trade_value=tv, raw_close=c, market=market,
                     meta=meta, supply=supply, earnings=earnings)


def test_build_signal_panels_matches_research_assembly():
    from research.experiments.convergence_run import relative_contraction as research_rel
    from research.signals.minervini_exits import contraction_low_level
    from research.signals.sepa import SepaPanels, skeleton_features
    from swing_it.features.earnings_leg import earnings_leg_panel, pead_rank
    from swing_it.features.flow_leg import flow_leg
    from swing_it.features.korea import universe_mask
    from swing_it.live.features import build_signal_panels

    inp = _raw_inputs()
    cfg = StrategyConfig(adv_floor=1.0)
    got = build_signal_panels(inp, cfg)

    p = SepaPanels(open=inp.open, high=inp.high, low=inp.low, close=inp.close, volume=inp.volume,
                   trade_value=inp.trade_value, market=inp.market)
    feats = skeleton_features(p)
    h5 = universe_mask(p.close, p.trade_value, inp.meta, raw_close=inp.raw_close, adv_floor=1.0)
    for k in ("tt", "rs", "breakout", "pivot", "breadth", "ma50"):
        pd.testing.assert_frame_equal(got[k], feats[k], check_names=False)
    pd.testing.assert_frame_equal(got["universe"], h5)
    pd.testing.assert_frame_equal(got["rel"], research_rel(p.high, p.low)[1])
    pd.testing.assert_frame_equal(got["contraction_low"], contraction_low_level(p.low, 10))
    fl = flow_leg(inp.supply, index=p.days, columns=p.close.columns, weight=p.trade_value)
    pd.testing.assert_frame_equal(got["F"], fl["F"])
    ep = earnings_leg_panel(inp.earnings, [d.strftime("%Y-%m-%d") for d in p.days])
    di = p.days.get_indexer(pd.to_datetime(ep["date"]))
    ci = p.close.columns.get_indexer(ep["code"])
    elig = h5.to_numpy()[di, ci]
    ep["pead_rank"] = pead_rank(ep, eligible=pd.Series(elig, index=ep.index))
    for _, r in ep.dropna(subset=["pead_rank"]).sample(200, random_state=0).iterrows():
        assert got["pead_rank"].loc[pd.Timestamp(r["date"]), r["code"]] == r["pead_rank"]
    assert got["pead_rank"].notna().sum().sum() == ep["pead_rank"].notna().sum()


def test_window_of_300_rows_equals_full_history_on_last_day():
    from swing_it.live.features import LOOKBACK_ROWS, RawInputs, build_signal_panels

    inp = _raw_inputs()
    cfg = StrategyConfig(adv_floor=1.0)
    full = build_signal_panels(inp, cfg)
    sl = slice(len(inp.days) - LOOKBACK_ROWS, None)
    win = RawInputs(**{k: (getattr(inp, k).iloc[sl] if isinstance(getattr(inp, k), pd.DataFrame)
                          and k not in ("meta", "supply", "earnings") else getattr(inp, k))
                       for k in ("open", "high", "low", "close", "volume", "trade_value", "raw_close", "market",
                                 "meta", "supply", "earnings")})
    part = build_signal_panels(win, cfg)
    d = inp.days[-1]
    for k in ("universe", "tt", "rs", "breakout", "pivot", "breadth", "rel", "F", "pead_rank", "contraction_low",
              "adv20"):
        a = full[k].loc[d].astype(float)
        b = part[k].loc[d].astype(float)
        pd.testing.assert_series_equal(a, b, check_names=False, obj=k)
    assert int(full["universe"].loc[d].sum()) > 0 and np.isfinite(full["rel"].loc[d]).any()


def test_min_lookback_rows_and_window_guard():
    from swing_it.live.features import LOOKBACK_ROWS, min_lookback_rows, signal_panels_for

    assert min_lookback_rows(CFG) == 250 + 11 + 10 == 271
    assert LOOKBACK_ROWS >= min_lookback_rows(CFG)
    with pytest.raises(ValueError, match="최소 271"):
        signal_panels_for(None, "2025-01-02", CFG, lookback=262)


def test_window_below_floor_changes_rel_silently():
    """262행 창은 오류 없이 다른 rel 을 낸다 — 그래서 하한을 명시한다(REVIEW_BACKTEST m1)."""
    from swing_it.live.features import RawInputs, build_signal_panels, min_lookback_rows

    inp = _raw_inputs()
    cfg = StrategyConfig(adv_floor=1.0)
    full = build_signal_panels(inp, cfg)
    d = inp.days[-1]

    def window(rows):
        sl = slice(len(inp.days) - rows, None)
        return build_signal_panels(RawInputs(**{k: (getattr(inp, k).iloc[sl] if k not in ("meta", "supply", "earnings",
                                                                                           "market")
                                                   else getattr(inp, k))
                                               for k in ("open", "high", "low", "close", "volume", "trade_value",
                                                         "raw_close", "market", "meta", "supply", "earnings")}), cfg)

    ok = window(min_lookback_rows(cfg))["rel"].loc[d]
    pd.testing.assert_series_equal(ok, full["rel"].loc[d], check_names=False)
    short = window(min_lookback_rows(cfg) - 9)["rel"].loc[d]
    assert not np.allclose(short.to_numpy(float), full["rel"].loc[d].to_numpy(float), equal_nan=True)


# ---------------------------------------------------------------------------
# 실적 문(GA2 유전자) — 기본 끔, 켜면 eps_yoy(PIT 주식수)·가속·매출 YoY 로 거른다
# ---------------------------------------------------------------------------

def test_earnings_gates_off_by_default_and_panels_present():
    from swing_it.live.features import build_signal_panels

    assert CFG.eps_yoy_min is None and not CFG.eps_accel and CFG.rev_yoy_min is None
    got = build_signal_panels(_raw_inputs(), StrategyConfig(adv_floor=1.0))
    for k in ("eps_yoy", "accel", "rev_yoy", "eps_from_netinc"):
        assert got[k].shape == got["close"].shape
    assert got["eps_from_netinc"].stack().dropna().eq(1.0).all()   # 주식수 없음 → 전부 순이익 YoY 대체
    assert got["eps_yoy"].notna().sum().sum() > 0


def test_scan_applies_earnings_gates_when_on():
    days, pn = _scan_panels()

    def P(vals):
        return pd.DataFrame([vals] * 3, index=days, columns=pn["close"].columns, dtype=float)

    pn["eps_yoy"] = P([0.5, 0.1, np.nan, 0.9, 0.9])
    pn["accel"] = P([1, 0, 1, 1, 1])
    pn["rev_yoy"] = P([0.2, 0.2, -0.3, 0.2, 0.2])
    base = [c.code for c in scan(pn, days[1], CFG)]
    assert base == ["000030", "000020", "000010"]                                 # 끔 = 그대로
    assert [c.code for c in scan(pn, days[1], StrategyConfig(eps_yoy_min=0.3))] == ["000010"]   # NaN 탈락
    assert [c.code for c in scan(pn, days[1], StrategyConfig(eps_accel=True))] == ["000030", "000010"]
    assert [c.code for c in scan(pn, days[1], StrategyConfig(rev_yoy_min=0.0))] == ["000020", "000010"]
    both = StrategyConfig(eps_yoy_min=0.0, eps_accel=True, rev_yoy_min=0.0)
    assert [c.code for c in scan(pn, days[1], both)] == ["000010"]
