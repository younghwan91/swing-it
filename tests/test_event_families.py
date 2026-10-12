"""13세대 사건 가족(EVENT_FAMILIES.md 정정 E-1) — 엔진 종가 진입·오버나이트 시가 판단."""
import numpy as np
import pytest

from research.signals import pattern_fast as PF
from tests.test_pattern_ga_fwm import _book, _px, _run


def _close_px(hold):
    return _px(X_ENTRY_CLOSE=1.0, X_ON_HOLD=hold, X_TRAIL=PF.TRAIL_MA50)


def test_close_entry_ignores_signal_day_range():
    # s 봉 저가가 손절 아래여도(종가에 샀으니) 손절하지 않는다 — 다음날 시가에 판다(p_hold 0.02, 시가 +1%)
    rows = [(100, 106, 90, 100), (101, 103, 99, 102), (102, 104, 100, 103)]
    out = _run(_book(rows), _close_px(0.02), e=100.0, sp=95.0)
    assert out[PF.O_X] == 1 and out[PF.O_EXIT_OPEN] == 1 and out[PF.O_FIN] == pytest.approx(101.0)
    assert out[PF.O_MFE] <= 101.0 or out[PF.O_MFE] == pytest.approx(103.0)    # s 봉 고가 106 은 MFE 에 안 든다
    assert out[PF.O_MFE] != pytest.approx(106.0)


def test_close_entry_gap_down_through_stop_sells_at_open():
    rows = [(100, 101, 99, 100), (90, 92, 88, 91), (91, 93, 90, 92)]
    out = _run(_book(rows), _close_px(0.02), e=100.0, sp=95.0)
    assert out[PF.O_X] == 1 and out[PF.O_REASON] == PF.R_STOP
    assert out[PF.O_FIN] < 91.0                                              # 시가 90 (− 미끄러짐·호가)


def test_close_entry_holds_when_gap_up_enough():
    rows = [(100, 101, 99, 100), (104, 106, 103, 105), (105, 107, 104, 106)]
    out = _run(_book(rows), _close_px(0.02), e=100.0, sp=95.0)
    assert out[PF.O_X] == 2                                                  # 시가 +4% ≥ +2% → 스윙으로 넘김(데이터 끝)


def test_close_entry_off_hold_is_swing():
    rows = [(100, 101, 99, 100), (100.5, 102, 100, 101), (101, 103, 100, 102)]
    out = _run(_book(rows), _close_px(np.nan), e=100.0, sp=95.0)
    assert out[PF.O_X] == 2                                                  # NaN = 시가 판단 없음


def test_stop_buy_entry_unchanged():
    rows = [(100, 106, 90, 100), (101, 103, 99, 102)]
    out = _run(_book(rows), _px(), e=100.0, sp=95.0)                         # 기존 경로: MFE 는 진입일 고가부터
    assert out[PF.O_MFE] == pytest.approx(106.0)


def test_lock_to_raises_breakeven_stop():
    # 1일째 고가 +1R(105) → 본전 발동, lock_to 0.5 → 2일째부터 손절 102.5
    rows = [(100, 101, 99, 100), (101, 105, 100, 104), (104, 104, 102, 103), (103, 104, 101, 102)]
    out = _run(_book(rows), _px(X_BE_R=1.0, X_LOCK_TO=0.5), e=100.0, sp=95.0)
    assert out[PF.O_X] == 2 and out[PF.O_REASON] == PF.R_BE


def test_reversal_bar_exit_at_close():
    rows = [(100, 101, 99, 100), (101, 106, 100, 105), (105, 108, 104, 104.5), (104, 105, 103, 104)]
    out = _run(_book(rows), _px(X_REV_R=1.0, X_REV_NEXT=0.0), e=100.0, sp=95.0)
    assert out[PF.O_X] == 2 and out[PF.O_REASON] == PF.R_REV and out[PF.O_FIN] == pytest.approx(104.5 * (1 - PF.LIVE_X[PF.X_SELL_EXTRA]))


def test_prior_low_trail():
    rows = [(100, 101, 99, 100), (101, 108, 103, 107), (107, 109, 102, 105), (105, 106, 104, 105)]
    out = _run(_book(rows), _px(X_PLOW_R=1.5), e=100.0, sp=95.0)
    assert out[PF.O_X] == 2 and out[PF.O_REASON] == PF.R_TRAIL                 # 전일 저가 103 이탈


def test_close_cut():
    rows = [(100, 101, 99, 100), (99, 100, 96, 97), (97, 98, 96, 97)]
    out = _run(_book(rows), _px(X_CLOSE_CUT=0.5), e=100.0, sp=95.0)
    assert out[PF.O_X] == 1 and out[PF.O_FIN] == pytest.approx(97.0)          # 종가 97 < 100 − 2.5


def test_new_exit_genes_off_by_default():
    for k in ("X_LOCK_TO", "X_REV_R", "X_PLOW_R", "X_CLOSE_CUT", "X_ON_HOLD"):
        assert np.isnan(PF.LIVE_X[getattr(PF, k)])


def test_trail_gate_waits_for_ma_above_entry():
    # 16b(미너비니 Champion p162): MA50 이 진입가(100) 아래면 MA50 종가 이탈로 팔지 않는다 — 손절선이 지킨다
    rows = [(100, 101, 99, 100), (100, 101, 98, 98.5), (98.5, 99, 97, 98), (98, 99, 97, 98)]
    ma50 = [99, 99, 99, 99]
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=0.0, X_TRAIL_GATE=0.0))
    assert int(out[PF.O_REASON]) == PF.R_TRAIL                                # 끔: 종가 98.5 < MA50 99 → 추적 매도
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=0.0, X_TRAIL_GATE=1.0))
    assert int(out[PF.O_REASON]) != PF.R_TRAIL and int(out[PF.O_X]) == 3      # 켬: MA50 < 진입가 → 추적 없음(데이터 끝)
