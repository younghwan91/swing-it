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
