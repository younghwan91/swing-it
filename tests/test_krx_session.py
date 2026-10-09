"""KRX 정규장 일봉 보정 — NXT 시간대 봉은 버리고, 조정계수는 시가 비율로."""

import pandas as pd

from swing_it.krx_session import patch_prices, session_bars_from_minutes


def _minutes():
    rows = [
        ("A", "2026-09-17 08:30:00", 99, 99, 99, 99, 5, 495),       # NXT 장전 — 버림
        ("A", "2026-09-17 09:00:00", 100, 103, 99, 101, 10, 1010),
        ("A", "2026-09-17 15:20:00", 101, 104, 100, 102, 10, 1020),
        ("A", "2026-09-17 15:30:00", 102, 102, 102, 102, 20, 2040),  # 종가 단일가
        ("A", "2026-09-17 19:59:00", 110, 120, 90, 115, 7, 805),    # NXT 야간 — 버림
    ]
    return pd.DataFrame(rows, columns=["code", "ts", "open", "high", "low", "close", "volume", "trade_value"])


def test_session_bars_drop_nxt_hours():
    s = session_bars_from_minutes(_minutes())
    r = s.iloc[0]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100, 104, 99, 102, 40)
    assert abs(r["trade_value"] - 4070 / 1e6) < 1e-12


def test_patch_scales_by_open_ratio_and_flags():
    s = session_bars_from_minutes(_minutes())
    prices = pd.DataFrame({
        "code": ["A", "A", "B"], "date": ["2026-09-11", "2026-09-17", "2026-09-17"],
        "open": [50.0, 50.0, 7.0], "high": [51.0, 60.0, 8.0], "low": [49.0, 45.0, 6.0], "close": [50.5, 57.5, 7.5],
        "volume": [1, 99, 3], "trade_value": [1.0, 9.0, 2.0]})
    out = patch_prices(prices, s)
    a_old, a_new, b = out.iloc[0], out.iloc[1], out.iloc[2]
    assert not a_old["krx_patched"] and a_old["close"] == 50.5          # 시작일 전은 그대로
    assert a_new["krx_patched"] and a_new["close"] == 102 * 0.5 and a_new["high"] == 104 * 0.5
    assert a_new["volume"] == 40
    assert not b["krx_patched"] and b["close"] == 7.5                    # 분봉 없는 종목은 원값 + 표시


def test_patch_accepts_integer_columns():
    """DB 원 일봉은 int64 열 — 실수 대입이 실패하지 않아야 한다(두 러너가 각자 우회했던 버그)."""
    s = session_bars_from_minutes(_minutes())
    prices = pd.DataFrame({"code": ["A"], "date": ["2026-09-17"], "open": [50], "high": [60], "low": [45],
                           "close": [57], "volume": [99], "trade_value": [9]})
    out = patch_prices(prices, s)
    assert out.iloc[0]["krx_patched"] and out.iloc[0]["close"] == 102 * 0.5
