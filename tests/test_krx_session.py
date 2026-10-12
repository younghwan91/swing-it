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


def test_extended_closing_auction_at_1532_is_the_close():
    """VI 로 종가 단일가가 연장되면 15:32 에 찍힌다 — 그 값이 종가여야 한다. 15:35 NXT 애프터마켓 봉은 버린다."""
    rows = [
        ("A", "2026-06-12 09:00:00", 100, 101, 99, 100, 10, 1000),
        ("A", "2026-06-12 15:19:00", 102, 103, 101, 102, 10, 1020),
        ("A", "2026-06-12 15:32:00", 130, 130, 130, 130, 50, 6500),   # 연장된 종가 단일가(상한가)
        ("A", "2026-06-12 15:35:00", 128, 129, 127, 128, 5, 640),     # NXT 애프터마켓 — 버림
    ]
    m = pd.DataFrame(rows, columns=["code", "ts", "open", "high", "low", "close", "volume", "trade_value"])
    r = session_bars_from_minutes(m).iloc[0]
    assert r["close"] == 130 and r["high"] == 130 and r["volume"] == 70


def test_raw_close_prefers_session_then_supply_then_bars():
    from swing_it.krx_session import combine_raw_close
    sd = pd.DataFrame({"code": ["A", "B"], "date": ["2017-03-31", "2026-09-17"], "close": [-2_060_000, 250]})
    bars = pd.DataFrame({"code": ["A", "B", "C"], "date": ["2017-03-31", "2026-09-17", "2017-03-31"], "close": [41_200, 260, 900]})
    sess = pd.DataFrame({"code": ["B"], "date": ["2026-09-17"], "close": [252.0]})
    out = combine_raw_close(sd, bars, sess).set_index("code")
    assert out.loc["A", "close"] == 2_060_000 and out.loc["A", "raw_src"] == "supply_demand"   # 부호 제거·수급 우선
    assert out.loc["B", "close"] == 252 and out.loc["B", "raw_src"] == "session"              # NXT 이후 분봉 우선
    assert out.loc["C", "close"] == 900 and out.loc["C", "raw_src"] == "daily_bars"           # 없을 때만 일봉


def test_raw_close_canary_rejects_retro_adjusted_price():
    import pytest

    from swing_it.krx_session import check_raw_close_canary
    bad = pd.DataFrame({"code": ["005930"], "date": ["2017-03-31"], "close": [41_200.0]})
    with pytest.raises(AssertionError):
        check_raw_close_canary(bad)
    check_raw_close_canary(bad.assign(close=2_060_000.0))
