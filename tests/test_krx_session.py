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


def test_split_consistent_cap_bridges_stale_shares():
    import numpy as np

    from swing_it.krx_session import split_consistent_cap
    days = pd.DatetimeIndex(["2018-04-30", "2018-05-04", "2018-12-28"])
    raw = pd.DataFrame({"S": [2_500_000.0, 51_900.0, 38_700.0]}, index=days)          # 5/4 50:1 분할
    adj = pd.DataFrame({"S": [50_000.0, 51_900.0, 38_700.0]}, index=days)              # 연속 수정 가격
    sh = pd.DataFrame({"code": ["S"], "date": ["2017-12-31"], "shares_outstanding": [129_000_000]})   # 분할 전 주식수(늦게 바뀜)
    sh = pd.concat([sh, pd.DataFrame({"code": ["S"], "date": ["2018-04-30"], "shares_outstanding": [129_000_000]})])
    cap, bad = split_consistent_cap(raw, adj, sh, asof="date")
    assert np.isclose(cap["S"].iloc[0], 2_500_000 * 129e6)                 # 분할 전 = 원 종가 × 주식수
    assert np.isclose(cap["S"].iloc[1], 51_900 * 129e6 * 50)               # 분할 뒤에도 주식수가 50배로 이어진다
    assert bad == 0


def test_factor_cap_reads_split_from_rebuilt_adjusted():
    import numpy as np

    from swing_it.krx_session import factor_cap
    days = pd.DatetimeIndex(["2018-04-30", "2018-05-04", "2018-12-28"])
    raw = pd.DataFrame({"S": [2_500_000.0, 51_900.0, 38_700.0]}, index=days)          # 5/4 50:1 분할
    adj = pd.DataFrame({"S": [50_000.0, 51_900.0, 38_700.0]}, index=days)              # 원 × 누적계수(v2)
    sh = pd.DataFrame({"code": ["S", "S"], "date": ["2017-12-31", "2018-04-30"], "shares_outstanding": [129_000_000, 129_000_000]})
    cap, F = factor_cap(raw, adj, sh, asof="date")
    assert np.isclose(cap["S"].iloc[0], 2_500_000 * 129e6)
    assert np.isclose(cap["S"].iloc[1], 51_900 * 129e6 * 50)                # 늦은 주식수(분할 전)가 50배로 이어진다
    assert np.isclose(F["S"].iloc[2] / F["S"].iloc[0], 50)
    sh2 = pd.DataFrame({"code": ["S"], "date": ["2018-12-28"], "shares_outstanding": [6_450_000_000]})   # 분할 뒤 행만 있는 경우
    cap2, _ = factor_cap(raw, adj, sh2, asof="date")
    assert np.isclose(cap2["S"].iloc[2], 38_700 * 6.45e9)


def test_factor_cap_row_before_data_start_uses_first_factor():
    """검증 29 C1: 주식수 행이 데이터 시작 전이면 F 대체값은 1 이 아니라 그 종목의 첫 F(v2 는 최신일 F = 1)."""
    import numpy as np

    from swing_it.krx_session import factor_cap
    days = pd.DatetimeIndex(["2017-01-02", "2018-05-04", "2023-06-01"])
    raw = pd.DataFrame({"S": [100_000.0, 21_000.0, 30_000.0]}, index=days)             # 2018-05-04 5:1 분할
    adj = pd.DataFrame({"S": [20_000.0, 21_000.0, 30_000.0]}, index=days)              # v2: 최신 F = 1, 분할 전 F = 0.2
    sh = pd.DataFrame({"code": ["S"], "date": ["2016-03-31"], "shares_outstanding": [25_000_000]})   # 데이터 시작 전 행(분할 전 주식수)
    cap, _ = factor_cap(raw, adj, sh, asof="date")
    assert np.isclose(cap["S"].iloc[0], 100_000 * 25e6)                    # 분할 전 = 원 × 행 주식수
    assert np.isclose(cap["S"].iloc[2], 30_000 * 25e6 * 5)                 # 분할 뒤 주식수 5배로 이어진다


def test_cap_functions_require_explicit_asof():
    """as-of 키를 기본값으로 숨기지 않는다 — "date" 는 기준일이라 dart 행은 석 달 미래 정보다(호출부가 고르게 한다)."""
    import pytest

    from swing_it.krx_session import factor_cap, split_consistent_cap
    days = pd.DatetimeIndex(["2018-04-30"])
    raw = adj = pd.DataFrame({"S": [1.0]}, index=days)
    sh = pd.DataFrame({"code": ["S"], "date": ["2018-04-30"], "shares_outstanding": [1]})
    with pytest.raises(TypeError, match="asof"):
        split_consistent_cap(raw, adj, sh)
    with pytest.raises(TypeError, match="asof"):
        factor_cap(raw, adj, sh)
