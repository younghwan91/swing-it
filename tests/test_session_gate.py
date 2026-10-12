"""정문의 정규장 보정(KRX 애프터마켓 16:00~20:00, 2026-09-14~) — read_prices·read_prices_wide·read_supply_demand·항등식.

2026-09-14 부터 키움 일봉·분봉·수급 종가가 애프터마켓 체결을 포함한다. 이 사실을 CLAUDE.md 에 적어 두었는데도 모의매매 첫 장부가
오염을 읽었다(730def6). 그래서 정문이 기본으로 고치고, 원값은 ``session="raw"`` 로 **명시**해야 받게 했다. 아래는 그 계약을 지킨다.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from swing_it.krx_session import (
    combine_raw_close,
    patch_prices,
    regular_close_from_supply_demand,
    session_bars_from_minutes,
)
from swing_it.storage import (
    SessionUnavailable,
    connect,
    read_prices,
    read_prices_wide,
    read_supply_demand,
)

COLS = ("code", "date", "open", "high", "low", "close", "volume", "trade_value", "source")


def _bars(con, rows, table="daily_bars_adjusted"):
    con.executemany(f"INSERT INTO {table}({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})", rows)  # noqa: S608
    con.commit()


def _minutes_table(con, rows):
    con.execute("CREATE TABLE minute_bars (code TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, "
                "volume REAL, trade_value REAL)")
    con.executemany("INSERT INTO minute_bars VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()


#: 조정 테이블 행: 2026-09-11(보정 전 날짜), 2026-09-15 A(분봉 있음)·B(분봉 없음). 조정계수 0.5(행 시가 50 / 정규장 시가 100).
_ADJ_ROWS = [
    ("A", "2026-09-11", 50, 51, 49, 50.5, 7, 3, "kiwoom"),
    ("A", "2026-09-15", 50, 60, 45, 57.5, 99, 9, "kiwoom"),   # 종가 57.5 = 애프터마켓 19:59 (×0.5)
    ("B", "2026-09-15", 7, 8, 6, 7.5, 3, 2, "kiwoom"),
]
_MIN_ROWS = [
    ("A", "2026-09-15 09:00:00", 100, 103, 99, 101, 10, 1010),
    ("A", "2026-09-15 15:30:00", 102, 104, 102, 102, 20, 2040),
    ("A", "2026-09-15 19:59:00", 110, 120, 90, 115, 7, 805),          # 애프터마켓 — 버림
]


def test_regular_is_default_and_patches_after_aftermarket_start(tmp_path):
    con = connect(tmp_path / "t.db")
    _bars(con, _ADJ_ROWS)
    _minutes_table(con, _MIN_ROWS)
    df = read_prices(con)
    a = df[(df["code"] == "A") & (df["date"] == "2026-09-15")].iloc[0]
    assert a["close"] == 102 * 0.5 and a["high"] == 104 * 0.5 and a["volume"] == 30
    assert a["krx_patched"]
    assert df.attrs["krx_session"] == {"session": "regular", "since": "2026-09-14", "rows": 2, "patched": 1, "unpatched": 1}
    assert df.loc[df["date"] == "2026-09-11", "close"].iloc[0] == 50.5        # 보정 전 날짜는 그대로
    assert df["date"].tolist() == [r[1] for r in _ADJ_ROWS]                    # 날짜 열 모양은 입력 그대로
    with pytest.raises(SessionUnavailable, match="strict"):
        read_prices(con, strict=True)


def test_window_before_aftermarket_is_byte_identical_to_raw(tmp_path):
    """09-14 전 창은 기본 모드가 raw 와 dtype 까지 같다 — 옛 연구 결과가 기본값 때문에 흔들리지 않는다."""
    con = connect(tmp_path / "t.db")
    _bars(con, _ADJ_ROWS)
    a = read_prices(con, end="2026-09-12")
    b = read_prices(con, end="2026-09-12", session="raw")
    pd.testing.assert_frame_equal(a, b)
    assert list(a.dtypes) == list(b.dtypes)


def test_sqlite_without_minute_bars_refuses_late_rows(tmp_path):
    """분봉 표가 없는 sqlite 에 09-14 이후 행이 있으면 조용히 오염값을 주지 않는다 — raw 를 명시해야 받는다."""
    con = connect(tmp_path / "t.db")
    _bars(con, _ADJ_ROWS)
    with pytest.raises(SessionUnavailable, match="session='raw'"):
        read_prices(con)
    with pytest.raises(SessionUnavailable):
        read_prices_wide(con, cols=("close",))
    raw = read_prices(con, session="raw")
    assert raw.loc[raw["date"] == "2026-09-15", "close"].tolist() == [57.5, 7.5]
    assert "krx_patched" not in raw.columns


def test_patching_is_idempotent_to_the_last_bit(tmp_path):
    """정문 출력에 patch_prices 를 한 번 더 걸어도 값이 비트 단위로 같다(자체 보정 경로와 겹쳐도 안전)."""
    con = connect(tmp_path / "t.db")
    rows = [("A", "2026-09-15", 33.3, 40.1, 30.7, 37.9, 99, 9, "kiwoom")]         # 조정계수 0.333 — 끝자리가 흔들리는 값
    _bars(con, rows)
    _minutes_table(con, [("A", "2026-09-15 09:00:00", 100.7, 103.3, 99.1, 101.9, 10, 1010),
                         ("A", "2026-09-15 15:30:00", 102.3, 104.9, 102.3, 102.3, 20, 2040)])
    once = read_prices(con)
    sess = session_bars_from_minutes(pd.read_sql_query("SELECT * FROM minute_bars", con))
    twice = patch_prices(once, sess)
    for c in ("open", "high", "low", "close", "volume", "trade_value"):
        assert np.array_equal(once[c].to_numpy(), twice[c].to_numpy()), c
    # 표시된 행은 다시 계산하지 않는다(조정계수를 다시 구하면 (so·fac)/so ≠ fac 로 끝자리가 흔들릴 수 있다 — 약 12%) —
    # 다른 분봉 값을 줘도 그대로인지로 확인한다.
    assert once["krx_patched"].all()
    other = sess.assign(close=sess["close"] * 2)
    assert np.array_equal(patch_prices(once, other)["close"].to_numpy(), once["close"].to_numpy())


def test_adjusted_table_without_open_column_reads_open_internally(tmp_path):
    """조정계수(시가 비율)가 필요해 안에서 open 을 더 읽지만 반환에는 요청한 열만."""
    con = connect(tmp_path / "t.db")
    _bars(con, _ADJ_ROWS)
    _minutes_table(con, _MIN_ROWS)
    df = read_prices(con, cols=("code", "date", "close"))
    assert list(df.columns) == ["code", "date", "close", "krx_patched"]
    assert df.loc[(df["code"] == "A") & (df["date"] == "2026-09-15"), "close"].iloc[0] == 51.0


def test_wide_regular_patches_in_place_and_reports(tmp_path):
    con = connect(tmp_path / "t.db")
    _bars(con, _ADJ_ROWS)
    _minutes_table(con, _MIN_ROWS)
    w = read_prices_wide(con, cols=("close", "volume"))
    assert set(w) == {"close", "volume"}                                          # 안에서 읽은 open 은 버린다
    assert w["close"].loc["2026-09-15", "A"] == 51.0 and w["volume"].loc["2026-09-15", "A"] == 30
    assert w["close"].loc["2026-09-15", "B"] == 7.5                              # 분봉 없음 → 원값, 미보정으로 셈
    assert w.krx_session["patched"] == 1 and w.krx_session["unpatched"] == 1
    raw = read_prices_wide(con, cols=("close",), session="raw")
    assert type(raw) is dict and raw["close"].loc["2026-09-15", "A"] == 57.5


def test_bad_session_name_is_rejected(tmp_path):
    con = connect(tmp_path / "t.db")
    with pytest.raises(ValueError, match="session"):
        read_prices(con, session="krx")


# --- 정규장 종가 항등식 ---------------------------------------------------------------------------------------

def _sd(rows):
    return pd.DataFrame(rows, columns=["code", "date", "close", "flu_rt"])


def test_identity_recovers_regular_close_from_next_row():
    """정규장 종가(t) = 호가 반올림(|close(t+1)| / (1 + flu_rt(t+1)/1e4)). 마지막 행은 값 없음."""
    sd = _sd([
        ("A", "2026-09-15", 25_650, 150.0),          # 애프터마켓 섞인 종가
        ("A", "2026-09-16", -25_850, 99.6),          # 기준가 = 25,850 / 1.00996 = 25,595.1 → 50원 호가 → 25,600
        ("A", "2026-09-17", 26_000, 50.0),
    ])
    out = regular_close_from_supply_demand(sd)
    assert out["date"].dt.strftime("%m-%d").tolist() == ["09-15", "09-16"]
    assert out["close"].tolist()[0] == 25_600.0
    assert out["close"].tolist()[1] == pytest.approx(round(26_000 / 1.005 / 50) * 50)


def test_identity_drops_split_days():
    """t+1 에 분할로 기준가가 조정되면 항등식 값이 원 종가의 몇 분의 1 — 버린다."""
    sd = _sd([("A", "2026-09-15", 50_000, 0.0), ("A", "2026-09-16", 10_100, 100.0)])   # 1:5 분할
    assert regular_close_from_supply_demand(sd).empty


def test_supply_demand_regular_mode(tmp_path):
    con = connect(tmp_path / "t.db")
    rows = [("A", "2026-09-15", 57, 1400.0, 999), ("A", "2026-09-16", 103, 98.0, 50),
            ("B", "2026-09-15", 2_010, 50.0, 7), ("B", "2026-09-16", -2_000, -24.88, 8)]
    con.executemany("INSERT INTO supply_demand(code, date, close, flu_rt, acc_trde_qty, foreign_, institution) "
                    "VALUES (?,?,?,?,?,0,0)", rows)
    con.commit()
    _minutes_table(con, [("A", "2026-09-15 09:00:00", 100, 103, 99, 101, 10, 1010),
                         ("A", "2026-09-15 15:30:00", 102, 104, 102, 102, 20, 2040)])
    df = read_supply_demand(con, cols=("code", "date", "close", "flu_rt", "acc_trde_qty"))
    a15 = df[(df["code"] == "A") & (df["date"] == "2026-09-15")].iloc[0]
    assert a15["close"] == 102 and a15["acc_trde_qty"] == 30                     # 분봉 정규장
    base_a = 57 / 1.14
    assert a15["flu_rt"] == pytest.approx((102 / base_a - 1) * 1e4)
    b15 = df[(df["code"] == "B") & (df["date"] == "2026-09-15")].iloc[0]
    assert b15["close"] == 2_005                                                  # 분봉 없음 → 항등식(2000/0.997512 → 5원 호가)
    assert np.isnan(b15["acc_trde_qty"]) and not b15["krx_patched"]              # 거래량은 분봉 없이는 못 고친다 — NaN
    b16 = df[(df["code"] == "B") & (df["date"] == "2026-09-16")].iloc[0]
    assert np.isnan(b16["close"])                                                 # 마지막 날 + 분봉 없음 → 오염값 대신 NaN
    assert df.attrs["krx_session"]["rows"] == 4 and df.attrs["krx_session"]["patched"] == 1
    raw = read_supply_demand(con, cols=("code", "date", "close", "flu_rt", "acc_trde_qty"), session="raw")
    assert raw["close"].tolist() == [57, 103, 2_010, -2_000]
    # 순매매 열만 읽으면 손대지 않는다
    flows = read_supply_demand(con, cols=("code", "date", "foreign_", "institution"))
    assert "krx_patched" not in flows.columns


def test_combine_raw_close_identity_priority_and_legacy_default():
    sess = pd.DataFrame({"code": ["A"], "date": ["2026-09-15"], "close": [100.0]})
    ident = pd.DataFrame({"code": ["A", "B"], "date": ["2026-09-15", "2026-09-15"], "close": [99.0, 50.0]})
    sd = pd.DataFrame({"code": ["A", "B", "B"], "date": ["2026-09-15", "2026-09-15", "2026-09-11"], "close": [111, 55, 40]})
    bars = pd.DataFrame({"code": ["B", "C"], "date": ["2026-09-15", "2026-09-15"], "close": [56, 70]})
    legacy = combine_raw_close(sd, bars, sess)
    assert dict(zip(legacy["code"] + legacy["date"].dt.strftime("%d"), legacy["raw_src"], strict=True)) == {
        "A15": "session", "B11": "supply_demand", "B15": "supply_demand", "C15": "daily_bars"}
    new = combine_raw_close(sd, bars, sess, identity=ident, sd_until="2026-09-14")
    got = {(c, d.day): (v, s) for c, d, v, s in new[["code", "date", "close", "raw_src"]].itertuples(index=False)}
    assert got[("A", 15)] == (100.0, "session")
    assert got[("B", 15)] == (50.0, "identity")
    assert got[("B", 11)] == (40.0, "supply_demand")                              # 09-14 전 수급 종가는 그대로 쓴다
    assert got[("C", 15)] == (70.0, "daily_bars")


def test_sqlite_connection_type_guard():
    """sqlite 판정은 연결 타입으로 — 정문이 Postgres 문법(::time)을 sqlite 에 보내지 않는다."""
    con = sqlite3.connect(":memory:")
    from swing_it.krx_session import read_session_bars_any
    assert read_session_bars_any(con, start="2026-09-14") is None
