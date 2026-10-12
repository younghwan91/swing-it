"""Storage layer (read side): connect() dispatch, market_cap_asof. No network.

Write-side tests (upsert idempotency, to_int/to_float, ON CONFLICT SQL) moved
to quant-airflow/tests/test_storage.py along with the upsert_* functions
themselves — see quant-airflow/collectors/storage.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from swing_it.storage import connect, market_cap_asof


def test_connect_dispatches_postgres_dsn_to_psycopg2():
    """A postgresql:// path opens Postgres instead of sqlite — no real connection made."""
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"psycopg2": fake_module}):
        connect("postgresql://user:pw@localhost:5432/swing_it")
    fake_module.connect.assert_called_once_with("postgresql://user:pw@localhost:5432/swing_it")


def _insert_bar(con, code, date, close):
    con.execute(
        "INSERT INTO daily_bars(code, date, open, high, low, close, volume, trade_value) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
        (code, date, close, close, close, close),
    )
    con.commit()


def _insert_shares(con, code, date, shares):
    con.execute(
        "INSERT INTO shares_outstanding_history(code, date, shares_outstanding) VALUES (?, ?, ?)",
        (code, date, shares),
    )
    con.commit()


def test_market_cap_asof_normal_case(tmp_path):
    con = connect(tmp_path / "t.db")
    _insert_bar(con, "005930", "2026-02-01", 70000)
    _insert_shares(con, "005930", "2026-02-01", 1000000)

    assert market_cap_asof(con, "005930", "2026-02-01", legacy=True) == 70000 * 1000000
    con.close()


def test_market_cap_asof_avoids_lookahead_bias(tmp_path):
    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1000000)
    _insert_shares(con, "005930", "2026-03-01", 2000000)  # simulates a later stock split
    _insert_bar(con, "005930", "2026-02-01", 70000)

    # 2026-02-01 is between the two share counts — must use the earlier
    # (on-or-before) 1,000,000 figure, never the later 2,000,000 one.
    assert market_cap_asof(con, "005930", "2026-02-01", legacy=True) == 70000 * 1000000
    con.close()


def test_market_cap_asof_returns_none_without_shares_data(tmp_path):
    con = connect(tmp_path / "t.db")
    _insert_bar(con, "005930", "2026-02-01", 70000)

    assert market_cap_asof(con, "005930", "2026-02-01", legacy=True) is None
    con.close()


def test_market_cap_asof_treats_zero_shares_as_missing(tmp_path):
    """0 주식수는 결측이다 — close*0 = 0 이 정상 시총으로 나가면 안 된다.

    조회가 `date <= 조회일 ORDER BY date DESC LIMIT 1` 이라, 0 행이 한 번 들어오면
    그 종목의 최신 점이 되어 이후 모든 조회를 이긴다. None 만 보는 가드로는 못 막는다.
    """
    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1000000)
    _insert_shares(con, "005930", "2026-01-15", 0)      # 파싱 실패가 0 으로 적재된 행
    _insert_bar(con, "005930", "2026-02-01", 70000)

    assert market_cap_asof(con, "005930", "2026-02-01", legacy=True) is None
    con.close()


def test_market_cap_asof_bulk_matches_per_row_on_zero_shares(tmp_path):
    """벌크도 같은 semantics — 더 오래된 유효 행으로 조용히 대체하지 않는다."""
    import pandas as pd

    from swing_it.storage import market_cap_asof_bulk

    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1000000)
    _insert_shares(con, "005930", "2026-01-15", 0)
    _insert_shares(con, "000660", "2026-01-01", 500000)
    _insert_bar(con, "005930", "2026-02-01", 70000)
    _insert_bar(con, "000660", "2026-02-01", 20000)

    df = pd.DataFrame({"code": ["005930", "000660"], "date": ["2026-02-01", "2026-02-01"]})
    out = market_cap_asof_bulk(con, df, legacy=True)

    assert pd.isna(out.iloc[0])                       # 0 행이 이긴 종목 → 결측
    assert out.iloc[1] == 20000 * 500000              # 정상 종목은 그대로
    assert pd.isna(out.iloc[0]) == (market_cap_asof(con, "005930", "2026-02-01", legacy=True) is None)
    con.close()



# --- 새 정의(기본): 원 종가 × 알려진 주식수 — legacy=True 는 위의 옛 정의 ------------------------------------------

def _insert_sd_close(con, code, date, close):
    con.execute("INSERT INTO supply_demand(code, date, close) VALUES (?, ?, ?)", (code, date, close))
    con.commit()


def _insert_shares_kd(con, code, date, shares, knowledge_date):
    con.execute("INSERT INTO shares_outstanding_history(code, date, shares_outstanding, source, knowledge_date) "
                "VALUES (?, ?, ?, 'dart', ?)", (code, date, shares, knowledge_date))
    con.commit()


def test_market_cap_uses_raw_close_not_back_adjusted_daily_bars(tmp_path):
    """daily_bars.close 는 이후 분할이 소급 반영된 가격이다(005930 2017 41,200 vs 실제 206만). 원 종가(수급표)를 쓴다."""
    con = connect(tmp_path / "t.db")
    _insert_bar(con, "005930", "2017-03-31", 41_200)              # 소급 수정 가격
    _insert_sd_close(con, "005930", "2017-03-31", -2_060_000)     # 원 종가(키움 부호 포함)
    _insert_shares(con, "005930", "2016-12-31", 140_000_000)
    assert market_cap_asof(con, "005930", "2017-03-31") == 2_060_000 * 140_000_000
    assert market_cap_asof(con, "005930", "2017-03-31", legacy=True) == 41_200 * 140_000_000


def test_market_cap_uses_knowledge_date_not_report_date(tmp_path):
    """dart 행은 기준일(date)보다 늦게 알려진다 — 알려지기 전 날짜의 시총에 쓰면 미래 정보다."""
    con = connect(tmp_path / "t.db")
    _insert_bar(con, "000660", "2026-02-01", 20_000)
    _insert_shares(con, "000660", "2025-06-30", 500_000)                          # kiwoom(knowledge_date NULL) = 기준일에 안다
    _insert_shares_kd(con, "000660", "2025-12-31", 900_000, "2026-03-20")        # 기준일 12-31, 공시 03-20
    assert market_cap_asof(con, "000660", "2026-02-01") == 20_000 * 500_000
    assert market_cap_asof(con, "000660", "2026-02-01", legacy=True) == 20_000 * 900_000   # 옛 정의 = 미래 정보


def test_market_cap_new_definition_skips_zero_share_rows(tmp_path):
    """새 정의는 0 이하 주식수 행을 결측으로 **빼고**(read_shares) 그 전의 알려진 값을 쓴다."""
    con = connect(tmp_path / "t.db")
    _insert_shares(con, "005930", "2026-01-01", 1_000_000)
    _insert_shares(con, "005930", "2026-01-15", 0)
    _insert_bar(con, "005930", "2026-02-01", 70_000)
    assert market_cap_asof(con, "005930", "2026-02-01") == 70_000 * 1_000_000


def test_read_shares_asof_column_and_filters(tmp_path):
    from swing_it.storage import read_shares

    con = connect(tmp_path / "t.db")
    _insert_shares(con, "A", "2026-01-01", 10)
    _insert_shares(con, "A", "2026-01-02", 0)
    _insert_shares_kd(con, "B", "2025-12-31", 20, "2026-03-20")
    df = read_shares(con, with_asof=True, parse_dates=True)
    assert set(df["code"]) == {"A", "B"} and len(df) == 2                         # 0 주식수 행은 뺀다
    b = df[df["code"] == "B"].iloc[0]
    assert str(b["asof"].date()) == "2026-03-20"                                 # coalesce(knowledge_date, date)
    a = df[df["code"] == "A"].iloc[0]
    assert str(a["asof"].date()) == "2026-01-01"
    assert len(read_shares(con, include_nonpositive=True)) == 3
    assert set(read_shares(con, codes=["B"])["code"]) == {"B"}
    with pytest.raises(ValueError):
        read_shares(con, cols=("code; DROP TABLE x",))


def _supply_schema(con):
    # connect() 가 이미 실 스키마(17컬럼)를 만든다 — 없을 때만 최소 형태로 보강한다.
    con.execute(
        "CREATE TABLE IF NOT EXISTS supply_demand("
        "code TEXT, date TEXT, individual REAL, foreign_ REAL, institution REAL, source TEXT)")
    con.commit()


def _insert_supply(con, code, date, individual, foreign_, institution, source):
    con.execute(
        "INSERT INTO supply_demand(code, date, individual, foreign_, institution, source) "
        "VALUES (?,?,?,?,?,?)",
        (code, date, individual, foreign_, institution, source))
    con.commit()


def test_read_supply_demand_blocks_silent_individual_survivorship(tmp_path):
    """개인 수급 + 폐지 커버리지는 현 데이터로 양립 불가 — 조용히 넘기지 않는다."""
    from swing_it.storage import read_supply_demand

    con = connect(tmp_path / "t.db")
    _supply_schema(con)
    with pytest.raises(AssertionError, match="individual"):
        read_supply_demand(con, cols=("code", "date", "individual"))
    con.close()


def test_read_supply_demand_allows_individual_when_declared(tmp_path):
    from swing_it.storage import read_supply_demand

    con = connect(tmp_path / "t.db")
    _supply_schema(con)
    _insert_supply(con, "005930", "2026-08-26", 10.0, 5.0, -15.0, "kiwoom")
    df = read_supply_demand(con, cols=("code", "date", "individual"),
                            allow_individual_survivorship=True)
    assert len(df) == 1
    con.close()


def test_read_supply_demand_catches_missing_delisted(tmp_path):
    """폐지 종목 수급이 구간에 있는데 결과에 없으면 터진다."""
    from swing_it.storage import _assert_supply_has_delisted

    con = connect(tmp_path / "t.db")
    _supply_schema(con)
    _insert_supply(con, "900010", "2026-01-05", None, 3.0, 4.0, "naver")   # 폐지분
    _insert_supply(con, "005930", "2026-01-05", 1.0, 2.0, 3.0, "kiwoom")

    import pandas as pd
    survivors_only = pd.DataFrame({"code": ["005930"], "date": ["2026-01-05"]})
    with pytest.raises(AssertionError, match="폐지 종목 수급"):
        _assert_supply_has_delisted(con, survivors_only, start=None, end=None)
    con.close()


def test_read_supply_demand_window_scopes_the_delisted_expectation(tmp_path):
    """'오늘 하루' 시점 조회는 그 구간에 폐지분이 없으므로 헛되이 터지지 않는다."""
    from swing_it.storage import read_supply_demand

    con = connect(tmp_path / "t.db")
    _supply_schema(con)
    _insert_supply(con, "900010", "2026-01-05", None, 3.0, 4.0, "naver")   # 오래전 폐지
    _insert_supply(con, "005930", "2026-08-26", None, 2.0, 3.0, "kiwoom")

    df = read_supply_demand(con, cols=("code", "date", "institution"),
                            start="2026-08-26", end="2026-08-26")
    assert list(df["code"]) == ["005930"]
    con.close()


def test_read_earnings_with_known_dates(tmp_path):
    """known_date = max(avail_date, 정기보고서 첫 rcept_dt) 다음 거래일, event_date = 첫 공시 다음 거래일 — eps_asof 와 같은 규칙."""
    import pandas as pd

    from swing_it.storage import read_earnings

    con = connect(tmp_path / "t.db")
    con.executemany("INSERT INTO earnings(code, period, avail_date, knowledge_date, netinc, netinc_prior) VALUES (?,?,?,?,?,?)",
                    [("A", "2024Q1", "2024-05-15", "2024-05-20", 10, 5),     # 정기보고서가 마감일보다 늦다(5/17 금)
                     ("B", "2024Q1", "2024-05-15", "2024-05-20", 3, 4),      # 공시 기록 없음 → avail_date 로 돌아간다
                     ("C", "2024Q1", "2024-05-15", "2024-05-20", 1, 1)])     # 잠정실적이 먼저(4/30)
    con.executemany("INSERT INTO earnings_disclosures(code, period, kind, rcept_dt) VALUES (?,?,?,?)",
                    [("A", "2024Q1", "periodic", "2024-05-17"), ("A", "2024Q1", "periodic", "2024-05-30"),
                     ("C", "2024Q1", "prelim", "2024-04-30"), ("C", "2024Q1", "periodic", "2024-05-14")])
    con.commit()
    cal = pd.DatetimeIndex(["2024-04-30", "2024-05-02", "2024-05-14", "2024-05-16", "2024-05-17", "2024-05-20", "2024-05-21"])
    df = read_earnings(con, cols=("code", "period", "netinc"), with_known=True, calendar=cal).set_index("code")
    assert list(df.columns) == ["period", "netinc", "known_date", "event_date"]       # 안에서 읽은 avail_date 는 버린다
    assert df.loc["A", "known_date"] == pd.Timestamp("2024-05-20")      # max(5/15, 5/17) 다음 거래일(주말 건너뜀)
    assert df.loc["A", "event_date"] == pd.Timestamp("2024-05-20")
    assert df.loc["B", "known_date"] == pd.Timestamp("2024-05-16")      # avail_date 5/15 다음 거래일
    assert pd.isna(df.loc["B", "event_date"])
    assert df.loc["C", "known_date"] == pd.Timestamp("2024-05-16")      # max(5/15, 5/14) = 5/15
    assert df.loc["C", "event_date"] == pd.Timestamp("2024-05-02")      # 잠정 4/30 다음 거래일
