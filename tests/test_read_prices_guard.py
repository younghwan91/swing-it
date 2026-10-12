"""``read_prices`` — 유니버스에 상장폐지 종목이 들어있는지 로딩 시점에 검사.

생존편향은 검증 스택이 못 잡는다. walk-forward·음성대조·fragility 는 전부 **넘겨받은
트레이드**만 보므로, 유니버스가 이미 생존자만 담고 있으면 모든 게이트가 초록불이면서
성적만 부풀려진다(GUARDRAILS §4 공백 2). 잡을 수 있는 유일한 지점이 데이터를 읽는
자리라, 그 자리에서 assert 한다.
"""

from __future__ import annotations

import pytest

from swing_it.storage import DAILY_BAR_COLUMNS, connect, read_prices

COLS = [*DAILY_BAR_COLUMNS, "source"]


def _row(code, date, source):
    return (code, date, 100, 110, 90, 100, 10, 1, source)


def _seed(con, rows):
    ph = ",".join(["?"] * len(COLS))
    con.executemany(
        f"INSERT INTO daily_bars_adjusted({','.join(COLS)}) VALUES({ph})", rows)  # noqa: S608 — 모듈 상수
    con.commit()


def test_universe_with_delisted_names_loads(tmp_path):
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"),
                _row("000030", "2020-01-02", "naver")])
    df = read_prices(con)
    assert set(df["code"]) == {"000020", "000030"}
    con.close()


def test_missing_delisted_names_raises(tmp_path):
    """DB 엔 폐지 종목이 있는데 로딩 결과에 없으면 = 유니버스가 좁혀졌다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"),
                _row("000030", "2020-01-02", "naver")])

    # 생존자만 남기는 조회를 흉내낸다 — read_prices 를 우회한 자리와 같은 상황.
    import pandas as pd
    survivors = pd.read_sql_query(
        "SELECT code, date, close FROM daily_bars_adjusted WHERE source = 'kiwoom'", con)
    from swing_it.storage import _assert_universe_has_delisted
    with pytest.raises(AssertionError, match="생존편향"):
        _assert_universe_has_delisted(con, survivors, table="daily_bars_adjusted")
    con.close()


def test_db_without_any_delisted_rows_passes(tmp_path):
    """폐지 시세가 DB 에 아예 없으면 통과 — 없는 걸 요구할 수는 없다(백필이 선행)."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom")])
    assert len(read_prices(con)) == 1
    con.close()


def test_opt_out_is_explicit(tmp_path):
    """의도적으로 좁힌 유니버스는 인자로 명시해야 한다 — 조용히 되면 안 된다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"),
                _row("000030", "2020-01-02", "naver")])
    df = read_prices(con, require_delisted=False)
    assert len(df) == 2
    con.close()


def test_default_table_is_split_adjusted():
    """백테스트가 원자료를 읽으면 액면분할이 수익률로 둔갑한다."""
    from swing_it.storage import PRICE_TABLE
    assert PRICE_TABLE == "daily_bars_adjusted"


def test_read_prices_wide_sqlite_fallback_pivots_and_checks(tmp_path):
    """와이드판 정문(sqlite 폴백) — 같은 폐지 검사, 날짜×종목 패널 반환."""
    from swing_it.storage import read_prices_wide

    con = connect(tmp_path / "t.db")
    _seed(con, [_row("000020", "2020-01-02", "kiwoom"), _row("000030", "2020-01-02", "naver"),
                _row("000020", "2020-01-03", "kiwoom")])
    panels = read_prices_wide(con, cols=("close", "volume"))
    assert set(panels) == {"close", "volume"}
    assert list(panels["close"].columns) == ["000020", "000030"]
    assert panels["close"].shape == (2, 2)
    assert panels["close"].loc["2020-01-03", "000030"] != panels["close"].loc["2020-01-03", "000030"]  # NaN
    con.close()


def _tv_rows(n, *, unit=1.0, start="2020-01-"):
    """현실 단위 행: trade_value(백만원) = close × volume / 1e6 × unit."""
    return [(f"{i:06d}", f"{start}{(i % 28) + 1:02d}", 10_000, 10_100, 9_900, 10_000, 50_000,
             10_000 * 50_000 / 1e6 * unit, "kiwoom") for i in range(n)]


def test_trade_value_unit_canary_passes_real_units(tmp_path):
    con = connect(tmp_path / "t.db")
    _seed(con, _tv_rows(40))
    assert len(read_prices(con, cols=("code", "date", "close", "volume", "trade_value"))) == 40
    assert len(read_prices(con, cols=("code", "date", "trade_value"))) == 40     # close·volume 없으면 표본 질의로 잰다


def test_trade_value_unit_canary_catches_won_units(tmp_path):
    """trade_value 가 원 단위로 적재되면(백만원의 1e6 배) 읽는 자리에서 시끄럽게 실패한다(4f6ef61 배선 오류의 거울)."""
    from swing_it.storage import TradeValueUnitError

    con = connect(tmp_path / "t.db")
    _seed(con, _tv_rows(40, unit=1e6))
    with pytest.raises(TradeValueUnitError, match="백만원"):
        read_prices(con, cols=("code", "date", "trade_value"))
    with pytest.raises(TradeValueUnitError):
        read_prices(con, cols=("code", "date", "close", "volume", "trade_value"))
    assert len(read_prices(con, cols=("code", "date", "close"))) == 40           # trade_value 를 안 읽으면 검사하지 않는다
