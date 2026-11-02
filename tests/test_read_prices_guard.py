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


# --- 2단계 검증 지적: end·정렬·가드의 구멍 ---

def _adj(code, date, source="kiwoom"):
    return _row(code, date, source)


def test_end_is_a_ceiling_and_rows_are_ordered(tmp_path):
    con = connect(tmp_path / "t.db")
    _seed(con, [_adj("000030", "2020-01-03"), _adj("000020", "2020-01-06"), _adj("000020", "2020-01-02"), _adj("000040", "2020-01-02", "naver")])
    df = read_prices(con, end="2020-01-03")
    assert list(zip(df["code"], df["date"])) == [("000020", "2020-01-02"), ("000030", "2020-01-03"), ("000040", "2020-01-02")]
    con.close()


@pytest.mark.parametrize("end", ["2026-09-14", "2026-10-01"])
def test_end_in_the_aftermarket_window_is_refused(tmp_path, end):
    """정문은 2026-09-14~ 정규장 값을 만들지 못한다. 그 구간을 달라는 것은 거부한다."""
    from swing_it.pipeline.checks import CheckError
    con = connect(tmp_path / "t.db")
    _seed(con, [_adj("000020", "2020-01-02")])
    with pytest.raises(CheckError, match="애프터마켓"):
        read_prices(con, end=end)
    con.close()


def test_end_just_before_the_aftermarket_window_is_allowed(tmp_path):
    con = connect(tmp_path / "t.db")
    _seed(con, [_adj("000020", "2026-09-11")])
    assert len(read_prices(con, end="2026-09-13")) == 1
    con.close()


def test_a_garbage_end_is_refused(tmp_path):
    from swing_it.pipeline.checks import CheckError
    con = connect(tmp_path / "t.db")
    with pytest.raises(CheckError, match="날짜"):
        read_prices(con, end="soon")
    con.close()


def test_without_end_the_old_callers_still_read_the_whole_table(tmp_path):
    """기본값 정책 — ``end=None`` 은 옛 호출 경로 그대로다(표 전체). 새 러너는 린트 (k) 가 ``end=`` 를 요구한다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_adj("000020", "2026-10-01")])
    assert len(read_prices(con)) == 1
    con.close()


def test_guard_range_follows_end(tmp_path):
    """``end`` 전에 폐지 종목이 없었으면 그 구간에서는 요구할 수 없다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_adj("000020", "2019-01-02"), _adj("000030", "2020-06-01", "naver")])
    assert len(read_prices(con, end="2019-12-31")) == 1
    con.close()


def test_a_query_without_code_cannot_pass_the_survivor_guard(tmp_path):
    """code 를 안 뽑으면 폐지 포함을 볼 수 없다 — 조용히 통과시키지 않는다."""
    con = connect(tmp_path / "t.db")
    _seed(con, [_adj("000020", "2020-01-02"), _adj("000030", "2020-01-02", "naver")])
    with pytest.raises(AssertionError, match="code"):
        read_prices(con, cols=("date", "close"))
    assert len(read_prices(con, cols=("date", "close"), require_delisted=False)) == 2
    con.close()


def test_a_table_without_delisted_rows_cannot_pass_when_the_default_table_has_them(tmp_path):
    """``daily_bars_raw`` 에는 폐지 백필이 없다(source=kiwoom_raw 뿐) — 가드가 "폐지 시세 없음"으로 통과해 생존자만 읽혔다."""
    con = connect(tmp_path / "t.db")
    con.execute("CREATE TABLE daily_bars_raw AS SELECT * FROM daily_bars_adjusted WHERE 0")
    _seed(con, [_adj("000030", "2020-01-02", "naver")])
    con.execute(f"INSERT INTO daily_bars_raw({','.join(COLS)}) VALUES({','.join(['?'] * len(COLS))})", _adj("000020", "2020-01-02", "kiwoom_raw"))
    with pytest.raises(AssertionError, match="생존편향"):
        read_prices(con, table="daily_bars_raw")
    assert len(read_prices(con, table="daily_bars_raw", require_delisted=False)) == 1
    con.close()
