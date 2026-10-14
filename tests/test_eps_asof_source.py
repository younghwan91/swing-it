"""``eps_asof`` 의 실적 원천 인자(30 보충 7~9) — ``earnings_dart_doc`` 는 접수일 **다음 거래일**부터, 정정 행은 정정 접수일부터,
``netinc_prior`` 결측이면 후보 아님. 기본 원천 ``earnings`` 경로는 그대로(같은 픽스처로 회귀)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing_it.features.eps_asof import eps_ok, eps_yoy, known_table
from swing_it.storage import connect, read_earnings

DAYS = pd.DatetimeIndex(pd.bdate_range("2010-05-10", "2010-06-30"))
# (code, period, knowledge_date, rcept_no, is_correction, basis, netinc, netinc_prior, prior_basis_note, shares_common, shares_total, shares_asof, source)
ORIG = ("000020", "2010Q1", "2010-05-14", "20100514000001", 0, "ofs", 100.0, 90.0, None, 1000, 1000, "2010-03-31", "dart_doc")
CORR = ("000020", "2010Q1", "2010-06-01", "20100601000002", 1, "ofs", 80.0, 90.0, None, 1000, 1000, "2010-03-31", "dart_doc")
NOPRIOR = ("000030", "2010Q1", "2010-05-14", "20100514000003", 0, "cfs", 100.0, None, "basis=cfs:prior_no_column", 500, 500, "2010-03-31", "dart_doc")


@pytest.fixture
def con():
    c = connect(":memory:")
    c.executemany("INSERT INTO earnings_dart_doc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [ORIG, CORR, NOPRIOR])
    c.executemany("INSERT INTO earnings VALUES (?,?,?,?,?,?,?,?,?,?)",
                  [("000020", "2010Q1", "2010-05-14", "2010-05-14", 100.0, 90.0, 1.0, 1.0, 1.0, 1.0)])
    yield c
    c.close()


def _row(d: str) -> int:
    return int(DAYS.get_loc(pd.Timestamp(d)))


def test_read_earnings_dart_doc_requires_all_versions(con):
    df = read_earnings(con, cols=("code", "period", "knowledge_date", "netinc", "netinc_prior"), all_versions=True, parse_dates=True,
                       table="earnings_dart_doc")
    assert len(df) == 3 and str(df["knowledge_date"].dtype).startswith("datetime64")
    with pytest.raises(ValueError):
        read_earnings(con, table="earnings_dart_doc")
    with pytest.raises(ValueError):
        read_earnings(con, table="earnings_dart_doc", all_versions=True, with_known=True)
    with pytest.raises(ValueError):
        read_earnings(con, table="nope", all_versions=True)


def test_dart_doc_known_next_trading_day_and_correction_from_its_own_date(con):
    codes = ["000020", "000030"]
    s = np.array([_row("2010-05-14"), _row("2010-05-17"), _row("2010-05-31"), _row("2010-06-01"), _row("2010-06-02")])
    ci = np.zeros(len(s), int)
    ok = eps_ok(codes, DAYS, ci, s, con, source="earnings_dart_doc")
    # 접수일(05-14 금) 당일은 모름 → 다음 거래일 05-17 부터 True(100 > 90 > 0) → 정정(06-01, 80 < 90) 접수일 당일까지 원본, 06-02 부터 False
    assert ok.tolist() == [False, True, True, True, False]
    yoy = eps_yoy(codes, DAYS, ci, s, con, source="earnings_dart_doc")
    assert np.isnan(yoy[0]) and yoy[1] == pytest.approx(100 / 90 - 1) and yoy[4] == pytest.approx(80 / 90 - 1)


def test_dart_doc_prior_missing_is_not_a_candidate(con):
    ok = eps_ok(["000020", "000030"], DAYS, np.array([1]), np.array([_row("2010-06-30")]), con, source="earnings_dart_doc")
    assert ok.tolist() == [False]
    kt = known_table(con, DAYS, source="earnings_dart_doc")
    assert list(kt.columns) >= ["code", "period", "knowledge_date", "is_correction", "netinc", "netinc_prior", "known", "krow"]
    assert kt["krow"].is_monotonic_increasing


def test_default_source_unchanged(con):
    """기본 ``earnings`` 경로: avail_date(05-14) 다음 거래일부터 — 원천 인자를 주지 않은 호출과 같다."""
    s = np.array([_row("2010-05-14"), _row("2010-05-17")])
    a = eps_ok(["000020"], DAYS, np.zeros(2, int), s, con)
    b = eps_ok(["000020"], DAYS, np.zeros(2, int), s, con, source="earnings")
    assert a.tolist() == b.tolist() == [False, True]
    with pytest.raises(ValueError):
        known_table(con, DAYS, source="nope")
