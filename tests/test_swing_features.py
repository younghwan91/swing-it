"""스윙 1호 피처 — 공시 촉매 분류·거래일 정렬, 테크니컬, 수급, 트레이드 시뮬레이터."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from swing_it.features.catalysts import (
    align_to_trading_days, catalyst_events, classify_catalyst, read_dart)
from swing_it.features.flows import net_to_volume, smart_money_net
from swing_it.features.technical import average_trade_value, moving_average, simple_return


@pytest.mark.parametrize("name, fam", [
    ("단일판매ㆍ공급계약체결", "supply_contract"),
    ("단일판매ㆍ공급계약체결(자율공시)", "supply_contract"),
    ("단일판매ㆍ공급계약해지", None),
    ("[기재정정]단일판매ㆍ공급계약체결", None),
    ("주요사항보고서(자기주식취득결정)", "buyback"),
    ("주요사항보고서(자기주식처분결정)", None),
    ("주식소각결정", "buyback"),
    ("신규시설투자등", "capex"),
    ("주요사항보고서(유형자산양도결정)", None),
    ("현금ㆍ현물배당결정", None),
    ("", None),
    (None, None),
])
def test_classify_catalyst(name, fam):
    assert classify_catalyst(name) == fam


def _dart_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE disclosures (rcept_no TEXT PRIMARY KEY, rcept_dt TEXT, code TEXT, corp_name TEXT,"
                " corp_cls TEXT, report_nm TEXT, material_type TEXT, is_correction INTEGER)")
    con.executemany("INSERT INTO disclosures VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def test_read_dart_merges_files_and_dedupes(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    _dart_db(a, [("1", "20240105", "000001", "", "Y", "신규시설투자등", "시설투자", 0)])
    _dart_db(b, [("1", "20240105", "000001", "", "Y", "신규시설투자등", "시설투자", 0),
                 ("2", "20240106", "000002", "", "K", "주식소각결정", "주주환원", 1)])
    df = read_dart([str(a), str(b), str(tmp_path / "missing.db")])
    assert list(df["rcept_no"]) == ["1", "2"]
    ev = catalyst_events(df)
    assert list(ev["code"]) == ["000001"]  # 정정(is_correction=1) 은 빠진다


def test_align_to_trading_days_rolls_weekend_and_merges():
    days = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    ev = pd.DataFrame({
        "code": ["A", "A", "B", "C"],
        "rcept_dt": pd.to_datetime(["2024-01-06", "2024-01-08", "2024-01-05", "2024-01-10"]),
        "family": ["capex", "buyback", "capex", "capex"],
        "report_nm": ["x"] * 4,
    })
    out = align_to_trading_days(ev, days).set_index("code")
    assert out.loc["A", "event_day"] == pd.Timestamp("2024-01-08")  # 토요일 접수 → 월요일, 같은 날 공시와 합침
    assert out.loc["A", "family"] == "buyback+capex" and out.loc["A", "n_filings"] == 2
    assert out.loc["B", "event_day"] == pd.Timestamp("2024-01-05")
    assert "C" not in out.index  # 마지막 거래일 뒤 접수는 버린다


def test_technical_is_point_in_time():
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=pd.date_range("2024-01-01", periods=4))
    ma = moving_average(close, 3)
    assert np.isnan(ma.iloc[1, 0]) and ma.iloc[2, 0] == pytest.approx(11.0)
    assert simple_return(close, 1).iloc[1, 0] == pytest.approx(0.1)
    tv = average_trade_value(close, 2)
    assert np.isnan(tv.iloc[0, 0]) and tv.iloc[1, 0] == pytest.approx(10.5)


def test_smart_money_net_keeps_null_unknown():
    sd = pd.DataFrame({"code": ["A", "A", "B"], "date": ["2024-01-02", "2024-01-03", "2024-01-02"],
                       "foreign_": [5, None, -3], "institution": [2, 4, 1]})
    net = smart_money_net(sd)
    assert net.loc[pd.Timestamp("2024-01-02"), "A"] == 7
    assert np.isnan(net.loc[pd.Timestamp("2024-01-03"), "A"])  # 한쪽 모름 → 합도 모름(0 아님)
    vol = pd.DataFrame({"A": [70.0, 0.0], "B": [10.0, 5.0]}, index=net.index)
    r = net_to_volume(net, vol)
    assert r.loc[pd.Timestamp("2024-01-02"), "A"] == pytest.approx(0.1)
    assert np.isnan(r.loc[pd.Timestamp("2024-01-03"), "B"])  # B 그날 수급 없음
