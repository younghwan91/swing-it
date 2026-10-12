"""``earnings_leg`` — SEPA 합류 실적 다리(E)·가속·PEAD 랭크의 시점 안전성.

결과(수익률)를 보지 않는 피처라도 누출은 조용하다 — 정정본이 과거 셀에 들어가거나 공시 전 분기가 보이면
백테스트가 답을 보고 문제를 푼다. 합성 데이터로 명시 검증하고, 검증된 PEAD 경로
(``_yoy_vec`` → ``dropna`` → ``earnings_yoy_panel(knowledge_col=)``)와 ``yoy``·``age_days`` 가 같음을 못박는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing_it.features.earnings_leg import (
    E_MAX_AGE_DAYS,
    accel_flag,
    add_earnings_flags,
    e_flag,
    earnings_leg_panel,
    pead_cross_events,
    pead_rank,
    period_ordinal,
)
from swing_it.features.fundamentals import _yoy_vec, earnings_yoy_panel

KN = "knowledge_date"
COLS = ["code", "period", "avail_date", KN, "netinc", "netinc_prior",
        "revenue", "revenue_prior", "op_income", "op_income_prior"]


def _rows(*recs) -> pd.DataFrame:
    """(code, period, avail, knowledge, netinc, netinc_prior[, rev, rev_p, op, op_p])."""
    full = [tuple(r) + (np.nan,) * (len(COLS) - len(r)) for r in recs]
    return pd.DataFrame(full, columns=COLS)


def _at(panel, date, code="A"):
    row = panel[(panel["date"] == date) & (panel["code"] == code)]
    assert len(row) == 1, f"{code} {date} 행이 {len(row)}개"
    return row.iloc[0]


def _bdays(lo, hi):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(lo, hi)]


# --- 공시 전 사용 금지 ---------------------------------------------------------

def test_not_visible_before_avail_date():
    ea = _rows(("A", "2022Q1", "2022-05-16", "2022-05-16", 150.0, 100.0))
    p = earnings_leg_panel(ea, ["2022-05-13", "2022-05-16", "2022-05-17"], avail_col="avail_date")
    assert p[p["date"] == "2022-05-13"].empty, "공시 전 날짜에 행이 있으면 안 된다(= 모름)"
    r = _at(p, "2022-05-16")
    assert r["yoy"] == pytest.approx(0.5)
    assert r["age_days"] == 0 and r["age_tdays"] == 0
    assert _at(p, "2022-05-17")["age_days"] == 1


def test_newer_quarter_supersedes_on_its_avail_date():
    ea = _rows(
        ("A", "2022Q1", "2022-05-16", "2022-05-16", 150.0, 100.0),
        ("A", "2022Q2", "2022-08-15", "2022-08-15", 300.0, 100.0),
    )
    p = earnings_leg_panel(ea, ["2022-08-12", "2022-08-15"], avail_col="avail_date")
    assert _at(p, "2022-08-12")["period"] == "2022Q1"
    r = _at(p, "2022-08-15")
    assert r["period"] == "2022Q2" and r["yoy"] == pytest.approx(2.0)
    assert r["yoy_prev"] == pytest.approx(0.5)


def test_late_collected_filing_waits_for_knowledge_date():
    """공시일보다 늦게 수집된 행은 '알게 된 날' 부터. 그러나 age 는 공시일부터 잰다."""
    ea = _rows(("A", "2022Q1", "2022-05-16", "2022-09-01", 150.0, 100.0))
    p = earnings_leg_panel(ea, ["2022-06-01", "2022-09-01"], avail_col="avail_date")
    assert p[p["date"] == "2022-06-01"].empty
    r = _at(p, "2022-09-01")
    assert r["age_days"] == (pd.Timestamp("2022-09-01") - pd.Timestamp("2022-05-16")).days


def test_age_tdays_counts_from_first_trading_day_on_or_after_avail():
    ea = _rows(("A", "2022Q1", "2022-05-15", "2022-05-15", 150.0, 100.0))  # 일요일
    p = earnings_leg_panel(ea, _bdays("2022-05-10", "2022-05-20"), avail_col="avail_date")
    assert _at(p, "2022-05-16")["age_tdays"] == 0
    assert _at(p, "2022-05-16")["age_days"] == 1
    assert _at(p, "2022-05-20")["age_tdays"] == 4


# --- 정정공시(이중 시간축) -------------------------------------------------------

def test_restatement_of_latest_quarter_flips_only_after_knowledge_date():
    ea = _rows(
        ("A", "2026Q1", "2026-05-15", "2026-05-15", 110.0, 100.0),   # +10%
        ("A", "2026Q1", "2026-05-15", "2026-07-01", 200.0, 100.0),   # 정정 → +100%
    )
    p = earnings_leg_panel(ea, ["2026-06-30", "2026-07-01", "2026-07-02"], avail_col="avail_date")
    assert _at(p, "2026-06-30")["yoy"] == pytest.approx(0.10)
    assert _at(p, "2026-07-01")["yoy"] == pytest.approx(1.00)
    # 신선도는 정정으로 리셋되지 않는다.
    assert _at(p, "2026-07-02")["age_days"] == (pd.Timestamp("2026-07-02") - pd.Timestamp("2026-05-15")).days


def test_old_quarter_restatement_never_displaces_newest_quarter():
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 110.0, 100.0),
        ("A", "2024Q4", "2025-03-31", "2025-03-31", 150.0, 100.0),
        ("A", "2020Q1", "2020-05-15", "2025-06-01", 999.0, 100.0),   # 옛 분기 정정
    )
    p = earnings_leg_panel(ea, ["2025-04-01", "2025-06-02"], avail_col="avail_date")
    assert _at(p, "2025-06-02")["period"] == "2024Q4"
    assert _at(p, "2025-06-02")["yoy"] == pytest.approx(0.5)


def test_restated_previous_quarter_changes_yoy_prev_only_after_knowledge_date():
    ea = _rows(
        ("A", "2024Q1", "2024-05-15", "2024-05-15", 120.0, 100.0),   # +20%
        ("A", "2024Q2", "2024-08-14", "2024-08-14", 150.0, 100.0),   # +50%
        ("A", "2024Q1", "2024-05-15", "2024-09-02", 180.0, 100.0),   # Q1 정정 → +80%
    )
    p = add_earnings_flags(earnings_leg_panel(ea, ["2024-08-30", "2024-09-02"], avail_col="avail_date"))
    before, after = _at(p, "2024-08-30"), _at(p, "2024-09-02")
    assert before["yoy_prev"] == pytest.approx(0.2) and bool(before["accel"])
    assert after["yoy_prev"] == pytest.approx(0.8) and not bool(after["accel"])
    assert before["yoy"] == after["yoy"] == pytest.approx(0.5)


def test_restatement_does_not_disturb_unrelated_dates():
    base = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 110.0, 100.0),
        ("A", "2021Q1", "2021-05-15", "2021-05-15", 120.0, 100.0),
        ("A", "2023Q1", "2023-05-15", "2023-05-15", 130.0, 100.0),
    )
    restated = pd.concat(
        [base, _rows(("A", "2020Q1", "2020-05-15", "2025-01-01", 999.0, 100.0))], ignore_index=True)
    dates = ["2020-06-01", "2021-06-01", "2024-06-01", "2026-06-01"]
    pd.testing.assert_frame_equal(earnings_leg_panel(base, dates, avail_col="avail_date"), earnings_leg_panel(restated, dates, avail_col="avail_date"))


# --- PEAD 와 같은 YoY 정의 -------------------------------------------------------

def test_negative_prior_uses_abs_denominator_like_pead():
    ea = _rows(
        ("A", "2022Q1", "2022-05-16", "2022-05-16", 10.0, -100.0),    # 흑자 전환
        ("B", "2022Q1", "2022-05-16", "2022-05-16", -50.0, -100.0),   # 적자 축소
    )
    p = add_earnings_flags(earnings_leg_panel(ea, ["2022-05-16"], avail_col="avail_date"))
    a, b = _at(p, "2022-05-16", "A"), _at(p, "2022-05-16", "B")
    assert a["yoy"] == pytest.approx(1.1) and bool(a["e_flag"])
    assert b["yoy"] == pytest.approx(0.5) and not bool(b["e_flag"]), "적자면 YoY 가 커도 E 가 아니다"
    expect = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    assert list(p.sort_values("code")["yoy"]) == pytest.approx(list(expect))


def test_zero_or_missing_prior_falls_back_to_older_quarter_like_pead():
    """PEAD 는 YoY 가 NaN 인 행을 패널 전에 버린다 → 그 전 분기가 '최신' 으로 남는다."""
    ea = _rows(
        ("A", "2022Q1", "2022-05-16", "2022-05-16", 150.0, 100.0),
        ("A", "2022Q2", "2022-08-15", "2022-08-15", 150.0, 0.0),      # 분모 0
        ("B", "2022Q1", "2022-05-16", "2022-05-16", 150.0, 100.0),
        ("B", "2022Q2", "2022-08-15", "2022-08-15", 150.0, np.nan),   # 분모 결측
    )
    p = earnings_leg_panel(ea, ["2022-09-01"], avail_col="avail_date")
    for code in ("A", "B"):
        r = _at(p, "2022-09-01", code)
        assert r["period"] == "2022Q1" and r["yoy"] == pytest.approx(0.5)
        assert r["age_days"] == (pd.Timestamp("2022-09-01") - pd.Timestamp("2022-05-16")).days


def test_parity_with_validated_pead_panel():
    """yoy·age_days 가 검증된 PEAD 경로와 같다 — 정정·늦은 수집·NaN 분기가 섞여도."""
    ea = _rows(
        ("A", "2020Q1", "2020-05-15", "2020-05-15", 110.0, 100.0),
        ("A", "2020Q2", "2020-08-14", "2020-08-14", 90.0, -100.0),
        ("A", "2020Q3", "2020-11-14", "2020-11-14", 90.0, 0.0),
        ("A", "2020Q4", "2021-03-31", "2021-03-31", 400.0, 300.0),
        ("A", "2020Q2", "2020-08-14", "2021-02-01", 50.0, -100.0),    # 옛 분기 정정
        ("A", "2021Q1", "2021-05-15", "2021-05-15", 130.0, 110.0),
        ("A", "2021Q1", "2021-05-15", "2021-06-10", 160.0, 110.0),    # 최신 분기 정정
        ("B", "2020Q1", "2020-05-15", "2020-07-01", 50.0, 100.0),     # 늦은 수집
        ("B", "2020Q2", "2020-08-14", "2020-08-14", 70.0, 100.0),
        ("B", "2020Q3", "2020-11-14", "2020-11-14", 80.0, 100.0),
        ("C", "2020Q4", "2021-03-31", "2021-03-31", -5.0, 10.0),
    )
    dates = _bdays("2020-04-01", "2021-08-31")
    mine = earnings_leg_panel(ea, dates, avail_col="avail_date")

    ref_in = ea.assign(yoy=_yoy_vec(ea["netinc"], ea["netinc_prior"])).dropna(subset=["yoy"])
    ref = earnings_yoy_panel(ref_in, dates, knowledge_col=KN, avail_col="avail_date").dropna(subset=["yoy"])
    m = ref.merge(mine, on=["code", "date"], how="outer", suffixes=("_ref", ""), indicator=True)
    assert (m["_merge"] == "both").all(), "행 집합이 다르다"
    np.testing.assert_allclose(m["yoy"], m["yoy_ref"])
    np.testing.assert_array_equal(m["age_days"].astype(int), m["age_days_ref"].astype(int))


# --- 가속·E·랭크 -----------------------------------------------------------------

def test_accel_flag_requires_consecutive_finite_quarters():
    ea = _rows(
        ("A", "2022Q4", "2023-03-31", "2023-03-31", 130.0, 100.0),    # +30%
        ("A", "2023Q1", "2023-05-15", "2023-05-15", 150.0, 100.0),    # +50% > +30% (Q4→Q1 연속)
        ("B", "2022Q4", "2023-03-31", "2023-03-31", 130.0, 0.0),      # 직전 YoY 불가
        ("B", "2023Q1", "2023-05-15", "2023-05-15", 150.0, 100.0),
        ("C", "2022Q3", "2022-11-14", "2022-11-14", 110.0, 100.0),    # Q4 빠짐 → 연속 아님
        ("C", "2023Q1", "2023-05-15", "2023-05-15", 150.0, 100.0),
        ("D", "2022Q4", "2023-03-31", "2023-03-31", 190.0, 100.0),    # 감속
        ("D", "2023Q1", "2023-05-15", "2023-05-15", 150.0, 100.0),
    )
    p = add_earnings_flags(earnings_leg_panel(ea, ["2023-05-16"], avail_col="avail_date"))
    got = p.set_index("code")
    assert got.loc["A", "yoy_prev"] == pytest.approx(0.3) and bool(got.loc["A", "accel"])
    assert np.isnan(got.loc["B", "yoy_prev"]) and not bool(got.loc["B", "accel"])
    assert np.isnan(got.loc["C", "yoy_prev"]) and not bool(got.loc["C", "accel"])
    assert not bool(got.loc["D", "accel"])
    assert accel_flag(p).tolist() == p["accel"].tolist()
    assert int(period_ordinal(pd.Series(["2023Q1"]))[0] - period_ordinal(pd.Series(["2022Q4"]))[0]) == 1


def test_e_flag_thresholds():
    avail = pd.Timestamp("2022-05-16")
    ea = _rows(
        ("A", "2022Q1", "2022-05-16", "2022-05-16", 125.0, 100.0),    # 정확히 +25%
        ("B", "2022Q1", "2022-05-16", "2022-05-16", 124.9, 100.0),    # 문턱 아래
        ("C", "2022Q1", "2022-05-16", "2022-05-16", 0.0, -100.0),     # YoY +100% 이지만 순이익 0
    )
    on_edge = (avail + pd.Timedelta(days=E_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    past = (avail + pd.Timedelta(days=E_MAX_AGE_DAYS + 1)).strftime("%Y-%m-%d")
    p = earnings_leg_panel(ea, ["2022-05-16", on_edge, past], avail_col="avail_date")
    p["e"] = e_flag(p)
    e = p.set_index(["code", "date"])["e"]
    assert e[("A", "2022-05-16")] and e[("A", on_edge)]
    assert not e[("A", past)], "공시 후 90일 초과면 꺼진다"
    assert not e[("B", "2022-05-16")]
    assert not e[("C", "2022-05-16")], "당기 순이익 > 0 이어야 한다(0 은 아님)"


def test_pead_rank_is_daily_percentile_among_valued_and_eligible():
    ea = _rows(
        ("A", "2022Q1", "2022-05-16", "2022-05-16", 110.0, 100.0),
        ("B", "2022Q1", "2022-05-16", "2022-05-16", 150.0, 100.0),
        ("C", "2022Q1", "2022-05-16", "2022-05-16", 300.0, 100.0),
        ("D", "2022Q1", "2022-05-16", "2022-05-16", 900.0, 100.0),
    )
    p = earnings_leg_panel(ea, ["2022-05-16"], avail_col="avail_date")
    r = pead_rank(p).set_axis(p["code"])
    assert list(r[["A", "B", "C", "D"]]) == pytest.approx([0.25, 0.5, 0.75, 1.0])
    r2 = pead_rank(p, eligible=p["code"] != "D").set_axis(p["code"])
    assert np.isnan(r2["D"]) and r2["C"] == pytest.approx(1.0)


def test_revenue_and_op_yoy_come_from_the_same_row():
    ea = _rows(
        ("A", "2022Q1", "2022-05-16", "2022-05-16", 150.0, 100.0, 220.0, 200.0, 30.0, -10.0),
        ("A", "2022Q2", "2022-08-15", "2022-08-15", 150.0, 0.0, 999.0, 1.0, 999.0, 1.0),  # 순이익 YoY 불가
    )
    r = _at(earnings_leg_panel(ea, ["2022-09-01"], avail_col="avail_date"), "2022-09-01")
    assert r["period"] == "2022Q1"
    assert r["rev_yoy"] == pytest.approx(0.1) and r["op_yoy"] == pytest.approx(4.0)


def test_empty_input_returns_schema():
    p = earnings_leg_panel(_rows(("A", "2022Q1", "2022-05-16", "2022-05-16", 1.0, 0.0)), ["2022-06-01"], avail_col="avail_date")
    assert p.empty and "yoy" in p.columns


# --- pead_cross_events (탐색 기준선 pead_new) ---

def test_pead_cross_events_up_crossings_only():
    d = [f"2024-01-0{i}" for i in range(1, 9)]
    panel = pd.DataFrame({
        "code": ["A"] * 8 + ["B"] * 3,
        "date": d + d[:3],
        #        아래  위(사건) 유지  아래  NaN  위(사건) 정확히 문턱 유지  | B: 첫 행부터 위(사건), 유지, 아래
        "pead_rank": [0.5, 0.95, 0.99, 0.80, np.nan, 0.91, 0.90, 0.97] + [0.92, 0.93, 0.10],
    })
    ev = pead_cross_events(panel.sample(frac=1.0, random_state=0), threshold=0.90)   # 입력 순서 무관
    assert list(zip(ev["code"], ev["date"], strict=True)) == [
        ("A", "2024-01-02"), ("A", "2024-01-06"), ("B", "2024-01-01")]
    assert list(ev.columns) == list(panel.columns)
    assert pead_cross_events(panel.iloc[:0]).empty


def test_pead_cross_events_threshold_is_inclusive_and_nan_is_below():
    panel = pd.DataFrame({"code": ["A"] * 3, "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                          "pead_rank": [np.nan, 0.90, np.nan]})
    ev = pead_cross_events(panel, threshold=0.90)
    assert ev["date"].tolist() == ["2024-01-02"]
