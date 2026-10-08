"""스윙 3호 뉴스 다리 N(:mod:`swing_it.news.daily_leg`) — 합성 데이터만, DB·가격·수익 없음.

시점 매핑(장 마감 후·주말·휴장일·overnight), 60일 창 경계, judge 우선순위, 오류 행 = 모름, 필터, news_known.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from krx_quant_core.market.session import KST

from swing_it.news.daily_leg import (
    JUDGE_PRIORITY,
    daily_counts,
    is_n_article,
    map_signal_days,
    n_flag_panel,
    n_flag_values,
    pick_judgments,
    signal_day,
)
from swing_it.news.prefilter import NO_FILTER

# 2025-10-01(수) ~ : 10-03(금) 개천절, 10-06~10-08 추석 연휴, 10-09(목) 한글날 휴장 → 10-02 다음 거래일 = 10-10
TD = [dt.date(2025, 9, 29), dt.date(2025, 9, 30), dt.date(2025, 10, 1), dt.date(2025, 10, 2), dt.date(2025, 10, 10),
      dt.date(2025, 10, 13), dt.date(2025, 10, 14)]


def kst(y, m, d, hh, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=KST)


N_OUT = {"subject": "main", "sentiment_direction": 1, "persistence": "multi_quarter", "freshness": "new",
         "reports_price_move": False}


def out(**kw):
    return {**N_OUT, **kw}


# --- 시점 매핑 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "want"),
    [
        (kst(2025, 10, 1, 9, 0), dt.date(2025, 10, 1)),  # 장중 → 그날
        (kst(2025, 10, 1, 15, 29), dt.date(2025, 10, 1)),
        (kst(2025, 10, 1, 15, 30), dt.date(2025, 10, 2)),  # 15:30 정각부터 다음 거래일
        (kst(2025, 10, 1, 18, 0), dt.date(2025, 10, 2)),  # 장 마감 후
        (kst(2025, 10, 1, 7, 0), dt.date(2025, 10, 1)),  # 장 전 → 그날
        (kst(2025, 10, 2, 16, 0), dt.date(2025, 10, 10)),  # 연휴 전 마감 후 → 연휴 뒤 첫 거래일
        (kst(2025, 10, 4, 12, 0), dt.date(2025, 10, 10)),  # 토요일
        (kst(2025, 10, 9, 10, 0), dt.date(2025, 10, 10)),  # 휴장일(한글날) 장중 시각
        (kst(2025, 10, 11, 10, 0), dt.date(2025, 10, 13)),  # 토 → 월
        (kst(2025, 10, 14, 15, 31), None),  # 달력 뒤
    ],
)
def test_close_mode(t, want):
    assert signal_day(t, TD, "close") == want


@pytest.mark.parametrize(
    ("t", "want"),
    [
        (kst(2025, 10, 1, 18, 0), dt.date(2025, 10, 1)),  # 마감 후 → 다음날 시가 진입에 쓴다 → 신호일 = 그날
        (kst(2025, 10, 2, 7, 59), dt.date(2025, 10, 1)),  # 진입일 08:00 전 → 전 거래일 신호
        (kst(2025, 10, 2, 8, 0), dt.date(2025, 10, 1)),  # 08:00 정각 포함
        (kst(2025, 10, 2, 8, 1), dt.date(2025, 10, 2)),  # 08:00 넘으면 그날 신호(다음날 진입)
        (kst(2025, 10, 4, 12, 0), dt.date(2025, 10, 2)),  # 연휴 중 → 10-10 시가 진입 → 신호일 10-02
        (kst(2025, 9, 29, 7, 0), None),  # 진입일이 달력 첫날 → 전 거래일 모름
        (kst(2025, 10, 14, 9, 0), None),  # 진입일이 달력 뒤 → 모름(마지막 날로 몰지 않는다)
    ],
)
def test_overnight_mode(t, want):
    assert signal_day(t, TD, "overnight") == want


def test_close_never_earlier_than_overnight():
    """close 는 모든 기사를 overnight 와 같거나 늦은 신호일에 배정한다(더 보수적)."""
    ts = [kst(2025, 9, 29, 9) + dt.timedelta(minutes=37 * i) for i in range(400)]
    c = map_signal_days(ts, TD, "close")
    o = map_signal_days(ts, TD, "overnight")
    both = c.notna() & o.notna()
    assert both.sum() > 100
    assert all(a >= b for a, b in zip(c[both], o[both], strict=True))


def test_naive_time_refused():
    with pytest.raises(ValueError):
        map_signal_days([dt.datetime(2025, 10, 1, 10)], TD)
    with pytest.raises(ValueError):
        map_signal_days([kst(2025, 10, 1, 10)], TD, "weird")


def test_utc_input_converted():
    # 2025-10-01 06:29 UTC = 15:29 KST → 그날, 06:30 UTC = 15:30 KST → 다음 거래일
    u = dt.timezone.utc
    assert signal_day(dt.datetime(2025, 10, 1, 6, 29, tzinfo=u), TD) == dt.date(2025, 10, 1)
    assert signal_day(dt.datetime(2025, 10, 1, 6, 30, tzinfo=u), TD) == dt.date(2025, 10, 2)


# --- N 기사 정의 --------------------------------------------------------------------------


def test_is_n_article():
    assert is_n_article(out())
    assert is_n_article(out(subject="partial", persistence="structural"))
    assert is_n_article(out(sentiment_direction="1"))  # 문자열도 정수로
    assert is_n_article(out(reports_price_move="false"))
    for bad in (dict(subject="mention"), dict(sentiment_direction=0), dict(sentiment_direction=-1),
                dict(persistence="one_off"), dict(freshness="follow_up"), dict(freshness="repeat"),
                dict(reports_price_move=True), dict(reports_price_move=None)):
        assert not is_n_article(out(**bad)), bad
    assert not is_n_article(None)
    assert not is_n_article({})


# --- judge 우선순위 · 오류 -----------------------------------------------------------------


def jrows(rows):
    return pd.DataFrame(rows, columns=["article_id", "code", "judge", "output", "error"])


def test_priority_prefers_v2r3():
    assert JUDGE_PRIORITY == ("v2r3-haiku55", "v2-haiku45")
    p = pick_judgments(jrows([
        ("a", "000001", "v2-haiku45", out(freshness="repeat"), None),
        ("a", "000001", "v2r3-haiku55", out(), None),
    ]))
    assert len(p) == 1 and p.loc[0, "judge"] == "v2r3-haiku55" and p.loc[0, "status"] == "ok"
    assert is_n_article(p.loc[0, "output"])


def test_priority_falls_back_on_error():
    """우선 judge 가 오류면 다음 judge 의 성공을 쓴다. 둘 다 오류면 status=error(모름)."""
    p = pick_judgments(jrows([
        ("a", "000001", "v2r3-haiku55", None, "schema: x"),
        ("a", "000001", "v2-haiku45", out(), None),
        ("b", "000001", "v2r3-haiku55", None, "schema: x"),
        ("b", "000001", "v2-haiku45", None, "schema: y"),
        ("c", "000001", "v2-haiku55", out(), None),  # 우선순위 밖 judge 는 버린다
    ])).set_index("article_id")
    assert p.loc["a", "judge"] == "v2-haiku45" and p.loc["a", "status"] == "ok"
    assert p.loc["b", "status"] == "error" and p.loc["b", "judge"] == "v2r3-haiku55" and p.loc["b", "output"] is None
    assert "c" not in p.index


def test_priority_is_per_code():
    p = pick_judgments(jrows([
        ("a", "000001", "v2-haiku45", out(), None),
        ("a", "000002", "v2r3-haiku55", out(), None),
    ])).set_index("code")
    assert p.loc["000001", "judge"] == "v2-haiku45" and p.loc["000002", "judge"] == "v2r3-haiku55"


# --- 일별 건수 ----------------------------------------------------------------------------


def feed(rows):
    df = pd.DataFrame(rows, columns=["article_id", "code", "available_at", "title", "related_n"])
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True).dt.tz_convert(KST)
    return df


def test_daily_counts_and_errors():
    f = feed([
        ("a", "000001", kst(2025, 10, 1, 10), "호재 A", 1),
        ("b", "000001", kst(2025, 10, 1, 18), "호재 B", 1),  # 마감 후 → 10-02
        ("c", "000001", kst(2025, 10, 1, 11), "악재", 1),
        ("d", "000001", kst(2025, 10, 1, 12), "오류", 1),
        ("e", "000001", kst(2025, 10, 1, 13), "미판정", 1),
        ("a", "000001", kst(2025, 10, 1, 10), "호재 A", 1),  # 중복 피드 행은 한 번
    ])
    picked = pick_judgments(jrows([
        ("a", "000001", "v2r3-haiku55", out(), None),
        ("b", "000001", "v2-haiku45", out(persistence="one_off"), None),
        ("c", "000001", "v2r3-haiku55", out(sentiment_direction=-1), None),
        ("d", "000001", "v2r3-haiku55", None, "schema: z"),
        ("z", "000009", "v2r3-haiku55", out(), None),  # 피드에 없는 판정 → 가용 시각 모름 → 버림
    ]))
    d = daily_counts(f, picked, TD).set_index("date")
    r = d.loc[dt.date(2025, 10, 1)]
    assert (r.n_feed, r.n_articles, r.n_error, r.n_pos_persistent, r.n_pos, r.n_neg) == (4, 2, 1, 1, 1, 1)
    assert r.judge == "v2r3-haiku55"
    r2 = d.loc[dt.date(2025, 10, 2)]
    assert (r2.n_feed, r2.n_articles, r2.n_pos_persistent, r2.n_pos) == (1, 1, 0, 1)
    assert r2.judge == "v2-haiku45"
    assert set(d["code"]) == {"000001"}


def test_daily_counts_mixed_judges_joined():
    f = feed([("a", "000001", kst(2025, 10, 1, 10), "x", 1), ("b", "000001", kst(2025, 10, 1, 11), "y", 1)])
    picked = pick_judgments(jrows([("a", "000001", "v2r3-haiku55", out(), None),
                                   ("b", "000001", "v2-haiku45", out(), None)]))
    d = daily_counts(f, picked, TD)
    assert d.loc[0, "judge"] == "v2-haiku45+v2r3-haiku55" and d.loc[0, "n_pos_persistent"] == 2


def test_daily_counts_prefilter_both_sides():
    """STAGE1 필터는 피드 분모와 판정 쌍 모두에 걸린다(필터 걸린 판정 N 기사도 세지 않는다)."""
    f = feed([
        ("a", "000001", kst(2025, 10, 1, 10), "[특징주] 급등", 1),  # 스윙 태그
        ("b", "000001", kst(2025, 10, 1, 10), "여러 종목", 12),  # 팬아웃 > 11
        ("c", "000001", kst(2025, 10, 1, 10), "수주 공시", 1),
    ])
    picked = pick_judgments(jrows([(x, "000001", "v2-haiku45", out(), None) for x in "abc"]))
    d = daily_counts(f, picked, TD)
    assert (d.loc[0, "n_feed"], d.loc[0, "n_pos_persistent"]) == (1, 1)
    d0 = daily_counts(f, picked, TD, cfg=NO_FILTER)
    assert (d0.loc[0, "n_feed"], d0.loc[0, "n_pos_persistent"]) == (3, 3)


def test_daily_counts_overnight_mode():
    f = feed([("b", "000001", kst(2025, 10, 1, 18), "x", 1)])
    picked = pick_judgments(jrows([("b", "000001", "v2r3-haiku55", out(), None)]))
    assert daily_counts(f, picked, TD, mode="close").loc[0, "date"] == dt.date(2025, 10, 2)
    assert daily_counts(f, picked, TD, mode="overnight").loc[0, "date"] == dt.date(2025, 10, 1)


# --- 60일 창 · news_known --------------------------------------------------------------------


def daily(rows):
    cols = ["code", "date", "n_pos_persistent", "n_pos", "n_neg", "n_articles", "n_error", "n_feed", "judge"]
    return pd.DataFrame(rows, columns=cols)


def test_window_60_calendar_days_boundaries():
    s = dt.date(2025, 1, 10)
    d = daily([("000001", s, 1, 1, 0, 1, 0, 1, "v2-haiku45")])
    dates = [s - dt.timedelta(days=1), s, s + dt.timedelta(days=59), s + dt.timedelta(days=60)]
    p = n_flag_panel(d, dates).set_index("date")
    assert not p.loc[dates[0], "n_flag"]  # 기사 전 — 미래 기사를 쓰지 않는다
    assert p.loc[s, "n_flag"]  # 같은 날(매핑 신호일 = d) 포함
    assert p.loc[dates[2], "n_flag"]  # 59일 뒤 포함
    assert not p.loc[dates[3], "n_flag"]  # 60일 뒤 = (d−60, d] 밖
    assert p.loc[dates[2], "judges_60"] == "v2-haiku45" and p.loc[dates[3], "judges_60"] == ""


def test_window_sums_and_coverage():
    s = dt.date(2025, 3, 1)
    d = daily([
        ("000001", s, 0, 1, 0, 2, 1, 4, "v2r3-haiku55"),
        ("000001", s + dt.timedelta(days=10), 1, 1, 0, 1, 0, 1, "v2-haiku45"),
        ("000002", s, 0, 0, 0, 0, 0, 5, ""),  # 피드는 있는데 판정 0 (유니버스 밖) → 모름
    ])
    t = s + dt.timedelta(days=10)
    p = n_flag_panel(d, [s, t], codes=["000001", "000002", "000003"]).set_index(["code", "date"])
    a = p.loc[("000001", s)]
    assert (a.n_articles_60, a.n_error_60, a.n_feed_60) == (2, 1, 4) and a.cov_60 == 0.5
    assert not a.n_flag and a.news_known  # cov 0.5 ≥ 0.5
    b = p.loc[("000001", t)]
    assert b.n_flag and b.n_pos_persistent_60 == 1 and b.judges_60 == "v2-haiku45+v2r3-haiku55"
    c = p.loc[("000002", s)]
    assert c.cov_60 == 0 and not c.news_known  # 판정 안 된 피드 → 없는 뉴스가 아니라 모름
    z = p.loc[("000003", s)]
    assert z.n_feed_60 == 0 and np.isnan(z.cov_60) and z.news_known  # 피드 0건 → 0 으로 안다
    v = n_flag_values(p.reset_index()).to_numpy()
    assert list(np.nan_to_num(v, nan=-1)) == [0.0, 1.0, -1.0, -1.0, 0.0, 0.0]


def test_errors_lower_coverage_to_unknown():
    s = dt.date(2025, 3, 1)
    d = daily([("000001", s, 0, 0, 0, 1, 3, 4, "v2r3-haiku55")])
    p = n_flag_panel(d, [s])
    assert p.loc[0, "cov_60"] == 0.25 and not p.loc[0, "news_known"]
    assert n_flag_panel(d, [s], min_coverage=0.2).loc[0, "news_known"]


def test_positive_is_known_even_with_low_coverage():
    s = dt.date(2025, 3, 1)
    d = daily([("000001", s, 1, 1, 0, 1, 0, 10, "v2r3-haiku55")])
    p = n_flag_panel(d, [s])
    assert p.loc[0, "n_flag"] and p.loc[0, "news_known"] and p.loc[0, "cov_60"] == 0.1


def test_end_to_end_pit_no_lookahead():
    """장 마감 후 기사는 그날 신호의 N 에 들어가지 않는다(close 기본)."""
    f = feed([("a", "000001", kst(2025, 10, 2, 16, 0), "수주", 1)])
    picked = pick_judgments(jrows([("a", "000001", "v2r3-haiku55", out(), None)]))
    d = daily_counts(f, picked, TD)
    p = n_flag_panel(d, TD).set_index("date")
    assert not p.loc[dt.date(2025, 10, 2), "n_flag"]
    assert p.loc[dt.date(2025, 10, 10), "n_flag"]
