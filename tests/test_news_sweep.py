"""쌍 키·유니버스·순회 도달 재현 — daytrade-it ``test_universe.py`` 이식(+ DB id 경로)."""

from __future__ import annotations

import datetime as dt

import pytest
from krx_quant_core.market.session import KST

from swing_it.news.sweep import (
    SWEEP_INTERVAL,
    SweepCursor,
    SweepRequest,
    build_article_url,
    db_article_id,
    live_article_id,
    news_id_from_url,
    pair_key_article,
    select_universe,
    simulate_sweep,
)


class TestArticleKeys:
    def test_matches_operational_key_formula(self) -> None:
        # daytrade-it 실측 고정값(운영 데몬 파일명 = 그 기사의 uuid5 키).
        assert live_article_id("newspim_20260915000210") == "006f4150-d62b-5770-a154-fb84bd7b23a5"
        assert live_article_id("seokyung_2KHIC2PH6Z") == "595eead7-3889-57cd-bdc5-fc299f660c70"

    def test_db_id_path_equals_news_id_path(self) -> None:
        db_id = db_article_id("seokyung_2KHIC2PH6Z")
        assert db_id.startswith("toss:") and len(db_id) == 5 + 12
        assert pair_key_article(db_id) == live_article_id("seokyung_2KHIC2PH6Z")

    def test_url_round_trip(self) -> None:
        url = build_article_url("abc")
        assert url == "https://tossinvest.com/news?contentType=news&contentParams=%7B%22id%22%3A%22abc%22%7D"
        assert news_id_from_url(url) == "abc"
        assert news_id_from_url(None) is None
        assert news_id_from_url("https://example.com/x") is None

    def test_different_ids_differ(self) -> None:
        assert live_article_id("abc_123") != live_article_id("abc_124")


class TestSelectUniverse:
    def test_sorts_by_value_desc_then_code_asc_on_ties(self) -> None:
        values = {"005930": 100.0, "000660": 100.0, "035420": 200.0}
        assert select_universe(values) == ["035420", "000660", "005930"]

    def test_caps_at_size(self) -> None:
        values = {f"{i:06d}": float(1000 - i) for i in range(10)}
        assert select_universe(values, size=3) == ["000000", "000001", "000002"]

    def test_ignores_non_positive_values_and_non_six_digit_codes(self) -> None:
        assert select_universe({"005930": 0.0, "000660": -5.0, "035420": 1.0}) == ["035420"]
        assert select_universe({"00593": 100.0, "0059300": 100.0, "AB1234": 100.0, "005930": 50.0}) == ["005930"]

    def test_default_size_is_300(self) -> None:
        values = {f"{i:06d}": float(2000 - i) for i in range(350)}
        assert len(select_universe(values)) == 300


DAY = dt.date(2026, 9, 16)
STALE = dt.datetime(2026, 9, 15, 15, 20, tzinfo=KST)


def t(h: int, m: int, s: int = 0) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(h, m, s), tzinfo=KST)


def _reqs(trace: list[SweepRequest]) -> list[tuple]:
    return [(r.kind, r.code, r.page, r.news_id) for r in trace]


class TestSweepCursor:
    def test_visits_codes_in_order_and_detail_follows_its_page(self) -> None:
        cur = SweepCursor(["A", "B"], STALE, page_size=3)
        r = cur.next_request()
        assert (r.kind, r.code, r.page) == ("page", "A", 1)
        assert cur.page_result(r, [("n1", t(8, 0)), ("n0", STALE - dt.timedelta(seconds=1))]) == ["n1"]
        d = cur.next_request()
        assert (d.kind, d.code, d.news_id) == ("detail", "A", "n1")
        assert cur.next_request().code == "B"

    def test_seen_via_any_code_costs_nothing_and_stale_or_undated_skipped(self) -> None:
        cur = SweepCursor(["A", "B"], STALE, page_size=3)
        cur.page_result(cur.next_request(), [("n1", t(8, 0))])
        cur.next_request()  # detail n1
        r = cur.next_request()
        assert cur.page_result(r, [("n1", t(8, 0)), ("n2", None), (None, t(8, 0))]) == []
        assert cur.next_request() == SweepRequest("page", "A", 1)

    def test_pages_on_while_the_page_is_full_of_new_items(self) -> None:
        cur = SweepCursor(["A"], STALE, page_size=2)
        r = cur.next_request()
        cur.page_result(r, [("n4", t(8, 0)), ("n3", t(8, 0))])
        assert [cur.next_request().news_id for _ in range(2)] == ["n4", "n3"]
        r2 = cur.next_request()
        assert (r2.kind, r2.code, r2.page) == ("page", "A", 2)
        cur.page_result(r2, [("n2", t(8, 0)), ("n1", STALE - dt.timedelta(1))])  # one stale: stop
        assert cur.next_request().news_id == "n2"
        assert cur.next_request() == SweepRequest("page", "A", 1)

    def test_page_result_is_required_before_the_next_request(self) -> None:
        cur = SweepCursor(["A"], STALE)
        cur.next_request()
        with pytest.raises(RuntimeError):
            cur.next_request()


class TestSimulateSweep:
    def test_detail_costs_an_interval_and_discovery_is_detail_completion(self) -> None:
        arrivals = {"A": [("n1", t(7, 0))], "B": [("n2", t(7, 30)), ("n1", t(7, 0))]}
        trace: list[SweepRequest] = []
        found = simulate_sweep(["A", "B"], DAY, arrivals, STALE, end=dt.time(8, 0, 6), trace=trace)
        assert _reqs(trace) == [
            ("page", "A", 1, None),
            ("detail", "A", 1, "n1"),
            ("page", "B", 1, None),
            ("detail", "B", 1, "n2"),
            ("page", "A", 1, None),
            ("page", "B", 1, None),
        ]
        assert found == {"n1": t(8, 0, 2), "n2": t(8, 0, 4)}

    def test_article_not_visible_before_created(self) -> None:
        found = simulate_sweep(["A"], DAY, {"A": [("late", t(8, 0, 1))]}, STALE, end=dt.time(8, 0, 5))
        assert found == {"late": t(8, 0, 3)}

    def test_paging_through_a_full_backlog(self) -> None:
        arrivals = {"A": [(f"n{i}", t(7, 0, i)) for i in range(5)]}
        trace: list[SweepRequest] = []
        found = simulate_sweep(["A"], DAY, arrivals, STALE, end=dt.time(8, 1), page_size=2, trace=trace)
        assert [(r.kind, r.page, r.news_id) for r in trace[:9]] == [
            ("page", 1, None),
            ("detail", 1, "n4"),
            ("detail", 1, "n3"),
            ("page", 2, None),
            ("detail", 2, "n2"),
            ("detail", 2, "n1"),
            ("page", 3, None),
            ("detail", 3, "n0"),
            ("page", 1, None),
        ]
        assert found["n0"] == t(8, 0, 8)

    def test_stops_at_end_and_excludes_stale(self) -> None:
        arrivals = {"A": [("old", STALE - dt.timedelta(seconds=1)), ("n", t(15, 19, 59))]}
        assert simulate_sweep(["A"], DAY, arrivals, STALE, start=dt.time(15, 19, 58)) == {}

    def test_empty_universe_and_interval(self) -> None:
        assert simulate_sweep([], DAY, {}, STALE) == {}
        assert dt.timedelta(seconds=1) == SWEEP_INTERVAL
