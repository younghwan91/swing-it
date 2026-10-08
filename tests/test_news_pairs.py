"""쌍 생성기의 순수 핵심 — 창·유니버스·수집 원장 커버리지·순회 도달·필터·출력 형식. DB 불요.

DB 위 패리티(2026-10-08): 1단계 창 2025-09-01..2026-02-28 재생성 164,864쌍 = 옛 파일 164,864쌍(키 완전 일치),
1단계 판정 캐시 36,171행의 input_hash 100% 재현. 전 구간 2023-03-20..2026-02-27 도 637,343 = 637,343.
"""

from __future__ import annotations

import datetime as dt

from krx_quant_core.market.session import KST

from swing_it.news.prefilter import NO_FILTER, STAGE1
from swing_it.news.pairs import (
    Article,
    DayReport,
    Stock,
    bucket_articles,
    build_pairs,
    coverage_from_rows,
    covered,
    day_pairs,
    fetch_from_row,
    parse_days,
)
from swing_it.news.sweep import pair_key_article

PREV = dt.date(2025, 9, 1)
DAY = dt.date(2025, 9, 2)
NEXT = dt.date(2025, 9, 3)
LO = dt.datetime.combine(PREV, dt.time(15, 20), tzinfo=KST)
HI = dt.datetime.combine(DAY, dt.time(15, 20), tzinfo=KST)
DONE_AT = dt.datetime(2026, 9, 17, 2, 0, tzinfo=KST)
KEYS = {"key", "code", "name", "market", "sector", "title", "body", "summary", "category", "day", "month",
        "session", "published_at", "related_n"}

STOCKS = {
    "000001": Stock("가나", "거래소", "전기/전자"),
    "000002": Stock("다라", "코스닥", None),
    "000003": Stock("마바", "코스닥", "금융"),
}


def at(d: dt.date, h: int, m: int = 0) -> dt.datetime:
    return dt.datetime.combine(d, dt.time(h, m), tzinfo=KST)


def art(i: int, pub: dt.datetime, codes: tuple[str, ...], title: str = "", summary: str = "요약") -> Article:
    return Article(db_id=f"toss:{i:012x}", news_id=f"n{i}", title=title or f"제목 {i}", summary=summary,
                   published_at=pub, codes=codes)


def full_cov(*codes: str) -> dict:
    return coverage_from_rows((c, "2023-03-16", "done", None, DONE_AT) for c in codes)


class TestCoverage:
    def test_done_covers_from_since_midnight(self) -> None:
        f = fetch_from_row("2023-03-16", "done", None, DONE_AT)
        assert f.start == dt.datetime(2023, 3, 16, tzinfo=KST)

    def test_page_cap_starts_at_oldest(self) -> None:
        oldest = at(DAY, 10)
        f = fetch_from_row(dt.date(2023, 3, 16), "page_cap", oldest, DONE_AT)
        assert f.start == oldest
        cov = coverage_from_rows([("A", "2023-03-16", "page_cap", oldest, DONE_AT)])
        assert not covered("A", cov, LO, HI)  # 창이 oldest 보다 먼저 시작 — 일부만 받았다
        assert covered("A", cov, at(DAY, 10), at(NEXT, 15, 20))

    def test_failed_and_missing_cover_nothing(self) -> None:
        cov = coverage_from_rows([("A", "2023-03-16", "failed", None, DONE_AT)])
        assert cov == {"A": []}
        assert not covered("A", cov, LO, HI) and not covered("B", cov, LO, HI)
        assert covered("B", None, LO, HI)

    def test_fetch_must_finish_after_the_window(self) -> None:
        """끝난 시각 < 창 끝이면 그 창 뒷부분은 못 받았다(옛 생성기엔 없던 조건 — 1단계엔 영향 없음)."""
        cov = coverage_from_rows([("A", "2023-03-16", "done", None, at(DAY, 12))])
        assert not covered("A", cov, LO, HI)
        cov["A"] += coverage_from_rows([("A", "2025-09-02", "done", None, at(DAY, 18))])["A"]
        assert not covered("A", cov, LO, HI)  # 증분은 9/2 00:00 부터라 창 시작(9/1 15:20)을 못 덮는다
        cov["A"] += coverage_from_rows([("A", "2025-09-01", "done", None, at(DAY, 18))])["A"]
        assert covered("A", cov, LO, HI)


def test_bucket_articles_assigns_each_article_to_one_window() -> None:
    a = [art(1, at(PREV, 15, 19), ("000001",)), art(2, at(PREV, 15, 20), ("000001",)), art(3, at(DAY, 15, 20), ("000001",))]
    b = bucket_articles(a, [PREV, DAY, NEXT])
    assert [x.db_id for x in b[PREV]] == [a[0].db_id]
    assert [x.db_id for x in b[DAY]] == [a[1].db_id]
    assert [x.db_id for x in b[NEXT]] == [a[2].db_id]


class TestDayPairs:
    def test_window_universe_coverage_and_row_shape(self) -> None:
        arts = [
            art(1, at(DAY, 8, 30), ("000001", "000002", "999999")),  # 999999: 유니버스 밖
            art(2, at(PREV, 15, 19), ("000001",)),  # 창 이전
            art(3, at(DAY, 15, 20), ("000001",)),  # 창 끝(배타)
            art(4, at(DAY, 10, 0), ("000003",), summary=""),  # 장중, 요약 없음 → body = 제목
        ]
        rep = DayReport(DAY)
        rows = day_pairs(arts, DAY, PREV, ["000001", "000002", "000003"], full_cov("000001", "000003"), STOCKS,
                         sweep=False, report=rep)
        assert [r["code"] for r in rows] == ["000001", "000003"]  # 000002 는 수집 원장에 없다
        r = rows[0]
        assert set(r) == KEYS
        assert r["key"] == f"{pair_key_article(arts[0].db_id)}|000001"
        assert (r["name"], r["market"], r["sector"], r["category"]) == ("가나", "거래소", "전기/전자", None)
        assert (r["day"], r["month"], r["session"], r["related_n"]) == ("2025-09-02", "2025-09", "pre_market", 3)
        assert r["published_at"] == "2025-09-02T08:30:00+09:00"
        assert rows[1]["session"] == "intraday" and rows[1]["body"] == "제목 4" and rows[1]["summary"] == ""
        assert rep.stats["excluded_pairs"] == 1 and rep.missing == 1 and rep.uncovered == 1
        assert "pairs 2" in rep.line()

    def test_related_n_counts_every_feed_code_and_drives_fanout(self) -> None:
        codes = tuple(f"{i:06d}" for i in range(1, 13))  # 12 종목 피드
        stocks = {c: Stock(c, None, None) for c in codes}
        a = [art(1, at(DAY, 9), codes)]
        assert day_pairs(a, DAY, PREV, ["000001"], None, stocks, sweep=False) == []
        rows = day_pairs(a, DAY, PREV, ["000001"], None, stocks, sweep=False, cfg=NO_FILTER)
        assert rows[0]["related_n"] == 12

    def test_stage1_filters_and_reports_reasons(self) -> None:
        a = [art(1, at(DAY, 9), ("000001",), title="[특징주] 가나 급등"),
             art(2, at(DAY, 9), ("000001",), title="[표] 순매수 상위"),
             art(3, at(DAY, 9), ("000001",), title="가나 수주")]
        rep = DayReport(DAY)
        rows = day_pairs(a, DAY, PREV, ["000001"], None, STOCKS, cfg=STAGE1, sweep=False, report=rep)
        assert [r["title"] for r in rows] == ["가나 수주"]
        assert rep.stats["skip:swing"] == 1 and rep.stats["skip:roundup"] == 1 and rep.stats["prefiltered"] == 2

    def test_not_in_master_is_dropped_and_counted(self) -> None:
        rep = DayReport(DAY)
        assert day_pairs([art(1, at(DAY, 9), ("123456",))], DAY, PREV, ["123456"], None, STOCKS,
                         sweep=False, report=rep) == []
        assert rep.stats["not_in_master"] == 1

    def test_sweep_drops_articles_the_live_sweep_never_reaches(self) -> None:
        # 15:19:59 게시: 그 초의 페이지가 보지만 상세 요청은 15:20 — 순회기는 그 시각부터 요청하지 않는다.
        late = art(1, at(DAY, 15, 19) + dt.timedelta(seconds=59), ("000001",))
        early = art(2, at(DAY, 7), ("000001",))
        rep = DayReport(DAY)
        rows = day_pairs([late, early], DAY, PREV, ["000001"], None, STOCKS, report=rep)
        assert [r["key"].split("|")[0] for r in rows] == [pair_key_article(early.db_id)]
        assert rep.stats["unswept"] == 1
        assert len(day_pairs([late, early], DAY, PREV, ["000001"], None, STOCKS, sweep=False)) == 2

    def test_rows_are_ordered_by_discovery_then_key(self) -> None:
        a = [art(2, at(DAY, 9), ("000001",)), art(1, at(DAY, 8, 30), ("000001", "000002"))]
        rows = day_pairs(a, DAY, PREV, ["000001", "000002"], None, STOCKS, sweep=False)
        assert [r["published_at"][11:16] for r in rows] == ["08:30", "08:30", "09:00"]
        assert rows[0]["key"] < rows[1]["key"]


def test_build_pairs_uses_previous_trading_day_universe_and_dedupes() -> None:
    values = {PREV: {"000001": 10.0, "000002": 5.0}, DAY: {"000002": 10.0, "000001": 1.0}}
    arts = [art(1, at(DAY, 9), ("000001", "000002")), art(2, at(NEXT, 9), ("000001", "000002"))]
    logs: list[str] = []
    rows = build_pairs([DAY, NEXT], [PREV, DAY, NEXT], values, arts, None, STOCKS, universe_size=1, sweep=False,
                       log=logs.append)
    assert [(r["day"], r["code"]) for r in rows] == [("2025-09-02", "000001"), ("2025-09-03", "000002")]
    assert len(logs) == 2
    assert build_pairs([PREV], [PREV], values, arts, None, STOCKS, log=logs.append) == []
    assert "no previous trading day" in logs[-1]


def test_parse_days_range_and_list_restricted_to_trading_days() -> None:
    trading = [PREV, DAY, NEXT]
    assert parse_days("2025-09-02..2025-09-10", trading) == [DAY, NEXT]
    assert parse_days("2025-09-03,2025-09-06", trading) == [NEXT]
