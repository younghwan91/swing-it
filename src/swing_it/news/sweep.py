"""토스 종목별 피드의 쌍 키·유니버스·순회 도달 재현 — daytrade-it ``application/news/universe.py`` 의 사본.

**출처:** daytrade-it 브랜치 ``research/news-persistence-judge`` @ ba0662a. 판정 캐시의 쌍 키
(``live_article_id|code``)와 스윙 2호 1단계 쌍 집합을 바이트 단위로 재현하는 데 필요한 것만 옮겼다.

- :func:`live_article_id` — 판정 캐시 키의 앞부분. ``uuid5(NS, "toss:" + md5(build_article_url(newsId))[:12])``
  이고, ``"toss:" + md5(...)[:12]`` 는 quant-airflow ``news_articles.id`` 그 자체다 → DB id 에서 바로 만든다
  (:func:`pair_key_article`).
- :func:`select_universe` — 전 거래일 거래대금 내림차순, 동률은 코드 오름차순, 6자리 숫자 코드·양수만.
- :func:`simulate_sweep` — **장중 순회기 도달 재현.** daytrade-it 운영 순회기(08:00~15:20, 1초 1요청, 유니버스
  순서대로 페이지 → 새 기사마다 상세 1회)가 그날 15:20 전까지 *발견하지 못하는* 기사는 옛 쌍 생성기
  (``build_universe_events.day_pairs``)가 쌍에서 뺐다. 스윙 연구엔 본래 필요 없는 장중 제약이지만 1단계 쌍
  파일이 이걸 거쳤으므로 **재현을 위해** 옮겼다 — :mod:`swing_it.news.pairs` 에서 ``sweep=False`` 로 끌 수 있다.

순수 함수만 — 네트워크·DB·I/O 없음.
"""

from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import urllib.parse
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid5

from krx_quant_core.market.session import KST

#: daytrade-it ``toss_news_client.TOSS_NEWS_NAMESPACE`` — 운영 데몬의 기사 키 네임스페이스.
TOSS_NEWS_NAMESPACE = UUID("b6a0f6a0-8e3b-4b1a-9b8a-6a2f2f2b9a1a")

#: 스윙 2호 유니버스 크기(전 거래일 거래대금 상위 N).
UNIVERSE_SIZE = 300
#: 유니버스 계산에만 쓰이는 지수 ETF — 피드를 받지 않는다(quant-airflow 수집기·daytrade-it 크론 규칙).
MARKET_ETF = "069500"

SWEEP_INTERVAL = dt.timedelta(seconds=1)
SWEEP_START = dt.time(8, 0)
#: 쌍 창의 경계이자 순회기가 요청을 멈추는 시각.
CUTOFF = dt.time(15, 20)
#: 운영 순회기의 페이지 크기(``size``).
PAGE_SIZE = 20


def build_article_url(news_id: str) -> str:
    """krx-news-client ``scrapers.toss.build_article_url`` 과 같은 URL(DB ``news_articles.url``)."""
    content_params = json.dumps({"id": news_id}, ensure_ascii=False, separators=(",", ":"))
    query = urllib.parse.urlencode({"contentType": "news", "contentParams": content_params})
    return f"https://tossinvest.com/news?{query}"


def news_id_from_url(url: str | None) -> str | None:
    """:func:`build_article_url` 의 역 — 토스 newsId, 못 읽으면 None."""
    if not url:
        return None
    try:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        return str(json.loads(params["contentParams"][0])["id"])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def db_article_id(news_id: str) -> str:
    """quant-airflow ``news_articles.id`` (= krx-news-client ``make_article_id('toss', url)``)."""
    digest = hashlib.md5(build_article_url(news_id).encode()).hexdigest()[:12]  # noqa: S324 — id 유도, 보안 아님
    return f"toss:{digest}"


def pair_key_article(db_id: str) -> str:
    """DB 기사 id → 판정 캐시 키의 기사 부분(uuid5)."""
    return str(uuid5(TOSS_NEWS_NAMESPACE, db_id))


def live_article_id(news_id: str) -> str:
    """토스 newsId → 운영 데몬의 기사 키(uuid5). ``pair_key_article(db_article_id(news_id))`` 와 같다."""
    return pair_key_article(db_article_id(news_id))


def select_universe(daily_values: dict[str, float], size: int = UNIVERSE_SIZE) -> list[str]:
    """그날 유니버스 — 전 거래일 거래대금 내림차순, 동률은 코드 오름차순, 상위 ``size``.

    6자리 숫자 코드·양수 거래대금만. ``069500`` 은 **여기서 빼지 않는다**: 운영 순회기는 이 목록 순서대로
    페이지를 요청하므로(:func:`simulate_sweep`) 순서 재현에는 그대로 둬야 한다. 피드가 없으니 쌍은 안 생긴다.
    """
    candidates = [c for c, v in daily_values.items() if v > 0 and len(c) == 6 and c.isdigit()]
    candidates.sort(key=lambda c: (-daily_values[c], c))
    return candidates[:size]


@dataclass(frozen=True)
class SweepRequest:
    """One Toss request of the sweep: a company news page, or an article detail."""

    kind: str  # "page" | "detail"
    code: str
    page: int = 1
    news_id: str | None = None


class SweepCursor:
    """The sweep's request order (live runner and :func:`simulate_sweep` share it).

    Rules (each request costs one interval, enforced by the caller's clock):
      * codes are visited in universe order, repeating;
      * after a page, every item not seen earlier today (via any code), not
        stale (published before `stale_before`, or undated) gets one detail
        request, in page order, before anything else;
      * if the page was full (`page_size` items) and every item on it was new,
        the next page of the same code follows those details;
      * otherwise the next code's page 1.
    `page_result` must be called for each page request before `next_request`.
    """

    def __init__(self, universe: list[str], stale_before: dt.datetime, page_size: int = PAGE_SIZE) -> None:
        if not universe:
            raise ValueError("empty universe")
        self.universe = list(universe)
        self.stale_before = stale_before
        self.page_size = page_size
        self.seen: set[str] = set()
        self._pending: deque[SweepRequest] = deque()
        self._index = 0
        self._awaiting: SweepRequest | None = None

    def next_request(self) -> SweepRequest:
        if self._awaiting is not None:
            raise RuntimeError(f"page_result not recorded for {self._awaiting}")
        if not self._pending:
            code = self.universe[self._index % len(self.universe)]
            self._index += 1
            self._pending.append(SweepRequest("page", code, 1))
        request = self._pending.popleft()
        if request.kind == "page":
            self._awaiting = request
        return request

    def page_result(
        self, request: SweepRequest, items: Sequence[tuple[str | None, dt.datetime | None]]
    ) -> list[str]:
        """Record a page's (news id, createdAt) items (empty on an HTTP error); new ids."""
        if request != self._awaiting:
            raise RuntimeError(f"unexpected page result for {request}")
        self._awaiting = None
        new: list[str] = []
        for news_id, created in items:
            if not news_id or news_id in self.seen:
                continue
            if created is None or created < self.stale_before:
                continue
            self.seen.add(news_id)
            new.append(news_id)
        self._pending.extend(SweepRequest("detail", request.code, request.page, n) for n in new)
        if len(items) >= self.page_size and len(new) == len(items):
            self._pending.append(SweepRequest("page", request.code, request.page + 1))
        return new


def page_items(
    listing: Sequence[tuple[dt.datetime, str]], at: dt.datetime, page: int, page_size: int = PAGE_SIZE
) -> list[tuple[str, dt.datetime]]:
    """Page `page` (1-based) of a code's newest-first list as of `at`.

    `listing` is the code's (createdAt, news id) sorted ascending; an article is
    listed once createdAt <= `at`.
    """
    visible = bisect.bisect_right(listing, (at, "\U0010ffff"))
    hi = visible - (page - 1) * page_size
    if hi <= 0:
        return []
    lo = max(0, hi - page_size)
    return [(news_id, created) for created, news_id in reversed(listing[lo:hi])]


def simulate_sweep(
    universe: list[str],
    day: dt.date,
    arrivals: dict[str, list[tuple[str, dt.datetime]]],
    stale_before: dt.datetime,
    start: dt.time = SWEEP_START,
    end: dt.time = CUTOFF,
    interval: dt.timedelta = SWEEP_INTERVAL,
    page_size: int = PAGE_SIZE,
    trace: list[SweepRequest] | None = None,
) -> dict[str, dt.datetime]:
    """News id -> when the live sweep would have discovered it on `day`.

    Deterministic replay of `SweepCursor` on a synthetic clock: request k starts
    at `start + k * interval` (only if that is before `end`) and completes one
    interval later; a page lists the code's `arrivals` published by the
    request's start (`page_items`); an article's discovery time is when its
    detail request completes. Articles never reached before `end` are absent.
    """
    found: dict[str, dt.datetime] = {}
    if not universe:
        return found
    listings = {code: sorted((c, n) for n, c in rows) for code, rows in arrivals.items()}
    cursor = SweepCursor(universe, stale_before, page_size)
    now = dt.datetime.combine(day, start, tzinfo=KST)
    stop = dt.datetime.combine(day, end, tzinfo=KST)
    while now < stop:
        request = cursor.next_request()
        if trace is not None:
            trace.append(request)
        if request.kind == "page":
            items = page_items(listings.get(request.code, ()), now, request.page, page_size)
            cursor.page_result(request, items)
        else:
            assert request.news_id is not None
            found[request.news_id] = now + interval
        now += interval
    return found
