"""판정 쌍 생성기 — quant-airflow DB 의 토스 종목별 피드에서 (기사, 종목) 쌍을 만든다.

    # 1단계 구간 재생성(DB 읽기 전용)
    python -m swing_it.news.pairs --days 2023-03-20..2026-02-27 \\
        --out data/eval/persistence_swing/pairs/swing_2023-03_2026-02.db.jsonl

daytrade-it ``scripts/eval/build_universe_events.py --pairs-out`` (+ 스윙 태그 필터) 를 대체한다. 그쪽은
daytrade-it 파일 아카이브(``data/toss_universe_42m/<code>.jsonl`` + ``.done``)를 읽었고, 이쪽은 같은 아카이브를
옮겨 담은 DB 를 읽는다(quant-airflow ``sql/migrations/015_news_company_feed.sql``):

- ``news_articles`` — 기사 본체. 토스 기사 id = ``"toss:" + md5(build_article_url(newsId))[:12]``.
  **먼저 본 판을 남긴다**(토스가 기사를 고치면 createdAt·제목이 바뀌는데 나중 판은 lookahead).
- ``news_company_feed`` — (기사, 종목) = 그 기사가 그 종목의 토스 피드에 떴다. 옛 생성기의 "관련 종목"
  (``related_codes``) 이고, 기사별 종목 수가 팬아웃(``related_n``)이다.
- ``news_company_feed_fetches`` — 종목별 수집 원장. 원장에 없는 종목·``failed`` 는 "못 받았다"(0건이 아니라 모름).
  ``page_cap`` 은 토스 100페이지 상한에 걸린 것이라 ``oldest_at`` 이후만 덮는다.

사건일 d 의 쌍 (옛 ``day_pairs`` 의미 그대로):

1. 유니버스 = 전 거래일(삼성전자 일봉이 있는 날) ``daily_bars_adjusted.trade_value`` 상위 N(기본 300),
   동률은 코드 오름차순(:func:`swing_it.news.sweep.select_universe`). ``069500`` 은 피드가 없어 쌍이 안 생긴다.
2. 기사 = createdAt ∈ [전 거래일 15:20, d 15:20) KST.
3. 쌍 = 기사의 피드 종목 ∩ 유니버스 ∩ **그 창을 덮는 수집이 있는 종목**(:func:`covered`).
4. (기본 켬) 운영 순회기가 d 15:20 전에 발견하지 못하는 기사는 뺀다(:func:`~swing_it.news.sweep.simulate_sweep`) —
   1단계 쌍 파일이 거친 장중 제약이라 재현용으로 둔다. ``--no-sweep`` 으로 끈다.
5. 판정 전 필터(:class:`~swing_it.news.prefilter.PairFilter`, 기본 = 1단계 조합).

출력 JSONL 키는 옛 파일(``data/eval/persistence_swing/pairs/swing_2023-03_2026-02.jsonl``)과 같다: key, code, name, market, sector, title,
body(요약, 없으면 제목), summary, category(피드에 분류가 없어 None), day, month, session, published_at, related_n.

DB 는 읽기만 한다. 가격은 :func:`swing_it.storage.read_prices` 정문으로 읽는다(가드레일 g).
"""

from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from krx_quant_core.market.session import KST

from swing_it.news.prefilter import STAGE1, PairFilter, add_filter_args, filter_from_args, skip_reason
from swing_it.news.sweep import (
    CUTOFF,
    MARKET_ETF,
    SWEEP_START,
    UNIVERSE_SIZE,
    news_id_from_url,
    pair_key_article,
    select_universe,
    simulate_sweep,
)

OPEN = dt.time(9, 0)
#: 수집 원장에서 "그 창을 받았다" 로 치는 상태. page_cap 은 oldest_at 이후만.
TRUNCATED = frozenset({"page_cap", "max_pages"})
#: 삼성전자 — 옛 생성기의 거래일 달력(이 종목 일봉이 있는 날).
CALENDAR_CODE = "005930"


@dataclass(frozen=True)
class Article:
    """기사 한 건 + 그 기사가 뜬 피드 종목들."""

    db_id: str
    news_id: str  # 토스 newsId(URL 에서 복원) — 순회 재현의 동률 정렬 키. 못 읽으면 db_id.
    title: str
    summary: str
    published_at: dt.datetime
    codes: tuple[str, ...]
    category: str | None = None


@dataclass(frozen=True)
class Stock:
    name: str
    market: str | None
    sector: str | None


@dataclass
class Fetch:
    """수집 원장 한 줄을 '이 시각부터 이 시각까지 빠짐없이 받았다' 로 읽은 것."""

    start: dt.datetime
    finished_at: dt.datetime
    status: str


@dataclass
class DayReport:
    day: dt.date
    universe: int = 0
    missing: int = 0  # 원장에 한 줄도 없는 유니버스 종목
    truncated: int = 0
    uncovered: int = 0
    stats: Counter = field(default_factory=Counter)

    def line(self) -> str:
        s = self.stats
        return (
            f"{self.day}: universe {self.universe} truncated {self.truncated} missing {self.missing} "
            f"uncovered {self.uncovered} pairs {s['pairs']} excluded_pairs {s['excluded_pairs']} "
            f"not_in_master {s['not_in_master']} unswept {s['unswept']} empty_summary {s['empty_summary']} "
            f"prefiltered {s['prefiltered']} "
            + " ".join(f"{k} {v}" for k, v in sorted(s.items()) if k.startswith("skip:"))
        ).rstrip()


# --- 순수 함수 ---------------------------------------------------------------------------


def fetch_from_row(since: dt.date | str, status: str, oldest_at: dt.datetime | None,
                   finished_at: dt.datetime) -> Fetch | None:
    """원장 행 → 덮는 구간. ``failed`` 는 None(아무것도 덮지 않는다).

    시작 = since 00:00 KST, page_cap 이면 max(그것, oldest_at) — 옛 ``_coverage_entry`` 와 같다.
    """
    if status not in ("done", *TRUNCATED):
        return None
    since = since if isinstance(since, dt.date) else dt.date.fromisoformat(str(since)[:10])
    start = dt.datetime.combine(since, dt.time(0), tzinfo=KST)
    if status in TRUNCATED and oldest_at is not None:
        start = max(start, oldest_at)
    return Fetch(start=start, finished_at=finished_at, status=status)


def coverage_from_rows(rows: Iterable[tuple]) -> dict[str, list[Fetch]]:
    """(code, since, status, oldest_at, finished_at) 원장 행들 → 종목별 덮는 구간."""
    out: dict[str, list[Fetch]] = {}
    for code, since, status, oldest_at, finished_at in rows:
        f = fetch_from_row(since, status, oldest_at, finished_at)
        if f is not None:
            out.setdefault(code, []).append(f)
        else:
            out.setdefault(code, [])
    return out


def covered(code: str, coverage: dict[str, list[Fetch]] | None, lo: dt.datetime, hi: dt.datetime) -> bool:
    """``code`` 의 피드를 [lo, hi) 창 전체에 대해 받았나.

    받은 수집 중 하나라도 시작 ≤ lo 이고 **끝난 시각 ≥ hi** 여야 한다. 옛 생성기에는 뒤쪽 조건이 없었다 —
    파일이 매일 이어 붙는다고 가정했지만, 유니버스에서 빠진 종목은 크론이 더 받지 않아 그 뒤 창이 "0건" 으로
    보였다. 1단계 창(2025-09~2026-02)은 아카이브 백필(전 종목 2026-09-17 완료)이 전부 덮어서 둘이 같다.
    ``coverage=None`` 이면 전부 덮였다고 본다(테스트·파일 입력용).
    """
    if coverage is None:
        return True
    return any(f.start <= lo and f.finished_at >= hi for f in coverage.get(code, ()))


def bucket_articles(articles: Iterable[Article], trading_days: list[dt.date]) -> dict[dt.date, list[Article]]:
    """기사마다 [전 거래일 15:20, d 15:20) 창을 담는 거래일 d 하나에 배정한다."""
    days = sorted(trading_days)
    cutoffs = [dt.datetime.combine(d, CUTOFF, tzinfo=KST) for d in days]
    out: dict[dt.date, list[Article]] = {}
    for a in articles:
        i = bisect.bisect_right(cutoffs, a.published_at)
        if i < len(days):
            out.setdefault(days[i], []).append(a)
    return out


def day_pairs(
    articles: Iterable[Article],
    day: dt.date,
    prev_day: dt.date,
    universe: list[str],
    coverage: dict[str, list[Fetch]] | None,
    stocks: dict[str, Stock],
    *,
    cfg: PairFilter = STAGE1,
    sweep: bool = True,
    report: DayReport | None = None,
) -> list[dict]:
    """사건일 ``day`` 의 쌍 행(출력 JSONL 형식). 옛 ``day_pairs`` + ``pair_rows`` + 스윙 태그 필터."""
    lo = dt.datetime.combine(prev_day, CUTOFF, tzinfo=KST)
    hi = dt.datetime.combine(day, CUTOFF, tzinfo=KST)
    open_ = dt.datetime.combine(day, OPEN, tzinfo=KST)
    members = set(universe)
    stats: Counter = report.stats if report is not None else Counter()
    if report is not None:
        real = [c for c in universe if c != MARKET_ETF]
        report.universe = len(universe)
        report.missing = sum(1 for c in real if coverage is not None and not coverage.get(c))
        report.truncated = sum(
            1 for c in real if coverage is not None and any(f.status in TRUNCATED for f in coverage.get(c, ()))
        )
        report.uncovered = sum(1 for c in real if not covered(c, coverage, lo, hi))
    candidates: list[tuple[Article, list[str]]] = []
    arrivals: dict[str, list[tuple[str, dt.datetime]]] = {}
    for a in articles:
        if not lo <= a.published_at < hi:
            continue
        in_universe = [c for c in a.codes if c in members]
        codes = [c for c in in_universe if covered(c, coverage, lo, hi)]
        stats["excluded_pairs"] += len(in_universe) - len(codes)
        if not codes:
            continue
        candidates.append((a, codes))
        for code in codes:
            arrivals.setdefault(code, []).append((a.news_id, a.published_at))
    found = (
        simulate_sweep(universe, day, arrivals, stale_before=lo, start=SWEEP_START, end=CUTOFF) if sweep else None
    )
    out: list[tuple[tuple, dict]] = []
    for a, codes in candidates:
        discovered = found.get(a.news_id) if found is not None else a.published_at
        if discovered is None:
            stats["unswept"] += len(codes)
            continue
        aid = pair_key_article(a.db_id)
        related_n = len(a.codes)
        for code in codes:
            s = stocks.get(code)
            if s is None:
                stats["not_in_master"] += 1
                continue
            reason = skip_reason(a.title, related_n, cfg)
            if reason is not None:
                stats["prefiltered"] += 1
                stats["skip:" + reason.split(":", 1)[0]] += 1
                continue
            key = f"{aid}|{code}"
            row = {
                "key": key,
                "code": code,
                "name": s.name,
                "market": s.market,
                "sector": s.sector,
                "title": a.title,
                "body": a.summary or a.title,
                "summary": a.summary,
                "category": a.category,
                "day": str(day),
                "month": str(day)[:7],
                "session": "pre_market" if a.published_at < open_ else "intraday",
                "published_at": a.published_at.astimezone(KST).isoformat(),
                "related_n": related_n,
            }
            stats["empty_summary"] += not a.summary
            out.append(((discovered, key), row))
    out.sort(key=lambda t: t[0])
    stats["pairs"] += len(out)
    return [row for _, row in out]


def parse_days(spec: str, trading_days: list[dt.date]) -> list[dt.date]:
    """``A..B`` (양끝 포함) 또는 쉼표 목록, 거래일만."""
    valid = set(trading_days)
    if ".." in spec:
        a, b = (dt.date.fromisoformat(x) for x in spec.split("..", 1))
        return sorted(d for d in valid if a <= d <= b)
    return sorted(d for x in spec.split(",") if x and (d := dt.date.fromisoformat(x)) in valid)


def build_pairs(
    days: list[dt.date],
    trading_days: list[dt.date],
    values_by_day: dict[dt.date, dict[str, float]],
    articles: Iterable[Article],
    coverage: dict[str, list[Fetch]] | None,
    stocks: dict[str, Stock],
    *,
    universe_size: int = UNIVERSE_SIZE,
    cfg: PairFilter = STAGE1,
    sweep: bool = True,
    log=print,
) -> list[dict]:
    """여러 날의 쌍 — 키는 처음 나온 날 한 번만(옛 ``--pairs-out`` 과 같다)."""
    trading = sorted(trading_days)
    buckets = bucket_articles(articles, trading)
    seen: set[str] = set()
    rows: list[dict] = []
    for day in days:
        i = bisect.bisect_left(trading, day)
        if i == 0:
            log(f"{day}: no previous trading day -- skipped")
            continue
        prev = trading[i - 1]
        universe = select_universe(values_by_day.get(prev, {}), universe_size)
        rep = DayReport(day)
        got = day_pairs(buckets.get(day, []), day, prev, universe, coverage, stocks, cfg=cfg, sweep=sweep, report=rep)
        log(rep.line())
        for r in got:
            if r["key"] not in seen:
                seen.add(r["key"])
                rows.append(r)
    return rows


# --- DB ----------------------------------------------------------------------------------


def _aware(t: dt.datetime | None) -> dt.datetime | None:
    if t is None:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def load_prices_by_day(con, lo: dt.date, hi: dt.date) -> tuple[list[dt.date], dict[dt.date, dict[str, float]]]:
    """(거래일 달력, {날짜: {code: 거래대금}}) — read_prices 정문(폐지 포함 검사)으로 읽는다."""
    import pandas as pd  # noqa: PLC0415

    from swing_it.storage import read_prices  # noqa: PLC0415

    pr = read_prices(con, cols=("code", "date", "trade_value"))
    pr["date"] = pd.to_datetime(pr["date"]).dt.date
    trading = sorted(set(pr.loc[pr["code"] == CALENDAR_CODE, "date"]))
    win = pr[(pr["date"] >= lo) & (pr["date"] <= hi) & pr["trade_value"].notna()]
    values: dict[dt.date, dict[str, float]] = {}
    for d, g in win.groupby("date"):
        values[d] = dict(zip(g["code"], g["trade_value"].astype(float), strict=True))
    return trading, values


def load_stocks(con) -> dict[str, Stock]:
    """종목 마스터 — 현재 상장(``stocks``) 우선, 없으면 폐지 마스터(업종 없음)."""
    cur = con.cursor()
    out: dict[str, Stock] = {}
    cur.execute("SELECT code, name, market FROM delisted_stocks")
    for code, name, market in cur.fetchall():
        out[code] = Stock(name=name or code, market=market, sector=None)
    cur.execute("SELECT code, name, market, sector FROM stocks")
    for code, name, market, sector in cur.fetchall():
        out[code] = Stock(name=name or code, market=market, sector=sector)
    return out


def load_coverage(con) -> dict[str, list[Fetch]]:
    cur = con.cursor()
    cur.execute("SELECT code, since, status, oldest_at, finished_at FROM news_company_feed_fetches")
    return coverage_from_rows(
        (c, s, st, _aware(o), _aware(f)) for c, s, st, o, f in cur.fetchall()
    )


def load_articles(con, lo: dt.datetime, hi: dt.datetime, slack: dt.timedelta = dt.timedelta(days=7)) -> list[Article]:
    """창 [lo, hi) 의 피드 기사 + 각 기사의 피드 종목 전부(팬아웃은 창과 무관하게 전체 피드로 센다).

    같은 id 가 news_articles 에 두 판 있으면(PK 가 (id, published_at)) **먼저 수집된 판**을 쓴다.
    """
    cur = con.cursor()
    cur.execute(
        "SELECT a.id, a.url, a.title, a.summary, a.published_at, a.collected_at "
        "FROM news_articles a WHERE a.published_at >= %s AND a.published_at < %s "
        "AND EXISTS (SELECT 1 FROM news_company_feed f WHERE f.article_id = a.id)",
        (lo - slack, hi + slack),
    )
    best: dict[str, tuple] = {}
    for aid, url, title, summary, pub, coll in cur.fetchall():
        pub, coll = _aware(pub), _aware(coll)
        rank = (coll or pub, pub)
        if aid not in best or rank < best[aid][0]:
            best[aid] = (rank, url, title, summary, pub)
    ids = [a for a, v in best.items() if lo <= v[4] < hi]
    codes: dict[str, set[str]] = {}
    for i in range(0, len(ids), 5000):
        part = ids[i : i + 5000]
        cur.execute("SELECT article_id, code FROM news_company_feed WHERE article_id = ANY(%s)", (part,))
        for aid, code in cur.fetchall():
            codes.setdefault(aid, set()).add(code)
    out = []
    for aid in ids:
        _, url, title, summary, pub = best[aid]
        out.append(
            Article(
                db_id=aid,
                news_id=news_id_from_url(url) or aid,
                title=title or "",
                summary=summary or "",
                published_at=pub.astimezone(KST),
                codes=tuple(sorted(codes.get(aid, ()))),
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", required=True, help="A..B (양끝 포함) 또는 쉼표 목록, 거래일만")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--universe-size", type=int, default=UNIVERSE_SIZE)
    ap.add_argument("--no-sweep", action="store_true", help="장중 순회 도달 재현을 끈다(1단계와 달라진다)")
    ap.add_argument("--strict", action="store_true", help="원장에 없는 유니버스 종목이 있는 날이면 멈춘다")
    ap.add_argument("--db", default=None, help="DSN (기본 KR_QUANT_DB)")
    add_filter_args(ap)
    args = ap.parse_args(argv)
    cfg = filter_from_args(args)

    from swing_it.storage import connect, db_default, load_env_db  # noqa: PLC0415

    load_env_db()
    con = connect(args.db or db_default())
    spec = args.days
    lo_day = dt.date.fromisoformat(spec.split("..")[0].split(",")[0])
    hi_day = dt.date.fromisoformat(spec.split("..")[-1].split(",")[-1])
    trading, values = load_prices_by_day(con, lo_day - dt.timedelta(days=30), hi_day)
    days = parse_days(spec, trading)
    if not days:
        ap.error(f"--days {spec}: 거래일이 없다")
    prev_first = trading[bisect.bisect_left(trading, days[0]) - 1]
    lo = dt.datetime.combine(prev_first, CUTOFF, tzinfo=KST)
    hi = dt.datetime.combine(days[-1], CUTOFF, tzinfo=KST)
    coverage = load_coverage(con)
    if not coverage:
        sys.exit("REFUSED: news_company_feed_fetches 가 비었다 — 피드 이전(import)이 끝나지 않았다")
    stocks = load_stocks(con)
    articles = load_articles(con, lo, hi)
    print(f"{len(articles):,} feed articles in [{lo:%Y-%m-%d %H:%M}, {hi:%Y-%m-%d %H:%M}); "
          f"{len(coverage):,} codes in ledger; {len(days)} trading days; filter {cfg.describe()} sweep {not args.no_sweep}")

    def log(line: str) -> None:
        print(line, flush=True)
        if args.strict and " missing 0 " not in line and "missing" in line:
            sys.exit(f"REFUSED (--strict): {line}")

    rows = build_pairs(days, trading, values, articles, coverage, stocks,
                       universe_size=args.universe_size, cfg=cfg, sweep=not args.no_sweep, log=log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    print(f"wrote {len(rows):,} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
