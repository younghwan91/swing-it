"""스윙 3호(SEPA 합류) 뉴스 다리 N — 판정 원장을 (종목, 신호일) 일별 건수와 60일 N 플래그로 만든다.

설계: ``docs/superpowers/specs/2026-10-09-sepa-confluence.md`` §3 "N 뉴스". 이 모듈은 **피처와 건수만** 만든다 —
가격·수익은 읽지 않는다(사전등록 전 단계 1). 스윙 2호(``research/logs/news_persistence_swing``)의 평가와 무관하다:
그 연구의 쌍 파일·사건일(15:20 창)·사건 정의를 쓰지 않고, 원장(``article_judgments``)의 판정 출력만 읽는다.

결정(전부 여기 고정 — 바꾸면 새 정의다):

1. **judge 우선순위** :data:`JUDGE_PRIORITY` = ``v2r3-haiku55`` > ``v2-haiku45``. (기사, 종목)마다 우선순위대로
   **성공 행**이 있는 첫 judge 를 쓴다(행마다 ``judge`` 로 노출). 어느 judge 에도 성공 행이 없고 오류 행만 있으면
   ``error`` — 모름(unknown)으로 센다(0 도 1 도 아니다). ``v2-haiku55``(2026-04 파일럿 500쌍)는 쓰지 않는다 —
   같은 500쌍에 ``v2-haiku45`` 가 있고, 한 시기만 다른 모델이 섞이면 judge 단절이 하나 더 생긴다.
   두 judge 는 같은 (기사, 종목)을 공유하지 않는다(2026-10-09 원장: 겹침 0) — 우선순위는 사실상 기간 구분이다.

2. **기사 가용 시각 = ``news_articles.published_at``** (토스 createdAt, KST 로 읽는다). 설계서는 ``first_seen_at`` 을
   적었지만 ``news_company_feed.first_seen_at`` 은 아카이브 이전(``toss_archive_import``) 행에서 **이전 작업 시각**
   (2026-09-16 이후)이라 과거 기사의 가용 시각이 아니다. ``collected_at`` 도 같은 이유(2026-09-06 이후)로 못 쓴다.
   토스가 기사를 고치면 createdAt 이 늦어질 수 있다 — 아카이브는 백필 때 본 판이라 그 경우 시각은 **늦게**(보수적으로)
   잡힌다.

3. **시점 매핑(PIT)** — 신호일 d 는 d 종가에 계산되고 진입은 다음 거래일 시가다.

   - ``mode="close"`` (**기본**, 설계서 문구 그대로): 가용 시각 t 가 거래일 d 의 15:30 KST **전**이면 d, 15:30 이후·
     주말·휴장일이면 그 다음 거래일. 즉 d = t < d 15:30 을 만족하는 첫 거래일.
   - ``mode="overnight"`` (민감도): d 장 마감 후 ~ 다음 거래일 08:00 KST 까지의 기사도 신호일 d 에 넣는다 — 진입
     주문(08:30 장전 동시호가) 전에 읽을 수 있으므로 시점상 합법이다. d = (t ≤ e 08:00 인 첫 거래일 e)의 전 거래일.
   - 기본을 close 로 둔 이유: (a) 사전등록 설계서 문구다. (b) 60일 창에서 경계 기사 몇 건의 이득은 작고,
     createdAt 이 실제 노출보다 이를 수 있는 위험(정정·재송고)은 마감 직후~새벽 구간에서 가장 크다. (c) close 는
     overnight 보다 모든 기사를 같거나 늦은 신호일에 배정한다 — 미래 정보가 들어올 수 없는 쪽이다.

4. **창**: 신호일 d 의 N 은 매핑된 신호일 s ∈ (d − 60일, d] (역일 60일, d 포함) 의 기사로 정한다.

5. **N 기사**(:func:`is_n_article`): subject ∈ {main, partial} ∧ sentiment_direction = +1 ∧
   persistence ∈ {multi_quarter, structural} ∧ freshness = new ∧ reports_price_move = false.
   ``n_pos`` = subject ∈ {main, partial} ∧ direction +1, ``n_neg`` = subject ∈ {main, partial} ∧ direction −1.

6. **판정 전 필터를 양쪽에 같게**: 두 judge 의 쌍 집합은 필터가 달랐다(v2r3 = 운영 3규칙 + 스윙 태그,
   v2-haiku45 = 필터 없음/운영 3규칙). 기간마다 다른 기사 모집단이 섞이지 않게 피드 쌍과 판정 쌍 **모두**에
   :data:`~swing_it.news.prefilter.STAGE1` 을 걸고 남은 쌍만 센다(팬아웃 = 그 기사의 피드 종목 수 전체).

7. **커버리지(news_known)**: 분모 = 그 창의 필터 통과 피드 쌍(``n_feed``), 분자 = 성공 판정(``n_articles``).
   오류 행·판정 안 된 쌍(유니버스 밖, 표본 추출에서 빠짐, 순회 미도달, 두 기간 사이 공백)은 분모에만 있다.
   ``news_known`` = N 이 참(양성은 커버리지와 무관하게 안다) ∨ 창에 피드 기사가 0건(없는 뉴스는 판정할 것도 없다)
   ∨ ``cov_60 ≥ min_coverage``. **N 이 거짓이고 news_known 도 거짓인 행은 0 이 아니라 모름(NaN)이다.**

DB 는 읽기만 한다. DSN 은 :func:`swing_it.storage.load_env_db` / ``KR_QUANT_DB`` 로만 읽고 찍지 않는다.

    # 커버리지 건수(수익 없음) — research/logs/sepa_confluence/coverage_news.md 의 원자료
    uv run --extra pg python -m swing_it.news.daily_leg --coverage
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from krx_quant_core.market.session import KST

from swing_it.news.prefilter import STAGE1, PairFilter, skip_reason

#: (기사, 종목)마다 성공 행이 있는 첫 judge 를 쓴다.
JUDGE_PRIORITY: tuple[str, ...] = ("v2r3-haiku55", "v2-haiku45")
#: 정규장 마감 — close 모드에서 이 시각 이후 기사는 다음 거래일 신호.
CLOSE_CUTOFF = dt.time(15, 30)
#: overnight 모드 — 진입일 이 시각까지의 기사는 전 거래일 신호에 쓴다(장전 동시호가 08:30 전).
ENTRY_CUTOFF = dt.time(8, 0)
MODES = ("close", "overnight")
#: N 창(역일).
WINDOW_DAYS = 60
#: news_known 의 창 커버리지 하한 기본값(근거: coverage_news.md — 1단계 기술 통계에서만 쓰고 판정에 걸지 않는다).
MIN_COVERAGE = 0.5
#: 거래일 달력 종목(삼성전자 일봉이 있는 날) — :mod:`swing_it.news.pairs` 와 같다.
CALENDAR_CODE = "005930"

SUBJECTS = frozenset({"main", "partial"})
PERSISTENT = frozenset({"multi_quarter", "structural"})

DAILY_COLS = ("code", "date", "n_pos_persistent", "n_pos", "n_neg", "n_articles", "n_error", "n_feed", "judge")
PANEL_COLS = ("code", "date", "n_flag", "news_known", "n_pos_persistent_60", "n_articles_60", "n_error_60",
              "n_feed_60", "cov_60", "judges_60")


# --- 판정 출력 ----------------------------------------------------------------------------


def _subject_dir(out: dict | None) -> tuple[bool, int | None]:
    if not out:
        return False, None
    d = out.get("sentiment_direction")
    try:
        d = int(d) if d is not None else None
    except (TypeError, ValueError):
        d = None
    return out.get("subject") in SUBJECTS, d


def _false(v) -> bool:
    """reports_price_move 가 명시적으로 거짓인가(None·누락은 거짓이 아니다 — N 에 넣지 않는다)."""
    return v is False or (isinstance(v, str) and v.strip().lower() == "false")


def is_n_article(out: dict | None) -> bool:
    """N 기사 정의(모듈 docstring 5) — 출력이 없으면 False."""
    about, d = _subject_dir(out)
    return (
        about
        and d == 1
        and out.get("persistence") in PERSISTENT
        and out.get("freshness") == "new"
        and _false(out.get("reports_price_move"))
    )


def is_pos(out: dict | None) -> bool:
    about, d = _subject_dir(out)
    return about and d == 1


def is_neg(out: dict | None) -> bool:
    about, d = _subject_dir(out)
    return about and d == -1


# --- judge 우선순위 -----------------------------------------------------------------------


def pick_judgments(rows: pd.DataFrame, priority: Sequence[str] = JUDGE_PRIORITY) -> pd.DataFrame:
    """(기사, 종목)당 한 줄로 — 우선순위대로 성공 행이 있는 첫 judge, 없으면 오류(모름).

    ``rows``: article_id · code · judge · output(dict 또는 None) · error(str 또는 None).
    우선순위 밖 judge 행은 버린다. 반환: article_id · code · judge · status(``ok``/``error``) · output.
    ``status == "error"`` 의 judge 는 오류 행이 있는 가장 앞 judge 다.
    """
    rank = {j: i for i, j in enumerate(priority)}
    df = rows[rows["judge"].isin(rank)].copy()
    if df.empty:
        return pd.DataFrame(columns=["article_id", "code", "judge", "status", "output"])
    df["ok"] = df["output"].map(lambda o: o is not None and not (isinstance(o, float) and np.isnan(o)))
    df["rank"] = df["judge"].map(rank)
    # 성공이 오류를 이기고(ok 내림차순), 같은 종류면 우선순위.
    df = df.sort_values(["article_id", "code", "ok", "rank"], ascending=[True, True, False, True], kind="mergesort")
    df = df.drop_duplicates(["article_id", "code"], keep="first")
    df["status"] = np.where(df["ok"], "ok", "error")
    df.loc[~df["ok"], "output"] = None
    return df[["article_id", "code", "judge", "status", "output"]].reset_index(drop=True)


# --- 시점 매핑 ----------------------------------------------------------------------------


def _ns(stamps: Iterable[dt.datetime]) -> np.ndarray:
    return pd.DatetimeIndex(list(stamps)).tz_convert("UTC").as_unit("ns").asi8


def _to_utc_ns(times) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(times)))
    if idx.tz is None:
        raise ValueError("가용 시각은 tz-aware 여야 한다(naive 는 KST/UTC 를 추측해야 한다)")
    return idx.tz_convert("UTC").as_unit("ns").asi8


def map_signal_days(times, trading_days: Sequence[dt.date], mode: str = "close") -> pd.Series:
    """가용 시각(tz-aware) → 그 기사를 쓸 수 있는 첫 신호일(``datetime.date``). 달력 밖이면 None.

    close: t < d 15:30 KST 인 첫 거래일 d. overnight: t ≤ e 08:00 KST 인 첫 거래일 e 의 전 거래일
    (e 가 달력 첫날이면 그 전 거래일을 모르므로 None).
    """
    if mode not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나: {mode!r}")
    days = sorted(trading_days)
    t = _to_utc_ns(times)
    if not days or len(t) == 0:
        return pd.Series([None] * len(t), dtype=object)
    if mode == "close":
        cut = _ns(dt.datetime.combine(d, CLOSE_CUTOFF, tzinfo=KST) for d in days)
        idx = np.searchsorted(cut, t, side="right")  # 첫 cut > t
    else:
        cut = _ns(dt.datetime.combine(d, ENTRY_CUTOFF, tzinfo=KST) for d in days)
        e = np.searchsorted(cut, t, side="left")  # 첫 cut ≥ t = 진입일
        # 진입일이 달력 뒤면 다음 거래일을 몰라 신호일도 모른다(마지막 날로 몰지 않는다).
        idx = np.where(e >= len(days), len(days), e - 1)
    table = np.empty(len(days) + 1, dtype=object)
    table[: len(days)] = days
    table[len(days)] = None  # 달력 뒤
    idx = np.where(idx < 0, len(days), idx)  # 달력 앞(overnight 첫날 진입)
    return pd.Series(table[idx], dtype=object)


def signal_day(t: dt.datetime, trading_days: Sequence[dt.date], mode: str = "close") -> dt.date | None:
    """:func:`map_signal_days` 의 한 건 판."""
    return map_signal_days([t], trading_days, mode).iloc[0]


# --- 일별 건수 ----------------------------------------------------------------------------


def daily_counts(
    feed: pd.DataFrame,
    picked: pd.DataFrame,
    trading_days: Sequence[dt.date],
    *,
    mode: str = "close",
    cfg: PairFilter | None = STAGE1,
) -> pd.DataFrame:
    """(종목, 신호일)별 건수 — 긴 프레임 :data:`DAILY_COLS`.

    ``feed``: 피드 쌍 article_id · code · available_at(tz-aware) · title · related_n (기사의 피드 종목 수).
    ``picked``: :func:`pick_judgments` 결과. 피드에 없는 판정 쌍은 가용 시각을 모르므로 버린다.
    ``cfg``: 판정 전 필터(None = 끔). 피드 쌍·판정 쌍 모두에 같은 필터.

    ``n_articles`` = 성공 판정, ``n_error`` = 오류만 있는 쌍, ``n_feed`` = 필터 통과 피드 쌍 전부(판정 여부 무관).
    ``judge`` = 그날 성공 판정에 쓴 judge(여럿이면 '+' 로 이은 정렬 문자열, 없으면 빈 문자열).
    """
    f = feed.drop_duplicates(["article_id", "code"]).copy()
    if cfg is not None and len(f):
        keep = [
            skip_reason(t, None if pd.isna(n) else int(n), cfg) is None
            for t, n in zip(f["title"], f["related_n"], strict=True)
        ]
        f = f[np.asarray(keep, dtype=bool)]
    f["date"] = map_signal_days(f["available_at"], trading_days, mode).to_numpy()
    f = f[f["date"].notna()]
    j = f[["article_id", "code", "date"]].merge(picked, on=["article_id", "code"], how="left")
    ok = j["status"].eq("ok")
    outs = j["output"].where(ok, None)
    j["n_articles"] = ok.astype(int)
    j["n_error"] = j["status"].eq("error").astype(int)
    j["n_feed"] = 1
    j["n_pos_persistent"] = outs.map(is_n_article).astype(int)
    j["n_pos"] = outs.map(is_pos).astype(int)
    j["n_neg"] = outs.map(is_neg).astype(int)
    j["judge_ok"] = j["judge"].where(ok, None)
    if j.empty:
        return pd.DataFrame(columns=list(DAILY_COLS))
    g = j.groupby(["code", "date"], sort=True)
    out = g[["n_pos_persistent", "n_pos", "n_neg", "n_articles", "n_error", "n_feed"]].sum()
    out["judge"] = g["judge_ok"].agg(lambda s: "+".join(sorted(set(s.dropna()))))
    return out.reset_index()[list(DAILY_COLS)]


# --- 60일 N 플래그 -------------------------------------------------------------------------


def n_flag_panel(
    daily: pd.DataFrame,
    dates: Sequence[dt.date],
    *,
    codes: Iterable[str] | None = None,
    window_days: int = WINDOW_DAYS,
    min_coverage: float = MIN_COVERAGE,
) -> pd.DataFrame:
    """신호일 ``dates`` × 종목의 N 플래그 패널 — :data:`PANEL_COLS`.

    창 = 매핑 신호일 s ∈ (d − window_days, d]. ``n_flag`` = 창의 N 기사 ≥ 1. ``news_known`` 은 모듈 docstring 7.
    ``codes`` 를 안 주면 ``daily`` 에 나온 종목 전부. ``daily`` 에 없는 종목·날은 창 건수 0 (피드 0건 → known).
    ``cov_60`` = n_articles_60 / n_feed_60 (피드 0건이면 NaN). ``judges_60`` = 창 안 성공 판정의 judge 들.
    """
    dates = sorted(set(dates))
    d64 = np.array(dates, dtype="datetime64[D]")
    lo64 = d64 - np.timedelta64(window_days, "D")
    codes = sorted(set(daily["code"]) if codes is None else set(codes))
    num = ["n_pos_persistent", "n_articles", "n_error", "n_feed"]
    by_code = {c: g for c, g in daily.groupby("code", sort=False)} if len(daily) else {}
    frames = []
    for code in codes:
        g = by_code.get(code)
        res = {k: np.zeros(len(dates), dtype=int) for k in num}
        judges = [""] * len(dates)
        if g is not None and len(g):
            g = g.sort_values("date")
            s64 = np.array(list(g["date"]), dtype="datetime64[D]")
            hi = np.searchsorted(s64, d64, side="right")
            lo = np.searchsorted(s64, lo64, side="right")
            for k in num:
                cs = np.concatenate([[0], np.cumsum(g[k].to_numpy(dtype=int))])
                res[k] = cs[hi] - cs[lo]
            jv = g["judge"].fillna("").to_numpy()
            for i, (a, b) in enumerate(zip(lo, hi, strict=True)):
                if b > a:
                    judges[i] = "+".join(sorted({x for s in jv[a:b] if s for x in s.split("+")}))
        frames.append(pd.DataFrame({"code": code, "date": dates, **{f"{k}_60": v for k, v in res.items()},
                                    "judges_60": judges}))
    if not frames:
        return pd.DataFrame(columns=list(PANEL_COLS))
    p = pd.concat(frames, ignore_index=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p["cov_60"] = np.where(p["n_feed_60"] > 0, p["n_articles_60"] / p["n_feed_60"].where(p["n_feed_60"] > 0), np.nan)
    p["n_flag"] = p["n_pos_persistent_60"] >= 1
    p["news_known"] = p["n_flag"] | (p["n_feed_60"] == 0) | (p["cov_60"] >= min_coverage)
    return p[list(PANEL_COLS)]


def n_flag_values(panel: pd.DataFrame) -> pd.Series:
    """N 을 float 로 — 참 1.0, 거짓이면서 알면 0.0, 모르면 NaN(없는 뉴스 ≠ 판정 안 됨)."""
    return pd.Series(np.where(panel["n_flag"], 1.0, np.where(panel["news_known"], 0.0, np.nan)), index=panel.index)


# --- DB -----------------------------------------------------------------------------------


def load_trading_days(con) -> list[dt.date]:
    """거래일 달력 — :func:`swing_it.storage.read_prices` 정문, 삼성전자 일봉이 있는 날."""
    from swing_it.storage import read_prices  # noqa: PLC0415

    pr = read_prices(con, cols=("code", "date"))
    return sorted(set(pd.to_datetime(pr.loc[pr["code"] == CALENDAR_CODE, "date"]).dt.date))


def load_judgments(con, judges: Sequence[str] = JUDGE_PRIORITY) -> pd.DataFrame:
    """원장의 judge 들 행(성공·오류 모두) — article_id · code · judge · output · error."""
    cur = con.cursor()
    cur.execute(
        "SELECT article_id, code, judge, output, error FROM article_judgments WHERE judge = ANY(%s)", (list(judges),)
    )
    rows = [(a, c, j, o if o is None or isinstance(o, dict) else json.loads(o), e) for a, c, j, o, e in cur.fetchall()]
    return pd.DataFrame(rows, columns=["article_id", "code", "judge", "output", "error"])


def load_feed(con, lo: dt.datetime, hi: dt.datetime) -> pd.DataFrame:
    """published_at ∈ [lo, hi) 기사의 피드 쌍 — article_id · code · available_at(KST) · title · related_n.

    related_n 은 그 기사의 피드 종목 수 전체(창과 무관). 같은 (기사, 종목)이 두 source 에 있으면 한 줄.
    """
    cur = con.cursor()
    cur.execute(
        "SELECT f.article_id, f.code, a.published_at, a.title FROM news_company_feed f "
        "JOIN news_articles a ON a.id = f.article_id WHERE a.published_at >= %s AND a.published_at < %s",
        (lo, hi),
    )
    df = pd.DataFrame(cur.fetchall(), columns=["article_id", "code", "available_at", "title"])
    df = df.drop_duplicates(["article_id", "code"])
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True).dt.tz_convert(KST)
    df["related_n"] = df.groupby("article_id")["code"].transform("size")
    return df.reset_index(drop=True)


def build_daily(con, *, lo: dt.datetime, hi: dt.datetime, mode: str = "close",
                priority: Sequence[str] = JUDGE_PRIORITY, cfg: PairFilter | None = STAGE1,
                trading_days: Sequence[dt.date] | None = None) -> pd.DataFrame:
    """DB → :func:`daily_counts`. 가용 시각 [lo, hi) 의 피드 쌍 전부(판정 여부 무관)를 분모로 센다."""
    days = list(trading_days) if trading_days is not None else load_trading_days(con)
    picked = pick_judgments(load_judgments(con, priority), priority)
    return daily_counts(load_feed(con, lo, hi), picked, days, mode=mode, cfg=cfg)


# --- 커버리지(건수만) -----------------------------------------------------------------------


def _coverage(con, lo: dt.datetime, hi: dt.datetime) -> None:  # pragma: no cover — DB 리포트
    """coverage_news.md 의 원자료를 출력한다. 가격·수익은 읽지 않는다."""
    days = load_trading_days(con)
    raw = load_judgments(con, ("v2r3-haiku55", "v2-haiku45", "v2-haiku55"))
    picked = pick_judgments(raw)
    feed = load_feed(con, lo, hi)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # judge 별 가용 시각 범위·월별 건수(필터 전, 판정 행 그대로)
    meta = feed.drop_duplicates("article_id").set_index("article_id")["available_at"]
    r = raw.copy()
    r["available_at"] = r["article_id"].map(meta)
    print("## 원장 행(필터 전) — 가용 시각 없음:", int(r["available_at"].isna().sum()))
    r = r.dropna(subset=["available_at"])
    r["ok"] = r["output"].notna()
    r["month"] = r["available_at"].dt.strftime("%Y-%m")
    print(r.groupby("judge").agg(rows=("ok", "size"), ok=("ok", "sum"), first=("available_at", "min"),
                                 last=("available_at", "max"), codes=("code", "nunique")))
    print(r.pivot_table(index="month", columns="judge", values="ok", aggfunc="size", fill_value=0))

    for mode in MODES:
        d = daily_counts(feed, picked, days, mode=mode)
        d["month"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m")
        m = d.groupby("month").agg(
            feed=("n_feed", "sum"), judged=("n_articles", "sum"), err=("n_error", "sum"),
            pos=("n_pos", "sum"), neg=("n_neg", "sum"), npos=("n_pos_persistent", "sum"),
            codes_feed=("code", "nunique"),
        )
        m["codes_judged"] = d[d["n_articles"] > 0].groupby("month")["code"].nunique()
        m["cov"] = (m["judged"] / m["feed"]).round(3)
        m["n_rate"] = (m["npos"] / m["judged"]).round(4)
        m["pos_rate"] = (m["pos"] / m["judged"]).round(4)
        print(f"\n## 월별(필터 STAGE1 후, mode={mode})")
        print(m[(m.index >= "2025-08") & (m.index <= "2026-10")])
        jd = d[d["n_articles"] > 0].groupby("judge").agg(
            first=("date", "min"), last=("date", "max"), judged=("n_articles", "sum"), npos=("n_pos_persistent", "sum"),
            pos=("n_pos", "sum"))
        jd["n_rate"] = (jd["npos"] / jd["judged"]).round(4)
        print(jd)
        if mode == "close":
            close_daily = d
    # overnight 와 close 의 신호일이 다른 N 기사 수
    fk = feed.drop_duplicates(["article_id", "code"])
    sc = map_signal_days(fk["available_at"], days, "close")
    so = map_signal_days(fk["available_at"], days, "overnight")
    fk = fk.assign(sc=sc.to_numpy(), so=so.to_numpy()).merge(picked[picked["status"] == "ok"], on=["article_id", "code"])
    fk["isn"] = fk["output"].map(is_n_article)
    print("\n## 판정 쌍 중 신호일이 모드 간 다른 것:", int((fk["sc"] != fk["so"]).sum()), "/", len(fk),
          "· 그중 N 기사:", int(((fk["sc"] != fk["so"]) & fk["isn"]).sum()), "/", int(fk["isn"].sum()))

    # 패널(신호일 = 판정이 있는 거래일 범위)
    span = [x for x in days if dt.date(2025, 9, 1) <= x <= dt.date(2026, 9, 30)]
    p = n_flag_panel(close_daily, span)
    p["month"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m")
    for thr in (0.2, 0.5, 0.9):
        k = p["n_flag"] | (p["n_feed_60"] == 0) | (p["cov_60"] >= thr)
        print(f"known@{thr}:", p.assign(k=k).groupby("month")["k"].mean().round(3).to_dict())
    print("\n## 패널 월별: 종목-일 중 N 참 비율, known(0.5) 비율, 피드>0 종목-일 중 known 비율")
    q = p.assign(feed_pos=p["n_feed_60"] > 0)
    print(q.groupby("month").agg(rows=("n_flag", "size"), n_true=("n_flag", "mean"), known=("news_known", "mean"),
                                 n_true_cnt=("n_flag", "sum")))

    # judge 단절: 같은 500쌍에서 v2-haiku45 vs v2-haiku55
    a = raw[(raw["judge"] == "v2-haiku45") & raw["output"].notna()].set_index(["article_id", "code"])["output"]
    b = raw[(raw["judge"] == "v2-haiku55") & raw["output"].notna()].set_index(["article_id", "code"])["output"]
    both = a.index.intersection(b.index)
    na, nb = a.loc[both].map(is_n_article), b.loc[both].map(is_n_article)
    pa, pb = a.loc[both].map(is_pos), b.loc[both].map(is_pos)
    print(f"\n## 500 파일럿 겹침 {len(both)}쌍: N 기사 haiku45 {int(na.sum())} · haiku55 {int(nb.sum())} · 둘 다 "
          f"{int((na & nb).sum())} | pos haiku45 {int(pa.sum())} · haiku55 {int(pb.sum())} · 둘 다 {int((pa & pb).sum())}")
    for fld in ("persistence", "freshness", "sentiment_direction", "subject", "reports_price_move"):
        ct = pd.crosstab(a.loc[both].map(lambda o, f=fld: str(o.get(f))), b.loc[both].map(lambda o, f=fld: str(o.get(f))),
                         rownames=["haiku45"], colnames=["haiku55"])
        print(fld, "\n", ct)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coverage", action="store_true", help="커버리지 건수 출력(수익 없음)")
    ap.add_argument("--from", dest="lo", default="2025-06-01")
    ap.add_argument("--to", dest="hi", default="2026-10-10")
    args = ap.parse_args(argv)
    if not args.coverage:
        ap.error("지금은 --coverage 만 있다")
    from swing_it.news.judgments_db import connect_db  # noqa: PLC0415

    con = connect_db()
    lo = dt.datetime.combine(dt.date.fromisoformat(args.lo), dt.time(0), tzinfo=KST)
    hi = dt.datetime.combine(dt.date.fromisoformat(args.hi), dt.time(0), tzinfo=KST)
    _coverage(con, lo, hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
