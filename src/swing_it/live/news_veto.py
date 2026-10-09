"""뉴스 거부권(veto) — 라이브 신호 경로의 마지막 문 (설계 §7 · ``research/logs/sepa_explore/NEWS_VETO.md`` 사전등록).

뉴스는 매수 조건이 아니라 **거부권**이다. :func:`swing_it.live.strategy.scan` 이 낸 후보마다 최근 20거래일 기사를 보고

- 악재 = 판정 출력의 subject ∈ {main, partial} ∧ sentiment_direction = −1 (:func:`swing_it.news.daily_leg.is_neg` 그대로),
- 신호일 d 기준 **직전 5거래일 안에 악재 ≥ 1건** 또는 **20거래일 안에 악재 ≥ 2건** 이면 거부(진입하지 않는다),
- 모름(기사 없음 · 판정 실패 · 키 없음 · 판정 한도 초과)은 **거부하지 않는다** — 모름 ≠ 악재.

기사의 신호일 매핑은 연구와 같다(``daily_leg.map_signal_days(mode="close")`` — published_at < d 15:30 KST 면 d, 그 뒤면 다음
거래일). 판정 전 필터도 같다(:data:`swing_it.news.prefilter.STAGE1`). 판정자 = ``v2r3-haiku55`` 와 **같은 정의**(운영 v2 system +
r3 보충, Haiku 5.5, thinking disabled) 를 이름 ``v2r3-haiku55-live`` 로 원장(``article_judgments``)에 쓴다 — 이름을 나누는 이유는
연구 배치(기간·표본이 고정)와 라이브 일별 판정을 원장에서 구분하려는 것이고, 정의가 같으므로 입력 해시도 같다.

**이미 판정된 쌍은 다시 판정하지 않는다:** ``v2r3-haiku55-live`` 성공 행 → 그대로, 없으면 같은 정의의 ``v2r3-haiku55`` 성공 행
(2025-09~2026-02 연구 배치)을 재사용한다. 둘 다 없는 쌍만 그날 판정한다(비배치 Messages API — 배치는 수 시간이 걸려 07:35 신호
작업에 못 쓴다). 판정 실패(스키마·전송 오류)는 원장에 실패 행으로 적고 모름으로 센다.

**비용·안전 가드:** 하루 판정 상한 :data:`MAX_JUDGED_PER_DAY` (넘으면 나머지는 모름 + 경보), ``ANTHROPIC_API_KEY`` 가 없으면
전 후보 모름 + 경보 — **신호 작업은 어떤 경우에도 막지 않는다**(거부권 층이 죽으면 "모름" 으로 접힌다). 키는 어디에도 찍지 않는다.

거부된 후보는 ``vetoed`` 로 따로 돌려준다 — 런타임이 **가상 진입**(섀도 포지션)으로 추적해 전진 데이터에서 거부군 vs 진입군을
비교한다(합격 바: NEWS_VETO.md).

백테스트(``swing_it.live.backtest.run_backtest(veto=...)``)는 API 를 부르지 않고 **과거 원장**(judge 우선순위 ``v2r3-haiku55`` →
``v2-haiku45``, :func:`veto_panel_from_daily`)으로 같은 규칙을 적용한다 — 판정 원장이 있는 2025-09 이후에만 뜻이 있다.

경계 규칙: ``research/`` 를 import 하지 않는다. DB 읽기·쓰기는 :mod:`swing_it.news.judgments_db` 의 창구로만.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

import pandas as pd

from swing_it.news.daily_leg import CLOSE_CUTOFF, JUDGE_PRIORITY, is_neg, map_signal_days
from swing_it.news.judgments_db import OWNER, V2R3_HAIKU55, JudgeSpec, JudgmentLedger, read_judgments
from swing_it.news.prefilter import STAGE1, skip_reason
from swing_it.news.prompt import ADDENDUM_R3, SchemaError, parse_features
from swing_it.news.sweep import pair_key_article

KST = dt.timezone(dt.timedelta(hours=9))
API = "https://api.anthropic.com"
MODEL = "claude-haiku-5-5"
ADDENDUM = ADDENDUM_R3.strip()
LIVE_JUDGE = "v2r3-haiku55-live"
#: 재사용할 성공 판정(우선순위) — 라이브 행, 그다음 같은 정의의 연구 배치 행.
REUSE_JUDGES: tuple[str, ...] = (LIVE_JUDGE, "v2r3-haiku55")
#: 하루 판정 상한(쌍 수). 후보 ≤ 수십 종목 × 20일 기사라 보통 수십~수백. 넘으면 나머지 모름 + 경보.
MAX_JUDGED_PER_DAY = 500
#: 거부 경계(사전등록): 직전 5거래일 악재 ≥ 1 · 20거래일 악재 ≥ 2.
NEG5_MIN = 1
NEG20_MIN = 2
WINDOW_SHORT = 5
WINDOW_LONG = 20
#: 요청 하나의 타임아웃(초)과 하루 판정 전체의 시간 예산(초) — 넘으면 나머지 모름 + 경보.
REQUEST_TIMEOUT = 30.0
TIME_BUDGET = 600.0

LIVE_SPEC = JudgeSpec(
    judge=LIVE_JUDGE,
    model=V2R3_HAIKU55.model,
    system_prompt=V2R3_HAIKU55.system_prompt,
    user_template=V2R3_HAIKU55.user_template,
    params=V2R3_HAIKU55.params,
    output_schema="v2",
    owner_repo=OWNER,
    description=(
        "스윙 라이브 뉴스 거부권(설계 §7, NEWS_VETO.md) 일별 판정 — v2r3-haiku55 와 같은 정의(운영 v2 system + r3 보충, "
        "Haiku 5.5, thinking disabled). 비배치 Messages API, judged_at = 응답 시각(exact TRUE). 신호 작업이 후보 종목의 "
        "최근 20거래일 미판정 기사만 판정한다."
    ),
)


# ---------------------------------------------------------------------------
# 순수 로직
# ---------------------------------------------------------------------------

@dataclass
class CodeNews:
    """한 종목의 거부권 판단 재료 — 신호일 매핑된 (기사, 판정) 목록."""

    code: str
    neg5: int = 0
    neg20: int = 0
    judged: int = 0          # 성공 판정 쌍 수(20일)
    unknown: int = 0         # 판정 없음·실패 쌍 수(20일)
    feed: int = 0            # 필터 통과 피드 쌍 수(20일)

    @property
    def vetoed(self) -> bool:
        return veto_decision(self.neg5, self.neg20)


def veto_decision(neg5: int, neg20: int) -> bool:
    """직전 5거래일 악재 ≥ 1 또는 20거래일 악재 ≥ 2 → 거부. 모름은 세지 않으므로 0 = 거부 안 함."""
    return neg5 >= NEG5_MIN or neg20 >= NEG20_MIN


def window_days(trading_days: Sequence[dt.date], d: dt.date, n: int) -> list[dt.date]:
    """``d`` 포함 직전 ``n`` 거래일(오름차순). ``d`` 가 달력에 없으면 d 이하 마지막 거래일 기준."""
    days = sorted(trading_days)
    i = max(k for k, x in enumerate(days) if x <= d) if any(x <= d for x in days) else -1
    if i < 0:
        return []
    return days[max(i - n + 1, 0): i + 1]


def count_news(pairs: pd.DataFrame, judged: dict[str, dict], trading_days: Sequence[dt.date], d: dt.date,
               *, cfg=STAGE1) -> dict[str, CodeNews]:
    """쌍 표 + 판정 → 종목별 :class:`CodeNews`.

    ``pairs``: ``key``(uuid|code) · ``code`` · ``available_at``(tz-aware) · ``title`` · ``related_n``.
    ``judged``: 키 → 판정 output(dict) — 성공 행만. 없는 키는 모름.
    창: 매핑 신호일 ∈ 직전 20거래일(d 포함); 5일 창은 그 안의 마지막 5거래일.
    """
    out: dict[str, CodeNews] = {}
    if pairs.empty:
        return out
    p = pairs.drop_duplicates("key").copy()
    if cfg is not None:
        keep = [skip_reason(t, None if pd.isna(n) else int(n), cfg) is None
                for t, n in zip(p["title"], p["related_n"], strict=True)]
        p = p[pd.Series(keep, index=p.index).to_numpy()]
    if p.empty:
        return out
    p["signal_day"] = map_signal_days(p["available_at"], trading_days, "close").to_numpy()
    long = set(window_days(trading_days, d, WINDOW_LONG))
    short = set(window_days(trading_days, d, WINDOW_SHORT))
    p = p[[s in long for s in p["signal_day"]]]
    for code, g in p.groupby("code", sort=False):
        c = CodeNews(code=str(code))
        for key, s in zip(g["key"], g["signal_day"], strict=True):
            c.feed += 1
            o = judged.get(key)
            if o is None:
                c.unknown += 1
                continue
            c.judged += 1
            if is_neg(o):
                c.neg20 += 1
                if s in short:
                    c.neg5 += 1
        out[c.code] = c
    return out


def split_candidates(cands: list, news: dict[str, CodeNews]) -> tuple[list, list]:
    """후보 → (진입 후보, 거부 후보). 재료가 없는 종목은 모름 = 진입."""
    kept, vetoed = [], []
    for c in cands:
        n = news.get(c.code)
        (vetoed if (n is not None and n.vetoed) else kept).append(c)
    return kept, vetoed


# ---------------------------------------------------------------------------
# 판정(비배치 Messages API) — 주입 가능한 judge_fn 으로 테스트한다
# ---------------------------------------------------------------------------

JudgeFn = Callable[[dict], tuple[dict | None, str | None, dict | None]]


def api_key_from_env() -> str | None:
    """``ANTHROPIC_API_KEY`` 환경변수, 없으면 ``../quant-airflow/.env``. 찍지 않는다."""
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    from pathlib import Path  # noqa: PLC0415

    env = Path(__file__).resolve().parents[3].parent / "quant-airflow" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def messages_judge(api_key: str, *, timeout: float = REQUEST_TIMEOUT) -> JudgeFn:
    """비배치 ``/v1/messages`` 로 한 쌍을 판정하는 함수 — 요청 본문은 배치 판정기와 같다(``judge.build_request``)."""
    import httpx  # noqa: PLC0415

    from swing_it.news.judge import build_request  # noqa: PLC0415

    client = httpx.Client(timeout=timeout)
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def judge(pair: dict) -> tuple[dict | None, str | None, dict | None]:
        params = build_request(pair, MODEL, ADDENDUM)["params"]
        try:
            resp = client.post(f"{API}/v1/messages", headers=headers, json=params)
        except httpx.HTTPError as exc:
            return None, f"transport: {type(exc).__name__}", None
        if resp.status_code != 200:
            return None, f"http {resp.status_code}: {resp.text[:200]}", None
        body = resp.json()
        usage = body.get("usage")
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        try:
            feat = parse_features(text)
        except SchemaError as exc:
            return None, f"schema: {exc}", usage
        return {**asdict(feat), "score": feat.score()}, None, usage

    return judge


def judge_pending(pairs: list[dict], judge_fn: JudgeFn, ledger: "JudgmentLedger | None", *,
                  cap: int = MAX_JUDGED_PER_DAY, time_budget: float = TIME_BUDGET,
                  now: Callable[[], dt.datetime] | None = None) -> tuple[dict[str, dict], dict]:
    """미판정 쌍을 판정하고 원장에 쓴다 → (키 → 성공 output, 정보). ``cap``·``time_budget`` 을 넘기면 나머지는 모름.

    ``ledger`` 가 None 이면 원장에 쓰지 않는다(테스트·드라이런). 원장 쓰기 실패는 경보로 적고 판정 결과는 그대로 쓴다
    (그날 거부권엔 쓰고, 원장엔 다음 날 다시 판정해 쓴다 — 신호 작업을 막지 않는다).
    """
    from swing_it.news.judge import pair_input_hash  # noqa: PLC0415

    now = now or (lambda: dt.datetime.now(dt.UTC))
    info = {"n_pending": len(pairs), "n_judged": 0, "n_ok": 0, "n_error": 0, "n_skipped_cap": 0, "alerts": [],
            "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                      "output_tokens": 0}}
    outputs: dict[str, dict] = {}
    entries: list[dict] = []
    t0 = time.monotonic()
    for i, pair in enumerate(pairs):
        if i >= cap or time.monotonic() - t0 > time_budget:
            info["n_skipped_cap"] = len(pairs) - i
            info["alerts"].append(f"뉴스 거부권: 판정 상한(쌍 {cap}·{time_budget:.0f}s) 초과 — {len(pairs) - i}쌍 모름 처리")
            break
        output, error, usage = judge_fn(pair)
        info["n_judged"] += 1
        if usage:
            for k in info["usage"]:
                info["usage"][k] += int(usage.get(k) or 0)
        if output is not None:
            outputs[pair["key"]] = output
            info["n_ok"] += 1
        else:
            info["n_error"] += 1
        entries.append({"key": pair["key"], "input_hash": pair_input_hash(pair, MODEL, ADDENDUM), "output": output,
                        "error": error, "judged_at": now(), "judged_at_exact": True, "batch_id": None, "usage": usage})
    if ledger is not None and entries:
        try:
            info["ledger"] = ledger.record(entries)
        except Exception as exc:  # noqa: BLE001 — 원장 실패가 신호를 막지 않는다
            info["alerts"].append(f"뉴스 거부권: 원장 쓰기 실패({type(exc).__name__}: {str(exc)[:120]})")
    return outputs, info


# ---------------------------------------------------------------------------
# DB — 후보 종목의 최근 기사 쌍
# ---------------------------------------------------------------------------

def load_candidate_pairs(con, codes: Iterable[str], lo: dt.datetime, hi: dt.datetime) -> pd.DataFrame:
    """후보 종목의 피드 쌍(published_at ∈ [lo, hi)) + 판정 요청에 필요한 열(이름·시장·업종·제목·요약).

    열: ``key`` · ``article_id`` · ``code`` · ``available_at``(KST) · ``title`` · ``summary`` · ``body`` · ``name`` ·
    ``market`` · ``sector`` · ``related_n``(기사의 피드 종목 수 전체). 같은 id 가 두 판이면 먼저 수집된 판.
    """
    from swing_it.news.pairs import load_stocks  # noqa: PLC0415

    codes = sorted(set(map(str, codes)))
    cols = ["key", "article_id", "code", "available_at", "title", "summary", "body", "name", "market", "sector",
            "related_n"]
    if not codes:
        return pd.DataFrame(columns=cols)
    cur = con.cursor()
    cur.execute(
        "SELECT f.article_id, f.code, a.published_at, a.collected_at, a.title, a.summary FROM news_company_feed f "
        "JOIN news_articles a ON a.id = f.article_id WHERE f.code = ANY(%s) AND a.published_at >= %s AND a.published_at < %s",
        (codes, lo, hi),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["article_id", "code", "available_at", "collected_at", "title", "summary"])
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    df = df.sort_values(["article_id", "code", "collected_at"]).drop_duplicates(["article_id", "code"])
    ids = sorted(set(df["article_id"]))
    cur.execute("SELECT article_id, COUNT(DISTINCT code) FROM news_company_feed WHERE article_id = ANY(%s) GROUP BY article_id",
                (ids,))
    fan = dict(cur.fetchall())
    stocks = load_stocks(con)
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True).dt.tz_convert(KST)
    df["related_n"] = df["article_id"].map(fan).fillna(1).astype(int)
    df["key"] = [f"{pair_key_article(a)}|{c}" for a, c in zip(df["article_id"], df["code"], strict=True)]
    df["title"] = df["title"].fillna("")
    df["summary"] = df["summary"].fillna("")
    df["body"] = df["summary"].where(df["summary"].str.strip() != "", df["title"])
    df["name"] = [getattr(stocks.get(c), "name", c) for c in df["code"]]
    df["market"] = [getattr(stocks.get(c), "market", None) for c in df["code"]]
    df["sector"] = [getattr(stocks.get(c), "sector", None) for c in df["code"]]
    return df[cols].reset_index(drop=True)


def pair_dicts(pairs: pd.DataFrame) -> list[dict]:
    """쌍 표 행 → 판정 요청용 dict(``judge.build_request`` 입력)."""
    return [{"key": r.key, "code": r.code, "name": r.name, "market": r.market, "sector": r.sector, "title": r.title,
             "body": r.body, "summary": r.summary, "category": None} for r in pairs.itertuples(index=False)]


# ---------------------------------------------------------------------------
# 라이브 진입점
# ---------------------------------------------------------------------------

@dataclass
class VetoResult:
    kept: list
    vetoed: list
    news: dict[str, CodeNews] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def alerts(self) -> list[str]:
        return list(self.info.get("alerts", []))


def apply_live_veto(con, cands: list, d: dt.date, trading_days: Sequence[dt.date], *,
                    judge_fn: "JudgeFn | None" = None, api_key: "str | None" = None, write_ledger: bool = True,
                    cap: int = MAX_JUDGED_PER_DAY, time_budget: float = TIME_BUDGET) -> VetoResult:
    """신호일 ``d`` 의 후보에 뉴스 거부권을 건다(모듈 설명). 어떤 실패도 예외로 올리지 않는다 — 모름 + 경보.

    ``judge_fn`` 을 주면 그걸로 판정(테스트). 없으면 ``api_key``(없으면 환경)로 :func:`messages_judge`; 키가 없으면 전부 모름.
    """
    info: dict = {"alerts": [], "n_candidates": len(cands)}
    if not cands:
        return VetoResult(kept=[], vetoed=[], info=info)
    days = sorted(trading_days)
    win = window_days(days, d, WINDOW_LONG)
    if not win:
        info["alerts"].append("뉴스 거부권: 거래일 달력이 비어 있다 — 전 후보 모름")
        return VetoResult(kept=list(cands), vetoed=[], info=info)
    lo = dt.datetime.combine(win[0], dt.time(0), tzinfo=KST) - dt.timedelta(days=4)
    hi = dt.datetime.combine(win[-1], CLOSE_CUTOFF, tzinfo=KST)
    codes = [c.code for c in cands]
    try:
        pairs = load_candidate_pairs(con, codes, lo, hi)
    except Exception as exc:  # noqa: BLE001
        info["alerts"].append(f"뉴스 거부권: 기사 읽기 실패({type(exc).__name__}) — 전 후보 모름")
        return VetoResult(kept=list(cands), vetoed=[], info=info)
    info["n_pairs"] = int(len(pairs))
    # 필터 통과 쌍만 판정 대상(연구와 같은 모집단)
    if len(pairs):
        keep = [skip_reason(t, int(n), STAGE1) is None for t, n in zip(pairs["title"], pairs["related_n"], strict=True)]
        pairs = pairs[pd.Series(keep, index=pairs.index).to_numpy()].reset_index(drop=True)
    info["n_pairs_filtered"] = int(len(pairs))
    judged: dict[str, dict] = {}
    if len(pairs):
        keys = list(pairs["key"])
        try:
            for j in REUSE_JUDGES:
                for k, row in read_judgments(con, j, keys).items():
                    if k not in judged and row.get("output"):
                        judged[k] = row["output"]
        except Exception as exc:  # noqa: BLE001
            info["alerts"].append(f"뉴스 거부권: 원장 읽기 실패({type(exc).__name__}) — 기존 판정 없이 진행")
        info["n_reused"] = len(judged)
        pending = [p for p in pair_dicts(pairs) if p["key"] not in judged]
        info["n_pending"] = len(pending)
        if pending:
            fn = judge_fn
            if fn is None:
                key = api_key or api_key_from_env()
                if not key:
                    info["alerts"].append(f"뉴스 거부권: ANTHROPIC_API_KEY 없음 — 미판정 {len(pending)}쌍 모름 처리")
                else:
                    fn = messages_judge(key)
            if fn is not None:
                ledger = JudgmentLedger(con, LIVE_SPEC) if write_ledger else None
                new, jinfo = judge_pending(pending, fn, ledger, cap=cap, time_budget=time_budget)
                judged.update(new)
                info["judge"] = {k: v for k, v in jinfo.items() if k != "alerts"}
                info["alerts"].extend(jinfo["alerts"])
    news = count_news(pairs, judged, days, d, cfg=None)        # 필터는 위에서 이미 걸었다
    kept, vetoed = split_candidates(cands, news)
    info["n_vetoed"] = len(vetoed)
    info["vetoed"] = {c.code: {"neg5": news[c.code].neg5, "neg20": news[c.code].neg20} for c in vetoed}
    return VetoResult(kept=kept, vetoed=vetoed, news=news, info=info)


# ---------------------------------------------------------------------------
# 백테스트 — 과거 원장으로 같은 규칙
# ---------------------------------------------------------------------------

def veto_panel_from_daily(daily: pd.DataFrame, trading_days: Sequence[dt.date], codes: Iterable[str]
                          ) -> pd.DataFrame:
    """``daily_leg.build_daily`` 결과(종목·신호일·n_neg) → 날짜×종목 bool 패널(거부 여부). 창은 거래일 기준(5·20)."""
    days = pd.DatetimeIndex(sorted(pd.to_datetime(list(trading_days))))
    codes = pd.Index(sorted(set(map(str, codes))))
    out = pd.DataFrame(False, index=days, columns=codes)
    if daily is None or daily.empty:
        return out
    d = daily[daily["code"].isin(codes)].copy()
    d["date"] = pd.to_datetime(d["date"])
    neg = d.pivot_table(index="date", columns="code", values="n_neg", aggfunc="sum").reindex(index=days, columns=codes)
    neg = neg.fillna(0.0)
    neg5 = neg.rolling(WINDOW_SHORT, min_periods=1).sum()
    neg20 = neg.rolling(WINDOW_LONG, min_periods=1).sum()
    return (neg5 >= NEG5_MIN) | (neg20 >= NEG20_MIN)


def veto_panel_from_ledger(con, lo: dt.datetime, hi: dt.datetime, trading_days: Sequence[dt.date],
                           codes: Iterable[str], priority: Sequence[str] = JUDGE_PRIORITY) -> pd.DataFrame:
    """DB 원장(judge 우선순위 v2r3-haiku55 → v2-haiku45) → 거부 패널. 백테스트 러너가 부른다."""
    from swing_it.news.daily_leg import build_daily  # noqa: PLC0415

    daily = build_daily(con, lo=lo, hi=hi, mode="close", priority=priority, trading_days=list(trading_days))
    return veto_panel_from_daily(daily, trading_days, codes)
