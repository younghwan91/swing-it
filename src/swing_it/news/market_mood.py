"""LLM 시장 기사 기분 지수 — 적응 층 B (``research/logs/sepa_explore/ADAPTIVE.md`` §1-B, 설계 고정 후 자료 준비).

매일 시황 기사(제목에 코스피·코스닥·증시·마감·외국인) 중 **발행 시각 순 10건**을 Haiku 5.5 로 판정한다 — 고정 스키마
``market_direction(−1/0/1)``·``risk_appetite(0~2)``·``foreign_flow_tone(−1/0/1)``·``breadth_tone(−1/0/1)``·``event_risk(0/1)``,
JSON 만. 일별 집계 = 평균, 20일 지수이동평균 = 지수.

**시점(PIT):** 기사의 신호일은 :func:`swing_it.news.daily_leg.map_signal_days` (``mode="close"``) 그대로 — 발행 시각
< d 15:30 KST 면 d, 그 뒤면 다음 거래일. ``d`` 행은 ``d`` 15:30 전 기사만 담는다(경계 15:30 정각은 다음날 — daily_leg 와 같다).

**선택:** 제목 :data:`INCLUDE` 매치 ∧ :data:`EXCLUDE` 비매치(종목별 수급 포착·특징주 같은 종목 기사 태그 — 설계 패턴
"외국인" 이 종목 기사도 잡기 때문). 같은 날 안에서 거의 같은 제목(정규화 후 동일, 또는 문자 2-gram 자카드 ≥ :data:`DEDUPE_SIM`)
은 먼저 난 것만. 그다음 발행 시각 순 앞 :data:`PER_DAY` 건.

**판정 원장:** judge ``market-mood-haiku55`` (:data:`SPEC` — 모델·system·user 템플릿·파라미터로 정의, 프롬프트 해시 고정)
로 ``article_judgments`` 에 ``code='MARKET'`` 행을 쓴다(:mod:`swing_it.news.judgments_db` 창구만). 쌍 키는 다른 판정과 같은
``uuid5(기사 id)|MARKET`` 이고, 역사전은 선택한 기사 id 로 직접 만든다(시황 기사는 종목 피드에 없을 수 있다).

**두 경로:** 파일럿은 비배치 ``/v1/messages`` (:func:`messages_judge` — :mod:`swing_it.live.news_veto` 와 같은 모양, 즉시 결과·
실측 usage), 전 구간은 Message Batches (:class:`MarketBatchJudge` — :class:`swing_it.news.judge.BatchJudge` 의 제출·대기·
상태 파일·재시도 금지를 그대로 물려받고 요청 생성·파싱만 바꾼다). system 은 ``cache_control`` 블록으로 보낸다.

**비용 가드:** 추정(:func:`estimate_usd`, 글자수 기반·보수적)이 하드캡을 넘으면 아무것도 보내지 않는다. 비배치는 실측 누계가
예산을 넘으면 그 자리에서 멈춘다. 키는 ``ANTHROPIC_API_KEY`` 또는 ``../quant-airflow/.env`` (:func:`swing_it.news.judge._api_key`),
어디에도 찍지 않는다. 수익·트레이드 결과는 이 모듈 어디에도 없다.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import pandas as pd

from swing_it.news.daily_leg import map_signal_days
from swing_it.news.judge import (
    BATCH_DISCOUNT,
    CACHE_READ_MULT,
    CACHE_WRITE_MULT,
    USAGE_FIELDS,
    BatchJudge,
    custom_id_of,
    prices_for,
)
from swing_it.news.judgments_db import OWNER, JudgeSpec, JudgmentLedger, read_judgments, register_judge
from swing_it.news.prompt import SchemaError, _extract_json_object, judgment_input_hash
from swing_it.news.sweep import pair_key_article

KST = dt.timezone(dt.timedelta(hours=9))
API = "https://api.anthropic.com"
MODEL = "claude-haiku-5-5"
JUDGE = "market-mood-haiku55"
CODE = "MARKET"
PER_DAY = 10
MAX_OUTPUT_TOKENS = 160
DEDUPE_SIM = 0.6
REQUEST_TIMEOUT = 30.0
#: 추정용(보수적): 글자당 1 토큰(실측 ~1.3 글자/토큰), 출력 60 토큰.
EST_CHARS_PER_TOKEN = 1.0
EST_OUTPUT_TOKENS = 60

#: 설계 패턴 — 제목에 코스피·코스닥·증시·마감·외국인.
INCLUDE = re.compile(r"코스피|코스닥|증시|마감|외국인")
#: 종목 기사 태그·종목 수급 포착·주가 등락 보도(설계 패턴 "외국인" 이 잡는 종목 기사) — 시황이 아니다.
EXCLUDE = re.compile(
    r"장중수급포착|특징주|급등주|급락주|상한가|하한가|52주|신고가|신저가|목표가|리포트|공시|순매[수도]행진|연속\s*순매[수도]"
    r"|주가\s*[+\-−]?\s*\d|\[속보\]\s*\S+\s*(?:주가|급등|급락)"
)

FIELDS: dict[str, tuple[int, ...]] = {
    "market_direction": (-1, 0, 1),
    "risk_appetite": (0, 1, 2),
    "foreign_flow_tone": (-1, 0, 1),
    "breadth_tone": (-1, 0, 1),
    "event_risk": (0, 1),
}

SYSTEM = """당신은 한국 주식시장 시황 기사를 읽고 그날 시장의 분위기를 고정 스키마로 추출하는 판정기다.
기사 텍스트(제목·요약)에 적힌 내용만 근거로 한다. 본문에 없는 내용은 추정하지 않는다. 개별 종목의 호재·악재가 아니라
**시장 전체**의 분위기를 읽는다.

필드(정수만):
- market_direction: 기사가 전하는 시장(코스피·코스닥) 방향. −1 = 하락·약세, 0 = 보합·혼조·방향 언급 없음, 1 = 상승·강세.
- risk_appetite: 위험 선호. 0 = 위험 회피(공포·투매·안전자산·급락·변동성 확대), 1 = 중립, 2 = 위험 선호(매수세 유입·신고점·
  낙관·강한 순매수).
- foreign_flow_tone: 외국인 수급 톤. −1 = 외국인 순매도·이탈, 0 = 언급 없음·혼조, 1 = 외국인 순매수·유입.
- breadth_tone: 시장 폭. −1 = 대형주·소수 종목 쏠림(지수만 오르고 종목은 약세, 반도체 쏠림 등), 0 = 중립·언급 없음,
  1 = 순환매·중소형주 동반 강세·오르는 종목이 많음.
- event_risk: 거시·정책 충격 언급. 1 = 전쟁·관세·금리 급변·정책 발표·규제·환율 급변 같은 외부 충격이 기사 주제, 0 = 아니다.

출력은 아래 모양의 JSON 객체 하나뿐이다. 설명·접두어·코드펜스 없이 JSON 만 쓴다.
{"market_direction":0,"risk_appetite":1,"foreign_flow_tone":0,"breadth_tone":0,"event_risk":0}"""

USER_TEMPLATE = """발행 시각(KST): {published}
제목: {title}
요약: {summary}"""

SPEC = JudgeSpec(
    judge=JUDGE,
    model=MODEL,
    system_prompt=SYSTEM,
    user_template=USER_TEMPLATE,
    params={"max_tokens": MAX_OUTPUT_TOKENS, "thinking": {"type": "disabled"}},
    output_schema="market_mood_v1",
    owner_repo=OWNER,
    description=(
        "적응 층 B(ADAPTIVE.md §1-B) 시황 기사 기분 판정 — 제목·요약만 입력, 고정 스키마 5필드(market_direction·risk_appetite·"
        "foreign_flow_tone·breadth_tone·event_risk), Haiku 5.5, thinking disabled, temperature 없음. code='MARKET'. "
        "system 은 cache_control 블록으로 보내지만 모델 입력은 같다. 파일럿(2026-07~09, 비배치 Messages, judged_at exact TRUE) "
        "뒤 전 구간(2023-03~, Batch API, judged_at = 배치 종료 시각)."
    ),
)

JudgeFn = Callable[[dict], tuple["dict | None", "str | None", "dict | None"]]


# ---------------------------------------------------------------------------
# 선택 (순수)
# ---------------------------------------------------------------------------

_TAG = re.compile(r"\[[^\]]*\]|【[^】]*】|\([^)]*\)")
_NONWORD = re.compile(r"[^0-9a-z가-힣]+")


def normalize_title(title: str) -> str:
    """대괄호 태그·괄호·공백·구두점 제거, 소문자 — 거의 같은 제목을 같은 문자열로."""
    return _NONWORD.sub("", _TAG.sub("", str(title)).lower())


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def title_similarity(a: str, b: str) -> float:
    """정규화 제목의 문자 2-gram 자카드."""
    x, y = _bigrams(normalize_title(a)), _bigrams(normalize_title(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def is_market_title(title: str) -> bool:
    t = str(title)
    return bool(INCLUDE.search(t)) and not EXCLUDE.search(t)


def select_market_articles(articles: pd.DataFrame, trading_days: Sequence[dt.date], *, per_day: int = PER_DAY,
                           sim: float = DEDUPE_SIM) -> pd.DataFrame:
    """후보 기사(``id``·``published_at``(tz-aware)·``title``·``summary``) → 신호일별 발행 순 앞 ``per_day`` 건.

    반환 열: ``id``·``published_at``·``title``·``summary``·``signal_day``(datetime.date)·``key``. 달력 밖 기사는 버린다."""
    cols = ["id", "published_at", "title", "summary", "signal_day", "key"]
    if articles.empty:
        return pd.DataFrame(columns=cols)
    a = articles.drop_duplicates("id").copy()
    a = a[a["title"].map(is_market_title)]
    a["published_at"] = pd.to_datetime(a["published_at"], utc=True).dt.tz_convert(KST)
    a = a.sort_values(["published_at", "id"]).reset_index(drop=True)
    a["signal_day"] = map_signal_days(a["published_at"], trading_days, mode="close").to_numpy()
    a = a[a["signal_day"].notna()]
    keep: list[int] = []
    for _, g in a.groupby("signal_day", sort=True):
        chosen: list[str] = []
        for i, title in zip(g.index, g["title"]):
            if len(chosen) >= per_day:
                break
            norm = normalize_title(title)
            if any(normalize_title(c) == norm or title_similarity(c, title) >= sim for c in chosen):
                continue
            chosen.append(title)
            keep.append(i)
    out = a.loc[keep].copy()
    out["key"] = out["id"].map(article_key)
    return out[cols].reset_index(drop=True)


def article_key(article_id: str) -> str:
    """원장 쌍 키 — 다른 판정과 같은 ``uuid5(기사 id)|MARKET``."""
    return f"{pair_key_article(article_id)}|{CODE}"


# ---------------------------------------------------------------------------
# 요청·파싱 (순수)
# ---------------------------------------------------------------------------

def user_prompt(article: dict) -> str:
    pub = article["published_at"]
    if isinstance(pub, (pd.Timestamp, dt.datetime)):
        pub = pd.Timestamp(pub).tz_convert(KST).strftime("%Y-%m-%d %H:%M") if pd.Timestamp(pub).tzinfo else \
            pd.Timestamp(pub).strftime("%Y-%m-%d %H:%M")
    return USER_TEMPLATE.format(published=pub, title=str(article["title"]).strip(),
                                summary=str(article.get("summary") or "").strip() or "(없음)")


def build_request(article: dict, *, cache_system: bool = True) -> dict:
    """한 기사의 요청(배치 한 줄 모양 ``{custom_id, params}`` — 비배치는 ``params`` 만 보낸다)."""
    return {
        "custom_id": custom_id_of(article["key"]),
        "params": {
            "model": MODEL,
            "system": ([{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
                       if cache_system else SYSTEM),
            "messages": [{"role": "user", "content": user_prompt(article)}],
            **SPEC.params,
        },
    }


def input_hash(article: dict) -> str:
    """(모델, 평문 system, user) 해시 — 캐싱 블록이든 평문이든 같은 값."""
    return judgment_input_hash(MODEL, SYSTEM, user_prompt(article))


def parse_mood(raw: "str | dict") -> dict:
    """응답 텍스트 → 5필드 dict. 필드 누락·범위 밖·정수 아님이면 :class:`SchemaError`."""
    obj = raw if isinstance(raw, dict) else _extract_json_object(raw)
    out = {}
    for name, allowed in FIELDS.items():
        if name not in obj:
            raise SchemaError(f"missing field {name}")
        v = obj[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            raise SchemaError(f"{name} must be an integer, got {v!r}")
        if int(v) not in allowed:
            raise SchemaError(f"{name} out of range: {v!r}")
        out[name] = int(v)
    return out


# ---------------------------------------------------------------------------
# 비용
# ---------------------------------------------------------------------------

def usage_usd(usage: dict, *, batch: bool) -> float:
    """실측 usage → 달러(캐시 쓰기 1.25×·읽기 0.1×, 배치면 50% 할인)."""
    in_usd, out_usd = prices_for(MODEL)
    billed_in = (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) * CACHE_WRITE_MULT
                 + usage.get("cache_read_input_tokens", 0) * CACHE_READ_MULT)
    usd = (billed_in * in_usd + usage.get("output_tokens", 0) * out_usd) / 1e6
    return usd * BATCH_DISCOUNT if batch else usd


def estimate_usd(articles: Iterable[dict], *, batch: bool, cache_hit_rate: float = 1.0,
                 chars_per_token: float = EST_CHARS_PER_TOKEN, output_tokens: int = EST_OUTPUT_TOKENS) -> dict:
    """**추정** 비용(글자수 ÷ chars_per_token, system 은 첫 요청 쓰기·나머지 ``cache_hit_rate`` 만큼 읽기). 실측은 :func:`usage_usd`."""
    users = [len(user_prompt(a)) for a in articles]
    n = len(users)
    in_usd, out_usd = prices_for(MODEL)
    sys_tok = len(SYSTEM) / chars_per_token
    user_tok = sum(users) / chars_per_token
    if n:
        rest = n - 1
        sys_billed = sys_tok * (CACHE_WRITE_MULT * (1 + rest * (1 - cache_hit_rate)) + CACHE_READ_MULT * rest * cache_hit_rate)
    else:
        sys_billed = 0.0
    usd = ((user_tok + sys_billed) * in_usd + n * output_tokens * out_usd) / 1e6
    if batch:
        usd *= BATCH_DISCOUNT
    return {"requests": n, "input_tokens": round(user_tok + n * sys_tok), "output_tokens": n * output_tokens,
            "usd": usd, "is_estimate": True}


def add_usage(total: dict, usage: "dict | None") -> None:
    for k in USAGE_FIELDS:
        total[k] = total.get(k, 0) + int((usage or {}).get(k) or 0)


# ---------------------------------------------------------------------------
# 비배치 판정 (파일럿)
# ---------------------------------------------------------------------------

def messages_judge(api_key: str, *, timeout: float = REQUEST_TIMEOUT, cache_system: bool = True) -> JudgeFn:
    """비배치 ``/v1/messages`` 로 한 기사를 판정 → ``(output | None, error | None, usage | None)``."""
    import httpx  # noqa: PLC0415

    client = httpx.Client(timeout=timeout)
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}

    def judge(article: dict) -> tuple[dict | None, str | None, dict | None]:
        params = build_request(article, cache_system=cache_system)["params"]
        for attempt in range(3):
            try:
                resp = client.post(f"{API}/v1/messages", headers=headers, json=params)
            except httpx.HTTPError as exc:
                return None, f"transport: {type(exc).__name__}", None
            if resp.status_code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(5.0 * (attempt + 1))
                continue
            break
        if resp.status_code != 200:
            return None, f"http {resp.status_code}: {resp.text[:200]}", None
        body = resp.json()
        usage = body.get("usage")
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        try:
            return parse_mood(text), None, usage
        except SchemaError as exc:
            return None, f"schema: {exc}", usage

    return judge


class CostCapExceeded(RuntimeError):
    pass


def judge_articles(selected: pd.DataFrame, judge_fn: JudgeFn, ledger: "JudgmentLedger | None", *,
                   budget_usd: float, hard_cap_usd: float, already: "dict[str, dict] | None" = None,
                   flush_every: int = 25, now: "Callable[[], dt.datetime] | None" = None,
                   log: Callable[[str], None] = print) -> tuple[dict[str, dict], dict]:
    """비배치 판정 루프 → (키 → output, 정보). ``already`` 의 키는 건너뛰고 그 output 을 쓴다.

    추정이 ``hard_cap_usd`` 를 넘으면 아무것도 보내지 않고 :class:`CostCapExceeded`. 실측 누계가 ``budget_usd`` 를 넘으면 멈춘다
    (나머지는 미판정). ``flush_every`` 건마다 원장에 쓴다 — 중간에 죽어도 쓴 만큼은 남는다. 원장 쓰기 실패는 예외로 올린다."""
    now = now or (lambda: dt.datetime.now(dt.UTC))
    outputs: dict[str, dict] = dict(already or {})
    todo = [r for r in selected.to_dict("records") if r["key"] not in outputs]
    est = estimate_usd(todo, batch=False)
    info = {"n_selected": len(selected), "n_reused": len(selected) - len(todo), "n_todo": len(todo), "estimate": est,
            "n_judged": 0, "n_ok": 0, "n_error": 0, "n_unjudged": 0, "usage": {}, "measured_usd": 0.0,
            "stopped": None, "ledger": {"inserted": 0, "updated": 0, "unchanged": 0}}
    log(f"selected {len(selected)}, reused {info['n_reused']}, to judge {len(todo)}; "
        f"estimate ${est['usd']:.4f} (hard cap ${hard_cap_usd:.2f}, budget ${budget_usd:.2f})")
    if est["usd"] > hard_cap_usd:
        raise CostCapExceeded(f"estimate ${est['usd']:.4f} > hard cap ${hard_cap_usd:.2f} — nothing sent")
    entries: list[dict] = []

    def flush() -> None:
        if ledger is not None and entries:
            counts = ledger.record(entries)
            for k, v in counts.items():
                info["ledger"][k] += v
        entries.clear()

    for i, art in enumerate(todo):
        if info["measured_usd"] > budget_usd:
            info["stopped"] = f"measured ${info['measured_usd']:.4f} > budget ${budget_usd:.2f}"
            info["n_unjudged"] = len(todo) - i
            log(info["stopped"])
            break
        output, error, usage = judge_fn(art)
        info["n_judged"] += 1
        add_usage(info["usage"], usage)
        info["measured_usd"] = usage_usd(info["usage"], batch=False)
        if output is not None:
            outputs[art["key"]] = output
            info["n_ok"] += 1
        else:
            info["n_error"] += 1
        entries.append({"key": art["key"], "input_hash": input_hash(art), "output": output, "error": error,
                        "judged_at": now(), "judged_at_exact": True, "batch_id": None, "usage": usage})
        if len(entries) >= flush_every:
            flush()
    flush()
    return outputs, info


# ---------------------------------------------------------------------------
# 배치 판정 (전 구간) — BatchJudge 의 제출·대기·상태를 물려받고 요청·파싱만 바꾼다
# ---------------------------------------------------------------------------

class MarketBatchJudge(BatchJudge):
    """시황 기사용 배치 판정기. ``run_articles`` 가 입구 — 상태 파일(``<run>.state.json``)로 재개 가능.

    부모의 ``run``/``collect`` 는 종목 쌍 전용이라 쓰지 않는다(``submit``·``wait``·``_http``·상태 파일·usage 로그만 재사용)."""

    _ledger_counts: "dict[str, int]" = {}

    def run_articles(self, selected: pd.DataFrame, *, hard_cap_usd: float, already: "dict[str, dict] | None" = None,
                     dry_run: bool = False, cache_hit_rate: float = 0.5) -> tuple[dict[str, dict], dict]:
        outputs: dict[str, dict] = dict(already or {})
        state = self._load_state()
        pending = [b for b in state.get("batches", []) if not b["processed"]]
        todo = [r for r in selected.to_dict("records") if r["key"] not in outputs]
        est = estimate_usd(todo, batch=True, cache_hit_rate=cache_hit_rate)
        info = {"n_selected": len(selected), "n_reused": len(selected) - len(todo), "n_todo": len(todo), "estimate": est,
                "submitted": 0, "n_ok": 0, "n_error": 0, "usage": {}, "measured_usd": 0.0, "batches": [],
                "ledger": {"inserted": 0, "updated": 0, "unchanged": 0}}
        self._ledger_counts = info["ledger"]
        self.log(f"selected {len(selected)}, reused {info['n_reused']}, to judge {len(todo)}; "
                 f"estimate ${est['usd']:.4f} (batch, hard cap ${hard_cap_usd:.2f}); pending batches {len(pending)}")
        if pending:
            # 재개: 이미 제출된 배치부터 거둔다(제출된 것은 과금됐다)
            for entry in pending:
                outs, ok, bad = self.collect_articles(entry)
                outputs.update(outs)
                info["n_ok"] += ok
                info["n_error"] += bad
            return outputs, self._bank(info)
        if est["usd"] > hard_cap_usd:
            raise CostCapExceeded(f"estimate ${est['usd']:.4f} > hard cap ${hard_cap_usd:.2f} — nothing sent")
        if dry_run or not todo:
            return outputs, info
        by_key = {r["key"]: r for r in todo}
        keys = list(by_key)
        chunks = [keys[i:i + self.chunk] for i in range(0, len(keys), self.chunk)]
        state = self._load_state()
        state["run"] = {"model": self.model, "estimate": est, "hard_cap_usd": hard_cap_usd, "n": len(keys),
                        "created_at": dt.datetime.now(dt.UTC).isoformat(), "judge": JUDGE}
        state.setdefault("batches", [])
        self._save_state(state)
        for i, chunk_keys in enumerate(chunks):
            reqs = [build_request(by_key[k], cache_system=self.cache_system) for k in chunk_keys]
            self._resolve_uncertain(i)
            self.submit(reqs, {custom_id_of(k): k for k in chunk_keys}, i)
            info["submitted"] += 1
        for entry in [b for b in self._load_state()["batches"] if not b["processed"]]:
            outs, ok, bad = self.collect_articles(entry)
            outputs.update(outs)
            info["n_ok"] += ok
            info["n_error"] += bad
        return outputs, self._bank(info)

    def _bank(self, info: dict) -> dict:
        state = self._load_state()
        for b in state.get("batches", []):
            if b.get("processed") and b.get("usage"):
                add_usage(info["usage"], b["usage"])
                info["batches"].append({"id": b["id"], "requests": len(b["custom_ids"]), "measured_usd": b.get("measured_usd")})
        info["measured_usd"] = usage_usd(info["usage"], batch=True)
        return info

    def collect_articles(self, entry: dict) -> tuple[dict[str, dict], int, int]:
        """배치 하나를 끝까지 기다려 결과를 파싱, **원장에 먼저** 쓰고 상태를 갱신 → (키 → output, 성공, 실패)."""
        batch = self.wait(entry["id"])
        text = self._http("GET", batch["results_url"]).text
        ended = batch.get("ended_at")
        judged_at = dt.datetime.fromisoformat(str(ended).replace("Z", "+00:00")) if ended else dt.datetime.now(dt.UTC)
        keys, hashes = entry["custom_ids"], entry.get("input_hashes") or {}
        outputs: dict[str, dict] = {}
        ledger_rows: list[dict] = []
        usage_rows: list[dict] = []
        usage: dict = {}
        seen: set[str] = set()
        ok = bad = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["custom_id"]
            key = keys.get(cid)
            if key is None:
                continue
            seen.add(cid)
            result = row.get("result") or {}
            u = (result.get("message") or {}).get("usage")
            if u:
                add_usage(usage, u)
                usage_rows.append({"key": key, "batch_id": entry["id"], "result": result.get("type"),
                                   "usage": {k: int(u.get(k) or 0) for k in USAGE_FIELDS}})
            output = error = None
            if result.get("type") != "succeeded":
                error = f"{result.get('type') or 'unknown'}: {json.dumps(result.get('error'), ensure_ascii=False)}"
            else:
                blocks = (result.get("message") or {}).get("content") or []
                content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                try:
                    output = parse_mood(content)
                except SchemaError as exc:
                    error = f"schema: {exc}"
            if output is not None:
                outputs[key] = output
                ok += 1
            else:
                bad += 1
            if (h := hashes.get(cid)) is not None:
                ledger_rows.append({"key": key, "input_hash": h, "output": output, "error": error, "batch_id": entry["id"],
                                    "judged_at": judged_at, "judged_at_exact": True, "usage": u})
        for cid in set(keys) - seen:
            bad += 1
            if (h := hashes.get(cid)) is not None:
                ledger_rows.append({"key": keys[cid], "input_hash": h, "output": None, "error": "missing_result: null",
                                    "batch_id": entry["id"], "judged_at": judged_at, "judged_at_exact": True, "usage": None})
        if self.ledger is not None:
            counts = self.ledger.record(ledger_rows)  # 실패하면 예외 — 배치는 미처리로 남는다
            for k, v in counts.items():
                self._ledger_counts[k] = self._ledger_counts.get(k, 0) + v
            self.log(f"batch {entry['id']}: ledger {counts}")
        for r in usage_rows:
            self._append(self.usage_log, r)
        measured = usage_usd(usage, batch=True)
        state = self._load_state()
        for b in state["batches"]:
            if b["id"] == entry["id"]:
                b["processed"] = True
                b["usage"] = usage
                b["measured_usd"] = measured
        self._save_state(state)
        self.log(f"batch {entry['id']}: {ok} ok, {bad} failed; billed {usage} = ${measured:.4f} measured")
        return outputs, ok, bad


# ---------------------------------------------------------------------------
# 집계 (순수)
# ---------------------------------------------------------------------------

def daily_mood(selected: pd.DataFrame, outputs: dict[str, dict]) -> pd.DataFrame:
    """신호일별 집계 — ``n_selected``·``n_judged`` 와 5필드 평균(판정 성공분만). index = signal_day(Timestamp)."""
    rows = []
    for key, day in zip(selected["key"], selected["signal_day"]):
        o = outputs.get(key)
        rows.append({"signal_day": pd.Timestamp(day), "judged": o is not None, **{f: (o or {}).get(f) for f in FIELDS}})
    df = pd.DataFrame(rows, columns=["signal_day", "judged", *FIELDS])
    if df.empty:
        return pd.DataFrame(columns=["n_selected", "n_judged", *FIELDS]).rename_axis("signal_day")
    g = df.groupby("signal_day")
    out = pd.DataFrame({"n_selected": g.size(), "n_judged": g["judged"].sum().astype(int)})
    for f in FIELDS:
        out[f] = g[f].mean()
    return out


def mood_score(daily: pd.DataFrame) -> pd.Series:
    """기술용 합성 점수(설계에 없음 — 가중치 1·1·1·1·−1 은 임의): 방향 + (위험선호 − 1) + 외국인 + 폭 − 이벤트."""
    return (daily["market_direction"] + (daily["risk_appetite"] - 1.0) + daily["foreign_flow_tone"]
            + daily["breadth_tone"] - daily["event_risk"])


def mood_index(daily: pd.DataFrame, trading_days: Sequence[dt.date], *, span: int = 20) -> pd.DataFrame:
    """거래일 달력 위 일별 평균과 ``span`` 일 지수이동평균(``*_ema20``) — 기사 없는 날은 EMA 가 전날 값을 유지(인과)."""
    idx = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in trading_days))
    d = daily.reindex(idx)
    out = d[["n_selected", "n_judged", *FIELDS]].copy()
    out["n_selected"] = out["n_selected"].fillna(0).astype(int)
    out["n_judged"] = out["n_judged"].fillna(0).astype(int)
    out["mood_score"] = mood_score(d)
    for f in [*FIELDS, "mood_score"]:
        out[f"{f}_ema{span}"] = out[f].ewm(span=span, adjust=False, ignore_na=True).mean().ffill()
    return out.rename_axis("date")


# ---------------------------------------------------------------------------
# DB (읽기: 기사 · 쓰기: 원장만)
# ---------------------------------------------------------------------------

_SQL_ARTICLES = (
    "SELECT DISTINCT ON (id) id, published_at, title, summary FROM news_articles "
    "WHERE published_at >= %s AND published_at < %s "
    "AND (title ILIKE '%%코스피%%' OR title ILIKE '%%코스닥%%' OR title ILIKE '%%증시%%' OR title ILIKE '%%마감%%' "
    "OR title ILIKE '%%외국인%%') ORDER BY id, collected_at"
)


def load_market_articles(con, lo: dt.datetime, hi: dt.datetime) -> pd.DataFrame:
    """제목 패턴(DB 단 1차 거름)에 맞는 기사 — ``id``·``published_at``·``title``·``summary``. 같은 id 가 두 판이면 먼저 수집된 판."""
    cur = con.cursor()
    cur.execute(_SQL_ARTICLES, (lo, hi))
    df = pd.DataFrame(cur.fetchall(), columns=["id", "published_at", "title", "summary"])
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def ledger_for(con, article_ids: Iterable[str]) -> JudgmentLedger:
    """선택 기사 id 로 역사전을 만든 원장 창구 — 시황 기사는 종목 피드에 없을 수 있어 ``key_map`` 을 쓰지 않는다."""
    return JudgmentLedger(con, SPEC, kmap={pair_key_article(a): a for a in article_ids})


def existing_outputs(con) -> dict[str, dict]:
    """원장의 ``market-mood-haiku55`` 성공 행 → {키: output} (재판정 건너뛰기)."""
    rows = read_judgments(con, JUDGE)
    return {k: r["output"] for k, r in rows.items() if r.get("output") is not None}


def register(con) -> bool:
    """judge 정의를 ``judges`` 에 등록(이미 같은 정의면 no-op, 다른 정의면 :class:`~swing_it.news.judgments_db.JudgeCollision`)."""
    return register_judge(con, SPEC)


def api_key() -> "str | None":
    from swing_it.news.judge import _api_key  # noqa: PLC0415

    return _api_key()


def run_dir() -> Path:
    from swing_it.news.judge import RUNS  # noqa: PLC0415

    return RUNS / JUDGE
