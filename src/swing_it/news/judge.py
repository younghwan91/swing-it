"""(기사, 종목) 쌍을 Message Batches API 로 판정한다 — daytrade-it ``scripts/eval/judge_batch.py`` 이식.

    # 비용 추정만(API 호출 없음)
    python -m swing_it.news.judge --pairs data/eval/persistence_swing/pairs/swing_2023-03_2026-02.jsonl \\
        --days 2025-03-01..2025-08-31 --sample 29932 --seed 1 --budget-usd 8 --run stage2 --dry-run
    # 제출·폴링·캐시 기록(재개 가능: 죽으면 같은 명령을 다시 돌린다)
    python -m swing_it.news.judge ... (같은 인자, --dry-run 없이)

**출처:** daytrade-it 브랜치 ``research/news-persistence-judge`` @ ba0662a. 안전 동작은 그대로다 —
예산 가드, 재개 가능한 상태 파일, 배치 생성은 절대 재시도하지 않음(5xx·전송 오류 → ``SubmitUncertain``,
``--adopt-batch``), 스키마 오류 파일, 입력 해시로 캐시 행 검증, ``MAX_ATTEMPTS``. raw httpx(SDK 없음).
기본값은 스윙 2호 판정 그대로다: 모델 ``claude-haiku-5-5``(``temperature`` 를 받지 않아 빼고,
``thinking: disabled`` — 운영 v2 를 판정한 Haiku 4.5 는 생각하지 않았다), 운영 v2 프롬프트 +
사전등록 보충 ``research/logs/news_persistence_swing/prompt_r3.txt``.

**판정의 정본은 DB 원장이다(2026-10-09).** 배치를 수집할 때마다 결과를 quant-airflow ``article_judgments`` 에
judge 이름(:mod:`swing_it.news.judgments_db` — 모델·system·파라미터로 정해진다, 기본 = ``v2r3-haiku55``)으로 **먼저**
쓰고(실측 usage 토큰, judged_at = 배치 종료 시각·exact TRUE), 그다음 JSONL 작업 상태를 붙인다. DB 쓰기가 실패하면
배치를 처리 완료로 표시하지 않고 실패한다. 작업 상태 파일: ``data/eval/judge_runs/<judge>/`` 의 ``judgments.jsonl``·
``errors.jsonl`` (judge 단위 — 재판정 건너뛰기)과 ``<run>.state.json``·``<run>.batches.log``·``<run>.usage.jsonl``.

**이식하며 바뀐 것(2026-10-08):**

- **프롬프트 캐싱.** ``system`` 을 ``[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]``
  블록으로 보낸다. Haiku 5.5 실측: 보충 포함 system ≈1,044 토큰이 캐시된다(두 번째 요청 cache_read 1,044),
  user 턴(~1,030 토큰)은 캐시되지 않는다. 캐시 행의 ``input_hash`` 는 **평문 system 문자열**로 계산한다 —
  보내는 모양만 바뀌고 모델 입력은 같으므로 기존 판정 행이 그대로 유효하다(``--no-cache-system`` 으로 끈다).
- **비용은 추정이 아니라 측정한다.** 결과마다 실제 과금 usage(input / cache_creation / cache_read / output 토큰)를
  ``<state>.usage.jsonl`` 에 남기고(스키마 오류 응답도 과금되므로 포함), 배치별 합계·측정 금액을 상태 파일과
  요약에 싣는다. 1단계의 "$6.26" 은 글자수/토큰 추정이었고 실제 출력은 쌍당 ~97 토큰(가정 200)이었다.
- :func:`estimate_cost` 는 **여전히 추정**이다: 입력 = 글자수 ÷ ``--chars-per-token``(기본 1.0, 실측 ~1.3 이라
  ~30% 보수적), 출력 = ``--est-output-tokens``, 캐시되는 system 토큰은 첫 요청 1.25×(쓰기)·나머지
  ``--cache-hit-rate`` 비율만큼 0.1×(읽기). 배치 할인 50%.
- 판정 전 필터는 ``--prefilter`` 로만 켠다(기본 끔 — 쌍 생성기 :mod:`swing_it.news.pairs` 가 이미 거른다;
  1단계 실행도 ``prefiltered 0``). 켜면 :class:`~swing_it.news.prefilter.PairFilter` 파라미터를 받는다.

나머지 동작(daytrade-it 원문):

- Pairs already in the cache (rows ``{"key", "input_hash", "v2"}`` with v2 = asdict(features) + "score") are
  skipped only if the row's input_hash = sha256 of (model, system, user) of the request this pair would send
  now (``cache_row_valid``; legacy rows without a hash count only if the pair's input is unchanged by the body
  fallback); so are pairs that already failed in ``MAX_ATTEMPTS`` (3) distinct batches (errors file).
- Stratified sample: proportional allocation over the ``--stratify`` fields (largest remainder), seeded
  shuffle within each stratum -- deterministic for a given (pairs set, sample, seed) regardless of file order.
- A batch create is never retried: a 5xx/transport error records the attempt and fails loudly
  (``SubmitUncertain``); a rerun resubmits that chunk only if no unrecorded batch of that size is listed,
  else ``--adopt-batch ID`` records it.
- ``--max-submit N`` submits only N chunks per invocation and collects them (~100k requests-in-flight cap).
- Resumable, no double spend: the chosen sample (keys, chunking, seed, args) is saved to the state file
  before the first submit; each batch id is appended to ``<state>.batches.log`` as soon as the create call
  returns, then recorded in the state. A completed run blocks drawing a fresh sample unless ``--new-run``.
- Key from ``ANTHROPIC_API_KEY`` or ``../quant-airflow/.env``. 키는 어디에도 찍지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import httpx

from swing_it.news.judgments_db import JudgmentLedger, connect_db, spec_for_request
from swing_it.news.prefilter import PairFilter, add_filter_args, filter_from_args, keep_pair
from swing_it.news.prompt import (
    _MAX_OUTPUT_TOKENS,
    _SYSTEM,
    SchemaError,
    build_prompt,
    judgment_input_hash,
    parse_features,
    read_addendum,
    system_with_addendum,
)

API = "https://api.anthropic.com"
REPO = Path(__file__).resolve().parents[3]
#: 판정 작업 상태(정본은 DB 원장): judge_runs/<judge>/{judgments.jsonl, errors.jsonl, <run>.state.json,
#: <run>.batches.log, <run>.usage.jsonl}. judgments/errors 는 judge 단위(재판정 건너뛰기), 나머지는 런 단위.
RUNS = Path("data/eval/judge_runs")
#: 사전등록(수정 1)이 고정한 보충. data/eval/persistence_swing/prompts/r3.txt 와 바이트 동일.
ADDENDUM = REPO / "research" / "logs" / "news_persistence_swing" / "prompt_r3.txt"
MODEL = "claude-haiku-5-5"

INPUT_USD_PER_M = 1.0  # Haiku 4.5
OUTPUT_USD_PER_M = 5.0
#: (input, output) $/M by model (standard; the Batch API halves both) — Haiku 5.5 (2026-10-07) is 1/10 of 4.5.
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
#: Models that reject `temperature` (400 "`temperature` is deprecated for this model").
NO_TEMPERATURE = {"claude-haiku-5-5"}
BATCH_DISCOUNT = 0.5
#: Prompt-cache multipliers on the input price: 5-minute write, read.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1
MAX_ATTEMPTS = 3  # same as live score_article (_MAX_RETRIES + 1)
CHUNK = 10_000  # requests per batch (API limit 100k / 256 MB)
USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


class BudgetExceeded(RuntimeError):
    pass


class SubmitUncertain(RuntimeError):
    """A batch create may or may not have happened (5xx / transport error): never retried."""


#: How far before a failed create a listed batch may claim to be created (clock skew).
UNCERTAIN_SKEW = dt.timedelta(minutes=10)

#: A 429 on a batch create is a rejection (nothing queued, nothing billed), so -- unlike a
#: 5xx -- it is safe to retry: the concurrent-batch quota frees as earlier batches end.
CREATE_429_DELAY = 60.0
CREATE_429_MAX_DELAY = 600.0
CREATE_429_RETRIES = 60

#: Per-request `errored` results that say nothing about the pair: the request was
#: rejected by the account, not judged (no credit, bad key, revoked permission).
#: They must not burn one of MAX_ATTEMPTS -- unlike `overloaded_error`/`api_error`,
#: which mean the model was actually asked.
NON_ATTEMPT_ERROR_TYPES = frozenset(
    {
        "invalid_request_error",  # includes "Your credit balance is too low"
        "authentication_error",
        "permission_error",
        "billing_error",
    }
)


def counts_as_attempt(row: dict) -> bool:
    """Whether an error row means this pair really got a judgement attempt."""
    if row.get("reason") != "errored":
        return True  # schema / missing_result: the model did answer (or the batch did)
    detail = row.get("detail")
    error = detail.get("error") if isinstance(detail, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else None
    return error_type not in NON_ATTEMPT_ERROR_TYPES


def retry_after_seconds(resp: httpx.Response, default: float) -> float:
    """`retry-after` in seconds, or `default` when the header is absent or unparsable."""
    try:
        return max(float(resp.headers.get("retry-after")), 1.0)
    except (TypeError, ValueError):
        return default


def custom_id_of(key: str) -> str:
    """`uuid|code` -> API-safe custom_id (^[a-zA-Z0-9_-]{1,64}$)."""
    return key.replace("|", "_")


def build_request(pair: dict, model: str, addendum: str | None = None, cache_system: bool = True) -> dict:
    """한 쌍의 배치 요청. system = 운영 v2 ``_SYSTEM`` (+ ``"\\n\\n" + addendum``).

    ``cache_system`` 이면 system 을 ``cache_control: ephemeral`` 블록 하나로 보낸다 — 모델이 읽는 텍스트는
    평문 문자열과 같다(입력 해시도 평문으로 계산한다, :func:`request_input_hash`).
    """
    system, user = build_prompt(
        code=pair["code"],
        name=pair["name"],
        market=pair.get("market"),
        sector=pair.get("sector"),
        title=pair["title"],
        body=pair.get("body") or "",
        category=pair.get("category"),
    )
    system = system_with_addendum(system, addendum)
    return {
        "custom_id": custom_id_of(pair["key"]),
        "params": {
            "model": model,
            "system": (
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if cache_system
                else system
            ),
            "messages": [{"role": "user", "content": user}],
            **request_params(model),
        },
    }


def request_params(model: str) -> dict:
    """모델 입력 외 판정 파라미터 — judge 정의(``judges.params``)의 일부."""
    return {
        "max_tokens": _MAX_OUTPUT_TOKENS,
        **({} if model in NO_TEMPERATURE else {"temperature": 0}),
        # Haiku 5.5 thinks by default and the 400-token cap went entirely to thinking (pilot 2026-10-08:
        # 320/500 empty, 90 truncated). The live v2 judge (Haiku 4.5) never thought — disable to replicate it.
        **({"thinking": {"type": "disabled"}} if model in NO_TEMPERATURE else {}),
    }


def judge_spec(model: str, addendum: str | None):
    """(모델, 보충) → 등록된 judge 정의. 맞는 이름이 없으면 :class:`JudgeCollision`."""
    return spec_for_request(model, system_with_addendum(_SYSTEM, addendum), request_params(model))


def system_text(params: dict) -> str:
    """요청의 system 을 평문으로 — 문자열이든 텍스트 블록 목록이든."""
    system = params["system"]
    if isinstance(system, str):
        return system
    return "".join(b["text"] for b in system if b.get("type") == "text")


def request_input_hash(req: dict) -> str:
    """(model, 평문 system, user) 해시 — 캐싱 블록이든 평문이든 같은 값(기존 캐시 행과 호환)."""
    p = req["params"]
    return judgment_input_hash(p["model"], system_text(p), p["messages"][0]["content"])


def pair_input_hash(pair: dict, model: str, addendum: str | None = None) -> str:
    """Hash of the exact input `build_request` would send for this pair."""
    return request_input_hash(build_request(pair, model, addendum))


def cache_row_valid(row: dict, pair: dict, model: str, addendum: str | None = None) -> bool:
    """Whether a cache row judged this pair's current input.

    Rows carry `input_hash`. Legacy rows (written before hashing) are valid only
    if the pair's current input hash equals the hash of the input the old
    pipeline sent -- body = the raw C1 summary (no title fallback) -- i.e. the
    pair's input didn't change.

    An **empty row is never valid**: callers that look a key up with
    ``cache.get(key, {})`` would otherwise fall into the legacy branch, where
    the absent row hashes to the pair's own current input and every unjudged
    pair reports as judged.
    """
    if not row:
        return False
    current = pair_input_hash(pair, model, addendum)
    if "input_hash" in row:
        return row["input_hash"] == current
    legacy = {**pair, "body": pair["summary"]} if "summary" in pair else pair
    return pair_input_hash(legacy, model, addendum) == current


def _chars(req: dict) -> int:
    p = req["params"]
    return len(system_text(p)) + sum(len(m["content"]) for m in p["messages"])


def _cached_system(req: dict) -> bool:
    system = req["params"]["system"]
    return not isinstance(system, str) and any("cache_control" in b for b in system)


def prices_for(model: str) -> tuple[float, float]:
    """(input, output) 표준 $/M — 배치 할인 전."""
    return PRICES.get(model, (INPUT_USD_PER_M, OUTPUT_USD_PER_M))


def estimate_cost(
    requests: list[dict], chars_per_token: float, output_tokens: int, cache_hit_rate: float = 1.0
) -> dict:
    """**추정** 비용(배치 할인 포함). 실제 과금은 :func:`usage_cost` 로 결과의 usage 에서 잰다.

    입력 토큰 = 글자수 ÷ ``chars_per_token``. system 을 캐싱하는 요청이면 system 토큰을 따로 떼어,
    첫 요청은 캐시 쓰기(1.25×), 나머지는 ``cache_hit_rate`` 만큼 읽기(0.1×)·나머지 쓰기로 친다. 배치는 동시에
    처리돼 캐시 적중이 보장되지 않는다 — 적중률을 낮춰 잡으면 보수적이다. (system 이 모델의 최소 캐시 길이보다
    짧으면 실제로는 캐시되지 않는다 — 운영 v2 평문 system 만으로는 짧다.)
    """
    model = requests[0]["params"]["model"] if requests else ""
    in_usd, out_usd = prices_for(model)
    total = round(sum(_chars(r) for r in requests) / chars_per_token)
    cached = [r for r in requests if _cached_system(r)]
    sys_tokens = round(sum(len(system_text(r["params"])) for r in cached) / chars_per_token)
    uncached = total - sys_tokens
    if cached:
        per = sys_tokens / len(cached)
        rest = len(cached) - 1
        writes = per * (1 + rest * (1 - cache_hit_rate))
        reads = per * rest * cache_hit_rate
    else:
        writes = reads = 0.0
    out = len(requests) * output_tokens
    usd = (
        (uncached + writes * CACHE_WRITE_MULT + reads * CACHE_READ_MULT) * in_usd + out * out_usd
    ) / 1e6 * BATCH_DISCOUNT
    return {
        "requests": len(requests),
        "input_tokens": total,
        "cached_system_tokens": sys_tokens,
        "output_tokens": out,
        "usd": usd,
        "is_estimate": True,
    }


def add_usage(total: dict, usage: dict | None) -> None:
    for k in USAGE_FIELDS:
        total[k] = total.get(k, 0) + int((usage or {}).get(k) or 0)


def usage_cost(usage: dict, model: str) -> float:
    """측정 비용(배치 할인 포함) — 결과 usage 의 토큰 × 모델 가격, 캐시 쓰기 1.25×·읽기 0.1×."""
    in_usd, out_usd = prices_for(model)
    billed_in = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0) * CACHE_WRITE_MULT
        + usage.get("cache_read_input_tokens", 0) * CACHE_READ_MULT
    )
    return (billed_in * in_usd + usage.get("output_tokens", 0) * out_usd) / 1e6 * BATCH_DISCOUNT


def select_days(pairs: list[dict], spec: str) -> list[dict]:
    """The pairs whose ``day`` is in ``spec``: ``A..B`` (inclusive) or a comma list.

    A7 needs *whole* days judged: the stratified sample spreads a partial budget
    evenly over every day, so a wave that stops early leaves no day complete and
    no day usable for the universe backtest. Restricting the pairs file to a few
    days first makes those days finish. File order is kept, so the downstream
    sample stays deterministic. A row without ``day`` raises ``KeyError``.
    """
    if ".." in spec:
        lo, hi = (dt.date.fromisoformat(x) for x in spec.split("..", 1))

        def wanted(day: dt.date) -> bool:
            return lo <= day <= hi
    else:
        days = {dt.date.fromisoformat(x) for x in spec.split(",") if x}

        def wanted(day: dt.date) -> bool:
            return day in days

    return [p for p in pairs if wanted(dt.date.fromisoformat(p["day"]))]


def _stratum(pair: dict, fields: list[str]) -> tuple[str, ...]:
    return tuple(str(pair.get(f)) for f in fields)


def stratified_sample(pairs: list[dict], n: int, fields: list[str], seed: int) -> list[dict]:
    """Proportional stratified sample, sorted by key. All pairs if n >= len(pairs)."""
    ordered = sorted(pairs, key=lambda p: p["key"])
    if n >= len(ordered):
        return ordered
    strata: dict[tuple[str, ...], list[dict]] = {}
    for p in ordered:
        strata.setdefault(_stratum(p, fields), []).append(p)
    total = len(ordered)
    quotas = {s: n * len(ps) / total for s, ps in strata.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    rest = n - sum(alloc.values())
    for s in sorted(strata, key=lambda s: (-(quotas[s] - alloc[s]), s))[:rest]:
        alloc[s] += 1
    rng = random.Random(seed)
    picked = []
    for s in sorted(strata):
        picked += rng.sample(strata[s], alloc[s])
    return sorted(picked, key=lambda p: p["key"])


class BatchJudge:
    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str | None,
        model: str,
        cache: Path,
        errors: Path,
        state: Path,
        budget_usd: float,
        chars_per_token: float = 1.0,
        est_output_tokens: int = 200,
        chunk: int = CHUNK,
        addendum: str | None = None,
        cache_system: bool = True,
        cache_hit_rate: float = 1.0,
        prefilter: PairFilter | None = None,
        ledger: JudgmentLedger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.api_key = api_key
        self.model = model
        self.cache, self.errors, self.state = cache, errors, state
        self.budget_usd = budget_usd
        self.chars_per_token = chars_per_token
        self.est_output_tokens = est_output_tokens
        self.chunk = chunk
        self.addendum = addendum
        self.cache_system = cache_system
        self.cache_hit_rate = cache_hit_rate
        self.prefilter = prefilter
        self.ledger = ledger
        self.sleep = sleep
        self.log = log

    # --- local files ----------------------------------------------------------

    def _load_state(self) -> dict:
        if self.state.exists():
            return json.loads(self.state.read_text())
        return {"batches": []}

    def _save_state(self, state: dict) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state)

    def cached_rows(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        if self.cache.exists():
            for line in self.cache.open():
                if line.strip():
                    row = json.loads(line)
                    out.setdefault(row["key"], []).append(row)
        return out

    def is_cached(self, pair: dict, rows: dict[str, list[dict]]) -> bool:
        return any(cache_row_valid(r, pair, self.model, self.addendum) for r in rows.get(pair["key"], []))

    def _estimate(self, requests: list[dict]) -> dict:
        return estimate_cost(requests, self.chars_per_token, self.est_output_tokens, self.cache_hit_rate)

    def _sibling(self, suffix: str) -> Path:
        """상태 파일 옆 런 파일: ``<run>.state.json`` → ``<run><suffix>`` (옛 이름 ``x.json`` → ``x<suffix>``)."""
        name = self.state.name
        base = name[: -len(".state.json")] if name.endswith(".state.json") else self.state.stem
        return self.state.with_name(base + suffix)

    @property
    def usage_log(self) -> Path:
        """결과별 실제 과금 usage — ``{key, batch_id, result, usage}`` 한 줄씩."""
        return self._sibling(".usage.jsonl")

    def failure_counts(self) -> Counter[str]:
        if not self.errors.exists():
            return Counter()
        seen = {
            (r["key"], r.get("batch_id"))
            for line in self.errors.open()
            if line.strip() and counts_as_attempt(r := json.loads(line))
        }
        return Counter(key for key, _ in seen)

    def _append(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- HTTP -----------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _http(self, method: str, url: str, **kw) -> httpx.Response:
        delay = 5.0
        for attempt in range(6):
            resp = self.client.request(method, url, headers=self._headers(), timeout=120.0, **kw)
            if resp.status_code not in (429, 500, 502, 503, 529) or attempt == 5:
                resp.raise_for_status()
                return resp
            self.log(f"HTTP {resp.status_code} on {method} {url} -- retry in {delay:.0f}s")
            self.sleep(delay)
            delay = min(delay * 2, 300.0)
        raise AssertionError("unreachable")

    def count_tokens(self, req: dict) -> int:
        p = req["params"]
        body = {"model": p["model"], "system": p["system"], "messages": p["messages"]}
        return int(
            self._http("POST", f"{API}/v1/messages/count_tokens", json=body).json()["input_tokens"]
        )

    @property
    def batches_log(self) -> Path:
        return self._sibling(".batches.log")

    def submit(self, requests: list[dict], keys: dict[str, str], chunk: int) -> str:
        """Create one batch. Exactly one POST: a create is billable, so it is never retried.

        A 5xx or transport error leaves it unknown whether the batch exists: the
        attempt is recorded in the state (`uncertain_submits`) and in the
        batches log, and `SubmitUncertain` is raised. A rerun lists recent
        batches before submitting that chunk again (`_resolve_uncertain`).
        """
        attempted_at = dt.datetime.now(dt.UTC).isoformat()
        custom_ids = {r["custom_id"]: keys[r["custom_id"]] for r in requests}
        hashes = {r["custom_id"]: request_input_hash(r) for r in requests}
        delay = CREATE_429_DELAY
        for attempt in range(CREATE_429_RETRIES + 1):
            try:
                resp = self.client.request(
                    "POST",
                    f"{API}/v1/messages/batches",
                    headers=self._headers(),
                    timeout=120.0,
                    json={"requests": requests},
                )
            except httpx.TransportError as exc:
                self._uncertain(
                    chunk, custom_ids, hashes, attempted_at, f"{type(exc).__name__}: {exc}"
                )
                raise AssertionError("unreachable") from exc
            if resp.status_code >= 500:
                self._uncertain(chunk, custom_ids, hashes, attempted_at, f"HTTP {resp.status_code}")
            if resp.status_code != 429 or attempt == CREATE_429_RETRIES:
                break
            wait = retry_after_seconds(resp, delay)
            self.log(f"HTTP 429 creating chunk {chunk} -- retry in {wait:.0f}s")
            self.sleep(wait)
            delay = min(delay * 2, CREATE_429_MAX_DELAY)
        resp.raise_for_status()  # other 4xx: rejected, nothing was created
        batch = resp.json()
        self._record_batch(batch["id"], chunk, custom_ids, hashes)
        return batch["id"]

    def _record_batch(
        self, batch_id: str, chunk: int, custom_ids: dict[str, str], hashes: dict[str, str]
    ) -> None:
        now = dt.datetime.now(dt.UTC).isoformat()
        # the batch is billable from here: log its id before anything else can fail
        self._append(
            self.batches_log,
            {"id": batch_id, "chunk": chunk, "requests": len(custom_ids), "submitted_at": now},
        )
        state = self._load_state()
        state["batches"].append(
            {
                "id": batch_id,
                "chunk": chunk,
                "custom_ids": custom_ids,
                "input_hashes": hashes,
                "submitted_at": now,
                "processed": False,
            }
        )
        pending = [u for u in state.get("uncertain_submits", []) if u["chunk"] != chunk]
        if pending:
            state["uncertain_submits"] = pending
        else:
            state.pop("uncertain_submits", None)
        self._save_state(state)
        self.log(f"submitted batch {batch_id} chunk {chunk} ({len(custom_ids)} requests)")

    def _uncertain(
        self,
        chunk: int,
        custom_ids: dict[str, str],
        hashes: dict[str, str],
        attempted_at: str,
        error: str,
    ) -> None:
        entry = {
            "chunk": chunk,
            "requests": len(custom_ids),
            "attempted_at": attempted_at,
            "error": error,
            "custom_ids": custom_ids,
            "input_hashes": hashes,
        }
        self._append(
            self.batches_log,
            {
                "uncertain": True,
                **{k: v for k, v in entry.items() if k not in ("custom_ids", "input_hashes")},
            },
        )
        state = self._load_state()
        state["uncertain_submits"] = [
            u for u in state.get("uncertain_submits", []) if u["chunk"] != chunk
        ] + [entry]
        self._save_state(state)
        candidates = self._candidates(entry)
        raise SubmitUncertain(self._uncertain_message(entry, candidates))

    def _candidates(self, entry: dict) -> list[str] | None:
        """Unknown recent batches that could be this create (None if listing failed)."""
        try:
            listed = self._http("GET", f"{API}/v1/messages/batches", params={"limit": 100}).json()
        except (httpx.HTTPError, ValueError) as exc:
            self.log(f"listing batches failed: {exc}")
            return None
        known = {b["id"] for b in self._load_state().get("batches", [])}
        since = dt.datetime.fromisoformat(entry["attempted_at"]) - UNCERTAIN_SKEW
        out = []
        for b in listed.get("data") or []:
            try:
                created = dt.datetime.fromisoformat(str(b["created_at"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                created = None
            total = sum((b.get("request_counts") or {}).values())
            if b.get("id") in known or (created is not None and created < since):
                continue
            if total == entry["requests"]:
                out.append(b["id"])
        return out

    def _uncertain_message(self, entry: dict, candidates: list[str] | None) -> str:
        if candidates is None:
            found = "could not list batches"
        elif candidates:
            found = f"unrecorded batch(es) that may be it: {', '.join(candidates)}"
        else:
            found = "no matching batch listed"
        return (
            f"batch create for chunk {entry['chunk']} ({entry['requests']} requests) at "
            f"{entry['attempted_at']} failed with {entry['error']} and was NOT retried; {found}. "
            f"Check GET /v1/messages/batches and {self.batches_log}. If a batch was created, "
            "record it with --adopt-batch ID; otherwise rerun (it resubmits only if no "
            "matching batch is listed)."
        )

    def _resolve_uncertain(self, chunk: int) -> None:
        """Before resubmitting a chunk whose create was uncertain: refuse if a batch may exist."""
        entry = next(
            (u for u in self._load_state().get("uncertain_submits", []) if u["chunk"] == chunk),
            None,
        )
        if entry is None:
            return
        candidates = self._candidates(entry)
        if candidates is None or candidates:
            raise SubmitUncertain(self._uncertain_message(entry, candidates))
        self.log(f"chunk {chunk}: no batch from the failed create is listed -- resubmitting")

    def adopt_batch(self, batch_id: str) -> None:
        """Record an existing batch for the uncertain create it belongs to (matched by size)."""
        batch = self._http("GET", f"{API}/v1/messages/batches/{batch_id}").json()
        total = sum((batch.get("request_counts") or {}).values())
        state = self._load_state()
        if batch_id in {b["id"] for b in state.get("batches", [])}:
            raise ValueError(f"{batch_id} is already recorded")
        matches = [u for u in state.get("uncertain_submits", []) if u["requests"] == total]
        if len(matches) != 1:
            raise ValueError(
                f"{batch_id} has {total} requests: {len(matches)} uncertain create(s) match"
            )
        u = matches[0]
        self._record_batch(batch_id, u["chunk"], u["custom_ids"], u["input_hashes"])

    def wait(self, batch_id: str, first: float = 30.0, cap: float = 600.0) -> dict:
        delay = first
        while True:
            batch = self._http("GET", f"{API}/v1/messages/batches/{batch_id}").json()
            if batch["processing_status"] == "ended":
                return batch
            self.log(
                f"batch {batch_id}: {batch['processing_status']} {batch.get('request_counts', '')}"
            )
            self.sleep(delay)
            delay = min(delay * 1.5, cap)

    def collect(self, entry: dict) -> tuple[int, int]:
        """Poll one state entry to completion and write its results. -> (succeeded, failed).

        순서: 결과를 전부 메모리에 모은 뒤 **판정 원장(DB)에 먼저** 쓰고(``ledger`` 가 있으면), 그다음 JSONL
        작업 상태(캐시·오류·usage)를 붙이고 배치를 처리 완료로 표시한다. DB 쓰기가 실패하면 예외가 그대로 올라가고
        배치는 미처리로 남는다 — 다시 돌리면 다시 수집한다(파일만 성공한 척하지 않는다).
        원장의 judged_at = 배치 종료 시각(``ended_at`` — 결과를 알 수 있게 된 시각), judged_at_exact TRUE.
        """
        batch = self.wait(entry["id"])
        text = self._http("GET", batch["results_url"]).text
        ended = batch.get("ended_at")
        judged_at = (
            dt.datetime.fromisoformat(str(ended).replace("Z", "+00:00")) if ended else dt.datetime.now(dt.UTC)
        )
        keys = entry["custom_ids"]
        hashes = entry.get("input_hashes") or {}
        ok = bad = 0
        seen = set()
        usage: dict = {}
        n_usage = n_cache_read = 0
        cache_rows: list[dict] = []
        error_rows: list[dict] = []
        usage_rows: list[dict] = []
        ledger_rows: list[dict] = []

        def to_ledger(key: str, cid: str, u: dict | None, *, output: dict | None = None, error: str | None = None):
            h = hashes.get(cid)
            if h is None:  # 해시 이전 상태 파일 — 무엇을 판정했는지 모르니 원장에 넣지 않는다
                return
            ledger_rows.append({"key": key, "input_hash": h, "output": output, "error": error,
                                "batch_id": entry["id"], "judged_at": judged_at, "judged_at_exact": True,
                                "usage": u})

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
            # 과금은 응답이 있는 모든 결과(스키마 오류 포함)에 붙는다 — 측정은 파싱 전에 한다.
            u = (result.get("message") or {}).get("usage")
            if u:
                add_usage(usage, u)
                n_usage += 1
                n_cache_read += bool(u.get("cache_read_input_tokens"))
                usage_rows.append(
                    {"key": key, "batch_id": entry["id"], "result": result.get("type"),
                     "usage": {k: int(u.get(k) or 0) for k in USAGE_FIELDS}}
                )
            if result.get("type") != "succeeded":
                err = {"key": key, "reason": result.get("type") or "unknown", "detail": result.get("error"),
                       "batch_id": entry["id"]}
                error_rows.append(err)
                if counts_as_attempt(err):  # 계정 거절(크레딧·인증)은 판정 시도가 아니다 — 원장에 넣지 않는다
                    to_ledger(key, cid, u, error=f"{err['reason']}: {json.dumps(err['detail'], ensure_ascii=False)}")
                bad += 1
                continue
            blocks = (result.get("message") or {}).get("content") or []
            content = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            try:
                features = parse_features(content)
            except SchemaError as exc:
                error_rows.append({"key": key, "reason": "schema", "detail": str(exc), "batch_id": entry["id"]})
                to_ledger(key, cid, u, error=f"schema: {exc}")
                bad += 1
                continue
            cached: dict = {"key": key}
            # the hash of the input actually submitted (entries from before hashing have none)
            if (h := hashes.get(cid)) is not None:
                cached["input_hash"] = h
            cached["v2"] = {**asdict(features), "score": features.score()}
            cache_rows.append(cached)
            to_ledger(key, cid, u, output=cached["v2"])
            ok += 1
        for cid in set(keys) - seen:
            error_rows.append({"key": keys[cid], "reason": "missing_result", "detail": None, "batch_id": entry["id"]})
            to_ledger(keys[cid], cid, None, error="missing_result: null")
            bad += 1
        if self.ledger is not None:
            counts = self.ledger.record(ledger_rows)  # 실패하면 여기서 예외 — 아래 파일·상태는 건드리지 않는다
            self.log(f"batch {entry['id']}: ledger {self.ledger.spec.judge} {counts}")
        for r in cache_rows:
            self._append(self.cache, r)
        for r in error_rows:
            self._append(self.errors, r)
        for r in usage_rows:
            self._append(self.usage_log, r)
        state = self._load_state()
        model = (state.get("run") or {}).get("model", self.model)
        measured = usage_cost(usage, model)
        for b in state["batches"]:
            if b["id"] == entry["id"]:
                b["processed"] = True
                b["usage"] = {**usage, "results_with_usage": n_usage, "results_with_cache_read": n_cache_read}
                b["measured_usd"] = measured
        self._save_state(state)
        self._last_usage = {**usage, "results_with_usage": n_usage, "results_with_cache_read": n_cache_read}
        self.log(
            f"batch {entry['id']}: {ok} cached, {bad} failed; billed {usage} "
            f"(cache_read on {n_cache_read}/{n_usage}) = ${measured:.4f} measured"
        )
        return ok, bad

    def _bank_usage(self, summary: dict) -> None:
        """방금 collect 한 배치의 실제 usage 를 요약에 더한다(측정 금액 포함)."""
        u = getattr(self, "_last_usage", None) or {}
        total = summary.setdefault("usage", {})
        for k, v in u.items():
            total[k] = total.get(k, 0) + v
        model = (self._load_state().get("run") or {}).get("model", self.model)
        summary["measured_usd"] = usage_cost(total, model)
        self._last_usage = {}

    # --- orchestration ------------------------------------------------------------

    def collect_pending(self) -> dict:
        """Collect every batch not processed yet; submit nothing.

        For a run stopped by something a resubmit cannot fix (credit exhausted,
        a pricing surprise): the batches already created are paid for, so their
        results are banked, while the unsubmitted chunks stay unsubmitted and
        the run stays incomplete for a later resume.
        """
        summary: dict = {
            "resumed": 1,
            "submitted": 0,
            "succeeded": 0,
            "failed": 0,
            "collect_only": True,
        }
        for entry in [b for b in self._load_state()["batches"] if not b["processed"]]:
            ok, bad = self.collect(entry)
            self._bank_usage(summary)
            summary["succeeded"] += ok
            summary["failed"] += bad
        return summary

    def run(
        self,
        pairs: list[dict],
        *,
        sample: int,
        stratify: list[str],
        seed: int,
        dry_run: bool = False,
        calibrate: int = 0,
        new_run: bool = False,
        collect_only: bool = False,
        max_submit: int | None = None,
    ) -> dict:
        """Draw + persist a sample, submit its chunks, collect. Reruns continue the stored run.

        State `run` holds the chosen keys and chunking (written before the first
        submit). A rerun with an incomplete run submits only chunks that have no
        batch yet and collects unprocessed batches -- the CLI's sample/seed are
        ignored then. A completed run blocks a fresh sample unless `new_run`.
        `collect_only` banks the batches already created and submits nothing.
        `max_submit` caps how many chunks this invocation submits (the Batch API
        caps requests in flight at ~100k): the wave is collected, the run stays
        incomplete, and the next invocation submits the next wave.
        """
        summary: dict = {"resumed": 0, "submitted": 0, "succeeded": 0, "failed": 0}
        if collect_only:
            return self.collect_pending()
        state = self._load_state()
        run = state.get("run")
        if run and not run.get("completed"):
            if dry_run:
                self.log(f"incomplete run in {self.state} -- a real run resumes it")
                summary["pending_run"] = True
                return summary
            summary["resumed"] = 1
            self.log(f"resuming stored run (seed {run['seed']}, {len(run['keys'])} keys)")
            return self._finish(run, {p["key"]: p for p in pairs}, summary, max_submit=max_submit)
        if run and not new_run:
            self.log(f"completed run in {self.state}; pass --new-run to draw a fresh sample")
            summary["refused"] = True
            return summary

        cached = self.cached_rows()
        failures = self.failure_counts()
        unique = {p["key"]: p for p in pairs}
        todo = [
            p
            for k, p in unique.items()
            if failures[k] < MAX_ATTEMPTS and not self.is_cached(p, cached)
        ]
        # Pre-LLM filter (off by default -- the pair builder already filtered): drop pairs
        # that can never be an event before they cost anything. `related_n` comes from the
        # pairs file; without it the fan-out/index rules are inert (None = unknown).
        kept = (
            todo
            if self.prefilter is None
            else [p for p in todo if keep_pair(p.get("title"), p.get("related_n"), self.prefilter)]
        )
        prefiltered = len(todo) - len(kept)
        if prefiltered:
            self.log(f"prefilter skipped {prefiltered} of {len(todo)} eligible pairs")
        summary["prefiltered"] = prefiltered
        todo = kept
        chosen = stratified_sample(todo, sample, stratify, seed)
        requests = [build_request(p, self.model, self.addendum, self.cache_system) for p in chosen]
        if calibrate and requests:  # count_tokens is free: also in --dry-run
            probe = random.Random(seed).sample(requests, min(calibrate, len(requests)))
            ratio = sum(_chars(r) for r in probe) / sum(self.count_tokens(r) for r in probe)
            self.chars_per_token = ratio / 1.1  # 10% safety margin
            self.log(f"calibrated chars/token {ratio:.3f} -> using {self.chars_per_token:.3f}")
        est = self._estimate(requests)
        summary.update(
            {
                "pairs": len(unique),
                "cached": len(unique) - len(todo),
                "eligible": len(todo),
                "sample": len(requests),
                "estimate": est,
            }
        )
        self.log(
            f"pairs {len(unique)}, eligible {len(todo)}, sample {len(requests)}; "
            f"estimate {est['input_tokens']:,} in + {est['output_tokens']:,} out tokens "
            f"(system {est['cached_system_tokens']:,} cached) = ${est['usd']:.2f} ESTIMATE (batch-discounted) "
            f"vs budget ${self.budget_usd:.2f}"
        )
        if dry_run:
            summary["over_budget"] = est["usd"] > self.budget_usd
            return summary
        if est["usd"] > self.budget_usd:
            raise BudgetExceeded(f"estimate ${est['usd']:.2f} > budget ${self.budget_usd:.2f}")
        if not chosen:
            self.log("nothing to judge")
            return summary
        keys = [p["key"] for p in chosen]
        run = {
            "seed": seed,
            "sample": sample,
            "stratify": stratify,
            "model": self.model,
            "system_addendum": self.addendum,
            "cache_system": self.cache_system,
            "prefilter": None if self.prefilter is None else self.prefilter.describe(),
            "estimate": est,
            "budget_usd": self.budget_usd,
            "keys": keys,
            "chunks": [keys[i : i + self.chunk] for i in range(0, len(keys), self.chunk)],
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "completed": False,
        }
        history = state.get("history", [])
        if state.get("run"):
            history.append({"run": state["run"], "batches": state["batches"]})
        self._save_state({"run": run, "batches": [], "history": history})
        return self._finish(run, unique, summary, max_submit=max_submit)

    def _finish(
        self,
        run: dict,
        by_key: dict[str, dict],
        summary: dict,
        *,
        max_submit: int | None = None,
    ) -> dict:
        submitted = {b["chunk"] for b in self._load_state()["batches"]}
        remaining: list[tuple[int, list[str], list[dict]]] = []
        for i, chunk_keys in enumerate(run["chunks"]):
            if i in submitted:
                continue
            missing = [k for k in chunk_keys if k not in by_key]
            if missing:
                raise KeyError(
                    f"{len(missing)} stored sample key(s) not in --pairs, e.g. {missing[0]}"
                )
            # a resume sends exactly what the stored run chose (addendum, caching), not today's CLI flags
            addendum = run.get("system_addendum", self.addendum)
            cache_system = run.get("cache_system", self.cache_system)
            reqs = [build_request(by_key[k], run["model"], addendum, cache_system) for k in chunk_keys]
            remaining.append((i, chunk_keys, reqs))
        if remaining:
            est = self._estimate([r for *_, reqs in remaining for r in reqs])
            self.log(
                f"run original estimate ${run['estimate']['usd']:.2f} under budget "
                f"${run['budget_usd']:.2f}; remaining {len(remaining)}/{len(run['chunks'])} "
                f"chunk(s) estimate ${est['usd']:.2f} vs current budget ${self.budget_usd:.2f}"
            )
            summary["remaining_estimate"] = est
            # a resume spends real money too: check the unsubmitted part against today's budget
            if est["usd"] > self.budget_usd:
                raise BudgetExceeded(
                    f"remaining estimate ${est['usd']:.2f} > budget ${self.budget_usd:.2f}"
                )
        wave = remaining if max_submit is None else remaining[:max_submit]
        if len(wave) < len(remaining):
            self.log(f"submitting {len(wave)} of {len(remaining)} remaining chunk(s) this wave")
        for i, chunk_keys, reqs in wave:
            self._resolve_uncertain(i)
            self.submit(reqs, {custom_id_of(k): k for k in chunk_keys}, i)
            summary["submitted"] += 1
        del remaining, wave  # the requests are large: free them before the long poll
        for entry in [b for b in self._load_state()["batches"] if not b["processed"]]:
            ok, bad = self.collect(entry)
            self._bank_usage(summary)
            summary["succeeded"] += ok
            summary["failed"] += bad
        state = self._load_state()
        done = {b["chunk"] for b in state["batches"]}
        left = [i for i in range(len(run["chunks"])) if i not in done]
        summary["chunks_left"] = len(left)
        if left:
            self.log(f"{len(left)} chunk(s) not submitted yet -- rerun to continue")
        else:
            state["run"]["completed"] = True
        self._save_state(state)
        return summary


def _api_key() -> str | None:
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    env = REPO.parent / "quant-airflow" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return None


def api_key_for(
    dry_run: bool, calibrate: int, lookup: Callable[[], str | None] = _api_key
) -> str | None:
    """A dry run gets the key only for --calibrate (free count_tokens); it never creates batches."""
    return lookup() if (not dry_run or calibrate) else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--budget-usd", type=float)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--stratify", default="month,sector,session")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL, help="스윙 2호 = claude-haiku-5-5 (Haiku 4.5 도 허용)")
    ap.add_argument("--run", default=None,
                    help="런 이름(예: stage2) → judge_runs/<judge>/<run>.state.json. 구간(단계)마다 따로 — 끝난 런은 "
                    "--new-run 없이 새 표본을 막는다")
    ap.add_argument("--cache", type=Path, default=None, help="기본 judge_runs/<judge>/judgments.jsonl")
    ap.add_argument("--errors", type=Path, default=None, help="기본 judge_runs/<judge>/errors.jsonl")
    ap.add_argument("--state", type=Path, default=None, help="기본 judge_runs/<judge>/<run>.state.json")
    ap.add_argument("--chars-per-token", type=float, default=1.0)
    ap.add_argument("--est-output-tokens", type=int, default=200)
    ap.add_argument("--cache-hit-rate", type=float, default=1.0,
                    help="추정용 system 캐시 적중률(첫 요청은 항상 쓰기). 측정치는 결과 usage 에 남는다")
    ap.add_argument("--calibrate", type=int, default=0, help="count_tokens on N sampled requests")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--system-addendum",
        type=Path,
        default=ADDENDUM,
        help="운영 system 뒤에 붙일 보충(기본 = 사전등록 r3). --no-addendum 이면 운영 v2 그대로",
    )
    ap.add_argument("--no-addendum", action="store_true")
    ap.add_argument("--no-cache-system", action="store_true", help="system 프롬프트 캐싱을 끈다")
    ap.add_argument("--prefilter", action="store_true",
                    help="판정 전 필터를 여기서도 적용(기본 끔 — 쌍 생성기가 이미 거른다). 아래 파라미터로 조정")
    add_filter_args(ap)
    ap.add_argument(
        "--adopt-batch",
        default=None,
        help="record an existing batch for a failed (uncertain) create, then resume",
    )
    ap.add_argument(
        "--new-run",
        action="store_true",
        help="draw a fresh sample even though the state file holds a completed run",
    )
    ap.add_argument(
        "--max-submit",
        type=int,
        default=None,
        help="submit at most N chunks this run (Batch API caps ~100k requests in flight)",
    )
    ap.add_argument(
        "--days",
        default=None,
        help="restrict the pairs file to A..B (inclusive) or a comma list of days, so a "
        "partial budget finishes whole days instead of thinning every day",
    )
    ap.add_argument(
        "--collect-only",
        action="store_true",
        help="collect the batches already created and submit nothing (no spend, no --pairs read)",
    )
    args = ap.parse_args(argv)
    if not args.collect_only:  # a collect spends nothing and touches no pair
        missing = [f for f in ("pairs", "budget_usd", "sample") if getattr(args, f) is None]
        if missing:
            ap.error("required unless --collect-only: " + ", ".join(f"--{m}" for m in missing))
    # Haiku 4.5 (daytrade-it 운영 판정)와 5.5(스윙 2호)는 캐시를 공유하지 않는다 — 입력 해시에 모델이 들어간다.
    if not (args.model.startswith("claude-haiku-4-5") or args.model == "claude-haiku-5-5"):
        sys.exit(f"model must be Haiku 4.5 or 5.5, got {args.model}")
    addendum = None if args.no_addendum else read_addendum(args.system_addendum)
    spec = judge_spec(args.model, addendum)
    run_dir = RUNS / spec.judge
    if args.state is None and args.run is None:
        ap.error("--run NAME (또는 --state PATH) 이 필요하다")
    state = args.state or run_dir / f"{args.run}.state.json"
    cache = args.cache or run_dir / "judgments.jsonl"
    errors = args.errors or run_dir / "errors.jsonl"

    # --collect-only touches no pair: skip reading the (large) pairs file entirely
    pairs = (
        []
        if args.collect_only
        else [json.loads(line) for line in args.pairs.open() if line.strip()]
    )
    if args.days and not args.collect_only:
        pairs = select_days(pairs, args.days)
        if not pairs:
            ap.error(f"--days {args.days} selects no pair in {args.pairs}")
    with httpx.Client() as client:
        judge = BatchJudge(
            client=client,
            api_key=api_key_for(args.dry_run, args.calibrate),
            model=args.model,
            cache=cache,
            errors=errors,
            state=state,
            budget_usd=args.budget_usd or 0.0,
            chars_per_token=args.chars_per_token,
            est_output_tokens=args.est_output_tokens,
            chunk=args.chunk,
            addendum=addendum,
            cache_system=not args.no_cache_system,
            cache_hit_rate=args.cache_hit_rate,
            prefilter=filter_from_args(args) if args.prefilter else None,
            # 판정 원장(DB)이 정본 — 결과를 모으는 모든 실행이 쓴다. 연결이 안 되면 여기서 실패한다.
            ledger=None if args.dry_run else JudgmentLedger(connect_db(), spec),
        )
        try:
            if args.adopt_batch:
                judge.adopt_batch(args.adopt_batch)
            summary = judge.run(
                pairs,
                sample=args.sample or 0,
                stratify=[f for f in args.stratify.split(",") if f],
                seed=args.seed,
                dry_run=args.dry_run,
                calibrate=args.calibrate,
                new_run=args.new_run,
                collect_only=args.collect_only,
                max_submit=args.max_submit,
            )
        except (BudgetExceeded, SubmitUncertain) as exc:
            sys.exit(f"ABORT: {exc}")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
