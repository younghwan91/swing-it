"""배치 판정기 — Batch API 를 httpx.MockTransport 로만 친다(네트워크 절대 안 씀).

daytrade-it ``tests/unit/scripts/test_judge_batch.py`` 의 안전 동작 테스트를 그대로 옮기고(예산·재개·재시도 금지·
오류 파일·입력 해시), 이식하며 더한 것 — system 프롬프트 캐싱, 실측 usage 기록, 캐시 인지 비용 추정 — 을 덧붙였다.
"""

import datetime as dt
import json
from pathlib import Path

import httpx
import pytest

import swing_it.news.judge as judge_batch
from swing_it.news.judge import (
    BatchJudge,
    BudgetExceeded,
    SubmitUncertain,
    build_request,
    cache_row_valid,
    custom_id_of,
    estimate_cost,
    pair_input_hash,
    request_input_hash,
    retry_after_seconds,
    stratified_sample,
    usage_cost,
)
from swing_it.news.prefilter import PairFilter
from swing_it.news.prompt import (
    _MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    ArticleFeatures,
    build_prompt,
    judgment_input_hash,
)

FEATS = {
    "subject": "main",
    "reports_price_move": False,
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "surprise": 1,
    "sentiment_direction": 1,
    "sentiment_strength": 2,
    "scale_vs_size": "medium",
}
UUID = "595eead7-3889-57cd-bdc5-fc299f660c70"


def _pair(i: int, month: str = "2026-09", sector: str = "IT", session: str = "intraday") -> dict:
    return {
        "key": f"{UUID[:-4]}{i:04d}|{i:06d}",
        "code": f"{i:06d}",
        "name": f"회사{i}",
        "market": "KOSPI",
        "sector": sector,
        "title": f"제목 {i}",
        "body": "요약 " * 50,
        "category": None,
        "month": month,
        "session": session,
    }


def test_build_request_uses_live_prompt_and_limits() -> None:
    p = _pair(1)
    req = build_request(p, model="claude-haiku-4-5")
    system, user = build_prompt(
        code=p["code"],
        name=p["name"],
        market=p["market"],
        sector=p["sector"],
        title=p["title"],
        body=p["body"],
        category=p["category"],
    )
    assert req["custom_id"] == custom_id_of(p["key"])
    assert len(req["custom_id"]) <= 64 and "|" not in req["custom_id"]
    assert req["params"] == {
        "model": "claude-haiku-4-5",
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "temperature": 0,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": user}],
    }
    plain = build_request(p, model="claude-haiku-4-5", cache_system=False)
    assert plain["params"]["system"] == system


def test_stratified_sample_is_proportional_seeded_and_covers_strata() -> None:
    pairs = [_pair(i, month="2026-08" if i < 80 else "2026-09") for i in range(100)]
    a = stratified_sample(pairs, 10, ["month", "sector", "session"], seed=7)
    b = stratified_sample(list(reversed(pairs)), 10, ["month", "sector", "session"], seed=7)
    assert [p["key"] for p in a] == [p["key"] for p in b]
    months = [p["month"] for p in a]
    assert months.count("2026-08") == 8 and months.count("2026-09") == 2
    assert stratified_sample(pairs, 500, ["month"], seed=1) == sorted(pairs, key=lambda p: p["key"])


def test_estimate_cost_is_batch_discounted_haiku_price() -> None:
    reqs = [build_request(_pair(i), model="m", cache_system=False) for i in range(10)]
    est = estimate_cost(reqs, chars_per_token=1.0, output_tokens=200)
    chars = sum(
        len(r["params"]["system"]) + len(r["params"]["messages"][0]["content"]) for r in reqs
    )
    assert est["input_tokens"] == chars and est["cached_system_tokens"] == 0 and est["is_estimate"]
    assert est["usd"] == pytest.approx((chars * 1.0 + 10 * 200 * 5.0) / 1e6 * 0.5)


def test_estimate_cost_prices_cached_system_as_one_write_then_reads() -> None:
    reqs = [build_request(_pair(i), model="claude-haiku-5-5") for i in range(10)]
    sys_chars = len(reqs[0]["params"]["system"][0]["text"])
    user = sum(len(r["params"]["messages"][0]["content"]) for r in reqs)
    est = estimate_cost(reqs, chars_per_token=1.0, output_tokens=100)
    assert est["cached_system_tokens"] == 10 * sys_chars
    billed_in = user + sys_chars * 1.25 + 9 * sys_chars * 0.1
    assert est["usd"] == pytest.approx((billed_in * 0.10 + 1000 * 0.50) / 1e6 * 0.5)
    # 적중률 0 이면 매번 쓰기(1.25×) — 캐싱 안 한 것보다 비싸다
    worst = estimate_cost(reqs, chars_per_token=1.0, output_tokens=100, cache_hit_rate=0.0)
    flat = estimate_cost([build_request(_pair(i), "claude-haiku-5-5", cache_system=False) for i in range(10)],
                         chars_per_token=1.0, output_tokens=100)
    assert worst["usd"] > flat["usd"] > est["usd"]


class FakeAPI:
    """Minimal Message Batches API: create -> in_progress once -> ended."""

    def __init__(self, result_for=None, prefix: str = "b") -> None:
        self.prefix = prefix
        self.created: list[dict] = []
        self.created_at: list[str] = []
        self.posts = 0
        self.polls = 0
        self.result_for = result_for or (lambda _cid: {"type": "succeeded", "message": _msg(FEATS)})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "k"
        assert request.headers["anthropic-version"] == "2023-06-01"
        path = request.url.path
        if request.method == "GET" and path == "/v1/messages/batches":
            data = [
                {
                    "id": f"{self.prefix}{i + 1}",
                    "created_at": self.created_at[i],
                    "request_counts": {"processing": len(b["requests"])},
                }
                for i, b in reversed(list(enumerate(self.created)))
            ]
            return httpx.Response(200, json={"data": data, "has_more": False})
        if request.method == "POST" and path == "/v1/messages/batches":
            self.posts += 1
            body = json.loads(request.content)
            self.created.append(body)
            self.created_at.append(dt.datetime.now(dt.UTC).isoformat())
            return httpx.Response(
                200,
                json={
                    "id": f"{self.prefix}{len(self.created)}",
                    "processing_status": "in_progress",
                },
            )
        if (
            request.method == "GET"
            and path.startswith("/v1/messages/batches/")
            and not path.endswith("/results")
        ):
            bid = path.rsplit("/", 1)[1]
            self.polls += 1
            status = "ended" if self.polls % 2 == 0 else "in_progress"
            return httpx.Response(
                200,
                json={
                    "id": bid,
                    "processing_status": status,
                    "request_counts": {
                        "processing": len(
                            self.created[int(bid[len(self.prefix) :]) - 1]["requests"]
                        )
                    },
                    "results_url": f"https://api.anthropic.com/v1/messages/batches/{bid}/results"
                    if status == "ended"
                    else None,
                },
            )
        if path.endswith("/results"):
            bid = path.split("/")[-2]
            reqs = self.created[int(bid[len(self.prefix) :]) - 1]["requests"]
            lines = [
                json.dumps({"custom_id": r["custom_id"], "result": self.result_for(r["custom_id"])})
                for r in reqs
            ]
            return httpx.Response(200, text="\n".join(lines) + "\n")
        return httpx.Response(404)


def _msg(payload: dict | str) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1300, "output_tokens": 150},
    }


def _judge(tmp_path: Path, api: FakeAPI, budget: float = 100.0, chunk: int = 10_000) -> BatchJudge:
    client = httpx.Client(transport=httpx.MockTransport(api), base_url="https://api.anthropic.com")
    return BatchJudge(
        client=client,
        api_key="k",
        model="claude-haiku-4-5",
        cache=tmp_path / "cache.jsonl",
        errors=tmp_path / "errors.jsonl",
        state=tmp_path / "state.json",
        budget_usd=budget,
        chunk=chunk,
        sleep=lambda _s: None,
    )


def test_budget_abort_submits_nothing(tmp_path: Path) -> None:
    api = FakeAPI()
    j = _judge(tmp_path, api, budget=0.0001)
    with pytest.raises(BudgetExceeded):
        j.run([_pair(i) for i in range(5)], sample=5, stratify=["month"], seed=1)
    assert api.created == []
    assert not (tmp_path / "state.json").exists()


def test_run_submits_polls_and_writes_cache_skipping_cached(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(4)]
    (tmp_path / "cache.jsonl").write_text(
        json.dumps({"key": pairs[0]["key"], "v2": {"score": 0}}) + "\n"
    )
    api = FakeAPI()
    j = _judge(tmp_path, api)
    summary = j.run(pairs, sample=100, stratify=["month"], seed=1)
    assert len(api.created) == 1
    assert len(api.created[0]["requests"]) == 3
    rows = [json.loads(line) for line in (tmp_path / "cache.jsonl").open()]
    assert [r["key"] for r in rows[1:]] == sorted(p["key"] for p in pairs[1:])
    assert rows[1]["v2"]["score"] == pytest.approx(ArticleFeatures(**FEATS).score())
    assert set(rows[1]["v2"]) == set(FEATS) | {"score"}
    assert summary["succeeded"] == 3
    state = json.loads((tmp_path / "state.json").read_text())
    assert all(b["processed"] for b in state["batches"])


def test_results_errored_and_schema_failures_go_to_errors_file(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(3)]
    cids = [custom_id_of(p["key"]) for p in sorted(pairs, key=lambda p: p["key"])]

    def result_for(cid: str) -> dict:
        if cid == cids[0]:
            return {
                "type": "errored",
                "error": {"type": "error", "error": {"type": "overloaded_error", "message": "x"}},
            }
        if cid == cids[1]:
            return {"type": "succeeded", "message": _msg({**FEATS, "persistence": "forever"})}
        return {"type": "succeeded", "message": _msg(FEATS)}

    api = FakeAPI(result_for)
    summary = _judge(tmp_path, api).run(pairs, sample=10, stratify=["month"], seed=1)
    assert (summary["succeeded"], summary["failed"]) == (1, 2)
    errs = [json.loads(line) for line in (tmp_path / "errors.jsonl").open()]
    assert {e["reason"] for e in errs} == {"errored", "schema"}
    # MAX_ATTEMPTS=3 (live score_article tries 3 times): two more explicit runs
    # resubmit both, a fourth submits nothing
    for n in range(2):
        api_n = FakeAPI(result_for, prefix=f"r{n}b")
        _judge(tmp_path, api_n).run(pairs, sample=10, stratify=["month"], seed=1, new_run=True)
        assert len(api_n.created[0]["requests"]) == 2
    api4 = FakeAPI(result_for, prefix="r4b")
    _judge(tmp_path, api4).run(pairs, sample=10, stratify=["month"], seed=1, new_run=True)
    assert api4.created == []


def test_credit_and_auth_errors_do_not_burn_an_attempt(tmp_path: Path) -> None:
    """A per-request API error that is not the pair's fault was never a judgement.

    The batch run of 2026-09-16 came back with 61,515 rows of
    `invalid_request_error: "Your credit balance is too low"`. Counting those as
    attempts would lock those pairs out of MAX_ATTEMPTS for good, although not one
    of them was ever judged.
    """

    def errored(err_type: str) -> dict:
        return {"type": "error", "error": {"type": err_type, "message": "x"}}

    rows = [
        {
            "key": "a|1",
            "reason": "errored",
            "detail": errored("invalid_request_error"),
            "batch_id": "b1",
        },
        {
            "key": "a|1",
            "reason": "errored",
            "detail": errored("authentication_error"),
            "batch_id": "b2",
        },
        {
            "key": "b|1",
            "reason": "errored",
            "detail": errored("overloaded_error"),
            "batch_id": "b1",
        },
        {
            "key": "c|1",
            "reason": "schema",
            "detail": "persistence outside allowed set",
            "batch_id": "b1",
        },
        {"key": "d|1", "reason": "missing_result", "detail": None, "batch_id": "b1"},
    ]
    (tmp_path / "errors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = _judge(tmp_path, FakeAPI()).failure_counts()
    assert counts["a|1"] == 0  # credit / auth: never judged
    assert (counts["b|1"], counts["c|1"], counts["d|1"]) == (1, 1, 1)


def test_max_submit_caps_the_wave_and_leaves_the_run_resumable(tmp_path: Path) -> None:
    """The Batch API caps requests in flight (~100k), so a run submits in waves."""
    pairs = [_pair(i) for i in range(6)]
    api = FakeAPI()
    summary = _judge(tmp_path, api, chunk=2).run(
        pairs, sample=6, stratify=["month"], seed=1, max_submit=2
    )
    assert len(api.created) == 2  # 3 chunks planned, 2 submitted
    assert summary["submitted"] == 2 and summary["chunks_left"] == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["run"]["completed"] is False
    assert all(b["processed"] for b in state["batches"])  # the wave was collected

    summary = _judge(tmp_path, api, chunk=2).run(pairs, sample=6, stratify=["month"], seed=1)
    assert len(api.created) == 3 and summary["submitted"] == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["run"]["completed"] is True
    keys = {json.loads(line)["key"] for line in (tmp_path / "cache.jsonl").open()}
    assert keys == {p["key"] for p in pairs}


def test_failure_counts_dedupe_by_key_and_batch(tmp_path: Path) -> None:
    rows = [
        {"key": "a|1", "reason": "schema", "batch_id": "b1"},
        {"key": "a|1", "reason": "schema", "batch_id": "b1"},
        {"key": "a|1", "reason": "errored", "batch_id": "b2"},
    ]
    (tmp_path / "errors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert _judge(tmp_path, FakeAPI()).failure_counts()["a|1"] == 2


class CrashOnSecondCreate(FakeAPI):
    def __init__(self) -> None:
        super().__init__()
        self.crash = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.crash and request.method == "POST" and len(self.created) == 1:
            raise httpx.ConnectError("boom")
        return super().__call__(request)


def test_crash_after_first_chunk_rerun_submits_only_the_rest_of_the_same_sample(
    tmp_path: Path,
) -> None:
    pairs = [_pair(i) for i in range(4)]
    api = CrashOnSecondCreate()
    with pytest.raises(SubmitUncertain):
        _judge(tmp_path, api, chunk=2).run(pairs, sample=4, stratify=["month"], seed=1)
    assert len(api.created) == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["run"]["keys"] == sorted(p["key"] for p in pairs)
    log = [json.loads(line) for line in (tmp_path / "state.batches.log").open()]
    assert [r["id"] for r in log if "id" in r] == ["b1"]
    assert [r["chunk"] for r in log if r.get("uncertain")] == [1]
    first = {r["custom_id"] for r in api.created[0]["requests"]}

    api.crash = False
    # a different seed/sample on rerun must not matter: the stored sample wins
    summary = _judge(tmp_path, api, chunk=2).run(
        pairs + [_pair(9)], sample=1, stratify=["month"], seed=99
    )
    assert len(api.created) == 2
    second = {r["custom_id"] for r in api.created[1]["requests"]}
    assert first.isdisjoint(second) and len(second) == 2
    assert summary["succeeded"] == 4
    keys = {json.loads(line)["key"] for line in (tmp_path / "cache.jsonl").open()}
    assert keys == {p["key"] for p in pairs}


def test_completed_run_refuses_fresh_sample_without_new_run(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(4)]
    api = FakeAPI()
    _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=1)
    assert len(api.created) == 1
    summary = _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=2)
    assert len(api.created) == 1
    assert summary["refused"] is True
    _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=2, new_run=True)
    assert len(api.created) == 2
    assert len(api.created[1]["requests"]) == 2


def test_dry_run_makes_no_http_calls(tmp_path: Path) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise AssertionError("dry run must not call the API")

    client = httpx.Client(transport=httpx.MockTransport(boom))
    j = BatchJudge(
        client=client,
        api_key=None,
        model="claude-haiku-4-5",
        cache=tmp_path / "c.jsonl",
        errors=tmp_path / "e.jsonl",
        state=tmp_path / "s.json",
        budget_usd=0.0,
        sleep=lambda _s: None,
    )
    summary = j.run(
        [_pair(i) for i in range(20)],
        sample=5,
        stratify=["month", "sector", "session"],
        seed=3,
        dry_run=True,
    )
    assert summary["sample"] == 5
    assert summary["estimate"]["usd"] > 0


def test_resume_aborts_when_remaining_chunks_exceed_current_budget(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(4)]
    api = CrashOnSecondCreate()
    with pytest.raises(SubmitUncertain):
        _judge(tmp_path, api, chunk=2).run(pairs, sample=4, stratify=["month"], seed=1)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["run"]["budget_usd"] == 100.0 and state["run"]["estimate"]["usd"] > 0
    api.crash = False
    logs: list[str] = []
    j = _judge(tmp_path, api, chunk=2, budget=0.0001)
    j.log = logs.append
    with pytest.raises(BudgetExceeded):
        j.run(pairs, sample=4, stratify=["month"], seed=1)
    assert len(api.created) == 1  # remaining chunk not submitted
    assert any("original estimate" in m for m in logs)
    assert any("remaining" in m for m in logs)


# --- F5: cache rows are keyed by the judged input ------------------------------


def test_cache_rows_store_the_input_hash_of_what_was_judged(tmp_path: Path) -> None:
    pairs = [_pair(1)]
    _judge(tmp_path, FakeAPI()).run(pairs, sample=10, stratify=["month"], seed=1)
    (row,) = [json.loads(line) for line in (tmp_path / "cache.jsonl").open()]
    req = build_request(pairs[0], "claude-haiku-4-5", cache_system=False)["params"]
    expected = judgment_input_hash(req["model"], req["system"], req["messages"][0]["content"])
    assert row["input_hash"] == expected == pair_input_hash(pairs[0], "claude-haiku-4-5")


def test_changed_input_is_rejudged(tmp_path: Path) -> None:
    pair = _pair(1)
    _judge(tmp_path, FakeAPI()).run([pair], sample=10, stratify=["month"], seed=1)
    api = FakeAPI()
    changed = {**pair, "title": "다른 제목"}
    _judge(tmp_path, api).run([changed], sample=10, stratify=["month"], seed=1, new_run=True)
    assert len(api.created) == 1 and len(api.created[0]["requests"]) == 1
    # the same input again: nothing to judge
    api2 = FakeAPI()
    _judge(tmp_path, api2).run([changed], sample=10, stratify=["month"], seed=1, new_run=True)
    assert api2.created == []


def test_legacy_rows_without_hash_valid_only_if_input_unchanged(tmp_path: Path) -> None:
    same = {**_pair(1), "summary": "요약 " * 50}  # body == summary: pre-F5 input unchanged
    fallback = {**_pair(2), "summary": "", "body": "제목 2"}  # pre-F5 body was ""
    (tmp_path / "cache.jsonl").write_text(
        "".join(json.dumps({"key": p["key"], "v2": {"score": 0}}) + "\n" for p in (same, fallback))
    )
    api = FakeAPI()
    _judge(tmp_path, api).run([same, fallback], sample=10, stratify=["month"], seed=1)
    assert [r["custom_id"] for r in api.created[0]["requests"]] == [custom_id_of(fallback["key"])]


# --- minor: batch create is never retried ---------------------------------------


class CreateFails(FakeAPI):
    """POST /batches answers `status`; with `phantom` the batch was created anyway."""

    def __init__(self, status: int, phantom: bool = False) -> None:
        super().__init__()
        self.status = status
        self.phantom = phantom
        self.failing = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.failing and request.method == "POST" and request.url.path == "/v1/messages/batches":
            if self.phantom:
                super().__call__(request)
            else:
                self.posts += 1
            return httpx.Response(self.status)
        return super().__call__(request)


def test_create_5xx_is_not_retried_and_fails_loudly(tmp_path: Path) -> None:
    api = CreateFails(500)
    with pytest.raises(SubmitUncertain, match="batches"):
        _judge(tmp_path, api).run(
            [_pair(i) for i in range(2)], sample=2, stratify=["month"], seed=1
        )
    assert api.posts == 1
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["uncertain_submits"][0]["chunk"] == 0 and state["batches"] == []


def test_rerun_after_uncertain_create_refuses_when_an_unknown_batch_exists(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(2)]
    api = CreateFails(529, phantom=True)  # the batch exists server-side
    with pytest.raises(SubmitUncertain):
        _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=1)
    api.failing = False
    with pytest.raises(SubmitUncertain, match="b1"):
        _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=1)
    assert api.posts == 1  # never resubmitted

    # adopting the batch records it for the chunk and collects it: no second create
    j = _judge(tmp_path, api)
    j.adopt_batch("b1")
    summary = j.run(pairs, sample=2, stratify=["month"], seed=1)
    assert api.posts == 1 and summary["succeeded"] == 2
    assert "uncertain_submits" not in json.loads((tmp_path / "state.json").read_text())


def test_rerun_after_uncertain_create_resubmits_when_no_batch_was_created(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(2)]
    api = CreateFails(503)
    with pytest.raises(SubmitUncertain):
        _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=1)
    api.failing = False
    summary = _judge(tmp_path, api).run(pairs, sample=2, stratify=["month"], seed=1)
    assert api.posts == 2 and summary["succeeded"] == 2


def test_create_4xx_raises_without_retry(tmp_path: Path) -> None:
    api = CreateFails(400)
    with pytest.raises(httpx.HTTPStatusError):
        _judge(tmp_path, api).run([_pair(1)], sample=1, stratify=["month"], seed=1)
    assert api.posts == 1
    assert "uncertain_submits" not in json.loads((tmp_path / "state.json").read_text())


def test_dry_run_calibrate_counts_tokens_but_never_creates_a_batch(tmp_path: Path) -> None:
    """--dry-run --calibrate N: only the free count_tokens endpoint, no batch create."""
    calls: list[str] = []

    def api(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/v1/messages/count_tokens":
            body = json.loads(request.content)
            system = body["system"]  # 캐싱 블록 그대로 count_tokens 에 간다
            assert system[0]["cache_control"] == {"type": "ephemeral"}
            chars = len(system[0]["text"]) + len(body["messages"][0]["content"])
            return httpx.Response(200, json={"input_tokens": round(chars / 1.32)})
        raise AssertionError(f"dry run must not call {request.url.path}")

    j = _judge(tmp_path, api, budget=0.0)
    summary = j.run(
        [_pair(i) for i in range(20)],
        sample=5,
        stratify=["month", "sector", "session"],
        seed=3,
        dry_run=True,
        calibrate=3,
    )
    assert calls == ["POST /v1/messages/count_tokens"] * 3
    assert j.chars_per_token == pytest.approx(1.32 / 1.1, rel=0.02)
    assert summary["over_budget"] is True
    assert not (tmp_path / "state.json").exists()


def test_cli_dry_run_passes_the_api_key_only_when_calibrating() -> None:
    from swing_it.news.judge import api_key_for

    assert api_key_for(dry_run=True, calibrate=0, lookup=lambda: "k") is None
    assert api_key_for(dry_run=True, calibrate=20, lookup=lambda: "k") == "k"
    assert api_key_for(dry_run=False, calibrate=0, lookup=lambda: "k") == "k"


# --- a 429 on create is a rejection, so it is retried (a 5xx never is) ---------


class RateLimitedCreates(FakeAPI):
    """Rejects the first `refusals` creates with 429: nothing queued, nothing billed."""

    def __init__(self, refusals: int = 2, retry_after: str | None = None) -> None:
        super().__init__()
        self.refusals = refusals
        self.retry_after = retry_after
        self.rejected = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if (
            request.method == "POST"
            and request.url.path == "/v1/messages/batches"
            and self.rejected < self.refusals
        ):
            self.rejected += 1
            headers = {"retry-after": self.retry_after} if self.retry_after else {}
            return httpx.Response(429, json={"type": "error"}, headers=headers)
        return super().__call__(request)


def test_create_retries_a_429_honouring_retry_after_without_double_creating(
    tmp_path: Path,
) -> None:
    pairs = [_pair(i) for i in range(4)]
    api = RateLimitedCreates(refusals=2, retry_after="30")
    j = _judge(tmp_path, api)
    waits: list[float] = []
    j.sleep = waits.append
    summary = j.run(pairs, sample=4, stratify=["month"], seed=1)
    assert api.rejected == 2
    assert len(api.created) == 1 and len(api.created[0]["requests"]) == 4
    assert waits[:2] == [30.0, 30.0]
    assert summary["succeeded"] == 4
    log = [json.loads(line) for line in (tmp_path / "state.batches.log").open()]
    assert len(log) == 1  # the rejected attempts are not billable and are not logged


def test_create_backs_off_then_gives_up_on_a_persistent_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(judge_batch, "CREATE_429_RETRIES", 2)
    api = RateLimitedCreates(refusals=99)
    j = _judge(tmp_path, api)
    waits: list[float] = []
    j.sleep = waits.append
    with pytest.raises(httpx.HTTPStatusError):
        j.run([_pair(i) for i in range(2)], sample=2, stratify=["month"], seed=1)
    assert api.created == []
    assert waits == [judge_batch.CREATE_429_DELAY, judge_batch.CREATE_429_DELAY * 2]


def test_retry_after_seconds_falls_back_when_the_header_is_missing_or_junk() -> None:
    assert retry_after_seconds(httpx.Response(429, headers={"retry-after": "12"}), 60.0) == 12.0
    assert retry_after_seconds(httpx.Response(429), 60.0) == 60.0
    assert retry_after_seconds(httpx.Response(429, headers={"retry-after": "soon"}), 60.0) == 60.0


# --- --collect-only: bank what is already paid for, submit nothing -------------


def _stopped_run(tmp_path: Path) -> CrashOnSecondCreate:
    """A run whose first chunk got a batch and whose second chunk never did."""
    api = CrashOnSecondCreate()
    with pytest.raises(SubmitUncertain):
        _judge(tmp_path, api, chunk=2).run(
            [_pair(i) for i in range(4)], sample=4, stratify=["month"], seed=1
        )
    assert len(api.created) == 1
    api.crash = False
    return api


def test_collect_only_collects_pending_batches_and_submits_nothing(tmp_path: Path) -> None:
    api = _stopped_run(tmp_path)
    summary = _judge(tmp_path, api, chunk=2).run(
        [], sample=0, stratify=["month"], seed=1, collect_only=True
    )
    assert len(api.created) == 1  # the unsubmitted chunk stays unsubmitted
    assert summary["submitted"] == 0 and summary["succeeded"] == 2
    keys = {json.loads(line)["key"] for line in (tmp_path / "cache.jsonl").open()}
    submitted = {r["custom_id"] for r in api.created[0]["requests"]}
    assert {custom_id_of(k) for k in keys} == submitted
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["run"]["completed"] is False  # the rest can still be submitted later
    assert all(b["processed"] for b in state["batches"])


def test_collect_only_is_idempotent_and_reads_no_pairs(tmp_path: Path) -> None:
    api = _stopped_run(tmp_path)
    _judge(tmp_path, api, chunk=2).run([], sample=0, stratify=["month"], seed=1, collect_only=True)
    rows = len(list((tmp_path / "cache.jsonl").open()))
    summary = _judge(tmp_path, api, chunk=2).run(
        [], sample=0, stratify=["month"], seed=1, collect_only=True
    )
    assert summary["succeeded"] == 0  # nothing left unprocessed
    assert len(list((tmp_path / "cache.jsonl").open())) == rows


# --- A7: day filter ------------------------------------------------------


def _day_pairs() -> list[dict]:
    return [
        {"key": "a|000660", "day": "2026-03-17"},
        {"key": "b|005930", "day": "2026-03-18"},
        {"key": "c|005930", "day": "2026-03-19"},
        {"key": "d|005930", "day": "2026-04-01"},
    ]


def test_select_days_range_is_inclusive() -> None:
    kept = judge_batch.select_days(_day_pairs(), "2026-03-18..2026-03-19")
    assert [p["key"] for p in kept] == ["b|005930", "c|005930"]


def test_select_days_comma_list_keeps_file_order() -> None:
    kept = judge_batch.select_days(_day_pairs(), "2026-04-01,2026-03-17")
    assert [p["key"] for p in kept] == ["a|000660", "d|005930"]


def test_select_days_rejects_a_pairs_row_without_a_day() -> None:
    with pytest.raises(KeyError):
        judge_batch.select_days([{"key": "a|000660"}], "2026-03-17")


def test_select_days_can_select_nothing() -> None:
    assert judge_batch.select_days(_day_pairs(), "2026-05-01") == []


def test_cache_row_valid_rejects_an_empty_row() -> None:
    """`cache.get(key, {})` must not report an unjudged pair as judged.

    Without this guard the empty row misses `input_hash`, falls into the legacy
    branch, and hashes to the pair's own current input -- so every pair with no
    cache row at all looked cached.
    """
    pair = {
        "key": "k",
        "code": "005930",
        "name": "삼성전자",
        "market": "KOSPI",
        "sector": "반도체",
        "title": "제목",
        "body": "본문",
        "category": "",
    }
    assert cache_row_valid({}, pair, DEFAULT_MODEL) is False
    assert (
        cache_row_valid(
            {"input_hash": pair_input_hash(pair, DEFAULT_MODEL)},
            pair,
            DEFAULT_MODEL,
        )
        is True
    )


# --- 이식하며 더한 것: Haiku 5.5 요청 모양, system 캐싱, 실측 usage --------------------------


def test_haiku55_request_has_no_temperature_thinking_disabled_and_addendum() -> None:
    p = _pair(1)
    req = build_request(p, "claude-haiku-5-5", addendum="보충")["params"]
    system, user = build_prompt(code=p["code"], name=p["name"], market=p["market"], sector=p["sector"],
                                title=p["title"], body=p["body"], category=p["category"])
    assert "temperature" not in req
    assert req["thinking"] == {"type": "disabled"}
    assert req["system"] == [{"type": "text", "text": system + "\n\n보충", "cache_control": {"type": "ephemeral"}}]
    assert req["messages"] == [{"role": "user", "content": user}]


def test_input_hash_is_over_the_plain_system_so_old_cache_rows_stay_valid() -> None:
    """캐싱은 보내는 모양만 바꾼다 — 1단계 캐시 행(평문 system 으로 보낸 요청)의 해시와 같아야 한다."""
    p = _pair(1)
    cached = build_request(p, "claude-haiku-5-5", addendum="보충")
    plain = build_request(p, "claude-haiku-5-5", addendum="보충", cache_system=False)
    assert request_input_hash(cached) == request_input_hash(plain) == pair_input_hash(p, "claude-haiku-5-5", "보충")
    assert pair_input_hash(p, "claude-haiku-5-5", "보충") != pair_input_hash(p, "claude-haiku-5-5")
    row = {"key": p["key"], "input_hash": request_input_hash(plain)}
    assert cache_row_valid(row, p, "claude-haiku-5-5", "보충")
    assert not cache_row_valid(row, p, "claude-haiku-5-5")  # 보충이 다르면 다른 입력


USAGE = {"input_tokens": 1030, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1044,
         "output_tokens": 97}


def _msg_usage(payload: dict | str, usage: dict = USAGE) -> dict:
    return {**_msg(payload), "usage": usage}


def test_collect_records_billed_usage_per_result_in_state_and_summary(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(3)]
    cids = sorted(custom_id_of(p["key"]) for p in pairs)

    def result_for(cid: str) -> dict:
        if cid == cids[0]:  # 스키마 오류도 과금된다
            return {"type": "succeeded", "message": _msg_usage({**FEATS, "persistence": "forever"})}
        if cid == cids[1]:  # 첫 요청: 캐시 쓰기
            return {"type": "succeeded", "message": _msg_usage(FEATS, {**USAGE, "cache_creation_input_tokens": 1044,
                                                                          "cache_read_input_tokens": 0})}
        return {"type": "succeeded", "message": _msg_usage(FEATS)}

    api = FakeAPI(result_for)
    summary = _judge(tmp_path, api).run(pairs, sample=10, stratify=["month"], seed=1)
    assert (summary["succeeded"], summary["failed"]) == (2, 1)
    u = summary["usage"]
    assert u["input_tokens"] == 3 * 1030 and u["output_tokens"] == 3 * 97
    assert u["cache_creation_input_tokens"] == 1044 and u["cache_read_input_tokens"] == 2 * 1044
    assert u["results_with_usage"] == 3 and u["results_with_cache_read"] == 2
    assert summary["measured_usd"] == pytest.approx(usage_cost(u, "claude-haiku-4-5"))
    rows = [json.loads(line) for line in (tmp_path / "state.usage.jsonl").open()]
    assert len(rows) == 3 and {r["batch_id"] for r in rows} == {"b1"}
    assert set(rows[0]["usage"]) == set(judge_batch.USAGE_FIELDS)
    (batch,) = json.loads((tmp_path / "state.json").read_text())["batches"]
    assert batch["usage"]["cache_read_input_tokens"] == 2 * 1044
    assert batch["measured_usd"] == pytest.approx(summary["measured_usd"])


def test_usage_cost_uses_batch_prices_and_cache_multipliers() -> None:
    # claude-haiku-5-5 batch: $0.05/M input, $0.25/M output; cache read 0.1x, write 1.25x
    u = {"input_tokens": 1_000_000, "cache_creation_input_tokens": 1_000_000,
         "cache_read_input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert usage_cost(u, "claude-haiku-5-5") == pytest.approx(0.05 + 0.05 * 1.25 + 0.05 * 0.1 + 0.25)


def test_resume_sends_the_stored_runs_addendum_not_todays_flag(tmp_path: Path) -> None:
    pairs = [_pair(i) for i in range(4)]
    api = CrashOnSecondCreate()
    j = _judge(tmp_path, api, chunk=2)
    j.addendum = "r3"
    with pytest.raises(SubmitUncertain):
        j.run(pairs, sample=4, stratify=["month"], seed=1)
    api.crash = False
    j2 = _judge(tmp_path, api, chunk=2)  # 오늘의 CLI 는 보충 없음
    j2.run(pairs, sample=4, stratify=["month"], seed=1)
    second = api.created[1]["requests"][0]["params"]["system"][0]["text"]
    assert second.endswith("\n\nr3")
    assert json.loads((tmp_path / "state.json").read_text())["run"]["system_addendum"] == "r3"


def test_prefilter_is_off_by_default_and_a_parameter(tmp_path: Path) -> None:
    pairs = [{**_pair(i), "title": "[특징주] 급등", "related_n": 1} for i in range(2)] + [_pair(9)]
    s = _judge(tmp_path / "a", FakeAPI()).run(pairs, sample=10, stratify=["month"], seed=1, dry_run=True)
    assert (s["sample"], s["prefiltered"]) == (3, 0)
    j = _judge(tmp_path / "b", FakeAPI())
    j.prefilter = PairFilter()
    s = j.run(pairs, sample=10, stratify=["month"], seed=1, dry_run=True)
    assert (s["sample"], s["prefiltered"]) == (1, 2)
