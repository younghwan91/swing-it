"""옛 배치 캐시 이전 — 배치 귀속(judged_at)·오류 행 규칙. DB 없이."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

from swing_it.news.import_legacy import attribute, batch_index, build_rows, pair_hash
from swing_it.news.judgments_db import V2_HAIKU45, V2R3_HAIKU55
from swing_it.news.prompt import build_prompt, judgment_input_hash

T0 = "2026-09-15T19:26:56+00:00"
T1 = "2026-09-15T23:45:33+00:00"
MTIME = dt.datetime(2026, 9, 17, tzinfo=dt.UTC)


def _state(tmp_path: Path) -> Path:
    s = {
        "batches": [
            {"id": "b1", "submitted_at": T1, "custom_ids": {"a_1": "a|1", "b_1": "b|1"},
             "input_hashes": {"a_1": "ha", "b_1": "hb"}},
        ],
        "history": [{"batches": [
            {"id": "b0", "submitted_at": T0, "custom_ids": {"a_1": "a|1", "c_1": "c|1"},
             "input_hashes": {"a_1": "ha", "c_1": "hc"}},
        ]}],
    }
    p = tmp_path / "s.state.json"
    p.write_text(json.dumps(s))
    return p


def _err(key: str, batch: str, kind: str) -> dict:
    if kind == "credit":
        return {"key": key, "reason": "errored", "batch_id": batch,
                "detail": {"type": "error", "error": {"type": "invalid_request_error"}}}
    return {"key": key, "reason": "schema", "batch_id": batch, "detail": "persistence outside allowed set"}


def test_success_is_attributed_to_the_batch_where_it_did_not_error(tmp_path: Path) -> None:
    by_key, submitted = batch_index([_state(tmp_path)])
    assert submitted["b0"] < submitted["b1"]
    errored = {("a|1", "b0")}  # b0 에선 크레딧 부족, b1 에서 성공
    assert attribute("a|1", "ha", by_key, errored)[1] == "b1"
    assert attribute("a|1", "ha", by_key, set())[1] == "b0"  # 둘 다 가능하면 가장 이른 것
    assert attribute("a|1", "other-hash", by_key, set()) is None
    assert attribute("zz|1", "ha", by_key, set()) is None


def test_build_rows_rules(tmp_path: Path) -> None:
    by_key, submitted = batch_index([_state(tmp_path)])
    cache = [{"key": "a|1", "input_hash": "ha", "v2": {"score": 0.1}},
             {"key": "d|1", "input_hash": "hd", "v2": {"score": 0.0}}]  # 어느 상태에도 없다 → mtime
    errors = [_err("a|1", "b0", "credit"), _err("a|1", "old", "schema"),  # 성공이 있으니 오류는 빠진다
              _err("b|1", "b1", "credit"),  # 판정 시도가 아니다
              _err("c|1", "b0", "schema"), _err("c|1", "b0", "schema")]  # 같은 오류 두 줄 → 한 행
    resolved = {k: (f"toss:{k[0]}", "000001") for k in ("a|1", "b|1", "c|1", "d|1")}
    stats: Counter = Counter()
    rows = build_rows(V2_HAIKU45, cache, errors, resolved, by_key, submitted, MTIME, stats)
    ok = {r["article_id"]: r for r in rows if r["output"] is not None}
    bad = {r["article_id"]: r for r in rows if r["error"] is not None}
    assert set(ok) == {"toss:a", "toss:d"} and set(bad) == {"toss:c"}
    assert ok["toss:a"]["batch_id"] == "b1" and ok["toss:a"]["judged_at"] == dt.datetime.fromisoformat(T1)
    assert ok["toss:d"]["batch_id"] is None and ok["toss:d"]["judged_at"] == MTIME
    assert all(r["judged_at_exact"] is False and r["judge"] == "v2-haiku45" for r in rows)
    assert bad["toss:c"]["input_hash"] == "hc" and bad["toss:c"]["error"].startswith("schema:")
    assert stats["error_not_an_attempt_skipped"] == 2 and stats["error_superseded_by_success"] == 1
    assert stats["ok_judged_at_mtime"] == 1 and stats["ok_judged_at_batch"] == 1


def test_pair_hash_is_the_judge_definition_over_the_pair_as_judged() -> None:
    p = {"code": "005930", "name": "삼성전자", "market": "거래소", "sector": "전기/전자", "title": "t", "body": "b",
         "category": None}
    _, user = build_prompt(code="005930", name="삼성전자", market="거래소", sector="전기/전자", title="t", body="b",
                           category=None)
    assert pair_hash(V2R3_HAIKU55, p) == judgment_input_hash("claude-haiku-5-5", V2R3_HAIKU55.system_prompt, user)
