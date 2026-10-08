"""판정 원장 — judge 정의·이름 충돌·키 풀기·행 정리. DB 없이(가짜 커서) 검사한다.

실제 DB 이전 결과(2026-10-09): v2-haiku45 157,834행(성공 156,605 + 오류 1,229), v2r3-haiku55 36,214
(36,171 + 43), v2-haiku55 500(496 + 4). 미해결 키 0. 재실행은 전부 unchanged(멱등).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from swing_it.news.judge import judge_spec, request_params
from swing_it.news.judgments_db import (
    KNOWN_JUDGES,
    V2_HAIKU45,
    V2_HAIKU55,
    V2R3_HAIKU55,
    JudgeCollision,
    JudgmentLedger,
    UnmappedKeys,
    dedupe_rows,
    judgment_row,
    register_judge,
    resolve_keys,
    spec_for_request,
    write_judgments,
)
from swing_it.news.prompt import _SYSTEM, ADDENDUM_R3, read_addendum
from swing_it.news.sweep import pair_key_article

AT = dt.datetime(2026, 10, 9, tzinfo=dt.UTC)


def test_judge_definitions_are_pinned() -> None:
    assert V2_HAIKU45.params == {"max_tokens": 400, "temperature": 0}
    assert V2_HAIKU45.owner_repo == "daytrade-it" and V2_HAIKU45.model == "claude-haiku-4-5-20251001"
    assert V2R3_HAIKU55.system_prompt == _SYSTEM + "\n\n" + ADDENDUM_R3.strip()
    assert V2R3_HAIKU55.params == {"max_tokens": 400, "thinking": {"type": "disabled"}}
    assert V2R3_HAIKU55.system_sha256 == hashlib.sha256(V2R3_HAIKU55.system_prompt.encode()).hexdigest()
    assert V2_HAIKU55.system_prompt == _SYSTEM
    assert set(KNOWN_JUDGES) == {"v2-haiku45", "v2r3-haiku55", "v2-haiku55"}


def test_r3_constant_is_the_committed_prereg_file() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "research" / "logs" / "news_persistence_swing" / "prompt_r3.txt"
    assert ADDENDUM_R3 == path.read_text(encoding="utf-8")
    assert read_addendum(path) == ADDENDUM_R3.strip()


def test_judge_cli_definitions_resolve_to_the_registered_names() -> None:
    assert judge_spec("claude-haiku-5-5", ADDENDUM_R3.strip()) is V2R3_HAIKU55
    assert judge_spec("claude-haiku-5-5", None) is V2_HAIKU55
    assert judge_spec("claude-haiku-4-5-20251001", None) is V2_HAIKU45
    with pytest.raises(JudgeCollision):
        judge_spec("claude-haiku-5-5", "다른 보충")
    with pytest.raises(JudgeCollision):
        spec_for_request("claude-haiku-5-5", _SYSTEM, {**request_params("claude-haiku-5-5"), "max_tokens": 500})


class FakeCursor:
    def __init__(self, db: dict) -> None:
        self.db = db
        self._rows: list = []

    def execute(self, sql: str, args=()) -> None:
        if sql.startswith("SELECT model"):
            row = self.db.get(args[0])
            self._rows = [row] if row else []
        elif sql.startswith("INSERT INTO judges"):
            judge, model, sha, _system, user, params = args[:6]
            self.db[judge] = (model, sha, user, json.loads(params))
        else:
            raise AssertionError(sql)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeCon:
    def __init__(self) -> None:
        self.db: dict = {}
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.db)

    def commit(self) -> None:
        self.commits += 1


def test_register_judge_inserts_once_and_rejects_a_changed_definition() -> None:
    con = FakeCon()
    assert register_judge(con, V2R3_HAIKU55) is True
    assert register_judge(con, V2R3_HAIKU55) is False  # 같은 정의 — 그대로
    m, sha, user, params = con.db["v2r3-haiku55"]
    con.db["v2r3-haiku55"] = (m, sha, user, {**params, "max_tokens": 500})
    with pytest.raises(JudgeCollision):
        register_judge(con, V2R3_HAIKU55)


def test_resolve_keys_never_guesses() -> None:
    kmap = {pair_key_article("toss:aaaaaaaaaaaa"): "toss:aaaaaaaaaaaa"}
    good = f"{pair_key_article('toss:aaaaaaaaaaaa')}|005930"
    found, missing = resolve_keys([good, f"{pair_key_article('toss:bbbbbbbbbbbb')}|005930", "junk"], kmap)
    assert found == {good: ("toss:aaaaaaaaaaaa", "005930")}
    assert len(missing) == 2


def _row(output=None, error=None, at=AT):
    return judgment_row("toss:a", "005930", "v2r3-haiku55", "h", output=output, error=error, judged_at=at,
                        judged_at_exact=True, usage={"input_tokens": 1, "cache_read_input_tokens": 2,
                                                     "output_tokens": 3})


def test_judgment_row_maps_api_usage_names() -> None:
    r = _row(output={"x": 1})
    assert (r["input_tokens"], r["cache_creation_tokens"], r["cache_read_tokens"], r["output_tokens"]) == (1, None, 2, 3)


def test_dedupe_prefers_success_then_latest() -> None:
    ok, err = _row(output={"x": 1}), _row(error="schema: x")
    assert dedupe_rows([ok, err]) == [ok]
    assert dedupe_rows([err, ok]) == [ok]
    later = _row(output={"x": 2})
    assert dedupe_rows([ok, later]) == [later]


def test_write_rejects_rows_with_both_or_neither() -> None:
    pytest.importorskip("psycopg2")
    with pytest.raises(ValueError):
        write_judgments(FakeCon(), [{**_row(output={"x": 1}), "error": "e"}])


def test_ledger_refuses_judges_owned_by_another_repo() -> None:
    with pytest.raises(PermissionError):
        JudgmentLedger(FakeCon(), V2_HAIKU45)


def test_ledger_refuses_unmapped_keys_before_writing() -> None:
    con = FakeCon()
    led = JudgmentLedger(con, V2R3_HAIKU55, kmap={})
    with pytest.raises(UnmappedKeys):
        led.record([{"key": "u|005930", "input_hash": "h", "output": {}, "judged_at": AT, "judged_at_exact": True}])
    assert con.db == {}  # 등록도 안 했다
