"""의미 정답지 도구 — 요청 모양·사건 정의·LLM 없는 기준선. 네트워크 없음."""

from __future__ import annotations

from swing_it.news import labelset
from swing_it.news.prompt import _SYSTEM, RUBRIC, build_prompt

P = {"key": "k|005930", "code": "005930", "name": "삼성전자", "market": "거래소", "sector": "전기/전자",
     "title": "삼성전자, 美 공장 증설", "body": "삼성전자가 증설한다.", "category": None}


def test_haiku_request_v2_is_the_live_prompt_with_thinking_off(tmp_path, monkeypatch) -> None:
    req = labelset.haiku_request(P, "v2")
    system, user = build_prompt(code=P["code"], name=P["name"], market=P["market"], sector=P["sector"],
                                title=P["title"], body=P["body"], category=None)
    assert req == {"model": "claude-haiku-5-5", "max_tokens": 400, "system": system,
                   "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": user}]}


def test_haiku_request_file_variant_appends_stripped_addendum(tmp_path, monkeypatch) -> None:
    (tmp_path / "r9.txt").write_text("\n보충 정의\n", encoding="utf-8")
    monkeypatch.setattr(labelset, "PROMPTS", tmp_path)
    assert labelset.haiku_request(P, "r9")["system"] == _SYSTEM + "\n\n보충 정의"


def test_event_definition_matches_the_prereg() -> None:
    ev = {"subject": "main", "sentiment_direction": 1, "freshness": "new", "reports_price_move": False}
    assert labelset.is_event(ev)
    assert not labelset.is_event({**ev, "freshness": "repeat"})
    assert not labelset.is_event({**ev, "reports_price_move": True})
    assert not labelset.is_event({**ev, "subject": "partial"})
    assert not labelset.is_event({**ev, "sentiment_direction": 0})


def test_rules_baseline_uses_text_only() -> None:
    lab = labelset.rules_label(P)
    assert lab["subject"] == "main" and lab["persistence"] == "multi_quarter" and lab["sentiment_direction"] == 1
    assert labelset.rules_label({**P, "title": "삼성전자 상한가"})["reports_price_move"] is True


def test_teacher_rubric_is_the_shared_constant() -> None:
    assert labelset.RUBRIC is RUBRIC
