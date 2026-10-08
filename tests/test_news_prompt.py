"""판정 프롬프트 — daytrade-it 운영 v2 의 바이트 동일 사본인지, 1단계 판정 캐시 해시를 재현하는지.

문자열 한 글자가 바뀌면 판정 캐시 36,171 행이 전부 "다른 입력" 이 된다. 고정값은 이식 시점(2026-10-08)에
daytrade-it 원본(@ ba0662a)에서 잰 sha256 이다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from swing_it.news.prompt import (
    _MAX_OUTPUT_TOKENS,
    _SYSTEM,
    _USER_TEMPLATE,
    DEFAULT_MODEL,
    MATERIAL_TYPES,
    RUBRIC,
    ArticleFeatures,
    SchemaError,
    build_prompt,
    judgment_input_hash,
    parse_features,
    read_addendum,
    system_with_addendum,
)

R3 = Path(__file__).resolve().parents[1] / "research" / "logs" / "news_persistence_swing" / "prompt_r3.txt"

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


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def test_prompt_strings_are_byte_identical_to_the_live_v2_prompt() -> None:
    assert _sha(_SYSTEM) == "5fd6eb64ceb03dc3e8de232de5c5fa241132176a774983f09f520b566428c851"
    assert _sha(_USER_TEMPLATE) == "11278821c2f537177b891e22e4e2f62c675190d3f482f02db996c96e247213cf"
    assert _sha(RUBRIC) == "c5c0df271aa478a69b7fc88942470dfed0d8d8d071ebaa497e99bd8d0ff30313"
    assert MATERIAL_TYPES == ("수주", "실적", "신약", "증설", "정책", "테마", "루머", "지분", "기타")
    assert _MAX_OUTPUT_TOKENS == 400
    assert DEFAULT_MODEL == "claude-haiku-4-5-20251001"


def test_prereg_addendum_r3_is_the_committed_file() -> None:
    assert _sha(R3.read_text(encoding="utf-8")) == "0417bf688a11e7c14ccbbf1f8a012f56a78ef15e5409b2a8c767cf885a0bd67f"


def test_reproduces_a_real_stage1_cache_row_hash() -> None:
    """2025-08-29 롯데지주 쌍 — 1단계 판정 캐시(persistence_v2.jsonl)에 있는 실제 행의 input_hash."""
    system, user = build_prompt(
        code="004990",
        name="롯데지주",
        market="거래소",
        sector="금융",
        title="Lotus Technology 2분기 매출 1억2550만 미 달러 >LOT",
        body="(MORE TO FOLLOW) Dow Jones Newswires\n\nAugust 29, 2025 06:00 ET (10:00 GMT)",
        category=None,
    )
    system = system_with_addendum(system, read_addendum(R3))
    assert (
        judgment_input_hash("claude-haiku-5-5", system, user)
        == "e8f1fcd23c392e5ed010d4f41d87ebfaf37327d371b6f748980ebee3790d4c53"
    )


def test_build_prompt_renders_fallbacks_and_strips() -> None:
    system, user = build_prompt(
        code=" 005930 ", name=" 삼성전자 ", market=None, sector=None, title=" 제목 ", body=" 본문 ", category=None
    )
    assert system == _SYSTEM
    assert user.startswith("다음은 삼성전자(005930, -, 업종: -) 관련으로 분류된 기사다.\n\n제목: 제목\n본문(요약): 본문\n분류: \n")
    assert "수주 / 실적 / 신약" in user


def test_addendum_join_and_none() -> None:
    assert system_with_addendum("S", None) == "S"
    assert system_with_addendum("S", "A") == "S\n\nA"
    assert read_addendum(None) is None


def test_parse_features_accepts_fenced_json_and_scores() -> None:
    f = parse_features("```json\n" + str(FEATS).replace("'", '"').replace("False", "false") + "\n```")
    assert f == ArticleFeatures(**FEATS)
    assert f.score() == pytest.approx(2 / 3 * (0.5 + 0.5 * f.magnitude()))
    assert ArticleFeatures(**{**FEATS, "subject": "mention"}).score() == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        {**FEATS, "persistence": "forever"},
        {**FEATS, "specificity": 9},
        {**FEATS, "reports_price_move": "maybe"},
        {k: v for k, v in FEATS.items() if k != "surprise"},
    ],
)
def test_parse_features_rejects_schema_violations(payload: dict) -> None:
    with pytest.raises(SchemaError):
        parse_features(payload)


def test_parse_features_rejects_no_json() -> None:
    with pytest.raises(SchemaError):
        parse_features("no json here")


def test_judgment_input_hash_covers_model_system_and_user() -> None:
    h = judgment_input_hash("m", "sys", "user")
    assert len(h) == 64 and h == judgment_input_hash("m", "sys", "user")
    assert len({h, judgment_input_hash("m2", "sys", "user"), judgment_input_hash("m", "sys2", "user"),
                judgment_input_hash("m", "sys", "user2")}) == 4
    # unambiguous framing: moving text across the boundary changes the hash
    assert judgment_input_hash("m", "ab", "c") != judgment_input_hash("m", "a", "bc")
    assert h != hashlib.sha256(b"msysuser").hexdigest()
