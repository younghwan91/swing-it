"""뉴스 판정 프롬프트 — daytrade-it 운영 v2 추출 스키마의 사본 + 스윙 보충(addendum).

**출처(바이트 동일 사본, 2026-10-08):** daytrade-it ``src/daytrade_it/infrastructure/external/
anthropic_client.py`` (브랜치 ``research/news-persistence-judge`` @ ba0662a) 의 ``_SYSTEM``·
``_USER_TEMPLATE``·``build_prompt``·``MATERIAL_TYPES``·``ArticleFeatures``·``parse_features``·
``SchemaError``·``_MAX_OUTPUT_TOKENS``, 그리고 ``application/news/universe.py`` 의
``judgment_input_hash``. 판정 캐시 행은 (모델, system, user) 해시로 검증되므로 **문자열이 한
글자라도 바뀌면 기존 판정 36,171 행이 전부 "다른 입력" 이 된다.** 고치지 말 것 —
``tests/test_news_prompt.py`` 가 해시 고정값으로 지킨다.

스윙 2호(사전등록 ``research/logs/news_persistence_swing/VERDICT.md``)의 판정 프롬프트는
운영 ``_SYSTEM`` 뒤에 ``"\\n\\n" + r3.txt.strip()`` 을 붙인 것이다(:func:`system_with_addendum`).
보충 파일은 ``data/eval/persistence_swing/prompts/`` (r1..r5; 사전등록이 고정한 것은 r3 —
같은 내용이 ``research/logs/news_persistence_swing/prompt_r3.txt`` 로 커밋돼 있다).

시세 판단 언어는 프롬프트에 없다 — LLM 은 기사 사실만 채우고, 점수는 코드(:meth:`ArticleFeatures.score`)가 낸다.
순수 모듈: I/O·네트워크 없음.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

PROMPT_VERSION = "v2"

# --- Fixed schema -----------------------------------------------------------

MATERIAL_TYPES: tuple[str, ...] = (
    "수주",
    "실적",
    "신약",
    "증설",
    "정책",
    "테마",
    "루머",
    "지분",
    "기타",
)
SUBJECT_LEVELS: tuple[str, ...] = ("main", "partial", "mention")
PERSISTENCE_LEVELS: tuple[str, ...] = ("one_off", "multi_quarter", "structural")
FRESHNESS_LEVELS: tuple[str, ...] = ("repeat", "follow_up", "new")
SCALE_LEVELS: tuple[str, ...] = ("unknown", "small", "medium", "large")

_CATEGORICAL = {
    "subject": SUBJECT_LEVELS,
    "material_type": MATERIAL_TYPES,
    "persistence": PERSISTENCE_LEVELS,
    "freshness": FRESHNESS_LEVELS,
    "scale_vs_size": SCALE_LEVELS,
}
_NUMERIC = {
    "specificity": (0, 1, 2, 3),
    "surprise": (0, 1, 2),
    "sentiment_direction": (-1, 0, 1),
    "sentiment_strength": (0, 1, 2, 3),
}
_BOOLEAN = ("reports_price_move",)

_PERSISTENCE_SCORE = {"one_off": 0.0, "multi_quarter": 1.0, "structural": 2.0}
_SCALE_SCORE = {"unknown": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
_MAGNITUDE_WEIGHTS = {"persistence": 0.5, "scale": 0.3, "specificity": 0.2}
_MAGNITUDE_MAX = (
    _MAGNITUDE_WEIGHTS["persistence"] * max(_PERSISTENCE_SCORE.values())
    + _MAGNITUDE_WEIGHTS["scale"] * max(_SCALE_SCORE.values())
    + _MAGNITUDE_WEIGHTS["specificity"] * max(_NUMERIC["specificity"])
)


class SchemaError(ValueError):
    """LLM output didn't match the fixed schema -- caller's cue to retry."""


@dataclass(frozen=True)
class ArticleFeatures:
    """Fields extracted from one (article, company) pair."""

    subject: str
    reports_price_move: bool
    material_type: str
    persistence: str
    specificity: int
    freshness: str
    surprise: int
    sentiment_direction: int
    sentiment_strength: int
    scale_vs_size: str

    def magnitude(self) -> float:
        """How large/durable/specific the material is, in [0, 1]."""
        raw = (
            _MAGNITUDE_WEIGHTS["persistence"] * _PERSISTENCE_SCORE[self.persistence]
            + _MAGNITUDE_WEIGHTS["scale"] * _SCALE_SCORE[self.scale_vs_size]
            + _MAGNITUDE_WEIGHTS["specificity"] * self.specificity
        )
        return raw / _MAGNITUDE_MAX

    def score(self) -> float:
        """Signed material score in [-1, 1] -- the evaluated v2 rule (캐시 행의 ``score``)."""
        if self.subject == "mention":
            return 0.0
        s = self.sentiment_direction * (self.sentiment_strength / 3.0)
        s *= 0.5 + 0.5 * self.magnitude()
        if self.freshness == "repeat":
            s *= 0.5
        return max(-1.0, min(1.0, s))


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of `text` (tolerates fences/prose)."""
    start = text.find("{")
    if start < 0:
        raise SchemaError(f"No JSON object in response: {text[:120]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    result: dict[str, Any] = json.loads(text[start : i + 1])
                    return result
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"JSON parse failed: {exc}") from exc
    raise SchemaError(f"Unterminated JSON object: {text[:120]!r}")


def _as_int(name: str, value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} is not an int: {value!r}") from exc


def _as_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    raise SchemaError(f"{name} is not a bool: {value!r}")


def parse_features(raw: str | dict[str, Any]) -> ArticleFeatures:
    payload = raw if isinstance(raw, dict) else _extract_json_object(raw)

    for field in fields(ArticleFeatures):
        if field.name not in payload:
            raise SchemaError(f"Missing field: {field.name}")

    values: dict[str, Any] = {}
    for name, allowed_strings in _CATEGORICAL.items():
        value = str(payload[name]).strip()
        if value not in allowed_strings:
            raise SchemaError(f"{name} outside allowed set: {value!r}")
        values[name] = value
    for name, allowed_numbers in _NUMERIC.items():
        number = _as_int(name, payload[name])
        if number not in allowed_numbers:
            raise SchemaError(f"{name} outside allowed range: {number}")
        values[name] = number
    for name in _BOOLEAN:
        values[name] = _as_bool(name, payload[name])

    return ArticleFeatures(**values)


# --- Prompt (운영 v2 — 바이트 동일, 위 docstring 참조) ---------------------------------------

_SYSTEM = """너는 한국 상장사 기사에서 고정 스키마 필드를 채우는 추출기다.

규칙:
- 주어진 텍스트에 적힌 내용만 쓴다. 배경지식으로 보충하지 않는다.
- 이 작업에 시세 판단은 포함되지 않는다. 스키마에 없는 것은 답하지 않는다.
- 근거가 없으면 가장 보수적인 값(unknown 또는 0)을 고른다.
- 출력은 JSON 객체 하나뿐이다. 코드펜스·설명·주석을 붙이지 않는다."""

_USER_TEMPLATE = """다음은 {name}({code}, {market}, 업종: {sector}) 관련으로 분류된 기사다.

제목: {title}
본문(요약): {body}
분류: {category}

아래 필드를 채운 JSON 객체 하나만 출력하라. 모든 판단은 **{name}** 기준이다.

subject: 이 기사에서 {name}의 비중. main(기사의 주인공) / partial(여러 회사 중 하나로 비중 있게 다룸) / mention(업계 나열·지나가는 언급·다른 회사가 주인공)
reports_price_move: 기사가 {name}의 **이미 일어난** 주가 움직임(급등·급락·상한가·하한가·신고가·장 마감 시세)을 주요 내용으로 보도하는가. true / false
material_type: 재료 유형. 다음 중 하나 — {material_types}
persistence: 실적에 반영되는 지속성. one_off / multi_quarter / structural
specificity: 구체성. 0(방향만) / 1(대상·시기 중 하나) / 2(둘 다, 금액 없음) / 3(금액·기간·상대방 모두)
freshness: new(처음 알려짐) / follow_up(기존 건 진행 경과) / repeat(이미 알려진 내용의 재보도·해설·칼럼)
surprise: 이 내용이 기존 기대와 얼마나 다른가. 0(예상된 일·일정대로) / 1(다소 뜻밖) / 2(크게 뜻밖)
sentiment_direction: {name} 사업에 미치는 방향. -1 / 0 / 1
sentiment_strength: 그 방향의 강도. 0 / 1 / 2 / 3
scale_vs_size: {name} 규모 대비 사안의 크기. unknown / small / medium / large

규칙: 본문에 없는 내용은 추정하지 않는다. 인터뷰·칼럼·업계 동향 기사는 대체로 repeat 이고 강도가 낮다.

출력 형식 예시:
{{"subject":"main","reports_price_move":false,"material_type":"수주","persistence":"multi_quarter","specificity":3,"freshness":"new","surprise":1,"sentiment_direction":1,"sentiment_strength":2,"scale_vs_size":"medium"}}"""


def build_prompt(
    *,
    code: str,
    name: str,
    market: str | None,
    sector: str | None,
    title: str,
    body: str,
    category: str | None,
) -> tuple[str, str]:
    """(system, user) — 운영 v2 그대로. (운영의 ``full_body`` 그림자 변형 v4 는 옮기지 않았다.)"""
    user = _USER_TEMPLATE.format(
        code=code.strip(),
        name=name.strip(),
        market=market or "-",
        sector=sector or "-",
        title=title.strip(),
        body=body.strip(),
        category=category or "",
        material_types=" / ".join(MATERIAL_TYPES),
    )
    return _SYSTEM, user


#: 운영 v2 를 판정한 모델(Haiku 4.5 스냅샷). 스윙 2호는 ``claude-haiku-5-5`` 로 판정한다.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_OUTPUT_TOKENS = 400


# --- 스윙 보충(addendum) -------------------------------------------------------------------


def system_with_addendum(system: str, addendum: str | None) -> str:
    """운영 system 뒤에 보충을 붙인다 — ``judge_batch --system-addendum`` 과 같은 결합."""
    return system + "\n\n" + addendum if addendum else system


def read_addendum(path: str | Path | None) -> str | None:
    """보충 파일 → 결합할 문자열(앞뒤 공백 제거). ``None`` 이면 보충 없음(운영 v2 그대로)."""
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8").strip() or None


#: 의미 정답지(교사 라벨) 채점표 — daytrade-it ``scripts/eval/persistence_labelset.py`` 의 ``RUBRIC`` 바이트 사본.
RUBRIC = """채점 기준(정답지 작성용 — 신중히, 기사 텍스트만 근거로):
- subject: main = 기사의 주인공이 이 회사다(제목·첫 문장의 주어). partial = 여러 회사 중 하나로 실질적 비중.
  mention = 업종 나열·시황 기사·다른 회사가 주인공·지나가는 언급.
- persistence (실적에 반영되는 기간):
  · one_off = 한 번의 사건으로 끝난다 — 일회성 이익/손실, 단발 이벤트·행사, 주가·수급 시황, 루머, 단기 테마 언급,
    인사·소송 같은 실적 무관 소식, 단건 소규모 계약.
  · multi_quarter = 여러 분기에 걸쳐 매출·이익에 반영된다 — 다년 공급계약·수주잔고, 신제품/신약 출시 후 판매,
    증설 가동, 가격 인상, 분기를 넘는 실적 개선 추세.
  · structural = 사업 구조·경쟁 지위가 장기적으로 바뀐다 — 신시장 진입·플랫폼 전환, 대형 M&A·지배구조 재편,
    규제 체제 변화로 인한 장기 수혜/피해, 핵심 기술 확보.
  애매하면 낮은 쪽. 기사가 회사 실적과 무관하면 one_off.
- sentiment_direction: 이 회사 *사업*(주가 아님)에 미치는 방향 −1/0/1. 시황·주가 보도만 있으면 0.
- freshness: new = 처음 알려지는 사실, follow_up = 이미 알려진 건의 진행 경과, repeat = 재보도·해설·칼럼·요약.
- reports_price_move: 기사의 주요 내용이 이미 일어난 주가 움직임 보도인가."""


def judgment_input_hash(model: str, system: str, user: str) -> str:
    """sha256 over the judged input (model, system prompt, user prompt), length-framed.

    Judgment caches key rows by (pair key, this hash) so a pair whose prompt
    inputs changed (e.g. the body fallback) is re-judged instead of reusing a
    judgment of different text.
    """
    h = hashlib.sha256()
    for part in (model, system, user):
        data = part.encode()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()
