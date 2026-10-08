"""LLM 판정 전 결정론 필터 — (기사, 종목) 쌍 중 판정할 가치가 없는 것을 돈 쓰기 전에 뺀다.

**출처:** daytrade-it ``src/daytrade_it/application/news/mention_prefilter.py`` (브랜치
``research/news-persistence-judge`` @ ba0662a) 의 세 규칙 + 스윙 전용 태그. 태그 목록·정규식·
판정 순서는 그대로 옮겼고, 바뀐 것은 **스위치를 어디서 읽나** 하나다:

- daytrade-it 은 ``DAYTRADE_NEWS_PREFILTER*`` 환경변수로 켜고 임계값을 바꿨다(운영 데몬과 백테스트가
  같은 값을 읽게 하려고). swing-it 에는 운영 데몬이 없으므로 환경변수를 읽지 않는다 — 모든 규칙이
  :class:`PairFilter` 의 **명시 파라미터**다. 스윙용으로 규칙을 다시 볼 때 무엇을 바꿨는지가 코드·
  커맨드라인에 남는다.
- :data:`STAGE1` (= ``PairFilter()`` 기본값)이 스윙 2호 1단계 쌍 파일을 만든 조합 그대로다:
  운영 사전필터 3규칙(roundup 태그, 팬아웃 > 11, 지수 시황 ≥ 5) **그리고** 스윙 전용 태그 14종.
  (옛 경로: ``build_universe_events --prefilter`` → ``pairs_2023-03_2026-02.jsonl`` 670,463쌍 →
  ``swing_skip_tag is None`` 로 거른 ``_swing.jsonl`` 637,343쌍. 2026-10-08 재확인: 옛 파일에
  운영 3규칙 위반 0건, 스윙 태그 필터 결과 637,343 일치.)

규칙의 측정 근거(daytrade-it 원문 요약):

1. **roundup 태그** — 시황·표 칼럼(``[표]``·``[마감시황]``·``[0519마감체크]`` …). 97,721 판정 쌍에서
   BUY 비율 ≤0.22% (기준 17.30%).
2. **팬아웃** — 토스가 한 기사에 태그한 종목 수. 1종목 기사 BUY 41.3%, 20종목 초과 0.07%.
   > 11 이면 뺀다(3규칙 합계: 쌍 29.00% 스킵, BUY 재현율 손실 0.373%).
3. **지수 시황** — 제목에 ``코스피``/``코스닥`` 이 있고 팬아웃 ≥ 5 이면 지수 해설.
4. **스윙 전용 태그** — 장중 판정에선 일부러 안 거른 ``[특징주]``·``[장중수급포착]`` 등. 스윙 사건
   ("주가 보도가 아닌 main·호재·비반복")은 이 태그에 거의 없다: 2026-03~09 Haiku 4.5 판정 115,197쌍에서
   사건(기준 16.3%) 비율 ≤1% · n≥50 인 14종(합계 약 8.5천 쌍, 잃는 사건 ~5건).

규칙은 **제목과 팬아웃만** 읽는다 — 본문은 운영(전문)과 아카이브(약 100자 요약)가 달라서, 본문 규칙은
두 세계에서 다른 쌍을 거른다. 순수 모듈: I/O·시계·모델 호출 없음.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

# Syndicated market-roundup / price-table columns. Each measured at <= 0.22% BUY
# rate against a 17.30% base, over 97,721 judged pairs.
ROUNDUP_TAGS: frozenset[str] = frozenset(
    {
        "표",  # n=7395, 0.00% BUY -- price/flow tables
        "개장시황",  # n=646,  0.00%
        "마감시황",  # n=631,  0.16%
        "오늘의 IR",  # n=434,  0.00%
        "오늘증시",  # n=253,  0.00%
        "티타임",  # n=246,  0.00%
        "食전食후",  # n=183,  0.00%
        "오늘의 증시일정",  # n=126,  0.00%
        "연합뉴스 이 시각 헤드라인",  # n=124,  0.00%
        "AI MY 증시전망",  # n=108,  0.00%
        "장중시황",  # n=54,   0.00%
        "ETF 시황",  # n=45,   0.00%
        "주간수급리포트",  # n=38,   0.00%
        "주간증시전망",  # n=32,   0.00%
    }
)

# Dated variants of the same columns: "[0519마감체크]", "[0810개장체크]", ...
DATED_ROUNDUP_RE = re.compile(r"^\d{4}(?:마감체크|개장체크|주요일정|장마감|마감|개장)$")

# A leading bracketed tag, in the half-width, full-width and 【】 forms Korean
# wires use interchangeably.
BRACKET_RE = re.compile(r"^\s*[\[\(（【]([^\]\)）】]{1,24})[\]\)）】]")

#: Articles tagged with more than this many companies are dropped (0 disables).
FANOUT_MAX = 11

#: Index names; a title naming one on an article tagged to INDEX_FANOUT_MIN+ companies is index commentary.
INDEX_TOKENS: frozenset[str] = frozenset({"코스피", "코스닥"})

#: Fan-out floor for the index rule (0 disables).
INDEX_FANOUT_MIN = 5

# 스윙 전용(2026-10-08, 스윙 2호) — 장중 판정과 **따로** 둔다: [특징주]·[장중수급포착] 은 장중 BUY 의
# 실제 원천이라 운영 필터는 일부러 거르지 않았다. 측정: 위 모듈 docstring 4.
SWING_SKIP_TAGS: frozenset[str] = frozenset(
    {
        "장중수급포착",  # n=2890, 사건 0
        "특징주",  # n=2461, 3
        "증시키워드",  # n=531, 0
        "마켓뷰",  # n=169, 0
        "ET특징주",  # n=158, 1
        "급등락주 짚어보기",  # n=158, 0
        "테마시황",  # n=134, 0
        "오늘의 주요공시",  # n=121, 0
        "기고",  # n=64, 0
        "전국 주요 신문 톱뉴스",  # n=59, 0
        "정책의속살",  # n=55, 0
        "MBN골드",  # n=55, 0
        "핫타임",  # n=55, 0
        "이슈플러스",  # n=53, 0
    }
)


@dataclass(frozen=True)
class PairFilter:
    """판정 전 필터 설정. 기본값 = 스윙 2호 1단계(:data:`STAGE1`).

    Attributes:
        roundup: roundup 태그 규칙(1) 켜기.
        fanout_max: 팬아웃 상한(2). 이보다 많은 종목에 태그된 기사는 뺀다. 0 = 끔.
        index_fanout_min: 지수 시황 규칙(3)의 팬아웃 하한. 0 = 끔.
        swing_tags: 스윙 전용 태그 규칙(4) 켜기. (roundup 태그와 독립 — 옛 ``swing_skip_tag`` 는
            roundup 도 함께 거렀지만 여기선 규칙 1 이 맡는다. 기본 조합의 합집합은 같다.)
        swing_tag_set: 규칙 4 가 거르는 태그 집합.
    """

    roundup: bool = True
    fanout_max: int = FANOUT_MAX
    index_fanout_min: int = INDEX_FANOUT_MIN
    swing_tags: bool = True
    swing_tag_set: frozenset[str] = SWING_SKIP_TAGS

    def describe(self) -> dict:
        """원장·로그용 — 무엇을 켰는지 그대로 남긴다."""
        return {
            "roundup": self.roundup,
            "fanout_max": self.fanout_max,
            "index_fanout_min": self.index_fanout_min,
            "swing_tags": self.swing_tags,
            "swing_tag_set": sorted(self.swing_tag_set) if self.swing_tag_set != SWING_SKIP_TAGS else "default",
        }


#: 스윙 2호 1단계 쌍 파일을 만든 조합. 바꾸면 새 쌍 집합이다(사전등록 수정 2).
STAGE1 = PairFilter()
#: 아무것도 거르지 않는다(측정·비교용).
NO_FILTER = PairFilter(roundup=False, fanout_max=0, index_fanout_min=0, swing_tags=False)


def bracket_tag(title: str | None) -> str | None:
    """제목 맨 앞 괄호 태그(공백 제거), 없으면 None."""
    match = BRACKET_RE.match(title or "")
    return None if match is None else match.group(1).strip()


def roundup_tag(title: str | None) -> str | None:
    """The roundup column tag this title belongs to, or None."""
    tag = bracket_tag(title)
    if tag is not None and (tag in ROUNDUP_TAGS or DATED_ROUNDUP_RE.match(tag)):
        return tag
    return None


def index_token(title: str | None, related_count: int | None, floor: int) -> str | None:
    """The index name that makes this a market-commentary mention, or None."""
    if floor <= 0 or related_count is None or related_count < floor:
        return None
    text = title or ""
    for token in sorted(INDEX_TOKENS):
        if token in text:
            return token
    return None


def swing_skip_tag(title: str | None, tags: frozenset[str] = SWING_SKIP_TAGS) -> str | None:
    """스윙 판정에서 뺄 태그(장중 roundup 포함), 아니면 None — daytrade-it 원본과 같은 함수."""
    tag = roundup_tag(title)
    if tag is not None:
        return tag
    tag = bracket_tag(title)
    return tag if tag is not None and tag in tags else None


def skip_reason(title: str | None, related_count: int | None, cfg: PairFilter = STAGE1) -> str | None:
    """이 쌍을 뺄 이유, 아니면 None(판정한다).

    ``"roundup:<tag>"`` · ``"fanout:<n>"`` · ``"index:<token>:<n>"`` · ``"swing:<tag>"`` — 가장 좁은 규칙부터
    본다(운영 순서 그대로, 스윙 태그는 그 뒤). ``related_count=None`` (팬아웃을 모름)이면 팬아웃·지수 규칙은
    절대 발동하지 않는다.
    """
    if cfg.roundup and (tag := roundup_tag(title)) is not None:
        return f"roundup:{tag}"
    if cfg.fanout_max > 0 and related_count is not None and related_count > cfg.fanout_max:
        return f"fanout:{related_count}"
    if (token := index_token(title, related_count, cfg.index_fanout_min)) is not None:
        return f"index:{token}:{related_count}"
    if cfg.swing_tags and (tag := bracket_tag(title)) is not None and tag in cfg.swing_tag_set:
        return f"swing:{tag}"
    return None


def keep_pair(title: str | None, related_count: int | None, cfg: PairFilter = STAGE1) -> bool:
    """True 면 판정한다. :func:`skip_reason` 이 None 인 것과 같다."""
    return skip_reason(title, related_count, cfg) is None


# --- CLI 공용(쌍 생성기·판정기) ------------------------------------------------------------


def filter_from_args(args: argparse.Namespace) -> PairFilter:
    return PairFilter(
        roundup=not args.no_roundup,
        fanout_max=args.fanout_max,
        index_fanout_min=args.index_fanout_min,
        swing_tags=not args.no_swing_tags,
    )


def add_filter_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("판정 전 필터 (기본 = 스윙 2호 1단계)")
    g.add_argument("--no-roundup", action="store_true", help="roundup 태그 규칙 끔")
    g.add_argument("--fanout-max", type=int, default=STAGE1.fanout_max, help="팬아웃 상한, 0 = 끔")
    g.add_argument("--index-fanout-min", type=int, default=STAGE1.index_fanout_min, help="지수 시황 하한, 0 = 끔")
    g.add_argument("--no-swing-tags", action="store_true", help="스윙 전용 태그 규칙 끔")
