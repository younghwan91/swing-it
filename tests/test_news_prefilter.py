"""판정 전 필터 — daytrade-it ``test_mention_prefilter.py`` 의 경계값 이식 + 파라미터화.

기본값(:data:`STAGE1`)이 1단계 쌍 파일을 만든 조합 그대로인지가 핵심이다. 2026-10-08 실측: 옛 운영 필터만
거친 쌍 670,463 에 STAGE1 을 입히면 정확히 1단계 스윙 쌍 637,343 (키 집합 일치).
"""

from __future__ import annotations

import argparse

import pytest

from swing_it.news.prefilter import (
    FANOUT_MAX,
    INDEX_FANOUT_MIN,
    NO_FILTER,
    ROUNDUP_TAGS,
    STAGE1,
    SWING_SKIP_TAGS,
    PairFilter,
    add_filter_args,
    filter_from_args,
    keep_pair,
    roundup_tag,
    skip_reason,
    swing_skip_tag,
)

REAL = "기아 ‘PV7’, 獨 하노버서 세계 첫 공개"


class TestRoundupTag:
    @pytest.mark.parametrize(
        "title",
        [
            "[표] 코스피 기관·외국인·개인 순매수도 상위종목",
            "[개장시황] 엔비디아 호실적 영향 코스피, 2%대 상승 출발",
            "[마감시황] 코스피, 삼성전자 주주환원 기대에 6900선 회복",
            "[오늘의 IR] 삼성전자 등",
            "[티타임] 어쩌고",
            "[食전食후] 어쩌고",
            "[오늘의 증시일정] 9월 16일",
            "[연합뉴스 이 시각 헤드라인] 오전 9시",
            "[AI MY 증시전망] 9월 16일",
            "[장중시황] 코스피 보합",
            "[ETF 시황] 어쩌고",
            "[주간수급리포트] 어쩌고",
            "[주간증시전망] 어쩌고",
            "[오늘증시] 어쩌고",
            "[0519마감체크]",
            "[0810개장체크] 어쩌고",
            "[0703주요일정] 어쩌고",
        ],
    )
    def test_matches_known_roundup_tags(self, title: str) -> None:
        assert roundup_tag(title) is not None

    @pytest.mark.parametrize(
        "title",
        [
            "[단독]\"전세계인 다 먹은 꼴\"…'매운맛 뚝심' 불닭, 100억개 신화 썼다",
            "[특징주] 삼성전기, 10%대 급등에 신고가 경신",
            "[리포트 브리핑] 어쩌고",
            "[장중수급포착] 한화생명, 외국인/기관 동시 순매수… 주가 +8.15%",
            "HD현대, 1분기 실적 '역대 최대'…영업익 2조8348억원",
            "",
        ],
    )
    def test_does_not_match_tradeable_titles(self, title: str) -> None:
        assert roundup_tag(title) is None

    def test_tolerates_leading_whitespace_full_width_brackets_and_none(self) -> None:
        assert roundup_tag("  [표] 어쩌고") == "표"
        assert roundup_tag("【표】 어쩌고") == "표"
        assert roundup_tag(None) is None
        assert "표" in ROUNDUP_TAGS and isinstance(ROUNDUP_TAGS, frozenset)


class TestStage1Defaults:
    """측정으로 고른 상수 — 바꾸면 1단계와 다른 쌍 집합이다."""

    def test_thresholds(self) -> None:
        assert FANOUT_MAX == 11 and INDEX_FANOUT_MIN == 5
        assert STAGE1 == PairFilter() == PairFilter(True, 11, 5, True, SWING_SKIP_TAGS)
        assert len(SWING_SKIP_TAGS) == 14

    def test_fanout_boundary_eleven_kept_twelve_dropped(self) -> None:
        assert keep_pair(REAL, 11) is True
        assert skip_reason(REAL, 12) == "fanout:12"

    def test_index_boundary_is_five_codes(self) -> None:
        title = "코스피, 7999 찍고 7643 마감…외국인 5.7조 순매도"
        assert keep_pair(title, 4) is True
        assert skip_reason(title, 5) == "index:코스피:5"
        assert skip_reason("코스닥 900선 회복…개인 순매수", 7) == "index:코스닥:7"
        assert keep_pair("한미반도체, 코스피 시총 10위 진입", 2) is True

    def test_unknown_fanout_never_fires_fanout_or_index(self) -> None:
        for title in ("코스피 급락", "코스닥 900선 회복", REAL):
            assert keep_pair(title, None) is True

    def test_rule_order_narrowest_first(self) -> None:
        assert skip_reason("[표] 어쩌고", 40) == "roundup:표"
        assert skip_reason("코스피 급락", 40) == "fanout:40"
        assert skip_reason("[특징주] 삼성전기, 10%대 급등", 40) == "fanout:40"
        assert skip_reason("[특징주] 삼성전기, 10%대 급등", 1) == "swing:특징주"

    def test_real_dropped_disclosure_roundup(self) -> None:
        title = "[오늘의 주요공시] SK하이닉스·HLB·진에어·포스코홀딩스·네이버·대우건설 등"
        assert skip_reason(title, 28) == "fanout:28"
        assert skip_reason(title, 1) == "swing:오늘의 주요공시"

    def test_real_news_is_kept(self) -> None:
        assert keep_pair("삼성 3조 수주", 2) is True
        assert keep_pair("[단독] 삼성전자, 美 공장 증설", 1) is True


class TestSwingSkipTag:
    """옛 ``swing_skip_tag`` 그대로(roundup 포함) — 1단계 _swing.jsonl 을 만든 함수."""

    @pytest.mark.parametrize(
        "title, tag",
        [
            ("[장중수급포착] 한화생명, 외국인/기관 동시 순매수… 주가 +8.15%", "장중수급포착"),
            ("[특징주] 삼성전기, 10%대 급등에 신고가 경신", "특징주"),
            ("【마켓뷰】 코스피 숨고르기", "마켓뷰"),
            ("[표] 외국인 순매수 상위", "표"),
        ],
    )
    def test_skips(self, title: str, tag: str) -> None:
        assert swing_skip_tag(title) == tag

    @pytest.mark.parametrize("title", ["[단독] 삼성전자, 美 공장 증설", "[컨콜] SK하이닉스 HBM 수요", "LG엔솔 수주"])
    def test_keeps(self, title: str) -> None:
        assert swing_skip_tag(title) is None

    def test_union_with_live_rules_equals_stage1(self) -> None:
        """옛 경로(운영 3규칙 → swing_skip_tag) 와 STAGE1 한 번이 같은 쌍을 남긴다."""
        titles = ["[표] x", "[특징주] x", "코스피 급락", "진짜 뉴스", "[마켓뷰] x", "[0519마감체크]", "[단독] x"]
        live = PairFilter(swing_tags=False)
        for t in titles:
            for n in (None, 1, 5, 11, 12, 40):
                old = keep_pair(t, n, live) and swing_skip_tag(t) is None
                assert keep_pair(t, n) is old


class TestParameters:
    def test_each_rule_switches_off_independently(self) -> None:
        assert keep_pair("[표] x", 1, PairFilter(roundup=False, swing_tags=False)) is True
        assert keep_pair(REAL, 99, PairFilter(fanout_max=0)) is True
        assert keep_pair(REAL, 20, PairFilter(fanout_max=25)) is True
        assert keep_pair(REAL, 6, PairFilter(fanout_max=5)) is False
        assert keep_pair("코스피 급락", 9, PairFilter(index_fanout_min=0)) is True
        assert keep_pair("[특징주] x", 1, PairFilter(swing_tags=False)) is True
        assert keep_pair("[특징주] x", 1, PairFilter(swing_tag_set=frozenset({"마켓뷰"}))) is True

    def test_no_filter_keeps_everything(self) -> None:
        for t in ("[표] x", "[특징주] x", "코스피 급락"):
            assert keep_pair(t, 999, NO_FILTER) is True

    def test_cli_args_round_trip(self) -> None:
        ap = argparse.ArgumentParser()
        add_filter_args(ap)
        assert filter_from_args(ap.parse_args([])) == STAGE1
        cfg = filter_from_args(ap.parse_args(["--no-swing-tags", "--fanout-max", "0", "--index-fanout-min", "3"]))
        assert cfg == PairFilter(roundup=True, fanout_max=0, index_fanout_min=3, swing_tags=False)
        assert STAGE1.describe()["swing_tag_set"] == "default"
