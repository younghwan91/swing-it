"""시장 기분 판정(``swing_it.news.market_mood``) — 선택·PIT 매핑·중복 제거·파싱·비용 가드·집계 (가짜 판정기·가짜 원장, 네트워크 없음)."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from swing_it.news import market_mood as mm
from swing_it.news.judgments_db import KNOWN_JUDGES
from swing_it.news.prompt import SchemaError

KST = mm.KST
DAYS = [d.date() for d in pd.bdate_range("2026-06-29", "2026-07-10")]


def _art(i: int, when: dt.datetime, title: str, summary: str = "요약") -> dict:
    return {"id": f"toss:{i:012x}", "published_at": when, "title": title, "summary": summary}


def _at(day: dt.date, hh: int, mi: int = 0) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(hh, mi), tzinfo=KST)


_SYL = "가나다라마바사아자차카타파하"


def _title(i: int) -> str:
    """서로 충분히 다른 제목(2-gram 자카드 < 0.6) — "증시 기사 3" 식 번호만 다른 제목은 의도대로 중복 처리된다."""
    return f"증시 {i}번 기사 {_SYL[i % 14]}{chr(65 + i % 26)}"


# --- 선택 ---

def test_include_exclude_patterns():
    assert mm.is_market_title("코스피, 외인·기관 매도에 6750선 약세")
    assert mm.is_market_title("[마감시황] 코스닥 1%대 하락")
    assert not mm.is_market_title("[장중수급포착] SK, 외국인 6일 연속 순매수행진... 주가 +5.32%")
    assert not mm.is_market_title("[특징주] 삼성전자, 외국인 매수에 강세")
    assert not mm.is_market_title("현대차 3분기 실적 발표")      # 패턴 없음


def test_select_maps_pit_cutoff_and_caps_per_day():
    d0, d1 = DAYS[1], DAYS[2]
    arts = [_art(0, _at(d0, 15, 29), "코스피 마감 A")] + \
           [_art(1, _at(d0, 15, 30), "코스피 마감 B")] + \
           [_art(10 + i, _at(d1, 9, i), _title(i)) for i in range(12)]
    sel = mm.select_market_articles(pd.DataFrame(arts), DAYS, per_day=10)
    by_day = sel.groupby("signal_day").size()
    assert by_day[d0] == 1                   # 15:29 → d0
    assert by_day[d1] == 10                  # 15:30 → d1 로 넘어가 d1 의 첫 기사가 되고, 총 13건 중 10건
    first = sel[sel["signal_day"] == d1].iloc[0]
    assert first["title"] == "코스피 마감 B"  # 발행 시각 순
    assert sel["key"].str.endswith("|MARKET").all()
    assert sel["id"].is_unique


def test_select_dedupes_near_identical_titles():
    d = DAYS[1]
    arts = [_art(0, _at(d, 9, 1), "코스피, 중동 긴장에 6,780선 약세…코스닥 1%대↓(종합)"),
            _art(1, _at(d, 9, 2), "[속보] 코스피 중동 긴장에 6780선 약세 코스닥 1%대↓"),
            _art(2, _at(d, 9, 3), "외국인 순매도 전환, 증시 하락 출발"),
            _art(3, _at(d, 9, 4), "코스피, 중동 긴장에 6,780선 약세…코스닥 1%대↓(종합)")]
    sel = mm.select_market_articles(pd.DataFrame(arts), DAYS)
    assert list(sel["id"]) == [arts[0]["id"], arts[2]["id"]]


def test_select_drops_articles_outside_calendar_and_empty():
    assert mm.select_market_articles(pd.DataFrame(columns=["id", "published_at", "title", "summary"]), DAYS).empty
    late = _art(0, _at(DAYS[-1], 16), "코스피 마감")   # 마지막 거래일 15:30 뒤 → 달력 밖
    assert mm.select_market_articles(pd.DataFrame([late]), DAYS).empty


# --- 요청·파싱·judge 정의 ---

def test_request_shape_and_hash_independent_of_cache_block():
    a = {**_art(1, _at(DAYS[1], 9), "코스피 상승"), "key": "k|MARKET"}
    req = mm.build_request(a)
    p = req["params"]
    assert p["model"] == mm.MODEL and p["thinking"] == {"type": "disabled"} and "temperature" not in p
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"} and p["system"][0]["text"] == mm.SYSTEM
    assert "2026-06-30 09:00" in p["messages"][0]["content"]
    plain = mm.build_request(a, cache_system=False)["params"]
    from swing_it.news.judge import request_input_hash
    assert request_input_hash(req) == request_input_hash({"params": plain}) == mm.input_hash(a)


def test_parse_mood_validates_schema():
    good = '{"market_direction":1,"risk_appetite":2,"foreign_flow_tone":0,"breadth_tone":-1,"event_risk":0}'
    assert mm.parse_mood(good) == {"market_direction": 1, "risk_appetite": 2, "foreign_flow_tone": 0,
                                   "breadth_tone": -1, "event_risk": 0}
    assert mm.parse_mood("설명 " + good + " 끝")["risk_appetite"] == 2
    for bad in ('{"market_direction":2,"risk_appetite":1,"foreign_flow_tone":0,"breadth_tone":0,"event_risk":0}',
                '{"market_direction":1,"risk_appetite":1,"foreign_flow_tone":0,"breadth_tone":0}',
                '{"market_direction":"up","risk_appetite":1,"foreign_flow_tone":0,"breadth_tone":0,"event_risk":0}',
                "no json"):
        with pytest.raises(SchemaError):
            mm.parse_mood(bad)


def test_spec_is_distinct_judge_owned_by_swing_it():
    assert mm.SPEC.judge == "market-mood-haiku55" and mm.SPEC.owner_repo == "swing-it"
    assert all(mm.SPEC.identity() != s.identity() for s in KNOWN_JUDGES.values())
    assert mm.SPEC.params == {"max_tokens": mm.MAX_OUTPUT_TOKENS, "thinking": {"type": "disabled"}}


# --- 비용 ---

def test_cost_estimate_and_measured():
    arts = [{**_art(i, _at(DAYS[1], 9), "코스피 상승"), "key": f"k{i}|MARKET"} for i in range(600)]
    est = mm.estimate_usd(arts, batch=False)
    assert est["requests"] == 600 and 0 < est["usd"] < 0.5          # 파일럿은 $1 안
    assert mm.estimate_usd(arts, batch=True)["usd"] == pytest.approx(est["usd"] * 0.5)
    u = {"input_tokens": 1000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 100}
    assert mm.usage_usd(u, batch=False) == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    assert mm.usage_usd(u, batch=True) == pytest.approx((1000 * 0.10 + 100 * 0.50) / 2e6)


class FakeLedger:
    def __init__(self):
        self.entries = []
        self.spec = mm.SPEC

    def record(self, entries):
        self.entries += list(entries)
        return {"inserted": len(entries), "updated": 0, "unchanged": 0}


def _selected(n: int) -> pd.DataFrame:
    arts = [_art(i, _at(DAYS[1 + i // 10], 9, i % 10), _title(i)) for i in range(n)]
    return mm.select_market_articles(pd.DataFrame(arts), DAYS)


def test_judge_articles_with_fake_judge_writes_ledger_and_reuses():
    sel = _selected(12)
    calls = []

    def fake(a):
        calls.append(a["key"])
        if a["title"].startswith("증시 3번"):
            return None, "schema: boom", {"input_tokens": 50, "output_tokens": 10}
        return {"market_direction": 1, "risk_appetite": 2, "foreign_flow_tone": 1, "breadth_tone": 0, "event_risk": 0}, None, \
            {"input_tokens": 50, "output_tokens": 10}

    ledger = FakeLedger()
    already = {sel["key"].iloc[0]: {"market_direction": -1, "risk_appetite": 0, "foreign_flow_tone": -1,
                                     "breadth_tone": 0, "event_risk": 1}}
    outputs, info = mm.judge_articles(sel, fake, ledger, budget_usd=1.0, hard_cap_usd=2.0, already=already,
                                      flush_every=5, log=lambda *_: None)
    assert info["n_reused"] == 1 and info["n_judged"] == 11 and len(calls) == 11
    assert info["n_ok"] == 10 and info["n_error"] == 1
    assert len(outputs) == 11 and outputs[sel["key"].iloc[0]]["market_direction"] == -1
    assert len(ledger.entries) == 11
    err = [e for e in ledger.entries if e["output"] is None]
    assert len(err) == 1 and err[0]["error"].startswith("schema") and err[0]["judged_at_exact"] is True
    assert all(e["input_hash"] for e in ledger.entries)
    assert info["measured_usd"] == pytest.approx(11 * (50 * 0.10 + 10 * 0.50) / 1e6)


def test_judge_articles_hard_cap_sends_nothing():
    sel = _selected(5)
    calls = []
    with pytest.raises(mm.CostCapExceeded):
        mm.judge_articles(sel, lambda a: calls.append(a) or (None, "x", None), FakeLedger(), budget_usd=1.0,
                          hard_cap_usd=0.0, log=lambda *_: None)
    assert not calls


def test_judge_articles_stops_when_measured_exceeds_budget():
    sel = _selected(20)
    big = {"input_tokens": 10_000_000, "output_tokens": 0}   # 한 건에 $1
    fake = lambda a: ({"market_direction": 0, "risk_appetite": 1, "foreign_flow_tone": 0, "breadth_tone": 0,  # noqa: E731
                       "event_risk": 0}, None, big)
    outputs, info = mm.judge_articles(sel, fake, None, budget_usd=1.5, hard_cap_usd=100.0, log=lambda *_: None)
    assert info["n_judged"] == 2 and info["n_unjudged"] == 18 and info["stopped"]


# --- 집계 ---

def test_daily_mood_and_ema_are_causal():
    sel = _selected(25)   # 3일(10·10·5)
    out = {k: {"market_direction": 1, "risk_appetite": 2, "foreign_flow_tone": 1, "breadth_tone": 1, "event_risk": 0}
           for k in sel["key"]}
    out.pop(sel["key"].iloc[0])   # 하나 미판정
    daily = mm.daily_mood(sel, out)
    assert list(daily["n_selected"]) == [10, 10, 5] and list(daily["n_judged"]) == [9, 10, 5]
    assert daily["market_direction"].iloc[0] == 1.0
    idx = mm.mood_index(daily, DAYS)
    assert len(idx) == len(DAYS) and idx["n_selected"].iloc[0] == 0
    assert np.isnan(idx["market_direction_ema20"].iloc[0])          # 첫 거래일엔 기사 없음
    assert idx["mood_score"].dropna().iloc[0] == pytest.approx(1 + 1 + 1 + 1 - 0)
    # 미래 기사를 바꿔도 앞 행은 그대로
    out2 = {k: {**v, "market_direction": -1} for k, v in out.items()}
    for k in sel["key"].iloc[:20]:
        out2[k] = out[k] if k in out else out2.get(k)
    out2 = {k: v for k, v in out2.items() if v is not None}
    idx2 = mm.mood_index(mm.daily_mood(sel, out2), DAYS)
    cut = pd.Timestamp(DAYS[2])
    pd.testing.assert_frame_equal(idx.loc[:cut], idx2.loc[:cut])
    assert idx2["market_direction_ema20"].iloc[-1] < idx["market_direction_ema20"].iloc[-1]


def test_mood_index_carries_ema_over_days_without_articles():
    sel = _selected(10)   # 하루치
    out = {k: {"market_direction": 1, "risk_appetite": 1, "foreign_flow_tone": 0, "breadth_tone": 0, "event_risk": 0}
           for k in sel["key"]}
    idx = mm.mood_index(mm.daily_mood(sel, out), DAYS)
    d = pd.Timestamp(DAYS[1])
    assert idx.loc[d, "market_direction_ema20"] == 1.0          # adjust=False: 첫 관측이 EMA 초기값
    assert (idx["market_direction_ema20"].loc[d:] == 1.0).all()   # 기사 없는 날은 유지(ffill)
    assert np.isnan(idx.loc[idx.index[-1], "market_direction"])


def test_messages_judge_parses_and_reports_errors(monkeypatch):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["x-api-key"] == "k" and body["model"] == mm.MODEL
        if "오류" in body["messages"][0]["content"]:
            return httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}], "usage": {"input_tokens": 1}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(
            {"market_direction": -1, "risk_appetite": 0, "foreign_flow_tone": -1, "breadth_tone": 0, "event_risk": 1})}],
            "usage": {"input_tokens": 300, "cache_read_input_tokens": 250, "output_tokens": 40}})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    judge = mm.messages_judge("k")
    a = {**_art(1, _at(DAYS[1], 9), "코스피 급락"), "key": "k|MARKET"}
    out, err, usage = judge(a)
    assert out["market_direction"] == -1 and err is None and usage["cache_read_input_tokens"] == 250
    out, err, usage = judge({**a, "title": "오류"})
    assert out is None and err.startswith("schema") and usage == {"input_tokens": 1}
