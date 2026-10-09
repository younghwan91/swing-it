"""뉴스 거부권 — 순수 규칙·창·판정 가드(가짜 판정기·가짜 원장)·백테스트 패널·후보 분리 (설계 §7, NEWS_VETO.md)."""

import datetime as dt

import numpy as np
import pandas as pd

from swing_it.live import news_veto as nv
from swing_it.live.backtest import run_backtest
from swing_it.live.config import StrategyConfig
from swing_it.live.strategy import Candidate

KST = nv.KST


def _cand(code, rel=0.5, rs=90.0):
    return Candidate(code=code, signal_day=pd.Timestamp("2026-03-10"), market="KOSPI", close=100.0, pivot=99.0,
                     contraction_low=95.0, rel=rel, rs=rs, adv20=6000.0, pead_rank=0.9, flow_f=1.0)


def _neg(direction=-1, subject="main"):
    return {"subject": subject, "sentiment_direction": direction, "persistence": "one_off", "freshness": "new",
            "reports_price_move": False, "material_type": "기타", "specificity": 1, "surprise": 0,
            "sentiment_strength": 2, "scale_vs_size": "small", "score": -0.3}


DAYS = [d.date() for d in pd.bdate_range("2026-01-05", "2026-03-31")]
D = dt.date(2026, 3, 10)


def _pair(key, code, when: dt.datetime, title="실적 악화", related_n=1):
    return {"key": key, "code": code, "available_at": when, "title": title, "related_n": related_n,
            "summary": "요약", "body": "요약", "name": "종목", "market": "KOSPI", "sector": None}


def _at(day: dt.date, hh=10):
    return dt.datetime.combine(day, dt.time(hh), tzinfo=KST)


def test_veto_decision_boundaries():
    assert not nv.veto_decision(0, 0) and not nv.veto_decision(0, 1)
    assert nv.veto_decision(1, 1) and nv.veto_decision(0, 2)


def test_window_days_trading_calendar():
    w = nv.window_days(DAYS, D, 5)
    assert len(w) == 5 and w[-1] == D and w[0] == dt.date(2026, 3, 4)
    assert nv.window_days(DAYS, dt.date(2026, 3, 14), 1) == [dt.date(2026, 3, 13)]   # 토요일 → 금요일
    assert nv.window_days(DAYS, dt.date(2025, 1, 1), 5) == []


def test_count_news_windows_pit_and_unknown():
    w20 = nv.window_days(DAYS, D, 20)
    pairs = pd.DataFrame([
        _pair("a|X", "X", _at(w20[0], 9)),                       # 20일 창 첫날 → neg20
        _pair("b|X", "X", _at(dt.date(2026, 3, 9), 16)),         # 15:30 뒤 → 신호일 3-10 → neg5
        _pair("c|X", "X", _at(D, 15, ) + dt.timedelta(minutes=29)),  # 15:29 → 3-10 → neg5
        _pair("d|X", "X", _at(D, 16)),                           # 15:30 뒤 → 3-11 → 창 밖
        _pair("e|X", "X", _at(dt.date(2026, 3, 6))),             # 판정 없음 → 모름
        _pair("f|X", "X", _at(dt.date(2026, 3, 6)), related_n=12),   # 필터 탈락(팬아웃 > 11)
        _pair("g|Y", "Y", _at(dt.date(2026, 3, 6))),             # 긍정
    ])
    judged = {"a|X": _neg(), "b|X": _neg(), "c|X": _neg(subject="partial"), "d|X": _neg(), "g|Y": _neg(1)}
    out = nv.count_news(pairs, judged, DAYS, D)
    x = out["X"]
    assert (x.neg5, x.neg20, x.judged, x.unknown, x.feed) == (2, 3, 3, 1, 4)
    assert x.vetoed
    assert out["Y"].neg20 == 0 and not out["Y"].vetoed
    kept, vetoed = nv.split_candidates([_cand("X"), _cand("Y"), _cand("Z")], out)
    assert [c.code for c in kept] == ["Y", "Z"] and [c.code for c in vetoed] == ["X"]   # Z 재료 없음 = 모름 = 진입


def test_mention_or_neutral_is_not_negative():
    pairs = pd.DataFrame([_pair("a|X", "X", _at(D)), _pair("b|X", "X", _at(D)), _pair("c|X", "X", _at(D))])
    judged = {"a|X": _neg(subject="mention"), "b|X": _neg(0), "c|X": _neg(-1, "partial")}
    assert nv.count_news(pairs, judged, DAYS, D)["X"].neg5 == 1


class FakeLedger:
    def __init__(self, fail=False):
        self.entries, self.fail = [], fail

    def record(self, entries):
        if self.fail:
            raise RuntimeError("db down")
        self.entries.extend(entries)
        return {"inserted": len(entries), "updated": 0, "unchanged": 0}


def _fake_judge(outputs: dict):
    calls = []

    def judge(pair):
        calls.append(pair["key"])
        o = outputs.get(pair["key"], "err")
        if o == "err":
            return None, "schema: bad", {"input_tokens": 10, "output_tokens": 5}
        return o, None, {"input_tokens": 10, "output_tokens": 5}

    judge.calls = calls
    return judge


def test_judge_pending_writes_ledger_and_caps():
    pairs = [_pair(f"k{i}|X", "X", _at(D)) for i in range(4)]
    for p in pairs:
        p["category"] = None
    judge = _fake_judge({"k0|X": _neg(), "k1|X": "err", "k2|X": _neg(1)})
    led = FakeLedger()
    out, info = nv.judge_pending(pairs, judge, led, cap=3)
    assert set(out) == {"k0|X", "k2|X"}
    assert (info["n_judged"], info["n_ok"], info["n_error"], info["n_skipped_cap"]) == (3, 2, 1, 1)
    assert info["alerts"] and "상한" in info["alerts"][0]
    assert len(led.entries) == 3 and led.entries[1]["error"] == "schema: bad" and led.entries[1]["output"] is None
    assert all(e["judged_at_exact"] and e["input_hash"] for e in led.entries)
    assert info["usage"]["input_tokens"] == 30
    # 원장 실패는 경보일 뿐 판정은 쓴다
    out2, info2 = nv.judge_pending(pairs[:1], judge, FakeLedger(fail=True))
    assert set(out2) == {"k0|X"} and any("원장" in a for a in info2["alerts"])


def test_judge_pending_time_budget_marks_rest_unknown():
    pairs = [_pair(f"k{i}|X", "X", _at(D)) for i in range(3)]
    for p in pairs:
        p["category"] = None
    out, info = nv.judge_pending(pairs, _fake_judge({}), None, time_budget=-1.0)
    assert out == {} and info["n_judged"] == 0 and info["n_skipped_cap"] == 3


class FakeCon:
    """apply_live_veto 가 부르는 DB 함수를 monkeypatch 로 바꾸므로 연결 자체는 더미."""


def test_apply_live_veto_end_to_end_with_fake_judge(monkeypatch):
    w20 = nv.window_days(DAYS, D, 20)
    pairs = pd.DataFrame([
        _pair("a|X", "X", _at(dt.date(2026, 3, 9))),             # 미판정 → 가짜 판정기 → 악재(5일)
        _pair("b|Y", "Y", _at(w20[2])), _pair("c|Y", "Y", _at(w20[3])),   # 재사용 성공 행 2건 악재(20일 ≥ 2)
        _pair("d|Z", "Z", _at(D)),                               # 판정 실패 → 모름
    ])
    pairs["article_id"] = ["a", "b", "c", "d"]
    monkeypatch.setattr(nv, "load_candidate_pairs", lambda con, codes, lo, hi: pairs[pairs["code"].isin(codes)])
    reuse = {"v2r3-haiku55": {"b|Y": {"output": _neg()}, "c|Y": {"output": _neg()}}}
    monkeypatch.setattr(nv, "read_judgments", lambda con, j, keys: {k: v for k, v in reuse.get(j, {}).items()
                                                                     if k in set(keys)})
    monkeypatch.setattr(nv, "JudgmentLedger", lambda con, spec: FakeLedger())
    judge = _fake_judge({"a|X": _neg()})
    cands = [_cand("X"), _cand("Y"), _cand("Z"), _cand("W")]
    r = nv.apply_live_veto(FakeCon(), cands, D, DAYS, judge_fn=judge)
    assert [c.code for c in r.vetoed] == ["X", "Y"] and [c.code for c in r.kept] == ["Z", "W"]
    assert sorted(judge.calls) == ["a|X", "d|Z"]                 # 재사용 행은 다시 판정하지 않는다
    assert r.info["n_reused"] == 2 and r.info["judge"]["n_error"] == 1
    assert r.info["vetoed"] == {"X": {"neg5": 1, "neg20": 1}, "Y": {"neg5": 0, "neg20": 2}}
    assert r.alerts == []


def test_apply_live_veto_without_api_key_is_unknown_and_alerts(monkeypatch):
    pairs = pd.DataFrame([_pair("a|X", "X", _at(D))])
    pairs["article_id"] = ["a"]
    monkeypatch.setattr(nv, "load_candidate_pairs", lambda con, codes, lo, hi: pairs)
    monkeypatch.setattr(nv, "read_judgments", lambda con, j, keys: {})
    monkeypatch.setattr(nv, "api_key_from_env", lambda: None)
    r = nv.apply_live_veto(FakeCon(), [_cand("X")], D, DAYS)
    assert [c.code for c in r.kept] == ["X"] and r.vetoed == []
    assert any("ANTHROPIC_API_KEY" in a for a in r.alerts)


def test_apply_live_veto_db_failure_never_blocks(monkeypatch):
    def boom(con, codes, lo, hi):
        raise RuntimeError("db")

    monkeypatch.setattr(nv, "load_candidate_pairs", boom)
    r = nv.apply_live_veto(FakeCon(), [_cand("X")], D, DAYS, judge_fn=_fake_judge({}))
    assert [c.code for c in r.kept] == ["X"] and any("기사 읽기 실패" in a for a in r.alerts)
    assert nv.apply_live_veto(FakeCon(), [], D, DAYS).kept == []


def test_live_spec_is_same_definition_as_research_judge():
    from swing_it.news.judgments_db import V2R3_HAIKU55

    assert nv.LIVE_SPEC.identity() == V2R3_HAIKU55.identity()
    assert nv.LIVE_SPEC.judge == "v2r3-haiku55-live" and nv.LIVE_SPEC.owner_repo == "swing-it"


def test_veto_panel_from_daily_trading_day_windows():
    days = DAYS
    daily = pd.DataFrame([
        {"code": "X", "date": dt.date(2026, 2, 2), "n_neg": 1},
        {"code": "X", "date": dt.date(2026, 2, 20), "n_neg": 1},
        {"code": "Y", "date": dt.date(2026, 3, 2), "n_neg": 1},
    ])
    p = nv.veto_panel_from_daily(daily, days, ["X", "Y", "Z"])
    ts = pd.Timestamp
    assert p.loc[ts("2026-02-02"), "X"] and p.loc[ts("2026-02-06"), "X"]        # 5거래일 창 안
    assert not p.loc[ts("2026-02-09"), "X"]                                      # 6번째 거래일 — 20일 창 1건뿐
    assert p.loc[ts("2026-02-20"), "X"] and p.loc[ts("2026-02-27"), "X"]        # 2건(20거래일 창 02-02..02-27) → 거부
    assert not p.loc[ts("2026-03-02"), "X"]                                      # 21번째 거래일 — 2-02 가 창 밖, 1건뿐
    assert p.loc[ts("2026-03-02"), "Y"] and not p.loc[ts("2026-03-09"), "Y"]
    assert not p["Z"].any()
    assert nv.veto_panel_from_daily(pd.DataFrame(), days, ["X"]).shape == (len(days), 1)


def _flat_panels():
    days = pd.bdate_range("2025-01-01", periods=30)
    codes = ["000010", "000020"]
    one = pd.DataFrame(100.0, index=days, columns=codes)
    const = lambda v: pd.DataFrame(v, index=days, columns=codes)  # noqa: E731
    pn = {"open": one.copy(), "high": one * 1.005, "low": one * 0.995, "close": one.copy(), "volume": one * 0 + 1000.0,
          "market": pd.Series("KOSPI", index=codes), "ma50": const(90.0), "universe": const(False), "tt": const(False),
          "breakout": const(False), "rs": const(80.0), "breadth": const(0.6), "pead_rank": const(0.8), "rel": const(0.5),
          "F": const(1.0), "adv20": const(10_000.0), "pivot": const(99.0), "contraction_low": const(95.0)}
    for k in ("universe", "tt", "breakout"):
        pn[k].loc[days[2], :] = True
    return days, codes, pn


def test_backtest_applies_veto_panel_and_records_virtual_entries():
    days, codes, pn = _flat_panels()
    veto = pd.DataFrame(False, index=days, columns=codes)
    veto.loc[days[2], "000010"] = True
    res = run_backtest(pn, StrategyConfig(), cost_fn=None, veto=veto)
    assert list(res.trades["code"]) == ["000020"]
    assert res.skips["skip_news_veto"] == 1
    assert list(res.vetoed["code"]) == ["000010"] and res.vetoed.iloc[0]["event_day"] == days[2]
    off = run_backtest(pn, StrategyConfig(news_veto=False), cost_fn=None, veto=veto)
    assert sorted(off.trades["code"]) == codes and off.skips["skip_news_veto"] == 0
    # 패널 밖 종목·날짜는 모름 = 거부 안 함
    part = run_backtest(pn, StrategyConfig(), cost_fn=None, veto=veto[["000020"]])
    assert sorted(part.trades["code"]) == codes


def test_messages_judge_parses_and_reports_errors(monkeypatch):
    import httpx

    class Resp:
        def __init__(self, status, body):
            self.status_code, self._body, self.text = status, body, str(body)

        def json(self):
            return self._body

    class Client:
        def __init__(self, timeout):
            self.calls = []

        def post(self, url, headers, json):
            self.calls.append((url, headers, json))
            text = '{"subject":"main","reports_price_move":false,"material_type":"기타","persistence":"one_off",' \
                   '"specificity":1,"freshness":"new","surprise":0,"sentiment_direction":-1,"sentiment_strength":2,' \
                   '"scale_vs_size":"small"}'
            if json["messages"][0]["content"].startswith("다음은 BAD"):
                return Resp(500, {"error": "x"})
            return Resp(200, {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 1}})

    made = {}
    monkeypatch.setattr(httpx, "Client", lambda timeout: made.setdefault("c", Client(timeout)))
    judge = nv.messages_judge("sk-test")
    pair = {"key": "a|X", "code": "X", "name": "종목", "market": "KOSPI", "sector": None, "title": "t", "body": "b",
            "summary": "b", "category": None}
    out, err, usage = judge(pair)
    assert err is None and out["sentiment_direction"] == -1 and out["subject"] == "main" and "score" in out
    url, headers, body = made["c"].calls[0]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "sk-test"
    assert body["model"] == "claude-haiku-5-5" and body["thinking"] == {"type": "disabled"} and "temperature" not in body
    assert body["system"][0]["text"] == nv.LIVE_SPEC.system_prompt
    out, err, _ = judge({**pair, "name": "BAD"})
    assert out is None and err.startswith("http 500")
    assert np.isfinite(usage["input_tokens"])
