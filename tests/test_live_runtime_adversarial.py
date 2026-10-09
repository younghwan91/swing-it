"""스윙 라이브 런타임 공격 시나리오 — ``research/logs/live_swing/REVIEW_RUNTIME.md`` 의 11개(C1·C2·M1~M7 + 매도 지정가).

검토자가 재현한 실패 모드를 그대로 가져와, 고친 뒤의 **올바른** 동작을 단언한다. 하네스(``Env``·``FakeKiwoom``·``cand``)는
``tests/test_live_runtime.py`` 의 것이다(작업마다 새 런타임 = 새 프로세스, 상태는 파일).
"""

from __future__ import annotations

import pytest
from krx_quant_core.execution.events import OpenOrder

import test_live_runtime as H
from swing_it.live.runtime import SignalBatch, SwingRuntime
from swing_it.live.state import StateStore

D1, D2, D3, SIGNAL_D = H.D1, H.D2, H.D3, H.SIGNAL_D
cand, Env, FakeKiwoom, _closed = H.cand, H.Env, H.FakeKiwoom, H._closed


def _live_entry(tmp_path, fk, codes=("AAAAAA",)):
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand(c) for c in codes],
                                          trail_closes={c: [9_000.0] * 50 for c in codes})
    for c in codes:
        env.signals.trail[c] = [9_000.0] * 50
        env.q(c, 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    assert env.run("preopen", D1, "08:35") == 0
    return env


# ---------------------------------------------------------------------- 1. C1 멱등성: POST 와 스냅샷 저장 사이 크래시


def test_crash_between_post_and_state_save_does_not_duplicate_buy(tmp_path, monkeypatch):
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    # 주문 POST 는 나갔는데 그 뒤 state.json 저장에서 프로세스가 죽는다(SIGKILL·전원). POST 전 저장은 성공한다.
    real_save = StateStore.save

    def crash_after_post(self, st):
        if any(i.side == "buy" for i in fk.submitted):
            raise KeyboardInterrupt()
        real_save(self, st)

    monkeypatch.setattr(StateStore, "save", crash_after_post)
    with pytest.raises(KeyboardInterrupt):
        env.run("preopen", D1, "08:35")
    assert len([i for i in fk.submitted if i.side == "buy"]) == 1
    monkeypatch.setattr(StateStore, "save", real_save)
    st = env.state()
    [o] = st.orders.values()
    assert o.status == "intent" and o.ord_no is None and st.positions["AAAAAA"].status == "entry_pending"
    # 크론이 다시 띄운다 — 저장된 주문 의도를 브로커 미체결에서 찾아 이어받고 **다시 사지 않는다**
    assert env.run("preopen", D1, "08:36") == 0
    assert len([i for i in fk.submitted if i.side == "buy"]) == 1 and len(fk.orders) == 1
    st = env.state()
    [o] = st.orders.values()
    assert (o.status, o.ord_no) == ("submitted", "0000001") and "1" in st.own_ord_nos   # 정규화 번호
    assert env.events("order_resumed") and any(a[0] == "error" and "이어받음" in a[1] for a in env.alerts)
    fk.match("AAAAAA", 10_100)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    assert env.state().positions["AAAAAA"].status == "open" and env.book().position("AAAAAA").qty == 100


def test_unresolvable_intent_is_marked_unknown_and_never_reposted(tmp_path, monkeypatch):
    """의도는 저장됐는데 브로커 미체결에도 없다(POST 자체가 안 나갔거나 이미 체결) → unknown + 경보, 재발주 없음."""
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    real_submit = fk.submit
    monkeypatch.setattr(fk, "submit", lambda intent: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        env.run("preopen", D1, "08:35")
    monkeypatch.setattr(fk, "submit", real_submit)
    assert env.run("preopen", D1, "08:36") == 0
    assert fk.submitted == []
    [o] = env.state().orders.values()
    assert o.status == "unknown" and env.events("order_unresolved")
    assert any(a[0] == "error" and "주문 미해결" in a[1] for a in env.alerts) and "AAAAAA" in env.state().blocked_codes


# ---------------------------------------------------------------------- 2. M1 거부된 매도: 경보 + 회로 차단


def test_rejected_sell_alerts_and_stops_after_three(tmp_path):
    fk = FakeKiwoom({}, reject_sells=True)
    env = _live_entry(tmp_path, fk)
    fk.match("AAAAAA", 10_100)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    env.alerts.clear()
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    for i in range(5):
        env.clock.set(D1, "10:30", i)
        assert env.run("monitor", D1, "10:30", once=True) == 0
    sells = [i for i in fk.submitted if i.side == "sell"]
    assert len(sells) == 3                                           # 3회 거부 뒤 멈춘다
    it = env.state().positions["AAAAAA"].exit_intent
    assert it["kind"] == "stop" and it["halted"]
    errors = [a for a in env.alerts if a[0] == "error"]
    assert any("매도 주문 rejected" in a[1] for a in errors) and any("needs_human" in a[1] for a in errors)
    # 다음날 preopen 이 한 번 풀어 다시 시도한다
    H._eod(env, D1, {"AAAAAA": 9_100})
    env.q("AAAAAA", 9_100, prev_close=9_100)
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    assert not env.state().positions["AAAAAA"].exit_intent.get("halted")
    assert len([i for i in fk.submitted if i.side == "sell"]) == 4  # 장전 동시호가 매도 1건 더(거부)


# ---------------------------------------------------------------------- 3. C2 취소-체결 경합 + 잔고 지연 → 사용자 주식을 팔지 않는다


def test_cancel_reject_with_lagging_holdings_never_resells(tmp_path):
    fk = FakeKiwoom({}, lag=2)
    env = _live_entry(tmp_path, fk)
    fk.match("AAAAAA", 10_100)
    fk.publish()                          # 전략 100주 체결(잔고 반영)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    assert env.state().positions["AAAAAA"].qty == 100
    fk.user_trade("AAAAAA", 100)
    fk.publish()                        # 사용자가 장중 같은 종목 100주 매수
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)                        # 손절 매도 100 @9,140; 잔고 +100 은 사용자 몫
    assert env.state().baseline["user"] == {"AAAAAA": 100}
    fk.match("AAAAAA", 9_140)                                         # 체결 — 잔고 조회에는 2번 더 늦게 보인다
    env.clock.set(D1, "10:31", 1)
    env.q("AAAAAA", 9_100, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:31", once=True)                        # 재가격 → 취소 거부 → 잔고 재조회(≥3회, ≥10초) → 체결 확인
    sells = [i for i in fk.submitted if i.side == "sell"]
    assert len(sells) == 1, "second sell must not be submitted while the first may have filled"
    assert fk.orders == {} and fk._h["AAAAAA"].qty == 100             # 사용자 100주는 그대로
    st = env.state()
    assert "AAAAAA" not in st.positions and env.book().positions() == {}
    [a] = _closed(env, "AAAAAA")
    assert a.exit_reason == "stop"
    assert st.baseline["user"] == {"AAAAAA": 100}
    assert not [a for a in env.alerts if "전략 몫이 모자람" in a[2]]
    assert env.clock().strftime("%H:%M:%S") >= "10:31:05"            # 체결이 보일 때까지 기다렸다(5초 간격 재조회)


def test_cancel_reject_unexplained_releases_reservation_and_stop_goes_out(tmp_path):
    """검토 F3: 취소 거부인데 체결도 미체결도 아니면(사용자 HTS 취소) 예약을 풀고(released) 손절을 다시 낸다 — 교착 없음."""
    fk = FakeKiwoom({})
    env = _live_entry(tmp_path, fk)
    fk.match("AAAAAA", 10_100)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    [o] = env.state().live_orders()
    fk.orders.pop(o.ord_no)
    fk.remaining.pop(o.ord_no)                                        # HTS 수동 취소(잔고 변화 없음)
    reads_before = fk.holdings_calls
    env.q("AAAAAA", 9_100, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:31", once=True)                        # 재가격 → 취소 거부 → 재조회 3회 → released → 새 손절
    assert fk.holdings_calls - reads_before >= 4
    assert "10:31:10" <= env.clock().strftime("%H:%M:%S") <= "10:32:00"   # 예약이 청산을 막은 시간 ≈ 10초
    st = env.state()
    sells = sorted((o for o in st.orders.values() if o.side == "sell"), key=lambda o: o.oid)
    assert [o.status for o in sells] == ["released", "submitted"] and sells[1].qty == 100
    assert env.events("order_released") and any("예약 해제" in a[1] for a in env.alerts)
    fk.match("AAAAAA", 9_050)
    env.q("AAAAAA", 9_050, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:33", once=True)
    assert "AAAAAA" not in env.state().positions and env.book().positions() == {}


# ---------------------------------------------------------------------- 4. 진입 취소-체결 경합 + 잔고 지연 → 전략 주식을 잃지 않는다


def test_entry_cancel_reject_with_lagging_holdings_keeps_position(tmp_path):
    fk = FakeKiwoom({}, lag=3)
    env = _live_entry(tmp_path, fk)
    fk.match("AAAAAA", 10_100)                                        # 09:00 단일가 체결 — 잔고 조회에는 3번 뒤에 보인다
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    st = env.state()
    assert st.positions["AAAAAA"].status == "open" and st.positions["AAAAAA"].qty == 100   # missed 가 아니다
    assert st.baseline["user"] == {} and env.book().position("AAAAAA").qty == 100
    env.q("AAAAAA", 10_200, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    assert env.state().baseline["user"] == {} and not env.events("mismatch")


# ---------------------------------------------------------------------- 5. M3 가짜 휴장 → 감시는 계속된다


def test_false_holiday_at_open_check_is_not_sticky(tmp_path):
    env = Env(tmp_path)
    H._enter(env, {"AAAAAA": {}})
    H._eod(env, D1, {"AAAAAA": 10_000})
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    # D2 09:02: ka10081 이 아직 오늘 봉을 안 주고(지연) 시세 조회도 실패 → "의심" — 취소도, 판정도 없다
    env.quotes.set_bars(H.REF_CODE, [H.DailyBar(d, 1, 1, 1, 1) for d in H.SESSIONS if d < D2])
    env.quotes.fail.add("AAAAAA")
    assert env.run("open_check", D2, "09:02") == 0
    assert env.state().sessions[D2.isoformat()] == "suspect" and env.events("holiday_suspect")
    assert env.state().positions["AAAAAA"].status == "open"
    # 10:30 오늘 일자가 찍힌 시세가 온다(봉은 아직 없다) → 거래일 확정 → 손절가 아래 → 매도
    env.quotes.fail.clear()
    env.q("AAAAAA", 8_000, open=9_900, prev_close=10_000, date=D2)
    assert env.run("monitor", D2, "10:30", once=True) == 0
    st = env.state()
    assert st.sessions[D2.isoformat()] == "open"
    assert "AAAAAA" not in st.positions and _closed(env, "AAAAAA")[0].exit_reason == "stop"


# ---------------------------------------------------------------------- 6. M4 open_pric 없음 → missed 가 아니라 경보 + 보류 + 대용


def test_missing_open_price_alerts_and_resolves_with_proxy_after_0905(tmp_path):
    env = Env(tmp_path)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA"), cand("BBBBBB")])
    for c in ("AAAAAA", "BBBBBB"):
        env.q(c, 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    env.alerts.clear()
    env.q("AAAAAA", 10_100, prev_close=10_000)                        # open=None (필드명 틀림/미제공)
    env.q("BBBBBB", 10_300, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    st = env.state()
    assert {p.status for p in st.positions.values()} == {"entry_pending"} and len(st.live_orders()) == 2
    assert not env.events("entry_missed")
    assert [a for a in env.alerts if a[0] == "error" and "open_pric" in a[1]]
    # 09:03 monitor: 아직 대용 금지 → 그대로. 09:06: 현재가를 단일가 대용으로(경보) → A 체결, B 미체결 → missed
    assert env.run("monitor", D1, "09:03", once=True) == 0
    assert len(env.state().live_orders()) == 2
    assert env.run("monitor", D1, "09:06", once=True) == 0
    st = env.state()
    assert st.positions["AAAAAA"].status == "open" and st.positions["AAAAAA"].entry_price == 10_100
    [b] = _closed(env, "BBBBBB")
    assert b.status == "missed" and env.events("open_proxy")
    assert any(a[0] == "warn" and "시가 대용" in a[1] for a in env.alerts)


# ---------------------------------------------------------------------- 7. M2 같은 종목의 사용자 미체결 매도가 손절을 막지 않는다


def test_user_pending_sell_on_same_code_does_not_block_strategy_stop(tmp_path):
    fk = FakeKiwoom({}, foreign=[OpenOrder("U000001", "AAAAAA", "sell", 100, 100, 12_000)])
    env = _live_entry(tmp_path, fk)
    fk.match("AAAAAA", 10_100)
    fk.user_trade("AAAAAA", 100)                                     # 사용자도 100주 (그 위에 12,000 매도 걸어둠)
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    sells = [o for o in env.state().orders.values() if o.side == "sell"]
    assert sells and sells[-1].status == "submitted" and sells[-1].qty == 100
    assert [i.qty for i in fk.submitted if i.side == "sell"] == [100]


# ---------------------------------------------------------------------- 8. M7 크론 지연 → 시간창 밖 → 경보


def test_late_cron_alerts(tmp_path):
    env = Env(tmp_path)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.alerts.clear()
    assert env.run("preopen", D1, "09:00") == 2                       # 30초 늦었다
    assert [a for a in env.alerts if a[0] == "warn" and "시간창 밖" in a[1]]
    assert not env.state().live_orders()


# ---------------------------------------------------------------------- 9. M6 부분 체결 + 사용자 매수 같은 날 → 브로커 잔량으로 상한


def test_partial_fill_plus_user_buy_same_day_books_only_filled_qty(tmp_path):
    fk = FakeKiwoom({})
    env = _live_entry(tmp_path, fk)
    fk.partial_fill("AAAAAA", 60, 10_100)                            # 단일가에 60주만 체결(부분) — 미체결 잔량 40
    fk.user_trade("AAAAAA", 50)                                      # 같은 아침 사용자 50주 매수
    env.q("AAAAAA", 10_150, open=10_100, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    st = env.state()
    assert env.book().position("AAAAAA").qty == 60                   # 전략 장부 60 (브로커 잔량 40 → 체결 60 까지만)
    assert st.baseline["user"] == {"AAAAAA": 50}                      # 사용자 몫 50
    assert st.positions["AAAAAA"].qty == 60 and fk.orders == {}       # 잔량 40 은 취소됐다
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    [s] = [i for i in fk.submitted if i.side == "sell"]
    assert s.qty == 60                                               # 사용자 50주는 팔지 않는다


# ---------------------------------------------------------------------- 10. M5 signal 하루 결손 → 다음 signal 이 따라잡는다


def test_missed_signal_job_catches_up_trail_exit(tmp_path):
    env = Env(tmp_path)
    H._enter(env, {"CCCCCC": {"trail": [11_800.0] * 49}})
    env.q("CCCCCC", 11_900, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)                       # 절반 익절 → 이후 MA50 추적
    assert env.state().positions["CCCCCC"].partial_done
    H._eod(env, D1, {"CCCCCC": 11_000})
    # D2 07:35 signal 이 안 돌았다(daily_finalize 지연/크론 미실행). D1 종가 11,000 < MA50 ≈ 11,784 → 연구는 D2 시가 매도.
    env.q("CCCCCC", 11_000, prev_close=11_000)
    env.run("preopen", D2, "08:35")
    env.q("CCCCCC", 11_000, open=11_000, prev_close=11_000)
    env.run("open_check", D2, "09:02")
    H._eod(env, D2, {"CCCCCC": 12_500})                              # D2 종가는 MA50 위로 복귀
    # D3 signal: D1·D2 종가를 차례로 평가 → D1 에서 추적 결정이 살아난다(하루 늦게 D3 시가 매도)
    env.q("CCCCCC", 12_500, prev_close=12_500)
    assert env.run("signal", D3, "07:35") == 0
    st = env.state()
    assert st.positions["CCCCCC"].exit_intent["kind"] == "trail_next_open"
    assert [r["date"] for r in st.equity_history][-2:] == [D1.isoformat(), D2.isoformat()]
    assert st.positions["CCCCCC"].last_eod == D2.isoformat()
    env.run("preopen", D3, "08:35")
    [o] = env.state().live_orders()
    assert (o.code, o.side, o.purpose, o.session) == ("CCCCCC", "sell", "trail_next_open", "open_auction")


def test_stale_signal_data_still_settles_held_positions(tmp_path):
    """DB 최신일이 직전 거래일보다 이르면(daily_finalize 지연) 신규 진입은 중단하되 있는 날짜까지의 보유 결정·자산은 한다."""
    env = Env(tmp_path)
    H._enter(env, {"CCCCCC": {"trail": [11_800.0] * 49}})
    env.q("CCCCCC", 11_900, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    H._eod(env, D1, {"CCCCCC": 11_000})
    env.q("CCCCCC", 11_000, prev_close=11_000)
    env.run("preopen", D2, "08:35")
    env.q("CCCCCC", 11_000, open=11_000, prev_close=11_000)
    env.run("open_check", D2, "09:02")
    H._eod(env, D2, {"CCCCCC": 12_500})
    env.signals.batches[D3] = SignalBatch(data_date=D1, candidates=[cand("EEEEEE", d=D1)],
                                          trail_closes={"CCCCCC": [11_800.0] * 49 + [11_000.0]})
    env.q("CCCCCC", 12_500, prev_close=12_500)
    assert env.run("signal", D3, "07:35") == 4                                  # 신규 진입 중단 = 종료 코드 4
    st = env.state()
    assert "EEEEEE" not in st.positions and env.events("signal_abort")
    assert st.positions["CCCCCC"].exit_intent["kind"] == "trail_next_open"
    assert st.equity_history[-1]["date"] == D1.isoformat()


# ---------------------------------------------------------------------- 11. 매도 지정가: 틱·하한가 경계


def test_exit_limit_ticks_and_limit_down_floor(tmp_path):
    env = Env(tmp_path)
    rt = SwingRuntime(env.cfg, env.bundle, clock=env.clock, alert=lambda *a: None)
    assert rt._exit_limit(9_153, 1) == 9_140                         # 내림 9,150 → 1틱 아래
    assert rt._exit_limit(2_000, 1) == 1_999                         # 밴드 경계: 아래 밴드 틱
    assert rt._exit_limit(1, 1) == 1                                 # 더 못 내리면 그대로
    assert rt._exit_limit(10_000, 2) == 9_950                        # 0.5%
    assert rt._exit_limit(10_000, 7) == 9_700                        # 3% 상한
    assert rt._exit_limit(10_000, 50) == 9_700


# ====================================================================== 최종 재검토(REVIEW_RUNTIME §9) 재현 시나리오


class TimeoutKiwoom(FakeKiwoom):
    """첫 매수 POST 가 '타임아웃' 으로 예외를 내지만 브로커엔 접수된다(응답 유실)."""

    def submit(self, intent):
        r = super().submit(intent)
        if intent.side == "buy" and len(self.submitted) == 1:
            raise TimeoutError("read timed out")
        return r


def _opened(env, fk, code="AAAAAA"):
    fk.match(code, 10_100)
    fk.publish()
    env.q(code, 10_150, open=10_100, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    assert env.state().positions[code].qty == 100


def test_f_c1_crash_after_post_is_adopted_not_duplicated(tmp_path, monkeypatch):
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    real = StateStore.save
    n = {"saves": 0}

    def crash_after_post(self, st):          # 의도 저장(1회)은 성공, 결과 저장(2회째)부터 프로세스가 죽는다(finally 저장도 못 한다)
        n["saves"] += 1
        if n["saves"] >= 2:
            raise KeyboardInterrupt()
        real(self, st)

    monkeypatch.setattr(StateStore, "save", crash_after_post)
    with pytest.raises(KeyboardInterrupt):
        env.run("preopen", D1, "08:35")
    monkeypatch.setattr(StateStore, "save", real)
    assert env.run("preopen", D1, "08:36") == 0
    assert len([i for i in fk.submitted if i.side == "buy"]) == 1          # 재발주 없음
    [o] = env.state().orders.values()
    assert o.status == "submitted" and o.ord_no == "0000001"               # 브로커 주문을 이어받았다(error 경보)
    assert any(a[0] == "error" and "이어받음" in a[1] for a in env.alerts)
    assert env.state().positions["AAAAAA"].status == "entry_pending"


def test_f1_resume_never_adopts_users_identical_order(tmp_path, monkeypatch):
    """F1: 사용자가 같은 종목·수량·가격으로 걸어 둔 주문(번호 순서를 못 가림)은 이어받지 않는다 → needs_human + 종목 차단."""
    fk = FakeKiwoom({}, foreign=[OpenOrder("U000009", "AAAAAA", "buy", 100, 100, 10_290)])
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    real = StateStore.save
    n = {"saves": 0}

    def crash_before_post(self, st):         # 의도 저장 직후(POST 전) 죽는다 → 우리 주문은 안 나갔다
        n["saves"] += 1
        real(self, st)
        if n["saves"] == 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(StateStore, "save", crash_before_post)
    with pytest.raises(KeyboardInterrupt):
        env.run("preopen", D1, "08:35")
    monkeypatch.setattr(StateStore, "save", real)
    assert fk.submitted == []
    assert env.run("preopen", D1, "08:36") == 0
    st = env.state()
    [o] = st.orders.values()
    assert o.status == "needs_human" and o.ord_no is None                 # 남의 주문을 가져가지 않았다
    assert fk.submitted == [] and "AAAAAA" in st.blocked_codes             # 재발주도 없고 종목은 차단
    assert any(a[0] == "error" and "주문 미해결" in a[1] for a in env.alerts)
    # 차단은 다음 날 신규 계획도 막는다; 사람이 release-order 로 풀면 그 다음부터 산다
    env.signals.batches[D2] = SignalBatch(data_date=D1, candidates=[cand("AAAAAA", d=D1)])
    env.run("signal", D2, "07:35")
    env.run("preopen", D2, "08:35")
    assert fk.submitted == [] and _closed(env, "AAAAAA")[-1].exit_reason.startswith("blocked")
    assert env.run("release_order", D2, "10:00", oid=o.oid) == 0
    st = env.state()
    assert st.orders[o.oid].status == "released" and "AAAAAA" not in st.blocked_codes
    env.signals.batches[D3] = SignalBatch(data_date=D2, candidates=[cand("AAAAAA", d=D2)])
    env.run("signal", D3, "07:35")
    env.run("preopen", D3, "08:35")
    assert [i.code for i in fk.submitted] == ["AAAAAA"]


def test_f1_resume_adopts_only_newer_single_numeric_order(tmp_path, monkeypatch):
    """두 후보(우리 번호보다 작은 것·큰 것)가 있으면 못 가린다 → needs_human; 하나뿐이고 번호가 더 크면 이어받는다."""
    fk = FakeKiwoom({}, foreign=[OpenOrder("0000001", "AAAAAA", "buy", 100, 100, 10_290)])
    env = Env(tmp_path, mode="live", broker=fk)
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA"), cand("BBBBBB")])
    for c in ("AAAAAA", "BBBBBB"):
        env.q(c, 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    real = StateStore.save
    seen = {"n": 0}

    def crash_on_second_result(self, st):   # A: 의도+결과 저장 OK(번호 0000001 → 가짜도 0000001 로 겹침 방지 위해 n=1), B: 결과 저장에서 죽음
        seen["n"] += 1
        if seen["n"] == 4:
            raise KeyboardInterrupt()
        real(self, st)

    fk.n = 5                                                               # 우리 주문은 0000006 부터
    monkeypatch.setattr(StateStore, "save", crash_on_second_result)
    with pytest.raises(KeyboardInterrupt):
        env.run("preopen", D1, "08:35")
    monkeypatch.setattr(StateStore, "save", real)
    assert env.run("preopen", D1, "08:36") == 0
    st = env.state()
    by_code = {o.code: o for o in st.orders.values()}
    assert by_code["AAAAAA"].status == "submitted" and by_code["AAAAAA"].ord_no == "0000006"
    assert by_code["BBBBBB"].status == "submitted" and by_code["BBBBBB"].ord_no == "0000007"   # 유일·더 큰 번호 → 채택
    assert len(fk.submitted) == 2 and "BBBBBB" not in st.blocked_codes


def test_f2_unknown_without_ord_no_is_resumed_and_cancelled_at_open(tmp_path):
    fk = TimeoutKiwoom({})
    env = _live_entry(tmp_path, fk)
    [o] = env.state().orders.values()
    assert o.status == "unknown" and o.ord_no is None and list(fk.orders) == ["0000001"]
    env.q("AAAAAA", 10_350, open=10_300, prev_close=10_000)                # 시가 > 지정가 → 미체결로 남는 주문
    assert env.run("open_check", D1, "09:02") == 0
    [o] = env.state().orders.values()
    assert o.status == "canceled" and o.ord_no == "0000001"                # 대조로 번호를 찾아 09:02 에 취소했다
    assert fk.orders == {}
    [a] = _closed(env, "AAAAAA")
    assert a.status == "missed"
    env.q("AAAAAA", 10_290, open=10_300, prev_close=10_000)                # 11:00 가격이 내려와도 추격 체결 없음
    fk.match("AAAAAA", 10_290)
    env.run("monitor", D1, "11:00", once=True)
    assert env.book().positions() == {} and fk._h.get("AAAAAA") is None and env.state().baseline["user"] == {}


def test_f2_unknown_without_ord_no_unresolvable_blocks_code_and_is_never_expired(tmp_path):
    """접수 여부를 끝내 모르면(미체결에도 없음) unknown 그대로 + 종목 차단 + 경보; expired 로 바꾸지 않는다."""
    class VanishKiwoom(FakeKiwoom):
        def submit(self, intent):
            self.submitted.append(intent)
            raise TimeoutError("read timed out")

    fk = VanishKiwoom({})
    env = _live_entry(tmp_path, fk)
    env.q("AAAAAA", 10_350, open=10_300, prev_close=10_000)
    env.run("open_check", D1, "09:02")
    env.run("eod", D1, "15:40")
    st = env.state()
    [o] = st.orders.values()
    assert o.status == "unknown" and o.ord_no is None and "AAAAAA" in st.blocked_codes
    assert any(a[0] == "error" and "주문 미해결" in a[1] for a in env.alerts)


def test_f3_fill_pending_never_deadlocks_across_days(tmp_path):
    """사용자가 HTS 에서 우리 매도를 취소 → 예약 해제 → 그날 다시 손절이 나간다(검토 C2 교착 시나리오)."""
    fk = FakeKiwoom({})
    env = _live_entry(tmp_path, fk)
    _opened(env, fk)
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    [o] = env.state().live_orders()
    fk.orders.pop(o.ord_no)
    fk.remaining.pop(o.ord_no)                                            # HTS 수동 취소(잔고 변화 없음)
    env.q("AAAAAA", 9_100, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:31", once=True)                             # 재가격 → 취소 거부 → released → 새 손절
    sells = [i for i in fk.submitted if i.side == "sell"]
    assert len(sells) == 2 and list(fk.orders)                            # 두 번째 손절이 브로커에 살아 있다
    assert not [o for o in env.state().orders.values() if o.status == "fill_pending"]


def test_f3_release_order_cli_flag_required():
    from swing_it.live import cli
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        cli.main(["release-order"])
    assert parser.parse_args(["release-order", "--oid", "L000001"]).oid == "L000001"


def test_f4_real_holiday_with_stale_open_does_not_fill_paper(tmp_path):
    """달력에 없는 휴장일: ka10081 은 어제 봉까지, ka10001 은 어제 시가를 그대로(캐시) → 세션 미확정 → 체결·day_no 전진 없음."""
    env = Env(tmp_path)
    env.quotes.set_bars(H.REF_CODE, [H.DailyBar(d, 1, 1, 1, 1) for d in H.SESSIONS if d < D1])   # D1 봉 없음
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    env.q("AAAAAA", 10_000, open=9_950, prev_close=10_000)                # 어제 시가·종가가 그대로 온다(일자 없음)
    assert env.run("open_check", D1, "09:02") == 0
    st = env.state()
    assert st.sessions[D1.isoformat()] == "suspect" and env.events("holiday_suspect")
    assert st.positions["AAAAAA"].status == "entry_pending" and not env.events("fill")
    for hhmm in ("09:06", "10:30"):
        assert env.run("monitor", D1, hhmm, once=True) == 0
    assert env.run("close", D1, "15:21") == 0
    assert env.run("eod", D1, "15:40") == 0
    st = env.state()
    assert st.sessions[D1.isoformat()] == "suspect" and not env.events("fill") and not st.live_orders()
    assert env.run("signal", D2, "07:35") == 4      # 가짜 원천의 DB 최신일(D1)이 키움 직전 거래일(10-09)과 달라 진입 중단
    assert [p.exit_reason for p in env.state().history] == ["stale_plan"]


def test_f4_real_holiday_live_orders_are_released_not_stuck(tmp_path):
    """같은 상황 실계좌: 주문은 브로커에 없고(휴장 거부) 취소도 거부 → eod 가 released 로 풀고 경보(fill_pending 교착 없음)."""
    fk = FakeKiwoom({})
    env = Env(tmp_path, mode="live", broker=fk)
    env.quotes.set_bars(H.REF_CODE, [H.DailyBar(d, 1, 1, 1, 1) for d in H.SESSIONS if d < D1])
    env.signals.batches[D1] = SignalBatch(data_date=SIGNAL_D, candidates=[cand("AAAAAA")])
    env.q("AAAAAA", 10_000, prev_close=10_000)
    env.run("signal", D1, "07:35")
    env.run("preopen", D1, "08:35")
    fk.orders.clear()
    fk.remaining.clear()                                                   # 키움이 휴장일 주문을 거부/소멸
    env.q("AAAAAA", 10_000, open=9_950, prev_close=10_000)
    assert env.run("open_check", D1, "09:02") == 0
    [o] = env.state().orders.values()
    assert o.status == "submitted" and env.state().positions["AAAAAA"].status == "entry_pending"   # 세션 미확정 → 손대지 않음
    env.run("eod", D1, "15:40")
    [o] = env.state().orders.values()
    assert o.status == "released" and env.events("order_released")
    env.run("signal", D2, "07:35")
    assert [p.exit_reason for p in env.state().history] == ["stale_plan"] and env.state().blocked_codes == {}


def _catch_up_env(tmp_path, outage_days: int):
    """CCCCCC 절반 익절 뒤 MA50 추적 단계. D1 signal 까지 평가됨. 그 뒤 outage_days 거래일 결손, 결손 2일째 종가 < MA50.
    배치 크기는 실제 원천처럼 결손 길이에 맞춘다(m10: trail_ma + max(15, 결손 + 5))."""
    import pandas as pd
    from swing_it.live.runtime import CATCHUP_DAYS

    env = Env(tmp_path)
    H._enter(env, {"CCCCCC": {"trail": [11_800.0] * 49}})
    env.q("CCCCCC", 11_900, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    assert env.state().positions["CCCCCC"].partial_done
    H._eod(env, D1, {"CCCCCC": 11_900})
    env.q("CCCCCC", 11_900, prev_close=11_900)
    env.run("signal", D2, "07:35")
    assert env.state().positions["CCCCCC"].last_eod == D1.isoformat()
    days = [d.date() for d in pd.bdate_range(D2, periods=outage_days)]
    back = [d.date() for d in pd.bdate_range(days[-1], periods=2)][-1]
    n = 50 + max(CATCHUP_DAYS, outage_days + 5)
    tail = [11_900.0] + [11_000.0 if i == 1 else 12_600.0 for i in range(outage_days)]   # 결손 2일째만 MA 아래
    closes = ([11_800.0] * (n - len(tail)) + tail)[-n:]
    dts = [str(x.date()) for x in pd.bdate_range(end=days[-1], periods=len(closes))]
    env.signals.batches[back] = SignalBatch(data_date=days[-1], candidates=[], trail_closes={"CCCCCC": closes},
                                            trail_dates={"CCCCCC": dts})
    env.quotes.set_bars(H.REF_CODE, [H.DailyBar(d, 1, 1, 1, 1) for d in H.SESSIONS] + [H.DailyBar(d, 1, 1, 1, 1) for d in days])
    env.q("CCCCCC", 12_600, prev_close=12_600)
    assert env.run("signal", back, "07:35") == 0
    return env, days


def test_m5_catch_up_two_week_outage_sells_what_it_should(tmp_path):
    env, days = _catch_up_env(tmp_path, 10)
    st = env.state()
    assert st.positions["CCCCCC"].last_eod == days[-1].isoformat()
    it = st.positions["CCCCCC"].exit_intent
    assert it and it["kind"] == "trail_next_open" and it["qty"] == 50 and it["session"] == "open_auction"
    assert len([a for a in env.alerts if "MA50 계산 불가" in a[1]]) == 0
    assert len([e for e in env.events("action") if e["kind"] == "trail_next_open"]) == 1       # 한 번만
    assert env.signals.since == D1                                            # 원천에 결손 시작일을 넘겼다


def test_m10_catch_up_beyond_catchup_days_keeps_ma(tmp_path):
    from swing_it.live.runtime import CATCHUP_DAYS

    env, days = _catch_up_env(tmp_path, CATCHUP_DAYS + 3)
    st = env.state()
    assert st.positions["CCCCCC"].last_eod == days[-1].isoformat()
    assert st.positions["CCCCCC"].exit_intent["kind"] == "trail_next_open"   # 결손 2일째 추적 매도가 살아 있다
    assert not [a for a in env.alerts if "MA50 계산 불가" in a[1]]
    assert [a for a in env.alerts if "장기 결손" in a[1]]


def test_m10_features_source_sizes_batch_by_gap(monkeypatch):
    import swing_it.live.features as features
    import swing_it.live.strategy as strategy
    import swing_it.storage as storage
    from swing_it.live.runtime import FeaturesSignalSource

    import pandas as pd
    days = pd.bdate_range("2026-01-05", "2026-10-09")
    close = pd.DataFrame({"AAAAAA": range(len(days))}, index=days, dtype=float)
    panels = {"close": close, "patch_info": {"nxt_rows": 1, "nxt_patched": 1, "nxt_unpatched": pd.DataFrame(columns=["code", "date"])}}

    class _Con:
        def close(self):
            pass

    monkeypatch.setattr(storage, "connect", lambda db=None: _Con())
    monkeypatch.setattr(storage, "db_default", lambda: "x")
    monkeypatch.setattr(features, "signal_panels_for", lambda con, d, cfg: (panels, days[-1]))
    monkeypatch.setattr(strategy, "scan", lambda p, d, cfg: [])
    b = FeaturesSignalSource().load(D1, held=["AAAAAA"], since=H.date(2026, 8, 1))   # 50거래일 결손
    gap = int(__import__("numpy").busday_count(H.date(2026, 8, 1), D1))
    assert len(b.trail_closes["AAAAAA"]) == 50 + gap + 5 and len(b.trail_dates["AAAAAA"]) == len(b.trail_closes["AAAAAA"])
    assert b.trail_dates["AAAAAA"][-1] == "2026-10-09"
    b2 = FeaturesSignalSource().load(D1, held=["AAAAAA"])
    assert len(b2.trail_closes["AAAAAA"]) == 65


def test_ledger_oms_keeps_core_buy_guard_and_only_counts_own_sells(tmp_path):
    from krx_quant_core.execution.orders import OrderIntent

    from swing_it.live.brokers import entry_guard_config

    fk = FakeKiwoom({}, foreign=[OpenOrder("U000001", "AAAAAA", "sell", 100, 100, 12_000)])
    env = _live_entry(tmp_path, fk)
    _opened(env, fk)
    env.q("AAAAAA", 9_150, open=10_100, prev_close=10_000)
    env.run("monitor", D1, "10:30", once=True)
    [s] = [i for i in fk.submitted if i.side == "sell"]
    assert s.qty == 100                                                     # 남의 미체결은 예약이 아니다
    rt = SwingRuntime(env.cfg, env.bundle, clock=env.clock, alert=lambda *a: None)
    st = env.state()
    ctx = rt._make_ctx("monitor", D1, st, StateStore(env.cfg.mode_dir, clock=env.clock).load_book(st))
    ctx.oms.guard.config = entry_guard_config(equity=10_000_000, price=10_000, hard_cap_frac=0.12,
                                              price_band_pct=0.06, max_orders_total=20)
    m = ctx.oms.buy("BBBBBB", 121, 10_000, ref_price=10_000)
    assert m.result.blocked and "하드캡" in m.result.blocked_reason
    m = ctx.oms.buy("BBBBBB", 10, 10_700, ref_price=10_000)
    assert m.result.blocked and "밖" in m.result.blocked_reason
    assert OrderIntent(side="sell", code="AAAAAA", qty=1, price=1).exchange == "KRX"
