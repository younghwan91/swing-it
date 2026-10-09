"""스윙 라이브 상태(``swing_it.live.state``) — 스냅샷 복원·손상 거부·기준선·다일 낙폭 킬."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from krx_quant_core.execution.events import Fill
from krx_quant_core.market.session import KST

from swing_it.live.state import (
    LiveState,
    OrderRec,
    PositionRec,
    StateCorrupt,
    StateStore,
    compute_baseline_user,
    drawdown_check,
)

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=KST)


def _store(tmp_path):
    return StateStore(tmp_path / "paper", clock=lambda: T0)


def test_fresh_state_uses_capital_once(tmp_path):
    s = _store(tmp_path)
    st = s.load(mode="paper", capital=5_000_000)
    assert st.capital == 5_000_000 and st.equity_peak == 5_000_000
    s.save(st)
    # 두 번째 실행은 다른 자본을 줘도 스냅샷 값이 정본
    assert s.load(mode="paper", capital=1).capital == 5_000_000


def test_roundtrip_positions_orders(tmp_path):
    s = _store(tmp_path)
    st = s.load(mode="paper", capital=1e7)
    st.positions["111111"] = PositionRec(code="111111", market="KOSPI", signal_date="2026-10-09",
                                         trade_date="2026-10-12", status="open", qty=10,
                                         strat={"phase": "full"}, exit_intent={"kind": "stop", "qty": 10},
                                         trail_closes=[1.0, 2.0])
    oid = st.next_oid()
    st.orders[oid] = OrderRec(oid=oid, trade_date="2026-10-12", ts=T0.isoformat(), code="111111", side="sell",
                              qty=10, price=9000, purpose="stop", session="continuous", status="submitted",
                              ord_no="P000001")
    st.own_ord_nos = ["P000001"]
    s.save(st)
    back = s.load(mode="paper", capital=0)
    assert back.positions["111111"].exit_intent == {"kind": "stop", "qty": 10}
    assert back.positions["111111"].trail_closes == [1.0, 2.0]
    assert back.orders[oid].live and back.orders[oid].remaining == 10
    assert back.seq == 1 and back.next_oid() == "L000002"
    assert (tmp_path / "paper" / "state.json.bak").exists() is False  # 첫 저장엔 백업 없음
    s.save(back)
    assert (tmp_path / "paper" / "state.json.bak").exists()


def test_corrupt_snapshot_refused(tmp_path):
    s = _store(tmp_path)
    s.root.mkdir(parents=True)
    s.state_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StateCorrupt):
        s.load(mode="paper", capital=1e7)


def test_mode_mismatch_refused(tmp_path):
    s = _store(tmp_path)
    s.save(LiveState(mode="paper", capital=1e7))
    with pytest.raises(StateCorrupt, match="모드"):
        s.load(mode="live", capital=1e7)


def test_missing_snapshot_with_ledger_refused(tmp_path):
    """스냅샷이 사라졌는데 원장이 있으면 새 상태로 시작하지 않는다(보유를 잊고 다시 사는 사고)."""
    s = _store(tmp_path)
    s.event("order", code="111111")
    with pytest.raises(StateCorrupt, match="스냅샷 유실"):
        s.load(mode="paper", capital=1e7)


def test_ledger_skips_torn_last_line(tmp_path):
    s = _store(tmp_path)
    s.event("a", x=1)
    with open(s.ledger_path, "a", encoding="utf-8") as f:
        f.write('{"ts": "2026-10-12", "event": "b"')   # 크래시로 잘린 줄
    assert [e["event"] for e in s.events()] == ["a"]


def test_book_journal_restores_with_market(tmp_path):
    s = _store(tmp_path)
    st = LiveState(mode="paper", capital=1e7, market_of={"111111": "코스닥"})
    book = s.load_book(st)
    book.apply(Fill(ord_no="P1", code="111111", side="buy", qty=10, price=10_000, ts=T0))
    book2 = s.load_book(st)
    assert book2.position("111111").qty == 10
    pnl = book2.apply(Fill(ord_no="P2", code="111111", side="sell", qty=10, price=11_000, ts=T0))
    assert 0 < pnl < 10_000          # 수수료·세금 차감
    assert s.load_book(st).position("111111") is None


def test_baseline_user_split():
    user, short = compute_baseline_user({"A": 130, "B": 50, "C": 5}, {"A": 100, "C": 10})
    assert user == {"A": 30, "B": 50}
    assert short == {"C": 5}


def test_drawdown_kill_triggers_and_ack(tmp_path):
    stop = tmp_path / "swing_paper_STOP"
    st = LiveState(mode="paper", capital=1e7, equity_peak=1e7)
    fired = []
    drawdown_check(st, date="2026-10-12", equity=9_000_000, cash=0, market_value=0, kill_pct=0.15,
                   stop_file=stop, on_trigger=fired.append)
    assert not stop.exists() and not fired
    row = drawdown_check(st, date="2026-10-13", equity=8_400_000, cash=0, market_value=0, kill_pct=0.15,
                         stop_file=stop, on_trigger=fired.append)
    assert row["dd"] == pytest.approx(-0.16)
    assert stop.exists() and fired and st.drawdown_kill["active"]
    # 파일이 남아 있으면 계속 킬
    drawdown_check(st, date="2026-10-14", equity=8_500_000, cash=0, market_value=0, kill_pct=0.15, stop_file=stop)
    assert st.drawdown_kill["active"] and st.equity_peak == 1e7
    # 사람이 지움 = 확인 → 고점을 지금 자산으로
    stop.unlink()
    drawdown_check(st, date="2026-10-15", equity=8_500_000, cash=0, market_value=0, kill_pct=0.15, stop_file=stop)
    assert not st.drawdown_kill["active"] and st.equity_peak == 8_500_000 and not stop.exists()
    # 같은 날 다시 부르면 덮어쓴다(멱등)
    drawdown_check(st, date="2026-10-15", equity=8_600_000, cash=0, market_value=0, kill_pct=0.15, stop_file=stop)
    assert [r["date"] for r in st.equity_history].count("2026-10-15") == 1


def test_state_json_is_plain_json(tmp_path):
    s = _store(tmp_path)
    s.save(LiveState(mode="paper", capital=1e7))
    raw = json.loads(s.state_path.read_text(encoding="utf-8"))
    assert raw["schema"] == 1 and raw["mode"] == "paper"
