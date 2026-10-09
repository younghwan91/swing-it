"""스윙 라이브 브로커 배선(``swing_it.live.brokers``) — 모드 안전 불변식·페이퍼 영속·동시호가 에뮬·REST 파싱."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from krx_quant_core.execution.guards import OrderGuard
from krx_quant_core.execution.kiwoom_broker import KiwoomBroker
from krx_quant_core.execution.orders import OrderIntent
from krx_quant_core.market.session import KST

from swing_it.live.brokers import (
    REF_CODE,
    DailyBar,
    FakeQuoteProvider,
    PersistentPaperBroker,
    RestQuoteProvider,
    build_bundle,
    entry_guard_config,
    feed_auction,
    session_date,
)

T9 = datetime(2026, 10, 12, 9, 0, tzinfo=KST)


class _NoNetApi:
    """주문·조회가 불리면 실패하는 가짜 키움 api — dry-run 이 POST 하지 않음을 증명한다."""

    def __init__(self):
        self.order = SimpleNamespace(buy_order=self._boom, sell_order=self._boom, cancel_order=self._boom)

    @staticmethod
    def _boom(**kw):
        raise AssertionError(f"주문 POST 가 나갔다: {kw}")


def test_live_requires_confirm_real(tmp_path):
    fq = FakeQuoteProvider()
    with pytest.raises(ValueError):
        build_bundle("live", confirm_real=False, paper_path=tmp_path / "p.json", api=_NoNetApi(), quotes=fq)
    with pytest.raises(ValueError):
        build_bundle("paper", confirm_real=True, paper_path=tmp_path / "p.json", quotes=fq)
    with pytest.raises(ValueError):
        build_bundle("dry-run", confirm_real=True, paper_path=tmp_path / "p.json", api=_NoNetApi(), quotes=fq)
    with pytest.raises(ValueError):
        build_bundle("real", confirm_real=False, paper_path=tmp_path / "p.json", quotes=fq)


def test_dry_run_never_posts(tmp_path):
    b = build_bundle("dry-run", confirm_real=False, paper_path=tmp_path / "p.json", api=_NoNetApi(),
                     quotes=FakeQuoteProvider())
    assert isinstance(b.broker, KiwoomBroker) and b.broker.dry_run and b.fills_from_holdings
    r = b.broker.submit(OrderIntent(side="buy", code="111111", qty=1, price=1000))
    assert r.dry_run and not r.submitted and r.intent.exchange == "KRX" and r.intent.body()["trde_tp"] == "0"


def test_live_bundle_is_not_dry_run(tmp_path):
    b = build_bundle("live", confirm_real=True, paper_path=tmp_path / "p.json", api=_NoNetApi(),
                     quotes=FakeQuoteProvider())
    assert isinstance(b.broker, KiwoomBroker) and not b.broker.dry_run


def test_paper_bundle_refuses_real_broker(tmp_path):
    with pytest.raises(TypeError):
        build_bundle("paper", confirm_real=False, paper_path=tmp_path / "p.json", quotes=FakeQuoteProvider(),
                     broker=KiwoomBroker(_NoNetApi(), dry_run=False))


def test_paper_broker_persists(tmp_path):
    path = tmp_path / "paper.json"
    b = PersistentPaperBroker.load(path)
    r1 = b.submit(OrderIntent(side="buy", code="111111", qty=10, price=10_290))
    b.submit(OrderIntent(side="buy", code="222222", qty=5, price=20_000))
    feed_auction(b, "111111", T9, 10_100)          # 체결 → 버퍼에 남긴 채 저장
    b.save()
    b2 = PersistentPaperBroker.load(path)
    assert b2.holdings()["111111"].qty == 10
    assert [o.code for o in b2.open_orders()] == ["222222"]
    fills = b2.poll_fills()
    assert len(fills) == 1 and fills[0].ord_no == r1.ord_no and fills[0].price == 10_100
    r3 = b2.submit(OrderIntent(side="sell", code="111111", qty=10, price=9_000))
    assert r3.ord_no == "P000003"                   # 주문번호가 이어진다
    assert b2.submit(OrderIntent(side="sell", code="111111", qty=1, price=9_000)).blocked  # 걸린 매도 포함 보유 초과


def test_auction_fill_rule(tmp_path):
    b = PersistentPaperBroker(None)
    b.submit(OrderIntent(side="buy", code="A00001", qty=1, price=10_290))
    b.submit(OrderIntent(side="buy", code="B00001", qty=1, price=10_290))
    assert feed_auction(b, "A00001", T9, 10_290)[0].price == 10_290   # 시가 = 지정가 → 체결
    assert feed_auction(b, "B00001", T9, 10_300) == []                 # 시가 > 지정가 → 미체결(추격 없음)
    b2 = PersistentPaperBroker(None, holdings=b.holdings())
    b2.submit(OrderIntent(side="sell", code="A00001", qty=1, price=9_000))
    assert feed_auction(b2, "A00001", T9, 9_500)[0].price == 9_500     # 매도: 단일가 ≥ 지정가 → 단일가


def test_entry_guard_config_sized_per_order():
    cfg = entry_guard_config(equity=10_000_000, price=10_290, hard_cap_frac=0.12, price_band_pct=0.06,
                             max_orders_total=20)
    assert cfg.max_qty == cfg.max_qty_hard_cap == 116
    g = OrderGuard(cfg)
    assert g.reason(OrderIntent(side="buy", code="111111", qty=100, price=10_290), 10_000) is None
    assert "하드캡" in g.reason(OrderIntent(side="buy", code="111111", qty=117, price=10_290), 10_000)
    assert "밖" in g.reason(OrderIntent(side="buy", code="111111", qty=10, price=10_700), 10_000)


def test_rest_quote_parsing():
    api = SimpleNamespace(
        stock_info=SimpleNamespace(basic_stock_info=lambda **kw: {
            "return_code": 0, "cur_prc": "-10,200", "open_pric": "+10100", "high_pric": "+10400",
            "low_pric": "-9900", "base_pric": "10000"}),
        chart=SimpleNamespace(stock_daily_chart=lambda **kw: {"stk_dt_pole_chart_qry": [
            {"dt": "20261012", "open_pric": "+1", "high_pric": "2", "low_pric": "1", "cur_prc": "-2", "trde_qty": "5"},
            {"dt": "20261009", "open_pric": "1", "high_pric": "1", "low_pric": "1", "cur_prc": "1"},
            {"dt": "", "cur_prc": "1"},
        ]}),
    )
    rq = RestQuoteProvider(api, clock=lambda: T9)
    q = rq.quote("111111")
    assert (q.last, q.open, q.high, q.low, q.prev_close) == (10_200, 10_100, 10_400, 9_900, 10_000)
    bars = rq.daily_bars(REF_CODE)
    assert [b.date for b in bars] == [date(2026, 10, 9), date(2026, 10, 12)]
    assert session_date(rq, date(2026, 10, 12)) == date(2026, 10, 12)
    assert session_date(rq, date(2026, 10, 11)) == date(2026, 10, 9)


def test_rest_quote_without_price_raises():
    api = SimpleNamespace(stock_info=SimpleNamespace(basic_stock_info=lambda **kw: {"return_code": 0}))
    with pytest.raises(ValueError):
        RestQuoteProvider(api, clock=lambda: T9).quote("111111")


def test_session_date_failure_is_none():
    fq = FakeQuoteProvider(clock=lambda: T9)
    fq.fail.add(REF_CODE)
    assert session_date(fq, date(2026, 10, 12)) is None
    fq.fail.clear()
    fq.set_bars(REF_CODE, [DailyBar(date(2026, 10, 9), 1, 1, 1, 1)])
    assert session_date(fq, date(2026, 10, 12)) == date(2026, 10, 9)


def test_rest_quote_rejects_nxt_or_integrated_codes():
    """시세는 KRX 만 — ``_NX``(NXT)·``_AL``(통합) 접미 코드는 조회 전에 거부한다."""
    calls = []
    api = SimpleNamespace(stock_info=SimpleNamespace(basic_stock_info=lambda **kw: calls.append(kw) or {}),
                          chart=SimpleNamespace(stock_daily_chart=lambda **kw: calls.append(kw) or {}))
    rq = RestQuoteProvider(api, clock=lambda: T9)
    for bad in ("005930_NX", "005930_AL", "5930"):
        with pytest.raises(ValueError):
            rq.quote(bad)
        with pytest.raises(ValueError):
            rq.daily_bars(bad)
    assert calls == []
