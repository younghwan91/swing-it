"""스윙 라이브 브로커 배선 — 페이퍼(코어 ``PaperBroker`` + 동시호가 에뮬)·키움(``KiwoomBroker``)·시세 공급자.

모드 셋(:data:`MODES`):

* ``paper`` (기본) — 주문은 프로세스 밖으로 **절대** 나가지 않는다. 코어 :class:`PaperBroker` 를 파일에 저장·복원하는
  :class:`PersistentPaperBroker` 가 받는다(크론 작업이 프로세스마다 따로 떠서, 메모리 브로커로는 08:30 에 낸 주문을
  09:01 작업이 볼 수 없다). 시세는 실제 REST 조회(읽기 전용)거나 테스트의 가짜.
* ``dry-run`` — :class:`KiwoomBroker` ``dry_run=True``. 실계좌 **조회**(잔고·미체결)는 하고 주문·취소는 POST 하지 않는다.
* ``live`` — :class:`KiwoomBroker` ``dry_run=False``. :func:`build_bundle` 은 ``confirm_real=True`` 일 때만 이것을 만든다
  (CLI 는 ``--execute --confirm-real`` 두 플래그가 다 있을 때만 ``confirm_real=True`` 를 넘긴다).

동시호가 에뮬레이션(페이퍼):

* 장전(08:30~09:00) 지정가 매수는 09:00 단일가에 **시가 ≤ 지정가일 때만 시가로** 체결된다. 그래서 open_check 작업이
  그날 시가를 ``Quote(ts=09:00, bid=시가, ask=시가)`` 로 한 번 먹인다 — 코어 판정(``ask ≤ 지정가`` → ``ask`` 체결)이 곧
  "시가 ≤ 지정가 → 시가 체결" 이다. 장전 매도(전일 결정 추적 청산)도 같은 시세로 ``bid(=시가) ≥ 지정가`` 면 시가 체결.
* 종가 동시호가(15:20~15:30) 매도는 eod 작업이 그날 종가를 ``Quote(ts=15:30, bid=종가, ask=종가)`` 로 먹인다.
* 장중 시세는 호가(bid/ask)가 있으면 그것을, 없으면 ``bid = ask = 현재가`` 로 먹인다(근사 — 스프레드 1틱 무시).

KRX 정규장만(2026-09-14~ 키움 일봉에 KRX 애프터마켓(16:00~20:00) 체결이 섞였다 — :mod:`swing_it.krx_session`):

* 키움 REST 의 종목코드 규약(키움 REST 문서): ``"005930"`` = KRX, ``"005930_NX"`` = NXT, ``"005930_AL"`` = SOR(통합).
  :class:`RestQuoteProvider` 는 **6자리 숫자 코드만** 받는다(접미 붙은 코드는 ``ValueError``) — 시세는 항상 KRX 다.
  이 규약은 문서 기준이고 이 레포에서 실호출로 확인하지 않았다(scalp-it 이 6자리로 ``ka10001`` 을 쓰고 있다).
* 주문 거래소는 코어 ``OrderIntent`` 기본 ``dmst_stex_tp="KRX"`` (실주문으로 확인된 값). 런타임은 바꾸지 않는다.
* ``ka10081`` 일봉은 **가격에 쓰지 않는다**(전일까지의 종가가 애프터마켓 19:59 체결일 수 있다). 날짜(``dt``)만 휴장 판정에
  쓴다 — quant-airflow 일봉 수집기의 실측("장중에도 오늘 봉을 첫 행으로 준다", 2026-08-24)에 기댄다.
* 장 마감 판단 가격(eod)은 15:31~16:00 사이 ``ka10001`` ``cur_prc`` = KRX 15:30 단일가. 16:00 부터는 KRX 시간외
  단일가가 시작되므로 eod 시간창이 16:00 에 닫힌다.

미검증 필드(REST, 실호출 확인 안 됨 — :class:`RestQuoteProvider`):

* ``ka10001`` ``cur_prc`` 는 scalp-it 이 실호출로 확인했다(2026-08-25). ``open_pric``·``high_pric``·``low_pric``·
  ``base_pric``(기준가=전일 KRX 종가)은 키움 문서 필드명이고 이 응답에서는 **미확인**. 장전(08:30~09:00)의 ``base_pric``·
  ``cur_prc`` 의미도 미확인 — preopen 은 ``base_pric`` 가 없으면 계획 기준가(DB 보정 종가)를 쓴다.
* ``ka10081`` (``stk_dt_pole_chart_qry`` · ``dt``)은 quant-airflow 가 매일 쓰는 필드라 확인된 것으로 본다.
* 호가(``ka10004`` ``sel_fpr_bid``/``buy_fpr_bid``)는 쓰지 않는다(daytrade-it 도 README 관례 추정값). 매도 지정가는 현재가
  기준 할인으로 정한다 — 연속매매에서 최우선 매수호가보다 낮은 매도 지정가는 최우선 매수호가에 체결된다.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from krx_quant_core.execution.events import Fill, Holding
from krx_quant_core.execution.guards import OrderGuardConfig
from krx_quant_core.execution.kiwoom_broker import KiwoomBroker
from krx_quant_core.execution.orders import OrderIntent
from krx_quant_core.execution.paper import PaperBroker, _Pending
from krx_quant_core.market.session import KST, now_kst

from .state import atomic_write_json

__all__ = [
    "MODES",
    "REPLAY_JOB_TIMES",
    "BrokerBundle",
    "DBQuoteProvider",
    "SimClock",
    "DailyBar",
    "FakeQuoteProvider",
    "PersistentPaperBroker",
    "QuoteProvider",
    "QuoteSnap",
    "RestQuoteProvider",
    "build_bundle",
    "entry_guard_config",
    "feed_auction",
    "feed_quote",
    "make_kiwoom_api",
]

MODES = ("paper", "dry-run", "live")

#: 휴장 판정·세션 날짜용 기준 종목(삼성전자) — quant-airflow ``daily_bars._REF_CODE`` 와 같다.
REF_CODE = "005930"


# --------------------------------------------------------------------------- 시세


@dataclass(frozen=True)
class QuoteSnap:
    """한 종목의 현재 시세 한 장. 없는 값은 ``None``."""

    code: str
    ts: datetime
    last: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    prev_close: float | None = None
    bid: float | None = None
    ask: float | None = None
    #: 시세 자체가 단 일자(KST). 오늘과 같을 때만 "장이 열렸다" 는 증거로 쓴다(검토 F4). ``None`` = 모름.
    date: date | None = None


@dataclass(frozen=True)
class DailyBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class QuoteProvider(Protocol):
    """시세 공급자. 실구현은 :class:`RestQuoteProvider`, 테스트는 :class:`FakeQuoteProvider`."""

    def quote(self, code: str) -> QuoteSnap:
        """현재 시세. 실패하면 예외(호출부가 그 종목만 건너뛴다)."""
        ...

    def daily_bars(self, code: str, n: int = 60) -> list[DailyBar]:
        """일봉(오래된 것 → 최신). 장중·장 마감 직후엔 **오늘 봉이 마지막**일 수 있다."""
        ...


def session_date(quotes: QuoteProvider, asof: date) -> date | None:
    """기준 종목 일봉의 최신 날짜(≤ ``asof``). 장이 열린 날 09:00 이후면 ``asof`` 다. 조회 실패면 ``None``."""
    try:
        bars = quotes.daily_bars(REF_CODE, 10)
    except Exception:  # noqa: BLE001 — 모르면 None(호출부가 보수적으로 판단)
        return None
    days = [b.date for b in bars if b.date <= asof]
    return max(days) if days else None


def _ymd(value: Any) -> date | None:
    """``"20261012"`` → ``date``. 형식이 아니면 ``None``."""
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    """키움 부호 붙은 문자열(``"+10200"``·``"-9,900"``) → 절대값 float. 비었으면 ``None``."""
    if value in (None, ""):
        return None
    text = str(value).strip().lstrip("+-").replace(",", "")
    if not text:
        return None
    try:
        out = abs(float(text))
    except ValueError:
        return None
    return out if math.isfinite(out) else None


class RestQuoteProvider:
    """키움 REST 시세(읽기 전용). ``api`` 는 ``kiwoom_client.KiwoomAPI`` 호환 객체.

    현재가 ``ka10001``(``api.stock_info.basic_stock_info``), 일봉 ``ka10081``(``api.chart.stock_daily_chart``,
    수정주가 ``upd_stkpc_tp="1"``). 종목코드는 6자리 그대로 → KRX 시세(NXT·통합은 ``_NX``·``_AL`` 접미).
    """

    def __init__(self, api: Any = None, *, api_factory: Callable[[], Any] | None = None,
                 clock: Callable[[], datetime] = now_kst) -> None:
        """``api`` 를 바로 주거나 ``api_factory`` 로 **지연** 생성한다 — 첫 시세 조회 전엔 키움(토큰 발급 포함)을 부르지 않는다
        (통합 검토 M2: signal 작업·DB 재생은 키움 없이 떠야 한다)."""
        if api is None and api_factory is None:
            raise ValueError("api 또는 api_factory 가 필요하다")
        self._api_obj = api
        self._api_factory = api_factory
        self._clock = clock

    @property
    def _api(self) -> Any:
        if self._api_obj is None:
            assert self._api_factory is not None
            self._api_obj = self._api_factory()
        return self._api_obj

    @staticmethod
    def _krx_code(code: str) -> str:
        """6자리 숫자만(= KRX 시세). ``_NX``·``_AL`` 접미(NXT·통합)는 거부한다."""
        c = str(code).strip()
        if len(c) != 6 or not c.isalnum() or "_" in c:
            raise ValueError(f"KRX 6자리 종목코드만 허용: {code!r}")
        return c

    def quote(self, code: str) -> QuoteSnap:
        code = self._krx_code(code)
        raw = self._api.stock_info.basic_stock_info(stk_cd=code)
        last = _num(raw.get("cur_prc"))
        if not last:
            raise ValueError(f"ka10001 현재가 없음: {code} {raw.get('return_msg', '')}")
        return QuoteSnap(
            code=code,
            ts=self._clock(),
            last=last,
            date=_ymd(raw.get("dt") or raw.get("date") or raw.get("base_dt")),   # **미검증** 필드 — 없으면 None(증거 아님)
            open=_num(raw.get("open_pric")),
            high=_num(raw.get("high_pric")),
            low=_num(raw.get("low_pric")),
            prev_close=_num(raw.get("base_pric")),
        )

    def daily_bars(self, code: str, n: int = 60) -> list[DailyBar]:
        """``ka10081`` 일봉. **가격은 KRX 애프터마켓 체결이 섞일 수 있다** — 런타임은 날짜(휴장 판정)에만 쓴다."""
        code = self._krx_code(code)
        today = self._clock().astimezone(KST).strftime("%Y%m%d")
        raw = self._api.chart.stock_daily_chart(stk_cd=code, base_dt=today, upd_stkpc_tp="1")
        out: list[DailyBar] = []
        for row in raw.get("stk_dt_pole_chart_qry") or []:
            dt = str(row.get("dt") or "")
            close = _num(row.get("cur_prc"))
            if len(dt) != 8 or not dt.isdigit() or not close:
                continue
            out.append(DailyBar(
                date=date(int(dt[:4]), int(dt[4:6]), int(dt[6:])),
                open=_num(row.get("open_pric")) or close,
                high=_num(row.get("high_pric")) or close,
                low=_num(row.get("low_pric")) or close,
                close=close,
                volume=_num(row.get("trde_qty")) or 0.0,
            ))
            if len(out) >= n:
                break
        return sorted(out, key=lambda b: b.date)


class FakeQuoteProvider:
    """테스트·리허설용 시세. ``set_quote``·``set_bars`` 로 채운다. 호출 기록은 ``calls``."""

    def __init__(self, *, clock: Callable[[], datetime] = now_kst) -> None:
        self._clock = clock
        self.quotes: dict[str, QuoteSnap] = {}
        self.bars: dict[str, list[DailyBar]] = {}
        self.fail: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def set_quote(self, code: str, last: float, **kw: Any) -> None:
        self.quotes[code] = QuoteSnap(code=code, ts=kw.pop("ts", None) or self._clock(), last=last, **kw)

    def set_bars(self, code: str, bars: list[DailyBar]) -> None:
        self.bars[code] = sorted(bars, key=lambda b: b.date)

    def quote(self, code: str) -> QuoteSnap:
        self.calls.append(("quote", code))
        if code in self.fail or code not in self.quotes:
            raise RuntimeError(f"fake: 시세 없음 {code}")
        q = self.quotes[code]
        return replace(q, ts=self._clock())

    def daily_bars(self, code: str, n: int = 60) -> list[DailyBar]:
        self.calls.append(("daily_bars", code))
        if code in self.fail:
            raise RuntimeError(f"fake: 일봉 없음 {code}")
        return list(self.bars.get(code, []))[-n:]


#: 재생에서 작업별 시각(크론 예시와 같다). ``--quotes db`` 는 시계를 이 시각에 둔다 — 벽시계로 과거일을 재생하면 매 작업이
#: 그날 전체 봉을 본다(재검토 잔여 3).
REPLAY_JOB_TIMES = {"signal": "07:35", "preopen": "08:35", "open_check": "09:02", "monitor": "09:05",
                    "close": "15:21", "eod": "15:40", "reconcile": "15:50", "status": "16:00", "release_order": "16:00"}


class SimClock:
    """재생용 시계 — ``set(day, "HH:MM")`` 로 두고 ``sleep(초)`` 로 진행한다(monitor 루프가 실제로 자지 않는다). KST tz-aware."""

    def __init__(self, day: date, hhmm: str = "07:35") -> None:
        self.now = datetime(day.year, day.month, day.day, tzinfo=KST)
        self.set(day, hhmm)

    def set(self, day: date, hhmm: str) -> None:
        h, m = (int(x) for x in hhmm.split(":"))
        self.now = datetime(day.year, day.month, day.day, h, m, tzinfo=KST)

    def sleep(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=float(seconds))

    def __call__(self) -> datetime:
        return self.now


class DBQuoteProvider:
    """DB 재생 시세(통합 검토 M2) — 키움을 전혀 부르지 않는다. ``sw-live … --quotes db --asof <과거일>``.

    ``minute_bars``(KST naive, 09:00~15:32 정규장 봉 — :data:`swing_it.krx_session.SESSION_CLOSE`, VI 로 연장된 종가 단일가 포함)로 ``asof`` 하루를 재생한다: 시계 시각 t 까지의 봉으로 시가(첫 봉)·고가·저가·
    현재가(t 이하 마지막 봉 종가)를 만들고, ``date=asof``(F4 신선도 증거)를 단다. 09:00 전이거나 봉이 없으면 현재가 = 직전 거래일
    종가, 시가 없음(실제 장전 시세와 같은 모양). 기준 종목 일봉은 ``daily_bars``(날짜만 쓴다). Postgres·sqlite 둘 다 된다.
    """

    def __init__(self, con: Any, asof: date, *, clock: Callable[[], datetime] = now_kst) -> None:
        self._con = con
        self._asof = asof
        self._clock = clock
        self._cache: dict[str, list[tuple[datetime, float, float, float, float]]] = {}
        self._prev_close: dict[str, float | None] = {}
        self._pg = type(con).__module__.split(".")[0] != "sqlite3"

    def _rows(self, sql: str, params: tuple) -> list[tuple]:
        cur = self._con.cursor()
        cur.execute(sql.replace("?", "%s") if self._pg else sql, params)
        return list(cur.fetchall())

    def _minutes(self, code: str) -> list[tuple[datetime, float, float, float, float]]:
        if code not in self._cache:
            from swing_it.krx_session import SESSION_CLOSE, SESSION_OPEN  # noqa: PLC0415 — 정규장 구간의 정본

            lo = f"{self._asof.isoformat()} {SESSION_OPEN}"
            hi = f"{self._asof.isoformat()} {SESSION_CLOSE}"     # 15:30 이면 VI 연장 종가(15:32)를 놓친다(REVIEW_META_GEN2 N1)
            rows = self._rows("SELECT ts, open, high, low, close FROM minute_bars WHERE code = ? AND ts >= ? AND ts <= ? "
                              "ORDER BY ts", (code, lo, hi))
            out = []
            for ts, o, h, lo_, c in rows:
                t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                out.append((t.replace(tzinfo=None), float(o), float(h), float(lo_), float(c)))
            self._cache[code] = out
        return self._cache[code]

    def _prev(self, code: str) -> float | None:
        if code not in self._prev_close:
            from swing_it.storage import last_close_before  # noqa: PLC0415 — 가격 테이블은 storage 정문으로만

            self._prev_close[code] = last_close_before(self._con, code, self._asof.isoformat())
        return self._prev_close[code]

    def quote(self, code: str) -> QuoteSnap:
        now = self._clock()
        t = (now.astimezone(KST) if now.tzinfo else now).replace(tzinfo=None)
        bars = [b for b in self._minutes(code) if b[0] <= t]
        prev = self._prev(code)
        if not bars:
            if prev is None:
                raise ValueError(f"db: {code} {self._asof} 시세 없음")
            return QuoteSnap(code=code, ts=now, last=prev, prev_close=prev)
        return QuoteSnap(code=code, ts=now, last=bars[-1][4], open=bars[0][1], high=max(b[2] for b in bars),
                         low=min(b[3] for b in bars), prev_close=prev, date=self._asof)

    def daily_bars(self, code: str, n: int = 60) -> list[DailyBar]:
        from swing_it.storage import reference_daily_bars  # noqa: PLC0415 — 가격 테이블은 storage 정문으로만

        out = []
        for dd, o, h, lo_, c in reference_daily_bars(self._con, code, end=self._asof.isoformat(), limit=n):
            if c is None:
                continue
            out.append(DailyBar(date=dd, open=float(o or c), high=float(h or c), low=float(lo_ or c), close=float(c)))
        return sorted(out, key=lambda b: b.date)


# --------------------------------------------------------------------------- 페이퍼 브로커


class PersistentPaperBroker(PaperBroker):
    """파일로 저장·복원되는 코어 :class:`PaperBroker`.

    체결 판정은 코어 그대로다(상속, 판정 메서드 안 건드림). 저장하는 것은 보유·대기 주문(도착시각 포함)·다음
    주문번호·아직 안 빼간 체결 버퍼. 코어 내부 필드(``_holdings``·``_orders``·``_next_ord_no``·``_fills_buffer``)를
    직접 읽고 쓴다 — ``krx-quant-core==0.6.2`` 고정이라 안전하고, 필드가 바뀌면 ``tests/test_live_brokers.py`` 가 깨진다.
    """

    def __init__(self, path: Path | str | None = None, **kw: Any) -> None:
        super().__init__(fill_basis=kw.pop("fill_basis", "through"), latency_sec=0.0, **kw)
        self.path = Path(path) if path is not None else None

    @classmethod
    def load(cls, path: Path | str) -> PersistentPaperBroker:
        broker = cls(path)
        p = Path(path)
        if not p.exists():
            return broker
        raw = json.loads(p.read_text(encoding="utf-8"))
        broker._holdings = {
            c: Holding(code=c, qty=int(h["qty"]), avg_price=float(h["avg_price"]))
            for c, h in (raw.get("holdings") or {}).items()
        }
        broker._orders = {}
        for o in raw.get("orders") or []:
            intent = OrderIntent(side=o["side"], code=o["code"], qty=int(o["qty"]), price=int(o["price"]))
            arrival = datetime.fromisoformat(o["arrival_ts"]) if o.get("arrival_ts") else None
            broker._orders[o["ord_no"]] = _Pending(intent=intent, ord_no=o["ord_no"], arrival_ts=arrival)
        broker._next_ord_no = int(raw.get("next_ord_no", 1))
        broker._fills_buffer = [
            Fill(ord_no=f["ord_no"], code=f["code"], side=f["side"], qty=int(f["qty"]),
                 price=int(f["price"]), ts=datetime.fromisoformat(f["ts"]))
            for f in raw.get("fills_buffer") or []
        ]
        return broker

    def save(self) -> None:
        if self.path is None:
            return
        atomic_write_json(self.path, {
            "holdings": {c: {"qty": h.qty, "avg_price": h.avg_price} for c, h in self._holdings.items()},
            "orders": [
                {"ord_no": p.ord_no, "side": p.intent.side, "code": p.intent.code, "qty": p.intent.qty,
                 "price": p.intent.price,
                 "arrival_ts": p.arrival_ts.isoformat() if p.arrival_ts else None}
                for p in self._orders.values()
            ],
            "next_ord_no": self._next_ord_no,
            "fills_buffer": [
                {"ord_no": f.ord_no, "code": f.code, "side": f.side, "qty": f.qty, "price": f.price,
                 "ts": f.ts.isoformat()}
                for f in self._fills_buffer
            ],
        })


def feed_auction(broker: PaperBroker, code: str, ts: datetime, price: float) -> list[Fill]:
    """동시호가 단일가 체결 에뮬 — ``bid = ask = 단일가`` 시세 한 장. 매수는 ``가격 ≤ 지정가``, 매도는 ``가격 ≥ 지정가`` 면
    그 단일가로 체결된다(코어 테이커 판정)."""
    return broker.on_quote(code, ts, bid=float(price), ask=float(price), last=None)


def feed_quote(broker: PaperBroker, snap: QuoteSnap) -> list[Fill]:
    """장중 시세 한 장. 호가가 없으면 ``bid = ask = 현재가``."""
    bid = snap.bid if snap.bid and snap.bid > 0 else snap.last
    ask = snap.ask if snap.ask and snap.ask > 0 else snap.last
    return broker.on_quote(snap.code, snap.ts, bid=float(bid), ask=float(ask), last=None)


# --------------------------------------------------------------------------- 가드


def entry_guard_config(
    *,
    equity: float,
    price: float,
    hard_cap_frac: float,
    price_band_pct: float,
    max_orders_total: int,
    max_orders_per_code: int = 3,
) -> OrderGuardConfig:
    """매수 한 건의 가드 설정. 수량 상한 = ``floor(자산 × hard_cap_frac / 지정가)``(사이징 10% 위 여유).

    코어 기본값(최대 1주·하드캡 10·밴드 3%)은 1주 단타용이라 그대로 쓰면 모든 진입이 막힌다. 그렇다고 하드캡을
    큰 상수로 풀면 가드가 무의미해진다 — 그래서 **주문마다** 자산·가격으로 상한을 다시 잡는다.
    """
    cap = max(1, int(math.floor(equity * hard_cap_frac / max(float(price), 1.0))))
    return OrderGuardConfig(
        max_qty=cap,
        max_qty_hard_cap=cap,
        price_band_pct=price_band_pct,
        max_orders_per_code=max_orders_per_code,
        max_orders_total=max_orders_total,
    )


# --------------------------------------------------------------------------- 배선


@dataclass
class BrokerBundle:
    """한 작업이 쓰는 브로커·시세 묶음."""

    mode: str
    broker: Any                       # Broker 프로토콜(PersistentPaperBroker | KiwoomBroker | 테스트 가짜)
    quotes: QuoteProvider
    paper: PersistentPaperBroker | None = None
    #: 체결을 잔고 차이로 판정하나(키움: ka10076 미검증이라 poll_fills 가 꺼져 있다).
    fills_from_holdings: bool = False

    @property
    def dry_run(self) -> bool:
        return bool(getattr(self.broker, "dry_run", False))

    def persist(self) -> None:
        if self.paper is not None:
            self.paper.save()


def make_kiwoom_api() -> Any:
    """``kiwoom_client.KiwoomAPI`` 를 **지연** import 로 만든다(페이퍼 테스트·CI 는 키움 패키지 없이 돈다).

    환경변수: ``KIWOOM_APP_KEY``·``KIWOOM_APP_SECRET``(필수), ``SWING_KIWOOM_IS_MOCK``(``true`` 면 모의서버, 기본 실서버 —
    페이퍼도 실시세를 읽어야 해서). 토큰 폐기(logout/revoke)는 **절대 부르지 않는다**(다른 데몬과 앱키 공유).
    """
    from kiwoom_client import KiwoomAPI  # noqa: PLC0415 — 지연 import(위 docstring)

    key = os.environ.get("KIWOOM_APP_KEY", "")
    secret = os.environ.get("KIWOOM_APP_SECRET", "")
    if not key or not secret:
        raise RuntimeError("KIWOOM_APP_KEY / KIWOOM_APP_SECRET 이 없다")
    is_mock = os.environ.get("SWING_KIWOOM_IS_MOCK", "false").strip().lower() in ("1", "true", "yes")
    return KiwoomAPI(app_key=key, app_secret=secret, is_mock=is_mock)


def build_bundle(
    mode: str,
    *,
    confirm_real: bool,
    paper_path: Path | str,
    api: Any = None,
    quotes: QuoteProvider | None = None,
    broker: Any = None,
    clock: Callable[[], datetime] = now_kst,
) -> BrokerBundle:
    """모드에 맞는 브로커·시세를 만든다.

    안전 불변식: ``mode == "live"`` 는 ``confirm_real=True`` 일 때만, ``confirm_real=True`` 는 ``mode == "live"`` 일 때만.
    어긋나면 ``ValueError`` — CLI 검증을 우회한 호출도 여기서 막힌다.

    ``broker``·``quotes`` 를 주면 그것을 쓴다(테스트). 페이퍼는 ``broker`` 를 주지 않으면 ``paper_path`` 에서 복원한다.
    """
    if mode not in MODES:
        raise ValueError(f"알 수 없는 모드: {mode!r} (가능: {MODES})")
    if (mode == "live") != bool(confirm_real):
        raise ValueError("실주문(live)은 confirm_real=True 와 정확히 짝이어야 한다 — --execute --confirm-real")

    if quotes is None:
        # api 를 안 줬으면 지연 생성 — 시세를 실제로 읽는 작업에서만 키움이 뜬다(토큰 발급 포함).
        quotes = RestQuoteProvider(api, api_factory=None if api is not None else make_kiwoom_api, clock=clock)

    if mode == "paper":
        paper = broker if broker is not None else PersistentPaperBroker.load(paper_path)
        if not isinstance(paper, PaperBroker):
            raise TypeError("paper 모드 브로커는 PaperBroker 여야 한다(실브로커를 페이퍼로 쓰는 사고 방지)")
        return BrokerBundle(mode=mode, broker=paper, quotes=quotes,
                            paper=paper if isinstance(paper, PersistentPaperBroker) else None,
                            fills_from_holdings=False)

    if broker is None:
        if api is None:
            api = make_kiwoom_api()
        # fills_verified=False: ka10076 미검증 — poll_fills 는 [] 이고 체결은 잔고 차이로 판정한다.
        broker = KiwoomBroker(api, dry_run=(mode != "live"), fills_verified=False)
    if mode == "dry-run" and not getattr(broker, "dry_run", False):
        raise ValueError("dry-run 모드인데 브로커가 dry_run 이 아니다")
    if isinstance(broker, PaperBroker):
        raise TypeError("dry-run/live 모드에 PaperBroker 를 줄 수 없다")
    return BrokerBundle(mode=mode, broker=broker, quotes=quotes, fills_from_holdings=True)


def auction_ts(day: date, hhmm: str) -> datetime:
    """``day`` 의 KST 시각(동시호가 체결 시각 표기용)."""
    return datetime(day.year, day.month, day.day, int(hhmm[:2]), int(hhmm[2:]), tzinfo=KST)


def stale(ts: datetime, now: datetime, *, max_age: timedelta) -> bool:
    return now - ts > max_age
