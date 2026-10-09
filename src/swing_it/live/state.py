"""스윙 라이브 상태 — 전략 원장(스냅샷 + 사건 JSONL)·포지션 장부 저널·기준선·다일 낙폭 킬.

크론이 작업마다 **새 프로세스**로 띄우므로(signal·preopen·open_check·monitor·close·eod), 프로세스 사이에 넘어가야
하는 것은 전부 여기 파일에 있다. 메모리에만 있는 것은 없다고 가정한다.

파일(모드별 디렉터리 ``<data_dir>/<mode>/`` — 페이퍼·드라이런·실계좌가 서로의 상태를 절대 섞지 않는다):

* ``state.json`` — 전략 원장 **스냅샷**. 원자적 교체(tmp → fsync → rename)로만 쓴다. 직전 판은 ``state.json.bak``.
  재시작 복원은 이 파일 하나를 읽는다. 깨져 있으면 :class:`StateCorrupt` — 호출부는 신규 진입을 막고 사람을 부른다
  (추측해서 복구하지 않는다).
* ``ledger.jsonl`` — 사건 원장(append + fsync). 주문·체결·포지션 전이·킬·기준선이 한 줄씩. **감사 추적**이고 복원에는
  쓰지 않는다(스냅샷과 원장이 갈라지면 원장이 사람이 읽는 증거다).
* ``book.jsonl`` — 코어 :class:`~krx_quant_core.execution.book.PositionBook` 저널(체결 재생으로 평단·실현손익 복원).
* ``paper_broker.json`` — 페이퍼 모드의 모의 브로커 상태(:mod:`.brokers`).

포지션 수량 규약(공유 계좌): 계좌 보유 = **사용자 몫(기준선)** + 이 전략 몫. 아침 기준선은
``사용자 몫 = 계좌 보유 − 전략 장부 보유`` 로 잡고 그날 내내 고정한다. 이 전략은 사용자 몫이 있는 종목을 **사지 않고**,
팔 때는 **전략 몫까지만** 판다(daytrade-it AutoTrader C-2 규약과 같다).

다일 낙폭 킬: 장 마감 자산(전략 자본 기준, 계좌 전체가 아니다)이 고점 대비 ``−drawdown_kill_pct`` 이하면 정지 파일을
만들고 신규 진입을 막는다. 청산은 막지 않는다. 사람이 정지 파일을 지우면 그것을 **확인(ack)** 으로 보고 고점을 현재
자산으로 다시 잡는다 — 지운 직후 다음 장 마감에 같은 낙폭으로 즉시 다시 걸리지 않게.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.events import Fill
from krx_quant_core.market.codes import Market

__all__ = [
    "LiveState",
    "OrderRec",
    "PositionRec",
    "StateCorrupt",
    "StateStore",
    "drawdown_check",
]

SCHEMA_VERSION = 1

#: 포지션 상태. planned(신호만) → entry_pending(주문 나감) → open(체결) → closed.
#: 진입 안 된 것은 missed(시가 > 지정가)·skipped(킬·기준선·현금 등)·expired(휴장·주문 실패)로 끝난다.
POSITION_STATUSES = ("planned", "entry_pending", "open", "closed", "missed", "skipped", "expired")
#: 열린(관리 대상) 상태.
ACTIVE_STATUSES = ("planned", "entry_pending", "open")
#: 주문이 아직 살아 있는(체결 대기·예약 유지) 상태. ``intent`` = POST 전에 저장된 의도(검토 C1), ``fill_pending`` = 취소가
#: 거부돼 체결 확인을 기다리는 중(검토 C2) — 둘 다 같은 수량을 다시 내지 않도록 예약으로 센다.
LIVE_ORDER_STATUSES = ("submitted", "unknown", "partial", "intent", "fill_pending")


class StateCorrupt(RuntimeError):
    """스냅샷을 읽을 수 없다 — 신규 진입 금지, 사람 확인 필요."""


@dataclass
class OrderRec:
    """이 전략이 낸(또는 낼 뻔한) 주문 하나. ``oid`` 는 로컬 키(브로커 주문번호가 없을 수도 있어서)."""

    oid: str
    trade_date: str
    ts: str
    code: str
    side: str                       # buy | sell
    qty: int
    price: int
    purpose: str                    # entry | stop | partial | trail_next_open | time_close | max_hold | skip_below_low | exit
    session: str                    # open_auction | continuous | close_auction
    status: str                     # intent | submitted | unknown | partial | fill_pending | filled | canceled | rejected | blocked | dry_run | expired | needs_human | released
    ord_no: str | None = None
    filled_qty: int = 0
    attempt: int = 0
    ref_price: float | None = None  # 제출 시점 참조가(매수 가드 기준가·매도 호가) — 체결가 추정에 쓴다
    reason: str = ""

    @property
    def remaining(self) -> int:
        return max(0, self.qty - self.filled_qty)

    @property
    def live(self) -> bool:
        return self.status in LIVE_ORDER_STATUSES


@dataclass
class PositionRec:
    """전략 포지션 하나(신호 → 진입 → 청산 전 생애). 전략 상태기계는 ``strat`` 에 직렬화돼 있다."""

    code: str
    market: str
    signal_date: str
    trade_date: str                 # 진입 예정(=신호 다음 거래일)
    status: str = "planned"
    name: str = ""
    plan: dict[str, Any] = field(default_factory=dict)   # 피벗·수축저점·d_est·지정가·계획 수량 등
    entry_date: str | None = None
    entry_price: float | None = None
    qty: int = 0                    # 지금 들고 있는 전략 몫
    entered_qty: int = 0            # 진입 체결 누계
    last_day: str | None = None     # 전략 day_no 를 마지막으로 올린 거래일(하루 한 번만 올린다)
    last_eod: str | None = None
    #: MA50 추적용 KRX 정규장 종가(오래된 것 → 최신, 최대 trail_ma − 1 개). signal 작업이 DB(NXT 보정본)에서
    #: 새로 채우고 eod 가 그날 KRX 종가를 덧붙인다. 키움 일봉(ka10081)은 2026-09-14 부터 NXT 가 섞여 쓰지 않는다.
    trail_closes: list[float] = field(default_factory=list)
    strat: dict[str, Any] | None = None   # 직렬화된 PositionState
    stop: float | None = None       # 관측용 사본(전략 상태가 정본)
    target: float | None = None
    partial_done: bool = False
    breakeven: bool = False
    exit_intent: dict[str, Any] | None = None  # {"kind","qty","session","created","reason","halted"}
    exit_attempt: int = 0
    exit_rejects: int = 0           # 연속 거부·차단 매도 수(검토 M1 회로 차단기)
    #: 계획 시점의 StrategyConfig(dict) 와 그 해시 — 이 포지션의 청산 규칙은 **여기 고정**된다(통합 검토 M3: 롤링 재적합이
    #: 열린 포지션의 규칙을 소급해 바꾸지 않는다). None 이면(옛 스냅샷) 현재 설정.
    cfg: dict[str, Any] | None = None
    cfg_hash: str | None = None
    realized: float = 0.0
    closed_date: str | None = None
    exit_reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES


@dataclass
class LiveState:
    """전략 원장 스냅샷 전체."""

    mode: str
    capital: float
    schema: int = SCHEMA_VERSION
    seq: int = 0
    positions: dict[str, PositionRec] = field(default_factory=dict)   # 코드 → 관리 중 포지션
    history: list[PositionRec] = field(default_factory=list)          # 끝난 포지션
    orders: dict[str, OrderRec] = field(default_factory=dict)         # oid → 주문
    own_ord_nos: list[str] = field(default_factory=list)              # 재시작 뒤 OMS.own_orders 복원용
    baseline: dict[str, Any] = field(default_factory=dict)            # {"date","user":{code:qty},"account":{...}}
    equity_peak: float = 0.0
    equity_history: list[dict[str, Any]] = field(default_factory=list)  # [{"date","equity","cash","mv","dd"}]
    drawdown_kill: dict[str, Any] = field(default_factory=dict)       # {"active","date","equity","peak","dd"}
    jobs_done: dict[str, str] = field(default_factory=dict)           # 작업 → 마지막 완료 거래일
    last_signal: dict[str, Any] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)            # 거래일 → "open"|"closed"|"suspect"(휴장 판정 결과)
    #: 뉴스 거부권(설계 §7)에 걸린 후보의 **가상** 포지션 — 주문 없음. 실포지션과 같은 상태기계로 추적해 "거부군 vs 진입군" 을 비교한다.
    shadow: dict[str, PositionRec] = field(default_factory=dict)
    shadow_history: list[PositionRec] = field(default_factory=list)
    blocked_codes: dict[str, str] = field(default_factory=dict)       # 코드 → 사유. 미해결 주문이 있는 종목(신규 진입 금지, 사람이 해제)
    config_hash: str | None = None                                    # 마지막 작업이 쓴 전략 설정 해시(신규 계획용)
    market_of: dict[str, str] = field(default_factory=dict)           # 코드 → KOSPI/KOSDAQ(세금 계산용, 닫힌 것도 유지)

    # ----- 키·조회 -------------------------------------------------------------

    def next_oid(self) -> str:
        self.seq += 1
        return f"L{self.seq:06d}"

    def active_positions(self) -> list[PositionRec]:
        return [p for p in self.positions.values() if p.active]

    def open_positions(self) -> list[PositionRec]:
        return [p for p in self.positions.values() if p.status == "open"]

    def live_orders(self, code: str | None = None, side: str | None = None) -> list[OrderRec]:
        return [
            o
            for o in self.orders.values()
            if o.live and (code is None or o.code == code) and (side is None or o.side == side)
        ]

    def baseline_user(self) -> dict[str, int]:
        return {k: int(v) for k, v in (self.baseline.get("user") or {}).items()}

    def close_position(self, code: str, *, date: str, reason: str) -> PositionRec:
        """``positions`` 에서 빼 ``history`` 로 옮긴다."""
        pos = self.positions.pop(code)
        if pos.status == "open" or pos.entered_qty > 0:
            pos.status = "closed"
        pos.closed_date = date
        pos.exit_reason = pos.exit_reason or reason
        pos.exit_intent = None
        self.history.append(pos)
        return pos

    # ----- 직렬화 --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LiveState:
        data = dict(raw)
        schema = int(data.get("schema", 0))
        if schema != SCHEMA_VERSION:
            raise StateCorrupt(f"알 수 없는 스냅샷 스키마: {schema} (기대 {SCHEMA_VERSION})")
        data["positions"] = {k: _make(PositionRec, v) for k, v in (data.get("positions") or {}).items()}
        data["history"] = [_make(PositionRec, v) for v in (data.get("history") or [])]
        data["orders"] = {k: _make(OrderRec, v) for k, v in (data.get("orders") or {}).items()}
        data["shadow"] = {k: _make(PositionRec, v) for k, v in (data.get("shadow") or {}).items()}
        data["shadow_history"] = [_make(PositionRec, v) for v in (data.get("shadow_history") or [])]
        return _make(cls, data)


def _make(cls: type, raw: dict[str, Any]) -> Any:
    """알 수 없는 키는 버리고 dataclass 를 만든다(필드 추가에 대한 전방 호환)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def _fsync_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def atomic_write_json(path: Path, obj: Any) -> None:
    """tmp → fsync → rename. 중간에 죽어도 이전 판이나 새 판 둘 중 하나만 남는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:  # 디렉터리 엔트리까지 내린다(전원 차단 대비). 실패해도 데이터는 이미 tmp 에서 옮겨졌다.
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


class StateStore:
    """모드 디렉터리 하나의 파일들(스냅샷·사건 원장·장부 저널)을 다룬다."""

    def __init__(self, root: Path | str, *, clock: Callable[[], datetime]) -> None:
        self.root = Path(root)
        self._clock = clock
        #: 모든 사건 행에 붙는 공통 필드(예: ``cfg`` = 전략 설정 해시 8자리).
        self.default_fields: dict[str, Any] = {}

    # ----- 경로 ----------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    @property
    def book_path(self) -> Path:
        return self.root / "book.jsonl"

    @property
    def order_log_path(self) -> Path:
        return self.root / "orders.jsonl"

    @property
    def paper_path(self) -> Path:
        return self.root / "paper_broker.json"

    @property
    def shadow_path(self) -> Path:
        """거부권 섀도 원장(설계 §7) — 가상 진입·청산 사건. 실원장과 섞지 않는다."""
        return self.root / "shadow.jsonl"

    @property
    def shadow_book_path(self) -> Path:
        return self.root / "shadow_book.jsonl"

    @property
    def lock_path(self) -> Path:
        return self.root / "swing_live.lock"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    # ----- 스냅샷 --------------------------------------------------------------

    def load(self, *, mode: str, capital: float) -> LiveState:
        """스냅샷 복원. 없으면 새 상태(첫 실행). 깨졌으면 :class:`StateCorrupt`.

        모드가 다르면 거부한다 — 페이퍼 상태로 실계좌를 굴리는 사고를 막는다(디렉터리도 모드별이지만 이중 확인).
        ``capital`` 은 **첫 실행에만** 쓴다. 이후엔 스냅샷 값이 정본이다(실행마다 다른 값을 줘서 자산이 튀지 않게).
        """
        if not self.state_path.exists():
            if self.ledger_path.exists() and self.ledger_path.stat().st_size > 0:
                raise StateCorrupt(
                    f"{self.state_path} 가 없는데 사건 원장({self.ledger_path})은 있다 — 스냅샷 유실. "
                    "state.json.bak 을 확인해 사람이 복구할 것"
                )
            return LiveState(mode=mode, capital=float(capital), equity_peak=float(capital))
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            state = LiveState.from_dict(raw)
        except StateCorrupt:
            raise
        except Exception as e:  # noqa: BLE001 — 어떤 파싱 실패든 같은 결론(사람 확인)
            raise StateCorrupt(f"스냅샷을 읽을 수 없다: {self.state_path}: {type(e).__name__}: {e}") from e
        if state.mode != mode:
            raise StateCorrupt(f"스냅샷 모드 {state.mode!r} ≠ 실행 모드 {mode!r} — 디렉터리를 확인할 것")
        return state

    def save(self, state: LiveState) -> None:
        if self.state_path.exists():
            try:
                atomic_write_json(self.state_path.with_name("state.json.bak"),
                                  json.loads(self.state_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 — 백업 실패가 저장을 막으면 안 된다
                pass
        atomic_write_json(self.state_path, state.to_dict())

    # ----- 사건 원장 -----------------------------------------------------------

    def event(self, event: str, **fields_: Any) -> dict[str, Any]:
        row = {"ts": self._clock().isoformat(timespec="seconds"), "event": event, **self.default_fields, **fields_}
        _fsync_append(self.ledger_path, row)
        return row

    def shadow_event(self, event: str, **fields_: Any) -> dict[str, Any]:
        row = {"ts": self._clock().isoformat(timespec="seconds"), "event": event, **self.default_fields, **fields_}
        _fsync_append(self.shadow_path, row)
        return row

    def shadow_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.shadow_path)

    def load_shadow_book(self, state: LiveState) -> PositionBook:
        """섀도 장부 — 실장부와 같은 비용 모델(수수료·세금)로 가상 체결의 순손익을 낸다."""
        return PositionBook.load(self.shadow_book_path, market_of=market_lookup(state))

    def events(self) -> list[dict[str, Any]]:
        """사건 원장 전체. 마지막 줄이 쓰다 만 줄이면(크래시) 건너뛴다."""
        return _read_jsonl(self.ledger_path)

    # ----- 포지션 장부 ---------------------------------------------------------

    def load_book(self, state: LiveState) -> PositionBook:
        """코어 장부를 저널 재생으로 복원. 세금 계산용 시장은 원장의 ``market_of`` 에서."""
        return PositionBook.load(self.book_path, market_of=market_lookup(state))


def market_lookup(state: LiveState) -> Callable[[str], str]:
    """``PositionBook`` 의 ``market_of``. 모르는 코드는 KOSDAQ(보수적 기본값 — 사건 원장에 경고를 남길 것)."""

    def _of(code: str) -> str:
        raw = state.market_of.get(code)
        if not raw:
            return Market.KOSDAQ.value
        try:
            return Market.parse(raw).value
        except ValueError:
            return Market.KOSDAQ.value

    return _of


def strategy_holdings(book: PositionBook) -> dict[str, int]:
    return {c: h.qty for c, h in book.positions().items()}


def compute_baseline_user(account: dict[str, int], own: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """``(사용자 몫, 부족분)``. 부족분 = 전략 장부가 계좌보다 많이 든 종목(대사 이상 — 사람 확인)."""
    user: dict[str, int] = {}
    short: dict[str, int] = {}
    for code, qty in account.items():
        rest = int(qty) - int(own.get(code, 0))
        if rest > 0:
            user[code] = rest
    for code, qty in own.items():
        have = int(account.get(code, 0))
        if have < qty:
            short[code] = qty - have
    return user, short


def drawdown_check(
    state: LiveState,
    *,
    date: str,
    equity: float,
    cash: float,
    market_value: float,
    kill_pct: float,
    stop_file: Path,
    on_trigger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """장 마감 자산을 기록하고 다일 낙폭 킬을 판정한다. 기록 행을 돌려준다.

    * 같은 날 두 번 부르면 그날 행을 덮어쓴다(멱등).
    * 킬이 이미 걸려 있는데 정지 파일이 없으면 = 사람이 지웠다(확인) → 고점을 지금 자산으로 다시 잡고 킬 해제.
    * 고점 대비 ``−kill_pct`` 이하면 정지 파일 생성 + ``drawdown_kill.active``.
    """
    active = bool(state.drawdown_kill.get("active"))
    if active and not stop_file.exists():
        state.drawdown_kill = {**state.drawdown_kill, "active": False, "acked": date}
        state.equity_peak = float(equity)
    state.equity_peak = max(float(state.equity_peak or 0.0), float(equity))
    dd = equity / state.equity_peak - 1.0 if state.equity_peak > 0 else 0.0
    row = {"date": date, "equity": round(equity, 2), "cash": round(cash, 2),
           "mv": round(market_value, 2), "peak": round(state.equity_peak, 2), "dd": round(dd, 6)}
    state.equity_history = [r for r in state.equity_history if r.get("date") != date] + [row]
    if not state.drawdown_kill.get("active") and dd <= -abs(kill_pct) + 1e-12:
        state.drawdown_kill = {"active": True, "date": date, "equity": equity,
                               "peak": state.equity_peak, "dd": dd}
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text(
            f"swing-live 다일 낙폭 킬 {date}: 자산 {equity:,.0f} / 고점 {state.equity_peak:,.0f} = {dd:.2%}\n"
            "신규 진입 정지(청산은 계속). 원인 확인 후 이 파일을 지우면 다음 장 마감에 고점을 현재 자산으로 다시 잡는다.\n",
            encoding="utf-8",
        )
        if on_trigger is not None:
            on_trigger(f"다일 낙폭 킬: {dd:.2%} (자산 {equity:,.0f}, 고점 {state.equity_peak:,.0f})")
    return row


def strategy_hash(cfg_dict: dict[str, Any]) -> str:
    """전략 설정 dict 의 정규화 sha256(전체). 원장엔 앞 8자리를 쓴다. inf/NaN 은 문자열로 직렬화해 안정적이다."""
    canon = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False
                       ) if _finite_only(cfg_dict) else json.dumps(_stringify_nonfinite(cfg_dict), sort_keys=True,
                                                                   ensure_ascii=False, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _finite_only(d: dict[str, Any]) -> bool:
    import math  # noqa: PLC0415

    return all(not (isinstance(v, float) and not math.isfinite(v)) for v in d.values())


def _stringify_nonfinite(d: dict[str, Any]) -> dict[str, Any]:
    import math  # noqa: PLC0415

    return {k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def trade_stats(records: list[PositionRec]) -> dict[str, Any]:
    """진입이 있었던 포지션들의 ``n``·평균 순수익률·손절 비율(``NEWS_VETO.md`` 합격 바: 거부군 vs 진입군).

    순수익률 = 장부 실현 순손익(수수료·세금 차감) / (진입 수량 × 진입가). 열린 포지션은 실현분만(미실현 제외) — 비교는 닫힌 것끼리.
    손절 = exit_reason ∈ {stop, below_low}.
    """
    done = [p for p in records if p.entered_qty > 0 and p.entry_price and p.status == "closed"]
    if not done:
        return {"n": 0, "mean_net": float("nan"), "stop_rate": float("nan")}
    rets = [p.realized / (p.entered_qty * p.entry_price) for p in done]
    stops = sum(1 for p in done if p.exit_reason in ("stop", "below_low"))
    return {"n": len(done), "mean_net": sum(rets) / len(rets), "stop_rate": stops / len(done)}


def fill_record(fill: Fill, **extra: Any) -> dict[str, Any]:
    return {"ord_no": fill.ord_no, "code": fill.code, "side": fill.side, "qty": fill.qty,
            "price": fill.price, "fill_ts": fill.ts.isoformat(), **extra}
