"""스윙 라이브 전략 v0 — 순수 로직(진입 판정·주문 계획·포지션 상태 전이). 설계 §2·§3.

백테스트(:mod:`swing_it.live.backtest`)와 데몬(``swing_it.live.runtime``)이 **둘 다 이 모듈만** 부른다. DB·브로커·
시계에 손대지 않는다 — 입력은 패널·가격·숫자, 출력은 후보·계획·:class:`Action`.

정본(연구)과의 관계: 진입·청산 규칙은 ``research/signals/minervini_exits.simulate_minervini`` 기본값(원전 프로토콜)
과 ``research/signals/sepa.py`` 체결 규약을 그대로 옮긴다. 둘의 일치는 ``tests/test_live_parity.py`` 가 못박는다
(자본 무제한·슬롯 무제한 백테스트 트레이드 = 연구 트레이드 — 진입일·청산일·사유·수익 1e-9).

**진입(신호일 d, d 종가까지 — :func:`scan`):** 위생 유니버스(ADV20 ≥ 50억) ∧ 추세 템플릿 ∧ RS ≥ 70 ∧ 피벗 돌파
(거래량 1.5×) ∧ 그 시장 breadth > 0.5 ∧ PEAD 순위 ≥ 0.70 ∧ rel ≤ 1.0 ∧ F = 1 ∧ 수축 저점 정의됨.

**주문 계획(:func:`plan_entry`):** 장전 동시호가 지정가 매수. "시가 ≤ 지정가일 때만 시가 체결" = 연구의 추격 금지.

- 라이브(호가 격자, ``tick_grid=True``): 지정가 = 호가 **내림**(min(피벗 × 1.05, 상한가 − 1틱)). 내림인 이유: 실제
  시가는 호가 위에 있으므로 "시가 ≤ 내림(피벗×1.05)" ⟺ "시가 ≤ 피벗×1.05" — 연구의 추격 금지와 같은 집합이다.
  상한가 − 1틱 상한은 상한가 시가(연구 ``skip_limit_up``)를 체결에서 뺀다.
- 백테스트(분할조정 패널, ``tick_grid=False``): 과거 조정가는 호가 격자 위에 있지 않아 틱 반올림·정확 상한가가 틀린
  값을 낸다(:data:`research.signals.sepa.LIMIT_UP_RATIO` 설명과 같은 이유). 그래서 지정가 = 피벗 × 1.05 (반올림
  없음), 상한가 = 전일종가 × ``cfg.limit_up_ratio`` (1.30 − 0.007) 비율 판정 — 체결 조건 "시가 ≤ 지정가 ∧ 시가 <
  상한가 비율" 이 연구의 ``skip_chase``·``skip_limit_up`` 과 **같은 식**이다.

**포지션 상태(:class:`PositionState`) — 연구 docstring 6~11 의 스트리밍판:**

- 체결(:meth:`PositionState.on_fill_open`): d = clip(1 − 수축저점×0.99 / 진입가, 3%, 10%), 손절가 = 진입가×(1−d),
  목표가 = 진입가×(1+2d). **진입가 ≤ 수축 저점이면 그날 바로 전량 청산**(``below_low``) — 연구는 이 사건을 아예
  건너뛴다(``skip_below_low``). 라이브는 지정가가 이미 체결된 뒤라 건너뛸 수 없으므로 "체결 직후 청산" 이 가장
  가까운 행동이다. 백테스트는 그 청산을 **시가(=진입가)** 로 기록한다(총수익 0, 비용만 낸다). 패리티 테스트는 이
  사건을 비교에서 빼고 개수를 센다.
- 일봉(:meth:`PositionState.on_bar` — 백테스트): 전량 단계에서 저가 ≤ 손절 → 손절(갭이면 시가), 고가 ≥ 목표 →
  절반 익절(갭 상승이면 시가), **같은 날 둘 다면 손절 먼저**(갭 상승 시가 익절일도 저가가 손절에 닿았으면 손절).
  본전 손절은 익절 **다음 거래일부터**(예외: 갭 상승 익절일엔 그날 저가 ≤ 진입가면 그날 진입가). 나머지 절반은
  종가 < MA50 → 다음 거래일 시가(``trail_next_open``) — 1 ~ max_hold−1 일째 종가에서만. 시간 손절: 20일째(정지면 다음
  거래일) 종가에 익절·손절 전 ∧ 종가 ≤ 진입가 → 그 종가(한 번만 본다). 120일째(정지면 다음 거래일) 종가에 전부.
  정지일(``tradable=False``)은 아무것도 안 하고 날짜만 센다(연구와 같이 정지일도 보유일로 센다).
- 장중·장전·장후(:meth:`on_open` · :meth:`on_quote` · :meth:`on_close` — 라이브): 같은 상태·같은 가격 수준을 실제
  시간 순서로 본다. 익절일의 나머지 절반에는 **원래 손절가가 그날 내내** 걸려 있고(본전은 다음 거래일부터) — 같은 날 익절
  뒤 손절가를 찍으면 나머지 절반을 손절가에 판다. 일봉 모형과의 **알려진 차이**: 일봉은 그날 전량을 손절가에 판 것으로
  보고(비관), 라이브는 절반은 목표가(또는 시가)에 이미 팔았다 — 청산일·사유(``stop``)는 같고 수익은 라이브가 유리하다.
  장중 폴링(60초) 사이의 터치는 놓칠 수 있다. 시간 손절·최대 보유는 15:20 종가
  동시호가 — 판단 가격은 호출부가 넘기는 가격(보통 15:20 직전 현재가)이다.

수익 기록: 트레이드 총수익 = Σ (매도 수량 / 최초 수량) × (매도가 / 진입가 − 1). 소수 수량(자본 무제한 패리티 모드)
에서 익절 비율 0.5 면 연구의 ``0.5 × (ret_half1 + ret_half2)`` 와 같다. 정수 수량에서 1주는 나눌 수 없어 익절 매도가
0주가 된다 — 그래도 상태는 익절 뒤 단계로 넘어간다(손절 → 본전, MA50 추적).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from swing_it.live.config import DEFAULT_STRATEGY, StrategyConfig
from swing_it.live.sizing import position_size, stop_width

#: :func:`scan` 이 읽는 패널 이름(날짜×종목 DataFrame) — :func:`swing_it.live.features.build_signal_panels` 가 만든다.
#: ``market`` 은 code → 시장 Series.
SCAN_KEYS = ("close", "pivot", "tt", "rs", "breakout", "breadth", "universe", "pead_rank", "rel", "F",
             "contraction_low", "adv20")
#: 수급 유전자(``smart_min``·``foreign_pos``·``persist_min``)가 설정됐을 때만 읽는 패널.
FLOW_KEYS = ("smart", "foreign", "persist")
#: 실적 문(GA2 유전자)이 켜졌을 때만 읽는 패널 — :func:`swing_it.live.features.earnings_panels`.
EARNINGS_KEYS = ("eps_yoy", "accel", "rev_yoy")

ActionKind = Literal["stop", "partial", "breakeven", "trail_next_open", "failed_breakout_next_open", "time_close",
                     "max_hold", "below_low", "delisted", "data_end"]
ActionAt = Literal["open", "intraday", "close", "next_open"]

#: 마지막 청산 Action 종류 → 연구 ``exit_reason`` (패리티·리포트용). ``partial`` 은 마지막 청산이 될 수 없다.
RESEARCH_REASON: dict[str, str] = {
    "stop": "stop", "breakeven": "breakeven", "trail_next_open": "trail", "failed_breakout_next_open": "failed_breakout",
    "time_close": "time_stop",
    "max_hold": "time", "below_low": "below_low", "delisted": "delisted", "data_end": "data_end",
}


# ---------------------------------------------------------------------------
# 값 객체
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """신호일 ``signal_day`` 종가까지로 정해진 진입 후보 한 종목."""

    code: str
    signal_day: pd.Timestamp
    market: str
    close: float               # 신호일 종가(분할조정) = 다음날의 전일종가
    pivot: float               # 직전 50일 최고 고가
    contraction_low: float     # min(low[t−10..t])
    rel: float                 # 상대 수렴(NaN 가능 — rel 조건을 끈 변형)
    rs: float
    adv20: float               # 백만원
    pead_rank: float
    flow_f: float              # 1.0 / 0.0 / NaN(모름)

    def rank_key(self) -> tuple:
        """후보 순위 키 — rel 오름차순(더 강한 수렴, NaN 은 뒤) → RS 내림차순 → 코드(결정성)."""
        rel = self.rel if np.isfinite(self.rel) else math.inf
        return (rel, -self.rs, self.code)


@dataclass(frozen=True)
class EntryPlan:
    """다음 거래일 장전 동시호가 지정가 매수 계획.

    체결 규칙(:meth:`fills_at`): 시가 ≤ ``limit_px`` ∧ 시가 < ``limit_up_px`` 이면 **시가**에 ``qty`` 전량 체결,
    아니면 미체결(그날 취소 — 다음 날로 미루지 않는다, 연구 ``skip_no_open``·``skip_chase``·``skip_limit_up``).
    """

    code: str
    signal_day: pd.Timestamp
    market: str
    limit_px: float
    limit_up_px: float
    qty: float                 # 정수 주(라이브·백테스트) — 패리티 모드에선 소수 허용
    ref_px: float              # 사이징 기준가(신호일 종가)
    d_est: float               # 신호일 기준 예상 손절폭
    pivot: float
    contraction_low: float
    rel: float
    rs: float

    def fills_at(self, open_px: float) -> bool:
        """그날 시가로 체결되는가(장전 지정가 → 시가 ≤ 지정가면 시가 체결)."""
        return bool(np.isfinite(open_px) and open_px > 0 and open_px <= self.limit_px and open_px < self.limit_up_px)


@dataclass(frozen=True)
class Action:
    """상태 전이가 낸 매도 행동.

    - ``kind``: stop / partial / breakeven / trail_next_open / time_close / max_hold / below_low (+ 백테스트 전용
      delisted / data_end).
    - ``qty_frac``: **행동 직전 보유 수량** 대비 매도 비율(1.0 = 남은 전부, 익절 = ``cfg.partial_frac``).
    - ``price``: 체결가(일봉 모형) 또는 판단 기준가(라이브 장중). ``at == "next_open"`` 이면 ``None`` — 다음 거래일 시가
      주문을 내라는 결정(MA50 추적).
    - ``qty``: 매도 주수(상태가 계산한 값, 정수 수량에서 익절 0주 가능).
    """

    kind: str
    qty_frac: float
    price: "float | None"
    at: str = "intraday"
    qty: float = 0.0
    day_no: int = 0


# ---------------------------------------------------------------------------
# 진입 판정
# ---------------------------------------------------------------------------

def _row(panel: pd.DataFrame, i: int) -> np.ndarray:
    v = panel.iloc[i]
    if str(v.dtype) == "boolean":
        return v.astype("float64").to_numpy(dtype=float, na_value=np.nan)
    return v.to_numpy(dtype=float, na_value=np.nan)


def scan(panels: Mapping[str, "pd.DataFrame | pd.Series"], d, cfg: StrategyConfig = DEFAULT_STRATEGY
         ) -> list[Candidate]:
    """신호일 ``d`` 의 진입 후보(전부 ``d`` 종가까지) → :meth:`Candidate.rank_key` 순 정렬 목록.

    Args:
        panels: :data:`SCAN_KEYS` 이름의 날짜×종목 패널(전부 같은 모양) + ``market`` (code → 시장 Series).
            :func:`swing_it.live.features.build_signal_panels` 결과를 그대로 넘긴다(라이브 = 그날치 창, 백테스트 =
            전 기간). bool 패널(``tt``·``breakout``·``universe``)은 0/1 로 읽고 NaN 은 거짓이다.
        d: 신호일. 패널에 없는 날이면 빈 목록.
        cfg: 전략 설정. ``rel_max=None`` 이면 rel 조건, ``require_flow=False`` 면 F 조건을 끈다.

    조건(전부 AND; 켜진 문턱에서 NaN 은 통과하지 않고, ``None`` = 끔인 문턱은 조건 자체가 없다 — GA2 마스크와 같다):
    ``universe`` ∧ ``tt``(cfg 의 4성분으로 features 가 만든다) ∧ ``breakout``(cfg: 돌파·거래량 조건, 둘 다 끔이면 전부 참) ∧
    ``rs ≥ rs_min`` ∧ ``breadth > breadth_min`` ∧ ``pead_rank ≥ pead_rank_min`` ∧ ``rel ≤ rel_max`` ∧ 수급(``F == 1``, 또는
    수급 유전자가 있으면 ``smart ≥ smart_min`` ∧ (``foreign > 0``) ∧ ``persist ≥ persist_min``) ∧ 수축 저점·피벗·종가 유한
    ∧ (켜진 실적 문: ``eps_yoy ≥ eps_yoy_min`` · ``accel == 1`` · ``rev_yoy ≥ rev_yoy_min`` — :data:`EARNINGS_KEYS` 패널).
    수축 저점이 없으면 연구는 ``skip_no_contraction`` 으로 버린다 — 여기선 후보에서 뺀다(결과 같음).
    """
    close = panels["close"]
    d = pd.Timestamp(d)
    if d not in close.index:
        return []
    i = int(close.index.get_loc(d))
    r = {k: _row(panels[k], i) for k in SCAN_KEYS}
    flow_genes = cfg.smart_min is not None or cfg.foreign_pos is not None or cfg.persist_min
    with np.errstate(invalid="ignore"):
        ok = ((r["universe"] == 1) & (r["tt"] == 1) & (r["breakout"] == 1)
              & np.isfinite(r["contraction_low"]) & np.isfinite(r["pivot"]) & np.isfinite(r["close"]))
        if cfg.rs_min is not None:
            ok &= r["rs"] >= cfg.rs_min
        if cfg.breadth_min is not None:
            ok &= r["breadth"] > cfg.breadth_min
        if cfg.pead_rank_min is not None:
            ok &= r["pead_rank"] >= cfg.pead_rank_min
        if cfg.rel_max is not None:
            ok &= r["rel"] <= cfg.rel_max
        if flow_genes:
            if cfg.smart_min is not None:
                ok &= _row(panels["smart"], i) >= cfg.smart_min
            if cfg.foreign_pos:
                ok &= _row(panels["foreign"], i) > 0
            if cfg.persist_min:
                ok &= _row(panels["persist"], i) >= cfg.persist_min
        elif cfg.require_flow:
            ok &= r["F"] == 1
        if cfg.eps_yoy_min is not None:
            ok &= _row(panels["eps_yoy"], i) >= cfg.eps_yoy_min
        if cfg.eps_accel:
            ok &= _row(panels["accel"], i) == 1
        if cfg.rev_yoy_min is not None:
            ok &= _row(panels["rev_yoy"], i) >= cfg.rev_yoy_min
    idx = np.flatnonzero(ok)
    codes = close.columns
    market = panels["market"]
    out = [Candidate(code=str(codes[j]), signal_day=d, market=str(market.get(codes[j], "X")),
                     close=float(r["close"][j]), pivot=float(r["pivot"][j]),
                     contraction_low=float(r["contraction_low"][j]), rel=float(r["rel"][j]), rs=float(r["rs"][j]),
                     adv20=float(r["adv20"][j]), pead_rank=float(r["pead_rank"][j]), flow_f=float(r["F"][j]))
           for j in idx]
    return sorted(out, key=Candidate.rank_key)


# ---------------------------------------------------------------------------
# 주문 계획
# ---------------------------------------------------------------------------

def floor_tick(price: float) -> float:
    """호가 **내림**(현행 KRX 호가표, ``kiwoom_client.round_to_tick``). 0 이하·NaN 은 NaN."""
    if not (np.isfinite(price) and price > 0):
        return math.nan
    from krx_quant_core.market.ticks import round_to_tick  # noqa: PLC0415 — 호가 정본은 코어(kiwoom_client)

    return float(round_to_tick(price))


def entry_limit(pivot: float, prev_close: float, cfg: StrategyConfig = DEFAULT_STRATEGY, *,
                tick_grid: bool = True) -> "tuple[float, float]":
    """``(지정가, 상한가 기준)``. 체결 = 시가 ≤ 지정가 ∧ 시가 < 상한가 기준.

    ``tick_grid=True`` (라이브 원자료): 지정가 = 호가 내림(min(피벗×(1+chase), 상한가 − 1틱)), 상한가 기준 =
    ``krx_quant_core.market.limits.limit_up_price(전일종가)``.
    ``tick_grid=False`` (분할조정 백테스트): 지정가 = 피벗 × (1+chase) 그대로, 상한가 기준 = 전일종가 ×
    ``cfg.limit_up_ratio`` — 연구 ``sepa``/``minervini_exits`` 의 추격·상한가 판정과 같은 식(같은 부동소수).
    """
    cap = pivot * (1.0 + cfg.chase) if np.isfinite(cfg.chase) else math.inf   # chase = inf → 추격 금지 없음
    if not tick_grid:
        lu = prev_close * cfg.limit_up_ratio if (np.isfinite(prev_close) and prev_close > 0) else math.inf
        # 추격 금지 없음: 지정가 = 상한가 기준(체결 조건 "시가 < 상한가" 와 같은 집합, 현금 예약은 유한하게)
        return (cap if np.isfinite(cap) else lu), lu
    from krx_quant_core.market.limits import limit_up_price  # noqa: PLC0415
    from krx_quant_core.market.ticks import shift_ticks  # noqa: PLC0415

    lu = float(limit_up_price(prev_close))
    below = float(shift_ticks(int(lu), -1))
    return floor_tick(min(cap, below)), lu


def plan_entry(c: Candidate, equity: float, prev_close: float, cfg: StrategyConfig = DEFAULT_STRATEGY, *,
               tick_grid: bool = True, fractional: bool = False) -> EntryPlan:
    """후보 → 지정가 매수 계획(수량 포함). 슬롯·현금 제약은 :func:`swing_it.live.sizing.allocate` 몫이다.

    수량 = floor(min(위험 ``risk_per_trade`` × 자산 / (기준가 × d_est), ``max_position_frac`` × 자산 / 기준가)),
    기준가 = ``prev_close`` (신호일 종가 — 진입가는 체결 전엔 모른다), d_est = :func:`swing_it.live.sizing.stop_width`
    (기준가, 수축 저점). ``fractional`` 이면 내림 없이 소수 수량(패리티 모드).
    """
    limit_px, lu = entry_limit(c.pivot, prev_close, cfg, tick_grid=tick_grid)
    d_est = stop_width(prev_close, c.contraction_low, cfg)
    qty = position_size(equity, prev_close, d_est, cfg, fractional=fractional)
    return EntryPlan(code=c.code, signal_day=c.signal_day, market=c.market, limit_px=limit_px, limit_up_px=lu,
                     qty=qty, ref_px=prev_close, d_est=d_est, pivot=c.pivot, contraction_low=c.contraction_low,
                     rel=c.rel, rs=c.rs)


# ---------------------------------------------------------------------------
# 포지션 상태
# ---------------------------------------------------------------------------

Phase = Literal["pending", "full", "half", "closed"]


@dataclass
class PositionState:
    """한 트레이드의 상태(직렬화 가능 — ``state.py`` 원장이 :meth:`to_dict` / :meth:`from_dict` 로 저장·복원).

    생성: 계획(:class:`EntryPlan`) 의 ``code``·``contraction_low``·``qty`` 로 만들고(``phase="pending"``), 체결되면
    :meth:`on_fill_open`. 날짜 번호 ``day_no`` 는 **진입일 = 1** 이고 정지일도 센다(연구 오프셋 + 1).
    """

    code: str
    contraction_low: float
    qty: float
    market: str = "X"
    signal_day: "str | None" = None
    pivot: float = math.nan                # 돌파 기준가(계획의 pivot) — 돌파 실패 판정
    fractional: bool = False               # True = 소수 수량(패리티 모드) — 익절 수량을 내리지 않는다
    cfg: StrategyConfig = field(default=DEFAULT_STRATEGY, repr=False)

    phase: str = "pending"
    entry_px: float = math.nan
    entry_day: "str | None" = None
    d: float = math.nan
    stop_px: float = math.nan
    target_px: float = math.nan
    qty_open: float = 0.0
    day_no: int = 0
    partial_day_no: "int | None" = None
    breakeven_from: "int | None" = None    # 본전 손절이 걸리는 첫 day_no
    trail_pending: bool = False            # 다음 거래일 시가 매도 결정(MA50 추적 또는 돌파 실패)
    pending_kind: "str | None" = None      # 그 결정의 종류: trail_next_open / failed_breakout_next_open
    time_checked: bool = False             # 시간 손절은 한 번만 본다
    exits: list = field(default_factory=list)  # [{day_no, kind, qty, price}]

    # --- 직렬화 ---
    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("cfg")
        return out

    @classmethod
    def from_dict(cls, data: dict, cfg: StrategyConfig = DEFAULT_STRATEGY) -> "PositionState":
        kw = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "cfg"}
        kw["exits"] = [dict(x) for x in kw.get("exits", [])]
        return cls(cfg=cfg, **kw)

    @classmethod
    def from_plan(cls, plan: EntryPlan, cfg: StrategyConfig = DEFAULT_STRATEGY, *,
                  qty: "float | None" = None, fractional: bool = False) -> "PositionState":
        return cls(code=plan.code, contraction_low=plan.contraction_low, qty=plan.qty if qty is None else qty,
                   market=plan.market, signal_day=str(pd.Timestamp(plan.signal_day).date()), pivot=float(plan.pivot),
                   fractional=fractional, cfg=cfg)

    # --- 조회 ---
    @property
    def is_open(self) -> bool:
        return self.phase in ("full", "half")

    @property
    def partial_hit(self) -> bool:
        return self.partial_day_no is not None

    def gross_return(self) -> float:
        """총수익(비용 전) = Σ (매도 수량 / 최초 수량) × (매도가 / 진입가 − 1). 아직 열려 있으면 실현분만."""
        if not self.exits or not np.isfinite(self.entry_px) or self.qty <= 0:
            return math.nan
        return float(sum((x["qty"] / self.qty) * (x["price"] / self.entry_px - 1.0) for x in self.exits))

    def last_exit(self) -> "dict | None":
        return self.exits[-1] if self.exits else None

    def time_stop_due(self, day_no: "int | None" = None) -> bool:
        """오늘 종가에 시간 손절을 봐야 하나(익절 전 ∧ 아직 안 봄 ∧ day_no ≥ time_stop_days)."""
        n = self.day_no if day_no is None else day_no
        ts = self.cfg.time_stop_days
        return self.phase == "full" and ts is not None and not self.time_checked and n >= ts

    def max_hold_due(self, day_no: "int | None" = None) -> bool:
        """오늘 종가가 최대 보유 청산인가."""
        n = self.day_no if day_no is None else day_no
        return self.is_open and n >= self.cfg.max_hold

    # --- 내부 전이(모든 진입점이 공유) ---
    def _sell(self, kind: str, qty: float, price: "float | None", at: str) -> Action:
        before = self.qty_open
        frac = (qty / before) if before > 0 else 0.0
        if price is not None:
            self.qty_open = before - qty
            self.exits.append({"day_no": self.day_no, "kind": kind, "qty": qty, "price": float(price)})
        return Action(kind=kind, qty_frac=frac, price=None if price is None else float(price), at=at, qty=qty,
                      day_no=self.day_no)

    def _exit_all(self, kind: str, price: float, at: str) -> Action:
        a = self._sell(kind, self.qty_open, price, at)
        self.qty_open = 0.0
        self.phase = "closed"
        self.trail_pending = False
        self.pending_kind = None
        return a

    def _partial_qty(self) -> float:
        q = self.qty_open * self.cfg.partial_frac
        return q if self.fractional else float(math.floor(q + 1e-9))

    def _take_partial(self, price: float, at: str, *, gap: bool) -> Action:
        a = self._sell("partial", self._partial_qty(), price, at)
        self.phase = "half"
        self.partial_day_no = self.day_no
        # 본전은 다음 거래일부터 — 단 시가 익절(갭 상승)은 시가가 먼저인 게 확정이라 그날부터.
        self.breakeven_from = self.day_no if gap else self.day_no + 1
        if self.qty_open <= 0:          # 소수 수량 끝자리 등으로 남은 게 없으면 닫는다
            self.phase = "closed"
        return a

    def _breakeven_active(self) -> bool:
        return self.phase == "half" and self.breakeven_from is not None and self.day_no >= self.breakeven_from

    # --- 체결 ---
    def on_fill_open(self, price: float, *, day_no: int = 1, entry_day=None) -> list[Action]:
        """장전 지정가가 시가 ``price`` 에 체결됐다 → 손절·목표 확정. 시가 ≤ 수축 저점이면 그날 전량 청산.

        d = clip(1 − 수축저점 × stop_mult / 진입가, stop_floor, stop_cap) — 연구와 같은 식·같은 순서로 계산한다.
        """
        if self.phase != "pending":
            raise ValueError(f"{self.code}: 이미 체결된 포지션({self.phase})")
        if not np.isfinite(self.contraction_low):
            raise ValueError(f"{self.code}: 수축 저점이 없다(연구 skip_no_contraction — scan 이 걸렀어야 한다)")
        if self.cfg.failed_breakout and not np.isfinite(self.pivot):
            raise ValueError(f"{self.code}: 돌파 실패 판정에 피벗이 필요하다(계획의 pivot)")
        cfg = self.cfg
        entry = float(price)
        self.entry_px = entry
        self.entry_day = None if entry_day is None else str(pd.Timestamp(entry_day).date())
        self.day_no = day_no
        self.qty_open = float(self.qty)
        self.phase = "full"
        self.d = float(min(max(1.0 - self.contraction_low * cfg.stop_mult / entry, cfg.stop_floor), cfg.stop_cap))
        self.stop_px = entry * (1.0 - self.d)
        self.target_px = entry * (1.0 + cfg.take_r * self.d)   # take_r = inf → 목표 없음(익절 없음)
        if entry <= self.contraction_low:
            return [self._exit_all("below_low", entry, "open")]
        return []

    # --- 일봉(백테스트) ---
    def on_bar(self, open: float, high: float, low: float, close: float, ma50: float, day_no: int, *,
               tradable: bool = True) -> list[Action]:
        """하루 일봉으로 상태 전이(연구 시뮬레이터와 같은 비관적 일봉 순서). 진입일엔 체결 뒤 같은 봉으로 부른다.

        ``tradable=False`` (정지·시세 없음)면 날짜만 센다. 반환 Action 의 ``price`` 는 체결가이고, ``next_open`` 은
        다음 거래일 시가 매도 결정(그날 :meth:`on_bar` 가 시가로 체결한다).
        """
        self.day_no = day_no
        if not self.is_open or not tradable:
            return []
        acts: list[Action] = []
        if self.trail_pending:                                           # 전날 결정(추적·돌파 실패) → 시가
            return self._pending_exit(open)
        if self.phase == "full":
            if low <= self.stop_px or open <= self.stop_px:              # 같은 날이면 손절 먼저
                return [self._exit_all("stop", min(open, self.stop_px), "intraday")]
            if self.cfg.partial_on and (high >= self.target_px or open >= self.target_px):
                gap = open >= self.target_px
                acts.append(self._take_partial(open if gap else self.target_px, "open" if gap else "intraday",
                                               gap=gap))
                if not self.is_open:
                    return acts
                if gap and low <= self.entry_px:                          # 갭 익절일: 그날 본전 체결 확정
                    acts.append(self._exit_all("breakeven", self.entry_px, "intraday"))
                    return acts
                acts.extend(self._close_half(close, ma50))
                return acts
            return self._close_full(close, ma50)
        # 익절 뒤 나머지
        if self._breakeven_active() and (low <= self.entry_px or open <= self.entry_px):
            return [self._exit_all("breakeven", min(open, self.entry_px), "intraday")]
        return self._close_half(close, ma50)

    def _pending_exit(self, open: float) -> list[Action]:
        """전날 결정(MA50 추적·돌파 실패)을 시가에 실행 — 그날 손절·목표보다 먼저."""
        kind = self.pending_kind or "trail_next_open"
        return [self._exit_all(kind, open, "open")]

    def _decide_next_open(self, kind: str) -> list[Action]:
        self.trail_pending = True
        self.pending_kind = kind
        return [self._sell(kind, self.qty_open, None, "next_open")]

    def _close_full(self, close: float, ma: float = math.nan) -> list[Action]:
        """전량 단계의 종가 판단: 시간 손절(켜져 있으면, 한 번) → 최대 보유 → 돌파 실패(종가 < 피벗 → 다음 거래일 시가)
        → (익절 없음 ∧ 추적 있음) 전량 추적: 종가 < MA → 다음 거래일 시가(GA 커널 ``j_t1`` — 같은 날이면 돌파 실패가 먼저)."""
        if self.time_stop_due():
            self.time_checked = True
            if close <= self.entry_px:
                return [self._exit_all("time_close", close, "close")]
        if self.max_hold_due():
            return [self._exit_all("max_hold", close, "close")]
        if self.trail_pending:
            return []
        if self.cfg.failed_breakout and close < self.pivot:
            return self._decide_next_open("failed_breakout_next_open")
        if not self.cfg.partial_on and self.cfg.has_trail and np.isfinite(ma) and close < ma:
            return self._decide_next_open("trail_next_open")
        return []

    def _close_half(self, close: float, ma50: float) -> list[Action]:
        """익절 뒤 나머지의 종가 판단: 최대 보유 → 그 종가, 아니면 종가 < MA → 다음 거래일 시가 결정."""
        if self.max_hold_due():
            return [self._exit_all("max_hold", close, "close")]
        if self.cfg.has_trail and np.isfinite(ma50) and close < ma50 and not self.trail_pending:
            return self._decide_next_open("trail_next_open")
        return []

    # --- 라이브 이벤트(장전 시가 / 장중 현재가 / 종가) ---
    def on_open(self, open: float, day_no: int) -> list[Action]:
        """거래일 시작(시가 확인). 전일 결정(MA50 추적·돌파 실패)이면 시가 매도, 갭 하락 ≤ 손절이면 손절, 갭 상승 ≥ 목표면 익절,
        본전 활성 중 시가 ≤ 진입가면 본전. 진입일엔 부르지 않는다(:meth:`on_fill_open` 이 그 몫)."""
        self.day_no = day_no
        if not self.is_open:
            return []
        if self.trail_pending:
            return self._pending_exit(open)
        if self.phase == "full":
            if open <= self.stop_px:
                return [self._exit_all("stop", open, "open")]
            if self.cfg.partial_on and open >= self.target_px:
                return [self._take_partial(open, "open", gap=True)]
            return []
        if self._breakeven_active() and open <= self.entry_px:
            return [self._exit_all("breakeven", open, "open")]
        if open <= self.stop_px:                                  # 본전 전(익절일) — 원래 손절이 남아 있다
            return [self._exit_all("stop", open, "open")]
        return []

    def on_quote(self, last: float) -> list[Action]:
        """장중 현재가. 손절가 이하 → 손절, 목표가 이상 → 절반 익절(본전은 다음 거래일부터), 본전 활성 중 진입가
        이하 → 본전. ``price`` 는 판단 기준가(현재가)다 — 실제 체결가는 브로커 체결로 확정한다."""
        if not self.is_open or not np.isfinite(last):
            return []
        if self.phase == "full":
            if last <= self.stop_px:
                return [self._exit_all("stop", last, "intraday")]
            if self.cfg.partial_on and last >= self.target_px:
                return [self._take_partial(last, "intraday", gap=False)]
            return []
        if self._breakeven_active() and last <= self.entry_px:
            return [self._exit_all("breakeven", last, "intraday")]
        # 익절일 나머지 절반: 본전은 다음 거래일부터지만 **원래 손절은 그날 내내 산다**(REVIEW_BACKTEST M1 — 일봉 모형의
        # "같은 날 손절 먼저" 와 같은 보호. 라이브는 이미 절반을 팔았으므로 남은 절반만 손절가에 판다).
        if last <= self.stop_px:
            return [self._exit_all("stop", last, "intraday")]
        return []

    def on_close(self, close: float, ma50: float, day_no: "int | None" = None) -> list[Action]:
        """종가 판단(라이브: 15:20 종가 동시호가 = 시간 손절·최대 보유, 장후/다음날 아침 = MA50 추적·돌파 실패 결정).

        전량 단계: 시간 손절(켜져 있으면, 한 번) → 최대 보유 → **돌파 실패**(종가 < 피벗 → ``failed_breakout_next_open``,
        다음 거래일 시가). 익절 뒤: 최대 보유 → 종가 < MA → ``trail_next_open``. 둘 다 ``at="next_open"`` 결정이고 다음
        :meth:`on_open` 이 시가에 실행한다.
        """
        if day_no is not None:
            self.day_no = day_no
        if not self.is_open:
            return []
        if self.phase == "full":
            return self._close_full(close, ma50)
        return self._close_half(close, ma50)

    # --- 백테스트 전용 ---
    def force_close(self, kind: str, price: float, day_no: "int | None" = None) -> list[Action]:
        """데이터 끝(``data_end``)·상장폐지(``delisted``) — 마지막 거래 가능일 종가로 남은 전부를 닫는다."""
        if day_no is not None:
            self.day_no = day_no
        if not self.is_open:
            return []
        return [self._exit_all(kind, price, "close")]
