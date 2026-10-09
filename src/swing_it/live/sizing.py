"""스윙 라이브 사이징 — 위험 1% / 종목 10% 상한 / 최대 10종목 / 현금 제약 / 정수 주 (설계 §2).

- 예상 손절폭 ``d_est = clip(1 − 수축저점 × 0.99 / 기준가, 3%, 10%)`` — 체결가 대신 신호일 종가(기준가)로 미리 잰다.
  실제 손절폭은 체결가로 다시 잰다(:meth:`swing_it.live.strategy.PositionState.on_fill_open`).
- 주수 = ``floor(min(위험 × 자산 / (기준가 × d_est), 상한 × 자산 / 기준가))``.
- :func:`allocate`: 후보가 빈 슬롯보다 많으면 rel 오름차순(더 강한 수렴) → RS 내림차순으로 자른다. 현금이 모자라면
  그 계획의 수량을 **현금이 허용하는 만큼 줄이고**(최소 1주), 1주도 못 사면 건너뛰고 다음 후보를 본다. 현금 예약은
  ``수량 × 지정가`` (최악 체결가 — 시가 ≤ 지정가라 실제 지출은 이 이하).

순수 함수. :mod:`swing_it.live.strategy` 가 import 하므로 여기서 strategy 를 import 하지 않는다(덕 타이핑).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from swing_it.live.config import DEFAULT_STRATEGY, StrategyConfig


def stop_width(ref_px: float, contraction_low: float, cfg: StrategyConfig = DEFAULT_STRATEGY) -> float:
    """손절폭 ``clip(1 − 수축저점 × stop_mult / 기준가, stop_floor, stop_cap)``. 입력이 비정상이면 상한(보수적)."""
    if not (np.isfinite(ref_px) and ref_px > 0 and np.isfinite(contraction_low)):
        return cfg.stop_cap
    return float(min(max(1.0 - contraction_low * cfg.stop_mult / ref_px, cfg.stop_floor), cfg.stop_cap))


def position_size(equity: float, ref_px: float, d_est: float, cfg: StrategyConfig = DEFAULT_STRATEGY, *,
                  fractional: bool = False) -> float:
    """주수 = floor(min(위험 × 자산 / (기준가 × d_est), 상한 × 자산 / 기준가)). 비정상 입력·음수 자산은 0."""
    if not (np.isfinite(equity) and equity > 0 and np.isfinite(ref_px) and ref_px > 0
            and np.isfinite(d_est) and d_est > 0):
        return 0.0
    q = min(cfg.risk_per_trade * equity / (ref_px * d_est), cfg.max_position_frac * equity / ref_px)
    return float(q) if fractional else float(math.floor(q + 1e-9))


def rank_plans(plans: Iterable) -> list:
    """rel 오름차순(NaN 뒤) → RS 내림차순 → 코드. ``rel``·``rs``·``code`` 속성만 본다(Candidate·EntryPlan 공용)."""
    def key(p):
        rel = p.rel if np.isfinite(p.rel) else math.inf
        return (rel, -p.rs, p.code)

    return sorted(plans, key=key)


def allocate(plans: Iterable, *, cash: float, open_slots: int, held: "set[str] | None" = None,
             fractional: bool = False) -> list:
    """계획 목록 → 실제로 낼 주문 목록(순위 순). 슬롯·현금·보유 중 종목을 거른다.

    Args:
        plans: :class:`swing_it.live.strategy.EntryPlan` 들(``qty``·``limit_px``·``code``·``rel``·``rs``).
        cash: 주문에 쓸 수 있는 현금(원).
        open_slots: 빈 슬롯 수(``max_positions − 보유 종목 수``). 음수면 0.
        held: 이미 보유 중인 종목(같은 종목 재진입 없음 — 연구 ``no_overlap``).
        fractional: 소수 수량 허용(패리티 모드 — 현금 제약을 끄려면 ``cash=inf``).

    Returns:
        수량이 조정된 계획 목록(``dataclasses.replace`` 로 ``qty`` 만 바꾼 사본). 수량 0 인 계획은 빠진다.
    """
    from dataclasses import replace  # noqa: PLC0415

    held = held or set()
    out = []
    left = float(cash)
    slots = max(int(open_slots), 0) if open_slots != math.inf else math.inf
    for p in rank_plans(plans):
        if len(out) >= slots:
            break
        if p.code in held or p.qty <= 0 or not (np.isfinite(p.limit_px) and p.limit_px > 0):
            continue
        qty = p.qty
        if math.isfinite(left):
            afford = left / p.limit_px
            if not fractional:
                afford = math.floor(afford + 1e-9)
            qty = min(qty, afford)
        if qty <= 0 or (not fractional and qty < 1):
            continue
        left -= qty * p.limit_px
        out.append(replace(p, qty=float(qty)))
    return out
