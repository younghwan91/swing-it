"""8단계 거래비용 모델 — core ``costs.impact.trade_cost_bp``(정가). 기본 핀 0.7.0 에는 없어 부를 때 import 한다.

할인·면제 인자는 두지 않는다. 비용은 정가로 보고, 판정은 ``multiplier=2`` 로 한다(레포 CLAUDE.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from swing_it.pipeline.contracts import STAGE_CONTRACTS, CostEstimate, TargetWeights, core_layer, expect
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.COSTS
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "research/experiments/pead_optimizer_b.py::trade_costs",
    "research/experiments/slippage_check.py",
)


def run(study: str, target: TargetWeights, w_prev: Any, *, aum: float, adv_value: Any, daily_vol: Any, market: Any,
        trade_date: Any, half_spread_bp: Any, y: float = 1.0, multiplier: float = 1.0) -> CostEstimate:
    """``w_prev`` → ``target`` 거래의 종목별 편도 비용과 NAV 대비 합계."""
    require_stage(study, STAGE)
    expect(target, TargetWeights, "target")
    trade_cost_bp = core_layer("krx_quant_core.costs.impact").trade_cost_bp
    dw = np.asarray(target.weights, dtype=np.float64) - np.asarray(w_prev, dtype=np.float64)
    side = np.where(dw >= 0, "BUY", "SELL")
    bp = trade_cost_bp(side, np.abs(dw) * aum, adv_value, daily_vol, market, trade_date,
                       half_spread_bp=half_spread_bp, y=y)
    nav_cost = float(np.sum(np.abs(dw) * bp) * 1e-4 * multiplier)
    return CostEstimate(trade_date=trade_date, codes=target.codes, one_way_bp=bp, nav_cost=nav_cost,
                        multiplier=multiplier)
