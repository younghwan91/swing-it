"""7단계 포트폴리오 최적화 — core ``portfolio.optimize``. 기본 핀 0.7.0 에는 없어 부를 때 import 한다.

순위 북(트랜치 동일가중)은 이 단계를 일부러 건너뛴다. 원장에 skip 과 사유를 적는다.
"""

from __future__ import annotations

from typing import Any

from swing_it.pipeline.contracts import STAGE_CONTRACTS, RiskModelOutput, TargetWeights, core_layer, expect
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.PORTFOLIO
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "research/experiments/pead_optimizer_b.py::o_target",
    "research/experiments/pead_optimizer_b_feasibility.py::solve_at",
)


def run(study: str, alpha: Any, risk: RiskModelOutput, w_prev: Any, **kwargs: Any) -> TargetWeights:
    """리밸런스 한 번. ``kwargs`` 는 ``optimize`` 인자. 풀이 실패는 ``status`` 로 돌아오고 비중은 ``w_prev`` 다."""
    require_stage(study, STAGE)
    expect(risk, RiskModelOutput, "risk")
    optimize = core_layer("krx_quant_core.portfolio").optimize
    res = optimize(alpha, risk.model, w_prev, **kwargs)
    return TargetWeights(asof=risk.asof, codes=risk.model.codes, weights=res.weights, method="optimizer",
                         status=res.status)
