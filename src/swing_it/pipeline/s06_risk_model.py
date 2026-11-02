"""6단계 팩터 위험모델 — core ``risk.factor_model.build_risk_model``. 기본 핀 0.7.0 에는 없어 부를 때 import 한다."""

from __future__ import annotations

from typing import Any

from swing_it.pipeline.contracts import STAGE_CONTRACTS, RiskModelOutput, core_layer
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.RISK_MODEL
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "research/experiments/pead_optimizer_po3.py",
    "research/experiments/pead_optimizer_po4.py",
)


def run(study: str, R: Any, X_style: Any, ind_labels: Any, cap: Any, mask: Any, asof_index: int, *, asof: str,
        label: str, codes: Any = None, **kwargs: Any) -> RiskModelOutput:
    """``asof_index`` 까지만 써서 다음 날 위험모델을 만든다. ``kwargs`` 는 ``build_risk_model`` 인자."""
    require_stage(study, STAGE)
    build = core_layer("krx_quant_core.risk.factor_model").build_risk_model
    model = build(R, X_style, ind_labels, cap, mask, asof_index, codes=codes, **kwargs)
    return RiskModelOutput(asof=asof, model=model, label=label)
