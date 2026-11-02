"""11단계 성과 기여·모니터링 — core ``risk.attribution.factor_attribution``. 기본 핀 0.7.0 에는 없어 부를 때 import 한다."""

from __future__ import annotations

from typing import Any, Sequence

from swing_it.pipeline.contracts import STAGE_CONTRACTS, Attribution, core_layer
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.ATTRIBUTION
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = ("research/experiments/pead_optimizer_b_report.py",)


def run(study: str, w: Any, X: Any, f: Any, u: Any, *, names: Sequence[str], dates: Sequence) -> Attribution:
    """날짜별 팩터 기여·특이 기여·합계. 입력 모양은 core 가 검사한다."""
    require_stage(study, STAGE)
    attribute = core_layer("krx_quant_core.risk.attribution").factor_attribution
    return Attribution(frame=attribute(w, X, f, u, names=names, dates=dates), names=tuple(names))
