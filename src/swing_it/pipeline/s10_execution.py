"""10단계 주문 실행 — 구현 없음. 주문·키움 호출·로그인을 하지 않는다.

``docs/execution/DESIGN.md`` 는 종가 동시호가 바구니의 dry-run 만 먼저 만든다고 정했다. 실주문 경로가 생기면 그 첫 문장은
:func:`~swing_it.pipeline.gate.require_execution_approval` 이어야 한다(``require_stage`` 는 앞 단계만 본다).
"""

from __future__ import annotations

from swing_it.pipeline.contracts import STAGE_CONTRACTS, OrderPlan, TargetWeights
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.EXECUTION
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = ("docs/execution/DESIGN.md",)


def run(study: str, target: TargetWeights) -> OrderPlan:
    require_stage(study, STAGE)
    raise NotImplementedError("10단계는 아직 없다. docs/execution/DESIGN.md §4 의 dry-run 부터 만든다 — 실주문은 그 뒤 사용자 승인으로만.")
