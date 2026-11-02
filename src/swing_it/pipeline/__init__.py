"""파이프라인 11단계의 순서 가드레일. 사용법은 ``docs/PIPELINE.md``.

여기서는 게이트만 내보낸다(표준 라이브러리만 쓴다). 단계의 실제 구현은 ``swing_it.features``·``swing_it.book``·``swing_it.validation`` 에
있고, 표는 :data:`swing_it.pipeline.stages.STAGE_FUNCTIONS` 다. 경계 검사는 ``swing_it.pipeline.checks`` 다(numpy·pandas 필요).
"""

from __future__ import annotations

from swing_it.pipeline import gate
from swing_it.pipeline.gate import (
    LedgerCorrupt,
    RecordRejected,
    StageGateError,
    StageState,
    record_stage,
    require_execution_approval,
    require_stage,
    study_states,
    verify,
)
from swing_it.pipeline.stages import Stage

__all__ = [
    "LedgerCorrupt",
    "RecordRejected",
    "Stage",
    "StageGateError",
    "StageState",
    "gate",
    "record_stage",
    "require_execution_approval",
    "require_stage",
    "study_states",
    "verify",
]
