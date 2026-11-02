"""파이프라인 11단계 — 단계별 모듈과 순서 가드레일. 사용법은 ``docs/PIPELINE.md``.

여기서는 게이트만 내보낸다(표준 라이브러리만 쓴다). 계약과 단계 모듈은 numpy·pandas 가 필요하니
``from swing_it.pipeline import s04_alpha_validation`` 처럼 필요한 것만 부른다.
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
