"""1단계 유니버스 — ``features.universe.smallmid_universe`` 를 계약으로 감싼다."""

from __future__ import annotations

from typing import Any

import pandas as pd

from swing_it.features.universe import smallmid_universe
from swing_it.pipeline.contracts import STAGE_CONTRACTS, UniverseMask
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.UNIVERSE
CONTRACT = STAGE_CONTRACTS[STAGE]
#: 이 단계의 기존 구현들. src 는 research 를 import 하지 않으니 경로만 적는다(테스트가 실재를 확인한다).
REFERENCES = (
    "src/swing_it/features/universe.py::smallmid_universe",
    "src/swing_it/validation/universe_hygiene.py::assert_point_in_time",
    "research/experiments/pead_optimizer_b.py::eligible",
)


def run(study: str, cap_panel: pd.DataFrame, adv_panel: pd.DataFrame, *, definition: str, survivorship: str,
        **kwargs: Any) -> UniverseMask:
    """시총·ADV 긴 형식 패널에서 날짜별 적격 마스크를 만든다. ``kwargs`` 는 ``smallmid_universe`` 인자."""
    require_stage(study, STAGE)
    frame = smallmid_universe(cap_panel, adv_panel, **kwargs)
    return UniverseMask(frame=frame, definition=definition, survivorship=survivorship)
