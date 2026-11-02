"""3단계 알파 발굴 — 구현 없음. 신호는 러너가 만들어 :class:`~swing_it.pipeline.contracts.Signal` 로 넘긴다.

이 자리에 있던 ``earnings_yoy_panel`` 래퍼를 지웠다. 실적을 쓸 수 있는 첫날은 max(avail_date, 정기보고서 첫 접수일) 다음
거래일인데 그 래퍼는 avail_date 만 봤다(2019~ 종목-일의 3.6% 가 달랐고, 807건은 알기 전에 썼다). 그 규칙의 구현은
``yoy_signal_strict`` 다.
"""

from __future__ import annotations

from swing_it.pipeline.contracts import STAGE_CONTRACTS, DataPanels, Signal
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.ALPHA
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "research/experiments/pead_optimizer_po4.py::yoy_signal_strict",
    "src/swing_it/features/fundamentals.py::earnings_yoy_panel",
)


def run(study: str, panels: DataPanels) -> Signal:
    require_stage(study, STAGE)
    raise NotImplementedError("3단계 공용 구현은 없다. 알게 된 날 규칙을 지키는 구현은 "
                              "research/experiments/pead_optimizer_po4.py::yoy_signal_strict 다.")
