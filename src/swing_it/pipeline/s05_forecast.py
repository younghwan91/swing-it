"""5단계 ML 수익률 예측 — main 에 구현이 없다. 건너뛰려면 원장에 skip 과 사유를 적는다."""

from __future__ import annotations

from swing_it.pipeline.contracts import STAGE_CONTRACTS, DataPanels, Forecast, Signal
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.FORECAST
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES: tuple[str, ...] = ()
POINTER = ("브랜치 younghwan91/ml-forecast-prereg 의 research/experiments/ml_forecast_*.py·"
           "research/logs/ml_forecast/PREREG.md (2단계 진행 중, 미머지)")


def run(study: str, signal: Signal, panels: DataPanels) -> Forecast:
    require_stage(study, STAGE)
    raise NotImplementedError(f"5단계 구현은 아직 main 에 없다: {POINTER}")
