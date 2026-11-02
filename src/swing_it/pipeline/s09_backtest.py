"""9단계 백테스트 — ``strategies.pead.staggered_backtest`` + 다중검정 원장.

이 경로는 **비용을 떼지 않는다**(``cost_basis="gross"``). 비용 1·2배 판정은 ``REFERENCES`` 의 연구 하네스에 있다
(옮기면 바이트 동일 재현이 깨지니 그대로 둔다). pass 에는 검증 문서가 필요하다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from swing_it.diagnostics.trials import count_trials, record_trial
from swing_it.pipeline.contracts import STAGE_CONTRACTS, BacktestResult, DataPanels, Signal, expect
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage
from swing_it.strategies.pead import staggered_backtest

STAGE = Stage.BACKTEST
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "src/swing_it/strategies/pead.py::staggered_backtest",
    "research/experiments/pead_optimizer_b.py::simulate",
    "research/experiments/prop_gate.py::prop_gate",
)


def run(study: str, signal: Signal, panels: DataPanels, *, config: Mapping[str, Any],
        logs_dir: "Path | str | None" = None) -> BacktestResult:
    """``config`` 를 원장에 적고 돌린다. ``config`` 는 ``staggered_backtest`` 인자이고, 시행 수는 원장에서 읽는다."""
    require_stage(study, STAGE)
    expect(signal, Signal, "signal")
    expect(panels, DataPanels, "panels")
    record_trial(study, dict(config), note="pipeline s09", logs_dir=logs_dir)
    args = {"delisting_exit": True, **config}
    periods, summary = staggered_backtest(panels.prices, None, signal_panel=signal.frame, **args)
    return BacktestResult(label=study, periods=periods, summary=summary, cost_basis="gross",
                          n_trials=count_trials(study, logs_dir=logs_dir), config=dict(config))
