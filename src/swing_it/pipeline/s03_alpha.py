"""3단계 알파 발굴 — 현재 등록된 신호는 분기 순이익 YoY(PEAD) 하나다. 새 신호는 같은 :class:`Signal` 을 내면 된다."""

from __future__ import annotations

from swing_it.features.fundamentals import _yoy_vec, earnings_yoy_panel
from swing_it.pipeline.contracts import STAGE_CONTRACTS, DataPanels, Signal, expect
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.ALPHA
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "src/swing_it/features/fundamentals.py::earnings_yoy_panel",
    "research/experiments/pead_optimizer_b.py::tranche_names",
)


def run(study: str, panels: DataPanels, *, name: str = "pead_yoy") -> Signal:
    """지식 시점 as-of 로 YoY 패널을 만들고, 유니버스 밖은 NaN 으로 둔다(PO-1.4 패리티와 같은 방식)."""
    require_stage(study, STAGE)
    expect(panels, DataPanels, "panels")
    ea = panels.earnings.assign(yoy=_yoy_vec(panels.earnings["netinc"], panels.earnings["netinc_prior"]))
    long = earnings_yoy_panel(ea.dropna(subset=["yoy"]), panels.trading_dates(), knowledge_col=panels.knowledge_col)
    frame = long.rename(columns={"yoy": "signal"})[["code", "date", "signal"]].merge(
        panels.universe.frame[["code", "date", "eligible"]], on=["code", "date"], how="left")
    frame["signal"] = frame["signal"].where(frame["eligible"].eq(True))
    return Signal(frame=frame[["code", "date", "signal"]], name=name,
                  known_at=f"avail_date·{panels.knowledge_col} 가 그날 이하인 값만")
