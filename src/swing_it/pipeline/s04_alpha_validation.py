"""4단계 알파 검증(IC/ICIR) — core ``rank_ic`` 를 계약으로 감싼다. 합격 바는 없다. 사람이 읽고 원장에 적는다."""

from __future__ import annotations

import inspect

import numpy as np
from krx_quant_core.backtest.crosssectional import rank_ic

from swing_it.engine.panels import price_arrays, resolve_signal
from swing_it.pipeline.contracts import STAGE_CONTRACTS, DataPanels, ICReport, Signal, expect
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage

STAGE = Stage.ALPHA_VALIDATION
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "src/swing_it/strategies/pead.py::pead_rank_ic",
    "research/experiments/pead_optimizer_b.py::daily_rank_ic",
    "research/experiments/pead_optimizer_b.py::verdict",
)


def run(study: str, signal: Signal, panels: DataPanels, *, horizon: int = 40, adv_floor: float = 5000.0,
        adv_window: int = 20, start_index: int = 130, n_regimes: int = 4) -> ICReport:
    """일별 횡단면 순위 IC·Newey-West t(lag=horizon)·국면 분해. ICIR 은 일별 IC 열을 주는 core 에서만 낸다."""
    require_stage(study, STAGE)
    expect(signal, Signal, "signal")
    expect(panels, DataPanels, "panels")
    pa = price_arrays(panels.prices)
    sig, _ = resolve_signal(None, signal.frame, pa.codes, pa.dates)
    series = {"return_series": True} if "return_series" in inspect.signature(rank_ic).parameters else {}
    r = rank_ic(pa.C, pa.V, sig, pa.dates, horizon=horizon, adv_floor=adv_floor, adv_window=adv_window,
                start_index=start_index, n_regimes=n_regimes, **series)
    icir = None
    if series and r["n_days"] >= 2:
        ic = np.asarray(r["ic"], dtype=np.float64)
        sd = float(ic.std(ddof=1))
        icir = float(ic.mean() / sd) if sd > 0 else None
    return ICReport(signal_name=signal.name, horizon=horizon, ic_mean=float(r["ic_mean"]),
                    ic_nw_t=float(r["ic_nw_t"]), n_days=int(r["n_days"]), frac_positive=float(r["frac_positive"]),
                    regimes=tuple(r["regimes"]), icir=icir)
