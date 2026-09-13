"""Backtest performance metrics — thin re-export of ``krx_quant_core.stats.metrics``.

The single source of truth moved to the shared package krx-quant-core (numerically
identical: signatures and defaults unchanged, ``PPY=12`` monthly). scalp-it and
daytrade-it import the same functions, so one return series gives one Sharpe in
every repo. This module stays so ``swing_it.engine.metrics`` imports keep working;
do not re-implement anything here.
"""

from __future__ import annotations

from krx_quant_core.stats.metrics import (
    PPY,
    ann_sharpe,
    cagr,
    max_drawdown,
    newey_west_t,
    paired_bootstrap,
    quantile_summary,
    regime_buckets,
    spearman,
    summarize_periods,
)

__all__ = [
    "PPY",
    "ann_sharpe",
    "cagr",
    "max_drawdown",
    "newey_west_t",
    "paired_bootstrap",
    "quantile_summary",
    "regime_buckets",
    "spearman",
    "summarize_periods",
]
