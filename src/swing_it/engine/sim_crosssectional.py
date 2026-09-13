"""Cross-sectional rank-tilt simulation — re-export of ``krx_quant_core.backtest.crosssectional``.

The loops (entry at ``t+1``, ADV floor, turnover cost, borrow drag, long-only
excess, staggered tranches with ``delisting_exit``) now live in krx-quant-core,
numerically identical to the former copy here. The strategy-level wrappers in
``strategies.pead`` still do the DataFrame→array conversion via ``engine.panels``.

``delisting_exit`` defaults to ``False`` because published figures and
``tests/test_parity_*.py`` are pinned to that path (GUARDRAILS §8).

``cost_one_way`` default (0.0023) is unchanged. For a dated KRX sell-tax schedule
see ``krx_quant_core.costs.round_trip_cost`` — do not change the default here, the
published numbers are tied to it.
"""

from __future__ import annotations

from krx_quant_core.backtest.crosssectional import (
    rank_ic,
    rank_tilt_backtest,
    staggered_tranche_backtest,
)

__all__ = ["rank_ic", "rank_tilt_backtest", "staggered_tranche_backtest"]
