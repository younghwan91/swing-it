"""Panel construction helpers + session-scoped caching for the backtest engine.

Long (``code``/``date``/value) frames pivot to ``code × date`` numpy panels the
same way everywhere so no experiment re-derives the abs/reindex conventions.
A content-keyed session cache lets a parameter sweep pivot the same DB-loaded
frame once instead of on every run.

The market-generic pivots (``panel_pivot`` · ``forward_returns`` · ``lookup_panel``
· ``adv_panel``) live in ``krx_quant_core.backtest.panels`` and are re-exported
here. PEAD-specific panels (``yoy_panels``/``resolve_signal``) and the session
cache (``PanelCache``/``price_arrays``) stay in swing-it — the consumer process
owns cache state.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from krx_quant_core.backtest.panels import (
    adv_panel,
    forward_returns,
    lookup_panel,
    panel_pivot,
)

__all__ = [
    "PANEL_CACHE",
    "PanelCache",
    "PriceArrays",
    "adv_panel",
    "cached_panel_pivot",
    "forward_returns",
    "lookup_panel",
    "panel_pivot",
    "price_arrays",
    "resolve_signal",
    "yoy_panels",
]


def yoy_panels(earnings_panel, codes, dates):
    """Pivot the long earnings panel to code×date ``yoy`` and ``age_days`` arrays."""
    yoy = (
        earnings_panel.pivot_table(index="code", columns="date", values="yoy", aggfunc="first")
        .reindex(index=codes, columns=dates).to_numpy(float)
    )
    if "age_days" in earnings_panel.columns:
        age = (
            earnings_panel.pivot_table(index="code", columns="date", values="age_days", aggfunc="first")
            .reindex(index=codes, columns=dates).to_numpy(float)
        )
    else:
        age = np.full_like(yoy, np.nan)
    return yoy, age


def resolve_signal(earnings_panel, signal_panel, codes, dates):
    """Return ``(sig, age)`` code×date arrays from the YoY panel or a precomputed
    ``signal_panel`` (long ``code``/``date``/``signal`` — e.g. a PEAD+value blend
    from :func:`swing_it.features.fundamentals.blend_rank`). Freshness (``age``)
    only applies to the raw YoY path; it is ``NaN`` for a precomputed signal.
    """
    if signal_panel is not None:
        sig = (
            signal_panel.pivot_table(index="code", columns="date", values="signal", aggfunc="first")
            .reindex(index=codes, columns=dates).to_numpy(float)
        )
        return sig, np.full_like(sig, np.nan)
    return yoy_panels(earnings_panel, codes, dates)


# --- Session-scoped LRU cache for panel construction ---------------------------
#
# DataFrames are unhashable, so functools.lru_cache can't wrap panel_pivot
# directly. This is a small content-keyed LRU: a parameter sweep that repeatedly
# pivots the same DB-loaded frame (same content) hits the cache instead of
# re-pivoting. Keyed on the value column + shape + a stable content digest so two
# equal-content-but-distinct frame objects share an entry.


def _panel_cache_key(prices: pd.DataFrame, value: str) -> tuple:
    sub = prices[["code", "date", value]]
    digest = hashlib.sha1(
        pd.util.hash_pandas_object(sub, index=False).to_numpy().tobytes()
    ).hexdigest()
    return (value, sub.shape, digest)


class PanelCache:
    """A tiny content-keyed LRU cache for :func:`panel_pivot` results."""

    #: 기본 4 = (close, trade_value) x 서로 다른 prices 프레임 2개. 스윕은 같은
    #: 프레임을 재사용하므로 이걸로 캐시 이득은 그대로 나온다. 전 종목 code x date
    #: float64 피벗 하나가 ~50MB 라, 이 상한이 곧 상주 메모리 상한이다 — 모든 전략
    #: 진입점이 이 캐시를 타게 된 뒤로는 상한이 넉넉할수록 안전한 게 아니라 위험하다.
    def __init__(self, maxsize: int = 4) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def panel_pivot(self, prices: pd.DataFrame, value: str) -> pd.DataFrame:
        key = _panel_cache_key(prices, value)
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        result = panel_pivot(prices, value)
        self._store[key] = result
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)  # evict least-recently-used
        return result

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0


# Module-level session cache. Import and call ``PANEL_CACHE.panel_pivot(...)`` to
# share pivots across a sweep; call ``PANEL_CACHE.clear()`` between sessions.
PANEL_CACHE = PanelCache()


def cached_panel_pivot(prices: pd.DataFrame, value: str) -> pd.DataFrame:
    """Session-scoped cached wrapper over :func:`panel_pivot` (see :data:`PANEL_CACHE`)."""
    return PANEL_CACHE.panel_pivot(prices, value)


@dataclass(frozen=True)
class PriceArrays:
    """close/trade_value numpy panels plus their shared axes.

    A plain struct, not a closure — it holds only the four fields it needs, so
    it never pins an enclosing scope's frames alive.
    """

    C: np.ndarray          # close, codes x dates
    V: np.ndarray          # trade_value, aligned to C
    codes: list
    dates: list


def price_arrays(prices: pd.DataFrame, *, cache: bool = True) -> PriceArrays:
    """The close/trade_value prelude every backtest entry point opens with.

    Nine copies of these six lines had drifted apart on which frame supplied the
    axes; one helper means one convention. ``cache`` routes the two pivots
    through :data:`PANEL_CACHE`, so a parameter sweep over a fixed ``prices``
    pivots once instead of once per call.

    Both arrays own their data. :meth:`pandas.DataFrame.to_numpy` can return a
    view of the frame it came from, so handing back an un-copied array would let
    one backtest's write corrupt the next one's prices through the shared cache.
    ``V`` happens to be materialised by the reindex today, but callers must not
    have to know which of the two is safe — so the rule is uniform.
    """
    pivot = cached_panel_pivot if cache else panel_pivot
    close = pivot(prices, "close")
    tval = pivot(prices, "trade_value")
    dates = list(close.columns)
    codes = list(close.index)
    C = close.to_numpy(float).copy()
    V = tval.reindex(index=codes, columns=dates).to_numpy(float).copy()
    return PriceArrays(C, V, codes, dates)
