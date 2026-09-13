"""Walk-forward validation — re-export of ``krx_quant_core.stats.cv`` + frozen ``FOLDS``.

Folds, no-lookahead slicing, purge/embargo (AFML §7.4) and fold-consistency now
live in krx-quant-core (numerically identical). What stays here is the research
discipline, not the math:

1. **Folds are a frozen default, not a per-experiment knob.** ``FOLDS`` is produced
   once by ``rolling_folds()`` (core defaults equal the historical 6-fold) and
   shared. Handing each experiment a mutable fold config invites fold-shopping —
   silently widening/shifting windows until the OOS number looks good.
2. **TRAIN-only fit, OOS-only eval.** ``walk_forward`` fits on ``[train_lo, train_hi)``
   and evaluates on ``[test_lo, test_hi)``; ``train_hi <= test_lo``.
"""

from __future__ import annotations

from krx_quant_core.stats.cv import (
    Fold,
    FoldMask,
    Simulate,
    _expectancy,
    entry_mask,
    fold_consistency,
    fold_slices,
    oos_fixed,
    purge_embargo,
    rdist,
    resolve_exit_dates,
    rolling_folds,
    slice_by_entry,
    walk_forward,
)

__all__ = [
    "FOLDS",
    "Fold",
    "FoldMask",
    "Simulate",
    "_expectancy",
    "entry_mask",
    "fold_consistency",
    "fold_slices",
    "oos_fixed",
    "purge_embargo",
    "rdist",
    "resolve_exit_dates",
    "rolling_folds",
    "slice_by_entry",
    "walk_forward",
]

# frozen 기본 fold 세트(fold-shopping 방지). 실험마다 새로 만들지 말 것.
FOLDS: "tuple[Fold, ...]" = rolling_folds()
