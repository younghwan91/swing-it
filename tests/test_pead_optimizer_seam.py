"""PEAD 최적화 PO-2.1 이음매 가드 — 이음매는 버리고, 사건이 설명하는 조정·판정 불가 셀은 남긴다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.pead_optimizer_seam import seam_guard

DATES = pd.DatetimeIndex(["2021-07-27", "2021-07-28", "2021-07-29", "2021-07-30", "2021-08-02"])


def _ret(c):
    r = np.full_like(c, np.nan)
    r[:, 1:] = c[:, 1:] / c[:, :-1] - 1.0
    return r


def test_seam_dropped_split_kept_undeterminable_kept():
    adj = np.array([
        [1780.0, 1780.0, 29179.0, 29300.0, 29000.0],  # 025620 형 이음매(원 종가는 +1.7%)
        [1000.0, 1000.0, 200.0, 202.0, 204.0],        # 5:1 분할 — 사건 있음
        [500.0, 510.0, 900.0, 905.0, 910.0],          # 원 종가 없음 → 판정 불가
    ])
    raw = np.array([
        [1780.0, 1780.0, 1810.0, 1817.0, 1798.0],
        [1000.0, 1000.0, 1000.0, 1010.0, 1020.0],     # 수정 −80%, 원 0% — 그러나 사건이 설명한다
        [np.nan] * 5,
    ])
    ev = pd.DataFrame({"code": ["B"], "date": ["2021-07-29"]})
    ret = _ret(adj)
    rf, cf, seam, info = seam_guard(np.array(["A", "B", "C"]), DATES, ret, adj, [raw], ev)
    assert seam[0, 2] and np.isnan(rf[0, 2])
    assert seam.sum() == 1
    assert info["cells_big_dev_explained_kept"] == 1 and np.isfinite(rf[1, 2])
    assert info["cells_undeterminable"] == 4 and np.isfinite(rf[2, 2])
    # 이음매 뒤 수준은 그 날 수익 0 으로 이어진다 — 다음 날 수익은 그대로
    assert np.isclose(cf[0, 2], cf[0, 1])
    assert np.isclose(cf[0, 3] / cf[0, 2], adj[0, 3] / adj[0, 2])
    assert np.allclose(cf[1:], adj[1:], equal_nan=True)


def test_fallback_source_is_not_mixed_across_days():
    adj = np.array([[100.0, 100.0, 300.0, 300.0, 300.0]])
    src1 = np.array([[100.0, 100.0, np.nan, 101.0, 101.0]])  # d=2 결측 → 이 원천으로는 (1,2) 쌍이 없다
    src2 = np.array([[100.0, 100.0, 101.0, 101.0, 101.0]])
    _, _, seam, info = seam_guard(np.array(["A"]), DATES, _ret(adj), adj, [src1, src2], pd.DataFrame(
        {"code": [], "date": []}))
    assert seam[0, 2] and info["seam_cells"] == 1
