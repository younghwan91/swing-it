"""PEAD 최적화 PO-2.1 이음매 가드 — 이음매는 버리고, 사건이 설명하는 조정·판정 불가 셀은 남긴다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from swing_it.book.seam import seam_guard

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


def test_secondary_source_needs_impossible_move_po3():
    """PO-3.3 — supply_demand(두 번째 원천)만 근거면 |수정 수익| > 0.3 일 때만 이음매. 첫 원천 규칙은 그대로."""
    adj = np.array([
        [16950.0, 16950.0, 14645.0, 14700.0, 14800.0],  # 221610 형: 재개일 −13.6% 는 실제 움직임
        [1780.0, 1780.0, 29179.0, 29300.0, 29000.0],    # 두 번째 원천만 있는데 +1539% — 이음매
        [1000.0, 1000.0, 1300.0, 1310.0, 1320.0],       # 첫 원천 근거 +30% 대 원 +1% — 이음매(그대로)
    ])
    src1 = np.array([[np.nan] * 5, [np.nan] * 5, [1000.0, 1000.0, 1010.0, 1020.0, 1030.0]])
    src2 = np.array([
        [339.0, 339.0, 14650.0, 14700.0, 14800.0],      # 정지 중 339 에 멈췄다가 재개일에 따라잡는다(원 +42.2)
        [1780.0, 1780.0, 1810.0, 1817.0, 1798.0],
        [np.nan] * 5,
    ])
    ev = pd.DataFrame({"code": [], "date": []})
    codes = np.array(["221610", "025620", "X"])
    _, _, old, _ = seam_guard(codes, DATES, _ret(adj), adj, [src1, src2], ev)
    assert old[0, 2] and old[1, 2] and old[2, 2]  # PO-2 동작: 셋 다 이음매
    rf, _, seam, info = seam_guard(codes, DATES, _ret(adj), adj, [src1, src2], ev, secondary_min_abs_ret=0.3)
    assert not seam[0, 2] and np.isclose(rf[0, 2], 14645.0 / 16950.0 - 1.0)
    assert seam[1, 2] and seam[2, 2] and seam.sum() == 2
    assert info["seam_cells_by_source"] == [1, 1] and info["candidates_by_source"] == [1, 2]
    assert info["kept_secondary_undeterminable"] == 1 and info["kept_secondary_abs_ret_gt_0.1"] == 1
