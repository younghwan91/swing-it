import numpy as np
import pandas as pd

from research.combined import data as cd


def test_eras_shape():
    A, B, C = cd.ERAS_C["A"], cd.ERAS_C["B"], cd.ERAS_C["C"]
    assert len(A["folds"]) == 5 and A["folds"][-1][2] == "2016-01-01"
    assert len(B["folds"]) == 5 and B["folds"][-1] == ("2022-01-01", "2024-01-01", "2025-01-01")
    assert B["data"][1] == "2024-12-31"
    assert C["folds"] == [("2023-01-01", "2025-01-02", "2026-09-12")]
    assert C["eps"] == "earnings" and C["data"] == ("2016-09-01", "2026-09-11")


def test_view_keys_have_no_dates_or_codes():
    n, t = 3, 4
    D = {k: np.ones((n, t)) for k in cd.VIEW_KEYS_C}
    D.update(kx=np.ones(t), kq=np.ones(t), regime_on=np.ones(t, bool),
             days=pd.date_range("2020-01-01", periods=t), codes=["000001"] * n)
    V = cd.view_c(D)
    assert set(V) == set(cd.VIEW_KEYS_C) | {"kx", "kq", "regime_on"}
    assert all(isinstance(v, np.ndarray) for v in V.values())
    V["gap"][0, 0] = 99
    assert D["gap"][0, 0] == 1


def test_repo_keys_cover_new_arrays():
    new = set(cd.VIEW_KEYS_C) - set(cd.VIEW_KEYS_V4)
    assert new == {k for v in cd.REPO_KEYS.values() for k in v}


def test_raw_fill_removes_delisting_flag_and_no_flag_key():
    import pytest
    from research.combined.gaps import gap_arrays
    n, t = 2, 6
    C = np.full((n, t), 1000.0)
    raw = C.copy()
    raw[1] = np.nan                                   # 폐지 종목: 원주가 전부 결측
    raw2, fb = cd.fill_raw_with_adj(raw, C)
    assert fb[1].all() and not fb[0].any()
    g = gap_arrays(C, C, raw2, np.zeros((n, t), bool), np.full(t, 1.29), pd.date_range("2020-01-01", periods=t), np.zeros(n, int))
    assert np.isfinite(g["tickp"][1, 1:]).all()
    uni = np.ones((n, t), bool)
    cd.check_tickp_leak(g["tickp"], uni, np.array([False, True]))
    bad = g["tickp"].copy()
    bad[1] = np.nan
    with pytest.raises(ValueError):
        cd.check_tickp_leak(bad, uni, np.array([False, True]))
    assert not [k for k in cd.VIEW_KEYS_C if "fallback" in k or "src" in k or "miss" in k]


def test_market_mapping_explicit():
    import pytest
    m = pd.Series({"A": "코스닥", "B": "거래소", "C": "유가증권", "D": "코넥스"})
    arr, miss = cd.market_codes(m, ["A", "B", "C", "D", "E"])
    assert arr.tolist() == [1, 0, 0, 1, 0] and miss == ["E"]
    with pytest.raises(ValueError):
        cd.market_codes(pd.Series({"A": "ETF"}), ["A"])
