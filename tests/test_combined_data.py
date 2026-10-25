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
