import os

import numpy as np
import pandas as pd
import pytest

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
    raw2, fb = cd.fill_raw_with_adj(raw, C, pd.date_range("2020-01-01", periods=t), np.zeros(n, int))
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


# ---------------------------------------------------------------- 정정 0 C2: 대체 원주가를 호가 격자에 맞춘다

PRE = pd.bdate_range("2015-01-05", periods=8)        # 옛 호가표 구간


def _tickp(raw, days, mkt):
    from research.combined.gaps import gap_arrays
    n, t = raw.shape
    return gap_arrays(raw, raw, raw, np.zeros((n, t), bool), np.full(t, 1.29), days, mkt)["tickp"]


def _offgrid(tp):
    inv = 1.0 / tp.astype(float)
    return np.isfinite(inv) & (np.abs(inv - np.round(inv)) > 1e-2)


def test_snap_to_tick_lands_on_own_grid_across_bands_and_reform():
    from research.combined.gaps import tick_dated_vec
    rng = np.random.default_rng(0)
    days = pd.DatetimeIndex(["2015-03-02", "2023-01-24", "2023-01-25", "2024-06-03"])
    p = np.exp(rng.uniform(np.log(50), np.log(2e6), (400, len(days))))
    edges = np.array([999.6, 1000.4, 1999.7, 4999.8, 5002.0, 9999.0, 19999.0, 49990.0, 50010.0, 99990.0, 199990.0, 499900.0, 500100.0])
    p[:len(edges), :] = edges[:, None]
    mkt = (np.arange(400) % 2).astype(int)
    q = cd.snap_to_tick(p, days, mkt)
    tk = tick_dated_vec(q, days, mkt)
    assert np.all(np.abs(q / tk - np.round(q / tk)) < 1e-9)                 # 자기 틱의 격자 위(구간 경계를 넘어도)
    assert np.all(np.abs(q - p) <= tick_dated_vec(p, days, mkt) / 2 + 1e-9)  # 반 틱 안에서만 움직인다
    assert cd.snap_to_tick(np.array([[0.3]]), days[:1], np.zeros(1, int))[0, 0] == 1.0   # 최소 1틱
    assert np.isnan(cd.snap_to_tick(np.array([[np.nan]]), days[:1], np.zeros(1, int))[0, 0])


def test_adj_fill_on_grid_kills_offgrid_delisting_flag():
    """폐지 종목은 원주가가 없어 수정 종가(격자 밖)로 채워졌다 → |1/tickp − 정수| 가 폐지 표지(0.501 vs 0.001). 격자에 맞춘 대체는 그 차이를 없앤다."""
    n, t = 40, len(PRE)
    rng = np.random.default_rng(1)
    raw_alive = np.round(rng.uniform(2_000, 40_000, (n // 2, t)) / 50) * 50          # 생존: 실제 원주가(격자 위)
    adj_dead = rng.uniform(2_000, 40_000, (n // 2, t)) * 0.873                       # 폐지: 수정 종가(격자 밖)
    raw = np.vstack([raw_alive, np.full_like(adj_dead, np.nan)])
    C = np.vstack([raw_alive, adj_dead])
    mkt, dl = np.zeros(n, int), np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    uni = np.ones((n, t), bool)
    unsnapped = np.where(np.isfinite(raw), raw, C)                                   # 옛 대체(위반 주입)
    with pytest.raises(ValueError, match="offgrid"):
        cd.check_tickp_leak(_tickp(unsnapped, PRE, mkt), uni, dl)
    filled, fb = cd.fill_raw_with_adj(raw, C, PRE, mkt)
    tp = _tickp(filled, PRE, mkt)
    assert fb[n // 2:].all() and not _offgrid(tp[:, 1:]).any()
    out = cd.check_tickp_leak(tp, uni, dl, tol={"ks": 1.0})                          # 격자 검사만(장난감 분포 차는 별도 시험)
    assert out["offgrid_delisted"] == 0.0 and out["offgrid_alive"] == 0.0


def test_tickp_leak_check_each_measure_fires():
    n, t = 60, 30
    rng = np.random.default_rng(2)
    tp = (1.0 / rng.integers(200, 2_000, (n, t))).astype(np.float32)                  # 격자 위 tickp(1/정수)
    dl = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    uni = np.ones((n, t), bool)
    ok = cd.check_tickp_leak(tp, uni, dl)
    assert ok["ks"] < cd.TICKP_LEAK_TOL["ks"] and ok["nan_delisted"] == ok["offgrid_delisted"] == 0.0
    bad = tp.copy()
    bad[n // 2:, :3] = np.nan                                                          # NaN 몫 10% 차
    with pytest.raises(ValueError, match="nan"):
        cd.check_tickp_leak(bad, uni, dl)
    bad = tp.copy()
    bad[n // 2:, :3] = 1.0 / 333.3                                                     # 격자 밖 10%
    with pytest.raises(ValueError, match="offgrid"):
        cd.check_tickp_leak(bad, uni, dl)
    bad = tp.copy()
    bad[n // 2:] = (1.0 / rng.integers(20, 200, (n // 2, t))).astype(np.float32)       # 격자 위지만 분포가 다르다(10배 큰 tickp)
    with pytest.raises(ValueError, match="KS"):
        cd.check_tickp_leak(bad, uni, dl)
    assert cd.check_tickp_leak(bad, np.zeros((n, t), bool) | ~dl[:, None], dl)["ks"] == 0.0   # uni 밖 칸은 안 본다


# ---------------------------------------------------------------- 정정 0 M4·M6a: 잠금 절반 디스크 캐시, 전진 빌더 끝 날짜

def _half_toy():
    from research.autoresearch.prepare import half_of
    codes = [f"{i:06d}" for i in range(20)]
    half = np.array([half_of(c) for c in codes], np.int8)
    assert set(half) == {0, 1}
    t = 5
    return {"codes": codes, "days": pd.bdate_range("2020-01-01", periods=t), "C": np.arange(20 * t, dtype=float).reshape(20, t),
            "half": half, "mkt": np.zeros(20, np.int64), "mkt_real": np.zeros(20, np.int8), "era": "Z"}


def test_locked_half_written_to_separate_dir_and_sha_checked(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(cd, "CACHE", str(tmp_path / "combined"))
    monkeypatch.setattr(cd, "LOCKED_CACHE", str(tmp_path / "combined_locked"))
    F = _half_toy()
    cd.write_halves(F, "Z")
    p0, p1 = cd.cache_path("Z", 0), cd.cache_path("Z", 1)
    assert os.path.dirname(p0) == cd.CACHE and os.path.dirname(p1) == cd.LOCKED_CACHE      # 루프 격리는 CACHE 만 다시 보인다
    s0, s1 = cd.load("Z", 0), cd.load("Z", 1, sha=hashlib.sha256(open(p1, "rb").read()).hexdigest())
    assert set(s0["codes"]).isdisjoint(s1["codes"]) and sorted(s0["codes"] + s1["codes"]) == F["codes"]
    assert (s1["half"] == 1).all() and s1["mkt_real"].shape == (len(s1["codes"]),)
    with pytest.raises(SystemExit, match="sha"):
        cd.load("Z", 1, sha="0" * 64)


def test_build_era_end_override_rolls_only_end(monkeypatch):
    import research.experiments.indep30 as I30
    seen = {}

    class Stop(Exception):
        pass

    def fake(con, data, eps):
        seen.update(data=data, eps=eps)
        raise Stop
    monkeypatch.setattr(I30, "build", fake)
    with pytest.raises(Stop):
        cd.build_era(None, "C", end="2026-10-23")
    assert seen == {"data": ("2016-09-01", "2026-10-23"), "eps": "earnings"}
    with pytest.raises(Stop):
        cd.build_era(None, "C")
    assert seen["data"] == cd.ERAS_C["C"]["data"]


def test_news_exclusion_single_switch(tmp_path, monkeypatch):
    import json
    (tmp_path / "000001.jsonl").write_text(json.dumps({"id": 1, "title": "a", "createdAt": "2023-03-16T09:00:00"}) + "\n")
    days = pd.bdate_range("2023-03-01", periods=30)
    uni = np.ones((2, 30), bool)
    u2, det = cd.apply_news_exclusion(uni, ["000001", "999999"], days, tmp_path)
    w = np.asarray(days >= "2023-03-15")
    assert det["enabled"] and det["nofile"] == ["999999"] and det["uni_off_cells"] == int(w.sum())
    assert u2[0].all() and u2[1, ~w].all() and not u2[1, w].any()
    monkeypatch.setattr(cd, "NEWS_EXCLUDE", False)                                   # 사용자가 뉴스 부품을 빼면 끄는 스위치 하나
    u3, det = cd.apply_news_exclusion(uni, ["000001", "999999"], days, tmp_path)
    assert det == {"enabled": False} and u3.all()
