"""research/experiments/sepa_momentum 의 순수 헬퍼 — 합성 데이터(DB 없음).

- 12-1 모멘텀이 t−21 종가까지만 본다(미래·최근 1개월을 바꿔도 값이 같다).
- 날짜 × 시장 분위·중앙값 2분위가 시장별로, 유니버스 마스크 안에서만 잘린다.
- 주도주 마스크, 층 후보 수(자기 종목 제외·NaN 층 = 0).
- core matched_control 의 얇은 층 동작(후보 < n_per 면 복원 추출로 반복) — 리포트 서술의 근거.
- 방향 가족이 정확히 12칸.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from krx_quant_core.stats.matched_null import matched_control

from research.experiments.sepa_explore import build_cells
from research.experiments.sepa_momentum import (
    candidate_counts,
    day_market_median_half,
    day_market_quantiles,
    is_family,
    leader_mask,
    momentum_12_1,
)


def _close(n_days=300, n_codes=4, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_days)
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(n_days, n_codes)), axis=0))
    return pd.DataFrame(x, index=days, columns=[f"C{i}" for i in range(n_codes)])


def test_momentum_formula_and_warmup():
    c = _close()
    m = momentum_12_1(c)
    t = 280
    exp = c.iloc[t - 21] / c.iloc[t - 252] - 1.0
    assert np.allclose(m.iloc[t].to_numpy(), exp.to_numpy())
    assert m.iloc[:252].isna().all().all()          # t−252 행이 없으면 NaN
    assert m.iloc[252].notna().all()


def test_momentum_no_lookahead():
    c = _close()
    m = momentum_12_1(c)
    t = 270
    c2 = c.copy()
    c2.iloc[t - 20:] *= 3.0                         # 최근 20일 + t + 미래를 바꾼다
    m2 = momentum_12_1(c2)
    assert np.allclose(m.iloc[: t + 1].to_numpy(), m2.iloc[: t + 1].to_numpy(), equal_nan=True)
    c3 = c.copy()
    c3.iloc[t - 21] *= 2.0                          # t−21 은 쓴다
    assert not np.allclose(momentum_12_1(c3).iloc[t].to_numpy(), m.iloc[t].to_numpy())


def test_momentum_nonpositive_base_is_nan():
    c = _close(n_days=260, n_codes=2)
    c.iloc[0, 0] = 0.0
    assert np.isnan(momentum_12_1(c).iloc[252, 0])


def _two_markets(n_per_mkt=20):
    codes = [f"K{i}" for i in range(n_per_mkt)] + [f"Q{i}" for i in range(n_per_mkt)]
    market = pd.Series(["KOSPI"] * n_per_mkt + ["KOSDAQ"] * n_per_mkt, index=codes)
    days = pd.bdate_range("2021-01-01", periods=3)
    # KOSPI 값 0..19, KOSDAQ 값 100..119 — 시장을 섞어 자르면 KOSPI 가 전부 아래 반이 된다
    v = np.vstack([np.r_[np.arange(n_per_mkt), 100 + np.arange(n_per_mkt)]] * 3).astype(float)
    return pd.DataFrame(v, index=days, columns=codes), market


def test_quantiles_are_per_market_and_masked():
    v, mk = _two_markets()
    mask = pd.DataFrame(True, index=v.index, columns=v.columns)
    q = day_market_quantiles(v, mask, mk, n_q=10)
    row = q.iloc[0]
    assert row["K0"] == 0 and row["K19"] == 9 and row["Q0"] == 0 and row["Q19"] == 9
    assert (row[[f"K{i}" for i in range(20)]].value_counts() == 2).all()     # 20종목 → 10분위 × 2
    mask.loc[:, "K19"] = False
    q2 = day_market_quantiles(v, mask, mk, n_q=10)
    assert np.isnan(q2.iloc[0]["K19"])
    assert q2.iloc[0]["K18"] == 9                                            # 마스크 밖은 경계에서 빠진다


def test_quantiles_too_few_is_nan_and_rowwise():
    v, mk = _two_markets(n_per_mkt=9)
    mask = pd.DataFrame(True, index=v.index, columns=v.columns)
    assert day_market_quantiles(v, mask, mk, n_q=10).isna().all().all()     # 9 < 10
    v2, mk2 = _two_markets()
    mask2 = pd.DataFrame(True, index=v2.index, columns=v2.columns)
    q = day_market_quantiles(v2, mask2, mk2, n_q=10)
    v3 = v2.copy()
    v3.iloc[2] = v3.iloc[2][::-1].to_numpy()                                 # 다른 날을 바꿔도 그날 분위는 그대로
    q3 = day_market_quantiles(v3, mask2, mk2, n_q=10)
    assert q.iloc[:2].equals(q3.iloc[:2])


def test_median_half_per_market_ties_and_mask():
    v, mk = _two_markets(n_per_mkt=5)              # KOSPI 0..4, KOSDAQ 100..104
    mask = pd.DataFrame(True, index=v.index, columns=v.columns)
    h = day_market_median_half(v, mask, mk).iloc[0]
    assert list(h[[f"K{i}" for i in range(5)]]) == [0, 0, 1, 1, 1]            # 중앙값 2 → 2 이상이 1
    assert list(h[[f"Q{i}" for i in range(5)]]) == [0, 0, 1, 1, 1]
    mask.loc[:, ["K3", "K4"]] = False              # KOSPI 유효 0,1,2 → 중앙값 1
    h2 = day_market_median_half(v, mask, mk).iloc[0]
    assert list(h2[["K0", "K1", "K2"]]) == [0, 1, 1] and np.isnan(h2["K3"])
    mask.loc[:, ["K1", "K2"]] = False              # 유효 1종목 → NaN
    assert day_market_median_half(v, mask, mk).iloc[0][[f"K{i}" for i in range(5)]].isna().all()


def test_leader_mask():
    idx, cols = pd.bdate_range("2021-01-01", periods=1), ["A", "B", "C", "D"]
    uni = pd.DataFrame([[True, True, True, False]], index=idx, columns=cols)
    tt = pd.DataFrame([[True, True, False, True]], index=idx, columns=cols)
    rs = pd.DataFrame([[70.0, 69.9, 90.0, 99.0]], index=idx, columns=cols)
    assert list(leader_mask(uni, tt, rs).iloc[0]) == [True, False, False, False]
    rs.iloc[0, 0] = np.nan
    assert not leader_mask(uni, tt, rs).iloc[0, 0]


def test_candidate_counts_excludes_self_and_nan():
    pool = pd.DataFrame({"date": [1, 1, 1, 2], "s": [0, 0, 1, 0]}, index=["a", "b", "c", "d"])
    t = pd.DataFrame({"date": [1, 1, 2, 3, 1], "s": [0, 1, 0, 0, np.nan]}, index=["a", "x", "d", "y", "z"])
    n = candidate_counts(t, pool, ["date", "s"])
    assert n.to_dict() == {"a": 1, "x": 1, "d": 0, "y": 0, "z": 0}


def test_core_thin_stratum_samples_with_replacement():
    pool = pd.DataFrame({"date": [1, 1], "s": [0, 0]}, index=["a", "b"])
    t = pd.DataFrame({"date": [1, 1], "s": [0, 0]}, index=["a", "z"])
    ctl = matched_control(t, pool, strata=["date", "s"], n_per=5, seed=0, exclude_self=True)
    a = ctl[ctl["trade_idx"] == "a"]
    assert len(a) == 5 and set(a["control_idx"]) == {"b"}   # 후보 1개 → 같은 대조 5번
    assert len(ctl[ctl["trade_idx"] == "z"]) == 5
    assert ctl.attrs["unmatched"] == []


def test_family_is_twelve_cells():
    fam = [c for c in build_cells() if is_family(c)]
    assert len(fam) == 12
    assert {c["entry"] for c in fam} == {"breakout"}
    assert {c["gate"] for c in fam} == {"E", "pead≥0.70", "pead≥0.90"}
    assert {c["exit"] for c in fam} == {"⑥ 손절없음+60일", "⑦ 손절없음+120일"}
    assert all(c["stop"] == pytest.approx(1.0) for c in fam)
