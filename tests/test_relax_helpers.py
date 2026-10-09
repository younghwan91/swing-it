"""research/experiments/relax_run 의 순수 헬퍼 — 합성 데이터(DB 없음).

- 하나씩 빼기 칸: F 없음판 9칸·F 포함판 10칸, 각 칸이 기준에서 조건 하나만 다르다.
- 사건 마스크: 조건별 켜고 끄기, NaN 은 켜진 조건을 통과하지 않음, 돌파·거래량 성분 분해.
- 추격 금지 사후 거르기 = 시뮬레이터 식.
- 층: 띠 안에서 따로 자른 분위·중앙값 2분위, 층 키 정수화.
- 날짜군집 비율 CI·차이 부트스트랩, 경험적 베이즈 축소(닫힌 꼴과 대조), BO 목적값 재계산, 판단 규칙 산술.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments.relax_run import (
    BASE,
    BASE_NAME,
    CONDS,
    FAM_F,
    FAM_NOF,
    SPACE,
    band_strata,
    bo_config,
    bo_objective_value,
    cell_ledger_config,
    chase_ok,
    decision_numbers,
    diff_boot,
    eb_shrink,
    eb_tau2,
    event_mask,
    loo_families,
    purge_train,
    ratio_ci,
    stratum_id,
)
from swing_it.diagnostics.trials import config_fingerprint
from swing_it.validation.optimization import make_objective


# --- 칸 ---

def test_loo_families_have_design_cell_counts_and_differ_by_one_condition():
    fams = loo_families()
    nof, f = fams[FAM_NOF], fams[FAM_F]
    assert len(nof) == 9 and len(f) == 10
    assert list(nof)[0] == BASE_NAME and list(f)[0] == BASE_NAME
    assert nof[BASE_NAME] == BASE
    assert f[BASE_NAME]["flow"] is True and f[BASE_NAME]["event_lo"] == "2024-02-19"
    for fam in (nof, f):
        base = fam[BASE_NAME]
        for name, c in fam.items():
            if name == BASE_NAME:
                continue
            diff = {k for k in base if base[k] != c[k]}
            assert len(diff) == 1, (name, diff)
    # F 포함판의 '− 수급 F' 는 F 없음 기준을 2024-02-19+ 로 자른 것
    assert f["− 수급 F=1"] == {**BASE, "event_lo": "2024-02-19"}
    assert "− 수급 F=1" not in nof
    assert [k for k, _, _ in CONDS] == ["tt", "rs", "breakout", "vol", "breadth", "pead", "rel", "chase", "flow"]


def test_loo_debug_window_clips_and_fingerprints_are_distinct():
    fams = loo_families("2021-07-01", "2022-06-30")
    assert fams[FAM_NOF][BASE_NAME]["event_lo"] == "2021-07-01"
    assert fams[FAM_F][BASE_NAME]["event_lo"] == "2024-02-19"     # 디버그 하한이 더 이르면 F 하한 유지
    assert fams[FAM_NOF][BASE_NAME]["event_hi"] == "2022-06-30"
    fps = {config_fingerprint(cell_ledger_config(c, fam, n)) for fam, cells in loo_families().items()
           for n, c in cells.items()}
    assert len(fps) == 19


def test_bo_config_overrides_only_space_keys():
    params = {"rs_min": 55.5, "vol_mult": 1.1, "pead_min": 0.2, "rel_max": 2.5, "breadth_min": 0.1, "pivot_window": 33}
    c = bo_config(params)
    assert set(params) == set(SPACE)
    for k, v in params.items():
        assert c[k] == v
    for k in ("tt", "breakout", "chase", "flow", "event_lo", "event_hi"):
        assert c[k] == BASE[k]
    assert isinstance(c["pivot_window"], int)


# --- 사건 마스크 ---

def _cols(n=8):
    return {
        "event_day": pd.to_datetime(["2020-01-02"] * n).to_numpy(),
        "tt": np.ones(n, bool), "rs": np.full(n, 80.0), "close": np.full(n, 11.0), "pivot": np.full(n, 10.0),
        "volume": np.full(n, 200.0), "vol_avg": np.full(n, 100.0), "breadth": np.full(n, 0.6),
        "pead_rank": np.full(n, 0.8), "rel": np.full(n, 0.9), "F": np.ones(n),
    }


def test_event_mask_each_condition_and_nan_fails_only_when_on():
    cfg = {**BASE, "flow": True, "event_lo": "2019-01-01", "event_hi": "2021-01-01"}
    cols = _cols()
    assert event_mask(cols, cfg).all()
    breakers = {"tt": ("tt", False), "rs": ("rs", 69.9), "breakout": ("close", 10.0), "vol": ("volume", 149.0),
                "breadth": ("breadth", 0.5), "pead": ("pead_rank", 0.69), "rel": ("rel", 1.01), "flow": ("F", 0.0)}
    for i, (_, (col, val)) in enumerate(breakers.items()):
        cols[col] = cols[col].copy()
        cols[col][i] = val
    mk = event_mask(cols, cfg)
    assert not mk[:8].any()
    # 조건을 끄면 그 행만 되살아난다
    offs = {"tt": {"tt": False}, "rs": {"rs_min": None}, "breakout": {"breakout": False}, "vol": {"vol_mult": None},
            "breadth": {"breadth_min": None}, "pead": {"pead_min": None}, "rel": {"rel_max": None}, "flow": {"flow": False}}
    for i, k in enumerate(breakers):
        assert np.flatnonzero(event_mask(cols, {**cfg, **offs[k]})).tolist() == [i], k
    # NaN: 켜진 조건은 통과 못 함, 꺼지면 통과
    c2 = _cols(3)
    c2["rel"] = np.array([np.nan, 0.5, 2.0])
    c2["pead_rank"] = np.array([0.9, np.nan, 0.9])
    assert event_mask(c2, cfg).tolist() == [False, False, False]
    assert event_mask(c2, {**cfg, "rel_max": None}).tolist() == [True, False, True]
    assert event_mask(c2, {**cfg, "rel_max": 3.0, "pead_min": 0.0}).tolist() == [False, False, True]
    # 구간
    assert not event_mask(_cols(2), {**cfg, "event_lo": "2020-01-03"}).any()


def test_event_mask_breakout_parts_match_pivot_breakout():
    from swing_it.features.minervini import pivot_breakout, pivot_level

    rng = np.random.default_rng(1)
    d = pd.bdate_range("2020-01-01", periods=200)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (200, 3)), axis=0)), index=d, columns=list("abc"))
    high = close * (1 + rng.uniform(0, 0.02, close.shape))
    vol = pd.DataFrame(rng.integers(0, 1000, close.shape).astype(float), index=d, columns=close.columns)
    w = 30
    brk, _ = pivot_breakout(high, close, vol, window=w, vol_mult=1.5)
    piv = pivot_level(high, w)
    vavg = vol.shift(1).rolling(w, min_periods=w).mean()
    n = close.size
    cols = {"event_day": np.repeat(d.to_numpy(), 3), "tt": np.ones(n, bool), "rs": np.full(n, 99.0),
            "close": close.to_numpy().ravel(), "pivot": piv.to_numpy().ravel(), "volume": vol.to_numpy().ravel(),
            "vol_avg": vavg.to_numpy().ravel(), "breadth": np.ones(n), "pead_rank": np.ones(n), "rel": np.zeros(n),
            "F": np.ones(n)}
    cfg = {**BASE, "event_lo": "2000-01-01", "event_hi": "2100-01-01", "vol_mult": 1.5}
    assert (event_mask(cols, cfg) == brk.to_numpy().ravel()).all()
    assert brk.to_numpy().sum() > 0


def test_chase_ok_matches_simulator_rule():
    e = np.array([105.0, 105.01, 90.0, 120.0])
    pv = np.array([100.0, 100.0, np.nan, np.nan])
    assert chase_ok(e, pv, 0.05).tolist() == [True, False, True, True]
    assert chase_ok(e, pv, None).all()


# --- 층 ---

def test_band_strata_cuts_within_band_and_market():
    d = pd.bdate_range("2021-01-01", periods=2)
    cols = [f"c{i}" for i in range(20)]
    vals = pd.DataFrame(np.tile(np.arange(20, dtype=float), (2, 1)), index=d, columns=cols)
    market = pd.Series(["KOSPI"] * 10 + ["KOSDAQ"] * 10, index=cols)
    b1 = pd.DataFrame(False, index=d, columns=cols)
    b1[cols[:5] + cols[10:15]] = True                 # 띠 1: 시장마다 낮은 5개
    b0 = ~b1
    q = band_strata(vals, {1: b1, 0: b0}, market, n_q=5)
    # 각 띠·시장 안 5종목 → 0..4
    assert q.loc[d[0], cols[:5]].tolist() == [0, 1, 2, 3, 4]
    assert q.loc[d[0], cols[5:10]].tolist() == [0, 1, 2, 3, 4]
    assert q.loc[d[0], cols[10:15]].tolist() == [0, 1, 2, 3, 4]
    h = band_strata(vals, {1: b1, 0: b0}, market, n_q=None)
    assert h.loc[d[0], cols[:5]].tolist() == [0, 0, 1, 1, 1]       # 중앙값(2) 이상 → 1
    assert h.loc[d[0], cols[5:10]].tolist() == [0, 0, 1, 1, 1]
    # 띠 밖은 NaN
    none = band_strata(vals, {1: b1}, market, n_q=5)
    assert none.loc[:, cols[5:10]].isna().all().all()


def test_stratum_id_unique_and_missing_is_minus_one():
    import itertools

    combos = list(itertools.product(range(3), ["KOSPI", "KOSDAQ"], [0, 1], range(5), [0, 1], range(3)))
    a = np.array(combos, dtype=object)
    sid = stratum_id(a[:, 0].astype(int), a[:, 1], a[:, 2].astype(float), a[:, 3].astype(float),
                     a[:, 4].astype(float), a[:, 5].astype(float))
    assert len(set(sid.tolist())) == len(combos) and (sid >= 0).all()
    bad = stratum_id([0, 0, 0], ["KOSPI", "X", "KOSDAQ"], [1.0, 1.0, np.nan], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                     [2.0, 2.0, 2.0])
    assert bad[0] >= 0 and bad[1] == -1 and bad[2] == -1


# --- 부트스트랩 ---

def test_ratio_ci_point_is_sum_over_sum_and_ci_brackets():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(50), 4)
    num = rng.normal(0.01, 0.05, 200)
    den = rng.integers(5, 40, 200).astype(float)
    pt, lo, hi = ratio_ci(num, den, cl)
    assert pt == pytest.approx(num.sum() / den.sum())
    assert lo < pt < hi
    assert np.isnan(ratio_ci([], [], [])[0])


def test_diff_boot_identical_groups_zero_and_shift_detected():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(60), 3)
    x = rng.normal(0, 0.05, 180)
    one = np.ones(180)
    d = diff_boot((x, one, cl), (x, one, cl))
    assert d["point"] == 0 and d["se"] == pytest.approx(0.0, abs=1e-15)
    d2 = diff_boot((x + 0.02, one, cl), (x, one, cl))
    assert d2["point"] == pytest.approx(0.02) and d2["se"] < 1e-12       # 같은 군집을 함께 뽑으면 상수 차이는 분산 0
    d3 = diff_boot((x[:90], one[:90], cl[:90]), (x[90:], one[90:], cl[90:]))
    assert d3["se"] > 0 and d3["lo"] < d3["point"] < d3["hi"]


# --- 경험적 베이즈 ---

def test_eb_tau2_matches_closed_form_with_equal_se():
    rng = np.random.default_rng(3)
    s = 0.01
    delta = rng.normal(0, 0.03, 40)
    want = max(0.0, float(np.mean(delta ** 2)) - s ** 2)       # 사전 평균 0, 같은 s 의 MLE
    assert eb_tau2(delta, np.full(40, s)) == pytest.approx(want, rel=1e-4)


def test_eb_shrink_posterior_formulas_and_probabilities():
    delta = np.array([0.05, -0.02, 0.0, 0.03])
    se = np.array([0.01, 0.02, 0.01, 0.05])
    df = eb_shrink(delta, se)
    t2 = df["tau2"].iloc[0]
    assert t2 > 0
    B = t2 / (t2 + se ** 2)
    assert np.allclose(df["B"], B)
    assert np.allclose(df["post_mean"], B * delta)
    assert np.allclose(df["post_sd"], np.sqrt(B) * se)
    assert (np.abs(df["post_mean"]) <= np.abs(delta) + 1e-15).all()          # 0 쪽으로만
    assert df["p_pos"].iloc[0] > 0.99 and df["p_pos"].iloc[1] < 0.5 and df["p_pos"].iloc[2] == pytest.approx(0.5)
    assert (df["lo90"] <= df["post_mean"]).all() and (df["post_mean"] <= df["hi90"]).all()


def test_eb_shrink_degenerate_when_noise_dominates_and_nan_rows():
    df = eb_shrink([0.001, -0.001, 0.0005], [0.05, 0.05, 0.05])
    assert df["tau2"].iloc[0] == 0.0
    assert df["degenerate"].all() and (df["p_pos"] == 0.5).all() and (df["post_mean"] == 0).all()
    df2 = eb_shrink([0.05, np.nan, 0.02], [0.01, 0.01, 0.0])
    assert np.isfinite(df2["p_pos"].iloc[0]) and np.isnan(df2["p_pos"].iloc[1]) and np.isnan(df2["p_pos"].iloc[2])


# --- BO 목적·판단 규칙 ---

def test_bo_objective_value_equals_make_objective():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2019-01-01", "2023-12-31")
    ed = rng.choice(days, 400).astype("datetime64[D]").astype(str)
    rets = rng.normal(0.01, 0.1, 400)
    rets[:10] = np.nan

    class _T:
        def suggest_float(self, name, lo, hi):
            return lo

        def suggest_int(self, name, lo, hi):
            return lo

        def set_user_attr(self, *a):
            pass

    obj = make_objective(lambda p: (rets, ed), {"x": (0.0, 1.0)}, train_hi="2022-01-01", floor=150)
    v, n = bo_objective_value(rets, ed)
    assert v == obj(_T()) and n == int(((ed < "2022-01-01") & np.isfinite(rets)).sum())
    small, n2 = bo_objective_value(rets[:50], ed[:50])
    assert small == pytest.approx(-1.0 + n2 / 150 * 1e-3) and n2 < 150


def test_purge_train_drops_trades_straddling_boundary_only():
    r = np.array([0.1, 0.2, 0.3, 0.4])
    ed = np.array(["2021-12-01", "2021-12-20", "2022-01-03", "2021-06-01"])
    xd = np.array(["2021-12-30", "2022-01-05", "2022-02-01", "2022-01-01"])
    out = purge_train(r, ed, xd)
    assert np.isnan(out[1]) and np.isnan(out[3])          # 청산 ≥ 경계(같은 날 포함)
    assert out[0] == 0.1 and out[2] == 0.3                 # TRAIN 안 · 경계 뒤 진입은 그대로
    assert r[1] == 0.2                                     # 입력은 안 바뀐다
    v, n = bo_objective_value(out, ed, floor=1)
    assert n == 1


def test_decision_numbers_arithmetic():
    d = decision_numbers({"a": 0.49, "b": 0.5, "c": 0.9, "d": np.nan}, 0.01, 0.002, 0.1)
    assert d["drop"] == ["a"] and d["all_three"]
    d2 = decision_numbers({}, 0.01, -0.001, 0.3)
    assert not d2["all_three"] and not d2["small_pos"]
    assert not decision_numbers({}, 0.01, 0.01, 0.0)["rho_pos"]
    assert not decision_numbers({}, np.nan, 0.01, 0.5)["liq_pos"]
