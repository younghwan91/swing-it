"""골든 임박 백테스트 워크포워드·널·바 — 합성 자료(DB 없음). research/signals/golden_bt_wf.py."""
import numpy as np
import pandas as pd
import pytest

from research.signals import golden_bt_engine as E
from research.signals import golden_bt_wf as W

COST = 0.004


def test_menu_48_order_and_fields():
    M = W.menu()
    assert len(M) == 48 and len({tuple(sorted(c.items())) for c in M}) == 48
    assert M[0] == {"th_in": 0.3, "th_out": 0.1, "sizing": "eq", "exit": "dead_p", "th_d": 0.5}
    assert M[-1] == {"th_in": 0.9, "th_out": 0.2, "sizing": "prob", "exit": "ma200", "th_d": None}
    assert all((c["exit"] == "ma200") == (c["th_d"] is None) for c in M)
    assert {c["th_d"] for c in M if c["exit"] == "dead_p"} == {0.5, 0.7}


def test_neighbors_axis_steps():
    M = W.menu()
    nb = W.neighbors(0)
    assert nb[0] == 0
    got = [M[j] for j in nb[1:]]
    # θ_in 0.3 → 0.5 만(0.7 은 두 칸), θ_out 다른 값, sizing 다른 값, exit 다른 두 값
    assert sorted((g["th_in"], g["th_out"], g["sizing"], g["exit"], g["th_d"]) for g in got) == sorted([
        (0.5, 0.1, "eq", "dead_p", 0.5), (0.3, 0.2, "eq", "dead_p", 0.5), (0.3, 0.1, "prob", "dead_p", 0.5),
        (0.3, 0.1, "eq", "dead_p", 0.7), (0.3, 0.1, "eq", "ma200", None)])
    mid = M.index({"th_in": 0.5, "th_out": 0.2, "sizing": "prob", "exit": "ma200", "th_d": None})
    assert len(W.neighbors(mid)) == 1 + 2 + 1 + 1 + 2
    for i in range(48):                                    # 대칭
        for j in W.neighbors(i):
            assert i in W.neighbors(j)


def test_select_plateau_over_spike():
    s = np.zeros(48)
    s[0] = 100.0                                           # 고립 극값(이웃 0)
    mid = 26
    for j in W.neighbors(mid):
        s[j] = 5.0                                         # 고원
    assert W.select(s) == mid
    assert W.select(np.zeros(48)) == 0                     # 동률 → 메뉴 앞
    with pytest.raises(ValueError):
        W.select(np.zeros(47))


def test_fold_windows():
    days = pd.bdate_range("2017-09-01", "2023-12-29")
    f = W.fold_windows(days, (2020, 2021, 2022, 2023), "2017-10-02", "2023-12-28")
    assert [x["year"] for x in f] == [2020, 2021, 2022, 2023]
    a = f[0]
    assert days[a["train"][0]] == pd.Timestamp("2017-10-02") and days[a["train"][1]] == pd.Timestamp("2019-12-31")
    assert days[a["test"][0]] == pd.Timestamp("2020-01-01") and days[a["test"][1]] == pd.Timestamp("2023-12-28")
    assert days[a["entry_end"]] == pd.Timestamp("2020-12-31")
    assert days[f[3]["entry_end"]] == pd.Timestamp("2023-12-28")


# --------------------------------------------------------------------------------------------- 합성 패널

def rand_D(n=8, T=300, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-01-01", periods=T)
    C = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (n, T)), axis=1))
    O = C * np.exp(rng.normal(0, 0.005, (n, T)))  # noqa: E741
    O[:, 1:] = np.minimum(O[:, 1:], C[:, :-1] * 1.2)
    ma50 = pd.DataFrame(C.T).rolling(20, min_periods=1).mean().to_numpy().T
    ma200 = pd.DataFrame(C.T).rolling(60, min_periods=1).mean().to_numpy().T
    pool = (C > ma50) & (rng.random((n, T)) < 0.3)
    PG = rng.random((n, T)).astype(np.float32)
    PD = rng.random((n, T)).astype(np.float32)
    D = {"days": days, "codes": [f"{i:06d}" for i in range(n)], "O": O, "C": C, "ma50": ma50, "ma200": ma200, "pool": pool,
         "PG": PG, "PD": PD, "lu": np.full(T, 1.29), "cost": np.full((2, T), COST), "mkt": np.zeros(n, np.int64)}
    D["last_valid"] = E.last_valid_index(C)
    return D


def test_train_window_truncation():
    """학습 끝 뒤 가격·확률·풀을 바꿔도 학습 점수는 같다(끝 보유분은 끝 종가 평가)."""
    D = rand_D()
    cfg = W.menu()[5]
    end = 200
    a = W.train_score(D, cfg, 1, end)
    D2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in D.items()}
    for k in ("O", "C", "ma50", "ma200", "PG", "PD"):
        D2[k][:, end + 1 :] = D2[k][:, end + 1 :] * 0.3 + 1.0
    D2["pool"][:, end + 1 :] = True
    assert W.train_score(D2, cfg, 1, end) == a
    tr, _ = E.simulate(D, cfg, 2.0, 1, end)
    assert len(tr) > 0 and (pd.to_datetime(tr["exit_day"]) <= D["days"][end]).all()
    assert W.train_score(D, cfg, 1, end, cost_mult=1.0) > a     # 비용 2배가 판정 기본


def test_null1_draws_per_fold():
    T = np.arange(48 * 4, dtype=float).reshape(48, 4)
    x = W.null1_folds(T, n=300, seed=1)
    assert x.shape == (300,)
    rng = np.random.default_rng(1)
    pick = rng.integers(0, 48, size=(300, 4))
    assert np.allclose(x, T[pick, np.arange(4)].sum(axis=1))
    assert np.array_equal(W.null1_folds(T, 300, 1), x)


def test_null2_matches_daily_counts():
    D = rand_D(n=30, seed=2)
    D["O"] = D["C"].copy()
    D["O"][:, 1:] = D["C"][:, :-1] * 1.001                 # 전부 체결 가능(손절선 = MA 아래, 상한가 아님)
    D["ma50"] = D["C"] * 0.9
    D["pool"][:] = False
    D["pool"][:, [10, 50, 90]] = True
    eq_, pr = [(0, 1.0)], [(0, 0.7), (3, 0.9)]
    plan = [(10, [eq_] * 3, 299), (50, [pr] * 5, 299), (90, [eq_], 299)]
    r = W.null2_rep(D, plan, np.array([5, 10]), np.random.default_rng(0))
    assert r["n_fill"] == 9 and r["n_short"] == 0 and np.isfinite(r["R_sum"])
    D["pool"][:, 50] = False
    D["pool"][:2, 50] = True                               # 그 날 풀이 2 뿐 → 모자람 3
    r = W.null2_rep(D, plan, np.array([5]), np.random.default_rng(0))
    assert r["n_fill"] == 6 and r["n_short"] == 3


def test_null2_pg_zero_pool_still_matches_strategy_sizing():
    """GB-2 C1: 풀 종목 P_g 가 전부 0 이어도 무작위 쪽 비중 = 기준 × 짝 전략 거래의 진입 P_g — R 크기가 전략 규칙과 같다."""
    T = 40
    days = pd.bdate_range("2020-01-01", periods=T)
    n = 10
    C = np.full((n, T), 100.0)
    C[:, 12:] = 110.0
    D = {"days": days, "codes": [f"{i:06d}" for i in range(n)], "O": C.copy(), "C": C, "ma50": np.full((n, T), 99.0),
         "ma200": np.full((n, T), 120.0), "pool": np.zeros((n, T), bool), "PG": np.zeros((n, T), np.float32),
         "PD": np.full((n, T), np.nan, np.float32), "lu": np.full(T, 1.29), "cost": np.full((2, T), COST), "mkt": np.zeros(n, np.int64)}
    D["last_valid"] = E.last_valid_index(C)
    D["pool"][:, 5] = True                                  # 무작위 쪽 풀 — P_g 0
    strat = pd.DataFrame({"dec_day": [days[5], days[5]], "pg_entry": [0.6, 0.9], "ci": [3, 4], "t0": [6, 6], "base_w": [0.05, 0.05],
                          "e_ref": [1.0, 1.0], "fills": [[(6, 0.03, 100.0)], [(6, 0.045, 100.0)]], "reason": ["dead", "dead"], "hold": [20, 20]})
    plan = W.null2_plan(strat, days, T - 1)
    assert plan == [(5, [[(0, pytest.approx(0.9))], [(0, pytest.approx(0.6))]], T - 1)]   # 그날 순위 = 진입 P_g 내림차순
    r = W.null2_rep(D, plan, np.array([20]), np.random.default_rng(1))
    ret = 110 * (1 - 2 * COST) / 100 - 1                    # 기준 = 0.25%/(1 − 99/100) → 상한 5%
    assert r["n_fill"] == 2 and r["R_sum"] == pytest.approx((0.05 * 0.9 + 0.05 * 0.6) * ret / 0.0025)


def test_null2_plan_mult_path_from_fills_and_holds():
    days = pd.bdate_range("2020-01-01", periods=30)
    tr = pd.DataFrame({"dec_day": [days[3], days[3], days[7]], "pg_entry": [0.5, 0.7, 0.9], "ci": [1, 0, 2], "t0": [4, 4, 8],
                       "base_w": [0.04, 0.05, 0.02], "e_ref": [1.0, 2.0, 1.0],
                       "fills": [[(4, 0.02, 10.0), (9, 0.012, 11.0)], [(4, 0.07, 5.0)], [(8, 0.02, 3.0)]],
                       "reason": ["stop", "dead", "end"], "hold": [2, 9, 15]})
    plan = W.null2_plan(tr, days, 29)
    assert [p[0] for p in plan] == [3, 7] and plan[0][2] == 29
    a, b = plan[0][1]
    assert a == [(0, pytest.approx(0.7))]                    # P_g 0.7 이 먼저(순위), 0.07 / (0.05 × 2)
    assert b == [(0, pytest.approx(0.5)), (5, pytest.approx(0.8))]
    assert plan[1][1] == [[(0, pytest.approx(1.0))]]
    h, src = W.null2_holds(tr)
    assert list(h) == [9] and src == "non_stop"             # 손절·끝 절단 제외
    h, src = W.null2_holds(tr.assign(reason="stop"))
    assert list(h) == [2, 9, 15] and src == "all_trades_fallback"


def _bars_inputs(n_trades, seed=0, mu=0.3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", "2023-12-28")
    ed = days[rng.integers(0, len(days) - 5, n_trades)]
    tr = pd.DataFrame({"code": [f"{i % 40:06d}" for i in range(n_trades)], "R": rng.normal(mu, 1.0, n_trades), "entry_day": ed})
    eq = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.005, len(days))), index=days)
    eq.attrs["expo"] = np.full(len(days), 0.5)
    mk = pd.Series(np.cumprod(1 + rng.normal(0.0001, 0.01, len(days))), index=days)
    kx = mk * 100
    m = W.monthly(eq, kx, kx * 0.5)
    F = pd.DataFrame(rng.normal(0, 0.03, (len(m), 4)), index=m.index, columns=["mom", "smb", "lv", "rev"])
    return tr, eq, mk, m, F


def test_bars_void_under_100():
    tr, eq, mk, m, F = _bars_inputs(99)
    b = W.bars_main(tr, [1.0, 1.0, 1.0, 1.0], eq, mk, m, F, np.zeros(300), np.zeros(300), 362)
    assert b["void"] and not b["pass"]
    tr, eq, mk, m, F = _bars_inputs(100)
    b = W.bars_main(tr, [1.0, 1.0, 1.0, 1.0], eq, mk, m, F, np.zeros(300), np.zeros(300), 362)
    assert not b["void"]
    for k in ("b1_nulls", "b2_cluster_boot", "b3_top5_removed", "b4_factor_alpha", "b5_cagr_vs_expo_market", "b6_folds", "b7_dsr"):
        assert isinstance(b[k]["pass"], bool)
    assert b["pass"] == all(b[k]["pass"] for k in b if k.startswith("b"))


def test_bar_nulls_need_both():
    assert W.bar_nulls(10.0, np.arange(100.0) / 20, np.arange(100.0) / 20)["pass"]
    assert not W.bar_nulls(10.0, np.arange(100.0) / 20, np.arange(100.0))["pass"]
    assert not W.bar_nulls(10.0, np.arange(100.0), np.zeros(10))["pass"]


def test_fold_bar_3_of_4():
    tr, eq, mk, m, F = _bars_inputs(200)
    nul = np.zeros(10)
    assert W.bars_main(tr, [1, 1, 1, -1], eq, mk, m, F, nul, nul, 10)["b6_folds"]["pass"]
    assert not W.bars_main(tr, [1, 1, 0, -1], eq, mk, m, F, nul, nul, 10)["b6_folds"]["pass"]


def test_top5_removal():
    tr = pd.DataFrame({"code": list("ABCDEFG") + ["A"], "R": [10.0, 9, 8, 7, 6, -1, -2, 1]})
    mean, top = W.top_removed_mean(tr)
    assert top == ["A", "B", "C", "D", "E"] and mean == pytest.approx(-1.5)


def test_cluster_boot_by_month():
    R = np.r_[np.full(50, 1.0), np.full(50, -1.0)]
    cl = np.r_[np.full(50, 202001), np.full(50, 202002)]
    m, lo, hi = W.cluster_boot(R, cl, n_boot=4000, seed=0)
    assert m == 0.0 and lo == pytest.approx(-1.0) and hi == pytest.approx(1.0)  # 군집 두 개 — 거래 수가 아니라 달 수가 표본
    m, lo, hi = W.cluster_boot(np.full(10, 0.5), np.arange(10))
    assert lo == hi == 0.5


def test_factor_alpha_recovers_alpha():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2015-01-31", periods=96, freq="ME")
    F = pd.DataFrame(rng.normal(0, 0.04, (96, 4)), index=idx, columns=["mom", "smb", "lv", "rev"])
    k, q = rng.normal(0, 0.05, 96), rng.normal(0, 0.06, 96)
    s = 0.02 + 0.8 * k + 0.3 * q - 0.5 * F["rev"].to_numpy() + rng.normal(0, 0.005, 96)
    m = pd.DataFrame({"s": s, "k": k, "q": q}, index=idx)
    r = W.factor_alpha(m, F)
    assert r["n"] == 96 and abs(r["alpha"] - 0.02) < 0.003 and r["t_ols"] > 10 and r["t_nw"] > 10
    r0 = W.factor_alpha(m.assign(s=m["s"] - 0.02), F)
    assert abs(r0["t_ols"]) < 3


def test_factor_rev_definition():
    """REV = 직전 달 수익 하위 30% 다음 달 평균 − 상위 30% (uni 안 동일가중)."""
    days = pd.bdate_range("2020-01-01", "2020-04-30")
    n = 40
    C = np.ones((n, len(days))) * 100.0
    me = pd.Series(range(len(days)), index=days).resample("ME").last().to_numpy()
    jan, feb, mar = me[0], me[1], me[2]
    C[:, jan + 1 : feb + 1] = (100 * (1 + np.linspace(-0.2, 0.2, n)))[:, None]         # 2월 수익: 종목별 −20%..+20%
    C[:, feb + 1 :] = C[:, [feb]] * (1 + np.where(np.arange(n) < 20, 0.1, -0.1))[:, None]  # 3월: 앞 절반 +10%, 뒤 −10%
    D = {"days": days, "C": C, "cap": np.ones_like(C), "uni": np.ones_like(C, bool)}
    F = W.factor_months_rev(D)
    assert F.loc[pd.Timestamp("2020-03-31"), "rev"] == pytest.approx(0.1 - (-0.1))
    assert set(F.columns) >= {"mom", "smb", "lv", "rev"}
    assert mar > feb


def test_monthly_includes_first_month_and_stitch():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    a = pd.Series(np.linspace(1.0, 1.1, len(days)), index=days)
    m = W.monthly(a, a * 100, a * 50)
    assert m.index[0] == pd.Timestamp("2020-01-31") and len(m) == 3
    d1, d2 = pd.bdate_range("2020-01-01", "2020-12-31"), pd.bdate_range("2021-01-01", "2021-12-31")
    s1 = pd.Series(np.linspace(1, 2, len(d1)), index=d1)
    s2 = pd.Series(np.linspace(1, 1.5, len(d2)), index=d2)
    st = W.stitch([(s1, np.ones(len(d1))), (s2, np.zeros(len(d2)))])
    assert st.iloc[-1] == pytest.approx(3.0) and len(st.attrs["expo"]) == len(st)
    assert W.yearly_returns(st)[2021] == pytest.approx(0.5)


def test_expo_market():
    days = pd.bdate_range("2020-01-01", periods=4)
    eq = pd.Series([1.0, 1.0, 1.0, 1.0], index=days)
    eq.attrs["expo"] = np.array([0.5, 1.0, 0.0, 0.0])
    cw = pd.Series([0.1, 0.1, 0.1, 0.1], index=days)
    x = W.expo_market(eq, cw)
    assert np.allclose(x.to_numpy(), [1.0, 1.05, 1.05 * 1.1, 1.05 * 1.1])


def test_report_and_shakeout():
    D = rand_D(n=3, T=60)
    D["C"][0, 20:40] = 200.0
    tr = pd.DataFrame({"code": ["000000", "000001"], "ci": [0, 1], "R": [-1.0, 2.0], "reason": ["stop", "dead"],
                       "exit_day": [D["days"][18], D["days"][30]], "entry_day": [D["days"][10], D["days"][12]],
                       "entry_px": [150.0, 1e9], "d_fill": [0.03, 0.12], "hold": [8, 18], "golden_day": [pd.NaT, D["days"][20]]})
    r = W.report(D, tr, 59)
    assert r["shakeout"] == {"n_stop": 1, "rate": 1.0, "horizon": 20}
    assert r["by_reason"]["stop"]["n"] == 1 and r["all"]["payoff"] == pytest.approx(2.0)
    assert r["by_d_fill"]["0.00-0.05"]["n"] == 1 and r["by_d_fill"]["0.10-0.15"]["n"] == 1
    assert r["golden_share"] == 0.5


# --------------------------------------------------------------------------------------------- 정정 GB-1: 폴드 밖 연속 자산

def _flat_D(T=60):
    days = pd.bdate_range("2020-12-01", periods=T)
    C = np.full((2, T), 100.0)
    D = {"days": days, "codes": ["000000", "000001"], "O": C.copy(), "C": C, "ma50": np.full((2, T), 95.0), "ma200": np.full((2, T), 98.0),
         "pool": np.zeros((2, T), bool), "PG": np.full((2, T), np.nan, np.float32), "PD": np.full((2, T), np.nan, np.float32),
         "lu": np.full(T, 1.29), "cost": np.full((2, T), COST), "mkt": np.zeros(2, np.int64)}
    D["last_valid"] = E.last_valid_index(C)
    return D


def test_oof_series_trade_spans_fold_boundary():
    """폴드 2020 에 들어가 2021 에 나간 거래의 손익은 2021 자산에도 들어간다(해 구간만 잇던 예전 정의와 다름)."""
    D = _flat_D()
    days = D["days"]
    t_dec = int(days.searchsorted(pd.Timestamp("2020-12-28")))
    D["pool"][0, t_dec] = True
    D["PG"][0, t_dec] = 0.8
    y21 = int(days.searchsorted(pd.Timestamp("2021-01-04")))
    D["C"][0, y21 + 3 :] = 110.0                                          # 2021 에 오른다
    D["C"][0, y21 + 10] = 94.0                                            # 손절 → 다음 시가 청산
    D["O"][0, y21 + 11] = 105.0
    tr, _ = E.simulate(D, {"th_in": 0.5, "th_out": 0.2, "sizing": "eq", "exit": "dead_p", "th_d": 0.5}, 2.0, 1, len(days) - 1,
                       entry_end=y21 - 1)
    assert len(tr) == 1 and pd.Timestamp(tr.iloc[0]["entry_day"]).year == 2020 and pd.Timestamp(tr.iloc[0]["exit_day"]).year == 2021
    tr = tr.assign(fold=2020)
    eq = W.oof_series(D, tr, 1, len(days) - 1)
    r = eq.attrs["daily_ret"]
    w = tr.iloc[0]["cost_basis"] / tr.iloc[0]["e_ref"]
    assert r.sum() == pytest.approx(tr.iloc[0]["pnl"] / tr.iloc[0]["e_ref"])  # 거래 손익 = 일별 기여 합
    i21 = y21 - 1
    assert r[i21:].sum() == pytest.approx(r.sum())                         # 손익이 전부 2021 날짜에(2020 은 평평)
    assert eq.loc[days[y21 + 3]] / eq.loc[days[y21 + 2]] - 1 == pytest.approx(w * 0.10)
    ex = eq.attrs["expo"]
    assert ex.max() == pytest.approx(w * 1.10) and ex[-1] == 0.0
    assert W.yearly_returns(eq)[2021] != 0.0


def test_oof_series_sums_overlapping_folds_and_topups():
    D = _flat_D()
    D["pool"][0, 3] = True
    D["PG"][0, 3] = 0.5
    D["ma50"][0, 3] = 99.0
    D["PG"][0, 6] = 0.9                                                   # 7 시가 추가
    D["pool"][1, 10] = True
    D["PG"][1, 10] = 0.9
    D["C"][:, 20:] = 120.0
    cfg = {"th_in": 0.5, "th_out": 0.2, "sizing": "prob", "exit": "dead_p", "th_d": 0.5}
    a, _ = E.simulate(D, cfg, 2.0, 1, 59, entry_end=8)
    b, _ = E.simulate(D, cfg, 2.0, 9, 59)
    tr = pd.concat([a.assign(fold=1), b.assign(fold=2)], ignore_index=True)
    assert len(tr) == 2 and tr.iloc[0]["n_adds"] == 1
    eq = W.oof_series(D, tr, 1, 59)
    r = eq.attrs["daily_ret"]
    assert r.sum() == pytest.approx((tr["pnl"] / tr["e_ref"]).sum())
    ex = eq.attrs["expo"]
    assert ex[25 - 1] == pytest.approx((tr["cost_basis"] / tr["e_ref"] * 1.2).sum())   # 두 폴드 거래가 같은 날 겹친다
    with pytest.raises(ValueError):
        W.oof_series(D, tr, 5, 59)
