"""골든 임박 백테스트 엔진 — 합성 패널·손 계산 기대값(DB 없음). research/signals/golden_bt_engine.py."""
import numpy as np
import pandas as pd
import pytest

from research.signals import golden_bt_engine as E

COST = 0.004
CFG = {"th_in": 0.5, "th_out": 0.2, "sizing": "eq", "exit": "dead_p", "th_d": 0.5}


def mk(n=3, T=40, px=100.0):
    """평평한 가격 100 · MA50 95 < MA200 98(골든 없음) · 확률 NaN · 상한 1.29 · 비용 0.4%."""
    days = pd.bdate_range("2020-01-01", periods=T)
    O = np.full((n, T), px)  # noqa: E741
    return {"days": days, "codes": [f"{i:06d}" for i in range(n)], "O": O, "C": O.copy(),
            "ma50": np.full((n, T), 95.0), "ma200": np.full((n, T), 98.0), "pool": np.zeros((n, T), bool),
            "PG": np.full((n, T), np.nan, np.float32), "PD": np.full((n, T), np.nan, np.float32),
            "lu": np.full(T, 1.29), "cost": np.full((2, T), COST), "mkt": np.zeros(n, np.int64)}


def enter(D, ci=0, t=5, pg=0.8):
    D["pool"][ci, t] = True
    D["PG"][ci, t] = pg


def run(D, cfg=CFG, cm=1.0, start=1, end=None, entry_end=None):
    return E.simulate(D, cfg, cm, start, D["C"].shape[1] - 1 if end is None else end, entry_end)


def w_eq(stop, o):
    return min(0.05, 0.0025 / (1 - stop / o))


# ---------------------------------------------------------------------------------------------- 진입

def test_next_open_fill_and_weight():
    D = mk()
    enter(D)
    D["O"][0, 6] = 101.0
    tr, eq = run(D)
    assert len(tr) == 1
    r = tr.iloc[0]
    assert r["dec_day"] == D["days"][5] and r["entry_day"] == D["days"][6] and r["entry_px"] == 101.0
    w = w_eq(95.0, 101.0)
    assert r["w0"] == pytest.approx(w) and r["d_fill"] == pytest.approx(1 - 95 / 101) and r["stop"] == 95.0
    ret = 100 * (1 - COST) / 101 - 1
    assert r["reason"] == "end" and r["ret"] == pytest.approx(ret) and r["R"] == pytest.approx(w * ret / 0.0025)
    assert eq.index[0] == D["days"][1] and eq.index[-1] == D["days"][-1]
    assert eq.iloc[-1] == pytest.approx(1 + w * ret)                     # 마지막 날 = 비용 뺀 순가치


def test_skip_when_open_at_or_below_stop():
    for o in (94.0, 95.0):
        D = mk()
        enter(D)
        D["O"][0, 6] = o
        tr, eq = run(D)
        assert tr.empty and eq.attrs["skips"]["below_stop"] == 1
    D = mk()
    enter(D)
    D["O"][0, 6] = 95.01
    assert len(run(D)[0]) == 1


def test_limit_up_open_no_buy():
    D = mk()
    enter(D)
    D["O"][0, 6] = 129.0                                                  # ≥ 100 × 1.29
    tr, eq = run(D)
    assert tr.empty and eq.attrs["skips"]["limit_up"] == 1
    D["O"][0, 6] = 128.9
    assert len(run(D)[0]) == 1


def test_entry_needs_pool_and_threshold():
    D = mk()
    D["PG"][0, 5] = 0.9                                                   # 풀 밖
    D["pool"][1, 5] = True
    D["PG"][1, 5] = 0.49                                                  # θ_in 미만
    D["pool"][2, 5] = True                                                # P_g NaN
    assert run(D)[0].empty
    D["PG"][1, 5] = 0.5                                                   # 같음은 들어간다(≥)
    assert list(run(D)[0]["ci"]) == [1]


# ---------------------------------------------------------------------------------------------- 손절

def test_stop_next_open_gap_below_stop_and_never_raised():
    D = mk()
    enter(D)
    D["ma50"][0, 6:] = 99.5                                               # 진입 뒤 MA50 상승 — 손절선은 95 그대로
    D["ma200"][0, 6:] = 101.0                                             # 골든은 없게(MA50 < MA200)
    D["C"][0, 8] = 97.0                                                   # 새 MA50 아래지만 95 위 → 청산 없음
    D["C"][0, 10] = 94.0
    D["O"][0, 11] = 80.0                                                  # 갭 하락: 시가 그대로 체결
    tr, _ = run(D)
    r = tr.iloc[0]
    assert r["reason"] == "stop" and r["exit_day"] == D["days"][11] and r["exit_px"] == 80.0
    w = w_eq(95.0, 100.0)
    ret = 80 * (1 - COST) / 100 - 1
    assert r["R"] == pytest.approx(w * ret / 0.0025) and r["R"] < -1.0


def test_stop_applies_after_golden_too():
    D = mk()
    enter(D)
    D["ma50"][0, 7:] = 99.0                                               # 7 에 골든
    D["C"][0, 12] = 94.9
    tr, _ = run(D, {**CFG, "exit": "ma200"})
    assert tr.iloc[0]["reason"] == "stop" and tr.iloc[0]["exit_day"] == D["days"][13]


# ---------------------------------------------------------------------------------------------- 예측 붕괴·데드·MA200

def test_collapse_exit_only_before_golden():
    D = mk()
    enter(D)
    D["PG"][0, 8] = 0.1
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "collapse" and tr.iloc[0]["exit_day"] == D["days"][9]
    D["PG"][0, 8] = 0.2                                                   # θ_out 과 같음 → 붕괴 아님(<)
    assert run(D)[0].iloc[0]["reason"] == "end"
    D = mk()
    enter(D)
    D["ma50"][0, 7:] = 99.0                                               # 7 종가에 골든
    D["PG"][0, 8] = 0.1
    tr, _ = run(D, {**CFG, "exit": "ma200"})
    assert tr.iloc[0]["reason"] == "end" and tr.iloc[0]["golden_day"] == D["days"][7]


def test_dead_exit_only_after_golden():
    D = mk()
    enter(D)
    D["PD"][0, 8] = 0.9                                                   # 골든 전 — 무시
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "end"
    D["ma50"][0, 7:] = 99.0
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "dead" and tr.iloc[0]["exit_day"] == D["days"][9]
    D["PD"][0, 8] = 0.49
    assert run(D)[0].iloc[0]["reason"] == "end"
    assert run(D, {**CFG, "th_d": 0.49})[0].iloc[0]["reason"] == "dead"


def test_golden_day_switches_rules_same_day():
    """골든이 된 판정일부터 데드 규칙(그날 P_d ≥ θ_d 면 다음 시가), 붕괴는 그날부터 꺼진다."""
    D = mk()
    enter(D)
    D["ma50"][0, 7:] = 99.0
    D["PD"][0, 7] = 0.9
    D["PG"][0, 7] = 0.05
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "dead" and tr.iloc[0]["exit_day"] == D["days"][8]


def test_ma200_exit_only_after_golden():
    D = mk()
    enter(D)
    D["C"][0, 8] = 97.0                                                   # 골든 전 MA200 아래 — 무시
    cfg = {**CFG, "exit": "ma200"}
    assert run(D, cfg)[0].iloc[0]["reason"] == "end"
    D["ma50"][0, 10:] = 99.0
    D["C"][0, 12] = 97.0                                                  # 골든 뒤 MA200(98) 아래, 손절(95) 위
    tr, _ = run(D, cfg)
    assert tr.iloc[0]["reason"] == "ma200" and tr.iloc[0]["exit_day"] == D["days"][13]


def test_nan_prob_does_not_exit_and_halt_day_skips_decision():
    D = mk()
    enter(D)
    D["ma50"][0, 7:] = 99.0
    tr, _ = run(D)                                                        # P_d 전부 NaN → 데드 없음
    assert tr.iloc[0]["reason"] == "end"
    D["C"][0, 9] = np.nan                                                 # 정지일 종가 NaN — 판정 없음(손절도)
    D["O"][0, 9] = np.nan
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "end"


# ---------------------------------------------------------------------------------------------- 비중

def test_prob_topup_to_target_capped_5pct():
    D = mk()
    enter(D, pg=0.5)
    D["ma50"][0, 5] = 99.0                                                # d_fill 1% → 기준 = min(5%, 25%) = 5%
    D["PG"][0, 6] = 0.5                                                   # 목표 2.5% = 현재 → 추가 없음
    D["PG"][0, 7] = 0.9                                                   # 목표 4.5% → 8 시가 2%p 추가
    D["PG"][0, 9] = 1.0                                                   # 목표 5% → 10 시가 0.5%p
    D["PG"][0, 11] = 1.0                                                  # 이미 5%
    tr, eq = run(D, {**CFG, "sizing": "prob"})
    r = tr.iloc[0]
    assert r["base_w"] == pytest.approx(0.05) and r["w0"] == pytest.approx(0.025)
    assert r["n_adds"] == 2 and r["cost_basis"] == pytest.approx(0.05)
    ret = 100 * (1 - COST) / 100 - 1
    assert r["R"] == pytest.approx(0.05 * ret / 0.0025)
    tr_eq, _ = run(D)                                                     # eq: 추가 없음, 처음부터 5%
    assert tr_eq.iloc[0]["n_adds"] == 0 and tr_eq.iloc[0]["cost_basis"] == pytest.approx(0.05)


def test_prob_topup_stops_after_golden_and_respects_fill_rules():
    D = mk()
    enter(D, pg=0.5)
    D["ma50"][0, 5] = 99.0
    D["ma50"][0, 7:] = 99.0                                               # 7 에 골든 → 7 판정부터 추가 없음
    D["ma200"][0, :] = 98.5
    D["PG"][0, 7] = 0.9
    tr, _ = run(D, {**CFG, "sizing": "prob", "exit": "ma200"})
    assert tr.iloc[0]["n_adds"] == 0
    D = mk()
    enter(D, pg=0.5)
    D["ma50"][0, 5] = 99.0
    D["PG"][0, 7] = 0.9
    D["O"][0, 8] = 129.1                                                  # 추가 체결일 상한가 시가 → 추가 안 함
    tr, _ = run(D, {**CFG, "sizing": "prob"})
    assert tr.iloc[0]["n_adds"] == 0


def test_weight_eq_formula_cap():
    D = mk()
    enter(D)
    D["ma50"][0, 5] = 80.0                                                # d = 20% → 0.25%/0.2 = 1.25%
    tr, _ = run(D)
    assert tr.iloc[0]["w0"] == pytest.approx(0.0125)
    D = mk()
    enter(D)
    D["ma50"][0, 5] = 99.9                                                # d = 0.1% → 상한 5%
    assert run(D)[0].iloc[0]["w0"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------------------------- 슬롯

def test_20_slots_ranked_by_pg():
    D = mk(n=25)
    D["pool"][:, 5] = True
    D["PG"][:, 5] = (0.5 + 0.01 * np.arange(25)).astype(np.float32)
    tr, eq = run(D)
    assert sorted(tr["ci"]) == list(range(5, 25)) and eq.attrs["n_orders"] == 20
    D = mk(n=22)                                                          # 같은 P_g 는 종목 행 순
    D["pool"][:, 5] = True
    D["PG"][:, 5] = 0.7
    assert sorted(run(D)[0]["ci"]) == list(range(20))


def test_freed_slot_usable_next_day():
    D = mk(n=21)
    D["pool"][:20, 5] = True
    D["PG"][:20, 5] = 0.9
    D["pool"][20, 5:25] = True
    D["PG"][20, 5:25] = 0.6
    D["C"][0, 10] = 90.0                                                  # 0 손절 → 11 시가 매도
    tr, _ = run(D)
    r20 = tr[tr["ci"] == 20].iloc[0]
    assert tr[tr["ci"] == 0].iloc[0]["exit_day"] == D["days"][11]
    assert r20["dec_day"] == D["days"][11] and r20["entry_day"] == D["days"][12]


def test_reentry_after_exit_fill():
    D = mk()
    enter(D)
    D["PG"][0, 8] = 0.1                                                   # 붕괴 → 9 시가 매도
    D["pool"][0, 8:11] = True
    D["PG"][0, 9] = 0.8                                                   # 매도 체결일 종가 판정에서 다시 → 10 시가
    tr, _ = run(D)
    assert list(tr["entry_day"]) == [D["days"][6], D["days"][10]]


def test_entry_end_stops_new_entries_but_holds():
    D = mk(n=2)
    enter(D, 0, 5)
    enter(D, 1, 20)
    tr, _ = run(D, entry_end=15)
    assert list(tr["ci"]) == [0] and tr.iloc[0]["exit_day"] == D["days"][-1]
    tr, _ = run(D, entry_end=21)
    assert sorted(tr["ci"]) == [0, 1]


# ---------------------------------------------------------------------------------------------- 정지·끝·비용

def test_suspension_delays_sell():
    D = mk()
    enter(D)
    D["C"][0, 10] = 94.0
    D["O"][0, 11] = np.nan
    D["C"][0, 11] = np.nan
    D["O"][0, 12] = 93.0
    tr, _ = run(D)
    r = tr.iloc[0]
    assert r["reason"] == "stop" and r["exit_day"] == D["days"][12] and r["exit_px"] == 93.0


def test_suspended_open_no_buy():
    D = mk()
    enter(D)
    D["O"][0, 6] = np.nan
    tr, eq = run(D)
    assert tr.empty and eq.attrs["skips"]["halt"] == 1


def test_delist_closes_at_last_price():
    D = mk()
    enter(D)
    D["C"][0, 15] = 90.0
    D["O"][0, 16:] = np.nan
    D["C"][0, 16:] = np.nan
    tr, _ = run(D)
    r = tr.iloc[0]
    assert r["reason"] == "delist" and r["exit_day"] == D["days"][16] and r["exit_px"] == 90.0


def test_mark_at_data_end_with_cost():
    D = mk()
    enter(D)
    D["C"][0, 30] = 110.0
    tr, eq = run(D, end=30)
    r = tr.iloc[0]
    assert r["reason"] == "end" and r["exit_day"] == D["days"][30] and r["exit_px"] == 110.0
    assert r["ret"] == pytest.approx(110 * (1 - COST) / 100 - 1)
    assert eq.index[-1] == D["days"][30]


def test_cost_multiplier():
    D = mk()
    enter(D)
    D["C"][0, 10] = 94.0
    a = run(D, cm=1.0)[0].iloc[0]
    b = run(D, cm=2.0)[0].iloc[0]
    assert a["ret"] == pytest.approx(100 * (1 - COST) / 100 - 1)
    assert b["ret"] == pytest.approx(100 * (1 - 2 * COST) / 100 - 1)


def test_cost_uses_exit_day_and_market():
    D = mk(n=2)
    D["mkt"][1] = 1
    D["cost"][1, :] = 0.01
    D["cost"][1, 11] = 0.02
    enter(D, 1, 5)
    D["C"][1, 10] = 94.0
    r = run(D)[0].iloc[0]
    assert r["ret"] == pytest.approx(100 * (1 - 0.02) / 100 - 1)


def test_bad_window_and_cfg():
    D = mk()
    with pytest.raises(ValueError):
        E.simulate(D, CFG, 1.0, 0, 10)
    with pytest.raises(ValueError):
        E.simulate(D, {**CFG, "sizing": "x"}, 1.0, 1, 10)


def test_equity_marks_open_positions_daily():
    D = mk()
    enter(D)
    D["C"][0, 8] = 104.0
    _, eq = run(D)
    w = w_eq(95.0, 100.0)
    assert eq.loc[D["days"][8]] == pytest.approx(1 + w * 0.04)
    assert eq.attrs["expo"][8 - 1] == pytest.approx(w * 1.04 / (1 + w * 0.04))


# ---------------------------------------------------------------------------------------------- 널 ② 거래 경로

def _lv(D):
    D["last_valid"] = E.last_valid_index(D["C"])
    return D


EQ = [(0, 1.0)]


def test_trade_path_hold_exit():
    D = _lv(mk())
    D["C"][0, 9] = 103.0
    D["O"][0, 10] = 104.0
    r = E.trade_path(D, 0, 5, 4, EQ, 1.0, 39)                         # 체결 6, 4 거래일째 시가 = 10
    w = w_eq(95.0, 100.0)
    assert r["t0"] == 6 and r["t1"] == 10 and r["reason"] == "hold"
    assert r["R"] == pytest.approx(w * (104 * (1 - COST) / 100 - 1) / 0.0025)


def test_trade_path_stop_first_and_fill_rules():
    D = _lv(mk())
    D["C"][0, 7] = 94.0
    D["O"][0, 8] = 90.0
    r = E.trade_path(D, 0, 5, 10, EQ, 2.0, 39)
    assert r["reason"] == "stop" and r["t1"] == 8
    assert r["ret"] == pytest.approx(90 * (1 - 2 * COST) / 100 - 1)
    D["O"][0, 6] = 129.0
    assert E.trade_path(D, 0, 5, 10, EQ, 1.0, 39) is None
    D["O"][0, 6] = 94.0
    assert E.trade_path(D, 0, 5, 10, EQ, 1.0, 39) is None
    with pytest.raises(ValueError):
        E.trade_path(_lv(mk()), 0, 5, 10, [(2, 1.0)], 1.0, 39)


def test_trade_path_suspension_end_and_mult_path_adds():
    D = _lv(mk())
    D["O"][0, 10] = np.nan
    D["C"][0, 10] = np.nan
    r = E.trade_path(D, 0, 5, 4, EQ, 1.0, 39)
    assert r["t1"] == 11
    r = E.trade_path(D, 0, 5, 100, EQ, 1.0, 20)
    assert r["reason"] == "end" and r["t1"] == 20
    D = _lv(mk())
    D["ma50"][0, 5] = 99.0                                                # 기준 5%
    r = E.trade_path(D, 0, 5, 20, [(0, 0.5), (2, 0.9)], 1.0, 39)          # 진입 2.5%, 체결 뒤 2일째(8) 원가 4.5% 까지
    assert r["ret"] == pytest.approx(100 * (1 - COST) / 100 - 1)
    assert r["R"] == pytest.approx(0.045 * r["ret"] / 0.0025)
    r = E.trade_path(D, 0, 5, 1, [(0, 0.5), (2, 0.9)], 1.0, 39)           # 추가 날 전에 청산 → 추가 없음
    assert r["R"] == pytest.approx(0.025 * r["ret"] / 0.0025)
    D["O"][0, 8] = 129.1                                                  # 추가 체결일 상한가 → 그 추가 없음
    r = E.trade_path(D, 0, 5, 20, [(0, 0.5), (2, 0.9)], 1.0, 39)
    assert r["R"] == pytest.approx(0.025 * (100 * (1 - COST) / 100 - 1) / 0.0025)


def test_trade_path_ignores_own_pg():
    """GB-2 C1: 무작위 종목 자신의 P_g(0 이든 NaN 이든)는 비중에 쓰지 않는다."""
    D = _lv(mk())
    D["ma50"][0, 5] = 99.0
    for pg in (0.0, np.nan, 1.0):
        D["PG"][0, :] = pg
        r = E.trade_path(D, 0, 5, 20, [(0, 0.8)], 1.0, 39)
        assert r["R"] == pytest.approx(0.04 * (100 * (1 - COST) / 100 - 1) / 0.0025)


# ---------------------------------------------------------------------------------------------- 정정 GB-2: 불타기 = 원가 기준

def test_gb2_no_add_when_pg_flat_and_price_falls():
    """P_g 0.8 그대로, 가격만 100 → 97(손절 95 위): 시가 평가 비중은 줄지만 원가 비중은 그대로 → 추가 없음(물타기 금지)."""
    D = mk(T=20)
    enter(D, pg=0.8)
    D["PG"][0, 6:12] = 0.8
    D["C"][0, 6:] = 97.0
    D["O"][0, 7:] = 97.0
    tr, _ = run(D, {**CFG, "sizing": "prob"}, end=12)
    r = tr.iloc[0]
    assert r["n_adds"] == 0 and r["cost_basis"] == pytest.approx(r["w0"])


def test_gb2_add_when_pg_rises_buys_cost_basis_difference():
    D = mk(T=20)
    enter(D, pg=0.5)
    D["ma50"][0, 5] = 99.0                                                # 기준 5%, 진입 2.5%
    D["C"][0, 6:] = 120.0                                                 # 가격이 올라 시가 평가 비중은 3% — 원가 비중 2.5% 로 비교
    D["O"][0, 7:] = 120.0
    D["PG"][0, 7] = 0.6                                                   # 목표 3% → 원가 차이 0.5%p 추가(시가 평가라면 추가 없음)
    tr, _ = run(D, {**CFG, "sizing": "prob"}, end=15)
    r = tr.iloc[0]
    assert r["n_adds"] == 1 and r["cost_basis"] == pytest.approx(0.03)
    assert r["fills"][1][0] == 8 and r["fills"][1][1] == pytest.approx(0.005)


# ---------------------------------------------------------------------------------------------- 정정 GB-1: 정지일 창 대체 청산

def test_valid_ma_skips_nan_days():
    C = np.array([[1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0]])
    v = E.valid_ma(C, 3)
    assert np.isnan(v[0, :3]).all() and v[0, 3] == pytest.approx(2.0) and v[0, 4] == pytest.approx(3.0)
    assert np.isnan(v[0, 5]) and v[0, 6] == pytest.approx(4.0)


def test_gb1_nan_prob_before_golden_holds_with_stop():
    D = mk()
    enter(D)
    D["PG"][0, 6:] = np.nan                                               # 골든 전 확률 NaN — 보유
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "end" and tr.iloc[0]["nan_dec"] > 0
    D["C"][0, 12] = 94.0                                                  # 손절은 그대로
    tr, _ = run(D)
    r = tr.iloc[0]
    assert r["reason"] == "stop" and r["exit_day"] == D["days"][13] and r["nan_dec"] == 12 - 6


def test_gb1_nan_after_golden_falls_back_to_valid_ma200():
    for cfg in (CFG, {**CFG, "exit": "ma200"}):
        D = mk()
        enter(D)
        D["ma50"][0, :] = np.nan                                          # 정지일이 창에 낀 정규 MA — NaN
        D["ma200"][0, 7:] = np.nan
        D["ma50v"] = np.full_like(D["C"], 95.0)
        D["ma200v"] = np.full_like(D["C"], 98.0)
        D["ma50v"][0, 7:] = 99.0                                          # 건너뛴 MA 로 골든 감지(7)
        D["C"][0, 10] = 97.0                                              # 건너뛴 MA200(98) 아래, 손절(95) 위
        D["ma50"][0, 5] = 95.0                                            # 진입 판정일 손절선
        tr, _ = run(D, cfg)
        r = tr.iloc[0]
        assert r["golden_day"] == D["days"][7]
        assert r["reason"] == "ma200_fb" and r["exit_day"] == D["days"][11] and r["nan_dec"] == 1 + (10 - 7 + 1)   # 6(골든 전 P_g NaN) + 7..10
    D = mk()                                                              # 신호가 있으면 대체를 쓰지 않는다
    enter(D)
    D["ma50"][0, 7:] = 99.0
    D["PG"][0, 6] = 0.8
    D["PD"][0, 7:] = 0.1
    D["ma200v"] = D["ma200"].copy()
    D["ma50v"] = D["ma50"].copy()
    D["C"][0, 10] = 97.0
    tr, _ = run(D)
    assert tr.iloc[0]["reason"] == "end" and tr.iloc[0]["nan_dec"] == 0


def test_fills_ledger_sums_to_cost_basis():
    D = mk()
    enter(D, pg=0.5)
    D["ma50"][0, 5] = 99.0
    D["PG"][0, 7] = 0.9
    tr, _ = run(D, {**CFG, "sizing": "prob"})
    r = tr.iloc[0]
    assert [f[0] for f in r["fills"]] == [6, 8] and sum(f[1] for f in r["fills"]) == pytest.approx(r["cost_basis"])
    assert r["t0"] == 6 and r["t1"] == len(D["days"]) - 1
