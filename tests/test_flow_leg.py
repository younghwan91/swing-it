"""F(수급) 다리 — NaN 전파·창 완전성·룩어헤드 없음·F 플래그·기준 단절 불변(가중 모드)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing_it.features.flow_leg import (
    f_flag,
    flow_known,
    flow_leg,
    flow_wide,
    foreign_net,
    foreign_ratio,
    smart_persistence,
    smart_ratio,
)
from swing_it.features.flows import smart_money_net

W = 5  # 테스트는 짧은 창으로 — 로직은 창 길이와 무관하다.


def _sd(n=12, codes=("AAA", "BBB"), seed=0):
    """정문 결과 모양의 long 프레임(``code``·``date``·``foreign_``·``institution``·``acc_trde_qty``)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for c in codes:
        for d in dates:
            vol = int(rng.integers(1_000, 10_000))
            rows.append({"code": c, "date": d.strftime("%Y-%m-%d"),
                         "foreign_": int(rng.integers(-500, 500)),
                         "institution": int(rng.integers(-500, 500)),
                         "acc_trde_qty": vol})
    return pd.DataFrame(rows), dates


def _panels(sd, index=None):
    w = flow_wide(sd, index=index)
    return w["foreign"], w["institution"], w["volume"]


# --- 기본 정의 ------------------------------------------------------------------

def test_smart_ratio_is_sum_net_over_sum_volume():
    sd, dates = _sd()
    f, i, v = _panels(sd)
    sr = smart_ratio(f, i, v, window=W)
    t = dates[7]
    win = slice(dates[3], dates[7])
    exp = (f.loc[win, "AAA"] + i.loc[win, "AAA"]).sum() / v.loc[win, "AAA"].sum()
    assert sr.loc[t, "AAA"] == pytest.approx(exp)


def test_net_matches_flows_smart_money_net():
    """같은 정의(외국인+기관, NaN 전파)를 flows 와 공유한다."""
    sd, _ = _sd()
    sd.loc[3, "institution"] = np.nan
    f, i, _v = _panels(sd)
    pd.testing.assert_frame_equal((f + i), smart_money_net(sd).astype(float), check_names=False)


def test_warmup_rows_are_nan_and_unknown():
    sd, dates = _sd()
    f, i, v = _panels(sd)
    sr = smart_ratio(f, i, v, window=W)
    k = flow_known(f, i, v, window=W)
    assert sr.iloc[: W - 1].isna().all().all()
    assert not k.iloc[: W - 1].any().any()
    assert sr.iloc[W - 1:].notna().all().all()
    assert k.iloc[W - 1:].all().all()


# --- NULL = 모름 ----------------------------------------------------------------

def test_null_poisons_exactly_the_windows_that_contain_it():
    sd, dates = _sd()
    bad = dates[6]
    sd.loc[(sd.code == "AAA") & (sd.date == bad.strftime("%Y-%m-%d")), "foreign_"] = np.nan
    f, i, v = _panels(sd)
    sr = smart_ratio(f, i, v, window=W)
    k = flow_known(f, i, v, window=W)
    fn = foreign_net(f, window=W)
    pers = smart_persistence(f, i, window=W)
    hit = dates[6:6 + W]  # bad 를 창에 담는 행 t ∈ [bad, bad+W−1]
    for out in (sr, fn, pers):
        assert out.loc[hit, "AAA"].isna().all()
        assert out.loc[dates[W - 1:6], "AAA"].notna().all()
        assert out.loc[dates[6 + W:], "AAA"].notna().all()
        assert out.loc[dates[W - 1:], "BBB"].notna().all()  # 다른 종목은 오염되지 않는다
    assert not k.loc[hit, "AAA"].any()
    assert k.loc[dates[6 + W:], "AAA"].all()


def test_institution_null_makes_smart_unknown_but_not_foreign_net():
    sd, dates = _sd()
    sd.loc[(sd.code == "AAA") & (sd.date == dates[6].strftime("%Y-%m-%d")), "institution"] = np.nan
    out = flow_leg(sd, window=W)
    assert np.isnan(out["smart_ratio_20"].loc[dates[8], "AAA"])
    assert np.isnan(out["F"].loc[dates[8], "AAA"])
    assert not out["flow_known"].loc[dates[8], "AAA"]
    assert not np.isnan(out["foreign_net_20"].loc[dates[8], "AAA"])  # 외국인 자체는 알려졌다


def test_missing_row_is_unknown_when_reindexed_to_calendar():
    """행이 아예 없으면(2017~2023 미수집) 달력으로 reindex 해야 '모름' 이 된다 — 0 도, 건너뛰기도 아니다."""
    sd, dates = _sd()
    gone = dates[6].strftime("%Y-%m-%d")
    sd2 = sd[~((sd.code == "AAA") & (sd.date == gone))]
    out = flow_leg(sd2, index=dates, window=W)
    assert out["smart_ratio_20"].loc[dates[6:6 + W], "AAA"].isna().all()
    assert not out["flow_known"].loc[dates[6:6 + W], "AAA"].any()


def test_null_is_never_filled_with_zero():
    sd, dates = _sd()
    sd["foreign_"] = sd["foreign_"].astype(float)
    sd.loc[sd.code == "AAA", "foreign_"] = np.nan
    out = flow_leg(sd, window=W)
    for key in ("smart_ratio_20", "foreign_net_20", "foreign_ratio_20", "F", "smart_pos_days_20"):
        assert out[key]["AAA"].isna().all(), key
    assert not out["flow_known"]["AAA"].any()


def test_duplicate_code_date_fails_loudly():
    sd, _ = _sd()
    with pytest.raises(ValueError):
        flow_wide(pd.concat([sd, sd.iloc[:1]]))


# --- 룩어헤드 없음 ----------------------------------------------------------------

def test_no_lookahead_future_rows_do_not_change_past_values():
    sd, dates = _sd(n=15)
    base = flow_leg(sd, window=W)
    cut = dates[9]
    sd2 = sd.copy()
    future = pd.to_datetime(sd2.date) > cut
    sd2.loc[future, ["foreign_", "institution"]] = sd2.loc[future, ["foreign_", "institution"]] * -50 + 7
    sd2.loc[future, "acc_trde_qty"] = 1
    alt = flow_leg(sd2, window=W)
    for key, panel in base.items():
        pd.testing.assert_frame_equal(panel.loc[:cut], alt[key].loc[:cut], obj=key)


def test_value_at_t_uses_t_inclusive_window():
    """행 t 는 t 당일을 포함한다(t+1 시가 진입 전제). t 당일 값을 바꾸면 t 가 바뀐다."""
    sd, dates = _sd()
    t = dates[8]
    a = flow_leg(sd, window=W)["smart_ratio_20"].loc[t, "AAA"]
    sd2 = sd.copy()
    sd2.loc[(sd2.code == "AAA") & (sd2.date == t.strftime("%Y-%m-%d")), "foreign_"] += 10_000
    b = flow_leg(sd2, window=W)["smart_ratio_20"].loc[t, "AAA"]
    assert b > a


# --- F 플래그 -------------------------------------------------------------------

def test_f_flag_truth_table_and_nan():
    idx = pd.DatetimeIndex(["2024-01-01"])
    smart = pd.DataFrame([[0.1, 0.1, -0.1, 0.0, np.nan, 0.1]], index=idx, columns=list("abcdef"))
    fore = pd.DataFrame([[5.0, -5.0, 5.0, 5.0, 5.0, np.nan]], index=idx, columns=list("abcdef"))
    out = f_flag(smart, fore).iloc[0]
    assert out["a"] == 1.0
    assert out["b"] == 0.0 and out["c"] == 0.0
    assert out["d"] == 0.0  # 경계: 0 은 > 0 이 아니다
    assert np.isnan(out["e"]) and np.isnan(out["f"])  # 모름은 거짓이 아니다


def test_f_flag_end_to_end_on_constructed_flows():
    dates = pd.bdate_range("2024-01-01", periods=W)
    rows = []
    for c, f, i in (("BUY", 100, 50), ("INST_ONLY", -10, 200), ("SELL", -100, -50)):
        for d in dates:
            rows.append({"code": c, "date": d, "foreign_": f, "institution": i, "acc_trde_qty": 1_000})
    out = flow_leg(pd.DataFrame(rows), window=W)
    last = out["F"].iloc[-1]
    assert last["BUY"] == 1.0
    assert last["INST_ONLY"] == 0.0  # 합은 양수지만 외국인이 판다 → 설계서 F 아님
    assert last["SELL"] == 0.0
    assert out["smart_ratio_20"].iloc[-1]["BUY"] == pytest.approx(0.15)
    assert out["foreign_net_20"].iloc[-1]["BUY"] == pytest.approx(100 * W)
    assert out["smart_pos_days_20"].iloc[-1].tolist() == [W, W, 0]


def test_qty_mode_foreign_ratio_sign_equals_foreign_net_sign():
    sd, _ = _sd(n=30)
    f, _i, v = _panels(sd)
    fr, fn = foreign_ratio(f, v, window=W), foreign_net(f, window=W)
    m = fr.notna()
    assert m.iloc[W - 1:].all().all()
    assert (np.sign(fr) == np.sign(fn))[m].all().all()


# --- 거래량 0·전 창 정지 ------------------------------------------------------------

def test_all_halted_window_is_known_but_ratio_nan():
    dates = pd.bdate_range("2024-01-01", periods=W)
    sd = pd.DataFrame({"code": "H", "date": dates, "foreign_": 0, "institution": 0, "acc_trde_qty": 0})
    out = flow_leg(sd, window=W)
    assert out["flow_known"].iloc[-1]["H"]
    assert np.isnan(out["smart_ratio_20"].iloc[-1]["H"])
    assert np.isnan(out["F"].iloc[-1]["H"])
    assert out["smart_pos_days_20"].iloc[-1]["H"] == 0


# --- 가중 모드: 기준 단절 불변 -----------------------------------------------------

def test_weighted_mode_is_invariant_to_basis_break_inside_window():
    """창 중간에 1:5 분할 기준 단절 — 단절 전 행의 수량만 1/5 로 기록됐다고 하자(수집 뒤 원자료가 이어 붙은 경우).

    수량 합 비율은 단절 전 날들을 1/5 로 가볍게 세서 부호가 뒤집히지만, 거래대금 가중 하루 비율은 그대로다.
    """
    dates = pd.bdate_range("2024-01-01", periods=W)
    # 단절 전 2일: 외국인 대량 매수, 단절 후 3일: 소량 매도. 같은 기준이면 합은 순매수.
    f_true = [400, 400, -100, -100, -100]
    vol_true = [1_000] * W
    scale = [0.2, 0.2, 1, 1, 1]  # 단절 전 행은 다른 주식수 기준(×0.2)으로 저장됐다
    rows_true, rows_broken = [], []
    for d, f, v, s in zip(dates, f_true, vol_true, scale):
        rows_true.append({"code": "X", "date": d, "foreign_": f, "institution": 0, "acc_trde_qty": v})
        rows_broken.append({"code": "X", "date": d, "foreign_": f * s, "institution": 0, "acc_trde_qty": v * s})
    tv = pd.DataFrame({"X": [5_000.0] * W}, index=dates)  # 거래대금은 주식수 기준이 없다
    t, b = pd.DataFrame(rows_true), pd.DataFrame(rows_broken)
    qty_true = flow_leg(t, window=W)["smart_ratio_20"].iloc[-1]["X"]
    qty_broken = flow_leg(b, window=W)["smart_ratio_20"].iloc[-1]["X"]
    assert qty_true > 0 and qty_broken < 0  # 수량 모드는 단절에 부호가 뒤집힌다
    w_true = flow_leg(t, window=W, weight=tv)["smart_ratio_20"].iloc[-1]["X"]
    w_broken = flow_leg(b, window=W, weight=tv)["smart_ratio_20"].iloc[-1]["X"]
    assert w_true == pytest.approx(w_broken) and w_true > 0
    assert flow_leg(b, window=W, weight=tv)["F"].iloc[-1]["X"] == 1.0


def test_weighted_mode_equals_qty_mode_with_constant_volume_and_weight():
    dates = pd.bdate_range("2024-01-01", periods=W + 3)
    rng = np.random.default_rng(3)
    sd = pd.DataFrame({"code": "Y", "date": dates, "foreign_": rng.integers(-50, 50, len(dates)),
                       "institution": rng.integers(-50, 50, len(dates)), "acc_trde_qty": 1_000})
    tv = pd.DataFrame({"Y": 7.0}, index=dates)
    a = flow_leg(sd, window=W)["smart_ratio_20"]["Y"]
    b = flow_leg(sd, window=W, weight=tv)["smart_ratio_20"]["Y"]
    pd.testing.assert_series_equal(a, b)


def test_weighted_mode_unknown_weight_is_unknown():
    sd, dates = _sd()
    tv = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB"])
    tv.loc[dates[6], "AAA"] = np.nan
    out = flow_leg(sd, window=W, weight=tv)
    assert out["smart_ratio_20"].loc[dates[6:6 + W], "AAA"].isna().all()
    assert not out["flow_known"].loc[dates[6:6 + W], "AAA"].any()
    assert out["smart_ratio_20"].loc[dates[W - 1:], "BBB"].notna().all()


def test_weighted_mode_zero_volume_with_nonzero_net_is_unknown():
    dates = pd.bdate_range("2024-01-01", periods=W)
    sd = pd.DataFrame({"code": "Z", "date": dates, "foreign_": [10, 10, 10, 10, 10],
                       "institution": 0, "acc_trde_qty": [100, 100, 0, 100, 100]})
    tv = pd.DataFrame({"Z": 1.0}, index=dates)
    assert np.isnan(flow_leg(sd, window=W, weight=tv)["smart_ratio_20"].iloc[-1]["Z"])
    sd.loc[2, "foreign_"] = 0  # 거래 없음 ∧ 순매수 0 → 알려진 0
    assert flow_leg(sd, window=W, weight=tv)["smart_ratio_20"].iloc[-1]["Z"] == pytest.approx(0.08)
