"""ML 예측 피처 — 절단 동일성·알게 된 날·F6 비중첩·부호표 해시(DB 불요, 합성 패널)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("krx_quant_core.risk.factor_model")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import ml_forecast_common as C  # noqa: E402
import ml_forecast_features as FT  # noqa: E402
from pead_optimizer_c0 import yoy_wide  # noqa: E402

N, T = 40, 400


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=T)
    codes = np.array([f"{i:06d}" for i in range(N)])
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, (N, T)), axis=1))
    close[3, :100] = np.nan  # 늦게 상장
    close[5, 300:] = np.nan  # 폐지
    tv = rng.uniform(1e9, 1e11, (N, T))
    tv[7, 200:210] = 0.0  # 정지
    shares = np.full((N, T), 1e7)
    shares[2, :50] = np.nan
    fnet = rng.normal(0, 1e4, (N, T))
    fnet[rng.random((N, T)) < 0.05] = np.nan
    inet = rng.normal(0, 1e4, (N, T))
    hold = 20 + np.cumsum(rng.normal(0, 0.1, (N, T)), axis=1)
    hold[rng.random((N, T)) < 0.03] = np.nan
    Xs = {k: rng.normal(0, 1, (N, T)) for k in C.STYLE_FEATURES}
    yoy = rng.normal(0, 1, (N, T))
    n_rep = 800
    rep = pd.DataFrame({"research_id": np.arange(n_rep),
                        "code": rng.choice(codes, n_rep),
                        "broker": rng.choice(["A", "B", "C"], n_rep),
                        "write_date": rng.choice(dates, n_rep),
                        "goal_price": rng.integers(0, 3, n_rep) * rng.integers(10000, 90000, n_rep)})
    per = [f"{y}Q{q}" for y in (2020, 2021, 2022) for q in (1, 2, 3, 4)]
    disc = pd.DataFrame({"code": rng.choice(codes, 300), "period": rng.choice(per, 300),
                         "first": rng.choice(dates, 300)})
    disc = disc.groupby(["code", "period"], as_index=False)["first"].min()
    return {"codes": codes, "dates": dates, "close": close, "tv": tv, "shares": shares, "fnet": fnet, "inet": inet,
            "hold": hold, "Xs": Xs, "yoy": yoy, "reports": rep, "disc": disc}


def _cut(P, d):
    """열 d 까지 자른 패널 — 원천 프레임은 그날까지의 행만."""
    day = P["dates"][d]
    out = {"codes": P["codes"], "dates": P["dates"][: d + 1]}
    for k in ("close", "tv", "shares", "fnet", "inet", "hold", "yoy"):
        out[k] = P[k][:, : d + 1]
    out["Xs"] = {k: v[:, : d + 1] for k, v in P["Xs"].items()}
    out["reports"] = P["reports"][pd.to_datetime(P["reports"]["write_date"]) <= day]
    out["disc"] = P["disc"][pd.to_datetime(P["disc"]["first"]) <= day]
    return out


def _bits_equal(a, b):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
        a[~np.isnan(a)].view(np.uint64), b[~np.isnan(b)].view(np.uint64))


@pytest.mark.parametrize("d", [150, 233, 300, 399])
def test_truncation_identity_all_features(d):
    """잘라서 다시 낸 피처 = 전체 패널 피처의 d 열(모든 F, 비트 단위)."""
    P = _panel()
    full = FT.raw_features(P)
    cut = FT.raw_features(_cut(P, d))
    for k in C.FEATURES:
        assert _bits_equal(full[k][:, d], cut[k][:, d]), k


def test_truncation_catches_future_reading_feature():
    """위반 주입 — 미래 열을 읽는 피처는 이 비교에서 걸린다(검사가 살아 있다)."""
    P = _panel()
    d = 233
    leak = lambda c: np.roll(c, -1, axis=1)  # noqa: E731 — t+1 종가를 읽는 피처
    assert not _bits_equal(FT.f_rev21(leak(P["close"]))[:, d], FT.f_rev21(leak(P["close"][:, : d + 1]))[:, d])


def test_tp_rev_known_next_trading_day_and_pairing():
    dates = pd.bdate_range("2022-01-03", periods=200)
    codes = np.array(["000001"])
    rep = pd.DataFrame({"research_id": [1, 2, 3], "code": ["000001"] * 3, "broker": ["A", "A", "B"],
                        "write_date": [dates[10], dates[40], dates[41]], "goal_price": [100, 200, 300]})
    out = FT.f_tp_rev(rep, codes, dates)[0]
    # 첫 리포트는 짝이 없다. 둘째 리포트(A 100→200)는 작성일 다음 거래일부터. B 는 짝이 없다.
    assert np.isnan(out[40])
    assert out[41] == pytest.approx(np.log(2.0))
    # 90일(달력) 뒤에는 사라진다
    gone = dates.searchsorted(dates[40] + pd.Timedelta(days=91))
    assert np.isnan(out[gone])
    assert out[gone - 1] == pytest.approx(np.log(2.0))


def test_tp_rev_ignores_zero_goal_and_collapses_same_day():
    dates = pd.bdate_range("2022-01-03", periods=60)
    rep = pd.DataFrame({"research_id": [1, 2, 3, 4], "code": ["000001"] * 4, "broker": ["A"] * 4,
                        "write_date": [dates[5], dates[6], dates[10], dates[10]],
                        "goal_price": [100, 0, 150, 200]})
    out = FT.f_tp_rev(rep, np.array(["000001"]), dates)[0]
    # 목표가 0 은 빠지고, 같은 날은 research_id 큰 200 하나 → log(200/100)
    assert out[11] == pytest.approx(np.log(2.0))


def test_ann_age_known_next_day_capped_and_latest_quarter():
    dates = pd.bdate_range("2022-01-03", periods=300)
    disc = pd.DataFrame({"code": ["000001"] * 2, "period": ["2022Q1", "2021Q4"],
                         "first": [dates[20], dates[50]]})  # 옛 분기가 나중에 알려져도 최근 분기는 그대로
    out = FT.f_ann_age(disc, np.array(["000001"]), dates)[0]
    assert np.isnan(out[20])
    assert out[21] == 0  # 다음 거래일에 처음 보이고 그날 0
    assert out[60] == 39
    assert out[250] == C.ANN_AGE_CAP


def test_frgn_own_does_not_overlap_flow_window():
    """F6 는 h(t−20)−h(t−60) — F5 창(t−19..t) 안의 보유율을 읽지 않는다."""
    rng = np.random.default_rng(1)
    hold = 10 + rng.normal(0, 1, (5, 200))
    t = 150
    a = FT.f_frgn_own(hold)[:, t]
    h2 = hold.copy()
    h2[:, t - 19: t + 1] = 99.0
    assert _bits_equal(a, FT.f_frgn_own(h2)[:, t])
    h3 = hold.copy()
    h3[:, t - 20] = 99.0
    assert not _bits_equal(a, FT.f_frgn_own(h3)[:, t])
    h4 = hold.copy()
    h4[:, t - 60] = 99.0
    assert not _bits_equal(a, FT.f_frgn_own(h4)[:, t])


def test_flow20_min_obs_and_bad_shares():
    net = np.ones((2, 30))
    net[1, 10:25] = np.nan  # 창 20 중 관측 부족
    sh = np.full((2, 30), 100.0)
    out = FT.f_flow20(net, sh)
    assert out[0, 19] == pytest.approx(0.2)
    assert np.isnan(out[1, 24])
    sh[0, 25] = 0.0
    assert np.isnan(FT.f_flow20(net, sh)[0, 25])


def test_ew_sign_table_hash_regression():
    """부호·방향 피처·피처 순서가 바뀌면 실패한다(§3 '해시 회귀 테스트로 고정')."""
    assert C.sign_table_hash() == "484a74bcb8c7cd73"
    assert C.EW_SIGNS["rev21"] == -1 and sum(v > 0 for v in C.EW_SIGNS.values()) == 5
    assert C.S_FEATURES == ("vol_shock", "size", "beta", "momentum", "resid_vol", "liquidity", "value")


def test_ew_score_missing_rules():
    k = len(C.FEATURES)
    Z = np.full((3, k), np.nan)
    Z[0, C.FEATURES.index("rev21")] = 1.0
    Z[0, C.FEATURES.index("hi52")] = 3.0
    Z[1, C.FEATURES.index("size")] = 2.0  # 방향 피처가 아니다 → 다 결측
    out = FT.ew_score(Z)
    assert out[0] == pytest.approx((-1.0 + 3.0) / 2)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_weighted_score_renormalizes_missing():
    Z = np.array([[1.0, np.nan, 3.0]])
    out = FT.weighted_score(Z, np.array([0.5, 0.25, 0.25]), np.array([1.0, 1.0, -1.0]))
    assert out[0] == pytest.approx((0.5 * 1 - 0.25 * 3) / 0.75)


def test_rank_gauss_keeps_nan_and_is_symmetric():
    v = np.array([1.0, np.nan, 2.0, 3.0])
    z = C.rank_gauss(v)
    assert np.isnan(z[1])
    assert z[0] == pytest.approx(-z[3]) and z[2] == pytest.approx(0.0)


def test_yoy_wide_known_next_day_no_fallback_and_restatement():
    """알게 된 날 = max(avail, 정기보고서 첫 접수일) 다음 거래일. 접수일 없는 분기는 avail 로 돌아가지 않는다. 정정판은 그 뒤만 바꾼다."""
    dates = pd.bdate_range("2022-01-03", periods=120)
    d = dates.strftime("%Y-%m-%d")
    ea = pd.DataFrame({"code": ["A", "A", "A", "B"], "period": ["2021Q4", "2021Q4", "2022Q1", "2021Q4"],
                       "avail_date": [d[10], d[10], d[60], d[10]],
                       "knowledge_date": [d[10], d[80], d[60], d[10]],  # 둘째 행이 정정판
                       "netinc": [20.0, 40.0, 60.0, 5.0], "netinc_prior": [10.0, 10.0, 10.0, 10.0]})
    per = pd.DataFrame({"code": ["A", "A"], "period": ["2021Q4", "2022Q1"], "per": [dates[30], dates[60]]})
    wide, _ = yoy_wide(ea.copy(), per.copy(), ["A", "B"], dates, d[-1], fallback=False)
    assert np.isnan(wide[0, 30])  # avail 은 일렀다 — 정기보고서 접수일 다음날부터
    assert wide[0, 31] == pytest.approx(1.0)
    assert np.isnan(wide[1]).all()  # B 는 접수일이 없다 → avail 로 돌아가지 않고 빠진다
    assert wide[0, 61] == pytest.approx(5.0)  # 2022Q1 이 다음 거래일부터
    # 정정판이 무관한 과거를 바꾸지 않는다
    wide2, _ = yoy_wide(ea[ea["knowledge_date"] != d[80]].copy(), per.copy(), ["A", "B"], dates, d[-1], fallback=False)
    assert _bits_equal(wide[0, :80], wide2[0, :80])
    wide_f, _ = yoy_wide(ea.copy(), per.copy(), ["A", "B"], dates, d[-1], fallback=True)
    assert wide_f[1, 11] == pytest.approx(-0.5)  # fallback 기본 동작은 그대로


def _tied_earnings(n_other=400):
    """종목 A 의 두 분기가 같은 날 알려진다 + 동률 정렬을 흔드는 다른 행들."""
    dates = pd.bdate_range("2022-01-03", periods=120)
    d = dates.strftime("%Y-%m-%d")
    rows = [("A", "2021Q4", d[20], d[20], 20.0, 10.0), ("A", "2022Q1", d[20], d[20], 40.0, 10.0)]
    rng = np.random.default_rng(0)
    for i in range(n_other):
        rows.append((f"B{i:03d}", "2021Q4", d[int(rng.integers(0, 40))], d[0], 15.0, 10.0))
    ea = pd.DataFrame(rows, columns=["code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior"])
    per = ea[["code", "period"]].assign(per=pd.to_datetime(ea["avail_date"]))
    return dates, ea, per


@pytest.mark.parametrize("seed", range(8))
def test_yoy_wide_same_day_quarters_take_later_quarter_whatever_the_row_order(seed):
    """정정 ML-2 — 같은 날 알려진 두 분기는 입력 순서·길이와 무관하게 늦은 분기다."""
    dates, ea, per = _tied_earnings()
    shuffled = ea.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    wide, _ = yoy_wide(shuffled, per.copy(), ["A"], dates, str(dates[-1].date()), fallback=False)
    assert np.isnan(wide[0, 20])
    assert wide[0, 21] == pytest.approx(3.0)  # 2022Q1 (40−10)/10


def test_yoy_panel_default_path_keeps_no_tie_rule():
    """tie_col 기본값은 그대로다 — PEAD 패리티 경로의 호출이 인자 없이 도는지(동률 규칙이 켜지지 않는지) 본다."""
    import inspect

    from swing_it.features.fundamentals import earnings_yoy_panel

    assert inspect.signature(earnings_yoy_panel).parameters["tie_col"].default is None


def test_f1_cut_catches_broken_tie_rule():
    """위반 주입 — 전체 패널에 옛 분기가 들어 있으면 (a2) 비교가 실패한다."""
    import ml_forecast_stage0 as S0

    dates, ea, per = _tied_earnings(n_other=5)
    codes = ["A"]
    full, _ = yoy_wide(ea.copy(), per.copy(), codes, dates, str(dates[-1].date()), fallback=False)
    S0._F1.clear()
    S0._F1.update(ea=ea, per=per, codes=codes, dates=dates, full=full)
    assert S0._f1_cut(30)[1] is True
    broken = full.copy()
    broken[0, 30] = 1.0  # 옛 분기(2021Q4) 값
    S0._F1["full"] = broken
    t, ok, n = S0._f1_cut(30)
    assert (t, ok, n) == (30, False, 1)
