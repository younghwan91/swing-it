"""시장 국면 패널(``swing_it.features.regime``) — 룩어헤드 없음·돌파 성공률 PIT·스펙트럼 인과성 (합성 데이터만)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swing_it.features import regime as rg
from swing_it.features.minervini import market_breadth


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2019-01-01", periods=n)


def _inputs(n: int = 700, k: int = 24, seed: int = 0) -> rg.RegimeInputs:
    rng = np.random.default_rng(seed)
    idx = _days(n)
    codes = [f"{i:06d}" for i in range(k)]
    drift = np.linspace(-0.001, 0.002, k)
    close = 100 * np.exp(np.cumsum(drift + 0.02 * rng.standard_normal((n, k)), axis=0))
    high = close * (1 + 0.01 * rng.random((n, k)))
    low = close * (1 - 0.01 * rng.random((n, k)))
    vol = rng.integers(50, 300, (n, k)).astype(float)

    def f(a):
        return pd.DataFrame(a, index=idx, columns=codes)

    market = pd.Series(["KOSPI", "KOSDAQ"] * (k // 2), index=codes)
    tv = f(vol * close * rng.uniform(0.5, 2.0, k))
    net = f(rng.integers(-50, 50, (n, k)).astype(float))
    off = pd.DataFrame({"KOSPI": close[:, 0], "KOSDAQ": close[:, 1]}, index=idx)
    return rg.RegimeInputs(close=f(close), high=f(high), low=f(low), volume=f(vol), market=market, trade_value=tv,
                           flow_net=net, flow_volume=f(vol), official=off)


def _truncate(inp: rg.RegimeInputs, m: int) -> rg.RegimeInputs:
    cut = lambda df: None if df is None else df.iloc[:m]  # noqa: E731
    return rg.RegimeInputs(close=cut(inp.close), high=cut(inp.high), low=cut(inp.low), volume=cut(inp.volume),
                           market=inp.market, trade_value=cut(inp.trade_value), flow_net=cut(inp.flow_net),
                           flow_volume=cut(inp.flow_volume), official=cut(inp.official))


# --- 전체 패널: 뒤를 잘라도 앞 행은 그대로(룩어헤드 없음) ---

def test_panel_rows_do_not_depend_on_future_rows(monkeypatch):
    monkeypatch.setattr(rg, "MIN_INDEX_N", 5)          # 합성 24종목에서도 지수·스펙트럼 열이 값을 갖게
    inp = _inputs()
    full = rg.build_regime_panel(inp)
    part = rg.build_regime_panel(_truncate(inp, 600))
    d = inp.close.index[599]
    a = full[full["date"] <= d].reset_index(drop=True)
    b = part.reset_index(drop=True)
    assert len(a) == len(b) == 600 * 3
    pd.testing.assert_frame_equal(a, b)


def test_panel_shape_and_columns():
    inp = _inputs(n=400)
    p = rg.build_regime_panel(inp)
    assert list(p.columns) == ["date", "market", *rg.FEATURES]
    assert set(p["market"]) == set(rg.MARKETS)
    assert (p.groupby("market").size() == 400).all()


# --- breadth ---

def test_breadth_matches_minervini_market_breadth():
    inp = _inputs(n=300)
    mine = rg._breadth_all(inp.close, inp.market, 50)
    ref = market_breadth(inp.close, inp.market, window=50)
    for m in ("KOSPI", "KOSDAQ"):
        pd.testing.assert_series_equal(mine[m], ref[m], check_names=False)
    assert mine["ALL"].dropna().between(0, 1).all()


# --- 시총가중 지수: 전일 가중, 수익률 평균 ---

# --- 돌파 성공률: 결과가 확정된 날부터만 ---

def test_breakout_success_is_pit_until_horizon_passes():
    n, w, h = 120, 50, 5
    idx = _days(n)
    close = pd.Series(100.0, index=idx)
    high = close * 1.01
    vol = pd.Series(100.0, index=idx)
    b = 80  # 돌파일
    close.iloc[b] = 110.0
    high.iloc[b] = 111.0
    vol.iloc[b] = 500.0
    close.iloc[b + 1:] = 110.0
    high.iloc[b + 1:] = 110.5
    frame = lambda s: s.to_frame("A")  # noqa: E731
    market = pd.Series({"A": "KOSPI"})
    out = rg.breakout_success(frame(high), frame(close), frame(vol), market, window=w, horizon=h, lookback=20)
    cnt, ok = out["breakouts20"]["ALL"], out["breakout_success20"]["ALL"]
    # 돌파 뒤 h−1 일까지는 아직 결과 없음(분모 0 → 성공률 NaN)
    assert (cnt.iloc[b: b + h] == 0).all() and ok.iloc[b: b + h].isna().all()
    # b+h 에 결과 확정: 종가 110 > 피벗 101 → 성공
    assert cnt.iloc[b + h] == 1 and ok.iloc[b + h] == 1.0
    # 20일 창(b ∈ [d−19, d−5]) 밖으로 나가면 다시 0
    assert cnt.iloc[b + 20] == 0
    # 미래 종가를 바꿔도 b+h 전 행은 안 바뀐다
    close2 = close.copy()
    close2.iloc[b + h:] = 50.0
    out2 = rg.breakout_success(frame(high), frame(close2), frame(vol), market, window=w, horizon=h, lookback=20)
    pd.testing.assert_frame_equal(out["breakouts20"].iloc[: b + h], out2["breakouts20"].iloc[: b + h])
    assert out2["breakout_success20"]["ALL"].iloc[b + h] == 0.0  # 실패로 확정


# --- flow20: since 전은 NaN, 창은 since 뒤에서 시작 ---

def test_universe_flow_starts_after_since():
    idx = _days(60)
    net = pd.DataFrame({"A": 1.0, "B": -0.5}, index=idx)
    vol = pd.DataFrame({"A": 10.0, "B": 10.0}, index=idx)
    market = pd.Series({"A": "KOSPI", "B": "KOSDAQ"})
    since = idx[30]
    f = rg.universe_flow(net, vol, market, window=20, since=since.strftime("%Y-%m-%d"))
    assert f.iloc[:49].isna().all().all()                 # since 뒤 20일째에야 첫 값
    assert f.loc[idx[49], "ALL"] == pytest.approx(0.5 / 20)
    assert f.loc[idx[49], "KOSPI"] == pytest.approx(0.1)


# --- 신고가/신저가 ---

def test_new_high_low_counts_and_ratio():
    n = 260
    idx = _days(n)
    base = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    high, low = base.copy(), base.copy()
    high.iloc[-1, 0] = 101.0   # A 신고가
    low.iloc[-1, 1] = 99.0     # B 신저가
    market = pd.Series({"A": "KOSPI", "B": "KOSPI"})
    r = rg.new_high_low(high, low, market)
    assert r["new_highs"]["KOSPI"].iloc[-1] == 1 and r["new_lows"]["KOSPI"].iloc[-1] == 1
    assert r["nh_nl_ratio"]["KOSPI"].iloc[-1] == 0.5
    assert np.isnan(r["nh_nl_ratio"]["KOSPI"].iloc[-2])   # 둘 다 0
    assert np.isnan(r["new_highs"]["KOSPI"].iloc[0]) or r["new_highs"]["KOSPI"].iloc[0] == 0


# --- 스펙트럼 ---

def test_periodogram_finds_sine_period_40():
    n = 600
    t = np.arange(n)
    x = pd.Series(np.sin(2 * np.pi * t / 40.0) + 0.05 * np.random.default_rng(1).standard_normal(n))
    pg = rg.rolling_periodogram(x)
    assert pg["period"].iloc[: rg.SPEC_WINDOW - 1].isna().all()
    got = pg["period"].iloc[rg.SPEC_WINDOW - 1:]
    assert got.notna().all()
    assert (got - 40.0).abs().max() < 3.0
    assert pg["share"].iloc[-1] > 0.3


def test_periodogram_ignores_linear_trend_and_restricts_band():
    n = 400
    t = np.arange(n, dtype=float)
    x = pd.Series(0.01 * t + np.sin(2 * np.pi * t / 20.0))
    pg = rg.rolling_periodogram(x)
    assert abs(pg["period"].iloc[-1] - 20.0) < 2.0
    assert pg["period"].dropna().between(rg.SPEC_MIN_PERIOD, rg.SPEC_MAX_PERIOD).all()


def test_spectral_features_are_causal():
    n = 500
    rng = np.random.default_rng(2)
    x = pd.Series(np.cumsum(rng.standard_normal(n)))
    y = x.copy()
    y.iloc[400:] += 1000.0 * rng.standard_normal(100)   # 미래만 바꾼다
    a, b = rg.spectral_block(x, "s"), rg.spectral_block(y, "s")
    pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400])
    assert not a.iloc[400:].equals(b.iloc[400:])


def test_haar_shares_sum_to_one_and_track_scale():
    n = 600
    t = np.arange(n)
    hz = rg.haar_variance_share(pd.Series(np.sin(2 * np.pi * t / 32.0)))
    tail = hz.dropna()
    assert len(tail) > 0
    assert np.allclose(tail.sum(axis=1), 1.0)
    assert tail.iloc[-1].idxmax() == 4   # 32일 사인은 가장 긴 척도(16일 블록 차)가 우세


def test_haar_details_use_only_past():
    idx = _days(40)
    x = pd.Series(np.arange(40, dtype=float), index=idx)
    d = rg.haar_details(x, levels=2)
    # 단계 1: x[t] − x[t−1] = 1; 단계 2: mean(t−1,t) − mean(t−3,t−2) = 2
    assert d[1].iloc[1] == 1.0 and d[2].iloc[3] == 2.0
    assert np.isnan(d[2].iloc[2])


# --- PIT 주식수 ---

def test_coverage_by_year_is_fraction():
    inp = _inputs(n=520)
    cov = rg.coverage_by_year(rg.build_regime_panel(inp))
    assert cov.index.name == "year" and ((cov >= 0) & (cov <= 1)).all().all()
    assert cov["breadth_ma50"].iloc[-1] == 1.0


# --- 자료 사건 가드 ---

def test_patch_wide_nxt_replaces_late_rows_with_session_bars():
    idx = _days(4)
    since = idx[2].strftime("%Y-%m-%d")
    panels = {c: pd.DataFrame({"A": [100.0, 100.0, 200.0, 200.0]}, index=idx) for c in rg.PRICE_COLS}
    panels["volume"] = pd.DataFrame({"A": [10.0, 10.0, 50.0, 50.0]}, index=idx)
    # 정규장 분봉 집계: 시가 100(조정 전 원값) → 조정계수 2, 종가 90 → 180, 거래량 7
    sess = pd.DataFrame({"code": ["A"], "date": [idx[2]], "open": [100.0], "high": [95.0], "low": [85.0], "close": [90.0],
                         "volume": [7.0], "trade_value": [1.0]})
    info = rg.patch_wide_nxt(panels, sess, since=since)
    assert info == {"nxt_rows": 2, "nxt_patched": 1}
    assert panels["close"].loc[idx[2], "A"] == pytest.approx(180.0)
    assert panels["volume"].loc[idx[2], "A"] == 7.0
    assert panels["close"].loc[idx[3], "A"] == 200.0 and panels["close"].loc[idx[1], "A"] == 100.0


# --- 지수 대용(거래대금 가중, 리뷰 C1) ---

def test_capped_weights_caps_and_renormalizes():
    rng = np.random.default_rng(3)
    w = rng.pareto(1.2, (50, 30)) + 1e-3
    w[:, 0] = 1e6                                   # 한 종목이 압도
    w[3, 5:] = np.nan                               # 5종목만 → 10% 상한 불가 → NaN
    out = rg.capped_weights(w, 0.10)
    ok = ~np.isnan(out).all(axis=1)
    assert not ok[3] and ok.sum() == 49
    assert np.allclose(out[ok].sum(axis=1), 1.0)
    assert (out[ok] <= 0.10 + 1e-12).all()
    assert np.allclose(out[ok, 0], 0.10)
    # 안 깎인 종목끼리는 원 가중 비례가 유지된다
    r = 7
    free = out[r] < 0.10 - 1e-9
    assert np.allclose(out[r, free] / out[r, free].sum(), w[r, free] / w[r, free].sum())


def test_tv_index_uses_lagged_trade_value_and_drops_limit_breaches():
    idx = _days(25)
    codes = [f"{i:02d}" for i in range(12)]
    close = pd.DataFrame(100.0, index=idx, columns=codes)
    tv = pd.DataFrame(1.0, index=idx, columns=codes)
    market = pd.Series("KOSPI", index=codes)
    close.iloc[22:, 0] = 110.0                      # d=22: 종목 00 +10%
    close.iloc[22:, 1] = 200.0                      # d=22: 종목 01 +100% → 자료 사건, 수익·가중 둘 다 제외
    tv.iloc[22, 2] = 1e9                            # d=22 당일 거래대금 폭증 — 가중은 d−1 까지만 보므로 무관
    out = rg.tv_weighted_index(close, tv, market, window=20)
    ret, n, mw = out["index_ret"]["KOSPI"], out["index_n"]["KOSPI"], out["index_maxw"]["KOSPI"]
    assert ret.iloc[:20].isna().all()               # 20일 평균(행 0..19)이 행 20 의 "직전 20일" 가중 → 수익은 행 20 부터
    assert ret.iloc[20] == 0.0 and n.iloc[20] == 12
    assert ret.iloc[22] == pytest.approx(0.10 / 11) and n.iloc[22] == 11
    assert mw.iloc[22] == pytest.approx(1 / 11)
    # 다음 날: 종목 02 의 폭증이 20일 평균에 들어가 가중이 상한 10% 에 걸린다
    assert out["index_maxw"]["KOSPI"].iloc[23] == pytest.approx(0.10)
    assert np.isnan(out["index_ret"]["KOSDAQ"]).all()


def test_tv_index_future_trade_value_does_not_change_past():
    inp = _inputs(n=200)
    a = rg.tv_weighted_index(inp.close, inp.trade_value, inp.market)["index_ret"]
    tv2 = inp.trade_value.copy()
    tv2.iloc[150:] *= np.random.default_rng(9).uniform(0.01, 100, tv2.iloc[150:].shape)
    b = rg.tv_weighted_index(inp.close, tv2, inp.market)["index_ret"]
    pd.testing.assert_frame_equal(a.iloc[:151], b.iloc[:151])   # 행 150 의 가중은 ≤ 149 거래대금
    assert not a.iloc[151:].equals(b.iloc[151:])


def test_panel_index_sanity_on_synthetic(monkeypatch):
    monkeypatch.setattr(rg, "MIN_INDEX_N", 5)
    inp = _inputs(n=320)
    p = rg.build_regime_panel(inp)
    s = rg.index_sanity(p, since=str(inp.close.index[0].date()))
    assert (s["max_weight"] <= rg.INDEX_CAP + 1e-9).all()
    assert (s["max_abs_ret"] <= rg.RET_LIMIT).all()
    assert s["max_weight"].notna().all() and s.loc["ALL", "ok_corr"]   # ALL 은 공식 지수 없음(NaN) → 통과
    a = p[p["market"] == "ALL"].set_index("date")
    assert a["index_n"].iloc[-1] == 24 and np.isnan(a["official_ret"]).all()


def test_build_chunked_equals_unchunked():
    inp = _inputs(n=320)
    pd.testing.assert_frame_equal(rg.build_regime_panel(inp, chunk=5), rg.build_regime_panel(inp, chunk=10_000))


def test_index_requires_minimum_breadth(monkeypatch):
    monkeypatch.setattr(rg, "MIN_INDEX_N", 20)
    a = rg.build_regime_panel(_inputs(n=60))
    k = a[a["market"] == "KOSPI"]                   # 12종목 < 20 → NaN
    assert k["index_ret"].isna().all() and (k["index_n"].iloc[20:] == 12).all()


def test_idx_sign_ema20_is_ema_of_return_sign(monkeypatch):
    monkeypatch.setattr(rg, "MIN_INDEX_N", 5)
    inp = _inputs(n=120)
    a = rg.build_regime_panel(inp)
    k = a[a["market"] == "KOSPI"].set_index("date")
    want = np.sign(k["index_ret"]).ewm(span=20, adjust=False, ignore_na=True).mean().where(k["index_ret"].notna().cummax())
    assert k["idx_sign_ema20"].notna().sum() == 100
    pd.testing.assert_series_equal(k["idx_sign_ema20"], want, check_names=False)
    assert k["idx_sign_ema20"].dropna().between(-1, 1).all()


# --- 스펙트럼 유효성(리뷰 M4) ---

def test_rw_threshold_is_deterministic_and_rejects_random_walks():
    thr = rg.rw_share_threshold(256)
    assert thr == rg.rw_share_threshold(256) and 0 < thr < 1
    tab = rg.rw_threshold_table((128, 256))
    assert list(tab.index) == [128, 256] and tab.loc[256] == thr
    rng = np.random.default_rng(77)
    walk = pd.Series(np.cumsum(rng.standard_normal(1500)))
    blk = rg.spectral_block(walk, "w")
    valid = blk["w_period"].notna().sum() / blk["w_share"].notna().sum()
    assert valid < 0.25                               # 무작위 보행은 대부분 무효(평균 ~5%, 창이 겹쳐 출렁인다)
    assert (blk.loc[blk["w_period"].notna(), "w_share"] > thr).all()


def test_sine_period_survives_validity_filter():
    t = np.arange(600)
    x = pd.Series(np.sin(2 * np.pi * t / 40.0) + 0.05 * np.random.default_rng(1).standard_normal(600))
    blk = rg.spectral_block(x, "s")
    got = blk["s_period"].iloc[rg.SPEC_WINDOW - 1:]
    assert got.notna().all() and (got - 40).abs().max() < 3


# --- 실자료 위생 바(커밋된 npz) — 지수 대용 C1 수용 기준을 고정한다 ---

NPZ = Path(__file__).resolve().parents[1] / "research/logs/sepa_explore/regime/regime_panel.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="regime_panel.npz 없음")
def test_committed_panel_meets_index_sanity_bars():
    z = np.load(NPZ, allow_pickle=False)
    feats = list(z["features"])
    for col in ("index_maxw", "official_ret", "idx_sign_ema20"):
        assert col in feats, f"{col} 없음 — 옛(주식수 가중) 패널이다"
    panel = pd.DataFrame(z["values"], columns=feats)
    panel["date"] = pd.to_datetime(z["date"])
    panel["market"] = np.array(z["markets"])[z["market"]]
    s = rg.index_sanity(panel)
    assert (s["max_weight"] <= rg.INDEX_CAP + 1e-9).all(), s
    assert (s["max_abs_ret"] <= rg.RET_LIMIT).all(), s
    for m in ("KOSPI", "KOSDAQ"):
        assert s.loc[m, "n_official"] > 500 and s.loc[m, "corr_official"] >= rg.INDEX_MIN_CORR, s
