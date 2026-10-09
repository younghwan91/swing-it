"""시장 국면 패널(``swing_it.features.regime``) — 룩어헤드 없음·돌파 성공률 PIT·스펙트럼 인과성 (합성 데이터만)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing_it.features import regime as rg
from swing_it.features.minervini import market_breadth


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2019-01-01", periods=n)


def _inputs(n: int = 700, k: int = 8, seed: int = 0) -> rg.RegimeInputs:
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
    shares = f(np.tile(rng.integers(1_000, 10_000, k).astype(float), (n, 1)))
    net = f(rng.integers(-50, 50, (n, k)).astype(float))
    off = pd.DataFrame({"KOSPI": close[:, 0], "KOSDAQ": close[:, 1]}, index=idx)
    return rg.RegimeInputs(close=f(close), high=f(high), low=f(low), volume=f(vol), market=market, shares=shares,
                           flow_net=net, flow_volume=f(vol), official=off)


def _truncate(inp: rg.RegimeInputs, m: int) -> rg.RegimeInputs:
    cut = lambda df: None if df is None else df.iloc[:m]  # noqa: E731
    return rg.RegimeInputs(close=cut(inp.close), high=cut(inp.high), low=cut(inp.low), volume=cut(inp.volume),
                           market=inp.market, shares=cut(inp.shares), flow_net=cut(inp.flow_net),
                           flow_volume=cut(inp.flow_volume), official=cut(inp.official))


# --- 전체 패널: 뒤를 잘라도 앞 행은 그대로(룩어헤드 없음) ---

def test_panel_rows_do_not_depend_on_future_rows():
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

def test_cap_weighted_index_uses_lagged_weights():
    idx = _days(3)
    close = pd.DataFrame({"A": [100.0, 110.0, 110.0], "B": [100.0, 90.0, 99.0]}, index=idx)
    shares = pd.DataFrame({"A": [3.0, 3.0, 3.0], "B": [1.0, 1.0, 1.0]}, index=idx)
    market = pd.Series({"A": "KOSPI", "B": "KOSPI"})
    r = rg.cap_weighted_index(close, shares, market)
    assert np.isnan(r.loc[idx[0], "KOSPI"])
    # d1: 전일 시총 A 300, B 100 → (0.75·0.10 + 0.25·(−0.10)) = 0.05
    assert r.loc[idx[1], "KOSPI"] == pytest.approx(0.05)
    # d2: 전일 시총 A 330, B 90 → (330·0 + 90·0.1)/420
    assert r.loc[idx[2], "KOSPI"] == pytest.approx(9.0 / 420)
    assert np.isnan(r.loc[idx[1], "KOSDAQ"])


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

def test_shares_asof_uses_knowledge_date():
    days = _days(5)
    sh = pd.DataFrame({"code": ["A", "A"], "date": [days[0], days[1]],
                       "shares_outstanding": [100, 200], "knowledge_date": [days[0], days[3]]})
    w = rg.shares_asof(sh, days, pd.Index(["A"]))
    assert list(w["A"]) == [100.0, 100.0, 100.0, 200.0, 200.0]


def test_coverage_by_year_is_fraction():
    inp = _inputs(n=520)
    cov = rg.coverage_by_year(rg.build_regime_panel(inp))
    assert cov.index.name == "year" and ((cov >= 0) & (cov <= 1)).all().all()
    assert cov["breadth_ma50"].iloc[-1] == 1.0


# --- 자료 사건 가드 ---

def test_shares_asof_drops_implausible_counts():
    days = _days(3)
    sh = pd.DataFrame({"code": ["A", "B"], "date": [days[0], days[0]],
                       "shares_outstanding": [5.9e9, 7.4e13], "knowledge_date": [None, None]})
    w = rg.shares_asof(sh, days, pd.Index(["A", "B"]))
    assert w["A"].iloc[-1] == 5.9e9 and np.isnan(w["B"].iloc[-1])


def test_valid_returns_excludes_beyond_price_limit():
    idx = _days(3)
    close = pd.DataFrame({"A": [100.0, 129.0, 20.0]}, index=idx)
    r = rg.valid_returns(close)
    assert r["A"].iloc[1] == pytest.approx(0.29) and np.isnan(r["A"].iloc[2])
    # 지수 대용에도 반영: 자료 사건 종목만 있는 날은 NaN
    shares = pd.DataFrame({"A": 1.0}, index=idx)
    ret = rg.cap_weighted_index(close, shares, pd.Series({"A": "KOSPI"}))
    assert np.isnan(ret["KOSPI"].iloc[2]) and ret["KOSPI"].iloc[1] == pytest.approx(0.29)


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


def test_shares_asof_ratio_puts_records_in_adjusted_basis():
    """2017 삼성전자: 기록 1.28억 주(분할 전) × 원종가/조정종가(= 50) → 조정 기준 64억 주. 기록 사이에 분할이 있어도 이어진다."""
    days = _days(6)
    raw = pd.DataFrame({"A": [2_000_000.0] * 4 + [40_000.0] * 2}, index=days)      # 4일째(금) 뒤 50:1 분할
    adj = pd.DataFrame({"A": [40_000.0] * 6}, index=days)
    ratio = raw / adj
    sh = pd.DataFrame({"code": ["A"], "date": [days[0]], "shares_outstanding": [1.28e8], "knowledge_date": [days[1]]})
    w = rg.shares_asof(sh, days, pd.Index(["A"]), ratio=ratio)
    assert np.isnan(w["A"].iloc[0]) and w["A"].iloc[1] == pytest.approx(6.4e9)
    assert np.allclose(w["A"].iloc[1:], 6.4e9)
    assert (adj["A"] * w["A"]).iloc[-1] == pytest.approx(40_000.0 * 6.4e9)           # = 실제 시총
    # 기준일이 주말(분할 전 토요일)이면 직전 거래일(금) 비율 50
    sh2 = pd.DataFrame({"code": ["A"], "date": [days[3] + pd.Timedelta(days=1)], "shares_outstanding": [1.28e8],
                        "knowledge_date": [None]})
    w2 = rg.shares_asof(sh2, days, pd.Index(["A"]), ratio=ratio)
    assert w2["A"].iloc[-1] == pytest.approx(6.4e9)


def test_build_chunked_equals_unchunked_and_reports_index_n():
    inp = _inputs(n=320)
    a = rg.build_regime_panel(inp, chunk=3)
    b = rg.build_regime_panel(inp, chunk=10_000)
    pd.testing.assert_frame_equal(a, b)
    assert (a.loc[a["market"] == "ALL", "index_n"].iloc[1:] == 8).all()


def test_index_requires_minimum_breadth(monkeypatch):
    monkeypatch.setattr(rg, "MIN_INDEX_N", 5)
    inp = _inputs(n=60, k=8)
    inp.shares.iloc[:, 4:] = np.nan          # 4종목만 가중치 → 지수 NaN
    a = rg.build_regime_panel(inp)
    assert a.loc[a["market"] == "ALL", "index_ret"].isna().all()
    assert (a.loc[a["market"] == "ALL", "index_n"].iloc[1:] == 4).all()
