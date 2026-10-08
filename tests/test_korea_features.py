"""스윙 3호 한국 위생 피처 — 유니버스 필터·ADV 창·섹터 주도 상위 1/3·상한가 시가·비용표(합성 데이터)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from swing_it.features.korea import (
    cost_rt, cost_schedule, exclusion_reason, is_limit_up_open, limit_up_price, listing_age,
    normalize_market, sector_leadership, sector_returns, universe_mask)


def _days(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _meta(rows: list[tuple[str, str, str, "str | None"]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["code", "name", "market", "sector"]).set_index("code")
    df["market"] = df["market"].map(normalize_market)
    df["excluded"] = [exclusion_reason(c, n) for c, n in zip(df.index, df["name"])]
    return df


# --- 보통주 판별 ---

@pytest.mark.parametrize("code, name, reason", [
    ("005930", "삼성전자", None),
    ("005935", "삼성전자우", "preferred"),
    ("00088K", "한화3우B", "preferred"),
    ("0155E0", "해치텍", None),            # 새 형식 영숫자 보통주
    ("115960", "연우", None),               # '우' 로 끝나도 끝자리 0 이면 보통주
    ("458650", "성우", None),
    ("404950", "DB금융스팩10호", "spac"),
    ("347140", "케이프이에스제4호", "fund"),  # 이름에 '스팩' 없는 SPAC — N호 규칙이 잡는다
    ("083350", "동북아10호", "fund"),
    ("204210", "JK리버스톤리츠", "reit"),
    ("369370", "블리츠웨이엔터테인먼트", None),  # '리츠' 부분일치 — 끝맺음만 본다
    ("069500", "KODEX 200", "etf_etn"),
    ("123450", None, "unknown"),
])
def test_exclusion_reason(code, name, reason):
    assert exclusion_reason(code, name) == reason


@pytest.mark.parametrize("raw, norm", [("거래소", "KOSPI"), ("유가증권", "KOSPI"), ("코스닥", "KOSDAQ"),
                                       ("코넥스", "KONEX"), ("KOSDAQ", "KOSDAQ"), (None, None), ("??", None)])
def test_normalize_market(raw, norm):
    assert normalize_market(raw) == norm


# --- 유니버스 ---

def test_listing_age_counts_from_first_bar_including_halts():
    d = _days(6)
    close = pd.DataFrame({"A": [np.nan, np.nan, 10, np.nan, 11, 12], "B": [np.nan] * 6}, index=d)
    age = listing_age(close)
    assert age["A"].tolist()[2:] == [1.0, 2.0, 3.0, 4.0]      # 정지일(NaN 종가)도 상장 일수로 센다
    assert age["A"].iloc[:2].isna().all()
    assert age["B"].isna().all()


def test_universe_filters_each_rule():
    n = 30
    d = _days(n)
    codes = ["000010", "000015", "000020", "000030", "000040", "000050", "000060"]
    close = pd.DataFrame(5_000.0, index=d, columns=codes)
    tv = pd.DataFrame(30_000.0, index=d, columns=codes)
    close["000030"] = 900.0                        # 1,000원 미만
    tv["000040"] = 10_000.0                        # ADV 미달
    meta = _meta([
        ("000010", "보통", "거래소", "화학"),
        ("000015", "보통우", "거래소", "화학"),     # 우선주(코드 끝자리)
        ("000020", "OO스팩1호", "코스닥", "금융"),  # 스팩
        ("000030", "싼주", "코스닥", "화학"),
        ("000040", "얇은주", "코스닥", "화학"),
        ("000050", "코넥스주", "코넥스", None),     # 시장 제외
        # 000060 은 메타에 없다 → 제외
    ])
    u = universe_mask(close, tv, meta, min_history=5, adv_window=3)
    last = u.iloc[-1]
    assert last.to_dict() == {"000010": True, "000015": False, "000020": False, "000030": False,
                              "000040": False, "000050": False, "000060": False}
    # 이력: 5거래일째부터(첫날 = 1)
    assert not u["000010"].iloc[3] and u["000010"].iloc[4]


def test_universe_raw_close_drives_price_floor():
    d = _days(10)
    close = pd.DataFrame({"000010": [500.0] * 10}, index=d)        # 뒤에 1:10 분할 → 조정가는 500원
    raw = pd.DataFrame({"000010": [5_000.0] * 10}, index=d)        # 그날 실제 호가는 5,000원
    tv = pd.DataFrame({"000010": [50_000.0] * 10}, index=d)
    meta = _meta([("000010", "보통", "거래소", "화학")])
    kw = dict(min_history=1, adv_window=1)
    assert not universe_mask(close, tv, meta, **kw).to_numpy().any()
    assert universe_mask(close, tv, meta, raw_close=raw, **kw).to_numpy().all()


def test_adv_window_is_trailing_including_today_and_needs_full_window():
    d = _days(25)
    tv = pd.Series(10_000.0, index=d)
    tv.iloc[20] = 10_000.0 + 20 * 10_000.0      # 21번째 날 거래대금 급증 → 그날 평균 정확히 20,000
    close = pd.DataFrame({"000010": 2_000.0}, index=d)
    meta = _meta([("000010", "보통", "거래소", "화학")])
    u = universe_mask(close, tv.to_frame("000010"), meta, min_history=1)
    assert not u["000010"].iloc[:20].any()        # 창(20)이 차기 전·급증 전엔 미달
    assert u["000010"].iloc[20]                   # 급증 당일 포함(d 종가까지 — 시점 안전)
    assert u["000010"].iloc[21:].all()            # 20일 동안 창 안에 남는다
    tv2 = tv.copy()
    tv2.iloc[10] = np.nan                         # 거래정지 → 그 날을 포함하는 창은 값 없음
    u2 = universe_mask(close, tv2.to_frame("000010"), meta, min_history=1)
    assert not u2["000010"].iloc[20]


def test_universe_does_not_peek_future():
    d = _days(40)
    tv = pd.DataFrame({"000010": 1_000.0}, index=d)
    close = pd.DataFrame({"000010": 2_000.0}, index=d)
    meta = _meta([("000010", "보통", "거래소", "화학")])
    base = universe_mask(close, tv, meta, min_history=1)
    tv2 = tv.copy()
    tv2.iloc[30:] = 1e9                           # 미래만 바꾼다
    alt = universe_mask(close, tv2, meta, min_history=1)
    pd.testing.assert_frame_equal(base.iloc[:30], alt.iloc[:30])


# --- K 섹터 주도 ---

def _sector_panel(n_sectors: int = 6, per: int = 3, window: int = 5):
    d = _days(window + 2)
    cols, sec, growth = [], {}, {}
    for s in range(n_sectors):
        for j in range(per):
            c = f"{s:02d}{j:03d}0"
            cols.append(c)
            sec[c] = f"S{s}"
            growth[c] = 1.0 + 0.01 * s            # 섹터 번호가 클수록 강하다
    close = pd.DataFrame({c: 100.0 * growth[c] ** np.arange(len(d)) for c in cols}, index=d)
    return close, pd.Series(sec)


def test_sector_leadership_top_third():
    close, sec = _sector_panel(n_sectors=6)
    k = sector_leadership(close, sec, window=5)
    last = k.iloc[-1]
    on = last.fillna(False).astype(bool)
    leaders = set(sec[on.reindex(sec.index)])
    assert leaders == {"S5", "S4"}                # 6개 중 상위 1/3 = 2개
    assert last.notna().all()
    assert k.iloc[:5].isna().all().all()          # 창(5일) 전엔 정의 안 됨


def test_sector_leadership_na_for_unknown_and_thin_sectors():
    close, sec = _sector_panel(n_sectors=6)
    close["999990"] = 100.0                      # 섹터 모름
    thin = sec.copy()
    thin = thin[~((thin == "S5") & (thin.index != "050000"))]  # S5 를 1종목으로
    k = sector_leadership(close, thin, window=5, min_members=3)
    last = k.iloc[-1]
    assert pd.isna(last["999990"])
    assert pd.isna(last["050000"])              # 구성 1 < 3 → 순위 제외(NA)
    on = last.fillna(False).astype(bool)
    leaders = set(thin[on.reindex(thin.index)])
    assert leaders == {"S4"}                      # 남은 5개 중 상위 5/3 → 1개


def test_sector_returns_use_universe_members_only():
    close, sec = _sector_panel(n_sectors=3)
    uni = pd.DataFrame(True, index=close.index, columns=close.columns)
    boom = "000000"
    close[boom] = close[boom] * np.r_[np.ones(len(close) - 1), 10.0]   # S0 한 종목이 마지막 날 10배
    full = sector_returns(close, sec, window=5, min_members=1)
    uni.loc[:, boom] = False
    excl = sector_returns(close, sec, universe=uni, window=5, min_members=1)
    assert full["S0"].iloc[-1] > excl["S0"].iloc[-1]
    assert excl["S0"].iloc[-1] == pytest.approx(1.0 ** 5 - 1.0)


# --- 상한가 ---

def test_limit_up_price_matches_core():
    from krx_quant_core.market.limits import limit_up_price as core

    rng = np.random.default_rng(0)
    prices = np.r_[rng.integers(1_000, 900_000, 3_000), [1_538, 1_539, 3_846, 15_384, 38_461, 153_846, 384_615]]
    ours = limit_up_price(prices.astype(float))
    assert np.array_equal(ours, np.array([core(float(p)) for p in prices]))


def test_is_limit_up_open():
    prev = pd.Series([10_000.0, 10_000.0, 10_000.0, 7_777.0, np.nan, 10_000.0])
    lim = limit_up_price(prev)
    opn = pd.Series([13_000.0, 12_950.0, 12_000.0, float(lim.iloc[3]), 13_000.0, np.nan])
    assert is_limit_up_open(opn, prev, exact=True).tolist() == [True, False, False, True, False, False]
    # 비율 판정: +29.3% 이상은 상한가로 본다(조정가·옛 호가표에서 격자가 깨져도 잡히게)
    assert is_limit_up_open(opn, prev).tolist() == [True, True, False, True, False, False]
    # 분할조정으로 격자가 깨진 상한가(실제 13,000/10,000 을 1/3 로 조정)도 잡는다
    assert bool(is_limit_up_open(np.array([13_000 / 3]), np.array([10_000 / 3]))[0])


# --- 비용 ---

def test_cost_rt_baseline_and_schedule():
    assert cost_rt("2026-03-02", "KOSPI") == pytest.approx(0.00415)    # 레포 기준선 41.5bp
    assert cost_rt(date(2026, 3, 2), "코스닥") == pytest.approx(0.00415)
    assert cost_rt("2024-06-03", "KOSDAQ") == pytest.approx(0.0018 + 0.0003 + 0.00185)
    assert cost_rt("2025-06-02", "KOSPI") == pytest.approx(0.0015 + 0.0003 + 0.00185)
    assert cost_rt("2018-01-02", "KOSPI") > cost_rt("2023-06-01", "KOSPI")
    sch = cost_schedule()
    assert set(sch["market"]) == {"KOSPI", "KOSDAQ"}
    # 합계 세율은 두 시장이 늘 같다 → 시장 분류(PIT 아님)가 틀려도 비용은 안 틀린다
    assert sch.groupby(sch["effective"].fillna("min"))["tax_bp"].nunique().eq(1).all()
