"""smallcap 부품 손계산 벡터 — 합성 패널. DB·네트워크 불요."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import smallcap_lib as L  # noqa: E402


def mk(C, *, V=None, TV=None, H=None, L_=None, RAW=None, start="2012-01-02", market=None, distress=None, excluded=None):
    C = np.atleast_2d(np.asarray(C, dtype=float))
    N, T = C.shape
    dates = pd.bdate_range(start, periods=T).to_numpy().astype("datetime64[ns]")
    return L.Panel(
        dates=dates, codes=[f"{i:05d}0" for i in range(N)],
        H=C * 1.03 if H is None else np.atleast_2d(H), L=C * 0.99 if L_ is None else np.atleast_2d(L_), C=C,
        V=np.where(np.isfinite(C), 1000.0, np.nan) if V is None else np.atleast_2d(V),
        TV=np.where(np.isfinite(C), 10000.0, np.nan) if TV is None else np.atleast_2d(TV),
        RAW=np.where(np.isfinite(C), 20000.0, np.nan) if RAW is None else np.atleast_2d(RAW),
        market=np.array(["KOSPI"] * N if market is None else market),
        excluded=np.zeros(N, dtype=bool) if excluded is None else np.asarray(excluded),
        distress=np.zeros(N, dtype=bool) if distress is None else np.asarray(distress),
    )


def flat_then(n=100, drop_at=70, drop=98.0):
    c = np.full(n, 100.0)
    c[drop_at:] = drop
    return c


@pytest.mark.parametrize("code,half", [("005930", 1), ("000660", 0), ("086520", 1), ("035720", 0)])
def test_half_of_vectors(code, half):
    # sha256 첫 바이트(셸 sha256sum): 005930 ad · 000660 50 · 086520 03 · 035720 c0
    assert L.half_of(code) == half


@pytest.mark.parametrize("name,hit", [("메리츠화재", False), ("메리츠증권", False), ("맥쿼리인프라", True), ("케이탑리츠", True),
                                      ("하나머스트스팩", True), ("대우기업인수목적1호", True), ("삼성전자", False), ("KODEX 200", True),
                                      (None, False)])
def test_name_filter(name, hit):
    assert L.name_excluded(name) is hit


def test_market_group_rejects_unknown():
    assert L.market_group("거래소") == "KOSPI"
    assert L.market_group("코스닥") == "KOSDAQ"
    with pytest.raises(ValueError):
        L.market_group("나스닥")


def test_half_spread_and_impact_by_hand():
    assert L.half_spread_bp(500.0, 20000.0, "KOSDAQ") == 50.0
    assert L.half_spread_bp(1000.0, 20000.0, "KOSDAQ") == 30.0
    assert L.half_spread_bp(5000.0, 20000.0, "KOSDAQ") == 15.0
    assert L.half_spread_bp(5000.0, 1000.0, "KOSDAQ") == 25.0   # 5원 호가 / (2×1000) = 25bp
    assert L.half_spread_bp(500.0, float("nan"), "KOSDAQ") == 50.0
    assert L.impact_bp(1.5e7, 300.0, 0.03) == pytest.approx(0.03 * np.sqrt(0.05) * 1e4)
    assert L.tick_size(60000, "KOSDAQ") == 100.0
    assert L.tick_size(60000, "KOSPI") == 100.0
    assert L.tick_size(200000, "KOSPI") == 500.0


def test_cost_one_way_by_hand():
    d = pd.Timestamp("2012-03-02").date()
    sig, adv = 0.03, 1000.0
    imp = 0.03 * np.sqrt(1.5e7 / 1e9) * 1e4
    assert L.cost_one_way("BUY", d, "KOSDAQ", adv, sig, 20000.0) == pytest.approx((1.5 + 30 + imp) * 1e-4)
    assert L.cost_one_way("SELL", d, "KOSDAQ", adv, sig, 20000.0) == pytest.approx((1.5 + 30 + 30 + imp) * 1e-4)


def test_entry_and_exit_by_hand():
    p = mk(flat_then())
    o = L.outcome(p, 0, 59)
    assert o["e"] == 60 and o["x"] == 71 and o["reason"] == "signal"
    tp = 100 * (1.03 + 0.99 + 1) / 3
    assert o["gross"] == pytest.approx(100 / tp * 0.98 * (98 * (1.03 + 0.99 + 1) / 3) / 98 - 1)
    assert o["gross"] == pytest.approx(-0.02)
    assert o["cb"] == pytest.approx((1.5 + 15) * 1e-4)
    r = np.zeros(71)
    r[70] = -0.02
    sig = np.std(r[8:71], ddof=1)
    imp = sig * np.sqrt(1.5e7 / 1e10) * 1e4
    assert o["cs"] == pytest.approx((1.5 + 30 + 15 + imp) * 1e-4)
    assert L.net(o, 2.0) == pytest.approx(0.98 * (1 - 2 * o["cs"]) / (1 + 2 * o["cb"]) - 1)


def test_close_proxy_label_uses_close():
    o = L.outcome(mk(flat_then()), 0, 59, close_fill=True)
    assert o["gross"] == pytest.approx(-0.02)
    assert o["x"] == 71


def test_limit_up_close_is_no_fill_and_threshold_changes_2015():
    c = np.full(100, 100.0)
    c[60:] = 114.0                  # +14% — ±15% 시대 문턱
    assert L.outcome(mk(c), 0, 59) == "limit"
    c2 = np.full(100, 100.0)
    c2[60:] = 113.9
    assert isinstance(L.outcome(mk(c2), 0, 59), dict)
    # 2015-06-15 뒤에는 +14% 는 체결, +29% 는 미체결
    p = mk(c, start="2015-08-03")
    assert isinstance(L.outcome(p, 0, 59), dict)
    c3 = np.full(100, 100.0)
    c3[60:] = 129.0
    assert L.outcome(mk(c3, start="2015-08-03"), 0, 59) == "limit"


def test_entry_cap_at_five_percent_of_tv():
    tv = np.full(100, 10000.0)
    tv[60] = 299.0                 # 2.99억 × 5% < 1,500만
    assert L.outcome(mk(flat_then(), TV=tv), 0, 59) == "cap"
    tv[60] = 300.0
    assert isinstance(L.outcome(mk(flat_then(), TV=tv), 0, 59), dict)
    tv[60] = np.nan
    assert L.outcome(mk(flat_then(), TV=tv), 0, 59) == "cap"


def test_locked_limit_down_sells_at_low_when_released():
    c = np.full(100, 100.0)
    c[70] = 98.0                   # 신호
    c[71] = 84.0                   # −14.3%: 못 판다
    c[72:] = 84.0                  # 풀림: 그날 저가(0.99C)
    o = L.outcome(mk(c), 0, 59)
    assert o["x"] == 72
    tp = 100 * (1.03 + 0.99 + 1) / 3
    assert o["gross"] == pytest.approx(100 / tp * 0.84 * 0.99 - 1)


def test_sell_slices_at_participation_cap():
    tv = np.full(100, 10000.0)
    tv[71] = 150.0                 # 하루 750만원 = 주문의 약 절반
    o = L.outcome(mk(flat_then(), TV=tv), 0, 59)
    assert o["x"] == 72
    val = 100 / (100 * (1.03 + 0.99 + 1) / 3) * 0.98
    take = 7.5e6 / (1.5e7 * val)
    ratio = (1.03 + 0.99 + 1) / 3
    assert o["gross"] == pytest.approx(take * val * ratio + (1 - take) * val * ratio - 1)


def test_seam_keeps_loss_and_zeroes_gain():
    c = np.full(100, 100.0)
    c[65:] = 150.0                 # 제한폭 초과 +50% 이음매: 이익 0
    p = mk(c)
    assert p.seam[0, 65]
    assert p.Cp[0, 70] == pytest.approx(p.Cp[0, 60])
    assert p.Cb[0, 70] == pytest.approx(p.Cb[0, 60])
    c2 = np.full(100, 100.0)
    c2[65:] = 50.0                 # −50% 이음매: 포지션은 손실 유지, 벤치는 그 칸을 뺀다
    p2 = mk(c2)
    assert p2.Cp[0, 70] / p2.Cp[0, 60] == pytest.approx(0.5)
    assert p2.Cb[0, 70] / p2.Cb[0, 60] == pytest.approx(1.0)


def test_reopen_day_is_seam():
    c = np.full(100, 100.0)
    v = np.full(100, 1000.0)
    v[64:66] = 0.0
    c[66:] = 110.0
    p = mk(c, V=v)
    assert p.seam[0, 66]
    assert p.Cp[0, 80] / p.Cp[0, 60] == pytest.approx(1.0)


def test_series_end_uses_last_close_without_limit_lock_in_tail():
    c = np.full(100, 100.0)
    c[80:] = np.nan
    c[75:80] = [80, 60, 40, 25, 15]          # 정리매매: 제한폭 넘는 하락 그대로
    p = mk(c, distress=[True])
    assert p.last[0] == 79 and p.tail[0, 79] and not p.seam[0, 76]
    o = L.outcome(p, 0, 59)
    assert o["reason"] == "signal"           # 75 신호, 76 에 팔린다(잠김 적용 안 함)
    assert o["x"] == 76
    q = mk(np.where(np.arange(100) < 80, 100.0, np.nan), distress=[True])
    o2 = L.outcome(q, 0, 59)
    assert o2["reason"] == "series_end" and o2["x"] == 79 and o2["dist"]
    tp = 100 * (1.03 + 0.99 + 1) / 3
    assert o2["gross"] == pytest.approx(100 / tp - 1)
    assert L.net(o2, 2.0, dist100=True) == pytest.approx(0.0 * 1 - 1.0)


def test_tail_over_limit_gain_is_zero_but_loss_stays():
    c = np.full(100, 100.0)
    c[80:] = np.nan
    c[77:80] = [40.0, 80.0, 40.0]            # 꼬리: −60%(유지), +100%(이익 0), −50%(유지)
    p = mk(c)
    assert p.tail[0, 78]
    assert p.Cp[0, 79] / p.Cp[0, 76] == pytest.approx(0.4 * 1.0 * 0.5)
    assert p.Cb[0, 79] / p.Cb[0, 76] == pytest.approx(0.4 * 1.0 * 0.5)


def test_bad_adjusted_price_leaves_universe():
    c = np.full(100, 100.0)
    p = mk(c, RAW=np.full(100, 6000.0))      # 수정가/원가 1.7%
    assert not p.universe((300.0, 20000.0)).any()
    q = mk(c, RAW=np.full(100, 4000.0))      # 2.5%: 남는다
    assert q.universe((300.0, 20000.0)).any()


def test_bench_buy_hold_with_delisted_flat_and_dist100():
    a = np.full(100, 100.0)
    a[70:] = 200.0 / 1  # 이음매라 0 — 아래 b 로 본다
    a = np.linspace(100, 110, 100)
    b = np.linspace(100, 90, 100)
    b[80:] = np.nan
    p = mk(np.vstack([a, b]), distress=[False, True])
    u = np.ones_like(p.traded)
    be = L.Bench(p, u)
    want = ((a[90] / a[60] - 1) + (b[79] / b[60] - 1)) / 2
    assert be.ret(59, 60, 90) == pytest.approx(want)
    be100 = L.Bench(p, u, dist100=True)
    assert be100.ret(59, 60, 90) == pytest.approx(((a[90] / a[60] - 1) + (-1.0)) / 2)


def test_tv_is_excluded_from_its_own_adv_and_nan_leaves_universe():
    tv = np.full(100, 1000.0)
    tv[50] = 9000.0
    tv[70] = np.nan
    p = mk(flat_then(), TV=tv)
    assert p.ADV[0, 50] == pytest.approx(1000.0)
    assert p.ADV[0, 51] == pytest.approx((19 * 1000.0 + 9000.0) / 20)
    u = p.universe((300.0, 2000.0))
    assert u[0, 60] and not u[0, 70]
    assert not p.universe((3000.0, 4000.0))[0, 60]
    assert not mk(flat_then(), TV=tv, excluded=[True]).universe((300.0, 2000.0))[0, 60]


def test_hvnm_event_needs_new_max_small_move_and_no_action():
    n = 120
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    tv = np.full(n, 1000.0)
    t_ev = 100
    tv[t_ev] = 5000.0
    c[t_ev] = c[t_ev - 1] * 1.0005
    c[t_ev + 1:] = c[t_ev] * np.cumprod(1 + rng.normal(0, 0.01, n - t_ev - 1))
    p = mk(c, TV=tv, RAW=c * 100)
    ev = L.hvnm_events(p, p.universe((300.0, 2000.0)))
    assert list(ev.t) == [t_ev]
    assert ev["rank"][0] == pytest.approx(5.0)
    # 같은 거래대금이라도 3σ 움직인 날은 사건이 아니다
    c2 = c.copy()
    c2[t_ev:] = c2[t_ev:] * 1.05
    p2 = mk(c2, TV=tv, RAW=c2 * 100)
    assert t_ev not in set(L.hvnm_events(p2, p2.universe((300.0, 2000.0))).t)
    # 수정 사건 칸(수정가/원가 계단 > 1%)도 아니다
    raw = c * 100
    raw[t_ev:] = raw[t_ev:] * 0.5
    p3 = mk(c, TV=tv, RAW=raw)
    assert p3.action[0, t_ev]
    assert t_ev not in set(L.hvnm_events(p3, p3.universe((300.0, 2000.0))).t)


def test_book_slots_rank_held_and_cash():
    N = 3
    c = np.vstack([flat_then(), flat_then(), flat_then()])
    p = mk(c)
    u = np.ones_like(p.traded)
    be = L.Bench(p, u)
    ev = pd.DataFrame({"c": [0, 1, 2, 0], "t": [59, 59, 59, 62], "rank": [1.0, 3.0, 2.0, 9.0]})
    fills, fails = L.run_book(p, ev, be, slots=2)
    assert list(fills.c) == [1, 2]       # 같은 날 rank 큰 순, 2슬롯, 넘치면 버림
    assert fails["slots"] == 2
    fills2, fails2 = L.run_book(p, ev, be, slots=N)
    assert fails2["held"] == 1
    assert len(fills2) == 3
    assert fills2.net.iloc[0] == pytest.approx(L.net(L.outcome(p, 0, 59), 2.0))


def test_event_study_excludes_re_events_while_held():
    p = mk(flat_then())
    be = L.Bench(p, np.ones_like(p.traded))
    ev = pd.DataFrame({"c": [0, 0, 0], "t": [59, 62, 75], "rank": [1.0, 1.0, 1.0]})
    df, fails = L.event_study(p, ev, be)
    assert list(df.t) == [59, 75]
    assert fails["held"] == 1


def test_random_events_keep_month_counts_and_stay_in_universe():
    p = mk(np.vstack([flat_then(), flat_then()]))
    u = np.ones_like(p.traded)
    u[1, :] = False
    ev = pd.DataFrame({"c": [0, 0], "t": [10, 40], "rank": [1.0, 1.0]})
    out = L.random_events(p, u, ev, np.random.default_rng(0))
    assert set(out.c) == {0}
    months = p.dates.astype("datetime64[M]")
    assert set(months[out.t]) == {months[10], months[40]}


def test_shifted_events_keep_gaps_inside_universe():
    p = mk(flat_then())
    u = np.ones_like(p.traded)
    ev = pd.DataFrame({"c": [0, 0], "t": [10, 13], "rank": [1.0, 2.0]})
    out = L.shifted_events(p, u, ev, np.random.default_rng(0))
    diff = sorted((out.t.iloc[1] - out.t.iloc[0]) % 100 for _ in [0])
    assert diff[0] in (3, 97)


def test_book_daily_total_equals_fill_net():
    p = mk(flat_then())
    be = L.Bench(p, np.ones_like(p.traded))
    ev = pd.DataFrame({"c": [0], "t": [59], "rank": [1.0]})
    fills, _ = L.run_book(p, ev, be)
    day = L.book_daily(p, fills)
    assert day.pnl.sum() == pytest.approx(L.ORDER_WON * fills.net.iloc[0])
    assert day.exp_kospi.iloc[60] == pytest.approx(L.ORDER_WON)
    assert day.exp_kosdaq.sum() == 0


def test_ols_hac_constant_alpha():
    y = np.array([0.01, 0.02, 0.0, 0.03, 0.01, 0.02, 0.0, 0.01, 0.02, 0.03])
    out = L.ols_hac(y, np.arange(10.0)[:, None] * 0 + np.random.default_rng(0).normal(size=(10, 1)))
    assert out["n"] == 10 and np.isfinite(out["t_alpha"])


def test_assert_explore_refuses_locked_code():
    L.assert_explore(["000660"])
    with pytest.raises(AssertionError):
        L.assert_explore(["005930"])
