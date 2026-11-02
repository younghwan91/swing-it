"""thinband 슬롯 북 손계산 벡터 — 합성 패널. DB·네트워크 불요."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import smallcap_lib as L  # noqa: E402
import thinband_lib as B  # noqa: E402


def mk(C, *, V=None, TV=None, RAW=None, H=None, Lo=None, start="2017-01-02", distress=None, market=None):
    C = np.atleast_2d(np.asarray(C, dtype=float))
    N, T = C.shape
    dates = pd.bdate_range(start, periods=T).to_numpy().astype("datetime64[ns]")
    return L.Panel(dates=dates, codes=[f"{i:05d}0" for i in range(N)],
                   H=C if H is None else np.atleast_2d(H), L=C if Lo is None else np.atleast_2d(Lo), C=C,
                   V=np.where(np.isfinite(C), 1000.0, np.nan) if V is None else np.atleast_2d(V),
                   TV=np.where(np.isfinite(C), 1000.0, np.nan) if TV is None else np.atleast_2d(TV),
                   RAW=np.where(np.isfinite(C), np.nan_to_num(C) * 1.0, np.nan) if RAW is None else np.atleast_2d(RAW),
                   market=np.array(["KOSPI"] * N if market is None else market), excluded=np.zeros(N, dtype=bool),
                   distress=np.zeros(N, dtype=bool) if distress is None else np.asarray(distress), sig_start="2017-02-13")


def ctx(p, cfg=None, doom=None, up=False):
    T = p.T
    idx = {"KOSPI": np.full(T, 100.0) if not up else np.linspace(100, 110, T), "KOSDAQ": np.full(T, 100.0)}
    return B.Ctx(p, idx, cfg or B.Cfg(band=(300.0, 2000.0)), doom)


def one_name(n=130, **kw):
    return mk(np.full(n, 20000.0), **kw)


def test_periods_merge_remainder():
    assert B.periods_of(0, 122, 60) == [(0, 60), (60, 121)]
    assert B.periods_of(0, 160, 60) == [(0, 60), (60, 120), (120, 159)]
    assert B.periods_of(10, 130, 60, 10) == [(20, 80), (80, 129)]
    # 실제 표본 모양: 1,802행이면 30기간
    assert len(B.periods_of(0, 1802, 60)) == 30


def test_tick_reform_dates():
    from datetime import date
    assert L.tick_size(3000, "KOSDAQ", date(2023, 1, 24)) == 5
    assert L.tick_size(1500, "KOSDAQ", date(2023, 1, 24)) == 5
    assert L.tick_size(1500, "KOSDAQ", date(2023, 1, 25)) == 1
    assert L.tick_size(15000, "KOSDAQ", date(2023, 1, 25)) == 10
    assert L.tick_size(15000, "KOSDAQ", date(2023, 1, 24)) == 50
    assert L.tick_size(150000, "KOSPI", date(2023, 1, 25)) == 100
    assert L.tick_size(150000, "KOSPI") == 500
    assert L.half_spread_bp(5000.0, 1500.0, "KOSDAQ", date(2023, 1, 25)) == 15.0


def test_buy_splits_over_rows_at_five_percent():
    tv = np.full(130, 100.0)        # 하루 5% = 500만원 → 3행
    p = one_name(TV=tv)
    u = p.universe((50.0, 2000.0))
    assert u[0, p.sig_t0]
    c = B.Ctx(p, {"KOSPI": np.full(p.T, 100.0)}, B.Cfg(band=(50.0, 2000.0)))
    run = B.simulate(c, np.random.default_rng(0))
    a = c.periods[0][0]
    assert run.expo[a + 1] == 0.0
    assert run.expo[a + 2] == pytest.approx(5e6)
    assert run.expo[a + 3] == pytest.approx(1e7)
    assert run.expo[a + 4] == pytest.approx(1.5e7)
    assert run.cost[a + 1] > 0 and run.cost[a + 3] > 0 and run.cost[a + 4] == 0.0


def test_buy_gives_up_after_ten_rows_and_rebalances_next_grid():
    tv = np.full(130, 10.0)         # 하루 50만원 → 10행에 500만원만
    p = one_name(TV=tv)
    c = B.Ctx(p, {"KOSPI": np.full(p.T, 100.0)}, B.Cfg(band=(5.0, 2000.0)))
    run = B.simulate(c, np.random.default_rng(0))
    a, b = c.periods[0]
    assert run.expo[a + 11] == pytest.approx(5e6)
    assert run.expo[b] == pytest.approx(5e6)     # 그 뒤 채우지 않는다


def test_buy_limit_up_day_skipped_old_rule_is_one_shot():
    c0 = np.full(130, 20000.0)
    p0 = one_name()
    a = B.periods_of(p0.sig_t0, p0.T)[0][0]
    c0[a + 1:] = 25000.0              # a+1 +25%: 2015 뒤 문턱(29%) 아래라 체결
    run = B.simulate(ctx(mk(c0)), np.random.default_rng(0))
    assert run.expo[a + 2] > 0
    c1 = np.full(130, 20000.0)
    c1[a + 1:] = 26000.0              # +30%: 그날 못 산다, 다음 날 산다
    run1 = B.simulate(ctx(mk(c1)), np.random.default_rng(0))
    assert run1.cost[a + 1] == 0.0 and run1.cost[a + 2] > 0
    # 옛 규칙: 그 하루만, 재시도 없음
    runo = B.simulate(ctx(mk(c1), B.Cfg(band=(300.0, 2000.0), entry="old")), np.random.default_rng(0))
    assert runo.cost[a + 1:a + 40].sum() == 0.0


def test_rebalance_trades_only_the_difference_and_costs_are_by_hand():
    c = np.full(130, 20000.0)
    p0 = one_name()
    (a, b), (a2, _) = B.periods_of(p0.sig_t0, p0.T)[:2]
    c[a + 10:] = 22000.0              # +10%: 1,650만원
    p = mk(c)
    run = B.simulate(ctx(p), np.random.default_rng(0))
    # 첫 매도일 a2+1 에 150만원 매도
    assert run.expo[a2] == pytest.approx(1.5e7 * 1.1)
    assert run.cost[a2 + 1] == pytest.approx(1.5e6 * L.cost_one_way("SELL", p.day_py[a2 + 1], "KOSPI", 1000.0, p.SIG20[0, a2 + 1], 22000.0, 1.5e6))
    assert run.expo[a2 + 2] == pytest.approx(1.5e7)


def test_exit_when_leaving_band_and_slot_refilled():
    tv = np.full((2, 130), 1000.0)
    p0 = one_name()
    (a, b), (a2, _) = B.periods_of(p0.sig_t0, p0.T)[:2]
    tv[0, a2 - 20:] = 3000.0          # 종목 0 은 a2 에 대역 밖
    p = mk(np.full((2, 130), 20000.0), TV=tv)
    c = ctx(p)
    assert c.u[0, a] and not c.u[0, a2] and c.u[1, a2]
    run = B.simulate(c, np.random.default_rng(1))
    assert run.expo[a + 5] == pytest.approx(3e7)
    assert run.expo[a2 + 5] == pytest.approx(1.5e7)   # 종목 1 만 남는다


def test_halted_name_holds_slot_no_trade():
    c = np.full((1, 130), 20000.0)
    v = np.full((1, 130), 1000.0)
    p0 = one_name()
    (a, b), (a2, b2) = B.periods_of(p0.sig_t0, p0.T)[:2]
    c[0, a + 10:] = 22000.0
    v[0, a2 - 5:a2 + 5] = 0.0
    p = mk(c, V=v, RAW=c.copy())
    run = B.simulate(ctx(p), np.random.default_rng(0))
    assert run.halted[a2] > 0
    assert run.cost[a2 + 1:a2 + 5].sum() == 0.0      # 정지 중 복원 없음
    assert run.expo[a2 + 3] == pytest.approx(run.expo[a2])


def test_series_end_cashes_at_last_close_and_frees_slot():
    c = np.full(130, 20000.0)
    p0 = one_name()
    a = B.periods_of(p0.sig_t0, p0.T)[0][0]
    c[a + 30:] = np.nan
    p = mk(c)
    assert p.last[0] == a + 29
    run = B.simulate(ctx(p), np.random.default_rng(0))
    assert run.cost[a + 29] > 0
    assert run.expo[a + 40] == 0.0
    assert any(h[2] == a + 29 for h in run.holds)


def test_doom_name_written_off_at_sample_end_and_holds_slot():
    c = np.full(130, 20000.0)
    v = np.full(130, 1000.0)
    p0 = one_name()
    a = B.periods_of(p0.sig_t0, p0.T)[0][0]
    v[a + 30:] = 0.0                  # 거래 정지 뒤 표본 끝까지
    p = mk(c, V=v, distress=[True])
    doom = np.array([True])
    run = B.simulate(ctx(p, doom=doom), np.random.default_rng(0))
    assert run.doom_writeoff == 1
    assert run.gross[-1] == pytest.approx(-1.5e7)
    assert run.halted[a + 50] == pytest.approx(1.5e7)
    # 비부실이면 마지막 값, 수를 센다
    run2 = B.simulate(ctx(p), np.random.default_rng(0))
    assert run2.doom_writeoff == 0
    assert run2.censored_halted_nondistress + len(run2.holds) >= 1


def test_doom_mask_by_kind_date_and_series_end():
    c = np.full((2, 130), 20000.0)
    v = np.full((2, 130), 1000.0)
    v[0, 100:] = 0.0
    v[1, 100:] = 0.0
    p = mk(c, V=v, distress=[True, False])
    kind = pd.DataFrame({"code": ["000000", "000010"], "delist_date": pd.to_datetime(["2024-05-01", "2024-05-01"]), "reason": ["x", "합병"]})
    assert B.doom_mask(p, kind).tolist() == [True, False]
    kind2 = kind.assign(delist_date=pd.to_datetime(["2017-03-01", "2024-05-01"]))
    assert B.doom_mask(p, kind2).tolist() == [False, False]


def test_period_series_total_matches_recs_and_index_active():
    c = np.linspace(20000.0, 24000.0, 130)
    p = mk(c)
    x = ctx(p, up=True)
    run = B.simulate(x, np.random.default_rng(0))
    s = B.period_series(x, run, 2.0)
    den = B.SLOTS * L.ORDER_WON
    tot = sum(r[2] - 2.0 * r[3] for r in run.recs) / den
    assert s["net"].sum() + 0 == pytest.approx(tot - 0, abs=1e-9)
    assert (s["active"] < s["net"]).all()      # 지수가 오르니 능동이 작다


def test_cost_multiple_does_not_change_path():
    p = mk(np.linspace(20000.0, 24000.0, 130))
    x = ctx(p)
    r = B.simulate(x, np.random.default_rng(0))
    a, b = B.period_series(x, r, 1.0), B.period_series(x, r, 2.0)
    assert (a["exposure"] == b["exposure"]).all()
    assert (a["net"] >= b["net"]).all()


def test_same_seed_same_run():
    p = mk(np.vstack([np.linspace(20000.0, 21000.0 + i, 130) for i in range(40)]))
    x = ctx(p)
    r1 = B.simulate(x, np.random.default_rng(5))
    r2 = B.simulate(x, np.random.default_rng(5))
    assert (r1.gross == r2.gross).all()
    assert r1.expo[x.periods[0][0] + 5] == pytest.approx(B.SLOTS * L.ORDER_WON, rel=0.01)  # 40종목 중 20


def test_buffer_label_keeps_name_inside_stay_band():
    tv = np.full(130, 1000.0)
    p0 = one_name()
    (a, _), (a2, _) = B.periods_of(p0.sig_t0, p0.T)[:2]
    tv[a2 - 20:] = 3000.0              # 20억 위, 40억 아래
    p = mk(np.full(130, 20000.0), TV=tv)
    keep = B.simulate(ctx(p, B.Cfg(band=(300.0, 2000.0), stay=(150.0, 4000.0))), np.random.default_rng(0))
    drop = B.simulate(ctx(p), np.random.default_rng(0))
    assert keep.expo[a2 + 10] == pytest.approx(1.5e7)
    assert drop.expo[a2 + 10] == 0.0


def test_kill_bars_and_void():
    st = pd.bdate_range("2016-09-01", periods=30, freq="60B").to_numpy().astype("datetime64[ns]")
    u = np.full(30, 0.01)
    u2 = np.full(30, 0.0)
    k = B.kill_bars(u, u2, st, 0.9, -0.30, -0.25, -5e7)
    assert k["K1_ok"] and k["K2_ok"] and k["K3_ok"] and k["K4_ok"] and not k["VOID"]
    assert k["K4_mdd_limit"] == pytest.approx(0.325)
    no_mdd = B.kill_bars(u, u2, st, 0.9, -0.33, -0.25, -5e7)      # 0.33 > 0.25 × 1.3
    assert not no_mdd["K4a_ok"] and no_mdd["K4b_ok"] and not no_mdd["K4_ok"]
    no_pnl = B.kill_bars(u, u2, st, 0.9, -0.30, -0.25, -1.01e8)
    assert no_pnl["K4a_ok"] and not no_pnl["K4b_ok"] and not no_pnl["K4_ok"]
    assert B.kill_bars(u, u2, st, 0.9, -0.30, -0.25, -1e8)["K4b_ok"]
    u3 = u.copy()
    u3[st >= np.datetime64("2020-01-01")] = -0.01
    assert not B.kill_bars(u3, u2, st, 0.9, -0.3, -0.25, -5e7)["K2_ok"]
    assert B.kill_bars(u, u2, st, 0.5, -0.3, -0.25, -5e7)["VOID"]
    assert B.kill_bars(u[:27], u2[:27], st[:27], 0.9, -0.3, -0.25, -5e7)["VOID"]


def test_jump_equals_daily_marking_on_random_panel(monkeypatch):
    rng = np.random.default_rng(3)
    n, T = 60, 400
    r = rng.normal(0.0005, 0.02, (n, T))
    c = 20000 * np.cumprod(1 + r, axis=1)
    v = np.full((n, T), 1000.0)
    tv = rng.uniform(200, 2500, (n, T))
    for i in range(0, n, 7):
        s = rng.integers(100, 300)
        v[i, s:s + rng.integers(5, 40)] = 0.0
    for i in range(3, n, 9):
        c[i, rng.integers(200, 380):] = np.nan
    v = np.where(np.isfinite(c), v, np.nan)
    tv = np.where(np.isfinite(c), tv, np.nan)
    p = mk(c, V=v, TV=tv, H=c * 1.02, Lo=c * 0.98, distress=rng.random(n) < 0.3)
    doom = p.distress & (p.last < T - 1) & (rng.random(n) < 0.5)
    x = ctx(p, doom=doom)
    fast = B.simulate(x, np.random.default_rng(9))
    monkeypatch.setattr(B, "_JUMP", False)
    slow = B.simulate(x, np.random.default_rng(9))
    for name in ("gross", "cost", "idx", "expo", "halted"):
        assert np.allclose(getattr(fast, name), getattr(slow, name), atol=1e-4), name


def test_curve_metrics_by_hand():
    p = mk(np.full(130, 20000.0))
    x = ctx(p)
    run = B.simulate(x, np.random.default_rng(0))
    t0 = p.sig_t0
    run.gross[:] = 0.0
    run.cost[:] = 0.0
    run.idx[:] = 0.0
    run.gross[t0 + 1] = 3e7       # +10%
    run.gross[t0 + 2] = -6.6e7    # 1.10 → 0.88: −20%
    m = B.curve_metrics(x, run, 2.0)
    assert m["mdd"] == pytest.approx(-0.2)
    assert m["total"] == pytest.approx(-0.12)
    assert m["underwater"] == p.T - (t0 + 2)
