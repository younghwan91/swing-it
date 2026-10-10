"""research/experiments/meta_gen3_run 의 순수 헬퍼 + 합성 자료 위 월 북(상위 20 진입 · 30 안 유지 · 상한가 시가 건너뜀) — DB 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.experiments import meta_gen3_run as m3
from research.signals import meta_fast as mf


def test_ranked_codes_order_and_ties():
    s = np.array([0.5, np.nan, 0.9, 0.5, 0.1])
    ok = np.array([True, True, True, True, False])
    assert m3.ranked_codes(s, ok).tolist() == [2, 0, 3]          # NaN·자격 없음 제외, 동률은 종목 번호 오름차순


def test_lu_open_band_has_upper_cap():
    o = np.array([129.3, 130.0, 130.5, 131.0, 150.0, np.nan, 129.0])
    c = np.full(len(o), 100.0)
    assert m3.lu_open(o, c).tolist() == [True, True, True, False, False, False, False]


def test_mix_monthly_resets_weights_each_month():
    mon = np.array([0, 0, 1])
    R = np.array([[0.10, 0.0], [0.10, 0.0], [0.10, 0.0]])
    out = m3.mix_monthly(R, [0.5, 0.5], mon)
    # 첫날 0.5·10%; 둘째 날 비중이 0.55/1.05 로 표류; 새 달 첫날 다시 0.5
    np.testing.assert_allclose(out, [0.05, 0.55 * 0.10 / 1.05, 0.05])
    np.testing.assert_allclose(m3.mix_monthly(np.array([[np.nan, 0.02]]), [0.2, 0.8], np.array([0])), [0.016])


def test_mdd_match_weight_exact_and_cap():
    bench = np.array([0.0, -0.20, 0.10])
    cash = np.zeros(3)
    mon = np.zeros(3, np.int64)
    assert m3.mdd_match_weight(bench, cash, -0.10, mon) == pytest.approx(0.50)
    assert m3.mdd_match_weight(bench, cash, -0.50, mon) == pytest.approx(1.0)      # 더 깊으면 100%
    assert m3.mdd_match_weight(bench, cash, 0.0, mon) == pytest.approx(0.0)
    assert np.isnan(m3.mdd_match_weight(bench, cash, np.nan, mon))


def test_boot_beat_dominance():
    rng = np.random.default_rng(1)
    rb = rng.normal(0, 0.01, 300)
    mon = np.repeat(np.arange(15), 20)
    b = m3.boot_beat(rb + 0.002, rb, mon, n=200)
    assert b["p_cagr"] == 1.0 and b["months"] == 15 and b["dcagr_lo"] > 0
    b2 = m3.boot_beat(rb, rb, mon, n=200)
    assert b2["p_cagr"] == 0.0                                                     # 같으면 '이김' 없음(> 0 엄격)


def test_tiers():
    assert m3.tier1(2.1, 0.10, 0.09) == {"t1_alpha": True, "t1_null": True, "t1": True}
    assert m3.tier1(1.9, 0.10, 0.09)["t1"] is False
    assert m3.tier1(np.nan, 0.10, np.nan) == {"t1_alpha": False, "t1_null": False, "t1": False}
    ok = m3.tier2(0.8, 0.03, 0.02, 0.04, 0.02, 0.025, 0.02)
    assert ok["t2"] and ok["t2_b1"]
    assert not m3.tier2(0.79, 0.03, 0.02, 0.04, 0.02, 0.025, 0.02)["t2"]
    assert not m3.tier2(0.9, 0.019, 0.02, 0.04, 0.02, 0.025, 0.02)["t2_2024"]
    assert not m3.tier2(0.9, 0.03, 0.02, 0.02, 0.02, 0.025, 0.02)["t2_ex25"]   # 같으면 실패(> 엄격)
    assert not m3.tier2(0.9, 0.03, 0.02, 0.04, 0.02, 0.01, 0.02)["t2_x2"]


def test_masks_and_wstats():
    d = pd.DatetimeIndex(["2019-12-30", "2020-01-02", "2024-12-30", "2025-01-02", "2026-01-02"])
    assert m3.window_mask(d, "2020-01-01", "2025-12-31").tolist() == [False, True, True, True, False]
    sm = m3.sub_masks(d)
    assert sm["2024"].tolist() == [False, True, True, False, False]
    assert sm["ex25"].tolist() == [True, True, True, False, True]
    s = m3.wstats(np.array([0.01, 0.0, -0.02, 0.05, 0.0]), np.full(5, 0.0001), d)
    assert s["cagr_ex25"] == pytest.approx(mf.equity_stats(np.array([0.01, 0.0, -0.02, 0.0]))["cagr"])
    assert s["cash_cagr"] > 0 and np.isfinite(s["sharpe_m"])


def test_cd91_extension_keeps_reviewer_values():
    from research.experiments.meta_gen2_run import CD91
    for y, v in CD91.items():
        assert m3.CD91_EXT[y] == v
    assert set(m3.CD91_EXT) >= {2017, 2018, 2019}


# ---------------------------------------------------------------------------
# 합성 월 북 — World.run 의 진입·유지·매도·상한가 건너뜀
# ---------------------------------------------------------------------------

def _world(order_oct: list[int]) -> "m3.World":
    days = pd.bdate_range("2017-08-01", "2017-11-30")
    n, nc = len(days), 40
    C = np.full((nc, n), 100.0)
    O = C.copy()  # noqa: E741
    sep1 = int(days.searchsorted(pd.Timestamp("2017-09-01")))
    O[1, sep1] = 130.0                                   # 종목 1: 9월 첫 거래일 상한가 시가 → 못 산다
    C[:, sep1 + 3] = 101.0                               # 9월 넷째 거래일 모두 +1% (이후 유지)
    C[:, sep1 + 4:] = 101.0
    O[:, sep1 + 4:] = 101.0
    pead = np.full((nc, n), np.nan)
    pead[:, :] = ((nc - np.arange(nc)) / nc)[:, None]    # 8월 신호: 0 이 최고
    oct_sig = int(days.searchsorted(pd.Timestamp("2017-09-29")))
    sc = np.empty(nc)
    sc[np.array(order_oct)] = np.linspace(1.0, 0.01, nc)
    pead[:, oct_sig:] = sc[:, None]
    inp = {"days": days, "codes": pd.Index([f"{i:06d}" for i in range(nc)]), "O": O, "H": np.maximum(O, C), "L": np.minimum(O, C),
           "C": C, "TR": np.ones((nc, n), bool), "last_present": np.ones(nc, bool), "market": np.zeros(nc, np.int8),
           "cost_tab": np.zeros((n, 2)), "h5": np.ones((nc, n), bool), "tt": np.ones((nc, n), bool), "pead": pead,
           "proxy": np.zeros((n, 3)), "adv20": np.full((nc, n), 1e5), "info": {}}
    return m3.World(inp)


def test_world_monthly_book_entry_keep_sell_and_lu_skip():
    order = [25, 0, 1, 2, 3, 4] + list(range(7, 20)) + [20, 21, 22, 23, 24, 5] + list(range(26, 40))
    order.insert(35, 6)                                  # 종목 5 = 24위(유지), 종목 6 = 35위(매도)
    W = _world(order)
    assert len(W.rb_d) == 3                              # 9·10·11월 진입
    res = W.run("Q1")
    L = W.lists["Q1"]
    sel, op = res["sel"], res["opened"]
    codes = L["pc"][sel][op]
    ent = L["pd"][sel][op] + 1
    sep1, oct1 = (int(W.days.searchsorted(pd.Timestamp(d))) for d in ("2017-09-01", "2017-10-02"))
    assert sorted(codes[ent == sep1].tolist()) == [0] + list(range(2, 20))       # 상위 20 − 상한가 종목 1
    assert sorted(codes[ent == oct1].tolist()) == [1, 25]                         # 빈 슬롯 2개 = 순위 순 25, 1(20 은 못 들어감)
    assert (ent > oct1).sum() == 0                                                # 11월은 거래 없음
    i6 = int(np.flatnonzero(codes == 6)[0])
    q6 = op[i6]
    assert res["x"][q6] == oct1 and res["reason"][q6] == mf.R_REBAL              # 35위 → 10월 첫날 시가 매도
    i5 = int(np.flatnonzero(codes == 5)[0])
    assert res["x"][op[i5]] == W.cut - 1                                          # 24위 → 계속 보유(끝까지)
    assert L["lu"].sum() >= 1
    # 9월 넷째 거래일 +1%: 19종목 × 5% → +0.95%, 현금 5% 에 금리
    k = sep1 + 3 - (W.t0 + 1)
    assert res["r"][k] == pytest.approx(0.0095, rel=1e-9)
    assert res["rc"][k] == pytest.approx(0.0095 + (1 - res["gross"][k - 1]) * W.cash[k], rel=1e-9)
    assert res["gross"][k - 1] == pytest.approx(0.95)


def test_world_null_ranks_keep_same_list():
    order = list(range(40))
    W = _world(order)
    pair, rk = W.random_ranks(W.lists["Q1"], np.random.default_rng(0))
    L = W.lists["Q1"]
    for k in range(len(W.rb_d)):
        a, b = L["ptr"][k], L["ptr"][k + 1]
        assert sorted(L["pc"][pair[a:b]].tolist()) == sorted(L["pc"][a:b].tolist())
        assert sorted(rk[k][L["pc"][a:b]].tolist()) == list(range(b - a))
    res = W.run("Q1", 1.0, (pair, rk))
    assert len(res["opened"]) >= 19


def test_kospi_hybrid_drift_and_official():
    d = pd.DatetimeIndex(["2020-06-01", "2020-12-30", "2024-01-25", "2024-01-26", "2024-02-01"])
    px = np.array([0.01, -0.02, 0.005, 0.03, np.nan])
    off = np.array([np.nan, np.nan, 0.5, -0.01, 0.02])
    k = m3.kospi_hybrid(px, off, d, {2019: 100.0, 2020: 110.0})
    assert np.prod(1 + k[:2]) == pytest.approx(1.10)                               # 2020 = 연말 비율에 맞춤
    assert k[2] == pytest.approx(0.005)                                            # 2024-01-26 전 = 대용 그대로(공식 있어도)
    np.testing.assert_allclose(k[3:], [-0.01, 0.02])                               # 이후 = 공식
