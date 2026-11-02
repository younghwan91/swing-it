"""옮긴 함수의 규칙 — 기본 핀에서 돈다. 숫자 재현은 ``test_pead_optimizer_book_golden.py``, 실데이터는 ``real_db/``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_fixtures import synthetic_ctx  # noqa: E402

from swing_it.book.accounting import check_invariants, simulate  # noqa: E402
from swing_it.book.context import models_from_mask  # noqa: E402
from swing_it.book.costs import COMMISSION_BP, linear_bp, trailing_vol  # noqa: E402
from swing_it.book.metrics import holding_intervals, ir, oos, payoff_ratio, step_returns  # noqa: E402
from swing_it.book.optimized import capped_rescale, clean_dust, rank_z  # noqa: E402
from swing_it.book.rank import tranche_names  # noqa: E402
from swing_it.book.spec import DEFAULT_SPEC, BookSpec, Flags  # noqa: E402
from swing_it.engine.panels import ffill_axis1  # noqa: E402
from swing_it.features.earnings_asof import REPRODUCE_B, load_yoy_strict, yoy_strict_wide  # noqa: E402
from swing_it.features.universe import m0_mask, rank_band_eligible, u_exclusion, u_exclusion_causes  # noqa: E402
from swing_it.storage import connect, read_periodic_filings  # noqa: E402
from swing_it.validation.ic import fold_ic  # noqa: E402
from swing_it.validation.keys import SEED, keyed_rng  # noqa: E402

NO_COSTS = Flags(costs=False)


# --- 1 유니버스 ---

def test_rank_band_is_cap_order_inside_the_liquid_signal_universe():
    sig = np.array([[1.0], [1.0], [np.nan], [1.0], [1.0]])
    adv = np.array([[30000.0], [30000.0], [30000.0], [10.0], [30000.0]])
    cap = np.array([[3.0], [9.0], [8.0], [7.0], [5.0]])
    in_u = np.ones(5, bool)
    assert list(rank_band_eligible(0, sig, adv, cap, in_u, cap_rank=(0, 2))) == [1, 4]   # NaN 신호·ADV 미달을 뺀 뒤 순위
    assert list(rank_band_eligible(0, sig, adv, cap, in_u, cap_rank=(1, 3))) == [4, 0]
    assert list(rank_band_eligible(0, sig, adv, cap, np.array([1, 0, 1, 1, 1], bool), cap_rank=(0, 1))) == [4]
    assert list(m0_mask(0, sig, adv, in_u)) == [True, True, False, False, True]
    assert u_exclusion(0, sig, adv, np.array([1, 0, 1, 1, 1], bool)) == pytest.approx(1 / 3)


def test_u_exclusion_causes_split_missing_cap_from_a_risk_model_drop():
    """같은 U_t 탈락이라도 주식수 공백(원천 문제)과 위험모델 탈락은 고칠 곳이 다르다. 합은 u_exclusion 과 같다."""
    sig = np.ones((5, 1))
    adv = np.full((5, 1), 30000.0)
    cap = np.array([[9.0], [np.nan], [7.0], [np.nan], [5.0]])
    in_u = np.array([1, 0, 0, 1, 1], bool)
    got = u_exclusion_causes(0, sig, adv, cap, in_u)
    assert got == {"cap_missing": pytest.approx(1 / 5), "risk_model": pytest.approx(1 / 5), "cap_missing_in_u": pytest.approx(1 / 5)}
    assert got["cap_missing"] + got["risk_model"] == pytest.approx(u_exclusion(0, sig, adv, in_u))
    empty = u_exclusion_causes(0, np.full((2, 1), np.nan), np.full((2, 1), 30000.0), np.ones((2, 1)), np.ones(2, bool))
    assert all(np.isnan(v) for v in empty.values())


def test_simulate_records_the_u_exclusion_causes_without_touching_u_excl():
    ctx = synthetic_ctx(0)
    ctx.cap[3, :] = np.nan
    recs = simulate(ctx, "R", flags=NO_COSTS)
    assert any(r.u_excl_cap_missing > 0 or r.u_excl_model > 0 for r in recs)
    for r in recs:
        assert r.u_excl_cap_missing + r.u_excl_model == pytest.approx(r.u_excl)


# --- 3 알파: 알게 된 날 ---

def _earnings(rows):
    return pd.DataFrame(rows, columns=["code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior"])


DAYS = pd.bdate_range("2020-03-02", periods=30)
END = str(DAYS[-1].date())


def _at(sig, day):
    return sig[0, list(DAYS).index(pd.Timestamp(day))]


def test_value_is_usable_the_trading_day_after_the_later_of_avail_and_filing():
    """리뷰 M4 의 그 규칙 — avail_date 보다 정기보고서가 늦으면 그 다음 거래일부터다. 먼저 쓰면 미래 정보다."""
    ea = _earnings([("A", "2020Q1", "2020-03-04", "2020-03-04", 120.0, 100.0)])
    per = pd.DataFrame([("A", "2020Q1", "2020-03-10")], columns=["code", "period", "per"])
    sig, _ = yoy_strict_wide(ea, per, ["A"], DAYS, END)
    assert np.isnan(_at(sig, "2020-03-04")) and np.isnan(_at(sig, "2020-03-10"))
    assert _at(sig, "2020-03-11") == pytest.approx(0.2)


def test_filing_before_avail_still_waits_for_avail():
    ea = _earnings([("A", "2020Q1", "2020-03-10", "2020-03-10", 120.0, 100.0)])
    per = pd.DataFrame([("A", "2020Q1", "2020-03-04")], columns=["code", "period", "per"])
    sig, _ = yoy_strict_wide(ea, per, ["A"], DAYS, END)
    assert np.isnan(_at(sig, "2020-03-10")) and _at(sig, "2020-03-11") == pytest.approx(0.2)


def test_quarter_without_a_filing_is_dropped_from_2019q1_on_but_falls_back_before():
    ea = _earnings([("A", "2019Q1", "2020-03-04", "2020-03-04", 150.0, 100.0), ("B", "2018Q3", "2020-03-04", "2020-03-04", 150.0, 100.0)])
    per = pd.DataFrame(columns=["code", "period", "per"])
    sig, info = yoy_strict_wide(ea, per, ["A", "B"], DAYS, END)
    assert not np.isfinite(sig[0]).any(), "2019Q1 이후는 정기보고서가 없으면 뺀다"
    assert sig[1, list(DAYS).index(pd.Timestamp("2020-03-05"))] == pytest.approx(0.5)
    assert info["dropped_2019Q1plus_no_periodic_le_end"] == 1


def test_restatement_is_used_the_trading_day_after_it_was_known():
    """새 경로 기본 — 정정본도 알게 된 다음 거래일부터다(첫 공시와 같은 규칙). 재현 경로는 옛 당일 사용을 유지한다."""
    ea = _earnings([("A", "2020Q1", "2020-03-04", "2020-03-04", 120.0, 100.0), ("A", "2020Q1", "2020-03-04", "2020-03-20", 180.0, 100.0)])
    per = pd.DataFrame([("A", "2020Q1", "2020-03-04")], columns=["code", "period", "per"])
    sig, _ = yoy_strict_wide(ea, per, ["A"], DAYS, END)
    assert _at(sig, "2020-03-20") == pytest.approx(0.2) and _at(sig, "2020-03-23") == pytest.approx(0.8)
    old, _ = yoy_strict_wide(ea, per, ["A"], DAYS, END, **REPRODUCE_B)
    assert _at(old, "2020-03-19") == pytest.approx(0.2) and _at(old, "2020-03-20") == pytest.approx(0.8)


def _truncated_same(ea, per, days, cut, **kw):
    t = days[cut]
    e2 = ea[(pd.to_datetime(ea["avail_date"]) <= t) & (pd.to_datetime(ea["knowledge_date"]) <= t)]
    p2 = per[pd.to_datetime(per["per"]) <= t]
    full, _ = yoy_strict_wide(ea, per, ["A"], days, str(days[-1].date()), **kw)
    short, _ = yoy_strict_wide(e2, p2, ["A"], days[:cut + 1], str(t.date()), **kw)
    return np.array_equal(short, full[:, :cut + 1], equal_nan=True)


def test_truncation_identity_holds_at_the_2018q4_seam_only_with_the_new_default():
    """2018Q4 사업보고서는 2019-04 까지 나온다. end 2019-04~05 로 자르면 옛 기본(2019Q1)은 정기보고서를 몰라 avail_date 로 돌아가
    값이 더 일찍 보인다. 긴 패널에서는 늦은 날에 보이니 과거가 end 에 따라 바뀐다(검증 3단계)."""
    days = pd.bdate_range("2019-03-01", periods=80)
    ea = _earnings([("A", "2018Q4", "2019-03-29", "2019-03-29", 130.0, 100.0)])
    per = pd.DataFrame([("A", "2018Q4", "2019-05-10")], columns=["code", "period", "per"])
    cut = list(days).index(pd.Timestamp("2019-04-15"))
    assert _truncated_same(ea, per, days, cut)
    assert not _truncated_same(ea, per, days, cut, periodic_required_from="2019Q1")


def test_truncation_fuzz_on_the_new_default_is_identical():
    """2018Q3 이전 분기는 정기보고서 날짜가 없다(실데이터). 그 전제 위에서 어느 end 로 잘라도 과거 칸이 같다."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2019-01-01", periods=260)
    periods = [f"{y}Q{q}" for y in (2018, 2019) for q in (1, 2, 3, 4)]
    for _ in range(25):
        rows, filed = [], []
        for p in periods:
            a = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(-60, 340)))
            k = a + pd.Timedelta(days=int(rng.integers(0, 40)))
            rows.append(("A", p, str(a.date()), str(k.date()), float(rng.normal(100, 50)), 100.0))
            if p >= "2018Q4" and rng.random() < 0.85:
                filed.append(("A", p, str((a + pd.Timedelta(days=int(rng.integers(-30, 60)))).date())))
        ea, per = _earnings(rows), pd.DataFrame(filed, columns=["code", "period", "per"])
        ea = ea[ea["avail_date"] <= str(days[-1].date())]
        ea = ea[ea["knowledge_date"] <= str(days[-1].date())]
        per = per[per["per"] <= str(days[-1].date())]
        for cut in (70, 137, 200):
            t = days[cut]
            if not ((pd.to_datetime(ea["avail_date"]) <= t) & (pd.to_datetime(ea["knowledge_date"]) <= t)).any():
                continue
            assert _truncated_same(ea, per, days, cut)


def test_max_age_days_stops_a_value_from_living_for_years():
    """새 분기가 안 오면 값이 몇 년이고 그대로다(폐지·미제출). 나이 상한을 주면 그 뒤 칸은 NaN 이다."""
    ea = _earnings([("A", "2018Q3", "2020-03-04", "2020-03-04", 150.0, 100.0)])
    per = pd.DataFrame(columns=["code", "period", "per"])
    sig, _ = yoy_strict_wide(ea, per, ["A"], DAYS, END, max_age_days=7)
    assert _at(sig, "2020-03-05") == pytest.approx(0.5) and _at(sig, "2020-03-12") == pytest.approx(0.5)
    assert np.isnan(_at(sig, "2020-03-13"))
    assert np.isfinite(yoy_strict_wide(ea, per, ["A"], DAYS, END)[0][0, -1])


@pytest.mark.parametrize("call,exc,match", [
    (lambda: yoy_strict_wide(_earnings([]), pd.DataFrame(columns=["code", "period", "per"]), ["A"], DAYS, END), ValueError, "실적"),
    (lambda: yoy_strict_wide(_earnings([("A", "2018Q1", "2020-03-04", "2020-03-04", 1.0, 1.0)]), pd.DataFrame(columns=["code", "period", "per"]),
                             ["A"], pd.DatetimeIndex([]), END), ValueError, "dates"),
    (lambda: yoy_strict_wide(_earnings([("A", "2018Q1", "2020-03-04", "2020-03-04", 1.0, 1.0)]), pd.DataFrame(columns=["code", "period", "per"]),
                             ["A"], DAYS[::-1], END), ValueError, "단조"),
])
def test_bad_inputs_say_what_is_wrong(call, exc, match):
    with pytest.raises(exc, match=match):
        call()


def test_two_quarters_known_the_same_day_use_the_later_one_by_default():
    """정정 F1-2. ``tie_col=None`` 은 옛 동작이고 단계 B 재현에만 쓴다."""
    ea = _earnings([("A", "2020Q1", "2020-03-04", "2020-03-04", 120.0, 100.0), ("A", "2020Q2", "2020-03-04", "2020-03-04", 300.0, 100.0)])
    per = pd.DataFrame([("A", "2020Q1", "2020-03-10"), ("A", "2020Q2", "2020-03-10")], columns=["code", "period", "per"])
    sig, _ = yoy_strict_wide(ea, per, ["A"], DAYS, END)
    assert _at(sig, "2020-03-11") == pytest.approx(2.0)


def test_values_past_end_are_refused():
    ea = _earnings([("A", "2020Q1", "2020-03-04", "2020-03-04", 120.0, 100.0)])
    per = pd.DataFrame(columns=["code", "period", "per"])
    with pytest.raises(AssertionError):
        yoy_strict_wide(ea, per, ["A"], DAYS, "2020-03-03")


def test_loader_reads_through_the_front_door_and_stops_at_end(tmp_path):
    con = connect(tmp_path / "t.db")
    con.execute("CREATE TABLE earnings_disclosures(code TEXT, period TEXT, kind TEXT, rcept_dt TEXT)")
    con.executemany("INSERT INTO earnings_disclosures VALUES(?,?,?,?)",
                    [("A", "2020Q1", "periodic", "2020-03-10"), ("A", "2020Q1", "periodic", "2020-03-12"),
                     ("A", "2020Q1", "other", "2020-03-01"), ("A", "2020Q2", "periodic", "2099-01-01")])
    con.executemany("INSERT INTO earnings(code,period,avail_date,knowledge_date,netinc,netinc_prior) VALUES(?,?,?,?,?,?)",
                    [("A", "2020Q1", "2020-03-04", "2020-03-04", 120.0, 100.0), ("A", "2020Q2", "2099-01-01", "2099-01-01", 999.0, 100.0)])
    con.commit()
    per = read_periodic_filings(con, END)
    assert per.to_dict("records") == [{"code": "A", "period": "2020Q1", "per": "2020-03-10"}]
    sig, info = load_yoy_strict(con, ["A"], DAYS, END)
    assert _at(sig, "2020-03-10") != _at(sig, "2020-03-10") and _at(sig, "2020-03-11") == pytest.approx(0.2)
    assert info["earnings_rows_read_le_end"] == 1
    con.close()


# --- 공용 ---

def test_ffill_matches_the_definition():
    c = np.array([[np.nan, 1.0, np.nan, 3.0, np.nan], [2.0, np.nan, np.nan, np.nan, 5.0]])
    want = np.array([[np.nan, 1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0, 5.0]])
    assert np.array_equal(ffill_axis1(c), want, equal_nan=True)


def test_ffill_is_bitwise_core_when_core_has_it():
    core = pytest.importorskip("krx_quant_core.backtest.crosssectional")
    if not hasattr(core, "_ffill_axis1_nb"):
        pytest.skip("core 0.7.0 에는 없다")
    rng = np.random.default_rng(0)
    c = rng.normal(0, 1, (40, 60))
    c[rng.random(c.shape) < 0.3] = np.nan
    assert ffill_axis1(c).tobytes() == core._ffill_axis1_nb(c).tobytes()


def test_keyed_rng_is_the_prereg_seed():
    assert SEED == 20261101
    assert keyed_rng("x", 1).random() == keyed_rng("x", 1).random()
    assert keyed_rng("x", 1).random() != keyed_rng("x", 2).random() != keyed_rng("y", 1).random()


def test_fold_ic_purges_the_label_window_and_clips():
    days, ics = np.arange(10), np.full(10, 0.5)
    assert fold_ic(days, ics, 8, 3) == pytest.approx(0.10), "상한 자르기"
    assert fold_ic(days, ics, 2, 3) == 0.0, "퍼지 뒤 표본 없음"
    assert fold_ic(days, np.full(10, -0.5), 8, 3) == 0.0, "음수는 0"
    assert fold_ic(days, ics, 8, 3, clip=(0.0, 1.0)) == pytest.approx(0.5)


def test_rank_z_and_rescale_and_dust():
    assert list(np.round(rank_z(np.array([1.0, np.nan, 3.0])), 6)) == [-0.67449, 0.0, 0.67449]
    assert capped_rescale(np.array([0.5, 0.5]), 1.0, 0.6).sum() == pytest.approx(1.0)
    assert capped_rescale(np.array([0.5, 0.5]), 1.0, 0.4).sum() == pytest.approx(0.8), "모두 상한이면 남는다"
    w, n = clean_dust(np.array([1e-9, 0.5, 0.5]), np.array([1.0, 1.0, 1.0]), 3e8, np.zeros(3, bool), cap=None)
    assert n == 1 and w[0] == 0.0 and w.sum() == pytest.approx(1.0 + 1e-9)


def test_linear_cost_taxes_only_the_sell_side():
    day = pd.Timestamp("2023-06-01").date()
    buy, sell = linear_bp(np.array(["BUY", "SELL"]), np.array(["KOSPI", "KOSPI"], dtype=object), day, np.array([5.0, 5.0]))
    assert buy == pytest.approx(COMMISSION_BP + 5.0) and sell > buy


def test_trailing_vol_needs_enough_observations():
    ret = np.full((1, 100), np.nan)
    ret[0, 90:] = 0.01
    assert np.isnan(trailing_vol(ret, 99)[0])
    assert np.isfinite(trailing_vol(ret, 99, min_obs=5)[0])


def test_payoff_ir_and_intervals():
    assert np.isnan(ir(np.array([0.1])))
    assert ir(np.array([0.1, 0.2]), ann=1.0) == pytest.approx(0.15 / np.std([0.1, 0.2], ddof=1))
    assert payoff_ratio(np.array([0.2, -0.1])) == pytest.approx(2.0)
    assert np.isnan(payoff_ratio(np.array([0.2])))


# --- 7·9 북 ---

def test_tranche_is_top_n_by_rank_signal_with_eligibility_from_the_true_signal():
    ctx = synthetic_ctx(1)
    t = ctx.grid[3]
    names = tranche_names(t, ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t))
    assert names.size == DEFAULT_SPEC.top_n
    vals = ctx.sig[names, t]
    assert np.all(vals[:-1] >= vals[1:])
    assert tranche_names(t, ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t), 5).size == 5
    assert tranche_names(t, ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t), 10, 10_000) is None, "적격 부족이면 트랜치 없음"
    other = tranche_names(t, -ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t))
    assert not set(names) & set(other), "순위 신호만 바꾸면 반대쪽을 고른다"


def test_rank_book_accounting_identities_hold():
    ctx = synthetic_ctx(0)
    recs = simulate(ctx, "R", flags=NO_COSTS)
    assert check_invariants(recs, "R", costs=False) == []
    for r in recs:
        assert r.f == r.t + 1, "t 종가까지의 정보로 정하고 t+1 종가에 체결한다"
        assert sum(w * g for _, w, g, _ in r.contrib) == pytest.approx(r.gross)
    net = step_returns(recs, 2.0)
    assert net["a"].to_numpy() == pytest.approx((net["net"] - net["bench"]).to_numpy(), nan_ok=True)
    assert len(oos(net, ("2019-08-01", "2019-12-31"))) < len(net)
    assert holding_intervals(recs, 1.0).size > 0


def test_delisted_holding_goes_to_cash_at_its_last_close():
    """폐지 종목을 보유 가치로 남기면 생존편향이 회계로 들어온다. 마지막 종가로 현금화하고 다음 스텝에 안 든다."""
    ctx = synthetic_ctx(0, delist=False, gaps=False)
    recs = simulate(ctx, "R", flags=NO_COSTS)
    k = 3
    name = next(i for i, w, _, _ in recs[k].contrib if w > 0)
    last = recs[k + 1].f - 1                                # 다음 체결일 앞에서 끝난다
    ctx.C[name, last + 1:] = np.nan
    ctx.tv_won[name, last + 1:] = np.nan
    ctx.delist_after[name] = last
    ctx.Cff = ffill_axis1(ctx.C)
    after = simulate(ctx, "R", flags=NO_COSTS)
    assert after[k + 1].n_delist == 1 and after[k + 1].delist_mass > 0
    assert after[k + 1].n_frozen == 0, "얼린 보유가 아니라 현금이다"
    assert name not in [i for i, w, _, _ in after[k + 1].contrib if w > 0]


def test_spec_changes_reach_the_book():
    """규칙이 정말 인자로 닿는가 — 기본값에 묶여 있으면 새 연구가 못 쓴다."""
    ctx = synthetic_ctx(0)
    small = simulate(ctx, "R", flags=NO_COSTS, spec=BookSpec(top_n=3))
    base = simulate(ctx, "R", flags=NO_COSTS)
    assert max(r.n_names for r in small) < max(r.n_names for r in base)
    short = simulate(ctx, "R", flags=NO_COSTS, spec=BookSpec(horizon=20))
    assert all(r.n_tranches <= 1 for r in short) and check_invariants(short, "R", spec=BookSpec(horizon=20), costs=False) == []


def test_rank_book_runs_without_a_risk_model():
    """6단계를 건너뛴 연구 — 종목 집합만 넘긴다. 사전 위험은 NaN 이고 수익은 같다."""
    ctx = synthetic_ctx(0)
    with_model = simulate(ctx, "R", flags=NO_COSTS)
    ctx.models = models_from_mask(np.column_stack([ctx.in_u(t) if t in ctx.grid else np.zeros(ctx.C.shape[0], bool)
                                                   for t in range(ctx.C.shape[1])]), ctx.grid)
    ctx._u = {}
    no_model = simulate(ctx, "R", flags=NO_COSTS)
    assert [r.gross for r in no_model] == [r.gross for r in with_model]
    assert all(np.isnan(r.exante_20) for r in no_model)


def test_costs_cannot_be_silently_absent(monkeypatch):
    """비용 켠 회계는 비용을 낸다 — 레이어가 없으면 조용히 0 이 아니라 오류다."""
    from swing_it.book import costs
    from swing_it.core_layers import CoreLayerMissing

    monkeypatch.setattr(costs, "core_layer", lambda m: (_ for _ in ()).throw(CoreLayerMissing(m)))
    with pytest.raises(CoreLayerMissing):
        simulate(synthetic_ctx(0), "R")


# --- 4단계 널의 인자 ---

def test_permutation_null_checks_invariants_with_the_books_own_rules():
    """비용 끈 널에 비용 불변식을 걸면 위반으로 나왔다. 상한도 러너의 ``spec`` 값을 봐야 한다."""
    from swing_it.book.nulls import permutation_nulls

    ctx = synthetic_ctx(0)
    out = permutation_nulls(ctx, None, None, None, 1, 2, window=("2019-01-01", "2030-01-01"), kinds=("R",), flags=NO_COSTS)
    assert out["violations"] == []


def test_permutation_null_passes_spec_max_weight_to_the_invariant(monkeypatch):
    from swing_it.book import nulls

    seen = {}

    def spy(recs, kind, cell, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(nulls, "check_invariants", spy)
    spec = BookSpec(max_weight=0.03)
    ctx = synthetic_ctx(0)
    nulls._G.update({"ctx": ctx, "cell": None, "ic": None, "sr": None, "window": ("2019-01-01", "2030-01-01"), "spec": spec, "flags": NO_COSTS, "ann": 1.0})
    try:
        nulls._perm_task((0, "R"))
    finally:
        nulls._G.clear()
    assert seen["max_weight"] == 0.03 and seen["costs"] is False and seen["spec"] is spec


def test_permutation_null_uses_the_annualisation_it_is_given():
    from swing_it.book.metrics import ANN_20
    from swing_it.book.nulls import permutation_nulls

    ctx = synthetic_ctx(0)
    win = ("2019-01-01", "2030-01-01")
    a = permutation_nulls(ctx, None, None, None, 1, 2, window=win, kinds=("R",), flags=NO_COSTS)
    b = permutation_nulls(ctx, None, None, None, 1, 2, window=win, kinds=("R",), flags=NO_COSTS, ann=1.0)
    np.testing.assert_allclose(a["R"], b["R"] * ANN_20)


def test_permutation_null_on_an_empty_window_is_an_error_not_nans():
    from swing_it.book.nulls import permutation_nulls

    with pytest.raises(ValueError, match="창"):
        permutation_nulls(synthetic_ctx(0), None, None, None, 1, 2, window=("2040-01-01", "2041-01-01"), kinds=("R",), flags=NO_COSTS)


@pytest.mark.parametrize("a,b,match", [
    (np.zeros(5), np.zeros(6), "길이"),
    (np.array([0.1, np.nan, 0.2]), np.zeros(3), "NaN"),
    (np.array([]), np.array([]), "비었다"),
])
def test_boot_refuses_inputs_it_would_silently_misread(a, b, match):
    from swing_it.book.nulls import stationary_boot_dir

    with pytest.raises(ValueError, match=match):
        stationary_boot_dir(a, b, n=5)
