"""전진 팔 pead_rank_F1(research/paper/PEAD_FORWARD_PREREG.md) — 결정성·미래 정보 없음·덧붙이기·시작일·판정 규칙. DB 불요(합성 패널)."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import research.paper.pead_paper as PP


def synth(N: int = 70, T: int = 260, seed: int = 0) -> PP.Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=T)
    r = rng.normal(0.0005, 0.02, size=(N, T))
    C = 1000.0 * np.exp(np.cumsum(r, axis=1))
    C[3, 150:] = np.nan                               # 3: 폐지(마지막 거래일 149)
    C[5, 100:104] = np.nan                            # 5: 4일 거래정지
    tv = rng.uniform(3e10, 9e10, size=(N, T))         # 원 — ADV 300~900억
    tv[~np.isfinite(C)] = np.nan
    tv[7, :] = 1e9                                    # 7: ADV 10억 → 적격 밖
    ret = np.full_like(C, np.nan)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    cap = np.linspace(5e13, 1e11, N)[:, None] * (C / 1000.0)
    sig = rng.normal(0.1, 0.5, size=(N, T))
    sig[::9, :] = np.nan                              # 신호 없는 종목
    has_sec = np.ones((N, T), dtype=bool)
    has_sec[11, 120:] = False                         # 11: 위험모델 유니버스 이탈 → 강제 매도
    delist_after = np.full(N, 10**9, dtype=np.int64)
    delist_after[3] = 149
    market = np.array(["KOSPI" if i % 2 else "KOSDAQ" for i in range(N)], dtype=object)
    return PP.Panel(dates=dates, codes=np.array([f"{i:06d}" for i in range(N)]), C=C, tv_won=tv, ret=ret, cap=cap, sig=sig,
                    has_sec=has_sec, market=market, delist_after=delist_after)


def grid_of(p: PP.Panel, start: int = 70) -> list[int]:
    return list(range(start, len(p.dates), PP.STEP))


def as_tuple(st: PP.Step):
    return (st.k, st.t, st.f, st.f_next, st.cost1, st.turnover, st.gross, st.bench, tuple(sorted(st.weights.items())),
            tuple(st.contrib), tuple(st.book_path), tuple(st.bench_path), st.n_forced, st.n_delist, st.n_frozen)


def test_prereg_constants_fixed():
    assert PP.ARM == "pead_rank_F1"
    assert PP.ARM_START == "2026-11-05" and PP.GRID_ANCHOR == "2026-10-07" and PP.SIM_FROM == "2026-08-06"
    assert (PP.STEP, PP.HORIZON, PP.TOP_N, PP.MIN_NAMES, PP.CAP_RANK, PP.ADV_FLOOR, PP.AUM) == (20, 60, 10, 20, (20, 100), 20000.0, 3e8)
    assert PP.PRICE == {"session": "regular", "unpatched": "nan", "aftermarket_fix": True}
    assert PP.LOOKS == (12, 24, 36) and PP.OBF_SHAPE == (2.36, 1.67, 1.36) and PP.NW_LAG == 2
    assert PP.LEDGER == "research/paper/pead_rank_f1_ledger.jsonl"
    assert PP.BOUNDARIES == (3.068, 2.171, 1.768) and PP.FUTILITY == {24: 0.0}


def test_boundaries_match_calibration_file():
    from pathlib import Path

    cal = json.loads((Path(PP.__file__).parent / "pead_forward_calib.json").read_text(encoding="utf-8"))
    assert tuple(cal["boundaries"]) == PP.BOUNDARIES
    assert {int(k): v for k, v in cal["futility"].items()} == PP.FUTILITY
    assert cal["f_boot"]["multiplier"] == max(1.30, cal["f_boot"]["f_boot"])
    assert tuple(cal["obf_shape"]) == PP.OBF_SHAPE and tuple(cal["looks"]) == PP.LOOKS


def test_eligible_rank_window_and_filters():
    p = synth()
    t = 100
    el = PP.eligible(p, t)
    assert el.size > PP.MIN_NAMES
    ok = np.isfinite(p.sig[:, t]) & (p.adv[:, t] >= PP.ADV_FLOOR) & p.in_u(t)
    liq = np.flatnonzero(ok)
    rank = {i: r for r, i in enumerate(liq[np.argsort(-p.cap[liq, t], kind="stable")])}
    assert all(20 <= rank[i] < 100 for i in el)          # 0-기준 20..99 = 21~100위
    assert 7 not in el                                    # ADV 미달
    assert not any(np.isnan(p.sig[i, t]) for i in el)
    tr = PP.tranche_names(p, t)
    assert len(tr) == PP.TOP_N
    assert list(p.sig[tr, t]) == sorted(p.sig[el, t], reverse=True)[: PP.TOP_N]


def test_r_target_three_tranches_sum_one():
    p = synth()
    g = grid_of(p)
    tranches = {k: PP.tranche_names(p, t) for k, t in enumerate(g[:3])}
    w, n = PP.r_target(p, 2, g[2], tranches, g[2] + 1)
    assert n == 3 and abs(w.sum() - 1.0) < 1e-12 and w.max() <= 1.0 / 10 + 1e-12


def test_simulate_deterministic():
    p = synth()
    a, b = PP.simulate(p, grid_of(p)), PP.simulate(synth(), grid_of(p))
    assert [as_tuple(s) for s in a.steps] == [as_tuple(s) for s in b.steps]
    assert a.day_gross == b.day_gross and a.day_cost1 == b.day_cost1
    assert np.array_equal(a.w_last, b.w_last)


@pytest.mark.parametrize("cut", [129, 150, 171, 200, 233])
def test_no_lookahead_truncation(cut):
    """asof 에서 자른 패널의 결과는 그 뒤 데이터를 바꿔도 같다 — 끝난 스텝·그날 수익·마지막 보유 모두."""
    full = synth()
    other = synth()
    rng = np.random.default_rng(99)
    other.C[:, cut + 1:] *= rng.uniform(0.5, 2.0, size=other.C[:, cut + 1:].shape)   # 미래를 마구 바꾼다
    other.sig[:, cut + 1:] = rng.normal(size=other.sig[:, cut + 1:].shape)
    other.tv_won[:, cut + 1:] = 1.0
    other = PP.Panel(dates=other.dates, codes=other.codes, C=other.C, tv_won=other.tv_won, ret=other.ret, cap=other.cap, sig=other.sig,
                     has_sec=other.has_sec, market=other.market, delist_after=other.delist_after)
    g = grid_of(full)
    a = PP.simulate(full.truncate(cut), [x for x in g if x <= cut])
    b = PP.simulate(other.truncate(cut), [x for x in g if x <= cut])
    assert [as_tuple(s) for s in a.steps] == [as_tuple(s) for s in b.steps]
    assert a.day_gross == b.day_gross and np.array_equal(a.w_last, b.w_last) and a.pending == b.pending
    # 자른 결과의 끝난 스텝은 전체 패널 결과와도 같다(엔진이 체결일 f 밖의 미래 열을 안 본다)
    c = PP.simulate(full, g)
    done = [s for s in a.steps if s.complete]
    assert [as_tuple(s) for s in done] == [as_tuple(s) for s in c.steps[: len(done)]]


def test_step_net_equals_daily_rows_product():
    p = synth()
    out = PP.simulate(p, grid_of(p))
    done = [s for s in out.steps if s.complete]
    assert len(done) >= 5
    for s in done:
        prod = np.prod([1.0 + out.day_gross[d] for d in range(s.f + 1, s.f_next + 1)])
        for m in PP.COST_MULTS:
            assert abs((1.0 - m * out.day_cost1[s.f]) * prod - 1.0 - s.net(m)) < 1e-12
        bprod = np.prod([1.0 + out.day_bench[d] for d in range(s.f + 1, s.f_next + 1)])
        assert abs(bprod - 1.0 - s.bench) < 1e-12
        assert abs(sum(s.weights.values()) + s.cash - 1.0) < 1e-9


def test_forced_sale_and_delisting():
    p = synth()
    out = PP.simulate(p, grid_of(p))
    for s in out.steps:
        if s.t >= 120:
            assert 11 not in s.weights                    # U_t 이탈 → 다음 체결에서 판다
        if s.f > 149:
            assert 3 not in s.weights                     # 폐지
    assert sum(s.n_delist for s in out.steps) <= 1


def test_pending_on_decision_day():
    p = synth()
    g = grid_of(p)
    t = g[5]
    out = PP.simulate(p.truncate(t), [x for x in g if x <= t])
    assert out.pending is not None and out.pending["t"] == t
    assert abs(sum(out.pending["target"].values()) - 1.0) < 1e-9
    assert all(s.f <= t for s in out.steps)               # 결정일엔 체결하지 않는다


def test_build_row_step_closed_once_and_after_start(tmp_path):
    p = synth()
    g = grid_of(p)
    names = {}
    seen = []
    start = str(p.dates[g[3]].date())
    for cut in range(g[1], len(p.dates)):
        q = p.truncate(cut)
        out = PP.simulate(q, [x for x in g if x <= cut])
        row = PP.build_row(q, out, names, {}, arm_start=start)
        if "step_closed" in row:
            sc = row["step_closed"]
            assert sc["t"] >= start
            assert str(q.dates[cut - 1].date()) == sc["f_next"]    # 스텝이 끝난 다음 거래일 행
            seen.append(sc["t"])
        json.dumps(row, default=PP._json_default)
    assert seen == sorted(set(seen)) and len(seen) >= 3      # 스텝마다 한 번


def test_catchup_when_next_day_row_missing():
    p = synth()
    g = grid_of(p)
    start = str(p.dates[g[2]].date())
    out_all = PP.simulate(p, g)
    s0 = [s for s in out_all.steps if s.complete and str(p.dates[s.t].date()) >= start][0]
    cut = s0.f_next + 3                                     # 스텝이 끝난 다음 날 행을 건너뛰었다
    q = p.truncate(cut)
    out = PP.simulate(q, [x for x in g if x <= cut])
    row = PP.build_row(q, out, {}, {}, arm_start=start, written=set())
    assert "step_closed" not in row and [x["t"] for x in row["step_closed_catchup"]] == [str(p.dates[s0.t].date())]
    row2 = PP.build_row(q, out, {}, {}, arm_start=start, written={str(p.dates[s0.t].date())})
    assert "step_closed_catchup" not in row2


def test_ledger_steps_first_written_wins():
    rows = [{"date": "2026-12-08", "arm": PP.ARM, "step_closed": {"t": "2026-11-05", "a_2x": 0.01}},
            {"date": "2026-12-10", "arm": PP.ARM, "step_closed_catchup": [{"t": "2026-11-05", "a_2x": 0.99}, {"t": "2026-12-04", "a_2x": 0.02}]}]
    st = PP.ledger_steps(rows)
    assert [(s["t"], s["a_2x"]) for s in st] == [("2026-11-05", 0.01), ("2026-12-04", 0.02)]
    # 파일 순서가 기준이다: 뒤에 덧붙인 앞 날짜 백필 행은 먼저 적힌 따라잡기 값을 이기지 못한다
    late_backfill = [{"date": "2026-12-10", "arm": PP.ARM, "step_closed_catchup": [{"t": "2026-11-05", "a_2x": 0.5}]},
                     {"date": "2026-12-08", "arm": PP.ARM, "backfilled": True, "step_closed": {"t": "2026-11-05", "a_2x": 0.7}}]
    assert PP.ledger_steps(late_backfill)[0]["a_2x"] == 0.5


def test_catchup_then_asof_backfill_cannot_overwrite(tmp_path):
    """정정 F1-1 M2: 따라잡기로 먼저 적힌 스텝은, 그 뒤 빠진 날을 --asof 로 채워도 step_closed 로 다시 나오지 않는다."""
    p = synth()
    g = grid_of(p)
    start = str(p.dates[g[2]].date())
    s0 = [s for s in PP.simulate(p, g).steps if s.complete and str(p.dates[s.t].date()) >= start][0]
    path = str(tmp_path / "l.jsonl")
    late = s0.f_next + 3
    q = p.truncate(late)
    rows = PP.read_ledger(path)
    r1 = PP.build_row(q, PP.simulate(q, [x for x in g if x <= late]), {}, {}, arm_start=start,
                      written={s["t"] for s in PP.ledger_steps(rows)})
    assert "step_closed_catchup" in r1
    assert PP.append_row(path, r1, arm_start=start).startswith("기록")
    day = s0.f_next + 1                                  # 빠졌던 날을 백필
    q2 = p.truncate(day)
    rows = PP.read_ledger(path)
    r2 = PP.build_row(q2, PP.simulate(q2, [x for x in g if x <= day]), {}, {}, arm_start=start, backfilled=True,
                      written={s["t"] for s in PP.ledger_steps(rows)})
    assert "step_closed" not in r2 and "step_closed_catchup" not in r2
    PP.append_row(path, r2, arm_start=start)
    st = PP.ledger_steps(PP.read_ledger(path))
    assert [x["t"] for x in st] == [str(p.dates[s0.t].date())] and st[0] == json.loads(json.dumps(r1["step_closed_catchup"][0]))


def test_holding_intervals_active_and_stored_k():
    steps = [{"k": 5, "bench": 0.02, "contrib": [("A", 0.5, 0.10, 0.001), ("B", 0.5, -0.02, 0.001)]},
             {"k": 6, "bench": -0.01, "contrib": [("A", 0.5, 0.03, 0.0), ("B", 0.0, 0.0, 0.002)]},
             {"k": 8, "bench": 0.0, "contrib": [("A", 1.0, 0.01, 0.0)]}]           # k 7 건너뜀 → A 구간이 끊긴다
    iv = sorted(PP.holding_intervals(steps, 2.0).round(10))
    a1 = 0.5 * (0.10 - 0.02) - 0.002 + 0.5 * (0.03 + 0.01)
    b1 = 0.5 * (-0.02 - 0.02) - 0.002 - 0.004
    assert iv == sorted(np.round([a1, b1, 0.01], 10))


def test_forward_report_missing_is_explicit():
    st = [{"k": k, "bench": 0.0, "a_2x": x, "a_1x": x, "contrib": []} for k, x in enumerate([0.01, float("nan"), 0.02])]
    r = PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY)
    assert r["verdict"] == "판단 불가(결측)" and len(r["missing_steps"]) == 1


def test_check_calendar_guard():
    d = pd.bdate_range("2026-08-03", periods=30)
    assert PP.check_calendar(d, d[:-1], str(d[0].date()))["n_days"] == 29     # 지수가 하루 늦은 것은 통과
    with pytest.raises(ValueError):
        PP.check_calendar(d.delete(10), d, str(d[0].date()))                     # 가격에 거래일 하나가 빠졌다
    with pytest.raises(ValueError):
        PP.check_calendar(d, d.delete(10), str(d[0].date()))


def test_plan_mismatch():
    plan = {"tranche_new": [{"code": "A"}, {"code": "B"}], "target": {"A": 0.5, "B": 0.5}}
    rec = {"tranche": ["A", "C"], "contrib": [("A", 0.6, 0.0, 0.0), ("C", 0.4, 0.0, 0.0), ("B", 0.0, 0.0, 0.001)]}
    m = PP.plan_mismatch(plan, rec)
    assert m["tranche_symdiff"] == 2 and math.isclose(m["weight_l1"], 0.1 + 0.5 + 0.4)


def test_mask_last_day_non_session():
    df = pd.DataFrame({"code": ["A", "B", "C", "A"], "date": pd.to_datetime(["2026-10-30"] * 3 + ["2026-10-29"]),
                       "close": [1.0, 2.0, 3.0, 4.0], "trade_value": [1.0] * 4, "krx_src": ["session", "identity", "none", "identity"]})
    out, n = PP.mask_last_day(df, "2026-10-30", "krx_src", ("close", "trade_value"))
    assert n == 2 and out["close"].tolist()[0] == 1.0 and np.isnan(out["close"].iloc[1]) and out["close"].iloc[3] == 4.0
    assert PP.mask_last_day(df, "2023-12-28", "krx_src", ("close",))[1] == 0      # 애프터마켓 전 날짜는 건드리지 않는다


def test_live_recommend_failure_does_not_block(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(PP, "live_recommend", boom)
    assert PP.live_recommend_safe(None, {}, None, 0, [])["error"].startswith("RuntimeError")


def test_append_row_dedupe_and_arm_start(tmp_path):
    path = str(tmp_path / "x.jsonl")
    r1 = {"date": "2026-11-04", "arm": PP.ARM, "v": 1}
    assert PP.append_row(path, r1).startswith("시작 전")
    r2 = {"date": "2026-11-05", "arm": PP.ARM, "v": 2}
    assert PP.append_row(path, r2).startswith("기록")
    r3 = {"date": "2026-11-05", "arm": PP.ARM, "v": 3}
    assert PP.append_row(path, r3).startswith("이미 기록됨")
    lines = [json.loads(x) for x in open(path, encoding="utf-8")]
    assert lines == [r2]


def test_yoy_signal_asof_and_periodic_rule():
    days = pd.bdate_range("2026-04-01", periods=40)
    ea = pd.DataFrame({"code": ["A", "A", "B", "C"], "period": ["2025Q4", "2026Q1", "2026Q1", "2018Q4"],
                       "avail_date": ["2026-03-31", "2026-04-10", "2026-04-10", "2019-03-01"],
                       "knowledge_date": ["2026-03-31", "2026-04-10", "2026-04-10", "2019-03-01"],
                       "netinc": [120.0, 150.0, 90.0, 50.0], "netinc_prior": [100.0, 100.0, 100.0, 40.0]})
    # A 2026Q1 정기보고서 05-15 접수 → 다음 거래일부터, B 2026Q1 정기보고서 없음 → 뺀다(2019Q1 이후), C 2018Q4 는 avail 로
    per = pd.DataFrame({"code": ["A", "A"], "period": ["2025Q4", "2026Q1"], "per": ["2026-03-31", "2026-05-15"]})
    sig, info = PP.yoy_from_frames(ea, per, np.array(["A", "B", "C"]), days)
    d = list(days.strftime("%Y-%m-%d"))
    i515 = d.index("2026-05-15")
    assert math.isclose(sig[0, i515], 0.2) and math.isclose(sig[0, i515 + 1], 0.5)
    assert np.isnan(sig[1]).all() and info["dropped_2019Q1plus_no_periodic"] == 1
    assert math.isclose(sig[2, 0], 0.25)


def test_seam_guard_drops_unexplained_jump():
    dates = pd.bdate_range("2026-01-05", periods=5)
    C = np.array([[100.0, 101.0, 1010.0, 1020.0, 1030.0], [100.0, 50.0, 51.0, 52.0, 53.0]])
    raw = np.array([[100.0, 101.0, 102.0, 103.0, 104.0], [100.0, 100.0, 102.0, 104.0, 106.0]])
    ret = np.full_like(C, np.nan)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    ev = pd.DataFrame({"code": ["1"], "date": [dates[1]]})          # 종목 1 의 분할은 사건으로 설명된다
    r, c, n = PP.seam_guard(np.array(["0", "1"]), dates, ret, C, raw, ev)
    assert n == 1 and np.isnan(r[0, 2])                         # 종목 0 의 ×10 은 사건 없는 이음매 → 버린다
    assert math.isclose(c[0, 2], c[0, 1]) and math.isclose(c[0, 3] / c[0, 2], 1020.0 / 1010.0)   # 이어 붙인 종가
    assert math.isclose(r[1, 1], -0.5)                          # 설명된 분할은 그대로(수정주가 표가 맞다)


def test_make_grid_phase_guard():
    dates = pd.bdate_range("2026-01-05", periods=100)
    g = PP.make_grid(dates, str(dates[60].date()), str(dates[20].date()))
    assert g[0] == 20 and g[2] == 60 and all(b - a == 20 for a, b in zip(g, g[1:]))
    with pytest.raises(ValueError):
        PP.make_grid(dates, str(dates[60].date()), str(dates[21].date()))


def test_forward_report_rules():
    def steps(a, codes=None):
        codes = codes or [f"S{k}" for k in range(len(a))]           # 스텝마다 다른 종목 = 보유 구간 하나씩
        return [{"k": k, "bench": 0.0, "a_2x": x, "a_1x": x + 0.001, "contrib": [(c, 0.1, x * 10, 0.0)]}
                for k, (x, c) in enumerate(zip(a, codes))]
    b = (2.8, 2.0, 1.6)
    fut = {24: 0.0}
    rng = np.random.default_rng(1)
    strong = list(0.03 + rng.normal(0, 0.01, 12))
    r = PP.forward_report(steps(strong), b, fut)
    assert r["verdict"] == "엣지 확인" and r["looks"][0]["K"] == 12 and r["looks"][0]["EV_drop_top5_2x"] > 0
    assert PP.forward_report(steps(strong[:11]), b, fut)["verdict"] == "계속"
    # 경계는 넘었지만 기대값이 상위 5 구간에서만 나온다 → 확인 아님(취약), 계속
    frag = [0.5] * 5 + [-0.001 + 1e-5 * k for k in range(7)]
    rf = PP.forward_report(steps(frag), (0.1, 99.0, 99.0), fut)
    assert rf["verdict"] == "계속" and "취약" in rf["verdict_note"]
    weak = list(-0.01 + rng.normal(0, 0.01, 24))
    assert PP.forward_report(steps(weak), b, fut)["verdict"] == "엣지 없음(무용 중단)"
    flat = list(rng.normal(0, 0.03, 36))
    flat = [x - np.mean(flat) + 0.001 for x in flat]      # t 작고 양수
    r = PP.forward_report(steps(flat), (99.0, 99.0, 99.0), {})
    assert r["verdict"] == "판단 불가(검정력 부족)" and [lk["K"] for lk in r["looks"]] == [12, 24, 36]
    frag36 = [0.5] * 5 + [-0.001] * 31
    assert PP.forward_report(steps(frag36), (99.0, 99.0, 0.1), {})["verdict"] == "판단 불가(취약 — 상위 5 보유 구간 의존)"
    neg = [x - 0.002 for x in flat]
    assert PP.forward_report(steps(neg), (99.0, 99.0, 99.0), {})["verdict"] == "엣지 없음"


def test_nw_t_matches_core():
    from krx_quant_core.stats.metrics import newey_west_t

    x = np.random.default_rng(3).normal(0.01, 0.04, 30)
    assert math.isclose(PP.nw_t(x), newey_west_t(x, 2)[1], rel_tol=1e-12)


def test_trade_cost_bp_matches_core_impact_if_available():
    impact = pytest.importorskip("krx_quant_core.costs.impact")
    rng = np.random.default_rng(5)
    n = 50
    side = np.where(rng.random(n) > 0.5, "BUY", "SELL")
    notional, adv, vol = rng.uniform(1e6, 5e8, n), rng.uniform(1e9, 1e11, n), rng.uniform(0.005, 0.08, n)
    mk = np.where(rng.random(n) > 0.5, "KOSPI", "KOSDAQ").astype(object)
    hs = np.where(mk == "KOSPI", 5.0, 15.0)
    for day in (pd.Timestamp("2026-11-06").date(), pd.Timestamp("2024-06-03").date()):
        a = PP.trade_cost_bp(side, notional, adv, vol, mk, day, half_spread_bp=hs, y=1.0)
        b = impact.trade_cost_bp(side, notional, adv, vol, mk, day, half_spread_bp=hs, y=1.0)
        assert np.array_equal(a, b)


def test_trade_cost_bp_formula():
    d = pd.Timestamp("2026-11-06").date()
    from krx_quant_core.costs.tax import sell_tax_rate

    b = PP.trade_cost_bp(np.array(["SELL"]), np.array([1e8]), np.array([1e10]), np.array([0.02]), np.array(["KOSPI"], dtype=object), d,
                         half_spread_bp=np.array([5.0]))
    assert math.isclose(b[0], 1.5 + sell_tax_rate(d, "KOSPI") * 1e4 + 5.0 + 0.02 * math.sqrt(0.01) * 1e4, rel_tol=1e-12)
    with pytest.raises(ValueError):
        PP.trade_cost_bp(np.array(["BUY"]), np.array([1e8]), np.array([20000.0]), np.array([0.02]), np.array(["KOSPI"], dtype=object), d,
                         half_spread_bp=np.array([5.0]))
