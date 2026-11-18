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
    assert (PP.FIRST_FORWARD_K, PP.N_FORWARD_STEPS, PP.VOID_MISSING) == (3, 36, 4) and PP.N_FORWARD_STEPS == PP.LOOKS[-1]


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


def test_yoy_tie_uses_later_period_regardless_of_input_order():
    """정정 F1-2: 패널 시작 전 공시는 모두 첫날로 몰려 동률이 된다 — 늦은 분기를 쓰고, 입력 순서·길이에 따라 바뀌지 않는다."""
    days = pd.bdate_range("2026-03-02", periods=6)
    codes = np.array(["008770"])
    rows = [("008770", "2016Q3", "2016-11-14", "2016-11-14", 3.0, 1.0), ("008770", "2017Q3", "2017-11-14", "2017-11-14", 1.17, 1.0)]
    ea = pd.DataFrame(rows, columns=["code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior"])
    per = pd.DataFrame(columns=["code", "period", "per"])
    for frame in (ea, ea.iloc[::-1]):
        out, _ = PP.yoy_from_frames(frame, per, codes, days)
        assert np.allclose(out[0], 0.17)  # 2017Q3 (늦은 분기), 2016Q3 의 2.0 이 아니라


# ── 정정 F1-3 ────────────────────────────────────────────────────────────────


def fstep(k, x=0.01):
    return {"k": k, "t": f"s{k:03d}", "bench": 0.0, "a_2x": x, "a_1x": x, "contrib": [(f"C{k}", 0.1, x * 10, 0.0)]}


def test_forward_report_gap_does_not_shift_into_k():
    """k=3 이 빠진 35 스텝 — 예전엔 missing_steps=None·"계속" 에 뒤 스텝이 K 안으로 당겨졌다."""
    a = list(np.random.default_rng(0).normal(0.01, 0.01, 36))
    st = [fstep(k, a[k - 3]) for k in range(3, 39) if k != 5]
    r = PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY, first_k=3)
    assert r["verdict"] == "판단 불가(결측)" and r["missing_steps"] == [5] and r["n_missing"] == 1 and r["looks"] == [] and not r["void"]


def test_forward_report_missing_first_step_is_seen_only_with_first_k():
    st = [fstep(k) for k in range(4, 20)]
    assert PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY, first_k=3)["missing_steps"] == [3]


def test_forward_report_tail_missing_uses_through_k():
    st = [fstep(k) for k in range(3, 10)]
    assert PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY, first_k=3, through_k=9)["missing_steps"] == []
    assert PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY, first_k=3, through_k=11)["missing_steps"] == [10, 11]


@pytest.mark.parametrize("n,verdict", [(1, "판단 불가(결측)"), (3, "판단 불가(결측)"), (4, "VOID"), (6, "VOID")])
def test_forward_report_void_boundary(n, verdict):
    gone = set(range(10, 10 + n))
    r = PP.forward_report([fstep(k) for k in range(3, 39) if k not in gone], PP.BOUNDARIES, PP.FUTILITY, first_k=3)
    assert r["verdict"] == verdict and r["void"] == (n >= 4) and r["n_missing"] == n and r["looks"] == []


def test_forward_report_nan_counts_as_missing_for_void():
    st = [fstep(k, float("nan") if k in (4, 5) else 0.01) for k in range(3, 39) if k not in (7, 8)]
    r = PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY, first_k=3)
    assert r["verdict"] == "VOID" and r["missing_steps"] == [4, 5, 7, 8]


def test_forward_report_void_counts_only_first_36():
    a = np.random.default_rng(5).normal(0.0, 0.01, 36)
    st = [fstep(k, a[k - 3]) for k in range(3, 39)] + [fstep(50)]        # 36 뒤 결손은 세지 않는다
    r = PP.forward_report(st, PP.BOUNDARIES, {}, first_k=3, through_k=55)
    assert not r["void"] and r["missing_steps"] == [] and r["n_steps"] == 36 and [x["K"] for x in r["looks"]][-1] == 36


def test_forward_report_broken_k_order_is_not_judged():
    r = PP.forward_report([fstep(5), fstep(4), fstep(6)], PP.BOUNDARIES, PP.FUTILITY, first_k=3)
    assert r["verdict"] == "판단 불가(결측)" and "k" in r["verdict_note"]


def test_forward_report_continuous_same_as_before():
    """결손이 없으면 값은 first_k 를 주든 안 주든 같다 — 경계·판정 경로를 건드리지 않았다."""
    a = list(np.random.default_rng(2).normal(0.02, 0.01, 24))
    st = [fstep(k + 3, x) for k, x in enumerate(a)]
    x, y = PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY, first_k=3), PP.forward_report(st, PP.BOUNDARIES, PP.FUTILITY)
    assert x == y and x["missing_steps"] == [] and x["looks"][0]["K"] == 12 and x["looks"][0]["boundary"] == 3.068


def test_ledger_report_entry_empty_and_cli(tmp_path, monkeypatch, capsys):
    r = PP.ledger_report([])
    assert r["n_steps"] == 0 and r["verdict"] == "계속" and r["through_k"] is None and r["first_k"] == 3
    path = tmp_path / "l.jsonl"
    rows = [{"date": f"2027-01-{d:02d}", "arm": PP.ARM, "step": {"k": 9, "t": "x", "f": "2027-01-02", "f_next": None},
             **({"step_closed": fstep(3)} if d == 3 else {})} for d in (2, 3)]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["pead_paper", "--report", "--ledger", str(path)])
    monkeypatch.setitem(__import__("sys").modules, "psycopg2", None)   # DB 를 안 건드린다
    PP.main()
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "VOID" and out["through_k"] == 8 and out["missing_steps"] == [4, 5, 6, 7, 8]


def test_closed_through_k_matches_build_row():
    """장부를 처음부터 쌓을 때 closed_through_k 까지의 스텝은 언제나 적혀 있다(거짓 결손 없음), 한 칸도 앞서가지 않는다."""
    p = synth()
    g = grid_of(p)
    start = str(p.dates[g[2]].date())
    rows: list = []
    for cut in range(g[2], len(p.dates)):
        q = p.truncate(cut)
        out = PP.simulate(q, [x for x in g if x <= cut])
        rows.append(PP.build_row(q, out, {}, {}, arm_start=start, written={s["t"] for s in PP.ledger_steps(rows)}))
        assert PP.missing_steps(PP.ledger_steps(rows), 2, PP.closed_through_k(rows)) == []
    done = PP.ledger_steps(rows)
    assert len(done) >= 4 and PP.closed_through_k(rows) == done[-1]["k"]


def decision_panel(blank_last=False, partial_sec=False):
    p = synth()
    g = grid_of(p)
    t = g[5]
    C, tv, sec = p.C.copy(), p.tv_won.copy(), p.has_sec.copy()
    if blank_last:                      # 결정일 분봉 미적재 → mask_last_day 가 그날 종가를 전부 지운 상태
        C[:, t] = np.nan
        tv[:, t] = np.nan
    if partial_sec:
        sec[10:, t:] = False
    q = PP.Panel(dates=p.dates, codes=p.codes, C=C, tv_won=tv, ret=p.ret, cap=p.cap, sig=p.sig, has_sec=sec, market=p.market, delist_after=p.delist_after)
    return q, g, t


def test_decision_gate_healthy_day_passes():
    q, g, t = decision_panel()
    out = PP.simulate(q.truncate(t), [x for x in g if x <= t])
    assert PP.decision_gate(out.pending, {"wics_partial": False}) == []


def test_decision_gate_no_last_day_bars():
    q, g, t = decision_panel(blank_last=True)
    out = PP.simulate(q.truncate(t), [x for x in g if x <= t])
    why = PP.decision_gate(out.pending, {})
    assert out.pending["n_eligible"] == 0 and any("적격" in w for w in why) and any("트랜치" in w for w in why)


def test_decision_gate_partial_universe_and_flags():
    q, g, t = decision_panel(partial_sec=True)
    out = PP.simulate(q.truncate(t), [x for x in g if x <= t])
    assert any("u_excl" in w for w in PP.decision_gate(out.pending, {}))
    good = {"n_eligible": 60, "tranche": [1], "target": {1: 1.0}, "u_excl": 0.1}
    assert PP.decision_gate(good, {}) == []
    assert PP.decision_gate({**good, "n_eligible": PP.MIN_NAMES}, {}) == []                   # 경계: 최소는 통과
    assert len(PP.decision_gate({**good, "n_eligible": PP.MIN_NAMES - 1}, {})) == 1
    assert PP.decision_gate({**good, "u_excl": PP.U_EXCL_MAX}, {}) == []
    assert len(PP.decision_gate({**good, "u_excl": PP.U_EXCL_MAX + 0.01}, {})) == 1
    assert "WICS" in PP.decision_gate(good, {"wics_partial": True})[0]


def test_wics_size_check_partial():
    def w(sizes):
        return pd.DataFrame([{"date": f"2026-{m + 1:02d}-01", "code": f"{i:06d}"} for m, n in enumerate(sizes) for i in range(n)])
    assert PP.wics_size_check(w([100] * 5 + [90]))["wics_partial"] is False       # 경계: 90% 는 통과
    assert PP.wics_size_check(w([100] * 5 + [89]))["wics_partial"] is True
    assert PP.wics_size_check(w([100]))["wics_partial"] is False                 # 비교 대상 없음
    assert PP.wics_size_check(w([100]))["wics_last_n"] == 100
    assert PP.wics_size_check(pd.DataFrame(columns=["date", "code"]))["wics_partial"] is True


@pytest.fixture
def fake_db(monkeypatch):
    """run_day 를 합성 패널로 — load_panel·지수·격자만 바꾼다. 나머지 경로(게이트·장부·따라잡기·줄 찍기)는 실제 코드."""
    import swing_it.storage as ST

    state = {"p": synth(), "last": None, "idx_last": None, "calls": []}
    g = grid_of(state["p"])
    state["g"] = g

    def load(con, start, end):
        p = state["p"]
        cut = p.dates.get_loc(pd.Timestamp(end))
        state["calls"].append(end)
        return p.truncate(cut), {"wics_stale": False, "wics_stale_days": 3, "wics_last_n": 2456, "wics_partial": False}, {}

    def read_index(con, codes, cols=None, start=None, end=None, **kw):
        d = state["p"].dates
        d = d[d <= pd.Timestamp(state["idx_last"] or state["last"])]       # 기본: 지수와 가격이 같은 날까지
        return pd.DataFrame({"date": d[d <= pd.Timestamp(end)], "close": 1.0})

    monkeypatch.setattr(PP, "load_panel", load)
    monkeypatch.setattr(PP, "last_price_day", lambda con: state["last"])
    monkeypatch.setattr(PP, "make_grid", lambda dates, a, s: [x for x in g if x < len(dates)])
    monkeypatch.setattr(PP, "check_calendar", lambda *a: {"n_days": 0})
    monkeypatch.setattr(PP, "index_returns", lambda con, day: {})
    monkeypatch.setattr(PP, "live_recommend_safe", lambda *a: {"error": "x"})
    monkeypatch.setattr(ST, "read_index", read_index)
    return state


def ds(state, col):
    return str(state["p"].dates[col].date())


def blank_day(state, col):
    p = state["p"]
    C, tv = p.C.copy(), p.tv_won.copy()
    C[:, col] = np.nan
    tv[:, col] = np.nan
    state["p"] = PP.Panel(dates=p.dates, codes=p.codes, C=C, tv_won=tv, ret=p.ret, cap=p.cap, sig=p.sig, has_sec=p.has_sec, market=p.market, delist_after=p.delist_after)


def test_run_day_bad_decision_day_fails_loud_then_next_run_catches_up(fake_db, tmp_path, capsys):
    good = fake_db["p"]
    t = fake_db["g"][4]
    start = ds(fake_db, t)
    led = str(tmp_path / "l.jsonl")
    blank_day(fake_db, t)
    fake_db["last"] = start
    assert PP.run_day(None, None, led, False, arm_start=start) is False
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert any(x.startswith("실패:") and "결정일 행을 쓰지 않는다" in x for x in lines)
    assert any(x.startswith("감시:") for x in lines) and not any(x.startswith("기록") for x in lines)
    assert PP.read_ledger(led) == []
    # 분봉이 들어왔다 — 다음 날 실행이 지나간 결정일을 먼저 채우고 오늘 행을 쓴다
    fake_db["p"] = good
    fake_db["last"] = ds(fake_db, t + 1)
    assert PP.run_day(None, None, led, False, arm_start=start) is True
    rows = PP.read_ledger(led)
    assert [r["date"] for r in rows] == [start, ds(fake_db, t + 1)]
    assert rows[0]["phase"] == "decision" and rows[0]["backfilled"] is True and len(rows[0]["decision"]["tranche_new"]) == PP.TOP_N
    assert "backfilled" not in rows[1]


def test_run_day_catchup_still_bad_keeps_today_row_and_stays_failed(fake_db, tmp_path, capsys):
    t = fake_db["g"][4]
    start = ds(fake_db, t)
    led = str(tmp_path / "l.jsonl")
    blank_day(fake_db, t)
    fake_db["last"] = ds(fake_db, t + 1)
    assert PP.run_day(None, None, led, False, arm_start=start) is False
    assert [r["date"] for r in PP.read_ledger(led)] == [ds(fake_db, t + 1)]
    assert "실패:" in capsys.readouterr().out


def test_run_day_bad_data_does_not_block_prestart_or_existing_row(fake_db, tmp_path, capsys):
    t = fake_db["g"][4]
    day = ds(fake_db, t)
    led = str(tmp_path / "l.jsonl")
    fake_db["last"] = day
    assert PP.run_day(None, None, led, False, arm_start=day) is True
    first = PP.read_ledger(led)
    blank_day(fake_db, t)                                    # 이미 쓴 결정일에 나중에 나빠진 DB — 재실행은 실패가 아니다
    assert PP.run_day(None, None, led, False, arm_start=day) is True
    assert PP.read_ledger(led) == first and "이미 기록됨" in capsys.readouterr().out
    later = ds(fake_db, t + 40)
    assert PP.run_day(None, None, str(tmp_path / "l2.jsonl"), False, arm_start=later) is True    # 시작 전 결정일은 거부 대상이 아니다
    assert "시작 전" in capsys.readouterr().out


def test_run_day_price_behind_index_is_failure_even_when_prestart(fake_db, tmp_path, capsys):
    """가격이 지수보다 뒤처지면 "시작 전"·"이미 기록됨" 이 DB 지연을 숨기지 못한다."""
    t = fake_db["g"][3] + 1
    fake_db["last"] = ds(fake_db, t)
    fake_db["idx_last"] = ds(fake_db, t + 1)
    start = ds(fake_db, t + 30)
    assert PP.run_day(None, None, str(tmp_path / "l.jsonl"), False, arm_start=start) is False
    lines = capsys.readouterr().out.splitlines()
    assert any(x.startswith("실패:") and "DB 지연" in x for x in lines) and any(x.startswith("시작 전") for x in lines)


def test_label_existing_splits_rerun_from_no_new_day():
    rows = [{"date": "2026-11-05", "run_at": "2026-11-06T07:58:00"}]
    st = "이미 기록됨: 2026-11-05 pead_rank_F1 — 덧붙이지 않는다"
    assert PP.label_existing(st, rows, "2026-11-05", "2026-11-06").startswith("이미 기록됨(재실행)")
    assert PP.label_existing(st, rows, "2026-11-05", "2026-11-09").startswith("이미 기록됨(새 거래일 없음)")
    assert PP.label_existing("기록: x", rows, "2026-11-05", "2026-11-06") == "기록: x"


def test_monitor_line_is_one_line_with_flags_and_no_returns():
    row = {"date": "2026-12-08", "arm": PP.ARM, "phase": "hold", "ret_today_gross": 0.12345,
           "step_closed": {"t": "2026-11-05", "a_2x": 0.98765, "plan_mismatch": {"tranche_symdiff": 2, "weight_l1": 0.5}}}
    line = PP.monitor_line(row, {"wics_stale": True, "wics_stale_days": 30, "wics_partial": False, "wics_last_n": 2400}, [7])
    assert "\n" not in line and line.startswith("감시:")
    assert "wics_stale=True" in line and '"tranche_symdiff": 2' in line and "missing_k=[7]" in line
    assert "0.12345" not in line and "0.98765" not in line


def pead_step_of_paper_daily() -> str:
    from pathlib import Path

    text = (Path(__file__).parents[1] / "scripts" / "paper_daily.sh").read_text(encoding="utf-8")
    lines = [x for x in text.splitlines() if "pead_paper.py" in x and not x.startswith("#")]
    assert len(lines) == 1
    # 팔 실행은 공용 함수 run_arm(시간 제한·pipefail·실패 줄) — 스크립트의 정의를 그대로 떼어 같이 돌린다
    body = text.splitlines()
    a = body.index("run_arm() {")
    b = a + body[a:].index("}")
    return "FAILS=()\n" + "\n".join(body[a:b + 1]) + "\n" + lines[0]


@pytest.mark.parametrize("py_out,rc,want,unwanted", [
    ("기록: d\\n감시: x\\n {json", 0, ["기록: d", "감시: x"], ["실패", "{json"]),
    ("이미 기록됨(새 거래일 없음): d\\n감시: x", 0, ["이미 기록됨(새 거래일 없음): d"], ["실패"]),
    ("실패: d 결정일 행을 쓰지 않는다\\n감시: x", 1, ["실패: d 결정일", "실패: pead_rank_F1"], []),
    ("기록: d", 1, ["기록: d", "실패: pead_rank_F1"], []),                      # 파이썬이 뒤에서 죽었다 — grep 이 삼키지 않는다
    ("", 0, ["실패: pead_rank_F1"], []),
])
def test_paper_daily_pead_step_keeps_lines_and_failures(tmp_path, py_out, rc, want, unwanted):
    import subprocess

    uv = tmp_path / "uv"
    uv.write_text(f"#!/bin/bash\nprintf '{py_out}\\n'\nexit {rc}\n", encoding="utf-8")
    uv.chmod(0o755)
    done = subprocess.run(["bash", "-c", f"UV={uv}\n{pead_step_of_paper_daily()}"], capture_output=True, text=True, timeout=30)
    for w in want:
        assert w in done.stdout
    for u in unwanted:
        assert u not in done.stdout
