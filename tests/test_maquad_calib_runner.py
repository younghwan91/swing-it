"""관문 0 실행기 — 합성 패널만(DB 없음). research/experiments/maquad_calib.py."""
import json

import numpy as np
import pandas as pd
import pytest

from research.experiments import maquad_calib as R

LABEL_END = pd.Timestamp(R.LABEL_END)
EVENT_START = pd.Timestamp(R.EVENT_START)


# --------------------------------------------------------------------------- 합성 자료

def _synth(n_codes=3, seed=0, perturb_after=None):
    """2013-06 ~ 2024-01 영업일, 추세+파동 로그 가격(정배열·비정배열이 둘 다 나온다)."""
    days = pd.bdate_range("2013-06-03", "2024-01-31")
    rng = np.random.default_rng(seed)
    t = np.arange(len(days))
    lc = []
    for i in range(n_codes):
        wave = 0.25 * np.sin(2 * np.pi * t / (180 + 40 * i)) + 0.0004 * t
        lc.append(np.log(5000.0) + wave + np.cumsum(rng.normal(0, 0.012, len(days))))
    C = np.exp(np.array(lc))
    if perturb_after is not None:
        late = days > perturb_after
        C[:, late] *= np.exp(np.random.default_rng(99).normal(0, 0.05, (n_codes, int(late.sum()))))
    codes = [f"{i:06d}" for i in range(n_codes)]
    meta = pd.DataFrame({"market": "KOSPI", "excluded": None}, index=pd.Index(codes, name="code"))
    return {"days": days, "codes": codes, "C": C, "raw_close": C.copy(), "tv": np.full_like(C, 5000.0), "meta": meta}


def _all_eps(ci, s):
    return np.ones(len(ci), bool)


@pytest.fixture(scope="module")
def X():
    return R.prepare(_synth(), _all_eps)


def _mask_X(n_codes=4):
    """build_events 거름만 시험하는 손 패널(가격 없이 상태 배열만)."""
    days = pd.bdate_range("2016-12-01", "2024-01-31")
    shape = (n_codes, len(days))
    return {"days": days, "codes": [f"{i:06d}" for i in range(n_codes)],
            "U": np.ones(shape, bool), "eps": np.ones(shape, bool), "aligned": np.zeros(shape, bool),
            "d": np.full(shape, 0.05), "f200": np.ones(shape, bool), "tmpl": np.zeros(shape, bool),
            "r3": np.ones(shape, bool)}


# --------------------------------------------------------------------------- build_events

@pytest.mark.parametrize("k", R.KS)
def test_events_respect_label_end(k):
    Xm = _mask_X()
    days = Xm["days"]
    for pop in ("judged", "no_f200", "tmpl"):
        Xm["tmpl"][:] = True
        ci, s = R.build_events(Xm, k, pop=pop)
        assert len(s) > 0
        assert (days[s + k] <= LABEL_END).all() and (days[s] >= EVENT_START).all()
        # 경계: 라벨 끝에 정확히 닿는 마지막 t 는 들어간다
        last = int(days.searchsorted(LABEL_END, side="right")) - 1 - k
        assert s.max() == last
    ci, s = R.build_events_all(Xm, k)
    assert (days[s + k] <= LABEL_END).all() and (days[s] >= EVENT_START).all()


def test_events_label_end_on_real_prepare(X):
    days = X["days"]
    for k in R.KS:
        ci, s = R.build_events(X, k)
        assert len(s) > 0
        assert (s + k < len(days)).all() and (days[s + k] <= LABEL_END).all() and (days[s] >= EVENT_START).all()


def test_events_are_precandidates():
    Xm = _mask_X()
    days = Xm["days"]
    t0 = int(days.searchsorted(pd.Timestamp("2018-03-02")))
    Xm["aligned"][0, t0] = True     # 정배열 → 빠짐
    Xm["d"][1, t0] = 0.0            # d = 0 → 빠짐(0 < d)
    Xm["d"][2, t0] = 0.10           # d = 0.10 → 들어감(≤ 0.10)
    Xm["d"][3, t0] = 0.1000001      # d > 0.10 → 빠짐
    Xm["f200"][2, t0 + 1] = False   # 200일 거름 실패 → 판정 모집단에서 빠짐, R1 에는 남음
    Xm["U"][2, t0 + 2] = False      # 유니버스 밖 → 빠짐
    Xm["eps"][2, t0 + 3] = False    # EPS 아님 → 빠짐
    Xm["d"][0, t0 + 4] = np.nan     # d 결측 → 빠짐
    Xm["tmpl"][1, t0 + 5] = True    # 템플릿 부분집합은 이 칸 하나
    k = 5
    ev = set(zip(*R.build_events(Xm, k)))
    assert (0, t0) not in ev and (1, t0) not in ev and (3, t0) not in ev
    assert (2, t0) in ev
    assert (2, t0 + 1) not in ev and (2, t0 + 2) not in ev and (2, t0 + 3) not in ev and (0, t0 + 4) not in ev
    r1 = set(zip(*R.build_events(Xm, k, pop="no_f200")))
    assert (2, t0 + 1) in r1 and (2, t0 + 2) not in r1 and r1 >= ev
    r2 = set(zip(*R.build_events(Xm, k, pop="tmpl")))
    assert r2 == {(1, t0 + 5)}
    # 사건은 날짜 순(달별 묶음 전제)
    ci, s = R.build_events(Xm, k)
    assert (np.diff(s) >= 0).all()


def test_events_union_flags_consistent(X):
    for k in (3, 10):
        ev = R.events_union(X, k)
        got = set(zip(ev["ci"][ev["judged"]], ev["s"][ev["judged"]]))
        assert got == set(zip(*R.build_events(X, k)))
        got = set(zip(ev["ci"][ev["r1"]], ev["s"][ev["r1"]]))
        assert got == set(zip(*R.build_events(X, k, pop="no_f200")))
        got = set(zip(ev["ci"][ev["r2"]], ev["s"][ev["r2"]]))
        assert got == set(zip(*R.build_events(X, k, pop="tmpl")))
        got = set(zip(ev["ci"][ev["r3"]], ev["s"][ev["r3"]]))
        assert got == set(zip(*R.build_events_all(X, k)))
        assert (ev["r1"] | ev["r3"]).all()            # 합집합 밖 사건 없음


def test_r3_sample_is_deterministic_20pct():
    days = pd.bdate_range("2017-01-02", "2017-12-28")
    codes = [f"{i:06d}" for i in range(40)]
    ci, s = np.nonzero(np.ones((len(codes), len(days)), bool))
    a = R.hash_sample(codes, days, ci, s, 5)
    b = R.hash_sample(codes, days, ci, s, 5)
    assert (a == b).all()
    assert 0.17 < a.mean() < 0.23


# --------------------------------------------------------------------------- 미래 누출

def test_probs_ignore_prices_after_t():
    """t 이후 가격을 바꿔도 t ≤ T 사건의 확률·사건 집합은 그대로(라벨만 바뀐다)."""
    T = pd.Timestamp("2019-06-14")
    Xa = R.prepare(_synth(), _all_eps)
    Xb = R.prepare(_synth(perturb_after=T), _all_eps)
    k = 5
    ea, eb = R.events_union(Xa, k), R.events_union(Xb, k)
    ka = Xa["days"][ea["s"]] <= T
    kb = Xb["days"][eb["s"]] <= T
    assert ka.sum() > 0
    for f in ("ci", "s", "judged", "r1", "r2", "r3"):
        assert (ea[f][ka] == eb[f][kb]).all()
    for q, N in ((2, 20), (1, 20), (0, 0)):
        pa = R.cell_probs(Xa, q, N, k, ea["ci"][ka], ea["s"][ka])
        pb = R.cell_probs(Xb, q, N, k, eb["ci"][kb], eb["s"][kb])
        assert np.isfinite(pa["p_align"]).sum() > 0
        for key in R.KEYS:
            np.testing.assert_array_equal(pa[key], pb[key])


# --------------------------------------------------------------------------- 짝짓기

def test_pairing_drops_event_missing_for_any_q():
    n = 6
    y = np.array([1.0, 0.0, 1.0, np.nan, 0.0, 1.0])
    def P(nan_at=(), nopool_at=()):
        d = {key: np.full(n, 0.4) for key in R.KEYS}
        for key in R.KEYS:
            d[key][list(nan_at) + list(nopool_at)] = np.nan
        d["nopool"] = np.zeros(n, bool)
        d["nopool"][list(nopool_at)] = True
        return d
    probs = {2: P(), 1: P(nan_at=(1,)), 0: P(nopool_at=(4,))}
    ok, drops = R.pair(probs, y)
    assert ok.tolist() == [True, False, True, False, False, True]
    assert drops["n_in"] == 6 and drops["n_paired"] == 3
    assert drops["label"] == 1 and drops["q1_nan"] == 1 and drops["q0_nopool"] == 1 and drops["q2_nan"] == 0


def test_pairing_rejects_key_mismatch():
    n = 3
    probs = {q: {key: np.full(n, 0.4) for key in R.KEYS} | {"nopool": np.zeros(n, bool)} for q in R.QS}
    probs[2]["p_up_5_20"][0] = np.nan  # p_align 는 유한한데 다른 키만 NaN — 커널 규약 위반
    with pytest.raises(AssertionError):
        R.pair(probs, np.ones(n))


# --------------------------------------------------------------------------- run_cell

@pytest.fixture(scope="module")
def cell(X, tmp_path_factory):
    d = tmp_path_factory.mktemp("scratch")
    return R.run_cell(X, 20, 5, d), d


def test_run_cell_writes_arrays(cell):
    c, d = cell
    for key in ("N", "k", "verdict", "verdict_q1", "judged", "reports", "p_up", "drops", "n_events"):
        assert key in c
    assert set(c["reports"]) == {"r1", "r2", "r3"}
    assert set(c["p_up"]) == {"p_up_5_20", "p_up_20_60", "p_up_60_120"}
    assert c["verdict"]["n"] == c["n_events"]["judged"] > 0
    j = c["judged"]
    for key in ("bss_2v1", "bss_1v0", "resolution", "reliability", "verdict_2v0", "verdict_1v0"):
        assert key in j
    assert set(j["resolution"]) == {"q2", "q1", "q0"}
    z = np.load(d / "cell_N20_k5.npz")
    n = c["n_events"]["judged"]
    for key in ("ci", "s", "month", "year", "judged", "r1", "r2", "r3"):
        assert key in z.files
    for q in R.QS:
        for key in R.KEYS:
            assert f"q{q}_{key}" in z.files
    for key in R.KEYS:
        assert f"y_{key}" in z.files
    assert int(z["judged"].sum()) == n
    # 짝지은 사건만 저장: 세 q 전부 유한
    for q in R.QS:
        assert np.isfinite(z[f"q{q}_p_align"]).all()
    for q, N in ((2, 20), (1, 20), (0, 0)):
        assert (d / f"probs_q{q}_N{N}_k5.npz").exists()


def test_run_cell_months_and_years_in_window(cell):
    _, d = cell
    z = np.load(d / "cell_N20_k5.npz")
    assert (z["year"] >= 2017).all() and (z["year"] <= 2023).all()
    assert ((z["month"] // 100) == z["year"]).all()


# --------------------------------------------------------------------------- 출력

def test_json_strict_nan_to_null(tmp_path):
    p = tmp_path / "x.json"
    R.write_json({"a": np.nan, "b": [np.inf, np.float32(1.5), -np.inf], "c": {(10, 3): np.int64(2)}, "d": np.bool_(True),
                  "e": np.array([1.0, np.nan])}, p)
    txt = p.read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    assert json.loads(txt) == {"a": None, "b": [None, 1.5, None], "c": {"10,3": 2}, "d": True, "e": [1.0, None]}


def test_trials_append_one_line(tmp_path):
    p = tmp_path / "TRIALS.jsonl"
    p.write_text('{"trial":"old1"}\n{"trial":"old2"}\n')
    R.append_trial(p, {"trial": "gate0", "verdict": {"pass": False, "x": np.nan}})
    lines = p.read_text().splitlines()
    assert lines[:2] == ['{"trial":"old1"}', '{"trial":"old2"}']
    assert len(lines) == 3 and json.loads(lines[2])["trial"] == "gate0"
    assert json.loads(lines[2])["verdict"]["x"] is None
    q = tmp_path / "BAD.jsonl"
    q.write_text('{"trial":"old"}')          # 끝 줄바꿈 없음 → 덧붙이면 두 줄이 붙는다
    with pytest.raises(ValueError):
        R.append_trial(q, {"trial": "gate0"})
    assert q.read_text() == '{"trial":"old"}'


def test_rerun_guard(tmp_path):
    R.guard_rerun(tmp_path)                   # 비어 있으면 통과
    R.append_trial(tmp_path / "TRIALS.jsonl", {"trial": "gate0"})
    with pytest.raises(SystemExit):
        R.guard_rerun(tmp_path)


def test_results_md_and_gate(cell, tmp_path):
    c, _ = cell
    cells = {(N, k): c for N in R.NS for k in R.KS}
    res = R.summarize(cells)
    assert set(res["gate"]) == {"pass", "n_pass", "largest_block"}
    md = R.results_md(res, meta={"git": "abc", "data": {}})
    assert "(검증 전)" in md
    for N in R.NS:
        assert f"| {N} |" in md
    R.write_json(res, tmp_path / "r.json")


# --------------------------------------------------------------------------- 정지일(정정 MQ-2)

def test_mask_suspensions_counts_and_nans():
    C = np.array([[10.0, 11.0, 12.0, np.nan], [5.0, 5.0, 6.0, 7.0]])
    V = np.array([[100.0, 0.0, np.nan, np.nan], [1.0, 1.0, 0.0, 2.0]])
    PX = C.copy()
    n = R.mask_suspensions(C, V, PX)
    assert n == 3                                     # (0,1) 거래량 0 · (0,2) 결측 · (1,2) 0 — (0,3) 은 이미 NaN
    exp = np.array([[10.0, np.nan, np.nan, np.nan], [5.0, 5.0, np.nan, 7.0]])
    np.testing.assert_array_equal(C, exp)
    np.testing.assert_array_equal(PX, exp)


def _suspend(raw, c, b):
    V = np.ones_like(raw["C"])
    V[c, b] = 0.0
    raw = dict(raw, C=raw["C"].copy(), raw_close=raw["raw_close"].copy())
    R.mask_suspensions(raw["C"], V, raw["raw_close"])
    return raw


def test_suspension_in_ma120_window_removes_event():
    raw = _synth()
    base = R.prepare(raw, _all_eps)
    k = 5
    ci, s = R.build_events(base, k)
    j = len(s) // 2
    c, t = int(ci[j]), int(s[j])
    b = t - 100                                        # MA120 창 안(적합 창·σ 창 밖)
    X = R.prepare(_suspend(raw, c, b), _all_eps)
    ev = set(zip(*R.build_events(X, k, pop="no_f200")))
    assert (c, t) not in ev
    # 정지일이 MA120 창에 드는 이 종목의 t 는 하나도 없다
    for pop in ("judged", "no_f200"):
        cc, ss = R.build_events(X, k, pop=pop)
        assert not ((cc == c) & (ss >= b) & (ss <= b + 119)).any()
    # R3(정배열 아님 ∧ 유니버스, d 거름 없음)는 사건으로 뽑히지만 확률·라벨이 NaN → 짝짓기에서 빠진다
    cc, ss = R.build_events_all(X, k)
    m = (cc == c) & (ss >= b) & (ss <= b + 119 - k)
    assert m.sum() > 0
    for q, N in ((2, 20), (1, 20), (0, 0)):
        assert np.isnan(R.cell_probs(X, q, N, k, cc[m], ss[m])["p_align"]).all()
    assert np.isnan(R.MQ.realized(X["C"], cc[m], ss[m], k)["p_align"]).all()


def test_suspension_removes_library_origins_covering_it():
    from research.signals import ma_quad as MQ
    raw = _synth()
    days = raw["days"]
    c, N, k = 0, 20, 5
    b = int(days.searchsorted(pd.Timestamp("2018-03-05")))
    t_m = int(MQ.month_starts(days)[np.searchsorted(MQ.month_starts(days), b + 120)])

    def origins(Xp):
        fc = MQ.poly_forecast(Xp["y"], N, k, 2)
        z = MQ.forecast_z(Xp["y"], fc, Xp["sigma"])
        _, o = MQ.month_pool(days, z, Xp["member"][N + 60], t_m, k, return_origins=True)
        return o[o[:, 0] == c, 1]

    def hit(u):                                        # 라벨 창 u+1..u+k · 적합 창 u−N+1..u · σ 창 u−60..u 가 b 를 덮는 u
        return (u >= b - k) & (u <= b + 60)
    u0 = origins(R.prepare(raw, _all_eps))
    assert hit(u0).sum() == 60 + k + 1                 # 정지 전에는 그 원점이 전부 있다
    u1 = origins(R.prepare(_suspend(raw, c, b), _all_eps))
    assert hit(u1).sum() == 0
    assert set(u1.tolist()) == set(u0[~hit(u0)].tolist())         # 덮지 않는 원점은 그대로


def test_m_sens_default_on_for_real_run_off_for_smoke(monkeypatch, tmp_path):
    seen = {}

    def stop(*a, **kw):
        raise SystemExit(0)

    monkeypatch.setattr(R, "guard_rerun", lambda *a, **kw: None)
    monkeypatch.setattr(R, "git_sha", lambda: {"sha": "x", "dirty": False})
    monkeypatch.setattr(R, "_connect", stop)
    real_parse = R.argparse.ArgumentParser.parse_args

    def spy(self, argv=None, namespace=None):
        ns = real_parse(self, argv, namespace)
        seen["ns"] = ns
        return ns

    monkeypatch.setattr(R.argparse.ArgumentParser, "parse_args", spy)
    for argv, exp in (([], True), (["--smoke"], False), (["--no-m-sens"], False), (["--smoke", "--m-sens"], True)):
        with pytest.raises(SystemExit):
            R.main(argv + ["--scratch", str(tmp_path / "s"), "--logs", str(tmp_path / "l")])
        assert seen["ns"].m_sens is exp, argv
