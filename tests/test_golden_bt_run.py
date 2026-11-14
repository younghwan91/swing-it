"""골든 임박 백테스트 실행기 — 합성 패널만(DB 없음). research/experiments/golden_bt_run.py."""
import json

import numpy as np
import pandas as pd
import pytest

from research.experiments import golden_bt_run as R
from research.signals import golden_bt_wf as W
from research.signals import ma_quad as MQ

METRIC_WORDS = ("R 합", "R_sum", "R 평균", "cagr", "CAGR", "연복리", "alpha", "알파", "pass", "통과", "DSR", "payoff", "손익비", "수익")


# --------------------------------------------------------------------------------------------- 합성 자료

def _raw(win, n_codes=6, seed=0):
    days = pd.bdate_range(*win["data"])
    rng = np.random.default_rng(seed)
    t = np.arange(len(days))
    lc = [np.log(5000.0) + 0.18 * np.sin(2 * np.pi * t / (260 + 45 * i)) + np.cumsum(rng.normal(0, 0.012, len(days))) for i in range(n_codes)]
    C = np.exp(np.array(lc))
    O = C * np.exp(rng.normal(0, 0.004, C.shape))  # noqa: E741
    codes = [f"{i:06d}" for i in range(n_codes)]
    meta = pd.DataFrame({"market": ["KOSPI", "KOSDAQ"] * (n_codes // 2), "excluded": None}, index=pd.Index(codes, name="code"))
    kx = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))) * 2000, index=days)
    info = {"first_day": str(days[0].date()), "last_day": str(days[-1].date()), "n_codes": n_codes, "n_days": len(days)}
    return {"days": days, "codes": codes, "O": O, "C": C, "raw_close": C.copy(), "tv": np.full_like(C, 5000.0), "cap": np.full_like(C, 1e11),
            "meta": meta, "kx": kx, "kq": kx * 0.4, "info": info, "win": win}


def _all_eps(ci, s):
    return np.ones(len(ci), bool)


@pytest.fixture(scope="module")
def X1():
    return R.prepare(_raw(R.MAIN), _all_eps)


@pytest.fixture(scope="module")
def X2():
    return R.prepare(_raw(R.REP, seed=1), _all_eps)


# --------------------------------------------------------------------------------------------- 자료 규약

def test_windows_never_2024():
    assert R.MAIN["data"][1] < "2024-01-01" and R.MAIN["end"] == "2023-12-28" and R.REP["data"] == ("2007-11-22", "2016-08-31")
    R.check_window(pd.bdate_range(*R.MAIN["data"]), R.MAIN)
    with pytest.raises(RuntimeError):
        R.check_window(pd.bdate_range("2013-06-03", "2024-01-03"), R.MAIN)


def _fake_db(monkeypatch, seen, close, vol, opn, pl=()):
    import swing_it.features.korea as K
    import swing_it.storage as ST

    def fake_wide(con, *, cols, table, start, end, session, **kw):
        seen.append(("wide", table, start, end, session))
        m = {"close": close, "open": opn, "trade_value": close * 0 + 5000.0, "raw_close": close.copy(), "volume": vol}
        return {c: m[c].reindex(pd.bdate_range(start, end)).copy() for c in cols}

    def fake_long(con, *, cols, table, start, end, session, parse_dates):
        seen.append(("long", table, start, end, session))
        return pd.DataFrame({"code": [p[0] for p in pl], "date": [p[1] for p in pl], "close_src": ["pre_listing"] * len(pl)})

    def fake_shares(con, **kw):
        seen.append(("shares", kw.get("with_asof")))
        return pd.DataFrame({"code": ["000001"], "date": [pd.Timestamp("2016-03-02")], "shares_outstanding": [1e7],
                             "knowledge_date": [pd.Timestamp("2016-03-02")], "asof": [pd.Timestamp("2016-03-02")]})

    def fake_earn(con, **kw):
        seen.append(("earn", kw.get("table")))
        return pd.DataFrame({"code": ["000001"], "period": ["2008Q1"], "knowledge_date": [pd.Timestamp("2008-05-15")], "shares_common": [1e7]})

    def fake_index(con, code, **kw):
        seen.append(("index", code, kw.get("start"), kw.get("end"), kw.get("proxy") is not None))
        d = pd.bdate_range(kw["start"], kw["end"])
        return pd.DataFrame({"date": d, "close": 100, "level": 1.0})

    monkeypatch.setattr(ST, "read_prices_wide", fake_wide)
    monkeypatch.setattr(ST, "read_prices", fake_long)
    monkeypatch.setattr(ST, "read_shares", fake_shares)
    monkeypatch.setattr(ST, "read_earnings", fake_earn)
    monkeypatch.setattr(ST, "read_index", fake_index)
    monkeypatch.setattr(K, "load_listing_meta", lambda con: pd.DataFrame({"market": "KOSPI", "excluded": None},
                                                                         index=pd.Index(["000001", "000002"], name="code")))


def test_load_requests_only_its_window_and_masks(monkeypatch):
    for win in (R.MAIN, R.REP):
        seen = []
        days = pd.bdate_range(*win["data"])
        close = pd.DataFrame({"000001": 1000.0, "000002": 2000.0}, index=days)
        opn = close * 1.01
        vol = close * 0 + 10.0
        vol.iloc[100, 0] = 0.0                                   # 정지일 → 시가·종가 NaN
        opn.iloc[150, 1] = 0.0                                   # 시가 0(네이버 정지일) → NaN, 종가 유지
        close.iloc[5, 1] = np.nan                                # pre_listing(이미 NaN 이어야)
        close.iloc[200, 1] = -1.0
        _fake_db(monkeypatch, seen, close, vol, opn, pl=[("000002", days[5])])
        X = R.load(None, win)
        assert np.isnan(X["C"][0, 100]) and np.isnan(X["O"][0, 100]) and np.isfinite(X["C"][0, 99])
        assert np.isnan(X["O"][1, 150]) and np.isfinite(X["C"][1, 150])
        assert np.isnan(X["C"][1, 200]) and np.isnan(X["O"][1, 200]) and np.isnan(X["C"][1, 5])
        assert X["info"]["n_suspension_cells_nan"] == 1 and X["info"]["n_pre_listing_cells"] == 1
        for s in seen:
            if s[0] in ("wide", "long"):
                assert s[1] == "daily_bars_adj_v2" and s[4] == "raw" and (s[2], s[3]) == win["data"]
            if s[0] == "index":
                assert s[3] == win["data"][1] < "2024-01-01" and s[4]
        assert ("earn", "earnings_dart_doc") in seen if win is R.REP else ("shares", True) in seen
        assert X["days"][-1] < pd.Timestamp("2024-01-01")


def test_prepare_pool_definition(X1):
    s0, e0 = X1["bounds"]
    assert X1["days"][s0] == pd.Timestamp("2017-10-02") and X1["days"][e0] == pd.Timestamp("2023-12-28")
    ci, s = np.nonzero(X1["pool"])
    assert len(ci) > 0
    gap, d = X1["gap"][ci, s], 1 - X1["ma50"][ci, s] / X1["C"][ci, s]
    assert ((gap > 0) & (gap <= 0.05)).all() and ((d > 0) & (d <= 0.15)).all() and X1["U"][ci, s].all()
    assert s.min() >= s0 - 1 and s.max() <= e0 - 1                 # 판정일 창
    assert X1["mkt"].tolist() == [0, 1] * 3
    assert X1["lu"][0] == 1.14 and X1["lu"][-1] == 1.29 and X1["cost"].shape == (2, len(X1["days"]))


def test_lu_and_cost_by_date():
    d = pd.DatetimeIndex(["2015-06-12", "2015-06-15", "2023-01-02"])
    assert R.lu_array(d).tolist() == [1.14, 1.29, 1.29]
    from swing_it.features.korea import cost_rt
    c = R.cost_table(d)
    assert c[0, 2] == cost_rt("2023-01-02", "KOSPI") and c[1, 0] == cost_rt("2015-06-12", "KOSDAQ")


# --------------------------------------------------------------------------------------------- 확률 패널

def test_prob_panels_match_golden_hmm_q0(X2):
    """P_g = golden_hmm 관문 0 의 q0_probs 와 같은 값(같은 풀·같은 색인), P_d = 같은 경로 dead=True. 계산 칸 밖은 NaN."""
    pytest.importorskip("hmmlearn")
    from research.experiments import golden_hmm_gate0 as GHG

    cells = R.prob_cells(X2)
    months = sorted(cells)[::7][:4]
    P = R.prob_panels(X2, 1, months=months)
    ci = np.concatenate([cells[m][0] for m in months])
    s = np.concatenate([cells[m][1] for m in months])
    q, _, _ = GHG.q0_probs(X2, R.K, ci, s, M=R.M_PATHS)
    got = P["PG"][ci, s].astype(np.float64)
    assert np.isfinite(got).sum() > 0
    assert np.allclose(got, q.astype(np.float32), equal_nan=True)
    mask = np.zeros(X2["C"].shape, bool)
    mask[ci, s] = True
    assert np.isnan(P["PG"][~mask]).all() and np.isnan(P["PD"][~mask]).all()
    assert np.array_equal(np.isfinite(P["PG"]), np.isfinite(P["PD"]))


def test_prob_cells_cover_held_days(X1):
    cells = R.prob_cells(X1)
    ci = np.concatenate([v[0] for v in cells.values()])
    s = np.concatenate([v[1] for v in cells.values()])
    want = np.zeros(X1["C"].shape, bool)
    want[ci, s] = True
    pool = X1["pool"]
    assert (want | ~pool).all()                                   # 풀 칸은 전부 계산
    _, e0 = X1["bounds"]
    for c in np.flatnonzero(pool.any(axis=1)):
        f = int(np.argmax(pool[c]))
        assert (want[c, f : e0 + 1] == np.isfinite(X1["C"][c, f : e0 + 1])).all()


def test_probs_ignore_prices_after_t(X2):
    """달 t_m 칸의 확률은 그 칸 날짜 뒤 가격을 쓰지 않는다."""
    cells = R.prob_cells(X2)
    m = sorted(cells)[len(cells) // 2]
    ci, s = cells[m]
    j = int(np.argmax(s))
    a = R.prob_panels(X2, 1, months=[m])["PG"][ci[j], s[j]]
    Y = dict(X2)
    C = X2["C"].copy()
    C[:, s[j] + 1 :] *= 1.7
    Y["C"] = C
    with np.errstate(invalid="ignore", divide="ignore"):
        Y["y"] = np.where(C > 0, np.log(C), np.nan)
    Y["sigma"] = MQ.rolling_sigma(Y["y"], R.SIGMA_W)
    b = R.prob_panels(Y, 1, months=[m])["PG"][ci[j], s[j]]
    assert np.isfinite(a) and a == b


# --------------------------------------------------------------------------------------------- 전 과정(작은 합성)

@pytest.fixture(scope="module")
def full(X1, X2):
    mp = pytest.MonkeyPatch()
    mp.setattr(W, "N_NULL", 12)
    try:
        P1, P2 = R.prob_panels(X1, 4), R.prob_panels(X2, 4)
        D1, D2 = R.sim_data(X1, P1), R.sim_data(X2, P2)
        res = R.pipeline(X1, D1, X2, D2, P1["info"], P2["info"], 4)
    finally:
        mp.undo()
    return res


def test_pipeline_assembles_strict_json_and_md(full, tmp_path):
    out = {k: v for k, v in full.items() if not k.startswith("_")}
    meta = {"git": "abc", "rulings": R.RULINGS, "config": R.config()}
    text = json.dumps(R.MQC._clean({**meta, **out}), allow_nan=False, ensure_ascii=False)
    back = json.loads(text)
    assert set(back["verdict"]) == {"main_pass", "main_void", "rep_pass", "rep_void", "both_pass"}
    assert len(back["main"]["selection"]) == 4 and back["rep"]["cfg_i"] == back["main"]["selection"][-1]["cfg_i"]
    md = R.results_md(out, meta)
    assert md.startswith("# 골든 임박 자리 포지션 백테스트 결과 (검증 전)")
    tr = full["_trades"]["main"]
    assert set(tr["fold"]) <= set(R.FOLD_YEARS)
    for y in R.FOLD_YEARS:                                          # 폴드 Y 진입은 그 해 안
        ed = pd.to_datetime(tr.loc[tr["fold"] == y, "entry_day"])
        assert (ed.dt.year == y).all()
    assert (pd.to_datetime(tr["exit_day"]) <= pd.Timestamp("2023-12-28")).all()
    rt = full["_trades"]["rep"]
    assert (pd.to_datetime(rt["exit_day"]) <= pd.Timestamp("2016-08-31")).all()
    assert (pd.to_datetime(rt["entry_day"]) >= pd.Timestamp("2009-01-02")).all()


def test_selection_uses_train_scores_only(full):
    sc = np.asarray(full["main"]["train_scores"])
    for f, s in enumerate(full["main"]["selection"]):
        assert s["cfg_i"] == W.select(sc[:, f])


# --------------------------------------------------------------------------------------------- 스모크·출력·가드

def test_smoke_prints_no_metrics(X1, X2, capsys):
    class A:
        workers = 2
    R._smoke({"main": X1, "rep": X2}, A())
    out = capsys.readouterr().out
    assert "예상" in out
    for w in METRIC_WORDS:
        assert w not in out, w


def test_guard_and_write_outputs(tmp_path):
    logs = tmp_path / "logs"
    R.guard_rerun(logs)
    R.write_outputs(logs, "{}\n", "# x\n", json.dumps({"trial": R.TRIAL_NAME}))
    assert (logs / R.OUT_JSON).exists() and (logs / R.OUT_MD).exists()
    lines = (logs / R.OUT_TRIALS).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["trial"] == R.TRIAL_NAME
    with pytest.raises(SystemExit):
        R.guard_rerun(logs)
    with pytest.raises(RuntimeError):
        R.write_outputs(logs, "{}\n", "# x\n", json.dumps({"trial": R.TRIAL_NAME}))
    other = tmp_path / "o"
    other.mkdir()
    (other / R.OUT_TRIALS).write_text(json.dumps({"trial": "other"}) + "\n")
    R.guard_rerun(other)                                             # 다른 시도 줄은 막지 않는다
    (other / R.OUT_MD).write_text("half")
    with pytest.raises(SystemExit):
        R.guard_rerun(other)                                         # 반쯤 쓴 출력도 막는다


def test_write_order_trials_last(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(R.MQC, "_write_new", lambda p, t: order.append(p.name) or p.write_text(t))
    monkeypatch.setattr(R.MQC, "append_line", lambda p, t: order.append(p.name))
    R.write_outputs(tmp_path / "l", "{}", "#", "{}")
    assert order == [R.OUT_JSON, R.OUT_MD, R.OUT_TRIALS]


def test_real_run_requires_clean_tree_before_db(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "require_clean_tree", lambda *a, **kw: seen.append("clean") or (_ for _ in ()).throw(SystemExit("dirty")))
    monkeypatch.setattr(R, "_connect", lambda: seen.append("db"))
    monkeypatch.setattr(R, "guard_rerun", lambda *a, **kw: None)
    with pytest.raises(SystemExit, match="dirty"):
        R.main(["--scratch", str(tmp_path / "s")])
    assert seen == ["clean"]


def test_real_run_refuses_noncanonical_logs(tmp_path):
    with pytest.raises(SystemExit, match="정본"):
        R.main(["--logs", str(tmp_path), "--scratch", str(tmp_path / "s")])


def test_rulings_and_config():
    for k in ("probs", "fill", "sizing_d", "prob_topup", "golden", "null2", "bar4", "bar7", "oos_equity", "rep", "neighbors"):
        assert k in R.RULINGS
    c = R.config()
    assert c["n_trials"] == sum(R.DSR_N.values()) and "07f7ce9" in c["prereg"]
    assert c["slots"] == 20 and c["w_cap"] == 0.05 and c["risk"] == 0.0025 and c["cost_judge"] == 2.0 and c["n_null"] == 300
