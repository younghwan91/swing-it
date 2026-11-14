"""골든 HMM 관문 0 실행기 — 합성 패널만(DB 없음). research/experiments/golden_hmm_gate0.py."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from research.experiments import golden_hmm_gate0 as R  # noqa: E402

LABEL_END = pd.Timestamp(R.LABEL_END)
EVENT_START = pd.Timestamp(R.EVENT_START)
METRIC_WORDS = ("bss", "brier", "ece", "base_rate", "y_rate", "p_mean", "reliab", "확률 평균", "실현율")


# --------------------------------------------------------------------------- 합성 자료

def _panel(start, end, n_codes=4, seed=0):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    t = np.arange(len(days))
    lc = [np.log(5000.0) + 0.2 * np.sin(2 * np.pi * t / (230 + 50 * i)) + np.cumsum(rng.normal(0, 0.012, len(days)))
          for i in range(n_codes)]
    C = np.exp(np.array(lc))
    codes = [f"{i:06d}" for i in range(n_codes)]
    meta = pd.DataFrame({"market": "KOSPI", "excluded": None}, index=pd.Index(codes, name="code"))
    info = {"first_day": str(days[0].date()), "last_day": str(days[-1].date()), "n_codes": n_codes, "n_days": len(days),
            "n_pre_listing_cells": 0, "n_suspension_cells_nan": 0, "n_close_nonpositive": 0, "n_price_rule_from_adj_close": 0}
    return {"days": days, "codes": codes, "C": C, "raw_close": C.copy(), "tv": np.full_like(C, 5000.0), "meta": meta, "info": info}


def _all_eps(ci, s):
    return np.ones(len(ci), bool)


@pytest.fixture(scope="module")
def J():
    return R.prepare_judge(_panel(*R.JUDGE_DATA), _all_eps)


def _mask_X(n_codes=4):
    days = pd.bdate_range("2008-06-02", "2016-08-31")
    shape = (n_codes, len(days))
    return {"days": days, "codes": [f"{i:06d}" for i in range(n_codes)], "U": np.ones(shape, bool), "eps": np.ones(shape, bool),
            "gap": np.full(shape, 0.03)}


@pytest.fixture
def small_fit(monkeypatch):
    monkeypatch.setattr(R, "S_RANGE", (1, 2))
    monkeypatch.setattr(R, "N_ITER", 15)
    monkeypatch.setattr(R, "SMOKE_TRAIN_MOD", 1)


# --------------------------------------------------------------------------- 사건·라벨 (Review Focus 3)

@pytest.mark.parametrize("k", R.KS)
def test_label_end_2016_08_31(k):
    Xm = _mask_X()
    days = Xm["days"]
    ci, s = R.build_events(Xm, k)
    assert len(s) > 0
    assert (days[s + k] <= LABEL_END).all() and (days[s] >= EVENT_START).all()
    last = int(days.searchsorted(LABEL_END, side="right")) - 1 - k
    assert s.max() == last                                    # 경계에 정확히 닿는 마지막 t 는 들어간다
    # 판정 패널이 2016-08-31 뒤까지 있어도(가정) 라벨 끝은 넘지 않는다
    Xl = _mask_X()
    Xl["days"] = pd.bdate_range("2008-06-02", periods=len(days) + 40)
    for key in ("U", "eps", "gap"):
        Xl[key] = np.concatenate([Xl[key], Xl[key][:, :40]], axis=1)
    ci, s = R.build_events(Xl, k)
    assert (Xl["days"][s + k] <= LABEL_END).all()


def test_label_end_on_real_prepare(J):
    days = J["days"]
    assert days[-1] <= LABEL_END
    for k in R.KS:
        ci, s = R.build_events(J, k)
        assert len(s) > 0
        assert (s + k < len(days)).all() and (days[s + k] <= LABEL_END).all() and (days[s] >= EVENT_START).all()
        y = R.labels(J, ci, s, k)
        assert set(np.unique(y[np.isfinite(y)])) <= {0.0, 1.0}


def test_events_are_precandidates():
    Xm = _mask_X()
    days = Xm["days"]
    t0 = int(days.searchsorted(pd.Timestamp("2012-03-02")))
    Xm["gap"][0, t0] = 0.0          # MA50 = MA200 → 빠짐(엄격 아래)
    Xm["gap"][1, t0] = -0.01        # MA50 > MA200 → 빠짐
    Xm["gap"][2, t0] = 0.05         # 격차 5% → 들어감(≤ 0.05)
    Xm["gap"][3, t0] = 0.0500001    # 5% 초과 → 빠짐
    Xm["U"][2, t0 + 1] = False      # 유니버스 밖
    Xm["eps"][2, t0 + 2] = False    # EPS 아님
    Xm["gap"][2, t0 + 3] = np.nan   # 이평선 결측
    ev = set(zip(*R.build_events(Xm, 10)))
    assert (0, t0) not in ev and (1, t0) not in ev and (3, t0) not in ev and (2, t0) in ev
    assert (2, t0 + 1) not in ev and (2, t0 + 2) not in ev and (2, t0 + 3) not in ev
    ci, s = R.build_events(Xm, 10)
    assert (np.diff(s) >= 0).all()


def test_prepare_gap_and_universe(J):
    ma50, ma200 = J["ma"][50], J["ma"][200]
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.allclose(J["gap"], 1 - ma50 / ma200, equal_nan=True)
    assert J["U"].dtype == bool and J["member"].dtype == bool
    assert (J["member"] | ~J["U"]).all()                      # 상장 70일 모집단 ⊇ 252일 유니버스
    assert J["obs"].shape == J["C"].shape + (3,)


# --------------------------------------------------------------------------- 학습 구간 (Review Focus 5)

def test_train_window_disjoint():
    assert pd.Timestamp(R.TRAIN_DATA[0]) > pd.Timestamp(R.JUDGE_DATA[1])
    assert pd.Timestamp(R.TRAIN_DATA[1]) <= pd.Timestamp("2023-12-31")
    assert pd.Timestamp(R.LABEL_END) <= pd.Timestamp(R.JUDGE_DATA[1]) < pd.Timestamp(R.TRAIN_DATA[0])
    R.check_windows(pd.bdate_range(*R.TRAIN_DATA), pd.bdate_range(*R.JUDGE_DATA))
    with pytest.raises(RuntimeError):
        R.check_windows(pd.bdate_range("2016-08-01", "2023-12-29"), pd.bdate_range(*R.JUDGE_DATA))
    with pytest.raises(RuntimeError):
        R.check_windows(pd.bdate_range("2016-09-01", "2024-01-02"), pd.bdate_range(*R.JUDGE_DATA))


def test_load_requests_only_its_window(monkeypatch):
    """정문에 넘기는 구간: 학습 2016-09-01~2023-12-31, 판정 2007-11-22~2016-08-31 — 2024~ 요청 없음."""
    import swing_it.features.korea as K
    import swing_it.storage as ST

    seen = []

    def fake_wide(con, *, cols, table, start, end, session, **kw):
        seen.append(("wide", table, start, end, session))
        days = pd.bdate_range(start, end)
        df = pd.DataFrame(1000.0, index=days, columns=["000001"])
        return {c: df.copy() for c in cols}

    def fake_long(con, *, cols, table, start, end, session, parse_dates):
        seen.append(("long", table, start, end, session))
        return pd.DataFrame({"code": pd.Series([], dtype=str), "date": pd.Series([], dtype="datetime64[ns]"),
                             "close_src": pd.Series([], dtype=str)})

    monkeypatch.setattr(ST, "read_prices_wide", fake_wide)
    monkeypatch.setattr(ST, "read_prices", fake_long)
    monkeypatch.setattr(K, "load_listing_meta", lambda con: pd.DataFrame({"market": ["KOSPI"], "excluded": [None]},
                                                                         index=pd.Index(["000001"], name="code")))
    a = R.load(None, R.TRAIN_DATA)
    b = R.load(None, R.JUDGE_DATA)
    assert a["days"][0] >= pd.Timestamp("2016-09-01") and a["days"][-1] <= pd.Timestamp("2023-12-31")
    assert b["days"][-1] <= pd.Timestamp("2016-08-31")
    for _, table, start, end, session in seen:
        assert table == "daily_bars_adj_v2" and session == "raw"
        assert pd.Timestamp(end) < pd.Timestamp("2024-01-01")
        assert (start, end) in (R.TRAIN_DATA, R.JUDGE_DATA)


def test_train_set_uses_train_panel_only(small_fit):
    T = _panel(*R.TRAIN_DATA, seed=3)
    seqs, info = R.train_set(T)
    assert info["first_day"] >= R.TRAIN_DATA[0] and info["last_day"] <= R.TRAIN_DATA[1]
    assert info["n_seq"] >= 1 and all(len(q) >= R.TRAIN_MIN_LEN for q in seqs)


# --------------------------------------------------------------------------- 정지일·짝짓기

def test_load_suspension_and_prelisting_nan(monkeypatch):
    import swing_it.features.korea as K
    import swing_it.storage as ST

    days = pd.bdate_range(*R.JUDGE_DATA)
    close = pd.DataFrame({"000001": 1000.0, "000002": 2000.0}, index=days)
    vol = pd.DataFrame({"000001": 10.0, "000002": 10.0}, index=days)
    vol.iloc[100, 0] = 0.0                                    # 정지일 → 종가 NaN
    close.iloc[5, 1] = np.nan                                 # pre_listing(이미 NaN 이어야 한다)
    close.iloc[200, 1] = -1.0                                 # 0 이하 → NaN
    rc = close.copy()
    rc["000002"] = np.nan                                     # 폐지(원 종가 없음) → 1,000원 규칙은 수정 종가

    def fake_wide(con, *, cols, table, start, end, session, **kw):
        m = {"close": close, "trade_value": close * 0 + 5000.0, "raw_close": rc, "volume": vol}
        return {c: m[c].copy() for c in cols}

    def fake_long(con, *, cols, table, start, end, session, parse_dates):
        return pd.DataFrame({"code": ["000002"], "date": [days[5]], "close_src": ["pre_listing"]})

    monkeypatch.setattr(ST, "read_prices_wide", fake_wide)
    monkeypatch.setattr(ST, "read_prices", fake_long)
    monkeypatch.setattr(K, "load_listing_meta", lambda con: pd.DataFrame({"market": "KOSPI", "excluded": None},
                                                                         index=pd.Index(["000001", "000002"], name="code")))
    X = R.load(None, R.JUDGE_DATA)
    assert np.isnan(X["C"][0, 100]) and np.isfinite(X["C"][0, 99])
    assert np.isnan(X["C"][1, 200]) and np.isnan(X["C"][1, 5])
    assert X["info"]["n_suspension_cells_nan"] == 1 and X["info"]["n_pre_listing_cells"] == 1
    assert np.allclose(X["raw_close"][1, 300], 2000.0)


def test_pairing_drop_reasons():
    y = np.array([1.0, 0.0, np.nan, 1.0, 0.0, 1.0])
    probs = {"hmm": np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5]),
             "hmm1": np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5]),
             "q0": np.array([0.2, 0.2, 0.2, np.nan, np.nan, 0.6])}
    nopool = np.array([False, False, False, True, False, False])
    ok, drops = R.pair(probs, nopool, y)
    assert ok.tolist() == [True, False, False, False, False, True]
    assert drops == {"n_in": 6, "label": 1, "hmm_nan": 1, "q0_nopool": 1, "q0_nan": 1, "hmm1_nan": 1, "n_paired": 2}


# --------------------------------------------------------------------------- 판정 규칙

def test_cell_years_bar_is_6_of_8(monkeypatch):
    base = {"calib_ok": True, "skill_ok": True, "years_ok": True, "pass": True, "years_pos": 5, "n": 10}
    monkeypatch.setattr(R.CAL, "cell_verdict", lambda *a, **kw: dict(base))
    v = R.verdict(*(np.zeros(3),) * 5)
    assert v["years_ok"] is False and v["pass"] is False and v["years_min_pos"] == 6
    base["years_pos"] = 6
    v = R.verdict(*(np.zeros(3),) * 5)
    assert v["years_ok"] is True and v["pass"] is True


def test_gate_requires_k10_and_two_cells():
    assert R.gate({5: True, 10: True, 15: False})["pass"]
    assert not R.gate({5: True, 10: False, 15: True})["pass"]
    assert not R.gate({5: False, 10: True, 15: False})["pass"]
    assert R.gate({5: True, 10: True, 15: True})["n_pass"] == 3


# --------------------------------------------------------------------------- 출력·재실행 방지

def test_json_strict_nan_to_null(tmp_path):
    p = tmp_path / "x.json"
    R.write_json({"a": np.nan, "b": {5: np.float32(0.5)}, "c": [np.inf]}, p)
    assert json.loads(p.read_text()) == {"a": None, "b": {"5": 0.5}, "c": [None]}


def test_real_run_refuses_noncanonical_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_connect", lambda: (_ for _ in ()).throw(AssertionError("DB 에 붙으면 안 된다")))
    monkeypatch.setattr(R, "require_clean_tree", lambda *a, **kw: None)
    with pytest.raises(SystemExit, match="정본"):
        R.main(["--logs", str(tmp_path), "--scratch", str(tmp_path / "s")])


def test_real_run_requires_clean_tree_before_db(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(R, "require_clean_tree", lambda *a, **kw: seen.append("clean") or (_ for _ in ()).throw(SystemExit("dirty")))
    monkeypatch.setattr(R, "_connect", lambda: seen.append("db"))
    with pytest.raises(SystemExit, match="dirty"):
        R.main(["--scratch", str(tmp_path / "s")])
    assert seen == ["clean"]


def _fake_db(monkeypatch):
    monkeypatch.setattr(R, "_connect", lambda: (_FakeCon(), lambda codes, days, ci, s, con, source: np.ones(len(ci), bool)))
    monkeypatch.setattr(R, "load", lambda con, data: _panel(*data, seed=1 if data == R.JUDGE_DATA else 2))


class _FakeCon:
    def close(self):
        pass


def test_end_to_end_writes_once_and_refuses_rerun(tmp_path, monkeypatch, small_fit, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "PREREG.md").write_text("x")
    (logs / "TRIALS.jsonl").write_text('{"trial":"prereg"}\n')
    monkeypatch.setattr(R, "LOGS", logs)
    monkeypatch.setattr(R, "require_clean_tree", lambda *a, **kw: None)
    monkeypatch.setattr(R, "MSENS", {**R.MSENS, "mod": 3})
    _fake_db(monkeypatch)
    assert R.main(["--logs", str(logs), "--scratch", str(tmp_path / "s"), "--workers", "2"]) == 0
    res = json.loads((logs / R.OUT_JSON).read_text())
    assert set(res["cells"]) == {"5", "10", "15"} and set(res["gate"]) >= {"pass", "n_pass", "k10_pass"}
    for k in ("5", "10", "15"):
        c = res["cells"][k]
        assert c["n_events"] > 0 and c["verdict"]["years_min_pos"] == 6
        assert set(c["reports"]) >= {"s1_vs_q0", "hmm_vs_s1", "gap_bins", "yearly_bss", "reliability"}
        assert c["drops"]["n_paired"] == c["n_events"]
    assert {"S", "bic_table", "fits", "selected", "s1", "train"} <= set(res["hmm"])
    assert res["m_sens"]["n_sample"] >= 1
    md = (logs / R.OUT_MD).read_text()
    assert "(검증 전)" in md
    lines = (logs / "TRIALS.jsonl").read_text().splitlines()
    assert len(lines) == 2 and lines[0] == '{"trial":"prereg"}' and json.loads(lines[1])["trial"] == "gate0"
    def _bad(c):
        raise AssertionError(f"엄격 JSON 아님: {c}")
    json.loads((logs / R.OUT_JSON).read_text(), parse_constant=_bad)      # NaN·Infinity 토큰 없음
    with pytest.raises(SystemExit):
        R.main(["--logs", str(logs), "--scratch", str(tmp_path / "s")])
    assert len((logs / "TRIALS.jsonl").read_text().splitlines()) == 2


def test_smoke_prints_no_metrics_and_writes_no_logs(tmp_path, monkeypatch, small_fit, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(R, "LOGS", logs)
    monkeypatch.setattr(R, "require_clean_tree", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("스모크는 트리 검사 안 함")))
    _fake_db(monkeypatch)
    assert R.main(["--smoke", "--scratch", str(tmp_path / "s"), "--workers", "2"]) == 0
    out = capsys.readouterr().out.lower()
    assert "예상" in out and "hmm" in out
    for w in METRIC_WORDS:
        assert w not in out, w
    assert list(logs.iterdir()) == []


def test_config_names_prereg():
    cfg = R.config()
    assert cfg["prereg"].startswith("research/logs/golden_hmm/PREREG.md")
    assert cfg["ks"] == (5, 10, 15) and cfg["gap_max"] == 0.05 and cfg["years_min_pos"] == 6
    assert cfg["eps_source"] == "earnings_dart_doc" and cfg["M"] == 200
