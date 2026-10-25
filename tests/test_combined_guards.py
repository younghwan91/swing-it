"""결합 루프 평가기 가드 — 위반을 주입해 실제로 막히는지 본다(GUARDRAILS §4.5: 통과만 보는 테스트는 규칙이 죽어도 초록)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from research.autoresearch import prepare as V4
from research.combined import prepare as P

_REAL_PIN = P.pin_ref
TOY_FOLDS = [("2022-01-03", "2023-01-02", "2024-01-01"), ("2022-01-03", "2024-01-01", "2025-01-01")]


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    """원장·BEST 는 임시 경로로, 청산선 캐시는 비운다(장난감끼리 era·모양이 같아 섞이지 않게)."""
    monkeypatch.setattr(P, "RUNS", str(tmp_path / "RUNS.jsonl"))
    monkeypatch.setattr(P, "BEST", str(tmp_path / "BEST.json"))
    monkeypatch.setattr(P, "SHADOW", str(tmp_path / "shadow"))
    monkeypatch.setattr(P, "BEST_HISTORY", str(tmp_path / "BEST_HISTORY.jsonl"))
    monkeypatch.setattr(P, "REFERENCE", str(tmp_path / "REFERENCE.json"))
    monkeypatch.setattr(P, "pin_ref", lambda sha, gs: None)
    monkeypatch.setattr(P, "current_branch", lambda: "combined-loop-1")             # 평가는 섬 브랜치에서(섬 시험은 따로 바꾼다)                       # 진짜 레포에 refs 를 만들지 않는다(전용 시험은 _REAL_PIN)
    V4._EXIT.clear()
    yield
    V4._EXIT.clear()


def _toy(n: int = 3, t: int = 60, start: str = "2012-01-02", seed: "int | None" = None, era: str = "T") -> dict:
    days = pd.bdate_range(start, periods=t)
    if seed is None:
        C = np.tile(np.linspace(100, 160, t), (n, 1))
    else:
        C = 100 * np.cumprod(1 + np.random.default_rng(seed).normal(0.0005, 0.02, (n, t)), axis=1)
    ones = np.ones((n, t), bool)
    D = {"days": days, "codes": [f"{i:06d}" for i in range(n)], "O": C * 0.98, "H": C * 1.01, "L": C * 0.97, "C": C, "TR": ones,
         "uni": ones.copy(), "eps_up": ones.copy(), "hi20": ones.copy(), "cost": np.full((2, t), 0.004), "mkt": np.zeros(n, np.int64),
         "lu": np.full(t, 1.29), "regime_on": np.ones(t, bool), "kx": np.linspace(1000, 1100, t), "kq": np.linspace(500, 520, t),
         "era": era, "tv": C, "adv": C, "cap": C, "eps_yoy": C * 0}
    rng = np.random.default_rng(99 if seed is None else seed + 1)
    D["gap"] = rng.normal(0, 0.01, (n, t)).astype(np.float32)
    D["tickp"] = np.full((n, t), 0.002, np.float32)
    D["gap_event"] = rng.random((n, t)) < 0.1
    for k in ("n_all", "n_feat", "n_move", "n_excl"):
        D[k] = rng.integers(0, 3, (n, t)).astype(np.uint16)
    D["news_known"] = rng.random((n, t)) < 0.7
    return D


def _write_train(path, body: str) -> str:
    path.write_text("import numpy as np\nimport pandas as pd\n" + body +
                    "def get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': lambda x: {'pyramid': 0}}\n")
    return str(path)


# ---------------------------------------------------------------- 체결

def test_entry_is_close():
    D = _toy()
    sig = {"cand": np.zeros_like(D["C"], bool), "score": np.ones_like(D["C"])}
    sig["cand"][0, 10] = True
    _, tr = P.simulate_c(D, sig, {"pyramid": 0}, str(D["days"][1].date()), str(D["days"][-1].date()))
    assert len(tr) == 1 and tr["entry"].iloc[0] == str(D["days"][10].date())        # 신호일 = 진입일
    px_in = D["C"][0, -1] * (1 - 0.004 * P.COST_MULT) / (1 + tr["ret"].iloc[0])        # 끝까지 보유 → 종가 청산에서 역산
    assert px_in == pytest.approx(D["C"][0, 10]) and px_in != pytest.approx(D["O"][0, 11])


def test_eps_guard_and_mask_cannot_be_bypassed():
    D = _toy()
    D["eps_up"][1] = False
    sig = {"cand": np.ones_like(D["C"], bool), "score": np.tile([[1.0], [9.0], [2.0]], (1, D["C"].shape[1]))}
    lo, hi = str(D["days"][1].date()), str(D["days"][-1].date())
    _, tr = P.simulate_c(D, sig, {"max_pos": 5}, lo, hi)
    assert len(tr) and "000001" not in set(tr["code"])
    one = {"cand": np.zeros_like(D["C"], bool), "score": np.ones_like(D["C"])}
    one["cand"][0, 10] = True
    mask = np.zeros_like(D["C"], bool)
    mask[0, 10] = True
    _, tr = P.simulate_c(D, one, {}, lo, hi, mask=mask)
    assert len(tr) == 0                                                              # 상위 1% 제거용 mask 가 그 칸을 끈다


# ---------------------------------------------------------------- 부품 제거

def test_ablation_neutral_values():
    D = _toy()
    V = P.neutral_view(D, "daytrade")
    assert not V["news_known"].any() and all((V[k] == 0).all() for k in ("n_all", "n_feat", "n_move", "n_excl"))
    assert np.array_equal(V["gap"], D["gap"]) and np.array_equal(V["gap_event"], D["gap_event"])   # 다른 레포 키는 그대로
    V = P.neutral_view(D, "scalp")
    assert (V["gap"] == 0).all() and (V["tickp"] == 0).all() and not V["gap_event"].any()
    assert np.array_equal(V["news_known"], D["news_known"]) and V["n_feat"].dtype == D["n_feat"].dtype
    with pytest.raises(ValueError):
        P.neutral_view(D, "swing")


def _toy_loop(monkeypatch):
    D = _toy(n=6, t=780, start="2022-01-03", seed=3)
    monkeypatch.setattr(P, "ERAS", {"A": {"folds": TOY_FOLDS}})
    return {"A": D}


def test_ablation_flags_decorative_component(tmp_path, monkeypatch):
    Ds = _toy_loop(monkeypatch)
    fixed = [{"era": "A", "params": {}}] * 2
    deco = _write_train(tmp_path / "deco.py", "def signals(V, x):\n    g = V['gap'] * 1.0\n    k = V['news_known'] | True\n"
                                              "    return {'cand': (V['C'] > 0) & k, 'score': V['C'] + 0 * np.nan_to_num(g)}\n")
    res = P.walk_forward_c(Ds, 1, deco, fixed=fixed)
    ab = P.ablation(Ds, res, deco)
    assert ab["scalp"]["changed"] == 0 and ab["scalp"]["ok"] is False
    assert ab["daytrade"]["ok"] is False and ab["eligible"] is False
    # 양성 대조: 갭을 실제로 쓰면 체결 집합이 바뀐다(검사가 늘 False 인 것은 아님)
    used = _write_train(tmp_path / "used.py", "def signals(V, x):\n    return {'cand': np.nan_to_num(V['gap']) > 0.005, 'score': V['C']}\n")
    res = P.walk_forward_c(Ds, 1, used, fixed=fixed)
    neu = P.walk_forward_c(Ds, 1, used, fixed=fixed, neutral="scalp")
    assert [f["trade_key"] for f in res["folds"]] != [f["trade_key"] for f in neu["folds"]]
    assert P.ablation(Ds, res, used)["scalp"]["changed"] >= 0.05


def test_active_folds_news_only_2023_2024():
    folds = [{"test": [f"{y}-01-01", f"{y + 1}-01-01"]} for y in (2011, 2015, 2020, 2022, 2023, 2024)]
    assert P.active_folds(folds, "daytrade") == [4, 5]
    assert P.active_folds(folds, "scalp") == list(range(6))


# ---------------------------------------------------------------- keep

def _f(i, alpha, key):
    return {"era": "A" if i < 5 else "B", "test": [f"{2011 + i}-01-01", f"{2012 + i}-01-01"], "alpha": alpha, "trade_key": key}


CONC0 = {"top1pct": 0.02, "top_stock": 0.02, "top10d": 0.02}


def _best(score=0.0, conc=None):
    P._append(P.BEST_HISTORY, {"score": score, "conc": conc or CONC0, "folds": [_f(i, 0.0, f"k{i}") for i in range(10)]})


def test_keep_counts_changed_folds_only():
    _best()
    folds = [_f(i, 0.0, f"k{i}") for i in range(10)]
    folds[3] = _f(3, 0.01, "new3")
    folds[7] = _f(7, 0.01, "new7")
    ok, why = P.beats_c({"void": False, "score": 0.006}, folds, CONC0)
    assert ok, why
    folds[7] = _f(7, -0.01, "new7")                                                   # 바뀐 2개 중 1개만 개선 = 과반 아님
    assert not P.beats_c({"void": False, "score": 0.006}, folds, CONC0)[0]
    same = [_f(i, 0.5, f"k{i}") for i in range(10)]                                  # 체결이 하나도 안 바뀌면 keep 아님
    assert not P.beats_c({"void": False, "score": 0.5}, same, CONC0)[0]
    folds[7] = _f(7, 0.01, "new7")
    assert not P.beats_c({"void": False, "score": 0.004}, folds, CONC0)[0]            # Δscore < 0.005
    assert not P.beats_c({"void": True, "score": 0.5}, folds, CONC0)[0]


def test_keep_refuses_misaligned_folds():
    _best()
    folds = [_f(i, 0.05, f"n{i}") for i in range(10)]
    assert P.beats_c({"void": False, "score": 0.5}, folds[:9], CONC0) == (False, "fold mismatch")
    folds[0] = {**folds[0], "test": ["2016-01-01", "2016-09-01"]}
    assert P.beats_c({"void": False, "score": 0.5}, folds, CONC0) == (False, "fold mismatch")


def test_keep_concentration_vs_best():
    _best()
    folds = [_f(i, 0.01, f"n{i}") for i in range(10)]
    s = {"void": False, "score": 0.01}
    assert not P.beats_c(s, folds, {**CONC0, "top_stock": 0.02 - 0.011})[0]
    assert P.beats_c(s, folds, {**CONC0, "top_stock": 0.02 - 0.009})[0]
    for k in ("top1pct", "top10d"):                                                  # BEST 대비(자기 점수 대비가 아니다)
        assert not P.beats_c(s, folds, {**CONC0, k: 0.0})[0]


def test_best_bar_survives_deleting_worktree_files():
    """BEST.json·작업 트리 BEST_HISTORY 를 지워도 사본으로 비교한다 — 바를 되돌릴 수 없다."""
    _best(score=0.30)
    for f in (P.BEST, P.BEST_HISTORY):
        if os.path.exists(f):
            os.remove(f)
    folds = [_f(i, 0.01, f"n{i}") for i in range(10)]
    ok, why = P.beats_c({"void": False, "score": 0.01}, folds, CONC0)
    assert not ok and why != "첫 유효 실행"


def test_first_valid_only_without_any_keep_row():
    folds = [_f(i, 0.01, f"n{i}") for i in range(10)]
    s = {"void": False, "score": 0.01}
    assert P.beats_c(s, folds, {k: -1 for k in CONC0}) == (True, "첫 유효 실행")
    P._append(P.RUNS, {"train_sha": "b", "beats_best": True, "score": 0.30, "conc": CONC0,   # accept 없이 지나간 keep 도 바다
                       "folds": [_f(i, 0.0, f"k{i}") for i in range(10)]})
    ok, why = P.beats_c(s, folds, CONC0)
    assert not ok and why != "첫 유효 실행"


# ---------------------------------------------------------------- 시행 상한

def _eval(sha):
    P._append(P.RUNS, {"train_sha": sha, "folds": [], "beats_best": False})


def test_trial_cap_and_same_sha(monkeypatch):
    _eval("a")
    _eval("a")
    assert P.trial_state()["n"] == 1
    for i in range(149):
        _eval(f"s{i}")
    st = P.trial_state()
    assert st["n"] == 150 and st["stop"]
    os.remove(P.RUNS)                                                                # 작업 트리 원장을 지워도 사본이 센다
    assert P.trial_state()["stop"]
    monkeypatch.setattr(P, "integrity", lambda *a, **k: pytest.fail("상한 검사가 무결성·계산보다 먼저여야 한다"))
    with pytest.raises(SystemExit, match="상한 도달 — 평가 거부"):
        P.evaluate(1)


def test_crash_counts_and_hides_stdout(tmp_path, monkeypatch, capsys):
    """crash 도 시행으로 센다. 격리 실행의 stdout 은 버리고 stderr 끝 20줄만 보인다(스스로 잰 구간 성적을 흘리는 통로 차단)."""
    tr = tmp_path / "train.py"
    tr.write_text("# x\n")
    monkeypatch.setattr(P, "TRAIN", str(tr))
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    leak = "import sys\nprint('SECRET_FOLD_SCORE')\nfor i in range(50):\n    print(f'err{i}', file=sys.stderr)\nsys.exit(3)\n"
    monkeypatch.setattr(P, "sandbox_cmd_c", lambda args: [sys.executable, "-c", leak])
    with pytest.raises(SystemExit):
        P.evaluate(1)
    out = capsys.readouterr()
    assert "SECRET_FOLD_SCORE" not in out.out + out.err
    assert "err49" in out.out and "err29" not in out.out
    st = P.trial_state()
    assert st["n"] == 1
    with pytest.raises(SystemExit):
        P.evaluate(1)                                                                # 같은 sha 재 crash 는 늘지 않는다
    assert P.trial_state()["n"] == 1


def test_trial_cap_since_first_keep():
    for i in range(40):                                                              # keep 전에는 무 keep 이 쌓여도 멈추지 않는다
        _eval(f"pre{i}")
    assert not P.trial_state()["stop"]
    P._append(P.RUNS, {"train_sha": "k", "folds": [], "beats_best": True})          # keep = beats_best 행(--accept 를 건너뛰어도 센다)
    for i in range(29):
        _eval(f"x{i}")
    _eval("x0")                                                                      # 같은 sha 재실행은 세지 않는다
    st = P.trial_state()
    assert st["since_keep"] == 29 and not st["stop"]
    _eval("x29")
    assert P.trial_state()["stop"]


# ---------------------------------------------------------------- 미래정보

def test_lookahead_injection_voids(tmp_path, monkeypatch):
    Ds = _toy_loop(monkeypatch)
    body = ("def signals(V, x):\n    nxt = pd.DataFrame(V['C'].T).shift(-1).to_numpy().T\n"
            "    return {'cand': V['uni'] & ~(nxt < V['C'] * 0.98), 'score': V['C']}\n")
    peek = _write_train(tmp_path / "peek.py", body)
    out = P.run_all(Ds, 1, peek)
    s = out["summary"]
    assert s["void"] and s["score"] == -1.0 and not s["lookahead_check"].startswith("ok"), s["lookahead_check"]
    assert out["ablation"]["eligible"] is False
    honest = _write_train(tmp_path / "honest.py", body.replace("shift(-1)", "shift(1)"))
    assert P.run_all(Ds, 1, honest)["summary"]["lookahead_check"].startswith("ok")


@pytest.mark.parametrize("key,thr", [("gap", 0.0), ("gap_event", 0.5), ("news_known", 0.5), ("n_feat", 0.5)])
def test_lookahead_sees_news_and_gap_keys(tmp_path, monkeypatch, key, thr):
    """새 V 키(갭·뉴스)도 절단 검사 대상이다 — 내일 값을 읽는 train 이 걸려야 한다."""
    Ds = _toy_loop(monkeypatch)
    peek = _write_train(tmp_path / "pn.py", f"def signals(V, x):\n    nxt = pd.DataFrame(V['{key}'].T.astype(float)).shift(-1).to_numpy().T\n"
                                            f"    return {{'cand': V['uni'] & ~(nxt > {thr}), 'score': V['C']}}\n")
    ok, why = P.causality_c(Ds, {"folds": [{"era": "A", "params": {}}]}, peek, workers=4, n_cut=4)
    assert not ok, why


PROBE = "/tmp/claude-1000/-home-young-git/ece1241f-233e-4b80-ba5e-41e11ecd7401/scratchpad/probe/stash_train.py"
STASH_BODY = ("def signals(V, x):\n    C = V['C']\n    d = np.typecodes\n    box = d.setdefault('zz_stash_t4', [])\n"
              "    if not box or box[0].shape[1] < C.shape[1]:\n        box[:] = [C.copy()]\n"
              "    F = box[0]\n    t = C.shape[1]\n    nxt = pd.DataFrame(F.T).shift(-1).to_numpy().T[:, :t]\n"
              "    return {'cand': V['uni'] & ~(nxt < C * 0.98), 'score': C}\n")


def test_lint_rejects_module_state_stash():
    src = ("import numpy as np\nimport pandas as pd\ndef signals(V, x):\n    if not hasattr(np, 'zz_full'):\n"
           "        np.zz_full = V['C'].copy()\n    return {'cand': V['uni'], 'score': V['C']}\n")
    if os.path.exists(PROBE):
        src = open(PROBE, encoding="utf-8").read()                                    # 리뷰가 확인한 실제 우회 파일
    assert P.lint_train(src)
    for bad in ("import numpy as np\nnp.zz = 1\n", "import numpy as np\nnp.random.zz = 1\n", "import numpy as np\nm = np\n",
                "import numpy as np\nnp.typecodes['zz'] = 1\n", "import numpy as np\ndel np.zz\n", "x = 1\nprint(x)\n",
                "import numpy as np\nf = lambda: np\n"):
        assert P.lint_train(bad), bad
    assert P.lint_train("import numpy as np\nd = {}\ndef f():\n    x = np.zeros(3)\n    d['a'] = np.ones(2) + x\n") == []


def test_causality_isolates_processes_against_stash(tmp_path, monkeypatch):
    """lint 를 피한 상태 은닉(np.typecodes 안에 전체 C 를 숨김)도 — 절단 검사가 새 프로세스에서 돌아서 걸린다."""
    Ds = _toy_loop(monkeypatch)
    stash = _write_train(tmp_path / "stash.py", STASH_BODY)
    assert P.lint_train(open(stash, encoding="utf-8").read()) == []
    try:
        ok, why = P.causality_c(Ds, {"folds": [{"era": "A", "params": {}}]}, stash, workers=4, n_cut=4)
        assert not ok, why
        out = P.run_all(Ds, 8, stash)
        assert out["summary"]["void"] and not out["summary"]["lookahead_check"].startswith("ok")
    finally:
        np.typecodes.pop("zz_stash_t4", None)


def test_causality_one_task_per_process(tmp_path, monkeypatch):
    """작업자 하나여도(재사용되면 전체 신호 작업의 은닉 상태가 절단 작업으로 넘어간다) 걸린다 — 작업 하나 = 프로세스 하나."""
    Ds = _toy_loop(monkeypatch)
    stash = _write_train(tmp_path / "stash1.py", STASH_BODY.replace("zz_stash_t4", "zz_stash_w1"))
    try:
        ok, why = P.causality_c(Ds, {"folds": [{"era": "A", "params": {}}]}, stash, workers=1, n_cut=2)
        assert not ok, why
    finally:
        np.typecodes.pop("zz_stash_w1", None)


def test_concentration_rerun_detects_state(tmp_path, monkeypatch):
    """평가 실행과 같은 V·파라미터의 재실행 신호가 다르면(호출 횟수로 바뀜) StateLeak — 쏠림·비용 1배 숫자를 믿을 수 없다."""
    Ds = _toy_loop(monkeypatch)
    body = ("def signals(V, x):\n    d = np.typecodes\n    box = d.setdefault('zz_calls_c4', [])\n    box.append(1)\n"
            "    late = len(box) > 2\n    return {'cand': V['uni'] & (V['C'] > (1e9 if late else 0)), 'score': V['C']}\n")
    tr = _write_train(tmp_path / "late.py", body)
    try:
        res = P.walk_forward_c(Ds, 1, tr, fixed=[{"era": "A", "params": {}}] * 2)
        assert res["trades"]
        with pytest.raises(P.StateLeak):
            P.concentration(Ds, res, tr)
    finally:
        np.typecodes.pop("zz_calls_c4", None)


CFG_STASH = ("import numpy as np\nimport pandas as pd\n_K = {}\n"
             "def signals(V, x):\n    _K['c'] = V['C']\n    return {'cand': V['uni'] & (V['C'] > 0), 'score': V['C']}\n"
             "def cfg(x):\n    if 'c' in _K:\n        hi = float(np.nanmean(_K['c'][:, -1]) > float(np.nanmean(_K['c'][:, 0])))\n"
             "        return {'hard': 0.10 + 0.20 * hi, 'pyramid': 0}\n    return {'hard': 0.20, 'pyramid': 0}\n"
             "def get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': cfg}\n")


def test_cfg_reading_signal_stash_is_neutralised_and_voided(tmp_path, monkeypatch):
    """signals 가 모듈 dict 에 숨긴 전체 기간 C 로 cfg(손절 폭)를 고르는 train(리뷰 IMPORTANT A) —
    채점 cfg 는 signals 를 부른 적 없는 새 프로세스 값이고, 평가 프로세스 cfg 와 다르니 void."""
    Ds = _toy_loop(monkeypatch)
    (tmp_path / "cs.py").write_text(CFG_STASH)
    honest = tmp_path / "ch.py"
    honest.write_text(CFG_STASH.replace("    if 'c' in _K:", "    if False:"))
    cs = str(tmp_path / "cs.py")
    assert P.lint_train(CFG_STASH) == []
    assert P.fresh_cfgs(cs, [{}]) == [{"hard": 0.20, "pyramid": 0}]
    fixed = [{"era": "A", "params": {}}] * 2
    a = P.walk_forward_c(Ds, 1, cs, fixed=fixed)
    b = P.walk_forward_c(Ds, 1, str(honest), fixed=fixed)
    assert [f["alpha"] for f in a["folds"]] == [f["alpha"] for f in b["folds"]]       # 채점은 숨긴 값과 무관
    assert a["cfg_mismatch"] and not b["cfg_mismatch"]
    s = P.run_all(Ds, 8, cs)["summary"]
    assert s["void"] and s["lookahead_check"].startswith("fail: cfg"), s["lookahead_check"]


def test_fresh_cfg_ignores_process_global_stash(tmp_path):
    """lint 를 피한 프로세스 전역 은닉(np.typecodes)도 — cfg 는 새 프로세스에서 계산해 숨긴 값을 못 본다."""
    src = ("import numpy as np\nimport pandas as pd\n"
           "def signals(V, x):\n    d = np.typecodes\n    d.setdefault('zz_cfg_t4', []).append(1)\n    return {'cand': V['uni'], 'score': V['C']}\n"
           "def cfg(x):\n    return {'hard': 0.30 if np.typecodes.get('zz_cfg_t4') else 0.20}\n"
           "def get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': cfg}\n")
    (tmp_path / "g.py").write_text(src)
    assert P.lint_train(src) == []
    S = P.load_strategy(str(tmp_path / "g.py"))
    try:
        S["signals"](P.view_c(_toy()), {})                                            # 이 프로세스에 은닉
        assert S["cfg"]({})["hard"] == 0.30
        assert P.fresh_cfgs(str(tmp_path / "g.py"), [{}]) == [{"hard": 0.20}]
    finally:
        np.typecodes.pop("zz_cfg_t4", None)


PLATEAU_STEER = ("def signals(V, x):\n    nxt = pd.DataFrame(V['C'].T).shift(-1).to_numpy().T\n"
                 "    cand = V['uni'] & (V['C'] > 0)\n    if x['a'] != 5:\n        cand = cand & ~(nxt < V['C'] * 0.98)\n"
                 "    return {'cand': cand, 'score': V['C']}\n")


def test_causality_checks_training_points(tmp_path, monkeypatch):
    """고른 값(a=5)은 정직하고, 안 고른 학습 점만 미래 값으로 비우는 train(리뷰 IMPORTANT B) — 학습 점 표본 검사가 잡는다."""
    Ds = _toy_loop(monkeypatch)
    def write(name, body):
        (tmp_path / name).write_text("import numpy as np\nimport pandas as pd\n" + body +
                                     "def get_strategy():\n    return {'params': {'a': ('int', 0, 10)}, 'signals': signals, "
                                     "'cfg': lambda x: {'pyramid': 0}}\n")
        return str(tmp_path / name)
    res = {"folds": [{"era": "A", "params": {"a": 5}}] * 2}
    ok, why = P.causality_c(Ds, res, write("steer.py", PLATEAU_STEER), workers=4, n_cut=4)
    assert not ok, why
    ok, why = P.causality_c(Ds, res, write("fine.py", PLATEAU_STEER.replace("shift(-1)", "shift(1)")), workers=4, n_cut=4)
    assert ok, why
    assert {x["a"] for x in P.check_points({"a": ("int", 0, 10)}, 2, 0, seed=1)} - {5}   # 표본에 안 고른 점이 있다


HEAD_A = "import numpy as np\nimport pandas as pd\n"
TAIL_A = "def get_strategy():\n    return {'params': {'a': ('int', 0, 10)}, 'signals': signals, 'cfg': lambda x: {'pyramid': 0}}\n"
RR5 = (HEAD_A + "np.typecodes.setdefault('zz_rr5', []).append(1)\n"
       "def signals(V, x):\n    nxt = pd.DataFrame(V['C'].T).shift(-1).to_numpy().T\n"
       "    cand = V['uni'] & (V['C'] > 0)\n    if x['a'] != 5 and len(np.typecodes['zz_rr5']) >= 2:\n"
       "        cand = cand & ~(nxt > V['C'])\n    return {'cand': cand, 'score': V['C']}\n" + TAIL_A)
# lint 를 피한 같은 우회: 적재 횟수를 get_strategy 안에서 별칭으로 센다(부모·fork 작업자 ≥ 2, 새 프로세스 1)
RR5_ALIAS = (HEAD_A + "def signals(V, x):\n    d = np.typecodes\n    nxt = pd.DataFrame(V['C'].T).shift(-1).to_numpy().T\n"
             "    cand = V['uni'] & (V['C'] > 0)\n    if x['a'] != 5 and len(d.get('zz_rr5b', [])) >= 2:\n"
             "        cand = cand & ~(nxt > V['C'])\n    return {'cand': cand, 'score': V['C']}\n"
             "def get_strategy():\n    d = np.typecodes\n    d.setdefault('zz_rr5b', []).append(1)\n"
             "    return {'params': {'a': ('int', 0, 10)}, 'signals': signals, 'cfg': lambda x: {'pyramid': 0}}\n")
# 모듈 자기 _C 에 호출 수를 세고 첫 호출 뒤에만 속이는 train(부모가 데운 뒤 fork 작업자는 늘 ≥ 2)
COUNTER = (HEAD_A + "_C: dict = {}\n"
           "def signals(V, x):\n    _C['n'] = _C.get('n', 0) + 1\n    nxt = pd.DataFrame(V['C'].T).shift(-1).to_numpy().T\n"
           "    cand = V['uni'] & (V['C'] > 0)\n    if x['a'] != 5 and _C['n'] >= 2:\n"
           "        cand = cand & ~(nxt > V['C'])\n    return {'cand': cand, 'score': V['C']}\n" + TAIL_A)


def _salt(value="test-salt-0"):
    os.makedirs(P.SHADOW, exist_ok=True)
    with open(os.path.join(P.SHADOW, "causality_salt"), "w") as f:
        f.write(value)


def test_lint_rejects_module_top_code_and_module_mutation():
    assert P.lint_train(RR5)                                                         # 리뷰 rr5 탐침: 모듈 최상위 실행
    for bad in ("import numpy as np\nnp.typecodes.setdefault('z', [])\n", "import numpy as np\nX = np.zeros(3)\n",
                "import numpy as np\ndef f():\n    np.typecodes.get('z').append(1)\n",
                "import pandas as pd\ndef f():\n    pd.options.display.__setitem__('a', 1)\n",
                "import numpy as np\ndef f():\n    np.typecodes.update({'z': 1})\n", "for i in range(2):\n    pass\n",
                "import numpy as np\nif True:\n    pass\n"):
        assert P.lint_train(bad), bad
    ok = ('"""설명"""\nfrom __future__ import annotations\nimport numpy as np\n_C: dict = {}\nK = (1, -2.5, "a")\nM = {"a": [1, 2]}\n'
          "def f(a):\n    b = np.sort(a)\n    c = np.append(b, 1)\n    d = {}\n    d.update({'x': 1})\n    return c\n")
    assert P.lint_train(ok) == []


def test_rr5_alias_variant_voided(tmp_path, monkeypatch):
    """lint 를 피한 rr5(적재 횟수로 부모·fork 에서만 속임) — 고원 선택에 쓴 학습 점 신호 ≠ 새 프로세스 신호로 void."""
    Ds = _toy_loop(monkeypatch)
    _salt()
    (tmp_path / "rr5b.py").write_text(RR5_ALIAS)
    assert P.lint_train(RR5_ALIAS) == []
    try:
        s = P.run_all(Ds, 8, str(tmp_path / "rr5b.py"))["summary"]
        assert s["void"] and s["lookahead_check"].startswith("fail"), s["lookahead_check"]
    finally:
        np.typecodes.pop("zz_rr5b", None)


def test_call_counter_cheat_voided(tmp_path, monkeypatch):
    Ds = _toy_loop(monkeypatch)
    _salt()
    (tmp_path / "cnt.py").write_text(COUNTER)
    assert P.lint_train(COUNTER) == []
    s = P.run_all(Ds, 8, str(tmp_path / "cnt.py"))["summary"]
    assert s["void"] and s["lookahead_check"].startswith("fail"), s["lookahead_check"]
    honest = tmp_path / "hon.py"
    honest.write_text(COUNTER.replace("shift(-1)", "shift(1)").replace(" and _C['n'] >= 2", ""))
    assert P.run_all(Ds, 8, str(honest))["summary"]["lookahead_check"].startswith("ok")


def test_sampled_point_hash_must_match_worker(tmp_path, monkeypatch):
    """fork 작업자가 고원 선택에 쓴 신호 지문(point_hash)이 새 프로세스 지문과 다르면 실패 — 절단 검사는 통과하는 경우."""
    Ds = _toy_loop(monkeypatch)
    tr = str(tmp_path / "fine.py")
    (tmp_path / "fine.py").write_text(HEAD_A + PLATEAU_STEER.replace("shift(-1)", "shift(1)") + TAIL_A)
    pts = [x for fi in range(2) for x in P.training_points({"a": ("int", 0, 10)}, fi, 0)[0]]
    res = {"folds": [{"era": "A", "params": {"a": 5}}] * 2}
    assert P.causality_c(Ds, res, tr, workers=4, n_cut=2, check_seed=7)[0]
    bad = {**res, "point_hash": {"A": {P._xkey(x): "worker-saw-something-else" for x in pts}}}
    ok, why = P.causality_c(Ds, bad, tr, workers=4, n_cut=2, check_seed=7)
    assert not ok and "학습 점" in why, why


def test_causality_uses_salted_sample_by_default(tmp_path, monkeypatch):
    """check_seed 없이 부르면 비밀 소금 시드의 표본을 검사한다 — 소금 표본에만 있는 점의 지문이 틀리면 걸린다."""
    Ds = _toy_loop(monkeypatch)
    _salt("salt-for-default")
    tr = str(tmp_path / "fine.py")
    (tmp_path / "fine.py").write_text(HEAD_A + PLATEAU_STEER.replace("shift(-1)", "shift(1)") + TAIL_A)
    spec = {"a": ("int", 0, 10)}
    salted = P.check_points(spec, 2, 0, P.causality_seed(tr))
    other = {P._xkey(x) for x in P.check_points(spec, 2, 0, 0)}
    only = [x for x in salted if P._xkey(x) not in other and x != {"a": 5}]
    assert only
    res = {"folds": [{"era": "A", "params": {"a": 5}}] * 2, "point_hash": {"A": {P._xkey(x): "bogus" for x in only}}}
    ok, why = P.causality_c(Ds, res, tr, workers=4, n_cut=2)
    assert not ok and "학습 점" in why, why


def test_signals_error_only_when_truncated_fails(tmp_path, monkeypatch):
    """절단된 V 에서만 예외를 내는 train(길이·시점을 보고 다르게 군다) — 오류 불일치로 실패."""
    Ds = _toy_loop(monkeypatch)
    body = ("def signals(V, x):\n    if V['C'].shape[1] < 700:\n        raise ValueError('short')\n"
            "    return {'cand': V['uni'], 'score': V['C']}\n")
    ok, why = P.causality_c(Ds, {"folds": [{"era": "A", "params": {}}]}, _write_train(tmp_path / "err.py", body), workers=4, n_cut=2)
    assert not ok and "오류 불일치" in why, why


def test_check_sample_depends_on_secret(tmp_path):
    spec = {"a": ("int", 0, 1000)}
    assert P.check_points(spec, 2, 0, seed=1) != P.check_points(spec, 2, 0, seed=2)
    assert len(P.check_points(spec, 2, 0, seed=1)) == 2 * P.TRAIN_CHECK_PER_FOLD == 24
    tr = tmp_path / "t.py"
    tr.write_text("# a\n")
    _salt("s1")
    a = P.causality_seed(str(tr))
    _salt("s2")
    assert P.causality_seed(str(tr)) != a                                           # 소금이 바뀌면 표본도 바뀐다
    tr.write_text("# b\n")
    assert P.causality_seed(str(tr)) != a


def test_evaluate_sends_secret_seed_on_stdin(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    got = tmp_path / "stdin.txt"
    prog = f"import sys\nopen({str(got)!r}, 'w').write(sys.stdin.read())\nsys.exit(1)\n"
    seen = []
    monkeypatch.setattr(P, "sandbox_cmd_c", lambda args: seen.append(list(args)) or [sys.executable, "-c", prog])
    with pytest.raises(SystemExit):
        P.evaluate(1)
    seed = str(P.causality_seed(P.TRAIN))
    assert got.read_text().strip() == seed
    assert seen and seed not in " ".join(seen[0])                                    # 격리 명령 인자에는 시드가 없다(stdin 만)


# 리뷰 rr6: train 이 돌려준 cand 배열을 자기 _C 에 쥐고 있다가, 평가기가 지문을 뜬 뒤 부르는 cfg(x) 에서 그 배열을 내일 종가로 덮어쓴다
CFGMUT = (HEAD_A + "_C: dict = {}\n"
          "def signals(V, x):\n    C = V['C']\n    nxt = pd.DataFrame(C.T).shift(-1).to_numpy().T\n"
          "    cand = V['uni'] & (C > 0)\n    _C['cand'] = cand\n    _C['cheat'] = cand & (nxt > C)\n"
          "    return {'cand': cand, 'score': C}\n"
          "def cfg(x):\n    if 'cand' in _C:\n        c = _C['cand']\n        c[:] = _C['cheat']\n    return {'pyramid': 0}\n"
          "def get_strategy():\n    return {'params': {'a': ('int', 0, 10)}, 'signals': signals, 'cfg': cfg}\n")
CFGHON = CFGMUT.replace("        c[:] = _C['cheat']\n", "        c = c\n")
# 읽을 때마다 바뀌는 신호 객체: 첫 읽기(지문)는 정직, 그 뒤(시뮬레이션)는 내일 종가
_FLIP_SIG = ("def signals(V, x):\n    C = V['C']\n    nxt = pd.DataFrame(C.T).shift(-1).to_numpy().T\n"
             "    _N['h'] = V['uni'] & (C > 0)\n    _N['c'] = _N['h'] & (nxt > C)\n    _N['k'] = 0\n")
FLIP_OBJ = (HEAD_A + "_N: dict = {}\n"
            "class Flip:\n    def __array__(self, dtype=None, copy=None):\n        _N['k'] = _N['k'] + 1\n"
            "        a = _N['h'] if _N['k'] == 1 else _N['c']\n        return a if dtype is None else a.astype(dtype)\n"
            + _FLIP_SIG + "    return {'cand': Flip(), 'score': C}\n" + TAIL_A)
FLIP_SUB = (HEAD_A + "_N: dict = {}\n"
            "class Sub(np.ndarray):\n    def __array__(self, dtype=None, copy=None):\n        _N['k'] = _N['k'] + 1\n"
            "        a = _N['h'] if _N['k'] == 1 else _N['c']\n        return a if dtype is None else a.astype(dtype)\n"
            + _FLIP_SIG + "    return {'cand': _N['h'].view(Sub), 'score': C}\n" + TAIL_A)


def test_cfg_mutating_returned_cand_cannot_reach_scoring(tmp_path, monkeypatch):
    """리뷰 rr6: 지문 뒤 cfg 가 train 이 쥔 cand 를 제자리에서 고쳐도 평가기 사본은 그대로 — 정직한 train 과 같은 채점."""
    Ds = _toy_loop(monkeypatch)
    _salt()
    out = {}
    for name, src in (("mut", CFGMUT), ("hon", CFGHON)):
        assert P.lint_train(src) == []
        (tmp_path / f"{name}.py").write_text(src)
        out[name] = P.run_all(Ds, 8, str(tmp_path / f"{name}.py"))
    mut, hon = out["mut"], out["hon"]
    assert [f["trade_key"] for f in mut["folds"]] == [f["trade_key"] for f in hon["folds"]]
    assert [f["alpha"] for f in mut["folds"]] == [f["alpha"] for f in hon["folds"]]
    assert [f["params"] for f in mut["folds"]] == [f["params"] for f in hon["folds"]]
    assert mut["conc"] == hon["conc"] or all(np.isnan(v) for v in mut["conc"].values())


def test_own_signals_copies_and_rejects_non_plain_arrays():
    D = _toy()
    shp = D["C"].shape
    cand, score = np.ones(shp, bool), np.ones(shp)
    own = P.own_signals({"cand": cand, "score": score}, shp)
    cand[:] = False
    score[:] = 7.0
    assert own["cand"].all() and (own["score"] == 1.0).all()                          # 사본 — 원본을 고쳐도 그대로
    assert own["cand"].dtype == bool and own["score"].dtype == float

    class DictSub(dict):
        pass
    bad = [DictSub(cand=cand, score=score), [cand, score], {"cand": cand.view(np.matrix) if cand.ndim == 2 else cand, "score": score},
           {"cand": cand.tolist(), "score": score}, {"cand": cand, "score": score.astype(object)}, {"cand": cand[:, :-1], "score": score},
           {"cand": cand, "score": score[0]}, {"cand": cand}]
    for b in bad:
        with pytest.raises(TypeError):
            P.own_signals(b, shp)


@pytest.mark.parametrize("src", [FLIP_OBJ, FLIP_SUB], ids=["array-protocol", "ndarray-subclass"])
def test_signals_reading_differently_on_each_access_rejected(tmp_path, monkeypatch, src):
    """읽을 때마다 바뀌는 신호(__array__ 객체·ndarray 하위 클래스)는 거부 — 학습 점은 오류, 시험 폴드는 crash(시행으로 셈)."""
    Ds = _toy_loop(monkeypatch)
    _salt()
    assert P.lint_train(src) == []
    (tmp_path / "flip.py").write_text(src)
    with pytest.raises(TypeError):
        P.run_all(Ds, 8, str(tmp_path / "flip.py"))


def test_causality_worker_holds_owned_copies(tmp_path):
    """새 프로세스 작업자는 앞 x 의 전체 신호를 쥔 채 다음 x 로 signals 를 부른다 — train 이 앞서 돌려준 배열을 뒤 호출에서 고쳐도
    비교에 쓰는 배열은 그대로(소유 사본)."""
    D = _toy(n=4, t=80, seed=5)
    src = HEAD_A + ("_C: dict = {}\n"
                    "def signals(V, x):\n    for c in _C.get('prev', []):\n        c[:] = False\n"
                    "    cand = V['uni'] & (V['C'] > 0)\n    _C['prev'] = _C.get('prev', []) + [cand]\n"
                    "    return {'cand': cand, 'score': V['C']}\n" + TAIL_A[TAIL_A.index("def get_strategy"):])
    assert P.lint_train(src) == []
    (tmp_path / "hold.py").write_text(src)
    _, _, _, out = P._cause_task(("A", D, -1, "full", [{"a": 1}, {"a": 2}, {"a": 3}], str(tmp_path / "hold.py")))
    assert all(c.all() for c, _, _ in out)
    assert len({h for _, _, h in out}) == 1


def test_signals_get_a_copy_of_params(tmp_path, monkeypatch):
    """signals·cfg 가 받은 x 를 고쳐도 평가기가 기록·재실행에 쓰는 파라미터는 그대로."""
    Ds = _toy_loop(monkeypatch)
    _salt()
    src = HEAD_A + ("def signals(V, x):\n    x['a'] = 99\n    return {'cand': V['uni'] & (V['C'] > 0), 'score': V['C']}\n"
                    "def cfg(x):\n    x['a'] = 77\n    return {'pyramid': 0}\n"
                    "def get_strategy():\n    return {'params': {'a': ('int', 0, 10)}, 'signals': signals, 'cfg': cfg}\n")
    (tmp_path / "px.py").write_text(src)
    out = P.run_all(Ds, 8, str(tmp_path / "px.py"))
    assert all(0 <= f["params"]["a"] <= 10 for f in out["folds"])


def _keep_row(score):
    return {"train_sha": f"s{score}", "beats_best": True, "score": score, "conc": CONC0, "folds": [_f(i, 0.0, f"k{i}") for i in range(10)]}


def test_best_is_latest_keep_in_either_order():
    """BEST = 마지막 keep. accept 뒤에 accept 안 된 beats_best 행이 오면 그 행, 그 반대면 BEST_HISTORY(리뷰 IMPORTANT C)."""
    P._append(P.RUNS, _keep_row(0.30))
    P._append(P.RUNS, {"train_sha": "s0.3", "accept": True})
    P._append(P.BEST_HISTORY, {"score": 0.30, "conc": CONC0, "folds": [_f(i, 0.0, f"k{i}") for i in range(10)]})
    assert P.best_record()["score"] == 0.30
    P._append(P.RUNS, _keep_row(0.50))                                                # accept 안 된 더 늦은 keep
    assert P.best_record()["score"] == 0.50
    folds = [_f(i, 0.01, f"n{i}") for i in range(10)]
    assert not P.beats_c({"void": False, "score": 0.40}, folds, CONC0)[0]             # 0.30 바였다면 keep 이었을 점수
    P._append(P.RUNS, {"train_sha": "s0.5", "accept": True})
    P._append(P.BEST_HISTORY, {"score": 0.50, "conc": CONC0, "folds": [_f(i, 0.0, f"k{i}") for i in range(10)], "tag": "h"})
    assert P.best_record().get("tag") == "h"


def test_state_dependent_signals_void(tmp_path, monkeypatch):
    """새 프로세스 첫 호출과 평가 프로세스의 신호가 다르면 void — 호출 횟수·숨긴 상태로 바꾸는 train."""
    Ds = _toy_loop(monkeypatch)
    body = ("def signals(V, x):\n    d = np.typecodes\n    box = d.setdefault('zz_calls_t4', [])\n    box.append(1)\n"
            "    good = len(box) > 1\n    return {'cand': V['uni'] & (V['C'] > (0 if good else 1e9)), 'score': V['C']}\n")
    tr = _write_train(tmp_path / "calls.py", body)
    assert P.lint_train(open(tr, encoding="utf-8").read()) == []
    try:
        s = P.run_all(Ds, 8, tr)["summary"]
        assert s["void"] and s["lookahead_check"].startswith("fail"), s["lookahead_check"]
    finally:
        np.typecodes.pop("zz_calls_t4", None)


# ---------------------------------------------------------------- 무결성·격리

def _git(root, *a):
    return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", "-c", "user.email=t@example.com", "-c", "user.name=t", *a],
                          check=True, capture_output=True, text=True)


def test_integrity_refuses_changed_guarded_file(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel in P.GUARDED_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    man = root / "research/combined/MANIFEST.json"
    man.write_text(json.dumps({rel: P.sha_file(str(root / rel)) for rel in P.GUARDED_FILES}))
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "prereg")
    _git(root, "tag", P.PREREG_TAG)
    monkeypatch.setattr(P, "ROOT", str(root))
    monkeypatch.setattr(P, "MANIFEST", str(man))
    P.integrity(check_cache=False)                                                   # 손대지 않으면 통과
    gaps = root / "research/combined/gaps.py"
    gaps.write_text("# 바뀜\n")
    with pytest.raises(SystemExit, match="평가 거부"):
        P.integrity(check_cache=False)                                               # 작업 트리 변경
    _git(root, "commit", "-qam", "tweak gaps")
    with pytest.raises(SystemExit, match="평가 거부"):
        P.integrity(check_cache=False)                                               # 태그 뒤 커밋
    man.write_text(json.dumps({rel: P.sha_file(str(root / rel)) for rel in P.GUARDED_FILES}))
    _git(root, "commit", "-qam", "manifest too")                                     # 매니페스트까지 맞춰도 태그 검사가 막는다
    with pytest.raises(SystemExit, match="사전등록"):
        P.integrity(check_cache=False)


def test_guarded_files_cover_evaluator_engine_and_data():
    need = {f"research/combined/{m}.py" for m in ("prepare", "data", "gaps", "news", "select")} | {
        "research/autoresearch/prepare.py", "research/experiments/leader21.py", "research/experiments/indep30.py"} | set(V4.GUARDED_FILES)
    assert need <= set(P.GUARDED_FILES)
    assert P.PREREG_TAG == "combined-prereg-1"


def _pairs(cmd, flag):
    return [(cmd[i + 1], cmd[i + 2] if flag == "--ro-bind" else None) for i, x in enumerate(cmd) if x == flag]


def test_confirm_cache_hidden_in_sandbox():
    cmd = P.sandbox_cmd_c(["--inner"])
    binds = [a for a, _ in _pairs(cmd, "--ro-bind")]
    c_path = os.path.join(P.CACHE, "era_C_s0.pkl")
    assert c_path not in binds and all(os.path.join(P.CACHE, f"era_{e}_s0.pkl") in binds for e in ("A", "B"))
    tmpfs = [a for a, _ in _pairs(cmd, "--tmpfs")]
    assert P.CACHE in tmpfs and cmd.index(P.CACHE) < cmd.index(binds[-1])            # 캐시 디렉터리를 가린 뒤 A·B 만 다시 보인다
    home = os.path.expanduser("~")
    for h in (P.SHADOW, f"{home}/git/daytrade-it", f"{home}/git/quant-airflow", f"{home}/.config", f"{home}/.ssh", V4.SHADOW,
              f"{home}/.cache/cc-swing/scratch/autoresearch", f"{home}/.cache/cc-swing/autoresearch_archive"):
        assert h in P.sandbox_hide()
        if os.path.isdir(h):
            assert h in tmpfs
    assert "--unshare-net" in cmd and cmd[cmd.index("--unsetenv") + 1] == "KR_QUANT_DB"


@pytest.mark.skipif(shutil.which("bwrap") is None or not os.path.exists(os.path.join(P.CACHE, "era_C_s0.pkl")),
                    reason="bwrap·확인 캐시 없음")
def test_confirm_cache_unreadable_inside_real_sandbox():
    cmd = P.sandbox_cmd_c(["--inner"])
    py = cmd.index(os.path.join(P.ROOT, ".venv", "bin", "python"))
    probe = cmd[:py] + ["/usr/bin/test", "-r"]
    assert subprocess.run(probe + [os.path.join(P.CACHE, "era_C_s0.pkl")]).returncode != 0
    assert subprocess.run(probe + [os.path.join(P.CACHE, "era_A_s0.pkl")]).returncode == 0


# ---------------------------------------------------------------- 쏠림·기준선

def test_top10d_zeroes_pooled_best_days():
    daily = [np.r_[np.full(5, 0.10), np.zeros(5)], np.r_[np.full(8, 0.05), np.full(2, -0.01)]]
    s = P.top10d_score(daily, [1.0, 1.0])
    # 상위 10일 = 앞 폴드 0.10 × 5 + 뒤 폴드 0.05 × 5 → 남은 알파 0.0, 0.05×3 − 0.02
    assert s == pytest.approx(P.score_of([0.0, 0.13]))


def test_baseline_train_passes_lint_and_uses_no_repo_keys():
    src = open(P.TRAIN, encoding="utf-8").read()
    assert P.lint_train(src) == []
    assert not any(f"'{k}'" in src or f'"{k}"' in src for keys in P.REPO_KEYS.values() for k in keys)
    S = P.load_strategy(P.TRAIN)
    assert set(S["params"]) == {"theta", "N", "ma_exit", "hard", "exit_kind", "low_w"}
    assert "entry" not in S["cfg"]({"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "exit_kind": "ma", "low_w": 50})


# ---------------------------------------------------------------- 참고 실행(--reference)

def test_reference_once_not_a_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    fake = json.dumps({"summary": {"score": 0.0}, "folds": []})
    monkeypatch.setattr(P, "sandbox_cmd_c", lambda args: [sys.executable, "-c", f"print('@@RESULT@@' + {fake!r})"])
    P.reference(1)
    assert os.path.exists(P.REFERENCE) and os.path.exists(os.path.join(P.SHADOW, "REFERENCE.json"))
    assert P.trial_state()["n"] == 0 and not os.path.exists(P.RUNS)                # 시행으로 세지 않는다
    os.remove(P.REFERENCE)
    with pytest.raises(SystemExit, match="이미"):
        P.reference(1)                                                               # 사본이 있으면 다시 못 돈다


def test_reference_refuses_changed_train(tmp_path, monkeypatch):
    tr = tmp_path / "train.py"
    tr.write_text(open(P.TRAIN, encoding="utf-8").read() + "\n# 바뀜\n")
    monkeypatch.setattr(P, "TRAIN", str(tr))
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="기준선"):
        P.reference(1)


def test_reference_params_are_leader_c():
    from research.paper.leader_paper import ARMS
    c = ARMS["leader_C"]
    r = P.REF_PARAMS
    assert (r["theta"], r["N"], r["ma_exit"], r["hard"], r["exit_kind"]) == (c["theta"], c["N"], c["ma_exit"], c["hard"], "ma")
    S = P.load_strategy(P.TRAIN)
    cfg = P.clamp_cfg(S["cfg"](r))
    assert (cfg["max_pos"], cfg["pyramid"], cfg["regime"], cfg["exit_w"]) == (10, c["pyramid"], c["regime"], c["ma_exit"])
    assert P.BASELINE_TRAIN_SHA == P.hashlib.sha256(open(P.TRAIN, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- select.py 용 인터페이스

def test_git_state_records_commit_and_dirty(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "research/combined").mkdir(parents=True)
    tr = root / "research/combined/train.py"
    tr.write_text("# a\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "a")
    monkeypatch.setattr(P, "ROOT", str(root))
    monkeypatch.setattr(P, "TRAIN", str(tr))
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    assert P.git_state() == {"commit": head[:12], "dirty": False}
    tr.write_text("# b\n")
    assert P.git_state() == {"commit": head[:12], "dirty": True}


def test_crash_row_carries_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    monkeypatch.setattr(P, "sandbox_cmd_c", lambda args: [sys.executable, "-c", "raise SystemExit(2)"])
    with pytest.raises(SystemExit):
        P.evaluate(1)
    row = P._read_jsonl(P.RUNS)[-1]
    assert row["crash"] and {"commit", "dirty"} <= set(row)


def test_eras_keyword_runs_other_era_without_global_swap(tmp_path):
    D = _toy(n=6, t=780, start="2022-01-03", seed=3)
    eras = {"Z": {"folds": TOY_FOLDS}}
    used = _write_train(tmp_path / "used.py", "def signals(V, x):\n    return {'cand': np.nan_to_num(V['gap']) > 0.005, 'score': V['C']}\n")
    fixed = [{"era": "Z", "params": {}}] * 2
    res = P.walk_forward_c({"Z": D}, 1, used, fixed=fixed, eras=eras)
    assert [f["era"] for f in res["folds"]] == ["Z", "Z"] and set(P.ERAS) == {"A", "B"}
    assert set(P.concentration({"Z": D}, res, used, eras=eras)) == set(P.CONC_KEYS)
    assert P.ablation({"Z": D}, res, used, eras=eras)["scalp"]["changed"] >= 0.05
    ok, why = P.causality_c({"Z": D}, res, used, workers=4, n_cut=4, eras=eras)
    assert ok, why


# ---------------------------------------------------------------- 정정 0: 잠금 캐시 격리·MANIFEST, 시간 초과, refs, 모양 고정

def test_locked_cache_hidden_in_sandbox():
    from research.combined.data import FROZEN, LOCKED_CACHE
    assert LOCKED_CACHE in P.sandbox_hide() and not LOCKED_CACHE.startswith(P.CACHE + os.sep)
    assert FROZEN in P.sandbox_hide() and not FROZEN.startswith(P.CACHE + os.sep)    # 얼린 제외 목록 = 미래 폐지 표지
    cmd = P.sandbox_cmd_c(["--inner"])
    binds = [a for a, _ in _pairs(cmd, "--ro-bind")]
    assert not any(b.startswith(LOCKED_CACHE) for b in binds)
    if os.path.isdir(LOCKED_CACHE):
        assert LOCKED_CACHE in [a for a, _ in _pairs(cmd, "--tmpfs")]


def test_manifest_and_integrity_cover_locked_cache(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel in P.GUARDED_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    cache, locked = tmp_path / "combined", tmp_path / "combined_locked"
    cache.mkdir()
    locked.mkdir()
    for e in P.CACHE_ERAS:
        (cache / f"era_{e}_s0.pkl").write_bytes(f"s0{e}".encode())
        (locked / f"era_{e}_s1.pkl").write_bytes(f"s1{e}".encode())
    man = root / "research/combined/MANIFEST.json"
    monkeypatch.setattr(P, "ROOT", str(root))
    monkeypatch.setattr(P, "MANIFEST", str(man))
    monkeypatch.setattr(P, "CACHE", str(cache))
    monkeypatch.setattr(P, "LOCKED_CACHE", str(locked))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for f in P.FROZEN_FILES:
        (frozen / f).write_text("{}")
    monkeypatch.setattr(P, "FROZEN", str(frozen))
    monkeypatch.setattr(V4, "EXT_GUARDED", ())
    P.write_manifest()
    m = json.load(open(man))
    assert {f"locked/era_{e}_s1.pkl" for e in P.CACHE_ERAS} <= set(m) and {f"cache/era_{e}_s0.pkl" for e in P.CACHE_ERAS} <= set(m)
    assert {f"frozen/{f}" for f in P.FROZEN_FILES} <= set(m)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "prereg")
    _git(root, "tag", P.PREREG_TAG)
    P.integrity(check_locked=True)
    (locked / "era_B_s1.pkl").write_bytes(b"tampered")
    P.integrity()                                                                    # 루프 평가는 잠금 캐시를 해시하지 않는다
    with pytest.raises(SystemExit, match="잠금 캐시"):
        P.integrity(check_locked=True)
    (frozen / P.FROZEN_FILES[0]).write_text('{"000001": "ok"}')                      # 얼린 뉴스 분류를 바꾸면 루프 평가도 거부
    with pytest.raises(SystemExit, match="얼린"):
        P.integrity()


def test_sandbox_timeout_is_a_crash_trial(tmp_path, monkeypatch, capsys):
    tr = tmp_path / "train.py"
    tr.write_text("# slow\n")
    monkeypatch.setattr(P, "TRAIN", str(tr))
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    monkeypatch.setattr(P, "SANDBOX_TIMEOUT_S", 1)
    monkeypatch.setattr(P, "sandbox_cmd_c", lambda args: [sys.executable, "-c", "import time, sys; print('SECRET'); sys.stdout.flush(); time.sleep(30)"])
    with pytest.raises(SystemExit):
        P.evaluate(1)
    row = P._read_jsonl(P.RUNS)[-1]
    assert row["crash"] is True and row["returncode"] == "timeout"
    out = capsys.readouterr()
    assert "SECRET" not in out.out + out.err and "시간 초과" in out.out
    assert P.trial_state()["n"] == 1


def test_pin_ref_keeps_discarded_commit_reachable(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "research/combined").mkdir(parents=True)
    tr = root / "research/combined/train.py"
    tr.write_text("# base\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    tr.write_text("# exp\n")
    _git(root, "commit", "-qam", "exp")
    monkeypatch.setattr(P, "ROOT", str(root))
    monkeypatch.setattr(P, "TRAIN", str(tr))
    gs = P.git_state()
    ref = _REAL_PIN("a" * 16, gs)
    assert ref == f"refs/combined/{'a' * 16}/{gs['commit']}"
    _git(root, "reset", "-q", "--hard", "HEAD~1")                                    # 루프의 discard
    _git(root, "reflog", "expire", "--expire=now", "--all")
    _git(root, "gc", "-q", "--prune=now")
    assert _git(root, "show", f"{gs['commit']}:research/combined/train.py").stdout == "# exp\n"
    tr.write_text("# dirty\n")
    assert _REAL_PIN("b" * 16, P.git_state()) is None                               # dirty 는 붙잡지 않는다
    assert _REAL_PIN("c" * 16, {"commit": None, "dirty": None}) is None


def test_signals_cannot_swap_V_C_to_pass_shape_check():
    """기대 모양은 train 코드가 돌기 전에 정한다 — signals 가 V['C'] 를 작은 배열로 바꿔 끼우고 그 모양으로 돌려줘도 거부(Task 4 미뤄 둔 사소)."""
    D = _toy()
    V = P.neutral_view(D)

    def sneaky(V, x):
        V["C"] = V["C"][:1, :5]
        return {"cand": np.ones((1, 5), bool), "score": np.ones((1, 5))}
    with pytest.raises(TypeError, match="ndarray"):
        P.call_signals({"signals": sneaky}, V, {})
    V = P.neutral_view(D)
    with pytest.raises(TypeError):
        P.call_signals({"signals": sneaky}, V, {}, D["C"].shape)
    ok = P.call_signals({"signals": lambda V, x: {"cand": np.ones(V["C"].shape, bool), "score": V["C"]}}, P.neutral_view(D), {})
    assert ok["cand"].shape == D["C"].shape


def test_ablation_requires_removed_run_to_keep_trading(tmp_path, monkeypatch):
    """정정 0 M1: 중립 입력에서 거래를 멈추는 뉴스 문(cand & news_known)은 changed·dscore 가 공짜로 서도 자격이 아니다."""
    D = _toy(n=6, t=780, start="2022-01-03", seed=None)                              # 고르게 오르는 종목 → 거래하면 알파 > 0
    D["kx"] = np.full(780, 1000.0)
    D["news_known"][:] = True
    monkeypatch.setattr(P, "ERAS", {"A": {"folds": TOY_FOLDS}})
    Ds = {"A": D}
    fixed = [{"era": "A", "params": {}}] * 2
    gate = _write_train(tmp_path / "gate.py", "def signals(V, x):\n    return {'cand': (V['C'] > 0) & V['news_known'], 'score': V['C']}\n")
    res = P.walk_forward_c(Ds, 1, gate, fixed=fixed)
    ab = P.ablation(Ds, res, gate)["daytrade"]
    assert ab["changed"] >= P.ABL_MIN_CHANGED and ab["dscore"] > 0                     # 옛 규칙이면 ok 였다(위반 주입이 실제로 선다)
    assert ab["n_removed"] == 0 and ab["trades_ok"] is False and ab["ok"] is False
    # 통과(pass-through) 부품: 중립 입력에서 아무것도 막지 않는다 → 제거 실행이 여전히 거래한다
    thru = _write_train(tmp_path / "thru.py", "def signals(V, x):\n    bad = V['news_known'] & (V['n_feat'] > 1)\n"
                                             "    return {'cand': (V['C'] > 0) & ~bad, 'score': V['C']}\n")
    ab2 = P.ablation(Ds, P.walk_forward_c(Ds, 1, thru, fixed=fixed), thru)["daytrade"]
    assert ab2["trades_ok"] is True and ab2["n_removed"] >= ab2["n_orig"] * P.ABL_MIN_KEEP_TRADES


def test_evaluate_pins_commit_even_on_crash(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(P, "pin_ref", lambda sha, gs: seen.append((sha, set(gs))))
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    monkeypatch.setattr(P, "sandbox_cmd_c", lambda args: [sys.executable, "-c", "raise SystemExit(2)"])
    with pytest.raises(SystemExit):
        P.evaluate(1)
    cur = P.hashlib.sha256(open(P.TRAIN, "rb").read()).hexdigest()[:16]
    assert seen == [(cur, {"commit", "dirty"})]


# ---------------------------------------------------------------- 정정 0-2 K1: 가격 배수(미래 수정 사건 표지 제거)

def test_price_mult_salted_deterministic_in_range(tmp_path):
    from research.combined import data as cd
    salt = cd.price_salt(str(tmp_path / "led" / "price_salt"))
    assert oct(os.stat(tmp_path / "led" / "price_salt").st_mode & 0o777) == "0o600"
    assert cd.price_salt(str(tmp_path / "led" / "price_salt")) == salt                 # 한 번만 만든다
    codes = [f"{i:06d}" for i in range(500)]
    m = cd.price_mult(codes, salt)
    assert ((m >= 1) & (m < 2)).all() and len(set(m)) == 500 and m.std() > 0.2
    assert np.array_equal(cd.price_mult(codes[::-1], salt), m[::-1])                  # 종목마다 같은 배수(시대·절반 무관)
    assert not np.allclose(cd.price_mult(codes, salt + "x"), m)                       # 소금 없이는 모른다


def test_view_scales_prices_only_and_baseline_trades_invariant(tmp_path, monkeypatch):
    from research.combined import data as cd
    D = _toy(n=6, t=780, start="2022-01-03", seed=3)
    V0 = cd.view_c(D)
    D2 = {**D, "vmult": cd.price_mult(D["codes"], "s")}
    V = cd.view_c(D2)
    for k in ("O", "H", "L", "C"):
        assert np.allclose(V[k], V0[k] * D2["vmult"][:, None]) and not np.allclose(V[k], V0[k])
    assert np.allclose(V["C"][:, 1:] / V["C"][:, :-1], V0["C"][:, 1:] / V0["C"][:, :-1])   # 수익률 그대로
    for k in set(V) - {"O", "H", "L", "C"}:
        assert np.array_equal(V[k], V0[k], equal_nan=True)
    monkeypatch.setattr(P, "ERAS", {"A": {"folds": TOY_FOLDS}})
    fixed = [{"era": "A", "params": dict(P.REF_PARAMS)}] * 2
    r0 = P.walk_forward_c({"A": D}, 1, P.TRAIN, fixed=fixed)
    r1 = P.walk_forward_c({"A": D2}, 1, P.TRAIN, fixed=fixed)
    assert [f["trade_key"] for f in r0["folds"]] == [f["trade_key"] for f in r1["folds"]] and sum(f["n"] for f in r0["folds"]) > 0


def test_price_frac_leak_check_fires_without_multiplier():
    from research.combined import data as cd
    rng = np.random.default_rng(0)
    n, t = 40, 200
    C = np.round(rng.uniform(1_000, 50_000, (n, t)) / 50) * 50                       # 수정 사건 없음: 정수(호가 격자)
    C[n // 2:] = C[n // 2:] * 0.8731                                                  # 뒤에 분할·증자: 소급 수정 → 비정수
    uni = np.ones((n, t), bool)
    with pytest.raises(ValueError, match="소수부"):
        cd.check_price_frac_leak(C, np.ones(n), uni)                                  # 위반 주입: 배수 없음
    out = cd.check_price_frac_leak(C, cd.price_mult([f"{i:06d}" for i in range(n)], "salt"), uni)
    assert out["ks"] < cd.FRAC_LEAK_TOL and 0.45 < out["nonint_share_uni"] < 0.55


def test_cache_without_multiplier_refused(tmp_path, monkeypatch):
    from research.combined import data as cd
    monkeypatch.setattr(cd, "CACHE", str(tmp_path))
    import pickle
    with open(tmp_path / "era_Z_s0.pkl", "wb") as f:
        pickle.dump({"codes": ["000001"], "C": np.ones((1, 3))}, f)
    with pytest.raises(SystemExit, match="vmult"):
        cd.load("Z")
    with pytest.raises(SystemExit, match="vmult"):
        P._cause_task(("Z", str(tmp_path / "era_Z_s0.pkl"), -1, "full", [], P.TRAIN))   # 미래정보 검사 작업자도


# ---------------------------------------------------------------- 정정 0-2: 섬(island) 모형

def _row(sha, island, beats=False, score=0.0, **kw):
    return {"train_sha": sha, "island": island, "folds": [_f(i, score, f"{sha}{i}") for i in range(10)], "beats_best": beats,
            "score": score, "conc": CONC0, **kw}


def test_island_from_branch_and_refusal(monkeypatch):
    for b in ("combined-loop-1", "combined-loop-4"):
        monkeypatch.setattr(P, "current_branch", lambda b=b: b)
        assert P.island_of("f" * 16) == b
    for b in ("combined-loop-5", "feature-combined-strategy", "combined-loop-1x", ""):
        monkeypatch.setattr(P, "current_branch", lambda b=b: b)
        with pytest.raises(SystemExit, match="섬이 아니다"):
            P.island_of("f" * 16)
        assert P.island_of(P.BASELINE_TRAIN_SHA) == "main"                           # 기준선(스모크·참고)만 main
    monkeypatch.setattr(P, "current_branch", lambda: "feature-combined-strategy")
    tr = P.TRAIN
    monkeypatch.setattr(P, "TRAIN", tr)
    monkeypatch.setattr(P, "integrity", lambda *a, **k: pytest.fail("거부는 무결성·계산보다 앞"))
    monkeypatch.setattr(P, "BASELINE_TRAIN_SHA", "0" * 16)
    with pytest.raises(SystemExit, match="섬이 아니다"):
        P.evaluate(1)
    assert not os.path.exists(P.RUNS)


def test_default_workers_split_four_ways():
    n = os.cpu_count() or 2
    assert P.default_workers("combined-loop-2") == max(1, (n - 1) // 4)
    assert P.default_workers("feature-combined-strategy") == max(1, n - 1)


def test_per_island_bar_starts_from_baseline():
    P._append(P.RUNS, _row("base", "main", beats=True, score=0.10))
    P._append(P.RUNS, _row("i1a", "combined-loop-1", beats=True, score=0.30))
    assert P.best_record("combined-loop-1")["train_sha"] == "i1a"
    assert P.best_record("combined-loop-2")["train_sha"] == "base"                    # 다른 섬의 keep 은 이 섬의 바가 아니다
    s = {"void": False, "score": 0.20}
    conc = dict(CONC0)
    f2 = [_f(i, 0.5, f"new{i}") for i in range(10)]
    assert P.beats_c(s, f2, conc, "combined-loop-2")[0] is True                       # 기준선 0.10 대비 keep
    assert P.beats_c(s, f2, conc, "combined-loop-1")[0] is False                      # 섬 1 의 0.30 대비 아님


def test_accept_per_island(monkeypatch):
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    cur = P.hashlib.sha256(open(P.TRAIN, "rb").read()).hexdigest()[:16]
    P._append(P.RUNS, _row(cur, "combined-loop-1", beats=True, score=0.2, evaluator="t", eligible=False))
    P._append(P.RUNS, _row("other", "combined-loop-2", beats=False, score=0.1))      # 다른 섬 행이 뒤에 와도 막지 않는다
    P.accept()
    assert os.path.exists(P.best_path("combined-loop-1")) and not os.path.exists(P.BEST)
    assert P._read_jsonl(P.BEST_HISTORY)[-1]["island"] == "combined-loop-1"
    monkeypatch.setattr(P, "current_branch", lambda: "combined-loop-2")
    with pytest.raises(SystemExit, match="combined-loop-2"):
        P.accept()


def test_global_cap_across_islands_and_recheck_before_append(monkeypatch):
    for i in range(149):
        P._append(P.RUNS, {"train_sha": f"s{i}", "island": f"combined-loop-{1 + i % 4}", "folds": [], "beats_best": False})
    st = P.trial_state()
    assert st["n"] == 149 and not st["stop"]                                          # 섬을 합쳐 센다
    monkeypatch.setattr(P, "integrity", lambda *a, **k: None)
    cur = P.hashlib.sha256(open(P.TRAIN, "rb").read()).hexdigest()[:16]

    def other_island_fills_cap(args, c, stdin="", island="main"):
        P._append(P.RUNS, {"train_sha": "late", "island": "combined-loop-3", "folds": [], "beats_best": False})
        return {"summary": {"train_sha": c}, "folds": [], "conc": {}, "ablation": {}}
    monkeypatch.setattr(P, "_sandboxed", other_island_fills_cap)
    with pytest.raises(SystemExit, match="상한"):
        P.evaluate(1)
    assert cur not in {r["train_sha"] for r in P._read_jsonl(P.RUNS)} and P.trial_state()["n"] == 150   # 151 번째가 끼지 못했다


_APPENDER = """
import sys
sys.path[:0] = [{root!r}, {src!r}]
from research.combined import prepare as P
P.SHADOW = {shadow!r}
for i in range({n}):
    P._append({runs!r}, {{"who": {who!r}, "i": i, "pad": "x" * 2000}})
"""


def test_flock_concurrent_appends_from_two_processes(tmp_path):
    runs, shadow = str(tmp_path / "RUNS.jsonl"), str(tmp_path / "shadow")
    procs = [subprocess.Popen([sys.executable, "-c", _APPENDER.format(root=P.ROOT, src=os.path.join(P.ROOT, "src"), shadow=shadow, n=300,
                                                                       runs=runs, who=w)]) for w in ("a", "b")]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    for path in (runs, os.path.join(shadow, "RUNS.jsonl")):
        rows = [json.loads(x) for x in open(path, encoding="utf-8")]                 # 줄이 섞이면 json 이 깨진다
        assert len(rows) == 600 and {(r["who"], r["i"]) for r in rows} == {(w, i) for w in "ab" for i in range(300)}
    a = [json.loads(x)["who"] for x in open(runs, encoding="utf-8")]
    b = [json.loads(x)["who"] for x in open(os.path.join(shadow, "RUNS.jsonl"), encoding="utf-8")]
    assert a == b                                                                    # 두 사본의 순서가 같다(자물쇠가 두 쓰기를 묶는다)


_HOLDER = """
import sys, time
sys.path[:0] = [{root!r}, {src!r}]
from research.combined import prepare as P
P.SHADOW = {shadow!r}
with P.ledger_lock():
    open({flag!r}, "w").write("held")
    time.sleep(2.0)
"""


def test_append_waits_for_ledger_lock_held_by_other_process(tmp_path, monkeypatch):
    """다른 섬 프로세스가 자물쇠를 쥐고 있으면 덧붙이기는 기다린다(자물쇠를 빼면 바로 써서 이 시험이 깨진다)."""
    import time
    shadow, flag = str(tmp_path / "shadow"), str(tmp_path / "flag")
    monkeypatch.setattr(P, "SHADOW", shadow)
    h = subprocess.Popen([sys.executable, "-c", _HOLDER.format(root=P.ROOT, src=os.path.join(P.ROOT, "src"), shadow=shadow, flag=flag)])
    t0 = time.time()
    while not os.path.exists(flag):
        assert time.time() - t0 < 60 and h.poll() is None
        time.sleep(0.02)
    t1 = time.time()
    P._append(str(tmp_path / "RUNS.jsonl"), {"x": 1})
    waited = time.time() - t1
    assert h.wait(timeout=30) == 0 and waited > 1.0


def test_cut_view_keeps_price_multiplier():
    """절단·섞기 V 도 같은 가격 배수를 써야 한다 — 빠지면 절단 V 만 배수 없는 가격이 되어 부동소수 같음 비교가 뒤집혀 정직한 train 이 void 된다
    (2026-10-25 태그 전 스모크에서 실제로 era A cut 731 cand 1칸 불일치)."""
    from research.combined import data as cd
    D = _toy(n=4, t=60, seed=5)
    D["vmult"] = cd.price_mult(D["codes"], "s")
    full = cd.view_c(D)
    for mode in (None, 7):
        V = cd.view_c(P._cut(D, 30, shuffle_seed=mode))
        for k in ("O", "H", "L", "C"):
            assert np.array_equal(V[k][:, :31], full[k][:, :31])
