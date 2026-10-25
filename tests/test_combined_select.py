"""사후 후보 선택기(select.py) 가드 — 위반을 주입해 실제로 막히는지 본다(GUARDRAILS §4.5).

DB·캐시 없이: walk_forward_c·적재·쏠림·팩터·DSR 은 가짜로 바꾸고, 원장·사본은 임시 경로로 돌린다.
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd
import pytest

from research.combined import prepare as P
from research.combined import select as S


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    """원장·사본·산출물은 임시 경로로. 적재는 가짜(잠금 절반은 호출 횟수를 센다), 무결성 검사는 끈다."""
    monkeypatch.setattr(P, "RUNS", str(tmp_path / "RUNS.jsonl"))
    monkeypatch.setattr(P, "SHADOW", str(tmp_path / "shadow"))
    monkeypatch.setattr(S, "CONFIRM", str(tmp_path / "CONFIRM.jsonl"))
    monkeypatch.setattr(S, "LOCKBOX", str(tmp_path / "LOCKBOX.jsonl"))
    monkeypatch.setattr(S, "SCREEN", str(tmp_path / "SCREEN.jsonl"))
    monkeypatch.setattr(S, "SELECTION", str(tmp_path / "SELECTION.json"))
    monkeypatch.setattr(S, "SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setattr(P, "integrity", lambda check_cache=True: None)
    monkeypatch.setattr(S, "load_search", lambda era: {"era": era, "half": 0})
    calls = {"locked": []}
    monkeypatch.setattr(S, "load_locked", lambda era: calls["locked"].append(era) or {"era": era, "half": 1})
    yield calls


# ---------------------------------------------------------------- 가짜 계산

def _fake_wf(alpha_of, n=30, ret=0.02):
    """walk_forward_c 대역 — 지금 P.ERAS 의 폴드마다 alpha_of(train_sha, era, half, null) 로 폴드 alpha 를 낸다."""
    calls = []

    def wf(D_by_era, workers, train_path=P.TRAIN, fixed=None, null=None, neutral=None, cost_mult=P.COST_MULT, exclude=(), mask=None):
        sha = hashlib.sha256(open(train_path, "rb").read()).hexdigest()[:16]
        calls.append({"sha": sha, "eras": tuple(P.ERAS), "null": null, "fixed": fixed is not None,
                      "half": {e: D_by_era[e]["half"] for e in P.ERAS}})
        res = {"folds": [], "trades": [], "months": [], "daily": []}
        for era, E in P.ERAS.items():
            for lo_tr, lo, hi in E["folds"]:
                a = alpha_of(sha, era, D_by_era[era]["half"], null)
                gi = len(res["folds"])
                res["folds"].append({"era": era, "test": [lo, hi], "params": {"w": 20, "era": era}, "alpha": a, "excess": a,
                                     "years": 1.0, "n": n, "half": D_by_era[era]["half"], "trade_key": f"{sha}{gi}"})
                res["trades"].append(pd.DataFrame({"ret": [ret] * n, "code": [f"{i:06d}" for i in range(n)], "exit_dist": 0.1,
                                                   "ci": range(n), "ti": range(n), "era": era, "fold": gi}))
                res["months"].append(pd.DataFrame())
        return res
    wf.calls = calls
    return wf


def _stub_stats(monkeypatch, conc=0.05):
    monkeypatch.setattr(S, "concentration", lambda D, res, train_path: {"top1pct": conc, "top_stock": conc, "top10d": conc})
    monkeypatch.setattr(S, "factor_record", lambda D, res: {"alpha_m": 0.01, "t_nw": 2.5})
    monkeypatch.setattr(S, "dsr_record", lambda res, n: {"prob_deflated_sharpe": 0.9, "n_trials": n})


def _train(tmp_path, tag: str, extra_lines: int = 0) -> "tuple[str, str]":
    p = tmp_path / f"train_{tag}.py"
    p.write_text(f"# {tag}\n" + "x = 1\n" * extra_lines + "def get_strategy():\n    return {}\n")
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _run(sha, score, eligible=True, void=False, beats=False, **kw):
    return {"train_sha": sha, "score": score, "void": void, "eligible": eligible, "beats_best": beats, "folds": [], **kw}


def _write_runs(rows):
    for r in rows:
        P._append(P.RUNS, r)


# ---------------------------------------------------------------- 풀

def test_pool_only_eligible_max10():
    runs = [_run(f"s{i:02d}", 0.10 + i * 0.01) for i in range(14)]
    runs += [_run("inelig", 9.0, eligible=False), _run("void", 9.0, void=True), _run("elig_none", 9.0, eligible=None),
             {"train_sha": "s13", "accept": True, "score": 0.23}, _run("ref", 9.0, reference=True),
             _run("s05", 0.15)]                                                    # 같은 sha 재평가 — 한 번만
    pool = S.pool(runs)
    shas = [r["train_sha"] for r in pool]
    assert len(pool) == 10 and len(set(shas)) == 10
    assert not {"inelig", "void", "elig_none", "ref"} & set(shas)
    assert [r["score"] for r in pool] == sorted((r["score"] for r in pool), reverse=True)
    assert shas[0] == "s13" and "s03" not in shas                                  # 상위 10 = s04..s13


def test_pool_keeps_below_top10_but_caps_by_score():
    runs = [_run(f"s{i:02d}", 0.50 + i * 0.01) for i in range(10)] + [_run("keep_low", 0.01, beats=True)]
    assert "keep_low" not in {r["train_sha"] for r in S.pool(runs)}               # 합집합 11 → score 순 10
    runs = [_run(f"s{i:02d}", 0.50 + i * 0.01) for i in range(5)] + [_run("keep_low", 0.01, beats=True)]
    assert "keep_low" in {r["train_sha"] for r in S.pool(runs)}


# ---------------------------------------------------------------- 거르개

def test_screen_null_bar(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    _stub_stats(monkeypatch)

    def alpha(s, era, half, null):                                                # 실제 0.20, 널 100회 0.00~0.99 균등 → p95 ≈ 0.94
        return 0.20 if null is None else (null[1] % 100) / 100.0
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(alpha))
    rec = _run(sha, 0.20)
    out = S.screen(rec, {"A": {"half": 0}, "B": {"half": 0}}, path)
    assert out["bars"]["beats_null_p95"] is False and out["pass"] is False
    assert out["null_n"] == S.N_NULL and all(c["null"][0] == "params" for c in P.walk_forward_c.calls if c["null"])

    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.20 if null is None else -0.05))
    out = S.screen(rec, {"A": {"half": 0}, "B": {"half": 0}}, path)
    assert out["pass"] is True and all(out["bars"].values())
    assert {c["half"]["A"] for c in P.walk_forward_c.calls} == {0}                # 거르개는 탐색 절반만


@pytest.mark.parametrize("bad", ["score", "ev", "conc", "trades", "repro"])
def test_screen_each_bar_fails(tmp_path, monkeypatch, bad):
    path, sha = _train(tmp_path, "a")
    _stub_stats(monkeypatch, conc=-0.01 if bad == "conc" else 0.05)
    a = -0.05 if bad == "score" else 0.20
    wf = _fake_wf(lambda s, e, h, null: a if null is None else -0.5, n=1 if bad == "trades" else 30, ret=-0.01 if bad == "ev" else 0.02)
    monkeypatch.setattr(P, "walk_forward_c", wf)
    rec = _run(sha, 0.999 if bad == "repro" else P.score_of([a] * 10))
    out = S.screen(rec, {"A": {"half": 0}, "B": {"half": 0}}, path)
    assert out["pass"] is False


# ---------------------------------------------------------------- 확인 1회

def test_confirm_refuses_same_sha_twice(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    row = S.confirm(sha, path)
    assert row["era"] == "C" and row["test"] == ["2025-01-02", "2026-09-12"] and row["params"]["era"] == "C"
    assert P.walk_forward_c.calls[-1]["eras"] == ("C",) and P.walk_forward_c.calls[-1]["half"] == {"C": 0}
    assert P.ERAS.keys() == {"A", "B"}                                             # 시대 바꿔치기는 원래대로
    with pytest.raises(SystemExit, match="이미"):
        S.confirm(sha, path)
    os.remove(S.CONFIRM)                                                           # 작업 트리 원장을 지워도 사본이 막는다
    with pytest.raises(SystemExit, match="이미"):
        S.confirm(sha, path)
    assert len(P.walk_forward_c.calls) == 1


def test_confirm_refuses_wrong_train_file(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    with pytest.raises(SystemExit, match="sha"):
        S.confirm("0" * 16, path)
    assert not P.walk_forward_c.calls


# ---------------------------------------------------------------- 잠금 1회

def test_lockbox_once(tmp_path, monkeypatch, _iso):
    path, sha = _train(tmp_path, "a")
    path2, sha2 = _train(tmp_path, "b")
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    with pytest.raises(SystemExit, match="확인"):                                  # 확인 없이 잠금은 못 연다
        S.lockbox(sha, path)
    assert _iso["locked"] == []
    S.confirm(sha, path)
    S.confirm(sha2, path2)
    row = S.lockbox(sha, path)
    assert row["pass"] is True and len(row["folds"]) == 11
    assert sorted(_iso["locked"]) == ["A", "B", "C"]
    assert {c["half"][e] for c in P.walk_forward_c.calls[2:] for e in c["half"]} == {1}
    with pytest.raises(SystemExit, match="잠금"):
        S.lockbox(sha2, path2)                                                    # 다른 sha 도 두 번째는 거부
    os.remove(S.LOCKBOX)
    with pytest.raises(SystemExit, match="잠금"):
        S.lockbox(sha2, path2)
    assert sorted(_iso["locked"]) == ["A", "B", "C"]                               # 거부는 잠금 절반을 만들기 전
    with pytest.raises(SystemExit, match="잠금"):
        S.confirm(sha2 + "x", path2)                                              # 잠금이 열린 뒤 확인도 닫힌다


def test_lockbox_bars(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1 if h == 0 else -0.1))
    S.confirm(sha, path)
    row = S.lockbox(sha, path)
    assert row["pass"] is False and row["bars"]["score_pos"] is False


# ---------------------------------------------------------------- main: 사람 열람·대체 없음

def _setup_main(tmp_path, monkeypatch, confirm_alpha, lock_alpha):
    """풀 셋(sha 셋, 둘 통과) — git 대신 미리 쓴 train.py 를 꺼내 준다."""
    _stub_stats(monkeypatch)
    trains = {t: _train(tmp_path, t, extra_lines=i) for i, t in enumerate(("hi", "lo", "bad"))}
    by_sha = {sha: p for p, sha in trains.values()}
    monkeypatch.setattr(S, "train_commits", lambda: {sha: f"c_{sha[:6]}" for sha in by_sha})
    monkeypatch.setattr(S, "extract_train", lambda commit, sha: by_sha[sha])
    monkeypatch.setattr(P, "trial_state", lambda: {"n": 3, "since_keep": 30, "stop": True})
    good = {trains["hi"][1], trains["lo"][1]}

    def alpha(s, era, half, null):
        if null is not None:
            return -0.5
        if era == "C" and half == 0:
            return confirm_alpha[s == trains["hi"][1]]
        if half == 1:
            return lock_alpha
        return 0.2 if s in good else -0.1
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(alpha))
    _write_runs([_run(trains["hi"][1], 0.2), _run(trains["lo"][1], 0.2), _run(trains["bad"][1], 0.3)])
    return trains


def test_reviewed_sha_required(tmp_path, monkeypatch, _iso):
    trains = _setup_main(tmp_path, monkeypatch, (0.1, 0.3), 0.1)
    S.main(["--screen"])
    scr = P._read_jsonl(S.SCREEN)[-1]
    assert set(scr["survivors"]) == {trains["hi"][1], trains["lo"][1]}
    assert not os.path.exists(S.CONFIRM) and _iso["locked"] == []                 # 거르개는 확인·잠금을 열지 않는다
    for bad in ("", trains["hi"][1], f"{trains['hi'][1]},{'f' * 16}", f"{trains['hi'][1]},{trains['bad'][1]}"):
        with pytest.raises(SystemExit, match="reviewed"):
            S.main(["--confirm", "--reviewed", bad])
    assert not os.path.exists(S.CONFIRM) and _iso["locked"] == []
    S.main(["--confirm", "--reviewed", f"{trains['lo'][1]},{trains['hi'][1]}"])
    sel = json.load(open(S.SELECTION))
    assert sel["chosen"]["train_sha"] == trains["hi"][1] and sel["frozen_params"]["era"] == "C"
    assert len(P._read_jsonl(S.CONFIRM)) == 2 and len(P._read_jsonl(S.LOCKBOX)) == 1


def test_confirm_refuses_stale_screen(tmp_path, monkeypatch):
    trains = _setup_main(tmp_path, monkeypatch, (0.1, 0.3), 0.1)
    S.main(["--screen"])
    path, sha = _train(tmp_path, "new")
    _write_runs([_run(sha, 0.9)])                                                  # 거르개 뒤 루프가 더 돌았다
    with pytest.raises(SystemExit, match="screen"):
        S.main(["--confirm", "--reviewed", f"{trains['lo'][1]},{trains['hi'][1]}"])


def test_screen_refuses_while_loop_running(tmp_path, monkeypatch):
    _setup_main(tmp_path, monkeypatch, (0.1, 0.3), 0.1)
    monkeypatch.setattr(P, "trial_state", lambda: {"n": 3, "since_keep": 0, "stop": False})
    with pytest.raises(SystemExit, match="루프"):
        S.main(["--screen"])


def test_no_fallback_after_lockbox_fail(tmp_path, monkeypatch, _iso):
    trains = _setup_main(tmp_path, monkeypatch, (0.1, 0.3), -0.1)                  # 1위(hi)가 잠금에서 실패
    S.main(["--screen"])
    S.main(["--confirm", "--reviewed", f"{trains['hi'][1]},{trains['lo'][1]}"])
    sel = json.load(open(S.SELECTION))
    assert sel["chosen"] is None and sel["frozen_params"] is None and sel["picked"]["train_sha"] == trains["hi"][1]
    locked_shas = {c["sha"] for c in P.walk_forward_c.calls if 1 in c["half"].values()}
    assert locked_shas == {trains["hi"][1]}                                        # 2위(lo)는 잠금에서 돌리지 않았다
    assert len(P._read_jsonl(S.LOCKBOX)) == 1
    with pytest.raises(SystemExit, match="잠금"):
        S.main(["--confirm", "--reviewed", f"{trains['hi'][1]},{trains['lo'][1]}"])


def test_tie_breaks_on_shorter_train(tmp_path, monkeypatch):
    trains = _setup_main(tmp_path, monkeypatch, (0.2, 0.2), 0.1)                   # 확인 score 동점 → 줄 수 적은 hi
    S.main(["--screen"])
    S.main(["--confirm", "--reviewed", f"{trains['hi'][1]},{trains['lo'][1]}"])
    assert json.load(open(S.SELECTION))["chosen"]["train_sha"] == trains["hi"][1]


def test_no_survivors_writes_selection_without_holdouts(tmp_path, monkeypatch, _iso):
    _setup_main(tmp_path, monkeypatch, (0.1, 0.3), 0.1)
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: -0.1))
    S.main(["--screen"])
    S.main(["--confirm", "--reviewed", ""])
    sel = json.load(open(S.SELECTION))
    assert sel["chosen"] is None and sel["reasons"] and _iso["locked"] == [] and not os.path.exists(S.CONFIRM)


# ---------------------------------------------------------------- git: 버린 시도(reset --hard)도 찾는다

def test_train_commits_finds_reset_commit(tmp_path, monkeypatch):
    import subprocess

    repo = tmp_path / "repo"
    (repo / "research" / "combined").mkdir(parents=True)

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    if subprocess.run(["git", "-C", str(repo), "config", "user.email"], capture_output=True, text=True).returncode != 0:
        pytest.skip("git 신원 설정 없음 — 신원을 덮어쓰지 않는다")
    tp = repo / "research" / "combined" / "train.py"
    shas = []
    for body in ("v1\n", "v2_discarded\n"):
        tp.write_text(body)
        git("add", "research/combined/train.py")
        git("commit", "-q", "-m", body.strip())
        shas.append((hashlib.sha256(body.encode()).hexdigest()[:16], git("rev-parse", "HEAD")))
    git("reset", "-q", "--hard", "HEAD~1")                                        # 루프가 버린 시도
    monkeypatch.setattr(S, "ROOT", str(repo))
    found = S.train_commits()
    assert found[shas[0][0]] == shas[0][1] and found[shas[1][0]] == shas[1][1]
    p = S.extract_train(shas[1][1], shas[1][0])
    assert open(p).read() == "v2_discarded\n"
    with pytest.raises(SystemExit, match="sha"):
        S.extract_train(shas[0][1], shas[1][0])                                   # 내용이 다른 commit 은 거부
