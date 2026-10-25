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

def _fake_wf(alpha_of, n=30, ret=0.02, boom=None):
    """walk_forward_c 대역 — 받은 ``eras`` 의 폴드마다 alpha_of(train_sha, era, half, null) 로 폴드 alpha. boom(eras) 참이면 예외."""
    calls = []

    def wf(D_by_era, workers, train_path=P.TRAIN, fixed=None, null=None, neutral=None, cost_mult=P.COST_MULT, exclude=(), mask=None,
           eras=None):
        assert eras is not None, "select 는 eras= 를 늘 넘긴다"
        sha = hashlib.sha256(open(train_path, "rb").read()).hexdigest()[:16]
        calls.append({"sha": sha, "eras": tuple(eras), "null": null, "fixed": fixed is not None,
                      "half": {e: D_by_era[e]["half"] for e in eras}, "data_eras": {e: D_by_era[e]["era"] for e in eras}})
        if boom and boom(tuple(eras)):
            raise RuntimeError("계산 중 죽음(주입)")
        res = {"folds": [], "trades": [], "months": [], "daily": []}
        for era, E in eras.items():
            for _lo_tr, lo, hi in E["folds"]:
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
    def c(D, res, train_path, eras):
        assert tuple(eras) == ("A", "B")
        return {"top1pct": conc, "top_stock": conc, "top10d": conc}
    monkeypatch.setattr(S, "concentration", c)
    monkeypatch.setattr(S, "factor_record", lambda D, res, eras: {"alpha_m": 0.01, "t_nw": 2.5})
    monkeypatch.setattr(S, "dsr_record", lambda res, n: {"prob_deflated_sharpe": 0.9, "n_trials": n})


def _train(tmp_path, tag: str, extra_lines: int = 0) -> "tuple[str, str]":
    p = tmp_path / f"train_{tag}.py"
    p.write_text(f"# {tag}\n" + "K = 1\n" * extra_lines + "def signals(V, x):\n    return {}\n\n\ndef cfg(x):\n    return {}\n\n\n"
                 "def get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': cfg}\n")
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _run(sha, score, eligible=True, void=False, beats=False, commit="abc123def456", dirty=False, **kw):
    return {"train_sha": sha, "score": score, "void": void, "eligible": eligible, "beats_best": beats, "folds": [],
            "commit": commit, "dirty": dirty, **kw}


def _write_runs(rows):
    for r in rows:
        P._append(P.RUNS, r)


def _screened(shas: list) -> None:
    """원장 행 + 그 sha 들이 통과한 거르개 기록(직접 쓴다 — confirm·lockbox 단위 시험용)."""
    _write_runs([_run(s, 0.2) for s in shas])
    P._append(S.SCREEN, {"pool": [r["train_sha"] for r in S.pool(P._read_jsonl(P.RUNS))], "n_trials": P.trial_state()["n"],
                         "survivors": list(shas), "review": [], "results": {}, "pool_rows": []})


# ---------------------------------------------------------------- 풀

def test_pool_only_eligible_max10():
    runs = [_run(f"s{i:02d}", 0.10 + i * 0.01) for i in range(14)]
    runs += [_run("inelig", 9.0, eligible=False), _run("void", 9.0, void=True), _run("elig_none", 9.0, eligible=None),
             {"train_sha": "s13", "accept": True, "score": 0.23}, _run("ref", 9.0, reference=True),
             {"train_sha": "crash", "crash": True, "score": 9.0, "eligible": True, "folds": []},
             _run("s05", 0.15)]                                                    # 같은 sha 재평가 — 한 번만
    pool = S.pool(runs)
    shas = [r["train_sha"] for r in pool]
    assert len(pool) == 10 and len(set(shas)) == 10
    assert not {"inelig", "void", "elig_none", "ref", "crash"} & set(shas)
    assert [r["score"] for r in pool] == sorted((r["score"] for r in pool), reverse=True)
    assert shas[0] == "s13" and "s03" not in shas                                  # 상위 10 = s04..s13


def test_pool_refuses_dirty_or_no_commit():
    runs = [_run("ok", 0.1), _run("dirty", 0.9, dirty=True), _run("nocommit", 0.8, commit=None), _run("unknown", 0.7, dirty=None)]
    assert [r["train_sha"] for r in S.pool(runs)] == ["ok"]
    rej = {x["train_sha"]: x["reason"] for x in S.pool_rejects(runs)}
    assert set(rej) == {"dirty", "nocommit", "unknown"} and "commit" in rej["nocommit"] and "dirty" in rej["dirty"]


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
    out = S.screen(rec, {"A": {"half": 0, "era": "A"}, "B": {"half": 0, "era": "B"}}, path)
    assert out["bars"]["beats_null_p95"] is False and out["pass"] is False
    assert out["null_n"] == S.N_NULL and all(c["null"][0] == "params" for c in P.walk_forward_c.calls if c["null"])
    assert {c["eras"] for c in P.walk_forward_c.calls} == {("A", "B")}

    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.20 if null is None else -0.05))
    out = S.screen(rec, {"A": {"half": 0, "era": "A"}, "B": {"half": 0, "era": "B"}}, path)
    assert out["pass"] is True and all(out["bars"].values())
    assert {c["half"]["A"] for c in P.walk_forward_c.calls} == {0}                # 거르개는 탐색 절반만


def test_screen_null_too_few_finite_fails(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    _stub_stats(monkeypatch)
    nan = float("nan")
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.20 if null is None else (nan if null[1] > 85 else -0.5)))
    out = S.screen(_run(sha, 0.20), {"A": {"half": 0, "era": "A"}, "B": {"half": 0, "era": "B"}}, path)
    assert out["pass"] is False and out["null_n"] == 85 and any("null 표본 부족" in r for r in out["reasons"])


@pytest.mark.parametrize("bad", ["score", "ev", "conc", "trades", "repro"])
def test_screen_each_bar_fails(tmp_path, monkeypatch, bad):
    path, sha = _train(tmp_path, "a")
    _stub_stats(monkeypatch, conc=-0.01 if bad == "conc" else 0.05)
    a = -0.05 if bad == "score" else 0.20
    wf = _fake_wf(lambda s, e, h, null: a if null is None else -0.5, n=1 if bad == "trades" else 30, ret=-0.01 if bad == "ev" else 0.02)
    monkeypatch.setattr(P, "walk_forward_c", wf)
    rec = _run(sha, 0.999 if bad == "repro" else P.score_of([a] * 10))
    out = S.screen(rec, {"A": {"half": 0, "era": "A"}, "B": {"half": 0, "era": "B"}}, path)
    assert out["pass"] is False and out["reasons"]


# ---------------------------------------------------------------- 확인 1회

def test_confirm_refuses_same_sha_twice(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    _screened([sha])
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    row = S.confirm(sha, path, {sha})
    assert row["era"] == "C" and row["test"] == ["2025-01-02", "2026-09-12"] and row["params"]["era"] == "C"
    assert P.walk_forward_c.calls[-1]["eras"] == ("C",) and P.walk_forward_c.calls[-1]["half"] == {"C": 0}
    assert P.walk_forward_c.calls[-1]["data_eras"] == {"C": "C"}
    assert [r["status"] for r in P._read_jsonl(S.CONFIRM)] == ["opened", "result"]   # 열기 전에 기록
    with pytest.raises(SystemExit, match="이미"):
        S.confirm(sha, path, {sha})
    os.remove(S.CONFIRM)                                                           # 작업 트리 원장을 지워도 사본이 막는다
    with pytest.raises(SystemExit, match="이미"):
        S.confirm(sha, path, {sha})
    assert len(P.walk_forward_c.calls) == 1


def test_confirm_crash_burns_attempt(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    _screened([sha])
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1, boom=lambda eras: True))
    with pytest.raises(RuntimeError):
        S.confirm(sha, path, {sha})
    assert [r["status"] for r in P._read_jsonl(S.CONFIRM)] == ["opened"]
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    with pytest.raises(SystemExit, match="이미"):
        S.confirm(sha, path, {sha})
    assert not P.walk_forward_c.calls


def test_confirm_refuses_wrong_train_file(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    _screened([sha, "0" * 16])
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    with pytest.raises(SystemExit, match="sha"):
        S.confirm("0" * 16, path, {"0" * 16})
    assert not P.walk_forward_c.calls and not os.path.exists(S.CONFIRM)


def test_confirm_lockbox_require_screen_and_reviewed(tmp_path, monkeypatch, _iso):
    path, sha = _train(tmp_path, "a")
    path2, sha2 = _train(tmp_path, "b")
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    with pytest.raises(SystemExit, match="거르개"):                                # 거르개 기록 없음
        S.confirm(sha, path, {sha})
    _screened([sha])
    _write_runs([_run(sha2, 0.05)])                                                # 거르개 뒤 원장 변경 → 낡음
    with pytest.raises(SystemExit, match="screen"):
        S.confirm(sha, path, {sha})
    P._append(S.SCREEN, {"pool": [r["train_sha"] for r in S.pool(P._read_jsonl(P.RUNS))], "n_trials": P.trial_state()["n"],
                         "survivors": [sha], "review": [], "results": {}, "pool_rows": []})
    with pytest.raises(SystemExit, match="거르개 통과"):                           # import 로 통과 밖 sha 를 열 수 없다
        S.confirm(sha2, path2, {sha2})
    with pytest.raises(SystemExit, match="reviewed"):                              # 통과했어도 reviewed 밖이면 거부
        S.confirm(sha, path, set())
    S.confirm(sha, path, {sha})
    with pytest.raises(SystemExit, match="reviewed"):
        S.lockbox(sha, path, set())
    assert _iso["locked"] == [] and len(P.walk_forward_c.calls) == 1


# ---------------------------------------------------------------- 잠금 1회

def test_lockbox_once(tmp_path, monkeypatch, _iso):
    path, sha = _train(tmp_path, "a")
    path2, sha2 = _train(tmp_path, "b")
    _screened([sha, sha2])
    rv = {sha, sha2}
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    with pytest.raises(SystemExit, match="확인"):                                  # 확인 없이 잠금은 못 연다
        S.lockbox(sha, path, rv)
    assert _iso["locked"] == []
    S.confirm(sha, path, rv)
    S.confirm(sha2, path2, rv)
    row = S.lockbox(sha, path, rv)
    assert row["pass"] is True and len(row["folds"]) == 11
    assert sorted(_iso["locked"]) == ["A", "B", "C"]
    lk = P.walk_forward_c.calls[2:]
    assert [c["eras"] for c in lk] == [("A", "B"), ("C",)] and {h for c in lk for h in c["half"].values()} == {1}
    assert [r["status"] for r in P._read_jsonl(S.LOCKBOX)] == ["opened", "result"]
    with pytest.raises(SystemExit, match="잠금"):
        S.lockbox(sha2, path2, rv)                                                # 다른 sha 도 두 번째는 거부
    os.remove(S.LOCKBOX)
    with pytest.raises(SystemExit, match="잠금"):
        S.lockbox(sha2, path2, rv)
    assert sorted(_iso["locked"]) == ["A", "B", "C"]                               # 거부는 잠금 절반을 만들기 전
    with pytest.raises(SystemExit, match="잠금"):
        S.confirm(sha2 + "x", path2, rv)                                          # 잠금이 열린 뒤 확인도 닫힌다


def test_lockbox_crash_burns_attempt(tmp_path, monkeypatch, _iso):
    path, sha = _train(tmp_path, "a")
    _screened([sha])
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1))
    S.confirm(sha, path, {sha})
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1, boom=lambda eras: eras == ("C",)))
    with pytest.raises(RuntimeError):
        S.lockbox(sha, path, {sha})                                               # A·B 를 돈 뒤 C 에서 죽음
    assert [r["status"] for r in P._read_jsonl(S.LOCKBOX)] == ["opened"]
    with pytest.raises(SystemExit, match="잠금"):
        S.lockbox(sha, path, {sha})
    assert len(P.walk_forward_c.calls) == 2


def test_lockbox_bars(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    _screened([sha])
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1 if h == 0 else -0.1))
    S.confirm(sha, path, {sha})
    row = S.lockbox(sha, path, {sha})
    assert row["pass"] is False and row["bars"]["score_pos"] is False


def test_pinned_detects_change_during_compute(tmp_path, monkeypatch):
    path, sha = _train(tmp_path, "a")
    real = P.load_strategy
    n = {"k": 0}

    def flaky(p):                                                                  # 두 번째 로드(계산 뒤 대조)는 다른 sha
        n["k"] += 1
        S_ = real(p)
        return {**S_, "sha": S_["sha"] if n["k"] == 1 else "f" * 16}
    monkeypatch.setattr(P, "load_strategy", flaky)
    with pytest.raises(SystemExit, match="바뀌었다"):
        with S._pinned(path, sha) as tp:
            assert open(tp, "rb").read() == open(path, "rb").read() and not os.access(tp, os.W_OK)


# ---------------------------------------------------------------- main: 사람 열람·대체 없음

def _setup_main(tmp_path, monkeypatch, confirm_alpha, lock_alpha):
    """풀 셋(sha 셋, 둘 통과) — git 대신 미리 쓴 train.py 를 꺼내 준다."""
    _stub_stats(monkeypatch)
    trains = {t: _train(tmp_path, t, extra_lines=i) for i, t in enumerate(("hi", "lo", "bad"))}
    by_sha = {sha: p for p, sha in trains.values()}
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
    _write_runs([_run(trains["hi"][1], 0.2), _run(trains["lo"][1], 0.2), _run(trains["bad"][1], 0.3),
                 _run("d" * 16, 0.9, dirty=True)])
    return trains


def test_reviewed_sha_required(tmp_path, monkeypatch, _iso):
    trains = _setup_main(tmp_path, monkeypatch, (0.1, 0.3), 0.1)
    S.main(["--screen"])
    scr = P._read_jsonl(S.SCREEN)[-1]
    assert set(scr["survivors"]) == {trains["hi"][1], trains["lo"][1]} and "d" * 16 not in scr["pool"]
    assert not os.path.exists(S.CONFIRM) and _iso["locked"] == []                 # 거르개는 확인·잠금을 열지 않는다
    for bad in ("", trains["hi"][1], f"{trains['hi'][1]},{'f' * 16}", f"{trains['hi'][1]},{trains['bad'][1]}"):
        with pytest.raises(SystemExit, match="reviewed"):
            S.main(["--confirm", "--reviewed", bad])
    assert not os.path.exists(S.CONFIRM) and _iso["locked"] == []
    S.main(["--confirm", "--reviewed", f"{trains['lo'][1]},{trains['hi'][1]}"])
    sel = json.load(open(S.SELECTION))
    assert sel["chosen"]["train_sha"] == trains["hi"][1] and sel["frozen_params"]["era"] == "C"
    assert any("dirty" in r for r in sel["reasons"])                               # 풀 제외 사유가 남는다
    assert len(P._read_jsonl(S.CONFIRM)) == 4 and len(P._read_jsonl(S.LOCKBOX)) == 2


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


def test_resume_stops_on_opened_without_result(tmp_path, monkeypatch):
    trains = _setup_main(tmp_path, monkeypatch, (0.1, 0.3), 0.1)
    S.main(["--screen"])
    rv = f"{trains['hi'][1]},{trains['lo'][1]}"
    wf_ok = P.walk_forward_c
    monkeypatch.setattr(P, "walk_forward_c", _fake_wf(lambda s, e, h, null: 0.1, boom=lambda eras: eras == ("C",)))
    with pytest.raises(RuntimeError):
        S.main(["--confirm", "--reviewed", rv])
    monkeypatch.setattr(P, "walk_forward_c", wf_ok)
    n0 = len(wf_ok.calls)
    with pytest.raises(SystemExit, match="확인 중단 — 사람 판단"):
        S.main(["--confirm", "--reviewed", rv])
    assert len(wf_ok.calls) == n0 and not os.path.exists(S.LOCKBOX)


def test_no_fallback_after_lockbox_fail(tmp_path, monkeypatch, _iso):
    trains = _setup_main(tmp_path, monkeypatch, (0.1, 0.3), -0.1)                  # 1위(hi)가 잠금에서 실패
    S.main(["--screen"])
    S.main(["--confirm", "--reviewed", f"{trains['hi'][1]},{trains['lo'][1]}"])
    sel = json.load(open(S.SELECTION))
    assert sel["chosen"] is None and sel["frozen_params"] is None and sel["picked"]["train_sha"] == trains["hi"][1]
    locked_shas = {c["sha"] for c in P.walk_forward_c.calls if 1 in c["half"].values()}
    assert locked_shas == {trains["hi"][1]}                                        # 2위(lo)는 잠금에서 돌리지 않았다
    assert len(P._read_jsonl(S.LOCKBOX)) == 2
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


# ---------------------------------------------------------------- git: RUNS 의 commit 에서 꺼낸다

def test_extract_train_from_commit(tmp_path, monkeypatch):
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
        shas.append((hashlib.sha256(body.encode()).hexdigest()[:16], git("rev-parse", "HEAD")[:12]))
    git("reset", "-q", "--hard", "HEAD~1")                                        # 루프가 버린 시도 — commit 은 원장에 있다
    monkeypatch.setattr(S, "ROOT", str(repo))
    p = S.extract_train(shas[1][1], shas[1][0])
    assert open(p).read() == "v2_discarded\n"
    with pytest.raises(SystemExit, match="sha"):
        S.extract_train(shas[0][1], shas[1][0])                                   # 내용이 다른 commit 은 거부
