"""결합 루프 평가기 가드 — 위반을 주입해 실제로 막히는지 본다(GUARDRAILS §4.5: 통과만 보는 테스트는 규칙이 죽어도 초록)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from research.autoresearch import prepare as V4
from research.combined import prepare as P

TOY_FOLDS = [("2022-01-03", "2023-01-02", "2024-01-01"), ("2022-01-03", "2024-01-01", "2025-01-01")]


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    """원장·BEST 는 임시 경로로, 청산선 캐시는 비운다(장난감끼리 era·모양이 같아 섞이지 않게)."""
    monkeypatch.setattr(P, "RUNS", str(tmp_path / "RUNS.jsonl"))
    monkeypatch.setattr(P, "BEST", str(tmp_path / "BEST.json"))
    monkeypatch.setattr(P, "SHADOW", str(tmp_path / "shadow"))
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

def _best(score=0.0, conc=None):
    conc = conc or {"top1pct": 0.02, "top_stock": 0.02, "top10d": 0.02}
    json.dump({"score": score, "conc": conc, "folds": [{"alpha": 0.0, "trade_key": f"k{i}"} for i in range(10)]}, open(P.BEST, "w"))


def test_keep_counts_changed_folds_only():
    _best()
    conc = {"top1pct": 0.02, "top_stock": 0.02, "top10d": 0.02}
    folds = [{"alpha": 0.0, "trade_key": f"k{i}"} for i in range(10)]
    folds[3] = {"alpha": 0.01, "trade_key": "new3"}
    folds[7] = {"alpha": 0.01, "trade_key": "new7"}
    ok, why = P.beats_c({"void": False, "score": 0.006}, folds, conc)
    assert ok, why
    folds[7] = {"alpha": -0.01, "trade_key": "new7"}                                  # 바뀐 2개 중 1개만 개선 = 과반 아님
    assert not P.beats_c({"void": False, "score": 0.006}, folds, conc)[0]
    same = [{"alpha": 0.5, "trade_key": f"k{i}"} for i in range(10)]                 # 체결이 하나도 안 바뀌면 keep 아님
    assert not P.beats_c({"void": False, "score": 0.5}, same, conc)[0]
    folds[7] = {"alpha": 0.01, "trade_key": "new7"}
    assert not P.beats_c({"void": False, "score": 0.004}, folds, conc)[0]             # Δscore < 0.005
    assert not P.beats_c({"void": True, "score": 0.5}, folds, conc)[0]


def test_keep_concentration_vs_best():
    _best()
    folds = [{"alpha": 0.01, "trade_key": f"n{i}"} for i in range(10)]
    s = {"void": False, "score": 0.01}
    assert not P.beats_c(s, folds, {"top1pct": 0.02, "top_stock": 0.02 - 0.011, "top10d": 0.02})[0]
    assert P.beats_c(s, folds, {"top1pct": 0.02, "top_stock": 0.02 - 0.009, "top10d": 0.02})[0]
    for k in ("top1pct", "top10d"):                                                  # BEST 대비(자기 점수 대비가 아니다)
        c = {"top1pct": 0.02, "top_stock": 0.02, "top10d": 0.02, k: 0.0}
        assert not P.beats_c(s, folds, c)[0]
    os.remove(P.BEST)
    assert P.beats_c(s, folds, {"top1pct": -1, "top_stock": -1, "top10d": -1})[0]   # 첫 유효 실행


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


def test_trial_cap_since_first_keep():
    for i in range(40):                                                              # keep 전에는 무 keep 이 쌓여도 멈추지 않는다
        _eval(f"pre{i}")
    assert not P.trial_state()["stop"]
    _eval("k")
    P._append(P.RUNS, {"train_sha": "k", "accept": True})
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


def test_lookahead_sees_news_and_gap_keys(tmp_path, monkeypatch):
    """새 V 키(갭·뉴스)도 절단 검사 대상이다 — 내일 뉴스를 읽는 train 이 걸려야 한다."""
    Ds = _toy_loop(monkeypatch)
    peek = _write_train(tmp_path / "pn.py", "def signals(V, x):\n    nxt = pd.DataFrame(V['n_feat'].T.astype(float)).shift(-1).to_numpy().T\n"
                                            "    return {'cand': V['uni'] & ~(nxt > 1), 'score': V['C']}\n")
    ok, _ = P.causality_c(Ds, {"folds": [{"era": "A", "params": {}}]}, peek)
    assert not ok


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
        "research/autoresearch/prepare.py", "research/experiments/leader21.py", "research/experiments/indep30.py"}
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
    for h in (P.SHADOW, f"{home}/git/daytrade-it", f"{home}/git/quant-airflow", f"{home}/.config", f"{home}/.ssh"):
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
