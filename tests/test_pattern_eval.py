"""pattern_eval — Engine.evaluate_fast(njit 한 번 호출) 가 evaluate_ref(파이썬 접착)와 비트 동일한지.

실캐시 대조(simnode FWM 캐시, PATTERN_GA_VARIANT=fwm, G3A·G7 메뉴, 폴드 1~4 · 무작위·선별·CMA 탐색 config
14,040개 중 체결 있는 config 9,170개 · counts_only 둘 다 · 시험창 · 재평가 시드 · 널 2b 섞은 F — 불일치 0)와 run_evolve
후보 전체 바이트 비교(G7 메뉴, 원래 경로 vs 빠른 평가 + 코어 bipop: trades CSV sha256 같음, JSON 같음)는 러너 밖에서
했다. 여기서는 합성 캐시로 같은 대조를 CI 에서 돈다.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "research/experiments")

from research.signals import pattern_eval as PEV  # noqa: E402

from test_pattern_ga import _loose, _synthetic_cache  # noqa: E402

KEYS = ("n", "n_pass", "pen", "feasible", "fit", "lower")


def _same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and math.isnan(x):
            if not (isinstance(y, float) and math.isnan(y)):
                return False
        elif x != y or type(x) is not type(y):
            return False
    return True


@pytest.mark.parametrize("m", list(range(0, 20)) + [63, 64, 65, 127])
def test_year_mean_matches_numpy_mean(m):
    """벌점 연도 평균 = np.mean(list) 의 덧셈 순서(8 미만 순차, 8 이상 8 갈래)."""
    rng = np.random.default_rng(m)
    for _ in range(300):
        v = rng.random(max(m, 1)) * rng.choice([1.0, 1e-3, 7.0])
        want = float(np.mean(list(v[:m]))) if m else 0.0
        assert PEV.year_mean_np(v, m) == want


def test_evaluate_fast_bit_identical_synthetic(tmp_path):
    from swing_it.validation.walkforward import Fold

    G, E = _synthetic_cache(tmp_path)
    folds = [Fold("2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01"),
             Fold("2017-06-01", "2019-06-01", "2019-06-01", "2020-06-01")]
    wins = G.fold_windows(E, folds)
    rng = np.random.default_rng(7)
    X = G.random_x(rng, 240)
    cfgs = [G.decode(x) for x in X[:80]]                       # 무작위(대부분 진입 0)
    for x in X[80:]:                                           # 느슨한 거르기 + 무작위 청산 유전자(체결 있음)
        rg = G.decode(x)
        g = _loose(G, entry=("horiz", "trend")[int(x[0] > 0.5)])
        g.update({k: rg[k] for k in G.EXIT_GENES if k in rg})
        cfgs.append(G.canonical(g))
    n_trades = 0
    for tr, te in wins:
        for g in cfgs:
            for w in (tr, te):
                for co in (False, True):
                    for seed in (G.FIT_SEED, G.REEVAL_SEED):
                        a = E.evaluate_ref(g, w, seed=seed, counts_only=co)
                        b = E.evaluate_fast(g, w, seed=seed, counts_only=co)
                        assert _same(a, b), (a, b, g)
                        n_trades += (not co) and a["n"] > 0
    assert n_trades > 50                              # 실제로 포트폴리오·꼬리 경로를 탔다


def test_evaluate_dispatch_and_switch(tmp_path, monkeypatch):
    G, E = _synthetic_cache(tmp_path)
    from swing_it.validation.walkforward import Fold

    tr, _ = G.fold_windows(E, [Fold("2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01")])[0]
    g = G.decode(G.x0_feasible())
    calls = []
    monkeypatch.setattr(E, "evaluate_fast", lambda *a, **k: calls.append("fast") or {"fit": 0.0})
    E.evaluate(g, tr)
    assert calls == ["fast"]
    E.evaluate(g, tr, want_monthly=True)              # 월 배열이 필요하면 원래 경로
    assert calls == ["fast"]
    monkeypatch.setattr(G, "FAST_EVAL", False)
    E.evaluate(g, tr)
    assert calls == ["fast"]


def test_evaluate_fast_fw_branch_rep_and_seal(tmp_path, monkeypatch):
    """F1 하락 쐐기 거르기(FW)·F1 끔 짝(fw_off)·널 2b(섞은 F)·봉인 자르기(LIM) 경로도 비트 동일."""
    from swing_it.validation.walkforward import Fold

    G, E = _synthetic_cache(tmp_path)
    monkeypatch.setattr(G, "FW", True)
    tr, te = G.fold_windows(E, [Fold("2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01")])[0]
    rng = np.random.default_rng(11)
    cfgs = []
    for x in G.random_x(rng, 60):
        rg = G.decode(x)
        g = _loose(G, entry=("horiz", "trend")[int(x[0] > 0.5)])
        g.update({k: rg[k] for k in G.EXIT_GENES if k in rg})
        g = G.canonical(g)
        g["fw_lo"] = float(rng.uniform(-0.15, 0.15))
        cfgs.append(g)
    checked = 0
    for rep in (None, 1, 2, None):
        E.set_rep(rep)
        for fw_off in (False, True):
            E.fw_off = fw_off
            for g in cfgs:
                for w in (tr, te):
                    a, b = E.evaluate_ref(g, w), E.evaluate_fast(g, w)
                    assert _same(a, b), (rep, fw_off, a, b)
                    checked += a["n_pass"] > 0
    E.fw_off = False
    E.LIM = np.minimum(np.asarray(E.LIM), E.row("2019-06-03")).astype(np.int64)   # 봉인 자르기(run_evolve 와 같은 식)
    for g in cfgs:
        assert _same(E.evaluate_ref(g, te), E.evaluate_fast(g, te))
    assert checked > 0


@pytest.mark.parametrize("floors", [(12.0, 4.0), (2.0, 0.5)])
def test_screen_fast_matches_row_evaluate(tmp_path, monkeypatch, floors):
    """건수만 선별 덩어리(연도 조각 가지치기)의 실현 가능·진입 수 = 행마다 evaluate_ref(counts_only) — 재진입·FW·널 2b 포함."""
    from swing_it.validation.walkforward import Fold

    G, E = _synthetic_cache(tmp_path)
    monkeypatch.setattr(G, "FLOOR_PER_YEAR", floors[0])
    monkeypatch.setattr(G, "FLOOR_YEAR", floors[1])
    folds = [Fold("2018-01-01", "2019-09-01", "2019-09-01", "2020-06-01"),        # 합성 사건이 있는 해(실현 가능 행이 나오게)
             Fold("2017-06-01", "2019-06-01", "2019-06-01", "2020-06-01")]
    wins = G.fold_windows(E, folds)
    rng = np.random.default_rng(5)
    X = G.random_x(rng, 400)
    gs = [G.decode(x) for x in X[:150]]
    for x in X[150:]:
        rg = G.decode(x)
        g = _loose(G, entry=("horiz", "trend", "bottom")[int(3 * x[0]) % 3])
        g.update({k: rg[k] for k in G.EXIT_GENES if k in rg})
        g["reentry"] = ("off", "F", "line")[int(3 * x[1]) % 3]
        g["stop_anchor"] = ("E", "lower", "dynamic")[int(3 * x[2]) % 3]
        gs.append(G.canonical(g))
    seen = {"pruned": 0, "feasible": 0, "kept_infeasible": 0, "re_feasible": 0}
    gs_fw = [{**g, "fw_lo": float(v)} for g, v in zip(gs, rng.uniform(-0.15, 0.15, len(gs)))]
    for rep, fw in ((None, False), (2, False), (None, True)):
        E.set_rep(rep)
        monkeypatch.setattr(G, "FW", fw)
        cfgs = gs_fw if fw else gs
        for tr, te in wins:
            for w in (tr, te):
                feas, nn = E.screen_fast(cfgs, w)
                for g, f, n in zip(cfgs, feas, nn):
                    r = E.evaluate_ref(g, w, counts_only=True)
                    assert bool(f) == r["feasible"], (g, r)
                    if n >= 0:
                        assert n == r["n"]
                    seen["pruned"] += n < 0
                    seen["feasible"] += bool(f)
                    seen["kept_infeasible"] += n >= 0 and not f
                    seen["re_feasible"] += bool(f) and g.get("reentry") != "off"
    E.set_rep(None)
    assert seen["pruned"] > 100
    if floors[1] < 4.0:                               # 낮은 하한: 실현 가능(재진입 포함)·버리지 못한 실현 불가 경로도 탄다
        assert seen["feasible"] > 20 and seen["re_feasible"] > 0 and seen["kept_infeasible"] > 0


def test_screen_chunk_fast_equals_reference(tmp_path, monkeypatch):
    """screen_chunk — 빠른 경로(덩어리 njit)와 원래 경로(PATTERN_GA_FAST=0, 행마다 evaluate_ref)의 (벡터, 진입 수)가 같다."""
    from swing_it.validation.walkforward import Fold

    G, E = _synthetic_cache(tmp_path)
    monkeypatch.setattr(G, "FLOOR_PER_YEAR", 2.0)
    monkeypatch.setattr(G, "FLOOR_YEAR", 0.5)
    base = G.random_x

    def rx(rng, n):                                   # 3행마다 느슨한 거르기(체결 있는 행이 나오게) + 그 행의 청산 유전자
        X = base(rng, n)
        for i in range(0, n, 3):
            rg = G.decode(X[i])
            g = _loose(G, entry=("horiz", "trend", "bottom")[(i // 3) % 3])
            g.update({k: rg[k] for k in G.EXIT_GENES if k in rg})
            X[i] = G.encode(G.canonical(g))
        return X

    monkeypatch.setattr(G, "random_x", rx)
    tr, _ = G.fold_windows(E, [Fold("2018-01-01", "2019-09-01", "2019-09-01", "2020-06-01")])[0]
    Xa, na = G.screen_chunk(E, tr, 0, 3, 3000)
    assert len(Xa) > 0
    monkeypatch.setattr(G, "FAST_EVAL", False)
    Xb, nb = G.screen_chunk(E, tr, 0, 3, 3000)
    assert np.array_equal(Xa, Xb) and np.array_equal(na, nb)


def _decode_np_clip(G, x):
    """옛 decode(스칼라 np.clip) — 가속 판과 대조용."""
    out = {}
    for i, (name, lo, hi, off) in enumerate(G.CONT):
        u = float(np.clip(x[i], 0.0, 1.0))
        if (off == "lo" and u <= G.OFF_ZONE) or (off == "hi" and u >= 1.0 - G.OFF_ZONE):
            out[name] = None
        else:
            out[name] = round(lo + u * (hi - lo), 4)
    j = len(G.CONT)
    for name, levels in G.NOM:
        out[name] = levels[int(np.argmax(x[j:j + len(levels)]))]
        j += len(levels)
    for q, (name, levels) in enumerate(G.DISC):
        out[name] = levels[int(np.clip(round(float(x[G.N_CONT + q])), 0, len(levels) - 1))]
    return G.canonical(out)


def test_decode_matches_np_clip_version():
    """decode(min·max) = 옛 decode(np.clip) — 범위 밖·−0·경계·반올림 동점 근처 포함, 값과 타입 모두."""
    G = __import__("pattern_ga_run")
    rng = np.random.default_rng(3)
    X = G.random_x(rng, 3000)
    X[:1000, :G.N_CONT] = rng.uniform(-0.4, 1.4, (1000, G.N_CONT))
    X[:1000, G.N_CONT:] += rng.uniform(-3, 3, (1000, len(G.DISC)))
    X[1000:1100, :G.N_CONT] = rng.choice([-0.0, 0.0, 1.0, G.OFF_ZONE, 1 - G.OFF_ZONE, 0.5], (100, G.N_CONT))
    X[1100:1200, G.N_CONT:] = rng.integers(0, 4, (100, len(G.DISC))) + 0.5
    for x in X:
        a, b = _decode_np_clip(G, x), G.decode(x)
        assert repr(a) == repr(b) and [type(v) for v in a.values()] == [type(v) for v in b.values()]
