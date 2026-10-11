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
