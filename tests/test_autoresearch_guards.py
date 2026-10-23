"""autoresearch 평가기 가드 — 위반을 주입해 실제로 막히는지 본다(GUARDRAILS §4.5: 통과만 보는 테스트는 규칙이 죽어도 초록)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.autoresearch import prepare as P


def _toy(n: int = 3, t: int = 60) -> dict:
    days = pd.bdate_range("2012-01-02", periods=t)
    C = np.tile(np.linspace(100, 160, t), (n, 1))
    ones = np.ones((n, t), bool)
    return {"days": days, "codes": [f"{i:06d}" for i in range(n)], "O": C.copy(), "H": C * 1.01, "L": C * 0.99, "C": C, "TR": ones,
            "uni": ones.copy(), "eps_up": ones.copy(), "hi20": ones.copy(), "cost": np.full((2, t), 0.004), "mkt": np.zeros(n, np.int64),
            "lu": np.full(t, 1.29), "regime_on": np.ones(t, bool), "kx": np.linspace(1000, 1100, t)}


def test_clamp_cfg_cuts_narrow_stop_and_unknown_keys():
    c = P.clamp_cfg({"hard": 0.03, "risk": 0.5, "max_pos": 100, "stop": "struct", "rank_score": "x", "time_stop": 5})
    assert c["hard"] == 0.10 and c["risk"] == 0.01 and c["max_pos"] == 20
    assert not {"stop", "rank_score", "time_stop"} & set(c)


def test_clamp_cfg_rejects_choice_outside_menu():
    with pytest.raises(ValueError):
        P.clamp_cfg({"entry": "vwap"})


def test_eps_guard_cannot_be_bypassed_by_cand():
    D = _toy()
    D["eps_up"][1] = False                       # 종목 1 은 EPS 상승 아님
    sig = {"cand": np.ones_like(D["C"], bool), "score": np.tile([[1.0], [9.0], [2.0]], (1, D["C"].shape[1])),
           "exit_line": np.zeros_like(D["C"])}
    _, tr = P.simulate(D, sig, {"max_pos": 5}, str(D["days"][1].date()), str(D["days"][-1].date()))
    assert len(tr) and "000001" not in set(tr["code"])   # 순위가 가장 높아도 못 산다


def test_uni_guard_cannot_be_bypassed_by_cand():
    D = _toy()
    D["uni"][:] = False
    sig = {"cand": np.ones_like(D["C"], bool), "score": np.ones_like(D["C"]), "exit_line": np.zeros_like(D["C"])}
    _, tr = P.simulate(D, sig, {}, str(D["days"][1].date()), str(D["days"][-1].date()))
    assert len(tr) == 0


def test_plateau_pick_prefers_plateau_over_spike():
    rng = np.random.default_rng(0)
    U = rng.random((60, 2))
    y = np.where(U[:, 0] > 0.6, 0.10, -0.10)     # 넓은 고원
    U = np.vstack([U, [[0.05, 0.05]]])
    y = np.append(y, 1.0)                        # 외딴 한 점 극값
    assert U[P.plateau_pick(U, y), 0] > 0.6


def test_half_split_is_stable_and_balanced():
    codes = [f"{i:06d}" for i in range(4000)]
    h = np.array([P.half_of(c) for c in codes])
    assert 0.45 < h.mean() < 0.55
    assert [P.half_of(c) for c in codes[:50]] == list(h[:50])


def test_summarize_voids_thin_trading():
    folds = [{"era": "A", "test": ["2011-01-01", "2012-01-01"], "excess": 0.05, "years": 1.0, "n": 3} for _ in range(10)]
    s = P.summarize({"folds": folds, "trades": []})
    assert s["void"] and s["score"] == -1.0
