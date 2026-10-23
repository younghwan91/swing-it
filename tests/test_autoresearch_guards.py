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
            "lu": np.full(t, 1.29), "regime_on": np.ones(t, bool), "kx": np.linspace(1000, 1100, t), "kq": np.linspace(500, 520, t),
            "era": "T", "tv": C, "adv": C, "cap": C, "eps_yoy": C * 0}


def _sig(D, score=None):
    return {"cand": np.ones_like(D["C"], bool), "score": np.ones_like(D["C"]) if score is None else score}


def test_clamp_cfg_cuts_narrow_stop_tight_exit_and_unknown_keys():
    c = P.clamp_cfg({"hard": 0.03, "risk": 0.5, "max_pos": 100, "stop": "struct", "rank_score": "x", "time_stop": 5,
                     "exit_kind": "low", "exit_w": 3, "entry": "close"})
    assert c["hard"] == 0.10 and c["risk"] == 0.01 and c["max_pos"] == 20 and c["exit_w"] == 20
    assert not {"stop", "rank_score", "time_stop", "entry"} & set(c)


def test_clamp_cfg_rejects_choice_outside_menu():
    with pytest.raises(ValueError):
        P.clamp_cfg({"exit_kind": "trail"})


def test_eps_guard_cannot_be_bypassed_by_cand():
    D = _toy()
    D["eps_up"][1] = False                       # 종목 1 은 EPS 상승 아님
    sig = _sig(D, np.tile([[1.0], [9.0], [2.0]], (1, D["C"].shape[1])))
    _, tr = P.simulate(D, sig, {"max_pos": 5}, str(D["days"][1].date()), str(D["days"][-1].date()))
    assert len(tr) and "000001" not in set(tr["code"])   # 순위가 가장 높아도 못 산다


def test_uni_guard_cannot_be_bypassed_by_cand():
    D = _toy()
    D["uni"][:] = False
    _, tr = P.simulate(D, _sig(D), {}, str(D["days"][1].date()), str(D["days"][-1].date()))
    assert len(tr) == 0


@pytest.mark.parametrize("src", [
    "import os\n", "import pickle\n", "from prepare import COST_MULT\n", "x = open('a')\n", "y = getattr(1, 'real')\n",
    "import numpy as np\nnp.load('x.npy')\n", "import pandas as pd\npd.read_pickle('x')\n", "a = (1).__class__\n",
    "BAD = ['005930']\n", "Y = 2022\n", "S = '2020-03'\n", "def f(V):\n    return V['days'].year\n",
])
def test_lint_rejects_escape_and_hindsight(src):
    assert P.lint_train(src)


def test_lint_accepts_baseline():
    assert P.lint_train(open(P.TRAIN, encoding="utf-8").read()) == []


def test_causality_catches_lookahead(tmp_path):
    D = _toy(t=120)
    rng = np.random.default_rng(0)
    D["C"] = np.cumprod(1 + rng.normal(0, 0.02, D["C"].shape), axis=1)
    peek = tmp_path / "peek.py"
    peek.write_text("import numpy as np\nimport pandas as pd\n"
                    "def signals(V, x):\n    f = pd.DataFrame(V['C'].T).shift(-1).to_numpy().T\n"
                    "    return {'cand': f > V['C'], 'score': f}\n"
                    "def get_strategy():\n    return {'params': {}, 'signals': signals, 'cfg': lambda x: {}}\n")
    ok, _ = P.causality({"A": D, "B": D}, {"folds": [{"era": "A", "params": {}}, {"era": "B", "params": {}}]}, str(peek))
    assert not ok
    honest = tmp_path / "honest.py"
    honest.write_text(peek.read_text().replace("shift(-1)", "shift(1)"))
    ok, _ = P.causality({"A": D, "B": D}, {"folds": [{"era": "A", "params": {}}, {"era": "B", "params": {}}]}, str(honest))
    assert ok


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
    folds = [{"era": "A", "test": ["2011-01-01", "2012-01-01"], "alpha": 0.05, "excess": 0.05, "years": 1.0, "n": 3} for _ in range(10)]
    s = P.summarize({"folds": folds, "trades": []})
    assert s["void"] and s["score"] == -1.0


def test_beats_requires_paired_fold_improvement(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BEST", str(tmp_path / "BEST.json"))
    best = [{"alpha": 0.0} for _ in range(10)]
    (tmp_path / "BEST.json").write_text(pd.Series({"score": 0.0, "top5": 0.0, "folds": best}).to_json())
    one_big = [{"alpha": 0.5}] + [{"alpha": -0.001} for _ in range(9)]   # 한 폴드만 크게 좋아짐(잡음 겨냥)
    ok, _ = P.beats({"void": False, "score": 0.02}, one_big, 0.0)
    assert not ok
    broad = [{"alpha": 0.01} for _ in range(10)]
    ok, _ = P.beats({"void": False, "score": 0.01}, broad, 0.0)
    assert ok
