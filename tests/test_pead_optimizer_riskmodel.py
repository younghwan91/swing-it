"""6단계 모델 빌드 — 캐시는 입력에 묶이고, 옮긴 함수가 옮기기 전과 같은 값을 낸다. core 위험모델 층이 필요하다(``research`` 잡, 건너뛰면 실패)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("krx_quant_core.risk.factor_model")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_fixtures import SECTORS  # noqa: E402

from swing_it.book.riskmodel import as_risk_model, build_models, model_key  # noqa: E402


def _panels(seed=0, nN=120, nD=420):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 0.02, (nN, nD))
    cap = np.exp(rng.normal(27, 1, (nN, 1))) * np.ones((nN, nD))
    lab = np.array([SECTORS[i % 3] for i in range(nN)], dtype=object)[:, None].repeat(nD, axis=1)
    Xs = {"size": rng.normal(0, 1, (nN, nD)), "momentum": rng.normal(0, 1, (nN, nD))}
    return {"R": R, "cap": cap, "lab": lab, "Xs": Xs, "msk": np.ones((nN, nD), bool)}


NEED = [400, 410]


def test_cache_returns_the_same_models_without_rebuilding(tmp_path, monkeypatch):
    log = []
    c = _panels()
    a = build_models(c, NEED, 1, tmp_path / "m.pkl", log.append, shrink=0.1)
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    b = build_models(c, NEED, 1, tmp_path / "m.pkl", log.append, shrink=0.1)
    for t in NEED:
        np.testing.assert_array_equal(a[t][3], b[t][3])
    assert len(log) == 1, "두 번째는 읽기만 한다"


@pytest.mark.parametrize("change", ["shrink", "vra", "need", "returns", "lab"])
def test_a_changed_input_does_not_read_the_old_cache(tmp_path, change):
    """PO-3 은 경로만 보고 설정을 무시했다 — 다른 shrink 로 불러도 옛 모델을 돌려줬다."""
    c = _panels()
    kw = {"shrink": 0.1, "vra": None}
    need = NEED
    base = model_key(c, need, **kw)
    if change == "shrink":
        kw["shrink"] = 0.3
    elif change == "vra":
        kw["vra"] = 21
    elif change == "need":
        need = [400]
    elif change == "returns":
        c = {**c, "R": c["R"].copy()}
        c["R"][0, 0] += 1e-9
    else:
        c = {**c, "lab": c["lab"].copy()}
        c["lab"][0, 0] = SECTORS[1]
    assert model_key(c, need, **kw) != base


def test_different_shrink_really_gives_different_models_through_the_cache(tmp_path):
    c = _panels()
    a = build_models(c, NEED, 1, tmp_path / "m.pkl", lambda m: None, shrink=0.1)
    b = build_models(c, NEED, 1, tmp_path / "m.pkl", lambda m: None, shrink=0.9)
    assert not np.array_equal(a[400][3], b[400][3])


def test_cache_key_ignores_unrelated_entries():
    c = _panels()
    assert model_key({**c, "dates": [1, 2], "raw": {"x": 1}}, NEED, shrink=0.1) == model_key(c, NEED, shrink=0.1)


def test_as_risk_model_keeps_the_coordinates(tmp_path):
    c = _panels()
    models = build_models(c, NEED, 1, tmp_path / "m.pkl", lambda m: None, shrink=0.1)
    m, idx = as_risk_model(models, 400)
    np.testing.assert_array_equal(m.codes, idx)
    np.testing.assert_array_equal(m.D, models[400][3])
    assert m.names == models[400][4]


def test_old_cache_with_the_old_name_is_not_read(tmp_path):
    c = _panels()
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps({400: "옛 모델"}))
    got = build_models(c, NEED, 1, path, lambda m: None, shrink=0.1)
    assert got[400] != "옛 모델"
