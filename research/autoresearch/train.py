"""autoresearch 실험 파일 — 에이전트가 고치는 유일한 파일(program.md).

기준선 = 23세대 leader_C 뼈대(추세 템플릿 · N일 돌파 · RS 백분위 ≥ θ · MA 청산 · 하드 손절 · 종가 진입 · 불타기 · 국면).
숫자 파라미터는 ``params`` 범위만 정하고 값은 prepare 가 폴드마다 학습 구간에서 고른다 — 여기서 숫자를 손으로 맞추지 않는다.

계약(prepare.load_strategy):
  get_strategy() → {"params": {이름: ("int"|"float", lo, hi) | ("choice", (값, ...))},
                    "signals": f(V, x) → {"cand": bool[N,T], "score": float[N,T], "exit_line": float[N,T]},
                    "cfg": f(x) → 엔진 설정(max_pos·hard·risk·pos_cap·entry·pyramid·regime 만, 범위는 prepare 가 자른다)}
  V = 탐색 절반 종목 × 날짜 배열(O·H·L·C·tv·adv·cap·eps_yoy·uni·TR, days·codes·kx·regime_on·era). 읽기 전용.
  cand 는 prepare 가 uni ∧ TR ∧ EPS 상승과 AND 한다. score 가 클수록 빈 슬롯에 먼저 산다. 종가 < exit_line 이면 다음 날 시가 청산.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_C: dict = {}


def _roll(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    return getattr(pd.DataFrame(a.T).rolling(w, min_periods=w), fn)().to_numpy().T


def _cached(V: dict, key, fn):
    k = (V["era"], len(V["codes"]), key)
    if k not in _C:
        _C[k] = fn()
    return _C[k]


def _base(V: dict) -> dict:
    def mk():
        C, H, L = V["C"], V["H"], V["L"]
        ma = {w: _roll(C, w, "mean") for w in (50, 100, 150, 200)}
        ma200_prev = np.concatenate([np.full((C.shape[0], 20), np.nan), ma[200][:, :-20]], axis=1)
        hi252, lo252 = _roll(H, 252, "max"), _roll(L, 252, "min")

        def ret(k):
            return C / np.concatenate([np.full((C.shape[0], k), np.nan), C[:, :-k]], axis=1) - 1.0
        raw = 2 * ret(63) + ret(126) + ret(189) + ret(252)
        rs = pd.DataFrame(np.where(V["uni"] & np.isfinite(raw), raw, np.nan).T).rank(axis=1, pct=True).to_numpy().T * 100
        with np.errstate(invalid="ignore"):
            tt = (C > ma[50]) & (ma[50] > ma[150]) & (ma[150] > ma[200]) & (ma[200] > ma200_prev) & (C >= 0.75 * hi252) & (C >= 1.30 * lo252)
        return {"ma": ma, "rs": rs, "tt": tt}
    return _cached(V, "base", mk)


def _brk(V: dict, n: int) -> np.ndarray:
    return _cached(V, ("brk", n), lambda: V["C"] >= _roll(V["C"], n, "max") - 1e-9)


def signals(V: dict, x: dict) -> dict:
    B = _base(V)
    with np.errstate(invalid="ignore"):
        cand = B["tt"] & _brk(V, x["N"]) & (B["rs"] >= x["theta"])
    return {"cand": cand, "score": B["rs"], "exit_line": B["ma"][x["ma_exit"]]}


def cfg(x: dict) -> dict:
    return {"max_pos": 10, "hard": x["hard"], "risk": 0.005, "pos_cap": 0.10, "entry": "close", "pyramid": 1, "regime": 1}


def get_strategy() -> dict:
    return {"params": {"theta": ("int", 60, 95), "N": ("choice", (20, 50, 120)), "ma_exit": ("choice", (50, 100, 150, 200)),
                       "hard": ("float", 0.10, 0.25)},
            "signals": signals, "cfg": cfg}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import prepare
    prepare.evaluate(os.path.abspath(__file__))
