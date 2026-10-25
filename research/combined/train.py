"""세 레포 종합 autoresearch 실험 파일 — 에이전트가 고치는 유일한 파일(program.md). 실행은 평가기: ``uv run python research/combined/prepare.py``.

기준선 = leader_C 구조(추세 템플릿 · N일 돌파 · RS 백분위 ≥ θ · MA 청산 · 하드 손절 · 불타기 · 국면). 진입은 **신호일 종가**(평가기가 강제 — 여기서 정하지 않는다).
숫자 파라미터는 ``params`` 범위만 정하고 값은 평가기가 폴드마다 학습 구간에서 고른다 — 여기서 숫자를 손으로 맞추지 않는다.
이 기준선은 갭·뉴스 키를 쓰지 않으므로 후보 자격(eligible)이 없다 — 정상이다. 세 레포 부품을 실제로 쓰는 구조로 바꾸는 것이 루프의 일이다.

계약(prepare.load_strategy — AST 허용 목록: numpy·pandas·math·functools·itertools·collections 만, 파일·OS·getattr·더블 언더스코어·연도·종목코드 리터럴 금지):
  get_strategy() → {"params": {이름: ("int"|"float", lo, hi) | ("choice", (값, ...))},
                    "signals": f(V, x) → {"cand": bool[N,T], "score": float[N,T]},
                    "cfg": f(x) → {max_pos 5~20, hard 0.10~0.30, risk 0.002~0.01, pos_cap 0.05~0.20, pyramid 0/1, regime 0/1,
                                   exit_kind "ma"|"low", exit_w (ma 20~250, low 20~120)}}
  V = 탐색 절반 종목 × 날짜 배열 + 날짜 배열 kx·kq(KOSPI·KOSDAQ 종가)·regime_on. 날짜·종목코드는 없다.
    swing: O·H·L·C·tv·adv·cap·eps_yoy·uni·TR
    scalp: gap(t 시가 / t-1 종가 − 1, 수정가, 재개일 NaN) · tickp(t-1 원주가 호가 / 원주가, 주문일 호가표) · gap_event(|gap| ≥ 제한폭 절반, 대칭)
    daytrade: n_all·n_feat([특징주])·n_move(급등·상한가 등)·n_excl([단독]) = t-1 15:10 < 기사 ≤ t 15:10 개수 ·
              news_known(아카이브가 그 종목·날을 안다 — False 면 0 은 "모름"이지 "기사 없음"이 아니다; 아카이브 시작 전은 전부 False)
  cand 는 평가기가 uni ∧ TR ∧ EPS 상승과 AND 한다. score 가 클수록 빈 슬롯에 먼저 산다. 신호는 t 종가까지의 값만(평가기가 절단 재계산으로 검사).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_C: dict = {}


def _rmean(a: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(a.T).rolling(w, min_periods=w).mean().to_numpy().T


def _rmax(a: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(a.T).rolling(w, min_periods=w).max().to_numpy().T


def _rmin(a: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(a.T).rolling(w, min_periods=w).min().to_numpy().T


def _base(V: dict) -> dict:
    if "base" not in _C:
        C, H, L = V["C"], V["H"], V["L"]
        ma = {w: _rmean(C, w) for w in (50, 150, 200)}
        ma200_prev = np.concatenate([np.full((C.shape[0], 20), np.nan), ma[200][:, :-20]], axis=1)
        hi252, lo252 = _rmax(H, 252), _rmin(L, 252)

        def ret(k):
            return C / np.concatenate([np.full((C.shape[0], k), np.nan), C[:, :-k]], axis=1) - 1.0
        raw = 2 * ret(63) + ret(126) + ret(189) + ret(252)
        rs = pd.DataFrame(np.where(V["uni"] & np.isfinite(raw), raw, np.nan).T).rank(axis=1, pct=True).to_numpy().T * 100
        with np.errstate(invalid="ignore"):
            tt = (C > ma[50]) & (ma[50] > ma[150]) & (ma[150] > ma[200]) & (ma[200] > ma200_prev) & (C >= 0.75 * hi252) & (C >= 1.30 * lo252)
        _C["base"] = {"rs": rs, "tt": tt}
    return _C["base"]


def _brk(V: dict, n: int) -> np.ndarray:
    if ("brk", n) not in _C:
        _C[("brk", n)] = V["C"] >= _rmax(V["C"], n) - 1e-9
    return _C[("brk", n)]


def signals(V: dict, x: dict) -> dict:
    B = _base(V)
    with np.errstate(invalid="ignore"):
        cand = B["tt"] & _brk(V, x["N"]) & (B["rs"] >= x["theta"])
    return {"cand": cand, "score": B["rs"]}


def cfg(x: dict) -> dict:
    return {"max_pos": 10, "hard": x["hard"], "risk": 0.005, "pos_cap": 0.10, "pyramid": 1, "regime": 1,
            "exit_kind": x["exit_kind"], "exit_w": x["ma_exit"] if x["exit_kind"] == "ma" else x["low_w"]}


def get_strategy() -> dict:
    return {"params": {"theta": ("int", 60, 95), "N": ("choice", (20, 50, 120)), "ma_exit": ("int", 50, 200),
                       "hard": ("float", 0.10, 0.25), "exit_kind": ("choice", ("ma", "low")), "low_w": ("int", 20, 120)},
            "signals": signals, "cfg": cfg}
