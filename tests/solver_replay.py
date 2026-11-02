"""솔버 응답 녹화·재생 — O 북 골든이 솔버 판과 무관하게 솔버 **밖** 코드만 재게 한다.

``krx_quant_core.portfolio.optimize.optimize`` 자리에 넣는다. 호출 순서대로 녹화한 응답을 돌려주고, 호출마다 입력 요약을 녹화값과 1e-9 로 맞댄다.
해시 키로 찾지 않는다 — 끝자리가 달라도 맞아야 하고, 순서가 틀어지면 찾지 못한 것이 아니라 어긋난 것이어야 한다.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np

SUMMARY = ("lam", "n", "alpha_sum", "alpha_sq", "w_prev_sum", "budget")
TOL = 1e-9


def input_summary(alpha, w_prev, kw: dict) -> tuple:
    """녹화·재생이 맞대는 입력 요약: λ, 종목 수, 선형항 합·제곱합, 직전 비중 합, 예산."""
    alpha, w_prev = np.asarray(alpha, dtype=np.float64), np.asarray(w_prev, dtype=np.float64)
    return (float(kw["risk_aversion"]), int(alpha.size), float(alpha.sum()), float((alpha * alpha).sum()), float(w_prev.sum()),
            float(kw["constraints"].budget))


def recorder(real, path: Path):
    """실제 솔버를 감싸 호출을 ``path`` 에 덧붙인다. 파일에 적으므로 포크한 널 작업자 안에서도 녹화된다."""

    def optimize(alpha, model, w_prev, **kw):
        res = real(alpha, model, w_prev, **kw)
        with open(path, "ab") as fh:
            pickle.dump((input_summary(alpha, w_prev, kw), np.asarray(res.weights, dtype=np.float64), str(res.status), float(res.est_cost)), fh)
        return res

    return optimize


def read_records(path: Path) -> list:
    out = []
    with open(path, "rb") as fh:
        while True:
            try:
                out.append(pickle.load(fh))
            except EOFError:
                return out


def to_arrays(key: str, records: list) -> dict:
    """npz 한 스트림. 비중은 호출마다 길이가 달라 이어 붙이고 끝 위치를 적는다."""
    return {f"{key}.summary": np.array([r[0] for r in records], dtype=np.float64),
            f"{key}.weights": np.concatenate([r[1] for r in records]),
            f"{key}.ends": np.cumsum([r[1].size for r in records]),
            f"{key}.status": np.array([r[2] for r in records]),
            f"{key}.est_cost": np.array([r[3] for r in records], dtype=np.float64)}


class Replay:
    """녹화 스트림 하나를 순서대로 돌려준다. 호출 수는 공유 카운터라 포크한 작업자가 쓴 것도 부모에서 보인다(널은 ``workers=1``)."""

    def __init__(self, data, key: str, result_cls):
        self.key = key
        self.summary = data[f"{key}.summary"]
        self.weights = data[f"{key}.weights"]
        self.ends = data[f"{key}.ends"]
        self.status = data[f"{key}.status"]
        self.est_cost = data[f"{key}.est_cost"]
        self.result_cls = result_cls
        self.calls = mp.get_context("fork").Value("i", 0)

    def __call__(self, alpha, model, w_prev, **kw):
        with self.calls.get_lock():
            i = self.calls.value
            self.calls.value += 1
        assert i < len(self.summary), f"{self.key}: 녹화보다 호출이 많다({i + 1} > {len(self.summary)})"
        got = input_summary(alpha, w_prev, kw)
        for name, g, w in zip(SUMMARY, got, self.summary[i]):
            assert abs(g - w) <= TOL, f"{self.key} 호출 {i}: 입력 {name} {g} != 녹화 {w}"
        start = int(self.ends[i - 1]) if i else 0
        nan = float("nan")
        return self.result_cls(weights=self.weights[start: int(self.ends[i])].copy(), expected_alpha=nan, expected_risk=nan,
                               est_cost=float(self.est_cost[i]), turnover=nan, status=str(self.status[i]))

    def assert_used_up(self) -> None:
        assert self.calls.value == len(self.summary), f"{self.key}: 호출 {self.calls.value}번, 녹화 {len(self.summary)}번"
