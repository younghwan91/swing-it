"""23세대 실행 개선(META_EVOLUTION.md 23세대 — 결과 전 고정): leader_A(21세대 격자 215) 골격 고정, 진입 방식만 바꾼 짝 비교.

팔: A = 다음날 시가(기준) · B = 다음날 시가 ∧ 갭 ≤ +3% · C = 신호일 종가(종가 단일가) · D = C ∧ (다음날 판단 없음 — C 와 같다, 제외).
평가 = 2017-10 ~ 2023-12 전체(같은 데이터 위 짝 비교 — 설정 선택 없음), 연복리·낙폭·연도별 짝 차이, 상위 5 종목 제거, 노출.

사용: ``uv run --with numba --extra pg python research/experiments/leader23.py --cache <v13> --out <dir>`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

import research.experiments.leader21 as L  # noqa: E402

BASE = {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "univ": "top200", "regime": 1}
ARMS = {"A_open": {}, "B_open_gap3": {"gap_max": 0.03}, "C_close": {"entry": "close"}}
WIN = ("2017-10-02", "2024-01-01")


def drop_top(eq: pd.Series, tr: pd.DataFrame, k: int) -> float:
    """상위 k 종목(순손익 기여) 거래를 '안 산 것' 으로 근사: 그 거래들의 기여를 뺀 연복리(가산 근사)."""
    if tr.empty:
        return float("nan")
    contrib = (tr["ret"] * tr["w_peak"]).groupby(tr["code"]).sum().sort_values(ascending=False)
    top = set(contrib.index[:k])
    x = eq[(eq.index >= WIN[0]) & (eq.index < WIN[1])]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    total = x.iloc[-1] / x.iloc[0] - 1 - float(contrib[contrib.index.isin(top)].sum())
    return float((1 + total) ** (1 / yrs) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = L.prepare(a.cache)
    out, curves = {}, {}
    for nm, ov in ARMS.items():
        for cm in (1.0, 2.0):
            eq, tr = L.simulate(D, {**BASE, **ov}, cm)
            x = eq[(eq.index >= WIN[0]) & (eq.index < WIN[1])]
            m = L.metrics(x)
            trw = tr[(tr["entry"] >= WIN[0]) & (tr["entry"] < WIN[1])]
            m.update(n_trades=int(len(trw)), drop_top5_cagr=drop_top(x, trw, 5), win=float((trw["ret"] > 0).mean()) if len(trw) else np.nan)
            out[f"{nm}_{int(cm)}x"] = m
            curves[f"{nm}_{int(cm)}x"] = x
            tr.to_csv(os.path.join(a.out, f"trades_{nm}_{int(cm)}x.csv.gz"), index=False) if os.makedirs(a.out, exist_ok=True) is None else None
    B = L.benchmarks(D)
    Bw = B[(B.index >= WIN[0]) & (B.index < WIN[1])]
    out["bench"] = {k: L.metrics(Bw[k]) for k in Bw.columns}
    pd.DataFrame(curves).to_csv(os.path.join(a.out, "curves.csv"))
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for k, v in out.items():
        if k != "bench":
            print(k, round(v["cagr"], 4), round(v["mdd"], 4), "top5-", round(v["drop_top5_cagr"], 4), {y: round(r, 3) for y, r in v["by_year"].items()})
    for k, v in out["bench"].items():
        print("bench", k, round(v["cagr"], 4), round(v["mdd"], 4))


if __name__ == "__main__":
    main()
