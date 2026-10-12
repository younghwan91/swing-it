"""25세대 코어-위성(META_EVOLUTION.md 25세대 — 결과 전 고정): 위성 = 23세대 C(신호일 종가 진입) 그대로,
코어 = 국면 켬(시총 가중 시장 ≥ 200일선, 전날 판정)일 때 위성이 안 쓰는 자산(1 − 위성 노출, 전날 종가)을 시총 상위 200 시총 가중 지수(KOSPI200 대용)에.
코어 비중 변화량에 왕복 0.1%(ETF 수수료·스프레드 근사) 비용. 국면 꺼짐 = 전부 현금(위성도 국면으로 이미 청산).

비교: 시총 가중 시장(전체, 단순 보유) · 상위 200 시총 가중(단순 보유) · 국면 타이밍 상위 200(전액) · 위성 C 단독.
사용: ``uv run --with numba --extra pg python research/experiments/core25.py --cache <v13> --out <dir>`` (KR_QUANT_DB)
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

SAT = {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "univ": "top200", "regime": 1, "entry": "close"}
CORE_COST = 0.001
WINS = {"design_2017_2023": ("2017-10-02", "2024-01-01"), "contaminated_2024_2026": ("2024-01-01", "2026-10-09")}


def top200_cw(D: dict) -> np.ndarray:
    """그날 시총 순위 ≤ 200(전날 기준) 종목의 전날 시총 가중 일간 수익."""
    C = D["C"]
    r = np.concatenate([np.full((C.shape[0], 1), np.nan), C[:, 1:] / C[:, :-1] - 1], axis=1)
    rk = np.concatenate([np.full((C.shape[0], 1), np.nan), D["crank"][:, :-1]], axis=1)
    cap_prev = np.concatenate([np.full((C.shape[0], 1), np.nan), D["cap"][:, :-1]], axis=1)
    w = np.where(np.isfinite(r) & (np.nan_to_num(rk, nan=1e9) <= 200) & np.isfinite(cap_prev), cap_prev, 0.0)
    out = np.where(w.sum(0) > 0, (np.nan_to_num(np.clip(r, -0.3, 0.3)) * w).sum(0) / np.maximum(w.sum(0), 1e-12), 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = L.prepare(a.cache)
    days = D["days"]
    idx_r = pd.Series(top200_cw(D), index=days)
    reg = pd.Series(D["regime_on"].astype(float), index=days)
    out, curves = {}, {}
    for cm in (1.0, 2.0):
        eq, _ = L.simulate(D, SAT, cm, end="2026-10-08")
        expo = pd.Series(eq.attrs["expo"], index=eq.index)
        sat_r = eq.pct_change().fillna(0)
        core_w = ((1 - expo.shift(1)) * reg.reindex(eq.index).shift(1)).fillna(0).clip(0, 1)
        core_r = core_w * idx_r.reindex(eq.index).fillna(0) - core_w.diff().abs().fillna(0) * CORE_COST * cm
        cs = (1 + sat_r + core_r).cumprod()
        curves[f"core_sat_{int(cm)}x"] = cs
        curves[f"sat_only_{int(cm)}x"] = eq / eq.iloc[0]
        curves[f"expo_total_{int(cm)}x"] = expo.shift(1).fillna(0) + core_w
    idx_eq = (1 + idx_r).cumprod()
    B = L.benchmarks(D)
    curves["top200_cw_bh"] = idx_eq.reindex(curves["core_sat_2x"].index)
    curves["top200_cw_regime"] = (1 + idx_r * reg.shift(1).fillna(0)).cumprod().reindex(curves["core_sat_2x"].index)
    curves["market_cw_bh"] = B["market_cw"].reindex(curves["core_sat_2x"].index)
    Cv = pd.DataFrame(curves)
    os.makedirs(a.out, exist_ok=True)
    Cv.to_csv(os.path.join(a.out, "curves.csv"))
    for wn, (lo, hi) in WINS.items():
        x = Cv[(Cv.index >= lo) & (Cv.index < hi)]
        out[wn] = {k: L.metrics(x[k]) for k in x.columns if not k.startswith("expo")}
        out[wn]["avg_total_exposure_2x"] = float(x["expo_total_2x"].mean())
        # 월 알파 vs 상위 200 단순 보유
        m = x.resample("ME").last().pct_change().dropna()
        y, xx = m["core_sat_2x"], m["top200_cw_bh"]
        b = np.polyfit(xx, y, 1)
        res = y - (b[0] * xx + b[1])
        se = res.std(ddof=2) / np.sqrt(len(y)) if len(y) > 2 else np.nan
        out[wn]["alpha_m_vs_top200"] = {"beta": float(b[0]), "alpha_m": float(b[1]), "t": float(b[1] / se) if se else np.nan}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for wn, v in out.items():
        print("==", wn, "avg expo", round(v["avg_total_exposure_2x"], 3), "alpha", {k: round(x, 4) for k, x in v["alpha_m_vs_top200"].items()})
        for k, m in v.items():
            if isinstance(m, dict) and "cagr" in m:
                print(f"  {k:20s} cagr {m['cagr']:+.3f} mdd {m['mdd']:+.3f} sharpe {m['sharpe_m']:.2f}")


if __name__ == "__main__":
    main()
