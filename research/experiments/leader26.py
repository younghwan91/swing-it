"""26세대(META_EVOLUTION.md 26세대 — 결과 전 고정): 국면 필터의 역할 — 전량 청산 vs 신규 매수만 막기 vs 약한 것만 매도.

골격 = C(신호일 종가 진입, leader_C). 팔: liquidate(기준) · gate · weak. 지표: 2017-10 ~ 2023 연복리·낙폭·연도 짝,
**대장주 포착률**(해마다 연초 시총 상위 200 중 그해 상승률 상위 30 종목의 로그 상승분 중 보유 중이던 몫, 평균) — 연도별 30 관측.
2024-01 ~ 2026-10 은 오염(이 가설은 그 구간을 포함한 진단에서 나왔다) — 기술.

사용: ``uv run --with numba --extra pg python research/experiments/leader26.py --cache <v13> --out <dir>`` (KR_QUANT_DB)
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

BASE = {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "univ": "top200", "regime": 1, "entry": "close"}
ARMS = {"liquidate": {}, "gate": {"regime_mode": "gate"}, "weak": {"regime_mode": "weak"}}
WINS = {"design": ("2017-10-02", "2024-01-01"), "contaminated": ("2024-01-01", "2026-10-09")}


def capture(D, tr: pd.DataFrame, years) -> dict:
    days, C, codes = D["days"], D["C"], D["codes"]
    held = np.zeros(C.shape, bool)
    for r in tr.itertuples():
        i = codes.index(r.code)
        held[i, days.get_loc(pd.Timestamp(r.entry)):days.get_loc(pd.Timestamp(r.exit))] = True
    rr = np.concatenate([np.full((C.shape[0], 1), np.nan), C[:, 1:] / C[:, :-1] - 1], axis=1)
    out = {}
    for y in years:
        m = np.asarray(days.year == y)
        t0, t1 = np.flatnonzero(m)[0], np.flatnonzero(m)[-1]
        ok = np.nan_to_num(D["crank"][:, t0], nan=1e9) <= 200
        yr = C[:, t1] / C[:, t0] - 1
        top = np.argsort(-np.where(ok & np.isfinite(yr), yr, -9))[:30]
        lr = np.log1p(np.nan_to_num(rr[:, m]))
        out[y] = [float((lr[i] * held[i, m]).sum() / max(lr[i].sum(), 1e-9)) for i in top]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = L.prepare(a.cache)
    res, curves = {}, {}
    for nm, ov in ARMS.items():
        eq, tr = L.simulate(D, {**BASE, **ov}, 2.0, end="2026-10-08", paper=True)
        curves[nm] = eq / eq.iloc[0]
        cap = capture(D, tr, range(2018, 2027))
        r = {"capture": {y: float(np.mean(v)) for y, v in cap.items()}, "capture_raw": cap}
        for wn, (lo, hi) in WINS.items():
            x = eq[(eq.index >= lo) & (eq.index < hi)]
            r[wn] = L.metrics(x)
        r["n_regime_exits"] = int((tr["why"] == "regime").sum())
        res[nm] = r
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(curves).to_csv(os.path.join(a.out, "curves.csv"))
    # 짝 비교: 포착률 대장주 단위 차이(설계 구간 2018~2023)
    for nm in ("gate", "weak"):
        d = []
        for y in range(2018, 2024):
            d += list(np.array(res[nm]["capture_raw"][y]) - np.array(res["liquidate"]["capture_raw"][y]))
        d = np.array(d)
        res[nm]["capture_diff_design"] = {"mean": float(d.mean()), "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), "n": len(d)}
        yd = {y: res[nm]["design"]["by_year"].get(y, 0) - res["liquidate"]["design"]["by_year"].get(y, 0) for y in res["liquidate"]["design"]["by_year"]}
        res[nm]["year_diff_design"] = yd
        res[nm]["pass"] = bool(res[nm]["design"]["cagr"] >= res["liquidate"]["design"]["cagr"] + 0.02
                               and res[nm]["design"]["mdd"] >= res["liquidate"]["design"]["mdd"] - 0.08
                               and res[nm]["capture_diff_design"]["t"] >= 2
                               and sum(v > 0 for v in yd.values()) >= 4)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for nm, r in res.items():
        print(nm, "design", round(r["design"]["cagr"], 4), round(r["design"]["mdd"], 4), "contam", round(r["contaminated"]["cagr"], 4),
              round(r["contaminated"]["mdd"], 4), "regime exits", r["n_regime_exits"], "capture", {y: round(v, 3) for y, v in r["capture"].items()},
              r.get("capture_diff_design"), r.get("pass"))


if __name__ == "__main__":
    main()
