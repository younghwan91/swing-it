"""20세대 메타 진화(META_EVOLUTION.md 20세대 — 결과 전 고정): '규칙' 이 아니라 '배우는 방식' 을 진화시키고, 그 선택까지 해마다 바꾼다.

사용자: "시장은 유기체야 계속 바뀐다" · "하나의 규칙으론 안돼 계속 바뀌어야 해" · "진화만 하지 말고 진화의 방식을 진화시켜".

1층(규칙) = 18세대 격자 5,120 설정(미너비니 책 범위 진입·청산, 모집단 M0/tight). EPS 는 **새 as-of**(정정 G19-1).
2층(배우는 방식) P = 학습 창 W ∈ {12, 24, 36개월, 확장(2017~)} × 선택 기준 S ∈ {평균 최대, P(R ≥ 4) 최대} × 관망 A ∈ {켬, 끔} = 16 방식.
  매년 1월 첫 거래일 Y 에 P 가 과거로만(청산이 Y 전인 사건만 — 누수 정화) 설정 하나를 고른다. 학습 사건 < 50 × 창 연수면 그 설정 제외.
  관망 켬: 고른 설정의 학습 평균 ≤ 0 이면 그해 거래 안 함(0).
3층(방식 선택) = 해마다 Y 에 16 방식 중 **직전 2년(Y−2, Y−1) 표본 밖 평균**(사건 가중)이 가장 높은 방식으로 갈아탄다(동률 = 거래 많은 쪽).
  2층 표본 밖은 2018~ 부터 쌓이므로 3층은 2020~ 부터 작동.

**봉인 사용(사용자 승인 2026-10-12 "응, 돌려"):** 데이터 끝 2026-10-08. 2024-01 ~ 2026-10 = 3층 절차의 진짜 표본 밖(한 번).
지표 = 순 2배 비용 × min(1, 5%/손절폭), 사건 단위. 포트폴리오(10 슬롯, 청산일 실현)는 기술.

사용: ``PATTERN_GA_VARIANT=fwm PATTERN_GA_MENU="$(cat menu_g16s.json)" uv run --with numba --extra pg python research/experiments/meta20.py
--cache <v13> --out <dir> --workers 10`` (KR_QUANT_DB 필요 — EPS as-of)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

import research.experiments.adapt18 as A  # noqa: E402

SEED = 20261020
END = "2026-10-09"                                   # 캐시 끝 다음날(봉인 해제)
Y0, Y1 = 2018, 2026
WINDOWS = (12, 24, 36, 0)                            # 개월, 0 = 확장(2017-01~)
CRITERIA = ("mean", "pR4")
ABSTAIN = (1, 0)
META_K = 2                                           # 3층: 직전 2년 표본 밖
MIN_PER_YEAR = 50
NM = (Y1 - 2017 + 1) * 12                            # 진입 월 칸(2017-01 ~ 2026-12)


def _task(a: tuple) -> dict:
    i, cfg = a
    t = A.run_cfg(cfg, detail=True)
    days = A._W["days"]
    dd = days[t["d"].to_numpy()]
    mo = ((dd.year - 2017) * 12 + dd.month - 1).to_numpy()
    xr = t["x"].to_numpy()
    n2s, big = t["n2s"].to_numpy(), (t["R"].to_numpy() >= 4).astype(float)
    cut = {Y: int(days.searchsorted(pd.Timestamp(f"{Y}-01-01"))) for Y in range(Y0, Y1 + 1)}
    tr = np.zeros((Y1 - Y0 + 1, NM, 3))
    for k, Y in enumerate(range(Y0, Y1 + 1)):
        m = xr < cut[Y]
        np.add.at(tr[k, :, 0], mo[m], n2s[m])
        np.add.at(tr[k, :, 1], mo[m], 1.0)
        np.add.at(tr[k, :, 2], mo[m], big[m])
    oos = np.zeros((Y1 - 2017 + 1, 3))
    yi = dd.year.to_numpy() - 2017
    np.add.at(oos[:, 0], yi, n2s)
    np.add.at(oos[:, 1], yi, 1.0)
    np.add.at(oos[:, 2], yi, big)
    return {"i": i, "tr": tr.astype(np.float32), "oos": oos}


def choose(TR, k: int, Y: int, W: int, crit: str):
    """방식 (W, crit) 로 Y 년 초에 고른 설정 번호와 학습 평균(없으면 None)."""
    hi = (Y - 2017) * 12
    lo = 0 if W == 0 else max(0, hi - W)
    yrs = (hi - lo) / 12.0
    s = TR[:, k, lo:hi, :].sum(axis=1)                     # 설정 × (합, 수, 큰 승자)
    ok = s[:, 1] >= MIN_PER_YEAR * yrs
    if not ok.any():
        return None, None
    mean = np.where(ok, s[:, 0] / np.maximum(s[:, 1], 1), -np.inf)
    score = mean if crit == "mean" else np.where(ok, s[:, 2] / np.maximum(s[:, 1], 1), -np.inf)
    j = int(np.argmax(score))
    return j, float(mean[j])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    keys = list(A.MENU)
    cfgs = [dict(zip(keys, v, strict=True)) for v in itertools.product(*A.MENU.values())]
    if a.limit:
        cfgs = cfgs[:a.limit]
    with cf.ProcessPoolExecutor(a.workers, initializer=A._init, initargs=(a.cache, END, True)) as ex:
        res = sorted(ex.map(_task, list(enumerate(cfgs)), chunksize=8), key=lambda r: r["i"])
    TR = np.stack([r["tr"] for r in res])                  # 설정 × 컷 연도 × 진입 월 × 3
    OOS = np.stack([r["oos"] for r in res])                # 설정 × 진입 연도 × 3
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "agg.npz"), TR=TR, OOS=OOS)
    procs = list(itertools.product(WINDOWS, CRITERIA, ABSTAIN))
    # 2층: 방식마다 연도별 선택·표본 밖
    lay2 = {}
    for p in procs:
        W, crit, ab = p
        rows = {}
        for k, Y in enumerate(range(Y0, Y1 + 1)):
            j, trm = choose(TR, k, Y, W, crit)
            yi = Y - 2017
            if j is None or (ab and trm <= 0):
                rows[Y] = {"cfg": None, "sum": 0.0, "n": 0, "big": 0.0, "train_mean": trm}
            else:
                rows[Y] = {"cfg": j, "sum": float(OOS[j, yi, 0]), "n": int(OOS[j, yi, 1]), "big": float(OOS[j, yi, 2]), "train_mean": trm}
        lay2[p] = rows
    # 3층: 직전 2년 표본 밖 평균으로 방식 선택
    lay3 = {}
    for Y in range(Y0 + META_K, Y1 + 1):
        best, key = None, None
        for p in procs:
            s_ = sum(lay2[p][y]["sum"] for y in range(Y - META_K, Y))
            n_ = sum(lay2[p][y]["n"] for y in range(Y - META_K, Y))
            sc = (s_ / n_ if n_ else 0.0, n_)
            if key is None or sc > key:
                best, key = p, sc
        lay3[Y] = {"proc": best, **lay2[best][Y], "meta_score": key[0]}
    def pooled(years):
        s_ = sum(lay3[y]["sum"] for y in years)
        n_ = sum(lay3[y]["n"] for y in years)
        return {"n": n_, "mean_pct": (s_ / n_ * 100) if n_ else 0.0}
    summ = {"n_cfgs": len(cfgs), "sealed_2024_2026": pooled([2024, 2025, 2026]), "design_2020_2023": pooled([2020, 2021, 2022, 2023]),
            "by_year": {int(y): {"proc": list(v["proc"]), "cfg": (cfgs[v["cfg"]] if v["cfg"] is not None else None), "n": v["n"],
                                 "mean_pct": (v["sum"] / v["n"] * 100) if v["n"] else 0.0, "train_mean_pct": (v["train_mean"] * 100 if v["train_mean"] is not None else None),
                                 "meta_score_pct": v["meta_score"] * 100} for y, v in lay3.items()},
            "layer2": {"|".join(map(str, p)): {int(y): ((r["sum"] / r["n"] * 100) if r["n"] else 0.0, r["n"]) for y, r in rows.items()}
                       for p, rows in lay2.items()}}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("sealed_2024_2026", "design_2020_2023", "by_year")}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
