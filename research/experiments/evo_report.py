"""진화 후보 보강 보고(META_EVOLUTION.md 정정 G12-1) — 판정 바 밖의 기술 항목.

- 시드별 체결 수 중앙값(표본 바 둘째 조건 ≥ 30).
- 리더 마진: 슬롯 대비 거래당 (후보 − 기준) 과 두 후보 각각의 주문일 군집 부트스트랩 SE 를 합친 SE(짝 없음).
- 2021 제외 슬롯 t, 2022~23 부분집합 평균·건수, 모든 시드 공통 거래 vs 한 시드만 산 거래 평균.
- 폴드별 시험 체결(``fold_n``). 학습창 하한 결합 비율은 결과 JSON 에 학습 체결이 없어 검증 에이전트가 따로 잰다.

사용: ``uv run python research/experiments/evo_report.py G7A G12A G12B …`` (첫 인자가 기준)
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["."]
from research.experiments.evo_judge import EVO, SLOT_W, boot_t  # noqa: E402

N_SEED_MIN = 30


def load_rows(c: str) -> "tuple[pd.DataFrame, list[int]]":
    ts, ns = [], []
    for f in sorted(glob.glob(f"{EVO}/{c}_s*.json")):
        off = int(f.rsplit("_s", 1)[1].split(".")[0])
        t = pd.read_csv(f.replace(".json", "_trades.csv"), dtype={"code": str})
        t["off"] = off
        ts.append(t)
        ns.append(len(t))
    t = pd.concat(ts) if ts else pd.DataFrame()
    if len(t):
        amb = t["amb"].astype(bool)
        p = 0.275
        t["n2"] = t["net2"].where(~amb, p * t["net2_stop0"] + (1 - p) * t["net2"])
        if "adv20" in t:                               # evo_judge 와 같은 소형주 추가 미끄러짐
            t["n2"] = t["n2"] - 0.004 * t["wb"] * (t["adv20"] < 1000.0)
        t["n2s"] = t["n2"] * (t["alloc"] / SLOT_W).clip(upper=1.0)
    return t, ns


def uniq(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby(["code", "order_day"], as_index=False).agg(n2s=("n2s", "mean"), k=("off", "nunique"))


def se_of(x: pd.Series, g: pd.Series) -> float:
    m, tt = boot_t(x, g)
    return abs(m / tt) if tt and np.isfinite(tt) and tt != 0 else float("nan")


def main(cands: list[str]) -> None:
    base = cands[0]
    tb, _ = load_rows(base)
    ub = uniq(tb)
    mb, seb = ub["n2s"].mean(), se_of(ub["n2s"], ub["order_day"])
    rows = []
    for c in cands:
        t, ns = load_rows(c)
        if not len(t):
            continue
        u = uniq(t)
        nset = t["off"].nunique()
        yr = pd.to_datetime(u["order_day"]).dt.year
        m, s = u["n2s"].mean(), se_of(u["n2s"], u["order_day"])
        x21 = u[yr != 2021]
        _, t21 = boot_t(x21["n2s"], x21["order_day"]) if len(x21) >= 3 else (np.nan, np.nan)
        bear = u[yr.isin([2022, 2023])]
        allk, onek = u[u["k"] == nset], u[u["k"] == 1]
        marg = (m - mb) / np.hypot(s, seb) if c != base else np.nan
        rows.append({"후보": c, "시드별 체결": "/".join(map(str, ns)), "중앙값": st.median(ns),
                     "중앙값 ≥ 30": st.median(ns) >= N_SEED_MIN, "고유": len(u), "슬롯 거래당%": m * 100,
                     "마진(기준 대비 SE배)": marg, "2021 제외 t": t21,
                     "2022~23 n·평균%": f"{len(bear)} · {bear['n2s'].mean() * 100:+.2f}" if len(bear) else "0",
                     "공통 n·평균%": f"{len(allk)} · {allk['n2s'].mean() * 100:+.2f}" if len(allk) else "0",
                     "한 시드 n·평균%": f"{len(onek)} · {onek['n2s'].mean() * 100:+.2f}" if len(onek) else "0"})
        fn = []
        for f in sorted(glob.glob(f"{EVO}/{c}_s*.json")):
            fn.append(json.load(open(f, encoding="utf-8"))["metrics"].get("fold_n"))
        rows[-1]["폴드별 체결"] = " ".join(str(x) for x in fn)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 80)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
