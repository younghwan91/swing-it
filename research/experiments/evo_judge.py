"""진화 후보 판정(META_EVOLUTION.md 정정 G6-1) — 시드 집합을 합친 고유 거래(종목, 주문일) 기준 표본 바·우위 바 + race(기술).

- 표본 바: 고유 거래 ≥ 60(4년). 못 미치면 '판단 불가(체결 부족)'.
- 우위 바: 고유 거래당 순(2배 비용, 모호 거래 p 혼합)·슬롯 대비 손익의 주문일 군집 부트스트랩 t ≥ 1.
- race(k = 3, min_sets = 5): 공유 오프셋 연환산 짝 차이, 표본 바를 넘은 후보끼리(기술).

사용: ``uv run python research/experiments/evo_judge.py G4C G6B G6C …``
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys

import numpy as np
import pandas as pd

EVO = "research/logs/sepa_explore/pattern_ga_evo"
N_MIN, T_MIN, K, MIN_SETS, BOOT, SEED, SLOT_W = 60, 1.0, 3.0, 5, 2000, 20261011, 0.10
SMALL_ADV, SMALL_SLIP = 1000.0, 0.004


def load(c: str):
    anns, trades = {}, []
    for f in sorted(glob.glob(f"{EVO}/{c}_s*.json")):
        off = int(f.rsplit("_s", 1)[1].split(".")[0])
        d = json.load(open(f, encoding="utf-8"))
        anns[off] = d["metrics"]["ann"] * 100
        t = pd.read_csv(f.replace(".json", "_trades.csv"), dtype={"code": str})
        trades.append(t)
    t = pd.concat(trades) if trades else pd.DataFrame()
    if len(t):
        amb = t["amb"].astype(bool)
        p = 0.275                                     # 엔진 pmix(캐시 meta 0.27523) — 모호 거래만 영향(대부분 0건)
        t["n2"] = t["net2"].where(~amb, p * t["net2_stop0"] + (1 - p) * t["net2"])
        if "adv20" in t:                               # 14세대(사전등록): ADV20 < 10억 소형주는 왕복 추가 미끄러짐 0.4%(첫 단위 비중)
            t["n2"] = t["n2"] - SMALL_SLIP * t["wb"] * (t["adv20"] < SMALL_ADV)
        t["n2s"] = t["n2"] * (t["alloc"] / SLOT_W).clip(upper=1.0)
        # 검증 10차 MINOR 1(정정 G8-1): 같은 (종목, 주문일)은 시드마다 청산 설정이 달라 손익이 다르다 → 시드 평균 한 행
        t = t.groupby(["code", "order_day"], as_index=False).agg(n2=("n2", "mean"), n2s=("n2s", "mean"))
    return anns, t


def boot_t(x: pd.Series, groups: pd.Series) -> tuple[float, float]:
    """주문일 군집 부트스트랩 → (평균, 평균/표준오차)."""
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    g = groups.to_numpy()
    u = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in u}
    xv = x.to_numpy()
    ms = []
    for _ in range(BOOT):
        pick = rng.choice(u, size=len(u), replace=True)
        ms.append(xv[np.concatenate([idx[k] for k in pick])].mean())
    se = float(np.std(ms, ddof=1))
    m = float(xv.mean())
    return m, (m / se if se > 0 else float("nan"))


def main(cands: list[str]) -> None:
    rows, A = [], {}
    for c in cands:
        anns, t = load(c)
        A[c] = anns
        n = len(t)
        m2, t2 = boot_t(t["n2"], t["order_day"]) if n else (np.nan, np.nan)
        ms, ts = boot_t(t["n2s"], t["order_day"]) if n else (np.nan, np.nan)
        loss = t.loc[t["n2s"] < -0.005, "n2s"] if n else pd.Series(dtype=float)
        top = t["n2s"].quantile(0.95) if n else np.nan
        rows.append({"후보": c, "연환산 평균": st.mean(anns.values()) if anns else np.nan,
                     "sd": st.stdev(anns.values()) if len(anns) > 1 else np.nan, "고유 거래": n,
                     "거래당(2배)": m2 * 100, "t": t2, "슬롯 대비 거래당": ms * 100, "슬롯 t": ts,
                     "손실 평균(슬롯)": loss.mean() * 100 if len(loss) else np.nan,
                     "손실 최악(슬롯)": t["n2s"].min() * 100 if n else np.nan,
                     "상위 5% 문턱": top * 100,
                     "표본 바": n >= N_MIN, "우위 바": (n >= N_MIN) and (ts >= T_MIN)})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    ok = [r["후보"] for r in rows if r["표본 바"] and len(A[r["후보"]]) >= MIN_SETS]
    if len(ok) >= 2:
        lead = max(ok, key=lambda c: st.mean(A[c].values()))
        print(f"\nrace(기술, 표본 바 통과 {ok}) 리더 {lead}")
        for c in ok:
            if c == lead:
                continue
            offs = sorted(set(A[c]) & set(A[lead]))
            d = [A[c][o] - A[lead][o] for o in offs]
            m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
            print(f"  {c} − {lead}: {m:+.2f} ± {se:.2f} → m + {K:g}SE {m + K * se:+.2f} {'탈락' if m + K * se < 0 else '유지'}")


if __name__ == "__main__":
    main(sys.argv[1:])
