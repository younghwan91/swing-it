"""시드 합의 앙상블 판정(META_EVOLUTION.md 15세대) — 독립 GA 시드 집합들이 함께 고른 거래만 실행하는 절차.

절차(결과 전 고정): 같은 메뉴·같은 폴드 학습으로 시드 집합 N 개를 돌린다. 시험창에서 (종목, 주문일) 을 고른 시드 집합 수가 ≥ k 인
거래만 실행하고, 포지션은 동의한 시드 집합 수만큼 나눠 각 하위 묶음이 자기 설정의 청산을 따른다. 그래서 거래 손익 = 동의한 시드
판본들의 평균(라이브에서 하위 묶음 분할로 실행 가능).

지표는 :mod:`evo_judge` 와 같다(슬롯 대비 순 2배 비용 + 소형주 미끄러짐, 주문일 군집 부트스트랩 t). k 를 1..N 으로 훑어 기술로 낸다.

사용: ``uv run python research/experiments/evo_consensus.py G15C --k 5``
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["."]
from research.experiments.evo_judge import EVO, N_MIN, SLOT_W, SMALL_ADV, SMALL_SLIP, T_MIN, boot_t  # noqa: E402


def seed_trades(c: str) -> pd.DataFrame:
    ts = []
    for f in sorted(glob.glob(f"{EVO}/{c}_s*_trades.csv")):
        t = pd.read_csv(f, dtype={"code": str})
        t["off"] = int(f.rsplit("_s", 1)[1].split("_")[0])
        ts.append(t)
    t = pd.concat(ts)
    amb = t["amb"].astype(bool)
    p = 0.275
    t["n2"] = t["net2"].where(~amb, p * t["net2_stop0"] + (1 - p) * t["net2"])
    t["n2"] = t["n2"] - SMALL_SLIP * t["wb"] * (t["adv20"] < SMALL_ADV)
    t["n2s"] = t["n2"] * (t["alloc"] / SLOT_W).clip(upper=1.0)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand")
    ap.add_argument("--k", type=int, required=True, help="주 판정 합의 문턱(시드 집합 수)")
    a = ap.parse_args()
    t = seed_trades(a.cand)
    n_sets = t["off"].nunique()
    u = t.groupby(["code", "order_day"], as_index=False).agg(n2s=("n2s", "mean"), k=("off", "nunique"))
    u["year"] = pd.to_datetime(u["order_day"]).dt.year
    rows = []
    for k in range(1, n_sets + 1):
        x = u[u["k"] >= k]
        m, tt = boot_t(x["n2s"], x["order_day"]) if len(x) >= 3 else (np.nan, np.nan)
        x21 = x[x["year"] != 2021]
        _, t21 = boot_t(x21["n2s"], x21["order_day"]) if len(x21) >= 3 else (np.nan, np.nan)
        rows.append({"k ≥": k, "고유 거래": len(x), "연평균 거래": len(x) / 4.0, "슬롯 거래당%": m * 100, "t": tt,
                     "승률": float((x["n2s"] > 0).mean()) if len(x) else np.nan, "2021 제외 t": t21,
                     "2022~23 n": int(x["year"].isin([2022, 2023]).sum()),
                     "주": "◀" if k == a.k else ""})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"{a.cand}: 시드 집합 {n_sets}개, 시드 판본 거래 {len(t)}")
    print(df.round(3).to_string(index=False))
    r = df[df["k ≥"] == a.k].iloc[0]
    ok_n, ok_t = r["고유 거래"] >= N_MIN, (r["고유 거래"] >= N_MIN) and (r["t"] >= T_MIN)
    print(f"\n주 판정(k ≥ {a.k}): 표본 바 {'✓' if ok_n else '✗'} · 우위 바 {'✓' if ok_t else '✗'}")


if __name__ == "__main__":
    main()
