"""전진 팔 leader_M 과거 기술(META_EVOLUTION.md '전진 팔 leader_M 사전등록' — 판정 무관, 등록 뒤 한 번, 보고만).

모의매매와 같은 경로(leader_paper.load_db + leader21.build + 실제 KOSPI 200일선 국면, 보정 정문 가격)로 2017-10 ~ 마지막 거래일을
leader_M · leader_C_rs · leader_C20_rs 설정으로 돌려 연도별 체결 수 · 거래당 · 손익비 · 연복리 · 낙폭을 낸다.
2017~23 = 설계 구간, 2024~ = 봉인 소모(오염) — 둘 다 증거가 아니다. 이 숫자로 팔 설정을 바꾸지 않는다.
사용: ``uv run --with numba --extra pg python research/experiments/leaderM_desc.py --out <dir>``
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
import research.paper.leader_paper as P  # noqa: E402

ARMS = ("leader_M", "leader_C_rs", "leader_C20_rs")
PERIODS = (("2017-10-02", "2024-01-01", "설계 2017-10~2023"), ("2024-01-01", "2100-01-01", "오염 2024~"))


def trade_stats(tr: pd.DataFrame, hard: float) -> dict:
    if len(tr) == 0:
        return {"n": 0}
    r = tr["ret"].to_numpy(float)
    win, loss = r[r > 0], r[r <= 0]
    return {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)), "win_rate": float(len(win) / len(r)),
            "avg_win": float(win.mean()) if len(win) else 0.0, "avg_loss": float(loss.mean()) if len(loss) else 0.0,
            "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "mean_R": float((r / hard).mean()), "hard_stop_share": float((tr["why"] == "hard_stop").mean())}


def main() -> None:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    price = P.ARM_PRICE["leader_M"]                 # *_rs 와 같은 보정 정문 인자
    days, codes, O, H, L_, C, TR, cost, mkt = P.load_db(con, session=price["session"], unpatched=price["unpatched"])  # noqa: E741
    D = L.build(days, codes, O, H, L_, C, TR, cost, mkt, con, pit_shares=True, krx_patch=True, raw_close=True,
                session=price["session"], aftermarket_fix=price["aftermarket_fix"], unpatched=price["unpatched"])
    from swing_it.storage import read_index
    kx = read_index(con, "001", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,
                    proxy="시총 가중 대용(leader21.build regime_on) — 지수 200일선이 생긴 날부터만 덮어쓴다").set_index("date")["close"]
    kma = kx.rolling(200, min_periods=200).mean()
    for d_, v in (kx >= kma).where(kma.notna()).dropna().items():
        if d_ in days:
            D["regime_on"][days.get_loc(d_)] = bool(v)
    end = str(days[-1].date())
    out: dict = {"end": end, "arms": {}}
    mkt_cw = pd.Series(np.cumprod(1 + D["cw"]), index=days)       # 시총 가중 시장 대용(공식 KOSPI 는 DB 에 2024-01-26 부터)
    for arm in ARMS:
        cfg = P.ARMS[arm]
        res = {}
        for cm in (1.0, 2.0):
            eq, tr = L.simulate(D, cfg, cm, end=end)
            tr = tr.assign(entry=pd.to_datetime(tr["entry"]), exit=pd.to_datetime(tr["exit"]))
            eq.to_frame("eq").to_csv(os.path.join(a.out, f"eq_{arm}_{int(cm)}x.csv"))
            tr.to_csv(os.path.join(a.out, f"trades_{arm}_{int(cm)}x.csv.gz"), index=False)
            per = {}
            for lo, hi, nm in PERIODS:
                m = L.metrics(eq, lo, hi)
                t = tr[(tr["entry"] >= lo) & (tr["entry"] < hi)]
                yrs = max((min(pd.Timestamp(hi), eq.index[-1]) - pd.Timestamp(lo)).days / 365.25, 1e-9)
                k = mkt_cw.loc[lo:hi]
                per[nm] = {"cagr": m["cagr"], "mdd": m["mdd"], "trades_per_year": len(t) / yrs, **trade_stats(t, cfg["hard"]),
                           "mkt_cw_cagr": float((k.iloc[-1] / k.iloc[0]) ** (1 / yrs) - 1)}
            by_year = tr.groupby(tr["entry"].dt.year)["ret"].agg(["count", "mean"]).round(4)
            res[f"{int(cm)}x"] = {"periods": per, "by_year_trades": by_year["count"].to_dict(), "by_year_mean_ret": by_year["mean"].to_dict(),
                                  "by_year_cagr": {int(k): float(v) for k, v in L.metrics(eq, "2017-10-02", None)["by_year"].items()}}
        out["arms"][arm] = res
        print(arm, json.dumps(res["2x"]["periods"], ensure_ascii=False, default=float))
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
