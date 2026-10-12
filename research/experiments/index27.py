"""27세대 지수 편입 스윙(META_EVOLUTION.md 27세대 — 결과 전 고정). 이 정보원은 이전 세대에서 한 번도 보지 않았다 → 2016~2026 전체 단일 시험.

사건 = KOSPI200·KOSDAQ150 정기변경 편입(index_changes action='add'). 진입 = 발표일 다음 거래일 종가, 청산 = 적용일 전 거래일 종가
(패시브 매수가 몰리는 마감 동시호가). 비교 = 같은 시장·시가총액 순위 ±50 의 비편입 종목들의 같은 구간 수익 중앙값(규모 맞춘 대조).
순 = 수익 − 왕복 0.415% × 2(2배 비용). 군집 = 정기변경(지수, 적용일).

사용: ``uv run --extra pg python research/experiments/index27.py --out <dir>`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

COST2 = 2 * 0.00415
MAIN = ("KOSPI200", "KOSDAQ150")


def load(con):
    from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg
    ev = pd.read_sql("select index_name, effective_date, action, code, announce_date from index_changes", con,
                     parse_dates=["effective_date", "announce_date"])
    from swing_it.storage import read_prices
    px = read_prices(con, session="raw", cols=("code", "date", "open", "close"), start="2015-10-01", parse_dates=True)
    px = patch_prices(px, read_session_bars_agg(con, start=NXT_START), adjusted=True)
    raw = read_prices(con, session="raw", cols=("code", "date", "close"), table="daily_bars", start="2015-10-01", parse_dates=True)
    from swing_it.storage import read_shares
    sh = read_shares(con, parse_dates=True)  # 정문(가드레일 (k)) — 예전 직접 SELECT 와 같은 행(0 이하 주식수 없음 확인)·순서
    stk = pd.read_sql("select code, market from stocks", con)
    return ev, px, raw, sh, stk


def main() -> None:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    ev, px, raw, sh, stk = load(con)
    C = px.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    days = C.index
    R = raw.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days)
    sh["date"] = sh["knowledge_date"].fillna(sh["date"])
    S = sh.pivot_table(index="date", columns="code", values="shares_outstanding", aggfunc="last")
    S = S.reindex(index=S.index.union(days)).sort_index().ffill().reindex(index=days)
    cap = (R * S.reindex(columns=R.columns))
    mkt = stk.set_index("code")["market"].to_dict()
    adds = ev[ev["action"] == "add"].copy()
    rows = []
    event_codes = set(adds["code"])
    for r in adds.itertuples():
        i_in = days.searchsorted(r.announce_date, side="right")                 # 발표일 다음 거래일
        i_out = days.searchsorted(r.effective_date, side="left") - 1           # 적용일 전 거래일
        if i_in >= len(days) or i_out <= i_in or r.code not in C.columns:
            continue
        p0, p1 = C[r.code].iloc[i_in], C[r.code].iloc[i_out]
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
            continue
        ret = p1 / p0 - 1
        # 규모 맞춘 대조: 같은 시장, 진입일 시총 순위 ±50, 사건 종목 제외
        capd = cap.iloc[i_in]
        same = [c for c in capd.index if mkt.get(c) == mkt.get(r.code) and np.isfinite(capd[c]) and c not in event_codes]
        rk = capd[same + ([r.code] if np.isfinite(capd.get(r.code, np.nan)) else [])].rank(ascending=False)
        if r.code in rk.index:
            me = rk[r.code]
            peers = [c for c in same if abs(rk[c] - me) <= 50]
        else:
            peers = []
        cr = (C[peers].iloc[i_out] / C[peers].iloc[i_in] - 1).replace([np.inf, -np.inf], np.nan).dropna() if peers else pd.Series(dtype=float)
        ctrl = float(cr.median()) if len(cr) >= 10 else np.nan
        rows.append({"index": r.index_name, "eff": str(r.effective_date.date()), "ann": str(r.announce_date.date()), "code": r.code,
                     "entry": str(days[i_in].date()), "exit": str(days[i_out].date()), "hold": int(i_out - i_in), "ret": ret,
                     "ctrl": ctrl, "excess_net": ret - COST2 - ctrl if np.isfinite(ctrl) else np.nan, "n_peers": len(cr)})
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "events.csv"), index=False)
    out = {}
    for nm, sub in (("main", t[t["index"].isin(MAIN)]), ("KOSPI200", t[t["index"] == "KOSPI200"]), ("KOSDAQ150", t[t["index"] == "KOSDAQ150"]),
                    ("KRX300", t[t["index"] == "KRX300"])):
        s = sub.dropna(subset=["excess_net"])
        if s.empty:
            continue
        cl = s.groupby(["index", "eff"])["excess_net"].mean()
        mean_cl, se_cl = cl.mean(), cl.std(ddof=1) / np.sqrt(len(cl))
        yr = pd.to_datetime(s["eff"]).dt.year
        out[nm] = {"n": int(len(s)), "n_rebal": int(len(cl)), "mean_excess_net_pct": float(s["excess_net"].mean() * 100),
                   "median_excess_net_pct": float(s["excess_net"].median() * 100), "cluster_mean_pct": float(mean_cl * 100),
                   "cluster_t": float(mean_cl / se_cl) if se_cl > 0 else np.nan, "rebal_pos_share": float((cl > 0).mean()),
                   "win": float((s["excess_net"] > 0).mean()), "hold_median": float(s["hold"].median()),
                   "early_2016_2020_pct": float(s[yr <= 2020]["excess_net"].mean() * 100), "late_2021_2026_pct": float(s[yr >= 2021]["excess_net"].mean() * 100),
                   "by_year_pct": {int(k): float(v * 100) for k, v in s.groupby(yr)["excess_net"].mean().items()}}
    m = out.get("main", {})
    out["pass"] = bool(m and m["mean_excess_net_pct"] >= 1.0 and m["cluster_t"] >= 2 and m["rebal_pos_share"] >= 0.6
                       and m["early_2016_2020_pct"] > 0 and m["late_2021_2026_pct"] > 0)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
