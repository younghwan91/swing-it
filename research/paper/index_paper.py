"""27세대 정정 사양(G27-1) 지수 편입 — 전진 모의매매(실주문 없음). META_EVOLUTION.md 28세대 '전진 등록'.

KOSPI200·KOSDAQ150 정기변경 편입(index_changes action='add')마다: 발표 다음 거래일 KRX 정규장 종가 매수 → 적용 전 거래일 종가 매도.
대조 = 같은 시장, 진입일 시총 순위 ±50 동료 **평균**(그 변경의 편입·편출 종목 제외) 같은 구간 수익. 순 = 수익 − 왕복 0.415% − 대조.
기록은 **덧붙이기 전용**(`research/paper/index27_ledger.jsonl`): 진입이 확정된 날 'entry' 행, 청산이 확정된 날 'exit' 행. 이미 쓴 (종류, 지수, 적용일, 종목)
은 다시 쓰지 않는다. 2026-12 정기변경(발표 2026-11 하순)부터 — 그 전 변경은 무시(``--since``).

사용: ``export $(grep -v '^#' ~/git/quant-airflow/.env | xargs) && uv run --extra pg python research/paper/index_paper.py``
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]
warnings.filterwarnings("ignore")

LEDGER = "research/paper/index27_ledger.jsonl"
COST_RT = 0.00415
IDX_MKT = {"KOSPI200": "거래소", "KOSDAQ150": "코스닥"}


def main() -> None:
    import psycopg2
    from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg
    from swing_it.storage import read_prices
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--since", default="2026-11-01", help="이 날짜 이후 발표된 변경만(사전등록 시작 = 2026-12 정기변경)")
    a = ap.parse_args()
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    q = lambda s, **k: pd.read_sql(s, con, **k)  # noqa: E731
    ch = q("select index_name, effective_date, action, code, name, announce_date from index_changes where announce_date >= %(s)s",
           params={"s": a.since}, parse_dates=["effective_date", "announce_date"])
    ch = ch[ch["index_name"].isin(IDX_MKT)]
    if ch.empty:
        print(f"{a.since} 이후 발표된 KOSPI200·KOSDAQ150 정기변경 없음 — 기록할 것 없음")
        return
    start = str((ch["announce_date"].min() - pd.Timedelta(days=10)).date())
    px = read_prices(con, session="raw", cols=("code", "date", "open", "close"), start=start, parse_dates=True)
    px = patch_prices(px, read_session_bars_agg(con, start=max(start, NXT_START)), adjusted=True)
    raw = read_prices(con, session="raw", cols=("code", "date", "open", "close"), table="daily_bars", start=start, parse_dates=True)
    raw = patch_prices(raw, read_session_bars_agg(con, start=max(start, NXT_START)), adjusted=False)
    sh = q("select code, coalesce(knowledge_date, date) d, shares_outstanding from shares_outstanding_history", parse_dates=["d"])
    stk = q("select code, market, name from stocks")
    C = px.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    days = C.index
    R = raw.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days, columns=C.columns)
    S = sh.pivot_table(index="d", columns="code", values="shares_outstanding", aggfunc="last")
    S = S.reindex(index=S.index.union(days)).sort_index().ffill().bfill().reindex(index=days, columns=C.columns)   # 기록 전 = 첫 기록
    CAP = R * S
    mkt = stk.set_index("code")["market"].to_dict()
    done = set()
    if os.path.exists(a.ledger):
        with open(a.ledger, encoding="utf-8") as fh:
            done = {(r["kind"], r["index"], r["eff"], r["code"]) for r in map(json.loads, filter(str.strip, fh))}
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    new = []
    for (idx, eff), g in ch.groupby(["index_name", "effective_date"]):
        ann = g["announce_date"].min()
        i_in = days.searchsorted(ann, side="right")
        i_out = days.searchsorted(eff, side="left") - 1
        changed = set(g["code"])
        for r in g[g["action"] == "add"].itertuples():
            key_e = ("entry", idx, str(eff.date()), r.code)
            if i_in >= len(days) or r.code not in C.columns:
                continue                                                    # 진입일 종가 아직 없음
            p0 = C[r.code].iloc[i_in]
            capd = CAP.iloc[i_in]
            pool = [c for c in capd.index if mkt.get(c) == IDX_MKT[idx] and np.isfinite(capd[c]) and c not in changed]
            rk = capd[pool + [r.code]].rank(ascending=False)
            peers = [c for c in pool if abs(rk[c] - rk[r.code]) <= 50] if np.isfinite(rk.get(r.code, np.nan)) else []
            base = {"index": idx, "eff": str(eff.date()), "ann": str(ann.date()), "code": r.code, "name": r.name, "git": sha,
                    "run_at": pd.Timestamp.now().isoformat(timespec="seconds"), "krx_patched": True}
            if key_e not in done:
                new.append({"kind": "entry", **base, "date": str(days[i_in].date()), "px": float(p0), "n_peers": len(peers)})
            key_x = ("exit", idx, str(eff.date()), r.code)
            if key_x in done or i_out >= len(days) or i_out <= i_in:
                continue
            p1 = C[r.code].iloc[i_out]
            cr = (C[peers].iloc[i_out] / C[peers].iloc[i_in] - 1).replace([np.inf, -np.inf], np.nan).dropna()
            ctrl, ctrl_kind = (float(cr.mean()), "peer_mean") if len(cr) >= 10 else (float("nan"), "")
            if not np.isfinite(ctrl):                                       # 동료 부족 → 시장 동일 가중(검증 28차 대체 대조)
                allm = [c for c in C.columns if mkt.get(c) == IDX_MKT[idx] and c not in changed]
                mr = (C[allm].iloc[i_out] / C[allm].iloc[i_in] - 1).replace([np.inf, -np.inf], np.nan).dropna().clip(-0.9, 5)
                ctrl, ctrl_kind = float(mr.mean()), "market_ew"
            ret = float(p1 / p0 - 1)
            new.append({"kind": "exit", **base, "date": str(days[i_out].date()), "entry_px": float(p0), "px": float(p1), "ret": ret,
                        "ctrl": ctrl, "ctrl_kind": ctrl_kind, "excess_net": ret - COST_RT - ctrl, "n_peers": int(len(cr))})
    if new:
        os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
        with open(a.ledger, "a", encoding="utf-8") as fh:
            for row in new:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"새 기록 {len(new)} 행 (데이터 마지막 날 {days[-1].date()})")
    for row in new:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
