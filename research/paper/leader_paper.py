"""21세대 주도주 보유형 — 전진 모의매매(실주문 없음). 장 마감·DB 갱신 뒤 하루 한 번 돌린다.

매번 DB 최신 수정 주가로 2017-10 부터 같은 규칙을 다시 시뮬레이션해(결정적·인과적) 마지막 날의 보유·다음날 주문·당일 수익을
**덧붙이기 전용 장부**(`research/paper/leader21_ledger.jsonl`)에 남긴다. 이미 기록한 날은 다시 쓰지 않는다 — 전진 성적은 장부의
`ret_today` 를 이어 붙여 잰다(과거 수정 주가 재조정에 흔들리지 않게).

사용: ``export $(grep -v '^#' ~/git/quant-airflow/.env | xargs) && uv run --with numba --extra pg python research/paper/leader_paper.py``
(``--arm`` 은 사전등록 팔 이름, 기본 META_EVOLUTION.md 21세대 모의매매 등록)
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

LEDGER = "research/paper/leader21_ledger.jsonl"
COST_RT = 0.00415                                     # 캐시 cost_tab 최근 값(왕복, 시장 공통)
ARMS = {                                              # 사전등록(META_EVOLUTION.md 21세대 모의매매) — 결과 보고 바꾸지 않는다
    "leader_A": {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "univ": "top200", "regime": 1},   # 격자 215(검증 24차 권고)
    # 23세대 C(정정 G23-1): 신호일 KRX 종가(종가 단일가) 매수 — 체결 충실도 팔. 체결가 = krx_session 정규장 종가.
    "leader_C": {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "univ": "top200", "regime": 1, "entry": "close"},
    # 전진 팔(META_EVOLUTION.md 'leader_C20 사전등록'): C 와 같고 슬롯 20 · 모든 비중 절반 — 장부 leader_c20_ledger.jsonl
    "leader_C20": {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "univ": "top200", "regime": 1, "entry": "close",
                   "max_pos": 20, "pos_cap": 0.05, "risk": 0.0025},
}
ARM_LEDGER = {"leader_C20": "research/paper/leader_c20_ledger.jsonl"}


def load_db(con) -> tuple:
    from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg
    px = pd.read_sql("select code, date, open, high, low, close, volume from daily_bars_adjusted where date >= '2016-08-01'", con,
                     parse_dates=["date"])
    # 2026-09-14+ 일봉에 NXT 야간 체결이 섞였다(종가 85~88% 오염) → 분봉 정규장(09:00~15:32)으로 다시 만든다(조정계수 = 시가 비율)
    px = patch_prices(px, read_session_bars_agg(con, start=NXT_START), adjusted=True)
    stk = pd.read_sql("select code, market from stocks", con)
    days = pd.DatetimeIndex(sorted(px["date"].unique()))
    codes = sorted(px["code"].astype(str).unique())
    def wide(col):
        return px.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=codes).to_numpy(float).T
    O, H, L_, C, V = (wide(k) for k in ("open", "high", "low", "close", "volume"))
    TR = np.isfinite(C) & (np.nan_to_num(V) > 0)
    mk = stk.set_index("code")["market"].reindex(codes).fillna("거래소").to_numpy()
    mkt = np.where(mk == "코스닥", 1, 0).astype(np.int64)
    cost = np.full((2, len(days)), COST_RT)
    return days, codes, O, H, L_, C, TR, cost, mkt


def main() -> None:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="leader_A")
    ap.add_argument("--ledger", default=None, help="기본: 팔별 장부(ARM_LEDGER), 없으면 leader21_ledger.jsonl")
    ap.add_argument("--legacy-close", action="store_true", help="정정 G28-3 전 결함 가격(daily_bars.close)으로 시총 — 비교용만")
    a = ap.parse_args()
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    days, codes, O, H, L_, C, TR, cost, mkt = load_db(con)
    D = L.build(days, codes, O, H, L_, C, TR, cost, mkt, con, pit_shares=True, krx_patch=True, raw_close=not a.legacy_close)
    # 국면 = 실제 KOSPI(sector_index 001) 종가 ≥ 200일선(검증 24차 MUST-FIX) — 지수 200일선이 생긴 날부터, 그 전은 시총 가중 대용
    kx = pd.read_sql("select date, close from sector_index where code='001' order by date", con, parse_dates=["date"]).set_index("date")["close"]
    kma = kx.rolling(200, min_periods=200).mean()
    kon = (kx >= kma).where(kma.notna())
    for d_, v in kon.dropna().items():
        if d_ in days:
            D["regime_on"][days.get_loc(d_)] = bool(v)
    cfg = ARMS[a.arm]
    a.ledger = a.ledger or ARM_LEDGER.get(a.arm, LEDGER)
    st: dict = {}
    eq, tr = L.simulate(D, cfg, 1.0, end=str(days[-1].date()), state=st, paper=True)
    ret_today = float(eq.iloc[-1] / eq.iloc[-2] - 1)
    ix = pd.read_sql("select code, date, close from sector_index where code in ('001','101') and date >= %s order by date", con,
                     params=(str(days[-5].date()),), parse_dates=["date"])
    bench = {}
    for code, nm in (("001", "kospi"), ("101", "kosdaq")):
        z = ix[ix["code"] == code].set_index("date")["close"]
        if len(z) >= 2 and z.index[-1] == days[-1]:
            bench[nm] = float(z.iloc[-1] / z.iloc[-2] - 1)
    names = pd.read_sql("select code, name from stocks", con).set_index("code")["name"].to_dict()
    import subprocess  # noqa: PLC0415
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {"date": st["last_day"], "arm": a.arm, "cfg": cfg, "git": sha, "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "krx_patched": True, "raw_close": not a.legacy_close, "ret_today": ret_today, "equity_sim": float(st["equity"]),
           "cash_w": float(st["cash"] / st["equity"]), "regime_on": st["regime_on"],
           "positions": {c: {"name": names.get(c, ""), "w": round(float(p["w"]), 4), "units": int(p["units"]),
                             "entry": str(days[p["entry_t"]].date()), "avg": float(p["cost_basis"] / p["qty"]), "last": float(p["last_px"])}
                         for c, p in st["pos"].items()},
           "orders_next_open": {"buy": [(c, names.get(c, "")) for c in st["buy"]], "sell": [(c, names.get(c, "")) for c in st["sell"]],
                                "add": [(c, names.get(c, "")) for c in st["add"]]},
           "bench_ret_today": bench}
    done = set()
    if os.path.exists(a.ledger):
        with open(a.ledger, encoding="utf-8") as fh:
            done = {(json.loads(x)["date"], json.loads(x)["arm"]) for x in fh if x.strip()}
    if (row["date"], row["arm"]) in done:
        print(f"이미 기록됨: {row['date']} {row['arm']} — 덧붙이지 않는다")
    else:
        os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
        with open(a.ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"기록: {row['date']} {row['arm']}")
    print(json.dumps(row, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
