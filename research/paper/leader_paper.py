"""21세대 주도주 보유형 — 전진 모의매매(실주문 없음). 장 마감·DB 갱신 뒤 하루 한 번 돌린다.

매번 DB 최신 수정 주가로 2017-10 부터 같은 규칙을 다시 시뮬레이션해(결정적·인과적) 마지막 날의 보유·다음날 주문·당일 수익을
**덧붙이기 전용 장부**(`research/paper/leader21_ledger.jsonl`)에 남긴다. 이미 기록한 날은 다시 쓰지 않는다 — 전진 성적은 장부의
`ret_today` 를 이어 붙여 잰다(과거 수정 주가 재조정에 흔들리지 않게).

사용: ``export $(grep -v '^#' ~/git/quant-airflow/.env | xargs) && uv run --with numba --extra pg python research/paper/leader_paper.py``
(``--arm`` 은 사전등록 팔 이름, 기본 META_EVOLUTION.md 21세대 모의매매 등록)

쌍 팔 ``*_rs``(META_EVOLUTION.md '전진 팔 *_rs 사전등록'): 기존 팔과 cfg·코드 경로가 같고 정문 인자만
``read_prices(session="regular")``·``read_raw_close(aftermarket_fix=True)`` 로 다르다(:data:`ARM_PRICE`). 장부 ``leader_rs_ledger.jsonl``,
첫 기록 거래일 2026-10-13(:data:`ARM_START` — 그 전 날짜 행은 쓰지 않는다).
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
# 쌍 팔(META_EVOLUTION.md '전진 팔 *_rs 사전등록', 2026-10-13): cfg 는 짝 팔과 같은 사전, 정문 애프터마켓 보정 인자만 다르다
RS_PAIR = {"leader_A_rs": "leader_A", "leader_C_rs": "leader_C", "leader_C20_rs": "leader_C20"}
ARMS.update({k: dict(ARMS[v]) for k, v in RS_PAIR.items()})
ARM_LEDGER = {"leader_C20": "research/paper/leader_c20_ledger.jsonl",
              **{k: "research/paper/leader_rs_ledger.jsonl" for k in RS_PAIR}}
PRICE_RAW = {"session": "raw", "aftermarket_fix": False}          # 기존 팔 고정값(바꾸지 않는다)
ARM_PRICE = {k: {"session": "regular", "aftermarket_fix": True} for k in RS_PAIR}
ARM_START = {k: "2026-10-13" for k in RS_PAIR}                     # 첫 기록 거래일 — 그 전 날짜 행은 장부에 쓰지 않는다(소급 성적 없음)


def load_db(con, session: str = "raw", info: "dict | None" = None) -> tuple:
    from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg
    from swing_it.storage import read_prices
    px = read_prices(con, session=session, cols=("code", "date", "open", "high", "low", "close", "volume"), start="2016-08-01", parse_dates=True)
    if info is not None:
        info.update(px.attrs.get("krx_session", {}))
    # 2026-09-14+ 일봉에 KRX 애프터마켓(16:00~20:00) 체결이 섞였다(종가 85~88% 오염) → 분봉 정규장(09:00~15:32)으로 다시 만든다(조정계수 = 시가 비율)
    # (session="regular" 이면 정문이 이미 보정했다 — patch_prices 는 krx_patched 행을 건드리지 않는다(멱등))
    px = patch_prices(px, read_session_bars_agg(con, start=NXT_START), adjusted=True)
    stk = pd.read_sql("select code, market from stocks", con)
    days = pd.DatetimeIndex(sorted(px["date"].unique()))
    codes = sorted(px["code"].astype(str).unique())
    def wide(col):
        return px.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=codes).to_numpy(float).T
    O, H, L_, C, V = (wide(k) for k in ("open", "high", "low", "close", "volume"))  # noqa: E741
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
    price = ARM_PRICE.get(a.arm, PRICE_RAW)
    if a.legacy_close and a.arm in ARM_PRICE:
        raise SystemExit(f"{a.arm}: --legacy-close 는 쌍 팔에 쓰지 않는다(원 종가 정문의 애프터마켓 보정이 이 팔의 정의)")
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    px_info: dict = {}
    days, codes, O, H, L_, C, TR, cost, mkt = load_db(con, session=price["session"], info=px_info)  # noqa: E741
    D = L.build(days, codes, O, H, L_, C, TR, cost, mkt, con, pit_shares=True, krx_patch=True, raw_close=not a.legacy_close,
                session=price["session"], aftermarket_fix=price["aftermarket_fix"])
    # 국면 = 실제 KOSPI(sector_index 001) 종가 ≥ 200일선(검증 24차 MUST-FIX) — 지수 200일선이 생긴 날부터, 그 전은 시총 가중 대용
    from swing_it.storage import read_index
    kx = read_index(con, "001", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,   # ×100 정수(옛 값 그대로)
                    proxy="시총 가중 대용(leader21.build regime_on) — 지수 200일선이 생긴 날부터만 덮어쓴다").set_index("date")["close"]
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
    ix = read_index(con, ["001", "101"], cols=("code", "date", "close"), start=str(days[-5].date()), with_level=False,
                    order_by_date=True, parse_dates=True)
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
    if a.arm in ARM_PRICE:                          # 쌍 팔만 — 기존 팔 행은 열이 하나도 바뀌지 않는다
        row.update({"pair": RS_PAIR[a.arm], **price, "price_session": px_info, "raw_src": D.get("raw_src_diag", {})})
    done = set()
    if os.path.exists(a.ledger):
        with open(a.ledger, encoding="utf-8") as fh:
            done = {(json.loads(x)["date"], json.loads(x)["arm"]) for x in fh if x.strip()}
    if (row["date"], row["arm"]) in done:
        print(f"이미 기록됨: {row['date']} {row['arm']} — 덧붙이지 않는다")
    elif row["date"] < ARM_START.get(a.arm, ""):
        print(f"시작 전: {row['date']} < {ARM_START[a.arm]} {row['arm']} — 사전등록 첫 기록 거래일 전이라 쓰지 않는다")
    else:
        os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
        with open(a.ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"기록: {row['date']} {row['arm']}")
    print(json.dumps(row, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
