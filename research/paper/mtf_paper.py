"""전진 모의매매 팔 ``mtf30_fwd`` — 일봉 상승(U3) 중 30분봉 역→정배열 전환(실주문 없음). 정본 ``research/paper/MTF30_FORWARD_PREREG.md``.

매일 한 번(DB 확정 뒤 장 전) 고정 시작점(분봉 2026-09-01~)부터 전날까지 분봉으로 2026-11-17 진입분부터 계좌를 처음부터 다시 돌리고,
``research/paper/mtf30_ledger.jsonl`` 에 그 거래일 한 줄을 **덧붙인다**(이미 쓴 거래일은 다시 쓰지 않는다). 앞 행에서 청산된 거래가
다시 계산한 장부와 다르면 ``drift`` 를 적고 ``감시:`` 를 찍는다.

사용(작업 폴더에서): ``set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --with numba --extra pg python research/paper/mtf_paper.py [--dry-run] [--fwd-start YYYY-MM-DD]``
``--dry-run`` 은 장부를 쓰지 않고 건수·시간만 찍는다(``--fwd-start`` 를 바꾼 실행은 언제나 dry-run).
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from research.signals import mtf_transition as T  # noqa: E402

ARM = "mtf30_fwd"
LEDGER = REPO / "research" / "paper" / "mtf30_ledger.jsonl"
PREREG = "research/paper/MTF30_FORWARD_PREREG.md"
FWD_START = "2026-11-17"
MINUTE_START, DAILY_START, IDX_START = "2026-09-01", "2025-06-02", "2026-01-02"
INTRA_OPEN, INTRA_CLOSE_EXCL = "09:00:00", "15:33:00"      # 실행기 mtf_transition_wf 와 같다(15:30~15:32 단일가 → 15:29 봉)
CFG = {"tf": "30m", "up": "U3", "spread": True, "overnight": False, "exit": "X3", "stop": "SF", "regime": "on"}   # MT-2 설정 189
SHADOW = {**CFG, "regime": "ignore"}
COST_JUDGE = 2.0
COVER_MIN, LAST_BAR = 0.95, "15:00"
DRIFT_TOL = 1e-6


def _git() -> tuple:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "research/signals", "research/paper/mtf_paper.py"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    return sha, dirty


# =============================================================================================== 순수 부품(DB 없음 — 테스트 대상)

def complete_days(m15: pd.DataFrame, expect: dict, cover_min: float = COVER_MIN, last_bar: str = LAST_BAR) -> list:
    """거래일마다 기대 종목(그날 일봉 거래대금 > 0 인 패널 종목) 중 ``last_bar`` 봉까지 분봉이 있는 몫이 ``cover_min`` 이상인 날."""
    x = m15.assign(day=m15["ts"].dt.normalize())
    lb = pd.Timedelta(hours=int(last_bar[:2]), minutes=int(last_bar[3:]))
    late = x[x["ts"] - x["day"] >= lb].groupby("day")["code"].agg(lambda s: set(s))
    out = []
    for d, codes in expect.items():
        if not codes:
            continue
        have = late.get(pd.Timestamp(d), set())
        if len(have & codes) / len(codes) >= cover_min:
            out.append(pd.Timestamp(d))
    return sorted(out)


def book(m15: pd.DataFrame, dc: pd.DataFrame, tv: pd.DataFrame, elig: list, kosdaq: pd.Series, mkt: np.ndarray, cost_fn,
         fwd_start: str = FWD_START, topn: int = T.TOPN) -> dict:
    """고정 규칙 계좌. ``cost_fn(udates) -> 거래일 × 시장 왕복 비용``. 반환: 패널 S·국면·main(비용 2배·1배)·shadow 계좌와 자산."""
    codes = sorted(m15["code"].unique())
    S = T.build_tf(T.aggregate(m15, 30), dc, tv, elig, "30m", codes=codes, topn=topn)
    ud = S["udates"]
    costs = cost_fn(ud)
    RG = T.regime_on(kosdaq, ud, T.u1_share(S))
    S["regime"] = RG["on"]
    T.add_floor_stop(S, costs, mkt)
    OCs, dayclose = T.prepare([S], stops=("SF",))
    d0 = int(np.searchsorted(ud, np.datetime64(pd.Timestamp(fwd_start))))
    d1 = len(ud) - 1
    out = {"S": S, "RG": RG, "costs": costs, "dayclose": dayclose, "d0": d0, "d1": d1}
    for name, cfg, cm in (("main", CFG, COST_JUDGE), ("main_c1", CFG, 1.0), ("shadow", SHADOW, COST_JUDGE)):
        c = T.candidates(S, OCs[0], cfg, 0, costs, mkt, cm)
        acc = T.account(T.take(c, c["eday"] >= d0), dayclose)
        eq = T.equity_curve(acc, dayclose, ud, d0, d1, costs, mkt, cm) if d0 <= d1 else pd.DataFrame(columns=["equity", "expo"])
        out[name] = acc
        out[name + "_eq"] = eq
    return out


def next_regime(B: dict, kosdaq: pd.Series, tv: pd.DataFrame, elig: list, topn: int = T.TOPN) -> dict:
    """다음 거래일 국면(자료 마지막 날 d 까지로 계산 가능): 지수 조건 = d 의 KOSDAQ 종가 > MA50, U1 비율 = d 행 U1 ∧ DEF, 대상 = d 까지 거래대금 상위."""
    S = B["S"]
    nxt = np.datetime64(pd.Timestamp(S["udates"][-1]) + pd.offsets.BDay(1), "ns")
    Md, _ = T.universe(tv, elig, S["codes"], np.array([nxt]), topn)
    u1 = S["DS"]["up_same"][0][:, -1]
    den = Md[:, 0].sum()
    share = float((u1 & Md[:, 0]).sum() / den) if den else float("nan")
    r = T.regime_on(kosdaq, np.array([nxt]), np.array([share]))
    return {"on": bool(r["on"][0]), "idx_on": bool(r["idx_on"][0]), "u1_share": share}


def _num(v, nd: int):
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def trade_rows(B: dict, name: str, names: dict | None = None) -> list:
    """계좌 거래를 장부 행으로(가격은 분할 보정 패널 기준 — 표류 점검은 비율·시각만 본다)."""
    S, acc = B["S"], B[name]
    names = names or {}
    ts = S["days"]
    rows = []
    for k in np.argsort(acc["entry_ts"], kind="stable"):
        code = S["codes"][int(acc["ci"][k])]
        real = bool(acc["realized"][k])
        rows.append({"code": code, "name": names.get(code), "entry": str(ts[int(acc["eb"][k])]), "entry_px": _num(acc["entry_px"][k], 4),
                     "stop": _num(acc["stop"][k], 4), "dist": _num(acc["dist"][k], 6), "size": _num(acc["size"][k], 0),
                     "exit": str(ts[int(acc["xb"][k])]) if real else None, "exit_px": _num(acc["exit_px"][k], 4),
                     "why": T.WHY[int(acc["why"][k])], "realized": real, "ret": _num(acc["ret"][k], 8),
                     "Racct": _num(acc["Racct"][k], 6), "reg_on": bool(acc["reg_on"][k])})
    return rows


def summary(B: dict, name: str) -> dict:
    acc, eq = B[name], B[name + "_eq"]
    real = acc["realized"]
    last = eq.iloc[-1] if len(eq) else None
    return {"n_entries": int(len(acc["ci"])), "n_realized": int(real.sum()), "n_open": int((~real).sum()),
            "n_realized_on": int((real & acc["reg_on"]).sum()), "Racct_sum_realized": round(float(acc["Racct"][real].sum()), 6),
            "equity": round(float(last["equity"]), 0) if last is not None else T.CASH0,
            "expo": round(float(last["expo"]), 6) if last is not None else 0.0}


def drift(prev_rows: list, now_rows: list, prev_asof: str | None = None, tol: float = DRIFT_TOL) -> list:
    """앞 행과 지금 장부의 대조(정정 MTF30-1, 리뷰 M1). ① 앞 행에서 청산된 거래: (종목, 진입, 청산, 사유, 수익률) 같음.
    ② 앞 행의 미청산 거래: 지금도 있고 손절 거리·비중이 같음. ③ 진입이 ``prev_asof`` 이하인 거래 집합이 같음(새로 생김 = '추가')."""
    now = {(r["code"], r["entry"]): r for r in now_rows}
    bad = []
    for r in prev_rows:
        q = now.get((r["code"], r["entry"]))
        if q is None:
            bad.append({"code": r["code"], "entry": r["entry"], "what": "사라짐"})
        elif r["realized"]:
            if (q["exit"] != r["exit"] or q["why"] != r["why"] or not q["realized"] or q["ret"] is None or r["ret"] is None
                    or abs(q["ret"] - r["ret"]) > tol):
                bad.append({"code": r["code"], "entry": r["entry"], "what": "바뀜", "prev": [r["exit"], r["why"], r["ret"]],
                            "now": [q["exit"], q["why"], q["ret"]]})
        elif q["dist"] is None or r["dist"] is None or abs(q["dist"] - r["dist"]) > tol or abs((q["size"] or 0) - (r["size"] or 0)) > 1:
            bad.append({"code": r["code"], "entry": r["entry"], "what": "보유 바뀜", "prev": [r["dist"], r["size"]], "now": [q["dist"], q["size"]]})
    if prev_asof is not None:
        hi = str(pd.Timestamp(prev_asof) + pd.Timedelta(days=1))
        seen = {(r["code"], r["entry"]) for r in prev_rows}
        bad += [{"code": c, "entry": e, "what": "추가"} for (c, e) in now if e < hi and (c, e) not in seen]
    return bad


def newly_realized(prev_rows: list, now_rows: list) -> list:
    """이번 행에서 처음 실현된 거래(판정은 이 값으로 동결 — 정정 MTF30-1 M2)."""
    done = {(r["code"], r["entry"]) for r in prev_rows if r["realized"]}
    return [r for r in now_rows if r["realized"] and (r["code"], r["entry"]) not in done]


def rows_ok(m15: pd.DataFrame, day: pd.Timestamp, frac: float = 0.90) -> bool:
    """그날 15분봉 행 수 ≥ 직전 5거래일 중앙값 × ``frac``(장중 구멍 보조 점검, 리뷰 m3)."""
    n = m15.groupby(m15["ts"].dt.normalize()).size()
    prior = n[n.index < day].tail(5)
    return day in n.index and (len(prior) == 0 or n[day] >= frac * float(prior.median()))


def read_ledger(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append_row(path: Path, row: dict) -> None:
    """덧붙이기 전용(같은 asof 가 있으면 쓰지 않는다)."""
    if any(r.get("asof") == row["asof"] for r in read_ledger(path)):
        raise ValueError(f"이미 기록됨: {row['asof']}")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


# =============================================================================================== DB

def _connect():
    from swing_it.storage import connect

    if not os.environ.get("KR_QUANT_DB"):
        raise SystemExit("실패: KR_QUANT_DB 가 없다 — set -a; . ~/git/quant-airflow/.env; set +a")
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return con


def load(con) -> dict:
    from swing_it.features.korea import UNIVERSE_MARKETS, load_listing_meta
    from swing_it.storage import read_index, read_prices_wide

    pw = read_prices_wide(con, cols=("close", "trade_value"), start=DAILY_START, session="regular", unpatched="nan")
    dc, tv = pw["close"], pw["trade_value"]
    dc.index, tv.index = pd.DatetimeIndex(dc.index), pd.DatetimeIndex(tv.index)
    meta = load_listing_meta(con)
    ok = meta["excluded"].isna() & meta["market"].isin(UNIVERSE_MARKETS)
    elig = [c for c in tv.columns if c in set(meta.index[ok])]
    adv = tv[elig].rolling(T.ADV_WINDOW, min_periods=T.ADV_MIN_PERIODS).mean().shift(1)
    rk = adv.rank(axis=1, ascending=False, method="first")
    mem = (rk <= T.TOPN) & (adv.index >= MINUTE_START)[:, None]
    union = sorted(mem.columns[mem.any(axis=0)])
    q = ("SELECT code, time_bucket('15 minutes', CASE WHEN ts::time >= '15:30:00' THEN ts::date + time '15:29:00' ELSE ts END) AS ts, "
         "first(open, ts) AS open, max(high) AS high, min(low) AS low, last(close, ts) AS close, sum(volume)::float8 AS volume FROM minute_bars "
         "WHERE code = ANY(%(codes)s) AND ts >= %(lo)s AND ts::time >= %(o)s AND ts::time < %(c)s GROUP BY 1, 2 ORDER BY 1, 2")
    parts = [pd.read_sql_query(q, con, params={"codes": union[a:a + 40], "lo": MINUTE_START, "o": INTRA_OPEN, "c": INTRA_CLOSE_EXCL})
             for a in range(0, len(union), 40)]
    m = pd.concat(parts, ignore_index=True)
    m["ts"] = pd.to_datetime(m["ts"])
    ix = read_index(con, ["001", "101"], cols=("code", "date", "close"), start=IDX_START, with_level=True, order_by_date=True, parse_dates=True)
    w = ix.pivot(index="date", columns="code", values="level").rename(columns={"001": "kospi", "101": "kosdaq"})
    w.index = pd.DatetimeIndex(w.index)
    return {"dc": dc, "tv": tv, "meta": meta, "elig": elig, "union": union, "m15": m, "ix": w, "krx_session": dict(pw.krx_session)}


def cut_to(D: dict, asof: pd.Timestamp) -> dict:
    hi = asof + pd.Timedelta(days=1)
    return {**D, "m15": D["m15"][D["m15"]["ts"] < hi], "dc": D["dc"][D["dc"].index <= asof], "tv": D["tv"][D["tv"].index <= asof],
            "ix": D["ix"][D["ix"].index <= asof]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fwd-start", default=FWD_START, help="바꾸면 언제나 dry-run(스모크용)")
    ap.add_argument("--ledger", default=str(LEDGER))
    a = ap.parse_args(argv)
    dry = a.dry_run or a.fwd_start != FWD_START
    t0 = time.time()
    con = _connect()
    D = load(con)
    m = D["m15"]
    days = sorted(m["ts"].dt.normalize().unique())
    tail = [pd.Timestamp(d) for d in days[-3:]]
    expect = {}
    for d in tail:
        if d in D["tv"].index:
            row = D["tv"].loc[d].reindex(D["union"])
            expect[d] = set(row.index[row.to_numpy(dtype=float, na_value=np.nan) > 0])
    ok_days = [d for d in complete_days(m[m["ts"] >= tail[0]], expect) if rows_ok(m, d)]
    if not ok_days:
        print(f"실패: 최근 3거래일({', '.join(str(d.date()) for d in tail)}) 분봉 완결 날 없음")
        return 1
    asof = ok_days[-1]
    if asof != tail[-1]:
        print(f"감시: 분봉 마지막 날 {tail[-1].date()} 미완결 — {asof.date()} 까지만 쓴다")
    if asof < pd.Timestamp(a.fwd_start) and not dry:
        print(f"시작 전: 자료 마지막 {asof.date()} < 시작 {a.fwd_start}")
        return 0
    path = Path(a.ledger)
    led = [r for r in read_ledger(path) if r.get("arm") == ARM]
    if any(r["asof"] == str(asof.date()) for r in led) and not dry:
        print(f"이미 기록됨: {asof.date()}")
        return 0
    D = cut_to(D, asof)
    if D["ix"].index.max() != asof:
        print(f"감시: 지수 마지막 행 {D['ix'].index.max().date()} ≠ 자료 마지막 날 {asof.date()} — 국면 계산이 직전 지수 행을 쓴다")
    from swing_it.features.korea import cost_rt

    codes = sorted(D["m15"]["code"].unique())
    mk = D["meta"]["market"].reindex(codes)
    if not mk.isin(["KOSPI", "KOSDAQ"]).all():
        print("실패: 패널 종목 시장이 KOSPI/KOSDAQ 밖")
        return 1
    mkt = (mk == "KOSDAQ").astype(np.int64).to_numpy()
    B = book(D["m15"], D["dc"], D["tv"], D["elig"], D["ix"]["kosdaq"], mkt,
             lambda ud: np.array([[cost_rt(pd.Timestamp(d).date(), x) for x in ("KOSPI", "KOSDAQ")] for d in ud]), fwd_start=a.fwd_start)
    names = D["meta"]["name"].to_dict()
    rows = {k: trade_rows(B, k, names) for k in ("main", "shadow")}
    RG = B["RG"]
    nr = next_regime(B, D["ix"]["kosdaq"], D["tv"], D["elig"])
    if dry:
        print(f"dry-run: asof {asof.date()} 시작 {a.fwd_start}, 패널 {len(codes)} 종목 × {len(B['S']['days'])} 봉, "
              f"main 진입 {len(rows['main'])} shadow 진입 {len(rows['shadow'])}, 다음 날 국면 {'ON' if nr['on'] else 'OFF'} ({time.time() - t0:.0f}s)")
        return 0
    prev = led[-1] if led else None
    dr = {k: drift(prev[k]["trades"], rows[k], prev["asof"]) for k in ("main", "shadow")} if prev else {"main": [], "shadow": []}
    nr_ = {k: newly_realized(prev[k]["trades"] if prev else [], rows[k]) for k in ("main", "shadow")}
    sha, dirty = _git()
    last = -1
    row = {"arm": ARM, "asof": str(asof.date()), "run_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "git": sha, "dirty": dirty,
           "prereg": PREREG, "cfg": CFG, "fwd_start": FWD_START,
           "data": {"minute_from": MINUTE_START, "panel_codes": len(codes), "union": len(D["union"]),
                    "cover_asof": round(len(expect.get(asof, set()) & set(D["m15"][D["m15"]["ts"] >= asof + pd.Timedelta(hours=15)]["code"]))
                                        / max(len(expect.get(asof, set())), 1), 4),
                    "krx_session": D["krx_session"]},
           "regime": {"asof": {"on": bool(RG["on"][last]), "idx_on": bool(RG["idx_on"][last]), "u1_share": _num(RG["share"][last], 4)},
                      "next": {k: (_num(v, 4) if isinstance(v, float) else v) for k, v in nr.items()}},
           "main": {**summary(B, "main"), "trades": rows["main"]}, "main_c1": summary(B, "main_c1"),
           "shadow": {**summary(B, "shadow"), "trades": rows["shadow"]}, "drift": dr, "newly_realized": nr_,
           "note": "가격은 분할 보정 패널 기준(판정은 비율만), regime.next 의 대상은 패널 종목 한정 근사"}
    append_row(path, row)
    s = row["main"]
    print(f"기록: {ARM} {asof.date()} — main 진입 {s['n_entries']}(실현 {s['n_realized']}, 보유 {s['n_open']}), 자산 {s['equity']:,.0f}, "
          f"다음 날 국면 {'ON' if nr['on'] else 'OFF'} ({time.time() - t0:.0f}s)")
    if dr["main"] or dr["shadow"]:
        print(f"감시: 표류 main {len(dr['main'])} shadow {len(dr['shadow'])} — 행 drift 참조")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
