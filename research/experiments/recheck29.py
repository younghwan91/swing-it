"""29 재확인(META_EVOLUTION.md '29 재확인 사전등록' + 보충 — 결과 전 고정): 재구성 수정 주가(daily_bars_adj_v2) 위 23세대 A·C 단 한 번.

v13 캐시의 날짜·종목·TR·비용표는 그대로, O·H·L·C 만 v2 로 교체(종목별 적용률 ≥ 99% 일 때만, 아니면 캐시 가격). 시총 = 원 종가 × 분할 일관
주식수(F = v2 종가 / 원 종가, ``krx_session.factor_cap``). 카나리아가 하나라도 실패하면 실행하지 않는다. 결과와 무관하게 이 데이터로 새 세대 없음.

사용: ``uv run --with numba --extra pg python research/experiments/recheck29.py --cache <v13> --out <dir>`` (KR_QUANT_DB)
      ``--repro-only`` = 현 캐시로 코드 경로 재현 카나리아만(G28-5 D2 C 2배 14.7%; 새 표 불요)
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
from research.experiments.leader23 import ARMS, BASE, WIN, drop_top  # noqa: E402

V2, RAW, CA = "daily_bars_adj_v2", "daily_bars_raw", "corporate_actions"
REPRO_C2 = 0.147                                    # G28-5 D2 C_close 2배 2017-10~2023
COVER_MIN = 0.99


def derive(C: np.ndarray, cap: np.ndarray) -> dict:
    """leader21.build 의 시총 파생(순위·시총 가중 시장·국면) — 같은 식."""
    cap = np.where(cap <= 2e15, cap, np.nan)
    crank = pd.DataFrame(cap.T).rank(axis=1, ascending=False).to_numpy().T
    r1 = np.concatenate([np.full((C.shape[0], 1), np.nan), C[:, 1:] / C[:, :-1] - 1], axis=1)
    cprev = np.concatenate([np.full((C.shape[0], 1), np.nan), cap[:, :-1]], axis=1)
    w = np.where(np.isfinite(r1) & np.isfinite(cprev) & (cprev > 0), cprev, 0.0)
    cw = np.where(w.sum(0) > 0, (np.nan_to_num(r1) * w).sum(0) / np.maximum(w.sum(0), 1e-12), 0.0)
    cw = np.clip(cw, -0.2, 0.2)
    cwi = np.cumprod(1 + cw)
    regime_on = cwi >= pd.Series(cwi).rolling(200, min_periods=200).mean().to_numpy()
    return {"cap": cap, "crank": crank, "cw": cw, "regime_on": regime_on}


def load_cache(cache: str):
    from research.signals import pattern_events as PE
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    codes = [str(c) for c in np.load(f"{cache}/codes.npy", allow_pickle=True)]
    Op, H, Lo, C = (np.load(f"{cache}/{k}.npy").astype(float) for k in ("O", "H", "L", "C"))
    TR = np.load(f"{cache}/TR.npy").astype(bool)
    cost = np.load(f"{cache}/cost_tab.npy")
    F = np.load(f"{cache}/F.npy", mmap_mode="r")
    I_ = np.load(f"{cache}/I.npy", mmap_mode="r")
    mk_of = dict(zip(np.asarray(I_[:, PE.II["ci"]]).astype(int), np.asarray(F[:, PE.FI["market"]]).astype(int), strict=True))
    mkt = np.array([mk_of.get(i, 0) for i in range(len(codes))], np.int64)
    return days, codes, Op, H, Lo, C, TR, cost, mkt


def win_metrics(eq: pd.Series) -> dict:
    return L.metrics(eq, *WIN)


def expo_matched(D: dict, eq: pd.Series) -> pd.Series:
    """전날 종가 위성 노출 × 시총 가중 시장 일수익."""
    ix = D["days"].get_indexer(eq.index)
    expo = np.asarray(eq.attrs["expo"], float)
    r = np.concatenate([[0.0], expo[:-1] * D["cw"][ix[1:]]])
    return pd.Series(np.cumprod(1 + r), index=eq.index)


def alpha_t(eq: pd.Series, mkt: pd.Series) -> "tuple[float, float]":
    """월 수익 OLS: 전략 = a + b·시장 → (a 월, t)."""
    x = pd.concat([eq, mkt], axis=1).loc[WIN[0]:WIN[1]]
    m = x.resample("ME").last().pct_change().dropna()
    y, X = m.iloc[:, 0].to_numpy(), np.column_stack([np.ones(len(m)), m.iloc[:, 1].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    s2 = e @ e / (len(y) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return float(beta[0]), float(beta[0] / se)


def run_arms(D: dict, tag: str, out_dir: str) -> dict:
    res = {}
    for nm in ("A_open", "C_close"):
        cfg = {**BASE, **ARMS[nm]}
        for cm in (1.0, 2.0):
            eq, tr = L.simulate(D, cfg, cm)
            trw = tr[(tr["entry"] >= WIN[0]) & (tr["entry"] < WIN[1])]
            m = win_metrics(eq)
            m["n_trades"] = int(len(trw))
            if cm == 2.0:
                m["drop_top5_cagr"], m["top5"] = drop_top(D, cfg, cm, trw, 5)
                m["drop_top10_cagr"], _ = drop_top(D, cfg, cm, trw, 10)
                m["drop_scan"] = {k: drop_top(D, cfg, cm, trw, k)[0] for k in range(3, 9)}   # 정정 R29-1: 판정 무관 보고 열
                em = expo_matched(D, eq)
                m["expo_mkt_cagr"] = win_metrics(em)["cagr"]
                mk = pd.Series(np.cumprod(1 + D["cw"]), index=D["days"]).reindex(eq.index)
                m["alpha_m"], m["alpha_t"] = alpha_t(eq, mk)
                c = (trw["ret"] * trw["w_peak"]).groupby(trw["code"]).sum().sort_values(ascending=False)
                m["contrib_top5"] = {k: round(float(v), 4) for k, v in c.head(5).items()}
                tr.to_csv(os.path.join(out_dir, f"trades_{tag}_{nm}_2x.csv.gz"), index=False)
                eq.to_frame("eq").assign(expo_mkt=em).to_csv(os.path.join(out_dir, f"eq_{tag}_{nm}_2x.csv"))
            res[f"{nm}_{int(cm)}x"] = m
    B = L.benchmarks(D)
    res["bench"] = {k: win_metrics(B[k])["cagr"] for k in B.columns}
    mc = res["bench"]["market_cw"]
    res["mkt_multiple_C2"] = res["C_close_2x"]["cagr"] / mc if mc > 0 else float("nan")   # 기록만(2017~23 시장 ≈ 0 → 판정에 안 씀)
    return res


def bars(r: dict) -> dict:
    a, c = r["A_open_2x"], r["C_close_2x"]
    yd = [c["by_year"][y] - a["by_year"][y] for y in range(2017, 2024) if y in c["by_year"] and y in a["by_year"]]
    b1 = (c["cagr"] >= a["cagr"] + 0.02 and c["drop_top5_cagr"] >= a["drop_top5_cagr"] + 0.01
          and sum(d > 0 for d in yd) >= 4 and c["mdd"] >= a["mdd"] - 0.05)
    b2 = c["drop_top5_cagr"] > c["expo_mkt_cagr"]
    b3 = c["alpha_t"] >= 2.0
    verdict = "유지(시장 알파 포함)" if b1 and b2 and b3 else "실행 개선만 유지" if b1 else "23세대 통과는 데이터 결함 산물"
    return {"bar1": bool(b1), "bar2": bool(b2), "bar3": bool(b3), "year_diff": [round(d, 4) for d in yd], "verdict": verdict}


def main() -> None:
    import psycopg2

    from swing_it.krx_session import factor_cap, read_raw_close
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repro-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    days, codes, Op, H, Lo, C, TR, cost, mkt = load_cache(a.cache)

    # 카나리아 1: 코드 경로 재현(현 캐시, G28-5 D2) + 파생 함수 동일성
    D2 = L.build(days, codes, Op, H, Lo, C, TR, cost, mkt, con, raw_close=True)
    dv = derive(D2["C"], D2["cap"])
    same = all(np.array_equal(dv[k], D2[k], equal_nan=True) for k in ("crank", "cw", "regime_on"))
    c2 = win_metrics(L.simulate(D2, {**BASE, **ARMS["C_close"]}, 2.0)[0])["cagr"]
    can = {"repro_C2": round(c2, 4), "repro_ok": abs(c2 - REPRO_C2) <= 0.003, "derive_identical": bool(same)}
    print("canary repro", can)
    if a.repro_only or not (can["repro_ok"] and same):
        json.dump({"canary": can}, open(os.path.join(a.out, "canary.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(0 if a.repro_only and can["repro_ok"] and same else 1)

    cur = con.cursor()
    cur.execute("select table_name from information_schema.tables where table_name in (%s, %s, %s)", (V2, RAW, CA))
    have = {r[0] for r in cur.fetchall()}
    if have != {V2, RAW, CA}:
        print("새 표 없음:", sorted({V2, RAW, CA} - have), "— 실행 안 함")
        sys.exit(2)
    v2 = pd.read_sql(f"select code, date, open, high, low, close from {V2} where date >= '2016-08-01'", con, parse_dates=["date"])
    rw = pd.read_sql(f"select code, date, close from {RAW} where date >= '2016-08-01'", con, parse_dates=["date"])
    ca = pd.read_sql(f"select code, date from {CA}", con, parse_dates=["date"])
    W = {k: v2.pivot_table(index="date", columns="code", values=k, aggfunc="last").reindex(index=days, columns=codes)
         for k in ("open", "high", "low", "close")}

    # 카나리아 2: 데이터
    def px(tab, code, d):
        s = tab.get(code)
        return float(s.get(pd.Timestamp(d), np.nan)) if s is not None else np.nan
    rwp = rw.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    rc = read_raw_close(con, start="2016-08-01")
    rcp = rc.pivot_table(index="date", columns="code", values="close", aggfunc="last")
    raw = rwp.reindex(index=days, columns=codes).combine_first(rcp.reindex(index=days, columns=codes))
    vc = W["close"]
    k1 = abs(px(vc, "005930", "2018-05-04") / px(vc, "005930", "2018-05-03") - 1)
    k2 = px(raw, "005930", "2017-03-31")
    k3 = abs(px(vc, "001230", "2021-07-29") / vc["001230"].loc[:"2021-07-28"].dropna().iloc[-1] - 1)
    can.update(samsung_split_ret=k1, samsung_raw_2017=k2, seam_001230=k3)
    can["data_ok"] = bool(k1 < 0.3 and abs(k2 / 2_060_000 - 1) <= 0.03 and k3 <= 0.3)
    print("canary data", can)
    json.dump({"canary": can}, open(os.path.join(a.out, "canary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    if not can["data_ok"]:
        sys.exit(1)

    # 가격 교체(종목별 적용률)
    lo, hi = days.searchsorted(pd.Timestamp("2016-08-01")), days.searchsorted(pd.Timestamp(WIN[1]))
    valid = np.isfinite(C[:, lo:hi])
    vcv = vc.to_numpy().T
    cover = (np.isfinite(vcv[:, lo:hi]) & valid).sum(1) / np.maximum(valid.sum(1), 1)
    use = (cover >= COVER_MIN) & (valid.sum(1) > 0)
    Cn = pd.DataFrame(vcv.T).ffill().to_numpy().T
    def pick(k, old):
        x = W[k].to_numpy().T
        x = np.where(np.isfinite(x), x, Cn)
        return np.where(use[:, None], x, old)
    O2, H2, L2, C2 = pick("open", Op), pick("high", H), pick("low", Lo), np.where(use[:, None], Cn, C)
    DN = L.build(days, codes, O2, H2, L2, C2, TR, cost, mkt, con, raw_close=False)
    cur.execute("select column_name from information_schema.columns where table_name = 'shares_outstanding_history'")
    has_orig = "orig_rcept_dt" in {r[0] for r in cur.fetchall()}
    sh = pd.read_sql("select code, date, shares_outstanding" + (", orig_rcept_dt" if has_orig else "") + " from shares_outstanding_history",
                     con, parse_dates=["date"] + (["orig_rcept_dt"] if has_orig else []))
    sh = sh[(sh["shares_outstanding"] > 0) & (sh["shares_outstanding"] <= 1e10)]
    if has_orig:                                    # 사전등록: 원 정기보고서 접수일 다음 날부터(없는 행은 date)
        sh["asof"] = pd.to_datetime(sh["orig_rcept_dt"]) + pd.Timedelta(days=1)
    diag_asof = "orig_rcept_dt+1" if has_orig else "date(≈3개월 미래 정보)"
    capv2, Fw = factor_cap(raw, pd.DataFrame(C2.T, index=days, columns=codes), sh, asof="asof" if has_orig else "date")
    cap = np.where(use[:, None], capv2.to_numpy().T, D2["cap"])     # v2 없는 종목 = D2 시총 그대로(검증 29 MINOR)
    DN.update(derive(C2, cap))
    # 카나리아 3(정정 R29-1, 검증 29 M2): 시총 경로 — 상위 200 자카드(대 D2) 평균 ≥ 0.85 ∧ 시총 점프 위반 ≤ D2
    w0, w1 = days.searchsorted(pd.Timestamp(WIN[0])), days.searchsorted(pd.Timestamp(WIN[1]))
    t200n, t200o = DN["crank"][:, w0:w1] <= 200, D2["crank"][:, w0:w1] <= 200
    jac = float(np.nanmean((t200n & t200o).sum(0) / np.maximum((t200n | t200o).sum(0), 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        rw_ = raw.to_numpy(float).T
        jump, r1 = DN["cap"][:, 1:] / DN["cap"][:, :-1], rw_[:, 1:] / rw_[:, :-1]
        n_bad = int((((jump > 3) | (jump < 1 / 3)) & (r1 < 1.3) & (r1 > 1 / 1.3)).sum())
    can.update(cap_jaccard_vs_D2=jac, cap_jump_bad=n_bad, cap_jump_bad_D2=int(D2["cap_jump_bad"]), shares_asof=diag_asof)
    can["cap_ok"] = bool(jac >= 0.85 and n_bad <= D2["cap_jump_bad"])
    print("canary cap", can)
    json.dump({"canary": can}, open(os.path.join(a.out, "canary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    if not can["cap_ok"]:
        sys.exit(3)                                 # 판정 전에 원인 조사(실행 안 함)
    # 진단: corporate_actions 밖 F 변화
    with np.errstate(divide="ignore", invalid="ignore"):
        dF = np.abs(np.log(Fw / Fw.shift(1)))
    evset = set(zip(ca["code"], ca["date"]))
    st = dF.stack()
    st = st[st > 0.01]
    off = int(sum((c, d) not in evset for d, c in st.index))
    diag = {"shares_asof": diag_asof, "n_codes": len(codes), "v2_codes": int(use.sum()), "fallback_codes": int((~use & (valid.sum(1) > 0)).sum()),
            "F_change_outside_events": off, "F_change_total": int(len(st))}

    D0 = L.build(days, codes, Op, H, Lo, C, TR, cost, mkt, con, raw_close=False)
    out = {"canary": can, "diag": diag, "v2": run_arms(DN, "v2", a.out), "D2_old": run_arms(D2, "D2", a.out),
           "D0_defect": run_arms(D0, "D0", a.out)}
    out["judge"] = bars(out["v2"])
    fb_codes = {c for c, u, v in zip(codes, use, valid.sum(1) > 0, strict=True) if not u and v}
    trc = pd.read_csv(os.path.join(a.out, "trades_v2_C_close_2x.csv.gz"), dtype={"code": str})
    trc = trc[(trc["entry"] >= WIN[0]) & (trc["entry"] < WIN[1])]
    diag["fallback_trades_C2"] = int(trc["code"].isin(fb_codes).sum())
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"diag": diag, "judge": out["judge"],
                      **{f"{t}|{k}": (round(v["cagr"], 4), round(v["mdd"], 4), v["n_trades"]) for t in ("v2", "D2_old", "D0_defect")
                         for k, v in out[t].items() if k.endswith("x")}}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
