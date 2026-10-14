"""30 독립 표본 실행기 — META_EVOLUTION.md '30 독립 표본 사전등록' + 보충 1~10, 정정 GP-1(leader_P 주 판정), '32 RL 단계'(RL-1 평가).

판정 구간 2009-01-02 ~ 2016-08-31(2008 = 준비 기간). 가격 = daily_bars_adj_v2(상장 = 원 일봉 × 누적계수, 폐지 = 네이버 수정가;
close_src='pre_listing' 행 제외). 실적 = earnings_dart_doc 한 원천(eps_asof source, 보충 7~9). 주식수 = earnings_dart_doc.shares_common
knowledge_date as-of(shares_outstanding_history 는 2016-03 부터라 이 구간에 없다). 정지(전날 거래량 0) 뒤 재개 첫날 수익률은 시총 가중
시장 수익에서 NaN(인계 주의); 그 날을 보유 중에 지난 거래는 ``susp`` 로 표시해 뺀 결과를 보고 열로 낸다(판정은 포함).
이 파일은 데이터가 다 들어온 뒤 **한 번만** 돈다(사전등록). ``--smoke`` 는 이미 본 2016~2017 구간에서 코드가 도는지만 보고 성적을 내지 않는다.
사용: ``uv run --with numba --with lightgbm --extra pg python research/experiments/indep30.py --out <dir> [--smoke]``
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]
import research.experiments.leader21 as L  # noqa: E402
import research.paper.leader_paper as P  # noqa: E402
from research.experiments.leader23 import ARMS as A23  # noqa: E402
from research.experiments.leader23 import BASE as B23  # noqa: E402

JUDGE = ("2009-01-02", "2016-08-31")
HALVES = (("2009-01-02", "2013-01-01"), ("2013-01-01", "2016-09-01"))
DATA_START = "2008-01-01"
SMOKE = {"data": ("2016-09-01", "2017-12-29"), "judge": ("2017-09-01", "2017-12-29"),
         "halves": (("2017-09-01", "2017-11-01"), ("2017-11-01", "2017-12-30"))}
COST_RT = 0.00515                                  # 30 사전등록: 2016 전 왕복(매도세 0.3% 시대) — 시장 공통
LU_CUT = "2015-06-15"                              # 가격제한폭 ±15% → ±30%
N_NULL = 200
UNIT_RISK_P = 0.0025
EPS_SOURCE = "earnings_dart_doc"                   # 30 보충 7: 판정 구간 전체 한 원천(2016 분기도 earnings 와 섞지 않는다)
ARMS = {"A_open": {**B23, **A23["A_open"]}, "C_close": {**B23, **A23["C_close"]},
        "leader_M": P.ARMS["leader_M"], "leader_P": P.ARMS["leader_P"], "leader_P_RL": P.ARMS["leader_P_RL"]}
_G: dict = {}
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.0f}s] {msg}", flush=True)


def load(con, data: tuple, eps_source: str = EPS_SOURCE) -> dict:
    """v2 → 종목×날짜 배열, 시총(원 종가 또는 수정 종가 × 주식수 knowledge_date as-of), 공식 KOSPI 국면, 정지 재개 마스크."""
    from swing_it.storage import read_earnings, read_index, read_prices, read_shares
    v2 = read_prices(con, cols=("code", "date", "open", "high", "low", "close", "volume", "trade_value", "raw_close", "close_src", "source"),
                     table="daily_bars_adj_v2", start=data[0], end=data[1], parse_dates=True, session="raw")   # 09-14~ 는 판정 구간 밖
    n_pre = int((v2["close_src"] == "pre_listing").sum())
    v2 = v2[v2["close_src"] != "pre_listing"]        # KRX 상장 전 K-OTC·프리보드 시세(OHLCV NULL) 제외(인계 10-14)
    for k in ("open", "high", "low"):                # 네이버 정지일 O·H·L = 0 → NaN(체결 불가, 30 보충 2)
        v2.loc[v2[k] <= 0, k] = np.nan
    days = pd.DatetimeIndex(sorted(v2["date"].unique()))
    codes = sorted(v2["code"].astype(str).unique())
    delisted = sorted(v2.loc[v2["source"] == "naver_adj", "code"].astype(str).unique())

    def wide(col):
        return v2.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=codes).to_numpy(float).T
    O, H, L_, C, V, TV, RC = (wide(k) for k in ("open", "high", "low", "close", "volume", "trade_value", "raw_close"))  # noqa: E741
    TR = np.isfinite(C) & (np.nan_to_num(V) > 0)
    if eps_source == "earnings_dart_doc":           # 주식수 = DART 원문 보고서의 보통주 수, 접수일(knowledge_date) as-of(보충 5 '주식수 knowledge_date as-of')
        sh = read_earnings(con, cols=("code", "period", "knowledge_date", "shares_common"), all_versions=True, parse_dates=True,
                           table="earnings_dart_doc")
        sh = sh[(sh["shares_common"] > 0) & (sh["shares_common"] <= 1e10)].sort_values(["knowledge_date", "period"], kind="stable")
        shw = sh.pivot_table(index="knowledge_date", columns="code", values="shares_common", aggfunc="last")
        shares_src = "earnings_dart_doc.shares_common@knowledge_date"
    else:                                           # 스모크(2016~2017): 기존 주식수 표
        sh = read_shares(con, parse_dates=True)
        sh["date"] = sh["knowledge_date"].fillna(sh["date"])
        sh = sh[(sh["shares_outstanding"] > 0) & (sh["shares_outstanding"] <= 1e10)]
        shw = sh.pivot_table(index="date", columns="code", values="shares_outstanding", aggfunc="last")
        shares_src = "shares_outstanding_history@knowledge_date"
    shw = shw.reindex(index=shw.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes).to_numpy(float).T
    px_cap = np.where(np.isfinite(RC), RC, C)       # 폐지 종목은 원 종가가 없다 — 수정 종가(분할 뒤 틀릴 수 있음, 보충 2 에 공개)
    cap = px_cap * shw
    cost = np.full((2, len(days)), COST_RT)
    mkt = np.zeros(len(codes), np.int64)
    lu = np.where(days < pd.Timestamp(LU_CUT), 1.15, 1.29)
    kx = read_index(con, "001", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,
                    proxy="30 독립 표본: 공식 KOSPI(2008~) — 없으면 시총 가중 대용").set_index("date")["close"]
    pad = np.full((C.shape[0], 1), np.nan)
    Cp, Vp = np.concatenate([pad, C[:, :-1]], axis=1), np.concatenate([pad, V[:, :-1]], axis=1)
    reopen = np.isfinite(C) & np.isfinite(Cp) & ~(np.nan_to_num(Vp) > 0)     # 정지(전날 거래량 0 또는 결측) 뒤 재개 첫날
    return {"days": days, "codes": codes, "O": O, "H": H, "L": L_, "C": C, "TR": TR, "tv": TV, "cap": cap, "cost": cost, "mkt": mkt,
            "lu": lu, "kx": kx, "reopen": reopen, "delisted": delisted, "shares_src": shares_src, "n_pre_listing_rows": n_pre,
            "delisted_cap_from_adj": int((~np.isfinite(RC) & np.isfinite(C)).sum()),
            "cap_missing_share": float((~np.isfinite(cap) & np.isfinite(C)).sum() / max(np.isfinite(C).sum(), 1))}


def market_cw(C: np.ndarray, cap: np.ndarray, drop: "np.ndarray | None" = None) -> np.ndarray:
    """leader21.build 의 시총 가중 시장 일수익(전날 시총 가중, ±20% 클립) — ``drop`` 칸(정지 재개 첫날)의 수익률은 NaN(가중 0)."""
    cap = np.where(cap <= 2e15, cap, np.nan)
    r1 = np.concatenate([np.full((C.shape[0], 1), np.nan), C[:, 1:] / C[:, :-1] - 1], axis=1)
    if drop is not None:
        r1 = np.where(drop, np.nan, r1)
    cprev = np.concatenate([np.full((C.shape[0], 1), np.nan), cap[:, :-1]], axis=1)
    w = np.where(np.isfinite(r1) & np.isfinite(cprev) & (cprev > 0), cprev, 0.0)
    cw = np.where(w.sum(0) > 0, (np.nan_to_num(r1) * w).sum(0) / np.maximum(w.sum(0), 1e-12), 0.0)
    return np.clip(cw, -0.2, 0.2)


def build(con, data: tuple, eps_source: str = EPS_SOURCE) -> dict:
    X = load(con, data, eps_source)
    log(f"load: {len(X['codes'])} codes × {len(X['days'])} days, delisted {len(X['delisted'])}, shares={X['shares_src']}")
    D = L.build(X["days"], X["codes"], X["O"], X["H"], X["L"], X["C"], X["TR"], X["cost"], X["mkt"], con, pit_shares=True,
                tv_wide=X["tv"], cap_wide=X["cap"], eps_source=eps_source)
    D["lu"], D["tv"], D["reopen"], D["delisted"] = X["lu"], X["tv"], X["reopen"], X["delisted"]
    cw0 = D["cw"]
    D["cw"] = market_cw(D["C"], D["cap"], drop=X["reopen"])          # 인계 주의: 정지 뒤 재개 첫날 수익률 버림(시장 대용·알파 회귀·대용 국면)
    kx, days = X["kx"], X["days"]
    kma = kx.rolling(200, min_periods=200).mean()
    on = (kx >= kma).where(kma.notna()).reindex(days)
    D["regime_src"] = "official" if on.notna().mean() > 0.9 else "proxy"
    if D["regime_src"] == "official":
        D["regime_on"] = on.ffill().fillna(False).to_numpy(bool)
    else:
        cwi = np.cumprod(1 + D["cw"])
        D["regime_on"] = cwi >= pd.Series(cwi).rolling(200, min_periods=200).mean().to_numpy()
    D["kx"] = kx.reindex(days).ffill()
    D["diag"] = {"n_codes": len(X["codes"]), "n_days": len(days), "delisted_codes": len(X["delisted"]), "delisted_cap_from_adj": X["delisted_cap_from_adj"],
                 "cap_missing_share": X["cap_missing_share"], "shares_src": X["shares_src"], "n_pre_listing_rows": X["n_pre_listing_rows"],
                 "eps_source": eps_source, "regime_src": D["regime_src"], "official_cover": float(on.notna().mean()),
                 "reopen_cells": int(X["reopen"].sum()), "cw_mask_abs_diff_sum": float(np.abs(D["cw"] - cw0).sum()),
                 "regime_on_share": float(D["regime_on"].mean())}
    return D


def metrics(eq: pd.Series, lo: str, hi: str) -> dict:
    x = eq[(eq.index >= lo) & (eq.index < hi)]
    return L.metrics(x / x.iloc[0])


def expo_market(D: dict, eq: pd.Series) -> pd.Series:
    ix = D["days"].get_indexer(eq.index)
    expo = np.asarray(eq.attrs["expo"], float)
    r = np.concatenate([[0.0], expo[:-1] * D["cw"][ix[1:]]])
    return pd.Series(np.cumprod(1 + r), index=eq.index)


def alpha_t(eq: pd.Series, mk: pd.Series, lo: str, hi: str) -> "tuple[float, float]":
    m = pd.concat([eq, mk], axis=1).loc[lo:hi].resample("ME").last().pct_change().dropna()
    y, X = m.iloc[:, 0].to_numpy(), np.column_stack([np.ones(len(m)), m.iloc[:, 1].to_numpy()])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(e @ e / (len(y) - 2) * np.linalg.inv(X.T @ X)[0, 0])
    return float(b[0]), float(b[0] / se)


def r_of(arm: str, tr: pd.DataFrame) -> pd.Series:
    """M·C·A = 투입 R(ret × units / 0.15, 30 보충 4) · P·P_RL = 손익 ÷ 자산 0.25%(정정 GP-1 M4)."""
    if arm in ("leader_P", "leader_P_RL"):
        return tr["ret"] * tr["units"] * tr["unit_w"] / UNIT_RISK_P
    return tr["ret"] * tr["units"] / 0.15


def mark_susp(D: dict, tr: pd.DataFrame) -> pd.DataFrame:
    """보유 중(진입 다음 날 ~ 청산일)에 정지 재개 첫날을 지난 거래 → ``susp`` (인계 주의; 뺀 결과는 보고 열, 판정은 포함)."""
    if tr.empty:
        return tr.assign(susp=pd.Series(dtype=bool))
    ci = pd.Index(D["codes"]).get_indexer(tr["code"].astype(str))
    t0 = D["days"].get_indexer(pd.to_datetime(tr["entry"]))
    t1 = D["days"].get_indexer(pd.to_datetime(tr["exit"]))
    tr = tr.copy()
    tr["susp"] = [bool(D["reopen"][c, a + 1:b + 1].any()) for c, a, b in zip(ci, t0, t1, strict=True)]
    return tr


def run(arm: str, cfg: dict, cm: float, judge: tuple) -> "tuple[pd.Series, pd.DataFrame]":
    D = _G["D"]
    eq, tr = L.simulate(D, cfg, cm, end=judge[1], start=judge[0])
    if len(tr):
        tr = tr[(tr["entry"] >= judge[0]) & (tr["entry"] <= judge[1])]
    else:
        tr = pd.DataFrame(columns=["code", "entry", "exit", "why", "ret", "units", "hold", "w_peak", "stop_dist", "unit_w"])
    return eq, tr


def drop_top_cagr(arm, cfg, cm, tr, k, judge) -> float:
    if tr.empty:
        return float("nan")
    contrib = (tr["ret"] * tr["w_peak"]).groupby(tr["code"]).sum().sort_values(ascending=False)
    eq, _ = run(arm, {**cfg, "exclude": list(contrib.index[:k])}, cm, judge)
    return metrics(eq, *judge)["cagr"]


def drop_top_rsum(arm, cfg, cm, tr, k, judge) -> float:
    contrib = r_of(arm, tr).groupby(tr["code"]).sum().sort_values(ascending=False)
    _, t2 = run(arm, {**cfg, "exclude": list(contrib.index[:k])}, cm, judge)
    return float(r_of(arm, t2).sum()) if len(t2) else 0.0


def _null_task(args) -> dict:
    kind, seed, judge = args
    D = _G["D"]
    rng = np.random.default_rng(seed)
    if kind == "thin":                               # 정정 GP-1: P 필터 대신 같은 통과율 무작위 솎기(구조 손절 · 위험 비중 그대로, 손절 거리 상한 없음)
        D["null_mask"] = rng.random(D["C"].shape) < _G["pass_rate"]
        cfg = {**ARMS["leader_P"], "chase_max": None, "stop_max": 1.0, "cand_and": "null_mask"}
        eq, tr = run("leader_P", cfg, 2.0, judge)
        R = r_of("leader_P", tr) if len(tr) else pd.Series(dtype=float)
        x = eq[(eq.index >= judge[0])]
        return {"seed": seed, "rsum": float(R.sum()), "n": int(len(tr)), "mdd": float((x / x.cummax() - 1).min())}
    D["score_rand"] = np.where(np.isfinite(D["score_rl32"]), rng.random(D["C"].shape), np.nan)   # 32: 같은 순위 자리에 무작위 점수
    eq, tr = run("leader_P", {**ARMS["leader_P"], "rank_score": "score_rand"}, 2.0, judge)
    return {"seed": seed, "rsum": float(r_of("leader_P", tr).sum()) if len(tr) else 0.0, "n": int(len(tr))}


def boot_lo(x: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return float("nan")
    return float(np.quantile(rng.choice(x, (n, len(x))).mean(1), 0.05))


def arm_report(arm: str, cfg: dict, eq: pd.Series, tr: pd.DataFrame, D: dict, judge: tuple, halves: tuple) -> dict:
    """2배 비용 보고 열(사전등록 '보고 열' + 보충 2·4·10): 시장 대조 · 알파 t · 반기 · R · 상위 k 제거 · 기여 상위 10 · 폐지 · 정지."""
    cm = 2.0
    r: dict = {}
    em = expo_market(D, eq)
    mk = pd.Series(np.cumprod(1 + D["cw"]), index=D["days"]).reindex(eq.index)
    r["expo_mkt_cagr"] = metrics(em, *judge)["cagr"]
    r["mkt_cw_cagr"] = metrics(mk, *judge)["cagr"]
    r["alpha_m"], r["alpha_t"] = alpha_t(eq, mk, *judge)
    r["halves"] = [{"cagr": metrics(eq, lo, hi)["cagr"], "expo_mkt": metrics(em, lo, hi)["cagr"], "n": int(((tr["entry"] >= lo) & (tr["entry"] < hi)).sum())}
                   for lo, hi in halves]
    R = r_of(arm, tr) if len(tr) else pd.Series(dtype=float)
    yrs = (pd.Timestamp(judge[1]) - pd.Timestamp(judge[0])).days / 365.25
    r.update(R_mean=float(R.mean()) if len(R) else float("nan"), R_boot_lo=boot_lo(R.to_numpy()), R_sum=float(R.sum()), R_sum_per_year=float(R.sum() / yrs),
             R_halves=[float(R[(tr["entry"] >= lo) & (tr["entry"] < hi)].mean()) if len(tr) else float("nan") for lo, hi in halves],
             win_rate=float((tr["ret"] > 0).mean()) if len(tr) else float("nan"),
             R_by_year={int(y): float(v) for y, v in R.groupby(pd.to_datetime(tr["entry"]).dt.year).sum().items()} if len(tr) else {})
    if len(tr):
        susp = tr["susp"].to_numpy(bool)
        r["n_susp"] = int(susp.sum())
        r["R_mean_ex_susp"] = float(R[~susp].mean()) if (~susp).any() else float("nan")
        r["R_sum_ex_susp"] = float(R[~susp].sum())
        contrib = (tr["ret"] * tr["w_peak"]).groupby(tr["code"]).sum().sort_values(ascending=False)
        r["contrib_top10"] = {k: round(float(v), 4) for k, v in contrib.head(10).items()}
        r["R_contrib_top10"] = {k: round(float(v), 3) for k, v in R.groupby(tr["code"]).sum().sort_values(ascending=False).head(10).items()}
        dl = set(D["delisted"])
        r["n_delisted_trades"] = int(tr["code"].astype(str).isin(dl).sum())
    r["drop_top5_cagr"] = drop_top_cagr(arm, cfg, cm, tr, 5, judge)
    r["drop_top10_cagr"] = drop_top_cagr(arm, cfg, cm, tr, 10, judge)
    r["drop_top5_rsum"] = drop_top_rsum(arm, cfg, cm, tr, 5, judge) if len(tr) else 0.0
    if arm in ("A_open", "C_close"):
        r["drop_scan_cagr"] = {k: drop_top_cagr(arm, cfg, cm, tr, k, judge) for k in range(3, 9)}
        dl_traded = sorted(set(tr["code"].astype(str)) & set(D["delisted"])) if len(tr) else []
        if dl_traded:                              # 보충 2: 폐지 종목(수정가 × 주식수 시총) 거래를 뺀 결과 — 보고 열
            eq2, _ = run(arm, {**cfg, "exclude": dl_traded}, cm, judge)
            r["drop_delisted_cagr"] = metrics(eq2, *judge)["cagr"]
        else:
            r["drop_delisted_cagr"] = metrics(eq, *judge)["cagr"]
    r["mkt_multiple_expo"] = float(metrics(eq, *judge)["cagr"] / r["expo_mkt_cagr"]) if r["expo_mkt_cagr"] > 0 else float("nan")
    r["mkt_multiple_cw"] = float(metrics(eq, *judge)["cagr"] / r["mkt_cw_cagr"]) if r["mkt_cw_cagr"] > 0 else float("nan")
    return r


def main() -> None:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="이미 본 2016~2017 로 코드 경로만 점검(성적 미출력)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, judge, halves = (SMOKE["data"], SMOKE["judge"], SMOKE["halves"]) if a.smoke else ((DATA_START, JUDGE[1]), JUDGE, HALVES)
    n_null = 4 if a.smoke else N_NULL
    eps_source = "earnings" if a.smoke else EPS_SOURCE   # 스모크 구간(2016-09~)은 dart_doc 범위 밖 — 경로 점검만
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    D = build(con, data, eps_source)
    log(f"build: {D['diag']}")
    from research.experiments.rl32 import cand_mask, score_array
    D["score_rl32"] = score_array(D, ARMS["leader_P"], con)
    log("rl32 score done")
    _G["D"] = D
    j0, j1 = D["days"].searchsorted(pd.Timestamp(judge[0])), D["days"].searchsorted(pd.Timestamp(judge[1]), side="right")
    top = np.nan_to_num(D["crank"], nan=1e9)[:, j0:j1] <= 200
    dl = np.isin(np.asarray(D["codes"]), list(D["delisted"]))
    D["diag"]["delisted_in_top200_share"] = float((top & dl[:, None]).sum() / max(top.sum(), 1))
    base_all = D["uni"] & D["TR"]
    D["diag"]["eps_cand_share_by_year"] = {int(y): float(D["cand"][:, (D["days"].year == y)].sum() / max(base_all[:, (D["days"].year == y)].sum(), 1))
                                           for y in sorted(set(D["days"][j0:j1].year))}
    D["diag"]["kospi_official_cagr"] = metrics(D["kx"].astype(float), *judge)["cagr"] if D["regime_src"] == "official" else None
    out: dict = {"diag": D["diag"], "judge": judge, "cost_rt": COST_RT, "arms": {}}
    trades = {}
    for arm, cfg in ARMS.items():
        res = {}
        for cm in (1.0, 2.0):
            eq, tr = run(arm, cfg, cm, judge)
            tr = mark_susp(D, tr)
            m = metrics(eq, *judge)
            r = {"cagr": m["cagr"], "mdd": m["mdd"], "n": int(len(tr)), "by_year": m["by_year"], "total": m["total"], "sharpe_m": m["sharpe_m"]}
            if cm == 2.0:
                trades[arm] = tr
                tr.to_csv(os.path.join(a.out, f"trades_{arm}_2x.csv.gz"), index=False)
                r.update(arm_report(arm, cfg, eq, tr, D, judge, halves))
                pd.DataFrame({"eq": eq, "expo_mkt": expo_market(D, eq)}).to_csv(os.path.join(a.out, f"eq_{arm}_2x.csv"))
            res[f"{int(cm)}x"] = r
            log(f"arm {arm} {int(cm)}x: n={len(tr)}")
        out["arms"][arm] = res
    # 널(정정 GP-1 · 32): P 통과율 = 판정 구간 M 기준 후보 중 P 필터 통과 비율
    mP, _, _ = cand_mask(D, ARMS["leader_P"])
    base = D["cand"] & D["brk"][20] & (D["rs"] >= 70) & D["regime_on"][None, :]
    _G["pass_rate"] = float(mP[:, j0:j1].sum() / max(base[:, j0:j1].sum(), 1))
    with mp.get_context("fork").Pool(a.workers) as pool:
        thin = pool.map(_null_task, [("thin", s, judge) for s in range(n_null)])
        log(f"null thin {len(thin)} done")
        rand = pool.map(_null_task, [("rand", s, judge) for s in range(n_null)])
        log(f"null rand {len(rand)} done")
    pd.DataFrame(thin).to_csv(os.path.join(a.out, "null_thin.csv"), index=False)
    pd.DataFrame(rand).to_csv(os.path.join(a.out, "null_randscore.csv"), index=False)
    out["null"] = {"pass_rate": _G["pass_rate"], "n": n_null, "thin_rsum_p95": float(np.quantile([x["rsum"] for x in thin], 0.95)),
                   "thin_rsum_median": float(np.median([x["rsum"] for x in thin])), "thin_n_median": float(np.median([x["n"] for x in thin])),
                   "thin_mdd_median": float(np.median([x["mdd"] for x in thin])),
                   "rand_rsum_p95": float(np.quantile([x["rsum"] for x in rand], 0.95)), "rand_rsum_median": float(np.median([x["rsum"] for x in rand]))}
    # 판정(2배)
    g = out["arms"]
    C2, M2, P2, PR2 = g["C_close"]["2x"], g["leader_M"]["2x"], g["leader_P"]["2x"], g["leader_P_RL"]["2x"]
    RP, RPR = r_of("leader_P", trades["leader_P"]), r_of("leader_P_RL", trades["leader_P_RL"])
    rng = np.random.default_rng(1)
    diff = rng.choice(RPR.to_numpy(), (10000, len(RPR))).mean(1) - rng.choice(RP.to_numpy(), (10000, len(RP))).mean(1) if len(RP) and len(RPR) else np.array([np.nan])
    out["bars"] = {
        "30_C_alpha": {"b1": C2["cagr"] > C2["expo_mkt_cagr"] + 0.03, "b2": C2["drop_top5_cagr"] > C2["expo_mkt_cagr"], "b3": C2["alpha_t"] >= 2.0,
                       "b4": all(h["cagr"] - h["expo_mkt"] > 0 for h in C2["halves"])},
        "30s4_M": {"b1": M2["R_boot_lo"] > 0, "b2": M2["R_sum_per_year"] >= C2["R_sum_per_year"], "b3": M2["drop_top5_rsum"] > 0,
                   "b4": all(x > 0 for x in M2["R_halves"])},
        "GP1_P": {"b1": float(RP.sum()) > out["null"]["thin_rsum_p95"], "b2": P2["drop_top5_rsum"] > 0,
                  "b3": P2["mdd"] > out["null"]["thin_mdd_median"]},
        "32_RL1": {"b1": float(np.quantile(diff, 0.05)) > 0, "b2": float(RPR.sum()) > out["null"]["rand_rsum_p95"], "b3": PR2["n"] >= 0.9 * P2["n"],
                   "diff_lo": float(np.quantile(diff, 0.05))},
    }
    out["bars"] = {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv) for kk, vv in v.items()} for k, v in out["bars"].items()}
    out["elapsed_s"] = time.time() - _T0
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    if a.smoke:                                       # 성적은 출력하지 않는다 — 경로 점검만
        print(json.dumps({"diag": D["diag"], "n_trades": {k: v["2x"]["n"] for k, v in g.items()}, "pass_rate": _G["pass_rate"],
                          "null_n": len(thin)}, ensure_ascii=False, default=float))
    else:
        print(json.dumps(out["bars"], ensure_ascii=False))
    log("done")


if __name__ == "__main__":
    main()
