"""30 독립 표본 실행기 — META_EVOLUTION.md '30 독립 표본 사전등록' + 보충 1~4, 정정 GP-1(leader_P 주 판정), '32 RL 단계'(RL-1 평가).

판정 구간 2009-01-02 ~ 2016-08-31(2008 = 준비 기간). 가격 = daily_bars_adj_v2(상장 = 원 일봉 × 누적계수, 폐지 = 네이버 수정가).
이 파일은 데이터가 다 들어온 뒤 **한 번만** 돈다(사전등록). ``--smoke`` 는 이미 본 2016~2017 구간에서 코드가 도는지만 보고 성적을 내지 않는다.
사용: ``uv run --with numba --with lightgbm --extra pg python research/experiments/indep30.py --out <dir> [--smoke]``
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

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
ARMS = {"A_open": {**B23, **A23["A_open"]}, "C_close": {**B23, **A23["C_close"]},
        "leader_M": P.ARMS["leader_M"], "leader_P": P.ARMS["leader_P"], "leader_P_RL": P.ARMS["leader_P_RL"]}
_G: dict = {}


def load(con, data: tuple) -> dict:
    """v2 → 종목×날짜 배열, 시총(원 종가 또는 수정 종가 × 주식수 knowledge_date as-of), 공식 KOSPI 국면."""
    from swing_it.storage import read_index, read_prices, read_shares
    v2 = read_prices(con, cols=("code", "date", "open", "high", "low", "close", "volume", "trade_value", "raw_close"), table="daily_bars_adj_v2",
                     start=data[0], end=data[1], parse_dates=True, session="raw")   # v2 의 09-14~ 는 이미 분봉 정규장 값(close_src) — 판정 구간 밖
    for k in ("open", "high", "low"):                # 네이버 정지일 O·H·L = 0 → NaN(체결 불가, 30 보충 2)
        v2.loc[v2[k] <= 0, k] = np.nan
    days = pd.DatetimeIndex(sorted(v2["date"].unique()))
    codes = sorted(v2["code"].astype(str).unique())

    def wide(col):
        return v2.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=codes).to_numpy(float).T
    O, H, L_, C, V, TV, RC = (wide(k) for k in ("open", "high", "low", "close", "volume", "trade_value", "raw_close"))  # noqa: E741
    TR = np.isfinite(C) & (np.nan_to_num(V) > 0)
    sh = read_shares(con, parse_dates=True)
    sh["date"] = sh["knowledge_date"].fillna(sh["date"])
    sh = sh[(sh["shares_outstanding"] > 0) & (sh["shares_outstanding"] <= 1e10)]
    shw = sh.pivot_table(index="date", columns="code", values="shares_outstanding", aggfunc="last")
    shw = shw.reindex(index=shw.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes).to_numpy(float).T
    px_cap = np.where(np.isfinite(RC), RC, C)       # 폐지 종목은 원 종가가 없다 — 수정 종가(분할 뒤 틀릴 수 있음, 보충 2 에 공개)
    cap = px_cap * shw
    cost = np.full((2, len(days)), COST_RT)
    mkt = np.zeros(len(codes), np.int64)
    lu = np.where(days < pd.Timestamp(LU_CUT), 1.15, 1.29)
    kx = read_index(con, "001", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,
                    proxy="30 독립 표본: 공식 KOSPI(2008~) — 없으면 시총 가중 대용").set_index("date")["close"]
    return {"days": days, "codes": codes, "O": O, "H": H, "L": L_, "C": C, "TR": TR, "tv": TV, "cap": cap, "cost": cost, "mkt": mkt,
            "lu": lu, "kx": kx, "delisted_cap_from_adj": int((~np.isfinite(RC) & np.isfinite(C)).sum())}


def build(con, data: tuple) -> dict:
    X = load(con, data)
    D = L.build(X["days"], X["codes"], X["O"], X["H"], X["L"], X["C"], X["TR"], X["cost"], X["mkt"], con, pit_shares=True,
                tv_wide=X["tv"], cap_wide=X["cap"])
    D["lu"], D["tv"] = X["lu"], X["tv"]
    kx, days = X["kx"], X["days"]
    kma = kx.rolling(200, min_periods=200).mean()
    on = (kx >= kma).where(kma.notna()).reindex(days)
    D["regime_src"] = "official" if on.notna().mean() > 0.9 else "proxy"
    if D["regime_src"] == "official":
        D["regime_on"] = on.ffill().fillna(False).to_numpy(bool)
    D["diag"] = {"n_codes": len(X["codes"]), "n_days": len(days), "delisted_cap_from_adj": X["delisted_cap_from_adj"],
                 "regime_src": D["regime_src"], "official_cover": float(on.notna().mean())}
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


def run(arm: str, cfg: dict, cm: float, judge: tuple) -> "tuple[pd.Series, pd.DataFrame]":
    D = _G["D"]
    eq, tr = L.simulate(D, cfg, cm, end=judge[1], start=judge[0])
    if len(tr):
        tr = tr[(tr["entry"] >= judge[0]) & (tr["entry"] <= judge[1])]
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
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    D = build(con, data)
    from research.experiments.rl32 import cand_mask, score_array
    D["score_rl32"] = score_array(D, ARMS["leader_P"], con)
    _G["D"] = D
    out: dict = {"diag": D["diag"], "judge": judge, "arms": {}}
    trades = {}
    for arm, cfg in ARMS.items():
        res = {}
        for cm in (1.0, 2.0):
            eq, tr = run(arm, cfg, cm, judge)
            m = metrics(eq, *judge)
            r = {"cagr": m["cagr"], "mdd": m["mdd"], "n": int(len(tr)), "by_year": m["by_year"]}
            if cm == 2.0:
                trades[arm] = tr
                tr.to_csv(os.path.join(a.out, f"trades_{arm}_2x.csv.gz"), index=False)
                em = expo_market(D, eq)
                mk = pd.Series(np.cumprod(1 + D["cw"]), index=D["days"]).reindex(eq.index)
                r["expo_mkt_cagr"] = metrics(em, *judge)["cagr"]
                r["alpha_m"], r["alpha_t"] = alpha_t(eq, mk, *judge)
                r["halves"] = [{"cagr": metrics(eq, lo, hi)["cagr"], "expo_mkt": metrics(em, lo, hi)["cagr"]} for lo, hi in halves]
                R = r_of(arm, tr) if len(tr) else pd.Series(dtype=float)
                yrs = (pd.Timestamp(judge[1]) - pd.Timestamp(judge[0])).days / 365.25
                r.update(R_mean=float(R.mean()) if len(R) else float("nan"), R_boot_lo=boot_lo(R.to_numpy()), R_sum_per_year=float(R.sum() / yrs),
                         R_halves=[float(R[(tr["entry"] >= lo) & (tr["entry"] < hi)].mean()) for lo, hi in halves])
                if arm == "C_close":
                    r["drop_top5_cagr"] = drop_top_cagr(arm, cfg, cm, tr, 5, judge)
                    r["drop_top10_cagr"] = drop_top_cagr(arm, cfg, cm, tr, 10, judge)
                if arm in ("leader_M", "leader_P"):
                    r["drop_top5_rsum"] = drop_top_rsum(arm, cfg, cm, tr, 5, judge)
                pd.DataFrame({"eq": eq, "expo_mkt": em}).to_csv(os.path.join(a.out, f"eq_{arm}_2x.csv"))
            res[f"{int(cm)}x"] = r
        out["arms"][arm] = res
    # 널(정정 GP-1 · 32): P 통과율 = 판정 구간 M 기준 후보 중 P 필터 통과 비율
    mP, _, _ = cand_mask(D, ARMS["leader_P"])
    base = D["cand"] & D["brk"][20] & (D["rs"] >= 70) & D["regime_on"][None, :]
    j0, j1 = D["days"].searchsorted(pd.Timestamp(judge[0])), D["days"].searchsorted(pd.Timestamp(judge[1]), side="right")
    _G["pass_rate"] = float(mP[:, j0:j1].sum() / max(base[:, j0:j1].sum(), 1))
    with mp.get_context("fork").Pool(a.workers) as pool:
        thin = pool.map(_null_task, [("thin", s, judge) for s in range(n_null)])
        rand = pool.map(_null_task, [("rand", s, judge) for s in range(n_null)])
    pd.DataFrame(thin).to_csv(os.path.join(a.out, "null_thin.csv"), index=False)
    pd.DataFrame(rand).to_csv(os.path.join(a.out, "null_randscore.csv"), index=False)
    out["null"] = {"pass_rate": _G["pass_rate"], "thin_rsum_p95": float(np.quantile([x["rsum"] for x in thin], 0.95)),
                   "thin_mdd_median": float(np.median([x["mdd"] for x in thin])),
                   "rand_rsum_p95": float(np.quantile([x["rsum"] for x in rand], 0.95))}
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
        "GP1_P": {"b1": P2["R_sum_per_year"] * 0 + float(RP.sum()) > out["null"]["thin_rsum_p95"], "b2": P2["drop_top5_rsum"] > 0,
                  "b3": P2["mdd"] > out["null"]["thin_mdd_median"]},
        "32_RL1": {"b1": float(np.quantile(diff, 0.05)) > 0, "b2": float(RPR.sum()) > out["null"]["rand_rsum_p95"], "b3": PR2["n"] >= 0.9 * P2["n"],
                   "diff_lo": float(np.quantile(diff, 0.05))},
    }
    out["bars"] = {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv) for kk, vv in v.items()} for k, v in out["bars"].items()}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    if a.smoke:                                       # 성적은 출력하지 않는다 — 경로 점검만
        print(json.dumps({"diag": D["diag"], "n_trades": {k: v["2x"]["n"] for k, v in g.items()}, "pass_rate": _G["pass_rate"],
                          "null_n": len(thin)}, ensure_ascii=False))
    else:
        print(json.dumps(out["bars"], ensure_ascii=False))


if __name__ == "__main__":
    main()
