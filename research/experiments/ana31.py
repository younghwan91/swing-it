"""31 새 정보원 실행기 — META_EVOLUTION.md '31 새 정보원 사전등록' + 정정 31-1(F_ana 단일 · 투입 R) + 31 보충 2(커버리지 · 리포트 있는 후보 널).

F_ana(t) = [t−20, t−1] 에 알게 된(write_date 다음 거래일 ≤ t−1) 목표가 리포트 중 상향 건수 − 하향 건수 ≥ 1(목표가 · 직전 목표가 둘 다 > 0).
팔 = leader_M 설정 · 엔진 · 가격(leaderM_desc 와 같은 경로) 그대로에 ``cand_and = F_ana``. 리포트가 없으면 거짓.
판정 거래 = 진입일 2017-10-02 ~ 마지막 거래일(두 반기 2017-10~2023 · 2024~ 의 합 — leaderM_desc 와 같은 시작).
널 = 날마다 F_ana 와 같은 통과 비율로 leader_M 후보를 무작위로 남기는 필터 200(시드 0~199). 보고 널 = 20일 창에 리포트가 있는 후보 안에서 같은 방식.
이 파일은 naver_research_reports 수집 완료 신호 뒤 **한 번만** 돈다. ``--smoke`` 는 F 를 무작위 마스크로 바꿔 경로만 점검한다(성적 미출력).
사용: ``NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --with numba --extra pg python research/experiments/ana31.py --out <dir> [--smoke]``
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

JUDGE0 = "2017-10-02"
HALVES = (("2017-10-02", "2024-01-01"), ("2024-01-01", "2100-01-01"))
WIN = 20                                            # 신호 창(사전등록)
COV_WIN = (20, 90)                                  # 커버리지 보고 창(31 보충 2)
N_NULL = 200
MIN_TRADES, MIN_PER_YEAR = 30, 36
_G: dict = {}


def report_events(con, days: pd.DatetimeIndex, codes) -> "tuple[np.ndarray, np.ndarray]":
    """종목 × 행 배열 두 개: net(상향 +1 · 하향 −1, 알게 된 행에) · cnt(목표가 있는 리포트 수, 알게 된 행에)."""
    rr = pd.read_sql("select code, write_date, goal_price, prev_goal_price from naver_research_reports where goal_price is not null",
                     con, parse_dates=["write_date"])
    ix = pd.Index([str(c) for c in codes])
    rr = rr[rr["code"].astype(str).isin(ix)]
    ci = ix.get_indexer(rr["code"].astype(str))
    k = days.searchsorted(rr["write_date"].to_numpy(), side="right")      # write_date 다음 거래일 행
    ok = k < len(days)
    g, pg = rr["goal_price"].to_numpy(float), rr["prev_goal_price"].to_numpy(float)
    both = (g > 0) & (pg > 0)
    sgn = np.where(both & (g > pg), 1, np.where(both & (g < pg), -1, 0))
    net = np.zeros((len(ix), len(days)), np.int32)
    cnt = np.zeros((len(ix), len(days)), np.int32)
    np.add.at(net, (ci[ok], k[ok]), sgn[ok])
    np.add.at(cnt, (ci[ok], k[ok]), 1)
    return net, cnt


def window_sum(a: np.ndarray, w: int) -> np.ndarray:
    """out[:, t] = a[:, t−w : t].sum() — t−1 까지(그날 아침 공개분은 다음 날부터)."""
    cs = np.concatenate([np.zeros((a.shape[0], 1), np.int64), np.cumsum(a, axis=1, dtype=np.int64)], axis=1)
    t = np.arange(a.shape[1])
    return cs[:, t] - cs[:, np.maximum(t - w, 0)]


def r_cap(tr: pd.DataFrame) -> pd.Series:
    return tr["ret"] * tr["units"] / 0.15                              # 투입 R(정정 31-1)


def run(cfg: dict, cm: float, end: str) -> "tuple[pd.Series, pd.DataFrame]":
    eq, tr = L.simulate(_G["D"], cfg, cm, end=end)
    if len(tr):
        tr = tr.assign(entry=pd.to_datetime(tr["entry"]), exit=pd.to_datetime(tr["exit"]))
        tr = tr[tr["entry"] >= JUDGE0]
    return eq, tr


def stats(tr: pd.DataFrame, yrs: float) -> dict:
    if len(tr) == 0:
        return {"n": 0}
    r, u = tr["ret"].to_numpy(float), tr["units"].to_numpy(float)
    win, loss = r[r > 0], r[r <= 0]
    R = r_cap(tr)
    return {"n": int(len(r)), "per_year": len(r) / yrs, "R_mean": float(R.mean()), "R_sum": float(R.sum()),
            "ret_mean_ew": float(r.mean()), "ret_mean_cw": float((r * u).sum() / u.sum()),
            "win_rate": float(len(win) / len(r)), "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "R_halves": [float(R[(tr["entry"] >= lo) & (tr["entry"] < hi)].mean()) for lo, hi in HALVES]}


def _null_task(args) -> dict:
    kind, seed = args
    D = _G["D"]
    rng = np.random.default_rng(seed)
    pool = _G["base"] if kind == "all" else (_G["base"] & _G["cov20"])
    rate = _G["rate_all"] if kind == "all" else _G["rate_cov"]
    D["null_mask"] = pool & (rng.random(pool.shape) < rate[None, :])
    _, tr = run({**_G["cfg"], "cand_and": "null_mask"}, 2.0, _G["end"])
    R = r_cap(tr) if len(tr) else pd.Series(dtype=float)
    return {"seed": seed, "n": int(len(tr)), "R_mean": float(R.mean()) if len(R) else float("nan"), "R_sum": float(R.sum())}


def main() -> None:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="F 대신 무작위 마스크 — 경로만(성적 미출력)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    price = P.ARM_PRICE["leader_M"]
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
    cfg = P.ARMS["leader_M"]
    if a.smoke:                                                             # 스모크는 리포트 표를 읽지 않는다(값 미조회 유지)
        rng = np.random.default_rng(7)
        net = (rng.random(C.shape) < 0.002).astype(np.int32)
        cnt = (rng.random(C.shape) < 0.004).astype(np.int32)
    else:
        net, cnt = report_events(con, days, codes)
    D["F_ana"] = window_sum(net, WIN) >= 1
    cov = {w: window_sum(cnt, w) >= 1 for w in COV_WIN}
    base = D["cand"] & D["brk"][cfg["N"]] & (D["rs"] >= cfg["theta"]) & D["regime_on"][None, :]
    j0 = days.searchsorted(pd.Timestamp(JUDGE0))
    with np.errstate(invalid="ignore", divide="ignore"):
        nb = base.sum(0)
        rate_all = np.where(nb > 0, (base & D["F_ana"]).sum(0) / nb, 0.0)
        nc = (base & cov[20]).sum(0)
        rate_cov = np.where(nc > 0, (base & D["F_ana"]).sum(0) / nc, 0.0)
    _G.update(D=D, cfg=cfg, end=end, base=base, cov20=cov[20], rate_all=rate_all, rate_cov=rate_cov)
    listed = np.isfinite(C[:, -1])                                         # 마지막 거래일 가격 있음 = 상장(없음 = 폐지 근사)
    yr = days.year.to_numpy()
    coverage = {}
    for w in COV_WIN:
        for grp, msk in (("상장", listed), ("폐지", ~listed)):
            b = base[msk][:, j0:]
            c = (base & cov[w])[msk][:, j0:]
            coverage[f"{w}d_{grp}"] = {int(y): float(c[:, yr[j0:] == y].sum() / max(b[:, yr[j0:] == y].sum(), 1)) for y in np.unique(yr[j0:])}
    yrs = (days[-1] - pd.Timestamp(JUDGE0)).days / 365.25
    out: dict = {"end": end, "smoke": a.smoke, "pass_rate_mean": float((base & D["F_ana"])[:, j0:].sum() / max(base[:, j0:].sum(), 1)),
                 "coverage": coverage, "arms": {}}
    trades = {}
    for arm, c_ in (("leader_M", cfg), ("M_ana", {**cfg, "cand_and": "F_ana"})):
        res = {}
        for cm in (1.0, 2.0):
            eq, tr = run(c_, cm, end)
            m = L.metrics(eq.loc[JUDGE0:] / eq.loc[JUDGE0:].iloc[0])
            res[f"{int(cm)}x"] = {"cagr": m["cagr"], "mdd": m["mdd"], **stats(tr, yrs)}
            if cm == 2.0:
                trades[arm] = tr
                if not a.smoke:
                    tr.to_csv(os.path.join(a.out, f"trades_{arm}_2x.csv.gz"), index=False)
                    eq.to_frame("eq").to_csv(os.path.join(a.out, f"eq_{arm}_2x.csv"))
                if len(tr):
                    contrib = r_cap(tr).groupby(tr["code"]).sum().sort_values(ascending=False)
                    _, t5 = run({**c_, "exclude": list(contrib.index[:5])}, 2.0, end)
                    res["2x"]["drop_top5_R_sum"] = float(r_cap(t5).sum()) if len(t5) else 0.0
                    res["2x"]["by_year"] = tr.groupby(tr["entry"].dt.year).apply(lambda g: {"n": len(g), "R_mean": float(r_cap(g).mean())}).to_dict()
        out["arms"][arm] = res
    n_null = 4 if a.smoke else N_NULL
    with mp.get_context("fork").Pool(a.workers) as pool:
        nul = pool.map(_null_task, [("all", s) for s in range(n_null)])
        nul_cov = pool.map(_null_task, [("cov", s) for s in range(n_null)])
    out["null"] = {"all_R_mean_p95": float(np.nanquantile([x["R_mean"] for x in nul], 0.95)),
                   "cov_R_mean_p95": float(np.nanquantile([x["R_mean"] for x in nul_cov], 0.95)),
                   "all_n_median": float(np.median([x["n"] for x in nul]))}
    A, M = out["arms"]["M_ana"]["2x"], out["arms"]["leader_M"]["2x"]
    if A["n"] < MIN_TRADES:
        out["verdict"] = "VOID(체결 < 30)"
    else:
        bars = {"b1_R_gt_null95": A["R_mean"] > out["null"]["all_R_mean_p95"], "b2_ret_cw_gt0": A["ret_mean_cw"] > 0,
                "b3_per_year_ge36": A["per_year"] >= MIN_PER_YEAR,
                "b4_halves_gt_M": all(x > y for x, y in zip(A["R_halves"], M["R_halves"]))}
        out["bars"] = bars
        out["verdict"] = "통과(전진 팔 등록)" if all(bars.values()) else "실패(기각 목록)"
    if a.smoke:
        print("smoke ok", {k: out["arms"][k]["2x"].get("n") for k in out["arms"]}, "null", len(nul), len(nul_cov))
        return
    pd.DataFrame(nul).to_csv(os.path.join(a.out, "null_all.csv"), index=False)
    pd.DataFrame(nul_cov).to_csv(os.path.join(a.out, "null_cov.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"verdict": out["verdict"], "bars": out.get("bars"), "null": out["null"]}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
