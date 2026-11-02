"""thinband 9단계 — 무작위 20종목 슬롯 북, 판정 실행 1회. 결과는 (검증 전). 표본 끝 2023-12-28, 2024 이후 안 읽는다."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402
import thinband_lib as B  # noqa: E402
from dataclasses import replace  # noqa: E402

from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample, t_haircut  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402

PIPELINE = ("thinband", Stage.BACKTEST)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "thinband"
WIDE_N = 33149 + 73 + 15 + 288 + 38 + 79 + 999 + 8     # PREREG §5 다중검정: fcs·main·ML·루프·scalp·daytrade·smallcap
_G: dict = {}


def _task(args) -> dict:
    key, seed = args
    ctx = _G[key]
    run = B.simulate(ctx, np.random.default_rng(seed))
    s2, s1 = B.period_series(ctx, run, 2.0), B.period_series(ctx, run, 1.0)
    den = B.SLOTS * ctx.cfg.order
    daily = (run.gross - 2.0 * run.cost) / den
    aum = B.SLOTS * ctx.cfg.order
    return {"a2": s2["active"], "a1": s1["active"], "n2": s2["net"], "exp": s2["exposure"], "curve": B.curve_metrics(ctx, run, 2.0, aum),
            "tail": B.tail_metrics(run, ctx.cfg.order), "daily": daily.astype(np.float32), "doom": run.doom_writeoff,
            "cens": run.censored_halted_nondistress, "cost_total": float(run.cost.sum() * 2.0 / aum), "turn": float(len(run.holds))}


def run_many(pool, key: str, n: int, seed: int) -> list:
    return pool.map(_task, [(key, int(s)) for s in np.random.SeedSequence(seed).generate_state(n)], chunksize=10)


def q3(v) -> list:
    return [float(x) for x in np.quantile(np.asarray(v, dtype=float), [.05, .5, .95])]


def summary(res: list, years: float) -> dict:
    a2 = np.array([r["a2"] for r in res])
    a1 = np.array([r["a1"] for r in res])
    n2 = np.array([r["n2"] for r in res])
    per_run = a2.mean(axis=1)
    pm = a2.mean(axis=0)
    w, l_ = pm[pm > 0], pm[pm < 0]
    return {"periods": a2.shape[1], "active2": float(a2.mean()), "active1": float(a1.mean()), "net2": float(n2.mean()),
            "active2_runs_q": q3(per_run), "net2_runs_q": q3(n2.mean(axis=1)), "share_runs_active_pos": float((per_run > 0).mean()),
            "period_median": float(np.median(pm)), "period_gain": float(w.mean()) if len(w) else float("nan"), "period_loss": float(l_.mean()) if len(l_) else float("nan"),
            "period_payoff": float(w.mean() / -l_.mean()) if len(w) and len(l_) else float("nan"),
            "exposure": float(np.mean([r["exp"].mean() for r in res])), "cost_drag_yr": float(np.mean([r["cost_total"] for r in res]) / years),
            "holds_mean": float(np.mean([r["turn"] for r in res]))}


def ols_factors(con, p, day_mean) -> dict:
    """전 시장 월 3분위 롱숏, 2017-09~. 주식수 = shares_outstanding_history as-of(알게 된 날 다음부터)."""
    ix = _G["idx"]
    sh = pd.read_sql_query("select code,knowledge_date,shares_outstanding from shares_outstanding_history where shares_outstanding>0 and code = any(%s)",
                           con, params=(p.codes,))
    sh["knowledge_date"] = pd.to_datetime(sh.knowledge_date)
    sh = sh.sort_values("knowledge_date")
    by = {c: (g.knowledge_date.to_numpy(), g.shares_outstanding.to_numpy(dtype=float)) for c, g in sh.groupby("code")}
    months = pd.Series(p.dates).dt.to_period("M")
    t0 = int(np.searchsorted(p.dates, np.datetime64(B.FACTOR_START)))
    ends = [i for i in range(t0 - 1, p.T) if i == p.T - 1 or months[i] != months[i + 1]]
    rows, cover = [], []
    for a, b in zip(ends[:-1], ends[1:]):
        names = np.flatnonzero(p.traded[:, a] & np.isfinite(p.ADV[:, a]) & ~p.excluded & np.isfinite(p.Cb[:, a]) & (p.ADV[:, a] >= 100.0))
        ret = p.Cb[names, b] / p.Cb[names, a] - 1.0
        cap = np.full(len(names), np.nan)
        for k, i in enumerate(names):
            got = by.get(p.codes[i])
            if got is not None:
                j = np.searchsorted(got[0], p.dates[a], side="left") - 1
                if j >= 0:
                    cap[k] = got[1][j] * p.RAW[i, a]
        cover.append(float(np.isfinite(cap).mean()))
        with np.errstate(invalid="ignore", divide="ignore"):
            mom = p.Cb[names, a - 21] / p.Cb[names, max(a - 252, 0)] - 1.0 if a >= 252 else np.full(len(names), np.nan)
        mom = np.where(p.Cb[names, max(a - 252, 0)] > 0, mom, np.nan)

        def ls(sig, hi_long=True):
            ok = np.isfinite(sig)
            if ok.sum() < 30:
                return np.nan
            lo, hi = np.quantile(sig[ok], [1 / 3, 2 / 3])
            top, bot = ret[ok & (sig >= hi)].mean(), ret[ok & (sig <= lo)].mean()
            return top - bot if hi_long else bot - top

        rows.append({"y": float(day_mean[a + 1:b + 1].sum()), "kospi": ix["KOSPI"][b] / ix["KOSPI"][a] - 1, "kosdaq": ix["KOSDAQ"][b] / ix["KOSDAQ"][a] - 1,
                     "mom": ls(mom), "size": ls(cap, False), "lowvol": ls(p.SIG63[names, a], False), "liq": ls(p.ADV[names, a], False)})
    f = pd.DataFrame(rows)
    five = ["kospi", "kosdaq", "mom", "size", "lowvol"]
    return {"months": len(f), "cap_coverage": float(np.mean(cover)), "five": L.ols_hac(f.y, f[five], lag=3), "two": L.ols_hac(f.y, f[["kospi", "kosdaq"]], lag=3),
            "five_liq": L.ols_hac(f.y, f[[*five, "liq"]], lag=3)}


def ew_label(p, ctx) -> dict:
    """U 전체 동일가중(라벨, 종가 대용, 잔류 복원 비용 포함), 2배."""
    prev: dict = {}
    rows = []
    for a, b in ctx.periods:
        names = ctx.ids[a]
        g = p.Cb[names, b] / p.Cb[names, a] - 1.0
        ix = np.array([ctx.I[i][b] / ctx.I[i][a] - 1.0 for i in names])
        w = 1.0 / len(names)
        cost = 0.0
        d = p.day_py[a]
        cur = {int(i): w for i in names}
        for i in set(cur) | set(prev):
            dw = cur.get(i, 0.0) - prev.get(i, 0.0)
            if abs(dw) > 0:
                adv = max(np.nan_to_num(p.ADV[i, a], nan=L.ADV_FLOOR_M), L.ADV_FLOOR_M)
                sig = min(np.nan_to_num(p.SIG20[i, a], nan=L.VOL_FALLBACK), 1.0)
                cost += abs(dw) * L.cost_one_way("BUY" if dw > 0 else "SELL", d, p.market[i], adv, sig, p.RAW[i, a])
        rows.append((float(g.mean() - ix.mean() - 2 * cost), float(np.median(g - ix) - 2 * cost)))
        prev = {int(i): w * (1 + float(x)) / (1 + float(g.mean())) for i, x in zip(names, g)}
    x = np.array(rows)
    return {"active2": float(x[:, 0].mean()), "median_name_active2": float(x[:, 1].mean())}


def main() -> int:
    require_stage(*PIPELINE)
    t_start = time.time()
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, _, kind = B.load_panel(con)
    ix = pd.read_sql_query("select code,date,close from sector_index where code in ('001','101') and date between %s and %s", con, params=(B.SIG_START, B.DATA_END))
    ix["date"] = pd.to_datetime(ix.date)
    w = ix.pivot(index="date", columns="code", values="close").reindex(pd.DatetimeIndex(p.dates)).ffill().bfill()
    idx = {"KOSPI": w["001"].to_numpy(float), "KOSDAQ": w["101"].to_numpy(float)}
    assert np.isfinite(idx["KOSPI"]).all() and np.isfinite(idx["KOSDAQ"]).all()
    _G["idx"] = idx
    assert (p.market == "KOSPI").sum() > 0 and (p.market == "KOSDAQ").sum() > 0    # 시장 매핑
    doom = B.doom_mask(p, kind)
    base = B.Cfg()
    _G["U"] = B.Ctx(p, idx, base, doom)
    _G["U2"] = B.Ctx(p, idx, replace(base, band=L.BAND_U2), doom)
    labels = B.label_cfgs(base)
    for k, c in labels.items():
        _G[k] = B.Ctx(p, idx, c, None if k == "no_doom" else doom)
    years = (pd.Timestamp(p.dates[-1]) - pd.Timestamp(p.dates[p.sig_t0])).days / 365.25
    record_trial("thinband", {"arm": "random20", "band": "U", "step": B.STEP, "prereg": "TB-1"}, note="판정")
    record_trial("thinband", {"arm": "random20", "band": "U2", "step": B.STEP, "prereg": "TB-1"}, note="K3")
    with mp.get_context("fork").Pool(12) as pool:
        resU = run_many(pool, "U", B.N_RUNS, B.SEED)
        resU2 = run_many(pool, "U2", B.N_RUNS, B.SEED + 1)
        print(f"[{time.time() - t_start:.0f}s] 주 팔 끝", flush=True)
        lab = {}
        for k in labels:
            record_trial("thinband", {"arm": "label", "label": k, "prereg": "TB-1"}, note="라벨")
            lab[k] = summary(run_many(pool, k, B.N_LABEL, B.SEED + 2), years)
    print(f"[{time.time() - t_start:.0f}s] 라벨 끝", flush=True)
    per = _G["U"].periods
    starts = np.array([p.dates[a] for a, _ in per])
    au = np.array([r["a2"] for r in resU]).mean(axis=0)
    au2 = np.array([r["a2"] for r in resU2]).mean(axis=0)
    a1u = np.array([r["a1"] for r in resU]).mean(axis=0)
    su, su2 = summary(resU, years), summary(resU2, years)
    q5_final = float(np.quantile([r["curve"]["total"] for r in resU], 0.05)) * L.AUM_WON
    mdd_med = float(np.median([r["curve"]["mdd"] for r in resU]))
    cmdd_med = float(np.median([r["curve"]["ctrl_mdd"] for r in resU]))
    kb = B.kill_bars(au, au2, starts, su["exposure"], mdd_med, cmdd_med, q5_final)
    u2_run = np.array([r["a2"].mean() for r in resU2])
    day_mean = np.mean([r["daily"] for r in resU], axis=0)
    yrs = starts.astype("datetime64[Y]").astype(int) + 1970
    by_year = {int(y): {"U_active2": float(au[yrs == y].mean()), "U_active1": float(a1u[yrs == y].mean()), "U2_active2": float(au2[yrs == y].mean())} for y in np.unique(yrs)}
    curve = {k: q3([r["curve"][k] for r in resU]) for k in ("mdd", "ctrl_mdd", "underwater", "worst250", "halted_mean", "halted_max", "halt_longest", "liq_rows_max", "locks", "total", "total_ctrl", "hold_med")}
    seg = {s: {"book": q3([r["curve"]["segments"][s]["book"] for r in resU]), "ctrl": q3([r["curve"]["segments"][s]["ctrl"] for r in resU])} for s in B.SEGMENTS}
    tail = {k: q3([r["tail"][k] for r in resU]) for k in ("n_minus50", "n_minus90", "distressed", "slot_mean", "slot_median", "slot_mean_ex5", "slot_mean_ex20")}
    n_narrow = count_trials("thinband")
    res = {"years": years, "bars": kb, "U": su, "U2": su2, "labels": lab, "by_year": by_year, "curve": curve, "segments": seg, "tail": tail,
           "periods_active_U": au.tolist(), "periods_active_U2": au2.tolist(), "starts": [str(s)[:10] for s in starts],
           "period_block_U": B.period_block_interval(au), "label_T2_U2_run_q95": float(np.quantile(u2_run, .95)), "factors": ols_factors(con, p, day_mean),
           "ew_U": ew_label(p, _G["U"]), "ew_U2": ew_label(p, _G["U2"]), "doom_names": int(doom.sum()),
           "doom_writeoff_mean": float(np.mean([r["doom"] for r in resU])), "censored_halted_nondistress_mean": float(np.mean([r["cens"] for r in resU])),
           "deflation": {"narrow": {"n": n_narrow, **{k: v for k, v in t_haircut(max(n_narrow, 2)).items() if isinstance(v, (int, float))}},
                         "wide": {"n": WIDE_N, **{k: v for k, v in t_haircut(WIDE_N).items() if isinstance(v, (int, float))}},
                         "dsr_narrow": _clean(deflated_sharpe_from_sample(au, max(n_narrow, 2))), "dsr_wide": _clean(deflated_sharpe_from_sample(au, WIDE_N))},
           "runtime_s": time.time() - t_start}
    (OUT / "B.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(kb, default=float), f"runtime {res['runtime_s']:.0f}s")
    return 0


def _clean(d: dict) -> dict:
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating))}


if __name__ == "__main__":
    raise SystemExit(main())
