"""smallcap 4단계 — 탐색 절반 입구 관문과 자유도 지도. 잠금 절반은 읽지 않는다. 결과는 (검증 전)."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from swing_it.diagnostics.trials import count_trials, record_trial
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE = ("smallcap", Stage.ALPHA_VALIDATION)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "smallcap"
N_NULL = 300
SEED = 20261102
EW_STEP = 60
FULL_YEARS = range(2009, 2016)
_G: dict = {}


def _init(p, us, evs) -> None:
    _G.update(p=p, us=us, evs=evs, bench={k: L.Bench(p, u) for k, u in us.items()})


def _null_task(args) -> dict:
    sig, kind, seed = args
    p, u, ev, be = _G["p"], _G["us"]["U"], _G["evs"][sig], _G["bench"]["U"]
    rng = np.random.default_rng(seed)
    nev = L.random_events(p, u, ev, rng) if kind == "B5" else L.shifted_events(p, u, ev, rng)
    fills, _ = L.run_book(p, nev, be)
    if fills.empty:
        return {"sig": sig, "kind": kind, "ev_active": float("nan"), "ev_net": float("nan"), "n": 0}
    return {"sig": sig, "kind": kind, "ev_active": float(fills.active.mean()), "ev_net": float(fills.net.mean()), "n": len(fills)}


def years_span(p) -> float:
    d = pd.DatetimeIndex(p.dates[p.sig_t0:])
    return (d[-1] - d[0]).days / 365.25


def ew_arm(p, u, step: int = EW_STEP) -> pd.DataFrame:
    """U(t) 전체 동일가중, step 행마다 재구성, 지수 대비. 체결은 종가 대용 라벨(§6). 비용은 들고 나는 종목만 2배."""
    idx = _G["idx"]
    grid = list(range(p.sig_t0, p.T - 1, step)) + [p.T - 1]
    prev: set = set()
    rows = []
    for a, b in zip(grid[:-1], grid[1:]):
        names = np.flatnonzero(u[:, a] & np.isfinite(p.Cb[:, a]))
        if len(names) == 0:
            continue
        g = p.Cb[names, b] / p.Cb[names, a] - 1.0
        ix = np.array([idx[p.market[i]][b] / idx[p.market[i]][a] - 1.0 for i in names])
        cur = set(int(i) for i in names)
        cost = 0.0
        day = p.day_py[a]
        for i in cur - prev:
            cost += L.cost_one_way("BUY", day, p.market[i], max(np.nan_to_num(p.ADV[i, a], nan=L.ADV_FLOOR_M), L.ADV_FLOOR_M),
                                   min(np.nan_to_num(p.SIG20[i, a], nan=L.VOL_FALLBACK), 1.0), p.RAW[i, a])
        for i in prev - cur:
            cost += L.cost_one_way("SELL", day, p.market[i], max(np.nan_to_num(p.ADV[i, a], nan=L.ADV_FLOOR_M), L.ADV_FLOOR_M),
                                   min(np.nan_to_num(p.SIG20[i, a], nan=L.VOL_FALLBACK), 1.0), p.RAW[i, a])
        gross, bench = float(g.mean()), float(ix.mean())
        c2 = 2.0 * cost / len(names)
        rows.append({"year": int(str(p.dates[a])[:4]), "gross": gross, "bench": bench, "gross_active": gross - bench,
                     "net": gross - c2, "active": gross - c2 - bench, "cost": c2, "n": len(names), "turn": len(cur - prev) / len(names)})
        prev = cur
    return pd.DataFrame(rows)


def pay(s: pd.Series) -> float:
    w, l_ = s[s > 0], s[s < 0]
    return float(w.mean() / -l_.mean()) if len(w) and len(l_) else float("nan")


def curve(p, ev_fills: pd.DataFrame, be) -> dict:
    out = {}
    for h in (20, 60):
        vals = []
        for f in ev_fills.itertuples():
            k = f.e + h
            if k >= p.T:
                continue
            g = p.C[f.c, f.e] / p.TP[f.c, f.e] * p.Cp[f.c, k] / p.Cp[f.c, f.e] - 1.0
            vals.append(g - be.ret(f.t, f.e, k))
        out[f"{h}d"] = float(np.nanmean(vals))
    out["exit"] = float(ev_fills.gross_active.mean())
    return out


def year_block_delta(a: pd.DataFrame, b: pd.DataFrame, col: str, rng) -> dict:
    x = a.groupby("year")[col].mean()
    y = b.groupby("year")[col].mean()
    yrs = sorted(set(x.index) & set(y.index))
    d = (x.reindex(yrs) - y.reindex(yrs)).to_numpy()
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    return {"delta": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)), "years": len(d)}


def factors(con, p, u, be, day) -> dict:
    """월말 3분위 롱숏(12-1·소형·저변동, U(t)·탐색 절반 안) + KOSPI·KOSDAQ. 시총 = 원 가격 × as-of 주식수(접수 다음날부터)."""
    sh = pd.read_sql_query("select code,knowledge_date,shares_common from earnings_dart_doc where shares_common is not null and code = any(%s)",
                           con, params=(p.codes,))
    L.assert_explore(sh.code.unique())
    sh["knowledge_date"] = pd.to_datetime(sh.knowledge_date)
    sh = sh.sort_values("knowledge_date")
    by = {c: (g.knowledge_date.to_numpy(), g.shares_common.to_numpy(dtype=float)) for c, g in sh.groupby("code")}
    months = pd.Series(p.dates).dt.to_period("M")
    ends = [i for i in range(p.sig_t0, p.T) if i == p.T - 1 or months[i] != months[i + 1]]
    rows = []
    cap_ok = []
    for a, b in zip(ends[:-1], ends[1:]):
        names = np.flatnonzero(u[:, a] & np.isfinite(p.Cb[:, a]))
        ret = p.Cb[names, b] / p.Cb[names, a] - 1.0
        cap = np.full(len(names), np.nan)
        for k, i in enumerate(names):
            got = by.get(p.codes[i])
            if got is not None:
                j = np.searchsorted(got[0], p.dates[a], side="left") - 1
                if j >= 0:
                    cap[k] = got[1][j] * p.RAW[i, a]
        cap_ok.append(float(np.isfinite(cap).mean()))
        mom = np.where(p.first[names] + 252 <= a, p.Cb[names, max(a - 21, 0)] / p.Cb[names, max(a - 252, 0)] - 1.0, np.nan)
        vol = p.SIG63[names, a]

        def ls(sig, hi_is_long=True):
            ok = np.isfinite(sig)
            if ok.sum() < 30:
                return np.nan
            lo_cut, hi_cut = np.quantile(sig[ok], [1 / 3, 2 / 3])
            top, bot = ret[ok & (sig >= hi_cut)].mean(), ret[ok & (sig <= lo_cut)].mean()
            return top - bot if hi_is_long else bot - top

        ix = _G["idx"]
        pnl = day.pnl.to_numpy()[a + 1: b + 1].sum() / L.AUM_WON
        rows.append({"y": pnl, "kospi": ix["KOSPI"][b] / ix["KOSPI"][a] - 1, "kosdaq": ix["KOSDAQ"][b] / ix["KOSDAQ"][a] - 1,
                     "mom": ls(mom), "size": ls(cap, False), "lowvol": ls(vol, False), "ew": float(ret.mean())})
    f = pd.DataFrame(rows)
    return {"five": L.ols_hac(f.y, f[["kospi", "kosdaq", "mom", "size", "lowvol"]], lag=3),
            "two_label": L.ols_hac(f.y, f[["kosdaq", "ew"]], lag=3), "cap_coverage": float(np.mean(cap_ok)), "months": len(f)}


def main() -> int:
    require_stage(*PIPELINE)
    t_start = time.time()
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, _ = L.load_explore_panel(con)
    L.assert_explore(p.codes)
    idx_df = pd.read_sql_query("select code,date,close from sector_index where code in ('001','101') and date<='2016-08-31'", con)
    idx_df["date"] = pd.to_datetime(idx_df.date)
    wide = idx_df.pivot(index="date", columns="code", values="close").reindex(pd.DatetimeIndex(p.dates)).ffill()
    _G["idx"] = {"KOSPI": wide["001"].to_numpy(float), "KOSDAQ": wide["101"].to_numpy(float)}
    assert np.isfinite(_G["idx"]["KOSPI"][p.sig_t0:]).all() and np.isfinite(_G["idx"]["KOSDAQ"][p.sig_t0:]).all()
    assert (p.market == "KOSPI").sum() > 0 and (p.market == "KOSDAQ").sum() > 0    # B7 매핑

    us = {"U": p.universe(L.BAND_U), "U2": p.universe(L.BAND_U2)}
    bench = {k: L.Bench(p, u) for k, u in us.items()}
    years = years_span(p)
    sigs = {"HVNM": L.hvnm_events, "MOM": L.mom_events}
    evs, map_rows, study, fails_all = {}, {}, {}, {}
    for s, fn in sigs.items():
        for un, u in us.items():
            ev = fn(p, u)
            key = f"{s}_{un}"
            evs[key] = ev
            fills, fails = L.event_study(p, ev, bench[un])
            study[key], fails_all[key] = fills, fails
            map_rows[key] = L.summarize(fills, years=years)
            record_trial("smallcap", {"signal": s, "universe": un, "view": "event_study", "half": 0, "prereg": "SC-1"}, note="지도")
    ew = {un: ew_arm(p, u) for un, u in us.items()}
    for un, df in ew.items():
        record_trial("smallcap", {"signal": "EW", "universe": un, "view": "ew_vs_index", "half": 0, "prereg": "SC-1"}, note="지도")
    print(f"[{time.time() - t_start:.0f}s] 지도 끝", flush=True)

    # 입구 관문: 후보 후보는 U 의 HVNM·MOM. 북·널 300(B5), 지속성 널 300(B6 라벨).
    gate = {}
    book = {}
    for s in sigs:
        fills, fails = L.run_book(p, evs[f"{s}_U"], bench["U"])
        book[s] = fills
        gate[s] = {"book": L.summarize(fills, years=years), "fails": fails, "frag": L.fragility(fills),
                   "fills_by_year": {int(k): int(v) for k, v in fills.groupby("year").size().items()}}
    tasks = [(s, kind, int(sd)) for s in sigs for kind in ("B5", "B6")
             for sd in np.random.SeedSequence(SEED).generate_state(N_NULL)]
    null_evs = {s: evs[f"{s}_U"] for s in sigs}
    with mp.get_context("fork").Pool(12, initializer=_init, initargs=(p, {"U": us["U"]}, null_evs)) as pool:
        nulls = pd.DataFrame(pool.map(_null_task, tasks, chunksize=5))
    print(f"[{time.time() - t_start:.0f}s] 널 끝", flush=True)
    for s in sigs:
        g = gate[s]
        for kind in ("B5", "B6"):
            d = nulls[(nulls.sig == s) & (nulls.kind == kind)]
            g[kind] = {"q50": float(d.ev_active.quantile(.5)), "q95": float(d.ev_active.quantile(.95)), "n_median": float(d.n.median()),
                       "pctile": float((d.ev_active < g["book"]["ev_active"]).mean() * 100), "q95_net": float(d.ev_net.quantile(.95))}
        es = map_rows[f"{s}_U"]
        fy = g["fills_by_year"]
        g["G1_event_active_gt0"] = bool(es["ev_active"] > 0)
        g["G1_wins_active"] = es["ev_active_wins"]
        g["G2_book_beats_B5_q95"] = bool(g["book"]["ev_active"] > g["B5"]["q95"])
        g["G3_fills"] = bool(g["book"]["n"] >= 200 and all(fy.get(y, 0) >= 15 for y in FULL_YEARS))
        g["G0_events_ge_300"] = bool(es["n"] >= 300)
        g["pass"] = bool(g["G0_events_ge_300"] and g["G1_event_active_gt0"] and g["G2_book_beats_B5_q95"] and g["G3_fills"])
        g["curve"] = curve(p, study[f"{s}_U"], bench["U"])
        day = L.book_daily(p, book[s])
        r1 = pd.Series(_G["idx"]["KOSPI"]).pct_change().fillna(0).to_numpy()
        r2 = pd.Series(_G["idx"]["KOSDAQ"]).pct_change().fillna(0).to_numpy()
        ctrl = float((day.exp_kospi.shift(1).fillna(0).to_numpy() * r1 + day.exp_kosdaq.shift(1).fillna(0).to_numpy() * r2).sum())
        g["B7_label"] = {"book_pnl_won": float(day.pnl.sum()), "exposure_ctrl_won": ctrl,
                         "mean_exposure": float((day.exp_kospi + day.exp_kosdaq).iloc[p.sig_t0:].mean() / L.AUM_WON)}
        g["factor"] = factors(con, p, us["U"], bench["U"], day)
        # 라벨: 종가 대용(체결 불가), 부실 −100%(벤치도 같게), 비용 1배
        lab, _ = L.event_study(p, evs[f"{s}_U"], bench["U"], close_fill=True)
        g["label_close_proxy_ev_active"] = float(lab.active.mean())
        lab2, _ = L.event_study(p, evs[f"{s}_U"], L.Bench(p, us["U"], dist100=True), dist100=True)
        g["label_dist100_ev_active"] = float(lab2.active.mean())
        lab3, _ = L.event_study(p, evs[f"{s}_U"], bench["U"], m=1.0)
        g["label_cost1x_ev_active"] = float(lab3.active.mean())
        record_trial("smallcap", {"signal": s, "universe": "U", "view": "book", "half": 0, "prereg": "SC-1"}, note="입구 관문")

    rng = np.random.default_rng(SEED)
    deltas = {}
    for s in sigs:
        a, b = study[f"{s}_U"], study[f"{s}_U2"]
        deltas[s] = {c: year_block_delta(a, b, c, rng) for c in ("gross_active", "active")}
    deltas["EW"] = {c: year_block_delta(ew["U"], ew["U2"], c, rng) for c in ("gross_active", "active")}
    candidate = next((s for s in ("HVNM", "MOM") if gate[s]["pass"]), None)
    ps = 250.0 / EW_STEP
    res = {"years": years, "map": map_rows, "fails": fails_all, "gate": gate, "delta": deltas, "candidate": candidate,
           "ew": {un: {"periods": len(d), "ev_gross": float(d.gross.mean()), "ev_gross_active": float(d.gross_active.mean()), "ev_net": float(d.net.mean()),
                       "ev_active": float(d.active.mean()), "payoff_abs": pay(d.net), "payoff_active": pay(d.active),
                       "ann_gross_active": float(d.gross_active.mean() * ps), "ann_active": float(d.active.mean() * ps),
                       "turn": float(d.turn.iloc[1:].mean()), "cost_per_period": float(d.cost.mean()),
                       "by_year_active": {int(k): float(v) for k, v in d.groupby("year").active.mean().items()}} for un, d in ew.items()},
           "by_year_event_active": {k: {int(y): float(v) for y, v in d.groupby("year").active.mean().items()} for k, d in study.items()},
           "n_trials_smallcap": count_trials("smallcap"), "runtime_s": time.time() - t_start}
    (OUT / "S4.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    cache = Path(os.environ.get("SMALLCAP_CACHE", "/tmp"))
    with open(cache / "smallcap_s4_fills.pkl", "wb") as fh:
        pickle.dump({"study": study, "book": book, "codes": p.codes}, fh)
    print(json.dumps({"candidate": candidate, "runtime_s": res["runtime_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
