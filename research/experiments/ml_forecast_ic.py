"""ML 수익률 예측 1단계 — IC(비용 없음, 5거래일 격자). PREREG §7·§8·§9, 정정 ML-1, DESIGN §4-3.

순서: 관문 확인(0단계 통과·조정주가 검사) → 원장(진단 → E → M) → 라벨 → OOS 예측(M·E·S·R·L·P) → N1·결정성 → N2 1,000·N3 300 → I-1~I-5 →
``S1.{json,md}``(검증 전). **한 번만 돈다** — ``S1.json`` 이 있으면 멈춘다. 2024-01 이후 데이터는 어떤 계산에도 쓰지 않는다.
2단계(북)는 여기서 돌리지 않는다. 판정 문장은 사전등록 규칙의 기계적 적용이다(GUARDRAILS §8).

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" --with lightgbm==4.7.0 \\
      python research/experiments/ml_forecast_ic.py --cache <레포 밖> [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import time
from pathlib import Path

import ml_forecast_common as C
import ml_forecast_models as MD
import ml_forecast_nulls as NL
import numpy as np
from krx_quant_core.stats.metrics import newey_west_t
from ml_forecast_common import END, LABEL, MIN_IC_NAMES, N1_REPS, N2_REPS, N3_REPS, NW_LAG, PREREG_SHA, SEEDS, T_BAR, TEST_YEARS
from ml_forecast_target import labels, tradable

OUT = Path(__file__).resolve().parents[1] / "logs" / "ml_forecast"
MODELS = ("S", "R", "L", "P", "E", "M")  # 원장 순서: 진단 먼저, 1차 마지막(GUARDRAILS §4 ②)
CANDIDATES = ("M", "E")
_N: dict = {}  # 널 작업자 공유(fork)


# ── IC ──────────────────────────────────────────────────────────────────────────


def day_ic(pred_t: np.ndarray, y_u: np.ndarray, tr_u: np.ndarray, min_names: int = MIN_IC_NAMES) -> tuple[float, int]:
    """(Spearman, 유효 수). 유효 = 예측·y 유한 ∧ t+1 거래량 > 0. 유효 < min_names 면 NaN."""
    ok = np.isfinite(pred_t) & np.isfinite(y_u) & tr_u
    n = int(ok.sum())
    if n < max(min_names, 3):
        return float("nan"), n
    return C.spearman(pred_t[ok], y_u[ok]), n


def ic_series(pred: dict, y: dict, U: dict, vol, days: list[int], window: C.Window) -> dict[int, float]:
    """격자일 → IC. 유효 종목이 모자란 날은 뺀다(정정 ML-1 ⑨)."""
    C.require_eval(window, "IC")
    out = {}
    for t in days:
        ic, _ = day_ic(pred[t], y[t][U[t]], tradable(vol, t)[U[t]])
        if np.isfinite(ic):
            out[t] = ic
    return out


def stat_of(ics: dict[int, float], dates) -> dict:
    v = np.array(list(ics.values()), dtype=np.float64)
    by = {}
    for yr in TEST_YEARS:
        s = [x for t, x in ics.items() if dates[t].year == yr]
        by[str(yr)] = float(np.mean(s)) if s else float("nan")
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return {"mean": float(v.mean()) if v.size else float("nan"), "sd": sd,
            "icir": float(v.mean() / sd) if v.size > 1 and sd > 0 else float("nan"),
            "n_days": int(v.size), "by_year": by, "years_positive": int(sum(x > 0 for x in by.values()))}


def delta_nw(a: dict[int, float], b: dict[int, float]) -> dict:
    """ΔIC(a − b) 격자일 시계열의 평균·NW(lag 6) t. 두 쪽 다 있는 날만."""
    days = sorted(set(a) & set(b))
    d = np.array([a[t] - b[t] for t in days], dtype=np.float64)
    mu, t = newey_west_t(d, NW_LAG)
    return {"mean": mu, "t_nw6": t, "n_days": len(days)}


def bars_for(name: str, s: dict, null: dict, n3_q95: float | None, d_me: dict | None, d_ms: dict | None) -> dict:
    """I-1~I-5(§7). 숫자 비교만 한다. 값이 NaN 이면 미달."""
    i1 = s["mean"] > null["q95_mean"]
    if name == "E":
        i1 = i1 and s["mean"] > (n3_q95 if n3_q95 is not None else float("inf"))
    out = {"I-1": bool(i1), "I-2": bool(s["icir"] > null["q95_icir"]), "I-3": bool(s["years_positive"] >= 3)}
    if name == "M":
        out["I-4"] = bool(d_me is not None and d_me["mean"] > 0 and d_me["t_nw6"] >= T_BAR)
        out["I-5"] = bool(d_ms is not None and d_ms["mean"] > 0 and d_ms["t_nw6"] >= T_BAR)
    return out


def enters_stage2(name: str, bars: dict) -> bool:
    """§7 — E: I-1~I-3, M: I-1~I-3 + I-5."""
    need = ("I-1", "I-2", "I-3") if name == "E" else ("I-1", "I-2", "I-3", "I-5")
    return all(bars[k] for k in need)


def verdict_stage1(void: list[str], entered: dict[str, bool]) -> str:
    if void:
        return "VOID"
    if not any(entered.values()):
        return "채택 불가(예측력 없음)"
    return "2단계 진입: " + ", ".join(k for k, v in entered.items() if v)


# ── 널 작업자 ──────────────────────────────────────────────────────────────────


def _n2_task(r: int) -> tuple:
    c = _N
    pi = NL.n2_perm(c["u_star"], r, c["N"])
    res, fills = {}, {}
    for name in c["names"]:
        ics = []
        fl = np.zeros(len(c["days"][name]), dtype=np.int32)
        for j, t in enumerate(c["days"][name]):
            out, nf = NL.n2_assign(c["pred"][name][t], c["U"][t], pi, r, t, c["N"])
            fl[j] = nf
            ok = np.isfinite(out) & c["ok_base"][t]
            ics.append(C.spearman(out[ok], c["y_u"][t][ok]) if ok.sum() >= 3 else np.nan)
        v = np.asarray(ics, dtype=np.float64)
        v = v[np.isfinite(v)]
        sd = v.std(ddof=1) if v.size > 1 else np.nan
        res[name] = (float(v.mean()), float(v.mean() / sd) if sd and sd > 0 else np.nan)
        fills[name] = fl
    return r, res, fills


def _n3_task(r: int) -> tuple:
    c = _N
    s = NL.n3_scores({t: c["Z"][t] for t in c["days_all"]}, r)
    ics = []
    for t in c["days_all"]:
        ok = np.isfinite(s[t]) & c["ok_base"][t]
        if ok.sum() >= MIN_IC_NAMES:
            ics.append(C.spearman(s[t][ok], c["y_u"][t][ok]))
    v = np.asarray(ics, dtype=np.float64)
    v = v[np.isfinite(v)]
    return r, float(v.mean()), int(v.size)


def seam_in_tails(y: dict, U: dict, vol, seam, days: list[int]) -> dict:
    """가드가 걸린 셀이 든 라벨 행의 수와, 그 행이 그날 y 상·하위 1% 에 든 비율(DESIGN §6)."""
    rows = in_tail = 0
    for t in days:
        idx = U[t]
        ok = np.isfinite(y[t][idx]) & tradable(vol, t)[idx]
        if ok.sum() == 0:
            continue
        v = y[t][idx]
        lo, hi = np.quantile(v[ok], [0.01, 0.99])
        s = seam[idx, t + 2: t + 22].any(axis=1) & ok
        rows += int(s.sum())
        in_tail += int((s & ((v <= lo) | (v >= hi))).sum())
    return {"rows": rows, "in_top_bottom_1pct": in_tail, "share": in_tail / rows if rows else None}


# ── 실행 ───────────────────────────────────────────────────────────────────────


def data_check() -> dict:
    """§8·정정 ML-1 ⑭ — quant-airflow ``collectors.data_checks``(읽기 전용)를 실행일에 돌려 결과를 남긴다."""
    qa = Path.home() / "git" / "quant-airflow"
    r = subprocess.run([str(qa / ".venv" / "bin" / "python"), "-m", "collectors.data_checks"], cwd=qa,
                       capture_output=True, text=True, check=False, timeout=600, env=os.environ.copy())
    lines = [ln for ln in r.stdout.splitlines() if "adjusted" in ln or "통과" in ln or "[FAIL" in ln]
    return {"exit": r.returncode, "adjusted_lines": lines}


def run(args, log) -> int:
    import ml_forecast_data as D
    import ml_forecast_features as FT
    from swing_it.diagnostics.trials import record_trial

    window = C.assert_window(END, "eval")
    t_start = time.time()
    if (OUT / "S1.json").exists():
        raise SystemExit("S1.json 이 이미 있다 — 1단계는 한 번만 돈다")
    s0p = OUT / "stage0.json"
    if not s0p.exists() or not json.loads(s0p.read_text()).get("pass"):
        raise SystemExit("0단계 관문이 통과하지 않았다 — 라벨을 계산하지 않는다(§8)")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    assert Path(__file__).resolve().parents[2] not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    run_md: list[str] = ["# ML 예측 실행 기록 (RUN)", "", f"- 1단계 시작 {time.strftime('%F %T')}, 코어 16, 작업자 {args.workers}."]

    dq = data_check()
    log(f"조정주가 검사 exit {dq['exit']}")
    fp0 = D.fingerprint_range("2017-01-01", END)
    P = D.build_panels(END, cache, log)
    dates, vol = P["dates"], P["vol"]
    grid, rows = C.ic_grid(dates), C.phase_rows(dates)
    assert set(grid) <= set(rows)
    oos = C.oos_days(dates, grid)
    fr = D.v4_factor_returns(P)
    U = D.universe(P, fr, rows)
    raw = FT.raw_features(P)
    Z = FT.transform(raw, U)
    del raw

    # 원장 — 진단 → E → M. 숫자를 내기 전에 적는다(§9).
    base = {"stage": "S1-IC", "prereg": PREREG_SHA, "corrections": ["ML-1"], "end": END, "universe": "cap400_adv5e9",
            "target": "v4_resid_t+2..t+21", "grid": C.IC_STEP}
    for m in MODELS:
        record_trial(LABEL, {**base, "model": m}, note=f"S1 {'진단' if m not in CANDIDATES else '후보'} {m}")

    # 라벨
    t1 = time.time()
    fry = D.v4_factor_returns(P, D.label_mask(P))
    u_full = np.full_like(P["ret"], np.nan)
    u_full[:, P["first"]:] = fry.u
    y, rep_all = labels(u_full, vol, P["close"], P["delist_last"], U, P["seam"])
    _, rep_oos = labels(u_full, vol, P["close"], P["delist_last"], {t: U[t] for t in oos}, P["seam"])
    log(f"라벨 {time.time() - t1:.0f}s")
    # 위 labels 는 격자 전체에 대해 같은 값을 낸다 — y 는 첫 호출 것을 쓴다.
    seam_rep = seam_in_tails(y, U, vol, P["seam"], oos)
    u_median = float(np.median([len(U[t]) for t in oos]))

    # 학습·예측
    train = {yr: MD.train_days(rows, MD.t0_of(dates, grid, yr)) for yr in TEST_YEARS}
    test = {yr: C.oos_days(dates, grid, yr) for yr in TEST_YEARS}
    train_info = {}
    for yr in TEST_YEARS:
        X, Y, w, ids = MD.train_matrix(train[yr], Z, U, y, vol)
        train_info[str(yr)] = {"rows": int(len(Y)), "days": int(len(np.unique(ids))), "first": str(dates[int(ids.min())].date()),
                               "last": str(dates[int(ids.max())].date()), "t0": str(dates[MD.t0_of(dates, grid, yr)].date())}
        assert int(ids.max()) + C.LABEL_LAST <= MD.t0_of(dates, grid, yr), "퍼지 위반"
    MD.CTX.clear()
    MD.CTX.update(train=train, test=test, Z=Z, U=U, y=y, vol=vol, rng=C.keyed_rng)

    # 계산량 실측(§10) — 맞춤 1회(가장 큰 폴드)를 작업자에서 잰다.
    with mp.get_context("fork").Pool(1) as pool:
        t2 = time.time()
        pool.apply(MD.seeds_task, (("M", 2023, 1, None, 9),))
        fit_s = time.time() - t2
    n_fits = 2 * 4 * 5 * 2 + N1_REPS * 4 * 5
    est_fit_min = n_fits * fit_s / max(args.workers, 1) / 60
    run_md += [f"- 맞춤 1회(2023 폴드, {train_info['2023']['rows']}행) 실측 {fit_s:.1f}s. 맞춤 {n_fits}회(M·S 각 20 ×2(결정성) + N1 400) → 작업자 {args.workers} 외삽 약 {est_fit_min:.1f}분(상한 쪽, 2023 폴드 기준)."]
    log(run_md[-1])

    tasks = [(m, yr, s, None, 0) for m in ("M", "S") for yr in TEST_YEARS for s in SEEDS]
    tasks += [(m, yr, s, None, 1) for m in ("M", "S") for yr in TEST_YEARS for s in SEEDS]  # 결정성(§8)
    tasks += [("M", yr, s, r, 0) for r in range(N1_REPS) for yr in TEST_YEARS for s in SEEDS]
    got = {}
    t3 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(MD.seeds_task, tasks, chunksize=1)):
            name, yr, seed, r, salt, pred = res
            got[(name, yr, seed, r, salt)] = pred
            if (k + 1) % 80 == 0:
                log(f"  맞춤 {k + 1}/{len(tasks)}")
    run_md.append(f"- 맞춤 {len(tasks)}회 실제 벽시계 {time.time() - t3:.0f}s.")
    log(run_md[-1])

    def merged(name, seed=None, r=None, salt=0) -> dict[int, np.ndarray]:
        out = {}
        for yr in TEST_YEARS:
            if seed is None:
                avg = MD.average_seed_preds({s: got[(name, yr, s, r, salt)] for s in SEEDS})
                out.update(avg)
            else:
                out.update(got[(name, yr, seed, r, salt)])
        return out

    preds = {"M": merged("M"), "S": merged("S")}
    preds["E"] = {t: MD.score_E(Z[t]) for t in oos}
    preds["R"] = {t: MD.score_R(Z[t]) for t in oos}
    preds["P"] = {t: MD.score_P(Z[t]) for t in oos}
    L = {}
    for yr in TEST_YEARS:
        X, Y, w, _ = MD.train_matrix(train[yr], Z, U, y, vol)
        beta = MD.fit_ridge(X, Y, w)
        L.update({t: MD.predict_ridge(beta, Z[t]) for t in test[yr]})
    preds["L"] = L
    pickle.dump({"preds": preds, "oos": oos, "codes": P["codes"], "dates": dates, "U": U},
                open(cache / "s1_preds.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # 결정성(§8)
    det_same = True
    det_cells = 0
    for m in ("M", "S"):
        for yr in TEST_YEARS:
            for s in SEEDS:
                a, b = got[(m, yr, s, None, 0)], got[(m, yr, s, None, 1)]
                for t in a:
                    det_cells += 1
                    det_same &= a[t].tobytes() == b[t].tobytes()
    log(f"결정성 같음 {det_same} ({det_cells}칸)")

    # IC 와 진단·시드
    ics = {m: ic_series(preds[m], y, U, vol, oos, window) for m in MODELS}
    stats = {m: stat_of(ics[m], dates) for m in MODELS}
    seed_ic = {m: {str(s): stat_of(ic_series(merged(m, seed=s), y, U, vol, oos, window), dates)["mean"] for s in SEEDS}
               for m in ("M", "S")}
    d_me = delta_nw(ics["M"], ics["E"])
    d_ms = delta_nw(ics["M"], ics["S"])
    d_ml = delta_nw(ics["M"], ics["L"])

    # 널 공유 입력
    y_u = {t: y[t][U[t]] for t in oos}
    ok_base = {t: np.isfinite(y_u[t]) & tradable(vol, t)[U[t]] for t in oos}
    u_star = np.unique(np.concatenate([U[t] for t in oos]))
    N = len(P["codes"])
    days_by = {m: sorted(ics[m]) for m in MODELS}
    _N.clear()
    _N.update(u_star=u_star, N=N, names=list(MODELS), days=days_by, pred=preds, U=U, y_u=y_u, ok_base=ok_base,
              Z=Z, days_all=oos)

    # 널 실측 3회 → 외삽 → RUN.md(§10)
    t4 = time.time()
    with mp.get_context("fork").Pool(3) as pool:
        _ = pool.map(_n2_task, [0, 1, 2], chunksize=1)
    n2_s = (time.time() - t4)  # 3회를 작업자 3개로 — 한 번의 벽시계
    run_md.append(f"- N2 3회 실측(후보·진단 6개, 작업자 3): 회당 약 {n2_s:.1f}s → 1,000회 / 작업자 {args.workers} 외삽 약 {n2_s * N2_REPS / args.workers / 60:.1f}분.")
    log(run_md[-1])
    run_md.append("- 로컬 외삽이 12시간을 훨씬 밑돈다 — Modal 불요(§10).")

    # N1(§8) — 판정 전에 평가
    n1_means = []
    for r in range(N1_REPS):
        p = merged("M", r=r)
        n1_means.append(stat_of(ic_series(p, y, U, vol, oos, window), dates)["mean"])
    n1_mean = float(np.mean(n1_means))

    # N2·N3
    t5 = time.time()
    n2 = {m: np.full((N2_REPS, 2), np.nan) for m in MODELS}
    fill = {m: np.zeros(len(days_by[m]), dtype=np.int64) for m in MODELS}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, (r, res, fl) in enumerate(pool.imap_unordered(_n2_task, range(N2_REPS), chunksize=8)):
            for m in MODELS:
                n2[m][r] = res[m]
                fill[m] += fl[m]
            if (k + 1) % 200 == 0:
                log(f"  N2 {k + 1}/{N2_REPS}")
    run_md.append(f"- N2 {N2_REPS}회 실제 벽시계 {time.time() - t5:.0f}s.")
    t6 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        n3_rows = list(pool.imap_unordered(_n3_task, range(N3_REPS), chunksize=4))
    run_md.append(f"- N3 {N3_REPS}회 실제 벽시계 {time.time() - t6:.0f}s.")
    n3_means = np.array([x[1] for x in sorted(n3_rows)])
    n3_q95 = NL.q95(n3_means)

    null = {}
    fill_rep = {}
    for m in MODELS:
        null[m] = {"q95_mean": NL.q95(n2[m][:, 0]), "q95_icir": NL.q95(n2[m][:, 1]), "mean_of_means": float(np.nanmean(n2[m][:, 0]))}
        sizes = np.array([len(U[t]) for t in days_by[m]], dtype=np.float64)
        share = fill[m] / (N2_REPS * sizes)
        fill_rep[m] = {"overall": float(fill[m].sum() / (N2_REPS * sizes.sum())), "by_day_min": float(share.min()),
                       "by_day_median": float(np.median(share)), "by_day_max": float(share.max())}
    n1_q = null["M"]["q95_mean"]

    # VOID(§8) 먼저
    void = []
    if len(oos) < C.IC_DAYS_MIN:
        void.append(f"IC OOS 격자일 {len(oos)} < {C.IC_DAYS_MIN}")
    for m in CANDIDATES:
        if stats[m]["n_days"] < C.IC_DAYS_MIN:
            void.append(f"후보 {m} 유효 IC 격자일 {stats[m]['n_days']} < {C.IC_DAYS_MIN}")
    if u_median < C.U_MEDIAN_MIN:
        void.append(f"U(t) 중앙값 {u_median} < {C.U_MEDIAN_MIN}")
    if n1_mean > n1_q:
        void.append(f"N1 누출 카나리아 {n1_mean:.5f} > M 의 N2 q95 {n1_q:.5f}")
    if not det_same:
        void.append("같은 시드 재적합 예측이 비트 단위로 다르다")
    if dq["exit"] != 0:
        void.append(f"조정주가 검사 실패(exit {dq['exit']})")
    fp1 = D.fingerprint_range("2017-01-01", END)
    if fp0 != fp1:
        void.append("실행 중 표 지문이 바뀌었다")

    bars = {"M": bars_for("M", stats["M"], null["M"], None, d_me, d_ms),
            "E": bars_for("E", stats["E"], null["E"], n3_q95, None, None)}
    entered = {m: enters_stage2(m, bars[m]) for m in CANDIDATES}
    verdict = verdict_stage1(void, entered)
    interact = bool(d_ml["mean"] > 0 and d_ml["t_nw6"] >= T_BAR)

    res = {
        "status": "검증 전", "prereg": PREREG_SHA, "corrections": ["ML-1"], "end": END, "core_ref": args.core_ref,
        "lightgbm": _lgb_version(), "grid": {"oos_days": len(oos), "first": str(dates[oos[0]].date()), "last": str(dates[oos[-1]].date()),
                                             "train_phase_days": len(rows)},
        "universe": {"median_oos": u_median, "U_star": int(u_star.size)},
        "train": train_info, "label_report_all_phase_days": rep_all, "label_report_oos": rep_oos, "seam_in_tails": seam_rep,
        "stats": stats, "null_n2": null, "null_n3": {"q95_mean": n3_q95, "mean": float(np.mean(n3_means)), "reps": len(n3_means),
                                                     "days_min": int(min(x[2] for x in n3_rows))},
        "fill": fill_rep, "seed_ic_mean": seed_ic,
        "delta": {"M-E": d_me, "M-S": d_ms, "M-L": d_ml},
        "bars": bars, "enters_stage2": entered, "verdict": verdict, "void": void,
        "claim_label": {"M>E(I-4)": bars["M"]["I-4"], "M-L_t≥2(상호작용·비선형 이득 이름)": interact},
        "n1": {"reps": N1_REPS, "mean": n1_mean, "means": n1_means, "m_n2_q95": n1_q, "void": bool(n1_mean > n1_q)},
        "determinism": {"same": bool(det_same), "cells": det_cells},
        "data_check": dq, "fingerprints": {"before": fp0, "after": fp1, "panel_2019_end": P["info"]["fingerprint"]},
        "elapsed_s": time.time() - t_start,
        "label_public": "재사용 창 — 알파 주장 불가. 2020~23 은 진화 폴드·B·옛 ML-PEAD 가 모두 썼다.",
        "limits": ["N1 은 피처가 미래와 연결된 누출을 못 잡는다(§8). 절단 동일성은 원천이 나중에 고친 값을 못 잡는다(§8) — 예: 2026-10 백필 naver_investor_trend, 정정 리포트 목표가."],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "S1.md").write_text(render(json.loads((OUT / "S1.json").read_text())))
    run_md.append(f"- 1단계 전체 벽시계 {res['elapsed_s']:.0f}s.")
    with open(OUT / "RUN.md", "w") as fh:
        fh.write("\n".join(run_md) + "\n")
    log(f"1단계 판정 {verdict}")
    return 0


def _lgb_version() -> str:
    import lightgbm

    assert lightgbm.__version__ == C.LGB_VERSION, f"LightGBM {lightgbm.__version__} ≠ 고정 {C.LGB_VERSION}"
    return lightgbm.__version__


def _yes(b) -> str:
    return "통과" if b else "미달"


def render(r: dict) -> str:
    """S1.json → 표. 숫자는 JSON 에서만 온다."""
    st, nl = r["stats"], r["null_n2"]
    L = [f"# ML 수익률 예측 1단계 — IC ({r['status']})", "",
         f"> **{r['status']}.** 리뷰(`quant-reviewer`)·재계산(`haiku-verifier`) 전 숫자다. {r['label_public']}", "",
         f"사전등록 `{r['prereg']}` + 정정 {', '.join(r['corrections'])}. END {r['end']}, core `{r['core_ref']}`, LightGBM {r['lightgbm']}. 원본 `S1.json`.", "",
         f"**1단계 판정: {r['verdict']}**" + (f" — VOID 사유: {r['void']}" if r["void"] else ""), "",
         f"OOS IC 격자 {r['grid']['oos_days']}일({r['grid']['first']}~{r['grid']['last']}, 5거래일). U(t) 중앙 {r['universe']['median_oos']:.0f}, U* {r['universe']['U_star']}.", "",
         "## 표 1 — IC 와 널", "",
         "| 모델 | 평균 IC | sd | ICIR | N2 q95 평균 | N2 q95 ICIR | 2020 | 2021 | 2022 | 2023 | 양(+) 연수 | 격자일 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("M", "E", "S", "R", "L", "P"):
        s, n = st[m], nl[m]
        y = s["by_year"]
        L.append(f"| {m} | {s['mean']:+.4f} | {s['sd']:.4f} | {s['icir']:+.3f} | {n['q95_mean']:+.4f} | {n['q95_icir']:+.3f} | "
                 f"{y['2020']:+.4f} | {y['2021']:+.4f} | {y['2022']:+.4f} | {y['2023']:+.4f} | {s['years_positive']}/4 | {s['n_days']} |")
    d = r["delta"]
    L += ["", f"N3 q95(평균 IC, E 전용) {r['null_n3']['q95_mean']:+.4f} ({r['null_n3']['reps']}회, 결합 평균 {r['null_n3']['mean']:+.4f}).", "",
          "## 표 2 — 바", "",
          "| 바 | M | E |", "|---|---|---|",
          f"| I-1 평균 IC > N2 q95 (E 는 N3 q95 도) | {_yes(r['bars']['M']['I-1'])} | {_yes(r['bars']['E']['I-1'])} |",
          f"| I-2 ICIR > N2 q95 | {_yes(r['bars']['M']['I-2'])} | {_yes(r['bars']['E']['I-2'])} |",
          f"| I-3 연도별 IC > 0 이 ≥ 3/4 | {_yes(r['bars']['M']['I-3'])} | {_yes(r['bars']['E']['I-3'])} |",
          f"| I-4 ΔIC(M−E) 평균 {d['M-E']['mean']:+.4f}, NW t {d['M-E']['t_nw6']:+.2f} (M 선택용) | {_yes(r['bars']['M']['I-4'])} | — |",
          f"| I-5 ΔIC(M−S) 평균 {d['M-S']['mean']:+.4f}, NW t {d['M-S']['t_nw6']:+.2f} | {_yes(r['bars']['M']['I-5'])} | — |",
          f"| 2단계 진입 | {_yes(r['enters_stage2']['M'])} | {_yes(r['enters_stage2']['E'])} |", "",
          f"라벨(바 아님): ΔIC(M−L) 평균 {d['M-L']['mean']:+.4f}, NW t {d['M-L']['t_nw6']:+.2f}. "
          f"\"상호작용·비선형 이득\"이라 쓰려면 t ≥ 2.0 — {r['claim_label']['M-L_t≥2(상호작용·비선형 이득 이름)']}.", "",
          "## VOID 관문(§8)", "",
          f"- N1 학습 널(라벨 섞기 {r['n1']['reps']}회) OOS 평균 IC {r['n1']['mean']:+.5f}, M 의 N2 q95 {r['n1']['m_n2_q95']:+.5f}: "
          f"{'**VOID**' if r['n1']['void'] else '걸리지 않음'}. 20회 범위 [{min(r['n1']['means']):+.4f}, {max(r['n1']['means']):+.4f}].",
          f"- 결정성: 같은 시드 재적합 예측 바이트 같음 {r['determinism']['same']} ({r['determinism']['cells']}칸).",
          f"- 조정주가 검사(`collectors.data_checks`) exit {r['data_check']['exit']}: {r['data_check']['adjusted_lines']}.",
          f"- 표 지문(2017-01-01~END) 시작=끝: {r['fingerprints']['before'] == r['fingerprints']['after']}. 2019-01-01~END: {r['fingerprints']['panel_2019_end']}.", "",
          "## 널 채움 비율 N2(§7)", "", "| 모델 | 전체 | 날짜 최소 | 날짜 중앙 | 날짜 최대 |", "|---|---|---|---|---|"]
    for m, v in r["fill"].items():
        L.append(f"| {m} | {v['overall']:.3f} | {v['by_day_min']:.3f} | {v['by_day_median']:.3f} | {v['by_day_max']:.3f} |")
    lr = r["label_report_oos"]
    L += ["", "## 라벨 보고(§4, OOS 격자일 U(t) 행)", "",
          f"행 {lr['rows']}, 유한·진입 가능 {lr['rows_finite']}, 정지일 셀 {lr['halt_days']}, 정지일 있는 라벨 행 {lr['rows_with_halt']}, "
          f"진입 불가로 뺀 행 {lr['rows_entry_excluded']}, 15일 규칙으로 NaN {lr['rows_nan_15']}, 창 안 폐지 {lr['rows_delist_in_window']}"
          f"(15일 미만인데 보존 {lr['rows_delist_kept_by_exception']}), 이음매 셀이 든 라벨 행 {lr['rows_seam']}.",
          f"이음매 셀이 든 라벨 행 중 그날 y 상·하위 1% 비율: {r['seam_in_tails']}.", "",
          "## 학습(§6)", "", "| 시험 연도 | T0 | 학습 행 | 학습 날짜 | 처음 | 마지막 |", "|---|---|---|---|---|---|"]
    for yr, v in r["train"].items():
        L.append(f"| {yr} | {v['t0']} | {v['rows']} | {v['days']} | {v['first']} | {v['last']} |")
    L += ["", "2020 폴드 학습은 2019 한 해 안쪽으로 얇다(§6).", "",
          "시드별 OOS 평균 IC(선택에 쓰지 않는다): " + "; ".join(
              f"{m} " + ", ".join(f"{k}:{v:+.4f}" for k, v in r["seed_ic_mean"][m].items()) for m in ("M", "S")), "",
          "## 한계(§8 공개)", ""] + [f"- {x}" for x in r["limits"]] + ["", f"_{r['elapsed_s']:.0f}초_"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--core-ref", default="13c56a7")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        (OUT / "S1.md").write_text(render(json.loads((OUT / "S1.json").read_text())))
        return 0
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    return run(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
