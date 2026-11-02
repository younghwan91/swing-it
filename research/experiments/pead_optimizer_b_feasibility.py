"""PEAD 최적화 단계 B — O 실행 가능성 스모크(정정 PO-7, B 결과 전). **수익을 읽지 않는다.**

몇 개 리밸런스일에서 O 를 **사전 정보만으로** 풀어 리밸런스당 풀이 시간·``optimal`` 비율·λ 맞춤 성공을 잰다.
B 의 성과(수익·IR·EV·비용 실현·널)는 하나도 내지 않는다.

- α: t 에 알려진 YoY 신호의 순위 z, t 의 v4 특이분산. **IC 는 추정하지 않는다** — IC_k 는 학습 구간 앞 수익으로 정하는 값이라
  여기서는 [0, 0.10] 자르기 범위 안의 자리표시 값 {0.02, 0.05, 0.10} 으로 바꿔 가며 푼다(α 크기는 λ 맞춤이 상쇄하지만 λ 위치를 옮긴다).
- 위험: t 의 v4 모델(B 격자 캐시 ``models_v4_grid.pkl``). λ 목표: 같은 날 R 결정 목표 비중의 사전 능동위험(t 정보만).
- 비용: 최적화기 안의 벌점(1배, AUM 칸 값). ``no_trade``·변동성·ADV 는 t 까지의 관측만 쓴다(엔진의 t+1 규칙 대신 t — 수익을 피하려고).
- 상태 두 가지: (i) 첫 구축(w_prev=0, 회전율 상한 없음), (ii) 정상 상태 — 직전 격자일 O 해(표류 없음 — 표류는 지난 수익이라 쓰지 않는다)를
  w_prev 로, 회전율 상한 2/3.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" \\
      python research/experiments/pead_optimizer_b_feasibility.py --cache <scratch>/bparity --po3-cache <scratch>/po3cache

출력: ``research/logs/pead_optimizer/B_feasibility.json``·``B_feasibility.md``(시간·개수만).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pead_optimizer_b as B

N_DATES = 6
IC_PLACEHOLDER = (0.0, 0.01, 0.02, 0.05, 0.10)  # 0 = IC_k 가 0 으로 잘린 폴드(α=0)
CELLS = (B.PRIMARY, B.Cell(20, 0.05), B.Cell(60, 0.10, 1e10))
RUNS = {"primary": 1, "sensitivity": 6, "null_O": B.N_PERM}
OUT = B.OUT


def pick_dates(ctx: B.Ctx, n: int = N_DATES) -> list[int]:
    """OOS(2020-01~2023-12) 격자 순번 중 고르게 n 개(첫 구축 앞 격자가 있도록 k ≥ 3)."""
    ks = [k for k, t in enumerate(ctx.grid) if B.EVAL[0] <= str(ctx.dates[t].date()) <= B.EVAL[1] and k >= 3]
    pos = np.linspace(0, len(ks) - 1, n).round().astype(int)
    return [ks[i] for i in pos]


def r_target_exante(ctx: B.Ctx, k: int) -> np.ndarray:
    """R 결정 목표 비중(결정일 t, 합 1, U_t 안) — 활성 트랜치(k, k−1, k−2), t 까지의 정보만."""
    t = ctx.grid[k]
    in_u = ctx.in_u(t)
    gone = B.delisted(ctx, t)
    parts = []
    for j in range(k, max(-1, k - B.R_HORIZON // B.STEP), -1):
        tj = ctx.grid[j]
        b = B.tranche_names(tj, ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(tj))
        if b is None:
            continue
        b = np.asarray(b, dtype=int)
        b = b[in_u[b] & ~gone[b] & np.isfinite(ctx.Cff[b, t])]
        if b.size:
            parts.append(b)
    w = np.zeros(ctx.C.shape[0])
    for b in parts:
        w[b] += 1.0 / (len(parts) * b.size)
    return w


def sigma_r_exante(ctx: B.Ctx, k: int) -> float:
    """같은 날 R 목표 비중의 사전 20일 능동위험(``simulate`` 의 R 사전 위험 블록과 같은 식)."""
    t = ctx.grid[k]
    in_u = ctx.in_u(t)
    el = B.eligible(t, ctx.sig, ctx.adv, ctx.cap, in_u)
    w = r_target_exante(ctx, k) * in_u
    if w.sum() <= 0 or el.size == 0:
        return float("nan")
    w = w / w.sum()
    nN = ctx.C.shape[0]
    rows = np.flatnonzero(in_u & ((w > 0) | np.isin(np.arange(nN), el)))
    sub = B.sub_model(ctx.models[t], rows, B.STEP)
    b = np.zeros(nN)
    b[el] = 1.0 / el.size
    return B.active_risk(sub, (w - b)[rows])


def solve_at(ctx: B.Ctx, cell: B.Cell, k: int, ic: float, w_prev: np.ndarray | None, sigma_r: float) -> dict:
    """O 를 결정일 t 에서 한 번 푼다(사전 정보만). 체결일 대신 t 를 쓴다."""
    t = ctx.grid[k]
    in_u = ctx.in_u(t)
    el = B.eligible(t, ctx.sig, ctx.adv, ctx.cap, in_u)
    nN = ctx.C.shape[0]
    w_t = np.zeros(nN) if w_prev is None else w_prev
    held = w_t > 0
    tradeable = np.isfinite(ctx.C[:, t]) & (np.nan_to_num(ctx.tv_won[:, t]) > 0)
    vol = B.trailing_vol(ctx.ret, t)
    adv_won = ctx.adv[:, t] * 1e6
    costable = np.isfinite(vol) & np.isfinite(adv_won) & (adv_won >= 1e6) & (vol <= 1.0)
    tradeable &= costable
    frozen = held & ~tradeable
    n_solves = [0]
    orig = B.match_lambda

    def counting(solve, risk_of, target):
        out = orig(solve, risk_of, target)
        n_solves[0] = out[3]
        return out

    B.match_lambda = counting
    t0 = time.perf_counter()
    try:
        want, info = B.o_target(ctx, cell, t, t, el, w_t, held, in_u, tradeable, costable, frozen, vol, ctx.sig,
                                {int(ctx.dates[t].year): ic}, {t: sigma_r}, w_prev is not None, None)
    finally:
        B.match_lambda = orig
    dt = time.perf_counter() - t0
    return {"k": k, "date": str(ctx.dates[t].date()), "cell": cell.name, "ic": ic, "state": "steady" if w_prev is not None else "build",
            "seconds": dt, "n_solves": n_solves[0], "status": info["status"], "lam": info["lam"], "lam_hit": bool(info["lam_hit"]),
            "lam_cause": info.get("lam_cause", ""), "n_dust": int(info.get("n_dust", 0)),
            "n_names": int((want > 0).sum()),
            "n_universe": int(el.size), "risk_ratio": float(info.get("exante_20", float("nan")) / sigma_r),
            "weights": want}


_G: dict = {}


def _task(a):
    cell_i, k, ic = a
    ctx = _G["ctx"]
    cell = CELLS[cell_i]
    out = []
    s_prev = sigma_r_exante(ctx, k - 1)
    s_now = sigma_r_exante(ctx, k)
    r1 = solve_at(ctx, cell, k - 1, ic, None, s_prev)
    w1 = r1.pop("weights")
    out.append(r1)
    w_prev = w1 if r1["status"] == "optimal" else None
    r2 = solve_at(ctx, cell, k, ic, w_prev, s_now)
    r2.pop("weights")
    r2["prev_ok"] = w_prev is not None
    out.append(r2)
    return out


def summarize(rows: list[dict], n_grid: int, workers: int) -> dict:
    steady = [r for r in rows if r["state"] == "steady"]
    by = {}
    for key in sorted({(r["cell"], r["ic"]) for r in rows}):
        rr = [r for r in steady if (r["cell"], r["ic"]) == key]
        by[f"{key[0]}|ic{key[1]}"] = {
            "n": len(rr), "sec_median": float(np.median([r["seconds"] for r in rr])),
            "sec_max": float(np.max([r["seconds"] for r in rr])),
            "solves_median": float(np.median([r["n_solves"] for r in rr])),
            "optimal_share": float(np.mean([r["status"] == "optimal" for r in rr])),
            "lam_hit_share": float(np.mean([r["lam_hit"] for r in rr])),
            "causes": {c: int(sum(r["lam_cause"] == c for r in rr)) for c in ("up", "down", "gap", "solver")},
            "names_median": float(np.median([r["n_names"] for r in rr]))}
    sec = np.array([r["seconds"] for r in steady])
    per_reb = float(np.median(sec))
    per_reb_p90 = float(np.quantile(sec, 0.9))
    n_o_runs = RUNS["primary"] + RUNS["sensitivity"] + RUNS["null_O"]
    cpu_h = n_o_runs * n_grid * per_reb / 3600.0
    cpu_h90 = n_o_runs * n_grid * per_reb_p90 / 3600.0
    return {
        "by_cell_ic": by,
        "steady": {"n": len(steady), "sec_median": per_reb, "sec_p90": per_reb_p90,
                   "optimal_share": float(np.mean([r["status"] == "optimal" for r in steady])),
                   "lam_hit_share": float(np.mean([r["lam_hit"] for r in steady]))},
        "build": {"n": len(rows) - len(steady),
                  "optimal_share": float(np.mean([r["status"] == "optimal" for r in rows if r["state"] == "build"])),
                  "lam_hit_share": float(np.mean([r["lam_hit"] for r in rows if r["state"] == "build"]))},
        "extrapolation": {"o_runs": n_o_runs, "rebalances_per_run": n_grid, "cpu_hours_median": cpu_h,
                          "cpu_hours_p90": cpu_h90, "workers": workers,
                          "wall_hours_median": cpu_h / workers, "wall_hours_p90": cpu_h90 / workers,
                          "note": "R 북·R 널 300·토글은 최적화가 없어 무시할 만하다. 체결일 표류·강제 매도는 문제 크기를 거의 안 바꾼다."},
    }


def render(r: dict) -> str:
    s, b, e = r["summary"]["steady"], r["summary"]["build"], r["summary"]["extrapolation"]
    L = ["# 단계 B — O 실행 가능성 스모크 (정정 PO-7, B 결과 전 — 수익을 읽지 않았다)", "",
         f"core `{r['core_ref']}`, END {r['end']}, 표 {r['table']}. 리밸런스일 {len(r['dates'])}개: {', '.join(r['dates'])}.",
         "α 는 t 의 YoY 순위 z·v4 특이분산, IC 는 **추정하지 않은 자리표시 값** "
         f"{list(IC_PLACEHOLDER)}. λ 목표 = 같은 날 R 결정 목표의 사전 능동위험. 비용은 최적화기 벌점(1배).",
         "정상 상태 = 직전 격자일 O 해(표류 없음)를 w_prev 로, 회전율 상한 2/3. 첫 구축 = w_prev 0. "
         f"λ 탐색 [{r['lam_lo']:g}, {B.LAM_HI:g}], 최대 {B.LAM_ITERS}회, ±{B.LAM_TOL:.0%}.", "",
         "| 칸 · IC | 정상 상태 n | 초(중앙) | 초(최대) | 풀이 수(중앙) | optimal | λ 맞춤 | 실패 원인 up/down/gap/solver | 보유 수(중앙) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, d in r["summary"]["by_cell_ic"].items():
        c = d["causes"]
        L.append(f"| {key} | {d['n']} | {d['sec_median']:.2f} | {d['sec_max']:.2f} | {d['solves_median']:.0f} | "
                 f"{d['optimal_share']:.2f} | {d['lam_hit_share']:.2f} | {c['up']}/{c['down']}/{c['gap']}/{c['solver']} | {d['names_median']:.0f} |")
    L += ["", "아래 '전체' 두 줄은 자리표시 IC 다섯 값을 섞은 것이다(IC 0·0.01 이 실패를 끌어올린다). 칸·IC 별 표를 본다.",
          f"- 정상 상태 전체: n {s['n']}, 리밸런스당 {s['sec_median']:.2f}초(중앙)·{s['sec_p90']:.2f}초(90%), "
          f"optimal {s['optimal_share']:.3f}, λ 맞춤 {s['lam_hit_share']:.3f}.",
          f"- 첫 구축: n {b['n']}, optimal {b['optimal_share']:.3f}, λ 맞춤 {b['lam_hit_share']:.3f}.",
          f"- 외삽: O 실행 {e['o_runs']}회(1차 1 + 민감도 6 + O 널 {RUNS['null_O']}) × 리밸런스 {e['rebalances_per_run']} → "
          f"CPU {e['cpu_hours_median']:.1f}시간(중앙)·{e['cpu_hours_p90']:.1f}시간(90%), 작업자 {e['workers']} 이면 벽시계 "
          f"{e['wall_hours_median']:.1f}·{e['wall_hours_p90']:.1f}시간. {e['note']}", "",
          f"_{r['elapsed_s']:.0f}초_"]
    return "\n".join(L) + "\n"


def repo_of() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="B 격자 모델 캐시(레포 밖)")
    ap.add_argument("--po3-cache", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--core-ref", default="13c56a7")
    ap.add_argument("--lam-lo", type=float, default=None,
                    help="제안 검토용(PO-7): λ 탐색 하한을 바꿔 본다. 주면 출력은 --out 디렉터리(레포 밖)에만 쓴다")
    ap.add_argument("--out", default="", help="--lam-lo 를 줄 때의 출력 디렉터리(레포 밖)")
    args = ap.parse_args()
    out_dir = OUT
    if args.lam_lo is not None:
        assert args.out and repo_of() not in Path(args.out).resolve().parents, "--lam-lo 실행은 레포 밖에만 쓴다"
        B.LAM_LO = args.lam_lo  # match_lambda 가 모듈 전역을 읽는다 — 이 프로세스(와 fork 작업자)에서만
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    assert repo_of() not in Path(args.cache).resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    ctx, info, _ = B.build_ctx(Path(args.cache), Path(args.po3_cache), args.workers, log)
    ks = pick_dates(ctx)
    dates = [str(ctx.dates[ctx.grid[k]].date()) for k in ks]
    log(f"리밸런스일 {dates}")
    tasks = [(ci, k, ic) for ci in range(len(CELLS)) for k in ks for ic in IC_PLACEHOLDER]
    _G["ctx"] = ctx
    rows = []
    with mp.get_context("fork").Pool(min(args.workers, len(tasks))) as pool:
        for out in pool.imap_unordered(_task, tasks):
            rows += out
            log(f"  {out[-1]['cell']} {out[-1]['date']} ic {out[-1]['ic']}: {out[-1]['seconds']:.1f}s "
                f"{out[-1]['status']} λ맞춤 {out[-1]['lam_hit']} 풀이 {out[-1]['n_solves']}")
    _G.clear()
    rows.sort(key=lambda r: (r["cell"], r["ic"], r["k"], r["state"]))
    res = {"mode": "feasibility", "core_ref": args.core_ref, "end": B.END_B, "table": "daily_bars_adjusted(지금 표 — PR #12 전)",
           "dates": dates, "grid": info["grid"], "cells": [c.name for c in CELLS], "ic_placeholder": list(IC_PLACEHOLDER),
           "rows": rows, "summary": summarize(rows, len(ctx.grid), args.workers), "elapsed_s": time.time() - t0,
           "note": "수익·IR·EV·실현 비용·널은 계산하지 않았다. 시간·상태·개수만 적는다."}
    res = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else float(o)))
    res["lam_lo"] = B.LAM_LO
    (out_dir / "B_feasibility.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (out_dir / "B_feasibility.md").write_text(render(res))
    log(f"저장 {out_dir / 'B_feasibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
