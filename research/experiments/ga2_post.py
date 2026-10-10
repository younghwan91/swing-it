#!/usr/bin/env python
"""GA2 채택 규칙(교체판) 숫자 — ``ga2_run.py`` 실행이 끝난 뒤 그 산출물(``ga2/runs.csv``·``ga2/noise.csv``) 위에서 계산한다. **판정 아님.**

설계: ``research/logs/sepa_explore/GA2.md`` 마지막 절 "변경 — 채택 규칙 교체"(9d48f11, 리뷰어 B ``REVIEW_GA2_B.md``, GA2 결과 보기 전).
처리량 적합도 · 시드 유전자 중앙값 config · 연 폴드 6개 기준:

- (1) 6폴드 중 ≥ 4 폴드 OOS 포트폴리오 초과 > 0 (유지).
- (2′) 이어 붙인 OOS 연환산 초과 > **같은 청산 유전자를 공유하는 무작위 진입 설정 300개**의 95 백분위. 무작위 진입 = GA2 진입 유전자 공간
  (연속 [0,1] 균등 — 끔 구역 포함, 이산 단계 균등)에서 폴드마다 따로 뽑고, 청산 유전자는 그 폴드 중앙값 config 의 것. 그 무작위 설정을 실제와 **같은
  이어 붙인 OOS 절차**(하나의 연속 포트폴리오, 폴드 경계 시가평가)에 넣는다. **활동 하한:** 이어 붙인 OOS 진입 ≥ 100 인 설정만 300개가 찰 때까지
  뽑는다(거래가 거의 없는 무작위 설정은 연환산 0 근처라 기준선을 낮춘다 — 더 엄격한 쪽, 결과 보기 전 결정). 최대 3,000번 뽑기.
- (2″) 이어 붙인 OOS 날별 초과의 팩터 회귀(KOSPI·KOSDAQ 등가중 시장, 12-1 모멘텀 L/S, ADV 사이즈, 저변동; Newey-West 5) 알파 t > 2.
  팩터 = 리뷰어 A 의 ``factors.py`` 산출 ``factors.pkl``(``--factors``), 회귀 = 리뷰어 B ``factor_reg.py`` 의 ``nw_ols`` 와 같은 식.
- (3) WFE 삭제.
- (4) PBO = **각 폴드 학습창 안**에서 CMA-ES 가 평가한 memo config 들의 학습창 월별 포트폴리오 초과 행렬 CSCV(조각 12, Sharpe) — 폴드 PBO 의 중앙값 < 0.5.
  실행 중 프로세스가 memo 를 저장하지 않았으므로(런마다 해시만 남김) 같은 시드·예산으로 처리량 런을 **결정적으로 다시 돌려**(같은 난수·같은 적합도 →
  같은 평가 순서) memo 를 되살린다. 되살린 최적이 ``runs.csv`` 의 최적과 같은지 확인해 적는다.
- 노이즈: 이어 붙인 OOS 연환산 초과 > 노이즈 30회의 95 백분위(유지).
- DSR: N = 보고한 변형 수(≈ 108) 와 검증 구조 수(≈ 10) 두 가지(고유 config 수 아님).
- 통과가 입증하는 것/아닌 것: ``REVIEW_GA2_B.md`` §8 문구 그대로.

Run (``ga2_run.py`` 가 끝난 뒤):
  NUMBA_NUM_THREADS=8 uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/ga2_post.py \\
      --cache <npy 디렉터리> --factors <factors.pkl> --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import ga2_run as g2  # noqa: E402
import ga_run as g1  # noqa: E402


OUT_DIR = "research/logs/sepa_explore"
N_RANDOM, ACTIVE_MIN, MAX_DRAWS = 300, 100, 3000
RANDOM_SEED = 20261010
FACTOR_COLS = ["MKT_KOSPI_EW", "MKT_KOSDAQ_EW", "UMD", "SMB_ADV", "LOWVOL"]
N_VARIANTS, N_STRUCTURES = 108, 10
SECTION_START = "<!-- 채택 규칙 교체판 시작 -->"
SECTION_END = "<!-- 채택 규칙 교체판 끝 -->"


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 순수 헬퍼
# ---------------------------------------------------------------------------

def genes_of(row: pd.Series) -> dict:
    """``runs.csv`` 한 행 → 유전자 dict(연속 NaN = 끔, 이산 = 가장 가까운 단계; inf 문자열 처리)."""
    g = {}
    for name, lo, hi, off in g2.G2_CONT:
        v = row[name]
        g[name] = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
    for name, levels in g2.G2_DISC:
        v = float(row[name])
        g[name] = next((L for L in levels if (math.isinf(L) and math.isinf(v)) or (not math.isinf(L) and abs(L - v) < 1e-9)),
                       min(levels, key=lambda L: abs(L - v) if np.isfinite(v) and not math.isinf(L) else 1e9))
    return g


def random_entry(rng: np.random.Generator, exit_from: dict) -> dict:
    """GA2 진입 유전자 공간에서 균등 뽑기(연속 [0,1] — 끔 구역 포함, 이산 단계 균등) + ``exit_from`` 의 청산 유전자."""
    x = np.r_[rng.random(len(g2.G2_CONT)), [rng.integers(0, k) for k in g2.DISC_SIZES]].astype(float)
    g = g2.g2_decode(x)
    for k in g2.EXIT_GENES:
        g[k] = exit_from[k]
    return g


def nw_ols(y: np.ndarray, X: np.ndarray, lags: int = 5) -> dict:
    """OLS + Newey-West(Bartlett) 공분산 — 리뷰어 B ``factor_reg.nw_ols`` 와 같은 식. X 에 상수 열 포함."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None])
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = (X[lag:] * u[lag:, None]).T @ (X[:-lag] * u[:-lag, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    r2 = 1.0 - (u @ u) / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return {"b": b, "se": se, "t": b / np.where(se > 0, se, np.nan), "r2": float(r2), "n": int(len(y))}


def factor_regress(y: pd.Series, F: pd.DataFrame, lags: int = 5) -> dict:
    d = pd.concat([y.rename("y"), F[FACTOR_COLS]], axis=1, sort=True).dropna()
    if len(d) < 60:
        return {"n": int(len(d))}
    X = np.column_stack([np.ones(len(d)), d[FACTOR_COLS].to_numpy()])
    r = nw_ols(d["y"].to_numpy(), X, lags)
    fm = d[FACTOR_COLS].mean().to_numpy()
    return {"n": r["n"], "mean_ann": float(d["y"].mean() * 252), "alpha_ann": float(r["b"][0] * 252),
            "alpha_t": float(r["t"][0]), "r2": r["r2"], "explained_ann": float((r["b"][1:] * fm).sum() * 252),
            "betas": {c: float(v) for c, v in zip(FACTOR_COLS, r["b"][1:], strict=True)},
            "beta_t": {c: float(v) for c, v in zip(FACTOR_COLS, r["t"][1:], strict=True)}}


# ---------------------------------------------------------------------------
# 작업자(spawn) — memo 되살리기 · 무작위 기준선
# ---------------------------------------------------------------------------

def replay_task(t: dict) -> dict:
    """처리량 폴드 런을 같은 시드·예산으로 다시 돌려 memo(평가한 config 와 학습창 월별 포트폴리오 초과)를 되살린다."""
    E, pre = g2._W["E"], g2._W["pre"]
    tr = g2.Win(pre, *t["win"])
    memo: dict = {}

    def fn(x):
        genes = g2.g2_decode(x)
        key = g2.g2_key(genes)
        if key not in memo:
            spec = g2.g2_spec(genes)
            fit = g2.fitness(E, "port", spec, tr)[0]
            daily, _, _, _ = E.portfolio(spec, tr.i0, tr.i1, tr.d_lo, tr.end, tr.cut)
            memo[key] = (fit, E.monthly(daily, tr.d_lo, tr.d_first, tr.end).astype(np.float32))
        return memo[key][0]

    hist, _ = g1.cma_bipop(fn, len(g2.G2_CONT), g2.DISC_SIZES, budget=t["budget"], seed=t["seed"], stall=True)
    bi = int(np.argmax([h[2] for h in hist]))
    return {"fold": t["fold"], "seed": t["seed"], "best_key": g2.g2_key(g2.g2_decode(hist[bi][1])), "best_fit": hist[bi][2],
            "keys": list(memo), "fits": np.array([memo[k][0] for k in memo]),
            "M": np.column_stack([memo[k][1] for k in memo]) if memo else np.zeros((0, 0))}


def random_task(t: dict) -> dict:
    """무작위 진입 설정 한 묶음 → 각 설정의 이어 붙인 OOS 연환산 초과·진입 수."""
    E, pre = g2._W["E"], g2._W["pre"]
    wins = [(g2.Win(pre, *a), g2.Win(pre, *b)) for a, b in t["fold_args"]]
    rng = np.random.default_rng(t["seed"])
    out = []
    for _ in range(t["n"]):
        cfgs = [random_entry(rng, med) for med in t["meds"]]
        st = g2.stitched(E, wins, cfgs)
        out.append({"ann": st["port"]["ann"], "trades": st["port"]["trades"], "calm": st["calm_ann"],
                    "folds_pos": int(sum(1 for fp in st["fold_port"] if np.isfinite(fp["ann"]) and fp["ann"] > 0))})
    return {"rows": out}


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
    from swing_it.diagnostics.pbo import pbo_cscv
    from swing_it.validation.walkforward import FOLDS

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--factors", required=True, help="리뷰어 A factors.py 산출 pickle(날짜 × 팩터)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=int, default=0, help="0 = runs.csv 의 런 예산(원장 요약 행)")
    args = ap.parse_args(argv)
    t0 = time.time()
    out_dir = args.out
    runs = pd.read_csv(os.path.join(out_dir, "ga2", "runs.csv"))
    parts = runs["run"].str.split("/", expand=True)
    runs["tag"], runs["fold"], runs["kind"], runs["seed"] = parts[0], parts[1], parts[2], parts[3]
    wf = runs[(runs.tag == "wf") & (runs.kind == "port")].copy()
    wf["fold"], wf["seed"] = wf["fold"].astype(int), wf["seed"].astype(int)
    fold_ids = sorted(wf["fold"].unique())
    seeds = sorted(wf["seed"].unique())
    budget = args.budget or int(_budget_from_results(out_dir))
    folds = [FOLDS[k] for k in fold_ids]
    cuts = [FOLDS[k + 1].test_lo if k + 1 < len(FOLDS) else g2.EVENT_HI_NEXT for k in fold_ids]
    fold_args = [((f.train_lo, f.train_hi, f.test_lo), (f.test_lo, f.test_hi, ct)) for f, ct in zip(folds, cuts, strict=True)]
    bests = [[genes_of(wf[(wf.fold == k) & (wf.seed == s)].iloc[0]) for s in seeds] for k in fold_ids]
    meds = [g2.g2_median(b) for b in bests]

    g2.worker_init(args.cache, threads=int(os.environ.get("NUMBA_NUM_THREADS", "8")))
    E, pre = g2._W["E"], g2._W["pre"]
    wins = [(g2.Win(pre, *a), g2.Win(pre, *b)) for a, b in fold_args]
    real = g2.stitched(E, wins, meds)
    # 리뷰 A-M1: 진입 행 = 절단 행(창 마지막 신호일)인 쌍 — 이 판 코드는 뺀다, 실행 산출 표(아래 자동 생성 절)의 트레이드 행에는 들어 있다
    last_tr, last_te = wins[-1]
    sp_last = g2.g2_spec(meds[-1])
    fill_last, _ = E.fill_ctl(sp_last["ek"], sp_last["w_idx"], sp_last["c_idx"])
    s_ = slice(last_te.i0, last_te.i1)
    m_last = E.mask(sp_last, last_te.i0, last_te.i1) & fill_last[s_] & (pre.pd[s_] + 1 >= last_te.cut)
    am1 = {"pairs_in_window_at_cut": int((pre.pd[s_] + 1 >= last_te.cut).sum()), "median_cfg_signals_at_cut": int(m_last.sum())}
    # 분기 재적합(참고) — 하루짜리 조각 분기를 뺀 판(리뷰 A minor)
    qs = g2.quarters()
    q_args = []
    for i, (a, b) in enumerate(qs):
        tr_lo = (pd.Timestamp(a) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        nxt = qs[i + 1][0] if i + 1 < len(qs) else g2.EVENT_HI_NEXT
        q_args.append(((tr_lo, a, a), (a, b, nxt)))
    qrun = runs[(runs.tag == "q")].copy()
    qrun["fold"], qrun["seed"] = qrun["fold"].astype(int), qrun["seed"].astype(int)
    n_q_run = int(qrun["fold"].nunique())
    qps = None
    if n_q_run >= len(q_args):
        qwins = [(g2.Win(pre, *a), g2.Win(pre, *b)) for a, b in q_args]
        qb = [[genes_of(qrun[(qrun.fold == qi) & (qrun.seed == s)].iloc[0]) for s in seeds] for qi in range(len(q_args))]
        qps = g2.proc_stats(E, "port", qwins, qb, bands=("H5", "유동", "중소형"))
        _log(f"분기(조각 분기 제외 {len(q_args)}): 이어 붙인 OOS {qps['dec']['ann']:+.4f}", t0)
    real_ann = real["port"]["ann"]
    fold_pos = [bool(np.isfinite(fp["ann"]) and fp["ann"] > 0) for fp in real["fold_port"]]
    _log(f"실제(처리량·중앙값) 이어 붙인 OOS {real_ann:+.4f} · 폴드 양수 {sum(fold_pos)}/{len(fold_pos)}", t0)

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=g2.worker_init,
                                initargs=(args.cache, 1)) as ex:
        rep_f = [ex.submit(replay_task, {"fold": k, "seed": s, "win": fa[0], "budget": budget})
                 for k, fa in zip(fold_ids, fold_args, strict=True) for s in seeds]
        n_batch = 25
        rnd_f = [ex.submit(random_task, {"fold_args": fold_args, "meds": meds, "n": n_batch, "seed": RANDOM_SEED + j})
                 for j in range(MAX_DRAWS // n_batch)]
        rnd = []
        for f in rnd_f:                                   # 제출 순서대로 모아 300개 활동 설정이 차면 나머지는 취소
            rnd += f.result()["rows"]
            if sum(1 for r in rnd if r["trades"] >= ACTIVE_MIN) >= N_RANDOM:
                for g in rnd_f:
                    g.cancel()
                break
        _log(f"무작위 기준선: 뽑기 {len(rnd)} · 활동 {sum(1 for r in rnd if r['trades'] >= ACTIVE_MIN)}", t0)
        reps = [f.result() for f in rep_f]
    _log("memo 되살리기 끝", t0)

    # --- (2′) 무작위 진입 기준선 ---
    rdf = pd.DataFrame(rnd)
    act = rdf[rdf["trades"] >= ACTIVE_MIN].head(N_RANDOM)
    p95 = float(act["ann"].quantile(0.95)) if len(act) else np.nan
    base = {"n_draws": len(rdf), "n_active": len(act), "p50": float(act["ann"].median()) if len(act) else np.nan, "p95": p95,
            "p95_all_draws": float(rdf["ann"].quantile(0.95)), "pct_real": float((act["ann"] < real_ann).mean()) if len(act) else np.nan}

    # --- (4) 폴드 내 CSCV PBO ---
    pbo_rows, replay_ok = [], []
    for k in fold_ids:
        rs = [r for r in reps if r["fold"] == k]
        keys, cols, fits = {}, [], []
        for r in rs:
            want = g2.g2_key(bests[fold_ids.index(k)][seeds.index(r["seed"])])
            replay_ok.append(bool(r["best_key"] == want))
            for j, key in enumerate(r["keys"]):
                if key not in keys:
                    keys[key] = len(cols)
                    cols.append(r["M"][:, j])
                    fits.append(r["fits"][j])
        M = np.column_stack(cols).astype(float)
        fits = np.array(fits)
        valid = fits > -0.5
        Mv = M[:, valid]
        nb = 12 if M.shape[0] >= 24 else 8
        row = {"fold": k, "test": FOLDS[k].test_lo[:4], "months": M.shape[0], "n_unique": M.shape[1], "n_valid": int(valid.sum())}
        if Mv.shape[1] >= 4:
            pb = pbo_cscv(Mv, n_blocks=nb, metric="sharpe", std_floor=1e-6)
            row.update(pbo=pb["pbo"], pbo_mean=pbo_cscv(Mv, n_blocks=nb, metric="mean", std_floor=1e-6)["pbo"],
                       is_oos_corr=float(np.corrcoef(pb["is_perf"], pb["oos_perf"])[0, 1]))
            top = fits[valid] >= np.quantile(fits[valid], 0.5)
            if top.sum() >= 4:
                row["pbo_top50"] = pbo_cscv(Mv[:, top], n_blocks=nb, metric="sharpe", std_floor=1e-6)["pbo"]
        pbo_rows.append(row)
    pdf = pd.DataFrame(pbo_rows)
    pbo_med = float(pdf["pbo"].median()) if "pbo" in pdf else np.nan

    # --- (2″) 팩터 회귀 ---
    F = pd.read_pickle(args.factors)
    F.index = pd.to_datetime(F.index)
    tests = [te for _, te in wins]
    y = pd.Series(real["daily"], index=pre.days[real["t0"]:tests[-1].end]).iloc[1:]
    fac = {"full": factor_regress(y, F),
           "calm_2022_24": factor_regress(y[(y.index.year >= 2022) & (y.index.year <= 2024)], F),
           "bull_2020_21_2025_26": factor_regress(y[(y.index.year <= 2021) | (y.index.year >= 2025)], F)}

    # --- 노이즈 ---
    nz = pd.read_csv(os.path.join(out_dir, "ga2", "noise.csv"))
    nz_p95 = float(nz["ann"].quantile(0.95)) if len(nz) else np.nan

    # --- DSR (보고 변형 수 · 검증 구조 수) ---
    mo = real["port"]["monthly"]
    dsr = {N: deflated_sharpe_from_sample(mo, N) for N in (N_VARIANTS, N_STRUCTURES)}

    dec = {"r1": sum(fold_pos) >= 4, "r2p": bool(np.isfinite(p95) and real_ann > p95),
           "r2pp": bool(np.isfinite(fac["full"].get("alpha_t", np.nan)) and fac["full"]["alpha_t"] > 2),
           "r4": bool(np.isfinite(pbo_med) and pbo_med < 0.5), "rn": bool(np.isfinite(nz_p95) and real_ann > nz_p95)}
    dec["all"] = all(dec.values())
    global SECTION8_NOTE
    SECTION8_NOTE = "\n".join(x.lstrip("> ").rstrip() for x in _section8() if x.strip() != ">")
    lc = os.path.join(out_dir, "live_candidate.json")
    live_check = write_live(lc, dec, real_ann, p95, fac, pbo_med, nz_p95, fold_pos)
    _log(f"live_candidate.json(스키마 1, write_candidate) 왕복 검사: {live_check}", t0)
    L = render_section(locals())
    md = os.path.join(out_dir, g2.RESULTS_MD)
    text = open(md, encoding="utf-8").read() if os.path.exists(md) else ""
    text = re.sub(re.escape(SECTION_START) + ".*?" + re.escape(SECTION_END) + "\n?", "", text, flags=re.S)
    text = text.replace("## 판단 규칙 숫자 — 처리량 적합도", "## (구 규칙 — 교체됨, 숫자만 참고) 판단 규칙 숫자 — 처리량 적합도", 1)
    lines = text.split("\n")
    at = next((i for i, x in enumerate(lines) if x.startswith("## ")), len(lines))
    lines[at:at] = L
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    rdf.to_csv(os.path.join(out_dir, "ga2", "random_entry_baseline.csv"), index=False)
    pdf.to_csv(os.path.join(out_dir, "ga2", "pbo_fold.csv"), index=False)
    with open(os.path.join(out_dir, "ga2", "factor_reg.json"), "w", encoding="utf-8") as fh:
        json.dump(fac, fh, indent=1, default=float)
    print(f"채택 규칙(교체판): {dec} · 실제 {real_ann:+.4f} · 무작위 p95 {p95:+.4f} · 알파 t {fac['full'].get('alpha_t')} · "
          f"PBO 폴드 중앙 {pbo_med:.3f} · 노이즈 p95 {nz_p95:+.4f}")
    return 0


def write_live(path: str, dec: dict, real_ann: float, p95: float, fac: dict, pbo_med: float, nz_p95: float,
               fold_pos: list) -> str:
    """``ga2_run`` 이 쓴 옛 모양 파일(``config`` 키) → ``swing_it.live.candidate.write_candidate`` 스키마로 다시 쓰고 ``load_candidate`` 로 왕복 검사."""
    import datetime as dt  # noqa: PLC0415

    from swing_it.live.candidate import build_candidate, genes_from_legacy, load_candidate  # noqa: PLC0415

    if not os.path.exists(path):
        return "옛 파일 없음 — 건너뜀"
    old = json.load(open(path, encoding="utf-8"))
    if "config" not in old:                          # 이미 새 스키마(다시 돌린 경우) — 그 genes 를 그대로
        genes = genes_from_legacy({"genes": old["genes"]})
        fit_old, win_old = old.get("fitness", {}), old.get("window", {})
    else:
        genes = genes_from_legacy(old)
        fit_old = {"variant": "port (처리량 — 자본 제약 포트폴리오)", "median_fitness_train": old.get("median_fitness_train"),
                   "median_orders_train": old.get("median_orders_train"), "seed_fitness": old.get("seed_fitness"),
                   "seed_bests": old.get("seed_bests"), "budget_per_seed": old.get("budget_per_seed"),
                   "config_rule": old.get("config_rule")}
        win_old = old.get("window", {})
    fitness = {**fit_old, "decision_rule": "GA2.md '변경 — 채택 규칙 교체'(9d48f11) — 처리량·시드 중앙값·연 폴드",
               "decision_numbers": {"folds_pos": f"{sum(fold_pos)}/{len(fold_pos)}", "stitched_ann": real_ann,
                                    "random_entry_p95": p95, "factor_alpha_t": fac["full"].get("alpha_t"),
                                    "pbo_fold_median": pbo_med, "noise_p95": nz_p95},
               "arithmetic": {k: bool(v) for k, v in dec.items()}}
    doc = build_candidate(genes, fitness=fitness, window=win_old, valid_from=dt.date.today())
    doc["note"] = ("탐색 산출물 — 판정 아님. 이 config 자체의 표본 밖 성적은 없다; 신뢰 근거는 같은 절차의 이어 붙인 워크포워드 OOS 와 "
                   "그 기준선(무작위 진입 p95·노이즈·팩터 알파)이다(GA2_RESULTS.md 채택 규칙 교체판). 전진 페이퍼 전에는 실주문에 쓰지 않는다. "
                   "REVIEW_GA2_B.md §8: " + (SECTION8_NOTE or ""))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    cfg, back = load_candidate(path)
    same = genes_from_legacy({"genes": back["genes"]}) == genes_from_legacy({"genes": doc["genes"]})
    return f"load_candidate 통과 · 유전자 왕복 일치 {same} · strategy_hash {back['strategy_hash']} · code_hash {back['code_hash']}"


def _budget_from_results(out_dir: str) -> int:
    """원장 요약 행(마지막 실행)의 런 예산."""
    path = os.path.join(out_dir, "ga2", "TRIALS.jsonl")
    b = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                c = json.loads(line)["config"]
            except (json.JSONDecodeError, KeyError):
                continue
            if "budget" in c:
                b = int(c["budget"])
    if not b:
        raise SystemExit("원장에서 런 예산을 찾지 못했다 — --budget 을 준다")
    return b


SECTION8_NOTE = None    # REVIEW_GA2_B.md §8 문구(실행 때 파일에서 그대로 읽는다)


def _section8() -> list[str]:
    path = os.path.join(OUT_DIR, "REVIEW_GA2_B.md")
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"## 8\..*?\n(.*?)\n---", txt, flags=re.S)
    body = m.group(1).strip().split("\n") if m else ["(REVIEW_GA2_B.md §8 을 찾지 못함)"]
    return [x for x in body if x.startswith(">")]


def render_section(v: dict) -> list[str]:
    global SECTION8_NOTE
    dec, base, fac, pdf, nz, dsr = v["dec"], v["base"], v["fac"], v["pdf"], v["nz"], v["dsr"]
    f = g1._f
    pct = g1._pct
    s8 = _section8()
    SECTION8_NOTE = "\n".join(x.lstrip("> ").rstrip() for x in s8 if x.strip() != ">")
    ff = fac["full"]
    L = [SECTION_START,
         "## 채택 규칙 숫자 — 교체판 (GA2.md '변경 — 채택 규칙 교체', 9d48f11 · 산술 비교, 판정 아님)", "",
         "처리량 적합도 · 시드 유전자 중앙값 config · 연 폴드 6. 이 표만 ✓/✗ 를 단다 — 아래 문서의 다른 ✓/✗(구 규칙·시드별·분기·띠)는 숫자일 뿐 판정에 쓰지 않는다. "
         "계산: `research/experiments/ga2_post.py`(실행 산출물 위에서). 분기 재적합 변형은 판정에 쓰지 않는다(리뷰 B-M3).", "",
         "| 조건 | 문턱 | 숫자 | 산술 |", "|---|---|---|---|",
         f"| (1) 폴드별 OOS 포트폴리오 초과 > 0 | 6 중 ≥ 4 | {sum(v['fold_pos'])}/{len(v['fold_pos'])} | {'✓' if dec['r1'] else '✗'} |",
         f"| (2′) 이어 붙인 OOS 연환산 초과 > 같은 청산의 무작위 진입 300개 95 백분위 | 실제 > p95 | 실제 {pct(v['real_ann'])} vs p95 "
         f"{pct(base['p95'])} (중앙 {pct(base['p50'])}; 실제의 백분위 {f(base['pct_real'], '{:.0%}')}; 뽑기 {base['n_draws']} 중 활동 ≥ {ACTIVE_MIN} 진입 "
         f"{base['n_active']}개; 활동 제한 없는 p95 {pct(base['p95_all_draws'])}) | {'✓' if dec['r2p'] else '✗'} |",
         f"| (2″) 이어 붙인 OOS 날별 초과의 팩터 알파 HAC t | > 2 | 알파 {pct(ff.get('alpha_ann'))}/년, t {f(ff.get('alpha_t'))} "
         f"(평균 {pct(ff.get('mean_ann'))}/년 중 팩터 설명 {pct(ff.get('explained_ann'))}, R² {f(ff.get('r2'), '{:.3f}')}) | {'✓' if dec['r2pp'] else '✗'} |",
         f"| (4) 폴드 내 CSCV PBO(학습창 memo config, Sharpe, 조각 12) 폴드 중앙값 | < 0.5 | {f(v['pbo_med'], '{:.3f}')} | {'✓' if dec['r4'] else '✗'} |",
         f"| 노이즈: 이어 붙인 OOS 연환산 초과 > 노이즈 {len(nz)}회 95 백분위 | 실제 > p95 | 실제 {pct(v['real_ann'])} vs {pct(v['nz_p95'])} | "
         f"{'✓' if dec['rn'] else '✗'} |",
         f"| 전부 | | | {'✓' if dec['all'] else '✗'} |", "",
         f"- (3) WFE 는 삭제됐다(리뷰 B-M4). DSR(이어 붙인 OOS 월별 초과 {len(v['mo'])}개월, 중앙값 config): N = 보고 변형 {N_VARIANTS} → "
         f"P {f(dsr[N_VARIANTS]['prob_deflated_sharpe'], '{:.1%}')} (관측 월 Sharpe {f(dsr[N_VARIANTS]['observed_sharpe'], '{:.3f}')}, "
         f"H0 기대 최대 {f(dsr[N_VARIANTS]['expected_max_sharpe_h0'], '{:.3f}')}) · N = 검증 구조 {N_STRUCTURES} → "
         f"P {f(dsr[N_STRUCTURES]['prob_deflated_sharpe'], '{:.1%}')} (H0 기대 최대 {f(dsr[N_STRUCTURES]['expected_max_sharpe_h0'], '{:.3f}')}).",
         f"- 2022~24 조각만(참고, 리뷰 (1′)): {pct(v['real']['calm_ann'])}. 무작위 기준선의 (1) 통과율 "
         f"{f(float((v['rdf']['folds_pos'] >= 4).mean()) if len(v['rdf']) else np.nan, '{:.0%}')}.",
         "- 팩터 회귀 쪼개기: " + " · ".join(
             f"{lab} 알파 {pct(r.get('alpha_ann'))}/년 t {f(r.get('alpha_t'))}" for lab, r in
             (("2022~24", fac["calm_2022_24"]), ("2020~21+2025~26", fac["bull_2020_21_2025_26"]))) + " · 베타(전체): "
         + " · ".join(f"{c} {ff.get('betas', {}).get(c, np.nan):+.2f} (t {ff.get('beta_t', {}).get(c, np.nan):+.1f})" for c in FACTOR_COLS),
         f"- 팩터 = 리뷰어 A `factors.py` 산출(KRX 세션 보정 일봉, ADV ≥ 10억 시장 등가중, ADV ≥ 50억 안 12-1 모멘텀·ADV·60일 변동성 상하위 30% 롱숏, 월 리밸런스) "
         f"`{os.path.basename(v['args'].factors)}` — 이 저장소 밖 스크래치 산출물이라 재현에는 그 스크립트가 필요하다.",
         f"- memo 되살리기: 처리량 폴드 런 {len(v['replay_ok'])}개를 같은 시드·예산({v['budget']:,})으로 다시 돌려 최적이 `runs.csv` 와 같은 런 "
         f"{sum(v['replay_ok'])}/{len(v['replay_ok'])}.", "",
         "| 폴드 | 시험 | 학습 월 | memo 고유 config | 하한 위 | PBO(Sharpe) | PBO(평균) | 상위 50% PBO | IS↔OOS 상관 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in pdf.iterrows():
        L.append(f"| {r['fold']} | {r['test']} | {r['months']} | {r['n_unique']:,} | {r['n_valid']:,} | {f(r.get('pbo'), '{:.3f}')} | "
                 f"{f(r.get('pbo_mean'), '{:.3f}')} | {f(r.get('pbo_top50'), '{:.3f}')} | {f(r.get('is_oos_corr'), '{:+.2f}')} |")
    am1 = v["am1"]
    L += ["", f"- 리뷰 A-M1(진입 행 = 절단 행): 마지막 폴드 시험창에서 신호일 2026-04-01(진입 2026-04-02 = 절단 행) 쌍 {am1['pairs_in_window_at_cut']:,}개, "
          f"그중 중앙값 config 신호·체결 {am1['median_cfg_signals_at_cut']}개. 포트폴리오(판정 숫자)에서는 손익이 0 이었고, 이 절의 숫자는 그것을 신호에서 뺀 코드로 "
          "다시 계산했다. **아래 자동 생성 절(실행 당시 코드)의 트레이드 단위 OOS 행·수급 OOS·라이브 창 표에는 이 쌍들이 실제 청산 경로(2026-10 까지, NXT 보정 구간)로 들어 있다.**",
          f"- 아래 자동 생성 절의 분기 재적합 숫자는 하루짜리 조각 분기(2026-04-01 → 04-02, 런 {len(v['seeds'])}개)를 포함한다 — 그것을 뺀 숫자는 바로 아래 표. "
          "또 그 절의 '정체 재시작 … 30·16/λ … 50세대' 문구는 19유전자에선 30·19/λ → 58세대(696 평가)다(코드의 실제 조건은 차원을 그대로 쓴다)."]
    qps = v["qps"]
    if qps is not None:
        qd = qps["dec"]
        L += ["", f"분기 재적합(처리량, 참고 — 판정에 쓰지 않음; 조각 분기 제외 {qd['n_folds']}분기): 양수 분기 {qd['folds_pos']}/{qd['n_folds']} · "
              f"WFE {f(qd['wfe'])} · 2022~24 조각 {pct(qd['calm_ann'])}", "",
              "| 띠 | 연환산 초과 [95% CI] | MDD | 진입(주문) |", "|---|---|---|---|"]
        for b in ("H5", "유동", "중소형"):
            x = qps["st"][b]["port"]
            L.append(f"| {b} | {pct(x['ann'])} [{pct(x.get('lo'))}, {pct(x.get('hi'))}] | {pct(-x['mdd'])} | {x['trades']} ({x['orders']}) |")
    L += ["", f"라이브 인계: `live_candidate.json` 은 `swing_it.live.candidate.write_candidate` 스키마 1 로 썼고 `load_candidate` 왕복 검사 — {v['live_check']}. "
          "페이퍼 실행(판정과 무관하게 형식만): `sw-live <job> --paper --strategy-config research/logs/sepa_explore/live_candidate.json`."]
    L += ["", "### 통과가 확립하는 것과 확립하지 않는 것 (`REVIEW_GA2_B.md` §8 문구 그대로)", "", *s8, "", SECTION_END, ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
