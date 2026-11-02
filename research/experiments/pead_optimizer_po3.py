"""PEAD 최적화 정정 PO-3 — 위험모델 v4(특이위험 VRA·수축) 구창 격자 선택 + v4 신창 검정의 밴드·탐지(구창 모의).

판정 러너가 아니다. 사전등록: ``research/logs/pead_optimizer/PREREG.md`` 정정 PO-3(5판). 이 러너는 PO-3.7-2 만 낸다.
- PO-3.3 새 이음매 가드(``seam_guard(secondary_min_abs_ret=0.3)``), 가드 끈 R0(DB 확인), 코어 기본값이 ``d5c1533`` 과
  같은 모델을 내는지(PO-2 캐시의 가드 끈 모델과 비트 대조).
- PO-3.5 격자 12칸(VRA 반감기 {끔, 21, 42, 84} × spec_shrink {0.1, 0.3, 0.6}) — 칸마다 위험모델·σ_target·OB·실현
  b_OB21·b_OB5, 선택 규칙 L = |b21 − 1| + |b5 − 1|(동률 규칙 고정).
- PO-3.6 고른 v4 의 구창 구조 모의(T=31·주간 124, 10,000회): α·밴드·κ, D×1.5·D×0.67 탐지(VOID 판정), 보고만 D×0.5·D×2·F×1.5.
- **2023-12-28 뒤 데이터는 읽지 않는다**(PO-2 러너와 같은 읽기 함수).

숫자만 낸다(GUARDRAILS §8). 선택·VOID 는 사전등록 규칙의 기계적 적용이다.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" \\
      python research/experiments/pead_optimizer_po3.py --cache <scratch>/po3cache \\
        --po2-cache <scratch>/po2cache [--workers 16]

출력: ``research/logs/pead_optimizer/PO3.json``·``PO3.md``. 캐시는 레포 밖.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pead_optimizer_po2 as P2
from krx_quant_core.backtest.crosssectional import _trailing_adv
from krx_quant_core.data.risk_inputs import STYLE_NAMES, load_risk_inputs
from krx_quant_core.risk.factor_model import factor_returns
from pead_optimizer_c0 import (
    END,
    GRID_LAST,
    GRID_START,
    HORIZON,
    N_OB,
    N_P,
    OB_SECTOR_BAND,
    STEP,
    bias_rows,
    eligible_idx,
    keyed_rng,
    nu_from,
)
from pead_optimizer_seam import seam_guard

from swing_it.book import riskmodel as _riskmodel

GRID_VRA = (None, 21, 42, 84)
GRID_SHRINK = (0.1, 0.3, 0.6)
V3_CELL = (None, 0.3)
TIE = 0.002
SEC_MIN_ABS_RET = 0.3
OB_OK_MIN = 0.90
DETECT_MIN = 0.8
FAMILY_PASS = 0.90
VOID_WORLDS = ("Dx1.5", "Dx0.67")
SCEN = ["correct", "Dx1.5", "Dx0.67", "Dx0.5", "Dx2", "Fx1.5"]
OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"


def cell_name(cell) -> str:
    h, s = cell
    return f"vra{'off' if h is None else h}_s{s}"


# ── 위험모델(병렬, 칸 인자) ───────────────────────────────────────────────────

def build_models(c, need, workers, path: Path, log, *, vra=None, shrink=0.3):
    """구현은 :func:`swing_it.book.riskmodel.build_models` 로 옮겼다. 캐시가 입력 해시에 묶인다(옛 구현은 설정을 무시했다)."""
    return _riskmodel.build_models(c, need, workers, path, log, vra=vra, shrink=shrink)


def as_model(models, a):
    return _riskmodel.as_risk_model(models, a)


# ── σ_target·OB·실현 ───────────────────────────────────────────────────────────


def sigma_target_of(models, grid, sig, adv, cap, nN):
    """PO-1.6 — 격자일(k ≥ 2) R 목표 비중의 사전 21일 능동위험 중앙값(PO-2 러너 4단계와 같은 식)."""
    tranche = {}
    for t in grid:
        _, idx = as_model(models, t)
        in_u = np.zeros(nN, bool)
        in_u[idx] = True
        el = eligible_idx(t, sig, adv, cap, in_u)
        top = el[np.argsort(-sig[el, t], kind="stable")[:10]] if el.size >= 20 else None
        tranche[t] = (el, top)
    rows = []
    for k, t in enumerate(grid):
        if k < 2:
            continue
        m, idx = as_model(models, t)
        pos = {cc: i for i, cc in enumerate(idx)}
        el, _ = tranche[t]
        w = np.zeros(len(idx))
        nt = 0
        for j in range(3):
            top = tranche[grid[k - j]][1]
            if top is None:
                continue
            nt += 1
            for cc in top:
                if cc in pos:
                    w[pos[cc]] += 1.0 / len(top)
        if nt == 0 or w.sum() <= 0:
            continue
        w /= w.sum()
        b = np.zeros(len(idx))
        eu = [pos[cc] for cc in el if cc in pos]
        b[eu] = 1.0 / len(eu)
        rows.append(float(np.sqrt(HORIZON * m.var(w - b))))
    return float(np.median(rows)), rows


def b_universe(models, asofs, sig, adv, cap, nN):
    bu = {}
    for n, a in enumerate(asofs):
        _, idx = as_model(models, a)
        in_u = np.zeros(nN, bool)
        in_u[idx] = True
        el = eligible_idx(a, sig, adv, cap, in_u)
        pos = {cc: i for i, cc in enumerate(idx)}
        bu[n] = np.array([pos[cc] for cc in el], dtype=int)
    return bu


def solve_ob(models, asofs, bu, lab, sigma_target, workers, path: Path):
    if path.exists():
        return pickle.load(open(path, "rb"))
    tasks = []
    for n, a in enumerate(asofs):
        m, idx = as_model(models, a)
        s = bu[n]
        tasks.append((n, m.X[s], m.F, m.D[s], m.names, lab[idx[s], a].astype(object), sigma_target, OB_SECTOR_BAND))
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for n, W, st in pool.imap_unordered(P2._ob_task, tasks):
            res[n] = (W, st)
    pickle.dump(res, open(path, "wb"))
    return res


def evaluate(c, models, bu, ob, seam_s, U):
    """구창 월말 루프 — 실현 z(보고만 대상 + OB21·OB5), OB 모의 구조, 사전 몫, 실현 특이 z(PO-2 러너 7단계와 같은 식)."""
    R, cap, lab, asofs, sectors, first = c["R"], c["cap"], c["lab"], c["asofs"], c["sectors"], c["first"]
    rng_t = {k: keyed_rng(k) for k in ("P5", "P50", "P10B")}
    names_t = ([f"S:{k}" for k in STYLE_NAMES] + ["P5", "P50", "P10B", "HV", "SM"]
               + [f"IP:{g}" for g in sectors] + [f"IE:{g}" for g in sectors])
    Zr = {k: [] for k in names_t}
    A = len(asofs)
    Z21 = np.full((N_OB, A), np.nan)
    Z5 = np.full((N_OB, A, 4), np.nan)
    struct, spec_z = [], []
    expo = {"max_abs_sector_active": [], "sector_var_share": [], "factor_var_share": []}
    nan_seam = nan_other = 0
    for n, a in enumerate(asofs):
        m, idx = as_model(models, a)
        nA = len(idx)
        fr = R[idx, a + 1:a + 1 + HORIZON]
        bad = ~np.isfinite(fr)
        sw = seam_s[idx, a + 1:a + 1 + HORIZON]
        nan_seam += int((bad & sw).sum())
        nan_other += int((bad & ~sw).sum())
        fd = np.nan_to_num(fr, nan=0.0)
        fwd = fd.sum(axis=1)
        wk = np.stack([fd[:, 5 * k:5 * k + 5].sum(axis=1) for k in range(4)], axis=1)
        Wr = np.sqrt(cap[idx, a])
        P = np.linalg.pinv((m.X * Wr[:, None]).T @ m.X) @ (m.X * Wr[:, None]).T
        bU = np.full(nA, 1.0 / nA)

        def zs(Wm, fwd=fwd, m=m):
            Wm = np.atleast_2d(Wm)
            Y = Wm @ m.X
            v = np.einsum("pk,kl,pl->p", Y, m.F, Y) + (Wm * Wm) @ m.D
            return (Wm @ fwd) / np.sqrt(HORIZON * v)

        for k in STYLE_NAMES:
            Zr[f"S:{k}"].append(zs(P[m.names.index(k)]))
        for key, size in (("P5", 5), ("P50", 50)):
            Wm = np.zeros((N_P, nA))
            for p in range(N_P):
                Wm[p, rng_t[key].choice(nA, size, replace=False)] = 1.0 / size
            Zr[key].append(zs(Wm - bU))
        s = bu[n]
        bB = np.zeros(nA)
        bB[s] = 1.0 / len(s)
        Wm = np.zeros((N_P, nA))
        for p in range(N_P):
            Wm[p, rng_t["P10B"].choice(s, 10, replace=False)] = 0.1
        Zr["P10B"].append(zs(Wm - bB))
        for key, raw_k, top in (("HV", "resid_vol", True), ("SM", "size", False)):
            v = c["raw"][raw_k][idx, a + first]
            cut = np.nanquantile(v, 0.9 if top else 0.1)
            sel = (np.isfinite(v) & ((v >= cut) if top else (v <= cut))).astype(float)
            Zr[key].append(zs(sel / sel.sum() - bU))
        labs_a = lab[idx, a]
        for g in sectors:
            Zr[f"IP:{g}"].append(zs(P[m.names.index(g)]) if g in m.names else np.array([np.nan]))
            sel = (labs_a == g).astype(float)
            Zr[f"IE:{g}"].append(zs(sel / sel.sum() - bU) if sel.sum() > 0 else np.array([np.nan]))
        Xs_, Ds_ = m.X[s], m.D[s]
        sec_idx = [m.names.index(g) for g in sectors if g in m.names]
        Wob = ob[n][0]
        good = np.isfinite(Wob).all(axis=1)
        Wg = np.where(good[:, None], Wob, 0.0)
        Y = Wg @ Xs_
        fv = np.einsum("pk,kl,pl->p", Y, m.F, Y)
        sv = (Wg * Wg) @ Ds_
        p1 = np.sqrt(fv + sv)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z21[good, n] = (Wg @ fwd[s])[good] / (np.sqrt(21.0) * p1[good])
            Z5[good, n, :] = (Wg @ wk[s])[good] / (np.sqrt(5.0) * p1[good])[:, None]
            Ys = Y[:, sec_idx]
            Fs = m.F[np.ix_(sec_idx, sec_idx)]
            expo["max_abs_sector_active"].append(np.abs(Ys[good]).max(axis=1))
            expo["sector_var_share"].append((np.einsum("pk,kl,pl->p", Ys, Fs, Ys) / p1 ** 2)[good])
            expo["factor_var_share"].append((fv / p1 ** 2)[good])
        struct.append({"Y": Y, "Wd": Wg * np.sqrt(Ds_)[None, :], "F": m.F, "good": good,
                       "pred1": np.where(good, p1, 1.0),
                       "sec_pos": {g: m.names.index(g) for g in sectors if g in m.names}})
        u = U[idx, a + 1:a + 1 + HORIZON]
        okr = np.isfinite(u).all(axis=1)
        spec_z.append(u[okr].sum(axis=1) / np.sqrt(HORIZON * m.D[okr]))
    realized = {t: bias_rows(np.array([np.atleast_1d(v) for v in Zr[t]], dtype=float).T) for t in names_t}
    realized["OB21"] = bias_rows(Z21)
    realized["OB5"] = bias_rows(Z5.reshape(N_OB, -1))
    return {"realized": realized, "Zr": Zr, "Z21": Z21, "Z5": Z5, "struct": struct, "spec_z": spec_z,
            "expo": expo, "nan_to_zero": {"seam": nan_seam, "other": nan_other}}


# ── 선택·밴드 ─────────────────────────────────────────────────────────────────


def select_cell(rows: list[dict]) -> tuple:
    """PO-3.5 선택 규칙 — OB 성공 ≥ 90% 중 L 최소, (최소 + 0.002) 안 동률은 고정 순서."""
    ok = [r for r in rows if r["ob_ok_share"] >= OB_OK_MIN and np.isfinite(r["L"])]
    if not ok:
        return None, []
    best = min(r["L"] for r in ok)
    tied = [r for r in ok if r["L"] <= best + TIE]

    def key(r):
        h, s = r["cell"]
        knobs = (h is not None) + (s != 0.3)
        return (knobs, -(np.inf if h is None else h), abs(s - 0.3), s)

    tied.sort(key=key)
    return tuple(tied[0]["cell"]), [cell_name(r["cell"]) for r in tied]


def calibrate(sims, se_boot, sectors=None):
    """PO-2.4·PO-3.6 — 맞는 모델 모의로 α(두 검정 동시 통과 0.90, κ=1), κ 확대 최종 밴드, 세계별 탐지."""
    b21c, b5c, _ = sims["correct"]
    c21, c5 = float(np.median(b21c)), float(np.median(b5c))
    se21, se5 = float(np.std(b21c, ddof=1)), float(np.std(b5c, ddof=1))

    def bands(al, k21=1.0, k5=1.0):
        q21 = np.quantile(b21c, [al / 2, 1 - al / 2])
        q5 = np.quantile(b5c, [al / 2, 1 - al / 2])
        return ((c21 - k21 * (c21 - q21[0]), c21 + k21 * (q21[1] - c21)),
                (c5 - k5 * (c5 - q5[0]), c5 + k5 * (q5[1] - c5)))

    def inside(b, band_):
        return (b >= band_[0]) & (b <= band_[1])

    lo_a, hi_a = 1e-5, 0.5
    for _ in range(60):
        mid = 0.5 * (lo_a + hi_a)
        b1, b2 = bands(mid)
        if np.mean(inside(b21c, b1) & inside(b5c, b2)) > FAMILY_PASS:
            lo_a = mid
        else:
            hi_a = mid
    al = lo_a
    k21, k5 = max(1.0, se_boot["OB21"] / se21), max(1.0, se_boot["OB5"] / se5)
    B21, B5 = bands(al, k21, k5)
    det = {}
    for sc, (b21, b5, stopw) in sims.items():
        o21, o5 = ~inside(b21, B21), ~inside(b5, B5)
        det[sc] = {"family": float(np.mean(o21 | o5)), "primary": float(np.mean(o21)),
                   "secondary": float(np.mean(o5)), "b21_median": float(np.median(b21)),
                   "b5_median": float(np.median(b5)), "tripwire_prob": float(np.mean(stopw > 0)),
                   "tripwire_week_median": float(np.median(stopw[stopw > 0])) if (stopw > 0).any() else None}
    return {"alpha": al, "center": {"OB21": c21, "OB5": c5}, "se_sim": {"OB21": se21, "OB5": se5},
            "kappa": {"OB21": k21, "OB5": k5}, "band": {"OB21": B21, "OB5": B5},
            "h": {"OB21": [c21 - B21[0], B21[1] - c21], "OB5": [c5 - B5[0], B5[1] - c5]},
            "family_pass_correct_final": 1.0 - det["correct"]["family"], "detect": det}


# ── 본체 ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="레포 밖 캐시 디렉터리")
    ap.add_argument("--po2-cache", required=True, help="PO-2 캐시(가드 끈 모델 비트 대조용)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--render", action="store_true", help="PO3.json → PO3.md 만")
    args = ap.parse_args()
    if args.render:
        (OUT / "PO3.md").write_text(render_md(json.loads((OUT / "PO3.json").read_text())))
        return 0
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    assert repo not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    # 1. 입력 + 새 이음매 가드(PO-3.3) -------------------------------------------
    x0 = load_risk_inputs("2019-01-02", END)
    assert x0.dates[-1] <= pd.Timestamp(END), "END 뒤 날짜가 들어왔다"
    raw_src, events = P2.raw_closes_and_events(x0.codes, x0.dates)
    rf, cf, seam, seam_info = seam_guard(x0.codes, x0.dates, x0.ret, x0.close, raw_src, events,
                                         secondary_min_abs_ret=SEC_MIN_ABS_RET)
    _, _, seam_old, seam_info_old = seam_guard(x0.codes, x0.dates, x0.ret, x0.close, raw_src, events)
    del raw_src
    x = dataclasses.replace(x0, ret=rf, close=cf)
    log(f"이음매(새 규칙) {seam_info['seam_cells']}셀 원천별 {seam_info['seam_cells_by_source']}, "
        f"supply_demand 근거로 남김 {seam_info['kept_secondary_undeterminable']} (옛 규칙 {seam_info_old['seam_cells']})")

    # 2. 가드 끈 R0 + 코어 기본값 비트 대조 ---------------------------------------
    c_raw = P2.prep(x0)
    m_raw = build_models(c_raw, c_raw["asofs"], args.workers, cache / "models_noguard.pkl", log)
    r0_noguard = P2.r0_values(c_raw, m_raw)
    old = pickle.load(open(Path(args.po2_cache) / "models_noguard.pkl", "rb"))
    same_keys = sorted(old) == sorted(m_raw)
    bit_same = same_keys and all(
        all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(old[a][:4], m_raw[a][:4]))
        and old[a][4] == m_raw[a][4] for a in old)
    log(f"R0 가드 끔 최대 차 {r0_noguard['max_abs_diff']:.4f}, 코어 기본값 비트 동일 {bit_same}")
    del c_raw, m_raw, old
    if not (r0_noguard["match_0.005"] and bit_same):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "PO3.json").write_text(json.dumps({"R0_noguard": r0_noguard, "core_default_bit_same": bit_same,
                                                  "STOP": "R0 불일치 또는 코어 기본값 불일치"},
                                                 ensure_ascii=False, indent=1, default=float))
        return 1

    # 3. 가드 켠 입력 -----------------------------------------------------------------
    c = P2.prep(x)
    R, cap, lab, dates, asofs, sectors, first = (c["R"], c["cap"], c["lab"], c["dates"], c["asofs"],
                                                 c["sectors"], c["first"])
    seam_s = seam[:, first:]
    nN, nD = R.shape
    sig_all, sig_info = P2.yoy_signal_strict(x.codes, x.dates)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    sig, adv = sig_all[:, first:], adv_all[:, first:]
    t_0 = int(np.searchsorted(dates, np.datetime64(GRID_START)))
    grid = [t for t in range(t_0, nD, STEP) if dates[t] <= pd.Timestamp(GRID_LAST)]
    need = sorted(set(asofs) | set(grid))
    upath = cache / "resid_guard3.npy"
    if upath.exists():
        U = np.load(upath)
    else:
        U = factor_returns(R, c["Xs"], lab, cap, c["msk"]).u
        np.save(upath, U)

    # 4. 격자 12칸 -------------------------------------------------------------------
    rows, keep, models_by = [], {}, {}
    for s in GRID_SHRINK:
        for h in GRID_VRA:
            cell = (h, s)
            nm = cell_name(cell)
            models = build_models(c, need, args.workers, cache / f"models_{nm}.pkl", log, vra=h, shrink=s)
            models_by[nm] = models
            st, st_rows = sigma_target_of(models, grid, sig, adv, cap, nN)
            bu = b_universe(models, asofs, sig, adv, cap, nN)
            ob = solve_ob(models, asofs, bu, lab, st, args.workers, cache / f"ob_{nm}_{st:.6f}.pkl")
            ob_stat = {k: int(sum(ob[n][1][k] for n in ob)) for k in ("ok", "lam_fail", "solver_fail", "solves")}
            ev = evaluate(c, models, bu, ob, seam_s, U)
            b21, b5 = ev["realized"]["OB21"], ev["realized"]["OB5"]
            row = {"cell": [h, s], "name": nm, "OB21": b21, "OB5": b5, "L": abs(b21 - 1) + abs(b5 - 1),
                   "sigma_target": st, "sigma_target_q10_q90": [float(np.quantile(st_rows, 0.1)),
                                                                 float(np.quantile(st_rows, 0.9))],
                   "ob": ob_stat, "ob_ok_share": ob_stat["ok"] / (N_OB * len(asofs)),
                   "B_universe_size": [int(min(len(v) for v in bu.values())), int(max(len(v) for v in bu.values()))]}
            rows.append(row)
            keep[nm] = (ev, bu, st)
            log(f"칸 {nm}: OB21 {b21:.4f} OB5 {b5:.4f} L {row['L']:.4f} σ_t {st:.4f} OB 성공 {row['ob_ok_share']:.3f}")
            if cell != V3_CELL:
                keep[nm] = ({k: ev[k] for k in ("realized", "expo", "nan_to_zero")}, None, st)
    # λ_S² = 같은 shrink 의 VRA 끔 칸 대비 D 비(asof 마다 상수)
    for row in rows:
        h, s = row["cell"]
        if h is None:
            row["lambda2_q10_50_90"] = [1.0, 1.0, 1.0]
            continue
        mv, mo = models_by[row["name"]], models_by[cell_name((None, s))]
        lam2 = np.array([float(np.median(mv[a][3] / mo[a][3])) for a in asofs])
        row["lambda2_q10_50_90"] = [float(v) for v in np.quantile(lam2, [0.1, 0.5, 0.9])]
        row["lambda2_by_asof"] = [float(v) for v in lam2]
    chosen, tied = select_cell(rows)
    log(f"선택 {chosen} (동률 후보 {tied})")
    del models_by
    if chosen is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "PO3.json").write_text(json.dumps({"grid": rows, "STOP": "OB 성공 ≥ 90% 인 칸 없음"},
                                                 ensure_ascii=False, indent=1, default=float))
        return 1
    v4name = cell_name(chosen)
    if v4name == cell_name(V3_CELL):
        ev4 = keep[v4name][0]
    else:
        models4 = pickle.load(open(cache / f"models_{v4name}.pkl", "rb"))
        st4 = keep[v4name][2]
        bu4 = b_universe(models4, asofs, sig, adv, cap, nN)
        ob4 = solve_ob(models4, asofs, bu4, lab, st4, args.workers, cache / f"ob_{v4name}_{st4:.6f}.pkl")
        ev4 = evaluate(c, models4, bu4, ob4, seam_s, U)
        del models4
    ev3 = keep[cell_name(V3_CELL)][0]

    # 5. v4 꼬리·혼합(PO-2.2 추정식) ---------------------------------------------------
    cv, s_med = P2.cv_estimates(ev4["spec_z"])
    cv_s = cv["CV_med"]
    nu_u, k_u, n_u = nu_from(np.concatenate([z / np.sqrt(sv) for z, sv in zip(ev4["spec_z"], s_med)]))
    nu_u_unclipped = 4.0 + 6.0 / max(k_u, 0.2)
    fac_t = [f"S:{k}" for k in STYLE_NAMES] + [f"IP:{g}" for g in sectors]
    pooled = []
    for t in fac_t:
        zz = np.concatenate([np.atleast_1d(v) for v in ev4["Zr"][t]])
        zz = zz[np.isfinite(zz)]
        pooled.append(zz / zz.std(ddof=1))
    nu_f, k_f, n_f = nu_from(np.concatenate(pooled))
    log(f"v4 CV_med {cv_s:.3f} split {cv['CV_split']:.3f} ν_u {nu_u:.2f}(κ {k_u:.2f}) ν_f {nu_f:.2f}(κ {k_f:.2f})")

    # 6. v4 모의(PO-2.4 생성식, T=31, 10,000회) -----------------------------------------
    sig_ln = float(np.sqrt(np.log(1.0 + cv_s ** 2)))
    P2._SIM.clear()
    P2._SIM.update({"struct": {"v4": ev4["struct"]}, "nu_f": nu_f, "nu_u": nu_u, "sig_ln": {"v4": sig_ln}})
    sims = {}
    with mp.get_context("fork").Pool(min(args.workers, len(SCEN))) as pool:
        for _, sc, b21, b5, stopw in pool.imap_unordered(P2._sim_task, [("v4", sc) for sc in SCEN]):
            sims[sc] = (b21, b5, stopw)
            log(f"  모의 {sc} b21 중앙 {np.nanmedian(b21):.3f} b5 {np.nanmedian(b5):.3f}")
    P2._SIM.clear()
    se_boot = P2.boot_se_ob(ev4["Z21"], ev4["Z5"], P2.T_GATE)
    cal = calibrate(sims, se_boot)
    void_det = {w: cal["detect"][w]["family"] for w in VOID_WORLDS}
    void = {"detect_family": void_det,
            "rule": "D×1.5·D×0.67 세계에서 A 실패(OB21 또는 OB5 밴드 밖) 확률 ≥ 0.8, 아니면 VOID(신창 미사용, 전진만)",
            "void": bool(any(v < DETECT_MIN for v in void_det.values()))}
    b4 = cal["band"]
    in_band_old = {"OB21": bool(b4["OB21"][0] <= ev4["realized"]["OB21"] <= b4["OB21"][1]),
                   "OB5": bool(b4["OB5"][0] <= ev4["realized"]["OB5"] <= b4["OB5"][1])}
    log(f"밴드 OB21 {b4['OB21']} OB5 {b4['OB5']} 탐지 {void_det} VOID {void['void']}")

    # 7. 출력 -----------------------------------------------------------------------------
    def q(v):
        v = np.concatenate([np.atleast_1d(x_) for x_ in v])
        return [float(x_) for x_ in np.quantile(v, [0.1, 0.5, 0.9])]

    res = {
        "prereg": {"PO-2": "f34a9c4", "PO-2_result": "f1a08db", "PO-3": "4bdf40e"}, "core_ref": "13c56a7",
        "window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())], "n_asof": len(asofs),
        "data_end": str(dates[-1].date()), "sectors": sectors,
        "seam_new": seam_info, "seam_old_rule_cells": seam_info_old["seam_cells"],
        "seam_old_rule_by_source": seam_info_old["seam_cells_by_source"],
        "signal": sig_info, "R0_noguard": r0_noguard, "core_default_bit_same_as_d5c1533": bit_same,
        "grid": rows, "chosen": {"cell": list(chosen), "name": v4name, "tied_within": tied},
        "v3": {"realized": ev3["realized"], "nan_to_zero": ev3["nan_to_zero"],
               "exposure": {k: q(v) for k, v in ev3["expo"].items()}},
        "v4": {"realized": ev4["realized"], "nan_to_zero": ev4["nan_to_zero"],
               "exposure": {k: q(v) for k, v in ev4["expo"].items()},
               "tails": {**cv, "nu_u": nu_u, "nu_u_unclipped": nu_u_unclipped, "kurt_u": k_u, "n_u": n_u,
                         "nu_f": nu_f, "kurt_f": k_f, "n_f": n_f},
               "se_boot_T31": se_boot, "calibration": cal, "void_check": void,
               "old_realized_in_new_band_report_only": in_band_old},
        "sim": {"N_SIM": P2.N_SIM, "T_gate": P2.T_GATE, "blocks": list(P2.BLOCKS), "N_BOOT": P2.N_BOOT,
                "scenarios": SCEN},
        "elapsed_s": time.time() - t0,
    }

    def rnd(o):
        if isinstance(o, dict):
            return {str(k): rnd(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rnd(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return round(float(o), 5)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PO3.json").write_text(json.dumps(rnd(res), ensure_ascii=False, indent=1))
    (OUT / "PO3.md").write_text(render_md(json.loads((OUT / "PO3.json").read_text())))
    log(f"저장 {OUT / 'PO3.json'}")
    return 0


# ── PO3.md ────────────────────────────────────────────────────────────────────


def render_md(r: dict) -> str:
    """PO3.json → 표. 숫자는 JSON 에서만 온다."""
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    v3, v4 = r["v3"], r["v4"]
    cal, vc = v4["calibration"], v4["void_check"]
    s = r["seam_new"]
    ch = r["chosen"]
    L = ["# 정정 PO-3 — 위험모델 v4 구창 선택 + v4 신창 검정의 밴드·탐지 (검증 전)", "",
         f"사전등록 PO-3 `{r['prereg']['PO-3']}`(PO-2 `{r['prereg']['PO-2']}`, PO-2 결과 `{r['prereg']['PO-2_result']}`), "
         f"core `{r['core_ref']}`. 구창 asof {r['window'][0]}~{r['window'][1]} ({r['n_asof']}회), 데이터 끝 {r['data_end']}. "
         "**2023-12-28 뒤 데이터는 읽지 않았다. 신창은 읽지 않았다.** 재현: `research/experiments/pead_optimizer_po3.py` 머리 주석.", "",
         "> **표본 안 튜닝이다.** v4 의 구창 OB 값은 같은 52개월에서 12칸 중 고른 값이라 증거가 아니다. 검증은 PO-3.6 신창 검정(독립 리뷰 뒤)뿐이다.", "",
         "## 신창 검정 사전 숫자 (PO-3.6 — 신창 전에 고정)", "",
         "D·최적화 바이어스 검정 — F·업종은 O 에 무관(사전 몫 약 6%), 검정 안 함.", "",
         f"- v4 = **{ch['name']}** (VRA 반감기 {ch['cell'][0] if ch['cell'][0] is not None else '끔'}, spec_shrink {ch['cell'][1]}). "
         f"σ_target(고정) = **{next(g['sigma_target'] for g in r['grid'] if g['name'] == ch['name']):.4f}**.",
         f"- α = {cal['alpha']:.4f}(맞는 모델에서 두 검정 동시 통과 0.90, κ=1), κ 뒤 동시 통과 {cal['family_pass_correct_final']:.3f}.", "",
         "| 검정 | 중심 | 밴드 | h− | h+ | SE 모의 | SE 부트 | κ |", "|---|---|---|---|---|---|---|---|"]
    for k in ("OB21", "OB5"):
        L.append(f"| {k} | {cal['center'][k]:.3f} | [{cal['band'][k][0]:.3f}, {cal['band'][k][1]:.3f}] | {cal['h'][k][0]:.3f} | "
                 f"{cal['h'][k][1]:.3f} | {cal['se_sim'][k]:.3f} | {v4['se_boot_T31'][k]:.3f} | {cal['kappa'][k]:.2f} |")
    L += ["", "탐지확률(T=31, 10,000회, 공통 난수). VOID 판정은 가족(A 실패) 열이다.", "",
          "| 세계 | A 실패(가족) | 1차 OB21 | 보조 OB5 | b21 중앙 | b5 중앙 | 트립와이어 확률 | 트립와이어 주 중앙 |",
          "|---|---|---|---|---|---|---|---|"]
    for sc, d in cal["detect"].items():
        L.append(f"| {sc} | {d['family']:.3f} | {d['primary']:.3f} | {d['secondary']:.3f} | {d['b21_median']:.3f} | "
                 f"{d['b5_median']:.3f} | {d['tripwire_prob']:.3f} | {f3(d['tripwire_week_median'])} |")
    L += ["", f"**VOID 판정: {'VOID — 신창 미사용, v4 는 전진으로만' if vc['void'] else 'VOID 아님 — 신창 1회 사용 가능(독립 리뷰 뒤)'}** "
          f"(D×1.5 {vc['detect_family']['Dx1.5']:.3f}, D×0.67 {vc['detect_family']['Dx0.67']:.3f}, 바 ≥ 0.8).", "",
          "## 격자 (PO-3.5, 구창 표본 안)", "",
          "| 칸 | OB21 | OB5 | L | σ_target | OB 성공 | λ_S² 10·50·90% |", "|---|---|---|---|---|---|---|"]
    for g in r["grid"]:
        mark = " **(v4)**" if g["name"] == ch["name"] else (" (v3)" if g["cell"] == [None, 0.3] else "")
        L.append(f"| {g['name']}{mark} | {g['OB21']:.4f} | {g['OB5']:.4f} | {g['L']:.4f} | {g['sigma_target']:.4f} | "
                 f"{g['ob_ok_share']:.3f} | " + "·".join(f"{x:.3f}" for x in g["lambda2_q10_50_90"]) + " |")
    g_by = {g["name"]: g for g in r["grid"]}
    shr = sorted({g["cell"][1] for g in r["grid"]})
    off = [g_by[f"vraoff_s{x}"] for x in shr]
    L += ["", f"선택 규칙: L = |OB21 − 1| + |OB5 − 1| 최소(OB 성공 ≥ 0.90), 동률(+0.002) 후보 {', '.join(ch['tied_within'])}.", "",
          "**관찰(검증 전, 판정 아님).**",
          "- VRA 는 켤 때마다 나빠졌다. 시총가중 λ_S² 중앙값이 1 아래다(" + ", ".join(
              f"s={x}: {g_by[f'vra42_s{x}']['lambda2_q10_50_90'][1]:.3f}" for x in shr)
          + ", h=42). 시총 상위 종목의 특이위험은 과대예측이고, 그 승수가 B 유니버스(시총 21~100위)의 D 까지 낮춘다. "
          "이 시총가중 VRA 로는 OB 과소예측이 고쳐지지 않는다(국면 지연이 원인인지는 이것으로 가려지지 않는다).",
          "- 수축 강도 하나가 OB 를 거의 다 움직인다(VRA 끔: " + ", ".join(
              f"s={g['cell'][1]} OB21 {g['OB21']:.3f}·OB5 {g['OB5']:.3f}" for g in off)
          + "). 수축은 D̂ 을 규모 그룹 평균 쪽으로 눌러 종목 사이 차이를 지운다. OB 의 실현/사전 분산비는 대략 "
          "종목 평균 D_i/D̂_i 라서, D̂ 이 압축될수록 커진다(가설 — 이 러너는 기제를 따로 검증하지 않았다).",
          f"- **고른 칸은 격자 경계(shrink 최솟값 {shr[0]})다.** L 은 shrink 를 줄이는 쪽으로 단조 감소했다. 최적이 0.1 아래일 수 "
          "있고, 고원이 아니라 경계를 골랐다. 격자는 사전등록대로 넓히지 않는다.",
          "- 남은 것: OB5 가 OB21 보다 약 4% 크다(주간 쪽 과소예측이 남는다). 업종 순수 포트 G55 는 v4 에서 더 커졌다(보고만, 검정 안 함).",
          "- 코어 `spec_vra_half_life` 는 v4 에서 쓰이지 않는다(끔). v4 = v3 에 `spec_shrink=0.1` 만 바꾼 모델이다.", "",
          "## v3 대 v4 — 구창 실현 바이어스 (새 가드, 보고만)", "",
          "| 대상 | v3 | v4 |", "|---|---|---|"]
    for k in v3["realized"]:
        L.append(f"| {k} | {v3['realized'][k]:.3f} | {v4['realized'][k]:.3f} |")
    L += ["", f"v4 구창 실현이 v4 신창 밴드 안인가(보고만, T=52 대 T=31 근사): OB21 {v4['old_realized_in_new_band_report_only']['OB21']}, "
          f"OB5 {v4['old_realized_in_new_band_report_only']['OB5']}.", "",
          "| 모델 | OB 팩터 분산 몫 10·50·90% | 업종 몫 10·50·90% | 업종 능동 노출 최대 10·50·90% | NaN→0 셀 이음매/그 밖 |",
          "|---|---|---|---|---|"]
    for nm, v in (("v3", v3), ("v4", v4)):
        e = v["exposure"]
        L.append(f"| {nm} | " + " | ".join("·".join(f"{x:.3f}" for x in e[k])
                                         for k in ("factor_var_share", "sector_var_share", "max_abs_sector_active"))
                 + f" | {v['nan_to_zero']['seam']}/{v['nan_to_zero']['other']} |")
    t = v4["tails"]
    L += ["", "## v4 꼬리·혼합 (PO-2.2 추정식)", "",
          f"CV_s 1차(중앙값-z²) {t['CV_med']:.3f} [5·50·95% " + "·".join(f"{x:.3f}" for x in t["CV_med_q05_50_95"]) + "], "
          f"확인(윈저화 반분) {t['CV_split']:.3f} [" + "·".join(f"{x:.3f}" for x in t["CV_split_q05_50_95"]) + "]. "
          f"ν_u {t['nu_u']:.2f}(자르기 전 {t['nu_u_unclipped']:.2f}, 초과첨도 {t['kurt_u']:.2f}, n={t['n_u']}), "
          f"ν_f {t['nu_f']:.2f}(초과첨도 {t['kurt_f']:.2f}, n={t['n_f']}).", "",
          "## 이음매 가드 (PO-3.3 새 규칙)", "",
          f"이음매 {s['seam_cells']}셀({s['seam_codes']}종목) — 원천별(daily_bars_raw, supply_demand) {s['seam_cells_by_source']}, "
          f"후보 {s['candidates_by_source']}. supply_demand 근거라 남긴 셀 {s['kept_secondary_undeterminable']}"
          f"(그중 |수익| > 0.1 {s['kept_secondary_abs_ret_gt_0.1']}). 옛 규칙 {r['seam_old_rule_cells']}셀"
          f"(원천별 {r['seam_old_rule_by_source']}). 판정 불가 {s['cells_undeterminable']}"
          f"(|수익| > 0.3 {s['cells_undeterminable_abs_ret_gt_0.3']}, 남김).", "",
          f"R0 가드 끔 최대 차 {r['R0_noguard']['max_abs_diff']:.4f}(일치 {r['R0_noguard']['match_0.005']}), "
          f"코어 `13c56a7` 기본값 모델이 `d5c1533` 과 비트 동일: {r['core_default_bit_same_as_d5c1533']}.", "",
          f"_{f3(r['elapsed_s'])}초_"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
