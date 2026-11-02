"""PEAD 최적화 정정 PO-4 — v4 신창 단계 A(A1) 러너와 그 구창 재현 관문. 사전등록 ``PREREG.md`` 정정 PO-4(6판).

as-of 목록과 END 는 **인자**다. ``pead_optimizer_c0.END`` 를 쓰지 않는다 — END 를 굽던 PO-2 러너의 읽기 함수·OB 난수 키는
같은 식의 인자 판으로 바꿨다(``tests/test_pead_optimizer_po4.py`` 가 AST 로 본다).

- ``--mode repro`` (PO-4.1·4.3): 구창(END 2023-12-28, asof 52)에서 PO-3 v4 를 새 캐시로 처음부터 다시 내고, OB21 1.00823·OB5 1.04455·OB 성공 10,339,
  가드 끈 R0·모델 비트·밴드·탐지가 ``PO3.json``·PO-3 캐시와 정확히 같은지 본다. D×1.2·D×0.83 검정력과 판정 문장의 X·Y 를 낸다.
  출력 ``PO4_repro.json``·``PO4_repro.md``. **2023-12-28 뒤 데이터는 읽지 않는다.**
- ``--mode new`` (PO-3.6·PO-4.2): 먼저 구창에서 R0·모델 4개를 다시 확인하고, 표 지문을 잰 뒤 신창(END 2026-09-11, asof 31)을 한 번 판정한다.
  출력 ``A1.json``·``A1.md``. 판정은 사전등록 규칙의 기계적 적용이다(GUARDRAILS §8).

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" \\
      python research/experiments/pead_optimizer_po4.py --mode repro --end 2023-12-28 \\
        --asof-from 2019-07-01 --asof-to 2023-10-31 --n-asof 52 --cache <scratch>/po4repro --po3-cache <scratch>/po3cache
    ... --mode new --end 2026-09-11 --asof-from 2024-01-01 --asof-to 2026-07-31 --n-asof 31 \\
        --cache <scratch>/po4new --po3-cache <scratch>/po3cache

``daily_bars_adj_v2`` 는 v2 민감도(보고만)에서만 읽는다(정문이 없다 — 로더 종목 격자에 맞춰 붙인다).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing as mp
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pead_optimizer_po2 as P2
import pead_optimizer_po3 as P3
from krx_quant_core.backtest.crosssectional import _trailing_adv
from krx_quant_core.data.daily import AFTERMARKET_START, _fingerprint
from krx_quant_core.data.risk_inputs import STYLE_NAMES, load_risk_inputs, style_panels
from krx_quant_core.risk.factor_model import factor_returns
from pead_optimizer_c0 import (
    HORIZON,
    LAM_HI,
    LAM_ITERS,
    LAM_LO,
    LAM_TOL,
    N_OB,
    N_P,
    OB_MAX_WEIGHT,
    OB_SECTOR_BAND,
    STEP,
    Sub,
    bias_rows,
    exposures,
    keyed_rng,
    month_ends,
    nu_from,
)
from pead_optimizer_seam import seam_guard

OLD_END = "2023-12-28"
OLD_ASOF = ("2019-07-01", "2023-10-31", 52)
OLD_GRID = ("2019-08-01", "2023-10-31")  # PO-1.6 σ_target 격자(구창)
LOAD_START = "2019-01-02"
FP_START = date(2019, 1, 1)
V4 = {"vra": None, "shrink": 0.1}
V3 = {"vra": None, "shrink": 0.3}
SEC_MIN_ABS_RET = 0.3
OB_OK_MIN = 0.90
REPRO_TARGET = {"OB21": 1.00823, "OB5": 1.04455, "ob_ok": 10339}
REPRO_SCEN = ["correct", "Dx1.5", "Dx0.67", "Dx1.2", "Dx0.83"]
X_PAIRS = ((1.2, 0.83), (1.5, 0.67))
DETECT_MIN = 0.8
RECHECK_N = (0, 17, 34, 51)
ADV_LIQ = 20000.0  # 백만원 = 200억
CLUSTER_DEV, LIMIT_HIT, LIMIT_KEEP = 0.10, 0.29, 0.05
OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"


def _ts(s) -> pd.Timestamp:
    return pd.Timestamp(s)


def _check_end(end: str) -> None:
    assert _ts(end).date() < AFTERMARKET_START, f"END {end} ≥ {AFTERMARKET_START}(애프터마켓 혼입)"


# ── 읽기(모두 END 인자, 결과 날짜 단언) ─────────────────────────────────────────


def load_inputs(end: str):
    _check_end(end)
    x = load_risk_inputs(LOAD_START, end)
    assert x.dates[-1] <= _ts(end), "로더가 END 뒤 날짜를 냈다"
    return x


def raw_closes_and_events(codes, dates, end: str):
    """PO-2 러너 ``raw_closes_and_events`` 와 같은 식, 상한 = 인자 END."""
    _check_end(end)
    con, cur = P2._db()
    lo = str(dates[0].date())
    cidx = pd.Index([str(c) for c in codes])
    didx = pd.DatetimeIndex(dates)
    out = []
    for sql in ("SELECT code, date, close FROM daily_bars_raw WHERE date BETWEEN %s AND %s AND close > 0",
                "SELECT code, date, abs(close) FROM supply_demand WHERE date BETWEEN %s AND %s "
                "AND close IS NOT NULL AND close <> 0"):
        cur.execute(sql, (lo, end))
        df = pd.DataFrame(cur.fetchall(), columns=["code", "date", "close"])
        assert df.empty or pd.to_datetime(df["date"]).max() <= _ts(end)
        a = np.full((len(codes), len(dates)), np.nan)
        i = cidx.get_indexer(df["code"].astype(str))
        t = didx.get_indexer(pd.to_datetime(df["date"]))
        ok = (i >= 0) & (t >= 0)
        a[i[ok], t[ok]] = df["close"].to_numpy(float)[ok]
        out.append(a)
    cur.execute("SELECT code, date FROM corporate_actions WHERE date BETWEEN %s AND %s", (lo, end))
    ev = pd.DataFrame(cur.fetchall(), columns=["code", "date"])
    assert ev.empty or pd.to_datetime(ev["date"]).max() <= _ts(end)
    con.close()
    return out, ev


def yoy_signal_strict(codes, dates, end: str) -> tuple[np.ndarray, dict]:
    """PO-2.1-5 — PO-2 러너 ``yoy_signal_strict`` 와 같은 식, 상한 = 인자 END. 구현은 ``features.earnings_asof`` 로 옮겼다."""
    from swing_it.features.earnings_asof import load_yoy_strict

    _check_end(end)
    con, _ = P2._db()
    try:
        wide, info = load_yoy_strict(con, codes, dates, end, tie_col=None)
    finally:
        con.close()
    return wide, {"end": end, **info}


def v2_returns(codes, dates, end: str, lo: str) -> tuple[np.ndarray, dict]:
    """``daily_bars_adj_v2`` 종가로 로더 격자 위 일수익(보고만). 어느 날이든 없으면 NaN."""
    _check_end(end)
    con, cur = P2._db()
    cur.execute("SELECT code, date, close FROM daily_bars_adj_v2 WHERE date BETWEEN %s AND %s "
                "AND close IS NOT NULL AND close > 0", (lo, end))
    df = pd.DataFrame(cur.fetchall(), columns=["code", "date", "close"])
    con.close()
    assert not df.empty and pd.to_datetime(df["date"]).max() <= _ts(end)
    cl = np.full((len(codes), len(dates)), np.nan)
    i = pd.Index([str(c) for c in codes]).get_indexer(df["code"].astype(str))
    t = pd.DatetimeIndex(dates).get_indexer(pd.to_datetime(df["date"]))
    ok = (i >= 0) & (t >= 0)
    cl[i[ok], t[ok]] = df["close"].to_numpy(float)[ok]
    r = np.full_like(cl, np.nan)
    r[:, 1:] = cl[:, 1:] / cl[:, :-1] - 1.0
    return r, {"rows_read": int(len(df)), "rows_on_grid": int(ok.sum())}


def fingerprint(table: str, hi: str) -> list:
    con, _ = P2._db()
    try:
        return _fingerprint(con, table, FP_START, _ts(hi).date())
    finally:
        con.close()


# ── 준비(as-of 인자) ─────────────────────────────────────────────────────────


def prep(x, asof_from: str, asof_to: str, n_asof: int, end: str):
    """PO-2 러너 ``prep`` 와 같은 식. asof = [asof_from, asof_to] 의 월말 중 앞 21거래일이 END 안인 것."""
    sector = x.sector
    has_sec = np.vectorize(lambda s: s is not None, otypes=[bool])(sector)
    mask = has_sec & np.isfinite(x.cap) & (x.cap > 0) & np.isfinite(x.close)
    raw = style_panels(x)
    Xall = exposures(x, raw, mask)
    first = int(np.argmax(has_sec.any(axis=0)))
    sl = slice(first, None)
    c = {"R": x.ret[:, sl], "cap": x.cap[:, sl], "msk": mask[:, sl], "lab": sector[:, sl],
         "Xs": {k: v[:, sl] for k, v in Xall.items()}, "dates": x.dates[sl], "first": first, "raw": raw}
    nD = c["R"].shape[1]
    c["asofs"] = [a for a in month_ends(c["dates"], asof_from, nD - 1) if c["dates"][a] <= _ts(asof_to)]
    assert len(c["asofs"]) == n_asof, f"asof {len(c['asofs'])}개 ≠ {n_asof}"
    assert all(a + HORIZON < nD and c["dates"][a + HORIZON] <= _ts(end) for a in c["asofs"])
    c["sectors"] = sorted({s for s in c["lab"][:, -1] if s is not None})
    return c


def guarded(x0, end: str):
    raw_src, events = raw_closes_and_events(x0.codes, x0.dates, end)
    rf, cf, seam, info = seam_guard(x0.codes, x0.dates, x0.ret, x0.close, raw_src, events,
                                    secondary_min_abs_ret=SEC_MIN_ABS_RET)
    return dataclasses.replace(x0, ret=rf, close=cf), seam, info


# ── OB(α 난수 키 인자 — 식은 PO-2 러너 ``_ob_task`` 그대로) ─────────────────────


def _ob_task(args):
    from krx_quant_core.portfolio.optimize import Constraints, optimize

    n, X, F, D, names, sect, sigma_t, band, key = args
    nb = len(D)
    sub = Sub(X=X, F=21.0 * F, D=21.0 * D, names=names)
    b = np.full(nb, 1.0 / nb)
    Sb = sub.X @ (sub.F @ (sub.X.T @ b)) + sub.D * b
    con = Constraints(budget=1.0, long_only=True, max_weight=OB_MAX_WEIGHT,
                      sector=sect if band is not None else None, benchmark=b, sector_band=band)
    rng = keyed_rng(key, n)
    W = np.full((N_OB, nb), np.nan)
    stat = {"ok": 0, "lam_fail": 0, "solver_fail": 0, "solves": 0}

    def act_risk(w):
        d = w - b
        y = sub.X.T @ d
        return float(np.sqrt(y @ sub.F @ y + (d * d) @ sub.D))

    for p in range(N_OB):
        alpha = np.sqrt(21.0 * D) * rng.standard_normal(nb)
        lo, hi = np.log(LAM_LO), np.log(LAM_HI)
        got, r = None, None
        for _ in range(LAM_ITERS):
            mid = 0.5 * (lo + hi)
            lam = float(np.exp(mid))
            stat["solves"] += 1
            r = optimize(alpha + lam * Sb, sub, None, constraints=con, risk_aversion=lam)
            if not r.ok:
                stat["solver_fail"] += 1
                break
            risk = act_risk(r.weights)
            if abs(risk / sigma_t - 1.0) <= LAM_TOL:
                got = r.weights
                break
            if risk > sigma_t:
                lo = mid
            else:
                hi = mid
        if got is None:
            stat["lam_fail"] += 1 if (r is not None and r.ok) else 0
            continue
        W[p] = got - b
        stat["ok"] += 1
    return n, W, stat


def solve_ob(models, asofs, bu, lab, sigma_target, workers, path: Path, key: str):
    if path.exists():
        return pickle.load(open(path, "rb"))
    tasks = []
    for n, a in enumerate(asofs):
        m, idx = P3.as_model(models, a)
        s = bu[n]
        tasks.append((n, m.X[s], m.F, m.D[s], m.names, lab[idx[s], a].astype(object), sigma_target,
                      OB_SECTOR_BAND, key))
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for n, W, st in pool.imap_unordered(_ob_task, tasks):
            res[n] = (W, st)
    pickle.dump(res, open(path, "wb"))
    return res


# ── 평가(PO-3 러너 ``evaluate`` 와 같은 식 + 보고만 항목) ───────────────────────


def evaluate(c, models, bu, ob, seam_s, U, sfx: str, models3=None, R_v2=None, mret=None):
    """구창이면 sfx="" 로 PO-3 와 같은 난수 키를 쓴다. OB21·OB5 는 난수 키와 무관하다."""
    R, cap, lab, asofs, sectors, first = c["R"], c["cap"], c["lab"], c["asofs"], c["sectors"], c["first"]
    rng_t = {k: keyed_rng(k + sfx) for k in ("P5", "P50", "P10B", "LO5", "LO50")}
    names_t = ([f"S:{k}" for k in STYLE_NAMES] + ["P5", "P50", "P10B", "HV", "SM"]
               + [f"IP:{g}" for g in sectors] + [f"IE:{g}" for g in sectors])
    mkt_t = ["capw", "LO5", "LO50", "KOSPI"]
    Zr = {k: [] for k in names_t + mkt_t}
    A = len(asofs)
    Z = {k: np.full((N_OB, A), np.nan) for k in ("v4", "v3", "v2")}
    Z5 = {k: np.full((N_OB, A, 4), np.nan) for k in ("v4", "v3", "v2")}
    struct, spec_z, spec_codes = [], [], []
    expo = {"max_abs_sector_active": [], "sector_var_share": [], "factor_var_share": []}
    nan_all = {"seam": 0, "other": 0}
    nan_B = {"seam": 0, "other": 0, "v2": 0}
    for n, a in enumerate(asofs):
        m, idx = P3.as_model(models, a)
        nA = len(idx)
        fr = R[idx, a + 1:a + 1 + HORIZON]
        bad = ~np.isfinite(fr)
        sw = seam_s[idx, a + 1:a + 1 + HORIZON]
        nan_all["seam"] += int((bad & sw).sum())
        nan_all["other"] += int((bad & ~sw).sum())
        s = bu[n]
        nan_B["seam"] += int((bad[s] & sw[s]).sum())
        nan_B["other"] += int((bad[s] & ~sw[s]).sum())
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
        # 시장 수준(보고만): 시총가중, 롱온리 무작위 5·50, KOSPI 단독(시총가중 사전 위험으로 나눔 — 근사)
        wc = cap[idx, a] / cap[idx, a].sum()
        Zr["capw"].append(zs(wc))
        for key, size in (("LO5", 5), ("LO50", 50)):
            Wm = np.zeros((N_P, nA))
            for p in range(N_P):
                Wm[p, rng_t[key].choice(nA, size, replace=False)] = 1.0 / size
            Zr[key].append(zs(Wm))
        if mret is not None:
            km = np.nan_to_num(mret[a + 1:a + 1 + HORIZON], nan=0.0).sum()
            Zr["KOSPI"].append(np.array([km / np.sqrt(HORIZON * float(m.var(wc)))]))
        # OB
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
            Z["v4"][good, n] = (Wg @ fwd[s])[good] / (np.sqrt(21.0) * p1[good])
            Z5["v4"][good, n, :] = (Wg @ wk[s])[good] / (np.sqrt(5.0) * p1[good])[:, None]
            Ys = Y[:, sec_idx]
            Fs = m.F[np.ix_(sec_idx, sec_idx)]
            expo["max_abs_sector_active"].append(np.abs(Ys[good]).max(axis=1))
            expo["sector_var_share"].append((np.einsum("pk,kl,pl->p", Ys, Fs, Ys) / p1 ** 2)[good])
            expo["factor_var_share"].append((fv / p1 ** 2)[good])
        if models3 is not None:  # 같은 포트를 v3 예측으로(보고만)
            m3, idx3 = P3.as_model(models3, a)
            assert np.array_equal(idx3, idx), "v3·v4 모델 종목이 다르다"
            Y3 = Wg @ m3.X[s]
            p3 = np.sqrt(np.einsum("pk,kl,pl->p", Y3, m3.F, Y3) + (Wg * Wg) @ m3.D[s])
            with np.errstate(invalid="ignore", divide="ignore"):
                Z["v3"][good, n] = (Wg @ fwd[s])[good] / (np.sqrt(21.0) * p3[good])
                Z5["v3"][good, n, :] = (Wg @ wk[s])[good] / (np.sqrt(5.0) * p3[good])[:, None]
        if R_v2 is not None:  # 같은 포트·같은 v4 예측, 실현만 v2(보고만)
            f2 = R_v2[idx[s], a + 1:a + 1 + HORIZON]
            nan_B["v2"] += int((~np.isfinite(f2)).sum())
            f2 = np.nan_to_num(f2, nan=0.0)
            fw2 = f2.sum(axis=1)
            wk2 = np.stack([f2[:, 5 * k:5 * k + 5].sum(axis=1) for k in range(4)], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z["v2"][good, n] = (Wg @ fw2)[good] / (np.sqrt(21.0) * p1[good])
                Z5["v2"][good, n, :] = (Wg @ wk2)[good] / (np.sqrt(5.0) * p1[good])[:, None]
        struct.append({"Y": Y, "Wd": Wg * np.sqrt(Ds_)[None, :], "F": m.F, "good": good,
                       "pred1": np.where(good, p1, 1.0),
                       "sec_pos": {g: m.names.index(g) for g in sectors if g in m.names}})
        u = U[idx, a + 1:a + 1 + HORIZON]
        okr = np.isfinite(u).all(axis=1)
        spec_z.append(u[okr].sum(axis=1) / np.sqrt(HORIZON * m.D[okr]))
        spec_codes.append(idx[okr])
    realized = {t: bias_rows(np.array([np.atleast_1d(v) for v in Zr[t]], dtype=float).T)
                for t in names_t + mkt_t if Zr[t]}
    realized["OB21"] = bias_rows(Z["v4"])
    realized["OB5"] = bias_rows(Z5["v4"].reshape(N_OB, -1))
    return {"realized": realized, "Zr": Zr, "Z21": Z["v4"], "Z5": Z5["v4"], "Zall": Z, "Z5all": Z5,
            "struct": struct, "spec_z": spec_z, "spec_codes": spec_codes, "expo": expo,
            "nan_to_zero": nan_all, "nan_to_zero_B": nan_B}


def ob_pair(Z21, Z5, cols=None):
    cols = np.arange(Z21.shape[1]) if cols is None else np.asarray(cols)
    return {"OB21": bias_rows(Z21[:, cols]), "OB5": bias_rows(Z5[:, cols, :].reshape(N_OB, -1))}


def spec_z_median(spec_z, spec_codes, min_obs=12):
    """종목별 특이 z 의 시계열 표준편차(바이어스)의 종목 중앙값 — 설계 §6 '종목별 특이 z 중앙'."""
    z = np.concatenate(spec_z)
    cc = np.concatenate(spec_codes)
    df = pd.DataFrame({"c": cc, "z": z})
    g = df.groupby("c")["z"].agg(["std", "count"])
    g = g[g["count"] >= min_obs]
    return float(g["std"].median()), int(len(g))


def q3(v):
    v = np.concatenate([np.atleast_1d(x_) for x_ in v])
    return [float(x_) for x_ in np.quantile(v, [0.1, 0.5, 0.9])]


def rnd(o):
    if isinstance(o, dict):
        return {str(k): rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def r5(v) -> float:
    return round(float(v), 5)


def models_bit_same(a_models, b_models, keys) -> bool:
    return all(k in a_models and k in b_models
               and all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(a_models[k][:4], b_models[k][:4]))
               and a_models[k][4] == b_models[k][4] for k in keys)


def r0_matches_po3(r0, po3) -> tuple[bool, dict]:
    d = {k: r5(v) - po3["R0_noguard"]["values"][k] for k, v in r0["values"].items()}
    return all(abs(v) < 5e-6 for v in d.values()), d


def tails(spec_z):
    cv, s_med = P2.cv_estimates(spec_z)
    nu_u, k_u, n_u = nu_from(np.concatenate([z / np.sqrt(sv) for z, sv in zip(spec_z, s_med)]))
    return {**cv, "nu_u": nu_u, "nu_u_unclipped": 4.0 + 6.0 / max(k_u, 0.2), "kurt_u": k_u, "n_u": n_u}


def nu_f_of(Zr, sectors):
    pooled = []
    for t in [f"S:{k}" for k in STYLE_NAMES] + [f"IP:{g}" for g in sectors]:
        zz = np.concatenate([np.atleast_1d(v) for v in Zr[t]])
        zz = zz[np.isfinite(zz)]
        pooled.append(zz / zz.std(ddof=1))
    return nu_from(np.concatenate(pooled))


# ── repro ────────────────────────────────────────────────────────────────────


def run_repro(args, cache: Path, log) -> int:
    assert args.end == OLD_END and (args.asof_from, args.asof_to, args.n_asof) == OLD_ASOF
    po3 = json.loads((OUT / "PO3.json").read_text())
    po3c = Path(args.po3_cache)
    fp0 = fingerprint("daily_bars_adjusted", args.end)
    x0 = load_inputs(args.end)
    x, seam, seam_info = guarded(x0, args.end)
    log(f"이음매 {seam_info['seam_cells']}셀 (PO-3 {po3['seam_new']['seam_cells']})")
    # 가드 끈 R0·모델 비트
    c_raw = prep(x0, args.asof_from, args.asof_to, args.n_asof, args.end)
    m_raw = P3.build_models(c_raw, c_raw["asofs"], args.workers, cache / "models_noguard.pkl", log)
    r0 = P2.r0_values(c_raw, m_raw)
    r0_ok, r0_d = r0_matches_po3(r0, po3)
    old = pickle.load(open(po3c / "models_noguard.pkl", "rb"))
    noguard_bit = sorted(old) == sorted(m_raw) and models_bit_same(old, m_raw, list(old))
    del c_raw, m_raw, old
    log(f"R0 PO3 일치 {r0_ok}, 가드 끈 모델 비트 {noguard_bit}")
    # v4
    c = prep(x, args.asof_from, args.asof_to, args.n_asof, args.end)
    R, cap, lab, dates, asofs, sectors, first = (c["R"], c["cap"], c["lab"], c["dates"], c["asofs"],
                                                 c["sectors"], c["first"])
    nN, nD = R.shape
    sig_all, sig_info = yoy_signal_strict(x.codes, x.dates, args.end)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    sig, adv = sig_all[:, first:], adv_all[:, first:]
    t_0 = int(np.searchsorted(dates, np.datetime64(OLD_GRID[0])))
    grid = [t for t in range(t_0, nD, STEP) if dates[t] <= _ts(OLD_GRID[1])]
    need = sorted(set(asofs) | set(grid))
    models = P3.build_models(c, need, args.workers, cache / "models_v4.pkl", log, vra=V4["vra"], shrink=V4["shrink"])
    old = pickle.load(open(po3c / "models_vraoff_s0.1.pkl", "rb"))
    v4_bit = sorted(old) == sorted(models) and models_bit_same(old, models, list(old))
    del old
    st, st_rows = P3.sigma_target_of(models, grid, sig, adv, cap, nN)
    bu = P3.b_universe(models, asofs, sig, adv, cap, nN)
    ob = solve_ob(models, asofs, bu, lab, st, args.workers, cache / f"ob_v4_{st:.6f}.pkl", "OB")
    ob_stat = {k: int(sum(ob[n][1][k] for n in ob)) for k in ("ok", "lam_fail", "solver_fail", "solves")}
    U = factor_returns(R, c["Xs"], lab, cap, c["msk"]).u
    ev = evaluate(c, models, bu, ob, seam[:, first:], U, "")
    b21, b5 = ev["realized"]["OB21"], ev["realized"]["OB5"]
    repro = {"OB21": b21, "OB5": b5, "ob_ok": ob_stat["ok"],
             "OB21_match": r5(b21) == REPRO_TARGET["OB21"], "OB5_match": r5(b5) == REPRO_TARGET["OB5"],
             "ob_ok_match": ob_stat["ok"] == REPRO_TARGET["ob_ok"],
             "sigma_target_match": r5(st) == next(g["sigma_target"] for g in po3["grid"] if g["name"] == "vraoff_s0.1"),
             "R0_match": r0_ok, "R0_diff": r0_d, "noguard_models_bit_same": noguard_bit, "v4_models_bit_same": v4_bit,
             "nan_to_zero_all_match": ev["nan_to_zero"] == po3["v4"]["nan_to_zero"]}
    log(f"재현 OB21 {b21:.6f} OB5 {b5:.6f} OB 성공 {ob_stat['ok']} — {repro}")
    res = {"mode": "repro", "prereg": {"PO-3": "4bdf40e", "PO-3_result": "1c12207"}, "core_ref": "13c56a7",
           "end": args.end, "window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())], "n_asof": len(asofs),
           "data_end": str(dates[-1].date()), "fingerprint_before": fp0, "seam": seam_info, "signal": sig_info,
           "R0_noguard": r0, "sigma_target": st, "sigma_target_repr": repr(st), "ob": ob_stat, "repro": repro}
    core_ok = all(repro[k] for k in ("OB21_match", "OB5_match", "ob_ok_match", "sigma_target_match", "R0_match",
                                     "noguard_models_bit_same", "v4_models_bit_same"))
    if not core_ok:
        res["STOP"] = "구창 재현 불일치 — 신창을 읽지 않는다"
        return write(res, "PO4_repro", log, 1)
    # 모의(PO-3 와 같은 꼬리·같은 구조·공통 난수) + D×1.2·D×0.83
    t4 = tails(ev["spec_z"])
    nu_f, k_f, n_f = nu_f_of(ev["Zr"], sectors)
    sig_ln = float(np.sqrt(np.log(1.0 + t4["CV_med"] ** 2)))
    P2._SIM.clear()
    P2._SIM.update({"struct": {"v4": ev["struct"]}, "nu_f": nu_f, "nu_u": t4["nu_u"], "sig_ln": {"v4": sig_ln}})
    sims = {}
    with mp.get_context("fork").Pool(min(args.workers, len(REPRO_SCEN))) as pool:
        for _, sc, s21, s5, stopw in pool.imap_unordered(P2._sim_task, [("v4", sc) for sc in REPRO_SCEN]):
            sims[sc] = (s21, s5, stopw)
            log(f"  모의 {sc} b21 중앙 {np.nanmedian(s21):.3f} b5 {np.nanmedian(s5):.3f}")
    P2._SIM.clear()
    se_boot = P2.boot_se_ob(ev["Z21"], ev["Z5"], P2.T_GATE)
    cal = P3.calibrate(sims, se_boot)
    pc = po3["v4"]["calibration"]
    band_po3 = pc["band"]
    sim_match = {
        "band": all(r5(cal["band"][k][j]) == band_po3[k][j] for k in ("OB21", "OB5") for j in (0, 1)),
        "detect": all(r5(cal["detect"][w]["family"]) == pc["detect"][w]["family"] for w in ("correct", "Dx1.5", "Dx0.67")),
        "alpha": r5(cal["alpha"]) == pc["alpha"],
    }
    repro["sim_match"] = sim_match
    # 고정 밴드(PO3.json) 기준 탐지 — 판정에 쓰는 바로 그 숫자
    def outside(s21, s5):
        o21 = ~((s21 >= band_po3["OB21"][0]) & (s21 <= band_po3["OB21"][1]))
        o5 = ~((s5 >= band_po3["OB5"][0]) & (s5 <= band_po3["OB5"][1]))
        return {"family": float(np.mean(o21 | o5)), "primary": float(np.mean(o21)), "secondary": float(np.mean(o5)),
                "b21_median": float(np.median(s21)), "b5_median": float(np.median(s5))}
    power = {sc: outside(sims[sc][0], sims[sc][1]) for sc in REPRO_SCEN}
    c21, c5 = cal["center"]["OB21"], cal["center"]["OB5"]
    yw21 = sims["correct"][0] * (REPRO_TARGET["OB21"] / c21)
    yw5 = sims["correct"][1] * (REPRO_TARGET["OB5"] / c5)
    yworld = outside(yw21, yw5)
    pair = next(((hi, lo) for hi, lo in X_PAIRS
                 if power[f"Dx{hi}"]["family"] >= DETECT_MIN and power[f"Dx{lo}"]["family"] >= DETECT_MIN), None)
    X = None if pair is None else int(round(100 * max(math.sqrt(pair[0]) - 1, 1 - math.sqrt(pair[1]))))
    Y = int(round(100 * yworld["family"]))
    sd_med, sd_n = spec_z_median(ev["spec_z"], ev["spec_codes"])
    fp1 = fingerprint("daily_bars_adjusted", args.end)
    res.update({
        "fingerprint_after": fp1, "fingerprint_same": fp0 == fp1,
        "realized_v4": ev["realized"], "nan_to_zero_all": ev["nan_to_zero"], "nan_to_zero_B": ev["nan_to_zero_B"],
        "spec_bias_median": sd_med, "spec_bias_n": sd_n,
        "exposure": {k: q3(v) for k, v in ev["expo"].items()},
        "tails": {**t4, "nu_f": nu_f, "kurt_f": k_f, "n_f": n_f}, "se_boot_T31": se_boot,
        "calibration_recomputed": {k: cal[k] for k in ("alpha", "center", "se_sim", "kappa", "band", "h")},
        "band_fixed_PO3": band_po3, "power_fixed_band": power, "Y_world": {"scale": [REPRO_TARGET["OB21"] / c21,
                                                                                    REPRO_TARGET["OB5"] / c5], **yworld},
        "X": X, "X_pair": pair, "Y": Y,
        "sentence_pass": None if X is None else f"변동성 오차 약 {X}% 이상은 기각됐다. v4 가 맞게 보정됐다는 뜻은 아니다.",
        "sentence_fail": f"구창 표본 안 성적 그대로여도 약 {Y}% 확률로 실패한다.",
    })
    repro["all_ok"] = bool(core_ok and all(sim_match.values()) and fp0 == fp1)
    if not repro["all_ok"]:
        res["STOP"] = "모의 재현 또는 지문 불일치 — 신창을 읽지 않는다"
        return write(res, "PO4_repro", log, 1)
    return write(res, "PO4_repro", log, 0)


# ── new ──────────────────────────────────────────────────────────────────────


def recheck_old(args, cache: Path, po3, log) -> dict:
    """PO-4.2 실행 시작 확인 — 구창 가드 끈 R0(PO3.json)·구창 v4 모델 4개(PO-3 캐시 비트)."""
    x0 = load_inputs(OLD_END)
    c_raw = prep(x0, *OLD_ASOF, OLD_END)
    m_raw = P3.build_models(c_raw, c_raw["asofs"], args.workers, cache / "old_models_noguard.pkl", log)
    r0 = P2.r0_values(c_raw, m_raw)
    r0_ok, r0_d = r0_matches_po3(r0, po3)
    del c_raw, m_raw
    x, _, _ = guarded(x0, OLD_END)
    c = prep(x, *OLD_ASOF, OLD_END)
    keys = [c["asofs"][i] for i in RECHECK_N]
    m4 = P3.build_models(c, keys, args.workers, cache / "old_models_v4_4.pkl", log, vra=V4["vra"], shrink=V4["shrink"])
    old = pickle.load(open(Path(args.po3_cache) / "models_vraoff_s0.1.pkl", "rb"))
    bit = models_bit_same(old, m4, keys)
    return {"R0": r0["values"], "R0_diff_vs_PO3": r0_d, "R0_match": r0_ok, "v4_models_checked": RECHECK_N,
            "v4_models_bit_same": bit, "ok": bool(r0_ok and bit)}


def v2_tables(x0, x, R_v2, adv_all, c, bu, models, seam, first, lo: str, end: str):
    """PO-4.2 v2 차이 표·셀 집계(보고만). 패널은 로더 전체 격자(first 앞 포함), 여기서 신창 날짜로 자른다."""
    dates = x0.dates
    win = (dates >= _ts(lo)) & (dates <= _ts(end))
    ra, rg, rv = x0.ret, x.ret, R_v2
    liq = adv_all >= ADV_LIQ
    Bw = np.zeros_like(ra, dtype=bool)
    for n, a in enumerate(c["asofs"]):
        _, idx = P3.as_model(models, a)
        rows = idx[bu[n]]
        Bw[rows, first + a + 1:first + a + 1 + HORIZON] = True
    Bw &= win[None, :]

    def counts(ret, sel):
        ok = sel & np.isfinite(ret) & np.isfinite(rv)
        d = np.abs(ret - rv)
        return {"cells": int(ok.sum()), ">1%": int((ok & (d > 0.01)).sum()), ">5%": int((ok & (d > 0.05)).sum()),
                ">20%": int((ok & (d > 0.20)).sum())}

    rows = {}
    for nm, sel in (("all", win[None, :] & np.ones_like(ra, dtype=bool)), ("liquid_200eok", win[None, :] & liq),
                    ("B_fwd21", Bw)):
        rows[nm] = {"pre_guard": counts(ra, sel), "post_guard": counts(rg, sel)}
    okc = np.isfinite(ra) & np.isfinite(rv)
    clus = okc & (np.abs(ra - rv) > CLUSTER_DEV) & win[None, :]
    ld = okc & (rv <= -LIMIT_HIT) & (ra > -LIMIT_KEEP) & win[None, :]
    lu = okc & (rv >= LIMIT_HIT) & (ra < LIMIT_KEEP) & win[None, :]

    def cells(mask, cap=40):
        ii, tt = np.nonzero(mask)
        out = [{"code": str(x0.codes[i]), "date": str(dates[t].date()), "ret_adj": float(ra[i, t]),
                "ret_v2": float(rv[i, t]), "guard_nan": bool(seam[i, t])} for i, t in zip(ii, tt)]
        out.sort(key=lambda r: -abs(r["ret_adj"] - r["ret_v2"]))
        return out[:cap]

    by_date = pd.Series(dates[np.nonzero(clus)[1]].strftime("%Y-%m-%d")).value_counts()
    by_date_B = pd.Series(dates[np.nonzero(clus & Bw)[1]].strftime("%Y-%m-%d")).value_counts()
    d0108 = int(np.flatnonzero(dates == _ts("2024-01-08"))[0]) if (dates == _ts("2024-01-08")).any() else None
    return {
        "diff_table": rows,
        "cluster": {"rule": "|ret_adj − ret_v2| > 0.10", "all": int(clus.sum()), "B_fwd21": int((clus & Bw).sum()),
                    "B_fwd21_guard_nan": int((clus & Bw & seam).sum()),
                    "on_2024-01-08_all": None if d0108 is None else int(clus[:, d0108].sum()),
                    "on_2024-01-08_B_fwd21": None if d0108 is None else int((clus & Bw)[:, d0108].sum()),
                    "top_dates_all": [[k, int(v)] for k, v in by_date.head(10).items()],
                    "top_dates_B_fwd21": [[k, int(v)] for k, v in by_date_B.head(10).items()],
                    "cells_B_fwd21": cells(clus & Bw)},
        "erased_limit_down": {"rule": "ret_v2 ≤ −0.29 ∧ ret_adj > −0.05", "all": int(ld.sum()), "B_fwd21": int((ld & Bw).sum()),
                              "cells_all": cells(ld), "001380_2025-03-26_in_B_fwd21": bool(
                                  any(r["code"] == "001380" and r["date"] == "2025-03-26" for r in cells(ld & Bw, 10 ** 6)))},
        "erased_limit_up": {"rule": "ret_v2 ≥ 0.29 ∧ ret_adj < 0.05", "all": int(lu.sum()), "B_fwd21": int((lu & Bw).sum()),
                            "cells_all": cells(lu)},
    }


def run_new(args, cache: Path, log) -> int:
    """신창 A1. ``--mode smoke`` 는 같은 경로를 구창 끝 5개월(≤ 2023-12-28)로만 돌려 코드 오류를 신창 전에 잡는다(결과는 버린다)."""
    smoke = args.mode == "smoke"
    assert (args.end == OLD_END) if smoke else (args.end != OLD_END)
    po3 = json.loads((OUT / "PO3.json").read_text())
    band = po3["v4"]["calibration"]["band"]
    if smoke:
        st = next(g["sigma_target"] for g in po3["grid"] if g["name"] == "vraoff_s0.1")
        rep = {"sentence_pass": "(smoke)", "sentence_fail": "(smoke)", "power_fixed_band": {}, "X": None, "Y": None}
    else:
        rep = json.loads((OUT / "PO4_repro.json").read_text())
        assert rep.get("repro", {}).get("all_ok") and "STOP" not in rep, "구창 재현이 통과하지 않았다"
        st = float(rep["sigma_target_repr"])
        assert rep["band_fixed_PO3"] == band
    # 1. 지문(시작) + 구창 재확인 — 신창을 읽기 전
    fp0 = {t: fingerprint(t, args.end) for t in ("daily_bars_adjusted", "daily_bars_adj_v2")}
    log(f"지문(시작) {fp0['daily_bars_adjusted']}")
    rc = recheck_old(args, cache, po3, log)
    log(f"구창 재확인 R0 {rc['R0_match']} 모델 비트 {rc['v4_models_bit_same']}")
    if not rc["ok"]:
        return write({"mode": "new", "STOP": "구창 재확인 실패 — 신창 미사용", "recheck_old": rc,
                      "fingerprint_before": fp0}, args.stem, log, 1)
    # 2. 신창 입력
    x0 = load_inputs(args.end)
    x, seam, seam_info = guarded(x0, args.end)
    c = prep(x, args.asof_from, args.asof_to, args.n_asof, args.end)
    R, cap, lab, dates, asofs, sectors, first = (c["R"], c["cap"], c["lab"], c["dates"], c["asofs"],
                                                 c["sectors"], c["first"])
    if not smoke:
        assert str(dates[asofs[0]].date()) == "2024-01-31" and str(dates[asofs[-1]].date()) == "2026-07-31"
    nN, _ = R.shape
    sig_all, sig_info = yoy_signal_strict(x.codes, x.dates, args.end)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    sig, adv = sig_all[:, first:], adv_all[:, first:]
    models = P3.build_models(c, asofs, args.workers, cache / "models_v4.pkl", log, vra=V4["vra"], shrink=V4["shrink"])
    models3 = P3.build_models(c, asofs, args.workers, cache / "models_v3.pkl", log, vra=V3["vra"], shrink=V3["shrink"])
    bu = P3.b_universe(models, asofs, sig, adv, cap, nN)
    ob = solve_ob(models, asofs, bu, lab, st, args.workers, cache / f"ob_new_{st:.6f}.pkl",
                  "OB-smoke" if smoke else "OB-new")
    ob_stat = {k: int(sum(ob[n][1][k] for n in ob)) for k in ("ok", "lam_fail", "solver_fail", "solves")}
    U = factor_returns(R, c["Xs"], lab, cap, c["msk"]).u
    v2_lo, diff_lo = ("2023-05-01", "2023-06-01") if smoke else ("2023-12-01", "2024-01-02")  # 첫 asof 앞부터
    R_v2_full, v2_info = v2_returns(x.codes, x.dates, args.end, v2_lo)
    ev = evaluate(c, models, bu, ob, seam[:, first:], U, "-new", models3=models3, R_v2=R_v2_full[:, first:],
                  mret=x.market_ret[first:])
    v2t = v2_tables(x0, x, R_v2_full, adv_all, c, bu, models, seam, first, diff_lo, args.end)
    # 3. 지문(끝) — 다르면 결과를 쓰지 않는다
    fp1 = {t: fingerprint(t, args.end) for t in ("daily_bars_adjusted", "daily_bars_adj_v2")}
    if fp1["daily_bars_adjusted"] != fp0["daily_bars_adjusted"]:
        return write({"mode": "new", "STOP": "실행 중 daily_bars_adjusted 지문이 바뀌었다 — 결과 미기록(아무도 보지 않음)",
                      "fingerprint_before": fp0, "fingerprint_after": fp1}, args.stem, log, 2)
    # 4. 판정(PO-3.6·PO-4.3 기계적 적용)
    b21, b5 = ev["realized"]["OB21"], ev["realized"]["OB5"]
    ok_share = ob_stat["ok"] / (N_OB * len(asofs))
    in21 = band["OB21"][0] <= b21 <= band["OB21"][1]
    in5 = band["OB5"][0] <= b5 <= band["OB5"][1]
    if ok_share < OB_OK_MIN:
        verdict, sentence = "VOID", "VOID(OB 성공 < 90%) — v4 는 전진 A 로 판정한다(PO-3.6)."
    elif in21 and in5:
        verdict, sentence = "통과", rep["sentence_pass"]
    else:
        verdict, sentence = "실패", rep["sentence_fail"]
    fshare = q3(ev["expo"]["factor_var_share"])
    alarm = fshare[1] > 0.20
    years = {}
    yr = np.array([dates[a].year for a in asofs])
    mo = np.array([dates[a].month for a in asofs])
    for nm, cols in (("2024", yr == 2024), ("2025", yr == 2025), ("2026-01~07", yr == 2026),
                     ("2026-07 제외", ~((yr == 2026) & (mo == 7)))):
        years[nm] = {k: ob_pair(ev["Zall"][k], ev["Z5all"][k], np.flatnonzero(cols)) for k in ("v4", "v3", "v2")}
    t_new = tails(ev["spec_z"])
    nu_f, k_f, n_f = nu_f_of(ev["Zr"], sectors)
    sd_med, sd_n = spec_z_median(ev["spec_z"], ev["spec_codes"])
    r = ev["realized"]
    res = {
        "mode": "new", "prereg": {"PO-3": "4bdf40e", "PO-3_result": "1c12207", "PO-4": args.prereg_sha,
                                  "PO-4_repro": args.repro_sha},
        "core_ref": "13c56a7", "end": args.end, "window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())],
        "n_asof": len(asofs), "data_end": str(dates[-1].date()), "sectors": sectors,
        "label": "D·최적화 바이어스 검정 — F·업종은 O 에 무관(사전 몫 약 6%), 검정 안 함",
        "model": "v4 = vraoff_s0.1 (VRA 끔, spec_shrink 0.1) — 격자 경계(spec_shrink 최솟값 0.1)에서 골랐다, 고원이 아니다",
        "sigma_target_fixed": st, "band": band,
        "fingerprint_before": fp0, "fingerprint_after": fp1,
        "fingerprint_same": {t: fp0[t] == fp1[t] for t in fp0}, "recheck_old": rc,
        "seam": seam_info, "signal": sig_info, "ob": ob_stat, "ob_ok_share": ok_share,
        "B_universe_size": [int(min(len(v) for v in bu.values())), int(max(len(v) for v in bu.values()))],
        "OB21": b21, "OB5": b5, "OB21_in_band": bool(in21), "OB5_in_band": bool(in5),
        "verdict": verdict, "sentence": sentence,
        "factor_share_q10_50_90": fshare, "factor_share_alarm": bool(alarm),
        "factor_share_sentence": "A 는 이 포트들의 팩터 위험을 검증하지 않았다." if alarm else None,
        "exposure": {k: q3(v) for k, v in ev["expo"].items()},
        "power_report": {k: rep["power_fixed_band"][k] for k in rep["power_fixed_band"]}, "X": rep["X"], "Y": rep["Y"],
        "report_only": {
            "realized": r, "v3_same_ob": ob_pair(ev["Zall"]["v3"], ev["Z5all"]["v3"]),
            "v2_same_ob": ob_pair(ev["Zall"]["v2"], ev["Z5all"]["v2"]), "v2_read": v2_info,
            "years": years, "nan_to_zero_B": ev["nan_to_zero_B"], "nan_to_zero_all": ev["nan_to_zero"],
            "spec_bias_median": sd_med, "spec_bias_n": sd_n,
            "sector_direction": {"IP:G55>1": bool(r.get("IP:G55", np.nan) > 1), "IP:G25<1": bool(r.get("IP:G25", np.nan) < 1),
                                 "IP:G10<1": bool(r.get("IP:G10", np.nan) < 1)},
            "tails_post_hoc": {**t_new, "nu_f": nu_f, "kurt_f": k_f, "n_f": n_f},
            "v2_tables": v2t,
        },
    }
    return write(res, args.stem, log, 0)


# ── 출력 ──────────────────────────────────────────────────────────────────────


def write(res: dict, stem: str, log, code: int) -> int:
    out = _OUTDIR[0]
    out.mkdir(parents=True, exist_ok=True)
    res["elapsed_s"] = time.time() - _T0[0]
    (out / f"{stem}.json").write_text(json.dumps(rnd(res), ensure_ascii=False, indent=1))
    r = json.loads((out / f"{stem}.json").read_text())
    (out / f"{stem}.md").write_text(render_repro(r) if stem == "PO4_repro" else render_a1(r))
    log(f"저장 {out / stem}.json (종료 {code})")
    return code


def _f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def render_repro(r: dict) -> str:
    L = ["# 정정 PO-4 — 신창 러너의 구창 재현 관문 + D×1.2·D×0.83 검정력 (검증 전)", "",
         f"러너 `research/experiments/pead_optimizer_po4.py --mode repro`, core `{r.get('core_ref')}`, END {r.get('end')}, "
         f"asof {r.get('window')} ({r.get('n_asof')}회), 데이터 끝 {r.get('data_end')}. **2023-12-28 뒤 데이터는 읽지 않았다.**", ""]
    if "STOP" in r:
        L += [f"**멈춤: {r['STOP']}**", "", "```", json.dumps(r.get("repro"), ensure_ascii=False, indent=1), "```"]
        return "\n".join(L) + "\n"
    rp = r["repro"]
    L += ["## 재현 (PO-4.1 — 전부 정확히 맞아야 신창을 읽는다)", "", "| 항목 | 이번 | PO-3 | 일치 |", "|---|---|---|---|",
          f"| OB21 | {rp['OB21']:.5f} | 1.00823 | {rp['OB21_match']} |",
          f"| OB5 | {rp['OB5']:.5f} | 1.04455 | {rp['OB5_match']} |",
          f"| OB 성공 | {rp['ob_ok']} | 10339 | {rp['ob_ok_match']} |",
          f"| σ_target | {r['sigma_target_repr']} | 0.04013 | {rp['sigma_target_match']} |",
          f"| 가드 끈 R0 8개(5자리) | 최대 차 {max(abs(v) for v in rp['R0_diff'].values()):.5f} | — | {rp['R0_match']} |",
          f"| 가드 끈 모델 52개 비트 | — | — | {rp['noguard_models_bit_same']} |",
          f"| v4 모델 102개 비트 | — | — | {rp['v4_models_bit_same']} |",
          f"| NaN→0 전체 종목(이음매/그 밖) | {r['nan_to_zero_all']['seam']}/{r['nan_to_zero_all']['other']} | 57/431 | {rp['nan_to_zero_all_match']} |",
          f"| 모의 밴드·α·탐지(5자리) | — | — | {rp['sim_match']} |",
          f"| 지문 시작=끝 | {r['fingerprint_before']} | — | {r['fingerprint_same']} |", "",
          f"**재현 판정: {'통과 — 신창 1회 사용 가능' if rp['all_ok'] else '실패 — 멈춤'}**", "",
          "## 고정 밴드 기준 검정력 (PO-4.3, 10,000회, T=31, 공통 난수)", "",
          f"밴드(PO3.json 고정): OB21 [{r['band_fixed_PO3']['OB21'][0]:.5f}, {r['band_fixed_PO3']['OB21'][1]:.5f}], "
          f"OB5 [{r['band_fixed_PO3']['OB5'][0]:.5f}, {r['band_fixed_PO3']['OB5'][1]:.5f}].", "",
          "| 세계 | A 실패(가족) | 1차 OB21 | 보조 OB5 | b21 중앙 | b5 중앙 |", "|---|---|---|---|---|---|"]
    for sc, d in r["power_fixed_band"].items():
        L.append(f"| {sc} | {d['family']:.3f} | {d['primary']:.3f} | {d['secondary']:.3f} | {d['b21_median']:.3f} | {d['b5_median']:.3f} |")
    y = r["Y_world"]
    L.append(f"| 구창 표본 안 그대로(×{y['scale'][0]:.4f}·×{y['scale'][1]:.4f}) | {y['family']:.3f} | {y['primary']:.3f} | "
             f"{y['secondary']:.3f} | {y['b21_median']:.3f} | {y['b5_median']:.3f} |")
    L += ["", f"X 쌍 {r['X_pair']} → X = {r['X']}, Y = {r['Y']}.", "",
          "판정 문장(고정, PO-4.3):", f"- 통과 → \"{r['sentence_pass']}\"", f"- 실패 → \"{r['sentence_fail']}\"", "",
          "## 새 정의로 다시 센 구창 값 (보고만)", "",
          f"- NaN→0 셀 B 유니버스(이음매/그 밖): {r['nan_to_zero_B']['seam']}/{r['nan_to_zero_B']['other']} "
          f"(PO-3 은 모델 종목 전체 {r['nan_to_zero_all']['seam']}/{r['nan_to_zero_all']['other']}).",
          f"- 종목별 특이 z 바이어스 중앙값(v4, ≥12회 종목 {r['spec_bias_n']}): {r['spec_bias_median']:.3f}.",
          f"- 꼬리: CV_s 1차 {r['tails']['CV_med']:.3f}, ν_u {r['tails']['nu_u']:.2f}(자르기 전 {r['tails']['nu_u_unclipped']:.2f}), "
          f"ν_f {r['tails']['nu_f']:.2f}.", "", f"_{_f(r.get('elapsed_s'))}초_"]
    return "\n".join(L) + "\n"


# PO-1.5 — A 판정문에 그대로 쓰는 문장(PREREG 503~504·837행).
PO15_SENTENCE = "시장 대상 강등은 신창 정보 열람 후의 완화이고, A 통과는 시장 위험 예측에 대해 아무 말도 하지 않는다."

# A1 독립 리뷰(quant-reviewer)가 따로 계산해 전한 값 — 이 러너가 다시 내지 않는다. 렌더러는 글로만 옮긴다.
A1_REVIEW = {
    "descriptive": "점추정은 특이위험 과소예측(변동성 약 6%), 맞는 모델 대비 단측 p ≈ 0.05~0.11, 방향은 구창 v3 와 같다.",
    "tails": ("고정 밴드(구창 꼬리로 정함)는 신창 꼬리에서 조금 좁다 — 신창 꼬리로 모의한 맞는 모델의 실패 확률 0.104(명목 0.066). "
              "통과를 어렵게 하는 쪽이라 판정에는 보수적이다. 같은 꼬리에서 D×1.5·D×0.67 탐지는 0.997·0.994 라 "
              "\"변동성 오차 약 22% 이상은 기각됐다\" 는 그대로 선다."),
    "sim": ("리뷰 모의(맞는 모델, 구창 꼬리/신창 꼬리): P(b21 ≥ 1.0496) 0.091/0.112, P(b5 ≥ 1.0609) 0.049/0.071. 실현값은 "
            "D×1.13 세계의 중앙값 근처이고, 변동성 과소예측 약 10~14% 까지와 양립한다. 판정(밴드 안)은 바뀌지 않는다."),
    "sub2026": ("2026-01~07(7회) 값은 약 1.2 SE 라 따로 판정할 수 없다. 월별 OB z² 평균은 2026-07 이 0.45 로 낮고, "
                "높은 달은 2026-01 2.16·2026-04 2.42 다."),
    "smoke": ("신창 실행 전 `--mode smoke`(END 2023-12-28, asof 2023-06-30~2023-10-31 5회, 2026-11-02 05:03~05:04)를 "
              "PO-4 커밋(`b38817a`, 05:05) **전** 코드로 한 번 돌렸다. 코드 오류 확인용이고 출력은 레포 밖에 버렸다. "
              "그 실행의 지문 구간 끝이 2023-12-28 이라 신창을 읽지 않았다."),
    "renderer": ("이 렌더러는 결과를 본 뒤 두 번 고쳤다. (1) A1 결과 커밋 `63a05d7` 에서 시장 수준 대상 정의 한 줄. "
                 "(2) 이 정정에서 PO-1.5 문장·서술 한 줄·이 절. 둘 다 `--render`(JSON → MD)만 했고 숫자는 다시 계산하지 않았다."),
}


def render_a1(r: dict, review: bool | None = None) -> str:
    """``review`` 를 주지 않으면 원래 A1(신창 창·END)일 때 독립 리뷰 절을 넣는다. A1_repaired 는 ``review=False`` 로 부른다."""
    L = []
    if "STOP" in r:
        return f"# A1 — 멈춤\n\n**{r['STOP']}**\n"
    ro = r["report_only"]
    b = r["band"]
    if review is None:
        review = r.get("window") == ["2024-01-31", "2026-07-31"] and r.get("end") == "2026-09-11"
    L += [f"# 단계 A1 — v4 신창 D·최적화 바이어스 검정 ({'독립 리뷰 반영 — 판정 유지, 숫자 재계산 없음' if review else '검증 전'})", "",
          f"**{r['label']}**", "",
          f"사전등록 PO-3 `{r['prereg']['PO-3']}`·PO-4 `{r['prereg']['PO-4']}`, 재현 `{r['prereg']['PO-4_repro']}`, core `{r['core_ref']}`. "
          f"신창 asof {r['window'][0]}~{r['window'][1]} ({r['n_asof']}회), END {r['end']}(데이터 끝 {r['data_end']}). 신창은 이 한 번만 쓴다.", "",
          f"- 모델: {r['model']}.", f"- σ_target(구창 v4 고정) = {r['sigma_target_fixed']:.6f}, α 난수 `keyed_rng(\"OB-new\", n)`.", "",
          "## 판정 (PO-3.6·PO-4.3 기계적 적용)", "",
          "| 검정 | 실현 | 밴드(고정) | 안 |", "|---|---|---|---|",
          f"| 1차 A-OB21 | {r['OB21']:.4f} | [{b['OB21'][0]:.5f}, {b['OB21'][1]:.5f}] | {r['OB21_in_band']} |",
          f"| 보조 A-OB5 | {r['OB5']:.4f} | [{b['OB5'][0]:.5f}, {b['OB5'][1]:.5f}] | {r['OB5_in_band']} |",
          f"| OB 성공 | {r['ob']['ok']}/{200 * r['n_asof']} = {r['ob_ok_share']:.3f} | ≥ 0.90 (아니면 VOID) | {r['ob_ok_share'] >= 0.9} |", "",
          f"**A 판정: {r['verdict']}.** {r['sentence']}", ""]
    rz = ro["realized"]
    L.append(f"- 시장 수준 대상(보고만): capw {_f(rz['capw'])}, KOSPI {_f(rz['KOSPI'])}, LO50 {_f(rz['LO50'])}. {PO15_SENTENCE}")
    if review:
        L.append(f"- (서술, 판정 아님) {A1_REVIEW['descriptive']}")
    if r["verdict"] == "통과":
        L.append("- 결과 가지(PO-3.6): B 를 §2·PO-1 대로 열 수 있다 — **이 실행에서는 열지 않는다.** B 첫 줄은 \"위험모델 v4 — D·최적화 바이어스만 신창 OB 로 검증, F·업종 미검증\".")
    elif r["verdict"] == "실패":
        L.append("- 결과 가지(PO-3.6): v4 실패. 고친 v5 는 전진으로만 판정한다. B 는 역사 구간에서 열지 않는다.")
    else:
        L.append("- 결과 가지(PO-3.6): v4 를 전진 A 로 판정한다. B 는 전진 A 통과 뒤에만.")
    L += ["- 경계(PO-4.4): v4 는 격자 경계(spec_shrink 0.1)에서 골랐다. s<0.1 이나 s=0 은 신창 결과와 상관없이 v5 로만 다루고 전진으로만 판정한다.",
          "- 취약 라벨(PO-4.2): PR #12 배포 뒤 C0 1회 + A1 보고만 1회. 통과·실패가 바뀌면 \"취약\" → 전진.",
          f"- OB 사전 팩터 분산 몫 10·50·90% {'·'.join(f'{v:.3f}' for v in r['factor_share_q10_50_90'])} — 경보(중앙 > 20%) "
          f"{'**켜짐**: ' + r['factor_share_sentence'] if r['factor_share_alarm'] else '꺼짐'}.", ""]
    if review:
        tp = ro["tails_post_hoc"]
        ex = ro["years"]["2026-07 제외"]["v4"]
        L += ["## 독립 리뷰 반영 (정정 — 판정 유지, 숫자 재계산 없음)", "",
              "독립 리뷰(`quant-reviewer`): A 통과 유지, CRITICAL 없음. 아래의 확률·월별 값은 리뷰어가 따로 계산한 것이고, 이 러너가 다시 낸 것이 아니다. "
              "실행 숫자는 모두 `A1.json` 그대로다.", "",
              f"- **꼬리(m1).** 신창 꼬리(CV_s {_f(tp['CV_med'])}, ν_u 자르기 전 {_f(tp['nu_u_unclipped'], 2)})는 구창 v4(0.324, 4.42)보다 두껍다. "
              f"{A1_REVIEW['tails']}",
              f"- **점추정의 뜻(M2, 서술).** {A1_REVIEW['sim']}",
              f"- **2026 부분창(m4).** {A1_REVIEW['sub2026']} 그래서 \"2026-07 제외\" 값(OB21 {_f(ex['OB21'], 4)}·OB5 {_f(ex['OB5'], 4)})이 "
              f"전체(OB21 {r['OB21']:.4f}·OB5 {r['OB5']:.4f})보다 크다.",
              f"- **스모크 실행 공개(m2).** {A1_REVIEW['smoke']}",
              f"- **렌더러 수정 공개(m3).** {A1_REVIEW['renderer']}", ""]
    L += ["## 데이터 고정 (PO-4.2)", "",
          f"- 구창 재확인: 가드 끈 R0 PO3.json 일치 {r['recheck_old']['R0_match']}, v4 모델 {r['recheck_old']['v4_models_checked']} 비트 동일 "
          f"{r['recheck_old']['v4_models_bit_same']}.",
          f"- `daily_bars_adjusted` 2019-01-01~{r['end']} 지문(행 수, 최대 날짜, Σround(close×100), Σvolume): 시작 {r['fingerprint_before']['daily_bars_adjusted']}, "
          f"끝 {r['fingerprint_after']['daily_bars_adjusted']} — 같음 {r['fingerprint_same']['daily_bars_adjusted']}.",
          f"- `daily_bars_adj_v2` 지문(보고만): 시작 {r['fingerprint_before']['daily_bars_adj_v2']}, 같음 {r['fingerprint_same']['daily_bars_adj_v2']}.", "",
          "## 검정력 (재현 실행에서 신창 전에 고정, 보고만)", "",
          "| 세계 | A 실패(가족) | 1차 OB21 | 보조 OB5 |", "|---|---|---|---|"]
    for sc, d in r["power_report"].items():
        L.append(f"| {sc} | {d['family']:.3f} | {d['primary']:.3f} | {d['secondary']:.3f} |")
    L += ["", f"X = {r['X']}, Y = {r['Y']}(구창 표본 안 그대로인 세계의 실패 확률 %).", "",
          "## 보고만 — 같은 OB 포트를 다른 예측·다른 실현으로", "",
          "| 판 | OB21 | OB5 |", "|---|---|---|",
          f"| v4 (판정) | {r['OB21']:.4f} | {r['OB5']:.4f} |",
          f"| v3 예측(같은 포트) | {_f(ro['v3_same_ob']['OB21'], 4)} | {_f(ro['v3_same_ob']['OB5'], 4)} |",
          f"| v4 예측·`daily_bars_adj_v2` 실현 | {_f(ro['v2_same_ob']['OB21'], 4)} | {_f(ro['v2_same_ob']['OB5'], 4)} |", "",
          "| 구간 | v4 OB21 | v4 OB5 | v3 OB21 | v3 OB5 | v2 OB21 | v2 OB5 |", "|---|---|---|---|---|---|---|"]
    for nm, d in ro["years"].items():
        L.append(f"| {nm} | " + " | ".join(_f(d[k][t], 4) for k in ("v4", "v3", "v2") for t in ("OB21", "OB5")) + " |")
    L += ["", "2026-07 을 뺀 값은 실패를 구제하는 데 쓸 수 없다(§1). 판정은 31회 전체로만 한다.", "",
          "## 보고만 — 그 밖의 대상 (신창 실현 바이어스)", "",
          ", ".join(f"{k} {_f(v)}" for k, v in ro["realized"].items() if k not in ("OB21", "OB5")), "",
          "capw·LO5·LO50·KOSPI 는 시장 수준 대상(§0 — 국면에 지배, 보고만)이다. KOSPI 는 지수 21일 수익(일수익 합) / 시총가중 포트의 v4 사전 21일 위험(근사).", "",
          f"- 업종 방향 재현(구창 G55>1·G25<1·G10<1): {ro['sector_direction']}.",
          f"- 종목별 특이 z 바이어스 중앙값 {_f(ro['spec_bias_median'])}(종목 {ro['spec_bias_n']}).",
          f"- 앞 21일 NaN→0 셀 — B 유니버스: 이음매 {ro['nan_to_zero_B']['seam']}·그 밖 {ro['nan_to_zero_B']['other']}"
          f"(v2 실현에서 NaN {ro['nan_to_zero_B']['v2']}); 모델 종목 전체: {ro['nan_to_zero_all']['seam']}·{ro['nan_to_zero_all']['other']}.",
          f"- 꼬리(사후, 밴드에 안 씀): CV_s 1차 {_f(ro['tails_post_hoc']['CV_med'])} [{'·'.join(_f(v) for v in ro['tails_post_hoc']['CV_med_q05_50_95'])}], "
          f"확인 {_f(ro['tails_post_hoc']['CV_split'])}; ν_u {_f(ro['tails_post_hoc']['nu_u'], 2)}(자르기 전 {_f(ro['tails_post_hoc']['nu_u_unclipped'], 2)}, "
          f"초과첨도 {_f(ro['tails_post_hoc']['kurt_u'], 2)}); ν_f {_f(ro['tails_post_hoc']['nu_f'], 2)}. 구창 v4: CV_s 0.324, ν_u 4.50(자르기 전 4.42).",
          f"- 이음매 가드(신창 로드 전 구간): {r['seam']['seam_cells']}셀, 원천별 {r['seam']['seam_cells_by_source']}, 판정 불가 {r['seam']['cells_undeterminable']}.",
          f"- 실적: END 이하 {r['signal']['earnings_rows_read_le_end']}행, 뺀 행 {r['signal']['dropped_2019Q1plus_no_periodic_le_end']}, 쓴 행 {r['signal']['rows_used']}. "
          f"B 유니버스 {r['B_universe_size'][0]}~{r['B_universe_size'][1]}.", "",
          "## 보고만 — `daily_bars_adj_v2` 대비 일간 수익률 차이 (신창 2024-01-02~2026-09-11)", "",
          "| 칸 | 가드 | 비교 칸 | >1% | >5% | >20% |", "|---|---|---|---|---|---|"]
    v2t = ro["v2_tables"]
    for nm, d in v2t["diff_table"].items():
        for g, cc in d.items():
            L.append(f"| {nm} | {g} | {cc['cells']} | {cc['>1%']} | {cc['>5%']} | {cc['>20%']} |")
    cl, ld, lu = v2t["cluster"], v2t["erased_limit_down"], v2t["erased_limit_up"]
    L += ["", f"- 군집형 이음매({cl['rule']}): 전체 {cl['all']}, B 앞 21일 창 {cl['B_fwd21']}(가드가 NaN 으로 바꾼 것 {cl['B_fwd21_guard_nan']}). "
          f"2024-01-08: 전체 {cl['on_2024-01-08_all']}, B 창 {cl['on_2024-01-08_B_fwd21']}. 날짜 상위(B 창): "
          + ", ".join(f"{d} {n}" for d, n in cl["top_dates_B_fwd21"]) + ".",
          f"- 지워진 하한가({ld['rule']}): 전체 {ld['all']}, B 앞 21일 창 {ld['B_fwd21']}. 001380 2025-03-26 이 B 창 안: {ld['001380_2025-03-26_in_B_fwd21']}.",
          f"- 지워진 상한가({lu['rule']}): 전체 {lu['all']}, B 앞 21일 창 {lu['B_fwd21']}.", "",
          "| 종류 | 종목 | 날짜 | adjusted | v2 | 가드 NaN |", "|---|---|---|---|---|---|"]
    for kind, rows in (("군집(B 창)", cl["cells_B_fwd21"][:15]), ("하한가", ld["cells_all"][:15]), ("상한가", lu["cells_all"][:10])):
        for c_ in rows:
            L.append(f"| {kind} | {c_['code']} | {c_['date']} | {c_['ret_adj']:+.3f} | {c_['ret_v2']:+.3f} | {c_['guard_nan']} |")
    L += ["", f"_{_f(r.get('elapsed_s'))}초_"]
    return "\n".join(L) + "\n"


_T0 = [0.0]
_OUTDIR = [OUT]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("repro", "new", "smoke"), required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--asof-from", required=True)
    ap.add_argument("--asof-to", required=True)
    ap.add_argument("--n-asof", type=int, required=True)
    ap.add_argument("--cache", required=True, help="레포 밖, 새 디렉터리")
    ap.add_argument("--po3-cache", required=True, help="PO-3 캐시(비트 대조용)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prereg-sha", default="")
    ap.add_argument("--repro-sha", default="")
    ap.add_argument("--render", action="store_true", help="JSON → MD 만")
    ap.add_argument("--smoke-out", default="", help="smoke 출력 디렉터리(레포 밖)")
    args = ap.parse_args()
    _check_end(args.end)
    stem = {"repro": "PO4_repro", "new": "A1", "smoke": "A1_smoke"}[args.mode]
    args.stem = stem
    if args.mode == "smoke":
        assert args.smoke_out, "smoke 는 --smoke-out 필요"
        _OUTDIR[0] = Path(args.smoke_out)
    if args.render:
        r = json.loads((OUT / f"{stem}.json").read_text())
        (OUT / f"{stem}.md").write_text(render_repro(r) if stem == "PO4_repro" else render_a1(r))
        return 0
    if args.mode == "smoke":
        assert (args.asof_from, args.asof_to, args.n_asof) == ("2023-06-01", "2023-10-31", 5)
    cache = Path(args.cache)
    repo = Path(__file__).resolve().parents[2]
    assert repo not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    assert not cache.exists() or not any(cache.iterdir()), "캐시는 새 디렉터리여야 한다(처음부터 다시 만든다)"
    cache.mkdir(parents=True, exist_ok=True)
    _T0[0] = time.time()

    def log(msg):
        print(f"[{time.time() - _T0[0]:6.0f}s] {msg}", flush=True)

    return run_repro(args, cache, log) if args.mode == "repro" else run_new(args, cache, log)


if __name__ == "__main__":
    raise SystemExit(main())
