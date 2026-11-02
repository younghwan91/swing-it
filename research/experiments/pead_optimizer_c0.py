"""PEAD 최적화 사전등록 C0 — 구창(2019-07~2023-10) 보정. 판정 러너가 아니다.

사전등록: ``research/logs/pead_optimizer/PREREG.md`` 2판(d14b288) + 정정 PO-1(3판, 5f1458b).
이 러너는 PO-1.8-2 의 C0 항목만 낸다 — R0 재현, CV_s·ν_u·ν_f, 구창 블록 부트스트랩 SE, 구창 b(D=0),
σ_target, 구창 모델 구조 모의(T 격자 + 31 + 52 의 α·밴드·탐지확률), T_fwd, 구창 전지 실현값과
구창 밴드 대조(STOP 여부). **2023-12-28 뒤 가격·수익은 읽지 않는다** — 위험 입력은
``load_risk_inputs(…, END)`` 로, 실적은 알게 된 날 ≤ END 로 자른다.

숫자만 낸다(GUARDRAILS §8). STOP 은 사전등록 규칙의 기계적 적용 결과를 그대로 적는 것이다.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@d5c1533" \\
      python research/experiments/pead_optimizer_c0.py --cache <scratch> [--workers 16]

캐시(--cache)는 레포 밖에 둔다(위험모델·OB 포트). 출력: ``research/logs/pead_optimizer/C0.json``·``C0.md``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _trailing_adv
from krx_quant_core.data.risk_inputs import STYLE_NAMES, load_risk_inputs, style_panels
from krx_quant_core.risk.factor_model import (
    bias_statistic,
    build_risk_model,
    factor_returns,
    standardize,
)
from scipy import stats

from swing_it.book.risk import Sub
from swing_it.features.universe import rank_band_eligible
from swing_it.validation.keys import SEED, keyed_rng

__all__ = ["SEED", "Sub", "keyed_rng"]  # 옮긴 구현을 이 이름으로 부르는 러너가 있다

START, END = "2019-07-01", "2023-12-28"
HORIZON = 21
N_RANDOM, RANDOM_NAMES = 100, 50  # R0 (validate_risk_model.py 그대로)
N_P = 100  # P5·P50·P10B 각 포트 수
N_OB = 200
ADV_FLOOR = 20000.0  # 백만원
CAP_RANK = (20, 100)  # 시총 21~100위
GRID_START, GRID_LAST = "2019-08-01", "2023-10-31"
STEP = 20
OB_MAX_WEIGHT, OB_SECTOR_BAND = 0.10, 0.10
LAM_LO, LAM_HI, LAM_ITERS, LAM_TOL = 1e-2, 1e5, 40, 0.02
T_GRID = (24, 30, 36, 40, 48, 60)
T_REPORT = (31, 52)
N_SIM = 1000
N_BOOT = 2000
BOOT_BLOCKS = (3, 6)
FAMILY_PASS = 0.90
MAX_H = 0.35
DETECT_MIN = 0.8
SECTOR_VOL = 1.3
VOID_SET = ("Dx0", "Dx0.5", "Dx2", "D0pred", "Fx0.7", "Fx1.5")
R0_DESIGN = {  # 설계 §6 v3 열(krx-quant-core docs/superpowers/specs/2026-11-01-portfolio-layer-design.md)
    "random": 0.974, "capw": 1.002, "style:size": 1.012, "style:beta": 0.987,
    "style:momentum": 0.908, "style:resid_vol": 0.873, "style:liquidity": 0.834,
    "style:value": 0.905,
}
OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"


# ── 입력 ──────────────────────────────────────────────────────────────────────


def exposures(x, raw, mask):  # validate_risk_model.py 와 같은 규칙
    out = {}
    for k, v in raw.items():
        e = np.full_like(v, np.nan)
        for t in range(v.shape[1]):
            m = mask[:, t]
            if m.sum() < 30:
                continue
            col = standardize(v[:, t], x.cap[:, t], m)
            col[m & ~np.isfinite(col)] = 0.0
            e[:, t] = col
        out[k] = e
    return out


def month_ends(dates, lo, last_index):  # validate_risk_model.py 그대로
    idx = []
    for i in range(len(dates) - 1):
        if dates[i] >= np.datetime64(lo) and dates[i].month != dates[i + 1].month:
            if i + HORIZON <= last_index:
                idx.append(i)
    return idx


def yoy_signal(codes, dates, end: str = END, fallback: bool = True) -> tuple[np.ndarray, dict]:
    ea, per = yoy_inputs(end, fallback)
    return yoy_wide(ea, per, codes, dates, end, fallback)


def yoy_inputs(end: str = END, fallback: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """분기 순이익 YoY 패널(code × date). 알게 된 날 = max(avail, 정기보고서 첫 rcept_dt) 다음 거래일
    (fcs ``storage._attach_known_dates`` 와 같은 식). 정정 버전은 ``knowledge_col`` 로 날짜별 선택.

    ``end``: 실적 행을 자르는 상한(기본 ``END`` — 기존 동작). ``fallback=False`` 면 정기보고서 접수일이 없는 분기를
    avail_date 로 돌리지 않고 뺀다(ML 예측 사전등록 F1 MINOR f). 기본 ``True`` 는 기존 동작 그대로다."""
    from swing_it.storage import connect, db_default, read_earnings

    con = connect(db_default())
    con.set_session(readonly=True)
    cur = con.cursor()
    cur.execute("SET statement_timeout='60s'")
    ea = read_earnings(con, all_versions=True, until=None if fallback else end, cols=(
        "code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior"))
    # 정기보고서 접수일은 END 뒤 것도 읽는다(날짜만). avail ≤ END 인데 정기보고서가 END 뒤에 나온 분기는
    # 그래야 "END 전에는 몰랐다"로 빠진다 — 자르면 avail 로 돌아가 미래 정보가 된다.
    # fallback=False 면 접수일 없는 분기는 어차피 빠지므로 END 에서 잘라도 결과가 같다(END 뒤 행을 읽지 않는다).
    if fallback:
        cur.execute(
            "SELECT code, period, min(rcept_dt) FROM earnings_disclosures "
            "WHERE kind = 'periodic' GROUP BY code, period")
    else:
        cur.execute(
            "SELECT code, period, min(rcept_dt) FROM earnings_disclosures "
            "WHERE kind = 'periodic' AND rcept_dt <= %s GROUP BY code, period", (str(pd.Timestamp(end).date()),))
    per = pd.DataFrame(cur.fetchall(), columns=["code", "period", "per"])
    con.close()
    return ea, per


def yoy_wide(ea: pd.DataFrame, per: pd.DataFrame, codes, dates, end, fallback: bool = True) -> tuple[np.ndarray, dict]:
    """:func:`yoy_signal` 의 DB 뒤 부분(순수) — 실적 행 ``ea``·정기보고서 첫 접수일 ``per`` → (code × date 패널, 보고)."""
    from swing_it.features.fundamentals import _yoy_vec, earnings_yoy_panel

    end = pd.Timestamp(end)
    ea["avail_date"] = pd.to_datetime(ea["avail_date"])
    ea["knowledge_date"] = pd.to_datetime(ea["knowledge_date"])
    n_all = len(ea)
    ea = ea[(ea["avail_date"] <= end) & (ea["knowledge_date"] <= end)].copy()
    ea["code"] = ea["code"].astype(str)
    per["code"] = per["code"].astype(str)
    per["per"] = pd.to_datetime(per["per"])
    ea = ea.merge(per, on=["code", "period"], how="left")
    n_no_periodic = int(ea["per"].isna().sum())
    if not fallback:
        ea = ea[ea["per"].notna()].copy()
    known = ea[["avail_date", "per"]].max(axis=1).to_numpy(dtype="datetime64[ns]")
    days = pd.DatetimeIndex(dates)
    i = days.searchsorted(known, side="right")
    ok = (i < len(days)) & ~np.isnat(known)
    ea = ea[ok].copy()
    ea["known_date"] = days[i[ok]].strftime("%Y-%m-%d")
    ea["knowledge_date"] = ea["knowledge_date"].dt.strftime("%Y-%m-%d")
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    # fallback=False(ML 예측)만 동률을 늦은 분기로 정한다(정정 ML-2). 기본 경로는 PEAD 패리티에 묶여 그대로다.
    panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), list(days.strftime("%Y-%m-%d")),
                               avail_col="known_date", knowledge_col="knowledge_date",
                               tie_col=None if fallback else "period")
    wide = panel.pivot(index="code", columns="date", values="yoy")
    wide = wide.reindex(index=[str(c) for c in codes], columns=list(days.strftime("%Y-%m-%d")))
    info = {"earnings_rows_read": int(n_all), "earnings_rows_known_le_end": int(len(ea)),
            "periodic_rcept_missing_share": float(ea["per"].isna().mean())}
    if not fallback:
        info["dropped_no_periodic"] = n_no_periodic
    return wide.to_numpy(dtype=np.float64), info


# ── B 유니버스 ────────────────────────────────────────────────────────────────


def eligible_idx(t, sig, adv, cap, in_u):
    """eligible(t) — M0(신호 유한 ∧ ADV ≥ 200억 ∧ U_t) 안의 시총 21~100위(행 번호). 구현은 ``features.universe`` 로 옮겼다."""
    return rank_band_eligible(t, sig, adv, cap, in_u, adv_floor=ADV_FLOOR, cap_rank=CAP_RANK)


# ── OB(PO-1.6) ────────────────────────────────────────────────────────────────


def _ob_task(args):
    from krx_quant_core.portfolio.optimize import Constraints, optimize

    n, X, F, D, names, sect, sigma_t = args
    nb = len(D)
    sub = Sub(X=X, F=21.0 * F, D=21.0 * D, names=names)
    b = np.full(nb, 1.0 / nb)
    Sb = sub.X @ (sub.F @ (sub.X.T @ b)) + sub.D * b
    con = Constraints(budget=1.0, long_only=True, max_weight=OB_MAX_WEIGHT, sector=sect,
                      benchmark=b, sector_band=OB_SECTOR_BAND)
    rng = keyed_rng("OB", n)
    W = np.full((N_OB, nb), np.nan)
    stat = {"ok": 0, "lam_fail": 0, "solver_fail": 0, "solves": 0}

    def act_risk(w):
        d = w - b
        y = sub.X.T @ d
        return float(np.sqrt(y @ sub.F @ y + (d * d) @ sub.D))

    for p in range(N_OB):
        eps = rng.standard_normal(nb)
        alpha = np.sqrt(21.0 * D) * eps

        def solve(lam, alpha=alpha):
            stat["solves"] += 1
            r = optimize(alpha + lam * Sb, sub, None, constraints=con, risk_aversion=lam)
            return r

        lo, hi = np.log(LAM_LO), np.log(LAM_HI)
        got = None
        for _ in range(LAM_ITERS):
            mid = 0.5 * (lo + hi)
            r = solve(float(np.exp(mid)))
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
            stat["lam_fail"] += 1 if r.ok else 0
            continue
        W[p] = got - b
        stat["ok"] += 1
    return n, W, stat


# ── 통계 도우미 ───────────────────────────────────────────────────────────────


def nu_from(z):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    k = float(stats.kurtosis(z, fisher=True, bias=False))
    return float(np.clip(4.0 + 6.0 / max(k, 0.2), 4.5, 30.0)), k, int(z.size)


def bias_rows(Z):
    """Z (P, T) z → 포트별 std(ddof=1, 유한만)의 평균."""
    Z = np.atleast_2d(Z)
    out = []
    for row in Z:
        r = row[np.isfinite(row)]
        if r.size >= 2:
            out.append(r.std(ddof=1))
    return float(np.mean(out)) if out else float("nan")


def boot_se(Zs: dict, T: int, sim_center: dict, sim_se: dict, styles: list) -> dict:
    """PO-1.1-5 — 원형 블록 부트스트랩 SE(L ∈ {3,6} 최대). Zs: 대상 → (P, 52) z."""
    A = next(iter(Zs.values())).shape[1]
    out = {}
    for L in BOOT_BLOCKS:
        rng = keyed_rng("boot", T, L)
        nb = -(-T // L)
        starts = rng.integers(0, A, size=(N_BOOT, nb))
        idx = ((starts[:, :, None] + np.arange(L)[None, None, :]) % A).reshape(N_BOOT, -1)[:, :T]
        bstar = {}
        for key, Z in Zs.items():
            G = Z[:, idx]  # (P, B, T)
            with np.errstate(invalid="ignore"):
                sd = np.nanstd(G, axis=2, ddof=1)
            bstar[key] = np.nanmean(sd, axis=0)
        s_star = np.mean([(bstar[k] - sim_center[k]) / sim_se[k] for k in styles], axis=0)
        bstar["S*"] = s_star
        for key, v in bstar.items():
            out[key] = max(out.get(key, 0.0), float(np.nanstd(v, ddof=1)))
    return out


# ── 본체 ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="레포 밖 캐시 디렉터리")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--render", action="store_true", help="C0.json → C0.md 만")
    args = ap.parse_args()
    if args.render:
        (OUT / "C0.md").write_text(render_md(json.loads((OUT / "C0.json").read_text())))
        return 0
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    assert repo not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    # 1. 입력(구창만) ------------------------------------------------------------
    x = load_risk_inputs("2019-01-02", END)
    assert x.dates[-1] <= pd.Timestamp(END), "END 뒤 날짜가 들어왔다"
    sector = x.sector
    has_sec = np.vectorize(lambda s: s is not None, otypes=[bool])(sector)
    mask = has_sec & np.isfinite(x.cap) & (x.cap > 0) & np.isfinite(x.close)
    raw = style_panels(x)
    Xall = exposures(x, raw, mask)
    sig_all, sig_info = yoy_signal(x.codes, x.dates)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    first = int(np.argmax(has_sec.any(axis=0)))
    sl = slice(first, None)
    R, cap, msk, lab = x.ret[:, sl], x.cap[:, sl], mask[:, sl], sector[:, sl]
    Xs = {k: v[:, sl] for k, v in Xall.items()}
    sig, adv = sig_all[:, sl], adv_all[:, sl]
    dates = x.dates[sl]
    nD = len(dates)
    asofs = month_ends(dates, START, R.shape[1] - 1)
    sectors = sorted({s for s in lab[:, -1] if s is not None})
    G = len(sectors)
    log(f"입력 — 종목 {len(x.codes)}, 날짜 {nD}({dates[0].date()}~{dates[-1].date()}), asof {len(asofs)}")

    # 2. 위험모델(월말 + 격자일) — 캐시 -------------------------------------------
    t_0 = int(np.searchsorted(dates, np.datetime64(GRID_START)))
    grid = list(range(t_0, nD, STEP))
    grid = [t for t in grid if dates[t] <= pd.Timestamp(GRID_LAST)]
    need = sorted(set(asofs) | set(grid))
    mpath = cache / "models.pkl"
    if mpath.exists():
        models = pickle.load(open(mpath, "rb"))
    else:
        models = {}
        for j, a in enumerate(need):
            m = build_risk_model(R, Xs, lab, cap, msk, a, codes=np.arange(R.shape[0]))
            models[a] = (m.codes.astype(int), m.X, m.F, m.D, list(m.names))
            if j % 20 == 0:
                log(f"  모델 {j}/{len(need)} {dates[a].date()}")
        pickle.dump(models, open(mpath, "wb"))
    log(f"모델 {len(models)}개")

    def model(a):
        from krx_quant_core.risk.factor_model import RiskModel

        idx, X, F, D, names = models[a]
        return RiskModel(codes=idx, names=names, X=X, F=F, D=D), idx

    # 3. σ_target(PO-1.6) — 격자일 순위 북 R 목표 비중의 사전 21일 능동위험 --------
    tranche = {}
    for t in grid:
        _, idx = model(t)
        in_u = np.zeros(R.shape[0], bool)
        in_u[idx] = True
        el = eligible_idx(t, sig, adv, cap, in_u)
        top = el[np.argsort(-sig[el, t], kind="stable")[:10]] if el.size >= 20 else None
        tranche[t] = (el, top)
    sig_rows = []
    for k, t in enumerate(grid):
        if k < 2:
            continue
        m, idx = model(t)
        pos = {c: i for i, c in enumerate(idx)}
        el, _ = tranche[t]
        w = np.zeros(len(idx))
        nt = 0
        for j in range(3):
            top = tranche[grid[k - j]][1]
            if top is None:
                continue
            nt += 1
            for c in top:
                if c in pos:
                    w[pos[c]] += 1.0 / len(top)
        if nt == 0 or w.sum() <= 0:
            continue
        w /= w.sum()
        b = np.zeros(len(idx))
        eu = [pos[c] for c in el if c in pos]
        b[eu] = 1.0 / len(eu)
        sig_rows.append({"date": str(dates[t].date()), "sigma21": float(np.sqrt(HORIZON * m.var(w - b))),
                         "n_tranche": nt, "n_eligible": int(el.size)})
    sigma_target = float(np.median([r["sigma21"] for r in sig_rows]))
    log(f"σ_target(21일) {sigma_target:.4f} — 격자일 {len(sig_rows)}")

    # 4. OB 포트(월말) — 병렬·캐시 -----------------------------------------------
    bu = {}
    for n, a in enumerate(asofs):
        m, idx = model(a)
        in_u = np.zeros(R.shape[0], bool)
        in_u[idx] = True
        el = eligible_idx(a, sig, adv, cap, in_u)
        pos = {c: i for i, c in enumerate(idx)}
        bu[n] = np.array([pos[c] for c in el], dtype=int)
    opath = cache / f"ob_{sigma_target:.6f}.pkl"
    if opath.exists():
        ob = pickle.load(open(opath, "rb"))
    else:
        tasks = []
        for n, a in enumerate(asofs):
            m, idx = model(a)
            s = bu[n]
            tasks.append((n, m.X[s], m.F, m.D[s], m.names, lab[idx[s], a].astype(object), sigma_target))
        ob = {}
        with mp.get_context("fork").Pool(args.workers) as pool:
            for n, W, st in pool.imap_unordered(_ob_task, tasks):
                ob[n] = (W, st)
                if len(ob) % 8 == 0:
                    log(f"  OB {len(ob)}/{len(asofs)}")
        pickle.dump(ob, open(opath, "wb"))
    ob_stat = {k: int(sum(ob[n][1][k] for n in ob)) for k in ("ok", "lam_fail", "solver_fail", "solves")}
    log(f"OB {ob_stat}")

    # 5. 일별 잔차(실현 특이 z 용) --------------------------------------------------
    upath = cache / "resid.npy"
    if upath.exists():
        U = np.load(upath)
    else:
        U = factor_returns(R, Xs, lab, cap, msk).u
        np.save(upath, U)

    # 6. 월말 루프 — 실현 + 모의 구조 ---------------------------------------------
    rng_r0 = np.random.default_rng(SEED)
    rng_diag = np.random.default_rng(SEED + 1)
    rng_t = {k: keyed_rng(k) for k in ("P5", "P50", "P10B")}
    r0 = {"random": [], "capw": [], **{f"style:{k}": [] for k in STYLE_NAMES}}
    post = {"random5": [], "hivol_decile": [], "small_decile": [],
            **{f"sector_pure:{g}": [] for g in sectors}}
    # 1차 묶음 슬롯(고정 순서)
    slots = ([("S", k) for k in STYLE_NAMES] + [("P5", i) for i in range(N_P)]
             + [("P50", i) for i in range(N_P)] + [("P10B", i) for i in range(N_P)]
             + [("HV", 0), ("SM", 0)] + [("IP", g) for g in sectors] + [("IE", g) for g in sectors]
             + [("OB", i) for i in range(N_OB)])
    slot_target = [f"S:{k}" if t == "S" else (f"{t}:{k}" if t in ("IP", "IE") else t) for t, k in slots]
    targets = list(dict.fromkeys(slot_target))
    PS = len(slots)
    A = len(asofs)
    real = np.full((PS, A), np.nan)
    pred = np.full((PS, A), np.nan)
    pred0 = np.full((PS, A), np.nan)
    struct = []
    spec_z, spec_rows = [], []
    nan_cells = 0
    sec_names_idx = []
    for n, a in enumerate(asofs):
        m, idx = model(a)
        nA = len(idx)
        fwd_raw = R[idx, a + 1:a + 1 + HORIZON]
        nan_cells += int((~np.isfinite(fwd_raw)).sum())
        fwd = np.nan_to_num(fwd_raw, nan=0.0).sum(axis=1)

        def z(w, fwd=fwd, m=m):
            return float(w @ fwd), float(np.sqrt(HORIZON * m.var(w)))

        rnd = []
        for _ in range(N_RANDOM):
            w = np.zeros(nA)
            w[rng_r0.choice(nA, RANDOM_NAMES, replace=False)] = 1.0 / RANDOM_NAMES
            rnd.append(z(w))
        r0["random"].append(rnd)
        cw = cap[idx, a] / cap[idx, a].sum()
        r0["capw"].append(z(cw))
        Wr = np.sqrt(cap[idx, a])
        P = np.linalg.pinv((m.X * Wr[:, None]).T @ m.X) @ (m.X * Wr[:, None]).T
        for k in STYLE_NAMES:
            r0[f"style:{k}"].append(z(P[m.names.index(k)]))
        r5 = []
        for _ in range(N_RANDOM):
            w = np.zeros(nA)
            w[rng_diag.choice(nA, 5, replace=False)] = 0.2
            r5.append(z(w))
        post["random5"].append(r5)

        bU = np.full(nA, 1.0 / nA)
        Wm = np.zeros((PS, nA))
        row = 0
        for k in STYLE_NAMES:
            Wm[row] = P[m.names.index(k)]
            row += 1
        for key, size in (("P5", 5), ("P50", 50)):
            for _ in range(N_P):
                w = np.zeros(nA)
                w[rng_t[key].choice(nA, size, replace=False)] = 1.0 / size
                Wm[row] = w - bU
                row += 1
        s = bu[n]
        bB = np.zeros(nA)
        bB[s] = 1.0 / len(s)
        for _ in range(N_P):
            w = np.zeros(nA)
            w[rng_t["P10B"].choice(s, 10, replace=False)] = 0.1
            Wm[row] = w - bB
            row += 1
        for key, raw_k, top in (("hivol_decile", "resid_vol", True), ("small_decile", "size", False)):
            v = raw[raw_k][idx, a + first]
            ok = np.isfinite(v)
            cut = np.nanquantile(v, 0.9 if top else 0.1)
            sel = (ok & ((v >= cut) if top else (v <= cut))).astype(float)
            post[key].append(z(sel / sel.sum()))
            Wm[row] = sel / sel.sum() - bU
            row += 1
        labs_a = lab[idx, a]
        sec_pos = []
        for g in sectors:
            if g in m.names:
                Wm[row] = P[m.names.index(g)]
                post[f"sector_pure:{g}"].append(z(P[m.names.index(g)]))
            else:
                Wm[row] = np.nan
            sec_pos.append(m.names.index(g) if g in m.names else -1)
            row += 1
        for g in sectors:
            sel = (labs_a == g).astype(float)
            Wm[row] = sel / sel.sum() - bU if sel.sum() > 0 else np.nan
            row += 1
        Wob, _ = ob[n]
        for p in range(N_OB):
            if np.isfinite(Wob[p]).all():
                w = np.zeros(nA)
                w[s] = Wob[p]
                Wm[row] = w
            else:
                Wm[row] = np.nan
            row += 1
        assert row == PS
        good = np.isfinite(Wm).all(axis=1)
        Wg = np.where(good[:, None], Wm, 0.0)
        Y = Wg @ m.X
        fv = np.einsum("pk,kl,pl->p", Y, m.F, Y)
        sv = (Wg * Wg) @ m.D
        real[good, n] = (Wg @ fwd)[good]
        pred[good, n] = np.sqrt(HORIZON * (fv + sv))[good]
        pred0[good, n] = np.sqrt(HORIZON * fv)[good]
        struct.append({"Y": Y, "Wsd": Wg * np.sqrt(HORIZON * m.D)[None, :], "F": m.F, "good": good,
                       "pred": np.sqrt(HORIZON * (fv + sv)), "pred0": np.sqrt(HORIZON * fv)})
        sec_names_idx.append(sec_pos)
        # 실현 특이 z(PO-1.1-1)
        u = U[idx, a + 1:a + 1 + HORIZON]
        okr = np.isfinite(u).all(axis=1)
        zz = u[okr].sum(axis=1) / np.sqrt(HORIZON * m.D[okr])
        spec_z.append(zz)
        spec_rows.append((idx[okr], zz))
        if n % 6 == 0:
            log(f"  {dates[a].date()} 종목 {nA} B유니버스 {len(s)}")

    # 7. R0 ---------------------------------------------------------------------
    rr = np.array(r0["random"])
    r0_out = {"random": bias_statistic(rr[:, :, 1].T, rr[:, :, 0].T)["mean"]}
    for key in ["capw", *[f"style:{k}" for k in STYLE_NAMES]]:
        arr = np.array(r0[key])
        r0_out[key] = bias_statistic(arr[:, 1], arr[:, 0])["mean"]
    r0_diff = {k: r0_out[k] - R0_DESIGN[k] for k in R0_DESIGN}
    r0_ok = all(abs(v) <= 0.005 for v in r0_diff.values())
    r5 = np.array(post.pop("random5"))
    post_out = {"random5": bias_statistic(r5[:, :, 1].T, r5[:, :, 0].T)["mean"]}
    for key, vals in post.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        post_out[key] = bias_statistic(arr[:, 1], arr[:, 0])["mean"]
    log(f"R0 {'일치' if r0_ok else '불일치'} {r0_diff}")
    if not r0_ok:  # 사전등록 C0-1: DB 가 바뀐 것 — 원인을 적고 멈춘다
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "C0.json").write_text(json.dumps(
            {"R0": {"values": r0_out, "design": R0_DESIGN, "diff": r0_diff, "match_0.005": False},
             "STOP": True, "reason": "R0 불일치"}, ensure_ascii=False, indent=1, default=float))
        log("R0 불일치 — 멈춘다")
        return 1

    # 8. 실현 z·꼬리·혼합 ---------------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        Zr = real / pred
        Zr0 = real / pred0
    slot_arr = np.array(slot_target)
    Zt = {t: Zr[slot_arr == t] for t in targets}
    Zt0 = {t: Zr0[slot_arr == t] for t in targets}
    ms = np.array([np.mean(z_ * z_) for z_ in spec_z])
    vs = np.array([np.var(z_ * z_, ddof=1) for z_ in spec_z])
    ns = np.array([z_.size for z_ in spec_z])
    s2 = ms / ms.mean()
    cv2_raw = float(np.var(s2, ddof=1))
    noise = float(np.mean(vs / ns) / ms.mean() ** 2)
    cv_s = float(np.sqrt(max(cv2_raw - noise, 0.0)))
    nu_u, k_u, n_u = nu_from(np.concatenate([z_ / np.sqrt(s) for z_, s in zip(spec_z, s2)]))
    nu_u_raw, k_u_raw, _ = nu_from(np.concatenate(spec_z))
    fac_t = [f"S:{k}" for k in STYLE_NAMES] + [f"IP:{g}" for g in sectors]
    pooled = []
    for t in fac_t:
        zz = Zt[t].ravel()
        zz = zz[np.isfinite(zz)]
        pooled.append(zz / zz.std(ddof=1))
    nu_f, k_f, n_f = nu_from(np.concatenate(pooled))
    _, k_f_raw, _ = nu_from(np.concatenate([Zt[t].ravel() for t in fac_t]))
    # 종목별 특이 z(보고만): 종목마다 √mean z², 관측 ≥ 24
    acc: dict[int, list] = {}
    for codes_, zz in spec_rows:
        for c, v in zip(codes_, zz):
            acc.setdefault(int(c), []).append(v)
    per_stock = [np.sqrt(np.mean(np.square(v))) for v in acc.values() if len(v) >= 24]
    log(f"CV_s {cv_s:.3f}(raw {np.sqrt(cv2_raw):.3f}) ν_u {nu_u:.2f}(κ {k_u:.2f}) ν_f {nu_f:.2f}(κ {k_f:.2f})")

    # 9. 구창 실현 전지 -------------------------------------------------------------
    realized = {t: bias_rows(Zt[t]) for t in targets}
    realized_d0 = {t: bias_rows(Zt0[t]) for t in ("P5", "P50", "P10B", "HV")}

    # 10. 모의(구창 구조, PO-1.1·1.2) ---------------------------------------------
    scen = ["correct", "Dx0", "Dx0.5", "Dx2", "D0pred", "Fx0.7", "Fx1.5"] + [f"sec:{g}" for g in sectors]
    S = len(scen)
    Tmax = max(T_GRID + T_REPORT)
    checks = sorted(set(T_GRID) | set(T_REPORT))
    S1 = np.zeros((S, PS, N_SIM))
    S2 = np.zeros((S, PS, N_SIM))
    NN = np.zeros((PS, 1))
    sig_ln = np.sqrt(np.log(1.0 + cv_s ** 2))
    sims = {}
    for t in range(Tmax):
        st = struct[t % A]
        good = st["good"]
        rng = keyed_rng("sim", t)
        F21 = HORIZON * st["F"]
        ev, evec = np.linalg.eigh(0.5 * (F21 + F21.T))
        Gt = evec * np.sqrt(np.clip(ev, 0, None))[None, :]  # f = Gt z
        K = F21.shape[0]
        zf = rng.standard_normal((K, N_SIM))
        chi = rng.chisquare(nu_f, N_SIM)
        f = (Gt @ zf) * np.sqrt((nu_f - 2.0) / chi)[None, :]
        nA = st["Wsd"].shape[1]
        eps = rng.standard_t(nu_u, (nA, N_SIM)) * np.sqrt((nu_u - 2.0) / nu_u)
        s_t = np.sqrt(np.exp(rng.normal(-0.5 * sig_ln ** 2, sig_ln, N_SIM))) if sig_ln > 0 else np.ones(N_SIM)
        fac = st["Y"] @ f
        spec = (st["Wsd"] @ eps) * s_t[None, :]
        pr = st["pred"][:, None]
        pr0 = st["pred0"][:, None]
        for i, sc in enumerate(scen):
            if sc == "correct":
                num, den = fac + spec, pr
            elif sc.startswith("Dx"):
                num, den = fac + np.sqrt(float(sc[2:])) * spec, pr
            elif sc == "D0pred":
                num, den = fac + spec, pr0
            elif sc.startswith("Fx"):
                num, den = np.sqrt(float(sc[2:])) * fac + spec, pr
            else:
                j = sec_names_idx[t % A][sectors.index(sc[4:])]
                num = fac + spec
                if j >= 0:
                    num = num + (SECTOR_VOL - 1.0) * np.outer(st["Y"][:, j], f[j])
                den = pr
            with np.errstate(invalid="ignore", divide="ignore"):
                zz = np.where(good[:, None], num / den, 0.0)
            S1[i] += zz
            S2[i] += zz * zz
        NN[:, 0] += good
        if t + 1 in checks:
            T = t + 1
            with np.errstate(invalid="ignore", divide="ignore"):
                var = (S2 - S1 ** 2 / NN[None]) / (NN[None] - 1.0)
                sd = np.sqrt(np.maximum(var, 0.0))
            sd[:, (NN[:, 0] < 2)] = np.nan
            sims[T] = {tg: np.nanmean(sd[:, slot_arr == tg, :], axis=1) for tg in targets}  # (S, N_SIM)
            log(f"  모의 T={T}")

    # 11. 밴드·α·탐지(PO-1.1) -----------------------------------------------------
    scalar_t = [f"S:{k}" for k in STYLE_NAMES] + ["P5", "P50", "P10B", "HV", "SM", "OB"]
    style_t = [f"S:{k}" for k in STYLE_NAMES]
    ip_t = [f"IP:{g}" for g in sectors]
    ie_t = [f"IE:{g}" for g in sectors]
    allb = scalar_t + ip_t + ie_t
    n_out_max = int(np.floor(0.2 * G))

    def evaluate(b, bands, sthr, center, se):
        """b: 대상 → (N,) 또는 스칼라. 1차 바 전부 통과 여부 (N,)."""
        ok = np.ones(np.size(b[scalar_t[0]]), bool)
        for tg in scalar_t:
            lo, hi = bands[tg]
            ok &= (b[tg] >= lo) & (b[tg] <= hi)
        sstar = np.mean([(b[k] - center[k]) / se[k] for k in style_t], axis=0)
        ok &= np.abs(sstar) <= sthr
        for grp in (ip_t, ie_t):
            out1 = np.zeros_like(ok, dtype=int)
            out15 = np.zeros_like(ok, dtype=int)
            for tg in grp:
                lo, hi = bands[tg]
                c = center[tg]
                v = b[tg]
                out1 += ((v < lo) | (v > hi)).astype(int)
                out15 += ((v < c - 1.5 * (c - lo)) | (v > c + 1.5 * (hi - c))).astype(int)
            ok &= (out1 <= n_out_max) & (out15 == 0)
        return ok, sstar

    calib = {}
    for T in checks:
        bc = {tg: sims[T][tg][0] for tg in targets}
        center = {tg: float(np.median(bc[tg])) for tg in allb}
        se_sim = {tg: float(np.std(bc[tg], ddof=1)) for tg in allb}

        def bands_at(alpha, kappa=None, bc=bc, center=center):
            out = {}
            for tg in allb:
                lo, hi = np.quantile(bc[tg], [alpha / 2, 1 - alpha / 2])
                k = 1.0 if kappa is None else kappa[tg]
                c = center[tg]
                out[tg] = (c - k * (c - lo), c + k * (hi - c))
            return out

        def sthr_at(alpha, k=1.0, bc=bc, center=center, se_sim=se_sim):
            ss = np.mean([(bc[s] - center[s]) / se_sim[s] for s in style_t], axis=0)
            return k * float(np.quantile(np.abs(ss), 1 - alpha))

        lo_a, hi_a = 1e-5, 0.5
        for _ in range(60):
            mid = 0.5 * (lo_a + hi_a)
            p = evaluate(bc, bands_at(mid), sthr_at(mid), center, se_sim)[0].mean()
            if p > FAMILY_PASS:
                lo_a = mid
            else:
                hi_a = mid
        alpha = lo_a
        bse = boot_se({tg: Zt[tg] for tg in allb}, T, center, se_sim, style_t)
        ss_sim = np.mean([(bc[s] - center[s]) / se_sim[s] for s in style_t], axis=0)
        kappa = {tg: max(1.0, bse[tg] / se_sim[tg]) for tg in allb}
        k_s = max(1.0, bse["S*"] / float(np.std(ss_sim, ddof=1)))
        bands = bands_at(alpha, kappa)
        sthr = sthr_at(alpha, k_s)
        h = {tg: (center[tg] - bands[tg][0], bands[tg][1] - center[tg]) for tg in allb}
        hmax = {tg: max(v) for tg, v in h.items()}
        det = {}
        for i, sc in enumerate(scen):
            bs = {tg: sims[T][tg][i] for tg in targets}
            det[sc] = float(1.0 - evaluate(bs, bands, sthr, center, se_sim)[0].mean())
        fam_pass = 1.0 - det["correct"]
        b_d0 = sims[T]["P5"][scen.index("D0pred")]
        a6_void = float(np.mean(b_d0 <= bands["P5"][1]))
        sec_det = [det[f"sec:{g}"] for g in sectors]
        calib[T] = {
            "alpha": alpha, "family_pass_correct": fam_pass, "kappa": kappa, "kappa_Sstar": k_s,
            "center": center, "se_sim": se_sim, "se_boot": {k: bse[k] for k in allb},
            "bands": bands, "h": h, "h_max": hmax, "max_h": max(hmax.values()),
            "argmax_h": max(hmax, key=hmax.get), "Sstar_thr": sthr, "detect": det,
            "detect_sector_mean": float(np.mean(sec_det)), "detect_sector_min": float(np.min(sec_det)),
            "A6_void_prob_correct": a6_void,
            "void_ok": bool(max(hmax.values()) <= MAX_H and all(det[s] >= DETECT_MIN for s in VOID_SET)),
        }
        log(f"  T={T} α={alpha:.4f} 가족통과 {fam_pass:.3f} max h {max(hmax.values()):.3f}"
            f"({calib[T]['argmax_h']}) 탐지 " + " ".join(f"{s}:{det[s]:.2f}" for s in VOID_SET)
            + f" 업종평균 {np.mean(sec_det):.2f}")

    t_fwd = next((T for T in T_GRID if calib[T]["void_ok"]), None)

    # 12. 구창 전지 판정(T=52 밴드, PO-1.1 STOP) ----------------------------------
    c52 = calib[52]
    bands52 = c52["bands"]
    old = {}
    for tg in scalar_t:
        lo, hi = bands52[tg]
        old[tg] = {"b": realized[tg], "lo": lo, "hi": hi, "in": bool(lo <= realized[tg] <= hi)}
    sstar_real = float(np.mean([(realized[k] - c52["center"][k]) / c52["se_sim"][k] for k in style_t]))
    grp_res = {}
    for name, grp in (("IP", ip_t), ("IE", ie_t)):
        out1 = out15 = 0
        rows = {}
        for tg in grp:
            lo, hi = bands52[tg]
            c = c52["center"][tg]
            v = realized[tg]
            o1 = not (lo <= v <= hi)
            o15 = (v < c - 1.5 * (c - lo)) or (v > c + 1.5 * (hi - c))
            out1 += o1
            out15 += o15
            rows[tg] = {"b": v, "lo": lo, "hi": hi, "out": bool(o1), "out1.5": bool(o15)}
        grp_res[name] = {"rows": rows, "n_out": out1, "n_out1.5": out15,
                         "pass": bool(out1 <= n_out_max and out15 == 0)}
    a6_real = realized_d0["P5"]
    a6_ok = bool(a6_real > bands52["P5"][1])
    bars = {
        "A-1": all(old[k]["in"] for k in style_t) and abs(sstar_real) <= c52["Sstar_thr"],
        "A-2": all(old[k]["in"] for k in ("P5", "P50", "P10B")),
        "A-3": all(old[k]["in"] for k in ("HV", "SM")),
        "A-4": grp_res["IP"]["pass"], "A-5": grp_res["IE"]["pass"],
        "A-6_not_void": a6_ok, "A-7": old["OB"]["in"],
    }
    stop = (not r0_ok) or (not all(bars.values()))
    log(f"구창 전지 {bars} → STOP={stop}")

    # 13. 출력 --------------------------------------------------------------------
    def rnd(o):
        if isinstance(o, dict):
            return {str(k): rnd(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rnd(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return round(float(o), 5)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    res = {
        "prereg": {"v2": "d14b288", "v3": "5f1458b"}, "core_ref": "d5c1533",
        "window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())], "n_asof": A,
        "data_end": str(dates[-1].date()), "sectors": sectors,
        "signal": sig_info, "nan_to_zero_cells": nan_cells,
        "R0": {"values": r0_out, "design": R0_DESIGN, "diff": r0_diff, "match_0.005": r0_ok},
        "post_hoc_repro": post_out,
        "tails": {"CV_s": cv_s, "CV_s_raw": float(np.sqrt(cv2_raw)), "noise_var": noise,
                  "nu_u": nu_u, "kurt_u": k_u, "n_u": n_u, "nu_u_without_s": nu_u_raw,
                  "kurt_u_without_s": k_u_raw, "nu_f": nu_f, "kurt_f": k_f, "n_f": n_f,
                  "kurt_f_unstandardized": k_f_raw,
                  "spec_z_rms_median_per_stock": float(np.median(per_stock)),
                  "n_stock_spec": len(per_stock)},
        "sigma_target": {"value": sigma_target, "n": len(sig_rows),
                         "q10": float(np.quantile([r["sigma21"] for r in sig_rows], 0.1)),
                         "q90": float(np.quantile([r["sigma21"] for r in sig_rows], 0.9))},
        "OB": ob_stat,
        "B_universe_size": {"min": int(min(len(v) for v in bu.values())),
                            "max": int(max(len(v) for v in bu.values()))},
        "old_realized": realized, "old_realized_D0pred": realized_d0,
        "old_battery": {"scalar": old, "Sstar": sstar_real, "Sstar_thr": c52["Sstar_thr"],
                        "groups": grp_res, "A6_bD0_P5": a6_real, "A6_need_gt": bands52["P5"][1],
                        "bars": bars},
        "calibration": {str(T): calib[T] for T in checks},
        "T_fwd": t_fwd, "STOP": stop,
        "deferred": ["A0(신창 모델 구조의 α·밴드·탐지확률) — 신창 월말 예측이 필요하다",
                     "순차 경계 최종 배수 f_boot — B 의 2020~23 d_k 가 필요하다(PO-1.3)"],
        "elapsed_s": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C0.json").write_text(json.dumps(rnd(res), ensure_ascii=False, indent=1))
    log(f"저장 {OUT / 'C0.json'}")
    return 0


# ── C0.md ─────────────────────────────────────────────────────────────────────


def render_md(r: dict) -> str:
    """C0.json → 사람이 읽는 표. 숫자는 JSON 에서만 온다(손으로 옮기지 않는다)."""
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    L = ["# C0 구창 보정 — PEAD 최적화 사전등록 (판정 아님)", "",
         f"사전등록 2판 `{r['prereg']['v2']}` + 정정 PO-1 `{r['prereg']['v3']}`, core `{r['core_ref']}`. "
         f"구창 asof {r['window'][0]}~{r['window'][1]} ({r['n_asof']}회), 데이터 끝 {r['data_end']}.",
         "재현: `research/experiments/pead_optimizer_c0.py` 머리 주석. 숫자 원본은 `C0.json`.", "",
         "## R0 재현 (설계 §6 v3 열, 허용 0.005)", "", "| 대상 | 재현 | 설계 | 차이 |", "|---|---|---|---|"]
    for k, v in r["R0"]["values"].items():
        L.append(f"| {k} | {v:.4f} | {r['R0']['design'][k]:.3f} | {r['R0']['diff'][k]:+.4f} |")
    L += ["", f"일치: **{r['R0']['match_0.005']}**", ""]
    if "tails" not in r:
        return "\n".join(L)
    t = r["tails"]
    L += ["## 꼬리·혼합 (PO-1.1)", "", "| 양 | 값 |", "|---|---|",
          f"| CV_s (표본잡음 보정) | {t['CV_s']:.3f} |", f"| CV_s (보정 전) | {t['CV_s_raw']:.3f} |",
          f"| ν_u (z/ŝ, 초과첨도 {t['kurt_u']:.2f}, n={t['n_u']}) | {t['nu_u']:.2f} |",
          f"| ν_u 참고: ŝ 로 안 나눴을 때 (초과첨도 {t['kurt_u_without_s']:.2f}) | {t['nu_u_without_s']:.2f} |",
          f"| ν_f (S1~S6·IP_g, 대상별 표준화, 초과첨도 {t['kurt_f']:.2f}, n={t['n_f']}) | {t['nu_f']:.2f} |",
          f"| 참고: 표준화 전 초과첨도 | {t['kurt_f_unstandardized']:.2f} |",
          f"| 종목별 특이 z RMS 중앙값 (보고만, 설계 0.92) | {t['spec_z_rms_median_per_stock']:.3f} |",
          f"| 앞 수익 NaN→0 셀 수 | {r['nan_to_zero_cells']} |", "",
          "## σ_target·OB (PO-1.6)", "",
          f"σ_target(21일 사전 능동위험, 격자일 {r['sigma_target']['n']}개 중앙값) = **{r['sigma_target']['value']:.4f}**"
          f" (10~90%: {r['sigma_target']['q10']:.4f}~{r['sigma_target']['q90']:.4f}). "
          f"B 유니버스 크기 {r['B_universe_size']['min']}~{r['B_universe_size']['max']}.",
          f"OB 성공 {r['OB']['ok']}/{N_OB * r['n_asof']}, λ 실패 {r['OB']['lam_fail']}, 솔버 실패 {r['OB']['solver_fail']}, "
          f"풀이 {r['OB']['solves']}회.", "",
          "## 구창 b(D=0) (실현, D=0 예측)", "", "| 대상 | b(D=0) | b(v3) |", "|---|---|---|"]
    for k, v in r["old_realized_D0pred"].items():
        L.append(f"| {k} | {v:.3f} | {r['old_realized'][k]:.3f} |")
    c52 = r["calibration"]["52"]
    ob = r["old_battery"]
    L += ["", "## 구창 전지 (T=52, 구창 구조 밴드) — STOP 규칙", "",
          f"α = {c52['alpha']:.4f}, 맞는 모델 가족 통과 {c52['family_pass_correct']:.3f}.", "",
          "| 대상 | 실현 b | 밴드 하한 | 밴드 상한 | 중심 | 안 |", "|---|---|---|---|---|---|"]
    for k, v in ob["scalar"].items():
        L.append(f"| {k} | {v['b']:.3f} | {v['lo']:.3f} | {v['hi']:.3f} | {c52['center'][k]:.3f} | "
                 f"{'예' if v['in'] else '**아니오**'} |")
    L.append(f"| S* | {ob['Sstar']:+.3f} | | ±{ob['Sstar_thr']:.3f} | | "
             f"{'예' if abs(ob['Sstar']) <= ob['Sstar_thr'] else '**아니오**'} |")
    for name in ("IP", "IE"):
        gq = ob["groups"][name]
        L += ["", f"{name}_g: 밴드 밖 {gq['n_out']}개(허용 {int(np.floor(0.2 * len(r['sectors'])))}), "
              f"1.5배 밴드 밖 {gq['n_out1.5']}개(허용 0)", "", "| 업종 | 실현 b | 하한 | 상한 | 밖 | 1.5배 밖 |",
              "|---|---|---|---|---|---|"]
        for k, v in gq["rows"].items():
            L.append(f"| {k} | {v['b']:.3f} | {v['lo']:.3f} | {v['hi']:.3f} | {v['out']} | {v['out1.5']} |")
    L += ["", f"A-6: 구창 P5 의 D=0 예측 b = {ob['A6_bD0_P5']:.3f}, 필요 > {ob['A6_need_gt']:.3f}.", "",
          "| 바 | 구창 |", "|---|---|"]
    for k, v in ob["bars"].items():
        L.append(f"| {k} | {'통과' if v else '**미달**'} |")
    kmax = max(c52["kappa"], key=c52["kappa"].get)
    L += ["", f"**STOP = {r['STOP']}**", "",
          "> **정정 PO-2 공개(M3) — 구창 전지 통과는 정보가 없다.** 확대 계수 κ 를 같은 52개월에서 부트스트랩했으므로 "
          "가장 크게 틀린 대상이 가장 넓은 밴드를 받는다. 예: "
          f"{kmax} 실현 {r['old_realized'][kmax]:.3f} 이 T=52 밴드 [{c52['bands'][kmax][0]:.2f}, {c52['bands'][kmax][1]:.2f}]"
          f"(κ {c52['kappa'][kmax]:.2f}, T=31 κ {r['calibration']['31']['kappa'][kmax]:.2f}) 안이다. "
          "위 'STOP = False' 는 STOP 규칙에 걸리지 않았다는 뜻일 뿐, 위험모델이 맞다는 증거가 아니다. "
          "또 이 C0 는 `daily_bars_adjusted` 이음매(025620 2021-07-29 등)를 거르지 않은 값이다 — 이음매 가드 뒤 값은 `PO2.md`.", "",
          "## 구창 구조 모의 — T 별 (PO-1.1·1.2, 1,000회)", "",
          "max h(업종 제외)는 보고만 — VOID 바는 업종 포함 max h 로 본다. α 는 1,000회 모의의 순서통계량 계단"
          " (2k/999) 위에서 정해진다(T 마다 k=4 근처로 같다).", "",
          "| T | α | 가족 통과 | max h (대상) | max h 업종 제외 | D×0 | D×0.5 | D×2 | D=0 예측 | F×0.7 | F×1.5 | 업종×1.69 평균·최소 | A-6 VOID 확률 | VOID 조건 충족 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for T, c in sorted(r["calibration"].items(), key=lambda kv: int(kv[0])):
        d = c["detect"]
        nonsec = {k: v for k, v in c["h_max"].items() if not k.startswith(("IP:", "IE:"))}
        top = max(nonsec, key=nonsec.get)
        L.append(f"| {T} | {c['alpha']:.4f} | {c['family_pass_correct']:.3f} | {c['max_h']:.3f} ({c['argmax_h']}) | "
                 f"{nonsec[top]:.3f} ({top}) | "
                 + " | ".join(f"{d[s]:.2f}" for s in VOID_SET)
                 + f" | {c['detect_sector_mean']:.2f}·{c['detect_sector_min']:.2f} | {c['A6_void_prob_correct']:.2f} | "
                 + f"{'예' if c['void_ok'] else '아니오'} |")
    L += ["", f"**T_fwd = {r['T_fwd'] if r['T_fwd'] is not None else '없음 — 전진 A 판정 불가(설계 검정력 부족, PO-1.7)'}**", "",
          "> 정정 PO-2(M4): T_fwd 가 없다는 것은 SE(b) ≈ √((2+κ_f)/(4T)) 로 미리 알 수 있었다. 사전등록의 "
          "\"T_fwd ≥ 40 예상\"은 틀렸고 철회했다. 업종 ×1.69 탐지가 낮은 것은 IP/IE 묶음의 ⌊0.2G⌋개 밴드 밖 허용 때문이다. "
          "α 는 1,000회 모의의 순서통계량 위에 있다(PO-2 는 10,000회).", "",
          "## T=31 반폭과 확대 계수 (보고만 — 신창 A0 는 신창 구조로 다시 낸다)", "",
          "| 대상 | 중심 | h− | h+ | SE 모의 | SE 부트 | κ |", "|---|---|---|---|---|---|---|"]
    c31 = r["calibration"]["31"]
    for k in c31["h"]:
        L.append(f"| {k} | {c31['center'][k]:.3f} | {c31['h'][k][0]:.3f} | {c31['h'][k][1]:.3f} | "
                 f"{c31['se_sim'][k]:.3f} | {c31['se_boot'][k]:.3f} | {c31['kappa'][k]:.2f} |")
    L += ["", "## 미룬 것", ""] + [f"- {x}" for x in r["deferred"]]
    L += ["", "사후 진단 재현(보고만, 설계 §6 M2): " + ", ".join(f"{k} {v:.3f}" for k, v in r["post_hoc_repro"].items()),
          "", f"_{f3(r['elapsed_s'])}초_"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
