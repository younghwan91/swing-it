"""PEAD 최적화 정정 PO-2 — 이음매 가드 C0 다시 계산 + 신창 전 검정력 관문(구창 모의). 판정 러너가 아니다.

사전등록: ``research/logs/pead_optimizer/PREREG.md`` 정정 PO-2(4판). 이 러너는 PO-2.7-2 만 낸다.
- PO-2.1 이음매 가드(``pead_optimizer_seam.seam_guard``)를 로드 직후에 켜고, 가드 끈 R0(DB 확인)·가드 켠 R0,
  CV_s(중앙값-z² 1차·윈저화 반분 확인, 블록 부트스트랩 구간)·ν_u·ν_f·σ_target·OB(밴드 있음/없음)·구창 실현 바이어스를 낸다.
- PO-2.4 관문: 구창 52개월 구조로 T=31(주간 124) 모의 10,000회, 1차 A-OB21 탐지확률(F×1.5·업종 ×1.69 평균),
  보조 A-OB5, 밴드 업종 제약 효과, PO-2.5 전진 중단 규칙의 거짓 중단·탐지 확률.
- **2023-12-28 뒤 데이터는 읽지 않는다** — 가격은 ``load_risk_inputs(…, END)``, 원 종가·사건·정기보고서 접수일은
  SQL 에서 ≤ END, 실적은 ``read_earnings(all_versions=True, until=END)``.

숫자만 낸다(GUARDRAILS §8). 관문 결과는 사전등록 규칙의 기계적 적용이다.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@d5c1533" \\
      python research/experiments/pead_optimizer_po2.py --cache <scratch> [--workers 16]

출력: ``research/logs/pead_optimizer/PO2.json``·``PO2.md``. 캐시는 레포 밖.
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
from krx_quant_core.backtest.crosssectional import _trailing_adv
from krx_quant_core.data.risk_inputs import STYLE_NAMES, load_risk_inputs, style_panels
from krx_quant_core.risk.factor_model import RiskModel, bias_statistic, build_risk_model, factor_returns
from pead_optimizer_c0 import (
    END,
    GRID_LAST,
    GRID_START,
    HORIZON,
    LAM_HI,
    LAM_ITERS,
    LAM_LO,
    LAM_TOL,
    N_OB,
    N_P,
    N_RANDOM,
    OB_MAX_WEIGHT,
    OB_SECTOR_BAND,
    R0_DESIGN,
    RANDOM_NAMES,
    SEED,
    START,
    STEP,
    Sub,
    bias_rows,
    eligible_idx,
    exposures,
    keyed_rng,
    month_ends,
    nu_from,
)
from pead_optimizer_seam import seam_guard

T_GATE = 31
T_STOP_MONTHS = 39  # 156주
STOP_FIRST_K, STOP_RR = 26, 1.5
N_SIM = 10_000
CHUNK = 2_000
N_BOOT = 10_000
BOOT_BLOCKS = (3, 6)
CV_BOOT_BLOCK = 3
FAMILY_PASS = 0.90
DETECT_MIN = 0.8
OB_OK_MIN = 0.90
SECTOR_VOL = 1.3
BLOCKS = (5, 5, 5, 5, 1)  # 21일 = 주 4개 + 1일
WINSOR = 5.0
N_CVMED_BOOT = 200
OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"


# ── 입력 ──────────────────────────────────────────────────────────────────────


def _db():
    from swing_it.storage import connect, db_default

    con = connect(db_default())
    con.set_session(readonly=True)
    cur = con.cursor()
    cur.execute("SET statement_timeout='60s'")
    return con, cur


def yoy_signal_strict(codes, dates) -> tuple[np.ndarray, dict]:
    """PO-2.1-5 — 실적·정기보고서 접수일을 ≤ END 로 읽는다. 2019Q1 이후 분기는 END 이전 정기보고서가 없으면 뺀다.

    구현은 ``features.earnings_asof`` 로 옮겼다. ``tie_col=None`` 은 단계 B 재현에 묶인 옛 동작이다.
    """
    from swing_it.features.earnings_asof import REPRODUCE_B, load_yoy_strict

    con, _ = _db()
    try:
        return load_yoy_strict(con, codes, dates, END, **REPRODUCE_B)
    finally:
        con.close()


def raw_closes_and_events(codes, dates):
    """원 종가(``daily_bars_raw``·``supply_demand``)와 사건 날짜 — 모두 ≤ END."""
    con, cur = _db()
    lo, hi = str(dates[0].date()), END
    cidx = pd.Index([str(c) for c in codes])
    didx = pd.DatetimeIndex(dates)
    out = []
    for sql in ("SELECT code, date, close FROM daily_bars_raw WHERE date BETWEEN %s AND %s AND close > 0",
                "SELECT code, date, abs(close) FROM supply_demand WHERE date BETWEEN %s AND %s "
                "AND close IS NOT NULL AND close <> 0"):
        cur.execute(sql, (lo, hi))
        df = pd.DataFrame(cur.fetchall(), columns=["code", "date", "close"])
        a = np.full((len(codes), len(dates)), np.nan)
        i = cidx.get_indexer(df["code"].astype(str))
        t = didx.get_indexer(pd.to_datetime(df["date"]))
        ok = (i >= 0) & (t >= 0)
        a[i[ok], t[ok]] = df["close"].to_numpy(float)[ok]
        out.append(a)
    cur.execute("SELECT code, date FROM corporate_actions WHERE date BETWEEN %s AND %s", (lo, hi))
    ev = pd.DataFrame(cur.fetchall(), columns=["code", "date"])
    con.close()
    return out, ev


# ── 위험모델(병렬) ────────────────────────────────────────────────────────────

_CTX: dict = {}


def _build(a):
    c = _CTX
    m = build_risk_model(c["R"], c["Xs"], c["lab"], c["cap"], c["msk"], a, codes=np.arange(c["R"].shape[0]))
    return a, (m.codes.astype(int), m.X, m.F, m.D, list(m.names))


def prep(x):
    sector = x.sector
    has_sec = np.vectorize(lambda s: s is not None, otypes=[bool])(sector)
    mask = has_sec & np.isfinite(x.cap) & (x.cap > 0) & np.isfinite(x.close)
    raw = style_panels(x)
    Xall = exposures(x, raw, mask)
    first = int(np.argmax(has_sec.any(axis=0)))
    sl = slice(first, None)
    c = {"R": x.ret[:, sl], "cap": x.cap[:, sl], "msk": mask[:, sl], "lab": sector[:, sl],
         "Xs": {k: v[:, sl] for k, v in Xall.items()}, "dates": x.dates[sl], "first": first, "raw": raw}
    c["asofs"] = month_ends(c["dates"], START, c["R"].shape[1] - 1)
    c["sectors"] = sorted({s for s in c["lab"][:, -1] if s is not None})
    return c


def build_models(c, need, workers, path: Path, log):
    if path.exists():
        return pickle.load(open(path, "rb"))
    _CTX.clear()
    _CTX.update(c)
    with mp.get_context("fork").Pool(workers) as pool:
        models = dict(pool.imap_unordered(_build, need, chunksize=2))
    _CTX.clear()
    pickle.dump(models, open(path, "wb"))
    log(f"  모델 {len(models)}개 → {path.name}")
    return models


def as_model(models, a):
    idx, X, F, D, names = models[a]
    return RiskModel(codes=idx, names=names, X=X, F=F, D=D), idx


def r0_values(c, models):
    """C0·validate_risk_model.py 와 같은 R0(같은 난수열)."""
    rng = np.random.default_rng(SEED)
    R, cap = c["R"], c["cap"]
    r0 = {"random": [], "capw": [], **{f"style:{k}": [] for k in STYLE_NAMES}}
    for a in c["asofs"]:
        m, idx = as_model(models, a)
        nA = len(idx)
        fwd = np.nan_to_num(R[idx, a + 1:a + 1 + HORIZON], nan=0.0).sum(axis=1)

        def z(w, fwd=fwd, m=m):
            return float(w @ fwd), float(np.sqrt(HORIZON * m.var(w)))

        rnd = []
        for _ in range(N_RANDOM):
            w = np.zeros(nA)
            w[rng.choice(nA, RANDOM_NAMES, replace=False)] = 1.0 / RANDOM_NAMES
            rnd.append(z(w))
        r0["random"].append(rnd)
        r0["capw"].append(z(cap[idx, a] / cap[idx, a].sum()))
        Wr = np.sqrt(cap[idx, a])
        P = np.linalg.pinv((m.X * Wr[:, None]).T @ m.X) @ (m.X * Wr[:, None]).T
        for k in STYLE_NAMES:
            r0[f"style:{k}"].append(z(P[m.names.index(k)]))
    rr = np.array(r0["random"])
    out = {"random": bias_statistic(rr[:, :, 1].T, rr[:, :, 0].T)["mean"]}
    for key in ["capw", *[f"style:{k}" for k in STYLE_NAMES]]:
        arr = np.array(r0[key])
        out[key] = bias_statistic(arr[:, 1], arr[:, 0])["mean"]
    diff = {k: out[k] - R0_DESIGN[k] for k in R0_DESIGN}
    return {"values": out, "diff": diff, "max_abs_diff": max(abs(v) for v in diff.values()),
            "match_0.005": all(abs(v) <= 0.005 for v in diff.values())}


# ── OB(PO-1.6, 밴드 인자) ─────────────────────────────────────────────────────


def _ob_task(args):
    from krx_quant_core.portfolio.optimize import Constraints, optimize

    n, X, F, D, names, sect, sigma_t, band = args
    nb = len(D)
    sub = Sub(X=X, F=21.0 * F, D=21.0 * D, names=names)
    b = np.full(nb, 1.0 / nb)
    Sb = sub.X @ (sub.F @ (sub.X.T @ b)) + sub.D * b
    con = Constraints(budget=1.0, long_only=True, max_weight=OB_MAX_WEIGHT,
                      sector=sect if band is not None else None, benchmark=b, sector_band=band)
    rng = keyed_rng("OB", n)  # C0 와 같은 α 난수 — 밴드 있음/없음이 같은 α 를 쓴다
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


# ── 모의(PO-2.4) ──────────────────────────────────────────────────────────────

_SIM: dict = {}


def _sim_task(args):
    """시나리오 하나 × 반복 10,000. 공통 난수 keyed_rng("po2-sim", 달, 묶음)."""
    variant, sc = args
    S = _SIM["struct"][variant]
    nu_f, nu_u = _SIM["nu_f"], _SIM["nu_u"]
    sig_ln = _SIM["sig_ln"][variant]
    A = len(S)
    o = np.arange(N_SIM) % A
    S1, S2, NN, W1, W2, WN = (np.zeros((N_OB, N_SIM)) for _ in range(6))
    zstop = np.full((N_SIM, 4 * T_STOP_MONTHS), np.nan)
    pstop = np.arange(N_SIM) % N_OB
    fscale, dscale, sec_j = 1.0, 1.0, None
    if sc.startswith("Fx"):
        fscale = np.sqrt(float(sc[2:]))
    elif sc.startswith("Dx"):
        dscale = np.sqrt(float(sc[2:]))
    elif sc.startswith("sec:"):
        sec_j = sc[4:]
    for m in range(A):
        st = S[m]
        Y, Wd, good, p1 = st["Y"], st["Wd"], st["good"], st["pred1"]
        K, nb = Y.shape[1], Wd.shape[1]
        ev, evec = np.linalg.eigh(0.5 * (st["F"] + st["F"].T))
        Gt = evec * np.sqrt(np.clip(ev, 0, None))[None, :]
        jj = st["sec_pos"].get(sec_j, -1) if sec_j else -1
        for ch in range(N_SIM // CHUNK):
            rr = np.arange(ch * CHUNK, (ch + 1) * CHUNK)
            tpos = (m - o[rr]) % A
            rng = keyed_rng("po2-sim", m, ch)
            g = rng.standard_normal((len(BLOCKS), K, CHUNK))
            chi_f = rng.chisquare(nu_f, CHUNK)
            e = rng.standard_normal((len(BLOCKS), nb, CHUNK))
            chi_u = rng.chisquare(nu_u, (nb, CHUNK))
            ns = rng.standard_normal(CHUNK)
            s = np.sqrt(np.exp(sig_ln * ns - 0.5 * sig_ln ** 2))
            tf = np.sqrt((nu_f - 2.0) / chi_f)[None, :]
            tu = np.sqrt((nu_u - 2.0) / chi_u)
            r21 = np.zeros((N_OB, CHUNK))
            wk = []
            for k, ell in enumerate(BLOCKS):
                f = (Gt @ g[k]) * tf * np.sqrt(ell) * fscale
                if jj >= 0:
                    f[jj] *= SECTOR_VOL
                rk = Y @ f + (Wd @ (e[k] * tu)) * (np.sqrt(ell) * dscale) * s[None, :]
                r21 += rk
                if k < 4:
                    wk.append(rk / (np.sqrt(5.0) * p1)[:, None])
            z21 = r21 / (np.sqrt(21.0) * p1)[:, None]
            use = (tpos < T_GATE)[None, :] & good[:, None]
            z21 = np.where(use, z21, 0.0)
            S1[:, rr] += z21
            S2[:, rr] += z21 * z21
            NN[:, rr] += use
            for zk in wk:
                zk = np.where(use, zk, 0.0)
                W1[:, rr] += zk
                W2[:, rr] += zk * zk
                WN[:, rr] += use
            ss = tpos < T_STOP_MONTHS
            if ss.any():
                loc = np.flatnonzero(ss)
                pp = pstop[rr[loc]]
                okp = good[pp]
                for k in range(4):
                    v = wk[k][pp, loc]
                    zstop[rr[loc], 4 * tpos[loc] + k] = np.where(okp, v, np.nan)

    def bstat(s1, s2, nn):
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.sqrt(np.maximum((s2 - s1 ** 2 / nn) / (nn - 1.0), 0.0))
        sd[nn < 2] = np.nan
        return np.nanmean(sd, axis=0)

    # PO-2.5 중단 규칙(누적 sd, K ≥ 26, > 1.5)
    stop_week = np.full(N_SIM, -1)
    c1, c2, cn = np.zeros(N_SIM), np.zeros(N_SIM), np.zeros(N_SIM)
    for w in range(zstop.shape[1]):
        v = zstop[:, w]
        f = np.isfinite(v)
        c1[f] += v[f]
        c2[f] += v[f] ** 2
        cn[f] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            rr_k = np.sqrt(np.maximum((c2 - c1 ** 2 / cn) / (cn - 1.0), 0.0))
        hit = (cn >= STOP_FIRST_K) & (rr_k > STOP_RR) & (stop_week < 0) & f
        stop_week[hit] = w + 1
    return variant, sc, bstat(S1, S2, NN), bstat(W1, W2, WN), stop_week


# ── 통계 도우미 ───────────────────────────────────────────────────────────────


def circ_idx(rng, n_obs, T, L, B):
    nb = -(-T // L)
    starts = rng.integers(0, n_obs, size=(B, nb))
    return ((starts[:, :, None] + np.arange(L)[None, None, :]) % n_obs).reshape(B, -1)[:, :T]


def boot_se_ob(Z21, Z5, T):
    """PO-1.1-5 — 원형 블록 부트스트랩(달 단위, 주간은 그 달 4주를 함께) SE, L ∈ {3,6} 최대."""
    A = Z21.shape[1]
    out = {"OB21": 0.0, "OB5": 0.0}
    for L in BOOT_BLOCKS:
        rng = keyed_rng("po2-boot", T, L)
        idx = circ_idx(rng, A, T, L, N_BOOT)
        b21, b5 = [], []
        for lo in range(0, N_BOOT, 500):
            ii = idx[lo:lo + 500]
            G = Z21[:, ii]
            with np.errstate(invalid="ignore"):
                b21.append(np.nanmean(np.nanstd(G, axis=2, ddof=1), axis=0))
                G5 = Z5[:, ii, :].reshape(Z5.shape[0], ii.shape[0], -1)
                b5.append(np.nanmean(np.nanstd(G5, axis=2, ddof=1), axis=0))
        out["OB21"] = max(out["OB21"], float(np.std(np.concatenate(b21), ddof=1)))
        out["OB5"] = max(out["OB5"], float(np.std(np.concatenate(b5), ddof=1)))
    return out


def cv_estimates(spec_z):
    """PO-2.2 — 중앙값-z² CV(1차)·윈저화 반분 공분산 CV(확인), 각각 달 블록 부트스트랩 구간."""
    A = len(spec_z)
    med = np.array([np.median(z * z) for z in spec_z])
    vboot = np.empty(A)
    for a, z in enumerate(spec_z):
        rng = keyed_rng("cvmed", a)
        z2 = z * z
        bs = [np.median(z2[rng.integers(0, z2.size, z2.size)]) for _ in range(N_CVMED_BOOT)]
        vboot[a] = np.var(bs, ddof=1)
    h1, h2 = np.empty(A), np.empty(A)
    for a, z in enumerate(spec_z):
        rng = keyed_rng("split", a)
        perm = rng.permutation(z.size)
        zc = np.clip(z, -WINSOR, WINSOR) ** 2
        h1[a] = zc[perm[: z.size // 2]].mean()
        h2[a] = zc[perm[z.size // 2:]].mean()

    def cv_med(ix):
        m = med[ix]
        s2 = m / m.mean()
        v = np.var(s2, ddof=1) - np.mean(vboot[ix]) / m.mean() ** 2
        return float(np.sqrt(max(v, 0.0)))

    def cv_split(ix):
        a1, a2 = h1[ix] / h1[ix].mean(), h2[ix] / h2[ix].mean()
        v = np.cov(a1, a2, ddof=1)[0, 1]
        return float(np.sqrt(max(v, 0.0)))

    full = np.arange(A)
    rng = keyed_rng("cvboot")
    idx = circ_idx(rng, A, A, CV_BOOT_BLOCK, N_BOOT)
    bm = np.array([cv_med(i) for i in idx])
    bs = np.array([cv_split(i) for i in idx])
    q = lambda v: [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])]  # noqa: E731
    s_med = med / med.mean()
    raw_mean = np.array([np.mean(z * z) for z in spec_z])
    return {
        "CV_med": cv_med(full), "CV_med_q05_50_95": q(bm), "CV_med_share_zero": float(np.mean(bm == 0)),
        "CV_split": cv_split(full), "CV_split_q05_50_95": q(bs), "CV_split_share_zero": float(np.mean(bs == 0)),
        "CV_mean_raw_uncorrected": float(np.std(raw_mean / raw_mean.mean(), ddof=1)),
        "noise_share_med": float(np.mean(vboot) / med.mean() ** 2),
    }, s_med


# ── 본체 ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="레포 밖 캐시 디렉터리")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--render", action="store_true", help="PO2.json → PO2.md 만")
    args = ap.parse_args()
    if args.render:
        (OUT / "PO2.md").write_text(render_md(json.loads((OUT / "PO2.json").read_text())))
        return 0
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    assert repo not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    # 1. 입력 + 이음매 가드 -------------------------------------------------------
    x0 = load_risk_inputs("2019-01-02", END)
    assert x0.dates[-1] <= pd.Timestamp(END), "END 뒤 날짜가 들어왔다"
    raw_src, events = raw_closes_and_events(x0.codes, x0.dates)
    rf, cf, seam, seam_info = seam_guard(x0.codes, x0.dates, x0.ret, x0.close, raw_src, events)
    del raw_src
    x = dataclasses.replace(x0, ret=rf, close=cf)
    log(f"이음매 {seam_info['seam_cells']}셀/{seam_info['seam_codes']}종목, 사건 설명 "
        f"{seam_info['cells_big_dev_explained_kept']}, 판정 불가 {seam_info['cells_undeterminable']}")

    # 2. 가드 끈 R0(DB 확인) ------------------------------------------------------
    c_raw = prep(x0)
    m_raw = build_models(c_raw, c_raw["asofs"], args.workers, cache / "models_noguard.pkl", log)
    r0_noguard = r0_values(c_raw, m_raw)
    log(f"R0 가드 끔 최대 차 {r0_noguard['max_abs_diff']:.4f}")
    del c_raw, m_raw
    if not r0_noguard["match_0.005"]:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "PO2.json").write_text(json.dumps({"R0_noguard": r0_noguard, "STOP": "R0 불일치 — DB 변경"},
                                                 ensure_ascii=False, indent=1, default=float))
        return 1

    # 3. 가드 켠 입력·모델 --------------------------------------------------------
    c = prep(x)
    R, cap, lab, dates, asofs, sectors, first = (c["R"], c["cap"], c["lab"], c["dates"], c["asofs"],
                                                 c["sectors"], c["first"])
    seam_s = seam[:, first:]
    nD = len(dates)
    sig_all, sig_info = yoy_signal_strict(x.codes, x.dates)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    sig, adv = sig_all[:, first:], adv_all[:, first:]
    t_0 = int(np.searchsorted(dates, np.datetime64(GRID_START)))
    grid = [t for t in range(t_0, nD, STEP) if dates[t] <= pd.Timestamp(GRID_LAST)]
    models = build_models(c, sorted(set(asofs) | set(grid)), args.workers, cache / "models_guard.pkl", log)
    r0_guard = r0_values(c, models)
    log(f"R0 가드 켬 최대 차 {r0_guard['max_abs_diff']:.4f}")

    # 4. σ_target --------------------------------------------------------------
    tranche = {}
    for t in grid:
        _, idx = as_model(models, t)
        in_u = np.zeros(R.shape[0], bool)
        in_u[idx] = True
        el = eligible_idx(t, sig, adv, cap, in_u)
        top = el[np.argsort(-sig[el, t], kind="stable")[:10]] if el.size >= 20 else None
        tranche[t] = (el, top)
    sig_rows = []
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
        sig_rows.append(float(np.sqrt(HORIZON * m.var(w - b))))
    sigma_target = float(np.median(sig_rows))
    log(f"σ_target {sigma_target:.4f}({len(sig_rows)})")

    # 5. OB(밴드 0.10 / 없음) ---------------------------------------------------
    bu = {}
    for n, a in enumerate(asofs):
        _, idx = as_model(models, a)
        in_u = np.zeros(R.shape[0], bool)
        in_u[idx] = True
        el = eligible_idx(a, sig, adv, cap, in_u)
        pos = {cc: i for i, cc in enumerate(idx)}
        bu[n] = np.array([pos[cc] for cc in el], dtype=int)
    ob, ob_stat = {}, {}
    for variant, band in (("band", OB_SECTOR_BAND), ("noband", None)):
        opath = cache / f"ob_{variant}_{sigma_target:.6f}.pkl"
        if opath.exists():
            ob[variant] = pickle.load(open(opath, "rb"))
        else:
            tasks = []
            for n, a in enumerate(asofs):
                m, idx = as_model(models, a)
                s = bu[n]
                tasks.append((n, m.X[s], m.F, m.D[s], m.names, lab[idx[s], a].astype(object), sigma_target, band))
            res = {}
            with mp.get_context("fork").Pool(args.workers) as pool:
                for n, W, st in pool.imap_unordered(_ob_task, tasks):
                    res[n] = (W, st)
            pickle.dump(res, open(opath, "wb"))
            ob[variant] = res
        ob_stat[variant] = {k: int(sum(ob[variant][n][1][k] for n in ob[variant]))
                            for k in ("ok", "lam_fail", "solver_fail", "solves")}
        log(f"OB {variant} {ob_stat[variant]}")

    # 6. 잔차(실현 특이 z) --------------------------------------------------------
    upath = cache / "resid_guard.npy"
    if upath.exists():
        U = np.load(upath)
    else:
        U = factor_returns(R, c["Xs"], lab, cap, c["msk"]).u
        np.save(upath, U)

    # 7. 월말 루프 — 실현(보고만) + OB 모의 구조 ----------------------------------
    rng_t = {k: keyed_rng(k) for k in ("P5", "P50", "P10B")}
    names_t = ([f"S:{k}" for k in STYLE_NAMES] + ["P5", "P50", "P10B", "HV", "SM"]
               + [f"IP:{g}" for g in sectors] + [f"IE:{g}" for g in sectors])
    Zr = {k: [] for k in names_t}
    A = len(asofs)
    Z21 = {v: np.full((N_OB, A), np.nan) for v in ob}
    Z5 = {v: np.full((N_OB, A, 4), np.nan) for v in ob}
    struct = {v: [] for v in ob}
    expo = {v: {"max_abs_sector_active": [], "sector_var_share": [], "factor_var_share": []} for v in ob}
    spec_z = []
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
        wk = np.stack([fd[:, 5 * k:5 * k + 5].sum(axis=1) for k in range(4)], axis=1)  # (nA, 4)
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
        for v in ob:
            Wob = ob[v][n][0]  # (N_OB, nb) 능동 비중(B 유니버스 위)
            good = np.isfinite(Wob).all(axis=1)
            Wg = np.where(good[:, None], Wob, 0.0)
            Y = Wg @ Xs_
            fv = np.einsum("pk,kl,pl->p", Y, m.F, Y)
            sv = (Wg * Wg) @ Ds_
            p1 = np.sqrt(fv + sv)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z21[v][good, n] = (Wg @ fwd[s])[good] / (np.sqrt(21.0) * p1[good])
                Z5[v][good, n, :] = (Wg @ wk[s])[good] / (np.sqrt(5.0) * p1[good])[:, None]
            Ys = Y[:, sec_idx]
            Fs = m.F[np.ix_(sec_idx, sec_idx)]
            with np.errstate(invalid="ignore", divide="ignore"):
                expo[v]["max_abs_sector_active"].append(np.abs(Ys[good]).max(axis=1))
                expo[v]["sector_var_share"].append((np.einsum("pk,kl,pl->p", Ys, Fs, Ys) / p1 ** 2)[good])
                expo[v]["factor_var_share"].append((fv / p1 ** 2)[good])
            struct[v].append({"Y": Wg @ Xs_, "Wd": Wg * np.sqrt(Ds_)[None, :], "F": m.F, "good": good,
                              "pred1": np.where(good, p1, 1.0),
                              "sec_pos": {g: m.names.index(g) for g in sectors if g in m.names}})
        u = U[idx, a + 1:a + 1 + HORIZON]
        okr = np.isfinite(u).all(axis=1)
        spec_z.append(u[okr].sum(axis=1) / np.sqrt(HORIZON * m.D[okr]))
    log(f"월말 {A}개, NaN→0 셀 이음매 {nan_seam}·그 밖 {nan_other}")

    # 8. 꼬리·혼합 ----------------------------------------------------------------
    cv, s_med = cv_estimates(spec_z)
    cv_s = cv["CV_med"]
    nu_u, k_u, n_u = nu_from(np.concatenate([z / np.sqrt(sv) for z, sv in zip(spec_z, s_med)]))
    nu_u_raw, k_u_raw, _ = nu_from(np.concatenate(spec_z))
    fac_t = [f"S:{k}" for k in STYLE_NAMES] + [f"IP:{g}" for g in sectors]
    pooled = []
    for t in fac_t:
        zz = np.concatenate([np.atleast_1d(v) for v in Zr[t]])
        zz = zz[np.isfinite(zz)]
        pooled.append(zz / zz.std(ddof=1))
    nu_f, k_f, n_f = nu_from(np.concatenate(pooled))
    log(f"CV_med {cv_s:.3f} {cv['CV_med_q05_50_95']} split {cv['CV_split']:.3f} ν_u {nu_u:.2f}(κ {k_u:.2f}) "
        f"ν_f {nu_f:.2f}(κ {k_f:.2f})")
    realized = {t: bias_rows(np.array([np.atleast_1d(v) for v in Zr[t]], dtype=float).T) for t in names_t}
    for v in ob:
        realized[f"OB21:{v}"] = bias_rows(Z21[v])
        realized[f"OB5:{v}"] = bias_rows(Z5[v].reshape(N_OB, -1))

    # 9. 모의 ----------------------------------------------------------------------
    scen = ["correct", "Fx1.5", "Fx0.7", "Dx0.5", "Dx2"] + [f"sec:{g}" for g in sectors]
    sig_ln = float(np.sqrt(np.log(1.0 + cv_s ** 2)))
    cv_hi = cv["CV_med_q05_50_95"][2]
    _SIM.clear()
    _SIM.update({"struct": {"band": struct["band"], "noband": struct["noband"], "cvhi": struct["band"]},
                 "nu_f": nu_f, "nu_u": nu_u,
                 "sig_ln": {"band": sig_ln, "noband": sig_ln, "cvhi": float(np.sqrt(np.log(1.0 + cv_hi ** 2)))}})
    tasks = ([("band", sc) for sc in scen] + [("noband", sc) for sc in ["correct", "Fx1.5"] + scen[5:]]
             + [("cvhi", sc) for sc in ["correct", "Fx1.5"] + scen[5:]])
    sims = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for variant, sc, b21, b5, stopw in pool.imap_unordered(_sim_task, tasks):
            sims[(variant, sc)] = (b21, b5, stopw)
            log(f"  모의 {variant}:{sc} b21 중앙 {np.nanmedian(b21):.3f} b5 {np.nanmedian(b5):.3f}")
    _SIM.clear()

    # 10. 밴드·탐지·관문 ----------------------------------------------------------
    se_boot = boot_se_ob(Z21["band"], Z5["band"], T_GATE)

    def calibrate(variant, kappa_from_boot=True):
        b21c, b5c, _ = sims[(variant, "correct")]
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
            p = np.mean(inside(b21c, b1) & inside(b5c, b2))
            if p > FAMILY_PASS:
                lo_a = mid
            else:
                hi_a = mid
        al = lo_a
        if kappa_from_boot:
            k21, k5 = max(1.0, se_boot["OB21"] / se21), max(1.0, se_boot["OB5"] / se5)
        else:
            k21 = k5 = 1.0
        B21, B5 = bands(al, k21, k5)
        det = {}
        for (vv, sc), (b21, b5, stopw) in sims.items():
            if vv != variant:
                continue
            o21 = ~inside(b21, B21)
            o5 = ~inside(b5, B5)
            det[sc] = {"primary": float(np.mean(o21)), "secondary": float(np.mean(o5)),
                       "family": float(np.mean(o21 | o5)), "b21_median": float(np.median(b21)),
                       "b5_median": float(np.median(b5)),
                       "stop_prob": float(np.mean(stopw > 0)),
                       "stop_week_median": float(np.median(stopw[stopw > 0])) if (stopw > 0).any() else None}
        secs = [det[f"sec:{g}"]["primary"] for g in sectors if f"sec:{g}" in det]
        return {"alpha": al, "center": {"OB21": c21, "OB5": c5}, "se_sim": {"OB21": se21, "OB5": se5},
                "kappa": {"OB21": k21, "OB5": k5}, "band": {"OB21": B21, "OB5": B5},
                "h": {"OB21": [c21 - B21[0], B21[1] - c21], "OB5": [c5 - B5[0], B5[1] - c5]},
                "h_primary": max(c21 - B21[0], B21[1] - c21),
                "family_pass_correct_final": 1.0 - det["correct"]["family"],
                "detect": det, "detect_sector_mean": float(np.mean(secs)), "detect_sector_min": float(np.min(secs)),
                "detect_sector_argmin": sectors[int(np.argmin(secs))]}

    cal = {"band": calibrate("band"), "cvhi": calibrate("cvhi"), "noband": calibrate("noband", False)}
    cb = cal["band"]
    ob_ok_share = ob_stat["band"]["ok"] / (N_OB * A)
    gate = {
        "detect_F1.5": cb["detect"]["Fx1.5"]["primary"],
        "detect_sector_mean": cb["detect_sector_mean"],
        "ob_ok_share": ob_ok_share,
        "rule": "탐지(F×1.5) ≥ 0.8 ∧ 탐지(업종 ×1.69 평균) ≥ 0.8 ∧ OB 성공 ≥ 90%",
    }
    gate["open_new_window"] = bool(gate["detect_F1.5"] >= DETECT_MIN and gate["detect_sector_mean"] >= DETECT_MIN
                                   and ob_ok_share >= OB_OK_MIN)
    gate["decision"] = ("신창 1회 사용 — A0 → A1(A-OB21·A-OB5)" if gate["open_new_window"] else
                        "단계 A 생략(검정력 부족) — 신창 미사용, B 는 '위험모델 미검증', 전진 O 팔 주간 RR_K>1.5 중단")
    log(f"관문 {gate}")

    # 11. 출력 --------------------------------------------------------------------
    def q(v):
        v = np.concatenate([np.atleast_1d(x_) for x_ in v])
        return [float(x_) for x_ in np.quantile(v, [0.1, 0.5, 0.9])]

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

    res = {
        "prereg": {"PO-1": "5f1458b", "C0": "53bd09e", "PO-2": "f34a9c4"}, "core_ref": "d5c1533",
        "window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())], "n_asof": A,
        "data_end": str(dates[-1].date()), "sectors": sectors,
        "seam": seam_info, "nan_to_zero_cells": {"seam": nan_seam, "other": nan_other},
        "signal": sig_info,
        "R0_noguard": r0_noguard, "R0_guard": r0_guard, "R0_design": R0_DESIGN,
        "tails": {**cv, "nu_u": nu_u, "kurt_u": k_u, "n_u": n_u, "nu_u_without_s": nu_u_raw,
                  "kurt_u_without_s": k_u_raw, "nu_f": nu_f, "kurt_f": k_f, "n_f": n_f},
        "sigma_target": {"value": sigma_target, "n": len(sig_rows),
                         "q10": float(np.quantile(sig_rows, 0.1)), "q90": float(np.quantile(sig_rows, 0.9))},
        "OB": ob_stat, "B_universe_size": [int(min(len(v) for v in bu.values())),
                                           int(max(len(v) for v in bu.values()))],
        "old_realized_report_only": realized,
        "exposure": {v: {k: q(vals) for k, vals in d.items()} for v, d in expo.items()},
        "se_boot_T31": se_boot,
        "calibration": cal, "gate": gate,
        "sim": {"N_SIM": N_SIM, "T_gate": T_GATE, "T_stop_months": T_STOP_MONTHS, "blocks": list(BLOCKS),
                "N_BOOT": N_BOOT},
        "elapsed_s": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PO2.json").write_text(json.dumps(rnd(res), ensure_ascii=False, indent=1))
    (OUT / "PO2.md").write_text(render_md(json.loads((OUT / "PO2.json").read_text())))
    log(f"저장 {OUT / 'PO2.json'}")
    return 0


# ── PO2.md ────────────────────────────────────────────────────────────────────


def render_md(r: dict) -> str:
    """PO2.json → 표. 숫자는 JSON 에서만 온다."""
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    s = r["seam"]
    t = r["tails"]
    cb = r["calibration"]["band"]
    g = r["gate"]
    L = ["# 정정 PO-2 — 이음매 가드 C0 다시 계산 + 신창 전 검정력 관문 (검증 전)", "",
         f"사전등록 PO-2 `{r['prereg']['PO-2']}`(PO-1 `{r['prereg']['PO-1']}`, C0 `{r['prereg']['C0']}`), core "
         f"`{r['core_ref']}`. 구창 asof {r['window'][0]}~{r['window'][1]} ({r['n_asof']}회), 데이터 끝 {r['data_end']}. "
         "**2023-12-28 뒤 데이터는 읽지 않았다.** 재현: `research/experiments/pead_optimizer_po2.py` 머리 주석.", "",
         "## 관문 결과 (PO-2.4 규칙의 기계적 적용)", "",
         "| 양 | 값 | 바 |", "|---|---|---|",
         f"| 1차 A-OB21 탐지확률 — F×1.5 | {g['detect_F1.5']:.3f} | ≥ 0.8 |",
         f"| 1차 A-OB21 탐지확률 — 업종 ×1.69 (10개 평균) | {g['detect_sector_mean']:.3f} | ≥ 0.8 |",
         f"| 구창 OB 성공 비율 | {g['ob_ok_share']:.3f} | ≥ 0.90 |",
         f"| 1차 밴드 반폭 h (T=31) | {cb['h_primary']:.3f} | (보고) |", "",
         f"**결정: {g['decision']}**", "",
         "> **정정 PO-3(리뷰 반영, 재계산 없음).**",
         "> - **M1.** F·업종 오차는 O 사전 능동분산의 약 6%(구창 사전값)라 B 위험에 미치는 영향이 작다. A 생략은 위험모델을 "
         "O 에 대해 검증할 수 없어서가 아니라, 관문이 O 에 무관한 오차를 물었기 때문이다. 관문 실패는 C0 OB 비중으로 예측 가능했고, "
         "그 실패 가지가 B 를 여는 조건을 완화한다 — PO-3 이 그 가지를 철회했다(B 는 다시 A 통과 때만).",
         f"> - **M2.** v3 는 표본 안에서 최적화 포트의 주간 위험을 약 10% 과소예측한다. 구창 실현 OB5 "
         f"{r['old_realized_report_only']['OB5:band']:.3f} 은 밴드 상단 {cb['band']['OB5'][1]:.3f} 위다(OB21 "
         f"{r['old_realized_report_only']['OB21:band']:.3f}, 중심 {cb['center']['OB21']:.3f}). 앞 21일 NaN→0 셀 "
         f"{r['nan_to_zero_cells']['seam'] + r['nan_to_zero_cells']['other']}개가 실현값을 아래로 끄는데도 그렇다. "
         "최적화기가 이용하는 방향이다(OB 사전 능동분산의 약 96% 가 D).",
         "> - **M3.** 아래 이음매 가드의 `supply_demand` 근거 셀 일부는 정지 중 멈춘 종가 때문에 생긴 오판이다"
         "(221610 −13.6% 등). PO-3.3 이 규칙을 고쳤다. 이 표의 숫자는 옛 규칙 그대로다.", "",
         "## 이음매 가드 (PO-2.1)", "",
         "| 양 | 값 |", "|---|---|",
         f"| 이음매 셀(NaN 처리) | {s['seam_cells']} ({s['seam_codes']}종목) |",
         f"| 비 어긋남 > 0.2 이지만 사건이 설명해 남긴 셀 | {s['cells_big_dev_explained_kept']} |",
         f"| 판정 가능 / 판정 불가 셀 | {s['cells_determinable']} / {s['cells_undeterminable']} |",
         f"| 이음매 중 수정 수익 절댓값 > 0.3 / 원 수익 절댓값 > 0.3 | {s['seam_cells_adj_abs_ret_gt_0.3']} / "
         f"{s['seam_cells_raw_abs_ret_gt_0.3']} |",
         f"| 판정 불가 중 수익 절댓값 > 0.3 | {s['cells_undeterminable_abs_ret_gt_0.3']} |",
         f"| 월말 앞 21일 NaN→0 셀 — 이음매 / 그 밖 | {r['nan_to_zero_cells']['seam']} / {r['nan_to_zero_cells']['other']} |",
         "", "이음매 날짜 상위: " + ", ".join(f"{d} {n}" for d, n in s["seam_top_dates"]),
         "", "이음매 종목 상위: " + ", ".join(f"{d} {n}" for d, n in s["seam_top_codes"]), "",
         "| 종목 | 날짜 | 수정 수익 | 원 수익 |", "|---|---|---|---|"]
    for cc in s["seam_cells_largest"][:12]:
        L.append(f"| {cc['code']} | {cc['date']} | {cc['ret_adj']:+.3f} | {cc['ret_raw']:+.3f} |")
    L += ["", f"실적(PO-2.1-5): END 이하로 읽은 행 {r['signal']['earnings_rows_read_le_end']}, 2019Q1 이후 정기보고서 없음으로 뺀 행 "
          f"{r['signal']['dropped_2019Q1plus_no_periodic_le_end']}, 쓴 행 {r['signal']['rows_used']}.", "",
          "## R0 (설계 §6 v3 열, 허용 0.005)", "", "| 대상 | 설계 | 가드 끔 | 가드 켬 |", "|---|---|---|---|"]
    for k, v in r["R0_design"].items():
        L.append(f"| {k} | {v:.3f} | {r['R0_noguard']['values'][k]:.4f} | {r['R0_guard']['values'][k]:.4f} |")
    L += ["", f"가드 끔 최대 차 {r['R0_noguard']['max_abs_diff']:.4f}(일치 {r['R0_noguard']['match_0.005']}), "
          f"가드 켬 최대 차 {r['R0_guard']['max_abs_diff']:.4f}(보고만).", "",
          "## 꼬리·혼합 (PO-2.2)", "", "| 양 | 점추정 | 5% | 50% | 95% | 재표집 0 비율 |", "|---|---|---|---|---|---|",
          f"| CV_s 중앙값-z² (1차) | {t['CV_med']:.3f} | " + " | ".join(f"{v:.3f}" for v in t["CV_med_q05_50_95"])
          + f" | {t['CV_med_share_zero']:.3f} |",
          f"| CV_s 윈저화 반분 (확인) | {t['CV_split']:.3f} | " + " | ".join(f"{v:.3f}" for v in t["CV_split_q05_50_95"])
          + f" | {t['CV_split_share_zero']:.3f} |", "",
          f"보고만: 평균-z² 보정 전 {t['CV_mean_raw_uncorrected']:.3f}, 중앙값 표본잡음 몫 {t['noise_share_med']:.4f}. "
          f"ν_u {t['nu_u']:.2f}(z/ŝ 초과첨도 {t['kurt_u']:.2f}, n={t['n_u']}; ŝ 없이 {t['kurt_u_without_s']:.2f}), "
          f"ν_f {t['nu_f']:.2f}(초과첨도 {t['kurt_f']:.2f}, n={t['n_f']}).", "",
          f"σ_target = **{r['sigma_target']['value']:.4f}** (격자일 {r['sigma_target']['n']}, 10~90% "
          f"{r['sigma_target']['q10']:.4f}~{r['sigma_target']['q90']:.4f}), B 유니버스 {r['B_universe_size'][0]}~"
          f"{r['B_universe_size'][1]}.", ""]
    for v, st in r["OB"].items():
        L.append(f"OB({v}): 성공 {st['ok']}/{200 * r['n_asof']}, λ 실패 {st['lam_fail']}, 솔버 실패 {st['solver_fail']}.")
    L += ["", f"## 모의 (T=31·주간 124, {r['sim']['N_SIM']:,}회, 블록 {r['sim']['blocks']}, 공통 난수)", "",
          f"α = {cb['alpha']:.4f}(맞는 모델에서 두 검정 동시 통과 0.90, κ=1), κ 뒤 동시 통과 {cb['family_pass_correct_final']:.3f}.", "",
          "| 검정 | 중심 | 밴드 | h− | h+ | SE 모의 | SE 부트 | κ |", "|---|---|---|---|---|---|---|---|"]
    for k in ("OB21", "OB5"):
        L.append(f"| {k} | {cb['center'][k]:.3f} | [{cb['band'][k][0]:.3f}, {cb['band'][k][1]:.3f}] | {cb['h'][k][0]:.3f} | "
                 f"{cb['h'][k][1]:.3f} | {cb['se_sim'][k]:.3f} | {r['se_boot_T31'][k]:.3f} | {cb['kappa'][k]:.2f} |")
    L += ["", "탐지확률(밴드 0.10, 가드 뒤 구조). 중단 = PO-2.5 규칙(OB 포트 하나, 156주)의 중단 확률(보고만).", "",
          "| 세계 | 1차 OB21 | 보조 OB5 | 가족 | b21 중앙 | b5 중앙 | 전진 중단 확률 | 중단 주 중앙 |",
          "|---|---|---|---|---|---|---|---|"]
    for sc, d in cb["detect"].items():
        L.append(f"| {sc} | {d['primary']:.3f} | {d['secondary']:.3f} | {d['family']:.3f} | {d['b21_median']:.3f} | "
                 f"{d['b5_median']:.3f} | {d['stop_prob']:.3f} | {f3(d['stop_week_median'])} |")
    L += ["", f"업종 ×1.69 1차 탐지: 평균 {cb['detect_sector_mean']:.3f}, 최소 {cb['detect_sector_min']:.3f}"
          f"({cb['detect_sector_argmin']}).", "",
          "## 업종 밴드 0.10 의 효과 (보고만)", "",
          "| OB | 업종 능동 노출 절댓값 최대 10·50·90% | 업종 요인 분산 몫 10·50·90% | 요인 분산 몫 10·50·90% |",
          "|---|---|---|---|"]
    for v, d in r["exposure"].items():
        L.append(f"| {v} | " + " | ".join("·".join(f"{x:.3f}" for x in d[k])
                                         for k in ("max_abs_sector_active", "sector_var_share", "factor_var_share")) + " |")
    cn = r["calibration"]["noband"]
    L += ["", f"밴드 없음(κ=1, 자기 맞는 모델 밴드): F×1.5 탐지 {cn['detect']['Fx1.5']['primary']:.3f}, 업종 ×1.69 평균 "
          f"{cn['detect_sector_mean']:.3f}·최소 {cn['detect_sector_min']:.3f}. 업종 세계 b21 중앙값 이동(맞는 모델 대비):", "",
          "| 업종 | 밴드 0.10 | 밴드 없음 |", "|---|---|---|"]
    for sec in r["sectors"]:
        k = f"sec:{sec}"
        L.append(f"| {sec} | {cb['detect'][k]['b21_median'] - cb['detect']['correct']['b21_median']:+.4f} | "
                 f"{cn['detect'][k]['b21_median'] - cn['detect']['correct']['b21_median']:+.4f} |")
    ch = r["calibration"]["cvhi"]
    L += ["", f"민감도(보고만): CV_s = 95% 상단 {t['CV_med_q05_50_95'][2]:.3f} 이면 F×1.5 탐지 "
          f"{ch['detect']['Fx1.5']['primary']:.3f}, 업종 평균 {ch['detect_sector_mean']:.3f}, h {ch['h_primary']:.3f}.", "",
          "## 구창 실현 바이어스 — 가드 뒤 (보고만, PO-2.6 M3: 증거 아님)", "",
          ", ".join(f"{k} {v:.3f}" for k, v in r["old_realized_report_only"].items()), "",
          f"_{f3(r['elapsed_s'])}초_"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
