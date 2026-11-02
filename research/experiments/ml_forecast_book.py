"""ML 수익률 예측 2단계 — 비용 반영 북(20거래일 격자, 판정은 비용 2배). PREREG §7·§8·§9, 정정 ML-1~ML-4.

모드:

- ``--mode parity`` (S2-0): 주입 하네스의 None 경로로 B 상수를 다시 돌려 ``B.json`` 의 O·R 요약값과 1e-10 안인지 본다. 출력 ``S2_parity.json``.
- ``--mode run``: N2s 관문 → 널 시간 실측(외삽 > 12시간이면 멈춘다, §10) → 원장(민감도 → 진단 → M) → 북 → 널 → P-1~P-7·알파 관문 → ``S2.{json,md}``(검증 전).
  **한 번만 돈다** — ``S2.json`` 이 있으면 멈춘다. 예측은 실행된 ``s1_preds.pkl`` 을 그대로 쓴다(ML-4 ⑥). 2024-01 이후 데이터는 쓰지 않는다.

판정 문장은 사전등록 규칙의 기계적 적용이다(GUARDRAILS §8). 승률은 내지 않는다 — 손익비·기대값만.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" --with lightgbm==4.7.0 \\
      python research/experiments/ml_forecast_book.py --mode run --cache <레포 밖> [--workers 16]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import pickle
import subprocess
import time
from pathlib import Path

import ml_forecast_common as C
import ml_forecast_models as MD
import ml_forecast_nulls as NL
import numpy as np
import pandas as pd
import pead_optimizer_b as B
import pead_optimizer_b_report as BR
from krx_quant_core.backtest.crosssectional import _trailing_adv
from ml_forecast_common import END, IC0, LABEL, PREREG_SHA, SIGMA_TARGET, TEST_YEARS
from ml_forecast_target import labels

OUT = Path(__file__).resolve().parents[1] / "logs" / "ml_forecast"
CORRECTIONS = ["ML-1", "ML-2", "ML-3", "ML-4", "ML-4a"]
MAX_W = 0.05
BAND, H_ALPHA, AUM = 0.10, 20, 3e8
N_NULL = 300
NULL_KINDS = ("N2", "N2s")
GATE_TOL, GATE_REPS = 0.15, 300
ALPHA_T_LABEL = 2.0  # 관문이 아니라 라벨 문턱(정정 ML-4a)
PUBLIC_LABEL = "재사용 창 — 알파 주장 불가, 전진만이 증거"
WALL_LIMIT_H = 12.0
FACTORS = ("KOSPI", "KOSDAQ", "UMD", "SMB", "LOWVOL", "LIQ")
ALPHA_NW_LAG = 2
# 원장 순서: 민감도 → 진단 → 1차(GUARDRAILS §4 ②). (이름, 신호, IC₀, AUM)
BOOKS = (("M_ic0.015", "M", 0.015, AUM), ("M_ic0.06", "M", 0.06, AUM), ("M_aum1e10", "M", IC0, 1e10),
         ("R", "R", IC0, AUM), ("LS", "LS", IC0, AUM), ("L", "L", IC0, AUM), ("P", "P", IC0, AUM), ("E", "E", IC0, AUM),
         ("M", "M", IC0, AUM))
_G: dict = {}


# ── 순수 함수(판정·지표) ────────────────────────────────────────────────────────


def strip_build_cost(recs: list) -> list:
    """첫 구축의 비용을 평가에서 뺀다(§7). 총수익은 남는다. 첫 체결 스텝 하나만 — 입력은 그대로 둔다."""
    out = list(recs)
    for k, r in enumerate(out):
        if r.n_names > 0:
            out[k] = dataclasses.replace(r, cost1=0.0, contrib=[(i, w, g, 0.0) for i, w, g, _ in r.contrib])
            break
    return out


def name_contrib(recs: list, mult: float) -> dict[int, float]:
    """종목별 순기여 Σ(w·g − m·비용)."""
    out: dict[int, float] = {}
    for r in recs:
        for i, w, g, c in r.contrib:
            out[i] = out.get(i, 0.0) + w * g - mult * c
    return out


def fragility(a: np.ndarray, intervals: np.ndarray, contrib: dict[int, float]) -> dict:
    """P-3·P-4·P-5 재료(ML-4 ⑧). a = 스텝 능동수익(2배), intervals = 보유 구간 순기여."""
    a = np.asarray(a, dtype=np.float64)
    T = a.size
    ev = float(a.mean())
    srt = np.sort(np.asarray(intervals, dtype=np.float64))[::-1]
    out = {"EV": ev, "interval_payoff": B.payoff_ratio(srt), "n_intervals": int(srt.size)}
    for k in (5, 20):
        out[f"interval_mean_drop_top{k}"] = float(srt[k:].mean()) if srt.size > k else float("nan")
        out[f"step_EV_drop_top{k}"] = float((a.sum() - srt[:k].sum()) / T)
    top = max(contrib, key=contrib.get) if contrib else None
    tc = contrib[top] if top is not None else float("nan")
    out.update({"top_name_row": top, "top_name_contrib": tc, "EV_drop_top_name": float((a.sum() - tc) / T),
                "top_name_share_of_sum_a": float(tc / a.sum()) if a.sum() != 0 else float("nan")})
    return out


def bars_s2(ir_m: float, q_n2: float, q_n2s: float, fr: dict, ir_p: float, ir_ls: float) -> dict:
    """P-1~P-7(§7·ML-4). 값이 NaN 이면 미달."""
    ev = fr["EV"]
    return {
        "P-1": bool(ir_m > max(q_n2, q_n2s)),
        "P-2": bool(ev > 0),
        "P-3": bool(fr["interval_payoff"] >= 1.0),
        "P-4": bool(all(fr[f"{m}_drop_top{k}"] > 0 for k in (5, 20) for m in ("interval_mean", "step_EV"))),
        "P-5": bool(fr["EV_drop_top_name"] > 0 and fr["EV_drop_top_name"] >= 0.5 * ev),
        "P-6": bool(ir_m > ir_p),
        "P-7": bool(ir_m > ir_ls),
    }


def verdict_s2(void: list[str], bars: dict) -> dict:
    """ML-4a 판정표. 순서: VOID → P 실패 → 전진 후보. 알파 t 는 판정에 들어오지 않는다(:func:`forward_label`)."""
    failed = [k for k, ok in bars.items() if not ok]
    if void:
        return {"verdict": "VOID", "failed": failed, "forward_arm": False}
    if failed:
        return {"verdict": "ML 이득 없음", "failed": failed, "forward_arm": False}
    return {"verdict": "전진 후보", "failed": [], "forward_arm": True}


def forward_label(alpha_t: float) -> str:
    """ML-4a ④ — 전진 팔 라벨. t < 2.0(또는 NaN)이면 "스타일 미분리" 를 앞에 붙인다. 판정이 아니다."""
    if np.isfinite(alpha_t) and alpha_t >= ALPHA_T_LABEL:
        return PUBLIC_LABEL
    return f"스타일 미분리(알파 t = {alpha_t:.2f}) — 스타일 기울기일 수 있음. {PUBLIC_LABEL}"


def max_drawdown(step_ret) -> float:
    """스텝 복리 곡선(시작 1)의 최대 낙폭(음수 또는 0)."""
    r = np.asarray(step_ret, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    curve = np.r_[1.0, np.cumprod(1.0 + r)]
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def absolute_report(net, kospi, kosdaq, step: int = C.BOOK_STEP) -> dict:
    """ML-4a ③ 보고만 — 비용 2배 절대 순수익, 지수 대비, 최대 낙폭. 입력은 같은 길이의 스텝 수익."""
    net, kospi, kosdaq = (np.asarray(x, dtype=np.float64) for x in (net, kospi, kosdaq))
    per_year = 252.0 / step

    def cum(x):
        return float(np.prod(1.0 + np.where(np.isfinite(x), x, 0.0)) - 1.0)

    return {"steps": int(net.size), "net_step_mean": float(net.mean()), "net_ann_simple": float(net.mean() * per_year),
            "net_ann_geometric": float((1.0 + cum(net)) ** (per_year / net.size) - 1.0), "cum_book": cum(net),
            "cum_kospi": cum(kospi), "cum_kosdaq": cum(kosdaq),
            "vs_kospi_step_mean": float(np.nanmean(net - kospi)), "vs_kosdaq_step_mean": float(np.nanmean(net - kosdaq)),
            "mdd_book": max_drawdown(net), "mdd_kospi": max_drawdown(kospi), "mdd_kosdaq": max_drawdown(kosdaq)}


def persistence_gate(real: dict, null: dict, tol: float = GATE_TOL) -> dict:
    """ML-4 ① 관문 — 널 평균이 실제의 ±tol 안."""
    rows = {k: {"real": real[k], "null": null[k], "gap": null[k] - real[k], "pass": bool(abs(null[k] - real[k]) <= tol)} for k in real}
    return {"rows": rows, "pass": all(v["pass"] for v in rows.values())}


def acf1(x) -> float:
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    return float((x[1:] @ x[:-1]) / (x @ x))


def rank_autocorr(s: dict, U: dict, pairs: list) -> float:
    """쌍 (t, t′) 마다 두 날 다 U 에 있는 자리의 Spearman 평균."""
    v = []
    for a, b in pairs:
        _, i, j = np.intersect1d(U[a], U[b], return_indices=True)
        x, z = s[a][i], s[b][j]
        ok = np.isfinite(x) & np.isfinite(z)
        if ok.sum() >= 30:
            v.append(C.spearman(x[ok], z[ok]))
    return float(np.mean(v))


def panel_of(pred: dict, U: dict, days: list[int], first: int, shape: tuple) -> np.ndarray:
    """예측(전체 열 번호 → U 행 순서) → 위험모델 좌표(first 부터) 패널."""
    out = np.full(shape, np.nan)
    for t in days:
        out[U[t], t - first] = pred[t]
    return out


def book_days(dates, oos: list[int]) -> list[int]:
    """북 격자(기준일 + 20k) 중 OOS IC 격자일 — 부분집합이어야 한다(§2)."""
    s = set(oos)
    all_ = list(range(C.grid_phase(dates), len(dates), C.BOOK_STEP))
    return [t for t in all_ if t in s]


def run_book(ctx, U_s: dict, rank_sig: np.ndarray, ic0: float = IC0, aum: float = AUM) -> tuple[list, list[str]]:
    """O 북 한 번. (평가용 스텝 — 구축 비용 뺌, 불변식 위반). 불변식은 뺀 뒤가 아니라 실제 비용으로 본다."""
    cell = B.Cell(h_alpha=H_ALPHA, band=BAND, aum=aum)
    recs = B.simulate(ctx, "O", cell, rank_sig=rank_sig, ic_by_year={int(ctx.dates[t].year): ic0 for t in ctx.grid},
                      sigma_r={t: SIGMA_TARGET for t in ctx.grid}, universe=lambda t: U_s[t], max_weight=MAX_W)
    return strip_build_cost(recs), B.check_invariants(recs, "O", cell, max_weight=MAX_W)


def factor_rows(ctx, recs, U_s: dict, liq: np.ndarray, kospi, kosdaq) -> pd.DataFrame:
    """스텝 팩터 수익 — 풀은 U(t)(MINOR g·ML-4 ③). 정의는 B 와 같고 LIQ(저회전 − 고회전)를 더한다."""
    rows = []
    for r in recs:
        t, f, fn = r.t, r.f, r.f_next
        pool = np.asarray(U_s[t], dtype=int)
        mom = ctx.Cff[:, t - BR.MOM_SKIP] / ctx.Cff[:, t - BR.MOM_LOOKBACK] - 1.0 if t >= BR.MOM_LOOKBACK else np.full(ctx.C.shape[0], np.nan)
        kp, mk = BR.compound(kospi, f, fn)
        kq, mq = BR.compound(kosdaq, f, fn)
        rows.append({"t": t, "date": r.date, "KOSPI": kp, "KOSDAQ": kq,
                     "UMD": BR.long_short(mom, pool, ctx.Cff, ctx.C, f, fn),
                     "SMB": BR.long_short(-ctx.cap[:, t], pool, ctx.Cff, ctx.C, f, fn),
                     "LOWVOL": BR.long_short(-B.trailing_vol(ctx.ret, t), pool, ctx.Cff, ctx.C, f, fn),
                     "LIQ": BR.long_short(-liq[:, t], pool, ctx.Cff, ctx.C, f, fn),
                     "index_missing_days": mk + mq})
    return pd.DataFrame(rows)


def alpha_report(net: pd.DataFrame, fac: pd.DataFrame) -> dict:
    m = net.merge(fac, on="t", how="inner")
    r = BR.ols_hac(m["net"].to_numpy(), m[list(FACTORS)].to_numpy(), lag=ALPHA_NW_LAG)
    r["factors"] = list(FACTORS)
    r["alpha_ann"] = r["alpha"] * 252.0 / C.BOOK_STEP
    return r


def summarize(recs: list, fac: pd.DataFrame | None = None) -> dict:
    """한 북의 판정·보고 숫자(OOS). 승률은 내지 않는다."""
    s = B.book_summary(recs)
    o = [r for r in recs if B.EVAL[0] <= r.date <= B.EVAL[1]]
    out = {k: s[k] for k in ("IR_1x", "IR_2x", "EV_1x", "EV_2x", "sd_a_2x", "step_payoff_2x", "interval_payoff_1x", "interval_payoff_2x",
                             "folds_IR_2x", "oos_steps", "optimal_share", "lam_fail_share", "lam_up_share", "lam_fail_nonup_share",
                             "lam_causes", "exante_20_mean", "realized_over_exante", "turnover_mean", "cost1_mean", "est_cost_mean",
                             "names_mean", "n_forced", "n_delist", "n_frozen", "n_untradeable", "factor_share_q10_50_90", "n_buy_limit_up")}
    df = B.oos(B.step_returns(recs, 2.0))
    a = df["a"].to_numpy()
    fr = fragility(a, B.holding_intervals(o, 2.0), name_contrib(o, 2.0))
    out["fragility_2x"] = fr
    out["EV_by_year_2x"] = {str(y): float(df.loc[pd.to_datetime(df["date"]).dt.year == y, "a"].mean()) for y in TEST_YEARS}
    if fac is not None:
        net = df[["t", "date", "net"]]
        out["alpha_2x"] = alpha_report(net, fac)
        out["vs_kospi_2x"] = BR.vs_kospi(net, fac)
        m = net.merge(fac[["t", "KOSPI", "KOSDAQ"]], on="t", how="inner")
        out["absolute_2x"] = absolute_report(m["net"], m["KOSPI"], m["KOSDAQ"])
    return out


# ── 패리티(S2-0) ───────────────────────────────────────────────────────────────


def parity_compare(new: dict, old: dict, tol: float = B.PARITY_TOL) -> dict:
    """B.json 요약값(숫자)과 대조. 사전은 재귀, NaN 은 NaN 끼리 같다."""
    bad, n = [], 0

    def walk(a, b, path):
        nonlocal n
        if isinstance(b, dict):
            for k in b:
                walk(a.get(k) if isinstance(a, dict) else None, b[k], f"{path}.{k}")
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            n += 1
            if a is None or not (abs(float(a) - float(b)) <= tol or (np.isnan(float(a)) and np.isnan(float(b)))):
                bad.append({"path": path, "new": a, "old": b})

    walk(new, old, "")
    return {"compared": n, "max_tol": tol, "mismatch": bad[:30], "n_mismatch": len(bad), "pass": not bad}


def run_parity(args, log) -> int:
    import pead_optimizer_po4 as P4

    t0 = time.time()
    cache = Path(args.cache) / "bparity"
    cache.mkdir(parents=True, exist_ok=True)
    bj = json.loads((Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer" / "B.json").read_text())
    fp = P4.fingerprint("daily_bars_adjusted", B.END_B)
    ctx, info, _ = B.build_ctx(cache, None, args.workers, log, core_ref=args.core_ref)
    pc = B.full_parity(ctx)
    log(f"코어 패리티 {pc['pass']}")
    ic_by, ic_info = B.ic_by_year_for(args, log)
    sigma_r = B.r_sigma_series(ctx)
    recs_o = B.simulate(ctx, "O", B.PRIMARY, ic_by_year=ic_by[B.PRIMARY.h_alpha], sigma_r=sigma_r)
    recs_r = B.simulate(ctx, "R")
    keys = ("IR_1x", "IR_2x", "EV_1x", "EV_2x", "sd_a_1x", "sd_a_2x", "step_payoff_2x", "interval_payoff_2x", "interval_EV_drop_top5_2x",
            "folds_IR_2x", "turnover_mean", "cost1_mean", "names_mean", "exante_20_mean", "optimal_share", "lam_fail_share")
    res = {}
    for kind, recs in (("O", recs_o), ("R", recs_r)):
        s = B.book_summary(recs)
        res[kind] = parity_compare({k: s[k] for k in keys}, {k: bj["primary"][kind][k] for k in keys})
        log(f"{kind}: 대조 {res[kind]['compared']}, 어긋남 {res[kind]['n_mismatch']}")
    # B.json 은 JSON 왕복을 거쳐 키가 문자열이다 — 같은 왕복으로 맞춘다
    ic = parity_compare(json.loads(json.dumps(ic_info["ic_by_year"])), bj["ic"]["ic_by_year"])
    fp_same = fp == bj["fingerprints"]["daily_bars_adjusted_b_range"] or list(map(str, fp)) == list(map(str, bj["fingerprints"]["daily_bars_adjusted_b_range"]))
    out = {"core_ref": args.core_ref, "B_code": bj["code"], "fingerprint_same_as_B": bool(fp_same), "core_parity": pc, "O": res["O"], "R": res["R"], "ic_by_year": ic,
           "pass": bool(pc["pass"] and res["O"]["pass"] and res["R"]["pass"] and ic["pass"] and fp_same), "elapsed_s": time.time() - t0,
           "note": "B.json 에 스텝 열이 없어 요약값을 1e-10 으로 대조한다(정정 ML-4 ⑧). 숫자는 B 것이고 ML 예측 결과가 아니다."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S2_parity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"패리티 {'통과' if out['pass'] else '실패'}")
    return 0 if out["pass"] else 1


# ── 입력 ───────────────────────────────────────────────────────────────────────


def index_returns(dates) -> tuple[np.ndarray, np.ndarray]:
    """KOSPI·KOSDAQ 일수익(`sector_index` 001·101). 날짜 ≤ END 만."""
    import ml_forecast_data as D

    con = D._db()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT code, date, close FROM sector_index WHERE code IN ('001', '101') AND date BETWEEN %s AND %s",
                        (str(dates[0].date()), END))
            df = pd.DataFrame(cur.fetchall(), columns=["code", "date", "close"])
    finally:
        con.close()
    out = []
    for c in ("001", "101"):
        g = df[df["code"] == c]
        lvl = pd.Series(pd.to_numeric(g["close"]).to_numpy(float), index=pd.to_datetime(g["date"])).reindex(dates)
        out.append(lvl.pct_change(fill_method=None).to_numpy())
    return out[0], out[1]


def ls_predictions(Z, U, y, vol, dates, grid, rows) -> dict:
    """LS(ML-4 ②) — F9~F15 워크포워드 릿지. M 과 같은 폴드·행·가중."""
    cols = list(MD.S_COLS)
    out = {}
    for yr in TEST_YEARS:
        days = MD.train_days(rows, MD.t0_of(dates, grid, yr))
        X, Y, w, _ = MD.train_matrix(days, Z, U, y, vol)
        beta = MD.fit_ridge(X[:, cols], Y, w)
        for t in C.oos_days(dates, grid, yr):
            out[t] = MD.predict_ridge(beta, Z[t][:, cols])
    return out


def prepare(args, log) -> dict:
    import ml_forecast_data as D
    import ml_forecast_features as FT
    import pead_optimizer_po3 as P3

    cache = Path(args.cache)
    P = D.build_panels(END, cache, log)
    S = pickle.load(open(cache / "s1_preds.pkl", "rb"))
    assert np.array_equal(P["codes"], S["codes"]) and P["dates"].equals(S["dates"]), "패널이 S1 과 다르다"
    dates, vol = P["dates"], P["vol"]
    grid, rows, oos = C.ic_grid(dates), C.phase_rows(dates), S["oos"]
    assert oos == C.oos_days(dates, grid)
    fr = D.v4_factor_returns(P)
    U = D.universe(P, fr, rows)
    assert all(np.array_equal(U[t], S["U"][t]) for t in oos), "U(t) 가 S1 과 다르다"
    Z = FT.transform(FT.raw_features(P), U)
    fry = D.v4_factor_returns(P, D.label_mask(P))
    uf = np.full_like(P["ret"], np.nan)
    uf[:, P["first"]:] = fry.u
    y, _ = labels(uf, vol, P["close"], P["delist_last"], U, P["seam"])
    preds = dict(S["preds"])
    preds["LS"] = ls_predictions(Z, U, y, vol, dates, grid, rows)
    first = P["first"]
    sl = slice(first, None)
    book = book_days(dates, oos)
    c = {"R": P["ret"][:, sl], "cap": P["cap"][:, sl], "msk": P["msk"][:, sl], "lab": P["lab"][:, sl],
         "Xs": {k: v[:, sl] for k, v in P["Xs"].items()}}
    grid_s = [t - first for t in book]
    fp_h = D.code_hash()
    models = P3.build_models(c, grid_s, args.workers, cache / f"book_models_v4_{fp_h}_{len(grid_s)}.pkl", log, vra=None, shrink=B.V4_SHRINK)
    didx = pd.DatetimeIndex(dates[sl])
    adv = _trailing_adv(P["tv"] / 1e6, len(dates), 20, P["close"])[:, sl]
    market, last, mk_info = B.market_and_delisting(P["codes"])
    delist_after = np.full(len(P["codes"]), 10**9, dtype=np.int64)
    for i, cc in enumerate(P["codes"]):
        d = last.get(str(cc))
        if d is not None:
            j = int(didx.searchsorted(pd.Timestamp(d), side="right")) - 1
            if j < len(didx) - 1:
                delist_after[i] = j
    shape = c["R"].shape
    ctx = B.Ctx(dates=didx, C=P["close"][:, sl], tv_won=P["tv"][:, sl], adv=adv, cap=c["cap"], sig=np.full(shape, np.nan), ret=c["R"],
                lab=c["lab"], market=market, grid=grid_s, models=models, delist_after=delist_after)
    U_s = {t - first: U[t] for t in book}
    for t in book:  # U(t) 는 그날 v4 모델 종목 안이어야 한다
        assert ctx.in_u(t - first)[U[t]].all(), f"{dates[t].date()} U(t) 가 v4 모델 밖이다"
    return {"P": P, "ctx": ctx, "U": U, "U_s": U_s, "preds": preds, "book": book, "oos": oos, "first": first, "shape": shape,
            "fr": fr, "y": y, "dates": dates, "vol": vol, "market_info": mk_info, "prep": c}


def _null_panel(kind: str, r: int) -> np.ndarray:
    c = _G
    pm = c["pred"]
    if kind == "N2":
        pi = NL.n2_perm(c["u_star"], r, c["N"])
        s = {t: NL.n2_assign(pm[t], c["U"][t], pi, r, t, c["N"])[0] for t in c["book"]}
    else:
        s = NL.n2s_walk(pm, c["U"], c["book"], r, c["N"])
    return panel_of(s, c["U"], c["book"], c["first"], c["shape"])


def _null_task(a) -> tuple:
    kind, r = a
    c = _G
    t0 = time.time()
    recs, viol = run_book(c["ctx"], c["U_s"], _null_panel(kind, r))
    s = B.book_summary(recs)
    return kind, r, B.ir(B.oos(B.step_returns(recs, 2.0))["a"]), s["optimal_share"], s["lam_fail_nonup_share"], viol, time.time() - t0


def _gate_task(r: int) -> tuple:
    c = _G
    s = NL.n2s_walk(c["pred"], c["U"], c["oos"], r, c["N"])
    sb = NL.n2s_walk(c["pred"], c["U"], c["book"], r, c["N"])
    ic = [C.spearman(s[t][c["ok"][t] & np.isfinite(s[t])], c["y_u"][t][c["ok"][t] & np.isfinite(s[t])]) for t in c["oos"]]
    return acf1(ic), rank_autocorr(s, c["U"], c["p20"]), rank_autocorr(sb, c["U"], c["pb"])


def _git_sha() -> str:
    d = str(Path(__file__).resolve().parent)
    sha = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--", "research/experiments", "src"], capture_output=True, text=True, check=False).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def run(args, log) -> int:
    import ml_forecast_data as D
    import ml_forecast_ic as IC
    from ml_forecast_target import tradable
    from swing_it.diagnostics.trials import record_trial

    window = C.assert_window(END, "eval")
    C.require_eval(window, "북")
    t_start = time.time()
    if (OUT / "S2.json").exists():
        raise SystemExit("S2.json 이 이미 있다 — 2단계는 한 번만 돈다")
    s1 = json.loads((OUT / "S1.json").read_text())
    if not s1["enters_stage2"]["M"]:
        raise SystemExit("M 이 2단계 진입이 아니다")
    par = OUT / "S2_parity.json"
    if not par.exists() or not json.loads(par.read_text())["pass"]:
        raise SystemExit("S2-0 패리티가 통과하지 않았다 — VOID(§8)")
    run_md = [(OUT / "RUN.md").read_text().rstrip(), "", f"- 2단계 시작 {time.strftime('%F %T')}, 작업자 {args.workers}, 코드 {_git_sha()}."]
    dq = IC.data_check()
    fp0 = D.fingerprint_range("2017-01-01", END)
    A = prepare(args, log)
    ctx, U, U_s, preds, book, oos = A["ctx"], A["U"], A["U_s"], A["preds"], A["book"], A["oos"]
    y, vol, dates, first, shape = A["y"], A["vol"], A["dates"], A["first"], A["shape"]
    N = len(A["P"]["codes"])
    u_star = np.unique(np.concatenate([U[t] for t in oos]))
    log(f"북 격자 {len(book)}일 {dates[book[0]].date()}~{dates[book[-1]].date()}")

    # LS 배선 확인(리뷰어 IC +0.0345) — 판정에 쓰지 않는다
    ls_ic = IC.stat_of(IC.ic_series(preds["LS"], y, U, vol, oos, window), dates)
    m_ic = IC.stat_of(IC.ic_series(preds["M"], y, U, vol, oos, window), dates)
    assert abs(m_ic["mean"] - s1["stats"]["M"]["mean"]) < 1e-12, "M 의 IC 가 S1 과 다르다"
    d_ls = IC.delta_nw(IC.ic_series(preds["M"], y, U, vol, oos, window), IC.ic_series(preds["LS"], y, U, vol, oos, window))
    log(f"LS IC {ls_ic['mean']:+.4f}, ΔIC(M−LS) {d_ls['mean']:+.4f} t {d_ls['t_nw6']:+.2f}")

    # N2s 관문(ML-4 ①) — 북 전
    y_u = {t: y[t][U[t]] for t in oos}
    ok = {t: np.isfinite(y_u[t]) & tradable(vol, t)[U[t]] for t in oos}
    p20 = [(oos[i], oos[i + 4]) for i in range(len(oos) - 4) if oos[i + 4] - oos[i] == C.BOOK_STEP]
    pb = list(zip(book[:-1], book[1:]))
    _G.clear()
    _G.update(pred=preds["M"], U=U, U_s=U_s, N=N, oos=oos, book=book, ok=ok, y_u=y_u, p20=p20, pb=pb, u_star=u_star, ctx=ctx, first=first, shape=shape)
    with mp.get_context("fork").Pool(args.workers) as pool:
        g = np.array(pool.map(_gate_task, range(GATE_REPS), chunksize=4))
    real_ic = [C.spearman(preds["M"][t][ok[t]], y_u[t][ok[t]]) for t in oos]
    gate = persistence_gate({"ic_acf1": acf1(real_ic), "rank20_ic_grid": rank_autocorr(preds["M"], U, p20), "rank20_book_walk": rank_autocorr(preds["M"], U, pb)},
                            {"ic_acf1": float(g[:, 0].mean()), "rank20_ic_grid": float(g[:, 1].mean()), "rank20_book_walk": float(g[:, 2].mean())})
    log(f"N2s 관문 {'통과' if gate['pass'] else '실패'}")
    if not gate["pass"]:
        (OUT / "S2_gate_failed.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1, default=float))
        raise SystemExit("N2s 지속성 관문 실패 — VOID(ML-4 ①), 북을 돌리지 않는다")

    # 시간 실측(§10) — 널 3회씩
    t_m = time.time()
    with mp.get_context("fork").Pool(min(6, args.workers)) as pool:
        probe = pool.map(_null_task, [(k, r) for k in NULL_KINDS for r in range(3)], chunksize=1)
    each = float(np.mean([p[-1] for p in probe]))
    est_h = (2 * N_NULL + 2 * len(BOOKS)) * each / args.workers / 3600
    run_md.append(f"- 널 북 3회씩 실측: 회당 {each:.0f}s(벽시계 {time.time() - t_m:.0f}s). 널 {2 * N_NULL} + 북 {len(BOOKS)}·재현 → 작업자 {args.workers} 외삽 {est_h:.2f}시간. Modal 문턱 {WALL_LIMIT_H:.0f}시간(§10).")
    (OUT / "RUN.md").write_text("\n".join(run_md) + "\n")
    log(run_md[-1])
    if est_h > WALL_LIMIT_H:
        log("로컬 외삽이 문턱을 넘는다 — 멈춘다(Modal 판단)")
        return 3

    # 원장 → 북(민감도 → 진단 → 1차)
    base = {"stage": "S2-book", "prereg": PREREG_SHA, "corrections": CORRECTIONS, "end": END, "universe": "cap400_adv5e9",
            "max_weight": MAX_W, "band": BAND, "sigma_target": SIGMA_TARGET, "core": args.core_ref}
    panels = {k: panel_of(preds[k], U, book, first, shape) for k in ("M", "E", "L", "P", "R", "LS")}
    fac = None
    res_books, recs_by, viol = {}, {}, []
    ko, kq = index_returns(ctx.dates)
    liq = A["prep"]["Xs"]["liquidity"]
    for name, sig, ic0, aum in BOOKS:
        record_trial(LABEL, {**base, "book": name, "signal": sig, "ic0": ic0, "aum": aum}, note=f"S2 {'1차' if name == 'M' else '민감도·진단'} {name}")
    def _book_task_args():
        return [(name, sig, ic0, aum) for name, sig, ic0, aum in BOOKS]
    _G["panels"] = panels
    with mp.get_context("fork").Pool(min(len(BOOKS), args.workers)) as pool:
        for name, recs, v in pool.imap_unordered(_book_task, _book_task_args(), chunksize=1):
            recs_by[name] = recs
            viol += v
            log(f"  북 {name} 끝")
    fac = factor_rows(ctx, recs_by["M"], U_s, liq, ko, kq)
    for name, *_ in BOOKS:
        res_books[name] = summarize(recs_by[name], fac)
    # 결정성(§8) — M 을 다시 돌려 스텝 바이트 대조
    recs2, _ = run_book(ctx, U_s, panels["M"])
    det = bool(_step_bytes(recs2) == _step_bytes(recs_by["M"]))
    log(f"결정성 {det}")
    pickle.dump({n: r for n, r in recs_by.items()}, open(Path(args.cache) / "s2_recs.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # 널
    t_n = time.time()
    tasks = [(k, r) for k in NULL_KINDS for r in range(N_NULL)]
    null = {k: np.full(N_NULL, np.nan) for k in NULL_KINDS}
    nopt = {k: np.full(N_NULL, np.nan) for k in NULL_KINDS}
    nlam = {k: np.full(N_NULL, np.nan) for k in NULL_KINDS}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (kind, r, v, op, lf, vi, _) in enumerate(pool.imap_unordered(_null_task, tasks, chunksize=1)):
            null[kind][r], nopt[kind][r], nlam[kind][r] = v, op, lf
            viol += [f"{kind}{r} {x}" for x in vi]
            if (i + 1) % 60 == 0:
                log(f"  널 {i + 1}/{len(tasks)} ({(time.time() - t_n) / 60:.1f}분)")
    run_md.append(f"- 널 {len(tasks)}회 실제 벽시계 {(time.time() - t_n) / 60:.1f}분.")
    if viol:
        (Path(args.cache) / "S2_invariant_violations.txt").write_text("\n".join(viol))
        raise SystemExit(f"불변식 위반 {len(viol)}건 — 결과를 쓰지 않는다. 목록: {Path(args.cache) / 'S2_invariant_violations.txt'}")

    # 판정
    M = res_books["M"]
    q2, q2s = NL.q95(null["N2"]), NL.q95(null["N2s"])
    bars = bars_s2(M["IR_2x"], q2, q2s, M["fragility_2x"], res_books["P"]["IR_2x"], res_books["LS"]["IR_2x"])
    alpha_t = M["alpha_2x"]["t_alpha"]
    fp1 = D.fingerprint_range("2017-01-01", END)
    void = []
    if M["oos_steps"] < B.OOS_MIN:
        void.append(f"북 OOS 결정일 {M['oos_steps']} < {B.OOS_MIN}")
    if M["optimal_share"] < B.OPT_OK_MIN:
        void.append(f"optimal 비율 {M['optimal_share']:.3f} < {B.OPT_OK_MIN}")
    if M["lam_fail_nonup_share"] > B.LAM_FAIL_MAX:
        void.append(f"λ 실패(solver·gap·down) {M['lam_fail_nonup_share']:.3f} > {B.LAM_FAIL_MAX}")
    if not det:
        void.append("같은 설정 재실행이 비트 단위로 다르다")
    if dq["exit"] != 0:
        void.append(f"조정주가 검사 실패(exit {dq['exit']})")
    if fp0 != fp1:
        void.append("실행 중 표 지문이 바뀌었다")
    v = verdict_s2(void, bars)
    led = BR.ledger_n_ml()
    hur = {"narrow": BR.alpha_hurdle(led["narrow"]), "wide": BR.alpha_hurdle(led["wide"])}
    net2 = B.oos(B.step_returns(recs_by["M"], 2.0))["net"]
    aM, aE = (B.oos(B.step_returns(recs_by[k], 2.0))["a"].to_numpy() for k in ("M", "E"))
    boot = B.stationary_boot_dir(aM, aE, seed_key="ml-boot-dIR-M-E")
    m_e = s1["bars"]["M"]["I-4"] and M["IR_2x"] - res_books["E"]["IR_2x"] > 0
    o = [r for r in recs_by["M"] if B.EVAL[0] <= r.date <= B.EVAL[1]]
    attr = BR.attribution_report(ctx, o, A["fr"], {"lab": A["prep"]["lab"], "Xs": A["prep"]["Xs"]}, lambda t: U_s[t])
    top = M["fragility_2x"]["top_name_row"]
    res = {
        "status": "검증 전", "prereg": PREREG_SHA, "corrections": CORRECTIONS, "end": END, "core_ref": args.core_ref, "code_sha": _git_sha(),
        "verdict": v["verdict"], "failed": v["failed"], "forward_arm": v["forward_arm"], "void": void, "bars": bars,
        "grid": {"steps": len(book), "first": str(dates[book[0]].date()), "last": str(dates[book[-1]].date())},
        "books": res_books, "top_name_code": str(A["P"]["codes"][top]) if top is not None else None,
        "null": {k: {"reps": N_NULL, "q95_IR_2x": NL.q95(null[k]), "mean": float(np.nanmean(null[k])), "sd": float(np.nanstd(null[k], ddof=1)),
                     "optimal_share_mean": float(np.nanmean(nopt[k])), "lam_fail_nonup_mean": float(np.nanmean(nlam[k])),
                     "pct_rank_of_M": float(np.mean(null[k] < M["IR_2x"]))} for k in NULL_KINDS},
        "p1_bar": float(max(q2, q2s)), "gate_N2s": gate,
        "alpha_label": {"t": alpha_t, "label_threshold": ALPHA_T_LABEL, "style_unseparated": bool(not (np.isfinite(alpha_t) and alpha_t >= ALPHA_T_LABEL)),
                        "is_gate": False, "report": M["alpha_2x"]},
        "labels": {"public": PUBLIC_LABEL, "forward_arm_label": forward_label(alpha_t), "ledger": led, "hurdle_label_only": hur,
                   "t_over_narrow": bool(alpha_t > hur["narrow"]["hurdle"]), "t_over_wide": bool(alpha_t > hur["wide"]["hurdle"]),
                   "deflation_narrow": BR.deflation(net2, led["narrow"]), "deflation_wide": BR.deflation(net2, led["wide"]),
                   "M_over_E": bool(m_e), "S1_wording": "I-5 통과, 그러나 같은 입력의 선형 대조와 구별되지 않음"},
        "ls_check": {"ic_mean": ls_ic["mean"], "delta_M_LS": d_ls, "reviewer_ic": 0.0345, "reviewer_delta": 0.0065, "reviewer_t": 0.66},
        "dIR_M_E_boot_q05_50_95": [float(np.quantile(boot, q)) for q in (0.05, 0.5, 0.95)],
        "attribution_M": attr, "factor_index_missing_days": int(fac["index_missing_days"].sum()), "invariants": {"violations": 0},
        "determinism": det, "data_check": dq, "fingerprints": {"before": fp0, "after": fp1}, "market": A["market_info"],
        "elapsed_s": time.time() - t_start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "S2.md").write_text(render(json.loads((OUT / "S2.json").read_text())))
    run_md.append(f"- 2단계 전체 벽시계 {res['elapsed_s'] / 60:.1f}분.")
    (OUT / "RUN.md").write_text("\n".join(run_md) + "\n")
    log(f"2단계 판정 {v['verdict']} {v['failed']}")
    return 0


def _book_task(a) -> tuple:
    name, sig, ic0, aum = a
    recs, viol = run_book(_G["ctx"], _G["U_s"], _G["panels"][sig], ic0, aum)
    return name, recs, [f"{name} {x}" for x in viol]


def _step_bytes(recs) -> bytes:
    return np.array([[r.gross, r.cost1, r.bench, r.turnover] for r in recs], dtype=np.float64).tobytes()


def _yes(b) -> str:
    return "통과" if b else "**미달**"


def render(r: dict) -> str:
    """S2.json → 표. 숫자는 JSON 에서만 온다. 승률은 없다."""
    M = r["books"]["M"]
    F = M["fragility_2x"]
    n2, n2s = r["null"]["N2"], r["null"]["N2s"]
    A = r["alpha_label"]["report"]
    AB = M["absolute_2x"]
    L = [f"# ML 수익률 예측 2단계 — 비용 반영 북 ({r['status']})", "",
         f"> **{r['status']}.** 리뷰·재계산 전 숫자다. {r['labels']['public']}. S1: {r['labels']['S1_wording']}.", "",
         f"사전등록 `{r['prereg']}` + 정정 {', '.join(r['corrections'])}. END {r['end']}, core `{r['core_ref']}`, 코드 `{r['code_sha']}`. 원본 `S2.json`.", "",
         f"**판정: {r['verdict']}**" + (f" — 실패한 바 {r['failed']}" if r["failed"] else "") + (f" — VOID 사유 {r['void']}" if r["void"] else "")
         + f". 전진 팔 {'연다' if r['forward_arm'] else '열지 않는다'}.", "",
         f"북 격자 {r['grid']['steps']}스텝({r['grid']['first']}~{r['grid']['last']}), 벤치 U(t) 동일가중, AUM 3억, 상한 5%, 사전 능동위험 2%/20일, 첫 구축 비용 제외.", "",
         "## 표 1 — 북 (비용 2배, 스텝 = 20거래일)", "",
         "| 북 | 역할 | 스텝 EV | 손익비(보유 구간) | 손익비(스텝) | IR 2배 | IR 1배 | 상위 5 제거 EV | 상위 20 제거 EV | 1위 종목 몫 | 회전 | 비용 1배 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    roles = {"M": "후보", "E": "라벨(진입 실패)", "P": "P-6 기준", "LS": "P-7 기준", "L": "보고", "R": "보고", "M_ic0.015": "민감도", "M_ic0.06": "민감도", "M_aum1e10": "민감도"}
    for k in ("M", "LS", "P", "E", "L", "R", "M_ic0.015", "M_ic0.06", "M_aum1e10"):
        b = r["books"][k]
        f = b["fragility_2x"]
        L.append(f"| {k} | {roles[k]} | {b['EV_2x']:+.4%} | {b['interval_payoff_2x']:.2f} | {b['step_payoff_2x']:.2f} | {b['IR_2x']:+.3f} | {b['IR_1x']:+.3f} | "
                 f"{f['step_EV_drop_top5']:+.4%} | {f['step_EV_drop_top20']:+.4%} | {f['top_name_share_of_sum_a']:+.1%} | {b['turnover_mean']:.3f} | {b['cost1_mean']:.4%} |")
    L += ["", "\"상위 k 제거 EV\" 는 스텝 EV 정의다. B 정의(보유 구간 평균)는 아래 바 표에 있다.", "",
          "## 표 2 — 바 (M, 비용 2배)", "", "| 바 | 값 | 기준 | 결과 |", "|---|---|---|---|",
          f"| P-1 IR > max(N2 q95, N2s q95) | {M['IR_2x']:+.3f} | {r['p1_bar']:+.3f} (N2 {n2['q95_IR_2x']:+.3f}, N2s {n2s['q95_IR_2x']:+.3f}) | {_yes(r['bars']['P-1'])} |",
          f"| P-2 스텝 EV > 0 | {F['EV']:+.4%} | 0 | {_yes(r['bars']['P-2'])} |",
          f"| P-3 보유 구간 손익비 ≥ 1 | {F['interval_payoff']:.3f} | 1.0 | {_yes(r['bars']['P-3'])} |",
          f"| P-4 상위 5·20 제거 EV > 0 (두 정의) | 스텝 {F['step_EV_drop_top5']:+.4%}·{F['step_EV_drop_top20']:+.4%}; 구간 평균 {F['interval_mean_drop_top5']:+.5f}·{F['interval_mean_drop_top20']:+.5f} | 0 | {_yes(r['bars']['P-4'])} |",
          f"| P-5 1위 종목({r['top_name_code']}) 제거 EV > 0 ∧ ≥ 0.5·EV | {F['EV_drop_top_name']:+.4%} (몫 {F['top_name_share_of_sum_a']:+.1%}) | {0.5 * F['EV']:+.4%} | {_yes(r['bars']['P-5'])} |",
          f"| P-6 IR > IR(P) | {M['IR_2x']:+.3f} | {r['books']['P']['IR_2x']:+.3f} | {_yes(r['bars']['P-6'])} |",
          f"| P-7 IR > IR(LS) | {M['IR_2x']:+.3f} | {r['books']['LS']['IR_2x']:+.3f} | {_yes(r['bars']['P-7'])} |",
          "",
          f"**필수 라벨(바 아님, ML-4a):** 알파 HAC t {r['alpha_label']['t']:+.2f}"
          + (" — **스타일 미분리**" if r["alpha_label"]["style_unseparated"] else "") + f". 전진 팔이 열리면 라벨: \"{r['labels']['forward_arm_label']}\".", "",
          f"M 은 N2 널의 {n2['pct_rank_of_M']:.1%}, N2s 널의 {n2s['pct_rank_of_M']:.1%} 분위다.", "",
          "## 절대 수익(M, 비용 2배 — 보고만, ML-4a)", "",
          "| 항목 | 북 | KOSPI | KOSDAQ |", "|---|---|---|---|",
          f"| 누적 | {AB['cum_book']:+.1%} | {AB['cum_kospi']:+.1%} | {AB['cum_kosdaq']:+.1%} |",
          f"| 최대 낙폭(스텝 복리) | {AB['mdd_book']:.1%} | {AB['mdd_kospi']:.1%} | {AB['mdd_kosdaq']:.1%} |", "",
          f"순수익 스텝 평균 {AB['net_step_mean']:+.3%}, 연환산 단리 {AB['net_ann_simple']:+.1%}·기하 {AB['net_ann_geometric']:+.1%}. "
          f"스텝 평균 초과: KOSPI 대비 {AB['vs_kospi_step_mean']:+.3%}, KOSDAQ 대비 {AB['vs_kosdaq_step_mean']:+.3%}. 현금 금리는 빼지 않았다.", "",
          "## 알파 회귀(U(t) 안 팩터, 2배, NW lag 2 — 라벨)", "",
          f"스텝 알파 {A['alpha']:+.4%}(연 {A['alpha_ann']:+.2%}), t {A['t_alpha']:+.2f}, R² {A['r2']:.3f}, n {A['n']}.", "",
          "| 팩터 | β | t |", "|---|---|---|"]
    for k, b, t in zip(A["factors"], A["beta"], A["t_beta"]):
        L.append(f"| {k} | {b:+.3f} | {t:+.2f} |")
    h = r["labels"]["hurdle_label_only"]
    led = r["labels"]["ledger"]
    L += ["", f"**라벨만:** 좁은 N {led['narrow']} 문턱 {h['narrow']['hurdle']:.2f}, 넓은 N {led['wide']} 문턱 {h['wide']['hurdle']:.2f}. "
          f"Deflated Sharpe 좁은 {r['labels']['deflation_narrow'].get('deflated_sharpe', float('nan'))}, 넓은 {r['labels']['deflation_wide'].get('deflated_sharpe', float('nan'))}. "
          f"\"M 이 E 보다 낫다\"(I-4 ∧ ΔIR > 0): {r['labels']['M_over_E']}, ΔIR(M−E) 부트스트랩 5·50·95% {r['dIR_M_E_boot_q05_50_95']}.", "",
          "## 연도별(M, 2배)", "", "| 연도 | 스텝 EV | IR |", "|---|---|---|"]
    for y in ("2020", "2021", "2022", "2023"):
        L.append(f"| {y} | {M['EV_by_year_2x'][y]:+.4%} | {M['folds_IR_2x'][y]:+.3f} |")
    g = r["gate_N2s"]["rows"]
    L += ["", "## 관문(§8·ML-4)", "",
          f"- N2s 지속성: IC lag-1 실제 {g['ic_acf1']['real']:.3f}·널 {g['ic_acf1']['null']:.3f}; 20일 순위 IC 격자 {g['rank20_ic_grid']['real']:.3f}·{g['rank20_ic_grid']['null']:.3f}; 북 워크 {g['rank20_book_walk']['real']:.3f}·{g['rank20_book_walk']['null']:.3f}. 허용 ±0.15. 통과 {r['gate_N2s']['pass']}.",
          f"- optimal {M['optimal_share']:.3f}(바 0.95), λ 실패 {M['lam_fail_share']:.3f}, 그중 up 제외 {M['lam_fail_nonup_share']:.3f}(바 0.10), 원인 {M['lam_causes']}.",
          f"- 결정성 {r['determinism']}, 불변식 위반 {r['invariants']['violations']}, 조정주가 검사 exit {r['data_check']['exit']}, 표 지문 시작=끝 {r['fingerprints']['before'] == r['fingerprints']['after']}.",
          f"- 널 평균 IR N2 {n2['mean']:+.3f}(sd {n2['sd']:.3f}), N2s {n2s['mean']:+.3f}(sd {n2s['sd']:.3f}). 널 optimal 평균 {n2['optimal_share_mean']:.3f}·{n2s['optimal_share_mean']:.3f}.",
          f"- LS 배선: IC {r['ls_check']['ic_mean']:+.4f}(리뷰어 +0.0345), ΔIC(M−LS) {r['ls_check']['delta_M_LS']['mean']:+.4f}, t {r['ls_check']['delta_M_LS']['t_nw6']:+.2f}.", "",
          "## 보고만", "",
          f"- 사전 능동위험 평균 {M['exante_20_mean']:.4f}, 실현/사전 {M['realized_over_exante']:.2f}, 평균 종목 {M['names_mean']:.0f}, 추정 비용 {M['est_cost_mean']:.4%}.",
          f"- KOSPI 대비(2배) 스텝 평균 {M['vs_kospi_2x']['excess_mean']:+.4%}, IR {M['vs_kospi_2x']['excess_IR']:+.3f}, 누적 북 {M['vs_kospi_2x']['cum_book']:+.1%}, KOSPI {M['vs_kospi_2x']['cum_kospi']:+.1%}.",
          f"- 귀속: 특이 분산 몫 {r['attribution_M']['specific_share_var']:.3f}, 평균 몫 {r['attribution_M']['specific_share_mean']:.3f}, 묶음 평균 {r['attribution_M']['mean_by_group']}.",
          f"- 강제 매도 {M['n_forced']}, 폐지 {M['n_delist']}, 얼림 {M['n_frozen']}, 상한가 매수 {M['n_buy_limit_up']}.", "",
          f"_{r['elapsed_s'] / 60:.1f}분_"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("parity", "run", "render"), required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--core-ref", default="13c56a7")
    args = ap.parse_args()
    if args.mode == "render":
        (OUT / "S2.md").write_text(render(json.loads((OUT / "S2.json").read_text())))
        return 0
    assert args.cache and Path(__file__).resolve().parents[2] not in Path(args.cache).resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    return run_parity(args, log) if args.mode == "parity" else run(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
