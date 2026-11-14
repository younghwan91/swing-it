"""이평선 q차식 외삽 — 시간대별(주·일·60분·5분) 보정 **사후 진단 — 판정 무관**.

관문 0(`research/logs/ma_quad/RESULTS.md`, 일봉 20칸 불합격)을 다시 여는 것이 아니다. 같은 절차를 봉 단위만 바꿔 그려 보는 기술 진단이다.
판정·원장(`TRIALS.jsonl`·`calib_results.json`·`RESULTS.md`·`PREREG.md`)은 건드리지 않는다. 출력은 ``research/logs/ma_quad/tf_diag/`` 뿐.

절차(관문 0 과 같은 부품 — ``ma_quad``·``ma_quad_calib``·``maquad_calib.cell_probs``):
- 사건 = 지금 정배열 아님(봉 단위 MA5 > MA20 > MA60 > MA120 아님) ∧ 네 이평선 모두 정의 ∧ 유니버스. EPS·200일선·d 거름 없음.
- N = 20, k ∈ {3, 5, 10} 봉, q = 2·1·0 같은 사건 집합(세 확률·라벨 모두 유한). 풀 = 달 첫 봉 기준 ``month_pool``(3년, ≥ 120 봉).
- BSS(q2 대 q0, q1 대 q0) 군집 복원추출 95% — 군집은 주·일봉 = 달력 월, 분봉 = 거래일. q 별 신뢰도 표·ECE·P ≥ 0.5 해상도.
- 일봉: 관문 0 로더 ``maquad_calib.load``(정지일 NaN 규칙 포함), 유니버스 ADV20 ≥ 10억(``universe_mask``), 2017-01 ~ 2023-12.
- 주봉: 같은 일봉 패널에서 ISO 주 마지막 거래일 종가(끝난 주만), 유니버스 = 그 주 마지막 날 일봉 유니버스, 사건 2015 ~ 2023.
- 60분·5분: ``minute_bars``(KST naive, 시간대 변환 없음) 09:00 ≤ ts < 15:33, 2025-08-01 ~ 2026-11-13. 봉은 날을 넘겨 묶지 않지만
  이평선·적합은 날을 넘겨 이어진다. 그 구간 일 거래대금 중앙값 상위 300 종목(보통주·KOSPI/KOSDAQ). 분봉은 원 가격이라
  봉 사이 로그 변화가 가격제한폭(±30%) 밖인 종목(분할·병합 의심)은 뺀다. **오염 구간(2025-08~), 증거 아님.**

사용:
``set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --with numba --extra pg python research/experiments/maquad_tf_diag.py --tf daily [--workers 9]`` (weekly·60m·5m 도 같게),
끝나면 ``--report`` 로 그림·TF_DIAG.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path[:0] = [".", "src"]
from research.experiments import maquad_calib as G0  # noqa: E402
from research.signals import ma_quad as MQ  # noqa: E402
from research.signals import ma_quad_calib as CAL  # noqa: E402

N = 20
KS = (3, 5, 10)
QS = (2, 1, 0)
K_EX = 5
TFS = ("weekly", "daily", "60m", "5m")
INTRA = ("2025-08-01", "2026-11-13")
INTRA_OPEN, INTRA_CLOSE_EXCL = "09:00:00", "15:33:00"
TOP_CODES = 300
LIMIT_LO, LIMIT_HI = np.log(0.70) - 0.005, np.log(1.30) + 0.005     # 가격제한폭 ±30% 밖 = 기업행위 의심
EV = {"daily": (G0.EVENT_START, G0.LABEL_END), "weekly": ("2015-01-01", G0.LABEL_END),
      "60m": (INTRA[0], INTRA[1] + " 23:59"), "5m": (INTRA[0], INTRA[1] + " 23:59")}
CLUSTER = {"weekly": "month", "daily": "month", "60m": "day", "5m": "day"}

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research" / "logs" / "ma_quad" / "tf_diag"
SCRATCH = Path.home() / ".cache" / "cc-swing" / "scratch" / "ma_quad" / "tf_diag"
LABEL = "사후 진단 — 판정 무관"
LABEL_INTRA = "오염 구간(2025-08~), 증거 아님"

_G: dict = {}
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def _connect():
    from swing_it.storage import connect

    if not os.environ.get("KR_QUANT_DB"):
        raise SystemExit("KR_QUANT_DB 가 없다 — set -a; . ~/git/quant-airflow/.env; set +a")
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return con


# =============================================================================================== 패널

def _finish(tf: str, days: pd.DatetimeIndex, codes: list, C: np.ndarray, U: np.ndarray, member: dict, extra: dict) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(C > 0, np.log(C), np.nan)
    st = MQ.ma_state(C)
    ma = st["ma"]
    defined = np.isfinite(ma[5]) & np.isfinite(ma[20]) & np.isfinite(ma[60]) & np.isfinite(ma[120])
    if CLUSTER[tf] == "month":
        cl = np.asarray(days.year * 100 + days.month, dtype=np.int64)
    else:
        cl = np.asarray(days.year * 10000 + days.month * 100 + days.day, dtype=np.int64)
    X = {"tf": tf, "days": days, "codes": codes, "C": C, "y": y, "sigma": MQ.rolling_sigma(y, G0.SIGMA_W), "U": U,
         "member": member, "aligned": st["aligned"], "defined": defined, "cluster": cl,
         "ev_lo": EV[tf][0], "ev_hi": EV[tf][1]}
    X.update(extra)
    return X


def _daily_masks(raw: dict):
    from swing_it.features.korea import universe_mask

    days, codes = raw["days"], pd.Index(raw["codes"])
    dfc = pd.DataFrame(raw["C"].T, index=days, columns=codes)
    dftv = pd.DataFrame(raw["tv"].T, index=days, columns=codes)
    dfrc = pd.DataFrame(raw["raw_close"].T, index=days, columns=codes)

    def um(**kw):
        u = universe_mask(dfc, dftv, raw["meta"], raw_close=dfrc, adv_floor=G0.ADV_FLOOR, **kw)
        return np.ascontiguousarray(u.to_numpy(dtype=bool).T)

    return um(), {G0.MEMBER_HIST_Q0: um(min_history=G0.MEMBER_HIST_Q0), N + 60: um(min_history=N + 60)}


def build_daily(con) -> dict:
    raw = G0.load(con)
    log(f"일봉 load: {raw['info']['n_codes']:,} × {raw['info']['n_days']:,} · 정지일 NaN {raw['info']['n_suspension_cells_nan']:,}")
    U, member = _daily_masks(raw)
    return _finish("daily", pd.DatetimeIndex(raw["days"]), list(raw["codes"]), raw["C"], U, member,
                   {"info": raw["info"], "names": raw["meta"]["name"].to_dict()})


def week_ends(days: pd.DatetimeIndex) -> np.ndarray:
    """ISO 주마다 마지막 거래일의 행 번호 — 마지막 주는 패널이 주 중간에 끝날 수 있어 뺀다(끝난 주만)."""
    iso = days.isocalendar()
    key = np.asarray(iso["year"], dtype=np.int64) * 100 + np.asarray(iso["week"], dtype=np.int64)
    last = np.flatnonzero(np.r_[key[1:] != key[:-1], True])
    return last[:-1]


def build_weekly(con) -> dict:
    raw = G0.load(con)
    U, member_d = _daily_masks(raw)
    days = pd.DatetimeIndex(raw["days"])
    wk = week_ends(days)
    C = np.ascontiguousarray(raw["C"][:, wk])
    cnt = np.cumsum(np.isfinite(C), axis=1)
    mem80 = member_d[N + 60][:, wk]
    member = {h: np.ascontiguousarray(mem80 & (cnt >= h)) for h in (G0.MEMBER_HIST_Q0, N + 60)}
    log(f"주봉: {C.shape[0]:,} × {C.shape[1]:,} 주 ({days[wk[0]].date()} ~ {days[wk[-1]].date()})")
    return _finish("weekly", days[wk], list(raw["codes"]), C, np.ascontiguousarray(U[:, wk]), member,
                   {"info": raw["info"], "names": raw["meta"]["name"].to_dict()})


def top_codes(con) -> tuple[list, dict]:
    """구간 일 거래대금(정규장 분봉 집계) 중앙값 상위 — 보통주·KOSPI/KOSDAQ 만. 여유분까지 순위대로."""
    from swing_it.features.korea import UNIVERSE_MARKETS, load_listing_meta
    from swing_it.krx_session import read_session_bars_agg

    tvd = read_session_bars_agg(con, start=INTRA[0], end=INTRA[1])
    med = tvd.groupby("code")["trade_value"].median()
    meta = load_listing_meta(con)
    ok = meta["excluded"].isna() & meta["market"].isin(UNIVERSE_MARKETS)
    med = med[med.index.isin(meta.index[ok])].sort_values(ascending=False)
    return [str(c) for c in med.index[: TOP_CODES + 40]], {"median_tv_mn": med.iloc[: TOP_CODES + 40].to_dict(),
                                                         "names": meta["name"].to_dict()}


def read_5m(con, codes: list) -> pd.DataFrame:
    """정규장 5분봉(09:00 ≤ ts < 15:33, KST naive 그대로) — DB 에서 묶는다. 묶음은 하루 안에서만."""
    q = ("SELECT code, time_bucket('5 minutes', ts) AS ts, first(open, ts) AS open, max(high) AS high, min(low) AS low, "
         "last(close, ts) AS close, sum(volume)::float8 AS volume FROM minute_bars "
         "WHERE code = ANY(%(codes)s) AND ts >= %(lo)s AND ts < %(hi)s AND ts::time >= %(o)s AND ts::time < %(c)s "
         "GROUP BY 1, 2 ORDER BY 1, 2")
    hi = (pd.Timestamp(INTRA[1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    for a in range(0, len(codes), 20):
        parts.append(pd.read_sql_query(q, con, params={"codes": codes[a:a + 20], "lo": INTRA[0], "hi": hi,
                                                       "o": INTRA_OPEN, "c": INTRA_CLOSE_EXCL}))
        log(f"  5분봉 {min(a + 20, len(codes))}/{len(codes)} 종목")
    m = pd.concat(parts, ignore_index=True)
    m["ts"] = pd.to_datetime(m["ts"])
    return m


def to_60m(m5: pd.DataFrame) -> pd.DataFrame:
    """5분봉 → 60분봉(정시 시작, 하루 안에서만 — 15:00 봉은 15:00~15:32)."""
    m = m5.copy()
    m["ts"] = m["ts"].dt.floor("60min")
    g = m.sort_values(["code", "ts"]).groupby(["code", "ts"], sort=True)
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                 volume=("volume", "sum")).reset_index()


def _panel(m: pd.DataFrame, codes: list) -> tuple:
    grid = pd.DatetimeIndex(np.sort(m["ts"].unique()))
    ci = pd.Index(codes).get_indexer(m["code"])
    ti = grid.get_indexer(m["ts"])
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        a = np.full((len(codes), len(grid)), np.nan)
        a[ci, ti] = m[col].to_numpy(dtype=float)
        out[col] = a
    return grid, out


def build_intraday(con, tf: str) -> dict:
    cache = SCRATCH / "intraday_5m.pkl"
    if cache.exists():
        m5, ranked, meta = pd.read_pickle(cache)
        log(f"5분봉 캐시 {cache} — {len(m5):,} 행")
    else:
        ranked, meta = top_codes(con)
        log(f"상위 후보 {len(ranked)} 종목(중앙 거래대금 {ranked[0]} … {ranked[-1]})")
        m5 = read_5m(con, ranked)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((m5, ranked, meta), cache)
    m = m5 if tf == "5m" else to_60m(m5)
    grid, P = _panel(m, ranked)
    C = P["close"]
    bad_v = np.isfinite(C) & ~(np.nan_to_num(P["volume"], nan=0.0) > 0)    # 정지일 규칙의 봉 판: 거래량 0 이면 NaN
    C[bad_v] = np.nan
    # 기업행위 의심(원 가격): 이웃한 유한 봉 사이 로그 변화가 ±30% 밖
    flagged = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(len(ranked)):
            v = np.log(C[i][np.isfinite(C[i])])
            d = np.diff(v)
            if len(v) == 0 or ((d < LIMIT_LO) | (d > LIMIT_HI)).any():
                flagged.append(i)
    keep = [i for i in range(len(ranked)) if i not in set(flagged)][:TOP_CODES]
    codes = [ranked[i] for i in keep]
    sel = np.asarray(keep)
    C = np.ascontiguousarray(C[sel])
    Op, H, L, V = (np.ascontiguousarray(P[c][sel]) for c in ("open", "high", "low", "volume"))
    fin = np.isfinite(C)
    cnt = np.cumsum(fin, axis=1)
    member = {h: np.ascontiguousarray(fin & (cnt >= h)) for h in (G0.MEMBER_HIST_Q0, N + 60)}
    per_day = pd.Series(1, index=grid).groupby(grid.normalize()).size()
    info = {"n_codes": len(codes), "n_bars": len(grid), "first": str(grid[0]), "last": str(grid[-1]),
            "n_days": int(len(per_day)), "bars_per_day_median": float(per_day.median()),
            "bars_per_day_min": int(per_day.min()), "nan_frac": float(1 - fin.mean()), "n_volume0_nan": int(bad_v[sel].sum()),
            "flagged_corporate_action": [ranked[i] for i in flagged],
            "rank_of_last_kept": int(keep[-1]) + 1}
    log(f"{tf}: {len(codes)} 종목 × {len(grid):,} 봉 · 날 {info['n_days']} · 하루 봉 중앙 {info['bars_per_day_median']:.0f} "
        f"(최소 {info['bars_per_day_min']}) · NaN {info['nan_frac']:.3%} · 기업행위 의심 제외 {len(flagged)}")
    return _finish(tf, grid, codes, C, fin.copy(), member, {"info": info, "names": meta["names"], "O": Op, "H": H, "L": L, "V": V})


# =============================================================================================== 사건·확률

def events(X: dict, k: int):
    days = X["days"]
    n = len(days)
    t = np.arange(n)
    ok = np.asarray(days >= pd.Timestamp(X["ev_lo"])) & (t + k < n)
    ok[ok] &= np.asarray(days[t[ok] + k] <= pd.Timestamp(X["ev_hi"]))
    mask = X["U"] & ~X["aligned"] & X["defined"] & ok[None, :]
    s, ci = np.nonzero(mask.T)
    return ci.astype(np.int64), s.astype(np.int64)


def _task(args):
    q, k, out_dir = args
    X = _G["X"]
    ci, s = _G["EV"][k]
    t = time.time()
    P = G0.cell_probs(X, q, N, k, ci, s, M=G0.M_PATHS)
    np.savez(Path(out_dir) / f"probs_q{q}_k{k}.npz", ci=ci, s=s, **P)
    return q, k, time.time() - t, len(ci), int(np.isfinite(P["p_align"]).sum())


def run_probs(X: dict, EV: dict, out_dir: Path, workers: int) -> None:
    _G["X"], _G["EV"] = X, EV
    G0._warmup()
    tasks = sorted([(q, k, str(out_dir)) for q in QS for k in KS], key=lambda a: -(a[1] * (a[0] + 1)))
    t0 = time.time()
    with mp.get_context("fork").Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
        for i, (q, k, secs, n_ev, n_fin) in enumerate(pool.imap_unordered(_task, tasks), 1):
            log(f"[{i}/{len(tasks)}] q={q} k={k} 사건 {n_ev:,} · 확률 유한 {n_fin:,} · {secs:.1f}s (경과 {time.time() - t0:.0f}s)")


def _hi(p, y) -> dict:
    m = p >= 0.5
    n = int(m.sum())
    return {"n": n, "p_mean": float(p[m].mean()) if n else None, "y_rate": float(y[m].mean()) if n else None}


def metrics(X: dict, k: int, out_dir: Path) -> tuple[dict, dict]:
    ci, s = _G["EV"][k]
    probs = {}
    for q in QS:
        z = np.load(out_dir / f"probs_q{q}_k{k}.npz")
        assert np.array_equal(z["ci"], ci) and np.array_equal(z["s"], s)
        probs[q] = {"p": z["p_align"], "nopool": z["nopool"]}
    y = MQ.realized(X["C"], ci, s, k)["p_align"]
    ok = np.isfinite(y)
    drops = {"n_in": int(len(y)), "label_nan": int((~ok).sum())}
    for q in QS:
        f = np.isfinite(probs[q]["p"])
        drops[f"q{q}_nopool"] = int(probs[q]["nopool"].sum())
        ok &= f
    p2, p1, p0 = (probs[q]["p"][ok] for q in QS)
    yy, cl = y[ok], X["cluster"][s[ok]]
    res = {"k": k, "n": int(ok.sum()), "drops": drops, "base_rate": float(yy.mean()),
           "n_clusters": int(len(np.unique(cl))),
           "first_event": str(X["days"][s[ok]].min()), "last_event": str(X["days"][s[ok]].max()),
           "bss_2v0": list(CAL.bss_ci(p2, p0, yy, cl)), "bss_1v0": list(CAL.bss_ci(p1, p0, yy, cl)),
           "bss_2v1": list(CAL.bss_ci(p2, p1, yy, cl)),
           "brier": {f"q{q}": CAL.brier(p, yy) for q, p in zip(QS, (p2, p1, p0))},
           "ece": {f"q{q}": CAL.ece(p, yy) for q, p in zip(QS, (p2, p1, p0))},
           "reliability": {f"q{q}": CAL.reliability(p, yy).to_dict("records") for q, p in zip(QS, (p2, p1, p0))},
           "high": {f"q{q}": _hi(p, yy) for q, p in zip(QS, (p2, p1, p0))}}
    if X["tf"] == "daily":       # 교차 확인: 관문 0 R3(정배열 아닌 유니버스 20% 표본, N=20) 와 같은 표본
        keep = G0.hash_sample(X["codes"], X["days"], ci[ok], s[ok], G0.R3_MOD)
        res["r3_check"] = {"n": int(keep.sum()), "bss_2v0": CAL._bss(p2[keep], p0[keep], yy[keep])}
    arrays = {"ci": ci[ok], "s": s[ok], "p2": p2, "p1": p1, "p0": p0, "y": yy}
    return res, arrays


# =============================================================================================== 예시

def pick_example(arr: dict) -> int:
    """P(q2) ∈ [0.5, 0.9] 인 사건 중 P(q2) − P(q0) 가 가장 큰 것(동률이면 앞) — 외삽이 정배열을 내다보는 실제 장면."""
    m = (arr["p2"] >= 0.5) & (arr["p2"] <= 0.9)
    if not m.any():
        m = np.ones(len(arr["p2"]), dtype=bool)
    gap = np.where(m, arr["p2"] - arr["p0"], -np.inf)
    return int(np.argmax(gap))


def example(X: dict, arr: dict, con_fn=None) -> dict:
    j = pick_example(arr)
    c, t = int(arr["ci"][j]), int(arr["s"][j])
    k = K_EX
    days, code = X["days"], X["codes"][c]
    fc = MQ.poly_forecast(X["y"], N, k, 2)
    z = MQ.forecast_z(X["y"], fc, X["sigma"])
    ms = MQ.month_starts(days)
    t_m = int(ms[np.searchsorted(ms, t, side="right") - 1])
    pool = MQ.month_pool(days, z, X["member"][N + 60], t_m, k, years=G0.POOL_YEARS, min_dates=G0.POOL_MIN_DATES)
    idx = MQ.draw_index(code, days[t_m], 2, N, k, len(pool), G0.M_PATHS)
    fc64 = np.ascontiguousarray(fc, dtype=np.float64)
    p_re = MQ.event_probs(X["C"], fc64, X["sigma"], pool, idx[None, :], [c], [t], k)["p_align"][0]
    if abs(p_re - arr["p2"][j]) > 1e-12:
        raise RuntimeError(f"예시 재계산 P(q2) {p_re} ≠ 저장 {arr['p2'][j]}")
    paths = np.exp(fc64[c, t, :][None, :] + X["sigma"][c, t] * pool[idx].astype(np.float64))
    yw = X["y"][c, t - N + 1: t + 1]
    sx = np.arange(-N + 1, 1) / N
    coef = np.polyfit(sx, yw, 2)
    fit = np.exp(np.polyval(coef, sx))
    fc_chk = np.exp(np.polyval(coef, np.arange(1, k + 1) / N))
    if not np.allclose(fc_chk, np.exp(fc64[c, t, :]), rtol=1e-4):
        raise RuntimeError("예시 polyfit 외삽 ≠ poly_forecast")
    lo, hi = max(t - 130, 0), min(t + k + 5, len(days) - 1)
    C = X["C"][c]
    mas = {w: pd.Series(C).rolling(w, min_periods=w).mean().to_numpy()[lo:hi + 1] for w in (5, 20, 60, 120)}
    if "O" in X:
        Op, H, L, V = (X[a][c, lo:hi + 1] for a in ("O", "H", "L", "V"))
    else:
        from swing_it.storage import read_code_bars

        con = con_fn()
        rows = read_code_bars(con, code, start=str(days[lo].date()), end=str(days[hi].date()),
                              cols=("date", "open", "high", "low", "close", "volume"), table=G0.PRICE_TABLE)
        con.close()
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["date"])
        df = df.reindex(days[lo:hi + 1])
        Op, H, L, V = (df[a].to_numpy(dtype=float) for a in ("open", "high", "low", "volume"))
        cl = df["close"].to_numpy(dtype=float)
        okc = np.isfinite(cl) & np.isfinite(C[lo:hi + 1])
        if okc.any() and not np.allclose(cl[okc], C[lo:hi + 1][okc], rtol=1e-6):
            raise RuntimeError("예시 일봉 종가가 패널과 다르다")
    return {"tf": X["tf"], "code": code, "name": X["names"].get(code), "t": str(days[t]), "t_idx": t - lo, "k": k,
            "ts": [str(d) for d in days[lo:hi + 1]], "O": Op.tolist(), "H": H.tolist(), "L": L.tolist(),
            "C": C[lo:hi + 1].tolist(), "V": V.tolist(), "ma": {str(w): v.tolist() for w, v in mas.items()},
            "fit": fit.tolist(), "fc": np.exp(fc64[c, t, :]).tolist(),
            "band10": np.quantile(paths, 0.1, axis=0).tolist(), "band90": np.quantile(paths, 0.9, axis=0).tolist(),
            "p_q2": float(arr["p2"][j]), "p_q1": float(arr["p1"][j]), "p_q0": float(arr["p0"][j]),
            "realized": float(arr["y"][j]), "pool_P": int(len(pool)), "t_m": str(days[t_m])}


# =============================================================================================== 실행

def _clean(o):
    return G0._clean(o)


def run_tf(tf: str, workers: int) -> None:
    out_dir = SCRATCH / tf
    out_dir.mkdir(parents=True, exist_ok=True)
    con = _connect()
    t = time.time()
    X = {"daily": build_daily, "weekly": build_weekly}.get(tf, lambda c: build_intraday(c, tf))(con)
    con.close()
    log(f"{tf} 패널 {X['C'].shape} · {time.time() - t:.1f}s")
    EV_ = {k: events(X, k) for k in KS}
    for k in KS:
        log(f"k={k}: 사건 {len(EV_[k][0]):,}")
    t = time.time()
    run_probs(X, EV_, out_dir, workers)
    t_probs = time.time() - t
    res, ex = {}, None
    for k in KS:
        r, arr = metrics(X, k, out_dir)
        res[str(k)] = r
        log(f"k={k}: 짝 {r['n']:,} · 기준율 {r['base_rate']:.3f} · BSS 2v0 {r['bss_2v0'][0]:+.4f} "
            f"[{r['bss_2v0'][1]:+.4f}, {r['bss_2v0'][2]:+.4f}] · 1v0 {r['bss_1v0'][0]:+.4f}"
            + (f" · R3 교차 {r['r3_check']['bss_2v0']:+.4f} n={r['r3_check']['n']:,}" if "r3_check" in r else ""))
        if k == K_EX and tf != "weekly":
            ex = example(X, arr, _connect)
    doc = {"tf": tf, "N": N, "ks": KS, "M": G0.M_PATHS, "cluster": CLUSTER[tf], "events_window": EV[tf],
           "info": X["info"], "secs_probs": t_probs, "cells": res, "label": LABEL}
    (out_dir / "result.json").write_text(json.dumps(_clean(doc), ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    if ex is not None:
        (out_dir / "example.json").write_text(json.dumps(_clean(ex), ensure_ascii=False, allow_nan=False) + "\n")
    log(f"{tf} 끝 → {out_dir}")


# =============================================================================================== 그림·문서

COL = {"q2": "#2a78d6", "q1": "#eb6834", "q0": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MA_COL = {"5": "#2a78d6", "20": "#eb6834", "60": "#1baf7a", "120": "#eda100"}
TF_TITLE = {"weekly": "주봉", "daily": "일봉", "60m": "60분봉", "5m": "5분봉"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    if not names & {"NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"}:     # 사용자 글꼴 폴더는 캐시에 안 잡힌다
        for fp in sorted(Path.home().glob(".local/share/fonts/**/NotoSansKR-*.otf")):
            font_manager.fontManager.addfont(str(fp))
        names = {f.name for f in font_manager.fontManager.ttflist}
    for fn in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if fn in names:
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def _sub(tf: str) -> str:
    return LABEL + (" · " + LABEL_INTRA if tf in ("60m", "5m") else "")


def plot_bss(R: dict, path: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, tf in zip(axes, TFS):
        r = R.get(tf)
        ax.axhline(0, color=INK, lw=1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        if r is None:
            ax.set_title(f"{TF_TITLE[tf]} — 없음")
            continue
        x = np.arange(len(KS))
        for off, key, lab, col in ((-0.1, "bss_2v0", "q2 대 q0", COL["q2"]), (0.1, "bss_1v0", "q1 대 q0", COL["q1"])):
            b = np.array([r["cells"][str(k)][key] for k in KS], dtype=float)
            ax.errorbar(x + off, b[:, 0], yerr=[b[:, 0] - b[:, 1], b[:, 2] - b[:, 0]], fmt="o", ms=7, color=col, lw=2,
                        capsize=4, label=lab)
            for xi, v in zip(x + off, b[:, 0]):
                ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points", xytext=(8 if off > 0 else -8, 0),
                            ha="left" if off > 0 else "right", va="center", fontsize=8, color=INK2)
        ax.set_xticks(x, [f"k={k}" for k in KS])
        ax.set_xlim(-0.6, len(KS) - 0.4)
        ax.set_title(f"{TF_TITLE[tf]}" + ("\n" + LABEL_INTRA if tf in ("60m", "5m") else ""), color=INK, fontsize=11)
    axes[0].set_ylabel("BSS (기준 q0 무작위 보행)  ↑ 좋음")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, loc="lower center", ncol=2)
    fig.suptitle(f"이평선 외삽 정배열 확률 — 시간대별 BSS(N=20, 95% 군집 구간) · {LABEL}", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rel(R: dict, path: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharex=True, sharey=True)
    for ax, tf in zip(axes, TFS):
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="대각선(완전 보정)")
        ax.grid(color=GRID, lw=0.8)
        r = R.get(tf)
        if r is None:
            continue
        c = r["cells"][str(K_EX)]
        for q in ("q2", "q1", "q0"):
            rel = pd.DataFrame(c["reliability"][q])
            rel = rel[rel["n"] >= 50]
            ax.plot(rel["p_mean"], rel["y_rate"], "-o", ms=5, lw=2, color=COL[q], label=f"{q}  ECE {c['ece'][q]:.3f}")
        ax.set_title(f"{TF_TITLE[tf]} k={K_EX} · 기준율 {c['base_rate']:.3f}"
                     + ("\n" + LABEL_INTRA if tf in ("60m", "5m") else ""), color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlabel("예측 확률(구간 평균)")
    axes[0].set_ylabel("실현 정배열 비율")
    fig.suptitle(f"신뢰도 곡선(구간 n ≥ 50, N=20, k={K_EX}) · {LABEL}", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_example(e: dict, path: Path) -> None:
    plt = _style()
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
    n = len(e["C"])
    x = np.arange(n)
    Op, H, L, C, V = (np.array(e[a], dtype=float) for a in ("O", "H", "L", "C", "V"))
    up = C >= Op
    ax.vlines(x, L, H, color=INK2, lw=0.8)
    body_lo, body_h = np.minimum(Op, C), np.maximum(np.abs(C - Op), np.nanmax(H) * 1e-4)
    ax.bar(x[up], body_h[up], bottom=body_lo[up], width=0.7, facecolor="white", edgecolor=INK2, lw=0.8)
    ax.bar(x[~up], body_h[~up], bottom=body_lo[~up], width=0.7, color=INK2)
    for w, v in e["ma"].items():
        ax.plot(x, v, color=MA_COL[w], lw=1.6, label=f"MA{w}")
    t, k = e["t_idx"], e["k"]
    xf = np.arange(t - N + 1, t + 1)
    ax.plot(xf, e["fit"], color="#6250d6", lw=2.4, label="q2 적합(최근 20봉)")
    xe = np.arange(t, t + k + 1)
    ax.plot(xe, [e["fit"][-1]] + e["fc"], color="#6250d6", lw=2.4, ls="--", label=f"q2 외삽 {k}봉(적합선 연장)")
    ax.fill_between(np.arange(t + 1, t + k + 1), e["band10"], e["band90"], color="#6250d6", alpha=0.18,
                    label="q2 라이브러리 10–90%")
    ax.plot(np.arange(t, t + k + 1), C[t:t + k + 1], "-o", color=INK, lw=1.8, ms=4, label="실현 종가")
    for xi, lab in ((t, "t"), (t + k, f"t+{k}")):
        ax.axvline(xi, color=INK2, lw=1, ls=":")
        ax.annotate(lab, (xi, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", color=INK2)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    lo = min(np.nanmin(L[max(t - 60, 0):]), np.min(e["band10"]))
    hi = max(np.nanmax(H[max(t - 60, 0):]), np.max(e["band90"]), np.max(e["fc"]))
    ax.set_ylim(lo * 0.97, hi * 1.03)
    ax.set_xlim(max(t - 60, 0) - 1, n)
    axv.bar(x, V, width=0.7, color=np.where(up, "#b9b8b3", INK2))
    axv.set_ylabel("거래량")
    step = max(1, (n - max(t - 60, 0)) // 8)
    ticks = list(range(max(t - 60, 0), n, step))
    fmt = (lambda s: s[:10]) if e["tf"] == "daily" else (lambda s: s[5:16])
    axv.set_xticks(ticks, [fmt(e["ts"][i]) for i in ticks], rotation=0, fontsize=8)
    real = "정배열" if e["realized"] == 1 else "정배열 아님"
    ax.set_title(f"{TF_TITLE[e['tf']]} {e['code']} {e['name'] or ''} · t={e['t'][:10] if e['tf'] == 'daily' else e['t'][:16]} · P(q2)={e['p_q2']:.2f} · "
                 f"P(q0)={e['p_q0']:.2f} · 실현(t+{k}) {real}\n{_sub(e['tf'])}", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _f(v, fmt="+.3f"):
    return "—" if v is None else format(v, fmt)


def write_md(R: dict, E: dict, path: Path) -> None:
    L = ["# 이평선 외삽 — 시간대별 보정 사후 진단", "", f"**{LABEL}.** 관문 0(일봉, 20칸 불합격 — `../RESULTS.md`)을 다시 여는 것이 아니다. "
         "같은 부품(`research/signals/ma_quad.py`·`ma_quad_calib.py`·`maquad_calib.cell_probs`)으로 봉 단위만 바꿔 그린 기술 진단이다. "
         f"60분·5분봉은 **{LABEL_INTRA}** — 봉인 구간(2024-01~)이 이미 소모됐다.", "",
         "공통: 사건 = 정배열 아님(봉 단위 MA5>MA20>MA60>MA120 아님) ∧ 네 이평선 정의 ∧ 유니버스, EPS·200일선·d 거름 없음. "
         f"N=20, M={G0.M_PATHS}, q=2·1·0 같은 사건 집합. BSS 기준 = q0(무작위 보행). 구간 = 군집 복원추출 95%"
         "(주·일 = 달력 월, 분봉 = 거래일). 높은 구간 = P ≥ 0.5 사건의 평균 예측 → 실현율.", "", "## 자료", ""]
    for tf in TFS:
        r = R.get(tf)
        if r is None:
            L.append(f"- {TF_TITLE[tf]}: 실행 안 됨")
            continue
        i = r["info"]
        c = r["cells"][str(K_EX)]
        if tf in ("daily", "weekly"):
            src = (f"`maquad_calib.load`({i['first_day']} ~ {i['last_day']}, 정지일 NaN {i['n_suspension_cells_nan']:,} 칸), "
                   "유니버스 ADV20 ≥ 10억")
            if tf == "weekly":
                src += "의 ISO 주 마지막 거래일(끝난 주만)"
        else:
            src = (f"`minute_bars` 09:00 ≤ ts < 15:33, {i['first'][:10]} ~ {i['last'][:10]}, {i['n_codes']} 종목 × {i['n_bars']:,} 봉"
                   f"(거래일 {i['n_days']}, 하루 봉 중앙 {i['bars_per_day_median']:.0f}), 봉 NaN {i['nan_frac']:.2%}, "
                   f"가격제한폭 밖 변화로 뺀 종목 {len(i['flagged_corporate_action'])}(채운 순위 끝 {i['rank_of_last_kept']})")
        w = 10 if tf in ("daily", "weekly") else 16
        L.append(f"- {TF_TITLE[tf]}: {src}. 사건 {c['first_event'][:w]} ~ {c['last_event'][:w]}, 군집 {c['n_clusters']:,}.")
    L += ["", "## 표", "",
          "| 시간대 | k | n 사건 | 기준율 | BSS q2v0 [95%] | BSS q1v0 [95%] | ECE q0 / q2 | P≥0.5 q2: 예측→실현 (n) | P≥0.5 q0: 예측→실현 (n) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tf in TFS:
        r = R.get(tf)
        if r is None:
            continue
        for k in KS:
            c = r["cells"][str(k)]
            b2, b1 = c["bss_2v0"], c["bss_1v0"]
            h2, h0 = c["high"]["q2"], c["high"]["q0"]
            L.append(f"| {TF_TITLE[tf]} | {k} | {c['n']:,} | {c['base_rate']:.3f} | {b2[0]:+.3f} [{b2[1]:+.3f}, {b2[2]:+.3f}] | "
                     f"{b1[0]:+.3f} [{b1[1]:+.3f}, {b1[2]:+.3f}] | {c['ece']['q0']:.3f} / {c['ece']['q2']:.3f} | "
                     f"{_f(h2['p_mean'], '.3f')}→{_f(h2['y_rate'], '.3f')} ({h2['n']:,}) | "
                     f"{_f(h0['p_mean'], '.3f')}→{_f(h0['y_rate'], '.3f')} ({h0['n']:,}) |")
    d = R.get("daily")
    if d is not None and "r3_check" in d["cells"][str(K_EX)]:
        rc = {k: d["cells"][str(k)]["r3_check"] for k in KS}
        L += ["", "교차 확인(일봉): 관문 0 R3 과 같은 20% 표본(`blake2b(code|date) % 5 == 0`)으로 자르면 BSS q2v0 = "
              + ", ".join(f"k={k} {rc[k]['bss_2v0']:+.4f}(n={rc[k]['n']:,})" for k in KS)
              + " — `../RESULTS.md` R3 N=20 행(−0.4583, −0.3691, −0.5019)과 비교."]
    L += ["", "## 그림", "", "![BSS](bss_by_timeframe.png)", "", "![신뢰도](reliability_by_timeframe.png)", ""]
    for tf in ("daily", "60m", "5m"):
        if tf in E:
            e = E[tf]
            L += [f"![{TF_TITLE[tf]} 예시](example_{tf}.png)", "",
                  f"{TF_TITLE[tf]} 예시 — {e['code']} {e['name'] or ''} t={e['t'][:16]}: P(q2) {e['p_q2']:.3f}, P(q1) {e['p_q1']:.3f}, "
                  f"P(q0) {e['p_q0']:.3f}, 실현 {'정배열' if e['realized'] == 1 else '정배열 아님'}. 고른 규칙: P(q2) ∈ [0.5, 0.9] 중 "
                  "P(q2) − P(q0) 최대(대표 표본이 아니라 외삽이 가장 앞서 나간 장면). 띠 = 그 달 q2 라이브러리 200 경로의 10–90%.", ""]
    beat = [f"{TF_TITLE[tf]} k={k}" for tf in TFS if tf in R for k in KS if R[tf]["cells"][str(k)]["bss_2v0"][1] > 0]
    beat_pt = [f"{TF_TITLE[tf]} k={k}" for tf in TFS if tf in R for k in KS if R[tf]["cells"][str(k)]["bss_2v0"][0] > 0]
    L += ["## 읽기(사실만)", "",
          f"- q2 가 q0 를 이긴 칸(BSS 95% 하한 > 0): {', '.join(beat) if beat else '없음'}. 점추정만 > 0: {', '.join(beat_pt) if beat_pt else '없음'}.",
          f"- {LABEL}. 분봉 숫자는 {LABEL_INTRA}. 매매 권고 없음.", "",
          "## 주의", "",
          "- 주봉 사건은 2017-05 부터다 — 풀이 3년 창 안 원점 날짜 ≥ 120(주봉이면 120 주)을 요구해 그 전 달은 풀이 없다(q 별 `nopool`).",
          "- 분봉 종목은 구간 전체의 일 거래대금 중앙값으로 고른다(구간 뒤 정보 — 진단용). 중앙값은 거래한 날로만 잡혀 구간 중 상장·수집 "
          "시작 종목이 들어온다. 봉 NaN 의 대부분(후보 340 종목 5분봉 실측 91%)이 그런 종목의 첫 봉 이전 칸이다. 분봉은 원 가격(분할 미조정)이다 — "
          "가격제한폭 밖 변화가 있는 종목은 뺐지만 배당락 같은 작은 조정은 남는다. 수능일·새해 첫날처럼 장 시간이 바뀐 날도 09:00~15:32 만 쓴다.",
          "- 분봉 BSS 구간은 거래일 군집이라 날 사이 계열상관을 다 담지 못한다(구간이 좁게 나온다). 일·주봉은 달력 월 군집.",
          "- 예시 세 장은 고른 규칙상 극단 장면이다(대표 표본 아님). 그림의 q2 적합선은 종가가 아니라 적합값이라 t 에서 종가와 떨어질 수 있다.",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")


def report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    R, E = {}, {}
    for tf in TFS:
        p = SCRATCH / tf / "result.json"
        if p.exists():
            R[tf] = json.loads(p.read_text())
        pe = SCRATCH / tf / "example.json"
        if pe.exists():
            E[tf] = json.loads(pe.read_text())
    plot_bss(R, OUT / "bss_by_timeframe.png")
    plot_rel(R, OUT / "reliability_by_timeframe.png")
    for tf, e in E.items():
        plot_example(e, OUT / f"example_{tf}.png")
    write_md(R, E, OUT / "TF_DIAG.md")
    log(f"보고 → {OUT}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tf", choices=TFS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    if args.tf:
        run_tf(args.tf, args.workers)
    if args.report:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
