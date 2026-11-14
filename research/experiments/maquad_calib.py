"""관문 0 실행기 — 이평선 q차식 외삽 정배열 확률의 보정·기술(PREREG.md 관문 0 + 정정 MQ-1~MQ-4, 결과 전 고정).

판정: q=2 의 N{10,15,20,30,40} × k{3,5,7,10} 20칸, 대 B0(q=0) — `ma_quad_calib.cell_verdict` → `gate_verdict`.
보고(판정 무관): q=1 같은 20칸(대 B0), q=2 대 B1(q=1) BSS, 해상도(P ≥ 0.5 실현율), R1 200일선 거름 없는 원래 사전 후보,
R2 템플릿 충족 부분집합, R3 정배열 아닌 유니버스 전 종목-일(결정적 20% 표본), 쌍별 P_up 3키.

실행 규약(`research/logs/ma_quad/PREREG.md` — 정정 MQ-1 200일선 거름·템플릿 보고, MQ-2 정지일 결측·폐지 종목 가격·경계 여유,
MQ-3 관문 1 단순화, MQ-4 라이브러리 모집단 q=0 상장 ≥ 70일·짝짓기·R3 20% 표본·session="raw"·실행 출처):
- 같은 칸 (N, k) 의 q=2·1·0 은 **같은 사건 집합**(세 q 모두 확률 유한 ∧ 라벨 유한) — 빠진 수와 까닭을 적는다.
- 사건 t: 2017-01-02 ≤ t, t+k 거래일 ≤ 2023-12-28(라벨이 2024 가격을 쓰지 않게). 사건이 쓰는 것은 전부 t 까지, 라벨만 t+k.
- 라이브러리 모집단: 위생 ∧ ADV20 ≥ 10억 ∧ 이력 ≥ N+60(q=0 은 70). 풀은 달 첫 거래일 t_m 기준(ma_quad.month_pool).
- 첫 실제 실행이 곧 판정이다 — 정본 ``research/logs/ma_quad/`` 로만, ``research/ src/ tests/`` 가 깨끗한 작업 트리(커밋된 코드)에서만,
  결과 파일이나 원장 gate0 이 있으면 다시 돌지 않는다(`guard_rerun`). 출력은 메모리에서 다 만든 뒤 JSON → MD → 원장 순으로 쓴다.
  ``--smoke`` 는 모양·사건 수·시간만 출력한다.

사용(실제 실행은 컨트롤러가 한 번):
``set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --with numba --extra pg python research/experiments/maquad_calib.py [--smoke] [--workers 16] [--no-m-sens]``
(실제 실행은 ``--m-sens`` 가 기본 켬 — 한 번뿐이라 같이 낸다.)
"""
from __future__ import annotations

import os

# numba 를 부르기 전에(ma_quad 가 import 때 부른다) — 다중 작업자 스레드 경합 방지(CLAUDE.md 실행 환경 함정)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path[:0] = [".", "src"]
from research.signals import ma_quad as MQ  # noqa: E402
from research.signals import ma_quad_calib as CAL  # noqa: E402
from swing_it.features.korea import universe_mask  # noqa: E402
from swing_it.units import eok  # noqa: E402

# --- 사전등록 고정값 ------------------------------------------------------------------------------------------
DATA = ("2013-06-01", "2024-01-31")
PRICE_TABLE = "daily_bars_adj_v2"
EVENT_START = "2017-01-02"
LABEL_END = "2023-12-28"
NS = (10, 15, 20, 30, 40)
KS = (3, 5, 7, 10)
QS = (2, 1, 0)
M_PATHS = 200
D_MAX = 0.10
ADV_FLOOR = eok(10)            # ADV20 ≥ 10억(백만원 단위 1,000)
MEMBER_HIST_Q0 = 70            # q=0 라이브러리 이력 하한 = 가장 작은 N(10) + 60 (규정 C)
SIGMA_W = 60
POOL_YEARS = 3
POOL_MIN_DATES = 120
R3_MOD = 5                     # R3 = blake2b(code|date) % 5 == 0 (20%)
MSENS = {"q": 2, "N": 20, "k": 5, "mod": 100, "salt": "msens|", "M": (200, 1000)}
EPS_SOURCE = "earnings"
KEYS = MQ._PROB_KEYS
POPS = ("judged", "r1", "r2", "r3")

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "research" / "logs" / "ma_quad"
SCRATCH = Path.home() / ".cache" / "cc-swing" / "scratch" / "ma_quad"

RULINGS = {
    "session": "read_prices_wide(table=daily_bars_adj_v2, session='raw') — 창 끝 2024-01-31 이 KRX 애프터마켓 오염(2026-09-14~) 전이라 "
               "정규장 보정과 내용이 같다. 이 표는 session='regular' 를 받지 않는다(storage._resolve_session ValueError).",
    "pre_listing": "close_src 는 긴 정문 read_prices(cols=code,date,close_src, session='raw') 로 읽어 pre_listing 칸을 전 패널 NaN — "
                   "그 칸의 종가가 이미 NaN 인지 검사하고 아니면 멈춘다.",
    "raw_close": "1,000원 규칙 가격 = raw_close, 없으면(폐지 종목 naver_adj 는 raw_close 가 전부 NULL) 수정 종가 — indep30 load 선례. "
                 "raw_close 를 그대로 넘기면 폐지 종목이 유니버스에서 전부 빠진다(생존편향).",
    "pairing": "칸 (N,k) 마다 q=2·1·0 확률이 모두 유한하고 라벨이 유한한 사건만 — 같은 사건 집합.",
    "member": "라이브러리 모집단 universe_mask(adv_floor=10억, min_history=N+60), q=0 은 min_history=70.",
    "r3": "R3 = 정배열 아닌 유니버스 종목-일의 결정적 20% 표본(blake2b(code|date) % 5 == 0).",
    "label_end": "t+k 거래일 ≤ 2023-12-28, t ≥ 2017-01-02.",
    "suspension": "정정 MQ-2: 거래량 0 또는 결측인 날의 종가(와 원 종가)를 load 맨 앞에서 NaN — 적합·σ·이평선·라벨 창에 끼면 신호·라벨 없음. "
                  "평평한 종가 유지판 보고 열은 내지 않는다. 수는 data.n_suspension_cells_nan.",
    "eps_tolerance": "판정 문턱 비교는 부동소수 오차 1e-12 만큼 너그럽다(등호 포함 — ma_quad_calib._EPS).",
}

_G: dict = {}
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# =============================================================================================== 자료

def load(con) -> dict:
    """가격 패널 (n_codes, n_days) — 종가·거래대금(백만원)·1,000원 규칙 가격, 메타. 지표 없음."""
    from swing_it.features.korea import load_listing_meta
    from swing_it.storage import read_prices, read_prices_wide

    # session="raw": 창 끝(2024-01-31)이 애프터마켓 오염(2026-09-14~) 전이라 정규장과 같고, 이 표는 regular 를 받지 않는다.
    p = read_prices_wide(con, cols=("close", "trade_value", "raw_close", "volume"), table=PRICE_TABLE, start=DATA[0], end=DATA[1],
                         session="raw")
    days = pd.DatetimeIndex(p["close"].index)
    codes = pd.Index([str(c) for c in p["close"].columns])

    def arr(c):
        return np.array(p[c].to_numpy(dtype=float).T, dtype=np.float64, order="C", copy=True)   # 쓰기 가능한 사본

    C, TV, RC, V = arr("close"), arr("trade_value"), arr("raw_close"), arr("volume")
    p = None                                           # 와이드 DataFrame 해제
    C_src = C.copy()                                   # pre_listing 검사용 원 종가(정지일 NaN 전 — 검사가 헛돌지 않게)
    # 정정 MQ-2: 정지일(거래량 0 또는 결측)의 종가는 NaN — 맨 먼저. 그 날이 적합·σ·이평선·라벨 창에 끼면 신호·라벨 없음.
    # 원 종가(1,000원 규칙 원천)도 같이 비운다. 거래대금은 종가 파생이 아니라 그대로(0 이면 ADV20 평균에 0 으로 든다).
    n_susp = mask_suspensions(C, V, RC)
    del V
    cs = read_prices(con, cols=("code", "date", "close_src"), table=PRICE_TABLE, start=DATA[0], end=DATA[1],
                     session="raw", parse_dates=True)
    pl = cs.loc[cs["close_src"] == "pre_listing", ["code", "date"]]
    del cs
    di = days.get_indexer(pd.to_datetime(pl["date"]))
    cj = codes.get_indexer(pl["code"].astype(str))
    if (di < 0).any() or (cj < 0).any():
        raise RuntimeError("pre_listing 행이 와이드 패널 격자 밖이다")
    if np.isfinite(C_src[cj, di]).any():
        raise RuntimeError(f"pre_listing 칸 {int(np.isfinite(C_src[cj, di]).sum())} 개에 종가가 있다 — 자료가 바뀌었다, 멈춘다")
    del C_src
    for a in (C, TV, RC):
        a[cj, di] = np.nan
    n_nonpos = int((C <= 0).sum())
    C[C <= 0] = np.nan
    # 1,000원 규칙 가격: 원 종가, 없으면(폐지 종목) 수정 종가 — RULINGS["raw_close"]
    n_rc_fill = int((~np.isfinite(RC) & np.isfinite(C)).sum())
    PX = np.where(np.isfinite(RC), RC, C)
    meta = load_listing_meta(con)
    info = {"table": PRICE_TABLE, "start": DATA[0], "end": DATA[1], "first_day": str(days[0].date()),
            "last_day": str(days[-1].date()), "n_codes": len(codes), "n_days": len(days),
            "n_pre_listing_cells": len(pl), "n_suspension_cells_nan": n_susp, "n_close_nonpositive": n_nonpos,
            "n_price_rule_from_adj_close": n_rc_fill}
    return {"days": days, "codes": list(codes), "C": C, "raw_close": PX, "tv": TV, "meta": meta, "info": info}


def mask_suspensions(C: np.ndarray, V: np.ndarray, *derived: np.ndarray) -> int:
    """정정 MQ-2: 종가가 유한한데 거래량이 0 또는 결측인 칸을 C 와 ``derived`` 에서 제자리 NaN. NaN 으로 만든 칸 수."""
    bad = np.isfinite(C) & ~(np.nan_to_num(V, nan=0.0) > 0)
    for a in (C, *derived):
        a[bad] = np.nan
    return int(bad.sum())


def hash_sample(codes, days, ci, s, mod: int, salt: str = "") -> np.ndarray:
    """결정적 표본: blake2b(salt + code|YYYY-MM-DD) % mod == 0."""
    dstr = pd.DatetimeIndex(days).strftime("%Y-%m-%d")

    def h(c, t):
        key = f"{salt}{codes[c]}|{dstr[t]}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % mod == 0

    return np.fromiter((h(c, t) for c, t in zip(ci, s)), dtype=bool, count=len(ci))


def prepare(X: dict, eps_fn, ev_lo: str = EVENT_START, ev_hi: str = LABEL_END) -> dict:
    """상태·유니버스·라이브러리 모집단·σ·EPS↑·R3 표본. ``eps_fn(ci, s) -> bool`` (사전 후보 기초 칸에만 부른다)."""
    X = dict(X)
    days, codes = pd.DatetimeIndex(X["days"]), list(X["codes"])
    C = np.ascontiguousarray(X["C"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(C > 0, np.log(C), np.nan)
    st = MQ.ma_state(C)
    aligned, d, f200, tmpl = st["aligned"], st["d"], st["f200"], st["tmpl"]
    del st
    idx = pd.Index(codes)
    dfc = pd.DataFrame(C.T, index=days, columns=idx)
    dftv = pd.DataFrame(np.asarray(X["tv"], dtype=float).T, index=days, columns=idx)
    dfrc = pd.DataFrame(np.asarray(X["raw_close"], dtype=float).T, index=days, columns=idx)

    def um(**kw):
        u = universe_mask(dfc, dftv, X["meta"], raw_close=dfrc, adv_floor=ADV_FLOOR, **kw)
        return np.ascontiguousarray(u.to_numpy(dtype=bool).T)

    U = um()
    member = {mh: um(min_history=mh) for mh in sorted({MEMBER_HIST_Q0} | {n + 60 for n in NS})}
    dfc = dftv = dfrc = None                           # 해제(um 은 위에서 다 불렀다)
    sigma = MQ.rolling_sigma(y, SIGMA_W)
    in_win = np.asarray((days >= pd.Timestamp(ev_lo)) & (days <= pd.Timestamp(ev_hi)))
    with np.errstate(invalid="ignore"):
        dok = (d > 0) & (d <= D_MAX)
    base = U & ~aligned & dok & in_win[None, :]
    ci, s = np.nonzero(base)
    eps = np.zeros_like(U)
    if len(ci):
        eps[ci, s] = np.asarray(eps_fn(ci, s), dtype=bool)
    r3b = U & ~aligned & in_win[None, :]
    ci3, s3 = np.nonzero(r3b)
    keep = hash_sample(codes, days, ci3, s3, R3_MOD)
    r3 = np.zeros_like(U)
    r3[ci3[keep], s3[keep]] = True
    X.update(days=days, codes=codes, C=C, y=y, sigma=sigma, U=U, member=member, aligned=aligned, d=d, f200=f200, tmpl=tmpl,
             eps=eps, r3=r3, ev_lo=ev_lo, ev_hi=ev_hi)
    return X


# =============================================================================================== 사건

def _tvalid(X: dict, k: int) -> np.ndarray:
    days = pd.DatetimeIndex(X["days"])
    n = len(days)
    t = np.arange(n)
    ok = np.asarray(days >= pd.Timestamp(X.get("ev_lo", EVENT_START))) & (t + k < n)
    hi = pd.Timestamp(X.get("ev_hi", LABEL_END))
    ok[ok] &= np.asarray(days[t[ok] + k] <= hi)
    return ok


def _pop_mask(X: dict, k: int, pop: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        dok = (X["d"] > 0) & (X["d"] <= D_MAX)
    m = X["U"] & X["eps"] & ~X["aligned"] & dok
    if pop in ("judged", "tmpl"):
        m = m & X["f200"]
    if pop == "tmpl":
        m = m & X["tmpl"]
    elif pop not in ("judged", "no_f200"):
        raise ValueError(f"모집단 {pop!r}")
    return m & _tvalid(X, k)[None, :]


def _ordered(mask: np.ndarray):
    s, ci = np.nonzero(mask.T)                         # 날짜 순, 같은 날 안은 종목 순
    return ci.astype(np.int64), s.astype(np.int64)


def build_events(X: dict, k: int, pop: str = "judged"):
    """사전 후보 (ci, s). judged = U ∧ EPS↑ ∧ ¬정배열 ∧ 0 < d ≤ 0.10 ∧ 200일 거름(MQ-1) ∧ 날짜 창.
    no_f200 = 200일 거름 없는 원래 사전 후보(R1), tmpl = judged ∧ 미너비니 템플릿(R2)."""
    return _ordered(_pop_mask(X, k, pop))


def build_events_all(X: dict, k: int):
    """R3: 정배열 아닌 유니버스 종목-일(20% 결정적 표본) ∧ 날짜 창 — 보고용."""
    return _ordered(X["r3"] & _tvalid(X, k)[None, :])


def events_union(X: dict, k: int) -> dict:
    """R1 ∪ R3 한 번에 계산(judged·R2 ⊆ R1) — 사건별 확률은 모집단과 무관하므로 부분집합으로 잘라 같다."""
    m1 = _pop_mask(X, k, "no_f200")
    m3 = X["r3"] & _tvalid(X, k)[None, :]
    ci, s = _ordered(m1 | m3)
    r1 = m1[ci, s]
    judged = r1 & X["f200"][ci, s]
    return {"ci": ci, "s": s, "judged": judged, "r1": r1, "r2": judged & X["tmpl"][ci, s], "r3": m3[ci, s]}


# =============================================================================================== 확률

def cell_probs(X: dict, q: int, N: int, k: int, ci, s, M: int = M_PATHS) -> dict:
    """사건 (ci, s) 의 4키 확률(NaN = 계산 불가)과 ``nopool``(그 달 풀 없음). q=0 은 N 무관(적합 N=1, 색인 키 N=0)."""
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    days, codes = X["days"], X["codes"]
    n_fit, n_key = (N, N) if q > 0 else (1, 0)
    member = X["member"][MEMBER_HIST_Q0 if q == 0 else N + 60]
    fc = MQ.poly_forecast(X["y"], n_fit, k, q)
    z = MQ.forecast_z(X["y"], fc, X["sigma"])
    fc64 = np.ascontiguousarray(fc, dtype=np.float64)
    del fc
    out = {key: np.full(len(ci), np.nan) for key in KEYS}
    nopool = np.zeros(len(ci), dtype=bool)
    if len(ci) == 0:
        return out | {"nopool": nopool}
    ms = MQ.month_starts(days)
    tm = ms[np.searchsorted(ms, s, side="right") - 1]
    for t_m in np.unique(tm):
        sel = np.flatnonzero(tm == t_m)
        pool = MQ.month_pool(days, z, member, int(t_m), k, years=POOL_YEARS, min_dates=POOL_MIN_DATES)
        if pool is None or len(pool) == 0:
            nopool[sel] = True
            continue
        draws: dict = {}
        for c in np.unique(ci[sel]):                    # 같은 달 같은 종목 = 같은 색인(공통 난수)
            draws[int(c)] = MQ.draw_index(codes[c], days[t_m], q, n_key, k, len(pool), M)
        idx = np.stack([draws[int(c)] for c in ci[sel]])
        r = MQ.event_probs(X["C"], fc64, X["sigma"], pool, idx, ci[sel], s[sel], k)
        for key in KEYS:
            out[key][sel] = r[key]
    return out | {"nopool": nopool}


def probs_path(out_dir, q: int, N: int, k: int) -> Path:
    return Path(out_dir) / f"probs_q{q}_N{N if q > 0 else 0}_k{k}.npz"


def save_probs(path, P: dict, ci, s) -> None:
    np.savez(path, ci=ci, s=s, **P)


def load_probs(path, ev: dict) -> dict:
    z = np.load(path)
    if not (np.array_equal(z["ci"], ev["ci"]) and np.array_equal(z["s"], ev["s"])):
        raise RuntimeError(f"{path}: 사건 집합이 다르다")
    return {key: z[key] for key in (*KEYS, "nopool")}


# =============================================================================================== 짝짓기·판정

def pair(probs: dict, y) -> tuple:
    """세 q 모두 유한 ∧ 라벨 유한인 사건 마스크와 빠진 수(q 별 풀 없음·그 밖 NaN, 라벨 NaN — 겹치면 각각 센다)."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    drops = {"n_in": int(len(y)), "label": int((~ok).sum())}
    for q in QS:
        P = probs[q]
        f = np.isfinite(P["p_align"])
        for key in KEYS[1:]:
            if not np.array_equal(np.isfinite(P[key]), f):
                raise RuntimeError(f"q{q} {key}: 유한 마스크가 p_align 과 다르다")
        drops[f"q{q}_nopool"] = int(P["nopool"].sum())
        drops[f"q{q}_nan"] = int((~f & ~P["nopool"]).sum())
        ok &= f
    drops["n_paired"] = int(ok.sum())
    return ok, drops


def _resolution(p, y) -> dict:
    m = p >= 0.5
    n = int(m.sum())
    return {"n": n, "y_rate": float(y[m].mean()) if n else float("nan"), "p_mean": float(p[m].mean()) if n else float("nan")}


def block(pq: dict, y, month, year) -> dict:
    """한 모집단·한 키의 표: 판정형(2v0·1v0), BSS 2v1·1v0(월 군집 95%), Brier·ECE·신뢰도·해상도 q 별."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return {"n": 0}
    p2, p1, p0 = (np.asarray(pq[q], dtype=float) for q in QS)
    return {
        "n": n, "base_rate": float(y.mean()),
        "verdict_2v0": CAL.cell_verdict(p2, p0, y, month, year),
        "verdict_1v0": CAL.cell_verdict(p1, p0, y, month, year),
        "bss_2v1": list(CAL.bss_ci(p2, p1, y, month)),
        "bss_1v0": list(CAL.bss_ci(p1, p0, y, month)),
        "brier": {f"q{q}": CAL.brier(p, y) for q, p in zip(QS, (p2, p1, p0))},
        "ece": {f"q{q}": CAL.ece(p, y) for q, p in zip(QS, (p2, p1, p0))},
        "reliability": {f"q{q}": CAL.reliability(p, y).to_dict("records") for q, p in zip(QS, (p2, p1, p0))},
        "resolution": {f"q{q}": _resolution(p, y) for q, p in zip(QS, (p2, p1, p0))},
        "yearly_bss": {"2v0": _yearly(p2, p0, y, year), "2v1": _yearly(p2, p1, y, year)},   # 보고 전용 — 판정에 안 쓴다
    }


def _yearly(pA, pB, y, year) -> dict:
    """연도별 BSS(점추정) — 사건 없는 해는 NaN(→ JSON null)."""
    year = np.asarray(year)
    out = {}
    for yr in CAL.YEARS:
        m = year == yr
        out[int(yr)] = CAL._bss(pA[m], pB[m], y[m]) if m.any() else float("nan")
    return out


def assemble_cell(X: dict, N: int, k: int, ev: dict, Y: dict, probs: dict, out_dir) -> dict:
    """짝짓기 → 칸 배열 npz 저장 → 판정·보고 표."""
    days = pd.DatetimeIndex(X["days"])
    s = ev["s"]
    year = np.asarray(days.year[s], dtype=np.int64)
    month = year * 100 + np.asarray(days.month[s], dtype=np.int64)
    y = np.asarray(Y["p_align"], dtype=float)
    for key in KEYS[1:]:
        if not np.array_equal(np.isfinite(Y[key]), np.isfinite(y)):
            raise RuntimeError(f"라벨 유한 마스크가 {key} 와 p_align 에서 다르다")
    ok, _ = pair(probs, y)
    drops, cand, n_ev = {}, {}, {}
    for pop in POPS:
        m = ev[pop]
        _, drops[pop] = pair({q: {kk: v[m] for kk, v in probs[q].items()} for q in QS}, y[m])
        cand[pop] = int(m.sum())
        n_ev[pop] = int((m & ok).sum())
    arrays = {"ci": ev["ci"][ok], "s": s[ok], "month": month[ok], "year": year[ok]}
    arrays |= {pop: ev[pop][ok] for pop in POPS}
    arrays |= {f"q{q}_{key}": probs[q][key][ok] for q in QS for key in KEYS}
    arrays |= {f"y_{key}": np.asarray(Y[key], dtype=float)[ok] for key in KEYS}
    np.savez(Path(out_dir) / f"cell_N{N}_k{k}.npz", **arrays)

    def blk(pop, key="p_align"):
        m = arrays[pop]
        return block({q: arrays[f"q{q}_{key}"][m] for q in QS}, arrays[f"y_{key}"][m], arrays["month"][m], arrays["year"][m])

    judged = blk("judged")
    return {
        "N": N, "k": k, "n_candidates": cand, "n_events": n_ev, "drops": drops,
        "judged": judged, "verdict": judged.get("verdict_2v0", _empty_verdict()),
        "verdict_q1": judged.get("verdict_1v0", _empty_verdict()),
        "reports": {pop: blk(pop) for pop in ("r1", "r2", "r3")},
        "p_up": {key: blk("judged", key) for key in KEYS[1:]},
    }


def _empty_verdict() -> dict:
    return {"pass": False, "n": 0}


def run_cell(X: dict, N: int, k: int, out_dir) -> dict:
    """한 칸 (N, k) 을 순차로(q=2·1·0 확률 → npz → 짝짓기·판정) — 테스트·디버그용. 본 실행은 작업자 분배."""
    ev = events_union(X, k)
    Y = MQ.realized(X["C"], ev["ci"], ev["s"], k)
    probs = {}
    for q in QS:
        P = cell_probs(X, q, N, k, ev["ci"], ev["s"])
        save_probs(probs_path(out_dir, q, N, k), P, ev["ci"], ev["s"])
        probs[q] = P
    return assemble_cell(X, N, k, ev, Y, probs, out_dir)


def summarize(cells: dict) -> dict:
    """칸 dict {(N,k): cell} → 관문 판정(q=2) + 보고 관문(q=1, R1·R2·R3 — 판정 무관)."""
    def gate(get):
        return CAL.gate_verdict({c: bool(get(cells[c]).get("pass", False)) for c in cells}, Ns=NS, ks=KS)

    return {
        "gate": gate(lambda c: c["verdict"]),
        "report_gate_q1": gate(lambda c: c["verdict_q1"]),
        "report_gate_r1": gate(lambda c: c["reports"]["r1"].get("verdict_2v0", {})),
        "report_gate_r2": gate(lambda c: c["reports"]["r2"].get("verdict_2v0", {})),
        "report_gate_r3": gate(lambda c: c["reports"]["r3"].get("verdict_2v0", {})),
        "cells": {f"N{N}_k{k}": cells[(N, k)] for N in NS for k in KS},
    }


# =============================================================================================== 출력

def _key(k) -> str:
    return ",".join(map(str, k)) if isinstance(k, tuple) else str(k)


def _clean(o):
    if isinstance(o, dict):
        return {_key(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (pd.Timestamp, _dt.date)):
        return o.isoformat()
    return o


def write_json(obj, path) -> None:
    """엄격 JSON(NaN·inf → null, allow_nan=False)."""
    Path(path).write_text(json.dumps(_clean(obj), allow_nan=False, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def append_trial(path, rec: dict) -> None:
    """원장에 기록 한 줄 덧붙이기 — 기존 줄은 건드리지 않는다."""
    append_line(path, trial_line(rec))


OUT_JSON, OUT_MD, OUT_TRIALS = "calib_results.json", "RESULTS.md", "TRIALS.jsonl"


def _already(logs_dir) -> list[str]:
    """이미 있는 관문 0 출력(결과 JSON·MD 파일 — 반쯤 쓴 것 포함, 원장의 gate0 줄)."""
    logs_dir = Path(logs_dir)
    found = [n for n in (OUT_JSON, OUT_MD) if (logs_dir / n).exists()]
    t = logs_dir / OUT_TRIALS
    if t.exists():
        for line in t.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("trial") == "gate0":
                found.append(f"{OUT_TRIALS}(gate0)")
                break
    return found


def guard_rerun(logs_dir=LOGS) -> None:
    """관문 0 은 한 번만 — 결과 파일(반쯤 쓴 것 포함)이나 원장 gate0 줄이 있으면 멈춘다."""
    found = _already(logs_dir)
    if found:
        raise SystemExit(f"{logs_dir}: {found} 가 있다 — 관문 0 은 이미 돌았다(재실행 = 새 시도, 사전등록 필요)")


def trial_line(rec: dict) -> str:
    """원장 한 줄(엄격 JSON, 줄바꿈 없음)."""
    return json.dumps(_clean(rec), allow_nan=False, ensure_ascii=False)


def append_line(path, line: str) -> None:
    """원장에 한 줄 덧붙이기(mode "a"). 끝 줄바꿈이 없으면 멈춘다(두 줄이 붙는다)."""
    path = Path(path)
    if "\n" in line:
        raise ValueError("원장 줄에 줄바꿈이 있다")
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                raise ValueError(f"{path}: 끝 줄바꿈이 없다 — 덧붙이지 않는다")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_new(path, text: str) -> None:
    """새 파일로만 쓴다(``"x"`` — 있으면 FileExistsError). 열자마자 파일이 생기므로 도중 실패도 재실행을 막는다."""
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def write_outputs(logs_dir, json_text: str, md_text: str, trial: str) -> None:
    """메모리에서 다 만든 세 출력을 JSON → MD → 원장(마지막) 순으로 쓴다. 이미 있으면 아무것도 쓰지 않고 멈춘다.
    도중 실패는 이미 쓴 파일 목록과 함께 RuntimeError — 쓴 파일이 남아 재실행은 막힌다(손으로 확인)."""
    logs_dir = Path(logs_dir)
    found = _already(logs_dir)
    if found:
        raise RuntimeError(f"{logs_dir}: {found} 가 이미 있다 — 관문 0 출력을 쓰지 않는다(재실행 금지)")
    logs_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    try:
        _write_new(logs_dir / OUT_JSON, json_text)
        done.append(OUT_JSON)
        _write_new(logs_dir / OUT_MD, md_text)
        done.append(OUT_MD)
        append_line(logs_dir / OUT_TRIALS, trial)
        done.append(OUT_TRIALS)
    except Exception as e:
        raise RuntimeError(f"관문 0 출력 쓰기 실패 — 이미 쓴 것 {done}(그리고 실패한 파일이 반쯤 남았을 수 있다). "
                           f"재실행은 막힌다 — {logs_dir} 를 손으로 확인하라") from e


def _fmt(v: dict) -> str:
    if not v or not v.get("n"):
        return "—"
    mark = "✓" if v.get("pass") else "✗"
    return f"{mark} {v['bss']:+.4f} [{v['bss_lo']:+.4f}] n={v['n']:,}"


def _grid_md(res: dict, get, title: str) -> list[str]:
    out = [f"### {title}", "", "| N \\ k | " + " | ".join(str(k) for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for N in NS:
        out.append(f"| {N} | " + " | ".join(_fmt(get(res["cells"][f"N{N}_k{k}"])) for k in KS) + " |")
    return out + [""]


def _gate_line(g: dict) -> str:
    return f"{'합격' if g['pass'] else '불합격'} — 합격 칸 {g['n_pass']}/20, 최대 이웃 덩어리 {g['largest_block']}(바: ≥ 12 ∧ ≥ 8)"


def results_md(res: dict, meta: dict) -> str:
    L = ["# 관문 0 결과 (검증 전)", "",
         f"git `{meta.get('git')}` · 자료 {meta.get('data', {}).get('first_day')} ~ {meta.get('data', {}).get('last_day')} · "
         f"사건 {EVENT_START} ~ t+k ≤ {LABEL_END} · M={M_PATHS}", "",
         "칸 = `✓/✗ BSS(대 B0) [월 군집 95% 하한] n` — ✓ = 보정 ∧ 기술 ∧ 연도 ≥ 5/7 (PREREG 관문 0).", "",
         f"**관문 판정(q=2, 대 B0):** {_gate_line(res['gate'])}", ""]
    L += _grid_md(res, lambda c: c["verdict"], "판정 — q=2 대 B0(q=0), 사전 후보(MQ-1 200일선 거름 포함)")
    L += [f"보고 — q=1 관문형: {_gate_line(res['report_gate_q1'])}", ""]
    L += _grid_md(res, lambda c: c["verdict_q1"], "보고 — q=1 대 B0 (판정 무관)")
    L += ["### 보고 — q=2 대 B1(q=1) BSS [월 군집 95%] (판정 무관)", "",
          "| N \\ k | " + " | ".join(str(k) for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for N in NS:
        row = []
        for k in KS:
            b = res["cells"][f"N{N}_k{k}"]["judged"].get("bss_2v1")
            row.append("—" if not b else f"{b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]")
        L.append(f"| {N} | " + " | ".join(row) + " |")
    L.append("")
    for pop, title in (("r1", "R1 200일선 거름 없는 원래 사전 후보"), ("r2", "R2 미너비니 템플릿 충족 부분집합"),
                       ("r3", "R3 정배열 아닌 유니버스 전 종목-일(20% 표본)")):
        L += [f"보고 — {title} 관문형: {_gate_line(res[f'report_gate_{pop}'])}", ""]
        L += _grid_md(res, lambda c, p=pop: c["reports"][p].get("verdict_2v0", {}), f"보고 — {title}, q=2 대 B0 (판정 무관)")
    for key in KEYS[1:]:
        L += _grid_md(res, lambda c, kk=key: c["p_up"][kk].get("verdict_2v0", {}), f"보고 — {key} 사전 후보, q=2 대 B0 (판정 무관)")
    L += ["### 사건 수(짝지은 뒤 / 후보)", "", "| N \\ k | " + " | ".join(str(k) for k in KS) + " |", "|---|" + "---|" * len(KS)]
    for N in NS:
        cs = [res["cells"][f"N{N}_k{k}"] for k in KS]
        L.append(f"| {N} | " + " | ".join(f"{c['n_events']['judged']:,} / {c['n_candidates']['judged']:,}" for c in cs) + " |")
    return "\n".join(L) + "\n"


WATCHED = ("research", "src", "tests")


def require_clean_tree(repo=REPO) -> None:
    """실제 실행 출처(MQ-4): ``research/ src/ tests/`` 에 추적·미추적 변경이 있으면 멈춘다(.superpowers/ 는 gitignore)."""
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *WATCHED], cwd=repo,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git status 실패: {r.stderr.strip()}")
    if r.stdout.strip():
        raise SystemExit("작업 트리가 깨끗하지 않다(실제 실행은 커밋된 코드로만):\n" + r.stdout.rstrip())


def git_sha() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())
    return {"sha": sha, "dirty": dirty}


def config() -> dict:
    return {"Ns": NS, "ks": KS, "qs": QS, "M": M_PATHS, "d_max": D_MAX, "adv_floor_mn": ADV_FLOOR, "member_hist": "N+60 (q=0: 70)",
            "sigma_w": SIGMA_W, "pool_years": POOL_YEARS, "pool_min_dates": POOL_MIN_DATES, "event_start": EVENT_START,
            "label_end": LABEL_END, "data": DATA, "table": PRICE_TABLE, "eps_source": EPS_SOURCE, "years": CAL.YEARS,
            "r3_sample": f"blake2b(code|date) % {R3_MOD} == 0",
            "prereg": "research/logs/ma_quad/PREREG.md + 정정 MQ-1·MQ-2·MQ-3·MQ-4"}


# =============================================================================================== 병렬

def _warmup() -> None:
    """numba 컴파일을 부모에서 한 번 — fork 자식이 물려받는다."""
    y = np.log(np.linspace(100, 130, 300))[None, :].repeat(2, axis=0)
    fc = MQ.poly_forecast(y, 10, 3, 2).astype(np.float64)
    sg = MQ.rolling_sigma(y, SIGMA_W)
    MQ.event_probs(np.exp(y), fc, sg, np.zeros((5, 3), np.float32), np.zeros((1, 4), np.int64), [0], [250], 3)


def _task(args):
    q, N, k, out_dir = args
    X, ev = _G["X"], _G["EV"][k]
    t = time.time()
    P = cell_probs(X, q, N, k, ev["ci"], ev["s"])
    path = probs_path(out_dir, q, N, k)
    save_probs(path, P, ev["ci"], ev["s"])
    return q, N, k, str(path), time.time() - t, len(ev["ci"]), int(np.isfinite(P["p_align"]).sum())


def tasks_for(out_dir) -> list:
    t = [(q, N, k, str(out_dir)) for q in (2, 1) for N in NS for k in KS] + [(0, 0, k, str(out_dir)) for k in KS]
    return sorted(t, key=lambda a: -(max(a[1], 1) * a[2] * (a[0] + 1)))     # 무거운 것 먼저


def run_tasks(X: dict, EV: dict, tasks: list, workers: int) -> list:
    _G["X"], _G["EV"] = X, EV
    _warmup()
    res = []
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks), 1):
            q, N, k, _, secs, n_ev, n_fin = r
            log(f"[{i}/{len(tasks)}] q={q} N={N} k={k}  사건 {n_ev:,} · 확률 유한 {n_fin:,} · {secs:.1f}s (경과 {time.time() - t0:.0f}s)")
            res.append(r)
    return res


def m_sens(X: dict, ev: dict) -> dict:
    """칸 q=2 N=20 k=5, 판정 사건의 결정적 1% 표본에서 M=1000 대 200 의 |Δp| 분위 — 보고만."""
    j = ev["judged"]
    ci, s = ev["ci"][j], ev["s"][j]
    keep = hash_sample(X["codes"], X["days"], ci, s, MSENS["mod"], salt=MSENS["salt"])
    ci, s = ci[keep], s[keep]
    a = cell_probs(X, MSENS["q"], MSENS["N"], MSENS["k"], ci, s, M=MSENS["M"][0])
    b = cell_probs(X, MSENS["q"], MSENS["N"], MSENS["k"], ci, s, M=MSENS["M"][1])
    out = {"cell": {k: MSENS[k] for k in ("q", "N", "k")}, "n_sample": int(len(ci)), "M": MSENS["M"]}
    for key in KEYS:
        d = np.abs(a[key] - b[key])
        d = d[np.isfinite(d)]
        out[key] = {"n": int(len(d)), **({f"q{int(p * 100)}": float(np.quantile(d, p)) for p in (0.5, 0.9, 0.99)} if len(d) else {})}
    return out


# =============================================================================================== 본체

def _connect():
    from swing_it.features.eps_asof import eps_ok
    from swing_it.storage import connect

    if not os.environ.get("KR_QUANT_DB"):
        raise SystemExit("KR_QUANT_DB 가 없다 — set -a; . ~/git/quant-airflow/.env; set +a")
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return con, eps_ok


def _subset(X: dict, rows: np.ndarray, ev_hi: str) -> dict:
    Y = dict(X)
    for key in ("C", "y", "sigma", "U", "aligned", "d", "f200", "tmpl", "eps", "r3", "raw_close", "tv"):
        Y[key] = np.ascontiguousarray(X[key][rows])
    Y["member"] = {mh: np.ascontiguousarray(m[rows]) for mh, m in X["member"].items()}
    Y["codes"] = [X["codes"][i] for i in rows]
    Y["ev_hi"] = ev_hi
    return Y


def _smoke(X: dict, args, scratch: Path) -> None:
    """모양·사건 수·시간만. 지표·확률·실현율은 출력하지도 저장해 보지도 않는다(npz 는 scratch, 열지 않는다)."""
    # 1) 전 패널 한 칸 시간 탐침(가장 무거운 칸) — 확률은 버린다
    k = max(KS)
    t = time.time()
    ev = events_union(X, k)
    log(f"탐침 events_union k={k}: 사건 {len(ev['ci']):,} (judged {int(ev['judged'].sum()):,} · r1 {int(ev['r1'].sum()):,} · "
        f"r2 {int(ev['r2'].sum()):,} · r3 {int(ev['r3'].sum()):,}) {time.time() - t:.1f}s")
    _warmup()
    t = time.time()
    P = cell_probs(X, 2, max(NS), k, ev["ci"], ev["s"])
    probe = time.time() - t
    log(f"탐침 cell_probs q=2 N={max(NS)} k={k} 전 패널: 확률 유한 {int(np.isfinite(P['p_align']).sum()):,} · "
        f"풀 없음 {int(P['nopool'].sum()):,} · {probe:.1f}s")
    del P
    # 2) 50 종목 · 2017 만: 44 작업 병렬 → 짝짓기·판정 경로(출력은 수만)
    d17 = (X["days"] >= pd.Timestamp("2017-01-02")) & (X["days"] <= pd.Timestamp("2017-12-28"))
    active = np.flatnonzero(X["U"][:, np.asarray(d17)].any(axis=1))
    rows = active[np.linspace(0, len(active) - 1, min(50, len(active))).astype(int)]
    S = _subset(X, rows, "2017-12-28")
    EV = {k: events_union(S, k) for k in KS}
    for k in KS:
        log(f"스모크 k={k}: 사건 {len(EV[k]['ci']):,} · judged {int(EV[k]['judged'].sum()):,}")
    out = scratch / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    res = run_tasks(S, EV, tasks_for(out), args.workers)
    wall = time.time() - t
    t = time.time()
    for N in NS:
        for k in KS:
            ev = EV[k]
            Y = MQ.realized(S["C"], ev["ci"], ev["s"], k)
            probs = {q: load_probs(probs_path(out, q, N, k), ev) for q in QS}
            c = assemble_cell(S, N, k, ev, Y, probs, out)
            log(f"스모크 칸 N={N} k={k}: 후보 {c['n_candidates']} → 짝 {c['n_events']} · 빠짐(judged) "
                f"{ {kk: v for kk, v in c['drops']['judged'].items() if kk not in ('n_in', 'n_paired')} }")
    log(f"스모크 짝짓기·판정 경로 {time.time() - t:.1f}s (값은 출력하지 않음)")
    if args.m_sens:
        t = time.time()
        m_sens(S, EV[5])
        log(f"스모크 m-sens {time.time() - t:.1f}s (값은 출력하지 않음)")
    per = sorted(r[4] for r in res)
    n_tasks = len(tasks_for(out))
    log(f"스모크 작업 {len(res)} 개 벽시계 {wall:.1f}s · 작업당 중앙 {per[len(per) // 2]:.1f}s 최대 {per[-1]:.1f}s")
    log(f"예상(전 패널): 작업 {n_tasks} × 탐침 {probe:.0f}s / {args.workers} 작업자 ≈ {n_tasks * probe / args.workers / 60:.1f} 분"
        f"(상한 — q=0·작은 N·k 는 더 가볍다; 짝짓기·판정은 별도)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="모양·사건 수·시간만(50 종목·2017 + 전 패널 한 칸 시간 탐침)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--m-sens", action=argparse.BooleanOptionalAction, default=None,
                    help="q=2 N=20 k=5, 판정 사건 1%% 표본 M=1000 대 200 |Δp| (보고) — 실제 실행은 기본 켬, 스모크는 기본 끔")
    ap.add_argument("--scratch", default=str(SCRATCH))
    ap.add_argument("--logs", default=str(LOGS))
    args = ap.parse_args(argv)
    if args.m_sens is None:
        args.m_sens = not args.smoke                   # 실제 실행은 한 번뿐(재실행 막음) — m-sens 를 같이 낸다
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    scratch = Path(args.scratch)
    logs = Path(args.logs)
    if not args.smoke:
        if logs.resolve() != LOGS.resolve():
            raise SystemExit(f"실제 실행은 정본 {LOGS} 로만 — --logs 는 스모크에서만 바꾼다")
        require_clean_tree()
        guard_rerun(logs)
    scratch.mkdir(parents=True, exist_ok=True)
    gs = git_sha()
    con, eps_ok = _connect()
    t = time.time()
    raw = load(con)
    log(f"load: 종목 {raw['info']['n_codes']:,} × 거래일 {raw['info']['n_days']:,} "
        f"({raw['info']['first_day']} ~ {raw['info']['last_day']}) · pre_listing 칸 {raw['info']['n_pre_listing_cells']} · "
        f"정지일 NaN {raw['info']['n_suspension_cells_nan']:,} 칸 · "
        f"원 종가 없어 수정 종가로 1,000원 규칙 {raw['info']['n_price_rule_from_adj_close']:,} 칸 · {time.time() - t:.1f}s")
    t = time.time()
    codes, days = raw["codes"], raw["days"]
    X = prepare(raw, lambda ci, s: eps_ok(codes, days, ci, s, con, source=EPS_SOURCE))
    con.close()                                        # fork 전에 닫는다(자식은 DB 를 쓰지 않는다)
    del raw
    log(f"prepare: C {X['C'].shape} · U 종목-일 {int(X['U'].sum()):,} · EPS 표 {int(X['eps'].sum()):,} · R3 표본 {int(X['r3'].sum()):,} · "
        f"{time.time() - t:.1f}s")
    if args.smoke:
        _smoke(X, args, scratch)
        return 0

    out = scratch / "gate0"
    out.mkdir(parents=True, exist_ok=True)
    EV = {k: events_union(X, k) for k in KS}
    for k in KS:
        log(f"k={k}: 사건 {len(EV[k]['ci']):,} · judged {int(EV[k]['judged'].sum()):,}")
    run_tasks(X, EV, tasks_for(out), args.workers)
    cells = {}
    for N in NS:
        for k in KS:
            ev = EV[k]
            Y = MQ.realized(X["C"], ev["ci"], ev["s"], k)
            probs = {q: load_probs(probs_path(out, q, N, k), ev) for q in QS}
            cells[(N, k)] = assemble_cell(X, N, k, ev, Y, probs, out)
            log(f"칸 N={N} k={k} 조립 — 짝 {cells[(N, k)]['n_events']['judged']:,}")
    res = summarize(cells)
    meta = {"git": gs["sha"], "git_head": gs["sha"], "git_dirty": gs["dirty"], "git_clean_paths": WATCHED,
            "date": _dt.date.today().isoformat(), "data": X["info"], "config": config(), "rulings": RULINGS}
    if args.m_sens:
        res["m_sens"] = m_sens(X, EV[5])
    # 세 출력을 메모리에서 다 만든 뒤 쓴다(직렬화 실패는 쓰기 전에 난다)
    json_text = json.dumps(_clean({**meta, **res}), allow_nan=False, ensure_ascii=False, indent=1) + "\n"
    md_text = results_md(res, meta)
    trial = trial_line({"trial": "gate0", "date": meta["date"], "config": config(), "verdict": res["gate"], "git": gs["sha"]})
    write_outputs(logs, json_text, md_text, trial)
    log(f"관문 판정: {_gate_line(res['gate'])} → {logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
