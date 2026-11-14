"""골든 HMM 관문 0 실행기 — MA50 → MA200 교차 확률(국면 전환 HMM) 대 q0 의 보정·기술 (PREREG.md 4e09516, 결과 전 고정).

판정: k ∈ {5, 10, 15} 3칸, P_golden(k)(HMM, S = BIC) 대 q0(ma_quad 무조건 경험 오차 라이브러리 무작위 경로) —
`ma_quad_calib.cell_verdict(years=2009..2016)` 단 연도 바는 이 실행기에서 ``years_pos ≥ 6``(8년 중). 관문 = k=10 합격 ∧ 3칸 중 ≥ 2.
보고(판정 무관): S=1 가우시안 대 q0, HMM 대 S=1, 격차 구간(<1%·1~3%·3~5%), 연도별 BSS, 신뢰도, M=1000 민감도(k=10 1% 표본),
선택된 S·BIC 표·상태별 μ/Σ(원 단위), 짝짓기 탈락 사유.

자료 규약(계획 docs/superpowers/plans/2026-11-14-golden-hmm-gate0.md Global Constraints):
- 판정 패널 ``daily_bars_adj_v2`` 2007-11-22 ~ 2016-08-31, ``session="raw"``, 폐지 포함(naver_adj), pre_listing 칸 NaN, 종가 ≤ 0 → NaN,
  정지일(거래량 0·결측) 종가 NaN(ma_quad MQ-2), 1,000원 규칙 = 원 종가 없으면 수정 종가. 실적 ``eps_ok(source="earnings_dart_doc")``.
- HMM 학습 패널: 같은 표 2016-09-01 ~ 2023-12-31(라벨 미사용 — 수익·이평선만), 같은 규약. 2024~ 는 읽지 않는다.
  학습 수열 = 유니버스 ∧ 관측 유한 연속 구간(길이 ≥ 60). 판정 패널은 학습에 들어가지 않는다(``check_windows``).
- 사건: 유니버스(ADV20 ≥ 10억, 상장 korea.MIN_HISTORY = 252일) ∧ EPS↑ ∧ MA50 < MA200 ∧ 1 − MA50/MA200 ≤ 0.05 ∧ 2009-01-02 ≤ t ∧ t+k 거래일 ≤ 2016-08-31.
  라벨 = h ∈ 1..k 중 한 번이라도 MA50 > MA200(엄격), 창 NaN 이면 없음. 짝짓기 = HMM·q0 확률 ∧ 라벨이 모두 유한한 사건.
- 상태 사후확률은 **순방향 필터만**(golden_hmm.forward_filter_panel). 경로 M = 200, 시드 = blake2b(code|date|k|"hmm").
- 첫 실제 실행이 곧 판정 — 정본 ``research/logs/golden_hmm/`` 로만, ``research/ src/ tests/`` 가 깨끗한 트리에서만, 결과 파일이나 원장 gate0 줄이
  있으면 다시 돌지 않는다. 출력은 메모리에서 다 만든 뒤 JSON → MD → 원장(마지막) 순(ma_quad 실행기 함수 재사용).
  ``--smoke`` 는 모양·건수·시간만 출력한다(지표·확률·실현율 없음, 판정 경로 안 돎).

사용(실제 실행은 컨트롤러가 한 번):
``set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --with numba --extra pg --extra research python research/experiments/golden_hmm_gate0.py [--smoke] [--workers 16]``
"""
from __future__ import annotations

import os

# numba·BLAS 를 부르기 전에 — 다중 작업자 스레드 경합·fork 뒤 BLAS 잠금 방지(CLAUDE.md 실행 환경 함정)
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path[:0] = [".", "src"]
from research.experiments import maquad_calib as MQC  # noqa: E402
from research.signals import golden_hmm as GH  # noqa: E402
from research.signals import ma_quad as MQ  # noqa: E402
from research.signals import ma_quad_calib as CAL  # noqa: E402
from swing_it.features.korea import MIN_HISTORY, universe_mask  # noqa: E402
from swing_it.units import eok  # noqa: E402

# --- 사전등록 고정값 ------------------------------------------------------------------------------------------
JUDGE_DATA = ("2007-11-22", "2016-08-31")
TRAIN_DATA = ("2016-09-01", "2023-12-31")
PRICE_TABLE = "daily_bars_adj_v2"
EVENT_START = "2009-01-02"
LABEL_END = "2016-08-31"
KS = (5, 10, 15)
GATE_K = 10
GATE_MIN_CELLS = 2
M_PATHS = 200
GAP_MAX = 0.05
SHORT, LONG = 50, 200
ADV_FLOOR = eok(10)            # ADV20 ≥ 10억(백만원 단위 1,000)
MEMBER_HIST_Q0 = 70            # q0 라이브러리 모집단 상장 ≥ 70일(ma_quad q=0 그대로)
SIGMA_W = 60
POOL_YEARS = 3
POOL_MIN_DATES = 120
YEARS = tuple(range(2009, 2017))
YEARS_MIN_POS = 6              # 8년 중 6(cell_verdict 기본 5 는 7년용 — 여기서 다시 계산)
EPS_SOURCE = "earnings_dart_doc"
S_RANGE = (1, 5)
HMM_SEED = 0
N_ITER = 200
TRAIN_MIN_LEN = 60
TRAIN_CODE_MOD = 1             # 학습 종목 표본 blake2b("train|"+code) % MOD == 0 — 1 = 전 종목(스모크 실측으로 정함, 축소 안 함)
GAP_BINS = ((0.0, 0.01), (0.01, 0.03), (0.03, 0.05))
MSENS = {"k": 10, "mod": 100, "salt": "msens|", "M": (200, 1000)}
SMOKE_ITERS = 5
SMOKE_TRAIN_MOD = 20           # 스모크 학습 시간 탐침 = 종목 5% 표본

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "research" / "logs" / "golden_hmm"
SCRATCH = Path.home() / ".cache" / "cc-swing" / "scratch" / "golden_hmm"
OUT_JSON, OUT_MD, OUT_TRIALS = MQC.OUT_JSON, MQC.OUT_MD, MQC.OUT_TRIALS

RULINGS = {
    "session": "read_prices_wide(table=daily_bars_adj_v2, session='raw') — 두 패널 모두 2026-09-14 애프터마켓 오염 전.",
    "load": "ma_quad 실행기 load 와 같은 규약을 구간 인자로(정지일 NaN 맨 먼저 → pre_listing 칸 NaN(이미 NaN 인지 검사) → 종가 ≤ 0 NaN → "
            "1,000원 규칙 가격 = 원 종가 없으면 수정 종가). 시·고·저가는 읽지 않는다(종가만 쓴다 — O/H/L ≤ 0 규칙은 해당 없음).",
    "standardize": "HMM 은 학습 관측을 차원별 평균·표준편차로 표준화해 맞춘다(아핀 — BIC 순위·사후확률 불변). hmmlearn 의 min_covar 1e-3 이 "
                   "Δ5g(분산 ~1e-4)를 덮지 않게. 경로 z 주변분포는 원 단위로 되돌려 쓴다.",
    "filter_start": "필터는 종목의 첫 유한 관측 날 startprob 에서 시작, 관측 결측 날은 예측만(전이행렬), 첫 관측 전 사후 NaN.",
    "tol": "hmmlearn 기본 tol=1e-2(총 로그우도) — n_iter ≤ 200 상한. hmmlearn monitor_.converged 는 상한 도달도 True 라 쓰지 않고, "
           "converged = 마지막 개선폭 < tol(history 로 직접), hit_cap = (반복 = 상한)을 따로 적는다(리뷰 정정).",
    "q0_library": "q0 의 3년 경험 라이브러리는 판정 패널이 2007-11-22 에 시작해 2009~2010 사건 달에서 3년을 다 채우지 못한다(원점은 패널 첫날 + "
                  "σ 60일 예열 뒤부터). 코드 논리는 그대로(month_pool 의 3년 창 하한이 패널 앞이면 패널 첫날부터) — 사건 달·k 별 풀 크기 P, "
                  "가장 이른 원점 날짜, 원점 날짜 수를 JSON q0_pools 에 적고 RESULTS.md 에 2009~2010 행을 보인다.",
    "bic_tie": "BIC 동률이면 작은 S.",
    "pairing": "칸 k 마다 HMM(선택 S)·q0 확률과 라벨이 모두 유한한 사건만. S=1 보고 열은 같은 사건 집합(유한하지 않으면 멈춘다).",
    "q0": "ma_quad q=0 절차 그대로 — poly_forecast(y,1,k,0)·forecast_z·month_pool(3년·원점 ≥120·u ≤ t_m−k)·member 상장 ≥ 70·"
          "draw_index(code, 달, q=0, N=0, k) M=200, 교차 확률만 ma_quad.cross_probs.",
    "train_sample": f"학습 종목 표본 TRAIN_CODE_MOD={TRAIN_CODE_MOD}(1 = 전 종목).",
    "fit_parallel": "S 마다 별도 spawn 프로세스(작업자 수는 BLAS 반올림만 바꾼다).",
    "years": "연도 바 = 2009..2016 중 BSS(대 q0) > 0 인 해 ≥ 6 — cell_verdict 의 years_ok·pass 를 이 실행기에서 다시 계산.",
}

_G: dict = {}
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# =============================================================================================== 자료

def load(con, data: tuple) -> dict:
    """구간 ``data`` 의 종가·거래대금(백만원)·1,000원 규칙 가격 패널 (n_codes, n_days) + 메타. 지표 없음."""
    from swing_it.features.korea import load_listing_meta
    from swing_it.storage import read_prices, read_prices_wide

    p = read_prices_wide(con, cols=("close", "trade_value", "raw_close", "volume"), table=PRICE_TABLE, start=data[0], end=data[1],
                         session="raw")
    days = pd.DatetimeIndex(p["close"].index)
    codes = pd.Index([str(c) for c in p["close"].columns])

    def arr(c):
        return np.array(p[c].to_numpy(dtype=float).T, dtype=np.float64, order="C", copy=True)

    C, TV, RC, V = arr("close"), arr("trade_value"), arr("raw_close"), arr("volume")
    p = None
    C_src = C.copy()
    n_susp = MQC.mask_suspensions(C, V, RC)                  # MQ-2: 정지일 종가 NaN — 맨 먼저
    del V
    cs = read_prices(con, cols=("code", "date", "close_src"), table=PRICE_TABLE, start=data[0], end=data[1],
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
    n_rc_fill = int((~np.isfinite(RC) & np.isfinite(C)).sum())
    PX = np.where(np.isfinite(RC), RC, C)
    meta = load_listing_meta(con)
    info = {"table": PRICE_TABLE, "start": data[0], "end": data[1], "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
            "n_codes": len(codes), "n_days": len(days), "n_pre_listing_cells": len(pl), "n_suspension_cells_nan": n_susp,
            "n_close_nonpositive": n_nonpos, "n_price_rule_from_adj_close": n_rc_fill}
    return {"days": days, "codes": list(codes), "C": C, "raw_close": PX, "tv": TV, "meta": meta, "info": info}


def check_windows(train_days, judge_days) -> None:
    """학습 패널 ⊂ TRAIN_DATA, 판정 패널 ⊂ JUDGE_DATA, 둘은 겹치지 않고 2024~ 없음 — 어기면 멈춘다."""
    tr, ju = pd.DatetimeIndex(train_days), pd.DatetimeIndex(judge_days)
    if tr.min() < pd.Timestamp(TRAIN_DATA[0]) or tr.max() > pd.Timestamp(TRAIN_DATA[1]) or tr.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError(f"학습 패널 {tr.min().date()} ~ {tr.max().date()} 이 {TRAIN_DATA} 밖")
    if ju.min() < pd.Timestamp(JUDGE_DATA[0]) or ju.max() > pd.Timestamp(JUDGE_DATA[1]):
        raise RuntimeError(f"판정 패널 {ju.min().date()} ~ {ju.max().date()} 이 {JUDGE_DATA} 밖")
    if ju.max() >= tr.min():
        raise RuntimeError("학습·판정 패널이 겹친다")


def _universe(X: dict, **kw) -> np.ndarray:
    idx = pd.Index(X["codes"])
    days = pd.DatetimeIndex(X["days"])
    u = universe_mask(pd.DataFrame(X["C"].T, index=days, columns=idx), pd.DataFrame(np.asarray(X["tv"], float).T, index=days, columns=idx),
                      X["meta"], raw_close=pd.DataFrame(np.asarray(X["raw_close"], float).T, index=days, columns=idx),
                      adv_floor=ADV_FLOOR, **kw)
    return np.array(u.to_numpy(dtype=bool).T, order="C", copy=True)          # 쓰기 가능한 사본


def _code_sample(codes, mod: int) -> np.ndarray:
    if mod <= 1:
        return np.ones(len(codes), bool)
    return np.array([int.from_bytes(hashlib.blake2b(f"train|{c}".encode(), digest_size=8).digest(), "little") % mod == 0 for c in codes])


def train_set(T: dict, code_mod: int = TRAIN_CODE_MOD):
    """학습 패널 → 학습 수열(유니버스 ∧ 관측 유한 연속 구간 ≥ TRAIN_MIN_LEN). 라벨·EPS 를 쓰지 않는다."""
    C = np.ascontiguousarray(T["C"], dtype=np.float64)
    U = _universe(T)
    U = U & _code_sample(T["codes"], code_mod)[:, None]
    obs = GH.stack_obs(*GH.observations(C, SIGMA_W, SHORT, LONG))
    seqs, info = GH.train_sequences(obs, U, TRAIN_MIN_LEN)
    days = pd.DatetimeIndex(T["days"])
    info |= {"first_day": str(days[0].date()), "last_day": str(days[-1].date()),
             "first_obs_day": None if info["t_first"] is None else str(days[info["t_first"]].date()),
             "last_obs_day": None if info["t_last"] is None else str(days[info["t_last"]].date()), "n_codes_panel": len(T["codes"]),
             "n_codes_used": int(U.any(axis=1).sum()), "code_mod": int(code_mod), "U_cells": int(U.sum())}
    return seqs, info


def prepare_judge(X: dict, eps_fn, ev_lo: str = EVENT_START, ev_hi: str = LABEL_END) -> dict:
    """판정 패널 상태: σ·MA50·MA200·격차·유니버스·q0 모집단·관측·EPS↑(사전 후보 기초 칸에만 ``eps_fn(ci, s)``)."""
    X = dict(X)
    days = pd.DatetimeIndex(X["days"])
    C = np.ascontiguousarray(X["C"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(C > 0, np.log(C), np.nan)
    sigma = MQ.rolling_sigma(y, SIGMA_W)
    ma = {SHORT: MQ._rolling_mean(C, SHORT), LONG: MQ._rolling_mean(C, LONG)}
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = 1.0 - ma[SHORT] / ma[LONG]
    U = _universe(X)
    member = _universe(X, min_history=MEMBER_HIST_Q0)
    obs = GH.stack_obs(*GH.observations(C, SIGMA_W, SHORT, LONG))
    in_win = np.asarray((days >= pd.Timestamp(ev_lo)) & (days <= pd.Timestamp(ev_hi)))
    with np.errstate(invalid="ignore"):
        base = U & (gap > 0) & (gap <= GAP_MAX) & in_win[None, :]
    ci, s = np.nonzero(base)
    eps = np.zeros_like(U)
    if len(ci):
        eps[ci, s] = np.asarray(eps_fn(ci, s), dtype=bool)
    X.update(days=days, C=C, y=y, sigma=sigma, ma=ma, gap=gap, U=U, member=member, obs=obs, eps=eps, ev_lo=ev_lo, ev_hi=ev_hi)
    return X


# =============================================================================================== 사건·라벨

def _tvalid(X: dict, k: int) -> np.ndarray:
    days = pd.DatetimeIndex(X["days"])
    n = len(days)
    t = np.arange(n)
    ok = np.asarray(days >= pd.Timestamp(X.get("ev_lo", EVENT_START))) & (t + k < n)
    ok[ok] &= np.asarray(days[t[ok] + k] <= pd.Timestamp(X.get("ev_hi", LABEL_END)))
    return ok


def build_events(X: dict, k: int):
    """사전 후보 (ci, s) — U ∧ EPS↑ ∧ 0 < 1 − MA50/MA200 ≤ 0.05 ∧ 날짜 창. 날짜 순, 같은 날은 종목 순."""
    with np.errstate(invalid="ignore"):
        m = X["U"] & X["eps"] & (X["gap"] > 0) & (X["gap"] <= GAP_MAX)
    m = m & _tvalid(X, k)[None, :]
    s, ci = np.nonzero(m.T)
    return ci.astype(np.int64), s.astype(np.int64)


def labels(X: dict, ci, s, k: int) -> np.ndarray:
    return MQ.cross_realized(X["C"], ci, s, k, SHORT, LONG, ma=X["ma"])


# =============================================================================================== 확률

def q0_probs(X: dict, k: int, ci, s, M: int = M_PATHS) -> tuple:
    """(확률, 풀 없음, 달별 풀 정보) — ma_quad q=0 경험 라이브러리 경로의 교차 확률. 풀 정보 = 사건 달마다 P·가장 이른 원점·원점 날짜 수."""
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    days, codes = pd.DatetimeIndex(X["days"]), X["codes"]
    out = np.full(len(ci), np.nan)
    nopool = np.zeros(len(ci), dtype=bool)
    pinfo: list = []
    if len(ci) == 0:
        return out, nopool, pinfo
    fc = MQ.poly_forecast(X["y"], 1, k, 0)
    z = MQ.forecast_z(X["y"], fc, X["sigma"])
    ms = MQ.month_starts(days)
    tm = ms[np.searchsorted(ms, s, side="right") - 1]
    for t_m in np.unique(tm):
        sel = np.flatnonzero(tm == t_m)
        got = MQ.month_pool(days, z, X["member"], int(t_m), k, years=POOL_YEARS, min_dates=POOL_MIN_DATES, return_origins=True)
        pool, org = got if got is not None else (None, None)
        if pool is None or len(pool) == 0:
            pinfo.append({"month": f"{days[t_m]:%Y-%m}", "P": 0, "first_origin": None, "n_origin_dates": 0})
            nopool[sel] = True
            continue
        pinfo.append({"month": f"{days[t_m]:%Y-%m}", "P": int(len(pool)), "first_origin": f"{days[int(org[:, 1].min())]:%Y-%m-%d}",
                      "n_origin_dates": int(len(np.unique(org[:, 1])))})
        draws = {int(c): MQ.draw_index(codes[c], days[t_m], 0, 0, k, len(pool), M) for c in np.unique(ci[sel])}
        idx = np.stack([draws[int(c)] for c in ci[sel]])
        out[sel] = MQ.cross_probs(X["C"], fc, X["sigma"], pool, idx, ci[sel], s[sel], k, SHORT, LONG)
    return out, nopool, pinfo


def hmm_probs(X: dict, model, post, k: int, ci, s, M: int = M_PATHS) -> np.ndarray:
    return GH.hmm_cross_probs(model, X["C"], X["sigma"], post, ci, s, k, SHORT, LONG, M, codes=X["codes"], days=X["days"])


def _task(args):
    kind, k = args
    X, ev = _G["X"], _G["EV"][k]
    t = time.time()
    if kind == "q0":
        p, nopool, pinfo = q0_probs(X, k, ev[0], ev[1])
    else:
        p, nopool, pinfo = hmm_probs(X, _G["models"][kind], _G["posts"][kind], k, ev[0], ev[1]), None, None
    return kind, k, p, (nopool, pinfo), time.time() - t


def run_tasks(X: dict, EV: dict, models: dict, posts: dict, workers: int) -> dict:
    _G.update(X=X, EV=EV, models=models, posts=posts)
    tasks = [(kind, k) for k in sorted(KS, reverse=True) for kind in ("q0", "hmm", "hmm1")]
    res: dict = {}
    t0 = time.time()
    try:
        if workers > 1:
            _warmup()
            with mp.get_context("fork").Pool(min(workers, len(tasks)), maxtasksperchild=1) as pool:
                it = pool.imap_unordered(_task, tasks)
                for i, (kind, k, p, extra, secs) in enumerate(it, 1):
                    res[(kind, k)] = (p, *extra)
                    log(f"[{i}/{len(tasks)}] {kind} k={k}: 사건 {len(p):,} · 유한 {int(np.isfinite(p).sum()):,} · {secs:.1f}s "
                        f"(경과 {time.time() - t0:.0f}s)")
        else:
            for i, a in enumerate(tasks, 1):
                kind, k, p, extra, secs = _task(a)
                res[(kind, k)] = (p, *extra)
                log(f"[{i}/{len(tasks)}] {kind} k={k}: 사건 {len(p):,} · 유한 {int(np.isfinite(p).sum()):,} · {secs:.1f}s")
    finally:
        _G.clear()
    return res


def _warmup() -> None:
    """numba 컴파일을 부모에서 한 번 — fork 자식이 물려받는다."""
    C = np.exp(np.log(np.linspace(100, 130, 300)))[None, :].repeat(2, axis=0)
    MQ.cross_kernel_events(C, [0], [250], np.zeros((1, 3)), np.zeros(1), np.zeros((2, 3)), np.zeros((1, 2), np.int64), 3, 5, 20)
    P = GH.HMMParams(startprob=np.ones(1), transmat=np.ones((1, 1)), means=np.zeros((1, 3)), covars=np.eye(3)[None],
                     loc=np.zeros(3), scale=np.ones(3))
    GH.hmm_cross_probs(P, C, np.full(C.shape, 0.01), np.ones(C.shape + (1,)), [0], [250], 3, 5, 20, 2,
                       codes=["0", "1"], days=pd.bdate_range("2010-01-04", periods=300))


# =============================================================================================== 짝짓기·판정

def pair(probs: dict, nopool, y) -> tuple:
    """HMM·q0 확률과 라벨이 모두 유한한 사건 마스크와 빠진 수(겹치면 각각 센다). S=1(hmm1)은 보고 열 — 마스크에 안 든다."""
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(probs["hmm"]) & np.isfinite(probs["q0"])
    drops = {"n_in": int(len(y)), "label": int((~np.isfinite(y)).sum()), "hmm_nan": int((~np.isfinite(probs["hmm"])).sum()),
             "q0_nopool": int(np.asarray(nopool).sum()), "q0_nan": int((~np.isfinite(probs["q0"]) & ~np.asarray(nopool)).sum()),
             "hmm1_nan": int((~np.isfinite(probs["hmm1"])).sum()), "n_paired": int(ok.sum())}
    return ok, drops


def verdict(pA, pB, y, month, year) -> dict:
    """cell_verdict(years=2009..2016) + 연도 바 ≥ 6/8 로 years_ok·pass 다시 계산."""
    v = CAL.cell_verdict(pA, pB, y, month, year, years=YEARS)
    v["years_ok"] = bool(v["years_pos"] >= YEARS_MIN_POS)
    v["years_min_pos"] = YEARS_MIN_POS
    v["pass"] = bool(v["calib_ok"] and v["skill_ok"] and v["years_ok"])
    return v


def gate(passed: dict) -> dict:
    if set(passed) != set(KS):
        raise ValueError("칸 키가 KS 와 다르다")
    n = int(sum(bool(v) for v in passed.values()))
    k10 = bool(passed[GATE_K])
    return {"pass": bool(k10 and n >= GATE_MIN_CELLS), "n_pass": n, "k10_pass": k10, "cells": {k: bool(passed[k]) for k in KS}}


def _yearly(pA, pB, y, year) -> dict:
    return {int(yr): (CAL._bss(pA[year == yr], pB[year == yr], y[year == yr]) if (year == yr).any() else float("nan")) for yr in YEARS}


def _resolution(p, y) -> dict:
    m = p >= 0.5
    n = int(m.sum())
    return {"n": n, "y_rate": float(y[m].mean()) if n else float("nan"), "p_mean": float(p[m].mean()) if n else float("nan")}


def reports(P: dict, y, month, year, gap) -> dict:
    hmm, q0, s1 = P["hmm"], P["q0"], P["hmm1"]
    bins = {}
    for lo, hi in GAP_BINS:
        m = (gap > lo) & (gap <= hi)                                # (lo, hi] — 사건은 gap > 0
        key = f"{lo:.2f}-{hi:.2f}"
        if not m.any():
            bins[key] = {"n": 0}
            continue
        bins[key] = {"n": int(m.sum()), "base_rate": float(y[m].mean()),
                     "bss_hmm_vs_q0": list(CAL.bss_ci(hmm[m], q0[m], y[m], month[m])),
                     "bss_s1_vs_q0": list(CAL.bss_ci(s1[m], q0[m], y[m], month[m])),
                     "brier": {n_: CAL.brier(p[m], y[m]) for n_, p in P.items()},
                     "ece": {n_: CAL.ece(p[m], y[m]) for n_, p in P.items()}}
    return {
        "s1_vs_q0": verdict(s1, q0, y, month, year),
        "hmm_vs_s1": list(CAL.bss_ci(hmm, s1, y, month)),
        "gap_bins": bins,
        "yearly_bss": {"hmm_vs_q0": _yearly(hmm, q0, y, year), "s1_vs_q0": _yearly(s1, q0, y, year), "hmm_vs_s1": _yearly(hmm, s1, y, year)},
        "reliability": {n_: CAL.reliability(p, y).to_dict("records") for n_, p in P.items()},
        "brier": {n_: CAL.brier(p, y) for n_, p in P.items()},
        "ece": {n_: CAL.ece(p, y) for n_, p in P.items()},
        "resolution": {n_: _resolution(p, y) for n_, p in P.items()},
    }


def assemble_cell(X: dict, k: int, ev: tuple, y, probs: dict, nopool, out_dir) -> dict:
    ci, s = ev
    days = pd.DatetimeIndex(X["days"])
    ok, drops = pair(probs, nopool, y)
    if not np.isfinite(probs["hmm1"][ok]).all():
        raise RuntimeError(f"k={k}: 짝지은 사건에 S=1 확률 결측 — 사후 마스크가 다르다")
    year = np.asarray(days.year[s], dtype=np.int64)[ok]
    month = year * 100 + np.asarray(days.month[s], dtype=np.int64)[ok]
    yy = np.asarray(y, dtype=float)[ok]
    P = {n_: np.asarray(p, dtype=float)[ok] for n_, p in probs.items()}
    gap = X["gap"][ci, s][ok]
    np.savez(Path(out_dir) / f"cell_k{k}.npz", ci=ci[ok], s=s[ok], codes=np.asarray(X["codes"])[ci[ok]],
             dates=np.asarray(days[s[ok]].strftime("%Y-%m-%d")), y=yy, gap=gap, month=month, year=year, **P)
    if len(yy) == 0:
        return {"k": k, "n_candidates": int(len(ci)), "n_events": 0, "drops": drops, "verdict": {"pass": False, "n": 0}, "reports": {}}
    return {"k": k, "n_candidates": int(len(ci)), "n_events": int(ok.sum()), "drops": drops,
            "verdict": verdict(P["hmm"], P["q0"], yy, month, year), "reports": reports(P, yy, month, year, gap)}


def m_sens(X: dict, model, post, ev: tuple, ok) -> dict:
    """k=10 짝지은 사건의 결정적 1% 표본에서 HMM M=1000 대 200 의 |Δp| 분위 — 보고만."""
    ci, s = ev[0][ok], ev[1][ok]
    keep = MQC.hash_sample(X["codes"], X["days"], ci, s, MSENS["mod"], salt=MSENS["salt"])
    ci, s = ci[keep], s[keep]
    a = hmm_probs(X, model, post, MSENS["k"], ci, s, M=MSENS["M"][0])
    b = hmm_probs(X, model, post, MSENS["k"], ci, s, M=MSENS["M"][1])
    d = np.abs(a - b)
    d = d[np.isfinite(d)]
    return {"k": MSENS["k"], "n_sample": int(len(ci)), "M": MSENS["M"], "n": int(len(d)),
            **({f"q{int(p * 100)}": float(np.quantile(d, p)) for p in (0.5, 0.9, 0.99)} if len(d) else {})}


# =============================================================================================== 출력

write_json = MQC.write_json
require_clean_tree = MQC.require_clean_tree
guard_rerun = MQC.guard_rerun
git_sha = MQC.git_sha


def versions() -> dict:
    import hmmlearn
    import sklearn

    try:
        import numba
        nb = numba.__version__
    except ImportError:  # pragma: no cover — numba 는 선택(ma_quad 규약)
        nb = None
    return {"hmmlearn": hmmlearn.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__, "numba": nb}


def config() -> dict:
    return {"ks": KS, "gate_k": GATE_K, "gate_min_cells": GATE_MIN_CELLS, "M": M_PATHS, "gap_max": GAP_MAX, "short": SHORT, "long": LONG,
            "adv_floor_mn": ADV_FLOOR, "universe_min_history": MIN_HISTORY, "member_hist_q0": MEMBER_HIST_Q0, "sigma_w": SIGMA_W,
            "pool_years": POOL_YEARS, "pool_min_dates": POOL_MIN_DATES, "event_start": EVENT_START, "label_end": LABEL_END,
            "judge_data": JUDGE_DATA, "train_data": TRAIN_DATA, "table": PRICE_TABLE, "eps_source": EPS_SOURCE, "years": YEARS,
            "years_min_pos": YEARS_MIN_POS, "S_range": S_RANGE, "hmm_seed": HMM_SEED, "n_iter": N_ITER, "train_min_len": TRAIN_MIN_LEN,
            "train_code_mod": TRAIN_CODE_MOD, "gap_bins": GAP_BINS, "msens": MSENS, "path_seed": "blake2b(code|date|k|hmm)",
            "prereg": "research/logs/golden_hmm/PREREG.md (4e09516) + 계획 docs/superpowers/plans/2026-11-14-golden-hmm-gate0.md"}


def _fmt_v(v: dict) -> str:
    if not v or not v.get("n"):
        return "| — |"
    mark = "✓" if v.get("pass") else "✗"
    return (f"{mark} | {v['n']:,} | {v['bss']:+.4f} [{v['bss_lo']:+.4f}, {v['bss_hi']:+.4f}] | {v['ece']:.4f} | {v['base_rate']:.3f} | "
            f"{v['years_pos']}/8 | {'O' if v['calib_ok'] else 'X'}/{'O' if v['skill_ok'] else 'X'}/{'O' if v['years_ok'] else 'X'} |")


def results_md(res: dict, meta: dict) -> str:
    g = res["gate"]
    L = ["# 골든 HMM 관문 0 결과 (검증 전)", "",
         f"git `{meta.get('git')}` · 판정 자료 {meta.get('data', {}).get('first_day')} ~ {meta.get('data', {}).get('last_day')} · "
         f"사건 {EVENT_START} ~ t+k ≤ {LABEL_END} · 학습 {TRAIN_DATA[0]} ~ {TRAIN_DATA[1]} · M={M_PATHS}", "",
         f"**관문 판정(HMM 대 q0):** {'합격' if g['pass'] else '불합격'} — 합격 칸 {g['n_pass']}/3, k=10 {'합격' if g['k10_pass'] else '불합격'}"
         f"(바: k=10 합격 ∧ ≥ {GATE_MIN_CELLS})", "",
         "칸 = 보정(P≥0.5 구간 n≥300 마다 |예측−실현| ≤ 5pp ∧ ECE ≤ 2pp ∧ ≤ 기준율 20%) ∧ 기술(BSS ≥ 0.01 ∧ 월 군집 95% 하한 > 0) ∧ 연도 ≥ 6/8.", "",
         "| 모형 | k | 판정 | n | BSS(대 q0) [월 95%] | ECE | 기준율 | 연도+ | 보정/기술/연도 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in KS:
        c = res["cells"][k]
        L.append(f"| HMM S={res['hmm']['S']} | {k} | " + _fmt_v(c["verdict"]))
    for k in KS:
        c = res["cells"][k]
        L.append("| S=1 가우시안(보고) | " + f"{k} | " + _fmt_v(c["reports"].get("s1_vs_q0", {})))
    L += ["", "### 격차 구간별 BSS(HMM 대 q0, 월 95%) — 보고", "", "| k | " + " | ".join(f"{a:.0%}~{b:.0%}" for a, b in GAP_BINS) + " |",
          "|---|" + "---|" * len(GAP_BINS)]
    for k in KS:
        bins = res["cells"][k]["reports"].get("gap_bins", {})
        row = []
        for a, b in GAP_BINS:
            x = bins.get(f"{a:.2f}-{b:.2f}", {})
            row.append("—" if not x.get("n") else f"{x['bss_hmm_vs_q0'][0]:+.4f} [{x['bss_hmm_vs_q0'][1]:+.4f}] n={x['n']:,}")
        L.append(f"| {k} | " + " | ".join(row) + " |")
    L += ["", "### 연도별 BSS(HMM 대 q0) — 보고", "", "| k | " + " | ".join(str(y) for y in YEARS) + " |", "|---|" + "---|" * len(YEARS)]
    for k in KS:
        yb = res["cells"][k]["reports"].get("yearly_bss", {}).get("hmm_vs_q0", {})
        L.append(f"| {k} | " + " | ".join("—" if yb.get(y) is None or not np.isfinite(yb.get(y, np.nan)) else f"{yb[y]:+.4f}" for y in YEARS)
                 + " |")
    h = res["hmm"]
    L += ["", f"### HMM — 선택 S = {h['S']} (BIC 최소, 표준화 공간)", "",
          "| S | BIC | 반복 | 수렴(마지막 개선 < tol) | 상한 도달 | 마지막 개선 |", "|---|---|---|---|---|---|"]
    for S, b in h["bic_table"].items():
        f = h["fits"][S]
        L.append(f"| {S} | {b:,.1f} | {f['n_iter']}/{f['n_iter_max']} | {f['converged']} | {f['hit_cap']} | {f['delta_last']:.4g} |")
    L += ["", f"학습: 수열 {h['train']['n_seq']:,} · 관측 {h['train']['n_obs']:,} · 종목 {h['train']['n_codes_used']:,} "
          f"(관측 {h['train']['first_obs_day']} ~ {h['train']['last_obs_day']}; 패널 {h['train']['first_day']} ~ {h['train']['last_day']}).", "",
          "### 짝짓기 탈락", "", "| k | 후보 | 라벨 NaN | HMM NaN | q0 풀 없음 | q0 NaN | 짝 |", "|---|---|---|---|---|---|---|"]
    for k in KS:
        d = res["cells"][k]["drops"]
        L.append(f"| {k} | {d['n_in']:,} | {d['label']:,} | {d['hmm_nan']:,} | {d['q0_nopool']:,} | {d['q0_nan']:,} | {d['n_paired']:,} |")
    qp = res.get("q0_pools", {})
    L += ["", "### q0 라이브러리 예열(2009~2010 사건 달 — 3년 창이 패널 시작 2007-11-22 에 잘림)", "",
          "| 달 | " + " | ".join(f"k={k} P / 가장 이른 원점 / 원점 날짜" for k in KS) + " |", "|---|" + "---|" * len(KS)]
    by = {k: {r["month"]: r for r in qp.get(k, [])} for k in KS}
    months = sorted({m for k in KS for m in by[k] if m < "2011-01"})
    for m in months:
        cells_ = []
        for k in KS:
            r = by[k].get(m)
            cells_.append("—" if r is None else f"{r['P']:,} / {r['first_origin'] or '—'} / {r['n_origin_dates']}")
        L.append(f"| {m} | " + " | ".join(cells_) + " |")
    if "m_sens" in res:
        m = res["m_sens"]
        L += ["", f"M 민감도(k={m['k']}, 1% 표본 n={m['n']:,}): |Δp| 중앙 {m.get('q50', float('nan')):.4f} · 90% {m.get('q90', float('nan')):.4f} · "
              f"99% {m.get('q99', float('nan')):.4f}"]
    return "\n".join(L) + "\n"


# =============================================================================================== 본체

def _connect():
    from swing_it.features.eps_asof import eps_ok
    from swing_it.storage import connect

    if not os.environ.get("KR_QUANT_DB"):
        raise SystemExit("KR_QUANT_DB 가 없다 — set -a; . ~/git/quant-airflow/.env; set +a")
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return con, eps_ok


def _load_info(tag: str, raw: dict, secs: float) -> None:
    i = raw["info"]
    log(f"{tag} load: 종목 {i['n_codes']:,} × 거래일 {i['n_days']:,} ({i['first_day']} ~ {i['last_day']}) · pre_listing 칸 "
        f"{i['n_pre_listing_cells']:,} · 정지일 NaN {i['n_suspension_cells_nan']:,} · 종가≤0 {i['n_close_nonpositive']:,} · "
        f"1,000원 규칙 수정 종가 {i['n_price_rule_from_adj_close']:,} 칸 · {secs:.1f}s")


def _posts(X: dict, models: dict) -> dict:
    out = {}
    for name, P in models.items():
        t = time.time()
        out[name] = GH.forward_filter_panel(P, X["obs"])
        log(f"순방향 필터 {name}(S={P.S}): {out[name].shape} · {time.time() - t:.1f}s")
    return out


def _smoke(con, eps_ok, args) -> int:
    """모양·건수·시간만 — 지표·확률·실현율 출력 없음, 판정 경로(짝짓기 뒤 지표) 안 돎, 정본 logs 에 쓰지 않는다."""
    t = time.time()
    T = load(con, TRAIN_DATA)
    _load_info("학습", T, time.time() - t)
    t = time.time()
    seqs, info = train_set(T)
    log(f"학습 수열(전 종목): 수열 {info['n_seq']:,} · 관측 {info['n_obs']:,} · 종목 {info['n_codes_used']:,} · {time.time() - t:.1f}s")
    sub, sinfo = train_set(T, code_mod=SMOKE_TRAIN_MOD)
    train_days = T["days"]
    del T
    t = time.time()
    fit = GH.fit_hmm(sub, S_RANGE, seed=HMM_SEED, n_iter=SMOKE_ITERS, workers=min(args.workers, len(range(S_RANGE[0], S_RANGE[1] + 1))))
    wall = time.time() - t
    per_it = {S: f["secs"] / max(f["n_iter"], 1) for S, f in fit["fits"].items()}
    scale = info["n_obs"] / max(sinfo["n_obs"], 1)
    fit_est = max(per_it.values()) * N_ITER * scale
    log(f"HMM 학습 탐침: 표본 관측 {sinfo['n_obs']:,}(1/{SMOKE_TRAIN_MOD} 종목) · {SMOKE_ITERS} 반복 · 벽시계 {wall:.1f}s · 반복당 "
        + ", ".join(f"S={S} {v:.2f}s" for S, v in per_it.items()))
    log(f"HMM 학습 예상(전 관측 × {N_ITER} 반복 상한, S 병렬): ≈ {fit_est / 60:.1f} 분")
    t = time.time()
    raw = load(con, JUDGE_DATA)
    _load_info("판정", raw, time.time() - t)
    check_windows(train_days, raw["days"])
    t = time.time()
    codes, days = raw["codes"], raw["days"]
    X = prepare_judge(raw, lambda ci, s: eps_ok(codes, days, ci, s, con, source=EPS_SOURCE))
    del raw
    t_prep = time.time() - t
    log(f"prepare: C {X['C'].shape} · U 종목-일 {int(X['U'].sum()):,} · q0 모집단 {int(X['member'].sum()):,} · EPS 표 {int(X['eps'].sum()):,} · "
        f"{t_prep:.1f}s")
    EV = {k: build_events(X, k) for k in KS}
    for k in KS:
        y = labels(X, *EV[k], k)
        log(f"k={k}: 사건 {len(EV[k][0]):,} · 라벨 유한 {int(np.isfinite(y).sum()):,} (값은 출력하지 않음)")
    # 탐침 모형(5 반복 표본 적합 — 실제 모형 아님)으로 필터·경로 시간만 잰다. 확률은 버린다.
    models = {"hmm": fit["models"][S_RANGE[1]], "hmm1": fit["models"][S_RANGE[0]]}
    t = time.time()
    posts = _posts(X, models)
    t_filter = time.time() - t
    k = max(KS)
    t = time.time()
    p = hmm_probs(X, models["hmm"], posts["hmm"], k, *EV[k])
    t_hmm = time.time() - t
    log(f"탐침 hmm_cross_probs k={k} 전 사건: 유한 {int(np.isfinite(p).sum()):,} · {t_hmm:.1f}s")
    t = time.time()
    q, nopool, pinfo = q0_probs(X, k, *EV[k])
    t_q0 = time.time() - t
    log(f"탐침 q0 k={k} 전 사건: 유한 {int(np.isfinite(q).sum()):,} · 풀 없음 {int(nopool.sum()):,} · {t_q0:.1f}s · "
        f"사건 달 {len(pinfo)} · 첫 달 {pinfo[0]['month'] if pinfo else '—'} 풀 P {pinfo[0]['P'] if pinfo else 0:,}"
        f"(가장 이른 원점 {pinfo[0]['first_origin'] if pinfo else '—'}, 원점 날짜 {pinfo[0]['n_origin_dates'] if pinfo else 0})")
    y = labels(X, *EV[k], k)
    _, drops = pair({"hmm": p, "q0": q, "hmm1": p}, nopool, y)
    log(f"탐침 짝짓기 k={k}: {drops}")
    del p, q, y
    n_ev = sum(len(EV[kk][0]) for kk in KS)
    tasks = (t_hmm * 2 + t_q0) * n_ev / max(len(EV[k][0]), 1)            # 3 k × (hmm·hmm1·q0) — k=15 를 상한으로
    par = max(t_hmm, t_q0) + tasks / max(min(args.workers, 9), 1)
    msens = t_hmm * (1 + 5) / MSENS["mod"]
    total = fit_est + par + t_filter + t_prep + 120
    log(f"예상(실제 실행): HMM 학습 ≈ {fit_est / 60:.1f} 분 + 필터 {t_filter / 60:.1f} 분 + 확률 9 작업(작업자 {args.workers}) ≈ {par / 60:.1f} 분 "
        f"+ M 민감도 ≈ {msens / 60:.1f} 분 + 적재·준비 ≈ {(t_prep + 120) / 60:.1f} 분 → 총 ≈ {total / 60:.0f} 분(상한 — EM 이 200 반복 전에 "
        f"수렴하면 짧다)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="모양·건수·시간만(지표 없음)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scratch", default=str(SCRATCH))
    ap.add_argument("--logs", default=str(LOGS))
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    scratch, logs = Path(args.scratch), Path(args.logs)
    if not args.smoke:
        if logs.resolve() != Path(LOGS).resolve():
            raise SystemExit(f"실제 실행은 정본 {LOGS} 로만 — --logs 는 바꾸지 않는다")
        require_clean_tree()
        guard_rerun(logs)
    scratch.mkdir(parents=True, exist_ok=True)
    gs = git_sha()
    con, eps_ok = _connect()
    if args.smoke:
        try:
            return _smoke(con, eps_ok, args)
        finally:
            con.close()

    # 1) 학습 수열 — 학습 패널만(2016-09-01 ~ 2023-12-31), 라벨·EPS 없음
    t = time.time()
    T = load(con, TRAIN_DATA)
    _load_info("학습", T, time.time() - t)
    seqs, tinfo = train_set(T)
    train_days = T["days"]
    del T
    log(f"학습 수열: {tinfo['n_seq']:,} · 관측 {tinfo['n_obs']:,} · 종목 {tinfo['n_codes_used']:,}")

    # 2) 판정 패널(EPS 까지 DB 에서 읽고 연결을 닫는다 — 긴 학습 동안 연결을 들고 있지 않게)
    t = time.time()
    raw = load(con, JUDGE_DATA)
    _load_info("판정", raw, time.time() - t)
    check_windows(train_days, raw["days"])
    codes, days = raw["codes"], raw["days"]
    X = prepare_judge(raw, lambda ci, s: eps_ok(codes, days, ci, s, con, source=EPS_SOURCE))
    con.close()                                        # fork 전에 닫는다
    info = raw["info"]
    del raw
    log(f"prepare: C {X['C'].shape} · U 종목-일 {int(X['U'].sum()):,} · EPS 표 {int(X['eps'].sum()):,}")

    # 3) HMM 학습(학습 수열만) → 판정 패널에는 순방향 필터만
    t = time.time()
    fit = GH.fit_hmm(seqs, S_RANGE, seed=HMM_SEED, n_iter=N_ITER, workers=min(args.workers, S_RANGE[1] - S_RANGE[0] + 1))
    del seqs
    log(f"HMM 학습 {time.time() - t:.1f}s — 선택 S={fit['S']}")
    models = {"hmm": fit["model"], "hmm1": fit["models"][1]}
    posts = _posts(X, models)
    EV = {k: build_events(X, k) for k in KS}
    Y = {k: labels(X, *EV[k], k) for k in KS}
    for k in KS:
        log(f"k={k}: 사건 {len(EV[k][0]):,}")
    out = scratch / "gate0"
    out.mkdir(parents=True, exist_ok=True)
    R = run_tasks(X, EV, models, posts, args.workers)
    cells, oks = {}, {}
    for k in KS:
        probs = {kind: R[(kind, k)][0] for kind in ("hmm", "q0", "hmm1")}  # R[(kind, k)] = (확률, 풀 없음, 풀 정보)
        cells[k] = assemble_cell(X, k, EV[k], Y[k], probs, R[("q0", k)][1], out)
        oks[k], _ = pair(probs, R[("q0", k)][1], Y[k])
        log(f"칸 k={k} 조립 — 짝 {cells[k]['n_events']:,}")
    res = {"gate": gate({k: cells[k]["verdict"].get("pass", False) for k in KS}), "cells": cells,
           "q0_pools": {k: R[("q0", k)][2] for k in KS}, "versions": versions()}
    res["m_sens"] = m_sens(X, models["hmm"], posts["hmm"], EV[MSENS["k"]], oks[MSENS["k"]])
    res["hmm"] = {"S": fit["S"], "bic_table": fit["bic_table"], "fits": fit["fits"], "selected": fit["model"].report(),
                  "s1": fit["models"][1].report(), "all": {S: m.report() for S, m in fit["models"].items()}, "train": tinfo}
    meta = {"git": gs["sha"], "git_head": gs["sha"], "git_dirty": gs["dirty"], "git_clean_paths": MQC.WATCHED,
            "date": _dt.date.today().isoformat(), "data": info, "config": config(), "rulings": RULINGS}
    json_text = json.dumps(MQC._clean({**meta, **res}), allow_nan=False, ensure_ascii=False, indent=1) + "\n"
    md_text = results_md(res, meta)
    trial = MQC.trial_line({"trial": "gate0", "date": meta["date"], "config": config(), "verdict": res["gate"],
                            "hmm_S": fit["S"], "git": gs["sha"]})
    MQC.write_outputs(logs, json_text, md_text, trial)
    log(f"관문 판정: {'합격' if res['gate']['pass'] else '불합격'} → {logs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
