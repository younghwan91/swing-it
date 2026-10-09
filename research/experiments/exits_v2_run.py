#!/usr/bin/env python
"""미너비니 청산 v2 — 시험 1(청산만 바꾸기): 청산 격자 14,400 칸 전부 + 월 내 셔플 진입 널 30벌. **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/MINERVINI_EXITS_V2.md`` (2e47f59, 결과 보기 전 고정). 이 러너는 숫자를
``MINERVINI_EXITS_V2_RESULTS.md`` 에 적고 판단 규칙(미리 적음)의 산술 비교만 한다 — 바꿀지는 사람이 정한다.

- **진입 고정:** 라이브 기본 ``StrategyConfig`` (수급 F 제외판 = ``require_flow=False``) 의 GA2 유전자 19개 → ``ga2_run.g2_spec`` 마스크.
  사건 2017-09-15 ~ 2026-04-01, 유니버스 H5(위생 ∧ ADV20 ≥ 5,000 = 유동 ∪ 중소형), 추격 5%(피벗 50일), 상한가 시가 버림, 뉴스 거부권 없음
  (GA 와 같다). 사전계산은 GA 캐시(``ga_run.Pre`` — 읽기 전용 mmap)를 그대로 쓴다.
- **청산 격자:** be_r 5 × take_r 5 × take_frac 2 × 강세 매도 4 × fb_tol 4(끔 포함) × fb_days 2 × stop_cap 3 × trail_ma 3 = 14,400 칸 전부
  (:mod:`research.signals.exits_v2_fast` — 최대 보유 120 · 시간 손절 없음 · 손절 하한 3% · 수축 저점 × 0.99 고정). 칸마다 후보(마스크 통과 쌍)와
  그 대조 쌍만 시뮬한다.
- **칸별 숫자:** 겹침 제거 트레이드(자본·슬롯 무제한) n · 승률(순 = 수익 − 비용 > 0) · 평균 이익/손실(순) · 실현 손익비 · R(= 총수익/d) 평균·중앙값 ·
  꽉 찬 손절(남은 수량 전부가 기술적 손절) 비율 · 본전 청산 비율 · 보유 중앙값 · 모멘텀 정합 α(트레이드 수익 − 대조 5개 평균 — 대조도 **같은 칸 청산**,
  ``ga_run`` RELAX 층·풀·해시 추출 그대로) · 10슬롯 포트폴리오 연환산 정합 초과·MDD(``ga_fast`` 처리량 엔진의 v2 판 — 절단 없음, 전 구간).
- **널:** 진입 피처(추세·RS·breadth·PEAD·rel·돌파 비트·거래량·EPS)를 **같은 신호월 안에서** 쌍끼리 재배정(``ga2_run.set_noise`` 와 같은 식·같은
  시드 1000+k) 30벌 × 같은 14,400 칸 → 칸별 95 백분위, (대표 칸 − 현 프로토콜) 차이의 널 분포.
- **지형:** 유전자별 주효과(다른 유전자 전부에 대한 중앙값), 상위 칸의 한 유전자 이웃 중앙값.

Run:  NUMBA_NUM_THREADS=1 uv run --with numba --extra pg python research/experiments/exits_v2_run.py --cache <pre_v2> --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import ga_run as g1  # noqa: E402

from research.signals import exits_v2_fast as xv  # noqa: E402
from research.signals import ga_fast as gf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "exits_v2"
RESULTS_MD = "MINERVINI_EXITS_V2_RESULTS.md"
CELLS_CSV = "minervini_exits_v2_cells.csv"
DESIGN_DOC = "research/logs/sepa_explore/MINERVINI_EXITS_V2.md (2e47f59)"
N_NOISE = 30
NOISE_SEED0 = 1000                 # ga2_run.set_noise 와 같은 시드 → 같은 셔플 세계

# --- 격자(설계 표 그대로 — 탐색하지 않는다) ---
GENES = ("be_r", "take_r", "take_frac", "strength", "fb_tol", "fb_days", "stop_cap", "trail_ma")
LEVELS = {
    "be_r": (0.5, 1.0, 1.5, 2.0, math.inf),
    "take_r": (1.0, 1.5, 2.0, 3.0, math.inf),
    "take_frac": (1 / 3, 1 / 2),
    "strength": (0, 1, 2, 3),                       # 끔 · x 15% · y 30% · x 15% 또는 y 30%
    "fb_tol": (0.0, 0.02, 0.05, None),              # None = 돌파 실패 끔
    "fb_days": (1, 2),
    "stop_cap": (0.06, 0.08, 0.10),
    "trail_ma": (10, 20, 50),
}
SHAPE = tuple(len(LEVELS[g]) for g in GENES)
N_CELLS = int(np.prod(SHAPE))
#: 원전 대표 칸(설계에서 미리 고정) · 현 프로토콜(라이브 기본 청산).
REP = {"be_r": 1.0, "take_r": 2.0, "take_frac": 1 / 2, "strength": 3, "fb_tol": 0.02, "fb_days": 2, "stop_cap": 0.08,
       "trail_ma": 20}
CUR = {"be_r": math.inf, "take_r": 2.0, "take_frac": 1 / 2, "strength": 0, "fb_tol": 0.0, "fb_days": 1, "stop_cap": 0.10,
       "trail_ma": 50}
FIXED = {"stop_floor": 0.03, "stop_mult": 0.99, "max_hold": 120, "ts_days": 0, "lu_skip": True, "chase": 0.05,
         "pivot_window": 50}

METRICS = ("n", "win", "avg_win", "avg_loss", "payoff", "r_mean", "r_med", "r_mean_net", "gross", "net", "full_stop",
           "be_exit", "hold_med", "partial", "strength", "be_moved", "fb_exit", "trail_exit", "stop_exit", "alpha",
           "n_matched", "port_ann", "port_mdd", "orders", "opened", "unmatched")
MI = {k: i for i, k in enumerate(METRICS)}


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _pct(x, d: int = 2) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def _f(x, fmt: str = "{:.2f}") -> str:
    return "nan" if x is None or not np.isfinite(x) else fmt.format(x)


# ---------------------------------------------------------------------------
# 순수 헬퍼 — 칸 번호 ↔ 유전자 (tests)
# ---------------------------------------------------------------------------

def cell_genes(k: int) -> dict:
    idx = np.unravel_index(int(k), SHAPE)
    return {g: LEVELS[g][int(i)] for g, i in zip(GENES, idx, strict=True)}


def cell_index(genes: dict) -> int:
    idx = []
    for g in GENES:
        v = genes[g]
        lv = LEVELS[g]
        idx.append(next(i for i, x in enumerate(lv) if (x is None and v is None) or (x is not None and v is not None
                                                                                    and x == v)))
    return int(np.ravel_multi_index(tuple(idx), SHAPE))


def cell_params(genes: dict) -> dict:
    """칸 유전자 → v2 커널 인자(``exits_v2_fast.EXIT_V2_DEFAULT`` 모양)."""
    ssx, ssy = xv.STRENGTH_LEVELS[genes["strength"]]
    fb_on = genes["fb_tol"] is not None
    return {"stop_floor": FIXED["stop_floor"], "stop_cap": genes["stop_cap"], "stop_mult": FIXED["stop_mult"],
            "take_r": genes["take_r"], "take_frac": genes["take_frac"], "be_r": genes["be_r"], "ss_x": ssx, "ss_y": ssy,
            "fb": fb_on, "fb_tol": genes["fb_tol"] if fb_on else 0.0, "fb_days": genes["fb_days"],
            "ts_days": FIXED["ts_days"], "max_hold": FIXED["max_hold"], "has_trail": True, "lu_skip": FIXED["lu_skip"]}


def cell_str(g: dict) -> str:
    def inf(v):
        return "∞" if v == math.inf else f"{v:g}"

    fb = "끔" if g["fb_tol"] is None else f"{g['fb_tol']:g}×{g['fb_days']}일"
    return (f"be {inf(g['be_r'])} · 익절 {inf(g['take_r'])}R×{'1/3' if g['take_frac'] < 0.4 else '1/2'} · "
            f"강세 {xv.STRENGTH_NAMES[g['strength']]} · 돌파실패 {fb} · 손절상한 {g['stop_cap']:g} · 추적 MA{g['trail_ma']}")


def world_perm(month_of_pair: np.ndarray, rep: int) -> np.ndarray:
    """``ga2_run.set_noise`` 와 같은 순열 — 같은 신호월 안에서 쌍 번호를 섞는다(시드 1000 + rep)."""
    rng = np.random.default_rng(NOISE_SEED0 + rep)
    m = month_of_pair
    order = np.argsort(m, kind="stable")
    b = np.searchsorted(m[order], np.arange(m.max() + 2))
    perm = np.arange(len(m))
    for j in range(len(b) - 1):
        idx = order[b[j]:b[j + 1]]
        perm[idx] = rng.permutation(idx)
    return perm


def port_stats(daily: np.ndarray, month_id: np.ndarray, d_from: int, t0: int, t1: int) -> "tuple[float, float, np.ndarray]":
    """날별 정합 초과 → (연환산 = 월합 평균 × 12, MDD = 단리 누적의 고점 대비 최대 하락, 월합)."""
    m = month_id[d_from:t1]
    v = daily[d_from - t0:t1 - t0]
    mo = np.bincount(m - m[0], weights=v, minlength=m[-1] - m[0] + 1)
    cum = np.r_[0.0, np.cumsum(v)]
    return float(mo.mean() * 12), float(np.max(np.maximum.accumulate(cum) - cum)), mo


# ---------------------------------------------------------------------------
# 작업자 — 세계(진입 집합) × 칸 묶음
# ---------------------------------------------------------------------------

_W: dict = {}


def worker_init(pre_path: str, threads: int = 1) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(threads)
    pre = g1.Pre.load(pre_path)
    _W.update(pre=pre, MA={10: pre.MA10, 20: pre.MA20, 50: pre.MA50}, world=None)


def _world_arrays(w: dict) -> dict:
    """세계 dict(부모가 만든 — 후보 M·대조 번호·우선순위) → 시뮬용 연속 배열(작업자 캐시)."""
    if _W.get("world_id") == w["id"]:
        return _W["world"]
    pre = _W["pre"]
    U = w["U"]
    a = {"U": U, "pc": np.ascontiguousarray(pre.pc[U]), "pd": np.ascontiguousarray(pre.pd[U]),
         "clow": np.ascontiguousarray(pre.clow[U]), "piv": np.ascontiguousarray(pre.piv[U, g1.W50]),
         "mloc": w["mloc"], "cloc": w["cloc"], "fill": w["fill"], "ord": w["ord"], "rel": w["rel"], "rs": w["rs"],
         "market": np.ascontiguousarray(pre.market[w["M"]]), "day_lo": w["day_lo"], "day_hi": w["day_hi"],
         "pc_m": np.ascontiguousarray(pre.pc[w["M"]]), "pd_m": np.ascontiguousarray(pre.pd[w["M"]])}
    _W["world_id"], _W["world"] = w["id"], a
    return a


def eval_cell(a: dict, genes: dict, *, trades: bool = False):
    """한 세계 × 한 칸 → 숫자 벡터(:data:`METRICS` 순서). ``trades`` 면 (벡터, 트레이드 표 dict, 포트폴리오 날별) 도."""
    pre = _W["pre"]
    p = cell_params(genes)
    from research.signals.sepa import LIMIT_UP_RATIO  # noqa: PLC0415

    st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = xv.sim_pairs_v2(
        pre.O, pre.H, pre.L, pre.C, _W["MA"][genes["trail_ma"]], pre.MA50, pre.TR, pre.last_present, a["pc"], a["pd"],
        a["clow"], a["piv"], p["stop_floor"], p["stop_cap"], p["stop_mult"], p["take_r"], p["take_frac"], p["be_r"],
        p["ss_x"], p["ss_y"], p["fb"], p["fb_tol"], p["fb_days"], p["ts_days"], p["max_hold"], p["has_trail"],
        LIMIT_UP_RATIO, p["lu_skip"])
    ml, cl = a["mloc"], a["cloc"]
    out = np.full(len(METRICS), np.nan)
    # 대조 평균 수익(같은 칸 청산) — 대조는 체결 쌍 풀에서 뽑았다
    has_c = cl[:, 0] >= 0
    ctl = np.full(len(ml), np.nan)
    ctl[has_c] = ret[cl[has_c]].mean(axis=1)
    xm, retm = x[ml], ret[ml]
    # 트레이드(겹침 제거 — 자본·슬롯 무제한)
    sel = gf.greedy(a["fill"], a["pc_m"], a["pd_m"], xm.astype(np.int64), pre.n_codes, 0, len(ml))
    cost_m = np.where(xm >= 0, pre.cost_tab[np.maximum(xm, 0), np.maximum(a["market"], 0)], 0.0)
    if len(sel):
        r = retm[sel]
        d = dd[ml][sel]
        net = r - cost_m[sel]
        R = r / d
        rsn = rs[ml][sel]
        flg = fl[ml][sel]
        win, loss = net[net > 0], net[net < 0]
        ex = r - ctl[sel]
        okx = np.isfinite(ex)
        out[MI["n"]] = len(sel)
        out[MI["win"]] = float((net > 0).mean())
        out[MI["avg_win"]] = float(win.mean()) if len(win) else np.nan
        out[MI["avg_loss"]] = float(loss.mean()) if len(loss) else np.nan
        out[MI["payoff"]] = out[MI["avg_win"]] / -out[MI["avg_loss"]] if len(win) and len(loss) else np.nan
        out[MI["r_mean"]] = float(R.mean())
        out[MI["r_med"]] = float(np.median(R))
        out[MI["r_mean_net"]] = float((net / d).mean())
        out[MI["gross"]] = float(r.mean())
        out[MI["net"]] = float(net.mean())
        out[MI["full_stop"]] = float(((rsn == gf.R_STOP) & (ff[ml][sel] == 1.0)).mean())
        out[MI["be_exit"]] = float((rsn == gf.R_BREAKEVEN).mean())
        out[MI["stop_exit"]] = float((rsn == gf.R_STOP).mean())
        out[MI["fb_exit"]] = float((rsn == gf.R_FAILED).mean())
        out[MI["trail_exit"]] = float((rsn == gf.R_TRAIL).mean())
        out[MI["hold_med"]] = float(np.median(xm[sel] - a["pd_m"][sel]))
        out[MI["partial"]] = float(((flg & xv.F_PARTIAL) > 0).mean())
        out[MI["strength"]] = float(((flg & xv.F_STRENGTH) > 0).mean())
        out[MI["be_moved"]] = float(((flg & xv.F_BE_TRIG) > 0).mean())
        out[MI["alpha"]] = float(ex[okx].mean()) if okx.any() else np.nan
        out[MI["n_matched"]] = int(okx.sum())
    # 10슬롯 포트폴리오(처리량 엔진 v2, 전 구간 — 절단 없음)
    t0, t1 = a.get("t0", int(pre.pd[0])), pre.n_days
    daily, opened, n_ord, n_unm = xv.portfolio_v2(
        a["pd_m"], a["pc_m"], a["fill"], a["ord"], a["rel"], a["rs"], xm, np.ascontiguousarray(ea[ml]),
        np.ascontiguousarray(ef[ml]), np.ascontiguousarray(er[ml]), np.ascontiguousarray(ff[ml]),
        np.ascontiguousarray(fr[ml]), ctl, cost_m, a["day_lo"], a["day_hi"], pre.O, pre.C, t0, t1, gf.SLOTS, gf.SLOT_W)
    ann, mdd, mo = port_stats(daily, pre.month_id, t0 + 1, t0, t1)
    out[MI["port_ann"]], out[MI["port_mdd"]] = ann, mdd
    out[MI["orders"]], out[MI["opened"]], out[MI["unmatched"]] = n_ord, len(opened), n_unm
    if not trades:
        return out
    tr = {"sel": sel, "ret": retm[sel], "d": dd[ml][sel], "cost": cost_m[sel], "ctl": ctl[sel], "reason": rs[ml][sel],
          "x": xm[sel], "pd": a["pd_m"][sel], "flags": fl[ml][sel], "fin_frac": ff[ml][sel]}
    return out, tr, {"daily": daily, "monthly": mo, "t0": t0}


def cell_task(t: dict) -> dict:
    a = _world_arrays(t["world"])
    tt = time.time()
    res = np.array([eval_cell(a, cell_genes(k)) for k in t["cells"]])
    return {"wid": t["world"]["id"], "cells": t["cells"], "res": res, "sec": time.time() - tt}


# ---------------------------------------------------------------------------
# 부모 — 세계 만들기
# ---------------------------------------------------------------------------

def build_world(pre, E, spec: dict, fill_all: np.ndarray, cidx: np.ndarray, rep: "int | None") -> dict:
    """세계(실제 = None, 셔플 = rep) → 후보 M(마스크 통과 쌍)·시뮬 대상 U(M ∪ 대조)·우선순위·날짜 범위."""
    feats = ("nh", "al", "stack", "slope_bits", "rs", "brk_bits", "vol", "vavg", "pead", "rel", "breadth", "eps_yoy",
             "eps_acc", "rev_yoy")
    if rep is None:
        f = {k: getattr(pre, k) for k in feats}
    else:
        perm = world_perm(pre.month_id[pre.pd], rep)
        f = {k: np.ascontiguousarray(np.asarray(getattr(pre, k))[perm]) for k in feats}
    ones = np.ones(pre.P, np.bool_)
    m = gf.mask_ga2(0, pre.P, f["nh"], f["al"], f["stack"], f["slope_bits"], f["rs"], f["brk_bits"], f["vol"], f["vavg"],
                    f["pead"], f["rel"], f["breadth"], pre.band, spec["g"], -1, ones, f["eps_yoy"], f["eps_acc"],
                    f["rev_yoy"])
    M = np.flatnonzero(m).astype(np.int64)
    cm = cidx[M]
    U = np.unique(np.r_[M, cm[cm >= 0].ravel()]).astype(np.int64)
    mloc = np.searchsorted(U, M)
    cloc = np.where(cm >= 0, np.searchsorted(U, np.maximum(cm, 0)), -1).astype(np.int64)
    pdm = pre.pd[M]
    day_lo = np.searchsorted(pdm, np.arange(pre.n_days), "left").astype(np.int64)
    day_hi = np.searchsorted(pdm, np.arange(pre.n_days), "right").astype(np.int64)
    return {"id": -1 if rep is None else rep, "M": M, "U": U, "mloc": mloc, "cloc": cloc,
            "fill": np.ascontiguousarray(fill_all[M]), "ord": np.ascontiguousarray(E.orderable[M]),
            "rel": np.ascontiguousarray(np.asarray(f["rel"])[M], dtype=np.float64),
            "rs": np.ascontiguousarray(np.asarray(f["rs"])[M], dtype=np.float64), "day_lo": day_lo, "day_hi": day_hi}


# ---------------------------------------------------------------------------
# 실데이터 패리티(부모)
# ---------------------------------------------------------------------------

def parity_checks(pre, E, spec: dict, world: dict, t0: float) -> dict:
    """(a) 현 프로토콜 칸 v2 커널 = 옛 커널(비트), (b) v2 포트폴리오 = GA2 처리량 엔진(같은 칸), (c) 연구 시뮬레이터(v2) = v2 엔진 트레이드."""
    from research.signals.minervini_exits import simulate_minervini  # noqa: PLC0415

    out: dict = {}
    a = _world_arrays(world)
    # (a) 옛 커널(사전등록 + 돌파 실패 켬·시간 손절 없음 — 라이브 기본 청산)
    p_old = {**gf.EXIT_PREREG, "ts_days": None, "fb": True, "stop_cap": 0.10, "take_r": 2.0}
    o = gf.sim_book(pre.O, pre.H, pre.L, pre.C, pre.MA50, pre.TR, pre.last_present, a["pc"], a["pd"], a["clow"], p_old,
                    pivot=a["piv"])
    n = xv.sim_book_v2(pre.O, pre.H, pre.L, pre.C, pre.MA50, pre.MA50, pre.TR, pre.last_present, a["pc"], a["pd"],
                       a["clow"], cell_params(CUR), pivot=a["piv"])
    ok = o[0] == gf.ST_OK
    out["kernel"] = {"pairs": int(len(a["pc"])), "ok": int(ok.sum()),
                     "status_eq": bool((o[0] == n[0]).all()), "x_eq": bool((o[1][ok] == n[1][ok]).all()),
                     "reason_eq": bool((o[7][ok] == n[4][ok]).all()), "ret_bitwise": bool((o[2][ok] == n[2][ok]).all())}
    _log(f"패리티(a) 현 칸 v2 = 옛 커널: {out['kernel']}", t0)
    # (b) 포트폴리오: GA2 Engine.portfolio(옛 청산, 절단 없음) vs v2
    tt = int(pre.pd[0])
    d_old, op_old, nord_old, _ = E.portfolio(spec, 0, pre.P, tt, pre.n_days, None)
    _, _, pdict = eval_cell(a, CUR, trades=True)
    vec = eval_cell(a, CUR)
    out["portfolio"] = {"max_abs_daily": float(np.max(np.abs(pdict["daily"] - d_old))), "orders_old": int(nord_old),
                        "orders_v2": int(vec[MI["orders"]]), "opened_old": int(len(op_old)),
                        "opened_v2": int(vec[MI["opened"]])}
    _log(f"패리티(b) 포트폴리오 v2 = GA2 엔진: {out['portfolio']}", t0)
    # (c) 연구 시뮬레이터(v2 키워드) vs v2 엔진 트레이드 — 실제 세계 사건
    Opn, H, L, C, V = (pd.DataFrame(np.asarray(x).T, index=pre.days, columns=pre.codes)
                       for x in (pre.O, pre.H, pre.L, pre.C, np.where(pre.TR, 1.0, 0.0)))
    M = world["M"]
    ev = pd.DataFrame({"code": pre.codes[pre.pc[M]], "event_day": pre.days[pre.pd[M]], "pivot": pre.piv[M, g1.W50],
                       "contraction_low": pre.clow[M]})
    sma = pd.DataFrame(np.asarray(pre.MA50).T, index=pre.days, columns=pre.codes)
    rows = []
    for name, g in (("대표", REP), ("현", CUR), ("be0.5·익절∞·x15·tol0.05", {**REP, "be_r": 0.5, "take_r": math.inf,
                                                                       "strength": 1, "fb_tol": 0.05, "trail_ma": 10}),
                    ("be2·익절1R×1/3·y30·돌파끔", {**REP, "be_r": 2.0, "take_r": 1.0, "take_frac": 1 / 3, "strength": 2,
                                              "fb_tol": None, "trail_ma": 50, "stop_cap": 0.06}),
                    ("익절3R×1/3·tol0·2일", {**CUR, "take_r": 3.0, "take_frac": 1 / 3, "fb_days": 2, "trail_ma": 20})):
        p = cell_params(g)
        ma = pd.DataFrame(np.asarray({10: pre.MA10, 20: pre.MA20, 50: pre.MA50}[g["trail_ma"]]).T, index=pre.days,
                          columns=pre.codes)
        ref = simulate_minervini(Opn, H, L, C, ma, ev, volume=V, stop_cap=p["stop_cap"], take_r=p["take_r"],
                                 time_stop_days=None, max_hold=FIXED["max_hold"], chase=FIXED["chase"],
                                 failed_breakout=p["fb"], take_frac=p["take_frac"], be_r=p["be_r"],
                                 strength_ret=None if np.isnan(p["ss_x"]) else p["ss_x"],
                                 strength_ma_ext=None if np.isnan(p["ss_y"]) else p["ss_y"], fb_tol=p["fb_tol"],
                                 fb_days=p["fb_days"], strength_ma=sma, exits_v2=True)
        vec, tr, _ = eval_cell(a, g, trades=True)
        eng = pd.DataFrame({"code": pre.codes[pre.pc[M][tr["sel"]]], "event_day": pre.days[tr["pd"]],
                            "exit_day": pre.days[tr["x"]], "ret": tr["ret"],
                            "reason": [gf.REASON_NAMES[k] for k in tr["reason"]]})
        a_ = eng.sort_values(["event_day", "code"]).reset_index(drop=True)
        b_ = ref.sort_values(["event_day", "code"]).reset_index(drop=True)
        r = {"name": name, "events": len(ev), "n_engine": len(a_), "n_ref": len(b_)}
        if len(a_) == len(b_) and len(a_):
            r["key_eq"] = bool((a_["code"].to_numpy() == b_["code"].to_numpy()).all()
                               and (a_["event_day"].to_numpy() == pd.to_datetime(b_["event_day"]).to_numpy()).all())
            r["exit_eq"] = bool((a_["exit_day"].to_numpy() == pd.to_datetime(b_["exit_day"]).to_numpy()).all())
            r["reason_eq"] = bool((a_["reason"].to_numpy() == b_["exit_reason"].to_numpy()).all())
            r["ret_maxdiff"] = float(np.abs(a_["ret"].to_numpy() - b_["ret"].to_numpy(float)).max())
        r["ok"] = bool(len(a_) == len(b_) and r.get("key_eq") and r.get("exit_eq") and r.get("reason_eq")
                       and r.get("ret_maxdiff", 1.0) < 1e-9)
        r["reasons"] = b_["exit_reason"].value_counts().to_dict()
        rows.append(r)
        _log(f"패리티(c) {name}: 엔진 {len(a_)} · 연구 {len(b_)} · 일치 {r['ok']} (최대 차 {r.get('ret_maxdiff', np.nan):.1e})", t0)
    out["research"] = rows
    del Opn, H, L, C, V, sma
    return out


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def write_trials(path: str, t0: float) -> int:
    """원장: 칸마다 한 줄(평가 전 — 결과를 보기 전에 적는다)."""
    seen = g1.read_fingerprints(path)
    rows = []
    for k in range(N_CELLS):
        g = cell_genes(k)
        rows.append({"config": {"design": DESIGN_DOC, "cell": k,
                                "genes": {kk: (str(v) if v == math.inf else v) for kk, v in g.items()},
                                "entry": "라이브 기본 StrategyConfig(require_flow=False) GA2 유전자", "universe": "H5 ADV20≥5,000",
                                "events": [g1.EVENT_LO, g1.EVENT_HI]}})
    n = g1.write_ledger(path, rows, seen)
    _log(f"원장 {path}: 새 줄 {n}", t0)
    return n


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="ga_run 사전계산(.npy 디렉터리) — 읽기 전용")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--noise", type=int, default=N_NOISE)
    ap.add_argument("--chunk", type=int, default=480)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--save", default=None, help="널 지표 배열(.npz) 저장 경로(레포 밖)")
    ap.add_argument("--load", default=None, help="저장한 지표 배열로 보고서만 다시 쓴다")
    args = ap.parse_args(argv)
    t0 = time.time()
    import concurrent.futures as cf
    import multiprocessing as mp

    import ga2_run as g2

    from swing_it.live.config import DEFAULT_STRATEGY

    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(min(args.workers, 3))
    worker_init(args.cache, min(args.workers, 3))
    pre = _W["pre"]
    E = g1.Engine(pre, max_exit=2)
    genes19 = DEFAULT_STRATEGY.to_genes(flow=False)
    spec = g2.g2_spec(genes19)
    fill_all, cidx = E.fill_ctl(spec["ek"], spec["w_idx"], spec["c_idx"])
    _log(f"사전계산 쌍 {pre.P:,} · 진입 = 라이브 기본(수급 제외) {g2.gene_str(genes19)}", t0)
    worlds = [build_world(pre, E, spec, fill_all, cidx, None)]
    _log(f"실제 세계: 후보 {len(worlds[0]['M'])} (체결 {int(worlds[0]['fill'].sum())}) · 시뮬 대상 {len(worlds[0]['U'])}", t0)
    for r in range(args.noise):
        worlds.append(build_world(pre, E, spec, fill_all, cidx, r))
    _log(f"셔플 세계 {args.noise}벌: 후보 {[len(w['M']) for w in worlds[1:4]]}… · 시뮬 대상 중앙값 "
         f"{int(np.median([len(w['U']) for w in worlds[1:]])) if args.noise else 0}", t0)

    info: dict = {"t_start": time.strftime("%Y-%m-%d %H:%M"), "workers": args.workers, "cells": N_CELLS,
                  "noise": args.noise, "genes19": genes19, "n_cand": [len(w["M"]) for w in worlds],
                  "n_fill": [int(w["fill"].sum()) for w in worlds], "n_U": [len(w["U"]) for w in worlds]}
    if args.load:
        z = np.load(args.load, allow_pickle=False)
        res = z["res"]
        info.update(json.loads(str(z["info"])))
    else:
        write_trials(os.path.join(args.out, LABEL, "TRIALS.jsonl"), t0)
        info["parity"] = parity_checks(pre, E, spec, worlds[0], t0)
        # 시간 추정(실제 세계 칸 40개)
        tt = time.time()
        a = _world_arrays(worlds[0])
        for k in range(0, N_CELLS, N_CELLS // 40):
            eval_cell(a, cell_genes(k))
        per = (time.time() - tt) / 40
        _log(f"칸당 {per * 1e3:.1f}ms → 예상 {per * N_CELLS * len(worlds) / max(args.workers, 1) / 60:.1f}분(작업자 {args.workers})", t0)
        res = np.full((len(worlds), N_CELLS, len(METRICS)), np.nan)
        tasks = [{"world": w, "cells": list(range(i, min(i + args.chunk, N_CELLS)))}
                 for w in worlds for i in range(0, N_CELLS, args.chunk)]
        ctx = mp.get_context("spawn")
        done = 0
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=worker_init,
                                    initargs=(args.cache, 1)) as ex:
            futs = [ex.submit(cell_task, t) for t in tasks]
            for f in cf.as_completed(futs):
                r = f.result()
                wi = 0 if r["wid"] < 0 else r["wid"] + 1
                res[wi, r["cells"]] = r["res"]
                done += 1
                if done % 30 == 0 or done == len(tasks):
                    _log(f"작업 {done}/{len(tasks)}", t0)
        info["sec_grid"] = time.time() - tt
        if args.save:
            np.savez_compressed(args.save, res=res, info=json.dumps(info, default=str))
    info["sec_total"] = time.time() - t0
    # 대표·현 칸 상세(실제 세계)
    a = _world_arrays(worlds[0])
    detail = {}
    for name, g in (("cur", CUR), ("rep", REP)):
        vec, tr, pdict = eval_cell(a, g, trades=True)
        detail[name] = describe(pre, vec, tr, pdict)
    write_report(args.out, res, info, detail, t0)
    write_cells_csv(os.path.join(args.out, CELLS_CSV), res)
    _log("끝", t0)
    return 0


def describe(pre, vec: np.ndarray, tr: dict, pdict: dict) -> dict:
    """대표·현 칸 상세 — 연도별 R, 청산 사유, α 날짜 군집 CI, 포트폴리오 블록 부트스트랩 CI."""
    R = tr["ret"] / tr["d"]
    yrs = pre.days[tr["pd"] + 1].year.to_numpy()
    by_year = {int(y): {"n": int((yrs == y).sum()), "r_mean": float(R[yrs == y].mean()),
                        "r_med": float(np.median(R[yrs == y])),
                        "win": float(((tr["ret"] - tr["cost"])[yrs == y] > 0).mean())} for y in np.unique(yrs)}
    ex = tr["ret"] - tr["ctl"]
    ok = np.isfinite(ex)
    a, alo, ahi = g1.cluster_mean_ci(ex[ok], tr["pd"][ok])
    v = gf.block_boot(pdict["monthly"], 3, 2000, 0)
    reasons = pd.Series([gf.REASON_NAMES[k] for k in tr["reason"]]).value_counts().to_dict()
    return {"vec": vec, "by_year": by_year, "alpha": (a, alo, ahi), "port_ci": (float(np.percentile(v, 2.5)),
                                                                                  float(np.percentile(v, 97.5))),
            "reasons": reasons, "R": R}


# ---------------------------------------------------------------------------
# 보고
# ---------------------------------------------------------------------------

def write_cells_csv(path: str, res: np.ndarray) -> None:
    rows = []
    real = res[0]
    p95 = np.nanpercentile(res[1:], 95, axis=0) if res.shape[0] > 1 else np.full_like(real, np.nan)
    for k in range(N_CELLS):
        g = cell_genes(k)
        row = {"cell": k, **{kk: ("off" if v is None else v) for kk, v in g.items()}}
        row.update({m: real[k, i] for i, m in enumerate(METRICS)})
        row.update({f"null_p95_{m}": p95[k, MI[m]] for m in ("win", "r_mean", "r_med", "payoff", "alpha", "port_ann",
                                                              "be_exit", "full_stop")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


def _p95(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95)) if len(x) else np.nan


def landscape(res0: np.ndarray, metric: str) -> dict:
    """유전자별 주효과 — 단계마다 (그 단계 칸들의 중앙값, 25·75 백분위)."""
    v = res0[:, MI[metric]].reshape(SHAPE)
    out = {}
    for ax, g in enumerate(GENES):
        moved = np.moveaxis(v, ax, 0).reshape(SHAPE[ax], -1)
        out[g] = [(float(np.nanmedian(r)), float(np.nanpercentile(r, 25)), float(np.nanpercentile(r, 75))) for r in moved]
    return out


def neighbors(k: int) -> list[int]:
    idx = np.unravel_index(k, SHAPE)
    out = []
    for ax in range(len(SHAPE)):
        for j in range(SHAPE[ax]):
            if j != idx[ax]:
                nb = list(idx)
                nb[ax] = j
                out.append(int(np.ravel_multi_index(tuple(nb), SHAPE)))
    return out


def _lv(g: str, v) -> str:
    if g == "strength":
        return xv.STRENGTH_NAMES[v]
    if v is None:
        return "끔"
    if v == math.inf:
        return "∞"
    if g == "take_frac":
        return "1/3" if v < 0.4 else "1/2"
    return f"{v:g}"


def write_report(out_dir: str, res: np.ndarray, info: dict, detail: dict, t0: float) -> None:
    kc, kr = cell_index(CUR), cell_index(REP)
    real = res[0]
    null = res[1:]
    nn = null.shape[0]
    L: list[str] = []
    w = L.append
    w("# 미너비니 청산 v2 — 시험 1 결과 (러너 자동 생성, 판정 아님)")
    w("")
    w(f"설계: `{DESIGN_DOC}` — 결과 보기 전 고정. 러너: `research/experiments/exits_v2_run.py` · 커널: `research/signals/exits_v2_fast.py`. "
      f"생성 {date.today().isoformat()}. **이 문서는 숫자와 설계 판단 규칙의 산술 비교만 적는다 — 라이브 기본 청산을 바꿀지는 사람이 정한다.**")
    w("")
    w("## 설정")
    w("")
    w(f"- 진입 고정: 라이브 기본 `StrategyConfig`(수급 F 제외 — `require_flow=False`, 뉴스 거부권 없음) = GA2 유전자 "
      f"`{info['genes19']}`. 사건 {g1.EVENT_LO} ~ {g1.EVENT_HI}, 유니버스 H5(위생 ∧ ADV20 ≥ 5,000 = 유동 ∪ 중소형), 추격 5%(피벗 50일), "
      f"상한가 시가 버림. 사전계산 = GA 캐시 `pre_v2`(읽기 전용).")
    w(f"- 실제 세계 후보(마스크 통과 쌍) {info['n_cand'][0]} · 체결 {info['n_fill'][0]} · 시뮬 대상(후보 ∪ 대조) {info['n_U'][0]}.")
    w(f"- 청산 격자: {' × '.join(f'{g} {len(LEVELS[g])}' for g in GENES)} = **{N_CELLS:,} 칸 전부**. 고정: 손절 하한 3% · 수축 저점 × 0.99 · "
      "최대 보유 120 · 시간 손절 없음. 현 프로토콜 = `" + cell_str(CUR) + "` (칸 " + str(kc) + "), 원전 대표 칸 = `" + cell_str(REP)
      + "` (칸 " + str(kr) + ").")
    w("- 트레이드 숫자: 겹침 제거(같은 종목 보유 중 신호 버림, 자본·슬롯 무제한). 승률·평균 이익/손실·손익비 = **순**(수익 − `korea.cost_rt`(청산일)), "
      "R = 총수익 / d. 꽉 찬 손절 = 남은 수량 전부가 기술적 손절(앞선 강세 매도 없음). 본전 청산 = 마지막 청산이 본전선(익절 뒤 진입가 또는 be_r 의 진입가 × 1.005).")
    w("- α = 트레이드 총수익 − 대조 5개 평균(모멘텀 정합 — `ga_run` RELAX 층, 풀 = 체결 ∧ 추격 통과 H5 종목-일, 해시 추출 seed 0), **대조도 같은 칸 청산**으로 시뮬.")
    w("- 포트폴리오: GA2 처리량 엔진의 v2 판(`exits_v2_fast.portfolio_v2` — 10슬롯 × 10%, rel → RS 순, 미체결 주문은 그날 슬롯만, 날별 정합 초과 "
      "= 수익 − 대조(보유일 일할) − 비용(청산일), 짝 없는 포지션 0). 전 구간 — 절단 없음, 첫 진입일 ~ 데이터 끝(2026-10-08). 연환산 = 월합 평균 × 12, "
      "MDD = 단리 누적 초과의 고점 대비 최대 하락(자산 대비).")
    w(f"- 널: 진입 피처를 같은 신호월 안에서 쌍끼리 재배정(`ga2_run.set_noise` 와 같은 식·시드 1000+k) **{nn}벌 × 같은 {N_CELLS:,} 칸 전부**. "
      "칸별 95 백분위, 그리고 (대표 − 현) 차이의 세계별 분포.")
    w("")
    w("### 구현 해석(실행 전 결정 — 설계 문구가 정하지 않은 곳)")
    w("")
    w("- 강세 매도는 **한 트레이드에 한 번**(\"절반 매도 규칙\"), 남은 수량의 절반. 부분 익절도 **남은 수량** × take_frac(라이브 `_partial_qty` 와 같다). "
      "강세 매도는 익절 전·후 어느 단계에서도 걸린다. 하루 상승률 = 종가 / **직전 행** 종가 − 1(직전 행이 NaN 이면 그 조건 거짓), MA50 = 단순 50일 종가 평균.")
    w("- 본전선은 손절선을 **대체**한다(본전선 > 진입가 > 기술적 손절가라 같은 날 둘 다 닿으면 가격은 본전선을 먼저 지난다 → 본전선 체결, 사유 breakeven). "
      "be_r 발동일 당일은 기술적 손절 그대로(갭 예외 없음). 익절 뒤 본전(진입가, 다음 거래일부터 · 갭 익절일 예외)은 옛 규칙 그대로; be_r 도 발동했으면 높은 쪽(진입가 × 1.005).")
    w("- 같은 종가에 다음날 시가 결정이 여럿이면 하나만(강세 매도 > 돌파 실패 > 추적). 강세 매도가 이긴 날 돌파 실패·추적은 다음 종가에 다시 본다(연속 이탈 횟수는 계속 센다). "
      "돌파 실패는 익절 전에만(옛 규칙), 연속 이탈일은 거래 가능일 종가만 센다(정지일은 세지도 끊지도 않는다).")
    w("- fb_tol \"끔\" × fb_days 2단계, take_r ∞ × take_frac 2단계, 강세 매도 끔 등은 같은 결과의 칸이 겹친다 — 설계 격자를 그대로 두고 지형의 중앙값에도 그대로 넣었다.")
    w("")
    par = info.get("parity", {})
    if par:
        w("## 패리티(실데이터)")
        w("")
        k_ = par["kernel"]
        w(f"- v2 커널(현 칸 = v2 전부 끔) vs 옛 GA 커널(`ga_fast.sim_book`, 사전등록 + 돌파 실패 · 시간 손절 없음): 쌍 {k_['pairs']:,} (체결 {k_['ok']:,}) — "
          f"상태 {k_['status_eq']} · 청산 행 {k_['x_eq']} · 사유 {k_['reason_eq']} · 수익 비트 일치 {k_['ret_bitwise']}.")
        p_ = par["portfolio"]
        w(f"- v2 포트폴리오 vs GA2 `Engine.portfolio`(같은 진입·옛 청산·절단 없음): 날별 최대 차 {p_['max_abs_daily']:.1e} (옛 엔진 r2 float32), "
          f"주문 {p_['orders_old']} / {p_['orders_v2']}, 진입 {p_['opened_old']} / {p_['opened_v2']}.")
        w("- 연구 시뮬레이터 `simulate_minervini`(v2 키워드) vs v2 엔진 트레이드(실제 세계 사건 전부, 겹침 제거):")
        w("")
        w("| 칸 | 사건 | 엔진 n | 연구 n | 키·청산일·사유 | 수익 최대 차 | 일치 | 청산 사유(연구) |")
        w("|---|---|---|---|---|---|---|---|")
        for r in par["research"]:
            w(f"| {r['name']} | {r['events']} | {r['n_engine']} | {r['n_ref']} | {r.get('key_eq')}·{r.get('exit_eq')}·{r.get('reason_eq')} | "
              f"{r.get('ret_maxdiff', float('nan')):.1e} | {'✓' if r['ok'] else '✗'} | "
              + " · ".join(f"{k} {v}" for k, v in r["reasons"].items()) + " |")
        w("")
        w("합성 데이터 3중 패리티(라이브 백테스트 = 연구 시뮬 = v2 커널, v2 8칸 × 시드 2 — 대표 칸 포함)는 `tests/test_live_genes.py`.")
        w("")
    # --- 핵심 표 ---
    dc, dr = detail["cur"], detail["rep"]
    w("## 현 프로토콜 vs 원전 대표 칸 (실제 세계)")
    w("")
    w("| | n | 승률(순) | 평균 이익 | 평균 손실 | 실현 손익비 | R 평균 | R 중앙값 | R 평균(순) | 꽉 찬 손절 | 본전 청산 | 보유 중앙값 | α/트레이드 [95% CI] | 포트폴리오 연환산 초과 [95% CI] | MDD |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, dd_ in (("현 프로토콜", dc), ("원전 대표 칸", dr)):
        v = dd_["vec"]
        a, alo, ahi = dd_["alpha"]
        plo, phi = dd_["port_ci"]
        w(f"| {name} | {int(v[MI['n']])} | {v[MI['win']]:.1%} | {_pct(v[MI['avg_win']])} | {_pct(v[MI['avg_loss']])} | {_f(v[MI['payoff']])} | "
          f"{_f(v[MI['r_mean']], '{:+.3f}')} | {_f(v[MI['r_med']], '{:+.3f}')} | {_f(v[MI['r_mean_net']], '{:+.3f}')} | {v[MI['full_stop']]:.1%} | "
          f"{v[MI['be_exit']]:.1%} | {v[MI['hold_med']]:.0f} | {_pct(a)} [{_pct(alo)}, {_pct(ahi)}] | {_pct(v[MI['port_ann']])} "
          f"[{_pct(plo)}, {_pct(phi)}] | {_pct(v[MI['port_mdd']])} |")
    w("")
    for name, dd_ in (("현 프로토콜", dc), ("원전 대표 칸", dr)):
        v = dd_["vec"]
        w(f"- {name}: 청산 사유 " + " · ".join(f"{k} {n}" for k, n in dd_["reasons"].items())
          + f" · 익절 {v[MI['partial']]:.0%} · 강세 매도 {v[MI['strength']]:.0%} · be_r 발동 {v[MI['be_moved']]:.0%} · "
          f"포트폴리오 주문 {int(v[MI['orders']])} · 진입 {int(v[MI['opened']])} · 짝 없음 {int(v[MI['unmatched']])}.")
    w("")
    # --- 판단 규칙 ---
    w("## 설계 판단 규칙의 숫자 (미리 적은 규칙 — 산술 비교만)")
    w("")
    w("규칙: 원전 대표 칸이 현 프로토콜보다 (1) 거래당 R 평균이 높고 (2) 10슬롯 포트폴리오 초과가 높고 (3) 그 차이가 셔플 진입 널의 같은 차이 분포 95 백분위 위면 "
      "라이브 기본 청산을 원전 대표 칸으로 바꾸는 사전등록 후보로 올린다. 승률만 오르면 바꾸지 않는다.")
    w("")
    w(f"| 지표 | 현 | 대표 | 차이(대표 − 현) | 널 차이 95 백분위({nn}벌) | 널 차이 중앙값 | 널에서 차이 ≥ 실제 비율 | 차이 > 널 95 백분위 |")
    w("|---|---|---|---|---|---|---|---|")
    rule = {}
    for m, lab, fmt in (("r_mean", "R 평균", "{:+.3f}"), ("port_ann", "포트폴리오 연환산 초과", None), ("win", "승률(순)", None),
                        ("r_med", "R 중앙값", "{:+.3f}"), ("alpha", "α/트레이드", None), ("payoff", "실현 손익비", "{:.2f}"),
                        ("port_mdd", "MDD", None)):
        c, r = real[kc, MI[m]], real[kr, MI[m]]
        diff = r - c
        nd = null[:, kr, MI[m]] - null[:, kc, MI[m]] if nn else np.array([])
        p95 = _p95(nd) if nn else np.nan
        share = float(np.mean(nd >= diff)) if nn else np.nan
        rule[m] = (c, r, diff, p95)

        def fm(x):
            return _f(x, fmt) if fmt else _pct(x)
        w(f"| {lab} | {fm(c)} | {fm(r)} | {fm(diff)} | {fm(p95)} | {fm(float(np.nanmedian(nd)) if nn else np.nan)} | "
          f"{_f(share, '{:.2f}')} | {diff > p95 if np.isfinite(p95) else 'nan'} |")
    w("")
    c1 = rule["r_mean"][1] > rule["r_mean"][0]
    c2 = rule["port_ann"][1] > rule["port_ann"][0]
    c3r = rule["r_mean"][2] > rule["r_mean"][3]
    c3p = rule["port_ann"][2] > rule["port_ann"][3]
    w(f"- (1) R 평균 대표 > 현: **{c1}** ({_f(rule['r_mean'][1], '{:+.3f}')} vs {_f(rule['r_mean'][0], '{:+.3f}')}).")
    w(f"- (2) 포트폴리오 초과 대표 > 현: **{c2}** ({_pct(rule['port_ann'][1])} vs {_pct(rule['port_ann'][0])}).")
    w(f"- (3) 차이 > 널 같은 차이 95 백분위: R 평균 **{c3r}** ({_f(rule['r_mean'][2], '{:+.3f}')} vs {_f(rule['r_mean'][3], '{:+.3f}')}), "
      f"포트폴리오 초과 **{c3p}** ({_pct(rule['port_ann'][2])} vs {_pct(rule['port_ann'][3])}).")
    w(f"- 승률: 현 {rule['win'][0]:.1%} → 대표 {rule['win'][1]:.1%} (널 차이 95 백분위 {_pct(rule['win'][3], 1)}).")
    w("- 세 조건이 모두 참인지와 그 뒤의 행동(사전등록 후보로 올릴지)은 사람이 읽고 정한다.")
    w("")
    # --- 널: 칸별 95 백분위 ---
    w("## 셔플 진입 널 — 칸별 95 백분위")
    w("")
    if nn:
        w("| 칸 | 지표 | 실제 | 널 중앙값 | 널 95 백분위 | 실제 > 95 백분위 |")
        w("|---|---|---|---|---|---|")
        for name, k in (("현", kc), ("대표", kr)):
            for m, lab, fmt in (("win", "승률(순)", None), ("r_mean", "R 평균", "{:+.3f}"), ("r_med", "R 중앙값", "{:+.3f}"),
                                ("payoff", "손익비", "{:.2f}"), ("alpha", "α/트레이드", None), ("port_ann", "포트폴리오 초과", None),
                                ("be_exit", "본전 청산 비율", None), ("full_stop", "꽉 찬 손절 비율", None)):
                x = real[k, MI[m]]
                nv = null[:, k, MI[m]]

                def fm(y):
                    return _f(y, fmt) if fmt else _pct(y)
                w(f"| {name} | {lab} | {fm(x)} | {fm(float(np.nanmedian(nv)))} | {fm(_p95(nv))} | {x > _p95(nv)} |")
        w("")
        w("격자 전체에서 실제 값이 **자기 칸의** 널 95 백분위를 넘는 칸 수(진입 정보가 없으면 기대 ≈ 5% = 720):")
        w("")
        w("| 지표 | 넘는 칸 | 비율 | 실제 최댓값(칸) | 널 세계별 격자 최댓값의 중앙값 · 95 백분위 |")
        w("|---|---|---|---|---|")
        for m, lab in (("win", "승률(순)"), ("r_mean", "R 평균"), ("alpha", "α/트레이드"), ("port_ann", "포트폴리오 초과")):
            p95c = np.nanpercentile(null[:, :, MI[m]], 95, axis=0)
            over = real[:, MI[m]] > p95c
            mx = null[:, :, MI[m]].max(axis=1)
            kb = int(np.nanargmax(real[:, MI[m]]))
            fmt = (lambda y: _f(y, "{:+.3f}")) if m == "r_mean" else _pct
            w(f"| {lab} | {int(over.sum()):,} | {over.mean():.1%} | {fmt(real[kb, MI[m]])} (칸 {kb}) | "
              f"{fmt(float(np.median(mx)))} · {fmt(_p95(mx))} |")
        w("")
        w("기계적 효과 확인 — 승률을 올리는 축(be_r)의 널 중앙값(셔플 진입에서도 오르는가):")
        w("")
        w("| be_r | 실제 승률 중앙값 | 널 승률 중앙값 | 실제 R 평균 중앙값 | 널 R 평균 중앙값 | 실제 포트폴리오 중앙값 | 널 포트폴리오 중앙값 |")
        w("|---|---|---|---|---|---|---|")
        for j, lv in enumerate(LEVELS["be_r"]):
            sl = [slice(None)] * len(SHAPE)
            sl[0] = j
            ks = np.arange(N_CELLS).reshape(SHAPE)[tuple(sl)].ravel()
            w(f"| {_lv('be_r', lv)} | {np.nanmedian(real[ks, MI['win']]):.1%} | {np.nanmedian(null[:, ks, MI['win']]):.1%} | "
              f"{np.nanmedian(real[ks, MI['r_mean']]):+.3f} | {np.nanmedian(null[:, ks, MI['r_mean']]):+.3f} | "
              f"{_pct(np.nanmedian(real[ks, MI['port_ann']]))} | {_pct(np.nanmedian(null[:, ks, MI['port_ann']]))} |")
        w("")
    # --- 연도별 R ---
    w("## 연도별 R (진입 연도)")
    w("")
    w("| 연도 | 현 n | 현 R 평균 | 현 R 중앙값 | 현 승률 | 대표 n | 대표 R 평균 | 대표 R 중앙값 | 대표 승률 |")
    w("|---|---|---|---|---|---|---|---|---|")
    ys = sorted(set(dc["by_year"]) | set(dr["by_year"]))
    for y in ys:
        a = dc["by_year"].get(y, {"n": 0, "r_mean": np.nan, "r_med": np.nan, "win": np.nan})
        b = dr["by_year"].get(y, {"n": 0, "r_mean": np.nan, "r_med": np.nan, "win": np.nan})
        w(f"| {y} | {a['n']} | {_f(a['r_mean'], '{:+.3f}')} | {_f(a['r_med'], '{:+.3f}')} | {_f(a['win'], '{:.0%}')} | {b['n']} | "
          f"{_f(b['r_mean'], '{:+.3f}')} | {_f(b['r_med'], '{:+.3f}')} | {_f(b['win'], '{:.0%}')} |")
    w("")
    # --- 지형 ---
    w("## 지형 — 유전자별 주효과(다른 유전자 전부에 대한 중앙값 [25, 75 백분위])")
    w("")
    for m, lab, fmt in (("win", "승률(순)", None), ("r_mean", "R 평균", "{:+.3f}"), ("port_ann", "포트폴리오 연환산 초과", None),
                        ("alpha", "α/트레이드", None)):
        ls = landscape(real, m)
        w(f"**{lab}**")
        w("")
        w("| 유전자 | 단계별 중앙값 [25, 75] | 단계 중앙값 폭(최대 − 최소) |")
        w("|---|---|---|")

        def fm(y):
            return _f(y, fmt) if fmt else _pct(y)
        for g in GENES:
            cells_ = " · ".join(f"{_lv(g, lv)}: {fm(md)} [{fm(lo)}, {fm(hi)}]" for lv, (md, lo, hi) in zip(LEVELS[g], ls[g],
                                                                                                         strict=True))
            meds = [md for md, _, _ in ls[g]]
            w(f"| {g} | {cells_} | {fm(max(meds) - min(meds))} |")
        w("")
    w("### 고원 vs 봉우리 — 포트폴리오 초과 상위 20칸의 한 유전자 이웃")
    w("")
    w("이웃 = 유전자 하나만 다른 칸(모두 " + str(sum(s - 1 for s in SHAPE)) + "개). 이웃 중앙값이 칸 값에 가까우면 고원, 크게 떨어지면 봉우리.")
    w("")
    w("| 순위 | 칸 | 설정 | 포트폴리오 초과 | 이웃 중앙값 | 이웃 > 0 비율 | R 평균 | 승률 | 널 95 백분위(그 칸) |")
    w("|---|---|---|---|---|---|---|---|---|")
    pa = real[:, MI["port_ann"]]
    order = np.argsort(-np.nan_to_num(pa, nan=-np.inf))
    p95c = np.nanpercentile(null[:, :, MI["port_ann"]], 95, axis=0) if nn else np.full(N_CELLS, np.nan)
    for rnk, k in enumerate(order[:20], 1):
        nb = neighbors(int(k))
        w(f"| {rnk} | {k} | {cell_str(cell_genes(k))} | {_pct(pa[k])} | {_pct(float(np.nanmedian(pa[nb])))} | "
          f"{np.mean(pa[nb] > 0):.0%} | {real[k, MI['r_mean']]:+.3f} | {real[k, MI['win']]:.0%} | {_pct(p95c[k])} |")
    w("")
    rk = lambda k: int((pa > pa[k]).sum()) + 1  # noqa: E731
    w(f"- 현 프로토콜 순위 {rk(kc):,} / {N_CELLS:,} (이웃 중앙값 {_pct(float(np.nanmedian(pa[neighbors(kc)])))}), "
      f"원전 대표 칸 순위 {rk(kr):,} / {N_CELLS:,} (이웃 중앙값 {_pct(float(np.nanmedian(pa[neighbors(kr)])))}).")
    w(f"- 격자 전체 포트폴리오 초과: 중앙값 {_pct(float(np.nanmedian(pa)))} · > 0 비율 {np.mean(pa > 0):.0%} · 최댓값 {_pct(float(np.nanmax(pa)))}; "
      f"R 평균 중앙값 {np.nanmedian(real[:, MI['r_mean']]):+.3f} · 승률 중앙값 {np.nanmedian(real[:, MI['win']]):.1%}.")
    w("")
    w("## 실행")
    w("")
    w(f"- 시작 {info.get('t_start')} · 작업자 {info.get('workers')} · 격자 평가 {info.get('sec_grid', float('nan')) / 60:.1f}분 · 전체 {info['sec_total'] / 60:.1f}분 · "
      f"세계 {1 + nn}(실제 1 + 셔플 {nn}) × 칸 {N_CELLS:,} = {(1 + nn) * N_CELLS:,} 칸 평가.")
    w(f"- 칸 표: `{OUT_DIR}/{CELLS_CSV}`(git 무시 — 재생성: 위 Run 줄). 원장: `{OUT_DIR}/{LABEL}/TRIALS.jsonl` (칸마다 한 줄, 평가 전 기록).")
    w("")
    w("시험 2(GA2 메타 세대 1 에 고원인 축만 투입)는 이 지형을 사람이 읽은 뒤 — 이 러너는 하지 않는다.")
    path = os.path.join(out_dir, RESULTS_MD)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    _log(f"보고서 {path}", t0)


if __name__ == "__main__":
    raise SystemExit(main())
