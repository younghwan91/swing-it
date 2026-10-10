#!/usr/bin/env python
"""패턴 GA — 수렴 모양·천장·진입 기준선·돌파 실패·본전·확장 청산을 유전자로, CMAwM + BIPOP × 워크포워드. **리포터 — 판정 아님.**

사전등록: ``research/logs/sepa_explore/PATTERN_GA.md`` — aeaf8e5 + 정정 0 ab612dc + **정정 1 72fab71**(``REVIEW_PATTERN_GA.md`` 의
"정정 1 초안" 12항목 + MINOR 1~12 를 그대로 채택 — 그 리뷰 절이 정본) + 정정 2 c483edb(급등 뒤 쉼) + 정정 3 2154f77(천장 종류·H0 첫 점·
바닥 진입·위생), 모두 결과 보기 전. 숫자를 ``PATTERN_GA_RESULTS.md`` 에 적고 바 문턱과의 **산술 비교**만 한다.

- 사건 표: :mod:`research.signals.pattern_events` (θ 6 × 천장 창 3 × 기준선 4 = 72 표, 첫 관통일만, d−1 서술자) — 설정과 무관하게 한 번
  만들어 ``--cache`` 디렉터리에 .npy 로 두고 작업자가 memmap 으로 함께 읽는다.
- 평가: :mod:`research.signals.pattern_fast` (거르기 → 매수정지 체결 → 확장 청산 경로(들어간 트레이드만) → 10 슬롯 포트폴리오 − 종목별
  모멘텀 정합 바스켓). 적합도 = 학습창 일별 초과의 월합 3개월 블록 부트스트랩(500, 공통 난수 seed 0) 연환산 2.5% 하단 − 연속 벌점
  λ·[max(0, 12·연수 − n)/(12·연수) + 연평균 max(0, 4·연 비율 − n_y)/(4·연 비율)], λ = 0.20(리뷰 C3-4·M6-3). 비용 2×(주).
- 워크포워드: ``swing_it.validation.walkforward.FOLDS`` 6개(3년 학습 → 1년 시험, 마지막 시험 2025-01 ~ 사건 2026-08-31). 학습 트레이드 =
  진입 ∈ 학습창, 학습창 끝(test_lo 직전 거래일 종가) 시가평가(purge 금지). 폴드마다 CMAwM + BIPOP 시드 10 × 런당 10,000 평가.
  **절차 출력(M2)** = 시드 최적 10개를 새 부트스트랩 난수(seed :data:`REEVAL_SEED`)로 학습창에서 다시 평가한 1등. 표본 밖 = 폴드 출력
  config 를 시험 연도 신규 진입에만 쓰는 **하나의 연속 포트폴리오**(열린 포지션은 원 config 청산), 데이터 끝 2026-09-11 시가평가.
- 널: (2a) 폴드마다 균등 추출 풀(:data:`POOL_N`)에서 학습창 실현 가능(벌점 0) 설정을 폴드별 독립으로 뽑아 이어 붙인 OOS 300벌,
  (2b) 모양 서술자를 같은 달 안에서 섞은 사건 표 위에서 **절차 전체**(폴드 6 × 시드 10 × 10,000) 50벌. 기술: 같은 풀의 무작위 탐색
  best-of-2,000 300벌, 청산 전용 대조 300벌(GA 출력의 청산 유전자 고정, 진입·모양 무작위).
- 원장: ``research/logs/sepa_explore/pattern_ga/TRIALS.jsonl`` — 절차·널·민감도 칸과 설계 진단(건수만)을 결과 전에, 런 요약은 런이
  끝날 때마다(결과 문서보다 먼저).

Run:  uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/pattern_ga_run.py --feasibility-only
      uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/pattern_ga_run.py --workers 12
디버그: ... pattern_ga_run.py --debug --out <scratch>/pga_debug   (폴드 2 · 시드 2 · 예산 300 · 널 축소 — 콘솔은 건수·시간만)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from research.signals import ga_fast as gf  # noqa: E402
from research.signals import pattern_events as PE  # noqa: E402
from research.signals import pattern_fast as PF  # noqa: E402
from research.signals.sepa import LIMIT_UP_RATIO  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "pattern_ga"
SUB = "pattern_ga"
RESULTS_MD = "PATTERN_GA_RESULTS.md"
FEAS_MD = "FEASIBILITY.md"
PREREG_DOC = ("research/logs/sepa_explore/PATTERN_GA.md (aeaf8e5 + 정정 0 ab612dc + 정정 1 72fab71 = REVIEW_PATTERN_GA.md 정정 1 초안·"
              "MINOR 1~12 + 정정 2 c483edb + 정정 3 2154f77)")
DEFAULT_CACHE = os.environ.get("PATTERN_GA_CACHE", "/tmp/pattern_ga_cache")
CACHE_VERSION = "pga-v3"                # 사건 표 정의가 바뀌면 올린다(캐시 무효화)

# --- 사전등록 수치(탐색하지 않는다) ---
EVENT_LO, EVENT_HI = "2017-09-15", "2026-08-31"
END_DAY = "2026-09-11"                 # 데이터 끝(NXT 절단) — 시가평가(M2-3)
ADV_FLOOR = 1_000.0                    # ADV20 ≥ 10억(백만원)
MINUTE_LO = "2025-08-01"
COST_MULT = 2.0
SEEDS = tuple(range(10))
BUDGET = 10_000
N_NULL_A, N_NULL_B, N_RS, N_EXIT_CTL = 300, 50, 300, 300
POOL_N = 100_000                       # 폴드당 균등 추출 풀(리뷰 C1 풀 방식 10~20만)
RS_BEST_OF = 2_000
FIT_BOOT, FIT_SEED, REEVAL_SEED, CI_BOOT = 500, 0, 7_919, 2_000
LAMBDA, FLOOR_PER_YEAR, FLOOR_YEAR = 0.20, 12.0, 4.0
OFF_ZONE = 0.02
POOL_SEED, NULL_A_SEED, RS_SEED, EXIT_CTL_SEED, NULLB_SEED = 20261013, 20261014, 20261015, 20261016, 20261012
CHART_SEED, CHART_N = 20261011, 12
SLIP_SENS = (0.003, 0.005)
ORDER_SEED = 20261017
TOP_PBO, PBO_BLOCKS = 100, 12
FOLD_MIN_FILLS = 20                    # 바 4: 체결 < 20 폴드는 판단 불가
B6_N, B6_GL, B6_MARGIN, BE_BAND = 60, 2.0, 0.05, 0.005
DSR_STRUCTS = 12                       # 이 계열 검증 구조 수(리뷰 M3 — 10~12)
N_FEAS = 3_000                         # 실현 가능 비율 — 균등 추출 수(건수만)
FLAT = 0.15                            # 모양 이름: |ŝ| ≤ 0.15 = 평평(MINOR 9, 결과 전 고정)
REGION_RUNUP, REGION_REST = 0.30, 20.0  # 정정 2 착상 영역 좌표

# --- 유전자(정정 0·1·2·3) ---
#: 연속 (이름, 하한, 상한, 끔 끝) — [0, 1] 정규화, 끔 끝에서 :data:`OFF_ZONE` 안이면 조건 끔.
CONT = (("rs_min", 0.0, 90.0, "lo"), ("breadth_min", 0.0, 0.7, "lo"),
        ("sup_c", -1.5, 1.0, None), ("sup_hw", 0.05, 1.5, "hi"), ("slo_c", -1.0, 1.5, None), ("slo_hw", 0.05, 1.5, "hi"),
        ("rho_max", 0.2, 1.5, "hi"), ("span_c", math.log(15.0), math.log(120.0), None), ("span_hw", 0.1, 1.5, "hi"),
        ("ceil_min", 0.70, 1.00, "lo"), ("depth_max", 0.15, 0.50, "hi"), ("dry_max", 0.5, 1.5, "hi"),
        ("runup_min", 0.0, 1.0, "lo"), ("rest_gap_max", 3.0, 120.0, "hi"),
        ("gap_max", 0.01, 0.05, None), ("stop_cap", 0.05, 0.12, None), ("fb_tol", 0.0, 0.03, None), ("trail_k", 2.0, 5.0, None))
#: 명목형(M2-1) — 범주별 연속 로짓 [0, 1] → argmax(동률 = 앞).
NOM = (("stage", (0, 1, 2, 3)), ("entry", PE.BASES), ("trail", ("ma20", "ma50", "chand")),
       ("pyramid", ("off", "1r", "newhigh")))
#: 이산(순서형 — CMAwM 정수 축).
DISC = (("ceil_win", PE.CEIL_WINS), ("theta", PE.THETAS), ("runup_win", PE.RUNUP_WINS), ("touch_lo", (2, 3)),
        ("e_min", (None, 0.0, 0.25, 0.5)), ("e_accel", (0, 1)), ("stop_anchor", ("E", "lower")), ("fb_days", (1, 2, 3, 0)),
        ("be_r", (0.5, 1.0, 1.5, 2.0, math.inf)), ("take_r", (2.0, 3.0, 4.0, math.inf)), ("take_frac", (1 / 3, 1 / 2)),
        ("tighten", (0, 1)), ("ceil_kind", ("single", "resist")), ("h0_in_pattern", (0, 1)))
N_LOGIT = sum(len(v) for _, v in NOM)
N_CONT = len(CONT) + N_LOGIT
DISC_SIZES = tuple(len(v) for _, v in DISC)
DIM = N_CONT + len(DISC)
GENE_NAMES = tuple(n for n, *_ in CONT) + tuple(n for n, _ in NOM) + tuple(n for n, _ in DISC)
STAGE_NAMES = ("MA 정렬만", "+ 종가 > MA50", "추세 템플릿 전부(d−1)", "H0 일 TT ∧ d−1 MA 정렬")
#: 모양 유전자(M6 모양 비활성 짝·C1-2b 와 같은 묶음 — 정정 3 의 천장 종류·H0 첫 점 포함).
SHAPE_GENES = ("sup_c", "sup_hw", "slo_c", "slo_hw", "rho_max", "touch_lo", "span_c", "span_hw", "ceil_min", "depth_max",
               "dry_max", "ceil_kind", "h0_in_pattern")
#: 청산 유전자(M1-f 청산 전용 대조가 고정하는 묶음).
EXIT_GENES = ("fb_days", "fb_tol", "be_r", "take_r", "take_frac", "trail", "trail_k", "tighten", "pyramid")


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _pct(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def _f(x, fmt: str = "{:.2f}") -> str:
    return "nan" if x is None or not (isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x)) else fmt.format(x)


# ---------------------------------------------------------------------------
# 1. 부호화 · 복호화 (순수 — tests/test_pattern_ga.py)
# ---------------------------------------------------------------------------

def canonical(g: dict) -> dict:
    """조건부로 평평한 유전자를 ``None`` 으로(같은 표현형 = 같은 키 — GA2 D-m1 ii). H0 기준선 × 신고가 불타기는 금지(M1-c → 끔)."""
    g = dict(g)
    if g["sup_hw"] is None:
        g["sup_c"] = None
    if g["slo_hw"] is None:
        g["slo_c"] = None
    if g["span_hw"] is None:
        g["span_c"] = None
    if g["runup_min"] is None:
        g["runup_win"] = None
    if g["e_min"] is None:
        g["e_accel"] = None
    if g["take_r"] == math.inf:
        g["take_frac"] = None
    if g["trail"] != "chand":
        g["trail_k"] = None
        g["tighten"] = None
    if g["entry"] == "h0" and g["pyramid"] == "newhigh":
        g["pyramid"] = "off"
    if g["entry"] == "bottom":                       # 정정 3: 손절 = E 고정, 익절 전 돌파 실패 없음
        g["stop_anchor"] = None
        g["fb_days"] = None
        g["fb_tol"] = None
    elif g["fb_days"] == 0:
        g["fb_tol"] = None
    return g


def decode(x) -> dict:
    """CMA 벡터(연속 18 + 로짓 14 [0,1] + 이산 14 단계 인덱스) → 유전자 dict(:func:`canonical`)."""
    x = np.asarray(x, float)
    out: dict = {}
    for i, (name, lo, hi, off) in enumerate(CONT):
        u = float(np.clip(x[i], 0.0, 1.0))
        if (off == "lo" and u <= OFF_ZONE) or (off == "hi" and u >= 1.0 - OFF_ZONE):
            out[name] = None
        else:
            out[name] = round(lo + u * (hi - lo), 4)
    j = len(CONT)
    for name, levels in NOM:
        v = x[j:j + len(levels)]
        out[name] = levels[int(np.argmax(v))]
        j += len(levels)
    for q, (name, levels) in enumerate(DISC):
        k = int(np.clip(round(float(x[N_CONT + q])), 0, len(levels) - 1))
        out[name] = levels[k]
    return canonical(out)


def encode(g: dict) -> np.ndarray:
    """유전자 → CMA 벡터(끔 = 끔 끝 값, ``None`` 인 조건부 유전자 = 범위 가운데·첫 단계). :func:`decode` 의 역(4자리 반올림 안에서)."""
    v = []
    for name, lo, hi, off in CONT:
        val = g.get(name)
        if val is None:
            v.append(0.0 if off == "lo" else (1.0 if off == "hi" else 0.5))
        else:
            v.append((val - lo) / (hi - lo))
    for name, levels in NOM:
        oh = [0.0] * len(levels)
        oh[levels.index(g[name])] = 1.0
        v += oh
    for name, levels in DISC:
        val = g.get(name)
        v.append(float(levels.index(val)) if val in levels else 0.0)
    return np.array(v)


def gkey(g: dict) -> tuple:
    return tuple(g.get(k) for k in GENE_NAMES)


def ghash(g: dict) -> int:
    return int.from_bytes(hashlib.blake2b(repr(gkey(g)).encode(), digest_size=8).digest(), "little", signed=True)


def x0_feasible() -> np.ndarray:
    """CMA 첫 런 평균(리뷰 C3-4) — 모든 모양 구간·맥락 거르기 끔(실현 가능점), 나머지는 범위 가운데."""
    x = np.full(DIM, 0.5)
    for i, (name, lo, hi, off) in enumerate(CONT):
        if off == "lo":
            x[i] = OFF_ZONE / 2
        elif off == "hi":
            x[i] = 1.0 - OFF_ZONE / 2
    for q, (name, levels) in enumerate(DISC):
        x[N_CONT + q] = {"e_min": 0.0, "touch_lo": 0.0, "ceil_kind": 0.0}.get(name, (len(levels) - 1) / 2.0)
    return x


def gene_vec(g: dict, extra_slip: float = 0.0) -> np.ndarray:
    """유전자 → :mod:`pattern_fast` 거르기 벡터(``gv``). 끔 = NaN."""
    def nz(v):
        return np.nan if v is None else float(v)

    gv = np.full(PF.N_G, np.nan)
    gv[PF.G_STAGE] = g["stage"]
    gv[PF.G_RS] = nz(g["rs_min"])
    gv[PF.G_BR] = nz(g["breadth_min"])
    if g["sup_hw"] is not None:
        gv[PF.G_SUP_LO], gv[PF.G_SUP_HI] = g["sup_c"] - g["sup_hw"], g["sup_c"] + g["sup_hw"]
    if g["slo_hw"] is not None:
        gv[PF.G_SLO_LO], gv[PF.G_SLO_HI] = g["slo_c"] - g["slo_hw"], g["slo_c"] + g["slo_hw"]
    gv[PF.G_RHO] = nz(g["rho_max"])
    gv[PF.G_TOUCH] = g["touch_lo"]
    if g["span_hw"] is not None:
        gv[PF.G_SPAN_LO], gv[PF.G_SPAN_HI] = math.exp(g["span_c"] - g["span_hw"]), math.exp(g["span_c"] + g["span_hw"])
    gv[PF.G_CEIL] = nz(g["ceil_min"])
    gv[PF.G_DEPTH] = nz(g["depth_max"])
    gv[PF.G_DRY] = nz(g["dry_max"])
    gv[PF.G_RUNUP_COL] = PE.RUNUP_WINS.index(g["runup_win"]) if g["runup_win"] is not None else 0
    gv[PF.G_RUNUP] = nz(g["runup_min"])
    gv[PF.G_REST] = nz(g["rest_gap_max"])
    gv[PF.G_EMIN] = nz(g["e_min"])
    gv[PF.G_EACC] = float(g["e_accel"] or 0)
    gv[PF.G_GAP] = g["gap_max"]
    gv[PF.G_ANCHOR] = 1.0 if g["stop_anchor"] == "lower" else 0.0
    gv[PF.G_CAP] = g["stop_cap"]
    gv[PF.G_FB_TOL] = nz(g["fb_tol"]) if g["fb_tol"] is not None else 0.0
    gv[PF.G_SLIP] = 0.001 + extra_slip
    gv[PF.G_DFLOOR] = 0.02
    gv[PF.G_BASE] = PE.BASES.index(g["entry"])
    gv[PF.G_H0IN] = float(g["h0_in_pattern"])
    gv[PF.G_CEILKIND] = 1.0 if g["ceil_kind"] == "resist" else 0.0
    return gv


def exit_vec(g: dict, extra_slip: float = 0.0) -> np.ndarray:
    """유전자 → 청산 인자 벡터(``px``)."""
    px = np.zeros(PF.N_X)
    px[PF.X_TAKE_R] = g["take_r"]
    px[PF.X_TAKE_FRAC] = g["take_frac"] if g["take_frac"] is not None else 0.5
    px[PF.X_TRAIL] = ("ma20", "ma50", "chand").index(g["trail"])
    px[PF.X_TRAIL_K] = g["trail_k"] if g["trail_k"] is not None else 3.0
    px[PF.X_TIGHTEN] = float(g["tighten"] or 0)
    px[PF.X_PYR] = ("off", "1r", "newhigh").index(g["pyramid"])
    px[PF.X_BE_R] = g["be_r"]
    px[PF.X_FB_DAYS] = 0 if g["fb_days"] is None else g["fb_days"]
    px[PF.X_MAX_HOLD] = 120
    px[PF.X_FROM_ENTRY] = 1.0
    px[PF.X_SLIP] = 0.001 + extra_slip
    px[PF.X_SELL_SLIP] = 0.001
    px[PF.X_BE_MULT] = 1.0
    px[PF.X_TIGHT_R] = 3.0
    px[PF.X_PYR_R] = 1.0
    px[PF.X_WB] = PF.WB_PYR
    return px


def table_of(g: dict) -> int:
    return PE.table_id(PE.THETAS.index(g["theta"]), PE.CEIL_WINS.index(g["ceil_win"]), PE.BASES.index(g["entry"]))


def shape_off(g: dict) -> dict:
    """모양 비활성 짝(M6-1): 모양 유전자만 끔(구간 전체), 나머지 그대로."""
    h = dict(g)
    h.update({"sup_c": None, "sup_hw": None, "slo_c": None, "slo_hw": None, "rho_max": None, "touch_lo": 2, "span_c": None,
              "span_hw": None, "ceil_min": None, "depth_max": None, "dry_max": None, "ceil_kind": "single"})
    return canonical(h)


def shape_name(cu, cl) -> str:
    """정규화 좌표 모양 이름(MINOR 9 — 결과 전 고정, |ŝ| ≤ :data:`FLAT` = 평평)."""
    if cu is None or cl is None or not (np.isfinite(cu) and np.isfinite(cl)):
        return "모양 끔(구간 없음)" if (cu is None and cl is None) else "한쪽만"
    up, lo = (0 if abs(cu) <= FLAT else (1 if cu > 0 else -1)), (0 if abs(cl) <= FLAT else (1 if cl > 0 else -1))
    if up == 0 and lo == 1:
        return "평평한 천장"
    if up == 1 and lo == 1:
        return "상승 쐐기" if cl > cu else "상승 채널/확산"
    if up == -1 and lo == -1:
        return "하락 쐐기" if cu < cl else "하락 채널/확산"
    if up == -1 and lo == 1:
        return "대칭 삼각"
    return "기타"


def config_shape(g: dict) -> str:
    return shape_name(g.get("sup_c") if g.get("sup_hw") is not None else None,
                      g.get("slo_c") if g.get("slo_hw") is not None else None)


def _falling_interval(g: dict) -> bool:
    """두 구간이 켜져 있고 전부 음수 ∧ 중심 ŝ_up < ŝ_lo (정정 1 C2 하락 쐐기 좌표의 구간판)."""
    if g.get("sup_hw") is None or g.get("slo_hw") is None:
        return False
    return (g["sup_c"] + g["sup_hw"] < 0) and (g["slo_c"] + g["slo_hw"] < 0) and g["sup_c"] < g["slo_c"]


def _rest_ok(g: dict) -> bool:
    return (g.get("runup_min") is not None and g["runup_min"] >= REGION_RUNUP and g.get("rest_gap_max") is not None
            and g["rest_gap_max"] <= REGION_REST)


def user_region(g: dict) -> dict:
    """사용자 착상 영역(결과 전 고정): W = 정정 1 C2(하락 쐐기 ∧ 추세선 ∧ fb 2) ∧ 정정 2(급등 ≥ 30% ∧ 쉼 ≤ 20일),
    WD = 정정 3(H0 = 첫 점 ∧ 바닥 반전) ∧ 하락 쐐기 ∧ 정정 2."""
    w = _falling_interval(g) and g["entry"] == "trend" and g.get("fb_days") == 2 and _rest_ok(g)
    wd = _falling_interval(g) and g["entry"] == "bottom" and g.get("h0_in_pattern") == 1 and _rest_ok(g)
    return {"W": bool(w), "WD": bool(wd), "any": bool(w or wd)}


def event_region(F: np.ndarray, base: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """사건별 착상 영역(건수 세기용, 가장 느슨한 창 — 60일 급등): W·WD 행 bool."""
    fi = PE.FI
    su, sl = F[:, fi["s_up"]], F[:, fi["s_lo"]]
    with np.errstate(invalid="ignore"):
        fall = (su < 0) & (sl < 0) & (su < sl) & (F[:, fi["runup60"]] >= REGION_RUNUP) & (F[:, fi["rest_gap"]] <= REGION_REST)
    w = fall & (base == PE.B_TREND)
    wd = fall & (base == PE.B_BOTTOM) & (F[:, fi["h0in"]] == 1)
    return w, wd


def median_config(cfgs: "list[dict]") -> "dict | None":
    """M2-2: 시드 최적이 모두 같은 모양 이름일 때만 — 연속은 중앙값, 이산은 단계 중앙값(짝수면 아래), 명목은 최빈(동률 = 앞)."""
    if len({config_shape(c) for c in cfgs}) != 1:
        return None
    xs = np.array([encode(c) for c in cfgs])
    med = np.empty(DIM)
    med[:len(CONT)] = np.median(xs[:, :len(CONT)], axis=0)
    j = len(CONT)
    for name, levels in NOM:
        cnt = np.zeros(len(levels))
        for c in cfgs:
            cnt[levels.index(c[name])] += 1
        oh = np.zeros(len(levels))
        oh[int(np.argmax(cnt))] = 1.0
        med[j:j + len(levels)] = oh
        j += len(levels)
    for q in range(len(DISC)):
        s = np.sort(xs[:, N_CONT + q])
        med[N_CONT + q] = s[(len(s) - 1) // 2]
    return decode(med)


def gene_str(g: dict) -> str:
    def c(v, fmt="{:.2f}"):
        return "끔" if v is None else fmt.format(v)

    def inf(v):
        return "끔" if v is None or v == math.inf else f"{v:g}"

    sup = "끔" if g["sup_hw"] is None else f"[{g['sup_c'] - g['sup_hw']:+.2f},{g['sup_c'] + g['sup_hw']:+.2f}]"
    slo = "끔" if g["slo_hw"] is None else f"[{g['slo_c'] - g['slo_hw']:+.2f},{g['slo_c'] + g['slo_hw']:+.2f}]"
    trk = "" if g["trail_k"] is None else f" k{g['trail_k']:.2f}"
    span = "끔" if g["span_hw"] is None else f"[{math.exp(g['span_c'] - g['span_hw']):.0f},{math.exp(g['span_c'] + g['span_hw']):.0f}]"
    return (f"θ{g['theta']:.0%} 창{g['ceil_win']} {g['entry']} 단계{g['stage']} rs≥{c(g['rs_min'], '{:.0f}')} br≥{c(g['breadth_min'])} "
            f"ŝ↑{sup} ŝ↓{slo} ρ≤{c(g['rho_max'])} 닿음{g['touch_lo']} L{span} 천장≥{c(g['ceil_min'])} 깊이≤{c(g['depth_max'])} "
            f"마름≤{c(g['dry_max'])} 급등{g['runup_win'] or ''}≥{c(g['runup_min'])} 쉼≤{c(g['rest_gap_max'], '{:.0f}')} "
            f"천장{g['ceil_kind']} H0첫점{g['h0_in_pattern']} E≥{c(g['e_min'])}{'가속' if g['e_accel'] else ''} 갭≤{g['gap_max']:.3f} "
            f"손절{g['stop_anchor'] or 'E'}≤{g['stop_cap']:.3f} | fb{g['fb_days'] if g['fb_days'] is not None else '-'}·"
            f"{c(g['fb_tol'], '{:.3f}')} be{inf(g['be_r'])} 익절{inf(g['take_r'])}R×{c(g['take_frac'])} 추적{g['trail']}"
            f"{trk}{'조임' if g['tighten'] else ''} 불타기{g['pyramid']}")


# ---------------------------------------------------------------------------
# 2. CMA-ES(CMAwM + BIPOP) — ``ga_run.cma_bipop`` + 실현 가능 첫 평균(리뷰 C3-4)
# ---------------------------------------------------------------------------

def cma_bipop_x0(fitness, n_cont: int, disc_sizes: "tuple[int, ...]", *, budget: int, seed: int, x0=None,
                 sigma0: float = 0.3, stall: bool = True, opt_cls=None):
    """``ga_run.cma_bipop`` 과 같은 재시작 규칙(Hansen 2009 BIPOP, ``stall`` 정체 재시작) — 첫 런 평균만 ``x0``(실현 가능점).
    ``ask`` 는 (평가용 x, tell 용 연속 x) 를 돌려준다 — 적합도는 평가용으로 재고 **tell 에는 표본 연속 x** 를 넘긴다(GA2 D-m1 iv).
    ``opt_cls`` = 시험용 대역(기본 ``cmaes.CMAwM``). 반환 (이력[(번호, x_eval, 값)], 재시작 기록)."""
    if opt_cls is None:
        from cmaes import CMAwM as opt_cls  # noqa: PLC0415, N813 — 선택 의존성(uv run --with cmaes==0.13.1)
    dim = n_cont + len(disc_sizes)
    bounds = np.array([[0.0, 1.0]] * n_cont + [[0.0, float(k - 1)] for k in disc_sizes])
    steps = np.array([0.0] * n_cont + [1.0] * len(disc_sizes))
    scale = bounds[:, 1] - bounds[:, 0]
    cov0 = np.diag(scale ** 2)
    rng = np.random.default_rng(seed)
    lam0 = 4 + int(3 * math.log(dim))
    hist, restarts = [], []
    n_eval, large_used, small_used, large_pop = 0, 0, 0, lam0
    first = True
    while n_eval < budget:
        mean = bounds[:, 0] + rng.random(dim) * scale
        if first:
            regime, lam, sig = "first", lam0, sigma0
            if x0 is not None:
                mean = np.clip(np.asarray(x0, float), bounds[:, 0], bounds[:, 1])
            first = False
        elif small_used < large_used:
            u = rng.random()
            regime, lam, sig = "small", max(2, int(lam0 * (0.5 * large_pop / lam0) ** (u ** 2))), sigma0 * 10 ** (-2 * rng.random())
        else:
            large_pop *= 2
            regime, lam, sig = "large", large_pop, sigma0
        opt = opt_cls(mean=mean, sigma=sig, bounds=bounds, steps=steps, seed=int(rng.integers(2 ** 31 - 1)),
                      population_size=lam, cov=cov0.copy())
        used, gens, run_best, since = 0, 0, -np.inf, 0
        stall_n = 10 + math.ceil(30 * dim / lam)
        why = "budget"
        while n_eval < budget:
            if opt.should_stop():
                why = "should_stop"
                break
            if stall and since >= stall_n:
                why = "stall"
                break
            sols = []
            for _ in range(opt.population_size):
                if n_eval >= budget:
                    break
                x_eval, x_tell = opt.ask()
                v = float(fitness(x_eval))
                hist.append((n_eval, x_eval.copy(), v))
                n_eval += 1
                used += 1
                sols.append((x_tell, -v))
            if len(sols) < opt.population_size:
                break
            opt.tell(sols)
            gens += 1
            g_best = -min(v for _, v in sols)
            if g_best > run_best + 1e-12:
                run_best, since = g_best, 0
            else:
                since += 1
        restarts.append({"regime": regime, "popsize": lam, "sigma": sig, "evals": used, "gens": gens, "why": why})
        if regime == "small":
            small_used += used
        else:
            large_used += used
    return hist, restarts


# ---------------------------------------------------------------------------
# 3. 엔진 — 캐시(memmap) 위 평가
# ---------------------------------------------------------------------------

ARR_NAMES = ("O", "H", "L", "C", "MA20", "MA50", "ATR", "TR", "LIM", "strat_cs", "bs", "bn", "cost_tab", "days_ns",
             "month_of", "year_of", "F", "I", "off", "codes")


class Win:
    """창: 진입 행 ``[d_lo, d_hi)`` (사건 구간과 교집합), 포트폴리오 날 ``[t_lo, t_end)`` (``t_end`` 직전 거래일 종가 시가평가)."""

    def __init__(self, E: "Engine", lo: str, hi: str, end: "str | None"):
        self.lo, self.hi, self.end = lo, hi, end
        self.d_lo = max(E.row(lo), E.row(EVENT_LO))
        self.d_hi = min(E.row(hi), int(E.days.searchsorted(pd.Timestamp(EVENT_HI), side="right")))
        self.t_lo = self.d_lo
        self.t_end = min(E.row(end), E.end_row + 1) if end else E.end_row + 1


class Engine:
    """캐시 디렉터리의 배열(memmap)과 평가 함수. ``F`` 는 :meth:`set_rep` 로 모양 섞은 사본과 바꿔 끼운다(2b)."""

    def __init__(self, cache: str, mmap: bool = True):
        mode = "r" if mmap else None
        for nm in ARR_NAMES:
            setattr(self, nm, np.load(os.path.join(cache, f"{nm}.npy"), mmap_mode=mode, allow_pickle=False))
        with open(os.path.join(cache, "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.days = pd.DatetimeIndex(np.asarray(self.days_ns).astype("datetime64[ns]"))
        self.end_row = int(self.days.searchsorted(pd.Timestamp(END_DAY), side="right") - 1)
        self.F_orig = self.F
        self.rep = None
        self.pmix = float(self.meta.get("p", 0.5))
        self.n_years = int(np.max(self.year_of)) + 1
        mx = int(np.max(np.diff(self.off))) if len(self.off) > 1 else 0
        self._buf = {k: np.empty(max(mx, 1), t) for k, t in (("idx", np.int64), ("e", float), ("sp", float), ("fb", float),
                                                               ("amb", np.bool_), ("skip0", np.bool_))}
        mo = np.asarray(self.month_of)
        self.month_year = np.zeros(int(mo.max()) + 1, np.int64)
        self.month_year[mo] = self.days.year.to_numpy()
        ydays = pd.Series(1, index=self.days).groupby(self.days.year).sum()
        self.year_len = {int(y): float(max(v, 245)) for y, v in ydays.items()}

    def row(self, date_s: str) -> int:
        return int(self.days.searchsorted(pd.Timestamp(date_s)))

    def set_rep(self, rep: "int | None") -> None:
        if rep == self.rep:
            return
        self.F = self.F_orig if rep is None else PE.shuffle_shape(np.asarray(self.F_orig), np.asarray(self.off), rep,
                                                                  seed=NULLB_SEED)
        self.rep = rep

    # --- 후보 ---
    def candidates(self, g: dict, w: Win, *, par: int = 0, extra_slip: float = 0.0, allow=None) -> dict:
        tid = table_of(g)
        a, b = PE.table_slice(self.off, tid)
        dcol = self.I[a:b, PE.II["d"]]
        aa = a + int(np.searchsorted(dcol, w.d_lo))
        bb = a + int(np.searchsorted(dcol, w.d_hi))
        B = self._buf
        n = PF.filter_events(self.F, aa, bb, gene_vec(g, extra_slip), float(LIMIT_UP_RATIO), B["idx"], B["e"], B["sp"],
                             B["fb"], B["amb"], B["skip0"]) if bb > aa else 0
        idx = B["idx"][:n].copy()
        keep = np.ones(n, bool) if allow is None else np.asarray(allow)[idx]
        I_, F = self.I, self.F
        out = {"ev": idx[keep], "e": B["e"][:n][keep].copy(), "sp": B["sp"][:n][keep].copy(), "fb": B["fb"][:n][keep].copy(),
               "amb": B["amb"][:n][keep].copy(), "skip0": B["skip0"][:n][keep].copy(), "n_pass": int(keep.sum())}
        ev = out["ev"]
        out.update(d=I_[ev, PE.II["d"]].astype(np.int64), ci=I_[ev, PE.II["ci"]].astype(np.int64),
                   h0=np.asarray(F[ev, PE.FI["h0"]], float), strat=np.asarray(self.F_orig[ev, PE.FI["strat"]]).astype(np.int64),
                   mkt=np.asarray(self.F_orig[ev, PE.FI["market"]]).astype(np.int64), par=np.full(len(ev), par, np.int64),
                   pid=(I_[ev, PE.II["i0"]].astype(np.int64) * 100_000 + I_[ev, PE.II["iD"]].astype(np.int64)))
        return out

    # --- 포트폴리오 ---
    def run(self, cands: "list[dict]", pxs: "list[np.ndarray]", t_lo: int, t_end: int, *, cost_mult: float = COST_MULT,
            pmix: "float | None" = None, order_seed: "int | None" = None, exclude=None) -> dict:
        cat = {k: np.concatenate([c[k] for c in cands]) if cands else np.zeros(0) for k in
               ("d", "ci", "e", "sp", "fb", "h0", "amb", "skip0", "strat", "mkt", "par", "ev", "pid")}
        if exclude is not None and len(cat["ev"]):
            m = ~np.isin(cat["ev"], np.asarray(list(exclude), np.int64))
            cat = {k: v[m] for k, v in cat.items()}
        if order_seed is not None and len(cat["d"]):
            rk = np.random.default_rng(order_seed).random(len(cat["d"]))
            o = np.lexsort((rk, cat["d"]))
            cat = {k: v[o] for k, v in cat.items()}
        n = len(cat["d"])
        L = t_end - t_lo
        ret, bench, expo = np.zeros(L), np.zeros(L), np.zeros(L)
        rec = np.zeros((max(n, 1), PF.N_Q))
        ny = np.zeros(self.n_years, np.int64)
        PX = np.ascontiguousarray(np.vstack(pxs)) if pxs else np.zeros((1, PF.N_X))
        res = PF.portfolio(cat["d"].astype(np.int64), cat["ci"].astype(np.int64), cat["e"].astype(float),
                           cat["sp"].astype(float), cat["fb"].astype(float), cat["h0"].astype(float),
                           cat["amb"].astype(np.bool_), cat["skip0"].astype(np.bool_), cat["strat"].astype(np.int64),
                           cat["mkt"].astype(np.int64), cat["par"].astype(np.int64), cat["ev"].astype(np.int64),
                           cat["pid"].astype(np.int64), PX, self.O, self.H, self.L, self.C, self.MA20, self.MA50, self.ATR,
                           self.TR, self.LIM, self.bs, self.bn, self.strat_cs, self.cost_tab, float(cost_mult),
                           float(self.pmix if pmix is None else pmix), int(t_lo), int(t_end), PF.SLOTS, PF.SLOT_W, ret,
                           bench, expo, rec, self.year_of, ny)
        n_taken = int(res[0])
        return {"ret": ret, "bench": bench, "exc": ret - bench, "expo": expo, "rec": rec[:n_taken], "n": n_taken, "ny": ny,
                "sk_slot": int(res[1]), "sk_held": int(res[2]), "unmatched": int(res[3]), "n_cand": n, "t_lo": t_lo,
                "t_end": t_end}

    def monthly(self, x: np.ndarray, t_lo: int) -> "tuple[np.ndarray, np.ndarray]":
        mo = np.asarray(self.month_of[t_lo:t_lo + len(x)])
        u, inv = np.unique(mo, return_inverse=True)
        return np.bincount(inv, weights=x), u

    def year_frac(self, w: Win) -> dict:
        """창의 연도별 비율(그해 거래일 중 창 안 거래일, ≤ 1) — 창마다 한 번 계산해 둔다."""
        if getattr(w, "_frac", None) is None:
            yrs = self.days[w.t_lo:w.t_end].year
            cnt = pd.Series(1, index=yrs).groupby(level=0).sum()
            w._frac = {int(y): min(1.0, v / self.year_len[int(y)]) for y, v in cnt.items()}
        return w._frac

    def penalty(self, r: dict, w: Win) -> "tuple[float, dict]":
        frac = self.year_frac(w)
        y0 = int(self.days[0].year)
        F_ = FLOOR_PER_YEAR * sum(frac.values())
        pen = max(0.0, F_ - r["n"]) / F_ if F_ > 0 else 0.0
        py = [max(0.0, FLOOR_YEAR * f - r["ny"][y - y0]) / (FLOOR_YEAR * f) for y, f in frac.items() if f > 0]
        pen_y = float(np.mean(py)) if py else 0.0
        return LAMBDA * (pen + pen_y), {"floor": F_, "years": frac, "pen_total": pen, "pen_year": pen_y,
                                        "feasible": bool(pen == 0 and pen_y == 0)}

    def evaluate(self, g: dict, w: Win, *, seed: int = FIT_SEED, counts_only: bool = False, allow=None,
                 want_monthly: bool = False) -> dict:
        """학습창 적합도(모듈 설명). ``counts_only`` = 진입 수·벌점만(수익 숫자를 만들지 않는다 — 실현 가능 비율용)."""
        c = self.candidates(g, w, allow=allow)
        r = self.run([c], [exit_vec(g)], w.t_lo, w.t_end)
        pen, pinfo = self.penalty(r, w)
        out = {"n": r["n"], "n_pass": c["n_pass"], "pen": pen, "feasible": pinfo["feasible"]}
        if counts_only:
            return out
        mo, _ = self.monthly(r["exc"], w.t_lo)
        lower = float(np.percentile(gf.block_boot(np.ascontiguousarray(mo), 3, FIT_BOOT, seed), 2.5)) if len(mo) >= 6 else -1.0
        out.update(fit=lower - pen, lower=lower)
        if want_monthly:
            out["monthly"] = mo
        return out

    def stitched(self, cfgs: "list[dict]", tests: "list[Win]", *, extra_slip: float = 0.0, cost_mult: float = COST_MULT,
                 pmix: "float | None" = None, order_seed: "int | None" = None, exclude=None) -> dict:
        """폴드별 config → 이어 붙인 OOS 연속 포트폴리오(폴드 시험 창 진입만 그 config, 열린 포지션은 원 config 청산)."""
        cands = [self.candidates(c, w, par=k, extra_slip=extra_slip) for k, (c, w) in enumerate(zip(cfgs, tests, strict=True))]
        pxs = [exit_vec(c, extra_slip) for c in cfgs]
        t_lo, t_end = tests[0].t_lo, tests[-1].t_end
        r = self.run(cands, pxs, t_lo, t_end, cost_mult=cost_mult, pmix=pmix, order_seed=order_seed, exclude=exclude)
        mo, mid = self.monthly(r["exc"], t_lo)
        r["monthly"], r["month_ids"] = mo, mid
        r["ann"] = float(mo.mean() * 12) if len(mo) else np.nan
        r["fold_ann"], r["fold_n"] = [], []
        starts = [int(self.month_of[min(w.d_lo, len(self.days) - 1)]) for w in tests] + [10 ** 9]
        for k, w in enumerate(tests):
            sel = (mid >= starts[k]) & (mid < starts[k + 1])
            r["fold_ann"].append(float(mo[sel].mean() * 12) if sel.any() else np.nan)
            r["fold_n"].append(int(((r["rec"][:, PF.Q_D] >= w.d_lo) & (r["rec"][:, PF.Q_D] < w.d_hi)).sum()) if len(r["rec"]) else 0)
        years = self.month_year[mid] if len(mid) else np.zeros(0, np.int64)
        def ann(m):
            return float(mo[m].mean() * 12) if m.any() else np.nan

        r["calm_ann"] = ann(np.isin(years, (2022, 2023, 2024)))
        r["ex_bundles_ann"] = ann(~np.isin(years, (2020, 2021, 2025, 2026)))
        r["ex2021_ann"] = ann(years != 2021)
        r["years"] = years
        return r


def fold_windows(E: Engine, folds) -> "list[tuple[Win, Win]]":
    """FOLDS → (학습 창: 진입 [train_lo, train_hi), test_lo 직전 시가평가), (시험 창: 진입 [test_lo, test_hi) ∩ 사건 구간)."""
    out = []
    for f in folds:
        out.append((Win(E, f.train_lo, f.train_hi, f.test_lo), Win(E, f.test_lo, f.test_hi, None)))
    return out


# ---------------------------------------------------------------------------
# 4. 작업자(spawn — 캐시는 memmap 으로 공유)
# ---------------------------------------------------------------------------

_W: dict = {}


def worker_init(cache: str, folds_spec: list) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(1)
    E = Engine(cache)
    from swing_it.validation.walkforward import Fold  # noqa: PLC0415
    folds = [Fold(*f) for f in folds_spec]
    _W.update(E=E, wins=fold_windows(E, folds))


def ga_run_one(E: Engine, w: Win, seed: int, budget: int, *, top: int = 0) -> dict:
    """CMA 한 런(학습창 w) → 최적·적합도·재시작·고유 config 수·해시·(선택) 학습 적합도 상위 config."""
    memo: dict = {}
    tt = time.time()

    def fn(x):
        g = decode(x)
        k = gkey(g)
        if k not in memo:
            r = E.evaluate(g, w)
            memo[k] = (r["fit"], r["n"], g)
        return memo[k][0]

    hist, rst = cma_bipop_x0(fn, N_CONT, DISC_SIZES, budget=budget, seed=seed, x0=x0_feasible())
    bi = int(np.argmax([h[2] for h in hist]))
    best = decode(hist[bi][1])
    fit, n, _ = memo[gkey(best)]
    curve = np.maximum.accumulate([h[2] for h in hist])
    out = {"best": best, "fit": fit, "n": n, "restarts": rst, "uniq": len(memo), "sec": time.time() - tt,
           "n_eval": len(hist), "curve": {m: float(curve[m - 1]) for m in (100, 1000, 3000, 10000, len(curve)) if m <= len(curve)}}
    out["hashes"] = np.array([ghash(v[2]) for v in memo.values()], np.int64)
    if top:
        vals = sorted(memo.values(), key=lambda v: -v[0])[:top]
        out["top"] = [(v[0], v[2]) for v in vals]
    return out


def ga_task(t: dict) -> dict:
    E = _W["E"]
    E.set_rep(t.get("rep"))
    tr, _ = _W["wins"][t["fold"]]
    r = ga_run_one(E, tr, t["seed"], t["budget"], top=t.get("top", 0))
    r.update(fold=t["fold"], seed=t["seed"], rep=t.get("rep"))
    return r


def procedure(E: Engine, wins: "list[tuple[Win, Win]]", bests: "list[list[dict]]") -> dict:
    """M2: 폴드마다 시드 최적을 새 난수(:data:`REEVAL_SEED`)로 학습창 재평가 → 1등 = 절차 출력."""
    outs, re_fits = [], []
    for (tr, _), bk in zip(wins, bests, strict=True):
        f = [E.evaluate(b, tr, seed=REEVAL_SEED)["fit"] for b in bk]
        outs.append(bk[int(np.argmax(f))])
        re_fits.append(f)
    return {"cfgs": outs, "re_fits": re_fits}


def null_b_task(t: dict) -> dict:
    """(2b) 모양 섞은 널 한 벌 = 섞은 사건 표 위에서 절차 전체(폴드 × 시드 × 예산 → 재평가 1등 → 이어 붙인 OOS) + 모양 끔 짝."""
    E = _W["E"]
    E.set_rep(t["rep"])
    wins = _W["wins"]
    tt = time.time()
    bests, rsts = [], []
    for k, (tr, _) in enumerate(wins):
        bk = []
        for s in t["seeds"]:
            r = ga_run_one(E, tr, int(1000 * (t["rep"] + 1) + s), t["budget"])
            bk.append(r["best"])
            rsts.append(len(r["restarts"]))
        bests.append(bk)
    pr = procedure(E, wins, bests)
    tests = [te for _, te in wins]
    st = E.stitched(pr["cfgs"], tests)
    so = E.stitched([shape_off(c) for c in pr["cfgs"]], tests)
    cu = [c["sup_c"] for c in pr["cfgs"] if c["sup_hw"] is not None]
    cl = [c["slo_c"] for c in pr["cfgs"] if c["slo_hw"] is not None]
    E.set_rep(None)
    return {"rep": t["rep"], "ann": st["ann"], "calm_ann": st["calm_ann"], "n_oos": st["n"], "shape_off_ann": so["ann"],
            "shape_gain": st["ann"] - so["ann"], "sup_c_sd": float(np.std(cu)) if len(cu) > 1 else np.nan,
            "slo_c_sd": float(np.std(cl)) if len(cl) > 1 else np.nan,
            "shapes": [config_shape(c) for c in pr["cfgs"]], "restarts_mean": float(np.mean(rsts)), "sec": time.time() - tt}


def random_x(rng: np.random.Generator, n: int) -> np.ndarray:
    """같은 메뉴 균등 추출(연속·로짓 U[0,1], 이산 단계 균등)."""
    X = rng.random((n, DIM))
    for q, k in enumerate(DISC_SIZES):
        X[:, N_CONT + q] = rng.integers(0, k, size=n)
    return X


def pool_task(t: dict) -> dict:
    """풀 방식(리뷰 C1): 폴드 학습창에서 균등 추출 config 를 한 번씩 평가 → (적합도, 진입 수, 실현 가능)."""
    E = _W["E"]
    E.set_rep(None)
    tr, _ = _W["wins"][t["fold"]]
    X = random_x(np.random.default_rng([POOL_SEED, t["fold"], t["chunk"]]), t["n"])
    fit, pen, n, feas = np.empty(t["n"]), np.empty(t["n"]), np.empty(t["n"], np.int64), np.empty(t["n"], bool)
    for i in range(t["n"]):
        r = E.evaluate(decode(X[i]), tr, counts_only=t.get("counts_only", False))
        fit[i] = r.get("fit", np.nan)
        pen[i] = r["pen"]
        n[i] = r["n"]
        feas[i] = r["feasible"]
    return {"fold": t["fold"], "chunk": t["chunk"], "X": X, "fit": fit, "pen": pen, "n": n, "feas": feas}


def landscape_task(t: dict) -> list:
    """민감도 지형: 폴드 출력 config 에서 한 유전자씩 범위 전체 — 학습 적합도·시험 연도 연환산(기술)."""
    E = _W["E"]
    E.set_rep(None)
    tr, te = _W["wins"][t["fold"]]
    x0 = encode(t["base"])
    rows = []
    for gi in range(DIM):
        grid = np.linspace(0, 1, 6) if gi < N_CONT else np.arange(DISC_SIZES[gi - N_CONT])
        for u in grid:
            x = x0.copy()
            x[gi] = u
            g = decode(x)
            if gkey(g) == gkey(t["base"]) and not np.isclose(u, x0[gi]):
                continue
            r = E.evaluate(g, tr)
            st = E.stitched([g], [te])
            rows.append({"fold": t["fold"], "dim": gi, "u": float(u), "train_fit": r["fit"], "train_n": r["n"],
                         "oos_ann": st["ann"], "oos_n": st["n"], "is_base": bool(np.isclose(u, x0[gi]))})
    return rows


# ---------------------------------------------------------------------------
# 5. 캐시 만들기(DB — 한 번)
# ---------------------------------------------------------------------------

def build_cache(cache: str, t0: float, *, with_p: bool) -> dict:
    """DB → 가격·맥락·층·바스켓·비용 배열 + 사건 표 72개 → ``cache`` (.npy). 분봉 p 는 ``with_p`` 일 때만(전체 모드)."""
    import vcp_retest_run as R  # noqa: PLC0415
    from sepa_momentum import MOM_N_Q, day_market_median_half, day_market_quantiles, momentum_12_1  # noqa: PLC0415

    from research.signals import vcp as V  # noqa: PLC0415
    from swing_it.features.technical import moving_average  # noqa: PLC0415

    os.makedirs(cache, exist_ok=True)
    ctx = R.Ctx(t0)
    A, days, codes = ctx.A, ctx.days, ctx.codes
    p = ctx.p
    close = p.close.reindex(index=days, columns=codes)
    hi_, lo_ = p.high.reindex(index=days, columns=codes), p.low.reindex(index=days, columns=codes)
    m20 = moving_average(close, 20)
    m150, m200 = moving_average(close, 150).to_numpy(float), moving_average(close, 200).to_numpy(float)
    prev = close.shift(1).to_numpy(float)
    hv_, lv_ = hi_.to_numpy(float), lo_.to_numpy(float)
    with np.errstate(invalid="ignore"):
        tr_ = np.fmax(np.fmax(hv_ - lv_, np.abs(hv_ - prev)), np.abs(lv_ - prev))
    atr = pd.DataFrame(np.where(np.isfinite(hv_ - lv_), tr_, np.nan)).rolling(20, min_periods=20).mean()
    uni = ctx.universe(ADV_FLOOR)
    with np.errstate(invalid="ignore"):
        align = (ctx.ma50 > m150) & (m150 > m200)
    base = uni.to_numpy(bool) & align & ~ctx.split["recent"]
    mom = momentum_12_1(p.close)
    q = day_market_quantiles(mom, uni, p.market, n_q=MOM_N_Q).reindex(index=days, columns=codes).to_numpy(float)
    a2 = day_market_median_half(ctx.D.adv20, uni, p.market).reindex(index=days, columns=codes).to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(a2) & uni.to_numpy(bool)
    strat = np.where(ok, ctx.mkt[None, :] * 20 + np.nan_to_num(q).astype(np.int64) * 2 + np.nan_to_num(a2).astype(np.int64), -1)
    del mom
    _log(f"맥락·층 패널 — 기본 맥락 종목-일 {int(base.sum()):,}", t0)
    T = lambda x: np.ascontiguousarray(np.asarray(x).T)  # noqa: E731
    month_of = ((days.year - days[0].year) * 12 + days.month - 1).to_numpy().astype(np.int64)
    year_of = (days.year - days[0].year).to_numpy().astype(np.int64)
    ctxd = {"base": T(base), "above50": T(np.asarray(ctx.p.close.reindex(index=days, columns=codes).to_numpy(float) > ctx.ma50)),
            "tt": T(ctx.tt), "rs": T(ctx.rs), "breadth": T(ctx.br), "adv20": T(ctx.adv), "strat": T(strat),
            "market": ctx.mkt.astype(np.int64), "month": month_of}
    d_lo = int(days.searchsorted(pd.Timestamp(EVENT_LO)))
    d_hi = int(days.searchsorted(pd.Timestamp(EVENT_HI), side="right") - 1)
    tb = PE.build_tables(A.H, A.L, A.C, ctx.VOL, A.O, A.TR, ctxd, d_lo=d_lo, d_hi=d_hi, log=lambda m: _log(m, t0))
    F, I_ = tb["F"], tb["I"]
    ci, s = I_[:, PE.II["ci"]], I_[:, PE.II["d"]] - 1
    Ee = V.e_accel_asof(ctx.est, codes[ci], days[s])
    F[:, PE.FI["e_yoy"]] = Ee["e_yoy"].to_numpy(float)
    F[:, PE.FI["e_yoy_prev"]] = Ee["e_yoy_prev"].to_numpy(float)
    end_row = int(days.searchsorted(pd.Timestamp(END_DAY), side="right") - 1)
    LIM = np.minimum(A.CUT, end_row + 1).astype(np.int64)
    strat_cs = T(strat).astype(np.int64)
    bs, bn = PF.basket_tables(A.O, A.C, strat_cs, 40)
    arrs = {"O": A.O, "H": A.H, "L": A.L, "C": A.C, "MA20": T(m20.to_numpy(float)), "MA50": A.MA, "ATR": T(atr.to_numpy(float)),
            "TR": A.TR, "LIM": LIM, "strat_cs": strat_cs, "bs": bs, "bn": bn, "cost_tab": R.cost_table(days),
            "days_ns": days.values.astype("datetime64[ns]").astype(np.int64), "month_of": month_of, "year_of": year_of,
            "F": F, "I": I_, "off": tb["off"], "codes": np.asarray(codes.astype(str), dtype="<U12")}
    meta = {"version": CACHE_VERSION, "built": date.today().isoformat(), "adv_floor": ADV_FLOOR, "event": [EVENT_LO, EVENT_HI],
            "end_day": END_DAY, "n_events": int(len(F)), "n_scan_rows": tb["n_scan_rows"], "n_cut_stocks": ctx.n_cut_stocks,
            "funnel": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in tb["funnel"].items()}, "base_rows": int(base.sum())}
    if with_p:
        pinfo = R._estimate_p(ctx.con, A, uni, days, None, band_lo=0.90, min_age=0, params={"ref_order": True})
        meta.update(p=pinfo["p"], p_ci=list(pinfo.get("ci", (np.nan, np.nan))), p_n=pinfo["n_classified"],
                    p_counts=pinfo.get("counts", {}), p_window=pinfo.get("window"))
    write_cache(cache, arrs, meta)
    return meta


def write_cache(cache: str, arrs: dict, meta: dict) -> None:
    """배열(:data:`ARR_NAMES`)·메타를 ``cache`` 에 .npy·json 으로(작업자가 memmap 으로 읽는다)."""
    os.makedirs(cache, exist_ok=True)
    for k in ARR_NAMES:
        np.save(os.path.join(cache, f"{k}.npy"), np.ascontiguousarray(arrs[k]), allow_pickle=False)
    with open(os.path.join(cache, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)


def ensure_cache(cache: str, t0: float, *, with_p: bool) -> None:
    mp_ = os.path.join(cache, "meta.json")
    meta = json.load(open(mp_, encoding="utf-8")) if os.path.exists(mp_) else {}
    if meta.get("version") == CACHE_VERSION and all(os.path.exists(os.path.join(cache, f"{k}.npy")) for k in ARR_NAMES):
        if with_p and "p" not in meta:
            import vcp_retest_run as R  # noqa: PLC0415
            ctx = R.Ctx(t0)
            uni = ctx.universe(ADV_FLOOR)
            pinfo = R._estimate_p(ctx.con, ctx.A, uni, ctx.days, None, band_lo=0.90, min_age=0, params={"ref_order": True})
            meta.update(p=pinfo["p"], p_ci=list(pinfo.get("ci", (np.nan, np.nan))), p_n=pinfo["n_classified"],
                        p_counts=pinfo.get("counts", {}), p_window=pinfo.get("window"))
            with open(mp_, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)
        _log(f"캐시 재사용 {cache} (사건 {meta['n_events']:,})", t0)
        return
    build_cache(cache, t0, with_p=with_p)


# ---------------------------------------------------------------------------
# 6. 원장
# ---------------------------------------------------------------------------

def _record_design(out_dir: str) -> None:
    """MINOR 10·11 — 리뷰가 센 설계 진단(건수만)과 '차트 열람 → 정의 수정 → 같은 데이터 재시험' 사슬 한 줄."""
    from swing_it.diagnostics.trials import record_trial  # noqa: PLC0415
    for v_ in ("θ 3·5·8% × 천장 창 120·260 × 기준선(추세선·수평) 문법 사건 수", "무작위 설정 3,000 실현 가능성(사전등록 원 범위)"):
        record_trial(LABEL, {"prereg": PREREG_DOC, "design_diagnostic": v_, "evaluated": "건수만, 성과 미계산",
                             "source": "REVIEW_PATTERN_GA.md 1단계 0. 실현 가능성 실측"}, logs_dir=out_dir,
                     note=f"패턴 GA 설계 진단(건수만): {v_}")
    record_trial(LABEL, {"prereg": PREREG_DOC, "chain": "차트 열람 → 정의 수정 → 같은 데이터 재시험: VCP → 쐐기 → 쐐기 D → 패턴 GA",
                         "evaluated": "기록만"}, logs_dir=out_dir,
                 note="MINOR 10: 이 계열 차트 열람 → 정의 수정 → 같은 KRX 데이터 재시험 사슬(VCP → 쐐기 → 쐐기 D → 패턴 GA)")


def _proc_config(budget: int, seeds: tuple, fac_sha: str, pmix: float) -> dict:
    return {"prereg": PREREG_DOC, "universe": f"위생 ∧ ADV20 ≥ {ADV_FLOOR:,.0f} 백만원 ∧ d−1 MA50>MA150>MA200 ∧ 분할 60일 제외",
            "event": [EVENT_LO, EVENT_HI], "end_mark": END_DAY, "tables": f"θ {PE.THETAS} × 창 {PE.CEIL_WINS} × 기준선 {PE.BASES}",
            "hygiene": PE.HYGIENE, "genes": {"cont": [list(c[:3]) for c in CONT], "nominal": [[n, list(map(str, v))] for n, v in NOM],
                                              "disc": [[n, list(map(str, v))] for n, v in DISC]},
            "fitness": f"10 슬롯 포트폴리오 − 종목별 모멘텀 정합 바스켓, 월합 3개월 블록 부트스트랩 {FIT_BOOT} 2.5% 하단 − "
                       f"λ{LAMBDA}·(연 {FLOOR_PER_YEAR:g} 하한·연도별 {FLOOR_YEAR:g}) 연속 벌점, 비용 {COST_MULT:g}배, p 혼합 {pmix:.3f}",
            "optimizer": f"CMAwM+BIPOP(cmaes 0.13.1) 시드 {len(seeds)} × {budget}, 첫 평균 = 실현 가능점, 재평가 seed {REEVAL_SEED}",
            "folds": "swing_it.validation.walkforward.FOLDS(6), 학습창 끝 시가평가", "factors_sha": fac_sha}


# ---------------------------------------------------------------------------
# 7. 실현 가능성(건수만) — FEASIBILITY.md
# ---------------------------------------------------------------------------

def feasibility(E: Engine, wins, folds, args, t0: float) -> dict:
    """사건 표 건수·폴드×연도 건수·맥락 거르기 공백·무작위 설정 실현 가능 비율·착상 영역 학습창 연 체결 수·분봉 커버리지.
    **수익 숫자는 만들지 않는다**(포트폴리오는 슬롯·보유 거르기의 진입 수를 세는 데만 — 수익 열은 버린다)."""
    F, I_, off = np.asarray(E.F), np.asarray(E.I), np.asarray(E.off)
    days = E.days
    yr = days[I_[:, PE.II["d"]]].year.to_numpy()
    tid = I_[:, PE.II["table"]]
    base = tid % len(PE.BASES)
    res: dict = {"tables": [], "years": sorted(set(yr.tolist()))}
    for t in range(PE.N_TABLES):
        a, b = PE.table_slice(off, t)
        th, cw, bn = PE.table_key(t)
        k = f"{th}|{cw}|{bn}"
        fu = E.meta["funnel"].get(k, {})
        cnt = pd.Series(yr[a:b]).value_counts().to_dict()
        res["tables"].append({"theta": th, "cw": cw, "base": bn, "n": b - a, "h0in": int((F[a:b, PE.FI["h0in"]] == 1).sum()),
                              "years": {int(y): int(v) for y, v in cnt.items()}, "funnel": fu})
    # 폴드 × 연도(학습·시험) — 기준 표(θ 5% · 창 260) 기준선별 + 맥락 거르기(breadth > 0.5, d−1 TT, H0 일 TT)
    ctx_rows = []
    for bi, bn in enumerate(PE.BASES):
        a, b = PE.table_slice(off, PE.table_id(PE.THETAS.index(0.05), PE.CEIL_WINS.index(260), bi))
        sub = F[a:b]
        y = yr[a:b]
        for nm, m in (("전부", np.ones(b - a, bool)), ("breadth ≥ 0.5", sub[:, PE.FI["breadth"]] >= 0.5),
                      ("d−1 추세 템플릿", sub[:, PE.FI["tt_s"]] > 0.5), ("H0 일 추세 템플릿", sub[:, PE.FI["tt_h0"]] > 0.5),
                      ("RS ≥ 70", sub[:, PE.FI["rs"]] >= 70)):
            ctx_rows.append({"base": bn, "filter": nm, **{int(yy): int(((y == yy) & m).sum()) for yy in res["years"]}})
    res["ctx_years"] = ctx_rows
    # 착상 영역(정정 1 C2 + 정정 2 + 정정 3) — 폴드 학습창 연 체결(통과 사건 · 슬롯 진입 수), 가장 느슨한 나머지 유전자
    rw, rwd = event_region(F, base)
    loose = {"rs_min": None, "breadth_min": None, "sup_c": None, "sup_hw": None, "slo_c": None, "slo_hw": None, "rho_max": None,
             "span_c": None, "span_hw": None, "ceil_min": None, "depth_max": None, "dry_max": None, "runup_min": None,
             "runup_win": None, "rest_gap_max": None, "gap_max": 0.05, "stop_cap": 0.12, "fb_tol": 0.0, "trail_k": None,
             "stage": 0, "trail": "ma50", "pyramid": "off", "touch_lo": 2, "e_min": None, "e_accel": None, "stop_anchor": "E",
             "fb_days": 2, "be_r": math.inf, "take_r": 2.0, "take_frac": 0.5, "tighten": None, "ceil_kind": "single"}
    reg_rows = []
    for rname, mask, entry, h0in in (("W", rw, "trend", 0), ("W", rw, "trend", 1), ("WD", rwd, "bottom", 1)):
        for th in PE.THETAS:
            for cw in PE.CEIL_WINS:
                g = canonical({**loose, "theta": th, "ceil_win": cw, "entry": entry, "h0_in_pattern": h0in})
                row = {"region": rname, "h0in": h0in, "theta": th, "cw": cw}
                for k, (tr, _) in enumerate(wins):
                    r = E.evaluate(g, tr, counts_only=True, allow=mask)
                    yrs = sum(E.penalty({"n": 0, "ny": np.zeros(E.n_years, np.int64)}, tr)[1]["years"].values())
                    row[f"f{k}_pass_yr"] = r["n_pass"] / yrs
                    row[f"f{k}_taken_yr"] = r["n"] / yrs
                    row[f"f{k}_feasible"] = r["feasible"]
                reg_rows.append(row)
    res["region"] = reg_rows
    res["region_verdict"] = {}
    for rname in ("W", "WD"):
        rows = [r for r in reg_rows if r["region"] == rname]
        res["region_verdict"][rname] = [bool(any(r[f"f{k}_feasible"] for r in rows)) for k in range(len(wins))]
    # 무작위 설정 실현 가능 비율(같은 메뉴 균등 — 건수만)
    rng = np.random.default_rng([POOL_SEED, 999])
    X = random_x(rng, args.n_feas)
    t1 = time.time()
    rows = []
    for i in range(len(X)):
        g = decode(X[i])
        row = {"entry": g["entry"]}
        for k, (tr, _) in enumerate(wins):
            r = E.evaluate(g, tr, counts_only=True)
            row[f"f{k}_n"], row[f"f{k}_pass"], row[f"f{k}_feas"] = r["n"], r["n_pass"], r["feasible"]
        rows.append(row)
    sec = (time.time() - t1) / max(1, len(X) * len(wins))
    res["random"] = pd.DataFrame(rows)
    res["sec_per_eval_counts"] = sec
    # 평가 1회 시간(적합도 전부 — 부트스트랩 포함, 숫자는 버린다)
    t1 = time.time()
    m = min(300, len(X))
    for i in range(m):
        E.evaluate(decode(X[i]), wins[-1][0])
    res["sec_per_eval_fit"] = (time.time() - t1) / max(1, m)
    # 분봉 커버리지(10억~50억 · ≥ 50억, 2025-08+ 사건 (종목, d) 고유)
    res["minute"] = minute_coverage(E, args) if not args.no_db else {}
    return res


def minute_coverage(E: Engine, args) -> dict:
    """M7: 분봉 구간 사건(전 표 합집합 (종목, 주문일))의 ADV20 구간별 분봉 유무 건수."""
    from prop_swing_common import load_env_db  # noqa: PLC0415

    from swing_it.storage import connect, db_default  # noqa: PLC0415
    F, I_ = np.asarray(E.F), np.asarray(E.I)
    d = I_[:, PE.II["d"]]
    lo = E.row(MINUTE_LO)
    m = d >= lo
    u = pd.DataFrame({"ci": I_[m, PE.II["ci"]], "d": d[m], "adv": F[m, PE.FI["adv20"]]}).drop_duplicates(["ci", "d"]).reset_index(drop=True)
    u["band"] = np.where(u["adv"] >= 5_000, "≥50억", "10억~50억")
    load_env_db()
    con = connect(db_default())
    has = np.zeros(len(u), bool)
    codes = np.asarray(E.codes)
    q = "SELECT DISTINCT code FROM minute_bars WHERE ts >= %(lo)s AND ts < %(hi)s AND code = ANY(%(c)s)"
    for dd, gidx in u.groupby("d").groups.items():
        day = E.days[int(dd)]
        cs = [str(codes[c]) for c in u.loc[gidx, "ci"]]
        got = set(pd.read_sql_query(q, con, params={"lo": day.strftime("%Y-%m-%d"),
                                                     "hi": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "c": cs})["code"].astype(str))
        has[np.asarray(gidx)] = [c in got for c in cs]
    u["has"] = has
    return {b: {"n": int(len(g)), "minute": int(g["has"].sum())} for b, g in u.groupby("band")}


def write_feasibility(path: str, res: dict, E: Engine, wins, args, elapsed: float, est: dict) -> None:
    yrs = res["years"]
    L = [f"# 패턴 GA — 실현 가능성(건수만, 성과 미계산) {date.today().isoformat()}", "",
         f"`pattern_ga_run.py --feasibility-only` ({PREREG_DOC}). 사건(주문일) {EVENT_LO}~{EVENT_HI}, 유니버스 ADV20 ≥ 10억. "
         "포트폴리오는 슬롯·보유 거르기의 **진입 수를 세는 데만** 썼고, 수익·청산 사유·손절 여부는 어디에도 적지 않는다. "
         f"실행 {elapsed / 60:.1f}분. 사건 표 {E.meta['n_events']:,}행(훑은 종목-일 {E.meta['n_scan_rows']:,}, 기본 맥락 종목-일 "
         f"{E.meta['base_rows']:,}).", "",
         "## 1. 사건 표 건수 (θ × 천장 창 × 기준선) — 첫 관통 → 위생 → 맥락", "",
         "| θ | 창 | 기준선 | 첫 관통 | 간격<3 | 길이<15 | 무결성 상/하 | 종가 여유 | 바닥 >5일 | 맥락(d−1) | **사건** | H0 첫 점 | " + " | ".join(str(y) for y in yrs) + " |",
         "|" + "---|" * (12 + len(yrs))]
    for t in res["tables"]:
        fu = t["funnel"]
        L.append(f"| {t['theta']:.0%} | {t['cw']} | {t['base']} | {fu.get('first_cross', 0):,} | {fu.get('spacing<3', 0):,} | "
                 f"{fu.get('span<15', 0):,} | {fu.get('integrity_upper', 0):,}/{fu.get('integrity_lower', 0):,} | "
                 f"{fu.get('close>order*0.995', 0):,} | {fu.get('bottom_age>5', 0):,} | {fu.get('context(d-1)', 0):,} | "
                 f"**{t['n']:,}** | {t['h0in']:,} | " + " | ".join(f"{t['years'].get(y, 0):,}" for y in yrs) + " |")
    L += ["", "## 2. 맥락 거르기별 연도 건수(θ 5% · 창 260) — 폴드 공백 점검(쐐기 리뷰 F)", "",
          "| 기준선 | 거르기 | " + " | ".join(str(y) for y in yrs) + " |", "|" + "---|" * (2 + len(yrs))]
    for r in res["ctx_years"]:
        L.append(f"| {r['base']} | {r['filter']} | " + " | ".join(f"{r.get(y, 0):,}" for y in yrs) + " |")
    L += ["", "## 3. 사용자 착상 영역 — 폴드 학습창 연 체결 수(정정 1 C2 + 정정 2 + 정정 3, 쐐기 리뷰 F)", "",
          f"영역 W = 하락 쐐기(ŝ_up < 0 ∧ ŝ_lo < 0 ∧ ŝ_up < ŝ_lo) ∧ 상단 추세선 ∧ 60일 급등 ≥ {REGION_RUNUP:.0%} ∧ 쉼 ≤ {REGION_REST:.0f}일"
          "(fb 2 · H0 첫 점 0/1 둘 다), WD = 같은 모양·맥락 ∧ H0 = 첫 점 ∧ 바닥 반전. 나머지 유전자는 가장 느슨하게(단계 0, 거르기 끔, "
          "갭 5%, 손절 상한 12%, 라이브 청산 꼴). '통과' = 거르기·체결 조건 통과 사건(슬롯 전 상한), '진입' = 10 슬롯·보유 거르기 뒤. "
          f"**실현 가능** = 학습창 연 {FLOOR_PER_YEAR:g}건·연도별 {FLOOR_YEAR:g}건 하한(비율 보정) 통과.", "",
          "| 영역 | H0 첫 점 | θ | 창 | " + " | ".join(f"폴드{k + 1} 통과/진입 연" for k in range(len(wins))) + " |",
          "|" + "---|" * (4 + len(wins))]
    for r in res["region"]:
        L.append(f"| {r['region']} | {r['h0in']} | {r['theta']:.0%} | {r['cw']} | " + " | ".join(
            f"{r[f'f{k}_pass_yr']:.1f}/{r[f'f{k}_taken_yr']:.1f}{' ✓' if r[f'f{k}_feasible'] else ''}" for k in range(len(wins))) + " |")
    for rname, v in res["region_verdict"].items():
        bad = [k + 1 for k, ok in enumerate(v) if not ok]
        L.append("")
        L.append(f"- **영역 {rname}:** " + ("모든 폴드에서 실현 가능한 (θ, 창) 칸이 있다." if not bad else
                                          f"폴드 {bad} 는 어느 (θ, 창) 칸도 학습창 하한을 못 넘는다 → **그 폴드에서 이 영역은 이 데이터로 GA 판단 불가**"
                                          "(결과 전 표시 — GA 최적이 이 영역에 떨어질 수 없거나, 떨어져도 벌점 구역)."))
    rnd = res["random"]
    L += ["", f"## 4. 같은 메뉴 균등 무작위 설정 {len(rnd):,}개 — 실현 가능 비율(건수만)", "",
          "| 폴드(학습창) | 통과 사건 0 | 통과 사건 중앙/90% | 진입 중앙/90% | **실현 가능(벌점 0)** |", "|---|---|---|---|---|"]
    for k, (tr, _) in enumerate(wins):
        L.append(f"| {k + 1} ({tr.lo}~{tr.hi}) | {(rnd[f'f{k}_pass'] == 0).mean():.1%} | {rnd[f'f{k}_pass'].median():.0f}/"
                 f"{rnd[f'f{k}_pass'].quantile(0.9):.0f} | {rnd[f'f{k}_n'].median():.0f}/{rnd[f'f{k}_n'].quantile(0.9):.0f} | "
                 f"**{rnd[f'f{k}_feas'].mean():.1%}** |")
    by_e = rnd.groupby("entry")[[f"f{k}_feas" for k in range(len(wins))]].mean()
    L += ["", "기준선별 실현 가능 비율: " + " · ".join(f"{e} " + "/".join(f"{v:.0%}" for v in row) for e, row in by_e.iterrows())]
    mi = res.get("minute") or {}
    L += ["", f"## 5. 분봉 커버리지(M7 — {MINUTE_LO}+ 사건 (종목, 주문일) 고유, 전 표 합집합)", "",
          "| ADV20 구간 | 사건 | 분봉 있음 |", "|---|---|---|"]
    for b, v in mi.items():
        L.append(f"| {b} | {v['n']:,} | {v['minute']:,} ({v['minute'] / max(v['n'], 1):.1%}) |")
    L += ["", "분봉 없는 사건의 진입일 모호 처리: 분봉 p 는 분봉 있는 합성 사건으로 한 번 재고(전체 모드), 모든 모호 거래에 같은 p 혼합"
          "(보수 p=1 · 낙관 p=0 경계를 함께 보고).", "",
          "## 6. 시간 추정", "",
          f"- 평가 1회(건수만) {res['sec_per_eval_counts'] * 1e3:.2f} ms · 적합도 전부 {res['sec_per_eval_fit'] * 1e3:.2f} ms(단일 스레드, 폴드 6 학습창)",
          f"- 전체 실행: {est['lines']}", "",
          "## 공통", "",
          f"- NXT 미보정 종목 {E.meta.get('n_cut_stocks', 0):,}개 보유 경로 2026-09-11 절단, 전 종목 데이터 끝 {END_DAY} 시가평가.",
          "- 이 문서는 건수만 적는다. 판정은 사람이 한다.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def estimate(sec: float, workers: int, budget: int, seeds: int, n_null_b: int, pool_n: int) -> dict:
    ev_real = 6 * seeds * budget
    ev_null = n_null_b * ev_real
    ev_pool = 6 * pool_n
    ev_misc = 6 * DIM * 6 * 2 + 3 * (N_NULL_A + N_RS + N_EXIT_CTL) * 6 + 6 * seeds * (1 + TOP_PBO)
    tot = ev_real + ev_null + ev_pool + ev_misc
    h = tot * sec / workers / 3600.0
    return {"evals": tot, "hours": h, "lines": f"평가 ≈ {tot:,.0f}회(실제 {ev_real:,} · 모양 섞은 널 {ev_null:,} · 풀 {ev_pool:,} · 기타 "
                                               f"{ev_misc:,}) × {sec * 1e3:.2f} ms / 작업자 {workers} ≈ **{h:.1f}시간**(+ 캐시·분봉 p 적재·JIT)"}


# ---------------------------------------------------------------------------
# 8. 보고용 트레이드 표
# ---------------------------------------------------------------------------

def trade_frame(E: Engine, r: dict, cfgs: "list[dict]") -> pd.DataFrame:
    """포트폴리오 기록 → 트레이드 표(p 혼합 재료: ``ret``/``ret_stop0``·``net``/``net2``(1·2배)). 비율은 슬롯 배분 1 단위."""
    rec = r["rec"]
    if not len(rec):
        return pd.DataFrame()
    ev = rec[:, PF.Q_EV].astype(np.int64)
    F, I_ = np.asarray(E.F_orig), np.asarray(E.I)
    d, x = rec[:, PF.Q_D].astype(int), rec[:, PF.Q_X].astype(int)
    e, sp, wb = rec[:, PF.Q_E], rec[:, PF.Q_SP], rec[:, PF.Q_WB]
    added = rec[:, PF.Q_ADDED] > 0.5
    wa = np.where(added, 1.0 - wb, 0.0)
    c1 = rec[:, PF.Q_CX] * wb + rec[:, PF.Q_CADD] * wa
    ret0 = wb * (sp * 0.999 / e - 1.0)
    from research.signals import vcp as V  # noqa: PLC0415
    t = pd.DataFrame({"ev": ev, "code": np.asarray(E.codes)[I_[ev, PE.II["ci"]]], "ci": I_[ev, PE.II["ci"]], "d_row": d, "x_row": x,
                      "event_day": E.days[d - 1], "order_day": E.days[d], "entry_day": E.days[d], "exit_day": E.days[x],
                      "fold": rec[:, PF.Q_PAR].astype(int), "entry_px": e, "stop_px": sp, "d": (e - sp) / e,
                      "ret": rec[:, PF.Q_PNL], "ret_stop0": ret0, "amb": rec[:, PF.Q_AMB] > 0.5,
                      "exit_reason": np.asarray(V.REASONS, dtype=object)[rec[:, PF.Q_REASON].astype(int)],
                      "hold_days": x - (d - 1), "wb": wb, "added": added, "partial": rec[:, PF.Q_TP] >= 0,
                      "mfe_px": rec[:, PF.Q_MFE], "alloc": rec[:, PF.Q_ALLOC], "exit_open": rec[:, PF.Q_EXIT_OPEN] > 0.5,
                      "exit_px": rec[:, PF.Q_FIN], "partial_px": rec[:, PF.Q_PX1], "add_px": rec[:, PF.Q_PA]})
    t["net"] = t["ret"] - c1
    t["net2"] = t["ret"] - 2 * c1
    t["net_stop0"] = t["ret_stop0"] - rec[:, PF.Q_CD] * wb
    t["net2_stop0"] = t["ret_stop0"] - 2 * rec[:, PF.Q_CD] * wb
    t["mfe_R"] = (t["mfe_px"] - e) / (e - sp)
    t["real_R"] = t["net"] / (wb * t["d"])
    t["adv20"] = F[ev, PE.FI["adv20"]]
    t["s_up"], t["s_lo"] = F[ev, PE.FI["s_up"]], F[ev, PE.FI["s_lo"]]
    t["shape"] = [shape_name(a, b) for a, b in zip(t["s_up"], t["s_lo"], strict=True)]
    t["table"] = I_[ev, PE.II["table"]]
    t["base"] = [PE.BASES[k % len(PE.BASES)] for k in t["table"]]
    return t


def mfe_table(t: pd.DataFrame, p: float) -> dict:
    """M1-d: MFE 구간(<1, 1~2, 2~4, ≥4 R)별 실현 R 평균·되돌림, '+1R → 손실' 비율, 승자 포착률(기술) — 낙관 경로·p 혼합 순."""
    if t is None or t.empty:
        return {}
    mix = p * t["net_stop0"] + (1 - p) * t["net"]
    rr = mix / (t["wb"] * t["d"])
    b = pd.cut(t["mfe_R"], [-np.inf, 1, 2, 4, np.inf], right=False, labels=["<1R", "1~2R", "2~4R", "≥4R"])
    out = {"buckets": {str(k): {"n": int(len(g)), "real_R": float(rr[g.index].mean()),
                                "giveback_R": float((t.loc[g.index, "mfe_R"] - rr[g.index]).mean())}
                       for k, g in t.groupby(b, observed=True)},
           "plus1R_to_loss": float(((t["mfe_R"] >= 1) & (mix < 0)).mean())}
    win = mix > 0
    cap = (mix[win] / (t.loc[win, "wb"] * (t.loc[win, "mfe_px"] / t.loc[win, "entry_px"] - 1))).replace([np.inf, -np.inf], np.nan)
    out["winner_capture_median"] = float(cap.median()) if win.any() else np.nan
    return out


def pair_exits(E: Engine, t: pd.DataFrame, cfgs: "list[dict]", variant) -> dict:
    """같은 진입(OOS 트레이드)에 다른 청산을 붙인 짝 — ``variant(px, cfg) -> px``. 낙관 경로 총손익(배분 1 단위) 차이, 날짜군집 CI."""
    from krx_quant_core.stats.matched_null import cluster_bootstrap_ci  # noqa: PLC0415
    if t is None or t.empty:
        return {"n": 0}
    out = np.empty(PF.N_O)
    F = np.asarray(E.F_orig)
    a, b = [], []
    for r in t.itertuples(index=False):
        g = cfgs[int(r.fold)]
        px = exit_vec(g)
        fb = F[r.ev, PE.FI["level"]] * (1 - (g["fb_tol"] or 0.0)) if g["entry"] != "bottom" else -np.inf
        skip0 = bool(F[r.ev, PE.FI["open_d"]] <= F[r.ev, PE.FI["level"]] and F[r.ev, PE.FI["close_d"]] > r.stop_px)
        args = (E.O, E.H, E.L, E.C, E.MA20, E.MA50, E.ATR, E.TR, E.LIM, int(r.ci), int(r.d_row), float(r.entry_px),
                float(r.stop_px), float(fb), float(F[r.ev, PE.FI["h0"]]), skip0)
        PF.exit_path(*args, px, out)
        a.append(out[PF.O_PNL])
        PF.exit_path(*args, variant(px.copy(), g), out)
        b.append(out[PF.O_PNL])
    diff = np.array(b) - np.array(a)
    lo, hi, _ = cluster_bootstrap_ci(diff, t["order_day"].dt.strftime("%Y%m%d").to_numpy(), n_boot=CI_BOOT, seed=0)
    return {"n": int(len(diff)), "a": float(np.mean(a)), "b": float(np.mean(b)), "diff": float(diff.mean()), "lo": lo, "hi": hi}


def live_px(px: np.ndarray, g: dict) -> np.ndarray:
    p = PF.LIVE_X.copy()
    p[PF.X_FB_DAYS] = 0 if g["entry"] == "bottom" else 1
    return p


def pyr_toggle(px: np.ndarray, g: dict) -> np.ndarray:
    px[PF.X_PYR] = PF.PYR_OFF if px[PF.X_PYR] != PF.PYR_OFF else PF.PYR_1R
    return px


# ---------------------------------------------------------------------------
# 9. 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:  # noqa: C901 — 단계가 많은 리포터
    from swing_it.validation.walkforward import FOLDS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="사건 표·배열 .npy 디렉터리(없으면 DB 에서 만든다)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--feasibility-only", action="store_true", help="건수만(수익 미계산) → pattern_ga/FEASIBILITY.md")
    ap.add_argument("--debug", action="store_true", help="축소 실행(폴드 2 · 시드 2 · 예산 300 · 널 축소) — --out 을 탐색 디렉터리 밖으로")
    ap.add_argument("--n-feas", type=int, default=N_FEAS)
    ap.add_argument("--no-db", action="store_true", help="실현 가능 모드에서 분봉 커버리지 조회 생략(캐시만)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if args.debug and not debug:
        ap.error("--debug 는 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    sub_dir = os.path.join(out_dir, SUB)
    os.makedirs(sub_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()
    fold_ids = list(range(len(FOLDS)))
    seeds, budget, n_a, n_b, n_rs, n_ec, pool_n = SEEDS, BUDGET, N_NULL_A, N_NULL_B, N_RS, N_EXIT_CTL, POOL_N
    if args.debug:
        fold_ids, seeds, budget, n_a, n_b, n_rs, n_ec, pool_n = [4, 5], (0, 1), 300, 20, 2, 20, 20, 400
        args.n_feas = min(args.n_feas, 60)
    folds = [FOLDS[k] for k in fold_ids]
    folds_spec = [tuple(f) for f in folds]
    ensure_cache(args.cache, t0, with_p=not args.feasibility_only)
    _record_design(out_dir)
    E = Engine(args.cache)
    wins = fold_windows(E, folds)
    _log(f"엔진 — 사건 {len(E.F):,} · 표 {PE.N_TABLES} · 폴드 {len(wins)}", t0)

    if args.feasibility_only:
        E.evaluate(decode(x0_feasible()), wins[-1][0])              # JIT 예열(시간에서 뺀다)
        res = feasibility(E, wins, folds, args, t0)
        est = estimate(res["sec_per_eval_fit"], args.workers, BUDGET, len(SEEDS), N_NULL_B, POOL_N)
        path = os.path.join(sub_dir, FEAS_MD)
        write_feasibility(path, res, E, wins, args, time.time() - t0, est)
        res["random"].to_csv(os.path.join(sub_dir, "feasibility_random.csv"), index=False)
        pd.DataFrame(res["region"]).to_csv(os.path.join(sub_dir, "feasibility_region.csv"), index=False)
        from swing_it.diagnostics.trials import record_trial  # noqa: PLC0415
        record_trial(LABEL, {"prereg": PREREG_DOC, "feasibility": "사건 표 건수·무작위 설정 실현 가능 비율·착상 영역 연 체결·분봉 커버리지",
                             "n_feas": int(args.n_feas), "evaluated": "건수만, 성과 미계산"}, logs_dir=out_dir,
                     note="패턴 GA 실현 가능성(건수만, 결과 전)")
        rnd = res["random"]
        print(f"사건 표 {E.meta['n_events']:,}행 · 무작위 실현 가능(폴드별) " + " · ".join(
            f"{rnd[f'f{k}_feas'].mean():.1%}" for k in range(len(wins))) + f" · 착상 영역 {res['region_verdict']}")
        print(est["lines"])
        print(f"→ {path}")
        return 0

    return run_full(E, wins, folds_spec, args, out_dir, sub_dir, seeds, budget, n_a, n_b, n_rs, n_ec, pool_n, debug, t0)


def run_full(E, wins, folds_spec, args, out_dir, sub_dir, seeds, budget, n_a, n_b, n_rs, n_ec, pool_n, debug, t0) -> int:
    """전체(또는 --debug) 실행 — 원장 → 실제 GA → 풀 → 널 → 후처리 → 결과 문서."""
    import vcp_retest_run as R  # noqa: PLC0415

    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample  # noqa: PLC0415
    from swing_it.diagnostics.pbo import pbo_cscv  # noqa: PLC0415
    from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: PLC0415

    timings: dict = {}
    fac_sha = R._sha(R.FACTORS_PKL)
    pmix = E.pmix
    base_cfg = _proc_config(budget, seeds, fac_sha, pmix)
    record_trial(LABEL, {**base_cfg, "role": "절차(주) — 폴드별 시드 최적 재평가 1등, 이어 붙인 OOS"}, logs_dir=out_dir,
                 note="패턴 GA 절차(결과 전)")
    for nm, d_ in (("2a", f"실현 가능 조건부 같은 메뉴 무최적화 무작위 설정 {N_NULL_A}벌(풀 {POOL_N}/폴드, seed {NULL_A_SEED})"),
                   ("2b", f"모양 서술자 같은 달 섞음 × 절차 전체 {N_NULL_B}벌(seed {NULLB_SEED})"),
                   ("rs", f"무작위 탐색 best-of-{RS_BEST_OF} {N_RS}벌(기술)"), ("exit_ctl", f"청산 전용 대조 {N_EXIT_CTL}벌(기술)")):
        record_trial(LABEL, {**base_cfg, "null": nm, "desc": d_}, logs_dir=out_dir, note=f"패턴 GA 널 {nm}(결과 전)")
    for cell in ("비용 1배", "보수 p=1", "낙관 p=0", "미끄러짐 +0.3%", "미끄러짐 +0.5%", "같은 날 무작위 순서", "상위 5거래 제외",
                 "상위 5% 제외", "2021 제외", "모양 끔 짝", "중앙값 config(조건부)", "기준 청산 짝", "불타기 짝"):
        record_trial(LABEL, {**base_cfg, "sensitivity": cell}, logs_dir=out_dir, note=f"패턴 GA 민감도·기술: {cell}")
    led_dir = out_dir
    n_led0 = count_trials(LABEL, logs_dir=led_dir)

    # --- 실제 GA + 풀 + 널(2b) ---
    tasks = [{"fold": k, "seed": s, "budget": budget, "top": TOP_PBO} for k in range(len(wins)) for s in seeds]
    per_chunk = 2_000 if not args.debug else 200
    ptasks = [{"fold": k, "chunk": c, "n": min(per_chunk, pool_n - c * per_chunk)} for k in range(len(wins))
              for c in range(math.ceil(pool_n / per_chunk))]
    ntasks = [{"rep": r, "seeds": seeds, "budget": budget} for r in range(n_b)]
    res, pools, nulls_b = {}, {k: [] for k in range(len(wins))}, []
    hashes: set = set()
    ctx = mp.get_context("spawn")
    t1 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=worker_init,
                                initargs=(args.cache, folds_spec)) as ex:
        futs = {ex.submit(ga_task, t): t for t in tasks}
        pf = [ex.submit(pool_task, t) for t in ptasks]
        nf = [ex.submit(null_b_task, t) for t in ntasks]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            res[(r["fold"], r["seed"])] = r
            hashes.update(r["hashes"].tolist())
            record_trial(LABEL, {"prereg": PREREG_DOC, "run": f"fold{r['fold']}/seed{r['seed']}", "budget": budget,
                                 "n_unique_configs_evaluated": r["uniq"], "best": {k: str(v) for k, v in r["best"].items()},
                                 "restarts": len(r["restarts"])}, logs_dir=led_dir,
                         note="패턴 GA 런 요약(고유 config 수는 n_unique_configs_evaluated)")
            if i % 10 == 0 or i == len(futs):
                _log(f"실제 런 {i}/{len(futs)} · 재시작 {len(r['restarts'])} · {r['sec']:.0f}s", t0)
        timings["ga"] = time.time() - t1
        bests = [[res[(k, s)]["best"] for s in seeds] for k in range(len(wins))]
        pr = procedure(E, wins, bests)
        lf = [ex.submit(landscape_task, {"fold": k, "base": c}) for k, c in enumerate(pr["cfgs"])]
        for f in cf.as_completed(pf):
            r = f.result()
            pools[r["fold"]].append(r)
        timings["pool"] = time.time() - t1
        _log("풀 평가 끝", t0)
        for i, f in enumerate(cf.as_completed(nf), 1):
            nulls_b.append(f.result())
            if i % 5 == 0 or i == len(nf):
                _log(f"모양 섞은 널 {i}/{len(nf)}", t0)
        timings["null_b"] = time.time() - t1
        land = [row for f in lf for row in f.result()]
        timings["landscape"] = time.time() - t1

    # --- 후처리(부모) ---
    t1 = time.time()
    tests = [te for _, te in wins]
    out_cfgs = pr["cfgs"]
    st = E.stitched(out_cfgs, tests)
    tr_df = trade_frame(E, st, out_cfgs)
    v = gf.block_boot(np.ascontiguousarray(st["monthly"]), 3, CI_BOOT, 0)
    ci_lo, ci_hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    # (2a) · 무작위 탐색 · 청산 전용
    P = {k: {"X": np.vstack([c["X"] for c in pools[k]]), "fit": np.concatenate([c["fit"] for c in pools[k]]),
             "pen": np.concatenate([c["pen"] for c in pools[k]]), "feas": np.concatenate([c["feas"] for c in pools[k]])}
         for k in pools}
    pool_feas = {}
    for k in P:                                      # 실현 가능 0 인 폴드는 벌점 최소 1% 로 대신(결과 문서에 수를 적는다)
        fi = np.flatnonzero(P[k]["feas"])
        pool_feas[k] = int(len(fi))
        P[k]["pick"] = fi if len(fi) else np.flatnonzero(P[k]["pen"] <= np.quantile(P[k]["pen"], 0.01))
    rng_a = np.random.default_rng(NULL_A_SEED)
    null_a = []
    for _ in range(n_a):
        cfgs = []
        for k in range(len(wins)):
            cfgs.append(decode(P[k]["X"][rng_a.choice(P[k]["pick"])]))
        s_ = E.stitched(cfgs, tests)
        null_a.append({"ann": s_["ann"], "calm_ann": s_["calm_ann"], "n": s_["n"]})
    null_a = pd.DataFrame(null_a)
    rng_r = np.random.default_rng(RS_SEED)
    rs_rows = []
    for _ in range(n_rs):
        cfgs = []
        for k in range(len(wins)):
            idx = rng_r.choice(len(P[k]["fit"]), size=min(RS_BEST_OF, len(P[k]["fit"])), replace=False)
            cfgs.append(decode(P[k]["X"][idx[int(np.nanargmax(P[k]["fit"][idx]))]]))
        rs_rows.append({"ann": E.stitched(cfgs, tests)["ann"]})
    rs_df = pd.DataFrame(rs_rows)
    rng_e = np.random.default_rng(EXIT_CTL_SEED)
    ec_rows = []
    for _ in range(n_ec):
        cfgs = []
        for k in range(len(wins)):
            g = decode(P[k]["X"][rng_e.choice(P[k]["pick"])])
            g.update({x: out_cfgs[k][x] for x in EXIT_GENES})
            cfgs.append(canonical(g))
        ec_rows.append({"ann": E.stitched(cfgs, tests)["ann"]})
    ec_df = pd.DataFrame(ec_rows)
    nb = pd.DataFrame(nulls_b)
    timings["post_nulls"] = time.time() - t1

    # --- 기술: 모양 끔 짝·선택도, 중앙값, 민감도 ---
    so = E.stitched([shape_off(c) for c in out_cfgs], tests)
    sel_rows = []
    for k, (c, w) in enumerate(zip(out_cfgs, tests, strict=True)):
        a_, b_ = E.candidates(c, w), E.candidates(shape_off(c), w)
        sel_rows.append({"fold": k, "pass": a_["n_pass"], "pass_shape_off": b_["n_pass"],
                         "selectivity": a_["n_pass"] / b_["n_pass"] if b_["n_pass"] else np.nan})
    taken_so = set(so["rec"][:, PF.Q_EV].astype(int).tolist()) if len(so["rec"]) else set()
    overlap = float(np.mean([e in taken_so for e in tr_df["ev"]])) if len(tr_df) else np.nan
    meds = [median_config(bk) for bk in bests]
    med_ok = all(m is not None for m in meds) and all(E.evaluate(m, tr)["feasible"] for m, (tr, _) in zip(meds, wins, strict=True)) \
        if all(m is not None for m in meds) else False
    med_st = E.stitched(meds, tests) if med_ok else None
    sens = {"비용 1배": E.stitched(out_cfgs, tests, cost_mult=1.0), "보수 p=1": E.stitched(out_cfgs, tests, pmix=1.0),
            "낙관 p=0": E.stitched(out_cfgs, tests, pmix=0.0),
            **{f"미끄러짐 +{x:.1%}": E.stitched(out_cfgs, tests, extra_slip=x) for x in SLIP_SENS},
            "같은 날 무작위 순서": E.stitched(out_cfgs, tests, order_seed=ORDER_SEED)}
    if len(tr_df):
        contrib = (tr_df["ret"] * tr_df["alloc"]).to_numpy()
        top5 = tr_df["ev"].to_numpy()[np.argsort(-contrib)[:5]]
        top5p = tr_df["ev"].to_numpy()[np.argsort(-contrib)[:max(1, int(round(0.05 * len(tr_df))))]]
        sens["상위 5거래 제외"] = E.stitched(out_cfgs, tests, exclude=top5)
        sens["상위 5% 제외"] = E.stitched(out_cfgs, tests, exclude=top5p)
    mfe = {"GA": mfe_table(tr_df, pmix)}
    base_pair = pair_exits(E, tr_df, out_cfgs, live_px)
    pyr_pair = pair_exits(E, tr_df, out_cfgs, pyr_toggle)
    # 기준 청산의 MFE 표(같은 진입, 라이브 청산 — 낙관 경로 총손익만)
    # 팩터·코스피 TVW·DSR·PBO
    F_fac = pd.read_pickle(R.FACTORS_PKL) if os.path.exists(R.FACTORS_PKL) else None
    dates = E.days[st["t_lo"]:st["t_end"]]
    fac = R.nw_factor(pd.Series(st["ret"], dates), F_fac[R.FACTOR_COLS].reindex(dates)) if F_fac is not None else {}
    fac1 = (R.nw_factor(pd.Series(sens["비용 1배"]["ret"], dates), F_fac[R.FACTOR_COLS].reindex(dates))
            if F_fac is not None else {})
    kospi_exc = np.nan
    if F_fac is not None:
        mk = F_fac["MKT_KOSPI_TVW"].reindex(dates).fillna(0.0).to_numpy()
        ex_k = st["ret"] - np.r_[0.0, st["expo"][:-1]] * mk
        mo_k, _ = E.monthly(ex_k, st["t_lo"])
        kospi_exc = float(mo_k.mean() * 12)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in (*R.PRIOR_LEDGERS, ("vcp_retest", "research/logs/sepa_explore"),
                                                                  ("wedge", "research/logs/sepa_explore"))}
    n_led = count_trials(LABEL, logs_dir=led_dir)
    n_dsr_a = n_led + sum(n_prior.values())
    dsr = {}
    for tag, N in (("변형", n_dsr_a), ("구조", DSR_STRUCTS)):
        d_ = deflated_sharpe_from_sample(st["monthly"], N) if len(st["monthly"]) > 2 else None
        dsr[tag] = {"N": N, "prob": float(d_["prob_deflated_sharpe"]) if d_ else np.nan,
                    "sr": float(d_["observed_sharpe"]) if d_ else np.nan}
    pbo = []
    for k, (tr, _) in enumerate(wins):
        cand = {}
        for s in seeds:
            for fv, g in res[(k, s)].get("top", []):
                cand.setdefault(gkey(g), (fv, g))
        top = sorted(cand.values(), key=lambda v: -v[0])[:TOP_PBO]
        if len(top) >= 2:
            M = np.column_stack([E.evaluate(g, tr, want_monthly=True)["monthly"] for _, g in top])
            nbk = min(PBO_BLOCKS, (M.shape[0] // 2) * 2)
            pbo.append(pbo_cscv(M, n_blocks=nbk, metric="sharpe", std_floor=1e-6)["pbo"] if nbk >= 2 else np.nan)
        else:
            pbo.append(np.nan)
    # 트레이드 바(바 6)·통계
    ws = R.wstats(R.expand(tr_df, pmix), n_dsr_a) if len(tr_df) else {"n": 0}
    timings["post"] = time.time() - t1

    # --- 바 산술 ---
    def p95(s):
        return float(np.nanpercentile(s, 95)) if len(s) else np.nan

    def rank(s, v):
        s = pd.Series(s).dropna()
        return float((s < v).mean()) if len(s) and np.isfinite(v) else np.nan

    w2 = np.nan                                       # 바 6 승률 = 2배 순 기준, 본전(|순| < 0.5%) 제외(쐐기 시험과 같은 처리)
    if ws.get("n"):
        xx = R.expand(tr_df, pmix)
        ww, n2 = xx["w"].to_numpy(float), xx["net2"].to_numpy(float)
        wn, ls = ww[n2 >= BE_BAND].sum(), ww[n2 <= -BE_BAND].sum()
        w2 = float(wn / (wn + ls)) if wn + ls > 0 else np.nan
    gl = ws.get("gl_ratio2", np.nan)
    be_need = 1.0 / (1.0 + gl) + B6_MARGIN if np.isfinite(gl) else np.nan
    fold_judge = [n >= FOLD_MIN_FILLS for n in st["fold_n"]]
    bars = {"b1": bool(np.isfinite(st["ann"]) and st["ann"] > 0 and ci_lo > 0),
            "b2a": bool(st["ann"] > p95(null_a["ann"])) if len(null_a) else False,
            "b2b": bool(st["ann"] > p95(nb["ann"])) if len(nb) else False,
            "b3": bool(np.isfinite(fac.get("alpha_t", np.nan)) and fac["alpha_t"] > 2.0),
            "b4": bool(np.isfinite(st["calm_ann"]) and st["calm_ann"] > 0 and np.isfinite(st["ex_bundles_ann"])
                       and st["ex_bundles_ann"] > 0),
            "b6": bool(ws.get("n", 0) >= B6_N and np.isfinite(gl) and gl >= B6_GL and np.isfinite(w2) and w2 > be_need)}
    bars["b2"] = bars["b2a"] and bars["b2b"]
    bars["all"] = all(bars[k] for k in ("b1", "b2", "b3", "b4", "b6"))
    regions = [user_region(c) for c in out_cfgs]
    n_uniq = len(hashes)
    elapsed = time.time() - t0
    V = locals()
    L = render(V)
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    # --- 산출물 ---
    pd.DataFrame([{"fold": k, "seed": s, "train_fit": r["fit"], "train_n": r["n"], "uniq": r["uniq"], "sec": r["sec"],
                   "restarts": len(r["restarts"]), "why": ",".join(x["why"] for x in r["restarts"]),
                   "regimes": ",".join(x["regime"] for x in r["restarts"]), **{g: str(r["best"].get(g)) for g in GENE_NAMES}}
                  for (k, s), r in res.items()]).to_csv(os.path.join(sub_dir, "runs.csv"), index=False)
    null_a.to_csv(os.path.join(sub_dir, "null_2a.csv"), index=False)
    nb.to_csv(os.path.join(sub_dir, "null_2b.csv"), index=False)
    rs_df.to_csv(os.path.join(sub_dir, "random_search.csv"), index=False)
    ec_df.to_csv(os.path.join(sub_dir, "exit_control.csv"), index=False)
    pd.DataFrame(land).to_csv(os.path.join(sub_dir, "landscape.csv"), index=False)
    if len(tr_df):
        tr_df.to_csv(os.path.join(sub_dir, "oos_trades.csv"), index=False)
    pd.DataFrame({"ret": st["ret"], "bench": st["bench"], "expo": st["expo"]}, index=dates).to_csv(
        os.path.join(sub_dir, "oos_daily.csv"))
    with open(os.path.join(sub_dir, "procedure.json"), "w", encoding="utf-8") as fh:
        json.dump({"folds": [list(f) for f in folds_spec], "configs": [{k: (None if v is None else str(v) if v == math.inf else v)
                                                                       for k, v in c.items()} for c in out_cfgs],
                   "reeval_fits": pr["re_fits"], "regions": regions, "p": pmix}, fh, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(sub_dir, "chart_points.json"), "w", encoding="utf-8") as fh:
        json.dump(chart_points(E, tr_df, out_cfgs), fh, ensure_ascii=False, indent=1)
    print(f"\n=== 패턴 GA — {elapsed / 60:.1f}분 · 고유 config {n_uniq:,} · 원장 {n_led}줄 · 폴드 {len(wins)} · 시드 {len(seeds)} · 예산 {budget:,} ===")
    print("시간(s): " + " · ".join(f"{k} {v:.0f}" for k, v in timings.items()))
    print(f"OOS 진입 {st['n']} · 폴드별 {st['fold_n']} · 널 2a {len(null_a)} · 2b {len(nb)} · 무작위 탐색 {len(rs_df)} · 청산 대조 {len(ec_df)}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def chart_points(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> list:
    """MINOR 10: OOS 트레이드 무작위 :data:`CHART_N` 건(seed :data:`CHART_SEED`) — 반전점·선·주문·체결·손절·청산·봉(그리기용)."""
    if t is None or t.empty:
        return []
    rng = np.random.default_rng(CHART_SEED)
    pick = rng.choice(len(t), size=min(CHART_N, len(t)), replace=False)
    out = []
    F = np.asarray(E.F_orig)
    for i in sorted(pick):
        r = t.iloc[i]
        th, cw, bn = PE.table_key(int(r["table"]))
        ci, d = int(r["ci"]), int(r["d_row"])
        pt = PE.pattern_points(np.asarray(E.H), np.asarray(E.L), np.asarray(E.C), ci, d - 1, th, cw)
        lo = max(0, (pt.get("i0", d) if pt.get("i0", -1) >= 0 else d) - 20)
        hi = min(len(E.days) - 1, int(r["x_row"]) + 10)

        def dd(row):
            return str(E.days[min(int(row), len(E.days) - 1)].date())
        bars = [{"date": dd(j), "o": float(E.O[ci, j]), "h": float(E.H[ci, j]), "l": float(E.L[ci, j]), "c": float(E.C[ci, j])}
                for j in range(lo, hi + 1)]
        out.append({"code": str(r["code"]), "fold": int(r["fold"]), "table": {"theta": th, "ceil_win": cw, "entry": bn},
                    "signal_day": dd(d - 1), "order_day": dd(d), "shape": r["shape"],
                    "points": [{"date": dd(p["row"]), "price": p["price"], "kind": p["kind"]} for p in pt.get("points", [])],
                    "upper": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt.get("upper", {}).items()},
                    "lower": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt.get("lower", {}).items()},
                    "level": float(F[int(r["ev"]), PE.FI["level"]]), "order_px": float(F[int(r["ev"]), PE.FI["order"]]),
                    "entry_px": float(r["entry_px"]), "stop_px": float(r["stop_px"]),
                    "exit": {"date": dd(r["x_row"]), "price": float(r["exit_px"]),
                             "reason": str(r["exit_reason"]), "ambiguous_entry_day": bool(r["amb"]),
                             "partial": bool(r["partial"]), "added": bool(r["added"])},
                    "bars": bars})
    return out


# ---------------------------------------------------------------------------
# 10. 서식
# ---------------------------------------------------------------------------

CLAIM_SCOPE = """> **전 바 통과가 확립하는 것:** 2020-01 ~ 2026-08 한국 ADV ≥ 10억에서, 이 연구 계열이 같은 기간을 본 뒤 설계한 패턴·청산 메뉴의 값을 직전 3년 학습창에서 매년 다시 고르는 **절차**가 남긴 의사-표본 밖 기록 하나다. 그 기록이 다음 셋을 넘었다: (a) 같은 메뉴에서 최적화 없이 뽑은 설정들의 95 백분위, (b) 모양 서술자를 섞은 데이터에서 같은 절차를 돌린 결과의 95 백분위, (c) 팩터 절편 t > 2. 이 데이터에서 모양 유전자가 맥락·청산 적재 이상을 더했다는 **약한** 증거다.
>
> **확립하지 않는 것:** (i) 특정 모양(하락 쐐기 등)이 유효하다는 것. 특히 그 모양이 사용자 착상 영역 안이면, 착상 자체가 OOS 연도를 본 데서 나왔으므로 독립 증거가 아니다. (ii) VCP·쐐기 시험의 판정을 바꾸는 것. (iii) 마지막 학습창 config 의 특정 값이 유효하다는 것. 그 config 는 표본 밖 기록이 없다. (iv) 2020~21·2025~26 두 상승 묶음 밖에서 통한다는 것(2022~24 조각을 따로 본다). (v) 이 계열 수십 개 변형·10여 개 검증 구조 뒤의 다중성을 넘는다는 것. (vi) 용량·미끄러짐(10억 종목 돌파 체결)을 넘는다는 것.
>
> **불통과가 확립하는 것:** "이 메뉴에서 매년 다시 고르는 절차는 의사-OOS 에서 같은 메뉴 기준선·모양 섞은 널·팩터를 넘지 못했다." 다른 시드·적합도·기준선의 ✓ 는 판정에 쓰지 않는다. 같은 데이터에서 메뉴를 고쳐 다시 돌리지 않는다.
>
> **허용하는 행동:** 통과해도 전진 페이퍼(고정 config, 페이퍼 판정 규칙 사전등록)까지만이다. 독립 표본(미국)에 같은 사전등록을 한 번 적용하는 것이 다음 증거다."""


def _m(ok) -> str:
    return "✓" if ok else "✗"


def render(V: dict) -> "list[str]":  # noqa: C901
    st, bars, nb, null_a, ws = V["st"], V["bars"], V["nb"], V["null_a"], V["ws"]
    out_cfgs, wins, regions = V["out_cfgs"], V["wins"], V["regions"]
    L = [f"# 패턴 GA — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         f"사전등록 `{PREREG_DOC}` 의 공간·적합도·절차·널·합격 바를 그대로 계산한 숫자다. ✓/✗ 는 산술 비교일 뿐이다. "
         "재현: `uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/pattern_ga_run.py`.", ""]
    if V["debug"]:
        L += [f"> **디버그 실행** — 폴드 {len(wins)} · 시드 {len(V['seeds'])} · 예산 {V['budget']} · 널 2a {len(null_a)} · 2b {len(nb)}, "
              f"출력 `{V['out_dir']}`. 측정으로 쓰지 않는다.", ""]
    reg_line = " · ".join(f"폴드{k + 1} {'**안(' + ','.join(n for n in ('W', 'WD') if r[n]) + ')**' if r['any'] else '밖'}"
                          for k, r in enumerate(regions))
    L += [f"**착상 영역(정정 1 C2·정정 2·정정 3, 결과 전 좌표):** {reg_line}" + (
        " — 영역 안 폴드는 **착상 오염 가능 — 독립 표본 필요**(착상이 OOS 연도를 본 데서 나왔다)." if any(r["any"] for r in regions) else ""), "",
          "## 주장 범위 (리뷰 정본 문구 — 정정 1 C2)", "", CLAIM_SCOPE, "",
          "판정 되살리기 금지: GA 최적이 VCP·쐐기·쐐기 D 정의의 이웃에 떨어져도 그 판정을 바꾸지 않는다. 신뢰 근거는 전진 페이퍼·독립 표본뿐이다.", "",
          "## 합격 바 산술 — 절차 출력(시드 최적 재평가 1등) 이어 붙인 OOS, 2배 비용, 정합 바스켓 대조", "",
          "| 바 | 문턱 | 값 | 충족 |", "|---|---|---|---|",
          f"| 1 | 연환산 초과 > 0 ∧ 월 블록 95% CI 하단 > 0 | {_pct(st['ann'])} [{_pct(V['ci_lo'])}, {_pct(V['ci_hi'])}] | {_m(bars['b1'])} |",
          f"| 2a | > 같은 메뉴 무최적화 실현 가능 설정 p95 | {_pct(st['ann'])} vs {_pct(float(np.nanpercentile(null_a['ann'], 95)) if len(null_a) else np.nan)} "
          f"(백분위 {_f(float((null_a['ann'] < st['ann']).mean()) if len(null_a) else np.nan)}) | {_m(bars['b2a'])} |",
          f"| 2b | > 모양 섞은 널(절차 전체) p95 | {_pct(st['ann'])} vs {_pct(float(np.nanpercentile(nb['ann'], 95)) if len(nb) else np.nan)} "
          f"(백분위 {_f(float((nb['ann'] < st['ann']).mean()) if len(nb) else np.nan)}) | {_m(bars['b2b'])} |",
          f"| 3 | 팩터 α t > 2 (NW 5, {', '.join(['MKT_KOSPI_TVW', 'MKT_KOSDAQ_TVW', 'UMD', 'SMB_ADV', 'LOWVOL'])}) | "
          f"t {_f(V['fac'].get('alpha_t', np.nan))} · α {_pct(V['fac'].get('alpha_ann', np.nan))} (1배 t {_f(V['fac1'].get('alpha_t', np.nan))}) | {_m(bars['b3'])} |",
          f"| 4 | 2022~24 조각 > 0 ∧ 두 상승 묶음(2020~21·2025~26) 뺀 값 > 0 | {_pct(st['calm_ann'])} · {_pct(st['ex_bundles_ann'])} | {_m(bars['b4'])} |",
          f"| 6 | OOS 거래 ≥ {B6_N} ∧ 손익비(2배) ≥ {B6_GL:g} ∧ 승률(본전 제외) > 손익분기 + 5%p | n {ws.get('n', 0)} · "
          f"{_f(ws.get('gl_ratio2', np.nan))} · {_f(V['w2'])} vs {_f(V['be_need'])} | {_m(bars['b6'])} |",
          f"| **전부** | 1·2(a∧b)·3·4·6 | | **{_m(bars['all'])}** |", "",
          f"바 5(기술): 폴드 내 CSCV PBO {[_f(x) for x in V['pbo']]} (리뷰 M3 — 노이즈에서도 0.37~0.59, 과적합 부재의 증거 아님). WFE 삭제. "
          f"DSR(기술): N = 보고 변형 {V['dsr']['변형']['N']} → P {_f(V['dsr']['변형']['prob'], '{:.3f}')} · N = 검증 구조 {DSR_STRUCTS} → "
          f"P {_f(V['dsr']['구조']['prob'], '{:.3f}')}. 폴드 체결 < {FOLD_MIN_FILLS} = 판단 불가: "
          f"{[f'폴드{k + 1}' for k, ok in enumerate(V['fold_judge']) if not ok] or '없음'}.", "",
          "## 폴드별 절차 출력", "",
          "| 폴드 | 학습 | 시험 | 모양 이름 | 재평가 적합도(시드 최적 10) | OOS 연환산 | OOS 체결 | 착상 영역 | config |", "|---|---|---|---|---|---|---|---|---|"]
    for k, (c, (tr, te)) in enumerate(zip(out_cfgs, wins, strict=True)):
        rf = V["pr"]["re_fits"][k]
        L.append(f"| {k + 1} | {tr.lo}~{tr.hi} | {te.lo}~{te.hi} | {config_shape(c)} | 최고 {_f(max(rf), '{:+.3f}')} · 중앙 "
                 f"{_f(float(np.median(rf)), '{:+.3f}')} | {_pct(st['fold_ann'][k])} | {st['fold_n'][k]} | "
                 f"{','.join(n for n in ('W', 'WD') if regions[k][n]) or '밖'} | `{gene_str(c)}` |")
    shapes = pd.Series([config_shape(V["res"][(k, s)]["best"]) for k in range(len(wins)) for s in V["seeds"]])
    L += ["", "**모양 선택의 폴드·시드 분포(사전 예상 — 해석은 모양 섞은 널의 같은 분포와 함께, MINOR 12):** "
          + " · ".join(f"{k} {v}" for k, v in shapes.value_counts().items()),
          f"폴드 간 ŝ_up 중심 표준편차 {_f(float(np.std([c['sup_c'] for c in out_cfgs if c['sup_hw'] is not None])) if sum(c['sup_hw'] is not None for c in out_cfgs) > 1 else np.nan)}"
          f" · 널 2b 중앙 {_f(float(nb['sup_c_sd'].median()) if len(nb) else np.nan)}.",
          f"BIPOP 재시작(런당 평균) {np.mean([len(r['restarts']) for r in V['res'].values()]):.1f} — 0 인 런 "
          f"{sum(len(r['restarts']) <= 1 for r in V['res'].values())}개는 'BIPOP 아님'(M4-2). 고유 config {V['n_uniq']:,}.", "",
          "## 널·대조", "",
          "| 분포 | 벌 | 중앙 | p95 | 실제 백분위 |", "|---|---|---|---|---|"]
    for nm, df in (("(2a) 무최적화 실현 가능 설정", null_a), ("(2b) 모양 섞은 널 — 절차 전체", nb),
                   ("무작위 탐색 best-of-2,000 (기술)", V["rs_df"]), ("청산 전용 대조(GA 청산 고정, 진입·모양 무작위 — 기술)", V["ec_df"])):
        s = df["ann"] if len(df) else pd.Series(dtype=float)
        L.append(f"| {nm} | {len(s)} | {_pct(float(s.median()) if len(s) else np.nan)} | {_pct(float(np.nanpercentile(s, 95)) if len(s) else np.nan)} | "
                 f"{_f(float((s < st['ann']).mean()) if len(s) else np.nan)} |")
    L += ["", f"풀(폴드당 균등 {V['pool_n']:,}) 실현 가능 수: " + " · ".join(f"폴드{k + 1} {v:,}" for k, v in sorted(V["pool_feas"].items()))
          + " — 0 인 폴드는 벌점 최소 1% 설정으로 대신 뽑았다(그 폴드의 2a 는 '실현 가능 조건부' 가 아니다)."]
    so = V["so"]
    gain = st["ann"] - so["ann"]
    L += ["", f"**모양 비활성 짝(M6-1):** 출력 {_pct(st['ann'])} − 모양 끔 {_pct(so['ann'])} = {_pct(gain)} · 널 2b 같은 차이 p95 "
          f"{_pct(float(np.nanpercentile(nb['shape_gain'], 95)) if len(nb) else np.nan)} (백분위 "
          f"{_f(float((nb['shape_gain'] < gain).mean()) if len(nb) else np.nan)}). 모양 선택도(통과 사건 / 모양 끔 통과): "
          + " · ".join(f"폴드{r['fold'] + 1} {_f(r['selectivity'])}" for r in V["sel_rows"])
          + f" · 출력 체결 중 모양 끔 config 도 산 비율 {_f(V['overlap'])}.", ""]
    L += ["## 트레이드·청산 (OOS, p 혼합, 2배 비용)", "",
          f"- n {ws.get('n', 0)} · 승/본전/패 {_f(ws.get('win', np.nan))}/{_f(ws.get('be', np.nan))}/{_f(ws.get('lossc', np.nan))} · "
          f"순 2배 평균 {_pct(ws.get('net2_mean', np.nan))} · 손익비(1배 {_f(ws.get('gl_ratio', np.nan))} · 2배 {_f(ws.get('gl_ratio2', np.nan))}) · "
          f"평균 손실 R {_f(ws.get('avg_loss_R', np.nan), '{:+.2f}')} · 보유 중앙 {_f(ws.get('hold_median', np.nan), '{:.0f}')}일 · 청산 {ws.get('exits', {})}",
          f"- 노출 평균 {_f(float(np.mean(st['expo'])) if len(st['expo']) else np.nan)} → 노출당 연환산 초과 "
          f"{_pct(st['ann'] / float(np.mean(st['expo'])) if len(st['expo']) and np.mean(st['expo']) > 0 else np.nan)} (M1-b)",
          f"- 코스피 TVW × 노출 대조 연환산 초과(원 정의 — 기술) {_pct(V['kospi_exc'])}", ""]
    mf = V["mfe"]["GA"]
    if mf:
        L += ["| MFE 구간 | n | 실현 R 평균 | 되돌림 R |", "|---|---|---|---|"]
        for k, v in mf["buckets"].items():
            L.append(f"| {k} | {v['n']} | {_f(v['real_R'], '{:+.2f}')} | {_f(v['giveback_R'], '{:.2f}')} |")
        L += ["", f"'+1R → 손실' 비율 {_f(mf['plus1R_to_loss'])} · 승자 포착률 중앙(기술 — 승자 조건부) {_f(mf['winner_capture_median'])}", ""]
    for nm, pp in (("기준 청산 짝(라이브 프로토콜 − GA 청산, 같은 진입)", V["base_pair"]), ("불타기 짝(켬↔끔 바꿈 − 원래)", V["pyr_pair"])):
        if pp.get("n"):
            L.append(f"- {nm}: n {pp['n']} · 원래 {_pct(pp['a'])} → 바꿈 {_pct(pp['b'])} · 차이 {_pct(pp['diff'])} [{_pct(pp['lo'])}, {_pct(pp['hi'])}] "
                     "(낙관 경로 총손익, 배분 1 단위, 날짜군집 CI)")
    L += ["", "## 민감도 (기술 — 판정에 쓰지 않는다)", "", "| 칸 | OOS 연환산 초과 | 2022~24 | 체결 |", "|---|---|---|---|"]
    for nm, s in V["sens"].items():
        L.append(f"| {nm} | {_pct(s['ann'])} | {_pct(s['calm_ann'])} | {s['n']} |")
    L.append(f"| 2021 제외 | {_pct(st['ex2021_ann'])} | | |")
    med = V["med_st"]
    L.append(f"| 중앙값 config(모든 시드 같은 모양 ∧ 실현 가능일 때만) | {_pct(med['ann']) if med else '정의 불가'} | "
             f"{_pct(med['calm_ann']) if med else ''} | {med['n'] if med else ''} |")
    adv = V["tr_df"]["adv20"] if len(V["tr_df"]) else pd.Series(dtype=float)
    al = V["tr_df"]["alloc"] if len(V["tr_df"]) else pd.Series(dtype=float)
    L += ["", "## 용량·체결 (M7)", "",
          f"- 개인 계좌 NAV 1억·3억에서 슬롯 금액 / ADV20: 중앙 {_f(float((al * 100 / adv).median()) if len(adv) else np.nan, '{:.2%}')} · "
          f"{_f(float((al * 300 / adv).median()) if len(adv) else np.nan, '{:.2%}')}, 90% {_f(float((al * 300 / adv).quantile(0.9)) if len(adv) else np.nan, '{:.2%}')}(3억).",
          f"- 분봉 p = {V['pmix']:.3f} ({E_meta(V, 'p_window')}, 판별 {E_meta(V, 'p_n')}) — 10억~50억 분봉 커버리지는 `pattern_ga/FEASIBILITY.md`.", "",
          "## 구현 메모 (결과 전 결정)", "",
          "- 사건 표·위생·첫 관통·정규화 좌표 = `research/signals/pattern_events.py` 모듈 설명 1~12, 체결·청산·포트폴리오·대조 = "
          "`research/signals/pattern_fast.py` 모듈 설명. 유전자 부호화·벌점·창 = 이 러너 모듈 설명·상수.",
          "- 명목형(단계·기준선·추적·불타기)은 범주별 연속 로짓 → argmax(M2-1), 구간은 (중심, 반폭)(C3-2), 끔 = 끝 2% 구역. 조건부로 평평한 "
          "유전자는 키에서 `None`(같은 표현형 = 같은 평가).",
          f"- 원장 {V['n_led']}줄 · 이 계열 원장 합 {V['n_dsr_a']} · 실행 {V['elapsed'] / 60:.1f}분 (" + " · ".join(
              f"{k} {v:.0f}s" for k, v in V["timings"].items()) + ")",
          "- OOS 트레이드 차트: `pattern_ga/chart_points.json` (seed 20261011 무작위 12건). **차트를 본 뒤의 정의 수정은 새 사전등록 · 이 데이터 "
          "재시험 금지**(MINOR 10). 이 계열 사슬: VCP → 쐐기 → 쐐기 D → 패턴 GA.", ""]
    return L


def E_meta(V: dict, k: str):
    return V["E"].meta.get(k, "—")


if __name__ == "__main__":
    raise SystemExit(main())
