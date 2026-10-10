#!/usr/bin/env python
"""GA2 — 모든 변수 연속·다단계 + 진입 × 청산, CMA-ES(CMAwM + BIPOP) × 롤링 워크포워드. **탐색이지 판정이 아니다.**

설계: ``research/logs/sepa_explore/GA2.md`` — 421b01a(유전자·보고) → aa25dd0(세 번째 적합도: 자본 제약 처리량) → 172449b(최적화기 CMA-ES)
→ 162103a(검증 구조: 고정 FOLDS 롤링 워크포워드, purge, 라이브 후보, 대체 판단 규칙). 전부 결과 보기 전 커밋. 고정 TRAIN(< 2022) GA2 런은
설계대로 **뺐다**(마지막 변경 절). 이 러너는 **리포터**다 — 숫자를 ``GA2_RESULTS.md`` 에 적고 판단 규칙 문턱과 산술 비교만 한다.

엔진은 ``ga_run`` (사전계산·청산 LRU 캐시·대조 고정 추출·마스크·겹침 제거·포트폴리오)을 그대로 쓴다. 청산 조합은 처음 만날 때 후보 전체
(= 대조 풀)를 numba 로 시뮬해 캐시한다.

- **유전자 19개**(:data:`G2_CONT` 연속 9 — [0,1] 정규화, :data:`G2_DISC` 이산 10 — 단계 인덱스; 시간 손절 → 돌파 실패, 실적 3개 추가). "끔" 은 범위 끝: 연속 유전자는 끝에서
  :data:`OFF_ZONE` (범위의 2%) 안이면 조건 자체를 끈다(NaN 도 통과 — CMA 표본이 정확히 끝값을 거의 안 내므로 끔을 닿을 수 있게).
  이산 "없음/∞" 은 마지막(또는 0) 단계.
- **적합도 3종**(처리량을 먼저 적는다): ``port`` = 슬롯 10 × 10% 포트폴리오 TRAIN 일별 정합 초과의 월합 3개월 블록 부트스트랩(500) 연환산 2.5% 하단,
  ``trade`` = 트레이드 정합 초과 부트스트랩 2.5% 하단, ``day`` = Σ초과/Σ보유일 부트스트랩 2.5% 하단. 셋 다 거래수 < 150 이면 하한 페널티.
- **폴드:** ``swing_it.validation.walkforward.FOLDS`` 6개(새로 만들지 않는다). 학습 트레이드 = 진입 ∈ [train_lo, train_hi), test_lo 를 넘는 것은 그 직전 종가에서 마크투마켓
  (설계의 purge '청산 < test_lo' 는 리뷰 K-C1 — 긴 보유 승자를 버리는 결과 조건부 선택 — 로 실행 전 이렇게 바꿨다; 대조·포트폴리오도 같은 날).
  폴드 × 적합도 × 시드(0·1·2) = 54 런, 런당 1,600 평가(``ask`` 수, 캐시 적중 포함).
- **표본 밖:** 폴드 k 시험 연도 진입에 폴드 k config 를 한 번 — 이어 붙인 하나의 연속 포트폴리오(폴드가 바뀌면 신규 진입만 새 config, 열린
  포지션은 원래 규칙으로 청산, 슬롯 공유). 시드 최적 3개 각각과 시드 유전자 중앙값 config. 판단 규칙은 **처리량 · 유전자 중앙값** 시계열에 건다.
- **라이브 후보:** 사건 2023-04-01 ~ 2026-04-01, 청산 < 데이터 끝(마지막 거래일) — 같은 절차(처리량, 시드 3, 중앙값) → ``live_candidate.json``.
- **수급 시대:** 폴드 6(학습 2022~2024) 처리량 시드 s 최적 위에 수급 유전자 3개, 학습 진입 2024-02-19 ~ 2025-06-30(청산 < 2025-07-01), OOS 2025-07-01 ~,
  시드 3 × 450 평가.
- 원장: ``research/logs/sepa_explore/ga2/TRIALS.jsonl`` — 평가한 고유 config(유전자) 마다 한 줄.

Run:  NUMBA_NUM_THREADS=8 uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/ga2_run.py --cache <npz>
"""

from __future__ import annotations

import argparse
import hashlib
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
from ga_run import (  # noqa: E402
    CHASES,
    EVENT_HI,
    N_BOOT_FIT,
    W50,
    WINDOWS,
    Engine,
    _f,
    _log,
    _pct,
    cma_bipop,
    load_or_build,
    read_fingerprints,
    write_ledger,
)

from research.signals import ga_fast as gf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "ga2"
RESULTS_MD = "GA2_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/GA2.md (421b01a → aa25dd0 → 172449b → 162103a)"
PRIOR_LEDGERS = g1.PRIOR_LEDGERS + (("ga", "research/logs/sepa_explore"),)

FITS = ("port", "trade", "day")
FIT_NAMES = {"port": "처리량(포트폴리오 연환산 초과)", "trade": "트레이드당 정합 초과", "day": "보유 1일당 정합 초과"}
SEEDS = (0, 1, 2)
BUDGET = 1600
FLOW_BUDGET, FLOW_SEEDS = 450, (0, 1, 2)
LIVE_LO, LIVE_HI = "2023-04-01", "2026-04-01"
FLOW_TR_LO, FLOW_TR_HI = "2024-02-19", "2025-07-01"
OFF_ZONE = 0.02
CI_BOOT = 2000
#: 처리량 적합도 하한 — 채운 트레이드 150 이 아니라 슬롯을 받은 **주문(시도한 진입) 수 ≥ 60** (리뷰 K2-M1 반영, GA2 재실행 전 결정).
PORT_FLOOR = 60
EVENT_HI_NEXT = (pd.Timestamp(EVENT_HI) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
#: 학습 트레이드·대조·포트폴리오는 학습창 끝(test_lo 직전 거래일 종가)에서 마크투마켓 — purge(청산 < test_lo 만 남김) 대신(리뷰 K-C1).

#: 연속 유전자 (이름, 하한, 상한, 끔 끝 "lo"/"hi")
G2_CONT = (("near_high", 0.50, 0.95, "lo"), ("above_low", 1.00, 1.60, "lo"), ("rs_min", 0.0, 95.0, "lo"),
           ("vol_mult", 0.0, 2.5, "lo"), ("breadth_min", 0.0, 0.7, "lo"), ("pead_min", 0.0, 0.95, "lo"),
           ("rel_max", 0.5, 4.0, "hi"), ("eps_yoy_min", 0.0, 1.0, "lo"), ("rev_yoy_min", -0.2, 0.5, "lo"))
#: 이산 유전자 (이름, 단계) — 0 / inf = 없음
G2_DISC = (("eps_accel", (0, 1)), ("ma_stack", (0, 1, 2, 3)), ("ma200_slope_days", (0, 10, 20, 40)), ("pivot_window", (0, 20, 30, 40, 50, 60)),
           ("chase", (0.02, 0.05, 0.10, math.inf)), ("stop_cap", (0.06, 0.08, 0.10, 0.12, 0.15)),
           ("take_r", (1.5, 2.0, 3.0, 4.0, math.inf)), ("trail_ma", (10, 20, 50, 0)), ("failed_breakout", (0, 1)),
           ("max_hold", (20, 60, 120)))
ENTRY_GENES = tuple(n for n, *_ in G2_CONT) + ("eps_accel", "ma_stack", "ma200_slope_days", "pivot_window", "chase")
EXIT_GENES = ("stop_cap", "take_r", "trail_ma", "failed_breakout", "max_hold")
GENE_ORDER = tuple(n for n, *_ in G2_CONT) + tuple(n for n, _ in G2_DISC)
DISC_SIZES = tuple(len(v) for _, v in G2_DISC)
#: 수급 유전자: 연속 smart_ratio_min [−0.10, 0.20], 이산 foreign {끔, > 0}, persistence {0, 8, 10, 12}
FLOW_CONT = (("smart_min", -0.10, 0.20),)
FLOW_DISC = (("foreign_pos", (0, 1)), ("persist_min", (0, 8, 10, 12)))


# ---------------------------------------------------------------------------
# 순수 헬퍼 — 부호화/복호화 (tests/test_ga_run_helpers.py)
# ---------------------------------------------------------------------------

def g2_decode(x) -> dict:
    """CMA 벡터(연속 7 [0,1] + 이산 9 단계 인덱스) → 유전자 dict. 연속 끔 = ``None``, 이산은 단계 값 그대로."""
    x = np.asarray(x, float)
    out: dict = {}
    for i, (name, lo, hi, off) in enumerate(G2_CONT):
        u = float(np.clip(x[i], 0.0, 1.0))
        if (off == "lo" and u <= OFF_ZONE) or (off == "hi" and u >= 1.0 - OFF_ZONE):
            out[name] = None
        else:
            out[name] = round(lo + u * (hi - lo), 4)
    for j, (name, levels) in enumerate(G2_DISC):
        k = int(np.clip(round(float(x[len(G2_CONT) + j])), 0, len(levels) - 1))
        out[name] = levels[k]
    return out


def g2_encode(genes: dict) -> np.ndarray:
    """유전자 dict → CMA 벡터(끔 = 끝값 0 또는 1). :func:`g2_decode` 의 역(연속은 4자리 반올림 오차 안에서)."""
    v = []
    for name, lo, hi, off in G2_CONT:
        g = genes[name]
        v.append((0.0 if off == "lo" else 1.0) if g is None else (g - lo) / (hi - lo))
    for name, levels in G2_DISC:
        v.append(float(levels.index(genes[name])))
    return np.array(v)


def g2_key(genes: dict) -> tuple:
    return tuple(genes[k] for k in GENE_ORDER)


def g2_spec(genes: dict) -> dict:
    """유전자 → 엔진 사양(``mask_ga2`` 의 g 벡터·청산 키·추격 창/단계)."""
    def nz(v):
        return np.nan if v is None else float(v)

    pw = genes["pivot_window"]
    pw_idx = WINDOWS.index(pw) if pw else -1
    slope = {0: 0, 10: 1, 20: 2, 40: 3}[genes["ma200_slope_days"]]
    g = np.array([nz(genes["near_high"]), nz(genes["above_low"]), float(genes["ma_stack"]), float(slope),
                  nz(genes["rs_min"]), float(pw_idx), nz(genes["vol_mult"]), nz(genes["breadth_min"]),
                  nz(genes["pead_min"]), nz(genes["rel_max"]), nz(genes["eps_yoy_min"]), float(genes["eps_accel"]),
                  nz(genes["rev_yoy_min"])])
    w_idx = pw_idx if pw_idx >= 0 else W50
    fb = int(genes["failed_breakout"])
    # 청산 키: 시간 손절 없음(0) · 돌파 실패 켬이면 그 config 의 피벗 창(돌파 없음이면 50일) 피벗 기준
    ek = (float(genes["stop_cap"]), float(genes["take_r"]), int(genes["trail_ma"]), 0, int(genes["max_hold"]), fb,
          w_idx if fb else -1)
    return {"kind": "ga2", "g": g, "ek": ek, "w_idx": w_idx, "c_idx": CHASES.index(genes["chase"])}


def g2_median(cfgs: "list[dict]") -> dict:
    """시드 유전자 중앙값 config — 연속은 끔을 끔 끝값으로 두고 중앙값(끝 구역이면 다시 끔), 이산은 단계 인덱스 중앙값(짝수면 아래)."""
    xs = np.array([g2_encode(c) for c in cfgs])
    med = np.empty(xs.shape[1])
    nc = len(G2_CONT)
    med[:nc] = np.median(xs[:, :nc], axis=0)
    for j in range(nc, xs.shape[1]):
        s = np.sort(xs[:, j])
        med[j] = s[(len(s) - 1) // 2]
    return g2_decode(med)


def flow_decode(x) -> dict:
    x = np.asarray(x, float)
    name, lo, hi = FLOW_CONT[0]
    out = {name: round(lo + float(np.clip(x[0], 0, 1)) * (hi - lo), 4)}
    for j, (n, levels) in enumerate(FLOW_DISC):
        out[n] = levels[int(np.clip(round(float(x[1 + j])), 0, len(levels) - 1))]
    return out


def flow_extra(pre, fl: dict) -> np.ndarray:
    """수급 유전자 → 쌍별 추가 조건(NaN 은 통과 안 함)."""
    with np.errstate(invalid="ignore"):
        ok = pre.smart >= fl["smart_min"]
        if fl["foreign_pos"]:
            ok &= pre.foreign > 0
        if fl["persist_min"]:
            ok &= pre.persist >= fl["persist_min"]
    return np.ascontiguousarray(ok)


def gene_str(g: dict) -> str:
    def c(v, fmt="{:.2f}"):
        return "끔" if v is None else fmt.format(v)

    def inf(v):
        return "∞" if v == math.inf else f"{v:g}"

    return (f"nh≥{c(g['near_high'])} al≥{c(g['above_low'])} stack≥{g['ma_stack']} slope{g['ma200_slope_days'] or '끔'} "
            f"rs≥{c(g['rs_min'], '{:.0f}')} 창{g['pivot_window'] or '없음'} vol≥{c(g['vol_mult'])} br>{c(g['breadth_min'])} "
            f"pead≥{c(g['pead_min'])} rel≤{c(g['rel_max'])} eps≥{c(g['eps_yoy_min'])} 가속{'켬' if g['eps_accel'] else '끔'} "
            f"매출≥{c(g['rev_yoy_min'])} 추격{inf(g['chase'])} | 손절상한{g['stop_cap']:g} "
            f"익절{inf(g['take_r'])}R 추적{g['trail_ma'] or '없음'} 돌파실패{'켬' if g['failed_breakout'] else '끔'} 최대{g['max_hold']}")


def is_off(name: str, v) -> bool:
    return v is None or (name in ("ma_stack", "ma200_slope_days", "pivot_window", "trail_ma", "failed_breakout", "eps_accel")
                         and v == 0) \
        or (name in ("chase", "take_r") and v == math.inf)


def wfe(oos: "list[float]", ins: "list[float]") -> float:
    """워크포워드 효율 = Σ 폴드 OOS 연환산 초과 / Σ 폴드 IS 연환산 초과(점추정). 분모 ≤ 0 이면 NaN(전제 미충족)."""
    so, si = float(np.nansum(oos)), float(np.nansum(ins))
    return so / si if si > 0 else np.nan


# ---------------------------------------------------------------------------
# 창·평가
# ---------------------------------------------------------------------------

class Win:
    """창 한 벌: 쌍 범위·신호일 행·절단 행(``cut`` — 그 직전 거래일 종가에서 트레이드·대조·포트폴리오를 마크투마켓, None = 없음)."""

    def __init__(self, pre, lo: "str | None", hi: "str | None", cut: "str | None", *, by_event: bool = False):
        if by_event:
            d_lo = pre.day_idx(lo)
            d_hi = pre.day_idx((pd.Timestamp(hi) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            self.i0, self.i1 = int(pre.day_start[d_lo]), int(pre.day_start[d_hi])
            self.d_lo, self.d_hi = d_lo, d_hi
        else:
            self.i0, self.i1, self.d_lo, self.d_hi = pre.entry_window(lo, hi)
        self.cut = pre.day_idx(cut) if cut else None
        # 첫 진입일(사건은 2017-09-15 부터 — 폴드 1 학습창 앞 빈 달을 연환산에 넣지 않는다)
        self.d_first = max(self.d_lo + 1, int(pre.pd[self.i0]) + 1 if self.i0 < pre.P else self.d_lo + 1)
        self.end = self.cut if self.cut is not None else pre.n_days     # 포트폴리오 날 상한(제외)
        self.lo, self.hi, self.cut_date = lo, hi, cut


def fitness(E: Engine, kind: str, spec: dict, w: Win, *, extra=None, seed: int = 0):
    """적합도 3종 — (값, n). 처리량 n = 슬롯을 받은 주문 수(하한 :data:`PORT_FLOOR`), 나머지 n = 짝 있는 트레이드(하한 150)."""
    if kind == "port":
        daily, _, n_ord, _ = E.portfolio(spec, w.i0, w.i1, w.d_lo, w.end, w.cut, extra=extra)
        if n_ord < PORT_FLOOR:
            return gf.floor_penalty(n_ord, PORT_FLOOR), n_ord
        mo = E.monthly(daily, w.d_lo, w.d_first, w.end)
        if len(mo) < 4:
            return gf.floor_penalty(0, PORT_FLOOR), n_ord
        return float(np.percentile(gf.block_boot(mo, 3, N_BOOT_FIT, seed), 2.5)), n_ord
    return E.fit_trade(spec, w.i0, w.i1, w.cut, per_day=(kind == "day"), seed=seed, extra=extra)


def port_point(E: Engine, daily: np.ndarray, t0: int, d_from: int, d_to: int) -> dict:
    """포트폴리오 날별 초과 → 연환산(월 평균 × 12)·MDD(단리 누적, 초기 자산 대비)·월 3개월 블록 CI."""
    mo = E.monthly(daily, t0, d_from, d_to)
    seg = daily[d_from - t0:d_to - t0]
    cum = np.cumsum(seg)
    mdd = float(np.max(np.maximum.accumulate(np.r_[0.0, cum]) - np.r_[0.0, cum])) if len(cum) else np.nan
    out = {"ann": float(mo.mean() * 12) if len(mo) else np.nan, "months": len(mo), "mdd": mdd,
           "total": float(seg.sum()), "monthly": mo}
    if len(mo) >= 6:
        v = gf.block_boot(mo, 3, CI_BOOT, 0)
        out["lo"], out["hi"] = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    else:
        out["lo"] = out["hi"] = np.nan
    return out


# ---------------------------------------------------------------------------
# 이어 붙인 OOS (폴드별 config → 하나의 연속 포트폴리오, 폴드 경계 마크투마켓)
# ---------------------------------------------------------------------------

def composite(E: Engine, pairs: "list[tuple[dict, Win]]", band_sel: int = -1, extras=None) -> dict:
    """폴드별 (사양, 시험 창) → 쌍 배열을 창마다 그 config 값으로 채운 합성 배열(신호·체결·청산·대조 번호·절단 행)."""
    p = E.p
    arr = {"sig": np.zeros(p.P, np.bool_), "fill": np.zeros(p.P, np.bool_), "x": np.full(p.P, -1, np.int32),
           "pab": np.full(p.P, -1, np.int32), "ret": np.full(p.P, np.nan), "r2": np.full(p.P, np.nan, np.float32),
           "cidx": np.full((p.P, g1.N_PER), -1, np.int32), "cut": np.full(p.P, p.n_days + 1, np.int32),
           "reason": np.full(p.P, -1, np.int8)}
    for k, (spec, w) in enumerate(pairs):
        e = E.exit(spec["ek"], w.i0, w.i1)
        fill, cidx = E.fill_ctl(spec["ek"], spec["w_idx"], spec["c_idx"])
        s = slice(w.i0, w.i1)
        ex = None if extras is None else extras[k]
        arr["sig"][s] = E.mask(spec, w.i0, w.i1, band_sel, ex)
        if w.cut is not None:                       # 진입 행 ≥ 절단 행(창 마지막 신호일)은 신호에서 뺀다(리뷰 A-M1)
            arr["sig"][s] &= (E.p.pd[s] + 1 < w.cut)
        arr["fill"][s] = fill[s]
        arr["cidx"][s] = cidx[s]
        arr["cut"][s] = w.cut if w.cut is not None else p.n_days + 1
        for key in ("x", "pab", "ret", "r2", "reason"):
            arr[key][s] = e[key][s]
    return arr


def comp_trades(E: Engine, arr: dict, i0: int, i1: int) -> np.ndarray:
    """겹침 제거 트레이드 — 진입 행 ≥ 절단 행인 쌍(창 마지막 신호일)은 뺀다: 포트폴리오에서도 손익이 0 이고 시가평가를 우회한다(리뷰 A-M1)."""
    ok = arr["sig"] & arr["fill"] & (E.p.pd + 1 < arr["cut"])
    return gf.greedy(ok, E.p.pc, E.p.pd, arr["x"], E.p.n_codes, i0, i1)


def comp_report(E: Engine, arr: dict, sel: np.ndarray, n_dsr: "int | None" = None) -> dict:
    """합성 배열 위 트레이드 서술(``ga_run.report_from`` — 절단·대조 번호 반영)."""
    p = E.p
    if len(sel) == 0:
        return {"n": 0}
    sel = np.ascontiguousarray(sel, np.int64)
    ex, hold, cost, rr, xx = gf.trade_excess(sel, arr["cut"], p.pc, p.pd, arr["x"], arr["pab"], arr["ret"], arr["r2"],
                                             p.O, p.C, arr["cidx"], p.cost_tab, p.market)
    r = {"ex": ex, "hold": hold, "cost": cost, "ret": rr, "x": xx}
    reason = np.where(xx < arr["x"][sel], -2, arr["reason"][sel])
    return g1.report_from(r, reason, arr["pab"][sel], p.pd[sel], n_dsr)


def stitched(E: Engine, folds_w: "list[tuple[Win, Win]]", cfgs: "list[dict]", band_sel: int = -1,
             n_dsr: "int | None" = None) -> dict:
    """폴드별 config → 이어 붙인 OOS: 연속 포트폴리오(폴드 경계에서 마크투마켓 — 조각 = 그 폴드 config 만) + 겹침 제거 트레이드."""
    p = E.p
    tests = [te for _, te in folds_w]
    arr = composite(E, [(g2_spec(c), te) for c, te in zip(cfgs, tests, strict=True)], band_sel)
    t0, end = tests[0].d_lo, tests[-1].end
    daily, opened, n_ord, n_unm = E.portfolio(None, 0, 0, t0, end, None, composite=arr)
    d_from = t0 + 1
    out = {"port": port_point(E, daily, t0, d_from, end)}
    out["fold_port"] = [port_point(E, daily, t0, te.d_first, te.end) for te in tests]
    years = (end - d_from) / 252.0
    out["port"].update({"trades": int(len(opened)), "orders": int(n_ord), "unmatched": int(n_unm),
                        "turnover": 2 * gf.SLOT_W * len(opened) / years if years > 0 else np.nan})
    xe = np.minimum(arr["x"][opened], arr["cut"][opened] - 1)
    out["port"]["util"] = float((xe - p.pd[opened]).sum() / (gf.SLOTS * max(end - d_from, 1)))
    # 2020·2021·2025~ (상승장 연도) 를 뺀 시험 연도 조각만 이어 붙인 연환산
    calm = [fp["monthly"] for te, fp in zip(tests, out["fold_port"], strict=True)
            if str(te.lo)[:4] in ("2022", "2023", "2024")]
    out["calm_ann"] = float(np.concatenate(calm).mean() * 12) if calm else np.nan
    sel = comp_trades(E, arr, tests[0].i0, tests[-1].i1)
    out["trades"] = comp_report(E, arr, sel, n_dsr)
    out["fold_trades"] = [comp_report(E, arr, sel[(sel >= te.i0) & (sel < te.i1)]) for te in tests]
    out["daily"], out["t0"] = daily, t0
    return out


def is_point(E: Engine, cfg: dict, w: Win) -> dict:
    """학습창 포트폴리오 점추정(연환산, 학습창 끝 마크투마켓) + 같은 절단의 학습 트레이드 서술."""
    spec = g2_spec(cfg)
    daily, opened, n_ord, n_unm = E.portfolio(spec, w.i0, w.i1, w.d_lo, w.end, w.cut)
    pp = port_point(E, daily, w.d_lo, w.d_first, w.end)
    pp.update({"n": int(len(opened)), "orders": int(n_ord), "unmatched": int(n_unm)})
    sel = E.select(spec, w.i0, w.i1)
    return {"port": pp, "trades": g1.trade_report(E, spec, sel, cut=w.cut)}


# ---------------------------------------------------------------------------
# 병렬 작업자 (spawn — 사전계산은 .npy mmap 으로 공유, 작업자마다 Engine·청산 캐시)
# ---------------------------------------------------------------------------

NOISE_FEATS = ("tt", "rs", "breadth", "pead", "rel", "brk_bits", "vol", "vavg", "nh", "al", "stack", "slope_bits",
               "eps_yoy", "eps_acc", "rev_yoy")
_W: dict = {}


def worker_init(pre_path: str, threads: int = 1) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(threads)
    pre = g1.Pre.load(pre_path)
    _W.update(pre=pre, E=Engine(pre, max_exit=24), orig={k: getattr(pre, k) for k in NOISE_FEATS}, rep=None)


def set_noise(rep: "int | None") -> None:
    """노이즈 기준선: 진입 피처를 **같은 신호월 안에서** 쌍끼리 무작위 재배정(가격·체결·층·대조는 그대로 — 리뷰 noise_wfo 와 같은 식)."""
    if _W["rep"] == rep:
        return
    pre = _W["pre"]
    if rep is None:
        for k, v in _W["orig"].items():
            setattr(pre, k, v)
    else:
        rng = np.random.default_rng(1000 + rep)
        m = pre.month_id[pre.pd]
        order = np.argsort(m, kind="stable")
        b = np.searchsorted(m[order], np.arange(m.max() + 2))
        perm = np.arange(pre.P)
        for j in range(len(b) - 1):
            idx = order[b[j]:b[j + 1]]
            perm[idx] = rng.permutation(idx)
        for k, v in _W["orig"].items():
            setattr(pre, k, np.ascontiguousarray(np.asarray(v)[perm]))
    _W["rep"] = rep


def _key_hash(key: tuple) -> int:
    return int.from_bytes(hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "little", signed=True)


def cma_task(t: dict) -> dict:
    """CMA-ES 한 런(창·적합도·시드). 노이즈면 피처를 섞은 뒤. 반환 = 최적·적합도·수렴·재시작·평가한 고유 config 해시."""
    set_noise(t.get("rep"))
    E, pre = _W["E"], _W["pre"]
    w = Win(pre, *t["win"], by_event=t.get("by_event", False))
    memo: dict = {}
    base = t.get("base")
    tt = time.time()

    def fn(x):
        if base is not None:
            fl = flow_decode(x)
            key = (g2_key(base), tuple(fl.values()))
            if key not in memo:
                memo[key] = fitness(E, t["kind"], g2_spec(base), w, extra=flow_extra(pre, fl))
        else:
            genes = g2_decode(x)
            key = g2_key(genes)
            if key not in memo:
                memo[key] = fitness(E, t["kind"], g2_spec(genes), w)
        return memo[key][0]

    if base is not None:
        hist, rst = cma_bipop(fn, 1, tuple(len(v) for _, v in FLOW_DISC), budget=t["budget"], seed=t["seed"], stall=True)
        bi = int(np.argmax([h[2] for h in hist]))
        best = flow_decode(hist[bi][1])
        nkey = (g2_key(base), tuple(best.values()))
    else:
        hist, rst = cma_bipop(fn, len(G2_CONT), DISC_SIZES, budget=t["budget"], seed=t["seed"], stall=True)
        bi = int(np.argmax([h[2] for h in hist]))
        best = g2_decode(hist[bi][1])
        nkey = g2_key(best)
    curve = np.maximum.accumulate([h[2] for h in hist])
    marks = [m for m in (100, 400, 800, 1600, 3200, 6400, 12800) if m <= len(curve)] + [len(curve)]
    return {"id": t["id"], "best": best, "fit": hist[bi][2], "n": memo[nkey][1], "restarts": rst,
            "curve": {m: float(curve[m - 1]) for m in marks}, "sec": time.time() - tt, "n_eval": len(hist),
            "uniq": len(memo), "hashes": None if t.get("rep") is not None else np.array([_key_hash(k) for k in memo])}


def proc_stats(E: Engine, kind: str, wins: "list[tuple[Win, Win]]", bests: "list[list[dict]]", *, bands=("H5",),
               with_is: bool = True, n_dsr: "int | None" = None) -> dict:
    """창별 시드 최적 → 시드 중앙값·이어 붙인 OOS·IS 점추정·WFE·판단 규칙 숫자(노이즈와 실제가 같은 함수를 쓴다)."""
    meds = [g2_median(b) for b in bests]
    out = {"meds": meds, "is": [], "fit": []}
    if with_is:
        for (tr, _), m in zip(wins, meds, strict=True):
            out["is"].append(is_point(E, m, tr))
            out["fit"].append(fitness(E, kind, g2_spec(m), tr)[0])
    out["st"] = {b: stitched(E, wins, meds, band_sel={"H5": -1, "유동": 1, "중소형": 0}[b], n_dsr=n_dsr) for b in bands}
    sm = out["st"]["H5"]
    oos = [fp["ann"] for fp in sm["fold_port"]]
    ins = [x["port"]["ann"] for x in out["is"]] if with_is else []
    pos = [bool(np.isfinite(v) and v > 0) for v in oos]
    out["dec"] = {"folds_pos": int(sum(pos)), "n_folds": len(pos), "pos_frac": float(np.mean(pos)) if pos else np.nan,
                  "ci_lo": sm["port"]["lo"], "ann": sm["port"]["ann"], "wfe": wfe(oos, ins) if with_is else np.nan,
                  "is_sum": float(np.nansum(ins)) if with_is else np.nan, "is_mean": float(np.nanmean(ins)) if ins else np.nan,
                  "calm_ann": sm["calm_ann"]}
    return out


def pbo_of(E: Engine, cfgs: "list[dict]") -> dict:
    """후보 config 들의 전 구간(2017-10 ~ 2026-04, 2026-04-01 종가 마크투마켓) 월별 포트폴리오 초과 → CSCV."""
    from swing_it.diagnostics.pbo import pbo_cscv  # noqa: PLC0415
    pre = E.p
    full = Win(pre, None, None, EVENT_HI_NEXT)
    m0 = pre.day_idx("2017-10-01")
    cols = []
    for c in cfgs:
        daily, _, _, _ = E.portfolio(g2_spec(c), full.i0, full.i1, full.d_lo, full.end, full.cut)
        cols.append(E.monthly(daily, full.d_lo, m0, full.end))
    M = np.column_stack(cols)
    return {mt: pbo_cscv(M, n_blocks=16, metric=mt, std_floor=1e-6) for mt in ("sharpe", "mean")}


def noise_task(t: dict) -> dict:
    """노이즈 기준선 한 벌 = 섞은 피처 위에서 **같은 절차 전체**(연 폴드 6 × 처리량 × 시드 5 → 중앙값 → 이어 붙인 OOS·WFE·PBO)."""
    rep = t["rep"]
    set_noise(rep)
    E, pre = _W["E"], _W["pre"]
    tt = time.time()
    wins = [(Win(pre, *a, by_event=False), Win(pre, *b, by_event=False)) for a, b in t["folds"]]
    bests, fits, rsts = [], [], []
    for k, (tr, _) in enumerate(t["folds"]):
        bk = []
        for s in t["seeds"]:
            r = cma_task({"id": ("noise", rep, k, s), "rep": rep, "win": tr, "kind": "port", "seed": s, "budget": t["budget"]})
            bk.append(r["best"])
            fits.append(r["fit"])
            rsts.append(len(r["restarts"]))
        bests.append(bk)
    ps = proc_stats(E, "port", wins, bests)
    pb = pbo_of(E, [b for bk in bests for b in bk])
    d = {**ps["dec"], "pbo": pb["sharpe"]["pbo"], "fit_mean": float(np.mean(fits)), "restarts_mean": float(np.mean(rsts)),
         "fold_oos": [fp["ann"] for fp in ps["st"]["H5"]["fold_port"]], "sec": time.time() - tt}
    d["r1"] = d["folds_pos"] >= 4
    d["r2"] = bool(np.isfinite(d["ci_lo"]) and d["ci_lo"] > 0)
    d["r3"] = bool(np.isfinite(d["wfe"]) and d["wfe"] > 0.3)
    d["r4"] = bool(d["pbo"] < 0.5)
    d["all"] = d["r1"] and d["r2"] and d["r3"] and d["r4"]
    return {"id": ("noise", rep), "dec": d}


def landscape_task(t: dict) -> list:
    """폴드 중앙값 config 에서 한 유전자씩 범위 전체 — 학습 적합도·시험 연도 포트폴리오·트레이드 α."""
    set_noise(None)
    E, pre = _W["E"], _W["pre"]
    tr, te = Win(pre, *t["tr"]), Win(pre, *t["te"])
    base, kind = t["base"], t["kind"]
    x0 = g2_encode(base)
    rows = []
    for gi, name in enumerate(GENE_ORDER):
        grid = np.linspace(0, 1, 11) if gi < len(G2_CONT) else np.arange(DISC_SIZES[gi - len(G2_CONT)])
        for u in grid:
            x = x0.copy()
            x[gi] = u
            c = g2_decode(x)
            spec = g2_spec(c)
            fv, n = fitness(E, kind, spec, tr)
            daily, _, _, _ = E.portfolio(spec, te.i0, te.i1, te.d_lo, te.end, te.cut)
            oo = port_point(E, daily, te.d_lo, te.d_first, te.end)
            sel = E.select(spec, te.i0, te.i1)
            ex, _ = E.excess(spec, sel, te.cut)
            rows.append({"fitness": kind, "fold": t["fold"], "gene": name, "grid": float(u), "value": str(c[name]),
                         "train_fit": fv, "train_n": n, "oos_port_ann": oo["ann"],
                         "oos_alpha": float(np.nanmean(ex)) if len(ex) else np.nan, "oos_n": int(len(sel)),
                         "is_base": bool(np.isclose(u, x0[gi])), "hash": _key_hash(g2_key(c))})
    return rows


def quarters(lo: str = "2020-01-01", hi: str = EVENT_HI_NEXT) -> "list[tuple[str, str]]":
    """분기 시험창 [q, q+3개월) — 2020Q1 ~ 사건 상한. 상한 앞 30일 미만 조각 분기(예: 2026-04-01 → 04-02 하루)는 뺀다(리뷰 A minor)."""
    qs = pd.date_range(lo, hi, freq="QS")
    out = [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(qs[:-1], qs[1:])]
    if qs[-1] < pd.Timestamp(hi) and (pd.Timestamp(hi) - qs[-1]).days >= 30:
        out.append((qs[-1].strftime("%Y-%m-%d"), hi))
    return out


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    from swing_it.diagnostics.trials import count_trials
    from swing_it.validation.walkforward import FOLDS

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="사전계산 .npy 디렉터리(없으면 만든다)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=int, default=0, help="0 = 시간 추정으로 6,400 또는 3,200")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise", type=int, default=30)
    ap.add_argument("--flow-budget", type=int, default=FLOW_BUDGET)
    ap.add_argument("--folds", default=None, help="디버그 전용")
    ap.add_argument("--quarters", type=int, default=0, help="디버그 전용 — 분기 수 제한(0 = 전부)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.folds or args.quarters or args.noise != 30 or args.seeds != 5) and not debug:
        ap.error("축소 실행은 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    os.makedirs(os.path.join(out_dir, LABEL), exist_ok=True)
    t0 = time.time()
    pre = load_or_build(args.cache, t0)
    seeds = tuple(range(args.seeds))
    fold_ids = [int(x) for x in args.folds.split(",")] if args.folds else list(range(len(FOLDS)))
    folds = [FOLDS[k] for k in fold_ids]
    cuts = [FOLDS[k + 1].test_lo if k + 1 < len(FOLDS) else EVENT_HI_NEXT for k in fold_ids]
    fold_args = [((f.train_lo, f.train_hi, f.test_lo), (f.test_lo, f.test_hi, ct)) for f, ct in zip(folds, cuts, strict=True)]
    qs = quarters()
    if args.quarters:
        qs = qs[-args.quarters:]
    q_args = []
    for i, (a, b) in enumerate(qs):
        tr_lo = (pd.Timestamp(a) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
        nxt = qs[i + 1][0] if i + 1 < len(qs) else EVENT_HI_NEXT
        q_args.append(((tr_lo, a, a), (a, b, nxt)))

    # --- 시간 추정(작업자 1개 상당 = 단일 스레드, 처리량 런 400 평가) ---
    worker_init(args.cache, threads=1)
    cma_task({"id": "warm", "win": fold_args[-1][0], "kind": "port", "seed": 98, "budget": 30})     # JIT 예열(시간에서 뺀다)
    cal = cma_task({"id": "cal", "win": fold_args[-1][0], "kind": "port", "seed": 99, "budget": 400})
    s_eval = cal["sec"] / cal["n_eval"]
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", "8")))

    def est(budget: int) -> float:
        runs_real = len(fold_args) * len(FITS) * len(seeds) + len(q_args) * len(seeds) + len(seeds)
        runs_noise = args.noise * len(fold_args) * len(seeds)
        ev = (runs_real + runs_noise) * budget + len(FLOW_SEEDS) * args.flow_budget + len(FITS) * len(fold_args) * 130 * 3
        return ev * s_eval / args.workers / 3600.0

    budget = args.budget or (6400 if est(6400) <= 8.0 else 3200)
    msg = (f"시간 추정: 평가당 {s_eval * 1e3:.1f} ms(단일 스레드, 차가운 캐시) → 예산 6,400 이면 {est(6400):.1f}시간, "
           f"3,200 이면 {est(3200):.1f}시간 (작업자 {args.workers}) → 예산 {budget:,}")
    est_msg = msg
    print(msg, flush=True)
    _log(msg, t0)

    led_path = os.path.join(out_dir, LABEL, "TRIALS.jsonl")
    seen = read_fingerprints(led_path)
    n_led_before = count_trials(LABEL, logs_dir=out_dir)
    hashes: set = set()

    def ledger_row(r: dict, tag: str) -> dict:
        return {"config": {"design": DESIGN_DOC, "run": tag, "budget": budget, "n_unique_configs_evaluated": r["uniq"],
                           "best": {k: (str(v) if v == math.inf else v) for k, v in r["best"].items()},
                           "train_fitness": r["fit"], "restarts": len(r["restarts"]),
                           "universe": "H5 ADV20≥5,000", "matched": g1.CTL_DESC},
                "note": "GA2 3차(돌파 실패 유전자) — 런 요약 한 줄(고유 config 수는 n_unique_configs_evaluated, 해시로 런 사이 중복 제거해 N 에 넣는다)"}

    # --- 작업 목록 ---
    tasks = []
    for kpos, (k, (tr, te)) in enumerate(zip(fold_ids, fold_args, strict=True)):
        for kind in FITS:
            for s in seeds:
                tasks.append({"id": ("wf", k, kind, s), "win": tr, "kind": kind, "seed": s, "budget": budget})
    for qi, (tr, te) in enumerate(q_args):
        for s in seeds:
            tasks.append({"id": ("q", qi, "port", s), "win": tr, "kind": "port", "seed": s, "budget": budget})
    for s in seeds:
        tasks.append({"id": ("live", 0, "port", s), "win": (LIVE_LO, LIVE_HI, None), "by_event": True, "kind": "port",
                      "seed": s, "budget": budget})
    ctx = mp.get_context("spawn")
    res: dict = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=worker_init,
                                initargs=(args.cache, 1)) as ex:
        futs = {ex.submit(cma_task, t): t for t in tasks}
        nfut = {ex.submit(noise_task, {"rep": r, "folds": fold_args, "seeds": seeds, "budget": budget}): r
                for r in range(args.noise)}
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            res[r["id"]] = r
            hashes.update(r["hashes"].tolist())
            write_ledger(led_path, [ledger_row(r, "/".join(map(str, r["id"])))], seen)
            if i % 10 == 0 or i == len(futs):
                _log(f"실제 런 {i}/{len(futs)} — 최근 {r['id']} 적합도 {r['fit']:+.4f} · 재시작 {len(r['restarts'])} · {r['sec']:.0f}s", t0)
        t_real = time.time() - t0
        # 2단계: 수급 시대(폴드 6 처리량 시드 최적 위) · 민감도 지형(폴드 중앙값 주변)
        last = fold_ids[-1]
        flow_futs = {ex.submit(cma_task, {"id": ("flow", 0, "port", s), "win": (FLOW_TR_LO, FLOW_TR_HI, FLOW_TR_HI),
                                          "kind": "port", "seed": s, "budget": args.flow_budget,
                                          "base": res[("wf", last, "port", s)]["best"]}): s for s in FLOW_SEEDS if s in seeds}
        land_futs = []
        for kind in FITS:
            for kpos, (k, (tr, te)) in enumerate(zip(fold_ids, fold_args, strict=True)):
                med = g2_median([res[("wf", k, kind, s)]["best"] for s in seeds])
                land_futs.append(ex.submit(landscape_task, {"kind": kind, "fold": k, "tr": tr, "te": te, "base": med}))

        # --- 부모: 실제 결과 후처리(작업자는 노이즈를 계속 돈다) ---
        E = _W["E"]
        set_noise(None)
        wins = [(Win(pre, *a), Win(pre, *b)) for a, b in fold_args]
        qwins = [(Win(pre, *a), Win(pre, *b)) for a, b in q_args]
        n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
        flow_rows, land_rows = {}, []
        for f in cf.as_completed(flow_futs):
            r = f.result()
            s = flow_futs[f]
            base = res[("wf", last, "port", s)]["best"]
            flow_rows[s] = {"base": base, "flow": r["best"], "fit": r["fit"],
                            "base_fit": fitness(E, "port", g2_spec(base), Win(pre, FLOW_TR_LO, FLOW_TR_HI, FLOW_TR_HI))[0]}
            hashes.update(r["hashes"].tolist())
        for f in cf.as_completed(land_futs):
            rows = f.result()
            land_rows += rows
            hashes.update(x["hash"] for x in rows)
        n_uniq = len(hashes)
        n_dsr = n_uniq + n_led_before + sum(n_prior.values())
        n_dsr_alt = sum(v for k_, v in n_prior.items() if k_ != "ga") + len(tasks)

        out = {"folds": {}, "q": {}}
        for kind in FITS:
            bests = [[res[("wf", k, kind, s)]["best"] for s in seeds] for k in fold_ids]
            ps = proc_stats(E, kind, wins, bests, bands=("H5", "유동", "중소형"), n_dsr=n_dsr)
            ps["seed"] = {}
            for s in seeds:
                cfgs = [res[("wf", k, kind, s)]["best"] for k in fold_ids]
                st = {b: stitched(E, wins, cfgs, band_sel={"H5": -1, "유동": 1, "중소형": 0}[b]) for b in ("H5", "유동", "중소형")}
                ins = [is_point(E, c, tr)["port"]["ann"] for c, (tr, _) in zip(cfgs, wins, strict=True)]
                ps["seed"][s] = {"st": st, "wfe": wfe([fp["ann"] for fp in st["H5"]["fold_port"]], ins)}
            ps["pbo"] = pbo_of(E, [b for bk in bests for b in bk])
            ps["bests"] = bests
            out["folds"][kind] = ps
            _log(f"{kind}: 이어 붙인 OOS(중앙값) {_pct(ps['dec']['ann'])} · WFE {_f(ps['dec']['wfe'])} · PBO {ps['pbo']['sharpe']['pbo']:.3f}", t0)
        qb = [[res[("q", qi, "port", s)]["best"] for s in seeds] for qi in range(len(q_args))]
        qps = proc_stats(E, "port", qwins, qb, bands=("H5", "유동", "중소형"), n_dsr=n_dsr)
        qps["pbo"] = pbo_of(E, [b for bk in qb for b in bk]) if len(qb) * len(seeds) >= 2 else None
        out["q"] = qps
        _log(f"분기 재적합: 이어 붙인 OOS {_pct(qps['dec']['ann'])} · WFE {_f(qps['dec']['wfe'])}", t0)
        # 진입 × 청산 바꿔 끼우기(처리량·트레이드당·1일당 각각, 시드 i 진입 × 시드 j 청산)
        swap = {}
        for kind in FITS:
            tab = np.full((len(seeds), len(seeds)), np.nan)
            tab_a = np.full((len(seeds), len(seeds)), np.nan)
            for i in seeds:
                for j in seeds:
                    cfgs = []
                    for bk in out["folds"][kind]["bests"]:
                        cfgs.append({**{g: bk[i][g] for g in ENTRY_GENES}, **{g: bk[j][g] for g in EXIT_GENES}})
                    r = stitched(E, wins, cfgs)
                    tab[i, j], tab_a[i, j] = r["port"]["ann"], r["trades"].get("alpha", np.nan)
            swap[kind] = (tab, tab_a)
        # 라이브 후보
        live_runs = {s: res[("live", 0, "port", s)] for s in seeds}
        live_med = g2_median([r["best"] for r in live_runs.values()])
        live_w = Win(pre, LIVE_LO, LIVE_HI, None, by_event=True)
        lv_fit, lv_n = fitness(E, "port", g2_spec(live_med), live_w)
        # DSR(정합 초과): 이어 붙인 OOS 월별 포트폴리오 초과 · 트레이드(초과 − 비용)
        from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
        dsr = {}
        for kind in FITS:
            mo = out["folds"][kind]["st"]["H5"]["port"]["monthly"]
            dsr[kind] = {"months": len(mo)}
            for tag, N in (("", n_dsr), ("_alt", n_dsr_alt)):
                d = deflated_sharpe_from_sample(mo, N) if len(mo) > 2 else None
                dsr[kind].update({f"sr{tag}": float(d["observed_sharpe"]) if d else np.nan,
                                  f"sr0{tag}": float(d["expected_max_sharpe_h0"]) if d else np.nan,
                                  f"prob{tag}": float(d["prob_deflated_sharpe"]) if d else np.nan})
            tr_ = out["folds"][kind]["st"]["H5"]["trades"]
            dsr[kind]["trade_prob"] = tr_.get("dsr", np.nan)
            dsr[kind]["trade_n"] = tr_.get("n", 0)
        # 노이즈
        noise = []
        for f in cf.as_completed(nfut):
            noise.append(f.result()["dec"])
            if len(noise) % 5 == 0:
                _log(f"노이즈 {len(noise)}/{len(nfut)}", t0)

    dec = dict(out["folds"]["port"]["dec"])
    dec["pbo"] = out["folds"]["port"]["pbo"]["sharpe"]["pbo"]
    dec["r1"] = dec["folds_pos"] >= 4
    dec["r2"] = bool(np.isfinite(dec["ci_lo"]) and dec["ci_lo"] > 0)
    dec["r3"] = bool(np.isfinite(dec["wfe"]) and dec["wfe"] > 0.3)
    dec["r4"] = bool(dec["pbo"] < 0.5)
    dec["all"] = dec["r1"] and dec["r2"] and dec["r3"] and dec["r4"]
    nz = pd.DataFrame(noise)

    def pctile(col: str, val: float) -> float:
        v = nz[col].astype(float).dropna() if len(nz) else pd.Series(dtype=float)
        return float((v < val).mean() + 0.5 * (v == val).mean()) if len(v) and np.isfinite(val) else np.nan

    noise_cmp = {c: {"real": dec.get(c), "pct": pctile(c, dec.get(c, np.nan)) if c in nz else np.nan,
                     "q05": float(nz[c].quantile(0.05)) if c in nz else np.nan,
                     "q50": float(nz[c].median()) if c in nz else np.nan,
                     "q95": float(nz[c].quantile(0.95)) if c in nz else np.nan}
                 for c in ("folds_pos", "ci_lo", "ann", "wfe", "pbo", "is_mean", "calm_ann")}
    noise_pass = {k: float(nz[k].mean()) if k in nz else np.nan for k in ("r1", "r2", "r3", "r4", "all")}

    live = {"generated": date.today().isoformat(), "design": DESIGN_DOC, "fitness": "port (처리량 — 자본 제약 포트폴리오)",
            "window": {"events": [LIVE_LO, LIVE_HI], "cut": "없음 — 데이터 끝 미청산은 마지막 종가(data_end)",
                       "panel_end": str(pre.days[-1].date())},
            "budget_per_seed": budget,
            "config": {k: (None if v is None else ("inf" if v == math.inf else v)) for k, v in live_med.items()},
            "config_rule": f"시드 {len(seeds)}개 최적의 유전자 중앙값(연속 = 중앙값, 이산 = 단계 인덱스 중앙값)",
            "seed_bests": {str(s): {k: (None if v is None else ("inf" if v == math.inf else v)) for k, v in r["best"].items()}
                           for s, r in live_runs.items()},
            "seed_fitness": {str(s): r["fit"] for s, r in live_runs.items()},
            "median_fitness_train": lv_fit, "median_orders_train": lv_n,
            "note": ("이 config 자체의 표본 밖 성적은 잴 수 없다. 신뢰 근거는 같은 절차(폴드별 CMA-ES → 시드 중앙값)의 이어 붙인 워크포워드 "
                     "OOS 성적과 그 노이즈 기준선 대비 백분위다(GA2_RESULTS.md). 판단 규칙(GA2.md) 산술: " + json.dumps(
                         {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in dec.items()}, default=float)
                     + ". 탐색 산출물 — 판정 아님. 전진 페이퍼 전에는 실주문에 쓰지 않는다."),
            "genes_off_meaning": "None = 그 조건 끔(연속 유전자 끝 2% 구역), inf = 없음(추격 금지 없음/익절 없음), "
                                 "0 = 없음(돌파·추적·정렬·기울기·돌파 실패)"}
    with open(os.path.join(out_dir, "live_candidate.json"), "w", encoding="utf-8") as fh:
        json.dump(live, fh, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    land = pd.DataFrame(land_rows).drop(columns=["hash"], errors="ignore")
    land.to_csv(os.path.join(out_dir, LABEL, "landscape.csv"), index=False)
    rows = [{"run": "/".join(map(str, k)), "train_fit": r["fit"], "train_n": r["n"], "uniq": r["uniq"], "sec": r["sec"],
             "restarts": len(r["restarts"]), "why": ",".join(x["why"] for x in r["restarts"]),
             **{g: r["best"].get(g) for g in GENE_ORDER}} for k, r in res.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, LABEL, "runs.csv"), index=False)
    nz.to_csv(os.path.join(out_dir, LABEL, "noise.csv"), index=False)
    L = render(locals())
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n=== GA2 — {elapsed / 60:.1f}분 · 고유 config {n_uniq:,} · DSR N {n_dsr:,} · 예산 {budget:,} ===")
    print(f"판단 규칙(처리량·중앙값): {dec}")
    print(f"노이즈 통과율: {noise_pass}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _pp(d: dict) -> str:
    if not d or not np.isfinite(d.get("ann", np.nan)):
        return "nan"
    return f"{_pct(d['ann'])} [{_pct(d.get('lo'))}, {_pct(d.get('hi'))}] · MDD {_pct(-d['mdd'])}"


def _tt(s: dict) -> str:
    if not s or not s.get("n"):
        return "n 0"
    return (f"n {s['n']} · 손절 {s['stop_rate']:.0%} · 이익/손실 {_f(s['gl'])} · 순 {_pct(s['net'])} · α {_pct(s['alpha'])} "
            f"[{_pct(s['alpha_lo'])}, {_pct(s['alpha_hi'])}] · α/일 {_f(s['alpha_day'] * 1e4, '{:+.1f}')}bp")


def _ppt(d: dict) -> str:
    return f"{_pct(d['ann'])} [{_pct(d.get('lo'))}, {_pct(d.get('hi'))}]" if d and np.isfinite(d.get("ann", np.nan)) else "nan"


def render(v: dict) -> list[str]:
    pre, out, res, dec = v["pre"], v["out"], v["res"], v["dec"]
    folds, fold_ids, seeds, nz = v["folds"], v["fold_ids"], v["seeds"], v["nz"]
    L = [f"# GA2 — 연속·다단계 유전자 × 진입·청산, CMA-ES × 롤링 워크포워드 — 실행 결과 {date.today().isoformat()} "
         "(러너 자동 생성, 탐색 — 판정 아님)", "",
         "설계 `GA2.md`(421b01a → … → 162103a → 7e51eb4 정정 → e245076 돌파 실패 유전자, 전부 결과 보기 전)와 코디네이터 확대 지시"
         "(시드 5 · 예산 · 분기 재적합 · 노이즈 기준선)를 계산한 숫자다. 판단 규칙 행은 문턱과의 산술 비교이고 판정·후보 선정은 사람이 한다. "
         "**표본 밖(2020~2026)은 의사-OOS다** — 유전자 목록·범위·청산 부품·유니버스·대조 층은 이 계열의 전 구간 결과를 본 뒤 정해졌다. "
         "워크포워드는 문턱 선택의 과적합만 막는다. 신뢰 근거는 전진 페이퍼뿐이다.", "",
         f"재현: `NUMBA_NUM_THREADS=8 uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/ga2_run.py "
         f"--cache <npy 디렉터리> --workers {v['args'].workers}` (numba {gf.numba.__version__ if gf.HAVE_NUMBA else '없음'}, cmaes 0.13.1).", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 출력·원장 `{v['out_dir']}`. 측정으로 쓰지 않는다.", ""]
    nc, npass = v["noise_cmp"], v["noise_pass"]
    L += ["## 판단 규칙 숫자 — 처리량 적합도 · 시드 유전자 중앙값 config · 연 폴드 (GA2.md, 산술 비교) + 노이즈 기준선", "",
          f"노이즈 = 진입 피처를 같은 신호월 안에서 무작위 재배정한 데이터에 **같은 절차 전체**(폴드 {len(folds)} × 시드 {len(seeds)} × 예산 {v['budget']:,} → 중앙값 → "
          f"이어 붙인 OOS·WFE·PBO)를 {len(nz)}번. 백분위 = 실제 값보다 작은 노이즈 비율(동률 ½).", "",
          "| 조건 | 문턱 | 실제 | 산술 | 노이즈 중앙값 [5%, 95%] | 실제의 노이즈 백분위 | 노이즈 통과율 |", "|---|---|---|---|---|---|---|",
          f"| 폴드별 OOS 포트폴리오 초과 > 0 | 6 중 ≥ 4 | {dec['folds_pos']}/{dec['n_folds']} | {'✓' if dec['r1'] else '✗'} | "
          f"{_f(nc['folds_pos']['q50'], '{:.0f}')} [{_f(nc['folds_pos']['q05'], '{:.0f}')}, {_f(nc['folds_pos']['q95'], '{:.0f}')}] | "
          f"{_f(nc['folds_pos']['pct'], '{:.0%}')} | {_f(npass['r1'], '{:.0%}')} |",
          f"| 이어 붙인 OOS 연환산 초과 CI 하단 | > 0 | {_pct(dec['ci_lo'])} | {'✓' if dec['r2'] else '✗'} | "
          f"{_pct(nc['ci_lo']['q50'])} [{_pct(nc['ci_lo']['q05'])}, {_pct(nc['ci_lo']['q95'])}] | {_f(nc['ci_lo']['pct'], '{:.0%}')} | "
          f"{_f(npass['r2'], '{:.0%}')} |",
          f"| 워크포워드 효율 Σ OOS / Σ IS | > 0.3 (IS 합 > 0) | {_f(dec['wfe'])} (IS 합 {_pct(dec['is_sum'])}) | {'✓' if dec['r3'] else '✗'} | "
          f"{_f(nc['wfe']['q50'])} [{_f(nc['wfe']['q05'])}, {_f(nc['wfe']['q95'])}] | {_f(nc['wfe']['pct'], '{:.0%}')} | {_f(npass['r3'], '{:.0%}')} |",
          f"| PBO(폴드 × 시드 최적, 전 구간 월별, Sharpe) | < 0.5 | {dec['pbo']:.3f} | {'✓' if dec['r4'] else '✗'} | "
          f"{_f(nc['pbo']['q50'], '{:.3f}')} [{_f(nc['pbo']['q05'], '{:.3f}')}, {_f(nc['pbo']['q95'], '{:.3f}')}] | "
          f"{_f(nc['pbo']['pct'], '{:.0%}')} | {_f(npass['r4'], '{:.0%}')} |",
          f"| 넷 다 | | | {'✓' if dec['all'] else '✗'} | | | {_f(npass['all'], '{:.0%}')} |",
          f"| (참고) 이어 붙인 OOS 연환산 초과 | | {_pct(dec['ann'])} | | {_pct(nc['ann']['q50'])} [{_pct(nc['ann']['q05'])}, {_pct(nc['ann']['q95'])}] | "
          f"{_f(nc['ann']['pct'], '{:.0%}')} | |",
          f"| (참고) 학습창 IS 연환산 평균 | | {_pct(dec['is_mean'])} | | {_pct(nc['is_mean']['q50'])} [{_pct(nc['is_mean']['q05'])}, "
          f"{_pct(nc['is_mean']['q95'])}] | {_f(nc['is_mean']['pct'], '{:.0%}')} | |",
          f"| (참고) 2022·2023·2024 조각만 | | {_pct(dec['calm_ann'])} | | {_pct(nc['calm_ann']['q50'])} | {_f(nc['calm_ann']['pct'], '{:.0%}')} | |",
          "", f"라이브 후보(`live_candidate.json`, 사건 {LIVE_LO} ~ {LIVE_HI}, 처리량, 시드 {len(seeds)} 중앙값): `{gene_str(v['live_med'])}` — "
          f"학습 적합도 {v['lv_fit']:+.4f} (주문 {v['lv_n']})", "",
          "**실적(EPS) 피처 커버리지:** " + (lambda e: (f"실적 버전 행 중 분기말·전년 분기말 주식 수가 둘 다 있는 비율 {e['share_rows_pit']:.1%} · "
                                                 f"후보 쌍 중 실적 행이 있는 비율 {e['pairs_with_earn']:.1%} · 그중 주식 수로 EPS 를 낸 쌍 "
                                                 f"{e['pairs_eps_from_shares']:.1%} (나머지는 순이익 YoY 대체, `eps_ni`)"))(pre.info.get("eps", {
              "share_rows_pit": np.nan, "pairs_with_earn": np.nan, "pairs_eps_from_shares": np.nan})), ""]

    # 이어 붙인 OOS 요약
    L += ["## 이어 붙인 OOS — 재적합 주기(연 폴드 vs 분기) × 적합도", "",
          "하나의 연속 포트폴리오(슬롯 10, 진입 비중 10% 고정) — 창 k 의 진입은 창 k config, 창 경계(다음 시험 시작 직전 종가)에서 열린 포지션을 시가평가로 닫는다. "
          "연환산 = 월 초과 평균 × 12, CI = 월 3개월 블록 부트스트랩 2,000, MDD = 단리 누적 초과(초기 자산 대비).", "",
          "| 주기 · 적합도 · config | 띠 | 연환산 초과 [95% CI] | MDD | 진입(주문) · 짝 없음 | 회전율/년 · 슬롯 사용률 | 2022~24 조각 | 겹침 제거 트레이드 | WFE |",
          "|---|---|---|---|---|---|---|---|---|"]

    def st_rows(label, st, wfe_v):
        for b in ("H5", "유동", "중소형"):
            if b not in st:
                continue
            x = st[b]
            L.append(f"| {label} | {b} | {_ppt(x['port'])} | {_pct(-x['port']['mdd'])} | {x['port']['trades']} ({x['port']['orders']}) · "
                     f"{x['port']['unmatched']} | {_f(x['port']['turnover'])} · {x['port']['util']:.0%} | {_pct(x['calm_ann'])} | "
                     f"{_tt(x['trades'])} | {_f(wfe_v) if b == 'H5' else ''} |")

    for kind in FITS:
        ps = out["folds"][kind]
        st_rows(f"연 · {kind} · 중앙값", ps["st"], ps["dec"]["wfe"])
    st_rows("분기 · port · 중앙값", out["q"]["st"], out["q"]["dec"]["wfe"])
    for kind in FITS:
        for s in seeds:
            sd = out["folds"][kind]["seed"][s]
            st_rows(f"연 · {kind} · 시드 {s}", {"H5": sd["st"]["H5"]}, sd["wfe"])
    L.append("")
    qd = out["q"]["dec"]
    L += [f"분기 재적합(처리량, 학습 = 직전 3년, 시험 = 다음 분기, {qd['n_folds']}분기): 양수 분기 {qd['folds_pos']}/{qd['n_folds']} · "
          f"CI 하단 {_pct(qd['ci_lo'])} · WFE {_f(qd['wfe'])} · PBO "
          f"{_f(out['q']['pbo']['sharpe']['pbo'] if out['q']['pbo'] else np.nan, '{:.3f}')} (분기 × 시드 최적 후보). "
          "분기 재적합은 처리량 적합도만 돌렸다(판단 규칙 적합도 · 계산량).", ""]

    for kind in FITS:
        ps = out["folds"][kind]
        L += [f"## 적합도: {FIT_NAMES[kind]} — 연 폴드", "",
              "| 폴드 | 학습 → 시험 | 중앙값 config | 학습 적합도 | IS 연환산 (진입·주문) | IS 트레이드 | OOS 조각 연환산 | OOS 트레이드 | 시드별 OOS 조각 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, f in enumerate(folds):
            isp = ps["is"][k]
            fp = ps["st"]["H5"]["fold_port"][k]
            ft = ps["st"]["H5"]["fold_trades"][k]
            sd = " · ".join(_pct(ps["seed"][s]["st"]["H5"]["fold_port"][k]["ann"]) for s in seeds)
            L.append(f"| {fold_ids[k]} | {f.train_lo[:4]}~{int(f.train_hi[:4]) - 1} → {f.test_lo[:4]} | `{gene_str(ps['meds'][k])}` | "
                     f"{ps['fit'][k]:+.4f} | {_pct(isp['port']['ann'])} ({isp['port']['n']}·{isp['port']['orders']}) | {_tt(isp['trades'])} | "
                     f"{_pct(fp['ann'])} | {_tt(ft)} | {sd} |")
        pb, dd = ps["pbo"], v["dsr"][kind]
        fa = sum(1 for k_, r in res.items() if k_[0] == "wf" and k_[2] == kind and r["fit"] < -0.5)
        L += ["", f"- PBO(폴드 × 시드 최적 {pb['sharpe']['n_candidates']}개, 2017-10 ~ 2026-04 월별 포트폴리오 초과, 조각 16): Sharpe **{pb['sharpe']['pbo']:.3f}** · 평균 {pb['mean']['pbo']:.3f}",
              f"- DSR(정합 초과): 이어 붙인 OOS 월별 포트폴리오 초과 {dd['months']}개월 — 관측 Sharpe(월) {_f(dd['sr'], '{:.3f}')} · P **{_f(dd['prob'], '{:.1%}')}** "
              f"(N {v['n_dsr']:,}) · P {_f(dd['prob_alt'], '{:.1%}')} (참고 N {v['n_dsr_alt']:,}); 트레이드 {dd['trade_n']}건(정합 초과 − 비용) P {_f(dd['trade_prob'], '{:.1%}')} (N {v['n_dsr']:,})",
              f"- 최적이 거래수 하한 페널티 구역(적합도 < −0.5)인 폴드 런: {fa}/{len(folds) * len(seeds)}", ""]
        L += ["### 유전자 — 폴드별 시드 중앙값 · 시드 최적 중 끔 비율 · 시드 간 폭", "",
              "| 유전자 | " + " | ".join(f"폴드 {fold_ids[k]}" for k in range(len(folds))) + " | 끔 비율 | 시드 간 폭(범위 대비, 폴드 평균) |",
              "|---|" + "---|" * (len(folds) + 2)]
        for gi, name in enumerate(GENE_ORDER):
            vals = []
            for k in range(len(folds)):
                m = ps["meds"][k][name]
                vals.append("끔" if is_off(name, m) else (f"{m:g}" if not isinstance(m, float) else f"{m:.2f}"))
            allb = [b for bk in ps["bests"] for b in bk]
            off = np.mean([is_off(name, b[name]) for b in allb])
            spreads = []
            for bk in ps["bests"]:
                xs = np.array([g2_encode(b)[gi] for b in bk])
                rng = 1.0 if gi < len(G2_CONT) else DISC_SIZES[gi - len(G2_CONT)] - 1
                spreads.append((xs.max() - xs.min()) / rng)
            L.append(f"| {name} | " + " | ".join(vals) + f" | {off:.0%} | {np.mean(spreads):.0%} |")
        tab, tab_a = v["swap"][kind]
        L += ["", "### 진입 × 청산 바꿔 끼우기 (행 = 진입 유전자 시드, 열 = 청산 유전자 시드; 이어 붙인 OOS 포트폴리오 연환산 · 트레이드당 α)", "",
              "| 진입 \\ 청산 | " + " | ".join(f"시드 {s}" for s in seeds) + " |", "|---|" + "---|" * len(seeds)]
        for i in seeds:
            L.append(f"| 시드 {i} | " + " | ".join(f"{_pct(tab[i, j])} · {_pct(tab_a[i, j])}" for j in seeds) + " |")
        L.append("")
        land = v["land"]
        if len(land):
            lk = land[land["fitness"] == kind]
            L += ["### 민감도 지형 — 폴드 중앙값 config 에서 한 유전자씩 범위 전체 (폴드 평균: 학습 적합도 / 시험 연도 포트폴리오 연환산)", "",
                  "| 유전자 | 격자 값 → 학습 적합도 / OOS 연환산 |", "|---|---|"]
            for name in GENE_ORDER:
                g = lk[lk["gene"] == name].groupby("grid").agg(fit=("train_fit", "mean"), oos=("oos_port_ann", "mean"),
                                                                val=("value", "first"))
                cells = " · ".join(f"{('끔' if r_['val'] in ('None', 'inf') else r_['val'])}: {r_['fit']:+.3f}/{_pct(r_['oos'])}"
                                   for _, r_ in g.iterrows())
                L.append(f"| {name} | {cells} |")
            L += ["", "(폴드별 값: `ga2/landscape.csv`. 연속 유전자 격자 = [0,1] 11점, 끔 끝은 '끔'. 이산 0 = 없음.)", ""]

    # 분기 유전자 요약
    qps = out["q"]
    L += ["## 분기 재적합 — 분기별 중앙값 config 의 유전자 분포", "",
          "| 유전자 | 분기 중앙값의 끔 비율 | 값 분포(중앙값 [최소, 최대], 끔 제외) |", "|---|---|---|"]
    for name in GENE_ORDER:
        vals = [m[name] for m in qps["meds"]]
        on = [float(x) for x in vals if not is_off(name, x) and x is not None and x != math.inf]
        L.append(f"| {name} | {np.mean([is_off(name, x) for x in vals]):.0%} | "
                 + (f"{np.median(on):g} [{min(on):g}, {max(on):g}]" if on else "—") + " |")
    L.append("")

    L += [f"## 수급 시대 미니 탐색 (처리량, 폴드 6 처리량 시드 0~2 최적 위, 학습 진입 {FLOW_TR_LO} ~ 2025-06-30 · 2025-06-30 종가 시가평가, "
          "OOS 진입 2025-07-01 ~ 2026-04-01)", "",
          f"OOS 창은 감사 이력상 여러 번 읽힌 창이다(설계에 적힌 대로). 처리량 하한(주문 ≥ {PORT_FLOOR}). 예산 {v['args'].flow_budget} · 시드 {len(FLOW_SEEDS)}(설계 그대로).", "",
          "| 시드 | 기본 config | 수급 유전자 | 학습 적합도(기본 → 수급) | OOS 포트폴리오(기본) | OOS 포트폴리오(+수급) | OOS 트레이드(+수급) |",
          "|---|---|---|---|---|---|---|"]
    E = v["E"]
    te = Win(pre, FLOW_TR_HI, None, EVENT_HI_NEXT)
    for s, r in sorted(v["flow_rows"].items()):
        spec = g2_spec(r["base"])
        d0, _, _, _ = E.portfolio(spec, te.i0, te.i1, te.d_lo, te.end, te.cut)
        ex = flow_extra(pre, r["flow"])
        d1, _, _, _ = E.portfolio(spec, te.i0, te.i1, te.d_lo, te.end, te.cut, extra=ex)
        sel = E.select(spec, te.i0, te.i1, extra=ex)
        L.append(f"| {s} | `{gene_str(r['base'])}` | {r['flow']} | {r['base_fit']:+.4f} → {r['fit']:+.4f} | "
                 f"{_ppt(port_point(E, d0, te.d_lo, te.d_first, te.end))} | {_ppt(port_point(E, d1, te.d_lo, te.d_first, te.end))} | "
                 f"{_tt(g1.trade_report(E, spec, sel, cut=te.cut))} |")
    L.append("")

    runs = [r for k, r in res.items() if k[0] in ("wf", "q", "live")]
    nre = pd.Series([len(r["restarts"]) for r in runs])
    why = pd.Series([x["why"] for r in runs for x in r["restarts"]]).value_counts().to_dict()
    reg = pd.Series([x["regime"] for r in runs for x in r["restarts"]]).value_counts().to_dict()
    curves = pd.DataFrame([r["curve"] for k, r in res.items() if k[0] == "wf" and k[2] == "port"])
    L += ["## 최적화기 · 시간 · 자원", "",
          f"- {v['est_msg']}",
          f"- 실제 런: 연 폴드 {len(folds)} × 적합도 3 × 시드 {len(seeds)} = {len(folds) * 3 * len(seeds)} · 분기 {len(v['q_args'])} × 시드 {len(seeds)} = "
          f"{len(v['q_args']) * len(seeds)} · 라이브 후보 {len(seeds)} · 수급 {len(FLOW_SEEDS)} × {v['args'].flow_budget} · 민감도 지형 {len(v['land']):,} 점 · "
          f"노이즈 {len(nz)} × {len(folds) * len(seeds)} 런. 작업자 {v['args'].workers}(spawn, 사전계산 .npy mmap 공유, 작업자당 numba 1 스레드).",
          f"- 런당 시간 중앙값 {np.median([r['sec'] for r in runs]):.0f}s · 실제 런 단계 {v['t_real'] / 60:.0f}분 · 전체 {v['elapsed'] / 60:.0f}분.",
          f"- BIPOP: 런마다 CMAwM 런 수 분포 {nre.value_counts().sort_index().to_dict()} (평균 {nre.mean():.1f}) · 체제 {reg} · 끝난 이유 {why}. "
          "정체 재시작 = 그 CMAwM 런 최고값이 10 + ⌈30·19/λ⌉ 세대 동안 개선 없음(λ = 12 → 58세대 = 696 평가).",
          "- 수렴(연 폴드 처리량 런, 최고 적합도의 런 중앙값): " + " · ".join(f"{c}평가 {curves[c].median():+.4f}" for c in curves.columns) if len(curves) else "",
          "- 단순 GA(GA.md) vs CMA-ES 의 같은 공간·같은 예산 비교는 `GA_RESULTS.md` 마지막 표(6유전자·고정 TRAIN — 검증 구조가 달라 GA2 와 직접 비교 아님).",
          f"- DSR N = 이번 실행 고유 config {v['n_uniq']:,}(런 사이 해시 중복 제거, 노이즈 제외, 민감도 지형 포함) + ga2 원장 기존 행 {v['n_led_before']:,}"
          f"(중단된 1·2·3차 실행의 config 행 — 시간 손절 유전자 세대) + " + " + ".join(f"{k} {n:,}" for k, n in v["n_prior"].items())
          + f" = **{v['n_dsr']:,}** · 참고 N = 이산 칸 원장(ga 제외) + 이번 CMA-ES 런 수 = {v['n_dsr_alt']:,}", "",
          "## 실행 이력 (원장 `ga2/TRIALS.jsonl`, 덧붙이기 전용)", "",
          "- 1차(03:56, 시간 손절 유전자·purge·하한 150) — 라이브 후보창 버그(포트폴리오 날 상한)로 폴드 0 몇 런 뒤 중단, 그때 원장을 지웠다(사본 없음 — 규칙 위반을 여기 적는다). "
          "본 것: 로그의 학습 적합도 몇 줄(OOS 없음).",
          "- 2차(같은 판) — 리뷰 K-C1(purge 가 긴 보유 승자를 버림)로 폴드 0~2 에서 중단. 본 것: 로그의 학습 적합도·유전자 줄(OOS 없음). 원장 행 42,900 은 남아 N 에 들어간다.",
          "- 3차(시가평가 절단·처리량 주문 하한·짝 없는 포지션 제외·경계 시가평가·정합 초과 DSR·정체 재시작) — 사용자의 청산 유전자 변경(시간 손절 → 돌파 실패)으로 "
          "폴드 0~4 학습 단계에서 중단(이어 붙인 OOS 계산 전). 본 것: 로그의 학습 적합도 줄. 그 원장 행도 남는다.",
          "- 4차(이 문서) = 돌파 실패 유전자 · 시드 5 · 분기 재적합 · 노이즈 기준선. 원장에는 **런마다 요약 한 줄**(최적·평가한 고유 config 수)을 덧붙였다 — "
          "런당 수천 개 config 를 한 줄씩 쓰면 원장이 수백 MB 가 된다. 고유 config 수는 런 사이 해시로 중복 제거해 N 에 넣었다.", ""]

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          f"- **끔 구역:** 연속 유전자는 끔 끝에서 범위의 {OFF_ZONE:.0%} 안이면 조건을 끈다(NaN 도 통과). 그 밖은 엄격(NaN 은 통과 안 함) — 끔 끝에 NaN 절벽이 있다(리뷰 K-M4).",
          "- **ma_stack:** close > MA50, MA50 > MA150, MA150 > MA200 세 부등식 중 **참인 개수 ≥ k**. ma200_slope_days k: MA200 > MA200[t−k].",
          "- **near_high/above_low:** 장중 고·저 252일 최대·최소(`trend_template` 과 같은 창, 당일 포함).",
          "- **돌파 없음(창 0):** 거래량 배수의 평균 창·추격 금지 피벗·돌파 실패 피벗 = 50일(뼈대 기본). 돌파 있음이면 셋 다 그 창.",
          "- **실적 유전자(GA2.md 마지막 절):** EPS = 순이익 / 분기말 주식 수(`shares_outstanding_history` as-of 분기말·전년 분기말, 분할 조정계수 = 조정 종가/원 종가로 "
          "나눠 같은 척도). 주식 수가 없으면 순이익 YoY 로 대체(`eps_ni`). 버전·가용일·직전 분기는 `earnings_leg_panel` 그대로(순이익 자리에 EPS). "
          "eps_accel = EPS YoY > 직전 분기 EPS YoY(둘 다 유한). rev_yoy = 같은 행 매출 YoY. 실적 행 없음 = NaN → 켠 조건 통과 못 함. 주식 수 행은 그 날짜에 알려졌다고 본다"
          "(분기 보고서와 함께 공시 — 실적 가용일 이후에만 쓰인다). `shares_outstanding_history` 는 폐지 종목이 0건이라 폐지 종목은 전부 순이익 대체다.",
          "- **돌파 실패(failed_breakout, 시간 손절 대체):** `minervini_exits.simulate_minervini(failed_breakout=True)` 규칙 12 그대로 — 익절 전 1 ~ max_hold−1 일째 "
          "종가 < 피벗(그 config 창의 직전 최고 고가) 인 첫날(그날 장중 손절·익절보다 앞선 날만) → 다음 거래 가능일 시가 전량(패리티 테스트). "
          "대조 종목에도 **같은 프로토콜**(대조 자신의 그 창 피벗) — 돌파하지 않은 대조는 대개 첫날 종가가 피벗 아래라 다음날 시가에 나간다.",
          "- **익절 없음(∞) ∧ 추적 MA:** 원전 규칙 8 의 추적을 전량에 건다(`sepa` MA50 청산 식, 패리티 테스트). 돌파 실패와 겹치면 청산일이 이른 쪽, 같은 날이면 돌파 실패.",
          "- **대조:** 피벗 창 × 추격 단계마다 같은 층 풀에서 쌍별 해시 추출(쌍 번호) — 대조 수익은 그 청산 조합으로 다시 읽고 트레이드와 같은 날 시가평가로 끊는다.",
          "- **포트폴리오 회계:** 진입 비중 10% 고정(복리·재조정 없음), 날별 = 남은 주식 × Δ가격 / 진입가 — 날별 합 = Σ 10% × (수익 − 대조 − 비용)(테스트). "
          "주문은 신호일 종가에 빈 슬롯 수만큼 rel 오름차순 → RS 내림차순, 미체결이면 그 슬롯은 빈다. 수축 저점 없음·다음날 없음은 주문하지 않는다. "
          "**짝 없는 포지션은 손익·비용 모두 0**(슬롯은 차지).",
          "- **학습창 끝 시가평가(purge 대체, 리뷰 K-C1):** 진입 ∈ [train_lo, train_hi), test_lo 를 넘는 트레이드·대조는 test_lo 직전 거래일 종가에서 끊는다. 시험 쪽도 창 경계에서 같은 규칙.",
          f"- **거래수 하한:** 트레이드당·1일당 = 짝 있는 학습 트레이드 ≥ 150. 처리량 = 슬롯을 받은 주문 수 ≥ {PORT_FLOOR}.",
          "- **IS 연환산(WFE 분모):** 학습창 포트폴리오 점추정(월 평균 × 12, 첫 사건일부터). WFE = Σ OOS / Σ IS, IS 합 ≤ 0 이면 NaN.",
          "- **판단 규칙 대상:** 처리량 적합도 · 시드 유전자 중앙값 config · 연 폴드(라이브 후보와 같은 절차). 시드별·다른 적합도·분기 숫자는 참고.",
          "- **분기 재적합(코디네이터 지시):** 2020Q1 ~ 2026Q1 각 분기 = 직전 3년 학습(분기 시작 직전 종가 시가평가) → 그 분기 진입에 적용 → 다음 분기 시작 직전 종가에서 닫음. 처리량만.",
          "- **노이즈 기준선:** 진입 피처(tt·rs·breadth·pead·rel·돌파 비트·거래량·평균·nh·al·정렬·기울기)를 같은 신호월 안에서 쌍끼리 재배정(seed 1000+r) — 가격·체결·층·대조·우선순위 rel/RS 도 섞인 피처를 쓴다. "
          "연 폴드 · 처리량 · 시드 전부 · 같은 예산. 노이즈 런은 원장에 넣지 않는다.",
          "- **PBO:** 설계 문구대로 폴드 × 시드 최적을 후보로 전 구간 CSCV — 폴드 k 최적은 자기 학습 연도에 유리해 이 정의는 '어느 해에 학습됐나' 를 섞는다(리뷰 D-M1). 노이즈도 같은 정의.",
          "- **라이브 후보창:** 사건 2023-04-01 ~ 2026-04-01, 절단 없음(데이터 끝 미청산은 시뮬레이터가 마지막 종가로 닫는다 — `data_end`).",
          f"- 가격: `relax_run.KrxPatch` 보정판(2026-09-14+ 분봉 정규장) — {pre.info.get('nxt')}", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
