#!/usr/bin/env python
"""메타 세대 1 — 개인 트레이더 절차 비교: 국면 게이트·노출 조절 × 승자 보유 청산 × 단순 모멘텀. **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/META_GEN1.md`` — 466ddc1(결과 보기 전 고정) + 737624a "정정 — 설계 리뷰 반영"(``REVIEW_META_GEN1.md`` 1단계,
결과 보기 전). 이 러너는 숫자를 ``META_GEN1_RESULTS.md`` 에 적고 두 층 판단 규칙·리뷰어 페이퍼 추천 규칙의 문턱과 산술 비교만 한다.

**절차(18개 = 기본 3 + 국면 게이트 3 × 점수 3 + 노출 조절 2 × 점수 3). 주 점수 = RA, RS = 기준선, RB = 기술(정정 M2):**

- ``P1`` = 라이브 기본 ``StrategyConfig`` 수급 제외판(GA2 유전자 19개 → ``ga2_run.g2_spec`` 마스크), 현 청산(수렴 저점 손절 3~10% · 2R 절반 ·
  돌파 실패 · MA50 · 최대 120일). ``P1h`` = 같은 진입, 청산 = 본전 이동 없음 · 3R × 1/3 · 돌파 실패 허용 2% · 2일 · MA50(설계 표 그대로 — 사후 가설).
  청산은 :mod:`research.signals.exits_v2_fast` (``P1`` 칸은 옛 커널과 비트 일치), 체결·대조 번호는 ``ga_run.Engine.fill_ctl``.
- ``P2`` = 매월 마지막 거래일 종가(신호일)에 추세 템플릿 통과 종목 중 12-1 모멘텀 상위 10 → 다음날 시가 진입, 개별 손절 −15%, 다음 리밸런스에서
  상위 10 밖이면 그 다음날 시가 매도, 안이면 계속 보유(:func:`research.signals.meta_fast.p2_trades`). 체결(정정 M4): 상한가 시가 진입 불가, 하한가로
  잠긴 날은 손절·매도 불가(다음 잠기지 않은 거래일 시가), 조정 일봉 |수익| > 31% 는 그 종목 그날 수익 0 (``meta_fast.clean_jumps``).
- ``·G`` = 국면 점수 < 문턱이면 신규 진입 없음(P2 는 그 달 현금). ``·S`` = 슬롯 수 = round(10 × clip(a + b·점수, 0, 1)).
  문턱 {없음, 20, 40, 60 백분위}·(a, b) ∈ {0, .25, .5} × {.25, .5, 1} 은 폴드 학습창 **절대 포트폴리오 월 수익의 블록 부트스트랩 Calmar 하단**
  (3개월 블록 500회, 2.5%)으로 고른다(정정 C1 — 주문 하한 없음).

**포트폴리오:** 10슬롯 × 10%, 일 1회 신호(신호일 종가 → 다음날 시가), 비용 ``korea.cost_rt`` (청산일, 2배 스트레스 병기). 절대수익(복리 — 명목 =
신호일 자산의 10%)과 GA2 정합 초과 두 시계열(``meta_fast.portfolio_meta``). 학습창 끝·폴드 경계 시가평가는 GA2 와 같다.

**검증:** ``FOLDS`` 6개, 이어 붙인 OOS 2020-01 ~ 2026-04-01(+ 2020~24 · 2021~24 · 2025 제외). 기준선: 유니버스 등가중(월 재조정 = 주, 매일),
재구축 거래대금 가중 지수 대용, 2024-01-26+ 공식 지수. 국면 변형은 같은 평균 노출의 정적 축소(슬롯 수 고정)와 짝지은 블록 부트스트랩으로 비교.
널: 국면 변형 = 같은 달 순열을 모든 피처·시장에(재적합 포함) 300 + 무작위 오프셋 순환 300; ``P1``·``P1h`` = GA2 무작위 진입 300(활동 ≥ 100);
``P2`` = 무작위 추세 템플릿 10종목 300. 널 뽑기에도 층 2 규칙(국면 조건 제외)을 걸어 통과율을 적는다.

Run:  NUMBA_NUM_THREADS=8 uv run --with numba --with matplotlib --extra pg python research/experiments/meta_gen1_run.py \\
          --cache <pre_v2 npy 디렉터리> --factors <review_ga2/factors.pkl> --workers 12
      (DB 는 실행 용량 표(09:00 분봉)에만 읽기 전용으로 쓴다 — ``KR_QUANT_DB``.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import ga2_post as gp  # noqa: E402
import ga2_run as g2  # noqa: E402
import ga_run as g1  # noqa: E402

from research.signals import exits_v2_fast as xv  # noqa: E402
from research.signals import ga_fast as gf  # noqa: E402
from research.signals import meta_fast as mf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
SUB = "meta_gen1"
RESULTS_MD = "META_GEN1_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/META_GEN1.md (466ddc1 + 정정 737624a)"
REGIME_NPZ = "research/logs/sepa_explore/regime/regime_panel.npz"
MOOD_CSV = "research/logs/sepa_explore/regime/mood_avail_daily.csv"

# --- 설계 수치(META_GEN1.md — 탐색하지 않는다) ---
GATE_Q = (None, 20, 40, 60)
SCALE_GRID = tuple((a, b) for a in (0.0, 0.25, 0.5) for b in (0.25, 0.5, 1.0))
P2_TOP, P2_STOP = 10, 0.15
N_NULL = 300
ACTIVE_MIN, MAX_DRAWS = gp.ACTIVE_MIN, gp.MAX_DRAWS
SEED = 20261011
JUMP_LIMIT = 0.31
ROT_MIN = 63                       # 순환 널 최소 오프셋(3개월)
P_STATIC = 0.8                     # 노출 맞춤 비교 P(Calmar 더 큼) 문턱
NULL_PASS_MAX = 0.20               # 페이퍼 규칙 (iii)
RA_FEATS = ("breadth_ma50", "breakout_success20", "nh_nl_ratio", "index_vs_ma200")
MARKETS2 = ("KOSPI", "KOSDAQ")
PROXY_MK = ("KOSPI", "KOSDAQ", "ALL")
SCORES = ("RA", "RB", "RS")
FAMS = ("P1", "P1h", "P2")
EXIT_P1 = dict(xv.EXIT_V2_DEFAULT)
EXIT_P1H = {**xv.EXIT_V2_DEFAULT, "take_r": 3.0, "take_frac": 1 / 3, "fb_tol": 0.02, "fb_days": 2}
#: "가장 단순한 것" 의 파라미터 수(결과 보기 전 정의): P2 = 추세 템플릿 4 + 모멘텀 창 2 + 상위 N + 손절 + 리밸런스 주기 = 9,
#: P1 = GA2 유전자 19, P1h = 19 + 청산 v2 유전자 5 = 24. 국면 게이트 +1, 노출 조절 +2. 동률이면 점수 RS < RA < RB.
N_PARAMS = {"P2": 9, "P1": 19, "P1h": 24}
MODE_PARAMS = {"base": 0, "G": 1, "S": 2, "static": 1}
SCORE_RANK = {None: -1, "RS": 0, "RA": 1, "RB": 2}
CAPITALS = (1e8, 3e8)              # 실행 용량 표(원)
MINUTE_SINCE = "2025-08-01"        # minute_bars 시작
BIG = 1 << 30


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


_pct, _f = g1._pct, g1._f


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_meta_gen1_helpers.py)
# ---------------------------------------------------------------------------

def procedures() -> "list[dict]":
    """설계 표의 18 절차(이름·가족·국면 방식·점수). ``primary`` = 판정·추천 대상(기본 3 + RA 변형 5, 정정 M2)."""
    out = [{"name": f, "fam": f, "mode": "base", "score": None} for f in FAMS]
    out += [{"name": f"{f}·G[{s}]", "fam": f, "mode": "G", "score": s} for f in FAMS for s in SCORES]
    out += [{"name": f"{f}·S[{s}]", "fam": f, "mode": "S", "score": s} for f in ("P1", "P2") for s in SCORES]
    for p in out:
        p["primary"] = p["score"] in (None, "RA")
    return out


def n_params(proc: dict) -> int:
    return N_PARAMS[proc["fam"]] + MODE_PARAMS[proc["mode"]]


def zscore_mean(raw: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    """열마다 학습창 행의 평균·표준편차(ddof 0, NaN 무시)로 z → 행별 NaN 무시 평균(``RA``). 표준편차 0·NaN 인 열은 뺀다.

    값이 없는 피처(예: 2017-01 ~ 08 의 nh_nl_ratio)는 그날 평균에서 빠진다(있는 피처만 평균 — 리뷰 M2 질문)."""
    tr = raw[train_rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(tr, axis=0)
        sd = np.nanstd(tr, axis=0)
    ok = np.isfinite(sd) & (sd > 0)
    z = (raw[:, ok] - mu[ok]) / sd[ok]
    cnt = np.isfinite(z).sum(axis=1)
    return np.where(cnt > 0, np.nansum(z, axis=1) / np.maximum(cnt, 1), np.nan)


def gate_thresholds(score: np.ndarray, train_rows: np.ndarray) -> list:
    """{없음, 20, 40, 60} 백분위 문턱(학습창 유한값, ``np.percentile`` 선형 보간). 없음 = −∞. 유한값이 없으면 전부 −∞."""
    v = score[train_rows]
    v = v[np.isfinite(v)]
    return [-np.inf if q is None or len(v) == 0 else float(np.percentile(v, q)) for q in GATE_Q]


def gate_open(score: np.ndarray, thr: float) -> np.ndarray:
    """국면 점수 < 문턱이면 닫힘. 점수 NaN(자료 없음)은 열림 — 닫을 근거가 없다."""
    with np.errstate(invalid="ignore"):
        return ~(score < thr)


def scaler_slots(score: np.ndarray, a: float, b: float, slots: int = gf.SLOTS) -> np.ndarray:
    """슬롯 수 = round(slots × clip(a + b·점수, 0, 1)) — 반올림은 0.5 올림. 점수 NaN 이면 슬롯 전부."""
    with np.errstate(invalid="ignore"):
        v = np.floor(slots * np.clip(a + b * score, 0.0, 1.0) + 0.5)
    return np.where(np.isfinite(score), v, slots).astype(np.int64)


def month_ends(month_id: np.ndarray) -> np.ndarray:
    """각 달의 마지막 거래일 행(다음 행이 다른 달) — 월 리밸런스 신호일. 패널 마지막 행은 다음 달을 모르므로 뺀다."""
    return np.flatnonzero(np.diff(month_id) != 0).astype(np.int64)


def first_argmax(v: "list[float]") -> int:
    """최댓값의 첫 위치(동률이면 앞 — 옵션 순서가 단순한 쪽부터라 단순한 쪽을 고른다). NaN 은 −∞."""
    a = np.asarray(v, float)
    return int(np.argmax(np.where(np.isfinite(a), a, -np.inf)))


def monthly_compound(r: np.ndarray, month_of_day: np.ndarray) -> np.ndarray:
    """날별 수익 → 달별 복리 수익(달 번호 순)."""
    if not len(r):
        return np.zeros(0)
    m = month_of_day - month_of_day[0]
    return np.expm1(np.bincount(m, weights=np.log1p(r), minlength=int(m[-1]) + 1))


def static_slots(exp_variant: float, exp_base: float, slots: int = gf.SLOTS) -> int:
    """노출 맞춤 정적 축소 슬롯 수 = round(slots × 변형 평균 노출 / 기본 평균 노출) ∈ [1, slots] (정정 C4)."""
    if not (np.isfinite(exp_variant) and np.isfinite(exp_base)) or exp_base <= 0:
        return slots
    return int(min(max(np.floor(slots * exp_variant / exp_base + 0.5), 1), slots))


def null_pct(real: float, null) -> float:
    """실제 값의 널 백분위 = 실제보다 작은 널 비율 + ½·같은 비율."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(real):
        return np.nan
    return float((null < real).mean() + 0.5 * (null == real).mean())


def tier2_core(cagr, mdd, calmar, cagr_x2, cagr_ex25, cagr_2024, bar_calmar) -> dict:
    """층 2 의 국면 조건을 뺀 네 조건 — 실제와 널 뽑기에 같은 함수(정정 M6). (1) Calmar > 기준선 ∧ MDD ≥ −25%, (2) 비용 2× CAGR > 0,
    (4) 2025 제외 CAGR > 0, (5) 2020~24 CAGR > 0(정정 C2)."""
    def ok(x):
        return x is not None and np.isfinite(x)
    c1 = bool(ok(calmar) and ok(bar_calmar) and calmar > bar_calmar and mdd >= -0.25)
    c2 = bool(ok(cagr_x2) and cagr_x2 > 0)
    c4 = bool(ok(cagr_ex25) and cagr_ex25 > 0)
    c5 = bool(ok(cagr_2024) and cagr_2024 > 0)
    return {"c1": c1, "c2": c2, "c4": c4, "c5": c5, "core": c1 and c2 and c4 and c5}


def decide(rows: "list[dict]") -> "list[dict]":
    """두 층 판단 규칙 산술(미리 적은 문턱 그대로, 정정 반영). 층 1 은 P2 가족이면 해당 없음(정정 M7 — UMD 대비 알파는 정의상 0 근처).
    층 2 (3) 국면 변형 = ΔCalmar(기본 대비) > 가짜 국면 두 널(달 순열·순환)의 같은 Δ p95 둘 다 ∧ P(Calmar > 노출 맞춤 정적 축소) ≥ 0.8."""
    out = []
    for r in rows:
        d = dict(r)
        fin = lambda x: x is not None and np.isfinite(x)  # noqa: E731
        if r["fam"] == "P2":
            d["t1_alpha"] = d["t1_null"] = d["t1"] = None
        else:
            d["t1_alpha"] = bool(fin(r["alpha_t"]) and r["alpha_t"] > 2)
            d["t1_null"] = bool(fin(r["null_cagr_p95"]) and fin(r["null_calmar_p95"]) and r["cagr"] > r["null_cagr_p95"]
                                and r["calmar"] > r["null_calmar_p95"])
            d["t1"] = d["t1_alpha"] and d["t1_null"]
        core = tier2_core(r["cagr"], r["mdd"], r["calmar"], r["cagr_x2"], r["cagr_ex25"], r["cagr_2024"], r["bar_calmar"])
        d.update({f"t2_{k}": v for k, v in core.items()})
        if r["mode"] == "base":
            d["t2_c3a"] = d["t2_c3b"] = d["t2_c3"] = None
        else:
            d["t2_c3a"] = bool(fin(r["dcalmar"]) and fin(r["null_dcal_p95"]) and fin(r["rot_dcal_p95"])
                               and r["dcalmar"] > r["null_dcal_p95"] and r["dcalmar"] > r["rot_dcal_p95"])
            d["t2_c3b"] = bool(fin(r["p_static"]) and r["p_static"] >= P_STATIC)
            d["t2_c3"] = d["t2_c3a"] and d["t2_c3b"]
        d["t2"] = core["core"] and (d["t2_c3"] is not False)
        out.append(d)
    return out


def paper_rule(d: dict, p1_cagr_2024: float) -> dict:
    """리뷰어 페이퍼 추천 규칙(정정 마지막 줄, 결과 전 고정): 층 2 통과 ∧ (i) 2020~24 CAGR > 0 ∧ (ii) 노출 맞춤 비교 P ≥ 0.8
    (국면 변형 = 정적 슬롯 축소, 기본 절차 = 같은 평균 노출의 월 등가중 × k + 현금) ∧ (iii) 자기 널 층 2 통과율 < 20% 또는 자기 널 Calmar 중앙값 위
    ∧ (iv) P1h 계열이면 2020~24 CAGR 이 P1 보다 큼. 주 절차만 추천 대상."""
    fin = lambda x: x is not None and np.isfinite(x)  # noqa: E731
    r = {"r0": bool(d["t2"]), "r1": bool(fin(d["cagr_2024"]) and d["cagr_2024"] > 0),
         "r2": bool(fin(d["p_expo"]) and d["p_expo"] >= P_STATIC),
         "r3": bool((fin(d["null_pass"]) and d["null_pass"] < NULL_PASS_MAX)
                    or (fin(d["null_calmar_med"]) and d["calmar"] > d["null_calmar_med"])),
         "r4": None if d["fam"] != "P1h" else bool(fin(d["cagr_2024"]) and d["cagr_2024"] > p1_cagr_2024)}
    r["paper"] = bool(d.get("primary")) and r["r0"] and r["r1"] and r["r2"] and r["r3"] and (r["r4"] is not False)
    return r


def simplest(rows: "list[dict]", key: str) -> "dict | None":
    """통과 행(주 절차만) 중 파라미터 수 최소 → 동률이면 점수 RS < RA < RB → 이름 순."""
    ok = [r for r in rows if r.get(key) and r.get("primary")]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (r["n_params"], SCORE_RANK[r["score"]], r["name"]))[0]


def window_stats(r: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """CAGR·MDD·Calmar(전 OOS) + 2025 제외 · 2020~24 · 2021~24 (정정 C3)."""
    s = mf.equity_stats(r)
    for tag, m in (("ex25", dates.year != 2025), ("2024", dates.year <= 2024),
                   ("2124", (dates.year >= 2021) & (dates.year <= 2024))):
        e = mf.equity_stats(r[np.asarray(m)])
        s.update({f"cagr_{tag}": e["cagr"], f"mdd_{tag}": e["mdd"], f"calmar_{tag}": e["calmar"]})
    return s


# ---------------------------------------------------------------------------
# 자료 — 국면 패널·기분 지수·공식 지수·지수 대용
# ---------------------------------------------------------------------------

def load_regime(days: pd.DatetimeIndex) -> dict:
    """국면 원재료(날 × 열): RA 8열(특징 4 × 시장 2), RS 2열, RB 1열, 공식 지수 일수익률 2열, 재구축 지수 대용 일수익률 3열(KOSPI·KOSDAQ·ALL).
    재구축(거래대금 가중) 판 npz 의 열만 쓴다 — 주식수 시총가중 옛 판은 npz 에 없다."""
    z = np.load(REGIME_NPZ, allow_pickle=False)
    feats = [str(x) for x in z["features"]]
    mk = [str(x) for x in z["markets"]]
    dt = pd.to_datetime(z["date"])
    vals = z["values"]

    def col(feat: str, market: str) -> np.ndarray:
        sel = z["market"] == mk.index(market)
        s = pd.Series(vals[sel, feats.index(feat)], index=dt[sel])
        return s[~s.index.duplicated()].reindex(days).to_numpy(float)

    out = {"RA": np.column_stack([col(f, m) for f in RA_FEATS for m in MARKETS2]),
           "RS": np.column_stack([col("idx_sign_ema20", m) for m in MARKETS2]),
           "official": np.column_stack([col("official_ret", m) for m in MARKETS2]),
           "proxy": np.column_stack([col("index_ret", m) for m in PROXY_MK])}
    mood = pd.read_csv(MOOD_CSV, parse_dates=["date"]).set_index("date")["market_direction_ema20"]
    first = mood.first_valid_index()
    rb = mood.reindex(days)
    rb = rb.where(rb.index < first, rb.ffill())          # 기분 지수 첫 값 뒤 빈 날은 직전 EMA 유지
    out["RB"] = rb.to_numpy(float)
    out["rb_first"] = int(days.searchsorted(first)) if first is not None else len(days)
    return out


# ---------------------------------------------------------------------------
# 계산 문맥(부모·작업자 공용)
# ---------------------------------------------------------------------------

class Ctx:
    """사전계산 + 엔진 + 청산 캐시 + P2 표(정리된 가격·잠긴 하한가) + 국면 원재료 + 폴드 창."""

    def __init__(self, cache: str):
        from swing_it.live.config import DEFAULT_STRATEGY  # noqa: PLC0415
        from swing_it.validation.walkforward import FOLDS  # noqa: PLC0415

        from research.signals.sepa import LIMIT_UP_RATIO  # noqa: PLC0415

        pre = self.pre = g1.Pre.load(cache)
        self.E = g1.Engine(pre, max_exit=1)
        self.genes = DEFAULT_STRATEGY.to_genes(flow=False)
        self.spec = g2.g2_spec(self.genes)
        self._ex: OrderedDict = OrderedDict()
        self._rows: dict = {}
        cuts = [FOLDS[k + 1].test_lo if k + 1 < len(FOLDS) else g2.EVENT_HI_NEXT for k in range(len(FOLDS))]
        self.fold_args = [((f.train_lo, f.train_hi, f.test_lo), (f.test_lo, f.test_hi, ct)) for f, ct in zip(FOLDS, cuts, strict=True)]
        self.folds = FOLDS
        self.wins = [(g2.Win(pre, *a), g2.Win(pre, *b)) for a, b in self.fold_args]
        self.tests = [te for _, te in self.wins]
        self.t0, self.t1 = self.tests[0].d_lo, self.tests[-1].end
        self.n_days = pre.n_days
        self.reg = load_regime(pre.days)
        self.dom_lo = pre.day_idx(self.folds[0].train_lo)
        self.dom_hi = self.t1
        self.rb_lo = self.reg["rb_first"]
        self._score_cache: dict = {}
        # --- P2: 정리된 가격(|r| > 31% 날 수익 0)·잠긴 하한가·체결·대조 ---
        O2, H2, L2, C2, bad = mf.clean_jumps(np.asarray(pre.O), np.asarray(pre.H), np.asarray(pre.L), np.asarray(pre.C), JUMP_LIMIT)
        self.P2B = {"O": np.ascontiguousarray(O2), "H": np.ascontiguousarray(H2), "L": np.ascontiguousarray(L2),
                    "C": np.ascontiguousarray(C2)}
        del O2, H2, L2, C2
        self.bad = bad
        TR = np.asarray(pre.TR)
        self.LD = np.ascontiguousarray(mf.locked_down(self.P2B["O"], self.P2B["H"], self.P2B["L"], self.P2B["C"], TR))
        n = self.n_days
        pc, pdv = np.asarray(pre.pc), np.asarray(pre.pd)
        e_c = np.minimum(pdv + 1, n - 1)
        o_e = self.P2B["O"][pc, e_c]
        c_d = self.P2B["C"][pc, pdv]
        with np.errstate(invalid="ignore"):
            lu = np.isfinite(c_d) & (c_d > 0) & (o_e >= c_d * LIMIT_UP_RATIO)
        self.p2_fill = np.ascontiguousarray((pdv + 1 < n) & TR[pc, e_c] & np.isfinite(o_e) & (o_e > 0) & ~lu)
        members, ms = gf.stratum_index(pre.sid, self.p2_fill & (np.asarray(pre.sid) >= 0), gf.stratum_order(pre.sid))
        self.p2_cidx = gf.draw_idx(pre.sid, members, ms, g1.N_PER, g1.CTL_SEED)
        has = TR.any(axis=1)
        self.last_tr = np.where(has, TR.shape[1] - 1 - np.argmax(TR[:, ::-1], axis=1), -1).astype(np.int64)
        me = month_ends(pre.month_id)
        self.rb_d = me[(me >= pdv.min()) & (me <= pdv.max()) & (me + 1 < n)]
        K = len(self.rb_d)
        ptr, pairs, elig_n = [0], [], []
        self.rank_kc = np.full((K, pre.n_codes), BIG, np.int32)
        tt = np.asarray(pre.tt)
        C2 = self.P2B["C"]
        for k, d in enumerate(self.rb_d):
            idx = np.arange(int(pre.day_start[d]), int(pre.day_start[d + 1]))
            cs = pc[idx]
            mom = np.full(len(cs), np.nan)
            if d >= 252:
                with np.errstate(invalid="ignore", divide="ignore"):
                    den = C2[cs, d - 252]
                    mom = C2[cs, d - 21] / np.where(den > 0, den, np.nan) - 1.0
            ok = tt[idx] & np.isfinite(mom)
            idx, cs, mom = idx[ok], cs[ok], mom[ok]
            order = np.lexsort((cs, -mom))
            idx, cs = idx[order], cs[order]
            pairs.append(idx)
            self.rank_kc[k, cs] = np.arange(len(cs))
            ptr.append(ptr[-1] + len(idx))
            elig_n.append(len(idx))
        self.rb_ptr = np.array(ptr, np.int64)
        self.rb_pair = np.concatenate(pairs).astype(np.int64) if pairs else np.zeros(0, np.int64)
        self.rb_elig = np.array(elig_n)
        self.cap10 = np.full(n, gf.SLOTS, np.int64)

    # --- 청산 캐시(v2 커널, 전 쌍) ---
    def exits(self, fam: str, w_fb: int) -> dict:
        key = (fam, int(w_fb))
        if key in self._ex:
            self._ex.move_to_end(key)
            return self._ex[key]
        pre = self.pre
        p = EXIT_P1 if fam == "P1" else EXIT_P1H
        st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = xv.sim_book_v2(
            pre.O, pre.H, pre.L, pre.C, pre.MA50, pre.MA50, pre.TR, pre.last_present, pre.pc, pre.pd, pre.clow, p,
            pivot=pre.piv[:, int(w_fb)])
        self._ex[key] = {"x": x, "ea": ea, "ef": ef, "er": er, "ff": ff, "fr": fr}
        while len(self._ex) > 12:
            self._ex.popitem(last=False)
        return self._ex[key]

    # --- 후보 행 ---
    def sepa_rows(self, fam: str, spec: dict, w, cost_mult: float) -> dict:
        """P1·P1h: 창 ``w`` 의 마스크 ∧ 주문 가능 ∧ 진입 행 < 절단 쌍 → 절단 시가평가·대조·비용을 붙인 압축 행."""
        pre = self.pre
        m = self.E.mask(spec, w.i0, w.i1) & self.E.orderable[w.i0:w.i1]
        if w.cut is not None:
            m &= pre.pd[w.i0:w.i1] + 1 < w.cut
        idx = (np.flatnonzero(m) + w.i0).astype(np.int64)
        fill, cidx = self.E.fill_ctl(spec["ek"], spec["w_idx"], spec["c_idx"])
        ex = self.exits(fam, spec["ek"][6] if spec["ek"][5] else g1.W50)
        cut = w.cut if w.cut is not None else self.n_days + 1
        x, ea, ef, er, ff, fr, tot, ctl, cost = mf.cands(idx, np.full(len(idx), cut, np.int64), pre.pc, pre.pd, ex["x"],
                                                         ex["ea"], ex["ef"], ex["er"], ex["ff"], ex["fr"], pre.O, pre.C, cidx,
                                                         pre.cost_tab, pre.market, float(cost_mult))
        return {"idx": idx, "pd": np.ascontiguousarray(pre.pd[idx]), "pc": np.ascontiguousarray(pre.pc[idx]),
                "fill": np.ascontiguousarray(fill[idx]), "p1": np.ascontiguousarray(pre.rel[idx], dtype=np.float64),
                "p2": np.ascontiguousarray(pre.rs[idx], dtype=np.float64), "x": x, "ea": ea, "ef": ef, "er": er, "ff": ff,
                "fr": fr, "tot": tot, "ctl": ctl, "cost": cost, "early": np.zeros(len(idx), np.bool_),
                "reason": np.full(len(idx), -1, np.int8)}

    def base_rows(self, fam: str, key, w, cost_mult: float) -> dict:
        k = (fam, key, cost_mult)
        if k not in self._rows:
            self._rows[k] = self.sepa_rows(fam, self.spec, w, cost_mult)
        return self._rows[k]

    def p2_rows(self, w, N: np.ndarray, cost_mult: float, ranks: "tuple | None" = None) -> dict:
        """P2: 창 ``w`` 안 리밸런스마다 상위 ``N[k]`` 쌍 → 경로·대조·비용 행. ``ranks`` = (rb_ptr, rb_pair, rank_kc) 바꿔 끼우기(무작위 널)."""
        pre, B = self.pre, self.P2B
        rb_ptr, rb_pair, rank_kc = ranks if ranks is not None else (self.rb_ptr, self.rb_pair, self.rank_kc)
        cut = w.cut if w.cut is not None else self.n_days + 1
        ks = np.flatnonzero((self.rb_d >= w.d_lo) & (self.rb_d < w.d_hi) & (self.rb_d + 1 < cut))
        sel, sk, rk = [], [], []
        for k in ks:
            a = int(rb_ptr[k])
            nk = int(min(N[k], rb_ptr[k + 1] - a))
            if nk <= 0:
                continue
            sel.append(rb_pair[a:a + nk])
            sk.append(np.full(nk, k, np.int64))
            rk.append(np.arange(nk, dtype=np.float64))
        if not sel:
            return empty_rows()
        sel, sk, rk = np.concatenate(sel), np.concatenate(sk), np.concatenate(rk)
        x, ret, reason, early, xopen = mf.p2_trades(sel, sk, pre.pc, pre.pd, self.rb_d, np.ascontiguousarray(N, np.int64),
                                                    rank_kc, B["O"], B["H"], B["L"], B["C"], pre.TR, self.LD, self.last_tr,
                                                    pre.last_present, self.p2_fill, np.full(len(sel), cut, np.int64), P2_STOP)
        ctl = mf.p2_ctl(sel, x, xopen, self.p2_cidx, pre.pc, pre.pd, B["O"], B["L"], B["C"], pre.TR, self.LD, P2_STOP)
        mk = np.maximum(np.asarray(pre.market)[sel], 0)
        cost = np.where(x >= 0, pre.cost_tab[np.maximum(x, 0), mk] * cost_mult, 0.0)
        n = len(sel)
        return {"idx": sel, "pd": np.ascontiguousarray(pre.pd[sel]), "pc": np.ascontiguousarray(pre.pc[sel]),
                "fill": np.ascontiguousarray(self.p2_fill[sel] & (x >= 0)), "p1": rk, "p2": np.zeros(n), "x": x,
                "ea": np.full((n, 2), -1, np.int64), "ef": np.zeros((n, 2)), "er": np.full((n, 2), np.nan),
                "ff": np.ones(n), "fr": ret, "tot": ret, "ctl": ctl, "cost": cost, "early": early, "reason": reason}

    def random_ranks(self, rng: np.random.Generator) -> tuple:
        """P2 널: 리밸런스마다 추세 템플릿 통과 종목 순서를 무작위로(모멘텀 대신)."""
        pc = np.asarray(self.pre.pc)
        rank_kc = np.full_like(self.rank_kc, BIG)
        pair = self.rb_pair.copy()
        for k in range(len(self.rb_d)):
            a, b = int(self.rb_ptr[k]), int(self.rb_ptr[k + 1])
            pair[a:b] = rng.permutation(pair[a:b])
            rank_kc[k, pc[pair[a:b]]] = np.arange(b - a)
        return self.rb_ptr, pair, rank_kc

    # --- 커널·적합도 ---
    def kernel(self, r: dict, cap: np.ndarray, t0: int, t1: int, fam: str):
        lo, hi = mf.day_bounds(r["pd"], self.n_days)
        O, C = (self.P2B["O"], self.P2B["C"]) if fam == "P2" else (self.pre.O, self.pre.C)  # noqa: E741
        return mf.portfolio_meta(r["pd"], r["pc"], r["fill"], r["p1"], r["p2"], r["x"], r["ea"], r["ef"], r["er"], r["ff"],
                                 r["fr"], r["ctl"], r["cost"], r["early"], lo, hi, cap, O, C, t0, t1, gf.SLOT_W, 2 * gf.SLOTS)

    def sel_fitness(self, raw: np.ndarray, w) -> float:
        """국면 옵션 선택 적합도(정정 C1): 학습창 절대 포트폴리오 월 복리 수익의 블록 부트스트랩 Calmar 하단(3개월 블록 500회, 2.5%)."""
        r = raw[w.d_first - w.d_lo:w.end - w.d_lo]
        return mf.calmar_boot_lower(monthly_compound(r, self.pre.month_id[w.d_first:w.end]))

    # --- 국면 점수 ---
    def score(self, sk: str, k: int, raw: "dict | None" = None) -> "tuple[np.ndarray, str]":
        """폴드 ``k`` 의 점수 배열(전 날) · 실제로 쓴 점수(RB → RA 대체 기록). ``raw`` = 가짜 원재료(널 — z 재적합)."""
        use = sk
        if sk == "RB" and self.tests[k].d_lo + 1 < self.rb_lo:
            use = "RA"
        R = self.reg if raw is None else raw
        ck = (use, k)
        if raw is None and ck in self._score_cache:
            return self._score_cache[ck], use
        tr = self.wins[k][0]
        if use == "RA":
            s = zscore_mean(R["RA"], np.arange(tr.d_lo, tr.d_hi))
        elif use == "RS":
            cnt = np.isfinite(R["RS"]).sum(axis=1)
            s = np.where(cnt > 0, np.nansum(R["RS"], axis=1) / np.maximum(cnt, 1), np.nan)
        else:
            s = np.asarray(R["RB"], float)
        if raw is None:
            self._score_cache[ck] = s
        return s, use

    def fake_regime(self, kind: str, rng: np.random.Generator) -> dict:
        """가짜 국면 원재료(정정 M1): 절차 구간(첫 학습창 시작 ~ OOS 끝) 안에서 **하나의 달 순열**(kind 'shuffle') 또는 **하나의 무작위 오프셋
        원형 이동**(kind 'rotate', 오프셋 ∈ [63, 길이 − 63] 거래일)을 RA 8열·RS 2열·RB 에 똑같이 건다."""
        lo, hi = self.dom_lo, self.dom_hi
        if kind == "shuffle":
            ix = mf.month_block_perm(self.pre.month_id, lo, hi, rng)
        else:
            D = hi - lo
            off = int(rng.integers(ROT_MIN, D - ROT_MIN))
            ix = lo + (np.arange(D) - off) % D
        R = dict(self.reg)
        for key in ("RA", "RS", "RB"):
            a = self.reg[key].copy()
            a[lo:hi] = self.reg[key][ix]
            R[key] = a
        return R

    # --- 옵션 적용 ---
    def _apply(self, fam: str, mode: str, opt, s, w, cost_mult: float, key, ranks=None, N_full=None, cap_full=None,
               state=None):
        """옵션(문턱 / (a, b) / 정적 슬롯 수) → (행, 슬롯 상한, N) for 창 ``w``. ``*_full`` 이 있으면 그 창 날만 덮어쓴다(이어 붙이기).
        ``state`` (날별 0~1 — 신호일 기준 진입 허용 비율)도 그 창 날만 채운다."""
        K = len(self.rb_d)
        cap = self.cap10.copy() if cap_full is None else cap_full
        days = slice(w.d_lo, w.d_hi)
        if fam == "P2":
            N = np.full(K, P2_TOP, np.int64) if N_full is None else N_full
            inw = (self.rb_d >= w.d_lo) & (self.rb_d < w.d_hi)
            if mode == "G":
                N[inw] = np.where(gate_open(s[self.rb_d[inw]], opt), P2_TOP, 0)
            elif mode == "S":
                N[inw] = scaler_slots(s[self.rb_d[inw]], *opt, P2_TOP)
            elif mode == "static":
                N[inw] = int(opt)
            cap[self.rb_d[inw]] = N[inw]
            if state is not None:
                for k in np.flatnonzero(inw):
                    nxt = self.rb_d[k + 1] if k + 1 < K else self.n_days
                    state[self.rb_d[k]:nxt] = N[k] / P2_TOP
            return self.p2_rows(w, N, cost_mult, ranks), cap, N
        r = self.base_rows(fam, key, w, cost_mult)
        if mode == "G":
            op = gate_open(s, opt)
            r = sub_rows(r, op[r["pd"]])
            if state is not None:
                state[days] = op[days]
        elif mode == "S":
            cap[days] = scaler_slots(s[days], *opt)
            if state is not None:
                state[days] = cap[days] / gf.SLOTS
        elif mode == "static":
            cap[days] = int(opt)
            if state is not None:
                state[days] = int(opt) / gf.SLOTS
        return r, cap, None

    def run_procedure(self, proc: dict, cost_mult: float = 1.0, raw: "dict | None" = None, ranks=None) -> dict:
        """절차 한 벌: 폴드마다 학습창 옵션 선택(국면 변형) → 시험창 → 이어 붙인 OOS 포트폴리오. ``proc['slots']`` = 정적 축소 슬롯 수."""
        fam, mode, sk = proc["fam"], proc["mode"], proc["score"]
        K = len(self.rb_d)
        N_full = np.full(K, P2_TOP, np.int64)
        cap_full = self.cap10.copy()
        state = np.ones(self.n_days)
        rows, choices = [], []
        for k, (tr, te) in enumerate(self.wins):
            opt, fits, ords, use, opts, s = None, [], [], None, [None], None
            if mode == "static":
                opt = proc["slots"]
            elif mode != "base":
                s, use = self.score(sk, k, raw)
                opts = gate_thresholds(s, np.arange(tr.d_lo, tr.d_hi)) if mode == "G" else list(SCALE_GRID)
                for o in opts:
                    r_tr, cap_tr, _ = self._apply(fam, mode, o, s, tr, cost_mult, ("tr", k), ranks)
                    raw_tr, _, _, _, n_ord, _ = self.kernel(r_tr, cap_tr, tr.d_lo, tr.end, fam)
                    fits.append(self.sel_fitness(raw_tr, tr))
                    ords.append(int(n_ord))
                opt = opts[first_argmax(fits)]
            r_te, cap_full, N_new = self._apply(fam, mode, opt, s, te, cost_mult, ("te", k), ranks,
                                                N_full if fam == "P2" else None, cap_full, state)
            if N_new is not None:
                N_full = N_new
            rows.append(r_te)
            choices.append({"fold": k, "test": self.folds[k].test_lo[:4], "score_used": use, "opt": opt, "opts": opts,
                            "fits": fits, "orders": ords, "j": first_argmax(fits) if fits else None})
        allr = cat_rows(rows)
        raw_d, exc, gross, opened, n_ord, n_unm = self.kernel(allr, cap_full, self.t0, self.t1, fam)
        st = np.r_[np.nan, state[self.t0:self.t1 - 1]]          # 날 t 수익을 다스리는 상태 = 신호일 t−1 의 상태
        return {"raw": raw_d, "exc": exc, "gross": gross, "opened": opened, "n_ord": int(n_ord), "n_unm": int(n_unm),
                "rows": allr, "choices": choices, "state": st}


def empty_rows() -> dict:
    z = np.zeros(0)
    return {"idx": np.zeros(0, np.int64), "pd": np.zeros(0, np.int64), "pc": np.zeros(0, np.int64), "fill": np.zeros(0, np.bool_),
            "p1": z, "p2": z, "x": np.zeros(0, np.int64), "ea": np.zeros((0, 2), np.int64), "ef": np.zeros((0, 2)),
            "er": np.zeros((0, 2)), "ff": z, "fr": z, "tot": z, "ctl": z, "cost": z, "early": np.zeros(0, np.bool_),
            "reason": np.zeros(0, np.int8)}


def sub_rows(r: dict, m: np.ndarray) -> dict:
    return {k: np.ascontiguousarray(v[m]) for k, v in r.items()}


def cat_rows(rs: "list[dict]") -> dict:
    return {k: np.ascontiguousarray(np.concatenate([r[k] for r in rs])) for k in rs[0]}


# ---------------------------------------------------------------------------
# 성적
# ---------------------------------------------------------------------------

def oos_dates(ctx: Ctx) -> pd.DatetimeIndex:
    return ctx.pre.days[ctx.t0 + 1:ctx.t1]


def core_stats(r: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """window_stats + 월 Sharpe(월 복리 수익, ×√12)·연도별."""
    from krx_quant_core.stats.metrics import ann_sharpe  # noqa: PLC0415

    s = window_stats(r, dates)
    mo = pd.Series(r, dates).groupby(dates.to_period("M")).apply(lambda x: float(np.prod(1 + x) - 1))
    s["monthly"] = mo.to_numpy()
    s["sharpe_m"] = ann_sharpe(s["monthly"], 12)
    s["years"] = {int(y): mf.equity_stats(r[np.asarray(dates.year == y)]) for y in sorted(set(dates.year))}
    return s


def official_cmp(r: np.ndarray, dates: pd.DatetimeIndex, off: np.ndarray, since: str = "2024-01-26") -> dict:
    """2024-01-26+ 공식 코스피·코스닥(공식 일수익률이 둘 다 있는 날)과 같은 날만 — CAGR·MDD."""
    m = np.asarray(dates >= pd.Timestamp(since)) & np.isfinite(off).all(axis=1)
    out = {"n": int(m.sum()), "proc": mf.equity_stats(r[m])}
    for j, name in enumerate(MARKETS2):
        out[name] = mf.equity_stats(off[m, j])
    return out


def factor_timing(y: pd.Series, F: pd.DataFrame, state: pd.Series) -> dict:
    """정적 팩터 회귀 + 타이밍 항(시장 × 국면 상태, 정정 M7). 시장 = KOSPI·KOSDAQ 등가중 평균. Newey-West 5."""
    mkt = F[["MKT_KOSPI_EW", "MKT_KOSDAQ_EW"]].mean(axis=1)
    X = pd.concat([F[gp.FACTOR_COLS], (mkt * state.reindex(F.index)).rename("TIMING")], axis=1)
    d = pd.concat([y.rename("y"), X], axis=1, sort=True).dropna()
    if len(d) < 60 or d["TIMING"].std() < 1e-12:
        return {"n": int(len(d))}
    cols = gp.FACTOR_COLS + ["TIMING"]
    r = gp.nw_ols(d["y"].to_numpy(), np.column_stack([np.ones(len(d)), d[cols].to_numpy()]), 5)
    return {"n": r["n"], "alpha_ann": float(r["b"][0] * 252), "alpha_t": float(r["t"][0]), "timing_b": float(r["b"][-1]),
            "timing_t": float(r["t"][-1]), "r2": r["r2"]}


def summarize(ctx: Ctx, res: dict, F: "pd.DataFrame | None", proc: dict) -> dict:
    dates = oos_dates(ctx)
    r = res["raw"][1:]
    s = core_stats(r, dates)
    pp = g2.port_point(ctx.E, res["exc"], ctx.t0, ctx.t0 + 1, ctx.t1)
    years = len(r) / 252.0
    gross = res["gross"][1:]
    s.update({"exc_ann": pp["ann"], "exc_lo": pp.get("lo"), "exc_hi": pp.get("hi"), "orders": res["n_ord"],
              "opened": int(len(res["opened"])), "unmatched": res["n_unm"],
              "gross_max": float(np.nanmax(gross)) if len(gross) else np.nan,
              "exposure": float(np.nanmean(gross)) if len(gross) else np.nan,
              "on_rate": float(np.nanmean(res["state"][1:])),
              "turnover": 2 * gf.SLOT_W * len(res["opened"]) / years if years > 0 else np.nan,
              "expo_years": {int(y): float(np.nanmean(gross[np.asarray(dates.year == y)])) for y in sorted(set(dates.year))}})
    op = res["opened"]
    rr = res["rows"]
    if len(op):
        net = rr["tot"][op] - rr["cost"][op]
        s.update({"trade_net": float(np.mean(net)), "trade_win": float((net > 0).mean()),
                  "trade_hold": float(np.median(rr["x"][op] - rr["pd"][op]))})
        if proc["fam"] == "P2":
            rs = rr["reason"][op]
            names = {**{i: n for i, n in enumerate(gf.REASON_NAMES)}, **mf.REASON_EXTRA}
            s["p2_reasons"] = {names.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(rs, return_counts=True), strict=True)}
            s["p2_broken"] = int(sum(ctx.bad[c, max(int(d) - 252, 0):int(x) + 1].any()
                                     for c, d, x in zip(rr["pc"][op], rr["pd"][op], rr["x"][op], strict=True)))
            s["p2_ld_stops"] = int(sum(bool(ctx.LD[c, int(d) + 1:int(x)].any()) and rz == gf.R_STOP
                                       for c, d, x, rz in zip(rr["pc"][op], rr["pd"][op], rr["x"][op], rs, strict=True)))
    if F is not None:
        y = pd.Series(r, dates)
        s["fac"] = gp.factor_regress(y, F)
        s["fac_exc"] = gp.factor_regress(pd.Series(res["exc"][1:], dates), F)
        if proc["mode"] != "base":
            s["fac_timing"] = factor_timing(y, F, pd.Series(res["state"][1:], dates))
    s["official"] = official_cmp(r, dates, ctx.reg["official"][ctx.t0 + 1:ctx.t1])
    return s


def null_row(ctx: Ctx, res: dict, res2: dict) -> dict:
    """널 뽑기 한 벌의 숫자(층 2 핵심 조건에 필요한 것 전부)."""
    dates = oos_dates(ctx)
    s = window_stats(res["raw"][1:], dates)
    s2 = mf.equity_stats(res2["raw"][1:])
    exc_ann = float(ctx.E.monthly(res["exc"], ctx.t0, ctx.t0 + 1, ctx.t1).mean() * 12)
    g = res["gross"][1:]
    return {"cagr": s["cagr"], "mdd": s["mdd"], "calmar": s["calmar"], "cagr_x2": s2["cagr"], "cagr_ex25": s["cagr_ex25"],
            "cagr_2024": s["cagr_2024"], "cagr_2124": s["cagr_2124"], "exc_ann": exc_ann, "opened": int(len(res["opened"])),
            "exposure": float(np.nanmean(g)), "on_rate": float(np.nanmean(res["state"][1:])) if "state" in res else 1.0}


# ---------------------------------------------------------------------------
# 작업자(spawn) — 널
# ---------------------------------------------------------------------------

_C: dict = {}


def worker_init(cache: str, threads: int = 1) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(threads)
    _C["ctx"] = Ctx(cache)


def regime_null_task(t: dict) -> dict:
    ctx = _C["ctx"]
    out = []
    for rep in t["reps"]:
        rng = np.random.default_rng([SEED, t["pi"], rep, 0 if t["kind"] == "shuffle" else 1])
        raw = ctx.fake_regime(t["kind"], rng)
        res = ctx.run_procedure(t["proc"], 1.0, raw=raw)
        res2 = ctx.run_procedure(t["proc"], 2.0, raw=raw)
        out.append({"proc": t["proc"]["name"], "kind": t["kind"], "rep": rep, **null_row(ctx, res, res2)})
    return {"rows": out}


def entry_null_task(t: dict) -> dict:
    """GA2 무작위 진입 기준선 방식: 폴드마다 진입 유전자 무작위(``ga2_post.random_entry``), 청산 = 그 절차 것 고정, 이어 붙인 OOS."""
    ctx = _C["ctx"]
    rng = np.random.default_rng([SEED, 7, t["batch"], FAMS.index(t["fam"])])
    out = []
    for _ in range(t["n"]):
        cfgs = [gp.random_entry(rng, ctx.genes) for _ in ctx.wins]
        rows = cat_rows([ctx.sepa_rows(t["fam"], g2.g2_spec(c), te, 1.0) for c, te in zip(cfgs, ctx.tests, strict=True)])
        r1 = ctx.kernel(rows, ctx.cap10, ctx.t0, ctx.t1, t["fam"])
        r2 = ctx.kernel(dict(rows, cost=rows["cost"] * 2.0), ctx.cap10, ctx.t0, ctx.t1, t["fam"])
        res = {"raw": r1[0], "exc": r1[1], "gross": r1[2], "opened": r1[3]}
        out.append({"fam": t["fam"], "batch": t["batch"], **null_row(ctx, res, {"raw": r2[0]})})
    return {"rows": out}


def p2_null_task(t: dict) -> dict:
    ctx = _C["ctx"]
    out = []
    p2 = {"name": "P2", "fam": "P2", "mode": "base", "score": None}
    for rep in t["reps"]:
        ranks = ctx.random_ranks(np.random.default_rng([SEED, 9, rep]))
        res = ctx.run_procedure(p2, 1.0, ranks=ranks)
        res2 = ctx.run_procedure(p2, 2.0, ranks=ranks)
        out.append({"rep": rep, **null_row(ctx, res, res2)})
    return {"rows": out}


# ---------------------------------------------------------------------------
# 실행 용량(정정 M8)
# ---------------------------------------------------------------------------

def capacity(ctx: Ctx, res: dict) -> dict:
    """OOS 실제 진입의 주문액(자본 1억·3억 × 10% × 신호일 자산) — (a) 전 OOS: 신호일 ADV20 대비, (b) 2025-08+ : 진입일 09:00 분봉 거래대금 대비."""
    pre = ctx.pre
    rr, op = res["rows"], res["opened"]
    eq = np.cumprod(1 + res["raw"])                       # t0 기준 자산(신호일 t → eq[t − t0])
    pdv = rr["pd"][op]
    nav = eq[pdv - ctx.t0]
    adv = np.asarray(pre.adv20)[rr["idx"][op]] * 1e6     # 백만원 → 원
    out = {"n": int(len(op))}
    for K in CAPITALS:
        ratio = 0.1 * K * nav / adv
        out[f"adv_{int(K / 1e8)}"] = (float(np.nanmedian(ratio)) if len(ratio) else np.nan,
                                      float(np.nanpercentile(ratio, 90)) if len(ratio) else np.nan)
    ent_days = pre.days[pdv + 1]
    m = np.asarray(ent_days >= pd.Timestamp(MINUTE_SINCE))
    out["n_min"] = int(m.sum())
    if not m.any():
        return out
    try:
        import warnings  # noqa: PLC0415

        from swing_it.storage import connect, db_default  # noqa: PLC0415

        warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
        con = connect(db_default())
        codes = np.asarray(pre.codes)[rr["pc"][op][m]]
        need = pd.DataFrame({"code": codes, "ts": ent_days[m] + pd.Timedelta(hours=9), "nav": nav[m]})
        got = []
        for ts, g in need.groupby("ts"):
            q = pd.read_sql("SELECT code, trade_value FROM minute_bars WHERE ts = %(ts)s AND code = ANY(%(codes)s)", con,
                            params={"ts": ts.to_pydatetime(), "codes": list(g["code"])})
            q["ts"] = ts
            got.append(q)
        con.close()
        tv = pd.concat(got) if got else pd.DataFrame(columns=["code", "trade_value", "ts"])
        j = need.merge(tv, on=["code", "ts"], how="left")
        out["n_min_found"] = int(j["trade_value"].notna().sum())
        for K in CAPITALS:
            ratio = (0.1 * K * j["nav"] / j["trade_value"].astype(float)).to_numpy(float)
            ratio = ratio[np.isfinite(ratio)]
            out[f"open_{int(K / 1e8)}"] = ((float(np.median(ratio)), float(np.percentile(ratio, 90)), float((ratio > 0.1).mean()),
                                           float((ratio > 0.3).mean())) if len(ratio) else None)
    except Exception as e:  # noqa: BLE001 — 용량 표는 보조; DB 가 없으면 적고 넘어간다
        out["db_error"] = type(e).__name__
    return out


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def parity(ctx: Ctx) -> dict:
    """P1 · 폴드 0 학습창: 새 커널 정합 초과 = GA2 처리량 엔진(옛 청산 커널), 처리량 적합도 = ``ga2_run.fitness``."""
    tr = ctx.wins[0][0]
    daily, opened, n_ord, n_unm = ctx.E.portfolio(ctx.spec, tr.i0, tr.i1, tr.d_lo, tr.end, tr.cut)
    r = ctx.base_rows("P1", ("tr", 0), tr, 1.0)
    _, exc, _, op, n2, nu2 = ctx.kernel(r, ctx.cap10, tr.d_lo, tr.end, "P1")
    f_old = g2.fitness(ctx.E, "port", ctx.spec, tr)[0]
    mo = ctx.E.monthly(exc, tr.d_lo, tr.d_first, tr.end)
    f_new = float(np.percentile(gf.block_boot(mo, 3, g1.N_BOOT_FIT, 0), 2.5)) if n2 >= g2.PORT_FLOOR else gf.floor_penalty(n2, g2.PORT_FLOOR)
    return {"max_abs_diff": float(np.max(np.abs(exc - daily))), "orders": (int(n_ord), int(n2)),
            "opened_same": bool(len(opened) == len(op) and (r["idx"][op] == opened).all()), "unmatched": (int(n_unm), int(nu2)),
            "fit_old": f_old, "fit_new": f_new}


def main(argv: "list[str] | None" = None) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="ga_run 사전계산 .npy 디렉터리(읽기 전용 mmap)")
    ap.add_argument("--factors", required=True, help="리뷰어 A factors.py 산출 pickle")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nulls", type=int, default=N_NULL)
    args = ap.parse_args(argv)
    debug = os.path.normpath(args.out) != os.path.normpath(OUT_DIR)
    if args.nulls != N_NULL and not debug:
        ap.error("축소 실행은 --out 을 탐색 디렉터리 밖으로 둘 때만")
    sub = os.path.join(args.out, SUB)
    os.makedirs(sub, exist_ok=True)
    t0 = time.time()
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", "8")))
    ctx = Ctx(args.cache)
    pre = ctx.pre
    _log(f"문맥: 쌍 {pre.P:,} · 리밸런스 {len(ctx.rb_d)} · OOS {pre.days[ctx.t0 + 1].date()} ~ {pre.days[ctx.t1 - 1].date()} · "
         f"깨진 조정 일봉 {int(ctx.bad.sum()):,} · 잠긴 하한가 {int(ctx.LD.sum()):,}", t0)
    par = parity(ctx)
    _log(f"패리티(P1 폴드 0 학습창): {par}", t0)
    F = pd.read_pickle(args.factors)
    F.index = pd.to_datetime(F.index)
    dates = oos_dates(ctx)

    # --- 기준선 ---
    mem_d, mem_c = np.asarray(pre.pd), np.asarray(pre.pc)
    bench = {"EW 월": mf.ew_monthly(np.asarray(pre.C), mem_d, mem_c, ctx.rb_d, ctx.t0 + 1, ctx.t1, JUMP_LIMIT),
             "EW 매일": mf.ew_daily(np.asarray(pre.C), mem_d, mem_c, ctx.t0 + 1, ctx.t1, JUMP_LIMIT)}
    for j, m in enumerate(PROXY_MK):
        bench[f"대용 {m}"] = np.nan_to_num(ctx.reg["proxy"][ctx.t0 + 1:ctx.t1, j])
    bench_s = {k: core_stats(v, dates) for k, v in bench.items()}
    for k, v in bench.items():
        bench_s[k]["official"] = official_cmp(v, dates, ctx.reg["official"][ctx.t0 + 1:ctx.t1])
        bench_s[k]["fac"] = gp.factor_regress(pd.Series(v, dates), F)
    bar = bench_s["EW 월"]["calmar"]
    _log("기준선 계산 끝", t0)

    # --- 실제 절차 ---
    procs = procedures()
    real, summ = {}, {}
    for p in procs:
        for cm in (1.0, 2.0):
            res = ctx.run_procedure(p, cm)
            real[(p["name"], cm)] = res
            summ[(p["name"], cm)] = summarize(ctx, res, F if cm == 1.0 else None, p)
        _log(f"절차 {p['name']} 끝", t0)
    # 노출 맞춤: 국면 변형 = 정적 슬롯 축소, 기본 절차 = 월 등가중 × k
    static, cmp = {}, {}
    for p in procs:
        nm = p["name"]
        a = real[(nm, 1.0)]["raw"][1:]
        if p["mode"] == "base":
            k = summ[(nm, 1.0)]["exposure"]
            ewk = k * bench["EW 월"]
            cmp[nm] = {"expo": mf.paired_calmar_boot(a, ewk), "expo_k": k, "ew": mf.paired_calmar_boot(a, bench["EW 월"]),
                       "ewk_stats": mf.equity_stats(ewk)}
            continue
        S = static_slots(summ[(nm, 1.0)]["exposure"], summ[(p["fam"], 1.0)]["exposure"])
        key = (p["fam"], S)
        if key not in static:
            sp = {"name": f"{p['fam']} 정적 {S}슬롯", "fam": p["fam"], "mode": "static", "score": None, "slots": S}
            static[key] = {cm: ctx.run_procedure(sp, cm) for cm in (1.0, 2.0)}
        b = static[key][1.0]["raw"][1:]
        base = real[(p["fam"], 1.0)]["raw"][1:]
        kret = summ[(nm, 1.0)]["exposure"] / summ[(p["fam"], 1.0)]["exposure"]
        cmp[nm] = {"static_slots": S, "static": mf.paired_calmar_boot(a, b), "static_stats": window_stats(b, dates),
                   "static_exposure": float(np.nanmean(static[key][1.0]["gross"][1:])),
                   "scaled": mf.paired_calmar_boot(a, kret * base), "scaled_k": kret,
                   "scaled_126": mf.paired_calmar_boot(a, b, block=126), "base": mf.paired_calmar_boot(a, base)}
        cmp[nm]["expo"] = cmp[nm]["static"]
    t_real = time.time() - t0
    _log(f"실제 절차·정적 축소 끝 ({t_real / 60:.1f}분)", t0)

    # --- 널 ---
    n_null = args.nulls
    nb = 10
    reg_rows, ent_rows, p2_rows_ = [], {f: [] for f in ("P1", "P1h")}, []
    mpc = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mpc, initializer=worker_init,
                                initargs=(args.cache, 1)) as ex:
        ent_f = {f: [ex.submit(entry_null_task, {"fam": f, "batch": j, "n": 25}) for j in range(MAX_DRAWS // 25)]
                 for f in ("P1", "P1h")}
        p2_f = [ex.submit(p2_null_task, {"reps": list(range(a, min(a + nb, n_null)))}) for a in range(0, n_null, nb)]
        reg_f = [ex.submit(regime_null_task, {"proc": p, "pi": pi, "kind": kind, "reps": list(range(a, min(a + nb, n_null)))})
                 for kind in ("shuffle", "rotate") for pi, p in enumerate(procs) if p["mode"] != "base"
                 for a in range(0, n_null, nb)]
        for fam in ("P1", "P1h"):
            for f in ent_f[fam]:
                if f.cancelled():
                    continue
                ent_rows[fam] += f.result()["rows"]
                if sum(1 for r in ent_rows[fam] if r["opened"] >= ACTIVE_MIN) >= n_null:
                    for g in ent_f[fam]:
                        g.cancel()
                    break
            _log(f"무작위 진입 {fam}: 뽑기 {len(ent_rows[fam])}", t0)
        for f in p2_f:
            p2_rows_ += f.result()["rows"]
        _log(f"P2 무작위 {len(p2_rows_)}", t0)
        for i, f in enumerate(cf.as_completed(reg_f), 1):
            reg_rows += f.result()["rows"]
            if i % 60 == 0 or i == len(reg_f):
                _log(f"국면 널 묶음 {i}/{len(reg_f)}", t0)
    _log("널 끝", t0)

    # --- 널 요약 ---
    rdf = pd.DataFrame(reg_rows)
    edf = {f: pd.DataFrame(v) for f, v in ent_rows.items()}
    pdf2 = pd.DataFrame(p2_rows_)
    nulls = {}
    for p in procs:
        nm = p["name"]
        if p["mode"] != "base":
            d = rdf[(rdf["proc"] == nm) & (rdf["kind"] == "shuffle")]
            dr = rdf[(rdf["proc"] == nm) & (rdf["kind"] == "rotate")]
            kind = "가짜 국면(같은 달 순열)"
        elif nm in ("P1", "P1h"):
            d = edf[nm]
            d = d[d["opened"] >= ACTIVE_MIN].head(n_null)
            dr, kind = None, "무작위 진입(청산 고정)"
        else:
            d, dr, kind = pdf2, None, "무작위 TT 10종목"
        pass_ = [tier2_core(x["cagr"], x["mdd"], x["calmar"], x["cagr_x2"], x["cagr_ex25"], x["cagr_2024"], bar)["core"]
                 for _, x in d.iterrows()]
        nulls[nm] = {"kind": kind, "n": int(len(d)), "df": d, "rot": dr, "pass": float(np.mean(pass_)) if pass_ else np.nan}

    def p95(a):
        a = np.asarray(a, float)
        return float(np.nanpercentile(a, 95)) if np.isfinite(a).any() else np.nan

    # --- 판단 규칙 숫자 ---
    drows = []
    for p in procs:
        nm = p["name"]
        s, s2 = summ[(nm, 1.0)], summ[(nm, 2.0)]
        nl = nulls[nm]
        b = summ[(p["fam"], 1.0)]
        reg = p["mode"] != "base"
        fac = s.get("fac_timing", {}) if reg else s["fac"]
        d = nl["df"]
        drows.append({"name": nm, "fam": p["fam"], "mode": p["mode"], "score": p["score"], "primary": p["primary"],
                      "n_params": n_params(p), "alpha_t": fac.get("alpha_t", np.nan), "cagr": s["cagr"], "calmar": s["calmar"],
                      "mdd": s["mdd"], "null_cagr_p95": p95(d["cagr"]), "null_calmar_p95": p95(d["calmar"]),
                      "null_calmar_med": float(np.nanmedian(d["calmar"])) if len(d) else np.nan,
                      "bar_calmar": bar, "cagr_x2": s2["cagr"], "cagr_ex25": s["cagr_ex25"], "cagr_2024": s["cagr_2024"],
                      "cagr_2124": s["cagr_2124"],
                      "dcalmar": s["calmar"] - b["calmar"] if reg else np.nan,
                      "null_dcal_p95": p95(d["calmar"]) - b["calmar"] if reg else np.nan,
                      "rot_dcal_p95": p95(nl["rot"]["calmar"]) - b["calmar"] if reg and nl["rot"] is not None else np.nan,
                      "p_static": cmp[nm]["static"]["p_calmar"] if reg else np.nan,
                      "p_expo": cmp[nm]["expo"]["p_calmar"], "null_pass": nl["pass"],
                      "pct_cagr": null_pct(s["cagr"], d["cagr"]), "pct_calmar": null_pct(s["calmar"], d["calmar"])})
    dec = decide(drows)
    p1_2024 = summ[("P1", 1.0)]["cagr_2024"]
    for d in dec:
        d.update(paper_rule(d, p1_2024))
    pick1, pick2, pick_paper = simplest(dec, "t1"), simplest(dec, "t2"), simplest(dec, "paper")
    dsr = {p["name"]: {N: deflated_sharpe_from_sample(summ[(p["name"], 1.0)]["monthly"], N) for N in (8, len(procs))}
           for p in procs}

    # --- 실행 용량 ---
    cap_tab = {p["name"]: capacity(ctx, real[(p["name"], 1.0)]) for p in procs if p["primary"]}
    elapsed = time.time() - t0
    _log("실행 용량 표 끝", t0)

    # --- 산출물 ---
    srows = []
    for p in procs:
        for cm in (1.0, 2.0):
            s = summ[(p["name"], cm)]
            srows.append({"proc": p["name"], "cost_mult": cm, **{k: s.get(k) for k in (
                "cagr", "mdd", "calmar", "sharpe_m", "total", "cagr_ex25", "mdd_ex25", "cagr_2024", "mdd_2024", "cagr_2124",
                "mdd_2124", "exc_ann", "exc_lo", "exc_hi", "orders", "opened", "unmatched", "gross_max", "exposure", "on_rate",
                "turnover", "trade_net", "trade_win", "trade_hold")},
                "alpha_t": s.get("fac", {}).get("alpha_t"), "alpha_ann": s.get("fac", {}).get("alpha_ann"),
                "alpha_t_timing": s.get("fac_timing", {}).get("alpha_t")})
    pd.DataFrame(srows).to_csv(os.path.join(sub, "summary.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(sub, "decision.csv"), index=False)
    nav = pd.DataFrame(index=dates)
    for p in procs:
        for cm in (1.0, 2.0):
            nav[f"{p['name']} ×{cm:g}"] = np.cumprod(1 + real[(p["name"], cm)]["raw"][1:])
        nav[f"{p['name']} 초과"] = real[(p["name"], 1.0)]["exc"][1:]
    for (fam, S), v in static.items():
        for cm in (1.0, 2.0):
            nav[f"{fam} 정적 {S}슬롯 ×{cm:g}"] = np.cumprod(1 + v[cm]["raw"][1:])
    for k, v in bench.items():
        nav[k] = np.cumprod(1 + v)
    for j, mname in enumerate(MARKETS2):
        nav[f"공식 {mname} 일수익"] = ctx.reg["official"][ctx.t0 + 1:ctx.t1, j]
    nav.to_csv(os.path.join(sub, "nav.csv"))
    frows = []
    for p in procs:
        for c in real[(p["name"], 1.0)]["choices"]:
            frows.append({"proc": p["name"], "fold": c["fold"], "test": c["test"], "score_used": c["score_used"],
                          "opt": str(c["opt"]), "opts": json.dumps([str(o) for o in c["opts"]]),
                          "fits": json.dumps([round(float(x), 6) for x in c["fits"]]), "orders": json.dumps(c["orders"])})
    pd.DataFrame(frows).to_csv(os.path.join(sub, "folds.csv"), index=False)
    rdf.to_csv(os.path.join(sub, "null_regime.csv"), index=False)
    pd.concat(list(edf.values()), ignore_index=True).to_csv(os.path.join(sub, "null_entry.csv"), index=False)
    pdf2.to_csv(os.path.join(sub, "null_p2.csv"), index=False)
    yrows = []
    for nm, s in [(p["name"], summ[(p["name"], 1.0)]) for p in procs] + list(bench_s.items()):
        for y, v in s["years"].items():
            yrows.append({"proc": nm, "year": y, "ret": v["total"], "mdd": v["mdd"], "exposure": s.get("expo_years", {}).get(y)})
    pd.DataFrame(yrows).to_csv(os.path.join(sub, "peryear.csv"), index=False)
    png = plot_equity(nav, procs, sub)
    led = write_ledger_rows(sub, procs, real)
    L = render(locals())
    with open(os.path.join(args.out, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n=== 메타 세대 1 — {elapsed / 60:.1f}분 · 층 1 {pick1 and pick1['name']} · 층 2 {pick2 and pick2['name']} · "
          f"페이퍼 {pick_paper and pick_paper['name']} ===")
    print(f"→ {os.path.join(args.out, RESULTS_MD)}")
    return 0


def write_ledger_rows(sub: str, procs: list, real: dict) -> int:
    """원장 ``meta_gen1/TRIALS.jsonl`` — 절차(18) × 비용(2) 한 줄씩(폴드별 고른 옵션 포함, 결과 숫자는 넣지 않는다)."""
    path = os.path.join(sub, "TRIALS.jsonl")
    seen = g1.read_fingerprints(path)
    rows = []
    for p in procs:
        for cm in (1.0, 2.0):
            ch = [str(c["opt"]) for c in real[(p["name"], cm)]["choices"]]
            rows.append({"config": {"design": DESIGN_DOC, "procedure": p["name"], "family": p["fam"], "mode": p["mode"],
                                    "score": p["score"], "cost_mult": cm, "fold_choices": ch, "n_params": n_params(p),
                                    "universe": "H5 ADV20≥5,000", "slots": f"{gf.SLOTS}×{gf.SLOT_W:g}"},
                         "note": "메타 세대 1 — 절차 한 벌(국면 옵션은 폴드 학습창 절대 Calmar 하단으로 고름)"})
    return g1.write_ledger(path, rows, seen)


def plot_equity(nav: pd.DataFrame, procs: list, sub: str) -> "str | None":
    """절차별 작은 그림(이어 붙인 OOS 자산, 로그) — 절차 · 등가중 월 재조정 · 거래대금 가중 지수 대용(ALL). 범주 3색(참조 팔레트 1~3번).
    이 기계에 한글 글꼴이 없어 그림 글자는 영어로 둔다(한글 글꼴이 있으면 그것을 쓴다)."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib import font_manager  # noqa: PLC0415
    except ImportError:
        return None
    for fn in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
        if any(fn in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams["axes.unicode_minus"] = False
    refs = (("EW 월", "#eb6834", "Universe EW (monthly rebal.)"), ("대용 ALL", "#1baf7a", "Trade-value-weighted index proxy"))
    ncol = 6
    nrow = math.ceil(len(procs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.0 * nrow), sharex=True, sharey=True, facecolor="#fcfcfb")
    for ax, p in zip(axes.ravel(), procs, strict=False):
        ax.set_facecolor("#fcfcfb")
        for col, c, lab in refs:
            ax.plot(nav.index, nav[col].to_numpy(), color=c, lw=1.2, label=lab)
        ax.plot(nav.index, nav[f"{p['name']} ×1"].to_numpy(), color="#2a78d6", lw=2.0, label="Procedure (cost 1x)")
        ax.set_yscale("log")
        ax.set_title(p["name"].replace("·", ".") + ("" if p["primary"] else " (descriptive)"), fontsize=10, color="#0b0b0b")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    for ax in axes.ravel()[len(procs):]:
        ax.set_visible(False)
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("META GEN 1 - stitched OOS equity (log), 2020-01 ~ 2026-04-01", y=1.05, fontsize=12)
    fig.tight_layout()
    path = os.path.join(sub, "equity.png")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _ck(v) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def _opt_str(o) -> str:
    if o is None:
        return "없음"
    if isinstance(o, tuple):
        return f"a {o[0]:g} · b {o[1]:g}"
    if isinstance(o, (int, np.integer)):
        return f"{o}슬롯"
    return "없음(−∞)" if not np.isfinite(o) else f"{o:+.3f}"


def _q(a, x):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, x)) if np.isfinite(a).any() else np.nan


def render(v: dict) -> list[str]:
    ctx, procs, summ, dec, nulls, bench_s = v["ctx"], v["procs"], v["summ"], v["dec"], v["nulls"], v["bench_s"]
    args, par, real, cmp, dsr, cap_tab = v["args"], v["par"], v["real"], v["cmp"], v["dsr"], v["cap_tab"]
    pre = ctx.pre
    ewm = bench_s["EW 월"]
    L = [f"# 메타 세대 1 — 국면 게이트·노출 조절 × 승자 보유 청산 × 단순 모멘텀 — 실행 결과 {date.today().isoformat()} "
         "(러너 자동 생성, 판정 아님)", "",
         "설계 `META_GEN1.md`(466ddc1) + 정정 '설계 리뷰 반영'(737624a, `REVIEW_META_GEN1.md` 1단계, 둘 다 결과 보기 전)을 계산한 숫자다. "
         "✓/✗ 는 미리 적은 문턱과의 **산술 비교**이고 판정·후보 선정은 사람이 한다. **표본 밖(2020-01 ~ 2026-04)은 의사-OOS 다** — 진입 가족·청산 메뉴"
         "(P1h 는 청산 v2 지형을 본 뒤 고른 사후 가설)·유니버스는 이 기간을 본 뒤 정해졌다. 워크포워드는 국면 문턱·(a, b) 선택만 보호한다. "
         "6개 해에 걸린 숫자라 널 대비 p < 0.05 여도 필요조건일 뿐이다(리뷰 M1).", "",
         "**국면 점수:** 게이트·노출 조절의 **주 점수는 RA** (판정·추천 대상), RS(지수 부호 EMA20)는 기준선, RB(기분 지수)는 OOS 가 2025-01 ~ 2026-04 뿐이라 "
         "**기술로만** 적는다(정정 M2). 판정·추천 대상(주 절차) = P1 · P1h · P2 · P1·G[RA] · P1h·G[RA] · P2·G[RA] · P1·S[RA] · P2·S[RA].", "",
         f"재현: `NUMBA_NUM_THREADS=8 uv run --with numba --with matplotlib --extra pg python research/experiments/meta_gen1_run.py "
         f"--cache <pre_v2> --factors <factors.pkl> --workers {args.workers}` · 실행 {v['elapsed'] / 60:.1f}분(실제 절차·정적 축소 "
         f"{v['t_real'] / 60:.1f}분, 작업자 {args.workers}) · numba {gf.numba.__version__ if gf.HAVE_NUMBA else '없음'}.", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 널 {args.nulls}개, 출력 `{args.out}`. 측정으로 쓰지 않는다.", ""]

    pp = v["pick_paper"]
    L += ["## 페이퍼 추천 규칙 (리뷰어가 결과 전 고정 — 정정 마지막 줄, 산술)", "",
          "추천 = 주 절차 ∧ 층 2 통과 ∧ (i) 2020~24 CAGR > 0 ∧ (ii) 노출 맞춤 비교 P(Calmar 더 큼) ≥ 0.8 (국면 변형 = 같은 평균 노출의 정적 슬롯 축소, "
          "기본 절차 = 같은 평균 노출의 월 등가중 × k + 현금) ∧ (iii) 자기 널의 층 2 통과율 < 20% 또는 자기 널 Calmar 중앙값 위 ∧ (iv) P1h 계열이면 2020~24 CAGR > P1.", "",
          "| 절차 | 층 2 | (i) 2020~24 CAGR | (ii) 노출 맞춤 P | (iii) 널 층 2 통과율 · 널 Calmar 중앙 | (iv) | 추천 |", "|---|---|---|---|---|---|---|"]
    for d in dec:
        if not d["primary"]:
            continue
        L.append(f"| {d['name']} | {_ck(d['t2'])} | {_pct(d['cagr_2024'])} {_ck(d['r1'])} | {_f(d['p_expo'], '{:.2f}')} {_ck(d['r2'])} | "
                 f"{_f(d['null_pass'], '{:.0%}')} · {_f(d['null_calmar_med'])} (실제 {_f(d['calmar'])}) {_ck(d['r3'])} | {_ck(d['r4'])} | "
                 f"**{_ck(d['paper'])}** |")
    L += ["", f"**산술 결과:** 추천 통과 {sum(1 for d in dec if d.get('paper'))}개 → "
          + (f"가장 단순한 것 **{pp['name']}** (파라미터 {pp['n_params']})." if pp else
             "없음 → 리뷰어 규칙의 문구: **\"개인용 개별종목 스윙 절차 없음 — 지수 ETF 보유가 기본값\"**."), ""]

    L += ["## 판단 규칙 숫자 — 두 층 (META_GEN1.md '판정' + 정정, 산술 비교)", "",
          "**층 1 · 알파:** 팩터 알파 HAC t > 2 (절대수익 순 시계열, KOSPI·KOSDAQ 등가중·UMD·SMB_ADV·LOWVOL, Newey-West 5; 국면 변형은 **타이밍 항** 시장 × 국면 상태를 더한 회귀의 알파) "
          "∧ 해당 널 p95 위(CAGR **와** Calmar). P2 가족은 UMD 대비 알파가 정의상 0 근처라 **해당 없음**(정정 M7).  ",
          f"**층 2 · 개인용 절차:** (1) Calmar > 유니버스 등가중(**월 재조정**, 정정 C2) Calmar {_f(ewm['calmar'])} ∧ MDD ≥ −25% · (2) 비용 2배 CAGR > 0 · "
          "(3) 국면 변형: (3a) ΔCalmar(변형 − 같은 가족 기본) > 가짜 국면 두 널(같은 달 순열 · 무작위 오프셋 순환)의 같은 Δ p95 둘 다 ∧ (3b) 노출 맞춤 정적 슬롯 축소 대비 "
          "P(Calmar 더 큼) ≥ 0.8 (짝지은 정상 블록 부트스트랩, 평균 블록 63일, 2,000벌) · (4) 2025 제외 CAGR > 0 · (5) 2020~24 CAGR > 0.", "",
          "| 절차 | 주 | 파라미터 | 알파 t | CAGR · 널 p95 (백분위) | Calmar · 널 p95 (백분위) | 층 1 | (1) Calmar · MDD | (2) 비용 2× CAGR | "
          "(3a) ΔCalmar · 순열 p95 · 순환 p95 | (3b) P(> 정적) | (4) 2025 제외 | (5) 2020~24 | 층 2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in dec:
        reg = d["mode"] != "base"
        L.append(f"| {d['name']} | {'●' if d['primary'] else '○'} | {d['n_params']} | {_f(d['alpha_t'])} | {_pct(d['cagr'])} · "
                 f"{_pct(d['null_cagr_p95'])} ({_f(d['pct_cagr'], '{:.0%}')}) | {_f(d['calmar'])} · {_f(d['null_calmar_p95'])} "
                 f"({_f(d['pct_calmar'], '{:.0%}')}) | {_ck(d['t1'])} | {_f(d['calmar'])} · {_pct(d['mdd'])} {_ck(d['t2_c1'])} | "
                 f"{_pct(d['cagr_x2'])} {_ck(d['t2_c2'])} | "
                 + (f"{_f(d['dcalmar'], '{:+.2f}')} · {_f(d['null_dcal_p95'], '{:+.2f}')} · {_f(d['rot_dcal_p95'], '{:+.2f}')} {_ck(d['t2_c3a'])} | "
                    f"{_f(d['p_static'], '{:.2f}')} {_ck(d['t2_c3b'])}" if reg else "— | —")
                 + f" | {_pct(d['cagr_ex25'])} {_ck(d['t2_c4'])} | {_pct(d['cagr_2024'])} {_ck(d['t2_c5'])} | {_ck(d['t2'])} |")
    p1, p2 = v["pick1"], v["pick2"]
    L += ["", "**가장 단순한 통과 절차(주 절차만, 파라미터 수 최소, 동률이면 점수 RS < RA < RB):** "
          f"층 1 → {p1['name'] if p1 else '없음'} · 층 2 → {p2['name'] if p2 else '없음'}"
          + ("" if p2 else " (설계 문구: '이 데이터로 개인용 스윙 절차는 유니버스 보유를 이기지 못한다' 에 해당하는 산술)") + ". "
          "파라미터 수(결과 보기 전 정의): P2 = 9 · P1 = 19 · P1h = 24 · 게이트 +1 · 노출 조절 +2. ● 주 절차 / ○ 기술(RB)·기준선(RS).", "",
          "**널 뽑기의 층 2 통과율(정정 M6 — 국면 조건 (3) 은 널에 걸 수 없어 (1)(2)(4)(5) 만, 실제도 같은 네 조건):**", "",
          "| 절차 | 널 | n | 널 통과율 | 실제 네 조건 | 널 CAGR 중앙 | 널 Calmar 중앙 | DSR P (N 8 · 18, 월 Sharpe) |", "|---|---|---|---|---|---|---|---|"]
    for d in dec:
        nl = nulls[d["name"]]
        ds = dsr[d["name"]]
        L.append(f"| {d['name']} | {nl['kind']} | {nl['n']} | {_f(nl['pass'], '{:.0%}')} | {_ck(d['t2_core'])} | "
                 f"{_pct(_q(nl['df']['cagr'], 50))} | {_f(_q(nl['df']['calmar'], 50))} | "
                 f"{_f(ds[8]['prob_deflated_sharpe'], '{:.1%}')} · {_f(ds[len(procs)]['prob_deflated_sharpe'], '{:.1%}')} "
                 f"(SR {_f(ds[8]['observed_sharpe'], '{:.3f}')}) |")
    L.append("")

    L += ["## 기준선 (2020-01-02 ~ 2026-04-01)", "",
          "| 기준선 | CAGR | MDD | Calmar | 2020~24 CAGR | 2021~24 CAGR | 2025 제외 CAGR | 월 Sharpe | 2024-01-26+ CAGR · MDD |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in bench_s.items():
        o = s["official"]
        L.append(f"| {k} | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_pct(s['cagr_2024'])} | {_pct(s['cagr_2124'])} | "
                 f"{_pct(s['cagr_ex25'])} | {_f(s['sharpe_m'])} | {_pct(o['proc']['cagr'])} · {_pct(o['proc']['mdd'])} |")
    o = ewm["official"]
    L += [f"| 공식 코스피 (2024-01-26+ 만, {o['n']}일) | | | | | | | | {_pct(o['KOSPI']['cagr'])} · {_pct(o['KOSPI']['mdd'])} |",
          f"| 공식 코스닥 (2024-01-26+ 만) | | | | | | | | {_pct(o['KOSDAQ']['cagr'])} · {_pct(o['KOSDAQ']['mdd'])} |", "",
          "- 등가중: 유니버스 = H5(위생 ∧ ADV20 ≥ 50억, 전일 편입), 비용 없음, 조정 일봉 |r| > 31% 는 그 종목 그날 제외(월판은 수익 0). **월 재조정 = 주 기준선**(정정 C2).",
          "- 지수 대용 = 국면 패널의 재구축 거래대금(20일) 가중 지수(단일 종목 ≤ 10%) 일수익 — 삼성전자·SK하이닉스 비중이 공식보다 낮아 **지수 수준으로 읽지 않는다**(REGIME_PANEL.md).",
          "- 현금 수익은 시뮬에서 0 이다 — 2020~26 실제 CMA·MMF 금리(0.5 ~ 3.5%)를 빼먹어 저노출 변형이 연 1~1.5%p 불리하다(리뷰 C4c, 양방향 편향이라 규칙은 그대로).", ""]

    L += ["## 이어 붙인 OOS — 절차별 (2020-01-02 ~ 2026-04-01, 비용 1×)", "",
          "절대수익 = 복리(명목 = 신호일 자산의 10%), CAGR = 날 수 기준(252), MDD = 초기 자산 1 을 첫 고점으로, Calmar = CAGR / |MDD|, 월 Sharpe = 월 복리 수익 × √12. "
          "노출 = 날별 총 보유 가치 / 자산 평균, 켜짐 = 국면 상태 평균. 회전율 = 2 × 10% × 진입 수 / 년. 정합 초과 = GA2 처리량 엔진 시계열(월합 × 12, 3개월 블록 CI).", "",
          "| 절차 | CAGR | MDD | Calmar | 월 Sharpe | 2020~24 CAGR · MDD | 2021~24 CAGR | 2025 제외 CAGR | 비용 2× CAGR · Calmar | 노출 · 켜짐 | 회전율/년 | "
          "정합 초과/년 [95% CI] | 알파/년 (t) · 타이밍 회귀 | 진입(주문) · 짝 없음 · 최대 총 노출 | 트레이드 순평균 · 승률 · 보유 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in procs:
        s, s2 = summ[(p["name"], 1.0)], summ[(p["name"], 2.0)]
        fa, ft = s.get("fac", {}), s.get("fac_timing", {})
        L.append(f"| {p['name']} | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe_m'])} | "
                 f"{_pct(s['cagr_2024'])} · {_pct(s['mdd_2024'])} | {_pct(s['cagr_2124'])} | {_pct(s['cagr_ex25'])} | {_pct(s2['cagr'])} · "
                 f"{_f(s2['calmar'])} | {_f(s['exposure'], '{:.0%}')} · {_f(s['on_rate'], '{:.0%}')} | {_f(s['turnover'], '{:.1f}')} | "
                 f"{_pct(s['exc_ann'])} [{_pct(s['exc_lo'])}, {_pct(s['exc_hi'])}] | {_pct(fa.get('alpha_ann'))} ({_f(fa.get('alpha_t'))})"
                 + (f" · 알파 t {_f(ft.get('alpha_t'))}, 타이밍 t {_f(ft.get('timing_t'))}" if ft else "")
                 + f" | {s['opened']} ({s['orders']}) · {s['unmatched']} · {_f(s['gross_max'])} | {_pct(s.get('trade_net'))} · "
                 f"{_f(s.get('trade_win'), '{:.0%}')} · {_f(s.get('trade_hold'), '{:.0f}')}일 |")
    L += ["", "팩터 베타(절대수익 순, 전 OOS, 순서 " + " / ".join(gp.FACTOR_COLS) + "): " + " · ".join(
        f"{p['name']} " + "/".join(f"{summ[(p['name'], 1.0)]['fac'].get('betas', {}).get(c, np.nan):+.2f}" for c in gp.FACTOR_COLS)
        for p in procs if p["mode"] == "base") + ". 정합 초과 시계열 알파 t(참고, GA2 판정 축): " + " · ".join(
        f"{p['name']} {_f(summ[(p['name'], 1.0)]['fac_exc'].get('alpha_t'))}" for p in procs if p["primary"]) + ".", ""]
    p2s = summ[("P2", 1.0)]
    if "p2_reasons" in p2s:
        L += [f"P2 체결 점검(정정 M4): 청산 사유 {p2s['p2_reasons']} · 순위 창~보유 중 깨진 조정 일봉(|r| > 31%, 그날 수익 0 처리)이 있던 진입 {p2s.get('p2_broken')}건 · "
              f"보유 중 하한가 잠김이 있던 손절 {p2s.get('p2_ld_stops')}건 · 사전계산 전체 깨진 일봉 {int(ctx.bad.sum()):,} · 잠긴 하한가 {int(ctx.LD.sum()):,} (종목-일).", ""]

    L += ["## 노출 맞춤 비교 (정정 C4·M3 — 짝지은 정상 블록 부트스트랩, 평균 블록 63일, 2,000벌, ΔCalmar 90% CI · P(Δ > 0))", "",
          "| 절차 | 비교 대상 | 대상 CAGR · MDD · Calmar | ΔCalmar [5%, 50%, 95%] · P | ΔCAGR [5%, 95%] · P | 참고 |", "|---|---|---|---|---|---|"]
    for p in procs:
        c = cmp[p["name"]]
        if p["mode"] == "base":
            st, e = c["ewk_stats"], c["expo"]
            L.append(f"| {p['name']} | 월 등가중 × {c['expo_k']:.2f} + 현금 | {_pct(st['cagr'])} · {_pct(st['mdd'])} · {_f(st['calmar'])} | "
                     f"[{_f(e['dcalmar_lo'], '{:+.2f}')}, {_f(e['dcalmar_med'], '{:+.2f}')}, {_f(e['dcalmar_hi'], '{:+.2f}')}] · {e['p_calmar']:.2f} | "
                     f"[{_pct(e['dcagr_lo'])}, {_pct(e['dcagr_hi'])}] · {e['p_cagr']:.2f} | 월 등가중(노출 100%) 대비 P {c['ew']['p_calmar']:.2f} |")
            continue
        st, e = c["static_stats"], c["static"]
        L.append(f"| {p['name']} | {p['fam']} 정적 {c['static_slots']}슬롯(노출 {_f(c['static_exposure'], '{:.0%}')}) | {_pct(st['cagr'])} · "
                 f"{_pct(st['mdd'])} · {_f(st['calmar'])} | [{_f(e['dcalmar_lo'], '{:+.2f}')}, {_f(e['dcalmar_med'], '{:+.2f}')}, "
                 f"{_f(e['dcalmar_hi'], '{:+.2f}')}] · {e['p_calmar']:.2f} | [{_pct(e['dcagr_lo'])}, {_pct(e['dcagr_hi'])}] · {e['p_cagr']:.2f} | "
                 f"블록 126일 P {c['scaled_126']['p_calmar']:.2f} · 수익 × {c['scaled_k']:.2f} 축소 대비 P {c['scaled']['p_calmar']:.2f} · "
                 f"기본 대비 P {c['base']['p_calmar']:.2f} |")
    L.append("")

    L += ["## 널 분포 — 이어 붙인 OOS (절대수익, 비용 1×)", "",
          "| 절차 | 널 | n | CAGR 중앙 [p5, p95] | Calmar 중앙 [p5, p95] | MDD 중앙 | 정합 초과/년 p95 | 켜짐 · 노출(실제 / 널 중앙) | 순환 널 Calmar p95 · 켜짐 중앙 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in dec:
        nl = nulls[d["name"]]
        df = nl["df"]
        s = summ[(d["name"], 1.0)]
        rot = nl["rot"]
        L.append(f"| {d['name']} | {nl['kind']} | {nl['n']} | {_pct(_q(df['cagr'], 50))} [{_pct(_q(df['cagr'], 5))}, {_pct(_q(df['cagr'], 95))}] | "
                 f"{_f(_q(df['calmar'], 50))} [{_f(_q(df['calmar'], 5))}, {_f(_q(df['calmar'], 95))}] | {_pct(_q(df['mdd'], 50))} | "
                 f"{_pct(_q(df['exc_ann'], 95))} | {_f(s['on_rate'], '{:.0%}')} · {_f(s['exposure'], '{:.0%}')} / "
                 f"{_f(_q(df['on_rate'], 50), '{:.0%}')} · {_f(_q(df['exposure'], 50), '{:.0%}')} | "
                 + (f"{_f(_q(rot['calmar'], 95))} · {_f(_q(rot['on_rate'], 50), '{:.0%}')} (n {len(rot)})" if rot is not None else "—") + " |")
    er = v["ent_rows"]
    L += ["", "- 무작위 진입(P1·P1h): `ga2_post.random_entry` 그대로(GA2 진입 유전자 공간 균등, 폴드마다 따로), 청산은 그 절차 것 고정(돌파 실패 피벗 = 그 config 의 피벗 창). "
          "활동 ≥ 100 진입만 300개가 찰 때까지 뽑음 — 뽑기 " + " · ".join(f"{f} {len(er[f])}" for f in er) + ".",
          "- 가짜 국면(정정 M1): 국면 원재료(RA 8열·RS 2열·RB)에 **하나의 달 순열**(첫 학습창 시작 ~ OOS 끝)을 똑같이 걸고 z·문턱·(a, b) 를 다시 고른다. "
          "순환 널 = 같은 구간에서 하나의 무작위 오프셋(≥ 63거래일)으로 원형 이동 — 자기상관·구간 길이·켜짐 비율을 그대로 둔다.",
          "- P2 무작위: 리밸런스마다 추세 템플릿 통과 종목 순서를 무작위로(모멘텀 대신) 상위 10 — 손절·계속 보유·체결·대조 규칙은 P2 그대로.", ""]

    L += ["## 공식 지수 대비 (2024-01-26 ~ 2026-04-01, 공식 코스피·코스닥 일수익률이 있는 날, 대부분 강세장 — 같은 구간 등가중 병기)", "",
          "| 절차 | 날 수 | 절차 CAGR · MDD | 코스피 CAGR · MDD | 코스닥 CAGR · MDD | 코스피 대비 CAGR |", "|---|---|---|---|---|---|"]
    for nm, s in [(p["name"], summ[(p["name"], 1.0)]) for p in procs] + list(bench_s.items()):
        o = s["official"]
        L.append(f"| {nm} | {o['n']} | {_pct(o['proc']['cagr'])} · {_pct(o['proc']['mdd'])} | {_pct(o['KOSPI']['cagr'])} · "
                 f"{_pct(o['KOSPI']['mdd'])} | {_pct(o['KOSDAQ']['cagr'])} · {_pct(o['KOSDAQ']['mdd'])} | "
                 f"{_pct(o['proc']['cagr'] - o['KOSPI']['cagr'])} |")

    yrs = sorted(ewm["years"])
    L += ["", "## 연도별 수익 (MDD) [평균 노출] — 비용 1×", "", "| 절차 | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
    for nm, s in [(p["name"], summ[(p["name"], 1.0)]) for p in procs] + list(bench_s.items()):
        ex = s.get("expo_years", {})
        L.append(f"| {nm} | " + " | ".join(
            (f"{_pct(s['years'][y]['total'])} ({_pct(s['years'][y]['mdd'])})" + (f" [{ex[y]:.0%}]" if y in ex else ""))
            if y in s["years"] else "—" for y in yrs) + " |")

    L += ["", "## 폴드별 국면 옵션 선택 (학습창 절대 Calmar 하단 최대 · 비용 1×, 정정 C1 요구 표)", "",
          "| 절차 | " + " | ".join(f"폴드 {c['fold']} ({c['test']})" for c in real[(procs[3]['name'], 1.0)]["choices"]) + " | 게이트 '없음' 폴드 수 |",
          "|---|" + "---|" * (len(ctx.wins) + 1)]
    for p in procs:
        if p["mode"] == "base":
            continue
        cells, n_none = [], 0
        for c in real[(p["name"], 1.0)]["choices"]:
            fb = "" if c["score_used"] == p["score"] else f" [{c['score_used']} 대체]"
            j = c["j"]
            if p["mode"] == "G":
                lab = f"q{GATE_Q[j] if GATE_Q[j] is not None else '없음'} {_opt_str(c['opt'])}"
                n_none += GATE_Q[j] is None
            else:
                lab = _opt_str(c["opt"])
            cells.append(f"{lab} (Calmar 하단 {_f(c['fits'][j], '{:+.2f}')}, 주문 {c['orders'][j]}){fb}")
        L.append(f"| {p['name']} | " + " | ".join(cells) + f" | {n_none if p['mode'] == 'G' else '—'} |")
    L += ["", "(옵션별 학습 적합도·주문 수 전부: `meta_gen1/folds.csv`. 게이트 '없음' 을 6폴드 중 4개 이상 고른 변형은 기본 절차와 거의 같고 가짜 국면 비교가 퇴화한다 — 리뷰 C1(b).)", ""]

    L += ["## 실행 용량 (정정 M8 — 주 절차의 OOS 실제 진입, 주문액 = 자본 × 10% × 신호일 자산)", "",
          "| 절차 | 진입 | 주문 / 신호일 ADV20 (1억: 중앙 · p90 / 3억: 중앙 · p90) | 2025-08+ 진입(분봉 찾음) | 주문 / 09:00 분봉 거래대금 1억: 중앙 · p90 · >10% · >30% | 3억: 같은 순서 |",
          "|---|---|---|---|---|---|"]
    for nm, c in cap_tab.items():
        a1, a3 = c.get("adv_1", (np.nan, np.nan)), c.get("adv_3", (np.nan, np.nan))
        fo = lambda o: " · ".join(_f(x, "{:.1%}") for x in o) if o else (c.get("db_error") or "—")  # noqa: E731
        L.append(f"| {nm} | {c['n']} | {_f(a1[0], '{:.2%}')} · {_f(a1[1], '{:.2%}')} / {_f(a3[0], '{:.2%}')} · {_f(a3[1], '{:.2%}')} | "
                 f"{c.get('n_min', 0)} ({c.get('n_min_found', 0)}) | {fo(c.get('open_1'))} | {fo(c.get('open_3'))} |")
    L += ["", "- `minute_bars` 는 2025-08-01 부터라 09:00 분봉(시가 동시호가 체결 포함) 비율은 OOS 마지막 8개월 진입만이다. 09:00 ~ 09:30 VWAP 체결 민감도는 페이퍼 후보가 생길 때 낸다(리뷰 M8).", ""]

    L += ["## 검증·자원", "",
          f"- 패리티(P1 · 폴드 0 학습창): 새 커널 정합 초과 vs GA2 처리량 엔진(옛 청산 커널) 날별 최대 차 {par['max_abs_diff']:.2e}, 주문 {par['orders']}, "
          f"진입 같음 {par['opened_same']}, 짝 없음 {par['unmatched']}, GA2 처리량 적합도 {par['fit_old']:+.5f} vs 새 커널로 같은 식 {par['fit_new']:+.5f}.",
          f"- 사전계산 `{os.path.basename(os.path.normpath(args.cache))}`: 쌍 {pre.P:,}(사건 {g1.EVENT_LO} ~ {g1.EVENT_HI}, H5 = 위생 ∧ ADV20 ≥ 50억), "
          f"리밸런스 {len(ctx.rb_d)}회(추세 템플릿 ∧ 12-1 모멘텀 유한 종목 중앙 {int(np.median(ctx.rb_elig))}개). 가격 `relax_run.KrxPatch` 보정판.",
          f"- 산출물: `meta_gen1/` nav(절차별 일별 NAV 비용 1·2배 + 정적 축소 + 기준선)·summary·decision·folds·peryear·null_regime·null_entry·null_p2 CSV(git 무시), "
          f"자산 곡선 `{os.path.basename(v['png']) if v['png'] else '없음'}`, 원장 `meta_gen1/TRIALS.jsonl`(덧붙인 {v['led']}줄).", ""]

    L += ["## 구현 메모 — 모호했던 곳과 정한 것(결과 보기 전 결정)", "",
          "- **절대수익:** 같은 주문·슬롯·체결·청산·비용 규칙에서 명목 = 신호일 종가 자산의 10%(복리), 날별 손익 / 전일 자산(`meta_fast.portfolio_meta`). 현금 제약 없음 — 최대 총 노출을 표에 적었다.",
          "- **선택 적합도(정정 C1):** 학습창(진입 ∈ 학습창, 학습창 끝 시가평가) 절대 포트폴리오의 월 복리 수익 → 3개월 이동 블록(비원형) 부트스트랩 500회(시드 0)의 Calmar 2.5% 하단. "
          "MDD 하한 1%(낙폭 없는 표본의 분모 0 방지). 주문 수 하한 없음. 동률이면 앞 옵션(게이트: 없음 → 20 → 40 → 60, 노출: a 오름 → b 오름).",
          "- **폴드 경계:** GA2 규약 그대로 시험창 경계에서 열린 포지션을 시가평가로 닫고 비용을 매긴다 — 이어 붙인 OOS 에 연 5회 강제 청산이 들어 있다.",
          "- **비용 2×:** 옵션 선택부터 이어 붙인 OOS 까지 절차 전체를 비용 2배로(세금도 2배 — 리뷰 m3 의 '세금 + 2×(수수료+스프레드)' 판은 내지 않았다).",
          "- **노출 맞춤 비교(정정 C4):** 국면 변형의 평균 노출 / 같은 가족 기본의 평균 노출 × 10 을 반올림한 고정 슬롯 수로 기본 절차를 다시 돌린 것(P2 는 상위 N = 그 수)이 비교 대상. "
          "보조로 '기본 일별 수익 × 노출 비율 + 현금' 과 블록 126일 판도 적었다. 기본 절차의 (ii) 는 '월 등가중 × 그 절차 평균 노출 + 현금'(리뷰 C2b).",
          "- **층 2 (3):** 설계의 가짜 국면 널 p95 에 정정의 순환 널 p95 와 노출 맞춤 P ≥ 0.8 을 **모두** 요구(정정이 덧붙인 조건이라 AND). 층 2 (1) 의 기준선은 월 재조정 등가중(정정 C2 의 주 기준선).",
          "- **층 1 의 '해당 널':** P1·P1h = 무작위 진입, 국면 변형 = 같은 달 순열 가짜 국면. 널 통과 = CAGR 과 Calmar 둘 다 p95 초과. 국면 변형의 알파 t = 타이밍 항 회귀의 절편.",
          "- **국면 상태(타이밍 항·켜짐 비율):** 날 t 의 상태 = 신호일 t−1 의 게이트 열림(0/1) 또는 슬롯 상한 / 10; P2 는 직전 리밸런스의 N / 10. 시장 = KOSPI·KOSDAQ 등가중 팩터 평균.",
          "- **P1h:** 설계 표 그대로 '돌파 실패 2%·2일'(지형 최고 칸 13274 는 1일). 손절 상한 0.10·최대 보유 120·손절 하한 3% 는 현 프로토콜 값. 사후 가설 — 추천 규칙 (iv).",
          "- **P2 세부:** 신호일 = 매월 마지막 거래일 종가(그날 유니버스 ∧ 추세 템플릿 ∧ 정리된 가격의 12-1 모멘텀 유한), 진입 = 다음날 시가. 시가가 상한가(전일 종가 × 1.293 이상)면 미체결 "
          "(그 슬롯은 그 달 빈다). 하한가로 잠긴 날(한 가격 ∧ 종가 ≤ 직전 종가 × 0.707)은 손절·리밸런스 매도 불가 → 다음 잠기지 않은 거래일 시가. 조정 일봉 |r| > 31% 는 그날부터 뒤 가격을 그 비율로 나눠 "
          "그 종목 그날 수익만 0 (순위·보유·대조 모두). 계속 상위 10 인 보유는 거래하지 않는다(재가중 없음). 리밸런스 매도는 신호일 종가에 정해지므로 그날 주문의 빈 슬롯으로 센다(P1 계열은 끔). "
          "−15% 손절 뒤 슬롯은 다음 리밸런스까지 현금. 시세가 끊긴 보유는 마지막 거래 가능일 종가(폐지/데이터 끝). 대조 = 같은 날 같은 층(RELAX 층) 쌍, 같은 진입일 시가 → 트레이드와 같은 청산일, 자기 −15% 손절.",
          "- **P1 계열 체결:** 상한가 시가 진입은 GA 엔진 규약대로 버린다(`lu_skip`). 하한가 잠김 손절 지연은 P1 계열에 넣지 않았다(정정 M4 는 P2 체결 항목) — P1 손절은 장중 손절가 체결 그대로다.",
          "- **국면 게이트 P2:** 닫힌 리밸런스는 상위 0 — 보유도 다음날 시가에 판다('그 달 현금'), 판단은 신호일(전월 말) 점수. P1 계열 게이트는 신규 진입만 막는다.",
          "- **노출 조절:** 슬롯 상한은 신규 진입만 막는다(초과 보유를 팔지 않는다 — 리뷰 m1). P2·S 는 그 리밸런스의 상위 N = 슬롯 수(모멘텀 하위부터 밖으로 밀려 판다). 반올림 = 0.5 올림.",
          "- **점수:** RA = 폴드 학습창 신호일의 열별 평균·표준편차로 z(8열, 있는 피처만 평균 — 2017-01 ~ 08 은 nh_nl_ratio·index_vs_ma200 없이). RS = KOSPI·KOSDAQ `idx_sign_ema20` 평균, "
          "RB = `mood_avail_daily.csv` `market_direction_ema20` (둘 다 z 없음 — a + b·점수 는 원척도). RB 는 시험창 시작이 기분 지수 첫날 뒤인 폴드(4·5)만, 앞 폴드는 RA 대체. 점수 NaN 은 열림·슬롯 전부.",
          "- **고정 분모 R·자기 종목 대조 제외(리뷰 교훈):** R 배수를 쓰지 않는다(수익 %·포트폴리오만). 대조는 같은 날 같은 층에서 자기 쌍을 빼고 뽑는다(하루 한 종목 한 쌍).",
          "- **DSR:** 이어 붙인 OOS 월 복리 수익의 Sharpe, N = 주 절차 수 8 · 보고 변형 수 18(비용 2배는 별개 시행으로 세지 않음).",
          "- **팩터:** 리뷰어 A `factors.py` 산출 pickle(이 저장소 밖 스크래치) — 재현에는 그 스크립트가 필요하다.", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
