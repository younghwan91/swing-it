#!/usr/bin/env python
"""메타 세대 4 — 개인 코어 운용: 지수 타이밍(C1f·C1·C2·C4) × 코어-위성(C3). **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/META_GEN4.md`` (2cb127b + 정정 412b012 — 둘 다 결과 보기 전). 이 러너는 숫자를 ``META_GEN4_RESULTS.md`` 에 적고
미리 적은 판정 조건 (1)~(5)·위성 조건과의 **산술 비교**만 한다. 판정은 사람이 한다.

**자산:** 코어 = KOSPI200 가격 지수(``research/data/krx_index_daily_naver.csv`` 의 ``KPI200`` — 네이버 차트 일봉, DB ``sector_index`` 와 겹치는
655일 전부 일치). 현금 = CD91 연평균 근사(:data:`CD91` — 메타 세대 3 ``CD91_EXT`` 그대로, **기억값 · DB 밖**).

**체결:** 신호 = d 종가, 매매 = d+1 지수 시가. 날 t 수익 = (1 + p_{t−1}·밤 수익 + (1 − p_{t−1})·현금 날 금리) × (1 − 0.07%·|p_t − p_{t−1}|)
× (1 + p_t·(장중 수익 − 0.05%/252)) − 1 (``p_t`` = t 시가 매매 뒤 보유 = 신호 ``x_{t−1}``). 창 첫날 직전은 현금(p = 0).

**절차:** C0 매수보유 · C1f 종가 > MA200 · C1 MA n ∈ {50, 100, 150, 200} · C2 RA(breadth_ma50·breakout_success20·nh_nl_ratio × KOSPI·KOSDAQ,
학습창 z 평균) ≥ 문턱(학습창 {없음, 20, 40, 60} 백분위) · C4 듀얼 모멘텀(월말 12개월 수익 > 현금 12개월) · C3·X = 코어 X 80% + P1(메타 세대 1,
유휴 현금 금리 포함 NAV) 20%, 매월 첫 거래일 되돌림. C1·C2 선택 = 학습창 일별 경로의 월 블록 부트스트랩 500회 Calmar 5 백분위(정정), 동률이면 전환 적은 쪽.

**지표(정정):** Sharpe = 초과 Sharpe(월 복리 수익 − 같은 달 현금 복리) × √12. MDD·Calmar = 일별 경로. 부트스트랩 = 일별 수익을 달 블록(3개월, 참고 12개월)으로
재표본해 이어 붙인 일별 경로. 배당 = 1.8% 고정 보정 줄 + 일별 가산(연 1.8%/252 × 보유) 줄. 세후 = 현금 이자·배당 15.4%, ETF 매매차익 비과세.

**창:** 고정 규칙 = 전체 2017-01 ~ 2026-09(표본 내) + 2017~2024 · 2025~26. 적합 규칙 = ``FOLDS`` 이어 붙인 OOS 2020-01 ~ 2026-09
(+ 2020~24 · 2025~26). C3 = P1 NAV 가 있는 2020-01-02 ~ 2026-04-01. 기술 줄: ~ 2026-06-19 (2026-07 급락 전).

**널:** 무작위 타이밍 — 2-상태 마르코프 1,000벌(같은 시장 안 비율·평균 전환 빈도; C4 는 월말에만 전환) + 원형 이동 1,000벌(실제 보유 열을 무작위
오프셋만큼 돌림) · C2 국면 점수 월 블록 순열 300벌(z·문턱 재적합).

Run:  uv run python research/experiments/meta_gen4_run.py [--workers 10] [--check-p1c <review_meta1/indiv.csv>]
      (DB 접속 없음 — CSV·npz·메타 세대 1 NAV 만 읽는다.)
"""

from __future__ import annotations

import argparse
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
import meta_gen1_run as mg1  # noqa: E402
import meta_gen2_run as mg2  # noqa: E402
import meta_gen3_run as mg3  # noqa: E402

from research.signals import meta_fast as mf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
SUB = "meta_gen4"
RESULTS_MD = "META_GEN4_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/META_GEN4.md (2cb127b + 정정 412b012)"
INDEX_CSV = "research/data/krx_index_daily_naver.csv"
CORE_SYM = "KPI200"
P1_NAV = "research/logs/sepa_explore/meta_gen1/nav.csv"
P1_PERYEAR = "research/logs/sepa_explore/meta_gen1/peryear.csv"
P1_COL = "P1 ×1"
REGIME_NPZ = mg1.REGIME_NPZ

# --- 설계 수치(META_GEN4.md + 정정 — 탐색하지 않는다) ---
COST_1W = 0.0007                  # ETF 거래당 편도(수수료 + 스프레드)
FEE_YR = 0.0005                   # ETF 보수 연
DIV_YR = 0.018                    # 배당: 연 1.8%(고정 보정 줄 · 일별 가산 줄)
DIV_SENS = (0.015, 0.022)         # (3) 배당 민감도 참고 줄(리뷰 M1)
TAX = 0.154                       # 세후 줄: 현금(MMF) 이자·분배금 15.4%, 국내 주식형 ETF 매매차익 비과세
SAT_W = 0.20                      # C3 위성 비중
MA_FIXED = 200
MA_GRID = (50, 100, 150, 200)
DM_MONTHS = 12
GATE_Q = mg1.GATE_Q               # C2 문턱 격자(학습창 백분위 {없음, 20, 40, 60}) — 메타 세대 1 게이트와 같다
RA4_FEATS = ("breadth_ma50", "breakout_success20", "nh_nl_ratio")   # 지수 대용 열(index_vs_ma200) 제외
MARKETS2 = mg1.MARKETS2
FIT_N, FIT_BLOCK, FIT_Q, FIT_FLOOR = 500, 3, 5.0, 0.01   # C1·C2 선택 적합도: 일별 경로 월 블록 500회 Calmar 5 백분위(MDD 하한 1%)
N_RAND = 1000
N_PERM = 300
N_BOOT, BLOCK_M, BLOCK_ALT = 2000, 3, 12
ROT_MIN = 60                      # 원형 이동 최소 오프셋(거래일)
SEED = 20261014
P_BOOT = 0.8                      # (1)·위성: 부트스트랩 확률 문턱
MDD_GAIN = 0.10                   # (2) MDD 가 10%p 이상 얕음
CAGR_LOSS = 0.02                  # (3) CAGR 손실 ≤ 연 2%p(배당 보정 줄)
#: CD91 연평균 근사(비율) — 메타 세대 3 ``CD91_EXT``(2017~19 = 세대 3 러너 값, 2020~26 = 세대 1 리뷰어 ``review_meta1/indiv.py`` 값).
#: **원천은 기억값이다(DB 에 금리 테이블이 없다).** 표에 없는 해(2015·2016)는 가장 가까운 해(2017) 값.
CD91 = dict(mg3.CD91_EXT)
W_START, W_END = "2017-01-01", "2026-09-30"
W_OOS = "2020-01-01"
W_PRE_END, W_POST = "2024-12-31", "2025-01-01"
PRECRASH_END = "2026-06-19"       # 2026-06-22 ~ 07-30 급락(−40.9%, 27거래일) 전 마지막 날 — 기술 줄
CRASH = ("2026-06-22", "2026-07-30")
EVENT_DD = 0.15                   # 사건표: C0 낙폭 15% 넘는 구간
#: 절차 정의(코드 · 파라미터 수 · 종류). 순서 = 표 순서.
PROCS = (("C0", 0, "fixed"), ("C1f", 0, "fixed"), ("C4", 0, "fixed"), ("C1", 1, "fitted"), ("C2", 1, "fitted"))
TIMING = ("C1f", "C4", "C1", "C2")
COLORS = {"C0": "#2a78d6", "C1f": "#eb6834", "C4": "#1baf7a", "C1": "#eda100", "C2": "#e87ba4", "C3·C0": "#008300"}


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


_pct, _f = g1._pct, g1._f


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_meta_gen4_helpers.py)
# ---------------------------------------------------------------------------

def ma_signal(close: np.ndarray, n: int) -> np.ndarray:
    """신호 x_d = 종가_d > 그날까지 n일 단순 이동평균(종가_d 포함, 뒤쪽 창만). 이동평균이 없는 날(n일 미만)은 거짓(현금)."""
    c = np.asarray(close, float)
    ma = pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        return np.isfinite(ma) & (c > ma)


def dual_momentum_signal(close: np.ndarray, rate: np.ndarray, month_id: np.ndarray, months: int = DM_MONTHS) -> np.ndarray:
    """듀얼 모멘텀 신호: 매월 마지막 거래일 d 에 (종가_d / ``months`` 달 전 월말 종가 − 1) > 같은 기간 현금 복리 수익 이면 참.
    다음 월말 전날까지 유지(→ 체결 하루 밀림으로 다음 달 첫 거래일 시가 ~ 그 달 마지막 거래일 보유). 비교할 과거가 없으면 거짓."""
    c = np.asarray(close, float)
    me = mg1.month_ends(np.asarray(month_id))
    cum = np.cumprod(1.0 + np.asarray(rate, float))
    x = np.zeros(len(c), bool)
    for k in range(months, len(me)):
        d, d0 = me[k], me[k - months]
        nxt = me[k + 1] if k + 1 < len(me) else len(c)
        x[d:nxt] = (c[d] / c[d0] - 1.0) > (cum[d] / cum[d0] - 1.0)
    return x


def held(x: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """신호 → 보유: 날 t ∈ [i0, i1) 의 보유 p_t = x_{t−1} (d 종가 신호 → d+1 시가 매매). ``i0 ≥ 1``."""
    if i0 < 1:
        raise ValueError("i0 ≥ 1 이어야 한다(첫날 보유는 전날 신호)")
    return np.asarray(x[i0 - 1:i1 - 1], float)


def core_returns(ov: np.ndarray, iday: np.ndarray, rate: np.ndarray, p: np.ndarray, p_prev0: float = 0.0, *,
                 cost: float = COST_1W, fee: float = FEE_YR) -> np.ndarray:
    """코어(지수 ETF ↔ 현금) 날별 수익. ``p`` = (T,) 또는 (n, T) 보유 비율(그날 시가 매매 뒤), ``p_prev0`` = 창 첫날 직전 보유.

    r_t = (1 + p_{t−1}·ov_t + (1 − p_{t−1})·rate_t) × (1 − cost·|p_t − p_{t−1}|) × (1 + p_t·(iday_t − fee/252)) − 1,
    ov_t = 시가_t / 종가_{t−1} − 1(밤), iday_t = 종가_t / 시가_t − 1(장중). 현금 금리는 전일 밖 비율에 하루치(메타 세대 2 ``with_cash`` 와 같은 쪽)."""
    p = np.asarray(p, float)
    prev = np.concatenate([np.full(p.shape[:-1] + (1,), float(p_prev0)), p[..., :-1]], axis=-1)
    a = 1.0 + prev * ov + (1.0 - prev) * rate
    b = 1.0 - cost * np.abs(p - prev)
    c = 1.0 + p * (iday - fee / 252.0)
    return a * b * c - 1.0


def div_adjust(r: np.ndarray, p: np.ndarray, div: float = DIV_YR) -> np.ndarray:
    """배당 **고정 보정 줄**(설계 원문): 날별 수익에서 div/252 × (1 − p_t) 를 뺀다(가격 지수 대비 타이밍 유리분 제거). C0 는 그대로."""
    return np.asarray(r, float) - div / 252.0 * (1.0 - np.asarray(p, float))


def div_accrue(r: np.ndarray, p: np.ndarray, div: float = DIV_YR) -> np.ndarray:
    """배당 **일별 가산 줄**(정정): 시장 안 날마다 div/252 × p_t 를 더한다 — C0 는 총수익(TR) 근사가 된다."""
    return np.asarray(r, float) + div / 252.0 * np.asarray(p, float)


def timing_stats(p: np.ndarray) -> "tuple[float, float]":
    """(시장 안 비율 π = 평균 p, 전환 빈도 s = 연속한 날 중 p 가 바뀐 비율 — 첫날 진입은 세지 않는다). 마지막 축 기준."""
    p = np.asarray(p, float)
    if p.shape[-1] < 2:
        return float(p.mean()) if p.size else np.nan, 0.0
    return float(p.mean()), float((p[..., 1:] != p[..., :-1]).mean())


def markov_probs(pi: float, s: float) -> "tuple[float, float]":
    """정상 분포 π · 전환 빈도 s 인 2-상태 마르코프 체인의 (p01 밖→안, p10 안→밖). π·p10 = (1 − π)·p01 = s/2. [0, 1] 로 자른다."""
    p10 = s / (2.0 * pi) if pi > 0 else 0.0
    p01 = s / (2.0 * (1.0 - pi)) if pi < 1 else 0.0
    return float(min(max(p01, 0.0), 1.0)), float(min(max(p10, 0.0), 1.0))


def markov_paths(n: int, T: int, pi: float, s: float, rng: np.random.Generator) -> np.ndarray:
    """무작위 타이밍 보유 경로(n × T, 불리언). 첫 상태 ~ Bernoulli(π), 이후 :func:`markov_probs` 전이(단위 = 날 또는 달)."""
    p01, p10 = markov_probs(pi, s)
    u = rng.random((n, T))
    st = np.empty((n, T), bool)
    st[:, 0] = u[:, 0] < pi
    for t in range(1, T):
        st[:, t] = np.where(st[:, t - 1], u[:, t] >= p10, u[:, t] < p01)
    return st


def month_starts(mon: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(np.diff(np.asarray(mon))) + 1]


def markov_month_paths(n: int, p: np.ndarray, mon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """월말에만 바뀌는 규칙(C4)의 마르코프 널: 달 단위 보유(각 달 첫날 값)로 π·달 전환 빈도를 재고 달 체인을 만든 뒤 날로 펼친다."""
    ms = month_starts(mon)
    pm = np.asarray(p, float)[ms]
    pi, s = timing_stats(pm)
    P = markov_paths(n, len(ms), pi, s, rng)
    return np.repeat(P, np.diff(np.r_[ms, len(p)]), axis=1)


def circular_paths(p: np.ndarray, n: int, rng: np.random.Generator, min_shift: int = ROT_MIN) -> np.ndarray:
    """원형 이동 널(정정): 실제 보유 열을 오프셋 ∈ [min_shift, T − min_shift) 만큼 돌린다 — 체류 길이 분포·자기상관 보존, 날짜만 틀림."""
    p = np.asarray(p, float)
    T = len(p)
    ks = rng.integers(min_shift, T - min_shift, size=n)
    return p[(np.arange(T)[None, :] - ks[:, None]) % T]


def sharpe_rows(M: np.ndarray) -> np.ndarray:
    """행마다 월 (초과)수익 연환산 Sharpe = 평균 / 표준편차(ddof 0) × √12."""
    M = np.atleast_2d(np.asarray(M, float))
    sd = M.std(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 1e-12, M.mean(axis=1) / sd * np.sqrt(12.0), np.nan)


def monthly_rows(R: np.ndarray, mon: np.ndarray) -> np.ndarray:
    """날별 수익(… × T) → 달별 복리 수익(… × M)."""
    return np.expm1(np.add.reduceat(np.log1p(np.asarray(R, float)), month_starts(mon), axis=-1))


def excess_sharpe(r: np.ndarray, rate: np.ndarray, mon: np.ndarray) -> np.ndarray:
    """초과 Sharpe(정정) = (월 복리 수익 − 같은 달 현금 복리 수익)의 평균 / 표준편차 × √12. ``r`` 은 (T,) 또는 (n, T)."""
    return sharpe_rows(monthly_rows(np.atleast_2d(r), mon) - monthly_rows(rate, mon)[None, :])


def path_metrics(R: np.ndarray, mon: np.ndarray, rate: np.ndarray) -> dict:
    """날별 수익 경로들(n × T) → 행마다 CAGR(252/T)·MDD·Calmar(일별 경로, ``meta_fast.equity_stats`` 와 같은 정의)·초과 Sharpe."""
    R = np.atleast_2d(np.asarray(R, float))
    eq = np.concatenate([np.ones((R.shape[0], 1)), np.cumprod(1.0 + R, axis=1)], axis=1)
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cagr = np.where(eq[:, -1] > 0, eq[:, -1] ** (252.0 / R.shape[1]) - 1.0, -1.0)
        calmar = np.where(mdd < 0, cagr / np.abs(mdd), np.nan)
    return {"cagr": cagr, "mdd": mdd, "calmar": calmar, "sharpe": excess_sharpe(R, rate, mon)}


class MonthBoot:
    """일별 수익을 **달 블록 단위로** 재표본한 일별 경로(정정): 비원형 이동 블록 ``block`` 달(``meta_gen2_run.block_index``), ``n`` 벌.
    같은 객체를 여러 계열에 쓰면 같은 달 표본(짝지음·공통 난수). 길이가 다른 표본은 0 수익(자산·낙폭 불변)으로 채운다."""

    def __init__(self, mon: np.ndarray, block: int, n: int, seed: int):
        mon = np.asarray(mon) - np.asarray(mon)[0]
        ms = month_starts(mon)
        ends = np.r_[ms[1:], len(mon)]
        M = len(ms)
        self.T = len(mon)
        self.J = mg2.block_index(M, min(block, M), n, np.random.default_rng(seed))
        lens = (ends - ms)[self.J]
        self.L = lens.sum(axis=1)
        self.IDX = np.full((n, int(self.L.max())), self.T, np.int64)
        for i in range(n):
            ix = np.concatenate([np.arange(ms[j], ends[j]) for j in self.J[i]])
            self.IDX[i, :len(ix)] = ix
        self.mon = mon

    def calmar(self, r: np.ndarray, mdd_floor: float = 1e-9) -> np.ndarray:
        R = np.append(np.asarray(r, float), 0.0)[self.IDX]
        eq = np.cumprod(1.0 + R, axis=1)
        eq = np.concatenate([np.ones((len(eq), 1)), eq], axis=1)
        mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
        cagr = eq[:, -1] ** (252.0 / self.L) - 1.0
        return cagr / np.maximum(np.abs(mdd), mdd_floor)

    def sharpe(self, r: np.ndarray, rate: np.ndarray) -> np.ndarray:
        ex = monthly_rows(r, self.mon) - monthly_rows(rate, self.mon)
        return sharpe_rows(ex[self.J])


def paired_boot(ra: np.ndarray, rb: np.ndarray, rate: np.ndarray, B: MonthBoot) -> dict:
    """짝지은 달 블록 부트스트랩: P(초과 Sharpe a > b)·P(Calmar a > b)(일별 경로)·P(둘 다)와 Δ 5·50·95%."""
    ds = B.sharpe(ra, rate) - B.sharpe(rb, rate)
    dc = B.calmar(ra) - B.calmar(rb)
    q = lambda a: [float(v) for v in np.nanpercentile(a, [5, 50, 95])]  # noqa: E731
    return {"p_sharpe": float(np.mean(ds > 0)), "p_calmar": float(np.mean(dc > 0)), "p_both": float(np.mean((ds > 0) & (dc > 0))),
            "dsharpe": q(ds), "dcalmar": q(dc), "months": int(B.J.shape[1])}


def fit_lower(r: np.ndarray, B: MonthBoot, q: float = FIT_Q, floor: float = FIT_FLOOR) -> float:
    """학습창 적합도(정정) = 일별 경로 달 블록 부트스트랩 Calmar 의 ``q`` 백분위(MDD 하한 ``floor`` — 낙폭 0 표본의 분모 보호)."""
    return float(np.percentile(B.calmar(r, floor), q))


def pick_option(fits: "list[float]", switches: "list[int]") -> int:
    """적합도 최대 → 동률이면 전환 수 적은 쪽(정정) → 그래도 동률이면 앞. NaN 은 −∞."""
    f = np.where(np.isfinite(np.asarray(fits, float)), np.asarray(fits, float), -np.inf)
    best = np.flatnonzero(f == f.max())
    return int(best[np.argmin(np.asarray(switches)[best])])


def core_sat_mix(r_core: np.ndarray, p_core: np.ndarray, r_sat: np.ndarray, mon: np.ndarray, w: float = 1.0 - SAT_W,
                 cost: float = COST_1W) -> np.ndarray:
    """코어 ``w`` + 위성 (1 − w), 매달 첫날 비중 되돌림. 되돌림 비용 = cost × |표류한 코어 비중 − w| × 그날 코어 보유 p(코어가 현금이면 0 —
    현금끼리 옮김). 첫날은 배분만(비용 없음 — 두 다리 모두 이미 보유 중인 조각). NaN 수익은 0."""
    rc = np.nan_to_num(np.asarray(r_core, float))
    rs = np.nan_to_num(np.asarray(r_sat, float))
    pc = np.nan_to_num(np.asarray(p_core, float))
    out = np.empty(len(rc))
    v = np.array([w, 1.0 - w])
    for t in range(len(rc)):
        c = 0.0
        if t == 0 or mon[t] != mon[t - 1]:
            if t > 0:
                c = cost * abs(v[0] / v.sum() - w) * pc[t]
            v = np.array([w, 1.0 - w])
        tot = v.sum()
        out[t] = (1.0 - c) * (1.0 + (v[0] * rc[t] + v[1] * rs[t]) / tot) - 1.0
        v = v * (1.0 + np.array([rc[t], rs[t]]))
    return out


def p1_with_cash(nav: pd.Series, exposure: dict, rate: np.ndarray) -> np.ndarray:
    """P1 날별 수익 + (1 − 그 해 평균 노출) × 현금 날 금리 — 메타 세대 1 리뷰어 ``review_meta1/indiv.py`` 의 ``p1c`` 와 같은 구성.
    첫날 수익 = NAV_0 − 1, 결측 0. ``exposure`` = {연도: 평균 총 노출}(세대 1 ``peryear.csv``)."""
    r = nav.pct_change()
    r.iloc[0] = nav.iloc[0] - 1.0
    e = np.array([exposure[int(y)] for y in nav.index.year])
    return r.fillna(0.0).to_numpy() + (1.0 - e) * np.asarray(rate, float)


def criteria(st: dict, st0: dict, boot: dict, null_p95: float, cal_split: "tuple[float, float]",
             cal0_split: "tuple[float, float]", *, cagr_key: str = "cagr_div", cagr0_key: str = "cagr", joint: bool = True) -> dict:
    """판정 조건 (1)~(5) 산술(C0 대비). (1) P(Sharpe ∧ Calmar 둘 다 > C0) ≥ 0.8 (``joint`` — 리뷰 M2 의 엄격한 읽기; 아니면 각각 ≥ 0.8)
    · (2) MDD − MDD_C0 ≥ 10%p · (3) CAGR0 − CAGR(배당 줄) ≤ 2%p · (4) Calmar > 무작위 타이밍 널 Calmar p95 · (5) 두 분리 창 모두 Calmar ≥ C0."""
    fin = lambda x: x is not None and np.isfinite(x)  # noqa: E731
    c1 = bool(boot["p_both"] >= P_BOOT) if joint else bool(boot["p_sharpe"] >= P_BOOT and boot["p_calmar"] >= P_BOOT)
    c2 = bool(fin(st["mdd"]) and st["mdd"] - st0["mdd"] >= MDD_GAIN - 1e-12)
    c3 = bool(fin(st[cagr_key]) and st0[cagr0_key] - st[cagr_key] <= CAGR_LOSS + 1e-12)
    c4 = bool(fin(null_p95) and st["calmar"] > null_p95)
    c5 = bool(all(fin(a) and fin(b) and a >= b for a, b in zip(cal_split, cal0_split, strict=True)))
    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "all": c1 and c2 and c3 and c4 and c5}


def simplest_pass(rows: "list[dict]") -> "list[dict]":
    """통과 행 중 파라미터 수 최소 → 동률이면 전환/년이 적은 쪽(정정). 반환은 그 순서의 목록(첫째 = 고른 것)."""
    ok = [r for r in rows if r["crit"]["all"]]
    if not ok:
        return []
    k = min(r["n_params"] for r in ok)
    return sorted([r for r in ok if r["n_params"] == k], key=lambda r: r["sw_py"])


def drawdown_events(eq: np.ndarray, thr: float = EVENT_DD) -> "list[tuple[int, int, float]]":
    """자산 경로의 낙폭 사건: 새 고점 사이 구간마다 최저 낙폭이 −thr 보다 깊으면 (고점 행, 저점 행, 낙폭). 고점 행 = 그 구간 시작(고점일)."""
    eq = np.asarray(eq, float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    out = []
    new_hi = np.flatnonzero(eq >= peak - 1e-15)
    for a, b in zip(new_hi, np.r_[new_hi[1:], len(eq)], strict=True):
        if b - a > 1:
            t = a + int(np.argmin(dd[a:b]))
            if dd[t] < -thr:
                out.append((int(a), int(t), float(dd[t])))
    return out


# ---------------------------------------------------------------------------
# 자료 · 세계
# ---------------------------------------------------------------------------

def load_ra(dates: pd.DatetimeIndex) -> np.ndarray:
    """국면 원재료 RA(날 × 6): breadth_ma50·breakout_success20·nh_nl_ratio × KOSPI·KOSDAQ(재구축 판 npz, d 종가까지 자료)."""
    z = np.load(REGIME_NPZ, allow_pickle=False)
    feats = [str(x) for x in z["features"]]
    mk = [str(x) for x in z["markets"]]
    dt = pd.to_datetime(z["date"])
    vals = z["values"]

    def col(feat: str, market: str) -> np.ndarray:
        sel = z["market"] == mk.index(market)
        s = pd.Series(vals[sel, feats.index(feat)], index=dt[sel])
        return s[~s.index.duplicated()].reindex(dates).to_numpy(float)

    return np.column_stack([col(f, m) for f in RA4_FEATS for m in MARKETS2])


class World:
    """KOSPI200 일봉 · 현금 · 달 번호 · 폴드 행 · RA · 학습창 부트스트랩(폴드마다 하나 — 옵션·순열 널 공통)."""

    def __init__(self):
        from swing_it.validation.walkforward import FOLDS  # noqa: PLC0415

        x = pd.read_csv(INDEX_CSV, parse_dates=["date"])
        k = x[x["symbol"] == CORE_SYM].set_index("date").sort_index()
        self.dates = pd.DatetimeIndex(k.index)
        self.O, self.C = k["open"].to_numpy(float), k["close"].to_numpy(float)
        self.ov = np.r_[np.nan, self.O[1:] / self.C[:-1] - 1.0]
        self.iday = self.C / self.O - 1.0
        self.rate = mg2.cash_daily(self.dates, CD91)
        self.mon = mg2.month_ids(self.dates)
        d = self.dates
        self.i0 = int(d.searchsorted(pd.Timestamp(W_START)))
        self.i_end = int(d.searchsorted(pd.Timestamp(W_END), "right"))
        self.i_oos = int(d.searchsorted(pd.Timestamp(W_OOS)))
        self.folds = FOLDS
        self.fold_rows = []
        for f in FOLDS:
            tr = (int(d.searchsorted(pd.Timestamp(f.train_lo))), int(d.searchsorted(pd.Timestamp(f.train_hi))))
            te = (int(d.searchsorted(pd.Timestamp(f.test_lo))), min(int(d.searchsorted(pd.Timestamp(f.test_hi))), self.i_end))
            self.fold_rows.append((tr, te))
        assert self.fold_rows[0][1][0] == self.i_oos and self.fold_rows[-1][1][1] == self.i_end
        self.fit_boot = [MonthBoot(self.mon[a:b], FIT_BLOCK, FIT_N, 0) for (a, b), _ in self.fold_rows]
        self.RA = load_ra(self.dates)
        self.x_c0 = np.ones(len(d), bool)
        self.x_ma = {n: ma_signal(self.C, n) for n in MA_GRID}
        self.x_dm = dual_momentum_signal(self.C, self.rate, self.mon)

    def run(self, p: np.ndarray, i0: int, i1: int, p_prev0: float = 0.0, tax: bool = False) -> np.ndarray:
        rate = self.rate[i0:i1] * ((1.0 - TAX) if tax else 1.0)
        return core_returns(self.ov[i0:i1], self.iday[i0:i1], rate, p, p_prev0)

    def after_tax(self, p: np.ndarray, i0: int, i1: int) -> np.ndarray:
        """세후 줄: 현금 이자 × (1 − 15.4%) + 배당 일별 가산 × (1 − 15.4%), 가격 차익 비과세."""
        return div_accrue(self.run(p, i0, i1, tax=True), p, DIV_YR * (1.0 - TAX))

    def fitted(self, options_of) -> dict:
        """폴드마다 ``options_of(k) → [(라벨, 신호)]`` 중 학습창 적합도 최대(동률 = 학습창 전환 적은 쪽) → 시험창 보유 → 이어 붙인 OOS."""
        p = np.zeros(self.i_end - self.i_oos)
        choices = []
        for k, (tr, te) in enumerate(self.fold_rows):
            opts = options_of(k)
            fits, sws = [], []
            for _, x in opts:
                ph = held(x, *tr)
                fits.append(fit_lower(self.run(ph, *tr), self.fit_boot[k]))
                sws.append(int((ph[1:] != ph[:-1]).sum()))
            j = pick_option(fits, sws)
            p[te[0] - self.i_oos:te[1] - self.i_oos] = held(opts[j][1], *te)
            choices.append({"fold": k, "test": self.folds[k].test_lo[:4], "labels": [lab for lab, _ in opts], "fits": fits,
                            "sws": sws, "j": j})
        return {"p": p, "r": self.run(p, self.i_oos, self.i_end), "choices": choices}

    def c1(self) -> dict:
        return self.fitted(lambda k: [(f"MA{n}", self.x_ma[n]) for n in MA_GRID])

    def c2(self, RA: "np.ndarray | None" = None) -> dict:
        RA = self.RA if RA is None else RA

        def opts(k):
            tr = np.arange(*self.fold_rows[k][0])
            s = mg1.zscore_mean(RA, tr)
            thr = mg1.gate_thresholds(s, tr)
            return [("없음" if q is None else f"q{q} ({t:+.2f})", mg1.gate_open(s, t)) for q, t in zip(GATE_Q, thr, strict=True)]
        return self.fitted(opts)

    def fake_ra(self, rng: np.random.Generator) -> np.ndarray:
        """가짜 RA(메타 세대 1 정정 M1 'shuffle' 과 같다): 첫 학습창 시작 ~ OOS 끝 행을 **하나의 달 블록 순열**로 섞는다(6열 같이)."""
        lo, hi = self.fold_rows[0][0][0], self.i_end
        ix = mf.month_block_perm(self.mon, lo, hi, rng)
        a = self.RA.copy()
        a[lo:hi] = self.RA[ix]
        return a


_C: dict = {}


def worker_init() -> None:
    _C["W"] = World()


def perm_task(reps: "list[int]") -> "list[dict]":
    """C2 순열 널 묶음(작업자): 가짜 RA → z·문턱 재적합 → 이어 붙인 OOS 지표."""
    W = _C["W"]
    a, b = W.i_oos, W.i_end
    out = []
    for rep in reps:
        res = W.c2(W.fake_ra(np.random.default_rng([SEED, 2, rep])))
        m = path_metrics(res["r"], W.mon[a:b], W.rate[a:b])
        out.append({"rep": rep, "calmar": float(m["calmar"][0]), "sharpe": float(m["sharpe"][0]), "mdd": float(m["mdd"][0]),
                    "cagr": float(m["cagr"][0]), "tim": float(res["p"].mean()), "thr_j": [c["j"] for c in res["choices"]]})
    return out


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------

def wstats(r: np.ndarray, p: np.ndarray, dates: pd.DatetimeIndex, rate: np.ndarray, rt: "np.ndarray | None" = None) -> dict:
    """한 창: CAGR·MDD·Calmar(일별)·초과 Sharpe(+ 원시)·시장 안·연 전환 + 배당 고정 보정 줄·일별 가산 줄(1.8%, 1.5%·2.2% CAGR)·세후 줄."""
    from krx_quant_core.stats.metrics import ann_sharpe  # noqa: PLC0415

    r, p, rate = np.asarray(r, float), np.asarray(p, float), np.asarray(rate, float)
    mon = mg2.month_ids(dates)
    s = mf.equity_stats(r)
    s["sharpe"] = float(excess_sharpe(r, rate, mon)[0])
    s["sharpe_raw"] = ann_sharpe(mg1.monthly_compound(r, mon), 12)
    pi, sw = timing_stats(p)
    s.update({"tim": pi, "sw_py": sw * (len(p) - 1) / (len(p) / 252.0) if len(p) > 1 else 0.0})
    e = mf.equity_stats(div_adjust(r, p))
    s.update({"cagr_div": e["cagr"], "mdd_div": e["mdd"], "calmar_div": e["calmar"]})
    ra = div_accrue(r, p)
    e = mf.equity_stats(ra)
    s.update({"cagr_acc": e["cagr"], "mdd_acc": e["mdd"], "calmar_acc": e["calmar"], "sharpe_acc": float(excess_sharpe(ra, rate, mon)[0])})
    for y in DIV_SENS:
        s[f"cagr_acc{int(y * 1000)}"] = mf.equity_stats(div_accrue(r, p, y))["cagr"]
    if rt is not None:
        e = mf.equity_stats(rt)
        s.update({"cagr_tax": e["cagr"], "mdd_tax": e["mdd"], "calmar_tax": e["calmar"]})
    s.update({"start": str(dates[0].date()), "end": str(dates[-1].date())})
    return s


def null_metrics(W: World, P: np.ndarray, i0: int, i1: int) -> dict:
    """널 보유 경로 묶음 → 가격 줄·배당 가산 줄 지표 + 실현 π·전환 빈도."""
    R = W.run(P, i0, i1)
    m = path_metrics(R, W.mon[i0:i1], W.rate[i0:i1])
    ma = path_metrics(div_accrue(R, P), W.mon[i0:i1], W.rate[i0:i1])
    m.update({"calmar_acc": ma["calmar"], "pi_mean": float(P.mean()), "s_mean": float((P[:, 1:] != P[:, :-1]).mean())})
    return m


def qs(a, x):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, x)) if np.isfinite(a).any() else np.nan


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def plot_equity(panels: "list[tuple[str, pd.DatetimeIndex, dict]]", path: str) -> "str | None":
    """창별 자산(로그, 위)·낙폭(아래) 작은 그림. 색은 절차를 따른다(참조 팔레트 1~6번, 검증 통과 — 대비 3:1 미만 색은 범례·결과 표가 보완).
    회색 띠 = 2026-06-22 ~ 07-30 급락. 그림 글자는 영어(이 기계에 한글 글꼴이 없다)."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: PLC0415
    except ImportError:
        return None
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, len(panels), figsize=(5.6 * len(panels), 6.4), sharex="col", facecolor="#fcfcfb",
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    c0, c1 = pd.Timestamp(CRASH[0]), pd.Timestamp(CRASH[1])
    for j, (title, d, series) in enumerate(panels):
        a0, a1 = axes[0, j], axes[1, j]
        for nm, r in series.items():
            eq = np.cumprod(1.0 + r)
            lab = nm.replace("·", ".")
            a0.plot(d, eq, color=COLORS.get(nm, "#52514e"), lw=2.0 if nm == "C0" else 1.4, label=lab)
            a1.plot(d, eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1.0, color=COLORS.get(nm, "#52514e"), lw=1.2)
        a0.set_yscale("log")
        a0.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
        a0.yaxis.set_minor_formatter(NullFormatter())
        a1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        a0.set_title(title, fontsize=10, color="#0b0b0b")
        a1.set_ylabel("drawdown" if j == 0 else "", fontsize=9, color="#52514e")
        a0.set_ylabel("equity (log)" if j == 0 else "", fontsize=9, color="#52514e")
        for ax in (a0, a1):
            ax.set_facecolor("#fcfcfb")
            if d[0] <= c0 <= d[-1]:
                ax.axvspan(c0, c1, color="#e6e5e0", lw=0, zorder=0)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
        a0.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("META GEN 4 - KOSPI200 core: buy-and-hold vs timing vs core-satellite (price index, costs incl.; "
                 "grey band = 2026-06-22..07-30 crash)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--nulls", type=int, default=N_RAND)
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--boots", type=int, default=N_BOOT)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--check-p1c", default=None, help="세대 1 리뷰어 indiv.csv(p1c 열) — 위성 NAV 재현 대조(선택)")
    args = ap.parse_args(argv)
    debug = os.path.normpath(args.out) != os.path.normpath(OUT_DIR)
    if (args.nulls != N_RAND or args.perms != N_PERM or args.boots != N_BOOT) and not debug:
        ap.error("축소 실행은 --out 을 탐색 디렉터리 밖으로 둘 때만")
    sub = os.path.join(args.out, SUB)
    os.makedirs(sub, exist_ok=True)
    t0 = time.time()
    W = World()
    D = W.dates
    _log(f"KOSPI200 {D[0].date()} ~ {D[-1].date()} ({len(D)}일), 전체 창 {D[W.i0].date()} ~ {D[W.i_end - 1].date()}", t0)

    # --- C2 순열 널(작업자 — 먼저 띄워 두고 실제 계산과 겹친다) ---
    ctx = mp.get_context("fork")
    ex = cf.ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx, initializer=worker_init)
    chunks = [list(range(a, min(a + 10, args.perms))) for a in range(0, args.perms, 10)]
    perm_f = [ex.submit(perm_task, c) for c in chunks]

    # --- 실제 절차 ---
    full = (W.i0, W.i_end)
    oos = (W.i_oos, W.i_end)
    ser = {}                                    # 이름 → {a: 창 시작 행, p: 보유, r: 수익, rt: 세후 줄}
    for nm, x in (("C0", W.x_c0), ("C1f", W.x_ma[MA_FIXED]), ("C4", W.x_dm)):
        p = held(x, *full)
        ser[nm] = {"a": full[0], "p": p, "r": W.run(p, *full), "rt": W.after_tax(p, *full)}
    c1, c2 = W.c1(), W.c2()
    for nm, res in (("C1", c1), ("C2", c2)):
        ser[nm] = {"a": oos[0], "p": res["p"], "r": res["r"], "rt": W.after_tax(res["p"], *oos)}
    _log("실제 절차 끝", t0)

    def sl(nm: str, lo: str, hi: str) -> dict:
        s = ser[nm]
        d = D[s["a"]:s["a"] + len(s["r"])]
        m = np.asarray((d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi)))
        i = s["a"] + int(np.flatnonzero(m)[0]) if m.any() else None
        return {"r": s["r"][m], "p": s["p"][m], "rt": None if s.get("rt") is None else s["rt"][m], "d": d[m], "i0": i,
                "rate": W.rate[s["a"]:s["a"] + len(s["r"])][m]}

    WINS = {"full": ("2017-01-01", W_END, "전체 2017-01 ~ 2026-09"), "pre": ("2017-01-01", W_PRE_END, "2017~2024"),
            "post": (W_POST, W_END, "2025-01 ~ 2026-09"), "oos": (W_OOS, W_END, "OOS 2020-01 ~ 2026-09"),
            "oos_pre": (W_OOS, W_PRE_END, "OOS 2020~2024"),
            "full_pc": ("2017-01-01", PRECRASH_END, "기술: 2017-01 ~ 2026-06-19 (급락 전)"),
            "oos_pc": (W_OOS, PRECRASH_END, "기술: OOS 2020-01 ~ 2026-06-19 (급락 전)")}
    stats = {}
    for nm in ser:
        for wk, (lo, hi, _) in WINS.items():
            q = sl(nm, lo, hi)
            if len(q["r"]) and q["d"][0] <= pd.Timestamp(lo) + pd.Timedelta(days=7):
                stats[(nm, wk)] = wstats(q["r"], q["p"], q["d"], q["rate"], q["rt"])

    # --- 판정: 고정 = 전체(분리 = 2017~24 · 2025~26), 적합 = OOS(분리 = 2020~24 · 2025~26) ---
    main_win = {"C1f": "full", "C4": "full", "C1": "oos", "C2": "oos"}
    split_win = {"full": ("pre", "post"), "oos": ("oos_pre", "post")}
    npars = {a: b for a, b, _ in PROCS}
    bcache = {}
    boots, nulls, rows = {}, {}, []
    for k, nm in enumerate(TIMING):
        wk = main_win[nm]
        lo, hi, _ = WINS[wk]
        q, q0 = sl(nm, lo, hi), sl("C0", lo, hi)
        assert (q["d"] == q0["d"]).all()
        mon = mg2.month_ids(q["d"])
        for blk in (BLOCK_M, BLOCK_ALT):
            if (wk, blk) not in bcache:
                bcache[(wk, blk)] = MonthBoot(mon, blk, args.boots, SEED + blk)
        B3, B12 = bcache[(wk, BLOCK_M)], bcache[(wk, BLOCK_ALT)]
        boots[nm] = {"b3": paired_boot(q["r"], q0["r"], q["rate"], B3), "b12": paired_boot(q["r"], q0["r"], q["rate"], B12),
                     "acc": paired_boot(div_accrue(q["r"], q["p"]), div_accrue(q0["r"], q0["p"]), q["rate"], B3)}
        i0, i1 = q["i0"], q["i0"] + len(q["r"])
        rng = np.random.default_rng([SEED, 1, k])
        Pm = (markov_month_paths(args.nulls, q["p"], mon, rng) if nm == "C4"
              else markov_paths(args.nulls, len(q["p"]), *timing_stats(q["p"]), rng)).astype(float)
        Pc = circular_paths(q["p"], args.nulls, rng)
        pi, s = timing_stats(q["p"])
        nulls[nm] = {"markov": null_metrics(W, Pm, i0, i1), "circ": null_metrics(W, Pc, i0, i1), "pi": pi, "s": s,
                     "unit": "달" if nm == "C4" else "날"}
        st, st0 = stats[(nm, wk)], stats[("C0", wk)]
        sa, sb = split_win[wk]
        cs = (stats[(nm, sa)]["calmar"], stats[(nm, sb)]["calmar"])
        cs0 = (stats[("C0", sa)]["calmar"], stats[("C0", sb)]["calmar"])
        csa = (stats[(nm, sa)]["calmar_acc"], stats[(nm, sb)]["calmar_acc"])
        cs0a = (stats[("C0", sa)]["calmar_acc"], stats[("C0", sb)]["calmar_acc"])
        nm95, nc95 = qs(nulls[nm]["markov"]["calmar"], 95), qs(nulls[nm]["circ"]["calmar"], 95)
        acc_st = {**st, "calmar": st["calmar_acc"], "mdd": st["mdd_acc"]}
        acc0 = {**st0, "mdd": st0["mdd_acc"]}
        rows.append({"proc": nm, "n_params": npars[nm], "win": wk, "sw_py": st["sw_py"], "split": cs, "split0": cs0,
                     "null_p95": nm95, "circ_p95": nc95,
                     "null_pct": mg1.null_pct(st["calmar"], nulls[nm]["markov"]["calmar"]),
                     "circ_pct": mg1.null_pct(st["calmar"], nulls[nm]["circ"]["calmar"]),
                     "crit": criteria(st, st0, boots[nm]["b3"], nm95, cs, cs0),
                     "crit_each": criteria(st, st0, boots[nm]["b3"], nm95, cs, cs0, joint=False),
                     "crit_circ": criteria(st, st0, boots[nm]["b3"], nc95, cs, cs0),
                     "crit_b12": criteria(st, st0, boots[nm]["b12"], nm95, cs, cs0),
                     "crit_acc": criteria(acc_st, acc0, boots[nm]["acc"], qs(nulls[nm]["markov"]["calmar_acc"], 95), csa, cs0a,
                                          cagr_key="cagr_acc", cagr0_key="cagr_acc"),
                     "c3_sens": {y: bool(stats[("C0", wk)][f"cagr_acc{int(y * 1000)}"] - st[f"cagr_acc{int(y * 1000)}"] <= CAGR_LOSS)
                                 for y in DIV_SENS}})
    _log("부트스트랩 · 무작위 타이밍 널 끝", t0)

    # --- 위성 C3 ---
    N = pd.read_csv(P1_NAV, index_col=0, parse_dates=True)[P1_COL].dropna()
    PY = pd.read_csv(P1_PERYEAR).drop_duplicates()
    expo = {int(r.year): float(r.exposure) for r in PY[PY["proc"] == "P1"].itertuples()}
    sat_d = pd.DatetimeIndex(N.index)
    ia, ib = int(D.searchsorted(sat_d[0])), int(D.searchsorted(sat_d[-1], "right"))
    assert (D[ia:ib] == sat_d).all(), "P1 NAV 날짜 ≠ 지수 날짜"
    rate_s = W.rate[ia:ib]
    r_sat = p1_with_cash(N, expo, rate_s)
    p1c_check = None
    if args.check_p1c:
        ref = pd.read_csv(args.check_p1c, index_col=0, parse_dates=True)["p1c"].reindex(sat_d)
        p1c_check = float(np.nanmax(np.abs(ref.to_numpy() - r_sat)))
    mon_s = mg2.month_ids(sat_d)
    Bs = MonthBoot(mon_s, BLOCK_M, args.boots, SEED + 7)
    Bs12 = MonthBoot(mon_s, BLOCK_ALT, args.boots, SEED + 8)
    lo_s, hi_s = str(sat_d[0].date()), str(sat_d[-1].date())
    q0 = sl("C0", lo_s, hi_s)
    cash_mix = core_sat_mix(q0["r"], q0["p"], rate_s, mon_s)              # 대조: 80% C0 + 20% 현금
    sat = {}
    for core in ("C0",) + TIMING:
        q = sl(core, lo_s, hi_s)
        assert (q["d"] == sat_d).all()
        mix = core_sat_mix(q["r"], q["p"], r_sat, mon_s)
        mix_acc = core_sat_mix(div_accrue(q["r"], q["p"]), q["p"], r_sat, mon_s)
        pmix = (1.0 - SAT_W) * q["p"]
        ser[f"C3·{core}"] = {"a": ia, "p": pmix, "r": mix, "rt": None}
        sat[core] = {"st": wstats(mix, pmix, sat_d, rate_s), "core_st": wstats(q["r"], q["p"], sat_d, rate_s),
                     "boot": paired_boot(mix, q["r"], rate_s, Bs), "boot_c0": paired_boot(mix, q0["r"], rate_s, Bs),
                     "boot12": paired_boot(mix, q["r"], rate_s, Bs12),
                     "boot_acc": paired_boot(mix_acc, div_accrue(q["r"], q["p"]), rate_s, Bs),
                     "boot_cash": paired_boot(mix, cash_mix, rate_s, Bs) if core == "C0" else None}
    cash_st = wstats(cash_mix, 0.8 * q0["p"], sat_d, rate_s)
    sat_st = wstats(r_sat, np.ones(len(r_sat)), sat_d, rate_s)
    corr = float(np.corrcoef(mg1.monthly_compound(r_sat, mon_s), mg1.monthly_compound(q0["r"], mon_s))[0, 1])
    ex25 = np.asarray(sat_d.year != 2025)
    sat_ex25 = {nm: float(excess_sharpe(r[ex25], rate_s[ex25], mg2.month_ids(sat_d[ex25]))[0])
                for nm, r in (("C0", q0["r"]), ("C3·C0", ser["C3·C0"]["r"]), ("80/20 현금", cash_mix))}
    _log("위성 끝", t0)

    # --- 사건표(C0 낙폭 > 15%) ---
    qf = sl("C0", W_START, W_END)
    eq0 = np.cumprod(1.0 + qf["r"])
    events = []
    for a, t, dd in drawdown_events(eq0):
        da, dt_ = qf["d"][a], qf["d"][t]
        row = {"peak": str(da.date()), "trough": str(dt_.date()), "days": int(t - a), "c0": dd}
        for nm in TIMING:
            s = ser[nm]
            d = D[s["a"]:s["a"] + len(s["r"])]
            if da < d[0]:
                row[nm] = None
                continue
            j0, j1 = int(d.searchsorted(da)), int(d.searchsorted(dt_)) + 1
            e = np.cumprod(1.0 + s["r"][j0 + 1:j1])          # 고점일 종가 기준 → 다음날부터의 경로
            dd_r = float((e / np.maximum.accumulate(np.r_[1.0, e])[1:] - 1.0).min()) if len(e) else 0.0
            row[nm] = {"expo": float(s["p"][j0]), "dd": min(dd_r, 0.0), "tim": float(s["p"][j0 + 1:j1].mean()) if j1 > j0 + 1 else np.nan}
        events.append(row)

    # --- C2 순열 널 회수 ---
    perm = [r for f in perm_f for r in f.result()]
    ex.shutdown()
    pdf = pd.DataFrame(perm).sort_values("rep")
    st_c2 = stats[("C2", "oos")]
    perm_sum = {"p95": qs(pdf["calmar"], 95), "p50": qs(pdf["calmar"], 50), "pct": mg1.null_pct(st_c2["calmar"], pdf["calmar"]),
                "sh_p95": qs(pdf["sharpe"], 95), "sh_pct": mg1.null_pct(st_c2["sharpe"], pdf["sharpe"]),
                "mdd_p95": qs(pdf["mdd"], 95), "beat": bool(st_c2["calmar"] > qs(pdf["calmar"], 95))}
    _log(f"C2 순열 널 {args.perms}벌 회수", t0)

    # --- 산술 결과 ---
    passed = [r["proc"] for r in rows if r["crit"]["all"]]
    simp = [r["proc"] for r in simplest_pass(rows)]
    sat_rows = []
    for core in ("C0",) + TIMING:
        applies = core == "C0" or (bool(simp) and core == simp[0])
        sat_rows.append({"core": core, "applies": applies, "p_sharpe": sat[core]["boot"]["p_sharpe"],
                         "adds": bool(sat[core]["boot"]["p_sharpe"] >= P_BOOT)})

    # --- 산출물 ---
    pd.DataFrame([{"proc": nm, "win": wk, **{k: v for k, v in s.items() if not isinstance(v, (list, dict))}}
                  for (nm, wk), s in stats.items()]).to_csv(os.path.join(sub, "summary.csv"), index=False)
    pd.DataFrame([{"proc": r["proc"], "win": r["win"], "n_params": r["n_params"], **{f"crit_{k}": v for k, v in r["crit"].items()},
                   "crit_each_all": r["crit_each"]["all"], "crit_circ_all": r["crit_circ"]["all"], "crit_b12_all": r["crit_b12"]["all"],
                   "crit_acc_all": r["crit_acc"]["all"], "null_markov_calmar_p95": r["null_p95"], "null_circ_calmar_p95": r["circ_p95"],
                   **{f"b3_{k}": v for k, v in boots[r["proc"]]["b3"].items() if not isinstance(v, list)},
                   **{f"b12_{k}": v for k, v in boots[r["proc"]]["b12"].items() if not isinstance(v, list)}}
                  for r in rows]).to_csv(os.path.join(sub, "decision.csv"), index=False)
    pd.DataFrame([{"proc": nm, "null": kind, "calmar": c, "sharpe": s, "mdd": m}
                  for nm in nulls for kind in ("markov", "circ")
                  for c, s, m in zip(nulls[nm][kind]["calmar"], nulls[nm][kind]["sharpe"], nulls[nm][kind]["mdd"], strict=True)]).to_csv(
        os.path.join(sub, "null_random.csv"), index=False)
    pdf.to_csv(os.path.join(sub, "null_c2_perm.csv"), index=False)
    folds = [{"proc": nm, "fold": c["fold"], "test": c["test"], "pick": c["labels"][c["j"]],
              **{f"fit[{lab.split(' ')[0]}]": f for lab, f in zip(c["labels"], c["fits"], strict=True)},
              **{f"sw[{lab.split(' ')[0]}]": s for lab, s in zip(c["labels"], c["sws"], strict=True)}}
             for nm, res in (("C1", c1), ("C2", c2)) for c in res["choices"]]
    pd.DataFrame(folds).to_csv(os.path.join(sub, "folds.csv"), index=False)
    pd.DataFrame([{k: (v if not isinstance(v, dict) else f"{v['expo']:.0f}/{v['dd']:.4f}/{v['tim']:.3f}") for k, v in e.items()}
                  for e in events]).to_csv(os.path.join(sub, "events.csv"), index=False)
    nav = pd.DataFrame(index=D[W.i0:W.i_end])
    for nm, s in ser.items():
        idx = D[s["a"]:s["a"] + len(s["r"])]
        nav[nm] = pd.Series(np.cumprod(1.0 + s["r"]), index=idx).reindex(nav.index)
        nav[f"{nm} 보유"] = pd.Series(s["p"], index=idx).reindex(nav.index)
    nav.to_csv(os.path.join(sub, "nav.csv"))
    yrows = []
    for nm, s in ser.items():
        d = D[s["a"]:s["a"] + len(s["r"])]
        for y in sorted(set(d.year)):
            m = np.asarray(d.year == y)
            e = mf.equity_stats(s["r"][m])
            yrows.append({"proc": nm, "year": int(y), "ret": e["total"], "mdd": e["mdd"], "tim": float(np.mean(s["p"][m]))})
    ydf = pd.DataFrame(yrows)
    ydf.to_csv(os.path.join(sub, "peryear.csv"), index=False)

    def seg(nms, lo, hi):
        out = {nm: sl(nm, lo, hi) for nm in nms}
        return next(iter(out.values()))["d"], {nm: q["r"] for nm, q in out.items()}
    d_full, s_full = seg(("C0", "C1f", "C4"), W_START, W_END)
    d_oos, s_oos = seg(("C0", "C1f", "C4", "C1", "C2"), W_OOS, W_END)
    d_sat, s_sat = seg(("C0", "C3·C0"), lo_s, hi_s)
    fig = plot_equity([("Fixed rules, full 2017-01 ~ 2026-09", d_full, s_full),
                       ("Stitched OOS 2020-01 ~ 2026-09 (C1, C2 fitted per fold)", d_oos, s_oos),
                       ("Core-satellite 2020-01 ~ 2026-04 (P1 NAV range)", d_sat, s_sat)], os.path.join(sub, "equity.png"))
    led = write_ledger_rows(sub)
    elapsed = time.time() - t0
    L = render(locals())
    with open(os.path.join(args.out, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n=== 메타 세대 4 — {elapsed:.0f}초 · 통과(1–5): {', '.join(passed) or '없음'} · 고른 코어: {simp[0] if simp else 'C0'} · "
          + " · ".join(f"위성[{s['core']}] P(초과 Sharpe) {s['p_sharpe']:.2f}" for s in sat_rows) + " ===")
    print(f"→ {os.path.join(args.out, RESULTS_MD)}")
    return 0


def write_ledger_rows(sub: str) -> int:
    """원장 ``meta_gen4/TRIALS.jsonl`` — 절차 한 줄씩(결과 숫자는 넣지 않는다)."""
    path = os.path.join(sub, "TRIALS.jsonl")
    seen = g1.read_fingerprints(path)
    base = {"design": DESIGN_DOC, "asset": f"{CORE_SYM} 가격 지수(네이버)", "cost_1w": COST_1W, "fee_yr": FEE_YR, "div_yr": DIV_YR,
            "exec": "d 종가 신호 → d+1 시가", "cash": "CD91 근사(기억값)", "sharpe": "초과(현금 차감)", "calmar": "일별 경로"}
    fit = {"fit": f"학습창 일별 경로 월 블록({FIT_BLOCK}) {FIT_N}회 Calmar p{FIT_Q:g}", "tie": "전환 적은 쪽", "folds": "FOLDS"}
    cfgs = [{"procedure": "C0"}, {"procedure": "C1f", "ma": MA_FIXED}, {"procedure": "C4", "months": DM_MONTHS},
            {"procedure": "C1", "ma_grid": list(MA_GRID), **fit},
            {"procedure": "C2", "ra": list(RA4_FEATS), "markets": list(MARKETS2), "gate_q": list(GATE_Q), **fit}]
    cfgs += [{"procedure": f"C3·{c}", "satellite": "P1 + 유휴 현금 금리", "sat_w": SAT_W, "rebalance": "월"} for c in ("C0",) + TIMING]
    rows = [{"config": {**base, **c}, "note": "메타 세대 4 — 개인 코어 운용"} for c in cfgs]
    return g1.write_ledger(path, rows, seen)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _ck(v) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def _pp(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.1f}%p"


def _q3(b: dict, key: str) -> str:
    return " · ".join(_f(q) for q in b[key])


def render(v: dict) -> "list[str]":
    args, stats, rows, boots, nulls = v["args"], v["stats"], v["rows"], v["boots"], v["nulls"]
    sat, sat_rows, perm_sum, pdf, WINS = v["sat"], v["sat_rows"], v["perm_sum"], v["pdf"], v["WINS"]
    passed, simp, c1, c2, events = v["passed"], v["simp"], v["c1"], v["c2"], v["events"]
    L = [f"# 메타 세대 4 — 개인 코어 운용: KOSPI200 타이밍 × 코어-위성 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "설계 `META_GEN4.md`(2cb127b + 정정 412b012, 둘 다 결과 보기 전)를 계산한 숫자다. ✓/✗ 는 미리 적은 문턱과의 **산술 비교**이고 판정은 사람이 한다. "
         "고정 규칙(C1f·C4)은 전 구간이 표본 내다 — MA200·12개월 듀얼 모멘텀은 문헌에서 온 규칙이라 이 자료로 고른 것은 아니지만 이 자료로 '다시 본' 것이다.", "",
         f"재현: `uv run python research/experiments/meta_gen4_run.py --workers {args.workers}` · 실행 {v['elapsed']:.0f}초 · "
         f"무작위 타이밍 널 {args.nulls}벌 × 2종/규칙 · C2 순열 널 {args.perms}벌 · 부트스트랩 {args.boots}벌 · DB 접속 없음.", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 출력 `{args.out}`. 측정으로 쓰지 않는다.", ""]

    L += ["## 먼저 읽을 것 — 해석 한계(정정에 미리 적은 문장)", "",
          "**2026-06-22 ~ 07-30 KOSPI200 −40.9%(27거래일)가 전 구간 MDD 를 결정한다.** 이 표본의 큰 하락은 느린 약세 둘(2018·2022)과 빠른 급락 둘"
          "(2020-03·2026-07)이다. 판정 (2)(4)(5)의 Calmar·MDD 는 사실상 사건 넷 남짓으로 정해진다. **기각이면 문구는 \"이 표본에서 매수보유를 버릴 근거 없음\" 이지 "
          "\"타이밍이 안 된다\" 가 아니다.** 2025-01 ~ 2026-09 분리 창은 21개월 · 한 국면(CAGR +100% 안팎, 연 변동성 65% 안팎)이라 Calmar 비교에 짧다(리뷰 M6). "
          "아래 사건표와 '급락 전' 기술 줄을 함께 본다.", ""]

    L += ["## 판정 산술 (META_GEN4.md '판정 — 개인에게 맞는 코어 운용' + 정정)", "",
          "C0(KOSPI200 매수보유, 가격 지수) 대비 **모두**: (1) 짝지은 달 블록 부트스트랩(일별 경로를 3개월 블록으로 재표본, "
          f"{args.boots}벌)에서 P(초과 Sharpe > C0 **그리고** Calmar > C0) ≥ 0.8 — 리뷰 M2 의 엄격한 읽기(동시); '각각 ≥ 0.8' 읽기는 참고 열 · "
          "(2) MDD 가 10%p 이상 얕음 · (3) CAGR_C0 − CAGR(**배당 고정 보정 줄**) ≤ 2%p · (4) Calmar > 무작위 타이밍 널(마르코프) Calmar p95 · "
          "(5) 분리 창 둘 다 Calmar ≥ C0.  ",
          "**창:** 고정 규칙(C1f·C4) = 전체 2017-01 ~ 2026-09, 분리 = 2017~2024 · 2025-01 ~ 2026-09. 적합 규칙(C1·C2) = 이어 붙인 OOS 2020-01 ~ 2026-09, "
          "분리 = **OOS 2020~2024** · 2025-01 ~ 2026-09(리뷰 M5 — OOS 가 2020 부터). C0 는 같은 창.", "",
          "| 절차 | 파라미터 | 창 | (1) P(SR) · P(Calmar) · **P(둘 다)** | (2) ΔMDD | (3) CAGR_C0 − CAGR(고정 보정) | (4) Calmar · 마르코프 p95 (백분위) | "
          "(5) Calmar 분리 창 · C0 | **통과(1–5)** | 참고: (1) 각각 · (1) 12개월 블록 · (4) 원형 이동 · 배당 가산 줄로 전부 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        nm, wk = r["proc"], r["win"]
        st, st0, b, c = stats[(nm, wk)], stats[("C0", wk)], boots[nm]["b3"], r["crit"]
        L.append(f"| **{nm}** | {r['n_params']} | {WINS[wk][2]} | {b['p_sharpe']:.2f} · {b['p_calmar']:.2f} · **{b['p_both']:.2f}** {_ck(c['c1'])} | "
                 f"{_pp(st['mdd'] - st0['mdd'])} ({_pct(st['mdd'])} vs {_pct(st0['mdd'])}) {_ck(c['c2'])} | "
                 f"{_pp(st0['cagr'] - st['cagr_div'])} ({_pct(st0['cagr'])} − {_pct(st['cagr_div'])}) {_ck(c['c3'])} | "
                 f"{_f(st['calmar'])} · {_f(r['null_p95'])} ({r['null_pct'] * 100:.0f}%) {_ck(c['c4'])} | "
                 f"{_f(r['split'][0])} · {_f(r['split0'][0])} / {_f(r['split'][1])} · {_f(r['split0'][1])} {_ck(c['c5'])} | "
                 f"**{_ck(c['all'])}** | {_ck(r['crit_each']['all'])} · {_ck(r['crit_b12']['all'])} · {_ck(r['crit_circ']['all'])} · "
                 f"{_ck(r['crit_acc']['all'])} |")
    L += ["", f"**산술 결과:** 통과(1–5) = {', '.join(passed) or '없음'}. "
          + (f"파라미터 수 최소 · 전환 적은 순 = {', '.join(simp)} → 코어 **{simp[0]}**." if simp
             else "통과가 없으므로 설계 문장대로 코어 = **C0(매수보유)**, 위험은 비중으로 조절."), ""]
    L += ["(3) 배당 민감도(참고, 일별 가산 줄의 CAGR 차): " + " · ".join(
        f"{r['proc']} " + " / ".join(
            f"{y * 100:.1f}% {_pp(stats[('C0', r['win'])][f'cagr_acc{int(y * 1000)}'] - stats[(r['proc'], r['win'])][f'cagr_acc{int(y * 1000)}'])} "
            f"{_ck(r['c3_sens'][y])}" for y in DIV_SENS)
        + f" / 1.8% {_pp(stats[('C0', r['win'])]['cagr_acc'] - stats[(r['proc'], r['win'])]['cagr_acc'])} {_ck(r['crit_acc']['c3'])}"
        for r in rows) + ".", "",
          f"C2 국면 점수 월 블록 순열 널 {args.perms}벌(설계 '널' 절, 판정 조건 목록 밖): C2 OOS Calmar {_f(stats[('C2', 'oos')]['calmar'])} vs 순열 널 "
          f"p50 {_f(perm_sum['p50'])} · p95 {_f(perm_sum['p95'])} (백분위 {perm_sum['pct'] * 100:.0f}%) {_ck(perm_sum['beat'])}; "
          f"초과 Sharpe {_f(stats[('C2', 'oos')]['sharpe'])} vs p95 {_f(perm_sum['sh_p95'])} (백분위 {perm_sum['sh_pct'] * 100:.0f}%).", ""]

    L += ["## 낙폭 사건표 (C0 낙폭 > 15%, 리뷰 C1 요구 — 기술)", "",
          "규칙 칸 = 고점일 보유(1 = 안) / 고점일 종가 → 저점일 종가 구간 안 규칙 자신의 최대 낙폭 / 그 구간(고점 다음 날 ~ 저점일) 평균 보유. "
          "C1·C2 는 2020-01 OOS 부터.", "",
          "| 고점 → 저점 | 거래일 | C0 | " + " | ".join(TIMING) + " |", "|---|---|---|" + "---|" * len(TIMING)]
    for e in events:
        cells = ["—" if e[nm] is None else f"{e[nm]['expo']:.0f} / {_pct(e[nm]['dd'])} / {e[nm]['tim'] * 100:.0f}%" for nm in TIMING]
        L.append(f"| {e['peak']} → {e['trough']} | {e['days']} | {_pct(e['c0'])} | " + " | ".join(cells) + " |")
    L += [""]
    pcs = [(nm, "full_pc" if nm in ("C0", "C1f", "C4") else "oos_pc") for nm in ("C0", "C1f", "C4", "C1", "C2")]
    L += ["**급락 전 기술 줄(2026-06-19 까지):** " + " · ".join(
        f"{nm} [{'2017~' if wk == 'full_pc' else 'OOS 2020~'}] CAGR {_pct(stats[(nm, wk)]['cagr'])} · MDD {_pct(stats[(nm, wk)]['mdd'])} · "
        f"Calmar {_f(stats[(nm, wk)]['calmar'])}" for nm, wk in pcs)
          + f" (같은 창 C0 [OOS 2020~] Calmar {_f(stats[('C0', 'oos_pc')]['calmar'])}). 판정에 쓰지 않는다.", ""]

    L += ["## 위성 (C3 = 코어 80% + P1 20%, 월 되돌림)", "",
          "창 = **2020-01-02 ~ 2026-04-01**(메타 세대 1 P1 NAV 가 있는 구간 — 설계의 2017~ 창보다 짧다). 조건: P(초과 Sharpe C3·X > X) ≥ 0.8 이면 "
          "\"위성 20% 가 코어에 더한다\", 아니면 \"위성은 취미 비중\". 적용 행 = C0 와 고른 통과 코어. 나머지는 참고. 부트스트랩 = 같은 달 블록(3개월).", "",
          "| 코어 X | 적용 | C3·X CAGR · MDD · Calmar · 초과 SR | X 같은 창 CAGR · MDD · Calmar · 초과 SR | P(SR > X) · P(Calmar > X) | "
          "ΔSR 5·50·95% | P(SR > C0) | 참고: 12개월 블록 · 배당 가산 P(SR > X) | 산술 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sat_rows:
        x = sat[s["core"]]
        a, b, bt = x["st"], x["core_st"], x["boot"]
        L.append(f"| {s['core']} | {'적용' if s['applies'] else '참고'} | {_pct(a['cagr'])} · {_pct(a['mdd'])} · {_f(a['calmar'])} · "
                 f"{_f(a['sharpe'])} | {_pct(b['cagr'])} · {_pct(b['mdd'])} · {_f(b['calmar'])} · {_f(b['sharpe'])} | "
                 f"{bt['p_sharpe']:.2f} · {bt['p_calmar']:.2f} | {_q3(bt, 'dsharpe')} | {x['boot_c0']['p_sharpe']:.2f} | "
                 f"{x['boot12']['p_sharpe']:.2f} · {x['boot_acc']['p_sharpe']:.2f} | {'더한다' if s['adds'] else '취미 비중'} {_ck(s['adds'])} |")
    ss, cs, bc = v["sat_st"], v["cash_st"], sat["C0"]["boot_cash"]
    L += ["", f"**대조(리뷰 C2-3): 80% C0 + 20% 현금**(같은 되돌림) CAGR {_pct(cs['cagr'])} · MDD {_pct(cs['mdd'])} · Calmar {_f(cs['calmar'])} · "
          f"초과 SR {_f(cs['sharpe'])} (원시 SR {_f(cs['sharpe_raw'])} — C0 원시 {_f(sat['C0']['core_st']['sharpe_raw'])}). "
          f"P(C3·C0 초과 SR > 80/20 현금) = {bc['p_sharpe']:.2f}, P(Calmar) = {bc['p_calmar']:.2f}.  ",
          "2025 제외 이어 붙인 초과 SR: " + " · ".join(f"{k} {_f(val)}" for k, val in v["sat_ex25"].items()) + ".  ",
          f"P1(+ 유휴 현금 금리) 단독 같은 창: CAGR {_pct(ss['cagr'])} · MDD {_pct(ss['mdd'])} · Calmar {_f(ss['calmar'])} · 초과 SR {_f(ss['sharpe'])}; "
          f"월 수익 상관(P1c, C0) = {v['corr']:+.2f}."
          + (f" 리뷰어 `indiv.csv` p1c 와 날별 최대 차 {v['p1c_check']:.2e}." if v["p1c_check"] is not None else ""), ""]

    L += ["## 숫자 (비용·보수·현금 금리 포함)", "",
          "CAGR = 날 수 기준(252/T — 저장소 `equity_stats`; 달력일 기준보다 약 0.4%p 높게 나온다), MDD = 일별 경로(초기 자산 1 을 첫 고점으로), Calmar = CAGR / |MDD|, "
          "초과 SR = (월 복리 수익 − 같은 달 CD91 복리) 평균/표준편차 × √12, 원시 SR = 무위험 차감 없음(참고). 시장 안 = 평균 보유, 전환/년 = 보유가 바뀐 날 / 년. "
          "배당 고정 = 날별 − 1.8%/252 × (1 − 보유)(C0 불변), 배당 가산 = 날별 + 1.8%/252 × 보유(C0 = **배당 포함 매수보유 근사**), "
          "세후 = 배당 가산 줄에서 현금 이자·배당 15.4% 과세(ETF 매매차익 비과세).", ""]
    for wk, (lo, hi, lab) in WINS.items():
        names = [nm for nm, _, _ in PROCS if (nm, wk) in stats]
        if not names:
            continue
        L += [f"### {lab}", "", "| 절차 | CAGR | MDD | Calmar | 초과 SR | 원시 SR | 시장 안 | 전환/년 | CAGR 배당 고정 | CAGR · Calmar 배당 가산 | "
              "CAGR · Calmar 세후 | 기간 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for nm in names:
            s = stats[(nm, wk)]
            L.append(f"| {nm} | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe'])} | {_f(s['sharpe_raw'])} | "
                     f"{s['tim'] * 100:.0f}% | {s['sw_py']:.1f} | {_pct(s['cagr_div'])} | {_pct(s['cagr_acc'])} · {_f(s['calmar_acc'])} | "
                     f"{_pct(s.get('cagr_tax', np.nan))} · {_f(s.get('calmar_tax', np.nan))} | {s['start']} ~ {s['end']} |")
        L.append("")
    L += ["C1f·C4 의 OOS 행은 전체 창 계열을 자른 것(2020-01 이전부터 보유 중), C1·C2 는 2020-01-02 시가에 현금에서 시작(진입 비용 0.07% 한 번).", ""]

    L += ["## 무작위 타이밍 널 (규칙마다 1,000벌 × 2종)", "",
          "**마르코프**(설계): 판정 창 보유의 π·전환 빈도 s 에 맞춘 2-상태 체인, p10 = s/(2π), p01 = s/(2(1 − π)), 첫 상태 ~ Bernoulli(π) — C4 는 **달 단위 체인**"
          "(월말에만 전환, 정정). **원형 이동**(정정 추가): 실제 보유 열을 오프셋 ∈ [60, T − 60) 거래일만큼 돌림 — 체류 길이·휩쏘 군집 보존. 같은 엔진·비용·현금. "
          "p95 = '널 중 좋은 쪽' 끝(MDD 는 덜 깊은 쪽). π·전환 빈도는 날 기준으로 적는다(C4 체인은 달 단위로 맞추고 날로 펼침).", "",
          "| 절차 | 널 | 단위 | π 목표 · 널 실현 | 전환 빈도 목표 · 널 실현(날) | Calmar · p50 · p95 (백분위) | 초과 SR · p50 · p95 (백분위) | MDD · p50 · p95 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        nm, wk = r["proc"], r["win"]
        s = stats[(nm, wk)]
        for kind, lab in (("markov", "마르코프"), ("circ", "원형 이동")):
            n = nulls[nm][kind]
            L.append(f"| {nm} | {lab} | {nulls[nm]['unit'] if kind == 'markov' else '날'} | {nulls[nm]['pi']:.3f} · {n['pi_mean']:.3f} | "
                     f"{nulls[nm]['s']:.4f} · {n['s_mean']:.4f} | "
                     f"{_f(s['calmar'])} · {_f(qs(n['calmar'], 50))} · {_f(qs(n['calmar'], 95))} ({mg1.null_pct(s['calmar'], n['calmar']) * 100:.0f}%) | "
                     f"{_f(s['sharpe'])} · {_f(qs(n['sharpe'], 50))} · {_f(qs(n['sharpe'], 95))} ({mg1.null_pct(s['sharpe'], n['sharpe']) * 100:.0f}%) | "
                     f"{_pct(s['mdd'])} · {_pct(qs(n['mdd'], 50))} · {_pct(qs(n['mdd'], 95))} |")
    L += ["", "## 부트스트랩 상세 (C0 대비, 판정 창, 일별 경로 달 블록)", "",
          "| 절차 | 블록 | 달 | P(초과 SR) | ΔSR 5·50·95% | P(Calmar) | ΔCalmar 5·50·95% | P(둘 다) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for key, lab in (("b3", "3개월"), ("b12", "12개월"), ("acc", "3개월 · 배당 가산 줄")):
            b = boots[r["proc"]][key]
            L.append(f"| {r['proc']} | {lab} | {b['months']} | {b['p_sharpe']:.3f} | {_q3(b, 'dsharpe')} | {b['p_calmar']:.3f} | "
                     f"{_q3(b, 'dcalmar')} | {b['p_both']:.3f} |")

    L += ["", "## 폴드 선택 (적합 규칙)", "",
          f"학습창 적합도(정정) = 학습창(현금에서 시작) **일별 경로**를 {FIT_BLOCK}개월 블록으로 {FIT_N}회 재표본한 Calmar 의 {FIT_Q:g} 백분위(MDD 하한 1%, 폴드마다 "
          "같은 표본 — 옵션·순열 널 공통). 동률이면 학습창 전환 수가 적은 쪽.", "",
          "| 절차 | 시험 해 | 옵션별 적합도 [학습창 전환] | 고른 것 |", "|---|---|---|---|"]
    for nm, res in (("C1", c1), ("C2", c2)):
        for c in res["choices"]:
            L.append(f"| {nm} | {c['test']} | " + " · ".join(f"{lab} {_f(f)} [{sw}]" for lab, f, sw in zip(c["labels"], c["fits"], c["sws"], strict=True))
                     + f" | **{c['labels'][c['j']]}** |")
    L += ["", "C2 순열 널에서 고른 문턱 분포(폴드 × 벌, 옵션 0 = 없음 · 1~3 = q20·q40·q60): "
          + " · ".join(f"{k}: {int(n)}" for k, n in pd.Series(np.concatenate(pdf["thr_j"].to_numpy())).value_counts().sort_index().items())
          + f". 순열 널 평균 시장 안 비율 {pdf['tim'].mean() * 100:.0f}% (C2 실제 {stats[('C2', 'oos')]['tim'] * 100:.0f}%).", ""]

    ydf = v["ydf"]
    procs_y = [nm for nm in ("C0", "C1f", "C4", "C1", "C2", "C3·C0") if nm in set(ydf["proc"])]
    L += ["## 연도별 수익 (그 해 보유 비율)", "", "| 해 | " + " | ".join(procs_y) + " |", "|---|" + "---|" * len(procs_y)]
    for y in sorted(set(ydf["year"])):
        cells = []
        for nm in procs_y:
            q = ydf[(ydf["proc"] == nm) & (ydf["year"] == y)]
            cells.append("" if q.empty else f"{_pct(q['ret'].iloc[0])} ({q['tim'].iloc[0] * 100:.0f}%)")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    L += ["", "2026 = 01-02 ~ 09-30, C3·C0 의 2026 = 01-02 ~ 04-01(P1 NAV 끝, 보유 = 코어 다리 80%). C1·C2 는 2020 부터(OOS). 가격 지수(배당 제외)."]

    L += ["", "## 자료 · 가정", "",
          f"- **지수:** `{INDEX_CSV}` 의 `{CORE_SYM}`(KOSPI200 가격 지수, 네이버 차트 일봉 시가·종가). 체결 = 지수 시가(ETF 시가 대용 — 괴리·호가 단위 무시). "
          "2026 연 변동성이 2017~25 의 두세 배이고 2026-07-31 +20% 같은 날이 있다 — KOSPI·KOSDAQ 같은 날 같은 방향이고 리뷰어가 DB 종목 일봉으로 교차 확인했다.",
          "- **현금:** CD91 연평균 근사 " + " · ".join(f"{y} {r * 100:.2f}%" for y, r in sorted(CD91.items()))
          + " — 2017~19 = 메타 세대 3 러너 값, 2020~26 = 메타 세대 1 리뷰어 값(`review_meta1/indiv.py`). **원천은 기억값(DB 밖)**. 2015·16 은 2017 값"
          "(C4 의 2016 월말 12개월 현금 문턱에만 쓰인다). 날 금리 = 연 / 252, 전일 시장 밖 비율에. 초과 SR 의 무위험도 이 값.",
          f"- **비용:** 보유가 바뀐 날 시가에 편도 {COST_1W * 100:.2f}%, 보수 연 {FEE_YR * 100:.2f}% / 252 × 그날 보유.",
          f"- **배당:** 고정 보정 줄(연 {DIV_YR * 100:.1f}% × 시장 밖 비율을 타이밍에서 뺌 — 판정 (3)) + 일별 가산 줄(연 {DIV_YR * 100:.1f}%/252 × 보유, "
          "C0 = 배당 포함 근사; '배당 가산 줄로 전부' 참고 열). 배당락 12월 쏠림·연도별 수익률 차는 무시(리뷰 M1: 효과 0.1%p 안팎).",
          f"- **세후 줄(기술):** 일반 계좌 가정 — 현금(MMF/CMA) 이자·ETF 분배금 {TAX * 100:.1f}%, 국내 주식형 ETF 매매차익 비과세. ISA·연금계좌는 다르다.",
          "- **C2 RA:** `regime_panel.npz` 재구축 판의 breadth_ma50·breakout_success20·nh_nl_ratio × KOSPI·KOSDAQ 6열(지수 대용 index_vs_ma200 제외), "
          "폴드 학습창 평균·표준편차로 z → 있는 열 평균(`meta_gen1_run.zscore_mean`). 문턱 격자 = 학습창 {없음, 20, 40, 60} 백분위(메타 세대 1 게이트 격자 — "
          "설계는 '문턱은 학습창에서' 만 적었다). 점수 NaN 날은 열림(코어). '없음' = 늘 코어 = C0 보유. C2 의 학습창 선택도 C1 과 같은 적합도·동률 규칙.",
          "- **C4:** 매월 마지막 거래일 종가의 12개월 전 월말 종가 대비 수익 vs 같은 기간 현금 복리. 신호 다음날(다음 달 첫 거래일) 시가 매매.",
          "- **위성 P1:** 메타 세대 1 `nav.csv` 의 `P1 ×1`(2020-01-02 ~ 2026-04-01) 날별 수익 + (1 − 그 해 평균 노출) × 현금 금리 — 세대 1 리뷰어 "
          "`indiv.py` 의 `p1c` 구성(연 평균 노출 근사). P1 은 뉴스 거부권이 없는 판이다. C3 되돌림 비용 = 0.07% × |표류 코어 비중 − 80%| × 코어 보유.",
          "- **부트스트랩:** 판정 창 일별 수익을 달 블록(비원형 이동 블록, 3개월 · 참고 12개월)으로 재표본해 이어 붙인 일별 경로 → Calmar(일별 MDD, 하한 없음)·"
          "초과 SR(뽑힌 달들의 월 초과수익). 같은 창의 모든 비교에 같은 달 표본(공통 난수).",
          f"- **원장:** `{SUB}/TRIALS.jsonl` 새 줄 {v['led']}개. 산출: `{SUB}/` summary·decision·folds·events·null_random·null_c2_perm·nav·peryear.csv, `equity.png`.", ""]
    if v["fig"]:
        L += ["## 그림", "", f"![자산·낙폭]({SUB}/equity.png)", "",
              "왼쪽 = 고정 규칙 전체 창, 가운데 = 이어 붙인 OOS(C1·C2 는 2020-01 현금 시작), 오른쪽 = 코어-위성(P1 NAV 구간). 아래 = 낙폭. 회색 띠 = 2026-06-22 ~ 07-30.", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
