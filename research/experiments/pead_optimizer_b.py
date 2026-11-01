"""PEAD 최적화 단계 B 하네스 — 현행 순위 북 R 대 최적화 북 O. 사전등록 ``PREREG.md`` §2·PO-1.4·PO-1.6·PO-1.7·PO-3.4·PO-5·PO-6.

두 북·널·패리티가 **같은 적격 마스크 함수 하나**(:func:`eligible` = A 단계가 쓴 ``pead_optimizer_c0.eligible_idx``)를 쓴다.
회계는 두 북이 같은 엔진(:func:`simulate`)을 탄다. 구현 고정(모호했던 곳)은 정정 PO-6 에 적었다.

모드:

- ``--mode parity`` (PO-1.4): 패리티 검사만 한다. 하네스 패리티 모드(지연 0·무비용·강제 매도 끔·현금 없음)와 core
  ``staggered_tranche_backtest`` 의 스텝별 ``book``·``bench`` 가 1e-10 안인지, ``n_universe``·적격 집합이 같은지만 본다.
  **성과 숫자는 출력하지도 저장하지도 않는다**(차이·개수만). 출력 ``B_parity.json``·``B_parity.md``.
- ``--mode run``: 단계 B 전체(민감도 6칸 → 1차 → 순열 널 300 → 판정). PO-5.2 순서를 코드로 강제한다 — ``A1_repaired.json`` 이 있고
  그 판정이 ``통과`` 이며, 지금 ``daily_bars_adjusted`` 지문이 그 파일의 지문과 같을 때만 돈다.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" \\
      python research/experiments/pead_optimizer_b.py --mode parity --cache <scratch>/bparity --po3-cache <scratch>/po3cache

2024-01 이후 데이터는 어떤 계산에도 쓰지 않는다(END 2023-12-28, PREREG §2).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import krx_quant_core.portfolio  # noqa: F401 — portfolio-layer(PR #19) 필요. 없으면 여기서 ModuleNotFoundError
import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _ffill_axis1_nb, _trailing_adv, staggered_tranche_backtest
from krx_quant_core.costs.impact import trade_cost_bp
from krx_quant_core.costs.model import DEFAULT_COMMISSION_RATE, DEFAULT_KOSDAQ_SLIPPAGE_PCT, DEFAULT_KOSPI_SLIPPAGE_PCT
from krx_quant_core.costs.tax import sell_tax_rate
from pead_optimizer_c0 import LAM_HI, LAM_ITERS, LAM_LO, LAM_TOL, Sub, eligible_idx, keyed_rng
from scipy import stats

# ── 사전등록 상수(PREREG §2·PO-1.6·PO-1.7·PO-5) ──────────────────────────────────────

END_B = "2023-12-28"  # §2 — 2024-01 이후는 쓰지 않는다
LOAD_START = "2019-01-02"  # 위험모델 입력(v4 와 같은 시작)
IC_LOAD_START = "2016-11-01"  # IC 학습 패널(2017-01-02 부터 + ADV 20일 앞)
IC_START = "2017-01-02"
GRID_START = "2019-08-01"
EVAL = ("2020-01-01", "2023-12-31")
FOLD_YEARS = (2020, 2021, 2022, 2023)
STEP, R_HORIZON, TOP_N, MIN_NAMES = 20, 60, 10, 20
CAP_RANK, ADV_FLOOR = (20, 100), 20000.0  # 시총 21~100위, ADV 200억(백만원)
V4_SHRINK = 0.1  # 위험모델 v4 = vraoff_s0.1(PO-3.5)
MAX_W, MAX_TURNOVER, MAX_PART = 0.10, 2.0 / 3.0, 0.05
IC_CLIP = (0.0, 0.10)
VOL_WIN, VOL_MIN_OBS = 63, 20
HALF_SPREAD_BP = {"KOSPI": float(DEFAULT_KOSPI_SLIPPAGE_PCT) * 1e4, "KOSDAQ": float(DEFAULT_KOSDAQ_SLIPPAGE_PCT) * 1e4}
COMMISSION_BP = float(DEFAULT_COMMISSION_RATE) * 1e4
N_PERM = 300  # PO-1.7
NULL_Q = 0.95
NI_X_CAP, NI3_RATIO = 0.30, 1.25  # PO-1.6 x = min(IR(R)/3, 0.30), B-NI-3
OPT_OK_MIN, OOS_MIN, LAM_FAIL_MAX = 0.95, 40, 0.10  # VOID
FACTOR_ALARM = 0.20  # PO-3.4·PO-5.3
PARITY_TOL = 1e-10
ANN = math.sqrt(252.0 / STEP)
N_BOOT, BOOT_MEAN_BLOCK = 10_000, 3
DUST_EPS = 1e-6  # PO-7 M2 — 해 직후 |w| < max(1e-6, 1주 가격/AUM) 은 0(솔버 먼지)
INV_EPS = 1e-6  # PO-7 M3 — 불변식 허용오차
COST_BP_PER_TURN = (1.5, 300.0)  # PO-7 M3 — 1배 비용/회전(bp): 수수료 하한 ~ 상한
LIMIT_UP_RET = 0.29  # PO-7 MINOR 9 — 체결일 종가 수익 ≥ 29% 면 상한가 체결로 센다


@dataclass(frozen=True)
class Cell:
    """설정 한 칸(PREREG §4: α 지평 × 업종 밴드 × AUM). 1차 = (60, 0.10, 3억)."""

    h_alpha: int = 60
    band: float = 0.10
    aum: float = 3e8

    @property
    def name(self) -> str:
        return f"H{self.h_alpha}_band{self.band:.2f}_aum{self.aum:.0e}"


PRIMARY = Cell()
SENSITIVITY = (Cell(20, 0.05), Cell(20, 0.10), Cell(20, 0.20), Cell(60, 0.05), Cell(60, 0.20), Cell(60, 0.10, 1e10))


# ── 공유 적격 마스크(§2) ─────────────────────────────────────────────────────────


def m0_mask(t: int, sig: np.ndarray, adv: np.ndarray, in_u: np.ndarray) -> np.ndarray:
    """M0(t) = ① 신호 유한 ∧ ② 20일 ADV ≥ 200억(백만원, core ``_trailing_adv``) ∧ ③ U_t."""
    return np.isfinite(sig[:, t]) & (adv[:, t] >= ADV_FLOOR) & in_u


def eligible(t: int, sig: np.ndarray, adv: np.ndarray, cap: np.ndarray, in_u: np.ndarray) -> np.ndarray:
    """eligible(t) = M0(t) 안 시총 21~100위(행 번호, 시총 내림차순). A 단계(P10B·OB)와 같은 함수다."""
    return eligible_idx(t, sig, adv, cap, in_u)


def u_exclusion(t: int, sig: np.ndarray, adv: np.ndarray, in_u: np.ndarray) -> float:
    """§2 — ①∧② 중 ③(U_t) 때문에 빠진 비율."""
    base = np.isfinite(sig[:, t]) & (adv[:, t] >= ADV_FLOOR)
    n = int(base.sum())
    return float((base & ~in_u).sum() / n) if n else float("nan")


def parity_signal(sig: np.ndarray, adv: np.ndarray, grid: list[int], in_u_of) -> np.ndarray:
    """PO-1.4 — 격자일 t_k 의 M0 안에서만 유한한 신호 패널(core 에 넘긴다)."""
    out = np.full_like(sig, np.nan)
    for t in grid:
        m = m0_mask(t, sig, adv, in_u_of(t))
        out[m, t] = sig[m, t]
    return out


def tranche_names(t, rank_sig, sig, adv, cap, in_u, top_n=TOP_N, min_names=MIN_NAMES):
    """트랜치 = eligible(t) 의 신호 상위 ``top_n`` (core ``staggered_tranche_backtest.book`` 과 같은 정렬).

    ``rank_sig`` 는 순위에 쓰는 신호(진짜 또는 순열), 적격은 진짜 신호 ``sig`` 로 정한다(§2 널). 순열 값이 NaN 인 적격 종목은 고르지 않는다.
    """
    el = eligible(t, sig, adv, cap, in_u)
    if el.size < min_names:
        return None
    idx = np.sort(el)  # core: idx = np.where(ok)[0] (행 순서)
    vals = rank_sig[idx, t]
    keep = np.isfinite(vals)
    idx, vals = idx[keep], vals[keep]
    return idx[np.argsort(-vals)[:top_n]]


# ── α 지도(§2) ─────────────────────────────────────────────────────────────────


def rank_z(vals: np.ndarray) -> np.ndarray:
    """z_i = Φ⁻¹((순위_i − 0.5)/n), 오름차순 평균 순위. NaN 은 z=0(널의 순열 NaN, PO-6)."""
    vals = np.asarray(vals, dtype=np.float64)
    out = np.zeros(vals.shape)
    fin = np.isfinite(vals)
    n = int(fin.sum())
    if n:
        r = stats.rankdata(vals[fin], method="average")
        out[fin] = stats.norm.ppf((r - 0.5) / n)
    return out


def alpha_vec(ic: float, d_daily: np.ndarray, z: np.ndarray, h: int) -> np.ndarray:
    """α_i = IC_k · √(H·D_i) · z_i (Grinold)."""
    return ic * np.sqrt(h * np.asarray(d_daily, dtype=np.float64)) * z


def daily_rank_ic(close, adv, cap, sig, lo: int, hi: int, h: int):
    """일별 순위 IC(``rank_ic`` 정의: t 신호 대 C[t+1+h]/C[t+1]−1, Spearman, 20종목 이상). 적격 = U_t 없는 M0 의 시총 21~100위(PO-6).

    PO-7 MINOR 2 — 창 안에서 가격이 끊긴 종목(폐지·정지)은 마지막 관측 종가로 잰다(빼면 생존자 필터). C[t+1] 이 없으면 뺀다.
    """
    nN, nD = close.shape
    cff = _ffill_axis1_nb(np.asarray(close, dtype=np.float64))
    allu = np.ones(nN, bool)
    days, ics = [], []
    for t in range(lo, min(hi, nD - h - 2) + 1):
        el = eligible_idx(t, sig, adv, cap, allu)
        ret = cff[el, t + 1 + h] / close[el, t + 1] - 1.0
        ok = np.isfinite(ret)
        if ok.sum() < 20:
            continue
        a = stats.rankdata(sig[el[ok], t])
        b = stats.rankdata(ret[ok])
        if a.std() > 0 and b.std() > 0:
            ics.append(float(np.corrcoef(a, b)[0, 1]))
            days.append(t)
    return np.array(days, dtype=int), np.array(ics)


def fold_ic(days: np.ndarray, ics: np.ndarray, fold_start: int, h: int) -> float:
    """IC_k = 폴드 시작 (h+2) 거래일 앞까지 누적한 일별 IC 평균, [0, 0.10] 자르기."""
    sel = ics[days <= fold_start - (h + 2)]
    if sel.size == 0:
        return 0.0
    return float(np.clip(sel.mean(), *IC_CLIP))


# ── 위험·비용 도우미 ─────────────────────────────────────────────────────────────


def sub_model(model, rows: np.ndarray, h: int) -> Sub:
    """모델 ``(idx, X, F, D, names)`` 에서 행 ``rows``(전체 종목 번호)만, 지평 h 로 늘린 부분 모델."""
    idx, X, F, D, names = model
    pos = {int(c): i for i, c in enumerate(idx)}
    p = np.array([pos[int(r)] for r in rows], dtype=int)
    return Sub(X=X[p], F=h * F, D=h * D[p], names=list(names))


def active_risk(sub: Sub, d: np.ndarray) -> float:
    y = sub.X.T @ d
    return float(np.sqrt(max(y @ sub.F @ y + (d * d) @ sub.D, 0.0)))


def factor_share(sub: Sub, d: np.ndarray) -> float:
    """사전 능동분산 중 팩터 몫 yᵀFy / (yᵀFy + Σ d_i² D_i) (PO-3.4 — 지평 배수와 무관)."""
    y = sub.X.T @ d
    fv = float(y @ sub.F @ y)
    sv = float((d * d) @ sub.D)
    return fv / (fv + sv) if fv + sv > 0 else float("nan")


def top_exposures(sub: Sub, d: np.ndarray, k: int = 3) -> list:
    """팩터 몫 경보 날의 큰 노출 셋(PO-5.3·PO-7 MINOR 7): 능동 노출 y = Xᵀd 를 팩터 변동성 √F_kk 로 잰 크기 순."""
    y = sub.X.T @ d
    sz = np.abs(y) * np.sqrt(np.clip(np.diag(sub.F), 0.0, None))
    order = np.argsort(-sz)[:k]
    return [(str(sub.names[j]), float(y[j])) for j in order]


def clean_dust(w: np.ndarray, price: np.ndarray, aum: float, fixed: np.ndarray) -> tuple[np.ndarray, int]:
    """PO-7 M2 — 솔버 먼지 제거. 고정(``no_trade``) 아닌 종목의 w < max(1e-6, 1주 가격/AUM) 은 0, 음수도 0.

    지운 만큼은 남은(고정 아닌, 양수) 비중에 비례로 돌려 합을 지킨다. 반환 ``(w, 지운 수)``.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    thr = np.maximum(DUST_EPS, np.where(np.isfinite(price), price, 0.0) / aum)
    kill = ~fixed & (w < thr) & (w != 0.0)
    total = w.sum()
    w[kill] = 0.0
    keep = ~fixed & (w > 0)
    s_keep = w[keep].sum()
    if s_keep > 0:
        w[keep] *= (total - w[fixed].sum()) / s_keep
    return w, int(kill.sum())


def linear_bp(side: np.ndarray, market: np.ndarray, day: date, hs_bp: np.ndarray) -> np.ndarray:
    """수수료 + 매도세(매도만) + 반스프레드(bp) — 충격 없는 부분(폐지 청산에 쓴다)."""
    side = np.asarray(side, dtype=str)
    tax = np.array([sell_tax_rate(day, m) * 1e4 for m in market], dtype=np.float64)
    return COMMISSION_BP + np.where(side == "SELL", tax, 0.0) + np.asarray(hs_bp, dtype=np.float64)


def trade_costs(dw, adv_won, vol, market, day: date, hs_bp, aum: float) -> np.ndarray:
    """종목별 1배 비용(NAV 대비 분수). core ``trade_cost_bp`` 그대로(정가 수수료·일자별 매도세·반스프레드·제곱근 충격 y=1).

    2배는 이 값의 2배다 — 네 항(수수료·매도세·반스프레드·y)이 모두 배수에 선형이다(§2 "전부 2배").
    """
    dw = np.asarray(dw, dtype=np.float64)
    out = np.zeros(dw.shape)
    tr = dw != 0
    if tr.any():
        side = np.where(dw[tr] > 0, "BUY", "SELL")
        bp = trade_cost_bp(side, np.abs(dw[tr]) * aum, np.asarray(adv_won)[tr], np.asarray(vol)[tr],
                           np.asarray(market, dtype=object)[tr], day, half_spread_bp=np.asarray(hs_bp)[tr], y=1.0)
        out[tr] = bp / 1e4 * np.abs(dw[tr])
    return out


def trailing_vol(ret: np.ndarray, t: int) -> np.ndarray:
    """t 포함 직전 63거래일 일수익 표준편차(ddof=1). 유한 관측 < 20 이면 NaN(→ 그 날 거래 불가, PO-6)."""
    w = ret[:, max(0, t - VOL_WIN + 1): t + 1]
    n = np.isfinite(w).sum(axis=1)
    with np.errstate(invalid="ignore"):
        v = np.nanstd(w, axis=1, ddof=1)
    v[n < VOL_MIN_OBS] = np.nan
    return v


# ── λ 맞춤(§2) ─────────────────────────────────────────────────────────────────


def match_lambda(solve, risk_of, target: float):
    """로그 공간 이분법 [1e-2, 1e5], 최대 40회, ±2%. 위험은 λ 에 감소한다.

    ``optimal`` 이 아닌 풀이에서 멈추지 않는다(PO-7 MINOR 4): 그 반복은 세고, 가장 가까운 성공 λ 쪽으로 구간을 좁혀 계속한다.

    Returns: ``(res, lam, hit, n_solves, cause)``. 못 맞추면 ``optimal`` 인 반복 중 |log(위험/목표)| 가 가장 작은 것(PO-6), 그것도 없으면
    ``None``. ``cause`` ∈ {"", "solver"(성공 풀이 없음), "up"(모든 성공 풀이의 위험 < 목표 — 하한 λ 에서도 못 올림),
    "down"(모든 성공 풀이의 위험 > 목표), "gap"(목표를 사이에 두었지만 ±2% 창을 못 맞춤 — 위험이 λ 에 불연속)}.
    """
    lo, hi = math.log(LAM_LO), math.log(LAM_HI)
    best, best_gap, best_lam, n = None, math.inf, float("nan"), 0
    below = above = False
    for _ in range(LAM_ITERS):
        mid = 0.5 * (lo + hi)
        lam = math.exp(mid)
        res = solve(lam)
        n += 1
        if not res.ok:
            if best is None:
                hi = mid  # 성공 풀이가 아직 없으면 작은 λ 쪽부터
            elif best_lam > lam:
                lo = mid
            else:
                hi = mid
            continue
        risk = risk_of(res.weights)
        gap = abs(math.log(max(risk, 1e-300) / target))
        if gap < best_gap:
            best, best_gap, best_lam = res, gap, lam
        if abs(risk / target - 1.0) <= LAM_TOL:
            return res, lam, True, n, ""
        if risk > target:
            above = True
            lo = mid
        else:
            below = True
            hi = mid
    cause = "solver" if best is None else ("gap" if (above and below) else ("up" if below else "down"))
    return best, best_lam, False, n, cause


# ── 엔진 ───────────────────────────────────────────────────────────────────────


@dataclass
class Ctx:
    """``code × date`` 패널(위험모델 좌표, 열 = 거래일). 금액 단위는 이름에 적었다."""

    dates: pd.DatetimeIndex
    C: np.ndarray  # 수정 종가(이음매 가드 뒤)
    tv_won: np.ndarray  # 일 거래대금(원)
    adv: np.ndarray  # t 제외 20일 평균 거래대금(백만원, core _trailing_adv)
    cap: np.ndarray
    sig: np.ndarray  # 진짜 YoY 신호
    ret: np.ndarray  # 일수익(가드 뒤)
    lab: np.ndarray  # WICS 섹터
    market: np.ndarray  # "KOSPI"/"KOSDAQ"
    grid: list[int]
    models: dict  # t → (idx, X, F, D, names), 일 단위
    delist_after: np.ndarray  # 마지막 거래일 열 번호(폐지 아님 = 큰 수)
    Cff: np.ndarray = field(default=None)
    hs_bp: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.Cff is None:
            self.Cff = _ffill_axis1_nb(np.asarray(self.C, dtype=np.float64))
        if self.hs_bp is None:
            self.hs_bp = np.array([HALF_SPREAD_BP.get(m, HALF_SPREAD_BP["KOSDAQ"]) for m in self.market])
        self._u = {}

    def in_u(self, t: int) -> np.ndarray:
        if t not in self._u:
            m = np.zeros(self.C.shape[0], bool)
            m[np.asarray(self.models[t][0], dtype=int)] = True
            self._u[t] = m
        return self._u[t]


@dataclass(frozen=True)
class Flags:
    """실제 규칙 스위치(PO-1.4 ①~⑥). 기본 = 실제 규칙 전부 켬. 전부 끈 것이 패리티 모드다(:func:`parity_book`)."""

    lag: int = 1  # ① t+1 종가 체결
    drift: bool = True  # ② 스텝 사이 매수 보유 표류(값 회계)
    forced_sale: bool = True  # ③ 위험모델 이탈 강제 매도
    delist_cash: bool = True  # ④ 폐지 대금 현금 → 다음 리밸런스 재투자
    no_trade: bool = True  # ⑤ 체결일 거래량 0·가격 없음 → 거래 불가
    costs: bool = True  # ⑥ 비용(1배로 기록, 2배는 배수)


def parity_book(ctx: Ctx, lag: int = 0) -> pd.DataFrame:
    """패리티 모드(PO-1.4): 지연 ``lag``·무비용·강제 매도 끔·현금 없음, 트랜치는 결성일 목록, 스텝마다 수익 유한 종목 nanmean.

    ``lag=0`` 이 core ``staggered_tranche_backtest(..., start_index=t_0, min_names=20, delisting_exit=True)`` 와 같아야 한다.
    """
    C, Cff = ctx.C, ctx.Cff
    nD = C.shape[1]
    t0 = ctx.grid[0]
    books: dict[int, np.ndarray | None] = {}
    rows = []
    for t in range(t0, nD - STEP - 1 - lag, STEP):
        uni = eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t))
        if uni.size < MIN_NAMES:
            continue
        f, e = t + lag, t + lag + STEP
        ret = C[:, e] / C[:, f] - 1.0
        gone = np.isfinite(C[:, f]) & ~np.isfinite(C[:, e])
        ret[gone] = Cff[gone, e] / C[gone, f] - 1.0
        bench = float(np.nanmean(ret[uni]))
        tb = []
        for k in range(R_HORIZON // STEP):
            tj = t - k * STEP
            if tj < t0:
                continue
            if tj not in books:
                books[tj] = tranche_names(tj, ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(tj))
            b = books[tj]
            if b is not None:
                tb.append(float(np.nanmean(ret[b])))
        if tb:
            rows.append({"t": t, "date": str(ctx.dates[t].date()), "book": float(np.mean(tb)), "bench": bench,
                         "n_universe": int(uni.size)})
    return pd.DataFrame(rows)


def core_parity(ctx: Ctx) -> pd.DataFrame:
    """core ``staggered_tranche_backtest`` 를 PO-1.4 인자로(신호는 격자일 M0 안에서만 유한)."""
    sp = parity_signal(ctx.sig, ctx.adv, ctx.grid, ctx.in_u)
    tv_m = ctx.tv_won / 1e6
    per, _ = staggered_tranche_backtest(
        ctx.C, tv_m, sp, list(range(ctx.C.shape[1])), horizon=R_HORIZON, step=STEP, top_n=TOP_N, adv_floor=ADV_FLOOR,
        start_index=ctx.grid[0], min_names=MIN_NAMES, cap_array=ctx.cap, cap_rank=CAP_RANK, delisting_exit=True)
    per = per.rename(columns={"date": "t"})
    return per[["t", "book", "bench", "n_universe"]]


def parity_check(ctx: Ctx, core_adv_matches: bool | None = None) -> dict:
    """하네스 패리티 대 core. 반환은 차이·개수뿐(성과 숫자 없음)."""
    h = parity_book(ctx, 0)
    c = core_parity(ctx)
    same_t = h["t"].tolist() == c["t"].tolist()
    out = {"steps_harness": int(len(h)), "steps_core": int(len(c)), "same_steps": bool(same_t)}
    if same_t and len(h):
        db = np.abs(h["book"].to_numpy() - c["book"].to_numpy())
        dc = np.abs(h["bench"].to_numpy() - c["bench"].to_numpy())
        out.update({"max_abs_diff_book": float(np.nanmax(db)), "max_abs_diff_bench": float(np.nanmax(dc)),
                    "nan_mismatch": int((np.isnan(h["book"]) != np.isnan(c["book"])).sum()
                                        + (np.isnan(h["bench"]) != np.isnan(c["bench"])).sum()),
                    "n_universe_same": bool((h["n_universe"].to_numpy() == c["n_universe"].to_numpy()).all()),
                    "first_date": h["date"].iloc[0], "last_date": h["date"].iloc[-1]})
        out["pass"] = bool(out["max_abs_diff_book"] <= PARITY_TOL and out["max_abs_diff_bench"] <= PARITY_TOL
                           and out["nan_mismatch"] == 0 and out["n_universe_same"])
    else:
        out["pass"] = False
    if core_adv_matches is not None:
        out["adv_bytes_same"] = bool(core_adv_matches)
        out["pass"] = out["pass"] and bool(core_adv_matches)
    return out


def full_parity(ctx: Ctx) -> dict:
    """PO-1.4 패리티 + ADV 바이트 + 격자 전부 적격 집합(``--mode parity`` 와 ``--mode run`` 이 같은 검사를 쓴다, PO-7 MINOR 6).

    패리티는 **마스크·트랜치 선택·폐지 산술만** 검증한다(``parity_book`` 은 core 함수의 이식이다). 엔진(``simulate``·``o_target``·비용·널)은
    :func:`check_invariants` 와 스웜 재계산으로 검증한다(PO-7 M3).
    """
    tv_m = ctx.tv_won / 1e6
    adv_core = _trailing_adv(tv_m, tv_m.shape[1], 20, ctx.C)
    adv_same = bool(np.array_equal(adv_core[:, 20:], ctx.adv[:, 20:], equal_nan=True))
    pc = parity_check(ctx, adv_same)
    sp = parity_signal(ctx.sig, ctx.adv, ctx.grid, ctx.in_u)
    allu = np.ones(ctx.C.shape[0], bool)
    elig_same = all(np.array_equal(np.sort(eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t))),
                                   np.sort(eligible_idx(t, sp, ctx.adv, ctx.cap, allu))) for t in ctx.grid)
    pc["eligible_same_all_grid"] = bool(elig_same)
    pc["pass"] = bool(pc["pass"] and elig_same)
    return pc


def delisted(ctx: Ctx, f: int) -> np.ndarray:
    """f 에 폐지로 본다 = ``delisted_stocks.last_trade_date`` 가 f 전 ∧ f 에 가격 없음(코드 재사용 방어, PO-6)."""
    return (ctx.delist_after < f) & ~np.isfinite(ctx.C[:, f])


def r_target(ctx: Ctx, k: int, t: int, tranches: dict, flags: Flags, f: int, with_parts: bool = False):
    """R 목표 비중(체결일 f 기준, 합 1): 활성 트랜치(k, k−1, k−2)마다 1/n, 트랜치 안 동일가중. 강제 매도·폐지 종목은 뺀다.

    ``with_parts`` 면 ``(w, 활성 트랜치 수)`` 를 돌려준다(PO-7 M3 불변식).
    """
    nN = ctx.C.shape[0]
    in_u = ctx.in_u(t)
    gone = delisted(ctx, f)
    w = np.zeros(nN)
    parts = []
    for j in range(k, max(-1, k - R_HORIZON // STEP), -1):
        b = tranches.get(j)
        if b is None:
            continue
        b = np.asarray(b, dtype=int)
        if flags.forced_sale:
            b = b[in_u[b]]
        if flags.delist_cash:
            b = b[~gone[b]]
        b = b[np.isfinite(ctx.Cff[b, f])]
        if b.size:
            parts.append(b)
    for b in parts:
        w[b] += 1.0 / (len(parts) * b.size)
    return (w, len(parts)) if with_parts else w


@dataclass
class StepRec:
    k: int
    t: int
    date: str
    gross: float  # G_k − 1 (비용 전, 체결 뒤 → 다음 체결 전)
    cost1: float  # 체결일 비용(1배, NAV 대비)
    bench: float
    turnover: float
    exante_20: float  # 사전 20일 능동위험(목표 비중, 그 날 모델)
    factor_share: float
    status: str = "optimal"
    lam: float = float("nan")
    lam_hit: bool = True
    est_cost: float = float("nan")
    n_names: int = 0
    n_forced: int = 0
    n_delist: int = 0
    n_frozen: int = 0
    n_untradeable: int = 0
    u_excl: float = float("nan")
    perm_nan_share: float = float("nan")
    contrib: list = field(default_factory=list)  # (종목, 체결 뒤 비중, 종목 수익, 1배 비용)
    # PO-7 — 회계 창·불변식·보고
    f: int = -1  # 체결일 열
    f_next: int = -1  # 다음 체결일 열(스텝 끝)
    lam_cause: str = ""
    n_tranches: int = 0  # R: 활성 트랜치 수
    cash: float = 0.0  # 체결 뒤 현금 비중
    min_w: float = 0.0
    max_w: float = 0.0
    sector_dev: float = float("nan")  # O: max_g |Σw_g − b_g| (체결 뒤)
    sector_tol: float = float("nan")
    turn_bound: float = float("nan")  # O(구축 뒤): 2/3 + 강제 매도·폐지·표류·재비례 몫
    n_dust: int = 0  # O: 해 직후 0 으로 지운 먼지 비중 수
    dust_mass: float = 0.0  # O: 지운 먼지 비중 합
    delist_mass: float = 0.0  # 폐지 현금화 비중(비용은 있고 회전에는 안 든다)
    n_buy_limit_up: int = 0  # 체결일 종가가 상한가(+29% 이상)인 매수 수
    alarm_top3: list = field(default_factory=list)  # 팩터 몫 > 20% 인 날의 큰 노출 셋 (이름, 노출)


def simulate(ctx: Ctx, kind: str, cell: Cell = PRIMARY, flags: Flags = Flags(), rank_sig=None, ic_by_year=None,
             sigma_r=None, solver_opts=None) -> list[StepRec]:
    """한 북의 회계(PREREG §2 회계·PO-6). ``kind`` ∈ {"R", "O"}. ``rank_sig`` 는 순위에 쓰는 신호(널은 순열).

    단위: 체결 직후 NAV 를 1 로 두고(총액 경로), 스텝 순수익은 r_k(m) = (1 − m·cost1_k)·(1 + gross_k) − 1 로 낸다.
    비용은 NAV 에 비례해 빠지므로 비중 경로는 비용 배수와 무관하다 — 1·2배가 같은 거래열을 쓴다(최적화기는 1배로 푼다).
    """
    assert kind in ("R", "O")
    if kind == "O":
        assert flags == Flags(), "O 는 실제 규칙으로만 돈다"
        assert ic_by_year is not None and sigma_r is not None
    if not flags.drift:
        raise ValueError("표류 끈 규칙은 parity_book 이다")
    rank_sig = ctx.sig if rank_sig is None else rank_sig
    C, Cff = ctx.C, ctx.Cff
    nN, nD = C.shape
    lag = flags.lag
    u = np.zeros(nN)  # 단위 NAV 당 주식 수
    cash = 1.0
    tranches: dict[int, np.ndarray | None] = {}
    out: list[StepRec] = []
    built = False
    for k, t in enumerate(ctx.grid):
        f = t + lag
        f_next = (ctx.grid[k + 1] if k + 1 < len(ctx.grid) else t + STEP) + lag
        if f_next > nD - 1:
            break
        day = ctx.dates[f].date()
        in_u = ctx.in_u(t)
        el = eligible(t, ctx.sig, ctx.adv, ctx.cap, in_u)
        rec = StepRec(k=k, t=t, date=str(ctx.dates[t].date()), gross=float("nan"), cost1=0.0, bench=float("nan"),
                      turnover=0.0, exante_20=float("nan"), factor_share=float("nan"), f=f, f_next=f_next)
        rec.u_excl = u_exclusion(t, ctx.sig, ctx.adv, in_u)
        if kind == "R":
            tranches[k] = tranche_names(t, rank_sig, ctx.sig, ctx.adv, ctx.cap, in_u)
        # 결정일 t 의 표류 비중
        val_t = np.where(u > 0, u * Cff[:, t], 0.0)
        nav_t = val_t.sum() + cash
        w_t = val_t / nav_t
        held = w_t > 0
        # 체결일 f 의 값
        val_f = np.where(u > 0, u * Cff[:, f], 0.0)
        nav_f = val_f.sum() + cash
        w_f = val_f / nav_f
        cash_f = cash / nav_f
        cost_vec = np.zeros(nN)
        # ④ 폐지 청산(마지막 종가, 선형 매도 비용)
        if flags.delist_cash:
            gone = held & delisted(ctx, f)
            if gone.any():
                g = np.flatnonzero(gone)
                if flags.costs:
                    cost_vec[g] += linear_bp(np.full(g.size, "SELL"), ctx.market[g], day, ctx.hs_bp[g]) / 1e4 * w_f[g]
                cash_f += w_f[g].sum()
                rec.delist_mass = float(w_f[g].sum())
                w_f[g] = 0.0
                rec.n_delist = int(g.size)
        held_f = w_f > 0
        # ⑤ 거래 가능 여부(체결일)
        if flags.no_trade:
            tradeable = np.isfinite(C[:, f]) & (np.nan_to_num(ctx.tv_won[:, f]) > 0)
        else:
            tradeable = np.isfinite(Cff[:, f])
        vol = trailing_vol(ctx.ret, t)
        adv_won = ctx.adv[:, t] * 1e6
        costable = np.isfinite(vol) & np.isfinite(adv_won) & (adv_won >= 1e6) & (vol <= 1.0)
        if flags.costs:
            tradeable &= costable
        frozen = held_f & ~tradeable
        rec.n_frozen = int(frozen.sum())
        forced = held_f & ~in_u if flags.forced_sale else np.zeros(nN, bool)
        rec.n_forced = int((forced & tradeable).sum())

        # 목표 비중(체결일 f, 합 ≤ 1)
        trade = True
        if kind == "R":
            want, rec.n_tranches = r_target(ctx, k, t, tranches, flags, f, with_parts=True)
            rec.n_untradeable = int(((want > 0) & ~tradeable).sum())
            # 사전 위험(결정 목표 비중, 결정일 모델, U_t 안, 합 1 로 맞춤)
            w_r = want * in_u
            if w_r.sum() > 0 and el.size:
                w_r = w_r / w_r.sum()
                rows = np.flatnonzero(in_u & ((w_r > 0) | np.isin(np.arange(nN), el)))
                sub = sub_model(ctx.models[t], rows, STEP)
                b = np.zeros(nN)
                b[el] = 1.0 / el.size
                d = (w_r - b)[rows]
                rec.exante_20 = active_risk(sub, d)
                rec.factor_share = factor_share(sub, d)
                if rec.factor_share > FACTOR_ALARM:
                    rec.alarm_top3 = top_exposures(sub, d)
            want = np.where(tradeable, want, 0.0)
        else:
            want, info = o_target(ctx, cell, t, f, el, w_t, held, in_u, tradeable, costable, frozen, vol,
                                  rank_sig, ic_by_year, sigma_r, built, solver_opts)
            for key, v in info.items():
                setattr(rec, key, v)
            trade = rec.status == "optimal"
        renorm = 0.0
        if trade:
            free = tradeable & ~frozen
            room = 1.0 - w_f[frozen].sum()
            s = want[free].sum()
            new = np.where(frozen, w_f, 0.0)
            if s > 0:
                new[free] = want[free] * (room / s)
                renorm = abs(room - s)
            new_cash = 1.0 - new.sum()
        else:  # 최적화 실패 → 거래 안 함(강제 매도만, 대금은 현금)
            new = w_f.copy()
            sold = forced & tradeable
            new[sold] = 0.0
            new_cash = 1.0 - new.sum()
        dw = np.where(tradeable, new - w_f, 0.0)
        new = np.where(tradeable, new, w_f)
        if flags.costs and np.any(dw != 0):
            cost_vec += trade_costs(dw, np.where(costable, adv_won, 1e12), np.where(costable, vol, 0.0), ctx.market,
                                    day, ctx.hs_bp, cell.aum)
        rec.turnover = float(np.abs(dw).sum())
        rec.cost1 = float(cost_vec.sum())
        rec.n_names = int((new > 0).sum())
        # PO-7 M3 — 성과와 무관한 불변식 재료(체결 뒤)
        rec.cash = float(new_cash)
        rec.min_w = float(new.min()) if new.size else 0.0
        rec.max_w = float(new.max()) if new.size else 0.0
        if kind == "O" and rec.status == "optimal" and el.size:
            bvec = np.zeros(nN)
            bvec[el] = 1.0 / el.size
            labs = ctx.lab[:, t]
            groups = {g for g in labs[(new > 0) | (bvec > 0)] if g is not None}
            rec.sector_dev = max((abs(float(new[labs == g].sum() - bvec[labs == g].sum())) for g in groups), default=0.0)
            outside = float(w_f[frozen & ~np.isin(np.arange(nN), np.union1d(el, np.flatnonzero(held & in_u)))].sum())
            rec.sector_tol = cell.band + INV_EPS + outside + renorm
            if built:
                drift = float(np.abs(w_f - w_t).sum())  # t→f 표류 + 폐지 현금화
                rec.turn_bound = (MAX_TURNOVER + drift + float(w_f[forced & tradeable].sum()) + renorm
                                  + 2.0 * rec.dust_mass + INV_EPS)
        if flags.lag:
            rf = ctx.ret[:, f]
            rec.n_buy_limit_up = int(((dw > 0) & np.isfinite(rf) & (rf >= LIMIT_UP_RET)).sum())
        # 보유: 체결 뒤 NAV=1 → 다음 체결 전
        px = np.where(new > 0, Cff[:, f], np.nan)
        u = np.where(new > 0, new / px, 0.0)
        cash = max(new_cash, 0.0)
        g_i = np.where(new > 0, Cff[:, f_next] / Cff[:, f] - 1.0, 0.0)
        rec.gross = float((new * g_i).sum())  # 현금은 수익 0
        nz = np.flatnonzero((new > 0) | (cost_vec > 0))
        rec.contrib = [(int(i), float(new[i]), float(g_i[i]), float(cost_vec[i])) for i in nz]
        # 벤치: eligible(t) 동일가중 매수 보유, 체결일 가격 있는 종목
        bel = el[np.isfinite(C[el, f])]
        if bel.size:
            rec.bench = float(np.mean(Cff[bel, f_next] / C[bel, f] - 1.0))
        built = built or rec.n_names > 0
        out.append(rec)
    return out


def o_target(ctx, cell, t, f, el, w_t, held, in_u, tradeable, costable, frozen, vol_all, rank_sig, ic_by_year,
             sigma_r, built, solver_opts):
    """O 의 목표 비중(§2·PO-6). 문제 종목 P = eligible(t) ∪ (보유 ∩ U_t ∩ 폐지 아님)."""
    from krx_quant_core.portfolio.optimize import Constraints, CostParams, optimize

    nN = ctx.C.shape[0]
    info = {"status": "optimal", "lam": float("nan"), "lam_hit": True, "est_cost": float("nan")}
    alive = ~delisted(ctx, f)
    P = np.union1d(el, np.flatnonzero(held & in_u & alive)).astype(int)
    want = np.zeros(nN)
    if el.size == 0 or P.size == 0:
        info["status"] = "no_universe"
        return want, info
    is_el = np.isin(P, el)
    sub = sub_model(ctx.models[t], P, cell.h_alpha)
    D_daily = sub.D / cell.h_alpha
    z = np.zeros(P.size)
    zv = rank_z(rank_sig[el, t])
    info["perm_nan_share"] = float(np.mean(~np.isfinite(rank_sig[el, t])))
    pos = {int(c): i for i, c in enumerate(P)}
    for c_, zz in zip(el, zv):
        z[pos[int(c_)]] = zz
    ic = ic_by_year[int(ctx.dates[t].year)]
    alpha = alpha_vec(ic, D_daily, z, cell.h_alpha)
    b = np.where(is_el, 1.0 / el.size, 0.0)
    Sb = sub.X @ (sub.F @ (sub.X.T @ b)) + sub.D * b
    w_prev = w_t[P]
    no_trade = ~(tradeable[P] & costable[P]) | frozen[P]
    outside_frozen = frozen & ~np.isin(np.arange(nN), P)
    budget = 1.0 - float(w_t[outside_frozen].sum())
    con = Constraints(budget=budget, long_only=True, max_weight=MAX_W, sector=ctx.lab[P, t], benchmark=b,
                      sector_band=cell.band, max_turnover=MAX_TURNOVER if built else None, max_participation=MAX_PART,
                      no_buy=~is_el, no_trade=no_trade)
    adv_won = np.where(costable[P], ctx.adv[P, t] * 1e6, 1e12)
    vol = np.where(costable[P], vol_all[P], 0.0)
    cp_ = CostParams(trade_date=ctx.dates[f].date(), market=ctx.market[P], half_spread_bp=ctx.hs_bp[P], y=1.0)
    target = float(sigma_r[t]) * math.sqrt(cell.h_alpha / STEP)  # σ_R 은 20일 단위로 받는다

    def solve(lam):
        return optimize(alpha + lam * Sb, sub, w_prev, aum=cell.aum, adv_value=adv_won, daily_vol=vol, cost_params=cp_,
                        constraints=con, risk_aversion=lam, solver_opts=solver_opts)

    res, lam, hit, _, cause = match_lambda(solve, lambda w: active_risk(sub, w - b), target)
    info.update({"lam": lam, "lam_hit": bool(hit), "lam_cause": cause})
    if res is None:
        info["status"] = "solver_fail"
        return want, info
    w_sol = np.asarray(res.weights, dtype=np.float64)
    w_clean, n_dust = clean_dust(w_sol, ctx.Cff[P, t], cell.aum, no_trade)  # PO-7 M2
    want[P] = w_clean
    info.update({"n_dust": n_dust, "dust_mass": float(np.abs(w_clean - w_sol).sum()) / 2.0})
    info["est_cost"] = float(res.est_cost)
    d = w_clean - b
    info["exante_20"] = active_risk(sub, d) * math.sqrt(STEP / cell.h_alpha)
    info["factor_share"] = factor_share(sub, d)
    if info["factor_share"] > FACTOR_ALARM:
        info["alarm_top3"] = top_exposures(sub, d)
    return want, info


def check_invariants(recs: list[StepRec], kind: str, cell: Cell = PRIMARY) -> list[str]:
    """PO-7 M3 — 성과와 무관한 불변식(스텝마다). 위반 문장 목록을 돌려준다(비면 통과). ``--mode run`` 은 하나라도 있으면 결과를 쓰지 않는다.

    - Σw + 현금 = 1, w ≥ −ε. O: max w ≤ 0.10 + ε, 업종 |Σw_g − b_g| ≤ 밴드 + ε + (문제 밖 얼린 몫 + 재비례),
      구축 뒤 회전율 ≤ 2/3 + 표류·폐지 + 강제 매도 + 재비례 + 2·먼지.
    - 비용: cost1 ≥ 0, 회전이 있으면 cost1/회전 ∈ [1.5bp, 300bp](1배).
    - R: 2019 예열 뒤(k ≥ 2) 활성 트랜치 3.
    """
    out = []
    lo_bp, hi_bp = COST_BP_PER_TURN
    for r in recs:
        tag = f"{kind}:{cell.name if kind == 'O' else 'R'} k={r.k} {r.date}"
        tot = r.cash + (sum(w for _, w, _, _ in r.contrib if w > 0))
        if abs(tot - 1.0) > INV_EPS:
            out.append(f"{tag} Σw+현금 {tot:.9f} ≠ 1")
        if r.min_w < -INV_EPS or r.cash < -INV_EPS:
            out.append(f"{tag} 음수 비중 {r.min_w:.3e}·현금 {r.cash:.3e}")
        if r.cost1 < 0:
            out.append(f"{tag} 비용 음수 {r.cost1}")
        traded = r.turnover + r.delist_mass
        if traded > 1e-9:
            bp = r.cost1 / traded * 1e4
            if not (lo_bp - 1e-6 <= bp <= hi_bp):
                out.append(f"{tag} 비용/회전 {bp:.2f}bp 가 [{lo_bp}, {hi_bp}] 밖")
        if kind == "O" and r.status == "optimal" and r.n_names:
            if r.max_w > MAX_W + INV_EPS:
                out.append(f"{tag} max w {r.max_w:.6f} > {MAX_W}")
            if np.isfinite(r.sector_dev) and r.sector_dev > r.sector_tol:
                out.append(f"{tag} 업종 편차 {r.sector_dev:.6f} > 허용 {r.sector_tol:.6f}")
            if np.isfinite(r.turn_bound) and r.turnover > r.turn_bound:
                out.append(f"{tag} 회전율 {r.turnover:.6f} > 한도 {r.turn_bound:.6f}")
        if kind == "R" and r.k >= 2 and r.n_tranches != R_HORIZON // STEP:
            out.append(f"{tag} 활성 트랜치 {r.n_tranches} ≠ 3")
    return out


def r_sigma_series(ctx: Ctx) -> dict:
    """같은 날 R 목표 비중의 사전 20일 능동위험(λ 맞춤 목표 — 진짜 신호 R, 널 O 도 이 값에 맞춘다, PO-6)."""
    recs = simulate(ctx, "R")
    return {r.t: r.exante_20 for r in recs}


# ── 지표·판정(§2·PO-1.6·PO-5) ───────────────────────────────────────────────────


def step_returns(recs: list[StepRec], mult: float) -> pd.DataFrame:
    df = pd.DataFrame({"t": [r.t for r in recs], "date": [r.date for r in recs],
                       "net": [(1.0 - mult * r.cost1) * (1.0 + r.gross) - 1.0 for r in recs],
                       "bench": [r.bench for r in recs]})
    df["a"] = df["net"] - df["bench"]
    return df


def oos(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    return df[(d >= EVAL[0]) & (d <= EVAL[1])].reset_index(drop=True)


def ir(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * ANN)


def payoff_ratio(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos, neg = x[x > 0], x[x < 0]
    return float(pos.mean() / abs(neg.mean())) if pos.size and neg.size else float("nan")


def holding_intervals(recs: list[StepRec], mult: float) -> np.ndarray:
    """보유 구간(종목별 체결 뒤 비중 > 0 이 이어진 스텝) 순기여 = Σ(비중·종목 수익 − m·비용) (PO-6)."""
    cur: dict[int, float] = {}
    last: dict[int, int] = {}
    done = []
    for r in recs:
        seen = set()
        for i, w, g, c in r.contrib:
            if w > 0:
                if i in cur and last.get(i) != r.k - 1:
                    done.append(cur.pop(i))
                cur[i] = cur.get(i, 0.0) + w * g - mult * c
                last[i] = r.k
                seen.add(i)
            elif i in cur:  # 이번 체결에서 다 판 종목: 매도 비용까지 넣고 닫는다
                done.append(cur.pop(i) - mult * c)
        for i in [i for i in cur if i not in seen and last[i] != r.k]:
            done.append(cur.pop(i))
    done += list(cur.values())
    return np.array(done)


def stationary_boot_dir(aO, aR, n=N_BOOT, mean_block=BOOT_MEAN_BLOCK, seed_key="boot-dIR"):
    """ΔIR 의 정상(stationary) 블록 부트스트랩 분포(평균 블록 3스텝, 10,000회 — §2 보고만)."""
    rng = keyed_rng(seed_key)
    T = len(aO)
    p = 1.0 / mean_block
    out = np.empty(n)
    for b in range(n):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for s in range(T):
            if s and rng.random() < p:
                i = rng.integers(T)
            idx[s] = i
            i = (i + 1) % T
        out[b] = ir(aO[idx]) - ir(aR[idx])
    return out


def verdict(o: dict, r: dict, null_o: np.ndarray, null_r: np.ndarray, parity_pass: bool) -> dict:
    """§2·PO-1.6·PO-5 바의 기계적 적용. ``o``/``r`` 은 :func:`book_summary` 결과."""
    ir_r2, ir_o2, ir_r1, ir_o1 = r["IR_2x"], o["IR_2x"], r["IR_1x"], o["IR_1x"]
    x = min(ir_r2 / 3.0, NI_X_CAP)
    q_r, q_o = float(np.quantile(null_r, NULL_Q)), float(np.quantile(null_o, NULL_Q))
    bars = {
        "SIG": bool(ir_r2 > q_r),
        "B-NI-1": bool(ir_o2 - ir_r2 >= -x),
        "B-NI-2": bool(ir_o1 - ir_r1 >= -x),
        "B-NI-3": bool(o["sd_a_2x"] <= NI3_RATIO * r["sd_a_2x"]),
        "B-NI-4": bool(ir_o2 > q_o),
        "B-NI-5": bool(o["EV_2x"] > 0),
    }
    void = {"optimal_share<0.95": o["optimal_share"] < OPT_OK_MIN, "parity_fail": not parity_pass,
            "oos_steps<40": min(o["oos_steps"], r["oos_steps"]) < OOS_MIN, "lam_fail>0.10": o["lam_fail_share"] > LAM_FAIL_MAX}
    if any(void.values()):
        v = "VOID"
    elif not bars["SIG"]:
        v = "판단 불가(신호 약함)"
    elif all(bars[b] for b in ("B-NI-1", "B-NI-2", "B-NI-3", "B-NI-4", "B-NI-5")):
        v = "비열등 통과"
    else:
        v = "채택 불가"
    return {"x": x, "null_q95_R": q_r, "null_q95_O": q_o, "dIR_2x": ir_o2 - ir_r2, "dIR_1x": ir_o1 - ir_r1,
            "bars": bars, "void": void, "verdict": v,
            "failed": [b for b, ok in bars.items() if not ok and b != "SIG"],
            "B-NI-3_label": "표본 안에서 유리(v4 는 이 구간에서 튜닝됐다, PO-5.1)",
            "B-NI-1_note": "열등 아님 선별(검정 아님) — 참 ΔIR = −x 이면 이 선별을 통과할 확률은 0.5 다(PO-1.6)"}


def book_summary(recs: list[StepRec]) -> dict:
    """한 북의 OOS 요약(1·2배)."""
    out = {}
    for m in (1.0, 2.0):
        df = oos(step_returns(recs, m))
        tag = f"{int(m)}x"
        out[f"IR_{tag}"] = ir(df["a"])
        out[f"EV_{tag}"] = float(df["a"].mean())
        out[f"sd_a_{tag}"] = float(df["a"].std(ddof=1))
        out[f"step_payoff_{tag}"] = payoff_ratio(df["a"])
        hi = holding_intervals([r for r in recs if EVAL[0] <= r.date <= EVAL[1]], m)
        out[f"interval_payoff_{tag}"] = payoff_ratio(hi)
        srt = np.sort(hi)[::-1]
        out[f"interval_EV_drop_top5_{tag}"] = float(srt[5:].mean()) if srt.size > 5 else float("nan")
        out[f"interval_skew_{tag}"] = float(stats.skew(hi)) if hi.size > 2 else float("nan")
        out[f"folds_IR_{tag}"] = {str(y): ir(df.loc[pd.to_datetime(df["date"]).dt.year == y, "a"]) for y in FOLD_YEARS}
        out[f"abs_net_mean_{tag}"] = float(df["net"].mean())
    o = [r for r in recs if EVAL[0] <= r.date <= EVAL[1]]
    out["oos_steps"] = len(o)
    out["optimal_share"] = float(np.mean([r.status == "optimal" for r in o])) if o else float("nan")
    out["lam_fail_share"] = float(np.mean([not r.lam_hit for r in o])) if o else float("nan")
    fs = np.array([r.factor_share for r in recs], dtype=float)
    out["factor_share_q10_50_90"] = [float(np.nanquantile(fs, q)) for q in (0.1, 0.5, 0.9)] if np.isfinite(fs).any() else None
    alarm = [r for r in recs if np.isfinite(r.factor_share) and r.factor_share > FACTOR_ALARM]
    out["factor_alarm_n"], out["factor_alarm_dates"] = len(alarm), [r.date for r in alarm]
    out["factor_alarm_top3"] = [[r.date, r.alarm_top3] for r in alarm]
    out["lam_causes"] = {c: int(sum(r.lam_cause == c for r in o)) for c in ("up", "down", "gap", "solver")} if o else {}
    out["n_buy_limit_up"] = int(sum(r.n_buy_limit_up for r in recs))
    out["n_dust"] = int(sum(r.n_dust for r in recs))
    ex = np.array([r.exante_20 for r in o], dtype=float)
    out["exante_20_mean"] = float(np.nanmean(ex)) if o else float("nan")
    out["realized_over_exante"] = out["sd_a_1x"] / out["exante_20_mean"] if o else float("nan")
    out["turnover_mean"] = float(np.mean([r.turnover for r in o])) if o else float("nan")
    out["cost1_mean"] = float(np.mean([r.cost1 for r in o])) if o else float("nan")
    out["est_cost_mean"] = float(np.nanmean([r.est_cost for r in o])) if o else float("nan")
    out["names_mean"] = float(np.mean([r.n_names for r in o])) if o else float("nan")
    for key in ("n_forced", "n_delist", "n_frozen", "n_untradeable"):
        out[key] = int(sum(getattr(r, key) for r in recs))
    out["u_excl_mean"] = float(np.nanmean([r.u_excl for r in recs]))
    return out


# ── 순열 널(§2·PO-1.7) ─────────────────────────────────────────────────────────

_G: dict = {}


def _perm_task(a):
    r, kind = a
    ctx, cell, icy, sr = _G["ctx"], _G["cell"], _G["ic"], _G["sr"]
    pi = keyed_rng("perm", r).permutation(ctx.sig.shape[0])
    recs = simulate(ctx, kind, cell, rank_sig=ctx.sig[pi], ic_by_year=icy, sigma_r=sr)
    nan_share = float(np.nanmean([x.perm_nan_share for x in recs])) if kind == "O" else float("nan")
    viol = check_invariants(recs, kind, cell)
    return r, kind, ir(oos(step_returns(recs, 2.0))["a"]), nan_share, viol


def permutation_nulls(ctx, cell, ic_by_year, sigma_r, workers: int, n: int = N_PERM):
    """같은 π 를 O 와 R 에 쓴다(시드 ``keyed_rng("perm", r)``). 2배 IR 분포를 돌려준다."""
    _G.update({"ctx": ctx, "cell": cell, "ic": ic_by_year, "sr": sigma_r})
    tasks = [(r, k) for r in range(n) for k in ("R", "O")]
    with mp.get_context("fork").Pool(workers) as pool:
        res = list(pool.imap_unordered(_perm_task, tasks, chunksize=1))
    _G.clear()
    out = {"R": np.full(n, np.nan), "O": np.full(n, np.nan), "O_perm_nan_share": np.full(n, np.nan), "violations": []}
    for r, kind, v, s, viol in res:
        out[kind][r] = v
        out["violations"] += [f"perm{r} {x}" for x in viol]
        if kind == "O":
            out["O_perm_nan_share"][r] = s
    return out


# ── 실제 규칙과의 차이(PO-1.4, 보고만) ─────────────────────────────────────────


TOGGLES = (
    ("① t+1 체결", None),
    ("② 스텝 사이 표류", Flags(lag=1, drift=True, forced_sale=False, delist_cash=False, no_trade=False, costs=False)),
    ("③ 위험모델 이탈 강제 매도", Flags(lag=1, drift=True, forced_sale=True, delist_cash=False, no_trade=False, costs=False)),
    ("④ 폐지 대금 현금", Flags(lag=1, drift=True, forced_sale=True, delist_cash=True, no_trade=False, costs=False)),
    ("⑤ no_trade", Flags(lag=1, drift=True, forced_sale=True, delist_cash=True, no_trade=True, costs=False)),
    ("⑥ 비용(1배)", Flags()),
)


def toggle_report(ctx: Ctx) -> list[dict]:
    """패리티 모드에서 한 규칙씩 켜서(누적) R 의 스텝 능동수익 차이를 낸다(B 실행 때만 — 수익을 쓰므로 패리티 모드에서는 부르지 않는다)."""
    prev = parity_book(ctx, 0).set_index("t")
    prev_a = prev["book"] - prev["bench"]
    out = []
    for name, fl in TOGGLES:
        if fl is None:
            cur = parity_book(ctx, 1).set_index("t")
            cur_a = cur["book"] - cur["bench"]
            ev = {"steps": int(len(cur))}
        else:
            recs = simulate(ctx, "R", flags=fl)
            df = step_returns(recs, 1.0).set_index("t")
            cur_a = df["a"]
            ev = {k: int(sum(getattr(r, k) for r in recs)) for k in ("n_forced", "n_delist", "n_frozen")}
        d = (cur_a - prev_a).dropna()
        out.append({"rule": name, "events": ev, "diff_sum": float(d.sum()), "diff_mean": float(d.mean()),
                    "diff_max_abs": float(d.abs().max()) if len(d) else float("nan"), "steps": int(len(d))})
        prev_a = cur_a
    return out


# ── 데이터(실데이터 — DB) ─────────────────────────────────────────────────────


def _db():
    import pead_optimizer_po2 as P2

    return P2._db()


def market_and_delisting(codes) -> tuple[np.ndarray, np.ndarray, dict]:
    """시장(``stocks``(현재) 우선, 없으면 ``delisted_stocks``)과 폐지 마지막 거래일. 모르는 시장은 KOSDAQ(비용 큰 쪽, PO-6)."""
    from krx_quant_core.market.codes import Market

    con, cur = _db()
    cur.execute("SELECT code, market FROM stocks")
    cur_m = dict(cur.fetchall())
    cur.execute("SELECT code, market, last_trade_date FROM delisted_stocks")
    rows = cur.fetchall()
    con.close()
    del_m = {c: m for c, m, _ in rows}
    last = {c: d for c, _, d in rows if d is not None}
    mk, unknown = [], 0
    for c in codes:
        raw = cur_m.get(str(c), del_m.get(str(c)))
        try:
            m = Market.parse(raw).value if raw is not None else None
        except ValueError:
            m = None
        if m is None or m == "KONEX":
            unknown += m is None
            m = "KOSDAQ"
        mk.append(m)
    return np.array(mk, dtype=object), last, {"market_unknown": unknown, "delisted_with_last_date": len(last)}


def model_cache_name(fp: list, core_ref: str, stem: str = "models_v4_grid") -> str:
    """PO-7 M1 — 모델 캐시 이름에 표 지문 해시·core SHA 를 넣는다. 표가 바뀌면(PR #12) 옛 캐시를 읽지 않는다."""
    import hashlib

    h = hashlib.sha256(json.dumps(fp, default=str).encode()).hexdigest()[:12]
    return f"{stem}_fp{h}_core{core_ref}.pkl"


def build_ctx(cache: Path, po3_cache: Path, workers: int, log, end: str = END_B, core_ref: str = "13c56a7"):
    """B 입력: v4 와 같은 로더·가드·노출, 격자 t_k(2019-08-01 이후 첫 거래일 + 20k)마다 v4 모델.

    모델 캐시는 ``daily_bars_adjusted`` 지문(2019-01-01~END)·core SHA 에 묶인다(:func:`model_cache_name`).
    """
    import pead_optimizer_po3 as P3
    import pead_optimizer_po4 as P4

    assert end == END_B, "B 는 2023-12-28 까지만 쓴다(§2)"
    fp = P4.fingerprint("daily_bars_adjusted", end)
    x0 = P4.load_inputs(end)
    x, seam, seam_info = P4.guarded(x0, end)
    c = P4.prep(x, "2019-07-01", "2023-10-31", 52, end)
    first = c["first"]
    dates = c["dates"]
    nD = c["R"].shape[1]
    t_0 = int(np.searchsorted(dates, np.datetime64(GRID_START)))
    grid = list(range(t_0, nD, STEP))
    mpath = cache / model_cache_name(fp, core_ref)
    models = P3.build_models(c, grid, workers, mpath, log, vra=None, shrink=V4_SHRINK)
    old = pickle.load(open(po3_cache / "models_vraoff_s0.1.pkl", "rb"))
    common = sorted(set(old) & set(models))
    bit_same = bool(common) and all(
        all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(old[t][:4], models[t][:4])) and old[t][4] == models[t][4]
        for t in common)
    sig_all, sig_info = P4.yoy_signal_strict(x.codes, x.dates, end)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    market, last, mk_info = market_and_delisting(x.codes)
    didx = pd.DatetimeIndex(dates)
    delist_after = np.full(len(x.codes), 10**9, dtype=np.int64)
    for i, cc in enumerate(x.codes):
        d = last.get(str(cc))
        if d is not None:
            j = int(didx.searchsorted(pd.Timestamp(d), side="right")) - 1  # 마지막 거래일 열(그 날까지 거래)
            if j < len(didx) - 1:  # 창 끝 이후 폐지는 창 안에서 폐지가 아니다
                delist_after[i] = j
    sl = slice(first, None)
    ctx = Ctx(dates=didx, C=x.close[:, sl], tv_won=x.trade_value[:, sl], adv=adv_all[:, sl], cap=c["cap"],
              sig=sig_all[:, sl], ret=c["R"], lab=c["lab"], market=market, grid=grid, models=models,
              delist_after=delist_after)
    info = {"seam": {k: seam_info[k] for k in ("seam_cells", "seam_cells_by_source")}, "signal": sig_info,
            "market": mk_info, "grid": [str(dates[grid[0]].date()), str(dates[grid[-1]].date()), len(grid)],
            "models_bit_same_PO3_common": bit_same, "models_common_n": len(common), "first": int(first),
            "fingerprint_models": fp, "model_cache": mpath.name, "prep": c,
            "adv_full": adv_all, "trade_value_full": x.trade_value, "close_full": x.close}
    return ctx, info, x


# ── 실행 ───────────────────────────────────────────────────────────────────────

OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"


def run_parity(args, log) -> int:
    """PO-1.4 패리티 검사 — 차이·개수만 기록한다(성과 숫자 없음)."""
    import pead_optimizer_po4 as P4

    t0 = time.time()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fp0 = P4.fingerprint("daily_bars_adjusted", END_B)
    ctx, info, x = build_ctx(cache, Path(args.po3_cache), args.workers, log, core_ref=args.core_ref)
    log(f"격자 {info['grid']}, PO-3 캐시와 공통 모델 {info['models_common_n']}개 비트 동일 {info['models_bit_same_PO3_common']}")
    pc = full_parity(ctx)
    fp1 = P4.fingerprint("daily_bars_adjusted", END_B)
    res = {"mode": "parity", "core_ref": args.core_ref, "end": END_B, "table": "daily_bars_adjusted(지금 표 — PR #12 전)",
           "fingerprint_before": fp0, "fingerprint_after": fp1, "fingerprint_same": fp0 == fp1,
           "grid": info["grid"], "models_bit_same_PO3_common": info["models_bit_same_PO3_common"],
           "models_common_n": info["models_common_n"], "seam": info["seam"], "market": info["market"],
           "parity": pc, "elapsed_s": time.time() - t0,
           "note": "성과 숫자(book·bench·IR)는 계산 중에만 쓰고 출력·저장하지 않았다. 차이·개수만 적는다."}
    (OUT / "B_parity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "B_parity.md").write_text(render_parity(res))
    log(f"패리티 {'통과' if pc['pass'] else '실패'}: 최대 차 book {pc.get('max_abs_diff_book')}, bench {pc.get('max_abs_diff_bench')}")
    return 0 if pc["pass"] else 1


def render_parity(r: dict) -> str:
    p = r["parity"]
    L = ["# 단계 B 패리티 검사 (PO-1.4) — 코드 검사, 성과 숫자 없음", "",
         f"core `{r['core_ref']}`, END {r['end']}, 표 {r['table']}. 격자 {r['grid'][0]}~{r['grid'][1]} ({r['grid'][2]}일).",
         "하네스 패리티 모드(지연 0·무비용·강제 매도 끔·현금 없음, `start_index` = 첫 격자일, `min_names=20`)와 core "
         "`staggered_tranche_backtest(top_n=10, cap_rank=(20,100), horizon=60, step=20, adv_floor=20000, delisting_exit=True)` 의 스텝별 비교.",
         "**book·bench 값 자체는 출력·저장하지 않았다**(B 는 수리된 표 뒤에 연다, PO-5.2).", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 스텝 수(하네스/core) | {p['steps_harness']}/{p['steps_core']} (같은 날짜 {p['same_steps']}) |",
         f"| 기간 | {p.get('first_date')} ~ {p.get('last_date')} |",
         f"| book 최대 절대차 | {p.get('max_abs_diff_book', float('nan')):.3e} |",
         f"| bench 최대 절대차 | {p.get('max_abs_diff_bench', float('nan')):.3e} |",
         f"| NaN 불일치 | {p.get('nan_mismatch')} |",
         f"| n_universe 같음 | {p.get('n_universe_same')} |",
         f"| 적격 집합 같음(격자 전부) | {p.get('eligible_same_all_grid')} |",
         f"| ADV 바이트 같음(core `_trailing_adv`) | {p.get('adv_bytes_same')} |",
         "| 바 | 1e-10 |", "",
         f"**패리티: {'통과' if p['pass'] else '실패'}.**", "",
         f"- 표 지문 시작=끝: {r['fingerprint_same']} {r['fingerprint_before']}.",
         f"- 격자 모델이 PO-3 캐시(v4)와 공통 {r['models_common_n']}개 비트 동일: {r['models_bit_same_PO3_common']}.",
         f"- 이음매 가드: {r['seam']}. 시장 구분: {r['market']}.",
         "- 수리된 표(PR #12)가 배포되면 같은 검사를 다시 통과해야 B 를 돌린다(PO-5.2-4).", "",
         f"_{r['elapsed_s']:.1f}초_"]
    return "\n".join(L) + "\n"


def git_sha() -> str:
    import subprocess

    r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain", "--", "research/experiments"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return r.stdout.strip() + ("-dirty" if dirty else "")


def fingerprint_range(table: str, lo: str, hi: str) -> list:
    """core ``_fingerprint`` 를 임의 구간으로(IC 패널 2016-11-01~ 용, PO-7 MINOR 2)."""
    import pead_optimizer_po2 as P2
    from krx_quant_core.data.daily import _fingerprint

    con, _ = P2._db()
    try:
        return _fingerprint(con, table, pd.Timestamp(lo).date(), pd.Timestamp(hi).date())
    finally:
        con.close()


def gate_b(a1r: dict | None, fp_now: list, report_impl: dict) -> None:
    """PO-5.2·PO-7 실행 전 관문(순수 함수 — 시험 가능). 막히면 ``SystemExit``."""
    import pead_optimizer_b_report as BR

    missing = [k for k in BR.REQUIRED if not report_impl.get(k)]
    if missing:
        raise SystemExit(f"보고만 항목 미구현 {missing} — B 를 돌리지 않는다(PO-7 M4)")
    if a1r is None:
        raise SystemExit("A1_repaired.json 이 없다 — PR #12 배포와 강건성 재실행(PO-5.2) 전에는 B 를 돌리지 않는다")
    if "STOP" in a1r or a1r.get("verdict") != "통과":
        raise SystemExit(f"강건성 재실행 판정 {a1r.get('verdict')!r} — '취약', B 는 역사 구간에서 열지 않는다(PO-5.2-3)")
    rp = a1r.get("repaired")
    if not rp or rp.get("label") is not None or not rp.get("B_opens") or not rp.get("fingerprint_differs_from_pre"):
        raise SystemExit("A1_repaired.json 의 강건성 분기가 B 를 열지 않는다(라벨·지문) — 멈춘다(PO-5.2-3)")
    if fp_now != a1r["fingerprint_before"]["daily_bars_adjusted"]:
        raise SystemExit("지금 표 지문이 강건성 재실행 때와 다르다 — 멈춘다(PO-5.2-4)")


def run_b(args, log) -> int:  # pragma: no cover - B 는 PR #12 배포·강건성 재실행 뒤에만(PO-5.2)
    """단계 B 전체. PO-5.2 순서·PO-7 불변식·보고만 구현 여부를 코드로 막는다."""
    import pead_optimizer_b_report as BR
    import pead_optimizer_po4 as P4
    from swing_it.diagnostics.trials import record_trial

    t_start = time.time()
    rep = OUT / "A1_repaired.json"
    a1r = json.loads(rep.read_text()) if rep.exists() else None
    fp_now = P4.fingerprint("daily_bars_adjusted", "2026-09-11")
    gate_b(a1r, fp_now, BR.IMPLEMENTED)
    if (OUT / "B.json").exists():
        raise SystemExit("B.json 이 이미 있다 — B 는 한 번만 돈다")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fp0 = P4.fingerprint("daily_bars_adjusted", END_B)
    fp_ic0 = fingerprint_range("daily_bars_adjusted", IC_LOAD_START, END_B)
    ctx, info, x = build_ctx(cache, Path(args.po3_cache), args.workers, log, core_ref=args.core_ref)
    pc = full_parity(ctx)
    if not pc["pass"]:
        raise SystemExit("패리티 실패 — VOID(§2)")
    ic_by, ic_info = ic_by_year_for(args, log)
    sigma_r = r_sigma_series(ctx)
    viol: list[str] = []
    steps_checked = 0
    res = {"mode": "run", "parity": pc, "cells": {}, "ic": ic_info}
    for cell in SENSITIVITY + (PRIMARY,):  # GUARDRAILS §4.5 ② — 격자를 1차보다 먼저
        record_trial("pead_optimizer", {"stage": "B", "cell": cell.name, "core": args.core_ref},
                     note="B 민감도" if cell != PRIMARY else "B 1차")
        recs_o = simulate(ctx, "O", cell, ic_by_year=ic_by[cell.h_alpha], sigma_r=sigma_r)
        viol += check_invariants(recs_o, "O", cell)
        steps_checked += len(recs_o)
        res["cells"][cell.name] = {"O": book_summary(recs_o)}
        if cell == PRIMARY:
            recs_r = simulate(ctx, "R")
            viol += check_invariants(recs_r, "R")
            steps_checked += len(recs_r)
            o_s, r_s = book_summary(recs_o), book_summary(recs_r)
            nulls = permutation_nulls(ctx, cell, ic_by[cell.h_alpha], sigma_r, args.workers)
            viol += nulls["violations"]
            aO = oos(step_returns(recs_o, 2.0))["a"].to_numpy()
            aR = oos(step_returns(recs_r, 2.0))["a"].to_numpy()
            boot = stationary_boot_dir(aO, aR)
            res["primary"] = {"O": o_s, "R": r_s, "verdict": verdict(o_s, r_s, nulls["O"], nulls["R"], pc["pass"]),
                              "rho": float(np.corrcoef(aO, aR)[0, 1]), "boot_dIR_q05_50_95":
                              [float(np.quantile(boot, q)) for q in (0.05, 0.5, 0.95)],
                              "null_O_perm_nan_share_mean": float(np.nanmean(nulls["O_perm_nan_share"])),
                              "toggles": toggle_report(ctx)}
            res["primary"]["verdict"]["ic_zero_folds"] = ic_info["zero_folds"][cell.h_alpha]
            res["primary"]["report"] = b_report(ctx, info, x, recs_o, recs_r, aO - aR, log)
    if viol:
        (cache / "B_invariant_violations.txt").write_text("\n".join(viol))
        raise SystemExit(f"불변식 위반 {len(viol)}건 — 결과를 쓰지 않는다(PO-7 M3). 목록: {cache / 'B_invariant_violations.txt'}")
    fp1 = P4.fingerprint("daily_bars_adjusted", END_B)
    fp_ic1 = fingerprint_range("daily_bars_adjusted", IC_LOAD_START, END_B)
    if fp0 != fp1 or fp_ic0 != fp_ic1:
        raise SystemExit("실행 중 표 지문이 바뀌었다 — 결과를 쓰지 않는다(PO-4.2)")
    res["invariants"] = {"violations": 0, "steps_checked": steps_checked}
    res["code"] = {"swing_it_sha": git_sha(), "core_ref": args.core_ref, "model_cache": info["model_cache"]}
    res["fingerprints"] = {"daily_bars_adjusted_b_range": fp0, "daily_bars_adjusted_ic_range": fp_ic0,
                           "daily_bars_adjusted_full_now": fp_now, "A1_repaired": a1r["fingerprint_before"]["daily_bars_adjusted"],
                           "same_as_A1_repaired": fp_now == a1r["fingerprint_before"]["daily_bars_adjusted"],
                           "models": info["fingerprint_models"]}
    res["elapsed_s"] = time.time() - t_start
    (OUT / "B.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "B.md").write_text(BR.render_b(json.loads((OUT / "B.json").read_text())))
    return 0


def b_report(ctx, info, x, recs_o, recs_r, d2x, log) -> dict:  # pragma: no cover - B 실행 때만
    """보고만 항목(PO-7.4) — ``pead_optimizer_b_report``."""
    import pead_optimizer_b_report as BR
    import pead_optimizer_po2 as P2
    from krx_quant_core.risk.factor_model import factor_returns

    first = info["first"]
    kospi = np.asarray(x.market_ret[first:], dtype=np.float64)
    con, cur = P2._db()
    cur.execute("SELECT date, close FROM sector_index WHERE code = '101' AND date BETWEEN %s AND %s",
                (str(ctx.dates[0].date()), END_B))
    kq = pd.DataFrame(cur.fetchall(), columns=["date", "close"])
    con.close()
    lvl = pd.Series(kq["close"].to_numpy(float), index=pd.to_datetime(kq["date"])).reindex(ctx.dates)
    kosdaq = lvl.pct_change(fill_method=None).to_numpy()
    ro, rr = [r for r in recs_o if EVAL[0] <= r.date <= EVAL[1]], [r for r in recs_r if EVAL[0] <= r.date <= EVAL[1]]
    fac = BR.factor_step_returns(ctx, ro, kospi, kosdaq, trailing_vol, ADV_FLOOR)
    led = BR.ledger_n()
    hur = BR.alpha_hurdle(led["N"])
    out = {"ledger": led, "hurdle": hur, "factor_alpha": {}, "vs_kospi": {}, "deflation": {}, "attribution": {}}
    for b, recs in (("O", ro), ("R", rr)):
        out["factor_alpha"][b] = {}
        for m in (1.0, 2.0):
            net = step_returns(recs, m)[["t", "date", "net"]]
            out["factor_alpha"][b][f"{int(m)}x"] = BR.factor_alpha_report(net, fac)
        net2 = step_returns(recs, 2.0)[["t", "date", "net"]]
        out["vs_kospi"][b] = BR.vs_kospi(net2, fac)
        out["deflation"][b] = BR.deflation(net2["net"], led["N"])
    out["qualification"] = BR.pead_qualification(out["factor_alpha"]["O"]["2x"]["t_alpha"],
                                                 out["factor_alpha"]["R"]["2x"]["t_alpha"], hur["hurdle"])
    prep = info["prep"]
    fr = factor_returns(prep["R"], prep["Xs"], prep["lab"], prep["cap"], prep["msk"])
    for b, recs in (("O", ro), ("R", rr)):
        out["attribution"][b] = BR.attribution_report(
            ctx, recs, fr, prep, lambda t: eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t)))
    out["f_boot"] = BR.f_boot(d2x)
    out["factor_index_missing_days"] = int(fac["index_missing_days"].sum())
    log(f"보고만 끝: 원장 N {led['N']}, f_boot {out['f_boot']['f_boot']}")
    return out


def ic_by_year_for(args, log) -> tuple[dict, dict]:  # pragma: no cover - B 실행 때만
    """폴드(연도)마다 IC_k — 2017-01-02 부터 폴드 시작 (H+2) 거래일 앞까지(PO-6). H ∈ {20, 60}.

    PO-7 MINOR 2: IC 패널에도 위험모델과 같은 이음매 가드(PO-3.3)를 건다. 폐지는 마지막 종가(:func:`daily_rank_ic`).
    IC_k 가 0 으로 잘린 폴드는 α ≡ 0 이라 λ 맞춤이 그 폴드 전부에서 실패한다(PO-7.5) — 그 폴드를 적어 둔다.
    """
    import pead_optimizer_po4 as P4
    from krx_quant_core.data.risk_inputs import load_risk_inputs

    xi0 = load_risk_inputs(IC_LOAD_START, END_B)
    xi, _, seam_info = P4.guarded(xi0, END_B)
    sig, _ = P4.yoy_signal_strict(xi.codes, xi.dates, END_B)
    adv = _trailing_adv(xi.trade_value / 1e6, len(xi.dates), 20, xi.close)
    d = pd.DatetimeIndex(xi.dates)
    lo = int(d.searchsorted(pd.Timestamp(IC_START)))
    out, info = {}, {"seam_cells": seam_info["seam_cells"], "n_days": {}, "zero_folds": {}}
    for h in (20, 60):
        days, ics = daily_rank_ic(xi.close, adv, xi.cap, sig, lo, len(d) - 1, h)
        out[h] = {y: fold_ic(days, ics, int(d.searchsorted(pd.Timestamp(f"{y}-01-01"))), h)
                  for y in (2019,) + FOLD_YEARS}
        info["n_days"][h] = int(len(ics))
        info["zero_folds"][h] = [y for y, v in out[h].items() if v <= 0.0]
        log(f"IC 학습 H={h}: 일 {len(ics)}")
    info["ic_by_year"] = {h: {str(y): v for y, v in out[h].items()} for h in out}
    return out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("parity", "run"), required=True)
    ap.add_argument("--cache", required=True, help="레포 밖")
    ap.add_argument("--po3-cache", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--core-ref", default="13c56a7")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assert repo not in Path(args.cache).resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    return run_parity(args, log) if args.mode == "parity" else run_b(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
