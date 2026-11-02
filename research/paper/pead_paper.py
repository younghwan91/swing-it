"""PEAD 순위 북 R 전진 모의매매 — 팔 ``pead_rank_F1``(실주문 없음). 사전등록 ``research/paper/PEAD_FORWARD_PREREG.md``.

규칙은 단계 B(swing-it PR #13, ``pead_optimizer_b.py`` 의 ``simulate(kind="R")``·``r_target``·``tranche_names``·``eligible``)와 같다:
3트랜치 스태거드(20거래일 간격·지평 60), 트랜치 = 적격 종목의 분기 순이익 YoY 상위 10 동일가중, 적격 = 신호 유한 ∧ 20일 ADV ≥ 200억
∧ U_t(WICS 업종·시총·종가 있음) 안 시총 21~100위, t 종가 결정 → t+1 종가 체결, 폐지 = 마지막 종가 현금화, 비용 = core 정가 모델(1배 기록, 판정 2배).
B 와 다른 곳(데이터 경로만, 사전등록 §2 표)은 정규장 가격(2026-09-14~ 애프터마켓 보정)·U_t 재구성·시총 정문이다.

매 실행은 ``SIM_FROM`` 격자일부터 DB 마지막 날(또는 ``--asof``)까지 같은 규칙을 다시 시뮬레이션한다(결정적·인과적 — 입력을
asof 이하로 자른다; 정규장 종가 항등식의 마지막 날 예외는 :func:`load_panel`). 마지막 날 행 하나를 **덧붙이기 전용 장부**(``LEDGER``)에 남긴다. (date, arm) 이 이미 있으면 쓰지 않고,
``ARM_START`` 전 날짜는 쓰지 않는다. 판정 통계는 행의 ``step_closed``(스텝이 끝난 다음 거래일 행에 한 번 적힌다)의 ``a_2x`` 다.

사용: ``set -a; . ~/git/quant-airflow/.env; set +a; uv run --extra pg python research/paper/pead_paper.py [--asof YYYY-MM-DD,...] [--ledger PATH]``
과거 재현(설계 확인용, 장부에 쓰지 않음): ``--replicate 2019-08-01:2023-12-28 --out <json>``.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]

from krx_quant_core.backtest.crosssectional import _trailing_adv  # noqa: E402 — B 와 같은 ADV 정의(t 제외 20일 nanmean)
from krx_quant_core.costs.model import DEFAULT_COMMISSION_RATE, DEFAULT_KOSDAQ_SLIPPAGE_PCT, DEFAULT_KOSPI_SLIPPAGE_PCT  # noqa: E402
from krx_quant_core.costs.tax import sell_tax_rate  # noqa: E402

from swing_it.units import eok, mil  # noqa: E402

# ── 사전등록 상수(PEAD_FORWARD_PREREG.md §2 — 도중에 바꾸지 않는다) ─────────────────────────────

ARM = "pead_rank_F1"
LEDGER = "research/paper/pead_rank_f1_ledger.jsonl"
ARM_START = "2026-11-05"          # 첫 결정일 t_{k_s} — 이 전 날짜 행은 장부에 쓰지 않는다(소급 성적 없음)
GRID_ANCHOR = "2026-10-07"        # B 격자(2019-08-01 첫 거래일 + 20k, daily_bars_adjusted 달력)의 k=88 — 위상을 B 와 같게
SIM_FROM = "2026-08-06"           # 예열 시작 격자일 = k_s − 3(트랜치 3개·직전 보유가 차도록). GRID_ANCHOR − 40거래일이어야 한다
PANEL_START = "2026-03-03"        # 63일 변동성·20일 ADV·시총 원 종가가 SIM_FROM 전에 차도록
REPLICATE_GRID_START = "2019-08-01"
STEP, HORIZON, TOP_N, MIN_NAMES = 20, 60, 10, 20
CAP_RANK = (20, 100)              # 시총 21~100위
ADV_FLOOR = eok(200)              # 20일 평균 거래대금(백만원) ≥ 200억
AUM = 3e8                         # 충격 비용 계산용 고정 AUM(복리로 늘리지 않는다, B Cell 기본)
COST_MULTS = (1.0, 2.0)           # 1배 기록 · 2배 판정
VOL_WIN, VOL_MIN_OBS = 63, 20
SEAM_THR = 0.2                    # 이음매 가드(PO-2.1): |ρ_adj/ρ_raw − 1| > 0.2 ∧ 사건 없음 → 그 날 수익 0
CAP_MAX = 2e15                    # 시총 2,000조원 초과 = 오류(leader21 과 같다)
HALF_SPREAD_BP = {"KOSPI": float(DEFAULT_KOSPI_SLIPPAGE_PCT) * 1e4, "KOSDAQ": float(DEFAULT_KOSDAQ_SLIPPAGE_PCT) * 1e4}
COMMISSION_BP = float(DEFAULT_COMMISSION_RATE) * 1e4
WON_PER_MIL = 1e6
WICS_STALE_DAYS = 21
STATEMENT_TIMEOUT_S = 600   # 정정 F1-1 M3: 패널이 날마다 길어진다(실측 ~0.39초/거래일 × 2 조회) — 300초면 2029-09 께 닿는다
PRICE = {"session": "regular", "unpatched": "nan", "aftermarket_fix": True}   # 정문 인자(사전등록 §2)
RULES = {"step": STEP, "horizon": HORIZON, "top_n": TOP_N, "min_names": MIN_NAMES, "cap_rank": list(CAP_RANK),
         "adv_floor_mil": ADV_FLOOR, "aum": AUM, "fill": "t+1 close", "costs": "core trade_cost_bp y=1 (1x), judged at 2x",
         "grid_anchor": GRID_ANCHOR, "sim_from": SIM_FROM, **PRICE}


# ── 비용(core ``costs.impact.trade_cost_bp`` 를 그대로 옮김 — fcs 의 krx-quant-core 0.8.0 에는 impact 모듈이 없다) ─────────


def trade_cost_bp(side, notional, adv_value, daily_vol, market, trade_date: date, *, half_spread_bp, y: float = 1.0) -> np.ndarray:
    """편도 비용(bp) = 수수료 + 매도세(매도만) + 반스프레드 + 제곱근 충격 y·σ·√(|Q|/ADV). core 8286e3e ``costs/impact.py`` 와 같은 식·같은 가드.

    ``tests/test_pead_paper.py`` 가 core 에 impact 모듈이 있으면 같은 입력에서 값이 같은지 본다.
    """
    sides = np.char.upper(np.asarray(side, dtype=str))
    if np.any(~np.isin(sides, ("BUY", "SELL"))):
        raise ValueError("side 는 'BUY'/'SELL'")
    hs = np.asarray(half_spread_bp, dtype=np.float64)
    if np.any(hs < 0):
        raise ValueError("half_spread_bp 는 0 이상")
    liq = np.asarray(adv_value, dtype=np.float64)
    if np.any(liq < WON_PER_MIL):
        raise ValueError("adv_value 는 원 단위(1e6 원 미만 = 백만원을 그대로 넣었다)")
    vol = np.asarray(daily_vol, dtype=np.float64)
    if np.any(vol < 0) or np.any(vol > 1.0):
        raise ValueError("daily_vol 은 분수(0~1)")
    markets = np.asarray(market, dtype=object)
    if markets.ndim == 0:
        tax_bp = sell_tax_rate(trade_date, markets.item()) * 1e4
    else:
        flat = markets.ravel().tolist()
        rates = {m: sell_tax_rate(trade_date, m) * 1e4 for m in set(flat)}
        tax_bp = np.array([rates[m] for m in flat], dtype=np.float64).reshape(markets.shape)
    impact = y * vol * np.sqrt(np.abs(np.asarray(notional, dtype=np.float64)) / liq) * 1e4
    return COMMISSION_BP + np.where(sides == "SELL", tax_bp, 0.0) + hs + impact


def linear_bp(side: np.ndarray, market: np.ndarray, day: date, hs_bp: np.ndarray) -> np.ndarray:
    """수수료 + 매도세(매도만) + 반스프레드(bp) — 충격 없는 부분(폐지 청산, B ``linear_bp``)."""
    side = np.asarray(side, dtype=str)
    tax = np.array([sell_tax_rate(day, m) * 1e4 for m in market], dtype=np.float64)
    return COMMISSION_BP + np.where(side == "SELL", tax, 0.0) + np.asarray(hs_bp, dtype=np.float64)


def trade_costs(dw, adv_won, vol, market, day: date, hs_bp, aum: float = AUM) -> np.ndarray:
    """종목별 1배 비용(NAV 대비 분수) — B ``trade_costs``. 2배는 정확히 2배(네 항 모두 배수에 선형)."""
    dw = np.asarray(dw, dtype=np.float64)
    out = np.zeros(dw.shape)
    tr = dw != 0
    if tr.any():
        side = np.where(dw[tr] > 0, "BUY", "SELL")
        bp = trade_cost_bp(side, np.abs(dw[tr]) * aum, np.asarray(adv_won)[tr], np.asarray(vol)[tr],
                           np.asarray(market, dtype=object)[tr], day, half_spread_bp=np.asarray(hs_bp)[tr], y=1.0)
        out[tr] = bp / 1e4 * np.abs(dw[tr])
    return out


# ── 패널 ─────────────────────────────────────────────────────────────────────────


def ffill_cols(C: np.ndarray) -> np.ndarray:
    """종목(행)마다 시간축(열) 앞으로 채우기."""
    return pd.DataFrame(C.T).ffill().to_numpy().T


@dataclass
class Panel:
    """``code × date`` 패널(열 = 거래일, 마지막 열 = asof). 금액 단위는 이름에 적었다."""

    dates: pd.DatetimeIndex
    codes: np.ndarray
    C: np.ndarray            # 수정 종가(정규장·이음매 가드 뒤), 거래 없는 날 NaN
    tv_won: np.ndarray       # 일 거래대금(원)
    ret: np.ndarray          # 일수익(가드 뒤)
    cap: np.ndarray          # 시총(원)
    sig: np.ndarray          # 분기 순이익 YoY(as-of)
    has_sec: np.ndarray      # WICS 업종 있음(bool)
    market: np.ndarray       # "KOSPI"/"KOSDAQ"
    delist_after: np.ndarray  # 폐지 마지막 거래일 열(폐지 아님 = 큰 수)
    adv: np.ndarray = field(default=None)    # t 제외 20일 평균 거래대금(백만원)
    Cff: np.ndarray = field(default=None)
    hs_bp: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.adv is None:
            self.adv = _trailing_adv(self.tv_won / 1e6, len(self.dates), 20, self.C)
        if self.Cff is None:
            self.Cff = ffill_cols(np.asarray(self.C, dtype=np.float64))
        if self.hs_bp is None:
            self.hs_bp = np.array([HALF_SPREAD_BP.get(m, HALF_SPREAD_BP["KOSDAQ"]) for m in self.market])

    def in_u(self, t: int) -> np.ndarray:
        """U_t 재구성 = 위험모델 마스크(업종 있음 ∧ 시총 > 0 ∧ 종가 유한, core ``prep``·``build_risk_model``)."""
        cap = self.cap[:, t]
        return self.has_sec[:, t] & np.isfinite(cap) & (cap > 0) & np.isfinite(self.C[:, t])

    def truncate(self, last: int) -> "Panel":
        """열 0..last 만 남긴다(asof 자르기 — 시험·재현용). 파생 배열(adv·Cff)은 자른 입력에서 다시 계산한다."""
        s = slice(0, last + 1)
        da = np.where(self.delist_after <= last, self.delist_after, 10**9)
        return Panel(dates=self.dates[s], codes=self.codes, C=self.C[:, s], tv_won=self.tv_won[:, s], ret=self.ret[:, s],
                     cap=self.cap[:, s], sig=self.sig[:, s], has_sec=self.has_sec[:, s], market=self.market, delist_after=da)


def eligible(p: Panel, t: int) -> np.ndarray:
    """eligible(t) = 신호 유한 ∧ ADV ≥ 200억 ∧ U_t ∧ 시총 유한 안의 시총 21~100위(행 번호, 시총 내림차순·안정 정렬). B ``eligible_idx``."""
    ok = np.isfinite(p.sig[:, t]) & (p.adv[:, t] >= ADV_FLOOR) & p.in_u(t) & np.isfinite(p.cap[:, t])
    liq = np.flatnonzero(ok)
    order = liq[np.argsort(-p.cap[liq, t], kind="stable")]
    return order[CAP_RANK[0]:CAP_RANK[1]]


def u_exclusion(p: Panel, t: int) -> float:
    """①∧②(신호·ADV) 중 U_t 때문에 빠진 비율(보고만, B 와 같은 정의)."""
    base = np.isfinite(p.sig[:, t]) & (p.adv[:, t] >= ADV_FLOOR)
    n = int(base.sum())
    return float((base & ~p.in_u(t)).sum() / n) if n else float("nan")


def tranche_names(p: Panel, t: int):
    """트랜치 = eligible(t) 의 YoY 상위 10(행 순서 idx 에 argsort(−신호), core ``book`` 정렬). 적격 20 미만이면 None."""
    el = eligible(p, t)
    if el.size < MIN_NAMES:
        return None
    idx = np.sort(el)
    vals = p.sig[idx, t]
    keep = np.isfinite(vals)
    idx, vals = idx[keep], vals[keep]
    return idx[np.argsort(-vals)[:TOP_N]]


def delisted(p: Panel, f: int) -> np.ndarray:
    """f 에 폐지로 본다 = 마지막 거래일이 f 전 ∧ f 에 가격 없음(B ``delisted``)."""
    return (p.delist_after < f) & ~np.isfinite(p.C[:, f])


def r_target(p: Panel, k: int, t: int, tranches: dict, f: int):
    """R 목표 비중(체결일 f 기준, 합 1): 활성 트랜치(k, k−1, k−2)마다 1/n, 트랜치 안 동일가중. U_t 밖·폐지·가격 없는 종목은 뺀다(B ``r_target``)."""
    in_u = p.in_u(t)
    gone = delisted(p, f)
    w = np.zeros(p.C.shape[0])
    parts = []
    for j in range(k, max(-1, k - HORIZON // STEP), -1):
        b = tranches.get(j)
        if b is None:
            continue
        b = np.asarray(b, dtype=int)
        b = b[in_u[b]]
        b = b[~gone[b]]
        b = b[np.isfinite(p.Cff[b, f])]
        if b.size:
            parts.append(b)
    for b in parts:
        w[b] += 1.0 / (len(parts) * b.size)
    return w, len(parts)


def trailing_vol(ret: np.ndarray, t: int) -> np.ndarray:
    """t 포함 직전 63거래일 일수익 표준편차(ddof=1). 유한 관측 < 20 이면 NaN(→ 그 날 거래 불가)."""
    w = ret[:, max(0, t - VOL_WIN + 1): t + 1]
    n = np.isfinite(w).sum(axis=1)
    with np.errstate(invalid="ignore"):
        v = np.nanstd(w, axis=1, ddof=1) if w.shape[1] > 1 else np.full(w.shape[0], np.nan)
    v[n < VOL_MIN_OBS] = np.nan
    return v


def make_grid(dates: pd.DatetimeIndex, anchor: str, sim_from: str) -> list[int]:
    """격자 열 번호 = anchor 열 ± 20k 중 sim_from 이상. anchor·sim_from 은 패널 거래일이어야 하고 sim_from 은 격자 위여야 한다(달력 구멍 감지)."""
    a = dates.get_loc(pd.Timestamp(anchor))
    s = dates.get_loc(pd.Timestamp(sim_from))
    if (a - s) % STEP:
        raise ValueError(f"{sim_from} 은 {anchor} 격자 위가 아니다(열 차 {a - s}) — 달력이 바뀌었다")
    return list(range(s, len(dates), STEP))


# ── 회계(B ``simulate(kind="R")`` 의 실제 규칙 전부: t+1 체결·표류·U_t 강제 매도·폐지 현금·거래 불가·비용) ──────────


@dataclass
class Step:
    k: int                # 격자 번호(이 실행의 격자 안)
    t: int
    f: int
    f_next: int           # 다음 체결일 열(패널 밖이면 > 마지막 열)
    cost1: float = 0.0
    turnover: float = 0.0
    n_names: int = 0
    n_tranches: int = 0
    n_forced: int = 0
    n_delist: int = 0
    n_frozen: int = 0
    n_untradeable: int = 0
    delist_mass: float = 0.0
    cash: float = 0.0
    n_eligible: int = 0
    u_excl: float = float("nan")
    tranche: list = field(default_factory=list)          # 이 결정일에 새로 만든 트랜치(행 번호)
    weights: dict = field(default_factory=dict)          # 체결 뒤 비중 {행: w}
    contrib: list = field(default_factory=list)          # (행, 체결 뒤 비중, 종목 스텝 수익, 1배 비용) — 스텝이 끝났을 때만
    book_path: list = field(default_factory=list)        # 체결 뒤 NAV 경로(f 다음 날부터 min(f_next, 마지막 열)까지)
    bench_path: list = field(default_factory=list)
    gross: float = float("nan")                          # 스텝 비용 전 수익(끝났을 때만)
    bench: float = float("nan")

    @property
    def complete(self) -> bool:
        return math.isfinite(self.gross)

    def net(self, m: float) -> float:
        return (1.0 - m * self.cost1) * (1.0 + self.gross) - 1.0

    def active(self, m: float) -> float:
        return self.net(m) - self.bench


@dataclass
class SimOut:
    steps: list
    pending: dict | None       # 마지막 열이 결정일이면 다음 체결(t+1 종가) 계획
    w_last: np.ndarray         # 마지막 열 종가의 보유 비중(표류 뒤)
    cash_last: float
    day_gross: dict            # 열 → 그날 비용 전 수익(첫 체결 다음 날부터)
    day_cost1: dict            # 열 → 그날 종가 체결 비용(1배)
    day_bench: dict            # 열 → 그날 벤치(적격 동일가중 매수 보유) 수익


def simulate(p: Panel, grid: list[int]) -> SimOut:
    """R 북 회계. 체결 직후 NAV=1, 스텝 순수익 r_k(m) = (1 − m·cost1_k)(1 + gross_k) − 1(B 와 같은 식, 비중 경로는 비용 배수와 무관).

    마지막 열 T−1 까지만 본다 — 그 뒤 정보가 없다. 결정일 t_k = T−1 이면 체결하지 않고 ``pending`` 에 계획만 남긴다(체결일 정보는
    다음 실행이 본다). 진행 중 스텝은 경로만 있고 ``gross`` 는 NaN 이다.
    """
    C, Cff = p.C, p.Cff
    nN, nD = C.shape
    last = nD - 1
    u = np.zeros(nN)
    cash = 1.0
    tranches: dict[int, np.ndarray | None] = {}
    steps: list[Step] = []
    day_gross: dict[int, float] = {}
    day_cost1: dict[int, float] = {}
    day_bench: dict[int, float] = {}
    pending = None
    for k, t in enumerate(grid):
        if t > last:
            break
        tranches[k] = tranche_names(p, t)
        if t == last:   # 오늘이 결정일 — 체결일(t+1) 정보는 아직 없다
            want, n_tr = r_target(p, k, t, tranches, t)
            el = eligible(p, t)
            pending = {"k": k, "t": t, "target": {int(i): float(want[i]) for i in np.flatnonzero(want > 0)}, "n_tranches": n_tr,
                       "tranche": [] if tranches[k] is None else [int(i) for i in tranches[k]], "n_eligible": int(el.size),
                       "u_excl": u_exclusion(p, t)}
            break
        f = t + 1
        f_next = (grid[k + 1] if k + 1 < len(grid) else t + STEP) + 1
        day = p.dates[f].date()
        in_u = p.in_u(t)
        el = eligible(p, t)
        st = Step(k=k, t=t, f=f, f_next=f_next, n_eligible=int(el.size), u_excl=u_exclusion(p, t),
                  tranche=[] if tranches[k] is None else [int(i) for i in tranches[k]])
        # 체결일 f 의 표류 비중
        val_f = np.where(u > 0, u * Cff[:, f], 0.0)
        nav_f = val_f.sum() + cash
        w_f = val_f / nav_f
        cash_f = cash / nav_f
        held = w_f > 0
        cost_vec = np.zeros(nN)
        gone = held & delisted(p, f)       # ④ 폐지 청산(마지막 종가, 선형 매도 비용)
        if gone.any():
            g = np.flatnonzero(gone)
            cost_vec[g] += linear_bp(np.full(g.size, "SELL"), p.market[g], day, p.hs_bp[g]) / 1e4 * w_f[g]
            cash_f += w_f[g].sum()
            st.delist_mass = float(w_f[g].sum())
            w_f[g] = 0.0
            st.n_delist = int(g.size)
        held_f = w_f > 0
        tradeable = np.isfinite(C[:, f]) & (np.nan_to_num(p.tv_won[:, f]) > 0)   # ⑤ 체결일 거래 가능
        vol = trailing_vol(p.ret, t)
        adv_won = p.adv[:, t] * WON_PER_MIL
        costable = np.isfinite(vol) & np.isfinite(adv_won) & (p.adv[:, t] >= mil(1)) & (vol <= 1.0)
        tradeable &= costable
        frozen = held_f & ~tradeable
        st.n_frozen = int(frozen.sum())
        forced = held_f & ~in_u
        st.n_forced = int((forced & tradeable).sum())
        want, st.n_tranches = r_target(p, k, t, tranches, f)
        st.n_untradeable = int(((want > 0) & ~tradeable).sum())
        want = np.where(tradeable, want, 0.0)
        free = tradeable & ~frozen
        room = 1.0 - w_f[frozen].sum()
        s = want[free].sum()
        new = np.where(frozen, w_f, 0.0)
        if s > 0:
            new[free] = want[free] * (room / s)     # R: 상한 없이 비례(B capped_rescale(cap=None))
        new_cash = 1.0 - new.sum()
        dw = np.where(tradeable, new - w_f, 0.0)
        new = np.where(tradeable, new, w_f)
        if np.any(dw != 0):
            cost_vec += trade_costs(dw, np.where(costable, adv_won, 1e12), np.where(costable, vol, 0.0), p.market, day, p.hs_bp)
        st.turnover = float(np.abs(dw).sum())
        st.cost1 = float(cost_vec.sum())
        st.n_names = int((new > 0).sum())
        st.cash = float(new_cash)
        st.weights = {int(i): float(new[i]) for i in np.flatnonzero(new > 0)}
        day_cost1[f] = st.cost1
        # 보유: 체결 뒤 NAV=1 → min(f_next, last)
        px = np.where(new > 0, Cff[:, f], np.nan)
        u = np.where(new > 0, new / px, 0.0)
        cash = max(new_cash, 0.0)
        bel = el[np.isfinite(C[el, f])]
        end = min(f_next, last)
        prev_b, prev_v = 1.0, 1.0
        for d in range(f + 1, end + 1):
            v = float(np.where(u > 0, u * Cff[:, d], 0.0).sum() + cash)
            b = float(np.mean(Cff[bel, d] / C[bel, f])) if bel.size else float("nan")
            st.book_path.append(v)
            st.bench_path.append(b)
            day_gross[d] = v / prev_v - 1.0
            day_bench[d] = b / prev_b - 1.0 if bel.size else float("nan")
            prev_v, prev_b = v, b
        if f_next <= last:   # 스텝 끝남
            g_i = np.where(new > 0, Cff[:, f_next] / Cff[:, f] - 1.0, 0.0)
            st.gross = float((new * g_i).sum())
            st.bench = float(np.mean(Cff[bel, f_next] / C[bel, f] - 1.0)) if bel.size else float("nan")
            nz = np.flatnonzero((new > 0) | (cost_vec > 0))
            st.contrib = [(int(i), float(new[i]), float(g_i[i]), float(cost_vec[i])) for i in nz]
        steps.append(st)
    val = np.where(u > 0, u * Cff[:, last], 0.0)
    nav = val.sum() + cash
    return SimOut(steps=steps, pending=pending, w_last=val / nav, cash_last=float(cash / nav),
                  day_gross=day_gross, day_cost1=day_cost1, day_bench=day_bench)


# ── 판정 통계(사전등록 §3) ───────────────────────────────────────────────────────

LOOKS = (12, 24, 36)
OBF_SHAPE = (2.36, 1.67, 1.36)    # c·√(36/K), c = 1.36 — PO-1.3 모양
NW_LAG = 2
BOUNDARIES = (3.068, 2.171, 1.768)   # OBF 모양 × max(1.30, f_boot) — pead_forward_calib.json(정정 F1-1 m1, 올리기만 한다)
FUTILITY = {24: 0.0}                 # 비구속 무용 중단: t_24 < 0


def nw_t(x) -> float:
    """평균 / Newey-West(Bartlett, lag 2) HAC se — core ``newey_west_t`` 와 같은 식."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < NW_LAG + 2:
        return float("nan")
    d = x - x.mean()
    var = (d @ d) / n
    for k in range(1, NW_LAG + 1):
        var += 2.0 * (1.0 - k / (NW_LAG + 1.0)) * ((d[k:] @ d[:-k]) / n)
    se = math.sqrt(max(var, 0.0) / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def payoff_ratio(x) -> float:
    """이익 평균 / 손실 평균의 절댓값(0 은 빼고). 한쪽이 없으면 NaN."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    w, lo = x[x > 0], x[x < 0]
    return float(w.mean() / abs(lo.mean())) if w.size and lo.size else float("nan")


def holding_intervals(steps: list[dict], m: float = 2.0) -> np.ndarray:
    """보유 구간(종목별 체결 뒤 비중 > 0 이 이어진 스텝)의 **능동** 순기여 Σ(w·(g − b_k) − m·c).

    b_k = 그 스텝 벤치 수익. 한 스텝의 종목 합은 a_k(m) 와 같다(현금 0·비용 교차항 무시 — 재현 합 0.350 대 Σa_k 0.345).
    정정 F1-1 M1: B ``holding_intervals`` 의 총액 기준 Σ(w·g − m·c)(재현 합 0.515)를 의도적으로 능동 기준으로 바꿨다 — 판정 통계 a_k 와
    단위를 맞춘다. 구간 이어짐·판 스텝의 매도 비용 처리는 B 그대로. 스텝 번호는 기록의 ``k``(격자 번호)를 쓴다(정정 F1-1 m7).
    """
    cur: dict[str, float] = {}
    last: dict[str, int] = {}
    done = []
    for s in steps:
        k, b = int(s["k"]), float(s["bench"])
        seen = set()
        for code, w, g, c in s["contrib"]:
            if w > 0:
                if code in cur and last.get(code) != k - 1:
                    done.append(cur.pop(code))
                cur[code] = cur.get(code, 0.0) + w * (g - b) - m * c
                last[code] = k
                seen.add(code)
            elif code in cur:
                done.append(cur.pop(code) - m * c)
        for code in [c for c in cur if c not in seen and last[c] != k]:
            done.append(cur.pop(code))
    done += list(cur.values())
    return np.asarray(done, dtype=np.float64)


def forward_report(steps: list[dict], boundaries: "tuple[float, ...]", futility: "dict[int, float]") -> dict:
    """장부의 ``step_closed`` 목록(k 순) → 보는 시점마다 HAC t·경계·판정. 숫자와 사전등록 라벨만 낸다.

    판정(사전등록 §3): 보는 시점 K 에서 t_K ≥ b_K ∧ 상위 5 보유 구간 제거 EV(2배) > 0 → "엣지 확인". t_K < 무용 문턱 → "엣지 없음(무용 중단)".
    K=36 에서 확인이 없으면: 경계는 넘었는데 취약 → "판단 불가(취약 — 상위 5 보유 구간 의존)", t_36 ≤ 0 → "엣지 없음",
    나머지 → "판단 불가(검정력 부족)". 그 전 시점에서 어느 쪽도 아니면 "계속".
    """
    a2 = np.array([s["a_2x"] for s in steps], dtype=np.float64)
    a1 = np.array([s["a_1x"] for s in steps], dtype=np.float64)
    if not np.isfinite(a2).all() or not np.isfinite(a1).all():   # 정정 F1-1 m7: 결측을 조용히 NaN EV 로 두지 않는다
        bad = [s.get("t") for s, x in zip(steps, a2) if not math.isfinite(x)]
        return {"n_steps": int(a2.size), "verdict": "판단 불가(결측)", "missing_steps": bad, "looks": []}
    res = {"n_steps": int(a2.size), "EV_2x": float(a2.mean()) if a2.size else float("nan"),
           "EV_1x": float(a1.mean()) if a1.size else float("nan"),
           "step_payoff_2x": payoff_ratio(a2), "looks": [], "verdict": "계속"}
    if a2.size >= 2:
        res["IR_2x"] = float(a2.mean() / a2.std(ddof=1) * math.sqrt(252.0 / STEP))
    iv = holding_intervals(steps, 2.0)
    res["interval_payoff_2x"] = payoff_ratio(iv)
    for K, b in zip(LOOKS, boundaries):
        if a2.size < K:
            break
        sub = steps[:K]
        tK = nw_t(a2[:K])
        ivK = np.sort(holding_intervals(sub, 2.0))[::-1]
        # 상위 5 보유 구간 능동 순기여를 뺀 스텝당 EV(재분배 없음) = mean(a_k) − Σ 상위 5 / K (정정 F1-1 M1: 능동 기준)
        drop5 = float(a2[:K].mean() - ivK[:5].sum() / K) if ivK.size > 5 else float("nan")
        look = {"K": K, "t": tK, "boundary": b, "futility": futility.get(K), "EV_2x": float(a2[:K].mean()), "EV_drop_top5_2x": drop5}
        res["looks"].append(look)
        if tK >= b and drop5 > 0:
            res["verdict"] = "엣지 확인"
            break
        fragile = tK >= b
        if fragile:
            res["verdict_note"] = "경계는 넘었지만 상위 5 보유 구간을 빼면 EV ≤ 0 — 취약, 확인 아님"
        if K in futility and tK < futility[K]:
            res["verdict"] = "엣지 없음(무용 중단)"
            break
        if K == LOOKS[-1]:
            res["verdict"] = ("판단 불가(취약 — 상위 5 보유 구간 의존)" if fragile else
                              "엣지 없음" if tK <= 0 else "판단 불가(검정력 부족)")
    return res


# ── DB 읽기(asof 이하로 자른다 — 예외는 load_panel 독스트링) ─────────────────────────────────────────────


def _cur(con):
    cur = con.cursor()
    cur.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT_S}s'")
    return cur


def yoy_signal(con, codes: np.ndarray, days: pd.DatetimeIndex, end: str) -> tuple[np.ndarray, dict]:
    """B ``yoy_signal_strict``(PO-2.1-5) 와 같은 식: 실적 버전·정기보고서 접수일을 ≤ end 로 자르고, 2019Q1 이후 분기는 end 전 정기보고서가
    없으면 뺀다. 알게 된 날 = max(avail_date, 정기보고서 첫 접수일) 다음 거래일, 정정 버전은 ``knowledge_col`` 로 날짜별 선택."""
    from swing_it.storage import read_earnings

    ea = read_earnings(con, all_versions=True, cols=("code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior"))
    ea["avail_date"] = pd.to_datetime(ea["avail_date"])
    ea["knowledge_date"] = pd.to_datetime(ea["knowledge_date"])
    endt = pd.Timestamp(end)
    ea = ea[(ea["avail_date"] <= endt) & (ea["knowledge_date"] <= endt)].copy()
    cur = _cur(con)
    cur.execute("SELECT code, period, min(rcept_dt) FROM earnings_disclosures WHERE kind = 'periodic' AND rcept_dt <= %s "
                "GROUP BY code, period", (end,))
    per = pd.DataFrame(cur.fetchall(), columns=["code", "period", "per"])
    return yoy_from_frames(ea, per, codes, days)


def yoy_from_frames(ea: pd.DataFrame, per: pd.DataFrame, codes: np.ndarray, days: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
    """``yoy_signal`` 의 순수 부분(시험용). ``ea``·``per`` 는 이미 end 이하로 자른 것."""
    from swing_it.features.fundamentals import _yoy_vec, earnings_yoy_panel

    n_read = len(ea)
    ea = ea.assign(code=ea["code"].astype(str), avail_date=pd.to_datetime(ea["avail_date"]),
                   knowledge_date=pd.to_datetime(ea["knowledge_date"]))
    per = per.assign(code=per["code"].astype(str), per=pd.to_datetime(per["per"]))
    ea = ea.merge(per, on=["code", "period"], how="left")
    late = (ea["period"].astype(str) >= "2019Q1") & ea["per"].isna()
    n_drop = int(late.sum())
    ea = ea[~late].copy()
    known = ea[["avail_date", "per"]].max(axis=1).to_numpy(dtype="datetime64[ns]")
    i = days.searchsorted(known, side="right")
    ok = (i < len(days)) & ~np.isnat(known)
    ea = ea[ok].copy()
    ea["known_date"] = days[i[ok]].strftime("%Y-%m-%d")
    ea["knowledge_date"] = ea["knowledge_date"].dt.strftime("%Y-%m-%d")
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    cols = list(days.strftime("%Y-%m-%d"))
    panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), cols, avail_col="known_date", knowledge_col="knowledge_date",
                               tie_col="period")  # 정정 F1-2 — 같은 날 알려진 두 분기는 늦은 분기
    wide = panel.pivot(index="code", columns="date", values="yoy").reindex(index=[str(c) for c in codes], columns=cols)
    return wide.to_numpy(dtype=np.float64), {"earnings_rows_le_end": int(n_read), "dropped_2019Q1plus_no_periodic": n_drop,
                                             "rows_used": int(len(ea))}


def sector_panel(wics: pd.DataFrame, codes: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    """WICS 스냅샷 → 업종 있음(bool). 스냅샷은 그날부터 다음 스냅샷 전까지(core ``_sector_panel``)."""
    out = np.zeros((len(codes), len(days)), dtype=bool)
    if wics.empty:
        return out
    wide = wics.pivot_table(index="date", columns="code", values="sector_cd", aggfunc="first")
    snap = pd.to_datetime(wide.index)
    pos = snap.searchsorted(days, side="right") - 1
    vals = wide.reindex(columns=pd.Index([str(c) for c in codes])).notna().to_numpy()
    ok = pos >= 0
    out[:, ok] = vals[pos[ok]].T
    return out


def seam_guard(codes, dates: pd.DatetimeIndex, ret: np.ndarray, close: np.ndarray, raw: np.ndarray, events: pd.DataFrame,
               thr: float = SEAM_THR) -> tuple[np.ndarray, np.ndarray, int]:
    """이음매 가드(PO-2.1, ``pead_optimizer_seam.seam_guard`` 의 원천 하나 판): 수정 비와 원 종가 비가 thr 넘게 어긋나고 그 사이
    ``corporate_actions`` 사건이 없으면 그 수익을 NaN, 종가는 그 비를 걷어내 이어 붙인다. 반환 (ret, close, 이음매 셀 수)."""
    N, T = ret.shape
    rho_adj = 1.0 + ret
    with np.errstate(divide="ignore", invalid="ignore"):
        pair = raw[:, 1:] / raw[:, :-1]
    rho_raw = np.full((N, T), np.nan)
    okp = np.isfinite(pair) & (pair > 0)
    rho_raw[:, 1:][okp] = pair[okp]
    explained = np.zeros((N, T), dtype=bool)
    if len(events):
        ii = pd.Index([str(c) for c in codes]).get_indexer(events["code"].astype(str))
        kk = dates.searchsorted(pd.to_datetime(events["date"]).to_numpy(), side="left")
        keep = (ii >= 0) & (kk >= 1) & (kk < T)
        explained[ii[keep], kk[keep]] = True
    fin = np.isfinite(rho_adj) & (rho_adj > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(rho_adj / rho_raw - 1.0)
    seam = fin & np.isfinite(rho_raw) & (dev > thr) & ~explained
    ret_f = ret.copy()
    ret_f[seam] = np.nan
    jump = np.where(seam, np.log(np.where(seam, rho_adj, 1.0)), 0.0)
    close_f = np.asarray(close, dtype=np.float64) * np.exp(-np.cumsum(jump, axis=1))
    return ret_f, close_f, int(seam.sum())


def mask_last_day(df: pd.DataFrame, end: str, src_col: str, cols: tuple) -> tuple[pd.DataFrame, int]:
    """정정 F1-1 m2: end ≥ 2026-09-14 이면 end 날 행 중 원천이 분봉(``"session"``)이 아닌 행의 ``cols`` 를 NaN 으로. 반환 (표, 지운 행 수).
    원천 열이 없으면(그 날짜 보정이 없었다) 그대로."""
    from swing_it.krx_session import AFTERMARKET_START

    if end < AFTERMARKET_START or src_col not in df.columns:
        return df, 0
    m = (pd.to_datetime(df["date"]) == pd.Timestamp(end)).to_numpy() & (df[src_col] != "session").to_numpy()
    if m.any():
        df = df.copy()
        df.loc[m, list(cols)] = np.nan
    return df, int(m.sum())


def load_panel(con, start: str, end: str) -> tuple[Panel, dict, dict]:
    """[start, end] 패널. 반환 (패널, 진단, 이름).

    날짜가 있는 원천(가격·실적·정기보고서·주식수·WICS·사건)은 end 이하로 자른다. **예외 하나(정정 F1-1 m2):** 정규장 종가 항등식
    (``read_regular_close``)은 날짜 d 의 종가를 d+1 수급 행(end 뒤 31일까지 읽는다)으로 만든다. d < end 이면 d+1 ≤ end 라 문제없고,
    d = end 만 미래 행을 쓴다. 그래서 2026-09-14 이후 마지막 날은 분봉(정규장) 원천만 남기고 항등식·일봉 대체 값을 지운다(NaN) —
    아침 실행(그때는 d+1 행이 없다)과 뒤늦은 ``--asof`` 백필·재시뮬레이션이 같은 값을 보게 한다.
    """
    from swing_it.krx_session import read_raw_close, split_consistent_cap
    from swing_it.storage import read_prices, read_shares

    px = read_prices(con, session=PRICE["session"], unpatched=PRICE["unpatched"], cols=("code", "date", "close", "trade_value"),
                     start=start, end=end, parse_dates=True)
    info = {"price_session": px.attrs.get("krx_session", {})}
    px, info["last_day_non_session_masked"] = mask_last_day(px, end, "krx_src", ("close", "trade_value"))
    cur = _cur(con)
    cur.execute("SELECT code FROM stocks UNION SELECT code FROM delisted_stocks UNION SELECT code FROM kind_delistings")
    equities = {str(r[0]) for r in cur.fetchall()}                     # core _equity_codes
    px = px[px["code"].astype(str).isin(equities)]
    days = pd.DatetimeIndex(sorted(px["date"].unique()))
    codes = np.array(sorted(px["code"].astype(str).unique()))
    cidx = pd.Index(codes)

    def wide(df, col):
        return df.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=cidx)

    adj = wide(px.assign(code=px["code"].astype(str)), "close")
    tv = wide(px.assign(code=px["code"].astype(str)), "trade_value").to_numpy(float).T * 1e6
    C0 = adj.to_numpy(float).T
    ret0 = np.full_like(C0, np.nan)
    ret0[:, 1:] = C0[:, 1:] / C0[:, :-1] - 1.0
    rc = read_raw_close(con, start=start, end=end, aftermarket_fix=PRICE["aftermarket_fix"])
    rc = rc.assign(code=rc["code"].astype(str))
    rc, info["last_day_raw_non_session_masked"] = mask_last_day(rc, end, "raw_src", ("close",))   # 원 종가도(일봉 대체는 애프터마켓 오염)
    rcw = wide(rc, "close")
    src = rc.pivot_table(index="date", columns="code", values="raw_src", aggfunc="last").reindex(index=days, columns=cidx)
    cur.execute("SELECT code, date FROM corporate_actions WHERE date BETWEEN %s AND %s", (start, end))
    ev = pd.DataFrame(cur.fetchall(), columns=["code", "date"])
    ret, C, n_seam = seam_guard(codes, days, ret0, C0, rcw.to_numpy(float).T, ev)
    info["seam_cells"] = n_seam
    sh = read_shares(con, with_asof=True, parse_dates=True)
    sh = sh[pd.to_datetime(sh["asof"]) <= pd.Timestamp(end)]
    sh = sh[(sh["shares_outstanding"] > 0) & (sh["shares_outstanding"] <= 1e10)]
    sh = sh.assign(code=sh["code"].astype(str))
    capw, n_bad = split_consistent_cap(rcw, adj, sh, asof="asof", raw_src=src)
    cap = capw.to_numpy(float).T
    cap = np.where(cap <= CAP_MAX, cap, np.nan)
    info["cap_check_violations"] = int(n_bad)
    last_src = src.iloc[-1].value_counts().to_dict() if len(src) else {}
    info["raw_src_last_day"] = {str(k): int(v) for k, v in last_src.items()}
    sig, info["signal"] = yoy_signal(con, codes, days, end)
    lo = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    cur.execute("SELECT date, code, sector_cd FROM wics_membership WHERE date BETWEEN %s AND %s AND sector_cd IS NOT NULL", (lo, end))
    wics = pd.DataFrame(cur.fetchall(), columns=["date", "code", "sector_cd"])
    wics["code"] = wics["code"].astype(str)
    info["wics_last_snapshot"] = str(max(wics["date"])) if len(wics) else None
    if len(wics):                                    # m3: 스냅샷이 21일 넘게 묵으면 표시(신규 상장이 U_t 에 못 들어온다)
        info["wics_stale_days"] = int((pd.Timestamp(end) - pd.Timestamp(max(wics["date"]))).days)
        info["wics_stale"] = info["wics_stale_days"] > WICS_STALE_DAYS
    has_sec = sector_panel(wics, codes, days)
    cur.execute("SELECT code, market, name FROM stocks")
    cur_m = {str(c): (m, n) for c, m, n in cur.fetchall()}
    cur.execute("SELECT code, market, last_trade_date, name FROM delisted_stocks")
    rows = cur.fetchall()
    del_m = {str(c): (m, n) for c, m, _, n in rows}
    last = {str(c): d for c, _, d, _ in rows if d}
    from krx_quant_core.market.codes import Market
    mk, names = [], {}
    for c in codes:
        m, nm = cur_m.get(c, del_m.get(c, (None, "")))
        names[c] = nm or ""
        try:
            v = Market.parse(m).value if m is not None else None
        except ValueError:
            v = None
        mk.append("KOSDAQ" if v in (None, "KONEX") else v)   # 모름·코넥스 = KOSDAQ 비용(B)
    delist_after = np.full(len(codes), 10**9, dtype=np.int64)
    for i, c in enumerate(codes):
        d = last.get(c)
        if d is None:
            continue
        try:
            j = int(days.searchsorted(pd.Timestamp(str(d)[:10]), side="right")) - 1
        except ValueError:
            continue
        if j < len(days) - 1:   # 창 끝 이후 폐지는 창 안에서 폐지가 아니다(B)
            delist_after[i] = j
    p = Panel(dates=days, codes=codes, C=C, tv_won=tv, ret=ret, cap=cap, sig=sig, has_sec=has_sec,
              market=np.array(mk, dtype=object), delist_after=delist_after)
    return p, info, names


# ── 장부 행 ─────────────────────────────────────────────────────────────────────


def plan_mismatch(plan: dict, rec: dict) -> dict:
    """정정 F1-1 m3: 결정일 행의 계획(``decision``) 대 다시 시뮬레이션한 스텝 기록 — 새 트랜치 차집합 수·비중 L1(보고만).

    비중 차에는 체결일 거래 불가·얼림 조정이 정상으로 섞인다. 트랜치 차가 0 이 아니면 결정일 데이터가 뒤에 바뀐 것이다."""
    pt = {x["code"] for x in plan.get("tranche_new", [])}
    rt = set(rec["tranche"])
    pw = plan.get("target", {})
    rw = {c: w for c, w, _, _ in rec["contrib"] if w > 0}
    keys = set(pw) | set(rw)
    return {"tranche_symdiff": len(pt ^ rt), "weight_l1": float(sum(abs(rw.get(c, 0.0) - pw.get(c, 0.0)) for c in keys))}


def step_record(p: Panel, st: Step, plan: "dict | None" = None) -> dict:
    """끝난 스텝 → 장부 ``step_closed`` 블록. 판정 통계 = ``a_2x``. ``k`` = 이 실행 격자 번호(SIM_FROM 기준이라 실행마다 같다)."""
    D = p.dates
    rec = {"k": int(st.k), "t": str(D[st.t].date()), "f": str(D[st.f].date()), "f_next": str(D[st.f_next].date()),
            "gross": st.gross, "cost1": st.cost1, "net_1x": st.net(1.0), "net_2x": st.net(2.0), "bench": st.bench,
            "a_1x": st.active(1.0), "a_2x": st.active(2.0), "turnover": st.turnover, "n_names": st.n_names, "n_tranches": st.n_tranches,
            "n_forced": st.n_forced, "n_delist": st.n_delist, "n_frozen": st.n_frozen, "n_untradeable": st.n_untradeable,
            "n_eligible": st.n_eligible, "u_excl": st.u_excl, "cash": st.cash,
            "tranche": [str(p.codes[i]) for i in st.tranche],
            "contrib": [(str(p.codes[i]), w, r, c) for i, w, r, c in st.contrib]}
    if plan is not None:
        rec["plan_mismatch"] = plan_mismatch(plan, rec)
    return rec


def build_row(p: Panel, out: SimOut, names: dict, info: dict, *, arm: str = ARM, live: "dict | None" = None,
              bench_idx: "dict | None" = None, arm_start: str = ARM_START, backfilled: bool = False,
              written: "set[str] | None" = None, plans: "dict | None" = None) -> dict:
    """마지막 열(asof)의 장부 행. 그날 수익·포지션·결정(오늘이 결정일이면)·전날 끝난 스텝(있으면).

    ``written`` = 장부에 이미 있는 스텝(결정일 t). 그 밖의 끝난 전진 스텝이 장부에 없으면(그 다음 날 행이 빠졌다) ``step_closed_catchup`` 에
    지금 데이터로 계산해 넣는다 — 판정은 스텝마다 장부에 처음 적힌 값(:func:`ledger_steps`). ``written`` 에 있는 스텝은 ``step_closed`` 로도
    다시 내지 않는다(정정 F1-1 M2 — 뒤늦은 ``--asof`` 백필이 덮어쓰지 못하게). ``plans`` = 결정일 → 그날 행의 ``decision``(m3 대조)."""
    T = len(p.dates) - 1
    D = p.dates
    g = out.day_gross.get(T, float("nan"))
    c1 = out.day_cost1.get(T, 0.0)
    b = out.day_bench.get(T, float("nan"))
    cur = [s for s in out.steps if s.f <= T]
    st = cur[-1] if cur else None
    phase = "decision" if out.pending is not None else ("fill" if st is not None and st.f == T else "hold")
    row = {"date": str(D[T].date()), "arm": arm, "rules": RULES, "git": _git_sha(), "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "phase": phase, "ret_today_gross": g, "cost1_today": c1,
           "ret_today_1x": (1.0 + g) * (1.0 - c1) - 1.0 if math.isfinite(g) else -c1,
           "ret_today_2x": (1.0 + g) * (1.0 - 2.0 * c1) - 1.0 if math.isfinite(g) else -2.0 * c1,
           "bench_today": b, "bench_ret_today": bench_idx or {},
           "step": None if st is None else {"t": str(D[st.t].date()), "f": str(D[st.f].date()),
                                            "f_next": str(D[st.f_next].date()) if st.f_next <= T else None},
           "positions": {str(p.codes[i]): {"name": names.get(str(p.codes[i]), ""), "w": round(float(out.w_last[i]), 6),
                                           "last": float(p.Cff[i, T])} for i in np.flatnonzero(out.w_last > 0)},
           "cash_w": out.cash_last, "price": info}
    if out.pending is not None:
        pd_ = out.pending
        row["decision"] = {"t": str(D[pd_["t"]].date()), "fill": "다음 거래일 종가", "n_eligible": pd_["n_eligible"], "u_excl": pd_["u_excl"],
                           "n_tranches": pd_["n_tranches"],
                           "tranche_new": [{"code": str(p.codes[i]), "name": names.get(str(p.codes[i]), ""), "yoy": float(p.sig[i, T])}
                                           for i in pd_["tranche"]],
                           "target": {str(p.codes[i]): round(w, 6) for i, w in sorted(pd_["target"].items(), key=lambda x: -x[1])}}
        if live is not None:
            row["decision"]["live_recommend"] = live
    written = written or set()
    plans = plans or {}
    fwd = [s for s in out.steps if s.complete and s.f_next < T and str(D[s.t].date()) >= arm_start
           and str(D[s.t].date()) not in written]
    closed = [s for s in fwd if s.f_next == T - 1]
    if closed:
        row["step_closed"] = step_record(p, closed[-1], plans.get(str(D[closed[-1].t].date())))
    late = [step_record(p, s, plans.get(str(D[s.t].date()))) for s in fwd if s.f_next < T - 1]
    if late:
        row["step_closed_catchup"] = late
    if backfilled:
        row["backfilled"] = True
    return row


def append_row(path: str, row: dict, arm_start: str = ARM_START) -> str:
    """덧붙이기 전용. (date, arm) 이 있으면 쓰지 않고, arm_start 전 날짜도 쓰지 않는다. 반환 = 상태 문자열."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for x in fh:
                if x.strip():
                    r = json.loads(x)
                    done.add((r["date"], r["arm"]))
    if (row["date"], row["arm"]) in done:
        return f"이미 기록됨: {row['date']} {row['arm']} — 덧붙이지 않는다"
    if row["date"] < arm_start:
        return f"시작 전: {row['date']} < {arm_start} {row['arm']} — 사전등록 첫 기록 거래일 전이라 쓰지 않는다"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    return f"기록: {row['date']} {row['arm']}"


def read_ledger(path: str, arm: str = ARM) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [r for r in (json.loads(x) for x in fh if x.strip()) if r.get("arm") == arm]


def ledger_steps(rows: list[dict]) -> list[dict]:
    """장부 → 판정용 스텝 목록(결정일 t 순). 같은 스텝이 여러 번 나오면 **파일에 먼저 덧붙인 값**만 쓴다(사전등록 §3, 정정 F1-1 M2 —
    행 날짜 순이 아니다: 늦게 채운 앞 날짜 백필 행이 이미 적힌 값을 이기면 안 된다). ``rows`` 는 :func:`read_ledger` 의 파일 순서 그대로."""
    seen: dict[str, dict] = {}
    for r in rows:
        for s in ([r["step_closed"]] if "step_closed" in r else []) + r.get("step_closed_catchup", []):
            seen.setdefault(s["t"], s)
    return [seen[t] for t in sorted(seen)]


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def live_recommend(p: Panel, names: dict, con, T: int, tranche: list) -> dict:
    """현행 ``strategies.pead.recommend_holdings`` 를 같은 날 같은 가격·신호로 부른 결과와 새 트랜치의 겹침(보고만)."""
    from swing_it.storage import read_shares
    from swing_it.strategies.pead import recommend_holdings

    days = p.dates[: T + 1]
    long = pd.DataFrame({"code": np.repeat(p.codes, T + 1), "date": np.tile(days.strftime("%Y-%m-%d"), len(p.codes)),
                         "close": p.C[:, : T + 1].ravel(), "trade_value": (p.tv_won[:, : T + 1] / 1e6).ravel()}).dropna()
    ep = pd.DataFrame({"code": np.repeat(p.codes, 1), "date": days[-1].strftime("%Y-%m-%d"), "yoy": p.sig[:, T]}).dropna()
    sh = read_shares(con, with_asof=True, parse_dates=True)
    sh = sh[pd.to_datetime(sh["asof"]) <= days[-1]][["code", "date", "shares_outstanding"]]
    rec = recommend_holdings(long, ep, sh, asof=days[-1].strftime("%Y-%m-%d"))
    live = [str(c) for c in rec["code"]]
    mine = [str(p.codes[i]) for i in tranche]
    return {"codes": live, "overlap_with_tranche_new": len(set(live) & set(mine)),
            "note": "recommend_holdings: U_t 없음·최신 주식수(as-of 아님) × 수정 종가 시총·ADV 단순 평균 — 보고만"}


def live_recommend_safe(p: Panel, names: dict, con, T: int, tranche: list) -> dict:
    """정정 F1-1 m5: 보고용 대조가 실패해도 결정일 행은 쓴다 — 오류는 행에 문자열로."""
    try:
        return live_recommend(p, names, con, T, tranche)
    except Exception as e:  # noqa: BLE001 — 보고만 하는 항목
        return {"error": f"{type(e).__name__}: {e}"[:300]}


# ── 실행 ───────────────────────────────────────────────────────────────────────


def index_returns(con, day: pd.Timestamp) -> dict:
    from swing_it.storage import read_index

    ix = read_index(con, ["001", "101"], cols=("code", "date", "close"), start=str((day - pd.Timedelta(days=14)).date()),
                    end=str(day.date()), with_level=False, order_by_date=True, parse_dates=True)
    out = {}
    for code, nm in (("001", "kospi"), ("101", "kosdaq")):
        z = ix[ix["code"] == code].set_index("date")["close"]
        if len(z) >= 2 and z.index[-1] == day:
            out[nm] = float(z.iloc[-1] / z.iloc[-2] - 1)
    return out


def last_price_day(con) -> str:
    """가격 표(daily_bars_adjusted)의 마지막 거래일 — 기본 asof. 다른 원천(실적·주식수)도 이 날로 자른다(백필과 같은 결과)."""
    from swing_it.storage import read_prices

    lo = str((pd.Timestamp.now() - pd.Timedelta(days=30)).date())
    d = read_prices(con, session="raw", cols=("code", "date"), start=lo, require_delisted=False, parse_dates=True)["date"].max()
    return str(pd.Timestamp(d).date())


def check_calendar(dates: pd.DatetimeIndex, idx_dates, lo: str) -> dict:
    """정정 F1-1 m4: 가격 패널 거래일[lo, 끝] 이 KOSPI(001) 지수 거래일과 같은가. 지수의 마지막 날까지만 견준다(지수 적재가 하루 늦을 수 있다).
    다르면 격자 위상이 B 와 어긋난다 — ``ValueError``."""
    idx = pd.DatetimeIndex(sorted(set(pd.to_datetime(idx_dates))))
    idx = idx[idx >= pd.Timestamp(lo)]
    if not len(idx):
        raise ValueError("지수 001 거래일이 없다 — 달력을 확인할 수 없다")
    pan = dates[(dates >= pd.Timestamp(lo)) & (dates <= idx[-1])]
    miss_p = sorted(set(idx) - set(pan))
    miss_i = sorted(set(pan) - set(idx))
    if miss_p or miss_i:
        raise ValueError(f"달력 불일치(격자 위상) — 가격에 없는 지수 거래일 {[str(d.date()) for d in miss_p][:5]}, "
                         f"지수에 없는 가격 거래일 {[str(d.date()) for d in miss_i][:5]}")
    return {"checked_to": str(idx[-1].date()), "n_days": int(len(pan))}


def run_day(con, asof: "str | None", ledger: str, backfilled: bool) -> dict:
    end = asof or last_price_day(con)
    p, info, names = load_panel(con, PANEL_START, end)
    if str(p.dates[-1].date()) != end:
        raise SystemExit(f"asof {end} 는 가격 패널의 마지막 거래일이 아니다({p.dates[-1].date()}) — DB 거래일인지 확인")
    from swing_it.storage import read_index

    ix = read_index(con, "001", cols=("date", "close"), start=SIM_FROM, end=end, with_level=False, order_by_date=True, parse_dates=True)
    try:
        info["calendar"] = check_calendar(p.dates, ix["date"], SIM_FROM)
    except ValueError as e:
        raise SystemExit(f"멈춤: {e}") from e
    grid = make_grid(p.dates, GRID_ANCHOR, SIM_FROM)
    out = simulate(p, grid)
    T = len(p.dates) - 1
    live = live_recommend_safe(p, names, con, T, out.pending["tranche"]) if out.pending is not None else None
    rows = read_ledger(ledger)
    written = {s["t"] for s in ledger_steps(rows)}
    plans: dict = {}
    for r in rows:                                   # 결정일 계획 — 파일에 먼저 적힌 것
        if "decision" in r:
            plans.setdefault(r["decision"]["t"], r["decision"])
    row = build_row(p, out, names, info, live=live, bench_idx=index_returns(con, p.dates[-1]), backfilled=backfilled, written=written,
                    plans=plans)
    print(append_row(ledger, row))
    print(json.dumps(row, ensure_ascii=False, indent=1, default=_json_default))
    return row


def replicate(con, span: str, out_path: str) -> dict:
    """과거 재현(장부에 쓰지 않음): B 와 같은 격자(2019-08-01 첫 거래일 + 20k)로 R 을 돌려 2020~23 OOS 스텝 a_k 를 낸다.

    설계 확인(구현이 B 의 R 을 다시 내는가)과 순차 경계 보정(f_boot)에만 쓴다. B 숫자: IR(2배) 0.550, EV(2배) 0.00681, 48 스텝.
    """
    lo, hi = span.split(":")
    p, info, _ = load_panel(con, "2019-01-02", hi)
    t0 = int(p.dates.searchsorted(pd.Timestamp(lo)))
    grid = list(range(t0, len(p.dates), STEP))
    out = simulate(p, grid)
    D = p.dates
    recs = [step_record(p, s) for s in out.steps if s.complete]
    oos = [r for r in recs if "2020-01-01" <= r["t"] <= "2023-12-31" and r["f_next"] <= hi]
    a2 = np.array([r["a_2x"] for r in oos])
    a1 = np.array([r["a_1x"] for r in oos])
    res = {"span": span, "grid": [str(D[grid[0]].date()), str(D[grid[-1]].date()), len(grid)], "oos_steps": len(oos),
           "IR_2x": float(a2.mean() / a2.std(ddof=1) * math.sqrt(252 / STEP)), "EV_2x": float(a2.mean()),
           "IR_1x": float(a1.mean() / a1.std(ddof=1) * math.sqrt(252 / STEP)), "EV_1x": float(a1.mean()),
           "nw_t_2x": nw_t(a2), "step_payoff_2x": payoff_ratio(a2), "interval_payoff_2x": payoff_ratio(holding_intervals(oos, 2.0)),
           "cost1_mean": float(np.mean([r["cost1"] for r in oos])), "turnover_mean": float(np.mean([r["turnover"] for r in oos])),
           "u_excl_mean": float(np.mean([r["u_excl"] for r in oos])), "names_mean": float(np.mean([r["n_names"] for r in oos])),
           "IR_2x_by_year": {y: float(np.mean(v) / np.std(v, ddof=1) * math.sqrt(252 / STEP))
                             for y, v in pd.Series(a2).groupby([r["t"][:4] for r in oos]).agg(list).items()},
           "fragility_2x": {str(K): {"EV": float(a2[:K].mean()),
                                     "EV_drop_top5": float(a2[:K].mean() - np.sort(holding_intervals(oos[:K], 2.0))[::-1][:5].sum() / K)}
                            for K in (12, 24, 36, len(oos))},
           "interval_active_sum_2x": float(holding_intervals(oos, 2.0).sum()), "a_2x_sum": float(a2.sum()),
           "a_2x": a2.tolist(), "steps_t": [r["t"] for r in oos], "info": {k: v for k, v in info.items() if k != "price_session"},
           "B_reference": {"IR_2x": 0.549976582984937, "EV_2x": 0.006805529413322349, "IR_1x": 0.6729675242521727,
                           "EV_1x": 0.008355563539048183, "oos_steps": 48, "cost1_mean": 0.001526177066929591,
                           "turnover_mean": 0.5327701697208388, "u_excl_mean": 0.1379188501868525, "names_mean": 18.958333333333332}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=_json_default)
    print(json.dumps({k: v for k, v in res.items() if k not in ("a_2x", "steps_t")}, ensure_ascii=False, indent=1, default=_json_default))
    return res


def main() -> None:
    import psycopg2

    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--asof", default="", help="빠진 날 채우기: 쉼표로 이은 거래일 — 그날까지의 데이터만으로 다시 만들고 행에 backfilled=true")
    ap.add_argument("--replicate", default="", help="과거 재현 구간 lo:hi(장부에 쓰지 않는다)")
    ap.add_argument("--out", default="", help="--replicate 결과 JSON 경로")
    a = ap.parse_args()
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    _cur(con)                                   # 세션 statement_timeout(정문 pandas 조회에도 걸린다)
    if a.replicate:
        if not a.out:
            raise SystemExit("--replicate 는 --out 이 필요하다")
        replicate(con, a.replicate, a.out)
        return
    for end in [x for x in a.asof.split(",") if x] or [None]:
        run_day(con, end, a.ledger, backfilled=end is not None)


if __name__ == "__main__":
    main()
