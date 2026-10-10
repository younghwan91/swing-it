#!/usr/bin/env python
"""메타 세대 6 — 박스권 한국 시장의 리밸런싱 규칙(표류·달력·밴드 + 현행 R1+C10, × 목표 비중 3). **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/META_GEN6.md`` (f5e71ba + 정정 c2924d8 — 둘 다 결과 보기 전). 이 러너는 숫자를 ``META_GEN6_RESULTS.md`` 에
적고 미리 적은 채택 조건 1~5 와의 **산술 비교**만 한다. 판정은 사람이 한다. 규칙·격자·문턱은 설계 그대로(이 세대에서 고르는 것 없음).

**자산:** 1997-12 ~ 2016-12 = 코스피 가격 지수(``research/data/krx_index_daily_yahoo.csv`` 의 ``KOSPI`` — 세대 5 와 같은 파일),
2017-01 ~ 2026-09 = 코스피200 가격 지수(``research/data/krx_index_daily_naver.csv`` 의 ``KPI200`` — 세대 4 와 같은 파일).
주식(지수 ETF 대용) + 현금 두 자산, 목표 주식 비중 w ∈ {0.4, 0.6, 0.8}.

**현금 두 세계:** (i) 0%(보수 — Sharpe = 원 Sharpe), (ii) CD91 기억값(세대 5 러너 ``CD91_MEM`` 그대로 import — **기억값, DB 밖**; Sharpe = 초과).
조건 (4) 는 (ii) 세계(정정).

**체결(정정 '세부 고정'):** 모든 결정 = d 종가, 매매 = d+1 시가 — 달력 규칙도 기간 마지막 거래일 종가에 판단 → 기간 첫 거래일 시가.
날 t: 밤 수익(주식 조각 × 시가_t/종가_{t−1}, 현금 조각 × 날 금리) → 시가에 리밸런스면 목표로(매매 금액 × 편도 0.07% 를 NAV 에서)
→ 장중 수익(주식 조각 × (종가_t/시가_t − 0.05%/252)). **각 창은 시작일에 목표 비중으로 새로 시작**(창 첫날 직전 종가에 w 보유, 진입 비용 없음 —
분리 창도 따로 새로 시작). 비중은 그 사이 **표류**한다(:func:`simulate`).

**규칙:** R0 표류 · R1 매년 첫 거래일 · **R1+C10 = R1 + 종가 비중 > w + 10%p 면 다음 시가에 w(상한만 — 매매법 현행, 정정)** · R2 분기 ·
R3 월 · B5·B10 = 종가 비중이 w ± 5·10%p 밖 · B20 = 종가 비중 > 1.2w 또는 < 0.8w → 다음 시가에 w(:func:`band_limits`).

**지표:** CAGR(252/T)·MDD·Calmar(일별 경로)·Sharpe(월 복리 초과 수익 × √12 — 세대 4 ``excess_sharpe``)·리밸런스/년·평균·최대 주식 비중·회전율/년.
짝지은 월 블록(3개월, 참고 12개월) 부트스트랩 2,000벌(세대 4 ``MonthBoot``·``paired_boot``). 무작위 리밸런스 날짜 널 1,000벌(:func:`random_reb_dates`)
+ 정정 기술 줄: 비중 맞춘 R0 비교 · 연 1회 기준일 240개 분포 · 날 단위로 섞은 수익열에서의 널 기저 통과율 · 칸 상관을 감안한 순열(최대 백분위) 기준선.

Run:  uv run python research/experiments/meta_gen6_run.py [--workers 8]
      (DB 접속 없음 — CSV 만 읽는다.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import ga_run as g1  # noqa: E402
import meta_gen1_run as mg1  # noqa: E402
import meta_gen2_run as mg2  # noqa: E402
import meta_gen4_run as m4  # noqa: E402
import meta_gen5_run as m5  # noqa: E402

from research.signals import meta_fast as mf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
SUB = "meta_gen6"
RESULTS_MD = "META_GEN6_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/META_GEN6.md (f5e71ba + 정정 c2924d8)"
KOSPI_CSV, KOSPI_SYM = m5.INDEX_CSV, m5.CORE_SYM          # Yahoo ^KS11
K200_CSV, K200_SYM = m4.INDEX_CSV, m4.CORE_SYM            # 네이버 KPI200

# --- 설계 수치(META_GEN6.md + 정정 — 탐색하지 않는다) ---
COST_1W, FEE_YR, TAX = m4.COST_1W, m4.FEE_YR, m4.TAX     # 편도 0.07% · 보수 연 0.05% · 현금 이자 15.4%(기술 줄)
CD91_MEM = m5.CD91_MEM                                   # 기억값(DB 밖) — 세대 5 러너 값 그대로
W_GRID = (0.4, 0.6, 0.8)
CUR = "R1+C10"                                           # 매매법 현행(정정) — 정식 비교 기준
RULES = ("R0", "R1", CUR, "R2", "R3", "B5", "B10", "B20")
CALENDAR = {"R1": "Y", CUR: "Y", "R2": "Q", "R3": "M"}
BANDS = {CUR: ("up", 0.10), "B5": ("abs", 0.05), "B10": ("abs", 0.10), "B20": ("rel", 0.20)}
#: 채택 산술 두 벌: 정식(정정) = 현행 R1+C10 대비, 참고(원 설계 글자) = R1 대비. 각 벌의 후보 = 기준(R0·그 벌의 기준) 뺀 나머지.
TRACKS = {"cur": CUR, "r1": "R1"}
TRACK_LAB = {"cur": f"정식(정정) — 기준 = 현행 {CUR}", "r1": "참고(원 설계 글자) — 기준 = R1"}
OOS = ("1997-12-01", "2016-12-31")
H1 = ("1997-12-01", "2006-12-31")
H2 = ("2007-01-01", "2016-12-31")
G4 = ("2017-01-01", "2026-09-30")
PRECRASH = m4.PRECRASH_END                               # 2026-06-19 — 2026-06-22 급락 전 마지막 날(최대 표류 기술 줄)
P_BOOT = 0.8                                             # 조건 1 문턱
G4_SLACK = 0.02                                          # 조건 3: Sharpe ≥ 기준 − 0.02
NULL_Q = 95                                              # 조건 4: 널 Sharpe 95 백분위
N_CELLS_DESIGN = 21                                      # 원 설계 칸 수(7 × 3)
N_CELLS = len(RULES) * len(W_GRID)                       # 정정 뒤 격자 24 칸 — Bonferroni 는 이것으로(21 문턱도 함께 적음)
N_RAND, N_BOOT = m4.N_RAND, m4.N_BOOT                    # 1,000 · 2,000
BLOCK_M, BLOCK_ALT = m4.BLOCK_M, m4.BLOCK_ALT            # 3 · 12 개월
N_ANCHOR = 240                                           # 연 1회 기준일 = 그해 k 번째 거래일, k = 0..239
N_SHUF, N_SHUF_NULL, SHUF_CHUNK = 100, 200, 5            # 섞은 수익열 기저율: 섞기 100벌 × 널 200벌
SEED = 20261010
CASHES = {"zero": "현금 (i) 0% — 원 Sharpe", "cd91": "현금 (ii) CD91 기억값 — 초과 Sharpe"}
C4_WORLD = "cd91"                                        # 조건 4 세계(정정)
WINS = {"oos": (*OOS, "표본 밖 1997-12 ~ 2016-12 (코스피)", "kospi"), "h1": (*H1, "분리 1997-12 ~ 2006-12 (새로 시작)", "kospi"),
        "h2": (*H2, "분리 2007-01 ~ 2016-12 (새로 시작)", "kospi"), "g4": (*G4, "2017-01 ~ 2026-09 (코스피200)", "k200")}
#: 색 = 참조 팔레트 순서(규칙 고정 — 순위가 아니라 규칙을 따른다).
COLORS = {"R0": "#2a78d6", "R1": "#52514e", CUR: "#008300", "R2": "#eb6834", "R3": "#eda100", "B5": "#1baf7a", "B10": "#e87ba4",
          "B20": "#4a3aa7"}


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


_pct, _f = g1._pct, g1._f
_ck = m4._ck


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_meta_gen6_helpers.py)
# ---------------------------------------------------------------------------

def calendar_flags(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """달력 리밸런스 체결 날(그날 시가 — 판단은 전 거래일 종가): 연(``Y``)·분기(``Q``)·월(``M``) 이 바뀐 첫 거래일. 행 0(창 첫날)은 거짓."""
    d = pd.DatetimeIndex(dates)
    key = np.asarray({"Y": d.year, "Q": d.year * 4 + (d.month - 1) // 3, "M": d.year * 12 + d.month}[freq])
    return np.r_[False, key[1:] != key[:-1]]


def year_ordinal(dates: pd.DatetimeIndex) -> np.ndarray:
    """그해 몇 번째 거래일인가(0 = 첫 거래일). ``dates`` 는 연초부터 이어진 전체 일봉이어야 한다."""
    y = np.asarray(pd.DatetimeIndex(dates).year)
    return pd.Series(y).groupby(y).cumcount().to_numpy()


def anchor_flags(ordinal: np.ndarray, k: "int | np.ndarray") -> np.ndarray:
    """연 1회 기준일 k(그해 k 번째 거래일 시가 체결): (T,) 또는 k 배열이면 (len(k), T). 행 0(창 첫날)은 거짓 — R1 = k 0."""
    ks = np.atleast_1d(np.asarray(k))
    out = np.asarray(ordinal)[None, :] == ks[:, None]
    out[:, 0] = False
    return out[0] if np.ndim(k) == 0 else out


def band_limits(w: float, kind: str, width: float) -> "tuple[float, float]":
    """밴드 (아래, 위): ``abs`` = w ± width(%p), ``rel`` = w × (1 ∓ width), ``up`` = (−∞, w + width) 상한만(C10).
    종가 비중이 이 범위 **밖**(엄격)이면 다음날 시가 리밸런스."""
    if kind == "abs":
        return w - width, w + width
    if kind == "rel":
        return w * (1.0 - width), w * (1.0 + width)
    if kind == "up":
        return -np.inf, w + width
    raise ValueError(kind)


def simulate(ov: np.ndarray, iday: np.ndarray, rate: np.ndarray, w: float, reb: "np.ndarray | None" = None,
             band: "tuple[float, float] | None" = None, *, cost: float = COST_1W, fee: float = FEE_YR) -> dict:
    """주식 + 현금 표류·리밸런스 날별 시뮬레이터. 창 첫날 직전 종가에 NAV 1 을 비중 w 로 보유(진입 비용 없음 — 정정 '목표 비중으로 새로 시작').
    ``reb`` = (T,) 또는 (n, T) 시가 리밸런스 예정 날, ``band`` = (아래, 위) 종가 비중 경계 → 밖이면 다음날 시가 리밸런스(둘 다 주면 둘 중 하나).
    ``ov``·``iday``·``rate`` 는 (T,) 또는 (n, T)(섞은 수익열 등 경로마다 다른 시장).

    날 t: E ← E·(1 + ov_t), K ← K·(1 + rate_t) → (리밸런스면) 매매 = |w·N − E|, N ← N − cost·매매, E = w·N, K = (1 − w)·N
    → E ← E·(1 + iday_t − fee/252). 반환: ``r`` 날 수익 · ``wc`` 종가 비중 · ``reb`` 실행한 리밸런스 · ``turn`` 매매 금액 / 매매 전 NAV."""
    arrs = [np.atleast_2d(np.asarray(a, float)) for a in (ov, iday, rate)]
    sched = None if reb is None else np.atleast_2d(np.asarray(reb, bool))
    T = arrs[0].shape[1]
    n = max([a.shape[0] for a in arrs] + ([sched.shape[0]] if sched is not None else []))
    one = all(np.ndim(a) == 1 for a in (ov, iday, rate)) and (reb is None or np.ndim(reb) == 1)
    OV, ID, RT = (np.broadcast_to(a, (n, T)) for a in arrs)
    SC = np.broadcast_to(sched, (n, T)) if sched is not None else None
    E, K, nav = np.full(n, float(w)), np.full(n, 1.0 - w), np.ones(n)
    R, WC, TR = np.empty((n, T)), np.empty((n, T)), np.zeros((n, T))
    DO = np.zeros((n, T), bool)
    pend = np.zeros(n, bool)
    for t in range(T):
        E = E * (1.0 + OV[:, t])
        K = K * (1.0 + RT[:, t])
        N = E + K
        do = pend if SC is None else (SC[:, t] | pend)
        if do.any():
            tr = np.where(do, np.abs(w * N - E), 0.0)
            N2 = N - cost * tr
            E = np.where(do, w * N2, E)
            K = np.where(do, (1.0 - w) * N2, K)
            DO[:, t] = do
            TR[:, t] = tr / N
        E = E * (1.0 + ID[:, t] - fee / 252.0)
        v = E + K
        R[:, t] = v / nav - 1.0
        nav = v
        WC[:, t] = E / v
        if band is not None:
            pend = (WC[:, t] < band[0]) | (WC[:, t] > band[1])
    out = {"r": R, "wc": WC, "reb": DO, "turn": TR}
    return {k: v[0] for k, v in out.items()} if one else out


def rule_args(rule: str, w: float, dates: pd.DatetimeIndex) -> dict:
    """규칙 코드 → :func:`simulate` 의 ``reb``·``band``."""
    if rule not in RULES:
        raise ValueError(rule)
    return {"reb": calendar_flags(dates, CALENDAR[rule]) if rule in CALENDAR else None,
            "band": band_limits(w, *BANDS[rule]) if rule in BANDS else None}


def rule_sim(rule: str, w: float, dates: pd.DatetimeIndex, ov, iday, rate) -> dict:
    return simulate(ov, iday, rate, w, **rule_args(rule, w, dates))


def random_reb_dates(T: int, k: "int | np.ndarray", n: int, rng: "np.random.Generator | None" = None,
                     keys: "np.ndarray | None" = None) -> np.ndarray:
    """무작위 리밸런스 날짜 널: 행마다 정확히 ``k`` 개 날(행 1..T−1 에서 비복원 균등 — 행 0 = 창 첫날)을 고른 (n, T) 불리언.
    ``k`` 는 정수 또는 행마다 (n,). ``keys`` (n, T−1) 를 주면 그 순위로 고른다 — 같은 keys 를 여러 칸에 쓰면 칸끼리 날짜가 겹친다
    (k 작은 칸의 날 ⊂ k 큰 칸의 날 — 칸 상관을 살린 순열 기준선)."""
    ks = np.broadcast_to(np.asarray(k, np.int64), (n,))
    if (ks > T - 1).any():
        raise ValueError("k ≤ T − 1")
    U = rng.random((n, T - 1)) if keys is None else np.asarray(keys)
    rk = np.empty((n, T - 1), np.int64)
    np.put_along_axis(rk, np.argsort(U, axis=1), np.broadcast_to(np.arange(T - 1), (n, T - 1)), axis=1)
    return np.c_[np.zeros(n, bool), rk < ks[:, None]]


def harvest_term(w: float, var_ann: float) -> float:
    """이론 변동성 수확 항 ½·w(1 − w)·σ² (σ² = 연환산 일 수익 분산) — 고정 비중이 구성 자산 가중 기하평균보다 더 버는 몫(참고, R0 대비 아님)."""
    return 0.5 * w * (1.0 - w) * var_ann


def bonferroni_p(p: float, k: int = N_CELLS) -> float:
    """부트스트랩 확률 P(X > 기준) → 칸 ``k`` 개 Bonferroni 보정 확률 = 1 − k·(1 − P) (0 아래는 0)."""
    return float(max(0.0, 1.0 - k * (1.0 - p)))


def cell_criteria(ok: dict) -> dict:
    """한 (규칙, w) 칸: 조건 1~3 은 두 세계 모두(조건 5), 조건 4 는 ``C4_WORLD``(정정). ``ok[(조건, 세계)]`` = bool."""
    c = {f"c{i}": all(ok[(f"c{i}", w)] for w in CASHES) for i in (1, 2, 3)}
    c["c4"] = ok[("c4", C4_WORLD)]
    c["c4_both"] = all(ok[("c4", w)] for w in CASHES)
    c["all"] = c["c1"] and c["c2"] and c["c3"] and c["c4"]
    return c


def rule_verdict(cells: dict, degenerate: set, rule: str) -> "tuple[bool, list[float]]":
    """규칙 채택 산술: 퇴화 칸(정정 — R0 와 같은 경로)을 뺀 **나머지 w 모두** 통과, 남은 칸이 하나 이상. → (통과, 판정에 쓴 w 목록)."""
    ws = [w for w in W_GRID if (rule, w) not in degenerate]
    return bool(ws) and all(cells[(rule, w)]["all"] for w in ws), ws


def pick_fewest(passed: "list[str]", reb_py: dict) -> "list[str]":
    """통과 규칙 → 리밸런스/년(판정에 쓴 w 평균) 적은 순. 동률이면 설계 표 순서."""
    return sorted(passed, key=lambda r: (reb_py[r], RULES.index(r)))


def family_max_pct(S: np.ndarray) -> np.ndarray:
    """칸 × 널 Sharpe 행렬(c, n — 같은 열 = 같은 무작위 키) → 널 표본마다 '칸들 가운데 자기 칸 안 백분위 최대'(n,).
    이 분포의 95 백분위 = 칸 상관을 감안한 가족 단위 조건 4 문턱(최대 통계량 순열 기준선)."""
    S = np.atleast_2d(np.asarray(S, float))
    n = S.shape[1]
    rk = np.argsort(np.argsort(S, axis=1), axis=1)
    return ((rk + 0.5) / n).max(axis=0)


# ---------------------------------------------------------------------------
# 시장
# ---------------------------------------------------------------------------

class Market:
    """지수 일봉 한 계열 · 현금(0 또는 기억값 CD91) · 그해 거래일 순번."""

    def __init__(self, csv: str, sym: str, cash: str):
        x = pd.read_csv(csv, parse_dates=["date"])
        k = x[x["symbol"] == sym].set_index("date").sort_index()
        self.sym = sym
        self.dates = pd.DatetimeIndex(k.index)
        self.O, self.C = k["open"].to_numpy(float), k["close"].to_numpy(float)
        self.ov = np.r_[np.nan, self.O[1:] / self.C[:-1] - 1.0]
        self.iday = self.C / self.O - 1.0
        self.rate = np.zeros(len(self.dates)) if cash == "zero" else mg2.cash_daily(self.dates, CD91_MEM)
        self.yord = year_ordinal(self.dates)

    def rows(self, lo: str, hi: str) -> "tuple[int, int]":
        i0, i1 = int(self.dates.searchsorted(pd.Timestamp(lo))), int(self.dates.searchsorted(pd.Timestamp(hi), "right"))
        if i0 < 1:
            raise ValueError("창 첫날 전 종가가 필요하다")
        return i0, i1

    def mkt(self, lo: str, hi: str) -> dict:
        i0, i1 = self.rows(lo, hi)
        return {"d": self.dates[i0:i1], "ov": self.ov[i0:i1], "iday": self.iday[i0:i1], "rate": self.rate[i0:i1],
                "yord": self.yord[i0:i1], "mon": mg2.month_ids(self.dates[i0:i1]), "i0": i0}

    def run(self, rule: str, w: float, lo: str, hi: str, tax: bool = False) -> dict:
        m = self.mkt(lo, hi)
        rate = m["rate"] * ((1.0 - TAX) if tax else 1.0)
        s = rule_sim(rule, w, m["d"], m["ov"], m["iday"], rate)
        s.update({"d": m["d"], "rate": m["rate"], "mon": m["mon"]})
        return s


def wstats(s: dict) -> dict:
    """한 창: CAGR(252/T)·MDD·Calmar(일별)·Sharpe(그 세계 현금 초과 — 0 이면 원)·원 Sharpe·리밸런스/년·평균·최소·최대 종가 비중(+ 날)·회전율/년."""
    r, d = s["r"], s["d"]
    st = mf.equity_stats(r)
    yrs = len(r) / 252.0
    j = int(np.argmax(s["wc"]))
    pc = np.flatnonzero(d == pd.Timestamp(PRECRASH))
    st.update({"sharpe": float(m4.excess_sharpe(r, s["rate"], s["mon"])[0]),
               "sharpe_raw": float(m4.excess_sharpe(r, np.zeros(len(r)), s["mon"])[0]),
               "n_reb": int(s["reb"].sum()), "reb_py": float(s["reb"].sum() / yrs), "w_avg": float(np.mean(s["wc"])),
               "w_min": float(np.min(s["wc"])), "w_max": float(s["wc"][j]), "w_max_date": str(d[j].date()),
               "w_precrash": float(s["wc"][pc[0]]) if len(pc) else np.nan, "w_end": float(s["wc"][-1]),
               "turn_py": float(s["turn"].sum() / yrs), "start": str(d[0].date()), "end": str(d[-1].date())})
    return st


def index_var(M: Market, lo: str, hi: str) -> dict:
    """창 안 지수 종가 일 수익 연환산 분산 σ²·변동성·기하 CAGR(252/T)."""
    i0, i1 = M.rows(lo, hi)
    r = M.C[i0:i1] / M.C[i0 - 1:i1 - 1] - 1.0
    return {"var": float(np.var(r) * 252.0), "vol": float(np.std(r) * np.sqrt(252.0)), "cagr": mf.equity_stats(r)["cagr"], "n": len(r)}


# ---------------------------------------------------------------------------
# 섞은 수익열 기저율(병렬 작업 — 정정)
# ---------------------------------------------------------------------------

def shuffle_task(job: dict) -> dict:
    """한 칸(규칙, w): 표본 밖 (밤, 장중) 수익을 날 단위로 함께 섞은 수익열(현금 금리는 달력 순서 그대로) ``N_SHUF`` 벌에서
    규칙 X 의 Sharpe 가 같은 횟수 무작위 날짜 널(``N_SHUF_NULL`` 벌)의 95 백분위를 넘는 비율 = 조건 4 의 기저 통과율."""
    rule, w, ov, iday, rate, mon, d = job["rule"], job["w"], job["ov"], job["iday"], job["rate"], job["mon"], job["d"]
    ns, nn, ch = job["n_shuf"], job["n_null"], job["chunk"]
    rng = np.random.default_rng(job["seed"])
    T = len(ov)
    ra = rule_args(rule, w, d)
    passes, pcts, ks = [], [], []
    for a in range(0, ns, ch):
        m = min(ch, ns - a)
        P = np.stack([rng.permutation(T) for _ in range(m)])
        OV, ID = ov[P], iday[P]
        X = simulate(OV, ID, rate, w, **ra)
        k = X["reb"].sum(axis=1)
        sx = m4.excess_sharpe(X["r"], rate, mon)
        kk = np.repeat(k, nn)
        N = simulate(np.repeat(OV, nn, axis=0), np.repeat(ID, nn, axis=0), rate, w, reb=random_reb_dates(T, kk, m * nn, rng))
        sn = m4.excess_sharpe(N["r"], rate, mon).reshape(m, nn)
        for i in range(m):
            ks.append(int(k[i]))
            if k[i] > 0:
                passes.append(bool(sx[i] > np.nanpercentile(sn[i], NULL_Q)))
                pcts.append(mg1.null_pct(sx[i], sn[i]))
    return {"rule": rule, "w": w, "n_valid": len(passes), "pass_rate": float(np.mean(passes)) if passes else np.nan,
            "mean_pct": float(np.mean(pcts)) if pcts else np.nan, "k_mean": float(np.mean(ks))}


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def plot_equity(panels: "list[tuple[str, pd.DatetimeIndex, np.ndarray, dict]]", path: str) -> "str | None":
    """열 = 창, 위 = 지수 매수보유(100%)와 R1 의 세 w 자산(로그), 아래 세 줄 = w 별 'R1 대비 상대 자산'(규칙 / R1 − 1). 그림 글자는 영어."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: PLC0415
    except ImportError:
        return None
    plt.rcParams["axes.unicode_minus"] = False
    widths = [max(1.0, (d[-1] - d[0]).days / 365.25 / 10.0) for _, d, _, _ in panels]
    fig, axes = plt.subplots(1 + len(W_GRID), len(panels), figsize=(5.2 * sum(widths), 11.5), sharex="col", facecolor="#fcfcfb",
                             gridspec_kw={"height_ratios": [2.0] + [1.0] * len(W_GRID), "width_ratios": widths})
    wcol = {0.4: "#86b6ef", 0.6: "#2a78d6", 0.8: "#104281"}
    for j, (title, d, idx_r, ser) in enumerate(panels):
        a0 = axes[0, j]
        a0.plot(d, np.cumprod(1.0 + idx_r), color="#8a8984", lw=1.2, label="index 100% (no cost)")
        for w in W_GRID:
            a0.plot(d, np.cumprod(1.0 + ser[(w, "R1")]), color=wcol[w], lw=1.6, label=f"R1 annual, w={w:.1f}")
        a0.set_yscale("log")
        a0.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        a0.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
        a0.yaxis.set_minor_formatter(NullFormatter())
        a0.set_title(title, fontsize=10, color="#0b0b0b")
        a0.legend(frameon=False, fontsize=8, loc="upper left")
        for i, w in enumerate(W_GRID, 1):
            ax = axes[i, j]
            base = np.cumprod(1.0 + ser[(w, "R1")])
            for nm in RULES:
                if nm == "R1":
                    continue
                ax.plot(d, np.cumprod(1.0 + ser[(w, nm)]) / base, color=COLORS[nm], lw=1.6 if nm in ("R0", CUR) else 1.1, label=nm)
            ax.axhline(1.0, color="#52514e", lw=0.8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{(y - 1) * 100:+.3g}%"))
            ax.set_ylabel(f"vs R1, w={w:.1f}", fontsize=9, color="#52514e")
        for ax in axes[:, j]:
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
    axes[0, 0].set_ylabel("equity (log)", fontsize=9, color="#52514e")
    h, lab = axes[1, 0].get_legend_handles_labels()
    fig.suptitle("META GEN 6 - stock+cash rebalancing rules (CD91 memory cash, costs incl., each window starts at target weight)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.legend(h, [f"{x} / R1 - 1" for x in lab], loc="lower center", ncol=len(lab), frameon=False, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--nulls", type=int, default=N_RAND)
    ap.add_argument("--boots", type=int, default=N_BOOT)
    ap.add_argument("--shuffles", type=int, default=N_SHUF)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    debug = os.path.normpath(args.out) != os.path.normpath(OUT_DIR)
    if (args.nulls != N_RAND or args.boots != N_BOOT or args.shuffles != N_SHUF) and not debug:
        ap.error("축소 실행은 --out 을 탐색 디렉터리 밖으로 둘 때만")
    sub = os.path.join(args.out, SUB)
    os.makedirs(sub, exist_ok=True)
    t0 = time.time()
    MK = {c: {"kospi": Market(KOSPI_CSV, KOSPI_SYM, c), "k200": Market(K200_CSV, K200_SYM, c)} for c in CASHES}
    _log(f"코스피 {MK['zero']['kospi'].dates[0].date()} ~ {MK['zero']['kospi'].dates[-1].date()} · "
         f"코스피200 {MK['zero']['k200'].dates[0].date()} ~ {MK['zero']['k200'].dates[-1].date()}", t0)

    # --- 계열: 창마다 목표 비중으로 새로 시작(정정) ---
    ser, stats = {}, {}
    for c in CASHES:
        for w in W_GRID:
            for nm in RULES:
                for wk, (lo, hi, _, mk) in WINS.items():
                    s = MK[c][mk].run(nm, w, lo, hi)
                    ser[(c, w, nm, wk)] = s
                    stats[(c, w, nm, wk)] = wstats(s)
    _log("계열·창 통계 끝", t0)

    # --- 퇴화 칸(정정): 표본 밖 창에서 한 번도 리밸런스하지 않음(어느 세계든) → R0 와 같은 경로 → 판정에서 뺀다 ---
    degenerate = {(nm, w) for nm in RULES if nm != "R0" for w in W_GRID if any(stats[(c, w, nm, "oos")]["n_reb"] == 0 for c in CASHES)}
    #: 기준과 같은 경로인 칸(예: 상한이 한 번도 안 걸린 R1+C10 ≡ R1) — 판정은 그대로(조건 1 P = 0 으로 구조상 ✗), 문서에 적는다.
    same_as_base = sorted({(tk, nm, w) for tk, base in TRACKS.items() for nm in RULES if nm not in ("R0", base) for w in W_GRID
                           if all(np.array_equal(ser[(c, w, nm, "oos")]["r"], ser[(c, w, base, "oos")]["r"]) for c in CASHES)})
    cap_n = {(c, w, wk): int((ser[(c, w, CUR, wk)]["reb"] & ~calendar_flags(ser[(c, w, CUR, wk)]["d"], "Y")).sum())
             for c in CASHES for w in W_GRID for wk in WINS}
    zero_other = sorted({(nm, w, wk, c) for nm in RULES if nm != "R0" for w in W_GRID for wk in WINS for c in CASHES
                         if stats[(c, w, nm, wk)]["n_reb"] == 0 and (nm, w) not in degenerate})

    # --- 조건 1: 짝지은 월 블록 부트스트랩(표본 밖) · 조건 4: 무작위 리밸런스 날짜 널(세계마다 같은 키 — 칸 상관 보존) ---
    boots, nulls, nullS = {}, {}, {}
    for ci, c in enumerate(CASHES):
        q0 = ser[(c, W_GRID[0], "R0", "oos")]
        mon, rate = q0["mon"], q0["rate"]
        B3 = m4.MonthBoot(mon, BLOCK_M, args.boots, SEED + BLOCK_M)
        B12 = m4.MonthBoot(mon, BLOCK_ALT, args.boots, SEED + BLOCK_ALT)
        m = MK[c]["kospi"].mkt(*OOS)
        T = len(m["d"])
        keys = np.random.default_rng([SEED, ci]).random((args.nulls, T - 1))
        for w in W_GRID:
            for nm in RULES:
                s = ser[(c, w, nm, "oos")]
                for base in ("R0", "R1", CUR):
                    b = ser[(c, w, base, "oos")]
                    boots[(c, w, nm, base)] = {"b3": m4.paired_boot(s["r"], b["r"], rate, B3), "b12": m4.paired_boot(s["r"], b["r"], rate, B12)}
                kk = int(s["reb"].sum())
                N = simulate(m["ov"], m["iday"], rate, w, reb=random_reb_dates(T, kk, args.nulls, keys=keys))
                sh = m4.excess_sharpe(N["r"], rate, mon)
                pm = m4.path_metrics(N["r"], mon, rate)
                nullS[(c, w, nm)] = sh
                nulls[(c, w, nm)] = {"k": kk, "reb_py": float(N["reb"].sum(axis=1).mean() / (T / 252.0)), "calmar": pm["calmar"],
                                     "p95": m4.qs(sh, NULL_Q), "p50": m4.qs(sh, 50), "calmar_p95": m4.qs(pm["calmar"], NULL_Q),
                                     "pct": mg1.null_pct(stats[(c, w, nm, "oos")]["sharpe"], sh)}
        _log(f"부트스트랩 · 널 끝 ({c})", t0)

    # --- 조건 산술 두 벌(정식 = 현행 R1+C10 대비, 참고 = R1 대비) ---
    rows, cells, verdict = [], {}, {}
    for tk, base in TRACKS.items():
        cands = tuple(nm for nm in RULES if nm not in ("R0", base))
        ok = {}
        for c in CASHES:
            for w in W_GRID:
                for nm in cands:
                    st = {wk: stats[(c, w, nm, wk)] for wk in WINS}
                    sb = {wk: stats[(c, w, base, wk)] for wk in WINS}
                    pb, p0 = boots[(c, w, nm, base)]["b3"]["p_sharpe"], boots[(c, w, nm, "R0")]["b3"]["p_sharpe"]
                    o = {"c1": bool(pb >= P_BOOT and p0 >= P_BOOT),
                         "c2": bool(st["h1"]["sharpe"] >= sb["h1"]["sharpe"] - 1e-12 and st["h2"]["sharpe"] >= sb["h2"]["sharpe"] - 1e-12),
                         "c3": bool(st["g4"]["sharpe"] >= sb["g4"]["sharpe"] - G4_SLACK - 1e-12),
                         "c4": bool(np.isfinite(nulls[(c, w, nm)]["p95"]) and st["oos"]["sharpe"] > nulls[(c, w, nm)]["p95"])}
                    for key, v in o.items():
                        ok[(nm, w, key, c)] = v
                    rows.append({"track": tk, "base": base, "cash": c, "w": w, "rule": nm, "degenerate": (nm, w) in degenerate,
                                 "p_vs_base": pb, "p_vs_R0": p0, "p_vs_base_b12": boots[(c, w, nm, base)]["b12"]["p_sharpe"],
                                 "p_vs_R0_b12": boots[(c, w, nm, "R0")]["b12"]["p_sharpe"],
                                 "bonf_vs_base": bonferroni_p(pb), "bonf_vs_R0": bonferroni_p(p0),
                                 **{f"sharpe_{wk}": st[wk]["sharpe"] for wk in WINS}, **{f"base_sharpe_{wk}": sb[wk]["sharpe"] for wk in WINS},
                                 "null_k": nulls[(c, w, nm)]["k"], "null_sharpe_p50": nulls[(c, w, nm)]["p50"],
                                 "null_sharpe_p95": nulls[(c, w, nm)]["p95"], "null_pct": nulls[(c, w, nm)]["pct"], **o})
        cl = {(nm, w): cell_criteria({(key, c): ok[(nm, w, key, c)] for key in ("c1", "c2", "c3", "c4") for c in CASHES})
              for nm in cands for w in W_GRID}
        cells[tk] = cl
        reb_py, passed, used = {}, [], {}
        for nm in cands:
            p, ws = rule_verdict(cl, degenerate, nm)
            used[nm] = ws
            reb_py[nm] = float(np.mean([stats[(C4_WORLD, w, nm, "oos")]["reb_py"] for w in (ws or W_GRID)]))
            if p:
                passed.append(nm)
        verdict[tk] = {"base": base, "cands": cands, "passed": passed, "order": pick_fewest(passed, reb_py), "reb_py": reb_py, "used": used}
    for r in rows:
        for key in ("c1", "c2", "c3", "c4", "c4_both", "all"):
            r[f"cell_{key}"] = cells[r["track"]][(r["rule"], r["w"])][key]

    # --- 칸 상관을 감안한 순열 기준선(조건 4, CD91 세계): 정식 후보의 비퇴화 칸 ---
    fam = [(nm, w) for nm in verdict["cur"]["cands"] for w in W_GRID if (nm, w) not in degenerate]
    fmax = family_max_pct(np.stack([nullS[(C4_WORLD, w, nm)] for nm, w in fam]))
    fam_thr = float(np.percentile(fmax, 95))
    fam_hits = [(nm, w, nulls[(C4_WORLD, w, nm)]["pct"]) for nm, w in fam if nulls[(C4_WORLD, w, nm)]["pct"] > fam_thr]
    _log("조건 산술 · 순열 기준선 끝", t0)

    # --- 정정 기술 줄: 비중 맞춘 R0 비교 · 연 1회 기준일 분포 · 현행 상한의 위험 줄 ---
    wmatch = []
    for c in CASHES:
        for wk, (lo, hi, _, mk) in WINS.items():
            for w in W_GRID:
                s0 = stats[(c, w, "R0", wk)]
                s1 = wstats(MK[c][mk].run("R1", s0["w_avg"], lo, hi))
                wmatch.append({"cash": c, "win": wk, "w": w, "w_bar": s0["w_avg"], **{f"r0_{k}": s0[k] for k in ("cagr", "mdd", "calmar", "sharpe")},
                               **{f"r1m_{k}": s1[k] for k in ("cagr", "mdd", "calmar", "sharpe", "w_avg", "n_reb")}})
    anchors = []
    for c in CASHES:
        for wk, (lo, hi, _, mk) in WINS.items():
            m = MK[c][mk].mkt(lo, hi)
            F = anchor_flags(m["yord"], np.arange(N_ANCHOR))
            for w in W_GRID:
                A = simulate(m["ov"], m["iday"], m["rate"], w, reb=F)
                sh = m4.excess_sharpe(A["r"], m["rate"], m["mon"])
                assert abs(sh[0] - stats[(c, w, "R1", wk)]["sharpe"]) < 1e-9          # 기준일 0 = R1
                anchors.append({"cash": c, "win": wk, "w": w, "r1": float(sh[0]), "pct": mg1.null_pct(sh[0], sh[1:]),
                                "p5": m4.qs(sh, 5), "p50": m4.qs(sh, 50), "p95": m4.qs(sh, 95), "best_k": int(np.nanargmax(sh)),
                                "worst_k": int(np.nanargmin(sh)), **{f"k{k}": float(v) for k, v in enumerate(sh)}})
    _log("비중 맞춤 · 기준일 분포 끝", t0)

    # --- 정정 기술 줄: 섞은 수익열에서의 조건 4 기저 통과율(CD91 세계, 병렬) ---
    m = MK[C4_WORLD]["kospi"].mkt(*OOS)
    jobs = [{"rule": nm, "w": w, "ov": m["ov"], "iday": m["iday"], "rate": m["rate"], "mon": m["mon"], "d": m["d"],
             "n_shuf": args.shuffles, "n_null": N_SHUF_NULL if not debug else min(N_SHUF_NULL, args.nulls), "chunk": SHUF_CHUNK,
             "seed": [SEED, 77, RULES.index(nm), W_GRID.index(w)]} for nm in RULES if nm != "R0" for w in W_GRID]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        shuf = {(r["rule"], r["w"]): r for r in ex.map(shuffle_task, jobs)}
    _log("섞은 수익열 기저율 끝", t0)

    # --- 기술: 변동성 수확 항 · 세후(현금 이자 15.4%) ---
    harv = []
    for wk, (lo, hi, lab, mk) in WINS.items():
        iv = index_var(MK["zero"][mk], lo, hi)
        for w in W_GRID:
            harv.append({"win": wk, "w": w, "vol": iv["vol"], "var": iv["var"], "idx_cagr": iv["cagr"], "harvest": harvest_term(w, iv["var"]),
                         **{f"dcagr_{nm}": stats[("zero", w, nm, wk)]["cagr"] - stats[("zero", w, "R0", wk)]["cagr"] for nm in RULES}})
    tax = []
    for w in W_GRID:
        for nm in RULES:
            for wk in ("oos", "g4"):
                lo, hi, _, mk = WINS[wk]
                e = mf.equity_stats(MK["cd91"][mk].run(nm, w, lo, hi, tax=True)["r"])
                tax.append({"w": w, "rule": nm, "win": wk, "cagr": stats[("cd91", w, nm, wk)]["cagr"], "cagr_tax": e["cagr"], "mdd_tax": e["mdd"]})
    _log("기술 줄 끝", t0)

    # --- 산출물 ---
    pd.DataFrame([{"cash": c, "w": w, "rule": nm, "win": wk, **s} for (c, w, nm, wk), s in stats.items()]).to_csv(
        os.path.join(sub, "summary.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(sub, "decision.csv"), index=False)
    pd.DataFrame([{"cash": c, "w": w, "rule": nm, "k": nulls[(c, w, nm)]["k"], "sharpe": a, "calmar": b}
                  for (c, w, nm), sh in nullS.items() for a, b in zip(sh, nulls[(c, w, nm)]["calmar"], strict=True)]).to_csv(
        os.path.join(sub, "null_random.csv"), index=False, float_format="%.6g")
    pd.DataFrame([{"cash": c, "w": w, "rule": nm, "base": base, **{f"{k}_{x}": v for k, b in bb.items() for x, v in b.items()
                                                                  if not isinstance(v, list)},
                   **{f"{k}_dsharpe_q{q}": b["dsharpe"][i] for k, b in bb.items() for i, q in enumerate((5, 50, 95))}}
                  for (c, w, nm, base), bb in boots.items()]).to_csv(os.path.join(sub, "bootstrap.csv"), index=False)
    pd.DataFrame(wmatch).to_csv(os.path.join(sub, "weight_matched_r0.csv"), index=False)
    pd.DataFrame(anchors).to_csv(os.path.join(sub, "anchors.csv"), index=False, float_format="%.6g")
    pd.DataFrame(list(shuf.values())).to_csv(os.path.join(sub, "null_base_rate.csv"), index=False)
    pd.DataFrame(harv).to_csv(os.path.join(sub, "harvest.csv"), index=False)
    pd.DataFrame(tax).to_csv(os.path.join(sub, "after_tax.csv"), index=False)
    for wk in WINS:
        nav = pd.DataFrame(index=ser[("cd91", W_GRID[0], "R0", wk)]["d"])
        for c in CASHES:
            for w in W_GRID:
                for nm in RULES:
                    nav[f"{nm} w{w:.1f} [{c}]"] = np.cumprod(1.0 + ser[(c, w, nm, wk)]["r"])
                    if c == C4_WORLD:
                        nav[f"{nm} w{w:.1f} 비중"] = ser[(c, w, nm, wk)]["wc"]
        nav.to_csv(os.path.join(sub, f"nav_{wk}.csv"), float_format="%.6g")

    def panel(wk, title):
        lo, hi, _, mk = WINS[wk]
        M = MK[C4_WORLD][mk]
        i0, i1 = M.rows(lo, hi)
        return (title, M.dates[i0:i1], M.C[i0:i1] / M.C[i0 - 1:i1 - 1] - 1.0, {(w, nm): ser[(C4_WORLD, w, nm, wk)]["r"] for w in W_GRID for nm in RULES})
    fig = plot_equity([panel("oos", "Out-of-sample 1997-12 ~ 2016-12 (KOSPI)"), panel("g4", "2017-01 ~ 2026-09 (KOSPI200)")],
                      os.path.join(sub, "equity.png"))
    led = write_ledger_rows(sub)
    elapsed = time.time() - t0
    L = render(locals())
    with open(os.path.join(args.out, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    vc, vr = verdict["cur"], verdict["r1"]
    print(f"\n=== 메타 세대 6 — {elapsed:.0f}초 · 정식(기준 {CUR}) 통과: {', '.join(vc['passed']) or '없음'} → "
          f"{vc['order'][0] if vc['order'] else f'{CUR} 유지(근거 없음)'} · 참고(기준 R1) 통과: {', '.join(vr['passed']) or '없음'} ===")
    print(f"→ {os.path.join(args.out, RESULTS_MD)}")
    return 0


def write_ledger_rows(sub: str) -> int:
    """원장 ``meta_gen6/TRIALS.jsonl`` — (규칙, w) 칸 한 줄씩(결과 숫자는 넣지 않는다)."""
    path = os.path.join(sub, "TRIALS.jsonl")
    seen = g1.read_fingerprints(path)
    base = {"design": DESIGN_DOC, "asset": "KOSPI(Yahoo ^KS11) 1997-12~2016-12 · KPI200(네이버) 2017-01~2026-09", "window": f"{OOS[0]} ~ {OOS[1]}",
            "cost_1w": COST_1W, "fee_yr": FEE_YR, "exec": "d 종가 결정 → d+1 시가", "start": "창마다 목표 비중", "cash": "0% · CD91 기억값",
            "sharpe": "월 복리 초과"}
    rows = []
    for nm in RULES:
        for w in W_GRID:
            cfg = {**base, "procedure": nm, "w": w}
            if nm in CALENDAR:
                cfg["calendar"] = CALENDAR[nm]
            if nm in BANDS:
                cfg["band"] = list(BANDS[nm])
            rows.append({"config": cfg, "note": "메타 세대 6 — 리밸런싱 규칙"})
    return g1.write_ledger(path, rows, seen)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

_DESC = {"R0": "표류(안 함)", "R1": "연 1회", CUR: "연 1회 + 상한 +10%p(현행)", "R2": "분기 1회", "R3": "월 1회", "B5": "±5%p", "B10": "±10%p",
         "B20": "±20%(상대)"}


def _adopt_table(v: dict, tk: str) -> "list[str]":
    rows, cells, degenerate = v["rows"], v["cells"][tk], v["degenerate"]
    vd = v["verdict"][tk]
    base = vd["base"]
    rd = {(r["cash"], r["w"], r["rule"]): r for r in rows if r["track"] == tk}
    L = [f"### {TRACK_LAB[tk]}", "",
         f"| 규칙 | w | 1 P(>{base}) · P(>R0) [0% / CD91] | 2 Sharpe 97-06 · {base} / 07-16 · {base} [0% ‖ CD91] | 3 Sharpe 17-26 · {base} [0% / CD91] | "
         "4 Sharpe · 널 p95 (백분위) [CD91 정식 / 0% 참고] | 칸 통과 |", "|---|---|---|---|---|---|---|"]
    for nm in vd["cands"]:
        for w in W_GRID:
            z, k = rd[("zero", w, nm)], rd[("cd91", w, nm)]
            cc = cells[(nm, w)]
            tag = " · **퇴화(판정 제외)**" if (nm, w) in degenerate else ""
            L.append(f"| **{nm}** | {w:.1f}{tag} | {z['p_vs_base']:.2f} · {z['p_vs_R0']:.2f} / {k['p_vs_base']:.2f} · {k['p_vs_R0']:.2f} {_ck(cc['c1'])} | "
                     f"{_f(z['sharpe_h1'], '{:.3f}')} · {_f(z['base_sharpe_h1'], '{:.3f}')} / {_f(z['sharpe_h2'], '{:.3f}')} · {_f(z['base_sharpe_h2'], '{:.3f}')} ‖ "
                     f"{_f(k['sharpe_h1'], '{:.3f}')} · {_f(k['base_sharpe_h1'], '{:.3f}')} / {_f(k['sharpe_h2'], '{:.3f}')} · {_f(k['base_sharpe_h2'], '{:.3f}')} "
                     f"{_ck(cc['c2'])} | {_f(z['sharpe_g4'], '{:.3f}')} · {_f(z['base_sharpe_g4'], '{:.3f}')} / {_f(k['sharpe_g4'], '{:.3f}')} · "
                     f"{_f(k['base_sharpe_g4'], '{:.3f}')} {_ck(cc['c3'])} | "
                     f"{_f(k['sharpe_oos'], '{:.3f}')} · {_f(k['null_sharpe_p95'], '{:.3f}')} ({k['null_pct'] * 100:.0f}%) {_ck(k['c4'])} / "
                     f"{_f(z['sharpe_oos'], '{:.3f}')} · {_f(z['null_sharpe_p95'], '{:.3f}')} ({z['null_pct'] * 100:.0f}%) {_ck(z['c4'])} | "
                     f"**{_ck(cc['all'])}** |")
    L += ["", "**규칙 단위(퇴화 칸 뺀 나머지 w 모두):**", "",
          "| 규칙 | 판정에 쓴 w | 리밸런스/년 (그 w 평균, 표본 밖 CD91) | 1 | 2 | 3 | 4 | **채택 조건 1~5** |", "|---|---|---|---|---|---|---|---|"]
    for nm in vd["cands"]:
        ws = vd["used"][nm]
        f = lambda key, ws=ws, nm=nm: _ck(bool(ws) and all(cells[(nm, w)][key] for w in ws))  # noqa: E731
        p, _ = rule_verdict(cells, degenerate, nm)
        L.append(f"| {nm} {_DESC[nm]} | {', '.join(f'{w:.1f}' for w in ws) or '없음'} | {vd['reb_py'][nm]:.2f} | {f('c1')} | {f('c2')} | {f('c3')} | "
                 f"{f('c4')} | **{_ck(p)}** |")
    L += [""]
    if vd["order"]:
        L += [f"**산술 결과:** 통과 = {', '.join(vd['passed'])} → 리밸런스 적은 순 {', '.join(vd['order'])}. 설계 규칙상 리밸런싱 규칙 = "
              f"**{vd['order'][0]}**({_DESC[vd['order'][0]]}).", ""]
    else:
        L += [f"**산술 결과:** 통과한 규칙 없음 → 설계 규칙상 **{base} 유지, '다른 규칙이 {base} 보다 낫다는 근거는 없음'** "
              "(리뷰 MAJOR-1: '연 1회가 낫다' 가 아니라 '어느 규칙도 낫다는 근거가 없다' — 1월 날짜는 편의).", ""]
    return L


def render(v: dict) -> "list[str]":  # noqa: PLR0915
    args, stats, rows, boots, nulls = v["args"], v["stats"], v["rows"], v["boots"], v["nulls"]
    verdict, degenerate, zero_other = v["verdict"], v["degenerate"], v["zero_other"]
    harv, tax, wmatch, anchors, shuf = v["harv"], v["tax"], v["wmatch"], v["anchors"], v["shuf"]
    MK = v["MK"]
    L = [f"# 메타 세대 6 — 박스권 한국 시장의 리밸런싱 규칙 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "설계 `META_GEN6.md`(f5e71ba + 정정 c2924d8, 둘 다 결과 보기 전)를 계산한 숫자다. ✓/✗ 는 미리 적은 문턱과의 **산술 비교**이고 판정은 사람이 한다. "
         f"규칙 {len(RULES)} × 목표 비중 {len(W_GRID)} = {N_CELLS} 칸(원 설계 21 + 정정 현행 {CUR} 3)은 설계 그대로이고 이 세대에서 고른 것은 없다.", "",
         f"재현: `uv run python research/experiments/meta_gen6_run.py` · 실행 {v['elapsed']:.0f}초 · 무작위 리밸런스 날짜 널 {args.nulls}벌/칸·세계 · "
         f"부트스트랩 {args.boots}벌 · 섞은 수익열 {args.shuffles}벌 × 널 {N_SHUF_NULL}벌/칸 · DB 접속 없음.", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 출력 `{args.out}`. 측정으로 쓰지 않는다.", ""]
    L += ["## 먼저 읽을 것", "",
          f"- **채택 산술은 두 벌이다.** 정정이 '{CUR} 이 R1 대신 현행 비교 기준' 이라 했으므로 **정식 = 조건 1~3 의 기준을 {CUR} 로**(후보 R1·R2·R3·B5·B10·B20), "
          f"원 설계 글자 그대로 기준 = R1 인 표(후보 {CUR}·R2·R3·B5·B10·B20)를 **참고**로 나란히 둔다. 조건 1 의 'vs R0' 는 두 벌 모두 같다.",
          "- **현금 두 세계:** (i) 0% — Sharpe = 원 Sharpe, (ii) CD91 기억값(세대 5 러너 `CD91_MEM`, **기억값 · DB 밖**) — Sharpe = 월 복리 초과 Sharpe. "
          "조건 1~3 은 두 세계 모두(조건 5), **조건 4 = (ii) 세계**(정정).",
          "- **조건 4 의 널:** 표본 밖 창에서 X 가 실제로 한 리밸런스 횟수 k 와 **정확히 같은 수**의 날을 창 안 거래일(첫날 제외)에서 비복원 균등으로 뽑아, "
          "그날 시가에 w 로 되돌리는 경로 1,000벌(같은 엔진·비용·현금). 한 세계 안 모든 칸이 같은 무작위 키를 쓴다(칸 상관 보존 → 아래 순열 기준선). "
          "**이 널은 밴드 쪽으로 기운다** — 날 단위로 섞은(구조 없는) 수익열에서의 기저 통과율을 아래에 함께 적는다(정정).",
          "- **체결:** 모든 결정은 d 종가 → d+1 시가. 달력 규칙 = 기간 마지막 거래일 종가에 판단 → 기간 첫 거래일 시가. 밴드·상한 = 종가 비중이 경계 밖인 날 d → d+1 시가. "
          "리밸런스 = 그 시가 NAV 로 정확히 w(비용 = 매매 금액 × 0.07%, NAV 에서).",
          "- **창마다 목표 비중으로 새로 시작**(정정): 표본 밖·두 분리 창·2017~2026 모두 창 첫날 직전 종가에 w 를 들고 시작(진입 비용 없음). "
          "분리 창은 표본 밖 계열을 자른 것이 아니라 따로 돌린 것이다.",
          "- **퇴화 칸**(정정): 표본 밖 창에서 (어느 세계든) 한 번도 리밸런스하지 않은 칸은 R0 와 같은 경로라 판정에서 뺀다 → "
          + (", ".join(f"{nm} · w {w:.1f}" for nm, w in sorted(degenerate, key=lambda x: (RULES.index(x[0]), x[1]))) or "없음")
          + ". 규칙 채택은 남은 w 모두로 본다. 표본 밖에서는 리밸런스했지만 다른 창(새로 시작)에서 0회인 칸(그 창에서는 R0 와 같은 경로 — 비교는 그대로 씀): "
          + ("; ".join(f"{nm} w {w:.1f} {wk} [{c}]" for nm, w, wk, c in zero_other) or "없음") + ".",
          "- **기준과 같은 경로인 칸**(표본 밖, 두 세계 모두 — 판정은 그대로 두고 조건 1 P = 0 으로 구조상 ✗): "
          + ("; ".join(f"{nm} w {w:.1f} = {TRACKS[tk]}" for tk, nm, w in v["same_as_base"]) or "없음") + ".",
          "- **가격 지수(배당 제외).** 배당은 모든 규칙에 거의 같은 비율(평균 주식 비중 × 배당률)로 더해진다 — 규칙 간 Sharpe 비교에는 2차 효과. R0 는 비중이 높아 "
          "CAGR 에서 배당만큼 손해(리뷰 MINOR).", ""]

    L += ["## 채택 산술 (META_GEN6.md '판정 (미리)' + 정정)", "",
          "칸마다: **1** 표본 밖 짝지은 3개월 블록 부트스트랩 P(Sharpe X > 기준) ≥ 0.8 **그리고** P(Sharpe X > R0) ≥ 0.8 · **2** 두 분리 창(각각 새로 시작) "
          "Sharpe X ≥ 기준 · **3** 2017~2026(코스피200) Sharpe X ≥ 기준 − 0.02 · **4** 표본 밖 Sharpe X > 무작위 리밸런스 날짜 널 Sharpe p95(CD91 세계) · "
          "**5** 1~3 은 현금 (i)·(ii) 두 세계 모두. 규칙 채택 = 퇴화 칸을 뺀 **나머지 w 모두** 1~5. 통과가 여럿이면 리밸런스 횟수가 가장 적은 것.", ""]
    for tk in TRACKS:
        L += _adopt_table(v, tk)
    # Bonferroni + 순열
    rc = [r for r in rows if r["track"] == "cur"]
    L += ["### 다중검정", "",
          f"**Bonferroni:** 보정 확률 = 1 − k × (1 − P), k = {N_CELLS} 칸(정정 뒤 격자; 원 설계 21 칸이면 문턱 P ≥ {1 - 0.2 / N_CELLS_DESIGN:.4f}). "
          f"보정 뒤 ≥ 0.8 이려면 P ≥ {1 - 0.2 / N_CELLS:.4f}. 정식 후보 칸의 조건 1 두 확률 최솟값(세계별)과 보정값:", "",
          "| 규칙 | " + " | ".join(f"w {w:.1f}: min P [0% / CD91] → 보정" for w in W_GRID) + " |", "|---|" + "---|" * len(W_GRID)]
    rd = {(r["cash"], r["w"], r["rule"]): r for r in rc}
    for nm in verdict["cur"]["cands"]:
        cs = []
        for w in W_GRID:
            pz = min(rd[("zero", w, nm)]["p_vs_base"], rd[("zero", w, nm)]["p_vs_R0"])
            pk = min(rd[("cd91", w, nm)]["p_vs_base"], rd[("cd91", w, nm)]["p_vs_R0"])
            cs.append(f"{pz:.3f} / {pk:.3f} → {bonferroni_p(pz):.2f} / {bonferroni_p(pk):.2f} {_ck(min(bonferroni_p(pz), bonferroni_p(pk)) >= P_BOOT)}")
        L.append(f"| {nm} | " + " | ".join(cs) + " |")
    L += ["", f"**칸 상관을 감안한 순열 기준선(조건 4, CD91 세계):** 정식 후보의 비퇴화 {len(v['fam'])} 칸이 같은 무작위 키를 쓰므로, 널 표본마다 "
          "'칸들 가운데 자기 칸 널 분포 안 백분위 최대' 를 모으면 가족 단위 문턱이 된다(`family_max_pct`). "
          f"그 95 백분위 = **{v['fam_thr'] * 100:.1f}%**(칸 하나의 명목 95% 대신). 이 문턱을 넘는 칸: "
          + (", ".join(f"{nm} w {w:.1f} ({p * 100:.1f}%)" for nm, w, p in v["fam_hits"]) or "없음") + ".", ""]

    # --- 정정 기술 줄 ---
    L += ["## 조건 4 의 기저 통과율 — 날 단위로 섞은 수익열 (정정, CD91 세계, 표본 밖)", "",
          f"표본 밖 (밤, 장중) 수익 쌍을 날 단위로 섞어 시계열 구조(추세·평균회귀)를 없앤 수익열 {args.shuffles}벌(현금 금리는 달력 순서 그대로)마다, "
          f"규칙 X 를 돌리고 같은 횟수 무작위 날짜 널 {N_SHUF_NULL}벌의 95 백분위를 넘었는지 센다. 명목 = 5%. "
          "이 비율이 5% 보다 크면 조건 4 는 '언제를 잘 고른다' 가 아니라 '비중 이탈을 묶는다' 만으로도 통과한다.", "",
          "| 규칙 | " + " | ".join(f"w {w:.1f}: 기저 통과율 · 평균 백분위 · 평균 k (실제 백분위)" for w in W_GRID) + " |", "|---|" + "---|" * len(W_GRID)]
    for nm in RULES:
        if nm == "R0":
            continue
        cs = []
        for w in W_GRID:
            q = shuf[(nm, w)]
            cs.append(f"{_f(q['pass_rate'] * 100 if np.isfinite(q['pass_rate']) else np.nan, '{:.0f}')}% ({q['n_valid']}) · "
                      f"{_f(q['mean_pct'] * 100 if np.isfinite(q['mean_pct']) else np.nan, '{:.0f}')}% · {q['k_mean']:.1f} "
                      f"({nulls[(C4_WORLD, w, nm)]['pct'] * 100:.0f}%)")
        L.append(f"| {nm} | " + " | ".join(cs) + " |")
    L += ["", "괄호 (n) = k > 0 인 섞은 수익열 수(k = 0 이면 널이 정의되지 않아 뺀다).", ""]

    L += ["## 연 1회 기준일 운 (정정) — 그해 k 번째 거래일 시가 리밸런스, k = 0..239", "",
          "R1 = k 0(1월 첫 거래일). 같은 엔진·비용·창(새로 시작). 백분위 = 다른 239 기준일 가운데 R1 보다 Sharpe 가 낮은 비율(+ 같으면 ½). "
          "높으면 '1월 첫날' 이 운 좋은 기준일이었다는 뜻 — 조건 1·2 의 기준선이 기준일 분포의 위쪽이다.", "",
          "| 현금 | 창 | " + " | ".join(f"w {w:.1f}: R1 Sharpe · **백분위** (기준일 p5 / p50 / p95)" for w in W_GRID) + " |", "|---|---|" + "---|" * len(W_GRID)]
    ad = {(a["cash"], a["win"], a["w"]): a for a in anchors}
    for c in CASHES:
        for wk in WINS:
            cs = [f"{ad[(c, wk, w)]['r1']:.3f} · **{ad[(c, wk, w)]['pct'] * 100:.0f}%** ({ad[(c, wk, w)]['p5']:.3f} / {ad[(c, wk, w)]['p50']:.3f} / "
                  f"{ad[(c, wk, w)]['p95']:.3f})" for w in W_GRID]
            L.append(f"| {c} | {WINS[wk][2]} | " + " | ".join(cs) + " |")
    L += ["", "참고 — 기준일 중앙값(p50)과 후보의 Sharpe 를 견주면 'R1 이 아니라 연 1회 규칙의 기대값' 대비 위치가 된다(표본 밖 숫자는 아래 '숫자' 표).", ""]

    L += ["## 비중 맞춘 R0 비교 (정정) — R0 의 실현 평균 비중 w̄ 로 돌린 R1", "",
          "R0(표류)는 주식이 오르면 비중이 w 보다 커진다 → CAGR·MDD 비교에 비중 효과가 섞인다. 같은 창·세계에서 R0 의 평균 종가 비중 w̄ 를 목표로 R1 을 돌려 "
          "같은 위험 수준에서 잰다. 조건 1 의 'vs R0'(Sharpe)는 그대로 두고 이 줄은 옆에 둔다.", "",
          "| 현금 | 창 | w | R0: w̄ · CAGR · MDD · Calmar · Sharpe | R1 @ w̄: CAGR · MDD · Calmar · Sharpe | ΔCAGR · ΔMDD (R1@w̄ − R0) |",
          "|---|---|---|---|---|---|"]
    for x in wmatch:
        L.append(f"| {x['cash']} | {WINS[x['win']][2]} | {x['w']:.1f} | {x['w_bar']:.2f} · {_pct(x['r0_cagr'])} · {_pct(x['r0_mdd'])} · {_f(x['r0_calmar'])} · "
                 f"{_f(x['r0_sharpe'], '{:.3f}')} | {_pct(x['r1m_cagr'])} · {_pct(x['r1m_mdd'])} · {_f(x['r1m_calmar'])} · {_f(x['r1m_sharpe'], '{:.3f}')} | "
                 f"{(x['r1m_cagr'] - x['r0_cagr']) * 100:+.2f}%p · {(x['r1m_mdd'] - x['r0_mdd']) * 100:+.1f}%p |")
    L += [""]

    L += ["## 현행 상한(C10)의 위험 줄 (정정) — MDD · 최대 표류 비중", "",
          f"상한은 수익원이 아니라 **위험 예산 유지 장치**라 Sharpe 판정과 별개로 적는다. 최대 비중 = 창 안 종가 주식 비중 최대(날). "
          f"급락 전날 = {PRECRASH}(2026-06-22 ~ 07-30 급락 직전, 2017~2026 창만).", "",
          "| 현금 | 창 | w | R0: MDD · 최대 비중 (날) | R1: MDD · 최대 비중 (날) · 급락 전날 | " + f"{CUR}: MDD · 최대 비중 (날) · 급락 전날 · 상한 촉발 수 |",
          "|---|---|---|---|---|---|"]
    for c in CASHES:
        for wk in WINS:
            for w in W_GRID:
                a, b, x = stats[(c, w, "R0", wk)], stats[(c, w, "R1", wk)], stats[(c, w, CUR, wk)]
                pc = lambda s: f" · {s['w_precrash'] * 100:.0f}%" if np.isfinite(s["w_precrash"]) else ""  # noqa: E731
                L.append(f"| {c} | {WINS[wk][2]} | {w:.1f} | {_pct(a['mdd'])} · {a['w_max'] * 100:.0f}% ({a['w_max_date']}) | "
                         f"{_pct(b['mdd'])} · {b['w_max'] * 100:.0f}% ({b['w_max_date']}){pc(b)} | "
                         f"{_pct(x['mdd'])} · {x['w_max'] * 100:.0f}% ({x['w_max_date']}){pc(x)} · {v['cap_n'][(c, w, wk)]} |")
    L += ["", f"상한 촉발 수 = {CUR} 의 리밸런스 가운데 1월 첫 거래일이 아닌 것(상한으로만 생긴 매매). 최대 비중이 w + 10%p 를 조금 넘는 것은 "
          "그날 종가에 넘고 다음 시가에 되돌렸기 때문이다.", ""]

    # --- 숫자 ---
    L += ["## 숫자 (비용·보수 포함, 가격 지수)", "",
          "CAGR = 252/T, MDD = 일별 경로(초기 1 을 첫 고점으로), Calmar = CAGR/|MDD|, Sharpe = 월 복리 (초과)수익 평균/표준편차 × √12(그 세계 현금 기준; 0% 면 원), "
          "원 SR = 현금 0 기준. 리밸런스/년 = 실행한 리밸런스 수 / (거래일/252). 평균 비중 = 종가 주식 비중 평균(최소~최대). 회전율/년 = Σ 매매 금액/NAV / 년.", ""]
    for c, clab in CASHES.items():
        L += [f"### {clab}", ""]
        for wk, (lo, hi, lab, _) in WINS.items():
            L += [f"**{lab}**", "", "| w | 규칙 | CAGR | MDD | Calmar | Sharpe | 원 SR | 리밸런스/년 (횟수) | 평균 비중 (최소~최대) | 회전율/년 | 기간 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for w in W_GRID:
                for nm in RULES:
                    s = stats[(c, w, nm, wk)]
                    L.append(f"| {w:.1f} | {nm} | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe'], '{:.3f}')} | "
                             f"{_f(s['sharpe_raw'], '{:.3f}')} | {s['reb_py']:.2f} ({s['n_reb']}) | {s['w_avg'] * 100:.0f}% ({s['w_min'] * 100:.0f}~{s['w_max'] * 100:.0f}%) | "
                             f"{s['turn_py'] * 100:.1f}% | {s['start']} ~ {s['end']} |")
            L.append("")

    # --- 변동성 수확 ---
    L += ["## 참고: 이론 변동성 수확 항 ½·w(1 − w)·σ² (창별)", "",
          "σ² = 그 창 지수 종가 일 수익 분산 × 252. 이 항은 고정 비중이 **구성 자산 가중 기하평균**보다 더 버는 몫의 근사이고 **표류(R0)보다 나은 몫이 아니다** "
          "(그 차이는 추세·평균회귀 모양이 정한다 — 리뷰 MAJOR-6). 오른쪽 = 현금 0% 세계 실제 CAGR − R0 CAGR(비중 효과 섞임 — 위 '비중 맞춘 R0' 참고).", "",
          "| 창 | 지수 연 변동성 · 지수 CAGR | w | ½w(1−w)σ² | " + " | ".join(f"ΔCAGR {nm}" for nm in RULES if nm != "R0") + " |",
          "|---|---|---|---|" + "---|" * (len(RULES) - 1)]
    for h in harv:
        L.append(f"| {WINS[h['win']][2]} | {h['vol'] * 100:.1f}% · {_pct(h['idx_cagr'])} | {h['w']:.1f} | {h['harvest'] * 100:.2f}%p | "
                 + " | ".join(f"{h[f'dcagr_{nm}'] * 100:+.2f}%p" for nm in RULES if nm != "R0") + " |")
    L += [""]

    # --- 널 ---
    L += [f"## 무작위 리밸런스 날짜 널 (표본 밖, {args.nulls}벌/칸·세계)", "",
          "| 현금 | w | 규칙 | k(실제 리밸런스 수) | 널 리밸런스/년 | Sharpe · 널 p50 · p95 (백분위) | Calmar · 널 p95 |", "|---|---|---|---|---|---|---|"]
    for c in CASHES:
        for w in W_GRID:
            for nm in RULES:
                n, s = nulls[(c, w, nm)], stats[(c, w, nm, "oos")]
                L.append(f"| {c} | {w:.1f} | {nm} | {n['k']} | {n['reb_py']:.2f} | {_f(s['sharpe'], '{:.3f}')} · {_f(n['p50'], '{:.3f}')} · "
                         f"{_f(n['p95'], '{:.3f}')} ({n['pct'] * 100:.0f}%) | {_f(s['calmar'])} · {_f(n['calmar_p95'])} |")
    # --- 부트스트랩 ---
    L += ["", "## 부트스트랩 상세 (표본 밖, 짝지은 월 블록 — 같은 세계 안 모든 비교에 같은 달 표본)", "",
          "| 현금 | w | 규칙 | vs | P(SR) 3개월 · 12개월 | ΔSR 5·50·95% (3개월) |", "|---|---|---|---|---|---|"]
    for c in CASHES:
        for w in W_GRID:
            for nm in RULES:
                for base in ("R0", "R1", CUR):
                    if nm == base:
                        continue
                    b = boots[(c, w, nm, base)]
                    L.append(f"| {c} | {w:.1f} | {nm} | {base} | {b['b3']['p_sharpe']:.3f} · {b['b12']['p_sharpe']:.3f} | "
                             + " · ".join(f"{q:+.3f}" for q in b["b3"]["dsharpe"]) + " |")
    # --- 세후 ---
    L += ["", "## 기술: 세후(현금 이자 15.4%, CD91 세계) CAGR", "", "ETF 매매차익 비과세 가정 — 현금 조각 이자에만 15.4%. 가격 지수(분배금 없음).", "",
          "| w | 규칙 | 표본 밖 CAGR → 세후 | 2017~2026 CAGR → 세후 |", "|---|---|---|---|"]
    tdf = pd.DataFrame(tax)
    for w in W_GRID:
        for nm in RULES:
            a = tdf[(tdf["w"] == w) & (tdf["rule"] == nm)].set_index("win")
            L.append(f"| {w:.1f} | {nm} | {_pct(a.loc['oos', 'cagr'])} → {_pct(a.loc['oos', 'cagr_tax'])} | "
                     f"{_pct(a.loc['g4', 'cagr'])} → {_pct(a.loc['g4', 'cagr_tax'])} |")
    # --- 자료 ---
    Mk, M2 = MK["zero"]["kospi"], MK["zero"]["k200"]
    L += ["", "## 자료 · 가정", "",
          f"- **지수:** `{KOSPI_CSV}` 의 `{KOSPI_SYM}`({Mk.dates[0].date()} ~ {Mk.dates[-1].date()}, {len(Mk.dates)}일) — 표본 밖·분리 창. "
          f"`{K200_CSV}` 의 `{K200_SYM}`({M2.dates[0].date()} ~ {M2.dates[-1].date()}, {len(M2.dates)}일) — 2017~2026 창. 체결 = 지수 시가(ETF 시가 대용).",
          "- **현금:** (i) 0%. (ii) CD91 연평균 근사 " + " · ".join(f"{y} {r * 100:.2f}%" for y, r in sorted(CD91_MEM.items()))
          + " — **기억값(DB 밖)**, 세대 5 러너 `CD91_MEM` import. 날 금리 = 연 / 252, 현금 조각에 매일.",
          f"- **비용:** 리밸런스 매매 금액 × 편도 {COST_1W * 100:.2f}%(NAV 에서), 주식 조각에 보수 연 {FEE_YR * 100:.2f}%/252 매일. 창 시작은 w 보유(진입 비용 없음).",
          f"- **밴드:** B5 = |비중 − w| > 5%p, B10 > 10%p, B20 = 비중 > 1.2w 또는 < 0.8w(w 0.4 → 32~48%, 0.6 → 48~72%, 0.8 → 64~96%), "
          f"{CUR} = 매년 첫 거래일 + 비중 > w + 10%p(아래 없음). 경계 위 정확히 = 안.",
          f"- **부트스트랩:** 표본 밖 일별 수익을 달 블록(비원형 이동 3개월 · 참고 12개월)으로 재표본, {args.boots}벌(`meta_gen4_run.MonthBoot`·`paired_boot`). "
          "세계마다 하나의 표본 집합을 모든 (규칙, w, 기준) 비교에 공통으로 쓴다.",
          "- **연 1회 기준일:** 그해 거래일 순번(전체 일봉 기준, 0 = 첫 거래일) = k 인 날 시가. 해에 k 번째 날이 없으면 그해는 건너뜀. 창 첫날은 리밸런스 안 함.",
          f"- **원장:** `{SUB}/TRIALS.jsonl` 새 줄 {v['led']}개({N_CELLS} 칸). 산출: `{SUB}/` summary·decision·bootstrap·null_random·null_base_rate·anchors·"
          "weight_matched_r0·harvest·after_tax·nav_{oos,h1,h2,g4}.csv, `equity.png`.", ""]
    if v["fig"]:
        L += ["## 그림", "", f"![자산·상대 자산]({SUB}/equity.png)", "",
              "왼쪽 = 표본 밖 1997-12 ~ 2016-12(코스피), 오른쪽 = 2017-01 ~ 2026-09(코스피200). 맨 위 = 지수 100%(비용 없음)와 R1 의 세 w 자산(로그), "
              "아래 세 줄 = w 별 각 규칙 자산 / R1 자산 − 1. 현금 = CD91 기억값.", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
