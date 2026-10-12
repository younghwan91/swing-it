#!/usr/bin/env python
"""유전 알고리즘(GA) 문턱 최적화 — numba 사전계산 적합도. **탐색이지 판정이 아니다.**

설계: ``research/logs/sepa_explore/GA.md`` (결과 보기 전 커밋 bc29047). 유전자·GA 연산자·적합도·과최적 장치 1~7·판단 규칙을
그대로 옮긴다. 이 러너는 **리포터**다 — 숫자를 ``GA_RESULTS.md`` 에 적고 판단 규칙의 문턱과 산술 비교만 한다.

- 유니버스·가격·청산·대조는 ``RELAX.md`` 와 같다(H5 = 위생 ∧ ADV20 ≥ 5,000, ``krx_session`` 보정판 패널, 미너비니 원전 청산,
  모멘텀 정합 — 날짜 × 시장 × 띠 × 12-1 모멘텀 5분위 × ADV20 2분위 × 직전 5일 수익 3분위, 풀 = 사건 구간 H5 전 종목-일 같은 프로토콜,
  추격 금지는 config 의 피벗 창으로 사후). 층·보정·띠 정의는 ``relax_run`` 을 import 해 그대로 쓴다.
- **사전계산(한 번):** 사건 구간 H5 종목-일 전부를 청산 규칙으로 시뮬(``research.signals.ga_fast.sim_pairs``). 그 쌍들이 곧
  후보 표본이자 대조 풀이다(추세 템플릿은 마스크의 한 조건 — 후보를 미리 좁히지 않는다). 피처: 추세 템플릿·RS·돌파(창 20~60
  종가 > 피벗)·거래량과 창별 평균·PEAD 순위(H5 모집단)·rel·breadth·수급 F·띠·시장·추격 금지 비트(창 × 2/5/10%).
- **대조 고정:** 쌍마다 같은 층 풀(체결 ∧ 그 창 추격 금지 통과)에서 자기 제외 5개 복원 추출 — 난수는 (쌍 번호, 추출 번호) 해시라
  어떤 부분집합을 고르든 그 쌍의 대조가 같다. ``matched_control`` 의 순차 난수와 표본이 달라(기댓값 같음) RELAX 숫자와 잡음만큼 다르다.
- **적합도:** TRAIN(진입 < 2022-01-01) 트레이드(겹침 제거 후, 짝 있는 것) 정합 초과의 부트스트랩 2.5% 하단(500회, 고정 해시 난수),
  거래수 < 150 이면 ``−1 + n/150·1e−3`` (``make_objective`` 모양). OOS 는 적합도에 안 들어간다.
- **GA:** 개체 60 · 세대 30(초기 개체 = 1세대, 평가 60×30 = 1,800/시드) · 토너먼트 3 · 블렌드 교차 BLX-0.5 · 가우시안 변이(범위의 10%,
  유전자별 확률 0.2) · 엘리트 2 · 시드 0~4. 피벗 창은 단계 인덱스 0~4 를 실수로 섞고 평가 때 반올림.
- 원장: ``research/logs/sepa_explore/ga/TRIALS.jsonl`` — **평가한 고유 config 마다 한 줄**(평가 직후, 사람이 결과 보기 전).

비교 기준선(``GA2.md`` 최적화기 절): 같은 6유전자 공간·같은 적합도·같은 예산(1,800/시드)에서 CMA-ES(CMAwM + BIPOP) 5시드도 돌려 나란히 적는다.

Run:  NUMBA_NUM_THREADS=8 uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/ga_run.py
      (``--cache <npz>`` 를 주면 사전계산 배열을 저장/재사용 — DB 적재를 건너뛴다)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from research.signals import ga_fast as gf  # noqa: E402

TRAIN_HI = "2022-01-01"
EVENT_LO, EVENT_HI = "2017-09-15", "2026-04-01"
OUT_DIR = "research/logs/sepa_explore"
LABEL = "ga"
RESULTS_MD = "GA_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/GA.md (bc29047)"
PRIOR_LEDGERS = (("sepa_confluence", "research/logs"), ("sepa_explore", "research/logs"),
                 ("minervini_exits", "research/logs/sepa_explore"), ("convergence", "research/logs/sepa_explore"),
                 ("relax", "research/logs/sepa_explore"))

# --- 설계 수치(GA.md — 탐색하지 않는다) ---
POP, GENS, TOURN, BLX_ALPHA, MUT_SIGMA, MUT_P, ELITE = 60, 30, 3, 0.5, 0.10, 0.2, 2
SEEDS = (0, 1, 2, 3, 4)
N_BOOT_FIT = 500
TRADE_FLOOR = 150
N_PER = 5
CTL_SEED = 0
N_BOOT_CI = 2000
FLOW_LO = "2024-02-19"
LIQ_FLOOR = 20_000.0
WINDOWS = (20, 30, 40, 50, 60)
CHASES = (0.02, 0.05, 0.10, math.inf)
W50 = WINDOWS.index(50)
#: GA1 공간(= RELAX.md BO 공간). 피벗 창은 GA.md 대로 {20,30,40,50,60}.
GA1_GENES = (("rs_min", 40.0, 90.0), ("vol_mult", 0.8, 2.0), ("pead_min", 0.0, 0.9), ("rel_max", 0.6, 3.0),
             ("breadth_min", 0.0, 0.6), ("pivot_idx", 0.0, 4.0))
#: 사전등록 A − F (CONVERGENCE A 에서 수급 F 를 뺀 것) — GA1 공간의 한 점.
BASE_A = {"rs_min": 70.0, "vol_mult": 1.5, "pead_min": 0.70, "rel_max": 1.0, "breadth_min": 0.5, "pivot_window": 50}
#: 미너비니 원전 청산의 청산 키 (stop_cap, take_r, trail, time_stop, max_hold) — trail/time_stop 0 = 없음.
EK_PREREG = (0.10, 2.0, 50, 20, 120)
MARKET_CODE = {"KOSPI": 0, "KOSDAQ": 1}
CTL_DESC = ("모멘텀 정합(RELAX 층): 같은 날 × 시장 × 띠 × 12-1 모멘텀 5분위 × ADV20 2분위 × 직전 5일 수익 3분위, 풀 = 사건 구간 H5 "
            "전 종목-일 같은 청산(체결 ∧ config 피벗 창 추격 금지 통과), 자기 제외, 5개 복원 추출, (쌍, 추출) 해시 난수 seed 0")


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _pct(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def _f(x, fmt: str = "{:.2f}") -> str:
    return "nan" if x is None or not np.isfinite(x) else fmt.format(x)


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_ga_run_helpers.py)
# ---------------------------------------------------------------------------

def ga1_decode(v) -> dict:
    """GA1 실수 유전자 벡터 → config(연속 4자리 반올림, 피벗 창 = 단계 인덱스 반올림)."""
    out = {}
    for (name, lo, hi), x in zip(GA1_GENES, v, strict=True):
        x = float(min(max(x, lo), hi))
        if name == "pivot_idx":
            out["pivot_window"] = WINDOWS[int(round(x))]
        else:
            out[name] = round(x, 4)
    return out


def ga1_key(c: dict) -> tuple:
    return tuple(c[k] for k in ("rs_min", "vol_mult", "pead_min", "rel_max", "breadth_min", "pivot_window"))


def tournament(fit: np.ndarray, rng: np.random.Generator, k: int = TOURN) -> int:
    """토너먼트 선택 — 무작위 k 개체 중 적합도 최대(동률이면 먼저 뽑힌 것)."""
    idx = rng.integers(0, len(fit), size=k)
    return int(idx[np.argmax(fit[idx])])


def blx(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray,
        alpha: float = BLX_ALPHA) -> np.ndarray:
    """블렌드 교차 BLX-α: 유전자마다 [min − αI, max + αI] 균등, 범위로 자름."""
    a, b = np.minimum(p1, p2), np.maximum(p1, p2)
    span = b - a
    child = rng.uniform(a - alpha * span, b + alpha * span)
    return np.clip(child, lo, hi)


def mutate(x: np.ndarray, rng: np.random.Generator, lo: np.ndarray, hi: np.ndarray, *, sigma: float = MUT_SIGMA,
           p: float = MUT_P) -> np.ndarray:
    """가우시안 변이 — 유전자별 확률 p 로 N(0, (σ·범위)²) 를 더하고 범위로 자름."""
    m = rng.random(len(x)) < p
    y = x + m * rng.normal(0.0, sigma * (hi - lo))
    return np.clip(y, lo, hi)


def simple_ga(fitness, genes, *, seed: int, pop: int = POP, gens: int = GENS, elite: int = ELITE):
    """GA.md 의 단순 GA. ``fitness(vec) -> float`` (최대화). 반환 = (평가 이력[(세대, 벡터, 적합도)], 세대별 개체·적합도)."""
    rng = np.random.default_rng(seed)
    lo = np.array([g[1] for g in genes], float)
    hi = np.array([g[2] for g in genes], float)
    P = rng.uniform(lo, hi, size=(pop, len(genes)))
    hist, gens_out = [], []
    for g in range(gens):
        fit = np.array([fitness(x) for x in P])
        hist.extend((g, x.copy(), f) for x, f in zip(P, fit, strict=True))
        gens_out.append((P.copy(), fit.copy()))
        if g == gens - 1:
            break
        order = np.argsort(-fit, kind="stable")
        nxt = [P[i].copy() for i in order[:elite]]
        while len(nxt) < pop:
            a, b = P[tournament(fit, rng)], P[tournament(fit, rng)]
            nxt.append(mutate(blx(a, b, rng, lo, hi), rng, lo, hi))
        P = np.array(nxt)
    return hist, gens_out


def cma_bipop(fitness, n_cont: int, disc_sizes: "tuple[int, ...]", *, budget: int, seed: int, sigma0: float = 0.3,
              stall: bool = False):
    """CMA-ES with Margin(``cmaes.CMAwM``) + BIPOP 재시작. 연속 차원 [0,1], 이산 차원 단계 인덱스 [0, K−1](steps 1).

    ``fitness(x_eval) -> float`` (최대화 — CMA 에는 부호 반전). 예산 = ``ask`` 수(캐시 적중 포함). 재시작 규칙(Hansen 2009):
    첫 런 기본 개체 λ₀ = 4 + ⌊3 ln n⌋; 이후 "작은 체제" 가 쓴 평가 < "큰 체제" 가 쓴 평가면 작은 체제
    (λ = ⌊λ₀ (½ λ_큰/λ₀)^{U²}⌋, σ = σ₀·10^{−2U}), 아니면 큰 체제(λ_큰 ×2, σ₀). 평균은 범위 안 균등 무작위.
    공분산 초기값 = diag(범위²) — 이산 축도 σ₀ × 단계 수만큼 퍼진다.
    재시작 조건 = ``cmaes`` 의 ``should_stop()``; ``stall=True`` 면 그 런의 최고값이 10 + ⌈30n/λ⌉ 세대 동안 나아지지 않을 때도
    (Hansen TolStagnation 류 — ``should_stop`` 만으로는 1,600 평가 안에 재시작이 거의 안 난다; GA1 비교 런은 ``stall=False``).
    반환: (이력[(평가 번호, x_eval, 값)], 재시작 기록[dict — 체제·개체·σ·평가·세대·끝난 이유]).
    """
    from cmaes import CMAwM  # noqa: PLC0415 — 선택 의존성(uv run --with cmaes)

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
            first = False
        elif small_used < large_used:
            u = rng.random()
            regime, lam, sig = "small", max(2, int(lam0 * (0.5 * large_pop / lam0) ** (u ** 2))), sigma0 * 10 ** (-2 * rng.random())
        else:
            large_pop *= 2
            regime, lam, sig = "large", large_pop, sigma0
        opt = CMAwM(mean=mean, sigma=sig, bounds=bounds, steps=steps, seed=int(rng.integers(2 ** 31 - 1)),
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


def ga1_from_unit(x) -> dict:
    """CMA-ES 비교용 GA1 부호화: 연속 5개 [0,1] → 범위, 이산 1개(피벗 창 단계 인덱스 0~4)."""
    v = [lo + float(np.clip(u, 0, 1)) * (hi - lo) for (_, lo, hi), u in zip(GA1_GENES[:5], x[:5], strict=True)]
    return ga1_decode([*v, float(x[5])])


def rank_corr(a, b) -> float:
    """스피어만(유한 쌍만)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    from krx_quant_core.stats.metrics import spearman  # noqa: PLC0415
    return float(spearman(pd.Series(a[ok]), pd.Series(b[ok])))


def gene_spread(cfgs: "list[dict]", genes: "list[tuple[str, float, float]]") -> dict:
    """시드 최적들의 유전자별 (최소, 최대, 범위 대비 폭)."""
    out = {}
    for name, lo, hi in genes:
        v = np.array([c[name] for c in cfgs], float)
        out[name] = (float(v.min()), float(v.max()), float((v.max() - v.min()) / (hi - lo)))
    return out


def write_ledger(path: str, rows: "list[dict]", seen: set) -> int:
    """원장(JSONL, ``krx_quant_core.stats.trials`` 와 같은 줄 형식·지문)에 새 config 만 덧붙인다 → 덧붙인 수."""
    from swing_it.diagnostics.trials import config_fingerprint  # noqa: PLC0415

    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fp = config_fingerprint(r["config"])
            if fp in seen:
                continue
            seen.add(fp)
            row = {"fingerprint": fp, "config": r["config"]}
            if r.get("note"):
                row["note"] = r["note"]
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_fingerprints(path: str) -> set:
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)["fingerprint"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return out


# ---------------------------------------------------------------------------
# 사전계산
# ---------------------------------------------------------------------------

PAIR_KEYS = ("pc", "pd", "clow", "entry_open", "tt", "rs", "breadth", "pead", "rel", "F", "smart", "foreign",
             "persist", "brk_bits", "vol", "vavg", "chase_bits", "nh", "al", "stack", "slope_bits", "band", "market",
             "sid", "adv20", "piv", "eps_yoy", "eps_acc", "rev_yoy", "eps_ni")
BOOK_KEYS = ("O", "H", "L", "C", "TR", "last_present", "MA10", "MA20", "MA50", "cost_tab")


class Pre:
    """사전계산 배열 묶음(쌍 = 사건 구간 H5 종목-일, 신호일 → 종목 순). ``days``·``codes`` 와 ``info`` 도."""

    def __init__(self, arrays: dict, days: pd.DatetimeIndex, codes: pd.Index, info: dict):
        self.__dict__.update(arrays)
        self.days, self.codes, self.info = days, codes, info
        self.P = len(self.pc)
        self.n_days, self.n_codes = len(days), len(codes)
        self.day_start = np.searchsorted(self.pd, np.arange(self.n_days + 1)).astype(np.int64)
        per = days.to_period("M")
        self.month_id = (per.year * 12 + per.month - (per[0].year * 12 + per[0].month)).to_numpy(np.int64)
        self.months = pd.period_range(per[0], per[-1], freq="M")

    def save(self, path: str) -> None:
        arrs = {k: getattr(self, k) for k in PAIR_KEYS + BOOK_KEYS}
        np.savez(path, days=self.days.to_numpy(), codes=np.asarray(self.codes, dtype=str),
                 info=json.dumps(self.info, default=str), **arrs)

    @classmethod
    def load(cls, path: str) -> "Pre":
        if os.path.isdir(path):                     # .npy 디렉터리 — mmap 으로 여러 프로세스가 같은 물리 메모리를 쓴다
            arrays = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r") for k in PAIR_KEYS + BOOK_KEYS}
            meta = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
            return cls(arrays, pd.DatetimeIndex(pd.to_datetime(meta["days"])), pd.Index(meta["codes"]), meta["info"])
        z = np.load(path, allow_pickle=False)
        arrays = {k: z[k] for k in PAIR_KEYS + BOOK_KEYS}
        return cls(arrays, pd.DatetimeIndex(z["days"]), pd.Index(z["codes"].astype(str)), json.loads(str(z["info"])))

    def save_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for k in PAIR_KEYS + BOOK_KEYS:
            np.save(os.path.join(path, f"{k}.npy"), np.ascontiguousarray(getattr(self, k)))
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"days": [str(d.date()) for d in self.days], "codes": list(map(str, self.codes)), "info": self.info},
                      fh, default=str)

    def day_idx(self, d: str) -> int:
        """``d`` 이상 첫 거래일 행(없으면 n_days)."""
        return int(self.days.searchsorted(pd.Timestamp(d), "left"))

    def entry_window(self, lo: "str | None", hi: "str | None") -> "tuple[int, int, int, int]":
        """진입일 ∈ [lo, hi) 인 쌍 범위 → ``(i0, i1, d_lo, d_hi)`` (신호일 행 = 진입일 행 − 1)."""
        d_lo = max(self.day_idx(lo) - 1, 0) if lo else 0
        d_hi = (self.day_idx(hi) - 1) if hi else self.n_days
        return int(self.day_start[d_lo]), int(self.day_start[max(d_hi, d_lo)]), d_lo, d_hi


def eps_features(con, days: pd.DatetimeIndex, codes: pd.Index, di: np.ndarray, ci: np.ndarray,
                 adj_close: pd.DataFrame, raw_close: pd.DataFrame) -> dict:
    """GA2 실적 유전자 피처(GA2.md 마지막 절) — 쌍별 (EPS YoY, EPS 가속, 매출 YoY, 순이익 대체 표시).

    EPS = 순이익 / 주식 수. 주식 수 = ``shares_outstanding_history`` 의 분기말(그 분기·전년 같은 분기) as-of 값을 **분할 조정**
    (그 날의 조정 종가 / 원 종가 = 조정계수로 나눔 — 분할 전후 주식 수가 같은 척도가 된다). 분기말 주식 수는 그 분기 보고서와 함께
    알려지므로 실적 가용일 기준 시점 안전. 둘 중 하나라도 없으면 그 분기 행은 순이익 YoY 그대로(``eps_ni`` = 1).
    버전·가용일·직전 분기 처리는 ``earnings_leg_panel`` 을 그대로 탄다(순이익 자리에 EPS 를 넣어 부른다) — 가속 = EPS YoY > 직전 분기 EPS YoY.
    """
    from swing_it.features.earnings_leg import accel_flag, earnings_leg_panel  # noqa: PLC0415
    from swing_it.storage import read_earnings  # noqa: PLC0415

    ea = read_earnings(con, all_versions=True, cols=("code", "period", "avail_date", "knowledge_date", "netinc",
                                                     "netinc_prior", "revenue", "revenue_prior"))
    from swing_it.storage import read_shares  # noqa: PLC0415

    sh = read_shares(con, cols=("code", "date", "shares_outstanding"))     # 정문(린트 (k)) — 0 이하는 정문이 뺀다(아래 거르기와 같음)
    sh["date"] = pd.to_datetime(sh["date"]).astype("datetime64[ns]")
    sh = sh[sh["shares_outstanding"] > 0]
    # 분할 조정계수(조정 종가 / 원 종가) — 주식 수 날짜의 직전 거래일 값
    fac = (adj_close / raw_close.reindex_like(adj_close)).ffill()
    ci_s = fac.columns.get_indexer(sh["code"].astype(str))
    di_s = fac.index.searchsorted(sh["date"], "right") - 1
    ok = (ci_s >= 0) & (di_s >= 0)
    f = np.full(len(sh), np.nan)
    f[ok] = fac.to_numpy()[di_s[ok], ci_s[ok]]
    sh["sh_adj"] = sh["shares_outstanding"] / f
    sh = sh[np.isfinite(sh["sh_adj"]) & (sh["sh_adj"] > 0)].sort_values(["date", "code"])
    q = ea["period"].astype(str)
    pe = pd.to_datetime(q.str[:4] + "-" + (q.str[5].astype(int) * 3).astype(str).str.zfill(2) + "-01") + pd.offsets.MonthEnd(0)
    pe = pe.astype("datetime64[ns]")
    ea = ea.assign(code=ea["code"].astype(str), _pe=pe, _pe0=(pe - pd.DateOffset(years=1)).astype("datetime64[ns]"),
                   _row=np.arange(len(ea)))
    out = {}
    for col, key in (("_pe", "now"), ("_pe0", "prev")):
        m = pd.merge_asof(ea[["code", col, "_row"]].sort_values(col).rename(columns={col: "date"}),
                          sh[["code", "date", "sh_adj"]], on="date", by="code", direction="backward")
        out[key] = m.sort_values("_row")["sh_adj"].to_numpy()
    have = np.isfinite(out["now"]) & np.isfinite(out["prev"])
    ea2 = ea[["code", "period", "avail_date", "knowledge_date", "revenue", "revenue_prior"]].copy()
    ea2["netinc"] = np.where(have, ea["netinc"].to_numpy(float) / out["now"], ea["netinc"].to_numpy(float))
    ea2["netinc_prior"] = np.where(have, ea["netinc_prior"].to_numpy(float) / out["prev"], ea["netinc_prior"].to_numpy(float))
    ea2["op_income"] = np.where(have, 2.0, 1.0)          # op_yoy = 1 → EPS 기준, 0 → 순이익 대체(표시용 자리)
    ea2["op_income_prior"] = 1.0
    panel = earnings_leg_panel(ea2, [d.strftime("%Y-%m-%d") for d in days])
    pdi = days.get_indexer(pd.to_datetime(panel["date"]))
    pci = codes.get_indexer(panel["code"])
    okp = (pdi >= 0) & (pci >= 0)
    res = {}
    for name, vals, fill in (("eps_yoy", panel["yoy"].to_numpy(float), np.nan),
                             ("eps_acc", accel_flag(panel).to_numpy(float), 0.0),
                             ("rev_yoy", panel["rev_yoy"].to_numpy(float), np.nan),
                             ("eps_ni", (panel["op_yoy"].to_numpy(float) == 0).astype(float), 1.0)):
        grid = np.full((len(days), len(codes)), np.nan)
        grid[pdi[okp], pci[okp]] = vals[okp]
        v = grid[di, ci]
        res[name] = np.where(np.isfinite(v), v, fill)
    res["eps_acc"] = res["eps_acc"].astype(np.int8)
    res["eps_ni"] = res["eps_ni"].astype(np.int8)
    res["info"] = {"share_rows_pit": float(have.mean()),
                   "pairs_with_earn": float(np.isfinite(res["eps_yoy"]).mean()),
                   "pairs_eps_from_shares": float((np.isfinite(res["eps_yoy"]) & (res["eps_ni"] == 0)).mean())}
    return res


def add_eps(cache_dir: str) -> dict:
    """기존 .npy 사전계산 디렉터리에 실적 유전자 피처 4개를 덧붙인다(가격·피처는 그대로 — 쌍 순서 같음)."""
    from prop_swing_common import load_env_db  # noqa: PLC0415
    from relax_run import KrxPatch  # noqa: PLC0415

    from swing_it.storage import connect, db_default, read_prices  # noqa: PLC0415

    meta = json.load(open(os.path.join(cache_dir, "meta.json"), encoding="utf-8"))
    days = pd.DatetimeIndex(pd.to_datetime(meta["days"]))
    codes = pd.Index(meta["codes"])
    pc, pdd = np.load(os.path.join(cache_dir, "pc.npy")), np.load(os.path.join(cache_dir, "pd.npy"))
    load_env_db()
    con = connect(db_default())
    patch = KrxPatch(con)

    def panel(table):
        df = patch.apply(read_prices(con, session="raw", cols=("code", "date", "open", "close"), table=table), table)
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot(index="date", columns="code", values="close").astype(float).reindex(index=days, columns=codes)

    res = eps_features(con, days, codes, pdd, pc, panel("daily_bars_adjusted"), panel("daily_bars"))
    for k in ("eps_yoy", "eps_acc", "rev_yoy", "eps_ni"):
        np.save(os.path.join(cache_dir, f"{k}.npy"), res[k])
    meta["info"]["eps"] = res["info"]
    json.dump(meta, open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8"), default=str)
    return res["info"]


def build_pre(t0: float) -> Pre:
    """DB → 사전계산(``relax_run`` 과 같은 보정·유니버스·층). 오래 걸린다(수 분)."""
    from convergence_run import PEAD_POP_FLOOR, ADV_HI, ADV_LO, raw_close_panel, relative_contraction, repoint_pead
    from prop_swing_common import load_env_db
    from relax_run import MOM_Q, RET5_Q, KrxPatch, band_strata, stratum_id
    from sepa_confluence_gate import Data
    from sepa_momentum import momentum_12_1

    from research.signals.minervini_exits import contraction_low_level
    from research.signals.sepa import _Book
    from swing_it.features.korea import cost_rt, universe_mask
    from swing_it.features.minervini import pivot_level
    from swing_it.features.technical import moving_average
    from swing_it.storage import connect, db_default

    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    load_env_db()
    con = connect(db_default())
    patch = KrxPatch(con)
    with patch.installed():
        D = Data(con, t0)
        raw_close = raw_close_panel(con, D.p)
    p = D.p
    assert p.close.columns.is_monotonic_increasing
    info = {"nxt": [{k: v for k, v in s.items()} for s in patch.stats]}
    _log("NXT 보정: " + " · ".join(f"{s['table']} 늦은 행 {s['rows_late']:,} 보정 {s['patched']:,} 미보정 {s['unpatched']:,}"
                                    for s in patch.stats), t0)
    h5 = universe_mask(p.close, p.trade_value, D.meta, raw_close=raw_close, adv_floor=PEAD_POP_FLOOR)
    u_s = universe_mask(p.close, p.trade_value, D.meta, raw_close=raw_close, adv_floor=ADV_LO, adv_cap=ADV_HI)
    main_u = D.uni.astype(bool)
    assert bool(((u_s | main_u) == h5).to_numpy().all()) and not bool((u_s & main_u).to_numpy().any())
    band = pd.DataFrame(np.where(main_u.to_numpy(), 1.0, np.where(u_s.to_numpy(), 0.0, np.nan)),
                        index=p.days, columns=p.close.columns)
    repoint_pead(D, h5)
    bands = {1: main_u, 0: u_s}
    mom = momentum_12_1(p.close)
    strata = {"q_mom5": band_strata(mom, bands, p.market, n_q=MOM_Q),
              "adv2": band_strata(D.adv20, bands, p.market, n_q=None),
              "q_ret5_3": band_strata(D.ret5, bands, p.market, n_q=RET5_Q)}
    del mom
    _, rel = relative_contraction(p.high, p.low)
    c_low = contraction_low_level(p.low, 10)
    _log("H5·띠·층·rel·수축 저점", t0)
    dmask0 = np.asarray((p.days >= pd.Timestamp(EVENT_LO)) & (p.days <= pd.Timestamp(EVENT_HI)))
    di0, ci0 = np.nonzero(h5.to_numpy(bool) & dmask0[:, None])
    eps = eps_features(con, p.days, p.close.columns, di0, ci0, p.close, raw_close)
    del di0, ci0

    dmask = np.asarray((p.days >= pd.Timestamp(EVENT_LO)) & (p.days <= pd.Timestamp(EVENT_HI)))
    di, ci = np.nonzero(h5.to_numpy(bool) & dmask[:, None])          # 행 우선 → 신호일 → 종목 순
    A: dict = {"pc": ci.astype(np.int64), "pd": di.astype(np.int64)}

    def pick(panel, dtype=np.float64):
        return np.ascontiguousarray(panel.to_numpy(dtype=float)[di, ci].astype(dtype))

    A["clow"] = pick(c_low)
    for k in ("eps_yoy", "eps_acc", "rev_yoy", "eps_ni"):
        A[k] = eps[k]
    info["eps"] = eps["info"]
    A["tt"] = D.feats["tt"].to_numpy(bool)[di, ci]
    A["rs"] = pick(D.feats["rs"])
    A["breadth"] = pick(D.feats["breadth"])
    A["rel"] = pick(rel)
    A["F"] = pick(D.flow["F"].astype("float64"), np.float32)
    A["smart"] = pick(D.flow["smart_ratio_20"], np.float32)
    A["foreign"] = pick(D.flow["foreign_ratio_20"], np.float32)
    A["persist"] = pick(D.flow["smart_pos_days_20"].astype("float64"), np.float32)
    A["adv20"] = pick(D.adv20)
    ep = D.earn
    pr = np.full(p.close.shape, np.nan)
    edi = p.days.get_indexer(pd.to_datetime(ep["date"]))
    eci = p.close.columns.get_indexer(ep["code"])
    ok = (edi >= 0) & (eci >= 0)
    pr[edi[ok], eci[ok]] = ep["pead_rank"].to_numpy(float)[ok]
    A["pead"] = np.ascontiguousarray(pr[di, ci])
    del pr
    A["band"] = pick(band).astype(np.int8)
    mk_s = p.market.reindex(p.close.columns[ci])
    A["market"] = mk_s.map(MARKET_CODE).fillna(-1).to_numpy(np.int8)
    A["sid"] = stratum_id(di, mk_s.to_numpy(), A["band"].astype(float), pick(strata["q_mom5"]), pick(strata["adv2"]),
                          pick(strata["q_ret5_3"])).astype(np.int64)
    del strata, rel, c_low
    _log(f"쌍 {len(di):,} — 기본 피처", t0)

    # 체결가(진입 시가) — 보정판 패널
    book = _Book(p.open, p.high, p.low, p.close, D.feats["ma50"], p.volume)
    n_days = len(p.days)
    nxt = np.minimum(di + 1, n_days - 1)
    A["entry_open"] = np.where(di + 1 < n_days, book.O[ci, nxt], np.nan)

    close = p.close.to_numpy(float)[di, ci]
    vol = p.volume.to_numpy(float)
    A["vol"] = np.ascontiguousarray(vol[di, ci])
    brk = np.zeros(len(di), np.int8)
    chase_bits = np.zeros(len(di), np.int16)
    vavg = np.empty((len(di), len(WINDOWS)))
    pivs = np.empty((len(di), len(WINDOWS)))
    for wi, w in enumerate(WINDOWS):
        piv = pivot_level(p.high, w).to_numpy(float)[di, ci]
        pivs[:, wi] = piv
        with np.errstate(invalid="ignore"):
            brk |= ((close > piv).astype(np.int8) << wi).astype(np.int8)
        vavg[:, wi] = p.volume.shift(1).rolling(w, min_periods=w).mean().to_numpy(float)[di, ci]
        for k, c in enumerate(CHASES[:3]):
            okc = gf.chase_ok(A["entry_open"], piv, c)
            chase_bits |= (okc.astype(np.int16) << (wi * 3 + k)).astype(np.int16)
    A["brk_bits"], A["chase_bits"], A["vavg"] = brk, chase_bits, np.ascontiguousarray(vavg)
    A["piv"] = np.ascontiguousarray(pivs)      # 창별 피벗(돌파 실패 청산 기준)
    # 뼈대 돌파(창 50, 1.5×)와 성분 분해가 같은지 — pivot_breakout 그대로
    ref = D.feats["breakout"].to_numpy(bool)[di, ci]
    with np.errstate(invalid="ignore"):
        me = ((brk >> W50) & 1).astype(bool) & (vavg[:, W50] > 0) & (A["vol"] >= 1.5 * vavg[:, W50])
    info["breakout_mismatch"] = int((ref != me).sum())
    _log(f"돌파·추격 비트 (뼈대 돌파 대비 불일치 {info['breakout_mismatch']})", t0)

    # GA2 추세 성분
    hi252 = p.high.rolling(252, min_periods=252).max().to_numpy(float)[di, ci]
    lo252 = p.low.rolling(252, min_periods=252).min().to_numpy(float)[di, ci]
    with np.errstate(invalid="ignore", divide="ignore"):
        A["nh"] = (close / hi252).astype(np.float32)
        A["al"] = (close / lo252).astype(np.float32)
    ma50 = D.feats["ma50"].to_numpy(float)[di, ci]
    m150p = moving_average(p.close, 150)
    m200p = moving_average(p.close, 200)
    ma150 = m150p.to_numpy(float)[di, ci]
    ma200 = m200p.to_numpy(float)[di, ci]
    with np.errstate(invalid="ignore"):
        A["stack"] = ((close > ma50).astype(np.int8) + (ma50 > ma150).astype(np.int8)
                      + (ma150 > ma200).astype(np.int8)).astype(np.int8)
        sb = np.zeros(len(di), np.int8)
        for k, lag in enumerate((10, 20, 40)):
            prev = m200p.shift(lag).to_numpy(float)[di, ci]
            sb |= ((ma200 > prev).astype(np.int8) << k).astype(np.int8)
    A["slope_bits"] = sb
    del m150p, m200p
    _log("추세 성분(GA2)", t0)

    B = {"O": book.O, "H": book.H, "L": book.L, "C": book.C, "TR": book.tradable,
         "last_present": np.ascontiguousarray(book.present[:, -1]), "MA50": book.MA}
    for w in (10, 20):
        B[f"MA{w}"] = np.ascontiguousarray(moving_average(p.close, w).to_numpy(float).T)
    cost_tab = np.empty((n_days, 2))
    for j, d in enumerate(p.days):
        cost_tab[j, 0] = cost_rt(d, "KOSPI")
        cost_tab[j, 1] = cost_rt(d, "KOSDAQ")
    B["cost_tab"] = cost_tab
    info.update({"h5_days": int(h5.to_numpy().sum()), "pairs": int(len(di)), "panel": [str(p.days[0].date()),
                 str(p.days[-1].date())], "n_codes": int(len(p.close.columns))})
    days, codes = p.days, p.close.columns
    del D, p, book, h5, u_s, main_u, band
    return Pre({**A, **B}, days, codes, info)


# ---------------------------------------------------------------------------
# 엔진 — 청산 캐시·대조 캐시·마스크·겹침 제거·적합도
# ---------------------------------------------------------------------------

class Engine:
    """사전계산 위의 평가기.

    - 청산 조합 ``ek`` = (손절 상한, 익절 배수, 추적 MA(0 없음), 시간 손절(0 없음), 최대 보유, 돌파 실패(0/1), 돌파 실패 피벗 창(−1 = 끔))
      별 시뮬을 **사건 연도 덩어리 단위로 필요할 때만** 계산해 LRU 캐시(창이 닿는 해만 시뮬한다).
    - 체결 상태는 청산 규칙과 무관 → 한 번. 대조 추출(쌍 번호)은 체결 ∧ 추격(피벗 창, 추격 단계)에만 달려 있어 (창, 추격) 별로 한 번.
      대조의 수익은 평가 때 그 청산 조합 배열에서 읽는다(같은 날 = 같은 덩어리).
    """

    def __init__(self, pre: Pre, *, max_exit: int = 32, max_ctl: int = 0):
        self.p = pre
        self.max_exit = max_exit
        self.exits: OrderedDict = OrderedDict()
        self.ctls: dict = {}
        self.stats = {"exit_sims": 0, "exit_chunks": 0, "exit_sec": 0.0, "ctl_draws": 0, "ctl_sec": 0.0}
        self.sorder = None
        self._cuts: dict = {}
        yrs = pre.days.year.to_numpy()[pre.pd]
        self.chunk_b = np.r_[0, np.flatnonzero(np.diff(yrs)) + 1, pre.P].astype(np.int64)
        st = gf.sim_book(pre.O, pre.H, pre.L, pre.C, pre.MA50, pre.TR, pre.last_present, pre.pc, pre.pd, pre.clow,
                         gf.EXIT_PREREG)[0]
        self.status = st
        self.orderable = (st != gf.ST_NO_NEXT) & (st != gf.ST_NO_CONTRACTION)

    @staticmethod
    def norm_ek(ek: tuple) -> tuple:
        return tuple(ek) + (0, -1) if len(ek) == 5 else tuple(ek)

    # --- 청산 시뮬 ---
    def exit(self, ek: tuple, i0: int = 0, i1: "int | None" = None) -> dict:
        ek = self.norm_ek(ek)
        p = self.p
        i1 = p.P if i1 is None else i1
        if ek in self.exits:
            self.exits.move_to_end(ek)
            e = self.exits[ek]
        else:
            e = {"x": np.full(p.P, -1, np.int32), "ret": np.full(p.P, np.nan), "r2": np.full(p.P, np.nan, np.float32),
                 "pab": np.full(p.P, -1, np.int32), "reason": np.full(p.P, -1, np.int8),
                 "done": np.zeros(len(self.chunk_b) - 1, np.bool_)}
            self.exits[ek] = e
            while len(self.exits) > self.max_exit:
                self.exits.popitem(last=False)
        if i1 <= i0:
            return e
        c0 = int(np.searchsorted(self.chunk_b, i0, "right") - 1)
        c1 = int(np.searchsorted(self.chunk_b, i1 - 1, "right") - 1)
        todo = [c for c in range(c0, c1 + 1) if not e["done"][c]]
        if todo:
            stop_cap, take_r, trail, ts, max_hold, fb, w_fb = ek
            ma = {10: p.MA10, 20: p.MA20, 50: p.MA50}.get(trail, p.MA50)
            params = {**gf.EXIT_PREREG, "stop_cap": stop_cap, "take_r": take_r, "ts_days": ts or None,
                      "max_hold": max_hold, "has_trail": trail != 0, "fb": bool(fb)}
            t = time.time()
            a, b = int(self.chunk_b[todo[0]]), int(self.chunk_b[todo[-1] + 1])   # 연속 범위로 한 번에
            piv = np.ascontiguousarray(p.piv[a:b, w_fb]) if fb else None
            st, x, ret, r1, r2, dd, pab, rs = gf.sim_book(p.O, p.H, p.L, p.C, ma, p.TR, p.last_present, p.pc[a:b],
                                                          p.pd[a:b], p.clow[a:b], params, pivot=piv)
            e["x"][a:b] = x
            e["ret"][a:b] = ret
            e["r2"][a:b] = r2
            e["pab"][a:b] = pab
            e["reason"][a:b] = rs
            e["done"][todo[0]:todo[-1] + 1] = True
            self.stats["exit_sims"] += 1
            self.stats["exit_chunks"] += len(todo)
            self.stats["exit_sec"] += time.time() - t
        return e

    def fill_ctl(self, ek: tuple, w_idx: int, c_idx: int) -> "tuple[np.ndarray, np.ndarray]":
        """(체결 ∧ 추격 통과) bool, 대조 쌍 번호(P × 5, 없으면 −1) — 피벗 창 ``w_idx``·추격 ``c_idx`` (3 = 없음). 청산 조합과 무관."""
        key = (w_idx, c_idx)
        if key in self.ctls:
            return self.ctls[key]
        p = self.p
        fill = self.status == gf.ST_OK
        if c_idx < 3:
            fill = fill & (((p.chase_bits >> (w_idx * 3 + c_idx)) & 1) == 1)
        t = time.time()
        if self.sorder is None:
            self.sorder = gf.stratum_order(p.sid)
        members, ms = gf.stratum_index(p.sid, fill, self.sorder)
        cidx = gf.draw_idx(p.sid, members, ms, N_PER, CTL_SEED)
        self.stats["ctl_draws"] += 1
        self.stats["ctl_sec"] += time.time() - t
        self.ctls[key] = (fill, cidx)
        return fill, cidx

    def cutv(self, cut: "int | None") -> np.ndarray:
        """모든 쌍에 같은 절단 행(마크투마켓은 그 직전 거래일 종가). ``None`` = 절단 없음."""
        c = self.p.n_days + 1 if cut is None else int(cut)
        if c not in self._cuts:
            self._cuts[c] = np.full(self.p.P, c, np.int32)
        return self._cuts[c]

    # --- 마스크 ---
    def mask(self, spec: dict, i0: int, i1: int, band_sel: int = -1, extra: "np.ndarray | None" = None) -> np.ndarray:
        """config 사양 → [i0, i1) 진입 조건 bool(체결 미포함)."""
        p = self.p
        ex = extra if extra is not None else _ones(p.P)
        if spec["kind"] == "ga1":
            return gf.mask_ga1(i0, i1, p.tt, p.rs, p.brk_bits, p.vol, p.vavg, p.pead, p.rel, p.breadth, p.band,
                               float(spec["rs_min"]), int(spec["w_idx"]), float(spec["vol_mult"]),
                               float(spec["pead_min"]), float(spec["rel_max"]), float(spec["breadth_min"]),
                               int(band_sel), ex)
        return gf.mask_ga2(i0, i1, p.nh, p.al, p.stack, p.slope_bits, p.rs, p.brk_bits, p.vol, p.vavg, p.pead, p.rel,
                           p.breadth, p.band, spec["g"], int(band_sel), ex, p.eps_yoy, p.eps_acc, p.rev_yoy)

    def select(self, spec: dict, i0: int, i1: int, band_sel: int = -1, extra=None) -> np.ndarray:
        """겹침 제거 트레이드(쌍 번호). 체결 = 시뮬 체결 ∧ 그 config 추격 통과."""
        e = self.exit(spec["ek"], i0, i1)
        fill, _ = self.fill_ctl(spec["ek"], spec["w_idx"], spec["c_idx"])
        ok = np.zeros(self.p.P, np.bool_)
        ok[i0:i1] = self.mask(spec, i0, i1, band_sel, extra) & fill[i0:i1]
        return gf.greedy(ok, self.p.pc, self.p.pd, e["x"], self.p.n_codes, i0, i1)

    def excess_full(self, spec: dict, sel: np.ndarray, cut: "int | None" = None, cutv=None) -> dict:
        """선택 트레이드의 (끊은) 수익·정합 초과(짝 없으면 NaN)·보유일·비용·청산 행. ``cut`` 행 직전 종가 마크투마켓."""
        p = self.p
        sel = np.ascontiguousarray(sel, np.int64)
        e = self.exit(spec["ek"], int(sel.min()) if len(sel) else 0, int(sel.max()) + 1 if len(sel) else 0)
        _, cidx = self.fill_ctl(spec["ek"], spec["w_idx"], spec["c_idx"])
        cv = self.cutv(cut) if cutv is None else cutv
        ex, hold, cost, rr, xx = gf.trade_excess(np.ascontiguousarray(sel, np.int64), cv, p.pc, p.pd, e["x"], e["pab"],
                                                 e["ret"], e["r2"], p.O, p.C, cidx, p.cost_tab, p.market)
        return {"ex": ex, "hold": hold, "cost": cost, "ret": rr, "x": xx}

    def excess(self, spec: dict, sel: np.ndarray, cut: "int | None" = None) -> "tuple[np.ndarray, np.ndarray]":
        r = self.excess_full(spec, sel, cut)
        return r["ex"], r["hold"]

    # --- 적합도 ---
    def fit_trade(self, spec, i0, i1, cut: "int | None" = None, *, per_day: bool = False, floor: int = TRADE_FLOOR,
                  seed: int = 0, extra=None) -> "tuple[float, int]":
        """트레이드당(또는 Σ초과/Σ보유일) 정합 초과 부트스트랩 2.5% 하단. 학습창 끝(``cut``)을 넘는 트레이드·대조는 그 직전 종가에서 끊는다."""
        sel = self.select(spec, i0, i1, extra=extra)
        ex, hold = self.excess(spec, sel, cut)
        ok = np.isfinite(ex)
        n = int(ok.sum())
        if n < floor:
            return gf.floor_penalty(n, floor), n
        if per_day:
            return gf.boot_lower_ratio(ex[ok], hold[ok], n_boot=N_BOOT_FIT, seed=seed), n
        return gf.boot_lower(ex[ok], n_boot=N_BOOT_FIT, seed=seed), n

    def portfolio(self, spec, i0: int, i1: int, t0: int, t1: int, cut: "int | None" = None, band_sel: int = -1,
                  extra=None, composite: "dict | None" = None):
        """자본 제약 포트폴리오 → ``(daily[t0:t1], opened, n_orders, n_unmatched)``. ``composite`` = 폴드별 config 를 이어 붙인 배열."""
        p = self.p
        if composite is None:
            e = self.exit(spec["ek"], i0, i1)
            fill_full, cidx = self.fill_ctl(spec["ek"], spec["w_idx"], spec["c_idx"])
            sig = np.zeros(p.P, np.bool_)
            sig[i0:i1] = self.mask(spec, i0, i1, band_sel, extra)
            if cut is not None:                     # 진입 행 ≥ 절단 행은 창 밖(리뷰 A-M1)
                sig[i0:i1] &= p.pd[i0:i1] + 1 < cut
            arr = {"sig": sig, "fill": fill_full, "x": e["x"], "pab": e["pab"], "ret": e["ret"], "r2": e["r2"],
                   "cidx": cidx, "cut": self.cutv(cut)}
        else:
            arr = composite
        return gf.portfolio(arr["sig"], arr["fill"], self.orderable, p.pc, p.pd, p.day_start, t0, t1, p.rel, p.rs,
                            arr["x"], arr["pab"], arr["ret"], arr["r2"], p.O, p.C, arr["cidx"], p.cost_tab, p.market,
                            arr["cut"], gf.SLOTS, gf.SLOT_W)

    def monthly(self, daily: np.ndarray, t0: int, d_from: int, d_to: int) -> np.ndarray:
        """날별 → 월별 합(행 [d_from, d_to) 만)."""
        m = self.p.month_id[d_from:d_to]
        v = daily[d_from - t0:d_to - t0]
        return np.bincount(m - m[0], weights=v, minlength=m[-1] - m[0] + 1) if len(m) else np.zeros(0)


_ONES: dict = {}


def _ones(n: int) -> np.ndarray:
    if n not in _ONES:
        _ONES[n] = np.ones(n, np.bool_)
    return _ONES[n]


def ga1_spec(c: dict) -> dict:
    """GA1 config → 엔진 사양(청산 = 원전 고정, 추격 5%·피벗 창 = 유전자)."""
    w = WINDOWS.index(int(c["pivot_window"]))
    return {"kind": "ga1", "ek": EK_PREREG, "w_idx": w, "c_idx": 1, "rs_min": c["rs_min"], "vol_mult": c["vol_mult"],
            "pead_min": c["pead_min"], "rel_max": c["rel_max"], "breadth_min": c["breadth_min"]}


# ---------------------------------------------------------------------------
# 보고용 숫자
# ---------------------------------------------------------------------------

def cluster_mean_ci(x: np.ndarray, clusters: np.ndarray, *, n_boot: int = N_BOOT_CI, seed: int = 0):
    from krx_quant_core.stats.matched_null import cluster_bootstrap_ci  # noqa: PLC0415
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    lo, hi, _ = cluster_bootstrap_ci(x, clusters, n_boot=n_boot, seed=seed)
    return float(np.mean(x)), lo, hi


def ratio_ci(num, den, clusters, *, n_boot: int = N_BOOT_CI, seed: int = 0):
    num, den, clusters = np.asarray(num, float), np.asarray(den, float), np.asarray(clusters)
    if len(num) == 0 or den.sum() <= 0:
        return np.nan, np.nan, np.nan
    point = float(num.sum() / den.sum())
    uniq, inv = np.unique(clusters, return_inverse=True)
    if len(uniq) < 2:
        return point, np.nan, np.nan
    s = np.bincount(inv, weights=num, minlength=len(uniq))
    n = np.bincount(inv, weights=den, minlength=len(uniq))
    draw = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = s[draw].sum(1) / n[draw].sum(1)
    return point, float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def report_from(r: dict, reason: np.ndarray, pab: np.ndarray, clusters: np.ndarray, n_dsr: "int | None" = None) -> dict:
    """``Engine.excess_full`` 결과 → 서술 숫자. 손절 비율은 끊기지 않은 트레이드의 마지막 사유 = stop 비율(끊긴 것은 손절 아님).
    DSR 은 **정합 초과 − 비용**(짝 있는 트레이드) 표본으로."""
    n = len(r["ret"])
    if n == 0:
        return {"n": 0}
    ex, hold, cost, ret = r["ex"], r["hold"], r["cost"], r["ret"]
    net = ret - cost
    ok = np.isfinite(ex)
    a, alo, ahi = cluster_mean_ci(ex[ok], clusters[ok])
    dpt, dlo, dhi = ratio_ci(ex[ok], hold[ok], clusters[ok])
    g, lss = net[net > 0], net[net < 0]
    s = {"n": int(n), "n_matched": int(ok.sum()), "win": float((net > 0).mean()),
         "stop_rate": float((reason == gf.R_STOP).mean()),
         "avg_gain": float(g.mean()) if len(g) else np.nan, "avg_loss": float(lss.mean()) if len(lss) else np.nan,
         "gross": float(ret.mean()), "net": float(net.mean()), "net2": float((ret - 2 * cost).mean()),
         "alpha": a, "alpha_lo": alo, "alpha_hi": ahi, "alpha_day": dpt, "alpha_day_lo": dlo, "alpha_day_hi": dhi,
         "hold_med": float(np.median(hold)), "partial": float(((pab >= 0) & (pab <= r["x"])).mean())}
    s["gl"] = s["avg_gain"] / abs(s["avg_loss"]) if np.isfinite(s["avg_gain"]) and np.isfinite(s["avg_loss"]) else np.nan
    if n_dsr:
        from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample  # noqa: PLC0415
        v = (ex - cost)[ok]
        d = deflated_sharpe_from_sample(v, n_dsr) if len(v) > 2 else None
        s["dsr"] = float(d["prob_deflated_sharpe"]) if d else np.nan
        s["sr"] = float(d["observed_sharpe"]) if d else np.nan
    return s


def trade_report(E: Engine, spec: dict, sel: np.ndarray, n_dsr: "int | None" = None, cut: "int | None" = None) -> dict:
    """선택 트레이드의 서술 숫자 — n·승률·손절 비율·이익/손실·순평균(1×·2×)·정합 α(트레이드당·보유 1일당, 날짜군집 CI)."""
    if cut is not None:                             # 진입 행 ≥ 절단 행은 창 밖(시가평가 우회 — 리뷰 A-M1)
        sel = np.asarray(sel)[E.p.pd[np.asarray(sel, np.int64)] + 1 < cut]
    if len(sel) == 0:
        return {"n": 0}
    e = E.exit(spec["ek"], int(np.min(sel)), int(np.max(sel)) + 1)
    r = E.excess_full(spec, sel, cut)
    reason = np.where(r["x"] < e["x"][sel], -2, e["reason"][sel])
    return report_from(r, reason, e["pab"][sel], E.p.pd[sel], n_dsr)


def monthly_excess_col(E: Engine, spec: dict, sel: np.ndarray) -> np.ndarray:
    """PBO 열 — 진입월별 정합 초과 평균(짝 있는 것), 거래 없는 달 0. 길이 = 전 패널 월 수."""
    p = E.p
    ex, _ = E.excess(spec, sel)
    ok = np.isfinite(ex)
    m = p.month_id[p.pd[sel][ok] + 1]
    out = np.zeros(len(p.months))
    if ok.any():
        s = np.bincount(m, weights=ex[ok], minlength=len(out))
        c = np.bincount(m, minlength=len(out))
        out = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return out


def tstr(s: dict) -> str:
    if not s or not s.get("n"):
        return "n 0"
    return (f"n {s['n']} · 손절 {s['stop_rate']:.0%} · 이익/손실 {_f(s['gl'])} · 순 {_pct(s['net'])} · "
            f"α/트레이드 {_pct(s['alpha'])} [{_pct(s['alpha_lo'])}, {_pct(s['alpha_hi'])}] · "
            f"α/보유일 {_f(s['alpha_day'] * 1e4, '{:+.1f}')}bp [{_f(s['alpha_day_lo'] * 1e4, '{:+.1f}')}, "
            f"{_f(s['alpha_day_hi'] * 1e4, '{:+.1f}')}]")


def trow(s: dict) -> str:
    """표 한 행의 숫자 칸들."""
    if not s or not s.get("n"):
        return "0 | | | | | | | |"
    return (f"{s['n']} | {s['win']:.0%} | {s['stop_rate']:.0%} | {_f(s['gl'])} | {_pct(s['net'])} | {_pct(s['net2'])} | "
            f"{_pct(s['alpha'])} [{_pct(s['alpha_lo'])}, {_pct(s['alpha_hi'])}] | "
            f"{_f(s['alpha_day'] * 1e4, '{:+.1f}')}bp [{_f(s['alpha_day_lo'] * 1e4, '{:+.1f}')}, "
            f"{_f(s['alpha_day_hi'] * 1e4, '{:+.1f}')}] | {s['hold_med']:.0f}")


T_HEAD = ("| n | 승률(순) | 손절 비율 | 이익/손실 | 순평균 1× | 순평균 2× | α/트레이드 [95% CI] | α/보유일 [95% CI] | 보유 중앙값 |")


# ---------------------------------------------------------------------------
# 실데이터 패리티 (엔진 = simulate_minervini, 트레이드 단위)
# ---------------------------------------------------------------------------

def parity_real(E: Engine, specs: "list[tuple[str, dict, dict]]", t0: float) -> list[dict]:
    """config 마다 엔진 트레이드(전 구간) vs ``simulate_minervini``(같은 사건 = 마스크 통과 쌍, 같은 패널). 청산 조합 포함."""
    from research.signals.minervini_exits import simulate_minervini  # noqa: PLC0415

    p = E.p
    Opn, H, L, C, V = (pd.DataFrame(a.T, index=p.days, columns=p.codes) for a in (p.O, p.H, p.L, p.C, np.where(p.TR, 1.0, 0.0)))
    out = []
    for name, spec, prm in specs:
        sel = E.select(spec, 0, p.P)
        m = E.mask(spec, 0, p.P)
        ev_i = np.flatnonzero(m)
        w = WINDOWS[spec["w_idx"]]
        from swing_it.features.minervini import pivot_level  # noqa: PLC0415
        piv = pivot_level(H, w).to_numpy()  # 보정판 고가 — 사건일 ≤ 2026-04-01 이라 원판과 같다
        ev = pd.DataFrame({"code": p.codes[p.pc[ev_i]], "event_day": p.days[p.pd[ev_i]],
                           "pivot": piv[p.pd[ev_i], p.pc[ev_i]], "contraction_low": p.clow[ev_i]})
        trail = prm.get("trail", 50)
        ma = pd.DataFrame(({10: p.MA10, 20: p.MA20, 50: p.MA50}.get(trail, p.MA50) if trail else
                           np.full(p.C.shape, np.nan)).T, index=p.days, columns=p.codes)
        tt = time.time()
        ref = simulate_minervini(Opn, H, L, C, ma, ev, volume=V, stop_cap=prm.get("stop_cap", 0.10),
                                 take_r=prm.get("take_r", 2.0), time_stop_days=prm.get("ts", 20) or None,
                                 max_hold=prm.get("max_hold", 120), chase=CHASES[spec["c_idx"]] if spec["c_idx"] < 3 else None,
                                 limit_up_skip=True)
        t_ref = time.time() - tt
        e = E.exit(spec["ek"])
        eng = pd.DataFrame({"code": p.codes[p.pc[sel]], "event_day": p.days[p.pd[sel]], "exit_day": p.days[e["x"][sel]],
                            "ret": e["ret"][sel], "reason": [gf.REASON_NAMES[k] for k in e["reason"][sel]]})
        a = eng.sort_values(["event_day", "code"]).reset_index(drop=True)
        b = ref.sort_values(["event_day", "code"]).reset_index(drop=True)
        same_n = len(a) == len(b)
        r = {"name": name, "events": len(ev), "n_engine": len(a), "n_ref": len(b), "ref_sec": t_ref}
        if same_n and len(a):
            r["key_eq"] = bool((a["code"].to_numpy() == b["code"].to_numpy()).all()
                               and (a["event_day"].to_numpy() == pd.to_datetime(b["event_day"]).to_numpy()).all())
            r["exit_eq"] = bool((a["exit_day"].to_numpy() == pd.to_datetime(b["exit_day"]).to_numpy()).all())
            r["reason_eq"] = bool((a["reason"].to_numpy() == b["exit_reason"].to_numpy()).all())
            r["ret_maxdiff"] = float(np.abs(a["ret"].to_numpy() - b["ret"].to_numpy(float)).max())
        r["ok"] = bool(same_n and r.get("key_eq") and r.get("exit_eq") and r.get("reason_eq")
                       and r.get("ret_maxdiff", 1.0) < 1e-9) if len(a) else same_n
        out.append(r)
        _log(f"패리티 {name}: 사건 {len(ev):,} · 엔진 {len(a):,} · 원본 {len(b):,} · 일치 {r['ok']} "
             f"(최대 차 {r.get('ret_maxdiff', np.nan):.2e}, 원본 {t_ref:.1f}s)", t0)
    return out


# ---------------------------------------------------------------------------
# 본체 (GA1)
# ---------------------------------------------------------------------------

def load_or_build(cache: "str | None", t0: float) -> Pre:
    """``cache`` 가 디렉터리(.npy, mmap) 또는 .npz. 없으면 DB 에서 만들고 그 형식으로 저장."""
    if cache and os.path.exists(cache):
        pre = Pre.load(cache)
        _log(f"사전계산 캐시 {cache}: 쌍 {pre.P:,}", t0)
        return pre
    pre = build_pre(t0)
    if cache:
        pre.save(cache) if cache.endswith(".npz") else pre.save_dir(cache)
        _log(f"사전계산 저장 {cache}", t0)
    return pre


def prior_counts(out_dir_root: str = ".") -> dict:
    from swing_it.diagnostics.trials import count_trials  # noqa: PLC0415
    return {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=None, help="사전계산 npz(있으면 읽고 없으면 만든 뒤 저장)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--gens", type=int, default=GENS, help="디버그 전용(--out 을 탐색 디렉터리 밖으로)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if (args.gens != GENS or seeds != SEEDS) and not debug:
        ap.error("세대·시드 축소는 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    pre = load_or_build(args.cache, t0)
    t_pre = time.time() - t0
    E = Engine(pre)
    p = pre
    i_tr0, i_tr1, d_tr0, d_tr1 = p.entry_window(None, TRAIN_HI)
    d22 = p.day_idx(TRAIN_HI)
    _log(f"TRAIN 쌍 [{i_tr0:,}, {i_tr1:,}) / 전체 {p.P:,}", t0)

    # --- 사전계산 시뮬(원전 청산) 시간 ---
    tt = time.time()
    E.exit(EK_PREREG)
    for w in range(len(WINDOWS)):
        E.fill_ctl(EK_PREREG, w, 1)
    t_sim = time.time() - tt
    _log(f"원전 청산 시뮬 {p.P:,}쌍 + 대조 5창: {t_sim:.1f}s (numba {gf.HAVE_NUMBA})", t0)

    # --- 실데이터 패리티(GA1 원전 청산 3 + GA2 청산 조합 2) ---
    par_specs = [
        ("A−F (기준)", ga1_spec(BASE_A), {}),
        ("느슨 rs40 vm0.8 pead0 rel3 br0 w20", ga1_spec({"rs_min": 40.0, "vol_mult": 0.8, "pead_min": 0.0, "rel_max": 3.0,
                                                       "breadth_min": 0.0, "pivot_window": 20}), {}),
        ("중간 rs60 vm1.2 pead.5 rel1.5 br.3 w40", ga1_spec({"rs_min": 60.0, "vol_mult": 1.2, "pead_min": 0.5,
                                                          "rel_max": 1.5, "breadth_min": 0.3, "pivot_window": 40}), {}),
    ]
    loose = ga1_spec({"rs_min": 40.0, "vol_mult": 0.8, "pead_min": 0.0, "rel_max": 3.0, "breadth_min": 0.0,
                      "pivot_window": 30})
    for ek, prm in (((0.06, 1.5, 10, 10, 20), {"stop_cap": 0.06, "take_r": 1.5, "trail": 10, "ts": 10, "max_hold": 20}),
                    ((0.15, 4.0, 0, 0, 60), {"stop_cap": 0.15, "take_r": 4.0, "trail": 0, "ts": 0, "max_hold": 60})):
        par_specs.append((f"청산 {ek} · 진입 느슨 w30", {**loose, "ek": ek, "c_idx": 2}, prm))
    parity = parity_real(E, par_specs, t0)

    # --- 원장 ---
    led_path = os.path.join(out_dir, LABEL, "TRIALS.jsonl")
    seen = read_fingerprints(led_path)
    n_prior = prior_counts()

    def ledger_cfg(c: dict) -> dict:
        return {"design": DESIGN_DOC, "space": "GA1 = RELAX BO 공간(피벗 창 {20,30,40,50,60})", **c,
                "exit": "minervini_exits 원전(PREREG_MINERVINI)", "universe": "H5 ADV20≥5,000", "chase": 0.05,
                "fitness": "TRAIN(진입<2022-01-01) 정합 초과 부트스트랩 2.5% 하단(500, 해시 난수), 하한 150",
                "matched": CTL_DESC}

    # --- 적합도 ---
    cache: dict = {}
    eval_times = []

    def fit_cfg(c: dict) -> float:
        k = ga1_key(c)
        if k not in cache:
            tt = time.time()
            f, n = E.fit_trade(ga1_spec(c), i_tr0, i_tr1)
            eval_times.append(time.time() - tt)
            cache[k] = {"cfg": c, "fit": f, "n_train": n}
        return cache[k]["fit"]

    # --- GA 5시드 ---
    runs = {}
    for s in seeds:
        tt = time.time()
        hist, gens_out = simple_ga(lambda v: fit_cfg(ga1_decode(v)), GA1_GENES, seed=s, gens=args.gens)
        best_i = int(np.argmax([h[2] for h in hist]))
        best = ga1_decode(hist[best_i][1])
        curve = np.maximum.accumulate([h[2] for h in hist])
        runs[s] = {"best": best, "fit": hist[best_i][2], "curve": curve, "last": [ga1_decode(x) for x in gens_out[-1][0]],
                   "sec": time.time() - tt, "n_eval": len(hist), "uniq": len({ga1_key(ga1_decode(h[1])) for h in hist})}
        write_ledger(led_path, [{"config": ledger_cfg(v["cfg"]), "note": "GA1 평가"} for v in cache.values()], seen)
        _log(f"GA 시드 {s}: 최적 {best} · 적합도 {runs[s]['fit']:+.5f} · 고유 {runs[s]['uniq']} · {runs[s]['sec']:.0f}s", t0)
    n_ga_uniq = len(cache)

    # --- CMA-ES 비교(같은 공간·적합도·예산) ---
    cma_runs = {}
    cma_cache_before = set(cache)
    for s in seeds:
        tt = time.time()
        hist, rst = cma_bipop(lambda x: fit_cfg(ga1_from_unit(x)), 5, (5,), budget=POP * args.gens, seed=s)
        best_i = int(np.argmax([h[2] for h in hist]))
        cma_runs[s] = {"best": ga1_from_unit(hist[best_i][1]), "fit": hist[best_i][2],
                       "curve": np.maximum.accumulate([h[2] for h in hist]), "restarts": rst, "sec": time.time() - tt,
                       "uniq": len({ga1_key(ga1_from_unit(h[1])) for h in hist})}
        write_ledger(led_path, [{"config": ledger_cfg(v["cfg"]), "note": "GA1 공간 CMA-ES 비교 평가"}
                                for k, v in cache.items() if k not in cma_cache_before], seen)
        _log(f"CMA 시드 {s}: 최적 {cma_runs[s]['best']} · 적합도 {cma_runs[s]['fit']:+.5f} · 재시작 {len(rst)}", t0)
    n_uniq_all = len(cache)
    from swing_it.diagnostics.trials import count_trials  # noqa: PLC0415
    n_led = count_trials(LABEL, logs_dir=out_dir)
    n_dsr = n_led + sum(n_prior.values())

    # --- 과최적 2: 모든 고유 config 의 TRAIN 적합도 vs OOS α ---
    tt = time.time()
    rows = []
    for k, v in cache.items():
        spec = ga1_spec(v["cfg"])
        sel = E.select(spec, 0, p.P)
        sel_o = sel[p.pd[sel] + 1 >= d22]
        ex, _ = E.excess(spec, sel_o)
        ok = np.isfinite(ex)
        rows.append({**v["cfg"], "fit": v["fit"], "n_train": v["n_train"], "n_oos": int(ok.sum()),
                     "oos_alpha": float(ex[ok].mean()) if ok.any() else np.nan})
    tab = pd.DataFrame(rows)
    ga_only = tab.iloc[:n_ga_uniq]
    valid = ga_only["fit"] > -0.5
    rho_all = rank_corr(ga_only["fit"], ga_only["oos_alpha"])
    rho_valid = rank_corr(ga_only.loc[valid, "fit"], ga_only.loc[valid, "oos_alpha"])
    rho_cma = rank_corr(tab["fit"], tab["oos_alpha"])
    t_rank = time.time() - tt
    _log(f"순위 상관: GA 고유 {len(ga_only):,} ρ {rho_all:+.3f} (하한 위 {int(valid.sum())} ρ {rho_valid:+.3f})", t0)

    # --- 과최적 3: PBO(마지막 세대 × 5시드, 월별 정합 초과) ---
    from swing_it.diagnostics.pbo import pbo_cscv  # noqa: PLC0415
    last = {}
    for s in seeds:
        for c in runs[s]["last"]:
            last[ga1_key(c)] = c
    M = np.column_stack([monthly_excess_col(E, ga1_spec(c), E.select(ga1_spec(c), 0, p.P)) for c in last.values()])
    mo_ok = (p.months >= pd.Period(EVENT_LO, "M")) & (p.months <= pd.Period(EVENT_HI, "M"))
    M = M[np.asarray(mo_ok)]
    pbo = {mt: pbo_cscv(M, n_blocks=16, metric=mt, std_floor=1e-6) for mt in ("sharpe", "mean")}

    # --- 과최적 5·6·7: 시드 최적 · 기준 A−F · RELAX BO 최적 · CMA 최적 ---
    def block(c: dict, n_dsr_: int) -> dict:
        spec = ga1_spec(c)
        out = {}
        for bname, bsel in (("H5", -1), ("유동", 1), ("중소형", 0)):
            sel = E.select(spec, 0, p.P, band_sel=bsel)
            tr = sel[p.pd[sel] + 1 < d22]
            oo = sel[p.pd[sel] + 1 >= d22]
            out[("TRAIN", bname)] = trade_report(E, spec, tr)
            out[("OOS", bname)] = trade_report(E, spec, oo, n_dsr_)
        sel = E.select(spec, 0, p.P)
        out[("전체", "H5")] = trade_report(E, spec, sel, n_dsr_)
        flo = p.entry_window(FLOW_LO, None)
        fsel = E.select(spec, flo[0], flo[1], extra=(p.F == 1))
        out[("F=1 2024-02-19+", "H5")] = trade_report(E, spec, fsel)
        for bname, bsel in (("유동", 1), ("중소형", 0)):
            fsel = E.select(spec, flo[0], flo[1], band_sel=bsel, extra=(p.F == 1))
            out[("F=1 2024-02-19+", bname)] = trade_report(E, spec, fsel)
        return out

    best_blocks = {s: block(runs[s]["best"], n_dsr) for s in seeds}
    base_block = block(BASE_A, n_dsr)
    cma_blocks = {s: block(cma_runs[s]["best"], n_dsr) for s in seeds}
    relax_best, relax_csv = None, None
    rb_path = os.path.join(OUT_DIR, "relax_bo.csv")
    if os.path.exists(rb_path):
        rb = pd.read_csv(rb_path)
        r0 = rb.loc[rb["obj"].astype(float).idxmax()]
        relax_best = {"rs_min": float(r0["rs_min"]), "vol_mult": float(r0["vol_mult"]), "pead_min": float(r0["pead_min"]),
                      "rel_max": float(r0["rel_max"]), "breadth_min": float(r0["breadth_min"]),
                      "pivot_window": int(r0["pivot_window"])}
        relax_csv = {k: float(r0[k]) for k in r0.index if k.startswith(("OOS_", "TRAIN_H5")) or k in ("obj", "n_train")}
    relax_block = block(relax_best, n_dsr) if relax_best is not None and relax_best["pivot_window"] in WINDOWS else None

    # --- 판단 규칙 숫자 ---
    def pos(s, per, b):
        v = best_blocks[s][(per, b)].get("alpha", np.nan)
        return bool(np.isfinite(v) and v > 0)
    seeds_both = [s for s in seeds if pos(s, "OOS", "유동") and pos(s, "OOS", "중소형")]
    med_cfg = {k: float(np.median([runs[s]["best"][k] for s in seeds])) for k in
               ("rs_min", "vol_mult", "pead_min", "rel_max", "breadth_min")}
    med_cfg["pivot_window"] = int(np.median([runs[s]["best"]["pivot_window"] for s in seeds]))
    if med_cfg["pivot_window"] not in WINDOWS:
        med_cfg["pivot_window"] = min(WINDOWS, key=lambda w: abs(w - med_cfg["pivot_window"]))
    med_block = block(med_cfg, n_dsr)
    dec = {"n_both": len(seeds_both), "rule1": len(seeds_both) >= 4, "rho": rho_all, "rule2": bool(np.isfinite(rho_all) and rho_all > 0.2),
           "pbo": pbo["sharpe"]["pbo"], "rule3": bool(pbo["sharpe"]["pbo"] < 0.5)}
    dec["all"] = dec["rule1"] and dec["rule2"] and dec["rule3"]

    elapsed = time.time() - t0
    timing = {"precompute": t_pre, "sim_prereg": t_sim, "eval_ms_median": 1e3 * float(np.median(eval_times)),
              "eval_ms_mean": 1e3 * float(np.mean(eval_times)), "n_evals_computed": len(eval_times),
              "ga_sec": sum(r["sec"] for r in runs.values()), "cma_sec": sum(r["sec"] for r in cma_runs.values()),
              "rank_sec": t_rank, "total": elapsed, "ref_sec": [r["ref_sec"] for r in parity],
              "exit_stats": dict(E.stats)}
    L = render_ga1(locals())
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    tab.to_csv(os.path.join(out_dir, LABEL, "ga_configs.csv"), index=False)
    print(f"\n=== GA — {elapsed / 60:.1f}분 · 고유 config {n_uniq_all:,} (GA {n_ga_uniq:,}) · DSR N {n_dsr:,} ===")
    for s in seeds:
        b = best_blocks[s]
        print(f"시드 {s}: {runs[s]['best']} · TRAIN {runs[s]['fit']:+.5f} · OOS 유동 α {_pct(b[('OOS', '유동')].get('alpha'))}"
              f" · 중소형 α {_pct(b[('OOS', '중소형')].get('alpha'))}")
    print(f"ρ {rho_all:+.3f} · PBO {pbo['sharpe']['pbo']:.3f} · 규칙 {dec}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def _cfg_s(c: dict) -> str:
    return (f"rs≥{c['rs_min']:.1f} · vol≥{c['vol_mult']:.2f}× · pead≥{c['pead_min']:.2f} · rel≤{c['rel_max']:.2f} · "
            f"breadth>{c['breadth_min']:.2f} · 창 {c['pivot_window']}")


def render_ga1(v: dict) -> list[str]:
    p, runs, cma_runs, seeds = v["p"], v["runs"], v["cma_runs"], v["seeds"]
    tm = v["timing"]
    L = [f"# GA 문턱 최적화 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 탐색 — 판정 아님)", "",
         "설계 `GA.md`(bc29047, 결과 보기 전)를 그대로 계산한 숫자다. 판단 규칙 행은 미리 적은 문턱과의 **산술 비교**일 뿐이고 "
         "후보 선정은 사람이 한다. 재현: `NUMBA_NUM_THREADS=8 uv run --with numba --with cmaes==0.13.1 --extra pg python "
         "research/experiments/ga_run.py`.", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 출력·원장 `{v['out_dir']}`. 측정으로 쓰지 않는다.", ""]
    d = v["dec"]
    L += ["## 판단 규칙 숫자 (GA.md — 산술 비교, 판정 아님)", "",
          "| 조건 | 문턱 | 숫자 | 산술 |", "|---|---|---|---|",
          f"| 시드 최적이 OOS 유동·중소형 **둘 다** 모멘텀 α 점추정 > 0 | 5개 중 ≥ 4 | {d['n_both']}/5 "
          f"(시드 {', '.join(map(str, v['seeds_both'])) or '없음'}) | {'✓' if d['rule1'] else '✗'} |",
          f"| TRAIN 적합도 → OOS α 순위 상관(GA 가 평가한 고유 config 전부) | > 0.2 | {d['rho']:+.3f} | {'✓' if d['rule2'] else '✗'} |",
          f"| PBO (CSCV, 마지막 세대 × 5시드, 월별 정합 초과, Sharpe) | < 0.5 | {d['pbo']:.3f} | {'✓' if d['rule3'] else '✗'} |",
          f"| 셋 다 | | | {'✓ (규칙상 유전자 중앙값 config 를 후보로 올리는 산술 조건 충족)' if d['all'] else '✗ (규칙 문구: GA 로도 표본 밖에서 살아남는 문턱 없음)'} |",
          "", f"유전자 중앙값 config: {_cfg_s(v['med_cfg'])}", ""]

    L += ["## 가속·시간", "",
          f"- 사전계산(DB 적재·보정·피처·쌍 {p.P:,}): **{tm['precompute']:.0f}s** · 원전 청산 시뮬 {p.P:,}쌍 + 대조 5창 추출: "
          f"**{tm['sim_prereg']:.1f}s** (numba {gf.HAVE_NUMBA})",
          f"- 적합도 평가(마스크 → 겹침 제거 → 부트스트랩 500): 중앙값 **{tm['eval_ms_median']:.1f} ms** · 평균 {tm['eval_ms_mean']:.1f} ms "
          f"(실계산 {tm['n_evals_computed']:,}회, 나머지는 같은 config 캐시)",
          f"- 같은 일을 원본 `simulate_minervini` 로 한 번: 패리티 config 별 {', '.join(f'{x:.1f}s' for x in tm['ref_sec'])} "
          f"(사건 수에 비례 — 대조 풀 시뮬·정합 추출 제외)",
          f"- GA 5시드 {tm['ga_sec']:.0f}s · CMA-ES 5시드 {tm['cma_sec']:.0f}s · 순위 상관용 OOS 재평가 {tm['rank_sec']:.0f}s · "
          f"**총 {tm['total'] / 60:.1f}분**", ""]

    L += ["## 패리티 (실데이터 — 엔진 = `simulate_minervini`, 같은 사건·같은 보정판 패널)", "",
          "| config | 사건 | 엔진 n | 원본 n | 종목·신호일 | 청산일 | 사유 | 수익 최대 차 | 일치 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in v["parity"]:
        L.append(f"| {r['name']} | {r['events']:,} | {r['n_engine']:,} | {r['n_ref']:,} | {r.get('key_eq', '—')} | "
                 f"{r.get('exit_eq', '—')} | {r.get('reason_eq', '—')} | {r.get('ret_maxdiff', np.nan):.1e} | {'✓' if r['ok'] else '✗'} |")
    L += ["", f"돌파 성분 분해(창 50·1.5×) vs `pivot_breakout` 불일치: {p.info.get('breakout_mismatch')} 쌍. "
          "합성 데이터 패리티(청산 조합 5종 × 시드, 익절 없음 ∧ 추적 = `simulate_sepa`, numba = 순수 파이썬)는 `tests/test_ga_fast.py`.", ""]

    L += ["## 과최적 1 — 시드별 최적 config 와 흩어짐", "",
          "| 시드 | rs_min | vol_mult | pead_min | rel_max | breadth_min | 피벗 창 | TRAIN 적합도 | TRAIN n | 고유 평가 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        c = runs[s]["best"]
        n_tr = v["cache"][ga1_key(c)]["n_train"]
        L.append(f"| {s} | {c['rs_min']:.1f} | {c['vol_mult']:.2f} | {c['pead_min']:.2f} | {c['rel_max']:.2f} | "
                 f"{c['breadth_min']:.2f} | {c['pivot_window']} | {runs[s]['fit']:+.5f} | {n_tr} | {runs[s]['uniq']} |")
    sp = gene_spread([{**r["best"], "pivot_idx": WINDOWS.index(r["best"]["pivot_window"])} for r in runs.values()],
                     list(GA1_GENES))
    L += ["", "유전자별 시드 간 폭(최대 − 최소, 범위 대비): " + " · ".join(f"{k} {a:.2f}~{b:.2f} ({w:.0%})" for k, (a, b, w) in sp.items()), ""]

    L += ["## 과최적 2 — TRAIN 적합도 순위 vs OOS(2022-01-01 ~ 2026-04-01) 정합 α 순위", "",
          f"- GA 가 평가한 고유 config {len(v['ga_only']):,}개: 스피어만 **{v['rho_all']:+.3f}** (판단 규칙 값)",
          f"- 그중 거래수 하한 위(적합도 > −0.5) {int(v['valid'].sum()):,}개: {v['rho_valid']:+.3f}",
          f"- CMA-ES 평가까지 합친 {len(v['tab']):,}개: {v['rho_cma']:+.3f}",
          "- OOS α = 그 config 를 전 구간에 겹침 제거로 돌린 트레이드 중 진입 ≥ 2022-01-01, 짝 있는 것의 평균 초과(H5).", ""]

    pb = v["pbo"]
    L += ["## 과최적 3 — PBO (CSCV)", "",
          f"- 후보 = 5시드 마지막 세대 고유 config {pb['sharpe']['n_candidates']}개 · 행 = 진입월(2017-09 ~ 2026-04, 거래 없는 달 0) 정합 초과 평균 · "
          f"조각 16 · 조합 {pb['sharpe']['n_combos']:,}",
          f"- PBO(Sharpe) **{pb['sharpe']['pbo']:.3f}** · PBO(평균) {pb['mean']['pbo']:.3f}", ""]

    L += ["## 과최적 4 — DSR", "",
          f"- N = 이 원장 고유 config {v['n_led']:,} (GA {v['n_ga_uniq']:,} + CMA-ES 비교 추가분) + 계열 원장 "
          + " + ".join(f"{k} {n}" for k, n in v["n_prior"].items()) + f" = **{v['n_dsr']:,}**",
          "- 아래 표 'DSR' 열 = 그 config 의 OOS(H5) 트레이드 순수익(1× 비용) 표본의 deflated Sharpe 확률(N 위).", ""]

    def blk_rows(name: str, b: dict) -> list[str]:
        out = []
        for key in (("TRAIN", "H5"), ("OOS", "유동"), ("OOS", "중소형"), ("OOS", "H5"), ("F=1 2024-02-19+", "H5"),
                    ("F=1 2024-02-19+", "유동"), ("F=1 2024-02-19+", "중소형")):
            s = b.get(key, {"n": 0})
            dsr = _f(s.get("dsr", np.nan), "{:.1%}") if key == ("OOS", "H5") else ""
            out.append(f"| {name} | {key[0]} | {key[1]} | {trow(s)} | {dsr} |")
        return out

    L += ["## 과최적 5·6·7 — 시드 최적을 OOS 유동·중소형에 한 번, 기준(A − F)·RELAX BO 최적·CMA-ES 최적과 나란히, F 추가", "",
          "OOS = 진입 2022-01-01 ~ (사건 ≤ 2026-04-01). 유동 = ADV20 ≥ 200억, 중소형 = 50~200억 — 띠마다 따로 적용(겹침 제거도 띠 안). "
          "α = 모멘텀 정합 초과(비용 전), 순평균 = `cost_rt`(청산일·시장) 차감. 'F=1' = 그 config ∧ 수급 F=1, 사건 2024-02-19+.", "",
          "| config | 구간 | 띠 " + T_HEAD + " DSR |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        L += blk_rows(f"GA 시드 {s}", v["best_blocks"][s])
    L += blk_rows("유전자 중앙값", v["med_block"])
    L += blk_rows("기준 A − F", v["base_block"])
    if v["relax_block"] is not None:
        L += blk_rows(f"RELAX BO 최적 ({_cfg_s(v['relax_best'])})", v["relax_block"])
    for s in seeds:
        L += blk_rows(f"CMA-ES 시드 {s}", v["cma_blocks"][s])
    L.append("")
    if v["relax_block"] is None and v["relax_best"] is not None:
        rc = v["relax_csv"]
        L += [f"RELAX BO 최적(`relax_bo.csv` 목적값 최대, 목적 {rc['obj']:+.5f}): {_cfg_s(v['relax_best'])} — 피벗 창이 GA 단계 "
              "{20,30,40,50,60} 밖이라 이 엔진으로 다시 재지 않고 RELAX 러너 숫자를 옮긴다(대조 추출 난수가 달라 잡음만큼 다를 수 있다):", "",
              "| 구간·띠 | n | α/트레이드 | α/보유일 | 순평균 1× |", "|---|---|---|---|---|"]
        for per, b in (("TRAIN", "H5"), ("OOS", "H5"), ("OOS", "유동"), ("OOS", "중소형")):
            k = f"{per}_{b}"
            if f"{k}_n" in rc:
                L.append(f"| {per} {b} | {rc[f'{k}_n']:.0f} | {_pct(rc[f'{k}_alpha'])} | {_f(rc[f'{k}_pd'] * 1e4, '{:+.1f}')}bp | "
                         f"{_pct(rc[f'{k}_net'])} |")
        L.append("")
    elif v["relax_best"] is None:
        L += ["RELAX BO 최적: `relax_bo.csv` 가 이 실행 시점에 없었다 — 나란히 적지 못함.", ""]

    L += ["## CMA-ES(CMAwM + BIPOP) vs 단순 GA — 같은 6유전자 공간·같은 적합도·같은 예산(1,800/시드)", "",
          "| 시드 | GA TRAIN 최적 | CMA TRAIN 최적 | GA 최적 config | CMA 최적 config | CMA 재시작(체제·개체) |", "|---|---|---|---|---|---|"]
    for s in seeds:
        rst = " ".join(f"{r['regime'][0]}{r['popsize']}" for r in cma_runs[s]["restarts"])
        L.append(f"| {s} | {runs[s]['fit']:+.5f} | {cma_runs[s]['fit']:+.5f} | {_cfg_s(runs[s]['best'])} | "
                 f"{_cfg_s(cma_runs[s]['best'])} | {rst} |")
    marks = [100, 300, 600, 1000, 1400, 1800]
    L += ["", "수렴 곡선(최고 적합도, 평가 수 기준 — 시드 중앙값):", "",
          "| 평가 수 | " + " | ".join(str(m) for m in marks) + " |", "|---|" + "---|" * len(marks)]
    for nm, rr in (("단순 GA", runs), ("CMA-ES", cma_runs)):
        vals = []
        for m in marks:
            xs = [r["curve"][min(m, len(r["curve"])) - 1] for r in rr.values()]
            vals.append(f"{np.median(xs):+.5f}")
        L.append(f"| {nm} | " + " | ".join(vals) + " |")
    L.append("")

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          "- **후보 표본:** 사전계산은 추세 템플릿으로 미리 좁히지 않고 사건 구간 H5 종목-일 전부를 시뮬한다(그 쌍들이 대조 풀이기도 하다). "
          "추세 템플릿은 마스크 조건. 메모리: 쌍 배열 float64/int(통틀어 수백 MB).",
          "- **추격 금지:** 시뮬 결과는 피벗과 무관 → 창(5) × 추격(2/5/10%) 비트로 저장, 사건·대조 모두 config 피벗 창으로 사후(RELAX 와 같다).",
          "- **거래량 배수의 평균 창 = 피벗 창**(`pivot_breakout(window=창)` 과 같다). 50일 고정 해석이 아니다.",
          "- **NaN:** RS·PEAD(실적 행 없음)·rel·breadth NaN 은 조건 통과 안 함(RELAX·CONVERGENCE 와 같다) — rel_max 3.0 도 '조건 없음' 이 아니다.",
          "- **대조 추출:** 쌍별 해시 난수(위 설명). 짝 없는 쌍(층 비거나 자기뿐)은 적합도·α 에서 빠지지만 트레이드로는 남는다(겹침 제거에 참여).",
          "- **부트스트랩:** 고정 해시 난수 500회, 백분위 선형 보간 — `_boot_lower` 와 같은 모양, 난수열만 다르다.",
          "- **세대:** 초기 개체 = 1세대, 30세대 × 60 = 1,800 평가/시드(같은 config 재평가는 캐시). 엘리트 2 는 그대로 복사, 나머지는 토너먼트 3 두 번 → BLX-0.5 → 변이.",
          "- **PBO 열:** 진입월별 정합 초과 평균(짝 있는 것), 거래 없는 달 0 — `sepa_explore.monthly_excess` 와 같은 규약.",
          "- **원장:** 고유 config 마다 한 줄(GA + CMA-ES 비교 평가). DSR N = 이 원장 + 계열 원장(sepa_confluence·sepa_explore·minervini_exits·convergence·relax).",
          f"- 가격: `relax_run.KrxPatch` 로 2026-09-14+ 행을 분봉 정규장 값으로 — {p.info.get('nxt')}", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
