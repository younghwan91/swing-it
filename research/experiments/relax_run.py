#!/usr/bin/env python
"""제약 완화 — 하나씩 빼기 · 베이지안 최적화(TRAIN 전용) · 베이지안 축소. **탐색이지 판정이 아니다.**

설계: ``research/logs/sepa_explore/RELAX.md`` (결과 보기 전 커밋 d04e983 + 감사 반영 b4a3532, 둘 다 실행 전). 유니버스·가격 보정·청산·대조·칸·탐색 공간·
목적·평가·과최적 측정·축소·판단 규칙을 그대로 옮긴다. 이 러너는 **리포터**다 — 숫자를 ``RELAX_RESULTS.md`` 에 적고
판단 규칙의 문턱과 산술 비교만 한다(후보 선정은 사람이, 확인은 OOS·중소형·전진 페이퍼).

- 유니버스 H5: ``korea.universe_mask`` 위생 ∧ ADV20 ≥ 5,000(백만원). 결과는 유동(≥ 20,000)·중소형(5,000~20,000)으로도.
- 가격: ``krx_session.patch_prices`` 로 2026-09-14 이후 NXT 오염 행을 정규장(분봉) 값으로 바꾼 뒤 모든 패널을 만든다
  (조정 테이블은 ``adjusted=True``, 원 일봉 종가는 ``adjusted=False``). 바꾼/못 바꾼 행 수를 적는다.
- 청산: 미너비니 원전 프로토콜(``minervini_exits.PREREG_MINERVINI``) 고정.
- 대조: 모멘텀 정합 — 같은 날 × 시장 × 유동/중소형 띠 × 12-1 모멘텀 5분위 × ADV20 2분위 × 직전 5일 수익 3분위
  (분위는 그날 그 띠·그 시장 안에서 자른다), 자기 제외, 5배 복원 추출, seed 0, 날짜군집 부트스트랩 2,000. 대조에도 같은
  프로토콜(``control_minervini``). 대조 풀은 청산 규칙에만 달려 있으므로 **사건 구간의 H5 전 종목-일을 한 번** 시뮬해
  캐시하고, 추격 금지(피벗 창에 따라 다름)는 체결가 ≤ 피벗×1.05 로 사후에 거른다(대조는 겹침 없는 독립 시뮬이라 정확히 같다).
- 지표: 트레이드당 alpha(정합 초과 평균) 와 보유 1일당 alpha — **주 = Σ초과 / Σ보유일**, 참고 = 평균(초과/보유일).
- 1. 하나씩 빼기: 기준 = CONVERGENCE A 전체 조건(F 제외판 전 구간 9칸, F 포함판 2024-02-19+ 10칸).
- 2. BO: ``swing_it.validation.optimization.mini_bo`` 40회 seed 0, 목적 = ``make_objective``(TRAIN 진입 < 2022-01-01
  정합 초과의 부트스트랩 2.5% 하단, 거래수 하한 150). 최적 config 를 OOS 유동·중소형에 한 번. 40 config 순위 상관·
  PBO(CSCV, 월별 정합 초과)·DSR. F 는 최적 ∧ F=1 1칸(2024-02-19+).
- 3. 베이지안 축소: 하나씩 빼기의 조건 효과를 정규-정규(사전 평균 0) 경험적 베이즈로 축소, P(효과 > 0).
- 원장: ``research/logs/sepa_explore/relax/TRIALS.jsonl`` — 칸·BO config 를 **결과 보기 전에**(평가 직전) 적는다.

기존 러너(``sepa_confluence_gate``·``sepa_explore``·``sepa_momentum``·``minervini_exits_run``·``convergence_run``)는
import 만 한다(가격 보정은 그 모듈들의 ``read_prices`` 이름을 실행 중에만 보정판으로 바꿔 끼운다 — 파일은 그대로).

Run:  uv run --with optuna --extra pg python research/experiments/relax_run.py
디버그(구간 축소 — 탐색 디렉터리·원장을 더럽히지 않는다):
      ... relax_run.py --event-lo 2021-07-01 --event-hi 2022-06-30 --n-trials 3 --out <scratch>/relax_debug
"""

from __future__ import annotations

import argparse
import contextlib
import math
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

import convergence_run as cvr  # noqa: E402
import sepa_confluence_gate as scg  # noqa: E402
from convergence_run import (  # noqa: E402
    ADV_HI,
    ADV_LO,
    FLOW_LO,
    PEAD_POP_FLOOR,
    arm_stats,
    raw_close_panel,
    relative_contraction,
    repoint_pead,
)
from krx_quant_core.stats.metrics import spearman  # noqa: E402
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample  # noqa: E402
from minervini_exits_run import SKIP_ORDER, _alpha, _dist, _f  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402
from sepa_confluence_gate import N_BOOT, SEED, Data, _costs, _key, _log, _pct, sub_alpha  # noqa: E402
from sepa_explore import EVENT_HI, EVENT_LO, PBO_BLOCKS, PBO_STD_FLOOR, monthly_excess  # noqa: E402
from sepa_momentum import (  # noqa: E402
    candidate_counts,
    day_market_median_half,
    day_market_quantiles,
    matched_on,
    momentum_12_1,
)

import research.signals.sepa as sepa_mod  # noqa: E402
from research.signals.minervini_exits import (  # noqa: E402
    PREREG_MINERVINI,
    _run,
    contraction_low_level,
    control_minervini,
)
from research.signals.sepa import _Book  # noqa: E402
from swing_it.diagnostics.pbo import pbo_cscv  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.korea import universe_mask  # noqa: E402
from swing_it.features.minervini import pivot_level  # noqa: E402
from swing_it.krx_session import NXT_START, patch_prices, read_session_bars  # noqa: E402
from swing_it.storage import connect, db_default  # noqa: E402
from swing_it.validation.optimization import TRADE_FLOOR, TRAIN_HI, _boot_lower, mini_bo, split_stats  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "relax"
RESULTS_MD = "RELAX_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/RELAX.md (d04e983 + 감사 반영 b4a3532)"
#: 감사 반영 — 이 연구 계열의 계획 평가 수(GA 9,000 · CMA-ES 49,350 · 워크포워드 86,400). DSR 의 두 번째 N 에 이 BO 시행 수를 더해 쓴다.
PLANNED_EVALS = {"GA": 9_000, "CMA-ES": 49_350, "워크포워드": 86_400}
#: 결과 문서에서 진입 ≥ 2022-01-01 구간의 이름 — 표본 밖이 아니다(유동 2022~26·중소형 일부는 이전 시험에서 봤다).
UNSEEN = "미관측"
PRIOR_LEDGERS = (("sepa_confluence", "research/logs"), ("sepa_explore", "research/logs"),
                 ("minervini_exits", "research/logs/sepa_explore"), ("convergence", "research/logs/sepa_explore"))

# --- 설계 수치(RELAX.md — 탐색하지 않는다) ---
LIQ_FLOOR = ADV_HI                     # 유동 = ADV20 ≥ 20,000 (백만원 = 200억), 중소형 = 5,000 ≤ ADV20 < 20,000
MOM_Q, RET5_Q = 5, 3                   # 층: 12-1 모멘텀 5분위 · 직전 5일 수익 3분위
STRATA = ["date", "market", "band", "q_mom5", "adv2", "q_ret5_3"]
THIN_MIN, THIN_WARN = 3, 0.40          # 후보 < 3 인 트레이드 비율이 40% 를 넘으면 적는다
N_TRIALS, BO_SEED = 40, 0
SPACE: dict[str, tuple] = {            # RELAX.md 탐색 공간(순서 = optuna 제안 순서)
    "rs_min": (40.0, 90.0),
    "vol_mult": (0.8, 2.0),
    "pead_min": (0.0, 0.9),
    "rel_max": (0.6, 3.0),
    "breadth_min": (0.0, 0.6),
    "pivot_window": (20, 60),
}
EB_Z90 = 1.6448536269514722            # 90% 구간
SIM_RULES = {k: PREREG_MINERVINI[k] for k in ("stop_floor", "stop_cap", "stop_mult", "take_r", "time_stop_days",
                                               "max_hold", "limit_up_skip")}
SIM_RULES["fixed_stop"] = None
MARKET_CODE = {"KOSPI": 0, "KOSDAQ": 1}
BANDS = {1: "유동", 0: "중소형"}

#: 기준 = CONVERGENCE A 전체 조건(F 제외). ``None`` = 그 조건 없음.
BASE: dict = {"tt": True, "rs_min": 70.0, "breakout": True, "vol_mult": 1.5, "breadth_min": 0.5, "pead_min": 0.70,
              "rel_max": 1.0, "chase": 0.05, "flow": False, "pivot_window": 50,
              "event_lo": EVENT_LO, "event_hi": EVENT_HI}
#: 하나씩 빼기 조건(RELAX.md 순서) — (키, 이름, 기준에서 바꿀 값).
CONDS: tuple = (
    ("tt", "추세 템플릿", {"tt": False}),
    ("rs", "RS ≥ 70", {"rs_min": None}),
    ("breakout", "피벗 돌파", {"breakout": False}),
    ("vol", "거래량 1.5×", {"vol_mult": None}),
    ("breadth", "breadth > 0.5", {"breadth_min": None}),
    ("pead", "PEAD ≥ 0.70", {"pead_min": None}),
    ("rel", "rel ≤ 1.0", {"rel_max": None}),
    ("chase", "추격 금지 5%", {"chase": None}),
    ("flow", "수급 F=1", {"flow": False}),
)
BASE_NAME = "기준"
FAM_NOF, FAM_F = "F 없음(전 구간)", "F 포함(2024-02-19+)"

UNIVERSE_DESC = ("korea.universe_mask(raw_close=daily_bars.close, adv_floor=5000) — 보통주·KOSPI/KOSDAQ·ADV20≥5,000"
                 "(백만원)·원종가≥1000·상장252일; 띠: 유동 ADV20≥20,000 / 중소형 5,000~20,000")
MATCH_DESC = ("모멘텀 정합: 같은 날 × 시장 × 띠(유동/중소형) × 12-1 모멘텀 5분위 × ADV20 2분위 × 직전 5일 수익 3분위"
              "(분위는 날짜·띠·시장 안), 풀 = H5 전 종목-일 같은 프로토콜(control_minervini), 자기 제외, n_per 5, seed 0, "
              "날짜군집 부트스트랩 2000")
PRICE_DESC = "read_prices → krx_session.patch_prices(2026-09-14+, 분봉 정규장; 조정=adjusted, daily_bars=raw)"


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_relax_helpers.py)
# ---------------------------------------------------------------------------

def loo_cells(base: dict, conds=CONDS) -> dict[str, dict]:
    """하나씩 빼기 칸: 기준 + (기준 − 조건) 각각. 이름 = ``기준`` / ``− <조건 이름>``."""
    out = {BASE_NAME: dict(base)}
    for _, name, over in conds:
        out[f"− {name}"] = {**base, **over}
    return out


def loo_families(lo: "str | None" = None, hi: "str | None" = None) -> dict[str, dict[str, dict]]:
    """F 제외판(전 구간 — 조건 8개, 9칸) · F 포함판(2024-02-19+ — 조건 9개, 10칸). ``lo``/``hi`` 는 디버그 구간 축소."""
    def clip(c: dict) -> dict:
        c = dict(c)
        if lo is not None:
            c["event_lo"] = max(c["event_lo"], lo)
        if hi is not None:
            c["event_hi"] = min(c["event_hi"], hi)
        return c

    nof = loo_cells(BASE, tuple(c for c in CONDS if c[0] != "flow"))
    f = loo_cells({**BASE, "flow": True, "event_lo": FLOW_LO}, CONDS)
    return {FAM_NOF: {k: clip(v) for k, v in nof.items()}, FAM_F: {k: clip(v) for k, v in f.items()}}


def bo_config(params: dict, lo: "str | None" = None, hi: "str | None" = None) -> dict:
    """BO 제안값 → 칸 config(추세 템플릿·돌파·추격 금지·청산 고정, F 없음, 전 구간)."""
    c = {**BASE, **{k: (int(v) if k == "pivot_window" else float(v)) for k, v in params.items()}}
    if lo is not None:
        c["event_lo"] = max(c["event_lo"], lo)
    if hi is not None:
        c["event_hi"] = min(c["event_hi"], hi)
    return c


def event_mask(cols: dict, cfg: dict) -> np.ndarray:
    """사건 master 열(같은 길이 배열) → 그 칸의 진입 조건. NaN 은 조건을 통과하지 않는다(조건이 꺼져 있으면 안 본다).

    필요한 열: ``event_day``·``tt``·``rs``·``close``·``pivot``(그 칸의 피벗 창)·``volume``·``vol_avg``(같은 창)·``breadth``·
    ``pead_rank``·``rel``·``F``. 돌파 = 종가 > 피벗, 거래량 = 평균 > 0 ∧ 거래량 ≥ 배수 × 평균
    (:func:`swing_it.features.minervini.pivot_breakout` 의 두 성분을 따로 켜고 끈다).
    """
    day = np.asarray(cols["event_day"], dtype="datetime64[ns]")
    mk = (day >= np.datetime64(pd.Timestamp(cfg["event_lo"]))) & (day <= np.datetime64(pd.Timestamp(cfg["event_hi"])))

    def num(k):
        return np.asarray(cols[k], float)

    with np.errstate(invalid="ignore"):
        if cfg["tt"]:
            mk &= np.asarray(cols["tt"], bool)
        if cfg["rs_min"] is not None:
            mk &= num("rs") >= cfg["rs_min"]
        if cfg["breakout"]:
            mk &= num("close") > num("pivot")
        if cfg["vol_mult"] is not None:
            va = num("vol_avg")
            mk &= (va > 0) & (num("volume") >= cfg["vol_mult"] * va)
        if cfg["breadth_min"] is not None:
            mk &= num("breadth") > cfg["breadth_min"]
        if cfg["pead_min"] is not None:
            mk &= num("pead_rank") >= cfg["pead_min"]
        if cfg["rel_max"] is not None:
            mk &= num("rel") <= cfg["rel_max"]
        if cfg["flow"]:
            mk &= num("F") == 1
    return mk


def chase_ok(entry_px, pivot, chase: "float | None") -> np.ndarray:
    """추격 금지 사후 거르기 — 시뮬레이터(``minervini_exits._simulate_one``)와 같은 식: 피벗이 유한하고
    체결가 > 피벗 × (1 + chase) 면 버린다. ``chase=None`` 이면 전부 통과."""
    e = np.asarray(entry_px, float)
    if chase is None:
        return np.ones(len(e), bool)
    pv = np.asarray(pivot, float)
    with np.errstate(invalid="ignore"):
        return ~(np.isfinite(pv) & (e > pv * (1.0 + chase)))


def band_strata(values: pd.DataFrame, bands: dict, market: pd.Series, *, n_q: "int | None") -> pd.DataFrame:
    """띠별로 따로 자른 층 패널을 하나로. ``bands`` = {띠: bool 마스크}(서로 겹치지 않음). ``n_q`` 가 정수면
    날짜×시장 분위(:func:`sepa_momentum.day_market_quantiles`), ``None`` 이면 중앙값 2분위(``day_market_median_half``).
    띠 밖은 NaN."""
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for mask in bands.values():
        m = mask.reindex_like(values).fillna(False).astype(bool)
        q = (day_market_median_half(values, m, market) if n_q is None
             else day_market_quantiles(values, m, market, n_q=n_q))
        out = out.where(~m, q)
    return out


def stratum_id(day_idx, market, band, q_mom, adv2, q_ret) -> np.ndarray:
    """층 키 → 정수 하나(풀을 트레이드 층으로 미리 좁히는 데만 쓴다). 성분 하나라도 없으면 −1."""
    mk = pd.Series(market).map(MARKET_CODE).to_numpy(float)
    parts = [np.asarray(x, float) for x in (band, q_mom, adv2, q_ret)]
    ok = np.isfinite(mk) & np.all([np.isfinite(x) for x in parts], axis=0)
    d = np.asarray(day_idx, np.int64)
    b, qm, a, qr = (np.where(ok, x, 0).astype(np.int64) for x in parts)
    sid = ((((d * 2 + np.where(ok, mk, 0).astype(np.int64)) * 2 + b) * MOM_Q + qm) * 2 + a) * RET5_Q + qr
    return np.where(ok, sid, -1)


def _cluster_sums(num, den, clusters, uniq) -> "tuple[np.ndarray, np.ndarray]":
    i = np.searchsorted(uniq, clusters)
    k = len(uniq)
    return np.bincount(i, weights=num, minlength=k), np.bincount(i, weights=den, minlength=k)


def ratio_ci(num, den, clusters, *, n_boot: int = N_BOOT, seed: int = SEED, ci: float = 0.95
             ) -> "tuple[float, float, float]":
    """Σnum / Σden 과 날짜군집 부트스트랩 CI(군집을 복원 추출해 비율을 다시 잰다). 보유 1일당 alpha 용."""
    num, den, clusters = np.asarray(num, float), np.asarray(den, float), np.asarray(clusters)
    if len(num) == 0 or den.sum() <= 0:
        return (np.nan, np.nan, np.nan)
    point = float(num.sum() / den.sum())
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return (point, np.nan, np.nan)
    s, n = _cluster_sums(num, den, clusters, uniq)
    draw = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = s[draw].sum(1) / n[draw].sum(1)
    a = (1.0 - ci) / 2.0 * 100.0
    return (point, float(np.nanpercentile(r, a)), float(np.nanpercentile(r, 100.0 - a)))


def diff_boot(a: tuple, b: tuple, *, n_boot: int = N_BOOT, seed: int = SEED, ci: float = 0.95) -> dict:
    """두 칸의 비율(Σnum/Σden) 차이 a − b 와 **같은 날짜 군집을 함께** 뽑은 부트스트랩 표준오차·CI.

    ``a``·``b`` = ``(num, den, clusters)``. 트레이드당 alpha 는 den = 1, 보유일당은 den = 보유일. 두 칸이 같은 트레이드를
    많이 공유하므로 군집을 합집합에서 한 번 뽑아 양쪽에 같이 쓴다(공분산이 표준오차에 들어간다)."""
    (na, da, ca), (nb, db, cb) = ([np.asarray(x) for x in g] for g in (a, b))
    na, da, nb, db = (np.asarray(x, float) for x in (na, da, nb, db))
    if len(na) == 0 or len(nb) == 0 or da.sum() <= 0 or db.sum() <= 0:
        return {"point": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan}
    point = float(na.sum() / da.sum() - nb.sum() / db.sum())
    uniq = np.unique(np.concatenate([ca, cb]))
    if len(uniq) < 2:
        return {"point": point, "se": np.nan, "lo": np.nan, "hi": np.nan}
    sa, ta = _cluster_sums(na, da, ca, uniq)
    sb, tb = _cluster_sums(nb, db, cb, uniq)
    draw = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sa[draw].sum(1) / ta[draw].sum(1) - sb[draw].sum(1) / tb[draw].sum(1)
    q = (1.0 - ci) / 2.0 * 100.0
    return {"point": point, "se": float(np.nanstd(d, ddof=1)), "lo": float(np.nanpercentile(d, q)),
            "hi": float(np.nanpercentile(d, 100.0 - q))}


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def eb_tau2(delta, se) -> float:
    """정규-정규(사전 평균 0) 주변우도 최대 τ². δ_k ~ N(0, s_k² + τ²). 로그 격자 + 황금분할 다듬기, τ² ≥ 0."""
    d, s2 = np.asarray(delta, float), np.asarray(se, float) ** 2

    def ll(t2: float) -> float:
        v = s2 + t2
        return float(-0.5 * np.sum(np.log(v) + d ** 2 / v))

    hi = float(max(np.max(d ** 2), np.max(s2))) * 10.0 + 1e-300
    grid = np.concatenate([[0.0], np.geomspace(hi * 1e-12, hi, 4000)])
    vals = np.array([ll(g) for g in grid])
    i = int(np.argmax(vals))
    if i == 0:
        return 0.0
    a, b = grid[i - 1], grid[min(i + 1, len(grid) - 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        c, e = b - g * (b - a), a + g * (b - a)
        if ll(c) >= ll(e):
            b = e
        else:
            a = c
    t = 0.5 * (a + b)
    return t if ll(t) > ll(0.0) else 0.0


def eb_shrink(delta, se, *, z: float = EB_Z90) -> pd.DataFrame:
    """조건 효과 δ(표준오차 s)를 0 쪽으로 축소(정규-정규, 사전 N(0, τ²), τ² = 주변우도 최대).

    사후 평균 = B·δ, 사후 표준편차 = √B·s, B = τ²/(τ²+s²). P(효과 > 0) = Φ(사후 평균 / 사후 표준편차).
    τ̂² = 0 이면 사후가 0 한 점이라 P 는 정의되지 않는다 — 0.5 로 적고 ``degenerate`` 를 켠다.
    유한하지 않거나 s ≤ 0 인 행은 τ² 추정에서 빼고 결과는 NaN.
    """
    d, s = np.asarray(delta, float), np.asarray(se, float)
    ok = np.isfinite(d) & np.isfinite(s) & (s > 0)
    t2 = eb_tau2(d[ok], s[ok]) if ok.sum() >= 1 else np.nan
    rows = []
    for di, si, oi in zip(d, s, ok, strict=True):
        if not oi or not np.isfinite(t2):
            rows.append({"delta": di, "se": si, "tau2": t2, "B": np.nan, "post_mean": np.nan, "post_sd": np.nan,
                         "lo90": np.nan, "hi90": np.nan, "p_pos": np.nan, "degenerate": False})
            continue
        B = t2 / (t2 + si ** 2)
        pm, ps = B * di, math.sqrt(B) * si
        p = _norm_cdf(pm / ps) if ps > 0 else 0.5
        rows.append({"delta": di, "se": si, "tau2": t2, "B": B, "post_mean": pm, "post_sd": ps,
                     "lo90": pm - z * ps, "hi90": pm + z * ps, "p_pos": p, "degenerate": not ps > 0})
    return pd.DataFrame(rows)


def purge_train(rets, edates, xdates, *, train_hi: str = TRAIN_HI) -> np.ndarray:
    """감사 반영 purge — 진입 < ``train_hi`` 인데 청산 ≥ ``train_hi`` 인 트레이드의 값을 NaN 으로(목적함수의 TRAIN 에서
    빠진다 — ``split_stats`` 가 비유한 값을 버린다). 진입 ≥ ``train_hi`` 는 그대로(목적함수가 어차피 안 본다)."""
    r = np.asarray(rets, float).copy()
    e, x = np.asarray(edates).astype(str), np.asarray(xdates).astype(str)
    r[(e < train_hi) & (x >= train_hi)] = np.nan
    return r


def bo_objective_value(rets, edates, *, floor: int = TRADE_FLOOR, train_hi: str = TRAIN_HI) -> "tuple[float, int]":
    """``make_objective`` 와 같은 식으로 한 config 의 목적값을 다시 잰다(기록용) → ``(값, TRAIN n)``."""
    rt = split_stats(np.asarray(rets, float), np.asarray(edates), lo=None, hi=train_hi)
    n = len(rt)
    if n < floor:
        return -1.0 + n / floor * 1e-3, n
    return _boot_lower(rt, n_boot=1000, seed=0), n


def decision_numbers(cond_p: dict, oos_liq: float, oos_small: float, rho: float) -> dict:
    """RELAX.md 판단 규칙의 산술(판정 아님): 축소 후 P(효과>0) < 0.5 인 조건 목록 · 최적 config 의 OOS 유동·중소형
    alpha 점추정 > 0 ∧ TRAIN→OOS 순위 상관 > 0."""
    def pos(v):
        return bool(v is not None and np.isfinite(v) and v > 0)

    drop = [k for k, p in cond_p.items() if p is not None and np.isfinite(p) and p < 0.5]
    out = {"drop": drop, "liq_pos": pos(oos_liq), "small_pos": pos(oos_small), "rho_pos": pos(rho)}
    out["all_three"] = out["liq_pos"] and out["small_pos"] and out["rho_pos"]
    return out


# ---------------------------------------------------------------------------
# 가격 보정 (NXT 2026-09-14+)
# ---------------------------------------------------------------------------

class KrxPatch:
    """``read_prices`` 를 감싸 돌려받은 롱 가격표의 2026-09-14+ 행을 정규장 값으로 바꾼다(분봉은 한 번만 읽는다)."""

    def __init__(self, con):
        self.con = con
        self.sess: "pd.DataFrame | None" = None
        self.stats: list[dict] = []
        self.unpatched: dict[str, pd.DataFrame] = {}

    def wrap(self, orig):
        def read_prices_krx(con, *, cols=None, table="daily_bars_adjusted", **kw):
            df = orig(con, cols=cols, table=table, **kw)
            return self.apply(df, table)
        return read_prices_krx

    def apply(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        if self.sess is None:
            self.sess = read_session_bars(self.con, start=NXT_START)
        adjusted = table != "daily_bars"
        df = df.copy()
        for col in ("open", "high", "low", "close", "volume", "trade_value"):
            if col in df.columns:                    # DB 정수 열 → 실수(보정값은 조정계수가 곱해진 실수다)
                df[col] = df[col].astype(float)
        out = patch_prices(df, self.sess, adjusted=adjusted)
        late = out["date"] >= pd.Timestamp(NXT_START)
        self.stats.append({"table": table, "cols": ",".join(c for c in df.columns if c not in ("code", "date")),
                           "adjusted": adjusted, "rows_late": int(late.sum()),
                           "patched": int(out.loc[late, "krx_patched"].sum()),
                           "unpatched": int((late & ~out["krx_patched"]).sum())})
        self.unpatched[f"{table}:{self.stats[-1]['cols']}"] = (
            out.loc[late & ~out["krx_patched"], ["code", "date"]].reset_index(drop=True))
        return out.drop(columns="krx_patched")

    @contextlib.contextmanager
    def installed(self):
        """그 모듈들의 ``read_prices`` 이름만 실행 중 보정판으로 바꿨다가 되돌린다(파일은 안 고친다)."""
        mods = (sepa_mod, scg, cvr)
        saved = [m.read_prices for m in mods]
        try:
            for m, o in zip(mods, saved, strict=True):
                m.read_prices = self.wrap(o)
            yield self
        finally:
            for m, o in zip(mods, saved, strict=True):
                m.read_prices = o


# ---------------------------------------------------------------------------
# 사건 master · 대조 풀 · 시뮬
# ---------------------------------------------------------------------------

class World:
    """패널·사건 master·대조 풀·창별 캐시 한 벌. BO 평가는 이 위에서 마스크·시뮬·정합만 한다."""

    def __init__(self, D: Data, h5: pd.DataFrame, band: pd.DataFrame, strata: dict, rel: pd.DataFrame,
                 lo: str, hi: str, t0: float):
        self.D, self.p, self.t0 = D, D.p, t0
        p = D.p
        self.trail = D.feats["ma50"]
        self.c_low_panel = contraction_low_level(p.low, PREREG_MINERVINI["vcp_window"])
        self.book = _Book(p.open, p.high, p.low, p.close, self.trail, p.volume)
        self.c_low = self.c_low_panel.to_numpy(float)
        self.vol_np = p.volume.to_numpy(float)
        self.high = p.high

        # --- 사건 master: 사건 구간 H5 종목-일 전부 ---
        dmask = ((p.days >= pd.Timestamp(lo)) & (p.days <= pd.Timestamp(hi)))
        di, ci = np.nonzero(h5.to_numpy(bool) & np.asarray(dmask)[:, None])
        self.di, self.ci = di, ci
        ms = pd.DataFrame({"code": p.close.columns[ci], "event_day": p.days[di], "day_idx": di, "col_idx": ci})
        ms["market"] = p.market.reindex(ms["code"]).to_numpy()
        f = D.feats
        ms["tt"] = f["tt"].to_numpy(bool)[di, ci]
        ms["rs"] = f["rs"].to_numpy(float)[di, ci]
        ms["breadth"] = f["breadth"].to_numpy(float)[di, ci]
        ms["close"] = p.close.to_numpy(float)[di, ci]
        ms["volume"] = self.vol_np[di, ci]
        ms["rel"] = rel.to_numpy(float)[di, ci]
        ms["F"] = D.flow["F"].astype("float64").to_numpy()[di, ci]
        ep = D.earn
        pr = np.full(p.close.shape, np.nan)
        edi = p.days.get_indexer(pd.to_datetime(ep["date"]))
        eci = p.close.columns.get_indexer(ep["code"])
        ok = (edi >= 0) & (eci >= 0)
        pr[edi[ok], eci[ok]] = ep["pead_rank"].to_numpy(float)[ok]
        ms["pead_rank"] = pr[di, ci]
        del pr
        ms["band"] = band.to_numpy(float)[di, ci]
        for k, panel in strata.items():
            ms[k] = panel.to_numpy(float)[di, ci]
        ms["sid"] = stratum_id(di, ms["market"], ms["band"], ms["q_mom5"], ms["adv2"], ms["q_ret5_3"])
        self.master = ms
        self.cols = {k: ms[k].to_numpy() for k in ("event_day", "tt", "rs", "close", "volume", "breadth",
                                                    "pead_rank", "rel", "F")}
        self.win: dict[int, dict] = {}
        _log(f"사건 master(H5 종목-일, {lo}~{hi}) {len(ms):,}", t0)

        # --- 대조 풀: 같은 종목-일 전부를 같은 프로토콜로(추격 금지는 사후) ---
        pairs = pd.DataFrame({"code": ms["code"], "event_day": ms["event_day"], "pivot": np.nan,
                              "pdi": di, "pci": ci})
        ct, sk = control_minervini(p.open, p.high, p.low, p.close, self.trail, pairs, volume=p.volume,
                                   market=p.market, contraction_low=self.c_low_panel, return_skipped=True,
                                   chase=None, **SIM_RULES)
        pool = ct[["code", "event_day", "ret", "market", "entry_px", "hold_days", "exit_reason", "pdi", "pci"]].rename(
            columns={"event_day": "date"})
        pool["date"] = pd.to_datetime(pool["date"])
        pdi, pci = pool["pdi"].to_numpy(int), pool["pci"].to_numpy(int)
        pool["band"] = band.to_numpy(float)[pdi, pci]
        for k, panel in strata.items():
            pool[k] = panel.to_numpy(float)[pdi, pci]
        pool["sid"] = stratum_id(pdi, pool["market"], pool["band"], pool["q_mom5"], pool["adv2"], pool["q_ret5_3"])
        pool.index = _key(pool["date"], pool["code"])
        self.pool = pool
        self.pool_info = {"pairs": len(pairs), "sims": len(pool), "skips": sk["skip_reason"].value_counts().to_dict()}
        _log(f"대조 풀: 쌍 {len(pairs):,} → 시뮬 {len(pool):,}", t0)

    def window(self, w: int) -> dict:
        """피벗 창 ``w`` 의 사건 피벗·거래량 평균과 대조 풀의 피벗(추격 금지용) — 창별 캐시."""
        if w not in self.win:
            piv = pivot_level(self.high, w).to_numpy(float)
            vavg = self.p.volume.shift(1).rolling(w, min_periods=w).mean().to_numpy(float)
            self.win[w] = {"pivot": piv[self.di, self.ci], "vol_avg": vavg[self.di, self.ci],
                           "pool_pivot": piv[self.pool["pdi"].to_numpy(int), self.pool["pci"].to_numpy(int)]}
            del piv, vavg
        return self.win[w]

    def events(self, cfg: dict) -> pd.DataFrame:
        wc = self.window(int(cfg["pivot_window"]))
        cols = {**self.cols, "pivot": wc["pivot"], "vol_avg": wc["vol_avg"]}
        mk = event_mask(cols, cfg)
        ev = self.master.loc[mk, ["code", "event_day", "market", "day_idx", "col_idx", "band", "q_mom5", "adv2",
                                  "q_ret5_3", "sid", "rel", "F", "pead_rank"]].copy()
        ev["pivot"] = wc["pivot"][mk]
        return ev.reset_index(drop=True)

    def pool_for(self, cfg: dict) -> pd.DataFrame:
        if cfg["chase"] is None:
            return self.pool
        ok = chase_ok(self.pool["entry_px"], self.window(int(cfg["pivot_window"]))["pool_pivot"], cfg["chase"])
        return self.pool[ok]

    def simulate(self, ev: pd.DataFrame, cfg: dict) -> "tuple[pd.DataFrame, dict]":
        """사건 → 트레이드(``minervini_exits._run`` 을 캐시한 ``_Book`` 위에서 — ``simulate_minervini`` 와 같은 규칙)."""
        if ev.empty:
            return pd.DataFrame(), {}
        e = ev.copy()
        e["contraction_low"] = self.c_low[e["day_idx"].to_numpy(int), e["col_idx"].to_numpy(int)]
        t, sk = _run(self.book, e, no_overlap=True, rules={**SIM_RULES, "chase": cfg["chase"]})
        skips = sk["skip_reason"].value_counts().to_dict() if len(sk) else {}
        if t.empty:
            return t, skips
        for col in ("event_day", "entry_day", "exit_day"):
            t[col] = pd.to_datetime(t[col])
        t["cost"] = _costs(t)
        t["net"] = t["ret"] - t["cost"]
        t.index = _key(t["event_day"], t["code"])
        return t, skips

    def match(self, t: pd.DataFrame, cfg: dict) -> dict:
        """모멘텀 정합(core ``matched_control``·``matched_alpha`` — ``sepa_momentum.matched_on``). 풀은 트레이드가 있는
        층 키로만 미리 좁힌다(그 밖 층은 뽑히지 않으므로 결과가 같다). 트레이드별 정합 초과·후보 수를 붙인다."""
        pool = self.pool_for(cfg)
        sids = t["sid"].to_numpy(np.int64)
        sub = pool[pool["sid"].isin(np.unique(sids[sids >= 0]))]
        m = matched_on(t, sub, STRATA)
        ctl = m["ctl"]
        if len(ctl):
            cret = sub.loc[ctl["control_idx"].to_numpy(), "ret"].to_numpy(float)
            cm = pd.Series(cret).groupby(ctl["trade_idx"].to_numpy()).mean()
            t["excess"] = t["ret"].to_numpy(float) - cm.reindex(t.index).to_numpy(float)
        else:
            t["excess"] = np.nan
        t["cand"] = candidate_counts(m["t"], sub, STRATA).to_numpy()
        return m


# ---------------------------------------------------------------------------
# 칸 숫자
# ---------------------------------------------------------------------------

def period_masks(t: pd.DataFrame) -> dict[str, pd.Series]:
    e = t["entry_day"]
    h = pd.Timestamp(TRAIN_HI)
    # TRAIN = 진입 ∧ 청산 < 2022-01-01(감사 반영 purge). 진입 < 2022 ∧ 청산 ≥ 2022 는 TRAIN·미관측 어느 쪽도 아니다(전체에만).
    return {"TRAIN": (e < h) & (t["exit_day"] < h), "OOS": e >= h,
            "전체": pd.Series(True, index=t.index)}


def band_masks(t: pd.DataFrame) -> dict[str, pd.Series]:
    return {"H5": pd.Series(True, index=t.index), "유동": t["band"] == 1, "중소형": t["band"] == 0}


def block_stats(t: pd.DataFrame, m: dict, mask: pd.Series, n_dsr: int) -> dict:
    """부분집합(기간 × 띠) 숫자: 서술(``convergence_run.arm_stats`` — 1배 비용 순수익) + 트레이드당 alpha(짝 그대로
    ``sub_alpha``) + 보유일당 alpha(주: Σ초과/Σ보유일, 날짜군집 CI · 참고: 평균(초과/보유일)) + 얇은 층 비율."""
    sub = t[mask.to_numpy()]
    if sub.empty:
        return {"n": 0}
    s = arm_stats(sub, n_dsr)
    a = m["ma"] if bool(mask.all()) else sub_alpha(m, mask)
    s.update(_alpha("a_", a))
    x = sub["excess"].to_numpy(float)
    h = sub["hold_days"].to_numpy(float)
    ok = np.isfinite(x)
    s["pd_alpha"], s["pd_lo"], s["pd_hi"] = ratio_ci(x[ok], h[ok], sub["event_day"].to_numpy()[ok])
    s["pd_mean_ratio"] = float(np.mean(x[ok] / h[ok])) if ok.any() else np.nan
    s["hold_mean"] = float(h.mean())
    s["thin3"] = float((sub["cand"] < THIN_MIN).mean())
    s["cand0"] = float((sub["cand"] == 0).mean())
    return s


def light_stats(t: pd.DataFrame) -> dict:
    """BO 40 config 용 점추정만(부트스트랩 없음): 기간 × 띠별 n·정합 초과 평균·Σ초과/Σ보유일."""
    out = {}
    if t is None or t.empty:
        return out
    for pn, pm in period_masks(t).items():
        for bn, bm in band_masks(t).items():
            sub = t[(pm & bm).to_numpy()]
            x = sub["excess"].to_numpy(float)
            ok = np.isfinite(x)
            out[(pn, bn)] = {"n": len(sub), "n_m": int(ok.sum()),
                             "alpha": float(x[ok].mean()) if ok.any() else np.nan,
                             "pd": float(x[ok].sum() / sub["hold_days"].to_numpy(float)[ok].sum()) if ok.any() else np.nan,
                             "net": float(sub["net"].mean()) if len(sub) else np.nan}
    return out


class Cell:
    """한 칸의 사건·트레이드·정합·숫자."""

    def __init__(self, W: World, name: str, cfg: dict, n_dsr: int, *, full: bool = True,
                 periods: "tuple[str, ...]" = ("TRAIN", "OOS", "전체")):
        self.name, self.cfg = name, cfg
        ev = W.events(cfg)
        self.n_events = len(ev)
        self.t, self.skips = W.simulate(ev, cfg)
        self.m = W.match(self.t, cfg) if not self.t.empty else None
        self.blocks: dict = {}
        if full and self.m is not None:
            pm, bm = period_masks(self.t), band_masks(self.t)
            for pn in periods:
                for bn in bm:
                    self.blocks[(pn, bn)] = block_stats(self.t, self.m, pm[pn] & bm[bn], n_dsr)

    def eff_input(self, per_day: bool) -> tuple:
        """축소용 (num, den, clusters) — 짝 있는 전 기간 H5 트레이드."""
        if self.t.empty:
            return (np.empty(0), np.empty(0), np.empty(0, dtype="datetime64[ns]"))
        x = self.t["excess"].to_numpy(float)
        ok = np.isfinite(x)
        den = self.t["hold_days"].to_numpy(float)[ok] if per_day else np.ones(int(ok.sum()))
        return (x[ok], den, self.t["event_day"].to_numpy()[ok])


def cell_ledger_config(cfg: dict, family: str, name: str) -> dict:
    """원장 지문용 전체 config."""
    c = {"design": DESIGN_DOC, "family": family, "cell": name, "universe": UNIVERSE_DESC, "prices": PRICE_DESC,
         "event_lo": cfg["event_lo"], "event_hi": cfg["event_hi"],
         "trend_template": cfg["tt"], "rs_min": cfg["rs_min"], "pivot_breakout": cfg["breakout"],
         "pivot_window": cfg["pivot_window"], "vol_mult": cfg["vol_mult"], "breadth_min": cfg["breadth_min"],
         "pead_min": cfg["pead_min"], "pead_population": f"그날 위생 통과 ∧ ADV20 ≥ {PEAD_POP_FLOOR:,.0f} (백만원)",
         "rel_max": cfg["rel_max"], "rel": "rng=(max high[t−10..t−1]/min low[t−10..t−1]−1) ÷ median(rng[t−260..t−11], ≥120개)",
         "flow_F": "F=1 (flow_leg, 거래대금 가중)" if cfg["flow"] else None, "chase": cfg["chase"],
         "exit": "minervini_exits.PREREG_MINERVINI 기본값(VCP 문 없음)",
         **{k: v for k, v in PREREG_MINERVINI.items() if k not in ("chase",)},
         "matched": MATCH_DESC, "cost": "korea.cost_rt(청산일, 시장) 1배 · 2배", "train_hi": TRAIN_HI}
    return c


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(--out 을 탐색 디렉터리 밖으로)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS, help="디버그 전용 — BO 시행 수 축소")
    ap.add_argument("--out", default=OUT_DIR, help="결과·원장 디렉터리(기본 = 탐색 디렉터리)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.event_lo or args.event_hi or args.n_trials != N_TRIALS) and not debug:
        ap.error("구간·시행 수 축소는 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    os.makedirs(out_dir, exist_ok=True)
    lo = max(EVENT_LO, args.event_lo) if args.event_lo else EVENT_LO
    hi = min(EVENT_HI, args.event_hi) if args.event_hi else EVENT_HI
    n_trials = args.n_trials

    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()
    fams = loo_families(args.event_lo, args.event_hi)

    # --- 원장: 하나씩 빼기 19칸을 결과 전에 적는다(BO 는 평가 직전, F 추가 칸은 그 평가 직전) ---
    for fam, cells in fams.items():
        for name, cfg in cells.items():
            record_trial(LABEL, cell_ledger_config(cfg, f"loo: {fam}", name), logs_dir=out_dir,
                         note=f"하나씩 빼기 {fam}: {name}")
    n_loo = count_trials(LABEL, logs_dir=out_dir)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
    n_dsr = sum(n_prior.values()) + n_trials                   # 사용자 지시: 계열 원장 누적 + 40
    n_dsr_all = sum(n_prior.values()) + n_loo + n_trials + 1  # 참고: + 하나씩 빼기 19 + F 추가 1
    n_dsr_plan = sum(PLANNED_EVALS.values()) + n_trials        # 감사 반영: 계열 계획 평가 수 + 이 BO
    _log(f"원장 {LABEL}: {n_loo}칸 · " + " · ".join(f"{k} {v}" for k, v in n_prior.items()) + f" → DSR N {n_dsr}", t0)

    load_env_db()
    con = connect(db_default())
    patch = KrxPatch(con)
    with patch.installed():
        D = Data(con, t0)
        raw_close = raw_close_panel(con, D.p)
    p = D.p
    _log("NXT 보정: " + " · ".join(f"{s['table']}({s['cols']}) 늦은 행 {s['rows_late']:,} 보정 {s['patched']:,} "
                                    f"미보정 {s['unpatched']:,}" for s in patch.stats), t0)

    # --- 유니버스 H5 = U_s ∪ main(분리) ---
    h5 = universe_mask(p.close, p.trade_value, D.meta, raw_close=raw_close, adv_floor=PEAD_POP_FLOOR)
    u_s = universe_mask(p.close, p.trade_value, D.meta, raw_close=raw_close, adv_floor=ADV_LO, adv_cap=ADV_HI)
    del raw_close
    main_u = D.uni.astype(bool)
    assert bool(((u_s | main_u) == h5).to_numpy().all()) and not bool((u_s & main_u).to_numpy().any()), "U_s ∪ main ≠ H5"
    band = pd.DataFrame(np.where(main_u.to_numpy(), 1.0, np.where(u_s.to_numpy(), 0.0, np.nan)),
                        index=p.days, columns=p.close.columns)
    repoint_pead(D, h5)
    unp = patch.unpatched.get("daily_bars_adjusted:open,high,low,close,volume,trade_value", pd.DataFrame())
    unp_h5 = 0
    if len(unp):
        udi, uci = p.days.get_indexer(pd.to_datetime(unp["date"])), p.close.columns.get_indexer(unp["code"])
        ok = (udi >= 0) & (uci >= 0)
        unp_h5 = int(h5.to_numpy(bool)[udi[ok], uci[ok]].sum())
    _log(f"H5 종목-일 {int(h5.to_numpy().sum()):,} (유동 {int(main_u.to_numpy().sum()):,} · 중소형 "
         f"{int(u_s.to_numpy().sum()):,}) · 미보정 늦은 행 중 H5 {unp_h5:,}", t0)

    # --- 층 패널(띠·시장·날짜 안) ---
    bands = {1: main_u, 0: u_s}
    mom = momentum_12_1(p.close)
    strata = {"q_mom5": band_strata(mom, bands, p.market, n_q=MOM_Q),
              "adv2": band_strata(D.adv20, bands, p.market, n_q=None),
              "q_ret5_3": band_strata(D.ret5, bands, p.market, n_q=RET5_Q)}
    del mom
    _, rel = relative_contraction(p.high, p.low)
    _log("층·rel 패널", t0)

    W = World(D, h5, band, strata, rel, lo, hi, t0)
    del rel
    # 기준 돌파(창 50·거래량 1.5×)가 뼈대의 pivot_breakout 과 같은지 확인
    wc = W.window(50)
    brk_ref = D.feats["breakout"].to_numpy(bool)[W.di, W.ci]
    with np.errstate(invalid="ignore"):
        brk_me = (W.cols["close"] > wc["pivot"]) & (wc["vol_avg"] > 0) & (W.cols["volume"] >= 1.5 * wc["vol_avg"])
    assert bool((brk_ref == brk_me).all()), "돌파 성분 분해가 pivot_breakout 과 다르다"

    # --- 1. 하나씩 빼기 ---
    loo: dict[str, dict[str, Cell]] = {}
    for fam, cells in fams.items():
        loo[fam] = {}
        periods = ("TRAIN", "OOS", "전체") if fam == FAM_NOF else ("전체",)
        for name, cfg in cells.items():
            c = Cell(W, name, cfg, n_dsr, periods=periods)
            loo[fam][name] = c
            b = c.blocks.get(("전체", "H5"), {"n": 0})
            _log(f"[{fam}] {name}: 사건 {c.n_events:,} · n {b['n']:,} · α {_pct(b.get('a_alpha', np.nan))}", t0)

    # --- 3. 베이지안 축소(조건 효과 = 기준 α − (기준 − 조건) α) ---
    shrink = {}
    for fam, cells in loo.items():
        base = cells[BASE_NAME]
        for per_day in (False, True):
            keys, deltas, ses, raws = [], [], [], []
            for name, c in cells.items():
                if name == BASE_NAME:
                    continue
                d = diff_boot(base.eff_input(per_day), c.eff_input(per_day))
                keys.append(name[2:])
                deltas.append(d["point"])
                ses.append(d["se"])
                raws.append(d)
            df = eb_shrink(deltas, ses)
            df.insert(0, "cond", keys)
            df["raw_lo"] = [r["lo"] for r in raws]
            df["raw_hi"] = [r["hi"] for r in raws]
            shrink[(fam, per_day)] = df
    _log("베이지안 축소", t0)

    # --- 2. BO (TRAIN 전용) ---
    trials: list[dict] = []

    def simulate(params: dict):
        cfg = bo_config(params, args.event_lo, args.event_hi)
        record_trial(LABEL, cell_ledger_config(cfg, "bo", "BO"), logs_dir=out_dir,
                     note=f"BO 시행 {len(trials)} (seed {BO_SEED}, TPE)")
        c = Cell(W, f"BO {len(trials)}", cfg, n_dsr, full=False)
        t = c.t
        rets = t["excess"].to_numpy(float) if not t.empty else np.empty(0)
        ed = t["entry_day"].dt.strftime("%Y-%m-%d").to_numpy() if not t.empty else np.empty(0, dtype="<U10")
        xd = t["exit_day"].dt.strftime("%Y-%m-%d").to_numpy() if not t.empty else np.empty(0, dtype="<U10")
        rets = purge_train(rets, ed, xd)
        obj, n_tr = bo_objective_value(rets, ed)
        trials.append({"i": len(trials), "params": dict(params), "cfg": cfg, "cell": c, "obj": obj, "n_train": n_tr,
                       "light": light_stats(t), "n_events": c.n_events})
        _log(f"BO {len(trials) - 1}: {params} · n {len(t):,} · TRAIN n {n_tr} · 목적 {obj:+.5f}", t0)
        return rets, ed

    best_params, best_value = mini_bo(simulate, SPACE, None, TRAIN_HI, n_trials=n_trials, seed=BO_SEED,
                                      floor=TRADE_FLOOR)
    bi = next(tr["i"] for tr in trials if tr["params"] == best_params)
    assert abs(trials[bi]["obj"] - best_value) < 1e-12, "목적값 재계산 불일치"
    best = trials[bi]["cell"]
    pm, bm = period_masks(best.t), band_masks(best.t)
    for pn in ("TRAIN", "OOS", "전체"):
        for bn in bm:
            best.blocks[(pn, bn)] = block_stats(best.t, best.m, pm[pn] & bm[bn], n_dsr)
    _log(f"BO 최적 시행 {bi}: {best_params} · 목적 {best_value:+.5f}", t0)

    # --- F 추가 칸: 최적 ∧ F=1, 2024-02-19+ ---
    fcfg = {**trials[bi]["cfg"], "flow": True, "event_lo": max(FLOW_LO, lo)}
    record_trial(LABEL, cell_ledger_config(fcfg, "bo+F", "BO 최적 ∧ F=1"), logs_dir=out_dir,
                 note="BO 최적 config ∧ F=1 (2024-02-19+)")
    fcell = Cell(W, "BO 최적 ∧ F=1", fcfg, n_dsr, periods=("전체",))
    n_led = count_trials(LABEL, logs_dir=out_dir)

    # --- 과최적: 순위 상관·PBO·DSR ---
    tab = pd.DataFrame([{"i": tr["i"], **tr["params"], "obj": tr["obj"], "n_train": tr["n_train"],
                         "n_events": tr["n_events"], "n": len(tr["cell"].t),
                         **{f"{k[0]}_{k[1]}_{f}": v[f] for k, v in tr["light"].items() for f in ("n", "alpha", "pd", "net")}}
                        for tr in trials])
    for col in ("OOS_H5_alpha", "OOS_유동_alpha", "OOS_중소형_alpha", "TRAIN_H5_alpha", "OOS_H5_pd"):
        if col not in tab:
            tab[col] = np.nan
    rho = {}
    for lab, a, b in (("주: TRAIN 목적값 vs OOS α(H5)", "obj", "OOS_H5_alpha"),
                      ("TRAIN α vs OOS α (H5)", "TRAIN_H5_alpha", "OOS_H5_alpha"),
                      ("TRAIN 목적값 vs OOS α(유동)", "obj", "OOS_유동_alpha"),
                      ("TRAIN 목적값 vs OOS α(중소형)", "obj", "OOS_중소형_alpha"),
                      ("TRAIN 목적값 vs OOS α/보유일(H5)", "obj", "OOS_H5_pd")):
        ok = np.isfinite(tab[a].astype(float)) & np.isfinite(tab[b].astype(float))
        rho[lab] = (spearman(tab.loc[ok, a].astype(float), tab.loc[ok, b].astype(float)) if ok.sum() > 2 else np.nan,
                    int(ok.sum()))
    months = pd.period_range(pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=7), freq="M")
    M = pd.DataFrame({tr["i"]: monthly_excess(tr["cell"].t, months) for tr in trials})
    n_blocks = PBO_BLOCKS if len(M) >= PBO_BLOCKS else max(2, len(M) // 2 * 2)
    pbo = {mt: pbo_cscv(M.to_numpy(), n_blocks=n_blocks, metric=mt, std_floor=PBO_STD_FLOOR) for mt in ("sharpe", "mean")}
    dsr = {}
    bt = best.t
    oos = bt["entry_day"] >= pd.Timestamp(TRAIN_HI)
    for lab, x in (("순수익 1배 · 전 구간", bt["net"]), ("순수익 1배 · OOS", bt.loc[oos, "net"]),
                   ("정합 초과 · 전 구간", bt["excess"]), ("정합 초과 · OOS", bt.loc[oos, "excess"])):
        v = x.to_numpy(float)
        v = v[np.isfinite(v)]
        dsr[lab] = {"n": len(v), **{N: deflated_sharpe_from_sample(v, N) if len(v) > 2 else None
                                    for N in (n_dsr, n_dsr_all, n_dsr_plan)}}
    _log("순위 상관·PBO·DSR", t0)

    # --- 판단 규칙 숫자 ---
    def p_of(fam, per_day):
        df = shrink[(fam, per_day)]
        return dict(zip(df["cond"], df["p_pos"], strict=True))

    ob = best.blocks
    dec = decision_numbers(p_of(FAM_NOF, False), ob.get(("OOS", "유동"), {}).get("a_alpha", np.nan),
                           ob.get(("OOS", "중소형"), {}).get("a_alpha", np.nan), rho["주: TRAIN 목적값 vs OOS α(H5)"][0])

    elapsed = time.time() - t0
    L = render(locals_for_render(**locals()))
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    save_csvs(out_dir, loo, trials, best, fcell, tab, shrink)
    print(f"\n=== 제약 완화 — {elapsed / 60:.1f}분 · DSR N {n_dsr} ===")
    print(f"BO 최적 {bi}: {best_params} · 목적 {best_value:+.5f}")
    for pn, bn in (("OOS", "유동"), ("OOS", "중소형")):
        b = ob.get((pn, bn), {"n": 0})
        print(f"  {pn} {bn}: n {b['n']} · α {_pct(b.get('a_alpha', np.nan))}")
    print(f"  순위 상관(주) {rho['주: TRAIN 목적값 vs OOS α(H5)'][0]:+.3f} · PBO {pbo['sharpe']['pbo']:.3f}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def locals_for_render(**kw) -> dict:
    keep = ("loo", "shrink", "trials", "tab", "best", "bi", "best_params", "best_value", "fcell", "rho", "pbo", "dsr",
            "dec", "n_dsr", "n_dsr_all", "n_dsr_plan", "n_prior", "n_led", "n_loo", "patch", "unp_h5", "W", "p", "h5", "main_u", "u_s",
            "elapsed", "debug", "out_dir", "lo", "hi", "n_trials", "n_blocks", "M")
    return {k: kw[k] for k in keep}


def save_csvs(out_dir, loo, trials, best, fcell, tab, shrink) -> None:
    rows = []
    for fam, cells in loo.items():
        for name, c in cells.items():
            for (pn, bn), s in c.blocks.items():
                rows.append({"family": fam, "cell": name, "period": pn, "band": bn, "n_events": c.n_events,
                             **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    for c, fam in ((best, "BO 최적"), (fcell, "BO 최적 ∧ F")):
        for (pn, bn), s in c.blocks.items():
            rows.append({"family": fam, "cell": c.name, "period": pn, "band": bn, "n_events": c.n_events,
                         **{k: v for k, v in s.items() if not isinstance(v, dict)}})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "relax_cells.csv"), index=False)
    tab.to_csv(os.path.join(out_dir, "relax_bo.csv"), index=False)
    pd.concat([df.assign(family=fam, per_day=pdy) for (fam, pdy), df in shrink.items()]).to_csv(
        os.path.join(out_dir, "relax_shrink.csv"), index=False)
    tr = []
    for fam, cells in loo.items():
        for name in (BASE_NAME,):
            t = cells[name].t
            if not t.empty:
                tr.append(t.assign(cell=f"{fam} {name}"))
    for c in (best, fcell):
        if not c.t.empty:
            tr.append(c.t.assign(cell=c.name))
    if tr:
        pd.concat(tr, ignore_index=True).to_csv(os.path.join(out_dir, "relax_trades.csv"), index=False)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _a(s: dict, pre: str = "a_") -> str:
    a = s.get(f"{pre}alpha", np.nan)
    if a is None or not np.isfinite(a):
        return "nan"
    return f"{_pct(a)} [{_pct(s.get(f'{pre}lo'))}, {_pct(s.get(f'{pre}hi'))}]"


def _pd(s: dict) -> str:
    a = s.get("pd_alpha", np.nan)
    if a is None or not np.isfinite(a):
        return "nan"
    return f"{a * 1e4:+.1f}bp [{s.get('pd_lo', np.nan) * 1e4:+.1f}, {s.get('pd_hi', np.nan) * 1e4:+.1f}]"


def _bp(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 1e4:+.1f}bp"


def _thin(s: dict) -> str:
    v = s.get("thin3", np.nan)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.0%}" + (" ⚠" if v > THIN_WARN else "")


def _cfg_str(c: dict) -> str:
    parts = []
    parts.append("TT" if c["tt"] else "TT 없음")
    parts.append(f"RS≥{c['rs_min']:.1f}" if c["rs_min"] is not None else "RS 없음")
    parts.append(f"돌파(창 {c['pivot_window']})" if c["breakout"] else "돌파 없음")
    parts.append(f"거래량≥{c['vol_mult']:.2f}×" if c["vol_mult"] is not None else "거래량 없음")
    parts.append(f"breadth>{c['breadth_min']:.2f}" if c["breadth_min"] is not None else "breadth 없음")
    parts.append(f"PEAD≥{c['pead_min']:.2f}" if c["pead_min"] is not None else "PEAD 없음")
    parts.append(f"rel≤{c['rel_max']:.2f}" if c["rel_max"] is not None else "rel 없음")
    parts.append(f"추격 {c['chase']:.0%}" if c["chase"] is not None else "추격 없음")
    if c["flow"]:
        parts.append("F=1")
    return " ∧ ".join(parts)


DETAIL_HEAD = ("| 칸 | 사건 | n | 손절 비율 | 승률 | 이익/손실 | 기대값 R(순) | 순평균 1배 | 순평균 2배 | α/트레이드 [CI] | "
               "α/보유일 Σ/Σ (주) [CI] | 평균(초과/보유일) | 보유 평균 | 짝 없음 | 후보<3 |")
DETAIL_SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def detail_row(name: str, n_events, s: dict) -> str:
    if not s or not s.get("n"):
        return f"| {name} | {n_events} | 0 | | | | | | | | | | | | |"
    return (f"| {name} | {n_events} | {s['n']} | {s['stop_rate']:.1%} | {s['win']:.1%} | {_f(s['gl_ratio'])} | "
            f"{_f(s['r_net'], '{:+.3f}')} | {_pct(s['net_mean'])} | {_pct(s['net2_mean'])} | {_a(s)} | {_pd(s)} | "
            f"{_bp(s['pd_mean_ratio'])} | {s['hold_mean']:.1f} | {s.get('a_unmatched', 0)} | {_thin(s)} |")


def short(s: dict) -> str:
    if not s or not s.get("n"):
        return "0 | | "
    return f"{s['n']} | {_a(s)} | {_pd(s)}"


def render(v: dict) -> list[str]:
    loo, shrink, trials, tab = v["loo"], v["shrink"], v["trials"], v["tab"]
    best, bi, fcell, rho, pbo, dsr, dec = v["best"], v["bi"], v["fcell"], v["rho"], v["pbo"], v["dsr"], v["dec"]
    W, p = v["W"], v["p"]
    L = [f"# 제약 완화 — 하나씩 빼기 · BO(TRAIN 전용) · 베이지안 축소 — 실행 결과 {date.today().isoformat()} "
         "(러너 자동 생성, 탐색 — 판정 아님)", "",
         "설계 `RELAX.md`(d04e983 + 감사 반영 b4a3532, 둘 다 결과 보기 전) 의 유니버스·가격 보정·청산·대조·칸·탐색 공간·목적·평가·과최적 측정·축소·판단 규칙을 "
         "그대로 계산한 숫자다. 탐색이므로 여기 숫자는 후보를 만들 뿐이다 — 확인은 새 사전등록·전진 페이퍼다. 판단 규칙 표의 ✓/✗ 는 "
         "미리 적은 문턱과의 **산술 비교**일 뿐이다. 재현: `uv run --with optuna --extra pg python research/experiments/relax_run.py`.",
         "",
         f"> **'{UNSEEN}' = 진입 ≥ 2022-01-01 — BO 목적함수 미관측 구간일 뿐 표본 밖이 아니다.** 유동주 2022~26 은 이 계열의 모든 "
         "시험에서 봤고 중소형도 수렴 시험 B·D 에서 봤다(RELAX.md 감사 반영). 진짜 표본 밖은 전진 페이퍼뿐이다. "
         "TRAIN 은 진입 ∧ 청산 < 2022-01-01(purge) — 진입 2021 ∧ 청산 2022+ 트레이드는 어느 쪽에도 없고 '전체' 에만 있다.",
         ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 사건 구간 {v['lo']}~{v['hi']}, BO {v['n_trials']}회, 출력·원장 `{v['out_dir']}`. "
              "측정으로 쓰지 않는다.", ""]

    # --- 판단 규칙 ---
    ob = best.blocks
    liq, sm = ob.get(("OOS", "유동"), {"n": 0}), ob.get(("OOS", "중소형"), {"n": 0})
    r0, rn = rho["주: TRAIN 목적값 vs OOS α(H5)"]

    def mk(ok):
        return "✓" if ok else "✗"

    sp = shrink[(FAM_NOF, False)]
    L += ["## 판단 규칙 숫자 (RELAX.md — 미리 적은 문턱과 산술 비교, 판정 아님)", "",
          "| 규칙 | 문턱 | 숫자 | 산술 |", "|---|---|---|---|",
          "| 1. 표본만 깎는 조건 | 축소 후 P(효과 > 0) < 0.5 (주 = F 없음판, 전 구간, 트레이드당 α) | "
          + " · ".join(f"{r.cond} {r.p_pos:.2f}" for r in sp.itertuples()) + " | "
          + ("해당: " + ", ".join(dec["drop"]) if dec["drop"] else "해당 없음") + " |",
          f"| 2a. 최적 config OOS 유동 α 점추정 > 0 | > 0 | n {liq['n']} · {_pct(liq.get('a_alpha', np.nan))} | "
          f"{mk(dec['liq_pos'])} |",
          f"| 2b. 최적 config OOS 중소형 α 점추정 > 0 | > 0 | n {sm['n']} · {_pct(sm.get('a_alpha', np.nan))} | "
          f"{mk(dec['small_pos'])} |",
          f"| 2c. TRAIN→OOS 순위 상관(40 config) > 0 | > 0 | 스피어만 {_f(r0, '{:+.3f}')} (config {rn}) | {mk(dec['rho_pos'])} |",
          f"| 2. 셋 다 | 2a ∧ 2b ∧ 2c | | **{'셋 다 충족' if dec['all_three'] else '셋 다 충족 아님'}** |", "",
          "규칙 1 참고(다른 읽기): " + " / ".join(
              f"{fam} {'보유일당' if pdy else '트레이드당'}: "
              + (", ".join(f"{r.cond}" for r in df.itertuples() if np.isfinite(r.p_pos) and r.p_pos < 0.5) or "해당 없음")
              for (fam, pdy), df in shrink.items() if not (fam == FAM_NOF and not pdy)),
          "", "α 의 부호는 트레이드당(평균 초과)과 보유일당 Σ/Σ 가 항상 같다(보유일 > 0) — 규칙 2 는 어느 쪽으로 읽어도 같다.", ""]

    # --- 표본 ---
    L += ["## 표본·가격 보정·대조 풀", "",
          f"- 가격 패널 {p.days[0].date()}~{p.days[-1].date()} · 사건 구간 {v['lo']}~{v['hi']} · 실행 {v['elapsed'] / 60:.1f}분",
          "- **NXT 보정(2026-09-14+, `krx_session.patch_prices`):** " + " · ".join(
              f"`{s['table']}`({'조정' if s['adjusted'] else '원값'}, {s['cols']}) 늦은 행 {s['rows_late']:,} → 보정 "
              f"**{s['patched']:,}** · 미보정 **{s['unpatched']:,}**" for s in v["patch"].stats)
          + f" · 미보정 조정 행 중 H5 종목-일 {v['unp_h5']:,} (분봉 없는 종목 — 원값 그대로 남는다)",
          f"- H5 종목-일 {int(v['h5'].to_numpy().sum()):,} = 유동 {int(v['main_u'].to_numpy().sum()):,} + 중소형 "
          f"{int(v['u_s'].to_numpy().sum()):,} · 사건 master(사건 구간 H5) {len(W.master):,}",
          f"- 대조 풀(사건 구간 H5 전 종목-일, 추격 금지 전): 쌍 {W.pool_info['pairs']:,} → 시뮬 {len(W.pool):,} (버림 "
          f"{_dist(W.pool_info['skips'], SKIP_ORDER)}) · 층 = {' × '.join(STRATA)}",
          f"- **층 얇기:** 하루 층 칸 = 시장 2 × 띠 2 × 모멘텀 {MOM_Q} × ADV 2 × 5일 수익 {RET5_Q} = "
          f"{2 * 2 * MOM_Q * 2 * RET5_Q} 칸, 대조 풀 하루 종목-일 중앙값 {W.pool.groupby('date').size().median():.0f} → 칸당 평균 "
          f"≈ {W.pool.groupby('date').size().median() / (2 * 2 * MOM_Q * 2 * RET5_Q):.1f} 후보(추격 금지 전). 칸별 '후보<3' 비율은 "
          f"각 표의 마지막 열(> {THIN_WARN:.0%} 면 ⚠). 얇은 층은 복원 추출로 같은 대조를 반복해 뽑는다(편향 아님, 분산↑).",
          f"- 원장 `{v['out_dir']}/{LABEL}/TRIALS.jsonl` **{v['n_led']}** 줄(하나씩 빼기 {v['n_loo']} + BO {len(trials)} + F 추가 1, "
          "같은 지문은 한 줄) · 계열 원장 누적 " + " + ".join(f"{k} {n}" for k, n in v["n_prior"].items())
          + f" = {sum(v['n_prior'].values())}", ""]

    # --- 하나씩 빼기 ---
    for fam, cells in loo.items():
        per = ("TRAIN", "OOS", "전체") if fam == FAM_NOF else ("전체",)
        L += [f"## 1. 하나씩 빼기 — {fam}", "",
              "기준 = " + _cfg_str(cells[BASE_NAME].cfg) + f" · 사건 {cells[BASE_NAME].cfg['event_lo']}~"
              f"{cells[BASE_NAME].cfg['event_hi']}. α 는 비용 전 총수익의 모멘텀 정합 초과, CI 는 날짜군집 95%. "
              "α/보유일 = Σ초과/Σ보유일(주), 평균(초과/보유일)은 참고.", ""]
        for pn in per:
            L += [f"### {pn} · H5 전체", "", DETAIL_HEAD, DETAIL_SEP]
            L += [detail_row(name, c.n_events, c.blocks.get((pn, "H5"), {})) for name, c in cells.items()]
            L.append("")
        L += ["### 띠별 (유동 ≥ 200억 / 중소형 50~200억) — n · α/트레이드 [CI] · α/보유일 [CI]", "",
              "| 칸 | " + " | ".join(f"{pn} {bn} n | α | α/보유일" for pn in per for bn in ("유동", "중소형")) + " |",
              "|---|" + "---|---|---|" * (2 * len(per))]
        for name, c in cells.items():
            L.append(f"| {name} | " + " | ".join(short(c.blocks.get((pn, bn), {})) for pn in per for bn in ("유동", "중소형"))
                     + " |")
        L.append("")
        thin = [(name, c.blocks.get(("전체", "H5"), {}).get("thin3", np.nan)) for name, c in cells.items()]
        bad = [f"{n} {x:.0%}" for n, x in thin if np.isfinite(x) and x > THIN_WARN]
        L += [f"얇은 층(후보 < {THIN_MIN}) 비율이 {THIN_WARN:.0%} 를 넘는 칸: " + (", ".join(bad) if bad else "없음"), ""]

    # --- 축소 ---
    L += ["## 3. 베이지안 축소 (경험적 베이즈, 정규-정규, 사전 평균 0)", "",
          "조건 효과 δ = α(기준) − α(기준 − 조건) — **양수 = 그 조건이 alpha 를 올린다**(RELAX.md 의 \"칸 alpha − 기준 alpha\" 와 "
          "부호가 반대 — 판단 규칙의 \"P(효과 > 0) = 그 조건이 alpha 를 올릴 확률\" 에 맞춘 것, 아래 구현 메모). 표준오차 = 기준·칸 트레이드를 "
          "같은 날짜 군집으로 함께 뽑은 부트스트랩 2,000회. τ² = 주변우도 최대. P = Φ(사후 평균 / 사후 표준편차).", ""]
    for (fam, pdy), df in shrink.items():
        unit = "보유일당 Σ/Σ (bp)" if pdy else "트레이드당 (%)"

        def fmt(x, pdy=pdy):
            return _bp(x) if pdy else _pct(x)

        tau = df["tau2"].iloc[0] if len(df) else np.nan
        tau_s = (f"{math.sqrt(tau) * 1e4:.1f}bp" if pdy else f"{math.sqrt(tau) * 100:.2f}%") if np.isfinite(tau) else "nan"
        L += [f"### {fam} · {unit} · τ̂ = {tau_s}", "",
              "| 조건 | δ 원값 | 표준오차 | 원값 95% CI | 축소 계수 B | 사후 평균 | 사후 90% 구간 | P(효과 > 0) |",
              "|---|---|---|---|---|---|---|---|"]
        for r in df.itertuples():
            L.append(f"| {r.cond} | {fmt(r.delta)} | {fmt(r.se).lstrip('+')} | [{fmt(r.raw_lo)}, {fmt(r.raw_hi)}] | "
                     f"{_f(r.B, '{:.3f}')} | {fmt(r.post_mean)} | [{fmt(r.lo90)}, {fmt(r.hi90)}] | "
                     f"{_f(r.p_pos, '{:.3f}')}{' (τ̂²=0, 퇴화)' if r.degenerate else ''} |")
        L.append("")

    # --- BO ---
    obj = tab["obj"].astype(float).to_numpy()
    run_best = np.maximum.accumulate(obj)
    L += ["## 2. 베이지안 최적화 (TRAIN 진입 < 2022-01-01 전용)", "",
          f"`mini_bo` {len(trials)}회 · TPE seed {BO_SEED} · 목적 = `make_objective`(TRAIN 정합 초과 부트스트랩 2.5% 하단, "
          f"거래수 하한 {TRADE_FLOOR} — 미달은 −1 + n/{TRADE_FLOOR}·10⁻³). 탐색 공간: "
          + " · ".join(f"{k} [{a}, {b}]" for k, (a, b) in SPACE.items()) + ". 추세 템플릿·추격 금지 5%·청산 고정, F 없음.", "",
          f"- 하한 충족 config: {int((tab['n_train'] >= TRADE_FLOOR).sum())}/{len(tab)} · 목적값 중앙값 "
          f"{_f(float(np.median(obj)), '{:+.5f}')} · 최고 {_f(float(obj.max()), '{:+.5f}')} (시행 {v['bi']})",
          "- 최적 config: " + _cfg_str(best.cfg) + " — " + ", ".join(
              f"{k} {val:.4g}" if isinstance(val, float) else f"{k} {val}" for k, val in v["best_params"].items()), "",
          "| 시행 | rs_min | vol_mult | pead_min | rel_max | breadth_min | 창 | 사건 | n | TRAIN n | TRAIN α | 목적값 | 누적 최고 | "
          "OOS n | OOS α H5 | OOS α 유동 | OOS α 중소형 | OOS α/보유일 H5 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, g in enumerate(tab.to_dict("records")):
        L.append(f"| {g['i']}{' ★' if g['i'] == bi else ''} | {g['rs_min']:.1f} | {g['vol_mult']:.2f} | "
                 f"{g['pead_min']:.2f} | {g['rel_max']:.2f} | {g['breadth_min']:.2f} | {g['pivot_window']} | "
                 f"{g['n_events']} | {g['n']} | {g['n_train']} | "
                 f"{_pct(g.get('TRAIN_H5_alpha', np.nan))} | {g['obj']:+.5f} | {run_best[k]:+.5f} | "
                 f"{int(g.get('OOS_H5_n', 0) or 0)} | {_pct(g.get('OOS_H5_alpha', np.nan))} | "
                 f"{_pct(g.get('OOS_유동_alpha', np.nan))} | {_pct(g.get('OOS_중소형_alpha', np.nan))} | "
                 f"{_bp(g.get('OOS_H5_pd', np.nan))} |")
    L.append("")

    # --- OOS 평가: 최적 vs 기준 ---
    base = loo[FAM_NOF][BASE_NAME]
    L += ["## 최적 config vs 기준(A − F) — 같은 OOS (진입 ≥ 2022-01-01), 한 번", "", DETAIL_HEAD, DETAIL_SEP]
    for pn, bn in (("OOS", "유동"), ("OOS", "중소형"), ("OOS", "H5"), ("TRAIN", "H5"), ("전체", "H5")):
        L.append(detail_row(f"최적 · {pn} {bn}", best.n_events, best.blocks.get((pn, bn), {})))
        L.append(detail_row(f"기준 · {pn} {bn}", base.n_events, base.blocks.get((pn, bn), {})))
    L.append("")

    # --- F 추가 칸 ---
    fb = loo[FAM_F][BASE_NAME]
    L += [f"## F 추가 칸 — 최적 config ∧ F=1 ({fcell.cfg['event_lo']}~{fcell.cfg['event_hi']})", "", DETAIL_HEAD, DETAIL_SEP]
    for bn in ("H5", "유동", "중소형"):
        L.append(detail_row(f"최적 ∧ F · {bn}", fcell.n_events, fcell.blocks.get(("전체", bn), {})))
        L.append(detail_row(f"(참고) 기준 ∧ F · {bn}", fb.n_events, fb.blocks.get(("전체", bn), {})))
    L.append("")

    # --- 과최적 ---
    L += ["## 과최적 측정", "",
          f"### PBO (CSCV, 월별(진입월) 정합 초과 평균, 거래 없는 달 0, {v['n_blocks']} 조각, 40 config × {len(v['M'])} 달)", "",
          "| 지표 | PBO | 조합 수 | IS 최고 config 의 나머지 조각 성과 중앙값 |", "|---|---|---|---|"]
    for mt, r in pbo.items():
        L.append(f"| {mt} | **{r['pbo']:.3f}** | {r['n_combos']} | {np.median(r['oos_perf']):+.5f} |")
    L += ["", "### TRAIN → OOS 순위 상관 (BO 가 평가한 config 전부)", "", "| 비교 | 스피어만 | config 수 |", "|---|---|---|"]
    L += [f"| {lab} | {_f(r, '{:+.3f}')} | {n} |" for lab, (r, n) in rho.items()]
    plan = " · ".join(f"{k} {n:,}" for k, n in PLANNED_EVALS.items())
    L += ["", "### 최적 config DSR — N 세 가지", "",
          f"- **원장 N** = 계열 원장 누적 {sum(v['n_prior'].values())} + BO {len(trials)} = **{v['n_dsr']}** (사용자 지시)",
          f"- 원장 N(전부) = 원장 N + 하나씩 빼기 {v['n_loo']} + F 추가 1 = {v['n_dsr_all']}",
          f"- **계획 평가 N** = 이 계열 계획 평가 수({plan}) + BO {len(trials)} = **{v['n_dsr_plan']:,}** (감사 반영 — 원장 N 은 "
          "탐색을 크게 과소평가한다)", "",
          "| 표본 | n | 건당 Sharpe | SR0 원장 N | P(SR > SR0) 원장 N | P(SR > SR0) 원장 N(전부) | SR0 계획 N | "
          "P(SR > SR0) 계획 N | P(SR > 0) |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, d in dsr.items():
        a, b, c = d[v["n_dsr"]], d[v["n_dsr_all"]], d[v["n_dsr_plan"]]
        if a is None:
            L.append(f"| {lab} | {d['n']} | | | | | | | |")
            continue
        L.append(f"| {lab} | {d['n']} | {_f(a['observed_sharpe'], '{:+.4f}')} | {_f(a['expected_max_sharpe_h0'], '{:.4f}')} | "
                 f"{_f(a['prob_deflated_sharpe'], '{:.1%}')} | {_f(b['prob_deflated_sharpe'], '{:.1%}')} | "
                 f"{_f(c['expected_max_sharpe_h0'], '{:.4f}')} | {_f(c['prob_deflated_sharpe'], '{:.1%}')} | "
                 f"{_f(a['prob_sharpe_gt0'], '{:.1%}')} |")
    L.append("")

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          "- **가격 보정:** `Data`·`raw_close_panel` 이 부르는 `read_prices`(모듈 `research.signals.sepa`·`sepa_confluence_gate`·"
          "`convergence_run` 의 이름)를 실행 중에만 보정판으로 바꿔 끼운다 — 롱 가격표 단계에서 `patch_prices`(조정 테이블 "
          "`adjusted=True`: 계수 = 행 시가 / 정규장 시가, 원 일봉 `adjusted=False`) 후 패널을 만든다. 그래서 MA·피벗·돌파·거래량·"
          "ADV·유니버스·breadth·RS·rel·청산 전부가 보정 가격에서 나온다. 분봉 없는 종목-일은 원값(오염 가능)으로 남고 수를 적는다. "
          "수급 F·실적은 가격과 무관.",
          "- **유니버스·띠:** H5 = `universe_mask(adv_floor=5000)`, 유동 = 기존 `Data.uni`(ADV20 ≥ 20,000), 중소형 = "
          "`adv_floor=5000, adv_cap=20000` — 실행 중 분할을 assert. PEAD 순위 모집단 = H5(`convergence_run.repoint_pead`).",
          "- **조건 분해:** 피벗 돌파 = 종가 > 직전 창 최고 고가, 거래량 = 거래량 ≥ 배수 × 직전 창 평균(평균 > 0) — "
          "`pivot_breakout` 의 두 성분. 창 50·1.5× 에서 둘의 곱이 `skeleton_features` 의 `breakout` 과 같음을 실행 중 assert. "
          "BO 의 `pivot_window` 는 피벗·거래량 평균 창 둘 다 바꾸고(원 함수와 같다) 추격 금지 피벗도 같은 창이다. "
          "조건이 켜진 칸에서 NaN(RS·breadth·PEAD·rel·F 없음)은 통과하지 않는다 — `pead_min`=0 도 실적 행이 있어야, "
          "`rel_max`=3.0 도 rel 이력이 있어야 통과(설계의 '≈ 조건 없음').",
          "- **'− 추격 금지' 칸:** 트레이드와 대조 둘 다 추격 금지를 끈다(같은 프로토콜). 그 밖의 칸은 대조에도 그 칸의 "
          "피벗 창으로 추격 금지를 건다(사후 거르기 — 대조 시뮬은 겹침 없는 독립 시행이라 결과가 정확히 같다).",
          "- **대조 층:** 날짜 × 시장 × 띠 × 12-1 모멘텀 5분위 × ADV20 2분위 × 직전 5일 수익 3분위. 분위는 **그날 그 띠·그 "
          "시장 안**에서 자른다(유동 트레이드는 유동 대조와만 — 띠별 결과를 쓰려면 짝도 띠 안이어야 한다). 5일 수익 = "
          "`Data.ret5`(정지일 제외 종가). 풀은 트레이드가 있는 층 키로만 미리 좁혀 core `matched_control` 에 넘긴다 — 다른 층은 "
          "뽑히지 않으므로 같은 결과다. alpha 는 비용 전 총수익.",
          "- **TRAIN/OOS:** 경계 `optimization.TRAIN_HI` = 2022-01-01. TRAIN = 진입 < 경계 **∧ 청산 < 경계**(감사 반영 purge — "
          "BO 목적함수에도 같은 규칙: `make_objective` 는 진입일만 보므로 진입 < 경계 ∧ 청산 ≥ 경계 트레이드의 값을 NaN 으로 넘겨 "
          "`split_stats` 가 버리게 한다 — 함수는 그대로). OOS(= 미관측) = 진입 ≥ 경계 ∧ 사건 ≤ 2026-04-01. F 포함판은 전부 "
          "2024+ 라 '전체' 만 적는다.",
          "- **BO 목적의 수익:** 트레이드별 모멘텀 정합 초과(트레이드 총수익 − 짝 5개 평균). 짝 없는 트레이드는 NaN → "
          "`split_stats` 가 뺀다(하한 150 도 짝 있는 TRAIN 트레이드 수). `mini_bo` 그대로(floor=150) — 40 config 의 값은 "
          "`simulate` 콜백에서 같은 식(`split_stats`·`_boot_lower`)으로 다시 재 기록하고 최고값 일치를 assert 한다.",
          "- **순위 상관(주):** BO 가 실제로 순위를 매긴 TRAIN 목적값 vs OOS 트레이드당 α(H5). 하한 미달 config 도 페널티 값으로 "
          "순위에 들어간다. 다른 읽기(TRAIN α, 유동·중소형, 보유일당)는 참고.",
          "- **PBO:** 40 config × 진입월 정합 초과 평균(`sepa_explore.monthly_excess`, 거래 없는 달 0), 전 구간, 16조각, "
          "sharpe·mean 둘 다(탐색 러너와 같다).",
          "- **DSR:** 원장 N = 계열 원장(sepa_confluence·sepa_explore·minervini_exits·convergence) 누적 + 40, 계획 평가 N = "
          "RELAX.md 감사 반영의 계열 계획 평가 수(GA 9,000 · CMA-ES 49,350 · 워크포워드 86,400 = 144,750) + 40. 참고로 원장 N 에 "
          "하나씩 빼기 19칸·F 추가 1칸을 더한 N 도 적는다. 표본 = 최적 config 의 트레이드(건당 Sharpe), 시행별 Sharpe 표준편차는 "
          "코어 기본(1/√(T−1)).",
          "- **축소의 부호:** RELAX.md 3절은 효과를 '칸 alpha − 기준 alpha' 로 적고 같은 문단에서 P(효과 > 0) 을 '그 조건이 "
          "alpha 를 올린다' 의 확률이라 부른다. 칸 = 기준 − 조건이라 두 문장은 부호가 반대다. 판단 규칙('P < 0.5 → 표본만 깎는 "
          "조건')이 뜻을 갖는 쪽인 **δ = α(기준) − α(기준 − 조건)** 으로 셈한다(결과 보기 전 결정). 표의 '원값'도 이 부호다.",
          "- **축소 모형:** δ_k ~ N(θ_k, s_k²), θ_k ~ N(0, τ²), τ² = 주변우도 최대(≥ 0). 사후 평균 B·δ, 사후 sd √B·s. "
          "τ̂² = 0 이면 사후가 0 한 점 → P = 0.5 로 적고 '퇴화' 표시(규칙 1 문턱 < 0.5 에 걸리지 않는다). 주 = F 없음판·"
          "전 구간·트레이드당. F 포함판·보유일당은 참고로 같은 식. **주의(감사 반영):** 층이 얇고(위 '층 얇기') 칸끼리 트레이드를 "
          "많이 공유해 δ 의 표준오차가 크면 τ̂² 가 0 으로 무너져 모든 조건이 P ≈ 0.5(퇴화)로 나올 수 있다 — 그때 규칙 1 은 "
          "아무 조건도 거르지 못하며, 그것은 '조건이 일하지 않는다' 가 아니라 '이 표본으로는 가를 수 없다' 이다. 조건이 8~9개뿐이라 "
          "τ² 추정 자체도 거칠다.",
          "- **판단 규칙 2 의 α:** 최적 config 의 OOS 띠별 트레이드당 모멘텀 정합 α 점추정(보유일당 Σ/Σ 와 부호가 같다).",
          "- **비용:** 1배 = `korea.cost_rt(청산일, 시장)`, 2배 = 총수익 − 2×그 값. 기대값 R(순) = mean(1배 순수익 / d).", ""]
    # 감사 반영: 진입 ≥ 2022 구간을 'OOS' 라 부르지 않는다(표본 밖이 아니다) — 문서 전체에서 이름을 바꾼다.
    return [line.replace("OOS", UNSEEN) for line in L]


if __name__ == "__main__":
    raise SystemExit(main())
