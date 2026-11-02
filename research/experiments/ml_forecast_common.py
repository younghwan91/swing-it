"""ML 수익률 예측(파이프라인 5단계) — 사전등록 상수와 창 가드. 정본 ``research/logs/ml_forecast/PREREG.md`` 2판(6925e75).

여기 값을 바꾸면 사전등록을 바꾸는 것이다 — 결과를 본 뒤에는 바꾸지 않는다(정정 ML-n 으로만).
EW 부호표·피처 목록은 ``tests/test_ml_forecast_features.py`` 의 해시 회귀가 고정한다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pead_optimizer_c0 import keyed_rng  # noqa: F401 — 이 연구의 난수는 모두 이 키 생성기(PO-1.7 꼴)
from scipy import stats

PREREG_SHA = "6925e75"
LABEL = "ml_forecast"  # 다중검정 원장 라벨(§9)

# ── 창(§1·§2) ────────────────────────────────────────────────────────────────
LOAD_START = "2019-01-02"  # core load_risk_inputs 시작(준비 400일은 자동)
END = "2023-12-28"  # 데이터 끝 — 평가 모드는 이 뒤를 거부한다
SEALED_START = "2024-01-01"  # 봉인 소모 구간 시작
FIT_ONLY_END = "2026-09-11"  # 전진 재적합 전용 상한(§11) — 이 연구 1·2단계에서는 쓰지 않는다
GRID_START = "2019-08-01"  # 북·IC 격자 기준일(B 와 같다)
TEST_YEARS = (2020, 2021, 2022, 2023)

# ── 격자·지평(§2·§4) ──────────────────────────────────────────────────────────
BOOK_STEP = 20
IC_STEP = 5
HOLD = 20  # y = Σ_{s=t+2}^{t+21} u_s
LABEL_LAST = HOLD + 1  # t+21
MIN_PRICE_DAYS = 15  # 창 안 가격 행(정지일 포함)이 이 미만이면 NaN(폐지 예외)

# ── 유니버스(§2) ───────────────────────────────────────────────────────────────
ADV_MIN_WON = 5e9  # core adv_value(원, 20일 평균) ≥ 50억
CAP_TOP = 400
MIN_IC_NAMES = 100  # IC 격자일 유효 종목 하한(§7)

# ── 피처(§3) ───────────────────────────────────────────────────────────────────
FEATURES = ("eps_yoy", "ann_age", "rev21", "hi52", "frgn20", "frgn_own", "tp_rev", "inst20", "vol_shock",
            "size", "beta", "momentum", "resid_vol", "liquidity", "value")
STYLE_FEATURES = FEATURES[9:]  # F10~F15 = core STYLE_NAMES 순서
S_FEATURES = FEATURES[8:]  # 진단 S: F9~F15
EW_SIGNS = {"eps_yoy": 1, "rev21": -1, "hi52": 1, "frgn20": 1, "frgn_own": 1, "tp_rev": 1}  # 문헌 부호(§3)
ANN_AGE_CAP = 120
TP_LOOKBACK_DAYS = 90  # 달력일(정정 ML-1)

# ── 모델(§5) ───────────────────────────────────────────────────────────────────
LGB_VERSION = "4.7.0"
LGB_ROUNDS = 300
LGB_PARAMS = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4,
    "min_data_in_leaf": 500, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 10.0, "deterministic": True, "force_row_wise": True, "num_threads": 1, "verbosity": -1,
}
SEEDS = (1, 2, 3, 4, 5)
RIDGE_ALPHA = 0.1

# ── 판정·널(§7·§8) ─────────────────────────────────────────────────────────────
NW_LAG = 6
T_BAR = 2.0
NULL_Q = 0.95
N2_REPS, N3_REPS, N1_REPS = 1000, 300, 20
IC_DAYS_MIN, U_MEDIAN_MIN = 160, 150  # VOID
IC0, SIGMA_TARGET = 0.03, 0.020  # 2단계(여기서는 기록만)


def sign_table_hash() -> str:
    """EW 부호표·피처 목록의 지문(테스트가 고정한다)."""
    blob = json.dumps({"features": FEATURES, "signs": EW_SIGNS, "s": S_FEATURES}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ── 창 가드(DESIGN §2) ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    """데이터 창. ``mode="eval"`` 만 IC·북·적합도 함수를 부를 수 있다."""

    end: str
    mode: str = "eval"


def assert_window(end, mode: str = "eval") -> Window:
    """평가 모드는 END(2023-12-28) 초과나 2024-01-01 이후를 ``ValueError`` 로 거부한다.

    ``fit_only`` 는 전진 재적합(§11) 전용 — ≤ 2026-09-11 학습 행만 허락하고, 그 창으로 IC·북·적합도 함수를 부르면
    :func:`require_eval` 이 예외를 낸다.
    """
    e = pd.Timestamp(end)
    if mode == "eval":
        if e > pd.Timestamp(END) or e >= pd.Timestamp(SEALED_START):
            raise ValueError(f"평가 창 끝 {e.date()} 이 END {END} 뒤다 — 2024-01 이후는 계산하지 않는다(PREREG §1)")
    elif mode == "fit_only":
        if e > pd.Timestamp(FIT_ONLY_END):
            raise ValueError(f"fit_only 창 끝 {e.date()} 이 {FIT_ONLY_END} 뒤다")
    else:
        raise ValueError(f"모드 {mode!r} 는 eval·fit_only 중 하나")
    return Window(end=str(e.date()), mode=mode)


def require_eval(window: Window, what: str) -> None:
    """IC·북·적합도 계산의 첫 줄. 평가 모드 창이 아니면 거부한다(봉인 구간 지표 계산을 코드로 막는다)."""
    if not isinstance(window, Window) or window.mode != "eval":
        raise PermissionError(f"{what}: 평가 모드 창이 아니다 — fit_only 창으로는 IC·북·적합도를 계산하지 않는다(PREREG §1·§11)")
    assert_window(window.end, "eval")


# ── 공용 수치 ───────────────────────────────────────────────────────────────────


def rank_gauss(v: np.ndarray) -> np.ndarray:
    """Φ⁻¹((순위−0.5)/n), 평균 순위. NaN 은 NaN 으로 남긴다(§3)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    fin = np.isfinite(v)
    n = int(fin.sum())
    if n:
        r = stats.rankdata(v[fin], method="average")
        out[fin] = stats.norm.ppf((r - 0.5) / n)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터(유한만 넘긴다)의 Spearman — 평균 순위의 Pearson."""
    ra = stats.rankdata(a, method="average")
    rb = stats.rankdata(b, method="average")
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def grid_phase(dates: pd.DatetimeIndex) -> int:
    """격자 기준일 = GRID_START 이후 첫 거래일의 열 번호."""
    return int(dates.searchsorted(pd.Timestamp(GRID_START)))


def ic_grid(dates: pd.DatetimeIndex, end: str = END) -> list[int]:
    """IC 격자(5거래일 간격, 기준일부터) 중 t+21 ≤ END 인 것."""
    t0 = grid_phase(dates)
    last = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    return [t for t in range(t0, len(dates), IC_STEP) if t + LABEL_LAST <= last]


def phase_rows(dates: pd.DatetimeIndex, lo: str = LOAD_START, end: str = END) -> list[int]:
    """학습 행 날짜 — IC 격자와 같은 위상(기준일 ± 5k), lo 이후, t+21 ≤ END."""
    t0 = grid_phase(dates)
    first = int(dates.searchsorted(pd.Timestamp(lo)))
    last = int(dates.searchsorted(pd.Timestamp(end), side="right")) - 1
    start = t0 - ((t0 - first) // IC_STEP) * IC_STEP
    return [t for t in range(start, len(dates), IC_STEP) if t >= first and t + LABEL_LAST <= last]


def oos_days(dates: pd.DatetimeIndex, grid: list[int], year: int | None = None) -> list[int]:
    """OOS IC 격자일(2020~23, 또는 한 해)."""
    ys = TEST_YEARS if year is None else (year,)
    return [t for t in grid if dates[t].year in ys]
