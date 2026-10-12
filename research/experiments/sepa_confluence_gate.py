#!/usr/bin/env python
"""SEPA 합류(스윙 3호) — 미너비니 뼈대 × 실적 다리 × 수급·섹터·뉴스 확신 층, 사전등록 러너.

사전등록: ``research/logs/sepa_confluence/VERDICT.md`` (2026-10-09, 결과 보기 전 커밋 0a0f402).
설계서: ``docs/superpowers/specs/2026-10-09-sepa-confluence.md``.

배터리는 재발명하지 않는다 — ``prop_gate`` 가 비용 스윕·폴드·R 분포·취약성·손안댄창·DSR 을, core
``stats.matched_null`` 이 같은 날·같은 층 무작위 대조(주 대조)를 낸다. 민감도 8칸은 전부 ``gate_sim(config=)``
으로 같은 원장(``research/logs/sepa_confluence/TRIALS.jsonl``)에 남고, 옛 미너비니 시행(minervini_prop) 1회도
사전등록대로 같은 원장에 한 줄로 적는다 — DSR 의 N = 1 + 8 + 1.

이 러너는 **리포터**다. 합격 바 숫자를 계산해 ``RUN.md`` 에 적을 뿐 판정하지 않는다(판정은 VERDICT.md 에 사람이).

Run:  uv run --extra pg python research/experiments/sepa_confluence_gate.py
디버그(구간 축소·원장 분리 — 판정 원장을 더럽히지 않는다):
      ... sepa_confluence_gate.py --event-lo 2020-01-01 --event-hi 2020-06-30 --out <scratch>/sepa_debug
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from krx_quant_core.market.session import KST  # noqa: E402
from krx_quant_core.stats.matched_null import (  # noqa: E402
    cluster_bootstrap_ci, matched_alpha, matched_control)
from prop_gate import prop_gate  # noqa: E402
from prop_swing_common import (  # noqa: E402
    gate_sim, load_env_db, render_dist_fragility, render_gate_core, render_gate_report, render_untouched)

from research.signals.sepa import (  # noqa: E402
    PREREG_SKELETON, build_skeleton_events, control_trades, load_panels, simulate_sepa, skeleton_features)
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.earnings_leg import accel_flag, e_flag, earnings_leg_panel, pead_rank  # noqa: E402
from swing_it.features.flow_leg import flow_leg  # noqa: E402
from swing_it.features.korea import (  # noqa: E402
    cost_rt, load_listing_meta, sector_leadership, sector_of_code, sector_returns, universe_mask)
from swing_it.features.technical import average_trade_value, simple_return  # noqa: E402
from swing_it.news.daily_leg import build_daily, n_flag_panel, n_flag_values  # noqa: E402
from swing_it.storage import connect, db_default, read_earnings, read_prices, read_supply_demand  # noqa: E402

OUT_DIR = "research/logs/sepa_confluence"
LABEL = "sepa_confluence"
PANEL_START = "2016-09-01"          # 가격 패널 시작(데이터는 2016-09-09 부터) — 252일 워밍업 포함
EVENT_LO, EVENT_HI = "2017-09-15", "2026-07-01"   # 사전등록 사건 구간(양끝 포함)
FLOW_LO = "2024-02-19"              # F 를 쓰는 사건 하한(그 전 수급은 미래로 골라진 부분집합)
HEDGE_LO = "2024-01-26"             # sector_index(001/101) 시작
NEWS_LO = "2025-09-01"              # 뉴스 기술: 이 날 이후 진입한 S 트레이드
NEWS_FEED_LO, NEWS_FEED_HI = "2025-06-01", "2026-10-10"   # 피드 분모 범위(coverage_news.md 와 같다)
REGIMES = (("2017-09~2021", "2017-09-15", "2022-01-01"), ("2022~2026", "2022-01-01", "2027-01-01"))
N_PER, SEED, N_BOOT = 5, 0, 2000
VOID_MIN_N = 150
ML_FIRST_TEST_YEAR = 2020
ML_LAMBDA = 1.0

# --- 판정 config (1개 — 이게 테스트다). VERDICT.md "판정 config" 1~5 를 그대로 옮긴다 ---
PREREG: dict = {
    **PREREG_SKELETON,                 # rs_min 70 · breadth_min 0.5 · 피벗 50 · 거래량 1.5 · 손절 8% · 60일 · 추격 5% · 상한가 버림
    "event_lo": EVENT_LO,
    "event_hi": EVENT_HI,
    "ma50_exit": True,                 # 종가 < MA50 → 다음날 시가
    "e_mode": "e_flag",                # 실적 다리: earnings_leg.e_flag
    "e_min_yoy": 0.25,
    "e_max_age_days": 140,             # 사전등록 변경(90 → 140): avail_date 고정값의 137일 공백
    "pead_rank_min": None,
    "universe": "korea.universe_mask(raw_close=daily_bars.close) — 보통주·KOSPI/KOSDAQ·ADV20≥200억·원종가≥1000·상장252일",
    "cost": "korea.cost_rt(청산일, 시장)",
    "matched": "core matched_null — 같은 날×ADV20 4분위×직전5일 4분위×시장, 유니버스 풀(자기 종목 제외), "
               "n_per=5, seed=0, 같은 진입·청산(sepa.control_trades), 날짜군집 부트스트랩 2000",
}


def _cfg(**kw) -> dict:
    c = dict(PREREG)
    c.update(kw)
    return c


# 민감도 — 사전등록 8칸 그대로(승자선택 금지, 전부 원장).
SENS_CELLS: dict[str, dict] = {
    "E 끔(뼈대 단독, 새 청산)": _cfg(e_mode="off"),
    "보유 20일 고정(MA50 이탈 끔)": _cfg(max_hold=20, ma50_exit=False),
    "보유 60일 고정(MA50 이탈 끔)": _cfg(max_hold=60, ma50_exit=False),
    "손절 10%": _cfg(stop=0.10),
    "E 문턱 YoY ≥ 0": _cfg(e_min_yoy=0.0),
    "E 대신 PEAD 순위 ≥ 70": _cfg(e_mode="pead_rank", pead_rank_min=0.70),
    "breadth 끔": _cfg(breadth_min=-1.0),      # breadth ∈ [0,1] 이라 −1 은 항상 통과
    "RS ≥ 80": _cfg(rs_min=80.0),
}

# 옛 미너비니 시행 — 사전등록 "원장 N(이 판정의 민감도 칸 + 옛 미너비니 1회)". 원본 원장이 없어 여기 1줄로 남긴다.
OLD_MINERVINI_TRIAL: dict = {
    "trial": "minervini_prop — 옛 미너비니 시행(2026-08 기각, 러너 삭제). 사전등록대로 이 판정의 다중검정 N 에 1회로 넣는다",
    "source": "research/logs/minervini_prop/VERDICT.md",
    "trend_template": "close>MA50>MA150>MA200 ∧ MA200[t]>MA200[t-20]",
    "near_high": 0.90, "above_low": 1.25, "rs_min": 70,
    "breakout": "close > max(high[t-50..t-1])",
    "vol_dryup": "mean vol[t-5..t-1] < 0.70 × mean vol[t-50..t-1]",
    "adv_floor": 20000, "entry": "t+1 open", "dedup_gap": 10,
    "stop": 0.07, "hold": 10, "exit": "time",
}

ML_FEATURES = ("rs", "yoy_clip", "yoy_prev", "accel", "rev_yoy", "netinc_pos", "age_days", "sector_rank60",
               "breadth", "close_hi252", "log_adv20", "r20", "vol_mult", "smart_ratio_20", "foreign_ratio_20")
ML_FLOW_FEATURES = ("smart_ratio_20", "foreign_ratio_20")


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_sepa_gate_helpers.py)
# ---------------------------------------------------------------------------

def day_quartiles(values: pd.DataFrame, mask: pd.DataFrame, *, n_q: int = 4) -> pd.DataFrame:
    """날짜별 ``mask`` 안 종목 사이의 분위(0..n_q−1). 값이 유한한 종목이 ``n_q`` 미만인 날은 NaN.

    ``catalyst_flow.add_strata`` 와 같은 방식(같은 날 순위 ``method='first'`` → 등분) — 다만 풀이 아니라
    **유니버스 단면 전체**에서 경계를 정한다(트레이드와 대조가 같은 자로 잘린다).
    """
    v = values.where(mask.reindex_like(values).fillna(False).astype(bool))
    v = v.where(np.isfinite(v))
    r = v.rank(axis=1, method="first")
    n = v.notna().sum(axis=1).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.floor(n_q * (r.to_numpy() - 1.0) / n[:, None])
    q[n < n_q, :] = np.nan
    return pd.DataFrame(q, index=values.index, columns=values.columns)


def cluster_diff_ci(x1, c1, x2, c2, *, n_boot: int = N_BOOT, seed: int = SEED,
                    ci: float = 0.95) -> tuple[float, float, float]:
    """두 집단 평균 차이(x1 − x2)와 날짜 군집 부트스트랩 CI. 군집은 두 집단의 합집합에서 함께 뽑는다
    (같은 날의 두 집단 관측이 함께 들어오고 함께 빠진다). → ``(점추정, 하단, 상단)``."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    c1, c2 = np.asarray(c1), np.asarray(c2)
    k1, k2 = np.isfinite(x1), np.isfinite(x2)
    x1, c1, x2, c2 = x1[k1], c1[k1], x2[k2], c2[k2]
    if len(x1) == 0 or len(x2) == 0:
        return (np.nan, np.nan, np.nan)
    point = float(x1.mean() - x2.mean())
    uniq = np.unique(np.concatenate([c1, c2]))
    k = len(uniq)
    if k < 2:
        return (point, np.nan, np.nan)
    i1, i2 = np.searchsorted(uniq, c1), np.searchsorted(uniq, c2)
    s1, n1 = np.bincount(i1, weights=x1, minlength=k), np.bincount(i1, minlength=k).astype(float)
    s2, n2 = np.bincount(i2, weights=x2, minlength=k), np.bincount(i2, minlength=k).astype(float)
    draw = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = s1[draw].sum(1) / n1[draw].sum(1) - s2[draw].sum(1) / n2[draw].sum(1)
    a = (1.0 - ci) / 2.0 * 100.0
    return (point, float(np.nanpercentile(d, a)), float(np.nanpercentile(d, 100.0 - a)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))           # 수치 안정(오버플로 없음)


def fit_logistic_l2(X: np.ndarray, y: np.ndarray, *, lam: float = ML_LAMBDA, max_iter: int = 100,
                    tol: float = 1e-10) -> np.ndarray:
    """L2 로지스틱 회귀(뉴턴법). 목적 = Σ 로그손실 + (λ/2)·‖w‖² — 절편은 벌점 없음(sklearn ``C = 1/λ`` 규약).

    반환 ``w`` 는 ``[절편, 계수…]``. ``X`` 는 호출부가 표준화한다.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    reg = np.full(d + 1, float(lam))
    reg[0] = 0.0
    w = np.zeros(d + 1)
    for _ in range(max_iter):
        p = _sigmoid(Xb @ w)
        g = Xb.T @ (p - y) + reg * w
        h = Xb.T @ (Xb * (p * (1.0 - p))[:, None]) + np.diag(reg) + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(h, g)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return _sigmoid(w[0] + X @ w[1:])


def purged_train_mask(exit_day: pd.Series, test_year: int) -> pd.Series:
    """학습 표본 = **청산일 < 시험 연도 1월 1일** 인 사건(purge — 라벨이 시험 기간에 걸친 사건은 뺀다)."""
    return pd.to_datetime(exit_day) < pd.Timestamp(f"{test_year}-01-01")


def ml_design(train: pd.DataFrame, test: pd.DataFrame, features, flow_features) -> tuple[np.ndarray, np.ndarray, list]:
    """학습 통계만으로 결측 대체(학습 중앙값) → 수급 known 지시자 1개 추가 → 표준화(학습 평균·표준편차).

    지시자는 수급 피처에만 붙인다(사전등록). 학습 중앙값이 없는 피처(예: 2024 전 학습창의 수급)는 0 으로 채운다.
    표준편차 0 인 열은 1 로 나눈다(상수 열 — 계수가 학습에서 움직이지 않는다).
    """
    feats = list(features)
    tr = train[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    te = test[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    flow = [f for f in flow_features if f in feats]
    cols = feats.copy()
    if flow:
        tr["flow_known"] = tr[flow].notna().all(axis=1).astype(float)
        te["flow_known"] = te[flow].notna().all(axis=1).astype(float)
        cols.append("flow_known")
    med = tr[feats].median().fillna(0.0)
    tr[feats] = tr[feats].fillna(med)
    te[feats] = te[feats].fillna(med)
    mu = tr[cols].mean()
    sd = tr[cols].std(ddof=0)
    sd = sd.where((sd > 0) & np.isfinite(sd), 1.0)
    return ((tr[cols] - mu) / sd).to_numpy(), ((te[cols] - mu) / sd).to_numpy(), cols


def ml_walk_forward(df: pd.DataFrame, *, features=ML_FEATURES, flow_features=ML_FLOW_FEATURES,
                    first_test_year: int = ML_FIRST_TEST_YEAR, last_test_year: int | None = None,
                    lam: float = ML_LAMBDA, date_col: str = "event_day", exit_col: str = "exit_day",
                    label_col: str = "label") -> pd.DataFrame:
    """연 단위 확장 walk-forward. 시험 연도 y: 학습 = 청산일 < y-01-01 ∧ 라벨 있음, 시험 = 사건일 연도 y ∧ 라벨 있음.

    선택 = 시험 예측확률 ≥ 학습 표본 예측확률 중앙값. 반환: 시험 행 index · ``test_year`` · ``prob`` ·
    ``selected`` · ``n_train`` · ``train_pos_rate``.
    """
    ev_year = pd.to_datetime(df[date_col]).dt.year
    if last_test_year is None:
        last_test_year = int(ev_year.max())
    lab = df[label_col]
    out = []
    for y in range(first_test_year, last_test_year + 1):
        tr_m = purged_train_mask(df[exit_col], y) & lab.notna()
        te_m = (ev_year == y) & lab.notna()
        if tr_m.sum() < 2 or te_m.sum() == 0 or lab[tr_m].nunique() < 2:
            continue
        Xtr, Xte, _ = ml_design(df[tr_m], df[te_m], features, flow_features)
        ytr = lab[tr_m].to_numpy(float)
        w = fit_logistic_l2(Xtr, ytr, lam=lam)
        ptr, pte = predict_logistic(w, Xtr), predict_logistic(w, Xte)
        thr = float(np.median(ptr))
        out.append(pd.DataFrame({"test_year": y, "prob": pte, "selected": pte >= thr,
                                 "n_train": int(tr_m.sum()), "train_pos_rate": float(ytr.mean())},
                                index=df.index[te_m.to_numpy()]))
    if not out:
        return pd.DataFrame(columns=["test_year", "prob", "selected", "n_train", "train_pos_rate"])
    return pd.concat(out)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _key(day: pd.Series, code: pd.Series) -> pd.Index:
    return pd.Index(pd.to_datetime(day).dt.strftime("%Y%m%d") + "|" + code.astype(str))


class Data:
    """패널·피처 한 벌(전부 가격 패널 모양 날짜×종목, ``d`` 행 = ``d`` 종가까지)."""

    def __init__(self, con, t0: float):
        self.t0 = t0
        p = load_panels(con, start=PANEL_START)
        self.p = p
        _log(f"가격 패널 {p.close.shape} {p.days[0].date()}~{p.days[-1].date()}", t0)
        raw = read_prices(con, session="raw", cols=("code", "date", "close"), table="daily_bars")
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw[raw["date"] >= pd.Timestamp(PANEL_START)]
        raw_close = raw.pivot(index="date", columns="code", values="close").astype(float)
        raw_close = raw_close.reindex(index=p.days, columns=p.close.columns)
        del raw
        self.meta = load_listing_meta(con)
        self.uni = universe_mask(p.close, p.trade_value, self.meta, raw_close=raw_close)
        del raw_close
        _log(f"유니버스 종목-일 {int(self.uni.to_numpy().sum()):,}", t0)
        self.feats = skeleton_features(p)
        self.nan_ma = pd.DataFrame(np.nan, index=p.days, columns=p.close.columns)

        vol = p.volume
        close_tr = p.close.where(vol > 0)            # 정지일(거래량 0)은 수익률에서 빼낸다(catalyst_flow 와 같음)
        self.adv20 = average_trade_value(p.trade_value, 20)
        self.ret5 = simple_return(close_tr, 5)
        self.q_adv = day_quartiles(self.adv20, self.uni)
        self.q_ret5 = day_quartiles(self.ret5, self.uni)
        self.r20 = simple_return(p.close, 20)
        self.vol_mult = vol / vol.shift(1).rolling(50, min_periods=50).mean()
        self.close_hi252 = p.close / p.high.rolling(252, min_periods=252).max()

        sec = sector_of_code(self.meta, p.close.columns)
        self.K = sector_leadership(p.close, sec, universe=self.uni)
        sr = sector_returns(p.close, sec, universe=self.uni)
        srank = sr.rank(axis=1, pct=True)            # 클수록 강한 섹터
        sk = srank.reindex(columns=sec.to_numpy())
        sk.columns = p.close.columns
        sk.index = p.days
        self.sector_rank = sk
        _log("섹터 K·순위", t0)

        sd = read_supply_demand(con, session="raw", cols=("code", "date", "foreign_", "institution", "acc_trde_qty"),
                                start=PANEL_START)
        self.flow = flow_leg(sd, index=p.days, columns=p.close.columns, weight=p.trade_value)
        del sd
        _log("수급 F(거래대금 가중)", t0)

        ea = read_earnings(con, all_versions=True, cols=("code", "period", "avail_date", "knowledge_date", "netinc",
                                                       "netinc_prior", "revenue", "revenue_prior", "op_income",
                                                       "op_income_prior"))
        ep = earnings_leg_panel(ea, [d.strftime("%Y-%m-%d") for d in p.days], avail_col="avail_date")  # legacy-asof: 사전등록 재현 경로 — 옛 as-of 그대로
        del ea
        di = p.days.get_indexer(pd.to_datetime(ep["date"]))
        ci = p.close.columns.get_indexer(ep["code"])
        ok = (di >= 0) & (ci >= 0)
        un = self.uni.to_numpy()
        elig = np.zeros(len(ep), dtype=bool)
        elig[ok] = un[di[ok], ci[ok]]
        ep["accel"] = accel_flag(ep)
        ep["pead_rank"] = pead_rank(ep, eligible=pd.Series(elig, index=ep.index))
        self.earn = ep
        _log(f"실적 패널 {len(ep):,}행", t0)

    def pick(self, panel: pd.DataFrame, di: np.ndarray, ci: np.ndarray) -> np.ndarray:
        if isinstance(panel.dtypes.iloc[0], pd.BooleanDtype) or str(panel.dtypes.iloc[0]) == "boolean":
            arr = panel.astype("float64").to_numpy()
        else:
            arr = panel.to_numpy(dtype=float)
        return arr[di, ci]


def attach_features(D: Data, ev: pd.DataFrame) -> pd.DataFrame:
    """사건 표(``day_idx``·``col_idx`` 는 가격 패널 위치)에 E·K·F·층·정합 층·ML 피처를 사건일 기준으로 붙인다."""
    ev = ev.copy()
    di, ci = ev["day_idx"].to_numpy(), ev["col_idx"].to_numpy()
    for name, panel in (("adv20", D.adv20), ("ret5", D.ret5), ("q_adv", D.q_adv), ("q_ret5", D.q_ret5),
                        ("r20", D.r20), ("vol_mult", D.vol_mult), ("close_hi252", D.close_hi252),
                        ("K", D.K), ("sector_rank60", D.sector_rank),
                        ("F", D.flow["F"]), ("smart_ratio_20", D.flow["smart_ratio_20"]),
                        ("foreign_ratio_20", D.flow["foreign_ratio_20"])):
        ev[name] = D.pick(panel, di, ci)
    ev["log_adv20"] = np.log(ev["adv20"].where(ev["adv20"] > 0))
    ev["_date"] = ev["event_day"].dt.strftime("%Y-%m-%d")
    e = D.earn[["code", "date", "period", "yoy", "yoy_prev", "rev_yoy", "netinc", "netinc_pos", "age_days",
                "accel", "pead_rank"]].rename(columns={"date": "_date"})
    ev = ev.merge(e, on=["code", "_date"], how="left").drop(columns="_date")
    known = ev["yoy"].notna()
    ev["e_known"] = known
    base = e_flag(ev.fillna({"netinc_pos": False}), min_yoy=PREREG["e_min_yoy"],
                  max_age_days=PREREG["e_max_age_days"])
    ev["E"] = np.where(known, base.astype(float), np.nan)
    ev["accel"] = np.where(known, ev["accel"].astype(float), np.nan)
    ev["netinc_pos"] = np.where(known, ev["netinc_pos"].astype(float), np.nan)
    ev["yoy_clip"] = ev["yoy"].clip(-5.0, 5.0)
    return ev


def e_pass(ev: pd.DataFrame, cfg: dict) -> pd.Series:
    """판정·민감도 config 의 실적 문. E 가 모름(실적 행 없음)이면 통과하지 않는다."""
    mode = cfg["e_mode"]
    if mode == "off":
        return pd.Series(True, index=ev.index)
    if mode == "pead_rank":
        return (ev["pead_rank"] >= cfg["pead_rank_min"]).fillna(False)
    f = e_flag(ev.fillna({"netinc_pos": 0.0}).assign(netinc_pos=lambda x: x["netinc_pos"].astype(bool)),
               min_yoy=cfg["e_min_yoy"], max_age_days=cfg["e_max_age_days"])
    return (f & ev["e_known"]).astype(bool)


def skeleton_keys(D: Data, cfg: dict) -> pd.Index:
    """뼈대 S(``sepa.build_skeleton_events``) 사건 키 — 사건 구간·RS·breadth 는 cfg 그대로."""
    s = build_skeleton_events(D.p, cfg, universe=D.uni, feats=D.feats)
    return _key(s["event_day"], s["code"])


def rules_of(cfg: dict) -> tuple:
    return (float(cfg["stop"]), int(cfg["max_hold"]), bool(cfg["ma50_exit"]))


def simulate(D: Data, ev: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """사건 → 트레이드(``sepa.simulate_sepa``, 거래량 넘김 = 정지일 체결 불가). 비용·순수익 붙임, index = 키."""
    p = D.p
    stop, hold, ma_exit = rules_of(cfg)
    ma = D.feats["ma50"] if ma_exit else D.nan_ma
    t = simulate_sepa(p.open, p.high, p.low, p.close, ma, ev, volume=p.volume, stop=stop, max_hold=hold,
                      chase=cfg["chase"], limit_up_skip=cfg["limit_up_skip"])
    if t.empty:
        return t
    t["event_day"] = pd.to_datetime(t["event_day"])
    t["entry_day"] = pd.to_datetime(t["entry_day"])
    t["exit_day"] = pd.to_datetime(t["exit_day"])
    t["cost"] = _costs(t)
    t["net"] = t["ret"] - t["cost"]
    t.index = _key(t["event_day"], t["code"])
    return t


_COST_CACHE: dict = {}


def _costs(t: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(t))
    for i, (x, m) in enumerate(zip(t["exit_day"], t["market"], strict=True)):
        k = (x, m)
        if k not in _COST_CACHE:
            _COST_CACHE[k] = cost_rt(x, m)
        out[i] = _COST_CACHE[k]
    return out


class Pools:
    """정합 대조 풀 — (체결 규칙, 날짜) 별로 그날 **유니버스 전 종목**을 같은 규칙(``sepa.control_trades``)으로 시뮬."""

    def __init__(self, D: Data):
        self.D = D
        self.cache: dict[tuple, pd.DataFrame] = {}
        self.done: dict[tuple, set] = {}

    def get(self, rules: tuple, days) -> pd.DataFrame:
        D, p = self.D, self.D.p
        days = pd.DatetimeIndex(sorted(set(pd.to_datetime(days))))
        have = self.done.setdefault(rules, set())
        new = [d for d in days if d not in have]
        if new:
            di_days = p.days.get_indexer(pd.DatetimeIndex(new))
            un = D.uni.to_numpy()[di_days]
            r, c = np.nonzero(un)
            di = di_days[r]
            pairs = pd.DataFrame({"code": p.close.columns[c], "event_day": p.days[di],
                                  "pivot": D.feats["pivot"].to_numpy()[di, c],
                                  "q_adv": D.q_adv.to_numpy()[di, c], "q_ret5": D.q_ret5.to_numpy()[di, c]})
            stop, hold, ma_exit = rules
            ma = D.feats["ma50"] if ma_exit else D.nan_ma
            ct = control_trades(p.open, p.high, p.low, p.close, ma, pairs, volume=p.volume, market=p.market,
                                stop=stop, max_hold=hold, chase=PREREG["chase"],
                                limit_up_skip=PREREG["limit_up_skip"])
            ct = ct[["code", "event_day", "ret", "market", "q_adv", "q_ret5", "exit_reason"]].copy()
            ct["event_day"] = pd.to_datetime(ct["event_day"])
            ct = ct.rename(columns={"event_day": "date"})
            prev = self.cache.get(rules)
            self.cache[rules] = ct if prev is None else pd.concat([prev, ct], ignore_index=True)
            have.update(new)
        pool = self.cache[rules]
        pool = pool[pool["date"].isin(days)].copy()
        pool.index = _key(pool["date"], pool["code"])
        return pool


def matched(pools: Pools, t: pd.DataFrame, cfg: dict) -> dict:
    """정합 대조 alpha + 트레이드별 정합 초과(트레이드 − 자기 대조 평균). 자기 종목은 풀에서 뺀다(같은 키)."""
    pool = pools.get(rules_of(cfg), t["event_day"].unique())
    tt = t.assign(date=t["event_day"])
    strata = ["date", "q_adv", "q_ret5", "market"]
    ctl = matched_control(tt, pool, strata=strata, n_per=N_PER, seed=SEED, exclude_self=True)
    ma = matched_alpha(tt, pool, ctl, value="ret", cluster="date", n_boot=N_BOOT, seed=SEED)
    cm = (ctl.assign(cret=pool.loc[ctl["control_idx"].to_numpy(), "ret"].to_numpy())
             .groupby("trade_idx")["cret"].mean())
    excess = t["ret"] - cm.reindex(t.index)
    return {"ma": ma, "ctl": ctl, "pool": pool, "t": tt, "excess": excess, "control_mean": cm.reindex(t.index)}


def sub_alpha(m: dict, mask: pd.Series):
    """정합 alpha 를 트레이드 부분집합에서(대조는 이미 뽑힌 짝 그대로)."""
    idx = m["t"].index[mask.to_numpy()]
    ctl = m["ctl"][m["ctl"]["trade_idx"].isin(idx)]
    if len(ctl) == 0:
        return None
    return matched_alpha(m["t"], m["pool"], ctl, value="ret", cluster="date", n_boot=N_BOOT, seed=SEED)


def index_hedge(con, D: Data, t: pd.DataFrame) -> pd.Series:
    """진입일 ≥ 2024-01-26 트레이드: 같은 보유 구간(진입일 시가 → 청산일 종가) KOSPI 001 / KOSDAQ 101 수익을 뺀 값."""
    idx = pd.read_sql_query("SELECT code, date, open, close FROM sector_index WHERE code IN ('001','101')", con)
    idx["date"] = pd.to_datetime(idx["date"])
    op = idx.pivot(index="date", columns="code", values="open").astype(float)
    cl = idx.pivot(index="date", columns="code", values="close").astype(float)
    out = pd.Series(np.nan, index=t.index)
    for k, r in t[t["entry_day"] >= pd.Timestamp(HEDGE_LO)].iterrows():
        code = "001" if r["market"] == "KOSPI" else "101"
        o = op[code].get(r["entry_day"], np.nan)
        c = cl[code].get(r["exit_day"], np.nan)
        if np.isfinite(o) and np.isfinite(c) and o > 0:
            out[k] = r["ret"] - (c / o - 1.0)
    return out


# ---------------------------------------------------------------------------
# 요약·서식
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def _ci(lo, hi) -> str:
    return f"[{_pct(lo)}, {_pct(hi)}]"


def mean_ci(x, clusters) -> tuple[int, float, float, float]:
    x = np.asarray(x, float)
    c = np.asarray(clusters)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return 0, np.nan, np.nan, np.nan
    lo, hi, _ = cluster_bootstrap_ci(x[ok], c[ok], n_boot=N_BOOT, seed=SEED)
    return int(ok.sum()), float(x[ok].mean()), lo, hi


def _ma_str(ma) -> str:
    if ma is None or not np.isfinite(ma.alpha):
        return "nan"
    return f"{_pct(ma.alpha)} {_ci(ma.ci_low, ma.ci_high)}"


def _drop_top(x: np.ndarray, frac: float = 0.01) -> float:
    s = np.sort(x[np.isfinite(x)])
    k = max(1, int(np.ceil(frac * len(s))))
    return float(s[:-k].mean()) if len(s) > k else np.nan


def _row_stats(t: pd.DataFrame) -> dict:
    net = t["net"].to_numpy(float)
    return {"n": len(t), "mean": float(np.mean(net)) if len(t) else np.nan,
            "median": float(np.median(net)) if len(t) else np.nan,
            "win": float(np.mean(net > 0)) if len(t) else np.nan}


def _group_excess(df: pd.DataFrame, by, col: str = "excess") -> list[str]:
    L = []
    for k, g in df.groupby(by, dropna=False, sort=True):
        n, m, lo, hi = mean_ci(g[col], g["event_day"])
        L.append(f"| {k} | {n} | {_pct(m)} | {_ci(lo, hi)} |")
    return L


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(원장은 --out 으로 분리)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR, help="RUN.md·원장 디렉터리(기본 = 판정 디렉터리)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.event_lo or args.event_hi) and not debug:
        ap.error("구간 축소는 --out 을 판정 디렉터리 밖으로 둘 때만(판정 원장 보호)")
    os.makedirs(out_dir, exist_ok=True)
    led_label, led_dir = os.path.basename(os.path.normpath(out_dir)), os.path.dirname(os.path.normpath(out_dir))

    prereg = PREREG
    cells = SENS_CELLS
    if debug and (args.event_lo or args.event_hi):
        over = {k: v for k, v in (("event_lo", args.event_lo), ("event_hi", args.event_hi)) if v}
        prereg = {**PREREG, **over}
        cells = {k: {**v, **over} for k, v in SENS_CELLS.items()}

    t0 = time.time()
    load_env_db()
    con = connect(db_default())
    D = Data(con, t0)
    pools = Pools(D)
    p = D.p

    # --- 다중검정 원장: 시도 config 를 먼저 전부 적는다(1 판정 + 8 민감도 + 옛 미너비니 1) ---
    record_trial(led_label, OLD_MINERVINI_TRIAL, logs_dir=led_dir,
                 note="옛 미너비니 시행 1회 — 사전등록 '원장 N(민감도 칸 + 옛 미너비니 1회)'")
    for name, cfg in cells.items():
        record_trial(led_label, cfg, logs_dir=led_dir, note=f"민감도: {name}")

    # --- 사건 표: 전 피벗 돌파(사건 구간) + 피처, 뼈대 S 는 cfg 별로 build_skeleton_events 키로 고른다 ---
    master = build_skeleton_events(p, prereg, universe=D.uni, feats=D.feats, keep_all=True)
    if any(c["event_lo"] != prereg["event_lo"] or c["event_hi"] != prereg["event_hi"] for c in cells.values()):
        raise RuntimeError("민감도 칸의 사건 구간이 판정 config 와 다르다 — master 를 다시 만들어야 한다")
    master = attach_features(D, master)
    master.index = _key(master["event_day"], master["code"])
    _log(f"피벗 돌파 사건(구간 내) {len(master):,}", t0)

    def events_for(cfg: dict) -> pd.DataFrame:
        ev = master.loc[master.index.intersection(skeleton_keys(D, cfg))]
        ev = ev[e_pass(ev, cfg).to_numpy()]
        return ev.sort_values(["event_day", "code"]).reset_index(drop=True)

    S_keys = skeleton_keys(D, prereg)
    S_ev = master.loc[master.index.intersection(S_keys)]
    L: list[str] = [f"# 실행 결과 — {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
                    "사전등록 `VERDICT.md`(0a0f402) 의 판정 config·배터리·민감도·기술을 그대로 계산한 숫자다. "
                    "합격 여부는 여기 적지 않는다 — 판정은 VERDICT.md 에 사람이 쓴다.", ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 구간 {prereg['event_lo']}~{prereg['event_hi']}, 원장 `{out_dir}`. 판정에 쓰지 않는다.", ""]

    # --- 판정 config: S ∧ E ---
    ev_p = events_for(prereg)
    tr = simulate(D, ev_p, prereg)
    _, sk = simulate_sepa(p.open, p.high, p.low, p.close, D.feats["ma50"], ev_p, volume=p.volume,
                          stop=prereg["stop"], max_hold=prereg["max_hold"], chase=prereg["chase"],
                          limit_up_skip=prereg["limit_up_skip"], return_skipped=True)
    _log(f"판정 config: S 사건 {len(S_ev):,} · S∧E 사건 {len(ev_p):,} · 트레이드 {len(tr):,}", t0)
    m = matched(pools, tr, prereg)
    tr["control_mean"], tr["excess"] = m["control_mean"], m["excess"]
    ma = m["ma"]
    _log("정합 대조(판정 config)", t0)

    rep = prop_gate(tr["entry_day"].dt.strftime("%Y-%m-%d").to_numpy(), tr["ret"].to_numpy(), prereg["stop"],
                    label=LABEL, log_dir=out_dir, config=prereg)
    hedged = index_hedge(con, D, tr)
    tr["hedged"] = hedged
    tr["hedged_net"] = hedged - tr["cost"]

    cl = tr["event_day"].to_numpy()
    net = tr["net"].to_numpy(float)
    n_tr = len(tr)
    _, net_mean, net_lo, net_hi = mean_ci(net, cl)
    net2 = (tr["ret"] - 2.0 * tr["cost"]).to_numpy(float)
    drop1 = _drop_top(net)
    reg_rows = []
    for nm, lo, hi in REGIMES:
        mk = (tr["event_day"] >= pd.Timestamp(lo)) & (tr["event_day"] < pd.Timestamp(hi))
        reg_rows.append((nm, int(mk.sum()), sub_alpha(m, mk)))
    yr = tr.groupby(tr["event_day"].dt.year)["net"].agg(["mean", "size"])
    yr_pos = int((yr["mean"] > 0).sum())
    fb = rep["folds"]
    dfl = rep["gate_report"].get("deflation") or {}
    n_led = count_trials(led_label, logs_dir=led_dir)

    L += ["## 표본", "",
          f"- 사건 구간 {prereg['event_lo']} ~ {prereg['event_hi']} · 피벗 돌파(구간 내) {len(master):,} · 뼈대 S 사건 {len(S_ev):,} "
          f"· S∧E 사건 {len(ev_p):,} (E 모름 {int(S_ev['E'].isna().sum()):,} · E=0 {int((S_ev['E'] == 0).sum()):,} "
          f"· E=1 {int((S_ev['E'] == 1).sum()):,})",
          f"- **S∧E 트레이드 n = {n_tr}** (진입 {rep['entry_range'][0]} ~ {rep['entry_range'][1]}). "
          f"표본 하한 {VOID_MIN_N}(미만이면 VOID — 사전등록 규칙). 시장별: "
          + ", ".join(f"{k} {v}" for k, v in tr["market"].value_counts().items()),
          "- 버린 사건(사유별): " + (", ".join(f"{k} {v}" for k, v in sk["skip_reason"].value_counts().items())
                                 if len(sk) else "없음"),
          "- 청산 사유: " + ", ".join(f"{k} {v}" for k, v in tr["exit_reason"].value_counts().items())
          + f" · 보유 거래일 중앙값 {tr['hold_days'].median():.0f}",
          ""]

    L += ["## 합격 바 숫자 (사전등록 1~7 — 숫자만, 판정 아님)", "",
          "| 바 | 사전등록 정의 | 값 |", "|---|---|---|",
          f"| (1) | 정합 대조 alpha(비용 전) 날짜군집 95% CI | alpha {_pct(ma.alpha)} CI {_ci(ma.ci_low, ma.ci_high)} "
          f"(트레이드 {_pct(ma.trade_mean)} vs 대조 {_pct(ma.control_mean)}, p(>0) {ma.p_greater:.3f}, "
          f"대조 {ma.n_controls} · 날짜 {ma.n_clusters} · 짝 없음 {ma.n_unmatched}) |",
          f"| (2) | 연도별 비용(cost_rt) 차감 평균, 날짜군집 부트스트랩 CI | {_pct(net_mean)} {_ci(net_lo, net_hi)} |",
          f"| (3) | 비용 2배 차감 평균 | {_pct(float(np.mean(net2)) if n_tr else np.nan)} |",
          f"| (4) | 상위 1% 트레이드 제거 후 비용 차감 평균 | {_pct(drop1)} (제거 {max(1, int(np.ceil(0.01 * n_tr)))}건) |",
          "| (5a) | 레짐별 정합 alpha 점추정 | " + " · ".join(
              f"{nm}: n {n} alpha {_ma_str(a)}" for nm, n, a in reg_rows) + " |",
          f"| (5b) | 사건 연도별 비용 차감 평균 > 0 인 연도 | {yr_pos} / {len(yr)} — "
          + ", ".join(f"{y} {_pct(r['mean'])}·{int(r['size'])}" for y, r in yr.iterrows()) + " |",
          f"| (6) | prop_gate clean-OOS 폴드 양수(46bp) | {fb['clean_oos_positive']}/{fb['clean_oos_valid']} "
          f"(raw {fb['raw_positive']}/{fb['raw_valid']}) |",
          f"| (7) | 원장 N 으로 깎은 Deflated Sharpe 확률 | P(Sharpe>SR0) {dfl.get('prob_deflated_sharpe', np.nan):.1%} "
          f"(N={dfl.get('n_trials')}, 원장 현재 {n_led}; prop_gate OOS = 진입 ≥ 2022, 46bp) |",
          f"| 하한 | S∧E 트레이드 ≥ {VOID_MIN_N} | n = {n_tr} |",
          ""]
    h_ok = tr["hedged"].notna()
    hn, hm, hlo, hhi = mean_ci(tr.loc[h_ok, "hedged_net"], tr.loc[h_ok, "event_day"])
    _, hg, hglo, hghi = mean_ci(tr.loc[h_ok, "hedged"], tr.loc[h_ok, "event_day"])
    L += ["## 판정 config 상세", "",
          f"- 총수익(비용 전) 평균 {_pct(tr['ret'].mean())} · 중앙값 {_pct(tr['ret'].median())} · 승률(순) "
          f"{np.mean(net > 0):.1%} · 순수익 중앙값 {_pct(float(np.median(net)) if n_tr else np.nan)} · 평균 비용 "
          f"{tr['cost'].mean() * 1e4:.1f}bp",
          f"- 정합 초과(트레이드별, 짝 있는 것) 평균 {_pct(tr['excess'].mean())} · 중앙값 {_pct(tr['excess'].median())} "
          f"· > 0 비율 {np.mean(tr['excess'].dropna() > 0):.1%}",
          f"- **지수 헤지(진입 ≥ {HEDGE_LO}, n {hn}):** 비용 전 {_pct(hg)} {_ci(hglo, hghi)} · 비용 차감 {_pct(hm)} {_ci(hlo, hhi)}",
          f"- R(=순수익/0.08) 평균 {np.mean(net) / prereg['stop']:+.3f}",
          ""]

    L += render_gate_core(rep)
    L += render_dist_fragility(rep)
    L += render_untouched(rep)
    L += render_gate_report(rep)       # [6] 다중검정 보정(render_deflation) 포함

    # --- 층 기여(부) ---
    ev_ne = S_ev[(S_ev["E"] == 0).to_numpy()].sort_values(["event_day", "code"]).reset_index(drop=True)
    tr_ne = simulate(D, ev_ne, prereg)
    m_ne = matched(pools, tr_ne, prereg)
    tr_ne["excess"] = m_ne["excess"]
    d_a = cluster_diff_ci(tr["excess"], tr["event_day"], tr_ne["excess"], tr_ne["event_day"])
    _log("층 기여", t0)

    # --- 모든 S 트레이드(E 끔 칸과 같은 것) — 사다리·뉴스·ML 의 표본 ---
    s_cfg = cells["E 끔(뼈대 단독, 새 청산)"]
    ev_s = events_for(s_cfg)
    ts = simulate(D, ev_s, s_cfg)
    m_s = matched(pools, ts, s_cfg)
    ts["control_mean"], ts["excess"] = m_s["control_mean"], m_s["excess"]
    _log(f"S 트레이드 {len(ts):,} + 정합", t0)
    e1, e0 = ts[ts["E"] == 1], ts[ts["E"] == 0]
    d_b = cluster_diff_ci(e1["excess"], e1["event_day"], e0["excess"], e0["event_day"])

    L += ["## 층 기여 — 실적 다리 E 가 뼈대 위에 더하는 것 (부, 정합 초과 = 트레이드 − 자기 대조 평균)", "",
          "| 비교 | S∧E n · 평균 초과 | S∧¬E(E 알려짐) n · 평균 초과 | 차이 [날짜군집 95% CI] |", "|---|---|---|---|",
          f"| (가) 각자 독립 시뮬(S∧E = 판정 트레이드, S∧¬E = E=0 사건만 따로 시뮬) | {int(tr['excess'].notna().sum())} · "
          f"{_pct(tr['excess'].mean())} | {int(tr_ne['excess'].notna().sum())} · {_pct(tr_ne['excess'].mean())} | "
          f"{_pct(d_a[0])} {_ci(d_a[1], d_a[2])} |",
          f"| (나) S 트레이드 표 안에서 E 로 나눔 | {int(e1['excess'].notna().sum())} · {_pct(e1['excess'].mean())} | "
          f"{int(e0['excess'].notna().sum())} · {_pct(e0['excess'].mean())} | {_pct(d_b[0])} {_ci(d_b[1], d_b[2])} |",
          f"\nS 트레이드 중 E 모름 {int(ts['E'].isna().sum())}건은 두 비교 모두에서 빠진다.", ""]

    # --- 민감도 8칸 ---
    L += ["## 민감도 (사전등록 8칸 · 전부 원장 · 승자선택 금지)", "",
          "| 칸 | n | 비용차감 평균 | 중앙값 | 승률 | 정합 alpha [CI] |", "|---|---|---|---|---|---|"]
    st = _row_stats(tr)
    L.append(f"| (판정 config) | {st['n']} | {_pct(st['mean'])} | {_pct(st['median'])} | {st['win']:.1%} | {_ma_str(ma)} |")
    for name, cfg in cells.items():
        if name.startswith("E 끔"):
            t, mm = ts, m_s
        else:
            t = simulate(D, events_for(cfg), cfg)
            if t.empty:
                L.append(f"| {name} | 0 | | | | |")
                continue
            mm = matched(pools, t, cfg)
        gate_sim(t["entry_day"].dt.strftime("%Y-%m-%d").to_numpy(), t["ret"].to_numpy(), cfg["stop"],
                 LABEL + "_sens", config=cfg, log_dir=out_dir)
        st = _row_stats(t)
        L.append(f"| {name} | {st['n']} | {_pct(st['mean'])} | {_pct(st['median'])} | {st['win']:.1%} | "
                 f"{_ma_str(mm['ma'])} |")
        _log(f"민감도 {name}: n {st['n']}", t0)
    L += ["", f"원장 `{led_dir}/{led_label}/TRIALS.jsonl` 시도 config 수 = **{count_trials(led_label, logs_dir=led_dir)}** "
          "(판정 1 + 민감도 8 + 옛 미너비니 1).", ""]

    # --- 기술: 확신 사다리 ---
    lad = ts[ts["E"].notna() & ts["K"].notna()].copy()
    lad["score_EK"] = lad["E"] + lad["K"]
    L += ["## 기술 — 확신 사다리 (판정 아님, S 트레이드 · 평균 정합 초과)", "",
          f"점수 = E + K. E 모름 또는 K NA 인 S 트레이드 {len(ts) - len(lad)}건 제외(전체 {len(ts)}).", "",
          "| E+K | n | 평균 초과 | [CI] |", "|---|---|---|---|"]
    L += _group_excess(lad, "score_EK")
    post = lad[(lad["event_day"] >= pd.Timestamp(FLOW_LO)) & lad["F"].notna()].copy()
    post["score_EKF"] = post["E"] + post["K"] + post["F"]
    L += ["", f"사건 ≥ {FLOW_LO}, F 알려진 행만(n {len(post)}): 점수 = E + K + F", "",
          "| E+K+F | n | 평균 초과 | [CI] |", "|---|---|---|---|"]
    L += _group_excess(post, "score_EKF")
    pre = ts[(ts["event_day"] < pd.Timestamp(FLOW_LO)) & ts["F"].notna()]
    L += ["", f"참고 1칸 — 사건 < {FLOW_LO}, F 알려진 S 트레이드(수급 표본이 미래로 골라짐 — 판정·사다리에 안 씀): "
          + " · ".join(f"F={int(k)}: n {int(g['excess'].notna().sum())} 평균 초과 {_pct(g['excess'].mean())}"
                       for k, g in pre.groupby("F")) + f" (F 모름 {int(((ts['event_day'] < pd.Timestamp(FLOW_LO)) & ts['F'].isna()).sum())})",
          ""]

    # --- 기술: 뉴스 N ---
    nw = ts[ts["entry_day"] >= pd.Timestamp(NEWS_LO)].copy()
    if len(nw):
        lo = dt.datetime.combine(dt.date.fromisoformat(NEWS_FEED_LO), dt.time(0), tzinfo=KST)
        hi = dt.datetime.combine(dt.date.fromisoformat(NEWS_FEED_HI), dt.time(0), tzinfo=KST)
        daily = build_daily(con, lo=lo, hi=hi, mode="close", trading_days=[d.date() for d in p.days])
        npan = n_flag_panel(daily, sorted({d.date() for d in nw["event_day"]}), codes=set(nw["code"]))
        npan["N"] = n_flag_values(npan)
        npan["date"] = pd.to_datetime(npan["date"])
        nw = nw.merge(npan[["code", "date", "N", "news_known", "judges_60", "cov_60"]],
                      left_on=["code", "event_day"], right_on=["code", "date"], how="left").drop(columns="date")
        nw.index = _key(nw["event_day"], nw["code"])
        ts = ts.join(nw[["N", "news_known", "judges_60", "cov_60"]])
        nk = nw[nw["N"].notna()].copy()
        nk["judge"] = nk["judges_60"].replace("", "(창 안 판정 없음)")
        _log("뉴스 N", t0)
        L += ["## 기술 — 뉴스 N (판정 불가 · 숫자만)", "",
              f"진입 ≥ {NEWS_LO} S 트레이드 {len(nw)} · N 알려짐 {len(nk)} (모름 {int(nw['N'].isna().sum())}). "
              "판정자 = 신호일 60일 창 안 성공 판정의 judge.", "",
              "| (판정자, N) | n | 평균 초과 | [CI] |", "|---|---|---|---|"]
        L += _group_excess(nk, ["judge", "N"])
        L += ["", "| N (판정자 무시) | n | 평균 초과 | [CI] |", "|---|---|---|---|"]
        L += _group_excess(nk, "N")
        L.append("")
    else:
        for c in ("N", "news_known", "judges_60", "cov_60"):
            ts[c] = np.nan

    # --- 기술: ML 메타라벨 도전자 ---
    ml = ts.copy()
    pre_flow = ml["event_day"] < pd.Timestamp(FLOW_LO)
    for f in ML_FLOW_FEATURES:
        ml.loc[pre_flow, f] = np.nan               # 2024-02-19 전 수급은 쓰지 않는다(사전등록)
    ml["label"] = np.where(ml["excess"].notna(), (ml["excess"] > 0).astype(float), np.nan)
    wf = ml_walk_forward(ml)
    ts = ts.join(wf[["test_year", "prob", "selected"]].rename(columns={"prob": "ml_prob", "selected": "ml_selected"}))
    _log(f"ML 도전자 시험 행 {len(wf)}", t0)
    L += ["## 기술 — ML 메타라벨 도전자 (판정 아님 · 이겨도 판정 config 를 바꾸지 않는다)", "",
          f"L2 로지스틱(numpy 뉴턴법, Σ로그손실 + λ/2‖w‖², λ={ML_LAMBDA:g}, 절편 무벌점), 피처 {len(ML_FEATURES)}개 + 수급 known 지시자, "
          "학습 중앙값 대체·학습 통계 표준화, 라벨 = 정합 초과 > 0, 연 단위 확장 WF(청산일 < 시험연도 1/1 만 학습), "
          f"선택 = 예측확률 ≥ 학습 표본 예측확률 중앙값. 시험 연도 = 사건일 연도({ML_FIRST_TEST_YEAR}~).", "",
          "| 시험 연도 | 학습 n | 전체 S n · 평균 초과 | ML 선택 n · 평균 초과 | S∧E(판정) n · 평균 초과 |",
          "|---|---|---|---|---|"]
    test_years = sorted(wf["test_year"].unique()) if len(wf) else []
    allS = ml.loc[wf.index] if len(wf) else ml.iloc[:0]
    for y in test_years:
        w_y = wf[wf["test_year"] == y]
        a = ml.loc[w_y.index, "excess"]
        sel = ml.loc[w_y.index[w_y["selected"].to_numpy(bool)], "excess"]
        se = tr.loc[tr["event_day"].dt.year == y, "excess"].dropna()
        L.append(f"| {y} | {int(w_y['n_train'].iloc[0])} | {len(a)} · {_pct(a.mean())} | {len(sel)} · {_pct(sel.mean())} | "
                 f"{len(se)} · {_pct(se.mean())} |")
    if test_years:
        sel_all = ml.loc[wf.index[wf["selected"].to_numpy(bool)]]
        se_all = tr[(tr["event_day"].dt.year >= test_years[0]) & tr["excess"].notna()]
        rows = (("전체 S", allS), ("ML 선택", sel_all), ("S∧E(판정)", se_all))
        L += ["", f"통합(시험 {test_years[0]}~{test_years[-1]}):", "", "| 집합 | n | 평균 초과 | [날짜군집 CI] |", "|---|---|---|---|"]
        for nm, g in rows:
            n, mval, lo2, hi2 = mean_ci(g["excess"], g["event_day"])
            L.append(f"| {nm} | {n} | {_pct(mval)} | {_ci(lo2, hi2)} |")
        dsel = cluster_diff_ci(sel_all["excess"], sel_all["event_day"], allS["excess"], allS["event_day"])
        L.append(f"\nML 선택 − 전체 S: {_pct(dsel[0])} {_ci(dsel[1], dsel[2])}")
    L.append("")

    # --- 산출물 ---
    elapsed = time.time() - t0
    L += ["## 실행 메타", "",
          f"- 런타임 {elapsed / 60:.1f}분 · 가격 패널 {p.days[0].date()}~{p.days[-1].date()} {p.close.shape} · "
          f"유니버스 종목-일 {int(D.uni.to_numpy().sum()):,}",
          f"- 산출물: `{out_dir}/trades.csv`(판정 트레이드), `{out_dir}/s_trades.csv`(S 트레이드 + 피처·정합 초과·층·ML)", ""]
    keep = ["code", "event_day", "entry_day", "exit_day", "market", "entry_px", "exit_px", "pivot", "ret", "cost", "net",
            "exit_reason", "hold_days", "control_mean", "excess", "hedged", "hedged_net", "E", "K", "F", "rs", "breadth",
            "yoy", "yoy_prev", "accel", "rev_yoy", "netinc_pos", "age_days", "pead_rank", "period", "q_adv", "q_ret5",
            "adv20", "ret5"]
    tr[[c for c in keep if c in tr.columns]].to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    ts["in_primary"] = ts.index.isin(tr.index)
    s_keep = ["code", "event_day", "entry_day", "exit_day", "market", "ret", "cost", "net", "exit_reason", "hold_days",
              "control_mean", "excess", "in_primary", "E", "K", "F", "N", "news_known", "judges_60", "cov_60",
              *ML_FEATURES, "yoy", "period", "pead_rank", "q_adv", "q_ret5", "test_year", "ml_prob", "ml_selected"]
    ts[[c for c in dict.fromkeys(s_keep) if c in ts.columns]].to_csv(os.path.join(out_dir, "s_trades.csv"), index=False)
    with open(os.path.join(out_dir, "RUN.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
