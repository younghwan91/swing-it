#!/usr/bin/env python
"""메타 세대 2 — 상한가 다음날: (A) 보유 중 상한가면 다음날 시가 매도(청산 규칙) · (B) 상한가 다음날 시가 매수(진입). **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/META_GEN2.md`` (f8151d6, 결과 보기 전 고정). 이 러너는 숫자를 ``META_GEN2_RESULTS.md`` 에 적고 설계의
바(문턱)와 산술 비교만 한다. 판정은 사람이 한다.

**사건(LU):** 위생 ∧ ADV20 ≥ 50억(H5 — 라이브 연구 유니버스와 같다) 종목-일, 원 종가(``daily_bars`` — ``krx_session`` 보정판) ≥ 직전 원
종가 × (1.30 − 0.007) (:func:`research.signals.limitup_fast.lu_close`). 연속 상한가는 첫날만 사건(:func:`~research.signals.limitup_fast.lu_first`).
잠김 = 그날 고가 = 종가 ∧ 거래량 < 직전 20일 평균의 0.5 (보조 분류).

**(A) 청산 규칙:** P1(라이브 기본 ``StrategyConfig`` 수급 제외판, 현 청산)과 같은 진입·같은 엔진(메타 세대 1 ``Ctx`` — 10슬롯 × 10%, 복리 절대수익,
비용 ``korea.cost_rt`` 청산일, 이어 붙인 OOS 2020-01 ~ 2026-04-01, 폴드 경계 시가평가)에서 청산만 바꾼다 — P1 vs P1+A(전량) vs P1+A½(절반).
A 는 청산 v2 경로를 후처리(:func:`research.signals.limitup_fast.lu_exit` — 다른 청산보다 우선). 널 = GA2 무작위 진입 300(활동 ≥ 100,
메타 세대 1 P1 널과 **같은 뽑기**)에 같은 세 청산을 걸어 (A − P1) 차이 분포. 짝지은 월 블록 부트스트랩(3개월 블록, 2,000벌, 비원형).
보조: 유휴 현금(1 − 전일 총 노출)에 CD91 연평균 근사 금리(:data:`CD91` — 메타 세대 1 리뷰어 값, DB 밖).

**(B) 사건 연구:** LU 첫날 d (2017-09-15 ~ 2026-09-30) → d+1 시가 매수(상한가 시가면 체결 불가로 버림), 보유 h ∈ {1, 3, 5, 10} 거래일 종가,
손절 −10%(갭이면 시가), 하한가로 잠긴 날은 못 판다, 조정 일봉 |수익| > 31% 는 그 종목 그날 수익 0(``meta_fast.clean_jumps`` — 메타 세대 1 P2 와 같다).
정합 대조: 같은 날 × 시장 × 12-1 모멘텀 5분위 × ADV20 2분위 × 당일 수익 5분위(H5 날·시장 안 분위), 풀 = 같은 규칙으로 체결된 H5 종목-일, 자기 제외,
5개 복원 추출 (쌍, 추출) 해시 난수(``ga_fast.draw_idx`` seed 0). 날짜 군집 부트스트랩 2,000벌. 비용 41.5bp(+ 2배).

Run:  NUMBA_NUM_THREADS=8 uv run --with numba --extra pg python research/experiments/meta_gen2_run.py \\
          --cache <pre_v2 npy 디렉터리> --inputs <스크래치 디렉터리(DB 파생 배열 캐시)> --workers 12
      (DB 읽기 전용 — ``KR_QUANT_DB``. 원 종가·거래대금·거래량·상장 메타·09:00 분봉·뉴스 원장.)
"""

from __future__ import annotations

import argparse
import json
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

import ga2_post as gp  # noqa: E402
import ga2_run as g2  # noqa: E402
import ga_run as g1  # noqa: E402
import meta_gen1_run as mg1  # noqa: E402

from research.signals import exits_v2_fast as xv  # noqa: E402
from research.signals import ga_fast as gf  # noqa: E402
from research.signals import limitup_fast as lf  # noqa: E402
from research.signals import meta_fast as mf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
SUB = "meta_gen2"
RESULTS_MD = "META_GEN2_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/META_GEN2.md (f8151d6)"

# --- 설계 수치(META_GEN2.md — 탐색하지 않는다) ---
VARIANTS = ("P1", "P1+A", "P1+A½")
B_LO, B_HI = "2017-09-15", "2026-09-30"
HS = (1, 3, 5, 10)
H_PRIMARY = 5
B_STOP = 0.10
COST_B = 0.00415                  # 41.5bp(레포 기준 왕복 — 2026 세율)
N_PER = 5
CTL_SEED = 0
MOM_Q, RET_Q = 5, 5
LOCK_VOL = 0.5
HALF_SPLIT = "2022-01-01"        # 전반 2017-09 ~ 2021 · 후반 2022 ~ 2026
N_NULL = mg1.N_NULL
ACTIVE_MIN, MAX_DRAWS = gp.ACTIVE_MIN, gp.MAX_DRAWS
N_BOOT = 2000
BLOCK_M = 3
SEED = 20261012
MDD_FLOOR = 0.01
BONF = 4
NEWS_LO = "2025-09-01"           # 뉴스 판정 구간 시작(daily_leg 커버리지 표의 패널 범위)
CAPITALS = (1e8, 2e8, 3e8)
MINUTE_SINCE = "2025-08-01"
MARKET_CODE = {"KOSPI": 0, "KOSDAQ": 1}
#: CD91 연평균 근사(%/년 → 비율). **DB 밖 — 메타 세대 1 리뷰어(REVIEW_META_GEN1.md F3, 스크래치 review_meta1/indiv.py)가 쓴 기억 값.**
CD91 = {2020: 0.0092, 2021: 0.0085, 2022: 0.0258, 2023: 0.0372, 2024: 0.0348, 2025: 0.0275, 2026: 0.0260}
INPUT_KEYS = ("raw_close", "luc", "luf", "h5", "q_mom", "adv2", "q_ret", "vol", "vavg20", "close_db")


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


_pct, _f = g1._pct, g1._f


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_meta_gen2_helpers.py)
# ---------------------------------------------------------------------------

def cash_daily(dates: pd.DatetimeIndex, table: "dict | None" = None) -> np.ndarray:
    """날별 현금 금리 = 그 해 CD91 근사 / 252. 표에 없는 해는 가장 가까운 해 값."""
    table = CD91 if table is None else table
    ys = np.array(sorted(table))
    y = np.asarray(dates.year)
    k = ys[np.clip(np.searchsorted(ys, y), 0, len(ys) - 1)]
    k = np.where(np.isin(y, ys), y, k)
    return np.array([table[int(v)] for v in k]) / 252.0


def with_cash(raw: np.ndarray, gross_prev: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """유휴 현금 금리를 더한 날별 수익 = raw + max(0, 1 − 전일 총 노출) × 날별 금리. 노출 NaN 은 0(전부 현금)."""
    g = np.nan_to_num(np.asarray(gross_prev, float), nan=0.0)
    return np.asarray(raw, float) + np.clip(1.0 - g, 0.0, None) * np.asarray(rate, float)


def month_ids(dates: pd.DatetimeIndex) -> np.ndarray:
    """날짜 → 첫 달 0 부터 이어지는 달 번호."""
    per = dates.to_period("M")
    return (per.year * 12 + per.month - (per[0].year * 12 + per[0].month)).to_numpy(np.int64)


def block_index(M: int, block: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """비원형 이동 블록 부트스트랩 번호(n × M) — ``meta_fast.calmar_boot_lower`` 와 같은 모양."""
    nb = -(-M // block)
    starts = rng.integers(0, M - block + 1, size=(n, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :M]


def paired_month_boot(ra: np.ndarray, rb: np.ndarray, mon: np.ndarray, tm_a: np.ndarray, net_a: np.ndarray,
                      tm_b: np.ndarray, net_b: np.ndarray, *, block: int = BLOCK_M, n: int = N_BOOT, seed: int = SEED) -> dict:
    """짝지은 월 블록 부트스트랩 — 같은 달 번호로 두 절차를 함께 재표본 → (a − b) 의 CAGR·Calmar·트레이드당 순수익 차이 분포.

    ``ra``·``rb`` = 같은 날의 날별 수익, ``mon`` = 날별 달 번호(0..M−1). 트레이드는 진입 달 번호 ``tm_*`` 와 순수익 ``net_*``.
    CAGR = 월 경로 복리 연환산(12/M), Calmar = CAGR / max(|월 경로 MDD|, 1%). 트레이드당 = 뽑힌 달들의 Σ순수익 / Σ건수.
    반환: 지표마다 ``(lo95, lo90, med, hi95, P(>0))``.
    """
    ma = mg1.monthly_compound(np.asarray(ra, float), mon)
    mb = mg1.monthly_compound(np.asarray(rb, float), mon)
    M = len(ma)
    rng = np.random.default_rng(seed)
    idx = block_index(M, block, n, rng)
    pa, pb = ma[idx], mb[idx]
    cg = np.prod(1 + pa, axis=1) ** (12.0 / M) - np.prod(1 + pb, axis=1) ** (12.0 / M)
    cl = mf.calmar_paths(pa, 12, mdd_floor=MDD_FLOOR) - mf.calmar_paths(pb, 12, mdd_floor=MDD_FLOOR)
    sa = np.bincount(tm_a, weights=net_a, minlength=M)[:M]
    ca = np.bincount(tm_a, minlength=M)[:M].astype(float)
    sb = np.bincount(tm_b, weights=net_b, minlength=M)[:M]
    cb = np.bincount(tm_b, minlength=M)[:M].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tn = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)

    def q(d):
        d = d[np.isfinite(d)]
        if not len(d):
            return (np.nan,) * 5
        return (float(np.percentile(d, 2.5)), float(np.percentile(d, 5)), float(np.median(d)), float(np.percentile(d, 97.5)),
                float((d > 0).mean()))
    return {"cagr": q(cg), "calmar": q(cl), "net": q(tn), "months": M}


def a_bar(dlt: dict) -> dict:
    """(A) 바 산술: (1) 순수익/트레이드·CAGR·Calmar 차이 > 0, (2) 짝지은 월 블록 부트스트랩 95% CI 하단 > 0 (셋 다),
    (3) 2025 제외로도 차이 > 0 (셋 다). ``dlt`` = {지표: {"pt", "lo", "ex25"}}. 널 p95 비교는 바 밖(따로 적는다)."""
    keys = ("net", "cagr", "calmar")

    def pos(v):
        return v is not None and np.isfinite(v) and v > 0
    c1 = all(pos(dlt[k]["pt"]) for k in keys)
    c2 = all(pos(dlt[k]["lo"]) for k in keys)
    c3 = all(pos(dlt[k]["ex25"]) for k in keys)
    return {"c1": c1, "c2": c2, "c3": c3, "all": c1 and c2 and c3}


def b_bar(s: dict) -> dict:
    """(B) 바 산술(h = 5): (1) 정합 초과 날짜 군집 95% CI 하단 > 0, (2) 비용(41.5bp) 차감 평균 > 0, (3) 정합 초과 전반·후반 모두 > 0,
    (4) 정합 초과 2025 제외 > 0, (5) 정합 초과 양수 연도 > 연도 수의 절반."""
    def pos(v):
        return v is not None and np.isfinite(v) and v > 0
    b1 = pos(s.get("exc_lo"))
    b2 = pos(s.get("net1"))
    b3 = pos(s.get("exc_h1")) and pos(s.get("exc_h2"))
    b4 = pos(s.get("exc_ex25"))
    ny, npos = s.get("n_years", 0), s.get("years_pos", 0)
    b5 = bool(ny and npos > ny / 2)
    return {"b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "all": b1 and b2 and b3 and b4 and b5}


def stratum_b(d: np.ndarray, mk: np.ndarray, qm: np.ndarray, adv2: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """B 정합 층 키(날 × 시장 × 모멘텀 5분위 × ADV 2분위 × 당일 수익 5분위) → 정수. 성분 하나라도 없으면 −1."""
    mk = np.asarray(mk, float)
    parts = [np.asarray(x, float) for x in (qm, adv2, qr)]
    ok = (mk >= 0) & np.isfinite(mk) & np.all([np.isfinite(x) for x in parts], axis=0)
    a, b, c = (np.where(ok, x, 0).astype(np.int64) for x in parts)
    m = np.where(ok, mk, 0).astype(np.int64)
    sid = (((np.asarray(d, np.int64) * 2 + m) * MOM_Q + a) * 2 + b) * RET_Q + c
    return np.where(ok, sid, -1)


def window_counts(daily: pd.DataFrame, keys: pd.DataFrame, window_days: int = 60) -> pd.DataFrame:
    """일별 뉴스 건수(``daily_leg.daily_counts`` 모양) → (종목, 날) 마다 매핑 신호일 s ∈ (d − window, d] 의 n_neg·n_articles·n_feed 합.

    ``keys`` = code·date 열. 반환은 ``keys`` 순서 그대로 + ``n_neg_60``·``n_articles_60``·``n_feed_60``.
    """
    out = np.zeros((len(keys), 3), np.int64)
    cols = ("n_neg", "n_articles", "n_feed")
    by = {c: g.sort_values("date") for c, g in daily.groupby("code", sort=False)} if len(daily) else {}
    kd = pd.to_datetime(keys["date"]).to_numpy("datetime64[D]")
    for code, pos in keys.groupby("code", sort=False).indices.items():
        g = by.get(code)
        if g is None:
            continue
        s64 = pd.to_datetime(g["date"]).to_numpy("datetime64[D]")
        d64 = kd[pos]
        hi = np.searchsorted(s64, d64, side="right")
        lo = np.searchsorted(s64, d64 - np.timedelta64(window_days, "D"), side="right")
        for j, col in enumerate(cols):
            cs = np.concatenate([[0], np.cumsum(g[col].to_numpy(np.int64))])
            out[pos, j] = cs[hi] - cs[lo]
    res = keys.copy()
    for j, col in enumerate(cols):
        res[f"{col}_60"] = out[:, j]
    return res


def neg_flag(n_neg: np.ndarray, n_art: np.ndarray, n_feed: np.ndarray, min_cov: float = 0.5) -> np.ndarray:
    """뉴스 악재 플래그(float): 창에 악재 판정 ≥ 1 → 1, 아니고 아는 경우(피드 0건 또는 판정률 ≥ ``min_cov``) → 0, 모르면 NaN
    (``daily_leg.n_flag_values`` 와 같은 NA ≠ 0 원칙)."""
    n_neg, n_art, n_feed = (np.asarray(x, float) for x in (n_neg, n_art, n_feed))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(n_feed > 0, n_art / np.where(n_feed > 0, n_feed, 1.0), np.nan)
    known = (n_feed == 0) | (cov >= min_cov)
    return np.where(n_neg >= 1, 1.0, np.where(known, 0.0, np.nan))


def years_positive(values: np.ndarray, years: np.ndarray) -> "tuple[int, int, dict]":
    """연도별 평균 → (양수 연도 수, 연도 수, {연도: (평균, n)}). NaN 은 뺀다."""
    v = np.asarray(values, float)
    y = np.asarray(years)
    ok = np.isfinite(v)
    out = {}
    for yy in sorted(set(y[ok].tolist())):
        m = ok & (y == yy)
        out[int(yy)] = (float(v[m].mean()), int(m.sum()))
    return sum(1 for a, _ in out.values() if a > 0), len(out), out


# ---------------------------------------------------------------------------
# DB 파생 입력(원 종가·LU·H5·층·거래량) — 한 번 만들어 스크래치에 .npy 로
# ---------------------------------------------------------------------------

def build_inputs(pre, path: str, t0: float) -> dict:
    """DB(읽기 전용) → 사전계산 격자(종목 × 날, ``pre.codes`` × ``pre.days``) 위의 입력 배열. 있으면 mmap 으로 읽는다."""
    mp = os.path.join(path, "meta.json")
    if os.path.exists(mp):
        arr = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r") for k in INPUT_KEYS + ("market",)}
        arr["info"] = json.load(open(mp, encoding="utf-8"))
        return arr
    import convergence_run as cvr  # noqa: PLC0415
    import sepa_confluence_gate as scg  # noqa: PLC0415
    from prop_swing_common import load_env_db  # noqa: PLC0415
    from relax_run import KrxPatch  # noqa: PLC0415
    from sepa_momentum import day_market_median_half, day_market_quantiles, momentum_12_1  # noqa: PLC0415

    import research.signals.sepa as sepa_mod  # noqa: PLC0415
    from swing_it.features.korea import load_listing_meta, universe_mask  # noqa: PLC0415
    from swing_it.features.technical import average_trade_value  # noqa: PLC0415
    from swing_it.storage import connect, db_default  # noqa: PLC0415

    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    load_env_db()
    con = connect(db_default())
    patch = KrxPatch(con)
    with patch.installed():
        p = sepa_mod.load_panels(con, start=scg.PANEL_START)
        raw_close = cvr.raw_close_panel(con, p)
    meta = load_listing_meta(con)
    con.close()
    _log(f"DB 패널 {p.close.shape} {p.days[0].date()} ~ {p.days[-1].date()}", t0)
    h5 = universe_mask(p.close, p.trade_value, meta, raw_close=raw_close, adv_floor=cvr.PEAD_POP_FLOOR)
    mom = momentum_12_1(p.close)
    adv20 = average_trade_value(p.trade_value, 20)
    prev = p.close.shift(1)
    ret1 = p.close / prev.where(prev > 0) - 1.0
    out = {"q_mom": day_market_quantiles(mom, h5, p.market, n_q=MOM_Q), "adv2": day_market_median_half(adv20, h5, p.market),
           "q_ret": day_market_quantiles(ret1, h5, p.market, n_q=RET_Q),
           "vavg20": p.volume.shift(1).rolling(20, min_periods=20).mean(), "vol": p.volume, "raw_close": raw_close,
           "close_db": p.close}
    del mom, adv20, ret1, prev
    os.makedirs(path, exist_ok=True)

    def T(df, dtype=float):
        return np.ascontiguousarray(df.reindex(index=pre.days, columns=pre.codes).to_numpy(dtype=float).T.astype(dtype))
    arr = {}
    for k, v in out.items():
        arr[k] = T(v)
    arr["h5"] = np.ascontiguousarray(h5.reindex(index=pre.days, columns=pre.codes).fillna(False).to_numpy(bool).T)
    TR = np.asarray(pre.TR)
    arr["luc"] = np.ascontiguousarray(lf.lu_close(arr["raw_close"], TR))
    arr["luf"] = np.ascontiguousarray(lf.lu_first(arr["luc"], TR))
    arr["market"] = p.market.reindex(pre.codes).map(MARKET_CODE).fillna(-1).to_numpy(np.int8)
    for k, v in arr.items():
        np.save(os.path.join(path, f"{k}.npy"), v)
    info = {"nxt": patch.stats, "db_days": [str(p.days[0].date()), str(p.days[-1].date())],
            "db_codes": int(p.close.shape[1]), "codes_missing_in_db": int((~pre.codes.isin(p.close.columns)).sum())}
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(info, fh, default=str)
    _log("DB 파생 입력 저장", t0)
    return build_inputs(pre, path, t0)


# ---------------------------------------------------------------------------
# (A) 계산 문맥 — 메타 세대 1 Ctx + A 청산 변형
# ---------------------------------------------------------------------------

class Ctx2(mg1.Ctx):
    """메타 세대 1 문맥 + LU 행렬 + 청산 가족 ``P1``·``P1+A``·``P1+A½`` (``exits`` 덮어쓰기 — ``P1`` 은 메타 세대 1 과 같은 호출)."""

    def __init__(self, cache: str, inputs: str):
        super().__init__(cache)
        self.LUC = np.load(os.path.join(inputs, "luc.npy"), mmap_mode="r")
        self.dates = mg1.oos_dates(self)
        self.mon = month_ids(self.dates)
        self.cash = cash_daily(self.dates)

    def exits(self, fam: str, w_fb: int) -> dict:
        key = (fam, int(w_fb))
        if key in self._ex:
            self._ex.move_to_end(key)
            return self._ex[key]
        pre = self.pre
        if fam == "P1":
            st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = xv.sim_book_v2(
                pre.O, pre.H, pre.L, pre.C, pre.MA50, pre.MA50, pre.TR, pre.last_present, pre.pc, pre.pd, pre.clow,
                mg1.EXIT_P1, pivot=pre.piv[:, int(w_fb)])
            out = {"x": x, "ea": ea, "ef": ef, "er": er, "ff": ff, "fr": fr, "reason": rs,
                   "fired": np.zeros(len(x), np.int8)}
        else:
            b = self.exits("P1", w_fb)
            out = lf.lu_exit(pre.pc, pre.pd, b, pre.O, pre.TR, self.LUC, half=fam == "P1+A½")
        self._ex[key] = out
        while len(self._ex) > 16:
            self._ex.popitem(last=False)
        return out


def wfb_of(spec: dict) -> int:
    return int(spec["ek"][6] if spec["ek"][5] else g1.W50)


def a_stats(ctx: Ctx2, res: dict) -> dict:
    """한 절차 결과 → CAGR·MDD·Calmar(전 OOS·2025 제외·2020~24)·트레이드당 순수익(전·2025 제외)·노출·현금 금리판."""
    dates = ctx.dates
    r = res["raw"][1:]
    s = mg1.window_stats(r, dates)
    op = res["opened"]
    rr = res["rows"]
    net = rr["tot"][op] - rr["cost"][op]
    ent = ctx.pre.days[rr["pd"][op] + 1]
    ex25 = np.asarray(ent.year != 2025)
    g = res["gross"]
    rc = with_cash(r, g[:-1], ctx.cash)
    sc = mg1.window_stats(rc, dates)
    per, p0 = ent.to_period("M"), dates[0].to_period("M")
    tm = np.clip(np.asarray(per.year * 12 + per.month) - (p0.year * 12 + p0.month), 0, ctx.mon[-1])
    s.update({"n_trades": int(len(op)), "trade_net": float(np.mean(net)) if len(op) else np.nan,
              "trade_net_ex25": float(np.mean(net[ex25])) if ex25.any() else np.nan,
              "exposure": float(np.nanmean(g[1:])), "cash_cagr": sc["cagr"], "cash_mdd": sc["mdd"],
              "cash_calmar": sc["calmar"], "cash_cagr_ex25": sc["cagr_ex25"], "cash_calmar_ex25": sc["calmar_ex25"],
              "years": {int(y): mf.equity_stats(r[np.asarray(dates.year == y)])["total"] for y in sorted(set(dates.year))},
              "_r": r, "_rc": rc, "_tm": tm.astype(np.int64), "_net": net})
    return s


NULL_KEYS = ("cagr", "calmar", "mdd", "trade_net", "cagr_ex25", "calmar_ex25", "trade_net_ex25", "n_trades", "exposure")

_C: dict = {}


def worker_init(cache: str, inputs: str, threads: int = 1) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(threads)
    _C["ctx"] = Ctx2(cache, inputs)


def null_task(t: dict) -> dict:
    """GA2 무작위 진입 기준선(메타 세대 1 ``entry_null_task`` 의 P1 과 같은 난수 — [SEED, 7, 묶음, 0]): 폴드마다 진입 유전자 무작위,
    청산 = P1 · P1+A · P1+A½ 셋 다, 이어 붙인 OOS."""
    ctx = _C["ctx"]
    rng = np.random.default_rng([mg1.SEED, 7, t["batch"], 0])
    out = []
    for j in range(t["n"]):
        cfgs = [gp.random_entry(rng, ctx.genes) for _ in ctx.wins]
        row = {"batch": t["batch"], "j": j}
        for v in VARIANTS:
            rows = mg1.cat_rows([ctx.sepa_rows(v, g2.g2_spec(c), te, 1.0) for c, te in zip(cfgs, ctx.tests, strict=True)])
            k = ctx.kernel(rows, ctx.cap10, ctx.t0, ctx.t1, v)
            s = a_stats(ctx, {"raw": k[0], "exc": k[1], "gross": k[2], "opened": k[3], "rows": rows})
            row.update({f"{v}|{key}": s[key] for key in NULL_KEYS})
        out.append(row)
    return {"rows": out}


# ---------------------------------------------------------------------------
# (B) 사건 연구
# ---------------------------------------------------------------------------

def b_world(ctx: Ctx2, inp: dict, t0: float) -> dict:
    """H5 ∧ 거래 가능 종목-일(사건 구간) 전부 → 경로(4 지평)·층·사건(LU 첫날)·대조 번호."""
    pre = ctx.pre
    d_lo, d_hi = pre.day_idx(B_LO), pre.day_idx((pd.Timestamp(B_HI) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    TR = np.asarray(pre.TR)
    m = np.asarray(inp["h5"]) & TR
    m[:, :d_lo] = False
    m[:, d_hi:] = False
    ci, di = np.nonzero(m)
    order = np.lexsort((ci, di))
    ci, di = ci[order].astype(np.int64), di[order].astype(np.int64)
    mk = np.asarray(inp["market"])[ci].astype(float)
    sid = stratum_b(di, mk, np.asarray(inp["q_mom"])[ci, di], np.asarray(inp["adv2"])[ci, di], np.asarray(inp["q_ret"])[ci, di])
    B = ctx.P2B
    fill, ret, xab, why, luo = lf.b_paths(ci, di, B["O"], B["H"], B["L"], B["C"], TR, ctx.LD, np.array(HS), B_STOP)
    ev = np.asarray(inp["luf"])[ci, di]
    members, ms = gf.stratum_index(sid, fill & (sid >= 0), gf.stratum_order(sid))
    strat_ev = np.where(ev, sid, -1).astype(np.int64)
    cidx = gf.draw_idx(strat_ev, members, ms, N_PER, CTL_SEED)
    _log(f"B: H5 종목-일 {len(ci):,} · 체결 {int(fill.sum()):,} · 층 유효 {int((sid >= 0).sum()):,} · LU 첫날 {int(ev.sum()):,}", t0)
    return {"ci": ci, "di": di, "mk": mk, "sid": sid, "fill": fill, "ret": ret, "xab": xab, "why": why, "luo": luo, "ev": ev,
            "cidx": cidx, "d_lo": d_lo, "d_hi": d_hi}


def b_events(ctx: Ctx2, inp: dict, W: dict) -> pd.DataFrame:
    """사건 표: 사건마다 지평별 수익·대조 평균·정합 초과·손절 여부, 잠김·시장·국면 RA·갭."""
    pre = ctx.pre
    e = np.flatnonzero(W["ev"])
    ci, di = W["ci"][e], W["di"][e]
    B = ctx.P2B
    n = pre.n_days
    vol, vavg = np.asarray(inp["vol"]), np.asarray(inp["vavg20"])
    with np.errstate(invalid="ignore"):
        locked = (np.abs(B["H"][ci, di] / B["C"][ci, di] - 1.0) < 1e-9) & (vol[ci, di] < LOCK_VOL * vavg[ci, di])
        gap = np.where(di + 1 < n, B["O"][ci, np.minimum(di + 1, n - 1)] / B["C"][ci, di] - 1.0, np.nan)
        prev = mf.pd_ffill_rows(np.concatenate([np.full((B["C"].shape[0], 1), np.nan), B["C"][:, :-1]], axis=1))
        adj_ratio = B["C"][ci, di] / prev[ci, di]
        vol_ratio = vol[ci, di] / np.where(vavg[ci, di] > 0, vavg[ci, di], np.nan)
    df = pd.DataFrame({"code": np.asarray(pre.codes)[ci], "date": pre.days[di], "d": di, "c": ci,
                       "market": np.where(W["mk"][e] == 0, "KOSPI", np.where(W["mk"][e] == 1, "KOSDAQ", "?")),
                       "sid": W["sid"][e], "fill": W["fill"][e], "lu_open": W["luo"][e], "locked": locked, "gap": gap,
                       "adj_lu": adj_ratio >= lf.LU_RATIO, "vol_ratio": vol_ratio})
    cidx = W["cidx"][e]
    has = cidx[:, 0] >= 0
    for q, h in enumerate(HS):
        r = W["ret"][e, q]
        cr = np.where(cidx >= 0, W["ret"][np.maximum(cidx, 0), q], np.nan)
        ctl = np.where(has, cr.mean(axis=1), np.nan)
        df[f"ret{h}"] = r
        df[f"ctl{h}"] = ctl
        df[f"exc{h}"] = r - ctl
        df[f"stop{h}"] = W["why"][e, q] == lf.B_STOP
    df["matched"] = has
    # 국면 RA(기술): 사건 구간 전체로 z → 평균(메타 세대 1 zscore_mean, 학습창 대신 사건 구간 — 서술용)
    ra = mg1.zscore_mean(ctx.reg["RA"], np.arange(W["d_lo"], W["d_hi"]))
    df["ra"] = ra[di]
    df["year"] = df["date"].dt.year
    return df


def b_stats(df: pd.DataFrame, h: int, *, seed: int = SEED) -> dict:
    """한 지평의 숫자(체결 ∧ 짝 있음 ∧ 완결된 사건). 정합 초과 CI = 날짜 군집 부트스트랩(95%, Bonferroni ×4 = 98.75%)."""
    from krx_quant_core.stats.matched_null import cluster_bootstrap_ci  # noqa: PLC0415

    r, x = df[f"ret{h}"].to_numpy(float), df[f"exc{h}"].to_numpy(float)
    ok = df["fill"].to_numpy(bool) & np.isfinite(r)
    okx = ok & np.isfinite(x)
    out = {"n_ev": int(len(df)), "n_fill": int(ok.sum()), "n_match": int(okx.sum())}
    if not ok.any():
        return out
    out.update({"ret": float(r[ok].mean()), "med": float(np.median(r[ok])), "win": float((r[ok] > 0).mean()),
                "net1": float((r[ok] - COST_B).mean()), "net2": float((r[ok] - 2 * COST_B).mean()),
                "stop": float(df[f"stop{h}"].to_numpy(bool)[ok].mean())})
    if okx.sum() < 2:
        return out
    xv_, cl = x[okx], df["date"].to_numpy()[okx]
    lo, hi, _ = cluster_bootstrap_ci(xv_, cl, n_boot=N_BOOT, seed=seed, ci=0.95)
    blo, bhi, _ = cluster_bootstrap_ci(xv_, cl, n_boot=N_BOOT, seed=seed, ci=1 - 0.05 / BONF)
    dates = pd.DatetimeIndex(df["date"].to_numpy()[okx])
    h1 = np.asarray(dates < pd.Timestamp(HALF_SPLIT))
    ex = np.asarray(dates.year != 2025)
    yp, ny, yd = years_positive(xv_, dates.year)
    ypn, _, ydn = years_positive(r[okx] - COST_B, dates.year)
    out.update({"exc": float(xv_.mean()), "exc_lo": lo, "exc_hi": hi, "exc_blo": blo, "exc_bhi": bhi,
                "ctl": float(df[f"ctl{h}"].to_numpy(float)[okx].mean()), "n_days": int(len(np.unique(cl))),
                "exc_h1": float(xv_[h1].mean()) if h1.any() else np.nan, "n_h1": int(h1.sum()),
                "exc_h2": float(xv_[~h1].mean()) if (~h1).any() else np.nan, "n_h2": int((~h1).sum()),
                "exc_ex25": float(xv_[ex].mean()) if ex.any() else np.nan, "n_ex25": int(ex.sum()),
                "years_pos": yp, "n_years": ny, "years": yd, "years_pos_net": ypn, "years_net": ydn,
                "net1_m": float((r[okx] - COST_B).mean())})
    return out


def b_split(df: pd.DataFrame, mask: np.ndarray, h: int = H_PRIMARY) -> dict:
    """보조 분류 한 칸(기술 — CI 는 같은 날짜 군집 부트스트랩)."""
    return b_stats(df[np.asarray(mask, bool)], h)


# ---------------------------------------------------------------------------
# 뉴스 악재(기술) · 실행 용량
# ---------------------------------------------------------------------------

def news_split(df: pd.DataFrame, days: pd.DatetimeIndex, t0: float) -> "pd.DataFrame | None":
    """사건(2025-09-01 ~) × 60일 창 악재 판정 → 플래그(1 / 0 / 모름 NaN). DB 가 없으면 None."""
    import datetime as dt  # noqa: PLC0415

    from krx_quant_core.market.session import KST  # noqa: PLC0415

    from swing_it.news.daily_leg import build_daily  # noqa: PLC0415
    from swing_it.news.judgments_db import connect_db  # noqa: PLC0415

    sub = df[df["date"] >= pd.Timestamp(NEWS_LO)][["code", "date"]].reset_index()
    if sub.empty:
        return None
    try:
        con = connect_db()
        lo = dt.datetime.combine((pd.Timestamp(NEWS_LO) - pd.Timedelta(days=61)).date(), dt.time(0), tzinfo=KST)
        hi = dt.datetime.combine((sub["date"].max() + pd.Timedelta(days=1)).date(), dt.time(0), tzinfo=KST)
        daily = build_daily(con, lo=lo, hi=hi, mode="close", trading_days=[d.date() for d in days])
        con.close()
    except Exception as e:  # noqa: BLE001 — 기술 분류; DB 가 없으면 적고 넘어간다
        _log(f"뉴스 원장 읽기 실패: {type(e).__name__}", t0)
        return None
    daily = daily.assign(date=pd.to_datetime(daily["date"]))
    w = window_counts(daily, sub[["code", "date"]])
    w["neg"] = neg_flag(w["n_neg_60"], w["n_articles_60"], w["n_feed_60"])
    w.index = sub["index"].to_numpy()
    _log(f"뉴스: 사건 {len(sub)} · 악재 {int((w['neg'] == 1).sum())} · 없음 {int((w['neg'] == 0).sum())} · 모름 "
         f"{int(w['neg'].isna().sum())}", t0)
    return w


def capacity(pre, rows: "dict[str, pd.DataFrame]", t0: float) -> dict:
    """LU 다음날 09:00 분봉 거래대금(2025-08+) 대비 주문액(자본 1·2·3억 × 슬롯 10%) — 표마다 (n, 찾음, 중앙, p90, >10%, >30%)."""
    out = {}
    try:
        from swing_it.storage import connect, db_default  # noqa: PLC0415

        warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
        con = connect(db_default())
        for name, t in rows.items():
            t = t[t["day"] >= pd.Timestamp(MINUTE_SINCE)]
            need = pd.DataFrame({"code": t["code"].to_numpy(), "ts": t["day"] + pd.Timedelta(hours=9)})
            got = []
            for ts, g in need.groupby("ts"):
                q = pd.read_sql("SELECT code, trade_value FROM minute_bars WHERE ts = %(ts)s AND code = ANY(%(codes)s)", con,
                                params={"ts": ts.to_pydatetime(), "codes": list(g["code"])})
                q["ts"] = ts
                got.append(q)
            tv = pd.concat(got) if got else pd.DataFrame(columns=["code", "trade_value", "ts"])
            j = need.merge(tv, on=["code", "ts"], how="left")
            res = {"n": int(len(need)), "found": int(j["trade_value"].notna().sum()),
                   "tv_med": float(j["trade_value"].astype(float).median()) if len(j) else np.nan}
            for K in CAPITALS:
                ratio = (gf.SLOT_W * K / j["trade_value"].astype(float)).to_numpy(float)
                ratio = ratio[np.isfinite(ratio)]
                res[int(K / 1e8)] = ((float(np.median(ratio)), float(np.percentile(ratio, 90)), float((ratio > 0.1).mean()),
                                      float((ratio > 0.3).mean())) if len(ratio) else None)
            out[name] = res
        con.close()
    except Exception as e:  # noqa: BLE001 — 용량 표는 보조
        out["db_error"] = type(e).__name__
        _log(f"용량 표 DB 오류: {type(e).__name__}", t0)
    return out


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True, help="ga_run 사전계산 .npy 디렉터리(읽기 전용 mmap)")
    ap.add_argument("--inputs", required=True, help="DB 파생 입력 캐시 디렉터리(없으면 DB 에서 만든다 — 스크래치)")
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
    pre = g1.Pre.load(args.cache)
    inp = build_inputs(pre, args.inputs, t0)
    ctx = Ctx2(args.cache, args.inputs)
    pre = ctx.pre
    TR = np.asarray(pre.TR)
    cdb = np.asarray(inp["close_db"])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(cdb / np.asarray(pre.C) - 1.0)
    both = np.isfinite(rel)
    par_close = {"cells": int(both.sum()), "max": float(np.nanmax(rel)) if both.any() else np.nan,
                 "n_gt_1e6": int((rel[both] > 1e-6).sum())}
    luc, luf = np.asarray(inp["luc"]), np.asarray(inp["luf"])
    _log(f"문맥: OOS {ctx.dates[0].date()} ~ {ctx.dates[-1].date()} · DB 종가 대 사전계산 종가 최대 상대차 {par_close['max']:.2e} "
         f"(> 1e-6 칸 {par_close['n_gt_1e6']}) · LU 종가 {int(luc.sum()):,} · 첫날 {int(luf.sum()):,}", t0)

    # ======================= (A) =======================
    real, st = {}, {}
    for v in VARIANTS:
        for cm in (1.0, 2.0):
            res = ctx.run_procedure({"name": v, "fam": v, "mode": "base", "score": None}, cm)
            real[(v, cm)] = res
            st[(v, cm)] = a_stats(ctx, res)
        _log(f"(A) {v}: CAGR {_pct(st[(v, 1.0)]['cagr'])} · Calmar {_f(st[(v, 1.0)]['calmar'])} · 트레이드 "
             f"{st[(v, 1.0)]['n_trades']} · 순/트레이드 {_pct(st[(v, 1.0)]['trade_net'])}", t0)
    # 차이 · 짝지은 월 블록 부트스트랩
    dlt, boot, bars = {}, {}, {}
    b0 = st[("P1", 1.0)]
    for v in VARIANTS[1:]:
        a = st[(v, 1.0)]
        bt = paired_month_boot(a["_r"], b0["_r"], ctx.mon, a["_tm"], a["_net"], b0["_tm"], b0["_net"])
        btc = paired_month_boot(a["_rc"], b0["_rc"], ctx.mon, a["_tm"], a["_net"], b0["_tm"], b0["_net"])
        boot[v] = {"raw": bt, "cash": btc}
        dlt[v] = {"net": {"pt": a["trade_net"] - b0["trade_net"], "lo": bt["net"][0], "ex25": a["trade_net_ex25"] - b0["trade_net_ex25"]},
                  "cagr": {"pt": a["cagr"] - b0["cagr"], "lo": bt["cagr"][0], "ex25": a["cagr_ex25"] - b0["cagr_ex25"]},
                  "calmar": {"pt": a["calmar"] - b0["calmar"], "lo": bt["calmar"][0],
                             "ex25": a["calmar_ex25"] - b0["calmar_ex25"]}}
        bars[v] = a_bar(dlt[v])
    # A 발동 서술: P1 이 실제로 연 트레이드에 그 규칙을 걸었다면(같은 쌍 · 같은 절단)
    fire = {}
    spec = ctx.spec
    for v in VARIANTS[1:]:
        r0, r1 = real[("P1", 1.0)]["rows"], real[(v, 1.0)]["rows"]
        op = real[("P1", 1.0)]["opened"]
        fired = ctx.exits(v, wfb_of(spec))["fired"][r0["idx"][op]]
        d_tot = r1["tot"][op] - r0["tot"][op]
        fm = fired == 1
        fire[v] = {"n": int(len(op)), "fired": int(fm.sum()), "same": int((fired == 2).sum()), "overflow": int((fired < 0).sum()),
                   "d_mean_fired": float(d_tot[fm].mean()) if fm.any() else np.nan,
                   "d_pos": float((d_tot[fm] > 0).mean()) if fm.any() else np.nan,
                   "d_sum": float(d_tot.sum()), "hold_cut": float(np.median((r1["x"][op] - r0["x"][op])[fm])) if fm.any() else np.nan}
        ov = int((ctx.exits(v, wfb_of(spec))["fired"] < 0).sum())
        fire[v]["overflow_all"] = ov
    _log(f"(A) 실제 끝 — 발동 {fire['P1+A']['fired']}/{fire['P1+A']['n']} · 바 A {bars['P1+A']} · A½ {bars['P1+A½']}", t0)

    # --- 널(무작위 진입 300) ---
    nrows = []
    mpc = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mpc, initializer=worker_init,
                                initargs=(args.cache, args.inputs, 1)) as ex:
        futs = [ex.submit(null_task, {"batch": j, "n": 25}) for j in range(MAX_DRAWS // 25)]
        for f in futs:
            if f.cancelled():
                continue
            nrows += f.result()["rows"]
            if sum(1 for r in nrows if r["P1|n_trades"] >= ACTIVE_MIN) >= args.nulls:
                for g in futs:
                    g.cancel()
                break
    ndf_all = pd.DataFrame(nrows)
    ndf = ndf_all[ndf_all["P1|n_trades"] >= ACTIVE_MIN].head(args.nulls).reset_index(drop=True)
    for v in VARIANTS[1:]:
        for k in ("cagr", "calmar", "trade_net", "cagr_ex25", "calmar_ex25", "trade_net_ex25"):
            ndf[f"d|{v}|{k}"] = ndf[f"{v}|{k}"] - ndf[f"P1|{k}"]
    _log(f"(A) 널: 뽑기 {len(ndf_all)} · 활동 {len(ndf)}", t0)
    # 메타 세대 1 P1 널과 같은 뽑기인지(같은 난수 → 같은 CAGR)
    gen1_csv = os.path.join(OUT_DIR, "meta_gen1", "null_entry.csv")
    null_parity = None
    if os.path.exists(gen1_csv):
        g1n = pd.read_csv(gen1_csv)
        g1n = g1n[(g1n["fam"] == "P1") & (g1n["opened"] >= ACTIVE_MIN)].head(len(ndf))
        if len(g1n) == len(ndf):
            null_parity = float(np.nanmax(np.abs(g1n["cagr"].to_numpy(float) - ndf["P1|cagr"].to_numpy(float))))
    nullsum = {}
    for v in VARIANTS[1:]:
        nullsum[v] = {}
        for k, real_v in (("trade_net", dlt[v]["net"]["pt"]), ("cagr", dlt[v]["cagr"]["pt"]), ("calmar", dlt[v]["calmar"]["pt"]),
                          ("cagr_ex25", dlt[v]["cagr"]["ex25"])):
            col = ndf[f"d|{v}|{k}"].to_numpy(float)
            fin = col[np.isfinite(col)]
            nullsum[v][k] = {"p50": float(np.median(fin)) if len(fin) else np.nan,
                             "p95": float(np.percentile(fin, 95)) if len(fin) else np.nan,
                             "pos": float((fin > 0).mean()) if len(fin) else np.nan,
                             "pct": mg1.null_pct(real_v, fin), "real": real_v,
                             "above": bool(len(fin) and np.isfinite(real_v) and real_v > np.percentile(fin, 95))}
    t_a = time.time() - t0
    _log(f"(A) 끝 ({t_a / 60:.1f}분)", t0)

    # ======================= (B) =======================
    W = b_world(ctx, inp, t0)
    df = b_events(ctx, inp, W)
    bst = {h: b_stats(df, h) for h in HS}
    bbar = b_bar(bst[H_PRIMARY])
    _log(f"(B) h=5: n {bst[5].get('n_match')} · 초과 {_pct(bst[5].get('exc'))} [{_pct(bst[5].get('exc_lo'))}, "
         f"{_pct(bst[5].get('exc_hi'))}] · 순 {_pct(bst[5].get('net1'))} · 바 {bbar}", t0)
    # 조건부(기술)
    ra = df["ra"].to_numpy(float)
    ra_ok = np.isfinite(ra) & df["fill"].to_numpy(bool)
    ra_med = float(np.median(ra[ra_ok])) if ra_ok.any() else np.nan
    news = news_split(df, pre.days, t0)
    if news is not None:
        df["neg"] = np.nan
        df.loc[news.index, "neg"] = news["neg"].to_numpy()
    else:
        df["neg"] = np.nan
    splits = {"잠김": b_split(df, df["locked"]), "풀림(잠김 아님)": b_split(df, ~df["locked"]),
              "KOSPI": b_split(df, df["market"] == "KOSPI"), "KOSDAQ": b_split(df, df["market"] == "KOSDAQ"),
              "RA 상위 절반": b_split(df, np.isfinite(ra) & (ra >= ra_med)), "RA 하위 절반": b_split(df, np.isfinite(ra) & (ra < ra_med)),
              "뉴스 악재 있음(판정 구간)": b_split(df, df["neg"] == 1.0), "뉴스 악재 없음(판정 구간)": b_split(df, df["neg"] == 0.0),
              "뉴스 모름(판정 구간)": b_split(df, (df["date"] >= pd.Timestamp(NEWS_LO)) & df["neg"].isna())}
    # 갭 서술: 모든 H5 LU 종가(연속 포함, A 의 매도 시점) · 첫날(B) · 상한가 시가 비율
    n = pre.n_days
    ci_all, di_all = np.nonzero(luc & np.asarray(inp["h5"]))
    k_ok = di_all + 1 < n
    ci_all, di_all = ci_all[k_ok], di_all[k_ok]
    trn = TR[ci_all, di_all + 1]
    Bp = ctx.P2B
    with np.errstate(invalid="ignore", divide="ignore"):
        g_all = Bp["O"][ci_all, di_all + 1] / Bp["C"][ci_all, di_all] - 1.0
    inb = (di_all >= W["d_lo"]) & (di_all < W["d_hi"]) & trn
    gaps = {"all_n": int(inb.sum()), "all_mean": float(np.nanmean(g_all[inb])), "all_med": float(np.nanmedian(g_all[inb])),
            "all_luopen": float(np.nanmean(g_all[inb] >= lf.LU_RATIO - 1.0)),
            "first_n": int(np.isfinite(df["gap"]).sum()), "first_mean": float(np.nanmean(df["gap"])),
            "first_med": float(np.nanmedian(df["gap"])), "first_luopen": float(df["lu_open"].mean()),
            "first_fill": float(df["fill"].mean()), "adj_mismatch": int((~df["adj_lu"]).sum())}
    t_b = time.time() - t0 - t_a
    _log(f"(B) 끝 ({t_b / 60:.1f}분)", t0)

    # ======================= 실행 용량 =======================
    cap_rows = {"B 진입(LU 첫날 다음날 시가 매수, 체결분)": pd.DataFrame({"code": df.loc[df["fill"], "code"].to_numpy(),
                                                                "day": pre.days[df.loc[df["fill"], "d"].to_numpy() + 1]}),
                "A 청산(H5 LU 종가 다음날 시가 매도, 연속 포함)": pd.DataFrame({"code": np.asarray(pre.codes)[ci_all[inb]],
                                                                       "day": pre.days[di_all[inb] + 1]})}
    cap = capacity(pre, cap_rows, t0)
    elapsed = time.time() - t0

    # ======================= 산출물 =======================
    srows = []
    for v in VARIANTS:
        for cm in (1.0, 2.0):
            s = st[(v, cm)]
            srows.append({"variant": v, "cost_mult": cm, **{k: s[k] for k in ("cagr", "mdd", "calmar", "cagr_ex25", "calmar_ex25",
                                                                             "cagr_2024", "trade_net", "trade_net_ex25", "n_trades",
                                                                             "exposure", "cash_cagr", "cash_mdd", "cash_calmar")}})
    pd.DataFrame(srows).to_csv(os.path.join(sub, "a_summary.csv"), index=False)
    ndf_all.to_csv(os.path.join(sub, "a_null.csv"), index=False)
    nav = pd.DataFrame(index=ctx.dates)
    for v in VARIANTS:
        nav[f"{v} ×1"] = np.cumprod(1 + st[(v, 1.0)]["_r"])
        nav[f"{v} ×1 + 현금"] = np.cumprod(1 + st[(v, 1.0)]["_rc"])
    nav.to_csv(os.path.join(sub, "a_nav.csv"))
    yrows = [{"variant": v, "year": y, "ret": r} for v in VARIANTS for y, r in st[(v, 1.0)]["years"].items()]
    pd.DataFrame(yrows).to_csv(os.path.join(sub, "a_peryear.csv"), index=False)
    df.drop(columns=["c"]).to_csv(os.path.join(sub, "b_events.csv"), index=False)
    pd.DataFrame([{"h": h, **{k: v for k, v in s.items() if k not in ("years", "years_net")}} for h, s in bst.items()]).to_csv(
        os.path.join(sub, "b_summary.csv"), index=False)
    led = write_ledger_rows(sub)
    L = render(locals())
    with open(os.path.join(args.out, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n=== 메타 세대 2 — {elapsed / 60:.1f}분 · (A) 바 A {bars['P1+A']['all']} · A½ {bars['P1+A½']['all']} · "
          f"(B) 바 {bbar['all']} ===")
    print(f"→ {os.path.join(args.out, RESULTS_MD)}")
    return 0


def write_ledger_rows(sub: str) -> int:
    """원장 ``meta_gen2/TRIALS.jsonl`` — (A) 청산 변형 3 × 비용 2, (B) 지평 4 한 줄씩(결과 숫자는 넣지 않는다)."""
    path = os.path.join(sub, "TRIALS.jsonl")
    seen = g1.read_fingerprints(path)
    rows = [{"config": {"design": DESIGN_DOC, "part": "A", "variant": v, "cost_mult": cm, "entry": "P1(라이브 기본, 수급 제외)",
                        "universe": "H5 ADV20≥5,000", "slots": f"{gf.SLOTS}×{gf.SLOT_W:g}"},
             "note": "메타 세대 2 (A) — 보유 중 상한가 마감이면 다음날 시가 매도"} for v in VARIANTS for cm in (1.0, 2.0)]
    rows += [{"config": {"design": DESIGN_DOC, "part": "B", "h": h, "stop": B_STOP, "cost_rt": COST_B, "events": "LU 첫날 H5",
                         "match": "날×시장×모멘텀5×ADV2×당일수익5, 5개 해시"},
              "note": "메타 세대 2 (B) — 상한가 다음날 시가 매수"} for h in HS]
    return g1.write_ledger(path, rows, seen)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _ck(v) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def _bp(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 1e4:+.0f}bp"


def _ci(lo, hi) -> str:
    return f"[{_pct(lo)}, {_pct(hi)}]"


def render(v: dict) -> "list[str]":
    st, dlt, boot, bars, fire, nullsum = v["st"], v["dlt"], v["boot"], v["bars"], v["fire"], v["nullsum"]
    bst, bbar, splits, gaps, cap = v["bst"], v["bbar"], v["splits"], v["gaps"], v["cap"]
    ndf, args = v["ndf"], v["args"]
    try:
        import numba  # noqa: PLC0415
        nb_ver = numba.__version__
    except ImportError:
        nb_ver = "없음(순수 파이썬)"
    L = [f"# 메타 세대 2 — 상한가 다음날: (A) 청산 규칙 · (B) 진입 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "설계 `META_GEN2.md`(f8151d6, 결과 보기 전 고정)을 계산한 숫자다. ✓/✗ 는 설계 바와의 **산술 비교**이고 판정은 사람이 한다. "
         "(A) 의 OOS(2020-01 ~ 2026-04-01)는 메타 세대 1 과 같은 의사-OOS 다(진입·청산·유니버스가 이 기간을 본 뒤 정해졌다). "
         "단서(+6.0% 갭)는 이 기간 자료에서 나왔다 — (A)·(B) 모두 같은 자료에서 가설을 만들고 재는 것이라 널·정합 대조가 필요조건일 뿐이다.", "",
         f"재현: `NUMBA_NUM_THREADS=8 uv run --with numba --extra pg python research/experiments/meta_gen2_run.py --cache <pre_v2> "
         f"--inputs <스크래치> --workers {args.workers}` · 실행 {v['elapsed'] / 60:.1f}분((A) {v['t_a'] / 60:.1f}분 · (B) "
         f"{v['t_b'] / 60:.1f}분) · numba {nb_ver}. DB 종가 대 사전계산 종가 최대 상대차 {v['par_close']['max']:.1e}"
         f"(> 1e-6 칸 {v['par_close']['n_gt_1e6']}).", ""]

    # ---- 바 요약 ----
    L += ["## 바 요약 (산술)", "",
          "**(A)** 바: 순수익/트레이드 · 포트폴리오 CAGR · Calmar 차이(변형 − P1) **셋 다** > 0 (1) ∧ 짝지은 월 블록 부트스트랩 95% CI 하단 > 0 "
          "(셋 다) (2) ∧ 2025 제외 차이 > 0 (셋 다) (3). 널(무작위 진입 300) p95 비교는 설계 바 문장 밖이라 **따로** 적는다.", "",
          "| 변형 | Δ순/트레이드 · CI 하단 · 2025 제외 | ΔCAGR · CI 하단 · 2025 제외 | ΔCalmar · CI 하단 · 2025 제외 | (1) | (2) | (3) | 바 | "
          "널 p95 위(Δ순 · ΔCAGR · ΔCalmar) |", "|---|---|---|---|---|---|---|---|---|"]
    for k in VARIANTS[1:]:
        d = dlt[k]
        ns = nullsum[k]
        L.append(f"| {k} | {_pct(d['net']['pt'])} · {_pct(d['net']['lo'])} · {_pct(d['net']['ex25'])} | "
                 f"{_pct(d['cagr']['pt'])} · {_pct(d['cagr']['lo'])} · {_pct(d['cagr']['ex25'])} | "
                 f"{_f(d['calmar']['pt'], '{:+.3f}')} · {_f(d['calmar']['lo'], '{:+.3f}')} · {_f(d['calmar']['ex25'], '{:+.3f}')} | "
                 f"{_ck(bars[k]['c1'])} | {_ck(bars[k]['c2'])} | {_ck(bars[k]['c3'])} | **{_ck(bars[k]['all'])}** | "
                 f"{_ck(ns['trade_net']['above'])} · {_ck(ns['cagr']['above'])} · {_ck(ns['calmar']['above'])} |")
    s5 = bst[H_PRIMARY]
    L += ["", "**(B)** 바(h = 5): 정합 초과 날짜 군집 95% CI 하단 > 0 (1) ∧ 비용(41.5bp) 차감 평균 > 0 (2) ∧ 정합 초과 전반(2017-09 ~ 2021)·후반"
          "(2022 ~ 2026) 모두 > 0 (3) ∧ 정합 초과 2025 제외 > 0 (4) ∧ 정합 초과 양수 연도 과반 (5).", "",
          "| 지평 | n | 정합 초과 [95% CI] | 비용 차감 평균 | 전반 · 후반 | 2025 제외 | 양수 연도 | (1) | (2) | (3) | (4) | (5) | 바 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L.append(f"| h = 5 | {s5.get('n_match')} | {_pct(s5.get('exc'))} {_ci(s5.get('exc_lo'), s5.get('exc_hi'))} | {_pct(s5.get('net1'))} | "
             f"{_pct(s5.get('exc_h1'))} · {_pct(s5.get('exc_h2'))} | {_pct(s5.get('exc_ex25'))} | {s5.get('years_pos')}/{s5.get('n_years')} | "
             f"{_ck(bbar['b1'])} | {_ck(bbar['b2'])} | {_ck(bbar['b3'])} | {_ck(bbar['b4'])} | {_ck(bbar['b5'])} | **{_ck(bbar['all'])}** |")
    L += ["", f"**산술 결과:** (A) P1+A 바 {_ck(bars['P1+A']['all'])} · P1+A½ 바 {_ck(bars['P1+A½']['all'])} · (B) h=5 바 {_ck(bbar['all'])}.", ""]

    # ---- (A) ----
    L += ["## (A) 청산 규칙 — P1 vs P1+A vs P1+A½", "",
          "같은 진입(P1 = 라이브 기본 수급 제외판), 메타 세대 1 엔진(10슬롯 × 10%, 복리, 비용 `cost_rt` 청산일, 이어 붙인 OOS, 폴드 경계 시가평가). "
          "A = 보유 중 LU 종가 → 다음 첫 거래 가능일 시가 **전량**(다른 청산보다 우선 — 최대 보유 종가 청산과 겹치면 시가로 미룬다), "
          "A½ = 그 시가에 남은 수량의 절반 + 나머지 기존 규칙(트레이드당 한 번). LU = 원 종가 ≥ 직전 원 종가 × 1.293.", "",
          "| 변형 | 비용 | CAGR | MDD | Calmar | 2025 제외 CAGR · Calmar | 2020~24 CAGR | 트레이드 | 순수익/트레이드 (2025 제외) | 평균 노출 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in VARIANTS:
        for cm in (1.0, 2.0):
            s = st[(k, cm)]
            L.append(f"| {k} | ×{cm:g} | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_pct(s['cagr_ex25'])} · "
                     f"{_f(s['calmar_ex25'])} | {_pct(s['cagr_2024'])} | {s['n_trades']} | {_pct(s['trade_net'])} "
                     f"({_pct(s['trade_net_ex25'])}) | {_pct(s['exposure'])} |")
    L += ["", f"P1 ×1 은 메타 세대 1 P1(CAGR +7.32% · MDD −12.48% · Calmar 0.59)과 같은 계산이어야 한다 — 여기 {_pct(st[('P1', 1.0)]['cagr'])} · "
          f"{_pct(st[('P1', 1.0)]['mdd'])} · {_f(st[('P1', 1.0)]['calmar'])}.", "",
          "**차이(변형 − P1, 비용 ×1) — 짝지은 월 블록 부트스트랩**(같은 달 번호로 두 절차를 함께 재표본, 3개월 블록 비원형, 2,000벌; CAGR·Calmar 는 "
          "월 경로 — 점추정은 날 경로라 중앙값과 다를 수 있다, Calmar MDD 하한 1%; 트레이드당 = 진입 달 기준):", "",
          "| 변형 | 지표 | 점추정 | 95% CI | 90% 하단 | 부트스트랩 중앙 | P(> 0) |", "|---|---|---|---|---|---|---|"]
    for k in VARIANTS[1:]:
        bt = boot[k]["raw"]
        for name, key, pt in (("순수익/트레이드", "net", dlt[k]["net"]["pt"]), ("CAGR", "cagr", dlt[k]["cagr"]["pt"]),
                              ("Calmar", "calmar", dlt[k]["calmar"]["pt"])):
            q = bt[key]
            fmt = (lambda x: _f(x, "{:+.3f}")) if key == "calmar" else _pct
            L.append(f"| {k} | {name} | {fmt(pt)} | [{fmt(q[0])}, {fmt(q[3])}] | {fmt(q[1])} | {fmt(q[2])} | {_f(q[4])} |")
    L += ["", f"**널 — 무작위 진입 {len(ndf)}개(활동 ≥ {ACTIVE_MIN}, GA2 기준선 생성, 메타 세대 1 P1 널과 같은 난수"
          + (f" — P1 CAGR 최대 차 {v['null_parity']:.1e}" if v["null_parity"] is not None else "") + ")에 같은 세 청산을 건 차이 분포:**", "",
          "| 변형 | 지표 | 실제 Δ | 널 Δ 중앙 | 널 Δ p95 | 널 Δ > 0 비율 | 실제의 널 백분위 | 실제 > p95 |", "|---|---|---|---|---|---|---|---|"]
    for k in VARIANTS[1:]:
        for name, key in (("순수익/트레이드", "trade_net"), ("CAGR", "cagr"), ("Calmar", "calmar"), ("CAGR 2025 제외", "cagr_ex25")):
            q = nullsum[k][key]
            fmt = (lambda x: _f(x, "{:+.3f}")) if key == "calmar" else _pct
            L.append(f"| {k} | {name} | {fmt(q['real'])} | {fmt(q['p50'])} | {fmt(q['p95'])} | {_f(q['pos'], '{:.0%}')} | "
                     f"{_f(q['pct'], '{:.0%}')} | {_ck(q['above'])} |")
    L += ["", "**보조 — 유휴 현금 금리(CD91 연평균 근사: 2020 0.92 · 2021 0.85 · 2022 2.58 · 2023 3.72 · 2024 3.48 · 2025 2.75 · 2026 2.60%, "
          "**DB 밖 — 메타 세대 1 리뷰어 값** `REVIEW_META_GEN1.md` F3). 날 t 수익 += max(0, 1 − 전일 총 노출) × 연 금리/252:**", "",
          "| 변형 | CAGR | MDD | Calmar | 2025 제외 CAGR · Calmar | ΔCAGR [95% CI] | ΔCalmar [95% CI] |", "|---|---|---|---|---|---|---|"]
    for k in VARIANTS:
        s = st[(k, 1.0)]
        if k == "P1":
            dd = "— | —"
        else:
            bc = boot[k]["cash"]
            dd = (f"{_pct(s['cash_cagr'] - st[('P1', 1.0)]['cash_cagr'])} [{_pct(bc['cagr'][0])}, {_pct(bc['cagr'][3])}] | "
                  f"{_f(s['cash_calmar'] - st[('P1', 1.0)]['cash_calmar'], '{:+.3f}')} [{_f(bc['calmar'][0], '{:+.3f}')}, "
                  f"{_f(bc['calmar'][3], '{:+.3f}')}]")
        L.append(f"| {k} + 현금 | {_pct(s['cash_cagr'])} | {_pct(s['cash_mdd'])} | {_f(s['cash_calmar'])} | {_pct(s['cash_cagr_ex25'])} · "
                 f"{_f(s['cash_calmar_ex25'])} | {dd} |")
    L += ["", "**A 가 걸린 트레이드(기술):** P1 이 실제로 연 트레이드에 같은 쌍·같은 절단으로 규칙만 바꿨다면 — 포트폴리오에선 슬롯이 일찍 비어 "
          "뒤 진입이 달라지므로 위 표와 같은 숫자가 아니다.", "",
          "| 변형 | P1 트레이드 | 발동(경로 바뀜) | 발동했으나 값 같음 | 발동 트레이드 총수익 차 평균 | 그중 양수 비율 | 청산일 당김 중앙(행) | 매도분 칸 넘침(전 쌍) |",
          "|---|---|---|---|---|---|---|---|"]
    for k in VARIANTS[1:]:
        f = fire[k]
        L.append(f"| {k} | {f['n']} | {f['fired']} | {f['same']} | {_pct(f['d_mean_fired'])} | {_f(f['d_pos'], '{:.0%}')} | "
                 f"{_f(f['hold_cut'], '{:.0f}')} | {f['overflow_all']} |")
    ys = sorted(st[("P1", 1.0)]["years"])
    L += ["", "**연도별 수익(×1):**", "", "| 변형 | " + " | ".join(str(y) for y in ys) + " |", "|---|" + "---|" * len(ys)]
    for k in VARIANTS:
        L.append(f"| {k} | " + " | ".join(_pct(st[(k, 1.0)]["years"][y]) for y in ys) + " |")

    # ---- (B) ----
    L += ["", "## (B) 진입 — 상한가 첫날 다음날 시가 매수", "",
          f"사건 = H5 ∧ LU 첫날, 신호일 {B_LO} ~ {B_HI}. LU 사건 {len(v['df'])}개 중 다음날 시가가 "
          f"상한가(체결 불가로 버림) {_f(gaps['first_luopen'], '{:.1%}')}, 체결 {_f(gaps['first_fill'], '{:.1%}')}. 원 종가로는 LU 지만 조정 종가 "
          f"비율 < 1.293 인 사건 {gaps['adj_mismatch']}개(권리락 등 — 설계대로 원 종가 판정을 따른다).", "",
          "| 지평 h | 사건 · 체결 · 짝 | 평균 수익 | 중앙 | 승률 | 손절 비율 | 비용 차감 ×1 · ×2 | 대조 평균 | 정합 초과 [95% CI] | Bonferroni ×4 (98.75%) CI | "
          "전반 · 후반 | 2025 제외 | 양수 연도(초과 · 순) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in HS:
        s = bst[h]
        L.append(f"| {h}{' (주)' if h == H_PRIMARY else ''} | {s.get('n_ev')} · {s.get('n_fill')} · {s.get('n_match')} | {_pct(s.get('ret'))} | "
                 f"{_pct(s.get('med'))} | {_f(s.get('win'), '{:.0%}')} | {_f(s.get('stop'), '{:.0%}')} | {_pct(s.get('net1'))} · "
                 f"{_pct(s.get('net2'))} | {_pct(s.get('ctl'))} | {_pct(s.get('exc'))} {_ci(s.get('exc_lo'), s.get('exc_hi'))} | "
                 f"{_ci(s.get('exc_blo'), s.get('exc_bhi'))} | {_pct(s.get('exc_h1'))} · {_pct(s.get('exc_h2'))} | {_pct(s.get('exc_ex25'))} | "
                 f"{s.get('years_pos')}/{s.get('n_years')} · {s.get('years_pos_net')}/{s.get('n_years')} |")
    yd, ydn = s5.get("years", {}), s5.get("years_net", {})
    L += ["", "**h = 5 연도별(정합 초과 · 비용 차감 수익 · n):**", "", "| 연도 | " + " | ".join(str(y) for y in yd) + " |",
          "|---|" + "---|" * len(yd), "| 정합 초과 | " + " | ".join(_pct(a) for a, _ in yd.values()) + " |",
          "| 비용 차감 | " + " | ".join(_pct(ydn.get(y, (np.nan, 0))[0]) for y in yd) + " |",
          "| n | " + " | ".join(str(nn) for _, nn in yd.values()) + " |"]
    L += ["", "**보조 분류(기술만, h = 5):** 잠김 = 고가 = 종가 ∧ 거래량 < 직전 20일 평균 × 0.5. RA = 메타 세대 1 RA 4특징 × 2시장을 사건 구간 전체로 "
          "z 평균(서술용 — 학습창 아님), 체결 사건 중앙값으로 가름. 뉴스 악재 = 판정 원장(daily_leg, close 매핑) 60일 창 악재(주제 main/partial ∧ 방향 −1) "
          f"≥ 1, 판정률 < 50% 이고 악재 0 이면 모름 — {NEWS_LO} 이후 사건만.", "",
          "| 칸 | 사건 · 체결 · 짝 | 평균 수익 | 비용 차감 | 정합 초과 [95% CI] | 손절 비율 |", "|---|---|---|---|---|---|"]
    for name, s in splits.items():
        L.append(f"| {name} | {s.get('n_ev')} · {s.get('n_fill')} · {s.get('n_match')} | {_pct(s.get('ret'))} | {_pct(s.get('net1'))} | "
                 f"{_pct(s.get('exc'))} {_ci(s.get('exc_lo'), s.get('exc_hi'))} | {_f(s.get('stop'), '{:.0%}')} |")

    # ---- 개인 트레이더 관점 ----
    L += ["", "## 개인 트레이더 관점 — 갭의 비대칭과 체결 용량 (기술)", "",
          f"- LU 종가 다음날 시가 갭(정리된 조정가, 신호일 {B_LO} ~ {B_HI}, 다음날 거래 가능): H5 LU 종가 전부(연속 포함 — A 가 파는 자리) "
          f"n {gaps['all_n']:,} · 평균 {_pct(gaps['all_mean'])} · 중앙 {_pct(gaps['all_med'])} · 상한가 시가 {_f(gaps['all_luopen'], '{:.1%}')}; "
          f"LU 첫날(B 사건) 평균 {_pct(gaps['first_mean'])} · 중앙 {_pct(gaps['first_med'])}.",
          "- **비대칭:** A 는 그 갭을 **판다**(시가 매도 = 갭 수익 실현, 상한가 시가도 매도는 체결). B 는 갭이 난 **뒤의 시가에 산다** — 갭 자체는 "
          "B 수익에 들어가지 않고, 갭이 가장 큰 상한가 시가 사건은 체결 불가로 빠진다. 그래서 (A) 의 이득과 (B) 의 수익은 같은 갭을 정반대 쪽에서 본다.",
          "", f"**주문액(자본 × 슬롯 10%) / 다음날 09:00 분봉 거래대금** (분봉 {MINUTE_SINCE}+; 중앙 · p90 · > 10% 비율 · > 30% 비율):", "",
          "| 자리 | n · 분봉 찾음 | 09:00 거래대금 중앙 | 1억 | 2억 | 3억 |", "|---|---|---|---|---|---|"]
    if "db_error" in cap:
        L.append(f"| (DB 오류 {cap['db_error']}) | | | | | |")
    for name, c in cap.items():
        if name == "db_error":
            continue

        def cc(x):
            return "—" if x is None else f"{x[0]:.1%} · {x[1]:.1%} · {x[2]:.0%} · {x[3]:.0%}"
        L.append(f"| {name} | {c['n']} · {c['found']} | {_f(c['tv_med'] / 1e8 if np.isfinite(c['tv_med']) else np.nan, '{:.1f}')}억 | "
                 f"{cc(c[1])} | {cc(c[2])} | {cc(c[3])} |")

    # ---- 정의·모호한 점 ----
    L += ["", "## 정의와 모호한 점(결과 전 구현 선택 — 바꾸지 않았다)", "",
          "- **A 의 '첫 거래 가능 시가':** 다음날 시가가 상한가여도 판다(매수 대기가 쌓인 자리라 매도는 체결된다). 정지면 첫 거래 가능일 시가. 팔 날이 "
          "없으면(패널 끝·폐지) 원래 경로.",
          "- **A 우선순위:** LU 종가 다음날 시가는 그날의 돌파 실패·추적·강세 매도 시가 청산과 같은 가격이고 장중 손절·익절보다 먼저다. 최대 보유(120일째) "
          "종가 청산과 같은 날 LU 면 A 는 다음날 시가로 미룬다(우선), A½ 는 절반만 미룬다.",
          "- **A½:** 트레이드당 한 번(첫 LU). 이후 LU 에는 다시 팔지 않는다.",
          "- **(A) 바의 'CI':** 95%(2.5% 하단)로 읽었다(90% 하단도 적었다). 트레이드당 순수익의 부트스트랩은 진입 달 단위. 2025 제외 = 날 경로에서 2025 날을 뺀 CAGR·Calmar, "
          "진입 2025 를 뺀 트레이드 평균.",
          "- **(B) 보유 h:** 진입일을 1일째로 h 번째 거래일 종가(h = 1 = 시가→종가). 손절은 진입일부터, 잠긴 하한가 날은 팔지 못한다(메타 세대 1 정정 M4). "
          "h 번째 행이 패널 밖인 사건은 그 지평에서 뺀다.",
          "- **(B) '비용 차감 평균':** 사건 자체 수익 − 41.5bp 의 평균(레포 관례 — MINERVINI_EXITS 바 2). 전반·후반·2025 제외·연도 과반은 **주 지표(정합 초과)** "
          "로 셌다; 비용 차감 수익의 양수 연도도 옆에 적었다. 비용은 설계의 고정 41.5bp(2026 세율) — 2023 이전 세율은 더 높다(`cost_rt`).",
          "- **(B) 대조 층의 당일 수익:** 조정 종가 대비 전일 조정 종가(정지일 이월 포함). 분위는 그날 H5 시장 안에서. 대조 풀 = H5 ∧ 거래 가능 ∧ 같은 규칙 체결; "
          "다른 LU 종목도 같은 층이면 대조가 될 수 있다(설계는 자기 종목만 제외).",
          "- **LU 판정:** 원 종가(`daily_bars`, NXT 보정) 대 직전 유한 원 종가. 사건은 그날 거래 가능 ∧ H5. 연속 = 직전 거래 가능일도 LU.",
          "- **뉴스 악재 창:** daily_leg 의 60일(역일) 창을 그대로 썼다(설계에 창 길이가 없다).",
          "", f"산출물: `meta_gen2/` — a_summary.csv · a_null.csv · a_nav.csv · a_peryear.csv · b_events.csv · b_summary.csv · TRIALS.jsonl(덧붙인 {v['led']}줄)."]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
