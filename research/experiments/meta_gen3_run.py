#!/usr/bin/env python
"""메타 세대 3 — 분산 PEAD 를 지금의 엄격한 틀로 다시: Q1 PEAD 분산 북 · Q2 PEAD ∧ 추세 템플릿 · Q3 위성 20%. **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/META_GEN3.md`` (7608844, 결과 보기 전 고정). 이 러너는 숫자를 ``META_GEN3_RESULTS.md`` 에 적고 미리 적은
두 층 문턱과 **산술 비교**만 한다. 판정은 사람이 한다.

**절차(전부 H5 = 위생 ∧ ADV20 ≥ 50억, 2017-09 ~ 2026-09, 폴드 적합 없음 — 고정 규칙 한 벌):**

- ``Q1`` = 매월 마지막 거래일 종가(신호일)에 ``earnings_leg.pead_rank`` (GA 커널 EPS·PIT 정의 — :func:`swing_it.live.features.eps_adjusted_earnings`
  → ``earnings_leg_panel`` → ``pead_rank(eligible = 그날 H5)``) 상위 20 → 다음날(= 그달 첫 거래일) 시가 진입, 종목당 그때 자산의 5%. 다음 리밸런스에서
  순위 30 안이면 계속 보유(거래 없음, 재가중 없음), 밖이면 그 다음날 시가 매도. 손절 없음.
- ``Q2`` = 같은 규칙, 순위 목록 = H5 ∧ 추세 템플릿(``minervini.trend_template`` 원전 수치) 통과 종목만(PEAD 순위 순서 그대로).
- ``Q3`` = 자산 20% Q1(유휴 현금 금리 포함) + 80% 재구축 거래대금 가중 지수 대용(ALL), 매월 첫 거래일에 20/80 으로 되돌림.
- 체결(메타 세대 1 P2 제약): 상한가 시가(정리된 조정 시가 / 직전 조정 종가 ∈ [1.293, 1.305]) 진입 불가 — 그 슬롯은 그달 빈다; 하한가로 잠긴 날
  (한 가격 ∧ 종가 ≤ 직전 × 0.707) 매도 불가 → 다음 잠기지 않은 거래일 시가; 조정 일봉 |수익| > 31% 는 그 종목 그날 수익 0
  (``meta_fast.clean_jumps``). 비용 ``korea.cost_rt`` (청산일, 2배 병기). 엔진 = 메타 세대 1 P2 월 경로(``meta_fast.p2_trades`` +
  ``meta_fast.portfolio_meta``, 20슬롯 × 5%, 복리). 유휴 현금(1 − 전일 총 노출)에 CD91 연평균 근사 금리(:data:`CD91_EXT`).

**기준선:** B0 = 지수 대용(재구축 ALL·KOSPI·KOSDAQ) 보유 · 유니버스 월 등가중 · 2024-01-26+ 공식 코스피·코스닥. B1 = "지수 대용 ALL x% + 현금"
(매월 되돌림), x = 그 Q 의 MDD(창마다)에 맞춘 0.01 격자의 최대 비중.

**창:** 전체 2017-09-01 ~ 2026-09-30(표본 내) · 비교 창 2020-01-02 ~ 2026-09-30(메타 세대 1·2 와 같은 비교 창 — **층 판정 산술의 주 창**),
+ 2020~24 · 2025 제외. **널:** 무작위 20종목 300벌(같은 순위 목록을 리밸런스마다 무작위로 섞고 같은 유지 규칙 — Q2 는 추세 템플릿 통과 목록 안에서),
Q3 널 = 20% Q1 널 + 80% 대용.

Run:  NUMBA_NUM_THREADS=8 uv run --with numba --extra pg python research/experiments/meta_gen3_run.py \\
          --inputs <스크래치 디렉터리(DB 파생 배열 캐시)> --factors <review_ga2/factors.pkl> [--cache <pre_v2> (대조만)]
      (DB 읽기 전용 — ``KR_QUANT_DB``. 가격·원 종가·상장 메타·실적·주식수·09:00 분봉.)
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
import ga_run as g1  # noqa: E402
import meta_gen1_run as mg1  # noqa: E402
import meta_gen2_run as mg2  # noqa: E402

from research.signals import ga_fast as gf  # noqa: E402
from research.signals import meta_fast as mf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
SUB = "meta_gen3"
RESULTS_MD = "META_GEN3_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/META_GEN3.md (7608844)"

# --- 설계 수치(META_GEN3.md — 탐색하지 않는다) ---
TOP_N, KEEP_N = 20, 30
SLOT_W = 1.0 / TOP_N
SAT_W = 0.20                      # Q3 위성 비중
NO_STOP = 2.0                     # 손절가 = 진입가 × (1 − 2) < 0 → 손절 없음
N_NULL = 300
N_BOOT, BLOCK_M = 2000, 3
SEED = 20261013
JUMP_LIMIT = 0.31
LU_LO, LU_HI = 1.293, 1.305       # 상한가 시가 = 조정 비율이 이 구간(이음매 날짜의 가짜 상한가 방지 — 상한 둠)
W_FULL = ("2017-09-01", "2026-09-30")
W_CMP = ("2020-01-01", "2026-09-30")
WINDOWS = {"full": W_FULL, "cmp": W_CMP}
WIN_LABEL = {"full": "전체 2017-09 ~ 2026-09", "cmp": "비교 창 2020-01 ~ 2026-09"}
PRIMARY_WIN = "cmp"
P_B1 = 0.8                        # 층 2: P(B1 을 이김) 문턱
ALPHA_T = 2.0                     # 층 1: 팩터 알파 HAC t 문턱
PROCS = ("Q1", "Q2", "Q3")
CAPITALS = (1e8, 2e8, 3e8)
MINUTE_SINCE = "2025-08-01"
MDD_STEP = 0.01
#: CD91 연평균 근사(비율). 2020~2026 = 메타 세대 1 리뷰어 값(``REVIEW_META_GEN1.md`` F3, 스크래치 ``review_meta1/indiv.py`` — 메타 세대 2 ``CD91``
#: 그대로). 2017~2019 = 이 러너가 덧붙인 근사(한국은행 CD 91일물 연평균 기억 값, **DB 밖** — DB 에 금리 테이블이 없다).
CD91_EXT = {2017: 0.0144, 2018: 0.0168, 2019: 0.0169, **mg2.CD91}
#: 코스피 연말 종가(가격지수) — 메타 세대 1 리뷰어 ``indiv.py`` 값(2019~2023 공개값 기억, DB 밖). **참고 민감도(B1-K)에만** 쓴다.
KOSPI_YE = {2019: 2197.67, 2020: 2873.47, 2021: 2977.65, 2022: 2236.40, 2023: 2655.28}
MARKET_CODE = {"KOSPI": 0, "KOSDAQ": 1}
GRID_KEYS = ("O", "H", "L", "C", "TR", "h5", "tt", "eps", "pead", "adv20")
CODE_KEYS = ("market", "last_present")
DAY_KEYS = ("cost_tab", "proxy")
PROXY_MK = ("KOSPI", "KOSDAQ", "ALL")
BIG = 1 << 30


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


_pct, _f = g1._pct, g1._f


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_meta_gen3_helpers.py)
# ---------------------------------------------------------------------------

def ranked_codes(score: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """종목 번호 목록 — ``ok`` ∧ 점수 유한, 점수 내림차순(동률이면 종목 번호 오름차순)."""
    idx = np.flatnonzero(np.asarray(ok, bool) & np.isfinite(score))
    return idx[np.lexsort((idx, -score[idx]))].astype(np.int64)


def lu_open(o_entry: np.ndarray, c_prev: np.ndarray, lo: float = LU_LO, hi: float = LU_HI) -> np.ndarray:
    """상한가 시가 = 진입 시가 / 직전 종가 ∈ [lo, hi] (조정 가격 비율). 범위 밖(이음매 날짜의 큰 비율)·NaN 은 상한가 아님."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.asarray(o_entry, float) / np.asarray(c_prev, float)
    return np.isfinite(r) & (r >= lo) & (r <= hi)


def mix_monthly(R: np.ndarray, w, mon: np.ndarray) -> np.ndarray:
    """여러 날별 수익 열(T × k)을 비중 ``w`` 로 섞되 **매달 첫날 비중을 되돌리는** 포트폴리오의 날별 수익. NaN 수익은 0."""
    R = np.nan_to_num(np.asarray(R, float).reshape(len(mon), -1))
    w = np.asarray(w, float)
    out = np.empty(len(mon))
    v = w.copy()
    for t in range(len(mon)):
        if t == 0 or mon[t] != mon[t - 1]:
            v = w.copy()
        tot = v.sum()
        out[t] = (v * R[t]).sum() / tot if tot > 0 else 0.0
        v = v * (1.0 + R[t])
    return out


def mdd_match_weight(bench: np.ndarray, cash: np.ndarray, target_mdd: float, mon: np.ndarray, step: float = MDD_STEP) -> float:
    """"지수 x + 현금 (1 − x)" (매달 되돌림)의 MDD 가 ``target_mdd`` 보다 깊지 않은 x 중 최대(격자 ``step``, 0 ~ 1). 1 이어도 더 얕으면 1."""
    best = 0.0
    if not np.isfinite(target_mdd):
        return np.nan
    for x in np.round(np.arange(0.0, 1.0 + 1e-9, step), 10):
        m = mf.equity_stats(mix_monthly(np.column_stack([bench, cash]), [x, 1.0 - x], mon))["mdd"]
        if m >= target_mdd - 1e-12:
            best = float(x)
    return best


def boot_beat(ra: np.ndarray, rb: np.ndarray, mon: np.ndarray, *, block: int = BLOCK_M, n: int = N_BOOT, seed: int = SEED) -> dict:
    """짝지은 월 블록 부트스트랩(같은 달 번호로 둘을 함께 재표본, 비원형 이동 블록 ``block`` 달, ``n`` 벌 — 메타 세대 2 와 같은 모양)
    → P(CAGR a > b)·P(Calmar a > b)(MDD 하한 1%)와 ΔCAGR 5·50·95%."""
    ma = mg1.monthly_compound(np.asarray(ra, float), mon)
    mb = mg1.monthly_compound(np.asarray(rb, float), mon)
    M = len(ma)
    idx = mg2.block_index(M, block, n, np.random.default_rng(seed))
    pa, pb = ma[idx], mb[idx]
    dc = np.prod(1 + pa, axis=1) ** (12.0 / M) - np.prod(1 + pb, axis=1) ** (12.0 / M)
    dl = mf.calmar_paths(pa, 12, mdd_floor=0.01) - mf.calmar_paths(pb, 12, mdd_floor=0.01)
    return {"p_cagr": float((dc > 0).mean()), "p_calmar": float((dl > 0).mean()), "dcagr_lo": float(np.percentile(dc, 5)),
            "dcagr_med": float(np.median(dc)), "dcagr_hi": float(np.percentile(dc, 95)), "months": int(M)}


def tier1(alpha_t: float, cagr: float, null_p95: float) -> dict:
    """층 1: 팩터 알파 HAC t > 2 ∧ CAGR > 무작위 널 CAGR p95."""
    def fin(x):
        return x is not None and np.isfinite(x)
    a = bool(fin(alpha_t) and alpha_t > ALPHA_T)
    b = bool(fin(cagr) and fin(null_p95) and cagr > null_p95)
    return {"t1_alpha": a, "t1_null": b, "t1": a and b}


def tier2(p_b1: float, cagr_2024: float, cash_2024: float, cagr_ex25: float, cash_ex25: float, cagr_x2: float,
          cash_main: float) -> dict:
    """층 2(개인용): (1) P(B1 을 이김) ≥ 0.8, (2) 2020~24 CAGR > 같은 구간 현금 ∧ 2025 제외 CAGR > 같은 구간 현금, (3) 비용 2배 CAGR > 현금."""
    def fin(x):
        return x is not None and np.isfinite(x)
    c1 = bool(fin(p_b1) and p_b1 >= P_B1)
    c2a = bool(fin(cagr_2024) and fin(cash_2024) and cagr_2024 > cash_2024)
    c2b = bool(fin(cagr_ex25) and fin(cash_ex25) and cagr_ex25 > cash_ex25)
    c3 = bool(fin(cagr_x2) and fin(cash_main) and cagr_x2 > cash_main)
    return {"t2_b1": c1, "t2_2024": c2a, "t2_ex25": c2b, "t2_x2": c3, "t2": c1 and c2a and c2b and c3}


def kospi_hybrid(proxy_kospi: np.ndarray, official_kospi: np.ndarray, dates: pd.DatetimeIndex, year_end: dict,
                 since: str = "2024-01-26") -> np.ndarray:
    """메타 세대 1 리뷰어(``review_meta1/indiv.py``) 방식의 코스피 혼성 일수익(참고 — 설계 밖): ``since``+ 공식 지수 일수익(있는 날),
    ``year_end`` 에 전년·그해 연말 값이 다 있는 해는 대용 일수익에 연 상수 로그 드리프트를 더해 실제 연수익에 맞춤, 나머지 날은 대용 그대로."""
    k = np.nan_to_num(np.asarray(proxy_kospi, float)).copy()
    off = np.asarray(official_kospi, float)
    y = np.asarray(dates.year)
    for yy in sorted(set(y.tolist())):
        if yy in year_end and yy - 1 in year_end:
            m = y == yy
            lr = np.log1p(k[m])
            k[m] = np.expm1(lr + (np.log(year_end[yy] / year_end[yy - 1]) - lr.sum()) / m.sum())
    use = np.isfinite(off) & np.asarray(dates >= pd.Timestamp(since))
    k[use] = off[use]
    return k


def window_mask(dates: pd.DatetimeIndex, lo: str, hi: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))


def sub_masks(dates: pd.DatetimeIndex) -> dict:
    """창 안 부분: 2020~24(2020 ≤ 연도 ≤ 2024)·2025 제외."""
    y = np.asarray(dates.year)
    return {"2024": (y >= 2020) & (y <= 2024), "ex25": y != 2025}


def wstats(r: np.ndarray, cash: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """한 창의 날별 수익 → CAGR·MDD·Calmar·월 Sharpe + 2020~24·2025 제외 CAGR·MDD + 같은 구간 현금 CAGR."""
    from krx_quant_core.stats.metrics import ann_sharpe  # noqa: PLC0415

    s = mf.equity_stats(r)
    mo = pd.Series(r, dates).groupby(dates.to_period("M")).apply(lambda x: float(np.prod(1 + x) - 1)).to_numpy()
    s["sharpe_m"] = ann_sharpe(mo, 12)
    s["cash_cagr"] = mf.equity_stats(cash)["cagr"]
    for tag, m in sub_masks(dates).items():
        e = mf.equity_stats(r[m])
        s.update({f"cagr_{tag}": e["cagr"], f"mdd_{tag}": e["mdd"], f"cash_{tag}": mf.equity_stats(cash[m])["cagr"]})
    return s


# ---------------------------------------------------------------------------
# DB 파생 입력 — 한 번 만들어 스크래치에 .npy 로(종목 × 날)
# ---------------------------------------------------------------------------

def build_inputs(path: str, t0: float) -> dict:
    """DB(읽기 전용) → 가격(정규장 보정판 — ``relax_run.KrxPatch`` 가 쓰는 ``krx_session``)·H5·추세 템플릿·EPS YoY·PEAD 순위·ADV20·
    시장·비용표·지수 대용(재구축 식 :func:`swing_it.features.regime.tv_weighted_index`). 있으면 mmap 으로 읽는다."""
    mp = os.path.join(path, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
        arr = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r") for k in GRID_KEYS + CODE_KEYS + DAY_KEYS}
        arr["days"] = pd.DatetimeIndex(pd.to_datetime(meta["days"]))
        arr["codes"] = pd.Index(meta["codes"])
        arr["info"] = meta["info"]
        return arr
    import convergence_run as cvr  # noqa: PLC0415
    import sepa_confluence_gate as scg  # noqa: PLC0415
    from prop_swing_common import load_env_db  # noqa: PLC0415
    from relax_run import KrxPatch  # noqa: PLC0415

    import research.signals.sepa as sepa_mod  # noqa: PLC0415
    from research.signals.sepa import _Book  # noqa: PLC0415
    from swing_it.features.earnings_leg import earnings_leg_panel  # noqa: PLC0415
    from swing_it.features.korea import cost_rt, load_listing_meta, market_of_code, universe_mask  # noqa: PLC0415
    from swing_it.features.minervini import trend_template  # noqa: PLC0415
    from swing_it.features.regime import MIN_INDEX_N, tv_weighted_index  # noqa: PLC0415
    from swing_it.features.technical import average_trade_value  # noqa: PLC0415
    from swing_it.krx_session import SESSION_CLOSE  # noqa: PLC0415
    from swing_it.live.features import EARN_COLS, eps_adjusted_earnings, pead_rank_panel  # noqa: PLC0415
    from swing_it.storage import connect, db_default, read_earnings  # noqa: PLC0415

    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    load_env_db()
    con = connect(db_default())
    patch = KrxPatch(con)
    with patch.installed():
        p = sepa_mod.load_panels(con, start=scg.PANEL_START)
        raw_close = cvr.raw_close_panel(con, p)
    lmeta = load_listing_meta(con)
    ea = read_earnings(con, all_versions=True, cols=EARN_COLS)
    from swing_it.storage import read_shares  # noqa: PLC0415

    shares = read_shares(con)                       # 정문(린트 (k))
    con.close()
    days, codes = p.days, p.close.columns
    _log(f"DB 패널 {p.close.shape} {days[0].date()} ~ {days[-1].date()} · 정규장 종가 컷 {SESSION_CLOSE}", t0)
    h5 = universe_mask(p.close, p.trade_value, lmeta, raw_close=raw_close, adv_floor=cvr.PEAD_POP_FLOOR)
    tt = trend_template(p.close, high=p.high, low=p.low)
    _log("H5·추세 템플릿", t0)
    ep = earnings_leg_panel(eps_adjusted_earnings(ea, shares, p.close, raw_close), [d.strftime("%Y-%m-%d") for d in days])
    pead = pead_rank_panel(ep, h5)
    di = days.get_indexer(pd.to_datetime(ep["date"]))
    ci = codes.get_indexer(ep["code"])
    v = ep["yoy"].to_numpy(float)
    ok = (di >= 0) & (ci >= 0) & np.isfinite(v)
    eps = np.full((len(days), len(codes)), np.nan)
    eps[di[ok], ci[ok]] = v[ok]
    eps_info = {"rows": int(len(ep)), "eps_from_shares": float((ep["op_yoy"] == 1).mean()) if len(ep) else np.nan}
    del ep, ea, shares
    _log("EPS YoY·PEAD 순위(eligible = H5)", t0)
    tv = tv_weighted_index(p.close, p.trade_value, market_of_code(lmeta, codes))
    proxy = tv["index_ret"].where(tv["index_n"] >= MIN_INDEX_N)[list(PROXY_MK)]
    book = _Book(p.open, p.high, p.low, p.close, p.close, p.volume)
    adv20 = average_trade_value(p.trade_value, 20)

    def T(df, dtype=float):
        return np.ascontiguousarray(df.reindex(index=days, columns=codes).to_numpy(dtype=float).T.astype(dtype))
    arr = {"O": book.O, "H": book.H, "L": book.L, "C": book.C, "TR": book.tradable, "h5": T(h5.fillna(False), bool),
           "tt": T(tt.fillna(False), bool), "eps": np.ascontiguousarray(eps.T), "pead": T(pead), "adv20": T(adv20),
           "market": p.market.reindex(codes).map(MARKET_CODE).fillna(-1).to_numpy(np.int8),
           "last_present": np.ascontiguousarray(book.present[:, -1]),
           "cost_tab": np.array([[cost_rt(d, "KOSPI"), cost_rt(d, "KOSDAQ")] for d in days]),
           "proxy": np.ascontiguousarray(proxy.to_numpy(float))}
    os.makedirs(path, exist_ok=True)
    for k, a in arr.items():
        np.save(os.path.join(path, f"{k}.npy"), a)
    info = {"nxt": patch.stats, "session_close": SESSION_CLOSE, "panel": [str(days[0].date()), str(days[-1].date())],
            "n_codes": int(len(codes)), "eps": eps_info, "built": time.strftime("%Y-%m-%d %H:%M")}
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump({"days": [str(d.date()) for d in days], "codes": list(map(str, codes)), "info": info}, fh, default=str)
    _log("DB 파생 입력 저장", t0)
    return build_inputs(path, t0)


# ---------------------------------------------------------------------------
# 계산 문맥 — 월 북(P2 경로 재사용)
# ---------------------------------------------------------------------------

class World:
    """정리된 가격·잠긴 하한가·리밸런스 신호일·순위 목록(Q1·Q2)·체결 가능 표시 + 기준선 원재료."""

    def __init__(self, inp: dict):
        self.inp = inp
        self.days, self.codes = inp["days"], inp["codes"]
        n = self.n_days = len(self.days)
        O2, H2, L2, C2, bad = mf.clean_jumps(np.asarray(inp["O"]), np.asarray(inp["H"]), np.asarray(inp["L"]),
                                             np.asarray(inp["C"]), JUMP_LIMIT)
        self.B = {"O": np.ascontiguousarray(O2), "H": np.ascontiguousarray(H2), "L": np.ascontiguousarray(L2),
                  "C": np.ascontiguousarray(C2)}
        del O2, H2, L2, C2
        self.bad = bad
        self.TR = np.ascontiguousarray(np.asarray(inp["TR"], bool))
        self.LD = np.ascontiguousarray(mf.locked_down(self.B["O"], self.B["H"], self.B["L"], self.B["C"], self.TR))
        has = self.TR.any(axis=1)
        self.last_tr = np.where(has, self.TR.shape[1] - 1 - np.argmax(self.TR[:, ::-1], axis=1), -1).astype(np.int64)
        self.last_present = np.ascontiguousarray(np.asarray(inp["last_present"], bool))
        self.market = np.asarray(inp["market"])
        self.cost_tab = np.asarray(inp["cost_tab"])
        per = self.days.to_period("M")
        month_id = (per.year * 12 + per.month - (per[0].year * 12 + per[0].month)).to_numpy(np.int64)
        lo, hi = pd.Timestamp(W_FULL[0]), pd.Timestamp(W_FULL[1])
        me = mg1.month_ends(month_id)
        me = me[me + 1 < n]
        ent = self.days[me + 1]
        self.rb_d = me[np.asarray((ent >= lo) & (ent <= hi))].astype(np.int64)
        self.cut = int(self.days.searchsorted(hi, "right"))           # 마지막 날 다음 행 — 그 전날 종가 시가평가
        self.t0, self.t1 = int(self.rb_d[0]), self.cut
        self.dates = self.days[self.t0 + 1:self.t1]
        self.mon = mg2.month_ids(self.dates)
        self.cash = mg2.cash_daily(self.dates, CD91_EXT)
        h5, tt, pead = np.asarray(inp["h5"]), np.asarray(inp["tt"]), np.asarray(inp["pead"])
        self.lists = {"Q1": self._lists(pead, h5), "Q2": self._lists(pead, h5 & tt)}
        self.proxy = np.asarray(inp["proxy"])[self.t0 + 1:self.t1]

    def _lists(self, pead: np.ndarray, ok: np.ndarray) -> dict:
        """리밸런스마다 순위 목록(종목) → 쌍 배열(pc·pd·체결 가능)·포인터·순위표."""
        K = len(self.rb_d)
        pcs, ptr = [], [0]
        rank_kc = np.full((K, len(self.codes)), BIG, np.int32)
        for k, d in enumerate(self.rb_d):
            cs = ranked_codes(pead[:, d], ok[:, d])
            pcs.append(cs)
            rank_kc[k, cs] = np.arange(len(cs))
            ptr.append(ptr[-1] + len(cs))
        pc = np.concatenate(pcs).astype(np.int64)
        pdv = np.repeat(self.rb_d, np.diff(ptr)).astype(np.int64)
        e = pdv + 1
        o_e = self.B["O"][pc, e]
        lu = self.TR[pc, e] & lu_open(o_e, self.B["C"][pc, pdv])
        fill = self.TR[pc, e] & np.isfinite(o_e) & (o_e > 0) & ~lu
        return {"pc": pc, "pd": pdv, "fill": np.ascontiguousarray(fill), "ptr": np.array(ptr, np.int64),
                "pair": np.arange(len(pc), dtype=np.int64), "rank_kc": rank_kc, "n": np.diff(ptr), "lu": lu}

    def random_ranks(self, L: dict, rng: np.random.Generator) -> tuple:
        """널: 리밸런스마다 같은 순위 목록의 순서를 무작위로(PEAD 순위 대신) — 메타 세대 1 ``Ctx.random_ranks`` 와 같은 방식."""
        rank_kc = np.full_like(L["rank_kc"], BIG)
        pair = L["pair"].copy()
        for k in range(len(self.rb_d)):
            a, b = int(L["ptr"][k]), int(L["ptr"][k + 1])
            pair[a:b] = rng.permutation(pair[a:b])
            rank_kc[k, L["pc"][pair[a:b]]] = np.arange(b - a)
        return pair, rank_kc

    def run(self, kind: str, cost_mult: float = 1.0, ranks: "tuple | None" = None) -> dict:
        """월 북 한 벌 → 날별 수익(t0+1 ~ t1−1)·총 노출·진입 행. ``ranks`` = (pair 순서, 순위표) 바꿔 끼우기(널)."""
        L = self.lists[kind]
        pair, rank_kc = ranks if ranks is not None else (L["pair"], L["rank_kc"])
        K = len(self.rb_d)
        sel, sk, rk = [], [], []
        for k in range(K):
            a = int(L["ptr"][k])
            nk = int(min(TOP_N, L["ptr"][k + 1] - a))
            if nk <= 0:
                continue
            sel.append(pair[a:a + nk])
            sk.append(np.full(nk, k, np.int64))
            rk.append(np.arange(nk, dtype=np.float64))
        sel, sk, rk = np.concatenate(sel), np.concatenate(sk), np.concatenate(rk)
        B = self.B
        N = np.full(K, KEEP_N, np.int64)
        x, ret, reason, early, xopen = mf.p2_trades(sel, sk, L["pc"], L["pd"], self.rb_d, N, rank_kc, B["O"], B["H"], B["L"],
                                                    B["C"], self.TR, self.LD, self.last_tr, self.last_present, L["fill"],
                                                    np.full(len(sel), self.cut, np.int64), NO_STOP)
        mk = np.maximum(self.market[L["pc"][sel]], 0)
        cost = np.where(x >= 0, self.cost_tab[np.maximum(x, 0), mk] * cost_mult, 0.0)
        n = len(sel)
        cd_pd = np.ascontiguousarray(L["pd"][sel])
        lo, hi = mf.day_bounds(cd_pd, self.n_days)
        cap = np.full(self.n_days, TOP_N, np.int64)
        raw, exc, gross, opened, n_ord, n_unm = mf.portfolio_meta(
            cd_pd, np.ascontiguousarray(L["pc"][sel]), np.ascontiguousarray(L["fill"][sel] & (x >= 0)), rk, np.zeros(n), x,
            np.full((n, 2), -1, np.int64), np.zeros((n, 2)), np.full((n, 2), np.nan), np.ones(n), ret, np.full(n, np.nan),
            cost, early, lo, hi, cap, B["O"], B["C"], self.t0, self.t1, SLOT_W, 2 * TOP_N)
        r = raw[1:]
        rc = mg2.with_cash(r, gross[:-1], self.cash)
        return {"r": r, "rc": rc, "gross": gross[1:], "opened": opened, "n_ord": int(n_ord), "sel": sel, "x": x, "ret": ret,
                "reason": reason, "xopen": xopen, "cost": cost, "kind": kind}

    def q3(self, rc_q1: np.ndarray) -> np.ndarray:
        """Q3 = 20% Q1(현금 금리 포함) + 80% 대용 ALL, 매월 되돌림."""
        return mix_monthly(np.column_stack([rc_q1, np.nan_to_num(self.proxy[:, 2])]), [SAT_W, 1.0 - SAT_W], self.mon)


# ---------------------------------------------------------------------------
# 성적 · 널 · 용량
# ---------------------------------------------------------------------------

def series_of(W: World, res: dict, F: "pd.DataFrame | None") -> dict:
    """한 절차의 창별 숫자(현금 금리 포함 = 주 시계열, 현금 0 = 참고)."""
    out = {}
    for wk, (lo, hi) in WINDOWS.items():
        m = window_mask(W.dates, lo, hi)
        d = W.dates[m]
        s = wstats(res["rc"][m], W.cash[m], d)
        s["nocash"] = mf.equity_stats(res["r"][m]) if res.get("r") is not None else None
        if F is not None:
            s["fac"] = gp.factor_regress(pd.Series(res["rc"][m], d), F)
            if res.get("r") is not None:
                s["fac_nocash"] = gp.factor_regress(pd.Series(res["r"][m], d), F)
        out[wk] = s
    return out


def book_diag(W: World, res: dict) -> dict:
    """회전율·보유·노출·체결 점검(전체 창)."""
    L = W.lists[res["kind"]]
    op = res["opened"]
    sel = res["sel"]
    years = len(W.dates) / 252.0
    pdv = L["pd"][sel][op]
    x = res["x"][op]
    net = res["ret"][op] - res["cost"][op]
    rs = res["reason"][op]
    names = {**{i: nm for i, nm in enumerate(gf.REASON_NAMES)}, **mf.REASON_EXTRA}
    g = res["gross"]
    return {"entries": int(len(op)), "orders": res["n_ord"], "turnover_oneway": SLOT_W * len(op) / years,
            "hold_med": float(np.median(x - pdv)) if len(op) else np.nan, "trade_net": float(np.mean(net)) if len(op) else np.nan,
            "trade_win": float((net > 0).mean()) if len(op) else np.nan, "expo": float(np.nanmean(g)), "expo_max": float(np.nanmax(g)),
            "expo_min": float(np.nanmin(g[(W.dates >= W.days[int(pdv.min()) + 1]) & (W.dates < W.dates[-1])])) if len(op) else np.nan,
            "reasons": {names.get(int(k), str(k)): int(v) for k, v in zip(*np.unique(rs, return_counts=True), strict=True)},
            "elig_med": float(np.median(L["n"])), "elig_min": int(L["n"].min()), "short_months": int((L["n"] < TOP_N).sum()),
            "lu_skip": int(L["lu"][sel].sum()), "ld_delay": int(sum(bool(W.LD[c, int(d) + 1:int(xx)].any())
                                                   for c, d, xx in zip(L["pc"][sel][op], pdv, x, strict=True)))}


def null_task(t: dict) -> dict:
    """무작위 20종목 널 한 묶음 — 같은 목록·같은 유지 규칙, 비용 1·2배. Q1 널은 Q3 널(20% + 80% 대용)도 만든다."""
    W = _C["W"]
    out = []
    for rep in t["reps"]:
        rng = np.random.default_rng([SEED, PROCS.index(t["kind"]), rep])
        rk = W.random_ranks(W.lists[t["kind"]], rng)
        r1 = W.run(t["kind"], 1.0, rk)
        r2 = W.run(t["kind"], 2.0, rk)
        kinds = [(t["kind"], r1["rc"], r2["rc"])]
        if t["kind"] == "Q1":
            kinds.append(("Q3", W.q3(r1["rc"]), W.q3(r2["rc"])))
        for nm, a, b in kinds:
            row = {"proc": nm, "rep": rep}
            for wk, (lo, hi) in WINDOWS.items():
                m = window_mask(W.dates, lo, hi)
                s = wstats(a[m], W.cash[m], W.dates[m])
                row.update({f"{wk}|{k}": s[k] for k in ("cagr", "mdd", "calmar", "cagr_2024", "cagr_ex25")})
                row[f"{wk}|cagr_x2"] = mf.equity_stats(b[m])["cagr"]
            out.append(row)
    return {"rows": out}


_C: dict = {}


def worker_init(inputs: str, threads: int = 1) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(threads)
    _C["W"] = World(build_inputs(inputs, time.time()))


def capacity(W: World, res: dict, w: float, t0: float) -> dict:
    """시가 체결(진입 + 리밸런스 매도) 주문액(자본 K × 종목 비중 ``w``) — (a) 신호일 ADV20 대비(전체 창), (b) 2025-08+ 그날 09:00 분봉
    거래대금(시가 동시호가 체결 포함) 대비. 자본 K 는 그 거래 시점 계좌 크기(복리 성장 반영 안 함 — 메타 세대 2 와 같다)."""
    L = W.lists[res["kind"]]
    sel, op = res["sel"], res["opened"]
    pc, pdv, x = L["pc"][sel], L["pd"][sel], res["x"]
    ent = pd.DataFrame({"code": np.asarray(W.codes)[pc[op]], "day": W.days[pdv[op] + 1], "sig": pdv[op], "c": pc[op], "side": "매수"})
    sm = (res["reason"] == mf.R_REBAL) & res["xopen"] & (x >= 0)
    sm[np.setdiff1d(np.arange(len(sel)), op)] = False
    xs = pd.DataFrame({"code": np.asarray(W.codes)[pc[sm]], "day": W.days[x[sm]], "sig": x[sm] - 1, "c": pc[sm], "side": "매도"})
    tr = pd.concat([ent, xs], ignore_index=True)
    adv = np.asarray(W.inp["adv20"])[tr["c"].to_numpy(), tr["sig"].to_numpy()] * 1e6
    adv = np.where(adv > 0, adv, np.nan)
    out = {"n": int(len(tr)), "n_buy": int(len(ent)), "n_sell": int(len(xs))}
    for K in CAPITALS:
        ratio = w * K / adv
        ratio = ratio[np.isfinite(ratio)]
        out[f"adv_{int(K / 1e8)}"] = (float(np.median(ratio)), float(np.percentile(ratio, 90))) if len(ratio) else (np.nan, np.nan)
    t = tr[tr["day"] >= pd.Timestamp(MINUTE_SINCE)]
    out["n_min"] = int(len(t))
    if not len(t):
        return out
    try:
        from swing_it.storage import connect, db_default  # noqa: PLC0415

        warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
        con = connect(db_default())
        need = pd.DataFrame({"code": t["code"].to_numpy(), "ts": t["day"] + pd.Timedelta(hours=9)})
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
        out["tv_med"] = float(j["trade_value"].astype(float).median())
        for K in CAPITALS:
            ratio = (w * K / j["trade_value"].astype(float)).to_numpy(float)
            ratio = ratio[np.isfinite(ratio)]
            out[f"open_{int(K / 1e8)}"] = ((float(np.median(ratio)), float(np.percentile(ratio, 90)), float((ratio > 0.1).mean()),
                                           float((ratio > 0.3).mean())) if len(ratio) else None)
    except Exception as e:  # noqa: BLE001 — 용량 표는 보조; DB 가 없으면 적고 넘어간다
        out["db_error"] = type(e).__name__
        _log(f"용량 표 DB 오류: {type(e).__name__}", t0)
    return out


def parity(W: World, cache: "str | None") -> dict:
    """대조: (1) 지수 대용 재계산 vs 국면 npz(2026-09-14 전 날), (2) 사전계산 pre_v2(있으면)와 가격·H5 쌍·EPS YoY·추세 템플릿(2026-09-14 전)."""
    from swing_it.krx_session import NXT_START  # noqa: PLC0415

    out = {}
    reg = mg1.load_regime(W.days)
    old = reg["proxy"]
    new = np.asarray(W.inp["proxy"])
    pre_nxt = np.asarray(W.days < pd.Timestamp(NXT_START))
    late = np.asarray((W.days >= pd.Timestamp(NXT_START)) & (W.days <= pd.Timestamp(W_FULL[1])))
    both = np.isfinite(old) & np.isfinite(new)
    out["proxy_maxdiff_pre"] = float(np.nanmax(np.abs(old - new)[both & pre_nxt[:, None]]))
    out["proxy_maxdiff_late"] = float(np.nanmax(np.abs(old - new)[both & late[:, None]])) if (both & late[:, None]).any() else np.nan
    out["official"] = reg["official"]
    if not cache:
        return out
    pre = g1.Pre.load(cache)
    if not (len(pre.days) == len(W.days) and (pre.days == W.days).all() and (pre.codes == W.codes).all()):
        out["pre_grid"] = "격자 다름"
        return out
    cut = int(W.days.searchsorted(pd.Timestamp(NXT_START)))
    a, b = np.asarray(pre.C)[:, :cut], np.asarray(W.inp["C"])[:, :cut]
    m = np.isfinite(a) & np.isfinite(b)
    out["close_maxrel_pre"] = float(np.max(np.abs(a[m] / b[m] - 1)))
    a, b = np.asarray(pre.C)[:, cut:], np.asarray(W.inp["C"])[:, cut:]
    m = np.isfinite(a) & np.isfinite(b)
    out["close_cells_diff_late"] = int((np.abs(a[m] / b[m] - 1) > 1e-6).sum())
    pdv, pcv = np.asarray(pre.pd), np.asarray(pre.pc)
    keep = pdv < cut
    pdv, pcv = pdv[keep], pcv[keep]
    h5 = np.asarray(W.inp["h5"])
    lo_d, hi_d = int(pdv.min()), int(pdv.max())
    mine = h5[:, lo_d:hi_d + 1].sum()
    out["h5_pairs"] = (int(len(pdv)), int(mine), int(h5[pcv, pdv].sum()))
    e1, e2 = np.asarray(pre.eps_yoy)[keep], np.asarray(W.inp["eps"])[pcv, pdv]
    fe = np.isfinite(e1) & np.isfinite(e2)
    out["eps_same"] = (int(fe.sum()), int((np.isfinite(e1) != np.isfinite(e2)).sum()),
                       float(np.max(np.abs(e1[fe] - e2[fe]))) if fe.any() else np.nan)
    out["tt_mismatch"] = int((np.asarray(pre.tt)[keep] != np.asarray(W.inp["tt"])[pcv, pdv]).sum())
    return out


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    import concurrent.futures as cf
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, help="DB 파생 배열 캐시 디렉터리(없으면 만든다)")
    ap.add_argument("--factors", required=True, help="리뷰어 A factors.py 산출 pickle")
    ap.add_argument("--cache", default=None, help="ga_run 사전계산(pre_v2) — 대조에만")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=8)
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
    inp = build_inputs(args.inputs, t0)
    t_inp = time.time() - t0
    W = World(inp)
    _log(f"문맥: 날 {W.n_days} · 종목 {len(W.codes)} · 리밸런스 {len(W.rb_d)} ({W.days[W.rb_d[0] + 1].date()} ~ "
         f"{W.days[W.rb_d[-1] + 1].date()}) · 수익 {W.dates[0].date()} ~ {W.dates[-1].date()} · 깨진 조정 일봉 {int(W.bad.sum()):,} · "
         f"잠긴 하한가 {int(W.LD.sum()):,}", t0)
    par = parity(W, args.cache)
    off = par.pop("official")
    _log(f"대조: {par}", t0)
    F = pd.read_pickle(args.factors)
    F.index = pd.to_datetime(F.index)
    dates = W.dates

    # --- 실제 절차 ---
    real = {}
    for kind in ("Q1", "Q2"):
        for cm in (1.0, 2.0):
            real[(kind, cm)] = W.run(kind, cm)
    for cm in (1.0, 2.0):
        real[("Q3", cm)] = {"rc": W.q3(real[("Q1", cm)]["rc"]), "r": None, "kind": "Q3"}
    stats = {(k, cm): series_of(W, real[(k, cm)], F if cm == 1.0 else None) for k in PROCS for cm in (1.0, 2.0)}
    diag = {k: book_diag(W, real[(k, 1.0)]) for k in ("Q1", "Q2")}
    t_real = time.time() - t0
    _log("실제 절차 끝", t0)

    # --- 기준선 ---
    h5 = np.asarray(inp["h5"])
    md, mc = np.nonzero(h5.T)
    bench = {f"대용 {m}": np.nan_to_num(W.proxy[:, j]) for j, m in enumerate(PROXY_MK)}
    bench["유니버스 월 등가중"] = mf.ew_monthly(np.asarray(inp["C"]), md.astype(np.int64), mc.astype(np.int64), W.rb_d, W.t0 + 1, W.t1,
                                         JUMP_LIMIT)
    bench["현금(CD91 근사)"] = W.cash
    bench_s = {k: series_of(W, {"rc": np.nan_to_num(v), "r": None}, None if k.startswith("현금") else F) for k, v in bench.items()}
    off_w = np.asarray(off)[W.t0 + 1:W.t1]
    offc = {k: mg1.official_cmp(real[(k, 1.0)]["rc"], dates, off_w) for k in PROCS}
    offc.update({k: mg1.official_cmp(np.nan_to_num(v), dates, off_w) for k, v in bench.items()})
    _log("기준선 끝", t0)

    # --- B1(같은 MDD 의 대용 ALL + 현금) · 짝지은 월 블록 부트스트랩 ---
    b1 = {}
    for k in PROCS:
        for wk, (lo, hi) in WINDOWS.items():
            m = window_mask(dates, lo, hi)
            mon = mg2.month_ids(dates[m])
            a = real[(k, 1.0)]["rc"][m]
            x = mdd_match_weight(bench["대용 ALL"][m], W.cash[m], stats[(k, 1.0)][wk]["mdd"], mon)
            rb = mix_monthly(np.column_stack([bench["대용 ALL"][m], W.cash[m]]), [x, 1.0 - x], mon)
            b1[(k, wk)] = {"x": x, "stats": wstats(rb, W.cash[m], dates[m]), "boot": boot_beat(a, rb, mon),
                           "boot_x2": boot_beat(real[(k, 2.0)]["rc"][m], rb, mon), "r": rb}
    # 참고(설계 밖): B1-K = 코스피 혼성(리뷰어 방식) x% + 현금 — 비교 창만(연말 값이 2019+ 만 있다)
    kh = kospi_hybrid(W.proxy[:, 0], off_w[:, 0], dates, KOSPI_YE)
    b1k = {}
    m = window_mask(dates, *W_CMP)
    mon = mg2.month_ids(dates[m])
    kstats = wstats(kh[m], W.cash[m], dates[m])
    for k in PROCS:
        x = mdd_match_weight(kh[m], W.cash[m], stats[(k, 1.0)]["cmp"]["mdd"], mon)
        rb = mix_monthly(np.column_stack([kh[m], W.cash[m]]), [x, 1.0 - x], mon)
        b1k[k] = {"x": x, "stats": wstats(rb, W.cash[m], dates[m]), "boot": boot_beat(real[(k, 1.0)]["rc"][m], rb, mon)}
    fac_q3 = {"Q3": stats[("Q3", 1.0)]["cmp"].get("fac", {}).get("alpha_ann"),
              "Q1": stats[("Q1", 1.0)]["cmp"].get("fac", {}).get("alpha_ann"),
              "ALL": bench_s["대용 ALL"]["cmp"].get("fac", {}).get("alpha_ann")}
    first_h5 = W.days[int(np.argmax(np.asarray(inp["h5"]).any(axis=0)))]
    _log("B1 끝", t0)

    # --- 널 ---
    n_null = args.nulls
    nb = 10
    rows = []
    mpc = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mpc, initializer=worker_init,
                                initargs=(args.inputs, 1)) as ex:
        futs = [ex.submit(null_task, {"kind": kind, "reps": list(range(a, min(a + nb, n_null)))})
                for kind in ("Q1", "Q2") for a in range(0, n_null, nb)]
        for i, f in enumerate(cf.as_completed(futs), 1):
            rows += f.result()["rows"]
            if i % 10 == 0 or i == len(futs):
                _log(f"널 묶음 {i}/{len(futs)}", t0)
    ndf = pd.DataFrame(rows).sort_values(["proc", "rep"]).reset_index(drop=True)
    t_null = time.time() - t0 - t_real

    # --- 층 산술 ---
    dec = []
    for k in PROCS:
        nd = ndf[ndf["proc"] == k]
        for wk in WINDOWS:
            s, s2 = stats[(k, 1.0)][wk], stats[(k, 2.0)][wk]
            fac = s.get("fac", {})
            p95 = float(np.nanpercentile(nd[f"{wk}|cagr"], 95)) if len(nd) else np.nan
            d = {"proc": k, "win": wk, "alpha_t": fac.get("alpha_t", np.nan), "alpha_ann": fac.get("alpha_ann", np.nan),
                 "alpha_t_nocash": s.get("fac_nocash", {}).get("alpha_t", np.nan), "cagr": s["cagr"], "mdd": s["mdd"],
                 "calmar": s["calmar"], "null_p50": float(np.nanmedian(nd[f"{wk}|cagr"])) if len(nd) else np.nan, "null_p95": p95,
                 "null_pct": mg1.null_pct(s["cagr"], nd[f"{wk}|cagr"]), "null_pct_calmar": mg1.null_pct(s["calmar"], nd[f"{wk}|calmar"]),
                 "p_b1": b1[(k, wk)]["boot"]["p_cagr"], "p_b1_calmar": b1[(k, wk)]["boot"]["p_calmar"], "b1_x": b1[(k, wk)]["x"],
                 "cagr_2024": s["cagr_2024"], "cash_2024": s["cash_2024"], "cagr_ex25": s["cagr_ex25"], "cash_ex25": s["cash_ex25"],
                 "cagr_x2": s2["cagr"], "cash": s["cash_cagr"], "n_null": int(len(nd))}
            d.update(tier1(d["alpha_t"], d["cagr"], d["null_p95"]))
            d.update(tier2(d["p_b1"], d["cagr_2024"], d["cash_2024"], d["cagr_ex25"], d["cash_ex25"], d["cagr_x2"], d["cash"]))
            dec.append(d)
    cap_tab = {"Q1": capacity(W, real[("Q1", 1.0)], SLOT_W, t0), "Q2": capacity(W, real[("Q2", 1.0)], SLOT_W, t0)}
    cap_tab["Q3"] = capacity(W, real[("Q1", 1.0)], SAT_W * SLOT_W, t0)
    elapsed = time.time() - t0
    _log("용량 표 끝", t0)

    # --- 산출물 ---
    srows = []
    for k in PROCS:
        for cm in (1.0, 2.0):
            for wk in WINDOWS:
                s = stats[(k, cm)][wk]
                srows.append({"proc": k, "cost_mult": cm, "win": wk, **{c: s.get(c) for c in (
                    "cagr", "mdd", "calmar", "sharpe_m", "total", "cagr_2024", "mdd_2024", "cagr_ex25", "mdd_ex25", "cash_cagr",
                    "cash_2024", "cash_ex25")}, "alpha_t": s.get("fac", {}).get("alpha_t"), "alpha_ann": s.get("fac", {}).get("alpha_ann")})
    for k, s in bench_s.items():
        for wk in WINDOWS:
            srows.append({"proc": k, "cost_mult": 0.0, "win": wk, **{c: s[wk].get(c) for c in (
                "cagr", "mdd", "calmar", "sharpe_m", "total", "cagr_2024", "mdd_2024", "cagr_ex25", "mdd_ex25", "cash_cagr")},
                "alpha_t": s[wk].get("fac", {}).get("alpha_t"), "alpha_ann": s[wk].get("fac", {}).get("alpha_ann")})
    pd.DataFrame(srows).to_csv(os.path.join(sub, "summary.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(sub, "decision.csv"), index=False)
    ndf.to_csv(os.path.join(sub, "null.csv"), index=False)
    nav = pd.DataFrame(index=dates)
    for k in PROCS:
        for cm in (1.0, 2.0):
            nav[f"{k} ×{cm:g}"] = np.cumprod(1 + real[(k, cm)]["rc"])
    for k in ("Q1", "Q2"):
        nav[f"{k} 현금 0 ×1"] = np.cumprod(1 + real[(k, 1.0)]["r"])
        nav[f"{k} 총 노출"] = real[(k, 1.0)]["gross"]
    for k, v in bench.items():
        nav[k] = np.cumprod(1 + np.nan_to_num(v))
    for j, m in enumerate(("KOSPI", "KOSDAQ")):
        nav[f"공식 {m} 일수익"] = off_w[:, j]
    nav.to_csv(os.path.join(sub, "nav.csv"))
    yrows = []
    yrs = sorted(set(dates.year))
    allser = {**{k: real[(k, 1.0)]["rc"] for k in PROCS}, **{k: np.nan_to_num(v) for k, v in bench.items()}}
    for nm, r in allser.items():
        for y in yrs:
            m = np.asarray(dates.year == y)
            e = mf.equity_stats(r[m])
            yrows.append({"proc": nm, "year": y, "ret": e["total"], "mdd": e["mdd"]})
    ydf = pd.DataFrame(yrows)
    ydf.to_csv(os.path.join(sub, "peryear.csv"), index=False)
    led = write_ledger_rows(sub)
    L = render(locals())
    with open(os.path.join(args.out, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    pick = [f"{d['proc']}[{d['win']}] 층1 {d['t1']} 층2 {d['t2']}" for d in dec]
    print(f"\n=== 메타 세대 3 — {elapsed / 60:.1f}분 · " + " · ".join(pick) + " ===")
    print(f"→ {os.path.join(args.out, RESULTS_MD)}")
    return 0


def write_ledger_rows(sub: str) -> int:
    """원장 ``meta_gen3/TRIALS.jsonl`` — 절차(3) × 비용(2) 한 줄씩(결과 숫자는 넣지 않는다)."""
    path = os.path.join(sub, "TRIALS.jsonl")
    seen = g1.read_fingerprints(path)
    rows = [{"config": {"design": DESIGN_DOC, "procedure": k, "cost_mult": cm, "top_n": TOP_N, "keep_n": KEEP_N,
                        "satellite": SAT_W if k == "Q3" else None, "trend_template": k == "Q2", "universe": "H5 ADV20≥5,000",
                        "rank": "earnings_leg.pead_rank(EPS, eligible=H5)", "window": f"{W_FULL[0]}~{W_FULL[1]}"},
             "note": "메타 세대 3 — 고정 규칙 한 벌(폴드 적합 없음)"} for k in PROCS for cm in (1.0, 2.0)]
    return g1.write_ledger(path, rows, seen)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _ck(v) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def _q(a, x):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, x)) if np.isfinite(a).any() else np.nan


def render(v: dict) -> "list[str]":
    W, args, dec, stats, bench_s, b1 = v["W"], v["args"], v["dec"], v["stats"], v["bench_s"], v["b1"]
    ndf, diag, cap_tab, offc, par, ydf = v["ndf"], v["diag"], v["cap_tab"], v["offc"], v["par"], v["ydf"]
    info = W.inp["info"]
    D = {(d["proc"], d["win"]): d for d in dec}
    L = [f"# 메타 세대 3 — 분산 PEAD(Q1)·PEAD ∧ 추세(Q2)·위성 20%(Q3) — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "설계 `META_GEN3.md`(7608844, 결과 보기 전 고정)를 계산한 숫자다. ✓/✗ 는 미리 적은 문턱과의 **산술 비교**이고 판정은 사람이 한다. "
         "폴드 적합이 없는 고정 규칙이라 전 기간이 표본 내다 — PEAD 신호 자체가 이 저장소에서 2016~2026 자료로 찾은 것이므로 "
         "널·팩터 알파는 필요조건일 뿐이다.", "",
         f"재현: `NUMBA_NUM_THREADS=8 uv run --with numba --extra pg python research/experiments/meta_gen3_run.py --inputs <스크래치> "
         f"--factors <factors.pkl> --cache <pre_v2> --workers {args.workers}` · 실행 {v['elapsed'] / 60:.1f}분(입력 {v['t_inp'] / 60:.1f}분 · "
         f"실제 절차 {(v['t_real'] - v['t_inp']) / 60:.1f}분 · 널 {v['t_null'] / 60:.1f}분, 작업자 {args.workers}) · "
         f"numba {gf.numba.__version__ if gf.HAVE_NUMBA else '없음'}.", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 널 {args.nulls}개, 출력 `{args.out}`. 측정으로 쓰지 않는다.", ""]

    L += ["## 층 판정 산술 (META_GEN3.md '판정 (미리)')", "",
          "**층 1:** 팩터 알파 HAC t > 2 (현금 금리 포함 날별 순수익, KOSPI·KOSDAQ 등가중·UMD·SMB_ADV·LOWVOL, Newey-West 5 — 리뷰어 `factors.pkl`) "
          "∧ CAGR > 무작위 20종목 널 CAGR p95.  ",
          "**층 2(개인용):** (1) P(CAGR > B1) ≥ 0.8 — B1 = 같은 MDD 의 \"지수 대용 ALL x% + 현금\", 짝지은 월 블록 부트스트랩(3개월 블록, 2,000벌) · "
          "(2a) 2020~24 CAGR > 같은 구간 현금 · (2b) 2025 제외 CAGR > 같은 구간 현금 · (3) 비용 2배 CAGR > 현금. "
          f"**주 창 = {WIN_LABEL[PRIMARY_WIN]}**(설계의 '메타 세대 1·2 와 같은 비교 창'), 전체 창은 같은 산술을 병기.", "",
          "| 절차 | 창 | 알파/년 (t) | CAGR · 널 p50 · 널 p95 (백분위) | 층 1 | P(> B1) [x] · P(Calmar) | 2020~24 CAGR · 현금 | "
          "2025 제외 CAGR · 현금 | 비용 2× CAGR · 현금 | 층 2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in PROCS:
        for wk in (PRIMARY_WIN, "full"):
            d = D[(k, wk)]
            L.append(f"| {k} | {'**' + WIN_LABEL[wk] + '**' if wk == PRIMARY_WIN else WIN_LABEL[wk]} | {_pct(d['alpha_ann'])} ({_f(d['alpha_t'])}) "
                     f"{_ck(d['t1_alpha'])} | {_pct(d['cagr'])} · {_pct(d['null_p50'])} · {_pct(d['null_p95'])} ({_f(d['null_pct'], '{:.0%}')}) "
                     f"{_ck(d['t1_null'])} | **{_ck(d['t1'])}** | {_f(d['p_b1'])} [{_f(d['b1_x'], '{:.0%}')}] · {_f(d['p_b1_calmar'])} "
                     f"{_ck(d['t2_b1'])} | {_pct(d['cagr_2024'])} · {_pct(d['cash_2024'])} {_ck(d['t2_2024'])} | {_pct(d['cagr_ex25'])} · "
                     f"{_pct(d['cash_ex25'])} {_ck(d['t2_ex25'])} | {_pct(d['cagr_x2'])} · {_pct(d['cash'])} {_ck(d['t2_x2'])} | "
                     f"**{_ck(d['t2'])}** |")
    p1 = [k for k in PROCS if D[(k, PRIMARY_WIN)]["t1"]]
    p2 = [k for k in PROCS if D[(k, PRIMARY_WIN)]["t2"]]
    pf1 = [k for k in PROCS if D[(k, "full")]["t1"]]
    pf2 = [k for k in PROCS if D[(k, "full")]["t2"]]
    L += ["", f"**산술 결과(주 창):** 층 1 통과 {', '.join(p1) if p1 else '없음'} · 층 2 통과 {', '.join(p2) if p2 else '없음'}. "
          f"(전체 창: 층 1 {', '.join(pf1) if pf1 else '없음'} · 층 2 {', '.join(pf2) if pf2 else '없음'}.)"
          + ("" if (p1 or p2) else " → 설계 문구의 조건(\"둘 다 없으면 메타 멈춤 규칙 발동\")에 해당하는 **산술**이다."), "",
          "알파 t 참고(현금 0 — 시뮬 그대로의 순수익): " + " · ".join(
              f"{k}[{wk}] {_f(D[(k, wk)]['alpha_t_nocash'])}" for k in ("Q1", "Q2") for wk in (PRIMARY_WIN, "full")) + ". Q3 는 80% 대용이 "
          "회귀에 들어가 알파 대부분이 대용의 팩터 노출 잔차다(대용 자체 알파 t 는 기준선 표).", ""]

    L += ["## 절차 · 기준선 숫자 (비용 1×, 현금 금리 포함)", "",
          "CAGR = 날 수 기준(252), MDD = 초기 자산 1 을 첫 고점으로, Calmar = CAGR / |MDD|, 월 Sharpe = 월 복리 수익 × √12(무위험 차감 없음). "
          "B1 = 그 절차 MDD 에 맞춘 \"대용 ALL x% + 현금\"(매월 되돌림).", ""]
    for wk in (PRIMARY_WIN, "full"):
        L += [f"### {WIN_LABEL[wk]}", "",
              "| 시계열 | CAGR | MDD | Calmar | 월 Sharpe | 2020~24 CAGR · MDD | 2025 제외 CAGR · MDD | 비용 2× CAGR · MDD | 알파/년 (t) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in PROCS:
            s, s2 = stats[(k, 1.0)][wk], stats[(k, 2.0)][wk]
            fa = s.get("fac", {})
            L.append(f"| **{k}** | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe_m'])} | {_pct(s['cagr_2024'])} · "
                     f"{_pct(s['mdd_2024'])} | {_pct(s['cagr_ex25'])} · {_pct(s['mdd_ex25'])} | {_pct(s2['cagr'])} · {_pct(s2['mdd'])} | "
                     f"{_pct(fa.get('alpha_ann'))} ({_f(fa.get('alpha_t'))}) |")
            bb = b1[(k, wk)]
            s = bb["stats"]
            L.append(f"| B1 for {k} (대용 {bb['x']:.0%} + 현금) | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe_m'])} | "
                     f"{_pct(s['cagr_2024'])} · {_pct(s['mdd_2024'])} | {_pct(s['cagr_ex25'])} · {_pct(s['mdd_ex25'])} | — | — |")
        for nm, bs in bench_s.items():
            s = bs[wk]
            fa = s.get("fac", {})
            L.append(f"| B0 {nm} | {_pct(s['cagr'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe_m'])} | {_pct(s['cagr_2024'])} · "
                     f"{_pct(s['mdd_2024'])} | {_pct(s['cagr_ex25'])} · {_pct(s['mdd_ex25'])} | — | {_pct(fa.get('alpha_ann'))} ({_f(fa.get('alpha_t'))}) |")
        L.append("")
    L += ["- 지수 대용 = 재구축 거래대금(20일) 가중 지수(시장마다 단일 종목 ≤ 10%, |r| > 31% 종목 그날 제외) — 이 러너가 **정규장 보정 수정판 가격**으로 "
          "같은 함수(`regime.tv_weighted_index`)를 다시 돌린 값이다(아래 대조). 삼성전자·SK하이닉스 비중이 공식보다 낮아 **지수 수준으로 읽지 않는다**"
          "(REGIME_PANEL.md: 2024-01-26+ KOSPI 대용 +103.9% vs 공식 +167.3%).",
          "- 월 등가중 = H5(신호일 편입) 월 재조정, 비용 없음, |r| > 31% 날 수익 0(`meta_fast.ew_monthly`). 현금 = CD91 연평균 근사 / 252 매일.", ""]

    L += ["## B1 비교 상세 — 짝지은 월 블록 부트스트랩 (3개월 블록, 2,000벌, 비원형)", "",
          "| 절차 | 창 | x(대용 비중) | 절차 MDD · B1 MDD | 절차 CAGR · B1 CAGR | ΔCAGR [5%, 50%, 95%] | P(ΔCAGR > 0) | P(ΔCalmar > 0) | 비용 2× P(ΔCAGR > 0) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in PROCS:
        for wk in (PRIMARY_WIN, "full"):
            bb = b1[(k, wk)]
            s, sb, bo = stats[(k, 1.0)][wk], bb["stats"], bb["boot"]
            L.append(f"| {k} | {WIN_LABEL[wk]} | {bb['x']:.0%} | {_pct(s['mdd'])} · {_pct(sb['mdd'])} | {_pct(s['cagr'])} · {_pct(sb['cagr'])} | "
                     f"[{_pct(bo['dcagr_lo'])}, {_pct(bo['dcagr_med'])}, {_pct(bo['dcagr_hi'])}] | {bo['p_cagr']:.2f} | {bo['p_calmar']:.2f} | "
                     f"{bb['boot_x2']['p_cagr']:.2f} |")
    L += ["", "- x = 0.01 격자에서 B1 MDD 가 절차 MDD 보다 깊지 않은 최대 비중. 절차 MDD 가 대용 100% 보다 깊으면 x = 100%(같은 MDD 를 만들 수 없음 — 그대로 적는다).",
          "- '이김' 의 지표는 CAGR(같은 MDD 이므로 수익 비교) — Calmar 판(MDD 하한 1%)을 병기했다. 월 경로 CAGR = 12 / 달 수 연환산.", ""]

    b1k, kst, fq = v["b1k"], v["kstats"], v["fac_q3"]
    L += ["## 참고(설계 밖) — B1 의 지수를 '코스피 혼성' 으로 바꾼 민감도 (비교 창만)", "",
          "설계의 B1 은 재구축 대용 ALL 이다. 그 대용은 2024-01-26+ 공식 코스피보다 크게 덜 올랐다(아래 공식 지수 표) — 개인이 실제로 살 수 있는 지수 ETF 에 가까운 "
          "비교로 메타 세대 1 리뷰어(`review_meta1/indiv.py`) 방식의 코스피 혼성(2024-01-26+ 공식 일수익, 2020~2023 은 대용 KOSPI 에 연 상수 로그 드리프트를 더해 "
          "연말 종가 2019 2197.67 → 2023 2655.28 에 맞춤, 2024-01-02 ~ 01-25 는 대용 그대로; 가격지수라 배당 ~2%/년 빠짐)을 같은 산술에 넣었다. **층 판정에는 쓰지 않는다.**", "",
          f"코스피 혼성 자체(비교 창): CAGR {_pct(kst['cagr'])} · MDD {_pct(kst['mdd'])} · 2020~24 {_pct(kst['cagr_2024'])} · 2025 제외 {_pct(kst['cagr_ex25'])}.", "",
          "| 절차 | x(코스피 혼성 비중) | B1-K CAGR · MDD | B1-K 2020~24 · 2025 제외 | 절차 CAGR | ΔCAGR [5%, 50%, 95%] | P(ΔCAGR > 0) | P(ΔCalmar > 0) |",
          "|---|---|---|---|---|---|---|---|"]
    for k in PROCS:
        bb = b1k[k]
        s, bo = bb["stats"], bb["boot"]
        L.append(f"| {k} | {bb['x']:.0%} | {_pct(s['cagr'])} · {_pct(s['mdd'])} | {_pct(s['cagr_2024'])} · {_pct(s['cagr_ex25'])} | "
                 f"{_pct(stats[(k, 1.0)]['cmp']['cagr'])} | [{_pct(bo['dcagr_lo'])}, {_pct(bo['dcagr_med'])}, {_pct(bo['dcagr_hi'])}] | "
                 f"{bo['p_cagr']:.2f} | {bo['p_calmar']:.2f} |")
    L += ["", f"**Q3 알파 분해(비교 창, 연):** Q3 {_pct(fq['Q3'])} ≈ 20% × Q1 {_pct(fq['Q1'])} + 80% × 대용 ALL {_pct(fq['ALL'])} "
          f"= {_pct(SAT_W * (fq['Q1'] or np.nan) + (1 - SAT_W) * (fq['ALL'] or np.nan))} — 대용 ALL 자체가 이 팩터(등가중 시장) 대비 알파를 가진다"
          "(거래대금 가중 = 대형·유동 기울기). Q3 층 1 의 알파 t 는 PEAD 위성보다 대용의 몫이 크다.", ""]

    L += ["## 무작위 포트폴리오 널 (300벌, 같은 순위 목록 · 같은 유지 규칙)", "",
          "| 절차 | 창 | 널 | CAGR 중앙 [p5, p95] | 실제 CAGR (백분위) | Calmar 중앙 [p95] | 실제 Calmar (백분위) | 널 MDD 중앙 | 널 2025 제외 CAGR 중앙 · 실제 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in PROCS:
        nd = ndf[ndf["proc"] == k]
        kind = {"Q1": "H5 ∧ EPS YoY 있음 목록에서 무작위 20", "Q2": "H5 ∧ 추세 템플릿 ∧ EPS YoY 있음 목록에서 무작위 20",
                "Q3": "20% × Q1 널 + 80% 대용"}[k]
        for wk in (PRIMARY_WIN, "full"):
            d = D[(k, wk)]
            s = stats[(k, 1.0)][wk]
            L.append(f"| {k} | {WIN_LABEL[wk]} | {kind} (n {len(nd)}) | {_pct(_q(nd[f'{wk}|cagr'], 50))} [{_pct(_q(nd[f'{wk}|cagr'], 5))}, "
                     f"{_pct(_q(nd[f'{wk}|cagr'], 95))}] | {_pct(s['cagr'])} ({_f(d['null_pct'], '{:.0%}')}) | {_f(_q(nd[f'{wk}|calmar'], 50))} "
                     f"[{_f(_q(nd[f'{wk}|calmar'], 95))}] | {_f(s['calmar'])} ({_f(d['null_pct_calmar'], '{:.0%}')}) | {_pct(_q(nd[f'{wk}|mdd'], 50))} | "
                     f"{_pct(_q(nd[f'{wk}|cagr_ex25'], 50))} · {_pct(s['cagr_ex25'])} |")
    L += ["", "- 널 = 리밸런스마다 그 절차의 순위 목록(같은 자격 — H5 ∧ EPS YoY 유한, Q2 는 ∧ 추세 템플릿)을 무작위로 섞어 상위 20 진입·30 안이면 유지 — "
          "매달 새로 섞으므로 유지 확률이 낮아 실제보다 회전이 많다(같은 규칙을 무작위 순위에 건 결과). 체결·비용·현금 금리는 실제와 같다.",
          "- **널 해석 주의(독립 점검, 스크래치 `mg3/review/nullcheck.py` 40벌 — 설계 정의는 바꾸지 않음):** 이 널의 편도 회전율은 ~11.3/년(Q1 5.25)이라 비용 끌림이 "
          "~4.8%/년(Q1 ~2.4%)이다. 종목별 무작위 점수를 달마다 유지하는 널(회전 ~2.8/년)이면 비교 창 CAGR p95 ≈ +13.96% 로 Q1 실제(+11.53%)보다 높다 — "
          "층 1 의 '널 p95 위' 다리(특히 Q3)는 널 구성에 민감하다.", ""]

    L += ["## 북 점검 — 회전율·보유·체결 (전체 창, 비용 1×)", "",
          "| 절차 | 진입(주문) | 편도 회전율/년 | 보유 중앙(거래일) | 트레이드 순평균 · 승률 | 평균 총 노출 [최소(첫 진입 ~ 끝 전날), 최대] | 순위 목록 크기 중앙 · 최소 · 20 미만 달 | "
          "상한가 시가로 못 산 상위 20 후보 | 보유 중 잠긴 하한가가 있던 트레이드 | 청산 사유 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("Q1", "Q2"):
        g = diag[k]
        L.append(f"| {k} | {g['entries']} ({g['orders']}) | {g['turnover_oneway']:.2f} | {_f(g['hold_med'], '{:.0f}')} | {_pct(g['trade_net'])} · "
                 f"{_f(g['trade_win'], '{:.0%}')} | {_f(g['expo'], '{:.0%}')} [{_f(g['expo_min'], '{:.0%}')}, {_f(g['expo_max'], '{:.0%}')}] | "
                 f"{g['elig_med']:.0f} · {g['elig_min']} · {g['short_months']} | {g['lu_skip']} | {g['ld_delay']} | {g['reasons']} |")
    L += ["", f"- Q3 의 위성 회전율 = Q1 × {SAT_W:.0%}(+ 매월 20/80 되돌림 거래 — 비용 미반영, 대용은 ETF 가정). 편도 회전율 = 5% × 진입 수 / 년.",
          "- 총 노출 > 100%: 계속 보유 종목이 오른 상태에서 빈 슬롯을 그때 자산의 5% 로 채우면 생긴다(현금 제약 없음, 차입 이자 미반영 — 메타 세대 1 엔진과 같다). "
          "유휴 현금 금리는 max(0, 1 − 전일 총 노출) 에만 붙는다.", ""]

    L += ["## 공식 지수 대비 (2024-01-26 ~ 2026-09-30, 공식 코스피·코스닥 일수익률이 있는 날)", "",
          "| 시계열 | 날 수 | CAGR · MDD | 코스피 CAGR · MDD | 코스닥 CAGR · MDD | 코스피 대비 CAGR |", "|---|---|---|---|---|---|"]
    for nm, o in offc.items():
        L.append(f"| {nm} | {o['n']} | {_pct(o['proc']['cagr'])} · {_pct(o['proc']['mdd'])} | {_pct(o['KOSPI']['cagr'])} · {_pct(o['KOSPI']['mdd'])} | "
                 f"{_pct(o['KOSDAQ']['cagr'])} · {_pct(o['KOSDAQ']['mdd'])} | {_pct(o['proc']['cagr'] - o['KOSPI']['cagr'])} |")

    yrs = sorted(set(ydf["year"]))
    L += ["", "## 연도별 수익 (MDD) — 비용 1×, 현금 금리 포함", "", "| 시계열 | " + " | ".join(str(y) for y in yrs) + " |",
          "|---|" + "---|" * len(yrs)]
    for nm, g in ydf.groupby("proc", sort=False):
        g = g.set_index("year")
        L.append(f"| {nm} | " + " | ".join(f"{_pct(g.loc[y, 'ret'])} ({_pct(g.loc[y, 'mdd'])})" for y in yrs) + " |")
    L += ["", "(2017 = 9 ~ 12월, 2026 = 1 ~ 9월.)", ""]

    L += ["## 실행 용량 — 시가 체결(진입 + 리밸런스 매도) 주문액 / 그날 09:00 분봉 거래대금 (2025-08+)", "",
          "주문액 = 계좌 K × 종목 비중(Q1·Q2 5%, Q3 위성 1%). 09:00 분봉은 시가 동시호가 체결을 포함한다.", "",
          "| 절차 | 시가 체결(매수 · 매도) | 주문 / 신호일 ADV20 중앙 · p90 (1억 / 3억) | 2025-08+ 체결(분봉 찾음) | 1억: 중앙 · p90 · >10% · >30% | "
          "2억 | 3억 |", "|---|---|---|---|---|---|---|"]
    for nm, c in cap_tab.items():
        a1, a3 = c.get("adv_1", (np.nan, np.nan)), c.get("adv_3", (np.nan, np.nan))
        fo = lambda o: " · ".join(_f(x, "{:.1%}") for x in o) if o else (c.get("db_error") or "—")  # noqa: E731
        L.append(f"| {nm} | {c['n']} ({c['n_buy']} · {c['n_sell']}) | {_f(a1[0], '{:.3%}')} · {_f(a1[1], '{:.3%}')} / {_f(a3[0], '{:.3%}')} · "
                 f"{_f(a3[1], '{:.3%}')} | {c.get('n_min', 0)} ({c.get('n_min_found', 0)}) | {fo(c.get('open_1'))} | {fo(c.get('open_2'))} | "
                 f"{fo(c.get('open_3'))} |")
    L.append("")

    nx = info.get("nxt", [])
    L += ["## 검증·자원", "",
          f"- 입력: DB → `KrxPatch`(정규장 종가 컷 {info.get('session_close')} — 5eb2870 수정판) 가격 {info.get('panel')} · 종목 {info.get('n_codes')} · "
          "NXT 보정 " + " · ".join(f"{s['table']} 늦은 행 {s['rows_late']:,} 보정 {s['patched']:,} 미보정 {s['unpatched']:,}(분봉 없음 — 원값)" for s in nx) + f" · 만든 때 {info.get('built')}.",
          f"- 지수 대용 재계산 vs 국면 npz(옛 보정판): 2026-09-14 전 날 최대 차 {_f(par.get('proxy_maxdiff_pre'), '{:.1e}')}, 2026-09-14 ~ 09-30 최대 차 "
          f"{_f(par.get('proxy_maxdiff_late'), '{:.1e}')}(종가 컷 수정 효과).",
          (f"- 사전계산 pre_v2 대조(2026-09-14 전): 종가 최대 상대차 {_f(par.get('close_maxrel_pre'), '{:.1e}')} · H5 쌍 (pre · 이 러너 같은 구간 · pre 쌍 중 이 러너 H5) "
           f"{par.get('h5_pairs')} · EPS YoY (둘 다 유한 · 유한 여부 다름 · 최대 차) {par.get('eps_same')} · 추세 템플릿 불일치 {par.get('tt_mismatch')} · "
           f"2026-09-14+ 종가 다른 칸 {par.get('close_cells_diff_late')}(종가 컷 수정)." if "close_maxrel_pre" in par else
           f"- 사전계산 대조: {par.get('pre_grid', '하지 않음(--cache 없음)')}."),
          f"- 리밸런스 {len(W.rb_d)}회(신호일 {W.days[W.rb_d[0]].date()} ~ {W.days[W.rb_d[-1]].date()}), 수익 {W.dates[0].date()} ~ {W.dates[-1].date()} "
          f"({len(W.dates)}일), 마지막 날 종가 시가평가(비용 부과). 깨진 조정 일봉 {int(W.bad.sum()):,} · 잠긴 하한가 {int(W.LD.sum()):,} (종목-일, 전 패널).",
          f"- 산출물: `meta_gen3/` nav·summary·decision·null·peryear CSV, 원장 `meta_gen3/TRIALS.jsonl`(덧붙인 {v['led']}줄).", ""]

    L += ["## 검증된 PEAD 북(`docs/pead-strategy.md`·`strategies/pead.py`)과 Q1 의 차이", "",
          "| 항목 | 검증된 PEAD(문서) | Q1(이 설계) |", "|---|---|---|",
          "| 신호 | 분기 **순이익** YoY 횡단면 순위(`earnings_yoy_panel`) | **EPS** YoY(분기 말 분할조정 주식수, 없으면 순이익 대체 — GA 커널 정의) 순위, eligible = 그날 H5 |",
          "| 유니버스 | ADV ≥ 200억(실전형) · 중대형(시총 21~60위) 집중 권장 | H5 = 위생 ∧ ADV20 ≥ 50억(시총 티어 제한 없음 — 초대형 포함) |",
          "| 보유·리밸런스 | 60거래일 보유, 20일 간격 3트랜치 스태거드 | 월 1회(월말 신호 → 첫 거래일 시가), 상위 20 진입 · 30 안 유지, 재가중 없음 |",
          "| 집중 | top 5 / 10 / 20 / 40 등가중 | top 20 동일 비중(진입 시점 자산의 5%) |",
          "| 체결 | 종가 기준 수익(시가·상한가·하한가 제약 없음) | 다음날 시가, 상한가 시가 진입 불가, 잠긴 하한가 매도 불가, |r| > 31% 날 수익 0 |",
          "| 비용 | 편도 0.18 ~ 0.23%(측정 회전율) | `korea.cost_rt` 왕복(일자별 세율, 2026 41.5bp) 청산일 + 2배 병기 |",
          "| 비교 대상 | 등가중 유니버스·시총가중 대형주 프록시·중대형 동종그룹(Sharpe 1.29~1.49) | 팩터 알파 HAC t · 무작위 20종목 널 · 같은 MDD 지수 대용 + 현금(B1) · 현금 금리 |",
          "| 현금 | 없음 | 유휴 현금에 CD91 근사 |", ""]

    L += ["## 구현 메모 — 모호했던 곳과 정한 것(결과 보기 전 결정)", "",
          "- **주 창:** 설계가 전체(2017-09 ~ 2026-09)를 '표본 내', 2020-01 ~ 2026-09 를 '메타 세대 1·2 와 같은 비교 창' 이라 적었다 — 층 산술의 주 창을 비교 창으로 두고 "
          "전체 창을 같은 산술로 병기했다. 북은 2017-09 부터 한 번 돌리고 창은 그 날별 수익을 자른다(2020-01 첫날에 2019 보유가 이어진다).",
          f"- **첫 달:** H5 는 상장(패널 시작 2016-09-09) 후 252거래일이 필요해 첫 H5 날이 {v['first_h5'].date()} 다 — 2017-08-31 신호 목록이 비어 "
          "**2017-09 한 달은 전부 현금**(금리만)이고 첫 진입은 2017-10-02 시가다. 전체 창 숫자에 그 한 달이 들어 있다(기준선도 같은 날들).",
          "- **신호일·체결:** '매월 첫 거래일' = 체결일. 순위는 전월 마지막 거래일 종가까지의 자료(그날 H5·그날 알던 실적), 체결 = 첫 거래일 시가(메타 세대 1 P2 와 같은 시점).",
          "- **유지 규칙:** 순위 = 그 리밸런스 목록 안 위치(0 부터). 보유 중 종목은 위치 < 30 이면 계속(재가중 없음), 아니면(목록에서 빠짐 — H5 탈락·EPS 없음·Q2 추세 템플릿 "
          "탈락 포함) 다음날 시가 매도. 빈 슬롯은 상위 20 중 보유하지 않은 종목을 순위 순으로 채운다. 상한가 시가로 못 산 슬롯은 그달 빈다.",
          "- **Q2:** 목록 = H5 ∧ 추세 템플릿 ∧ EPS YoY 유한, 순서 = PEAD 순위(H5 안 백분위 — 부분집합 안 순서는 같다). 추세 템플릿을 잃은 보유는 목록 밖이라 다음 달 판다.",
          "- **Q3:** 매월 첫 거래일에 20% Q1(그 자체의 유휴 현금 금리 포함) / 80% 대용 ALL 로 되돌림. 위성 안 비용은 Q1 그대로, 되돌림 거래·대용(ETF) 비용은 0.",
          "- **현금 금리:** CD91 연평균 근사 — 2020~2026 은 메타 세대 1 리뷰어 값(`REVIEW_META_GEN1.md` F3, 스크래치 `review_meta1/indiv.py`; 메타 세대 2 `CD91`), "
          "2017 1.44% · 2018 1.68% · 2019 1.69% 는 이 러너가 덧붙인 기억 값(DB 에 금리 테이블 없음). 연 상수 / 252 매일.",
          "- **층 1 회귀:** 종속 = 현금 금리 포함 날별 순수익(절차의 정의 그대로), 팩터 = 리뷰어 `factors.pkl` 의 5열(`ga2_post.FACTOR_COLS`), 상수항 t(Newey-West 5). "
          "현금 0 판 t 를 참고로 적었다. 널 p95 = 같은 창·같은 현금 처리의 무작위 널 CAGR.",
          "- **층 2 (1):** '이김' = CAGR(월 경로). B1 비중 x 는 창마다 그 절차(현금 포함) MDD 에 맞춘다. 부트스트랩은 주 창·전체 창 각각의 달 전체(부분 창 없음).",
          "- **층 2 (2)·(3):** 비교 대상 '현금 금리' = 같은 구간 CD91 근사의 복리 CAGR. 2025 제외 = 그 창에서 2025 년 날을 뺀 이어 붙인 수익.",
          "- **상한가 시가:** 정리된 조정 시가 / 직전 정리된 조정 종가 ∈ [1.293, 1.305] (조정 가격 비율 — 상한 1.305 는 수집 이음매 날짜의 가짜 상한가를 막는다, 코디네이터 지시). "
          "잠긴 하한가·|r| > 31% 처리는 메타 세대 1 `meta_fast` 그대로.",
          "- **독립 재현(스크래치 `mg3/review/`, `meta_fast` 안 씀):** Q1·Q2 현금 0 날별 수익이 nav.csv 와 같다(최대 차 6e-16, 진입 927·1,309 같음), B1·부트스트랩 6행 같음, "
          "신호일 PEAD 순위 = H5 안 EPS YoY 백분위(109 신호일 불일치 0). 남은 편향: (a) 총 노출 > 100% 인 날이 86%(평균 ~104%)인데 초과분 차입 이자가 없다 — "
          "CD91 을 물리면 −0.11%/년, CD91 + 2% 면 −0.20%/년; (b) 비용을 진입 명목에 매김(청산 가치 기준이면 −0.04%/년); (c) 2019 연간 실적 가용일 2020-03-30 은 "
          "코로나 제출 연장 회사에겐 실제 공시보다 이를 수 있다(작은 룩어헤드 가능).",
          "- **널 Q3:** 각 Q1 널 뽑기를 같은 방식으로 20/80 에 섞은 것(대용 부분은 실제와 같다).",
          "- **실행 용량:** 주문액 = 계좌 K × 종목 비중(성장 반영 안 함), 대상 = 실제 시가 체결(진입 + 리밸런스 매도), 2025-08+ 만 09:00 분봉.",
          "- **팩터:** 리뷰어 A `factors.py` 산출 pickle(이 저장소 밖 스크래치, 정규장 보정 옛 판 — 2026-09-14+ 날 팩터 값은 종가 컷 수정 전 가격) — 재현에는 그 스크립트가 필요하다.", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
