#!/usr/bin/env python
"""수렴 진입 × 수급 × 실적 — 미시험 중소형(ADV20 50억~200억) 표본 밖 시험, 사전등록 러너.

사전등록: ``research/logs/sepa_explore/CONVERGENCE.md`` (결과 보기 전 커밋 ce9a688). 유니버스·상대 수렴 rel·진입·
청산·비용·대조·팔(A~E, R)·합격 바·VOID 를 그대로 옮긴다. 이 러너는 **리포터**다 — 바 숫자를 계산해
``CONVERGENCE_RESULTS.md`` 에 적을 뿐 판정하지 않는다(판정은 사람이, 판정 팔은 A 하나).

- 유니버스 U_s: ``korea.universe_mask`` 위생 그대로 + **5,000 ≤ ADV20 < 20,000 (백만원)** (``adv_cap``).
  RS·breadth 모집단은 기존과 같다(가격 패널 전체 / 시장별 전 종목 — ``sepa.skeleton_features`` 그대로).
- 상대 수렴 rel = (t−10..t−1 고가 최대 / 저가 최소 − 1) ÷ (그 값의 t−260..t−11 250일 이동 중앙값, 최소 120개).
- 진입(A): 뼈대 S ∧ breadth > 0.5 ∧ PEAD 순위 ≥ 0.70 (순위 모집단 = 그날 ADV20 ≥ 5,000 위생 통과 종목) ∧
  rel ≤ 1.0 ∧ 수급 F = 1 (``flow_leg`` 거래대금 가중) — 사건 2024-02-19 ~ 2026-04-01.
- 청산: :func:`research.signals.minervini_exits.simulate_minervini` 기본값(원전 프로토콜).
- 비용: ``korea.cost_rt`` 와 그 2배.
- 대조: 모멘텀 정합(같은 날 × 시장 × 12-1 모멘텀 10분위 × ADV20 2분위, 그 팔의 유니버스 풀, 자기 제외, 5배, seed 0,
  날짜군집 2,000회), 같은 프로토콜(``control_minervini``).
- 원장: ``research/logs/sepa_explore/convergence/TRIALS.jsonl`` (A·B·C·D·R 5줄, 결과 전에 적는다). E 는 A 의
  부분집합 기술이라 원장에 넣지 않는다.

기존 러너(``sepa_confluence_gate``·``sepa_explore``·``sepa_momentum``·``minervini_exits_run``)는 import 만 한다.

Run:  uv run --extra pg python research/experiments/convergence_run.py
디버그(구간 축소 — 탐색 디렉터리·원장을 더럽히지 않는다):
      ... convergence_run.py --event-lo 2025-01-01 --event-hi 2025-03-31 --out <scratch>/cv_debug
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
from swing_it.units import eok

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from krx_quant_core.market.session import KST  # noqa: E402
from minervini_exits_run import (  # noqa: E402
    EXIT_ORDER,
    SIM_COLS,
    SKIP_ORDER,
    _alpha,
    _dist,
    _f,
    cell_config,
    sim_kwargs,
    trade_stats,
    worst_decile_mean,
)
from prop_swing_common import load_env_db  # noqa: E402
from sepa_confluence_gate import (  # noqa: E402
    NEWS_FEED_HI,
    NEWS_FEED_LO,
    PANEL_START,
    Data,
    _costs,
    _key,
    _log,
    _pct,
    attach_features,
    e_pass,
    sub_alpha,
)
from sepa_explore import EVENT_HI, EVENT_LO  # noqa: E402
from sepa_momentum import (  # noqa: E402
    MOM_N_Q,
    STRATA_MOM,
    attach_strata,
    day_market_median_half,
    day_market_quantiles,
    matched_on,
    momentum_12_1,
)

from research.signals.minervini_exits import (  # noqa: E402
    PREREG_MINERVINI,
    contraction_low_level,
    control_minervini,
    simulate_minervini,
)
from research.signals.sepa import build_skeleton_events  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.earnings_leg import pead_rank  # noqa: E402
from swing_it.features.korea import universe_mask  # noqa: E402
from swing_it.news.daily_leg import build_daily, n_flag_panel, n_flag_values  # noqa: E402
from swing_it.storage import connect, db_default, read_prices  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "convergence"
RESULTS_MD = "CONVERGENCE_RESULTS.md"
PREREG_DOC = "research/logs/sepa_explore/CONVERGENCE.md (ce9a688)"
PRIOR_LEDGERS = (("sepa_confluence", "research/logs"), ("sepa_explore", "research/logs"),
                 ("minervini_exits", "research/logs/sepa_explore"))

# --- 사전등록 수치(CONVERGENCE.md — 탐색하지 않는다) ---
ADV_LO, ADV_HI = eok(50), eok(200)          # U_s: 5,000 ≤ ADV20 < 20,000 (백만원 = 50억~200억)
PEAD_POP_FLOOR = 5_000.0                    # PEAD 순위 모집단: 그날 ADV20 ≥ 5,000 위생 통과
FLOW_LO = "2024-02-19"                      # F 를 쓰는 팔의 사건 하한
NEWS_LO = "2025-09-01"                      # 팔 E(뉴스) 사건 하한
REL_WINDOW = 10                             # rel 분자: t−10..t−1 고저폭
REL_MED_WINDOW = 250                        # rel 분모: 그 값의 250일 이동 중앙값
REL_MIN_PERIODS = 120                       # 분모 최소 개수
REL_GAP = 11                                # 분모 창 = t−260..t−11 (끝 = t−11)
REL_MAX = 1.0                               # 주 팔: rel ≤ 1.0
REL_POSTHOC = 1.33                          # 팔 C: 사후 경계
VOID_MIN_N = 40                             # A n < 40 → VOID
GL_MIN = 2.0                                # 바 3: 이익/손실 ≥ 2
STOP_RATE_MAX = 0.5                         # 바 3: 손절 비율 < 50%
WORST_DECILE_MIN = -0.11                    # 바 4: 최악 10% 구간(1배 비용) ≥ −11%
MONTHS_POS_MIN = 0.5                        # 바 5: 월별 양수 비율 > 50%

PRIMARY = "A"
#: 팔 — 순서가 표의 순서. ``universe`` ∈ {"U_s", "main"}, ``rel_max`` None = rel 조건 없음.
ARMS: dict[str, dict] = {
    "A": {"universe": "U_s", "rel_max": REL_MAX, "flow": True, "event_lo": FLOW_LO,
          "desc": "주: S ∧ breadth ∧ PEAD≥0.70 ∧ rel≤1.0 ∧ F=1 (U_s)"},
    "B": {"universe": "U_s", "rel_max": REL_MAX, "flow": False, "event_lo": EVENT_LO,
          "desc": "A − 수급 F (수렴만, U_s)"},
    "C": {"universe": "U_s", "rel_max": REL_POSTHOC, "flow": True, "event_lo": FLOW_LO,
          "desc": "A 에서 rel ≤ 1.33 (사후 경계, U_s)"},
    "D": {"universe": "U_s", "rel_max": None, "flow": True, "event_lo": FLOW_LO,
          "desc": "A 에서 rel 조건 없음 (수급만, U_s)"},
    "R": {"universe": "main", "rel_max": REL_MAX, "flow": True, "event_lo": FLOW_LO,
          "desc": "A 정의를 ADV20 ≥ 200억(이미 본 구간)에 — 참고"},
}
UNIVERSE_DESC = {
    "U_s": "korea.universe_mask(raw_close=daily_bars.close, adv_floor=5000, adv_cap=20000) — 보통주·KOSPI/KOSDAQ·"
           "5,000 ≤ ADV20 < 20,000(백만원)·원종가≥1000·상장252일",
    "main": "korea.universe_mask(raw_close=daily_bars.close) — 보통주·KOSPI/KOSDAQ·ADV20≥200억·원종가≥1000·상장252일",
}


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_convergence_run_helpers.py)
# ---------------------------------------------------------------------------

def relative_contraction(high: pd.DataFrame, low: pd.DataFrame, *, window: int = REL_WINDOW,
                         med_window: int = REL_MED_WINDOW, min_periods: int = REL_MIN_PERIODS,
                         gap: int = REL_GAP) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """상대 수렴 → ``(고저폭 rng, rel)`` (날짜×종목, ``t`` 행 = ``t−1`` 종가까지 — 돌파일 자신은 안 들어간다).

    ``rng[t] = max(high[t−w..t−1]) / min(low[t−w..t−1]) − 1`` (창이 꽉 차야 값, 저가 ≤ 0 이면 NaN).
    ``rel[t] = rng[t] / median(rng[s], s ∈ t−gap−med_window+1 .. t−gap)`` — 기본 s ∈ t−260..t−11(250개), 유효
    ``min_periods`` 개 이상일 때만. 사후 분해(``convergence_posthoc.py``)는 끝을 t−10 으로 잡았지만 사전등록 문구
    "t−260..t−11" 을 따른다. 분모가 0 이하이면 NaN.
    """
    hi = high.shift(1).rolling(window, min_periods=window).max()
    lo = low.shift(1).rolling(window, min_periods=window).min()
    rng = hi / lo.where(lo > 0) - 1.0
    med = rng.rolling(med_window, min_periods=min_periods).median().shift(gap)
    rel = rng / med.where(med > 0)
    return rng, rel


def arm_config(name: str, lo: "str | None" = None, hi: "str | None" = None) -> dict:
    """한 팔의 전체 config(원장 지문용). 미너비니 원전 청산 칸 config(``cell_config`` — VCP 끔) 위에 이 시험의
    유니버스·rel·수급·PEAD 모집단·대조·비용을 덮는다. ``lo``/``hi`` 는 디버그 구간 축소(팔 하한보다 늦을 때만)."""
    arm = ARMS[name]
    a_lo = arm["event_lo"] if lo is None else max(arm["event_lo"], lo)
    c = cell_config(name, {"vcp": False}, a_lo, hi or EVENT_HI)
    c.pop("fixed", None)
    c.update({
        "prereg": PREREG_DOC, "cell": name, "arm": arm["desc"], "universe": UNIVERSE_DESC[arm["universe"]],
        "pead_population": f"그날 위생 통과 ∧ ADV20 ≥ {PEAD_POP_FLOOR:,.0f} (백만원)",
        "rel": f"rng=(max high[t−{REL_WINDOW}..t−1]/min low[t−{REL_WINDOW}..t−1]−1) ÷ median(rng[t−260..t−11], "
               f"≥{REL_MIN_PERIODS}개)",
        "rel_max": arm["rel_max"], "flow_F": "F=1 (flow_leg, 거래대금 가중)" if arm["flow"] else None,
        "cost": "korea.cost_rt(청산일, 시장) 1배 · 2배(주 비용)",
        "matched": f"모멘텀 정합: 같은 날 × 시장 × 12-1 모멘텀 10분위 × ADV20 2분위, 풀 = {arm['universe']} 유니버스, "
                   "자기 제외, n_per 5, seed 0, 같은 프로토콜(control_minervini), 날짜군집 부트스트랩 2000",
    })
    return c


def arm_mask(ev: pd.DataFrame, arm: dict, lo: str, hi: str) -> np.ndarray:
    """사건 표(``event_day``·``rel``·``F``·``pead_ok`` 열)에서 그 팔의 진입 조건. NaN rel·F 는 통과하지 않는다."""
    day = pd.to_datetime(ev["event_day"])
    mk = ev["pead_ok"].to_numpy(bool) & (day >= pd.Timestamp(lo)).to_numpy() & (day <= pd.Timestamp(hi)).to_numpy()
    if arm["rel_max"] is not None:
        mk &= (ev["rel"] <= arm["rel_max"]).fillna(False).to_numpy(bool)
    if arm["flow"]:
        mk &= (ev["F"] == 1).fillna(False).to_numpy(bool)
    return mk


def stop_rate(exit_reason: pd.Series) -> float:
    """손절 비율 = 마지막 청산 사유가 ``stop`` 인 비율(본전 ``breakeven`` 은 손절이 아니다)."""
    return float((exit_reason == "stop").mean()) if len(exit_reason) else np.nan


def convergence_bars(s: dict) -> dict:
    """사전등록 합격 바 1~5 + VOID 의 산술 조건(숫자 비교만 — 판정은 사람이).

    1. 모멘텀 α CI 하단 > 0. 2. 2배 비용 평균 > 0 ∧ 상위 1% 제거(1배) > 0. 3. 이익/손실 ≥ 2 ∧ 손절 비율 < 50%.
    4. 하위 10% 평균(1배) ≥ −11%. 5. 월별(진입월) 비용 차감 평균 양수 비율 > 50% — 1배(``months_pos``)로 셈하고
    2배(``months2_pos``)는 따로 ``b5_2x`` 로 적는다. VOID = n < 40.
    """
    def fin(v):
        return v is not None and np.isfinite(v)

    def gt0(v):
        return bool(fin(v) and v > 0)

    def share(k, n):
        return k / n if n else np.nan

    m1 = share(s.get("months_pos", 0), s.get("months_n", 0))
    m2 = share(s.get("months2_pos", 0), s.get("months_n", 0))
    out = {
        "void": s.get("n", 0) < VOID_MIN_N,
        "b1": gt0(s.get("mom_lo", np.nan)),
        "b2": gt0(s.get("net2_mean", np.nan)) and gt0(s.get("drop_top1", np.nan)),
        "b3": bool(fin(s.get("gl_ratio")) and s["gl_ratio"] >= GL_MIN
                   and fin(s.get("stop_rate")) and s["stop_rate"] < STOP_RATE_MAX),
        "b4": bool(fin(s.get("worst_decile")) and s["worst_decile"] >= WORST_DECILE_MIN),
        "b5": bool(fin(m1) and m1 > MONTHS_POS_MIN),
        "b5_2x": bool(fin(m2) and m2 > MONTHS_POS_MIN),
    }
    out["n_met"] = sum(out[f"b{i}"] for i in range(1, 6))
    return out


def per_year_counts(days: pd.Series) -> dict:
    """사건 연도별 건수."""
    return {int(k): int(v) for k, v in pd.to_datetime(days).dt.year.value_counts().sort_index().items()}


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def raw_close_panel(con, p) -> pd.DataFrame:
    """1,000원 규칙용 원자료 종가(``Data`` 와 같은 방식 — ``daily_bars``)를 가격 패널 모양으로."""
    raw = read_prices(con, session="raw", cols=("code", "date", "close"), table="daily_bars")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["date"] >= pd.Timestamp(PANEL_START)]
    out = raw.pivot(index="date", columns="code", values="close").astype(float)
    return out.reindex(index=p.days, columns=p.close.columns)


def repoint_pead(D: Data, eligible_panel: pd.DataFrame) -> None:
    """``D.earn.pead_rank`` 를 ``eligible_panel``(날짜×종목 bool) 모집단 순위로 바꾼다(원래 값은 ``pead_rank_main``)."""
    ep, p = D.earn, D.p
    di = p.days.get_indexer(pd.to_datetime(ep["date"]))
    ci = p.close.columns.get_indexer(ep["code"])
    ok = (di >= 0) & (ci >= 0)
    un = eligible_panel.to_numpy()
    elig = np.zeros(len(ep), dtype=bool)
    elig[ok] = un[di[ok], ci[ok]]
    ep["pead_rank_main"] = ep["pead_rank"]
    ep["pead_rank"] = pead_rank(ep, eligible=pd.Series(elig, index=ep.index))


def build_pool(D: Data, uni: pd.DataFrame, strata: dict, trail: pd.DataFrame, c_low: pd.DataFrame, cfg: dict,
               days) -> pd.DataFrame:
    """그 날들의 ``uni`` 전 종목을 같은 프로토콜(``control_minervini`` — 자기 수축 저점, 겹침 허용)로 시뮬한 대조 풀."""
    p = D.p
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(days))))
    di_days = p.days.get_indexer(days)
    r, c = np.nonzero(uni.to_numpy(dtype=bool)[di_days])
    di = di_days[r]
    pairs = pd.DataFrame({"code": p.close.columns[c], "event_day": p.days[di],
                          "pivot": D.feats["pivot"].to_numpy()[di, c]})
    for k, panel in strata.items():
        pairs[k] = panel.to_numpy(dtype=float)[di, c]
    ct, sk = control_minervini(p.open, p.high, p.low, p.close, trail, pairs, volume=p.volume, market=p.market,
                               contraction_low=c_low, return_skipped=True, **sim_kwargs(cfg))
    pool = ct[["code", "event_day", "ret", "market", *strata, "exit_reason"]].rename(columns={"event_day": "date"})
    pool["date"] = pd.to_datetime(pool["date"])
    pool.index = _key(pool["date"], pool["code"])
    pool.attrs["n_pairs"] = len(pairs)
    pool.attrs["skips"] = sk["skip_reason"].value_counts().to_dict()
    return pool


def arm_stats(t: pd.DataFrame, n_dsr: int) -> dict:
    """``trade_stats``(1배 비용 순수익 기준) + 손절 비율·2배 비용 월 양수·10% 분위·청산 사유·연도별 건수."""
    s = trade_stats(t, n_dsr)
    if t.empty:
        return s
    net2 = t["ret"] - 2 * t["cost"]
    mo2 = net2.groupby(t["entry_day"].dt.to_period("M")).mean()
    s.update({"stop_rate": stop_rate(t["exit_reason"]), "months2_pos": int((mo2 > 0).sum()),
              "p10": float(np.percentile(t["net"], 10)), "r_gross": float(t["R"].mean()),
              "worst_decile2": worst_decile_mean(net2.to_numpy(float)),
              "exits": t["exit_reason"].value_counts().to_dict(),
              "exit_net": t.groupby("exit_reason")["net"].mean().to_dict(),
              "years": per_year_counts(t["event_day"])})
    return s


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(--out 을 탐색 디렉터리 밖으로)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR, help="결과·원장 디렉터리(기본 = 탐색 디렉터리)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.event_lo or args.event_hi) and not debug:
        ap.error("구간 축소는 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    os.makedirs(out_dir, exist_ok=True)
    hi = args.event_hi or EVENT_HI
    cfgs = {name: arm_config(name, args.event_lo, hi) for name in ARMS}
    lo_of = {name: c["event_lo"] for name, c in cfgs.items()}

    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()

    # --- 원장: 5팔(A·B·C·D·R)을 결과 전에 적는다. E 는 A 의 부분집합 기술 — 원장 밖 ---
    for name, c in cfgs.items():
        record_trial(LABEL, c, logs_dir=out_dir, note=f"수렴×수급×실적 팔 {name}: {ARMS[name]['desc']}")
    n_led = count_trials(LABEL, logs_dir=out_dir)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
    n_dsr = n_led + sum(n_prior.values())
    _log(f"원장 {LABEL}: {n_led}팔 · " + " · ".join(f"{k} {v}" for k, v in n_prior.items()) + f" → DSR N {n_dsr}", t0)

    load_env_db()
    con = connect(db_default())
    D = Data(con, t0)
    p = D.p

    # --- 유니버스: U_s(구간)·H5(PEAD 모집단, ADV ≥ 5,000)·main(= D.uni) ---
    raw_close = raw_close_panel(con, p)
    h5 = universe_mask(p.close, p.trade_value, D.meta, raw_close=raw_close, adv_floor=PEAD_POP_FLOOR)
    u_s = universe_mask(p.close, p.trade_value, D.meta, raw_close=raw_close, adv_floor=ADV_LO, adv_cap=ADV_HI)
    del raw_close
    main_u = D.uni.astype(bool)
    chk = (u_s | main_u) == h5
    assert bool(chk.to_numpy().all()) and not bool((u_s & main_u).to_numpy().any()), "U_s ∪ main ≠ H5"
    unis = {"U_s": u_s, "main": main_u}
    _log(f"U_s 종목-일 {int(u_s.to_numpy().sum()):,} · main {int(main_u.to_numpy().sum()):,} · "
         f"H5 {int(h5.to_numpy().sum()):,}", t0)
    repoint_pead(D, h5)
    _log("PEAD 순위 모집단 → H5", t0)

    # --- 패널: rel·수축 저점·추적 MA·모멘텀 층(유니버스별) ---
    rng, rel = relative_contraction(p.high, p.low)
    c_low = contraction_low_level(p.low, PREREG_MINERVINI["vcp_window"])
    trail = D.feats["ma50"]
    mom = momentum_12_1(p.close)
    strata = {u: {"q_mom": day_market_quantiles(mom, m, p.market, n_q=MOM_N_Q),
                  "adv2": day_market_median_half(D.adv20, m, p.market)} for u, m in unis.items()}
    del mom
    _log("rel·수축 저점·모멘텀 층", t0)

    # --- 사건 master(유니버스별): 뼈대 S ∧ breadth > 0.5 + 피처 + rel + PEAD 문 ---
    base_lo = min(lo_of.values())
    masters, funnel = {}, {}
    for u, m in unis.items():
        cfg_u = dict(cfgs[PRIMARY], event_lo=base_lo, event_hi=hi)
        ms = build_skeleton_events(p, cfg_u, universe=m, feats=D.feats)
        ms = attach_features(D, ms)
        di, ci = ms["day_idx"].to_numpy(), ms["col_idx"].to_numpy()
        ms["rng10"] = rng.to_numpy()[di, ci]
        ms["rel"] = rel.to_numpy()[di, ci]
        ms["pead_ok"] = e_pass(ms, cfgs[PRIMARY]).to_numpy(bool)
        ms = ms.sort_values(["event_day", "code"]).reset_index(drop=True)
        masters[u] = ms
        post = ms["event_day"] >= pd.Timestamp(max(FLOW_LO, base_lo))
        pk = post & ms["pead_ok"]
        relk = pk & (ms["rel"] <= REL_MAX).fillna(False)
        funnel[u] = {"S_all": len(ms), "S_post": int(post.sum()), "pead_post": int(pk.sum()),
                     "rel_known_post": int((pk & ms["rel"].notna()).sum()), "rel1_post": int(relk.sum()),
                     "F_known": int((relk & ms["F"].notna()).sum()), "F1": int((relk & (ms["F"] == 1)).sum()),
                     "pead_all": int(ms["pead_ok"].sum()),
                     "rel1_all": int((ms["pead_ok"] & (ms["rel"] <= REL_MAX).fillna(False)).sum())}
        _log(f"master {u}: {funnel[u]}", t0)
    del rng, rel

    # --- 1차: 팔별 트레이드 ---
    trades, skips, n_ev = {}, {}, {}
    for name, arm in ARMS.items():
        ms = masters[arm["universe"]]
        ev = ms[arm_mask(ms, arm, lo_of[name], hi)].reset_index(drop=True)
        n_ev[name] = len(ev)
        t, sk = simulate_minervini(p.open, p.high, p.low, p.close, trail, ev[SIM_COLS + ["rel", "F", "rng10"]],
                                   volume=p.volume, contraction_low=c_low, return_skipped=True,
                                   **sim_kwargs(cfgs[name]))
        if not t.empty:
            for col in ("event_day", "entry_day", "exit_day"):
                t[col] = pd.to_datetime(t[col])
            t["cost"] = _costs(t)
            t["net"] = t["ret"] - t["cost"]
            t.index = _key(t["event_day"], t["code"])
            t = attach_strata(t, "event_day", p.days, p.close.columns, strata[arm["universe"]])
        trades[name] = t
        skips[name] = sk["skip_reason"].value_counts().to_dict() if len(sk) else {}
        _log(f"팔 {name}: 사건 {len(ev):,} · 트레이드 {len(t):,}", t0)

    # --- 2차: 대조 풀(유니버스별, 그 유니버스 팔들의 사건일 합집합 — 프로토콜은 전 팔 같음) ---
    pools = {}
    for u in unis:
        names = [n for n, a in ARMS.items() if a["universe"] == u and not trades[n].empty]
        if not names:
            continue
        days = pd.concat([trades[n]["event_day"] for n in names]).unique()
        pools[u] = build_pool(D, unis[u], strata[u], trail, c_low, cfgs[names[0]], days)
        _log(f"대조 풀 {u} {names}: 쌍 {pools[u].attrs['n_pairs']:,} → 시뮬 {len(pools[u]):,}", t0)

    # --- 3차: 팔 숫자 ---
    rows, moms = {}, {}
    for name, arm in ARMS.items():
        t = trades[name]
        s = {"arm": name, "n_events": n_ev[name], "skips": skips[name], **arm_stats(t, n_dsr)}
        if not t.empty:
            ms_, m_mom = _mom(t, pools[arm["universe"]])
            s.update(ms_)
            moms[name] = m_mom
            pool = pools[arm["universe"]]
            s["pool_n"] = len(pool)
            s["pool_stop"] = float((pool["exit_reason"] == "stop").mean())
        s.update({f"bar_{k}": v for k, v in convergence_bars(s).items()})
        rows[name] = s
        _log(f"팔 {name}: n {s['n']} · 순평균 {_pct(s.get('net_mean', np.nan))} · 모멘텀 α {_astr(s, 'mom_')}", t0)

    # --- 팔 E: A ∧ 사건 ≥ 2025-09-01, 뉴스 N=1 / N=0 (기술) ---
    e_rows = {}
    ta = trades[PRIMARY]
    nw = ta[ta["event_day"] >= pd.Timestamp(NEWS_LO)].copy() if not ta.empty else ta
    if len(nw):
        flo = dt.datetime.combine(dt.date.fromisoformat(NEWS_FEED_LO), dt.time(0), tzinfo=KST)
        fhi = dt.datetime.combine(dt.date.fromisoformat(NEWS_FEED_HI), dt.time(0), tzinfo=KST)
        daily = build_daily(con, lo=flo, hi=fhi, mode="close", trading_days=[d.date() for d in p.days])
        npan = n_flag_panel(daily, sorted({d.date() for d in nw["event_day"]}), codes=set(nw["code"]))
        npan["N"] = n_flag_values(npan)
        npan["event_day"] = pd.to_datetime(npan["date"])
        nw = nw.merge(npan[["code", "event_day", "N"]], on=["code", "event_day"], how="left")
        nw.index = _key(nw["event_day"], nw["code"])
        for lab, mk in (("E: N=1", nw["N"] == 1), ("E: N=0", nw["N"] == 0), ("E: N 모름", nw["N"].isna())):
            sub = nw[mk.to_numpy()]
            if sub.empty:
                e_rows[lab] = {"n": 0}
                continue
            s = arm_stats(sub, n_dsr)
            s.update(_alpha("mom_", sub_alpha(moms[PRIMARY], pd.Series(ta.index.isin(sub.index))) if PRIMARY in moms else None))
            e_rows[lab] = s
        e_rows["_all"] = {"n": len(nw)}
        _log(f"팔 E: A 의 {NEWS_LO}+ {len(nw)} · " + " · ".join(f"{k} {v['n']}" for k, v in e_rows.items()), t0)

    elapsed = time.time() - t0
    L = render(rows, e_rows, cfgs, lo_of, hi, funnel, pools, n_led, n_prior, n_dsr, out_dir, debug, elapsed, p)
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows.values()])
    flat.to_csv(os.path.join(out_dir, "convergence_arms.csv"), index=False)
    allt = pd.concat([t.assign(arm=n) for n, t in trades.items() if not t.empty] or [pd.DataFrame()],
                     ignore_index=True)
    allt.to_csv(os.path.join(out_dir, "convergence_trades.csv"), index=False)

    print(f"\n=== 수렴 × 수급 × 실적 (U_s 50억~200억) — {elapsed / 60:.1f}분 · DSR N {n_dsr} ===")
    for name, s in rows.items():
        print(f"팔 {name} n {s['n']:4d} · 순평균 {_pct(s.get('net_mean', np.nan))} · 2배 {_pct(s.get('net2_mean', np.nan))}"
              f" · 이익/손실 {_f(s.get('gl_ratio', np.nan))} · 손절 {_f(s.get('stop_rate', np.nan), '{:.1%}')}"
              f" · R {_f(s.get('r_net', np.nan), '{:+.3f}')} · 모멘텀 α {_astr(s, 'mom_')}"
              f" · 바 {s['bar_n_met']}/5{' · VOID(n<40)' if s['bar_void'] else ''}")
    for lab, s in e_rows.items():
        if lab != "_all":
            print(f"{lab} n {s['n']} · 순평균 {_pct(s.get('net_mean', np.nan))}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def _mom(t: pd.DataFrame, pool: pd.DataFrame) -> "tuple[dict, dict]":
    m = matched_on(t, pool, STRATA_MOM)
    return _alpha("mom_", m["ma"]), m


def _astr(s: dict, pre: str) -> str:
    a = s.get(f"{pre}alpha", np.nan)
    if a is None or not np.isfinite(a):
        return "nan"
    return f"{_pct(a)} [{_pct(s.get(f'{pre}lo'))}, {_pct(s.get(f'{pre}hi'))}]"


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _share(k, n) -> str:
    return f"{k}/{n} ({k / n:.0%})" if n else "—"


def render(rows, e_rows, cfgs, lo_of, hi, funnel, pools, n_led, n_prior, n_dsr, out_dir, debug, elapsed, p):
    L = [f"# 수렴 × 수급 × 실적 — 중소형(ADV20 50억~200억) 표본 밖 시험 — 실행 결과 {date.today().isoformat()} "
         "(러너 자동 생성, 판정 아님)", "",
         "사전등록 `CONVERGENCE.md`(ce9a688, 결과 보기 전) 의 유니버스·rel·진입·청산·비용·대조·팔·합격 바를 그대로 계산한 "
         "숫자다. 합격 여부는 여기 적지 않는다 — 판정은 사람이 쓴다(판정 팔 = A). 표의 ✓/✗ 는 사전등록 문턱과의 "
         "**산술 비교**일 뿐이다. 재현: `uv run --extra pg python research/experiments/convergence_run.py`.", ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 구간 축소(상한 {hi}), 출력·원장 `{out_dir}`. 측정으로 쓰지 않는다.", ""]

    a = rows[PRIMARY]
    L += ["## 판정 팔 A — 합격 바 숫자 (사전등록 1~5 + VOID — 숫자만, 판정 아님)", "",
          "| 바 | 사전등록 문턱 | A 숫자 | 산술 |", "|---|---|---|---|",
          f"| VOID | n < {VOID_MIN_N} → VOID | n = {a['n']} | {'**VOID**' if a['bar_void'] else '표본 충족'} |"]
    if a["n"]:
        L += [f"| 1 | 모멘텀 정합 α 95% CI 하단 > 0 | {_astr(a, 'mom_')} (정합 {a.get('mom_n', 0)}건, 미정합 "
              f"{a.get('mom_unmatched', 0)}) | {_mark(a['bar_b1'])} |",
              f"| 2 | 2배 비용 평균 > 0 ∧ 상위 1% 제거(1배) > 0 | {_pct(a['net2_mean'])} · {_pct(a['drop_top1'])} | "
              f"{_mark(a['bar_b2'])} |",
              f"| 3 | 이익/손실 ≥ 2 ∧ 손절 비율 < 50% | {_f(a['gl_ratio'])} · {a['stop_rate']:.1%} | {_mark(a['bar_b3'])} |",
              f"| 4 | 최악 10% 구간(1배) ≥ −11% | 하위 10% 평균 {_pct(a['worst_decile'])} (참고: 10% 분위 "
              f"{_pct(a['p10'])}) | {_mark(a['bar_b4'])} |",
              f"| 5 | 월별(진입월) 비용 차감 평균 양수 비율 > 50% | 1배 {_share(a['months_pos'], a['months_n'])} · 2배 "
              f"{_share(a['months2_pos'], a['months_n'])} | 1배 {_mark(a['bar_b5'])} · 2배 {_mark(a['bar_b5_2x'])} |",
              f"| 산술 충족 | 5개 전부 | **{a['bar_n_met']}/5** (바 5 는 1배 기준) | |"]
    L.append("")

    L += ["## 표본·깔때기", "",
          f"- 가격 패널 {p.days[0].date()}~{p.days[-1].date()} · 실행 {elapsed / 60:.1f}분 · 팔 사건 구간: "
          + " · ".join(f"{n} {lo_of[n]}~{hi}" for n in ARMS),
          f"- 원장 `{out_dir}/{LABEL}/TRIALS.jsonl` 팔 수 **{n_led}** · DSR N(기술) = {n_led} + "
          + " + ".join(f"{k} {v}" for k, v in n_prior.items()) + f" = **{n_dsr}**", "",
          "| 유니버스 | S∧breadth 전체 | ∧PEAD≥0.70 전체 | ∧rel≤1.0 전체 | S∧breadth 2024-02-19+ | ∧PEAD≥0.70 | rel 알려짐 | "
          "∧rel≤1.0 | F 알려짐 | ∧F=1 (= A·R 사건) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for u, f in funnel.items():
        L.append(f"| {u} | {f['S_all']:,} | {f['pead_all']:,} | {f['rel1_all']:,} | {f['S_post']:,} | {f['pead_post']:,} | "
                 f"{f['rel_known_post']:,} | {f['rel1_post']:,} | {f['F_known']:,} | {f['F1']:,} |")
    L += ["", "대조 풀: " + " · ".join(f"{u} 쌍 {pl.attrs['n_pairs']:,} → 시뮬 {len(pl):,} (버림 "
                                      f"{_dist(pl.attrs['skips'], SKIP_ORDER)})" for u, pl in pools.items()), ""]

    L += ["## 전 팔 — 숫자 (판정은 A 만, 나머지는 참고)", "",
          "| 팔 | 정의 | n | 승률 | 손절 비율 | 평균 이익 | 평균 손실 | 이익/손실 | 기대값 R(순) | 순평균 1배 | 순평균 2배 | "
          "상위1% 제거(1배) | 하위10% 평균(1배) | 최장 연패 | 모멘텀 α [CI] | 월 양수(1배) | 월 양수(2배) | 바 산술 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows.items():
        if not s["n"]:
            L.append(f"| {name} | {ARMS[name]['desc']} | 0 | | | | | | | | | | | | | | | VOID |")
            continue
        L.append(f"| {name} | {ARMS[name]['desc']} | {s['n']} | {s['win']:.1%} | {s['stop_rate']:.1%} | "
                 f"{_pct(s['avg_gain'])} | {_pct(s['avg_loss'])} | {_f(s['gl_ratio'])} | {_f(s['r_net'], '{:+.3f}')} | "
                 f"{_pct(s['net_mean'])} | {_pct(s['net2_mean'])} | {_pct(s['drop_top1'])} | {_pct(s['worst_decile'])} | "
                 f"{s['loss_streak']} | {_astr(s, 'mom_')} | {_share(s['months_pos'], s['months_n'])} | "
                 f"{_share(s['months2_pos'], s['months_n'])} | {s['bar_n_met']}/5{' · n<40' if s['bar_void'] else ''} |")
    L.append("")
    L += ["| 팔 | 사건 | 버림(사유) | 청산 사유 (건) | 청산 사유별 순평균 | 순중앙값 | 10% 분위 | 보유 중앙값 | d 중앙값 | "
          "절반 익절 | 총 R 평균 | 사건 연도별 건수 | 대조 풀 손절 비율 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows.items():
        if not s["n"]:
            L.append(f"| {name} | {s['n_events']} | {_dist(s['skips'], SKIP_ORDER)} | — | | | | | | | | | |")
            continue
        en = s["exit_net"]
        L.append(f"| {name} | {s['n_events']} | {_dist(s['skips'], SKIP_ORDER)} | {_dist(s['exits'], EXIT_ORDER)} | "
                 + " · ".join(f"{k} {_pct(en[k])}" for k in EXIT_ORDER if k in en)
                 + f" | {_pct(s['net_median'])} | {_pct(s['p10'])} | {s['hold_median']:.0f} | {_pct(s['d_median'])} | "
                 f"{s['partial_rate']:.1%} | {_f(s['r_gross'], '{:+.3f}')} | "
                 + " · ".join(f"{y} {c}" for y, c in s["years"].items())
                 + f" | {s.get('pool_stop', np.nan):.1%} |")
    L.append("")

    L += [f"## 팔 E — A ∧ 사건 ≥ {NEWS_LO}, 뉴스 N (기술만 · 원장 밖)", ""]
    if not e_rows:
        L += ["A 에 이 구간 트레이드가 없다 — 생략(사전등록: 표본 없으면 생략).", ""]
    else:
        L += [f"A 트레이드 중 사건 ≥ {NEWS_LO}: {e_rows['_all']['n']}건. N = `daily_leg` close 모드 60일 플래그"
              "(모름 = NaN). 모멘텀 α 는 A 의 정합 짝을 그대로 부분집합에서(`sub_alpha`).", "",
              "| 군 | n | 승률 | 손절 비율 | 이익/손실 | 기대값 R(순) | 순평균 1배 | 순평균 2배 | 하위10% 평균 | 모멘텀 α [CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for lab, s in e_rows.items():
            if lab == "_all":
                continue
            if not s["n"]:
                L.append(f"| {lab} | 0 | | | | | | | | |")
                continue
            L.append(f"| {lab} | {s['n']} | {s['win']:.1%} | {s['stop_rate']:.1%} | {_f(s['gl_ratio'])} | "
                     f"{_f(s['r_net'], '{:+.3f}')} | {_pct(s['net_mean'])} | {_pct(s['net2_mean'])} | "
                     f"{_pct(s['worst_decile'])} | {_astr(s, 'mom_')} |")
        L.append("")

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          "- **U_s:** `korea.universe_mask` 에 선택 인자 `adv_cap`(ADV20 < 상한, 기본 None = 기존 동작)을 더해 "
          "`adv_floor=5000, adv_cap=20000` 으로 만든다. 원종가 1,000원 규칙은 `Data` 와 같은 `daily_bars.close`. "
          "실행 중 U_s ∪ main = (ADV ≥ 5,000 위생) ∧ U_s ∩ main = ∅ 를 assert 로 확인한다.",
          "- **RS·breadth:** `sepa.skeleton_features` 그대로(가격 패널 전체·시장별 전 종목) — 유니버스와 무관.",
          "- **rel:** 사전등록 문구대로 분모 창 끝을 t−11 로(`shift(11)` — s ∈ t−260..t−11, 250개, 유효 ≥ 120). 사후 분해 "
          "스크립트는 `shift(10)`(t−259..t−10)이었다 — 하루 차이, 사전등록을 따른다. 고저는 분할조정 패널. rel 이 NaN"
          "(이력 부족)이면 rel 조건을 통과하지 않는다(팔 D 는 rel 을 안 보므로 들어간다).",
          "- **PEAD 순위 모집단:** 그날 위생 통과 ∧ ADV20 ≥ 5,000 (U_s ∪ main). **팔 R 도 같은 모집단**이다(사전등록 'A 와 "
          "같은 정의') — 그래서 R 의 PEAD 문은 이전 시험들(main 안 순위)과 다르다. 문턱 0.70, 실적 행 없음 → 통과 안 함"
          "(`e_pass`).",
          "- **수급 F:** `Data.flow['F']`(flow_leg, 거래대금 가중), 사건일 값. F = NaN(수급 행 없음)이면 통과 안 함.",
          "- **청산·체결:** `simulate_minervini` 사전등록 기본값(VCP 문 없음 — 이 시험은 rel 로 수렴을 본다). 같은 종목 "
          "보유 중 재진입 없음은 팔마다 따로 걸린다(팔마다 사건 집합이 달라 트레이드가 부분집합 관계가 아닐 수 있다).",
          "- **비용:** 1배 = `korea.cost_rt(청산일, 시장)`, 2배 = 총수익 − 2×그 값. 기대값 R(순) = mean(1배 순수익 / d), "
          "총 R = 시뮬레이터 `R`(비용 전).",
          "- **바 3 손절 비율:** 마지막 청산 사유 = `stop` 의 비율(본전 `breakeven`·`time_stop` 은 손절 아님 — 사후 분해와 같다).",
          "- **바 4 '최악 10% 구간':** 하위 10% 트레이드의 **평균**(`worst_decile_mean`, 최소 1건 — 미너비니 결과의 "
          "'하위10% 평균' 과 같은 정의). 10% 분위수는 참고로 함께 적는다.",
          "- **바 5:** 사전등록이 비용 배수를 적지 않았다 — 1배(`cost_rt`)로 셈하고 2배도 나란히 적는다. 산술 충족 수는 "
          "1배 기준. 진입월 = `entry_day` 의 달, 양수 = 그 달 트레이드 순평균 > 0, 비율 > 50%(같으면 미달).",
          "- **대조:** 풀 = 그 유니버스 팔들의 트레이드 사건일 합집합의 그날 유니버스 전 종목을 같은 프로토콜로 "
          "`control_minervini`(자기 수축 저점, 피벗 = 직전 50일 최고 고가, 겹침 허용). 층 = `STRATA_MOM`(날짜·시장·q_mom·"
          "adv2) — q_mom(10분위)·adv2(중앙값 2분위)는 **그 팔의 유니버스 안**(U_s 또는 main)에서 날짜×시장별로 자른다. "
          "alpha 는 비용 전 총수익(모멘텀 재측정·미너비니 청산과 같다). 정합 층이 비면 그 트레이드는 미정합으로 빠진다.",
          "- **팔 E:** A 트레이드 중 사건 ≥ 2025-09-01 을 N 으로 나눈 부분집합(따로 시뮬하지 않는다), 원장 밖.",
          "- **DSR:** 사전등록 바에 없다 — 기술용 N 만 적고 바에 쓰지 않는다.", ""]

    L += ["## 팔 config", "", "| 팔 | 유니버스 | rel 상한 | 수급 F | 사건 하한 |", "|---|---|---|---|---|"]
    for n, a_ in ARMS.items():
        L.append(f"| {n} | {a_['universe']} | {a_['rel_max'] if a_['rel_max'] is not None else '—'} | "
                 f"{'F=1' if a_['flow'] else '—'} | {lo_of[n]} |")
    L.append(f"| E | U_s (A 부분집합) | 1.0 | F=1 | {NEWS_LO} |")
    return L


if __name__ == "__main__":
    raise SystemExit(main())
