#!/usr/bin/env python
"""VCP 재시험 — 수렴 피벗 매수정지 × 분기 단독 실적 가속 × 기반 수급, 사전등록 러너. **리포터 — 판정 아님.**

사전등록: ``research/logs/sepa_explore/VCP_RETEST.md`` — 5632879 + 정정 7dbbd0c + 정정 2 aae4b8e(``REVIEW_VCP_RETEST.md`` 1·2단계),
모두 결과 보기 전. 이 러너는 숫자를 ``VCP_RETEST_RESULTS.md`` 에 적고 바 문턱과의 **산술 비교**만 한다.

- **판정 팔 = V0**(진입 역학, 정정 C1). V1·V2·V3 = 실적·수급 층 기술 전용. X1·X2(·X2L 문구판) = V0 셋업 위 진입 방식 분리.
  바 1~5 판정(바 2·3 = 2배 비용, 바 5 = 1배 판정·2배 기술), 바 6(포트폴리오 팩터) 기술.
- **유니버스 하한(정정 2):** ``--funnel-only`` 가 ADV20 하한 50억·30억·20억·10억마다 체결 **건수만** 세어 ``vcp_retest/FUNNEL.md`` 에
  적고, V0 체결 ≥ 60 ∧ 2017-09~2024-12 ≥ 30 인 가장 높은 하한을 판정 유니버스로 고정한다(넷 다 미달이면 V0 VOID). 전체 실행은
  ``--adv-floor`` 를 주지 않으면 그 값을 ``FUNNEL.md`` 에서 읽는다.
- 가격: ``Data``(``sepa_confluence_gate``) 를 ``relax_run.KrxPatch`` 아래서 읽는다. 2026-09-14+ 미보정(NXT) 행이 있는 종목의 보유 경로는
  2026-09-11 에서 ``data_end`` 로 끊는다(정정 2).
- 셋업·진입·청산: :mod:`research.signals.vcp`. 진입일 종가 ≤ 손절가 = 확정 손절, 모호(시가 ≤ 피벗 ∧ 저가 ≤ 손절가 ∧ 종가 > 손절가) =
  분봉(2025-08+) 유니버스 합성 사건으로 잰 p 혼합(주), 보수(p=1)·낙관(p=0) 경계 함께.
- 비용: ``korea.cost_rt(청산일, 시장)`` 1배·2배 (+ 매수 +0.1%, 손절·본전 매도 −0.1% — 대조도 같다).
- 같은 메뉴 무작위 셋업 널 300벌, 모멘텀 정합 대조, 10 슬롯 포트폴리오 + 팩터 회귀(TVW·원수익·현금 0%), 분봉 검증.
- 원장: ``research/logs/sepa_explore/vcp_retest/TRIALS.jsonl`` — 팔·민감도 칸을 실행 전에(깔때기 모드는 원장에 쓰지 않는다).

Run:  uv run --with numba --extra pg python research/experiments/vcp_retest_run.py --funnel-only     # 건수만 → FUNNEL.md
      uv run --with numba --extra pg python research/experiments/vcp_retest_run.py --workers 12      # 전체(하한 = FUNNEL.md)
디버그(구간 축소 — 탐색 디렉터리·원장을 더럽히지 않는다, 콘솔은 건수만):
      ... vcp_retest_run.py --event-lo 2025-08-01 --event-hi 2026-03-31 --adv-floor 5000 --out <scratch>/vcp_debug --n-null 8 \\
          --counts-only
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import multiprocessing as mp
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
import meta_gen2_run as mg2  # noqa: E402
import meta_gen5_run as m5  # noqa: E402
from convergence_run import raw_close_panel  # noqa: E402
from krx_quant_core.stats.matched_null import cluster_bootstrap_ci, matched_alpha, matched_control  # noqa: E402
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample  # noqa: E402
from minervini_exits_run import _alpha, _f  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402
from relax_run import KrxPatch  # noqa: E402
from sepa_confluence_gate import N_BOOT, SEED, Data, _key, _log, _pct  # noqa: E402
from sepa_momentum import (  # noqa: E402
    MOM_N_Q,
    N_PER,
    STRATA_MOM,
    day_market_median_half,
    day_market_quantiles,
    momentum_12_1,
)

from research.signals import vcp as V  # noqa: E402
from research.signals.meta_fast import equity_stats  # noqa: E402
from research.signals.minervini_exits import contraction_low_level  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.flow_leg import flow_wide  # noqa: E402
from swing_it.features.korea import cost_rt, tick_size_vec, universe_mask  # noqa: E402
from swing_it.krx_session import NXT_START, SESSION_CLOSE, SESSION_OPEN  # noqa: E402
from swing_it.storage import connect, db_default, read_earnings, read_supply_demand  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "vcp_retest"
SUB = "vcp_retest"
RESULTS_MD = "VCP_RETEST_RESULTS.md"
PREREG_DOC = "research/logs/sepa_explore/VCP_RETEST.md (5632879 + 정정 7dbbd0c + 정정 2 aae4b8e)"
#: DSR 누적 시행 수(기술) — 이 연구 계열 원장 전부(정정 2: ga·ga2·exits_v2·meta_gen1~6 포함).
PRIOR_LEDGERS = (("sepa_confluence", "research/logs"), ("sepa_explore", "research/logs"),
                 ("minervini_exits", "research/logs/sepa_explore"), ("convergence", "research/logs/sepa_explore"),
                 ("relax", "research/logs/sepa_explore"), ("exits_v2", "research/logs/sepa_explore"),
                 ("ga", "research/logs/sepa_explore"), ("ga2", "research/logs/sepa_explore"),
                 *((f"meta_gen{i}", "research/logs/sepa_explore") for i in range(1, 7)))
FUNNEL_MD = "research/logs/sepa_explore/vcp_retest/FUNNEL.md"
FACTORS_PKL = "research/logs/sepa_explore/vcp_retest/factors.pkl"       # 리뷰어 GA2 factors.py 산출(정정 M4 — 저장소로 옮김)
FACTORS_PY = "research/logs/sepa_explore/vcp_retest/factors.py.txt"   # 리뷰어 스크립트 원본 바이트(.py 면 import 스모크가 실행)
FACTOR_COLS = ["MKT_KOSPI_TVW", "MKT_KOSDAQ_TVW", "UMD", "SMB_ADV", "LOWVOL"]
EXTREME_DAYS = ("2026-03-04", "2026-07-31")
NAVER_CSV = "research/data/krx_index_daily_naver.csv"
P1_TRADES = "research/logs/live_swing/trades_b.csv"

# --- 사전등록 수치(VCP_RETEST.md + 정정 — 탐색하지 않는다) ---
EVENT_LO, EVENT_HI = "2017-09-15", "2026-08-31"   # 사건(주문일 d) 구간
CAP_SENS = "2026-04-30"                           # 사건 상한 민감도(정정 M6)
FLOW_LO = "2024-04-25"                            # 수급 팔 사건 하한(정정 M3)
SUB_HI = "2024-12-31"                             # 하위 구간 2017-09 ~ 2024-12
MINUTE_LO = "2025-08-01"                          # 분봉 시작
ADV_FLOOR = 5_000.0                               # ADV20 ≥ 50억 (백만원) — 사전 규칙이 VOID 일 때의 기본
ADV_FLOORS_FUNNEL = (5_000.0, 3_000.0, 2_000.0, 1_000.0)   # 정정 2: 하한 후보(50억·30억·20억·10억)
FLOOR_RULE_N, FLOOR_RULE_SUB = 60, 30             # 정정 2: V0 체결 ≥ 60 ∧ 2017-09~2024-12 ≥ 30 인 가장 높은 하한
NXT_LAST_DAY = "2026-09-11"                       # NXT 미보정 종목의 보유 경로 끝(정정 2 B-M3)
H0_WICK = 0.08                                    # H0 윗꼬리 플래그(기술 칸, 정정 2 A-M3) — (H0 − max(시가, 종가)) / H0 ≥ 8%
RS_MIN, BREADTH_MIN = 70.0, 0.5
SPLIT_EXCL, SPLIT_TOL = 60, 0.02                  # 조정계수 변화 뒤 60 거래일 셋업 제외(정정 M7), 변화 = |비 − 1| > 2%
SPLIT_BATCH_MIN = 5                               # 같은 날 ≥ 5건 = 수집 배치 경계(분할 아님, 정정 2)
X1_VOL_MULT, X1_VOL_WIN = 1.5, 50                 # X1 옛 시점: 신호일 거래량 ≥ 1.5 × 직전 50일 평균
N_NULL = 300
NULL_K = (10, 40)
NULL_J = (5, 15)
NULL_SEED = 20261010
NULL_EXPAND = 5                                   # 풀 < 3 × 필요면 ±5 거래일
NULL_POOL_MULT = 3
R_FIXED = 0.07                                    # 고정 분모 R
#: 분봉 p 추정 합성 사건의 (k, j) 격자 — 널 메뉴 범위(k 10~40, j 5~15)를 5·5 간격으로. 종목-일당 격자 순서 첫 모호 조합 하나.
P_GRID = tuple((k, j) for k in range(10, 41, 5) for j in (5, 10, 15))
SLOTS, SLOT_W = 10, 0.10
NW_LAGS = 5
BE_BAND = 0.005                                   # 본전 = |순| < 0.5%
BIG_LOSS = -0.12
B1_N, B1_SUB_N = 50, 30
V1_DESC_N = 40
GL_MIN, LOSS_MAX = 2.0, 0.5
AVG_LOSS_R_MIN, R_P5_MIN, BIG_LOSS_MAX = -1.0, -1.3, 0.05
T_BAR = 2.0
PRIMARY = "V0"
ARMS: dict[str, dict] = {
    "V0": {"e": False, "flow": False, "lo": EVENT_LO, "kind": "buystop", "desc": "추세 ∧ RS ∧ breadth ∧ VCP — **판정**"},
    "V1": {"e": True, "flow": False, "lo": EVENT_LO, "kind": "buystop", "desc": "V0 ∧ E 가속 (기술)"},
    "V2": {"e": True, "flow": True, "lo": FLOW_LO, "kind": "buystop", "desc": "V1 ∧ 수급 받침 (기술)"},
    "V3": {"e": False, "flow": True, "lo": FLOW_LO, "kind": "buystop", "desc": "V0 ∧ 수급 받침 (기술)"},
    "X1": {"e": False, "flow": False, "lo": EVENT_LO, "kind": "x1",
           "desc": "V0 셋업 · 옛 시점(종가>피벗 ∧ 거래량 1.5× → 다음날 시가) + 새 손절"},
    "X2": {"e": False, "flow": False, "lo": EVENT_LO, "kind": "x2",
           "desc": "V0 셋업 · 새 매수정지 + 옛 손절(min(low[d−10..d−1])×0.99, [3,10]%)"},
    "X2L": {"e": False, "flow": False, "lo": EVENT_LO, "kind": "x2l",
            "desc": "X2 문구판(min(low[d−10..d]) — 진입일 저가 포함, 기술)"},
}
NULL_ARMS = ("V0", "V1", "V2", "V3")
SENS_CELLS = ("V0 보수(p=1)", "V0 낙관(p=0)", "V0 완결 거래만", f"V0 사건 ≤ {CAP_SENS}", "V0 팩터 극단일 제외",
              "V0 2020~21 제외(기술)", "V0 H0 윗꼬리 제외(기술)", "V0 p 재가중(V0 모호 거래 d 구성)")
UNIVERSE_DESC = ("korea.universe_mask(raw_close=daily_bars.close, adv_floor={floor:.0f}) — 보통주·KOSPI/KOSDAQ·ADV20≥{floor:,.0f}"
                 "(백만원)·원종가≥1000·상장252일 (셋업 행 s = d−1), 분할(배치일 제외) 직후 60 거래일 제외")
PRICE_DESC = "read_prices → krx_session.patch_prices(2026-09-14+, 분봉 정규장) — relax_run.KrxPatch"


def _sha(path: str) -> str:
    if not os.path.exists(path):
        return "없음"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _arm_config(name: str, lo: str, hi: str, fac_sha: str, floor: float = ADV_FLOOR) -> dict:
    """원장 지문용 전체 config(팔마다 한 줄)."""
    a = ARMS[name]
    c = {"prereg": PREREG_DOC, "arm": name, "desc": a["desc"], "event_lo": lo, "event_hi": hi,
         "universe": UNIVERSE_DESC.format(floor=floor), "adv_floor": floor, "prices": PRICE_DESC, "trend_template": True, "rs_min": RS_MIN,
         "breadth_min": BREADTH_MIN, "vcp": dict(V.VCP_PARAMS), "split_excl": SPLIT_EXCL,
         "e_accel": dict(V.E_PARAMS) if a["e"] else None,
         "flow_base": {**V.FLOW_PARAMS, "weight": "trade_value"} if a["flow"] else None,
         "buystop": dict(V.BUYSTOP_PARAMS), "cost": "korea.cost_rt(청산일, 시장) 1배 · 2배(바 2)",
         "same_day": f"진입일 종가 ≤ 손절가 = 확정 손절; 모호(시가≤피벗∧저가≤손절가∧종가>손절가) = 분봉 p 혼합(합성 사건 (k, j) "
                     f"격자 {P_GRID}) · 보수 · 낙관", "nxt_cut": NXT_LAST_DAY, "bar5_cost": "1배(판정) · 2배(기술)",
         "kind": a["kind"], "judged": name == PRIMARY}
    if name in NULL_ARMS:
        c["null"] = (f"같은 메뉴 무작위 셋업 {N_NULL}벌, k~U{NULL_K}, j~U{NULL_J}, 근접 띠·9.5%·2%·3%·상한가·정지 같음, 월별 체결 수 × "
                     f"d 3분위 층화, 풀 < {NULL_POOL_MULT}×면 ±{NULL_EXPAND}일, seed {NULL_SEED}")
        c["matched"] = ("모멘텀 정합: 같은 날 × 시장 × 12-1 모멘텀 10분위 × ADV20 2분위(H5), 자기 제외, 5배, seed 0, d 일 시가·같은 d·"
                        "같은 청산(돌파 실패 피벗 = 진입가), 날짜군집 2000")
        c["portfolio"] = f"10 슬롯 × 10%, ADV20 큰 순, 현금 0%, 팩터 {FACTOR_COLS} 원수익 NW 5, factors.pkl sha256 {fac_sha}"
    return c


# ---------------------------------------------------------------------------
# 순수 헬퍼
# ---------------------------------------------------------------------------

def cost_table(days: pd.DatetimeIndex) -> np.ndarray:
    """``[시장(0 KOSPI, 1 KOSDAQ), 날짜 행]`` → 왕복 비용률(청산일 세율)."""
    out = np.empty((2, len(days)))
    for j, m in enumerate(("KOSPI", "KOSDAQ")):
        cache: dict = {}
        for i, d in enumerate(days):
            k = d.date()
            if k not in cache:
                cache[k] = cost_rt(k, m)
            out[j, i] = cache[k]
    return out


def wquantile(x, w, q: float) -> float:
    """가중 분위(가중 누적 ≥ q × 합 인 첫 값). 빈 입력 NaN."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    k = np.isfinite(x) & (w > 0)
    if not k.any():
        return np.nan
    o = np.argsort(x[k], kind="mergesort")
    xs, cw = x[k][o], np.cumsum(w[k][o])
    return float(xs[min(np.searchsorted(cw, q * cw[-1] - 1e-12), len(xs) - 1)])


EXP_COLS = ["w", "net", "net2", "ret", "d", "R", "exit_reason", "hold_days", "complete", "order_day", "entry_day", "key"]


def expand(t: pd.DataFrame, p: float) -> pd.DataFrame:
    """트레이드 → 가중 행. 모호(``amb``) 거래는 보수 행(가중 p, 진입일 손절)과 낙관 행(가중 1−p) 둘로, 나머지는 한 행(가중 1)."""
    if t is None or t.empty:
        return pd.DataFrame(columns=EXP_COLS)
    base = pd.DataFrame({"net": t["net"].to_numpy(), "net2": t["net2"].to_numpy(), "ret": t["ret"].to_numpy(),
                         "d": t["d"].to_numpy(), "exit_reason": t["exit_reason"].to_numpy(),
                         "hold_days": t["hold_days"].to_numpy(),
                         "complete": (t["exit_reason"] != "data_end").to_numpy(),
                         "order_day": t["order_day"].to_numpy(), "entry_day": t["entry_day"].to_numpy(),
                         "key": np.arange(len(t))})
    amb = t["amb"].to_numpy(bool)
    w = np.where(amb, 1.0 - p, 1.0)
    stop0 = base[amb].copy()
    stop0["net"] = t.loc[amb, "net_stop0"].to_numpy()
    stop0["net2"] = t.loc[amb, "net2_stop0"].to_numpy()
    stop0["ret"] = t.loc[amb, "ret_stop0"].to_numpy()
    stop0["exit_reason"] = "stop_entry"
    stop0["hold_days"] = 1
    stop0["complete"] = True
    out = pd.concat([base.assign(w=w), stop0.assign(w=p)], ignore_index=True)
    out = out[out["w"] > 0].reset_index(drop=True)
    out["R"] = out["net"] / out["d"]
    return out


def wstats(x: pd.DataFrame, n_dsr: int = 1) -> dict:
    """가중 행 → 거래 숫자(바 2·5 재료). 순 = 1배 비용, ``net2`` = 2배."""
    if x is None or x.empty:
        return {"n": 0}
    w = x["w"].to_numpy(float)
    W = w.sum()
    net, net2, R = (x[c].to_numpy(float) for c in ("net", "net2", "R"))
    gain, loss = net > 0, net < 0
    reason = x["exit_reason"].to_numpy()

    def wm(v, m=None):
        m = np.ones(len(v), bool) if m is None else m
        return float((v[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0 else np.nan

    ag, al = wm(net, gain), wm(net, loss)
    od = pd.to_datetime(pd.Series(x["order_day"].to_numpy()))
    yr = od.dt.year.to_numpy()
    sub = (od <= pd.Timestamp(SUB_HI)).to_numpy()
    s = {"n": int(round(W)), "n_rows": int(len(x)), "gross_mean": wm(x["ret"].to_numpy(float)), "net_mean": wm(net),
         "net2_mean": wm(net2), "net_median": wquantile(net, w, 0.5),
         "win": float(w[net >= BE_BAND].sum() / W), "be": float(w[np.abs(net) < BE_BAND].sum() / W),
         "lossc": float(w[net <= -BE_BAND].sum() / W), "loss_rate": float(w[loss].sum() / W),
         "gain_rate": float(w[gain].sum() / W),
         "avg_gain": ag, "avg_loss": al,
         "gl_ratio": ag / abs(al) if np.isfinite(ag) and np.isfinite(al) and al < 0 else np.nan,
         "avg_loss_R": wm(R, loss), "r_p5": wquantile(R, w, 0.05), "big_loss": float(w[net < BIG_LOSS].sum() / W),
         "r_mean": wm(R), "r_median": wquantile(R, w, 0.5), "r_fixed_mean": wm(net) / R_FIXED,
         "hold_median": wquantile(x["hold_days"].to_numpy(float), w, 0.5),
         "n_sub": int(round(w[sub].sum())), "net2_sub": wm(net2, sub),
         "net2_ex2025": wm(net2, yr != 2025), "net2_ex2021": wm(net2, (yr != 2020) & (yr != 2021)),
         "stop_rate": float(w[reason == "stop"].sum() / W), "stop_entry_rate": float(w[reason == "stop_entry"].sum() / W),
         "exits": {k: round(float(v), 2) for k, v in pd.Series(w).groupby(reason).sum().items()},
         "exit_net": {k: float(np.average(g["net"], weights=g["w"])) for k, g in x.groupby("exit_reason")},
         "years_n": {int(k): round(float(v), 1) for k, v in pd.Series(w).groupby(yr).sum().items()},
         "years_net": {int(k): float(np.average(g["net"], weights=g["w"])) for k, g in x.groupby(yr)},
         "years_net2": {int(k): float(np.average(g["net2"], weights=g["w"])) for k, g in x.groupby(yr)}}
    # 바 5 의 2배 비용판(기술, 정정 2): 같은 손실 지표를 2배 순수익으로
    R2 = net2 / x["d"].to_numpy(float)
    g2, l2 = net2 > 0, net2 < 0
    ag2, al2 = wm(net2, g2), wm(net2, l2)
    s.update({"loss_rate2": float(w[l2].sum() / W), "avg_loss2": al2,
              "gl_ratio2": ag2 / abs(al2) if np.isfinite(ag2) and np.isfinite(al2) and al2 < 0 else np.nan,
              "avg_loss_R2": wm(R2, l2), "r_p5_2": wquantile(R2, w, 0.05), "big_loss2": float(w[net2 < BIG_LOSS].sum() / W)})
    per = pd.Series(net * w).groupby(x["key"].to_numpy()).sum().to_numpy()     # 거래당 기대 순수익
    d_ = deflated_sharpe_from_sample(per, max(n_dsr, 1)) if len(per) > 2 else None
    s["dsr_prob"] = float(d_["prob_deflated_sharpe"]) if d_ is not None else np.nan
    return s


def bar_check(s: dict, null_p95: float, mom_lo: float, fac_t: float, *, judged: bool = True) -> dict:
    """사전등록(정정·정정 2) 바 1~5 산술(바 2·3 = 2배 비용, 바 5 = 1배 판정 · ``b5_2x`` 2배 기술) + 바 6 기술. 판정은 사람이."""
    def fin(v):
        return v is not None and np.isfinite(v)

    n, ns = s.get("n", 0), s.get("n_sub", 0)
    out = {"b1": bool(n >= B1_N and ns >= B1_SUB_N) if judged else bool(n >= V1_DESC_N),
           "b2": bool(fin(s.get("net2_mean")) and s["net2_mean"] > 0 and fin(s.get("net2_sub")) and s["net2_sub"] > 0
                      and ns >= B1_SUB_N),
           "b3": bool(fin(s.get("net2_mean")) and fin(null_p95) and s["net2_mean"] > null_p95),
           "b4": bool(fin(mom_lo) and mom_lo > 0),
           "b5": bool(fin(s.get("loss_rate")) and s["loss_rate"] < LOSS_MAX and fin(s.get("gl_ratio"))
                      and s["gl_ratio"] >= GL_MIN and fin(s.get("avg_loss_R")) and s["avg_loss_R"] >= AVG_LOSS_R_MIN
                      and fin(s.get("r_p5")) and s["r_p5"] >= R_P5_MIN and fin(s.get("big_loss"))
                      and s["big_loss"] <= BIG_LOSS_MAX),
           "b5_2x": bool(fin(s.get("loss_rate2")) and s["loss_rate2"] < LOSS_MAX and fin(s.get("gl_ratio2"))
                         and s["gl_ratio2"] >= GL_MIN and fin(s.get("avg_loss_R2")) and s["avg_loss_R2"] >= AVG_LOSS_R_MIN
                         and fin(s.get("r_p5_2")) and s["r_p5_2"] >= R_P5_MIN and fin(s.get("big_loss2"))
                         and s["big_loss2"] <= BIG_LOSS_MAX),
           "b6": bool(fin(fac_t) and fac_t > T_BAR)}
    out["n_met"] = int(sum(out[f"b{i}"] for i in range(1, 6)))
    return out


def slot_portfolio(tr: pd.DataFrame, C: np.ndarray, t_lo: int, t_hi: int, cost_mult: float,
                   slots: int = SLOTS, slot_w: float = SLOT_W, take_frac: float = V.BUYSTOP_PARAMS["take_frac"]
                   ) -> "tuple[np.ndarray, dict]":
    """10 슬롯 포트폴리오 날별 수익 ``[t_lo, t_hi]``(행). 열: ``ci, d_row, x_row, entry_px, partial_hit, part_row, partial_px,
    exit_px, cost, adv20``. 슬롯당 전일 종가 자산의 10%(현금이 모자라면 남은 현금), 같은 날 진입은 ADV20 큰 순, 그날 청산되는 자리는
    그날 새 진입에 쓰지 않는다(체결 순서 모름 — 보수), 비용 = ``cost_mult × cost`` × 배분액(청산일), 현금 0%."""
    by_day: dict[int, list] = {}
    for r in tr.sort_values(["d_row", "adv20"], ascending=[True, False]).itertuples(index=False):
        by_day.setdefault(int(r.d_row), []).append(r)
    eq_prev, cash = 1.0, 1.0
    pos: list[dict] = []
    out = np.zeros(t_hi - t_lo + 1)
    expo = np.zeros(t_hi - t_lo + 1)
    taken = skipped = 0
    for t in range(t_lo, t_hi + 1):
        for r in by_day.get(t, []):
            if len(pos) >= slots or cash <= 1e-12:
                skipped += 1
                continue
            alloc = min(slot_w * eq_prev, cash)
            cash -= alloc
            pos.append({"ci": int(r.ci), "u0": alloc / r.entry_px, "u": alloc / r.entry_px, "x": int(r.x_row),
                        "p": int(r.part_row) if bool(r.partial_hit) else -1, "px1": float(r.partial_px),
                        "fin": float(r.exit_px), "alloc": alloc, "cost": float(r.cost), "last": float(r.entry_px)})
            taken += 1
        keep = []
        for q in pos:
            if q["p"] == t:
                cash += take_frac * q["u0"] * q["px1"]
                q["u"] = (1.0 - take_frac) * q["u0"]
            if q["x"] == t:
                cash += q["u"] * q["fin"] - cost_mult * q["cost"] * q["alloc"]
                continue
            keep.append(q)
        pos = keep
        val = cash
        for q in pos:
            c = C[q["ci"], t]
            if np.isfinite(c):
                q["last"] = c
            val += q["u"] * q["last"]
        out[t - t_lo] = val / eq_prev - 1.0
        expo[t - t_lo] = (val - cash) / val if val > 0 else 0.0
        eq_prev = val
    return out, {"taken": taken, "skipped": skipped, "exposure": float(expo.mean()) if len(expo) else np.nan}


def nw_factor(y: pd.Series, X: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """``ga2_post.nw_ols`` 로 y ~ 상수 + X (Newey-West). 결측일은 뺀다."""
    d = pd.concat([y.rename("y"), X], axis=1, sort=True).dropna()
    if len(d) < 60:
        return {"n": int(len(d))}
    cols = list(X.columns)
    r = gp.nw_ols(d["y"].to_numpy(), np.column_stack([np.ones(len(d)), d[cols].to_numpy()]), lags)
    return {"n": r["n"], "alpha_ann": float(r["b"][0] * 252), "alpha_t": float(r["t"][0]), "r2": r["r2"],
            "betas": {c: float(v) for c, v in zip(cols, r["b"][1:], strict=True)}}


# ---------------------------------------------------------------------------
# 널 작업자(fork — 부모의 배열을 그대로 물려받는다; numba 병렬 커널은 쓰지 않는다)
# ---------------------------------------------------------------------------

_G: dict = {}


def _null_task(task: tuple) -> dict:
    arm, b = task
    g = _G
    A, P = g["A"], g["plan"][arm]
    rng = np.random.default_rng([NULL_SEED, NULL_ARMS.index(arm), b])
    ci, s = P["ci"], P["s"]
    m = len(ci)
    k = rng.integers(NULL_K[0], NULL_K[1] + 1, size=m)
    j = rng.integers(NULL_J[0], NULL_J[1] + 1, size=m)
    piv, ip, ll = V.null_levels(A, ci, s, k, j, parallel=False)
    cs = A.C[ci, s]
    with np.errstate(invalid="ignore"):
        band = (cs < piv) & (cs >= V.VCP_PARAMS["close_lo"] * piv) & (ip <= s - V.VCP_PARAMS["dry_min"])   # 피벗 뒤 ≥ 3일(A MINOR 4)
    chk = V.buystop_entry_check(A, ci, s + 1, piv, ll)
    ok = band & (chk["status"] == V.S_OK)
    if ARMS[arm]["flow"]:
        with np.errstate(invalid="ignore"):
            ok &= g["flow"].f_base(ci, s - k, s) > 0
    terc = np.searchsorted(P["edges"], chk["dd"], side="right")         # 0,1,2 — V 팔 d 3분위 경계
    rows = np.flatnonzero(ok)
    sel, pool_ratio = [], []
    short = 0
    for (mo, tq), need in P["need"].items():
        lo_r, hi_r = P["month_rows"][mo]
        d_r = s[rows] + 1
        cand = rows[(terc[rows] == tq) & (d_r >= lo_r) & (d_r <= hi_r)]
        if len(cand) < NULL_POOL_MULT * need:
            cand = rows[(terc[rows] == tq) & (d_r >= lo_r - NULL_EXPAND) & (d_r <= hi_r + NULL_EXPAND)]
        pool_ratio.append(len(cand) / need)
        take = min(need, len(cand))
        short += need - take
        if take:
            sel.append(rng.choice(cand, size=take, replace=False))
    if not sel:
        return {"arm": arm, "b": b, "n": 0, "short": short, "need": P["need_total"], "n_cut": 0}
    sel = np.unique(np.concatenate(sel))                    # ±5일 확장으로 겹친 후보는 한 번만
    sel = sel[np.lexsort((ci[sel], s[sel]))]
    r = V.run_buystop(A, ci[sel], s[sel] + 1, piv[sel], ll[sel], ip[sel], no_overlap=True)
    okk = r["status"] == V.S_OK
    if not okk.any():
        return {"arm": arm, "b": b, "n": 0, "short": short, "need": P["need_total"], "n_cut": 0}
    mk = g["mkt"][ci[sel][okk]]
    n_cut = int(((r["reason"][okk] == V.R_DATA_END) & (r["x"][okk] < A.CUT[ci[sel][okk]])).sum())
    c_opt = g["cost_tab"][mk, r["x"][okk]]
    amb = r["amb"][okk]
    c0 = np.where(amb, g["cost_tab"][mk, (s[sel] + 1)[okk]], c_opt)
    ret, ret0 = r["ret"][okk], r["ret_stop0"][okk]
    p = g["p"]
    out = {"arm": arm, "b": b, "n": int(okk.sum()), "short": short, "selected": int(len(sel)), "need": P["need_total"],
           "n_cut": n_cut,
           "pool_ratio_min": float(np.min(pool_ratio)), "pool_ratio_med": float(np.median(pool_ratio)),
           "amb_rate": float(amb.mean())}
    for tag, pp in (("", p), ("_cons", 1.0), ("_opt", 0.0)):
        out[f"net_mean{tag}"] = float((pp * (ret0 - c0) + (1 - pp) * (ret - c_opt)).mean())
        out[f"net2_mean{tag}"] = float((pp * (ret0 - 2 * c0) + (1 - pp) * (ret - 2 * c_opt)).mean())
    out["r_mean"] = float(np.mean((p * (ret0 - c0) + (1 - p) * (ret - c_opt)) / r["dd"][okk]))
    return out


# ---------------------------------------------------------------------------
# 본체 — 공용 자료(한 번) · 하한별 셋업 · 깔때기 · 전체
# ---------------------------------------------------------------------------

def split_flags(fac: pd.DataFrame, *, tol: float = SPLIT_TOL, excl: int = SPLIT_EXCL,
                batch_min: int = SPLIT_BATCH_MIN) -> dict:
    """조정계수(조정 종가 / 원 종가) 변화 → 분할 제외 패널(정정 M7 · 정정 2).

    변화 = |계수_t / 계수_{t−1} − 1| > ``tol``. 같은 날 ``batch_min`` 건 이상 몰린 날은 수집 배치 경계라 분할이 아니다(제외에서 뺀다).
    ``recent`` = 개별 변화 뒤 ``excl`` 거래일(변화일 포함) bool 패널(날짜×종목), ``cum`` = 개별 변화 누적합(행 r 까지 = cum[r+1],
    종목×날짜 — 셋업의 [ip−49, s] 창 검사용). 배치 이전의 실제 분할은 원 종가가 이미 조정돼 있어 잡지 못한다(한계)."""
    chg = (fac / fac.shift(1) - 1.0).abs() > tol
    per_day = chg.sum(axis=1)
    batch = (per_day >= batch_min).to_numpy()
    real = chg.to_numpy(bool) & ~batch[:, None]
    recent = pd.DataFrame(real.astype(float), index=fac.index).rolling(excl, min_periods=1).max().fillna(0).to_numpy(bool)
    cum = np.vstack([np.zeros((1, real.shape[1])), np.cumsum(real, axis=0)]).T.copy()
    return {"recent": recent, "cum": cum, "n_events": int(chg.to_numpy().sum()), "n_real": int(real.sum()),
            "batch_days": {str(d.date()): int(v) for d, v in per_day[batch].items()}}


def floor_rule(counts: "dict[float, dict]") -> "tuple[float | None, bool]":
    """정정 2 사전 규칙: V0 체결 ≥ 60 ∧ 2017-09~2024-12 체결 ≥ 30 인 가장 높은 하한. 없으면 (None, VOID=True)."""
    ok = [f for f, c in counts.items() if c["V0"] >= FLOOR_RULE_N and c["V0_sub"] >= FLOOR_RULE_SUB]
    return (max(ok), False) if ok else (None, True)


def read_floor_rule(path: str = FUNNEL_MD) -> "tuple[float | None, bool]":
    """``FUNNEL.md`` 의 기계 판독 줄 ``<!-- adv_floor_rule: 5000 · v0_void: false -->`` 를 읽는다. 없으면 (None, None)."""
    if not os.path.exists(path):
        return None, None
    import re  # noqa: PLC0415

    m = re.search(r"adv_floor_rule:\s*([0-9.]+|none)\s*·\s*v0_void:\s*(true|false)", open(path, encoding="utf-8").read())
    if not m:
        return None, None
    return (None if m.group(1) == "none" else float(m.group(1))), m.group(2) == "true"


class Ctx:
    """하한과 무관한 공용 자료 한 벌(가격·보정·분할·실적·수급·배열)."""

    def __init__(self, t0: float):
        load_env_db()
        self.con = connect(db_default())
        self.patch = KrxPatch(self.con)
        with self.patch.installed():
            self.D = Data(self.con, t0)
            self.p = self.D.p
            self.raw_close = raw_close_panel(self.con, self.p)
        p = self.p
        self.days, self.codes = p.days, p.close.columns
        self.split = split_flags(p.close / self.raw_close.where(self.raw_close > 0))
        self.A = V.Arrays(p.open, p.high, p.low, p.close, self.D.feats["ma50"], p.volume)
        key = next((k for k in self.patch.unpatched if k.startswith("daily_bars_adjusted")), None)
        un = self.patch.unpatched.get(key, pd.DataFrame(columns=["code", "date"])) if key else pd.DataFrame(columns=["code"])
        self.cut_codes = sorted(set(un["code"].astype(str))) if len(un) else []
        self.n_cut_stocks = self.A.set_cut(self.cut_codes, NXT_START)
        self.VOL = np.ascontiguousarray(p.volume.reindex(index=self.days, columns=self.codes).to_numpy(float).T)
        self.mkt = np.where(p.market.reindex(self.codes).to_numpy() == "KOSDAQ", 1, 0).astype(np.int64)
        lowp = p.low.reindex(index=self.days, columns=self.codes)
        self.low10_prev = lowp.rolling(10, min_periods=10).min().to_numpy(float)     # 행 s: min(low[s−9..s]) = min(low[d−10..d−1])
        self.low10_lit = contraction_low_level(lowp, 10).to_numpy(float)             # 행 d: min(low[d−10..d])(문구판)
        self.adv = self.D.adv20.reindex(index=self.days, columns=self.codes).to_numpy(float)
        self.ma50 = self.D.feats["ma50"].reindex(index=self.days, columns=self.codes).to_numpy(float)
        self.tt = self.D.feats["tt"].reindex(index=self.days, columns=self.codes).fillna(False).to_numpy(bool)
        self.rs = self.D.feats["rs"].reindex(index=self.days, columns=self.codes).to_numpy(float)
        self.br = self.D.feats["breadth"].reindex(index=self.days, columns=self.codes).to_numpy(float)
        ea = read_earnings(self.con, all_versions=True, cols=("code", "period", "avail_date", "knowledge_date", "netinc",
                                                             "netinc_prior", "revenue", "revenue_prior"))
        self.est = V.standalone_quarter_states(ea)
        del ea
        sd = read_supply_demand(self.con, session="raw", cols=("code", "date", "foreign_", "institution", "acc_trde_qty"),
                                start="2016-09-01")
        fw = flow_wide(sd, index=self.days, columns=self.codes)
        del sd
        self.FCw = V.FlowCum(fw["foreign"], fw["institution"], fw["volume"],
                             weight=p.trade_value.reindex(index=self.days, columns=self.codes))
        self.FCq = V.FlowCum(fw["foreign"], fw["institution"], fw["volume"])
        del fw

    def universe(self, floor: float) -> pd.DataFrame:
        return universe_mask(self.p.close, self.p.trade_value, self.D.meta, raw_close=self.raw_close, adv_floor=floor)


def build_setups(ctx: Ctx, floor: float, lo_all: str, hi: str, lo_of: dict) -> dict:
    """하한 ``floor`` 의 셋업 표(신호 행 s 후보 → VCP → E·수급·열)·팔 행·깔때기. 성과 계산 없음."""
    A, days, codes = ctx.A, ctx.days, ctx.codes
    nd = len(days)
    h5 = ctx.universe(floor)
    d_ok = np.asarray((days >= pd.Timestamp(lo_all)) & (days <= pd.Timestamp(hi)))
    s_ok = np.zeros(nd, bool)
    s_ok[:-1] = d_ok[1:]
    base_nosplit = (h5.to_numpy(bool) & ctx.tt & (ctx.rs >= RS_MIN) & (ctx.br > BREADTH_MIN) & s_ok[:, None])
    base = base_nosplit & ~ctx.split["recent"]
    si, ci = np.nonzero(base)
    sc = V.vcp_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci, si)
    no_lr = V.vcp_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci, si, {"low_rise": 0.0})
    S = pd.DataFrame({"code": codes[ci], "sig_idx": si, "day_idx": si + 1, "col_idx": ci, **sc})
    S["event_day"] = days[si]
    S["order_day"] = days[si + 1]
    S["valid"] = S["status"] == V.V_OK
    # 거래량 창 [ip−49, s] 안 개별 조정계수 변화(A MINOR 5) — VCP 통과 행만
    vv = S["valid"].to_numpy()
    ipv = np.maximum(S["ip"].to_numpy() - 49, 0)
    cum = ctx.split["cum"]
    win_chg = np.zeros(len(S), bool)
    win_chg[vv] = (cum[ci[vv], si[vv] + 1] - cum[ci[vv], ipv[vv]]) > 0
    S.loc[win_chg, "valid"] = False
    vv = S["valid"].to_numpy()
    S["market"] = ctx.p.market.reindex(S["code"]).to_numpy()
    lit = V.vcp_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci[vv], si[vv], {"confirmed_only": True})
    S["pivot_inprogress"] = False
    S.loc[vv, "pivot_inprogress"] = lit["ip"] != S.loc[vv, "ip"].to_numpy()
    i0 = np.maximum(S["i0"].to_numpy(), 0)
    S["h0_wick"] = (S["h0"] - np.maximum(A.O[ci, i0], A.C[ci, i0])) / S["h0"]
    S["h0_wick_flag"] = S["h0_wick"] >= H0_WICK
    E = V.e_accel_asof(ctx.est, S["code"].to_numpy(), S["event_day"].to_numpy())
    for c in E.columns:
        S[c] = E[c].to_numpy()
    S["e_accel_eps"] = False
    if vv.any():
        S.loc[vv, "e_accel_eps"] = _eps_flags(ctx.con, S.loc[vv])
    cv, iv, sv = S["col_idx"].to_numpy(), S["i0"].to_numpy(), S["sig_idx"].to_numpy()
    S["f_base"] = np.where(vv, ctx.FCw.f_base(cv, iv, sv), np.nan)
    S["f_base_qty"] = np.where(vv, ctx.FCq.f_base(cv, iv, sv), np.nan)
    S["adv20"] = ctx.adv[S["sig_idx"], S["col_idx"]]
    S["ma50_s"] = ctx.ma50[S["sig_idx"], S["col_idx"]]
    S["close_s"] = A.C[S["col_idx"], S["sig_idx"]]
    S["old_low"] = ctx.low10_prev[S["sig_idx"], S["col_idx"]]                       # X2 주: min(low[d−10..d−1])
    S["old_low_lit"] = ctx.low10_lit[np.minimum(S["day_idx"], nd - 1), S["col_idx"]]  # X2L 기술: min(low[d−10..d])
    S["base_len"] = S["sig_idx"] - S["i0"]
    dts = S["order_day"]
    rng_ = {n: (dts >= pd.Timestamp(lo_of[n])) & (dts <= pd.Timestamp(hi)) for n in ARMS}
    v0 = S["valid"]
    fpos = S["f_base"] > 0
    arm_rows = {"V0": v0 & rng_["V0"], "V1": v0 & S["e_accel"] & rng_["V1"],
                "V2": v0 & S["e_accel"] & fpos & rng_["V2"], "V3": v0 & fpos & rng_["V3"],
                "X1": v0 & rng_["X1"], "X2": v0 & rng_["X2"], "X2L": v0 & rng_["X2L"]}
    flow_period = v0 & rng_["V3"]
    bl = pd.cut(S["base_len"], [0, 30, 65, 130, 251], right=False, labels=["15~29", "30~64", "65~129", "130~250"])
    funnel = {"floor": floor, "base_nosplit": int(base_nosplit.sum()), "split_excl": int((base_nosplit & ctx.split["recent"]).sum()),
              "split_events": ctx.split["n_events"], "split_real": ctx.split["n_real"],
              "split_batch_days": ctx.split["batch_days"], "split_window_excl": int(win_chg.sum()), "base": int(len(S)),
              "status": {V.VCP_STATUS[k]: int(v) for k, v in S["status"].value_counts().items()},
              "valid": int(v0.sum()), "lower_low_excl": int((no_lr["status"] == V.V_OK).sum() - (sc["status"] == V.V_OK).sum()),
              "pivot_inprogress": int(S["pivot_inprogress"].sum()),
              "h0_wick_valid": int((v0 & S["h0_wick_flag"]).sum()),
              "e_known_valid": int((v0 & S["e_known"]).sum()), "e_accel_valid": int((v0 & S["e_accel"]).sum()),
              "e_accel_eps_valid": int((v0 & S["e_accel_eps"]).sum()),
              "flow_period_valid": int(flow_period.sum()),
              "flow_known_valid": int((flow_period & S["f_base"].notna()).sum()),
              "flow_pos_valid": int((flow_period & fpos).sum()),
              "flow_qty_pos_valid": int((flow_period & (S["f_base_qty"] > 0)).sum()),
              "flow_unknown_by_base": {str(k): int(v) for k, v in
                                       S.loc[flow_period & S["f_base"].isna(), "base_len"].pipe(
                                           lambda x: bl[x.index]).value_counts().sort_index().items()},
              "arms": {n: int(m.sum()) for n, m in arm_rows.items()}}
    return {"S": S, "h5": h5, "arm_rows": arm_rows, "rng": rng_, "funnel": funnel}


def simulate_arms(ctx: Ctx, B: dict) -> "tuple[dict, dict, int]":
    """팔별 시뮬(진입 + 보유 거르기에 쓰는 청산). 반환 = (트레이드 원표, 버림 사유 수, X1 종가 돌파 신호 수). 비용·순수익은 붙이지 않는다."""
    A, S, nd = ctx.A, B["S"], len(ctx.days)
    trades, skips = {}, {}
    for n in ("V0", "V1", "V2", "V3", "X2", "X2L"):
        ev = S[B["arm_rows"][n]].reset_index(drop=True)
        if n == "X2L":
            ev = ev.assign(old_low=ev["old_low_lit"])
        t, sk = V.simulate_buystop(A, ev, old_stop=n in ("X2", "X2L"))
        trades[n], skips[n] = t, (sk["skip_reason"].value_counts().to_dict() if len(sk) else {})
        if n == "V0":
            trades["_V0_skips"] = sk
    ev = S[B["arm_rows"]["X1"]].reset_index(drop=True)
    dv, cv_ = ev["day_idx"].to_numpy(), ev["col_idx"].to_numpy()
    ds_ = np.minimum(dv, nd - 1)
    vavg = np.array([ctx.VOL[c, d - X1_VOL_WIN:d].mean() if d >= X1_VOL_WIN else np.nan
                     for c, d in zip(cv_, ds_, strict=True)])
    with np.errstate(invalid="ignore"):
        brk = (dv < nd) & (A.C[cv_, ds_] > ev["pivot"].to_numpy()) & (ctx.VOL[cv_, ds_] >= X1_VOL_MULT * vavg)
    ev = ev[brk].assign(sig_row=lambda x: x["day_idx"]).reset_index(drop=True)
    tx, skx = V.simulate_x1(A, ev)
    trades["X1"], skips["X1"] = tx, (skx["skip_reason"].value_counts().to_dict() if len(skx) else {})
    return trades, skips, int(len(ev))


def amb_overlap_count(t: pd.DataFrame, sk: pd.DataFrame) -> int:
    """A MINOR 1: 보유 중으로 빠진 셋업 중 막은 거래가 모호 거래이고 보수 경로(진입일 손절)라면 비었을 것의 수."""
    if t is None or t.empty or sk is None or sk.empty:
        return 0
    o = sk[sk["skip_reason"] == "skip_overlap"]
    if o.empty:
        return 0
    q = pd.DataFrame({"code": o["code"].astype(str).to_numpy(), "s": (o["day_idx"] - 1).to_numpy(np.int64)})
    tr = pd.DataFrame({"code": t["code"].astype(str).to_numpy(), "s": t["entry_row"].to_numpy(np.int64),
                       "amb": t["amb"].to_numpy(bool), "entry_row": t["entry_row"].to_numpy(np.int64)})
    m = pd.merge_asof(q.sort_values("s"), tr.sort_values("s"), on="s", by="code", direction="backward")
    return int((m["amb"].fillna(False).astype(bool) & (m["s"] >= m["entry_row"])).sum())


def funnel_counts(ctx: Ctx, floor: float, lo_all: str, hi: str, lo_of: dict) -> dict:
    """``--funnel-only`` 한 하한 — 건수만(청산 결과·수익은 어디에도 쓰지 않는다; 보유 거르기의 청산일은 시뮬 안에서만)."""
    B = build_setups(ctx, floor, lo_all, hi, lo_of)
    trades, skips, n_x1sig = simulate_arms(ctx, B)
    t0 = trades["V0"]
    sub = (pd.to_datetime(t0["order_day"]) <= pd.Timestamp(SUB_HI)) if len(t0) else pd.Series(dtype=bool)
    out = {"funnel": B["funnel"], "x1_signals": n_x1sig, "skips": skips,
           **{n: int(len(trades[n])) for n in ("V0", "V1", "V2", "V3", "X1", "X2", "X2L")},
           "V0_sub": int(sub.sum()) if len(t0) else 0,
           "V0_years": {int(k): int(v) for k, v in pd.to_datetime(t0["order_day"]).dt.year.value_counts().sort_index().items()}
           if len(t0) else {},
           "V0_base_len": {q: float(t0["base_len"].quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if len(t0) else {},
           "V0_h0_wick": int(t0["h0_wick_flag"].sum()) if len(t0) else 0,
           "V0_pivot_inprogress": int(t0["pivot_inprogress"].sum()) if len(t0) else 0,
           "V0_liquid_1pct": int((t0["adv20"] > 1_000.0).sum()) if len(t0) else 0,
           "V0_adv_median": float(t0["adv20"].median()) if len(t0) else np.nan,
           "V0_amb_overlap": amb_overlap_count(t0, trades.get("_V0_skips"))}
    return out


def write_funnel(path: str, res: dict, lo_all: str, hi: str, rule: "tuple", elapsed: float, ctx: Ctx) -> None:
    """``FUNNEL.md`` — 하한별 건수와 사전 규칙의 산술(성과 숫자 없음)."""
    floor, void = rule
    fl = sorted(res, reverse=True)
    L = [f"# VCP 재시험 — 깔때기(건수만, 성과 미계산) {date.today().isoformat()}", "",
         f"`vcp_retest_run.py --funnel-only` (정정 2 aae4b8e 의 유니버스 하한 사전 규칙). 사건(주문일) {lo_all}~{hi}. 시뮬은 진입 체결과 "
         "보유 중 거르기(그 거르기에 필요한 청산일)까지만 쓰고, 청산 사유·수익·손절 여부는 어디에도 적지 않는다. 실행 "
         f"{elapsed / 60:.1f}분.", "",
         f"**사전 규칙:** V0 체결 ≥ {FLOOR_RULE_N} ∧ 2017-09~2024-12 체결 ≥ {FLOOR_RULE_SUB} 인 가장 높은 하한 → "
         + (f"**ADV20 ≥ {floor:,.0f} 백만원({floor / 100:,.0f}억)** 이 판정 유니버스." if floor is not None
            else "**넷 다 미달 → V0 VOID**(기술만; 사용자 가설은 '이 데이터의 진짜 VCP 가 너무 드물어 통계 판정 불가' 로 적고 전진 페이퍼로)."),
         f"<!-- adv_floor_rule: {'none' if floor is None else f'{floor:.0f}'} · v0_void: {'true' if void else 'false'} -->", "",
         "| ADV20 하한(백만원) | V0 체결 | 하위(≤2024-12) | 규칙 충족 | V1 | V2 | V3 | X1 | X2 | X2L(문구판) | VCP 셋업 행 | 셋업 후보 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fl:
        r = res[f]
        ok = r["V0"] >= FLOOR_RULE_N and r["V0_sub"] >= FLOOR_RULE_SUB
        L.append(f"| {f:,.0f} | {r['V0']} | {r['V0_sub']} | {'✓' if ok else '✗'} | {r['V1']} | {r['V2']} | {r['V3']} | {r['X1']} | "
                 f"{r['X2']} | {r['X2L']} | {r['funnel']['valid']:,} | {r['funnel']['base']:,} |")
    L += ["", "## 하한별 깔때기", ""]
    for f in fl:
        r, fu = res[f], res[f]["funnel"]
        st = fu["status"]
        bl = r["V0_base_len"]
        L += [f"### ADV20 ≥ {f:,.0f} 백만원", "",
              f"- 셋업 후보(신호 행 s: H5 ∧ 추세 ∧ RS ≥ 70 ∧ breadth > 0.5) {fu['base_nosplit']:,} − 분할 직후 60일 {fu['split_excl']:,} = "
              f"{fu['base']:,}",
              "- VCP 첫 탈락: " + " · ".join(f"{k} {v:,}" for k, v in sorted(st.items(), key=lambda kv: V.VCP_STATUS.index(kv[0]))),
              f"- VCP 통과(거래량 창 [ip−49, s] 안 개별 분할 {fu['split_window_excl']:,} 제외 뒤) **{fu['valid']:,}** · 하락 계단 조건이 지운 셋업 "
              f"{fu['lower_low_excl']:,} · 피벗 = 진행 중 고점 {fu['pivot_inprogress']:,} · H0 윗꼬리 ≥ {H0_WICK:.0%} {fu['h0_wick_valid']:,}",
              f"- E 알려짐 {fu['e_known_valid']:,} · E 가속 {fu['e_accel_valid']:,} (주당 기준 {fu['e_accel_eps_valid']:,}) · 수급 기간 VCP "
              f"{fu['flow_period_valid']:,} → F_base 알려짐 {fu['flow_known_valid']:,} · > 0 {fu['flow_pos_valid']:,} (수량 합 > 0 "
              f"{fu['flow_qty_pos_valid']:,}) · 모름 기반 길이별 {fu['flow_unknown_by_base']}",
              f"- 팔 셋업 행: {fu['arms']} · X1 종가 돌파 신호 {r['x1_signals']:,}",
              f"- **V0 체결 {r['V0']}** (하위 {r['V0_sub']}) · 연도별 {r['V0_years']}",
              "- V0 체결 기반 길이(거래일) 최소/25%/중앙/75%/최대: "
              + ("/".join(f"{bl[q]:.0f}" for q in (0.0, 0.25, 0.5, 0.75, 1.0)) if bl else "—"),
              f"- V0 체결 중 H0 윗꼬리 ≥ {H0_WICK:.0%}: {r['V0_h0_wick']} ({r['V0_h0_wick'] / r['V0']:.0%})" if r["V0"] else
              "- V0 체결 없음",
              f"- V0 체결 중 피벗 = 진행 중 고점 {r['V0_pivot_inprogress']} · 보유 중 거르기에서 모호 거래가 막은 셋업(보수 경로면 비었을 것) "
              f"{r['V0_amb_overlap']}",
              f"- 개인 체결 가능성: 슬롯당 1,000만원 < ADV20 의 1% (ADV20 > 1,000 백만원) 인 V0 체결 {r['V0_liquid_1pct']}/{r['V0']} · "
              f"ADV20 중앙 {r['V0_adv_median']:,.0f} 백만원",
              "- 주문·체결 결과(버림 사유): " + " · ".join(f"{n}: " + ", ".join(f"{k} {v}" for k, v in sorted(sk.items()))
                                                      for n, sk in r["skips"].items() if sk), ""]
    fu0 = res[fl[0]]["funnel"]
    L += ["## 공통", "",
          f"- 분할 조정계수 변화 {fu0['split_events']:,}건 중 같은 날 ≥ {SPLIT_BATCH_MIN}건인 수집 배치일 {fu0['split_batch_days']} 를 빼고 "
          f"개별 {fu0['split_real']:,}건만 제외에 쓴다(배치 이전의 실제 분할은 원 종가가 이미 조정돼 있어 못 잡는다 — 한계).",
          f"- NXT 미보정 행이 있는 종목 {ctx.n_cut_stocks:,}개 — 보유 경로는 {NXT_LAST_DAY} 에서 `data_end` 로 끊는다.",
          "- 판정은 사람이 한다. 이 문서는 사전 규칙의 산술만 적는다.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(--out 을 탐색 디렉터리 밖으로)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR, help="결과·원장 디렉터리(기본 = 탐색 디렉터리)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4), help="널 병렬 작업자 수")
    ap.add_argument("--factors", default=FACTORS_PKL)
    ap.add_argument("--n-null", type=int, default=N_NULL, help="디버그 전용 — 널 벌 수 축소")
    ap.add_argument("--p-hi", default=None, help="디버그 전용 — 분봉 p 추정 구간 상한 축소")
    ap.add_argument("--counts-only", action="store_true", help="콘솔에 건수·시간만(성과 숫자 안 찍음)")
    ap.add_argument("--adv-floor", default=None,
                    help="ADV20 하한(백만원). 전체 실행: 생략하면 FUNNEL.md 의 사전 규칙 값. --funnel-only: 쉼표 목록(기본 5000,3000,2000,1000)")
    ap.add_argument("--funnel-only", action="store_true", help="깔때기 건수만(널·대조·성과 미계산) → FUNNEL.md")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.event_lo or args.event_hi or args.n_null != N_NULL or args.p_hi) and not debug:
        ap.error("구간·널 축소는 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    sub_dir = os.path.join(out_dir, SUB)
    os.makedirs(sub_dir, exist_ok=True)
    lo_all = max(EVENT_LO, args.event_lo) if args.event_lo else EVENT_LO
    hi = min(EVENT_HI, args.event_hi) if args.event_hi else EVENT_HI
    lo_of = {n: max(a["lo"], lo_all) for n, a in ARMS.items()}
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()

    if args.funnel_only:
        floors = [float(x) for x in (args.adv_floor or ",".join(str(int(f)) for f in ADV_FLOORS_FUNNEL)).split(",")]
        ctx = Ctx(t0)
        _log(f"공용 자료 · NXT 절단 종목 {ctx.n_cut_stocks:,}", t0)
        res = {}
        for f in floors:
            res[f] = funnel_counts(ctx, f, lo_all, hi, lo_of)
            _log(f"하한 {f:,.0f}: V0 {res[f]['V0']} (하위 {res[f]['V0_sub']}) · V1 {res[f]['V1']} · V2 {res[f]['V2']} · V3 "
                 f"{res[f]['V3']} · X1 {res[f]['X1']} · X2 {res[f]['X2']}", t0)
        rule = floor_rule(res)
        path = os.path.join(sub_dir, "FUNNEL.md")
        write_funnel(path, res, lo_all, hi, rule, time.time() - t0, ctx)
        print(f"사전 규칙 → 하한 {rule[0]} · V0 VOID {rule[1]} → {path}")
        return 0

    if args.adv_floor is not None:
        floor, void = float(args.adv_floor), None
    else:
        floor, void = read_floor_rule()
        if floor is None and void is None:
            ap.error(f"--adv-floor 를 주거나 먼저 --funnel-only 로 {FUNNEL_MD} 를 만든다")
        if floor is None:
            floor = ADV_FLOOR
    timings: dict[str, float] = {}
    fac_sha = _sha(args.factors)

    # --- 원장: 팔 + 민감도 칸을 결과 전에 ---
    cfgs = {n: _arm_config(n, lo_of[n], hi, fac_sha, floor) for n in ARMS}
    for n, c in cfgs.items():
        record_trial(LABEL, c, logs_dir=out_dir, note=f"VCP 재시험 팔 {n}: {ARMS[n]['desc']}")
    for cell in SENS_CELLS:
        record_trial(LABEL, {**cfgs[PRIMARY], "sensitivity": cell}, logs_dir=out_dir, note=f"VCP 재시험 민감도: {cell}")
    n_led = count_trials(LABEL, logs_dir=out_dir)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
    n_dsr = n_led + sum(n_prior.values())
    _log(f"원장 {LABEL}: {n_led}줄 · DSR N {n_dsr}", t0)

    ctx = Ctx(t0)
    A, days, codes, mkt = ctx.A, ctx.days, ctx.codes, ctx.mkt
    p = ctx.p
    cost_tab = cost_table(days)
    timings["load"] = time.time() - t0
    B = build_setups(ctx, floor, lo_all, hi, lo_of)
    S, h5, rng_, funnel = B["S"], B["h5"], B["rng"], B["funnel"]
    funnel["floor_rule_void"] = void
    funnel["n_cut_stocks"] = ctx.n_cut_stocks
    funnel["nxt_checks"] = _nxt_checks(ctx)
    _log(f"하한 {floor:,.0f} · 셋업 후보 {len(S):,} · VCP {int(S['valid'].sum()):,} · 팔 {funnel['arms']}", t0)
    mom = momentum_12_1(p.close)
    q_mom = day_market_quantiles(mom, h5, p.market, n_q=MOM_N_Q).to_numpy(float)
    adv2 = day_market_median_half(ctx.D.adv20, h5, p.market).to_numpy(float)
    del mom

    # --- 분봉 p (정정 C3 · 정정 2 B-M1) ---
    t1 = time.time()
    pinfo = _estimate_p(ctx.con, A, h5, days, args.p_hi)
    timings["p_minute"] = time.time() - t1
    pval = pinfo["p"]
    _log(f"분봉 p: 단계 {pinfo['stage']} · 판별 {pinfo['n_classified']:,}", t0)

    # --- 1차: 시뮬 ---
    t1 = time.time()
    raw, skips, n_x1sig = simulate_arms(ctx, B)
    funnel["x1_signals"] = n_x1sig
    funnel["v0_amb_overlap"] = amb_overlap_count(raw["V0"], raw.get("_V0_skips"))
    trades = {n: _finish(raw[n], cost_tab, mkt, A.CUT) for n in ARMS}
    pinfo["p_reweighted"] = _p_reweight(pinfo, trades["V0"])
    timings["sim"] = time.time() - t1
    _log("팔 체결: " + " · ".join(f"{n} {len(t)}" for n, t in trades.items()), t0)

    # --- 2차: 같은 메뉴 무작위 셋업 널 ---
    t1 = time.time()
    v0 = S["valid"]
    pool_base = ~v0.to_numpy()
    valid_keys = set(zip(S.loc[v0, "col_idx"].to_numpy(), S.loc[v0, "sig_idx"].to_numpy(), strict=True))
    plan = {}
    month_rows = _month_rows(days)
    for n in NULL_ARMS:
        t = trades[n]
        if t.empty:
            continue
        pm = pool_base & rng_[n].to_numpy() & (S["e_accel"].to_numpy(bool) if ARMS[n]["e"] else True)
        pool_ci, pool_s = S["col_idx"].to_numpy()[pm], S["sig_idx"].to_numpy()[pm]
        assert not any((c_, s_) in valid_keys for c_, s_ in zip(pool_ci, pool_s, strict=True)), "널 풀에 VCP 셋업 (종목, 날짜)"
        dd = t["d"].to_numpy(float)
        edges = np.quantile(dd, [1 / 3, 2 / 3]) if len(t) >= 3 else np.array([np.inf, np.inf])
        terc = np.searchsorted(edges, dd, side="right")
        mo = t["order_day"].dt.to_period("M").astype(str).to_numpy()
        need = pd.Series(1, index=pd.MultiIndex.from_arrays([mo, terc])).groupby(level=[0, 1]).sum().to_dict()
        plan[n] = {"ci": pool_ci.astype(np.int64), "s": pool_s.astype(np.int64), "edges": edges, "need": need,
                   "need_total": int(len(t)), "month_rows": month_rows, "pool_rows": int(pm.sum())}
    _G.update({"A": A, "plan": plan, "flow": ctx.FCw, "cost_tab": cost_tab, "mkt": mkt, "p": pval})
    tasks = [(n, b) for n in plan for b in range(args.n_null)]
    rows_null: list[dict] = []
    if tasks:
        if args.workers > 1:
            with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as ex:
                rows_null = list(ex.map(_null_task, tasks, chunksize=4))
        else:
            rows_null = [_null_task(tk) for tk in tasks]
    nulls = pd.DataFrame(rows_null)
    funnel["null_pool_rows"] = {n: pl["pool_rows"] for n, pl in plan.items()}
    timings["null"] = time.time() - t1
    _log(f"널 {len(tasks)}벌 ({args.workers} 작업자) {timings['null']:.1f}s", t0)

    # --- 3차: 모멘텀 정합 대조 ---
    t1 = time.time()
    moms = {n: _momentum_control(trades[n], A, h5, q_mom, adv2, mkt, days, codes, pval) for n in NULL_ARMS}
    timings["matched"] = time.time() - t1

    # --- 4차: 포트폴리오 + 팩터 ---
    t1 = time.time()
    F = pd.read_pickle(args.factors) if os.path.exists(args.factors) else None
    kospi = _naver_returns("KOSPI")
    ports = {n: _portfolio(trades[n], A, days, lo_of[n], hi, F, kospi, pval) for n in ARMS}
    timings["portfolio"] = time.time() - t1

    # --- 5차: 전략 트레이드 분봉 검증(기술) ---
    t1 = time.time()
    minute = _minute_check(ctx.con, trades, A)
    timings["minute_trades"] = time.time() - t1

    # --- 숫자 ---
    rows: dict = {}
    for n in ARMS:
        t = trades[n]
        s = {"arm": n, "n_setups": funnel["arms"][n], "skips": skips[n], "n_fills": len(t)}
        for tag, pp in (("", pval), ("cons_", 1.0), ("opt_", 0.0)):
            s.update({f"{tag}{k}": v for k, v in wstats(expand(t, pp), n_dsr).items()})
        s.update(_trade_desc(t))
        if n in NULL_ARMS and not nulls.empty and (nulls["arm"] == n).any():
            z = nulls[(nulls["arm"] == n) & (nulls["n"] > 0)]
            za = nulls[nulls["arm"] == n]
            for tag in ("", "_cons", "_opt"):
                s[f"null_p95{tag}"] = float(np.nanpercentile(z[f"net2_mean{tag}"], 95)) if len(z) else np.nan
            s.update({"null_k": int(len(z)), "null_p50": float(np.nanmedian(z["net2_mean"])) if len(z) else np.nan,
                      "null_rank": float((z["net2_mean"] < s.get("net2_mean", np.nan)).mean()) if len(z) else np.nan,
                      "null_n_med": float(z["n"].median()) if len(z) else np.nan,
                      "null_fill_ratio_med": float((za["n"] / za["need"]).median()) if len(za) else np.nan,
                      "null_short_med": float(za["short"].median()),
                      "null_cut_med": float(za["n_cut"].median()) if "n_cut" in za else np.nan,
                      "null_pool_min": float(z["pool_ratio_min"].min()) if len(z) else np.nan,
                      "null_pool_med": float(z["pool_ratio_med"].median()) if len(z) else np.nan,
                      "null_r_p95": float(np.nanpercentile(z["r_mean"], 95)) if len(z) else np.nan})
        if moms.get(n):
            s.update(moms[n])
        s.update(ports[n]["stats"])
        if n in NULL_ARMS:
            for tag, nt in (("", ""), ("cons_", "_cons"), ("opt_", "_opt")):
                sub = {k[len(tag):]: v for k, v in s.items() if k.startswith(tag)} if tag else s
                b = bar_check(sub, s.get(f"null_p95{nt}", np.nan), s.get(f"{tag}mom_lo", np.nan),
                              s.get("fac_alpha_t", np.nan), judged=(n == PRIMARY))
                s.update({f"{tag}bar_{k}": v for k, v in b.items()})
        rows[n] = s
    sens = (_sensitivities(trades[PRIMARY], pval, n_dsr, rows[PRIMARY], pinfo.get("p_reweighted", np.nan))
            if not trades[PRIMARY].empty else {})
    pairs = {x: _pair_compare(trades["V0"], trades[x], pval) for x in ("X1", "X2", "X2L")}
    f_terc = _flow_terciles(trades["V0"], pval, lo_of["V3"])
    p1 = _p1_side(lo_of[PRIMARY], hi)

    elapsed = time.time() - t0
    L = render(rows, funnel, sens, pairs, f_terc, p1, pinfo, minute, ports, lo_of, hi, n_led, n_prior, n_dsr,
               out_dir, debug, elapsed, timings, ctx.patch, p, args, fac_sha)
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    for n, t in trades.items():
        t.to_csv(os.path.join(sub_dir, f"trades_{n}.csv"), index=False)
    if not nulls.empty:
        nulls.to_csv(os.path.join(sub_dir, "null_draws.csv"), index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows.values()]).to_csv(
        os.path.join(sub_dir, "arms.csv"), index=False)
    for n, pr in ports.items():
        if pr.get("daily") is not None:
            pr["daily"].to_csv(os.path.join(sub_dir, f"portfolio_{n}.csv"))
    if len(minute.get("rows", [])):
        minute["rows"].to_csv(os.path.join(sub_dir, "minute_check.csv"), index=False)
    if len(pinfo.get("rows", [])):
        pinfo["rows"].to_csv(os.path.join(sub_dir, "p_minute_events.csv"), index=False)
    S.loc[S["valid"]].to_csv(os.path.join(sub_dir, "setups_valid.csv"), index=False)

    print(f"\n=== VCP 재시험 — {elapsed / 60:.1f}분 · 하한 {floor:,.0f} · 원장 {n_led}줄 · DSR N {n_dsr} ===")
    print("시간(s): " + " · ".join(f"{k} {v:.1f}" for k, v in timings.items()))
    print(f"셋업 후보 {funnel['base']:,} · VCP {funnel['valid']:,} (하락 계단 제외 {funnel['lower_low_excl']:,}) · E 가속 "
          f"{funnel['e_accel_valid']:,} · 수급 기간 {funnel['flow_period_valid']:,}")
    print(f"분봉 p: 모호 사건 {pinfo['n_events']:,} · 판별 {pinfo['n_classified']:,} · 확정 손절 {pinfo.get('n_confirmed', 0):,}")
    for n, s in rows.items():
        line = f"팔 {n}: 셋업 행 {s['n_setups']:,} · 체결 {s['n_fills']:,} (모호 {s.get('amb_n', 0)}, NXT 절단 {s.get('nxt_cut', 0)})"
        if n in NULL_ARMS:
            line += f" · 널 {s.get('null_k', 0)}벌 · 정합 짝 {s.get('mom_pairs', 0)}"
        line += f" · 포트폴리오 편입 {ports[n]['stats'].get('port_taken', 0)}"
        if not args.counts_only:
            line += (f" · p {pval:.3f} · 순평균 2배 {_pct(s.get('net2_mean', np.nan))} · 널 p95 "
                     f"{_pct(s.get('null_p95', np.nan))} · 정합 α {_pct(s.get('mom_alpha', np.nan))}")
        print(line)
    print(f"분봉 검증(전략 체결 2025-08+) {minute['summary'].get('n', 0)} · 분봉 있음 {minute['summary'].get('n_minute', 0)}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def _nxt_checks(ctx: Ctx) -> dict:
    """정정 M6·B-M3 점검 한 줄씩: (1) 2026-09-14+ 정지일(거래량 0)인데 종가가 전일과 다른 행(NXT 이월 가짜 갭), (2) 그 구간 정규장 분봉에
    15:30 봉이 없는 (종목, 날짜) 수."""
    p = ctx.p
    late = p.close.index >= pd.Timestamp(NXT_START)
    cl, vol = p.close.to_numpy(float), p.volume.to_numpy(float)
    prev = np.vstack([np.full((1, cl.shape[1]), np.nan), cl[:-1]])
    with np.errstate(invalid="ignore"):
        fake = late[:, None] & (vol == 0) & np.isfinite(prev) & (np.abs(cl / prev - 1) > 0.005)
    q = ("SELECT count(*) FROM (SELECT code, ts::date AS d, max(ts::time) AS m FROM minute_bars WHERE ts >= %(lo)s "
         "AND ts::time BETWEEN %(o)s AND %(c)s GROUP BY 1, 2) x WHERE m < '15:30:00'")
    try:
        no1530 = int(pd.read_sql_query(q, ctx.con, params={"lo": NXT_START, "o": SESSION_OPEN, "c": SESSION_CLOSE}).iloc[0, 0])
    except Exception:  # noqa: BLE001 — 점검 줄이라 실패해도 본 계산은 계속
        no1530 = -1
    return {"halt_fake_gap": int(fake.sum()), "no_1530_bar": no1530}


# ---------------------------------------------------------------------------
# 단계 함수
# ---------------------------------------------------------------------------

def _month_rows(days: pd.DatetimeIndex) -> dict:
    per = np.asarray(days.to_period("M").astype(str))
    out = {}
    for m in pd.unique(per):
        idx = np.flatnonzero(per == m)
        out[m] = (int(idx[0]), int(idx[-1]))
    return out


def _finish(t: pd.DataFrame, cost_tab: np.ndarray, mkt: np.ndarray, cut: "np.ndarray | None" = None) -> pd.DataFrame:
    """비용(낙관 경로 청산일 · 보수 경로 = 진입일)·순수익·포트폴리오 열 + NXT 절단 표시(``nxt_cut``)."""
    if t.empty:
        return t
    for col in ("event_day", "entry_day", "exit_day", "order_day"):
        t[col] = pd.to_datetime(t[col])
    ci = t["col_idx"].to_numpy(int)
    m = mkt[ci]
    t["cost"] = cost_tab[m, t["exit_row"].to_numpy(int)]
    amb = t["amb"].to_numpy(bool)
    t["cost_stop0"] = np.where(amb, cost_tab[m, t["entry_row"].to_numpy(int)], t["cost"])
    t["net"] = t["ret"] - t["cost"]
    t["net2"] = t["ret"] - 2 * t["cost"]
    t["net_stop0"] = t["ret_stop0"] - t["cost_stop0"]
    t["net2_stop0"] = t["ret_stop0"] - 2 * t["cost_stop0"]
    t["ci"] = ci
    t["d_row"] = t["entry_row"].astype(int)
    t["x_row"] = t["exit_row"].astype(int)
    t["ext_ma50"] = t["entry_px"] / t["ma50_s"] - 1.0
    t["vi_zone"] = t["pivot"] / t["close_s"] >= 1.08
    t["nxt_cut"] = ((t["exit_reason"] == "data_end").to_numpy() & (t["x_row"].to_numpy() < cut[ci])) if cut is not None else False
    t.index = _key(t["event_day"], t["code"])
    return t


def _trade_desc(t: pd.DataFrame) -> dict:
    """경로와 무관한 트레이드 기술(d 분포·확장률·갭·모호·VI·검열·NXT 청산)."""
    if t is None or t.empty:
        return {"amb_n": 0}
    d = t["d"]
    return {"amb_n": int(t["amb"].sum()), "amb_rate": float(t["amb"].mean()),
            "d_q": {q: float(d.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "d_at_cap": float((d >= 0.0999).mean()), "d_over10": float((d > 0.1 + 1e-12).mean()),
            "d_floor": float((d <= 0.02 + 1e-12).mean()), "ext_mean": float(t["ext_ma50"].mean()),
            "ext_median": float(t["ext_ma50"].median()), "gap_rate": float(t["gap_fill"].mean()),
            "vi_rate": float(t["vi_zone"].mean()), "censored": int((t["exit_reason"] == "data_end").sum()),
            "exit_after_nxt": int((t["exit_day"] >= pd.Timestamp(NXT_START)).sum()),
            "nxt_cut": int(t["nxt_cut"].sum()) if "nxt_cut" in t else 0,
            "h0_wick_n": int(t["h0_wick_flag"].sum()) if "h0_wick_flag" in t else 0,
            "base_len_q": {q: float(t["base_len"].quantile(q)) for q in (0.0, 0.5, 1.0)} if "base_len" in t else {},
            "pivot_inprogress_rate": float(t["pivot_inprogress"].mean()) if "pivot_inprogress" in t else np.nan,
            "e_accel_eps_n": int(t["e_accel_eps"].sum()) if "e_accel_eps" in t else 0}


def _eps_flags(con, sv: pd.DataFrame) -> np.ndarray:
    """주당 기준 E 가속(기술, 정정 M2) — ``shares_outstanding_history`` as-of 네 시점."""
    codes = [str(c) for c in pd.unique(sv["code"])]
    sh = pd.read_sql_query("SELECT code, date, shares_outstanding FROM shares_outstanding_history WHERE code = ANY(%(c)s)",
                           con, params={"c": codes})
    if sh.empty:
        return np.zeros(len(sv), bool)
    sh["code"] = sh["code"].astype(str)
    sh["date"] = pd.to_datetime(sh["date"].astype(str)).astype("datetime64[ns]")
    sh["shares_outstanding"] = sh["shares_outstanding"].astype(float)
    sh = sh.sort_values("date")
    cc = sv["code"].astype(str).to_numpy()

    def asof(dates) -> np.ndarray:
        q = pd.DataFrame({"code": cc, "date": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]").to_numpy(),
                          "_i": np.arange(len(sv))}).dropna(subset=["date"])
        out = np.full(len(sv), np.nan)
        if q.empty:
            return out
        m = pd.merge_asof(q.sort_values("date"), sh, on="date", by="code", direction="backward")
        out[m["_i"].to_numpy()] = m["shares_outstanding"].to_numpy(float)
        return out

    yr = pd.Timedelta(days=365)
    ea = pd.to_datetime(pd.Series(sv["e_avail"].to_numpy()))
    pa = pd.to_datetime(pd.Series(sv["e_prev_avail"].to_numpy()))
    return V.eps_accel(sv.reset_index(drop=True), asof(ea), asof(ea - yr), asof(pa), asof(pa - yr))


def _estimate_p(con, A: V.Arrays, h5: pd.DataFrame, days: pd.DatetimeIndex, p_hi: "str | None", *,
                band_lo: float = V.VCP_PARAMS["close_lo"], min_age: int = V.VCP_PARAMS["dry_min"],
                params: "dict | None" = None, levels_fn=None) -> dict:
    """정정 C3·정정 2 B-M1 — 분봉 구간(2025-08+) 유니버스 합성 매수정지 돌파의 **모호** 사건(시뮬과 같은 식: 시가 ≤ 피벗 ∧ 저가 ≤ 손절가 ∧
    종가 > 손절가 — :func:`vcp.buystop_entry_check`)에서 손절가 터치가 체결 봉 **뒤**에 온 비율 p. 진입일 종가 ≤ 손절가(확정 손절)는
    분모에 넣지 않고 따로 센다. 합성 = H5 종목, 피벗 = 직전 k일 최고 고가(피벗 뒤 ≥ 3일), 손절 = 직전 j일 최저 저가 − 1호가(2% 하한),
    (k, j) = :data:`P_GRID` — 종목-일당 격자 순서로 처음 모호해지는 조합 하나. 근접 띠·9.5%·3% 갭·상한가·정지·NXT 절단 같음.
    같은 분봉 안에서만 닿으면 0.5. 분봉이 도달·터치를 못 보이면 뺀다. 95% CI = 날짜군집 부트스트랩 2,000."""
    lo = pd.Timestamp(MINUTE_LO)
    hi_ = pd.Timestamp(p_hi) if p_hi else days[-1]
    drows = np.flatnonzero((days >= lo) & (days <= hi_))
    hm = h5.to_numpy(bool)
    ev = []
    stage = {"cand": 0, "band": 0, "fill_open_le_pivot": 0, "confirmed": 0, "amb": 0}
    for d in drows:
        s = d - 1
        cs = np.flatnonzero(hm[s])
        if not len(cs):
            continue
        stage["cand"] += len(cs)
        seen = np.zeros(len(cs), bool)
        conf_seen = np.zeros(len(cs), bool)
        for k, j in P_GRID:
            if levels_fn is None:
                piv, ip, ll = V.null_levels(A, cs, np.full(len(cs), s), np.full(len(cs), k), np.full(len(cs), j))
                cl = A.C[cs, s]
                with np.errstate(invalid="ignore"):
                    band = (cl < piv) & (cl >= band_lo * piv) & (ip <= s - min_age)
            else:                                      # 다른 시험의 합성 사건 정의(예: 쐐기 D 바닥 반전) — (기준가, 손절 기준, 통과)
                piv, ll, band = levels_fn(A, cs, np.full(len(cs), s), np.full(len(cs), k), np.full(len(cs), j))
            chk = V.buystop_entry_check(A, cs, np.full(len(cs), d), piv, ll, params)
            ok = band & (chk["status"] == V.S_OK)
            stage["band"] += int(band.sum())
            stage["fill_open_le_pivot"] += int((ok & chk["open_le_pivot"]).sum())
            conf_seen |= ok & chk["stop0_confirmed"]
            amb = ok & chk["amb"] & ~seen
            if not amb.any():
                continue
            seen |= amb
            cc, pv = cs[amb], piv[amb]
            e, stop = chk["entry"][amb], chk["stop"][amb]
            ev.append(pd.DataFrame({"ci": cc, "d": d, "k": k, "j": j, "order": pv + tick_size_vec(pv), "stop": stop,
                                    "dd": (e - stop) / e, "bull": A.C[cc, d] >= A.O[cc, d]}))
        stage["confirmed"] += int((conf_seen & ~seen).sum())
        stage["amb"] += int(seen.sum())
    base = {"window": f"{lo.date()}~{hi_.date()}", "stage": stage, "n_confirmed": stage["confirmed"]}
    if not ev:
        return {**base, "p": 0.5, "n_events": 0, "n_classified": 0, "counts": {}, "strata": {}, "rows": pd.DataFrame(),
                "fallback": True, "ci": (np.nan, np.nan), "v": np.zeros(0), "dd": np.zeros(0)}
    E = pd.concat(ev, ignore_index=True)
    E["code"] = A.codes[E["ci"].to_numpy()]
    E["day"] = days[E["d"].to_numpy()]
    q = ("SELECT code, ts, open, high, low FROM minute_bars WHERE ts >= %(lo)s AND ts < %(hi)s "
         "AND code = ANY(%(codes)s) AND ts::time BETWEEN %(o)s AND %(c)s ORDER BY code, ts")
    cls = np.full(len(E), "", dtype=object)
    for day, g in E.groupby("day"):
        m = pd.read_sql_query(q, con, params={"lo": day.strftime("%Y-%m-%d"),
                                              "hi": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                              "codes": [str(c) for c in g["code"].unique()],
                                              "o": SESSION_OPEN, "c": SESSION_CLOSE})
        bym = {str(c): b for c, b in m.groupby("code")}
        for i, r in g.iterrows():
            b = bym.get(str(r["code"]))
            if b is None or not len(b) or not float(b["open"].iloc[0]) > 0:
                cls[i] = "no_minute"
                continue
            f = A.O[int(r["ci"]), int(r["d"])] / float(b["open"].iloc[0])
            hi_m, lo_m = b["high"].to_numpy(float) * f, b["low"].to_numpy(float) * f
            touch = np.flatnonzero(hi_m >= r["order"] - 1e-9)
            hit = np.flatnonzero(lo_m <= r["stop"] + 1e-9)
            if not len(touch):
                cls[i] = "no_touch"
                continue
            k = touch[0]
            cls[i] = ("after" if (hit > k).any() else "same" if (hit == k).any() else "before" if (hit < k).any()
                      else "no_stop_minute")
    E["cls"] = cls
    c = {str(k): int(v) for k, v in E["cls"].value_counts().items()}
    okc = E["cls"].isin(["after", "same", "before"]).to_numpy()
    v = E.loc[okc, "cls"].map({"after": 1.0, "same": 0.5, "before": 0.0}).to_numpy(float)
    den = int(okc.sum())
    p = float(v.mean()) if den else 0.5
    ci = (np.nan, np.nan)
    if den >= 2:
        lo_, hi2, _ = cluster_bootstrap_ci(v, E.loc[okc, "day"].dt.strftime("%Y%m%d").to_numpy(), n_boot=N_BOOT, seed=SEED)
        ci = (lo_, hi2)
    strata, edges = {}, None
    if den >= 3:
        E2 = E[okc].copy()
        E2["v"] = v
        edges = np.quantile(E2["dd"], [1 / 3, 2 / 3])
        E2["d3"] = np.searchsorted(edges, E2["dd"], side="right")
        strata = {"양봉(종가 ≥ 시가)": E2.groupby("bull")["v"].agg(["mean", "size"]).to_dict("index"),
                  "d 3분위": E2.groupby("d3")["v"].agg(["mean", "size"]).to_dict("index")}
    return {**base, "p": p, "n_events": int(len(E)), "n_classified": den, "counts": c, "strata": strata, "ci": ci,
            "rows": E.drop(columns=["ci", "d"]), "fallback": den == 0, "v": v, "dd": E.loc[okc, "dd"].to_numpy(float),
            "d_edges": edges}


def _p_reweight(pinfo: dict, t: pd.DataFrame) -> float:
    """정정 2 MINOR 6 — 합성 사건의 d 3분위별 p 를 V0 모호 거래의 d 3분위(같은 경계) 구성으로 재가중(민감도 한 줄). 판정은 주 p."""
    edges = pinfo.get("d_edges")
    if edges is None or t is None or t.empty or not t["amb"].any():
        return np.nan
    b_syn = np.searchsorted(edges, pinfo["dd"], side="right")
    p_b = {k: float(pinfo["v"][b_syn == k].mean()) for k in range(3) if (b_syn == k).any()}
    b_v0 = np.searchsorted(edges, t.loc[t["amb"].to_numpy(bool), "d"].to_numpy(float), side="right")
    w = np.bincount(b_v0, minlength=3).astype(float)
    num = sum(w[k] * p_b[k] for k in p_b)
    den = sum(w[k] for k in p_b)
    return float(num / den) if den > 0 else np.nan


def _momentum_control(t: pd.DataFrame, A: V.Arrays, h5: pd.DataFrame, q_mom: np.ndarray, adv2: np.ndarray,
                      mkt: np.ndarray, days: pd.DatetimeIndex, codes: pd.Index, pval: float,
                      params: "dict | None" = None) -> dict:
    """모멘텀 정합: 트레이드(신호 행 s)마다 같은 날 × 시장 × q_mom × adv2 의 H5 종목 5개(복원, 자기 제외, seed 0) → d 시가 진입·
    같은 손절폭·같은 청산 → ``matched_alpha``(총수익, 날짜군집 2000). 트레이드 값 = p 혼합 총수익(보수·낙관도)."""
    if t is None or t.empty:
        return {}
    gmix = pval * t["ret_stop0"] + (1 - pval) * t["ret"]
    tt = pd.DataFrame({"date": t["event_day"].to_numpy(), "market": t["market"].to_numpy(),
                       "q_mom": q_mom[t["sig_idx"], t["col_idx"]], "adv2": adv2[t["sig_idx"], t["col_idx"]],
                       "ret": gmix.to_numpy(), "ret_cons": t["ret_stop0"].to_numpy(), "ret_opt": t["ret"].to_numpy(),
                       "d": t["d"].to_numpy()}, index=t.index)
    sig_days = np.unique(t["sig_idx"].to_numpy())
    r_, c_ = np.nonzero(h5.to_numpy(bool)[sig_days])
    rr = sig_days[r_]
    cand = pd.DataFrame({"date": days[rr], "market": np.where(mkt[c_] == 1, "KOSDAQ", "KOSPI"), "q_mom": q_mom[rr, c_],
                         "adv2": adv2[rr, c_], "ci": c_, "s": rr},
                        index=_key(pd.Series(days[rr]), pd.Series(np.asarray(codes[c_]))))
    ctl = matched_control(tt, cand, strata=STRATA_MOM, n_per=N_PER, seed=SEED, exclude_self=True)
    unmatched = ctl.attrs.get("unmatched", [])
    if ctl.empty:
        return _alpha("mom_", None)
    cc = cand.loc[ctl["control_idx"].to_numpy()]
    dd = tt.loc[ctl["trade_idx"].to_numpy(), "d"].to_numpy()
    r = V.run_open_entry(A, cc["ci"].to_numpy(), cc["s"].to_numpy() + 1, dd, params)
    ok = r["status"] == V.S_OK
    pool = pd.DataFrame({"ret": r["ret"][ok], "hold": r["x"][ok] - cc["s"].to_numpy()[ok],
                         "reason": np.asarray(V.REASONS, dtype=object)[r["reason"][ok]]})
    pool["ret_cons"] = pool["ret"]
    pool["ret_opt"] = pool["ret"]
    ctl2 = ctl[ok].copy()
    ctl2["control_idx"] = np.arange(int(ok.sum()))
    ctl2.attrs["unmatched"] = unmatched
    out = {}
    for tag, col in (("mom_", "ret"), ("cons_mom_", "ret_cons"), ("opt_mom_", "ret_opt")):
        out.update(_alpha(tag, matched_alpha(tt, pool, ctl2, value=col, cluster="date", n_boot=N_BOOT, seed=SEED)))
    ccut = int(((r["reason"][ok] == V.R_DATA_END) & (r["x"][ok] < A.CUT[cc["ci"].to_numpy()[ok]])).sum())
    out.update({"mom_pairs": int(len(ctl)), "mom_pairs_dropped": int((~ok).sum()), "mom_unmatched_n": len(unmatched),
                "mom_ctl_nxt_cut": ccut,
                "mom_ctl_hold_median": float(pool["hold"].median()) if len(pool) else np.nan,
                "mom_ctl_hold_mean": float(pool["hold"].mean()) if len(pool) else np.nan,
                "mom_ctl_exits": pool["reason"].value_counts().to_dict(),
                "mom_trade_hold_median": float(t["hold_days"].median())})
    return out


def _naver_returns(sym: str) -> pd.Series:
    nv = pd.read_csv(NAVER_CSV, parse_dates=["date"])
    s = nv[nv["symbol"] == sym].set_index("date")["close"].astype(float).sort_index()
    return s.pct_change(fill_method=None).dropna()


def _port_table(t: pd.DataFrame, days: pd.DatetimeIndex, cons: bool) -> pd.DataFrame:
    pt = t[["ci", "d_row", "x_row", "entry_px", "partial_hit", "partial_px", "exit_px", "cost", "adv20"]].copy()
    pt["part_row"] = days.get_indexer(pd.to_datetime(t["partial_day"]))
    if cons:
        a = t["amb"].to_numpy(bool)
        pt.loc[a, "x_row"] = t.loc[a, "d_row"].to_numpy()
        pt.loc[a, "exit_px"] = (t.loc[a, "entry_px"] * (1 + t.loc[a, "ret_stop0"])).to_numpy()
        pt.loc[a, "partial_hit"] = False
        pt.loc[a, "cost"] = t.loc[a, "cost_stop0"].to_numpy()
    return pt


def _portfolio(t: pd.DataFrame, A: V.Arrays, days: pd.DatetimeIndex, lo: str, hi: str, F, kospi: pd.Series,
               pval: float) -> dict:
    """10 슬롯 포트폴리오(보수·낙관 경로 각각, 주 = 날별 수익의 p 혼합) + 같은 기간 코스피 + 팩터 회귀(정정 M4: TVW 시장 열, 원수익,
    현금 0%) + 극단일 제외 민감도."""
    if t is None or t.empty:
        return {"stats": {"port_taken": 0}, "daily": None}
    t_lo = int(days.searchsorted(pd.Timestamp(lo)))
    t_hi = min(int(max(t["x_row"].max(), days.searchsorted(pd.Timestamp(hi), side="right") - 1)), len(days) - 1)
    dates = days[t_lo:t_hi + 1]
    res = {}
    for tag, cons in (("cons", True), ("opt", False)):
        pt = _port_table(t, days, cons)
        r1, info = slot_portfolio(pt, A.C, t_lo, t_hi, 1.0)
        r2, _ = slot_portfolio(pt, A.C, t_lo, t_hi, 2.0)
        res[tag] = (r1, r2, info)
    r1 = pval * res["cons"][0] + (1 - pval) * res["opt"][0]
    r2 = pval * res["cons"][1] + (1 - pval) * res["opt"][1]
    e1, e2 = equity_stats(r1), equity_stats(r2)
    sd = float(np.std(r1, ddof=1)) if len(r1) > 2 else np.nan
    exc = r1 - mg2.cash_daily(dates, m5.CD91_MEM)                 # 초과 Sharpe(현금 CD91 기억값, 기술 — 정정 2)
    sde = float(np.std(exc, ddof=1)) if len(exc) > 2 else np.nan
    out = {"port_taken": res["opt"][2]["taken"], "port_skipped": res["opt"][2]["skipped"],
           "port_exposure": res["opt"][2]["exposure"], "port_days": len(dates),
           "port_cagr": e1["cagr"], "port_mdd": e1["mdd"], "port_cagr2": e2["cagr"], "port_mdd2": e2["mdd"],
           "port_sharpe": float(np.mean(r1) / sd * np.sqrt(252)) if sd and sd > 0 else np.nan,
           "port_sharpe_exc": float(np.mean(exc) / sde * np.sqrt(252)) if sde and sde > 0 else np.nan,
           "port_cagr_cons": equity_stats(res["cons"][0])["cagr"], "port_cagr_opt": equity_stats(res["opt"][0])["cagr"],
           "port_from": str(dates[0].date()), "port_to": str(dates[-1].date())}
    ek = equity_stats(kospi.reindex(dates).dropna().to_numpy())
    out.update({"kospi_cagr": ek["cagr"], "kospi_mdd": ek["mdd"]})
    if F is not None:
        y = pd.Series(r1, dates)
        X = F[FACTOR_COLS].reindex(dates)
        f1 = nw_factor(y, X)
        f2 = nw_factor(pd.Series(r2, dates), X)
        keep = ~dates.isin(pd.to_datetime(list(EXTREME_DAYS)))
        fx = nw_factor(y[keep], X[keep])
        out.update({"fac_alpha_ann": f1.get("alpha_ann", np.nan), "fac_alpha_t": f1.get("alpha_t", np.nan),
                    "fac_r2": f1.get("r2", np.nan), "fac_n": f1.get("n", 0), "fac_betas": f1.get("betas", {}),
                    "fac_alpha_t2": f2.get("alpha_t", np.nan), "fac_alpha_t_noext": fx.get("alpha_t", np.nan),
                    "fac_alpha_ann_noext": fx.get("alpha_ann", np.nan)})
    daily = pd.DataFrame({"ret_1x": r1, "ret_2x": r2, "ret_1x_cons": res["cons"][0], "ret_1x_opt": res["opt"][0]},
                         index=dates)
    return {"stats": out, "daily": daily}


def _sensitivities(t: pd.DataFrame, pval: float, n_dsr: int, s0: dict, p_rw: float = np.nan) -> dict:
    """정정 M6·정정 2 — 완결 거래만(가중 행 단위: 모호 거래의 보수 행은 완결) · 사건 ≤ 2026-04-30 · 2020~21 제외 · H0 윗꼬리 제외
    (기술) · p 재가중. 바 1·2·5 는 부분집합으로 다시(주 p 와 보수·낙관 경계), 바 3·4 문턱은 전체 값 그대로."""
    out = {}
    cells = [("완결 거래만", None, True),
             (f"사건 ≤ {CAP_SENS}", t["order_day"] <= pd.Timestamp(CAP_SENS), False),
             ("2020~21 제외(기술)", ~t["order_day"].dt.year.isin([2020, 2021]), False)]
    if "h0_wick_flag" in t:
        cells.append(("H0 윗꼬리 제외(기술)", ~t["h0_wick_flag"].astype(bool), False))
    for name, mk, complete_only in cells:
        sub = t if mk is None else t[mk.to_numpy()]
        res = {}
        for tag, pp in (("", pval), ("cons_", 1.0), ("opt_", 0.0)):
            x = expand(sub, pp)
            if complete_only and len(x):
                x = x[x["complete"].to_numpy(bool)]
            st = wstats(x, n_dsr)
            b = bar_check(st, s0.get("null_p95", np.nan), s0.get("mom_lo", np.nan), np.nan)
            res.update({f"{tag}{k}": v for k, v in st.items()})
            res.update({f"{tag}bar_{k}": v for k, v in b.items()})
        out[name] = res
    if np.isfinite(p_rw):
        st = wstats(expand(t, p_rw), n_dsr)
        b = bar_check(st, s0.get("null_p95", np.nan), s0.get("mom_lo", np.nan), np.nan)
        out[f"p 재가중 {p_rw:.3f}"] = {**st, **{f"bar_{k}": v for k, v in b.items()}}
    return out


def _pair_compare(a: pd.DataFrame, b: pd.DataFrame, pval: float) -> dict:
    """정정 M5 — 같은 피벗(종목, ``ip``)에서 V0 와 X 칸 둘 다 체결된 쌍: X − V0 의 순수익(1배, p 혼합)·고정 분모 R 차이,
    날짜군집(V0 주문일) 부트스트랩 95% CI."""
    if a.empty or b.empty:
        return {"n": 0}

    def keyed(t):
        x = t.assign(mix=pval * t["net_stop0"] + (1 - pval) * t["net"]).set_index(["code", "ip"])
        return x[~x.index.duplicated()]

    ka, kb = keyed(a), keyed(b)
    common = ka.index.intersection(kb.index)
    if not len(common):
        return {"n": 0, "n_a": int(len(ka)), "n_b": int(len(kb))}
    diff = kb.loc[common, "mix"].to_numpy() - ka.loc[common, "mix"].to_numpy()
    cl = ka.loc[common, "order_day"].dt.strftime("%Y%m%d").to_numpy()
    lo, hi, _ = cluster_bootstrap_ci(diff, cl, n_boot=N_BOOT, seed=SEED)
    return {"n": int(len(common)), "n_a": int(len(ka)), "n_b": int(len(kb)), "diff_mean": float(diff.mean()),
            "diff_lo": lo, "diff_hi": hi, "diff_r_fixed": float(diff.mean() / R_FIXED),
            "a_mean": float(ka.loc[common, "mix"].mean()), "b_mean": float(kb.loc[common, "mix"].mean()),
            "a_d_med": float(ka.loc[common, "d"].median()), "b_d_med": float(kb.loc[common, "d"].median())}


def _flow_terciles(t: pd.DataFrame, pval: float, flow_lo: str) -> dict:
    """정정 M3 — V0 체결 중 수급 기간(2024-04-25+) · F_base 알려진 것의 3분위별 기술."""
    if t.empty:
        return {"n": 0}
    x = t[(t["order_day"] >= pd.Timestamp(flow_lo)) & t["f_base"].notna()]
    if len(x) < 3:
        return {"n": int(len(x))}
    q = pd.qcut(x["f_base"], 3, labels=["하", "중", "상"], duplicates="drop")
    out = {"n": int(len(x))}
    for k, g in x.groupby(q, observed=True):
        s = wstats(expand(g, pval))
        out[str(k)] = {"n": s["n"], "net_mean": s["net_mean"], "loss_rate": s["loss_rate"],
                       "f_lo": float(g["f_base"].min()), "f_hi": float(g["f_base"].max())}
    return out


def _p1_side(lo: str, hi: str) -> dict:
    """같은 기간 위성 P1(라이브 백테스트 B 판 ``trades_b.csv``) 거래 숫자 — 나란히 기술."""
    if not os.path.exists(P1_TRADES):
        return {"n": 0}
    x = pd.read_csv(P1_TRADES, parse_dates=["entry_day", "exit_day"])
    x = x[(x["entry_day"] >= pd.Timestamp(lo)) & (x["entry_day"] <= pd.Timestamp(hi))]
    if x.empty:
        return {"n": 0}
    net = x["net_ret"].to_numpy(float)
    dd = x["d"].to_numpy(float)
    g, l_ = net[net > 0], net[net < 0]
    return {"n": int(len(x)), "net_mean": float(net.mean()), "loss_rate": float((net < 0).mean()),
            "gl_ratio": float(g.mean() / abs(l_.mean())) if len(g) and len(l_) else np.nan,
            "avg_loss_R": float((net[net < 0] / dd[net < 0]).mean()) if len(l_) else np.nan,
            "big_loss": float((net < BIG_LOSS).mean()), "from": str(x["entry_day"].min().date()),
            "to": str(x["entry_day"].max().date())}


def _minute_check(con, trades: dict, A: V.Arrays) -> dict:
    """2025-08+ 전략 매수정지 체결(V0~V3·X2 합집합): 분봉(09:00~15:32, 일봉 시가 비율로 조정 환산)으로 체결가 오차·같은 날 순서(기술)."""
    parts = [t[["code", "order_day", "ci", "d_row", "order_px", "fill_px", "stop_px", "amb"]]
             for n, t in trades.items() if n != "X1" and t is not None and not t.empty]
    if not parts:
        return {"summary": {"n": 0}, "rows": pd.DataFrame()}
    u = pd.concat(parts).drop_duplicates(["code", "order_day"])
    u = u[u["order_day"] >= pd.Timestamp(MINUTE_LO)].reset_index(drop=True)
    rows = []
    q = ("SELECT code, ts, open, high, low FROM minute_bars WHERE ts >= %(lo)s AND ts < %(hi)s "
         "AND code = ANY(%(codes)s) AND ts::time BETWEEN %(o)s AND %(c)s ORDER BY code, ts")
    for day, g in u.groupby("order_day"):
        m = pd.read_sql_query(q, con, params={"lo": day.strftime("%Y-%m-%d"),
                                              "hi": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                              "codes": [str(c) for c in g["code"].unique()],
                                              "o": SESSION_OPEN, "c": SESSION_CLOSE})
        for r in g.itertuples(index=False):
            b = m[m["code"].astype(str) == str(r.code)]
            row = {"code": r.code, "order_day": day, "amb": bool(r.amb), "has_minute": len(b) > 0}
            if len(b) and float(b["open"].iloc[0]) > 0:
                f = A.O[int(r.ci), int(r.d_row)] / float(b["open"].iloc[0])
                hi_m, lo_m, op_m = (b[c].to_numpy(float) * f for c in ("high", "low", "open"))
                touch = np.flatnonzero(hi_m >= r.order_px - 1e-9)
                hit = np.flatnonzero(lo_m <= r.stop_px + 1e-9)
                row["touch"] = len(touch) > 0
                if len(touch):
                    k = int(touch[0])
                    row.update({"fill_err": r.fill_px / max(op_m[k], r.order_px) - 1.0,
                                "stop_after": bool((hit > k).any()), "stop_same": bool((hit == k).any()),
                                "stop_before": bool((hit < k).any())})
            rows.append(row)
    df = pd.DataFrame(rows)
    sm = {"n": int(len(u)), "n_minute": int(df["has_minute"].sum()) if len(df) else 0}
    if len(df) and "fill_err" in df:
        t_ = df[df["fill_err"].notna()]
        fe = t_["fill_err"]
        sm.update({"n_touch": int(len(t_)), "fill_err_mean": float(fe.mean()), "fill_err_median": float(fe.median()),
                   "fill_err_absmax": float(fe.abs().max())})
        am = t_[t_["amb"]]
        sa, ss, sb = (am[c].astype(bool) for c in ("stop_after", "stop_same", "stop_before"))
        sm.update({"amb_n": int(len(am)), "amb_after": int(sa.sum()), "amb_same_only": int((ss & ~sa).sum()),
                   "amb_before_only": int((sb & ~sa & ~ss).sum())})
    return {"summary": sm, "rows": df}


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _mark(ok) -> str:
    return "✓" if ok else "✗"


def _dist(d: dict, order=V.REASONS) -> str:
    keys = [k for k in order if d.get(k, 0)] + [k for k in d if k not in order and d.get(k, 0)]
    return " · ".join(f"{k} {d[k]:g}" for k in keys) or "—"


def _r(x, fmt="{:.2f}") -> str:
    return _f(x, fmt) if x is not None else "nan"


def _sh(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.1%}"


def _bars_lines(s: dict) -> "list[str]":
    g = s.get
    return [f"| 1 표본 | n ≥ {B1_N} ∧ 2017-09~2024-12 n ≥ {B1_SUB_N} | n {g('n')} · 하위 n {g('n_sub')} | {_mark(g('bar_b1'))} |",
            f"| 2 순수익(2배) | 거래당 순수익(2배 비용) > 0 ∧ 하위 구간 > 0 | {_pct(g('net2_mean', np.nan))} · 하위 "
            f"{_pct(g('net2_sub', np.nan))} | {_mark(g('bar_b2'))} |",
            f"| 3 같은 메뉴 널(2배) | 2배 순수익(%) > 널 95 백분위 | {_pct(g('net2_mean', np.nan))} vs p95 {_pct(g('null_p95', np.nan))} "
            f"(널 백분위 {_sh(g('null_rank', np.nan))}) | {_mark(g('bar_b3'))} |",
            f"| 4 모멘텀 정합 | α 날짜군집 95% CI 하단 > 0 | {_pct(g('mom_alpha', np.nan))} [{_pct(g('mom_lo', np.nan))}, "
            f"{_pct(g('mom_hi', np.nan))}] | {_mark(g('bar_b4'))} |",
            f"| 5 손실(1배 판정) | 손실 청산 < 50% ∧ 손익비 ≥ 2 ∧ 평균 손실 ≥ −1.0R ∧ R 5백분위 ≥ −1.3R ∧ 순 < −12% ≤ 5% | "
            f"{_sh(g('loss_rate', np.nan))} · {_r(g('gl_ratio', np.nan))} · {_r(g('avg_loss_R', np.nan), '{:+.2f}')}R · "
            f"{_r(g('r_p5', np.nan), '{:+.2f}')}R · {_sh(g('big_loss', np.nan))} | {_mark(g('bar_b5'))} |",
            f"| 5′ 손실(2배, 기술) | 같은 문턱 | {_sh(g('loss_rate2', np.nan))} · {_r(g('gl_ratio2', np.nan))} · "
            f"{_r(g('avg_loss_R2', np.nan), '{:+.2f}')}R · {_r(g('r_p5_2', np.nan), '{:+.2f}')}R · {_sh(g('big_loss2', np.nan))} | "
            f"{_mark(g('bar_b5_2x'))} |",
            f"| 산술 충족 | 1~5 | **{g('bar_n_met')}/5** | |"]


def render(rows, funnel, sens, pairs, f_terc, p1, pinfo, minute, ports, lo_of, hi, n_led, n_prior, n_dsr, out_dir,
           debug, elapsed, timings, patch, p, args, fac_sha) -> list[str]:
    pval = pinfo["p"]
    v = rows[PRIMARY]
    fl = funnel["floor"]
    void = funnel.get("floor_rule_void")
    L = [f"# VCP 재시험 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "사전등록 `VCP_RETEST.md`(5632879 + 정정 7dbbd0c + 정정 2 aae4b8e, 모두 결과 보기 전)의 정의·팔·대조·널·합격 바를 그대로 계산한 "
         "숫자다. 합격 여부는 여기 적지 않는다 — ✓/✗ 는 사전등록 문턱과의 **산술 비교**일 뿐이다. 판정 팔 = **V0**(진입 역학), V1·V2·V3 = "
         "기술 전용(표본 한 자릿수 — 통계 판정 불가를 미리 적었다), X1·X2(·X2L 문구판) = 진입 방식 분리. 재현: `uv run --with numba "
         "--extra pg python research/experiments/vcp_retest_run.py --workers N`.", "",
         f"**유니버스 하한:** ADV20 ≥ {fl:,.0f} 백만원 — " + (
             "`FUNNEL.md` 사전 규칙(V0 체결 ≥ 60 ∧ 하위 ≥ 30 인 가장 높은 하한) 값." if void is False else
             "**사전 규칙상 V0 VOID**(넷 다 미달) — 아래 숫자는 기술만." if void is True else "명시 인자(`--adv-floor`)."), ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 구간 {min(lo_of.values())}~{hi}, 널 {args.n_null}벌, p 구간 {pinfo.get('window', '—')}, "
              f"출력·원장 `{out_dir}`. 측정으로 쓰지 않는다.", ""]

    L += [f"## 판정 팔 V0 — 합격 바 (주 = 분봉 p 혼합, p = {pval:.3f})", "",
          "| 바 | 문턱 | V0 | 산술 |", "|---|---|---|---|"] + _bars_lines(v)
    same = all(v.get(f"bar_b{i}") == v.get(f"cons_bar_b{i}") == v.get(f"opt_bar_b{i}") for i in range(1, 6))
    L += ["", f"**체결 가정 경계:** 보수(p = 1) {v.get('cons_bar_n_met')}/5 · 낙관(p = 0) {v.get('opt_bar_n_met')}/5 · 주 "
          f"{v.get('bar_n_met')}/5 → 바별 산술이 {'세 가정에서 같다(강건)' if same else '가정 사이에 갈린다(**가정 의존**)'}. "
          "혼합은 '낙관 경로의 거래 집합' 위의 혼합이다(보유 중 거르기가 낙관 경로 청산일로 걸린다 — 모호 거래가 막은 셋업 "
          f"{funnel.get('v0_amb_overlap', 0)}건).", "",
          "| 바 | 보수(p=1) | 낙관(p=0) |", "|---|---|---|"]
    for i, nm in enumerate(("표본", "순수익", "널", "정합", "손실"), start=1):
        L.append(f"| {i} {nm} | {_mark(v.get(f'cons_bar_b{i}'))} | {_mark(v.get(f'opt_bar_b{i}'))} |")
    L += ["", f"바 6(기술): 10 슬롯 포트폴리오 팩터 α HAC t = {_r(v.get('fac_alpha_t', np.nan))} (α "
          f"{_pct(v.get('fac_alpha_ann', np.nan))}/년, 문턱 t > 2 {_mark(v.get('bar_b6'))}) · 평균 노출 "
          f"{_sh(v.get('port_exposure', np.nan))} — 기대 노출이 낮아 검정력이 거의 없음을 미리 적었다.", ""]

    L += ["### 민감도 (정정 M6·정정 2 — 바 1·2·5 는 부분집합으로 다시, 바 3·4 문턱 값은 전체 그대로)", "",
          "| 칸 | n | 하위 n | 2배 순 | 하위 2배 | 손실 청산 | 손익비 | 평균 손실 R | R 5% | 순<−12% | 바 1·2·3·5 (주) | 충족 주/보수/낙관 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sens.items():
        if not s.get("n"):
            L.append(f"| {name} | 0 |" + " |" * 10)
            continue
        L.append(f"| {name} | {s['n']} | {s['n_sub']} | {_pct(s['net2_mean'])} | {_pct(s['net2_sub'])} | {_sh(s['loss_rate'])} | "
                 f"{_r(s['gl_ratio'])} | {_r(s['avg_loss_R'], '{:+.2f}')} | {_r(s['r_p5'], '{:+.2f}')} | {_sh(s['big_loss'])} | "
                 f"{_mark(s['bar_b1'])}{_mark(s['bar_b2'])}{_mark(s['bar_b3'])}{_mark(s['bar_b5'])} | "
                 f"{s.get('bar_n_met', '—')}/{s.get('cons_bar_n_met', '—')}/{s.get('opt_bar_n_met', '—')} |")
    L += [f"| 팩터 극단일({', '.join(EXTREME_DAYS)}) 제외 | — | | | | | | | | | 바 6 t {_r(v.get('fac_alpha_t_noext', np.nan))} | |", ""]

    c = pinfo.get("counts", {})
    ci = pinfo.get("ci", (np.nan, np.nan))
    L += ["## 진입일 손절 순서 — 분봉 p (정정 C3 · 정정 2 B-M1)", "",
          f"구간 {pinfo.get('window', '—')}, H5 전 종목 합성 매수정지(피벗 = 직전 k일 최고 고가·피벗 뒤 ≥ 3일, 손절 = 직전 j일 최저 저가 "
          f"− 1호가·2% 하한, (k, j) ∈ {{10,15,…,40}} × {{5,10,15}} 중 종목-일당 첫 모호 조합 하나, 근접 띠·9.5%·3% 갭·상한가·정지 같음). "
          f"진입일 종가 ≤ 손절가(확정 손절) 종목-일 **{pinfo.get('n_confirmed', 0):,}** 는 분모 밖. 모호 사건(시가 ≤ 피벗 ∧ 저가 ≤ 손절가 ∧ "
          f"종가 > 손절가 — 시뮬과 같은 식) **{pinfo['n_events']:,}**: 체결 봉 뒤 손절 터치 {c.get('after', 0):,} · 같은 봉만 "
          f"{c.get('same', 0):,}(0.5) · 체결 전만 {c.get('before', 0):,} · 분봉 미도달 {c.get('no_touch', 0):,} · 분봉 손절 미터치 "
          f"{c.get('no_stop_minute', 0):,} · 분봉 없음 {c.get('no_minute', 0):,} → **p = {pval:.3f}** (판별 n {pinfo['n_classified']:,}, "
          f"날짜군집 95% CI [{_r(ci[0], '{:.3f}')}, {_r(ci[1], '{:.3f}')}])"
          f"{' (판별 사건 없음 — 0.5 대체)' if pinfo.get('fallback') else ''}. V0 모호 거래의 d 3분위 구성으로 재가중한 p = "
          f"{_r(pinfo.get('p_reweighted', np.nan), '{:.3f}')}(민감도). 단계 건수: {pinfo.get('stage', {})}.", ""]
    for k, d in pinfo.get("strata", {}).items():
        L.append(f"- {k}: " + " · ".join(f"{kk} p {vv['mean']:.3f} (n {vv['size']})" for kk, vv in d.items()))
    L.append("")

    st = funnel["status"]
    nc = funnel.get("nxt_checks", {})
    L += ["## 표본·깔때기", "",
          f"- 가격 패널 {p.days[0].date()}~{p.days[-1].date()} · 실행 {elapsed / 60:.1f}분 (" +
          " · ".join(f"{k} {v_:.0f}s" for k, v_ in timings.items()) + ")",
          "- 사건(주문일) 구간: " + " · ".join(f"{n} {lo_of[n]}~{hi}" for n in ARMS),
          f"- 원장 `{out_dir}/{LABEL}/TRIALS.jsonl` **{n_led}**줄(팔 {len(ARMS)} + 민감도 {len(SENS_CELLS)}) · DSR N(기술, 이 계열 원장 전부) "
          f"= {n_led} + " + " + ".join(f"{k} {v_}" for k, v_ in n_prior.items()) + f" = **{n_dsr}** · V0 DSR 확률 "
          f"{_r(v.get('dsr_prob', np.nan), '{:.3f}')}",
          "- NXT 보정: " + " · ".join(f"{s_['table']}({s_['cols']}) 늦은 행 {s_['rows_late']:,} → 보정 {s_['patched']:,}·"
                                     f"미보정 {s_['unpatched']:,}" for s_ in patch.stats),
          f"- NXT 절단(정정 2 B-M3): 미보정 행이 있는 종목 {funnel.get('n_cut_stocks', 0):,} — 보유 경로를 {NXT_LAST_DAY} 에서 `data_end` 로 끊음. "
          "끊긴 건수: 트레이드 " + " · ".join(f"{n} {rows[n].get('nxt_cut', 0)}" for n in ARMS)
          + " · 널(벌 중앙) " + " · ".join(f"{n} {_r(rows[n].get('null_cut_med', np.nan), '{:.0f}')}" for n in NULL_ARMS)
          + " · 대조 " + " · ".join(f"{n} {rows[n].get('mom_ctl_nxt_cut', 0)}" for n in NULL_ARMS),
          f"- NXT 점검: 2026-09-14+ 정지일인데 종가가 전일과 다른 행(이월 가짜 갭) {nc.get('halt_fake_gap', '—')} · 정규장 분봉에 15:30 봉이 "
          f"없는 (종목, 날짜) {nc.get('no_1530_bar', '—')}",
          f"- factors.pkl sha256 `{fac_sha}` · factors.py sha256 `{_sha(FACTORS_PY)}` (`{FACTORS_PKL}`)", "",
          f"셋업 후보(신호 행 s: H5 ∧ 추세 ∧ RS ≥ 70 ∧ breadth > 0.5) {funnel['base_nosplit']:,} − 분할 직후 60일 {funnel['split_excl']:,} "
          f"(조정계수 변화 {funnel['split_events']:,}건 중 수집 배치일 {funnel['split_batch_days']} 제외 → 개별 {funnel['split_real']:,}건; "
          f"배치 이전 실제 분할은 못 잡음) = **{funnel['base']:,}** → VCP 첫 탈락: "
          + " · ".join(f"{k} {v_:,}" for k, v_ in sorted(st.items(), key=lambda kv: V.VCP_STATUS.index(kv[0]))),
          f"→ VCP 통과(거래량 창 [ip−49, s] 개별 분할 {funnel['split_window_excl']:,} 제외 뒤) **{funnel['valid']:,}** (하락 계단 조건이 지운 "
          f"셋업 {funnel['lower_low_excl']:,} · 피벗 = 진행 중 고점 {funnel['pivot_inprogress']:,} · H0 윗꼬리 ≥ {H0_WICK:.0%} "
          f"{funnel['h0_wick_valid']:,}) · E 알려짐 {funnel['e_known_valid']:,} · E 가속 {funnel['e_accel_valid']:,} (주당 기준 "
          f"{funnel['e_accel_eps_valid']:,}) · 수급 기간 VCP {funnel['flow_period_valid']:,} → F_base 알려짐 {funnel['flow_known_valid']:,} · "
          f"> 0 {funnel['flow_pos_valid']:,} (수량 합 > 0 {funnel['flow_qty_pos_valid']:,}; 모름 기반 길이별 {funnel['flow_unknown_by_base']}) · "
          "널 풀 행 " + " · ".join(f"{k} {v_:,}" for k, v_ in funnel.get("null_pool_rows", {}).items()), "",
          "| 팔 | 정의 | 셋업 행 | 체결 | 기반 길이 최소/중앙/최대 | H0 윗꼬리 | NXT 절단 | 주문·체결 결과 |", "|---|---|---|---|---|---|---|---|"]
    for n in ARMS:
        r = rows[n]
        sk = r.get("skips", {})
        bq = r.get("base_len_q", {})
        extra = f" (종가 돌파 신호 {funnel['x1_signals']:,})" if n == "X1" else ""
        L.append(f"| {n} | {ARMS[n]['desc']} | {funnel['arms'][n]:,}{extra} | {r['n_fills']:,} | "
                 + ("/".join(f"{bq[q]:.0f}" for q in (0.0, 0.5, 1.0)) if bq else "—")
                 + f" | {r.get('h0_wick_n', 0)} | {r.get('nxt_cut', 0)} | "
                 + (" · ".join(f"{k} {v_:,}" for k, v_ in sorted(sk.items())) or "—") + " |")
    L.append("")

    L += ["## 전 팔 — 거래 숫자 (주 = p 혼합, 보수·낙관은 열로)", "",
          "| 팔 | n | 승/본전/패 | 손실 청산 | stop | 진입일 손절 | 평균 이익 | 평균 손실 | 손익비 | 평균 손실 R | R 5% | 순<−12% | "
          "순 1배 | 보수 1배 | 낙관 1배 | 순 2배 | 보수 2배 | 낙관 2배 | 거래별 d 기준 R 평균 | R 중앙 | 보유 중앙 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in rows.items():
        if not s.get("n"):
            L.append(f"| {n} | 0 |" + " |" * 19)
            continue
        L.append(f"| {n} | {s['n']} | {_sh(s['win'])}/{_sh(s['be'])}/{_sh(s['lossc'])} | {_sh(s['loss_rate'])} | "
                 f"{_sh(s['stop_rate'])} | {_sh(s['stop_entry_rate'])} | {_pct(s['avg_gain'])} | {_pct(s['avg_loss'])} | "
                 f"{_r(s['gl_ratio'])} | {_r(s['avg_loss_R'], '{:+.2f}')} | {_r(s['r_p5'], '{:+.2f}')} | {_sh(s['big_loss'])} | "
                 f"{_pct(s['net_mean'])} | {_pct(s.get('cons_net_mean', np.nan))} | {_pct(s.get('opt_net_mean', np.nan))} | "
                 f"{_pct(s['net2_mean'])} | {_pct(s.get('cons_net2_mean', np.nan))} | {_pct(s.get('opt_net2_mean', np.nan))} | "
                 f"{_r(s['r_mean'], '{:+.3f}')} | {_r(s['r_median'], '{:+.3f}')} | {s['hold_median']:.0f} |")
    L += ["", "| 팔 | 청산 사유(가중 건) | 사유별 순 1배 | d 10/25/50/75/90 | d ≥ 9.99% | d > 10% | d = 2% | MA50 확장 평균/중앙 | "
          "갭 체결 | 모호 | 피벗/전일 종가 ≥ 1.08 | 검열(data_end) | 09-14+ 청산 | 2020~21 제외 2배 | 2025 제외 2배 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in rows.items():
        if not s.get("n"):
            continue
        dq = s["d_q"]
        L.append(f"| {n} | {_dist(s['exits'])} | " + " · ".join(f"{k} {_pct(v_)}" for k, v_ in s["exit_net"].items())
                 + " | " + "/".join(f"{dq[q] * 100:.1f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9))
                 + f"% | {_sh(s['d_at_cap'])} | {_sh(s['d_over10'])} | {_sh(s['d_floor'])} | {_pct(s['ext_mean'])} / "
                 f"{_pct(s['ext_median'])} | {_sh(s['gap_rate'])} | {s['amb_n']} ({_sh(s['amb_rate'])}) | {_sh(s['vi_rate'])} | "
                 f"{s['censored']} | {s['exit_after_nxt']} | {_pct(s['net2_ex2021'])} | {_pct(s['net2_ex2025'])} |")
    L += ["", "앞선 시험과의 비교: \"손절폭 93~98% 가 상한 10%\" → `d ≥ 9.99%` 열, \"진입 시 MA50 위 평균 18%\" → `MA50 확장` 열"
          "(진입가 / 신호 행 MA50 − 1). '거래별 d 기준 R' = 순 1배 / 그 거래의 d(고정 분모 R = % ÷ 0.07 은 % 와 같은 비교라 따로 적지 않는다).",
          "", "**연도별(주문 연도) 거래당 순수익 1배 / 2배 (가중 건):**", ""]
    yrs = sorted({y for s in rows.values() for y in s.get("years_n", {})})
    L += ["| 팔 | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + "---|" * len(yrs)]
    for n, s in rows.items():
        if s.get("n"):
            L.append(f"| {n} | " + " | ".join(
                f"{_pct(s['years_net'].get(y, np.nan))} / {_pct(s['years_net2'].get(y, np.nan))} ({s['years_n'].get(y, 0):g})"
                if y in s["years_n"] else "—" for y in yrs) + " |")
    v1 = rows["V1"]
    v1_txt = (f"n ≥ {V1_DESC_N} 이라 바 숫자를 기술로 적는다: {v1.get('bar_n_met')}/5" if v1.get("n", 0) >= V1_DESC_N
              else f"n < {V1_DESC_N} — 바 숫자 없음(통계 판정 불가, 사전 명시)")
    L += ["", f"V1(기술): n {v1.get('n', 0)} — {v1_txt} · 주당 기준 E 가속 통과 체결 {v1.get('e_accel_eps_n', 0)}.", ""]

    L += ["## 같은 메뉴 무작위 셋업 널 · 모멘텀 정합 대조", "",
          f"널(정정 M1): 같은 날 추세 ∧ RS ∧ breadth ∧ H5(분할 제외) **VCP 아닌** 종목-일(E 팔은 E 가속 통과, 수급 팔은 그 셋업의 k일 창 "
          f"거래대금 가중 F > 0), 피벗 = 직전 k일 최고 고가(k~U{{{NULL_K[0]}..{NULL_K[1]}}}, 피벗 뒤 ≥ 3일), 손절 = 직전 j일 최저 저가 − 1호가"
          f"(j~U{{{NULL_J[0]}..{NULL_J[1]}}}), 근접 띠·9.5%·2% 하한·3% 갭·상한가·정지·NXT 절단 같음. V 팔의 **월별 체결 수 × d 3분위**만큼 "
          f"체결되는 널 사건을 비복원 추출(풀 < {NULL_POOL_MULT}×필요면 그 달 ±{NULL_EXPAND} 거래일), 같은 청산·p 혼합·겹침 규칙. 비교 = 벌마다 "
          "2배 순평균(%, 주). 거래별 d 기준 R 은 기술.", "",
          "| 팔 | 실제 2배 | 널 벌 | 널 중앙 | 널 p95 (보수/낙관) | 실제의 널 백분위 | 실제 R 평균 · 널 R p95(거래별 d 기준) | 벌 체결/필요 중앙 | "
          "부족(중앙) | 풀/필요 최소·중앙 | 정합 α [95% CI] | 보수 α 하단 | 낙관 α 하단 | 짝(버림)·미정합 | 대조 보유 중앙/평균 | 트레이드 보유 중앙 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in NULL_ARMS:
        s = rows[n]
        if not s.get("n"):
            L.append(f"| {n} | — |" + " |" * 14)
            continue
        L.append(f"| {n} | {_pct(s['net2_mean'])} | {s.get('null_k', 0)} | {_pct(s.get('null_p50', np.nan))} | "
                 f"{_pct(s.get('null_p95', np.nan))} ({_pct(s.get('null_p95_cons', np.nan))}/{_pct(s.get('null_p95_opt', np.nan))}) | "
                 f"{_sh(s.get('null_rank', np.nan))} | {_r(s.get('r_mean', np.nan), '{:+.3f}')} · "
                 f"{_r(s.get('null_r_p95', np.nan), '{:+.3f}')} | {_sh(s.get('null_fill_ratio_med', np.nan))} | "
                 f"{_r(s.get('null_short_med', np.nan), '{:.0f}')} | "
                 f"{_r(s.get('null_pool_min', np.nan), '{:.1f}')}·{_r(s.get('null_pool_med', np.nan), '{:.1f}')} | "
                 f"{_pct(s.get('mom_alpha', np.nan))} [{_pct(s.get('mom_lo', np.nan))}, {_pct(s.get('mom_hi', np.nan))}] | "
                 f"{_pct(s.get('cons_mom_lo', np.nan))} | {_pct(s.get('opt_mom_lo', np.nan))} | {s.get('mom_pairs', 0)} "
                 f"({s.get('mom_pairs_dropped', 0)})·{s.get('mom_unmatched_n', 0)} | "
                 f"{_r(s.get('mom_ctl_hold_median', np.nan), '{:.0f}')}/{_r(s.get('mom_ctl_hold_mean', np.nan), '{:.1f}')} | "
                 f"{_r(s.get('mom_trade_hold_median', np.nan), '{:.0f}')} |")
    L += ["", "정합 alpha = 비용 전 총수익. 매수 +0.1%·손절 −0.1% 는 양쪽 같다(정정 2). 남은 비대칭(사건 쪽만 보수): 진입일 순서 모호성은 "
          "사건 쪽만 p 혼합, 매수정지 진입일 2R 익절 불인정(갭 체결 포함 — 대조는 시가 진입이라 인정). 대조는 진입가 기준 돌파 실패라 "
          "이틀 보유가 많다(리뷰 C1) — 보유일을 같이 적었다. 널 벌 체결 수는 겹침·같은 피벗 거르기로 필요 수보다 적을 수 있다(체결/필요 열).", ""]

    L += ["## 진입 방식 분리 — X1·X2 vs V0, 같은 피벗 쌍 (정정 M5 · 정정 2 B-M2)", "",
          "| 비교 | 공통 피벗 | V0 체결 · X 체결 | V0 순 1배 | X 순 1배 | X − V0 [날짜군집 95% CI] | 차이 / 0.07 | d 중앙 V0 / X |",
          "|---|---|---|---|---|---|---|---|"]
    for x, pr in pairs.items():
        if not pr.get("n"):
            L.append(f"| {x} − V0 | 0 | {pr.get('n_a', 0)} · {pr.get('n_b', 0)} | | | | | |")
            continue
        L.append(f"| {x} − V0 | {pr['n']} | {pr['n_a']} · {pr['n_b']} | {_pct(pr['a_mean'])} | {_pct(pr['b_mean'])} | "
                 f"{_pct(pr['diff_mean'])} [{_pct(pr['diff_lo'])}, {_pct(pr['diff_hi'])}] | {_r(pr['diff_r_fixed'], '{:+.3f}')} | "
                 f"{_pct(pr['a_d_med'])} / {_pct(pr['b_d_med'])} |")
    L += ["", "X2 주 = 옛 저점 min(low[d−10..d−1])(주문 시점에 알려진 값), X2L = 문구 그대로 min(low[d−10..d])(진입일 저가 포함 — 진입일 손절을 "
          "구조적으로 피한다, 기술).", ""]

    L += ["## 수급 F_base 3분위 (V0 체결, 2024-04-25+, 기술) · 같은 기간 위성 P1", ""]
    if f_terc.get("n", 0) >= 3:
        L += ["| F_base 3분위 | n | 범위 | 순 1배 | 손실 청산 |", "|---|---|---|---|---|"]
        for k in ("하", "중", "상"):
            if k in f_terc:
                d_ = f_terc[k]
                L.append(f"| {k} | {d_['n']} | {d_['f_lo']:+.3f}~{d_['f_hi']:+.3f} | {_pct(d_['net_mean'])} | {_sh(d_['loss_rate'])} |")
    else:
        L.append(f"V0 수급 기간 체결 중 F_base 알려짐 {f_terc.get('n', 0)}건 — 3분위 생략.")
    if p1.get("n"):
        L += ["", f"위성 P1(`{P1_TRADES}` — 라이브 백테스트 B 판, 진입 {p1['from']}~{p1['to']}): n {p1['n']} · 순 {_pct(p1['net_mean'])} · "
              f"손실 청산 {_sh(p1['loss_rate'])} · 손익비 {_r(p1['gl_ratio'])} · 평균 손실 R {_r(p1['avg_loss_R'], '{:+.2f}')} · 순<−12% "
              f"{_sh(p1['big_loss'])} (포트폴리오 체결·비용 기준이 달라 나란히 기술만)."]
    else:
        L += ["", "위성 P1: 같은 기간 거래 없음."]
    L.append("")

    L += ["## 10 슬롯 포트폴리오 · 팩터 회귀 (바 6 기술)", "",
          "| 팔 | 기간 | 편입/못 들어감 | 평균 노출 | CAGR 1배 (보수/낙관) | MDD 1배 | CAGR 2배 | MDD 2배 | Sharpe(원) | 초과 Sharpe(CD91) | "
          "코스피 CAGR | 코스피 MDD | 팩터 α/년 (t) | t 2배 | t 극단일 제외 | R² |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ARMS:
        s = ports[n]["stats"]
        if not s.get("port_taken"):
            L.append(f"| {n} | — |" + " |" * 14)
            continue
        L.append(f"| {n} | {s['port_from']}~{s['port_to']} ({s['port_days']}일) | {s['port_taken']}/{s['port_skipped']} | "
                 f"{_sh(s['port_exposure'])} | {_pct(s['port_cagr'])} ({_pct(s['port_cagr_cons'])}/{_pct(s['port_cagr_opt'])}) | "
                 f"{_pct(s['port_mdd'])} | {_pct(s['port_cagr2'])} | {_pct(s['port_mdd2'])} | {_r(s['port_sharpe'])} | "
                 f"{_r(s.get('port_sharpe_exc', np.nan))} | {_pct(s['kospi_cagr'])} | {_pct(s['kospi_mdd'])} | "
                 f"{_pct(s.get('fac_alpha_ann', np.nan))} ({_r(s.get('fac_alpha_t', np.nan))}) | {_r(s.get('fac_alpha_t2', np.nan))} | "
                 f"{_r(s.get('fac_alpha_t_noext', np.nan))} | {_r(s.get('fac_r2', np.nan))} |")
    L += ["", f"회귀(정정 M4): 날별 포트폴리오 원수익(현금 0%, 주 = 보수·낙관 경로 날별 수익의 p 혼합) ~ 상수 + "
          f"{', '.join(f'`{c}`' for c in FACTOR_COLS)}(원수익), Newey-West 5. 시총가중 열은 파일에 없다(TVW = 직전 ADV 가중). "
          "초과 Sharpe 의 현금 = CD91 기억값(`meta_gen5_run.CD91_MEM`, DB 밖). 코스피 = 네이버 공식 지수 원수익(기술).", ""]

    m = minute["summary"]
    L += ["## 전략 체결 분봉 검증 (2025-08+, 기술)", "",
          f"- 대상 {m.get('n', 0)}건 · 분봉 있음 {m.get('n_minute', 0)} · 분봉상 주문가 도달 {m.get('n_touch', 0)}",
          f"- 체결가 오차(일봉 가정 / 분봉 첫 도달 봉 max(시가, 주문가) − 1): 평균 {_pct(m.get('fill_err_mean', np.nan))} · 중앙 "
          f"{_pct(m.get('fill_err_median', np.nan))} · |최대| {_pct(m.get('fill_err_absmax', np.nan))}",
          f"- 모호 체결 {m.get('amb_n', 0)}건: 체결 봉 뒤 손절 터치 {m.get('amb_after', 0)} · 같은 봉만 {m.get('amb_same_only', 0)} · "
          f"체결 전만 {m.get('amb_before_only', 0)}", ""]

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          "- **셋업 행:** 주문일 d 의 셋업 = 신호 행 s = d−1 의 추세·RS·breadth·H5 와 s 종가까지의 VCP. 창 260일이 다 차지 않거나 결측이면 셋업 "
          "없음. 사건 구간은 **주문일** 기준. 분할 제외 = 조정 종가 / 원 종가 비가 2% 넘게 바뀐 날(같은 날 ≥ 5건인 수집 배치일 제외)부터 60 "
          "거래일 + VCP 통과 셋업의 거래량 창 [ip−49, s] 안 개별 변화.",
          "- **지그재그(정정 2 A-C1):** 봉마다 상태 전이 하나, 진행 방향 극값 먼저·반전 확인은 다음 봉부터(두 상태 대칭). 동률은 늦은 봉. "
          "피벗 = 오름 상태면 마지막 확정 저점 뒤 진행 중 최고 고가, 내림 상태면 마지막 확정 고점. **저점 상승(정정 2 A-M3):** L_k ≥ L_{k−1}×0.99. "
          f"H0 윗꼬리 = (H0 − max(시가, 종가)) / H0 ≥ {H0_WICK:.0%} 표시(기술).",
          "- **거래량 마름(정정 M7):** mean(vol[ip+1..s]) ≤ 0.8 × mean(vol[ip−49..ip]) — 비교 창은 피벗 봉을 포함한 50일.",
          "- **매수정지·손절:** 주문가 = 피벗 + `tick_size_vec(피벗)`, 손절가 = L_last − `tick_size_vec(L_last)`(현행 호가표를 분할조정가에 — "
          "리뷰 MINOR 3 한계). 진입가 = max(시가, 주문가) × 1.001, 손절·본전 매도가 × 0.999. 진입일 2R 익절 불인정, 진입일 종가 < 피벗 → 다음날 "
          "시가(정정 M8). 본전선 = 진입가(라이브 기본).",
          "- **진입일 순서(정정 C3·정정 2 B-M1):** 진입일 종가 ≤ 손절가 = 확정 손절(두 경로 같음). 모호 = 시가 ≤ 피벗 ∧ 저가 ≤ 손절가 ∧ 종가 > "
          "손절가. 한 시뮬에서 낙관 경로와 보수 경로(진입일 손절가 × 0.999, 비용은 진입일 세율)를 함께 내고, 바 숫자는 모호 거래를 가중 "
          "p·1−p 두 행으로 펼친 가중 통계. p 추정도 같은 식으로 모호 사건을 고르고 확정 손절은 분모 밖.",
          "- **바 5(정정 C4·정정 2):** 1배 순수익으로 판정(손실 = 순 < 0, 손익비, 평균 손실 R = 손실 거래 순/그 거래 d, R 5백분위, 순 < −12%), "
          "2배 값은 5′ 줄(기술). 바 2·3 = 2배. 3분류 본전 = |순| < 0.5%.",
          "- **완결 거래만:** 가중 행 단위로 낙관 경로 `data_end` 행만 뺀다(모호 거래의 보수 행은 완결).",
          "- **NXT(정정 2 B-M3):** 2026-09-14+ 미보정 행이 있는 종목은 보유 경로를 2026-09-11 종가에서 `data_end` 로 끊는다(트레이드·널·대조).",
          "- **실적(정정 M2):** 최신 분기 = 그날 보이는 가장 늦은 분기(미루지 않음), Q4 단독 = 연간 − 같은 해 Q1~Q3(전년 동기도 당해 보고서 비교 "
          "수치끼리), yoy_prev 는 직전 분기 전년 동기 > 0 일 때만, 140일 문 없음. 이중 시간축은 사실상 `avail_date`(기한 고정값) 한 축이다 — "
          "지연 공시자의 룩어헤드는 보정 불가(한계).",
          "- **수급(정정 M3):** F_base 주 = 하루 (외국인+기관)/`acc_trde_qty` 의 거래대금 가중평균(i0+1..s, 알려진 날 ≥ 80%), 모름은 팔에서 "
          "빠진다(모름 수를 기반 길이별로 깔때기에). 널의 수급 창 = 그 셋업의 k일 피벗 창.",
          "- **널(정정 M1):** 날짜 맞춤 단위 = 주문월, d 3분위 경계 = 그 V 팔 체결 d 의 1/3·2/3 분위, 널 피벗도 피벗 뒤 ≥ 3일. 널 풀은 VCP 셋업이 "
          "아닌 행이라 같은 (종목, 날짜)를 쓰지 않는다(러너 단언).",
          "- **X1:** V0 셋업 행 d 에서 종가 > 피벗 ∧ 거래량 ≥ 1.5 × 직전 50일(d−50..d−1) 평균 → d+1 시가, 추격 5%·상한가·정지 버림, "
          "손절 = 새 손절(2% 하한), d > 10% 면 안 산다, 피벗당 첫 신호 하나. **X2(정정 2 B-M2):** V0 와 같은 매수정지·주문 전 거르기, 손절만 "
          "min(low[d−10..d−1]) × 0.99 를 [3%, 10%] 로 자름. **X2L** = 문구판(진입일 저가 포함, 기술).",
          "- **포트폴리오:** 슬롯당 전일 종가 자산의 10%, 같은 날 ADV20 큰 순, 그날 청산 자리는 그날 안 씀, 비용 = 배수 × `cost_rt` × 배분액, 현금 0%.",
          "- **위성 P1:** `trades_b.csv`(라이브 백테스트 B 판)의 `net_ret` 을 같은 진입 기간으로 잘라 나란히 적는다(같은 청산 재시뮬 아님).", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
