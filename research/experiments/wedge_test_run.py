#!/usr/bin/env python
"""천장 아래 하락 쐐기 돌파 — 사용자 정의 수렴 진입, 사전등록 러너. **리포터 — 판정 아님.**

사전등록: ``research/logs/sepa_explore/WEDGE_TEST.md`` — 7196cad + 정정 0 d1b7db8(사용자 그림) + 정정 0b 0a911e2(W-be1) + 정정 1
7e99bdb(설계 리뷰) + 정정 2 bf62569(쐐기 = 마지막 다섯 점 A~E, H0→A 구간 제약 없음), 모두 성과 보기 전. 이 러너는 숫자를 ``WEDGE_TEST_RESULTS.md`` 에 적고 바 문턱과의 **산술 비교**만 한다.
**정정 1 C1 사전 약속:** 건수 규칙이 VOID 면 W 는 판정 없는 기술로 끝까지 돌리고, 결과를 본 뒤 이 데이터에서 정의·청산·하한을 바꿔
다시 돌리지 않는다. 판정 팔이 있다면 W(이틀)·W1d(하루) 둘 다 통과해야 한다(같은 표).

- 자료·유니버스·비용·NXT 절단·분할 제외·진입일 모호성 p 혼합·같은 메뉴 널·모멘텀 정합·포트폴리오 팩터는 VCP 재시험 러너
  (``vcp_retest_run``)의 같은 단계를 그대로 쓴다(import 만).
- 셋업: :mod:`research.signals.wedge` (하락 쐐기 §2) + 이 러너의 단계 2 맥락(§1): d−1 MA50 > MA150 > MA200 ∧ MA200 > 20일 전 ∧ breadth
  > 0.5 (후보 거르기), H0 일 추세 템플릿 ∧ RS ≥ 70 (스캔 뒤 i0 에서).
- 진입·청산: :func:`wedge.simulate_wedge` (주문가 = U(d) + 1호가, 9.5%·3% 갭 기준 = 주문가, 돌파 실패 이틀 규칙).
- 팔: W · W1d(하루 규칙) · W-be1(+1R 뒤 본전 이동, 정정 0b) · W-nc(수렴 조건 끔, 정정 1 M5) · W-tt · W-E · W-F. (V0r 삭제 — 정정 1 M4.)
- 같은 메뉴 널(정정 1 M5): 맥락 통과 ∧ 쐐기 아님 ∧ 깊이 띠 종목-일에서 하락 기준선 매수정지(:func:`wedge.null_lines`, k ~ U{15..60}).
- ``--funnel-only``: ADV20 하한 50억·30억·20억·10억 각각 체결 **건수만** → ``wedge/FUNNEL.md`` (W 체결 ≥ 80 ∧ ≤ 2024-12 ≥ 40 인 가장
  높은 하한이 판정 유니버스, 넷 다 미달이면 W VOID) + 차트용 ``wedge/chart_points.json``(기하만, 성과 없음).
- 원장: ``research/logs/sepa_explore/wedge/TRIALS.jsonl`` — 팔·민감도 칸을 실행 전에(깔때기 모드는 쓰지 않는다).

Run:  uv run --with numba --extra pg python research/experiments/wedge_test_run.py --funnel-only
      uv run --with numba --extra pg python research/experiments/wedge_test_run.py --workers 12
디버그: ... wedge_test_run.py --event-lo 2020-06-01 --event-hi 2021-06-30 --adv-floor 5000 --out <scratch>/w --n-null 8 \\
          --p-hi 2025-12-31 --counts-only
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
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

import vcp_retest_run as R  # noqa: E402
from sepa_confluence_gate import N_BOOT, SEED, _log, _pct  # noqa: E402
from sepa_momentum import MOM_N_Q, day_market_median_half, day_market_quantiles, momentum_12_1  # noqa: E402

from krx_quant_core.stats.matched_null import cluster_bootstrap_ci  # noqa: E402
from research.signals import vcp as V  # noqa: E402
from research.signals import wedge as Wd  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.technical import moving_average  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "wedge"
SUB = "wedge"
RESULTS_MD = "WEDGE_TEST_RESULTS.md"
FUNNEL_MD = "research/logs/sepa_explore/wedge/FUNNEL.md"
PREREG_DOC = ("research/logs/sepa_explore/WEDGE_TEST.md (7196cad + 정정 0 d1b7db8 + 정정 0b 0a911e2 + 정정 1 7e99bdb + "
              "정정 2 bf62569)")
PRIOR_LEDGERS = (*R.PRIOR_LEDGERS, ("vcp_retest", "research/logs/sepa_explore"))

# --- 사전등록 수치(WEDGE_TEST.md — 탐색하지 않는다) ---
EVENT_LO, EVENT_HI = R.EVENT_LO, R.EVENT_HI
CAP_SENS, FLOW_LO, SUB_HI = R.CAP_SENS, R.FLOW_LO, R.SUB_HI
RS_MIN, BREADTH_MIN = 70.0, 0.5
ADV_FLOORS_FUNNEL = (eok(50), eok(30), eok(20), eok(10))
FLOOR_RULE_N, FLOOR_RULE_SUB = 80, 40
B1_N, B1_SUB_N = 80, 40
GL_MIN, AVG_LOSS_R_MIN, R_P5_MIN, BIG_LOSS_MAX, BE_MARGIN = 2.0, -1.0, -1.3, 0.05, 0.05
NULL_BAND = Wd.WEDGE_PARAMS["close_lo"]
N_NULL, NULL_K, NULL_J, NULL_SEED = 300, (15, 60), (5, 15), 20261011     # 정정 1 M5: k ~ U{15..60}
NULL_EXPAND, NULL_POOL_MULT = 5, 3
VOL_BUCKETS = (1.0, 1.5)                          # 돌파일 거래량 / 직전 50일 평균: < 1, 1~1.5, ≥ 1.5
PRIMARY = "W"
BAR_ARMS = ("W", "W1d")                           # 정정 1 M4: 판정이 있다면 둘 다 통과해야 — 같은 표
P_W = {**Wd.WEDGE_BUYSTOP, "fb_days": 2}
ARMS: dict[str, dict] = {
    "W": {"lo": EVENT_LO, "params": P_W, "desc": "하락 쐐기(정정 1), 돌파 실패 이틀 규칙"},
    "W1d": {"lo": EVENT_LO, "params": {**P_W, "fb_days": 1}, "desc": "W, 돌파 실패 하루 규칙"},
    "W-be1": {"lo": EVENT_LO, "params": {**P_W, "be_r": 1.0}, "desc": "W, +1R 뒤 본전 이동(정정 0b, 기술)"},
    "W-nc": {"lo": EVENT_LO, "params": P_W, "desc": "W, 수렴 조건 끔(c_k 감소·기울기 순서 — 하락 계단, 정정 1 M5, 기술)"},
    "W-tt": {"lo": EVENT_LO, "params": P_W, "desc": "W ∧ d−1 추세 템플릿 전부 (기술)"},
    "W-E": {"lo": EVENT_LO, "params": P_W, "desc": "W ∧ 분기 단독 실적 가속 (기술)"},
    "W-F": {"lo": FLOW_LO, "params": P_W, "desc": "W ∧ 쐐기 기간(A~s) 거래대금 가중 수급 > 0 (기술)"},
}
NULL_ARMS = ("W",)
SENS_CELLS = ("W 보수(p=1)", "W 낙관(p=0)", "W 완결 거래만", f"W 사건 ≤ {CAP_SENS}", "W p 재가중", "W 팩터 극단일 제외")
#: 설계 변형이 정의된 사전등록 판 — 지문 고정용(사전등록 문자열이 늘어도 같은 변형이 원장에 두 번 들어가지 않게).
VARIANT_PREREG = "research/logs/sepa_explore/WEDGE_TEST.md (7196cad + 정정 0 d1b7db8 + 정정 0b 0a911e2 + 정정 1 7e99bdb)"
#: 정정 1 MINOR — 리뷰가 센 설계 변형 8개(건수만, 성과 미계산). 원장에 시행 수로 남긴다(나중에 변형으로 갈아타면 N 에 들어간다).
DESIGN_VARIANTS = ("W 문구대로(7196cad+정정 0)", "W + A 양쪽 ±3%", "W + E 뒤 상단선 장중 관통 없음(첫 돌파만)", "W + E 진행 중 허용",
                   "W + E 진행 중 + E 뒤 첫 돌파만", "수렴 조건 끔(하락 계단)", "끝 5점만 쐐기 + E 진행 중 + 첫 돌파만",
                   "끝 5점만 쐐기 + E 진행 중 + 첫 돌파만 + A 양쪽 ±3%")


def _arm_config(name: str, lo: str, hi: str, floor: float, fac_sha: str) -> dict:
    a = ARMS[name]
    return {"prereg": PREREG_DOC, "arm": name, "desc": a["desc"], "event_lo": lo, "event_hi": hi, "adv_floor": floor,
            "universe": R.UNIVERSE_DESC.format(floor=floor), "prices": R.PRICE_DESC, "wedge": dict(Wd.WEDGE_PARAMS),
            "context": "H0 일 trend_template ∧ RS≥70; d−1 MA50>MA150>MA200 ∧ MA200 > 20일 전 ∧ breadth>0.5",
            "buystop": {**V.BUYSTOP_PARAMS, **a["params"]}, "cost": "korea.cost_rt 1배 · 2배(바 2·3)",
            "same_day": "진입일 종가 ≤ 손절가 = 확정; 모호 = 분봉 p 혼합(합성 사건 근접 띠 0.92·주문가 기준) · 보수 · 낙관",
            "null": f"하락 기준선 매수정지 널 {N_NULL}벌 (맥락 통과·쐐기 아님·깊이 띠, 직전 k일 두 확정 고점 선 k~U{NULL_K}, "
                    f"손절 j~U{NULL_J}, 첫 관통·종가<선(d)·띠 0.92·9.5%·이틀 규칙, 월별 체결 × d 3분위, seed {NULL_SEED})"
            if name in BAR_ARMS else None,
            "factors": f"{R.FACTOR_COLS} 원수익 NW 5, factors.pkl sha256 {fac_sha}", "judged": name == PRIMARY}


# ---------------------------------------------------------------------------
# 순수 헬퍼
# ---------------------------------------------------------------------------

def win_ex_be(s: dict) -> float:
    """정정 1 MINOR: 손익분기 비교용 승률 = 승 / (승 + 패), 본전(|순| < 0.5%) 제외."""
    w, lo = s.get("win", np.nan), s.get("lossc", np.nan)
    return w / (w + lo) if np.isfinite(w) and np.isfinite(lo) and (w + lo) > 0 else np.nan


def wedge_bars(s: dict, null_p95: float, mom_lo: float, fac_t: float) -> dict:
    """사전등록 바 1~5 산술(+ 바 6 기술). 바 2·3 = 2배, 바 5 = 1배(손익분기 승률 여유 +5%p — 승률은 본전 제외)."""
    def fin(v):
        return v is not None and np.isfinite(v)

    n, ns = s.get("n", 0), s.get("n_sub", 0)
    gl, gr = s.get("gl_ratio", np.nan), win_ex_be(s)
    be_need = 1.0 / (1.0 + gl) + BE_MARGIN if fin(gl) else np.nan
    out = {"b1": bool(n >= B1_N and ns >= B1_SUB_N),
           "b2": bool(fin(s.get("net2_mean")) and s["net2_mean"] > 0 and fin(s.get("net2_sub")) and s["net2_sub"] > 0
                      and fin(s.get("net2_ex2021")) and s["net2_ex2021"] > 0),
           "b3": bool(fin(s.get("net2_mean")) and fin(null_p95) and s["net2_mean"] > null_p95),
           "b4": bool(fin(mom_lo) and mom_lo > 0),
           "b5": bool(fin(gl) and gl >= GL_MIN and fin(s.get("avg_loss_R")) and s["avg_loss_R"] >= AVG_LOSS_R_MIN
                      and fin(s.get("r_p5")) and s["r_p5"] >= R_P5_MIN and fin(s.get("big_loss"))
                      and s["big_loss"] <= BIG_LOSS_MAX and fin(gr) and fin(be_need) and gr > be_need),
           "b6": bool(fin(fac_t) and fac_t > 2.0), "be_need": be_need, "win_ex_be": gr}
    out["n_met"] = int(sum(out[f"b{i}"] for i in range(1, 6)))
    return out


def vol_bucket(ratio) -> np.ndarray:
    """돌파일 거래량 / 직전 50일 평균 → 0 (< 1), 1 (1~1.5), 2 (≥ 1.5), NaN → −1."""
    r = np.asarray(ratio, float)
    b = np.searchsorted(np.asarray(VOL_BUCKETS), r, side="right")
    return np.where(np.isfinite(r), b, -1)


# ---------------------------------------------------------------------------
# 자료 · 셋업
# ---------------------------------------------------------------------------

class WCtx(R.Ctx):
    """VCP 러너의 공용 자료 + 단계 2 맥락 패널(MA150·MA200·MA200 20일 전)."""

    def __init__(self, t0: float):
        super().__init__(t0)
        c = self.p.close.reindex(index=self.days, columns=self.codes)
        m150 = moving_average(c, 150)
        m200 = moving_average(c, 200)
        self.struct = ((self.ma50 > m150.to_numpy(float)) & (m150.to_numpy(float) > m200.to_numpy(float))
                       & (m200.to_numpy(float) > m200.shift(20).to_numpy(float)))


def build_wedge(ctx: WCtx, floor: float, lo_all: str, hi: str, lo_of: dict) -> dict:
    """하한 ``floor`` 의 쐐기 셋업 표·팔 행·깔때기(성과 계산 없음)."""
    A, days, codes = ctx.A, ctx.days, ctx.codes
    nd = len(days)
    h5 = ctx.universe(floor)
    d_ok = np.asarray((days >= pd.Timestamp(lo_all)) & (days <= pd.Timestamp(hi)))
    s_ok = np.zeros(nd, bool)
    s_ok[:-1] = d_ok[1:]
    base_nosplit = h5.to_numpy(bool) & ctx.struct & (ctx.br > BREADTH_MIN) & s_ok[:, None]
    base = base_nosplit & ~ctx.split["recent"]
    si, ci = np.nonzero(base)
    sc = Wd.wedge_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci, si)
    nc = Wd.wedge_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci, si, {"conv": False})       # W-nc(수렴 조건 끔)
    S = pd.DataFrame({"code": codes[ci], "sig_idx": si, "day_idx": si + 1, "col_idx": ci, **sc})
    S["event_day"] = days[si]
    S["order_day"] = days[si + 1]
    i0 = S["i0"].to_numpy()
    i0s = np.maximum(i0, 0)
    S["base_len"] = np.where(i0 >= 0, si - i0, -1)
    in_base = (S["base_len"] >= Wd.WEDGE_PARAMS["base_min"]) & (S["base_len"] <= Wd.WEDGE_PARAMS["base_max"])
    S["context"] = (in_base & (i0 >= 0) & ctx.tt[i0s, ci] & (ctx.rs[i0s, ci] >= RS_MIN)).to_numpy(bool)
    S["valid"] = (S["status"] == Wd.W_OK) & S["context"]
    S["valid_nc"] = (nc["status"] == Wd.W_OK) & S["context"].to_numpy(bool)
    for kk in ("u_d", "floor_low", "i_high", "i_low", "i_a"):           # W-nc 행의 쐐기 값(수렴 조건만 다르다)
        S[f"nc_{kk}"] = nc[kk]
    vv = S["valid"].to_numpy()
    S["market"] = ctx.p.market.reindex(S["code"]).to_numpy()
    S["tt_s"] = ctx.tt[si, ci]
    S["adv20"] = ctx.adv[si, ci]
    S["ma50_s"] = ctx.ma50[si, ci]
    S["close_s"] = A.C[ci, si]
    S["rs_s"] = ctx.rs[si, ci]
    a_row = S["i_a"].to_numpy()
    mn = Wd.min_low_since(A.L, ci, i0s, si)
    S["depth_any"] = np.where(i0 >= 0, (S["h0"] - mn) / S["h0"], np.nan)               # 널 풀 깊이 띠용
    E = V.e_accel_asof(ctx.est, S["code"].to_numpy(), S["event_day"].to_numpy())
    S["e_accel"] = E["e_accel"].to_numpy(bool)
    S["f_wedge"] = np.where(vv & (a_row >= 1), ctx.FCw.f_base(ci, np.maximum(a_row - 1, 0), si), np.nan)  # A..s
    dts = S["order_day"]
    rng_ = {n: (dts >= pd.Timestamp(lo_of[n])) & (dts <= pd.Timestamp(hi)) for n in ARMS}
    v = S["valid"]
    arm_rows = {"W": v & rng_["W"], "W1d": v & rng_["W1d"], "W-be1": v & rng_["W-be1"], "W-nc": S["valid_nc"] & rng_["W-nc"],
                "W-tt": v & S["tt_s"] & rng_["W-tt"], "W-E": v & S["e_accel"] & rng_["W-E"],
                "W-F": v & (S["f_wedge"] > 0) & rng_["W-F"]}
    status = {Wd.WEDGE_STATUS[k]: int(n_) for k, n_ in S["status"].value_counts().items()}
    funnel = {"floor": floor, "base_nosplit": int(base_nosplit.sum()), "split_excl": int((base_nosplit & ctx.split["recent"]).sum()),
              "base": int(len(S)), "status": status, "wedge_ok": int((S["status"] == Wd.W_OK).sum()),
              "context_fail": int(((S["status"] == Wd.W_OK) & ~S["context"]).sum()), "valid": int(vv.sum()),
              "context_pool": int((S["context"] & ~S["valid"]).sum()),
              "null_pool": int((S["context"] & ~S["valid"] & S["depth_any"].between(Wd.WEDGE_PARAMS["depth_lo"],
                                                                                    Wd.WEDGE_PARAMS["depth_hi"])).sum()),
              "valid_nc": int(S["valid_nc"].sum()), "e_inprogress_valid": int((v & (S["e_confirmed"] == 0)).sum()),
              "surge30_valid": int((v & (S["surge20"] > 0.30)).sum()),
              "e_accel_valid": int((v & S["e_accel"]).sum()), "tt_valid": int((v & S["tt_s"]).sum()),
              "flow_period_valid": int((v & rng_["W-F"]).sum()),
              "flow_known_valid": int((v & rng_["W-F"] & S["f_wedge"].notna()).sum()),
              "arms": {n: int(m.sum()) for n, m in arm_rows.items()}}
    return {"S": S, "h5": h5, "arm_rows": arm_rows, "rng": rng_, "funnel": funnel}


def simulate_all(ctx: WCtx, B: dict) -> "tuple[dict, dict]":
    """팔별 시뮬(진입 + 보유 거르기에 쓰는 청산). 비용·순수익은 붙이지 않는다(깔때기 모드에서도 쓴다)."""
    trades, skips = {}, {}
    for n in ARMS:
        ev = B["S"][B["arm_rows"][n]].reset_index(drop=True)
        if n == "W-nc":
            ev = ev.assign(**{kk: ev[f"nc_{kk}"] for kk in ("u_d", "floor_low", "i_high", "i_low", "i_a")})
        t, sk = V.simulate_buystop(ctx.A, Wd.wedge_events(ev), params=ARMS[n]["params"])
        trades[n], skips[n] = t, (sk["skip_reason"].value_counts().to_dict() if len(sk) else {})
    return trades, skips


def entry_desc(ctx: WCtx, t: pd.DataFrame) -> pd.DataFrame:
    """진입 기술 열(성과 아님): 돌파일 거래량 비·구간, 체결가 → H0 거리."""
    if t is None or t.empty:
        return t
    t = t.copy()
    ci, d = t["col_idx"].to_numpy(int), t["day_idx"].to_numpy(int)
    vavg = np.array([ctx.VOL[c, k - 50:k].mean() if k >= 50 else np.nan for c, k in zip(ci, d, strict=True)])
    with np.errstate(invalid="ignore", divide="ignore"):
        t["bo_vol_ratio"] = ctx.VOL[ci, d] / vavg
    t["bo_vol_bucket"] = vol_bucket(t["bo_vol_ratio"])
    if "h0" in t:
        t["h0_dist"] = t["h0"] / t["fill_px"] - 1.0
    return t


def chart_points(ctx: WCtx, t: pd.DataFrame, with_exit: bool = False) -> list:
    """차트용 — W 체결의 H0·A~E(진행 중 E 후보 포함)·상단선/하단선 끝점·U(d)·주문가·체결가·손절가(날짜·가격). ``with_exit`` 면
    청산(낙관 경로 청산일·가격·사유, 모호 거래의 보수 경로 표시)도 — 깔때기 모드는 넣지 않는다."""
    out = []
    days = ctx.days
    for r in t.itertuples(index=False):
        pt = Wd.wedge_points(ctx.A.H, ctx.A.L, int(r.col_idx), int(r.sig_idx))

        def dd(row):
            return str(days[min(int(row), len(days) - 1)].date())
        item = {"code": str(r.code), "signal_day": str(pd.Timestamp(r.event_day).date()),
                "order_day": str(pd.Timestamp(r.order_day).date()), "h0": {"date": dd(pt["i0"]), "price": pt["h0"]},
                "points": [{"date": dd(p["row"]), "price": p["price"], "kind": p["kind"]} for p in pt["points"]],
                "upper": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt.get("upper", {}).items()},
                "lower": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt.get("lower", {}).items()},
                "u_d": pt.get("u_d"), "order_px": float(r.order_px), "fill_px": float(r.fill_px), "stop_px": float(r.stop_px),
                "gap_fill": bool(r.gap_fill)}
        if with_exit:
            item["exit"] = {"date": str(pd.Timestamp(r.exit_day).date()), "price": float(r.exit_px),
                            "reason": str(r.exit_reason), "ambiguous_entry_day": bool(r.amb),
                            "partial": ({"date": str(pd.Timestamp(r.partial_day).date()), "price": float(r.partial_px)}
                                        if bool(r.partial_hit) else None)}
        out.append(item)
    return out


def funnel_counts(ctx: WCtx, floor: float, lo_all: str, hi: str, lo_of: dict) -> dict:
    """``--funnel-only`` 한 하한 — 건수만(청산 결과·수익은 어디에도 쓰지 않는다)."""
    B = build_wedge(ctx, floor, lo_all, hi, lo_of)
    trades, skips = simulate_all(ctx, B)
    t = entry_desc(ctx, trades["W"])
    sub = (pd.to_datetime(t["order_day"]) <= pd.Timestamp(SUB_HI)) if len(t) else pd.Series(dtype=bool)
    out = {"funnel": B["funnel"], "skips": skips, **{n: int(len(trades[n])) for n in ARMS},
           "W_sub": int(sub.sum()) if len(t) else 0,
           "W_years": ({int(k): int(v) for k, v in pd.to_datetime(t["order_day"]).dt.year.value_counts().sort_index().items()}
                       if len(t) else {}),
           "W_base_len": {q: float(t["base_len"].quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if len(t) else {},
           "W_h0_dist": {q: float(t["h0_dist"].quantile(q)) for q in (0.1, 0.5, 0.9)} if len(t) else {},
           "W_vol_buckets": {int(k): int(v) for k, v in pd.Series(t["bo_vol_bucket"]).value_counts().sort_index().items()}
           if len(t) else {},
           "W_n_contr": {int(k): int(v) for k, v in t["n_contr"].value_counts().sort_index().items()} if len(t) else {},
           "W_adv_median": float(t["adv20"].median()) if len(t) else np.nan,
           "W_surge30": int((t["surge20"] > 0.30).sum()) if len(t) else 0,
           "W_surge_q": {q: float(t["surge20"].quantile(q)) for q in (0.1, 0.5, 0.9)} if len(t) else {},
           "W_e_inprogress": int((t["e_confirmed"] == 0).sum()) if len(t) else 0,
           "W_gap_fill": int(t["gap_fill"].sum()) if len(t) else 0,
           "W_rs_q": {q: float(t["rs_s"].quantile(q)) for q in (0.1, 0.5, 0.9)} if len(t) else {},
           "W_liquid_1pct": int((t["adv20"] > eok(10)).sum()) if len(t) else 0,
           "chart": chart_points(ctx, t) if len(t) else []}
    return out


def write_funnel(path: str, res: dict, lo_all: str, hi: str, rule: tuple, elapsed: float, ctx: WCtx) -> None:
    floor, void = rule
    fl = sorted(res, reverse=True)
    L = [f"# 천장 아래 하락 쐐기 — 깔때기(건수만, 성과 미계산) {date.today().isoformat()}", "",
         f"`wedge_test_run.py --funnel-only` ({PREREG_DOC}). 사건(주문일) {lo_all}~{hi}. 시뮬은 진입 체결과 보유 중 거르기(그 거르기에 "
         f"필요한 청산일)까지만 쓰고, 청산 사유·수익·손절 여부는 어디에도 적지 않는다. 실행 {elapsed / 60:.1f}분.", "",
         f"**사전 규칙:** W 체결 ≥ {FLOOR_RULE_N} ∧ 2017-09~2024-12 체결 ≥ {FLOOR_RULE_SUB} 인 가장 높은 하한 → "
         + (f"**ADV20 ≥ {floor:,.0f} 백만원({floor / 100:,.0f}억)** 이 판정 유니버스." if floor is not None
            else "**넷 다 미달 → W VOID**(기술만)."),
         f"<!-- adv_floor_rule: {'none' if floor is None else f'{floor:.0f}'} · v0_void: {'true' if void else 'false'} -->", "",
         "| ADV20 하한(백만원) | W 체결 | 하위(≤2024-12) | 규칙 충족 | W1d | W-be1 | W-nc | W-tt | W-E | W-F | 쐐기 셋업 행(맥락 통과) | 셋업 후보 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fl:
        r = res[f]
        ok = r["W"] >= FLOOR_RULE_N and r["W_sub"] >= FLOOR_RULE_SUB
        L.append(f"| {f:,.0f} | {r['W']} | {r['W_sub']} | {'✓' if ok else '✗'} | {r['W1d']} | {r['W-be1']} | {r['W-nc']} | "
                 f"{r['W-tt']} | {r['W-E']} | {r['W-F']} | {r['funnel']['valid']:,} | {r['funnel']['base']:,} |")
    L += ["", "## 하한별 깔때기", ""]
    for f in fl:
        r, fu = res[f], res[f]["funnel"]
        st = fu["status"]
        bl, hd = r["W_base_len"], r["W_h0_dist"]
        L += [f"### ADV20 ≥ {f:,.0f} 백만원", "",
              f"- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) {fu['base_nosplit']:,} − 분할 직후 60일 "
              f"{fu['split_excl']:,} = {fu['base']:,}",
              "- 쐐기 첫 탈락: " + " · ".join(f"{k} {v:,}" for k, v in sorted(st.items(), key=lambda kv: Wd.WEDGE_STATUS.index(kv[0]))),
              f"- 쐐기 통과 {fu['wedge_ok']:,} − H0 맥락(추세 템플릿 ∧ RS ≥ 70) 탈락 {fu['context_fail']:,} = **{fu['valid']:,}** · "
              f"d−1 추세 템플릿 {fu['tt_valid']:,} · E 가속 {fu['e_accel_valid']:,} · 수급 기간 {fu['flow_period_valid']:,} (F 알려짐 "
              f"{fu['flow_known_valid']:,}) · 널 풀(맥락 통과·쐐기 아님 · 깊이 띠) {fu['null_pool']:,} · W-nc 셋업 행 {fu['valid_nc']:,} · "
              f"진행 중 E 셋업 행 {fu['e_inprogress_valid']:,} · H0 전 20일 급등 > 30% 셋업 행 {fu['surge30_valid']:,}",
              f"- 팔 셋업 행: {fu['arms']}",
              f"- **W 체결 {r['W']}** (하위 {r['W_sub']}) · 연도별 {r['W_years']}",
              "- W 체결 기반 길이(H0 → s, 거래일) 최소/25%/중앙/75%/최대: " + ("/".join(f"{bl[q]:.0f}" for q in (0.0, 0.25, 0.5, 0.75, 1.0)) if bl else "—"),
              f"- W 체결 쐐기 안 수축 수: {r['W_n_contr']} · 체결가 → H0 거리 10/50/90%: "
              + ("/".join(f"{hd[q]:+.1%}" for q in (0.1, 0.5, 0.9)) if hd else "—"),
              f"- 돌파일 거래량/직전 50일 구간(0: < 1, 1: 1~1.5, 2: ≥ 1.5): {r['W_vol_buckets']} · 갭 체결(시가 ≥ 주문가) "
              f"{r['W_gap_fill']} · 진행 중 E 체결 {r['W_e_inprogress']}",
              "- H0 전 20일 급등률 10/50/90%: " + ("/".join(f"{r['W_surge_q'][q]:+.0%}" for q in (0.1, 0.5, 0.9))
                                                if r["W_surge_q"] else "—") + f" · > 30% {r['W_surge30']}/{r['W']} · d−1 RS "
              "10/50/90: " + ("/".join(f"{r['W_rs_q'][q]:.0f}" for q in (0.1, 0.5, 0.9)) if r["W_rs_q"] else "—"),
              f"- 개인 체결 가능성: ADV20 > 1,000 백만원 {r['W_liquid_1pct']}/{r['W']} · ADV20 중앙 {r['W_adv_median']:,.0f} 백만원",
              "- 주문·체결 결과(버림 사유): " + " · ".join(f"{n}: " + ", ".join(f"{k} {v}" for k, v in sorted(sk.items()))
                                                      for n, sk in r["skips"].items() if sk), ""]
    L += ["## 공통", "",
          f"- NXT 미보정 행이 있는 종목 {ctx.n_cut_stocks:,}개 — 보유 경로는 {R.NXT_LAST_DAY} 에서 `data_end` 로 끊는다.",
          "- 차트용 반전점: `wedge/chart_points.json` (하한별 W 체결 전부, 기하만).",
          "- 판정은 사람이 한다. 이 문서는 사전 규칙의 산술만 적는다.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# 널 작업자(fork, 직렬 커널만)
# ---------------------------------------------------------------------------

_GW: dict = {}


def _null_task(task: tuple) -> dict:
    """같은 메뉴 널 한 벌(정정 1 M5): 풀(맥락 통과 ∧ 쐐기 아님 ∧ 깊이 띠)의 종목-일마다 k ~ U{15..60}, j ~ U{5..15} 로 하락 기준선 셋업
    (:func:`wedge.null_lines` — 첫 관통·종가 < 선(d)·0.92 띠) → 매수정지 체결 가능(9.5%·3% 갭·상한가·정지)한 것 중 W 의 월별 체결 수 ×
    d 3분위만큼 비복원 추출(풀 < 3× 면 ±5 거래일) → 같은 청산(이틀 규칙)·p 혼합·보유 중 거르기."""
    arm, b = task
    g = _GW
    A, P = g["A"], g["plan"][arm]
    rng = np.random.default_rng([NULL_SEED, b])
    ci, s = P["ci"], P["s"]
    m = len(ci)
    k = rng.integers(NULL_K[0], NULL_K[1] + 1, size=m)
    j = rng.integers(NULL_J[0], NULL_J[1] + 1, size=m)
    nl_ = Wd.null_lines(A, ci, s, k, j, parallel=False)
    chk = V.buystop_entry_check(A, ci, s + 1, nl_["u_d"], nl_["l_last"], ARMS[arm]["params"])
    ok = nl_["ok"] & (chk["status"] == V.S_OK)
    terc = np.searchsorted(P["edges"], chk["dd"], side="right")
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
    base = {"arm": arm, "b": b, "short": short, "need": P["need_total"], "line_ok": int(nl_["ok"].sum()),
            "fillable": int(ok.sum())}
    if not sel:
        return {**base, "n": 0, "n_cut": 0}
    sel = np.unique(np.concatenate(sel))
    sel = sel[np.lexsort((ci[sel], s[sel]))]
    pid = nl_["pid"][sel]                                   # 같은 기준선(마지막 고점 행) 첫 체결 하나
    r = V.run_buystop(A, ci[sel], s[sel] + 1, nl_["u_d"][sel], nl_["l_last"][sel], pid, no_overlap=True,
                      params=ARMS[arm]["params"])
    okk = r["status"] == V.S_OK
    if not okk.any():
        return {**base, "n": 0, "n_cut": 0}
    cc = ci[sel][okk]
    mk = g["mkt"][cc]
    c_opt = g["cost_tab"][mk, r["x"][okk]]
    amb = r["amb"][okk]
    c0 = np.where(amb, g["cost_tab"][mk, (s[sel] + 1)[okk]], c_opt)
    ret, ret0 = r["ret"][okk], r["ret_stop0"][okk]
    p = g["p"]
    out = {**base, "n": int(okk.sum()), "selected": int(len(sel)), "amb_rate": float(amb.mean()),
           "n_cut": int(((r["reason"][okk] == V.R_DATA_END) & (r["x"][okk] < A.CUT[cc])).sum()),
           "pool_ratio_min": float(np.min(pool_ratio)), "pool_ratio_med": float(np.median(pool_ratio))}
    for tag, pp in (("", p), ("_cons", 1.0), ("_opt", 0.0)):
        out[f"net_mean{tag}"] = float((pp * (ret0 - c0) + (1 - pp) * (ret - c_opt)).mean())
        out[f"net2_mean{tag}"] = float((pp * (ret0 - 2 * c0) + (1 - pp) * (ret - 2 * c_opt)).mean())
    out["r_mean"] = float(np.mean((p * (ret0 - c0) + (1 - p) * (ret - c_opt)) / r["dd"][okk]))
    return out


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def _record_variants(out_dir: str) -> None:
    """정정 1 MINOR — 리뷰가 센 설계 변형 8개를 원장에 '건수만, 성과 미계산' 으로(지문 중복은 원장이 거른다)."""
    for v_ in DESIGN_VARIANTS:
        record_trial(LABEL, {"prereg": VARIANT_PREREG, "design_variant": v_, "evaluated": "건수만, 성과 미계산",
                             "source": "REVIEW_WEDGE_TEST.md 1단계 0. 설계 변형 체결 수"},
                     logs_dir=out_dir, note=f"쐐기 설계 변형(건수만): {v_}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(--out 을 탐색 디렉터리 밖으로)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--factors", default=R.FACTORS_PKL)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--p-hi", default=None)
    ap.add_argument("--counts-only", action="store_true")
    ap.add_argument("--adv-floor", default=None, help="전체: 생략하면 wedge/FUNNEL.md 사전 규칙 값(VOID 면 5000 기술). --funnel-only: 쉼표 목록")
    ap.add_argument("--funnel-only", action="store_true")
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
    _record_variants(out_dir)

    if args.funnel_only:
        floors = [float(x) for x in (args.adv_floor or ",".join(str(int(f)) for f in ADV_FLOORS_FUNNEL)).split(",")]
        ctx = WCtx(t0)
        res = {}
        for f in floors:
            res[f] = funnel_counts(ctx, f, lo_all, hi, lo_of)
            _log(f"하한 {f:,.0f}: W {res[f]['W']} (하위 {res[f]['W_sub']}) · " + " · ".join(
                f"{n} {res[f][n]}" for n in ARMS if n != "W"), t0)
        ok = [f for f, r in res.items() if r["W"] >= FLOOR_RULE_N and r["W_sub"] >= FLOOR_RULE_SUB]
        rule = (max(ok), False) if ok else (None, True)
        path = os.path.join(sub_dir, "FUNNEL.md")
        write_funnel(path, res, lo_all, hi, rule, time.time() - t0, ctx)
        with open(os.path.join(sub_dir, "chart_points.json"), "w", encoding="utf-8") as fh:
            json.dump({f"{int(f)}": r["chart"] for f, r in res.items()}, fh, ensure_ascii=False, indent=1)
        print(f"사전 규칙 → 하한 {rule[0]} · W VOID {rule[1]} → {path}")
        return 0

    if args.adv_floor is not None:
        floor, void = float(args.adv_floor), None
    else:
        floor, void = R.read_floor_rule(FUNNEL_MD)
        if floor is None and void is None:
            ap.error(f"--adv-floor 를 주거나 먼저 --funnel-only 로 {FUNNEL_MD} 를 만든다")
        if floor is None:
            floor = R.ADV_FLOOR                        # 정정 1 C1: VOID 여도 기술 모드로 끝까지(50억)
    timings: dict[str, float] = {}
    fac_sha = R._sha(args.factors)
    cfgs = {n: _arm_config(n, lo_of[n], hi, floor, fac_sha) for n in ARMS}
    for n, c in cfgs.items():
        record_trial(LABEL, c, logs_dir=out_dir, note=f"쐐기 팔 {n}: {ARMS[n]['desc']}")
    for cell in SENS_CELLS:
        record_trial(LABEL, {**cfgs[PRIMARY], "sensitivity": cell}, logs_dir=out_dir, note=f"쐐기 민감도: {cell}")
    n_led = count_trials(LABEL, logs_dir=out_dir)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
    n_dsr = n_led + sum(n_prior.values())

    ctx = WCtx(t0)
    A, days, codes, mkt = ctx.A, ctx.days, ctx.codes, ctx.mkt
    cost_tab = R.cost_table(days)
    timings["load"] = time.time() - t0
    B = build_wedge(ctx, floor, lo_all, hi, lo_of)
    S, h5, funnel = B["S"], B["h5"], B["funnel"]
    funnel["floor_rule_void"] = void
    _log(f"하한 {floor:,.0f} · 셋업 후보 {len(S):,} · 쐐기 {funnel['valid']:,} · 팔 {funnel['arms']}", t0)
    mom = momentum_12_1(ctx.p.close)
    q_mom = day_market_quantiles(mom, h5, ctx.p.market, n_q=MOM_N_Q).to_numpy(float)
    adv2 = day_market_median_half(ctx.D.adv20, h5, ctx.p.market).to_numpy(float)
    del mom

    t1 = time.time()
    pinfo = R._estimate_p(ctx.con, A, h5, days, args.p_hi, band_lo=NULL_BAND, min_age=0, params=P_W)
    pval = pinfo["p"]
    timings["p_minute"] = time.time() - t1

    t1 = time.time()
    raw, skips = simulate_all(ctx, B)
    trades = {n: R._finish(entry_desc(ctx, raw[n]), cost_tab, mkt, A.CUT) for n in ARMS}
    pinfo["p_reweighted"] = R._p_reweight(pinfo, trades["W"])
    timings["sim"] = time.time() - t1
    _log("팔 체결: " + " · ".join(f"{n} {len(t)}" for n, t in trades.items()), t0)

    # --- 같은 메뉴 널(W): 하락 기준선 매수정지 ---
    t1 = time.time()
    v = S["valid"]
    valid_keys = set(zip(S.loc[v, "col_idx"].to_numpy(), S.loc[v, "sig_idx"].to_numpy(), strict=True))
    month_rows = R._month_rows(days)
    plan = {}
    t = trades["W"]
    if not t.empty:
        pm = (S["context"] & ~v & B["rng"]["W"]
              & S["depth_any"].between(Wd.WEDGE_PARAMS["depth_lo"], Wd.WEDGE_PARAMS["depth_hi"])).to_numpy(bool)
        pci, pss = S["col_idx"].to_numpy()[pm], S["sig_idx"].to_numpy()[pm]
        assert not any((c_, s_) in valid_keys for c_, s_ in zip(pci, pss, strict=True)), "널 풀에 쐐기 셋업 (종목, 날짜)"
        dd = t["d"].to_numpy(float)
        edges = np.quantile(dd, [1 / 3, 2 / 3]) if len(t) >= 3 else np.array([np.inf, np.inf])
        terc = np.searchsorted(edges, dd, side="right")
        mo = t["order_day"].dt.to_period("M").astype(str).to_numpy()
        need = pd.Series(1, index=pd.MultiIndex.from_arrays([mo, terc])).groupby(level=[0, 1]).sum().to_dict()
        plan["W"] = {"ci": pci.astype(np.int64), "s": pss.astype(np.int64), "edges": edges, "need": need,
                     "need_total": int(len(t)), "month_rows": month_rows, "pool_rows": int(pm.sum())}
    _GW.update({"A": A, "plan": plan, "cost_tab": cost_tab, "mkt": mkt, "p": pval})
    tasks = [(n, b) for n in plan for b in range(args.n_null)]
    rows_null: list = []
    if tasks:
        if args.workers > 1:
            with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as ex:
                rows_null = list(ex.map(_null_task, tasks, chunksize=4))
        else:
            rows_null = [_null_task(tk) for tk in tasks]
    nulls = pd.DataFrame(rows_null)
    timings["null"] = time.time() - t1

    t1 = time.time()
    moms = {n: R._momentum_control(trades[n], A, h5, q_mom, adv2, mkt, days, codes, pval, ARMS[n]["params"])
            for n in BAR_ARMS}
    timings["matched"] = time.time() - t1
    F = pd.read_pickle(args.factors) if os.path.exists(args.factors) else None
    kospi = R._naver_returns("KOSPI")
    ports = {n: R._portfolio(trades[n], A, days, lo_of[n], hi, F, kospi, pval) for n in ARMS}

    rows = {}
    for n in ARMS:
        tn = trades[n]
        s = {"arm": n, "n_setups": funnel["arms"].get(n, 0), "skips": skips[n], "n_fills": len(tn)}
        for tag, pp in (("", pval), ("cons_", 1.0), ("opt_", 0.0)):
            s.update({f"{tag}{k}": v_ for k, v_ in R.wstats(R.expand(tn, pp), n_dsr).items()})
        s.update(_desc(ctx, tn))
        if moms.get(n):
            s.update(moms[n])
        s.update(ports[n]["stats"])
        rows[n] = s
    if not nulls.empty:
        z = nulls[nulls["n"] > 0]
        nstat = {"null_k": int(len(z)), "null_p50": float(np.nanmedian(z["net2_mean"])) if len(z) else np.nan,
                 "null_fill_ratio_med": float((nulls["n"] / nulls["need"]).median()),
                 "null_cut_med": float(nulls["n_cut"].median()),
                 "null_pool_med": float(z["pool_ratio_med"].median()) if len(z) else np.nan,
                 "null_r_p95": float(np.nanpercentile(z["r_mean"], 95)) if len(z) else np.nan,
                 "null_pool_rows": plan["W"]["pool_rows"]}
        for tag in ("", "_cons", "_opt"):
            nstat[f"null_p95{tag}"] = float(np.nanpercentile(z[f"net2_mean{tag}"], 95)) if len(z) else np.nan
        for n in BAR_ARMS:                              # W1d 도 같은 널 p95 와 비교(널은 이틀 규칙 — 기술 주석)
            rows[n].update(nstat)
            rows[n]["null_rank"] = (float((z["net2_mean"] < rows[n].get("net2_mean", np.nan)).mean()) if len(z) else np.nan)
    for n in BAR_ARMS:
        w = rows[n]
        for tag, nt in (("", ""), ("cons_", "_cons"), ("opt_", "_opt")):
            sub = {k[len(tag):]: v_ for k, v_ in w.items() if k.startswith(tag)} if tag else w
            b = wedge_bars(sub, w.get(f"null_p95{nt}", np.nan), w.get(f"{tag}mom_lo", np.nan), w.get("fac_alpha_t", np.nan))
            w.update({f"{tag}bar_{k}": v_ for k, v_ in b.items()})
    sens = _sensitivities(trades["W"], pval, n_dsr, rows["W"], pinfo.get("p_reweighted", np.nan))
    be1 = _be1_pairs(ctx, trades["W"], trades["W-be1"], pval)
    vol_tab = {k: R.wstats(R.expand(g, pval)) for k, g in trades["W"].groupby("bo_vol_bucket")} if len(trades["W"]) else {}
    surge_tab = ({k: R.wstats(R.expand(g, pval)) for k, g in trades["W"].groupby(trades["W"]["surge20"] > 0.30)}
                 if len(trades["W"]) else {})

    elapsed = time.time() - t0
    L = render(rows, funnel, sens, be1, vol_tab, surge_tab, pinfo, ports, lo_of, hi, n_led, n_prior, n_dsr, out_dir, debug,
               elapsed, timings, ctx, args, fac_sha, floor)
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    for n, tn in trades.items():
        tn.to_csv(os.path.join(sub_dir, f"trades_{n}.csv"), index=False)
    if not nulls.empty:
        nulls.to_csv(os.path.join(sub_dir, "null_draws.csv"), index=False)
    pd.DataFrame([{k: v_ for k, v_ in r.items() if not isinstance(v_, dict)} for r in rows.values()]).to_csv(
        os.path.join(sub_dir, "arms.csv"), index=False)
    with open(os.path.join(sub_dir, "chart_points.json"), "w", encoding="utf-8") as fh:
        json.dump({f"{int(floor)}": chart_points(ctx, trades["W"], with_exit=True) if len(trades["W"]) else []}, fh,
                  ensure_ascii=False, indent=1)
    S.loc[S["valid"]].to_csv(os.path.join(sub_dir, "setups_valid.csv"), index=False)

    print(f"\n=== 하락 쐐기 — {elapsed / 60:.1f}분 · 하한 {floor:,.0f} · 원장 {n_led}줄 · DSR N {n_dsr} ===")
    print("시간(s): " + " · ".join(f"{k} {v_:.1f}" for k, v_ in timings.items()))
    print(f"셋업 후보 {funnel['base']:,} · 쐐기 통과 {funnel['wedge_ok']:,} · 맥락 통과 {funnel['valid']:,} · W-nc {funnel['valid_nc']:,}"
          f" · 널 풀 {funnel['null_pool']:,} · 분봉 p 판별 {pinfo['n_classified']:,}")
    for n, s in rows.items():
        line = f"팔 {n}: 셋업 행 {s['n_setups']:,} · 체결 {s['n_fills']:,} (모호 {s.get('amb_n', 0)}, NXT 절단 {s.get('nxt_cut', 0)})"
        if n in BAR_ARMS:
            line += f" · 널 {s.get('null_k', 0)}벌 · 정합 짝 {s.get('mom_pairs', 0)}"
        if not args.counts_only:
            line += f" · 순 2배 {_pct(s.get('net2_mean', np.nan))}"
        print(line)
    if not nulls.empty:
        print(f"널 벌 체결 중앙 {nulls['n'].median():.0f}/{nulls['need'].median():.0f} · 기준선 셋업 중앙 {nulls['line_ok'].median():.0f}")
    print(f"W-be1 짝 {be1.get('n', 0)} · → {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def _desc(ctx: WCtx, t: pd.DataFrame) -> dict:
    """진입·보유 기술(d 분포·H0 거리·보유 중 H0 돌파·모호·NXT 절단)."""
    if t is None or t.empty:
        return {"amb_n": 0, "nxt_cut": 0}
    H = ctx.A.H
    br = [bool(np.nanmax(H[c, d:x + 1]) >= h) for c, d, x, h in
          zip(t["col_idx"].astype(int), t["d_row"].astype(int), t["x_row"].astype(int), t["h0"], strict=True)] if "h0" in t else []
    return {"amb_n": int(t["amb"].sum()), "nxt_cut": int(t["nxt_cut"].sum()),
            "d_q": {q: float(t["d"].quantile(q)) for q in (0.1, 0.5, 0.9)},
            "h0_dist_q": {q: float(t["h0_dist"].quantile(q)) for q in (0.1, 0.5, 0.9)} if "h0_dist" in t else {},
            "h0_breach": float(np.mean(br)) if br else np.nan, "gap_rate": float(t["gap_fill"].mean()),
            "censored": int((t["exit_reason"] == "data_end").sum())}


def _sensitivities(t: pd.DataFrame, pval: float, n_dsr: int, s0: dict, p_rw: float) -> dict:
    out = {}
    if t.empty:
        return out
    for name, mk, comp in (("완결 거래만", None, True), (f"사건 ≤ {CAP_SENS}", t["order_day"] <= pd.Timestamp(CAP_SENS), False)):
        sub = t if mk is None else t[mk.to_numpy()]
        res = {}
        for tag, pp in (("", pval), ("cons_", 1.0), ("opt_", 0.0)):
            x = R.expand(sub, pp)
            if comp and len(x):
                x = x[x["complete"].to_numpy(bool)]
            st = R.wstats(x, n_dsr)
            b = wedge_bars(st, s0.get("null_p95", np.nan), s0.get("mom_lo", np.nan), np.nan)
            res.update({f"{tag}{k}": v for k, v in st.items()})
            res.update({f"{tag}bar_{k}": v for k, v in b.items()})
        out[name] = res
    if np.isfinite(p_rw):
        st = R.wstats(R.expand(t, p_rw), n_dsr)
        b = wedge_bars(st, s0.get("null_p95", np.nan), s0.get("mom_lo", np.nan), np.nan)
        out[f"p 재가중 {p_rw:.3f}"] = {**st, **{f"bar_{k}": v for k, v in b.items()}}
    return out


def _be1_pairs(ctx: WCtx, a: pd.DataFrame, b: pd.DataFrame, pval: float) -> dict:
    """정정 0b — W 와 W-be1 의 같은 쐐기(종목, ip) 짝: 순(p 혼합)·R·승률 차이(날짜군집 CI)와 '한때 +1R 이상 → 손실' 수(각 팔, 낙관 경로)."""
    if a.empty or b.empty:
        return {"n": 0}

    def keyed(t):
        x = t.assign(mix=pval * t["net_stop0"] + (1 - pval) * t["net"]).set_index(["code", "ip"])
        return x[~x.index.duplicated()]

    def turned(t):
        H = ctx.A.H
        mfe = np.array([np.nanmax(H[c, d:x + 1]) for c, d, x in
                        zip(t["col_idx"].astype(int), t["d_row"].astype(int), t["x_row"].astype(int), strict=True)])
        return int(((mfe >= t["entry_px"] * (1 + t["d"])) & (t["net"] < 0)).sum())

    ka, kb = keyed(a), keyed(b)
    common = ka.index.intersection(kb.index)
    out = {"n": int(len(common)), "turned_W": turned(a), "turned_be1": turned(b)}
    if not len(common):
        return out
    da, db = ka.loc[common], kb.loc[common]
    cl = da["order_day"].dt.strftime("%Y%m%d").to_numpy()
    for name, va, vb in (("net", da["mix"], db["mix"]), ("R", da["mix"] / da["d"], db["mix"] / db["d"]),
                         ("win", (da["mix"] > 0).astype(float), (db["mix"] > 0).astype(float))):
        diff = vb.to_numpy() - va.to_numpy()
        lo, hi_, _ = cluster_bootstrap_ci(diff, cl, n_boot=N_BOOT, seed=SEED)
        out.update({f"{name}_W": float(va.mean()), f"{name}_be1": float(vb.mean()), f"{name}_diff": float(diff.mean()),
                    f"{name}_lo": lo, f"{name}_hi": hi_})
    return out


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _m(ok) -> str:
    return "✓" if ok else "✗"


def _r(x, fmt="{:.2f}") -> str:
    return R._r(x, fmt)


def _sh(x) -> str:
    return R._sh(x)


def render(rows, funnel, sens, be1, vol_tab, surge_tab, pinfo, ports, lo_of, hi, n_led, n_prior, n_dsr, out_dir, debug,
           elapsed, timings, ctx, args, fac_sha, floor) -> list[str]:
    pval = pinfo["p"]
    void = funnel.get("floor_rule_void")
    L = [f"# 천장 아래 하락 쐐기 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         f"사전등록 `{PREREG_DOC}` 의 정의·팔·대조·널·합격 바를 그대로 계산한 숫자다. ✓/✗ 는 산술 비교일 뿐이다. **정정 1 C1 사전 약속:** "
         "건수 규칙이 VOID 면 판정 없는 기술이고, 결과를 본 뒤 이 데이터에서 정의·청산·하한을 바꿔 다시 돌리지 않는다. 판정이 있다면 "
         "W(이틀)·W1d(하루) 둘 다 통과해야 한다. 재현: `uv run --with numba --extra pg python research/experiments/wedge_test_run.py`.", "",
         f"**유니버스 하한:** ADV20 ≥ {floor:,.0f} 백만원 — " + (
             "`wedge/FUNNEL.md` 사전 규칙 값." if void is False else "**사전 규칙상 W VOID** — 아래는 전부 기술(판정 없음)."
             if void is True else "명시 인자."), ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 {min(lo_of.values())}~{hi}, 널 {args.n_null}벌, p 구간 {pinfo.get('window', '—')}, "
              f"출력·원장 `{out_dir}`. 측정으로 쓰지 않는다.", ""]
    L += [f"## 합격 바 산술 — W(이틀) · W1d(하루) 같은 표 (주 = 분봉 p 혼합, p = {pval:.3f})", "",
          "| 바 | 문턱 | W | W1d |", "|---|---|---|---|"]

    def cell(w, key, txt):
        return f"{txt} {_m(w.get(key))}" if w.get("n") else "—"

    a, b = rows["W"], rows["W1d"]
    L += [f"| 1 표본 | n ≥ {B1_N} ∧ ≤2024-12 n ≥ {B1_SUB_N} | " + " | ".join(
              cell(w, "bar_b1", f"n {w.get('n', 0)} · 하위 {w.get('n_sub', 0)}") for w in (a, b)) + " |",
          "| 2 순수익(2배) | > 0 ∧ 하위 > 0 ∧ 2020~21 제외 > 0 | " + " | ".join(
              cell(w, "bar_b2", f"{_pct(w.get('net2_mean', np.nan))} · {_pct(w.get('net2_sub', np.nan))} · "
                                f"{_pct(w.get('net2_ex2021', np.nan))}") for w in (a, b)) + " |",
          "| 3 같은 메뉴 널(2배) | > 하락 기준선 널 p95 | " + " | ".join(
              cell(w, "bar_b3", f"{_pct(w.get('net2_mean', np.nan))} vs {_pct(w.get('null_p95', np.nan))} (백분위 "
                                f"{_sh(w.get('null_rank', np.nan))})") for w in (a, b)) + " |",
          "| 4 모멘텀 정합 | α CI 하단 > 0 | " + " | ".join(
              cell(w, "bar_b4", f"{_pct(w.get('mom_alpha', np.nan))} [{_pct(w.get('mom_lo', np.nan))}, "
                                f"{_pct(w.get('mom_hi', np.nan))}]") for w in (a, b)) + " |",
          "| 5 손실·승률(1배) | 손익비 ≥ 2 ∧ 평균 손실 ≥ −1R ∧ R 5% ≥ −1.3R ∧ 순<−12% ≤ 5% ∧ 승률(본전 제외) > 1/(1+손익비) + 5%p | "
          + " | ".join(cell(w, "bar_b5", f"{_r(w.get('gl_ratio', np.nan))} · {_r(w.get('avg_loss_R', np.nan), '{:+.2f}')}R · "
                                         f"{_r(w.get('r_p5', np.nan), '{:+.2f}')}R · {_sh(w.get('big_loss', np.nan))} · 승률 "
                                         f"{_sh(w.get('bar_win_ex_be', np.nan))} vs {_sh(w.get('bar_be_need', np.nan))}")
                       for w in (a, b)) + " |",
          "| 산술 충족(주/보수/낙관) | 1~5 | " + " | ".join(
              f"**{w.get('bar_n_met')}/5** · {w.get('cons_bar_n_met')} · {w.get('opt_bar_n_met')}" for w in (a, b)) + " |",
          "", f"바 6(기술): 팩터 α t — W {_r(a.get('fac_alpha_t', np.nan))} (극단일 제외 {_r(a.get('fac_alpha_t_noext', np.nan))}) · "
          f"W1d {_r(b.get('fac_alpha_t', np.nan))}. 널은 이틀 규칙이라 W1d 의 바 3 은 청산이 다른 비교다(기술).", ""]
    L += ["### 민감도 (W)", "", "| 칸 | n | 2배 순 | 손익비 | 승률(본전 제외) | 충족 주/보수/낙관 |", "|---|---|---|---|---|---|"]
    for name, s in sens.items():
        L.append(f"| {name} | {s.get('n', 0)} | {_pct(s.get('net2_mean', np.nan))} | {_r(s.get('gl_ratio', np.nan))} | "
                 f"{_sh(win_ex_be(s))} | {s.get('bar_n_met', '—')}/{s.get('cons_bar_n_met', '—')}/{s.get('opt_bar_n_met', '—')} |")
    ci = pinfo.get("ci", (np.nan, np.nan))
    st = funnel["status"]
    L += ["", "## 진입일 손절 순서 — 분봉 p", "",
          f"구간 {pinfo.get('window')}, 합성 매수정지(수평 k일 고가 피벗, 근접 띠 0.92·9.5%/3% 갭 기준 = 주문가), 확정 손절 "
          f"{pinfo.get('n_confirmed', 0):,} 분모 밖, 모호 {pinfo['n_events']:,} → 판별 {pinfo['n_classified']:,} · {pinfo.get('counts', {})} → "
          f"**p = {pval:.3f}** [{_r(ci[0], '{:.3f}')}, {_r(ci[1], '{:.3f}')}] · 재가중 {_r(pinfo.get('p_reweighted', np.nan), '{:.3f}')}.", "",
          "## 깔때기", "",
          f"- 실행 {elapsed / 60:.1f}분 (" + " · ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + f") · 원장 {n_led}줄(팔 {len(ARMS)} · "
          f"민감도 {len(SENS_CELLS)} · 설계 변형 {len(DESIGN_VARIANTS)} 건수만) · DSR N {n_dsr}",
          f"- 셋업 후보 {funnel['base_nosplit']:,} − 분할 {funnel['split_excl']:,} = {funnel['base']:,} → 쐐기 첫 탈락: "
          + " · ".join(f"{k} {v:,}" for k, v in sorted(st.items(), key=lambda kv: Wd.WEDGE_STATUS.index(kv[0]))),
          f"- 쐐기 통과 {funnel['wedge_ok']:,} − H0 맥락 탈락 {funnel['context_fail']:,} = **{funnel['valid']:,}** (진행 중 E "
          f"{funnel['e_inprogress_valid']:,} · H0 전 20일 급등 > 30% {funnel['surge30_valid']:,}) · W-nc {funnel['valid_nc']:,} · "
          f"널 풀 {funnel['null_pool']:,} · 팔 셋업 행 {funnel['arms']}",
          f"- factors.pkl sha256 `{fac_sha}`", "",
          "## 전 팔", "",
          "| 팔 | 셋업 행 | 체결 | n | 승/본전/패 | 손익비 | 평균 손실 R | R 5% | 순 1배 | 보수/낙관 1배 | 순 2배 | 돌파 실패 | 손절 | 모호 | "
          "갭 체결 | NXT 절단 | H0 거리 10/50/90 | 보유 중 H0 돌파 | 정합 α [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in rows.items():
        if not s.get("n"):
            L.append(f"| {n} | {s['n_setups']} | 0 |" + " |" * 16)
            continue
        ex = s.get("exits", {})
        hq = s.get("h0_dist_q", {})
        L.append(f"| {n} | {s['n_setups']} | {s['n_fills']} | {s['n']} | {_sh(s['win'])}/{_sh(s['be'])}/{_sh(s['lossc'])} | "
                 f"{_r(s['gl_ratio'])} | {_r(s['avg_loss_R'], '{:+.2f}')} | {_r(s['r_p5'], '{:+.2f}')} | {_pct(s['net_mean'])} | "
                 f"{_pct(s.get('cons_net_mean', np.nan))}/{_pct(s.get('opt_net_mean', np.nan))} | {_pct(s['net2_mean'])} | "
                 f"{ex.get('failed_breakout', 0):g} | {ex.get('stop', 0) + ex.get('stop_entry', 0):g} | {s.get('amb_n', 0)} | "
                 f"{_sh(s.get('gap_rate', np.nan))} | {s.get('nxt_cut', 0)} | "
                 + ("/".join(f"{hq[q]:+.1%}" for q in (0.1, 0.5, 0.9)) if hq else "—")
                 + f" | {_sh(s.get('h0_breach', np.nan))} | {_pct(s.get('mom_alpha', np.nan))} [{_pct(s.get('mom_lo', np.nan))}, "
                 f"{_pct(s.get('mom_hi', np.nan))}] |")
    L += ["", "## W-be1 vs W (정정 0b, 같은 쐐기 짝)", ""]
    if be1.get("n"):
        L += ["| 지표 | W | W-be1 | 차이 [날짜군집 95% CI] |", "|---|---|---|---|"]
        for k, nm, fmt in (("net", "거래당 순 1배(p 혼합)", _pct), ("R", "거래당 R", lambda x: _r(x, "{:+.3f}")),
                           ("win", "승률(순 > 0)", _sh)):
            L.append(f"| {nm} | {fmt(be1[f'{k}_W'])} | {fmt(be1[f'{k}_be1'])} | {fmt(be1[f'{k}_diff'])} "
                     f"[{fmt(be1[f'{k}_lo'])}, {fmt(be1[f'{k}_hi'])}] |")
    L += [f"짝 {be1.get('n', 0)} · 한때 +1R 이상이었다가 손실로 끝난 거래(낙관 경로): W {be1.get('turned_W', 0)} · W-be1 "
          f"{be1.get('turned_be1', 0)}", "", "## 기술 분해 (W)", "",
          "| 구분 | 구간 | n | 순 1배 | 손익비 | 승률(본전 제외) |", "|---|---|---|---|---|---|"]
    names = {-1: "모름", 0: "< 1", 1: "1~1.5", 2: "≥ 1.5"}
    for k, s in sorted(vol_tab.items()):
        L.append(f"| 돌파일 거래량/50일 | {names.get(int(k), k)} | {s.get('n', 0)} | {_pct(s.get('net_mean', np.nan))} | "
                 f"{_r(s.get('gl_ratio', np.nan))} | {_sh(win_ex_be(s))} |")
    for k, s in sorted(surge_tab.items()):
        L.append(f"| H0 전 20일 급등 > 30% | {'예' if k else '아니오'} | {s.get('n', 0)} | {_pct(s.get('net_mean', np.nan))} | "
                 f"{_r(s.get('gl_ratio', np.nan))} | {_sh(win_ex_be(s))} |")
    w = rows["W"]
    L += ["", "## 같은 메뉴 널 — 하락 기준선 매수정지 (정정 1 M5)", "",
          f"풀 {w.get('null_pool_rows', 0):,} 종목-일(맥락 통과 ∧ 쐐기 아님 ∧ 깊이 띠) · 벌 {w.get('null_k', 0)} · 중앙 {_pct(w.get('null_p50', np.nan))} · "
          f"p95 {_pct(w.get('null_p95', np.nan))} (보수 {_pct(w.get('null_p95_cons', np.nan))} / 낙관 {_pct(w.get('null_p95_opt', np.nan))}) · "
          f"체결/필요 중앙 {_sh(w.get('null_fill_ratio_med', np.nan))} · 풀/필요 중앙 {_r(w.get('null_pool_med', np.nan), '{:.1f}')} · NXT "
          f"절단 중앙 {_r(w.get('null_cut_med', np.nan), '{:.0f}')} · 실제 R {_r(w.get('r_mean', np.nan), '{:+.3f}')} vs 널 R p95(거래별 d 기준) "
          f"{_r(w.get('null_r_p95', np.nan), '{:+.3f}')}", "",
          "## 10 슬롯 포트폴리오 (바 6 기술)", "", "| 팔 | 편입 | 노출 | CAGR 1배 | MDD | 초과 Sharpe | 팩터 α t |", "|---|---|---|---|---|---|---|"]
    for n in ARMS:
        s = ports[n]["stats"]
        if s.get("port_taken"):
            L.append(f"| {n} | {s['port_taken']} | {_sh(s['port_exposure'])} | {_pct(s['port_cagr'])} | {_pct(s['port_mdd'])} | "
                     f"{_r(s.get('port_sharpe_exc', np.nan))} | {_r(s.get('fac_alpha_t', np.nan))} |")
    L += ["", "## 구현 메모 (결과 전 결정)", "",
          "- 단계 2 맥락: d−1 MA 정렬·MA200 상승·breadth 는 셋업 후보 거르기, H0 일 추세 템플릿·RS 는 쐐기 스캔 뒤 i0 에서.",
          "- 쐐기 정의·해석은 `research/signals/wedge.py` 모듈 설명 1~16(정정 0·1·2 — 모양 조건은 다섯 점 A~E 에만, H0 → A 구간 제약 없음). "
          "쐐기 식별 = (종목, H0, D) — "
          "진행 중 E 가 내려가도 같은 쐐기라 첫 체결 하나. W-F 의 쐐기 기간 = A 의 행 ~ s.",
          "- 매수정지·청산은 `vcp.simulate_buystop`(``ref_order``·``fb_days`` 2·W-be1 은 ``be_r`` 1). 돌파 기준가 = U(d) 고정, 진입일 종가가 첫날. "
          "진입일 모호 = 시가 ≤ U(d) ∧ 저가 ≤ 손절가 ∧ 종가 > 손절가. 정합 대조도 같은 청산.",
          "- 널: 하락 기준선 = 직전 k 일 창 최고 고가에서 지그재그(θ 5%) → 마지막 두 확정 고점 직선(하락, 간격 ≥ 3일), 손절 = 직전 j 일 최저 "
          "저가, 첫 관통·종가 < 선(d)·0.92 띠·9.5%·3% 갭·이틀 규칙. 분봉 p 의 합성 사건은 수평 피벗(같은 진입일 모호 정의).",
          "- 바 5 승률 = 승 / (승 + 패)(본전 |순| < 0.5% 제외), 손익비 = 평균 이익 / |평균 손실|(1배 순).",
          "- '한때 +1R 이상 → 손실' = 낙관 경로 [진입일, 청산일] 최고 고가 ≥ 진입가 × (1 + d) ∧ 순 1배 < 0. 보유 중 H0 돌파 = 같은 창 최고 고가 ≥ H0.",
          "- 돌파일 거래량 = 주문일 거래량 / 직전 50일 평균 — 진입 조건이 아니라 기술 표. H0 전 20일 급등률 = H0 / 종가(i0 − 20) − 1.",
          f"- 원장에 리뷰 설계 변형 {len(DESIGN_VARIANTS)}개를 '건수만, 성과 미계산' 으로 남겼다: " + " · ".join(DESIGN_VARIANTS), ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
