#!/usr/bin/env python
"""하락 쐐기 바닥(D) 진입 — A = H0, 네 점 A·B·C·D, D 봉 고가 매수정지. 사전등록 러너(**탐색 표시**) — 리포터, 판정 아님.

사전등록: ``research/logs/sepa_explore/WEDGE_D_TEST.md`` (2154f77 + 정정 1 b008fce, 결과 보기 전 · 탐색 표시 — 사용자가 결과가 붙은 쐐기
차트 12건을 본 뒤 진입·청산을 함께 바꾼 정의라, 바를 넘어도 "전진 페이퍼·독립 표본 확인 후보" 까지만, 그것도 **열람 제외 표본**에서도 바 2·3
산술 충족일 때만). 정정 1 (a): 문구 유지 · 기술 전용. 숫자를 ``WEDGE_D_RESULTS.md`` 에 적고 바 문턱과의 산술 비교만 한다.

- 자료·유니버스·비용·NXT·분할·맥락(H0 일 추세 템플릿 ∧ RS ≥ 70, d−1 MA 정렬·breadth)은 쐐기 시험 러너(``wedge_test_run``)와 같다(import).
- 셋업: :func:`wedge.wd_scan_arrays` (모듈 하단 a~h). 진입: D 봉 고가 + 1호가 매수정지(9.5%·3% 갭 기준 = 주문가), 손절 = D − 1호가,
  2R 전 돌파 실패 규칙 없음(:data:`wedge.WD_BUYSTOP`), 2R 절반 → 본전 + MA50 추적, 120일. 진입일 모호성 = 분봉 p 혼합(바닥 반전 합성 사건으로
  다시 잰다).
- 팔: WD(주, 탐색) · WD-be1 · WD-up(같은 모양·같은 청산, 진입만 상단선 A–C 첫 돌파 매수정지 — 정정 1 M7) · WD-nc(수렴 조건 끔) ·
  WD-nocap(120 거래일 상한 없음, 조건 청산만 — 정정 1 MINOR).
- 같은 메뉴 널(정정 1 M6): 맥락 통과 ∧ WD 아님 종목-일, 직전 k 일(k ~ U{10..40}) 최고 고가 이후 되돌림의 새 저점(눌림 8~40%) 봉 고가 매수정지,
  두 번째 관통 차단·종가 < 주문가, 어제가 최저가인 날 우선, 층화 = 월 × 손절폭 3분위 × 주문가/MA50 3분위.
- **열람 제외 표본(정정 1 C2):** 쐐기 차트 12건 종목의 열람 창(``wedge_d/viewed_windows.csv`` 첫 봉~마지막 봉) ± 60 거래일과 보유 기간이 겹치는
  체결을 뺀 표본 — 모든 표에 함께. R(총, 비용 전) = (청산가 − 체결가)/(체결가 − 손절가) = 총수익/d, 순 R = 순/d(정정 1 M5); 바 5 의 R 바는 총 R.
- ``--funnel-only``: ADV20 하한 50억·30억·20억·10억 WD 체결 **건수만** → ``wedge_d/FUNNEL.md`` (≥ 80 ∧ ≤ 2024-12 ≥ 40 인 가장 높은 하한,
  아니면 VOID) + 차트 JSON(기하·봉은 주문일까지).

Run:  uv run --with numba --extra pg python research/experiments/wedge_d_run.py --funnel-only
      uv run --with numba --extra pg python research/experiments/wedge_d_run.py --workers 12
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
import wedge_test_run as WR  # noqa: E402
from sepa_confluence_gate import _log, _pct  # noqa: E402
from sepa_momentum import MOM_N_Q, day_market_median_half, day_market_quantiles, momentum_12_1  # noqa: E402

from research.signals import vcp as V  # noqa: E402
from research.signals import wedge as Wd  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "wedge_d"
SUB = "wedge_d"
RESULTS_MD = "WEDGE_D_RESULTS.md"
FUNNEL_MD = "research/logs/sepa_explore/wedge_d/FUNNEL.md"
PREREG_DOC = "research/logs/sepa_explore/WEDGE_D_TEST.md (2154f77 + 정정 1 b008fce, 탐색 표시)"
VIEWED_CSV = "research/logs/sepa_explore/wedge_d/viewed_windows.csv"
VIEWED_PAD = 60                                   # 열람 창 ± 60 거래일
PRIOR_LEDGERS = (*WR.PRIOR_LEDGERS, ("wedge", "research/logs/sepa_explore"))
CHAIN = "VCP(결과 차트) → 쐐기(결과 차트 12건) → 쐐기 D(진입·청산 동시 변경, 사용자 지시 예외 · 탐색)"
#: 리뷰(REVIEW_WEDGE_D_TEST.md 1단계)가 센 설계 변형 8개 — 원장에 '건수만, 성과 미계산'.
DESIGN_VARIANTS = ("WD 문구대로", "WD + 소비 차단(M3)", "WD + D ≤ 0.99B(M1)", "WD-nc(수렴 끔)", "WD, d − iD 제한 없음(≤ 60)",
                   "담기 완화(M2)", "담기 완화 + 소비 차단", "담기 완화 + 소비 차단 + D ≤ 0.99B")

EVENT_LO, EVENT_HI, FLOW_LO, SUB_HI, CAP_SENS = R.EVENT_LO, R.EVENT_HI, R.FLOW_LO, R.SUB_HI, R.CAP_SENS
RS_MIN, BREADTH_MIN = 70.0, 0.5
ADV_FLOORS_FUNNEL = (eok(50), eok(30), eok(20), eok(10))
FLOOR_RULE_N, FLOOR_RULE_SUB = 80, 40
N_NULL, NULL_K, NULL_PULL, NULL_PULL_HI, NULL_SEED = 300, (10, 40), 0.08, 0.40, 20261012
NULL_EXPAND, NULL_POOL_MULT = 5, 3
CHART_PRE, CHART_POST = 60, 25                    # 차트 봉: H0 − 60 ~ 청산 + 25 (깔때기 모드는 주문일까지)
PRIMARY = "WD"
P_WD = dict(Wd.WD_BUYSTOP)
ARMS: dict[str, dict] = {
    "WD": {"lo": EVENT_LO, "params": P_WD, "desc": "쐐기 D — D 봉 고가 매수정지, 2R 전 손절만 (주·탐색)"},
    "WD-be1": {"lo": EVENT_LO, "params": {**P_WD, "be_r": 1.0}, "desc": "WD, +1R 다음날부터 손절 = 진입가 × 1.005 (기술)"},
    "WD-up": {"lo": EVENT_LO, "params": P_WD,
              "desc": "같은 모양·같은 청산, 진입만 상단선 A–C 첫 돌파 매수정지 (기술 — 바닥 vs 돌파, 정정 1 M7)"},
    "WD-nc": {"lo": EVENT_LO, "params": P_WD, "desc": "WD, 수렴 조건(c_2 < c_1, 상단 < 하단) 끔 (기술)"},
    "WD-nocap": {"lo": EVENT_LO, "params": {**P_WD, "max_hold": 1_000_000}, "desc": "WD, 120 거래일 상한 없음(조건 청산만, 기술)"},
}
SENS_CELLS = ("WD 보수(p=1)", "WD 낙관(p=0)", "WD 완결 거래만", f"WD 사건 ≤ {CAP_SENS}", "WD p 재가중", "WD 팩터 극단일 제외")
SHAPE_PASSED = (Wd.D_OK, Wd.D_FAR, Wd.D_CLOSE)    # 거래량 마름까지 통과한 모양(WD-up 셋업 바탕)


def _arm_config(name: str, lo: str, hi: str, floor: float, fac_sha: str) -> dict:
    a = ARMS[name]
    return {"prereg": PREREG_DOC, "arm": name, "desc": a["desc"], "event_lo": lo, "event_hi": hi, "adv_floor": floor,
            "universe": R.UNIVERSE_DESC.format(floor=floor), "prices": R.PRICE_DESC,
            "wd": {**Wd.WD_PARAMS, "conv": name != "WD-nc"}, "buystop": {**V.BUYSTOP_PARAMS, **a["params"]},
            "context": "H0 일 trend_template ∧ RS≥70; d−1 MA50>MA150>MA200 ∧ MA200 > 20일 전 ∧ breadth>0.5",
            "same_day": "진입일 종가 ≤ 손절가 = 확정; 모호 = 분봉 p 혼합(바닥 반전 합성 사건) · 보수 · 낙관",
            "null": (f"되돌림 새 저점 널 {N_NULL}벌(맥락 통과·WD 아님, k~U{NULL_K} 최고 이후 최저 저가, 눌림 "
                     f"[{NULL_PULL:.0%},{NULL_PULL_HI:.0%}], 그 봉 고가 매수정지, d−행 ≤ 5, 두 번째 관통 차단, 어제 최저 우선, 층화 월 × d 3분위 × "
                     f"주문가/MA50 3분위, seed {NULL_SEED})") if name == PRIMARY else None,
            "r_def": "총 R = 총수익/d(비용 전), 순 R = 순/d; 바 5 R 바 = 총 R; 손절 체결 = min(시가, 손절가) − 1호가",
            "viewed_exclusion": f"{VIEWED_CSV} ± {VIEWED_PAD} 거래일 겹침 제외 표본 병기",
            "cost": "korea.cost_rt 1배 · 2배", "factors": f"{R.FACTOR_COLS} NW 5 sha256 {fac_sha}", "exploratory": True}


# ---------------------------------------------------------------------------
# 셋업 · 시뮬
# ---------------------------------------------------------------------------

def build_wd(ctx: "WR.WCtx", floor: float, lo_all: str, hi: str, lo_of: dict) -> dict:
    """하한 ``floor`` 의 쐐기 D 셋업 표·팔 행·깔때기(성과 계산 없음)."""
    A, days, codes = ctx.A, ctx.days, ctx.codes
    nd = len(days)
    h5 = ctx.universe(floor)
    d_ok = np.asarray((days >= pd.Timestamp(lo_all)) & (days <= pd.Timestamp(hi)))
    s_ok = np.zeros(nd, bool)
    s_ok[:-1] = d_ok[1:]
    base_nosplit = h5.to_numpy(bool) & ctx.struct & (ctx.br > BREADTH_MIN) & s_ok[:, None]
    base = base_nosplit & ~ctx.split["recent"]
    si, ci = np.nonzero(base)
    sc = Wd.wd_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci, si)
    nc = Wd.wd_scan_arrays(A.H, A.L, A.C, ctx.VOL, ci, si, {"conv": False})
    S = pd.DataFrame({"code": codes[ci], "sig_idx": si, "day_idx": si + 1, "col_idx": ci, **sc})
    S["event_day"] = days[si]
    S["order_day"] = days[si + 1]
    i0 = S["i0"].to_numpy()
    i0s = np.maximum(i0, 0)
    S["base_len"] = np.where(i0 >= 0, si - i0, -1)
    in_base = (S["base_len"] >= Wd.WD_PARAMS["base_min"]) & (S["base_len"] <= Wd.WD_PARAMS["base_max"])
    S["context"] = (in_base & (i0 >= 0) & ctx.tt[i0s, ci] & (ctx.rs[i0s, ci] >= RS_MIN)).to_numpy(bool)
    S["valid"] = (S["status"] == Wd.D_OK) & S["context"]
    S["valid_up"] = S["status"].isin(SHAPE_PASSED) & (S["up_ok"] == 1) & S["context"]
    S["valid_nc"] = (nc["status"] == Wd.D_OK) & S["context"].to_numpy(bool)
    for kk in ("d_high", "u_d", "dlow", "i_b", "i_c", "i_d"):
        S[f"nc_{kk}"] = nc[kk]
    S["market"] = ctx.p.market.reindex(S["code"]).to_numpy()
    S["adv20"] = ctx.adv[si, ci]
    S["ma50_s"] = ctx.ma50[si, ci]
    S["close_s"] = A.C[ci, si]
    S["rs_s"] = ctx.rs[si, ci]
    dts = S["order_day"]
    rng_ = {n: (dts >= pd.Timestamp(lo_of[n])) & (dts <= pd.Timestamp(hi)) for n in ARMS}
    arm_rows = {"WD": S["valid"] & rng_["WD"], "WD-be1": S["valid"] & rng_["WD-be1"], "WD-up": S["valid_up"] & rng_["WD-up"],
                "WD-nc": S["valid_nc"] & rng_["WD-nc"], "WD-nocap": S["valid"] & rng_["WD-nocap"]}
    status = {Wd.WD_STATUS[k]: int(n_) for k, n_ in S["status"].value_counts().items()}
    funnel = {"floor": floor, "base_nosplit": int(base_nosplit.sum()),
              "split_excl": int((base_nosplit & ctx.split["recent"]).sum()), "base": int(len(S)), "status": status,
              "wd_ok": int((S["status"] == Wd.D_OK).sum()), "context_fail": int(((S["status"] == Wd.D_OK) & ~S["context"]).sum()),
              "valid": int(S["valid"].sum()), "valid_up": int(S["valid_up"].sum()), "valid_nc": int(S["valid_nc"].sum()),
              "null_pool": int((S["context"] & ~S["valid"]).sum()), "arms": {n: int(m.sum()) for n, m in arm_rows.items()}}
    return {"S": S, "h5": h5, "arm_rows": arm_rows, "rng": rng_, "funnel": funnel}


def simulate_all(ctx: "WR.WCtx", B: dict) -> "tuple[dict, dict]":
    """팔별 시뮬(진입 + 보유 거르기에 쓰는 청산). 비용·순수익은 붙이지 않는다."""
    trades, skips = {}, {}
    for n in ARMS:
        ev = B["S"][B["arm_rows"][n]].reset_index(drop=True)
        if n == "WD-nc":
            ev = ev.assign(**{kk: ev[f"nc_{kk}"] for kk in ("d_high", "u_d", "dlow", "i_b", "i_c", "i_d")})
        t, sk = V.simulate_buystop(ctx.A, Wd.wd_events(ev, up=(n == "WD-up")), params=ARMS[n]["params"])
        trades[n], skips[n] = t, (sk["skip_reason"].value_counts().to_dict() if len(sk) else {})
    return trades, skips


def load_viewed(days: pd.DatetimeIndex, path: str = VIEWED_CSV, pad: int = VIEWED_PAD) -> dict:
    """열람 창 → {종목: [(행 하한, 행 상한), …]} (첫 봉 − pad ~ 마지막 봉 + pad 거래일)."""
    if not os.path.exists(path):
        return {}
    v = pd.read_csv(path, dtype={"code": str})
    out: dict = {}
    for r in v.itertuples(index=False):
        a = int(days.searchsorted(pd.Timestamp(r.first_bar))) - pad
        b = int(days.searchsorted(pd.Timestamp(r.last_bar), side="right")) - 1 + pad
        out.setdefault(str(r.code).zfill(6), []).append((a, b))
    return out


def viewed_flag(t: pd.DataFrame, win: dict, *, with_exit: bool) -> np.ndarray:
    """체결의 보유 구간 [진입 행, 청산 행](깔때기 모드는 진입 행만)이 그 종목 열람 창 ± pad 와 겹치면 True."""
    if t is None or t.empty or not win:
        return np.zeros(0 if t is None else len(t), bool)
    lo = t["entry_row"].to_numpy(int)
    hi = np.maximum(t["exit_row"].to_numpy(int), lo) if with_exit and "exit_row" in t else lo
    out = np.zeros(len(t), bool)
    for i, c in enumerate(t["code"].astype(str).to_numpy()):
        for a, b in win.get(c.zfill(6), ()):
            if lo[i] <= b and hi[i] >= a:
                out[i] = True
    return out


def gross_r(x: pd.DataFrame) -> dict:
    """정정 1 M5 — 가중 행의 총 R(= 총수익/d, 비용 전): 손실 거래(순 < 0) 평균 총 R, 전 거래 총 R 5% 분위·평균·중앙."""
    if x is None or x.empty:
        return {}
    w = x["w"].to_numpy(float)
    rg = (x["ret"] / x["d"]).to_numpy(float)
    loss = x["net"].to_numpy(float) < 0
    return {"rg_mean": float(np.average(rg, weights=w)), "rg_median": R.wquantile(rg, w, 0.5),
            "rg_p5": R.wquantile(rg, w, 0.05),
            "rg_loss_mean": float(np.average(rg[loss], weights=w[loss])) if loss.any() else np.nan}


def arm_numbers(t: pd.DataFrame, pp: float, n_dsr: int) -> dict:
    """가중 통계 + 총 R(바 5 의 R 바는 총 R — ``avg_loss_R``·``r_p5`` 를 총 R 로 바꾼 ``*_bar`` 칸)."""
    x = R.expand(t, pp)
    s = R.wstats(x, n_dsr)
    s.update(gross_r(x))
    return s


def bar_input(s: dict) -> dict:
    """바 5 R 바를 총 R 로: wedge_bars 에 넘길 dict(순 R 은 그대로 표에)."""
    return {**s, "avg_loss_R": s.get("rg_loss_mean", np.nan), "r_p5": s.get("rg_p5", np.nan)}


def wd_desc_cols(ctx: "WR.WCtx", t: pd.DataFrame) -> pd.DataFrame:
    """진입 기술 열(성과 아님): 체결가의 D 대비 위치, D → 체결 일수, H0 거리, 돌파일 거래량."""
    if t is None or t.empty:
        return t
    t = WR.entry_desc(ctx, t)
    t["fill_vs_d"] = t["fill_px"] / t["dlow"] - 1.0
    t["d_to_fill"] = t["day_idx"] - t["i_d"]
    return t


def _bars(ctx: "WR.WCtx", ci: int, lo: int, hi: int) -> list:
    days = ctx.days
    lo, hi = max(lo, 0), min(hi, len(days) - 1)
    A = ctx.A
    out = []
    for j in range(lo, hi + 1):
        o, h, low, c = A.O[ci, j], A.H[ci, j], A.L[ci, j], A.C[ci, j]
        if not np.isfinite(c):
            continue
        out.append({"date": str(days[j].date()), "o": float(o), "h": float(h), "l": float(low), "c": float(c),
                    "v": float(ctx.VOL[ci, j]), "ma50": (float(A.MA[ci, j]) if np.isfinite(A.MA[ci, j]) else None)})
    return out


def chart_points(ctx: "WR.WCtx", t: pd.DataFrame, with_exit: bool = False) -> list:
    """차트용 — A·B·C·D·상단선 A–C·하단선 B–D·주문가·체결가·손절가 + 봉(H0 − 60 ~ 청산 + 25, MA50). ``with_exit`` 아니면 봉은 주문일까지,
    청산은 넣지 않는다(깔때기 모드)."""
    days = ctx.days
    out = []

    def dd(row):
        return str(days[min(int(row), len(days) - 1)].date())

    for r in t.itertuples(index=False):
        pt = Wd.wd_points(ctx.A.H, ctx.A.L, int(r.col_idx), int(r.sig_idx), int(r.i_b), int(r.i_c), int(r.i_d))
        item = {"code": str(r.code), "signal_day": str(pd.Timestamp(r.event_day).date()),
                "order_day": str(pd.Timestamp(r.order_day).date()),
                "points": [{"date": dd(p["row"]), "price": p["price"], "kind": p["kind"]} for p in pt["points"]],
                "upper": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt["upper"].items()},
                "lower": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt["lower"].items()},
                "order_px": float(r.order_px), "fill_px": float(r.fill_px), "stop_px": float(r.stop_px),
                "gap_fill": bool(r.gap_fill)}
        end = int(r.day_idx)
        if with_exit:
            item["exit"] = {"date": str(pd.Timestamp(r.exit_day).date()), "price": float(r.exit_px), "reason": str(r.exit_reason),
                            "ambiguous_entry_day": bool(r.amb),
                            "partial": ({"date": str(pd.Timestamp(r.partial_day).date()), "price": float(r.partial_px)}
                                        if bool(r.partial_hit) else None)}
            end = int(r.exit_row) + CHART_POST
        item["bars"] = _bars(ctx, int(r.col_idx), int(r.i0) - CHART_PRE, end)
        out.append(item)
    return out


def funnel_counts(ctx: "WR.WCtx", floor: float, lo_all: str, hi: str, lo_of: dict) -> dict:
    """``--funnel-only`` 한 하한 — 건수만(청산 결과·수익은 어디에도 쓰지 않는다)."""
    B = build_wd(ctx, floor, lo_all, hi, lo_of)
    trades, skips = simulate_all(ctx, B)
    t = wd_desc_cols(ctx, trades["WD"])
    sub = (pd.to_datetime(t["order_day"]) <= pd.Timestamp(SUB_HI)) if len(t) else pd.Series(dtype=bool)

    def q3(col, fmt=float):
        return {q: fmt(t[col].quantile(q)) for q in (0.1, 0.5, 0.9)} if len(t) else {}

    vwin = load_viewed(ctx.days)
    viewed = {n: int(viewed_flag(trades[n], vwin, with_exit=False).sum()) if len(trades[n]) else 0 for n in ARMS}
    return {"funnel": B["funnel"], "skips": skips, **{n: int(len(trades[n])) for n in ARMS}, "viewed": viewed,
            "WD_sub": int(sub.sum()) if len(t) else 0,
            "WD_years": ({int(k): int(v) for k, v in pd.to_datetime(t["order_day"]).dt.year.value_counts().sort_index().items()}
                         if len(t) else {}),
            "WD_base_len": q3("base_len"), "WD_fill_vs_d": q3("fill_vs_d"), "WD_d_to_fill": q3("d_to_fill"),
            "WD_h0_dist": q3("h0_dist"), "WD_rise": {w: q3(f"rise{w}") for w in (20, 40, 60)},
            "WD_d_conf": int((t["d_confirmed"] == 1).sum()) if len(t) else 0,
            "WD_gap_fill": int(t["gap_fill"].sum()) if len(t) else 0,
            "WD_vol_buckets": ({int(k): int(v) for k, v in pd.Series(t["bo_vol_bucket"]).value_counts().sort_index().items()}
                               if len(t) else {}),
            "WD_adv_median": float(t["adv20"].median()) if len(t) else np.nan,
            "chart": chart_points(ctx, t) if len(t) else []}


def write_funnel(path: str, res: dict, lo_all: str, hi: str, rule: tuple, elapsed: float, ctx: "WR.WCtx") -> None:
    floor, void = rule
    fl = sorted(res, reverse=True)

    def qs(d, fmt):
        return "/".join(fmt.format(d[q]) for q in (0.1, 0.5, 0.9)) if d else "—"

    L = [f"# 쐐기 D(바닥 진입) — 깔때기(건수만, 성과 미계산) {date.today().isoformat()}", "",
         f"`wedge_d_run.py --funnel-only` ({PREREG_DOC}). 사건(주문일) {lo_all}~{hi}. 시뮬은 진입 체결과 보유 중 거르기(그 거르기에 필요한 "
         f"청산일)까지만 쓰고, 청산 사유·수익·손절 여부는 어디에도 적지 않는다. 실행 {elapsed / 60:.1f}분. 사슬: {CHAIN}.", "",
         f"**사전 규칙:** WD 체결 ≥ {FLOOR_RULE_N} ∧ 2017-09~2024-12 체결 ≥ {FLOOR_RULE_SUB} 인 가장 높은 하한 → "
         + (f"**ADV20 ≥ {floor:,.0f} 백만원({floor / 100:,.0f}억)**." if floor is not None else "**넷 다 미달 → WD VOID**(기술만)."),
         f"<!-- adv_floor_rule: {'none' if floor is None else f'{floor:.0f}'} · v0_void: {'true' if void else 'false'} -->", "",
         "| ADV20 하한(백만원) | WD 체결 | 하위(≤2024-12) | 규칙 충족 | WD 열람 겹침(진입일) | WD-be1 | WD-up | WD-nc | WD-nocap | "
         "WD 셋업 행(맥락 통과) | 셋업 후보 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in fl:
        r = res[f]
        ok = r["WD"] >= FLOOR_RULE_N and r["WD_sub"] >= FLOOR_RULE_SUB
        L.append(f"| {f:,.0f} | {r['WD']} | {r['WD_sub']} | {'✓' if ok else '✗'} | {r['viewed']['WD']} | {r['WD-be1']} | {r['WD-up']} | "
                 f"{r['WD-nc']} | {r['WD-nocap']} | {r['funnel']['valid']:,} | {r['funnel']['base']:,} |")
    L += ["", "## 하한별 깔때기", ""]
    for f in fl:
        r, fu = res[f], res[f]["funnel"]
        st = fu["status"]
        L += [f"### ADV20 ≥ {f:,.0f} 백만원", "",
              f"- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) {fu['base_nosplit']:,} − 분할 직후 60일 "
              f"{fu['split_excl']:,} = {fu['base']:,}",
              "- 첫 탈락: " + " · ".join(f"{k} {v:,}" for k, v in sorted(st.items(), key=lambda kv: Wd.WD_STATUS.index(kv[0]))),
              f"- 모양·진입 통과 {fu['wd_ok']:,} − H0 맥락 탈락 {fu['context_fail']:,} = **{fu['valid']:,}** · WD-up 셋업 행 {fu['valid_up']:,} · "
              f"WD-nc {fu['valid_nc']:,} · 널 풀(맥락 통과·WD 아님) {fu['null_pool']:,} · 팔 셋업 행 {fu['arms']}",
              f"- **WD 체결 {r['WD']}** (하위 {r['WD_sub']}) · 연도별 {r['WD_years']}",
              f"- 열람 창(`{VIEWED_CSV}` ± {VIEWED_PAD} 거래일)과 진입일이 겹치는 체결(깔때기 모드 — 보유 끝은 안 봄): "
              + " · ".join(f"{n} {v}" for n, v in r["viewed"].items()),
              f"- 기반 길이(H0 → s) 10/50/90: {qs(r['WD_base_len'], '{:.0f}')} · 체결가의 D 대비 10/50/90: {qs(r['WD_fill_vs_d'], '{:+.1%}')} · "
              f"D → 체결 일수 10/50/90: {qs(r['WD_d_to_fill'], '{:.0f}')} · 체결가 → H0 거리 10/50/90: {qs(r['WD_h0_dist'], '{:+.1%}')}",
              "- H0 전 상승률(저가 기준) 10/50/90 — " + " · ".join(f"{w}일 {qs(r['WD_rise'][w], '{:+.0%}')}" for w in (20, 40, 60)),
              f"- D 확정(5% 반등) 체결 {r['WD_d_conf']} · 갭 체결 {r['WD_gap_fill']} · 체결일 거래량/50일 구간 {r['WD_vol_buckets']} · "
              f"ADV20 중앙 {r['WD_adv_median']:,.0f} 백만원",
              "- 주문·체결 결과(버림 사유): " + " · ".join(f"{n}: " + ", ".join(f"{k} {v}" for k, v in sorted(sk.items()))
                                                      for n, sk in r["skips"].items() if sk), ""]
    L += ["## 공통", "",
          f"- NXT 미보정 행이 있는 종목 {ctx.n_cut_stocks:,}개 — 보유 경로는 {R.NXT_LAST_DAY} 에서 `data_end` 로 끊는다.",
          "- 차트 JSON: `wedge_d/chart_points.json` (하한별 WD 체결, 봉은 H0 − 60 ~ 주문일 — 깔때기 모드는 청산·이후 봉 없음).",
          "- 정정 1(b008fce) 반영: D ≤ 0.99·B, 같은 (A, C) 두 번째 관통 차단, 손절 체결 = 손절가 − 1호가, WD-up 같은 청산, WD-nocap 팔, "
          "열람 제외 표시.",
          "- 탐색 표시: 판정은 사람이, 이 문서는 사전 규칙의 산술만.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# 널 작업자(fork, 직렬 커널만)
# ---------------------------------------------------------------------------

_GD: dict = {}


def _null_task(task: tuple) -> dict:
    arm, b = task
    g = _GD
    A, P = g["A"], g["plan"][arm]
    rng = np.random.default_rng([NULL_SEED, b])
    ci, s = P["ci"], P["s"]
    m = len(ci)
    k = rng.integers(NULL_K[0], NULL_K[1] + 1, size=m)
    nb = Wd.null_bottoms(A, ci, s, k, pull=NULL_PULL, pull_hi=NULL_PULL_HI, parallel=False)
    chk = V.buystop_entry_check(A, ci, s + 1, nb["pivot"], nb["l_last"], ARMS[arm]["params"])
    ok = nb["ok"] & (chk["status"] == V.S_OK)
    terc = np.searchsorted(P["edges"], chk["dd"], side="right")
    order = nb["pivot"] + V.tick_size_vec(nb["pivot"])
    with np.errstate(invalid="ignore", divide="ignore"):
        mrat = order / g["ma50"][s, ci]
    mterc = np.searchsorted(P["edges_ma"], mrat, side="right")
    yday = nb["yday"]
    rows = np.flatnonzero(ok)
    sel, pool_ratio = [], []
    short = n_yday = 0
    for (mo, tq, mq), need in P["need"].items():
        lo_r, hi_r = P["month_rows"][mo]
        d_r = s[rows] + 1
        cell = (terc[rows] == tq) & (mterc[rows] == mq)
        cand = rows[cell & (d_r >= lo_r) & (d_r <= hi_r)]
        if len(cand) < NULL_POOL_MULT * need:
            cand = rows[cell & (d_r >= lo_r - NULL_EXPAND) & (d_r <= hi_r + NULL_EXPAND)]
        pool_ratio.append(len(cand) / need)
        take = min(need, len(cand))
        short += need - take
        if take:
            # 칸 안에서 '어제가 최저가' 후보 우선(정정 1 M6), 모자라면 나머지에서
            cy, cn = cand[yday[cand]], cand[~yday[cand]]
            ty = min(take, len(cy))
            pick = [rng.choice(cy, size=ty, replace=False)] if ty else []
            if take > ty:
                pick.append(rng.choice(cn, size=take - ty, replace=False))
            n_yday += ty
            sel.append(np.concatenate(pick))
    base = {"arm": arm, "b": b, "short": short, "need": P["need_total"], "setups_ok": int(nb["ok"].sum()),
            "fillable": int(ok.sum()), "yday_ok": int((ok & yday).sum()), "picked_yday": int(n_yday)}
    if not sel:
        return {**base, "n": 0, "n_cut": 0}
    sel = np.unique(np.concatenate(sel))
    sel = sel[np.lexsort((ci[sel], s[sel]))]
    r = V.run_buystop(A, ci[sel], s[sel] + 1, nb["pivot"][sel], nb["l_last"][sel], nb["pid"][sel], no_overlap=True,
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


def _bottom_levels(A, cs, ss, ks, js=None):
    """분봉 p 합성 사건(바닥 반전 = 널과 같은 '되돌림 새 저점' 정의, WD 와 같은 진입일 모호 정의) — (기준가 = 저점 봉 고가,
    손절 기준 = 그 저가, 통과). ``js`` 는 격자 호환용(쓰지 않음)."""
    o = Wd.null_bottoms(A, cs, ss, ks, pull=NULL_PULL, pull_hi=NULL_PULL_HI)
    return o["pivot"], o["l_last"], o["ok"]


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def _record_chain(out_dir: str) -> None:
    record_trial(LABEL, {"prereg": PREREG_DOC, "chain": CHAIN, "evaluated": "사슬 기록(사전등록 요구)"}, logs_dir=out_dir,
                 note=f"사슬: {CHAIN}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None)
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--factors", default=R.FACTORS_PKL)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--p-hi", default=None)
    ap.add_argument("--counts-only", action="store_true")
    ap.add_argument("--adv-floor", default=None, help="전체: 생략하면 wedge_d/FUNNEL.md 사전 규칙 값(VOID 면 5000 기술). --funnel-only: 쉼표 목록")
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
    _record_chain(out_dir)

    if args.funnel_only:
        floors = [float(x) for x in (args.adv_floor or ",".join(str(int(f)) for f in ADV_FLOORS_FUNNEL)).split(",")]
        ctx = WR.WCtx(t0)
        res = {}
        for f in floors:
            res[f] = funnel_counts(ctx, f, lo_all, hi, lo_of)
            _log(f"하한 {f:,.0f}: WD {res[f]['WD']} (하위 {res[f]['WD_sub']}) · " + " · ".join(
                f"{n} {res[f][n]}" for n in ARMS if n != "WD"), t0)
        ok = [f for f, r in res.items() if r["WD"] >= FLOOR_RULE_N and r["WD_sub"] >= FLOOR_RULE_SUB]
        rule = (max(ok), False) if ok else (None, True)
        path = os.path.join(sub_dir, "FUNNEL.md")
        write_funnel(path, res, lo_all, hi, rule, time.time() - t0, ctx)
        with open(os.path.join(sub_dir, "chart_points.json"), "w", encoding="utf-8") as fh:
            json.dump({f"{int(f)}": r["chart"] for f, r in res.items()}, fh, ensure_ascii=False)
        print(f"사전 규칙 → 하한 {rule[0]} · WD VOID {rule[1]} → {path}")
        return 0

    if args.adv_floor is not None:
        floor, void = float(args.adv_floor), None
    else:
        floor, void = R.read_floor_rule(FUNNEL_MD)
        if floor is None and void is None:
            ap.error(f"--adv-floor 를 주거나 먼저 --funnel-only 로 {FUNNEL_MD} 를 만든다")
        if floor is None:
            floor = R.ADV_FLOOR
    timings: dict[str, float] = {}
    fac_sha = R._sha(args.factors)
    cfgs = {n: _arm_config(n, lo_of[n], hi, floor, fac_sha) for n in ARMS}
    for n, c in cfgs.items():
        record_trial(LABEL, c, logs_dir=out_dir, note=f"쐐기 D 팔 {n}: {ARMS[n]['desc']}")
    for cell in SENS_CELLS:
        record_trial(LABEL, {**cfgs[PRIMARY], "sensitivity": cell}, logs_dir=out_dir, note=f"쐐기 D 민감도: {cell}")
    for dv in DESIGN_VARIANTS:
        record_trial(LABEL, {"prereg": PREREG_DOC, "design_variant": dv, "evaluated": "건수만, 성과 미계산"}, logs_dir=out_dir,
                     note=f"쐐기 D 설계 변형(리뷰 1단계, 건수만): {dv}")
    n_led = count_trials(LABEL, logs_dir=out_dir)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
    n_dsr = n_led + sum(n_prior.values())

    ctx = WR.WCtx(t0)
    A, days, codes, mkt = ctx.A, ctx.days, ctx.codes, ctx.mkt
    cost_tab = R.cost_table(days)
    timings["load"] = time.time() - t0
    B = build_wd(ctx, floor, lo_all, hi, lo_of)
    S, h5, funnel = B["S"], B["h5"], B["funnel"]
    funnel["floor_rule_void"] = void
    _log(f"하한 {floor:,.0f} · 셋업 후보 {len(S):,} · WD {funnel['valid']:,} · 팔 {funnel['arms']}", t0)
    mom = momentum_12_1(ctx.p.close)
    q_mom = day_market_quantiles(mom, h5, ctx.p.market, n_q=MOM_N_Q).to_numpy(float)
    adv2 = day_market_median_half(ctx.D.adv20, h5, ctx.p.market).to_numpy(float)
    del mom

    t1 = time.time()
    pinfo = R._estimate_p(ctx.con, A, h5, days, args.p_hi, params=P_WD, levels_fn=_bottom_levels)
    pval = pinfo["p"]
    timings["p_minute"] = time.time() - t1

    t1 = time.time()
    raw, skips = simulate_all(ctx, B)
    trades = {n: R._finish(wd_desc_cols(ctx, raw[n]), cost_tab, mkt, A.CUT) for n in ARMS}
    pinfo["p_reweighted"] = R._p_reweight(pinfo, trades["WD"])
    timings["sim"] = time.time() - t1

    t1 = time.time()
    v = S["valid"]
    valid_keys = set(zip(S.loc[v, "col_idx"].to_numpy(), S.loc[v, "sig_idx"].to_numpy(), strict=True))
    plan = {}
    t = trades["WD"]
    if not t.empty:
        pm = (S["context"] & ~v & B["rng"]["WD"]).to_numpy(bool)
        pci, pss = S["col_idx"].to_numpy()[pm], S["sig_idx"].to_numpy()[pm]
        assert not any((c_, s_) in valid_keys for c_, s_ in zip(pci, pss, strict=True)), "널 풀에 WD 셋업"
        dd = t["d"].to_numpy(float)
        edges = np.quantile(dd, [1 / 3, 2 / 3]) if len(t) >= 3 else np.array([np.inf, np.inf])
        terc = np.searchsorted(edges, dd, side="right")
        mrat = t["order_px"].to_numpy(float) / ctx.ma50[t["sig_idx"].to_numpy(int), t["col_idx"].to_numpy(int)]
        edges_ma = np.nanquantile(mrat, [1 / 3, 2 / 3]) if len(t) >= 3 else np.array([np.inf, np.inf])
        mterc = np.searchsorted(edges_ma, mrat, side="right")
        mo = t["order_day"].dt.to_period("M").astype(str).to_numpy()
        need = pd.Series(1, index=pd.MultiIndex.from_arrays([mo, terc, mterc])).groupby(level=[0, 1, 2]).sum().to_dict()
        plan["WD"] = {"ci": pci.astype(np.int64), "s": pss.astype(np.int64), "edges": edges, "edges_ma": edges_ma, "need": need,
                      "need_total": int(len(t)), "month_rows": R._month_rows(days), "pool_rows": int(pm.sum())}
    _GD.update({"A": A, "plan": plan, "cost_tab": cost_tab, "mkt": mkt, "p": pval, "ma50": ctx.ma50})
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
    vwin = load_viewed(days)
    vx_trades = {}
    for n, tn in trades.items():
        if len(tn):
            tn["viewed"] = viewed_flag(tn, vwin, with_exit=True)
            vx_trades[n] = tn[~tn["viewed"].to_numpy(bool)].reset_index(drop=True)
        else:
            vx_trades[n] = tn
    moms = {n: R._momentum_control(trades[n], A, h5, q_mom, adv2, mkt, days, codes, pval, ARMS[n]["params"])
            for n in ("WD", "WD-up")}
    moms_vx = {n: R._momentum_control(vx_trades[n], A, h5, q_mom, adv2, mkt, days, codes, pval, ARMS[n]["params"])
               for n in ("WD", "WD-up")}
    timings["matched"] = time.time() - t1
    F = pd.read_pickle(args.factors) if os.path.exists(args.factors) else None
    kospi = R._naver_returns("KOSPI")
    ports = {n: R._portfolio(trades[n], A, days, lo_of[n], hi, F, kospi, pval) for n in ARMS}
    ports_vx = {n: R._portfolio(vx_trades[n], A, days, lo_of[n], hi, F, kospi, pval) for n in ("WD", "WD-up")}

    rows = {}
    for n in ARMS:
        tn, tv = trades[n], vx_trades[n]
        s = {"arm": n, "n_setups": funnel["arms"].get(n, 0), "skips": skips[n], "n_fills": len(tn),
             "n_viewed": int(tn["viewed"].sum()) if len(tn) else 0, "vx_n_fills": len(tv)}
        for tag, pp in (("", pval), ("cons_", 1.0), ("opt_", 0.0)):
            s.update({f"{tag}{k}": v_ for k, v_ in arm_numbers(tn, pp, n_dsr).items()})
            s.update({f"vx_{tag}{k}": v_ for k, v_ in arm_numbers(tv, pp, n_dsr).items()})
        s.update(WR._desc(ctx, tn))
        s.update(_hold_desc(ctx, tn))
        s.update({f"vx_{k}": v_ for k, v_ in WR._desc(ctx, tv).items()})
        s.update({f"vx_{k}": v_ for k, v_ in _hold_desc(ctx, tv).items()})
        if moms.get(n):
            s.update(moms[n])
        if moms_vx.get(n):
            s.update({f"vx_{k}": v_ for k, v_ in moms_vx[n].items()})
        s.update(ports[n]["stats"])
        if n in ports_vx:
            s.update({f"vx_{k}": v_ for k, v_ in ports_vx[n]["stats"].items()})
        rows[n] = s
    w = rows["WD"]
    if not nulls.empty:
        z = nulls[nulls["n"] > 0]
        for tag in ("", "_cons", "_opt"):
            w[f"null_p95{tag}"] = float(np.nanpercentile(z[f"net2_mean{tag}"], 95)) if len(z) else np.nan
        w.update({"null_k": int(len(z)), "null_p50": float(np.nanmedian(z["net2_mean"])) if len(z) else np.nan,
                  "null_rank": float((z["net2_mean"] < w.get("net2_mean", np.nan)).mean()) if len(z) else np.nan,
                  "null_fill_ratio_med": float((nulls["n"] / nulls["need"]).median()),
                  "null_yday_ratio_med": float((nulls["picked_yday"] / nulls["need"]).median()),
                  "null_pool_rows": plan["WD"]["pool_rows"]})
    # 바 산술 — 바 5 R 바는 총 R(bar_input). 열람 제외 표본은 같은 널 p95(전체 WD 에 맞춘 널)·그 표본의 정합 α·팩터 t.
    for pre in ("", "vx_"):
        for tag, nt in (("", ""), ("cons_", "_cons"), ("opt_", "_opt")):
            key = f"{pre}{tag}"
            sub = {k[len(key):]: v_ for k, v_ in w.items() if k.startswith(key)} if key else w
            bb = WR.wedge_bars(bar_input(sub), w.get(f"null_p95{nt}", np.nan), w.get(f"{pre}mom_lo", np.nan),
                               w.get(f"{pre}fac_alpha_t", np.nan))
            w.update({f"{key}bar_{k}": v_ for k, v_ in bb.items()})
    sens = _sensitivities(trades["WD"], vx_trades["WD"], pval, n_dsr, w, pinfo.get("p_reweighted", np.nan))
    be1 = WR._be1_pairs(ctx, trades["WD"], trades["WD-be1"], pval)

    elapsed = time.time() - t0
    L = render(rows, funnel, sens, be1, pinfo, ports, lo_of, hi, n_led, n_prior, n_dsr, out_dir, debug, elapsed, timings,
               args, fac_sha, floor)
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    for n, tn in trades.items():
        tn.to_csv(os.path.join(sub_dir, f"trades_{n}.csv"), index=False)
    if not nulls.empty:
        nulls.to_csv(os.path.join(sub_dir, "null_draws.csv"), index=False)
    pd.DataFrame([{k: v_ for k, v_ in r.items() if not isinstance(v_, dict)} for r in rows.values()]).to_csv(
        os.path.join(sub_dir, "arms.csv"), index=False)
    with open(os.path.join(sub_dir, "chart_points.json"), "w", encoding="utf-8") as fh:
        json.dump({f"{int(floor)}": chart_points(ctx, trades["WD"], with_exit=True) if len(trades["WD"]) else []}, fh,
                  ensure_ascii=False)
    S.loc[S["valid"]].to_csv(os.path.join(sub_dir, "setups_valid.csv"), index=False)

    print(f"\n=== 쐐기 D — {elapsed / 60:.1f}분 · 하한 {floor:,.0f} · 원장 {n_led}줄 · DSR N {n_dsr} ===")
    print("시간(s): " + " · ".join(f"{k} {v_:.1f}" for k, v_ in timings.items()))
    print(f"셋업 후보 {funnel['base']:,} · WD 모양 {funnel['wd_ok']:,} · 맥락 통과 {funnel['valid']:,} · WD-up {funnel['valid_up']:,} · "
          f"WD-nc {funnel['valid_nc']:,} · 분봉 p 판별 {pinfo['n_classified']:,}")
    for n, s in rows.items():
        line = f"팔 {n}: 셋업 행 {s['n_setups']:,} · 체결 {s['n_fills']:,} (모호 {s.get('amb_n', 0)}, NXT 절단 {s.get('nxt_cut', 0)})"
        if not args.counts_only:
            line += f" · 순 2배 {_pct(s.get('net2_mean', np.nan))}"
        print(line)
    if not nulls.empty:
        print(f"널 벌 {len(nulls)} · 체결 중앙 {nulls['n'].median():.0f}/{nulls['need'].median():.0f}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def _sensitivities(t: pd.DataFrame, tv: pd.DataFrame, pval: float, n_dsr: int, s0: dict, p_rw: float) -> dict:
    """민감도(완결 거래만 · 사건 ≤ CAP_SENS · p 재가중) — 전체와 열람 제외, 바 5 는 총 R."""
    out = {}
    for lab, tt in (("", t), ("열람 제외 · ", tv)):
        if tt is None or tt.empty:
            continue
        cells = (("완결 거래만", None, True), (f"사건 ≤ {WR.CAP_SENS}", tt["order_day"] <= pd.Timestamp(WR.CAP_SENS), False))
        for name, mk, comp in cells:
            sub = tt if mk is None else tt[mk.to_numpy()]
            res = {}
            for tag, pp in (("", pval), ("cons_", 1.0), ("opt_", 0.0)):
                x = R.expand(sub, pp)
                if comp and len(x):
                    x = x[x["complete"].to_numpy(bool)]
                st = R.wstats(x, n_dsr)
                st.update(gross_r(x))
                b = WR.wedge_bars(bar_input(st), s0.get("null_p95", np.nan), s0.get("mom_lo", np.nan), np.nan)
                res.update({f"{tag}{k}": v for k, v in st.items()})
                res.update({f"{tag}bar_{k}": v for k, v in b.items()})
            out[f"{lab}{name}"] = res
        if np.isfinite(p_rw):
            st = arm_numbers(tt, p_rw, n_dsr)
            b = WR.wedge_bars(bar_input(st), s0.get("null_p95", np.nan), s0.get("mom_lo", np.nan), np.nan)
            out[f"{lab}p 재가중 {p_rw:.3f}"] = {**st, **{f"bar_{k}": v for k, v in b.items()}}
    return out


def _hold_desc(ctx: "WR.WCtx", t: pd.DataFrame) -> dict:
    """보유 중 상단선 A–C 돌파 여부(낙관 경로 [진입, 청산] 고가 > 상단선 연장)·D 대비 위치·D → 체결 일수(기술)."""
    if t is None or t.empty:
        return {}
    H = ctx.A.H
    br = []
    for r in t.itertuples(index=False):
        ci, d, x, i0 = int(r.col_idx), int(r.d_row), int(r.x_row), int(r.i0)
        su = np.log(r.c / r.h0) / (int(r.i_c) - i0)
        js = np.arange(d, x + 1)
        u = np.exp(np.log(r.h0) + su * (js - i0))
        br.append(bool(np.nanmax(H[ci, js] - u) > 0))
    return {"upper_breach": float(np.mean(br)), "fill_vs_d_q": {q: float(t["fill_vs_d"].quantile(q)) for q in (0.1, 0.5, 0.9)},
            "d_to_fill_q": {q: float(t["d_to_fill"].quantile(q)) for q in (0.1, 0.5, 0.9)},
            "rise_q": {w_: {q: float(t[f"rise{w_}"].quantile(q)) for q in (0.1, 0.5, 0.9)} for w_ in (20, 40, 60)}}


def _arm_table(rows: dict, key: str, title: str) -> list:
    """전 팔 표 한 벌 — ``key`` 접두("", "cons_", "vx_", "vx_cons_"). R 은 총(비용 전)과 순 나란히."""
    r_, sh = WR._r, WR._sh
    pre = "vx_" if key.startswith("vx_") else ""
    L = [f"### {title}", "",
         "| 팔 | 체결 | n | 승/본전/패 | 손익비 | 손실 평균 총 R | 총 R 5% | 손실 평균 순 R | 순 R 5% | 순 1배 | 순 2배 | 손절 | 모호 | "
         "보유 중 상단선 돌파 | 보유 중 H0 돌파 | 정합 α [CI] |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in rows.items():
        def g(k, default=np.nan):
            return s.get(f"{key}{k}", default)
        nf = s.get(f"{pre}n_fills", 0)
        if not g("n", 0):
            L.append(f"| {n} | {nf} | 0 |" + " |" * 13)
            continue
        ex = g("exits", {}) or {}
        L.append(f"| {n} | {nf} | {g('n')} | {sh(g('win'))}/{sh(g('be'))}/{sh(g('lossc'))} | {r_(g('gl_ratio'))} | "
                 f"{r_(g('rg_loss_mean'), '{:+.2f}')} | {r_(g('rg_p5'), '{:+.2f}')} | {r_(g('avg_loss_R'), '{:+.2f}')} | "
                 f"{r_(g('r_p5'), '{:+.2f}')} | {_pct(g('net_mean'))} | {_pct(g('net2_mean'))} | "
                 f"{ex.get('stop', 0) + ex.get('stop_entry', 0):g} | {s.get(f'{pre}amb_n', 0)} | {sh(s.get(f'{pre}upper_breach', np.nan))} | "
                 f"{sh(s.get(f'{pre}h0_breach', np.nan))} | {_pct(s.get(f'{pre}mom_alpha', np.nan))} [{_pct(s.get(f'{pre}mom_lo', np.nan))}, "
                 f"{_pct(s.get(f'{pre}mom_hi', np.nan))}] |")
    return L + [""]


def _m(ok) -> str:
    return "✓" if ok else "✗"


def render(rows, funnel, sens, be1, pinfo, ports, lo_of, hi, n_led, n_prior, n_dsr, out_dir, debug, elapsed, timings, args,
           fac_sha, floor) -> list[str]:
    w = rows["WD"]
    pval = pinfo["p"]
    void = funnel.get("floor_rule_void")
    r_, sh = WR._r, WR._sh
    L = [f"# 쐐기 D(바닥 진입) — 실행 결과 {date.today().isoformat()} (러너 자동 생성, **탐색** — 판정 아님)", "",
         f"사전등록 `{PREREG_DOC}`. 사슬: {CHAIN}. 바를 넘어도 '전진 페이퍼·독립 표본 확인 후보' 까지만. ✓/✗ 는 산술 비교.", "",
         f"**유니버스 하한:** ADV20 ≥ {floor:,.0f} 백만원 — " + (
             "`wedge_d/FUNNEL.md` 사전 규칙 값." if void is False else "**사전 규칙상 WD VOID** — 아래는 전부 기술."
             if void is True else "명시 인자."), ""]
    if debug:
        L += [f"> **별도 출력 디렉터리** — 사건 {min(lo_of.values())}~{hi}, 널 {args.n_null}벌, 출력·원장 `{out_dir}`.", ""]
    cols = (("", f"주(p={pval:.3f})"), ("cons_", "보수 경계(p=1)"), ("vx_", "열람 제외 · 주"), ("vx_cons_", "열람 제외 · 보수"))

    def g(key, k, default=np.nan):
        return w.get(f"{key}{k}", default)

    def mom_cell(key):
        pre = "vx_" if key.startswith("vx_") else ""
        return f"{_pct(w.get(f'{pre}mom_alpha', np.nan))} [{_pct(w.get(f'{pre}mom_lo', np.nan))}]"

    L += ["## 바 산술 (WD) — 주 · 보수 경계(p=1) · 열람 제외 표본", "",
          f"열람 제외 = `{VIEWED_CSV}` 12 종목의 열람 창 ± {VIEWED_PAD} 거래일과 보유 [진입, 청산] 이 겹치는 체결 제외 "
          f"(WD 제외 {w.get('n_viewed', 0)}건). 바 3 은 모든 칸이 같은 널(전체 WD 에 맞춘 널, 칸별 p) — 열람 제외 칸도 같은 p95. "
          "바 5 의 R 바 = **총 R**(비용 전, 정정 1 M5); 순 R 은 아래 표에.", "",
          "| 바 | 문턱 | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols),
          "| 1 표본 | n ≥ {} ∧ 하위 ≥ {} | ".format(WR.B1_N, WR.B1_SUB_N)
          + " | ".join(f"n {g(k, 'n', 0)} · 하위 {g(k, 'n_sub', 0)} {_m(g(k, 'bar_b1'))}" for k, _ in cols) + " |",
          "| 2 순 2배 | > 0 ∧ 하위 > 0 ∧ 2020~21 제외 > 0 | "
          + " | ".join(f"{_pct(g(k, 'net2_mean'))} · {_pct(g(k, 'net2_sub'))} · {_pct(g(k, 'net2_ex2021'))} {_m(g(k, 'bar_b2'))}"
                       for k, _ in cols) + " |",
          "| 3 바닥 반전 널 | > p95 | "
          + " | ".join(f"{_pct(g(k, 'net2_mean'))} vs {_pct(w.get('null_p95_cons' if 'cons' in k else 'null_p95', np.nan))} "
                       f"{_m(g(k, 'bar_b3'))}" for k, _ in cols) + " |",
          "| 4 모멘텀 정합 | α CI 하단 > 0 | " + " | ".join(f"{mom_cell(k)} {_m(g(k, 'bar_b4'))}" for k, _ in cols) + " |",
          "| 5 손실·승률(1배) | 손익비 ≥ 2 ∧ 손실 평균 총 R ≥ −1 ∧ 총 R 5% ≥ −1.3 ∧ 순<−12% ≤ 5% ∧ 승률(본전 제외) > 손익분기 + 5%p | "
          + " | ".join(f"{r_(g(k, 'gl_ratio'))} · {r_(g(k, 'rg_loss_mean'), '{:+.2f}')}R · {r_(g(k, 'rg_p5'), '{:+.2f}')}R · "
                       f"{sh(g(k, 'big_loss'))} · {sh(g(k, 'bar_win_ex_be'))} vs {sh(g(k, 'bar_be_need'))} {_m(g(k, 'bar_b5'))}"
                       for k, _ in cols) + " |",
          "| 충족 | /5 | " + " | ".join(f"**{g(k, 'bar_n_met', '—')}**" for k, _ in cols) + " |",
          f"| (낙관 p=0) | /5 | 전체 {w.get('opt_bar_n_met', '—')} · 열람 제외 {w.get('vx_opt_bar_n_met', '—')} | | | |", "",
          "### 민감도", "", "| 칸 | n | 2배 순 | 손익비 | 손실 평균 총 R | 충족 주/보수/낙관 |", "|---|---|---|---|---|---|"]
    for name, s in sens.items():
        L.append(f"| {name} | {s.get('n', 0)} | {_pct(s.get('net2_mean', np.nan))} | {r_(s.get('gl_ratio', np.nan))} | "
                 f"{r_(s.get('rg_loss_mean', np.nan), '{:+.2f}')} | "
                 f"{s.get('bar_n_met', '—')}/{s.get('cons_bar_n_met', '—')}/{s.get('opt_bar_n_met', '—')} |")
    ci = pinfo.get("ci", (np.nan, np.nan))
    st = funnel["status"]
    L += ["", "## 분봉 p (바닥 반전 합성 사건)", "",
          f"구간 {pinfo.get('window')}, 확정 손절 {pinfo.get('n_confirmed', 0):,} 분모 밖, 모호 {pinfo['n_events']:,} → 판별 "
          f"{pinfo['n_classified']:,} · {pinfo.get('counts', {})} → **p = {pval:.3f}** [{r_(ci[0], '{:.3f}')}, {r_(ci[1], '{:.3f}')}].", "",
          "## 깔때기", "",
          f"- 실행 {elapsed / 60:.1f}분 (" + " · ".join(f"{k} {v:.0f}s" for k, v in timings.items()) + f") · 원장 {n_led}줄 · DSR N {n_dsr}",
          f"- 셋업 후보 {funnel['base']:,} → 첫 탈락: " + " · ".join(
              f"{k} {v:,}" for k, v in sorted(st.items(), key=lambda kv: Wd.WD_STATUS.index(kv[0]))),
          f"- WD 모양·진입 {funnel['wd_ok']:,} − H0 맥락 {funnel['context_fail']:,} = **{funnel['valid']:,}** · WD-up {funnel['valid_up']:,} · "
          f"WD-nc {funnel['valid_nc']:,} · 팔 셋업 행 {funnel['arms']}", "",
          "## 전 팔", "",
          "열람 제외 체결 수: " + " · ".join(f"{n} {rr.get('n_viewed', 0)}/{rr.get('n_fills', 0)}" for n, rr in rows.items()), ""]
    for key, title in (("", f"p 혼합(p = {pval:.3f})"), ("cons_", "보수 경계(p = 1)"), ("vx_", "열람 제외 · p 혼합"),
                       ("vx_cons_", "열람 제외 · 보수 경계(p = 1)")):
        L += _arm_table(rows, key, title)
    d = w.get("fill_vs_d_q", {})
    L += ["", "체결가의 D 대비 10/50/90: " + ("/".join(f"{d[q]:+.1%}" for q in (0.1, 0.5, 0.9)) if d else "—")
          + " · D → 체결 일수 10/50/90: " + ("/".join(f"{w['d_to_fill_q'][q]:.0f}" for q in (0.1, 0.5, 0.9)) if d else "—")
          + " · H0 전 상승률(저가 기준) " + " · ".join(
              f"{k}일 " + "/".join(f"{w['rise_q'][k][q]:+.0%}" for q in (0.1, 0.5, 0.9)) for k in (20, 40, 60)) if d else "", "",
          "## WD-be1 vs WD", ""]
    if be1.get("n"):
        L += ["| 지표 | WD | WD-be1 | 차이 [95% CI] |", "|---|---|---|---|"]
        for k, nm, fmt in (("net", "거래당 순 1배", _pct), ("R", "R", lambda x: r_(x, "{:+.3f}")), ("win", "승률", sh)):
            L.append(f"| {nm} | {fmt(be1[f'{k}_W'])} | {fmt(be1[f'{k}_be1'])} | {fmt(be1[f'{k}_diff'])} [{fmt(be1[f'{k}_lo'])}, "
                     f"{fmt(be1[f'{k}_hi'])}] |")
    L += [f"짝 {be1.get('n', 0)} · 한때 +1R 이상 → 손실: WD {be1.get('turned_W', 0)} · WD-be1 {be1.get('turned_be1', 0)}", "",
          f"널: 풀 {w.get('null_pool_rows', 0):,} · 벌 {w.get('null_k', 0)} · 중앙 {_pct(w.get('null_p50', np.nan))} · p95 "
          f"{_pct(w.get('null_p95', np.nan))} · 보수 p95 {_pct(w.get('null_p95_cons', np.nan))} · 체결/필요 중앙 "
          f"{sh(w.get('null_fill_ratio_med', np.nan))} · 어제 최저 추출 비율 중앙 {sh(w.get('null_yday_ratio_med', np.nan))}", "",
          "## 10 슬롯 포트폴리오 (기술)", "", "| 팔 | 편입 | 노출 | CAGR 1배 | MDD | 초과 Sharpe | 팩터 α t |", "|---|---|---|---|---|---|---|"]
    for n in ARMS:
        s = ports[n]["stats"]
        if s.get("port_taken"):
            L.append(f"| {n} | {s['port_taken']} | {sh(s['port_exposure'])} | {_pct(s['port_cagr'])} | {_pct(s['port_mdd'])} | "
                     f"{r_(s.get('port_sharpe_exc', np.nan))} | {r_(s.get('fac_alpha_t', np.nan))} |")
    for n in ("WD", "WD-up"):
        s = {k[3:]: v for k, v in rows[n].items() if k.startswith("vx_port_") or k.startswith("vx_fac_")}
        if s.get("port_taken"):
            L.append(f"| {n} 열람 제외 | {s['port_taken']} | {sh(s['port_exposure'])} | {_pct(s['port_cagr'])} | {_pct(s['port_mdd'])} | "
                     f"{r_(s.get('port_sharpe_exc', np.nan))} | {r_(s.get('fac_alpha_t', np.nan))} |")
    L += ["", "## 구현 메모 (결과 전 결정)", "",
          "- 정의 해석은 `research/signals/wedge.py` 하단 쐐기 D 절 a~h. 네 점 연속 = 확정 고점이 정확히 A(H0)·C 둘. 같은 쐐기 = (종목, A, C).",
          "- 바닥 진입: 주문가 = D 봉 고가 + 1호가, 9.5%·3% 갭 기준 = 주문가, 진입일 모호 = 시가 ≤ D 봉 고가 ∧ 저가 ≤ 손절가 ∧ 종가 > 손절가. "
          "2R 전 돌파 실패 규칙 없음(``fb_days`` 사실상 무한). D ≤ 0.99·B. 같은 (A, C) 에서 D 봉 고가 + 1호가를 이미 넘었으면(두 번째 관통) 그 D 로 "
          "사지 않음. 손절·본전 체결 = min(시가, 손절가) − 1호가(매도 미끄럼 없음). WD-up = 같은 모양·같은 청산(WD 매개변수), 진입만 상단선 A–C "
          "첫 돌파 매수정지(종가 < U(d)·0.92 띠, d − 행(D) ≤ 5 없음), 주문 수명 = 셋업 유효일. WD-nocap = WD 에서 120 거래일 상한만 끔.",
          "- 널: 맥락 통과 ∧ WD 아님 종목-일, 직전 k 일 최고 고가 이후 최저 저가(되돌림 새 저점, 동률 늦은 봉, 눌림 8~40%) 봉 고가 + 1호가, "
          "손절 = 그 저가 − 1호가, d − 행 ≤ 5, 두 번째 관통 차단, 종가 < 주문가, 9.5%·2%·3%·상한가·정지·같은 청산. 층화 월 × d 3분위 × "
          "주문가/MA50 3분위, 칸 안에서 어제가 최저가인 후보 우선. 분봉 p 합성 사건도 같은 정의(격자 k).",
          f"- R: 총 R = (청산가 − 체결가)/(체결가 − 손절가) = 총수익/d(비용 전), 순 R = 순/d. 열람 제외 = `{VIEWED_CSV}` ± {VIEWED_PAD} 거래일.",
          "- 보유 중 상단선 돌파 = 낙관 경로 [진입, 청산] 고가 > 상단선 A–C 연장. 차트 JSON 봉 = H0 − 60 ~ 청산 + 25 (MA50 포함).", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
