#!/usr/bin/env python
"""미너비니 원전 청산 프로토콜 — 사전등록 러너(주 1칸 + 민감도 6칸).

사전등록: ``research/logs/sepa_explore/MINERVINI_EXITS.md`` (결과 보기 전 커밋 36b7601). 진입·손절·익절·관리·대조·
합격 바·민감도·기술 항목을 그대로 옮긴다. 이 러너는 **리포터**다 — 바 숫자를 계산해 ``MINERVINI_EXITS_RESULTS.md``
에 적을 뿐 판정하지 않는다(판정은 사람이).

- 진입: 스윙 3호 뼈대 S(유니버스 ∧ 추세 템플릿 ∧ RS≥70 ∧ 피벗 돌파·거래량 1.5×) ∧ breadth > 0.5 ∧ 실적 문(PEAD 순위
  ≥ 0.70) ∧ VCP 수축(:mod:`research.signals.minervini_exits`).
- 청산: :func:`research.signals.minervini_exits.simulate_minervini` (기술적 손절 3~10%·2R 절반 익절+본전·MA50 추적·
  20일 시간 손절·120일).
- 주 대조: 모멘텀 정합(``MOMENTUM.md`` — 같은 날 × 시장 × 12-1 모멘텀 10분위 × ADV20 2분위, 유니버스 풀, 자기 종목
  제외, 5배, seed 0, 날짜군집 부트스트랩 2,000). 대조에도 **같은 프로토콜**(:func:`control_minervini` — 자기 수축 저점).
  보조: 기존 정합(ADV20 4분위 × 직전 5일 수익 4분위 × 시장).
- 원장: ``research/logs/sepa_explore/minervini_exits/TRIALS.jsonl`` (7칸, 결과 전에 적는다). DSR N = 스윙 3호 원장 +
  탐색 원장 + 이 원장.

기존 러너(``sepa_confluence_gate``·``sepa_explore``·``sepa_momentum``)는 import 만 한다.

Run:  uv run --extra pg python research/experiments/minervini_exits_run.py
디버그(구간 축소 — 탐색 디렉터리·원장을 더럽히지 않는다):
      ... minervini_exits_run.py --event-lo 2020-01-01 --event-hi 2020-12-31 --out <scratch>/mx_debug
"""

from __future__ import annotations

import argparse
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

from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402
from sepa_confluence_gate import (  # noqa: E402
    Data,
    _costs,
    _drop_top,
    _key,
    _log,
    _pct,
    attach_features,
    e_pass,
    index_hedge,
    sub_alpha,
)
from sepa_explore import EVENT_HI, EVENT_LO, GATES, HALVES, HEDGE_LO, _base  # noqa: E402
from sepa_momentum import (  # noqa: E402
    MOM_N_Q,
    STRATA_MOM,
    STRATA_OLD,
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
    vcp_contraction,
    vcp_ok,
)
from research.signals.sepa import build_skeleton_events, simulate_sepa  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.technical import moving_average  # noqa: E402
from swing_it.storage import connect, db_default  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "minervini_exits"
RESULTS_MD = "MINERVINI_EXITS_RESULTS.md"
PREREG_DOC = "research/logs/sepa_explore/MINERVINI_EXITS.md (36b7601)"
VOID_MIN_N = 100
GL_MIN = 2.0                                  # 바 3: 평균 이익 / 평균 손실 ≥ 2
DSR_MIN = 0.5                                 # 바 6
PRIOR_LEDGERS = (("sepa_confluence", "research/logs"), ("sepa_explore", "research/logs"))
SIM_COLS = ["code", "event_day", "market", "pivot", "q_adv", "q_ret5"]

# --- 칸 (사전등록 "프로토콜 1개" + "민감도 시도 1 + 6" — 순서가 표의 순서) ---
PRIMARY = "주(사전등록)"
CELLS: dict[str, dict] = {
    PRIMARY: {},
    "VCP 끔": {"vcp": False},
    "실적 문 E": {"gate": "E"},
    "손절 고정 8%": {"fixed_stop": 0.08},
    "익절 3R": {"take_r": 3.0},
    "나머지 추적 MA20": {"trail": "ma20"},
    "시간 손절 끔": {"time_stop_days": None},
}
#: 시뮬레이터 인자로 넘기는 키(나머지는 진입 정의·기록용).
SIM_KEYS = ("stop_floor", "stop_cap", "stop_mult", "fixed_stop", "take_r", "time_stop_days", "max_hold", "chase",
            "limit_up_skip")


def cell_config(name: str, over: dict, lo: str = EVENT_LO, hi: str = EVENT_HI) -> dict:
    """한 칸의 전체 config(원장 지문용). 스윙 3호 판정 config(유니버스·뼈대·체결·비용) 위에 프로토콜을 덮는다."""
    c = _base(event_lo=lo, event_hi=hi, rs_min=70.0, require_breakout=True, breadth_min=0.5)
    for k in ("stop", "ma50_exit", "matched", "e_mode", "e_min_yoy", "e_max_age_days", "pead_rank_min"):
        c.pop(k, None)                         # 스윙 3호 청산·대조·실적 문 값은 이 실험이 쓰지 않는다
    c.update({"prereg": PREREG_DOC, "cell": name, "gate": "pead≥0.70", "vcp": True, "fixed_stop": None,
              "trail": "ma50", "fixed": "VCP 깊이 t−10..t−1 ≤ 12% ∧ 거래량 비(t−10..t−1 / t−60..t−11) < 0.8; "
                                         "손절 = min(low[t−10..t])×0.99 → d∈[3%,10%]",
              "matched": "모멘텀 정합(MOMENTUM.md): 같은 날 × 시장 × 12-1 모멘텀 10분위 × ADV20 2분위, 유니버스 풀, 자기 제외, "
                         "n_per 5, seed 0, 같은 프로토콜(control_minervini), 날짜군집 부트스트랩 2000",
              **{k: v for k, v in PREREG_MINERVINI.items() if k != "chase" and k != "limit_up_skip"}})
    c.update(over)
    c.update(GATES[c["gate"]])
    return c


def sim_kwargs(cfg: dict) -> dict:
    return {k: cfg[k] for k in SIM_KEYS}


def rules_key(cfg: dict) -> tuple:
    """대조 풀 캐시 키 — 시뮬 규칙 + 추적 MA."""
    return tuple(cfg[k] for k in SIM_KEYS) + (cfg["trail"],)


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_minervini_exits_run_helpers.py)
# ---------------------------------------------------------------------------

def longest_losing_streak(x) -> int:
    """연속 손실(값 < 0) 최장 길이. 순서는 넘긴 순서 그대로(호출부가 진입일 순으로 정렬한다). 0 은 연패를 끊는다."""
    best = cur = 0
    for v in np.asarray(x, float):
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def gain_loss(x) -> dict:
    """승률(값 > 0)·평균 이익(> 0 의 평균)·평균 손실(< 0 의 평균, 음수)·이익/손실 비(평균 이익 / |평균 손실|).

    0 은 이익도 손실도 아니다(``n_zero`` 로 따로 센다 — 본전 청산은 비용 차감 뒤라 실제로는 거의 없다)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    g, lo = x[x > 0], x[x < 0]
    ag = float(g.mean()) if len(g) else np.nan
    al = float(lo.mean()) if len(lo) else np.nan
    ratio = ag / abs(al) if len(g) and len(lo) else np.nan
    return {"win": float(np.mean(x > 0)) if len(x) else np.nan, "avg_gain": ag, "avg_loss": al, "gl_ratio": ratio,
            "n_gain": int(len(g)), "n_loss": int(len(lo)), "n_zero": int((x == 0).sum())}


def worst_decile_mean(x) -> float:
    """하위 10%(최소 1건) 평균."""
    s = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])
    if not len(s):
        return np.nan
    k = max(1, int(np.ceil(0.1 * len(s))))
    return float(s[:k].mean())


def years_positive(values: pd.Series, days: pd.Series) -> "tuple[int, int]":
    """사건 연도별 평균 > 0 인 연도 수 / 사건이 있는 연도 수."""
    yr = values.groupby(pd.to_datetime(days).dt.year).mean()
    return int((yr > 0).sum()), int(len(yr))


def bar_numbers(s: dict) -> dict:
    """사전등록 합격 바 1~6 과 VOID 의 산술 조건(숫자 비교만 — 판정은 사람이)."""
    def gt0(v):
        return bool(np.isfinite(v) and v > 0)

    out = {
        "void": s["n"] < VOID_MIN_N,
        "b1": gt0(s.get("mom_lo", np.nan)),
        "b2": gt0(s.get("net_mean", np.nan)) and gt0(s.get("net2_mean", np.nan)),
        "b3": bool(np.isfinite(s.get("gl_ratio", np.nan)) and s["gl_ratio"] >= GL_MIN) and gt0(s.get("r_net", np.nan)),
        "b4": gt0(s.get("drop_top1", np.nan)),
        "b5": gt0(s.get("mom_h1_alpha", np.nan)) and gt0(s.get("mom_h2_alpha", np.nan))
        and s.get("years_n", 0) > 0 and s.get("years_pos", 0) * 2 > s.get("years_n", 0),
        "b6": bool(np.isfinite(s.get("dsr_prob", np.nan)) and s["dsr_prob"] > DSR_MIN),
    }
    out["n_met"] = sum(out[f"b{i}"] for i in range(1, 7))
    return out


# ---------------------------------------------------------------------------
# 대조 풀
# ---------------------------------------------------------------------------

class MxPools:
    """(규칙, 날짜) 별 그날 **유니버스 전 종목**을 같은 프로토콜(``control_minervini``)로 시뮬한 풀 + 층 열."""

    def __init__(self, D: Data, trail: dict, c_low: pd.DataFrame, strata: dict[str, pd.DataFrame]):
        self.D, self.trail, self.c_low, self.strata = D, trail, c_low, strata
        self.cache: dict[tuple, pd.DataFrame] = {}

    def build(self, key: tuple, cfg: dict, days) -> pd.DataFrame:
        D, p = self.D, self.D.p
        days = pd.DatetimeIndex(sorted(set(pd.to_datetime(days))))
        di_days = p.days.get_indexer(days)
        r, c = np.nonzero(D.uni.to_numpy()[di_days])
        di = di_days[r]
        pairs = pd.DataFrame({"code": p.close.columns[c], "event_day": p.days[di],
                              "pivot": D.feats["pivot"].to_numpy()[di, c],
                              "q_adv": D.q_adv.to_numpy()[di, c], "q_ret5": D.q_ret5.to_numpy()[di, c]})
        for k, panel in self.strata.items():
            pairs[k] = panel.to_numpy(dtype=float)[di, c]
        ct, sk = control_minervini(p.open, p.high, p.low, p.close, self.trail[cfg["trail"]], pairs, volume=p.volume,
                                   market=p.market, contraction_low=self.c_low, return_skipped=True,
                                   **sim_kwargs(cfg))
        cols = ["code", "event_day", "ret", "market", "q_adv", "q_ret5", *self.strata, "exit_reason", "partial_hit"]
        pool = ct[cols].rename(columns={"event_day": "date"}).copy()
        pool["date"] = pd.to_datetime(pool["date"])
        pool.index = _key(pool["date"], pool["code"])
        pool.attrs["n_pairs"] = len(pairs)
        pool.attrs["skips"] = sk["skip_reason"].value_counts().to_dict()
        self.cache[key] = pool
        return pool


# ---------------------------------------------------------------------------
# 칸 숫자
# ---------------------------------------------------------------------------

def _alpha(prefix: str, a) -> dict:
    if a is None or not np.isfinite(a.alpha):
        return {f"{prefix}alpha": np.nan, f"{prefix}lo": np.nan, f"{prefix}hi": np.nan, f"{prefix}n": 0,
                f"{prefix}unmatched": 0}
    return {f"{prefix}alpha": a.alpha, f"{prefix}lo": a.ci_low, f"{prefix}hi": a.ci_high, f"{prefix}n": a.n_trades,
            f"{prefix}unmatched": a.n_unmatched}


def _halves(t: pd.DataFrame) -> list[pd.Series]:
    return [(t["event_day"] >= pd.Timestamp(lo)) & (t["event_day"] < pd.Timestamp(hi)) for _, lo, hi in HALVES]


def trade_stats(t: pd.DataFrame, n_dsr: int) -> dict:
    """비용 차감 순수익 기준 서술 숫자(대조 제외)."""
    if t.empty:
        return {"n": 0}
    t = t.sort_values(["entry_day", "code"])
    net = t["net"].to_numpy(float)
    s = {"n": len(t), "gross_mean": float(t["ret"].mean()), "net_mean": float(net.mean()),
         "net_median": float(np.median(net)), "net2_mean": float((t["ret"] - 2 * t["cost"]).mean()),
         "cost_bp": float(t["cost"].mean() * 1e4), "drop_top1": _drop_top(net),
         "r_net": float((t["net"] / t["d"]).mean()) if "d" in t else np.nan,
         "hold_median": float(t["hold_days"].median()), "loss_streak": longest_losing_streak(net),
         "worst_decile": worst_decile_mean(net)}
    s.update(gain_loss(net))
    s["years_pos"], s["years_n"] = years_positive(t["net"], t["event_day"])
    mo = t.groupby(t["entry_day"].dt.to_period("M"))["net"].mean()
    s["months_pos"], s["months_n"] = int((mo > 0).sum()), int(len(mo))
    if "d" in t:
        s["d_median"] = float(t["d"].median())
        s["partial_rate"] = float(t["partial_hit"].astype(bool).mean())
    d = deflated_sharpe_from_sample(net, n_dsr) if len(net) > 2 else None
    s["dsr_prob"] = float(d["prob_deflated_sharpe"]) if d is not None else np.nan
    s["sharpe_trade"] = float(d["observed_sharpe"]) if d is not None else np.nan
    s["dsr_sr0"] = float(d["expected_max_sharpe_h0"]) if d is not None else np.nan
    return s


def matched_stats(t: pd.DataFrame, pool: pd.DataFrame) -> "tuple[dict, dict, dict]":
    """모멘텀(주)·기존(보조) 정합 alpha + 전반/후반·연도별 모멘텀 alpha. ``(숫자, 모멘텀 m, 연도별 숫자)``."""
    m_mom = matched_on(t, pool, STRATA_MOM)
    m_old = matched_on(t, pool, STRATA_OLD)
    s = {**_alpha("mom_", m_mom["ma"]), **_alpha("old_", m_old["ma"])}
    for h, mk in zip(("mom_h1_", "mom_h2_"), _halves(t), strict=True):
        s.update(_alpha(h, sub_alpha(m_mom, mk) if mk.any() else None))
    yrs = {}
    for y in sorted(t["event_day"].dt.year.unique()):
        mk = t["event_day"].dt.year == y
        yrs[int(y)] = {"n": int(mk.sum()), "net_mean": float(t.loc[mk, "net"].mean()),
                       **_alpha("mom_", sub_alpha(m_mom, mk))}
    s["mom_years_pos"] = int(sum(1 for v in yrs.values() if np.isfinite(v["mom_alpha"]) and v["mom_alpha"] > 0))
    return s, m_mom, yrs


def nostop_compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """같은 진입 키에서 프로토콜(a) vs 손절 없음 + 120일(b). 공통 키만."""
    k = a.index.intersection(b.index)
    out = {"n_a": len(a), "n_b": len(b), "n_common": len(k)}
    for tag, t in (("a", a.loc[k]), ("b", b.loc[k])):
        net = t.sort_values(["entry_day", "code"])["net"].to_numpy(float)
        out[tag] = {"mean": float(net.mean()) if len(net) else np.nan,
                    "median": float(np.median(net)) if len(net) else np.nan,
                    "win": float(np.mean(net > 0)) if len(net) else np.nan,
                    "worst_decile": worst_decile_mean(net), "p10": float(np.percentile(net, 10)) if len(net) else np.nan,
                    "best_decile": -worst_decile_mean(-net) if len(net) else np.nan,
                    "min": float(net.min()) if len(net) else np.nan, "streak": longest_losing_streak(net),
                    "hold_median": float(t["hold_days"].median()) if len(t) else np.nan,
                    "drop_top1": _drop_top(net)}
    return out


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _a(s: dict, pre: str) -> str:
    a = s.get(f"{pre}alpha", np.nan)
    if not np.isfinite(a):
        return "nan"
    return f"{_pct(a)} [{_pct(s.get(f'{pre}lo'))}, {_pct(s.get(f'{pre}hi'))}]"


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _f(x, fmt: str = "{:.2f}") -> str:
    return "nan" if x is None or not np.isfinite(x) else fmt.format(x)


def _dist(d: dict, order) -> str:
    return " · ".join(f"{k} {d.get(k, 0)}" for k in order if d.get(k, 0)) or "—"


EXIT_ORDER = ("stop", "breakeven", "trail", "time_stop", "time", "delisted", "data_end")
SKIP_ORDER = ("skip_overlap", "skip_chase", "skip_limit_up", "skip_no_open", "skip_below_low", "skip_no_contraction",
              "skip_no_next_day")


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
    lo, hi = args.event_lo or EVENT_LO, args.event_hi or EVENT_HI
    cfgs = {name: cell_config(name, over, lo, hi) for name, over in CELLS.items()}
    assert len(cfgs) == 7

    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()

    # --- 원장: 7칸을 결과 전에 전부 적는다 ---
    for name, c in cfgs.items():
        record_trial(LABEL, c, logs_dir=out_dir, note=f"미너비니 청산 칸: {name}")
    n_led = count_trials(LABEL, logs_dir=out_dir)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in PRIOR_LEDGERS}
    n_dsr = n_led + sum(n_prior.values())
    _log(f"원장 {LABEL}: {n_led}칸 · " + " · ".join(f"{k} {v}" for k, v in n_prior.items()) + f" → DSR N {n_dsr}", t0)

    load_env_db()
    con = connect(db_default())
    D = Data(con, t0)
    p = D.p

    # --- 패널: VCP·수축 저점·추적 MA·모멘텀 층 (전부 d 종가까지) ---
    depth, vratio = vcp_contraction(p.high, p.low, p.volume, window=PREREG_MINERVINI["vcp_window"],
                                    vol_window=PREREG_MINERVINI["vcp_vol_window"])
    vcp = vcp_ok(depth, vratio, max_depth=PREREG_MINERVINI["vcp_max_depth"],
                 max_vol_ratio=PREREG_MINERVINI["vcp_max_vol_ratio"])
    c_low = contraction_low_level(p.low, PREREG_MINERVINI["vcp_window"])
    trail = {"ma50": D.feats["ma50"], "ma20": moving_average(p.close, 20)}
    mom = momentum_12_1(p.close)
    strata = {"q_mom": day_market_quantiles(mom, D.uni, p.market, n_q=MOM_N_Q),
              "adv2": day_market_median_half(D.adv20, D.uni, p.market)}
    del mom
    _log("VCP·수축 저점·MA20·모멘텀 층", t0)

    # --- 사건 master: 뼈대 S ∧ breadth > 0.5 (사건 구간) + 피처 ---
    base = cfgs[PRIMARY]
    master = build_skeleton_events(p, base, universe=D.uni, feats=D.feats)
    master = attach_features(D, master)
    di, ci = master["day_idx"].to_numpy(), master["col_idx"].to_numpy()
    master["vcp"] = vcp.to_numpy()[di, ci]
    master["vcp_depth"] = depth.to_numpy()[di, ci]
    master["vcp_vratio"] = vratio.to_numpy()[di, ci]
    master = master.sort_values(["event_day", "code"]).reset_index(drop=True)
    gate_ok = {g: e_pass(master, cfgs_g).to_numpy(bool)
               for g, cfgs_g in (("pead≥0.70", cfgs[PRIMARY]), ("E", cfgs["실적 문 E"]))}
    _log(f"사건 master(S ∧ breadth) {len(master):,} · PEAD≥0.70 {gate_ok['pead≥0.70'].sum():,} · E {gate_ok['E'].sum():,} "
         f"· VCP {int(master['vcp'].sum()):,}", t0)

    def events_for(c: dict) -> pd.DataFrame:
        mk = gate_ok[c["gate"]].copy()
        if c["vcp"]:
            mk &= master["vcp"].to_numpy(bool)
        return master[mk].reset_index(drop=True)

    # --- 1차: 칸별 트레이드 ---
    trades, skips, evs = {}, {}, {}
    for name, c in cfgs.items():
        ev = events_for(c)
        evs[name] = ev
        t, sk = simulate_minervini(p.open, p.high, p.low, p.close, trail[c["trail"]], ev[SIM_COLS], volume=p.volume,
                                   contraction_low=c_low, return_skipped=True, **sim_kwargs(c))
        if not t.empty:
            for col in ("event_day", "entry_day", "exit_day"):
                t[col] = pd.to_datetime(t[col])
            t["cost"] = _costs(t)
            t["net"] = t["ret"] - t["cost"]
            t.index = _key(t["event_day"], t["code"])
            t = attach_strata(t, "event_day", p.days, p.close.columns, strata)
        trades[name], skips[name] = t, sk["skip_reason"].value_counts().to_dict()
        _log(f"{name}: 사건 {len(ev):,} · 트레이드 {len(t):,}", t0)

    # --- 2차: 대조 풀(규칙별, 그 규칙 칸들의 사건일 합집합) ---
    pools = MxPools(D, trail, c_low, strata)
    by_key: dict[tuple, list[str]] = {}
    for name, c in cfgs.items():
        by_key.setdefault(rules_key(c), []).append(name)
    pool_of: dict[str, pd.DataFrame] = {}
    for key, names in by_key.items():
        days = pd.concat([trades[n]["event_day"] for n in names if not trades[n].empty] or [pd.Series([], dtype="M8[ns]")])
        if not len(days):
            continue
        pool = pools.build(key, cfgs[names[0]], days.unique())
        for n in names:
            pool_of[n] = pool
        _log(f"대조 풀 {names}: 쌍 {pool.attrs['n_pairs']:,} → 시뮬 {len(pool):,}", t0)

    # --- 3차: 칸 숫자 ---
    rows, years = {}, {}
    for name, c in cfgs.items():
        t = trades[name]
        s = {"cell": name, "n_events": len(evs[name]), **trade_stats(t, n_dsr)}
        if not t.empty:
            ms, _, yrs = matched_stats(t, pool_of[name])
            s.update(ms)
            years[name] = yrs
            hedged = index_hedge(con, D, t)
            ok = hedged.notna()
            s["hedge_n"] = int(ok.sum())
            s["hedge_net_mean"] = float((hedged[ok] - t.loc[ok, "cost"]).mean()) if ok.any() else np.nan
            s["exits"] = t["exit_reason"].value_counts().to_dict()
            s["exit_net"] = t.groupby("exit_reason")["net"].mean().to_dict()
            s["pool_n"] = len(pool_of[name])
            s["pool_exits"] = pool_of[name]["exit_reason"].value_counts(normalize=True).to_dict()
        s["skips"] = skips[name]
        s.update({f"bar_{k}": v for k, v in bar_numbers(s).items()})
        rows[name] = s
        _log(f"{name}: n {s['n']} · 순평균 {_pct(s.get('net_mean', np.nan))} · 모멘텀 α {_a(s, 'mom_')}", t0)

    # --- 기술: 같은 진입에 "손절 없음 + 120일" ---
    nostop = {}
    for name in (PRIMARY, "VCP 끔"):
        ev = evs[name]
        b = simulate_sepa(p.open, p.high, p.low, p.close, D.nan_ma, ev[SIM_COLS], volume=p.volume, stop=1.0,
                          max_hold=120, chase=PREREG_MINERVINI["chase"], limit_up_skip=PREREG_MINERVINI["limit_up_skip"])
        if b.empty or trades[name].empty:
            continue
        for col in ("event_day", "entry_day", "exit_day"):
            b[col] = pd.to_datetime(b[col])
        b["cost"] = _costs(b)
        b["net"] = b["ret"] - b["cost"]
        b.index = _key(b["event_day"], b["code"])
        nostop[name] = nostop_compare(trades[name], b)
    _log("손절 없음 + 120일 비교", t0)

    elapsed = time.time() - t0
    L = render(rows, years, nostop, cfgs, master, gate_ok, n_led, n_prior, n_dsr, lo, hi, out_dir, debug, elapsed, p)
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    flat = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in rows.values()])
    flat.to_csv(os.path.join(out_dir, "minervini_exits_cells.csv"), index=False)
    allt = pd.concat([t.assign(cell=n) for n, t in trades.items() if not t.empty], ignore_index=True)
    allt.to_csv(os.path.join(out_dir, "minervini_exits_trades.csv"), index=False)

    print(f"\n=== 미너비니 원전 청산 — {elapsed / 60:.1f}분 · DSR N {n_dsr} ===")
    for name, s in rows.items():
        print(f"{name:14s} n {s['n']:4d} · 순평균 {_pct(s.get('net_mean', np.nan))} · 2배비용 "
              f"{_pct(s.get('net2_mean', np.nan))} · 이익/손실 {_f(s.get('gl_ratio', np.nan))} · R {_f(s.get('r_net', np.nan), '{:+.3f}')} "
              f"· 모멘텀 α {_a(s, 'mom_')} · DSR {_f(s.get('dsr_prob', np.nan), '{:.1%}')} · 바 충족 {s['bar_n_met']}/6"
              f"{' · VOID(n<100)' if s['bar_void'] else ''}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def render(rows, years, nostop, cfgs, master, gate_ok, n_led, n_prior, n_dsr, lo, hi, out_dir, debug, elapsed, p):
    L = [f"# 미너비니 원전 청산 프로토콜 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "사전등록 `MINERVINI_EXITS.md`(36b7601, 결과 보기 전) 의 진입·청산·대조·합격 바·민감도·기술을 그대로 계산한 숫자다. "
         "합격 여부는 여기 적지 않는다 — 판정은 사람이 쓴다. 표의 ✓/✗ 는 사전등록 문턱과의 **산술 비교**일 뿐이다.", ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 구간 {lo}~{hi}, 출력·원장 `{out_dir}`. 측정으로 쓰지 않는다.", ""]
    pm = gate_ok["pead≥0.70"]
    vm = master["vcp"].to_numpy(bool)
    L += ["## 표본", "",
          f"- 사건 구간 {lo} ~ {hi} · 가격 패널 {p.days[0].date()}~{p.days[-1].date()} · 실행 {elapsed / 60:.1f}분",
          f"- 사건 master(뼈대 S ∧ breadth > 0.5, 유니버스): **{len(master):,}** · ∧ PEAD≥0.70 **{int(pm.sum()):,}** · "
          f"∧ E {int(gate_ok['E'].sum()):,} · ∧ VCP {int(vm.sum()):,} · PEAD∧VCP **{int((pm & vm).sum()):,}** · "
          f"E∧VCP {int((gate_ok['E'] & vm).sum()):,}",
          f"- VCP 성분(PEAD≥0.70 사건): 깊이 ≤ 12% {int((master['vcp_depth'][pm] <= 0.12).sum()):,} · 거래량 비 < 0.8 "
          f"{int((master['vcp_vratio'][pm] < 0.8).sum()):,} · 깊이 중앙값 {master['vcp_depth'][pm].median():.3f} · 거래량 비 중앙값 "
          f"{master['vcp_vratio'][pm].median():.3f}",
          f"- 원장 `{out_dir}/{LABEL}/TRIALS.jsonl` 칸 수 **{n_led}** · DSR N = {n_led} + "
          + " + ".join(f"{k} {v}" for k, v in n_prior.items()) + f" = **{n_dsr}**", ""]

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          "- **시뮬레이터:** `research/signals/minervini_exits.py`(6ce5b7c) 모듈 독스트링 1~11 그대로. 요점: VCP 거래량 비의 "
          "'직전 50일' = 수축 창 직전 t−60..t−11; 수축 저점 = min(low[t−10..t]) (돌파일 포함); d 를 [3%,10%] 로 자른 뒤 "
          "손절가 = 진입가×(1−d); 같은 날 손절·익절 → 손절; 익절 전엔 MA 추적 없음; 본전 손절은 익절 다음 거래일부터"
          "(갭 상승 시가 익절일만 그날); 시간 손절은 20일째 한 번.",
          "- **실적 문:** PEAD 순위 = `Data.earn.pead_rank`(스윙 3호·탐색과 같은 유니버스 안 순위). E = 탐색 격자의 E"
          "(YoY ≥ 25%, 나이 ≤ 140일, `e_pass`).",
          "- **손절 고정 8%:** 수축 저점 거르기(시가 ≤ 수축 저점 → 버림)는 그대로 — 칸끼리 진입 집합을 같게.",
          "- **대조:** 풀 = 트레이드 사건일마다 그날 유니버스 전 종목을 **같은 칸 규칙**으로 `control_minervini`(자기 수축 저점·"
          "피벗 = 직전 50일 최고 고가, 겹침 허용). 층·추출은 `sepa_momentum.matched_on`(모멘텀 = `STRATA_MOM`, 기존 = "
          "`STRATA_OLD`) 그대로. alpha 는 비용 전 총수익(탐색·모멘텀 재측정과 같다).",
          "- **비용:** `korea.cost_rt(청산일, 시장)` 1회(사전등록: 분할 청산도 총량 같음). 청산일 = 마지막(나머지 절반) 청산일.",
          "- **바 3:** 평균 이익·손실은 비용 차감 순수익(> 0 / < 0), 비 = 평균 이익 / |평균 손실|. 기대값 R = mean(순수익 / d).",
          "- **바 5 연도:** 사건 연도별 비용 차감 순평균 > 0 인 연도가 과반(탐색 '연도 양수' 와 같은 정의). 연도별 모멘텀 alpha "
          "양수 연도 수도 함께 적는다.",
          "- **바 6 DSR:** 건당 비용 차감 순수익, core `deflated_sharpe_from_sample`(시행별 Sharpe 표준편차 = 코어 기본), "
          "N = 원장에서 읽은 값.",
          f"- **지수 헤지:** 진입 ≥ {HEDGE_LO} 트레이드의 (총수익 − 진입일 시가→마지막 청산일 종가 KOSPI/KOSDAQ 지수 수익) − 비용. "
          "분할 청산의 절반은 더 일찍 나가므로 헤지 구간이 근사다.",
          "- **손절 없음 + 120일 비교:** 같은 사건 표에 `sepa.simulate_sepa(stop=1.0, max_hold=120, MA 청산 없음)`. 겹침 규칙이 "
          "보유 기간에 따라 다르게 걸리고(프로토콜은 일찍 나가 재진입이 더 많다) 수축 저점 거르기는 프로토콜에만 있으므로 "
          "**공통 사건 키**에서만 나란히 놓는다.", ""]

    L += ["## 합격 바 숫자 (사전등록 1~6 + VOID — 숫자만, 판정 아님)", "",
          "| 칸 | n (VOID < 100) | (1) 모멘텀 α [CI] — 하단 > 0 | (2) 순평균 · 2배 비용 > 0 | (3) 이익/손실 ≥ 2 · 순 R > 0 | "
          "(4) 상위1% 제거 > 0 | (5) 전반 α · 후반 α > 0 · 연도 양수 과반 | (6) DSR > 0.5 | 산술 충족 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows.items():
        if s["n"] == 0:
            L.append(f"| {name} | 0 VOID | | | | | | | 0/6 |")
            continue
        L.append(
            f"| {name} | {s['n']} {'VOID' if s['bar_void'] else ''} | {_a(s, 'mom_')} {_mark(s['bar_b1'])} | "
            f"{_pct(s['net_mean'])} · {_pct(s['net2_mean'])} {_mark(s['bar_b2'])} | {_f(s['gl_ratio'])} · "
            f"{_f(s['r_net'], '{:+.3f}')} {_mark(s['bar_b3'])} | {_pct(s['drop_top1'])} {_mark(s['bar_b4'])} | "
            f"{_pct(s['mom_h1_alpha'])} · {_pct(s['mom_h2_alpha'])} · {s['years_pos']}/{s['years_n']} {_mark(s['bar_b5'])} | "
            f"{_f(s['dsr_prob'], '{:.1%}')} {_mark(s['bar_b6'])} | {s['bar_n_met']}/6 |")
    L.append("")

    L += ["## 기술 — 청산·분포 (원전 목표: 평균 손실 5~6%, 이익/손실 ≥ 2, 평균 보유 ~20일)", "",
          "| 칸 | n | 승률 | 평균 이익 | 평균 손실 | 순중앙값 | 하위10% 평균 | 보유 중앙값 | 최장 연패 | d 중앙값 | 절반 익절 비율 | "
          "월 양수 | 기존 α [CI] | 모멘텀 α 양수 연도 | 지수 헤지 순평균 (n) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows.items():
        if s["n"] == 0:
            continue
        hedge = f"{_pct(s['hedge_net_mean'])} ({s['hedge_n']})" if s.get("hedge_n") else "—"
        L.append(f"| {name} | {s['n']} | {s['win']:.1%} | {_pct(s['avg_gain'])} | {_pct(s['avg_loss'])} | "
                 f"{_pct(s['net_median'])} | {_pct(s['worst_decile'])} | {s['hold_median']:.0f} | {s['loss_streak']} | "
                 f"{_pct(s['d_median'])} | {s['partial_rate']:.1%} | {s['months_pos']}/{s['months_n']} | {_a(s, 'old_')} | "
                 f"{s['mom_years_pos']}/{len(years.get(name, {}))} | {hedge} |")
    L += ["", "| 칸 | 사건 | 버림(사유) | 청산 사유 (건) | 청산 사유별 순평균 | 대조 풀 n · 손절 비율 |", "|---|---|---|---|---|---|"]
    for name, s in rows.items():
        ex = s.get("exits", {})
        en = s.get("exit_net", {})
        pe = s.get("pool_exits", {})
        L.append(f"| {name} | {s['n_events']} | {_dist(s['skips'], SKIP_ORDER)} | {_dist(ex, EXIT_ORDER)} | "
                 + " · ".join(f"{k} {_pct(en[k])}" for k in EXIT_ORDER if k in en) + " | "
                 + (f"{s['pool_n']:,} · {pe.get('stop', 0):.1%}" if "pool_n" in s else "—") + " |")
    L.append("")

    L += ["## 기술 — 같은 진입에 '손절 없음 + 120일' (공통 사건 키)", "",
          "| 진입 | 청산 | n (공통 / 전체) | 순평균 | 순중앙값 | 승률 | 하위10% 평균 | 10% 분위 | 최악 | 상위10% 평균 | "
          "상위1% 제거 평균 | 최장 연패 | 보유 중앙값 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, cmp_ in nostop.items():
        for tag, lab, n_all in (("a", "미너비니 프로토콜", cmp_["n_a"]), ("b", "손절 없음 + 120일", cmp_["n_b"])):
            r = cmp_[tag]
            L.append(f"| {name} | {lab} | {cmp_['n_common']} / {n_all} | {_pct(r['mean'])} | {_pct(r['median'])} | "
                     f"{r['win']:.1%} | {_pct(r['worst_decile'])} | {_pct(r['p10'])} | {_pct(r['min'])} | "
                     f"{_pct(r['best_decile'])} | {_pct(r['drop_top1'])} | {r['streak']} | {r['hold_median']:.0f} |")
    L.append("")

    for name in (PRIMARY, "VCP 끔"):
        yrs = years.get(name)
        if not yrs:
            continue
        L += [f"## 연도별 — {name}", "", "| 사건 연도 | n | 순평균 | 모멘텀 α [CI] |", "|---|---|---|---|"]
        for y, v in yrs.items():
            L.append(f"| {y} | {v['n']} | {_pct(v['net_mean'])} | {_a(v, 'mom_')} |")
        L.append("")
    L += ["## 칸 config", "", "| 칸 | 바꾼 값 |", "|---|---|"]
    L += [f"| {n} | {CELLS[n] or '— (사전등록 프로토콜)'} |" for n in cfgs]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
