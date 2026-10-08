#!/usr/bin/env python
"""SEPA 계열 조건 탐색 격자 126칸 — **탐색이지 판정이 아니다**.

격자 정본: ``research/logs/sepa_explore/EXPLORE.md`` (실행 전 고정, 커밋 e399c7a). 칸을 더하거나 빼지 않고, 정의를
결과를 보고 바꾸지 않는다. 이 러너는 리포터다 — 칸마다 숫자를 적고 격자 전체의 과최적(PBO·전반→후반 순위·DSR·
축별 주효과)을 잴 뿐, 통과·기각도 "최고 전략" 추천도 하지 않는다. 격자에서 나온 후보는 새 사전등록 + 전진 페이퍼로만
확인한다(EXPLORE.md).

공통 기계는 스윙 3호 러너(``sepa_confluence_gate``)를 그대로 import 한다 — 패널·유니버스·피처(``Data``,
``attach_features``), 실적 문(``e_pass``), 체결·비용(``simulate``), 정합 대조(``Pools``·``matched``·``sub_alpha``),
지수 헤지(``index_hedge``), 상위 1% 제거(``_drop_top``). 체결 규칙 자체는 ``research/signals/sepa.py``.

칸마다 ``prop_gate`` 를 돌리지 않는다(탐색 — 파일 이름도 ``*_gate.py`` 가 아니다). 대신 126칸 config 를 전부
원장 ``research/logs/sepa_explore/TRIALS.jsonl`` 에 먼저 적고, 최고 칸 DSR 의 N 은 원장에서 읽는다
(이 원장 + 스윙 3호 원장).

Run:  uv run --extra pg python research/experiments/sepa_explore.py
디버그(구간 축소·원장 분리 — 탐색 원장을 더럽히지 않는다):
      ... sepa_explore.py --event-lo 2020-01-01 --event-hi 2020-06-30 --out <scratch>/sepa_explore_debug
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

from krx_quant_core.stats.metrics import spearman  # noqa: E402
from krx_quant_core.stats.sharpe import deflated_sharpe, deflated_sharpe_from_sample  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402
from sepa_confluence_gate import (  # noqa: E402
    PREREG,
    Data,
    Pools,
    _ci,
    _drop_top,
    _key,
    _log,
    _pct,
    attach_features,
    e_pass,
    index_hedge,
    matched,
    simulate,
    sub_alpha,
)

from research.signals.sepa import build_skeleton_events  # noqa: E402
from swing_it.diagnostics.pbo import pbo_cscv  # noqa: E402
from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: E402
from swing_it.features.earnings_leg import pead_cross_events  # noqa: E402
from swing_it.storage import connect, db_default  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "sepa_explore"
CONFLUENCE_LABEL = "sepa_confluence"          # DSR 의 N 에 더하는 스윙 3호 원장
EVENT_LO, EVENT_HI = "2017-09-15", "2026-04-01"   # 모든 칸 공통(120일 보유가 데이터 끝 안에 닫힌다)
HALVES = (("전반 2017-09~2021", "2017-09-15", "2022-01-01"), ("후반 2022~2026-04", "2022-01-01", "2026-04-02"))
HEDGE_LO = "2024-01-26"
PBO_BLOCKS = 16
PBO_STD_FLOOR = 1e-6
TOP_K = 10

# --- 격자 축 (EXPLORE.md 그대로 — 순서가 표의 정렬 순서다) ---
ENTRIES = {
    "breakout": "뼈대 S(추세 템플릿 ∧ RS≥70 ∧ 피벗 돌파 ∧ 거래량 1.5×)",
    "trend": "추세 템플릿 ∧ RS≥70(돌파·거래량 없음)",
}
PEAD_NEW = "pead_new"
GATES: dict[str, dict] = {
    "off": {"e_mode": "off", "pead_rank_min": None},
    "E": {"e_mode": "e_flag", "e_min_yoy": 0.25, "e_max_age_days": 140, "pead_rank_min": None},
    "pead≥0.70": {"e_mode": "pead_rank", "pead_rank_min": 0.70},
    "pead≥0.90": {"e_mode": "pead_rank", "pead_rank_min": 0.90},
}
EXITS: dict[str, dict] = {
    "① 손절8%+MA50": {"stop": 0.08, "max_hold": 60, "ma50_exit": True},
    "② 손절8%+20일": {"stop": 0.08, "max_hold": 20, "ma50_exit": False},
    "③ 손절8%+60일": {"stop": 0.08, "max_hold": 60, "ma50_exit": False},
    "④ 손절15%+60일": {"stop": 0.15, "max_hold": 60, "ma50_exit": False},
    "⑤ 손절없음+20일": {"stop": 1.0, "max_hold": 20, "ma50_exit": False},
    "⑥ 손절없음+60일": {"stop": 1.0, "max_hold": 60, "ma50_exit": False},
    "⑦ 손절없음+120일": {"stop": 1.0, "max_hold": 120, "ma50_exit": False},
}
REGIMES = {"breadth>0.5": 0.5, "끔": -1.0}     # breadth ∈ [0,1] 이라 −1 은 항상 통과(스윙 3호 "breadth 끔" 과 같다)
PEAD_NEW_MIN = 0.90


def _base(**kw) -> dict:
    """스윙 3호 판정 config(유니버스·뼈대 수치·체결·비용·정합 대조 정의) 위에 탐색 공통값을 덮는다."""
    c = dict(PREREG)
    c.update({"event_lo": EVENT_LO, "event_hi": EVENT_HI, "rs_min": 70.0,
              "grid": "research/logs/sepa_explore/EXPLORE.md (e399c7a)"})
    c.update(kw)
    return c


def build_cells() -> list[dict]:
    """126칸 config 목록(축 순서). 각 칸은 ``entry/gate/exit/regime`` 이름과 시뮬·실적 문에 필요한 값을 다 담는다."""
    cells = []
    for entry in ENTRIES:
        for gate, g in GATES.items():
            for ex, x in EXITS.items():
                for reg, bmin in REGIMES.items():
                    cells.append(_base(entry=entry, entry_def=ENTRIES[entry], require_breakout=(entry == "breakout"),
                                       gate=gate, **g, exit=ex, **x, regime=reg, breadth_min=bmin))
    for ex, x in EXITS.items():
        for reg, bmin in REGIMES.items():
            cells.append(_base(entry=PEAD_NEW, require_breakout=None, rs_min=None,
                               entry_def=f"유니버스 종목의 PEAD 순위(유니버스 안)가 {PEAD_NEW_MIN:.2f} 이상으로 올라선 날"
                                         "(기술 조건 없음, earnings_leg.pead_cross_events)",
                               pead_cross_min=PEAD_NEW_MIN, gate="—", e_mode="off", pead_rank_min=None,
                               exit=ex, **x, regime=reg, breadth_min=bmin))
    return cells


def cell_id(c: dict) -> str:
    return f"{c['entry']} / {c['gate']} / {c['exit']} / {c['regime']}"   # 표 안에 들어가므로 | 를 쓰지 않는다


# ---------------------------------------------------------------------------
# 사건
# ---------------------------------------------------------------------------

def pead_new_events(D: Data, lo: str, hi: str) -> pd.DataFrame:
    """기준선 pead_new 사건 — PEAD 순위(유니버스 안 순위, ``Data.earn``)가 0.90 이상으로 올라선 날, 사건 구간 안.

    순위는 유니버스 밖에서 NaN 이라 사건은 자동으로 유니버스 종목이다(그래도 유니버스를 다시 확인한다).
    피벗 = 직전 50일 최고 고가(대조와 같은 식) — 추격 금지 규칙이 다른 칸과 같이 걸린다.
    """
    p = D.p
    cr = pead_cross_events(D.earn[["code", "date", "pead_rank"]], threshold=PEAD_NEW_MIN)
    cr = cr[(cr["date"] >= lo) & (cr["date"] <= hi)]
    di = p.days.get_indexer(pd.to_datetime(cr["date"]))
    ci = p.close.columns.get_indexer(cr["code"])
    ok = (di >= 0) & (ci >= 0)
    di, ci = di[ok], ci[ok]
    ev = pd.DataFrame({"code": p.close.columns[ci], "event_day": p.days[di], "day_idx": di, "col_idx": ci})
    ev["market"] = p.market.reindex(ev["code"]).to_numpy()
    ev["pivot"] = D.feats["pivot"].to_numpy()[di, ci]
    ev["close"] = p.close.to_numpy()[di, ci]
    ev["breadth"] = D.feats["breadth"].to_numpy()[di, ci]
    ev["universe"] = D.uni.to_numpy()[di, ci].astype(bool)
    ev = ev[ev["universe"]]
    return ev.sort_values(["event_day", "code"]).reset_index(drop=True)


def regime_mask(ev: pd.DataFrame, bmin: float) -> np.ndarray:
    """뼈대와 같은 식 ``(breadth > breadth_min)``, NaN 은 거짓."""
    return (ev["breadth"] > bmin).fillna(False).to_numpy(bool)


# ---------------------------------------------------------------------------
# 칸 숫자
# ---------------------------------------------------------------------------

SIM_COLS = ["code", "event_day", "market", "pivot", "q_adv", "q_ret5"]


def _alpha_fields(prefix: str, a) -> dict:
    if a is None or not np.isfinite(a.alpha):
        return {f"{prefix}alpha": np.nan, f"{prefix}lo": np.nan, f"{prefix}hi": np.nan, f"{prefix}n": 0}
    return {f"{prefix}alpha": a.alpha, f"{prefix}lo": a.ci_low, f"{prefix}hi": a.ci_high, f"{prefix}n": a.n_trades}


def cell_stats(t: pd.DataFrame, m: "dict | None", hedged: "pd.Series | None") -> dict:
    """EXPLORE.md "칸마다 적는 숫자" + 보조(청산 사유·보유·건당 Sharpe)."""
    if t is None or t.empty:
        return {"n": 0}
    net = t["net"].to_numpy(float)
    out = {"n": len(t), "net_mean": float(net.mean()), "net_median": float(np.median(net)),
           "win": float(np.mean(net > 0)), "gross_mean": float(t["ret"].mean()),
           "cost_bp": float(t["cost"].mean() * 1e4),
           "drop_top1_mean": _drop_top(net), "drop_top1_k": max(1, int(np.ceil(0.01 * len(net)))),
           "sharpe_trade": float(net.mean() / net.std(ddof=1)) if len(net) > 1 and net.std(ddof=1) > 0 else np.nan,
           "hold_median": float(t["hold_days"].median()),
           "entry_first": t["entry_day"].min().date().isoformat(), "entry_last": t["entry_day"].max().date().isoformat()}
    for r in ("stop", "ma50", "time", "delisted", "data_end"):
        out[f"exit_{r}"] = int((t["exit_reason"] == r).sum())
    yr = t.groupby(t["event_day"].dt.year)["net"].mean()
    out["years_pos"], out["years_n"] = int((yr > 0).sum()), int(len(yr))
    ma = m["ma"] if m is not None else None
    out.update(_alpha_fields("", ma))
    out["p_greater"] = float(ma.p_greater) if ma is not None and np.isfinite(ma.alpha) else np.nan
    out["n_unmatched"] = int(ma.n_unmatched) if ma is not None else 0
    out["excess_mean"] = float(t["excess"].mean()) if "excess" in t else np.nan
    for (_, lo, hi), pre in zip(HALVES, ("h1_", "h2_"), strict=True):
        mk = (t["event_day"] >= pd.Timestamp(lo)) & (t["event_day"] < pd.Timestamp(hi))
        out.update(_alpha_fields(pre, sub_alpha(m, mk) if (m is not None and mk.any()) else None))
    if hedged is not None:
        ok = hedged.notna()
        out["hedge_n"] = int(ok.sum())
        out["hedge_mean"] = float(hedged[ok].mean()) if ok.any() else np.nan
        out["hedge_net_mean"] = float((hedged[ok] - t.loc[ok, "cost"]).mean()) if ok.any() else np.nan
    return out


def monthly_excess(t: pd.DataFrame, months: pd.PeriodIndex) -> pd.Series:
    """PBO 행렬의 한 열 — 진입월별 정합 초과 평균(짝 있는 트레이드만), 거래 없는 달 0."""
    if t is None or t.empty or "excess" not in t:
        return pd.Series(0.0, index=months)
    x = t[["entry_day", "excess"]].dropna()
    s = x.groupby(x["entry_day"].dt.to_period("M"))["excess"].mean()
    return s.reindex(months).fillna(0.0)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _a(r, pre: str = "") -> str:
    a = r.get(f"{pre}alpha", np.nan)
    if not np.isfinite(a):
        return "nan"
    return f"{_pct(a)} {_ci(r.get(f'{pre}lo'), r.get(f'{pre}hi'))}"


def _row_md(r) -> str:
    if r["n"] == 0:
        return f"| {r['entry']} | {r['gate']} | {r['exit']} | {r['regime']} | 0 | | | | | | | | | |"
    hedge = (f"{_pct(r['hedge_net_mean'])} ({int(r['hedge_n'])})" if r.get("hedge_n", 0) else "—")
    return (f"| {r['entry']} | {r['gate']} | {r['exit']} | {r['regime']} | {int(r['n'])} | {_pct(r['net_mean'])} | "
            f"{_pct(r['net_median'])} | {r['win']:.1%} | {_a(r)} | {_pct(r['drop_top1_mean'])} | "
            f"{int(r['years_pos'])}/{int(r['years_n'])} | {_pct(r['h1_alpha'])} | {_pct(r['h2_alpha'])} | {hedge} |")


MAIN_HEAD = ["| 진입 | 실적 문 | 청산 | 국면 | n | 순평균 | 순중앙값 | 승률 | 정합 alpha [날짜군집 CI] | 상위1% 제거 | "
             "연도 양수 | 전반 α | 후반 α | 헤지 순평균 (n) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def axis_effects(df: pd.DataFrame) -> list[str]:
    """축 값마다 그 값을 가진 칸들의 정합 alpha 중앙값·IQR (칸 수 n). 실적 문 축은 pead_new 를 뺀 112칸."""
    L = ["| 축 | 값 | 칸 수 | 정합 alpha 중앙값 | IQR [25%, 75%] | alpha > 0 칸 | 전반 α 중앙값 | 후반 α 중앙값 |",
         "|---|---|---|---|---|---|---|---|"]
    order = {"entry": [*ENTRIES, PEAD_NEW], "gate": list(GATES), "exit": list(EXITS), "regime": list(REGIMES)}
    names = {"entry": "진입", "gate": "실적 문(112칸)", "exit": "청산", "regime": "국면"}
    for ax, vals in order.items():
        sub = df if ax != "gate" else df[df["entry"] != PEAD_NEW]
        for v in vals:
            g = sub.loc[sub[ax] == v, ["alpha", "h1_alpha", "h2_alpha"]].astype(float)
            a = g["alpha"].dropna()
            if a.empty:
                L.append(f"| {names[ax]} | {v} | 0 | | | | | |")
                continue
            L.append(f"| {names[ax]} | {v} | {len(a)} | {_pct(a.median())} | [{_pct(a.quantile(0.25))}, "
                     f"{_pct(a.quantile(0.75))}] | {int((a > 0).sum())}/{len(a)} | {_pct(g['h1_alpha'].median())} | "
                     f"{_pct(g['h2_alpha'].median())} |")
    return L


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(원장은 --out 으로 분리)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR, help="RESULTS.md·cells.csv·원장 디렉터리(기본 = 탐색 디렉터리)")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.event_lo or args.event_hi) and not debug:
        ap.error("구간 축소는 --out 을 탐색 디렉터리 밖으로 둘 때만(탐색 원장 보호)")
    os.makedirs(out_dir, exist_ok=True)
    led_label, led_dir = os.path.basename(os.path.normpath(out_dir)), os.path.dirname(os.path.normpath(out_dir))
    lo, hi = args.event_lo or EVENT_LO, args.event_hi or EVENT_HI

    cells = build_cells()
    if debug and (args.event_lo or args.event_hi):
        cells = [{**c, "event_lo": lo, "event_hi": hi} for c in cells]
    assert len(cells) == 126 and len({cell_id(c) for c in cells}) == 126

    # index_hedge 의 pd.read_sql_query(psycopg 연결) 경고가 칸마다 찍힌다 — 동작과 무관해 끈다.
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()
    load_env_db()
    con = connect(db_default())
    D = Data(con, t0)
    pools = Pools(D)
    p = D.p

    # --- 원장: 126칸 config 를 결과 전에 전부 적는다 ---
    for c in cells:
        record_trial(led_label, c, logs_dir=led_dir, note=f"탐색 칸: {cell_id(c)}")
    n_led = count_trials(led_label, logs_dir=led_dir)
    n_conf = count_trials(CONFLUENCE_LABEL, logs_dir="research/logs")
    _log(f"원장 {led_label}: {n_led}칸 · {CONFLUENCE_LABEL}: {n_conf}", t0)

    # --- 사건 표: 진입별 master(국면 끔 = breadth −1) + 피처, 국면은 build_skeleton_events 키로 고른다 ---
    masters: dict[str, pd.DataFrame] = {}
    for entry in ENTRIES:
        mcfg = _base(require_breakout=(entry == "breakout"), breadth_min=REGIMES["끔"], event_lo=lo, event_hi=hi)
        ev = build_skeleton_events(p, mcfg, universe=D.uni, feats=D.feats)
        ev = attach_features(D, ev)
        ev.index = _key(ev["event_day"], ev["code"])
        masters[entry] = ev
        _log(f"사건 master {entry}: {len(ev):,}", t0)
    pn = attach_features(D, pead_new_events(D, lo, hi))
    pn.index = _key(pn["event_day"], pn["code"])
    masters[PEAD_NEW] = pn
    _log(f"사건 master {PEAD_NEW}: {len(pn):,}", t0)

    def events_for(c: dict) -> pd.DataFrame:
        m = masters[c["entry"]]
        if c["entry"] == PEAD_NEW:
            ev = m[regime_mask(m, c["breadth_min"])]
        else:
            s = build_skeleton_events(p, c, universe=D.uni, feats=D.feats)    # 국면·RS·구간은 cfg 그대로
            ev = m.loc[m.index.intersection(_key(s["event_day"], s["code"]))]
            ev = ev[e_pass(ev, c).to_numpy()]
        return ev.sort_values(["event_day", "code"]).reset_index(drop=True)

    months = pd.period_range(pd.Timestamp(lo).to_period("M"), pd.Timestamp(hi).to_period("M"), freq="M")
    rows, monthly, nets = [], {}, {}
    ev_cache: dict[tuple, pd.DataFrame] = {}
    for i, c in enumerate(cells, 1):
        k = (c["entry"], c["gate"], c["regime"])
        if k not in ev_cache:
            ev_cache[k] = events_for(c)
        ev = ev_cache[k]
        t = simulate(D, ev[SIM_COLS], c) if len(ev) else pd.DataFrame()
        m = hedged = None
        if not t.empty:
            m = matched(pools, t, c)
            t["excess"] = m["excess"]
            hedged = index_hedge(con, D, t)
        st = cell_stats(t, m, hedged)
        cid = cell_id(c)
        rows.append({"cell": cid, "entry": c["entry"], "gate": c["gate"], "exit": c["exit"], "regime": c["regime"],
                     "n_events": len(ev), **st})
        monthly[cid] = monthly_excess(t, months)
        nets[cid] = t["net"].to_numpy(float) if not t.empty else np.empty(0)
        _log(f"[{i:3d}/126] {cid}: 사건 {len(ev):,} · n {st['n']:,} · α {_pct(st.get('alpha', np.nan))}", t0)

    df = pd.DataFrame(rows)
    M = pd.DataFrame(monthly)                    # 월 × 칸
    M.index = M.index.astype(str)

    # --- 과최적 1: PBO(CSCV) ---
    n_blocks = PBO_BLOCKS if len(M) >= PBO_BLOCKS else max(2, len(M) // 2 * 2)   # 줄이는 건 디버그 구간뿐
    pbo = {mt: pbo_cscv(M.to_numpy(), n_blocks=n_blocks, metric=mt, std_floor=PBO_STD_FLOOR)
           for mt in ("sharpe", "mean")}
    df["pbo_is_best_freq_sharpe"] = pbo["sharpe"]["best_freq"]
    df["pbo_is_best_freq_mean"] = pbo["mean"]["best_freq"]
    _log("PBO", t0)

    # --- 과최적 2: 전반 → 후반 순위 ---
    both = df[np.isfinite(df["h1_alpha"].astype(float)) & np.isfinite(df["h2_alpha"].astype(float))].copy()
    rho = spearman(both["h1_alpha"].astype(float), both["h2_alpha"].astype(float)) if len(both) > 2 else np.nan
    both["h2_rank"] = both["h2_alpha"].rank(ascending=False, method="min")    # 1 = 후반 최고
    h1_order = both.sort_values("h1_alpha", ascending=False)
    h1_best = h1_order.iloc[0] if len(both) else None
    top_h1 = h1_order.head(TOP_K)
    df = df.merge(both[["cell", "h2_rank"]], on="cell", how="left")
    df["h1_rank"] = df["h1_alpha"].rank(ascending=False, method="min")
    df["alpha_rank"] = df["alpha"].rank(ascending=False, method="min")

    # --- 과최적 3: 최고 칸(전 표본 정합 alpha)의 DSR ---
    n_dsr = n_led + n_conf
    best = df.loc[df["alpha"].astype(float).idxmax()]
    dsr = deflated_sharpe_from_sample(nets[best["cell"]], n_dsr)
    sr_cells = df["sharpe_trade"].astype(float)
    sr_std_x = float(sr_cells[np.isfinite(sr_cells)].std(ddof=1))
    R = nets[best["cell"]]
    R = R[np.isfinite(R)]
    mu, sd = R.mean(), R.std()
    dsr_x = deflated_sharpe(float(R.mean() / R.std(ddof=1)), n_dsr, len(R), float(((R - mu) ** 3).mean() / sd ** 3),
                            float(((R - mu) ** 4).mean() / sd ** 4), sr_std=sr_std_x)

    # --- 리포트 ---
    elapsed = time.time() - t0
    L = [f"# SEPA 계열 조건 탐색 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 탐색 — 판정 아님)", "",
         "격자는 `EXPLORE.md`(e399c7a, 실행 전 고정) 그대로 126칸이다. **어떤 칸도 전략이 아니다** — 2017~2026 은 스윙 3호에서 "
         "이미 봤고, 격자 최고 칸은 정의상 운으로 부풀려져 있다. 여기 숫자는 축별 지형과 과최적 측정용이다. 후보는 새 사전등록 + "
         "전진 페이퍼로만 확인한다.", ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 구간 {lo}~{hi}, 원장 `{out_dir}`. 탐색 결과로 쓰지 않는다.", ""]
    L += ["## 표본", "",
          f"- 사건 구간 {lo} ~ {hi}(모든 칸) · 가격 패널 {p.days[0].date()}~{p.days[-1].date()} · 유니버스 종목-일 "
          f"{int(D.uni.to_numpy().sum()):,}",
          "- 사건 master(국면 끔, 실적 문 전): " + " · ".join(f"{k} {len(v):,}" for k, v in masters.items()),
          f"- 칸별 트레이드 n: 중앙값 {int(df['n'].median()):,} · 최소 {int(df['n'].min()):,} · 최대 {int(df['n'].max()):,} "
          f"· 합 {int(df['n'].sum()):,}",
          f"- 원장 `{led_dir}/{led_label}/TRIALS.jsonl` 칸 수 = **{n_led}** · DSR N = {n_led} + 스윙 3호 원장 {n_conf} = {n_dsr}",
          ""]
    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          "- **trend 진입:** `sepa.build_skeleton_events(cfg require_breakout=False)` — 유니버스 ∧ 추세 템플릿 ∧ RS≥70 ∧ (국면). "
          "추격 금지의 피벗은 그대로 직전 50일 최고 고가라 시가가 그 위 5% 를 넘으면 버린다. 같은 종목 보유 중 재진입은 없고, "
          "청산 다음 거래일부터 다시 조건이 맞으면 다시 들어간다(의도된 동작).",
          f"- **pead_new:** 유니버스 안 PEAD 순위(`Data.earn.pead_rank`, 스윙 3호와 같은 순위)가 직전 거래일 < {PEAD_NEW_MIN:.2f}"
          f"(또는 NaN = 유니버스 밖·실적 모름)에서 ≥ {PEAD_NEW_MIN:.2f} 로 올라선 날마다 사건(`earnings_leg.pead_cross_events`). "
          "기술 조건 없음, 실적 문 축 없음. 국면은 그 종목 시장 breadth > 0.5. 추격 금지·상한가·정지 규칙은 다른 칸과 같다"
          "(피벗 = 직전 50일 최고 고가 — 대조와 같은 식).",
          "- **손절 없음** = 손절폭 1.0(가격 0 이어야 닿는다). MA50 이탈은 ① 에서만. ① 의 최대 보유는 원전 60일.",
          "- **국면 끔** = breadth_min −1(스윙 3호 'breadth 끔' 칸과 같은 식 — breadth NaN 인 날은 두 국면 모두에서 빠진다).",
          "- **정합 대조:** 스윙 3호 그대로(같은 날 × ADV20 4분위 × 직전 5일 수익 4분위 × 시장, 유니버스 풀, 자기 종목 제외, "
          "n_per 5, seed 0, **같은 진입·청산 규칙**). 대조 풀은 청산 규칙 7벌마다 따로 캐시한다. alpha 는 비용 전 총수익 기준.",
          "- **전반/후반:** 사건일 2017-09-15~2021-12-31 / 2022-01-01~2026-04-01, 각 반의 alpha 는 이미 뽑힌 대조 짝 그대로"
          "(`sub_alpha`).",
          f"- **지수 헤지:** 진입 ≥ {HEDGE_LO} 트레이드의 (총수익 − 같은 구간 KOSPI/KOSDAQ 지수 수익) − 비용 평균(표의 '헤지 순평균').",
          f"- **PBO:** 월 × 칸 행렬(진입월별 정합 초과 평균, 짝 없는 트레이드 제외, 거래 없는 달 0, {len(M)}개월), 연속 "
          f"{n_blocks} 조각, C({n_blocks},{n_blocks // 2}) = {pbo['sharpe']['n_combos']:,} 조합 전부. 조각-집합 성과는 "
          f"(가) 월 평균 ÷ 월 표준편차(ddof 1, 하한 {PBO_STD_FLOOR:g}) — 주(主), (나) 월 평균. λ ≤ 0 비율 = PBO.",
          "- **DSR:** 최고 칸 = 전 표본 정합 alpha 최대 칸, 건당 비용 차감 순수익. (가) 코어 기본(시행별 Sharpe 표준편차 ≈ 1/√(T−1)), "
          f"(나) 시행별 Sharpe 표준편차를 이 격자 126칸 건당 Sharpe 의 표준편차({sr_std_x:.3f})로 둔 것.",
          ""]

    L += ["## 축별 주효과 (칸들의 정합 alpha 분포 — 한 칸이 아니라 축의 방향)", ""]
    L += axis_effects(df)
    L += ["", "실적 문 축은 pead_new(실적 문 없음) 14칸을 뺀 112칸. 진입·청산·국면 축은 126칸 전부.", ""]

    ps, pm = pbo["sharpe"], pbo["mean"]
    L += ["## 과최적 측정 (격자 전체)", "",
          "### 1. PBO (CSCV)", "",
          "| 조각 성과 | PBO | λ 중앙값 | IS 최고 칸의 OOS 순위 중앙값 (1 = 최하, N = 126) | IS 최고로 가장 자주 뽑힌 칸 (비율) |",
          "|---|---|---|---|---|"]
    for nm, r in (("(가) 평균/표준편차", ps), ("(나) 평균", pm)):
        j = int(np.argmax(r["best_freq"]))
        L.append(f"| {nm} | **{r['pbo']:.3f}** | {np.median(r['logits']):+.3f} | {np.median(r['oos_rank']):.1f} | "
                 f"{df['cell'].iloc[j]} ({r['best_freq'][j]:.1%}) |")
    L += ["", f"IS 최고로 한 번이라도 뽑힌 칸 수: (가) {int((ps['best_freq'] > 0).sum())} · (나) {int((pm['best_freq'] > 0).sum())}. "
          "IS 최고 칸의 IS 성과 대비 OOS 성과 — (가) IS 중앙값 "
          f"{np.median(ps['is_perf']):+.3f} → OOS {np.median(ps['oos_perf']):+.3f} · (나) IS 중앙값 {_pct(np.median(pm['is_perf']))} "
          f"→ OOS {_pct(np.median(pm['oos_perf']))}.", ""]

    L += ["### 2. 전반 → 후반 순위", "",
          f"- 두 반 모두 alpha 가 있는 칸 {len(both)} · 스피어만 ρ(전반 alpha, 후반 alpha) = **{rho:+.3f}**"]
    if h1_best is not None:
        L += [f"- 전반 최고 칸 `{h1_best['cell']}`: 전반 {_pct(h1_best['h1_alpha'])} → 후반 {_pct(h1_best['h2_alpha'])}, "
              f"후반 순위 **{int(h1_best['h2_rank'])} / {len(both)}** (1 = 최고)",
              f"- 전반 상위 {TOP_K}칸의 후반 alpha 평균 {_pct(top_h1['h2_alpha'].mean())} · 중앙값 "
              f"{_pct(top_h1['h2_alpha'].median())} — 전체 칸 후반 평균 {_pct(both['h2_alpha'].mean())} · 중앙값 "
              f"{_pct(both['h2_alpha'].median())}",
              f"- 전반 상위 {TOP_K}칸의 후반 순위: " + ", ".join(str(int(x)) for x in top_h1["h2_rank"])]
    L.append("")

    L += ["### 3. 최고 칸의 Deflated Sharpe", "",
          f"최고 칸(전 표본 정합 alpha) `{best['cell']}`: n {int(best['n'])} · 정합 alpha {_a(best)} · 순평균 "
          f"{_pct(best['net_mean'])} · 건당 Sharpe {dsr['observed_sharpe']:+.4f}", "",
          "| 시행별 Sharpe 표준편차 | N | E[max SR | H0] | deflated SR | P(SR>0) | P(SR>SR0) |", "|---|---|---|---|---|---|"]
    for nm, d in (("(가) 1/√(T−1) (코어 기본)", dsr), (f"(나) 격자 칸 간 {sr_std_x:.3f}", dsr_x)):
        L.append(f"| {nm} | {d['n_trials']} | {d['expected_max_sharpe_h0']:.4f} | {d['deflated_sharpe']:+.4f} | "
                 f"{d['prob_sharpe_gt0']:.1%} | {d['prob_deflated_sharpe']:.1%} |")
    L.append("")

    L += ["## 전체 126칸 (축 순서 — 성과순 아님)", "",
          "순평균·중앙값·승률·상위1% 제거·연도 양수는 비용(`korea.cost_rt`) 차감 순수익, 정합 alpha 와 전반/후반 α 는 비용 전. "
          "연도 양수 = 사건 연도별 순평균 > 0 인 연도 수 / 사건이 있는 연도 수.", ""]
    L += MAIN_HEAD
    L += [_row_md(r) for r in df.to_dict("records")]
    L.append("")

    top = df.sort_values("alpha", ascending=False).head(TOP_K)
    L += [f"## 정합 alpha 상위 {TOP_K}칸 (선택편향 경고와 함께)", "",
          f"> **경고:** 126칸 중 사후에 고른 상위다. 위 PBO {ps['pbo']:.2f}(평균/표준편차) · 전반→후반 ρ {rho:+.2f} 와 함께 읽는다. "
          "이 표는 추천이 아니다 — 여기 오른 칸은 새 사전등록 + 전진 페이퍼 없이는 아무것도 확인하지 않는다.", ""]
    L += ["| 순위 | " + MAIN_HEAD[0][2:] + " 전반 순위 | 후반 순위 |",
          "|---|" + MAIN_HEAD[1][1:] + "---|---|"]
    for rk, r in enumerate(top.to_dict("records"), 1):
        h1r = "—" if not np.isfinite(r.get("h1_rank", np.nan)) else str(int(r["h1_rank"]))
        h2r = "—" if not np.isfinite(r.get("h2_rank", np.nan)) else str(int(r["h2_rank"]))
        L.append(f"| {rk} " + _row_md(r) + f" {h1r} | {h2r} |")
    L.append("")

    L += ["## 기준선 pead_new 14칸 (기술 조건 없음)", ""]
    L += MAIN_HEAD
    L += [_row_md(r) for r in df[df["entry"] == PEAD_NEW].to_dict("records")]
    L.append("")

    L += ["## 실행 메타", "",
          f"- 런타임 {elapsed / 60:.1f}분 · 칸 {len(df)} · 대조 풀 규칙 {len(pools.cache)}벌 "
          f"({', '.join(f'{k}: {len(v):,}' for k, v in pools.cache.items())})",
          f"- 산출물: `{out_dir}/cells.csv`(칸별 전 숫자), `{out_dir}/monthly_excess.csv`(PBO 행렬, 월 × 칸), "
          f"`{out_dir}/TRIALS.jsonl`(원장)",
          "- 재현: `uv run --extra pg python research/experiments/sepa_explore.py`", ""]

    df.to_csv(os.path.join(out_dir, "cells.csv"), index=False)
    M.to_csv(os.path.join(out_dir, "monthly_excess.csv"))
    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
