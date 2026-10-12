#!/usr/bin/env python
"""스윙 라이브 포트폴리오 백테스트 러너 — ``swing_it.live`` (라이브와 같은 전략 코드)를 실데이터에 돌린다.

설계: ``docs/superpowers/specs/2026-10-10-swing-live.md`` (112a0a6). **리포터다 — 판정하지 않는다.** 표본 내다
(ADV ≥ 200억 구간은 이미 본 데이터, 50~200억 구간은 ``CONVERGENCE.md`` 사전등록 시험이 같은 사건을 본다).

하는 일:

1. 피처 대조(``--skip-checks`` 가 아니면): 연구 경로(``sepa.load_panels`` · ``skeleton_features`` · ``korea.universe_mask`` ·
   ``flow_leg`` · ``earnings_leg``+``pead_rank`` · ``convergence_run.relative_contraction`` — CONVERGENCE A 와 같은 조립,
   유니버스만 ADV20 ≥ 5,000)의 사건 집합 vs 라이브 :func:`swing_it.live.strategy.scan` 사건 집합, 전 기간
   (NXT 보정 전 날짜) + 그날치 300행 창(:func:`swing_it.live.features.signal_panels_for`) 몇 날짜.
2. 실데이터 패리티: 최근 구간 사건을 ``run_backtest(unlimited=True)`` 와 ``simulate_minervini`` 에 같은 패널로 넣어 대조.
3. 포트폴리오 백테스트 두 변형(시작 1억, 비용 ``korea.cost_rt`` 매도 부분마다):
   (a) 2024-02-19 → 끝, 규칙 전부(F 포함) / (b) 2017-09-15 → 끝, F 없음(수급 F 는 2024-02-19 부터만 있다).
   비용 2배 민감도도 같이 적는다.
4. 지수 대비: ``sector_index`` '001'(코스피, 2024-01-26~). 그 전은 시총가중 코스피 대용(원자료 종가 × 상장주식수
   ``shares_outstanding_history`` as-of, 전일 시총 가중 × 분할조정 일수익). 겹치는 구간에서 대용과 실제 지수의 상관을 적는다.
5. ``research/logs/live_swing/BACKTEST.md`` + 트레이드·일자산 CSV + ``equity.png``.

Run:  export $(grep -v '^#' ~/git/quant-airflow/.env | xargs)
      uv run --extra pg --extra viz python research/experiments/live_portfolio_backtest.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from convergence_run import raw_close_panel, relative_contraction  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402

from research.signals.minervini_exits import simulate_minervini  # noqa: E402
from research.signals.sepa import build_skeleton_events, load_panels, skeleton_features  # noqa: E402
from swing_it.features.earnings_leg import earnings_leg_panel, pead_rank  # noqa: E402
from swing_it.features.flow_leg import flow_leg  # noqa: E402
from swing_it.features.korea import load_listing_meta, universe_mask  # noqa: E402
from swing_it.krx_session import NXT_START  # noqa: E402
from swing_it.live.backtest import default_cost, drawdown, perf_stats, run_backtest  # noqa: E402
from swing_it.live.config import DEFAULT_RISK, StrategyConfig  # noqa: E402
from swing_it.live.news_veto import veto_panel_from_ledger  # noqa: E402
from swing_it.live.features import (  # noqa: E402
    EARN_COLS,
    PANEL_START,
    SUPPLY_COLS,
    build_signal_panels,
    load_inputs,
    signal_panels_for,
)
from swing_it.live.strategy import scan  # noqa: E402
from swing_it.storage import connect, db_default, read_earnings, read_supply_demand  # noqa: E402

OUT_DIR = os.path.join(ROOT, "research", "logs", "live_swing")
KST = dt.timezone(dt.timedelta(hours=9))
FLOW_LO = "2024-02-19"
EVENT_LO = "2017-09-15"
INDEX_LO = "2024-01-26"
INITIAL = 100_000_000.0
SLIP_BPS = 50.0
RISK = DEFAULT_RISK                      # 손절·본전 미끄러짐(주 결과). 0bp 도 같이 적는다.
PARITY_LO = "2025-01-01"
VETO_LO = "2025-09-01"                      # 판정 원장이 있는 구간 — 뉴스 거부권 변형의 신호 하한
#: 그날치 창 대조 날짜 — 고정 2개 + 실행 때 라이브 사건이 있는 날 4개(변형 a 2 · b 2)를 더한다.
WINDOW_CHECK_DAYS = ("2021-06-15", "2026-09-11")
#: 청산 프로토콜 — new = 라이브 기본(설계 §6: 돌파 실패 켬, 시간 손절 끔) / old = 사전등록 원전(시간 손절 20일).
PROTOCOLS = {
    "new": {"label": "돌파 실패(종가<피벗→다음날 시가), 시간 손절 없음", "failed_breakout": True, "time_stop_days": None},
    "old": {"label": "옛 원전: 20일 시간 손절, 돌파 실패 없음", "failed_breakout": False, "time_stop_days": 20},
}
VARIANTS = {
    "a": {"label": "(a) 규칙 전부(F 포함)", "cfg": StrategyConfig(), "start": FLOW_LO},
    "b": {"label": "(b) F 없음(규칙 A − F)", "cfg": StrategyConfig(require_flow=False), "start": EVENT_LO},
}


_CON = {"con": None}


def db():
    """살아 있는 DB 연결. 긴 계산 뒤 서버가 끊은 연결(psycopg2 OperationalError)은 다시 연다."""
    con = _CON["con"]
    if con is not None:
        try:
            cur = con.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            return con
        except Exception:  # noqa: BLE001 — 끊긴 연결
            try:
                con.close()
            except Exception:  # noqa: BLE001
                pass
    _CON["con"] = connect(db_default())
    return _CON["con"]


def _scaled_cost(mult: float):
    def cost(day, market) -> float:
        return mult * default_cost(day, market)

    return cost


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _pct(x, nd=1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def _num(x, nd=2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# 연구 경로 사건(대조용)
# ---------------------------------------------------------------------------

def research_events(con, cfg: StrategyConfig, t0: float) -> "tuple[pd.DataFrame, object]":
    """CONVERGENCE A 조립(연구 함수)로 ADV20 ≥ 5,000 사건 표 — rel·F·pead_rank 열 포함. 보정 전 가격."""
    p = load_panels(con, start=PANEL_START)
    raw = raw_close_panel(con, p)
    meta = load_listing_meta(con)
    h5 = universe_mask(p.close, p.trade_value, meta, raw_close=raw, adv_floor=cfg.adv_floor)
    feats = skeleton_features(p)
    ms = build_skeleton_events(p, {"event_lo": EVENT_LO}, universe=h5, feats=feats)
    di, ci = ms["day_idx"].to_numpy(), ms["col_idx"].to_numpy()
    _log(f"연구 경로 뼈대 S {len(ms):,}", t0)
    sd = read_supply_demand(con, session="raw", cols=SUPPLY_COLS, start=PANEL_START)
    fl = flow_leg(sd, index=p.days, columns=p.close.columns, weight=p.trade_value)
    ms["F"] = fl["F"].to_numpy()[di, ci]
    del sd, fl
    ea = read_earnings(con, all_versions=True, cols=EARN_COLS)
    ep = earnings_leg_panel(ea, [d.strftime("%Y-%m-%d") for d in p.days])
    edi = p.days.get_indexer(pd.to_datetime(ep["date"]))
    eci = p.close.columns.get_indexer(ep["code"])
    ok = (edi >= 0) & (eci >= 0)
    elig = np.zeros(len(ep), dtype=bool)
    elig[ok] = h5.to_numpy()[edi[ok], eci[ok]]
    ep["pead_rank"] = pead_rank(ep, eligible=pd.Series(elig, index=ep.index))
    ms["_date"] = ms["event_day"].dt.strftime("%Y-%m-%d")
    ms = ms.merge(ep[["code", "date", "pead_rank"]].rename(columns={"date": "_date"}), on=["code", "_date"],
                  how="left").drop(columns="_date")
    ms["rel"] = relative_contraction(p.high, p.low)[1].to_numpy()[di, ci]
    _log("연구 경로 F·PEAD·rel", t0)
    return ms, p


def research_mask(ms: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    mk = (ms["pead_rank"] >= cfg.pead_rank_min).fillna(False).to_numpy(bool).copy()
    mk = mk & (ms["rel"] <= cfg.rel_max).fillna(False).to_numpy(bool)
    if cfg.require_flow:
        mk = mk & (ms["F"] == 1).fillna(False).to_numpy(bool)
    return mk


def live_events(panels: dict, cfg: StrategyConfig, lo: str, hi: "str | None" = None) -> pd.DataFrame:
    days = panels["close"].index
    days = days[(days >= pd.Timestamp(lo)) & ((days <= pd.Timestamp(hi)) if hi else True)]
    rows = [{"code": c.code, "event_day": c.signal_day, "market": c.market, "pivot": c.pivot,
             "contraction_low": c.contraction_low, "rel": c.rel, "rs": c.rs}
            for d in days for c in scan(panels, d, cfg)]
    return pd.DataFrame(rows, columns=["code", "event_day", "market", "pivot", "contraction_low", "rel", "rs"])


def _keys(df: pd.DataFrame) -> set:
    return set(zip(df["code"].astype(str), pd.to_datetime(df["event_day"])))


# ---------------------------------------------------------------------------
# 지수
# ---------------------------------------------------------------------------

def kospi_series(con, t0: float) -> "tuple[pd.Series, dict]":
    """코스피 레벨 = ``sector_index`` '001' 종가(2024-01-26~). **그 전엔 지수 비교를 하지 않는다**(REVIEW_BACKTEST M2 —
    원자료 종가 × as-of 주식수 시총가중 대용이 2019·2020·2022 를 6~9%p 과소했다; 원자료 종가의 조정 기준 혼재가 유력).
    (b) 의 지수 열은 2024-01-26 이후 겹치는 구간만 적는다."""
    idx = pd.read_sql_query("SELECT date, close FROM sector_index WHERE code = '001'", con)
    idx["date"] = pd.to_datetime(idx["date"])
    real = idx.set_index("date")["close"].astype(float).sort_index()
    info = {"index_lo": str(real.index[0].date()), "index_hi": str(real.index[-1].date()), "n": int(len(real))}
    _log(f"지수 001 {info['index_lo']}~{info['index_hi']} ({info['n']}일) — 그 전 대용 없음", t0)
    return real, info


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def plot_equity(results: dict, bench: pd.Series, path: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    c_strat, c_bench = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    fig.patch.set_facecolor(surface)
    c_old = "#1baf7a"
    for col, key in enumerate(VARIANTS):
        eq = results[(key, "new")]["base"].equity
        eq_old = results[(key, "old")]["base"].equity
        b = bench.reindex(eq.index)
        ax, axd = axes[0, col], axes[1, col]
        ax.plot(eq.index, eq / eq.iloc[0], color=c_strat, lw=2, label="전략 new(돌파 실패)")
        ax.plot(eq_old.index, eq_old / eq_old.iloc[0], color=c_old, lw=2, label="전략 old(시간 손절)")
        bb = b.dropna()
        if len(bb):
            ax.plot(bb.index, bb / bb.iloc[0] * (eq / eq.iloc[0]).loc[bb.index[0]], color=c_bench, lw=2,
                    label="코스피 001 (겹침 시점에 맞춤)")
        ax.set_title(VARIANTS[key]["label"] + " — 시작 = 1 (낙폭 = new)", color=ink, fontsize=11, loc="left")
        axd.fill_between(eq.index, drawdown(eq) * 100, 0, color=c_strat, alpha=0.35, lw=0)
        axd.set_ylabel("낙폭 %", color=ink2)
        for a in (ax, axd):
            a.set_facecolor(surface)
            a.grid(True, color=grid, lw=0.8)
            a.tick_params(colors=ink2)
            for s in a.spines.values():
                s.set_visible(False)
        ax.legend(frameon=False, labelcolor=ink)
    fig.suptitle("스윙 라이브 v0 포트폴리오 백테스트 — 표본 내, 판정 아님", color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)
    return True


def stats_rows(cols: list, bench_stats: dict) -> list[str]:
    """``cols`` = [(이름, stats)…] 전략 열들 + 코스피 열."""
    b = bench_stats or {}

    def row(label, f, bf=None):
        cells = [f(c) for _, c in cols]
        return f"| {label} | " + " | ".join(cells) + f" | {bf(b) if bf else ''} |"

    return [
        row("기간", lambda c: f"{c['start']} ~ {c['end']}"),
        row("최종 자산", lambda c: f"{c['final_equity']:,.0f} 원"),
        row("총수익", lambda c: _pct(c["total_return"]), lambda b: _pct(b.get("total_return"))),
        row("CAGR", lambda c: _pct(c["cagr"]), lambda b: _pct(b.get("cagr"))),
        row("연 변동성", lambda c: _pct(c["vol"]), lambda b: _pct(b.get("vol"))),
        row("Sharpe 일(√252, rf 0)", lambda c: _num(c["sharpe"]), lambda b: _num(b.get("sharpe"))),
        row("Sharpe 월(√12)", lambda c: _num(c["sharpe_monthly"]), lambda b: _num(b.get("sharpe_monthly"))),
        row("MDD", lambda c: f"{_pct(c['mdd'])} ({c.get('mdd_peak')}→{c.get('mdd_trough')})", lambda b: _pct(b.get("mdd"))),
        row("지수 상관 / 베타", lambda c: f"{_num(c.get('bench', {}).get('corr'))} / {_num(c.get('bench', {}).get('beta'))}"),
        row("평균 노출(보유/자산)", lambda c: _pct(c["exposure"], 0)),
        row("평균 보유 종목", lambda c: _num(c["avg_positions"], 1)),
        row("연 회전율", lambda c: f"{_num(c['turnover'], 1)}×"),
        row("트레이드", lambda c: str(c["n_trades"])),
        row("승률(순)", lambda c: _pct(c.get("win_rate"), 0)),
        row("평균 이익 / 손실(순)", lambda c: f"{_pct(c.get('avg_win'))} / {_pct(c.get('avg_loss'))} (비 {_num(c.get('gl_ratio'))})"),
        row("트레이드 평균 총/순", lambda c: f"{_pct(c.get('avg_gross'), 2)} / {_pct(c.get('avg_net'), 2)}"),
        row("평균 총(연구 정의 ret_half)", lambda c: _pct(c.get("avg_gross_half"), 2)),
        row("평균 보유일", lambda c: _num(c.get("avg_hold"), 1)),
        row("비용 / 미끄러짐 합계", lambda c: f"{c.get('cost_krw', 0):,.0f} / {c.get('slip_krw', 0):,.0f} 원"),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--skip-checks", action="store_true", help="피처 대조·패리티 생략(빠른 재실행)")
    ap.add_argument("--cache", default=None, help="원재료 pickle 경로(있으면 읽고, 없으면 만든 뒤 저장 — 디버그용)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()
    load_env_db()

    if args.cache and os.path.exists(args.cache):
        inp = pd.read_pickle(args.cache)
    else:
        inp = load_inputs(db(), start=PANEL_START, patch_nxt=True)
        if args.cache:
            pd.to_pickle(inp, args.cache)
    pinfo = inp.patch_info
    _log(f"원재료 {inp.close.shape} {inp.days[0].date()}~{inp.days[-1].date()} · NXT 행 {pinfo.get('nxt_rows', 0):,} "
         f"중 보정 {pinfo.get('nxt_patched', 0):,}", t0)
    base_cfg = StrategyConfig()
    panels = build_signal_panels(inp, base_cfg, verbose=True)
    _log("라이브 패널", t0)
    data_end = inp.days[-1]

    # --- NXT 보정 건수(유니버스 종목-일) ---
    nxt = {"rows": pinfo.get("nxt_rows", 0), "patched": pinfo.get("nxt_patched", 0)}
    un = pinfo.get("nxt_unpatched")
    uni = panels["universe"]
    late_days = inp.days[inp.days >= pd.Timestamp(NXT_START)]
    nxt["universe_days"] = int(uni.loc[late_days].to_numpy().sum()) if len(late_days) else 0
    if un is not None and len(un):
        di = uni.index.get_indexer(pd.to_datetime(un["date"]))
        ci = uni.columns.get_indexer(un["code"])
        ok = (di >= 0) & (ci >= 0)
        nxt["universe_unpatched"] = int(uni.to_numpy()[di[ok], ci[ok]].sum())
        unpatched_keys = set(zip(un["code"].astype(str), pd.to_datetime(un["date"])))
    else:
        nxt["universe_unpatched"] = 0
        unpatched_keys = set()

    checks: dict = {}
    if not args.skip_checks:
        # --- 1. 피처 대조: 전 기간 사건 집합 ---
        ms, rp = research_events(db(), base_cfg, t0)
        cut = pd.Timestamp(NXT_START)
        for key, v in VARIANTS.items():
            cfg = v["cfg"]
            r_ev = ms[research_mask(ms, cfg)]
            l_ev = live_events(panels, cfg, EVENT_LO)
            rk, lk = _keys(r_ev), _keys(l_ev)
            pre_r = {k for k in rk if k[1] < cut}
            pre_l = {k for k in lk if k[1] < cut}
            checks[f"events_{key}"] = {
                "research": len(rk), "live": len(lk), "pre_research": len(pre_r), "pre_live": len(pre_l),
                "pre_only_research": sorted(map(str, pre_r - pre_l))[:10], "pre_only_live": sorted(map(str, pre_l - pre_r))[:10],
                "pre_equal": pre_r == pre_l,
                "post_research": len(rk) - len(pre_r), "post_live": len(lk) - len(pre_l),
                "post_common": len((rk - pre_r) & (lk - pre_l)),
            }
            _log(f"사건 대조 {key}: {checks[f'events_{key}']}", t0)
        del rp
        # 그날치 창(300행) — 몇 날짜, 보정 끔(연구와 같은 입력)
        win = []
        ev_days = {}
        for key, v in VARIANTS.items():
            ed = sorted({k[1] for k in _keys(ms[research_mask(ms, v["cfg"])]) if k[1] < cut})
            ev_days[key] = [ed[len(ed) // 3], ed[2 * len(ed) // 3]] if len(ed) >= 3 else ed
        check_days = sorted(set(map(pd.Timestamp, WINDOW_CHECK_DAYS)) | set(ev_days["a"]) | set(ev_days["b"]))
        for d in (str(x.date()) for x in check_days):
            dd = pd.Timestamp(d)
            if dd not in inp.days:
                continue
            wp, _ = signal_panels_for(db(), dd, base_cfg, patch_nxt=False)
            for key, v in VARIANTS.items():
                lk = {(c.code, c.signal_day) for c in scan(wp, dd, v["cfg"])}
                rk = {k for k in _keys(ms[research_mask(ms, v["cfg"])]) if k[1] == dd}
                # 원 S(유니버스 ∧ 뼈대) 층도 대조 — 사건이 적은 날에도 피처가 같은지 보이게
                s_live = int((wp["universe"].loc[dd] & wp["tt"].loc[dd] & wp["breakout"].loc[dd]
                              & (wp["rs"].loc[dd] >= 70) & (wp["breadth"].loc[dd] > 0.5)).sum())
                s_res = int((ms["event_day"] == dd).sum())
                win.append({"day": d, "variant": key, "live": len(lk), "research": len(rk), "equal": lk == rk,
                            "S_live": s_live, "S_research": s_res})
            _log(f"창 대조 {d}: {win[-2:]}", t0)
        checks["window"] = win
        del ms

        # --- 2. 실데이터 패리티(같은 패널, 최근 구간) ---
        par = {}
        for (key, v), (pk, pr) in ((kv, pp) for kv in VARIANTS.items() for pp in PROTOCOLS.items()):
            cfg = replace(v["cfg"], failed_breakout=pr["failed_breakout"], time_stop_days=pr["time_stop_days"])
            res_u = run_backtest(panels, cfg, start=PARITY_LO, unlimited=True)
            ev = live_events(panels, cfg, PARITY_LO)
            rt, rs = simulate_minervini(panels["open"], panels["high"], panels["low"], panels["close"], panels["ma50"],
                                        ev[["code", "event_day", "market", "pivot", "contraction_low"]],
                                        volume=panels["volume"], return_skipped=True,
                                        failed_breakout=pr["failed_breakout"], time_stop_days=pr["time_stop_days"])
            key = f"{key}/{pk}"
            lt = res_u.trades
            n_below = int((lt["exit_reason"] == "below_low").sum())
            lt = lt[lt["exit_reason"] != "below_low"]
            a = lt.set_index(["code", "event_day"]).sort_index()
            b = rt.set_index(["code", "event_day"]).sort_index()
            common = a.index.intersection(b.index)
            a, b = a.loc[common], b.loc[common]
            match = ((a["entry_day"] == b["entry_day"]) & (a["exit_day"] == b["exit_day"])
                     & (a["exit_reason"] == b["exit_reason"])
                     & ((a["ret"].astype(float) - b["ret"].astype(float)).abs() <= 1e-9))
            par[key] = {"events": len(ev), "live_trades": len(lt), "research_trades": len(rt), "common": len(common),
                        "match": int(match.sum()), "below_low_live": n_below,
                        "skip_below_low_research": int((rs["skip_reason"] == "skip_below_low").sum()),
                        "only_live": len(lt) - len(common), "only_research": len(rt) - len(common)}
            par[key]["exits"] = rt["exit_reason"].value_counts().to_dict()
            _log(f"패리티 {key}: {par[key]}", t0)
        checks["parity"] = par

    # --- 3. 지수 ---
    bench, binfo = kospi_series(db(), t0)

    # --- 4. 백테스트 ---
    results = {}
    for (key, v), (pk, pr) in ((kv, pp) for kv in VARIANTS.items() for pp in PROTOCOLS.items()):
        cfg = replace(v["cfg"], failed_breakout=pr["failed_breakout"], time_stop_days=pr["time_stop_days"])
        out = {}
        for tag, cmult, bps, kill in (("base", 1.0, SLIP_BPS, None), ("slip0", 1.0, 0.0, None),
                                      ("cost2x", 2.0, SLIP_BPS, None), ("kill", 1.0, SLIP_BPS, RISK.drawdown_kill)):
            c = replace(cfg, stop_slippage_bps=bps)
            res = run_backtest(panels, c, start=v["start"], initial_equity=INITIAL, cost_fn=_scaled_cost(cmult),
                               dd_kill=kill, kill_resume_days=RISK.kill_resume_days,
                               kill_resume_recovery=RISK.kill_resume_recovery)
            out[tag] = res
            _log(f"백테스트 {key}/{pk}/{tag}: 트레이드 {len(res.trades)} · 최종 {res.equity.iloc[-1]:,.0f}", t0)
        results[(key, pk)] = out

    # --- 뉴스 거부권(설계 §7): (a)/new, 신호 2025-09-01~ — 거부권 있음/없음 + 거부군의 가상 결과 ---
    veto_res = {}
    try:
        tdays = [pd.Timestamp(x).date() for x in inp.days]
        vlo = dt.datetime.combine(dt.date.fromisoformat(VETO_LO) - dt.timedelta(days=45), dt.time(0), tzinfo=KST)
        vhi = dt.datetime.combine(data_end.date() + dt.timedelta(days=1), dt.time(0), tzinfo=KST)
        vpanel = veto_panel_from_ledger(db(), vlo, vhi, tdays, panels["close"].columns)
        cfg_v = replace(VARIANTS["a"]["cfg"], **{k: PROTOCOLS["new"][k] for k in ("failed_breakout", "time_stop_days")})
        with_v = run_backtest(panels, cfg_v, start=VETO_LO, initial_equity=INITIAL, cost_fn=_scaled_cost(1.0), veto=vpanel)
        without = run_backtest(panels, replace(cfg_v, news_veto=False), start=VETO_LO, initial_equity=INITIAL,
                               cost_fn=_scaled_cost(1.0))
        vt = with_v.vetoed
        shadow = pd.DataFrame()
        if vt is not None and len(vt):
            shadow, _ = simulate_minervini(panels["open"], panels["high"], panels["low"], panels["close"], panels["ma50"],
                                           vt[["code", "event_day", "market", "pivot", "contraction_low"]],
                                           volume=panels["volume"], return_skipped=True, failed_breakout=True,
                                           time_stop_days=None, no_overlap=False)
        entered = without.trades
        veto_res = {"with": perf_stats(with_v.daily, with_v.trades, bench=bench),
                    "without": perf_stats(without.daily, without.trades, bench=bench),
                    "n_vetoed": int(len(vt)) if vt is not None else 0, "skips_with": with_v.skips,
                    "veto_days": int(vpanel.loc[vpanel.index >= pd.Timestamp(VETO_LO)].to_numpy().sum()),
                    "shadow": {"n": int(len(shadow)),
                               "mean_ret": float(shadow["ret"].mean()) if len(shadow) else float("nan"),
                               "stop_rate": float((shadow["exit_reason"] == "stop").mean()) if len(shadow) else float("nan"),
                               "win": float((shadow["ret"] > 0).mean()) if len(shadow) else float("nan")},
                    "entered": {"n": int(len(entered)),
                                "mean_ret": float(entered["ret"].mean()) if len(entered) else float("nan"),
                                "stop_rate": float((entered["exit_reason"] == "stop").mean()) if len(entered) else float("nan"),
                                "win": float((entered["ret"] > 0).mean()) if len(entered) else float("nan")},
                    "vetoed_list": vt[["event_day", "code"]].astype(str).values.tolist() if vt is not None else []}
        _log(f"뉴스 거부권 (a)/new {VETO_LO}~: 거부 후보 {veto_res['n_vetoed']} · 거부군 n {veto_res['shadow']['n']} "
             f"평균 {veto_res['shadow']['mean_ret']:+.3f} · 진입군 n {veto_res['entered']['n']} "
             f"평균 {veto_res['entered']['mean_ret']:+.3f}", t0)
    except Exception as exc:  # noqa: BLE001 — 원장이 없으면 거부권 표만 비운다
        veto_res = {"error": f"{type(exc).__name__}: {exc}"}
        _log(f"뉴스 거부권 변형 실패: {veto_res['error']}", t0)

    stats = {}
    for (key, pk), out in results.items():
        res = out["base"]
        s = perf_stats(res.daily, res.trades, bench=bench if key == "a" else None)
        if key != "a":                                      # (b): 지수가 있는 2024-01-26~ 겹침 구간만 따로
            ov = res.daily[res.daily.index >= bench.index[0]]
            s["overlap"] = perf_stats(ov, res.trades[res.trades["entry_day"] >= bench.index[0]], bench=bench)
        s2 = perf_stats(out["cost2x"].daily, out["cost2x"].trades, bench=None)
        s["slip0"] = perf_stats(out["slip0"].daily, out["slip0"].trades, bench=None)
        s["kill"] = perf_stats(out["kill"].daily, out["kill"].trades, bench=None)
        s["kill"]["kills"] = out["kill"].params["kills"]
        dd = drawdown(res.equity)
        kill = dd[dd <= -RISK.drawdown_kill]
        s["dd_kill_first"] = str(kill.index[0].date()) if len(kill) else None
        s["cost2x"] = {k: s2[k] for k in ("cagr", "sharpe", "mdd", "total_return")}
        t = res.trades
        sig_late = t[t["event_day"] >= pd.Timestamp(NXT_START)]
        held_late = 0
        held_unp = 0
        for r in t.itertuples():
            for d in inp.days[(inp.days >= max(r.entry_day, pd.Timestamp(NXT_START))) & (inp.days <= r.exit_day)]:
                held_late += 1
                held_unp += (r.code, d) in unpatched_keys
        s["nxt"] = {"signals_late": int(len(sig_late)),
                    "signals_late_unpatched": int(sum((c, d) in unpatched_keys
                                                      for c, d in zip(sig_late["code"], sig_late["event_day"]))),
                    "held_days_late": held_late, "held_days_late_unpatched": held_unp}
        stats[(key, pk)] = s
        res.trades.to_csv(os.path.join(args.out, f"trades_{key}_{pk}.csv"), index=False)
        res.daily.to_csv(os.path.join(args.out, f"equity_{key}_{pk}.csv"))

    png = plot_equity(results, bench, os.path.join(args.out, "equity.png"))
    elapsed = time.time() - t0
    write_md(args.out, stats, results, checks, binfo, nxt, png, data_end, elapsed, veto_res)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": {f"{k}/{pk}": v for (k, pk), v in stats.items()}, "checks": checks, "bench": binfo,
                   "nxt": nxt, "veto": veto_res}, f, ensure_ascii=False, indent=1, default=str)
    _log(f"완료 → {args.out}", t0)
    return 0


def write_md(out_dir, stats, results, checks, binfo, nxt, png, data_end, elapsed, veto_res=None) -> None:
    L = ["# 스윙 라이브 v0 포트폴리오 백테스트 — 러너 자동 생성, 표본 내 — 판정 아님", "",
         f"러너 `research/experiments/live_portfolio_backtest.py` · 데이터 끝 {data_end.date()} · {elapsed / 60:.1f}분.",
         "전략 코드 = `src/swing_it/live/` (라이브 데몬과 같은 `scan`·`plan_entry`·`allocate`·`PositionState.on_bar`).",
         "",
         "**표본 내(IN-SAMPLE)다.** ADV20 ≥ 200억 구간은 스윙 3호·탐색·모멘텀·미너비니 청산에서 이미 봤고, 50~200억 구간은",
         "`research/logs/sepa_explore/CONVERGENCE.md` 사전등록 시험이 같은 사건을 본다. 아래 숫자로 전략을 고르거나 고치면",
         "다중검정이 된다 — 진짜 확인은 전진 페이퍼다(설계 §1, 사용자 결정).", "",
         "## 정의", "",
         "- 유니버스: `korea.universe_mask` 위생(보통주·KOSPI/KOSDAQ·원종가 ≥ 1,000·상장 252일) ∧ **ADV20 ≥ 5,000 백만원**.",
         "- 진입(신호일 d 종가까지): 추세 템플릿 ∧ RS ≥ 70 ∧ 피벗 돌파(거래량 1.5×) ∧ breadth > 0.5 ∧ PEAD 순위 ≥ 0.70 "
         "(모집단 = 그날 유니버스) ∧ rel ≤ 1.0 ∧ F = 1 (변형 b 는 F 없음).",
         "- 체결: d+1 장전 지정가 — 시가 ≤ 피벗×1.05 ∧ 시가 < 전일종가×(1.30−0.007) 이면 **시가 체결**, 아니면 취소. "
         "분할조정 패널이라 호가 반올림·정확 상한가 대신 비율(연구 `sepa.LIMIT_UP_RATIO` 와 같음).",
         "- 청산 **new(라이브 기본, 설계 §6)**: 수축 저점×0.99 손절 [3%,10%], 2R 절반 익절 + 본전, 나머지 MA50 추적(다음날 시가), "
         "**돌파 실패 = 익절 전 종가 < 피벗 → 다음날 시가 전량**, 시간 손절 없음, 120일 안전 상한. **old(사전등록 원전)**: 돌파 실패 "
         "대신 20일 시간 손절(종가 ≤ 진입가). 일봉 같은 날 손절·익절이면 손절 먼저. 시가 ≤ 수축 저점이면 체결 직후 시가 청산(`below_low`).",
         "- 사이징: **실질 = 종목당 자산 10% 고정**(코드의 min(위험 1%/(종가×d_est), 10%) 에서 d_est ≤ 10% 라 위험 항이 묶이지 "
         "않는다 — REVIEW_BACKTEST M5, 설계서 §2 갱신). 트레이드당 위험 = 10% × d ≈ 1%. 최대 10종목, 현금 예약 = 수량 × 지정가, "
         "후보 순위 rel 오름차순 → RS 내림차순.",
         "- 비용: 매도할 때마다 매도 금액 × `korea.cost_rt(매도일, 시장)`(왕복: 세금+수수료 3bp+스프레드 18.5bp). "
         "분할 청산은 부분마다. 매수 쪽 별도 비용 없음. 결제(T+2) 미모형.",
         f"- 손절·본전 미끄러짐: 주 결과 {SLIP_BPS:.0f}bp(그 매도가 × (1 − bp)), 0bp 도 같이 적는다 — 라이브는 60초 폴링으로 "
         "터치를 본 뒤 판다(감사 AUDIT.md). 트레이드 `ret` 은 미끄러짐 전(연구 정의), 순수익·자산은 미끄러짐·비용 후.",
         f"- 낙폭 킬 −{RISK.drawdown_kill:.0%}: 기본 실행은 끄고 걸렸을 날만 적는다. 재개 규칙(`RiskConfig`, REVIEW_BACKTEST M4): "
         f"킬 뒤 {RISK.kill_resume_days}거래일 경과 **또는** 자산이 고점 −{RISK.kill_resume_recovery:.0%} 이내 회복 중 먼저 오는 것 "
         "— 신규 진입이 없는 동안 자산이 평평해 '둘 중 늦은 것' 은 영원히 재개가 안 된다 — 재개 때 고점을 다시 잰다. "
         "킬+재개를 켠 실행을 변형마다 같이 적는다. 라이브는 사람이 `data/swing_STOP` 을 지워야 풀린다.",
         "- 정수 수량 효과(REVIEW_BACKTEST m2): 익절 = floor(보유 × 0.5) 주라 `ret`(수량 가중)이 연구 정의(두 절반 평균, "
         "`ret_half` 열)와 다를 수 있다 — 3주면 1/3 익절, 1주면 익절 0주. 표의 '연구 정의' 줄이 그 평균이다.",
         "- 시작 자산 1억 원. 상장폐지는 마지막 거래 가능일 종가(연구와 같음 — 정리매매보다 낙관).",
         f"- 지수: `sector_index` '001'({binfo['index_lo']}~). **그 전엔 신뢰할 지수 자료가 없어 비교하지 않는다** — 시총가중 "
         "대용(원자료 종가 × as-of 주식수)은 2019·2020·2022 공개 코스피를 6~9%p 과소했다(REVIEW_BACKTEST M2; 원자료 종가의 "
         "조정 기준 혼재가 유력)라 버렸다. (b) 는 2024-01-26 이후 겹침 구간만 지수와 비교한다.",
         f"- NXT 보정: {NXT_START} 이후 일봉 {nxt['rows']:,}행 중 {nxt['patched']:,}행을 분봉 정규장 값으로 바꿨다"
         f"(`swing_it.krx_session`). 그 구간 유니버스 종목-일 {nxt['universe_days']:,} 중 분봉이 없어 **원값(NXT 섞임)이 남은 것 "
         f"{nxt['universe_unpatched']:,}**.", ""]
    for key in ("a", "b"):
        sn, so = stats[(key, "new")], stats[(key, "old")]
        res = results[(key, "new")]["base"]
        bench_stats = sn.get("bench") or sn.get("overlap", {}).get("bench", {})
        cols = [(f"new {SLIP_BPS:.0f}bp", sn), ("new 0bp", sn["slip0"]), (f"old {SLIP_BPS:.0f}bp", so), ("old 0bp", so["slip0"])]
        L += [f"## 변형 {VARIANTS[key]['label']} — 신호 {VARIANTS[key]['start']} ~ {data_end.date()}", "",
              f"new = {PROTOCOLS['new']['label']} (라이브 기본) · old = {PROTOCOLS['old']['label']}. 코스피 열은 (a) 전 구간, "
              "(b) 는 2024-01-26~ 겹침 구간.", "",
              "| 지표 | " + " | ".join(n for n, _ in cols) + " | 코스피 |", "|---|" + "---|" * (len(cols) + 1),
              *stats_rows(cols, bench_stats), ""]
        for pk, st in (("new", sn), ("old", so)):
            L += [f"- **{pk}** 비용 2배({SLIP_BPS:.0f}bp): CAGR {_pct(st['cost2x']['cagr'])} · Sharpe {_num(st['cost2x']['sharpe'])} · "
                  f"MDD {_pct(st['cost2x']['mdd'])} · 총수익 {_pct(st['cost2x']['total_return'])}. 낙폭 킬 −{RISK.drawdown_kill:.0%} "
                  f"걸렸을 첫날(킬 끈 실행): {st['dd_kill_first'] or '없음'}. **킬+재개 켠 실행**(재개 = {RISK.kill_resume_days}거래일 "
                  f"또는 고점 −{RISK.kill_resume_recovery:.0%} 회복 중 먼저): 총수익 {_pct(st['kill']['total_return'])} · CAGR "
                  f"{_pct(st['kill']['cagr'])} · Sharpe 일 {_num(st['kill']['sharpe'])} / 월 {_num(st['kill']['sharpe_monthly'])} · "
                  f"MDD {_pct(st['kill']['mdd'])} · 트레이드 {st['kill']['n_trades']} · 킬/재개 "
                  + (", ".join(f"{a}→{b or '미재개'}" for a, b in st["kill"]["kills"]) or "없음") + "."]
        by = bench_stats.get("years", {})
        yrs = sorted(set(sn["years"]) | set(by))
        L += ["", f"| 연도 | new {SLIP_BPS:.0f}bp | new 0bp | new 킬+재개 | old {SLIP_BPS:.0f}bp | old 킬+재개 | 코스피 |",
              "|---|---|---|---|---|---|---|"]
        for y in yrs:
            L.append(f"| {y} | {_pct(sn['years'].get(y))} | {_pct(sn['slip0']['years'].get(y))} | {_pct(sn['kill']['years'].get(y))} | "
                     f"{_pct(so['years'].get(y))} | {_pct(so['kill']['years'].get(y))} | {_pct(by.get(y))} |")
        if "overlap" in sn:
            o, ob = sn["overlap"], sn["overlap"]["bench"]
            oo = so["overlap"]
            L += ["", f"지수가 있는 겹침 구간({o['start']} ~ {o['end']}, 자산 그 시점 기준): new 총수익 {_pct(o['total_return'])} "
                  f"(CAGR {_pct(o['cagr'])}, Sharpe 일 {_num(o['sharpe'])}/월 {_num(o['sharpe_monthly'])}, MDD {_pct(o['mdd'])}) · "
                  f"old {_pct(oo['total_return'])} (CAGR {_pct(oo['cagr'])}, MDD {_pct(oo['mdd'])}) vs 코스피 {_pct(ob['total_return'])} "
                  f"(CAGR {_pct(ob['cagr'])}, MDD {_pct(ob['mdd'])}), new 상관 {_num(ob['corr'])} / 베타 {_num(ob['beta'])}. "
                  "**2024-01-26 전 신뢰할 지수 자료가 없다 — 그 구간은 지수 비교 없음.**"]
        L += [""]
        for pk, st in (("new", sn), ("old", so)):
            L.append(f"청산 사유 {pk}: " + " · ".join(f"{k} {v}" for k, v in sorted(st.get("exits", {}).items(),
                                                                                   key=lambda kv: -kv[1])))
        L += ["체결·후보 탈락(new): " + " · ".join(f"{k} {v}" for k, v in res.skips.items()),
              f"NXT 구간(new): 신호 {sn['nxt']['signals_late']}건(그 신호 종목-일 중 미보정 {sn['nxt']['signals_late_unpatched']}) · "
              f"보유 종목-일 {sn['nxt']['held_days_late']}(미보정 {sn['nxt']['held_days_late_unpatched']}).", ""]
    if veto_res:
        L += [f"## 뉴스 거부권(설계 §7) — 변형 (a)/new, 신호 {VETO_LO} ~ {data_end.date()} (판정 원장이 있는 구간, 표본 내)", ""]
        if "error" in veto_res:
            L += [f"원장에서 거부 패널을 만들지 못했다: {veto_res['error']}", ""]
        else:
            w, wo = veto_res["with"], veto_res["without"]
            cols = [("거부권 켬", w), ("거부권 끔", wo)]
            L += ["과거 원장(judge 우선순위 v2r3-haiku55 → v2-haiku45)으로 같은 규칙(5거래일 악재 ≥ 1 또는 20거래일 ≥ 2)을 신호일에 "
                  "적용. 모름(판정 없음)은 거부 안 함 — 원장 커버리지(전일 거래대금 상위 300 종목, 2026-03 판정 단절)만큼만 걸린다.", "",
                  "| 지표 | " + " | ".join(n for n, _ in cols) + " | 코스피 |", "|---|---|---|---|",
                  *stats_rows(cols, w.get("bench", {})), "",
                  f"거부된 후보 {veto_res['n_vetoed']}건(구간 안 거부 패널 참 종목-일 {veto_res['veto_days']:,}). "
                  f"**거부군(가상 진입, 같은 프로토콜, 겹침 허용)** n {veto_res['shadow']['n']} · 평균 총수익 "
                  f"{_pct(veto_res['shadow']['mean_ret'], 2)} · 승률 {_pct(veto_res['shadow']['win'], 0)} · 손절 비율 "
                  f"{_pct(veto_res['shadow']['stop_rate'], 0)} vs **진입군(거부권 끔 트레이드)** n {veto_res['entered']['n']} · "
                  f"평균 {_pct(veto_res['entered']['mean_ret'], 2)} · 승률 {_pct(veto_res['entered']['win'], 0)} · 손절 비율 "
                  f"{_pct(veto_res['entered']['stop_rate'], 0)}. 합격 바(NEWS_VETO.md)는 전진 데이터에서 본다 — 여기는 사후.",
                  "거부 목록: " + (", ".join(f"{d[:10]} {c}" for d, c in veto_res["vetoed_list"]) or "없음"), ""]
    if checks:
        L += ["## 검증", "", "### 사건 집합 — 라이브 `scan` vs 연구 경로(CONVERGENCE A 조립, ADV20 ≥ 5,000)", "",
              f"NXT 보정 전 날짜(< {NXT_START})는 두 경로 입력이 같아야 하므로 집합이 같아야 한다. 그 뒤는 라이브만 보정한다.", "",
              "| 변형 | 연구 (보정 전 날짜) | 라이브 (보정 전 날짜) | 같은가 | 보정 후 연구 / 라이브 / 공통 |", "|---|---|---|---|---|"]
        for key in ("a", "b"):
            e = checks[f"events_{key}"]
            L.append(f"| {key} | {e['pre_research']} | {e['pre_live']} | {'예' if e['pre_equal'] else '아니오'} | "
                     f"{e['post_research']} / {e['post_live']} / {e['post_common']} |")
            if not e["pre_equal"]:
                L.append(f"| | 연구에만 {e['pre_only_research']} | 라이브에만 {e['pre_only_live']} | | |")
        L += ["", "### 그날치 창(300행, `signal_panels_for`, 보정 끔) vs 연구 전 기간", "",
              "| 날짜 | 변형 | 라이브 | 연구 | 같은가 | 뼈대 S 라이브 / 연구 |", "|---|---|---|---|---|---|"]
        for w in checks["window"]:
            L.append(f"| {w['day']} | {w['variant']} | {w['live']} | {w['research']} | {'예' if w['equal'] else '아니오'} | "
                     f"{w['S_live']} / {w['S_research']} |")
        L += ["", f"### 실데이터 패리티 — 신호 {PARITY_LO} ~, 자본·슬롯 무제한 `run_backtest` vs `simulate_minervini`(같은 패널, "
              "두 시뮬레이터에 같은 프로토콜 설정)",
              "", "| 변형/프로토콜 | 사건 | 라이브 트레이드 | 연구 트레이드 | 공통 | 일치(진입·청산일·사유·수익 1e-9) | "
              "below_low(라이브) / skip_below_low(연구) | 연구 청산 사유 |", "|---|---|---|---|---|---|---|---|"]
        for key, p in checks["parity"].items():
            L.append(f"| {key} | {p['events']} | {p['live_trades']} | {p['research_trades']} | {p['common']} | "
                     f"{p['match']} | {p['below_low_live']} / {p['skip_below_low_research']} | "
                     + " · ".join(f"{k} {v}" for k, v in p["exits"].items()) + " |")
        L.append("")
    L += ["## 알려진 차이·모호한 점", "",
          "- 돌파 실패 Action 종류는 `failed_breakout_next_open`(`at=\"next_open\"`, MA50 추적 `trail_next_open` 과 같은 결정·실행 경로 — "
          "d 종가 결정 → d+1 시가 매도). 런타임은 두 종류를 같은 '다음날 시가 매도' 로 맵하면 된다.",
          "- `below_low`: 연구는 시가 ≤ 수축 저점 사건을 버리고(`skip_below_low`), 라이브는 지정가 체결 뒤 바로 판다 — 백테스트는 "
          "그 시가에 청산(총수익 0, 비용만). 실제 라이브는 동시호가 직후 연속매매 가격이라 다를 수 있다.",
          "- 라이브 장중 감시(`on_quote`, 60초 폴링)는 실제 시간 순서를 따른다. 일봉 백테스트는 같은 날 손절·익절이면 손절 먼저 "
          "(비관) — 갭 상승 시가 익절 뒤 손절가까지 빠진 날 라이브는 절반을 이미 팔았다.",
          "- 정수 수량: 익절은 floor(보유 × 0.5) 주. 1주 포지션은 익절 매도 0주(상태만 본전·추적으로 넘어간다).",
          "- 사이징 기준가는 신호일 종가(진입가는 체결 전엔 모른다). 체결가가 더 높으면 종목 비중이 10% 를 조금 넘을 수 있다.",
          "- 백테스트 체결 모형은 분할조정 패널 위 비율 상한가·무반올림 지정가다. 라이브는 원자료 호가 내림 지정가 — 같은 집합"
          "(호가 위 시가)이지만 조정 패널에선 확인할 수 없다.",
          "- 상장폐지·정리매매·결제일·호가 스프레드 실측은 모형 밖이다.", ""]
    if png:
        L += ["## 자산 곡선", "", "![equity](equity.png)", ""]
    with open(os.path.join(out_dir, "BACKTEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
