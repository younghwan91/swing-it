#!/usr/bin/env python
"""공시 촉매 × 수급 확인 × 추세 스윙 — 사전등록 러너.

사전등록: ``research/logs/catalyst_flow_swing/VERDICT.md`` (2026-10-08, 결과 보기 전 커밋 2f43074).

배터리는 재발명하지 않는다 — ``prop_gate`` 가 비용 스윕·폴드·R 분포·취약성·DSR 을, core
``stats.matched_null`` 이 같은 날·같은 층 무작위 대조를 낸다(prop_gate 의 랜덤 널은 OOS 비교에
무효로 알려져 있어 정합 대조가 주 대조다). 민감도 칸은 전부 ``gate_sim(config=)`` 로 원장에 남는다.

결과는 ``research/logs/catalyst_flow_swing/RUN.md`` 에 쓴다(판정은 사람이 VERDICT 에 적는다).

Run:  uv run --extra pg python research/experiments/catalyst_flow_swing_gate.py
"""

from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from krx_quant_core.stats.matched_null import (  # noqa: E402
    cluster_bootstrap_ci, matched_alpha, matched_control)
from krx_quant_core.stats.trials import record_trial  # noqa: E402
from prop_gate import prop_gate  # noqa: E402
from prop_swing_common import gate_sim, load_env_db  # noqa: E402

from research.signals.catalyst_flow import (  # noqa: E402
    PREREG, add_strata, build_events, control_pool, load_panels, simulate_events, strata_from_pool)
from swing_it.features.catalysts import read_dart  # noqa: E402
from swing_it.storage import connect, db_default  # noqa: E402

OUT_DIR = "research/logs/catalyst_flow_swing"
LABEL = "catalyst_flow_swing"
COST_RT = 0.00415  # 왕복 41.5bp (세금 20 + 수수료 3 + 스프레드 18.5, cross-repo synthesis)


def _cfg(**kw) -> dict:
    c = dict(PREREG)
    c.update(kw)
    return c


def _trades(p, cfg, dart) -> pd.DataFrame:
    ev = build_events(p, cfg, dart=dart)
    return simulate_events(p, ev, hold=cfg["hold"], stop=cfg["stop"], min_gap=cfg["hold"])


def _matched(p, tr: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    pool = add_strata(control_pool(p, pd.DatetimeIndex(tr["event_day"]), cfg,
                                   hold=cfg["hold"], stop=cfg["stop"]), "date")
    t = strata_from_pool(tr, pool).reset_index(drop=True)
    pool = pool.reset_index(drop=True)
    strata = ["date", "q_adv", "q_ret5", "market"]
    ctl = matched_control(t, pool, strata=strata, n_per=5, seed=0, exclude_self=False)
    ma = matched_alpha(t, pool, ctl, value="ret", cluster="date", n_boot=2000, seed=0)
    return ma.__dict__, pool


def _index_hedged(con, tr: pd.DataFrame, p, hold: int) -> np.ndarray:
    """같은 보유 구간(진입일 시가 → hold 번째 날 종가) 시장 지수 수익을 뺀 값. 2024-01-26 이전 지수 없음."""
    idx = pd.read_sql_query(
        "SELECT code, date, open, close FROM sector_index WHERE code IN ('001','101')", con)
    idx["date"] = pd.to_datetime(idx["date"])
    op = idx.pivot(index="date", columns="code", values="open").reindex(p.days)
    cl = idx.pivot(index="date", columns="code", values="close").reindex(p.days)
    out = np.full(len(tr), np.nan)
    for i, r in enumerate(tr.itertuples()):
        k = "001" if r.market == "KOSPI" else "101"
        e = p.days.get_loc(r.entry_day)
        x = min(e + hold - 1, len(p.days) - 1)
        o, c = op[k].iloc[e], cl[k].iloc[x]
        if np.isfinite(o) and np.isfinite(c) and o > 0:
            out[i] = r.ret - (c / o - 1.0)
    return out


def _boot_ci(x: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x)
    lo, hi, _ = cluster_bootstrap_ci(x[ok], clusters[ok], n_boot=2000, seed=0)
    return lo, hi


def _describe(tr: pd.DataFrame, net: np.ndarray) -> dict:
    d = pd.to_datetime(tr["event_day"]).to_numpy()
    lo, hi = _boot_ci(net, d)
    srt = np.sort(net)
    k = max(1, int(np.ceil(0.01 * len(srt))))
    month = pd.Series(net, index=pd.to_datetime(tr["entry_day"])).groupby(pd.Grouper(freq="ME")).mean().dropna()
    year = pd.Series(net, index=pd.to_datetime(tr["entry_day"])).groupby(lambda x: x.year).agg(["mean", "size"])
    return {
        "n": int(len(net)), "mean": float(np.mean(net)), "ci": (lo, hi), "median": float(np.median(net)),
        "win": float(np.mean(net > 0)), "mean_drop_top1pct": float(np.mean(srt[:-k])) if len(srt) > k else np.nan,
        "months_pos": int((month > 0).sum()), "months": int(len(month)), "year": year,
    }


def _fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x * 100:+.2f}%"


def main() -> int:
    load_env_db()  # .env 의 KR_QUANT_DB — 없으면 db_default() 가 로컬 sqlite 로 떨어진다
    con = connect(db_default())
    dart = read_dart()
    p = load_panels(con, start="2023-06-01")
    L: list[str] = [f"# 실행 결과 — {date.today().isoformat()} (러너 자동 생성, 판정 아님)", ""]

    # --- 사전등록 config ---
    tr = _trades(p, PREREG, dart)
    rep = prop_gate(tr["entry_day"].dt.strftime("%Y-%m-%d").to_numpy(), tr["ret"].to_numpy(), PREREG["stop"],
                    label=LABEL, log_dir=OUT_DIR, config=PREREG)
    net = tr["ret"].to_numpy() - COST_RT
    desc = _describe(tr, net)
    ma, _pool = _matched(p, tr, PREREG)
    hedged = _index_hedged(con, tr, p, PREREG["hold"]) - COST_RT
    hlo, hhi = _boot_ci(hedged, pd.to_datetime(tr["event_day"]).to_numpy())
    gr = rep.get("gate_report", {}) or {}

    L += ["## 사전등록 config", "",
          f"- 트레이드 n = {desc['n']} (진입 {rep['entry_range']}), 가족별: "
          + ", ".join(f"{k} {v}" for k, v in tr["family"].value_counts().items()),
          f"- **정합 대조 초과(비용 전, 10일):** alpha {_fmt_pct(ma['alpha'])} "
          f"[{_fmt_pct(ma['ci_low'])}, {_fmt_pct(ma['ci_high'])}] · 트레이드 {_fmt_pct(ma['trade_mean'])} vs 대조 "
          f"{_fmt_pct(ma['control_mean'])} · p(>0) {ma['p_greater']:.3f} · 짝 없음 {ma.get('n_unmatched', 'nan')}",
          f"- **비용(41.5bp) 차감:** 평균 {_fmt_pct(desc['mean'])} [{_fmt_pct(desc['ci'][0])}, {_fmt_pct(desc['ci'][1])}] "
          f"· 중앙값 {_fmt_pct(desc['median'])} · 승률 {desc['win']:.1%} · 상위 1% 제거 {_fmt_pct(desc['mean_drop_top1pct'])}",
          f"- **지수 헤지(비용 차감):** 평균 {_fmt_pct(np.nanmean(hedged))} [{_fmt_pct(hlo)}, {_fmt_pct(hhi)}]",
          f"- 월별 양수 {desc['months_pos']}/{desc['months']}",
          "- 연도별(비용 차감 평균 · n): " + ", ".join(
              f"{y} {_fmt_pct(r['mean'])} · {int(r['size'])}" for y, r in desc["year"].iterrows()),
          f"- prop_gate: 엣지 사망 비용 {rep['cost_edge_dies']}, 원장 N {rep.get('n_trials')}, "
          f"gate_report 키 {sorted(gr)[:12]}",
          ""]
    cs = rep["cost_sweep"]
    L += ["### 비용 스윕 (prop_gate, OOS 진입≥2022 = 전 표본)", "", "| 왕복 | OOS n | 기대값 R | 승률 |", "|---|---|---|---|"]
    for row in cs:
        L.append(f"| {int(round(row['cost'] * 1e4))}bp | {row.get('oos_n')} | {row.get('exp_r', row.get('oos_exp_r', 'nan'))} | {row.get('win', '')} |")
    L.append("")

    # --- 민감도 (승자선택 금지, 전부 원장) ---
    L += ["## 민감도 (전부 원장 기록 · 승자선택 금지)", "", "| 칸 | n | 비용차감 평균 | 중앙값 | 승률 | 정합 alpha |", "|---|---|---|---|---|---|"]
    cells = {
        "보유 5일": _cfg(hold=5), "보유 20일": _cfg(hold=20, event_hi="2026-09-08"),
        "수급 조건 끔": _cfg(require_flow=False), "추세 조건 끔": _cfg(require_trend=False),
        "상승 상한 끔": _cfg(require_calm=False),
    }
    for name, cfg in cells.items():
        t = _trades(p, cfg, dart)
        if t.empty:
            L.append(f"| {name} | 0 | | | | |")
            continue
        gate_sim(t["entry_day"].dt.strftime("%Y-%m-%d").to_numpy(), t["ret"].to_numpy(), cfg["stop"],
                 LABEL + "_sens", config=cfg, log_dir=OUT_DIR)
        d = _describe(t, t["ret"].to_numpy() - COST_RT)
        m, _ = _matched(p, t, cfg)
        L.append(f"| {name} | {d['n']} | {_fmt_pct(d['mean'])} | {_fmt_pct(d['median'])} | {d['win']:.1%} | "
                 f"{_fmt_pct(m['alpha'])} [{_fmt_pct(m['ci_low'])}, {_fmt_pct(m['ci_high'])}] |")
    record_trial(LABEL + "_sens", {**PREREG, "family_split": True}, logs_dir=OUT_DIR)
    L += ["", "### 촉매 가족별 (사전등록 config, 1칸으로 셈)", "", "| 가족 | n | 비용차감 평균 | 중앙값 | 승률 |", "|---|---|---|---|---|"]
    for fam, g in tr.groupby("family"):
        d = _describe(g, g["ret"].to_numpy() - COST_RT)
        L.append(f"| {fam} | {d['n']} | {_fmt_pct(d['mean'])} | {_fmt_pct(d['median'])} | {d['win']:.1%} |")
    L.append("")

    # --- 참고: 2016~2023 촉매+추세+상한(수급 없음) — 판정에 안 씀 ---
    p_old = load_panels(con, start="2016-06-01")
    cfg_old = _cfg(require_flow=False, event_lo="2016-12-01", event_hi="2023-12-15")
    t_old = _trades(p_old, cfg_old, dart)
    if not t_old.empty:
        gate_sim(t_old["entry_day"].dt.strftime("%Y-%m-%d").to_numpy(), t_old["ret"].to_numpy(), cfg_old["stop"],
                 LABEL + "_sens", config=cfg_old, log_dir=OUT_DIR)
        d = _describe(t_old, t_old["ret"].to_numpy() - COST_RT)
        m, _ = _matched(p_old, t_old, cfg_old)
        L += ["## 참고 — 2016-12~2023-12 촉매+추세+상한(수급 없음, 판정 미사용)", "",
              f"- n {d['n']} · 비용차감 평균 {_fmt_pct(d['mean'])} [{_fmt_pct(d['ci'][0])}, {_fmt_pct(d['ci'][1])}] · "
              f"중앙값 {_fmt_pct(d['median'])} · 승률 {d['win']:.1%} · 정합 alpha {_fmt_pct(m['alpha'])} "
              f"[{_fmt_pct(m['ci_low'])}, {_fmt_pct(m['ci_high'])}]",
              "- 연도별: " + ", ".join(f"{y} {_fmt_pct(r['mean'])} · {int(r['size'])}" for y, r in d["year"].iterrows()), ""]

    tr.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    with open(os.path.join(OUT_DIR, "RUN.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
