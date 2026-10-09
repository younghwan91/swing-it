#!/usr/bin/env python
"""모멘텀 정합 대조 재측정 — 126칸 탐색(``sepa_explore``)의 같은 칸·같은 트레이드를 **다른 대조**로 다시 잰다.

사전등록: ``research/logs/sepa_explore/MOMENTUM.md`` (결과 보기 전 커밋 f43babd). 대조 정의·대상(126칸 전부, 판정은
"방향 가족" 12칸)·합격 바 3개·기술 항목을 그대로 옮긴다. 이 러너는 **리포터**다 — 바 숫자를 계산해
``MOMENTUM_RESULTS.md`` 에 적을 뿐 통과·기각을 쓰지 않는다(판정은 사람이).

세 대조(전부 같은 트레이드, 같은 대조 풀 = 그날 유니버스 전 종목을 같은 진입·청산 규칙으로 시뮬한 것):

- **기존** — 스윙 3호 그대로(같은 날 × ADV20 4분위 × 직전 5일 수익 4분위 × 시장). ``sepa_confluence_gate.matched``
  를 칸 순서 그대로 다시 불러 ``cells.csv`` 의 alpha 를 재현한다(재현 확인용 — 숫자는 기존 탐색과 같아야 한다).
- **모멘텀(이게 측정)** — 같은 날 × 시장 × 12-1 모멘텀 10분위 × ADV20 2분위(그날 시장 내 중앙값 이상/미만).
- **보조(기술만)** — 풀을 그날 추세 템플릿 ∧ RS ≥ 70 ∧ 유니버스 종목으로 좁히고, 층 = 같은 날 × 시장 × ADV20 2분위.

공통: 자기 종목 제외, n_per 5(복원 추출), seed 0, 날짜 군집 부트스트랩 2,000회(core ``matched_null``).
기존 러너(``sepa_explore``·``sepa_confluence_gate``)는 건드리지 않고 import 만 한다.

Run:  uv run --extra pg python research/experiments/sepa_momentum.py
디버그(구간 축소 — 탐색 디렉터리를 더럽히지 않는다):
      ... sepa_momentum.py --event-lo 2020-01-01 --event-hi 2020-06-30 --out <scratch>/sepa_momentum_debug
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from krx_quant_core.stats.matched_null import matched_alpha, matched_control  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402
from sepa_confluence_gate import (  # noqa: E402
    N_BOOT,
    N_PER,
    SEED,
    Data,
    Pools,
    _ci,
    _key,
    _log,
    _pct,
    attach_features,
    day_quartiles,
    e_pass,
    matched,
    simulate,
    sub_alpha,
)
from sepa_explore import (  # noqa: E402
    ENTRIES,
    EVENT_HI,
    EVENT_LO,
    EXITS,
    GATES,
    HALVES,
    PEAD_NEW,
    REGIMES,
    SIM_COLS,
    _base,
    build_cells,
    cell_id,
    pead_new_events,
    regime_mask,
)

from research.signals.sepa import build_skeleton_events  # noqa: E402
from swing_it.storage import connect, db_default  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
OLD_CELLS = "research/logs/sepa_explore/cells.csv"
MOM_SKIP, MOM_LOOKBACK = 21, 252           # 12-1 모멘텀: close[t−21] / close[t−252] − 1
MOM_N_Q = 10
RS_MIN_AUX = 70.0

# --- 방향 가족 12칸 (MOMENTUM.md "대상" — 축의 교집합) ---
FAMILY_ENTRY = "breakout"
FAMILY_GATES = ("E", "pead≥0.70", "pead≥0.90")
FAMILY_EXITS = ("⑥ 손절없음+60일", "⑦ 손절없음+120일")
FAMILY_REGIMES = ("breadth>0.5", "끔")
BAR2_MIN = 4                                # 바 2: 12칸 중 4칸 이상 CI 하단 > 0

STRATA_OLD = ["date", "q_adv", "q_ret5", "market"]
STRATA_MOM = ["date", "market", "q_mom", "adv2"]
STRATA_AUX = ["date", "market", "adv2"]


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_sepa_momentum_helpers.py)
# ---------------------------------------------------------------------------

def momentum_12_1(close: pd.DataFrame, *, skip: int = MOM_SKIP, lookback: int = MOM_LOOKBACK) -> pd.DataFrame:
    """12-1 모멘텀 = ``close[t−skip] / close[t−lookback] − 1`` (Jegadeesh-Titman, 최근 1개월 제외).

    ``t`` 행 값은 ``t−skip`` 행 종가까지만 본다(``t`` 종가도 안 쓴다 — 시점 안전). 어느 한쪽이 NaN 이거나
    ``close[t−lookback] ≤ 0`` 이면 NaN. 행 오프셋은 패널의 거래일 순서다(분할조정 종가를 넘긴다).
    """
    a = close.shift(skip)
    b = close.shift(lookback)
    return a / b.where(b > 0) - 1.0


def day_market_quantiles(values: pd.DataFrame, mask: pd.DataFrame, market: pd.Series, *,
                         n_q: int) -> pd.DataFrame:
    """날짜 × 시장마다 ``mask`` 안 종목 사이의 분위(0..n_q−1). ``day_quartiles`` 를 시장별 열 묶음에 그대로 쓴다
    (같은 날 순위 ``method='first'`` → 등분, 유효 종목이 ``n_q`` 미만인 날·시장은 NaN). 시장이 없는 열은 NaN."""
    mk = market.reindex(values.columns)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for m in pd.unique(mk.dropna()):
        cols = values.columns[(mk == m).to_numpy()]
        out[cols] = day_quartiles(values[cols], mask.reindex(columns=cols), n_q=n_q)
    return out


def day_market_median_half(values: pd.DataFrame, mask: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    """날짜 × 시장마다 ``mask`` 안 유효 종목의 중앙값 기준 2분위: 값 ≥ 중앙값 → 1, 미만 → 0.

    ``mask`` 밖·값 NaN 은 NaN. 그날 그 시장 유효 종목이 2 미만이면 NaN(가를 수 없다). 시장이 없는 열은 NaN.
    """
    mk = market.reindex(values.columns)
    m_ok = mask.reindex_like(values).fillna(False).astype(bool)
    v = values.where(m_ok)
    v = v.where(np.isfinite(v))
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for m in pd.unique(mk.dropna()):
        cols = values.columns[(mk == m).to_numpy()]
        sub = v[cols]
        med = sub.median(axis=1)
        cnt = sub.notna().sum(axis=1)
        half = (sub.ge(med, axis=0)).astype(float).where(sub.notna())
        half.loc[(cnt < 2).to_numpy()] = np.nan
        out[cols] = half
    return out


def leader_mask(uni: pd.DataFrame, tt: pd.DataFrame, rs: pd.DataFrame, *, rs_min: float = RS_MIN_AUX) -> pd.DataFrame:
    """보조 대조 풀 = 그날 유니버스 ∧ 추세 템플릿 ∧ RS ≥ ``rs_min`` (bool, NaN 은 거짓)."""
    u = uni.fillna(False).astype(bool)
    t = tt.reindex_like(u).fillna(False).astype(bool)
    r = (rs.reindex_like(u) >= rs_min).fillna(False)
    return u & t & r


def attach_strata(df: pd.DataFrame, day_col: str, days: pd.DatetimeIndex, codes: pd.Index,
                  panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """행마다 (날짜, 종목) 위치의 패널 값을 붙인다. 패널 밖(날짜·종목 없음)은 NaN."""
    df = df.copy()
    di = days.get_indexer(pd.to_datetime(df[day_col]))
    ci = codes.get_indexer(df["code"])
    ok = (di >= 0) & (ci >= 0)
    for name, panel in panels.items():
        arr = panel.to_numpy(dtype=float)
        vals = np.full(len(df), np.nan)
        vals[ok] = arr[di[ok], ci[ok]]
        df[name] = vals
    return df


def candidate_counts(t: pd.DataFrame, pool: pd.DataFrame, strata: list[str]) -> pd.Series:
    """트레이드마다 같은 층의 풀 후보 수(자기 종목 제외). 층 키에 NaN 이 있으면 0.

    core ``matched_control`` 은 후보가 1개 이상이면 ``n_per`` 개를 **복원 추출**한다 — 후보가 ``n_per`` 미만인
    층에서는 같은 대조가 여러 번 뽑힌다. 이 수는 그 "얇은 층" 비율을 적는 데 쓴다.
    """
    size = pool.groupby(strata, dropna=True).size().rename("_n")
    k = t[strata].copy()
    k["_self"] = t.index.isin(pool.index).astype(int)
    j = k.join(size, on=strata)
    n = (j["_n"].fillna(0) - j["_self"]).clip(lower=0)
    n[t[strata].isna().any(axis=1)] = 0
    return n.astype(int)


def matched_on(t: pd.DataFrame, pool: pd.DataFrame, strata: list[str]) -> dict:
    """``sepa_confluence_gate.matched`` 와 같은 모양(``sub_alpha`` 호환)을 임의 층·풀로."""
    tt = t.assign(date=t["event_day"])
    ctl = matched_control(tt, pool, strata=strata, n_per=N_PER, seed=SEED, exclude_self=True)
    ma = matched_alpha(tt, pool, ctl, value="ret", cluster="date", n_boot=N_BOOT, seed=SEED)
    return {"ma": ma, "ctl": ctl, "pool": pool, "t": tt}


def is_family(c: dict) -> bool:
    return (c["entry"] == FAMILY_ENTRY and c["gate"] in FAMILY_GATES and c["exit"] in FAMILY_EXITS
            and c["regime"] in FAMILY_REGIMES)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

def _fields(prefix: str, a) -> dict:
    if a is None or not np.isfinite(a.alpha):
        return {f"{prefix}alpha": np.nan, f"{prefix}lo": np.nan, f"{prefix}hi": np.nan, f"{prefix}n": 0}
    return {f"{prefix}alpha": a.alpha, f"{prefix}lo": a.ci_low, f"{prefix}hi": a.ci_high, f"{prefix}n": a.n_trades}


def _a(r: dict, pre: str) -> str:
    a = r.get(f"{pre}alpha", np.nan)
    if a is None or not np.isfinite(a):
        return "nan"
    return f"{_pct(a)} {_ci(r.get(f'{pre}lo'), r.get(f'{pre}hi'))}"


def _rate(k, n) -> str:
    return "—" if not n else f"{k / n:.1%}"


def _med(s: pd.Series) -> float:
    s = s.astype(float).dropna()
    return float(s.median()) if len(s) else np.nan


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--event-lo", default=None, help="디버그 전용 — 사건 구간 축소(--out 을 탐색 디렉터리 밖으로)")
    ap.add_argument("--event-hi", default=None)
    ap.add_argument("--out", default=OUT_DIR, help="MOMENTUM_RESULTS.md·cells_momentum.csv 디렉터리")
    args = ap.parse_args(argv)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if (args.event_lo or args.event_hi) and not debug:
        ap.error("구간 축소는 --out 을 탐색 디렉터리 밖으로 둘 때만")
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = args.event_lo or EVENT_LO, args.event_hi or EVENT_HI

    cells = build_cells()
    if args.event_lo or args.event_hi:
        cells = [{**c, "event_lo": lo, "event_hi": hi} for c in cells]
    assert len(cells) == 126 and len({cell_id(c) for c in cells}) == 126
    fam = [cell_id(c) for c in cells if is_family(c)]
    assert len(fam) == 12, fam

    t0 = time.time()
    load_env_db()
    con = connect(db_default())
    D = Data(con, t0)
    pools = Pools(D)
    p = D.p

    # --- 새 층 패널(전부 d 종가까지, 유니버스 단면에서 경계) ---
    mom = momentum_12_1(p.close)
    q_mom = day_market_quantiles(mom, D.uni, p.market, n_q=MOM_N_Q)
    adv2 = day_market_median_half(D.adv20, D.uni, p.market)
    leader = leader_mask(D.uni, D.feats["tt"], D.feats["rs"]).astype(float)
    spanel = {"q_mom": q_mom, "adv2": adv2, "leader": leader}
    ev_days = p.days[(p.days >= pd.Timestamp(lo)) & (p.days <= pd.Timestamp(hi))]
    u_ev = D.uni.loc[ev_days].to_numpy()
    n_uni = int(u_ev.sum())
    n_qmom_nan = int((u_ev & ~np.isfinite(q_mom.loc[ev_days].to_numpy())).sum())
    n_adv2_nan = int((u_ev & ~np.isfinite(adv2.loc[ev_days].to_numpy())).sum())
    n_leader = int(leader.loc[ev_days].to_numpy().sum())
    first_row = int(p.days.get_indexer([ev_days[0]])[0]) if len(ev_days) else -1
    _log(f"층 패널: 유니버스 종목-일(사건 구간) {n_uni:,} · 모멘텀 분위 NaN {n_qmom_nan:,} · ADV 2분위 NaN {n_adv2_nan:,} "
         f"· 주도주 풀 {n_leader:,}", t0)
    del mom

    # --- 사건 표: sepa_explore.main 과 같은 식(같은 master·같은 칸 순서 → 같은 트레이드·같은 대조 풀 순서) ---
    masters: dict[str, pd.DataFrame] = {}
    for entry in ENTRIES:
        mcfg = _base(require_breakout=(entry == "breakout"), breadth_min=REGIMES["끔"], event_lo=lo, event_hi=hi)
        ev = build_skeleton_events(p, mcfg, universe=D.uni, feats=D.feats)
        ev = attach_features(D, ev)
        ev.index = _key(ev["event_day"], ev["code"])
        masters[entry] = ev
    pn = attach_features(D, pead_new_events(D, lo, hi))
    pn.index = _key(pn["event_day"], pn["code"])
    masters[PEAD_NEW] = pn
    _log("사건 master " + " · ".join(f"{k} {len(v):,}" for k, v in masters.items()), t0)

    def events_for(c: dict) -> pd.DataFrame:
        m = masters[c["entry"]]
        if c["entry"] == PEAD_NEW:
            ev = m[regime_mask(m, c["breadth_min"])]
        else:
            s = build_skeleton_events(p, c, universe=D.uni, feats=D.feats)
            ev = m.loc[m.index.intersection(_key(s["event_day"], s["code"]))]
            ev = ev[e_pass(ev, c).to_numpy()]
        return ev.sort_values(["event_day", "code"]).reset_index(drop=True)

    years = list(range(pd.Timestamp(lo).year, pd.Timestamp(hi).year + 1))
    rows, year_rows = [], []
    ev_cache: dict[tuple, pd.DataFrame] = {}
    for i, c in enumerate(cells, 1):
        k = (c["entry"], c["gate"], c["regime"])
        if k not in ev_cache:
            ev_cache[k] = events_for(c)
        ev = ev_cache[k]
        cid = cell_id(c)
        row = {"cell": cid, "entry": c["entry"], "gate": c["gate"], "exit": c["exit"], "regime": c["regime"],
               "family": cid in fam, "n_events": len(ev)}
        t = simulate(D, ev[SIM_COLS], c) if len(ev) else pd.DataFrame()
        row["n"] = len(t)
        if t.empty:
            rows.append(row)
            continue
        # 기존 대조 — sepa_explore 와 같은 호출(풀 캐시도 같은 순서로 자란다)
        m_old = matched(pools, t, c)
        pool = attach_strata(m_old["pool"], "date", p.days, p.close.columns, spanel)
        tm = attach_strata(t.assign(date=t["event_day"]), "event_day", p.days, p.close.columns, spanel)
        m_mom = matched_on(tm, pool, STRATA_MOM)
        pool_l = pool[pool["leader"] == 1.0]
        m_aux = matched_on(tm, pool_l, STRATA_AUX)
        for pre, m in (("old_", m_old), ("mom_", m_mom), ("aux_", m_aux)):
            ma = m["ma"]
            row.update(_fields(pre, ma))
            row[f"{pre}p_greater"] = float(ma.p_greater) if np.isfinite(ma.alpha) else np.nan
            row[f"{pre}unmatched"] = int(ma.n_unmatched)
        for pre, m in (("mom_", m_mom), ("aux_", m_aux)):
            for (_, hlo, hhi), h in zip(HALVES, ("h1_", "h2_"), strict=True):
                mk = (t["event_day"] >= pd.Timestamp(hlo)) & (t["event_day"] < pd.Timestamp(hhi))
                row.update(_fields(pre + h, sub_alpha(m, mk) if mk.any() else None))
        row["diff_old_mom"] = row["old_alpha"] - row["mom_alpha"]
        cm, ca = candidate_counts(tm, pool, STRATA_MOM), candidate_counts(tm, pool_l, STRATA_AUX)
        row["mom_thin"] = int(((cm > 0) & (cm < N_PER)).sum())
        row["mom_cand_median"] = float(cm[cm > 0].median()) if (cm > 0).any() else np.nan
        row["aux_thin"] = int(((ca > 0) & (ca < N_PER)).sum())
        row["aux_cand_median"] = float(ca[ca > 0].median()) if (ca > 0).any() else np.nan
        row["mom_nan_strata"] = int(tm[STRATA_MOM].isna().any(axis=1).sum())
        row["aux_nan_strata"] = int(tm[STRATA_AUX].isna().any(axis=1).sum())
        rows.append(row)
        if cid in fam:
            for y in years:
                mk = t["event_day"].dt.year == y
                a = sub_alpha(m_mom, mk) if mk.any() else None
                year_rows.append({"cell": cid, "year": y, "n": int(mk.sum()), **_fields("", a)})
        _log(f"[{i:3d}/126] {cid}: n {len(t):,} · 기존 {_pct(row['old_alpha'])} · 모멘텀 {_pct(row['mom_alpha'])} "
             f"· 보조 {_pct(row['aux_alpha'])}", t0)

    df = pd.DataFrame(rows)
    yr = pd.DataFrame(year_rows)

    # --- 재현 확인: 기존 alpha 가 cells.csv(995b277) 와 같은가 ---
    repro = None
    if not debug and os.path.exists(OLD_CELLS):
        old = pd.read_csv(OLD_CELLS).set_index("cell")
        # cells.csv 의 n 은 cell_stats 에서 matched_alpha 의 n_trades(짝 있는 트레이드 수)로 덮인다 — old_n 과 비교한다.
        j = df.set_index("cell").join(old[["n", "alpha", "lo", "hi", "n_unmatched"]], rsuffix="_csv")
        repro = {"n_same": int((j["old_n"] == j["n_csv"]).sum()),
                 "unm_same": int((j["old_unmatched"] == j["n_unmatched"]).sum()),
                 "alpha_maxdiff": float((j["old_alpha"] - j["alpha"]).abs().max()),
                 "ci_maxdiff": float(np.nanmax(np.abs(np.r_[j["old_lo"] - j["lo"], j["old_hi"] - j["hi"]])))}

    # --- 합격 바 숫자 ---
    F = df[df["family"]].copy()
    b1 = {k: _med(F[c]) for k, c in (("all", "mom_alpha"), ("h1", "mom_h1_alpha"), ("h2", "mom_h2_alpha"))}
    b2_cells = F[F["mom_lo"].astype(float) > 0]["cell"].tolist()

    def axis_med(col: str) -> dict:
        out = {}
        out["entry"] = {v: _med(df.loc[df["entry"] == v, col]) for v in [*ENTRIES, PEAD_NEW]}
        g = df[df["entry"] != PEAD_NEW]
        out["gate"] = {v: _med(g.loc[g["gate"] == v, col]) for v in GATES}
        out["gate_3med"] = _med(pd.Series([out["gate"][v] for v in GATES if v != "off"]))
        out["gate_pooled3"] = _med(g.loc[g["gate"] != "off", col])
        out["exit"] = {v: _med(df.loc[df["exit"] == v, col]) for v in EXITS}
        out["regime"] = {v: _med(df.loc[df["regime"] == v, col]) for v in REGIMES}
        return out

    ax = {pre: axis_med(f"{pre}alpha") for pre in ("old_", "mom_", "aux_")}
    ax_h = {h: axis_med(f"mom_{h}alpha") for h in ("h1_", "h2_")}

    # --- 리포트 ---
    elapsed = time.time() - t0
    L = [f"# 모멘텀 정합 대조 재측정 — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "사전등록 `MOMENTUM.md`(f43babd, 결과 보기 전) 의 대조 정의·대상·합격 바·기술 항목을 그대로 계산한 숫자다. "
         "합격 여부는 여기 적지 않는다 — 판정은 사람이 쓴다.", ""]
    if debug:
        L += [f"> **디버그 실행** — 사건 구간 {lo}~{hi}, 출력 `{out_dir}`. 측정으로 쓰지 않는다.", ""]
    L += ["## 표본", "",
          f"- 칸 126(`EXPLORE.md` 격자 그대로) · 사건 구간 {lo} ~ {hi} · 가격 패널 {p.days[0].date()}~{p.days[-1].date()} "
          f"(사건 구간 첫날 = 패널 {first_row}번째 행, 0부터)",
          f"- 칸별 트레이드 n: 중앙값 {int(df['n'].median()):,} · 최소 {int(df['n'].min()):,} · 최대 {int(df['n'].max()):,} "
          f"· 합 {int(df['n'].sum()):,}",
          f"- 사건 구간 유니버스 종목-일 {n_uni:,} — 모멘텀 10분위 NaN {n_qmom_nan:,} ({_rate(n_qmom_nan, n_uni)}) · "
          f"ADV 2분위 NaN {n_adv2_nan:,} ({_rate(n_adv2_nan, n_uni)}) · 주도주 풀(추세 템플릿 ∧ RS≥70 ∧ 유니버스) "
          f"{n_leader:,} ({_rate(n_leader, n_uni)})"]
    if repro is not None:
        L.append(f"- **기존 alpha 재현(cells.csv, 995b277):** 짝 있는 트레이드 n 일치 {repro['n_same']}/126 · 짝 없음 일치 "
                 f"{repro['unm_same']}/126 · alpha 최대 차이 "
                 f"{repro['alpha_maxdiff']:.2e} · CI 끝 최대 차이 {repro['ci_maxdiff']:.2e}")
    L.append("")

    L += ["## 구현 메모 (모호했던 곳과 정한 것 — 실행 전 결정)", "",
          f"- **12-1 모멘텀** = `close[t−{MOM_SKIP}] / close[t−{MOM_LOOKBACK}] − 1`, 분할조정 종가(`read_prices`), 행 오프셋 = "
          "패널 거래일. t 종가는 쓰지 않는다. 정지일 이월 종가는 그대로(그날 아는 마지막 가격).",
          f"- **모멘텀 10분위:** 그날 **유니버스** 종목 중 모멘텀이 유효한 종목을 시장별로 순위(`method='first'`) → 등분 "
          f"(`day_quartiles` 와 같은 식, n_q {MOM_N_Q}). 유효 종목 < 10 인 날·시장은 NaN → 그 트레이드는 짝 없음.",
          "- **ADV 2분위:** 20일 평균 거래대금(`average_trade_value(…, 20)`, 기존 ADV20 과 같은 패널), 그날 유니버스 종목의 "
          "**시장별 중앙값 이상 = 1 / 미만 = 0**(동점은 이상 쪽). 보조 대조도 같은 자(유니버스 단면)로 자른다 — "
          "주도주 풀 안에서 다시 자르지 않는다(트레이드와 대조가 같은 자).",
          "- **풀:** 기존과 같은 대조 풀 객체(`Pools` — 그날 유니버스 전 종목을 칸의 진입·청산 규칙으로 `sepa.control_trades`). "
          "모멘텀·보조는 그 풀에 층 열만 붙여 다시 뽑는다(대조 수익은 기존 대조와 같은 시뮬).",
          f"- **보조 풀:** 그 풀 중 그날 추세 템플릿 ∧ RS ≥ {RS_MIN_AUX:.0f} ∧ 유니버스 종목(`skeleton_features` 의 `tt`·`rs`). "
          "트레이드 쪽은 거르지 않는다(pead_new·trend 트레이드도 그대로 주도주와 짝).",
          f"- **추출:** core `matched_control` — 층 후보(자기 종목 제외)가 1개 이상이면 {N_PER}개를 **복원 추출**한다. "
          f"후보가 {N_PER} 미만인 '얇은 층'에서는 같은 대조가 반복해 뽑힌다(짝 없음이 아니다). 후보 0 또는 층 키 NaN 일 때만 "
          "짝 없음. 얇은 층 트레이드 수는 아래 표에 따로 적는다.",
          "- **전반/후반·연도별:** 사건일 기준, 이미 뽑힌 짝 그대로(`sub_alpha`).",
          "- **바 3 실적 문 '3종 중앙값':** 주 = 세 값(E·pead≥0.70·pead≥0.90) 각각의 칸 중앙값들의 중앙값. "
          "참고로 세 값 84칸을 한데 모은 중앙값도 적는다. 실적 문 축은 pead_new 14칸을 뺀 112칸.",
          "- **차이** = 기존 alpha − 모멘텀 alpha(모멘텀이 설명한 몫). 짝 없음 비율 = 짝 없는 트레이드 / 칸 트레이드 n.",
          ""]

    L += ["## 합격 바 숫자 (사전등록 1~3 — 숫자만, 판정 아님)", "",
          "| 바 | 사전등록 정의 | 값 |", "|---|---|---|",
          f"| (1) | 방향 가족 12칸 모멘텀 alpha 중앙값 > 0, 전반·후반 각각 | 전 구간 **{_pct(b1['all'])}** · "
          f"{HALVES[0][0]} **{_pct(b1['h1'])}** · {HALVES[1][0]} **{_pct(b1['h2'])}** |",
          f"| (2) | 12칸 중 {BAR2_MIN}칸 이상 전 구간 모멘텀 alpha CI 하단 > 0 | **{len(b2_cells)} / {len(F)}** — "
          + (", ".join(f"`{x}`" for x in b2_cells) if b2_cells else "없음") + " |",
          f"| (3a) | 축 주효과(모멘텀): 돌파 > 추세 | 돌파 **{_pct(ax['mom_']['entry']['breakout'])}** vs 추세 "
          f"**{_pct(ax['mom_']['entry']['trend'])}** (칸 중앙값, 각 56칸) |",
          f"| (3b) | 축 주효과(모멘텀): 실적 문(3종 중앙값) > 끔 | 3종 중앙값 **{_pct(ax['mom_']['gate_3med'])}** "
          f"(E {_pct(ax['mom_']['gate']['E'])} · pead≥0.70 {_pct(ax['mom_']['gate']['pead≥0.70'])} · pead≥0.90 "
          f"{_pct(ax['mom_']['gate']['pead≥0.90'])}) vs 끔 **{_pct(ax['mom_']['gate']['off'])}** · 참고: 84칸 모음 중앙값 "
          f"{_pct(ax['mom_']['gate_pooled3'])} |",
          ""]

    # --- 방향 가족 표 ---
    head = ("| 칸 | n | 기존 alpha [CI] | 모멘텀 alpha [CI] | 보조 alpha [CI] | 차이(기존−모멘텀) | 모멘텀 전반 α | "
            "모멘텀 후반 α | 짝 없음 기존 · 모멘텀 · 보조 | 얇은 층(후보<5) 모멘텀 · 보조 |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    def row_md(r: dict) -> str:
        if not r.get("n"):
            return f"| {r['cell']} | 0 | | | | | | | | |"
        n = int(r["n"])
        return (f"| {r['cell']} | {n} | {_a(r, 'old_')} | {_a(r, 'mom_')} | {_a(r, 'aux_')} | {_pct(r['diff_old_mom'])} | "
                f"{_pct(r['mom_h1_alpha'])} | {_pct(r['mom_h2_alpha'])} | {_rate(r['old_unmatched'], n)} · "
                f"{_rate(r['mom_unmatched'], n)} · {_rate(r['aux_unmatched'], n)} | {_rate(r['mom_thin'], n)} · "
                f"{_rate(r['aux_thin'], n)} |")

    L += ["## 방향 가족 12칸 (판정 대상)", "",
          "alpha 는 비용 전 총수익 기준, CI 는 날짜 군집 부트스트랩 95%.", "", head, sep]
    L += [row_md(r) for r in F.to_dict("records")]
    L += ["", f"가족 12칸 중앙값 — 기존 {_pct(_med(F['old_alpha']))} · 모멘텀 {_pct(_med(F['mom_alpha']))} · 보조 "
          f"{_pct(_med(F['aux_alpha']))} · 차이 {_pct(_med(F['diff_old_mom']))}. 보조 CI 하단 > 0 칸 "
          f"{int((F['aux_lo'].astype(float) > 0).sum())} / {len(F)}.", ""]

    # --- 축 주효과 ---
    L += ["## 축 주효과 (칸들의 alpha 중앙값 — 기존 · 모멘텀 · 보조)", "",
          "| 축 | 값 | 칸 수 | 기존 | **모멘텀** | 보조 | 모멘텀 전반 | 모멘텀 후반 | 모멘텀 alpha > 0 칸 | 모멘텀 CI 하단 > 0 칸 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"entry": "진입", "gate": "실적 문(112칸)", "exit": "청산", "regime": "국면"}
    for axn, vals in (("entry", [*ENTRIES, PEAD_NEW]), ("gate", list(GATES)), ("exit", list(EXITS)),
                      ("regime", list(REGIMES))):
        sub = df if axn != "gate" else df[df["entry"] != PEAD_NEW]
        for v in vals:
            g = sub[sub[axn] == v]
            ma = g["mom_alpha"].astype(float).dropna()
            L.append(f"| {names[axn]} | {v} | {len(g)} | {_pct(ax['old_'][axn][v])} | **{_pct(ax['mom_'][axn][v])}** | "
                     f"{_pct(ax['aux_'][axn][v])} | {_pct(ax_h['h1_'][axn][v])} | {_pct(ax_h['h2_'][axn][v])} | "
                     f"{int((ma > 0).sum())}/{len(ma)} | {int((g['mom_lo'].astype(float) > 0).sum())}/{len(g)} |")
    L += ["", f"실적 문 3종 중앙값(세 값 중앙값들의 중앙값) — 기존 {_pct(ax['old_']['gate_3med'])} · 모멘텀 "
          f"{_pct(ax['mom_']['gate_3med'])} · 보조 {_pct(ax['aux_']['gate_3med'])} · 모멘텀 전반 "
          f"{_pct(ax_h['h1_']['gate_3med'])} · 후반 {_pct(ax_h['h2_']['gate_3med'])}.", ""]

    # --- 연도별 ---
    L += ["## 방향 가족 12칸 연도별 모멘텀 alpha (사건 연도, 비용 전 · 괄호 = 트레이드 n)", ""]
    if len(yr):
        L += ["| 칸 | " + " | ".join(str(y) for y in years) + " | 양수 연도 |",
              "|---|" + "---|" * len(years) + "---|"]
        for cid in fam:
            g = yr[yr["cell"] == cid].set_index("year")
            cells_y, pos, nn = [], 0, 0
            for y in years:
                if y not in g.index or not g.loc[y, "n"]:
                    cells_y.append("—")
                    continue
                a = g.loc[y, "alpha"]
                if np.isfinite(a):
                    nn += 1
                    pos += int(a > 0)
                cells_y.append(f"{_pct(a)} ({int(g.loc[y, 'n'])})")
            L.append(f"| {cid} | " + " | ".join(cells_y) + f" | {pos}/{nn} |")
        med_y = yr.groupby("year")["alpha"].median()
        L.append("| (12칸 중앙값) | " + " | ".join(_pct(med_y.get(y, np.nan)) for y in years) + " | |")
    L.append("")

    # --- 전체 126칸 ---
    L += ["## 전체 126칸 (축 순서 — 성과순 아님)", "", head, sep]
    L += [row_md(r) for r in df.to_dict("records")]
    L.append("")

    tot = df["n"].sum()
    L += ["## 짝 없음·얇은 층 (126칸 합)", "",
          f"- 짝 없음: 기존 {int(df['old_unmatched'].sum()):,} ({_rate(df['old_unmatched'].sum(), tot)}) · 모멘텀 "
          f"{int(df['mom_unmatched'].sum()):,} ({_rate(df['mom_unmatched'].sum(), tot)}) · 보조 "
          f"{int(df['aux_unmatched'].sum()):,} ({_rate(df['aux_unmatched'].sum(), tot)}) / 트레이드 {int(tot):,}",
          f"- 그중 층 키 NaN(모멘텀 분위·ADV 2분위 없음): 모멘텀 {int(df['mom_nan_strata'].sum()):,} · 보조 "
          f"{int(df['aux_nan_strata'].sum()):,}",
          f"- 얇은 층(후보 1~{N_PER - 1}, 복원 추출로 반복): 모멘텀 {int(df['mom_thin'].sum()):,} "
          f"({_rate(df['mom_thin'].sum(), tot)}) · 보조 {int(df['aux_thin'].sum()):,} ({_rate(df['aux_thin'].sum(), tot)})",
          f"- 칸별 후보 수 중앙값의 중앙값: 모멘텀 {np.nanmedian(df['mom_cand_median'].astype(float)):.0f} · 보조 "
          f"{np.nanmedian(df['aux_cand_median'].astype(float)):.0f}",
          f"- 방향 가족 12칸 짝 없음: 기존 {_rate(F['old_unmatched'].sum(), F['n'].sum())} · 모멘텀 "
          f"{_rate(F['mom_unmatched'].sum(), F['n'].sum())} · 보조 {_rate(F['aux_unmatched'].sum(), F['n'].sum())} · 얇은 층 "
          f"모멘텀 {_rate(F['mom_thin'].sum(), F['n'].sum())} · 보조 {_rate(F['aux_thin'].sum(), F['n'].sum())}",
          ""]

    L += ["## 실행 메타", "",
          f"- 런타임 {elapsed / 60:.1f}분 · 칸 {len(df)} · 대조 풀 규칙 {len(pools.cache)}벌 "
          f"({', '.join(f'{k}: {len(v):,}' for k, v in pools.cache.items())})",
          f"- 산출물: `{out_dir}/cells_momentum.csv`(칸별 세 대조 숫자), `{out_dir}/MOMENTUM_RESULTS.md`",
          "- 재현: `uv run --extra pg python research/experiments/sepa_momentum.py`", ""]

    df.to_csv(os.path.join(out_dir, "cells_momentum.csv"), index=False)
    with open(os.path.join(out_dir, "MOMENTUM_RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
