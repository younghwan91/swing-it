#!/usr/bin/env python
"""LLM 뉴스 지속성 × 다일 드리프트 — 표본 외 재현 러너 (스윙 2호).

사전등록: ``research/logs/news_persistence_swing/VERDICT.md`` (2026-10-08, 결과 보기 전 커밋).

입력(판정은 daytrade-it 브랜치 ``research/news-persistence-judge`` 의 ``judge_batch.py`` 가 만든다):
  - 쌍 파일   ``PERSIST_PAIRS`` (기본 ~/git/daytrade-it/data/eval/persistence_swing/pairs_2023-03_2026-02.jsonl)
  - 판정 캐시 ``PERSIST_CACHE`` (기본 ~/git/daytrade-it/data/eval/cache/haiku55/persistence_v2.jsonl)

사건 = subject main · freshness≠repeat · sentiment +1 · reports_price_move false, (종목, d) 당 최고 persistence.
수익 = d 다음 거래일 시가 → h 거래일째 종가(손절 없음). 대조 = core matched_null(같은 d × 시장 × 전일 거래대금
4분위 × 5일 수익률 4분위, 풀 = 그날 상위 300 유니버스, 5배). 1차 지표 = h=20 structural − one_off 의 정합
초과수익 차이, 월 블록 부트스트랩.

결과는 같은 폴더 ``RUN.md`` — 판정은 사람이 VERDICT 에 적는다.

Run:  uv run --extra pg python research/experiments/news_persistence_swing.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from krx_quant_core.stats.matched_null import matched_control  # noqa: E402
from krx_quant_core.stats.trials import record_trial  # noqa: E402
from prop_swing_common import load_env_db  # noqa: E402

from swing_it.storage import connect, db_default, read_prices  # noqa: E402

OUT_DIR = Path("research/logs/news_persistence_swing")
LABEL = "news_persistence_swing"
DT = Path.home() / "git" / "daytrade-it" / "data" / "eval"
PAIRS = Path(os.environ.get("PERSIST_PAIRS", DT / "persistence_swing" / "pairs_2023-03_2026-02.jsonl"))
CACHE = Path(os.environ.get("PERSIST_CACHE", DT / "cache" / "haiku55" / "persistence_v2.jsonl"))

HORIZONS = (5, 10, 20, 40)
PRIMARY_H = 20
COST_RT = 0.00415
PERSIST_RANK = {"one_off": 0, "multi_quarter": 1, "structural": 2}
CLASSES = ("one_off", "multi_quarter", "structural")
UNIVERSE_N = 300
HALVES = (("2023-03-01", "2024-08-31"), ("2024-09-01", "2026-02-28"))

PREREG = {"subject": "main", "drop_repeat": True, "sentiment": 1, "drop_price_move": True,
          "entry": "next_open", "horizons": list(HORIZONS), "primary_h": PRIMARY_H,
          "strata": ["date", "market", "q_adv", "q_ret5"], "n_per": 5, "universe": UNIVERSE_N}


def load_events(cfg: dict) -> pd.DataFrame:
    meta = {}
    with PAIRS.open() as fh:
        for line in fh:
            p = json.loads(line)
            meta[p["key"]] = (p["code"], p["day"])
    rows = []
    with CACHE.open() as fh:
        for line in fh:
            r = json.loads(line)
            m = meta.get(r["key"])
            v = r.get("v2") or {}
            if m is None or not v:
                continue
            if v.get("subject") != cfg["subject"] or v.get("sentiment_direction") != cfg["sentiment"]:
                continue
            if cfg["drop_repeat"] and v.get("freshness") == "repeat":
                continue
            if cfg["drop_price_move"] and v.get("reports_price_move"):
                continue
            if v.get("persistence") not in PERSIST_RANK:
                continue
            rows.append({"code": m[0], "day": pd.Timestamp(m[1]), "rank": PERSIST_RANK[v["persistence"]]})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev = ev.groupby(["code", "day"], as_index=False)["rank"].max()
    ev["persistence"] = ev["rank"].map({v: k for k, v in PERSIST_RANK.items()})
    return ev


def n_judged() -> int:
    with CACHE.open() as fh:
        return sum(1 for _ in fh)


class Px:
    def __init__(self, con):
        pr = read_prices(con, cols=("code", "date", "open", "close", "trade_value"))
        pr["date"] = pd.to_datetime(pr["date"])
        pr = pr[pr["date"] >= pd.Timestamp("2022-12-01")]
        self.open = pr.pivot(index="date", columns="code", values="open").sort_index().astype(float)
        self.close = pr.pivot(index="date", columns="code", values="close").reindex_like(self.open).astype(float)
        self.tv = pr.pivot(index="date", columns="code", values="trade_value").reindex_like(self.open).astype(float)
        self.days = self.open.index
        self.codes = self.open.columns
        self.O, self.C, self.TV = self.open.to_numpy(), self.close.to_numpy(), self.tv.to_numpy()
        self.ret5 = (self.close / self.close.shift(5) - 1).to_numpy()
        mk = pd.concat([pd.read_sql_query("SELECT code, market FROM delisted_stocks", con),
                        pd.read_sql_query("SELECT code, market FROM stocks", con)]).drop_duplicates("code", keep="last")
        m = mk.set_index("code")["market"].map(lambda x: "KOSPI" if x in ("거래소", "KOSPI", "유가증권") else "KOSDAQ")
        self.market = m.reindex(self.codes).fillna("KOSDAQ").to_numpy()

    def fwd(self, di: int, cols: np.ndarray, h: int, entry: str) -> np.ndarray:
        """d 행 기준 진입·청산 수익. 진입가 없음 → NaN, 중간 시세 끊김 → 마지막 종가."""
        if entry == "next_open":
            e = di + 1
            if e >= len(self.days):
                return np.full(len(cols), np.nan)
            px_in = self.O[e, cols]
            last = e + h - 1
        else:  # "close": d 종가 진입, h 거래일 뒤 종가
            e = di
            px_in = self.C[e, cols]
            last = e + h
        if last >= len(self.days):
            return np.full(len(cols), np.nan)  # 보유 끝이 데이터 밖 — 표본에서 뺀다
        window = self.C[e:last + 1, cols]
        filled = pd.DataFrame(window).ffill().to_numpy()[-1]
        ok = np.isfinite(px_in) & (px_in > 0) & np.isfinite(filled)
        out = np.full(len(cols), np.nan)
        out[ok] = filled[ok] / px_in[ok] - 1.0
        return out


def universe_at(px: Px, di: int) -> np.ndarray:
    """d 의 유니버스 = 전 거래일 거래대금 상위 300 (쌍 생성기와 같은 규칙)."""
    tv = px.TV[di - 1]
    ok = np.where(np.isfinite(tv) & (tv > 0))[0]
    return ok[np.argsort(-tv[ok], kind="stable")][:UNIVERSE_N]


def build(px: Px, ev: pd.DataFrame, h: int, entry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades, pools = [], []
    col_of = {c: i for i, c in enumerate(px.codes)}
    for d, g in ev.groupby("day"):
        if d not in px.days:
            continue
        di = px.days.get_loc(d)
        uni = universe_at(px, di)
        if len(uni) < 40:
            continue
        r_u = px.fwd(di, uni, h, entry)
        adv_prev = px.TV[di - 1, uni]
        pool = pd.DataFrame({"date": d, "code": px.codes[uni], "ret": r_u, "market": px.market[uni],
                             "adv": adv_prev, "ret5": px.ret5[di, uni]})
        pool["q_adv"] = pd.qcut(pool["adv"].rank(method="first"), 4, labels=False)
        pool["q_ret5"] = pd.qcut(pool["ret5"].rank(method="first"), 4, labels=False) if pool["ret5"].notna().sum() >= 8 else np.nan
        pools.append(pool.dropna(subset=["ret"]))
        cols = np.array([col_of[c] for c in g["code"] if c in col_of])
        gg = g[g["code"].isin(col_of)]
        r_t = px.fwd(di, cols, h, entry)
        t = gg.assign(date=d, ret=r_t).merge(pool[["code", "market", "q_adv", "q_ret5"]], on="code", how="left")
        trades.append(t)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    pl = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    tr = tr.dropna(subset=["ret", "q_adv", "q_ret5"]).reset_index(drop=True)
    return tr, pl.reset_index(drop=True)


def excess(tr: pd.DataFrame, pl: pd.DataFrame, n_per: int) -> pd.DataFrame:
    ctl = matched_control(tr, pl, strata=["date", "market", "q_adv", "q_ret5"], n_per=n_per, seed=0,
                          exclude_self=False)
    cm = ctl.assign(cret=pl.loc[ctl["control_idx"].to_numpy(), "ret"].to_numpy()).groupby("trade_idx")["cret"].mean()
    out = tr.loc[cm.index].copy()
    out["ctrl"] = cm.to_numpy()
    out["ex"] = out["ret"] - out["ctrl"]
    out["month"] = out["date"].dt.to_period("M")
    return out


def month_boot_diff(x: pd.DataFrame, a: str, b: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """mean(ex | a) − mean(ex | b), 월 블록 부트스트랩 95% CI."""
    months = x["month"].unique()
    by = {m: g for m, g in x.groupby("month")}
    rng = np.random.default_rng(seed)

    def stat(df):
        ea, eb = df.loc[df["persistence"] == a, "ex"], df.loc[df["persistence"] == b, "ex"]
        return ea.mean() - eb.mean() if len(ea) and len(eb) else np.nan

    point = stat(x)
    boots = []
    for _ in range(n):
        pick = rng.choice(months, size=len(months), replace=True)
        boots.append(stat(pd.concat([by[m] for m in pick], ignore_index=True)))
    boots = np.array([v for v in boots if np.isfinite(v)])
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def month_boot_mean(v: pd.Series, months: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    df = pd.DataFrame({"v": v.to_numpy(), "m": months.to_numpy()})
    by = [g["v"].to_numpy() for _, g in df.groupby("m")]
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([by[i] for i in rng.integers(0, len(by), len(by))]).mean() for _ in range(n)]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def bp(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x * 1e4:+.0f}bp"


def run_cfg(px: Px, cfg: dict, horizons=HORIZONS) -> dict:
    ev = load_events(cfg)
    res = {"n_events": len(ev), "by_class": ev["persistence"].value_counts().to_dict() if len(ev) else {}, "h": {}}
    for h in horizons:
        tr, pl = build(px, ev, h, cfg["entry"])
        x = excess(tr, pl, cfg["n_per"])
        cls = {c: (x.loc[x["persistence"] == c, "ex"].mean(), int((x["persistence"] == c).sum())) for c in CLASSES}
        d, lo, hi = month_boot_diff(x, "structural", "one_off")
        halves = []
        for a, b in HALVES:
            xx = x[(x["date"] >= a) & (x["date"] <= b)]
            halves.append(month_boot_diff(xx, "structural", "one_off", n=500)[0] if len(xx) else np.nan)
        s = x.loc[x["persistence"] == "structural"]
        s_lo, s_hi = month_boot_mean(s["ex"], s["month"]) if len(s) > 10 else (np.nan, np.nan)
        res["h"][h] = {"cls": cls, "diff": (d, lo, hi), "halves": halves, "struct_ci": (s_lo, s_hi),
                       "raw_struct": float(s["ret"].mean()) if len(s) else np.nan, "n_months": int(x["month"].nunique())}
    return res


def main() -> int:
    load_env_db()
    con = connect(db_default())
    px = Px(con)
    L = [f"# 실행 결과 — {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         f"- 판정된 쌍 {n_judged():,} / 쌍 파일 {sum(1 for _ in PAIRS.open()):,}", ""]
    record_trial(LABEL, PREREG, logs_dir=str(OUT_DIR))
    r = run_cfg(px, PREREG)
    L += ["## 사전등록 config", "", f"- 사건(종목·일) {r['n_events']:,} · 등급별 {r['by_class']}", "",
          "| h | one_off | multi_quarter | structural | **structural − one_off** [월 블록 95%] | 전반 · 후반 | structural CI | structural − 41.5bp |",
          "|---|---|---|---|---|---|---|---|"]
    for h, v in r["h"].items():
        c = v["cls"]
        d, lo, hi = v["diff"]
        L.append(f"| {h} | {bp(c['one_off'][0])} ({c['one_off'][1]}) | {bp(c['multi_quarter'][0])} ({c['multi_quarter'][1]}) | "
                 f"{bp(c['structural'][0])} ({c['structural'][1]}) | **{bp(d)}** [{bp(lo)}, {bp(hi)}] | "
                 f"{bp(v['halves'][0])} · {bp(v['halves'][1])} | [{bp(v['struct_ci'][0])}, {bp(v['struct_ci'][1])}] | "
                 f"{bp(c['structural'][0] - COST_RT)} |")
    L += ["", "값 = 정합 대조 초과수익 평균(괄호 = 건수). 월 수 " + str(r["h"][PRIMARY_H]["n_months"]), ""]

    L += ["## 민감도 (원장 기록 · 승자선택 금지) — h=20", "", "| 칸 | 사건 | structural − one_off | structural |", "|---|---|---|---|"]
    for name, kw in (("진입 = d 종가", {"entry": "close"}), ("주가 보도 기사 포함", {"drop_price_move": False}),
                     ("repeat 포함", {"drop_repeat": False})):
        cfg = {**PREREG, **kw}
        record_trial(LABEL + "_sens", cfg, logs_dir=str(OUT_DIR))
        rr = run_cfg(px, cfg, horizons=(PRIMARY_H,))
        v = rr["h"][PRIMARY_H]
        d, lo, hi = v["diff"]
        L.append(f"| {name} | {rr['n_events']:,} | {bp(d)} [{bp(lo)}, {bp(hi)}] | {bp(v['cls']['structural'][0])} |")
    (OUT_DIR / "RUN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
