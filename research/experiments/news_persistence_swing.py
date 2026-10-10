#!/usr/bin/env python
"""LLM 뉴스 지속성 × 다일 드리프트 — 표본 외 재현 러너 (스윙 2호).

사전등록: ``research/logs/news_persistence_swing/VERDICT.md`` (2026-10-08, 결과 보기 전 커밋).

입력(판정은 ``swing_it.news.judge`` 가 만든다 — 2026-10-08 daytrade-it 연구 브랜치에서 이식, VERDICT 의 이전 노트):
  - 쌍 파일 ``PERSIST_PAIRS`` (기본 data/eval/persistence_swing/pairs/swing_2023-03_2026-02.jsonl —
    daytrade-it 에서 바이트 그대로 복사한 사본. ``swing_it.news.pairs`` 가 DB 에서 같은 키로 다시 만든다)
  - 판정: quant-airflow DB 판정 원장 ``article_judgments`` 의 judge ``v2r3-haiku55`` (2026-10-09 부터 — 그 전엔
    JSONL 캐시를 읽었다. 1단계 36,171 행에서 키→출력이 파일과 완전히 같음을 확인했다, VERDICT 이전 노트 2).

사건 = subject main · freshness≠repeat · sentiment +1 · reports_price_move false, (종목, d) 당 최고 persistence.
수익 = d 다음 거래일 시가 → h 거래일째 종가(손절 없음). 대조 = core matched_null(같은 d × 시장 × 전일 거래대금
4분위 × 5일 수익률 4분위, 풀 = 그날 상위 300 유니버스, 5배). 1차 지표(사전등록 수정 3) = h=20
(multi_quarter ∪ structural) − one_off 의 정합 초과수익 차이, 월 블록 부트스트랩. 지수 헤지는 사전등록이 쓰지 않는다 —
원수익·유니버스 동일가중 평균 차감은 사전등록 밖 기술 통계로만 함께 적는다.

가격: ``read_prices`` → ``krx_session.patch_prices``(2026-09-14+ NXT 오염 행을 분봉 정규장 값으로). 사건은 2026-02 에서
끝나 h=40 청산도 그 전이라 보정은 결과에 닿지 않는다 — RUN.md 에 실제 최대 청산일을 적는다.

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

from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg  # noqa: E402
from swing_it.news.judgments_db import read_judgments  # noqa: E402
from swing_it.storage import connect, db_default, read_prices  # noqa: E402

OUT_DIR = Path("research/logs/news_persistence_swing")
LABEL = "news_persistence_swing"
DT = Path(HERE).resolve().parents[1] / "data" / "eval"
PAIRS = Path(os.environ.get("PERSIST_PAIRS", DT / "persistence_swing" / "pairs" / "swing_2023-03_2026-02.jsonl"))
#: 판정 원장의 judge 이름(사전등록 수정 1 의 정의: 운영 v2 + r3, Haiku 5.5, 생각 끔).
JUDGE = "v2r3-haiku55"

HORIZONS = (5, 10, 20, 40)
PRIMARY_H = 20
COST_RT = 0.00415
PERSIST_RANK = {"one_off": 0, "multi_quarter": 1, "structural": 2}
CLASSES = ("one_off", "multi_quarter", "structural")
UNIVERSE_N = 300
HALVES = (("2023-03-01", "2024-08-31"), ("2024-09-01", "2026-02-28"))
#: 판정 6구간(사전등록 — 최근부터 과거로 판정한 순서 그대로 1..6).
STAGES = (("2025-09-01", "2026-02-28"), ("2025-03-01", "2025-08-31"), ("2024-09-01", "2025-02-28"),
          ("2024-03-01", "2024-08-31"), ("2023-09-01", "2024-02-29"), ("2023-03-01", "2023-08-31"))

PREREG = {"prompt": "v2+r3", "primary": "hi_minus_lo", "subject": "main", "drop_repeat": True, "sentiment": 1, "drop_price_move": True,
          "entry": "next_open", "horizons": list(HORIZONS), "primary_h": PRIMARY_H,
          "strata": ["date", "market", "q_adv", "q_ret5"], "n_per": 5, "universe": UNIVERSE_N}


def load_judgments(con) -> dict[str, dict]:
    """판정 원장에서 {쌍 키: v2 출력} — 성공 행만."""
    return {k: r["output"] for k, r in read_judgments(con, JUDGE).items()}


JUDGED: dict[str, dict] = {}  # main() 이 원장에서 채운다


def load_events(cfg: dict, judged: dict[str, dict] | None = None) -> pd.DataFrame:
    judged = JUDGED if judged is None else judged
    meta = {}
    with PAIRS.open() as fh:
        for line in fh:
            p = json.loads(line)
            meta[p["key"]] = (p["code"], p["day"])
    rows = []
    for key, v in judged.items():
        m = meta.get(key)
        v = v or {}
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
    return len(JUDGED)


class Px:
    def __init__(self, con):
        pr = read_prices(con, cols=("code", "date", "open", "close", "trade_value"))
        pr["date"] = pd.to_datetime(pr["date"])
        pr = pr[pr["date"] >= pd.Timestamp("2022-12-01")]
        late = int((pr["date"] >= pd.Timestamp(NXT_START)).sum())
        pr = patch_prices(pr, read_session_bars_agg(con, start=NXT_START))
        self.nxt = {"rows": late, "patched": int(pr["krx_patched"].sum())}
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


def build(px: Px, ev: pd.DataFrame, h: int, entry: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """사건 → 트레이드(진입·청산 수익 + 층 열)와 그날 유니버스 풀. 세 번째 반환 = 표본 흐름(빠진 건수)."""
    trades, pools = [], []
    col_of = {c: i for i, c in enumerate(px.codes)}
    flow = {"events": len(ev), "no_price_day": 0, "no_code": 0}
    for d, g in ev.groupby("day"):
        if d not in px.days:
            flow["no_price_day"] += len(g)
            continue
        di = px.days.get_loc(d)
        uni = universe_at(px, di)
        if len(uni) < 40:
            flow["no_price_day"] += len(g)
            continue
        exit_day = px.days[di + h] if di + h < len(px.days) else pd.NaT  # 두 진입 규칙 모두 청산 = d+h 거래일 종가
        r_u = px.fwd(di, uni, h, entry)
        adv_prev = px.TV[di - 1, uni]
        pool = pd.DataFrame({"date": d, "code": px.codes[uni], "ret": r_u, "market": px.market[uni],
                             "adv": adv_prev, "ret5": px.ret5[di, uni]})
        pool["q_adv"] = pd.qcut(pool["adv"].rank(method="first"), 4, labels=False)
        pool["q_ret5"] = pd.qcut(pool["ret5"].rank(method="first"), 4, labels=False) if pool["ret5"].notna().sum() >= 8 else np.nan
        pools.append(pool.dropna(subset=["ret"]))
        gg = g[g["code"].isin(col_of)]
        flow["no_code"] += len(g) - len(gg)
        cols = np.array([col_of[c] for c in gg["code"]], dtype=int)
        r_t = px.fwd(di, cols, h, entry)
        t = gg.assign(date=d, ret=r_t, exit_day=exit_day).merge(
            pool[["code", "market", "q_adv", "q_ret5"]], on="code", how="left")
        trades.append(t)
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(
        columns=["code", "day", "rank", "persistence", "date", "ret", "exit_day", "market", "q_adv", "q_ret5"])
    pl = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    flow["not_in_universe"] = int(tr["q_adv"].isna().sum())
    flow["no_ret5_quartile"] = int((tr["q_adv"].notna() & tr["q_ret5"].isna()).sum())
    flow["no_return"] = int((tr["q_adv"].notna() & tr["q_ret5"].notna() & tr["ret"].isna()).sum())
    tr = tr.dropna(subset=["ret", "q_adv", "q_ret5"]).reset_index(drop=True)
    flow["trades"] = len(tr)
    return tr, pl.reset_index(drop=True), flow


def excess(tr: pd.DataFrame, pl: pd.DataFrame, n_per: int) -> tuple[pd.DataFrame, dict]:
    """정합 대조 초과수익 ``ex`` (사전등록) + 기술 통계용 ``uhedge`` (원수익 − 그날 유니버스 동일가중 평균).

    대조 풀은 사전등록 문언대로 그날 상위 300 **전 종목** — 사건 종목 자신도 포함된다(자기 선택 비율을 함께 반환).
    """
    ctl = matched_control(tr, pl, strata=["date", "market", "q_adv", "q_ret5"], n_per=n_per, seed=0,
                          exclude_self=False)
    cidx = ctl["control_idx"].to_numpy()
    ctl = ctl.assign(cret=pl.loc[cidx, "ret"].to_numpy(),
                     self_pick=pl.loc[cidx, "code"].to_numpy() == tr.loc[ctl["trade_idx"].to_numpy(), "code"].to_numpy())
    cm = ctl.groupby("trade_idx")["cret"].mean()
    out = tr.loc[cm.index].copy()
    out["ctrl"] = cm.to_numpy()
    out["ex"] = out["ret"] - out["ctrl"]
    out["uhedge"] = out["ret"] - out["date"].map(pl.groupby("date")["ret"].mean()).to_numpy()
    out["month"] = out["date"].dt.to_period("M")
    # 1차 지표(사전등록 수정 3): multi_quarter ∪ structural = "hi", one_off = "lo"
    out["grp"] = np.where(out["persistence"] == "one_off", "lo", "hi")
    info = {"unmatched": len(tr) - len(out), "self_share": float(ctl["self_pick"].mean()) if len(ctl) else np.nan}
    return out.reset_index(drop=True), info


def _month_sums(x: pd.DataFrame, mask: pd.Series, val: str, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = x.loc[mask].groupby("month")[val].agg(["sum", "count"]).reindex(months, fill_value=0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def month_boot_diff(x: pd.DataFrame, a: str, b: str, n: int = 2000, seed: int = 0,
                    col: str = "persistence", val: str = "ex") -> tuple[float, float, float]:
    """mean(val | col==a) − mean(val | col==b), 월 블록 부트스트랩 95% CI → (점추정, 하단, 상단).

    달을 복원 추출하고 그 달의 a·b 관측을 통째로 가져간다(보유창이 겹쳐 같은 달 안은 독립이 아니다).
    재추출 표본에 a 나 b 가 없으면 그 회차는 버린다.
    """
    months = np.sort(x["month"].unique()) if len(x) else np.array([])
    k = len(months)
    if k == 0:
        return np.nan, np.nan, np.nan
    sa, na = _month_sums(x, x[col] == a, val, months)
    sb, nb = _month_sums(x, x[col] == b, val, months)

    def stat(idx: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return sa[idx].sum(-1) / na[idx].sum(-1) - sb[idx].sum(-1) / nb[idx].sum(-1)

    point = float(stat(np.arange(k)))
    bs = stat(np.random.default_rng(seed).integers(0, k, size=(n, k)))
    bs = bs[np.isfinite(bs)]
    if len(bs) == 0:
        return point, np.nan, np.nan
    return point, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def month_boot_mean(v: pd.Series, months: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """평균의 월 블록 부트스트랩 95% CI → (하단, 상단)."""
    df = pd.DataFrame({"v": v.to_numpy(float), "month": months.to_numpy()})
    if df.empty:
        return np.nan, np.nan
    ms = np.sort(df["month"].unique())
    s, c = _month_sums(df, pd.Series(True, index=df.index), "v", ms)
    draw = np.random.default_rng(seed).integers(0, len(ms), size=(n, len(ms)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = s[draw].sum(1) / c[draw].sum(1)
    bs = bs[np.isfinite(bs)]
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def ordering_holds(one_off: float, multi_quarter: float, structural: float) -> bool:
    """사전등록 바 2 의 한 보유기간: one_off < multi_quarter < structural (점추정, 엄격 부등호). NaN 이면 False."""
    v = np.array([one_off, multi_quarter, structural], dtype=float)
    return bool(np.all(np.isfinite(v)) and v[0] < v[1] < v[2])


def month_signs(x: pd.DataFrame, val: str = "ex") -> tuple[int, int]:
    """달마다 mean(hi) − mean(lo) 의 부호 → (양수인 달 수, hi·lo 가 둘 다 있는 달 수)."""
    if x.empty:
        return 0, 0
    m = x.groupby(["month", "grp"])[val].mean().unstack()
    if "hi" not in m or "lo" not in m:
        return 0, 0
    d = (m["hi"] - m["lo"]).dropna()
    return int((d > 0).sum()), len(d)


def in_range(x: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return x[(x["date"] >= pd.Timestamp(a)) & (x["date"] <= pd.Timestamp(b))]


def bar_numbers(res: dict) -> dict:
    """사전등록 합격 바 1~4 의 숫자와 산술 비교(판정 아님 — 판정은 사람이 VERDICT 에 적는다)."""
    v = res["h"][PRIMARY_H]
    d, lo, hi = v["diff"]
    order = {h: vv["ordered"] for h, vv in res["h"].items()}
    k = sum(order.values())
    a, b = v["halves"]
    net = v["hi_mean"] - COST_RT
    return {
        1: {"value": (d, lo, hi), "met": bool(np.isfinite(lo) and lo > 0)},
        2: {"value": (k, len(order), order), "met": k >= 3},
        3: {"value": (a, b), "met": bool(np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0)},
        4: {"value": (net, v["hi_ci"][0] - COST_RT, v["hi_ci"][1] - COST_RT), "met": bool(np.isfinite(net) and net > 0)},
    }


def bp(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 1e4:+.0f}bp"


def ci(t: tuple) -> str:
    return f"{bp(t[0])} [{bp(t[1])}, {bp(t[2])}]"


def yn(b: bool) -> str:
    return "예" if b else "아니오"


def run_cfg(px: Px, cfg: dict, horizons=HORIZONS, detail: bool = True) -> dict:
    ev = load_events(cfg)
    res = {"n_events": len(ev), "by_class": ev["persistence"].value_counts().to_dict() if len(ev) else {}, "h": {}}
    for h in horizons:
        tr, pl, flow = build(px, ev, h, cfg["entry"])
        x, info = excess(tr, pl, cfg["n_per"])
        cls = {c: (x.loc[x["persistence"] == c, "ex"].mean(), int((x["persistence"] == c).sum())) for c in CLASSES}
        hi_x = x.loc[x["grp"] == "hi"]
        v = {"cls": cls, "diff": month_boot_diff(x, "hi", "lo", col="grp"),
             "ordered": ordering_holds(*(cls[c][0] for c in CLASSES)),
             "halves": [month_boot_diff(in_range(x, a, b), "hi", "lo", n=1, col="grp")[0] for a, b in HALVES],
             "hi_mean": float(hi_x["ex"].mean()) if len(hi_x) else np.nan,
             "hi_ci": month_boot_mean(hi_x["ex"], hi_x["month"]),
             "n_months": int(x["month"].nunique()), "flow": {**flow, **info},
             "max_exit": x["exit_day"].max() if len(x) else pd.NaT}
        if detail:
            v["diff_struct"] = month_boot_diff(x, "structural", "one_off")
            v["diff_mq"] = month_boot_diff(x, "multi_quarter", "one_off")
            v["cls_ci"] = {c: month_boot_mean(x.loc[x["persistence"] == c, "ex"], x.loc[x["persistence"] == c, "month"])
                           for c in CLASSES}
            v["raw"] = {c: float(x.loc[x["persistence"] == c, "ret"].mean()) for c in CLASSES}
            v["uh"] = {c: float(x.loc[x["persistence"] == c, "uhedge"].mean()) for c in CLASSES}
            v["diff_raw"] = month_boot_diff(x, "hi", "lo", col="grp", val="ret")
            v["diff_uh"] = month_boot_diff(x, "hi", "lo", col="grp", val="uhedge")
            v["signs"] = month_signs(x)
            v["half_n"] = [(int((in_range(x, a, b)["grp"] == "hi").sum()), int((in_range(x, a, b)["grp"] == "lo").sum()))
                           for a, b in HALVES]
            st = []
            for a, b in STAGES:
                xx = in_range(x, a, b)
                lo_, hi_ = xx.loc[xx["grp"] == "lo", "ex"], xx.loc[xx["grp"] == "hi", "ex"]
                st.append({"range": (a, b), "n_lo": len(lo_), "n_hi": len(hi_), "lo": lo_.mean(), "hi": hi_.mean(),
                           "diff": month_boot_diff(xx, "hi", "lo", n=1, col="grp")[0], "signs": month_signs(xx),
                           "cls": {c: xx.loc[xx["persistence"] == c, "ex"].mean() for c in CLASSES}})
            v["stages"] = st
        res["h"][h] = v
    return res


NOTES = [
    "보유: 진입 = d 다음 거래일 시가, 청산 = d 로부터 h 번째 거래일 종가(진입일을 1일째로 셈). 민감도 (a) 는 d 종가 진입 · 같은 날 청산.",
    "직전 5일 수익률 층 = d 종가 / d−5 거래일 종가 − 1 (d 당일 반응 포함 — 진입이 d+1 시가라 d 종가는 진입 전에 알려져 있다).",
    "전일 거래대금 층 = d−1 거래일 거래대금, 4분위는 그날 상위 300 유니버스 안(시장 구분 없이)에서 자른다. 시장 = 상장·폐지 마스터, 미상은 KOSDAQ.",
    "대조 풀 = 사전등록 문언대로 그날 유니버스 **전 종목**: 사건 종목 자신과 다른 사건 종목도 대조로 뽑힐 수 있다(자기 선택 비율은 표본 흐름 표). 복원 추출 5개, seed 0.",
    "지수 헤지는 사전등록이 쓰지 않는다(2024-01 이전 지수 테이블 없음). '유니버스 평균 차감' = 원수익 − 그날 상위 300 동일가중 평균 — 사전등록 밖 기술 통계.",
    "월 = 사건일 d 의 달. 월 블록 부트스트랩 2000회, seed 0, 퍼센타일 95%. 전반·후반·구간 값은 점추정만(사전등록 바 3 이 점추정).",
    "판정 원장 오류 행은 제외(성공 행만). 사건은 (종목, d) 당 최고 persistence 하나 — 등급별 사건 수는 가격 필터 전, 표의 건수는 가격·층 필터 후.",
    "진입가 없음(정지 등)·보유창이 데이터 밖이면 제외, 보유 중 시세 끊김(폐지 포함)은 마지막 종가로 청산.",
    "바 2 는 엄격 부등호 점추정(같은 값은 불성립). 바 4 의 CI 는 사전등록대로 보고만.",
]


def main() -> int:
    load_env_db()
    con = connect(db_default())
    JUDGED.update(load_judgments(con))
    n_err = sum(1 for r in read_judgments(con, JUDGE, include_errors=True).values() if r["output"] is None)
    px = Px(con)
    L = [f"# 실행 결과 — {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         f"- 판정 원장 judge `{JUDGE}`: 성공 {n_judged():,} · 오류 {n_err:,} / 쌍 파일 {sum(1 for _ in PAIRS.open()):,}",
         f"- 가격: read_prices → krx_session.patch_prices — {NXT_START} 이후 {px.nxt['rows']:,}행 중 {px.nxt['patched']:,}행 보정", ""]
    record_trial(LABEL, PREREG, logs_dir=str(OUT_DIR))
    r = run_cfg(px, PREREG)
    max_exit = max(v["max_exit"] for v in r["h"].values())
    L[-1:] = [f"- 사전등록 config 의 최대 청산일 {max_exit.date()} (< {NXT_START}: {yn(max_exit < pd.Timestamp(NXT_START))})", ""]

    b = bar_numbers(r)
    o = b[2]["value"][2]
    L += ["## 사전등록 합격 바 — 숫자 (산술 비교만, 판정 아님)", "",
          "| 바 | 사전등록 기준 | 값 | 기준 산술 |", "|---|---|---|---|",
          f"| 1 | h=20 (multi∪struct) − one_off 월 블록 95% CI 하단 > 0 | {ci(b[1]['value'])} | 하단 > 0: {yn(b[1]['met'])} |",
          f"| 2 | one_off < multi_quarter < structural 이 4개 보유기간 중 ≥ 3 (점추정) | {b[2]['value'][0]}/{b[2]['value'][1]} ("
          + ", ".join(f"h={h} {yn(v)}" for h, v in o.items()) + f") | ≥ 3: {yn(b[2]['met'])} |",
          f"| 3 | 전반(2023-03~2024-08)·후반(2024-09~2026-02) h=20 차이 점추정 둘 다 > 0 | {bp(b[3]['value'][0])} · {bp(b[3]['value'][1])} | 둘 다 > 0: {yn(b[3]['met'])} |",
          f"| 4 | (multi∪struct) h=20 alpha − 41.5bp > 0 (CI 보고만) | {ci(b[4]['value'])} | 점추정 > 0: {yn(b[4]['met'])} |", ""]

    L += ["## 사전등록 config — 정합 대조 초과수익", "", f"- 사건(종목·일) {r['n_events']:,} · 등급별 {r['by_class']}", "",
          "| h | one_off | multi_quarter | structural | 순서 | **(multi∪struct) − one_off** [월 블록 95%] | multi − one_off | structural − one_off | 전반 · 후반 | (multi∪struct) [95%] | (multi∪struct) − 41.5bp |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h, v in r["h"].items():
        c = v["cls"]
        L.append(f"| {h} | {bp(c['one_off'][0])} ({c['one_off'][1]}) | {bp(c['multi_quarter'][0])} ({c['multi_quarter'][1]}) | "
                 f"{bp(c['structural'][0])} ({c['structural'][1]}) | {yn(v['ordered'])} | **{ci(v['diff'])}** | {ci(v['diff_mq'])} | "
                 f"{ci(v['diff_struct'])} | {bp(v['halves'][0])} · {bp(v['halves'][1])} | "
                 f"{bp(v['hi_mean'])} [{bp(v['hi_ci'][0])}, {bp(v['hi_ci'][1])}] | {bp(v['hi_mean'] - COST_RT)} |")
    L += ["", f"값 = 정합 대조 초과수익 평균(괄호 = 건수). 월 수 {r['h'][PRIMARY_H]['n_months']}. 전반·후반 건수(hi, lo) h=20: "
          f"{r['h'][PRIMARY_H]['half_n']}", "",
          "### 등급별 정합 초과수익 월 블록 95% CI", "", "| h | one_off | multi_quarter | structural |", "|---|---|---|---|"]
    for h, v in r["h"].items():
        L.append(f"| {h} | " + " | ".join(f"{bp(v['cls'][c][0])} [{bp(v['cls_ci'][c][0])}, {bp(v['cls_ci'][c][1])}]"
                                         for c in CLASSES) + " |")

    L += ["", "## 원수익 · 유니버스 평균 차감 (사전등록 밖 기술 통계)", "",
          "| h | 원수익 one_off / multi / struct | (multi∪struct) − one_off 원수익 [95%] | 유니버스 차감 one_off / multi / struct | (multi∪struct) − one_off 유니버스 차감 [95%] |",
          "|---|---|---|---|---|"]
    for h, v in r["h"].items():
        L.append(f"| {h} | " + " / ".join(bp(v["raw"][c]) for c in CLASSES) + f" | {ci(v['diff_raw'])} | "
                 + " / ".join(bp(v["uh"][c]) for c in CLASSES) + f" | {ci(v['diff_uh'])} |")

    L += ["", "## 구간별 (판정 6단계, 정합 초과수익 점추정 — 기술 통계)", ""]
    for h in (PRIMARY_H, *[hh for hh in HORIZONS if hh != PRIMARY_H]):
        L += [f"### h={h}", "", "| 단계 | 기간 | one_off (n) | multi_quarter | structural | (multi∪struct) (n) | 차이 | 월 부호 + / 전체 |",
              "|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate(r["h"][h]["stages"], 1):
            L.append(f"| {i} | {s['range'][0][:7]}~{s['range'][1][:7]} | {bp(s['lo'])} ({s['n_lo']}) | {bp(s['cls']['multi_quarter'])} | "
                     f"{bp(s['cls']['structural'])} | {bp(s['hi'])} ({s['n_hi']}) | {bp(s['diff'])} | {s['signs'][0]}/{s['signs'][1]} |")
        L.append("")

    L += ["## 월별 부호 — (multi∪struct) − one_off 정합 초과수익 > 0 인 달", "",
          "| h | + 달 / 전체 |", "|---|---|"]
    for h, v in r["h"].items():
        L.append(f"| {h} | {v['signs'][0]}/{v['signs'][1]} |")

    L += ["", "## 표본 흐름", "",
          "| h | 사건 | 가격일 없음 | 코드 없음 | 유니버스 밖 | 5일 수익률 분위 없음 | 수익 없음 | 트레이드 | 대조 미매칭 | 대조 자기 선택 비율 | 최대 청산일 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h, v in r["h"].items():
        f = v["flow"]
        L.append(f"| {h} | {f['events']:,} | {f['no_price_day']} | {f['no_code']} | {f['not_in_universe']} | {f['no_ret5_quartile']} | "
                 f"{f['no_return']} | {f['trades']:,} | {f['unmatched']} | {f['self_share']:.1%} | {v['max_exit'].date()} |")

    L += ["", "## 민감도 (원장 기록 · 승자선택 금지) — h=20", "",
          "| 칸 | 사건 | 트레이드 | one_off | multi_quarter | structural | (multi∪struct) − one_off [95%] | (multi∪struct) [95%] |",
          "|---|---|---|---|---|---|---|---|"]
    for name, kw in (("(a) 진입 = d 종가", {"entry": "close"}), ("(b) 주가 보도 기사 포함", {"drop_price_move": False}),
                     ("(c) repeat 포함", {"drop_repeat": False})):
        cfg = {**PREREG, **kw}
        record_trial(LABEL + "_sens", cfg, logs_dir=str(OUT_DIR))
        rr = run_cfg(px, cfg, horizons=(PRIMARY_H,), detail=False)
        v = rr["h"][PRIMARY_H]
        L.append(f"| {name} | {rr['n_events']:,} | {v['flow']['trades']:,} | "
                 + " | ".join(f"{bp(v['cls'][c][0])} ({v['cls'][c][1]})" for c in CLASSES)
                 + f" | {ci(v['diff'])} | {bp(v['hi_mean'])} [{bp(v['hi_ci'][0])}, {bp(v['hi_ci'][1])}] |")

    L += ["", "## 구현 메모 (사전등록이 정하지 않은 세부 · 러너가 고른 것)", ""] + [f"- {n}" for n in NOTES]
    (OUT_DIR / "RUN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
