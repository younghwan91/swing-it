"""실적 as-of(정정 G19-1 · 프로젝트 기억 earnings-asof-rule): EPS 값은 max(avail_date, 정기보고서 rcept_dt) **다음 거래일**부터 쓴다.

``eps_ok(codes, days, ci, s, con)`` → 사건 (종목 ci, 신호 행 s) 마다 그 시점 최신 분기 순이익 > 전년 같은 분기 ∧ > 0(사용자 고정 규칙).
earnings.avail_date 는 제출 마감일 기반이라 정기보고서의 18% 가 그보다 늦게 나왔다(미래 정보) — 이 함수가 그것을 막는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def known_table(con, days: pd.DatetimeIndex) -> pd.DataFrame:
    earn = pd.read_sql("select code, period, avail_date, netinc, netinc_prior from earnings", con, parse_dates=["avail_date"])
    per = pd.read_sql("select code, period, min(rcept_dt) as per_dt from earnings_disclosures where kind='periodic' group by code, period",
                      con, parse_dates=["per_dt"])
    e = earn.merge(per, on=["code", "period"], how="left")
    e["known"] = e[["avail_date", "per_dt"]].max(axis=1)
    e["krow"] = days.searchsorted(e["known"].to_numpy(), side="right")          # 알게 된 날 다음 거래일 행
    return e.sort_values("krow")


def eps_ok(codes, days: pd.DatetimeIndex, ci: np.ndarray, s: np.ndarray, con) -> np.ndarray:
    kt = known_table(con, days)
    ev = pd.DataFrame({"code": [str(codes[i]) for i in ci], "s": np.asarray(s, np.int64), "k": np.arange(len(ci))}).sort_values("s")
    m = pd.merge_asof(ev, kt[["code", "krow", "netinc", "netinc_prior"]], left_on="s", right_on="krow", by="code", direction="backward")
    ok = ((m["netinc"] > 0) & (m["netinc"] > m["netinc_prior"])).to_numpy()
    out = np.zeros(len(ci), bool)
    out[m["k"].to_numpy()] = ok
    return out


def eps_yoy(codes, days: pd.DatetimeIndex, ci: np.ndarray, s: np.ndarray, con) -> np.ndarray:
    """22세대: 그 시점 최신 분기 순이익 YoY 증가율(전년 같은 분기 ≤ 0 이면 NaN) — 같은 as-of 규칙."""
    kt = known_table(con, days)
    ev = pd.DataFrame({"code": [str(codes[i]) for i in ci], "s": np.asarray(s, np.int64), "k": np.arange(len(ci))}).sort_values("s")
    m = pd.merge_asof(ev, kt[["code", "krow", "netinc", "netinc_prior"]], left_on="s", right_on="krow", by="code", direction="backward")
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(m["netinc_prior"] > 0, m["netinc"] / m["netinc_prior"] - 1.0, np.nan)
    out = np.full(len(ci), np.nan)
    out[m["k"].to_numpy()] = g
    return out
