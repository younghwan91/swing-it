"""실적 as-of(정정 G19-1 · 프로젝트 기억 earnings-asof-rule): EPS 값은 max(avail_date, 정기보고서 rcept_dt) **다음 거래일**부터 쓴다.

``eps_ok(codes, days, ci, s, con)`` → 사건 (종목 ci, 신호 행 s) 마다 그 시점 최신 분기 순이익 > 전년 같은 분기 ∧ > 0(사용자 고정 규칙).
earnings.avail_date 는 제출 마감일 기반이라 정기보고서의 18% 가 그보다 늦게 나왔다(미래 정보) — 이 함수가 그것을 막는다.

2026-10-12 ``research/signals/eps_asof.py`` 에서 옮겼다(내용 그대로 — leader21 모의매매 팔이 쓴다). 같은 규칙의 긴 표판은 정문
:func:`swing_it.storage.read_earnings` 의 ``with_known=True``(``known_date``·``event_date`` 열)다.

**원천 인자 ``source``(30 보충 7~9, 결과 전 고정):** 기본 ``"earnings"`` 는 위 경로 그대로(비트 동일). ``"earnings_dart_doc"`` 은
DART 원문 파싱 표(2007~2015) — 알게 된 날 = 그 행의 ``knowledge_date``(접수일; 정정 행은 정정 접수일) **다음 거래일**부터,
전년 같은 분기 = 같은 보고서의 비교값 ``netinc_prior``(결측이면 후보 아님 — NaN 비교는 False). 값·판정 규칙은 엔진 그대로
(``netinc > 0 ∧ netinc > netinc_prior``). 같은 날 여러 판이면 (period, 정정 여부) 순으로 뒤의 것이 이긴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 실적 원천(:func:`known_table` ``source``) — :data:`swing_it.storage.EARNINGS_TABLES` 와 같다.
SOURCES = ("earnings", "earnings_dart_doc")


def known_table(con, days: pd.DatetimeIndex, source: str = "earnings") -> pd.DataFrame:
    from swing_it.storage import read_earnings
    if source == "earnings_dart_doc":
        e = read_earnings(con, cols=("code", "period", "knowledge_date", "is_correction", "netinc", "netinc_prior"), all_versions=True,
                          parse_dates=True, table="earnings_dart_doc")
        e["code"] = e["code"].astype(str)
        e["known"] = pd.to_datetime(e["knowledge_date"])
        e["krow"] = days.searchsorted(e["known"].to_numpy(), side="right")      # 접수일 다음 거래일 행(정정 행은 정정 접수일)
        return e.sort_values(["krow", "period", "is_correction"], kind="stable")
    if source != "earnings":
        raise ValueError(f"실적 원천이 아니다: {source!r} — {SOURCES}")
    # 모든 정정 버전 그대로(예전 직접 SELECT 와 같은 SQL) — 버전·시점 선택은 아래 known 규칙이 한다
    earn = read_earnings(con, cols=("code", "period", "avail_date", "netinc", "netinc_prior"), all_versions=True, parse_dates=True)
    per = pd.read_sql("select code, period, min(rcept_dt) as per_dt from earnings_disclosures where kind='periodic' group by code, period",
                      con, parse_dates=["per_dt"])
    e = earn.merge(per, on=["code", "period"], how="left")
    e["known"] = e[["avail_date", "per_dt"]].max(axis=1)
    e["krow"] = days.searchsorted(e["known"].to_numpy(), side="right")          # 알게 된 날 다음 거래일 행
    return e.sort_values("krow")


def _latest(kt: pd.DataFrame, codes, ci: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    ev = pd.DataFrame({"code": [str(codes[i]) for i in ci], "s": np.asarray(s, np.int64), "k": np.arange(len(ci))}).sort_values("s")
    return pd.merge_asof(ev, kt[["code", "krow", "netinc", "netinc_prior"]], left_on="s", right_on="krow", by="code", direction="backward")


def eps_ok(codes, days: pd.DatetimeIndex, ci: np.ndarray, s: np.ndarray, con, source: str = "earnings") -> np.ndarray:
    m = _latest(known_table(con, days, source), codes, ci, s)
    ok = ((m["netinc"] > 0) & (m["netinc"] > m["netinc_prior"])).to_numpy()
    out = np.zeros(len(ci), bool)
    out[m["k"].to_numpy()] = ok
    return out


def eps_yoy(codes, days: pd.DatetimeIndex, ci: np.ndarray, s: np.ndarray, con, source: str = "earnings") -> np.ndarray:
    """22세대: 그 시점 최신 분기 순이익 YoY 증가율(전년 같은 분기 ≤ 0 이면 NaN) — 같은 as-of 규칙."""
    m = _latest(known_table(con, days, source), codes, ci, s)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(m["netinc_prior"] > 0, m["netinc"] / m["netinc_prior"] - 1.0, np.nan)
    out = np.full(len(ci), np.nan)
    out[m["k"].to_numpy()] = g
    return out
