"""실적을 **쓸 수 있는 첫날** 규칙 — 3단계 신호의 정문.

레포 CLAUDE.md "실적 as-of": ``earnings.avail_date`` 는 제출 마감일 고정값이라 정기보고서 18% 가 그보다 늦다. 값을 쓸 수 있는 첫날은
max(avail_date, 정기보고서 첫 접수일)의 **다음 거래일**이다. 2019Q1 이후 분기는 ``end`` 까지 정기보고서가 없으면 뺀다. 2018Q3 이전은
정기보고서 날짜가 없어 avail_date 로 돌아간다(미래 정보 위험이 남는다, PEAD PREREG §2).

``avail_date`` 만 보는 경로(``earnings_yoy_panel`` 그대로)는 이 규칙을 어긴다. 2019~ 종목-일의 3.6% 가 달랐다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from swing_it.features.fundamentals import _yoy_vec, earnings_yoy_panel
from swing_it.storage import read_earnings, read_periodic_filings

#: 정기보고서 접수일이 있어야 하는 첫 분기(PO-2.1-5).
PERIODIC_REQUIRED_FROM = "2019Q1"
YOY_COLS = ("code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior")


def yoy_strict_wide(ea: pd.DataFrame, per: pd.DataFrame, codes, dates, end: str, *,
                    tie_col: "str | None" = "period") -> tuple[np.ndarray, dict]:
    """분기 순이익 YoY 패널(code × date).

    Args:
        ea: 실적 전 버전(``YOY_COLS``, ``avail_date``·``knowledge_date`` ≤ ``end``).
        per: code·period·per(정기보고서 첫 접수일, ≤ ``end``).
        codes, dates: 패널 좌표. ``dates`` 는 거래일이다 — "다음 거래일"을 여기서 잰다.
        end: 상한. 넘는 값이 있으면 ``AssertionError``.
        tie_col: 같은 날 알려진 두 분기를 고르는 열. 기본은 늦은 분기다(정정 F1-2). ``None`` 은 옛 동작 — 단계 B 재현에만 쓴다.

    Returns:
        ``(패널, 보고)``. 그날 알 수 없는 칸은 NaN 이다.
    """
    end_ts = pd.Timestamp(end)
    ea = ea.copy()
    per = per.copy()
    ea["avail_date"] = pd.to_datetime(ea["avail_date"])
    ea["knowledge_date"] = pd.to_datetime(ea["knowledge_date"])
    assert ea["avail_date"].max() <= end_ts and ea["knowledge_date"].max() <= end_ts
    n_read = len(ea)
    ea["code"] = ea["code"].astype(str)
    per["code"] = per["code"].astype(str)
    per["per"] = pd.to_datetime(per["per"])
    assert per.empty or per["per"].max() <= end_ts
    ea = ea.merge(per, on=["code", "period"], how="left")
    late = (ea["period"].astype(str) >= PERIODIC_REQUIRED_FROM) & ea["per"].isna()
    n_drop = int(late.sum())
    ea = ea[~late].copy()
    known = ea[["avail_date", "per"]].max(axis=1).to_numpy(dtype="datetime64[ns]")
    days = pd.DatetimeIndex(dates)
    i = days.searchsorted(known, side="right")
    ok = (i < len(days)) & ~np.isnat(known)
    ea = ea[ok].copy()
    ea["known_date"] = days[i[ok]].strftime("%Y-%m-%d")
    ea["knowledge_date"] = ea["knowledge_date"].dt.strftime("%Y-%m-%d")
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), list(days.strftime("%Y-%m-%d")),
                               avail_col="known_date", knowledge_col="knowledge_date", tie_col=tie_col)
    wide = panel.pivot(index="code", columns="date", values="yoy")
    wide = wide.reindex(index=[str(c) for c in codes], columns=list(days.strftime("%Y-%m-%d")))
    info = {"earnings_rows_read_le_end": int(n_read), "dropped_2019Q1plus_no_periodic_le_end": n_drop,
            "rows_used": int(len(ea))}
    return wide.to_numpy(dtype=np.float64), info


def load_yoy_strict(con: Any, codes, dates, end: str, *, tie_col: "str | None" = "period") -> tuple[np.ndarray, dict]:
    """정문으로 읽어 :func:`yoy_strict_wide` 를 부른다. ``end`` 뒤 행은 메모리에 올리지 않는다."""
    ea = read_earnings(con, all_versions=True, until=end, cols=YOY_COLS)
    return yoy_strict_wide(ea, read_periodic_filings(con, end), codes, dates, end, tie_col=tie_col)
