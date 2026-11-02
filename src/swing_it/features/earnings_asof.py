"""실적을 **쓸 수 있는 첫날** 규칙 — 3단계 신호의 정문.

레포 CLAUDE.md "실적 as-of": ``earnings.avail_date`` 는 제출 마감일 고정값이라 정기보고서 18% 가 그보다 늦다. 값을 쓸 수 있는 첫날은
max(avail_date, 정기보고서 첫 접수일)의 **다음 거래일**이다. ``periodic_required_from`` 이후 분기는 ``end`` 까지 정기보고서가 없으면 뺀다.
그 앞 분기는 있으면 정기보고서 날짜를 쓰고, 없으면 avail_date 로 돌아간다(2018Q3 이전은 날짜가 없다 — 미래 정보 위험이 남는다, PEAD PREREG §2).

``avail_date`` 만 보는 경로(``earnings_yoy_panel`` 그대로)는 이 규칙을 어긴다. 2019~ 종목-일의 3.6% 가 달랐다.

**새 연구 기본과 재현 경로.** 단계 B·P4 를 만든 코드와 세 군데가 다르다. 재현은 :data:`REPRODUCE_B` 를 넘긴다.

- ``periodic_required_from``: 기본 2018Q4. 옛 값 2019Q1 은 절단 동일성이 end 2019-04~05 에서 깨진다 — 2018Q4 사업보고서가 그때 나오므로,
  자른 패널은 정기보고서를 몰라 avail_date 로 돌아가 값이 긴 패널보다 일찍 보인다.
- ``knowledge_next_day``: 기본 참. 정정본도 첫 공시처럼 알게 된 다음 거래일부터 쓴다. 옛 동작은 당일 사용이다(정정 이력이 있는 2026 만 해당).
- ``tie_col``: 기본 늦은 분기(정정 F1-2).

**tie_col 의 한계.** 같은 날 알려진 두 분기만 푼다. 옛 분기가 새 분기보다 **늦게** 알려진 역전(실측 3건)에서는 옛 분기 값이 최신으로 쓰인다.
값이 끊긴 뒤에도 그대로 이어진다 — 새 분기가 안 오면 몇 년 전 값이 살아 있다. 나이를 제한하려면 ``max_age_days`` 를 준다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from swing_it.features.fundamentals import _yoy_vec, earnings_yoy_panel
from swing_it.storage import read_earnings, read_periodic_filings

#: 정기보고서 접수일이 있어야 하는 첫 분기 — 새 연구 기본(절단 동일성, 모듈 설명).
PERIODIC_REQUIRED_FROM = "2018Q4"
#: 단계 B·P4 를 숫자 그대로 재현하는 인자(PO-2.1-5). 새 연구는 쓰지 않는다.
REPRODUCE_B = {"tie_col": None, "periodic_required_from": "2019Q1", "knowledge_next_day": False}
YOY_COLS = ("code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior")


def yoy_strict_wide(ea: pd.DataFrame, per: pd.DataFrame, codes, dates, end: str, *,
                    tie_col: "str | None" = "period", periodic_required_from: str = PERIODIC_REQUIRED_FROM,
                    knowledge_next_day: bool = True, max_age_days: "int | None" = None) -> tuple[np.ndarray, dict]:
    """분기 순이익 YoY 패널(code × date).

    Args:
        ea: 실적 전 버전(``YOY_COLS``, ``avail_date``·``knowledge_date`` ≤ ``end``).
        per: code·period·per(정기보고서 첫 접수일, ≤ ``end``).
        codes, dates: 패널 좌표. ``dates`` 는 단조 증가하는 거래일이다 — "다음 거래일"을 여기서 잰다.
        end: 상한. 넘는 값이 있으면 ``AssertionError``.
        tie_col: 같은 날 알려진 두 분기를 고르는 열. 기본은 늦은 분기다(정정 F1-2). ``None`` 은 옛 동작 — 단계 B 재현에만 쓴다.
        periodic_required_from: 이 분기부터 ``end`` 까지 정기보고서가 없으면 뺀다.
        knowledge_next_day: 정정본을 알게 된 다음 거래일부터 쓴다. 거짓은 당일 사용(재현 전용).
        max_age_days: 값을 쓸 수 있게 된 날부터 이 달력일이 지난 칸은 NaN. ``None`` 은 제한 없음.

    Returns:
        ``(패널, 보고)``. 그날 알 수 없는 칸은 NaN 이다. 보고의 ``dropped_2019Q1plus_no_periodic_le_end`` 는 재현 대조 때문에 이름을 유지했다 —
        실제로는 ``periodic_required_from`` 이후 분기를 센다.
    """
    if len(ea) == 0:
        raise ValueError("실적이 비었다 — 빈 응답을 성공으로 넘기지 않는다")
    if len(dates) == 0:
        raise ValueError("dates 가 비었다")
    days = pd.DatetimeIndex(dates)
    if not days.is_monotonic_increasing or days.has_duplicates:
        raise ValueError("dates 는 중복 없이 단조 증가해야 한다 — 다음 거래일을 잴 수 없다")
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
    late = (ea["period"].astype(str) >= periodic_required_from) & ea["per"].isna()
    n_drop = int(late.sum())
    ea = ea[~late].copy()
    known = ea[["avail_date", "per"]].max(axis=1).to_numpy(dtype="datetime64[ns]")
    i = days.searchsorted(known, side="right")
    ok = (i < len(days)) & ~np.isnat(known)
    ea = ea[ok].copy()
    ea["known_date"] = days[i[ok]].strftime("%Y-%m-%d")
    if knowledge_next_day:
        kn = days.searchsorted(ea["knowledge_date"].to_numpy(dtype="datetime64[ns]"), side="right")
        ea = ea[kn < len(days)].copy()
        ea["knowledge_date"] = days[kn[kn < len(days)]]
    ea["knowledge_date"] = ea["knowledge_date"].dt.strftime("%Y-%m-%d")
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    panel = earnings_yoy_panel(ea.dropna(subset=["yoy"]), list(days.strftime("%Y-%m-%d")),
                               avail_col="known_date", knowledge_col="knowledge_date", tie_col=tie_col)
    wide = panel.pivot(index="code", columns="date", values="yoy")
    wide = wide.reindex(index=[str(c) for c in codes], columns=list(days.strftime("%Y-%m-%d")))
    if max_age_days is not None:
        age = panel.pivot(index="code", columns="date", values="age_days").reindex(index=wide.index, columns=wide.columns)
        wide = wide.where(age.to_numpy(dtype=np.float64) <= max_age_days)
    info = {"earnings_rows_read_le_end": int(n_read), "dropped_2019Q1plus_no_periodic_le_end": n_drop,
            "rows_used": int(len(ea))}
    return wide.to_numpy(dtype=np.float64), info


def load_yoy_strict(con: Any, codes, dates, end: str, **kw: Any) -> tuple[np.ndarray, dict]:
    """정문으로 읽어 :func:`yoy_strict_wide` 를 부른다. ``end`` 뒤 행은 메모리에 올리지 않는다. 인자는 그 함수와 같다."""
    ea = read_earnings(con, all_versions=True, until=end, cols=YOY_COLS)
    return yoy_strict_wide(ea, read_periodic_filings(con, end), codes, dates, end, **kw)
