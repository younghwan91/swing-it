"""공시 촉매 — DART 접수 목록을 "호재 사건" 으로 바꾼다 (스윙 1호 가설의 뉴스 축).

**왜 DART 인가.** 이 저장소에 쓸 수 있는 뉴스 이력은 DB(``news_articles``·``news_judgments``)가
2026-09 부터 한 달뿐이고, 토스 기사 아카이브(daytrade-it)는 LLM 판정이 없다. 반면 scalp-it 이
모아 둔 DART 접수 목록은 **2016-09 부터 10년**이고 보고서 이름이 사건 종류를 말해 준다.
뉴스의 "무슨 일이 있었나" 를 공짜로, 시점 안전하게 얻는 유일한 장기 원천이다.

**분류는 ``material_type`` 이 아니라 ``report_nm`` 패턴으로 한다.** scalp-it 의 material_type 은
넓다 — ``주주환원`` 에 자기주식 *처분*·배당 기준일이, ``기술_인허가`` 에 가족친화인증이, ``시설투자`` 에
유형자산 *양도* 가 섞여 있다(2026-10-08 실측). 호재 방향이 분명한 보고서만 이름으로 고른다.

**시점:** 원천에는 접수일(``rcept_dt``)만 있고 시각이 없다. 그래서 사건일은 "접수일 이후 첫 거래일"
이고, 그날 종가까지의 정보로 판단해 **다음 거래일 시가**에 산다(전략 쪽 규약). 장중 공시엔 늦고
장후 공시엔 정시다 — 엣지를 과소평가하는 방향의 보수성이다.

순수 함수(+ sqlite 파일 리더 하나). kr_quant DB 는 건드리지 않는다 — 그건 :mod:`swing_it.storage` 몫이다.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

#: scalp-it 이 쌓는 DART 사본. ``SWING_DART_DBS`` (os.pathsep 구분)로 덮어쓴다.
DEFAULT_DART_DBS = tuple(
    str(Path.home() / "git" / "scalp-it" / "data" / name)
    for name in ("dart_full.db", "dart_2024.db", "dart.db")
)

#: 호재 촉매 가족 → (포함 패턴 중 하나, 제외 패턴 전부 미포함). 사전등록 2026-10-08 — 바꾸면 새 가설이다.
CATALYST_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "supply_contract": (("단일판매ㆍ공급계약체결",), ("해지",)),
    "buyback": (("자기주식취득결정", "자기주식취득신탁계약체결결정", "주식소각결정"), ("처분",)),
    "capex": (("신규시설투자등",), ()),
}

_CORRECTION_PREFIX = "[기재정정]"


def dart_db_paths() -> list[str]:
    env = os.environ.get("SWING_DART_DBS")
    return [p for p in (env.split(os.pathsep) if env else DEFAULT_DART_DBS) if p]


def read_dart(paths: "list[str] | None" = None, *, start: "str | None" = None,
              end: "str | None" = None) -> pd.DataFrame:
    """DART 사본 sqlite 들을 합쳐 읽는다. 없는 파일은 건너뛰고, 같은 ``rcept_no`` 는 하나만 남긴다.

    반환 열: ``rcept_no``·``rcept_dt``(Timestamp)·``code``·``report_nm``·``material_type``·``is_correction``.
    ``start``/``end`` 는 ``YYYY-MM-DD`` (포함).
    """
    frames = []
    for path in paths or dart_db_paths():
        if not Path(path).exists():
            continue
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            frames.append(pd.read_sql_query(
                "SELECT rcept_no, rcept_dt, code, report_nm, material_type, is_correction "
                "FROM disclosures", con))
    cols = ["rcept_no", "rcept_dt", "code", "report_nm", "material_type", "is_correction"]
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("rcept_no", keep="last")
    df["rcept_dt"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["rcept_dt", "code"])
    if start is not None:
        df = df[df["rcept_dt"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["rcept_dt"] <= pd.Timestamp(end)]
    df["is_correction"] = df["is_correction"].fillna(0).astype(int)
    return df[cols].sort_values(["rcept_dt", "code"]).reset_index(drop=True)


def classify_catalyst(report_nm: "str | None") -> "str | None":
    """보고서 이름 → 촉매 가족(호재만). 정정 공시·해당 없음은 ``None``."""
    if not report_nm or report_nm.startswith(_CORRECTION_PREFIX):
        return None
    for family, (include, exclude) in CATALYST_RULES.items():
        if any(p in report_nm for p in include) and not any(x in report_nm for x in exclude):
            return family
    return None


def catalyst_events(dart: pd.DataFrame) -> pd.DataFrame:
    """DART 접수 목록 → 호재 사건 ``code``·``rcept_dt``·``family``·``report_nm`` (정정 제외)."""
    if dart.empty:
        return pd.DataFrame(columns=["code", "rcept_dt", "family", "report_nm"])
    fam = dart["report_nm"].map(classify_catalyst)
    keep = fam.notna() & (dart["is_correction"] == 0)
    out = dart.loc[keep, ["code", "rcept_dt", "report_nm"]].copy()
    out.insert(2, "family", fam[keep].to_numpy())
    return out.reset_index(drop=True)


def align_to_trading_days(events: pd.DataFrame, trading_days: "pd.DatetimeIndex") -> pd.DataFrame:
    """``event_day`` = 접수일 **이후(같은 날 포함) 첫 거래일**. 거래일 밖(마지막 거래일 뒤)은 버린다.

    같은 종목·같은 사건일에 여러 공시가 있으면 한 행으로 합친다(``family`` 는 정렬된 가족명들을 ``+`` 로).
    """
    days = pd.DatetimeIndex(sorted(trading_days))
    if events.empty or len(days) == 0:
        return pd.DataFrame(columns=["code", "event_day", "family", "n_filings"])
    pos = days.searchsorted(pd.DatetimeIndex(events["rcept_dt"]), side="left")
    ok = pos < len(days)
    ev = events.loc[ok].copy()
    ev["event_day"] = days[pos[ok]]
    agg = (ev.groupby(["code", "event_day"])
             .agg(family=("family", lambda s: "+".join(sorted(set(s)))),
                  n_filings=("family", "size"))
             .reset_index())
    return agg
