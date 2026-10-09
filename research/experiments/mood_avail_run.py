#!/usr/bin/env python
"""신호용 기분 지수 mood_avail 재집계 — 리뷰 M1·M2 (``regime/REVIEW_REGIME_DATA.md``, ``ADAPTIVE.md`` §7). **API 호출 없음.**

원장(``article_judgments``, judge ``market-mood-haiku55``)에 이미 있는 판정 8,677건을 판단일 d = **d 15:30 ≤ 발행 < 다음 거래일 07:30 KST**
로 다시 묶고(:func:`swing_it.news.market_mood.mood_avail_daily`), ``[표]``·전부 중립 출력을 빼고, 남은 기사 < 5건이면 결측으로 둔 뒤
20일 EMA 를 그 위에서 다시 계산한다. 수익·트레이드 결과는 보지 않는다(KOSPI 수익 열도 넣지 않는다).

산출물(``research/logs/sepa_explore/regime/``):
- ``mood_avail_daily.csv`` — 신호용(판단일 d 의 기분, d+1 07:35 작업에서 쓴다).
- ``mood_daily.csv`` — 옛 판(행 d = 세션 d−1 묘사). 지우지 않고 첫 줄에 "신호에 쓰지 말 것" 주석을 단다(``pd.read_csv(..., comment="#")``).
- ``MOOD_PILOT.md`` 의 "신호용 mood_avail" 절.

Run:  .venv/bin/python research/experiments/mood_avail_run.py
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swing_it.news import market_mood as mm  # noqa: E402
from swing_it.news.daily_leg import map_signal_days  # noqa: E402
from swing_it.storage import connect, db_default, load_env_db  # noqa: E402

OUT_DIR = Path("research/logs/sepa_explore/regime")
PANEL_CSV = OUT_DIR / "regime_panel.csv"
OLD_CSV = OUT_DIR / "mood_daily.csv"
OLD_MARK = "# 세션 d−1 묘사, 신호에 쓰지 말 것 — 신호용은 mood_avail_daily.csv (REVIEW_REGIME_DATA.md M1)"


def md_table(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for i, r in df.iterrows():
        cells = ["" if pd.isna(v) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r]
        rows.append("| " + " | ".join([str(getattr(i, "date", lambda: i)())] + cells) + " |")
    return "\n".join([head, sep, *rows])


def mark_old_csv(path: Path) -> bool:
    """옛 CSV 첫 줄에 경고 주석(이미 있으면 그대로). 반환 = 새로 달았는지."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if text.startswith("# 세션 d−1"):
        return False
    path.write_text(OLD_MARK + "\n" + text, encoding="utf-8")
    return True


def write_section(path: Path, section: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"## {section}\n"
    if marker in text:
        pre, rest = text.split(marker, 1)
        nxt = rest.find("\n## ")
        text = pre + marker + body + ("\n" + rest[nxt + 1:] if nxt >= 0 else "")
    else:
        text = text.rstrip("\n") + "\n\n" + marker + body
    path.write_text(text, encoding="utf-8")


def main() -> int:
    load_env_db()
    con = connect(db_default())
    days = sorted(pd.to_datetime(pd.read_csv(PANEL_CSV, usecols=["date"])["date"].unique()).date)
    arts = mm.load_judged_articles(con)
    outs = mm.existing_outputs(con)
    con.close()
    print(f"judged articles {len(arts):,}, outputs {len(outs):,}, trading days {len(days):,}", flush=True)

    # 옛 대응과의 관계: 판단일 d 의 기사 집합 = 옛 행 next_td(d) 의 기사 중 next_td(d) 07:30 전 발행분
    arts["old_day"] = map_signal_days(arts["published_at"], days, mode="close").to_numpy()
    arts["new_day"] = mm.avail_decision_days(arts["published_at"], days).to_numpy()
    nxt = dict(zip(days[:-1], days[1:]))
    by_old = arts.groupby("old_day")["key"].apply(frozenset)
    by_new = arts.dropna(subset=["new_day"]).groupby("new_day")["key"].apply(frozenset)
    same = sum(1 for d, ks in by_new.items() if d in nxt and by_old.get(nxt[d]) == ks)
    comparable = sum(1 for d in by_new.index if d in nxt and nxt[d] in by_old.index)
    differ = [str(d) for d, ks in by_new.items() if d in nxt and nxt[d] in by_old.index and by_old.get(nxt[d]) != ks]
    dropped = int(arts["new_day"].isna().sum())

    daily = mm.mood_avail_daily(arts, outs, days)
    idx = mm.mood_index(daily, days)
    idx = idx.loc[idx.index >= daily.index.min()]          # 기사 아카이브 시작(2023-03-15) 전 행은 버린다
    idx.to_csv(OUT_DIR / "mood_avail_daily.csv", float_format="%.4f")
    raw = mm.mood_avail_daily(arts, outs, days, exclude=False)
    yearly = mm.exclusion_by_year(arts, outs, days)
    marked = mark_old_csv(OLD_CSV)
    mark_old_csv(OUT_DIR / "mood_pilot_daily.csv")       # 파일럿 CSV 도 같은 옛 정의

    first = daily.index.min()
    span = idx.loc[idx.index >= first]
    n_days = len(span)
    n_valid = int(span["market_direction"].notna().sum())
    n_short = int(((span["n_judged"] > 0) & (span["n_judged"] < mm.MIN_ARTICLES)).sum())
    n_empty = int((span["n_window"] == 0).sum())
    nyear = span.assign(y=span.index.year).groupby("y").apply(
        lambda g: pd.Series({"days": len(g), "valid_%": 100 * g["market_direction"].notna().mean(),
                             "n_used_median": g["n_judged"].median(), "n_window_median": g["n_window"].median()}),
        include_groups=False).rename_axis("year")
    corr_raw = pd.concat([daily["market_direction"], raw["market_direction"]], axis=1).dropna().corr().iloc[0, 1]
    tail = idx[["n_window", "n_judged", *mm.FIELDS, "mood_score"]].dropna(subset=["market_direction"]).tail(10).rename(
        columns={"n_window": "창", "n_judged": "n", "market_direction": "dir", "risk_appetite": "risk",
                 "foreign_flow_tone": "frgn", "breadth_tone": "brdth", "event_risk": "event", "mood_score": "score"})
    body = "\n".join([
        f"실행 {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · `research/experiments/mood_avail_run.py` · **API 호출 없음**(원장 판정 재사용)",
        "",
        "**신호에는 이것(`mood_avail_daily.csv`)만 쓴다.** 판단일 d = d 15:30 ≤ 발행 < 다음 거래일 07:30 KST 기사 → d 종가 결정(d+1 07:35 작업)에 "
        "쓰는 기분. 옛 `mood_daily.csv` 의 행 d 는 세션 d−1 의 마감 시황(리뷰 M1)이라 파일 첫 줄에 \"신호에 쓰지 말 것\" 주석을 달았다"
        + (" (이번 실행에서 달았다)" if marked else "") + " — 읽을 때 `pd.read_csv(..., comment=\"#\")`.",
        "",
        f"- 재배정: 판정 기사 {len(arts):,}건 중 {len(arts) - dropped:,}건이 판단일 창 안, {dropped}건은 창 밖(장중 07:30~15:30 발행 — 버림). "
        f"판단일 {len(by_new):,}일 중 옛 행 next_td(d) 와 기사 집합이 **같은 날 {same}/{comparable}**, 다른 날: {', '.join(differ) or '없음'} "
        "(리뷰가 짚은 2023 초 누출 후보 — 다음 날 개장 뒤 기사가 빠진다).",
        f"- 집계 규칙: `[표]` 제목·5필드 전부 중립 출력은 뺀다(리뷰 M2), 남은 기사 < {mm.MIN_ARTICLES}건이면 그날 5필드는 결측(EMA 는 전날 값 유지). "
        f"판단일 {n_days:,}일 중 값 있는 날 **{n_valid:,}**, 기사 있지만 < {mm.MIN_ARTICLES}건 {n_short}, 기사 0건 {n_empty}.",
        f"- 제외 전·후 일 평균 방향 상관 {corr_raw:.3f}.",
        "- 라이브: `market_mood.live_mood_avail` — `collected_at < 다음 거래일 07:30` 인 기사만, 같은 제외, < 5건이면 결측(0 이 아님). "
        "역사 구간의 `collected_at` 은 아카이브 이전 시각이라 발행 시각으로 대신했다 — 라이브와 백테스트의 가용 시각 정의가 다르다는 점을 적어 둔다.",
        "",
        "### 연도별 `[표]`·전부 중립 비중과 일 평균 |방향| (제외 전 → 후)",
        "",
        md_table(yearly.rename(columns={"articles": "기사", "abs_dir_before": "|dir| 전", "abs_dir_after": "|dir| 후"})),
        "",
        "### 연도별 판단일 커버리지",
        "",
        md_table(nyear),
        "",
        "### 마지막 10 판단일 (기술 — 수익 열 없음)",
        "",
        md_table(tail.rename_axis("date")),
        "",
        "남은 한계(리뷰 M3·m9, 이번에 고치지 않음): 하루 10건의 실효 표본 ≈ 1.6건(같은 장면을 여러 매체가 씀), 방향 필드 ≈ 그 세션 지수 부호 — "
        "비교 기준선 `idx_sign_ema20`(국면 패널)을 넘어야 뜻이 있다. `breadth_tone` 은 스키마 의미와 모델 사용이 달라 해석 불가로 둔다.",
        "",
    ])
    write_section(OUT_DIR / "MOOD_PILOT.md", "신호용 mood_avail (리뷰 M1·M2)", body)
    print(yearly.round(3).to_string())
    print(nyear.round(2).to_string())
    print(f"same-as-old-next-row {same}/{comparable}, differ {differ}, dropped {dropped}, valid days {n_valid}/{n_days}, "
          f"short {n_short}, empty {n_empty}, corr raw/excl {corr_raw:.3f}, old csv marked {marked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
