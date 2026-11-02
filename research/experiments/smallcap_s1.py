"""smallcap 1단계 — U(t) 커버리지만 센다(두 절반). OHLC·수익률을 읽지 않는다. U.md 와 name_filter.csv 를 쓴다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect, read_prices

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE = ("smallcap", Stage.UNIVERSE)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "smallcap"
BANDS = ((150, 300), (300, 750), (750, 2000), (2000, 5000), (5000, 10000), (10000, 20000), (20000, 10**9))


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    df = read_prices(con, cols=("code", "date", "volume", "source", "close_src"), table=L.V2, require_delisted=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df.date >= "2007-11-01") & (df.date <= L.DATA_END) & df.code.map(L.is_common) & (df.close_src != "pre_listing")]
    names = L.load_names(con)
    hit = names[names.code.isin(df.code.unique()) & names.name.fillna("").map(L.name_excluded)]
    hit[["code", "name", "market"]].sort_values("code").to_csv(OUT / "name_filter.csv", index=False)
    df = df[~df.code.isin(set(hit.code))]
    nv = pd.read_sql_query("select code,date,close*volume/1e6 as tv from naver_investor_trend "
                           "where date between '2007-11-01' and %s and close is not null", con, params=(L.DATA_END,))
    nv["date"] = pd.to_datetime(nv["date"])
    df = df.merge(nv, how="left", on=["code", "date"]).sort_values(["code", "date"])
    df["tv"] = df.tv.where(df.volume > 0)
    df["adv"] = df.groupby("code").tv.transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    df["half"] = df.code.map(L.half_of)
    w = df[(df.date >= L.SIG_START) & (df.volume > 0) & df.tv.notna()].copy()
    w["year"] = w.date.dt.year
    w["dl"] = w.source == "naver_adj"
    lines = []
    for name, band in (("U(t) ADV 3~20억", L.BAND_U), ("U2(t) ADV 20~100억", L.BAND_U2)):
        b = w[(w.adv >= band[0]) & (w.adv < band[1])]
        lines += [f"\n### {name}\n", "| 연도 | 잠금 종목-일 | 잠금 종목 | 잠금 폐지 비중 | 탐색 종목-일 | 탐색 종목 | 탐색 폐지 비중 |", "|---|---|---|---|---|---|---|"]
        for y, g in b.groupby("year"):
            row = []
            for h in (1, 0):
                x = g[g.half == h]
                row += [f"{len(x):,}", f"{x.code.nunique():,}", f"{x.dl.mean() * 100:.1f}%"]
            lines.append(f"| {y} | " + " | ".join(row) + " |")
        tot = b.groupby("half").size()
        lines.append(f"| 합계 | {tot[1]:,} | | | {tot[0]:,} | | |")
        per = b.groupby(["date", "half"]).size().unstack(fill_value=0)
        lines.append(f"\n하루 종목 수(5·50·95 백분위): 잠금 {per[1].quantile([.05, .5, .95]).round().astype(int).tolist()}, 탐색 {per[0].quantile([.05, .5, .95]).round().astype(int).tolist()}")
    lines += ["\n### ADV20 구간별 종목-일 (두 절반 합)\n", "| ADV20(백만원) | " + " | ".join(str(y) for y in sorted(w.year.unique())) + " |", "|---|" + "---|" * w.year.nunique()]
    for lo, hi in BANDS:
        c = w[(w.adv >= lo) & (w.adv < hi)].groupby("year").size()
        lines.append(f"| {lo}~{hi if hi < 10**9 else ''} | " + " | ".join(f"{int(c.get(y, 0)):,}" for y in sorted(w.year.unique())) + " |")
    all_rows = df[(df.date >= L.SIG_START) & (df.volume > 0)]
    miss = all_rows.groupby("source").tv.apply(lambda s: float(s.isna().mean()))
    lines.append(f"\n거래가 있는 v2 칸 중 네이버 값이 없어 U 밖으로 나가는 비중: {miss.round(5).to_dict()}")
    lines.append(f"\n이름 거르개 제외 {len(hit)}종목(`name_filter.csv`). 메리츠 포함 여부: {bool(hit.name.fillna('').str.contains('메리츠').any())}.")
    (OUT / "U_counts.tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(np.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
