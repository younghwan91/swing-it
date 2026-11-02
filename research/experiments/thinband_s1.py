"""thinband 1단계 — 2016-09~2023 U(t) 커버리지만 센다. OHLC·수익률을 읽지 않는다. U_counts 를 표준 출력으로 낸다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE = ("thinband", Stage.UNIVERSE)
START, SIG, END = "2016-05-01", "2016-09-01", "2023-12-31"


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    df = pd.read_sql_query(f"select code,date,volume,source,close_src from {L.V2} where date between %s and %s", con, params=(START, END))  # noqa: S608
    df["date"] = pd.to_datetime(df["date"])
    print("close_src", df.close_src.value_counts().to_dict())
    df = df[df.code.map(L.is_common) & ~df.close_src.isin(("pre_listing",))]
    names = L.load_names(con)
    hit = set(names[names.name.fillna("").map(L.name_excluded)].code)
    print("이름 제외(표본 안)", len(hit & set(df.code)), "이름 없음", len(set(df.code) - set(names.code)))
    every = set(df.code)
    df = df[~df.code.isin(hit)]
    nv = pd.read_sql_query("select code,date,close*volume/1e6 as tv from naver_investor_trend where date between %s and %s and close is not null",
                           con, params=(START, END))
    nv["date"] = pd.to_datetime(nv["date"])
    df = df.merge(nv, how="left", on=["code", "date"]).sort_values(["code", "date"])
    df["tv"] = df.tv.where(df.volume > 0)
    df["adv"] = df.groupby("code").tv.transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    w = df[(df.date >= SIG) & (df.volume > 0)].copy()
    print("거래 칸 중 네이버 없음", w.groupby("source").tv.apply(lambda s: round(float(s.isna().mean()), 5)).to_dict())
    w = w[w.tv.notna()]
    w["year"] = w.date.dt.year
    w["dl"] = w.source == "naver_adj"
    for name, band in (("U", L.BAND_U), ("U2", L.BAND_U2)):
        b = w[(w.adv >= band[0]) & (w.adv < band[1])]
        t = b.groupby("year").agg(name_days=("code", "size"), codes=("code", "nunique"), dl=("dl", "mean"))
        print(name, "합계", len(b))
        for y, r in t.iterrows():
            print(f"| {y} | {int(r.name_days):,} | {int(r.codes):,} | {r.dl * 100:.1f}% |")
        q = b.groupby("date").size().quantile([.05, .5, .95]).round().astype(int).tolist()
        print("하루 종목 수 5·50·95", q)
    kd = pd.read_sql_query("select code,delist_date from kind_delistings where delist_date between %s and %s", con, params=(SIG, END))
    kd = kd[kd.code.map(L.is_common)]
    kd["in_v2"] = kd.code.isin(every)
    kd["name_excluded"] = kd.code.isin(hit)
    print("KIND 폐지", len(kd), "이름 제외", int(kd.name_excluded.sum()), "v2 있음(이름 제외 뺀 뒤)", int((kd.in_v2 & ~kd.name_excluded).sum()), "/", int((~kd.name_excluded).sum()))
    dl = df[(df.date >= SIG) & (df.volume > 0) & (df.source == "naver_adj")]
    miss = dl.groupby("code").tv.apply(lambda s: float(s.isna().mean()))
    print("폐지 종목", len(miss), "네이버 전부 없음", int((miss == 1).sum()), "일부 없음", int(((miss > 0) & (miss < 1)).sum()))
    ix = pd.read_sql_query("select code,count(*) n,min(date) a,max(date) b from sector_index where code in ('001','101') and date between %s and %s group by 1",
                           con, params=(SIG, END))
    print(ix.to_dict("records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
