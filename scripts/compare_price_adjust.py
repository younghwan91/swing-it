#!/usr/bin/env python
"""price_adjust 재계산을 DB 에 쓰지 않고 메모리에서 돌려 ``daily_bars_adj_v2`` 와 맞춰 본다 — 읽기 전용.

``daily_bars_adj_v2``(quant-airflow ``collectors/adjusted_v2.py``: 원주가 × corporate_actions 계수)는 이음매가
없는 독립 기준이다. 의존하지 않고 **대조용으로만** 읽는다. 지금 운영 표(``daily_bars_adjusted``, 옛 규칙)와
새 계산을 같은 칸에서 v2 와 비교해 일간 수익률 차이 >1%/5%/20% 칸 수를 전·후로 적고, 남은 칸을 CSV 로 낸다.

    set -a; . ~/git/quant-airflow/.env; set +a
    uv run --extra pg python scripts/compare_price_adjust.py --out <scratchpad 디렉터리> [--cache <같은 곳>]

- 연결은 ``readonly`` 세션, 모든 문장에 ``statement_timeout`` 60s. 쓰기 없음.
- 비교 창 기본 2016-09-01~2026-09-11: 2026-09-14~ v2 는 KRX 정규장 분봉 종가, daily_bars 는 애프터마켓 통합
  종가라 기준이 다르다(설계상 차이).
- 출력(CSV·캐시)은 ``--out`` 에만 쓴다. 레포에 커밋하지 않는다(수집 데이터).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from swing_it import price_adjust as pa  # noqa: E402
from swing_it.storage import read_prices  # noqa: E402

THRESH = (0.01, 0.05, 0.20)


def _connect():
    import psycopg2

    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    cur = con.cursor()
    cur.execute("SET statement_timeout = '60s'")
    con.commit()
    return con


def load(cache: Path | None) -> tuple[pd.DataFrame, ...]:
    files = None if cache is None else [cache / f"{n}.pkl" for n in ("inputs", "old", "v2", "ca", "rawspan", "shares")]
    if files and all(f.exists() for f in files):
        return tuple(pd.read_pickle(f) for f in files)  # type: ignore[return-value]
    con = _connect()
    parts = []
    for y in range(2016, 2027):
        for lo, hi in (("01-01", "06-30"), ("07-01", "12-31")):  # 반년씩 — 한 문장 60s 안
            t0 = time.time()
            parts.append(pa.read_inputs(con, cols="d.code,d.date,d.close,d.volume,d.source",
                                        date_from=f"{y}-{lo}", date_to=f"{y}-{hi}"))
            print(f"inputs {y}-{lo}: {len(parts[-1]):,}행 {time.time() - t0:.0f}s", flush=True)
    inputs = pd.concat(parts, ignore_index=True)
    old = read_prices(con, cols=("code", "date", "close"), table="daily_bars_adjusted", require_delisted=False)
    v2 = read_prices(con, cols=("code", "date", "close", "close_src", "factor"), table="daily_bars_adj_v2",
                     require_delisted=False)
    # quant-airflow data_checks.check_adjusted_vs_raw 를 흉내 내려고 이벤트일만(대조에서 건너뛴다)
    ca = pd.read_sql("SELECT code, date FROM corporate_actions WHERE upd_rt <> 0", con)
    ca["date"] = ca["date"].astype(str)
    # 검사는 daily_bars_raw 가 있는 칸만 본다(폐지 종목의 supply_demand 대용 원주가는 안 본다)
    rawspan = pd.read_sql("SELECT code, min(date) AS rmin, max(date) AS rmax FROM daily_bars_raw GROUP BY code", con)
    rawspan[["rmin", "rmax"]] = rawspan[["rmin", "rmax"]].astype(str)
    shares = pa.read_shares(con)  # 심판 없는 병합의 심판(운영 재계산과 같은 입력)
    con.close()
    for d in (old, v2):
        d["date"] = d["date"].astype(str)
    if files:
        cache.mkdir(parents=True, exist_ok=True)
        for f, d in zip(files, (inputs, old, v2, ca, rawspan, shares)):
            d.to_pickle(f)
    return inputs, old, v2, ca, rawspan, shares


def _ret(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df.groupby("code")[col].shift(1)


def build(inputs: pd.DataFrame, old: pd.DataFrame, v2: pd.DataFrame, ca: pd.DataFrame | None = None,
          shares: pd.DataFrame | None = None) -> pd.DataFrame:
    new = pa.adjust_prices(inputs, cols=("close",), shares=shares)
    plan = pa._plan(inputs, shares)[["code", "date", "_factor", "_kind"]]
    df = (inputs[["code", "date", "close", "raw_close", "flu_rt", "source"]].rename(columns={"close": "db"})
          .assign(new=new["close"].to_numpy()))
    df = df.merge(old.rename(columns={"close": "old"}), on=["code", "date"], how="left")
    df = df.merge(v2.rename(columns={"close": "v2", "factor": "v2_factor"}), on=["code", "date"], how="left")
    df = df.merge(plan, on=["code", "date"], how="left")
    df = df.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    for c in ("db", "raw_close", "flu_rt", "new", "old", "v2", "v2_factor"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ("db", "raw_close", "new", "old", "v2"):
        df["r_" + c] = _ret(df, c)
    df["pdate"] = df.groupby("code")["date"].shift(1)
    # 기준이 다른 v2(2026-09-14~ 정규장 분봉)와도 비교할 수 있게 "조정 계수"의 일간 변화를 본다: 계수 = 조정가 / 원 종가.
    # v2 계수는 corporate_actions 누적계수라 기준가(정규장·통합)와 무관하다.
    for c in ("new", "old"):
        df["f_" + c] = df[c] / df["raw_close"]
        df["fs_" + c] = df["f_" + c] / df.groupby("code")["f_" + c].shift(1)
    df["fs_v2"] = df["v2_factor"] / df.groupby("code")["v2_factor"].shift(1)
    return df


def agreement(df: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    w = df[(df.date >= lo) & (df.date <= hi) & df.pdate.notna()]
    rows = []
    for name in ("old", "new"):
        ok = w["r_" + name].notna() & w.r_v2.notna()
        dev = (w["r_" + name] / w.r_v2 - 1).abs()[ok]
        row = {"표": "daily_bars_adjusted(지금)" if name == "old" else "새 계산", "비교 칸": int(ok.sum())}
        for t in THRESH:
            sel = w[ok][dev > t]
            row[f">{t:.0%} 칸"] = len(sel)
            row[f">{t:.0%} 종목"] = sel.code.nunique()
        rows.append(row)
    return pd.DataFrame(rows)


# (종목, 날짜, 기대 수익률 또는 None=|r|<0.3) — 조사에서 짚은 이음매·하한가·진짜 분할
CANARIES = [
    ("025620", "2021-07-29", None), ("258830", "2021-07-29", None), ("008830", "2019-02-27", None),
    ("001380", "2025-03-26", 0.70), ("086900", "2020-04-20", 0.70), ("258830", "2021-10-20", 6600 / 7000),
    ("008830", "2024-02-14", None),
    # quant-airflow 일일 검사(PR #23)가 짚은 칸
    ("0035S0", "2026-09-30", "raw"),   # 애프터마켓 뒤 진짜 상한가 — 원주가 비 +43.6% 그대로
    ("117670", "2026-10-06", "raw"),   # 16:00 수집 15일 창이 만든 daily_bars 자체 이음매
    ("298000", "2026-06-15", "db"),    # 키움 이벤트에 없는 기준가 재설정 — 수정주가가 맞다
    ("207940", "2026-09-17", "raw"),   # 15일 재기록 창 이음매 ×0.9923 ↔ 10-02 권리락 짝
    ("207940", "2026-10-02", "db"),
    # 리뷰(PR #12)가 짚은 칸
    ("076610", "2024-01-08", "flu"),   # 이음매 날 감자 — 두 후보 다 틀림, 그날 수익률 = flu_rt
    ("056730", "2017-01-13", "db"),    # 오랜 정지 뒤 감자 재상장 6.0035:1 — 주주 수익률 = r_db(m 은 평가가격 기준)
    ("107640", "2024-06-24", "raw"),   # 코넥스 → 이전상장 첫날 — 기준가 = 공모가, 원주가 움직임 그대로
    ("139050", "2023-06-28", None),    # 네이버 백필 12.77:1 병합(+1442%) — |r|<0.3
    ("138360", "2023-01-27", "db"),    # m 이 제한폭 밖(−52.9%) — 심판으로 안 쓴다
    # 심판 없는 병합 — 상장주식수 심판(정지 중 29,129,064 → 19,419, 그날 주주 수익률 = 원주가 비 / 주식수 비)
    ("052670", "2026-02-09", 625000 / (2080 * 29129064 / 19419)),
]


def canaries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, d, want in CANARIES:
        x = df[(df.code == code) & (df.date == d)]
        if x.empty:
            rows.append({"code": code, "date": d, "ok": False})
            continue
        x = x.iloc[0]
        if want is None:
            ok = abs(x.r_new - 1) < 0.3
        elif isinstance(want, str):
            ref = {"raw": x.r_raw_close, "db": x.r_db, "flu": 1 + x.flu_rt / 1e4}[want]
            ok = abs(x.r_new / ref - 1) < 0.005
        else:
            ok = abs(x.r_new / want - 1) < 0.005
        rows.append({"code": code, "date": d, "r_db": x.r_db, "r_raw": x.r_raw_close, "r_old": x.r_old,
                     "r_new": x.r_new, "r_v2": x.r_v2, "want": want, "ok": bool(ok)})
    return pd.DataFrame(rows)


def airflow_check(df: pd.DataFrame, ca: pd.DataFrame, rawspan: pd.DataFrame, col: str) -> pd.DataFrame:
    """quant-airflow ``check_adjusted_vs_raw`` 와 같은 규칙으로 깨진 칸: 이벤트 없음·가격제한폭 안(flu_rt 기준가)·
    |r_adj/r_raw − 1| > 20%. 반환은 깨진 칸."""
    w = df[df.pdate.notna() & df.r_raw_close.notna() & df["r_" + col].notna()]
    w = w.merge(rawspan, on="code", how="inner")
    w = w[(w.pdate >= w.rmin) & (w.date <= w.rmax)]
    # 이벤트가 (전일, 그날] 에 있으면 건너뛴다 — 이벤트일을 그 뒤 첫 거래일 행에 붙인다
    left = ca.assign(code=ca["code"].astype(object), date=pd.to_datetime(ca["date"])).sort_values("date")
    right = (w[["code", "date"]].reset_index()
             .assign(code=lambda x: x["code"].astype(object), date=lambda x: pd.to_datetime(x["date"])).sort_values("date"))
    e = pd.merge_asof(left, right, on="date", by="code", direction="forward")
    w = w.drop(index=e["index"].dropna().astype(int).unique())
    m = 1 + w.flu_rt / 1e4
    base_gap = (w.r_raw_close / m - 1).abs()
    gap_max = np.where(w.pdate >= pa.AFTERMARKET_FROM, 0.45, 0.15)
    band = np.where(w.flu_rt.notna(), (m >= 0.7) & (m <= 1.3) & (base_gap <= gap_max),
                    (w.r_raw_close >= 0.7) & (w.r_raw_close <= 1.3))
    w = w[band]
    return w[(w["r_" + col] / w.r_raw_close - 1).abs() > 0.2]


def factor_agreement(df: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    w = df[(df.date >= lo) & (df.date <= hi) & df.pdate.notna()]
    rows = []
    for name in ("old", "new"):
        ok = w["fs_" + name].notna() & w.fs_v2.notna()
        dev = (w["fs_" + name] / w.fs_v2 - 1).abs()[ok]
        row = {"표": "지금 표" if name == "old" else "새 계산", "비교 칸": int(ok.sum())}
        for t in THRESH:
            sel = w[ok][dev > t]
            row[f">{t:.0%} 칸"] = len(sel)
            row[f">{t:.0%} 종목"] = sel.code.nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def residual_class(r: pd.Series) -> str:
    if not np.isfinite(r.r_raw_close):
        return "원주가 없음(옛 규칙 경로)"
    if pd.isna(r.flu_rt):
        return "원주가 있음·심판 없음(2017 전/네이버)"
    if abs(r.r_new / r.r_raw_close - 1) < 0.005:
        return "새 계산=원주가 수익률(v2 는 이벤트로 봄)"
    if abs(r.r_new / r.r_db - 1) < 0.005:
        return "새 계산=daily_bars 수익률(v2 는 다르게 봄)"
    return "기타"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path, help="CSV 출력 디렉터리(레포 밖)")
    ap.add_argument("--cache", type=Path, default=None, help="DB 조회 결과 피클 캐시 디렉터리")
    ap.add_argument("--scan", type=Path, default=None, help="quant-airflow 검사가 짚은 칸 CSV(code,date,raw_prev,raw)")
    ap.add_argument("--from", dest="lo", default="2016-09-01")
    ap.add_argument("--to", dest="hi", default="2026-09-11")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    inputs, old, v2, ca, rawspan, shares = load(a.cache)
    print(f"입력 {len(inputs):,}행 {inputs.code.nunique():,}종목", flush=True)
    t0 = time.time()
    df = build(inputs, old, v2, shares=shares)
    print(f"재계산 {time.time() - t0:.0f}s")

    print("\n## quant-airflow check_adjusted_vs_raw 규칙(이벤트·제한폭 밖 제외, |r_adj/r_raw−1|>20%) 깨진 칸")
    for col, name in (("old", "지금 표"), ("new", "새 계산")):
        bad = airflow_check(df, ca, rawspan, col)
        last = sorted(df.date.unique())[-21:-1]  # 검사 창: T-1 까지 20거래일
        print(f"  {name}: 전 구간 {len(bad)}칸 {bad.code.nunique()}종목, 최근 20거래일 {bad.date.isin(last).sum()}칸")
        if col == "new":
            bad[["code", "pdate", "date", "db", "raw_close", "flu_rt", "r_db", "r_raw_close", "r_new", "r_old",
                 "r_v2", "_kind"]].to_csv(a.out / "airflow_check_bad_new.csv", index=False)
    if a.scan and a.scan.exists():
        sb = pd.read_csv(a.scan, dtype={"code": str})
        j = sb.merge(df[["code", "date", "r_new", "r_old", "r_raw_close", "_kind"]], on=["code", "date"], how="left")
        r_raw = j.raw / j.raw_prev
        j["fixed"] = (j.r_new / r_raw - 1).abs() <= 0.2
        print(f"  검사 스캔 {a.scan.name}: {len(j)}칸 중 새 계산 통과 {int(j.fixed.sum())}, 행 없음 {int(j.r_new.isna().sum())}")
        j[~j.fixed].to_csv(a.out / "scan_still_bad.csv", index=False)
    print("\n## 카나리아")
    print(canaries(df).to_string(index=False))
    print(f"\n## v2 대비 일간 수익률 차이 ({a.lo}~{a.hi})")
    print(agreement(df, a.lo, a.hi).to_string(index=False))
    print("\n## 같은 표, 2026-09-14 이후 포함 전 구간")
    print(agreement(df, "0000", "9999").to_string(index=False))
    print("\n## 조정 계수 일간 변화 vs v2 계수(기준가와 무관 — 애프터마켓 뒤도 비교 가능)")
    for lo, hi in ((a.lo, a.hi), ("2026-09-14", "9999")):
        print(f"[{lo}~{hi}]")
        print(factor_agreement(df, lo, hi).to_string(index=False))
    w0 = df[(df.date >= a.lo) & (df.date <= a.hi) & df.pdate.notna() & df.r_v2.notna()]
    dn, do = (w0.r_new / w0.r_v2 - 1).abs(), (w0.r_old / w0.r_v2 - 1).abs()
    worse = w0[(dn > 0.01) & (do <= 0.01)]
    print(f"\n## 지금 표보다 나빠진 칸(v2 대비 새 >1%·지금 ≤1%): {len(worse)}칸, 그중 새 >20%: {(dn[worse.index] > 0.2).sum()}칸")
    worse.assign(dn=dn[worse.index])[["code", "pdate", "date", "r_db", "r_raw_close", "flu_rt", "r_new", "r_old", "r_v2",
                                      "_kind", "dn"]].sort_values("dn", ascending=False).to_csv(a.out / "worse_than_old.csv",
                                                                                                 index=False)

    w = df[(df.date >= a.lo) & (df.date <= a.hi) & df.pdate.notna()].copy()
    chg = (w.r_new / w.r_old - 1).abs()
    print("\n## 배포 영향 — 지금 표 대비 일간 수익률이 바뀌는 칸")
    for t in (1e-6, 0.01, 0.05, 0.20):
        print(f"  >{t:g}: {(chg > t).sum():,}칸 {w[chg > t].code.nunique():,}종목")
    lvl = (df.new / df.old - 1).abs()
    print(f"  가격 수준이 1% 넘게 바뀌는 행(전 구간): {(lvl > 0.01).sum():,}행 {df[lvl > 0.01].code.nunique():,}종목")
    wa = df[df.pdate.notna()]
    chg_all = wa[(wa.r_new / wa.r_old - 1).abs() > 1e-6]
    print(f"  일간 수익률이 바뀌는 칸(전 구간): {len(chg_all)}칸")
    if len(chg_all):
        print(chg_all[["code", "pdate", "date", "raw_close", "r_raw_close", "r_old", "r_new", "r_v2", "_kind"]]
              .head(50).to_string(index=False))
    flags = df[df._kind.isin(pa.FLAG_KINDS)]
    print(f"\n## 미해결(곱하지 않음, {'·'.join(pa.FLAG_KINDS)}) {len(flags)}칸")
    if len(flags):
        print(flags[["code", "pdate", "date", "raw_close", "r_raw_close", "r_new", "r_v2", "source"]].to_string(index=False))

    adj = df[df._factor != 1]
    print("\n## 새 계산의 조정 지점(종류별)")
    print(adj.groupby("_kind").agg(칸=("code", "size"), 종목=("code", "nunique")).to_string())

    ok = w.r_new.notna() & w.r_v2.notna()
    res = w[ok & ((w.r_new / w.r_v2 - 1).abs() > 0.05)].copy()
    res["cls"] = res.apply(residual_class, axis=1)
    print(f"\n## 남은 칸 (v2 대비 >5%) {len(res)}칸")
    print(res.groupby("cls").agg(칸=("code", "size"), 종목=("code", "nunique")).to_string())
    cols = ["code", "pdate", "date", "db", "raw_close", "flu_rt", "new", "old", "v2", "close_src",
            "r_db", "r_raw_close", "r_new", "r_old", "r_v2", "_kind", "cls", "source"]
    res[cols].to_csv(a.out / "residual_gt5pct.csv", index=False)
    fixed = w[ok & w.r_old.notna() & ((w.r_old / w.r_v2 - 1).abs() > 0.05) & ((w.r_new / w.r_v2 - 1).abs() <= 0.01)]
    fixed[cols[:-2]].to_csv(a.out / "fixed_cells.csv", index=False)
    changed = w[chg > 0.01]
    changed[cols[:-2]].to_csv(a.out / "changed_gt1pct.csv", index=False)
    print(f"\nCSV: {a.out}/residual_gt5pct.csv, fixed_cells.csv, changed_gt1pct.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
