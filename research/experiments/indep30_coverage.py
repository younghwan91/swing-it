"""30 독립 표본 실행 조건 검사(META_EVOLUTION.md '30 보충 6·7·9') — **가격·수익률을 보지 않는다.** 실적 표·거래대금 표·카나리아만.

커버리지 바: 판정 구간 해마다(2009 ~ 2016-08), 20일 거래대금 ≥ 10억(엔진 ``uni`` 와 같은 정의: rolling(20, min 15) 평균 ≥ 1,000 백만원)
종목-일 중 비12월 결산 종목(정기보고서 메타데이터 = DART list 캐시의 사업보고서 기준월, quant-airflow ``select_filings`` 와 같은 규칙)을
분모에서 뺀 뒤, as-of 최신 행(``eps_asof.known_table(source="earnings_dart_doc")`` — 실행기와 **같은 코드**)에 ``netinc`` 와
``netinc_prior`` 가 둘 다 있는 비율. 문턱: 상장 ≥ 90% · 폐지 ≥ 85% · 폐지 ≥ 상장 − 5pp(해마다). 상장/폐지 = v2 ``source``
(kiwoom_raw / naver_adj). 카나리아: 005930 원 종가 연속(±15% 시대 초과 점프 수), KIND 폐지 목록 대비 v2 폐지 커버리지 ≥ 95%.
사용: ``uv run --extra pg python research/experiments/indep30_coverage.py --out <json>``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src", os.path.expanduser("~/git/quant-airflow")]
from swing_it.units import eok  # noqa: E402

V2 = "daily_bars_adj_v2"
WINDOW = ("2008-01-01", "2016-08-31")
YEARS = list(range(2009, 2017))
ADV_MIN = float(eok(10))                           # 10억 = 1,000 백만원(엔진 leader21.ADV_MIN 과 같다)
BARS = {"listed": 0.90, "delisted": 0.85, "gap_pp": 0.05}
DART_CACHE = Path(os.path.expanduser("~/data/dartdoc_cache"))
UNUSABLE_MAP = {"stock_code_unconfirmed", "unconfirmed", "ambiguous", "no_name_match", "not_verified", "class_share_unmapped"}


def corp_map(rows: list[dict], mapping: dict) -> dict[str, list[str]]:
    """quant-airflow ``collectors.dart_doc_run.corp_map`` 과 같은 규칙(그 모듈은 네트워크 의존이 있어 여기 옮김)."""
    out: dict[str, list[str]] = {}
    for r in rows:
        if r["stock_code"]:
            out.setdefault(r["stock_code"], []).append(r["corp_code"])
    for code, m in mapping.items():
        if m.get("status") in UNUSABLE_MAP or not m.get("corp_code"):
            out.pop(code, None)
        else:
            out[code] = [m["corp_code"]]
    return out


def fiscal_status(codes: list[str]) -> dict[str, dict[int, str]]:
    """종목 → {연도: 'ok' | 'non_dec' | 'no_report' | 'no_map' | 'not_run'} — 정기보고서 메타데이터(list 캐시)만."""
    from collectors import dart_doc as dd
    corp = corp_map(json.loads((DART_CACHE / "corpCode.json").read_text()), json.loads((DART_CACHE / "map_delisted.json").read_text()))
    bgn, end = "20070101", "20170630"                 # dart_doc_run.list_window(2007..2016)
    out: dict[str, dict[int, str]] = {}
    for code in codes:
        cps = corp.get(code, [])
        if not cps:
            out[code] = dict.fromkeys(YEARS, "no_map")
            continue
        items: list[dict] = []
        missing = False
        for cp in cps:
            p = DART_CACHE / f"list_{cp}_{bgn}_{end}.json"
            if not p.exists():
                missing = True
                break
            items += json.loads(p.read_text())
        if missing:
            out[code] = dict.fromkeys(YEARS, "not_run")
            continue
        dec, anyfye = dd.periodic_years(items, dec_only=True), dd.periodic_years(items, dec_only=False)
        out[code] = {y: ("no_report" if y not in anyfye else "non_dec" if y not in dec else "ok") for y in YEARS}
    return out


def main() -> None:
    import psycopg2

    from swing_it.features.eps_asof import known_table
    from swing_it.storage import read_prices
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    v2 = read_prices(con, cols=("code", "date", "trade_value", "raw_close", "close_src", "source"), table=V2, start=WINDOW[0], end=WINDOW[1],
                     parse_dates=True, session="raw")
    v2 = v2[v2["close_src"] != "pre_listing"]
    days = pd.DatetimeIndex(sorted(v2["date"].unique()))
    codes = sorted(v2["code"].astype(str).unique())
    ci_of = {c: i for i, c in enumerate(codes)}
    delisted = set(v2.loc[v2["source"] == "naver_adj", "code"].astype(str))
    tvw = v2.pivot_table(index="date", columns="code", values="trade_value", aggfunc="last").reindex(index=days, columns=codes)
    uni = (tvw.rolling(20, min_periods=15).mean() >= ADV_MIN).to_numpy().T          # 종목×날짜(엔진 leader21.build 의 uni 와 같은 식)

    # 카나리아 1: 005930 원 종가 연속(분할 없는 구간) — 수준 범위와 ±15% 시대 초과 점프 수만 본다
    s = v2.loc[v2["code"] == "005930", ["date", "raw_close"]].dropna().sort_values("date")
    jump = np.abs(np.log(s["raw_close"].to_numpy(float)[1:] / s["raw_close"].to_numpy(float)[:-1]))
    canary = {"samsung_days": int(len(s)), "samsung_cal_days": int(len(days)), "samsung_min": float(s["raw_close"].min()), "samsung_max": float(s["raw_close"].max()),
              "samsung_jumps_gt15pct": int((jump > np.log(1.16)).sum()), "samsung_jumps_gt30pct": int((jump > np.log(1.31)).sum())}
    kind = pd.read_sql("select code, delist_date from kind_delistings", con, parse_dates=["delist_date"])
    kind = kind[(kind["delist_date"] >= WINDOW[0]) & (kind["delist_date"] <= WINDOW[1]) & kind["code"].str.fullmatch(r"\d{6}")]
    kc = set(kind["code"].astype(str))
    kcc = {c for c in kc if c.endswith("0")}
    canary.update(kind_delist_2008_2016=len(kc), kind_in_v2=len(kc & set(codes)), kind_cover=round(len(kc & set(codes)) / max(len(kc), 1), 4),
                  kind_common=len(kcc), kind_common_cover=round(len(kcc & set(codes)) / max(len(kcc), 1), 4), v2_delisted_codes=len(delisted))
    canary["ok"] = bool(canary["samsung_jumps_gt30pct"] == 0 and canary["kind_cover"] >= 0.95 and canary["samsung_days"] >= 0.99 * len(days))

    # 실적 as-of 상태(실행기와 같은 known_table) → 종목×날짜 상태 배열(마지막 행 우선 = merge_asof 와 같다)
    kt = known_table(con, days, source="earnings_dart_doc")
    kt = kt[kt["code"].isin(ci_of)]
    n, T = len(codes), len(days)
    has_ni = np.full((n, T + 1), np.nan)
    has_both = np.full((n, T + 1), np.nan)
    is_q4 = np.full((n, T + 1), np.nan)
    ci = kt["code"].map(ci_of).to_numpy(int)
    kr = kt["krow"].to_numpy(int)
    ok_ni, ok_pr = kt["netinc"].notna().to_numpy(), kt["netinc_prior"].notna().to_numpy()
    q4 = kt["period"].str.endswith("Q4").to_numpy()
    for i in range(len(kt)):                        # krow 순 — 같은 (ci, krow) 는 뒤 행이 덮는다
        if kr[i] <= T:
            has_ni[ci[i], kr[i]] = ok_ni[i]
            has_both[ci[i], kr[i]] = ok_ni[i] and ok_pr[i]
            is_q4[ci[i], kr[i]] = q4[i]

    def ff(x):
        return pd.DataFrame(x[:, :T].T).ffill().to_numpy().T
    has_ni, has_both, is_q4 = ff(has_ni), ff(has_both), ff(is_q4)
    st = fiscal_status(codes)
    fy = {k: {y: v for y, v in d.items()} for k, d in st.items()}
    dl = np.array([c in delisted for c in codes])
    pref = np.array([not c.endswith("0") for c in codes])
    foreign = np.array([c.startswith("9") for c in codes])
    rows, pr_rows = [], []
    for y in YEARS:
        ty = np.asarray(days.year == y)
        for grp, gm in (("listed", ~dl), ("delisted", dl)):
            U = uni & gm[:, None] & ty[None, :]
            nd = np.array([fy[c].get(y) == "non_dec" for c in codes])
            noc = np.array([fy[c].get(y) in ("no_map", "not_run", "no_report") for c in codes])
            tot = int(U.sum())
            ex = int((U & nd[:, None]).sum())
            den = U & ~nd[:, None]
            d = int(den.sum())
            cov = int((den & (has_both == 1)).sum())
            ni_only = int((den & (has_ni == 1) & (has_both != 1)).sum())
            norow = int((den & ~(has_ni == 1)).sum())
            rows.append({"year": y, "group": grp, "stock_days": tot, "non_dec_days": ex, "non_dec_share": round(ex / tot, 4) if tot else None,
                         "denom": d, "covered": cov, "coverage": round(cov / d, 4) if d else None,
                         "prior_missing_share": round(ni_only / d, 4) if d else None, "no_row_share": round(norow / d, 4) if d else None,
                         "pref_share": round(float((den & pref[:, None]).sum() / d), 4) if d else None,
                         "foreign_share": round(float((den & foreign[:, None]).sum() / d), 4) if d else None,
                         "no_report_or_map_share": round(float((den & noc[:, None]).sum() / d), 4) if d else None,
                         "n_codes": int(den.any(1).sum()), "n_codes_non_dec": int((U & nd[:, None]).any(1).sum())})
            for qk, qm in (("Q4", is_q4 == 1), ("Q1-3", is_q4 == 0)):
                base = den & (has_ni == 1) & qm
                b = int(base.sum())
                pr_rows.append({"year": y, "group": grp, "latest_q": qk, "stock_days_with_netinc": b,
                                "prior_missing_share": round(float((base & (has_both != 1)).sum() / b), 4) if b else None})
    bars = []
    for y in YEARS:
        li = next(r for r in rows if r["year"] == y and r["group"] == "listed")
        de = next(r for r in rows if r["year"] == y and r["group"] == "delisted")
        cl, cd = li["coverage"] or 0.0, de["coverage"] or 0.0
        bars.append({"year": y, "listed": cl, "delisted": cd, "b_listed": cl >= BARS["listed"], "b_delisted": cd >= BARS["delisted"],
                     "b_gap": cd >= cl - BARS["gap_pp"], "pass": cl >= BARS["listed"] and cd >= BARS["delisted"] and cd >= cl - BARS["gap_pp"]})
    out = {"window": WINDOW, "adv_min": ADV_MIN, "bars": BARS, "canary": canary, "table": rows, "prior_missing": pr_rows, "by_year": bars,
           "all_pass": bool(all(b["pass"] for b in bars) and canary["ok"]),
           "dart_doc": {"rows": int(len(kt)), "codes": int(kt["code"].nunique()), "period_max": str(kt["period"].max()),
                        "known_after_window": int((kt["krow"] >= T).sum())}}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print("canary", json.dumps(canary, ensure_ascii=False))
    print("| 연도 | 구분 | 종목-일 | 비12월 제외 | 분모 | 커버 | 커버리지 | prior 결측 | 행 없음 | 우선주 | 보고서·매핑 없음 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['year']} | {r['group']} | {r['stock_days']:,} | {r['non_dec_share']:.1%} | {r['denom']:,} | {r['covered']:,} | {r['coverage']:.1%} | "
              f"{r['prior_missing_share']:.1%} | {r['no_row_share']:.1%} | {r['pref_share']:.1%} | {r['no_report_or_map_share']:.1%} |")
    print("| 연도 | 구분 | 최신 분기 | netinc 있는 종목-일 | prior 결측 |")
    print("|---|---|---|---|---|")
    for r in pr_rows:
        print(f"| {r['year']} | {r['group']} | {r['latest_q']} | {r['stock_days_with_netinc']:,} | {(r['prior_missing_share'] or 0):.1%} |")
    print("bars", json.dumps({"by_year": bars, "all_pass": out["all_pass"]}))


if __name__ == "__main__":
    main()
