"""28세대 지수 편입 예측(META_EVOLUTION.md 28세대 — 결과 전 고정). 발표 전 공개 데이터만으로 KOSPI200·KOSDAQ150 정기변경 편입을 예측해
예측일(기준일 다음 거래일) 종가에 사고, 실제 편입이면 적용 전 거래일 종가, 아니면 발표 다음 거래일 종가에 판다.

예측 = 직전 정기 구성 목록 비구성 ∧ 기준일까지 6개월 평균 시총 순위 ≤ 0.7 × 지수 크기 ∧ 유효일 ≥ 100 ∧ 평균 거래대금 ≥ 10억.
대조 = 점-시점 시장, 예측일 시총 순위 ±50 동료 평균(그 변경의 편입·편출·예측 종목 제외; 10 미만 → 시장 동일 가중).
순 = 수익 − 왕복 0.415% × 2 − 대조. 군집 = (지수, 적용일).

사용: ``uv run --extra pg python research/experiments/index28.py --out <dir>`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]
warnings.filterwarnings("ignore")

COST2 = 2 * 0.00415
SIZE = {"KOSPI200": 200, "KOSDAQ150": 150}
IDX_MKT = {"KOSPI200": "거래소", "KOSDAQ150": "코스닥"}
RANK_FRAC = 0.7
WIN = 120          # 기준일까지 평균 창(거래일, 약 6개월)
MIN_VALID = 100
MIN_TV = 1000.0    # 평균 거래대금(백만 원) = 10억


def load(con):
    from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg
    q = lambda s, **k: pd.read_sql(s, con, **k)  # noqa: E731
    from swing_it.storage import read_prices
    px = read_prices(con, cols=("code", "date", "open", "close"), start="2016-05-01", parse_dates=True)
    px = patch_prices(px, read_session_bars_agg(con, start=NXT_START), adjusted=True)
    raw = read_prices(con, cols=("code", "date", "close", "trade_value"), table="daily_bars", start="2016-05-01", parse_dates=True)
    sh = q("select code, date, shares_outstanding, knowledge_date from shares_outstanding_history", parse_dates=["date", "knowledge_date"])
    stk = q("select code, name, market from stocks")
    dl = q("select code, name, market from delisted_stocks")
    ch = q("select index_name, effective_date, action, code, announce_date from index_changes", parse_dates=["effective_date", "announce_date"])
    cons = q("select index_name, effective_date, code from index_constituents", parse_dates=["effective_date"])
    return px, raw, sh, stk, dl, ch, cons


def cl_stats(s: pd.DataFrame, col: str = "excess_net") -> dict:
    s = s.dropna(subset=[col])
    if s.empty:
        return {}
    cl = s.groupby(["index", "eff"])[col].mean()
    cd = s.groupby("eff")[col].mean()
    se = cl.std(ddof=1) / np.sqrt(len(cl))
    sed = cd.std(ddof=1) / np.sqrt(len(cd))
    yr = pd.to_datetime(s["eff"]).dt.year
    srt = s[col].sort_values(ascending=False)
    return {"n": int(len(s)), "n_rebal": int(len(cl)), "mean_pct": float(s[col].mean() * 100), "median_pct": float(s[col].median() * 100),
            "cluster_mean_pct": float(cl.mean() * 100), "cluster_t": float(cl.mean() / se) if se > 0 else float("nan"),
            "date_t": float(cd.mean() / sed) if len(cd) > 1 and sed > 0 else float("nan"), "rebal_pos_share": float((cl > 0).mean()),
            "win": float((s[col] > 0).mean()), "early_2017_2020_pct": float(s[yr <= 2020][col].mean() * 100),
            "late_2021_2026_pct": float(s[yr >= 2021][col].mean() * 100), "drop_top10_mean_pct": float(srt.iloc[10:].mean() * 100),
            "by_year_pct": {int(k): float(v * 100) for k, v in s.groupby(yr)[col].mean().items()}}


def main() -> None:
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", default="a", choices=("a", "b", "c"))   # 정정 G28-1: a 원 종가×점-시점 주식수 · b 수정 종가×최신 주식수 · c a 단 a/b 3배 이탈 → b
    a = ap.parse_args()
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    px, raw, sh, stk, dl, ch, cons = load(con)
    C = px.pivot_table(index="date", columns="code", values="close", aggfunc="last").sort_index()
    days = C.index
    codes = C.columns
    R = raw.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days, columns=codes)
    TV = raw.pivot_table(index="date", columns="code", values="trade_value", aggfunc="last").reindex(index=days, columns=codes)
    sh["d"] = sh["knowledge_date"].fillna(sh["date"])
    S = sh.pivot_table(index="d", columns="code", values="shares_outstanding", aggfunc="last")
    S = S.reindex(index=S.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
    S_bf = S.bfill()                                    # 아직 기록 없는 날 = 첫 기록 뒤채움(건수 보고)
    CAP = R * S_bf
    if a.cap in ("b", "c"):
        CAP_b = C * S_bf.ffill().iloc[-1]
        CAP = CAP_b if a.cap == "b" else CAP.where(((CAP / CAP_b) < 3) & ((CAP_b / CAP) < 3), CAP_b)
    BF = (S.isna() & S_bf.notna()).to_numpy()
    Cf = C.ffill(limit=10).to_numpy(float)              # 창 안 상폐·거래정지 → 직전 종가(최대 10일)
    CAPv, TVv = CAP.to_numpy(float), TV.to_numpy(float)
    name = {**dl.set_index("code")["name"].to_dict(), **stk.set_index("code")["name"].to_dict()}
    mkt = {r.code: {"유가증권": "거래소"}.get(r.market, r.market) for r in dl.itertuples()}
    mkt.update(stk.set_index("code")["market"].to_dict())
    spac = np.array(["스팩" in str(name.get(c, "")) for c in codes])
    col = {c: i for i, c in enumerate(codes)}
    rows = []
    rebals = ch[(ch["action"] == "add") & ch["index_name"].isin(SIZE)].groupby(["index_name", "effective_date"])["announce_date"].min()
    for (idx, eff), ann in rebals.items():
        prev = cons[(cons["index_name"] == idx) & (cons["effective_date"] < eff)]
        if prev.empty:
            continue
        members = set(prev[prev["effective_date"] == prev["effective_date"].max()]["code"])
        # 점-시점 시장: 직전 목록 소속이 현재 구분보다 우선(코스닥 → 거래소 이전 종목)
        prev_all = cons[cons["effective_date"] < eff]
        k150 = set(prev_all[(prev_all["index_name"] == "KOSDAQ150") & (prev_all["effective_date"] == prev_all[prev_all["index_name"] == "KOSDAQ150"]["effective_date"].max())]["code"])
        k200 = set(prev_all[(prev_all["index_name"] == "KOSPI200") & (prev_all["effective_date"] == prev_all[prev_all["index_name"] == "KOSPI200"]["effective_date"].max())]["code"])
        mk = np.array(["코스닥" if c in k150 else "거래소" if c in k200 else mkt.get(c) for c in codes], dtype=object)
        base_month = (eff - pd.offsets.MonthBegin(1) - pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)   # 두 달 전 말일
        i_base = days.searchsorted(base_month, side="right") - 1
        i_p = i_base + 1
        i_ann = days.searchsorted(ann, side="left")
        i_ann1 = days.searchsorted(ann, side="right")
        i_out = days.searchsorted(eff, side="left") - 1
        if i_base - WIN + 1 < 0 or i_p >= i_ann or i_out <= i_ann1 or i_out >= len(days):
            continue
        w = slice(i_base - WIN + 1, i_base + 1)
        capw = CAPv[w]
        valid = np.isfinite(capw).sum(0)
        avgcap = np.where(valid >= MIN_VALID, np.nanmean(np.where(np.isfinite(capw), capw, np.nan), 0), np.nan)
        avgtv = np.nanmean(TVv[w], 0)
        univ = (mk == IDX_MKT[idx]) & ~spac & np.isfinite(avgcap)
        rk = pd.Series(avgcap[univ], index=codes[univ]).rank(ascending=False)
        mem_in = [c for c in members if c in rk.index]
        agree = float(np.mean([rk[c] <= SIZE[idx] for c in mem_in])) if mem_in else np.nan   # G28-1 선택 기준(수익 아님)
        cut = RANK_FRAC * SIZE[idx]
        pred = [c for c, r_ in rk.items() if r_ <= cut and c not in members and np.nan_to_num(avgtv[col[c]]) >= MIN_TV]
        actual = set(ch[(ch["index_name"] == idx) & (ch["effective_date"] == eff) & (ch["action"] == "add")]["code"])
        changed = set(ch[(ch["effective_date"] == eff) & ch["index_name"].isin(SIZE)]["code"])
        # 대조 동료: 예측일 시총(뒤채움 포함) 순위 ±50, 같은 점-시점 시장, 변경·예측 종목 제외
        capp = CAPv[i_p]
        pool = (mk == IDX_MKT[idx]) & ~spac & np.isfinite(capp)
        prank = pd.Series(capp[pool], index=codes[pool]).rank(ascending=False)
        excl = changed | set(pred)
        for c in pred:
            j = col[c]
            hit = c in actual
            i_x = i_out if hit else i_ann1
            p0, p1 = Cf[i_p, j], Cf[i_x, j]
            if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
                continue
            ret = p1 / p0 - 1
            me = prank.get(c, np.nan)
            peers = [col[k] for k, r_ in prank.items() if k not in excl and np.isfinite(me) and abs(r_ - me) <= 50]
            cr = Cf[i_x, peers] / Cf[i_p, peers] - 1 if peers else np.array([])
            cr = cr[np.isfinite(cr)]
            if len(cr) >= 10:
                ctrl, ctrl_kind = float(cr.mean()), "peer_mean"
            else:
                allm = np.where(pool)[0]
                mr = Cf[i_x, allm] / Cf[i_p, allm] - 1
                ctrl, ctrl_kind = float(np.clip(mr[np.isfinite(mr)], -0.9, 5).mean()), "market_ew"
            seg = {}
            if hit:
                for nm_, i0, i1 in (("p_to_ann", i_p, i_ann), ("ann_gap", i_ann, i_ann1), ("ann1_to_out", i_ann1, i_out)):
                    seg[nm_] = float(Cf[i1, j] / Cf[i0, j] - 1) if np.isfinite(Cf[i1, j] / Cf[i0, j]) else np.nan
            rows.append({"index": idx, "eff": str(eff.date()), "ann": str(ann.date()), "code": c, "name": name.get(c, ""),
                         "hit": hit, "rank": float(rk[c]), "entry": str(days[i_p].date()), "exit": str(days[i_x].date()),
                         "hold": int(i_x - i_p), "ret": ret, "ctrl": ctrl, "ctrl_kind": ctrl_kind, "n_peers": int(len(cr)),
                         "excess_net": ret - COST2 - ctrl, "shares_backfilled": bool(BF[i_base, j]), **seg})
        rows.append({"index": idx, "eff": str(eff.date()), "ann": str(ann.date()), "code": "_REBAL_", "n_pred": len(pred),
                     "n_actual": len(actual), "n_hit": len(set(pred) & actual), "agree": agree})
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    t.to_csv(os.path.join(a.out, "events.csv"), index=False)
    ev = t[t["code"] != "_REBAL_"].copy()
    ev["hit"], ev["shares_backfilled"] = ev["hit"].astype(bool), ev["shares_backfilled"].astype(bool)
    rb = t[t["code"] == "_REBAL_"]
    out = {"main": cl_stats(ev)}
    for idx in SIZE:
        out[idx] = cl_stats(ev[ev["index"] == idx])
        r_ = rb[rb["index"] == idx]
        out[idx]["precision"] = float(r_["n_hit"].sum() / max(r_["n_pred"].sum(), 1))
        out[idx]["recall"] = float(r_["n_hit"].sum() / max(r_["n_actual"].sum(), 1))
    out["main"]["precision"] = float(rb["n_hit"].sum() / max(rb["n_pred"].sum(), 1))
    out["main"]["recall"] = float(rb["n_hit"].sum() / max(rb["n_actual"].sum(), 1))
    out["hits"] = cl_stats(ev[ev["hit"]])
    out["misses"] = cl_stats(ev[~ev["hit"]])
    h = ev[ev["hit"]]
    out["hit_segments_mean_pct"] = {k: float(h[k].mean() * 100) for k in ("p_to_ann", "ann_gap", "ann1_to_out") if k in h}
    out["cap"] = a.cap
    out["member_agree_mean"] = float(rb["agree"].mean())
    out["n_shares_backfilled"] = int(ev["shares_backfilled"].sum())
    out["n_ctrl_market_fallback"] = int((ev["ctrl_kind"] == "market_ew").sum())
    out["excl_backfilled"] = cl_stats(ev[~ev["shares_backfilled"]])
    m = out["main"]
    out["pass"] = bool(m and m["mean_pct"] >= 1.0 and m["cluster_t"] >= 2 and m["rebal_pos_share"] >= 0.6
                       and m["early_2017_2020_pct"] > 0 and m["late_2021_2026_pct"] > 0 and m["drop_top10_mean_pct"] > 0)
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
