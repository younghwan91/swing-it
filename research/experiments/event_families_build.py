"""13세대 사건 가족 캐시 — 기존 쐐기 캐시(``--base``) + P·C 사건 행 → ``--out`` (EVENT_FAMILIES.md 정정 E-1).

쐐기 배열은 심볼릭 링크로 공유하고 ``F``·``I``·``off``·``meta`` 만 새로 쓴다. 가족 서술자는 ``XF.npy``. 성과는 계산하지 않는다
(사건 수만 meta 에 기록).

사용: ``PATTERN_GA_VARIANT=fwm uv run --with numba --extra pg python research/experiments/event_families_build.py
--base <v10 캐시> --out <v11 캐시>``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE)), os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")]

import pattern_ga_run as G  # noqa: E402

from research.signals import event_families as EF  # noqa: E402
from research.signals import pattern_events as PE  # noqa: E402


def fam_hash() -> str:
    h = hashlib.sha256()
    for f in (EF.__file__, __file__):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def flows(ctx, days, codes) -> dict:
    """(기관+외국인) 순매수 수량 N일 합 / 거래량(acc_trde_qty) N일 합 — :func:`pattern_ga_run._fill_flows` 와 같은 정의(날짜×종목)."""
    q = pd.read_sql("SELECT code, date, foreign_, institution, acc_trde_qty FROM supply_demand", ctx.con, parse_dates=["date"])
    q["code"] = q["code"].astype(str)
    q = q[q["code"].isin(set(map(str, codes)))]

    def wide(col):
        return q.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=codes)
    smart = wide("foreign_").add(wide("institution"), fill_value=np.nan)
    vol = wide("acc_trde_qty")
    out = {}
    for n_ in (5, 20):
        mp_ = int(0.8 * n_)
        v = vol.rolling(n_, min_periods=mp_).sum()
        v = v.where(v > 0)
        out[n_] = (smart.rolling(n_, min_periods=mp_).sum() / v).to_numpy(float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    import vcp_retest_run as R  # noqa: PLC0415

    from swing_it.features.catalysts import read_dart  # noqa: PLC0415
    from swing_it.features.technical import moving_average  # noqa: PLC0415

    from research.signals import vcp as V  # noqa: PLC0415

    E = G.Engine(a.base)
    ctx = R.Ctx(t0)
    days, codes = ctx.days, ctx.codes
    if not (len(days) == len(E.days) and (days == E.days).all() and list(map(str, codes)) == list(map(str, E.codes))):
        raise SystemExit("기준 캐시와 날짜·종목이 다르다")
    G._log("문맥 적재", t0)
    nd, nc = len(days), len(codes)
    A = ctx.A
    O_, H_, L_, C_ = (np.asarray(x, float) for x in (A.O, A.H, A.L, A.C))          # 종목×날짜
    TR = np.asarray(E.TR)
    TV = np.ascontiguousarray(ctx.p.trade_value.reindex(index=days, columns=codes).to_numpy(float).T)
    close = ctx.p.close.reindex(index=days, columns=codes)
    m150, m200 = moving_average(close, 150).to_numpy(float), moving_average(close, 200).to_numpy(float)
    uni = ctx.universe(G.ADV_FLOOR).to_numpy(bool)
    with np.errstate(invalid="ignore"):
        base = (uni & (ctx.ma50 > m150) & (m150 > m200) & ~ctx.split["recent"]).T          # 종목×날짜
    base &= TR.astype(bool) & (np.nan_to_num(TV) > 0)
    base[:, :21] = False
    base[:, -1] = False
    # --- P: 느슨한 상위 집합(유전자 범위의 아래 끝) ---
    tvd = pd.DataFrame(np.nan_to_num(TV).T)
    avg = tvd.shift(1).rolling(20, min_periods=20).mean().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        tvx = np.where(avg > 0, np.nan_to_num(TV) / np.where(avg > 0, avg, 1.0), np.nan)
        cp = np.concatenate([np.full((nc, 1), np.nan), C_[:, :-1]], axis=1)
        ret = C_ / cp - 1.0
        rng = H_ - L_
        clv = np.where(rng > 0, (C_ - L_) / np.where(rng > 0, rng, 1.0), np.where(C_ >= cp, 1.0, 0.0))
    mx60 = pd.DataFrame(H_.T).shift(1).rolling(60, min_periods=60).max().to_numpy().T
    with np.errstate(invalid="ignore"):
        pm = base & (tvx >= EF.P_MIN["tvx"]) & (ret >= EF.P_MIN["ret"]) & (clv >= EF.P_MIN["clv"]) & (C_ >= mx60)
    pc, ps = np.nonzero(pm)
    G._log(f"P 사건 {len(pc):,}", t0)
    # --- C: DART 공급계약·잠정실적 → s = 접수일 이상 첫 거래일 ---
    dk = EF.dart_kinds(read_dart())
    pos = days.searchsorted(pd.DatetimeIndex(dk["rcept_dt"]), side="left")
    cmap = {str(c): i for i, c in enumerate(codes)}
    dk = dk.assign(s=pos, ci=dk["code"].astype(str).map(cmap))
    dk = dk[(dk["s"] < nd) & dk["ci"].notna()]
    dk = dk.groupby(["ci", "s"], as_index=False).agg(contract=("contract", "max"), prelim=("prelim", "max"), n=("contract", "size"))
    cc_, cs_ = dk["ci"].to_numpy(np.int64), dk["s"].to_numpy(np.int64)
    keep = base[cc_, cs_]
    dk, cc_, cs_ = dk[keep], cc_[keep], cs_[keep]
    G._log(f"C 사건 {len(cc_):,}", t0)
    # --- 합치기(P 먼저, C 뒤) — 표마다 d 순 정렬 ---
    fam = np.concatenate([np.zeros(len(pc)), np.ones(len(cc_))])
    ci = np.concatenate([pc, cc_]).astype(np.int64)
    s = np.concatenate([ps, cs_]).astype(np.int64)
    o = np.lexsort((-np.nan_to_num(tvx[ci, s]), s, fam))                            # 표 안: s 순 → 같은 날 강도(배수) 큰 순
    fam, ci, s = fam[o], ci[o], s[o]
    con_ = np.concatenate([np.zeros(len(pc)), dk["contract"].to_numpy(float)])[o]
    pre_ = np.concatenate([np.zeros(len(pc)), dk["prelim"].to_numpy(float)])[o]
    cn_ = np.concatenate([np.zeros(len(pc)), dk["n"].to_numpy(float)])[o]
    n = len(ci)
    desc = EF.price_descriptors(O_, H_, L_, C_, TV, s, ci, float(G.LIMIT_UP_RATIO))
    fl = flows(ctx, days, codes)
    sm1 = np.maximum(s - 1, 0)
    XF = np.full((n, len(EF.X_COLS)), np.nan)
    for k, v in desc.items():
        XF[:, EF.XI[k]] = v
    XF[:, EF.XI["fam"]] = fam
    XF[:, EF.XI["fl5_s"]], XF[:, EF.XI["fl20_s"]] = fl[5][s, ci], fl[20][s, ci]
    XF[:, EF.XI["fl5_p"]], XF[:, EF.XI["fl20_p"]] = fl[5][sm1, ci], fl[20][sm1, ci]
    XF[:, EF.XI["c_contract"]], XF[:, EF.XI["c_prelim"]], XF[:, EF.XI["c_n"]] = con_, pre_, cn_
    G._log("서술자", t0)
    # --- F·I 행(쐐기 표 열 의미에 맞춰) ---
    FI, II = PE.FI, PE.II
    Ff = np.full((n, len(PE.F_COLS)), np.nan)
    d1 = np.minimum(s + 1, nd - 1)
    hs, ls = H_[ci, s], L_[ci, s]
    mk = ctx.mkt[ci].astype(np.int64)
    tick = PE.tick_dated_vec(hs, d1, mk, E.reform)
    for k_, v in (("level", hs), ("order", hs + tick), ("h0", hs), ("u_d", hs), ("su", np.zeros(n)), ("E", ls), ("sl", np.zeros(n)),
                  ("hE", hs), ("stopE_raw", ls - PE.tick_dated_vec(ls, d1, mk, E.reform)), ("open_d", O_[ci, d1]),
                  ("high_d", H_[ci, d1]), ("low_d", L_[ci, d1]), ("close_d", C_[ci, d1]), ("close_s", C_[ci, s]),
                  ("adv20", ctx.adv[s, ci]), ("rs", ctx.rs[s, ci]), ("breadth", ctx.br[s, ci]),
                  ("tt_s", ctx.tt[s, ci].astype(float)), ("market", mk.astype(float)),
                  ("strat", np.asarray(E.strat_cs)[ci, s].astype(float)), ("month", np.asarray(E.month_of)[s].astype(float)),
                  ("atr_s", np.asarray(E.ATR)[ci, s]), ("fl_smart5", fl[5][s, ci]), ("fl_smart20", fl[20][s, ci])):
        Ff[:, FI[k_]] = v
    Ee = V.e_accel_asof(ctx.est, codes[ci], days[s])
    Ff[:, FI["e_q"]], Ff[:, FI["e_q_prev"]] = Ee["e_netinc"].to_numpy(float), Ee["e_netinc_prior"].to_numpy(float)
    Ff[:, FI["e_yoy"]], Ff[:, FI["e_yoy_prev"]] = Ee["e_yoy"].to_numpy(float), Ee["e_yoy_prev"].to_numpy(float)
    rg = pd.read_csv(G.REGIME_CSV, usecols=["date", "market", "breadth_ma200", "nh_nl_ratio", "breakout_success20"])
    rg = rg[rg["market"].isin(["KOSPI", "KOSDAQ"])].assign(date=lambda x: pd.to_datetime(x["date"]))
    key = pd.DataFrame({"date": days[s], "market": np.where(mk == 1, "KOSDAQ", "KOSPI")})
    mg = key.merge(rg, on=["date", "market"], how="left")
    Ff[:, FI["mkt_br200"]], Ff[:, FI["mkt_nhnl"]] = mg["breadth_ma200"].to_numpy(float), mg["nh_nl_ratio"].to_numpy(float)
    Ff[:, FI["mkt_bos"]] = mg["breakout_success20"].to_numpy(float)
    If = np.zeros((n, len(PE.I_COLS)), np.int64)
    If[:, II["d"]] = s + 1
    If[:, II["ci"]] = ci
    for k_ in ("i0", "iA", "iB", "iC", "iD", "iE", "iS"):
        If[:, II[k_]] = s
    If[:, II["iD"]] = -(fam.astype(np.int64) + 1)                                   # pid = s·10⁵ − (가족 + 1): (종목, s, 가족) 하나
    If[:, II["table"]] = np.where(fam == 0, EF.TID_P, EF.TID_C)
    # --- 쓰기 ---
    os.makedirs(a.out, exist_ok=True)
    for k_ in G.ARR_NAMES:
        if k_ in ("F", "I", "off"):
            continue
        src, dst = os.path.abspath(os.path.join(a.base, f"{k_}.npy")), os.path.join(a.out, f"{k_}.npy")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    F0, I0, off0 = np.asarray(E.F_orig), np.asarray(E.I), np.asarray(E.off)
    n_p = int((fam == 0).sum())
    off = np.concatenate([off0, [off0[-1] + n_p, off0[-1] + n]]).astype(off0.dtype)
    np.save(os.path.join(a.out, "F.npy"), np.ascontiguousarray(np.vstack([F0, Ff])), allow_pickle=False)
    np.save(os.path.join(a.out, "I.npy"), np.ascontiguousarray(np.vstack([I0, If.astype(I0.dtype)])), allow_pickle=False)
    np.save(os.path.join(a.out, "off.npy"), off, allow_pickle=False)
    np.save(os.path.join(a.out, "XF.npy"), XF, allow_pickle=False)
    yr = days[s].year
    meta = dict(E.meta)
    meta["families"] = {"version": 1, "fam_hash": fam_hash(), "base": os.path.abspath(a.base), "n_P": n_p, "n_C": int(n - n_p),
                        "tid": [EF.TID_P, EF.TID_C], "x_cols": list(EF.X_COLS), "p_min": EF.P_MIN,
                        "count_by_year": {f: {int(y): int(v) for y, v in pd.Series(yr[fam == k]).value_counts().sort_index().items()}
                                          for k, f in ((0, "P"), (1, "C"))}}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)
    G._log(f"끝 — P {n_p:,} · C {n - n_p:,} → {a.out}", t0)
    print(json.dumps(meta["families"]["count_by_year"], ensure_ascii=False))


if __name__ == "__main__":
    main()
