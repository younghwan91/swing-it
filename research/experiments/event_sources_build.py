"""16세대 원천 캐시 — 가족 캐시(``--base``, v11: 쐐기 + P·C) + 원천 S 표(:data:`event_sources.TID_S`) → ``--out``.

쐐기·가격 배열은 원본에 심볼릭 링크로 공유하고 ``F``·``I``·``off``·``XF``·``meta`` 만 새로 쓴다. 성과는 계산하지 않는다
(원천·연도별 사건 수만 meta 에 기록). 정의는 :mod:`research.signals.event_sources` 문서.

사용: ``PATTERN_GA_VARIANT=fwm uv run --with numba --extra pg python research/experiments/event_sources_build.py
--base <v11 캐시> --out <v13 캐시>``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE)), os.path.join(os.path.dirname(os.path.dirname(HERE)), "src")]

import event_families_build as FB  # noqa: E402
import pattern_ga_run as G  # noqa: E402

from research.signals import event_families as EF  # noqa: E402
from research.signals import event_sources as ES  # noqa: E402
from research.signals import pattern_events as PE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    import vcp_retest_run as R  # noqa: PLC0415

    from swing_it.features.catalysts import read_dart  # noqa: PLC0415
    from swing_it.features.technical import moving_average  # noqa: PLC0415

    E = G.Engine(a.base)
    if not E.fam_on or E.n_fam != EF.N_FAM_TABLES:
        raise SystemExit("기준은 13세대 가족 캐시(쐐기 + P·C)여야 한다")
    ctx = R.Ctx(t0)
    days, codes = ctx.days, ctx.codes
    if not (len(days) == len(E.days) and (days == E.days).all() and list(map(str, codes)) == list(map(str, E.codes))):
        raise SystemExit("기준 캐시와 날짜·종목이 다르다")
    G._log("문맥 적재", t0)
    nd, nc = len(days), len(codes)  # noqa: F841
    A = ctx.A
    O_, H_, L_, C_ = (np.asarray(x, float) for x in (A.O, A.H, A.L, A.C))          # 종목×날짜
    TR = np.asarray(E.TR)
    TV = np.ascontiguousarray(ctx.p.trade_value.reindex(index=days, columns=codes).to_numpy(float).T)
    close = ctx.p.close.reindex(index=days, columns=codes)
    m150, m200 = moving_average(close, 150).to_numpy(float), moving_average(close, 200).to_numpy(float)
    uni = ctx.universe(G.ADV_FLOOR).to_numpy(bool)
    with np.errstate(invalid="ignore"):                                            # P·C 와 같은 맥락(event_families_build)
        base = (uni & (ctx.ma50 > m150) & (m150 > m200) & ~ctx.split["recent"]).T
    base &= TR.astype(bool) & (np.nan_to_num(TV) > 0)
    base[:, :21] = False
    base[:, -1] = False
    tvd = pd.DataFrame(np.nan_to_num(TV).T)
    avg = tvd.shift(1).rolling(20, min_periods=20).mean().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        tvx = np.where(avg > 0, np.nan_to_num(TV) / np.where(avg > 0, avg, 1.0), np.nan)
        cp = np.concatenate([np.full((nc, 1), np.nan), C_[:, :-1]], axis=1)
        ret = C_ / cp - 1.0
        rng = H_ - L_
        clv = np.where(rng > 0, (C_ - L_) / np.where(rng > 0, rng, 1.0), np.where(C_ >= cp, 1.0, 0.0))
    fl = FB.flows(ctx, days, codes)                                                # 날짜×종목
    MA20 = np.asarray(E.MA20, float)
    VOL = np.asarray(ctx.VOL, float)
    G._log("배열", t0)
    t = ES.build(O_, H_, L_, C_, VOL, TV, MA20, base, tvx, ret, clv, fl[5].T, fl[20].T, read_dart(), days, codes)
    ci, s = t["ci"].to_numpy(np.int64), t["s"].to_numpy(np.int64)
    o = np.lexsort((-np.nan_to_num(tvx[ci, s]), s))                               # s 순 → 같은 날 배수 큰 순
    t, ci, s = t.iloc[o].reset_index(drop=True), ci[o], s[o]
    n = len(t)
    G._log(f"S 사건 {n:,} " + json.dumps(t["kind"].map(dict(enumerate(ES.KINDS))).value_counts().to_dict()), t0)
    XF = np.full((n, len(EF.X_COLS)), np.nan)
    XF[:, EF.XI["fam"]] = 2.0
    XF[:, EF.XI["tvx"]], XF[:, EF.XI["ret"]], XF[:, EF.XI["clv"]] = tvx[ci, s], ret[ci, s], clv[ci, s]
    sm1 = np.maximum(s - 1, 0)
    XF[:, EF.XI["fl5_s"]], XF[:, EF.XI["fl20_s"]] = fl[5][s, ci], fl[20][s, ci]
    XF[:, EF.XI["fl5_p"]], XF[:, EF.XI["fl20_p"]] = fl[5][sm1, ci], fl[20][sm1, ci]
    XF[:, EF.XI["s_kind"]] = t["kind"].to_numpy(float)
    XF[:, EF.XI["s_lvl"]], XF[:, EF.XI["s_stp"]] = t["lvl"].to_numpy(float), t["stp"].to_numpy(float)
    Ff = FB.f_rows(ctx, E, ci, s, fl, O_, H_, L_, C_)
    II = PE.II
    If = np.zeros((n, len(PE.I_COLS)), np.int64)
    If[:, II["d"]] = s + 1
    If[:, II["ci"]] = ci
    for k_ in ("i0", "iA", "iB", "iC", "iE", "iS"):
        If[:, II[k_]] = s
    If[:, II["iD"]] = -(3 + t["kind"].to_numpy(np.int64))                          # pid = s·10⁵ − (3 + 종류): P −1 · C −2 와 겹치지 않는다
    If[:, II["table"]] = ES.TID_S
    G._log("행", t0)
    os.makedirs(a.out, exist_ok=True)
    for k_ in G.ARR_NAMES:
        if k_ in ("F", "I", "off"):
            continue
        src, dst = os.path.realpath(os.path.join(a.base, f"{k_}.npy")), os.path.join(a.out, f"{k_}.npy")
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    F0, I0, off0 = np.asarray(E.F_orig), np.asarray(E.I), np.asarray(E.off)
    XF0 = np.asarray(E.XF)
    XF0 = np.hstack([XF0, np.full((len(XF0), len(EF.X_COLS) - XF0.shape[1]), np.nan)])
    off = np.concatenate([off0, [off0[-1] + n]]).astype(off0.dtype)
    np.save(os.path.join(a.out, "F.npy"), np.ascontiguousarray(np.vstack([F0, Ff])), allow_pickle=False)
    np.save(os.path.join(a.out, "I.npy"), np.ascontiguousarray(np.vstack([I0, If.astype(I0.dtype)])), allow_pickle=False)
    np.save(os.path.join(a.out, "off.npy"), off, allow_pickle=False)
    np.save(os.path.join(a.out, "XF.npy"), np.vstack([XF0, XF]), allow_pickle=False)
    meta = dict(E.meta)
    yr = days[s].year
    kn = t["kind"].to_numpy()
    meta["sources"] = {"version": 1, "fam_hash": FB.fam_hash(), "base": os.path.abspath(a.base), "n_S": n, "tid": ES.TID_S,
                       "kinds": list(ES.KINDS),
                       "count_by_year": {k: {int(y): int(v) for y, v in pd.Series(yr[kn == i]).value_counts().sort_index().items()}
                                         for i, k in enumerate(ES.KINDS)}}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)
    G._log(f"끝 — S {n:,} → {a.out}", t0)
    print(json.dumps(meta["sources"]["count_by_year"], ensure_ascii=False))


if __name__ == "__main__":
    main()
