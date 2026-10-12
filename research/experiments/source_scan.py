"""16세대 사건 원천 검사(META_EVOLUTION.md 16세대 — 결과 전 고정): GA 전에 원천 자체에 평균 우위가 있는지.

원천마다 같은 메뉴(``PATTERN_GA_MENU``)에서 무작위 설정 ``--n`` 개를 뽑고(원천 무관 같은 벡터), 그 원천 가족만 켜서 폴드 1~4 시험창
(이어 붙인 2020~2023, 봉인 2024-01 앞)을 돌린다. 거래 지표 = 슬롯 대비 순 2배 비용 + 모호 p 혼합 + 소형주 미끄러짐(:mod:`evo_judge`).

판정 단위 = 사건(종목, 주문일): 그 사건을 거래한 설정들의 평균 → 주문월 군집 부트스트랩. 원천 = ctx(맥락 기준선)·pb·tight·earn·
accum·retest·corp(S) · P · C.

사용: ``PATTERN_GA_VARIANT=fwm PATTERN_GA_MENU="$(cat menu_g16s.json)" uv run --with numba --extra pg python
research/experiments/source_scan.py --cache <v13> --n 200 --out <dir>``
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path[:0] = [".", "src"]

SEED = 20261016
#: 16b(META_EVOLUTION.md 16b — 결과 전 고정, 미너비니 세 책 정독 근거): 진입 = 피벗(원천 lvl) + 1호가 매수정지, 시가가 피벗보다
#: 3% 넘게 뜨면 안 산다(TLW p239·MM p94 "a few percent"), 추세 템플릿 ∧ RS ≥ 70(TLW p94). 손절 = 원천 손절선(박스 저점) − 1호가,
#: 7.5% 상한(= 평균 이익 15% 의 절반, TLW p314 · MM p98), 비중으로 슬롯 손실 ≤ 5%(사용자). +2R 장중에 절반 매도 + 본전(free roll,
#: TTC p163). 나머지 = MA50 종가 이탈, MA50 ≥ 진입가가 된 뒤부터(TTC p162). 급등 매도(근사, TTC p156) = 5일 +25% ∨ 당일 +15%,
#: +3R 위, 종가. 불타기 없음·첫 진입 100%·시장 필터 없음(숫자 규칙 없음)·시간 손절 없음(엔진 120일 데이터 상한만).
MINERVINI = {"fam_w": 0, "mkt": "off", "atr_max": None, "tt_req": 1, "stop_max": 0.075, "gap_max": 0.03, "stop_policy": "clamp",
             "be_r": 2.0, "be_basis": "high", "lock_to": None, "take_r": 2.0, "take_frac": 0.5, "trail": "ma50",
             "trail_basis": "close", "trail_start": 0.0, "trail_gate": 1, "exit_safe": 0, "tighten": 0, "trail_k": None,
             "pyramid": "off", "climax": 1, "climax_basis": "close", "climax_day": 0.15, "climax_5d": 0.25, "climax_r": 3.0,
             "stop_basis": "intraday", "fail_exit": None, "reentry": "off", "first_frac": 1.0, "pyr_close": 0,
             "s_entry": "stop", "s_vmult": None, "s_flow": None,
             "p_entry": "stop", "p_vmult": 2.0, "p_ret": 0.02, "p_clv": 0.5, "p_hiw": 60, "p_flow": None, "p_hold": None,
             "c_kind": "both", "c_entry": "stop", "c_ret": None, "c_vmult": None, "c_flow": None}
SOURCES = ("ctx", "pb", "tight", "earn", "accum", "retest", "corp", "P", "C")
N_BOOT = 2000
_W: dict = {}


def _init(cache: str) -> None:
    import numpy as np

    import research.experiments.pattern_ga_run as G
    from swing_it.validation.walkforward import FOLDS
    E = G.Engine(cache)
    wins = G.fold_windows(E, [FOLDS[k] for k in G.EVO_FOLDS])
    f_last = FOLDS[G.EVO_FOLDS[-1]]
    seal = E.row(f_last.test_hi)
    E.LIM = np.minimum(np.asarray(E.LIM), seal).astype(np.int64)
    E.end_row = seal - 1
    tests = [te for _, te in wins]
    tests[-1] = G.Win(E, f_last.test_lo, f_last.test_hi, f_last.test_hi)
    _W.update(G=G, E=E, tests=tests)


def cfg_for(G, g: dict, src: str) -> dict:
    from research.signals import event_sources as ES
    g = dict(g, fam_w=0, fam_p=int(src == "P"), fam_c=int(src == "C"), fam_s=int(src in ES.K))
    if src in ES.K:
        g["s_kind"] = ES.K[src]
    return G.canonical(g)


def _task(a: tuple) -> list:
    import numpy as np
    from research.experiments.evo_judge import SLOT_W, SMALL_ADV, SMALL_SLIP
    i, src, g0, oseed = a
    G, E, tests = _W["G"], _W["E"], _W["tests"]
    cfg = cfg_for(G, g0, src)
    st = E.stitched([cfg] * len(tests), tests, order_seed=oseed)
    t = G.trade_frame(E, st, [cfg] * len(tests))
    if not len(t):
        return []
    amb = t["amb"].astype(bool)
    n2 = t["net2"].where(~amb, E.pmix * t["net2_stop0"] + (1 - E.pmix) * t["net2"])
    n2 = n2 - SMALL_SLIP * t["wb"] * (t["adv20"] < SMALL_ADV)
    n2s = n2 * np.minimum(1.0, t["alloc"] / SLOT_W)
    n1 = t["net"].where(~amb, E.pmix * t["net_stop0"] + (1 - E.pmix) * t["net"])        # 기술: 1배 비용(미끄러짐 추가 없음)
    n1s = n1 * np.minimum(1.0, t["alloc"] / SLOT_W)
    return [{"src": src, "i": i, "code": c, "order_day": d.strftime("%Y-%m-%d"), "n2s": float(v), "n1s": float(v1), "R": float(r_),
             "hold": int(h), "reason": rs, "exit_day": x.strftime("%Y-%m-%d"), "e": float(e_), "sp": float(sp_)}
            for c, d, v, v1, r_, h, rs, x, e_, sp_ in zip(t["code"], t["order_day"], n2s, n1s, t["real_R_post"], t["hold_days"],
                                                         t["exit_reason"], t["exit_day"], t["entry_px"], t["stop_px"], strict=True)]


def boot(x, cl, rng):
    """군집 부트스트랩 평균의 SE(군집 = 주문월)."""
    import numpy as np
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, weights=x)
    n = np.bincount(inv)
    idx = rng.integers(0, len(u), (N_BOOT, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(x.mean()), float(m.std(ddof=1))


def summarize(tr, n_cfg: int) -> "tuple[dict, object]":
    import numpy as np
    rng = np.random.default_rng(SEED)
    ev = tr.groupby(["src", "code", "order_day"], as_index=False).agg(n2s=("n2s", "mean"), n1s=("n1s", "mean"), k=("i", "nunique"))
    ev["month"] = ev["order_day"].str[:7]
    ev["year"] = ev["order_day"].str[:4].astype(int)
    out = {}
    base = ev[ev["src"] == "ctx"]
    bm = base.groupby("month")["n2s"].mean()
    for src, x in ev.groupby("src"):
        m, se = boot(x["n2s"].to_numpy(), x["month"].to_numpy(), rng)
        cm = tr[tr["src"] == src].groupby("i")["n2s"].agg(["mean", "size"])
        # 기준선 대비: 월 평균 차(두 원천 모두 사건이 있는 달)의 군집(월) 부트스트랩
        xm = x.groupby("month")["n2s"].mean()
        j = xm.index.intersection(bm.index)
        dd = (xm[j] - bm[j]).to_numpy()
        db = rng.integers(0, len(dd), (N_BOOT, len(dd))) if len(dd) else None
        dse = float(dd[db].mean(1).std(ddof=1)) if len(dd) > 2 else float("nan")
        yr = x.groupby("year")["n2s"].mean()
        out[src] = {"events": int(len(x)), "events_per_year": len(x) / 4.0, "mean_pct": m * 100, "t": m / se if se else float("nan"),
                    "win": float((x["n2s"] > 0).mean()), "mean_1x_pct": float(x["n1s"].mean() * 100),   # 기술(바 아님)
                    "vs_ctx_pct": float(dd.mean() * 100) if len(dd) else float("nan"),
                    "t_vs_ctx": float(dd.mean() / dse) if len(dd) > 2 and dse else float("nan"),
                    "years_pos": int((yr > 0).sum()), "by_year_pct": {int(k): float(v * 100) for k, v in yr.items()},
                    "cfg_with_trades": int(len(cm)), "cfg_mean_median_pct": float(cm["mean"].median() * 100) if len(cm) else float("nan"),
                    "cfg_frac_pos": float((cm["mean"] > 0).mean()) if len(cm) else float("nan"),
                    "trades_per_cfg_year": float(cm["size"].mean() / 4.0) if len(cm) else 0.0}
        r = out[src]
        r["pass"] = bool(r["events"] >= 240 and r["mean_pct"] >= 0.5 and r["t"] >= 2.0 and src != "ctx"
                         and r["vs_ctx_pct"] >= 0.5 and r["t_vs_ctx"] >= 1.5 and r["years_pos"] >= 3)
    return out, ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--fixed", default="", choices=("", "minervini"), help="16b: 고정 설정 하나(--n 은 같은 날 순서 seed 수)")
    ap.add_argument("--mkt", default="off", help="16b 기술 변형: 시장 국면 유전자")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    os.environ.setdefault("PATTERN_GA_VARIANT", "fwm")
    import research.experiments.pattern_ga_run as G
    t0 = time.time()
    X = G.random_x(np.random.default_rng(SEED), a.n)
    cfgs = [G.decode(x) for x in X]
    srcs = a.sources.split(",")
    if a.fixed:                                       # 16b: 같은 설정 · 같은 날 순서 seed i(슬롯 제한 아래 다른 사건 표본)
        fx = {**cfgs[0], **MINERVINI, "mkt": a.mkt}
        tasks = [(i, s, fx, SEED + i) for s in srcs for i in range(a.n)]
        cfgs = [fx]
    else:
        tasks = [(i, s, c, None) for s in srcs for i, c in enumerate(cfgs)]
    rows = []
    with cf.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.cache,)) as ex:
        for r in ex.map(_task, tasks, chunksize=4):
            rows.extend(r)
    os.makedirs(a.out, exist_ok=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join(a.out, "trades.csv.gz"), index=False)
    summ, ev = summarize(tr, a.n)
    ev.to_csv(os.path.join(a.out, "events.csv.gz"), index=False)
    json.dump({"args": vars(a), "seed": SEED, "summary": summ, "elapsed": time.time() - t0,
               "cfgs": [cfg_for(G, c, "ctx") for c in cfgs[:3]]},
              open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    pd.set_option("display.width", 220)
    df = pd.DataFrame(summ).T.drop(columns=["by_year_pct"])
    print(df.round(3).to_string())
    print({k: {y: round(v, 2) for y, v in s["by_year_pct"].items()} for k, s in summ.items()})


if __name__ == "__main__":
    main()
