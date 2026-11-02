"""ML 1단계 사후 진단 — 판정에 쓰지 않는다(S1 결과를 본 뒤 만들었다, 정정 ML-3). 새 신호·설정·유니버스를 평가하지 않는다.

S1 예측(캐시)을 그대로 읽어 ① 학습 행이 사전등록 기대보다 적은 이유, ② N2 널 폭이 §12 근사보다 좁은 이유, ③ 이음매 행 0 을 잰다.
먼저 N2 를 같은 키로 다시 돌려 ``S1.json`` 의 q95 와 같은지 본다. 출력 ``S1_diag.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import ml_forecast_common as C
import ml_forecast_data as D
import ml_forecast_ic as IC
import ml_forecast_nulls as NL
import numpy as np
from krx_quant_core.stats.metrics import newey_west_t
from ml_forecast_target import labels, tradable

OUT = Path(__file__).resolve().parents[1] / "logs" / "ml_forecast"
_C: dict = {}


def acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, np.float64) - np.mean(x)
    return float((x[lag:] @ x[:-lag]) / (x @ x))


def _null_series(r: int) -> tuple:
    """같은 키의 N2 — 모델별 (평균 IC, ICIR, 격자일 IC 의 lag 1 자기상관, 격자일 IC sd)."""
    c = _C
    pi = NL.n2_perm(c["u_star"], r, c["N"])
    out = {}
    for name in ("M", "E"):
        v = []
        for t in c["days"][name]:
            p, _ = NL.n2_assign(c["pred"][name][t], c["U"][t], pi, r, t, c["N"])
            ok = np.isfinite(p) & c["ok"][t]
            v.append(C.spearman(p[ok], c["y_u"][t][ok]))
        v = np.asarray(v)
        out[name] = (float(v.mean()), float(v.mean() / v.std(ddof=1)), acf(v, 1), float(v.std(ddof=1)))
    return r, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    window = C.assert_window(C.END, "eval")
    cache = Path(args.cache)
    s1 = json.loads((OUT / "S1.json").read_text())
    P = D.build_panels(C.END, cache)
    S = pickle.load(open(cache / "s1_preds.pkl", "rb"))
    preds, oos, U = S["preds"], S["oos"], S["U"]
    dates, vol = P["dates"], P["vol"]
    assert np.array_equal(P["codes"], S["codes"])
    fry = D.v4_factor_returns(P, D.label_mask(P))
    u_full = np.full_like(P["ret"], np.nan)
    u_full[:, P["first"]:] = fry.u
    y, _ = labels(u_full, vol, P["close"], P["delist_last"], U, P["seam"])
    res: dict = {"status": "사후 진단 — 판정에 쓰지 않는다", "s1_code_sha": s1["code_sha"]}

    # ① 학습 행
    fr = D.v4_factor_returns(P)
    rows = C.phase_rows(dates)
    tab = []
    for t in rows:
        if dates[t].year != 2019:
            continue
        m = P["msk"][:, t]
        ok = D.v4_ok(P, fr, t)
        idx = np.flatnonzero(m)
        top = idx[np.argsort(-P["cap"][idx, t], kind="stable")[:C.CAP_TOP]]
        tab.append({"date": str(dates[t].date()), "mask": int(m.sum()), "v4_ok": int(ok.sum()), "U": int(len(U[t])),
                    "U_if_no_D_condition": int((P["adv"][top, t] >= C.ADV_MIN_WON).sum())})
    t0_2020 = next(t for t in oos if dates[t].year == 2020)
    tr = [t for t in rows if t + C.LABEL_LAST <= t0_2020]
    res["train_2019"] = {"by_date": tab, "first_nonempty_U": next(x["date"] for x in tab if x["U"] > 0),
                         "phase_days_2020_fold": len(tr), "rows_U": int(sum(len(U[t]) for t in tr)),
                         "rows_U_if_no_D_condition": int(sum(x["U_if_no_D_condition"] for x in tab if x["date"] <= str(dates[tr[-1]].date())))}

    # ② 널 폭
    y_u = {t: y[t][U[t]] for t in oos}
    ok = {t: np.isfinite(y_u[t]) & tradable(vol, t)[U[t]] for t in oos}
    ics = {m: IC.ic_series(preds[m], y, U, vol, oos, window) for m in preds}
    u_star = np.unique(np.concatenate([U[t] for t in oos]))
    _C.update(u_star=u_star, N=len(P["codes"]), days={m: sorted(ics[m]) for m in ics}, pred=preds, U=U, y_u=y_u, ok=ok)
    with mp.get_context("fork").Pool(16) as pool:
        got = pool.map(_null_series, range(C.N2_REPS), chunksize=8)
    res["null_width"] = {}
    for m in ("M", "E"):
        a = np.array([g[1][m] for g in sorted(got, key=lambda g: g[0])])
        real = np.array([ics[m][t] for t in sorted(ics[m])])
        q = NL.q95(a[:, 0])
        assert q == s1["null_n2"][m]["q95_mean"], f"{m} 널 재현 실패 {q} ≠ {s1['null_n2'][m]['q95_mean']}"
        res["null_width"][m] = {
            "q95_reproduced_exactly": True, "n2_q95_mean": q, "n2_sd_of_mean": float(a[:, 0].std(ddof=1)),
            "n2_mean_day_ic_sd": float(a[:, 3].mean()), "n2_mean_lag1_acf_of_ic": float(a[:, 2].mean()),
            "real_day_ic_sd": float(real.std(ddof=1)), "real_lag_acf": {str(k): acf(real, k) for k in (1, 2, 3, 4, 6, 8)},
            "real_mean_nw6_t": newey_west_t(real, 6)[1], "real_mean_nw12_t": newey_west_t(real, 12)[1],
            "prereg_sec12_q95_approx": float(1.645 * real.std(ddof=1) * 2 / np.sqrt(real.size)),
        }
    n1 = np.array(s1["n1"]["means"])
    res["n1_width"] = {"sd_of_means": float(n1.std(ddof=1)), "implied_q95_if_mean0": float(1.645 * n1.std(ddof=1)), "reps": int(n1.size)}

    # ③ 이음매
    seam = P["seam"]
    in_star = np.zeros(len(P["codes"]), bool)
    in_star[u_star] = True
    oos_lo, oos_hi = oos[0] + 2, oos[-1] + C.LABEL_LAST
    res["seam"] = {"cells_panel": int(seam.sum()), "cells_oos_label_span": int(seam[:, oos_lo: oos_hi + 1].sum()),
                   "cells_oos_span_in_U_star_codes": int(seam[in_star, oos_lo: oos_hi + 1].sum())}
    (OUT / "S1_diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
