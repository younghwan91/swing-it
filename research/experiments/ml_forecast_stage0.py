"""ML 수익률 예측 0단계 관문 — **라벨 없이** 돈다(PREREG §8 (a)(b)(c), DESIGN §4-2). u·y 는 계산하지 않는다.

- (a) 절단 재계산 동일성: IC 격자에서 무작위 결정일 5개(``keyed_rng("ML-S0-trunc")``). 데이터를 그날까지 잘라
  :func:`ml_forecast_data.build_panels` 를 다시 부르고, 그날 열의 U(t)·원 피처·rank-gauss 피처가 전체 패널과 비트 단위로 같은지 본다.
- (b) 피처별 U(t) 커버리지 중앙값: F7 ≥ 25%, F1 은 2020~ 격자일 ≥ 70%, 나머지 ≥ 80%(IC 격자 전체).
- (c) 폐지 종목(``delisted_stocks``)의 피처 결측률이 상장 종목보다 20%p 넘게 높으면 실패.
- 보고만: U(t) 크기, v4 유니버스 재구성 대 core ``build_risk_model`` 종목 대조(3일), 이음매 가드 셀 수, 표 지문.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" --with lightgbm==4.7.0 \\
      python research/experiments/ml_forecast_stage0.py --cache <레포 밖>

출력: ``research/logs/ml_forecast/stage0.json``·``stage0.md``. 2024-01 이후 데이터는 읽지 않는다(END 2023-12-28).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ml_forecast_common import END, FEATURES, PREREG_SHA, assert_window, ic_grid, keyed_rng, phase_rows

OUT = Path(__file__).resolve().parents[1] / "logs" / "ml_forecast"
N_TRUNC = 5
N_V4_CHECK = 3
COV_BAR = {"tp_rev": 0.25, "eps_yoy": 0.70}
COV_DEFAULT = 0.80
DELIST_GAP = 0.20


def coverage_gate(cov: dict[str, list[float]], days: list[int], dates) -> dict:
    """(b) — F1 은 2020~ 격자일, 나머지는 IC 격자 전체의 중앙값."""
    out = {}
    y2020 = np.array([dates[t].year >= 2020 for t in days])
    for k in FEATURES:
        v = np.asarray(cov[k])
        sel = v[y2020] if k == "eps_yoy" else v
        med = float(np.nanmedian(sel))
        bar = COV_BAR.get(k, COV_DEFAULT)
        out[k] = {"median": med, "bar": bar, "pass": bool(med >= bar),
                  "median_all_grid": float(np.nanmedian(v)), "min": float(np.nanmin(v))}
    return out


def delist_gate(raw, U, days, is_del) -> dict:
    """(c) — U(t) 행(IC 격자일 전체)에서 폐지·상장 종목의 결측률."""
    out = {}
    for k in FEATURES:
        miss_d = tot_d = miss_l = tot_l = 0
        for t in days:
            idx = U[t]
            m = ~np.isfinite(raw[k][idx, t])
            d = is_del[idx]
            miss_d += int(m[d].sum())
            tot_d += int(d.sum())
            miss_l += int(m[~d].sum())
            tot_l += int((~d).sum())
        rd = miss_d / tot_d if tot_d else float("nan")
        rl = miss_l / tot_l if tot_l else float("nan")
        out[k] = {"miss_delisted": rd, "miss_listed": rl, "gap": rd - rl, "rows_delisted": tot_d,
                  "rows_listed": tot_l, "pass": bool(not (rd - rl > DELIST_GAP))}
    return out


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and bool(np.array_equal(np.isnan(a), np.isnan(b))) and \
        bool(np.array_equal(a[~np.isnan(a)].view(np.uint64), b[~np.isnan(b)].view(np.uint64)))


def truncation_check(P, raw, U, Z, d: int, cache: Path, log) -> dict:
    """(a) 한 결정일 — 그날까지 자른 패널에서 다시 낸 값과 비트 대조."""
    import ml_forecast_data as D
    import ml_forecast_features as FT

    day = str(P["dates"][d].date())
    Pd = D.build_panels(day, cache / "trunc", log)
    assert Pd["dates"][-1] == P["dates"][d]
    frd = D.v4_factor_returns(Pd)
    dd = len(Pd["dates"]) - 1
    Ud = D.universe(Pd, frd, [dd])[dd]
    rawd = FT.raw_features(Pd)
    full_codes = P["codes"][U[d]]
    trunc_codes = Pd["codes"][Ud]
    res = {"date": day, "U_same": bool(np.array_equal(full_codes, trunc_codes)), "U_n": int(len(Ud)),
           "codes_trunc": int(len(Pd["codes"])), "codes_full": int(len(P["codes"]))}
    common = pd.Index(P["codes"]).get_indexer(Pd["codes"])
    assert (common >= 0).all(), "잘린 패널에 전체 패널에 없는 종목이 있다"
    feats = {}
    for k in FEATURES:
        a_all = raw[k][common, d]
        b_all = rawd[k][:, dd]
        a_u = raw[k][U[d], d]
        b_u = rawd[k][Ud, dd] if res["U_same"] else None
        feats[k] = {"raw_all_codes_same": _same(a_all, b_all),
                    "raw_U_same": bool(res["U_same"] and _same(a_u, b_u))}
    if res["U_same"]:
        Zd = FT.transform(rawd, {dd: Ud})[dd]
        res["rank_gauss_same"] = _same(Z, Zd)
    else:
        res["rank_gauss_same"] = False
    res["features"] = feats
    res["pass"] = bool(res["U_same"] and res["rank_gauss_same"] and all(v["raw_all_codes_same"] for v in feats.values()))
    # 꼬리 정보 — 처음 어긋난 피처
    res["first_diff"] = next((k for k, v in feats.items() if not v["raw_all_codes_same"]), None)
    return res


def run(args, log) -> int:
    import ml_forecast_data as D
    import ml_forecast_features as FT

    assert_window(END, "eval")
    t_start = time.time()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    assert repo not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    fp0 = D.fingerprint_range("2017-01-01", END)
    P = D.build_panels(END, cache, log)
    dates = P["dates"]
    grid = ic_grid(dates)
    rows = phase_rows(dates)
    fr = D.v4_factor_returns(P)
    log(f"factor_returns(v4 마스크) 끝 — 격자 {len(grid)}일({dates[grid[0]].date()}~{dates[grid[-1]].date()}), 학습 위상 {len(rows)}일")
    U = D.universe(P, fr, rows)
    sizes = np.array([len(U[t]) for t in grid])
    raw = FT.raw_features(P)

    # v4 유니버스 재구성 대조(보고만이지만 어긋나면 유니버스 정의가 틀린 것 — 실패로 센다)
    rng = keyed_rng("ML-S0-v4")
    chk = sorted(int(t) for t in rng.choice(grid, N_V4_CHECK, replace=False))
    v4 = []
    for t in chk:
        a = np.flatnonzero(D.v4_ok(P, fr, t))
        b = D.build_risk_model_codes(P, t)
        v4.append({"date": str(dates[t].date()), "same": bool(np.array_equal(a, b)), "n": int(len(a))})
        log(f"  v4 대조 {dates[t].date()}: {v4[-1]}")

    # (b)(c)
    cov = FT.coverage(raw, U, grid)
    cov_gate = coverage_gate(cov, grid, dates)
    is_del = D.delisted_ever(P["codes"])
    del_gate = delist_gate(raw, U, grid, is_del)
    log("커버리지 " + " ".join(f"{k}:{v['median']:.2f}" for k, v in cov_gate.items()))

    # (a)
    rng = keyed_rng("ML-S0-trunc")
    tdays = sorted(int(t) for t in rng.choice(grid, N_TRUNC, replace=False))
    trunc = []
    for d in tdays:
        Z = FT.transform(raw, {d: U[d]})[d]
        r = truncation_check(P, raw, U, Z, d, cache, log)
        trunc.append(r)
        log(f"  절단 {r['date']}: {'같다' if r['pass'] else '다르다 — ' + str(r['first_diff'])}")
    fp1 = D.fingerprint_range("2017-01-01", END)

    gates = {"a_truncation": all(r["pass"] for r in trunc),
             "b_coverage": all(v["pass"] for v in cov_gate.values()),
             "c_delisted_missing": all(v["pass"] for v in del_gate.values()),
             "v4_universe_matches_core": all(r["same"] for r in v4),
             "fingerprint_unchanged": fp0 == fp1}
    res = {
        "prereg": PREREG_SHA, "end": END, "core_ref": args.core_ref,
        "grid": {"ic_days": len(grid), "first": str(dates[grid[0]].date()), "last": str(dates[grid[-1]].date()),
                 "oos_days": int(sum(dates[t].year >= 2020 for t in grid)), "train_phase_days": len(rows)},
        "universe": {"median": float(np.median(sizes)), "min": int(sizes.min()), "max": int(sizes.max()),
                     "q10": float(np.quantile(sizes, 0.1)), "q90": float(np.quantile(sizes, 0.9)),
                     "U_star": int(len(np.unique(np.concatenate([U[t] for t in grid if dates[t].year >= 2020])))),
                     "by_year_median": {str(y): float(np.median([len(U[t]) for t in grid if dates[t].year == y]))
                                        for y in sorted({dates[t].year for t in grid})}},
        "v4_check": v4, "coverage": cov_gate, "delisted_missing": del_gate, "truncation": trunc,
        "gates": gates, "pass": bool(all(gates.values())),
        "data": P["info"], "fingerprint_2017_end": {"before": fp0, "after": fp1},
        "delisted_codes_in_panel": int(is_del.sum()), "codes": int(len(P["codes"])),
        "elapsed_s": time.time() - t_start,
        "note": "라벨(u·y)·수익·모델 출력은 계산하지 않았다.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "stage0.md").write_text(render(res))
    log(f"0단계 {'통과' if res['pass'] else '실패'} {gates}")
    return 0 if res["pass"] else 1


def render(r: dict) -> str:
    g = r["gates"]
    ok = lambda b: "통과" if b else "**실패**"  # noqa: E731
    L = ["# ML 수익률 예측 0단계 관문 (라벨 전)", "",
         f"사전등록 `{r['prereg']}`(2판) + 정정 ML-1. END {r['end']}, core `{r['core_ref']}`. 숫자 원본 `stage0.json`.",
         "**라벨(u·y)·수익·모델 출력은 계산하지 않았다.**", "",
         "| 관문 | 결과 |", "|---|---|",
         f"| (a) 절단 재계산 동일성(결정일 {len(r['truncation'])}개) | {ok(g['a_truncation'])} |",
         f"| (b) 피처별 U(t) 커버리지 | {ok(g['b_coverage'])} |",
         f"| (c) 폐지 대 상장 결측률 차 ≤ 20%p | {ok(g['c_delisted_missing'])} |",
         f"| v4 유니버스 재구성 = core `build_risk_model` 종목(3일) | {ok(g['v4_universe_matches_core'])} |",
         f"| 실행 중 표 지문 불변 | {ok(g['fingerprint_unchanged'])} |", "",
         f"**0단계: {'통과' if r['pass'] else '실패'}**", "",
         "## 격자·유니버스", "",
         f"IC 격자 {r['grid']['ic_days']}일({r['grid']['first']}~{r['grid']['last']}), 그중 OOS(2020~) {r['grid']['oos_days']}일. "
         f"학습 위상 날짜 {r['grid']['train_phase_days']}일.", "",
         f"U(t) 크기: 중앙 {r['universe']['median']:.0f}, 10~90% {r['universe']['q10']:.0f}~{r['universe']['q90']:.0f}, "
         f"최소 {r['universe']['min']}·최대 {r['universe']['max']}. U*(OOS 합집합) {r['universe']['U_star']}종목. "
         "연도별 중앙 " + ", ".join(f"{k} {v:.0f}" for k, v in r["universe"]["by_year_median"].items()) + ".", "",
         "## (b) 커버리지 (U(t) 안 유한 비율의 격자일 중앙값)", "",
         "| 피처 | 중앙값 | 바 | 결과 | 최소 |", "|---|---|---|---|---|"]
    for k, v in r["coverage"].items():
        L.append(f"| {k} | {v['median']:.3f} | ≥ {v['bar']:.2f} | {ok(v['pass'])} | {v['min']:.3f} |")
    L += ["", "F1(eps_yoy) 은 2020~ 격자일, 나머지는 IC 격자 전체(2019-08~)의 중앙값.", "",
          "## (c) 결측률 — 폐지 종목(`delisted_stocks`, 언제 폐지됐든) 대 상장 종목, U(t) 행", "",
          f"패널 {r['codes']}종목 중 폐지 {r['delisted_codes_in_panel']}.", "",
          "| 피처 | 폐지 결측 | 상장 결측 | 차 | 결과 | 폐지 행 |", "|---|---|---|---|---|---|"]
    for k, v in r["delisted_missing"].items():
        L.append(f"| {k} | {v['miss_delisted']:.3f} | {v['miss_listed']:.3f} | {v['gap']:+.3f} | {ok(v['pass'])} | {v['rows_delisted']} |")
    L += ["", "## (a) 절단 재계산 동일성", "", "| 결정일 | U(t) 같음 | 원 피처 전부 같음 | rank-gauss 같음 | U 크기 | 처음 어긋난 피처 |",
          "|---|---|---|---|---|---|"]
    for t in r["truncation"]:
        allraw = all(v["raw_all_codes_same"] for v in t["features"].values())
        L.append(f"| {t['date']} | {t['U_same']} | {allraw} | {t['rank_gauss_same']} | {t['U_n']} | {t['first_diff'] or '—'} |")
    L += ["", "## v4 유니버스 대조", ""] + [f"- {v['date']}: 같음 {v['same']}, {v['n']}종목" for v in r["v4_check"]]
    d = r["data"]
    L += ["", "## 데이터", "",
          f"- 이음매 가드(PO-3.3): 셀 {d['seam']['seam_cells']}, 원천별 {d['seam']['seam_cells_by_source']}, 종목 {d['seam']['seam_codes']}.",
          f"- 실적(F1): {d['signal']}.",
          f"- 투자자 동향 행 {d['investor_trend_rows']}, 리포트 행 {d['reports_rows']}, 공시 분기 {d['disclosure_periods']}.",
          f"- 표 지문(daily_bars_adjusted 2019-01-01~END): {d['fingerprint']}.",
          f"- 표 지문(2017-01-01~END) 시작=끝: {r['fingerprint_2017_end']['before'] == r['fingerprint_2017_end']['after']}.",
          "", f"_{r['elapsed_s']:.0f}초_"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--core-ref", default="13c56a7")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    return run(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
