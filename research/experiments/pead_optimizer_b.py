"""PEAD 최적화 단계 B 하네스 — 현행 순위 북 R 대 최적화 북 O. 사전등록 ``PREREG.md`` §2·PO-1.4·PO-1.6·PO-1.7·PO-3.4·PO-5·PO-6.

두 북·널·패리티가 **같은 적격 마스크 함수 하나**(:func:`eligible` = A 단계가 쓴 ``pead_optimizer_c0.eligible_idx``)를 쓴다.
회계는 두 북이 같은 엔진(:func:`simulate`)을 탄다. 구현 고정(모호했던 곳)은 정정 PO-6 에 적었다.

구현은 ``src/swing_it/book/`` 으로 옮겼고 여기서는 부르기만 한다. 사전등록 상수·판정(:func:`verdict`)·실데이터 입력·실행은 이 파일에 남는다.
옮긴 뒤에도 ``B.json`` 을 1e-10 안에서 재현한다(``tests/real_db/test_b_json_reproduction.py``).

모드:

- ``--mode parity`` (PO-1.4): 패리티 검사만 한다. 하네스 패리티 모드(지연 0·무비용·강제 매도 끔·현금 없음)와 core
  ``staggered_tranche_backtest`` 의 스텝별 ``book``·``bench`` 가 1e-10 안인지, ``n_universe``·적격 집합이 같은지만 본다.
  **성과 숫자는 출력하지도 저장하지도 않는다**(차이·개수만). 출력 ``B_parity.json``·``B_parity.md``.
- ``--mode run``: 단계 B 전체(민감도 6칸 → 1차 → 순열 널 300 → 판정). PO-5.2 순서를 코드로 강제한다 — ``A1_repaired.json`` 이 있고
  그 판정이 ``통과`` 이며, 지금 ``daily_bars_adjusted`` 지문이 그 파일의 지문과 같을 때만 돈다.

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" \\
      python research/experiments/pead_optimizer_b.py --mode parity --cache <scratch>/bparity --po3-cache <scratch>/po3cache

2024-01 이후 데이터는 어떤 계산에도 쓰지 않는다(END 2023-12-28, PREREG §2).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import krx_quant_core.portfolio  # noqa: F401 — portfolio-layer(PR #19) 필요. 없으면 여기서 ModuleNotFoundError
import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _ffill_axis1_nb, _trailing_adv
from pead_optimizer_c0 import LAM_HI, LAM_ITERS, LAM_LO, LAM_TOL, eligible_idx, keyed_rng

from swing_it.book import metrics as _metrics
from swing_it.book import nulls as _nulls
from swing_it.book.accounting import COST_BP_PER_TURN, INV_EPS, LIMIT_UP_RET, check_invariants, r_sigma_series, simulate
from swing_it.book.context import Ctx, StepRec, delisted
from swing_it.book.costs import COMMISSION_BP, HALF_SPREAD_BP, VOL_MIN_OBS, VOL_WIN, linear_bp, trade_costs, trailing_vol
from swing_it.book.metrics import LABEL_IC0, holding_intervals, ir, mask_ic_zero_folds, payoff_ratio, step_returns
from swing_it.book.nulls import BOOT_MEAN_BLOCK, N_BOOT, N_PERM, stationary_boot_dir
from swing_it.book.optimized import (
    DUST_EPS,
    FACTOR_ALARM,
    alpha_vec,
    capped_rescale,
    clean_dust,
    match_lambda,
    o_target,
    rank_z,
)
from swing_it.book.parity import PARITY_TOL, core_parity, full_parity, parity_book, parity_check, parity_signal
from swing_it.book.rank import r_target, tranche_names
from swing_it.book.risk import active_risk, factor_share, sub_model, top_exposures
from swing_it.book.spec import PRIMARY, Cell, Flags
from swing_it.features.universe import m0_mask, rank_band_eligible, u_exclusion
from swing_it.validation.ic import IC_CLIP, daily_rank_ic, fold_ic

# 옮긴 구현을 이 이름으로 부르는 곳(feasibility·ML 러너·시험)이 있다.
__all__ = [
    "BOOT_MEAN_BLOCK", "COMMISSION_BP", "COST_BP_PER_TURN", "DUST_EPS", "FACTOR_ALARM", "HALF_SPREAD_BP", "IC_CLIP", "INV_EPS",
    "LABEL_IC0", "LAM_HI", "LAM_ITERS", "LAM_LO", "LAM_TOL", "LIMIT_UP_RET", "N_BOOT", "PARITY_TOL", "PRIMARY", "VOL_MIN_OBS",
    "VOL_WIN", "Cell", "Ctx", "Flags", "StepRec", "_ffill_axis1_nb", "eligible_idx",
    "active_risk", "alpha_vec", "book_summary", "capped_rescale", "check_invariants", "clean_dust", "core_parity", "daily_rank_ic",
    "delisted", "eligible", "factor_share", "fold_ic", "full_parity", "holding_intervals", "ir", "keyed_rng", "linear_bp",
    "mask_ic_zero_folds", "match_lambda", "m0_mask", "o_target", "oos", "parity_book", "parity_check", "parity_signal",
    "payoff_ratio", "permutation_nulls", "r_sigma_series", "r_target", "rank_z", "simulate", "stationary_boot_dir", "step_returns",
    "sub_model", "top_exposures", "trade_costs", "trailing_vol", "tranche_names", "u_exclusion",
]

# ── 사전등록 상수(PREREG §2·PO-1.6·PO-1.7·PO-5) ──────────────────────────────────────

END_B = "2023-12-28"  # §2 — 2024-01 이후는 쓰지 않는다
LOAD_START = "2019-01-02"  # 위험모델 입력(v4 와 같은 시작)
IC_LOAD_START = "2016-11-01"  # IC 학습 패널(2017-01-02 부터 + ADV 20일 앞)
IC_START = "2017-01-02"
GRID_START = "2019-08-01"
EVAL = ("2020-01-01", "2023-12-31")
FOLD_YEARS = (2020, 2021, 2022, 2023)
STEP, R_HORIZON, TOP_N, MIN_NAMES = 20, 60, 10, 20
CAP_RANK, ADV_FLOOR = (20, 100), 20000.0  # 시총 21~100위, ADV 200억(백만원)
V4_SHRINK = 0.1  # 위험모델 v4 = vraoff_s0.1(PO-3.5)
MAX_W, MAX_TURNOVER, MAX_PART = 0.10, 2.0 / 3.0, 0.05
NULL_Q = 0.95
NI_X_CAP, NI3_RATIO = 0.30, 1.25  # PO-1.6 x = min(IR(R)/3, 0.30), B-NI-3
OPT_OK_MIN, OOS_MIN, LAM_FAIL_MAX = 0.95, 40, 0.10  # VOID
LABEL_RISK_UP = "위험 맞춘 비교 아님 (O 사전위험 < R)"  # PO-8 — λ 실패 원인 up
ANN = math.sqrt(252.0 / STEP)


SENSITIVITY = (Cell(20, 0.05), Cell(20, 0.10), Cell(20, 0.20), Cell(60, 0.05), Cell(60, 0.20), Cell(60, 0.10, 1e10))


# ── 이 연구의 평가 구간을 묶은 얇은 래퍼 ──────────────────────────────────────────


def eligible(t: int, sig: np.ndarray, adv: np.ndarray, cap: np.ndarray, in_u: np.ndarray) -> np.ndarray:
    """eligible(t) = M0(t) 안 시총 21~100위(행 번호, 시총 내림차순). A 단계(P10B·OB)와 같은 함수다."""
    return rank_band_eligible(t, sig, adv, cap, in_u, adv_floor=ADV_FLOOR, cap_rank=CAP_RANK)


def oos(df: pd.DataFrame) -> pd.DataFrame:
    return _metrics.oos(df, EVAL)


def book_summary(recs: list[StepRec]) -> dict:
    """한 북의 OOS 요약(1·2배)."""
    return _metrics.book_summary(recs, EVAL, FOLD_YEARS, ann=ANN)


def permutation_nulls(ctx, cell, ic_by_year, sigma_r, workers: int, n: int = N_PERM):
    """같은 π 를 O 와 R 에 쓴다(시드 ``keyed_rng("perm", r)``). 2배 IR 분포를 돌려준다."""
    return _nulls.permutation_nulls(ctx, cell, ic_by_year, sigma_r, workers, n, window=EVAL)




# ── 지표·판정(§2·PO-1.6·PO-5) ───────────────────────────────────────────────────


def verdict(o: dict, r: dict, null_o: np.ndarray, null_r: np.ndarray, parity_pass: bool, ic_zero_folds=()) -> dict:
    """§2·PO-1.6·PO-5·PO-8 바의 기계적 적용. ``o``/``r`` 은 :func:`book_summary` 결과.

    PO-8: λ 실패 VOID(> 10%)는 원인 solver·gap·down 에만 건다. 원인 up(가장 작은 λ 에서도 O 사전위험 < R — 비용 > α)은 세어
    보고하고 :data:`LABEL_RISK_UP` 라벨·실현 능동위험 비를 붙인다. ``ic_zero_folds`` 는 IC_k = 0 폴드(α ≡ 0) — 통합 결과에는 남기고 라벨을 붙인다.
    """
    ir_r2, ir_o2, ir_r1, ir_o1 = r["IR_2x"], o["IR_2x"], r["IR_1x"], o["IR_1x"]
    x = min(ir_r2 / 3.0, NI_X_CAP)
    q_r, q_o = float(np.quantile(null_r, NULL_Q)), float(np.quantile(null_o, NULL_Q))
    bars = {
        "SIG": bool(ir_r2 > q_r),
        "B-NI-1": bool(ir_o2 - ir_r2 >= -x),
        "B-NI-2": bool(ir_o1 - ir_r1 >= -x),
        "B-NI-3": bool(o["sd_a_2x"] <= NI3_RATIO * r["sd_a_2x"]),
        "B-NI-4": bool(ir_o2 > q_o),
        "B-NI-5": bool(o["EV_2x"] > 0),
    }
    void = {"optimal_share<0.95": o["optimal_share"] < OPT_OK_MIN, "parity_fail": not parity_pass,
            "oos_steps<40": min(o["oos_steps"], r["oos_steps"]) < OOS_MIN,
            "lam_fail(solver·gap·down)>0.10": o.get("lam_fail_nonup_share", o["lam_fail_share"]) > LAM_FAIL_MAX}
    if any(void.values()):
        v = "VOID"
    elif not bars["SIG"]:
        v = "판단 불가(신호 약함)"
    elif all(bars[b] for b in ("B-NI-1", "B-NI-2", "B-NI-3", "B-NI-4", "B-NI-5")):
        v = "비열등 통과"
    else:
        v = "채택 불가"
    return {"x": x, "null_q95_R": q_r, "null_q95_O": q_o, "dIR_2x": ir_o2 - ir_r2, "dIR_1x": ir_o1 - ir_r1,
            "bars": bars, "void": void, "verdict": v,
            "failed": [b for b, ok in bars.items() if not ok and b != "SIG"],
            "B-NI-3_label": "표본 안에서 유리(v4 는 이 구간에서 튜닝됐다, PO-5.1)",
            "B-NI-1_note": "열등 아님 선별(검정 아님) — 참 ΔIR = −x 이면 이 선별을 통과할 확률은 0.5 다(PO-1.6)",
            "lam_up_share": o.get("lam_up_share", 0.0),
            "risk_label": LABEL_RISK_UP if o.get("lam_up_share", 0.0) > 0 else None,
            "realized_active_risk_ratio_2x": (o["sd_a_2x"] / r["sd_a_2x"]) if r["sd_a_2x"] else float("nan"),
            "exante_risk_ratio": (o.get("exante_20_mean", float("nan")) / r["exante_20_mean"])
            if r.get("exante_20_mean") else float("nan"),
            "ic_zero_folds": list(ic_zero_folds),
            "ic_zero_label": (f"{LABEL_IC0} 폴드 {list(ic_zero_folds)} — 통합 결과에는 α ≡ 0 으로 남는다(PO-8)" if ic_zero_folds else None)}


# ── 실제 규칙과의 차이(PO-1.4, 보고만) ─────────────────────────────────────────


TOGGLES = (
    ("① t+1 체결", None),
    ("② 스텝 사이 표류", Flags(lag=1, drift=True, forced_sale=False, delist_cash=False, no_trade=False, costs=False)),
    ("③ 위험모델 이탈 강제 매도", Flags(lag=1, drift=True, forced_sale=True, delist_cash=False, no_trade=False, costs=False)),
    ("④ 폐지 대금 현금", Flags(lag=1, drift=True, forced_sale=True, delist_cash=True, no_trade=False, costs=False)),
    ("⑤ no_trade", Flags(lag=1, drift=True, forced_sale=True, delist_cash=True, no_trade=True, costs=False)),
    ("⑥ 비용(1배)", Flags()),
)


def toggle_report(ctx: Ctx) -> list[dict]:
    """패리티 모드에서 한 규칙씩 켜서(누적) R 의 스텝 능동수익 차이를 낸다(B 실행 때만 — 수익을 쓰므로 패리티 모드에서는 부르지 않는다)."""
    prev = parity_book(ctx, 0).set_index("t")
    prev_a = prev["book"] - prev["bench"]
    out = []
    for name, fl in TOGGLES:
        if fl is None:
            cur = parity_book(ctx, 1).set_index("t")
            cur_a = cur["book"] - cur["bench"]
            ev = {"steps": int(len(cur))}
        else:
            recs = simulate(ctx, "R", flags=fl)
            df = step_returns(recs, 1.0).set_index("t")
            cur_a = df["a"]
            ev = {k: int(sum(getattr(r, k) for r in recs)) for k in ("n_forced", "n_delist", "n_frozen")}
        d = (cur_a - prev_a).dropna()
        out.append({"rule": name, "events": ev, "diff_sum": float(d.sum()), "diff_mean": float(d.mean()),
                    "diff_max_abs": float(d.abs().max()) if len(d) else float("nan"), "steps": int(len(d))})
        prev_a = cur_a
    return out


# ── 데이터(실데이터 — DB) ─────────────────────────────────────────────────────


def _db():
    import pead_optimizer_po2 as P2

    return P2._db()


def market_and_delisting(codes) -> tuple[np.ndarray, np.ndarray, dict]:
    """시장(``stocks``(현재) 우선, 없으면 ``delisted_stocks``)과 폐지 마지막 거래일. 모르는 시장은 KOSDAQ(비용 큰 쪽, PO-6)."""
    from krx_quant_core.market.codes import Market

    con, cur = _db()
    cur.execute("SELECT code, market FROM stocks")
    cur_m = dict(cur.fetchall())
    cur.execute("SELECT code, market, last_trade_date FROM delisted_stocks")
    rows = cur.fetchall()
    con.close()
    del_m = {c: m for c, m, _ in rows}
    last = {c: d for c, _, d in rows if d is not None}
    mk, unknown = [], 0
    for c in codes:
        raw = cur_m.get(str(c), del_m.get(str(c)))
        try:
            m = Market.parse(raw).value if raw is not None else None
        except ValueError:
            m = None
        if m is None or m == "KONEX":
            unknown += m is None
            m = "KOSDAQ"
        mk.append(m)
    return np.array(mk, dtype=object), last, {"market_unknown": unknown, "delisted_with_last_date": len(last)}


def model_cache_name(fp: list, core_ref: str, stem: str = "models_v4_grid") -> str:
    """PO-7 M1 — 모델 캐시 이름에 표 지문 해시·core SHA 를 넣는다. 표가 바뀌면(PR #12) 옛 캐시를 읽지 않는다."""
    import hashlib

    h = hashlib.sha256(json.dumps(fp, default=str).encode()).hexdigest()[:12]
    return f"{stem}_fp{h}_core{core_ref}.pkl"


def build_ctx(cache: Path, po3_cache: Path, workers: int, log, end: str = END_B, core_ref: str = "13c56a7"):
    """B 입력: v4 와 같은 로더·가드·노출, 격자 t_k(2019-08-01 이후 첫 거래일 + 20k)마다 v4 모델.

    모델 캐시는 ``daily_bars_adjusted`` 지문(2019-01-01~END)·core SHA 에 묶인다(:func:`model_cache_name`).
    """
    import pead_optimizer_po3 as P3
    import pead_optimizer_po4 as P4

    assert end == END_B, "B 는 2023-12-28 까지만 쓴다(§2)"
    fp = P4.fingerprint("daily_bars_adjusted", end)
    x0 = P4.load_inputs(end)
    x, seam, seam_info = P4.guarded(x0, end)
    c = P4.prep(x, "2019-07-01", "2023-10-31", 52, end)
    first = c["first"]
    dates = c["dates"]
    nD = c["R"].shape[1]
    t_0 = int(np.searchsorted(dates, np.datetime64(GRID_START)))
    grid = list(range(t_0, nD, STEP))
    mpath = cache / model_cache_name(fp, core_ref)
    models = P3.build_models(c, grid, workers, mpath, log, vra=None, shrink=V4_SHRINK)
    # po3_cache=None: PO-3 캐시와의 비트 대조를 건너뛴다(ML 예측 S2-0 — 옛 캐시가 없어도 패리티를 본다). 숫자에는 영향 없다.
    old = {} if po3_cache is None else pickle.load(open(po3_cache / "models_vraoff_s0.1.pkl", "rb"))
    common = sorted(set(old) & set(models))
    bit_same = bool(common) and all(
        all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(old[t][:4], models[t][:4])) and old[t][4] == models[t][4]
        for t in common)
    sig_all, sig_info = P4.yoy_signal_strict(x.codes, x.dates, end)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    market, last, mk_info = market_and_delisting(x.codes)
    didx = pd.DatetimeIndex(dates)
    delist_after = np.full(len(x.codes), 10**9, dtype=np.int64)
    for i, cc in enumerate(x.codes):
        d = last.get(str(cc))
        if d is not None:
            j = int(didx.searchsorted(pd.Timestamp(d), side="right")) - 1  # 마지막 거래일 열(그 날까지 거래)
            if j < len(didx) - 1:  # 창 끝 이후 폐지는 창 안에서 폐지가 아니다
                delist_after[i] = j
    sl = slice(first, None)
    ctx = Ctx(dates=didx, C=x.close[:, sl], tv_won=x.trade_value[:, sl], adv=adv_all[:, sl], cap=c["cap"],
              sig=sig_all[:, sl], ret=c["R"], lab=c["lab"], market=market, grid=grid, models=models,
              delist_after=delist_after)
    info = {"seam": {k: seam_info[k] for k in ("seam_cells", "seam_cells_by_source")}, "signal": sig_info,
            "market": mk_info, "grid": [str(dates[grid[0]].date()), str(dates[grid[-1]].date()), len(grid)],
            "models_bit_same_PO3_common": bit_same, "models_common_n": len(common), "first": int(first),
            "fingerprint_models": fp, "model_cache": mpath.name, "prep": c,
            "adv_full": adv_all, "trade_value_full": x.trade_value, "close_full": x.close}
    return ctx, info, x


# ── 실행 ───────────────────────────────────────────────────────────────────────

OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"


def run_parity(args, log) -> int:
    """PO-1.4 패리티 검사 — 차이·개수만 기록한다(성과 숫자 없음)."""
    import pead_optimizer_po4 as P4

    t0 = time.time()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fp0 = P4.fingerprint("daily_bars_adjusted", END_B)
    ctx, info, x = build_ctx(cache, Path(args.po3_cache), args.workers, log, core_ref=args.core_ref)
    log(f"격자 {info['grid']}, PO-3 캐시와 공통 모델 {info['models_common_n']}개 비트 동일 {info['models_bit_same_PO3_common']}")
    pc = full_parity(ctx)
    fp1 = P4.fingerprint("daily_bars_adjusted", END_B)
    res = {"mode": "parity", "core_ref": args.core_ref, "end": END_B, "table": "daily_bars_adjusted(실행 시점 표 — 수리 전후는 아래 지문으로 가린다)",
           "fingerprint_before": fp0, "fingerprint_after": fp1, "fingerprint_same": fp0 == fp1,
           "grid": info["grid"], "models_bit_same_PO3_common": info["models_bit_same_PO3_common"],
           "models_common_n": info["models_common_n"], "seam": info["seam"], "market": info["market"],
           "parity": pc, "elapsed_s": time.time() - t0,
           "note": "성과 숫자(book·bench·IR)는 계산 중에만 쓰고 출력·저장하지 않았다. 차이·개수만 적는다."}
    (OUT / "B_parity.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "B_parity.md").write_text(render_parity(res))
    log(f"패리티 {'통과' if pc['pass'] else '실패'}: 최대 차 book {pc.get('max_abs_diff_book')}, bench {pc.get('max_abs_diff_bench')}")
    return 0 if pc["pass"] else 1


def render_parity(r: dict) -> str:
    p = r["parity"]
    L = ["# 단계 B 패리티 검사 (PO-1.4) — 코드 검사, 성과 숫자 없음", "",
         f"core `{r['core_ref']}`, END {r['end']}, 표 {r['table']}. 격자 {r['grid'][0]}~{r['grid'][1]} ({r['grid'][2]}일).",
         "하네스 패리티 모드(지연 0·무비용·강제 매도 끔·현금 없음, `start_index` = 첫 격자일, `min_names=20`)와 core "
         "`staggered_tranche_backtest(top_n=10, cap_rank=(20,100), horizon=60, step=20, adv_floor=20000, delisting_exit=True)` 의 스텝별 비교.",
         "**book·bench 값 자체는 출력·저장하지 않았다**(B 는 수리된 표 뒤에 연다, PO-5.2).", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 스텝 수(하네스/core) | {p['steps_harness']}/{p['steps_core']} (같은 날짜 {p['same_steps']}) |",
         f"| 기간 | {p.get('first_date')} ~ {p.get('last_date')} |",
         f"| book 최대 절대차 | {p.get('max_abs_diff_book', float('nan')):.3e} |",
         f"| bench 최대 절대차 | {p.get('max_abs_diff_bench', float('nan')):.3e} |",
         f"| NaN 불일치 | {p.get('nan_mismatch')} |",
         f"| n_universe 같음 | {p.get('n_universe_same')} |",
         f"| 적격 집합 같음(격자 전부) | {p.get('eligible_same_all_grid')} |",
         f"| ADV 바이트 같음(core `_trailing_adv`) | {p.get('adv_bytes_same')} |",
         "| 바 | 1e-10 |", "",
         f"**패리티: {'통과' if p['pass'] else '실패'}.**", "",
         f"- 표 지문 시작=끝: {r['fingerprint_same']} {r['fingerprint_before']}.",
         f"- 격자 모델이 PO-3 캐시(v4)와 공통 {r['models_common_n']}개 비트 동일: {r['models_bit_same_PO3_common']}.",
         f"- 이음매 가드: {r['seam']}. 시장 구분: {r['market']}.",
         "- B(`--mode run`)는 PR #12 수리된 표에서 이 검사를 다시 통과해야 돈다(PO-5.2-4) — 수리 전 지문 `fingerprint_before` 는 `A1.json`.", "",
         f"_{r['elapsed_s']:.1f}초_"]
    return "\n".join(L) + "\n"


def git_sha() -> str:
    import subprocess

    r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain", "--", "research/experiments"],
                           capture_output=True, text=True, check=False).stdout.strip()
    return r.stdout.strip() + ("-dirty" if dirty else "")


def fingerprint_range(table: str, lo: str, hi: str) -> list:
    """core ``_fingerprint`` 를 임의 구간으로(IC 패널 2016-11-01~ 용, PO-7 MINOR 2)."""
    import pead_optimizer_po2 as P2
    from krx_quant_core.data.daily import _fingerprint

    con, _ = P2._db()
    try:
        return _fingerprint(con, table, pd.Timestamp(lo).date(), pd.Timestamp(hi).date())
    finally:
        con.close()


def gate_b(a1r: dict | None, fp_now: list, report_impl: dict) -> None:
    """PO-5.2·PO-7 실행 전 관문(순수 함수 — 시험 가능). 막히면 ``SystemExit``."""
    import pead_optimizer_b_report as BR

    missing = [k for k in BR.REQUIRED if not report_impl.get(k)]
    if missing:
        raise SystemExit(f"보고만 항목 미구현 {missing} — B 를 돌리지 않는다(PO-7 M4)")
    if a1r is None:
        raise SystemExit("A1_repaired.json 이 없다 — PR #12 배포와 강건성 재실행(PO-5.2) 전에는 B 를 돌리지 않는다")
    if "STOP" in a1r or a1r.get("verdict") != "통과":
        raise SystemExit(f"강건성 재실행 판정 {a1r.get('verdict')!r} — '취약', B 는 역사 구간에서 열지 않는다(PO-5.2-3)")
    rp = a1r.get("repaired")
    if not rp or rp.get("label") is not None or not rp.get("B_opens") or not rp.get("fingerprint_differs_from_pre"):
        raise SystemExit("A1_repaired.json 의 강건성 분기가 B 를 열지 않는다(라벨·지문) — 멈춘다(PO-5.2-3)")
    if fp_now != a1r["fingerprint_before"]["daily_bars_adjusted"]:
        raise SystemExit("지금 표 지문이 강건성 재실행 때와 다르다 — 멈춘다(PO-5.2-4)")


def compute_b(args, ctx, info, x, pc, log) -> tuple[dict, list[str], int]:  # pragma: no cover - B 실행·재현 때만
    """단계 B 의 계산(관문·쓰기 없음). ``run_b`` 와 재현 시험(``tests/real_db/``)이 같은 코드를 탄다."""
    from swing_it.diagnostics.trials import record_trial

    ic_by, ic_info = ic_by_year_for(args, log)
    sigma_r = r_sigma_series(ctx)
    viol: list[str] = []
    steps_checked = 0
    res = {"mode": "run", "parity": pc, "cells": {}, "ic": ic_info}
    for cell in SENSITIVITY + (PRIMARY,):  # GUARDRAILS §4.5 ② — 격자를 1차보다 먼저
        record_trial("pead_optimizer", {"stage": "B", "cell": cell.name, "core": args.core_ref},
                     note="B 민감도" if cell != PRIMARY else "B 1차")
        recs_o = simulate(ctx, "O", cell, ic_by_year=ic_by[cell.h_alpha], sigma_r=sigma_r)
        viol += check_invariants(recs_o, "O", cell)
        steps_checked += len(recs_o)
        res["cells"][cell.name] = {"O": mask_ic_zero_folds(book_summary(recs_o), ic_info["zero_folds"][cell.h_alpha])}
        if cell == PRIMARY:
            recs_r = simulate(ctx, "R")
            viol += check_invariants(recs_r, "R")
            steps_checked += len(recs_r)
            zf = ic_info["zero_folds"][cell.h_alpha]
            o_s, r_s = mask_ic_zero_folds(book_summary(recs_o), zf), mask_ic_zero_folds(book_summary(recs_r), zf)
            nulls = permutation_nulls(ctx, cell, ic_by[cell.h_alpha], sigma_r, args.workers)
            viol += nulls["violations"]
            aO = oos(step_returns(recs_o, 2.0))["a"].to_numpy()
            aR = oos(step_returns(recs_r, 2.0))["a"].to_numpy()
            boot = stationary_boot_dir(aO, aR)
            res["primary"] = {"O": o_s, "R": r_s, "verdict": verdict(o_s, r_s, nulls["O"], nulls["R"], pc["pass"], zf),
                              "rho": float(np.corrcoef(aO, aR)[0, 1]), "boot_dIR_q05_50_95":
                              [float(np.quantile(boot, q)) for q in (0.05, 0.5, 0.95)],
                              "null_O_perm_nan_share_mean": float(np.nanmean(nulls["O_perm_nan_share"])),
                              "toggles": toggle_report(ctx)}
            res["primary"]["report"] = b_report(ctx, info, x, recs_o, recs_r, aO - aR, log)
    return res, viol, steps_checked


def run_b(args, log) -> int:  # pragma: no cover - B 는 PR #12 배포·강건성 재실행 뒤에만(PO-5.2)
    """단계 B 전체. PO-5.2 순서·PO-7 불변식·보고만 구현 여부를 코드로 막는다."""
    import pead_optimizer_b_report as BR
    import pead_optimizer_po4 as P4

    t_start = time.time()
    rep = OUT / "A1_repaired.json"
    a1r = json.loads(rep.read_text()) if rep.exists() else None
    fp_now = P4.fingerprint("daily_bars_adjusted", "2026-09-11")
    gate_b(a1r, fp_now, BR.IMPLEMENTED)
    if (OUT / "B.json").exists():
        raise SystemExit("B.json 이 이미 있다 — B 는 한 번만 돈다")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fp0 = P4.fingerprint("daily_bars_adjusted", END_B)
    fp_ic0 = fingerprint_range("daily_bars_adjusted", IC_LOAD_START, END_B)
    ctx, info, x = build_ctx(cache, Path(args.po3_cache), args.workers, log, core_ref=args.core_ref)
    pc = full_parity(ctx)
    if not pc["pass"]:
        raise SystemExit("패리티 실패 — VOID(§2)")
    res, viol, steps_checked = compute_b(args, ctx, info, x, pc, log)
    if viol:
        (cache / "B_invariant_violations.txt").write_text("\n".join(viol))
        raise SystemExit(f"불변식 위반 {len(viol)}건 — 결과를 쓰지 않는다(PO-7 M3). 목록: {cache / 'B_invariant_violations.txt'}")
    fp1 = P4.fingerprint("daily_bars_adjusted", END_B)
    fp_ic1 = fingerprint_range("daily_bars_adjusted", IC_LOAD_START, END_B)
    if fp0 != fp1 or fp_ic0 != fp_ic1:
        raise SystemExit("실행 중 표 지문이 바뀌었다 — 결과를 쓰지 않는다(PO-4.2)")
    res["invariants"] = {"violations": 0, "steps_checked": steps_checked}
    res["code"] = {"swing_it_sha": git_sha(), "core_ref": args.core_ref, "model_cache": info["model_cache"]}
    res["fingerprints"] = {"daily_bars_adjusted_b_range": fp0, "daily_bars_adjusted_ic_range": fp_ic0,
                           "daily_bars_adjusted_full_now": fp_now, "A1_repaired": a1r["fingerprint_before"]["daily_bars_adjusted"],
                           "same_as_A1_repaired": fp_now == a1r["fingerprint_before"]["daily_bars_adjusted"],
                           "models": info["fingerprint_models"]}
    res["elapsed_s"] = time.time() - t_start
    (OUT / "B.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    (OUT / "B.md").write_text(BR.render_b(json.loads((OUT / "B.json").read_text())))
    return 0


def b_report(ctx, info, x, recs_o, recs_r, d2x, log) -> dict:  # pragma: no cover - B 실행 때만
    """보고만 항목(PO-7.4) — ``pead_optimizer_b_report``."""
    import pead_optimizer_b_report as BR
    import pead_optimizer_po2 as P2
    from krx_quant_core.risk.factor_model import factor_returns

    first = info["first"]
    kospi = np.asarray(x.market_ret[first:], dtype=np.float64)
    con, cur = P2._db()
    cur.execute("SELECT date, close FROM sector_index WHERE code = '101' AND date BETWEEN %s AND %s",
                (str(ctx.dates[0].date()), END_B))
    kq = pd.DataFrame(cur.fetchall(), columns=["date", "close"])
    con.close()
    lvl = pd.Series(kq["close"].to_numpy(float), index=pd.to_datetime(kq["date"])).reindex(ctx.dates)
    kosdaq = lvl.pct_change(fill_method=None).to_numpy()
    ro, rr = [r for r in recs_o if EVAL[0] <= r.date <= EVAL[1]], [r for r in recs_r if EVAL[0] <= r.date <= EVAL[1]]
    fac = BR.factor_step_returns(ctx, ro, kospi, kosdaq, trailing_vol, ADV_FLOOR)
    led = BR.ledger_n()
    hur = BR.alpha_hurdle(led["N"])
    out = {"ledger": led, "hurdle": hur, "factor_alpha": {}, "vs_kospi": {}, "deflation": {}, "attribution": {}}
    for b, recs in (("O", ro), ("R", rr)):
        out["factor_alpha"][b] = {}
        for m in (1.0, 2.0):
            net = step_returns(recs, m)[["t", "date", "net"]]
            out["factor_alpha"][b][f"{int(m)}x"] = BR.factor_alpha_report(net, fac)
        net2 = step_returns(recs, 2.0)[["t", "date", "net"]]
        out["vs_kospi"][b] = BR.vs_kospi(net2, fac)
        out["deflation"][b] = BR.deflation(net2["net"], led["N"])
    out["qualification"] = BR.pead_qualification(out["factor_alpha"]["O"]["2x"]["t_alpha"],
                                                 out["factor_alpha"]["R"]["2x"]["t_alpha"], hur["hurdle"])
    prep = info["prep"]
    fr = factor_returns(prep["R"], prep["Xs"], prep["lab"], prep["cap"], prep["msk"])
    for b, recs in (("O", ro), ("R", rr)):
        out["attribution"][b] = BR.attribution_report(
            ctx, recs, fr, prep, lambda t: eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t)))
    out["f_boot"] = BR.f_boot(d2x)
    out["factor_index_missing_days"] = int(fac["index_missing_days"].sum())
    log(f"보고만 끝: 원장 N {led['N']}, f_boot {out['f_boot']['f_boot']}")
    return out


def ic_by_year_for(args, log) -> tuple[dict, dict]:  # pragma: no cover - B 실행 때만
    """폴드(연도)마다 IC_k — 2017-01-02 부터 폴드 시작 (H+2) 거래일 앞까지(PO-6). H ∈ {20, 60}.

    PO-7 MINOR 2: IC 패널에도 위험모델과 같은 이음매 가드(PO-3.3)를 건다. 폐지는 마지막 종가(:func:`daily_rank_ic`).
    IC_k 가 0 으로 잘린 폴드는 α ≡ 0 이라 λ 맞춤이 그 폴드 전부에서 실패한다(PO-7.5) — 그 폴드를 적어 둔다.
    """
    import pead_optimizer_po4 as P4
    from krx_quant_core.data.risk_inputs import load_risk_inputs

    xi0 = load_risk_inputs(IC_LOAD_START, END_B)
    xi, _, seam_info = P4.guarded(xi0, END_B)
    sig, _ = P4.yoy_signal_strict(xi.codes, xi.dates, END_B)
    adv = _trailing_adv(xi.trade_value / 1e6, len(xi.dates), 20, xi.close)
    d = pd.DatetimeIndex(xi.dates)
    lo = int(d.searchsorted(pd.Timestamp(IC_START)))
    out, info = {}, {"seam_cells": seam_info["seam_cells"], "n_days": {}, "zero_folds": {}}
    for h in (20, 60):
        days, ics = daily_rank_ic(xi.close, adv, xi.cap, sig, lo, len(d) - 1, h)
        out[h] = {y: fold_ic(days, ics, int(d.searchsorted(pd.Timestamp(f"{y}-01-01"))), h)
                  for y in (2019,) + FOLD_YEARS}
        info["n_days"][h] = int(len(ics))
        info["zero_folds"][h] = [y for y, v in out[h].items() if v <= 0.0]
        log(f"IC 학습 H={h}: 일 {len(ics)}")
    info["ic_by_year"] = {h: {str(y): v for y, v in out[h].items()} for h in out}
    return out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("parity", "run"), required=True)
    ap.add_argument("--cache", required=True, help="레포 밖")
    ap.add_argument("--po3-cache", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--core-ref", default="13c56a7")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assert repo not in Path(args.cache).resolve().parents, "캐시는 레포 밖에 둔다"
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    return run_parity(args, log) if args.mode == "parity" else run_b(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
