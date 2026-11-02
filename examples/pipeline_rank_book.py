#!/usr/bin/env python
"""새 연구의 모양 — 선언하고, 게이트를 부르고, 단계마다 ``src`` 함수를 부른다. 합성 데이터로 돌아 DB·네트워크가 필요 없다.

실제 연구는 ``synthetic_inputs`` 자리에 정문(``storage.read_prices``·``read_earnings``·``read_periodic_filings``)을 넣고, 신호 자리에 자기 신호를 넣는다.
단계마다 러너를 나눠도 된다. 그때 각 러너가 자기 단계를 선언한다. 표는 ``docs/PIPELINE.md`` 다.

    python examples/pipeline_rank_book.py            # 앞 단계 증거가 없으면 여기서 멈춘다
    python -m swing_it.pipeline record example_rank_book 1 pass --evidence ... --reason ...
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _trailing_adv

from swing_it.book.accounting import check_invariants, simulate
from swing_it.book.context import Ctx, models_from_mask
from swing_it.book.metrics import book_summary, oos, step_returns
from swing_it.book.nulls import permutation_nulls
from swing_it.book.spec import PRIMARY, BookSpec, Flags
from swing_it.diagnostics.hac import ols_hac
from swing_it.features.earnings_asof import yoy_strict_wide
from swing_it.features.universe import rank_band_eligible
from swing_it.pipeline import Stage, require_stage
from swing_it.pipeline.checks import check_panels, check_signal, check_weights
from swing_it.validation.ic import daily_rank_ic, fold_ic

PIPELINE = ("example_rank_book", Stage.BACKTEST)

SPEC = BookSpec(step=20, horizon=60, top_n=5, min_names=10, adv_floor=1000.0, cap_rank=(0, 40))
WINDOW = ("2020-06-01", "2021-12-31")
END = "2021-12-30"
ANN = float(np.sqrt(252.0 / SPEC.step))


def synthetic_inputs(seed: int = 7, n: int = 60, days: int = 520) -> dict:
    """2단계 자리 — 가격·거래대금·시총·실적·정기보고서 접수일. 폐지 종목 하나, 정지 구간 하나가 들어 있다."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=days)
    close = 1000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (n, days)), axis=1))
    close[3, 300:] = np.nan                                 # 폐지
    close[5, 200:210] = np.nan                              # 거래정지
    tv_won = rng.uniform(5e9, 5e10, (n, days))
    tv_won[~np.isfinite(close)] = np.nan
    cap = np.exp(rng.normal(27, 0.5, (n, 1))) * np.ones((n, days))
    codes = [f"{i:06d}" for i in range(n)]
    ea, per = [], []
    for i, code in enumerate(codes):
        for q, (period, avail) in enumerate((("2019Q4", "2020-03-30"), ("2020Q1", "2020-05-15"), ("2020Q2", "2020-08-14"),
                                             ("2020Q3", "2020-11-16"), ("2020Q4", "2021-03-31"), ("2021Q1", "2021-05-17"),
                                             ("2021Q2", "2021-08-17"), ("2021Q3", "2021-11-15"))):
            ea.append((code, period, avail, avail, float(rng.normal(100 + i, 30)), 100.0))
            per.append((code, period, avail))
    earnings = pd.DataFrame(ea, columns=["code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior"])
    periodic = pd.DataFrame(per, columns=["code", "period", "per"])
    return {"dates": dates, "close": close, "tv_won": tv_won, "cap": cap, "codes": codes, "earnings": earnings, "periodic": periodic}


def run(inputs: dict, *, costs: bool, n_null: int = 5) -> dict:
    """1~9단계를 차례로. 이 함수는 게이트 뒤에서만 불린다."""
    dates, close, tv_won, cap, codes = (inputs[k] for k in ("dates", "close", "tv_won", "cap", "codes"))
    in_end = dates <= pd.Timestamp(END)
    dates, close, tv_won, cap = dates[in_end], close[:, in_end], tv_won[:, in_end], cap[:, in_end]

    # 2 데이터 — 경계 검사
    prices = pd.DataFrame({"code": np.repeat(codes, len(dates)), "date": np.tile(dates.strftime("%Y-%m-%d"), len(codes)),
                           "close": close.ravel()}).dropna(subset=["close"])  # 정문처럼 가격 없는 날은 행이 없다
    check_panels(prices, inputs["earnings"], END)

    # 3 알파 — 알게 된 날 규칙을 지키는 YoY
    sig, sig_info = yoy_strict_wide(inputs["earnings"], inputs["periodic"], codes, dates, END)
    check_signal(sig)

    # 1 유니버스 — 신호·유동성 뒤에 정해지는 띠라 신호 다음에 재료를 묶는다
    adv = _trailing_adv(tv_won / 1e6, len(dates), 20, close)
    in_universe = np.isfinite(close)
    grid = list(range(int(np.searchsorted(dates, np.datetime64("2020-06-01"))), len(dates), SPEC.step))

    def eligible_at(t):
        return rank_band_eligible(t, sig, adv, cap, in_universe[:, t], adv_floor=SPEC.adv_floor, cap_rank=SPEC.cap_rank)

    # 4 알파 검증 — 순위 IC, 라벨 겹침 퍼지. 합격 바는 사전등록한다
    ic_days, ics = daily_rank_ic(close, adv, cap, sig, grid[0], len(dates) - 1, SPEC.step, adv_floor=SPEC.adv_floor,
                                 cap_rank=SPEC.cap_rank)
    fold = fold_ic(ic_days, ics, grid[len(grid) // 2], SPEC.step)

    # 5 ML·6 위험모델·7 최적화 — 이 북은 일부러 안 쓴다(원장에 skip 과 사유). 위험모델 없이 U_t 만 넘긴다
    ret = np.full_like(close, np.nan)
    ret[:, 1:] = close[:, 1:] / close[:, :-1] - 1.0
    delist_after = np.full(len(codes), 10**9, dtype=np.int64)
    delist_after[3] = 299
    ctx = Ctx(dates=dates, C=close, tv_won=tv_won, adv=adv, cap=cap, sig=sig, ret=ret,
              lab=np.full(close.shape, "G", dtype=object), market=np.array(["KOSPI"] * len(codes), dtype=object), grid=grid,
              models=models_from_mask(in_universe, grid), delist_after=delist_after)

    # 8 비용·9 백테스트 — 정가 비용은 core 충격 레이어가 있어야 한다. 없으면 --no-costs 로 총수익만 본다
    flags = Flags() if costs else Flags(costs=False)
    recs = simulate(ctx, "R", flags=flags, spec=SPEC)
    violations = check_invariants(recs, "R", spec=SPEC, costs=costs)
    for r in recs:
        check_weights(np.array([w for _, w, _, _ in r.contrib if w > 0]))
    net = oos(step_returns(recs, 2.0), WINDOW)
    factor = np.column_stack([net["bench"].to_numpy()])
    nulls = permutation_nulls(ctx, PRIMARY, None, None, workers=1, n=n_null, window=WINDOW, spec=SPEC, kinds=("R",), flags=flags)
    # 널의 불변식은 비용을 끈 경로에서 비용 칸을 본다 — 거기서는 비용/회전 위반만 따로 뺀다
    null_viol = [v for v in nulls["violations"] if costs or "비용/회전" not in v]
    return {"recs": recs, "summary": book_summary(recs, WINDOW, (2020, 2021), ann=ANN), "violations": violations + null_viol,
            "alpha": ols_hac(net["net"].to_numpy(), factor), "null_ir": nulls["R"], "ic_days": ic_days, "fold_ic": fold,
            "signal_info": sig_info, "eligible_first": eligible_at(grid[0])}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-costs", action="store_true", help="core 충격 레이어가 없을 때(0.7.0 핀). 판정에는 쓰지 않는다")
    args = ap.parse_args(argv)
    require_stage(*PIPELINE)
    out = run(synthetic_inputs(), costs=not args.no_costs)
    print({k: out["summary"][k] for k in ("IR_2x", "EV_2x", "oos_steps", "names_mean")})
    return 1 if out["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
