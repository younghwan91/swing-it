"""회계 엔진과 core ``staggered_tranche_backtest`` 의 패리티 — 차이·개수만 낸다.

패리티 모드(지연 0·무비용·강제 매도 끔·현금 없음)는 core 함수의 이식이다. 마스크·트랜치 선택·폐지 산술만 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _trailing_adv, staggered_tranche_backtest

from swing_it.book.context import Ctx
from swing_it.book.rank import tranche_names
from swing_it.book.spec import DEFAULT_SPEC, BookSpec
from swing_it.features.universe import m0_mask, rank_band_eligible

PARITY_TOL = 1e-10


def parity_signal(sig: np.ndarray, adv: np.ndarray, grid: list[int], in_u_of, *, adv_floor: float = DEFAULT_SPEC.adv_floor) -> np.ndarray:
    """격자일 t_k 의 M0 안에서만 유한한 신호 패널(core 에 넘긴다)."""
    out = np.full_like(sig, np.nan)
    for t in grid:
        m = m0_mask(t, sig, adv, in_u_of(t), adv_floor=adv_floor)
        out[m, t] = sig[m, t]
    return out


def parity_book(ctx: Ctx, lag: int = 0, spec: BookSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """패리티 모드: 지연 ``lag``·무비용·강제 매도 끔·현금 없음, 트랜치는 결성일 목록, 스텝마다 수익 유한 종목 nanmean.

    ``lag=0`` 이 core ``staggered_tranche_backtest(..., start_index=t_0, min_names=20, delisting_exit=True)`` 와 같아야 한다.
    """
    C, Cff = ctx.C, ctx.Cff
    nD = C.shape[1]
    t0 = ctx.grid[0]
    books: dict[int, np.ndarray | None] = {}
    rows = []
    for t in range(t0, nD - spec.step - 1 - lag, spec.step):
        uni = rank_band_eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t), adv_floor=spec.adv_floor, cap_rank=spec.cap_rank)
        if uni.size < spec.min_names:
            continue
        f, e = t + lag, t + lag + spec.step
        ret = C[:, e] / C[:, f] - 1.0
        gone = np.isfinite(C[:, f]) & ~np.isfinite(C[:, e])
        ret[gone] = Cff[gone, e] / C[gone, f] - 1.0
        bench = float(np.nanmean(ret[uni]))
        tb = []
        for k in range(spec.n_tranches):
            tj = t - k * spec.step
            if tj < t0:
                continue
            if tj not in books:
                books[tj] = tranche_names(tj, ctx.sig, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(tj), spec.top_n, spec.min_names,
                                         adv_floor=spec.adv_floor, cap_rank=spec.cap_rank)
            b = books[tj]
            if b is not None:
                tb.append(float(np.nanmean(ret[b])))
        if tb:
            rows.append({"t": t, "date": str(ctx.dates[t].date()), "book": float(np.mean(tb)), "bench": bench,
                         "n_universe": int(uni.size)})
    return pd.DataFrame(rows)


def core_parity(ctx: Ctx, spec: BookSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """core ``staggered_tranche_backtest`` 를 같은 규칙으로(신호는 격자일 M0 안에서만 유한)."""
    sp = parity_signal(ctx.sig, ctx.adv, ctx.grid, ctx.in_u, adv_floor=spec.adv_floor)
    tv_m = ctx.tv_won / 1e6
    per, _ = staggered_tranche_backtest(
        ctx.C, tv_m, sp, list(range(ctx.C.shape[1])), horizon=spec.horizon, step=spec.step, top_n=spec.top_n, adv_floor=spec.adv_floor,
        start_index=ctx.grid[0], min_names=spec.min_names, cap_array=ctx.cap, cap_rank=spec.cap_rank, delisting_exit=True)
    per = per.rename(columns={"date": "t"})
    return per[["t", "book", "bench", "n_universe"]]


def parity_check(ctx: Ctx, core_adv_matches: bool | None = None, spec: BookSpec = DEFAULT_SPEC) -> dict:
    """하네스 패리티 대 core. 반환은 차이·개수뿐(성과 숫자 없음)."""
    h = parity_book(ctx, 0, spec)
    c = core_parity(ctx, spec)
    same_t = h["t"].tolist() == c["t"].tolist()
    out = {"steps_harness": int(len(h)), "steps_core": int(len(c)), "same_steps": bool(same_t)}
    if same_t and len(h):
        db = np.abs(h["book"].to_numpy() - c["book"].to_numpy())
        dc = np.abs(h["bench"].to_numpy() - c["bench"].to_numpy())
        out.update({"max_abs_diff_book": float(np.nanmax(db)), "max_abs_diff_bench": float(np.nanmax(dc)),
                    "nan_mismatch": int((np.isnan(h["book"]) != np.isnan(c["book"])).sum()
                                        + (np.isnan(h["bench"]) != np.isnan(c["bench"])).sum()),
                    "n_universe_same": bool((h["n_universe"].to_numpy() == c["n_universe"].to_numpy()).all()),
                    "first_date": h["date"].iloc[0], "last_date": h["date"].iloc[-1]})
        out["pass"] = bool(out["max_abs_diff_book"] <= PARITY_TOL and out["max_abs_diff_bench"] <= PARITY_TOL
                           and out["nan_mismatch"] == 0 and out["n_universe_same"])
    else:
        out["pass"] = False
    if core_adv_matches is not None:
        out["adv_bytes_same"] = bool(core_adv_matches)
        out["pass"] = out["pass"] and bool(core_adv_matches)
    return out


def full_parity(ctx: Ctx, spec: BookSpec = DEFAULT_SPEC) -> dict:
    """패리티 + ADV 바이트 + 격자 전부 적격 집합.

    패리티는 **마스크·트랜치 선택·폐지 산술만** 검증한다(``parity_book`` 은 core 함수의 이식이다). 엔진(``simulate``·``o_target``·비용·널)은
    :func:`~swing_it.book.accounting.check_invariants` 와 재계산으로 검증한다.
    """
    tv_m = ctx.tv_won / 1e6
    adv_core = _trailing_adv(tv_m, tv_m.shape[1], 20, ctx.C)
    adv_same = bool(np.array_equal(adv_core[:, 20:], ctx.adv[:, 20:], equal_nan=True))
    pc = parity_check(ctx, adv_same, spec)
    sp = parity_signal(ctx.sig, ctx.adv, ctx.grid, ctx.in_u, adv_floor=spec.adv_floor)
    allu = np.ones(ctx.C.shape[0], bool)
    kw = {"adv_floor": spec.adv_floor, "cap_rank": spec.cap_rank}
    elig_same = all(np.array_equal(np.sort(rank_band_eligible(t, ctx.sig, ctx.adv, ctx.cap, ctx.in_u(t), **kw)),
                                   np.sort(rank_band_eligible(t, sp, ctx.adv, ctx.cap, allu, **kw))) for t in ctx.grid)
    pc["eligible_same_all_grid"] = bool(elig_same)
    pc["pass"] = bool(pc["pass"] and elig_same)
    return pc
