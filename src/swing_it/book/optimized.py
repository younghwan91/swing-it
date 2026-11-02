"""7단계(최적화 북) — α 지도, 먼지 처리, 위험 맞춤, 목표 비중. core ``portfolio.optimize`` 는 부를 때 import 한다.

**알려진 결함 — 재사용 전에 고칠 것.** 얼린 칸(거래 불가 보유)이 표류로 종목 상한·업종 밴드를 넘으면 문제가 infeasible 이 되어 그 스텝 전체가 거래를
멈춘다(B 2022-04-06 k=33 ``solver_fail``). 얼린 칸의 상한·밴드를 분리해 풀어야 하지만 O 숫자가 바뀌므로 B 재현 때문에 고치지 않았다. 새 연구는 이 수정과
사전등록 정정을 먼저 한다(``docs/PIPELINE.md`` 7단계).

**재현 조건.** O 는 솔버 판에 달린다. ``uv.lock`` 에 cvxpy·clarabel 이 없어 새 환경에서 O 값이 5e-6 안팎 움직였다. R 은 솔버를 안 탄다.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from swing_it.book.context import delisted
from swing_it.book.risk import active_risk, factor_share, sub_model, top_exposures
from swing_it.book.spec import DEFAULT_SPEC, BookSpec
from swing_it.core_layers import core_layer

DUST_EPS = 1e-6     # 해 직후 |w| < max(1e-6, 1주 가격/AUM) 은 0(솔버 먼지)
FACTOR_ALARM = 0.20  # 사전 능동분산 중 팩터 몫 경보
LAM_LO, LAM_HI, LAM_ITERS, LAM_TOL = 1e-2, 1e5, 40, 0.02
BUY_EPS = 1e-6      # 매수 금지 불변식의 솔버 허용오차


def rank_z(vals: np.ndarray) -> np.ndarray:
    """z_i = Φ⁻¹((순위_i − 0.5)/n), 오름차순 평균 순위. NaN 은 z=0(널의 순열 NaN, PO-6)."""
    vals = np.asarray(vals, dtype=np.float64)
    out = np.zeros(vals.shape)
    fin = np.isfinite(vals)
    n = int(fin.sum())
    if n:
        r = stats.rankdata(vals[fin], method="average")
        out[fin] = stats.norm.ppf((r - 0.5) / n)
    return out


def alpha_vec(ic: float, d_daily: np.ndarray, z: np.ndarray, h: int) -> np.ndarray:
    """α_i = IC_k · √(H·D_i) · z_i (Grinold)."""
    return ic * np.sqrt(h * np.asarray(d_daily, dtype=np.float64)) * z


def capped_rescale(x: np.ndarray, total: float, cap: float | None) -> np.ndarray:
    """PO-9 — 음 아닌 ``x`` 를 합 ``total`` 로 비례 조정하되 어느 칸도 ``cap`` 을 넘지 않게 한다(물 채우기).

    넘친 몫은 cap 에 닿지 않은 칸에 비례로 다시 나누고, 모두 cap 이면 남는다(합 < total — 부르는 쪽에서 현금).
    ``cap`` 이 None 이면 그냥 비례(R 북 — 상한 없음). 줄이는 방향(total < Σx)은 cap 을 건드리지 않는다.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    s = x.sum()
    if np.isnan(s) or np.isnan(total):
        raise ValueError("capped_rescale 에 NaN 이 들어왔다 — 합이 조용히 NaN 이 된다")
    if total <= 0:
        return np.zeros_like(x)     # 방이 없다. 입력을 돌려주면 얼린 칸과 합쳐 1 을 넘는다
    if s <= 0:
        return x
    if cap is None:
        return x * (total / s)
    x = np.minimum(x, cap)
    fixed = np.zeros(x.size, dtype=bool)
    for _ in range(x.size + 1):
        rem = total - x[fixed].sum()
        free = ~fixed & (x > 0)
        sf = x[free].sum()
        if sf <= 0 or rem <= 0:
            break
        y = x[free] * (rem / sf)
        over = y > cap
        if not over.any():
            x[free] = y
            break
        idx = np.flatnonzero(free)
        x[idx[over]] = cap
        fixed[idx[over]] = True
    return x


def clean_dust(w: np.ndarray, price: np.ndarray, aum: float, fixed: np.ndarray,
               cap: float | None = DEFAULT_SPEC.max_weight) -> tuple[np.ndarray, int]:
    """PO-7 M2 — 솔버 먼지 제거. 고정(``no_trade``) 아닌 종목의 w < max(1e-6, 1주 가격/AUM) 은 0, 음수도 0.

    지운 만큼은 남은(고정 아닌, 양수) 비중에 비례로 돌려 합을 지킨다 — PO-9: ``cap`` 을 넘기지 않게(:func:`capped_rescale`,
    모두 cap 이면 그 몫은 남아 현금이 된다). 반환 ``(w, 지운 수)``.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    thr = np.maximum(DUST_EPS, np.where(np.isfinite(price), price, 0.0) / aum)
    kill = ~fixed & (w < thr) & (w != 0.0)
    total = w.sum()
    w[kill] = 0.0
    keep = ~fixed & (w > 0)
    s_keep = w[keep].sum()
    if s_keep > 0:
        w[keep] = capped_rescale(w[keep], total - w[fixed].sum(), cap)
    return w, int(kill.sum())


def match_lambda(solve, risk_of, target: float, *, lo: float = LAM_LO, hi: float = LAM_HI, iters: int = LAM_ITERS,
                 tol: float = LAM_TOL, strict: bool = False, statuses: "dict | None" = None):
    """로그 공간 이분법 [1e-2, 1e5], 최대 40회, ±2%. 위험은 λ 에 감소한다.

    ``optimal`` 이 아닌 풀이에서 멈추지 않는다(PO-7 MINOR 4): 그 반복은 세고, 가장 가까운 성공 λ 쪽으로 구간을 좁혀 계속한다.

    Returns: ``(res, lam, hit, n_solves, cause)``. 못 맞추면 ``optimal`` 인 반복 중 |log(위험/목표)| 가 가장 작은 것(PO-6), 그것도 없으면
    ``None``. ``cause`` ∈ {"", "solver"(성공 풀이 없음), "up"(모든 성공 풀이의 위험 < 목표 — 하한 λ 에서도 못 올림),
    "down"(모든 성공 풀이의 위험 > 목표), "gap"(목표를 사이에 두었지만 ±2% 창을 못 맞춤 — 위험이 λ 에 불연속)}.

    ``statuses`` 에 사전을 주면 실패 풀이의 ``status`` 별 횟수를 채운다. 목표가 양의 유한수가 아니면 ``ValueError``.

    ``strict`` 는 새 연구용이다. 기본값은 B 재현이다 — ``lam_causes`` 가 B.json 에 있어 라벨을 바꿀 수 없다. ``strict`` 이면 성공 풀이가 없을 때 전부 infeasible 이면
    "infeasible"(문제 자체), 아니면 "solver" 이고, 실패 풀이가 섞이면 "up"·"down" 대신 "partial" 이다. 실패한 λ 에서 위험이 어땠는지 모르니 "하한에서도 못 올림"을 증명하지 못한다.
    """
    if not (math.isfinite(target) and target > 0):
        raise ValueError(f"위험 목표는 양의 유한수여야 한다: {target}")
    statuses = {} if statuses is None else statuses
    lo, hi = math.log(lo), math.log(hi)
    best, best_gap, best_lam, n = None, math.inf, float("nan"), 0
    below = above = False
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lam = math.exp(mid)
        res = solve(lam)
        n += 1
        if not res.ok:
            status = getattr(res, "status", "failed")
            statuses[status] = statuses.get(status, 0) + 1
            if best is None:
                hi = mid  # 성공 풀이가 아직 없으면 작은 λ 쪽부터
            elif best_lam > lam:
                lo = mid
            else:
                hi = mid
            continue
        risk = risk_of(res.weights)
        gap = abs(math.log(max(risk, 1e-300) / target))
        if gap < best_gap:
            best, best_gap, best_lam = res, gap, lam
        if abs(risk / target - 1.0) <= tol:
            return res, lam, True, n, ""
        if risk > target:
            above = True
            lo = mid
        else:
            below = True
            hi = mid
    cause = "solver" if best is None else ("gap" if (above and below) else ("up" if below else "down"))
    if strict and statuses:
        if best is None:
            cause = "infeasible" if set(statuses) == {"infeasible"} else "solver"
        elif cause in ("up", "down"):
            cause = "partial"
    return best, best_lam, False, n, cause


def o_target(ctx, cell, t, f, el, w_t, held, in_u, tradeable, costable, frozen, vol_all, rank_sig, ic_by_year,
             sigma_r, built, solver_opts, spec: BookSpec = DEFAULT_SPEC):
    """O 의 목표 비중(§2·PO-6). 문제 종목 P = eligible(t) ∪ (보유 ∩ U_t ∩ 폐지 아님)."""
    opt = core_layer("krx_quant_core.portfolio.optimize")
    Constraints, CostParams, optimize = opt.Constraints, opt.CostParams, opt.optimize

    nN = ctx.C.shape[0]
    info = {"status": "optimal", "lam": float("nan"), "lam_hit": True, "est_cost": float("nan")}
    alive = ~delisted(ctx, f)
    P = np.union1d(el, np.flatnonzero(held & in_u & alive)).astype(int)
    want = np.zeros(nN)
    if el.size == 0 or P.size == 0:
        info["status"] = "no_universe"
        return want, info
    is_el = np.isin(P, el)
    sub = sub_model(ctx.models[t], P, cell.h_alpha)
    D_daily = sub.D / cell.h_alpha
    z = np.zeros(P.size)
    zv = rank_z(rank_sig[el, t])
    info["perm_nan_share"] = float(np.mean(~np.isfinite(rank_sig[el, t])))
    pos = {int(c): i for i, c in enumerate(P)}
    for c_, zz in zip(el, zv):
        z[pos[int(c_)]] = zz
    ic = ic_by_year[int(ctx.dates[t].year)]
    alpha = alpha_vec(ic, D_daily, z, cell.h_alpha)
    b = np.where(is_el, 1.0 / el.size, 0.0)
    Sb = sub.X @ (sub.F @ (sub.X.T @ b)) + sub.D * b
    w_prev = w_t[P]
    no_trade = ~(tradeable[P] & costable[P]) | frozen[P]
    outside_frozen = frozen & ~np.isin(np.arange(nN), P)
    budget = 1.0 - float(w_t[outside_frozen].sum())
    con = Constraints(budget=budget, long_only=True, max_weight=spec.max_weight, sector=ctx.lab[P, t], benchmark=b,
                      sector_band=cell.band, max_turnover=spec.max_turnover if built else None, max_participation=spec.max_participation,
                      no_buy=~is_el, no_trade=no_trade)
    adv_won = np.where(costable[P], ctx.adv[P, t] * 1e6, 1e12)
    vol = np.where(costable[P], vol_all[P], 0.0)
    cp_ = CostParams(trade_date=ctx.dates[f].date(), market=ctx.market[P], half_spread_bp=ctx.hs_bp[P], y=1.0)
    target = float(sigma_r[t]) * math.sqrt(cell.h_alpha / spec.step)  # σ_R 은 20일 단위로 받는다

    def solve(lam):
        return optimize(alpha + lam * Sb, sub, w_prev, aum=cell.aum, adv_value=adv_won, daily_vol=vol, cost_params=cp_,
                        constraints=con, risk_aversion=lam, solver_opts=solver_opts)

    statuses: dict = {}
    res, lam, hit, _, cause = match_lambda(solve, lambda w: active_risk(sub, w - b), target, strict=spec.strict_lam_cause, statuses=statuses)
    info.update({"lam": lam, "lam_hit": bool(hit), "lam_cause": cause, "lam_statuses": statuses})
    if res is None:
        info["status"] = "solver_fail"
        return want, info
    w_sol = np.asarray(res.weights, dtype=np.float64)
    dw_sol = np.where(no_trade, 0.0, w_sol - w_prev)
    info["n_buy_outside"] = int(((dw_sol > BUY_EPS) & ~is_el).sum())    # 불변식 재료: 해가 no_buy·참여율을 지켰나
    info["max_part"] = float((np.abs(dw_sol) * cell.aum / adv_won).max())
    w_clean, n_dust = clean_dust(w_sol, ctx.Cff[P, t], cell.aum, no_trade, cap=spec.max_weight)  # PO-7 M2
    want[P] = w_clean
    info.update({"n_dust": n_dust, "dust_mass": float(np.abs(w_clean - w_sol).sum()) / 2.0})
    info["est_cost"] = float(res.est_cost)
    d = w_clean - b
    info["exante_20"] = active_risk(sub, d) * math.sqrt(spec.step / cell.h_alpha)
    info["factor_share"] = factor_share(sub, d)
    if info["factor_share"] > FACTOR_ALARM:
        info["alarm_top3"] = top_exposures(sub, d)
    return want, info
