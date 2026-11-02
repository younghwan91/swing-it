"""9단계 회계 엔진 — 두 북이 같은 회계를 탄다. 비용 1배로 기록하고 2배는 배수로 낸다(:func:`swing_it.book.metrics.step_returns`)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from swing_it.book.context import Ctx, StepRec, delisted, index_mask
from swing_it.book.costs import linear_bp, trade_costs, trailing_vol
from swing_it.book.optimized import FACTOR_ALARM, capped_rescale, o_target
from swing_it.book.rank import r_target, tranche_names
from swing_it.book.risk import active_risk, factor_share, sub_model, top_exposures
from swing_it.book.spec import DEFAULT_SPEC, PRIMARY, BookSpec, Cell, Flags
from swing_it.features.universe import rank_band_eligible, u_exclusion, u_exclusion_causes

INV_EPS = 1e-6      # 불변식 허용오차
COST_BP_PER_TURN = (1.5, 300.0)  # 1배 비용/회전(bp): 수수료 하한 ~ 상한
LIMIT_UP_RET = 0.29  # 체결일 종가 수익 ≥ 29% 면 상한가 체결로 센다


def _equal_weight(n: int, idx: np.ndarray) -> np.ndarray:
    """벤치 — ``idx`` 동일가중, 합 1. ``idx`` 가 비면 부르지 않는다."""
    b = np.zeros(n)
    b[idx] = 1.0 / idx.size
    return b


def _r_exante(ctx: Ctx, rec: StepRec, t: int, want: np.ndarray, in_u: np.ndarray, el: np.ndarray, spec: BookSpec) -> None:
    """R 의 사전 위험(결정 목표 비중, 결정일 모델, U_t 안, 합 1 로 맞춤). 모델이 없으면 칸을 NaN 으로 둔다."""
    w_r = want * in_u
    if not (w_r.sum() > 0 and el.size and ctx.models[t][1] is not None):
        return
    nN = want.size
    w_r = w_r / w_r.sum()
    rows = np.flatnonzero(in_u & ((w_r > 0) | index_mask(nN, el)))
    sub = sub_model(ctx.models[t], rows, spec.step)
    d = (w_r - _equal_weight(nN, el))[rows]
    rec.exante_20 = active_risk(sub, d)
    rec.factor_share = factor_share(sub, d)
    if rec.factor_share > FACTOR_ALARM:
        rec.alarm_top3 = top_exposures(sub, d)


def _rebalance(want: np.ndarray, w_f: np.ndarray, frozen: np.ndarray, tradeable: np.ndarray, forced: np.ndarray, *, trade: bool,
               cap: "float | None") -> tuple[np.ndarray, float, float]:
    """체결 뒤 비중·현금·재비례 몫. ``trade`` 가 거짓이면(최적화 실패) 거래 안 함 — 강제 매도만, 대금은 현금."""
    renorm = 0.0
    if trade:
        free = tradeable & ~frozen
        room = 1.0 - w_f[frozen].sum()
        s = want[free].sum()
        new = np.where(frozen, w_f, 0.0)
        if s > 0:  # PO-9: O 는 상한을 지키며 채운다(넘치는 몫은 현금)
            new[free] = capped_rescale(want[free], room, cap)
            renorm = abs(room - s)
    else:
        new = w_f.copy()
        new[forced & tradeable] = 0.0
    return new, 1.0 - new.sum(), renorm


def _o_invariant_inputs(ctx: Ctx, rec: StepRec, cell: Cell, spec: BookSpec, t: int, el: np.ndarray, new: np.ndarray, w_f: np.ndarray, w_t: np.ndarray,
                        frozen: np.ndarray, forced: np.ndarray, tradeable: np.ndarray, held: np.ndarray, in_u: np.ndarray, renorm: float, built: bool) -> None:
    """O 불변식 재료 — 업종 편차와 허용폭, 구축 뒤 회전율 한도."""
    nN = new.size
    bvec = _equal_weight(nN, el)
    labs = ctx.lab[:, t]
    groups = {g for g in labs[(new > 0) | (bvec > 0)] if g is not None}
    rec.sector_dev = max((abs(float(new[labs == g].sum() - bvec[labs == g].sum())) for g in groups), default=0.0)
    outside = float(w_f[frozen & ~index_mask(nN, np.union1d(el, np.flatnonzero(held & in_u)))].sum())
    rec.sector_tol = cell.band + INV_EPS + outside + renorm + rec.dust_mass  # PO-9: 먼지 재분배로 옮긴 몫
    if built:
        drift = float(np.abs(w_f - w_t).sum())  # t→f 표류 + 폐지 현금화
        rec.turn_bound = (spec.max_turnover + drift + float(w_f[forced & tradeable].sum()) + renorm
                          + 2.0 * rec.dust_mass + INV_EPS)


def simulate(ctx: Ctx, kind: str, cell: Cell = PRIMARY, flags: Flags = Flags(), rank_sig=None, ic_by_year=None,
             sigma_r=None, solver_opts=None, universe=None, max_weight: float | None = None,
             spec: BookSpec = DEFAULT_SPEC) -> list[StepRec]:
    """한 북의 회계. ``kind`` ∈ {"R", "O"}. ``rank_sig`` 는 순위에 쓰는 신호(널은 순열). 규칙은 ``spec``·``cell``·``flags`` 로 받는다.

    주입점(기본 ``None`` 이면 PEAD 단계 B 경로와 비트 동일): ``universe(t)`` = 적격·벤치 종목(행 번호),
    ``max_weight`` = 종목 상한. IC·목표 위험 상수는 ``ic_by_year``·``sigma_r`` 사전으로 이미 넣을 수 있다.

    단위: 체결 직후 NAV 를 1 로 두고(총액 경로), 스텝 순수익은 r_k(m) = (1 − m·cost1_k)·(1 + gross_k) − 1 로 낸다.
    비용은 NAV 에 비례해 빠지므로 비중 경로는 비용 배수와 무관하다 — 1·2배가 같은 거래열을 쓴다(최적화기는 1배로 푼다).
    """
    assert kind in ("R", "O")
    if kind == "O":
        assert flags == Flags(), "O 는 실제 규칙으로만 돈다"
        assert ic_by_year is not None and sigma_r is not None
    if not flags.drift:
        raise ValueError("표류 끈 규칙은 parity_book 이다")
    rank_sig = ctx.sig if rank_sig is None else rank_sig
    if max_weight is not None:
        spec = replace(spec, max_weight=max_weight)
    C, Cff = ctx.C, ctx.Cff
    nN, nD = C.shape
    lag = flags.lag
    u = np.zeros(nN)  # 단위 NAV 당 주식 수
    cash = 1.0
    tranches: dict[int, np.ndarray | None] = {}
    out: list[StepRec] = []
    built = False
    for k, t in enumerate(ctx.grid):
        f = t + lag
        f_next = (ctx.grid[k + 1] if k + 1 < len(ctx.grid) else t + spec.step) + lag
        if f_next > nD - 1:
            break
        day = ctx.dates[f].date()
        in_u = ctx.in_u(t)
        el = (rank_band_eligible(t, ctx.sig, ctx.adv, ctx.cap, in_u, adv_floor=spec.adv_floor, cap_rank=spec.cap_rank)
              if universe is None else np.asarray(universe(t), dtype=int))
        rec = StepRec(k=k, t=t, date=str(ctx.dates[t].date()), gross=float("nan"), cost1=0.0, bench=float("nan"),
                      turnover=0.0, exante_20=float("nan"), factor_share=float("nan"), f=f, f_next=f_next)
        if universe is None:
            rec.u_excl = u_exclusion(t, ctx.sig, ctx.adv, in_u, adv_floor=spec.adv_floor)
            why = u_exclusion_causes(t, ctx.sig, ctx.adv, ctx.cap, in_u, adv_floor=spec.adv_floor)
            rec.u_excl_cap_missing, rec.u_excl_model, rec.cap_missing_in_u = why["cap_missing"], why["risk_model"], why["cap_missing_in_u"]
        if kind == "R":
            tranches[k] = tranche_names(t, rank_sig, ctx.sig, ctx.adv, ctx.cap, in_u, spec.top_n, spec.min_names,
                                        adv_floor=spec.adv_floor, cap_rank=spec.cap_rank, min_after_nan=spec.min_names_after_nan)
        # 결정일 t 의 표류 비중
        val_t = np.where(u > 0, u * Cff[:, t], 0.0)
        nav_t = val_t.sum() + cash
        w_t = val_t / nav_t
        held = w_t > 0
        # 체결일 f 의 값
        val_f = np.where(u > 0, u * Cff[:, f], 0.0)
        nav_f = val_f.sum() + cash
        w_f = val_f / nav_f
        cost_vec = np.zeros(nN)
        # ④ 폐지 청산(마지막 종가, 선형 매도 비용)
        if flags.delist_cash:
            gone = held & delisted(ctx, f)
            if gone.any():
                g = np.flatnonzero(gone)
                if flags.costs:
                    cost_vec[g] += linear_bp(np.full(g.size, "SELL"), ctx.market[g], day, ctx.hs_bp[g]) / 1e4 * w_f[g]
                rec.delist_mass = float(w_f[g].sum())
                w_f[g] = 0.0
                rec.n_delist = int(g.size)
        held_f = w_f > 0
        # ⑤ 거래 가능 여부(체결일)
        if flags.no_trade:
            tradeable = np.isfinite(C[:, f]) & (np.nan_to_num(ctx.tv_won[:, f]) > 0)
        else:
            tradeable = np.isfinite(Cff[:, f])
        vol = trailing_vol(ctx.ret, t)
        adv_won = ctx.adv[:, t] * 1e6
        costable = np.isfinite(vol) & np.isfinite(adv_won) & (adv_won >= 1e6) & (vol <= 1.0)
        if flags.costs:
            tradeable &= costable
        frozen = held_f & ~tradeable
        rec.n_frozen = int(frozen.sum())
        forced = held_f & ~in_u if flags.forced_sale else np.zeros(nN, bool)
        rec.n_forced = int((forced & tradeable).sum())

        # 목표 비중(체결일 f, 합 ≤ 1)
        trade = True
        if kind == "R":
            want, rec.n_tranches = r_target(ctx, k, t, tranches, flags, f, spec=spec)
            rec.n_untradeable = int(((want > 0) & ~tradeable).sum())
            _r_exante(ctx, rec, t, want, in_u, el, spec)
            want = np.where(tradeable, want, 0.0)
        else:
            want, info = o_target(ctx, cell, t, f, el, w_t, held, in_u, tradeable, costable, frozen, vol,
                                  rank_sig, ic_by_year, sigma_r, built, solver_opts, spec)
            for key, v in info.items():
                setattr(rec, key, v)
            trade = rec.status == "optimal"
        new, new_cash, renorm = _rebalance(want, w_f, frozen, tradeable, forced, trade=trade, cap=spec.max_weight if kind == "O" else None)
        dw = np.where(tradeable, new - w_f, 0.0)
        new = np.where(tradeable, new, w_f)
        if flags.costs and np.any(dw != 0):
            cost_vec += trade_costs(dw, np.where(costable, adv_won, 1e12), np.where(costable, vol, 0.0), ctx.market,
                                    day, ctx.hs_bp, cell.aum)
        rec.turnover = float(np.abs(dw).sum())
        rec.cost1 = float(cost_vec.sum())
        rec.n_names = int((new > 0).sum())
        # PO-7 M3 — 성과와 무관한 불변식 재료(체결 뒤)
        rec.cash = float(new_cash)
        rec.min_w = float(new.min()) if new.size else 0.0
        ctrl = tradeable & ~frozen  # PO-9: 상한 불변식은 이 스텝에 최적화기가 정한 종목만(얼린 보유는 따로 적는다)
        rec.max_w = float(new[ctrl].max()) if ctrl.any() else 0.0
        rec.max_w_frozen = float(new[frozen].max()) if frozen.any() else 0.0
        if kind == "O" and rec.status == "optimal" and el.size:
            _o_invariant_inputs(ctx, rec, cell, spec, t, el, new, w_f, w_t, frozen, forced, tradeable, held, in_u, renorm, built)
        if flags.lag:
            rf = ctx.ret[:, f]
            rec.n_buy_limit_up = int(((dw > 0) & np.isfinite(rf) & (rf >= LIMIT_UP_RET)).sum())
        # 보유: 체결 뒤 NAV=1 → 다음 체결 전
        px = np.where(new > 0, Cff[:, f], np.nan)
        u = np.where(new > 0, new / px, 0.0)
        cash = max(new_cash, 0.0)
        g_i = np.where(new > 0, Cff[:, f_next] / Cff[:, f] - 1.0, 0.0)
        rec.gross = float((new * g_i).sum())  # 현금은 수익 0
        nz = np.flatnonzero((new > 0) | (cost_vec > 0))
        rec.contrib = [(int(i), float(new[i]), float(g_i[i]), float(cost_vec[i])) for i in nz]
        # 벤치: eligible(t) 동일가중 매수 보유, 체결일 가격 있는 종목
        bel = el[np.isfinite(C[el, f])]
        if bel.size:
            rec.bench = float(np.mean(Cff[bel, f_next] / C[bel, f] - 1.0))
        built = built or rec.n_names > 0
        out.append(rec)
    return out


def check_invariants(recs: list[StepRec], kind: str, cell: Cell = PRIMARY, max_weight: "float | None" = None,
                     spec: BookSpec = DEFAULT_SPEC, costs: bool = True) -> list[str]:
    """성과와 무관한 불변식(스텝마다). 위반 문장 목록을 돌려준다(비면 통과). 러너는 하나라도 있으면 결과를 쓰지 않는다.

    NaN 은 통과가 아니다 — 비교는 "지켜졌다"를 묻는 꼴로 쓴다. 스텝이 없어도 위반이다(빈 장부는 아무것도 증명하지 않는다). 상한 기본값은 ``spec.max_weight``.

    - Σw + 현금 = 1, w ≥ −ε. O: (거래 가능·안 얼린 종목) max w ≤ 상한 + ε, 업종 |Σw_g − b_g| ≤ 밴드 + ε + (문제 밖 얼린 몫 + 재비례 + 먼지),
      구축 뒤 회전율 ≤ 상한 + 표류·폐지 + 강제 매도 + 재비례 + 2·먼지. 해는 no_buy 종목을 사지 않고 참여율 상한을 지킨다.
    - 비용: cost1 ≥ 0, 회전이 있으면 cost1/회전 ∈ [1.5bp, 300bp](1배).
    - R: 예열 뒤(k ≥ n−1) 활성 트랜치 n.
    """
    if max_weight is not None:
        spec = replace(spec, max_weight=max_weight)
    out = []
    if not recs:
        out.append(f"{kind}: 스텝이 없다 — 검사할 장부가 없다")
    cap = spec.max_weight
    lo_bp, hi_bp = COST_BP_PER_TURN
    for r in recs:
        tag = f"{kind}:{cell.name if kind == 'O' else 'R'} k={r.k} {r.date}"
        tot = r.cash + (sum(w for _, w, _, _ in r.contrib if w > 0))
        if not abs(tot - 1.0) <= INV_EPS:
            out.append(f"{tag} Σw+현금 {tot:.9f} ≠ 1")
        if not (r.min_w >= -INV_EPS and r.cash >= -INV_EPS):
            out.append(f"{tag} 음수 비중 {r.min_w:.3e}·현금 {r.cash:.3e}")
        if not r.cost1 >= 0:
            out.append(f"{tag} 비용 음수·NaN {r.cost1}")
        if not (np.isfinite(r.gross) and r.turnover >= 0):
            out.append(f"{tag} 수익·회전 NaN: gross {r.gross}, turnover {r.turnover}")
        traded = r.turnover + r.delist_mass
        if costs and traded > 1e-9:     # costs=False 는 비용 끈 회계(Flags(costs=False))를 시험할 때만 쓴다
            bp = r.cost1 / traded * 1e4
            if not (lo_bp - 1e-6 <= bp <= hi_bp):
                out.append(f"{tag} 비용/회전 {bp:.2f}bp 가 [{lo_bp}, {hi_bp}] 밖")
        if kind == "O" and r.status == "optimal" and r.n_names:
            if not r.max_w <= cap + INV_EPS:
                out.append(f"{tag} max w {r.max_w:.6f} > {cap}")
            if np.isfinite(r.sector_dev) and r.sector_dev > r.sector_tol:
                out.append(f"{tag} 업종 편차 {r.sector_dev:.6f} > 허용 {r.sector_tol:.6f}")
            if np.isfinite(r.turn_bound) and r.turnover > r.turn_bound:
                out.append(f"{tag} 회전율 {r.turnover:.6f} > 한도 {r.turn_bound:.6f}")
            if r.n_buy_outside:
                out.append(f"{tag} 매수 금지 종목 {r.n_buy_outside}개를 샀다")
            if np.isfinite(r.max_part) and r.max_part > spec.max_participation + INV_EPS:
                out.append(f"{tag} 참여율 {r.max_part:.6f} > {spec.max_participation}")
        if kind == "R" and r.k >= spec.n_tranches - 1 and r.n_tranches != spec.n_tranches:
            out.append(f"{tag} 활성 트랜치 {r.n_tranches} ≠ {spec.n_tranches}")
    return out


def r_sigma_series(ctx: Ctx, spec: BookSpec = DEFAULT_SPEC) -> dict:
    """같은 날 R 목표 비중의 사전 능동위험(스텝 단위) — O 의 λ 맞춤 목표. 진짜 신호 R 의 값이고, 널 O 도 이 값에 맞춘다."""
    recs = simulate(ctx, "R", spec=spec)
    return {r.t: r.exante_20 for r in recs}
