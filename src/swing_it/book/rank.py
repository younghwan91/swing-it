"""7단계(순위 북) — 트랜치를 신호 순위로 짜고, 활성 트랜치를 동일가중한다. 최적화기를 쓰지 않는다."""

from __future__ import annotations

import numpy as np

from swing_it.book.context import Ctx, delisted
from swing_it.book.spec import DEFAULT_SPEC, BookSpec, Flags
from swing_it.features.universe import rank_band_eligible


def tranche_names(t, rank_sig, sig, adv, cap, in_u, top_n=DEFAULT_SPEC.top_n, min_names=DEFAULT_SPEC.min_names, *,
                  adv_floor=DEFAULT_SPEC.adv_floor, cap_rank=DEFAULT_SPEC.cap_rank):
    """트랜치 = eligible(t) 의 신호 상위 ``top_n`` (core ``staggered_tranche_backtest.book`` 과 같은 정렬).

    ``rank_sig`` 는 순위에 쓰는 신호(진짜 또는 순열), 적격은 진짜 신호 ``sig`` 로 정한다(널에서 신호만 섞는다). 순열 값이 NaN 인 적격 종목은 고르지 않는다.
    """
    el = rank_band_eligible(t, sig, adv, cap, in_u, adv_floor=adv_floor, cap_rank=cap_rank)
    if el.size < min_names:
        return None
    idx = np.sort(el)  # core: idx = np.where(ok)[0] (행 순서)
    vals = rank_sig[idx, t]
    keep = np.isfinite(vals)
    idx, vals = idx[keep], vals[keep]
    return idx[np.argsort(-vals)[:top_n]]


def r_target(ctx: Ctx, k: int, t: int, tranches: dict, flags: Flags, f: int, with_parts: bool = False,
             spec: BookSpec = DEFAULT_SPEC):
    """R 목표 비중(체결일 f 기준, 합 1): 활성 트랜치(k, k−1, … ``spec.n_tranches`` 개)마다 1/n, 트랜치 안 동일가중. 강제 매도·폐지 종목은 뺀다.

    ``with_parts`` 면 ``(w, 활성 트랜치 수)`` 를 돌려준다(PO-7 M3 불변식).
    """
    nN = ctx.C.shape[0]
    in_u = ctx.in_u(t)
    gone = delisted(ctx, f)
    w = np.zeros(nN)
    parts = []
    for j in range(k, max(-1, k - spec.n_tranches), -1):
        b = tranches.get(j)
        if b is None:
            continue
        b = np.asarray(b, dtype=int)
        if flags.forced_sale:
            b = b[in_u[b]]
        if flags.delist_cash:
            b = b[~gone[b]]
        b = b[np.isfinite(ctx.Cff[b, f])]
        if b.size:
            parts.append(b)
    for b in parts:
        w[b] += 1.0 / (len(parts) * b.size)
    return (w, len(parts)) if with_parts else w
