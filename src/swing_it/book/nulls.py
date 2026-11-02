"""4·9단계 널 — 신호를 종목끼리 섞은 같은 메뉴의 무작위 신호. 합격 바는 러너가 사전등록한다."""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from swing_it.book.accounting import check_invariants, simulate
from swing_it.book.metrics import ANN_20, ir, oos, step_returns
from swing_it.book.spec import DEFAULT_SPEC, BookSpec, Flags
from swing_it.validation.keys import keyed_rng

N_PERM = 300
N_BOOT, BOOT_MEAN_BLOCK = 10_000, 3

_G: dict = {}


def stationary_boot_dir(aO, aR, n=N_BOOT, mean_block=BOOT_MEAN_BLOCK, seed_key="boot-dIR", ann: float = ANN_20):
    """ΔIR 의 정상(stationary) 블록 부트스트랩 분포(보고만)."""
    rng = keyed_rng(seed_key)
    T = len(aO)
    p = 1.0 / mean_block
    out = np.empty(n)
    for b in range(n):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for s in range(T):
            if s and rng.random() < p:
                i = rng.integers(T)
            idx[s] = i
            i = (i + 1) % T
        out[b] = ir(aO[idx], ann) - ir(aR[idx], ann)
    return out


def _perm_task(a):
    r, kind = a
    ctx, cell, icy, sr, window, spec, flags = (_G[k] for k in ("ctx", "cell", "ic", "sr", "window", "spec", "flags"))
    pi = keyed_rng("perm", r).permutation(ctx.sig.shape[0])
    recs = simulate(ctx, kind, cell, flags, rank_sig=ctx.sig[pi], ic_by_year=icy, sigma_r=sr, spec=spec)
    nan_share = float(np.nanmean([x.perm_nan_share for x in recs])) if kind == "O" else float("nan")
    viol = check_invariants(recs, kind, cell, spec=spec)
    return r, kind, ir(oos(step_returns(recs, 2.0), window)["a"]), nan_share, viol


def permutation_nulls(ctx, cell, ic_by_year, sigma_r, workers: int, n: int = N_PERM, *, window: tuple[str, str],
                      spec: BookSpec = DEFAULT_SPEC, kinds: tuple[str, ...] = ("R", "O"), flags: Flags = Flags()):
    """같은 메뉴의 무작위 신호 널. 같은 π 를 두 북에 쓴다(시드 ``keyed_rng("perm", r)``). 평가 구간 2배 IR 분포를 돌려준다."""
    _G.update({"ctx": ctx, "cell": cell, "ic": ic_by_year, "sr": sigma_r, "window": window, "spec": spec, "flags": flags})
    tasks = [(r, k) for r in range(n) for k in kinds]
    with mp.get_context("fork").Pool(workers) as pool:
        res = list(pool.imap_unordered(_perm_task, tasks, chunksize=1))
    _G.clear()
    out = {"R": np.full(n, np.nan), "O": np.full(n, np.nan), "O_perm_nan_share": np.full(n, np.nan), "violations": []}
    for r, kind, v, s, viol in res:
        out[kind][r] = v
        out["violations"] += [f"perm{r} {x}" for x in viol]
        if kind == "O":
            out["O_perm_nan_share"][r] = s
    return out
