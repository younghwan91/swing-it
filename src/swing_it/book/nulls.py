"""9단계 널 — 신호를 종목끼리 섞은 같은 메뉴의 무작위 신호. 합격 바는 러너가 사전등록한다. 4단계 IC 널은 :func:`swing_it.validation.ic.ic_permutation_null`.

**π 는 신호의 베타·변동성·사이즈 상관을 지운다.** 알파 0 이어도 기울기가 있는 신호는 이 널을 그냥 넘는다(기울기 입력에서 95 백분위 기각률 23~28%, 검증 4단계).
이 널만으로 알파를 주장하지 않는다. 팩터 회귀(:func:`swing_it.diagnostics.hac.ols_hac`)와 함께 쓴다.
"""

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
    aO, aR = np.asarray(aO, dtype=np.float64), np.asarray(aR, dtype=np.float64)
    if aO.size != aR.size:
        raise ValueError(f"두 수익열의 길이가 다르다: {aO.size}·{aR.size} — 같은 스텝끼리 비교해야 한다")
    if aO.size == 0:
        raise ValueError("수익열이 비었다")
    if not (np.isfinite(aO).all() and np.isfinite(aR).all()):
        raise ValueError("수익열에 NaN·inf 가 있다 — ir 이 조용히 빼면 두 열의 표본이 달라진다")
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
    ctx, cell, icy, sr, window, spec, flags, ann = (_G[k] for k in ("ctx", "cell", "ic", "sr", "window", "spec", "flags", "ann"))
    pi = keyed_rng("perm", r).permutation(ctx.sig.shape[0])
    recs = simulate(ctx, kind, cell, flags, rank_sig=ctx.sig[pi], ic_by_year=icy, sigma_r=sr, spec=spec)
    nan_share = float(np.nanmean([x.perm_nan_share for x in recs])) if kind == "O" else float("nan")
    viol = check_invariants(recs, kind, cell, max_weight=spec.max_weight, spec=spec, costs=flags.costs)
    return r, kind, ir(oos(step_returns(recs, 2.0), window)["a"], ann), nan_share, viol


def permutation_nulls(ctx, cell, ic_by_year, sigma_r, workers: int, n: int = N_PERM, *, window: tuple[str, str],
                      spec: BookSpec = DEFAULT_SPEC, kinds: tuple[str, ...] = ("R", "O"), flags: Flags = Flags(), ann: float = ANN_20):
    """같은 메뉴의 무작위 신호 널. 같은 π 를 두 북에 쓴다(시드 ``keyed_rng("perm", r)``). 평가 구간 2배 IR 분포를 돌려준다.

    ``ann`` 은 스텝 연환산이다 — 간격이 20일이 아니면 러너의 값을 넘긴다. 평가 구간에 격자일이 없으면 ``ValueError``.
    """
    in_window = [window[0] <= str(ctx.dates[t].date()) <= window[1] for t in ctx.grid]
    if not any(in_window):
        raise ValueError(f"평가 창 {window} 에 격자일이 없다 — 널이 전부 NaN 이 된다")
    _G.update({"ctx": ctx, "cell": cell, "ic": ic_by_year, "sr": sigma_r, "window": window, "spec": spec, "flags": flags, "ann": ann})
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
