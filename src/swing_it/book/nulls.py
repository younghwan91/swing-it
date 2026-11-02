"""9단계 널 — 신호를 종목끼리 섞은 같은 메뉴의 무작위 신호. 합격 바는 러너가 사전등록한다. 4단계 IC 널은 :func:`swing_it.validation.ic.ic_permutation_null`.

**π 는 신호의 베타·변동성·사이즈 상관을 지운다.** 알파 0 이어도 기울기가 있는 신호는 이 널을 그냥 넘는다(기울기 입력에서 95 백분위 기각률 23~28%, 검증 4단계).
이 널만으로 알파를 주장하지 않는다. 팩터 회귀(:func:`swing_it.diagnostics.hac.ols_hac`)와 함께 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swing_it.book.accounting import check_invariants, simulate
from swing_it.book.context import Ctx
from swing_it.book.metrics import ANN_20, ir, oos, step_returns
from swing_it.book.pool import fork_pool
from swing_it.book.spec import DEFAULT_SPEC, BookSpec, Cell, Flags
from swing_it.validation.keys import keyed_rng

N_PERM = 300
N_BOOT, BOOT_MEAN_BLOCK = 10_000, 3


@dataclass(frozen=True)
class _PermJob:
    """널 작업자가 공유하는 입력."""

    ctx: Ctx
    cell: Cell
    ic_by_year: dict
    sigma_r: dict
    window: tuple[str, str]
    spec: BookSpec
    flags: Flags
    ann: float


_JOB: "_PermJob | None" = None


def _set_job(job: "_PermJob | None") -> None:
    """작업자 초기화. 부모에서는 부르지 않는다."""
    global _JOB
    _JOB = job


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


class _RowPermuted:
    """``sig[pi]`` 의 읽기 전용 보기 — ``view[rows, t]`` 만 받는다. 스텝마다 그 칸만 읽으니 작업마다 패널을 통째로 복사하지 않는다."""

    __slots__ = ("sig", "pi")

    def __init__(self, sig: np.ndarray, pi: np.ndarray):
        self.sig, self.pi = sig, pi

    def __getitem__(self, key):
        rows, t = key
        return self.sig[self.pi[rows], t]


def _perm_task(a):
    r, kind = a
    job = _JOB
    ctx, cell, spec, flags = job.ctx, job.cell, job.spec, job.flags
    pi = keyed_rng("perm", r).permutation(ctx.sig.shape[0])
    recs = simulate(ctx, kind, cell, flags, rank_sig=_RowPermuted(ctx.sig, pi), ic_by_year=job.ic_by_year, sigma_r=job.sigma_r, spec=spec)
    nan_share = float(np.nanmean([x.perm_nan_share for x in recs])) if kind == "O" else float("nan")
    viol = check_invariants(recs, kind, cell, spec=spec, costs=flags.costs)
    return r, kind, ir(oos(step_returns(recs, 2.0), job.window)["a"], job.ann), nan_share, viol


def permutation_nulls(ctx, cell, ic_by_year, sigma_r, workers: int, n: int = N_PERM, *, window: tuple[str, str],
                      spec: BookSpec = DEFAULT_SPEC, kinds: tuple[str, ...] = ("R", "O"), flags: Flags = Flags(), ann: float = ANN_20):
    """같은 메뉴의 무작위 신호 널. 같은 π 를 두 북에 쓴다(시드 ``keyed_rng("perm", r)``). 평가 구간 2배 IR 분포를 돌려준다.

    ``ann`` 은 스텝 연환산이다 — 간격이 20일이 아니면 러너의 값을 넘긴다. 평가 구간에 격자일이 없으면 ``ValueError``.
    """
    in_window = [window[0] <= str(ctx.dates[t].date()) <= window[1] for t in ctx.grid]
    if not any(in_window):
        raise ValueError(f"평가 창 {window} 에 격자일이 없다 — 널이 전부 NaN 이 된다")
    job = _PermJob(ctx, cell, ic_by_year, sigma_r, window, spec, flags, ann)
    for t in ctx.grid:      # 포크 전에 채워 작업자들이 같은 값을 공유한다
        ctx.vol(t)
    tasks = [(r, k) for r in range(n) for k in kinds]
    with fork_pool(workers, _set_job, (job,)) as pool:
        res = list(pool.imap_unordered(_perm_task, tasks, chunksize=1))
    out = {"R": np.full(n, np.nan), "O": np.full(n, np.nan), "O_perm_nan_share": np.full(n, np.nan), "violations": []}
    for r, kind, v, s, viol in res:
        out[kind][r] = v
        out["violations"] += [f"perm{r} {x}" for x in viol]
        if kind == "O":
            out["O_perm_nan_share"][r] = s
    return out
