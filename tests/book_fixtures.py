"""단계 B 회계를 시험하는 합성 패널 — DB·core 레이어 없이 돈다. 난수 순서를 바꾸면 ``golden/pead_b_golden.json`` 과 어긋난다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _trailing_adv

from swing_it.book.context import Ctx

SECTORS = ("G10", "G20", "G30")
NAMES = ["market", *SECTORS, "size", "momentum"]
STEP = 20


def synthetic_ctx(seed=0, nN=160, nD=330, gaps=True, delist=True, u_share=0.92):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=nD)
    r = rng.normal(0.0003, 0.02, (nN, nD))
    C = 1000.0 * np.exp(np.cumsum(r, axis=1))
    tv = rng.uniform(1.5e10, 2e11, (nN, nD))  # 원
    tv[: nN // 8] = rng.uniform(1e9, 1.5e10, (nN // 8, nD))  # ADV 미달 종목
    delist_after = np.full(nN, 10**9, dtype=np.int64)
    if gaps:
        for i in rng.choice(nN, 12, replace=False):  # 거래정지(가격 없음)
            s = rng.integers(130, nD - 30)
            C[i, s: s + rng.integers(1, 15)] = np.nan
    if delist:
        for i in rng.choice(nN, 6, replace=False):  # 폐지
            s = int(rng.integers(160, nD - 30))
            C[i, s + 1:] = np.nan
            delist_after[i] = s
    tv[~np.isfinite(C)] = np.nan
    cap = np.exp(rng.normal(27, 1.0, (nN, 1))) * np.exp(np.cumsum(rng.normal(0, 0.01, (nN, nD)), axis=1))
    sig = np.full((nN, nD), np.nan)
    for i in range(nN):  # 분기마다 바뀌는 YoY, 일부 결측
        cuts = np.sort(rng.choice(np.arange(1, nD), 5, replace=False))
        start = 0
        for c in list(cuts) + [nD]:
            if rng.random() > 0.1:
                sig[i, start:c] = rng.normal(0.1, 0.5)
            start = c
    ret = np.full_like(C, np.nan)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    lab = np.array([SECTORS[i % 3] for i in range(nN)], dtype=object)[:, None].repeat(nD, axis=1)
    adv = _trailing_adv(tv / 1e6, nD, 20, C)
    t0 = int(np.searchsorted(dates, np.datetime64("2019-06-03")))
    grid = list(range(t0, nD, STEP))
    models = {}
    for t in grid:
        idx = np.flatnonzero((rng.random(nN) < u_share) & np.isfinite(C[:, t]))
        X = np.zeros((idx.size, len(NAMES)))
        X[:, 0] = 1.0
        for g, s in enumerate(SECTORS):
            X[:, 1 + g] = lab[idx, t] == s
        X[:, 4:] = rng.normal(0, 1, (idx.size, 2))
        A = rng.normal(0, 0.004, (len(NAMES), len(NAMES)))
        F = A @ A.T + np.diag(np.full(len(NAMES), 1e-5))
        D = rng.uniform(1e-4, 9e-4, idx.size)
        models[t] = (idx, X, F, D, NAMES)
    market = np.array(["KOSPI" if i % 2 else "KOSDAQ" for i in range(nN)], dtype=object)
    return Ctx(dates=dates, C=C, tv_won=tv, adv=adv, cap=cap, sig=sig, ret=ret, lab=lab, market=market, grid=grid,
                 models=models, delist_after=delist_after)
