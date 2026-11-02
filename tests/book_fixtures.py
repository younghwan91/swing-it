"""단계 B 회계를 시험하는 합성 패널 — DB·core 레이어 없이 돈다. 난수 순서를 바꾸면 ``golden/pead_b_golden.json`` 과 어긋난다."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from krx_quant_core.backtest.crosssectional import _trailing_adv

from swing_it.book.context import Ctx, models_from_mask
from swing_it.book.spec import DEFAULT_SPEC

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


def hand_ctx():
    """손으로 계산한 5스텝 R 북 — 이름 A~F. 표류·폐지 현금화·정지 동결·거래량 0·상한가 체결이 한 번씩 있다.

    돌려주는 ``steps`` 는 스텝별 기대값이다(비용 전). ``trades`` 는 (이름, Δw, 결정일), ``delist`` 는 (이름, 비중) — 비용까지 볼 때 쓴다.
    """
    names = "ABCDEF"
    ix = {c: i for i, c in enumerate(names)}
    nN, nD = 6, 60
    dates = pd.bdate_range("2021-01-04", periods=nD)
    noise = np.ones(nD)
    for d in range(1, 31):
        noise[d] = noise[d - 1] * (1.01 if d % 2 else 1 / 1.01)
    C = np.array([100, 200, 100, 100, 100, 100.0]).reshape(-1, 1) * noise[None, :]
    C[:, 31:] = C[:, 30:31]
    for c, p in zip(names, (100.0, 200.0, 100.0, 100.0, 100.0, 100.0)):
        C[ix[c], 31:] = p
    C[ix["A"], 36:] = 110.0
    C[ix["A"], 46:] = 121.0
    C[ix["B"], 36:] = 190.0
    C[ix["B"], 44] = 152.0
    C[ix["B"], 45:] = np.nan
    C[ix["C"], 36] = 130.0                      # 체결일 +30% — 상한가 매수
    C[ix["C"], 37:41] = 117.0
    C[ix["C"], 41] = np.nan                     # 거래정지
    C[ix["C"], 42:46] = 117.0
    C[ix["C"], 46:] = 93.6
    C[ix["D"], 56:] = 105.0
    C[ix["E"], 56:] = 98.0
    tv = np.full((nN, nD), 1e11)
    tv[~np.isfinite(C)] = np.nan
    tv[ix["C"], 51] = 0.0                       # 거래량 0
    adv = np.full((nN, nD), 1e5)
    cap = np.tile(np.array([6, 5, 4, 3, 2, 1.0])[:, None], (1, nD)) * 1e12
    sig = np.full((nN, nD), np.nan)
    grid = [30, 35, 40, 45, 50]
    for t, order in zip(grid, ("AB", "AC", "AB", "AC", "DE")):
        sig[:, t] = 0.0
        for r, c in enumerate(order):
            sig[ix[c], t] = 10.0 - r
    sig[ix["B"], 45] = np.nan
    sig[ix["B"], 50] = np.nan
    ret = np.full_like(C, np.nan)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    delist_after = np.full(nN, 10**9, dtype=np.int64)
    delist_after[ix["B"]] = 44
    ctx = Ctx(dates=dates, C=C, tv_won=tv, adv=adv, cap=cap, sig=sig, ret=ret, lab=np.full((nN, nD), "G10", dtype=object),
              market=np.array(["KOSPI", "KOSDAQ"] * 3, dtype=object), grid=grid, models=models_from_mask(np.ones((nN, nD), bool), grid),
              delist_after=delist_after)
    spec = replace(DEFAULT_SPEC, step=5, horizon=5, top_n=2, min_names=2, adv_floor=1.0, cap_rank=(0, 6))

    def bench(el, f, fn):
        return float(np.mean([ctx.Cff[ix[c], fn] / C[ix[c], f] - 1.0 for c in el if np.isfinite(C[ix[c], f])]))

    steps = []
    # k=0 A·B 를 반씩 산다. 다음 체결까지 A +10%, B −5%
    g0 = 0.5 * 0.1 + 0.5 * -0.05
    steps.append({"gross": g0, "turnover": 1.0, "bench": bench(names, 31, 36), "n_frozen": 0, "n_delist": 0, "n_buy_limit_up": 0, "cash": 0.0,
                  "trades": [("A", 0.5, 30), ("B", 0.5, 30)], "delist": []})
    # k=1 표류 뒤 비중에서 A·C 반씩으로. C 는 체결일 +30%
    wa, wb = 0.5 * 1.1 / 1.025, 0.5 * 0.95 / 1.025
    d = {"A": 0.5 - wa, "B": -wb, "C": 0.5}
    steps.append({"gross": 0.5 * 0.0 + 0.5 * -0.1, "turnover": abs(d["A"]) + abs(d["B"]) + d["C"], "bench": bench(names, 36, 41), "n_frozen": 0, "n_delist": 0,
                  "n_buy_limit_up": 1, "cash": 0.0, "trades": [(k, v, 35) for k, v in d.items()], "delist": []})
    # k=2 C 정지(41) → 표류 비중으로 동결, A·B 가 나머지를 나눈다. B 는 폐지 전 마지막 152 까지 −20%
    wa2, wc2 = 0.5 / 0.95, 0.5 * 0.9 / 0.95
    room = 1.0 - wc2
    na = nb = room / 2.0
    g2 = na * 0.1 + nb * -0.2 + wc2 * -0.2
    steps.append({"gross": g2, "turnover": abs(na - wa2) + nb, "bench": bench(names, 41, 46), "n_frozen": 1, "n_delist": 0, "n_buy_limit_up": 0, "cash": 0.0,
                  "trades": [("A", na - wa2, 40), ("B", nb, 40)], "delist": []})
    # k=3 B 폐지 현금화, A·C 반씩
    den = 1.0 + g2
    wa3, wb3, wc3 = na * 1.1 / den, nb * 0.8 / den, wc2 * 0.8 / den
    steps.append({"gross": 0.0, "turnover": abs(0.5 - wa3) + abs(0.5 - wc3), "bench": bench("ACDEF", 46, 51), "n_frozen": 0, "n_delist": 1, "n_buy_limit_up": 0, "cash": 0.0,
                  "trades": [("A", 0.5 - wa3, 45), ("C", 0.5 - wc3, 45)], "delist": [("B", wb3)]})
    # k=4 C 거래량 0(51) → 동결, A 를 팔고 D·E 를 산다
    steps.append({"gross": 0.25 * 0.05 + 0.25 * -0.02, "turnover": 0.5 + 0.25 + 0.25, "bench": bench("ACDEF", 51, 56), "n_frozen": 1, "n_delist": 0, "n_buy_limit_up": 0, "cash": 0.0,
                  "trades": [("A", -0.5, 50), ("D", 0.25, 50), ("E", 0.25, 50)], "delist": []})
    return ctx, spec, steps, ix
