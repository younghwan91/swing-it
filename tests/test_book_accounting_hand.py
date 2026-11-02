"""9단계 회계를 손계산과 대조한다 — 비용 없이 기본 핀에서 돈다. 변이 시험이 여기를 가드로 쓴다. 비용까지 본 대조는 ``test_pead_optimizer_accounting_hand.py``."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_fixtures import hand_ctx  # noqa: E402

from swing_it.book.accounting import check_invariants, simulate  # noqa: E402
from swing_it.book.context import Ctx, StepRec, models_from_mask  # noqa: E402
from swing_it.book.spec import DEFAULT_SPEC, Cell, Flags  # noqa: E402

NO_COSTS = Flags(costs=False)


@pytest.mark.parametrize("field", ["gross", "turnover", "bench", "n_frozen", "n_delist", "n_buy_limit_up", "cash"])
def test_book_matches_the_hand_calculation(field):
    ctx, spec, steps, _ = hand_ctx()
    recs = simulate(ctx, "R", flags=NO_COSTS, spec=spec)
    assert len(recs) == len(steps)
    for r, want in zip(recs, steps):
        assert getattr(r, field) == pytest.approx(want[field], abs=1e-12), f"k={r.k} {field}"
    assert check_invariants(recs, "R", spec=spec, costs=False) == []


def test_zero_volume_at_the_fill_freezes_a_holding():
    """거래량 0 은 가격이 있어도 체결 불가다 — C 는 그 스텝에 비중 그대로 남는다."""
    ctx, spec, _, ix = hand_ctx()
    r = simulate(ctx, "R", flags=NO_COSTS, spec=spec)[4]
    weights = {i: w for i, w, _, _ in r.contrib if w > 0}
    assert weights[ix["C"]] == pytest.approx(0.5, abs=1e-12)


def _all_cash_ctx():
    """t=20 에 목표 C·D 가 체결일 거래량 0 — A·B 는 팔리고 그 스텝은 전액 현금이다. C 는 그 구간에 +20%."""
    nN, nD = 6, 40
    C = np.full((nN, nD), 100.0)
    C[0, 16:] = 110.0
    C[1, 16:] = 90.0
    C[2, 26:] = 120.0
    tv = np.full((nN, nD), 1e11)
    tv[2, 21] = tv[3, 21] = 0.0
    sig = np.zeros((nN, nD))
    grid = [10, 15, 20, 25, 30]
    for t, o in ((10, (0, 1)), (15, (0, 1)), (20, (2, 3)), (25, (2, 3)), (30, (2, 3))):
        sig[list(o), t] = [9.0, 8.0]
    ret = np.full_like(C, np.nan)
    ret[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    return Ctx(dates=pd.bdate_range("2021-01-04", periods=nD), C=C, tv_won=tv, adv=np.full((nN, nD), 1e5),
               cap=np.tile(np.array([6, 5, 4, 3, 2, 1.0])[:, None], (1, nD)) * 1e12, sig=sig, ret=ret, lab=np.full((nN, nD), "G", dtype=object),
               market=np.array(["KOSPI"] * nN, dtype=object), grid=grid, models=models_from_mask(np.ones((nN, nD), bool), grid),
               delist_after=np.full(nN, 10**9, dtype=np.int64))


def test_cash_earns_nothing_and_the_book_re_enters_when_it_can():
    spec = replace(DEFAULT_SPEC, step=5, horizon=5, top_n=2, min_names=2, adv_floor=1.0, cap_rank=(0, 6))
    r = simulate(_all_cash_ctx(), "R", flags=NO_COSTS, spec=spec)
    assert [x.gross for x in r] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)
    assert [x.turnover for x in r] == pytest.approx([1.0, 0.1, 1.0, 1.0, 0.0], abs=1e-12)
    assert [x.cash for x in r] == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0], abs=1e-12)
    assert [x.n_names for x in r] == [2, 2, 0, 2, 2]
    assert r[2].n_untradeable == 2


# --- 불변식 — NaN 은 통과가 아니다 ---

def _rec(**kw):
    base = dict(k=5, t=0, date="2020-01-02", gross=0.0, cost1=0.0, bench=0.0, turnover=0.0, exante_20=0.0, factor_share=0.0, cash=0.0,
                contrib=[(0, 1.0, 0.0, 0.0)], n_names=1, n_tranches=3)
    return StepRec(**{**base, **kw})


@pytest.mark.parametrize("kw", [
    {"cash": float("nan")}, {"min_w": float("nan")}, {"gross": float("nan")}, {"cost1": float("nan")}, {"turnover": float("nan")},
    {"contrib": [(0, float("nan"), 0.0, 0.0)]},
])
def test_nan_does_not_pass_the_invariants(kw):
    """``x > bound`` 는 NaN 에서 거짓이라 NaN 장부가 위반 0 으로 지나갔다."""
    assert check_invariants([_rec()], "R") == []
    assert check_invariants([_rec(**kw)], "R"), kw


def test_nan_in_the_optimized_book_bounds_does_not_pass():
    o = {"status": "optimal"}
    assert check_invariants([_rec(max_w=float("nan"), **o)], "O", Cell())


def test_no_steps_is_a_violation_not_a_pass():
    assert check_invariants([], "R")
    assert check_invariants([], "R", min_steps=0) == []


def test_invariant_cap_comes_from_the_spec():
    spec = replace(DEFAULT_SPEC, max_weight=0.03)
    assert any("max w" in v for v in check_invariants([_rec(status="optimal", max_w=0.05)], "O", Cell(), spec=spec))
    assert check_invariants([_rec(status="optimal", max_w=0.05)], "O", Cell(), max_weight=0.10, spec=spec) == []
