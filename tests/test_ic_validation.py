"""4단계 — 순위 IC 와 함께 보고할 평균 정보량·분위 초과·요약·IC 널. 기본 핀에서 돈다."""

from __future__ import annotations

import numpy as np
import pytest

from swing_it.diagnostics.hac import nw_t_rows
from swing_it.validation.ic import (
    daily_mean_ic,
    daily_rank_ic,
    fold_ic,
    ic_permutation_null,
    ic_summary,
    quintile_excess,
)

N, D = 50, 60
BAND = {"adv_floor": 1.0, "cap_rank": (0, N)}


def _panel(returns):
    """일 0·1 은 100, 일 2 에 종목별 수익 하나. t=0, h=1 의 라벨이 곧 ``returns``."""
    close = np.full((len(returns), 4), 100.0)
    close[:, 2:] = 100.0 * (1.0 + np.asarray(returns))[:, None]
    n = len(returns)
    return close, np.full((n, 4), 10.0), np.arange(n, 0, -1, dtype=float)[:, None].repeat(4, axis=1)


def test_one_outlier_lifts_mean_ic_but_not_rank_ic_or_its_winsorized_twin():
    """롱온리가 거두는 것은 평균이다. 꼬리 한 종목이 순위 IC 와 평균 IC 를 어떻게 갈라놓는지 손으로 안다."""
    n = 50
    sig = np.arange(n, dtype=float)[:, None].repeat(4, axis=1)
    rets = np.where(np.arange(n) % 2 == 0, 0.01, -0.01)
    rets[n - 1] = 5.0
    close, adv, cap = _panel(rets)
    _, rank = daily_rank_ic(close, adv, cap, sig, 0, 0, 1, adv_floor=1.0, cap_rank=(0, n))
    got = daily_mean_ic(close, adv, cap, sig, 0, 0, 1, adv_floor=1.0, cap_rank=(0, n))
    ranks = np.arange(1.0, n + 1)
    assert got["ic"][0] == pytest.approx(np.corrcoef(ranks, rets)[0, 1])
    assert got["ic"][0] > 0.2 and abs(rank[0]) < 0.15 and got["ic_winsor"][0] < got["ic"][0]
    assert list(got["days"]) == [0]


def test_quintile_excess_by_hand():
    n = 50
    sig = np.arange(n, dtype=float)[:, None].repeat(4, axis=1)
    rets = np.linspace(-0.1, 0.1, n)
    rets[0] = -0.1
    close, adv, cap = _panel(rets)
    q = quintile_excess(close, adv, cap, sig, 0, 0, 1, adv_floor=1.0, cap_rank=(0, n))
    assert q["q5_mean"][0] == pytest.approx(rets[40:].mean() - rets.mean())
    assert q["q1_mean"][0] == pytest.approx(rets[:10].mean() - rets.mean())
    assert q["q5_median"][0] == pytest.approx(np.median(rets[40:]) - np.median(rets))
    assert q["q1_median"][0] == pytest.approx(np.median(rets[:10]) - np.median(rets))
    rets2 = rets.copy()
    rets2[-1] = 5.0
    close2, adv2, cap2 = _panel(rets2)
    q2 = quintile_excess(close2, adv2, cap2, sig, 0, 0, 1, adv_floor=1.0, cap_rank=(0, n))
    assert q2["q5_winsor"][0] < q2["q5_mean"][0], "꼬리 한 종목은 윈저 값에서 줄어야 한다"


def test_ic_summary_uses_lag_h_and_drops_overlapping_days():
    rng = np.random.default_rng(0)
    days = np.arange(100)
    ics = rng.normal(0.02, 0.1, 100)
    s = ic_summary(days, ics, 20)
    assert s["n_days"] == 100 and s["mean"] == pytest.approx(ics.mean())
    assert s["nw_t"] == pytest.approx(nw_t_rows(ics[None, :], lag=20)[0])
    kept = ics[::20]
    assert s["n_nonoverlap"] == 5
    assert s["icir_nonoverlap"] == pytest.approx(kept.mean() / kept.std(ddof=1))


def test_ic_summary_on_too_few_days_is_nan_not_a_number():
    s = ic_summary(np.array([3]), np.array([0.1]), 20)
    assert s["n_days"] == 1 and np.isnan(s["icir_nonoverlap"]) and np.isnan(s["nw_t"])
    empty = ic_summary(np.array([], dtype=int), np.array([]), 20)
    assert empty["n_days"] == 0 and np.isnan(empty["mean"])


@pytest.mark.parametrize("call", [
    lambda: ic_summary(np.arange(3), np.arange(4.0), 5),
    lambda: ic_summary(np.arange(3), np.arange(3.0), 0),
    lambda: fold_ic(np.arange(3), np.arange(3.0), 2, 0),
    lambda: daily_rank_ic(*_panel(np.zeros(30)), np.ones((30, 4)), -1, 1, 1),
    lambda: daily_rank_ic(*_panel(np.zeros(30)), np.ones((30, 4)), 0, 1, 0),
])
def test_bad_window_arguments_are_refused(call):
    with pytest.raises(ValueError):
        call()


def test_fold_ic_detail_shows_what_clipping_hid():
    days, ics = np.arange(10), np.full(10, 0.5)
    assert fold_ic(days, ics, 8, 3) == pytest.approx(0.10)
    d = fold_ic(days, ics, 8, 3, detail=True)
    assert d == {"ic": pytest.approx(0.10), "raw": pytest.approx(0.5), "n": 4}
    none = fold_ic(days, ics, 2, 3, detail=True)
    assert none["ic"] == 0.0 and none["n"] == 0 and np.isnan(none["raw"])


def _noise_panel(seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (N, D)), axis=1))
    sig = rng.normal(size=(N, D))
    return close, np.full((N, D), 10.0), np.arange(N, 0, -1, dtype=float)[:, None].repeat(D, axis=1), sig


def test_ic_null_is_centred_on_zero_and_a_real_signal_sits_above_it():
    close, adv, cap, _ = _noise_panel()
    ahead = close[:, 7:] / close[:, 1:-6] - 1.0
    sig = np.full((N, D), np.nan)
    sig[:, :ahead.shape[1]] = ahead
    null = ic_permutation_null(close, adv, cap, sig, 0, 40, 5, n=30, **BAND)
    assert null["rank"].shape == (30,) and null["mean"].shape == (30,)
    assert abs(null["rank"].mean()) < 0.05
    _, real = daily_rank_ic(close, adv, cap, sig, 0, 40, 5, **BAND)
    assert real.mean() > np.quantile(null["rank"], 0.95)


def test_ic_null_does_not_depend_on_call_order():
    close, adv, cap, sig = _noise_panel(1)
    a = ic_permutation_null(close, adv, cap, sig, 0, 20, 5, n=5, **BAND)
    b = ic_permutation_null(close, adv, cap, sig, 0, 20, 5, n=5, **BAND)
    np.testing.assert_array_equal(a["rank"], b["rank"])
    assert np.isfinite(a["rank"]).all()
