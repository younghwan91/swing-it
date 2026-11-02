"""ML 예측 타깃 — 잔차 복원·창·정지일·폐지(DB 불요, 합성 패널)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

factor_model = pytest.importorskip("krx_quant_core.risk.factor_model")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
import ml_forecast_common as C  # noqa: E402
import ml_forecast_data as D  # noqa: E402
import ml_forecast_target as TG  # noqa: E402

BIG = 10**9


def _simple(N=4, T=60, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 0.01, (N, T))
    vol = np.full((N, T), 1000.0)
    close = 100 * np.exp(np.cumsum(u, axis=1))
    return u, vol, close, np.full(N, BIG, dtype=np.int64)


def test_window_is_exactly_t2_to_t21():
    u, vol, close, dl = _simple()
    t = 5
    y, *_ = TG.label_at(u, vol, close, dl, t)
    assert y[0] == pytest.approx(u[0, t + 2: t + 22].sum())
    u2 = u.copy()
    u2[0, t + 1] = 9.0  # t+1 은 진입일 — 안 들어간다
    u2[0, t + 22] = 9.0  # t+22 도
    assert TG.label_at(u2, vol, close, dl, t)[0][0] == pytest.approx(y[0])


def test_halt_days_count_zero_and_resume_day_stays():
    u, vol, close, dl = _simple()
    t = 5
    close[1, 10:14] = close[1, 9]  # 정지: 종가 고정
    vol[1, 10:14] = 0.0
    u[1, 10:14] = -0.5  # 가짜 평탄 잔차(−f) — 지워져야 한다
    u[1, 14] = 0.2  # 재개일 수익은 남는다
    y, halt, *_ = TG.label_at(u, vol, close, dl, t)
    expected = u[1, t + 2: t + 22].copy()
    expected[10 - (t + 2): 14 - (t + 2)] = 0.0
    assert y[1] == pytest.approx(expected.sum())
    assert halt[1].sum() == 4


def test_entry_volume_zero_is_not_tradable_and_missing_row_too():
    _, vol, _, _ = _simple()
    vol[2, 6] = 0.0
    vol[3, 6] = np.nan
    tr = TG.tradable(vol, 5)
    assert tr.tolist() == [True, True, False, False]


def test_fifteen_day_rule_and_delist_exception():
    u, vol, close, dl = _simple()
    t = 5
    lo = t + 2
    # 0: 가격 14일뿐, 폐지 아님 → NaN
    close[0, lo + 14: lo + 20] = np.nan
    u[0, lo + 14: lo + 20] = np.nan
    vol[0, lo + 14: lo + 20] = np.nan
    # 1: 가격 15일 → 남는다
    close[1, lo + 15: lo + 20] = np.nan
    u[1, lo + 15: lo + 20] = np.nan
    # 2: 창 안 폐지(가격 3일) → 관측 합으로 남는다
    close[2, lo + 3:] = np.nan
    u[2, lo + 3:] = np.nan
    vol[2, lo + 3:] = np.nan
    dl[2] = lo + 2
    # 3: 이미 폐지(창 전에 끝) → NaN
    close[3, t:] = np.nan
    u[3, t:] = np.nan
    vol[3, t:] = np.nan
    dl[3] = t - 1
    y, *_ = TG.label_at(u, vol, close, dl, t)
    assert np.isnan(y[0])
    assert y[1] == pytest.approx(u[1, lo: lo + 15].sum())
    assert y[2] == pytest.approx(u[2, lo: lo + 3].sum())
    assert np.isnan(y[3])


def test_delist_at_entry_day_keeps_zero_label():
    u, vol, close, dl = _simple()
    t = 5
    close[0, t + 2:] = np.nan
    u[0, t + 2:] = np.nan
    dl[0] = t + 1  # t+1 이 마지막 거래일 → 체결 뒤 전 구간 현금
    y, *_ = TG.label_at(u, vol, close, dl, t)
    assert y[0] == 0.0


def test_labels_report_counts_on_universe_rows():
    u, vol, close, dl = _simple(N=6)
    t = 5
    vol[1, t + 3] = 0.0
    vol[2, t + 1] = 0.0  # 진입 불가
    seam = np.zeros(u.shape, dtype=bool)
    seam[3, t + 4] = True
    U = {t: np.array([0, 1, 2, 3])}
    y, rep = TG.labels(u, vol, close, dl, U, seam)
    assert rep["rows"] == 4 and rep["rows_entry_excluded"] == 1 and rep["rows_finite"] == 3
    assert rep["rows_with_halt"] == 1 and rep["halt_days"] == 1 and rep["rows_seam"] == 1
    assert set(y) == {t}


def _factor_panel(seed=3):
    """심은 팩터 모델: r_is = f_s + β_i·g_s + e_is. 업종 2개·스타일 1개."""
    rng = np.random.default_rng(seed)
    N, T = 120, 80
    X = np.tile(rng.normal(0, 1, (N, 1)), (1, T))
    lab = np.empty((N, T), dtype=object)
    lab[: N // 2] = "G1"
    lab[N // 2:] = "G2"
    cap = np.full((N, T), 1e10)
    f = rng.normal(0, 0.01, T)
    g = rng.normal(0, 0.005, T)
    e = rng.normal(0, 0.005, (N, T))
    R = f[None, :] + X * g[None, :] + e
    R[:, 0] = np.nan
    return R, {"style": X}, lab, cap, e


def test_factor_returns_recovers_planted_residual():
    """심은 팩터 모델에서 u 가 특이수익을 복원한다."""
    R, Xs, lab, cap, e = _factor_panel()
    u = factor_model.factor_returns(R, Xs, lab, cap, np.ones(R.shape, bool)).u
    c = np.corrcoef(u[:, 1:].ravel(), e[:, 1:].ravel())[0, 1]
    assert c > 0.97


def test_label_mask_excludes_halt_day_from_factor_regression():
    """mask_y[:, s−1] = 마스크 ∧ 거래량_s > 0 — 정지일(s)이 회귀 행에서 빠지고 u_s 는 NaN, 다음 날은 돌아온다."""
    R, Xs, lab, cap, _ = _factor_panel()
    N, T = R.shape
    vol = np.full((N, T), 100.0)
    s = 40
    vol[0, s] = 0.0
    R[0, s] = 0.0  # 종가 고정
    P = {"msk": np.ones((N, T), bool), "vol": vol}
    m = D.label_mask(P)
    assert not m[0, s - 1] and m[0, s] and m[1, s - 1]
    assert not m[:, -1].any()
    u = factor_model.factor_returns(R, Xs, lab, cap, m).u
    assert np.isnan(u[0, s]) and np.isfinite(u[0, s + 1]) and np.isfinite(u[1, s])
    # 정지 종목이 빠지면 그날 다른 종목의 잔차가 달라진다(가짜 평탄 잔차가 회귀를 흔들지 않는다)
    full = factor_model.factor_returns(R, Xs, lab, cap, np.ones((N, T), bool)).u
    assert np.isfinite(full[0, s]) and full[0, s] != u[0, s]


def test_constants_match_prereg_window():
    assert C.HOLD == 20 and C.LABEL_LAST == 21 and C.MIN_PRICE_DAYS == 15
