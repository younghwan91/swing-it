"""research/signals/limitup_fast — 메타 세대 2 커널(합성 데이터, DB 없음).

- LU 판정: 원 종가 비율 · 정지일 이월 · 연속 상한가 첫날(직전 거래 가능일 기준).
- 청산 규칙 A: 보유 중 LU 종가 → 다음 첫 거래 가능일 시가 전량(매도분 보존 · 종가 청산과 겹치면 우선 · 장중 청산일은 보유 아님 · 팔 날 없음).
- A½: 그 시가에 남은 절반 + 뒤 경로 × ½ · 원래 시가 청산과 같으면 그대로 · 종가 청산과 겹침 · 칸 넘침 표시.
- 실제 v2 경로 위에서 비율 합 = 1, 총수익 항등식, 순수 파이썬 = JIT.
- B 경로: 시가 매수 · 상한가 시가 버림 · 손절(같은 날이면 손절 우선) · 잠긴 하한가 · 정지 · 패널 밖 미완.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.signals import exits_v2_fast as xv
from research.signals import ga_fast as gf
from research.signals import limitup_fast as lf
from tests.test_exits_v2_fast import _port_inputs


def test_lu_close_and_first_day_of_streak():
    raw = np.array([[100.0, 130.0, 169.0, 169.0, 169.0, 219.7, 100.0]])
    TR = np.array([[True, True, True, True, False, True, True]])
    lu = lf.lu_close(raw, TR)
    assert lu.tolist() == [[False, True, True, False, False, True, False]]
    first = lf.lu_first(lu, TR)
    # 2일째는 연속(직전 거래 가능일이 LU), 정지일 너머 5일째는 직전 거래 가능일(3일째)이 LU 아님 → 첫날
    assert first.tolist() == [[False, True, False, False, False, True, False]]
    # 허용폭: 1.293 경계, NaN·0 직전 종가는 거짓
    raw2 = np.array([[100.0, 129.3, np.nan, 0.0, 10.0]])
    lu2 = lf.lu_close(raw2, np.ones_like(raw2, bool))
    assert lu2.tolist() == [[False, True, False, False, False]]


def _one(n_days=14, opens=None):
    Op = np.arange(100.0, 100.0 + n_days)[None, :] if opens is None else np.asarray(opens, float)[None, :]
    TR = np.ones((1, n_days), bool)
    LUC = np.zeros((1, n_days), bool)
    return Op, TR, LUC


def _ex(x, reason, ev=(), fin_frac=None, fin_ret=0.05):
    ea = np.full((1, 2), -1, np.int32)
    ef = np.zeros((1, 2))
    er = np.full((1, 2), np.nan)
    for s, (a, f, r) in enumerate(ev):
        ea[0, s], ef[0, s], er[0, s] = a, f, r
    ff = 1.0 - ef.sum() if fin_frac is None else fin_frac
    return {"x": np.array([x], np.int32), "reason": np.array([reason], np.int8), "ea": ea, "ef": ef, "er": er,
            "ff": np.array([ff]), "fr": np.array([fin_ret])}


@pytest.mark.parametrize("py", [True, False])
def test_rule_a_full_sells_next_open_and_keeps_earlier_partial(py):
    Op, TR, LUC = _one()
    LUC[0, 6] = True
    ex = _ex(10, gf.R_TRAIL, ev=[(4, 0.5, 0.2)])
    r = lf.lu_exit(np.array([0]), np.array([0]), ex, Op, TR, LUC, half=False, py=py)
    assert r["x"][0] == 7 and r["reason"][0] == lf.R_LU and r["fired"][0] == 1
    assert r["ea"][0].tolist() == [4, -1] and r["ef"][0, 0] == 0.5
    assert r["ff"][0] == pytest.approx(0.5) and r["fr"][0] == pytest.approx(Op[0, 7] / Op[0, 1] - 1)
    # 입력은 그대로
    assert ex["x"][0] == 10
    # 정지일이 끼면 첫 거래 가능일
    TR[0, 7] = False
    r = lf.lu_exit(np.array([0]), np.array([0]), ex, Op, TR, LUC, half=False, py=py)
    assert r["x"][0] == 8


@pytest.mark.parametrize("py", [True, False])
def test_rule_a_close_exits_and_intraday_exits(py):
    Op, TR, LUC = _one()
    LUC[0, 9] = True
    pc, pd_ = np.array([0]), np.array([0])
    # 장중 청산(손절)일의 LU 종가는 보유 아님 → 그대로
    r = lf.lu_exit(pc, pd_, _ex(9, gf.R_STOP), Op, TR, LUC, half=False, py=py)
    assert r["x"][0] == 9 and r["fired"][0] == 0
    # 최대 보유 종가 청산과 겹치면 A 가 우선 → 다음날 시가
    r = lf.lu_exit(pc, pd_, _ex(9, gf.R_TIME), Op, TR, LUC, half=False, py=py)
    assert r["x"][0] == 10 and r["reason"][0] == lf.R_LU
    # 데이터 끝(뒤에 거래 가능일 없음) → 그대로
    TR[0, 10:] = False
    r = lf.lu_exit(pc, pd_, _ex(9, gf.R_DATA_END), Op, TR, LUC, half=False, py=py)
    assert r["x"][0] == 9 and r["fired"][0] == 0
    # 진입 전 LU 는 무시(신호일 0 → 진입 1)
    Op, TR, LUC = _one()
    LUC[0, 0] = True
    r = lf.lu_exit(pc, pd_, _ex(9, gf.R_TRAIL), Op, TR, LUC, half=False, py=py)
    assert r["fired"][0] == 0


@pytest.mark.parametrize("py", [True, False])
def test_rule_a_half(py):
    Op, TR, LUC = _one()
    pc, pd_ = np.array([0]), np.array([0])
    LUC[0, 3] = True
    ent = Op[0, 1]
    # LU 3 → 4 시가 절반, 4 장중 익절(원래 0.5)은 0.25 로, 최종 0.25
    r = lf.lu_exit(pc, pd_, _ex(10, gf.R_TRAIL, ev=[(4, 0.5, 0.2)], fin_ret=0.07), Op, TR, LUC, half=True, py=py)
    assert r["fired"][0] == 1 and r["x"][0] == 10 and r["reason"][0] == gf.R_TRAIL
    assert r["ea"][0].tolist() == [4, 4]
    np.testing.assert_allclose(r["ef"][0], [0.5, 0.25])
    np.testing.assert_allclose(r["er"][0], [Op[0, 4] / ent - 1, 0.2])
    assert r["ff"][0] == pytest.approx(0.25) and r["fr"][0] == pytest.approx(0.07)
    assert r["ef"][0].sum() + r["ff"][0] == pytest.approx(1.0)
    # 원래 최종 청산이 그 시가(추적 시가 청산) → 값이 같아 그대로(표시 2)
    pk = Op[0, 4] / ent - 1
    r = lf.lu_exit(pc, pd_, _ex(4, gf.R_TRAIL, fin_ret=pk), Op, TR, LUC, half=True, py=py)
    assert r["fired"][0] == 2 and r["x"][0] == 4 and r["ef"][0].sum() == 0
    # 종가 청산과 겹침: 절반은 그 종가, 절반은 다음날 시가
    r = lf.lu_exit(pc, pd_, _ex(3, gf.R_TIME, ev=[(2, 0.5, 0.2)], fin_ret=0.3), Op, TR, LUC, half=True, py=py)
    assert r["x"][0] == 4 and r["reason"][0] == lf.R_LU
    assert r["ea"][0].tolist() == [2, 3]
    np.testing.assert_allclose(r["ef"][0], [0.5, 0.25])
    assert r["er"][0, 1] == pytest.approx(0.3) and r["ff"][0] == pytest.approx(0.25)
    # 칸 넘침: 앞 매도분 둘 + A½ → 표시 −1, 그대로
    r = lf.lu_exit(pc, pd_, _ex(10, gf.R_TRAIL, ev=[(2, 0.25, 0.1), (3, 0.25, 0.1)]), Op, TR, LUC, half=True, py=py)
    assert r["fired"][0] == -1 and r["x"][0] == 10 and r["ea"][0].tolist() == [2, 3]


def test_rule_a_on_real_v2_paths_identity_and_jit_parity():
    b, lp, ev, pc, pd_ = _port_inputs(seed=7)
    clow, piv = ev["contraction_low"].to_numpy(float), ev["pivot"].to_numpy(float)
    st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = xv.sim_book_v2(
        b.O, b.H, b.L, b.C, b.MA, np.full(b.C.shape, np.nan), b.tradable, lp, pc, pd_, clow, {}, pivot=piv)
    base = {"x": x, "reason": rs, "ea": ea, "ef": ef, "er": er, "ff": ff, "fr": fr}
    raw = b.C.copy()
    LUC = lf.lu_close(raw, b.tradable, ratio=1.03)            # 합성 시세엔 상한가가 드물어 문턱을 낮춰 자주 걸리게
    LUC = np.ascontiguousarray(LUC)
    for half in (False, True):
        a = lf.lu_exit(pc, pd_, base, b.O, b.tradable, LUC, half=half, py=True)
        j = lf.lu_exit(pc, pd_, base, b.O, b.tradable, LUC, half=half)
        for k in a:
            np.testing.assert_array_equal(np.nan_to_num(a[k], nan=-9), np.nan_to_num(j[k], nan=-9))
        ok = a["x"] >= 0
        assert (a["fired"][ok] == 1).sum() > 20
        tot = np.nansum(a["ef"] * np.nan_to_num(a["er"]), axis=1) + a["ff"] * a["fr"]
        frac = a["ef"].sum(axis=1) + a["ff"]
        np.testing.assert_allclose(frac[ok], 1.0)
        assert np.isfinite(tot[ok]).all()
        m = ok & (a["fired"] == 1)
        if not half:
            ent = b.O[pc[m], pd_[m] + 1]
            np.testing.assert_allclose(a["fr"][m], b.O[pc[m], a["x"][m]] / ent - 1)
            assert (a["reason"][m] == lf.R_LU).all()
            # 청산이 늦어지는 것은 종가 청산과 겹친 경우뿐(하루 뒤 첫 거래 가능일)
            late = a["x"][m] > x[m]
            assert np.isin(rs[m][late], lf.CLOSE_EXITS).all()
        # 매도분은 시간 순
        both = (a["ea"][:, 0] >= 0) & (a["ea"][:, 1] >= 0)
        assert (a["ea"][both, 0] <= a["ea"][both, 1]).all()


def _bworld():
    n = 14
    O = np.full((2, n), 100.0)  # noqa: E741
    H = np.full((2, n), 102.0)
    L = np.full((2, n), 98.0)
    C = np.full((2, n), 101.0)
    C[:, 2] = 90.0                                    # 신호일 종가(진입 시가 100 < 90 × 1.293)
    TR = np.ones((2, n), bool)
    LD = np.zeros((2, n), bool)
    O[1, 3] = 120.0                                   # 종목 1: 상한가 시가 → 버림
    return O, H, L, C, TR, LD


@pytest.mark.parametrize("py", [True, False])
def test_b_paths_rules(py):
    O, H, L, C, TR, LD = _bworld()  # noqa: E741
    pc, pd_ = np.array([0, 1]), np.array([2, 2])
    hs = np.array([1, 3, 5, 10, 20])
    fill, ret, xab, why, luo = lf.b_paths(pc, pd_, O, H, L, C, TR, LD, hs, 0.10, py=py)
    assert fill.tolist() == [True, False] and luo.tolist() == [False, True]
    np.testing.assert_allclose(ret[0, :4], 0.01)                          # 종가 101 / 시가 100
    assert xab[0, :4].tolist() == [3, 5, 7, 12] and (why[0, :4] == lf.B_TIME).all()
    assert np.isnan(ret[0, 4]) and np.isnan(ret[1]).all()                  # 패널 밖 = 미완, 미체결 = NaN
    # 손절: 5행 장중 저가 89 → 90, h=3(만기 5행)도 같은 날 손절이 우선
    L[0, 5] = 89.0
    fill, ret, xab, why, luo = lf.b_paths(pc, pd_, O, H, L, C, TR, LD, hs[:4], 0.10, py=py)
    assert ret[0, 0] == pytest.approx(0.01) and why[0, 0] == lf.B_TIME
    np.testing.assert_allclose(ret[0, 1:], -0.10)
    assert (why[0, 1:] == lf.B_STOP).all() and (xab[0, 1:] == 5).all()
    # 잠긴 하한가(5행)는 못 판다 → 6행 시가 85 ≤ 손절가 → 85
    LD[0, 5] = True
    O[0, 6] = 85.0
    fill, ret, xab, why, luo = lf.b_paths(pc, pd_, O, H, L, C, TR, LD, hs[:4], 0.10, py=py)
    assert ret[0, 1] == pytest.approx(-0.15) and xab[0, 1] == 6
    # 만기 행이 정지면 다음 거래 가능일 종가
    O, H, L, C, TR, LD = _bworld()  # noqa: E741
    TR[0, 7] = False
    C[0, 8] = 110.0
    fill, ret, xab, why, luo = lf.b_paths(pc, pd_, O, H, L, C, TR, LD, np.array([5]), 0.10, py=py)
    assert xab[0, 0] == 8 and ret[0, 0] == pytest.approx(0.10)
    # 진입 뒤 시세가 끊기면(폐지) 마지막 거래 가능일 종가
    O, H, L, C, TR, LD = _bworld()  # noqa: E741
    TR[0, 5:] = False
    fill, ret, xab, why, luo = lf.b_paths(pc, pd_, O, H, L, C, TR, LD, np.array([5]), 0.10, py=py)
    assert xab[0, 0] == 4 and why[0, 0] == lf.B_END
    # 진입일 정지 → 버림
    O, H, L, C, TR, LD = _bworld()  # noqa: E741
    TR[0, 3] = False
    fill, *_ = lf.b_paths(pc, pd_, O, H, L, C, TR, LD, np.array([5]), 0.10, py=py)
    assert not fill[0]
