"""패턴 GA-FWM(미너비니 정합) — 손절폭 상한 5%, EPS 전년 같은 분기 대비 상승 필수, 장중 청산, 변형 메뉴."""
import os
import subprocess
import sys

import numpy as np
import pytest

from research.signals import pattern_events as PE
from research.signals import pattern_fast as PF


def _gv(**kw):
    gv = np.full(PF.N_G, np.nan)
    gv[PF.G_STAGE] = 0
    gv[PF.G_TOUCH] = 2
    gv[PF.G_RUNUP_COL] = 0
    gv[PF.G_H0IN] = np.nan
    gv[PF.G_EACC] = 0.0
    gv[PF.G_GAP] = 0.05
    gv[PF.G_ANCHOR] = 0.0
    gv[PF.G_CAP] = 0.25
    gv[PF.G_FB_TOL] = 0.0
    gv[PF.G_SLIP] = 0.0
    gv[PF.G_DFLOOR] = 0.02
    gv[PF.G_BASE] = PF.BASE_BOTTOM
    gv[PF.G_CEILKIND] = 0.0
    gv[PF.G_STOP_BUF] = 0.0
    gv[PF.G_STOP_CLOSE] = 0.0
    gv[PF.G_EQ] = 0.0
    for k, v in kw.items():
        gv[getattr(PF, k)] = v
    return gv


def _event(stop_raw=88.0, eq=10.0, eqp=8.0):
    F = np.full((1, len(PE.F_COLS)), np.nan)
    fi = PE.FI
    F[0, fi["level"]], F[0, fi["order"]], F[0, fi["open_d"]], F[0, fi["close_s"]] = 99.0, 100.0, 99.0, 98.0
    F[0, fi["close_d"]], F[0, fi["low_d"]], F[0, fi["stopE_raw"]] = 101.0, 98.5, stop_raw
    F[0, fi["e_q"]], F[0, fi["e_q_prev"]] = eq, eqp
    return F


def _filter(F, gv):
    o = [np.zeros(1, np.int64)] + [np.zeros(1) for _ in range(3)] + [np.zeros(1, np.bool_), np.zeros(1, np.bool_), np.zeros(1), np.zeros(1)]
    n = PF._filter_py(F, 0, 1, gv, 1.3, *o)
    return n, o[1][0], o[2][0]


def test_stop_cap_lifts_stop_to_5pct():
    n, e, sp = _filter(_event(stop_raw=88.0), _gv(G_SMAX=0.05))
    assert n == 1 and e == pytest.approx(100.0) and sp == pytest.approx(95.0)      # 12% → 5% (거래는 산다)
    n, e, sp = _filter(_event(stop_raw=97.0), _gv(G_SMAX=0.05))
    assert n == 1 and sp == pytest.approx(97.0)                                    # 더 가까운 구조 손절은 그대로
    n, e, sp = _filter(_event(stop_raw=88.0), _gv())
    assert n == 1 and sp == pytest.approx(88.0)                                    # 끔 = 원래대로


def test_market_regime_gene():
    F = _event()
    fi = PE.FI
    F[0, fi["mkt_nhnl"]], F[0, fi["mkt_bos"]], F[0, fi["mkt_br200"]] = 0.6, 0.55, 0.45
    assert _filter(F, _gv(G_MKT=0))[0] == 1
    assert _filter(F, _gv(G_MKT=1))[0] == 1                                       # 신고가 비율 0.6 > 0.5
    assert _filter(F, _gv(G_MKT=2))[0] == 0                                       # 돌파 성공률 0.55 < 0.6
    assert _filter(F, _gv(G_MKT=3))[0] == 0                                       # MA200 위 종목 45% < 절반
    F[0, fi["mkt_nhnl"]] = np.nan
    assert _filter(F, _gv(G_MKT=1))[0] == 0                                       # 모르면 통과 아님


def test_eps_qoq_required():
    assert _filter(_event(eq=10.0, eqp=8.0), _gv(G_EQ=1.0))[0] == 1
    assert _filter(_event(eq=8.0, eqp=10.0), _gv(G_EQ=1.0))[0] == 0                # 전년 같은 분기보다 감소
    assert _filter(_event(eq=-1.0, eqp=-5.0), _gv(G_EQ=1.0))[0] == 0               # 적자 축소는 상승 아님(흑자 필요)
    assert _filter(_event(eq=10.0, eqp=np.nan), _gv(G_EQ=1.0))[0] == 0             # 전년 같은 분기 모름
    assert _filter(_event(eq=8.0, eqp=10.0), _gv())[0] == 1                        # 끔 = 거르지 않음


def _book(rows, ma50=None):
    a = np.array(rows, float).T
    Op, H, L, C = (np.ascontiguousarray(a[i][None, :]) for i in range(4))
    n = Op.shape[1]
    ma = np.full((1, n), np.nan) if ma50 is None else np.array([ma50], float)
    return Op, H, L, C, ma, np.full((1, n), 2.0), np.ones((1, n), bool), np.array([n], np.int64)


def _px(**kw):
    px = PF.LIVE_X.copy()
    px[PF.X_FROM_ENTRY] = 1.0
    px[PF.X_TAKE_R] = np.inf
    px[PF.X_FB_DAYS] = 0
    for k, v in kw.items():
        px[getattr(PF, k)] = v
    return px


def _run(book, px, e=100.0, sp=95.0):
    Op, H, L, C, ma, atr, TR, LIM = book
    out = np.empty(PF.N_O)
    PF._path_py(Op, H, L, C, ma, ma, atr, TR, LIM, 0, 0, e, sp, -np.inf, np.inf, np.inf, 0.0, 1.0, 0, 0.0, 0.0, True, 0, 0, px, out)
    return out


def test_intraday_trail_sells_at_prior_ma():
    rows = [(100, 104, 99, 103), (103, 108, 102, 107), (107, 109, 101, 104), (104, 105, 103, 104)] + [(104, 105, 103, 104)] * 4
    ma50 = [100, 102, 102.5, 103, 103, 103, 103, 103]
    ks = 1.0 - PF.LIVE_X[PF.X_SELL_SLIP]
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0))
    assert int(out[PF.O_REASON]) == PF.R_TRAIL and int(out[PF.O_X]) == 2              # 행 2 저가 101 ≤ 전날 MA50 102
    assert out[PF.O_FIN] == pytest.approx(102 * ks) and out[PF.O_EXIT_OPEN] == 0
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=0.0))
    assert int(out[PF.O_X]) != 2                                                      # 일봉 모드: 종가가 MA50 위라 그날은 안 판다


def test_intraday_fail_exit_prev_low():
    rows = [(100, 102, 99, 101), (101, 103, 100, 102), (102, 102.5, 98, 101), (101, 102, 100, 101)] + [(101, 102, 100, 101)] * 4
    ks = 1.0 - PF.LIVE_X[PF.X_SELL_SLIP]
    out = _run(_book(rows), _px(X_FAIL_EXIT=1, X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0))
    assert int(out[PF.O_REASON]) == PF.R_FAILX and int(out[PF.O_X]) == 2               # 저가 98 < 전일 저가 100 → 100 에 장중 매도
    assert out[PF.O_FIN] == pytest.approx(100 * ks)
    out = _run(_book(rows), _px(X_FAIL_EXIT=1, X_TRAIL=PF.TRAIL_MA50, X_INTRA=0.0))
    assert int(out[PF.O_REASON]) != PF.R_FAILX or int(out[PF.O_X]) != 2              # 일봉 모드: 종가 101 ≥ 100 → 그날은 안 판다


def test_intraday_gap_below_line_sells_at_open():
    rows = [(100, 104, 99, 103), (103, 108, 102, 107), (99, 100, 97, 98)] + [(98, 99, 97, 98)] * 4
    ma50 = [100, 102, 102.5, 103, 103, 103, 103]
    ks = 1.0 - PF.LIVE_X[PF.X_SELL_SLIP]
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0))
    assert int(out[PF.O_X]) == 2 and out[PF.O_EXIT_OPEN] == 1 and out[PF.O_FIN] == pytest.approx(99 * ks)


def test_fwm_variant_menu():
    code = r"""
import sys, math
sys.path[:0] = ['.', 'src']
import numpy as np
import research.experiments.pattern_ga_run as G
from research.signals import pattern_fast as PF
assert G.FW and G.FWL and G.FWM and G.SUB == 'pattern_ga_fwm' and G.CACHE_VERSION == 'pga-v5-fwm'
assert G.FLOOR_PER_YEAR == 6.0 and G.FEAS_MIN == 300 and G.RESTART_SCREEN == 200_000 and G.HYGIENE_FWM == {'gap_min': 1, 'span_min': 5}
gs = [G.decode(x) for x in G.random_x(np.random.default_rng(0), 3000)]
assert all(g['stage'] in (3, 4) and 70.0 <= g['rs_min'] <= 90.0 for g in gs)
assert all(g.get('stop_basis') in (None, 'intraday') and g.get('stop_anchor') != 'dynamic' for g in gs)
assert all(g['pyramid'] != 'off' for g in gs)
assert {g['mkt'] for g in gs} == {'off', 'nhnl', 'bos', 'br200'}
assert any(g['e_min'] is None for g in gs) and any(g['e_min'] == 0.5 for g in gs)
assert min(math.exp(g['span_c']) for g in gs if g['span_c'] is not None) < 7.0
g = gs[0]
gv, px = G.gene_vec(g), G.exit_vec(g)
assert gv[PF.G_MKT] == ('off', 'nhnl', 'bos', 'br200').index(g['mkt']) and gv[PF.G_EQ] == 1.0 and gv[PF.G_SMAX] == 0.05 and px[PF.X_SMAX] == 0.05 and px[PF.X_INTRA] == 1.0
print('ok')
"""
    env = {**os.environ, "PATTERN_GA_VARIANT": "fwm"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_other_variants_keep_daily_exits():
    code = r"""
import sys, math
sys.path[:0] = ['.', 'src']
import numpy as np
import research.experiments.pattern_ga_run as G
from research.signals import pattern_fast as PF
g = G.decode(G.random_x(np.random.default_rng(1), 1)[0])
gv, px = G.gene_vec(g), G.exit_vec(g)
assert gv[PF.G_MKT] == 0 and 'mkt' not in G.GENE_NAMES and gv[PF.G_EQ] == 0.0 and math.isnan(gv[PF.G_SMAX]) and math.isnan(px[PF.X_SMAX]) and px[PF.X_INTRA] == 0.0
print('ok')
"""
    for var in ("", "fw", "fwl"):
        env = {**os.environ, "PATTERN_GA_VARIANT": var}
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, (var, r.stderr[-2000:])


def test_gene_menu_regression_other_variants():
    """리뷰 FWM M1: 줄 끝 주석이 다음 유전자(fail_exit)를 삼킨 사고 — 기존 변형의 차원·유전자 이름이 45a26ba 와 같아야 한다."""
    code = ("import sys, hashlib; sys.path[:0]=['.','src']; import research.experiments.pattern_ga_run as G; "
            "print(G.DIM, hashlib.sha256(repr(G.GENE_NAMES).encode()).hexdigest()[:16], 'fail_exit' in G.GENE_NAMES)")
    want = {"": "57 65c1beeff0784d87 True", "fw": "57 59668f18473cd611 True", "fwl": "55 59668f18473cd611 True"}
    for var, exp in want.items():
        env = {**os.environ, "PATTERN_GA_VARIANT": var}
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and r.stdout.strip() == exp, (var, r.stdout, r.stderr[-1500:])


def test_addon_stop_capped_at_5pct_of_add_price():
    """리뷰 FWM M2: +1R 불타기 추가분의 손절 = max(본전, 손절선, 추가가 × 0.95) — 갭 없이 −5% 넘는 추가분 손실이 없다."""
    rows = [(100, 101, 99.5, 100.5), (101, 106, 100.8, 105.5), (110, 111, 103, 104)] + [(104, 105, 103, 104)] * 5
    e, sp = 100.0, 95.0                                          # R = 5 → +1R = 105 (행 1 고가 106) → 행 2 시가 110 추가
    out = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_SMAX=0.05, X_INTRA=1.0), e=e, sp=sp)
    pa = 110 * (1 + PF.LIVE_X[PF.X_SLIP])
    assert int(out[PF.O_TA]) == 2 and out[PF.O_PA] == pytest.approx(pa)
    assert int(out[PF.O_TS]) == 2 and out[PF.O_PXS] == pytest.approx(pa * 0.95 * (1 - PF.LIVE_X[PF.X_SELL_SLIP]))   # 저가 103 ≤ 104.6
    out = _run(_book(rows), _px(X_PYR=PF.PYR_1R), e=e, sp=sp)
    assert int(out[PF.O_TS]) != 2                                # 끔: 추가분 선 = 본전 100 — 그날 안 판다


def test_intraday_line_gap_sells_before_open_add():
    """리뷰 FWM M3: 시가가 이미 장중 하방 선 아래면 추가하지 않고 시가 전량."""
    rows = [(100, 101, 99.5, 100.5), (101, 106, 100.8, 105.5), (101, 102, 99, 100)] + [(100, 101, 99, 100)] * 5
    ma50 = [90, 90, 102, 102, 102, 102, 102, 102]               # 행 2 의 선 = 전날(행 1) MA50 = 90 … 아래로 바꿈
    ma50[1] = 103.0                                              # 행 2 장중 선 = 103 > 시가 101
    out = _run(_book(rows, ma50), _px(X_PYR=PF.PYR_1R, X_TRAIL=PF.TRAIL_MA50, X_SMAX=0.05, X_INTRA=1.0))
    assert int(out[PF.O_X]) == 2 and out[PF.O_EXIT_OPEN] == 1 and out[PF.O_ADDED] == 0


def test_addon_line_hit_before_lower_lines_same_day():
    """리뷰 FWM S1: 같은 날 내려가는 길 — 더 높은 추가분 선(추가가 × 0.95)에서 추가분을 먼저 팔고, 나머지는 더 낮은 선에서."""
    rows = [(100, 101, 99.5, 100.5), (101, 106, 100.8, 105.5), (110, 111, 109, 110), (109, 109.5, 94, 95)] + [(95, 96, 94, 95)] * 4
    ks = 1 - PF.LIVE_X[PF.X_SELL_SLIP]
    out = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_SMAX=0.05, X_INTRA=1.0), e=100.0, sp=95.0)
    pa = 110 * (1 + PF.LIVE_X[PF.X_SLIP])
    assert int(out[PF.O_TA]) == 2 and int(out[PF.O_TS]) == 3 and out[PF.O_PXS] == pytest.approx(pa * 0.95 * ks)
    assert int(out[PF.O_X]) == 3                                  # 첫 단위는 같은 날 본전·손절선에서


def test_trail_start_waits_for_profit():
    """진화 1세대: trail_start = 1R 이면 고가가 진입가 + 1R 에 닿기 전엔 장중 추적선(전날 MA50)을 쓰지 않는다."""
    rows = [(100, 101, 99.5, 100.5), (100.5, 102, 99.6, 101), (101, 102, 99.7, 100)] + [(100, 101, 99.5, 100)] * 5
    ma50 = [100.2] * 8
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0, X_TRAIL_START=1.0))
    assert int(out[PF.O_REASON]) != PF.R_TRAIL
    out = _run(_book(rows, ma50), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0, X_TRAIL_START=0.0))
    assert int(out[PF.O_REASON]) == PF.R_TRAIL and int(out[PF.O_X]) == 1


def test_stop_policy_skip():
    assert _filter(_event(stop_raw=88.0), _gv(G_SMAX=0.05, G_SPOL=1.0))[0] == 0     # 구조 손절 12% > 5% → 안 산다
    assert _filter(_event(stop_raw=97.0), _gv(G_SMAX=0.05, G_SPOL=1.0))[0] == 1     # 3% → 산다


def test_safe_trail_waits_for_close_above_line():
    """2세대: 진입이 MA20 아래면 종가가 MA20 위에 한 번 선 뒤에야 추적(검증 차트: 한미반도체 2021-05 다음날 매도)."""
    rows = [(100, 101, 99.5, 100.5), (100.4, 101, 99.8, 100.6), (100.6, 104, 100.5, 103.5), (103.5, 104, 101, 101.5)] + [(101.5, 102, 101, 101.5)] * 4
    ma = [102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0]
    o = _run(_book(rows, ma), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0))
    assert int(o[PF.O_REASON]) == PF.R_TRAIL and int(o[PF.O_X]) == 1                 # 무장 없음: 다음날 바로
    o = _run(_book(rows, ma), _px(X_TRAIL=PF.TRAIL_MA50, X_INTRA=1.0, X_SAFE=1.0))
    assert int(o[PF.O_REASON]) == PF.R_TRAIL and int(o[PF.O_X]) == 3                 # 행 2 종가 103.5 > 102 → 행 3 장중 이탈에서


def test_safe_pyramid_needs_breakeven():
    rows = [(100, 101, 99.5, 100.5), (101, 106, 100.8, 105.5), (106, 107, 105, 106)] + [(106, 107, 105, 106)] * 5
    o = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_BE_R=2.0, X_SAFE=1.0), e=100.0, sp=95.0)
    assert o[PF.O_ADDED] == 0                                                         # +1R 닿았지만 본전(2R) 미확보 → 추가 없음
    o = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_BE_R=1.0, X_SAFE=1.0), e=100.0, sp=95.0)
    assert o[PF.O_ADDED] == 1


def test_be_close_basis():
    rows = [(100, 101, 99.5, 100.5), (101, 106, 100.8, 102), (102, 102.5, 99, 100)] + [(100, 101, 99, 100)] * 5
    o = _run(_book(rows), _px(X_BE_R=1.0), e=100.0, sp=95.0)
    assert int(o[PF.O_REASON]) == PF.R_BE                                             # 고가 106 ≥ 105 → 본전 → 행 2 저가 99 에서
    o = _run(_book(rows), _px(X_BE_R=1.0, X_BE_CLOSE=1.0), e=100.0, sp=95.0)
    assert int(o[PF.O_REASON]) != PF.R_BE                                             # 종가 102 < 105 → 본전 안 옮김


def test_atr_contraction_gene():
    F = _event()
    F[0, PE.FI["atr_s"]] = 98.0 * 0.03                                               # ATR20 3% of d−1 close 98
    assert _filter(F, _gv(G_ATRMAX=0.04))[0] == 1
    assert _filter(F, _gv(G_ATRMAX=0.025))[0] == 0
    assert _filter(F, _gv())[0] == 1                                                  # 끔


def test_gen3_profit_lock_at_3r():
    rows = [(100, 101, 99.5, 100.5), (101, 116, 100.8, 115), (115, 116, 104, 105)] + [(105, 106, 104, 105)] * 5
    o = _run(_book(rows), _px(X_BE_R=1.0, X_SAFE=2.0), e=100.0, sp=95.0)            # 행 1 고가 116 ≥ 115(+3R) → 행 2 손절선 105
    assert int(o[PF.O_X]) == 2 and o[PF.O_FIN] == pytest.approx(105 * (1 - PF.LIVE_X[PF.X_SELL_SLIP]))


def test_gen3_no_add_on_surge_day():
    rows = [(100, 101, 99.5, 100.5), (101, 112, 100.8, 111), (111, 112, 110, 111)] + [(111, 112, 110, 111)] * 5
    o = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_BE_R=1.0, X_SAFE=2.0), e=100.0, sp=95.0)
    assert o[PF.O_ADDED] == 0                                                         # 발동일 종가 +10.4% → 추가 안 함
    o = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_BE_R=1.0, X_SAFE=1.0), e=100.0, sp=95.0)
    assert o[PF.O_ADDED] == 1


def test_gen3_entry_quality_gene():
    F = _event()
    F[0, PE.FI["ma20_up"]], F[0, PE.FI["h0_wick"]] = 1.0, 0.2
    assert _filter(F, _gv(G_ENTQ=1.0))[0] == 1
    F[0, PE.FI["h0_wick"]] = 0.7
    assert _filter(F, _gv(G_ENTQ=1.0))[0] == 0
    assert _filter(F, _gv())[0] == 1


def test_gen3_avg_floor_survives_addon_sale():
    """검증 MUST-FIX 4 재현: 추가분이 추가가 −5% 에서 먼저 팔려도 첫 단위는 평단에서 나간다(순손익 ≥ 0)."""
    rows = [(100, 101, 99.5, 100.5), (101, 106, 100.8, 105.5), (110, 111, 108, 109), (108, 108.5, 104, 105), (104, 104.5, 99, 100)]
    rows += [(100, 101, 99, 100)] * 4
    o1 = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_BE_R=1.0, X_SAFE=1.0, X_SMAX=0.05, X_INTRA=1.0), e=100.0, sp=95.0)
    o2 = _run(_book(rows), _px(X_PYR=PF.PYR_1R, X_BE_R=1.0, X_SAFE=2.0, X_SMAX=0.05, X_INTRA=1.0), e=100.0, sp=95.0)
    assert o2[PF.O_ADDED] == 1 and o2[PF.O_PNL] > o1[PF.O_PNL] and o2[PF.O_PNL] >= -0.002


def test_gen4_lock_on_close_not_wick():
    """검증 MUST-FIX 6: 장중 윗꼬리로만 +3R 이면 4세대(exit_safe 3)는 잠그지 않는다."""
    rows = [(100, 101, 99.5, 100.5), (101, 116, 100.8, 106), (106, 107, 104, 105)] + [(105, 106, 104, 105)] * 5
    o2 = _run(_book(rows), _px(X_BE_R=1.0, X_SAFE=2.0), e=100.0, sp=95.0)
    o3 = _run(_book(rows), _px(X_BE_R=1.0, X_SAFE=3.0), e=100.0, sp=95.0)
    assert int(o2[PF.O_X]) == 2                                                       # 3세대: 윗꼬리로 잠금 → 행 2 저가 104 ≤ 105
    assert int(o3[PF.O_X]) != 2                                                       # 4세대: 종가 106 < 115 → 잠금 없음


def test_gen5_market_both():
    F = _event()
    fi = PE.FI
    F[0, fi["mkt_nhnl"]], F[0, fi["mkt_br200"]] = 0.6, 0.45
    assert _filter(F, _gv(G_MKT=1))[0] == 1 and _filter(F, _gv(G_MKT=4))[0] == 0
    F[0, fi["mkt_br200"]] = 0.55
    assert _filter(F, _gv(G_MKT=4))[0] == 1


def test_gen7_entry_quality_levels():
    F = _event()
    fi = PE.FI
    F[0, fi["ma20_up"]], F[0, fi["h0_wick"]], F[0, fi["runup20"]] = 0.0, 0.7, 0.35   # 정석 급등 뒤 긴 윗꼬리·MA20 평평
    assert _filter(F, _gv(G_ENTQ=1.0))[0] == 0
    assert _filter(F, _gv(G_ENTQ=2.0))[0] == 1                                      # 급등이 커서 단발 블로오프 아님
    F[0, fi["runup20"]] = 0.1
    assert _filter(F, _gv(G_ENTQ=2.0))[0] == 0
    assert _filter(F, _gv(G_ENTQ=3.0))[0] == 0


def test_gen8_climax_close_and_no_entry_day_be():
    # 행 1: 장중 고가 112(급등 선 위) 윗꼬리, 종가 104 → 종가 판단이면 안 판다
    rows = [(100, 101, 99.5, 100.5), (101, 112, 100.8, 104), (104, 106, 103, 105)] + [(105, 106, 104, 105)] * 5
    px_i = _px(X_CLIMAX=1.0, X_CLIMAX_DAY=0.10, X_CLIMAX_5D=0.5, X_CLIMAX_R=1.0, X_INTRA=1.0)
    o = _run(_book(rows), px_i, e=100.0, sp=95.0)
    assert int(o[PF.O_REASON]) == PF.R_CLIMAX and int(o[PF.O_X]) == 1
    px_c = _px(X_CLIMAX=1.0, X_CLIMAX_DAY=0.10, X_CLIMAX_5D=0.5, X_CLIMAX_R=1.0, X_INTRA=1.0, X_CLIMAX_CLOSE=1.0)
    o = _run(_book(rows), px_c, e=100.0, sp=95.0)
    assert int(o[PF.O_REASON]) != PF.R_CLIMAX or int(o[PF.O_X]) != 1
    # 진입일 고가 106 ≥ 1R(105) → safe 4 면 본전 미발동
    rows2 = [(100, 106, 99.5, 101), (101, 102, 99.8, 100.5), (100.5, 101, 99, 100)] + [(100, 101, 99.2, 100)] * 5
    o3 = _run(_book(rows2), _px(X_BE_R=1.0, X_SAFE=3.0), e=100.0, sp=95.0)
    o4 = _run(_book(rows2), _px(X_BE_R=1.0, X_SAFE=4.0), e=100.0, sp=95.0)
    assert int(o3[PF.O_REASON]) == PF.R_BE and int(o4[PF.O_REASON]) != PF.R_BE


def test_gen9_flow_genes():
    F = _event()
    fi = PE.FI
    F[0, fi["fl_smart20"]], F[0, fi["fl_indiv20"]], F[0, fi["fl_smart5"]], F[0, fi["fl_indiv5"]] = 0.05, -0.04, -0.02, 0.03
    assert _filter(F, _gv(G_FLMIN=0.0, G_FLWIN=1.0))[0] == 1                       # 20일 기관+외국인 순매수 5%
    assert _filter(F, _gv(G_FLMIN=0.0, G_FLWIN=0.0))[0] == 0                       # 5일은 순매도
    assert _filter(F, _gv(G_FLWIN=1.0, G_INDSELL=1.0))[0] == 1                     # 20일 개인 순매도
    assert _filter(F, _gv(G_FLWIN=0.0, G_INDSELL=1.0))[0] == 0                     # 5일 개인 순매수
    F[0, fi["fl_smart20"]] = np.nan
    assert _filter(F, _gv(G_FLMIN=0.1, G_FLWIN=1.0))[0] == 1                       # 모르면 통과(거부권)
