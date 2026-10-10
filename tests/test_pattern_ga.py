"""패턴 GA(``research/signals/pattern_events.py``·``pattern_fast.py``·``research/experiments/pattern_ga_run.py``) — 결과 전 리뷰 2단계 점검 항목.

사전등록 ``research/logs/sepa_explore/PATTERN_GA.md`` (원문 + 정정 0·1·2·3, 정정 1 = ``REVIEW_PATTERN_GA.md`` 정정 1 초안·MINOR 1~12):
- 사건 표가 설정과 무관하다(유전자는 표를 거르기만 한다) · 거르기는 d−1 값만 쓴다(d+1 이후 자료를 바꿔도 같은 사건·서술자)
- 학습창 적합도는 test_lo 이후 가격을 쓰지 않는다(시가평가) · 모양 섞은 널은 모양 열만 같은 달 안에서 섞는다
- CMAwM ``tell`` 은 표본 연속 x 를 받는다 · 청산 경로는 룩어헤드 없음 · 확장 청산의 라이브 설정 = ``vcp._exit_path`` 동치
- 첫 관통 소진(두 번째 관통 차단) · d−1 종가 여유 · 바닥 반전 · H0 첫 점 패턴 · 불타기·샹들리에 손계산 · 포트폴리오 회계
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest

from research.signals import pattern_events as PE
from research.signals import pattern_fast as PF
from research.signals import vcp as V

sys.path.insert(0, "research/experiments")


def _runner():
    import pattern_ga_run as G
    return G


# ---------------------------------------------------------------------------
# 합성 자료
# ---------------------------------------------------------------------------

def _rand_book(n=520, k=6, seed=3, drift=0.001, vol=0.025):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, vol, (n, k)), axis=0))
    h = c * (1 + np.abs(rng.normal(0, 0.012, (n, k))))
    lo = c * (1 - np.abs(rng.normal(0, 0.012, (n, k))))
    o = np.clip(c * (1 + rng.normal(0, 0.008, (n, k))), lo, h)
    T = lambda x: np.ascontiguousarray(x.T)  # noqa: E731
    return T(o), T(h), T(lo), T(c), T(rng.lognormal(7, 0.5, (n, k)))


def _ctx(k, n):
    return {"base": np.ones((k, n), bool), "above50": np.ones((k, n), bool), "tt": np.ones((k, n), bool),
            "rs": np.full((k, n), 80.0), "breadth": np.full((k, n), 0.6), "adv20": np.full((k, n), 2000.0),
            "strat": np.zeros((k, n), np.int64), "market": np.zeros(k, np.int64), "month": np.arange(n) // 21}


def _tables(Op, H, L, C, Vv, d_lo=270, d_hi=None, **kw):
    k, n = C.shape
    TR = np.isfinite(H) & np.isfinite(C)
    return PE.build_tables(H, L, C, Vv, Op, TR, _ctx(k, n), d_lo=d_lo, d_hi=d_hi or n - 2,
                           thetas=kw.get("thetas", (0.05,)), ceil_wins=kw.get("ceil_wins", (120,)))


def _pivot_path(points, n):
    xs, ys = zip(*points, strict=True)
    return np.interp(np.arange(n), xs, ys)


def _book1(points, n, band=0.004):
    c = _pivot_path(points, n)
    H, L, C = (np.ascontiguousarray(x[None, :]) for x in (c * (1 + band), c * (1 - band), c))
    Op = C.copy()
    Vv = np.full((1, n), 1000.0)
    return Op, H, L, C, Vv


# ---------------------------------------------------------------------------
# 사건 표 — 설정 무관 · d−1 · 첫 관통
# ---------------------------------------------------------------------------

def test_event_table_deterministic_and_config_free():
    """표는 가격·맥락만으로 정해진다(유전자 입력이 없다) — 두 번 만들어도 같고, 거르기 통과 행은 늘 그 표의 부분집합."""
    Op, H, L, C, Vv = _rand_book()
    a, b = _tables(Op, H, L, C, Vv), _tables(Op, H, L, C, Vv)
    assert np.array_equal(a["off"], b["off"])
    assert np.allclose(a["F"], b["F"], equal_nan=True) and np.array_equal(a["I"], b["I"])
    assert len(a["F"]) > 0
    G = _runner()
    rng = np.random.default_rng(1)
    for _ in range(30):
        g = G.decode(G.random_x(rng, 1)[0])
        g.update(theta=0.05, ceil_win=120)
        g = G.canonical(g)
        lo, hi = PE.table_slice(a["off"], G.table_of(g))
        buf = [np.empty(max(hi - lo, 1), t) for t in (np.int64, float, float, float, np.bool_, np.bool_)]
        n = PF.filter_events(a["F"], lo, hi, G.gene_vec(g), 1.3, *buf)
        assert np.all((buf[0][:n] >= lo) & (buf[0][:n] < hi))
        # 청산 유전자만 바꾸면 후보는 같다
        g2 = G.canonical({**g, "take_r": math.inf, "trail": "chand", "trail_k": 4.0, "pyramid": "1r"})
        buf2 = [np.empty(max(hi - lo, 1), t) for t in (np.int64, float, float, float, np.bool_, np.bool_)]
        n2 = PF.filter_events(a["F"], lo, hi, G.gene_vec(g2), 1.3, *buf2)
        assert n2 == n and np.array_equal(buf[0][:n], buf2[0][:n2])


def test_filters_use_only_d_minus_1():
    """d+1 이후 자료를 바꿔도 d 이전 사건과 서술자는 그대로. d 의 시가·저가·종가를 바꿔도 d−1 서술자(체결 열 제외)는 그대로."""
    Op, H, L, C, Vv = _rand_book(seed=7)
    base = _tables(Op, H, L, C, Vv)
    I0, F0 = base["I"], base["F"]
    assert len(I0) > 3
    cut = int(np.median(I0[:, PE.II["d"]]))
    O2, H2, L2, C2, V2 = (x.copy() for x in (Op, H, L, C, Vv))
    rng = np.random.default_rng(9)
    for X in (O2, H2, L2, C2):
        X[:, cut + 1:] *= np.exp(rng.normal(0, 0.05, X[:, cut + 1:].shape))
    H2[:, cut + 1:] = np.maximum.reduce([H2[:, cut + 1:], O2[:, cut + 1:], C2[:, cut + 1:], L2[:, cut + 1:]])
    V2[:, cut + 1:] *= 3.0
    new = _tables(O2, H2, L2, C2, V2)
    m0, m1 = I0[:, PE.II["d"]] <= cut, new["I"][:, PE.II["d"]] <= cut
    assert np.array_equal(I0[m0], new["I"][m1])
    assert np.allclose(F0[m0], new["F"][m1], equal_nan=True)
    # 사건일 d 의 시가·저가·종가만 바꾼다(고가는 유지 — 관통 유지): 체결 열 밖 서술자는 같다
    r = I0[0]
    ci, d = int(r[PE.II["ci"]]), int(r[PE.II["d"]])
    O3, L3, C3 = Op.copy(), L.copy(), C.copy()
    O3[ci, d] = L3[ci, d] = C3[ci, d] = min(Op[ci, d], L[ci, d]) * 0.97
    b3 = _tables(O3, H, L3, C3, Vv)
    row = np.flatnonzero((b3["I"][:, PE.II["ci"]] == ci) & (b3["I"][:, PE.II["d"]] == d) & (b3["I"][:, PE.II["table"]] == r[PE.II["table"]]))
    assert len(row) == 1
    fill_cols = [PE.FI[c] for c in ("open_d", "low_d", "close_d")]
    keep = [j for j in range(len(PE.F_COLS)) if j not in fill_cols]
    assert np.allclose(F0[0, keep], b3["F"][row[0], keep], equal_nan=True)


def _wedge_book(after, n=None):
    # 0~200 오름 → H0 100(200) → A 82(212) → B 93(224) → C 80.5(236) → D 88(248) → E 79.5(260) → 이후 ``after``
    pts = [(0, 40.0), (200, 100.0), (212, 82.0), (224, 93.0), (236, 80.5), (248, 88.0), (260, 79.5)] + after
    n = n or pts[-1][0] + 1
    return _book1(pts, n), n


def test_first_crossing_consumes_pattern_second_crossing_blocked():
    """추세선을 한 번 넘었다가(관통) 다시 내려와 또 넘어도 사건은 첫 관통 하나뿐(정정 3 — 두 번째 관통 차단)."""
    (Op, H, L, C, Vv), n = _wedge_book([(266, 84.0), (268, 86.5), (272, 82.0), (280, 84.0), (284, 90.0), (290, 90.0)])
    t = _tables(Op, H, L, C, Vv, d_lo=250, d_hi=n - 2, ceil_wins=(260,))
    a, b = PE.table_slice(t["off"], PE.table_id(PE.THETAS.index(0.05), PE.CEIL_WINS.index(260), PE.B_TREND))
    ds = t["I"][a:b, PE.II["d"]]
    pid = t["I"][a:b, PE.II["iD"]]
    assert len(np.unique(pid)) == len(pid)                     # 패턴(행(D))마다 사건 ≤ 1
    # 이 경로에서 D=248 패턴의 추세선 첫 관통은 268 근처 — 284 근처의 두 번째 관통은 사건이 아니다
    assert not np.any((pid == 248) & (ds > 275))


def test_close_room_hygiene_blocks_effective_open_buy():
    """d−1 종가가 주문가 × 0.995 위면(사실상 시가 매수) 위생 탈락(정정 3)."""
    (Op, H, L, C, Vv), n = _wedge_book([(266, 83.0), (267, 84.15), (268, 86.0), (270, 86.0)])
    t = _tables(Op, H, L, C, Vv, d_lo=250, d_hi=n - 2, ceil_wins=(260,))
    fu = t["funnel"][(0.05, 260, "trend")]
    a, b = PE.table_slice(t["off"], PE.table_id(PE.THETAS.index(0.05), PE.CEIL_WINS.index(260), PE.B_TREND))
    F = t["F"][a:b]
    assert np.all(F[:, PE.FI["close_s"]] <= F[:, PE.FI["order"]] * 0.995 + 1e-9)
    assert fu["first_cross"] == fu["ok"] + sum(v for k, v in fu.items() if k not in ("ok", "first_cross"))


def test_h0_in_pattern_kind_and_bottom_entry():
    """확정 고점 2(H0, D) 패턴 = H0 첫 점(h0in 1, 상단선 H0–D), 바닥 반전 기준선 = E 봉 고가, E 뒤 ≤ 5일."""
    pts = [(0, 40.0), (200, 100.0), (215, 84.0), (230, 92.0), (245, 80.0), (247, 81.6), (249, 83.5), (255, 86.0)]
    Op, H, L, C, Vv = _book1(pts, 256)
    t = _tables(Op, H, L, C, Vv, d_lo=240, d_hi=254, ceil_wins=(120,))
    a, b = PE.table_slice(t["off"], PE.table_id(PE.THETAS.index(0.05), PE.CEIL_WINS.index(120), PE.B_BOTTOM))
    assert b > a
    F, I_ = t["F"][a:b], t["I"][a:b]
    assert np.all(F[:, PE.FI["h0in"]] == 1)
    iE = I_[:, PE.II["iE"]]
    assert np.allclose(F[:, PE.FI["level"]], H[0, iE])
    assert np.all(I_[:, PE.II["d"]] - iE <= 5)
    assert np.all(np.isnan(F[:, PE.FI["touch"]]))                # 저점이 둘뿐 → 닿음 NaN(touch_lo 3 탈락)
    assert np.all(I_[:, PE.II["iS"]] == I_[:, PE.II["i0"]])      # 패턴 시작 = H0


def test_shuffle_touches_only_shape_columns_within_month():
    Op, H, L, C, Vv = _rand_book(n=700, k=10, seed=11)
    t = _tables(Op, H, L, C, Vv, thetas=(0.03, 0.05), ceil_wins=(60, 120))
    F = t["F"]
    G = PE.shuffle_shape(F, t["off"], rep=3)
    shape = [PE.FI[c] for c in PE.SHAPE_COLS]
    other = [j for j in range(F.shape[1]) if j not in shape]
    assert np.allclose(F[:, other], G[:, other], equal_nan=True)
    assert not np.allclose(F[:, shape], G[:, shape], equal_nan=True)
    mo = F[:, PE.FI["month"]]
    for tb in range(len(t["off"]) - 1):
        a, b = PE.table_slice(t["off"], tb)
        for m in np.unique(mo[a:b]):
            r = a + np.flatnonzero(mo[a:b] == m)
            fa = {tuple(np.nan_to_num(x, nan=-9e9)) for x in F[r][:, shape]}
            ga = {tuple(np.nan_to_num(x, nan=-9e9)) for x in G[r][:, shape]}
            assert fa == ga                                     # 같은 달 안 행 묶음 재배정
    assert np.allclose(PE.shuffle_shape(F, t["off"], rep=3), G, equal_nan=True)


# ---------------------------------------------------------------------------
# 청산 경로 — 라이브 동치 · 룩어헤드 없음 · 확장 유전자 손계산
# ---------------------------------------------------------------------------

def _arrays(Op, H, L, C):
    k, n = C.shape
    days = pd.bdate_range("2020-01-01", periods=n)
    mk = lambda x: pd.DataFrame(x.T, index=days, columns=[f"S{i}" for i in range(k)])  # noqa: E731
    ma = mk(C).rolling(50, min_periods=50).mean()
    A = V.Arrays(mk(Op), mk(H), mk(L), mk(C), ma, mk(np.full_like(C, 1000.0)))
    ma20 = np.ascontiguousarray(mk(C).rolling(20, min_periods=20).mean().to_numpy().T)
    return A, ma20


def _atr(H, L, C):
    prev = np.concatenate([np.full((C.shape[0], 1), np.nan), C[:, :-1]], axis=1)
    tr = np.fmax(np.fmax(H - L, np.abs(H - prev)), np.abs(L - prev))
    return np.ascontiguousarray(pd.DataFrame(tr.T).rolling(20, min_periods=20).mean().to_numpy().T)


@pytest.mark.parametrize("fb_days,be_r", [(1, np.inf), (2, np.inf), (1, 1.0), (2, 1.5)])
def test_live_exit_parity_with_vcp(fb_days, be_r):
    """라이브 설정(2R 절반 · 익절 뒤 MA50 · 돌파 실패 · be_r) 의 확장 청산 = :func:`vcp._exit_path_py` (낙관·보수 진입일 둘 다)."""
    Op, H, L, C, _ = _rand_book(n=400, k=5, seed=21, drift=0.0015)
    A, ma20 = _arrays(Op, H, L, C)
    atr = _atr(H, L, C)
    LIM = A.CUT.copy()
    px = PF.LIVE_X.copy()
    px[PF.X_FB_DAYS], px[PF.X_BE_R] = fb_days, be_r
    out = np.empty(PF.N_O)
    import inspect
    extra = (False,) if "sell_tick" in inspect.signature(V._exit_path_py).parameters else ()   # 쐐기 D 스위치(기본 끔)
    rng = np.random.default_rng(4)
    for _ in range(150):
        ci, d = int(rng.integers(0, 5)), int(rng.integers(60, 380))
        e = Op[ci, d] * 1.001
        sp = e * (1 - rng.uniform(0.02, 0.09))
        fb = e * (1 - rng.uniform(0, 0.03))
        for skip0 in (True, False):
            tgt = e * (1 + 2.0 * (e - sp) / e)
            ref = V._exit_path_py(A.O, A.H, A.L, A.C, A.MA, A.TR, A.PL, A.CUT, ci, d, e, sp, tgt, fb, skip0, False, 0.5, 120,
                                  e, 0.001, fb_days, be_r, *extra)
            PF._path_py(A.O, A.H, A.L, A.C, ma20, A.MA, atr, A.TR, LIM, ci, d, e, sp, fb, np.inf, skip0, px, out)
            reason = int(out[PF.O_REASON])
            ref_reason = ref[0] if ref[0] != V.R_DELISTED else V.R_DATA_END
            assert reason in (ref_reason, ref[0])
            assert int(out[PF.O_X]) == ref[1]
            assert (out[PF.O_TP] >= 0) == bool(ref[2])
            if ref[2]:
                assert int(out[PF.O_TP]) == ref[3] and out[PF.O_PX1] == pytest.approx(ref[4])
            assert out[PF.O_FIN] == pytest.approx(ref[5])
            ret = (0.5 * ref[4] / e + 0.5 * ref[5] / e - 1) if ref[2] else ref[5] / e - 1
            assert out[PF.O_PNL] == pytest.approx(ret)


def test_exit_path_no_lookahead():
    """청산 행 x 뒤의 가격을 바꿔도 경로 결과는 같다(인과)."""
    Op, H, L, C, _ = _rand_book(n=400, k=3, seed=5)
    A, ma20 = _arrays(Op, H, L, C)
    atr = _atr(H, L, C)
    G = _runner()
    rng = np.random.default_rng(8)
    out1, out2 = np.empty(PF.N_O), np.empty(PF.N_O)
    for _ in range(60):
        g = G.decode(G.random_x(rng, 1)[0])
        px = G.exit_vec(g)
        ci, d = int(rng.integers(0, 3)), int(rng.integers(60, 300))
        e, sp = Op[ci, d] * 1.001, Op[ci, d] * 0.95
        PF._path_py(A.O, A.H, A.L, A.C, ma20, A.MA, atr, A.TR, A.CUT, ci, d, e, sp, e * 0.99, H[ci, d] * 1.2, True, px, out1)
        x = int(out1[PF.O_X])
        O2, H2, L2, C2 = (z.copy() for z in (A.O, A.H, A.L, A.C))
        for z in (O2, H2, L2, C2):
            z[:, x + 1:] *= 1.5
        PF._path_py(O2, H2, L2, C2, ma20, A.MA, atr, A.TR, A.CUT, ci, d, e, sp, e * 0.99, H[ci, d] * 1.2, True, px, out2)
        assert np.allclose(out1, out2, equal_nan=True)


def _hand_book(rows, atr_v=2.0):
    a = np.array(rows, float).T
    Op, H, L, C = (np.ascontiguousarray(a[i][None, :]) for i in range(4))
    n = Op.shape[1]
    ma = np.full((1, n), np.nan)
    atr = np.full((1, n), atr_v)
    return Op, H, L, C, ma, atr, np.ones((1, n), bool), np.array([n], np.int64)


def _px(**kw):
    px = PF.LIVE_X.copy()
    px[PF.X_FROM_ENTRY] = 1.0
    for k, v in kw.items():
        px[getattr(PF, k)] = v
    return px


def test_pyramid_plus_1r_add_and_addon_stop():
    """+1R 도달 다음날 시가에 1/3 추가(처음 2/3) → 추가분 손절 = max(본전, 손절선) 에서 추가분만 판다."""
    rows = [(100, 101, 99.5, 100.5), (101, 111, 100.8, 110), (110, 112, 109, 111), (108, 109, 99.8, 103), (104, 105, 103, 104)]
    rows += [(104, 105, 103, 104)] * 5
    Op, H, L, C, ma, atr, TR, LIM = _hand_book(rows)
    e, sp = 100.0, 90.0                                         # R = 10 → +1R = 110 (행 1 고가 111)
    out = np.empty(PF.N_O)
    px = _px(X_PYR=PF.PYR_1R, X_TAKE_R=np.inf, X_FB_DAYS=0)
    PF._path_py(Op, H, L, C, ma, ma, atr, TR, LIM, 0, 0, e, sp, -np.inf, np.inf, True, px, out)
    assert int(out[PF.O_TA]) == 2 and out[PF.O_PA] == pytest.approx(110 * 1.001)
    assert int(out[PF.O_TS]) == 3 and out[PF.O_PXS] == pytest.approx(100 * 0.999)   # 저가 99.8 ≤ 본전 100
    wb = PF.WB_PYR
    exp = -wb - (1 - wb) + (1 - wb) / (110 * 1.001) * 100 * 0.999 + wb / e * out[PF.O_FIN]
    assert out[PF.O_PNL] == pytest.approx(exp)


def test_pyramid_cancelled_when_stop_zone_touched_same_day():
    rows = [(100, 101, 99.5, 100.5), (101, 111, 100.8, 110), (110, 112, 99.0, 105)] + [(104, 105, 103, 104)] * 5
    Op, H, L, C, ma, atr, TR, LIM = _hand_book(rows)
    out = np.empty(PF.N_O)
    PF._path_py(Op, H, L, C, ma, ma, atr, TR, LIM, 0, 0, 100.0, 90.0, -np.inf, np.inf, True,
                _px(X_PYR=PF.PYR_1R, X_TAKE_R=np.inf, X_FB_DAYS=0), out)
    assert int(out[PF.O_TA]) == -1 and out[PF.O_ADDED] == 0


def test_chandelier_trail_from_entry_and_tighten():
    """샹들리에 = 진입 뒤 최고 고가 − k × ATR20(전날), 진입 다음날부터, +3R 뒤 k 절반."""
    rows = [(100, 102, 99, 101), (101, 104, 100, 103), (103, 105, 102, 104), (104, 104.5, 99, 99.5), (99, 100, 98, 99)]
    rows += [(99, 100, 98, 99)] * 4
    Op, H, L, C, ma, atr, TR, LIM = _hand_book(rows)
    out = np.empty(PF.N_O)
    PF._path_py(Op, H, L, C, ma, ma, atr, TR, LIM, 0, 0, 100.0, 90.0, -np.inf, np.inf, True,
                _px(X_TRAIL=PF.TRAIL_CHAND, X_TRAIL_K=2.0, X_TAKE_R=np.inf, X_FB_DAYS=0), out)
    # 행 3: 최고 105 − 2×2 = 101 > 종가 99.5 → 행 4 시가 청산
    assert int(out[PF.O_REASON]) == V.R_TRAIL and int(out[PF.O_X]) == 4 and out[PF.O_FIN] == 99
    rows2 = [(100, 102, 99, 101), (101, 131, 100, 128), (128, 129, 125, 126.5), (126, 127, 124, 125)] + [(125, 126, 124, 125)] * 4
    Op, H, L, C, ma, atr, TR, LIM = _hand_book(rows2, atr_v=3.0)
    for tight, want in ((0, -1), (1, 3)):      # 조임 없음: 131 − 2×3 = 125 (종가 ≥ 125) · +3R = 130 ≤ 131 → 다음날부터 131 − 1×3 = 128 > 126.5
        PF._path_py(Op, H, L, C, ma, ma, atr, TR, LIM, 0, 0, 100.0, 90.0, -np.inf, np.inf, True,
                    _px(X_TRAIL=PF.TRAIL_CHAND, X_TRAIL_K=2.0, X_TIGHTEN=tight, X_TAKE_R=np.inf, X_FB_DAYS=0), out)
        if want > 0:
            assert int(out[PF.O_REASON]) == V.R_TRAIL and int(out[PF.O_X]) == want
        else:
            assert int(out[PF.O_REASON]) != V.R_TRAIL


# ---------------------------------------------------------------------------
# 엔진 — 합성 캐시 · 시가평가 · 포트폴리오 회계
# ---------------------------------------------------------------------------

def _synthetic_cache(tmp_path, seed=13, n=1400, k=12, bump=None):
    G = _runner()
    Op, H, L, C, Vv = _rand_book(n=n, k=k, seed=seed, drift=0.0008)
    if bump is not None:
        r0, f = bump
        for X in (Op, H, L, C):
            X[:, r0:] *= f
    days = pd.bdate_range("2016-09-01", periods=n)
    TR = np.isfinite(C)
    ctx = _ctx(k, n)
    ctx["strat"] = np.tile((np.arange(k) % 4)[:, None], (1, n)).astype(np.int64)
    ctx["month"] = ((days.year - days[0].year) * 12 + days.month - 1).to_numpy()
    d_lo = int(days.searchsorted(pd.Timestamp(G.EVENT_LO)))
    t = PE.build_tables(H, L, C, Vv, Op, TR, ctx, d_lo=d_lo, d_hi=n - 2, thetas=(0.03, 0.05), ceil_wins=(60, 120))
    ma20 = np.ascontiguousarray(pd.DataFrame(C.T).rolling(20, min_periods=20).mean().to_numpy().T)
    ma50 = np.ascontiguousarray(pd.DataFrame(C.T).rolling(50, min_periods=50).mean().to_numpy().T)
    strat = ctx["strat"]
    bs, bn = PF.basket_tables(Op, C, strat, 40)
    arrs = {"O": Op, "H": H, "L": L, "C": C, "MA20": ma20, "MA50": ma50, "ATR": _atr(H, L, C), "TR": TR,
            "LIM": np.full(k, n, np.int64), "strat_cs": strat, "bs": bs, "bn": bn, "cost_tab": np.full((2, n), 0.003),
            "days_ns": days.values.astype("datetime64[ns]").astype(np.int64), "month_of": ctx["month"].astype(np.int64),
            "year_of": (days.year - days[0].year).to_numpy().astype(np.int64), "F": t["F"], "I": t["I"], "off": t["off"],
            "codes": np.array([f"S{i:03d}" for i in range(k)], dtype="<U12")}
    G.write_cache(str(tmp_path), arrs, {"version": G.CACHE_VERSION, "n_events": int(len(t["F"])), "p": 0.4, "funnel": {},
                                         "n_scan_rows": 0, "base_rows": 0})
    return G, G.Engine(str(tmp_path))


def _loose(G, **kw):
    x = G.x0_feasible()
    g = G.decode(x)
    g.update(theta=0.03, ceil_win=60, entry="horiz", gap_max=0.05, stop_cap=0.12)
    g.update(kw)
    return G.canonical(g)


def test_train_fitness_ignores_prices_after_test_lo(tmp_path):
    """학습창 적합도는 학습창 끝(test_lo 직전 거래일 종가) 시가평가 — test_lo 이후 가격을 바꿔도 같은 값(purge 없이)."""
    from swing_it.validation.walkforward import Fold
    G, E = _synthetic_cache(tmp_path / "a")
    f = Fold("2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01")
    tr, _ = G.fold_windows(E, [f])[0]
    g = _loose(G)
    r1 = E.evaluate(g, tr)
    assert r1["n"] > 0
    row = E.row(f.test_lo)
    _, E2 = _synthetic_cache(tmp_path / "b", bump=(row, 1.7))
    # 사건 표도 test_lo 이후 사건만 달라지므로 학습창 진입 후보가 같다 — 적합도가 같아야 한다
    r2 = E2.evaluate(g, G.fold_windows(E2, [f])[0][0])
    assert r1["n"] == r2["n"] and r1["fit"] == pytest.approx(r2["fit"]) and r1["lower"] == pytest.approx(r2["lower"])


def test_portfolio_single_trade_accounting(tmp_path):
    """한 트레이드: 슬롯 10% · 비용 2배 · 진입일 모호 없음 → 끝 자산 = 1 + 0.1 × (손익 − 2 × 비용)."""
    G, E = _synthetic_cache(tmp_path)
    w = G.Win(E, "2017-09-15", "2021-01-01", None)
    g = _loose(G)
    c = E.candidates(g, w)
    nz = np.flatnonzero(~c["amb"])
    assert len(nz)
    one = {k: (v[nz[:1]] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    r = E.run([one], [G.exit_vec(g)], w.t_lo, E.end_row + 1)
    assert r["n"] == 1
    rec = r["rec"][0]
    eq = np.prod(1 + r["ret"])
    wb = rec[PF.Q_WB]
    assert eq == pytest.approx(1 + 0.1 * (rec[PF.Q_PNL] - 2 * 0.003 * wb), rel=1e-9)
    # 대조 0 이면 초과 = 수익
    assert np.allclose(r["exc"], r["ret"] - r["bench"])


def test_stitched_uses_each_fold_config_for_new_entries_only(tmp_path):
    G, E = _synthetic_cache(tmp_path)
    from swing_it.validation.walkforward import Fold
    folds = [Fold("2017-01-01", "2019-01-01", "2019-01-01", "2020-01-01"), Fold("2018-01-01", "2020-01-01", "2020-01-01", "2021-06-01")]
    wins = G.fold_windows(E, folds)
    a, b = _loose(G), _loose(G, entry="trend", theta=0.05, ceil_win=120)
    st = E.stitched([a, b], [w for _, w in wins])
    rec = st["rec"]
    assert len(rec)
    d = rec[:, PF.Q_D]
    assert np.all(rec[d < wins[1][1].d_lo, PF.Q_PAR] == 0) and np.all(rec[d >= wins[1][1].d_lo, PF.Q_PAR] == 1)
    tb = E.I[rec[:, PF.Q_EV].astype(int), PE.II["table"]]
    assert set(tb[rec[:, PF.Q_PAR] == 0]) <= {G.table_of(a)} and set(tb[rec[:, PF.Q_PAR] == 1]) <= {G.table_of(b)}


def test_shape_null_engine_swaps_only_shape_columns(tmp_path):
    G, E = _synthetic_cache(tmp_path)
    F0 = np.asarray(E.F).copy()
    E.set_rep(2)
    shape = [PE.FI[c] for c in PE.SHAPE_COLS]
    other = [j for j in range(F0.shape[1]) if j not in shape]
    assert np.allclose(np.asarray(E.F)[:, other], F0[:, other], equal_nan=True)
    E.set_rep(None)
    assert np.allclose(np.asarray(E.F), F0, equal_nan=True)


# ---------------------------------------------------------------------------
# CMA · 부호화
# ---------------------------------------------------------------------------

class _FakeCMA:
    """``CMAwM`` 대역: ask 는 (평가용, tell 용) 을 다르게 돌려주고 tell 인자를 기록한다."""
    told: list = []
    asked: list = []

    def __init__(self, mean, sigma, bounds, steps, seed, population_size, cov):
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        self.dim = len(mean)
        self.bounds = bounds
        self.n_gen = 0

    def should_stop(self):
        return self.n_gen >= 3

    def ask(self):
        x_tell = self.rng.normal(0.5, 0.3, self.dim)
        x_eval = np.clip(np.round(x_tell * 4) / 4, self.bounds[:, 0], self.bounds[:, 1])
        _FakeCMA.asked.append((x_eval.copy(), x_tell.copy()))
        return x_eval, x_tell

    def tell(self, sols):
        self.n_gen += 1
        _FakeCMA.told.extend(s[0].copy() for s in sols)


def test_cmawm_tell_gets_sampled_continuous_x():
    G = _runner()
    _FakeCMA.told, _FakeCMA.asked = [], []
    seen = []
    hist, rst = G.cma_bipop_x0(lambda x: (seen.append(x.copy()), -float(np.sum(x)))[1], 3, (4, 2), budget=60, seed=1,
                                x0=np.full(5, 0.5), opt_cls=_FakeCMA)
    assert len(hist) == 60 and len(_FakeCMA.told) > 0
    for told, (x_eval, x_tell) in zip(_FakeCMA.told, _FakeCMA.asked, strict=False):
        assert np.array_equal(told, x_tell)                     # tell = 표본 연속 x(순서 그대로)
        assert not np.array_equal(told, x_eval)                 # 평가용(반올림·자름) x 가 아니다
    assert all(np.array_equal(s, a) for s, (a, _) in zip(seen, _FakeCMA.asked, strict=True))   # 적합도는 평가용 x
    assert rst[0]["regime"] == "first"


def test_decode_encode_roundtrip_and_canonical():
    G = _runner()
    rng = np.random.default_rng(0)
    for x in G.random_x(rng, 200):
        g = G.decode(x)
        assert G.gkey(G.decode(G.encode(g))) == G.gkey(g)
        if g["entry"] == "h0":
            assert g["pyramid"] != "newhigh"                   # M1-c 금지
        if g["entry"] == "bottom":
            assert g["fb_days"] is None and g["stop_anchor"] is None
        if g["take_r"] == math.inf:
            assert g["take_frac"] is None
    g0 = G.decode(G.x0_feasible())                               # 실현 가능 첫 평균 = 모양·맥락 끔
    assert all(g0[k] is None for k in ("sup_hw", "slo_hw", "rho_max", "span_hw", "ceil_min", "depth_max", "dry_max", "rs_min",
                                       "breadth_min", "runup_min", "rest_gap_max", "e_min"))
    assert g0["touch_lo"] == 2 and g0["ceil_kind"] == "single"


def test_shape_names_and_user_region():
    G = _runner()
    assert G.shape_name(-0.6, -0.2) == "하락 쐐기"
    assert G.shape_name(-0.2, -0.6) == "하락 채널/확산"
    assert G.shape_name(0.05, 0.5) == "평평한 천장"
    assert G.shape_name(-0.5, 0.5) == "대칭 삼각"
    assert G.shape_name(0.3, 0.8) == "상승 쐐기"
    g = G.decode(G.x0_feasible())
    g.update(sup_c=-0.8, sup_hw=0.3, slo_c=-0.5, slo_hw=0.2, entry="trend", fb_days=2, runup_min=0.35, rest_gap_max=15.0)
    g = G.canonical(g)
    assert G.user_region(g) == {"W": True, "WD": False, "any": True}
    g2 = G.canonical({**g, "rest_gap_max": 30.0})
    assert not G.user_region(g2)["any"]
    g3 = G.canonical({**g, "entry": "bottom", "h0_in_pattern": 1})
    assert G.user_region(g3) == {"W": False, "WD": True, "any": True}
    off = G.shape_off(g)
    assert off["sup_hw"] is None and off["slo_hw"] is None and off["touch_lo"] == 2 and off["entry"] == "trend"
