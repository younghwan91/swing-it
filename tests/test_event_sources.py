"""16세대 사건 원천 S(event_sources.py) — 매수정지·손절 기준가, 종류 거르기, 실적 반응일·좁은 박스 판정."""
import numpy as np
import pandas as pd
import pytest

from research.signals import event_families as EF
from research.signals import event_sources as ES
from research.signals import pattern_events as PE


def _fill(entry, lvl, stp, O, H, L, C, smax=0.08):  # noqa: E741
    I_ = np.zeros((1, len(PE.I_COLS)), np.int64)
    I_[0, PE.II["ci"]], I_[0, PE.II["i0"]] = 0, 1
    TR = np.ones_like(C, bool)
    return EF.fills(np.array([0]), I_, PE.II, O, H, L, C, TR, entry, {"stop_max": smax, "gap_max": None}, 0.0, 1.3,
                    np.zeros(1, np.int64), 10**9, None if lvl is None else np.array([lvl]), None if stp is None else np.array([stp]))


def _bars():
    O = np.array([[100.0, 100.0, 101.0]])  # noqa: E741
    H = np.array([[101.0, 102.0, 110.0]])
    L = np.array([[99.0, 99.0, 100.0]])
    C = np.array([[100.0, 101.0, 108.0]])
    return O, H, L, C


def test_stop_entry_uses_source_level():
    O, H, L, C = _bars()  # noqa: E741
    fx = _fill("stop", 105.0, 96.0, O, H, L, C, smax=0.12)
    assert len(fx["rows"]) == 1
    assert fx["e"][0] == pytest.approx(105.0 + PE.tick_dated_vec(np.array([105.0]), np.array([2]), np.zeros(1, np.int64), 10**9)[0])
    assert fx["sp"][0] < 96.0                                                 # 기준 − 1호가 (8% 상한 안)


def test_default_level_is_signal_bar():
    O, H, L, C = _bars()  # noqa: E741
    a, b = _fill("stop", None, None, O, H, L, C), _fill("stop", 102.0, 99.0, O, H, L, C)
    assert a["e"][0] == pytest.approx(b["e"][0]) and a["sp"][0] == pytest.approx(b["sp"][0])


def test_stop_capped_by_stop_max():
    O, H, L, C = _bars()  # noqa: E741
    fx = _fill("open", None, 80.0, O, H, L, C, smax=0.05)
    assert fx["sp"][0] == pytest.approx(fx["e"][0] * 0.95)


def test_gene_mask_kind_and_thresholds():
    XF = np.full((3, len(EF.X_COLS)), np.nan)
    XF[:, EF.XI["fam"]] = 2
    XF[:, EF.XI["s_kind"]] = [1, 1, 2]
    XF[:, EF.XI["tvx"]] = [3.0, 1.0, 3.0]
    XF[:, EF.XI["fl5_s"]] = [np.nan, 0.5, 0.5]
    x = lambda k: XF[:, EF.XI[k]]  # noqa: E731
    assert ES.gene_mask_s(x, {"s_kind": 1}).tolist() == [True, True, False]
    assert ES.gene_mask_s(x, {"s_kind": 1, "s_vmult": 2.0}).tolist() == [True, False, False]
    assert ES.gene_mask_s(x, {"s_kind": 1, "s_flow": 0.1}).tolist() == [True, True, False]     # 수급 모르면 통과


def _series(n=320):
    """상승 뒤 마지막 15일 좁은 박스·거래량 마름 — tight 사건이 박스 끝에 생긴다."""
    c = np.concatenate([100.0 * np.exp(-0.006 * np.arange(n - 15)[::-1]), np.full(15, 99.0)])
    h, l_ = c * 1.01, c * 0.99
    h[-15:], l_[-15:] = 100.5, 96.0
    o = c.copy()
    v = np.full(n, 1000.0)
    v[-15:] = 300.0
    return [x[None, :] for x in (o, h, l_, c, v)]


def test_tight_box_detected_with_levels():
    O, H, L, C, V = _series()  # noqa: E741
    n = C.shape[1]
    base = np.ones_like(C, bool)
    z = np.zeros_like(C)
    t = ES.build(O, H, L, C, V, V * C, C * 0.98, base, z, z, z, z, z, pd.DataFrame(columns=["code", "rcept_dt", "report_nm",
                 "is_correction"]), pd.bdate_range("2020-01-01", periods=n), pd.Index(["000001"]))
    tg = t[t["kind"] == ES.K["tight"]]
    assert len(tg) and tg["s"].max() == n - 2                                 # 마지막 행은 다음날이 없어 뺀다
    r = tg[tg["s"] == n - 2].iloc[0]
    assert r["lvl"] == pytest.approx(100.5) and r["stp"] == pytest.approx(96.0)


def test_earn_reaction_takes_first_qualifying_day():
    n = 60
    O = np.full((1, n), 100.0)  # noqa: E741
    C = np.full((1, n), 100.0)
    H, L = C * 1.01, C * 0.99
    days = pd.bdate_range("2021-01-01", periods=n)
    C[0, 41], O[0, 41], H[0, 41] = 105.0, 103.0, 105.5                         # 공시 다음날 반응
    tvx = np.full((1, n), 2.0)
    ret = np.zeros((1, n))
    ret[0, 41] = 0.05
    clv = np.full((1, n), 0.9)
    dart = pd.DataFrame({"code": ["000001"], "rcept_dt": [days[40]], "report_nm": ["연결재무제표기준영업(잠정)실적(공정공시)"],
                         "is_correction": [0]})
    base = np.ones((1, n), bool)
    z = np.zeros((1, n))
    t = ES.build(O, H, L, C, np.ones((1, n)), C, C, base, tvx, ret, clv, z, z, dart, days, pd.Index(["000001"]))
    e = t[t["kind"] == ES.K["earn"]]
    assert e["s"].tolist() == [41]
