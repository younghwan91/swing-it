"""GA2 → 라이브 인계 — 유전자 19개(+ 수급 3개 + 청산 v2 5개) 1:1 대응, 인계 파일, 3자 적재 패리티(백테스트 = 연구 시뮬 = GA 커널 —
청산 v2 칸은 v2 커널 ``exits_v2_fast``; 모든 칸에서 v2 커널도 같이 대조), EPS 정의 패리티."""

import contextlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from research.signals import exits_v2_fast as xv
from research.signals import ga_fast as gf
from research.signals.minervini_exits import contraction_low_level, simulate_minervini
from research.signals.sepa import LIMIT_UP_RATIO, _Book
from swing_it.features.minervini import pivot_level
from swing_it.features.technical import moving_average
from swing_it.live import candidate as cd
from swing_it.live.backtest import run_backtest
from swing_it.live.config import (
    DEFAULT_STRATEGY,
    EXIT_V2_GENE_FIELDS,
    FLOW_GENE_FIELDS,
    GENE_FIELDS,
    GENE_LEVELS,
    StrategyConfig,
)
from swing_it.live.strategy import scan
from tests.test_live_parity import _synthetic

INF = math.inf
#: 기본 설정과 같은 유전자(ga2_run 이름).
DEFAULT_GENES = {"near_high": 0.75, "above_low": 1.30, "rs_min": 70.0, "vol_mult": 1.5, "breadth_min": 0.5, "pead_min": 0.70,
                 "rel_max": 1.0, "eps_yoy_min": None, "rev_yoy_min": None, "eps_accel": 0, "ma_stack": 3,
                 "ma200_slope_days": 20, "pivot_window": 50, "chase": 0.05, "stop_cap": 0.10, "take_r": 2.0, "trail_ma": 50,
                 "failed_breakout": 1, "max_hold": 120}
#: 모든 "끔/없음" 쪽 + 극단 단계의 유전자.
OFF_GENES = {"near_high": None, "above_low": None, "rs_min": None, "vol_mult": None, "breadth_min": None, "pead_min": None,
             "rel_max": None, "eps_yoy_min": None, "rev_yoy_min": None, "eps_accel": 0, "ma_stack": 0, "ma200_slope_days": 0,
             "pivot_window": 0, "chase": INF, "stop_cap": 0.15, "take_r": INF, "trail_ma": 0, "failed_breakout": 0,
             "max_hold": 20}


# ---------------------------------------------------------------------------
# 대응표
# ---------------------------------------------------------------------------

def test_gene_table_is_complete_19_plus_flow():
    from research.experiments.ga2_run import FLOW_CONT, FLOW_DISC, G2_CONT, G2_DISC, GENE_ORDER

    assert set(GENE_FIELDS) == set(GENE_ORDER) and len(GENE_FIELDS) == 19
    assert {n for n, *_ in FLOW_CONT} | {n for n, _ in FLOW_DISC} == set(FLOW_GENE_FIELDS)
    for name, levels in G2_DISC + FLOW_DISC:
        assert GENE_LEVELS[name] == levels, name
    for name, lo, hi, _ in G2_CONT:
        from swing_it.live.config import GENE_RANGES

        assert GENE_RANGES[name] == (lo, hi), name
    # 모든 대응 필드가 실제 StrategyConfig 필드다
    for fld in list(GENE_FIELDS.values()) + list(FLOW_GENE_FIELDS.values()):
        assert hasattr(DEFAULT_STRATEGY, fld), fld


def test_from_genes_default_roundtrip_and_sizing():
    cfg = StrategyConfig.from_genes(DEFAULT_GENES)
    for g, f in GENE_FIELDS.items():
        want = DEFAULT_GENES[g]
        got = getattr(cfg, f)
        assert got == (bool(want) if isinstance(got, bool) else want), (g, got, want)
    assert cfg.require_flow and cfg.smart_min is None                      # 수급 유전자 없음 → F 규칙 그대로
    assert cfg.risk_per_trade == cfg.max_position_frac == 0.10              # GA 슬롯 10% 고정과 같은 사이징
    assert cfg.to_genes() == {**DEFAULT_GENES}
    # 유전자 밖 값은 base 그대로
    assert (cfg.adv_floor, cfg.contraction_window, cfg.news_veto) == (5000.0, 10, True)
    # 기본 설정 자체도 왕복 동일(risk 만 다르다)
    back = StrategyConfig.from_genes(DEFAULT_STRATEGY.to_genes())
    assert back == StrategyConfig(risk_per_trade=0.10)


def test_from_genes_off_values_and_inf_strings():
    g = {**OFF_GENES, "chase": "inf", "take_r": "inf"}
    cfg = StrategyConfig.from_genes(g)
    assert cfg.near_high is None and cfg.rs_min is None and cfg.vol_mult is None and cfg.pead_rank_min is None
    assert cfg.ma_stack == 0 and cfg.ma200_slope_days == 0 and cfg.pivot_window == 0 and cfg.effective_pivot_window == 50
    assert cfg.chase == INF and cfg.take_r == INF and not cfg.partial_on and cfg.trail_ma == 0 and not cfg.has_trail
    assert cfg.failed_breakout is False and cfg.max_hold == 20 and cfg.stop_cap == 0.15
    assert StrategyConfig.from_genes(cfg.to_genes()) == cfg


def test_from_genes_flow_genes_replace_f_rule():
    cfg = StrategyConfig.from_genes({**DEFAULT_GENES, "smart_min": 0.02, "foreign_pos": 1, "persist_min": 10})
    assert not cfg.require_flow and cfg.smart_min == 0.02 and cfg.foreign_pos is True and cfg.persist_min == 10
    assert cfg.to_genes()["persist_min"] == 10 and cfg.to_genes()["foreign_pos"] == 1
    with pytest.raises(ValueError, match="수급"):
        StrategyConfig.from_genes({**DEFAULT_GENES, "smart_min": 0.02})


@pytest.mark.parametrize("bad", [
    {"bogus": 1}, {"rs_min": 120.0}, {"trail_ma": 30}, {"take_r": 2.5}, {"pivot_window": 25}, {"chase": 0.07},
    {"ma_stack": 4}, {"stop_cap": 0.2}, {"max_hold": 90}, {"near_high": 0.3}, {"eps_accel": None}, {"chase": "big"},
])
def test_from_genes_raises_on_unmapped_or_bad_level(bad):
    with pytest.raises(ValueError):
        StrategyConfig.from_genes({**DEFAULT_GENES, **bad})
    with pytest.raises(ValueError, match="빠진"):
        StrategyConfig.from_genes({k: v for k, v in DEFAULT_GENES.items() if k != "max_hold"})


#: 청산 v2 유전자 — 현 동작 값 / 원전 대표 칸(MINERVINI_EXITS_V2.md 미리 고정).
V2_OFF = {"be_r": INF, "take_frac": 0.5, "strength_sell": 0, "fb_tol": 0.0, "fb_days": 1}
V2_REP = {**DEFAULT_GENES, "be_r": 1.0, "take_r": 2.0, "take_frac": 0.5, "strength_sell": 3, "fb_tol": 0.02, "fb_days": 2,
          "stop_cap": 0.08, "trail_ma": 20}


def test_from_genes_exit_v2_genes():
    cfg = StrategyConfig.from_genes(V2_REP)
    assert (cfg.be_r, cfg.partial_frac, cfg.strength_ret, cfg.strength_ma_ext, cfg.fb_tol, cfg.fb_days) == (
        1.0, 0.5, 0.15, 0.30, 0.02, 2)
    assert cfg.stop_cap == 0.08 and cfg.trail_ma == 20 and cfg.strength_on and not cfg.exits_v2_default
    assert cfg.to_genes() == V2_REP and StrategyConfig.from_genes(cfg.to_genes()) == cfg
    # v2 유전자를 현 동작 값으로 주면 설정이 기본과 같고, to_genes 는 v2 를 넣지 않는다
    same = StrategyConfig.from_genes({**DEFAULT_GENES, **V2_OFF})
    assert same == StrategyConfig.from_genes(DEFAULT_GENES) and same.exits_v2_default and "be_r" not in same.to_genes()
    # take_r 1.0 은 v2 유전자가 있을 때만, 1/3 은 JSON 왕복 그대로
    one = StrategyConfig.from_genes({**DEFAULT_GENES, **V2_OFF, "take_r": 1.0, "take_frac": json.loads(json.dumps(1 / 3))})
    assert one.take_r == 1.0 and one.partial_frac == 1 / 3
    with pytest.raises(ValueError):
        StrategyConfig.from_genes({**DEFAULT_GENES, "take_r": 1.0})
    for bad in ({"be_r": 0.7}, {"take_frac": 0.4}, {"strength_sell": 4}, {"fb_tol": 0.03}, {"fb_days": 3}):
        with pytest.raises(ValueError):
            StrategyConfig.from_genes({**V2_REP, **bad})
    with pytest.raises(ValueError, match="다섯"):
        StrategyConfig.from_genes({**DEFAULT_GENES, "be_r": 1.0})
    with pytest.raises(ValueError, match="강세"):
        StrategyConfig(strength_ret=0.2).to_genes()
    assert set(EXIT_V2_GENE_FIELDS) == set(V2_OFF)


def test_be_cost_same_everywhere():
    from research.signals import minervini_exits as mx
    from swing_it.live import config

    assert xv.BE_COST == mx.BE_COST == config.BE_COST == 0.005


# ---------------------------------------------------------------------------
# 인계 파일
# ---------------------------------------------------------------------------

def test_candidate_write_load_and_validation(tmp_path):
    genes = {**DEFAULT_GENES, "take_r": INF, "pivot_window": 40}
    path = tmp_path / "live_candidate.json"
    doc = cd.write_candidate(path, genes, fitness={"port": 0.12}, window={"events": ["2023-04-01", "2026-04-01"]},
                             valid_from="2026-10-13", created="2026-10-10")
    assert doc["genes"]["take_r"] == "inf" and doc["strategy"]["take_r"] == "inf"
    raw = json.loads(path.read_text())
    assert set(raw) == set(cd.REQUIRED)
    cfg, loaded = cd.load_candidate(path)
    assert cfg == StrategyConfig.from_genes(genes) and loaded["valid_from"] == "2026-10-13"
    # 옛 모양(config 키)도 유전자로 읽힌다
    assert cd.genes_from_legacy({"config": {"take_r": "inf", "rs_min": 70}}) == {"take_r": INF, "rs_min": 70}

    def corrupt(mut):
        d = json.loads(path.read_text())
        mut(d)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(d))
        return p

    with pytest.raises(cd.CandidateError, match="strategy ≠"):
        cd.load_candidate(corrupt(lambda d: d["strategy"].update(max_hold=60)))
    with pytest.raises(cd.CandidateError, match="strategy_hash"):
        cd.load_candidate(corrupt(lambda d: d.update(strategy_hash="deadbeef")))
    with pytest.raises(cd.CandidateError, match="code_hash"):
        cd.load_candidate(corrupt(lambda d: d.update(code_hash="0" * 16)))
    cd.load_candidate(corrupt(lambda d: d.update(code_hash="0" * 16)), allow_code_mismatch=True)
    with pytest.raises(cd.CandidateError, match="모르는 키"):
        cd.load_candidate(corrupt(lambda d: d.update(extra=1)))
    with pytest.raises(cd.CandidateError, match="빠진 키"):
        cd.load_candidate(corrupt(lambda d: d.pop("window")))
    with pytest.raises(cd.CandidateError, match="유전자"):
        cd.load_candidate(corrupt(lambda d: d["genes"].update(trail_ma=30)))
    with pytest.raises(cd.CandidateError, match="날짜"):
        cd.load_candidate(corrupt(lambda d: d.update(valid_from="next monday")))
    with pytest.raises(cd.CandidateError):
        cd.load_candidate(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# 3자 적재 패리티 — run_backtest(unlimited) = simulate_minervini = ga_fast.engine_events
# ---------------------------------------------------------------------------

def _synthetic_v2(seed: int):
    """``_synthetic`` + 급등일(종목·날짜 1.5%: 시가 뒤 종가가 +16~24%) — 강세 매도(전일 대비 +15%, MA50 × 1.3)가 실제로 걸리게.

    급등일 이후 가격은 같은 배수로 이어지고(OHLC 일관), 정지일·폐지·시세 없음 구간은 ``_synthetic`` 그대로 둔다.
    """
    frame, ev_mask = _synthetic(seed)
    rng = np.random.default_rng(100 + seed)
    o, h, lo, c, v = (frame[k].to_numpy().copy() for k in ("open", "high", "low", "close", "volume"))
    n_days, n_codes = c.shape
    for j in range(n_codes):
        for t in np.flatnonzero(rng.random(n_days) < 0.015):
            if t == 0 or not (np.isfinite(c[t, j]) and v[t, j] > 0):
                continue
            k = float(np.exp(rng.uniform(0.15, 0.22)))
            c[t, j] *= k
            h[t, j] = max(h[t, j] * k, c[t, j])
            o[t + 1:, j] *= k
            h[t + 1:, j] *= k
            lo[t + 1:, j] *= k
            c[t + 1:, j] *= k
    idx, cols = frame["close"].index, frame["close"].columns
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)  # noqa: E731
    return {"open": mk(o), "high": mk(h), "low": mk(lo), "close": mk(c), "volume": mk(v)}, ev_mask


@contextlib.contextmanager
def _pure_v2():
    """v2 커널의 순수 파이썬 경로가 JIT 보조 함수를 부르지 않게(리뷰 A-M2) — numba 가 있어도 진짜 파이썬으로."""
    saved = {n: getattr(xv, n) for n in xv.HELPERS}
    try:
        for n, f in saved.items():
            setattr(xv, n, getattr(f, "py_func", f))
        yield
    finally:
        for n, f in saved.items():
            setattr(xv, n, f)


def _panels(frame, ev_mask, cfg: StrategyConfig):
    c = frame["close"]
    piv = pivot_level(frame["high"], cfg.effective_pivot_window)
    ok = pd.DataFrame(ev_mask, index=c.index, columns=c.columns) & piv.notna()
    const = lambda v: pd.DataFrame(v, index=c.index, columns=c.columns)  # noqa: E731
    ma = moving_average(c, cfg.trail_ma) if cfg.has_trail else const(np.nan)
    return {
        **frame, "market": pd.Series("KOSPI", index=c.columns), "ma50": ma,
        "universe": ok, "tt": ok, "breakout": ok,
        "rs": const(80.0), "breadth": const(0.6), "pead_rank": const(0.8), "rel": const(0.5), "F": const(1.0),
        "adv20": const(10_000.0), "pivot": piv, "contraction_low": contraction_low_level(frame["low"], 10),
    }


def _events(panels):
    ok = panels["universe"].to_numpy()
    di, ci = np.nonzero(ok)
    c = panels["close"]
    return pd.DataFrame({"code": c.columns[ci], "event_day": c.index[di], "market": "KOSPI",
                         "pivot": panels["pivot"].to_numpy()[di, ci],
                         "contraction_low": panels["contraction_low"].to_numpy()[di, ci]})


GENE_CASES = {
    "default": DEFAULT_GENES,
    "no_partial_trail20": {**DEFAULT_GENES, "take_r": INF, "trail_ma": 20},
    "no_partial_no_trail": {**DEFAULT_GENES, "take_r": INF, "trail_ma": 0, "failed_breakout": 0, "max_hold": 60},
    "no_trail_partial": {**DEFAULT_GENES, "trail_ma": 0},
    "no_breakout_chase_inf": {**DEFAULT_GENES, "pivot_window": 0, "chase": INF, "stop_cap": 0.15, "take_r": 4.0},
    "fb_off_pw30_tm10": {**DEFAULT_GENES, "failed_breakout": 0, "pivot_window": 30, "trail_ma": 10, "take_r": 1.5,
                         "stop_cap": 0.06, "max_hold": 20},
    "all_off": OFF_GENES,
    # 청산 v2(MINERVINI_EXITS_V2.md) — 원전 대표 칸 + 축마다 극단 단계
    "v2_rep": V2_REP,
    "v2_be05_fulltrail_x15": {**DEFAULT_GENES, "be_r": 0.5, "take_r": INF, "take_frac": 0.5, "strength_sell": 1,
                              "fb_tol": 0.05, "fb_days": 1, "stop_cap": 0.06, "trail_ma": 10},
    "v2_take1_third_y30_fboff": {**DEFAULT_GENES, "be_r": 2.0, "take_r": 1.0, "take_frac": 1 / 3, "strength_sell": 2,
                                 "fb_tol": 0.0, "fb_days": 1, "failed_breakout": 0, "trail_ma": 50},
    "v2_take3_third_days2": {**DEFAULT_GENES, "be_r": INF, "take_r": 3.0, "take_frac": 1 / 3, "strength_sell": 0,
                             "fb_tol": 0.0, "fb_days": 2, "stop_cap": 0.10, "trail_ma": 20},
    "v2_be15_take15_both_hold20": {**DEFAULT_GENES, "be_r": 1.5, "take_r": 1.5, "take_frac": 0.5, "strength_sell": 3,
                                   "fb_tol": 0.02, "fb_days": 1, "max_hold": 20, "trail_ma": 10},
    "v2_be1_nopartial_notrail": {**DEFAULT_GENES, "be_r": 1.0, "take_r": INF, "take_frac": 0.5, "strength_sell": 3,
                                 "fb_tol": 0.05, "fb_days": 2, "trail_ma": 0, "pivot_window": 30},
    "v2_be2_x15_chase_inf": {**DEFAULT_GENES, "be_r": 2.0, "take_r": 2.0, "take_frac": 1 / 3, "strength_sell": 1,
                             "fb_tol": 0.02, "fb_days": 2, "chase": INF, "stop_cap": 0.15, "max_hold": 60},
    "v2_off_given": {**DEFAULT_GENES, **V2_OFF},
}
V2_FEATURES = {"v2_rep": ("be", "strength", "fb2"), "v2_be05_fulltrail_x15": ("be", "strength"),
               "v2_take1_third_y30_fboff": ("be", "strength"), "v2_take3_third_days2": ("fb2",),
               "v2_be15_take15_both_hold20": ("be", "strength"), "v2_be1_nopartial_notrail": ("be", "strength", "fb2"),
               "v2_be2_x15_chase_inf": ("be", "strength", "fb2")}


def v2_params(cfg: StrategyConfig) -> dict:
    """설정 → v2 커널 인자(``exits_v2_fast.EXIT_V2_DEFAULT`` 모양)."""
    nz = lambda v: np.nan if v is None else float(v)  # noqa: E731
    return {"stop_floor": cfg.stop_floor, "stop_cap": cfg.stop_cap, "stop_mult": cfg.stop_mult, "take_r": cfg.take_r,
            "take_frac": cfg.partial_frac, "be_r": cfg.be_r, "ss_x": nz(cfg.strength_ret), "ss_y": nz(cfg.strength_ma_ext),
            "fb": cfg.failed_breakout, "fb_tol": cfg.fb_tol, "fb_days": cfg.fb_days, "ts_days": cfg.time_stop_days or 0,
            "max_hold": cfg.max_hold, "has_trail": cfg.has_trail, "lu_skip": True}


@pytest.mark.parametrize("name", list(GENE_CASES))
@pytest.mark.parametrize("seed", [0, 1])
def test_three_way_parity_for_loaded_gene_config(name, seed):
    cfg = StrategyConfig.from_genes(GENE_CASES[name])
    frame, ev_mask = (_synthetic_v2 if name.startswith("v2_") else _synthetic)(seed)
    panels = _panels(frame, ev_mask, cfg)
    ev = _events(panels)
    # 1. 라이브 백테스트(자본·슬롯 무제한)
    bt = run_backtest(panels, cfg, unlimited=True)
    live = bt.trades[bt.trades["exit_reason"] != "below_low"]
    # 2. 연구 시뮬레이터(같은 프로토콜)
    v2 = not cfg.exits_v2_default
    rs = simulate_minervini(frame["open"], frame["high"], frame["low"], frame["close"], panels["ma50"], ev,
                            volume=frame["volume"], stop_cap=cfg.stop_cap, take_r=cfg.take_r,
                            time_stop_days=cfg.time_stop_days, max_hold=cfg.max_hold,
                            chase=None if math.isinf(cfg.chase) else cfg.chase, failed_breakout=cfg.failed_breakout,
                            take_frac=cfg.partial_frac, be_r=cfg.be_r, strength_ret=cfg.strength_ret,
                            strength_ma_ext=cfg.strength_ma_ext, fb_tol=cfg.fb_tol, fb_days=cfg.fb_days)
    # 3. GA 커널(설정을 고른 엔진 자체, 순수 파이썬 경로) — v2 칸은 v2 커널, 모든 칸에서 v2 커널도 같이
    book = _Book(frame["open"], frame["high"], frame["low"], frame["close"], panels["ma50"], frame["volume"])
    ma50 = moving_average(frame["close"], 50).to_numpy(float).T
    with _pure_v2():
        g2 = xv.engine_events_v2(book, ev, v2_params(cfg), ma50=ma50, chase=cfg.chase, py=True)
    if v2:
        ga = g2
    else:
        params = {"stop_floor": cfg.stop_floor, "stop_cap": cfg.stop_cap, "stop_mult": cfg.stop_mult, "fixed_stop": np.nan,
                  "take_r": cfg.take_r, "ts_days": cfg.time_stop_days, "max_hold": cfg.max_hold,
                  "has_trail": cfg.has_trail, "lu_skip": True, "fb": cfg.failed_breakout}
        ga = gf.engine_events(book, ev, params, chase=cfg.chase, py=True)
    assert LIMIT_UP_RATIO == cfg.limit_up_ratio

    key = ["code", "event_day"]
    a, b, g, g2 = (x.set_index(key).sort_index() for x in (live, rs, ga, g2))
    assert list(a.index) == list(b.index) == list(g.index) == list(g2.index), (name, len(a), len(b), len(g), len(g2))
    for col in ("entry_day", "exit_day", "exit_reason", "hold_days"):
        assert (a[col] == b[col]).all() and (b[col] == g[col]).all() and (g[col] == g2[col]).all(), (name, col)
    for x, tol_d in ((b, 1e-12), (g, 1e-7), (g2, 1e-12)):            # GA 커널의 d 는 float32
        np.testing.assert_allclose(a["ret"].to_numpy(float), x["ret"].to_numpy(float), rtol=0, atol=1e-9)
        np.testing.assert_allclose(a["d"].to_numpy(float), x["d"].to_numpy(float), rtol=0, atol=tol_d)
    np.testing.assert_allclose(a["ret_half"].to_numpy(float), a["ret"].to_numpy(float), rtol=0, atol=1e-12)
    assert (a["partial_hit"].astype(bool) == g["partial_hit"].astype(bool)).all()
    assert (g2["partial_hit"] == g["partial_hit"].astype(bool)).all()
    if not v2:                                                       # v2 커널(전부 끔) = 옛 GA 커널, 비트 단위
        assert (g["ret"].to_numpy() == g2["ret"].to_numpy()).all(), name
    else:
        assert (b["strength_hit"].astype(bool) == g2["strength_hit"]).all()
        assert (b["be_moved"].astype(bool) == g2["be_moved"]).all()
        np.testing.assert_allclose(b["final_frac"].to_numpy(float), g2["fin_frac"].to_numpy(float), rtol=0, atol=1e-12)
        feats = V2_FEATURES.get(name, ())
        if "strength" in feats:
            assert g2["strength_hit"].sum() >= 3, (name, int(g2["strength_hit"].sum()))
        if "be" in feats:
            assert (g2["be_moved"] & (g2["exit_reason"] == "breakeven")).sum() >= 3, name
    assert len(a) > 50
    reasons = set(a["exit_reason"])
    if not cfg.partial_on and cfg.has_trail:
        assert "trail" in reasons and ("breakeven" not in reasons or math.isfinite(cfg.be_r))   # 본전은 be_r 로만
    if not cfg.has_trail:
        assert "trail" not in reasons
    if cfg.failed_breakout:
        assert "failed_breakout" in reasons


# ---------------------------------------------------------------------------
# 피처 — 추세 성분·돌파·수급 유전자가 scan 에 실린다 / EPS 정의 = GA 커널
# ---------------------------------------------------------------------------

def test_trend_and_breakout_masks_default_equal_research_and_genes_change_them():
    from research.signals.sepa import SepaPanels, skeleton_features
    from swing_it.live.features import breakout_mask, build_signal_panels, trend_components, trend_mask
    from tests.test_live_strategy import _raw_inputs

    inp = _raw_inputs()
    p = SepaPanels(open=inp.open, high=inp.high, low=inp.low, close=inp.close, volume=inp.volume,
                   trade_value=inp.trade_value, market=inp.market)
    feats = skeleton_features(p)
    tc = trend_components(inp.close, inp.high, inp.low)
    pd.testing.assert_frame_equal(trend_mask(tc, DEFAULT_STRATEGY), feats["tt"], check_names=False)
    brk, piv = breakout_mask(inp.high, inp.close, inp.volume, DEFAULT_STRATEGY)
    pd.testing.assert_frame_equal(brk, feats["breakout"], check_names=False)
    pd.testing.assert_frame_equal(piv, feats["pivot"], check_names=False)
    # 성분 끔 → 조건이 사라진다(NaN 통과), 돌파 끔 → 거래량 조건만(50일 창), 둘 다 끔 → 전부 참
    all_off = StrategyConfig.from_genes(OFF_GENES)
    assert trend_mask(tc, all_off).all().all()
    assert breakout_mask(inp.high, inp.close, inp.volume, all_off)[0].all().all()
    only_vol = StrategyConfig(pivot_window=0, vol_mult=1.5)
    b2, piv2 = breakout_mask(inp.high, inp.close, inp.volume, only_vol)
    pd.testing.assert_frame_equal(piv2, piv, check_names=False)
    assert b2.to_numpy().sum() > brk.to_numpy().sum()
    stack2 = StrategyConfig(ma_stack=2, near_high=None, above_low=None, ma200_slope_days=0)
    assert trend_mask(tc, stack2).to_numpy().sum() >= feats["tt"].to_numpy().sum()
    pw30 = StrategyConfig(pivot_window=30)
    got = build_signal_panels(inp, pw30)
    pd.testing.assert_frame_equal(got["pivot"], pivot_level(inp.high, 30), check_names=False)
    assert got["ma50"].isna().all().all() if not pw30.has_trail else True
    assert build_signal_panels(inp, StrategyConfig(trail_ma=0))["ma50"].isna().all().all()
    for k in ("smart", "foreign", "persist", "nh", "al", "stack", "slope10", "slope20", "slope40"):
        assert got[k].shape == inp.close.shape


def test_scan_flow_genes_and_none_thresholds_pass_nan():
    from tests.test_live_strategy import _scan_panels

    days, pn = _scan_panels()
    codes = pn["close"].columns

    def P(vals):
        return pd.DataFrame([vals] * 3, index=days, columns=codes, dtype=float)

    pn["smart"], pn["foreign"], pn["persist"] = P([0.05, -0.01, 0.05, 0.05, 0.05]), P([1, 1, -1, 1, 1]), P([12, 12, 12, 8, 12])
    pn["F"].loc[days[1], "000010"] = 0.0                                     # F 규칙이면 탈락
    flow = StrategyConfig(require_flow=False, smart_min=0.0, foreign_pos=True, persist_min=10)
    assert [c.code for c in scan(pn, days[1], flow)] == ["000010"]           # 000020 smart<0, 000030 foreign<0
    pn["rs"].loc[days[1], "000010"] = np.nan
    assert scan(pn, days[1], flow) == []
    assert [c.code for c in scan(pn, days[1], StrategyConfig(require_flow=False, smart_min=0.0, foreign_pos=True,
                                                             persist_min=10, rs_min=None))] == ["000010"]   # 끔 = NaN 통과


def _eps_fixture():
    days = pd.bdate_range("2023-01-02", periods=450)
    codes = pd.Index(["A", "B", "C"])
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.01, (450, 3)), axis=0) * 10_000, index=days, columns=codes)
    adj = raw.copy()
    adj.loc[: "2024-07-15", "A"] /= 2.0                                     # A 는 2024-07-16 에 1:2 분할(조정계수 0.5)
    ea = pd.DataFrame([
        ("A", "2023Q4", "20240331", "20240331", 200.0, 100.0, 1000.0, 800.0),
        ("A", "2024Q1", "20240515", "20240515", 120.0, 100.0, 1100.0, 900.0),
        ("A", "2024Q2", "20240814", "20240814", 150.0, 100.0, 1200.0, 1000.0),
        ("A", "2024Q3", "20241114", "20241114", 90.0, 100.0, 900.0, 1000.0),
        ("B", "2024Q1", "20240515", "20240515", 50.0, 40.0, 500.0, 450.0),
        ("B", "2024Q2", "20240814", "20240814", 60.0, 40.0, 520.0, 450.0),
        ("C", "2024Q2", "20240814", "20240814", 30.0, -10.0, 100.0, 90.0),
    ], columns=["code", "period", "avail_date", "knowledge_date", "netinc", "netinc_prior", "revenue", "revenue_prior"])
    # A: 분할 전 100 → 분할 후 200(원 주식수) — 조정하면 둘 다 200 → 희석 없음. B: 10 → 20 (2024-06-30, 분할 없음) → 희석.
    shares = pd.DataFrame({"code": ["A", "A", "A", "B", "B"],
                           "date": ["2023-03-31", "2023-12-31", "2024-07-31", "2023-03-31", "2024-06-30"],
                           "shares_outstanding": [100, 100, 200, 10, 20]})
    return days, codes, raw, adj, ea, shares


def test_eps_panels_match_ga_kernel_definition(monkeypatch):
    """같은 행에서 ``features.earnings_panels_ga`` = ``ga_run.eps_features``(DB 읽기만 가짜로)."""
    from research.experiments import ga_run
    from swing_it.live.features import earnings_panels_ga

    days, codes, raw, adj, ea, shares = _eps_fixture()
    monkeypatch.setattr(ga_run, "pd", pd)
    monkeypatch.setattr(ga_run.pd, "read_sql_query", lambda sql, con: shares.copy())
    import swing_it.storage as st

    monkeypatch.setattr(st, "read_earnings", lambda con, **kw: ea.copy())
    di, ci = np.meshgrid(np.arange(len(days)), np.arange(len(codes)), indexing="ij")
    di, ci = di.ravel(), ci.ravel()
    ref = ga_run.eps_features(None, days, codes, di, ci, adj, raw)
    got = earnings_panels_ga(ea, shares, adj, raw, days, codes)
    for mine, theirs in (("eps_yoy", "eps_yoy"), ("accel", "eps_acc"), ("rev_yoy", "rev_yoy"), ("eps_from_netinc", "eps_ni")):
        a = got[mine].to_numpy(float)[di, ci]
        b = np.asarray(ref[theirs], float)
        if mine in ("accel", "eps_from_netinc"):
            a = np.where(np.isnan(a), 0.0 if mine == "accel" else 1.0, a)   # GA 는 실적 행 없음을 0/1 로 채운다
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True, err_msg=mine)
    # 값이 뜻대로인지(2024-08-20, 최신 = 2024Q2): A 는 분할만 → 조정 주식수 200 = 200 → EPS YoY = 순이익 YoY 0.5(주식수 기준);
    # B 는 10 → 20 희석 → (60/20)/(40/10) − 1 = −0.25
    d = pd.Timestamp("2024-08-20")
    assert got["eps_yoy"].loc[d, "A"] == pytest.approx(0.5) and got["eps_from_netinc"].loc[d, "A"] == 0.0
    assert got["eps_yoy"].loc[d, "B"] == pytest.approx(-0.25) and got["eps_from_netinc"].loc[d, "B"] == 0.0
    assert got["eps_from_netinc"].loc[d, "C"] == 1.0 and got["rev_yoy"].loc[d, "C"] == pytest.approx(100 / 90 - 1)
