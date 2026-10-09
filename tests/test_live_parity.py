"""패리티 — 라이브 백테스트(자본·슬롯 무제한) 트레이드 = 연구 ``simulate_minervini`` 트레이드 (설계 §1·§5).

합성 패널(랜덤워크 + 갭·정지일·상한가 시가·상장폐지·데이터 끝)에 무작위 사건을 심고, 같은 사건을
``swing_it.live.backtest.run_backtest(unlimited=True)`` 와 ``research.signals.minervini_exits.simulate_minervini``
(기본값 = 원전 프로토콜, 겹침 금지, 거래량 넘김)에 넣어 사건마다 진입일·청산일·청산 사유·총수익(1e-9)·익절 여부를 대조한다.

두 프로토콜로 돈다 — ``new`` = 라이브 기본(돌파 실패 켬, 시간 손절 끔, 설계 §6) / ``old`` = 사전등록 원전(시간 손절 20일,
돌파 실패 끔). 두 시뮬레이터에 **같은** 설정을 준다.

**알려진 차이 하나(설계 §2):** 진입 시가 ≤ 수축 저점이면 연구는 사건을 버리고(``skip_below_low``) 라이브는 체결 직후
그 시가에 청산한다(``below_low``, 총수익 0). 그 사건은 비교에서 빼고 개수를 센다 — 연구의 ``skip_below_low`` 개수와 같아야 한다.
"""

import numpy as np
import pandas as pd
import pytest

from research.signals.minervini_exits import contraction_low_level, simulate_minervini
from swing_it.features.minervini import pivot_level
from swing_it.features.technical import moving_average
from swing_it.live.backtest import run_backtest
from swing_it.live.config import StrategyConfig
from swing_it.live.strategy import PositionState

PROTOCOLS = {
    "new": {"failed_breakout": True, "time_stop_days": None},
    "old": {"failed_breakout": False, "time_stop_days": 20},
}


def _synthetic(seed: int, n_days: int = 700, n_codes: int = 40):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"{i:05d}0" for i in range(n_codes)]
    OP = np.empty((n_days, n_codes))
    H, L, C, V = OP.copy(), OP.copy(), OP.copy(), OP.copy()
    for j in range(n_codes):
        px = 10_000.0 * np.exp(rng.normal(0, 0.3))
        vol_d = rng.uniform(0.015, 0.04)
        drift = rng.normal(0.0005, 0.001)
        for t in range(n_days):
            gap = rng.normal(0, vol_d * 0.6)
            if rng.random() < 0.02:
                gap += rng.choice([-1, 1]) * rng.uniform(0.05, 0.12)     # 큰 갭
            if rng.random() < 0.004:
                gap = 0.29                                               # 상한가 근처 시가
            o = px * np.exp(gap)
            c = o * np.exp(drift + rng.normal(0, vol_d))
            h = max(o, c) * np.exp(abs(rng.normal(0, vol_d * 0.7)))
            lo = min(o, c) * np.exp(-abs(rng.normal(0, vol_d * 0.7)))
            OP[t, j], H[t, j], L[t, j], C[t, j] = o, h, lo, c
            V[t, j] = rng.integers(1_000, 100_000)
            px = c
    # 정지일: 거래량 0, OHLC = 직전 종가 이월
    halt = rng.random((n_days, n_codes)) < 0.01
    halt[0] = False
    for t, j in zip(*np.nonzero(halt)):
        OP[t, j] = H[t, j] = L[t, j] = C[t, j] = C[t - 1, j]
        V[t, j] = 0
    # 상장폐지 3종목(패널 끝 전에 시세가 끊김), 시세 없음 구간 1종목
    for j, cut in ((1, 400), (2, 520), (3, 650)):
        OP[cut:, j] = H[cut:, j] = L[cut:, j] = C[cut:, j] = np.nan
        V[cut:, j] = np.nan
    OP[300:310, 4] = np.nan                                               # 시가 결측 = 체결 불가
    frame = {k: pd.DataFrame(v, index=days, columns=codes) for k, v in
             {"open": OP, "high": H, "low": L, "close": C, "volume": V}.items()}
    ev_mask = rng.random((n_days, n_codes)) < 0.03
    ev_mask[:60] = False                                                 # 피벗·수축 저점 이력
    for j, cut in ((1, 400), (2, 520), (3, 650)):
        ev_mask[cut:, j] = False
    return frame, ev_mask


def _panels(frame: dict, ev_mask: np.ndarray) -> dict:
    c = frame["close"]
    ev = pd.DataFrame(ev_mask, index=c.index, columns=c.columns)
    piv = pivot_level(frame["high"], 50)
    ok = ev & piv.notna()
    const = lambda v: pd.DataFrame(v, index=c.index, columns=c.columns)  # noqa: E731
    return {
        **frame,
        "market": pd.Series(["KOSPI" if i % 2 else "KOSDAQ" for i in range(len(c.columns))], index=c.columns),
        "ma50": moving_average(c, 50),
        "universe": ok, "tt": ok, "breakout": ok,
        "rs": const(80.0), "breadth": const(0.6), "pead_rank": const(0.8), "rel": const(0.5), "F": const(1.0),
        "adv20": const(10_000.0),
        "pivot": piv,
        "contraction_low": contraction_low_level(frame["low"], 10),
    }


def _research(frame: dict, panels: dict, proto: dict):
    ok = panels["universe"].to_numpy()
    di, ci = np.nonzero(ok)
    c = frame["close"]
    events = pd.DataFrame({"code": c.columns[ci], "event_day": c.index[di],
                           "market": panels["market"].reindex(c.columns[ci]).to_numpy(),
                           "pivot": panels["pivot"].to_numpy()[di, ci]})
    return simulate_minervini(frame["open"], frame["high"], frame["low"], c, panels["ma50"], events,
                              volume=frame["volume"], return_skipped=True, **proto)


@pytest.mark.parametrize("proto", list(PROTOCOLS))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_unlimited_backtest_matches_research_simulator(seed, proto):
    frame, ev_mask = _synthetic(seed)
    panels = _panels(frame, ev_mask)
    cfg = StrategyConfig(**PROTOCOLS[proto])
    res = run_backtest(panels, cfg, unlimited=True)
    rt, rs = _research(frame, panels, PROTOCOLS[proto])

    lt = res.trades
    below = lt[lt["exit_reason"] == "below_low"]
    lt = lt[lt["exit_reason"] != "below_low"]
    n_below_research = int((rs["skip_reason"] == "skip_below_low").sum())
    assert len(below) == n_below_research

    key = ["code", "event_day"]
    a = lt.set_index(key).sort_index()
    b = rt.set_index(key).sort_index()
    assert list(a.index) == list(b.index), (len(a), len(b))
    assert (a["entry_day"] == b["entry_day"]).all()
    assert (a["exit_day"] == b["exit_day"]).all()
    assert (a["exit_reason"] == b["exit_reason"]).all()
    np.testing.assert_allclose(a["ret"].to_numpy(float), b["ret"].to_numpy(float), rtol=0, atol=1e-9)
    np.testing.assert_allclose(a["entry_px"].to_numpy(float), b["entry_px"].to_numpy(float), rtol=0, atol=1e-9)
    np.testing.assert_allclose(a["d"].to_numpy(float), b["d"].to_numpy(float), rtol=0, atol=1e-12)
    assert (a["partial_hit"].astype(bool) == b["partial_hit"].astype(bool)).all()
    assert (a["hold_days"].astype(int) == b["hold_days"].astype(int)).all()

    # 체결 실패 건수도 같다(추격·상한가·정지)
    for k in ("skip_no_open", "skip_chase", "skip_limit_up"):
        assert res.skips[k] == int((rs["skip_reason"] == k).sum()), k
    assert res.skips["skip_held"] == int((rs["skip_reason"] == "skip_overlap").sum())
    # 분기를 실제로 지났는가(합성이 너무 순하면 패리티가 공허하다)
    reasons = set(b["exit_reason"])
    need = {"stop", "breakeven", "trail", "data_end"}
    # 돌파 실패가 켜지면 합성 랜덤워크에서 120일·폐지까지 살아남는 포지션이 거의 없다
    need |= {"failed_breakout"} if proto == "new" else {"time_stop", "time", "delisted"}
    assert need <= reasons, reasons
    assert len(b) > 100


# ---------------------------------------------------------------------------
# 라이브 이벤트 경로(on_open → on_quote → on_close) vs 일봉 경로(on_bar)
# ---------------------------------------------------------------------------

def _walk(pos_bar, pos_live, o, h, lo, c, ma, tr, cut, *, order="pessimistic"):
    """같은 봉 시퀀스를 두 상태기계에 먹인다. ``order`` = 장중 현재가 순서: pessimistic(시가 → 저가 → 고가 → 종가) /
    optimistic(시가 → 고가 → 저가 → 종가).

    Returns: (일봉 청산 [(day_no, kind)], 라이브 청산 [(day_no, kind)], 일봉 총수익, 라이브 총수익, 같은 날 익절+손절 날 수).
    """
    same_day = 0
    for k in range(len(o)):
        day_no = k + 1
        if k == 0:
            pos_bar.on_fill_open(o[0], day_no=1)
            pos_live.on_fill_open(o[0], day_no=1)
        if not tr[k]:
            pos_bar.on_bar(o[k], h[k], lo[k], c[k], ma[k], day_no, tradable=False)
            continue
        if pos_bar.phase == "full" and max(o[k], h[k]) >= pos_bar.target_px and lo[k] <= pos_bar.stop_px:
            same_day += 1
        pos_bar.on_bar(o[k], h[k], lo[k], c[k], ma[k], day_no)
        if k > 0:
            pos_live.on_open(o[k], day_no)
        first, second = (lo[k], h[k]) if order == "pessimistic" else (h[k], lo[k])
        pos_live.on_quote(first)
        pos_live.on_quote(second)
        pos_live.on_close(c[k], ma[k], day_no)
        if not pos_bar.is_open and not pos_live.is_open:
            break
        if k >= cut:
            break
    ex = lambda p: [(x["day_no"], x["kind"]) for x in p.exits]  # noqa: E731
    return ex(pos_bar), ex(pos_live), pos_bar.gross_return(), pos_live.gross_return(), same_day


def _positions(seed):
    frame, _ = _synthetic(seed, n_days=400, n_codes=30)
    ma = moving_average(frame["close"], 50)
    c_low = contraction_low_level(frame["low"], 10)
    piv = pivot_level(frame["high"], 50)
    OP, H, L, C, V, M, CL, PV = (x.to_numpy() for x in (frame["open"], frame["high"], frame["low"], frame["close"],
                                                        frame["volume"], ma, c_low, piv))
    for j in range(OP.shape[1]):
        for e in range(60, 300, 7):
            if not (np.isfinite(OP[e, j]) and V[e, j] > 0 and np.isfinite(CL[e - 1, j]) and OP[e, j] > CL[e - 1, j]):
                continue
            sl = slice(e, None)
            tr = np.isfinite(C[sl, j]) & np.isfinite(OP[sl, j]) & (V[sl, j] > 0)
            if not tr.all():
                last = int(np.argmin(np.isfinite(C[sl, j]))) if not np.isfinite(C[sl, j]).all() else len(tr)
                tr = tr[:last]
            n = len(tr)
            yield (j, e, (float(CL[e - 1, j]), float(PV[e - 1, j])), OP[sl, j][:n], H[sl, j][:n], L[sl, j][:n],
                   C[sl, j][:n], M[sl, j][:n], tr)


def _pair(lows, cfg):
    c_low, pivot = lows
    a = PositionState(code="X", contraction_low=c_low, qty=1.0, fractional=True, pivot=pivot, cfg=cfg)
    b = PositionState(code="X", contraction_low=c_low, qty=1.0, fractional=True, pivot=pivot, cfg=cfg)
    return a, b


def _stress(lows, o, h, lo, c, k, cfg, *, gap):
    """``k`` 일째(0 기준) 봉을 "익절 + 손절 같은 날" 로 바꾼다(``gap`` = 시가 ≥ 목표). 진입가 = o[0] 로 연구 식대로."""
    c_low = lows[0]
    o, h, lo, c = (x.astype(float).copy() for x in (o, h, lo, c))
    entry = o[0]
    d = min(max(1.0 - c_low * cfg.stop_mult / entry, cfg.stop_floor), cfg.stop_cap)
    stop, tgt = entry * (1 - d), entry * (1 + cfg.take_r * d)
    o[k] = tgt * 1.01 if gap else entry * 1.001
    h[k] = max(o[k], tgt) * 1.01
    lo[k] = stop * 0.99
    c[k] = stop * 1.001
    return o, h, lo, c


@pytest.mark.parametrize("proto", list(PROTOCOLS))
@pytest.mark.parametrize("seed", [0, 1])
def test_live_event_path_matches_daily_bar_path(seed, proto):
    """비관 순서(저가 먼저)에선 라이브 = 일봉: 같은 청산(날짜·종류). 라이브 가격은 손절가·목표가를 밟은 현재가라 수익은
    비교하지 않는다. 같은 날 익절+손절 날(자연 발생 + 2~4일째 주입)도 **제외하지 않는다**:

    - 장중 익절(시가 < 목표): 저가가 먼저 오면 두 경로 다 전량 손절 → 같다.
    - 갭 상승 익절(시가 ≥ 목표) + 손절 터치: 시가가 먼저인 건 두 경로 다 안다. 일봉은 "같은 날 둘 다면 손절 먼저" 로 전량
      손절, 라이브는 시가에 절반 익절 뒤 본전(진입가, 그날 활성 — 손절가보다 위)에 나머지. **문서화된 차이** — 모양을 그대로
      단언한다(일봉 ``[stop]`` vs 라이브 ``[partial, breakeven]`` 같은 날).
    """
    cfg = StrategyConfig(**PROTOCOLS[proto])
    n_cmp = n_same = n_gap = 0
    kinds = set()
    for j, e, c_low, o, h, lo, c, ma, tr in _positions(seed):
        variants = [(o, h, lo, c, False)]
        if len(tr) > 4 and tr[1:4].all():
            variants.append((*_stress(c_low, o, h, lo, c, 1 + (j % 3), cfg, gap=bool(j % 2)), bool(j % 2)))
        for oo, hh, ll, cc, gap in variants:
            a, b = _pair(c_low, cfg)
            bar, live, _, _, same = _walk(a, b, oo, hh, ll, cc, ma, tr, cut=len(tr) - 1)
            n_cmp += 1
            n_same += same > 0
            kinds |= {k for _, k in bar}
            if bar == live:
                continue
            # 주입한 갭 익절일까지 살아 있었던 경우만 모양이 다르다(문서화된 차이)
            assert gap and same, (j, e, bar, live)
            n_gap += 1
            k = bar[-1][0]
            assert bar == [(k, "stop")] and live == [(k, "partial"), (k, "breakeven")], (j, e, bar, live)
            assert b.exits[-1]["price"] <= b.entry_px            # 라이브 가격 = 터치한 현재가(판단 기준가)
    print(f"[{proto}] compared {n_cmp}, same-day partial+stop {n_same} (gap-up documented {n_gap}), kinds {sorted(kinds)}")
    # new 프로토콜은 돌파 실패로 일찍 나가 주입한 2~4일째 봉까지 살아남는 포지션이 적다
    assert n_cmp > 400 and n_same > 100 and n_gap > (30 if proto == "old" else 15)
    need = {"stop", "partial", "breakeven", "trail_next_open"}
    need |= {"failed_breakout_next_open"} if proto == "new" else {"time_close", "max_hold"}
    assert need <= kinds, kinds


@pytest.mark.parametrize("proto", list(PROTOCOLS))
@pytest.mark.parametrize("seed", [0, 1])
def test_live_partial_then_stop_same_day_closes_with_stop(seed, proto):
    """REVIEW_BACKTEST M1 — 낙관 순서(고가 먼저)로 "익절 + 손절 같은 날" 봉(자연 발생 + 주입: 장중 익절 / 갭 상승 익절).
    라이브 = 절반 익절 + 나머지 절반 **손절**(원래 손절가, 같은 날; 갭 익절일에 본전이 먼저 걸리면 breakeven), 일봉 = 전량 손절.
    측정: 두 경로 다 그날 닫히고, 일봉 사유 stop, 라이브 마지막 사유 stop/breakeven, 라이브 수익 ≥ 일봉 수익. 크기를 찍는다."""
    cfg = StrategyConfig(**PROTOCOLS[proto])
    n_case = n_gap = n_plain = 0
    diffs = []
    for j, e, c_low, o, h, lo, c, ma, tr in _positions(seed):
        variants = [(o, h, lo, c)]
        if len(tr) > 4 and tr[1:4].all():
            variants.append(_stress(c_low, o, h, lo, c, 1 + (j % 3), cfg, gap=bool(j % 2)))
        for oo, hh, ll, cc in variants:
            a, b = _pair(c_low, cfg)
            bar, live, r_bar, r_live, same = _walk(a, b, oo, hh, ll, cc, ma, tr, cut=len(tr) - 1, order="optimistic")
            if same == 0 or bar == live:
                # 그 사례가 없거나(또는 전날 결정된 시가 청산이 먼저라) 순서와 무관하게 같다
                assert bar == live, (j, e, bar, live)
                n_plain += 1
                continue
            n_case += 1
            assert a.phase == b.phase == "closed", (j, e)
            assert bar[-1] == (live[-1][0], "stop"), (j, e, bar, live)         # 같은 날, 일봉 = 전량 손절
            if len(live) == 1:                                                  # 시가 ≤ 손절: 둘 다 시가 손절
                assert live == bar
                continue
            assert live[0][1] == "partial" and live[1][1] in ("stop", "breakeven") and len(live) == 2, live
            n_gap += oo[live[0][0] - 1] >= a.target_px
            assert r_live >= r_bar - 1e-12, (j, e, r_live, r_bar)
            diffs.append(r_live - r_bar)
    print(f"[{proto}] same-day partial+stop cases {n_case} (gap-up {n_gap}, plain {n_plain}), "
          f"mean live−bar ret {np.mean(diffs):+.4f}, max {np.max(diffs):+.4f}")
    assert n_case > (100 if proto == "old" else 40) and n_gap > (30 if proto == "old" else 10)


def test_live_half_phase_stop_regression_case():
    """리뷰의 구성 사례: 진입 100, d 8.9%; 2일째 시가 100 → 119 → 85 → 종가 90. 라이브 = 익절 50% @119 + 손절 50% @85."""
    cfg = StrategyConfig()
    c_low = (1 - 0.089) * 100 / 0.99
    a, b = _pair((c_low, 95.0), cfg)
    bar, live, r_bar, r_live, same = _walk(a, b, np.array([100.0, 100.0]), np.array([101.0, 119.0]),
                                           np.array([99.0, 85.0]), np.array([100.0, 90.0]), np.array([95.0, 95.0]),
                                           np.array([True, True]), cut=1, order="optimistic")
    assert same == 1
    assert bar == [(2, "stop")] and live == [(2, "partial"), (2, "stop")]
    assert b.exits[1]["price"] == 85.0 and b.exits[1]["qty"] == 0.5
    assert r_bar == pytest.approx(-0.089) and r_live == pytest.approx(0.5 * 0.19 + 0.5 * -0.15)
