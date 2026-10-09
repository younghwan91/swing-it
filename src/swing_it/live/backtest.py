"""스윙 라이브 포트폴리오 백테스트 — 일 루프, 현금·슬롯, 동시호가 체결 모형, 일봉 손절/익절 (설계 §3 ``backtest.py``).

**같은 코드:** 진입 후보(:func:`~swing_it.live.strategy.scan`)·주문 계획(:func:`~swing_it.live.strategy.plan_entry`)·
사이징(:func:`~swing_it.live.sizing.allocate`)·포지션 전이(:class:`~swing_it.live.strategy.PositionState.on_bar`)는 전부
라이브와 같은 함수다. 이 모듈은 날짜를 돌리고 현금을 셀 뿐이다.

하루(``t``)의 순서:

1. **장전 동시호가:** 전일 종가에 낸 지정가 계획 → 그날 거래 가능(시세 ∧ 거래량 > 0)하고 시가 ≤ 지정가 ∧ 시가 < 상한가
   기준이면 **시가 체결**(:meth:`EntryPlan.fills_at`). 아니면 취소(다음 날로 미루지 않는다). 체결 직후
   :meth:`PositionState.on_fill_open` (시가 ≤ 수축 저점이면 그 시가에 바로 청산 — ``below_low``), 이어서 같은 봉으로
   :meth:`PositionState.on_bar` (진입일 = 1일째).
2. **보유 포지션:** :meth:`PositionState.on_bar` (전일 추적 결정은 시가, 손절 먼저, 익절, 본전, 종가 판단). 보유일
   ``day_no = t − 진입행 + 1`` (정지일도 센다 — 연구와 같다). 정지일은 ``tradable=False``.
3. **상장폐지:** 패널 마지막 날에 시세가 없는 종목은 그 종목의 마지막 거래 가능일 종가에 남은 전부를 닫는다(``delisted``
   — 연구와 같은 처리. 실제로는 정리매매 가격이 더 나쁠 수 있다 = 낙관 쪽).
4. **평가:** 자산 = 현금 + Σ 보유 수량 × 그날 종가(없으면 마지막 종가).
5. **신호(``start ≤ t ≤ end``):** 그날 종가까지로 :func:`scan` → 보유 중 종목 제외 → :func:`plan_entry` (자산 = 그날 평가,
   기준가 = 그날 종가) → :func:`allocate` (빈 슬롯·현금 — 현금 예약 = 수량 × 지정가).

미끄러짐: ``cfg.stop_slippage_bps`` (기본 50bp) — stop·breakeven 매도가에 ``× (1 − bps/1e4)`` (라이브 60초 폴링은 터치
뒤에 판다). 트레이드 ``ret`` 은 미끄러짐 전(연구와 같은 정의), ``pnl_krw``·``net_ret`` 은 미끄러짐·비용 후. 패리티 모드는 0.

비용: 매도할 때마다 그 매도 금액 × ``cost_fn(매도일, 시장)`` (기본 ``korea.cost_rt`` — 왕복 비용률: 매도세 + 수수료 왕복 +
스프레드). 분할 청산이면 **매도한 부분마다** 그 부분 금액에 비용을 문다(익절 절반 + 나머지 절반 = 전량 한 번과 총비용이
같다, 가격 차이만큼만 다르다). 매수 쪽은 따로 물지 않는다(왕복 비용률에 들어 있다). 결제(T+2)·미수는 모형하지 않는다 —
매도 대금은 그날 바로 현금이 된다.

**패리티 모드(``unlimited=True``):** 자본·슬롯 무제한, 계획마다 소수 수량 1.0, 비용 0 — 연구
``simulate_minervini`` (겹침 금지)와 트레이드별로 같아야 한다(``tests/test_live_parity.py``).

데이터 끝까지 열린 포지션은 마지막 거래 가능일 종가로 닫는다(``data_end``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from swing_it.live.config import DEFAULT_STRATEGY, StrategyConfig
from swing_it.live.sizing import allocate
from swing_it.live.strategy import RESEARCH_REASON, PositionState, plan_entry, scan

TRADE_COLS = ["code", "market", "event_day", "entry_day", "exit_day", "entry_px", "exit_px", "ret", "exit_reason",
              "hold_days", "d", "stop_px", "target_px", "contraction_low", "partial_hit", "partial_day", "partial_px",
              "ret_half", "qty", "cost_krw", "slip_krw", "pnl_krw", "net_ret", "rel", "rs"]
SKIP_KEYS = ("skip_no_next_day", "skip_no_open", "skip_chase", "skip_limit_up", "skip_slots_cash", "skip_held",
             "skip_news_veto")


def default_cost(day, market) -> float:
    """``korea.cost_rt(매도일, 시장)`` — (날짜, 시장) 캐시."""
    from swing_it.features.korea import cost_rt  # noqa: PLC0415

    key = (pd.Timestamp(day).date(), market)
    if key not in _COST_CACHE:
        _COST_CACHE[key] = cost_rt(key[0], market if market in ("KOSPI", "KOSDAQ") else "KOSPI")
    return _COST_CACHE[key]


_COST_CACHE: dict = {}


@dataclass
class _Book:
    """날짜×종목 numpy 배열(행 = 날짜)."""

    days: pd.DatetimeIndex
    codes: pd.Index
    op: np.ndarray
    hi: np.ndarray
    lo: np.ndarray
    cl: np.ndarray
    ma: np.ndarray
    tradable: np.ndarray
    last_tradable: np.ndarray       # 종목별 마지막 거래 가능 행(-1 = 없음)
    present_end: np.ndarray         # 패널 마지막 날 종가가 있나

    @classmethod
    def of(cls, panels: dict) -> "_Book":
        c = panels["close"]

        def arr(x):
            return x.reindex(index=c.index, columns=c.columns).to_numpy(dtype=float)

        op, hi, lo, cl, ma, vol = (arr(panels[k]) for k in ("open", "high", "low", "close", "ma50", "volume"))
        present = np.isfinite(cl)
        tr = present & np.isfinite(op) & np.isfinite(lo) & (op > 0) & (vol > 0)
        n = len(c.index)
        last = np.where(tr.any(axis=0), n - 1 - np.argmax(tr[::-1], axis=0), -1)
        return cls(c.index, c.columns, op, hi, lo, cl, ma, tr, last, present[-1])


@dataclass
class _Open:
    pos: PositionState
    ci: int
    sig_t: int
    entry_t: int
    plan: object
    cost_krw: float = 0.0
    slip_krw: float = 0.0
    proceeds: float = 0.0
    last_px: float = math.nan
    partial_t: "int | None" = None


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily: pd.DataFrame              # date, equity, cash, invested, n_pos, buys, sells
    skips: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    vetoed: "pd.DataFrame | None" = None      # 뉴스 거부권으로 뺀 후보(가상 진입군)

    @property
    def equity(self) -> pd.Series:
        return self.daily["equity"]


def run_backtest(panels: dict, cfg: StrategyConfig = DEFAULT_STRATEGY, *, start=None, end=None,
                 initial_equity: float = 100_000_000.0, unlimited: bool = False,
                 cost_fn: "Callable | None" = default_cost, dd_kill: "float | None" = None,
                 kill_resume_days: int = 60, kill_resume_recovery: float = 0.05,
                 veto: "pd.DataFrame | None" = None) -> BacktestResult:
    """포트폴리오 백테스트(모듈 설명의 하루 순서).

    Args:
        panels: :func:`swing_it.live.features.build_signal_panels` 결과(전 기간). ``open/high/low/close/volume/ma50``
            + :data:`~swing_it.live.strategy.SCAN_KEYS` + ``market``.
        cfg: 전략 설정(``require_flow=False`` = F 없는 변형).
        start, end: **신호일** 구간(포함). 보유는 데이터 끝까지 이어진다. None = 패널 처음/끝.
        initial_equity: 시작 자산(원).
        unlimited: 패리티 모드 — 자본·슬롯 무제한, 소수 수량 1.0, 비용 0.
        cost_fn: ``(매도일, 시장) → 비용률``. None = 0.
        dd_kill: 주면 자산이 고점 대비 이 비율 이상 빠지면 신규 진입을 멈춘다(설계 낙폭 킬). 재개 =
            ``kill_resume_days`` 거래일 경과 **또는** 자산 ≥ 고점 × (1 − ``kill_resume_recovery``) 중 먼저 오는 것
            (:class:`swing_it.live.config.RiskConfig`); 재개 때 고점을 그때 자산으로 다시 잰다. 기본 None = 끔.
            ``params["kills"]`` 에 (킬일, 재개일) 목록.
        veto: 날짜×종목 bool 패널(뉴스 거부권, :func:`swing_it.live.news_veto.veto_panel_from_ledger`). ``cfg.news_veto``
            가 참이고 패널을 주면 신호일에 거부된 후보를 진입에서 뺀다(``skip_news_veto``, ``result.vetoed`` 에 (신호일,
            종목, rel, rs) 기록 — 가상 진입군). 패널 밖 (날짜, 종목)은 모름 = 거부 안 함.
    """
    b = _Book.of(panels)
    days, codes = b.days, b.codes
    n = len(days)
    t_lo = 0 if start is None else int(days.searchsorted(pd.Timestamp(start), side="left"))
    t_hi = n - 1 if end is None else int(days.searchsorted(pd.Timestamp(end), side="right")) - 1
    col_of = {c: i for i, c in enumerate(codes)}
    if unlimited:
        cost_fn = None
    slip = 0.0 if unlimited else cfg.stop_slippage_bps / 1e4

    cash = 0.0 if unlimited else float(initial_equity)
    opened: dict[str, _Open] = {}
    pending: list = []                       # (plan, sig_t)
    trades: list[dict] = []
    rows: list[dict] = []
    skips = {k: 0 for k in SKIP_KEYS}
    peak = -math.inf
    killed_on = None
    kill_t = None
    kills: list = []
    vetoed_log: list[dict] = []
    veto_arr = None
    if veto is not None and cfg.news_veto:
        veto_arr = veto.reindex(index=days, columns=codes).fillna(False).astype(bool).to_numpy()

    def sell(o: _Open, acts, t: int) -> float:
        nonlocal cash
        got = 0.0
        for a in acts:
            if a.price is None or a.qty <= 0:
                continue
            px = a.price * (1.0 - slip) if a.kind in ("stop", "breakeven") else a.price
            val = a.qty * px
            o.slip_krw += a.qty * (a.price - px)
            cst = val * (cost_fn(days[t], o.pos.market) if cost_fn is not None else 0.0)
            o.cost_krw += cst
            o.proceeds += val
            cash += val - cst
            got += val
            if a.kind == "partial":
                o.partial_t = t
        return got

    def close_trade(o: _Open, t_exit: int) -> None:
        p = o.pos
        last = p.last_exit()
        partial_px = next((x["price"] for x in p.exits if x["kind"] == "partial"), math.nan)
        # 연구 정의 수익(정수 수량 무관): 익절이 났으면 두 절반 평균, 아니면 마지막 청산가 — REVIEW_BACKTEST m2
        ret_half = (0.5 * (partial_px / p.entry_px - 1.0) + 0.5 * (last["price"] / p.entry_px - 1.0)
                    if p.partial_hit else last["price"] / p.entry_px - 1.0)
        invested = p.qty * p.entry_px
        pnl = o.proceeds - o.cost_krw - invested
        trades.append({
            "code": p.code, "market": p.market, "event_day": days[o.sig_t], "entry_day": days[o.entry_t],
            "exit_day": days[t_exit], "entry_px": p.entry_px, "exit_px": last["price"], "ret": p.gross_return(),
            "exit_reason": RESEARCH_REASON[last["kind"]], "hold_days": t_exit - o.sig_t, "d": p.d,
            "stop_px": p.stop_px, "target_px": p.target_px, "contraction_low": p.contraction_low,
            "partial_hit": p.partial_hit, "partial_day": days[o.partial_t] if o.partial_t is not None else pd.NaT,
            "partial_px": partial_px, "ret_half": ret_half,
            "qty": p.qty, "cost_krw": o.cost_krw, "slip_krw": o.slip_krw, "pnl_krw": pnl,
            "net_ret": pnl / invested if invested > 0 else math.nan, "rel": o.plan.rel, "rs": o.plan.rs})

    for t in range(t_lo, n):
        day = days[t]
        buys = sells = 0.0
        # 1. 장전 동시호가 체결
        for plan, sig_t in pending:
            ci = col_of[plan.code]
            if not b.tradable[t, ci]:
                skips["skip_no_open"] += 1
                continue
            op = b.op[t, ci]
            if not plan.fills_at(op):
                skips["skip_chase" if op > plan.limit_px else "skip_limit_up"] += 1
                continue
            pos = PositionState.from_plan(plan, cfg, fractional=unlimited)
            o = _Open(pos=pos, ci=ci, sig_t=sig_t, entry_t=t, plan=plan, last_px=op)
            cash -= plan.qty * op
            buys += plan.qty * op
            acts = pos.on_fill_open(op, day_no=1, entry_day=day)
            sells += sell(o, acts, t)
            if not pos.is_open:
                close_trade(o, t)
                continue
            opened[plan.code] = o
        pending = []
        # 2·3. 보유 포지션 일봉
        for code in list(opened):
            o = opened[code]
            ci, p = o.ci, o.pos
            day_no = t - o.entry_t + 1                  # 진입일 = 1, 정지일도 센다
            tr = bool(b.tradable[t, ci])
            acts = p.on_bar(b.op[t, ci], b.hi[t, ci], b.lo[t, ci], b.cl[t, ci], b.ma[t, ci], day_no, tradable=tr)
            sells += sell(o, acts, t)
            if p.is_open and tr and t == b.last_tradable[ci] and not b.present_end[ci]:
                sells += sell(o, p.force_close("delisted", b.cl[t, ci], day_no), t)
            if np.isfinite(b.cl[t, ci]):
                o.last_px = b.cl[t, ci]
            if not p.is_open:
                close_trade(o, t)
                del opened[code]
        # 4. 평가
        invested = sum(o.pos.qty_open * o.last_px for o in opened.values())
        equity = cash + invested
        peak = max(peak, equity)
        if dd_kill is not None and not unlimited:
            if killed_on is None and equity <= peak * (1.0 - dd_kill):
                killed_on, kill_t = day, t
                kills.append([str(day.date()), None])
            elif killed_on is not None and (t - kill_t >= kill_resume_days
                                            or equity >= peak * (1.0 - kill_resume_recovery)):
                kills[-1][1] = str(day.date())
                killed_on, kill_t, peak = None, None, equity       # 재개: 고점을 다시 잰다
        rows.append({"date": day, "equity": equity, "cash": cash, "invested": invested, "n_pos": len(opened),
                     "buys": buys, "sells": sells})
        # 5. 신호
        if t > t_hi:
            continue
        cands = scan(panels, day, cfg)
        if t == n - 1:
            skips["skip_no_next_day"] += len(cands)
            continue
        if killed_on is not None:
            continue
        held = set(opened)
        skips["skip_held"] += sum(c.code in held for c in cands)
        cands = [c for c in cands if c.code not in held]
        if veto_arr is not None:
            vet = [c for c in cands if veto_arr[t, col_of[c.code]]]
            if vet:
                skips["skip_news_veto"] += len(vet)
                vetoed_log.extend({"event_day": day, "code": c.code, "market": c.market, "pivot": c.pivot,
                                   "contraction_low": c.contraction_low, "rel": c.rel, "rs": c.rs} for c in vet)
                cands = [c for c in cands if not veto_arr[t, col_of[c.code]]]
        if not cands:
            continue
        if unlimited:
            plans = [plan_entry(c, 1.0, c.close, cfg, tick_grid=False, fractional=True) for c in cands]
            plans = allocate([_with_qty(p, 1.0) for p in plans], cash=math.inf, open_slots=math.inf,
                             held=held, fractional=True)
        else:
            plans = [plan_entry(c, equity, c.close, cfg, tick_grid=False) for c in cands]
            sized = [p for p in plans if p.qty >= 1]
            plans = allocate(sized, cash=cash, open_slots=cfg.max_positions - len(opened), held=held)
            skips["skip_slots_cash"] += len(cands) - len(plans)
        pending = [(p, t) for p in plans]

    # 데이터 끝: 열린 포지션 → 마지막 거래 가능일 종가(data_end)
    for o in list(opened.values()):
        lt = int(b.last_tradable[o.ci])
        lt = max(lt, o.entry_t)
        acts = o.pos.force_close("data_end", b.cl[lt, o.ci], lt - o.entry_t + 1)
        got = sell(o, acts, n - 1)
        close_trade(o, lt)
        rows[-1]["sells"] += got
    if opened and rows:
        rows[-1]["equity"] = cash
        rows[-1]["cash"] = cash
        rows[-1]["invested"] = 0.0
        rows[-1]["n_pos"] = 0

    daily = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["equity", "cash", "invested", "n_pos", "buys", "sells"])
    tdf = pd.DataFrame(trades, columns=TRADE_COLS)
    if not tdf.empty:
        tdf = tdf.sort_values(["event_day", "code"]).reset_index(drop=True)
    params = {"cfg": cfg.to_dict(), "start": None if start is None else str(pd.Timestamp(start).date()),
              "end": None if end is None else str(pd.Timestamp(end).date()), "initial_equity": initial_equity,
              "unlimited": unlimited, "dd_kill": dd_kill, "kills": kills,
              "kill_resume_days": kill_resume_days, "kill_resume_recovery": kill_resume_recovery}
    res = BacktestResult(trades=tdf, daily=daily, skips=skips, params=params)
    res.vetoed = pd.DataFrame(vetoed_log, columns=["event_day", "code", "market", "pivot", "contraction_low", "rel", "rs"])
    return res


def _with_qty(plan, qty: float):
    from dataclasses import replace  # noqa: PLC0415

    return replace(plan, qty=qty)


# ---------------------------------------------------------------------------
# 성과 지표
# ---------------------------------------------------------------------------

TRADING_DAYS = 252


def _monthly_sharpe(eq: pd.Series) -> float:
    """월말 자산의 월수익 Sharpe(√12, 무위험 0) — 일수익이 덩어리(현금 날 0)라 일 Sharpe 가 부푼다(REVIEW_BACKTEST m3)."""
    if len(eq) < 40:
        return math.nan
    m = eq.resample("ME").last().pct_change().dropna()
    return float(m.mean() / m.std(ddof=1) * math.sqrt(12)) if len(m) > 2 and m.std(ddof=1) > 0 else math.nan


def drawdown(equity: pd.Series) -> pd.Series:
    """고점 대비 낙폭(0 이하)."""
    return equity / equity.cummax() - 1.0


def perf_stats(daily: pd.DataFrame, trades: "pd.DataFrame | None" = None, *,
               bench: "pd.Series | None" = None) -> dict:
    """일 자산 → CAGR·연 변동성·Sharpe(무위험 0, 일수익 √252)·MDD·노출·회전율 + 트레이드 통계 + 지수 대비.

    - 노출 = 평균(보유 평가액 / 자산). 회전율(연) = (매수 + 매도) / 2 / 평균 자산 × 252 / 일수.
    - 트레이드: 순수익(``net_ret`` — 비용 포함, 수량 가중 아님) 기준 승률·평균 이익·평균 손실·이익/손실 비.
    - ``bench``: 같은 날짜의 지수 **레벨**(또는 누적 지수). 주면 지수 CAGR·MDD·변동성·상관·베타를 붙인다.
    """
    eq = daily["equity"].astype(float)
    r = eq.pct_change().dropna()
    n = len(r)
    years = n / TRADING_DAYS if n else math.nan
    out = {
        "start": str(eq.index[0].date()) if len(eq) else None,
        "end": str(eq.index[-1].date()) if len(eq) else None,
        "final_equity": float(eq.iloc[-1]) if len(eq) else math.nan,
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else math.nan,
        "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if n else math.nan,
        "vol": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 else math.nan,
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 and r.std(ddof=1) > 0
        else math.nan,
        "mdd": float(drawdown(eq).min()) if len(eq) else math.nan,
        "sharpe_monthly": _monthly_sharpe(eq),
        "exposure": float((daily["invested"] / daily["equity"]).mean()) if len(eq) else math.nan,
        "turnover": float(((daily["buys"] + daily["sells"]).sum() / 2.0) / eq.mean() / years) if n else math.nan,
        "avg_positions": float(daily["n_pos"].mean()) if len(eq) else math.nan,
    }
    dd = drawdown(eq)
    if len(dd):
        trough = dd.idxmin()
        out["mdd_trough"] = str(trough.date())
        out["mdd_peak"] = str(eq.loc[:trough].idxmax().date())
    yr = eq.groupby(eq.index.year).last()
    first = eq.iloc[0] if len(eq) else math.nan
    prev = pd.concat([pd.Series([first], index=[yr.index[0] - 1]), yr.iloc[:-1]]) if len(yr) else yr
    out["years"] = {int(y): float(v) for y, v in zip(yr.index, yr.to_numpy() / prev.to_numpy() - 1.0)}
    if trades is not None and len(trades):
        x = trades["net_ret"].astype(float)
        win, loss = x[x > 0], x[x < 0]
        out.update({
            "n_trades": int(len(x)), "win_rate": float((x > 0).mean()),
            "avg_win": float(win.mean()) if len(win) else math.nan,
            "avg_loss": float(loss.mean()) if len(loss) else math.nan,
            "gl_ratio": float(win.mean() / -loss.mean()) if len(win) and len(loss) else math.nan,
            "avg_net": float(x.mean()), "avg_gross": float(trades["ret"].mean()),
            "avg_gross_half": float(trades["ret_half"].mean()) if "ret_half" in trades else math.nan,
            "avg_hold": float(trades["hold_days"].mean()),
            "exits": trades["exit_reason"].value_counts().to_dict(),
            "pnl_krw": float(trades["pnl_krw"].sum()), "cost_krw": float(trades["cost_krw"].sum()),
            "slip_krw": float(trades["slip_krw"].sum()) if "slip_krw" in trades else 0.0,
        })
    else:
        out["n_trades"] = 0
    if bench is not None:
        bz = bench.reindex(eq.index).ffill().dropna()
        if len(bz) > 1:
            br = bz.pct_change().dropna()
            yrs = len(br) / TRADING_DAYS
            rr = r.reindex(br.index).dropna()
            br2 = br.reindex(rr.index)
            cov = np.cov(rr, br2) if len(rr) > 2 else np.full((2, 2), np.nan)
            by = bz.groupby(bz.index.year).last()
            bprev = pd.concat([pd.Series([bz.iloc[0]], index=[by.index[0] - 1]), by.iloc[:-1]])
            out["bench"] = {
                "total_return": float(bz.iloc[-1] / bz.iloc[0] - 1.0),
                "cagr": float((bz.iloc[-1] / bz.iloc[0]) ** (1.0 / yrs) - 1.0),
                "vol": float(br.std(ddof=1) * math.sqrt(TRADING_DAYS)),
                "sharpe": float(br.mean() / br.std(ddof=1) * math.sqrt(TRADING_DAYS)),
                "sharpe_monthly": _monthly_sharpe(bz),
                "mdd": float(drawdown(bz).min()),
                "corr": float(np.corrcoef(rr, br2)[0, 1]) if len(rr) > 2 else math.nan,
                "beta": float(cov[0, 1] / cov[1, 1]) if len(rr) > 2 and cov[1, 1] > 0 else math.nan,
                "years": {int(y): float(v) for y, v in zip(by.index, by.to_numpy() / bprev.to_numpy() - 1.0)},
            }
    return out
