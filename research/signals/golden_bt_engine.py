"""골든 임박 자리 포지션 백테스트 엔진 — 순수 함수(DB·파일 없음). 정본 규칙: ``research/logs/golden_bt/PREREG.md``.

한 설정(cfg)을 [start, end] 거래일 구간에서 일 단위로 돌린다. 판정은 t 종가, 체결은 t+1 시가.

``D`` 키(전부 (종목, 거래일) 배열은 C 순서, 정지일은 O·C 가 NaN — ma_quad MQ-2):
  days, codes, O, C, ma50, ma200, pool(사전 후보 bool), PG(P_g), PD(P_d)(NaN = 계산 불가), lu(거래일별 상한가 시가 배수),
  cost((시장, 거래일) 왕복 비용 = korea.cost_rt(그날, 시장)), mkt(종목별 시장 행), last_valid(선택 — 종목별 마지막 유한 종가 행).
``cfg`` 키: th_in, th_out, sizing("eq"|"prob"), exit("dead_p"|"ma200"), th_d(dead_p 일 때).

규칙(PREREG + 실행기 RULINGS):
- 진입: t 에 pool ∧ P_g ≥ θ_in, P_g 내림차순(같으면 종목 행 순), 슬롯 20(보유 = 매도 대기 포함) — 빈 슬롯만큼 t+1 시가 주문.
  같은 종목은 한 포지션. 청산 체결로 빈 슬롯은 그날 종가 판정부터 쓴다(같은 날 시가 진입 없음).
- 체결 불가: t+1 시가 NaN(정지), 시가 ≥ 전일 종가 × lu(상한가), 시가 ≤ 손절선(d_fill ≤ 0). 못 사면 그 주문은 버린다.
- 손절선 = 진입 판정일 MA50(고정, 올리지 않음). d_fill = 1 − 손절선/체결 시가.
- 비중: eq = min(5%, 0.25%/d_fill), prob = 그 값 × 판정일 P_g. prob 는 골든 전 판정일마다 목표 = 기준 × P_g(t) 가
  원가 비중(Σ 매수 금액 ÷ 진입 때 자산)보다 크면 차이 × 진입 때 자산을 다음 시가에 추가(같은 체결 조건 — 정정 GB-2, 가격 하락만으로는
  사지 않는다). 목표 ≤ 기준 ≤ 5%. 첫 매수 금액 = 비중 × 전날 종가 자산, 현금이 모자라면 남은 현금까지.
- 청산(판정 t → t+1 시가): ① 종가 < 손절선 ② 골든 전 P_g < θ_out(예측 붕괴) ③ 골든 뒤 dead_p: P_d ≥ θ_d / ma200: 종가 < MA200.
  골든 = 진입 뒤 판정일 종가 기준 MA50 > MA200(엄격) 이 처음 된 날 — 그날부터 ③ 이 적용되고 ② 와 추가 매수는 멈춘다.
  골든 판정은 정지일을 건너뛴 이평선(``ma50v``·``ma200v`` — 유효 종가 마지막 50·200 개 평균, 없으면 ``ma50``·``ma200``)으로 한다.
  정정 GB-1: 보유 중 그날 판정 신호(골든 전 P_g, 골든 뒤 dead_p 의 P_d·ma200 의 MA200)가 NaN 이면 — 골든 전은 보유(손절만),
  골든 뒤는 대체 청산 종가 < ``ma200v`` → 다음 시가(사유 ma200_fb). 그런 판정일 수를 거래별 ``nan_dec`` 로 센다.
  종가가 NaN(정지)인 날은 아무 판정도 하지 않는다.
- 매도 대기 중 정지(시가 NaN)면 다음 체결 가능일로 미룬다. 마지막 유한 종가 뒤(폐지)에는 마지막 가격으로 그날 청산(delist).
- 비용: 청산일 왕복 비용 × cost_mult 를 매도 대금에서 뺀다. 끝(end) 보유분은 end 종가(없으면 마지막 가격)·비용 포함으로 청산(end).
- R = 손익 ÷ (0.25% × 첫 매수 때 쓴 자산). 자산은 1.0 에서 시작, 보유는 종가로 평가(마지막 날만 비용 뺀 순가치).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS = 20
W_CAP = 0.05
RISK = 0.0025
REASONS = ("stop", "collapse", "dead", "ma200", "ma200_fb", "delist", "end")

TRADE_COLS = ("code", "ci", "dec_day", "entry_day", "exit_day", "entry_px", "exit_px", "stop", "d_fill", "base_w", "w0", "pg_entry",
              "n_adds", "cost_basis", "proceeds", "ret", "pnl", "e_ref", "R", "reason", "golden_day", "hold", "nan_dec", "t0", "t1", "fills")


def last_valid_index(C: np.ndarray) -> np.ndarray:
    """종목별 마지막 유한 종가의 행(없으면 −1)."""
    f = np.isfinite(C)
    n = C.shape[1]
    return np.where(f.any(axis=1), n - 1 - np.argmax(f[:, ::-1], axis=1), -1)


def valid_ma(C: np.ndarray, w: int) -> np.ndarray:
    """정지일(NaN)을 건너뛴 이동평균 — 종목별 유효 종가 마지막 ``w`` 개 평균(유효 종가 < w 면 NaN). 종가 NaN 칸은 NaN(정정 GB-1)."""
    C = np.asarray(C, dtype=np.float64)
    out = np.full(C.shape, np.nan)
    for i in range(C.shape[0]):
        f = np.flatnonzero(np.isfinite(C[i]))
        if len(f) >= w:
            cs = np.cumsum(np.r_[0.0, C[i, f]])
            out[i, f[w - 1 :]] = (cs[w:] - cs[:-w]) / w
    return out


def _fillable(D: dict, ci: int, t: int, stop: float) -> "tuple[bool, str]":
    o = D["O"][ci, t]
    if not (np.isfinite(o) and o > 0):
        return False, "halt"
    pc = D["C"][ci, t - 1]
    if np.isfinite(pc) and o >= pc * D["lu"][t]:
        return False, "limit_up"
    if not o > stop:
        return False, "below_stop"
    return True, ""


def simulate(D: dict, cfg: dict, cost_mult: float, start: int, end: int, entry_end: "int | None" = None):
    """설정 하나 → (거래 DataFrame, 일별 자산 Series[days[start..end]]). 체결일 ∈ [start, end], 판정일 ∈ [start−1, end−1].

    ``entry_end``(행, 포함)가 있으면 그 뒤 체결일의 새 진입은 없다(폴드 시험 — 보유는 규칙대로 end 까지).
    자산 Series 의 ``attrs``: expo(종가 노출 배열), skips(체결 불가 사유별 수), n_orders(진입 주문 수).
    """
    O, C, PG, PD = D["O"], D["C"], D["PG"], D["PD"]  # noqa: E741
    ma50, ma200, pool = D["ma50"], D["ma200"], D["pool"]
    g50, g200 = D.get("ma50v", ma50), D.get("ma200v", ma200)
    cost, mkt, days, codes = D["cost"], D["mkt"], D["days"], D["codes"]
    n_days = C.shape[1]
    if not (1 <= start <= end < n_days):
        raise ValueError(f"구간 [{start}, {end}] 이 패널(1..{n_days - 1}) 밖")
    lv = D["last_valid"] if "last_valid" in D else last_valid_index(C)
    ee = end if entry_end is None else min(int(entry_end), end)
    th_in, th_out = float(cfg["th_in"]), float(cfg["th_out"])
    prob = cfg["sizing"] == "prob"
    if cfg["sizing"] not in ("eq", "prob") or cfg["exit"] not in ("dead_p", "ma200"):
        raise ValueError(f"잘못된 설정 {cfg}")
    dead = cfg["exit"] == "dead_p"
    th_d = float(cfg["th_d"]) if dead else np.nan

    cash = 1.0
    pos: dict = {}
    pend_buy: list = []
    pend_sell: dict = {}
    pend_add: dict = {}
    trades: list = []
    skips = {"halt": 0, "limit_up": 0, "below_stop": 0, "no_cash": 0}
    n_orders = 0
    eq = np.empty(end - start + 1)
    expo = np.empty(end - start + 1)
    e_prev = 1.0

    def close(ci: int, px: float, t: int, why: str) -> None:
        nonlocal cash
        p = pos.pop(ci)
        pend_add.pop(ci, None)
        c_rt = float(cost[mkt[ci], t]) * cost_mult
        proceeds = p["qty"] * px * (1.0 - c_rt)
        cash += proceeds
        pnl = proceeds - p["cost_basis"]
        trades.append({"code": codes[ci], "ci": ci, "dec_day": days[p["dec"]], "entry_day": days[p["t0"]], "exit_day": days[t],
                       "entry_px": p["px0"], "exit_px": float(px), "stop": p["stop"], "d_fill": p["d_fill"], "base_w": p["base"],
                       "w0": p["w0"], "pg_entry": p["pg"], "n_adds": p["n_adds"], "cost_basis": p["cost_basis"], "proceeds": proceeds,
                       "ret": proceeds / p["cost_basis"] - 1.0, "pnl": pnl, "e_ref": p["e_ref"], "R": pnl / (RISK * p["e_ref"]),
                       "reason": why, "golden_day": days[p["golden_t"]] if p["golden_t"] >= 0 else pd.NaT, "hold": t - p["t0"],
                       "nan_dec": p["nan_dec"], "t0": p["t0"], "t1": t, "fills": list(p["fills"])})

    for t in range(start - 1, end + 1):
        if t >= start:
            # 1) 시가 체결: 매도 → 폐지 정리 → 추가 → 신규
            for ci in list(pend_sell):
                o = O[ci, t]
                if np.isfinite(o) and o > 0:
                    close(ci, float(o), t, pend_sell.pop(ci))
            for ci in [c for c in pos if t > lv[c]]:
                pend_sell.pop(ci, None)
                close(ci, pos[ci]["last_px"], t, "delist")
            for ci, aw in list(pend_add.items()):
                p = pos[ci]
                ok, _ = _fillable(D, ci, t, p["stop"])
                amt = min(aw * p["e_ref"], cash)                    # GB-2: 원가 비중 차이 × 진입 때 자산
                if ok and amt > 0:
                    p["qty"] += amt / O[ci, t]
                    p["cost_basis"] += amt
                    p["n_adds"] += 1
                    p["fills"].append((t, amt, float(O[ci, t])))
                    cash -= amt
            pend_add.clear()
            for b in pend_buy:
                ci = b["ci"]
                ok, why = _fillable(D, ci, t, b["stop"])
                if not ok:
                    skips[why] += 1
                    continue
                o = float(O[ci, t])
                d_fill = 1.0 - b["stop"] / o
                base = min(W_CAP, RISK / d_fill)
                w = base * (b["pg"] if prob else 1.0)
                amt = min(w * e_prev, cash)
                if amt <= 0:
                    skips["no_cash"] += 1
                    continue
                pos[ci] = {"qty": amt / o, "cost_basis": amt, "px0": o, "t0": t, "dec": b["dec"], "stop": b["stop"], "d_fill": d_fill,
                           "base": base, "w0": w, "pg": b["pg"], "e_ref": e_prev, "n_adds": 0, "golden_t": -1, "last_px": o,
                           "nan_dec": 0, "fills": [(t, amt, o)]}
                cash -= amt
            pend_buy = []
            # 2) 종가 평가
            mv = 0.0
            for ci, p in pos.items():
                c = C[ci, t]
                if np.isfinite(c):
                    p["last_px"] = float(c)
                mv += p["qty"] * p["last_px"]
            e = cash + mv
            eq[t - start] = e
            expo[t - start] = mv / e if e > 0 else 0.0
        else:
            e = 1.0
        if t >= end:
            break
        # 3) 종가 판정 → 다음 시가 주문
        for ci, p in pos.items():
            if ci in pend_sell:
                continue
            c = C[ci, t]
            if not np.isfinite(c):
                continue
            if p["golden_t"] < 0 and g50[ci, t] > g200[ci, t]:
                p["golden_t"] = t
            if c < p["stop"]:
                pend_sell[ci] = "stop"
            elif p["golden_t"] < 0:
                pg = PG[ci, t]
                if np.isfinite(pg):
                    if pg < th_out:
                        pend_sell[ci] = "collapse"
                    elif prob:
                        add = p["base"] * float(pg) - p["cost_basis"] / p["e_ref"]   # GB-2: 원가 비중 < 목표일 때만(불타기, 물타기 아님)
                        if add > 1e-12:
                            pend_add[ci] = add
                else:
                    p["nan_dec"] += 1                                  # GB-1: 골든 전 P_g NaN → 보유(손절만)
            else:
                sig = PD[ci, t] if dead else ma200[ci, t]
                if np.isfinite(sig):
                    if (dead and sig >= th_d) or (not dead and c < sig):
                        pend_sell[ci] = "dead" if dead else "ma200"
                else:
                    p["nan_dec"] += 1                                  # GB-1: 골든 뒤 신호 NaN → 정지일 건너뛴 MA200 대체
                    fb = g200[ci, t]
                    if np.isfinite(fb) and c < fb:
                        pend_sell[ci] = "ma200_fb"
        if t + 1 <= ee:
            free = SLOTS - len(pos)
            if free > 0:
                pg_col = PG[:, t]
                with np.errstate(invalid="ignore"):
                    cs = np.flatnonzero(pool[:, t] & (pg_col >= th_in))
                cs = np.array([c for c in cs if c not in pos], dtype=np.int64)
                if len(cs):
                    cs = cs[np.lexsort((cs, -pg_col[cs].astype(np.float64)))][:free]
                    for ci in cs:
                        pend_buy.append({"ci": int(ci), "dec": t, "stop": float(ma50[ci, t]), "pg": float(pg_col[ci])})
                        n_orders += 1
        e_prev = e
    for ci in list(pos):
        close(ci, pos[ci]["last_px"], end, "end")
    eq[-1] = cash
    out = pd.Series(eq, index=days[start : end + 1])
    out.attrs.update(expo=expo, skips=skips, n_orders=n_orders, base=1.0)          # base = 첫날 전 자산(첫날 수익 포함)
    tr = pd.DataFrame(trades, columns=list(TRADE_COLS))
    return tr, out


# ============================================================================================ 널 ② — 거래 단위 경로

def trade_path(D: dict, ci: int, dec: int, hold: int, mult_path, cost_mult: float, end: int) -> "dict | None":
    """널 ② 한 거래: 판정일 ``dec`` 의 종목 ``ci`` 를 dec+1 시가에 사고, 손절(엔진과 같은 고정 MA50) 또는 체결 뒤 ``hold`` 거래일째
    시가(정지면 다음 가능일)에 판다. 끝 ``end`` 보유분은 end 종가·비용. 자산 1 고정(거래 단위).

    정정 GB-2(C1): 비중 = 이 종목의 기준 min(5%, 0.25%/d_fill) × **짝 전략 거래의 원가 배수 경로** ``mult_path`` =
    [(체결 뒤 거래일 수, 누적 원가 ÷ (기준 × 진입 자산))] — 첫 원소(0, m0)가 진입, 뒤 원소는 같은 날짜 차이의 체결일에 원가가 기준 × m 이
    되도록 추가(그날 체결 불가·이미 청산·손절 대기면 그 추가는 없다 — 엔진의 실패한 추가와 같다). eq 는 [(0, 1.0)](현금 부족 체결이면 < 1).
    체결 불가(정지·상한가·시가 ≤ 손절선)면 None. 반환 R = 손익 ÷ 0.25%(자산 1).
    """
    O, C = D["O"], D["C"]  # noqa: E741
    t0 = dec + 1
    if t0 > end:
        return None
    stop = float(D["ma50"][ci, dec])
    ok, _ = _fillable(D, ci, t0, stop)
    if not ok:
        return None
    path = sorted((int(a), float(m)) for a, m in mult_path)
    if not path or path[0][0] != 0:
        raise ValueError(f"배수 경로는 (0, m0) 로 시작해야 한다: {mult_path}")
    lv = int(D["last_valid"][ci]) if "last_valid" in D else int(last_valid_index(C[ci : ci + 1])[0])
    o = float(O[ci, t0])
    base = min(W_CAP, RISK / (1.0 - stop / o))
    w = base * path[0][1]
    if w <= 0:
        return {"ci": ci, "dec": dec, "t0": t0, "t1": t0, "R": 0.0, "ret": 0.0, "reason": "zero_weight"}
    adds = {t0 + a: m for a, m in path[1:]}
    qty, basis, last = w / o, w, o
    sell_at = t0 + max(int(hold), 1)
    pend_stop = False
    t = t0
    while True:
        if t > t0:
            if t > lv:
                return _tp_out(D, ci, qty * last, basis, t, cost_mult, "delist", t0, dec)
            ot = O[ci, t]
            tradable = np.isfinite(ot) and ot > 0
            if tradable and (pend_stop or t >= sell_at):
                return _tp_out(D, ci, qty * float(ot), basis, t, cost_mult, "stop" if pend_stop else "hold", t0, dec)
            if t in adds and not pend_stop:
                amt = base * adds[t] - basis
                ok, _ = _fillable(D, ci, t, stop)
                if ok and amt > 1e-12:
                    qty += amt / float(ot)
                    basis += amt
        c = C[ci, t]
        if np.isfinite(c):
            last = float(c)
        if t >= end:
            return _tp_out(D, ci, qty * last, basis, end, cost_mult, "end", t0, dec)
        if np.isfinite(c) and not pend_stop and c < stop:
            pend_stop = True
        t += 1


def _tp_out(D, ci, value, basis, t, cost_mult, why, t0, dec) -> dict:
    c_rt = float(D["cost"][D["mkt"][ci], t]) * cost_mult
    proceeds = value * (1.0 - c_rt)
    return {"ci": ci, "dec": dec, "t0": t0, "t1": t, "R": (proceeds - basis) / RISK, "ret": proceeds / basis - 1.0, "reason": why}
