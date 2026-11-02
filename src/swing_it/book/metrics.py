"""9단계 요약 — 스텝 순수익, IR, 손익비, 보유 구간 기여. 숫자만 낸다. 승률은 내지 않는다."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from swing_it.book.context import StepRec
from swing_it.book.optimized import FACTOR_ALARM

ANN_20 = math.sqrt(252.0 / 20)  # 20거래일 스텝의 연환산
LABEL_IC0 = "판단 불가(학습 IC 0)"  # IC_k 가 0 으로 잘린 폴드


def step_returns(recs: list[StepRec], mult: float) -> pd.DataFrame:
    df = pd.DataFrame({"t": [r.t for r in recs], "date": [r.date for r in recs],
                       "net": [(1.0 - mult * r.cost1) * (1.0 + r.gross) - 1.0 for r in recs],
                       "bench": [r.bench for r in recs]})
    df["a"] = df["net"] - df["bench"]
    return df


def oos(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    return df[(d >= window[0]) & (d <= window[1])].reset_index(drop=True)


def ir(a, ann: float = ANN_20) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / a.std(ddof=1) * ann)


def payoff_ratio(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos, neg = x[x > 0], x[x < 0]
    return float(pos.mean() / abs(neg.mean())) if pos.size and neg.size else float("nan")


def holding_intervals(recs: list[StepRec], mult: float) -> np.ndarray:
    """보유 구간(종목별 체결 뒤 비중 > 0 이 이어진 스텝) 순기여 = Σ(비중·종목 수익 − m·비용) (PO-6)."""
    cur: dict[int, float] = {}
    last: dict[int, int] = {}
    done = []
    for r in recs:
        seen = set()
        for i, w, g, c in r.contrib:
            if w > 0:
                if i in cur and last.get(i) != r.k - 1:
                    done.append(cur.pop(i))
                cur[i] = cur.get(i, 0.0) + w * g - mult * c
                last[i] = r.k
                seen.add(i)
            elif i in cur:  # 이번 체결에서 다 판 종목: 매도 비용까지 넣고 닫는다
                done.append(cur.pop(i) - mult * c)
        for i in [i for i in cur if i not in seen and last[i] != r.k]:
            done.append(cur.pop(i))
    done += list(cur.values())
    return np.array(done)


def mask_ic_zero_folds(summary: dict, years, label: str = LABEL_IC0) -> dict:
    """IC_k = 0 폴드는 폴드별 항목에서 빼고 라벨로 바꾼다(통합 항목은 그대로)."""
    years = {str(y) for y in years}
    for key in [k for k in summary if k.startswith("folds_IR_")]:
        summary[key] = {y: (label if y in years else v) for y, v in summary[key].items()}
    return summary


def book_summary(recs: list[StepRec], window: tuple[str, str], fold_years, *, ann: float = ANN_20) -> dict:
    """한 북의 평가 구간(``window``) 요약(비용 1·2배). ``fold_years`` 는 폴드별 IR 의 연도."""
    out = {}
    for m in (1.0, 2.0):
        df = oos(step_returns(recs, m), window)
        tag = f"{int(m)}x"
        out[f"IR_{tag}"] = ir(df["a"], ann)
        out[f"EV_{tag}"] = float(df["a"].mean())
        out[f"sd_a_{tag}"] = float(df["a"].std(ddof=1))
        out[f"step_payoff_{tag}"] = payoff_ratio(df["a"])
        hi = holding_intervals([r for r in recs if window[0] <= r.date <= window[1]], m)
        out[f"interval_payoff_{tag}"] = payoff_ratio(hi)
        srt = np.sort(hi)[::-1]
        out[f"interval_EV_drop_top5_{tag}"] = float(srt[5:].mean()) if srt.size > 5 else float("nan")
        out[f"interval_skew_{tag}"] = float(stats.skew(hi)) if hi.size > 2 else float("nan")
        out[f"folds_IR_{tag}"] = {str(y): ir(df.loc[pd.to_datetime(df["date"]).dt.year == y, "a"], ann) for y in fold_years}
        out[f"abs_net_mean_{tag}"] = float(df["net"].mean())
    o = [r for r in recs if window[0] <= r.date <= window[1]]
    out["oos_steps"] = len(o)
    out["optimal_share"] = float(np.mean([r.status == "optimal" for r in o])) if o else float("nan")
    out["lam_fail_share"] = float(np.mean([not r.lam_hit for r in o])) if o else float("nan")
    out["lam_up_share"] = float(np.mean([(not r.lam_hit) and r.lam_cause == "up" for r in o])) if o else float("nan")
    out["lam_fail_nonup_share"] = (float(np.mean([(not r.lam_hit) and r.lam_cause != "up" for r in o]))
                                   if o else float("nan"))
    fs = np.array([r.factor_share for r in recs], dtype=float)
    out["factor_share_q10_50_90"] = [float(np.nanquantile(fs, q)) for q in (0.1, 0.5, 0.9)] if np.isfinite(fs).any() else None
    alarm = [r for r in recs if np.isfinite(r.factor_share) and r.factor_share > FACTOR_ALARM]
    out["factor_alarm_n"], out["factor_alarm_dates"] = len(alarm), [r.date for r in alarm]
    out["factor_alarm_top3"] = [[r.date, r.alarm_top3] for r in alarm]
    out["lam_causes"] = {c: int(sum(r.lam_cause == c for r in o)) for c in ("up", "down", "gap", "solver")} if o else {}
    out["n_buy_limit_up"] = int(sum(r.n_buy_limit_up for r in recs))
    out["n_dust"] = int(sum(r.n_dust for r in recs))
    ex = np.array([r.exante_20 for r in o], dtype=float)
    out["exante_20_mean"] = float(np.nanmean(ex)) if o else float("nan")
    out["realized_over_exante"] = out["sd_a_1x"] / out["exante_20_mean"] if o else float("nan")
    out["turnover_mean"] = float(np.mean([r.turnover for r in o])) if o else float("nan")
    out["cost1_mean"] = float(np.mean([r.cost1 for r in o])) if o else float("nan")
    out["est_cost_mean"] = float(np.nanmean([r.est_cost for r in o])) if o else float("nan")
    out["names_mean"] = float(np.mean([r.n_names for r in o])) if o else float("nan")
    for key in ("n_forced", "n_delist", "n_frozen", "n_untradeable"):
        out[key] = int(sum(getattr(r, key) for r in recs))
    out["u_excl_mean"] = float(np.nanmean([r.u_excl for r in recs]))
    out["max_w_frozen_max"] = float(max((r.max_w_frozen for r in recs), default=0.0))  # 보고만
    return out
