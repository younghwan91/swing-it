"""ML 수익률 예측 — 피처 F1~F15 (PREREG §3). 순수 함수: 패널 → code × date.

규칙: **열 t 의 값은 열 ≤ t 의 입력만 읽는다**(rolling·누적은 앞에서부터, 이벤트는 "알게 된 날" 다음 거래일부터).
``tests/test_ml_forecast_features.py`` 가 합성 패널을 잘라 다시 낸 값과 비트 대조한다. 0단계 (a) 는 실데이터로 같은 것을 본다.

구현 고정(PREREG 가 정하지 않은 곳)은 정정 ML-1 에 적었다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ml_forecast_common import ANN_AGE_CAP, EW_SIGNS, FEATURES, STYLE_FEATURES, TP_LOOKBACK_DAYS, rank_gauss

FLOW_WIN, FLOW_MIN = 20, 15
HI52_WIN, HI52_MIN = 252, 126
HOLD_FFILL = 5
ADV_S, ADV_L = 5, 60


def _roll(a: np.ndarray, win: int, minp: int, how: str) -> np.ndarray:
    df = pd.DataFrame(np.asarray(a, dtype=np.float64).T)
    r = df.rolling(win, min_periods=minp)
    return getattr(r, how)().to_numpy().T


def f_rev21(close: np.ndarray) -> np.ndarray:
    """F3 = C_t / C_{t−21} − 1."""
    out = np.full(close.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 21:] = close[:, 21:] / close[:, :-21] - 1.0
    return out


def f_hi52(close: np.ndarray) -> np.ndarray:
    """F4 = C_t / max(C_{t−251..t}), 창 안 유한값 ≥ 126."""
    mx = _roll(close, HI52_WIN, HI52_MIN, "max")
    with np.errstate(invalid="ignore", divide="ignore"):
        return close / mx


def f_flow20(net: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """F5·F8 = Σ_{t−19..t} 순매수(주) / 그때 알던 상장주식수. 창 안 관측 ≥ 15."""
    s = _roll(net, FLOW_WIN, FLOW_MIN, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / shares
    out[~(shares > 0)] = np.nan
    return out


def f_frgn_own(hold: np.ndarray) -> np.ndarray:
    """F6 = h(t−20) − h(t−60). 보유율은 최대 5거래일 앞으로 채운 값(F5 창 (t−19, t] 와 겹치지 않는다)."""
    h = pd.DataFrame(np.asarray(hold, dtype=np.float64).T).ffill(limit=HOLD_FFILL).to_numpy().T
    out = np.full(h.shape, np.nan)
    out[:, 60:] = h[:, 40:-20] - h[:, :-60]
    return out


def f_vol_shock(tv: np.ndarray) -> np.ndarray:
    """F9 = log(ADV5 / ADV60), 거래대금 단순평균(절반 이상 관측). 0 이면 NaN."""
    a5 = _roll(tv, ADV_S, 3, "mean")
    a60 = _roll(tv, ADV_L, 30, "mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.log(a5 / a60)
    out[~np.isfinite(out)] = np.nan
    return out


def _period_ord(p: str) -> int | None:
    m = len(p) == 6 and p[4] == "Q" and p[:4].isdigit() and p[5] in "1234"
    return int(p[:4]) * 4 + int(p[5]) - 1 if m else None


def f_ann_age(disc: pd.DataFrame, codes, dates: pd.DatetimeIndex) -> np.ndarray:
    """F2 = 최근 분기(알게 된 것 중 가장 늦은 분기)의 첫 공시 뒤 지난 거래일 수, 최대 120.

    분기 첫 공시일 = min(rcept_dt) (prelim·pl_change·periodic). 알게 된 날 = 그 다음 거래일 k, 그날 값 0.
    """
    N, T = len(codes), len(dates)
    out = np.full((N, T), np.nan)
    if disc.empty:
        return out
    d = disc.assign(o=disc["period"].astype(str).map(_period_ord)).dropna(subset=["o"])
    d = d.assign(k=dates.searchsorted(pd.to_datetime(d["first"]).to_numpy(), side="right"))
    d = d[d["k"] < T]
    cidx = pd.Index([str(c) for c in codes])
    tt = np.arange(T)
    for code, g in d.groupby("code", sort=False):
        i = cidx.get_indexer([str(code)])[0]
        if i < 0:
            continue
        g = g.sort_values(["k", "o"], kind="stable")
        best_o, k_best = -1, None
        ks, kb = [], []
        for k, o in zip(g["k"].to_numpy(), g["o"].to_numpy()):
            if o > best_o:
                best_o, k_best = o, int(k)
                ks.append(int(k))
                kb.append(k_best)
        # 구간 [ks[j], ks[j+1]) 에서 최근 분기 첫 공시의 k = kb[j]
        ks_arr = np.asarray(ks)
        j = np.searchsorted(ks_arr, tt, side="right") - 1
        has = j >= 0
        age = np.where(has, tt - np.asarray(kb)[np.clip(j, 0, None)], np.nan)
        out[i] = np.where(has, np.minimum(age, ANN_AGE_CAP), np.nan)
    return out


def f_tp_rev(rep: pd.DataFrame, codes, dates: pd.DatetimeIndex) -> np.ndarray:
    """F7 = 최근 90일(달력) 안 리포트마다 log(목표가 / 같은 증권사의 직전 목표가) 평균. 짝이 없으면 NaN.

    - 목표가 > 0 인 리포트만 쓴다. 같은 (종목, 증권사, 작성일) 은 research_id 가 큰 하나로 접는다.
    - 직전 = 같은 (종목, 증권사) 의 더 이른 작성일 리포트(이 표 처음부터). 작성일에 시각이 없어 다음 거래일부터 쓴다:
      t 에서 쓰는 리포트 = write_date ∈ [date(t) − 90일, date(t)).
    """
    N, T = len(codes), len(dates)
    out = np.full((N, T), np.nan)
    r = rep[pd.to_numeric(rep["goal_price"], errors="coerce") > 0].copy()
    if r.empty:
        return out
    r["code"] = r["code"].astype(str)
    r["write_date"] = pd.to_datetime(r["write_date"])
    r["goal_price"] = r["goal_price"].astype(np.float64)
    r = r.sort_values(["code", "broker", "write_date", "research_id"], kind="stable")
    r = r.drop_duplicates(["code", "broker", "write_date"], keep="last")
    prev = r.groupby(["code", "broker"], sort=False)["goal_price"].shift(1)
    r["lr"] = np.log(r["goal_price"] / prev)
    r = r[np.isfinite(r["lr"])]
    cidx = pd.Index([str(c) for c in codes])
    day = dates.to_numpy("datetime64[ns]")
    lo_day = (dates - pd.Timedelta(days=TP_LOOKBACK_DAYS)).to_numpy("datetime64[ns]")
    for code, g in r.groupby("code", sort=False):
        i = cidx.get_indexer([code])[0]
        if i < 0:
            continue
        g = g.sort_values("write_date", kind="stable")
        wd = g["write_date"].to_numpy("datetime64[ns]")
        cs = np.r_[0.0, np.cumsum(g["lr"].to_numpy())]
        hi = np.searchsorted(wd, day, side="left")  # write_date < date(t)
        lo = np.searchsorted(wd, lo_day, side="left")  # write_date ≥ date(t) − 90일
        n = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)
    return out


def raw_features(P: dict) -> dict[str, np.ndarray]:
    """F1~F15 원값(변환 전), 열은 패널 전체 날짜."""
    codes, dates = P["codes"], P["dates"]
    out = {
        "eps_yoy": np.asarray(P["yoy"], dtype=np.float64),
        "ann_age": f_ann_age(P["disc"], codes, dates),
        "rev21": f_rev21(P["close"]),
        "hi52": f_hi52(P["close"]),
        "frgn20": f_flow20(P["fnet"], P["shares"]),
        "frgn_own": f_frgn_own(P["hold"]),
        "tp_rev": f_tp_rev(P["reports"], codes, dates),
        "inst20": f_flow20(P["inet"], P["shares"]),
        "vol_shock": f_vol_shock(P["tv"]),
    }
    for k in STYLE_FEATURES:
        out[k] = np.asarray(P["Xs"][k], dtype=np.float64)
    assert tuple(out) == FEATURES
    return out


def transform(raw: dict[str, np.ndarray], U: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """날짜 t → (|U(t)|, 15) rank-gauss 피처 행렬(U(t) 행 순서, NaN 유지)."""
    out = {}
    for t, idx in U.items():
        out[t] = np.column_stack([rank_gauss(raw[k][idx, t]) for k in FEATURES])
    return out


def ew_score(Z: np.ndarray) -> np.ndarray:
    """후보 E — 방향 피처 6개의 부호 붙인 rank-gauss 평균. 결측은 빼고, 다 결측이면 NaN."""
    cols = [FEATURES.index(k) for k in EW_SIGNS]
    s = np.array([EW_SIGNS[k] for k in EW_SIGNS], dtype=np.float64)
    V = Z[:, cols] * s[None, :]
    fin = np.isfinite(V)
    n = fin.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(fin, V, 0.0).sum(axis=1) / n, np.nan)


def weighted_score(Z: np.ndarray, w: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """N3 — 피처 15개의 부호·가중 rank-gauss 가중평균(결측은 E 처럼 빼고 남은 가중으로 나눈다)."""
    V = Z * sign[None, :]
    fin = np.isfinite(V)
    ww = np.where(fin, w[None, :], 0.0)
    den = ww.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.where(fin, V, 0.0) @ w / den, np.nan)


def coverage(raw: dict[str, np.ndarray], U: dict[int, np.ndarray], days: list[int]) -> dict[str, list[float]]:
    """피처별 날짜마다 U(t) 안 유한 비율."""
    return {k: [float(np.isfinite(raw[k][U[t], t]).mean()) if len(U[t]) else float("nan") for t in days]
            for k in FEATURES}
