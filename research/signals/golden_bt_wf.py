"""골든 임박 백테스트 — 메뉴 48·이웃·학습 점수·고원 선택·널 ①②·바 ①~⑦·VOID·보고 열. 순수 함수(DB 없음).

정본: ``research/logs/golden_bt/PREREG.md``. 해석은 실행기 ``research/experiments/golden_bt_run.py`` 의 RULINGS 에 적는다.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from research.signals import golden_bt_engine as E

# --------------------------------------------------------------------------------------------- 메뉴

AXES = {
    "th_in": (0.3, 0.5, 0.7, 0.9),
    "th_out": (0.1, 0.2),
    "sizing": ("eq", "prob"),
    "exit": (("dead_p", 0.5), ("dead_p", 0.7), ("ma200", None)),
}
ORDERED = ("th_in",)            # 한 칸 = 이웃 값 하나. 나머지 축은 범주 — 다른 값 전부가 이웃(RULINGS["neighbors"])
COST_JUDGE = 2.0
MIN_TRADES = 100
N_NULL = 300


def menu() -> list[dict]:
    """48 설정, 순서 고정(θ_in × θ_out × sizing × exit — PREREG 표기 순)."""
    out = []
    for a, b, c, (ex, td) in itertools.product(*AXES.values()):
        out.append({"th_in": a, "th_out": b, "sizing": c, "exit": ex, "th_d": td})
    return out


def _coords() -> list[tuple]:
    return list(itertools.product(*(range(len(v)) for v in AXES.values())))


def neighbors(i: int) -> list[int]:
    """자기 + 한 축만 한 칸 다른 설정. θ_in 은 순서 축(±1), θ_out·sizing·exit 는 범주 축(다른 값 전부)."""
    co = _coords()
    me = co[i]
    out = [i]
    for j, c in enumerate(co):
        diff = [k for k in range(len(c)) if c[k] != me[k]]
        if len(diff) != 1:
            continue
        k = diff[0]
        name = list(AXES)[k]
        if name in ORDERED and abs(c[k] - me[k]) != 1:
            continue
        out.append(j)
    return out


def neighbor_medians(scores) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if len(s) != len(_coords()):
        raise ValueError("점수 수가 메뉴와 다르다")
    s = np.where(np.isfinite(s), s, -np.inf)
    return np.array([np.median(s[neighbors(i)]) for i in range(len(s))])


def select(scores) -> int:
    """이웃 중앙값 최대(동률이면 메뉴 순서 앞)."""
    return int(np.argmax(neighbor_medians(scores)))


# --------------------------------------------------------------------------------------------- 폴드

def day_index(days, date: str, side: str) -> int:
    """``side="first"`` = date 이상 첫 거래일, ``"last"`` = date 이하 마지막 거래일."""
    days = pd.DatetimeIndex(days)
    if side == "first":
        i = int(days.searchsorted(pd.Timestamp(date), side="left"))
    else:
        i = int(days.searchsorted(pd.Timestamp(date), side="right")) - 1
    if not (0 <= i < len(days)):
        raise ValueError(f"{date} 이 패널 밖")
    return i


def fold_windows(days, years, train_start: str, eval_end: str) -> list[dict]:
    """폴드 Y: 학습 = [train_start, Y−1 말], 시험 = 체결 [Y 첫날, eval_end], 새 진입은 Y 말까지(entry_end)."""
    s0, e_all = day_index(days, train_start, "first"), day_index(days, eval_end, "last")
    out = []
    for y in years:
        te = day_index(days, f"{y - 1}-12-31", "last")
        ts = day_index(days, f"{y}-01-01", "first")
        ee = day_index(days, f"{y}-12-31", "last")
        out.append({"year": int(y), "train": (s0, te), "test": (ts, e_all), "entry_end": min(ee, e_all)})
    return out


def r_sum(trades: pd.DataFrame) -> float:
    return float(trades["R"].sum()) if len(trades) else 0.0


def train_score(D: dict, cfg: dict, start: int, end: int, cost_mult: float = COST_JUDGE) -> float:
    """학습 창 [start, end] 만 시뮬: 끝난 거래 + end 종가 평가 보유분(비용 포함)의 투입 R 합. end 뒤 가격은 쓰지 않는다."""
    tr, _ = E.simulate(D, cfg, cost_mult, start, end)
    return r_sum(tr)


# --------------------------------------------------------------------------------------------- 널

def null1_folds(rsum_table, n: int = N_NULL, seed: int = 0) -> np.ndarray:
    """널 ①: 폴드마다 메뉴에서 무작위 설정(복원) — (설정, 폴드) R 합 표에서 n 번 뽑아 폴드 합."""
    T = np.asarray(rsum_table, dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, T.shape[0], size=(n, T.shape[1]))
    return T[pick, np.arange(T.shape[1])[None, :]].sum(axis=1)


def null2_rep(D: dict, plan: list, holds: np.ndarray, rng, cost_mult: float = COST_JUDGE) -> dict:
    """널 ② 한 번: ``plan`` = [(판정일 행, 진입 수, sizing, end 행)] — 그 날 같은 풀(pool ∧ P_g 유한)에서 무작위로 같은 수를 산다
    (그날 안 비복원, 못 사면 다음 후보로). 보유 기간은 ``holds`` 에서 복원 추출. 반환 R 합·체결 수·모자란 수."""
    pool, PG = D["pool"], D["PG"]
    total, n_fill, short = 0.0, 0, 0
    for dec, cnt, sizing, end in plan:
        with np.errstate(invalid="ignore"):
            cand = np.flatnonzero(pool[:, dec] & np.isfinite(PG[:, dec]))
        got = 0
        for ci in rng.permutation(cand):
            if got >= cnt:
                break
            r = E.trade_path(D, int(ci), int(dec), int(holds[rng.integers(0, len(holds))]), sizing, cost_mult, int(end))
            if r is None:
                continue
            total += r["R"]
            got += 1
        n_fill += got
        short += cnt - got
    return {"R_sum": total, "n_fill": n_fill, "n_short": short}


def null2_plan(trades: pd.DataFrame, days, end: int) -> list:
    """선택 설정 거래(체결된 것)의 판정일별 진입 수 → 널 ② 계획. ``trades`` 에 sizing 열."""
    if trades.empty:
        return []
    d = pd.DatetimeIndex(days)
    dec = d.get_indexer(pd.to_datetime(trades["dec_day"]))
    g = pd.DataFrame({"dec": dec, "sizing": trades["sizing"].to_numpy()}).groupby(["dec", "sizing"]).size()
    return [(int(a), int(c), str(b), int(end)) for (a, b), c in g.items()]


def null2_holds(trades: pd.DataFrame) -> "tuple[np.ndarray, str]":
    """보유 기간 분포 = 손절이 아닌 청산의 체결 간 거래일 수. 없으면 전 거래(표시)."""
    if trades.empty:
        return np.array([1]), "empty"
    h = trades.loc[trades["reason"] != "stop", "hold"].to_numpy(int)
    if len(h) == 0:
        return np.maximum(trades["hold"].to_numpy(int), 1), "all_trades_fallback"
    return np.maximum(h, 1), "non_stop"


# --------------------------------------------------------------------------------------------- 통계

def cluster_boot(R, cluster, n_boot: int = 2000, seed: int = 0) -> "tuple[float, float, float]":
    """거래 R 평균과 군집(진입 연월) 복원 추출 95% 구간(점, 2.5%, 97.5%) — ma_quad_calib.bss_ci 규약."""
    R = np.asarray(R, dtype=float)
    if len(R) == 0:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(np.asarray(cluster), return_inverse=True)
    g = int(inv.max()) + 1
    s = np.bincount(inv, weights=R, minlength=g)
    c = np.bincount(inv, minlength=g).astype(float)
    pick = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    boot = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(R.mean()), float(lo), float(hi)


def top_removed_mean(trades: pd.DataFrame, k: int = 5) -> "tuple[float, list]":
    """종목별 R 합 상위 k 종목의 거래를 빼고 R 평균(거래 제거 — 재시뮬 아님)."""
    if trades.empty:
        return float("nan"), []
    top = trades.groupby("code")["R"].sum().sort_values(ascending=False, kind="stable").index[:k].tolist()
    rest = trades.loc[~trades["code"].isin(top), "R"]
    return (float(rest.mean()) if len(rest) else float("nan")), top


def cagr(eq: pd.Series) -> float:
    x = eq / eq.iloc[0]
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
    return float(x.iloc[-1] ** (1 / yrs) - 1) if x.iloc[-1] > 0 else -1.0


def stitch(parts: list) -> pd.Series:
    """[(자산 Series, 노출 배열)] 를 이어 붙인다 — 조각마다 1.0 으로 맞추고 앞 조각 끝 값에서 잇는다. attrs expo 같이."""
    out, ex, base = [], [], 1.0
    for s, x in parts:
        if len(s) == 0:
            continue
        out.append(pd.Series(s.to_numpy(float) / float(s.iloc[0]) * base, index=s.index))     # attrs 없는 사본(조각 attrs 는 길이가 달라 concat 이 깨진다)
        ex.append(np.asarray(x, dtype=float))
        base = float(out[-1].iloc[-1])
    res = pd.concat(out)
    res.attrs["expo"] = np.concatenate(ex)
    return res


def expo_market(eq: pd.Series, cw_by_day: pd.Series) -> pd.Series:
    """노출 맞춘 시장: 전날 노출 × 시총 가중 시장 일수익(첫날 0) 복리."""
    expo = np.asarray(eq.attrs["expo"], dtype=float)
    r = cw_by_day.reindex(eq.index).fillna(0.0).to_numpy(float)
    g = np.r_[0.0, expo[:-1] * r[1:]]
    return pd.Series(np.cumprod(1 + g), index=eq.index)


def monthly(eq: pd.Series, kx: pd.Series, kq: pd.Series) -> pd.DataFrame:
    """월 수익 s(전략)·k(KOSPI)·q(KOSDAQ) — 첫 달 포함(시작 값을 전달 말일에 붙인다, prepare.fold_stats 규약)."""
    i0 = pd.DatetimeIndex([eq.index[0].to_period("M").to_timestamp() - pd.Timedelta(days=1)])
    k = kx.reindex(eq.index).ffill().to_numpy(float)
    q = kq.reindex(eq.index).ffill().to_numpy(float)
    m = pd.DataFrame({"s": np.r_[eq.iloc[0], eq.to_numpy()], "k": np.r_[k[0], k], "q": np.r_[q[0], q]}, index=i0.append(eq.index))
    return m.resample("ME").last().pct_change().dropna()


def factor_months_rev(D: dict) -> pd.DataFrame:
    """autoresearch.prepare.factor_months(MOM 12-1·SMB·LV) + 1개월 반전 REV(직전 달 수익 하위 30% − 상위 30%, 같은 규약)."""
    from research.autoresearch.prepare import factor_months

    F = factor_months(D)
    C = pd.DataFrame(D["C"].T, index=D["days"])
    me = C.resample("ME").last()
    r = me.pct_change()
    uni = pd.DataFrame(D["uni"].T, index=D["days"]).resample("ME").last().astype(bool)
    f = []
    for t in range(1, len(me)):
        x = r.iloc[t - 1].where(uni.iloc[t - 1])
        x = x[np.isfinite(x)]
        if len(x) < 30:
            f.append(np.nan)
            continue
        q1, q3 = x.quantile(0.3), x.quantile(0.7)
        f.append(np.nanmean(r.iloc[t][x.index[x <= q1]]) - np.nanmean(r.iloc[t][x.index[x >= q3]]))
    F["rev"] = pd.Series(f, index=me.index[1:])
    return F


FACTOR_COLS = ("k", "q", "mom", "smb", "lv", "rev")


def factor_alpha(m: pd.DataFrame, F: pd.DataFrame, lags: int = 3) -> dict:
    """월 전략 수익 ~ KOSPI + KOSDAQ + MOM + SMB + LV + REV — 절편, OLS t, NW(Bartlett lag 3) t(prepare.factor_t 와 같은 식)."""
    j = m.join(F, how="inner").dropna()
    n, p = len(j), len(FACTOR_COLS) + 1
    if n <= p + 2:
        return {"alpha": float("nan"), "t_ols": float("nan"), "t_nw": float("nan"), "n": n}
    y = j["s"].to_numpy()
    X = np.column_stack([np.ones(n), j[list(FACTOR_COLS)].to_numpy()])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.pinv(X.T @ X)
    se_ols = np.sqrt(e @ e / (n - p) * XtX_inv[0, 0])
    Sg = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, lags + 1):
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        Sg += (1 - lag / (lags + 1)) * (G + G.T)
    V = XtX_inv @ Sg @ XtX_inv
    return {"alpha": float(b[0]), "t_ols": float(b[0] / se_ols), "t_nw": float(b[0] / np.sqrt(V[0, 0])), "n": n}


def dsr(ms: np.ndarray, n_trials: int) -> dict:
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample

    return deflated_sharpe_from_sample(np.asarray(ms, dtype=float), int(n_trials))


# --------------------------------------------------------------------------------------------- 바

def _p95(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, 0.95)) if len(x) else float("nan")


def bar_nulls(rs: float, null1, null2) -> dict:
    a, b = _p95(null1), _p95(null2)
    return {"R_sum": rs, "null1_p95": a, "null2_p95": b, "pass": bool(np.isfinite(a) and np.isfinite(b) and rs > a and rs > b)}


def bars_main(trades: pd.DataFrame, fold_rsum: list, eq: pd.Series, mkt_eq: pd.Series, m: pd.DataFrame, F: pd.DataFrame,
              null1, null2, n_trials: int) -> dict:
    """주 시험 바 ①~⑦(폴드 밖·비용 2배) + VOID(체결 < 100). 리포터 — 판정 문자열을 만들지 않는다(불리언)."""
    n = int(len(trades))
    b = {"n_trades": n, "void": bool(n < MIN_TRADES)}
    b["b1_nulls"] = bar_nulls(r_sum(trades), null1, null2)
    mean, lo, hi = cluster_boot(trades["R"], _ym(trades["entry_day"])) if n else (np.nan, np.nan, np.nan)
    b["b2_cluster_boot"] = {"mean": mean, "lo": lo, "hi": hi, "pass": bool(np.isfinite(lo) and lo > 0)}
    t5, top = top_removed_mean(trades)
    b["b3_top5_removed"] = {"mean": t5, "removed": top, "pass": bool(np.isfinite(t5) and t5 > 0)}
    fa = factor_alpha(m, F)
    b["b4_factor_alpha"] = {**fa, "pass": bool(np.isfinite(fa["t_ols"]) and fa["t_ols"] > 2 and np.isfinite(fa["t_nw"]) and fa["t_nw"] > 2)}
    cs, cm = cagr(eq), cagr(mkt_eq)
    b["b5_cagr_vs_expo_market"] = {"cagr": cs, "expo_market_cagr": cm, "pass": bool(cs > cm)}
    pos = int(sum(x > 0 for x in fold_rsum))
    b["b6_folds"] = {"fold_R_sum": list(fold_rsum), "n_pos": pos, "pass": bool(pos >= 3)}
    d = dsr(m["s"].to_numpy(), n_trials)
    pd_ = d.get("prob_deflated_sharpe")
    b["b7_dsr"] = {**d, "pass": bool(pd_ is not None and np.isfinite(pd_) and pd_ > 0.9)}
    b["pass"] = bool(not b["void"] and all(b[k]["pass"] for k in b if k.startswith("b")))
    return b


def bars_rep(trades: pd.DataFrame, eq: pd.Series, mkt_eq: pd.Series, m: pd.DataFrame, F: pd.DataFrame, null1, null2,
             halves: tuple) -> dict:
    """재현 바 ①②④⑤ + 두 반기 R 평균 > 0, VOID(체결 < 100)."""
    n = int(len(trades))
    b = {"n_trades": n, "void": bool(n < MIN_TRADES)}
    b["b1_nulls"] = bar_nulls(r_sum(trades), null1, null2)
    mean, lo, hi = cluster_boot(trades["R"], _ym(trades["entry_day"])) if n else (np.nan, np.nan, np.nan)
    b["b2_cluster_boot"] = {"mean": mean, "lo": lo, "hi": hi, "pass": bool(np.isfinite(lo) and lo > 0)}
    fa = factor_alpha(m, F)
    b["b4_factor_alpha"] = {**fa, "pass": bool(np.isfinite(fa["t_ols"]) and fa["t_ols"] > 2 and np.isfinite(fa["t_nw"]) and fa["t_nw"] > 2)}
    cs, cm = cagr(eq), cagr(mkt_eq)
    b["b5_cagr_vs_expo_market"] = {"cagr": cs, "expo_market_cagr": cm, "pass": bool(cs > cm)}
    hv = []
    ed = pd.to_datetime(trades["entry_day"]) if n else pd.Series(dtype="datetime64[ns]")
    for lo_, hi_ in halves:
        sel = trades.loc[(ed >= pd.Timestamp(lo_)) & (ed < pd.Timestamp(hi_)), "R"] if n else pd.Series(dtype=float)
        hv.append({"lo": lo_, "hi": hi_, "n": int(len(sel)), "mean": float(sel.mean()) if len(sel) else float("nan")})
    b["b_halves"] = {"halves": hv, "pass": bool(all(np.isfinite(h["mean"]) and h["mean"] > 0 for h in hv))}
    b["pass"] = bool(not b["void"] and all(b[k]["pass"] for k in b if k.startswith("b")))
    return b


def _ym(s) -> np.ndarray:
    d = pd.DatetimeIndex(pd.to_datetime(s))
    return np.asarray(d.year * 100 + d.month)


# --------------------------------------------------------------------------------------------- 보고 열

D_BINS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, np.inf))


def _rstats(R) -> dict:
    R = np.asarray(R, dtype=float)
    if len(R) == 0:
        return {"n": 0}
    w, lo = R[R > 0], R[R <= 0]
    return {"n": int(len(R)), "R_sum": float(R.sum()), "R_mean": float(R.mean()), "win_share": float(len(w) / len(R)),
            "avg_win_R": float(w.mean()) if len(w) else float("nan"), "avg_loss_R": float(lo.mean()) if len(lo) else float("nan"),
            "payoff": float(w.mean() / abs(lo.mean())) if len(w) and len(lo) and lo.mean() < 0 else float("nan")}


def shakeout(D: dict, trades: pd.DataFrame, end: int, horizon: int = 20) -> dict:
    """털기율: 손절 청산 중 청산 체결 뒤 ``horizon`` 거래일 안(end 까지) 종가가 진입가 이상으로 돌아온 비율."""
    st = trades[trades["reason"] == "stop"]
    if st.empty:
        return {"n_stop": 0, "rate": float("nan"), "horizon": horizon}
    days = pd.DatetimeIndex(D["days"])
    t1 = days.get_indexer(pd.to_datetime(st["exit_day"]))
    hit = []
    for ci, a, px in zip(st["ci"].to_numpy(int), t1, st["entry_px"].to_numpy(float), strict=True):
        seg = D["C"][ci, a : min(a + horizon, end) + 1]
        hit.append(bool(np.nanmax(seg) >= px) if np.isfinite(seg).any() else False)
    return {"n_stop": int(len(st)), "rate": float(np.mean(hit)), "horizon": horizon}


def report(D: dict, trades: pd.DataFrame, end: int) -> dict:
    """손익비·기대값·체결 수·청산 사유별·털기율·d 구간별·연도별(진입 연도)."""
    out = {"all": _rstats(trades["R"]) if len(trades) else {"n": 0}}
    if trades.empty:
        return out
    out["by_reason"] = {r: _rstats(trades.loc[trades["reason"] == r, "R"]) for r in E.REASONS if (trades["reason"] == r).any()}
    out["shakeout"] = shakeout(D, trades, end)
    out["by_d_fill"] = {f"{a:.2f}-{b:.2f}": _rstats(trades.loc[(trades["d_fill"] > a) & (trades["d_fill"] <= b), "R"]) for a, b in D_BINS}
    yr = pd.to_datetime(trades["entry_day"]).dt.year
    out["by_entry_year"] = {int(y): _rstats(trades.loc[yr == y, "R"]) for y in sorted(yr.unique())}
    out["hold_days"] = {"median": float(trades["hold"].median()), "mean": float(trades["hold"].mean())}
    out["golden_share"] = float(trades["golden_day"].notna().mean())
    return out


def yearly_returns(eq: pd.Series) -> dict:
    out, prev = {}, float(eq.iloc[0])
    for y, g in eq.groupby(eq.index.year):
        out[int(y)] = float(g.iloc[-1] / prev - 1)
        prev = float(g.iloc[-1])
    return out
