"""결합 autoresearch 시대 캐시 — v4(prepare.build_era) 단계 + 야간 갭(scalp) + 뉴스(daytrade) 배열.

시대 A(2008~2016-08)·B(2016-09~2024)·확인 C(2016-09~2026-09-11). 탐색 절반(half 0)만 디스크에 쓴다.
``prepare.build_era`` 는 전역 ``ERAS`` 를 읽으므로 같은 단계를 ``ERAS_C`` 로 재현한다(prepare.py 는 바꾸지 않는다).

원주가: ``read_raw_close(aftermarket_fix=True)`` (indep30.load 는 raw_close 를 밖으로 내지 않는다), 빈 칸(시대 A 전부)은 v2 raw_close.
실제 시장(mkt_real): ``stocks`` + ``delisted_stocks`` 의 market (코스닥=1, 거래소·유가증권·코넥스=0). 엔진용 ``mkt`` 는 v4 처럼 0.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

from research.autoresearch.prepare import VIEW_KEYS as VIEW_KEYS_V4
from research.autoresearch.prepare import half_of, log, slice_half
from research.combined.gaps import gap_arrays
from research.combined.news import archive_dir, news_arrays

CACHE = os.path.expanduser("~/.cache/cc-swing/scratch/combined")


def _folds(y0: int, y1: int) -> list:
    return [(f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(y0, y1)]


ERAS_C = {
    "A": {"data": ("2008-01-01", "2016-08-31"), "eps": "earnings_dart_doc", "folds": _folds(2011, 2016)},
    "B": {"data": ("2016-09-01", "2024-12-31"), "eps": "earnings", "folds": _folds(2020, 2025)},
    "C": {"data": ("2016-09-01", "2026-09-11"), "eps": "earnings", "folds": [("2023-01-01", "2025-01-02", "2026-09-12")]},
}
GAP_KEYS = ("gap", "tickp", "gap_event")
NEWS_KEYS = ("n_all", "n_feat", "n_move", "n_excl", "news_known")
VIEW_KEYS_C = (*VIEW_KEYS_V4, *GAP_KEYS, *NEWS_KEYS)
REPO_KEYS = {"scalp": GAP_KEYS, "daytrade": NEWS_KEYS}   # swing 은 엔진·청산 메뉴·EPS 강제가 늘 쓰여 제거 검사 대상이 아니다


def view_c(D: dict) -> dict:
    """train.py 에 넘기는 묶음 — 복사본, 날짜·종목코드 없음."""
    V = {k: np.array(D[k], copy=True) for k in VIEW_KEYS_C}
    V.update(kx=np.array(D["kx"], copy=True), kq=np.array(D["kq"], copy=True), regime_on=np.array(D["regime_on"], copy=True))
    return V


MARKET_MAP = {"코스닥": 1, "거래소": 0, "유가증권": 0, "코넥스": 1}


def market_codes(m: pd.Series, codes: list) -> "tuple[np.ndarray, list]":
    """code→market 문자열 표 → (int8[N], 시장 없음·ETF 등 미분류 코드 목록). 표 밖 문자열은 오류, 없음은 0(코스피 표)."""
    m = m.reindex(codes)
    bad = sorted(set(m.dropna().astype(str)) - set(MARKET_MAP))
    if bad:
        raise ValueError(f"예상 밖 market 문자열: {bad}")
    miss = m.isna().to_numpy()
    arr = m.map(MARKET_MAP).fillna(0).to_numpy(np.int8)
    return arr, [c for c, x in zip(codes, miss, strict=True) if x]


def _market_real(con, codes: list) -> "tuple[np.ndarray, list, int]":
    """종목별 실제(현재) 시장 int8 + 시장 없는 코드 + stocks/delisted_stocks 불일치 수."""
    mk = pd.read_sql_query("SELECT code, market FROM stocks", con)
    dk = pd.read_sql_query("SELECT code, market FROM delisted_stocks", con)
    both = mk.merge(dk, on="code")
    n_dis = int((both["market_x"] != both["market_y"]).sum())
    m = pd.concat([dk, mk]).drop_duplicates("code", keep="last").set_index("code")["market"]
    arr, miss = market_codes(m, codes)
    return arr, miss, n_dis


def fill_raw_with_adj(raw: np.ndarray, C: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """원주가 결측 칸을 수정 종가로 채운다(indep30 px_cap 과 같은 방식). 결측 여부가 미래 폐지를 알리는 표지가 되지 않게 한다."""
    miss = ~np.isfinite(raw)
    return np.where(miss, C, raw), miss


def check_tickp_leak(tickp: np.ndarray, uni: np.ndarray, delisted: np.ndarray, tol: float = 0.01) -> "tuple[float, float]":
    """uni 칸의 tickp NaN 몫이 폐지·생존 종목 사이에서 tol 이상 다르면 오류(NaN 이 폐지 표지가 되는 누수)."""
    nan = np.isnan(tickp)
    sh = [float(nan[g[:, None] & uni].mean()) if (g[:, None] & uni).any() else 0.0 for g in (delisted, ~delisted)]
    if abs(sh[0] - sh[1]) >= tol:
        raise ValueError(f"tickp NaN 몫이 폐지 {sh[0]:.4f} vs 생존 {sh[1]:.4f} — 폐지 표지 누수")
    return sh[0], sh[1]


def _raw_wide(con, data: tuple, days: pd.DatetimeIndex, codes: list) -> np.ndarray:
    from swing_it.krx_session import read_raw_close
    r = read_raw_close(con, start=data[0], end=data[1], aftermarket_fix=True)
    r["date"] = pd.to_datetime(r["date"])
    w = r.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days, columns=codes)
    raw = w.to_numpy(float).T
    # read_raw_close 는 2016-09-05 부터라 시대 A 는 비었다 — 그 칸은 v2 의 raw_close(겹치는 구간에서 99.9% 일치)로 채운다.
    # 자료가 모두 2026-09-14 이전이라 session="raw" 여도 애프터마켓 오염이 없다.
    from swing_it.storage import read_prices
    v = read_prices(con, cols=("code", "date", "raw_close"), table="daily_bars_adj_v2", start=data[0], end=data[1], parse_dates=True,
                    session="raw")
    v = v.pivot_table(index="date", columns="code", values="raw_close", aggfunc="last").reindex(index=days, columns=codes)
    return np.where(np.isfinite(raw), raw, v.to_numpy(float).T)


def build_era(con, era: str) -> dict:
    """prepare.build_era 단계 (ERAS_C) + 갭·뉴스 배열."""
    import research.experiments.indep30 as I30
    from swing_it.features.eps_asof import eps_ok, eps_yoy
    from swing_it.features.korea import cost_rt
    from swing_it.storage import read_index
    E = ERAS_C[era]
    D = I30.build(con, E["data"], E["eps"])
    days, codes = D["days"], D["codes"]
    D["lu"] = np.where(days < pd.Timestamp("2015-06-15"), 1.14, 1.29)
    tax = np.array([cost_rt(d, "KOSPI") for d in days])
    D["cost"], D["mkt"] = np.vstack([tax, tax]), np.zeros(len(codes), np.int64)
    ci, s = np.nonzero(D["uni"] & D["TR"])
    up, yoy = np.zeros(D["C"].shape, bool), np.full(D["C"].shape, np.nan, np.float32)
    up[ci, s] = eps_ok(codes, days, ci, s, con, source=E["eps"])
    yoy[ci, s] = eps_yoy(codes, days, ci, s, con, source=E["eps"])
    kq = read_index(con, "101", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,
                    proxy="autoresearch: 공식 KOSDAQ(팩터 회귀용)").set_index("date")["close"]
    keep = ("days", "codes", "O", "H", "L", "C", "TR", "cost", "mkt", "hi20", "adv", "uni", "cap", "tv", "lu", "regime_on", "kx", "diag")
    out = {k: D[k] for k in keep}
    out.update(eps_up=up, eps_yoy=yoy, half=np.array([half_of(c) for c in codes], np.int8), era=era,
               kq=pd.Series(kq).reindex(days).ffill().to_numpy(float))
    out["kx"] = pd.Series(out["kx"]).reindex(days).ffill().to_numpy(float)
    for k in ("adv", "cap", "tv"):
        out[k] = np.asarray(out[k], np.float32)
    # --- 결합 추가분
    mkt_real, miss, n_dis = _market_real(con, codes)
    raw, fb = fill_raw_with_adj(_raw_wide(con, E["data"], days, codes), D["C"])
    out["mkt_real"] = mkt_real
    out.update(gap_arrays(D["O"], D["C"], raw, D["reopen"], D["lu"], days, mkt_real))
    out.update(news_arrays(codes, days, archive_dir()))
    isdel = np.isin(codes, list(D["delisted"]))
    nd, na = check_tickp_leak(out["tickp"], D["uni"], isdel)
    uc = D["uni"].sum(1)
    out["diag"] = {**D["diag"], "eps_up_share": float(up[ci, s].mean()),
                   "tickp_adj_fallback_share_uni": float(fb[D["uni"]].mean()),
                   "tickp_nan_share_uni_delisted": nd, "tickp_nan_share_uni_alive": na,
                   "mkt_unmapped_uni_cells": {c: int(uc[codes.index(c)]) for c in miss if uc[codes.index(c)] > 0},
                   "mkt_stocks_vs_delisted_disagree": n_dis}
    return out


def _slice(D: dict, half: int) -> dict:
    out = slice_half(D, half)
    out["mkt_real"] = D["mkt_real"][D["half"] == half]
    return out


def diagnostics(D: dict) -> dict:
    """진단만 — 점수·수익률은 계산하지 않는다."""
    days = pd.DatetimeIndex(D["days"])
    uni = D["uni"]
    d = {"codes": len(D["codes"]), "days": len(days), "mkt_real_kosdaq_share": float(D["mkt_real"].mean()),
         "gap_nan_share_all": float(np.isnan(D["gap"]).mean()),
         "gap_nan_share_uni": float(np.isnan(D["gap"][uni]).mean()) if uni.any() else float("nan"),
         "tickp_nan_share_uni": float(np.isnan(D["tickp"][uni]).mean()) if uni.any() else float("nan"),
         "gap_event_share_uni": float(D["gap_event"][uni].mean()) if uni.any() else float("nan"),
         "reopen_cells": D["diag"].get("reopen_cells")}
    for k in ("tickp_adj_fallback_share_uni", "tickp_nan_share_uni_delisted", "tickp_nan_share_uni_alive",
              "mkt_unmapped_uni_cells", "mkt_stocks_vs_delisted_disagree"):
        d[k] = D["diag"].get(k)
    for y in (2023, 2024):
        m = np.asarray(days.year == y)[None, :] & uni
        if m.any():
            d[f"news_known_share_uni_{y}"] = float(D["news_known"][m].mean())
    return d


def build_all(eras: tuple = ("A", "B", "C")) -> None:
    import psycopg2
    os.makedirs(CACHE, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    for era in eras:
        F = build_era(con, era)
        for k, v in diagnostics(F).items():
            log(f"era {era} diag {k}: {v}")
        D = _slice(F, 0)                                    # 탐색 절반만 디스크에
        with open(os.path.join(CACHE, f"era_{era}_s0.pkl"), "wb") as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"era {era}: 탐색 절반 {len(D['codes'])} codes × {len(D['days'])} days")


def load(era: str, half: int = 0) -> dict:
    """half 0 = 디스크 캐시, half 1 = DB 에서 메모리로(잠금 절반 — select.py 만 부른다)."""
    if half == 0:
        with open(os.path.join(CACHE, f"era_{era}_s0.pkl"), "rb") as f:
            return pickle.load(f)
    import psycopg2
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return _slice(build_era(con, era), half)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--diag", action="store_true", help="캐시된 탐색 절반의 진단만 출력")
    a = ap.parse_args()
    if a.build:
        build_all()
    elif a.diag:
        for e in ERAS_C:
            for k, v in diagnostics(load(e)).items():
                print(f"era {e} {k}: {v}", file=sys.stderr)
