"""결합 autoresearch 시대 캐시 — v4(prepare.build_era) 단계 + 야간 갭(scalp) + 뉴스(daytrade) 배열.

시대 A(2008~2016-08)·B(2016-09~2024)·확인 C(2016-09~2026-09-11). 탐색 절반(half 0)은 ``CACHE``(루프 격리가 A·B 만 다시 보인다),
잠금 절반(half 1)은 ``LOCKED_CACHE``(격리가 통째로 가린다 — select.py 잠금만 읽고 MANIFEST sha 로 대조, 정정 0 M4).
뉴스 구간(2023-03-15~)에서 아카이브 없는 종목·잘린 page_cap 종목은 구간 전체 uni 를 끈다(``apply_news_exclusion``, 정정 0 C1) —
루프·기준선·참고·선택·전진 모두 이 캐시를 쓴다.
``prepare.build_era`` 는 전역 ``ERAS`` 를 읽으므로 같은 단계를 ``ERAS_C`` 로 재현한다(prepare.py 는 바꾸지 않는다).

원주가: ``read_raw_close(aftermarket_fix=True)`` (indep30.load 는 raw_close 를 밖으로 내지 않는다), 빈 칸(시대 A 전부)은 v2 raw_close.
실제 시장(mkt_real): ``stocks`` + ``delisted_stocks`` 의 market (코스닥=1, 거래소·유가증권·코넥스=0). 엔진용 ``mkt`` 는 v4 처럼 0.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

from research.autoresearch.prepare import VIEW_KEYS as VIEW_KEYS_V4
from research.autoresearch.prepare import half_of, log, slice_half
from research.combined.gaps import gap_arrays, tick_dated_vec
from research.combined.news import archive_dir, check_news_leak, news_arrays, news_excluded, news_uni_off

CACHE = os.path.expanduser("~/.cache/cc-swing/scratch/combined")
LOCKED_CACHE = os.path.expanduser("~/.cache/cc-swing/scratch/combined_locked")    # 잠금 절반(격리가 가린다)
NEWS_EXCLUDE = True    # 뉴스 구간 종목 제외 규칙 스위치(사용자가 뉴스 부품을 빼기로 하면 False — 그때는 PREREG 정정으로)
TICKP_LEAK_TOL = {"nan": 0.01, "offgrid": 0.01, "ks": 0.15}
# ks 0.15: 원주가가 거의 다 있는 시대 B·C 의 폐지/생존 tickp KS 가 0.07~0.11(폐지 종목이 싸서 생기는 정상 차이, 2026-10-25 캐시 실측) — 그 위 여유.


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


def snap_to_tick(p: np.ndarray, days: pd.DatetimeIndex, mkt: np.ndarray) -> np.ndarray:
    """가격을 그날 호가 격자에 맞춘다: round(p/tick)·tick(최소 1틱). 구간 경계가 두 틱의 공배수라 결과는 자기 틱의 격자 위다."""
    p = np.asarray(p, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = tick_dated_vec(p, days, mkt)
        return np.maximum(np.round(p / tk), 1.0) * tk


def fill_raw_with_adj(raw: np.ndarray, C: np.ndarray, days: pd.DatetimeIndex, mkt: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """원주가 결측 칸을 수정 종가로 채우되 **그날 호가 격자에 맞춘 값**으로(indep30 px_cap 방식 + 격자). 결측 여부(NaN)도,
    격자 밖 가격(|1/tickp − 정수| > 0)도 미래 폐지를 알리는 표지가 되지 않게 한다(정정 0 C2: 격자 밖 몫 폐지 0.501 / 생존 0.001)."""
    miss = ~np.isfinite(raw)
    return np.where(miss, snap_to_tick(C, days, mkt), raw), miss


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """두 표본 KS 통계(최대 ECDF 차)."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def check_tickp_leak(tickp: np.ndarray, uni: np.ndarray, delisted: np.ndarray, tol: "dict | None" = None) -> dict:
    """uni 칸 tickp 가 폐지 표지가 되는지 — 폐지·생존 사이 (1) NaN 몫 차, (2) 호가 격자 밖(|1/tickp − round| > 0.01) 몫 차,
    (3) 유한값 분포 KS 가 허용치(``TICKP_LEAK_TOL``) 이상이면 ValueError. 값 dict 를 돌려준다(진단용)."""
    tol = {**TICKP_LEAK_TOL, **(tol or {})}
    tp = np.asarray(tickp, float)
    uni = np.asarray(uni, bool)
    nan = np.isnan(tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / tp
        off = np.isfinite(inv) & (np.abs(inv - np.round(inv)) > 1e-2)
    out, bad, fin = {}, [], {}
    for name, g in (("delisted", delisted), ("alive", ~np.asarray(delisted, bool))):
        m = np.asarray(g, bool)[:, None] & uni
        out[f"nan_{name}"] = float(nan[m].mean()) if m.any() else 0.0
        f = m & ~nan
        out[f"offgrid_{name}"] = float(off[f].mean()) if f.any() else 0.0
        fin[name] = tp[f]
    out["ks"] = _ks(fin["delisted"], fin["alive"]) if len(fin["delisted"]) and len(fin["alive"]) else 0.0
    for k in ("nan", "offgrid"):
        if abs(out[f"{k}_delisted"] - out[f"{k}_alive"]) >= tol[k]:
            bad.append(f"{k} 몫 폐지 {out[f'{k}_delisted']:.4f} vs 생존 {out[f'{k}_alive']:.4f}(허용 {tol[k]})")
    if out["ks"] >= tol["ks"]:
        bad.append(f"분포 KS {out['ks']:.4f}(허용 {tol['ks']})")
    if bad:
        raise ValueError("tickp 폐지 표지 누수 — " + "; ".join(bad))
    return out


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


def apply_news_exclusion(uni: np.ndarray, codes: "list[str]", days: pd.DatetimeIndex, archive) -> "tuple[np.ndarray, dict]":
    """정정 0 C1 규칙 하나 — 뉴스 구간 전체에서 제외 종목(파일 없음 ∪ 잘린 page_cap)의 uni 를 끈다. ``NEWS_EXCLUDE`` 가 False 면 그대로.
    → (새 uni, 진단: 제외 코드 목록·끈 칸 수). 제외 집합은 빌드 때의 상수(시점 무관)다."""
    if not NEWS_EXCLUDE:
        return np.asarray(uni, bool), {"enabled": False}
    excl, det = news_excluded(codes, archive)
    u2, n_off = news_uni_off(uni, excl, days)
    return u2, {"enabled": True, "uni_off_cells": n_off, **det}


def _delisted_codes(con) -> set:
    return set(pd.read_sql_query("SELECT code FROM delisted_stocks", con)["code"].astype(str))


def build_era(con, era: str, end: "str | None" = None) -> dict:
    """prepare.build_era 단계 (ERAS_C) + 갭·뉴스 배열 + 뉴스 구간 uni 끄기 + 누수 검사(어기면 ValueError — 캐시를 쓰지 않는다).

    ``end`` = 데이터 끝 날짜 바꾸기(전진 빌더: 시대 C 를 끝 날짜만 굴린다 — PREREG 정정 0 M6a). 없으면 ERAS_C 그대로."""
    import research.experiments.indep30 as I30
    from swing_it.features.eps_asof import eps_ok, eps_yoy
    from swing_it.features.korea import cost_rt
    from swing_it.storage import read_index
    E = ERAS_C[era]
    data = (E["data"][0], end or E["data"][1])
    D = I30.build(con, data, E["eps"])
    days, codes = D["days"], D["codes"]
    isdel = np.isin(codes, sorted(set(D["delisted"]) | _delisted_codes(con)))
    arc = archive_dir()
    news = news_arrays(codes, days, arc)
    D["uni"], excl = apply_news_exclusion(D["uni"], codes, days, arc)        # EPS·후보 모두 끈 뒤의 uni 를 쓴다
    capped = np.isin(codes, excl.get("page_cap_kept", []) + excl.get("page_cap_late", []))
    news_leak = check_news_leak(news["n_all"], news["news_known"], D["uni"], days, isdel, capped=capped)
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
    raw, fb = fill_raw_with_adj(_raw_wide(con, data, days, codes), D["C"], days, mkt_real)
    out["mkt_real"] = mkt_real
    out.update(gap_arrays(D["O"], D["C"], raw, D["reopen"], D["lu"], days, mkt_real))
    out.update(news)
    tk = check_tickp_leak(out["tickp"], D["uni"], isdel)
    uc = D["uni"].sum(1)
    out["diag"] = {**D["diag"], "eps_up_share": float(up[ci, s].mean()),
                   "tickp_adj_fallback_share_uni": float(fb[D["uni"]].mean()),
                   **{f"tickp_leak_{k}": v for k, v in tk.items()},
                   **{f"news_exclusion_{k}": v for k, v in excl.items()}, **{f"news_leak_{k}": v for k, v in news_leak.items()},
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
    for k, v in D["diag"].items():
        if k.startswith(("tickp_", "news_", "mkt_")):
            d[k] = v
    for y in (2023, 2024):
        m = np.asarray(days.year == y)[None, :] & uni
        if m.any():
            d[f"news_known_share_uni_{y}"] = float(D["news_known"][m].mean())
    return d


def cache_path(era: str, half: int = 0) -> str:
    """탐색 절반 = CACHE/era_X_s0.pkl, 잠금 절반 = LOCKED_CACHE/era_X_s1.pkl(격리가 가리는 다른 디렉터리)."""
    return os.path.join(CACHE if half == 0 else LOCKED_CACHE, f"era_{era}_s{half}.pkl")


def write_halves(F: dict, era: str) -> None:
    """한 번 만든 시대 F 를 두 절반으로 나눠 쓴다 — 같은 빌드라 시점이 어긋나지 않는다(잠금 절반 DB 재구축 거부 위험 제거, M4)."""
    for half in (0, 1):
        p = cache_path(era, half)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        D = _slice(F, half)
        with open(p, "wb") as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"era {era}: 절반 {half} {len(D['codes'])} codes × {len(D['days'])} days → {p}")


def build_all(eras: tuple = ("A", "B", "C")) -> None:
    import psycopg2
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    for era in eras:
        F = build_era(con, era)
        for k, v in diagnostics(F).items():
            log(f"era {era} diag {k}: {v}")
        write_halves(F, era)


def load(era: str, half: int = 0, sha: "str | None" = None) -> dict:
    """디스크 캐시 적재. half 1(잠금 절반)은 select.py 만 부른다. ``sha`` 를 주면 읽은 바이트의 sha256 이 같아야 한다(읽은 그 바이트를 푼다)."""
    with open(cache_path(era, half), "rb") as f:
        b = f.read()
    if sha is not None and hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"캐시 era {era} 절반 {half} sha 불일치 — 거부")
    return pickle.loads(b)


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
