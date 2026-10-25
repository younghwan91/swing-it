"""결합 autoresearch 시대 캐시 — v4(prepare.build_era) 단계 + 야간 갭(scalp) + 뉴스(daytrade) 배열.

시대 A(2008~2016-08)·B(2016-09~2024)·확인 C(2016-09~2026-09-11). 탐색 절반(half 0)은 ``CACHE``(루프 격리가 A·B 만 다시 보인다),
잠금 절반(half 1)은 ``LOCKED_CACHE``(격리가 통째로 가린다 — select.py 잠금만 읽고 MANIFEST sha 로 대조, 정정 0 M4).
뉴스 구간(2023-03-15~)에서 아카이브 없는 종목·잘린 page_cap 종목은 구간 전체 uni 를 끈다(``apply_news_exclusion``, 정정 0 C1) —
루프·기준선·참고·선택·전진 모두 이 캐시를 쓴다.
``prepare.build_era`` 는 전역 ``ERAS`` 를 읽으므로 같은 단계를 ``ERAS_C`` 로 재현한다(prepare.py 는 바꾸지 않는다).

tickp 는 수정 종가로 계산하고 유효숫자 2자리로 거칠게 한다 — 원주가를 쓰지 않는다(원주가/수정가 = 미래 수정 계수 누수, 정정 0 판정 26).
V 의 O·H·L·C 는 종목별 비밀 배수(U[1,2), 숨긴 원장 디렉터리의 ``price_salt`` — 모든 시대·절반에서 같은 종목은 같은 배수)를 곱해 넘긴다
(정정 0-2 K1: 소급 수정가의 비정수 = t 이후 가격 수정 사건 있음 → 미래 정보). 수익률·비율은 그대로, 가격 수준은 의미를 잃는다.
뉴스는 증분 수집기와 같은 수집 대상(전일 거래대금 상위 300)일 때만 known(정정 0-2 M-b). 제외 분류·수집 대상은 ``FROZEN`` 에 얼려 MANIFEST 에.
실제 시장(mkt_real): ``stocks`` + ``delisted_stocks`` 의 market (코스닥=1, 거래소·유가증권·코넥스=0). 엔진용 ``mkt`` 는 v4 처럼 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from research.autoresearch.prepare import VIEW_KEYS as VIEW_KEYS_V4
from research.autoresearch.prepare import half_of, log, slice_half
from research.combined.gaps import gap_arrays, tick_dated_vec
from research.combined.news import (
    NEWS_START,
    apply_collect,
    archive_dir,
    check_news_leak,
    collect_universes,
    collection_mask,
    news_arrays,
    news_excluded,
    news_uni_off,
)

CACHE = os.path.expanduser("~/.cache/cc-swing/scratch/combined")
LOCKED_CACHE = os.path.expanduser("~/.cache/cc-swing/scratch/combined_locked")    # 잠금 절반(격리가 가린다)
FROZEN = os.path.expanduser("~/.cache/cc-swing/scratch/combined_frozen")        # 뉴스 제외 분류·수집 대상(격리가 가린다)
FROZEN_FILES = ("news_status.json", "news_universe.json")
FROZEN_ADDED = "news_added.jsonl"      # 전진 빌더가 처음 계산한 수집 대상·분류(덧붙이기 전용 — 먼저 적힌 값이 이긴다, 정정 0-3)
PRICE_SALT = os.path.expanduser("~/.cache/cc-swing/combined_ledger/price_salt")   # 원장 사본 디렉터리(격리가 가린다) — 출력 금지
FRAC_LEAK_TOL = 0.05
NEWS_EXCLUDE = True    # 뉴스 구간 종목 제외 규칙 스위치(사용자가 뉴스 부품을 빼기로 하면 False — 그때는 PREREG 정정으로)
TICKP_LEAK_TOL = {"nan": 0.01, "ks": 0.15}
# ks 0.15: 폐지 종목이 싸서 생기는 정상 차이(2026-10-25 캐시, 수정가 tickp 폐지/생존 KS A 0.04·B 0.03·C 0.06, 원주가 판 B·C 0.07~0.11) 위 여유.
# 격자 밖 검사는 뺐다 — 수정가 tickp 는 누구에게나 격자 밖이라 표지가 아니다.


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


PRICE_KEYS = ("O", "H", "L", "C")


def round_to_tick(p: np.ndarray, days, mkt: np.ndarray) -> np.ndarray:
    """수정가를 그날 호가표(시장별 옛 표·통합 표)의 자기 수준 격자로 반올림(단조 — H ≥ C ≥ L 유지; 구간 경계는 두 틱의 공배수)."""
    p = np.asarray(p, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        tk = tick_dated_vec(p, pd.DatetimeIndex(days), mkt)
        return np.round(p / tk) * tk


def view_c(D: dict) -> dict:
    """train.py 에 넘기는 묶음 — 복사본, 날짜·종목코드 없음. ``vmult`` 가 있으면 O·H·L·C 를 그날 호가 격자로 반올림한 뒤(정정 0-3 K1-bis:
    소급 수정가가 격자 밖이라 남던 미래 수정 사건·폐지 표지 제거) 종목별 비밀 배수(한 자릿수 log-균등)를 곱한다. 캐시(``load``)는 vmult 가
    없으면 거부한다. 장난감 D 처럼 vmult 키가 없으면 그대로."""
    V = {k: np.array(D[k], copy=True) for k in VIEW_KEYS_C}
    if "vmult" in D:
        m = np.asarray(D["vmult"], float)[:, None]
        for k in PRICE_KEYS:
            V[k] = round_to_tick(D[k], D["days"], D["mkt_real"]) * m
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


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """두 표본 KS 통계(최대 ECDF 차)."""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def check_tickp_leak(tickp: np.ndarray, uni: np.ndarray, delisted: np.ndarray, tol: "dict | None" = None) -> dict:
    """uni 칸 tickp 가 폐지 표지가 되는지 — 폐지·생존 사이 (1) NaN 몫 차, (2) 유한값 분포 KS 가 허용치(``TICKP_LEAK_TOL``) 이상이면
    ValueError. 값 dict 를 돌려준다(진단용). 수정가 tickp 는 C 의 함수라 통과가 기대값이다 — 원주가 등 다른 입력이 다시 들어오면 잡는다."""
    tol = {**TICKP_LEAK_TOL, **(tol or {})}
    tp = np.asarray(tickp, float)
    uni = np.asarray(uni, bool)
    nan = np.isnan(tp)
    out, bad, fin = {}, [], {}
    for name, g in (("delisted", delisted), ("alive", ~np.asarray(delisted, bool))):
        m = np.asarray(g, bool)[:, None] & uni
        out[f"nan_{name}"] = float(nan[m].mean()) if m.any() else 0.0
        fin[name] = tp[m & ~nan]
    out["ks"] = _ks(fin["delisted"], fin["alive"]) if len(fin["delisted"]) and len(fin["alive"]) else 0.0
    if abs(out["nan_delisted"] - out["nan_alive"]) >= tol["nan"]:
        bad.append(f"nan 몫 폐지 {out['nan_delisted']:.4f} vs 생존 {out['nan_alive']:.4f}(허용 {tol['nan']})")
    if out["ks"] >= tol["ks"]:
        bad.append(f"분포 KS {out['ks']:.4f}(허용 {tol['ks']})")
    if bad:
        raise ValueError("tickp 폐지 표지 누수 — " + "; ".join(bad))
    return out


def price_salt(path: str = PRICE_SALT) -> str:
    """종목별 가격 배수 소금 — 없으면 한 번 만든다(0600). 값은 어디에도 출력하지 않는다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, os.urandom(16).hex().encode())
            os.close(fd)
        except FileExistsError:
            pass
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def price_mult(codes: "list[str]", salt: str) -> np.ndarray:
    """종목별 배수 10^U[0,1)(한 자릿수 log-균등, 정정 0-3) — U = sha256(소금:코드) 앞 52비트 / 2^52. 같은 코드는 모든 시대·절반에서 같다.
    log-균등이라 격자 간격(틱 × 배수)의 가수가 종목과 무관하게 고르다(U[1,2) 는 가수 [1,2)·[5,10) 에 몰렸다)."""
    return np.array([10.0 ** (int(hashlib.sha256(f"{salt}:{c}".encode()).hexdigest()[:13], 16) / 16 ** 13) for c in codes])


def check_price_frac_leak(C: np.ndarray, VC: np.ndarray, uni: np.ndarray, tol: float = FRAC_LEAK_TOL) -> dict:
    """K1 빌드 검사: 원 수정가 C 의 비정수 칸(= t 뒤 가격 수정 사건 있음)과 정수 칸 사이에서, V 가 보는 종가 ``VC`` 의 소수부 분포
    KS 가 ``tol`` 이상이면 ValueError. 배수가 없으면 정수 칸 소수부가 전부 0 이라 걸린다."""
    C = np.asarray(C, float)
    ok = np.asarray(uni, bool) & np.isfinite(C)
    nonint = np.abs(C - np.round(C)) > 1e-6
    v = np.asarray(VC, float)
    fr = v - np.floor(v)
    a, b = fr[ok & nonint], fr[ok & ~nonint]
    out = {"nonint_share_uni": float(nonint[ok].mean()) if ok.any() else 0.0,
           "ks": _ks(a, b) if len(a) and len(b) else 0.0}
    if out["ks"] >= tol:
        raise ValueError(f"가격 소수부 누수 — 수정 사건 있음/없음 KS {out['ks']:.4f}(허용 {tol})")
    return out


LATTICE_W, LATTICE_N = 20, 1000
LATTICE_TOL_R, LATTICE_TOL_STRAT, LATTICE_PERM_N, LATTICE_PERM_Q, LATTICE_PERM_SEED = 0.02, 0.04, 200, 0.975, 20261026


def lattice_flags(V: dict, W: int = LATTICE_W) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """리뷰어(3차) 격자 탐지기: t 까지 W 일 창의 O·H·L·C(V) 0 아닌 최소 간격 d.
    격자 정수성 표지 = r = d/(tickp[t]·C[t−1]) 가 정수가 아니다(|r−k| > 0.12·max(k,1) 또는 k < 1), 가수 표지 = d 의 가수 ∈ [2.05, 4.95).
    → (정수성 표지, 가수 표지, 계산된 칸, N = C/d) 모두 [N,T]."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    P = np.stack([np.asarray(V[k], float) for k in PRICE_KEYS], -1)
    n, T, _ = P.shape
    fr, fm, ok, nn = np.zeros((n, T), bool), np.zeros((n, T), bool), np.zeros((n, T), bool), np.full((n, T), np.nan)
    if T < W + 1:
        return fr, fm, ok, nn
    t = np.arange(W - 1, T)
    tp = np.asarray(V["tickp"], float)
    for i in range(n):
        w = np.sort(swv(P[i], W, axis=0).reshape(T - W + 1, -1), 1)
        df = np.diff(w, axis=1)
        df = np.where(df > 1e-6 * np.abs(w[:, 1:]) + 1e-9, df, np.inf)
        d = df.min(1)
        with np.errstate(all="ignore"):
            r = d / (tp[i, t] * P[i, t - 1, 3])
            k = np.round(r)
            mant = d / 10 ** np.floor(np.log10(d))
            fr[i, t] = (np.abs(r - k) > 0.12 * np.maximum(k, 1)) | (k < 1)
            fm[i, t] = (mant >= 2.05) & (mant < 4.95)
            nn[i, t] = P[i, t, 3] / d
        ok[i, t] = np.isfinite(r) & np.all(np.isfinite(w), 1)
    return fr, fm, ok, nn


def _rate_diff(f: np.ndarray, sub: np.ndarray, L: np.ndarray) -> "tuple[float, float, int, int]":
    a, b = sub & L, sub & ~L
    return (float(f[a].mean()) if a.any() else float("nan"), float(f[b].mean()) if b.any() else float("nan"), int(a.sum()), int(b.sum()))


def _strat_diff(f: np.ndarray, sub: np.ndarray, L: np.ndarray, q: np.ndarray) -> float:
    """log10(가격/틱) 10분위 층마다 (라벨 − 그 밖) 표지율 차를 라벨 칸 수로 가중 평균."""
    x = np.log10(q[sub])
    if len(x) == 0:
        return float("nan")
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    s = np.full(q.shape, -1)
    s[sub] = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, max(len(edges) - 2, 0))
    num = den = 0.0
    for b in np.unique(s[sub]):
        m = sub & (s == b)
        a, o = m & L, m & ~L
        if a.any() and o.any():
            num += a.sum() * (f[a].mean() - f[o].mean())
            den += a.sum()
    return float(num / den) if den else float("nan")


def _perm_p(f: np.ndarray, sub: np.ndarray, L: np.ndarray, n: int = LATTICE_PERM_N, seed: int = LATTICE_PERM_SEED) -> "tuple[float, float, float]":
    """종목 단위 순열: 라벨 행을 종목끼리 섞은 표지율 차 분포 → (관측 차, |차| 의 97.5 백분위, 순열 p = P(|순열| ≥ |관측|))."""
    fa, fb, _, _ = _rate_diff(f, sub, L)
    obs = fa - fb
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        Lp = L[rng.permutation(L.shape[0])]
        a, b, _, _ = _rate_diff(f, sub, Lp)
        if np.isfinite(a) and np.isfinite(b):
            ds.append(abs(a - b))
    ds = np.asarray(ds)
    return obs, float(np.quantile(ds, LATTICE_PERM_Q)) if len(ds) else float("nan"), float((ds >= abs(obs)).mean()) if len(ds) else float("nan")


def check_lattice_leak(V: dict, uni: np.ndarray, labels: dict, halves: "np.ndarray | None" = None) -> dict:
    """빌드 수용 검사(판정 32·33). 라벨(미래 수정 사건 있음/없음, 폐지/생존)·절반마다:
    (r) 격자 정수성 표지율 차, uni 칸 전체 |차| < 0.02.
    (a) 가수 표지: 종목 단위 순열 200 회(고정 시드) — 관측 |차| ≤ 순열 |차| 97.5 백분위(종목별 배수의 무작위 효과라 정보가 아니다).
    (b) N = C/d > 1000 칸: log10(가격/틱) 10분위 층별 가중 정수성 표지율 차 |차| < 0.04(가격 구간 구성 통제). 가격/틱 = 1/tickp(V 에 있는 값).
    어기면 ValueError(캐시를 쓰지 않는다). 층 나누지 않은 차도 기록한다."""
    fr, fm, ok, nn = lattice_flags(V)
    ok = ok & np.asarray(uni, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 1.0 / np.asarray(V["tickp"], float)
    hv = np.zeros(ok.shape[0], int) if halves is None else np.asarray(halves, int)
    out, bad = {}, []
    for h in np.unique(hv):
        rows = (hv == h)[:, None]
        okh = ok & rows
        n1k = okh & (nn > LATTICE_N) & np.isfinite(q) & (q > 0)
        for lab, L in labels.items():
            L = np.broadcast_to(np.asarray(L, bool), ok.shape)
            key = f"{lab}_h{h}"
            ra, rb, na, nb = _rate_diff(fr, okh, L)
            ma, mb, _, _ = _rate_diff(fm, okh, L)
            obs_m, q975, p_m = _perm_p(fm, okh, L)
            n_a, n_b, _, _ = _rate_diff(fr, n1k, L)
            strat = _strat_diff(fr, n1k, L, q)
            out[key] = {"cells": na, "cells_other": nb, "r_diff": ra - rb, "m_diff": ma - mb, "m_perm_q975": q975, "m_perm_p": p_m,
                        "n1000_r_diff_raw": n_a - n_b, "n1000_r_diff_strat": strat}
            if not abs(ra - rb) < LATTICE_TOL_R:
                bad.append(f"{key} 정수성 차 {ra - rb:+.3f}")
            if not abs(obs_m) <= q975:
                bad.append(f"{key} 가수 차 {obs_m:+.3f} > 순열 97.5% {q975:.3f}")
            if not abs(strat) < LATTICE_TOL_STRAT:
                bad.append(f"{key} N>1000 층별 차 {strat:+.3f}")
    if bad:
        raise ValueError("격자 탐지기 수용 검사 실패 — " + "; ".join(bad))
    return out


def load_frozen(path: str = FROZEN) -> dict:
    """얼린 뉴스 제외 분류·수집 대상 → {"status": {코드: 분류}, "universe": {날: [코드]}}. 없으면 빈 dict(빌드가 채운다).
    태그 전 빌드의 두 JSON 위에 ``news_added.jsonl``(전진 빌더가 처음 계산한 값)을 얹는다 — 이미 있는 키는 덮지 않는다(먼저 적힌 값이 이긴다)."""
    fr = {}
    for k, f in zip(("status", "universe"), FROZEN_FILES, strict=True):
        q = os.path.join(path, f)
        if os.path.exists(q):
            with open(q, encoding="utf-8") as fh:
                fr[k] = json.load(fh)
    q = os.path.join(path, FROZEN_ADDED)
    if os.path.exists(q):
        with open(q, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    fr.setdefault(r["kind"], {}).setdefault(r["key"], r["value"])
    return fr


def append_frozen(frozen: dict, before: dict, path: str = FROZEN) -> int:
    """``before``(빌드 시작 때 키 집합) 뒤에 새로 계산된 분류·수집 대상을 ``news_added.jsonl`` 에 덧붙인다(전진 빌더용). 덧붙인 줄 수."""
    os.makedirs(path, exist_ok=True)
    n = 0
    with open(os.path.join(path, FROZEN_ADDED), "a", encoding="utf-8") as fh:
        for kind in ("status", "universe"):
            for k in sorted(set(frozen.get(kind, {})) - set(before.get(kind, ()))):
                fh.write(json.dumps({"kind": kind, "key": k, "value": frozen[kind][k]}, ensure_ascii=False) + "\n")
                n += 1
    return n


def write_frozen(fr: dict, path: str = FROZEN) -> None:
    os.makedirs(path, exist_ok=True)
    for k, f in zip(("status", "universe"), FROZEN_FILES, strict=True):
        with open(os.path.join(path, f), "w", encoding="utf-8") as fh:
            json.dump(dict(sorted(fr.get(k, {}).items())), fh, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _collect_mask(con, codes: list, days: pd.DatetimeIndex, frozen: dict) -> np.ndarray:
    """뉴스 구간 날마다 수집 대상(얼린 값 우선, 없는 날만 DB 에서 계산해 ``frozen`` 에 더한다) → bool[N,T]."""
    uv = frozen.setdefault("universe", {})
    need = [d for d in pd.DatetimeIndex(days).normalize() if d >= NEWS_START and str(d.date()) not in uv]
    if need:
        from swing_it.storage import read_prices
        tv = read_prices(con, cols=("date", "code", "trade_value"), start=str((min(need) - pd.Timedelta(days=21)).date()),
                         end=str(max(need).date()), parse_dates=True, session="raw")       # 수집기와 같은 테이블 원값
        uv.update(collect_universes(tv, need))
    return collection_mask(codes, days, uv)


def apply_news_exclusion(uni: np.ndarray, codes: "list[str]", days: pd.DatetimeIndex, archive,
                         frozen: "dict | None" = None) -> "tuple[np.ndarray, dict]":
    """정정 0 C1 규칙 하나 — 뉴스 구간 전체에서 제외 종목(파일 없음 ∪ 잘린 page_cap)의 uni 를 끈다. ``NEWS_EXCLUDE`` 가 False 면 그대로.
    → (새 uni, 진단: 제외 코드 목록·끈 칸 수). 제외 집합은 빌드 때의 상수(시점 무관)다."""
    if not NEWS_EXCLUDE:
        return np.asarray(uni, bool), {"enabled": False}
    excl, det = news_excluded(codes, archive, frozen)
    u2, n_off = news_uni_off(uni, excl, days)
    return u2, {"enabled": True, "uni_off_cells": n_off, **det}


def _delisted_codes(con) -> set:
    return set(pd.read_sql_query("SELECT code FROM delisted_stocks", con)["code"].astype(str))


def build_era(con, era: str, end: "str | None" = None, frozen: "dict | None" = None, salt: "str | None" = None,
              persist: "str | None" = None) -> dict:
    """prepare.build_era 단계 (ERAS_C) + 갭·뉴스 배열 + 뉴스 구간 uni 끄기 + 누수 검사 셋(어기면 ValueError — 캐시를 쓰지 않는다).

    ``end`` = 데이터 끝 날짜 바꾸기(전진 빌더: 시대 C 를 끝 날짜만 굴린다 — PREREG 정정 0 M6a). 없으면 ERAS_C 그대로.
    ``frozen`` = 얼린 뉴스 제외 분류·수집 대상(``load_frozen``) — 없는 항목은 계산해 이 dict 에 더한다(``build_all`` 이 파일로 쓴다).
    ``salt`` = 가격 배수 소금(없으면 ``price_salt()``).
    ``persist`` = 얼린 디렉터리 — 주면 이 빌드가 처음 계산한 수집 대상·분류를 거기 ``news_added.jsonl`` 에 덧붙인다(전진 빌더: 한 번 정한
    U(d) 를 다시 계산하지 않는다 — 정정 0-3 사소 2)."""
    import research.experiments.indep30 as I30
    from swing_it.features.eps_asof import eps_ok, eps_yoy
    from swing_it.features.korea import cost_rt
    from swing_it.storage import read_index
    frozen = {} if frozen is None else frozen
    before = {k: set(frozen.get(k, {})) for k in ("status", "universe")}
    E = ERAS_C[era]
    data = (E["data"][0], end or E["data"][1])
    D = I30.build(con, data, E["eps"])
    days, codes = D["days"], D["codes"]
    isdel = np.isin(codes, sorted(set(D["delisted"]) | _delisted_codes(con)))
    arc = archive_dir()
    news_raw = news_arrays(codes, days, arc)
    collect = _collect_mask(con, codes, days, frozen)
    news = apply_collect(news_raw, collect)                                  # 수집기와 같은 시점 규칙(M-b)
    D["uni"], excl = apply_news_exclusion(D["uni"], codes, days, arc, frozen)  # EPS·후보 모두 끈 뒤의 uni 를 쓴다
    capped = np.isin(codes, excl.get("page_cap_kept", []) + excl.get("page_cap_late", []))
    news_leak = check_news_leak(news["n_all"], news["news_known"], D["uni"], days, isdel, capped=capped, collect=collect,
                                archive_known=news_raw["news_known"])
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
    out["mkt_real"] = mkt_real
    Or, Cr = round_to_tick(D["O"], days, mkt_real), round_to_tick(D["C"], days, mkt_real)   # V 와 같은 반올림 가격으로(갭·tickp)
    out.update(gap_arrays(Or, Cr, D["reopen"], D["lu"], days, mkt_real))
    out.update(news)
    tk = check_tickp_leak(out["tickp"], D["uni"], isdel)
    out["vmult"] = price_mult(codes, price_salt() if salt is None else salt)
    Vfull = view_c(out)
    fl = check_price_frac_leak(D["C"], Vfull["C"], D["uni"])
    lat = check_lattice_leak(Vfull, D["uni"], {"event": np.abs(D["C"] - np.round(D["C"])) > 1e-6, "delisted": isdel[:, None]},
                             halves=out["half"])
    win = np.asarray(pd.DatetimeIndex(days) >= NEWS_START)[None, :] & D["uni"]
    uc = D["uni"].sum(1)
    out["diag"] = {**D["diag"], "eps_up_share": float(up[ci, s].mean()),
                   **{f"tickp_leak_{k}": v for k, v in tk.items()}, **{f"price_frac_leak_{k}": v for k, v in fl.items()},
                   **{f"price_lattice_{k}": v for k, v in lat.items()},
                   "news_collect_share_uni_window": float(collect[win].mean()) if win.any() else None,
                   **{f"news_exclusion_{k}": v for k, v in excl.items()}, **{f"news_leak_{k}": v for k, v in news_leak.items()},
                   "mkt_unmapped_uni_cells": {c: int(uc[codes.index(c)]) for c in miss if uc[codes.index(c)] > 0},
                   "mkt_stocks_vs_delisted_disagree": n_dis}
    if persist is not None:
        out["diag"]["news_frozen_appended"] = append_frozen(frozen, before, persist)
    return out


def _slice(D: dict, half: int) -> dict:
    out = slice_half(D, half)
    for k in ("mkt_real", "vmult"):
        if k in D:
            out[k] = D[k][D["half"] == half]
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
        if k.startswith(("tickp_", "news_", "mkt_", "price_")):
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
    """시대마다 한 빌드 → 두 절반. 뉴스 제외 분류·수집 대상은 새로 계산해 ``FROZEN`` 에 얼린다(전진 빌더는 그 파일을 쓴다)."""
    import psycopg2
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    frozen, salt = {}, price_salt()
    for era in eras:
        F = build_era(con, era, frozen=frozen, salt=salt)
        for k, v in diagnostics(F).items():
            if not (isinstance(v, list) and len(v) > 20):
                log(f"era {era} diag {k}: {v}")
            else:
                log(f"era {era} diag {k}: {len(v)} 개")
        write_halves(F, era)
    write_frozen(frozen)
    log(f"얼린 뉴스 파일: 분류 {len(frozen.get('status', {}))} 종목, 수집 대상 {len(frozen.get('universe', {}))} 일 → {FROZEN}")


def load(era: str, half: int = 0, sha: "str | None" = None) -> dict:
    """디스크 캐시 적재. half 1(잠금 절반)은 select.py 만 부른다. ``sha`` 를 주면 읽은 바이트의 sha256 이 같아야 한다(읽은 그 바이트를 푼다)."""
    with open(cache_path(era, half), "rb") as f:
        b = f.read()
    if sha is not None and hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"캐시 era {era} 절반 {half} sha 불일치 — 거부")
    return require_vmult(pickle.loads(b))


def require_vmult(D: dict) -> dict:
    """캐시에서 읽은 시대는 가격 배수를 가져야 한다(없으면 V 의 가격이 비정수 = 미래 수정 표지 — K1)."""
    if "vmult" not in D or len(D["vmult"]) != len(D["codes"]):
        raise SystemExit("캐시에 가격 배수(vmult)가 없다 — 재구축(data.build_all) 뒤 다시")
    return D


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
