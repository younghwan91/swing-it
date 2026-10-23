"""21세대 주도주 보유형 포트폴리오(META_EVOLUTION.md 21세대 — 결과 전 고정). 일별 시뮬레이터 + 격자 72 + 확장 워크포워드 + 기준선.

정의는 META_EVOLUTION.md 21세대 사전등록 그대로다. 캐시 배열은 종목×날짜(수정 주가), 거래대금(백만원)과 EPS as-of 는 DB.

사용: ``uv run --with numba --extra pg python research/experiments/leader21.py --cache <v13> --out <dir> --workers 10`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from swing_it.units import eok  # noqa: E402

sys.path[:0] = [".", "src"]

SEED = 20261021
START, END = "2017-10-02", "2026-10-08"
FOLDS = (("2017-10-02", "2020-01-01", "2022-01-01"), ("2017-10-02", "2022-01-01", "2024-01-01"))
GRID = {"theta": (70, 80, 90), "N": (20, 50, 120), "ma_exit": (50, 100), "hard": (0.15, 0.25), "pyramid": (0, 1),
        "univ": ("all", "top200"), "regime": (0, 1)}            # 보강(검증 23차 권고, 결과 전): 시가총액 상위 200 · 시장 국면(시총 가중 200일선)
MAX_POS, RISK, POS_CAP, ADV_MIN, LU = 10, 0.005, 0.10, eok(10), 1.29
SG = {"004690", "017390", "016710", "003100", "004360", "032190", "003380", "030210"}
_D: dict = {}


def _roll(df: pd.DataFrame, w: int, fn: str) -> np.ndarray:
    return getattr(df.rolling(w, min_periods=w), fn)().to_numpy().T


def prepare(cache: str, extras: bool = False, raw_close: bool = False) -> dict:
    import psycopg2

    from research.signals import pattern_events as PE
    days = pd.DatetimeIndex(np.load(f"{cache}/days_ns.npy"))
    codes = [str(c) for c in np.load(f"{cache}/codes.npy", allow_pickle=True)]
    O, H, L, C = (np.load(f"{cache}/{k}.npy").astype(float) for k in ("O", "H", "L", "C"))  # noqa: E741
    TR = np.load(f"{cache}/TR.npy").astype(bool)
    cost = np.load(f"{cache}/cost_tab.npy")
    F = np.load(f"{cache}/F.npy", mmap_mode="r")
    I_ = np.load(f"{cache}/I.npy", mmap_mode="r")
    mk_of = dict(zip(np.asarray(I_[:, PE.II["ci"]]).astype(int), np.asarray(F[:, PE.FI["market"]]).astype(int), strict=True))
    mkt = np.array([mk_of.get(i, 0) for i in range(len(codes))], np.int64)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return build(days, codes, O, H, L, C, TR, cost, mkt, con, extras=extras, raw_close=raw_close)


def build(days, codes, O, H, L, C, TR, cost, mkt, con, pit_shares: bool = False, extras: bool = False, krx_patch: bool = False,  # noqa: E741
          raw_close: bool = False, session: str = "raw", aftermarket_fix: bool = False, unpatched: str = "nan",
          tv_wide: "np.ndarray | None" = None, cap_wide: "np.ndarray | None" = None, eps_source: str = "earnings") -> dict:
    """가격 배열(종목×날짜) → 후보·지표 묶음. 백테스트(캐시)와 모의매매(DB) 공통.

    ``session``·``aftermarket_fix``·``unpatched`` 는 정문 인자 그대로(기본 = 기존 팔 고정값 ``"raw"``·``False``·``"nan"`` — raw 면
    unpatched 는 동작 없음). 모의매매 쌍 팔 ``*_rs``(META_EVOLUTION.md '전진 팔 *_rs 사전등록' + 보충)만 ``"regular"``·``True``·``"raw"``
    — 그 뒤의 ``krx_patch`` 분봉 보정은 멱등이라 그대로 둔다.

    ``tv_wide``(종목×날짜 거래대금, 백만원)·``cap_wide``(종목×날짜 시총)를 주면 DB 에서 읽지 않고 그것을 쓴다 — 30 독립 표본(2008~2016,
    가격 원천 = daily_bars_adj_v2)용. 기본 None = 기존 경로 그대로. ``eps_source`` 는 실적 원천(:mod:`swing_it.features.eps_asof`
    ``source``; 기본 ``"earnings"`` = 기존 경로, 30 독립 표본만 ``"earnings_dart_doc"`` — 반환 ``D["eps_source"]`` 로 rl32 특성도 같은 원천).
    """
    from swing_it.features.eps_asof import eps_ok
    from swing_it.storage import read_prices
    sess = None
    if tv_wide is None:
        tv = read_prices(con, session=session, cols=("code", "date", "trade_value"), start="2016-08-01", parse_dates=True, unpatched=unpatched)
        if krx_patch:                               # 모의매매: 2026-09-14+ KRX 애프터마켓 섞인 일봉 → 분봉 정규장 값(krx_session)
            from swing_it.krx_session import NXT_START, patch_prices, read_session_bars_agg
            sess = read_session_bars_agg(con, start=NXT_START)
            tv = patch_prices(tv, sess, adjusted=False)
        tvw = tv.pivot_table(index="date", columns="code", values="trade_value", aggfunc="last").reindex(index=days, columns=codes)
    else:
        tvw = pd.DataFrame(np.asarray(tv_wide, float).T, index=days, columns=codes)
    adv = tvw.rolling(20, min_periods=15).mean().to_numpy().T
    Cd = pd.DataFrame(C.T)
    ma = {w: _roll(Cd, w, "mean") for w in (50, 100, 150, 200)}
    ma200_prev = np.concatenate([np.full((C.shape[0], 20), np.nan), ma[200][:, :-20]], axis=1)
    hi252, lo252 = _roll(pd.DataFrame(H.T), 252, "max"), _roll(pd.DataFrame(L.T), 252, "min")
    def ret(k):
        return C / np.concatenate([np.full((C.shape[0], k), np.nan), C[:, :-k]], axis=1) - 1.0
    raw = 2 * ret(63) + ret(126) + ret(189) + ret(252)
    uni = np.nan_to_num(adv) >= ADV_MIN
    rs = pd.DataFrame(np.where(uni & np.isfinite(raw), raw, np.nan).T).rank(axis=1, pct=True).to_numpy().T * 100
    with np.errstate(invalid="ignore"):
        tt = ((C > ma[50]) & (ma[50] > ma[150]) & (ma[150] > ma[200]) & (ma[200] > ma200_prev) & (C >= 0.75 * hi252) & (C >= 1.30 * lo252))
    brk = {n: C >= _roll(Cd, n, "max") - 1e-9 for n in GRID["N"]}
    # 시가총액(원 종가 × 상장주식수 as-of) → 그날 순위 · 시총 가중 시장(KOSPI 대용) 200일선 국면
    from swing_it.storage import read_shares
    sh = read_shares(con, parse_dates=True) if cap_wide is None else None  # 정문(가드레일 (k)) — 예전 직접 SELECT 와 같은 행·순서
    if cap_wide is not None:                        # 30 독립 표본: 부르는 쪽이 만든 시총
        cap, n_bad = np.asarray(cap_wide, float), -1
    elif raw_close:                                   # 정정 G28-4: 원 종가 × 분할 일관 주식수(krx_session.split_consistent_cap)
        from swing_it.krx_session import read_raw_close, split_consistent_cap
        rc = read_raw_close(con, start="2016-08-01", aftermarket_fix=aftermarket_fix)
        rcw = rc.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days, columns=codes)
        src = rc.pivot_table(index="date", columns="code", values="raw_src", aggfunc="last").reindex(index=days, columns=codes)
        adw = pd.DataFrame(C.T, index=days, columns=codes)                               # 연속 수정 종가(이 build 의 C)
        capw, n_bad = split_consistent_cap(rcw, adw, sh, asof="knowledge_date" if pit_shares else "date", raw_src=src)
        cap = capw.to_numpy().T
        if aftermarket_fix:                         # 쌍 팔 진단: 애프터마켓 뒤 원 종가 원천별 (종목-일) 수 · 마지막 날 원천별 종목 수
            from swing_it.krx_session import AFTERMARKET_START
            late = src.loc[src.index >= pd.Timestamp(AFTERMARKET_START)]
            raw_src_diag = {"since": {k: int(v) for k, v in late.stack().value_counts().items()},
                            "last_day": {k: int(v) for k, v in src.iloc[-1].value_counts().items()}}
    else:                                           # 21~26세대 기록 재현용(결함: 소급 수정 가격 × 그 시점 주식수)
        px = read_prices(con, session="raw", cols=("code", "date", "close"), table="daily_bars", start="2016-08-01", parse_dates=True)
        if sess is not None:
            px = patch_prices(px, sess, adjusted=False)
        pxw = px.pivot_table(index="date", columns="code", values="close", aggfunc="last").reindex(index=days, columns=codes)
        if pit_shares:                              # 모의매매(검증 24차 MUST-FIX): 알게 된 날(knowledge_date) 기준, 없으면 date
            sh["date"] = sh["knowledge_date"].fillna(sh["date"])
        sh = sh[(sh["shares_outstanding"] > 0) & (sh["shares_outstanding"] <= 1e10)]   # 정제(21세대 결과 뒤 발견 — 082640 161조 주 등 오류 행)
        shw = sh.pivot_table(index="date", columns="code", values="shares_outstanding", aggfunc="last")
        shw = shw.reindex(index=shw.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
        cap = (pxw * shw).to_numpy().T
        n_bad = -1
    cap = np.where(cap <= 2e15, cap, np.nan)                                          # 시총 2,000조원 초과 = 오류
    crank = pd.DataFrame(cap.T).rank(axis=1, ascending=False).to_numpy().T
    r1 = C[:, 1:] / C[:, :-1] - 1
    r1 = np.concatenate([np.full((C.shape[0], 1), np.nan), r1], axis=1)
    cprev = np.concatenate([np.full((C.shape[0], 1), np.nan), cap[:, :-1]], axis=1)
    w = np.where(np.isfinite(r1) & np.isfinite(cprev) & (cprev > 0), cprev, 0.0)
    cw = np.where(w.sum(0) > 0, (np.nan_to_num(r1) * w).sum(0) / np.maximum(w.sum(0), 1e-12), 0.0)
    cw = np.clip(cw, -0.2, 0.2)
    cwi = np.cumprod(1 + cw)
    regime_on = cwi >= pd.Series(cwi).rolling(200, min_periods=200).mean().to_numpy()
    hi20 = C >= _roll(Cd, 20, "max") - 1e-9
    base = uni & tt & TR & (rs >= min(GRID["theta"]))
    ci, s = np.nonzero(base & brk[min(GRID["N"])] | base & brk[50] | base & brk[120])
    ok = eps_ok(codes, days, ci, s, con, source=eps_source)
    eps = np.zeros_like(base)
    eps[ci[ok], s[ok]] = True
    return {"days": days, "codes": codes, "O": O, "H": H, "L": L, "C": C, "TR": TR, "cost": cost, "mkt": mkt, "rs": rs,
            "cand": base & eps, "brk": brk, "hi20": hi20, "ma": ma, "adv": adv, "uni": uni, "crank": crank, "cap": cap, "cap_jump_bad": n_bad, "cw": cw,
            "regime_on": regime_on, "eps_source": eps_source, **({"raw_src_diag": raw_src_diag} if raw_close and aftermarket_fix else {}),
            **(_extras(days, codes, C, ma, uni, rs, cwi, regime_on, base & eps, con) if extras else {})}


def _extras(days, codes, C, ma, uni, rs, cwi, regime_on, cand, con) -> dict:
    """22세대 팔(결과 전 고정): 국면 변형 · 업종(WICS) 강도 · 실적 성장."""
    from swing_it.features.eps_asof import eps_yoy
    ma50i = pd.Series(cwi).rolling(50, min_periods=50).mean().to_numpy()
    regime_r50 = regime_on | (cwi >= ma50i)                                   # 200일선 위 ∨ 50일선 회복(V자 재진입)
    with np.errstate(invalid="ignore"):
        above = (C > ma[200]) & uni & np.isfinite(ma[200])
    breadth = above.sum(0) / np.maximum((uni & np.isfinite(ma[200])).sum(0), 1)
    breadth_on = breadth >= 0.40
    # WICS 업종(coalesce(industry_cd, wi26_cd)) 주간 스냅샷 → 날짜별 앞으로 채움 → 업종 평균 RS 상위 30%
    w = pd.read_sql("select date, code, coalesce(industry_cd, wi26_cd) as g from wics_membership", con, parse_dates=["date"])
    gid = {g: i for i, g in enumerate(sorted(w["g"].dropna().unique()))}
    w["gi"] = w["g"].map(gid)
    gw = w.pivot_table(index="date", columns="code", values="gi", aggfunc="last")
    gw = gw.reindex(index=gw.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
    G = gw.to_numpy().T                                                       # 종목×날짜, NaN = 모름
    strong = np.zeros_like(cand)
    ng = len(gid)
    for t in np.flatnonzero(cand.any(0)):
        g = G[:, t]
        ok = np.isfinite(g) & np.isfinite(rs[:, t])
        if ok.sum() < 50:
            strong[:, t] = True                                               # 업종 정보 없는 시기(2019 전) = 거르지 않음
            continue
        gi = g[ok].astype(int)
        cnt = np.bincount(gi, minlength=ng)
        mean = np.bincount(gi, weights=rs[ok, t], minlength=ng) / np.maximum(cnt, 1)
        valid = cnt >= 3
        if valid.sum() < 5:
            strong[:, t] = True
            continue
        thr = np.quantile(mean[valid], 0.70)
        top = valid & (mean >= thr)
        strong[ok, t] = top[gi]
    ci, s = np.nonzero(cand)
    yoy = eps_yoy(codes, days, ci, s, con)
    growth = np.zeros_like(cand)
    growth[ci, s] = np.nan_to_num(yoy, nan=-1) >= 0.25
    return {"regime_r50": regime_r50, "breadth_on": breadth_on, "breadth": breadth, "sector_strong": strong, "eps_growth": growth}


def simulate(D: dict, cfg: dict, cost_mult: float, end: str = END, state: "dict | None" = None,
             paper: bool = False, start: str = START, net_mtm: bool = False) -> "tuple[pd.Series, pd.DataFrame]":
    """``paper`` = 모의매매 보정(검증 24차): 거래정지일 매도 미룸 · 손절 비교 허용오차 · 국면 청산 사유 표시. 백테스트 기본은 끔(재현).

    ``net_mtm`` = 보유 평가를 청산 비용(그날 왕복 × ``cost_mult``)을 뺀 순가치로 — autoresearch 평가기(리뷰 M4: 폴드 끝 보유분 비용 누락).
    기본 끔 = 기존 경로 그대로."""
    days, O, H, L, C = D["days"], D["O"], D["H"], D["L"], D["C"]  # noqa: E741, F841
    t0, t1 = int(days.searchsorted(pd.Timestamp(start))), int(days.searchsorted(pd.Timestamp(end), side="right"))
    pos_cap = cfg.get("pos_cap", POS_CAP)             # 전진 팔 leader_C20: 비중 절반(pos_cap·risk) — 기본값 = 기존 상수
    unit = min(pos_cap, cfg.get("risk", RISK) / cfg["hard"])
    max_units = max(1, int(round(pos_cap / unit)))
    mae = D["ma"][cfg["ma_exit"]]
    cand = D["cand"] & D["brk"][cfg["N"]] & (D["rs"] >= cfg["theta"])
    if cfg.get("univ") == "top200":
        cand = cand & (np.nan_to_num(D["crank"], nan=1e9) <= 200)
    if cfg.get("exclude"):                          # 23세대 강건성: 이 종목들은 아예 안 산 것으로(반사실 재시뮬레이션)
        ex = np.isin(np.asarray(D["codes"]), list(cfg["exclude"]))
        cand = cand & ~ex[:, None]
    if cfg.get("cand_and"):                         # 30 독립 표본 널: 부르는 쪽이 만든 무작위 솎기 마스크(D[키]) — 없으면 기존 경로
        cand = cand & D[cfg["cand_and"]]
    if cfg.get("rank") == "rs_sector":
        cand = cand & D["sector_strong"]
    elif cfg.get("rank") == "rs_eps":
        cand = cand & D["eps_growth"]
    struct = cfg.get("stop") == "struct"            # 전진 팔 leader_P(META '전진 팔 leader_P 사전등록'): 구조 손절·위험 비중 — 없으면 기존 경로 그대로
    if cfg.get("chase_max") is not None:            # leader_P: 쫓지 않기 — 종가 ≤ 직전 20일 최고 종가(그날 제외) × (1 + chase_max)
        prev20 = pd.DataFrame(C.T).shift(1).rolling(20, min_periods=20).max().to_numpy().T
        with np.errstate(invalid="ignore"):
            cand = cand & (C <= prev20 * (1 + cfg["chase_max"]))
    if struct:                                      # 손절선 = 그날 포함 10일 최저 저가, 거리 > stop_max 면 후보에서 뺀다(슬롯 순위 전)
        assert cfg.get("entry") == "close", "구조 손절은 종가 진입 팔만"
        low10 = pd.DataFrame(L.T).rolling(10, min_periods=10).min().to_numpy().T
        with np.errstate(invalid="ignore", divide="ignore"):
            sdist = 1.0 - low10 / C
            cand = cand & (sdist > 0) & (sdist <= cfg["stop_max"])
    reg = cfg.get("regime", 0)
    reg_arr = {"k200": D["regime_on"], "k200r50": D.get("regime_r50"), "breadth": D.get("breadth_on")}.get(cfg.get("regime_kind", "k200"))
    max_pos = int(cfg.get("max_pos", MAX_POS))
    entry = cfg.get("entry", "open")                # 23세대: "open" = 다음날 시가(기존) · "close" = 신호일 종가(종가 단일가)
    gap_max = cfg.get("gap_max")
    lu_arr = D.get("lu")                            # 30 독립 표본: 날짜별 상한가 판정 배수(2015-06-15 전 ±15%) — 없으면 LU
    cash, pos = 1.0, {}                        # ci → {qty, cost_basis, first_px, units, entry_t, last_px}
    p_buy, p_sell, p_add = [], set(), set()
    sell_why: dict = {}
    eq, trades = np.zeros(t1 - t0), []
    expo = np.zeros(t1 - t0)                       # 25세대: 그날 종가 위성 노출(포지션 가치 / 자산)
    eq_prev = 1.0
    def close_pos(ci, px, t, why):
        nonlocal cash
        p = pos.pop(ci)
        c_rt = D["cost"][D["mkt"][ci], min(t, D["cost"].shape[1] - 1)] * cost_mult
        cash += p["qty"] * px * (1 - c_rt)
        trades.append({"code": D["codes"][ci], "entry": str(days[p["entry_t"]].date()), "exit": str(days[t].date()), "why": why,
                       "ret": px * (1 - c_rt) * p["qty"] / p["cost_basis"] - 1, "units": p["units"], "hold": t - p["entry_t"],
                       "w_peak": p["w_peak"], **({"stop_dist": p["sdist"], "unit_w": p["uw"]} if "stop" in p else {})})
    for k, t in enumerate(range(t0, t1)):
        # 1) 시가 체결: 매도 → 추가 → 매수
        keep = set()
        for ci in list(p_sell):
            if ci in pos:
                if paper and not D["TR"][ci, t]:
                    keep.add(ci)                           # 거래정지: 다음 거래 가능일로 미룬다
                    continue
                px = O[ci, t] if np.isfinite(O[ci, t]) else pos[ci]["last_px"]
                close_pos(ci, px, t, sell_why.get(ci, "ma_exit"))
        p_sell = keep
        sell_why = {c: w for c, w in sell_why.items() if c in keep}
        for ci in list(p_add):
            if ci in pos and np.isfinite(O[ci, t]) and D["TR"][ci, t]:
                amt = min(pos[ci].get("uw", unit) * eq_prev, cash)
                if amt > 0:
                    pos[ci]["qty"] += amt / O[ci, t]
                    pos[ci]["cost_basis"] += amt
                    pos[ci]["units"] += 1
                    cash -= amt
        p_add.clear()
        for ci in p_buy:
            if ci in pos or len(pos) >= max_pos:
                continue
            op, pc = O[ci, t], C[ci, t - 1]
            if not (np.isfinite(op) and op > 0 and D["TR"][ci, t]) or (np.isfinite(pc) and op >= pc * (LU if lu_arr is None else lu_arr[t])):
                continue
            if gap_max is not None and np.isfinite(pc) and op > pc * (1 + gap_max):     # 23세대: 갭 상승 진입 회피
                continue
            amt = min(unit * eq_prev, cash)
            if amt <= 0:
                continue
            pos[ci] = {"qty": amt / op, "cost_basis": amt, "first_px": op, "units": 1, "entry_t": t, "last_px": op, "w_peak": 0.0}
            cash -= amt
        p_buy = []
        # 2) 장중 하드 손절(평단 기준)
        for ci in list(pos):
            p = pos[ci]
            avg = p["cost_basis"] / p["qty"]
            stop = p["stop"] if "stop" in p else avg * (1 - cfg["hard"])     # leader_P: 진입 때 고정한 구조선(올리지 않는다)
            if np.isfinite(L[ci, t]) and L[ci, t] <= stop * (1 + (1e-9 if paper else 0.0)):
                px = min(O[ci, t], stop) if np.isfinite(O[ci, t]) else stop
                close_pos(ci, px, t, "hard_stop")
        # 3) 종가 평가 · 신호
        mtm = 0.0
        for ci, p in pos.items():
            if np.isfinite(C[ci, t]):
                p["last_px"] = C[ci, t]
            mtm += p["qty"] * p["last_px"] * ((1 - D["cost"][D["mkt"][ci], min(t, D["cost"].shape[1] - 1)] * cost_mult) if net_mtm else 1.0)
        e = cash + mtm
        eq[k] = e
        expo[k] = mtm / e if e > 0 else 0.0
        for ci, p in list(pos.items()):
            p["w_peak"] = max(p["w_peak"], p["qty"] * p["last_px"] / e)
            c_ = C[ci, t]
            if not np.isfinite(c_):
                p["nan"] = p.get("nan", 0) + 1
                if p["nan"] >= 5:
                    close_pos(ci, p["last_px"], t, "no_data")
                continue
            p["nan"] = 0
            if np.isfinite(mae[ci, t]) and c_ < mae[ci, t]:
                p_sell.add(ci)
            elif cfg["pyramid"] and p["units"] < p.get("mu", max_units) and c_ >= p["first_px"] * (1 + 0.10 * p["units"]) and D["hi20"][ci, t]:
                p_add.add(ci)
        if reg and not reg_arr[t]:                # 국면 꺼짐: 새로 안 산다 + regime_mode 에 따라 보유 처리
            rm = cfg.get("regime_mode", "liquidate")  # 26세대: liquidate(기존 전량) · gate(보유 유지) · weak(MA50 아래만 매도)
            if rm == "liquidate":
                victims = list(pos.keys())
            elif rm == "weak":
                victims = [c for c in pos if not (np.isfinite(D["ma"][50][c, t]) and C[c, t] >= D["ma"][50][c, t])]
            else:
                victims = []
            p_sell.update(victims)
            if paper:
                sell_why.update({c: "regime" for c in victims})
            p_add.clear()
            p_buy = []
            eq_prev = e
            continue
        free = max_pos - len(pos) + (len(p_sell) if entry == "open" else 0)
        if free > 0:
            cs = np.flatnonzero(cand[:, t])
            cs = [c for c in cs if c not in pos]
            rk = D[cfg["rank_score"]] if cfg.get("rank_score") else D["rs"]     # 32 RL-1: leader_P_RL 은 예측 R 순(없으면 RS — 기존)
            cs.sort(key=lambda c: -np.nan_to_num(rk[c, t], nan=-np.inf))
            p_buy = cs[:free]
        if entry == "close":                         # 종가 단일가 체결(신호는 같은 종가로 판정 — 15:19 대용 아님, 낙관 쪽 근사)
            for ci in p_buy:
                cl, pc = C[ci, t], C[ci, t - 1]
                if not (np.isfinite(cl) and cl > 0 and D["TR"][ci, t]) or (np.isfinite(pc) and cl >= pc * (LU if lu_arr is None else lu_arr[t])):
                    continue
                uw = min(pos_cap, cfg.get("risk", RISK) / sdist[ci, t]) if struct else unit
                amt = min(uw * e, cash)
                if amt <= 0:
                    continue
                pos[ci] = {"qty": amt / cl, "cost_basis": amt, "first_px": cl, "units": 1, "entry_t": t, "last_px": cl, "w_peak": 0.0}
                if struct:                              # 위험 비중: 첫 단위 = min(상한, 위험 / 손절 거리), 추가 단위도 같은 금액, 종목당 상한까지(최대 3)
                    pos[ci].update(stop=low10[ci, t], sdist=float(sdist[ci, t]), uw=uw, mu=max(1, min(3, int(pos_cap / uw + 1e-9))))
                cash -= amt
            p_buy = []
        eq_prev = e
    if state is not None:                          # 모의매매: 마지막 날 보유·다음날 주문을 넘긴다(청산하지 않음)
        state.update(pos={D["codes"][c]: {**p, "w": p["qty"] * p["last_px"] / eq_prev} for c, p in pos.items()}, cash=cash,
                     equity=eq_prev, buy=[D["codes"][c] for c in p_buy], sell=[D["codes"][c] for c in p_sell],
                     add=[D["codes"][c] for c in p_add], regime_on=bool(D["regime_on"][t1 - 1]), last_day=str(days[t1 - 1].date()))
        out_eq = pd.Series(eq, index=days[t0:t1])
        out_eq.attrs["expo"] = expo
        return out_eq, pd.DataFrame(trades)
    for ci in list(pos):
        close_pos(ci, pos[ci]["last_px"], t1 - 1, "end")
    out_eq = pd.Series(eq, index=days[t0:t1])
    out_eq.attrs["expo"] = expo
    return out_eq, pd.DataFrame(trades)


def metrics(eq: pd.Series, lo=None, hi=None) -> dict:
    x = eq[(eq.index >= (lo or eq.index[0])) & (eq.index < (hi or eq.index[-1] + pd.Timedelta(days=1)))]
    x = x / x.iloc[0]
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
    cagr = x.iloc[-1] ** (1 / yrs) - 1
    mdd = float((x / x.cummax() - 1).min())
    r = x.resample("ME").last().pct_change().dropna()
    by, prev = {}, 1.0
    for y, g in x.groupby(x.index.year):
        by[int(y)] = float(g.iloc[-1] / prev - 1)
        prev = float(g.iloc[-1])
    return {"cagr": float(cagr), "mdd": mdd, "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan, "total": float(x.iloc[-1] - 1),
            "sharpe_m": float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan, "by_year": by}


def _init(cache, extras=False):
    _D.update(prepare(cache, extras))


def _task(a):
    i, cfg = a
    out = {"i": i}
    for cm in (1.0, 2.0):
        eq, tr = simulate(_D, cfg, cm)
        out[f"eq{int(cm)}"] = eq.to_numpy()
        out[f"tr{int(cm)}"] = tr
    out["idx"] = eq.index
    return out


def benchmarks(D: dict, start: str = START, end: str = END) -> pd.DataFrame:
    days, C = D["days"], D["C"]
    t0, t1 = int(days.searchsorted(pd.Timestamp(start))), int(days.searchsorted(pd.Timestamp(end), side="right"))
    r = C[:, 1:] / C[:, :-1] - 1
    r = np.concatenate([np.full((C.shape[0], 1), np.nan), r], axis=1)
    r = np.where(np.isfinite(r), np.clip(r, -0.3, 0.3), np.nan)
    uni = D["uni"]
    ew = np.nanmean(np.where(np.concatenate([np.zeros((C.shape[0], 1), bool), uni[:, :-1]], axis=1), r, np.nan), axis=0)
    # 대형주 대용: 직전 달 말 ADV20 상위 30 동일 비중(월 재조정)
    month = days.to_period("M")
    lc = np.full(len(days), np.nan)
    sel = None
    for t in range(t0, t1):
        if t == t0 or month[t] != month[t - 1]:
            a = np.nan_to_num(D["adv"][:, t - 1])
            sel = np.argsort(-a)[:30]
        lc[t] = np.nanmean(r[sel, t])
    idx = days[t0:t1]
    return pd.DataFrame({"market_cw": (1 + pd.Series(D["cw"][t0:t1], index=idx)).cumprod(),
                         "market_ew": (1 + pd.Series(np.nan_to_num(ew[t0:t1]), index=idx)).cumprod(),
                         "largecap30": (1 + pd.Series(np.nan_to_num(lc[t0:t1]), index=idx)).cumprod()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    run(GRID, ap.parse_args())


def run(grid: dict, a, extras: bool = False, fixed: "dict | None" = None) -> dict:
    """격자 실행 · 확장 워크포워드 · 널 · 기준선(21·22세대 공통). ``fixed`` = 모든 설정에 덮어쓸 고정 유전자."""
    keys = list(grid)
    cfgs = [{**(fixed or {}), **dict(zip(keys, v, strict=True))} for v in itertools.product(*grid.values())]
    if a.limit:
        cfgs = cfgs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    with cf.ProcessPoolExecutor(min(a.workers, len(cfgs)), initializer=_init, initargs=(a.cache, extras)) as ex:
        res = sorted(ex.map(_task, list(enumerate(cfgs))), key=lambda r: r["i"])
    idx = res[0]["idx"]
    E1 = pd.DataFrame({i: r["eq1"] for i, r in enumerate(res)}, index=idx)
    E2 = pd.DataFrame({i: r["eq2"] for i, r in enumerate(res)}, index=idx)
    E1.to_pickle(os.path.join(a.out, "eq1.pkl"))
    E2.to_pickle(os.path.join(a.out, "eq2.pkl"))
    for cm in (1, 2):
        pd.concat([r[f"tr{cm}"].assign(cfg=r["i"]) for r in res]).to_csv(os.path.join(a.out, f"trades{cm}.csv.gz"), index=False)
    _init(a.cache, extras)
    B = benchmarks(_D)
    B.to_csv(os.path.join(a.out, "bench.csv"))
    rng = np.random.default_rng(SEED)
    sel, parts1, parts2 = [], [], []
    for lo, te0, te1 in FOLDS:
        sc = {i: metrics(E1[i], lo, te0)["calmar"] for i in E1.columns}
        best = max(sc, key=lambda i: -np.inf if not np.isfinite(sc[i]) else sc[i])
        sel.append(best)
        for E, parts in ((E1, parts1), (E2, parts2)):
            seg = E[best][(E.index >= te0) & (E.index < te1)]
            parts.append(seg / seg.iloc[0])
    def stitch(parts):
        out, base = [], 1.0
        for p in parts:
            out.append(p * base)
            base = float(out[-1].iloc[-1])
        return pd.concat(out)
    oos1, oos2 = stitch(parts1), stitch(parts2)
    def oos_of(i, E):
        ps = []
        for (lo, te0, te1) in FOLDS:
            seg = E[i][(E.index >= te0) & (E.index < te1)]
            ps.append(seg / seg.iloc[0])
        return ps
    null = []
    for _ in range(3000):
        ps = [oos_of(int(rng.integers(0, len(cfgs))), E2)[k] for k in range(len(FOLDS))]
        null.append(metrics(stitch(ps))["cagr"])
    m2 = metrics(oos2)
    Bo = B[(B.index >= "2020-01-01") & (B.index < "2024-01-01")]
    bm = {k: metrics(Bo[k]) for k in Bo.columns}
    summ = {"selected": [cfgs[i] for i in sel], "oos_1x": metrics(oos1), "oos_2x": m2, "bench_oos": bm,
            "null_cagr_median": float(np.median(null)), "null_cagr_p90": float(np.quantile(null, 0.9)), "p_null": float(np.mean(np.array(null) >= m2["cagr"])),
            "grid_oos_2x": {str(i): metrics(stitch(oos_of(i, E2)))["cagr"] for i in range(len(cfgs))},
            "full_2017_2026_2x": {str(i): metrics(E2[i])["cagr"] for i in range(len(cfgs))},
            "contaminated_2024_2026": {"selected_last": metrics(E2[sel[-1]], "2024-01-01", None), "bench": {k: metrics(B[k], "2024-01-01", None) for k in B.columns}}}
    summ["pass"] = bool(m2["cagr"] > max(bm["market_ew"]["cagr"], bm["largecap30"]["cagr"], bm["market_cw"]["cagr"]) and m2["mdd"] > bm["market_ew"]["mdd"]
                        and summ["p_null"] <= 0.10 and sum(v > 0 for v in m2["by_year"].values()) >= 3)
    pd.DataFrame({"oos_1x": oos1, "oos_2x": oos2}).to_csv(os.path.join(a.out, "oos.csv"))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("grid_oos_2x", "full_2017_2026_2x")}, ensure_ascii=False, indent=1, default=float))
    return summ


if __name__ == "__main__":
    main()
