"""16세대 사건 원천 S — 구조가 다른 진입 원천 7종(표 :data:`TID_S`, ``XF`` 행 fam = 2).

설계·사전등록: ``research/logs/sepa_explore/META_EVOLUTION.md`` 16세대(결과 전). 원천마다 '같은 메뉴 무작위 설정' 으로 먼저 재고,
바를 넘은 원천만 GA 에 넣는다.

**공통 맥락(base, 신호 행 s — 그날 종가까지):** P·C 와 같다 — 위생 ∧ ADV20 ≥ 하한 ∧ MA50 > MA150 > MA200 ∧ 분할 60일 아님 ∧ 거래
있음. EPS·시장 국면·ATR 은 :func:`event_families.gene_mask` 공통 거르기(EPS 필수).

원천(종류 번호 = ``s_kind``)과 주문 기준가 ``lvl``(매수정지 = lvl + 1호가)·손절 기준가 ``stp``(손절 = stp − 1호가, stop_max 상한):

0. ctx — 맥락 기준선(구조 널): base 인 날 중 (s + 7·종목) mod 10 = 0 표본. lvl = H(s), stp = L(s).
1. pb — 강한 추세의 MA20 눌림: C(s)/C(s−60) − 1 ≥ 0.20 ∧ C(s) ≥ 0.75 × 250일 최고 ∧ L(s) ≤ MA20 × 1.01 ∧ C(s) ≥ MA20 ∧
   C(s) ≤ 0.95 × 20일 최고 고가. lvl = H(s), stp = L(s).
2. tight — 신고가 근처 좁은 박스(VCP형) + 거래량 마름: C(s) ≥ 0.85 × 250일 최고 ∧ 10일 (최고 − 최저)/C ≤ 0.10 ∧
   10일 평균 거래량 / 50일 평균 ≤ 0.75 ∧ C(s)/C(s−60) − 1 ≥ 0.15. lvl = 10일 최고 고가, stp = 10일 최저 저가.
3. earn — 실적 공시 반응: DART (잠정)실적·매출액또는손익구조·분기/반기/사업보고서(정정 아님) 접수일 r, r0 = r 이상 첫 거래일,
   s ∈ {r0, r0 + 1} 중 첫 ( 시가 갭 ≥ 2% ∨ 등락 ≥ 3% ) ∧ 종가 위치 ≥ 0.5 ∧ 거래대금 배수 ≥ 1.5 인 날. lvl = H(s), stp = L(s).
4. accum — 기관+외국인 매집: 5일 수급 ≥ 0.15 ∧ 20일 수급 ≥ 0.05 ∧ C(s) ≥ 0.90 × 60일 최고 ∧ 5일 등락 ≤ +5% ∧ 전날은 아님(첫날).
   lvl = H(s), stp = 5일 최저 저가.
5. retest — 돌파 뒤 되돌림·재돌파: 돌파일 b ∈ [s−15, s−3] (P 상위 집합: 배수 ≥ 2 ∧ 등락 ≥ 2% ∧ 종가 위치 ≥ 0.5 ∧ C(b) ≥ 직전
   60일 최고 고가 m) ∧ min L(b+1..s) ≤ m × 1.03 ∧ C(s) > H(b) ∧ C(s−1) ≤ H(b). lvl = H(s), stp = min L(b+1..s).
6. corp — 주주환원·증설 공시: 자기주식취득(신탁 포함)·무상증자·신규시설투자(정정 아님), s = r0. lvl = H(s), stp = L(s).

같은 날 순서: 거래대금 배수 큰 순. 가격 조건 판정은 모두 s 종가까지.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals import pattern_events as PE

TID_S = PE.N_TABLES + 2
KINDS = ("ctx", "pb", "tight", "earn", "accum", "retest", "corp")
K = {k: i for i, k in enumerate(KINDS)}
S_COLS = ("s_kind", "s_lvl", "s_stp")
EARN_PAT = ("(잠정)실적", "매출액또는손익구조", "분기보고서", "반기보고서", "사업보고서")
CORP_PAT = ("자기주식취득결정", "자기주식취득신탁계약체결결정", "무상증자결정", "신규시설투자")


def _roll(a: np.ndarray, w: int, fn: str, shift: int = 0) -> np.ndarray:
    """종목×날짜 배열의 날짜 축 이동 창(행 s: [s−shift−w+1, s−shift])."""
    d = pd.DataFrame(a.T)
    if shift:
        d = d.shift(shift)
    r = d.rolling(w, min_periods=w)
    return getattr(r, fn)().to_numpy().T


def _dart_days(dart: pd.DataFrame, pats: tuple, days: pd.DatetimeIndex, cmap: dict) -> "tuple[np.ndarray, np.ndarray]":
    from swing_it.features.catalysts import _CORRECTION_PREFIX  # noqa: PLC0415
    nm = dart["report_nm"].fillna("")
    ok = (dart["is_correction"] == 0) & ~nm.str.startswith(_CORRECTION_PREFIX)
    hit = np.zeros(len(dart), bool)
    for p in pats:
        hit |= nm.str.contains(p, regex=False).to_numpy()
    d = dart.loc[ok.to_numpy() & hit, ["code", "rcept_dt"]]
    pos = days.searchsorted(pd.DatetimeIndex(d["rcept_dt"]), side="left")
    ci = d["code"].astype(str).map(cmap)
    m = (pos < len(days)) & ci.notna().to_numpy()
    u = pd.DataFrame({"ci": ci[m].astype(np.int64).to_numpy(), "s": pos[m]}).drop_duplicates()
    return u["ci"].to_numpy(np.int64), u["s"].to_numpy(np.int64)


def build(Op, H, L, C, V, TV, MA20, base, tvx, ret, clv, fl5, fl20, dart, days, codes) -> pd.DataFrame:
    """원천 사건 표 (ci, s, kind, lvl, stp) — 배열은 종목×날짜, ``fl5``·``fl20`` 은 종목×날짜 수급(s 확정값)."""
    nc, nd = C.shape
    cmap = {str(c): i for i, c in enumerate(codes)}
    out = []

    def add(mask, kind, lvl, stp):
        ci, s = np.nonzero(mask)
        out.append(pd.DataFrame({"ci": ci, "s": s, "kind": K[kind], "lvl": lvl[ci, s], "stp": stp[ci, s]}))

    with np.errstate(invalid="ignore", divide="ignore"):
        c60 = np.concatenate([np.full((nc, 60), np.nan), C[:, :-60]], axis=1)
        r60 = C / c60 - 1.0
        hi250 = _roll(H, 250, "max")
        hi20 = _roll(H, 20, "max")
        hi10, lo10 = _roll(H, 10, "max"), _roll(L, 10, "min")
        lo5 = _roll(L, 5, "min")
        v10, v50 = _roll(V, 10, "mean"), _roll(V, 50, "mean")
        mx60p = _roll(H, 60, "max", shift=1)                                    # 직전 60일(s 제외) 최고 고가
        hi60 = _roll(H, 60, "max")
        c5 = np.concatenate([np.full((nc, 5), np.nan), C[:, :-5]], axis=1)
        cp = np.concatenate([np.full((nc, 1), np.nan), C[:, :-1]], axis=1)
        gap = Op / cp - 1.0
        # 0 ctx
        samp = ((np.arange(nd)[None, :] + 7 * np.arange(nc)[:, None]) % 10) == 0
        add(base & samp, "ctx", H, L)
        # 1 pb
        add(base & (r60 >= 0.20) & (C >= 0.75 * hi250) & (L <= MA20 * 1.01) & (C >= MA20) & (C <= 0.95 * hi20), "pb", H, L)
        # 2 tight
        add(base & (C >= 0.85 * hi250) & ((hi10 - lo10) / C <= 0.10) & (v10 / v50 <= 0.75) & (r60 >= 0.15), "tight", hi10, lo10)
        # 4 accum(첫날)
        ac = (fl5 >= 0.15) & (fl20 >= 0.05) & (C >= 0.90 * hi60) & (C / c5 - 1.0 <= 0.05)
        ac_prev = np.concatenate([np.zeros((nc, 1), bool), ac[:, :-1]], axis=1)
        add(base & ac & ~ac_prev, "accum", H, lo5)
        # 5 retest
        brk = (tvx >= 2.0) & (ret >= 0.02) & (clv >= 0.5) & (C >= mx60p)
        bi, bs = np.nonzero(brk)
        rows = []
        for ci, b in zip(bi, bs, strict=True):
            m = mx60p[ci, b]
            hb = H[ci, b]
            lo_run = np.inf
            for s in range(b + 1, min(b + 16, nd)):
                lo_run = min(lo_run, L[ci, s])
                if s - b < 3:
                    continue
                if lo_run <= m * 1.03 and C[ci, s] > hb and C[ci, s - 1] <= hb and base[ci, s]:
                    rows.append((ci, s, lo_run))
                    break
        if rows:
            r_ = np.array(rows)
            ci_, s_ = r_[:, 0].astype(np.int64), r_[:, 1].astype(np.int64)
            out.append(pd.DataFrame({"ci": ci_, "s": s_, "kind": K["retest"], "lvl": H[ci_, s_], "stp": r_[:, 2]}))
        # 3 earn
        ci_, r0 = _dart_days(dart, EARN_PAT, days, cmap)
        ev = []
        for k in (0, 1):
            s = np.minimum(r0 + k, nd - 1)
            ok = base[ci_, s] & ((gap[ci_, s] >= 0.02) | (ret[ci_, s] >= 0.03)) & (clv[ci_, s] >= 0.5) & (tvx[ci_, s] >= 1.5)
            ev.append(pd.DataFrame({"ci": ci_, "r0": r0, "s": s, "k": k, "ok": ok}))
        ev = pd.concat(ev)
        ev = ev[ev["ok"]].sort_values(["ci", "r0", "k"]).drop_duplicates(["ci", "r0"]).drop_duplicates(["ci", "s"])
        ci_, s_ = ev["ci"].to_numpy(np.int64), ev["s"].to_numpy(np.int64)
        out.append(pd.DataFrame({"ci": ci_, "s": s_, "kind": K["earn"], "lvl": H[ci_, s_], "stp": L[ci_, s_]}))
        # 6 corp
        ci_, r0 = _dart_days(dart, CORP_PAT, days, cmap)
        ok = base[ci_, r0]
        ci_, s_ = ci_[ok], r0[ok]
        out.append(pd.DataFrame({"ci": ci_, "s": s_, "kind": K["corp"], "lvl": H[ci_, s_], "stp": L[ci_, s_]}))
    t = pd.concat(out, ignore_index=True)
    t = t[np.isfinite(t["lvl"]) & np.isfinite(t["stp"]) & (t["s"] >= 21) & (t["s"] < nd - 1)]
    return t.drop_duplicates(["ci", "s", "kind"]).reset_index(drop=True)


def gene_mask_s(x, g: dict) -> np.ndarray:
    """S 행 고유 거르기 — 종류 · 거래대금 배수 하한 · 수급 하한(모르면 통과). ``x(name)`` = XF 열."""
    with np.errstate(invalid="ignore"):
        m = (x("fam") == 2) & (x("s_kind") == int(g.get("s_kind", 0)))
        vm = g.get("s_vmult")
        if vm is not None:
            m &= x("tvx") >= vm
        fm = g.get("s_flow")
        if fm is not None:
            m &= ~(x("fl5_s") < fm)
    return m
