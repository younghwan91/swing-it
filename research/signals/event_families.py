"""13세대 사건 가족 — P(가격·거래대금)·C(공시) 사건 표와 유전자 → 체결 후보(``Engine.run`` 형식).

설계·사전등록: ``research/logs/sepa_explore/EVENT_FAMILIES.md`` (+ 정정 E-1, 검증 15차). 이 파일은 결과 전 해석을 담는다.

**표(신호 행 s — 그날 종가까지):** 사건 행은 기존 쐐기 표(``F``·``I``) 뒤에 붙는다(표 번호 :data:`TID_P`·:data:`TID_C`). ``I["d"]`` =
s + 1(다음날 주문 행), ``i0`` = s. 가족 서술자는 ``XF``(행 = 가족 표 행, 열 :data:`X_COLS`).

- P: 맥락(위생 ∧ ADV20 ≥ 10억 ∧ MA50 > MA150 > MA200 ∧ 분할 60일 아님) ∧ 거래대금(s)/평균(s−20..s−1) ≥ 2 ∧ 등락률 ≥ 2% ∧ 종가 위치 ≥ 0.5 ∧
  C(s) ≥ 직전 60일 최고 고가 — 유전자 범위의 상위 집합.
- C: DART 공급계약(``catalysts.CATALYST_RULES['supply_contract']``)·잠정실적("(잠정)실적") 접수일 r → s = r 이상 첫 거래일, 맥락 ∧ 정정 아님.

**체결(유전자 → 후보):**

- P 종가 진입(``p_entry`` = close): 주문 행 = s, e = C(s) × (1 + 미끄러짐), 거래대금 배수는 × 0.85(동시호가 몫 근사), 상한가 종가 제외,
  수급은 s−1 값. 손절 = max(L(s) − 1호가, e × (1 − stop_max)), 2% 하한. ``X_ENTRY_CLOSE``·``X_ON_HOLD`` = ``p_hold``.
- s+1 시가(open): 시가 ≥ 전일 종가 × 상한가 비 → 버림, 시가 > C(s) × (1 + gap_max) → 버림, e = 시가 × (1 + 미끄러짐).
- s+1 매수정지(stop): 주문가 = H(s) + 1호가, 고가 ≥ 주문가일 때 체결가 = max(시가, 주문가), 시가 > 주문가 × (1 + gap_max) → 버림.
  진입일 모호 = 시가 < 주문가 ∧ 저가 ≤ 손절 ∧ 종가 > 손절(쐐기와 같다).
- s+1 진입의 손절 = max(L(s) − 1호가, e × (1 − stop_max)), 2% 하한(s+1 시가 손절 아래면 버림). 수급은 s 값(장 마감 뒤 확정 → s+1 사용).
- 공통 거르기: EPS 최신 분기 > 전년 같은 분기 ∧ > 0(사용자 고정), 시장 국면 유전자(쐐기와 같은 정의), 수급 하한(모르면 통과).
- 같은 날 순서: W → C → P(사전 고정), 가족 안 = 거래대금 배수 큰 순(강도).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals import pattern_events as PE

TID_P, TID_C = PE.N_TABLES, PE.N_TABLES + 1
N_FAM_TABLES = 2
FAM_ORDER = {"W": 0, "C": 1, "P": 2}

X_COLS = ("fam", "tvx", "ret", "clv", "hi60", "hi250", "lu_close", "fl5_s", "fl20_s", "fl5_p", "fl20_p",
          "c_contract", "c_prelim", "ret_m1", "clv_m1", "hi60_m1", "hi250_m1", "c_n",
          "s_kind", "s_lvl", "s_stp")                      # 16세대 원천 S(event_sources.py) — v11 XF 는 앞 18 열만
XI = {k: i for i, k in enumerate(X_COLS)}
P_MIN = {"tvx": 2.0, "ret": 0.02, "clv": 0.5}
PRELIM_PAT = "(잠정)실적"


def price_descriptors(O, H, L, C, TV, s_idx, c_idx, lu_ratio: float) -> dict:
    """(종목 c, 행 s) 의 P 서술자 — 배열은 종목×날짜. 종가 1% 낮춘 판정(정정 E-1 4)도 함께."""
    n = len(s_idx)
    out = {k: np.full(n, np.nan) for k in ("tvx", "ret", "clv", "hi60", "hi250", "lu_close", "ret_m1", "clv_m1", "hi60_m1",
                                           "hi250_m1")}
    tv = np.nan_to_num(TV)
    cs = np.concatenate([np.zeros((tv.shape[0], 1)), np.cumsum(tv, axis=1)], axis=1)
    lo = np.maximum(s_idx - 20, 0)
    avg = (cs[c_idx, s_idx] - cs[c_idx, lo]) / np.maximum(s_idx - lo, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["tvx"] = np.where((s_idx >= 20) & (avg > 0), tv[c_idx, s_idx] / np.where(avg > 0, avg, 1.0), np.nan)
        cp = C[c_idx, s_idx - 1]
        c0 = C[c_idx, s_idx]
        h0, l0 = H[c_idx, s_idx], L[c_idx, s_idx]
        rng = h0 - l0
        for sfx, cc in (("", c0), ("_m1", c0 * 0.99)):
            out["ret" + sfx] = np.where(cp > 0, cc / np.where(cp > 0, cp, 1.0) - 1.0, np.nan)
            out["clv" + sfx] = np.where(rng > 0, (cc - l0) / np.where(rng > 0, rng, 1.0), np.where(cc >= cp, 1.0, 0.0))
        out["lu_close"] = (c0 >= cp * lu_ratio * 0.999).astype(float)
    for w in (60, 250):
        mx = np.full(n, np.nan)
        for j in range(n):                                  # 행 수가 수만이라 단순 반복(빌드 한 번)
            s, c = s_idx[j], c_idx[j]
            if s >= w:
                mx[j] = np.nanmax(H[c, s - w:s])
        out[f"hi{w}"] = (c0 >= mx).astype(float)
        out[f"hi{w}_m1"] = (c0 * 0.99 >= mx).astype(float)
    return out


def dart_kinds(dart: pd.DataFrame) -> pd.DataFrame:
    """DART 접수 목록 → (code, rcept_dt, contract, prelim) — 정정 공시 제외(``catalysts`` 규약)."""
    from swing_it.features.catalysts import _CORRECTION_PREFIX, classify_catalyst  # noqa: PLC0415
    nm = dart["report_nm"].fillna("")
    ok = (dart["is_correction"] == 0) & ~nm.str.startswith(_CORRECTION_PREFIX)
    con = nm.map(classify_catalyst).eq("supply_contract")
    pre = nm.str.contains(PRELIM_PAT, regex=False)
    d = dart.loc[ok & (con | pre), ["code", "rcept_dt"]].copy()
    d["contract"] = con[ok & (con | pre)].astype(int).to_numpy()
    d["prelim"] = pre[ok & (con | pre)].astype(int).to_numpy()
    return d


def gene_mask(XF: np.ndarray, Ff: np.ndarray, g: dict, fam: str, FI: dict) -> np.ndarray:
    """가족 표 행 중 유전자 거르기 통과(체결 전) — 신호 행 s 의 값만."""
    x = lambda k: XF[:, XI[k]]  # noqa: E731
    f = lambda k: Ff[:, FI[k]]  # noqa: E731
    with np.errstate(invalid="ignore"):
        eq, eqp = f("e_q"), f("e_q_prev")
        m = np.isfinite(eq) & np.isfinite(eqp) & (eq > 0) & (eq > eqp)            # 사용자 고정: EPS 전년 같은 분기 대비 상승·흑자
        mk = g.get("mkt", "off")
        if mk == "nhnl":
            m &= f("mkt_nhnl") > 0.5
        elif mk == "bos":
            m &= f("mkt_bos") > 0.6
        elif mk == "br200":
            m &= f("mkt_br200") > 0.5
        am = g.get("atr_max")
        if am is not None:
            cs_ = f("close_s")
            m &= ~(f("atr_s") / cs_ > am)
        if fam == "S":
            from research.signals.event_sources import gene_mask_s  # noqa: PLC0415
            m &= gene_mask_s(x, g)
        elif fam == "P":
            close_in = g.get("p_entry") == "close"
            tvx = x("tvx") * (0.85 if close_in else 1.0)
            m &= x("fam") == 0
            m &= (tvx >= g["p_vmult"]) & (x("ret") >= g["p_ret"]) & (x("clv") >= g["p_clv"])
            m &= x("hi250" if g["p_hiw"] == 250 else "hi60") > 0.5
            if close_in:
                m &= x("lu_close") < 0.5
            fm = g.get("p_flow")
            if fm is not None:
                fl = x("fl5_p") if close_in else x("fl5_s")
                m &= ~(fl < fm)                                                    # 모르면 통과
        else:
            m &= x("fam") == 1
            k = g.get("c_kind", "both")
            m &= (x("c_contract") > 0.5) if k == "contract" else (x("c_prelim") > 0.5) if k == "prelim" else \
                ((x("c_contract") > 0.5) | (x("c_prelim") > 0.5))
            if g.get("c_ret") is not None:
                m &= x("ret") >= g["c_ret"]
            if g.get("c_vmult") is not None:
                m &= x("tvx") >= g["c_vmult"]
            fm = g.get("c_flow")
            if fm is not None:
                m &= ~(x("fl5_s") < fm)
    return m


def fills(rows: np.ndarray, I_: np.ndarray, II: dict, Op, H, L, C, TR, entry: str, g: dict, slip: float, lu_ratio: float,
          mkt: np.ndarray, reform: int, lvl: "np.ndarray | None" = None, stp: "np.ndarray | None" = None) -> dict:
    """통과 행 → 체결(주문 행 d·e·sp·amb·skip0·s0). 버린 행은 빠진다. ``mkt`` = 그 행의 시장(0/1).
    ``lvl``·``stp`` = 매수정지 기준가·손절 기준가(16세대 S — 없으면 H(s)·L(s))."""
    ci = I_[rows, II["ci"]].astype(np.int64)
    s = I_[rows, II["i0"]].astype(np.int64)
    Hs = H[ci, s] if lvl is None else np.asarray(lvl, float)
    Ls = L[ci, s] if stp is None else np.asarray(stp, float)
    nd = Op.shape[1]
    smax = g.get("stop_max") or 0.05
    gap = g.get("gap_max")
    gap = np.inf if gap is None else gap
    tk_lo = PE.tick_dated_vec(Ls, s, mkt, reform)
    if entry == "close":
        d = s.copy()
        ok = TR[ci, s].astype(bool) & (s + 1 < nd)
        e = C[ci, s] * (1.0 + slip)
        amb = np.zeros(len(rows), bool)
        skip0 = np.ones(len(rows), bool)
    else:
        d = s + 1
        d_ = np.minimum(d, nd - 1)
        op, hi, lo, cl = Op[ci, d_], H[ci, d_], L[ci, d_], C[ci, d_]
        prev = C[ci, s]
        ok = (d < nd) & TR[ci, d_].astype(bool) & np.isfinite(op) & (op > 0)
        ok &= ~(op >= prev * lu_ratio)
        if entry == "open":
            ok &= ~(op > prev * (1.0 + gap))
            e = op * (1.0 + slip)
            amb = np.zeros(len(rows), bool)
            skip0 = np.zeros(len(rows), bool)
        else:
            hs = Hs
            order = hs + PE.tick_dated_vec(hs, d_, mkt, reform)
            ok &= (hi >= order) & ~(op > order * (1.0 + gap))
            f = np.maximum(op, order)
            e = f * (1.0 + slip)
            amb = np.zeros(len(rows), bool)       # 아래에서 손절 뒤 채운다
            skip0 = np.zeros(len(rows), bool)
    with np.errstate(invalid="ignore"):
        sp = np.maximum(Ls - tk_lo, e * (1.0 - smax))
        sp = np.minimum(sp, e * (1.0 - 0.02))
        ok &= np.isfinite(sp) & (sp > 0) & (sp < e)
        if entry != "close":
            d_ = np.minimum(d, nd - 1)
            ok &= Op[ci, d_] > sp                                                # 시가가 이미 손절 아래면 안 산다
            if entry == "stop":
                hs = Hs
                order = hs + PE.tick_dated_vec(hs, d_, mkt, reform)
                skip0 = (Op[ci, d_] < order) & (C[ci, d_] > sp)                      # 쐐기 규약: 장중 체결 ∧ 종가 > 손절 = 낙관 경로
                amb = skip0 & (L[ci, d_] <= sp)
    k = np.flatnonzero(ok)
    return {"rows": rows[k], "d": d[k], "ci": ci[k], "e": e[k], "sp": sp[k], "amb": amb[k], "skip0": skip0[k], "s0": sp[k]}
