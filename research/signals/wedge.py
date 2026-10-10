"""천장 아래 하락 쐐기 — 검출(상단·하단 추세선) · 차트용 반전점 · 하락 기준선 널. 매수정지·청산은 :mod:`research.signals.vcp` 커널.

사전등록 정본: ``research/logs/sepa_explore/WEDGE_TEST.md`` — 7196cad + 정정 0 d1b7db8(사용자 그림) + 정정 0b 0a911e2 + 정정 1
7e99bdb(``REVIEW_WEDGE_TEST.md`` 1단계) + **정정 2 bf62569**(쐐기 = 마지막 다섯 점), 모두 성과 보기 전. 이 파일은 §2(하락 쐐기)·
정정 0·1·2 를 옮긴 계산 기계다 — 수치는
:data:`WEDGE_PARAMS` 에 고정하고 탐색하지 않는다. §1 단계 2 맥락은 패널이 필요해 러너(``wedge_test_run.py``)가 건다.
numba 는 선택(:mod:`vcp` 와 같은 규약).

**날짜 규약:** 셋업 행 ``s = d − 1`` 종가까지의 자료, 주문은 ``d`` 장중 매수정지(셋업이 유효한 날만, 매일 새 U(d)).

**사전등록 문구 → 구현 해석(모호한 곳은 전부 여기 — 결과 보기 전 결정):**

1. 천장 H0 = ``s−259..s`` 창 최고 고가(동률 = 늦은 봉), ``s − i0`` ∈ [15, 250]. 창이 다 차지 않거나 결측이면 셋업 없음.
2. 지그재그 = :func:`vcp._zigzag_py`(θ 5%, 극값 먼저·반전 확인은 다음 봉부터, 동률 = 늦은 봉, i0 의 H0 에서 시작).
3. **쐐기 = 마지막 다섯 점(정정 2):** D = 마지막 확정 고점, B = 그 앞 확정 고점, C = B–D 사이 확정 저점, A = B 앞 확정 저점,
   **E = D 뒤 ~ s 의 최저 저가(진행 중 후보 허용 — 확정을 기다리지 않고, 그 뒤 반등 중일 필요 없음; 동률 = 늦은 봉, 정정 1 C2).**
   H0 → A 구간(쐐기 전)은 제약 없음(창 최고라 모든 가격 < H0 는 자동). 확정 고점 ≥ 3 (H0 … B, D — B 가 H0 뒤에 있어야 한다).
4. 고점 하락 D ≤ B × 0.99. 저점 하락 A > C > E (직전 미만). — 다섯 점에만.
5. **모양(정정 1 M3, 정정 2 로 다섯 점에만):** A→B, B→C, C→D, D→E 간격 ≥ 3 거래일, A → s ≥ 15 거래일.
6. 수축은 쐐기 안에서만: c_1 = (B − C)/B, c_2 = c_last = (D − E)/D — c_2 < c_1, c_1 ∈ [5%, 30%], c_last ≤ 15%.
7. 상단선 = (B, D) 의 로그가격 직선, 하단선 = (C, E) 의 로그가격 직선(행 번호가 x). 두 기울기 < 0 ∧ 상단 < 하단.
   **W-nc(분해 팔, ``conv=False``):** c_k 감소와 기울기 순서(상단 < 하단)를 보지 않는다(하락 계단) — 나머지는 같다.
8. **A 닿음(정정 1 M2):** |A / 하단선(A) − 1| ≤ 3% (A = B 앞 확정 저점).
9. 천장 아래: 진행 중 고점 후보까지 < H0. 깊이 = (H0 − min(low[i0..s]))/H0 ∈ [10%, 40%].
10. E 뒤 ``s − 행(E)`` ≥ 2.
11. **첫 돌파만(정정 1 C2):** 행(E) 다음날 ~ s 에 고가 > U(그날) × 1.005 인 봉이 있으면 그 쐐기는 끝(셋업 없음).
12. **무결성(정정 1 M2):** A ~ s 의 종가가 U(그날) × 1.005 위나 하단선(그날) × 0.995 아래인 날이 없다.
13. **진짜 돌파 전(정정 1 M1):** 종가(s) < U(d) ∧ 종가(s) ≥ U(s) × 0.92.
14. 거래량 마름(정정 1 MINOR): mean(vol[행(D)+1 .. s]) ≤ 0.9 × mean(vol[i0−50 .. i0−1]) ("H0 이전 50일", H0 일 제외).
15. 손절 기준 = E(= D 뒤 ~ s 최저 저가). 쐐기 식별 = (종목, i0, 행(D)) — E 후보가 내려가도 같은 쐐기(첫 체결 하나).
16. 기술: H0 전 20일 급등률 = H0 / 종가(i0 − 20) − 1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.signals import vcp as V
from research.signals.vcp import _jit, prange

WEDGE_PARAMS = {
    "win": 260, "base_min": 15, "base_max": 250, "theta": 0.05,
    "high_drop": 0.99,     # H0 뒤 고점 열: 직전 × 0.99 이하
    "a_tol": 0.03,         # |A / 하단선(A) − 1| ≤ 3%
    "c1_lo": 0.05, "c1_hi": 0.30, "last_max": 0.15,   # 쐐기 안 수축
    "depth_lo": 0.10, "depth_hi": 0.40,
    "close_lo": 0.92,      # 종가(s) ≥ U(s) × 0.92
    "after_low": 2,        # E 뒤 ≥ 2 거래일
    "gap_min": 3,          # 연속 반전점 간격 ≥ 3 거래일
    "a_span": 15,          # A → s ≥ 15 거래일
    "break_tol": 0.005,    # 첫 돌파: 고가 > U × 1.005 면 이미 돌파 / 무결성: 종가 > U × 1.005
    "lower_tol": 0.005,    # 무결성: 종가 < 하단선 × 0.995
    "vol_win": 50, "vol_max": 0.9,
    "surge_win": 20,       # 기술: H0 전 20일 급등률
    "conv": True,          # False = W-nc(c_k 감소·기울기 순서 끔)
}
#: 쐐기 매수정지·청산 인자(:data:`vcp.BUYSTOP_PARAMS` 위에 덮는다) — 9.5%·3% 갭 기준 = 주문가, 돌파 실패 이틀 규칙(기준가 = U(d) 고정).
WEDGE_BUYSTOP = {"ref_order": True, "fb_days": 2}

(W_OK, W_NO_DATA, W_BASE, W_FEW, W_HIGHS, W_LOWS, W_SPACING, W_C1, W_NOT_DECR, W_LAST, W_SLOPE, W_A_TOUCH, W_DEPTH,
 W_RECENT, W_BROKEN, W_INTEGRITY, W_CLOSE, W_DRY) = range(18)
WEDGE_STATUS = ("ok", "no_data", "base_len", "highs<3", "highs_not_falling", "lows_not_falling", "spacing", "c1_range",
                "not_decreasing", "last>15%", "slope", "A_touch", "depth", "E<2d", "already_broken", "integrity",
                "close_band", "volume_dryup")
INT_KEYS = ("status", "i0", "n_highs", "i_low", "i_high", "n_contr", "i_a", "e_confirmed")
FLT_KEYS = ("h0", "u_s", "u_d", "slope_u", "slope_l", "floor_low", "c1", "c_last", "depth", "vol_ratio", "low_d", "surge20",
            "a_ratio")
OUT_KEYS = INT_KEYS + FLT_KEYS


def _wedge_one_py(H, L, C, V_, ci, t, win, bmin, bmax, theta, hdrop, atol, c1lo, c1hi, lastmax, dlo, dhi, closelo, aflow,
                  gapmin, aspan, btol, ltol, volwin, volmax, surgewin, conv, k, oi, of):
    """종목 ``ci`` 행 ``t``(= d−1) 까지 자료로 하락 쐐기 → ``oi[k, :]``(:data:`INT_KEYS`)·``of[k, :]``(:data:`FLT_KEYS`) 를 채운다.
    규칙은 모듈 설명. ``status`` 는 처음 걸린 조건(:data:`WEDGE_STATUS`), 실패해도 그때까지 계산한 값은 채운다."""
    oi[k, 0] = W_NO_DATA
    s0 = t - win + 1
    if s0 < 0:
        return
    i0 = -1
    h0 = -np.inf
    for j in range(s0, t + 1):
        hj = H[ci, j]
        if not (hj == hj and L[ci, j] == L[ci, j] and C[ci, j] == C[ci, j]):
            return
        if hj >= h0:
            h0 = hj
            i0 = j
    oi[k, 1] = i0
    of[k, 0] = h0
    if i0 - surgewin >= 0 and C[ci, i0 - surgewin] > 0:
        of[k, 11] = h0 / C[ci, i0 - surgewin] - 1.0
    base = t - i0
    if base < bmin or base > bmax:
        oi[k, 0] = W_BASE
        return
    cap = base + 3
    hv = np.empty(cap)
    hi_ = np.empty(cap, dtype=np.int64)
    lv = np.empty(cap)
    li_ = np.empty(cap, dtype=np.int64)
    nh, nl, up, hc, ihc, lc, ilc = V._zigzag(H, L, ci, i0, t, theta, hv, hi_, lv, li_)
    oi[k, 2] = nh
    if nh < 3:                                         # H0 … B·D (B 앞 저점 A, B–D 사이 저점 C)
        oi[k, 0] = W_FEW
        return
    iD = hi_[nh - 1]
    D = hv[nh - 1]
    E = np.inf
    iE = -1
    for j in range(iD + 1, t + 1):                     # E = D 뒤 최저 저가(진행 중 후보 허용, 동률 = 늦은 봉)
        if L[ci, j] <= E:
            E = L[ci, j]
            iE = j
    if iE < 0:
        oi[k, 0] = W_FEW
        return
    oi[k, 3] = iE
    oi[k, 4] = iD
    oi[k, 5] = 2
    oi[k, 7] = 1 if (nl == nh and li_[nl - 1] == iE) else 0
    of[k, 5] = E
    iB = hi_[nh - 2]
    B = hv[nh - 2]
    iC = li_[nh - 2]
    Cl = lv[nh - 2]
    iA = li_[nh - 3]
    Al = lv[nh - 3]
    oi[k, 6] = iA
    if not D <= B * hdrop:                             # 다섯 점에만(정정 2)
        oi[k, 0] = W_HIGHS
        return
    if not (Cl < Al and E < Cl):                       # A > C > E
        oi[k, 0] = W_LOWS
        return
    if iB - iA < gapmin or iC - iB < gapmin or iD - iC < gapmin or iE - iD < gapmin or t - iA < aspan:
        oi[k, 0] = W_SPACING
        return
    c1 = (B - Cl) / B
    clast = (D - E) / D
    of[k, 6] = c1
    of[k, 7] = clast
    if c1 < c1lo or c1 > c1hi:
        oi[k, 0] = W_C1
        return
    if conv and not clast < c1:
        oi[k, 0] = W_NOT_DECR
        return
    if clast > lastmax:
        oi[k, 0] = W_LAST
        return
    su = (math.log(D) - math.log(B)) / (iD - iB)
    sl = (math.log(E) - math.log(Cl)) / (iE - iC)
    lnD = math.log(D)
    lnE = math.log(E)
    u_s = math.exp(lnD + su * (t - iD))
    u_d = math.exp(lnD + su * (t + 1 - iD))
    of[k, 1] = u_s
    of[k, 2] = u_d
    of[k, 3] = su
    of[k, 4] = sl
    of[k, 10] = math.exp(lnE + sl * (t + 1 - iE))
    if not (su < 0 and sl < 0 and ((not conv) or su < sl)):
        oi[k, 0] = W_SLOPE
        return
    a_ratio = Al / math.exp(lnE + sl * (iA - iE))
    of[k, 12] = a_ratio
    if abs(a_ratio - 1.0) > atol:
        oi[k, 0] = W_A_TOUCH
        return
    mn = np.inf
    for j in range(i0, t + 1):
        if L[ci, j] < mn:
            mn = L[ci, j]
    depth = (h0 - mn) / h0
    of[k, 8] = depth
    if not (hc < h0) or depth < dlo or depth > dhi:
        oi[k, 0] = W_DEPTH
        return
    if t - iE < aflow:
        oi[k, 0] = W_RECENT
        return
    for j in range(iE + 1, t + 1):                     # 첫 돌파만: E 뒤 이미 상단선 × 1.005 위 고가가 있으면 끝
        if H[ci, j] > math.exp(lnD + su * (j - iD)) * (1.0 + btol):
            oi[k, 0] = W_BROKEN
            return
    for j in range(iA, t + 1):                         # 무결성: A ~ s 종가가 상단선 × 1.005 위 · 하단선 × 0.995 아래 없음
        cj = C[ci, j]
        if cj > math.exp(lnD + su * (j - iD)) * (1.0 + btol) or cj < math.exp(lnE + sl * (j - iE)) * (1.0 - ltol):
            oi[k, 0] = W_INTEGRITY
            return
    ct = C[ci, t]
    if not (ct < u_d and ct >= closelo * u_s):
        oi[k, 0] = W_CLOSE
        return
    if i0 - volwin < 0:
        oi[k, 0] = W_DRY
        return
    vs = 0.0
    for j in range(iD + 1, t + 1):
        vs += V_[ci, j]
    vb = 0.0
    for j in range(i0 - volwin, i0):
        vb += V_[ci, j]
    vr = np.nan
    if vb > 0 and vs == vs:
        vr = (vs / (t - iD)) / (vb / volwin)
    of[k, 9] = vr
    if not vr <= volmax:
        oi[k, 0] = W_DRY
        return
    oi[k, 0] = W_OK


_wedge_one = _jit(_wedge_one_py)


def _wedge_batch_py(H, L, C, V_, cis, ts, win, bmin, bmax, theta, hdrop, atol, c1lo, c1hi, lastmax, dlo, dhi, closelo,
                    aflow, gapmin, aspan, btol, ltol, volwin, volmax, surgewin, conv, oi, of):
    for k in prange(len(cis)):
        _wedge_one(H, L, C, V_, cis[k], ts[k], win, bmin, bmax, theta, hdrop, atol, c1lo, c1hi, lastmax, dlo, dhi, closelo,
                   aflow, gapmin, aspan, btol, ltol, volwin, volmax, surgewin, conv, k, oi, of)


_wedge_batch = _jit(_wedge_batch_py, parallel=True)


def _args(p: dict) -> tuple:
    return (int(p["win"]), int(p["base_min"]), int(p["base_max"]), float(p["theta"]), float(p["high_drop"]),
            float(p["a_tol"]), float(p["c1_lo"]), float(p["c1_hi"]), float(p["last_max"]), float(p["depth_lo"]),
            float(p["depth_hi"]), float(p["close_lo"]), int(p["after_low"]), int(p["gap_min"]), int(p["a_span"]),
            float(p["break_tol"]), float(p["lower_tol"]), int(p["vol_win"]), float(p["vol_max"]), int(p["surge_win"]),
            bool(p["conv"]))


def wedge_scan_arrays(H: np.ndarray, L: np.ndarray, C: np.ndarray, Vv: np.ndarray, cis, ts,
                      params: "dict | None" = None) -> dict:
    """종목×날짜 배열에서 (종목 ``cis[i]``, 행 ``ts[i]``) 마다 하락 쐐기(행 ``ts`` 까지 자료). → :data:`OUT_KEYS` 배열 dict."""
    p = {**WEDGE_PARAMS, **(params or {})}
    m = len(cis)
    oi = np.full((m, len(INT_KEYS)), -1, np.int64)
    of = np.full((m, len(FLT_KEYS)), np.nan)
    oi[:, 2] = 0
    oi[:, 5] = 0
    oi[:, 7] = 0
    if m:
        _wedge_batch(H, L, C, Vv, np.asarray(cis, np.int64), np.asarray(ts, np.int64), *_args(p), oi, of)
    out = {kk: oi[:, i] for i, kk in enumerate(INT_KEYS)}
    out.update({kk: of[:, i] for i, kk in enumerate(FLT_KEYS)})
    return out


def wedge_one(H, L, C, Vv, ci: int, t: int, params: "dict | None" = None) -> dict:
    """한 행 — 순수 파이썬 경로(테스트·참조용)."""
    p = {**WEDGE_PARAMS, **(params or {})}
    oi = np.full((1, len(INT_KEYS)), -1, np.int64)
    of = np.full((1, len(FLT_KEYS)), np.nan)
    oi[0, 2] = 0
    oi[0, 5] = 0
    oi[0, 7] = 0
    _wedge_one_py(H, L, C, Vv, ci, t, *_args(p), 0, oi, of)
    return {**{kk: oi[0, i] for i, kk in enumerate(INT_KEYS)}, **{kk: of[0, i] for i, kk in enumerate(FLT_KEYS)}}


def wedge_id(i0, i_high) -> np.ndarray:
    """쐐기 식별 정수 = i0 × 100000 + 행(D) (같은 H0·같은 마지막 확정 고점 → 같은 쐐기; E 후보가 내려가도 같다)."""
    return np.asarray(i0, np.int64) * 100_000 + np.asarray(i_high, np.int64)


def wedge_points(H: np.ndarray, L: np.ndarray, ci: int, t: int, params: "dict | None" = None) -> dict:
    """차트용 — H0·A…D(확정)·E(D 뒤 최저, 진행 중 후보 포함)·상단선/하단선 끝점(d 까지)·U(d) (행 번호와 가격). 성과 정보 없음."""
    p = {**WEDGE_PARAMS, **(params or {})}
    s0 = t - p["win"] + 1
    w = H[ci, s0:t + 1]
    i0 = s0 + (len(w) - 1 - int(np.argmax(w[::-1])))
    cap = t - i0 + 3
    hv, hi_ = np.empty(cap), np.empty(cap, dtype=np.int64)
    lv, li_ = np.empty(cap), np.empty(cap, dtype=np.int64)
    nh, nl, up, hc, ihc, lc, ilc = V._zigzag_py(H, L, ci, i0, t, p["theta"], hv, hi_, lv, li_)
    pts = [{"row": int(i0), "price": float(H[ci, i0]), "kind": "H0"}]
    for q in range(nh - 1):                            # 쐐기 전 반전점은 l/h, 마지막 네 점은 A·B·C·D(정정 2)
        lk = {nh - 3: "A", nh - 2: "C"}.get(q, "l")
        hk = {nh - 3: "B", nh - 2: "D"}.get(q, "h")
        pts.append({"row": int(li_[q]), "price": float(lv[q]), "kind": lk})
        pts.append({"row": int(hi_[q + 1]), "price": float(hv[q + 1]), "kind": hk})
    out = {"i0": int(i0), "h0": float(H[ci, i0]), "points": pts}
    if nh >= 2:
        iD = int(hi_[nh - 1])
        seg = L[ci, iD + 1:t + 1]
        if len(seg):
            iE = iD + 1 + (len(seg) - 1 - int(np.argmin(seg[::-1])))
            confirmed = bool(nl == nh and li_[nl - 1] == iE)
            pts.append({"row": int(iE), "price": float(L[ci, iE]), "kind": "E", "confirmed": confirmed})
            if nh >= 3:
                iB, B, D, E = int(hi_[nh - 2]), float(hv[nh - 2]), float(hv[nh - 1]), float(L[ci, iE])
                iC, Cl = int(li_[nh - 2]), float(lv[nh - 2])
                su = (math.log(D) - math.log(B)) / (iD - iB)
                sl = (math.log(E) - math.log(Cl)) / (iE - iC)
                ud = math.exp(math.log(D) + su * (t + 1 - iD))
                out["upper"] = {"from": {"row": iB, "price": B}, "to": {"row": t + 1, "price": ud}}
                iA = int(li_[nh - 3])
                out["lower"] = {"from": {"row": iA, "price": math.exp(math.log(E) + sl * (iA - iE))},
                                "to": {"row": t + 1, "price": math.exp(math.log(E) + sl * (t + 1 - iE))}}
                out["u_d"] = ud
    return out


def wedge_setups(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
                 mask: "pd.DataFrame | None" = None, params: "dict | None" = None) -> pd.DataFrame:
    """편의 — 패널에서 (종목, 주문일 d) 셋업 표(신호 행 s = d−1 까지 자료). 맥락(§1)은 러너가 건다."""
    H, L, C, Vv = (V._arr(x, close) for x in (high, low, close, volume))
    mk = (np.ones(close.shape, bool) if mask is None
          else mask.reindex(index=close.index, columns=close.columns).fillna(False).to_numpy(bool).copy())
    mk[-1, :] = False
    si, ci = np.nonzero(mk)
    r = wedge_scan_arrays(H, L, C, Vv, ci, si, params)
    df = pd.DataFrame({"code": close.columns[ci], "event_day": close.index[si], "order_day": close.index[si + 1],
                       "sig_idx": si, "day_idx": si + 1, "col_idx": ci, **r})
    df["valid"] = df["status"] == W_OK
    return df.sort_values(["order_day", "code"]).reset_index(drop=True)


def wedge_events(S: pd.DataFrame) -> pd.DataFrame:
    """셋업 표 → :func:`vcp.simulate_buystop` 사건 열(``pivot`` = U(d) 돌파 기준가, ``l_last`` = E, ``ip`` = 쐐기 식별)."""
    return S.assign(pivot=S["u_d"], l_last=S["floor_low"], ip=wedge_id(S["i0"], S["i_high"]))


def simulate_wedge(A: "V.Arrays", S: pd.DataFrame, *, fb_days: int = 2, no_overlap: bool = True,
                   params: "dict | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """쐐기 매수정지(§3~§5·정정 1): 주문가 = U(d) + 1호가, 체결 = max(시가, 주문가), 시가 > 주문가 × 1.03 버림, (주문가 − 손절가)/주문가
    > 9.5% 주문 없음, 손절가 = E − 1호가(2% 하한), 같은 쐐기 첫 체결 하나, 보유 중 무시, 라이브 청산 + 돌파 실패 ``fb_days`` 연속 종가
    < U(d)(진입일 종가가 첫날) → 다음날 시가. 진입일 모호성은 :mod:`vcp` 와 같은 두 경로."""
    return V.simulate_buystop(A, wedge_events(S), no_overlap=no_overlap,
                              params={**WEDGE_BUYSTOP, "fb_days": fb_days, **(params or {})})


# ---------------------------------------------------------------------------
# 같은 메뉴 널(정정 1 M5) — 하락 기준선 매수정지
# ---------------------------------------------------------------------------

def _null_line_py(H, L, C, cis, ss, ks, js, theta, closelo, btol, gapmin, ok, ud, ll, pid):
    """(종목, 신호 행 s, k, j) → 직전 k 일 창의 하락 기준선: 창 최고 고가(늦은 동률)에서 지그재그(θ)를 시작해 **마지막 두 확정 고점**을 잇는
    로그 직선(기울기 < 0, 두 고점 간격 ≥ gapmin). E' = 마지막 확정 고점 뒤 최저 저가. 첫 관통만(E' 뒤 고가 > 선 × 1.005 없음),
    종가(s) < 선(d) ∧ ≥ 선(s) × 0.92. 손절 기준 = 직전 j 일 최저 저가. 통과면 ``ok``·선(d)·손절 기준·식별(마지막 고점 행)."""
    for i in prange(len(cis)):
        ok[i] = False
        ci = cis[i]
        s = ss[i]
        k = ks[i]
        j = js[i]
        w0 = s - k + 1
        if w0 < 0 or s - j + 1 < 0:
            continue
        iw = -1
        hw = -np.inf
        bad = False
        for q in range(w0, s + 1):
            hq = H[ci, q]
            if not (hq == hq and L[ci, q] == L[ci, q]):
                bad = True
                break
            if hq >= hw:
                hw = hq
                iw = q
        if bad:
            continue
        cap = s - iw + 3
        hv = np.empty(cap)
        hi_ = np.empty(cap, dtype=np.int64)
        lv = np.empty(cap)
        li_ = np.empty(cap, dtype=np.int64)
        nh, nl, up, hc, ihc, lc, ilc = V._zigzag(H, L, ci, iw, s, theta, hv, hi_, lv, li_)
        if nh < 2:
            continue
        i1, h1, i2, h2 = hi_[nh - 2], hv[nh - 2], hi_[nh - 1], hv[nh - 1]
        if i2 - i1 < gapmin or not h2 < h1:
            continue
        su = (math.log(h2) - math.log(h1)) / (i2 - i1)
        e = np.inf
        ie = -1
        for q in range(i2 + 1, s + 1):
            if L[ci, q] <= e:
                e = L[ci, q]
                ie = q
        if ie < 0:
            continue
        broken = False
        for q in range(ie + 1, s + 1):
            if H[ci, q] > math.exp(math.log(h2) + su * (q - i2)) * (1.0 + btol):
                broken = True
                break
        if broken:
            continue
        us = math.exp(math.log(h2) + su * (s - i2))
        u_d = math.exp(math.log(h2) + su * (s + 1 - i2))
        cs = C[ci, s]
        if not (cs < u_d and cs >= closelo * us):
            continue
        lo = np.inf
        for q in range(s - j + 1, s + 1):
            if L[ci, q] < lo:
                lo = L[ci, q]
        ok[i] = True
        ud[i] = u_d
        ll[i] = lo
        pid[i] = i2


_null_line = _jit(_null_line_py, parallel=True)
_null_line_serial = _jit(_null_line_py)


def null_lines(A: "V.Arrays", cis, ss, ks, js, *, parallel: bool = True, params: "dict | None" = None) -> dict:
    """하락 기준선 널 셋업(:func:`_null_line_py`). → ``ok, u_d, l_last, pid`` 배열."""
    p = {**WEDGE_PARAMS, **(params or {})}
    m = len(cis)
    o = {"ok": np.zeros(m, np.bool_), "u_d": np.full(m, np.nan), "l_last": np.full(m, np.nan),
         "pid": np.full(m, -1, np.int64)}
    if m:
        (_null_line if parallel else _null_line_serial)(
            A.H, A.L, A.C, np.asarray(cis, np.int64), np.asarray(ss, np.int64), np.asarray(ks, np.int64),
            np.asarray(js, np.int64), float(p["theta"]), float(p["close_lo"]), float(p["break_tol"]), int(p["gap_min"]),
            o["ok"], o["u_d"], o["l_last"], o["pid"])
    return o


def _depth_py(L, cis, i0s, ts, out):
    for i in prange(len(cis)):
        mn = np.inf
        for j in range(i0s[i], ts[i] + 1):
            if L[cis[i], j] < mn:
                mn = L[cis[i], j]
        out[i] = mn


_depth = _jit(_depth_py, parallel=True)


def min_low_since(L: np.ndarray, cis, i0s, ts) -> np.ndarray:
    """(종목, i0, s) 마다 min(low[i0..s]) — 널 풀의 깊이 띠용."""
    m = len(cis)
    out = np.full(m, np.nan)
    if m:
        _depth(L, np.asarray(cis, np.int64), np.asarray(i0s, np.int64), np.asarray(ts, np.int64), out)
    return out


# ---------------------------------------------------------------------------
# 쐐기 D(``WEDGE_D_TEST.md`` 2154f77 + 정정 1 b008fce, 탐색) — A = H0, 네 점 A·B·C·D 연속, 바닥(D 봉 고가) 반전 매수정지
# ---------------------------------------------------------------------------
#
# 사전등록 문구 → 구현 해석(결과 보기 전 결정):
#  a. A = H0(창 260 최고 고가, 동률 늦은 봉, 15~250일 앞). 지그재그(θ 5%, :func:`vcp._zigzag_py`)를 H0 에서 시작.
#  b. **네 점 연속:** 확정 고점이 정확히 둘(A = H0, C) — B = 첫 확정 저점(A 와 C 사이), C = 마지막 확정 고점. A–B, B–C 사이에 다른
#     반전점이 있으면 확정 고점이 셋 이상이 되므로 셋업 없음. D = C 뒤 ~ s 최저 저가(진행 중, 확정 불필요; 동률 = 늦은 봉).
#  c. C ≤ A × 0.99, **D ≤ B × 0.99**(정정 1 M1 — 평평한 하단 = 내림 삼각형 제외). c_1 = (A − B)/A ∈ [8%, 40%], c_2 = (C − D)/C ≤ 15% ∧ c_2 < c_1(수렴, WD-nc 는 끔).
#  d. 상단선 A–C·하단선 B–D 로그 직선, 둘 다 기울기 < 0 ∧ 상단 < 하단(WD-nc 는 순서 끔). 간격 A→B·B→C·C→D ≥ 3, A → s ≥ 15.
#  e. 무결성: A ~ s 종가 ≤ 상단선 × 1.005 ∧ ≥ 하단선 × 0.97. 거래량: mean(vol[C+1..s]) ≤ 0.9 × mean(vol[i0−50..i0−1]).
#  f. 진입(WD): D 봉 고가 = 매수정지 기준(주문가 = 그 + 1호가 — 호가는 래퍼가 붙인다), d − 행(D) ≤ 5, 종가(s) < 주문가(래퍼).
#     **두 번째 관통 차단(정정 1 M3):** 행(D) 다음날 ~ s 에 고가 ≥ 주문가인 봉이 있으면 그 D 로는 사지 않는다(래퍼, ``consumed_D``).
#     새 D(더 낮은 저가)가 생기면 그 D 봉 이후로 다시 잰다(재무장). "바닥 근처" 의 실제 뜻 = 최근 신저가 봉 고가 돌파.
#     손절·본전 체결가 = min(시가, 손절선) − 1호가(정정 1 M5, :data:`WD_BUYSTOP` ``sell_tick``, −0.1% 대신).
#  g. WD-up 진입(같은 모양): 쐐기 시험 W 방식 — D 뒤 고가 > 상단선 × 1.005 인 봉이 없고(첫 돌파), 종가(s) < U(d) ∧ ≥ U(s) × 0.92.
#     d − 행(D) ≤ 5 는 걸지 않는다(돌파 진입이라 바닥 근처 조건이 아니다). ``up_ok`` 로 따로 낸다.
#  h. 기술: H0 전 상승률(저가 기준) = H0 / min(low[i0−w .. i0−1]) − 1, w = 20·40·60.

WD_PARAMS = {
    "win": 260, "base_min": 15, "base_max": 250, "theta": 0.05, "high_drop": 0.99,
    "c1_lo": 0.08, "c1_hi": 0.40, "last_max": 0.15, "gap_min": 3, "a_span": 15,
    "upper_tol": 0.005, "lower_tol": 0.03, "vol_win": 50, "vol_max": 0.9,
    "low_drop": 0.99,      # D ≤ B × 0.99 (정정 1 M1)
    "near_d": 5,           # d − 행(D) ≤ 5
    "up_close_lo": 0.92,   # WD-up: 종가 ≥ U(s) × 0.92
    "conv": True,
}
#: 바닥 진입 매수정지·청산 — 9.5%·3% 갭 기준 = 주문가, 2R 전 돌파 실패 규칙 없음(fb_days 를 사실상 무한으로).
WD_BUYSTOP = {"ref_order": True, "fb_days": 1_000_000_000, "sell_slip": 0.0, "sell_tick": True}

(D_OK, D_NO_DATA, D_BASE, D_SHAPE, D_HIGHS, D_LOWS, D_SPACING, D_C1, D_NOT_DECR, D_LAST, D_SLOPE, D_INTEGRITY, D_DRY,
 D_FAR, D_CONSUMED, D_CLOSE) = range(16)
WD_STATUS = ("ok", "no_data", "base_len", "not_four_points", "C>A*0.99", "D>B*0.99", "spacing", "c1_range", "not_decreasing",
             "c2>15%", "slope", "integrity", "volume_dryup", "d-iD>5", "consumed_D", "close>=order")
WD_INT = ("status", "i0", "i_b", "i_c", "i_d", "up_ok", "d_confirmed")
WD_FLT = ("h0", "b", "c", "dlow", "d_high", "u_s", "u_d", "slope_u", "slope_l", "c1", "c2", "vol_ratio", "low_d",
          "rise20", "rise40", "rise60", "maxh_after_d")


def _wd_one_py(H, L, C, V_, ci, t, win, bmin, bmax, theta, hdrop, ldrop, c1lo, c1hi, lastmax, gapmin, aspan, utol, ltol,
               volwin, volmax, near_d, up_close_lo, conv, k, oi, of):
    """종목 ``ci`` 행 ``t`` 까지 자료로 쐐기 D 셋업 → ``oi[k]``(:data:`WD_INT`)·``of[k]``(:data:`WD_FLT`). 주문가(호가)·종가 < 주문가는
    래퍼(:func:`wd_scan_arrays`)가 건다. 모양을 통과하면 ``up_ok``(WD-up 진입 조건)를 진입 상태와 무관하게 채운다."""
    oi[k, 0] = D_NO_DATA
    s0 = t - win + 1
    if s0 < 0:
        return
    i0 = -1
    h0 = -np.inf
    for j in range(s0, t + 1):
        hj = H[ci, j]
        if not (hj == hj and L[ci, j] == L[ci, j] and C[ci, j] == C[ci, j]):
            return
        if hj >= h0:
            h0 = hj
            i0 = j
    oi[k, 1] = i0
    of[k, 0] = h0
    for q, w in enumerate((20, 40, 60)):
        if i0 - w >= 0:
            mn = np.inf
            for j in range(i0 - w, i0):
                if L[ci, j] < mn:
                    mn = L[ci, j]
            if mn > 0:
                of[k, 13 + q] = h0 / mn - 1.0
    base = t - i0
    if base < bmin or base > bmax:
        oi[k, 0] = D_BASE
        return
    cap = base + 3
    hv = np.empty(cap)
    hi_ = np.empty(cap, dtype=np.int64)
    lv = np.empty(cap)
    li_ = np.empty(cap, dtype=np.int64)
    nh, nl, up, hc, ihc, lc, ilc = V._zigzag(H, L, ci, i0, t, theta, hv, hi_, lv, li_)
    if nh != 2 or nl < 1:                              # 확정 고점 = A(H0)·C 정확히 둘, 그 사이 확정 저점 B
        oi[k, 0] = D_SHAPE
        return
    iB, B = li_[0], lv[0]
    iC, Cc = hi_[1], hv[1]
    D = np.inf
    iD = -1
    for j in range(iC + 1, t + 1):
        if L[ci, j] <= D:
            D = L[ci, j]
            iD = j
    if iD < 0:
        oi[k, 0] = D_SHAPE
        return
    oi[k, 2] = iB
    oi[k, 3] = iC
    oi[k, 4] = iD
    oi[k, 6] = 1 if (nl >= 2 and li_[1] == iD) else 0
    of[k, 1] = B
    of[k, 2] = Cc
    of[k, 3] = D
    of[k, 4] = H[ci, iD]
    mh = -np.inf
    for j in range(iD + 1, t + 1):
        if H[ci, j] > mh:
            mh = H[ci, j]
    of[k, 16] = mh
    if not Cc <= h0 * hdrop:
        oi[k, 0] = D_HIGHS
        return
    if not D <= B * ldrop:
        oi[k, 0] = D_LOWS
        return
    if iB - i0 < gapmin or iC - iB < gapmin or iD - iC < gapmin or t - i0 < aspan:
        oi[k, 0] = D_SPACING
        return
    c1 = (h0 - B) / h0
    c2 = (Cc - D) / Cc
    of[k, 9] = c1
    of[k, 10] = c2
    if c1 < c1lo or c1 > c1hi:
        oi[k, 0] = D_C1
        return
    if conv and not c2 < c1:
        oi[k, 0] = D_NOT_DECR
        return
    if c2 > lastmax:
        oi[k, 0] = D_LAST
        return
    lnA = math.log(h0)
    lnD = math.log(D)
    su = (math.log(Cc) - lnA) / (iC - i0)
    sl = (lnD - math.log(B)) / (iD - iB)
    of[k, 7] = su
    of[k, 8] = sl
    of[k, 5] = math.exp(lnA + su * (t - i0))
    of[k, 6] = math.exp(lnA + su * (t + 1 - i0))
    of[k, 12] = math.exp(lnD + sl * (t + 1 - iD))
    if not (su < 0 and sl < 0 and ((not conv) or su < sl)):
        oi[k, 0] = D_SLOPE
        return
    for j in range(i0, t + 1):
        cj = C[ci, j]
        if cj > math.exp(lnA + su * (j - i0)) * (1.0 + utol) or cj < math.exp(lnD + sl * (j - iD)) * (1.0 - ltol):
            oi[k, 0] = D_INTEGRITY
            return
    if i0 - volwin < 0:
        oi[k, 0] = D_DRY
        return
    vs = 0.0
    for j in range(iC + 1, t + 1):
        vs += V_[ci, j]
    vb = 0.0
    for j in range(i0 - volwin, i0):
        vb += V_[ci, j]
    vr = np.nan
    if vb > 0 and vs == vs:
        vr = (vs / (t - iC)) / (vb / volwin)
    of[k, 11] = vr
    if not vr <= volmax:
        oi[k, 0] = D_DRY
        return
    # WD-up 진입 조건(모양 통과 뒤, 바닥 진입 조건과 무관)
    upok = True
    for j in range(iD + 1, t + 1):
        if H[ci, j] > math.exp(lnA + su * (j - i0)) * (1.0 + utol):
            upok = False
            break
    ct = C[ci, t]
    if not (ct < of[k, 6] and ct >= up_close_lo * of[k, 5]):
        upok = False
    oi[k, 5] = 1 if upok else 0
    if t + 1 - iD > near_d:
        oi[k, 0] = D_FAR
        return
    oi[k, 0] = D_OK


_wd_one = _jit(_wd_one_py)


def _wd_batch_py(H, L, C, V_, cis, ts, win, bmin, bmax, theta, hdrop, ldrop, c1lo, c1hi, lastmax, gapmin, aspan, utol, ltol,
                 volwin, volmax, near_d, up_close_lo, conv, oi, of):
    for k in prange(len(cis)):
        _wd_one(H, L, C, V_, cis[k], ts[k], win, bmin, bmax, theta, hdrop, ldrop, c1lo, c1hi, lastmax, gapmin, aspan, utol,
                ltol, volwin, volmax, near_d, up_close_lo, conv, k, oi, of)


_wd_batch = _jit(_wd_batch_py, parallel=True)


def _wd_args(p: dict) -> tuple:
    return (int(p["win"]), int(p["base_min"]), int(p["base_max"]), float(p["theta"]), float(p["high_drop"]),
            float(p["low_drop"]), float(p["c1_lo"]), float(p["c1_hi"]), float(p["last_max"]), int(p["gap_min"]),
            int(p["a_span"]), float(p["upper_tol"]), float(p["lower_tol"]), int(p["vol_win"]), float(p["vol_max"]),
            int(p["near_d"]), float(p["up_close_lo"]), bool(p["conv"]))


def _wd_finish(C: np.ndarray, cis, ts, oi: np.ndarray, of: np.ndarray) -> dict:
    out = {kk: oi[:, i] for i, kk in enumerate(WD_INT)}
    out.update({kk: of[:, i] for i, kk in enumerate(WD_FLT)})
    dh = out["d_high"]
    out["order_px"] = dh + V.tick_size_vec(dh)
    st = out["status"]
    cs = C[np.asarray(cis, np.int64), np.asarray(ts, np.int64)] if len(cis) else np.zeros(0)
    with np.errstate(invalid="ignore"):
        used = (st == D_OK) & (out["maxh_after_d"] >= out["order_px"])   # 두 번째 관통 차단(정정 1 M3)
        st[used] = D_CONSUMED
        bad = (st == D_OK) & ~(cs < out["order_px"])          # 종가(s) < 주문가(D 봉 고가 + 1호가)
    st[bad] = D_CLOSE
    return out


def wd_scan_arrays(H: np.ndarray, L: np.ndarray, C: np.ndarray, Vv: np.ndarray, cis, ts,
                   params: "dict | None" = None) -> dict:
    """쐐기 D 셋업 — :data:`WD_INT` + :data:`WD_FLT` + ``order_px``(D 봉 고가 + 1호가)."""
    p = {**WD_PARAMS, **(params or {})}
    m = len(cis)
    oi = np.full((m, len(WD_INT)), -1, np.int64)
    of = np.full((m, len(WD_FLT)), np.nan)
    oi[:, 5] = 0
    oi[:, 6] = 0
    if m:
        _wd_batch(H, L, C, Vv, np.asarray(cis, np.int64), np.asarray(ts, np.int64), *_wd_args(p), oi, of)
    return _wd_finish(C, cis, ts, oi, of)


def wd_one(H, L, C, Vv, ci: int, t: int, params: "dict | None" = None) -> dict:
    """한 행 — 순수 파이썬 경로(테스트·참조용)."""
    p = {**WD_PARAMS, **(params or {})}
    oi = np.full((1, len(WD_INT)), -1, np.int64)
    of = np.full((1, len(WD_FLT)), np.nan)
    oi[0, 5] = 0
    oi[0, 6] = 0
    _wd_one_py(H, L, C, Vv, ci, t, *_wd_args(p), 0, oi, of)
    r = _wd_finish(C, [ci], [t], oi, of)
    return {kk: v[0] for kk, v in r.items()}


def wd_id(i0, i_c) -> np.ndarray:
    """쐐기 D 식별 = i0 × 100000 + 행(C) — 같은 A·B·C(D 가 내려가도 같은 쐐기, 첫 체결 하나)."""
    return np.asarray(i0, np.int64) * 100_000 + np.asarray(i_c, np.int64)


def wd_events(S: pd.DataFrame, *, up: bool = False) -> pd.DataFrame:
    """셋업 표 → :func:`vcp.simulate_buystop` 사건 열. 바닥 진입: ``pivot`` = D 봉 고가(주문가 = + 1호가), ``l_last`` = D.
    ``up=True`` (WD-up): ``pivot`` = U(d)(상단선 A–C 연장), ``l_last`` = D."""
    piv = S["u_d"] if up else S["d_high"]
    return S.assign(pivot=piv, l_last=S["dlow"], ip=wd_id(S["i0"], S["i_c"]))


def wd_points(H: np.ndarray, L: np.ndarray, ci: int, t: int, i_b: int, i_c: int, i_d: int) -> dict:
    """차트용 — A(H0)·B·C·D 와 상단선 A–C·하단선 B–D 의 d 까지 끝점(행·가격)."""
    s0 = t - WD_PARAMS["win"] + 1
    w = H[ci, s0:t + 1]
    i0 = s0 + (len(w) - 1 - int(np.argmax(w[::-1])))
    A, B, Cc, D = float(H[ci, i0]), float(L[ci, i_b]), float(H[ci, i_c]), float(L[ci, i_d])
    su = (math.log(Cc) - math.log(A)) / (i_c - i0)
    sl = (math.log(D) - math.log(B)) / (i_d - i_b)
    return {"points": [{"row": int(i0), "price": A, "kind": "A"}, {"row": int(i_b), "price": B, "kind": "B"},
                       {"row": int(i_c), "price": Cc, "kind": "C"}, {"row": int(i_d), "price": D, "kind": "D"}],
            "upper": {"from": {"row": int(i0), "price": A}, "to": {"row": t + 1, "price": math.exp(math.log(A) + su * (t + 1 - i0))}},
            "lower": {"from": {"row": int(i_b), "price": B}, "to": {"row": t + 1, "price": math.exp(math.log(D) + sl * (t + 1 - i_d))}}}


def _null_bottom_py(H, L, C, cis, ss, ks, pull_lo, pull_hi, near_d, ok, piv, ll, pid, mh):
    """WD 같은 메뉴 널(정정 1 M6): (종목, s, k) → 직전 k 일 최고 고가(동률 늦은 봉) 이후 ~ s 의 최저 저가(되돌림의 새 저점, 동률 늦은 봉)
    봉의 고가를 매수정지 기준으로. 조건: 눌림 1 − 저점/고점 ∈ [pull_lo, pull_hi], d − 행(저점) ≤ near_d. 손절 기준 = 그 저점.
    ``mh`` = 저점 봉 다음날 ~ s 최고 고가(두 번째 관통 차단은 래퍼). ``pid`` = 저점 행(어제가 최저면 s)."""
    for i in prange(len(cis)):
        ok[i] = False
        ci = cis[i]
        s = ss[i]
        k = ks[i]
        if s - k + 1 < 0:
            continue
        hk = -np.inf
        ih = -1
        bad = False
        for q in range(s - k + 1, s + 1):
            hq = H[ci, q]
            if not (hq == hq and L[ci, q] == L[ci, q]):
                bad = True
                break
            if hq >= hk:
                hk = hq
                ih = q
        if bad or ih >= s:
            continue
        lo = np.inf
        il = -1
        for q in range(ih + 1, s + 1):
            if L[ci, q] <= lo:
                lo = L[ci, q]
                il = q
        pull = 1.0 - lo / hk
        if pull < pull_lo or pull > pull_hi or s + 1 - il > near_d:
            continue
        m = -np.inf
        for q in range(il + 1, s + 1):
            if H[ci, q] > m:
                m = H[ci, q]
        ok[i] = True
        piv[i] = H[ci, il]
        ll[i] = lo
        pid[i] = il
        mh[i] = m


_null_bottom = _jit(_null_bottom_py, parallel=True)
_null_bottom_serial = _jit(_null_bottom_py)


def null_bottoms(A: "V.Arrays", cis, ss, ks, js=None, *, pull: float = 0.08, pull_hi: float = 0.40,
                 near_d: int = WD_PARAMS["near_d"], parallel: bool = True) -> dict:
    """WD 널 셋업(:func:`_null_bottom_py`) + 두 번째 관통 차단(저점 봉 뒤 고가 ≥ 주문가 없음) + 종가(s) < 주문가.
    → ``ok, pivot, l_last, pid, yday``(어제 저점이 그 되돌림 최저가 — 널 추출 우선순위). ``js`` 는 쓰지 않는다(옛 정의 호환)."""
    m = len(cis)
    o = {"ok": np.zeros(m, np.bool_), "pivot": np.full(m, np.nan), "l_last": np.full(m, np.nan),
         "pid": np.full(m, -1, np.int64), "yday": np.zeros(m, np.bool_)}
    if m:
        cis = np.asarray(cis, np.int64)
        ss = np.asarray(ss, np.int64)
        mh = np.full(m, np.nan)
        (_null_bottom if parallel else _null_bottom_serial)(
            A.H, A.L, A.C, cis, ss, np.asarray(ks, np.int64), float(pull), float(pull_hi), int(near_d),
            o["ok"], o["pivot"], o["l_last"], o["pid"], mh)
        order = o["pivot"] + V.tick_size_vec(o["pivot"])
        with np.errstate(invalid="ignore"):
            o["ok"] &= (A.C[cis, ss] < order) & ~(mh >= order)
        o["yday"] = o["ok"] & (o["pid"] == ss)
    return o
