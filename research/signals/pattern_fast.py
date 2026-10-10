"""패턴 GA 평가 커널 — 사건 표 거르기 · 매수정지 체결 · 확장 청산 경로 · 10 슬롯 포트폴리오(종목별 모멘텀 정합 바스켓 대조).

사전등록 정본: ``research/logs/sepa_explore/PATTERN_GA.md`` (원문 + 정정 0 + 정정 1 = ``REVIEW_PATTERN_GA.md`` 정정 1 초안 12항목·
MINOR 1~12 + 정정 2), 모두 결과 보기 전. 사건 표는 :mod:`research.signals.pattern_events`. numba 는 선택(:mod:`vcp` 규약).

**체결(사건 행 = 첫 관통일 d — 거르기는 전부 d−1 값):** 주문가 = 기준선(d) + 1호가, 시가 ≥ 전일 종가 × 상한가 비 → 버림,
시가 > 주문가 × (1 + ``gap_max``) → 버림(둘 다 소진), 손절 원값 = 기준(E 또는 하단선(d)) − 1호가, (주문가 − 손절 원값)/주문가 >
``stop_cap`` → 주문 없음(소진), 하단선 손절은 하단선(d) ≥ 종가(s) 면 무효(MINOR 4). 바닥 반전(정정 3)은 손절 = E − 1호가 고정·
익절 전 돌파 실패 없음. 정정 3 ``h0_in_pattern`` 은 사건의 패턴 종류와 정확히 같아야, ``ceil_kind`` = 수평 저항이면 저항 수 ≥ 2. 체결가 = max(시가, 주문가), 진입가 = 체결가 ×
(1 + 매수 미끄러짐), 손절폭 < 2% 면 손절가 = 진입가 × 0.98. 진입일 모호(시가 ≤ 기준선 ∧ 저가 ≤ 손절가 ∧ 종가 > 손절가)·확정 손절
(종가 ≤ 손절가)·진입일 익절 불인정은 :mod:`vcp` 와 같다.

**청산(정정 0·정정 1 M1 — 결과 보기 전 해석):** 하루 처리 순서 = 우선순위 표(리뷰 M1-c: 손절 > 본전 손절 > 익절 > 추가 > 돌파 실패 > 추적).

1. 전날 결정한 돌파 실패·추적 → 그날 시가 전량(``exit_open``).
2. 진입일: 낙관 경로(모호 거래)는 진입일 손절을 보지 않는다. 그 밖은 저가 ≤ 손절가 → 손절가 × (1 − 매도 미끄러짐).
3. 진입 다음날부터 손절선 = 원 손절 → (익절 다음날부터) 진입가 × ``be_mult``(1.0) → (``be_r`` 발동 다음날부터) 진입가 × 1.005 중 높은 것.
   시가 또는 저가 ≤ 손절선 → min(시가, 선) × (1 − 미끄러짐) 전량(추가분 포함 — 추가분 선은 이보다 높거나 같다).
4. 추가분 손절선 = max(진입가(본전), 그날 손절선)(정정 1 M1). 닿으면 추가분만 그 선(시가가 아래면 시가)에서 판다.
5. 부분 익절: 고가 ≥ 진입가 + ``take_r`` × R (R = 진입가 − 원 손절가) → ``take_frac`` 를 그 값(갭이면 시가)에 판다(그 시점 보유 전부의
   비율). 갭 익절일 저가 ≤ 본전선이면 그날 본전 전량(:mod:`vcp`). ``take_r`` = ∞ 면 없음.
6. 불타기(정정 0): 진입 다음날부터 고가 ≥ 진입가 + 1R(``pyr`` 1) 또는 고가 > H0(``pyr`` 2 — H0 기준선 진입에서는 러너가 금지)인 날
   → 다음 거래일 시가에 슬롯 자본의 1/3 을 산다(처음 진입은 2/3; 불타기 끔이면 100%). 그날 시가 ≤ 추가분 손절선이거나 그날 저가가
   추가분 손절선에 닿으면 취소(우선순위 손절 > 추가 — 그날 손절 구역을 건드린 추가는 하지 않는다). 진입일 고가는 발동에 쓰지
   않는다(체결과 순서 모름 — 진입일 익절 불인정과 같은 이유).
7. 돌파 실패(익절 전): 종가 < 기준가 × (1 − ``fb_tol``) 가 연속 ``fb_days`` 거래일(진입일 종가가 첫날, 정지일은 세지도 끊지도 않음)
   → 다음 거래일 시가 전량. ``fb_days`` = 0 이면 끔. 기준가(MINOR 5) = 추세선이면 U(d) 고정, 수평이면 D, H0 면 H0.
8. 추적(정정 1 M1 — **진입 다음날부터**, 익절 여부와 무관; ``from_entry`` = 0 은 라이브 동치 비교용 = 익절 뒤에만):
   MA20 / MA50 = 종가 < 그날 MA → 다음 거래일 시가. 샹들리에 = 종가 < 진입 뒤 최고 고가(진입일 고가 포함, 그날까지) − k × ATR20(전날 값)
   → 다음 거래일 시가. 조임(``tighten``): 최고 고가 ≥ 진입가 + 3R 인 날 다음날부터 k 를 절반.
9. 안전 상한 120 거래일째 종가(진입일 = 1일째, 조건 청산은 늘 있다 — 무기한 보유 방지), 데이터 끝(``LIM``) = 마지막 종가 ``data_end``.

**포트폴리오:** 10 슬롯 × 전일 자산의 10%(현금이 모자라면 남은 현금), 같은 날 신호 순서 = 사건 표 순서(ADV20 큰 순), 보유 중 같은 종목
무시, 그날 청산된 자리는 다음날부터, 현금 0%. 불타기 켬이면 슬롯의 1/3 은 추가 전까지 슬롯 안 현금. 진입일 모호 거래는 **포지션
안 혼합**: 진입 수량의 p 를 진입일 손절가로 팔고(보수 경로) 나머지 1 − p 가 낙관 경로를 따른다(슬롯 점유는 낙관 경로 — :mod:`vcp`
보유 거르기 규약과 같은 쪽). 비용 = ``cost_mult`` × ``korea.cost_rt``(그 묶음을 닫는 날) × 그 묶음의 매수 금액(부분 익절은 비용
없이 — 기존 ``slot_portfolio`` 와 같다).

**대조(정정 1 C4):** 그날 보유 종목마다 그 종목의 진입 신호일(s) 층 k = (시장, 12-1 모멘텀 10분위, ADV20 2분위)의 바스켓(전날 층
구성원, 자기 제외, 등가중) 그날 수익 × 그 포지션의 시작 비중(전일 종가 평가액 / 전일 자산)을 뺀다. 진입일·추가일 = 시가→종가,
시가 청산일 = 전일 종가→시가, 나머지 = 종가→종가. 바스켓이 비면 0(``unmatched`` 로 센다).
"""

from __future__ import annotations

import numpy as np

from research.signals.minervini_exits import BE_COST
from research.signals.pattern_events import FI
from research.signals.vcp import (
    R_BE,
    R_DATA_END,
    R_DELISTED,
    R_FB,
    R_STOP,
    R_STOP_ENTRY,
    R_TIME,
    R_TRAIL,
    _jit,
)

# --- 유전자 → 거르기 벡터(``gv``) 번호 ---
(G_STAGE, G_RS, G_BR, G_SUP_LO, G_SUP_HI, G_SLO_LO, G_SLO_HI, G_RHO, G_TOUCH, G_SPAN_LO, G_SPAN_HI, G_CEIL, G_DEPTH,
 G_DRY, G_RUNUP_COL, G_RUNUP, G_REST, G_EMIN, G_EACC, G_GAP, G_ANCHOR, G_CAP, G_FB_TOL, G_SLIP, G_DFLOOR, G_BASE,
 G_H0IN, G_CEILKIND) = range(28)
N_G = 28
# --- 청산 인자 벡터(``px``) 번호 ---
(X_TAKE_R, X_TAKE_FRAC, X_TRAIL, X_TRAIL_K, X_TIGHTEN, X_PYR, X_BE_R, X_FB_DAYS, X_MAX_HOLD, X_FROM_ENTRY, X_SLIP,
 X_SELL_SLIP, X_BE_MULT, X_TIGHT_R, X_PYR_R, X_WB) = range(16)
N_X = 16
TRAIL_MA20, TRAIL_MA50, TRAIL_CHAND = 0, 1, 2
PYR_OFF, PYR_1R, PYR_NEWHIGH = 0, 1, 2
# --- 경로 출력 번호 ---
(O_REASON, O_X, O_TP, O_PX1, O_TA, O_PA, O_TS, O_PXS, O_FIN, O_PNL, O_ADDED, O_MFE, O_EXIT_OPEN, O_WB, O_TF) = range(15)
N_O = 15
# --- 트레이드 기록 번호 ---
(Q_EV, Q_D, Q_X, Q_REASON, Q_E, Q_SP, Q_PNL, Q_WB, Q_AMB, Q_CX, Q_CD, Q_CADD, Q_ADDED, Q_MFE, Q_TP, Q_TA, Q_TS, Q_ALLOC,
 Q_PAR, Q_EXIT_OPEN, Q_TF, Q_PX1, Q_FIN, Q_PA, Q_PXS) = range(25)
N_Q = 25
#: 라이브 프로토콜(:data:`vcp.BUYSTOP_PARAMS` 기본) — 기준 청산 짝·동치 시험용.
SLOTS, SLOT_W = 10, 0.10
WB_PYR = 2.0 / 3.0                                    # 불타기 켬: 처음 2/3 · 추가 1/3(정정 0)
LIVE_X = np.array([2.0, 0.5, TRAIL_MA50, 3.0, 0, PYR_OFF, np.inf, 1, 120, 0, 0.001, 0.001, 1.0, 3.0, 1.0, WB_PYR])

F_LEVEL, F_ORDER, F_H0, F_SUP, F_SLO, F_RHO, F_TOUCH, F_SPAN = (FI["level"], FI["order"], FI["h0"], FI["s_up"], FI["s_lo"],
                                                               FI["rho"], FI["touch"], FI["span"])
F_CEIL, F_DEPTH, F_DRY, F_RUN20, F_REST = FI["ceil"], FI["depth"], FI["dry"], FI["runup20"], FI["rest_gap"]
F_LOWER, F_STOPE, F_STOPL, F_OPEN, F_HIGH, F_LOW, F_CLOSE, F_CLOSE_S = (FI["lower_d"], FI["stopE_raw"], FI["stopL_raw"],
                                                                       FI["open_d"], FI["high_d"], FI["low_d"],
                                                                       FI["close_d"], FI["close_s"])
F_RS, F_BR, F_ABOVE50, F_TTS, F_TTH0, F_EYOY, F_EPREV = (FI["rs"], FI["breadth"], FI["above50"], FI["tt_s"], FI["tt_h0"],
                                                         FI["e_yoy"], FI["e_yoy_prev"])
F_NRES, F_H0IN = FI["n_res"], FI["h0in"]
BASE_BOTTOM = 3


# ---------------------------------------------------------------------------
# 1. 거르기 + 체결
# ---------------------------------------------------------------------------

def _filter_py(F, a, b, gv, lu_ratio, oidx, oe, osp, ofb, oamb, oskip0):
    """표 행 ``[a, b)`` 를 유전자 ``gv`` 로 거르고 체결 재료를 채운다 → 통과 수. 모든 조건은 d−1 서술자(+ d 시가·체결 판정)."""
    n = 0
    stage = int(gv[G_STAGE])
    rs_min, br_min = gv[G_RS], gv[G_BR]
    su_lo, su_hi, sl_lo, sl_hi = gv[G_SUP_LO], gv[G_SUP_HI], gv[G_SLO_LO], gv[G_SLO_HI]
    rho_max, touch, sp_lo, sp_hi = gv[G_RHO], int(gv[G_TOUCH]), gv[G_SPAN_LO], gv[G_SPAN_HI]
    ceil_min, depth_max, dry_max = gv[G_CEIL], gv[G_DEPTH], gv[G_DRY]
    rcol, run_min, rest_max = F_RUN20 + int(gv[G_RUNUP_COL]), gv[G_RUNUP], gv[G_REST]
    emin, eacc = gv[G_EMIN], gv[G_EACC] > 0.5
    gap, anchor, cap, fbtol, slip, dfloor = gv[G_GAP], int(gv[G_ANCHOR]), gv[G_CAP], gv[G_FB_TOL], gv[G_SLIP], gv[G_DFLOOR]
    bottom = int(gv[G_BASE]) == BASE_BOTTOM
    if bottom:
        anchor = 0                                    # 정정 3: 바닥 반전 손절 = E − 1호가 고정
    h0in, ckind = gv[G_H0IN], int(gv[G_CEILKIND])
    for k in range(a, b):
        if h0in == h0in and F[k, F_H0IN] != h0in:
            continue
        if ckind == 1 and not F[k, F_NRES] >= 2:
            continue
        if stage == 1 and not F[k, F_ABOVE50] > 0.5:
            continue
        if stage == 2 and not F[k, F_TTS] > 0.5:
            continue
        if stage == 3 and not F[k, F_TTH0] > 0.5:
            continue
        if rs_min == rs_min and not F[k, F_RS] >= rs_min:
            continue
        if br_min == br_min and not F[k, F_BR] >= br_min:
            continue
        if su_hi == su_hi and not (F[k, F_SUP] >= su_lo and F[k, F_SUP] <= su_hi):
            continue
        if sl_hi == sl_hi and not (F[k, F_SLO] >= sl_lo and F[k, F_SLO] <= sl_hi):
            continue
        if rho_max == rho_max and not F[k, F_RHO] <= rho_max:
            continue
        if touch == 3 and not abs(F[k, F_TOUCH]) <= 0.03:
            continue
        if sp_hi == sp_hi and not (F[k, F_SPAN] >= sp_lo and F[k, F_SPAN] <= sp_hi):
            continue
        if ceil_min == ceil_min and not F[k, F_CEIL] >= ceil_min:
            continue
        if depth_max == depth_max and not F[k, F_DEPTH] <= depth_max:
            continue
        if dry_max == dry_max and not F[k, F_DRY] <= dry_max:
            continue
        if run_min == run_min and not F[k, rcol] >= run_min:
            continue
        if rest_max == rest_max and not F[k, F_REST] <= rest_max:
            continue
        if emin == emin:
            if not F[k, F_EYOY] >= emin:
                continue
            if eacc and not (F[k, F_EPREV] == F[k, F_EPREV] and F[k, F_EYOY] > F[k, F_EPREV]):
                continue
        level = F[k, F_LEVEL]
        order = F[k, F_ORDER]
        op = F[k, F_OPEN]
        prev = F[k, F_CLOSE_S]
        if not (op == op and order == order and order > 0):
            continue
        if prev == prev and prev > 0 and op >= prev * lu_ratio:
            continue
        if op > order * (1.0 + gap):
            continue
        if anchor == 1:
            lo_d = F[k, F_LOWER]
            if not (lo_d == lo_d and lo_d < prev):
                continue
            sr = F[k, F_STOPL]
        else:
            sr = F[k, F_STOPE]
        if not (sr == sr and sr > 0):
            continue
        if (order - sr) / order > cap:
            continue
        f = max(op, order)
        if sr >= f:
            continue
        e = f * (1.0 + slip)
        sp = sr
        if (e - sp) / e < dfloor:
            sp = e * (1.0 - dfloor)
        cl = F[k, F_CLOSE]
        open_ok = op <= level and cl > sp
        oidx[n] = k
        oe[n] = e
        osp[n] = sp
        ofb[n] = -np.inf if bottom else level * (1.0 - fbtol)       # 바닥 반전: 익절 전 돌파 실패 규칙 없음
        oskip0[n] = open_ok
        oamb[n] = open_ok and F[k, F_LOW] <= sp
        n += 1
    return n


filter_events = _jit(_filter_py)


# ---------------------------------------------------------------------------
# 2. 청산 경로(한 트레이드, 배분 1 단위)
# ---------------------------------------------------------------------------

def _path_py(Op, H, L, C, MA20, MA50, ATR, TR, LIM, ci, d, e, sp, fb_line, h0, skip0, px, out):
    """진입 행 ``d`` 진입가 ``e``·손절가 ``sp`` → ``out``(:data:`N_O`). 규칙은 모듈 설명 청산 1~9. 손익 ``O_PNL`` = 배분 1 당 총손익
    (비용 전, 낙관 경로; 불타기 켬이면 처음 ``wb`` 만 사고 1/3 은 추가 전까지 현금)."""
    n = Op.shape[1]
    lim = min(n, LIM[ci])
    Ln = lim - d
    max_hold = int(px[X_MAX_HOLD])
    k_last = max_hold - 1
    j_time = -1
    if k_last <= Ln - 1:
        for j in range(k_last, Ln):
            if TR[ci, d + j]:
                j_time = j
                break
    horizon = j_time if j_time >= 0 else Ln - 1
    sig_end = min(k_last, horizon + 1)
    ks = 1.0 - px[X_SELL_SLIP]
    slip = px[X_SLIP]
    R = e - sp
    take_r = px[X_TAKE_R]
    tgt = e + take_r * R if take_r < 1e18 else np.inf
    tf = px[X_TAKE_FRAC]
    be_r = px[X_BE_R]
    be_level = e + be_r * R if be_r < 1e18 else np.inf
    be_stop = e * (1.0 + BE_COST)
    be_line = e * px[X_BE_MULT]
    pyr = int(px[X_PYR])
    wb = px[X_WB] if pyr > 0 else 1.0
    wa = 1.0 - wb
    pyr_level = e + px[X_PYR_R] * R
    trail = int(px[X_TRAIL])
    kk = px[X_TRAIL_K]
    tighten = px[X_TIGHTEN] > 0.5
    tight_level = e + px[X_TIGHT_R] * R
    fb_days = int(px[X_FB_DAYS])
    from_entry = px[X_FROM_ENTRY] > 0.5
    for q in range(N_O):
        out[q] = np.nan
    out[O_TP] = -1
    out[O_TA] = -1
    out[O_TS] = -1
    out[O_ADDED] = 0
    out[O_EXIT_OPEN] = 0
    out[O_WB] = wb
    out[O_TF] = tf
    sb = wb / e
    sa = 0.0
    cash = -wb
    partial = False
    be_from = -1
    be_trig = -1
    pending = -1
    fb_run = 0
    last = 0
    add_state = 0                                     # 0 없음 · 1 다음날 시가 대기 · 2 보유 · 3 끝(손절·취소)
    hh = H[ci, d]
    mfe = H[ci, d]
    tight_j = -1
    for j in range(horizon + 1):
        t = d + j
        if not TR[ci, t]:
            continue
        last = j
        op = Op[ci, t]
        if pending >= 0:                              # 1. 전날 결정 → 시가 전량
            cash += (sb + sa) * op
            out[O_REASON] = pending
            out[O_X] = t
            out[O_FIN] = op
            out[O_EXIT_OPEN] = 1
            out[O_PNL] = cash
            out[O_MFE] = mfe
            return
        if H[ci, t] > mfe:
            mfe = H[ci, t]
        if j == 0:
            if (not skip0) and L[ci, t] <= sp:        # 2. 진입일 손절
                px_ = sp * ks
                cash += sb * px_
                out[O_REASON] = R_STOP_ENTRY
                out[O_X] = t
                out[O_FIN] = px_
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
        else:
            line = sp                                 # 3. 손절선
            why = R_STOP
            if partial and j >= be_from:
                line = be_line
                why = R_BE
            if be_trig >= 0 and j > be_trig and be_stop > line:
                line = be_stop
                why = R_BE
            if L[ci, t] <= line or op <= line:
                px_ = min(op, line) * ks
                cash += (sb + sa) * px_
                out[O_REASON] = why
                out[O_X] = t
                out[O_FIN] = px_
                out[O_EXIT_OPEN] = 1 if op <= line else 0
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
            al = max(e, line)                         # 4. 추가분 손절
            if add_state == 2 and (L[ci, t] <= al or op <= al):
                pxs = min(op, al) * ks
                cash += sa * pxs
                sa = 0.0
                add_state = 3
                out[O_TS] = t
                out[O_PXS] = pxs
            if (not partial) and (H[ci, t] >= tgt or op >= tgt):     # 5. 부분 익절
                g = op >= tgt
                p1 = op if g else tgt
                cash += tf * (sb + sa) * p1
                sb *= 1.0 - tf
                sa *= 1.0 - tf
                partial = True
                be_from = j + 1
                out[O_TP] = t
                out[O_PX1] = p1
                if g and L[ci, t] <= be_line:
                    cash += (sb + sa) * be_line * ks
                    out[O_REASON] = R_BE
                    out[O_X] = t
                    out[O_FIN] = be_line * ks
                    out[O_PNL] = cash
                    out[O_MFE] = mfe
                    return
            if add_state == 1:                        # 6. 추가(우선순위: 손절 > 익절 > 추가)
                if op > al and L[ci, t] > al:
                    pa = op * (1.0 + slip)
                    sa = wa / pa
                    cash -= wa
                    add_state = 2
                    out[O_TA] = t
                    out[O_PA] = pa
                    out[O_ADDED] = 1
                else:
                    add_state = 3
        if be_trig < 0 and (H[ci, t] >= be_level or op >= be_level):
            be_trig = j
        if add_state == 0 and pyr > 0 and j >= 1:
            if (pyr == PYR_1R and H[ci, t] >= pyr_level) or (pyr == PYR_NEWHIGH and H[ci, t] > h0):
                add_state = 1
        if H[ci, t] > hh:
            hh = H[ci, t]
        if tighten and tight_j < 0 and hh >= tight_level:
            tight_j = j
        if j == j_time:                               # 9. 안전 상한 종가
            cash += (sb + sa) * C[ci, t]
            out[O_REASON] = R_TIME
            out[O_X] = t
            out[O_FIN] = C[ci, t]
            out[O_PNL] = cash
            out[O_MFE] = mfe
            return
        if j >= sig_end:
            continue
        ct = C[ci, t]
        if fb_days > 0 and not partial:               # 7. 돌파 실패
            if ct < fb_line:
                fb_run += 1
                if fb_run >= fb_days:
                    pending = R_FB
            else:
                fb_run = 0
        act = (j >= 1) if from_entry else partial     # 8. 추적
        if act and pending < 0:
            if trail == TRAIL_CHAND:
                a = ATR[ci, t - 1] if t >= 1 else np.nan
                k_use = kk * 0.5 if (tight_j >= 0 and j > tight_j) else kk
                if a == a and ct < hh - k_use * a:
                    pending = R_TRAIL
            else:
                m = MA20[ci, t] if trail == TRAIL_MA20 else MA50[ci, t]
                if m == m and ct < m:
                    pending = R_TRAIL
    reason = R_DATA_END if (lim < n or d + last >= lim - 1) else R_DELISTED     # 끝까지 거래 = 데이터 끝, 앞서 끊김 = 상장폐지
    fin = C[ci, d + last]
    cash += (sb + sa) * fin
    out[O_REASON] = reason
    out[O_X] = d + last
    out[O_FIN] = fin
    out[O_PNL] = cash
    out[O_MFE] = mfe


exit_path = _jit(_path_py)


# ---------------------------------------------------------------------------
# 3. 포트폴리오
# ---------------------------------------------------------------------------

def _basket_py(bs, bn, strat_cs, Op, C, typ, t, k, ci):
    """층 ``k`` 바스켓의 그날 수익(전날 층 구성원, 자기 제외). 0 = 종가→종가, 1 = 시가→종가, 2 = 전일 종가→시가. → (값, 비었나)."""
    if k < 0 or t < 1:
        return 0.0, True
    s_ = bs[typ, t, k]
    n_ = bn[typ, t, k]
    if strat_cs[ci, t - 1] == k:
        if typ == 0:
            r = C[ci, t] / C[ci, t - 1] - 1.0
        elif typ == 1:
            r = C[ci, t] / Op[ci, t] - 1.0
        else:
            r = Op[ci, t] / C[ci, t - 1] - 1.0
        if r == r and abs(r) < 1e300:
            s_ -= r
            n_ -= 1
    if n_ <= 0:
        return 0.0, True
    return s_ / n_, False


basket = _jit(_basket_py)


def _portfolio_py(cd, cci, ce, csp, cfb, ch0, camb, cskip0, cstrat, cmkt, cpar, cev, cpid, PX, Op, H, L, C, MA20, MA50, ATR, TR,
                  LIM, bs, bn, strat_cs, cost_tab, cost_mult, pmix, t_lo, t_end, slots, slot_w, ret, bench, expo, rec,
                  year_of, ny_out):
    """후보(주문일 순·같은 날 우선순위 순)를 날마다 슬롯에 넣는다 — 들어간 트레이드만 그 자리에서 청산 경로를 계산한다(lazy).
    ``ret``·``bench``·``expo`` = 날 ``[t_lo, t_end)`` 의 포트폴리오 수익·정합 바스켓 대조·노출. ``rec`` = 들어간 트레이드 기록
    (:data:`N_Q`), ``ny_out`` = 진입 연도(``year_of`` 번호)별 진입 수. 반환 (진입 수, 슬롯·현금 부족 버림, 보유 중 버림, 대조 빈 날)."""
    n_c = len(cd)
    n_codes = Op.shape[0]
    held = np.zeros(n_codes, np.bool_)
    done_pid = np.full(n_codes, -1, np.int64)        # 같은 패턴(종목, i0, 행(D)) 체결 하나
    P_ON = np.zeros(slots, np.bool_)
    P_CI = np.zeros(slots, np.int64)
    P_K = np.zeros(slots, np.int64)
    P_M = np.zeros(slots, np.int64)
    P_REC = np.zeros(slots, np.int64)
    P_X = np.zeros(slots, np.int64)
    P_TP = np.zeros(slots, np.int64)
    P_TA = np.zeros(slots, np.int64)
    P_TS = np.zeros(slots, np.int64)
    P_EO = np.zeros(slots, np.bool_)
    P_F = np.zeros((slots, 12))                      # alloc, sb, sa, cash, last, px1, pa, pxs, fin, wa, tf, phi
    out = np.empty(N_O)
    eq_prev = 1.0
    cash = 1.0
    n_open = 0
    n_taken = 0
    sk_slot = 0
    sk_held = 0
    unm = 0
    ptr = 0
    while ptr < n_c and cd[ptr] < t_lo:
        ptr += 1
    for t in range(t_lo, t_end):
        bt = 0.0
        for q in range(slots):                        # 시작 비중 × 바스켓(전날까지 연 포지션)
            if not P_ON[q]:
                continue
            w = (P_F[q, 1] + P_F[q, 2]) * P_F[q, 4] / eq_prev
            typ = 2 if (P_X[q] == t and P_EO[q]) else 0
            b, emp = basket(bs, bn, strat_cs, Op, C, typ, t, P_K[q], P_CI[q])
            if emp:
                unm += 1
            bt += w * b
        while ptr < n_c and cd[ptr] == t:             # 진입
            i = ptr
            ptr += 1
            ci = cci[i]
            if n_open >= slots or cash <= 1e-12:
                sk_slot += 1
                continue
            if held[ci]:
                sk_held += 1
                continue
            if done_pid[ci] == cpid[i]:
                sk_held += 1
                continue
            q = 0
            while P_ON[q]:
                q += 1
            alloc = min(slot_w * eq_prev, cash)
            px = PX[cpar[i]]
            e = ce[i]
            sp = csp[i]
            exit_path(Op, H, L, C, MA20, MA50, ATR, TR, LIM, ci, t, e, sp, cfb[i], ch0[i], cskip0[i], px, out)
            wb = out[O_WB]
            m = cmkt[i]
            amb = camb[i]
            phi = 1.0 - pmix if amb else 1.0
            sb = wb * alloc / e
            pc = alloc - wb * alloc
            if amb:
                pc += pmix * sb * sp * (1.0 - px[X_SELL_SLIP]) - cost_mult * cost_tab[m, t] * pmix * wb * alloc
                sb *= phi
            P_ON[q] = True
            P_CI[q] = ci
            P_K[q] = cstrat[i]
            P_M[q] = m
            P_REC[q] = n_taken
            P_X[q] = int(out[O_X])
            P_TP[q] = int(out[O_TP])
            P_TA[q] = int(out[O_TA])
            P_TS[q] = int(out[O_TS])
            P_EO[q] = out[O_EXIT_OPEN] > 0.5
            P_F[q, 0] = alloc
            P_F[q, 1] = sb
            P_F[q, 2] = 0.0
            P_F[q, 3] = pc
            P_F[q, 4] = e
            P_F[q, 5] = out[O_PX1]
            P_F[q, 6] = out[O_PA]
            P_F[q, 7] = out[O_PXS]
            P_F[q, 8] = out[O_FIN]
            P_F[q, 9] = 1.0 - wb
            P_F[q, 10] = out[O_TF]
            P_F[q, 11] = phi
            b, emp = basket(bs, bn, strat_cs, Op, C, 1, t, cstrat[i], ci)
            if emp:
                unm += 1
            bt += wb * alloc / eq_prev * b
            r = rec[n_taken]
            r[Q_EV] = cev[i]
            r[Q_D] = t
            r[Q_X] = out[O_X]
            r[Q_REASON] = out[O_REASON]
            r[Q_E] = e
            r[Q_SP] = sp
            r[Q_PNL] = out[O_PNL]
            r[Q_WB] = wb
            r[Q_AMB] = 1.0 if amb else 0.0
            r[Q_CX] = cost_tab[m, int(out[O_X])]
            r[Q_CD] = cost_tab[m, t]
            r[Q_CADD] = cost_tab[m, int(out[O_TS])] if out[O_TS] >= 0 else cost_tab[m, int(out[O_X])]
            r[Q_ADDED] = out[O_ADDED]
            r[Q_MFE] = out[O_MFE]
            r[Q_TP] = out[O_TP]
            r[Q_TA] = out[O_TA]
            r[Q_TS] = out[O_TS]
            r[Q_ALLOC] = alloc / eq_prev
            r[Q_PAR] = cpar[i]
            r[Q_EXIT_OPEN] = out[O_EXIT_OPEN]
            r[Q_TF] = out[O_TF]
            r[Q_PX1] = out[O_PX1]
            r[Q_FIN] = out[O_FIN]
            r[Q_PA] = out[O_PA]
            r[Q_PXS] = out[O_PXS]
            ny_out[year_of[t]] += 1
            n_taken += 1
            held[ci] = True
            done_pid[ci] = cpid[i]
            n_open += 1
            cash -= alloc
        for q in range(slots):                        # 흐름: 추가분 손절 → 부분 익절 → 추가 → 청산
            if not P_ON[q]:
                continue
            m = P_M[q]
            alloc = P_F[q, 0]
            phi = P_F[q, 11]
            if P_TS[q] == t:
                P_F[q, 3] += P_F[q, 2] * P_F[q, 7] - cost_mult * cost_tab[m, t] * phi * P_F[q, 9] * alloc
                P_F[q, 2] = 0.0
            if P_TP[q] == t:
                tf = P_F[q, 10]
                P_F[q, 3] += tf * (P_F[q, 1] + P_F[q, 2]) * P_F[q, 5]
                P_F[q, 1] *= 1.0 - tf
                P_F[q, 2] *= 1.0 - tf
            if P_TA[q] == t:
                an = phi * P_F[q, 9] * alloc
                P_F[q, 2] = an / P_F[q, 6]
                P_F[q, 3] -= an
                b, emp = basket(bs, bn, strat_cs, Op, C, 1, t, P_K[q], P_CI[q])
                bt += an / eq_prev * b
            if P_X[q] == t:
                c_ = P_F[q, 3] + (P_F[q, 1] + P_F[q, 2]) * P_F[q, 8]
                wbq = 1.0 - P_F[q, 9]
                c_ -= cost_mult * cost_tab[m, t] * phi * wbq * alloc
                if P_TA[q] >= 0 and P_TS[q] < 0:
                    c_ -= cost_mult * cost_tab[m, t] * phi * P_F[q, 9] * alloc
                cash += c_
                P_ON[q] = False
                held[P_CI[q]] = False
                n_open -= 1
        val = cash
        inv = 0.0
        for q in range(slots):
            if not P_ON[q]:
                continue
            c = C[P_CI[q], t]
            if c == c and c > 0:
                P_F[q, 4] = c
            v = (P_F[q, 1] + P_F[q, 2]) * P_F[q, 4]
            val += P_F[q, 3] + v
            inv += v
        ret[t - t_lo] = val / eq_prev - 1.0
        bench[t - t_lo] = bt
        expo[t - t_lo] = inv / val if val > 0 else 0.0
        eq_prev = val
    return n_taken, sk_slot, sk_held, unm


portfolio = _jit(_portfolio_py)


# ---------------------------------------------------------------------------
# 4. 정합 바스켓 표(한 번)
# ---------------------------------------------------------------------------

def basket_tables(Op: np.ndarray, C: np.ndarray, strat_cs: np.ndarray, n_k: int) -> "tuple[np.ndarray, np.ndarray]":
    """층별 바스켓 합·수(``[유형, 날, 층]``) — 유형 0 종가→종가, 1 시가→종가, 2 전일 종가→시가. 구성원 = 전날(t−1) 층 = k 이고
    그날 수익이 유한한 종목. 배열은 종목×날짜."""
    n_c, n_d = C.shape
    bs = np.zeros((3, n_d, n_k))
    bn = np.zeros((3, n_d, n_k), np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rcc = C[:, 1:] / C[:, :-1] - 1.0
        roc = C[:, 1:] / Op[:, 1:] - 1.0
        rco = Op[:, 1:] / C[:, :-1] - 1.0
    k_prev = strat_cs[:, :-1]
    for typ, r in enumerate((rcc, roc, rco)):
        ok = (k_prev >= 0) & np.isfinite(r)
        ci, tt = np.nonzero(ok)
        kk = k_prev[ci, tt].astype(np.int64)
        np.add.at(bs[typ], (tt + 1, kk), r[ci, tt])
        np.add.at(bn[typ], (tt + 1, kk), 1)
    return bs, bn
