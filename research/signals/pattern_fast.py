"""패턴 GA 평가 커널 — 사건 표 거르기 · 매수정지 체결 · 확장 청산 경로 · 10 슬롯 포트폴리오(종목별 모멘텀 정합 바스켓 대조).

사전등록 정본: ``research/logs/sepa_explore/PATTERN_GA.md`` (원문 + 정정 0 + 정정 1 = ``REVIEW_PATTERN_GA.md`` 정정 1 초안 12항목·
MINOR 1~12 + 정정 2), 모두 결과 보기 전. 사건 표는 :mod:`research.signals.pattern_events`. numba 는 선택(:mod:`vcp` 규약).

**체결(사건 행 = 첫 관통일 d — 거르기는 전부 d−1 값):** 주문가 = 기준선(d) + 1호가, 시가 ≥ 전일 종가 × 상한가 비 → 버림,
시가 > 주문가 × (1 + ``gap_max``) → 버림(둘 다 소진), 손절 원값 = 기준(E 또는 하단선(d)) − 1호가, (주문가 − 손절 원값)/주문가 >
``stop_cap`` → 주문 없음(소진), 하단선 손절은 하단선(d) ≥ 종가(s) 면 무효(MINOR 4). 바닥 반전(정정 3)은 손절 = E − 1호가 고정·
익절 전 돌파 실패 없음. 정정 3 ``h0_in_pattern`` 은 사건의 패턴 종류와 정확히 같아야, ``ceil_kind`` = 수평 저항이면 저항 수 ≥ 2. 체결가 = max(시가, 주문가), 진입가 = 체결가 ×
(1 + 매수 미끄러짐), 손절폭 < 2% 면 손절가 = 진입가 × 0.98. 진입일 모호(시가 ≤ 기준선 ∧ 저가 ≤ 손절가 ∧ 종가 > 손절가)·확정 손절
(종가 ≤ 손절가)·진입일 익절 불인정은 :mod:`vcp` 와 같다 — 단 '시가 체결' 판정은 시가 ≥ 주문가(2단계 MINOR 6: 시가가 기준선과
주문가 사이면 장중 체결이라 모호로 센다). **R 기준(MINOR 3, 현행 명시):** ``stop_cap`` = (주문가 − 손절 원값)/주문가(호가 포함, 미끄러짐·
비용 전), 2% 하한 = (진입가 − 손절가)/진입가(호가·매수 미끄러짐 뒤, 비용 전); 보고의 ``real_R_post`` 분모도 뒤쪽 R 이다.

**청산(정정 0·정정 1 M1 — 결과 보기 전 해석):** 하루 처리 순서 = 우선순위 표(리뷰 M1-c: 손절 > 본전 손절 > 익절 > 추가 > 돌파 실패 > 추적).

1. 전날 결정한 돌파 실패·추적 → 그날 시가 전량(``exit_open``).
2. 진입일: 낙관 경로(모호 거래)는 진입일 손절을 보지 않는다. 그 밖은 저가 ≤ 손절가 → 손절가 × (1 − 매도 미끄러짐).
3. 진입 다음날부터 손절선 = 원 손절 → (익절 다음날부터) 진입가 × ``be_mult``(1.0) → (``be_r`` 발동 다음날부터) 진입가 × 1.005 중 높은 것.
   시가 ≤ 손절선 → 시가, 장중 저가 ≤ 손절선 → 선 값, 전량(추가분 포함 — 추가분 선은 이보다 높거나 같다). 손절·본전 체결가 =
   값 × (1 − 0.1%) − 그날 1호가(2단계 MINOR 1 — 쐐기 D ``sell_tick`` 과 같다, 라이브 동치 시험에서는 끈다). 시가 청산·익절·안전 상한
   종가에는 0.1% 미끄러짐이 없다(:mod:`vcp` 규약); 민감도 칸은 모든 매도에 같은 추가 미끄러짐(``X_SELL_EXTRA``)을 붙인다(MINOR 7).
4. 추가분 손절선 = max(진입가(본전), 그날 손절선)(정정 1 M1). 닿으면 추가분만 그 선(시가가 아래면 시가)에서 판다(그날 산 추가분 포함).
5. 부분 익절: 고가 ≥ 진입가 + ``take_r`` × R (R = 진입가 − 원 손절가) → ``take_frac`` 를 그 값(갭이면 시가)에 판다(그 시점 보유 전부의
   비율). 갭 익절일 저가 ≤ 본전선이면 그날 본전 전량(:mod:`vcp`). ``take_r`` = ∞ 면 없음.
6. 불타기(정정 0 · 2단계 리뷰 C1): 진입 다음날부터 고가 ≥ 진입가 + 1R(``pyr`` 1) 또는 고가 > H0(``pyr`` 2 — H0 기준선 진입에서는 러너가
   금지)인 날 → 다음 거래일 시가에 슬롯 자본의 1/3 을 산다(처음 진입은 2/3; 불타기 끔이면 100%). **추가 결정은 시가 정보만** 쓴다:
   시가 > 추가분 손절선이면 시가 × (1 + 미끄러짐)에 사고, 아니면 취소. 같은 날 저가가 추가분 손절선에 닿으면 그 추가분은 **그날**
   선 값(× (1 − 미끄러짐) − 1호가)에 판다. 같은 날 부분 익절은 추가분 포함 보유 전부에 ``take_frac`` 을 적용한다(시가 추가가 장중 익절보다
   먼저). 진입일 고가는 발동에 쓰지 않는다(체결과 순서 모름 — 진입일 익절 불인정과 같은 이유).
7. 돌파 실패(익절 전): 종가 < 기준가 × (1 − ``fb_tol``) 가 연속 ``fb_days`` 거래일(진입일 종가가 첫날, 정지일은 세지도 끊지도 않음)
   → 다음 거래일 시가 전량. ``fb_days`` = 0 이면 끔. 기준가(MINOR 5) = 추세선이면 U(d) 고정, 수평이면 D, H0 면 H0.
8. 추적(정정 1 M1 — **진입 다음날부터**, 익절 여부와 무관; ``from_entry`` = 0 은 라이브 동치 비교용 = 익절 뒤에만):
   MA20 / MA50 = 종가 < 그날 MA → 다음 거래일 시가. 샹들리에 = 종가 < 진입 뒤 최고 고가(진입일 고가 포함, 그날까지) − k × ATR20(전날 값)
   → 다음 거래일 시가. 조임(``tighten``): 최고 고가 ≥ 진입가 + 3R 인 날 다음날부터 k 를 절반.
10. **손절 기준(정정 5):** ``stop_basis`` = 종가면 손절선(본전 이동 포함)은 종가 < 선인 날 다음날 시가 전량(장중 저가로는 안 판다),
   장중에는 재난 손절선 e − ``stop_hard``·R 만 본다(시가가 그 아래면 시가). 추가분 손절도 종가 기준(다음날 시가). 손절 여유
   ``stop_buf`` × ATR20(d−1) 은 손절가에 들어가 R·9.5%·2% 가 모두 여유 포함 기준이다(:func:`_filter_py`). 진입일 모호는 장중 선 기준.
11. **클라이맥스(정정 4):** 진입 다음날부터 종가 ≥ 진입가 + ``climax_r``·R ∧ (전 거래일 종가 대비 ≥ ``climax_day`` ∨ 5행 전 종가 대비 ≥
   ``climax_5d``) → 다음날 시가 남은 수량 전량. 우선순위 손절 > 익절 > 클라이맥스 > 추적.
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

import math

import numpy as np

from research.signals.minervini_exits import BE_COST
from research.signals.pattern_events import FI, tick_dated
from research.signals.vcp import (
    REASONS as _V_REASONS,
)
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
 G_H0IN, G_CEILKIND, G_STOP_BUF, G_STOP_CLOSE, G_STOP_HARD, G_EQ, G_SMAX, G_MKT, G_SPOL) = range(35)
N_G = 35                                              # FWM: G_EQ(EPS 전년 같은 분기 대비 상승 필수) · G_SMAX(손절폭 상한) · G_MKT(시장 국면)
MKT_NHNL_MIN, MKT_BOS_MIN = 0.5, 0.6                  # FWM M7: 신고가 > 신저가 · 돌파 성공률 ≥ 60%(전 구간 중앙 ≈ 0.60, 결과 전 고정)
# --- 청산 인자 벡터(``px``) 번호 ---
(X_TAKE_R, X_TAKE_FRAC, X_TRAIL, X_TRAIL_K, X_TIGHTEN, X_PYR, X_BE_R, X_FB_DAYS, X_MAX_HOLD, X_FROM_ENTRY, X_SLIP,
 X_SELL_SLIP, X_BE_MULT, X_TIGHT_R, X_PYR_R, X_WB, X_SELL_TICK, X_SELL_EXTRA, X_CLIMAX, X_CLIMAX_DAY, X_CLIMAX_5D,
 X_CLIMAX_R, X_STOP_CLOSE, X_STOP_HARD, X_FAIL_EXIT, X_REENTRY, X_DYN, X_GAP, X_CAP, X_BUF, X_DFLOOR, X_REENTRY_WIN,
 X_SMAX, X_INTRA, X_TRAIL_START, X_TRAIL_CLOSE, X_SAFE, X_BE_CLOSE) = range(38)
N_X = 38                                              # FWM: X_SMAX = 재진입 손절에도 같은 손절폭 상한(NaN = 끔) · X_INTRA = 장중 청산
LINE_RE_CAP = 0.095                                   # 정정 9 선 복귀 재진입 9.5% 거르기
R_CLIMAX = 8                                          # 정정 4 급등(클라이맥스) 매도 — :data:`vcp.REASONS` 뒤에 붙인 사유
R_FAILX = 9                                           # 정정 6 반등 실패 조기 청산
REENTRY_WAIT = 20                                     # 정정 6 재진입 감시 창(청산 뒤 거래일) — 사전등록에 없어 결과 전 고정
TRAIL_MA20, TRAIL_MA50, TRAIL_CHAND = 0, 1, 2
PYR_OFF, PYR_1R, PYR_NEWHIGH, PYR_UPPER = 0, 1, 2, 3
# --- 경로 출력 번호 ---
(O_REASON, O_X, O_TP, O_PX1, O_TA, O_PA, O_TS, O_PXS, O_FIN, O_PNL, O_ADDED, O_MFE, O_EXIT_OPEN, O_WB, O_TF) = range(15)
N_O = 15
# --- 트레이드 기록 번호 ---
(Q_EV, Q_D, Q_X, Q_REASON, Q_E, Q_SP, Q_PNL, Q_WB, Q_AMB, Q_CX, Q_CD, Q_CADD, Q_ADDED, Q_MFE, Q_TP, Q_TA, Q_TS, Q_ALLOC,
 Q_PAR, Q_EXIT_OPEN, Q_TF, Q_PX1, Q_FIN, Q_PA, Q_PXS, Q_S0, Q_RE) = range(27)
N_Q = 27
#: 라이브 프로토콜(:data:`vcp.BUYSTOP_PARAMS` 기본) — 기준 청산 짝·동치 시험용.
REASONS = (*_V_REASONS, "climax", "fail_exit")
SLOTS, SLOT_W = 10, 0.10
WB_PYR = 2.0 / 3.0                                    # 불타기 켬: 처음 2/3 · 추가 1/3(정정 0)
LIVE_X = np.array([2.0, 0.5, TRAIL_MA50, 3.0, 0, PYR_OFF, np.inf, 1, 120, 0, 0.001, 0.001, 1.0, 3.0, 1.0, WB_PYR, 0, 0.0,
                   0, 0.15, 0.30, 4.0, 0, 2.0, 0, 0, 0, 0.03, 0.095, 0.0, 0.02, 0, np.nan, 0, 0.0, 0, 0, 0])

F_LEVEL, F_ORDER, F_H0, F_SUP, F_SLO, F_RHO, F_TOUCH, F_SPAN = (FI["level"], FI["order"], FI["h0"], FI["s_up"], FI["s_lo"],
                                                               FI["rho"], FI["touch"], FI["span"])
F_CEIL, F_DEPTH, F_DRY, F_RUN20, F_REST = FI["ceil"], FI["depth"], FI["dry"], FI["runup20"], FI["rest_gap"]
F_LOWER, F_STOPE, F_STOPL, F_OPEN, F_HIGH, F_LOW, F_CLOSE, F_CLOSE_S = (FI["lower_d"], FI["stopE_raw"], FI["stopL_raw"],
                                                                       FI["open_d"], FI["high_d"], FI["low_d"],
                                                                       FI["close_d"], FI["close_s"])
F_RS, F_BR, F_ABOVE50, F_TTS, F_TTH0, F_EYOY, F_EPREV = (FI["rs"], FI["breadth"], FI["above50"], FI["tt_s"], FI["tt_h0"],
                                                         FI["e_yoy"], FI["e_yoy_prev"])
F_NRES, F_H0IN, F_ATR = FI["n_res"], FI["h0in"], FI["atr_s"]
F_EQ, F_EQP = FI["e_q"], FI["e_q_prev"]
F_MBR200, F_MNH, F_MBOS = FI["mkt_br200"], FI["mkt_nhnl"], FI["mkt_bos"]
F_MA50UP = FI["ma50_up"]
BASE_BOTTOM = 3


# ---------------------------------------------------------------------------
# 1. 거르기 + 체결
# ---------------------------------------------------------------------------

def _filter_py(F, a, b, gv, lu_ratio, oidx, oe, osp, ofb, oamb, oskip0, os0, obuf):
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
    sbuf, sclose, shard = gv[G_STOP_BUF], gv[G_STOP_CLOSE] > 0.5, gv[G_STOP_HARD]
    eq, smax = gv[G_EQ] > 0.5, gv[G_SMAX]
    mkt = int(gv[G_MKT]) if gv[G_MKT] == gv[G_MKT] else 0
    spol = gv[G_SPOL] > 0.5                            # 진화 1세대: 손절폭 상한 정책 — 1 = 구조 손절이 상한보다 멀면 안 산다(미너비니), 0 = 끌어올림
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
        if stage == 4 and not (F[k, F_TTH0] > 0.5 and F[k, F_ABOVE50] > 0.5):   # FWM M4: H0 일 TT ∧ d−1 종가 > MA50
            continue
        if stage == 5 and not (F[k, F_TTH0] > 0.5 and F[k, F_ABOVE50] > 0.5 and F[k, F_MA50UP] > 0.5):   # 2세대: + MA50 상승
            continue
        if eq and not (F[k, F_EQ] > 0.0 and F[k, F_EQ] > F[k, F_EQP]):           # FWM M1: 흑자 ∧ 전년 같은 분기보다 증가
            continue
        if mkt == 1 and not F[k, F_MNH] > MKT_NHNL_MIN:                          # FWM M7 시장: 신고가 > 신저가
            continue
        if mkt == 2 and not F[k, F_MBOS] >= MKT_BOS_MIN:                         # 돌파가 먹히는 장
            continue
        if mkt == 3 and not F[k, F_MBR200] > 0.5:                                # MA200 위 종목 > 절반(리뷰 FWM M6: 지수 대용 편향 대신)
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
        if anchor >= 1:                               # 1 하단선(d) 고정 · 2 하단선 동적(정정 7 — 진입일 값이 R 기준)
            lo_d = F[k, F_LOWER]
            if not (lo_d == lo_d and lo_d < prev):
                continue
            sr = F[k, F_STOPL]
        else:
            sr = F[k, F_STOPE]
        bamt = 0.0
        if sbuf > 0:                                  # 정정 5 손절 여유: 기준 − 1호가 − buf × ATR20(d−1)
            bamt = sbuf * F[k, F_ATR]
            sr = sr - bamt
        if not (sr == sr and sr > 0):
            continue
        if (order - sr) / order > cap:
            continue
        f = max(op, order)
        if sr >= f:
            continue
        e = f * (1.0 + slip)
        sp = sr
        if smax == smax and (e - sp) / e > smax:      # FWM M2: 손절폭 상한 — 손절선을 진입가 × (1 − smax) 로 끌어올린다(거래는 산다)
            if spol:
                continue
            sp = e * (1.0 - smax)
        if (e - sp) / e < dfloor:
            sp = e * (1.0 - dfloor)
        cl = F[k, F_CLOSE]
        s0 = e - shard * (e - sp) if sclose else sp   # 진입일 장중 손절선: 장중 기준 = 손절가, 종가 기준 = 재난 손절선(정정 5)
        open_ok = op < order and cl > s0              # 2단계 MINOR 6: 시가 < 주문가 = 장중 체결(순서 모호), 그 밖은 시가 체결
        os0[n] = s0
        obuf[n] = bamt
        oidx[n] = k
        oe[n] = e
        osp[n] = sp
        ofb[n] = -np.inf if bottom else level * (1.0 - fbtol)       # 바닥 반전: 익절 전 돌파 실패 규칙 없음
        oskip0[n] = open_ok
        oamb[n] = open_ok and F[k, F_LOW] <= s0
        n += 1
    return n


filter_events = _jit(_filter_py)


# ---------------------------------------------------------------------------
# 2. 청산 경로(한 트레이드, 배분 1 단위)
# ---------------------------------------------------------------------------

def _sellp_py(v, ks, stick, t, mkt, reform):
    """손절·본전 체결가 = v × (1 − 매도 미끄러짐) [− 그날 1호가] (쐐기 D ``sell_tick`` 과 같은 규약, 2단계 MINOR 1)."""
    r = v * ks
    if stick:
        r -= tick_dated(v, t, mkt, reform)
    return r


_sellp = _jit(_sellp_py)


def _path_py(Op, H, L, C, MA20, MA50, ATR, TR, LIM, ci, d, e, sp, fb_line, h0, u_d, su, lE, liE, lsl, lbuf, skip0, mkt, reform,
             px, out):
    """진입 행 ``d`` 진입가 ``e``·손절가 ``sp`` → ``out``(:data:`N_O`). 규칙은 모듈 설명 청산 1~11. 손익 ``O_PNL`` = 배분 1 당 총손익
    (비용 전, 낙관 경로; 불타기 켬이면 처음 ``wb`` 만 사고 1/3 은 추가 전까지 현금).

    하루 순서(2단계 리뷰 C1 — 시가 결정은 시가까지의 정보만): 시가 = 전날 결정 청산(돌파 실패·추적·클라이맥스·종가 기준 손절) →
    종가 기준 추가분 매도 → 시가 ≤ 장중 손절선 전량 → 불타기 추가 / 장중 = 장중 손절선 전량 → 추가분 손절(장중 기준) → 부분 익절 /
    종가 = 종가 기준 손절 → 돌파 실패 → 클라이맥스 → 추적(정정 4 우선순위: 손절 > 익절 > 클라이맥스 > 추적).
    장중 손절선 = 장중 기준이면 손절선(본전 이동 포함), 종가 기준이면 재난 손절선 e − stop_hard·R(정정 5).
    ``u_d``·``su`` = 상단선(d 값·로그 기울기 — 정정 6 상단선 돌파 불타기), ``lE``·``liE``·``lsl``·``lbuf`` = 하단선(E·행·기울기)과
    손절 여유 금액(정정 7 하단선 동적 손절)."""
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
    kx = 1.0 - px[X_SELL_EXTRA]                       # 모든 매도의 추가 미끄러짐(민감도 칸 — 주 = 0)
    ks = kx - px[X_SELL_SLIP]                         # 손절·본전 매도
    stick = px[X_SELL_TICK] > 0.5
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
    smx = px[X_SMAX]                                  # 리뷰 FWM M2: 추가분에도 손절폭 상한(NaN = 끔)
    t_start = e + px[X_TRAIL_START] * R              # 진화 1세대: 추적은 고가가 진입가 + k·R 에 닿은 뒤부터(0 = 기존)
    t_close = px[X_TRAIL_CLOSE] > 0.5                 # 진화 1세대: 추적만 종가 판단(→ 다음날 시가), 손절·본전은 장중
    safe = px[X_SAFE] > 0.5                           # 진화 2세대(검증 차트): 추적선은 진입 뒤 종가가 그 위에 선 뒤부터 · 불타기는 본전 확보 뒤·갭 급등 시가 금지
    be_close = px[X_BE_CLOSE] > 0.5                   # 진화 2세대: 본전 이동 발동을 종가 기준으로(장중 고가 터치 털림 방지)
    armed = False
    ix = px[X_INTRA] > 0.5                            # FWM M6(사용자 "알고리즘 매매라 장중에도 판다"): 반등 실패·추적·클라이맥스를
    #                                                   전날까지 정보로 정한 선에 걸어 두고 장중에 닿으면 그 선(시가가 넘으면 시가)에 판다
    fb_days = int(px[X_FB_DAYS])
    from_entry = px[X_FROM_ENTRY] > 0.5
    climax = px[X_CLIMAX] > 0.5
    c_day, c_5d, c_lvl = px[X_CLIMAX_DAY], px[X_CLIMAX_5D], e + px[X_CLIMAX_R] * R
    sclose = px[X_STOP_CLOSE] > 0.5
    dyn = px[X_DYN] > 0.5                             # 정정 7: 매일 하단선 연장 − 1호가 − 여유 = 손절선, 재난 손절 필수
    hard = e - px[X_STOP_HARD] * R if (sclose or dyn) else -np.inf
    fx = int(px[X_FAIL_EXIT])                         # 정정 6: 0 끔 · 1 전일 저가 · 2 진입일 저가(바닥 반전만 — 러너가 건다)
    low_d = L[ci, d]
    prevl = np.nan
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
    add_sell = False                                  # 종가 기준: 추가분 다음날 시가 매도 대기
    fb_run = 0
    last = 0
    add_state = 0                                     # 0 없음 · 1 다음날 시가 대기 · 2 보유 · 3 끝(손절·취소)
    hh = H[ci, d]
    mfe = H[ci, d]
    tight_j = -1
    prevc = np.nan
    for j in range(horizon + 1):
        t = d + j
        if not TR[ci, t]:
            continue
        last = j
        op = Op[ci, t]
        if pending >= 0:                              # 1. 전날 결정 → 시가 전량
            px_ = _sellp(op, ks, stick, t, mkt, reform) if (pending == R_STOP or pending == R_BE) else op * kx
            cash += (sb + sa) * px_
            out[O_REASON] = pending
            out[O_X] = t
            out[O_FIN] = px_
            out[O_EXIT_OPEN] = 1
            out[O_PNL] = cash
            out[O_MFE] = mfe
            return
        if add_sell:                                  # 종가 기준 추가분 손절 → 그날 시가
            pxs = _sellp(op, ks, stick, t, mkt, reform)
            cash += sa * pxs
            sa = 0.0
            add_state = 3
            add_sell = False
            out[O_TS] = t
            out[O_PXS] = pxs
        if H[ci, t] > mfe:
            mfe = H[ci, t]
        line = sp                                     # 3. 손절선(본전 이동 포함)
        why = R_STOP
        if j > 0:
            if dyn:
                lv_ = lE * math.exp(lsl * (t - liE))
                line = lv_ - tick_dated(lv_, t, mkt, reform) - lbuf
            if partial and j >= be_from:
                line = be_line
                why = R_BE
            if be_trig >= 0 and j > be_trig and be_stop > line:
                line = be_stop
                why = R_BE
        intra = hard if sclose else (max(line, hard) if dyn else line)     # 장중 손절선
        if j == 0:
            if (not skip0) and L[ci, t] <= intra:     # 2. 진입일 손절(장중 선)
                px_ = _sellp(intra, ks, stick, t, mkt, reform)
                cash += sb * px_
                out[O_REASON] = R_STOP_ENTRY
                out[O_X] = t
                out[O_FIN] = px_
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
        else:
            if op <= intra:                           # 시가가 장중 선 아래 — 시가 전량(추가 없음)
                px_ = _sellp(op, ks, stick, t, mkt, reform)
                cash += (sb + sa) * px_
                out[O_REASON] = R_STOP if sclose else why
                out[O_X] = t
                out[O_FIN] = px_
                out[O_EXIT_OPEN] = 1
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
            lx = -np.inf                              # FWM M6 장중 하방 선 — 선 = max(반등 실패 기준, 추적선), 손절선보다 높을 때만
            why_x = -1
            if ix:
                if fx > 0 and hh < e + 2.0 * R:
                    ref = prevl if fx == 1 else low_d
                    if ref == ref and ref > lx:
                        lx = ref
                        why_x = R_FAILX
                if ((j >= 1) if from_entry else partial) and hh >= t_start and not t_close and (armed or not safe):
                    tl = np.nan
                    if trail == TRAIL_CHAND:
                        a = ATR[ci, t - 1]
                        k_use = kk * 0.5 if (tight_j >= 0 and j > tight_j) else kk
                        if a == a:
                            tl = hh - k_use * a
                    else:
                        tl = MA20[ci, t - 1] if trail == TRAIL_MA20 else MA50[ci, t - 1]
                    if tl == tl and tl > lx:
                        lx = tl
                        why_x = R_TRAIL
                if not (why_x >= 0 and lx > intra):
                    why_x = -1
                if why_x >= 0 and op <= lx:           # 시가가 이미 선 아래 — 시가 전량(리뷰 FWM M3: 추가 전에 판다)
                    px_ = _sellp(op, ks, stick, t, mkt, reform)
                    cash += (sb + sa) * px_
                    out[O_REASON] = why_x
                    out[O_X] = t
                    out[O_FIN] = px_
                    out[O_EXIT_OPEN] = 1
                    out[O_PNL] = cash
                    out[O_MFE] = mfe
                    return
            al = max(e, line)                         # 추가분 손절선 = max(본전, 그날 손절선)
            if add_state == 1:                        # 6. 시가 추가 — 시가 정보만(C1)
                if op > al and not (safe and prevc == prevc and op > prevc * 1.05):
                    pa = op * (1.0 + slip)
                    sa = wa / pa
                    cash -= wa
                    add_state = 2
                    out[O_TA] = t
                    out[O_PA] = pa
                    out[O_ADDED] = 1
                else:
                    add_state = 3
            if smx == smx and add_state == 2:         # 리뷰 FWM M2: 추가분 손절 = max(본전, 손절선, 추가가 × (1 − 상한))
                al = max(al, out[O_PA] * (1.0 - smx))
            if smx == smx and add_state == 2 and (L[ci, t] <= al or op <= al) and al >= intra and (why_x < 0 or al >= lx):
                pxs = _sellp(min(op, al), ks, stick, t, mkt, reform)   # 리뷰 FWM S1: 내려가는 길에 더 높은 추가분 선을 먼저 지난다
                cash += sa * pxs
                sa = 0.0
                add_state = 3
                out[O_TS] = t
                out[O_PXS] = pxs
            if why_x >= 0 and L[ci, t] <= lx:         # FWM M6 장중 — 손절선 위의 선을 먼저 지난다(내려가는 길)
                px_ = _sellp(lx, ks, stick, t, mkt, reform)
                cash += (sb + sa) * px_
                out[O_REASON] = why_x
                out[O_X] = t
                out[O_FIN] = px_
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
            if L[ci, t] <= intra:                     # 장중 선 — 전량(그날 산 추가분 포함)
                px_ = _sellp(intra, ks, stick, t, mkt, reform)
                cash += (sb + sa) * px_
                out[O_REASON] = R_STOP if sclose else why
                out[O_X] = t
                out[O_FIN] = px_
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
            if (not sclose) and add_state == 2 and (L[ci, t] <= al or op <= al):    # 4. 추가분 손절(장중 기준)
                pxs = _sellp(min(op, al), ks, stick, t, mkt, reform)
                cash += sa * pxs
                sa = 0.0
                add_state = 3
                out[O_TS] = t
                out[O_PXS] = pxs
            if (not partial) and (H[ci, t] >= tgt or op >= tgt):     # 5. 부분 익절(그 시점 보유 전부 — 그날 추가분 포함)
                g = op >= tgt
                p1 = (op if g else tgt) * kx
                cash += tf * (sb + sa) * p1
                sb *= 1.0 - tf
                sa *= 1.0 - tf
                partial = True
                be_from = j + 1
                out[O_TP] = t
                out[O_PX1] = p1
                if g and L[ci, t] <= be_line and not sclose:
                    px_ = _sellp(be_line, ks, stick, t, mkt, reform)
                    cash += (sb + sa) * px_
                    out[O_REASON] = R_BE
                    out[O_X] = t
                    out[O_FIN] = px_
                    out[O_PNL] = cash
                    out[O_MFE] = mfe
                    return
        if ix and climax and j >= 1:                  # FWM M6 장중 클라이맥스 매도: 가격 ≥ max(+kR, min(전일 종가 × (1+당일), 5일 전 종가 × (1+5일)))
            c5_ = C[ci, t - 5] if t >= 5 else np.nan
            lv1 = prevc * (1.0 + c_day) if prevc == prevc and prevc > 0 else np.inf
            lv5 = c5_ * (1.0 + c_5d) if c5_ == c5_ and c5_ > 0 else np.inf
            pst = max(c_lvl, min(lv1, lv5))
            if pst < 1e300 and H[ci, t] >= pst:       # 하방 선은 위에서 먼저 처리(같은 날 순서 모름 → 보수)
                p_ = (op if op >= pst else pst) * kx
                cash += (sb + sa) * p_
                out[O_REASON] = R_CLIMAX
                out[O_X] = t
                out[O_FIN] = p_
                out[O_EXIT_OPEN] = 1 if op >= pst else 0
                out[O_PNL] = cash
                out[O_MFE] = mfe
                return
        if be_trig < 0 and ((C[ci, t] >= be_level) if be_close else (H[ci, t] >= be_level or op >= be_level)):
            be_trig = j
        if add_state == 0 and pyr > 0 and j >= 1 and (be_trig >= 0 or not safe):
            if (pyr == PYR_1R and H[ci, t] >= pyr_level) or (pyr == PYR_NEWHIGH and H[ci, t] > h0) or \
                    (pyr == PYR_UPPER and C[ci, t] > u_d * math.exp(su * (t - d))):
                add_state = 1
        if H[ci, t] > hh:
            hh = H[ci, t]
        if tighten and tight_j < 0 and hh >= tight_level:
            tight_j = j
        ct = C[ci, t]
        if j == j_time:                               # 9. 안전 상한 종가
            cash += (sb + sa) * ct * kx
            out[O_REASON] = R_TIME
            out[O_X] = t
            out[O_FIN] = ct * kx
            out[O_PNL] = cash
            out[O_MFE] = mfe
            return
        if j >= sig_end:
            prevc = ct
            prevl = L[ci, t]
            continue
        if sclose:                                    # 10. 종가 기준 손절(정정 5) — 종가 < 손절선 → 다음날 시가 전량
            if ct < line:
                pending = why
            elif add_state == 2 and ct < max(e, line):
                add_sell = True
        if pending < 0 and fb_days > 0 and not partial:     # 7. 돌파 실패
            if ct < fb_line:
                fb_run += 1
                if fb_run >= fb_days:
                    pending = R_FB
            else:
                fb_run = 0
        if pending < 0 and fx > 0 and j >= 1 and hh < e + 2.0 * R and not ix:   # 정정 6 반등 실패(2R 전) — 장중 모드는 위에서
            ref = prevl if fx == 1 else low_d
            if ref == ref and ct < ref:
                pending = R_FAILX
        if pending < 0 and climax and j >= 1 and ct >= c_lvl and not ix:   # 11. 클라이맥스(정정 4): 당일 ∨ 5일 급등 ∧ +kR
            up1 = ct / prevc - 1.0 if prevc == prevc and prevc > 0 else np.nan
            c5 = C[ci, t - 5] if t >= 5 else np.nan
            up5 = ct / c5 - 1.0 if c5 == c5 and c5 > 0 else np.nan
            if (up1 == up1 and up1 >= c_day) or (up5 == up5 and up5 >= c_5d):
                pending = R_CLIMAX
        act = ((j >= 1) if from_entry else partial) and hh >= t_start and (t_close or not ix) and (armed or not safe)   # 8. 추적
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
        if safe and not armed:                        # 2세대: 오늘 종가가 오늘 추적선 위면 다음날부터 추적 유효
            if trail == TRAIL_CHAND:
                a2 = ATR[ci, t - 1] if t >= 1 else np.nan
                armed = a2 == a2 and ct > hh - kk * a2
            else:
                m2 = MA20[ci, t] if trail == TRAIL_MA20 else MA50[ci, t]
                armed = m2 == m2 and ct > m2
        prevc = ct
        prevl = L[ci, t]
    reason = R_DATA_END if d + last >= lim - 1 else R_DELISTED   # 끝까지 거래 = 데이터 끝, 앞서 끊김 = 상장폐지(MINOR 13)
    fin = C[ci, d + last] * kx
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


# 슬롯 정수 열(P_I)·실수 열(P_F)
(I_CI, I_K, I_M, I_REC, I_X, I_TP, I_TA, I_TS, I_EO, I_PAR, I_RS, I_RE, I_PID, I_EV, I_IE) = range(15)
(F_ALLOC, F_SB, F_SA, F_CASH, F_LAST, F_PX1, F_PA, F_PXS, F_FIN, F_WA, F_TF, F_PHI, F_H0, F_U, F_SU, F_E, F_SL, F_BUF) = range(18)


def _open_py(q, t, ci, e, sp, fb, h0, u, su, lE, liE, lsl, lbuf, amb, skip0, s0, k, m, par, ppar, ev, pid, is_re, alloc, eq_prev,
             pmix, cost_mult, cost_on_sell, reform, PX, Op, H, L, C, MA20, MA50, ATR, TR, LIM, cost_tab, bs, bn, strat_cs, out,
             P_I, P_F, rec, n_taken):
    """슬롯 ``q`` 에 진입 — 청산 경로를 그 자리에서 계산하고 장부·기록을 채운다. ``ppar`` = 청산 인자 행(재진입은 고정 손절 행),
    ``par`` = 원 config 번호(기록용). → (진입일 대조 기여, 대조 빈 날 여부)."""
    px = PX[ppar]
    exit_path(Op, H, L, C, MA20, MA50, ATR, TR, LIM, ci, t, e, sp, fb, h0, u, su, lE, liE, lsl, lbuf, skip0, m, reform, px, out)
    wb = out[O_WB]
    phi = 1.0 - pmix if amb else 1.0
    sb = wb * alloc / e
    pc = alloc - wb * alloc
    if amb:                                           # 보수 경로 몫 p: 진입일 장중 손절선에 판다
        s0p = _sellp(s0, 1.0 - px[X_SELL_EXTRA] - px[X_SELL_SLIP], px[X_SELL_TICK] > 0.5, t, m, reform)
        pc += pmix * sb * s0p
        pc -= cost_mult * cost_tab[m, t] * (pmix * sb * s0p if cost_on_sell else pmix * wb * alloc)
        sb *= phi
    P_I[q, I_CI] = ci
    P_I[q, I_K] = k
    P_I[q, I_M] = m
    P_I[q, I_REC] = n_taken
    P_I[q, I_X] = int(out[O_X])
    P_I[q, I_TP] = int(out[O_TP])
    P_I[q, I_TA] = int(out[O_TA])
    P_I[q, I_TS] = int(out[O_TS])
    P_I[q, I_EO] = 1 if out[O_EXIT_OPEN] > 0.5 else 0
    P_I[q, I_PAR] = par
    P_I[q, I_RS] = int(out[O_REASON])
    P_I[q, I_RE] = is_re
    P_I[q, I_PID] = pid
    P_I[q, I_EV] = ev
    P_I[q, I_IE] = liE
    P_F[q, F_ALLOC] = alloc
    P_F[q, F_SB] = sb
    P_F[q, F_SA] = 0.0
    P_F[q, F_CASH] = pc
    P_F[q, F_LAST] = e
    P_F[q, F_PX1] = out[O_PX1]
    P_F[q, F_PA] = out[O_PA]
    P_F[q, F_PXS] = out[O_PXS]
    P_F[q, F_FIN] = out[O_FIN]
    P_F[q, F_WA] = 1.0 - wb
    P_F[q, F_TF] = out[O_TF]
    P_F[q, F_PHI] = phi
    P_F[q, F_H0] = h0
    P_F[q, F_U] = u
    P_F[q, F_SU] = su
    P_F[q, F_E] = lE
    P_F[q, F_SL] = lsl
    P_F[q, F_BUF] = lbuf
    r = rec[n_taken]
    r[Q_EV] = ev
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
    r[Q_PAR] = par
    r[Q_EXIT_OPEN] = out[O_EXIT_OPEN]
    r[Q_TF] = out[O_TF]
    r[Q_PX1] = out[O_PX1]
    r[Q_FIN] = out[O_FIN]
    r[Q_PA] = out[O_PA]
    r[Q_PXS] = out[O_PXS]
    r[Q_S0] = s0
    r[Q_RE] = is_re
    b, emp = basket(bs, bn, strat_cs, Op, C, 1, t, k, ci)
    return wb * alloc / eq_prev * b, emp


open_slot = _jit(_open_py)


def _portfolio_py(cd, cci, ce, csp, cfb, ch0, cu, csu, cE, ciE, csl, cbuf, camb, cskip0, cs0, cstrat, cmkt, cpar, cev, cpid,
                  PX, Op, H, L, C, MA20, MA50, ATR, TR, LIM, bs, bn, strat_cs, cost_tab, cost_mult, pmix, t_lo, t_end, slots,
                  slot_w, ret, bench, expo, rec, year_of, ny_out, reform, cost_on_sell, lu_ratio, npar):
    """후보(주문일 순·같은 날 우선순위 순)를 날마다 슬롯에 넣는다 — 들어간 트레이드만 그 자리에서 청산 경로를 계산한다(lazy).
    ``ret``·``bench``·``expo`` = 날 ``[t_lo, t_end)`` 의 포트폴리오 수익·정합 바스켓 대조·노출. ``rec`` = 들어간 트레이드 기록
    (:data:`N_Q`), ``ny_out`` = 진입 연도(``year_of`` 번호)별 진입 수. 반환 (진입 수, 슬롯·현금 부족 버림, 보유 중 버림, 대조 빈 날).
    비용: 주 = 배수 × ``cost_rt``(그 묶음을 닫는 날) × 그 묶음의 **매수 금액**(부분 익절 비용 없음 — ``slot_portfolio`` 규약);
    ``cost_on_sell`` = 매도 때마다 배수 × ``cost_rt``(그날) × **매도 금액**(2단계 MINOR 8 기술 칸). 하루 흐름 순서 = 추가(시가) →
    추가분 손절 → 부분 익절 → 청산(:func:`_path_py` 와 같다).

    **바닥 재진입(정정 6 — 결과 전 해석):** ``X_REENTRY`` 켠 config 의 (재진입이 아닌) 트레이드가 손절·진입일 손절·반등 실패로 청산되면 그
    종목에 감시를 건다(청산일 뒤 :data:`REENTRY_WAIT` 거래일). 청산 다음날부터의 최저 저가 F(진행 중, 동률 = 늦은 봉)가 원 패턴 E × 0.99
    이하 ∧ 하단선 연장(F 행) × 0.97 이상이고, F 뒤 ≤ 5 거래일 안에 고가 ≥ F 봉 고가 + 1호가(그날 표)이며 전날 종가 ≤ 주문가 × 0.995 이고
    F 뒤 그 주문가를 먼저 넘은 봉이 없으면 그날 매수정지(체결·갭·상한가·손절 상한·2%·진입일 모호 규칙은 원 config 와 같다, 손절 = F −
    1호가 − 여유). 감시 한 번에 시도 한 번(관통했는데 못 사도 끝), 재진입 트레이드는 다시 감시를 걸지 않는다(1회). 같은 날에는 재진입을
    새 사건보다 먼저 처리하고, 슬롯 규칙(그날 청산 자리는 다음날부터·보유 중 무시·10 슬롯)은 같다. 손실 중 물타기는 없다(청산 뒤에만).
    재진입 트레이드의 하단선·상단선·H0 는 원 패턴 값이고, 청산 인자는 ``PX[par + npar]`` 행(하단선 동적 끔 — 재진입 손절은 고정)이다.

    **선 복귀 재진입(정정 9):** ``X_REENTRY`` = 2 인 config 의 손절·진입일 손절 청산(청산일 x) 뒤, x ~ x + ``reentry_win`` − 1 의 종가가 하단선
    연장(여유 없이) 위로 돌아오면 다음날 시가 × (1 + 미끄러짐)에 재진입. 손절 = 이탈 구간 최저 저가(장중 기준 x 부터, 종가 기준은 이탈
    종가일 x − 1 부터 그 종가일까지) − 1호가, (진입가 − 손절)/진입가 > 9.5% 면 안 산다, 2% 하한, 상한가 시가 버림, 1회."""
    n_c = len(cd)
    n_codes = Op.shape[0]
    held = np.zeros(n_codes, np.bool_)
    done_pid = np.full(n_codes, -1, np.int64)        # 같은 패턴(종목, i0, 행(D)) 체결 하나
    P_ON = np.zeros(slots, np.bool_)
    P_I = np.zeros((slots, 15), np.int64)
    P_F = np.zeros((slots, 18))
    W_ON = np.zeros(n_codes, np.bool_)               # 재진입 감시
    W_X = np.zeros(n_codes, np.int64)
    W_FI = np.full(n_codes, -1, np.int64)
    W_CROSS = np.zeros(n_codes, np.bool_)
    W_FMIN = np.full(n_codes, np.inf)
    W_I = np.zeros((n_codes, 15), np.int64)
    W_F = np.zeros((n_codes, 18))
    W_T = np.zeros(n_codes, np.int64)                # 감시 종류: 1 F 바닥 · 2 선 복귀
    act = np.zeros(n_codes, np.int64)
    na = 0
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
            w = (P_F[q, F_SB] + P_F[q, F_SA]) * P_F[q, F_LAST] / eq_prev
            typ = 2 if (P_I[q, I_X] == t and P_I[q, I_EO] == 1) else 0
            b, emp = basket(bs, bn, strat_cs, Op, C, typ, t, P_I[q, I_K], P_I[q, I_CI])
            if emp:
                unm += 1
            bt += w * b
        kk = 0                                        # 재진입(감시 종목) — 새 사건보다 먼저
        while kk < na:
            c = act[kk]
            par = W_I[c, I_PAR]
            px = PX[par]
            m = W_I[c, I_M]
            y = t - 1
            if W_T[c] == 2:                           # 정정 9 선 복귀
                win = int(px[X_REENTRY_WIN])
                if y - W_X[c] >= win:
                    W_ON[c] = False
                    act[kk] = act[na - 1]
                    na -= 1
                    continue
                fire = False
                if y >= W_X[c] and TR[c, y]:
                    if y > W_X[c] and L[c, y] < W_FMIN[c]:
                        W_FMIN[c] = L[c, y]
                    lv_ = W_F[c, F_E] * math.exp(W_F[c, F_SL] * (y - W_I[c, I_IE]))
                    fire = C[c, y] > lv_ and TR[c, t]
                if not fire:
                    kk += 1
                    continue
                W_ON[c] = False
                act[kk] = act[na - 1]
                na -= 1
                op = Op[c, t]
                prev = C[c, t - 1]
                if prev == prev and prev > 0 and op >= prev * lu_ratio:
                    continue
                e = op * (1.0 + px[X_SLIP])
                lo_ = W_FMIN[c]
                sr = lo_ - tick_dated(lo_, t, m, reform)
                if not (sr == sr and sr > 0 and sr < e) or (e - sr) / e > LINE_RE_CAP:
                    continue
                sp = sr
                if px[X_SMAX] == px[X_SMAX] and (e - sp) / e > px[X_SMAX]:
                    sp = e * (1.0 - px[X_SMAX])
                if (e - sp) / e < px[X_DFLOOR]:
                    sp = e * (1.0 - px[X_DFLOOR])
                s0 = e - px[X_STOP_HARD] * (e - sp) if px[X_STOP_CLOSE] > 0.5 else sp
                if n_open >= slots or cash <= 1e-12 or held[c]:
                    sk_slot += 1
                    continue
                q = 0
                while P_ON[q]:
                    q += 1
                alloc = min(slot_w * eq_prev, cash)
                b_, emp = open_slot(q, t, c, e, sp, -np.inf, W_F[c, F_H0], W_F[c, F_U], W_F[c, F_SU], W_F[c, F_E],
                                    W_I[c, I_IE], W_F[c, F_SL], 0.0, False, False, s0, W_I[c, I_K], m, par, par + npar,
                                    W_I[c, I_EV], W_I[c, I_PID], 1, alloc, eq_prev, pmix, cost_mult, cost_on_sell, reform, PX, Op,
                                    H, L, C, MA20, MA50, ATR, TR, LIM, cost_tab, bs, bn, strat_cs, out, P_I, P_F, rec, n_taken)
                bt += b_
                unm += 1 if emp else 0
                P_ON[q] = True
                ny_out[year_of[t]] += 1
                n_taken += 1
                held[c] = True
                n_open += 1
                cash -= alloc
                continue
            if t > W_X[c] + REENTRY_WAIT:
                W_ON[c] = False
                act[kk] = act[na - 1]
                na -= 1
                continue
            if y > W_X[c] and TR[c, y]:
                if L[c, y] <= W_FMIN[c]:
                    W_FMIN[c] = L[c, y]
                    W_FI[c] = y
                    W_CROSS[c] = False
                elif W_FI[c] >= 0:
                    hf = H[c, W_FI[c]]
                    if H[c, y] >= hf + tick_dated(hf, y, m, reform):
                        W_CROSS[c] = True
            fi = W_FI[c]
            fire = False
            order = 0.0
            if fi >= 0 and (not W_CROSS[c]) and t - fi <= 5 and TR[c, t]:
                Fv = W_FMIN[c]
                Ea = W_F[c, F_E]
                lowf = Ea * math.exp(W_F[c, F_SL] * (fi - W_I[c, I_IE]))
                if Fv <= Ea * 0.99 and Fv >= lowf * 0.97:
                    hf = H[c, fi]
                    order = hf + tick_dated(hf, t, m, reform)
                    if C[c, t - 1] <= order * 0.995 and H[c, t] >= order:
                        fire = True
            if not fire:
                kk += 1
                continue
            W_ON[c] = False                           # 감시 한 번에 시도 한 번
            act[kk] = act[na - 1]
            na -= 1
            op = Op[c, t]
            prev = C[c, t - 1]
            if prev == prev and prev > 0 and op >= prev * lu_ratio:
                continue
            if op > order * (1.0 + px[X_GAP]):
                continue
            Fv = W_FMIN[c]
            a_ = ATR[c, t - 1]
            bamt = px[X_BUF] * a_ if px[X_BUF] > 0 else 0.0
            sr = Fv - tick_dated(Fv, t, m, reform) - bamt
            if not (sr == sr and sr > 0) or (order - sr) / order > px[X_CAP]:
                continue
            f = max(op, order)
            if sr >= f:
                continue
            e = f * (1.0 + px[X_SLIP])
            sp = sr
            if px[X_SMAX] == px[X_SMAX] and (e - sp) / e > px[X_SMAX]:
                sp = e * (1.0 - px[X_SMAX])
            if (e - sp) / e < px[X_DFLOOR]:
                sp = e * (1.0 - px[X_DFLOOR])
            s0 = e - px[X_STOP_HARD] * (e - sp) if px[X_STOP_CLOSE] > 0.5 else sp
            open_ok = op < order and C[c, t] > s0
            amb = open_ok and L[c, t] <= s0
            if n_open >= slots or cash <= 1e-12 or held[c]:
                sk_slot += 1
                continue
            q = 0
            while P_ON[q]:
                q += 1
            alloc = min(slot_w * eq_prev, cash)
            b_, emp = open_slot(q, t, c, e, sp, -np.inf, W_F[c, F_H0], W_F[c, F_U], W_F[c, F_SU], W_F[c, F_E], W_I[c, I_IE],
                                W_F[c, F_SL], bamt, amb, open_ok, s0, W_I[c, I_K], m, par, par + npar, W_I[c, I_EV], W_I[c, I_PID],
                                1, alloc,
                                eq_prev, pmix, cost_mult, cost_on_sell, reform, PX, Op, H, L, C, MA20, MA50, ATR, TR, LIM, cost_tab,
                                bs, bn, strat_cs, out, P_I, P_F, rec, n_taken)
            bt += b_
            unm += 1 if emp else 0
            P_ON[q] = True
            ny_out[year_of[t]] += 1
            n_taken += 1
            held[c] = True
            n_open += 1
            cash -= alloc
        while ptr < n_c and cd[ptr] == t:             # 진입
            i = ptr
            ptr += 1
            ci = cci[i]
            if n_open >= slots or cash <= 1e-12:
                sk_slot += 1
                continue
            if held[ci] or done_pid[ci] == cpid[i]:
                sk_held += 1
                continue
            q = 0
            while P_ON[q]:
                q += 1
            alloc = min(slot_w * eq_prev, cash)
            b_, emp = open_slot(q, t, ci, ce[i], csp[i], cfb[i], ch0[i], cu[i], csu[i], cE[i], ciE[i], csl[i], cbuf[i], camb[i],
                                cskip0[i], cs0[i], cstrat[i], cmkt[i], cpar[i], cpar[i], cev[i], cpid[i], 0, alloc, eq_prev, pmix,
                                cost_mult,
                                cost_on_sell, reform, PX, Op, H, L, C, MA20, MA50, ATR, TR, LIM, cost_tab, bs, bn, strat_cs, out,
                                P_I, P_F, rec, n_taken)
            bt += b_
            unm += 1 if emp else 0
            P_ON[q] = True
            ny_out[year_of[t]] += 1
            n_taken += 1
            held[ci] = True
            done_pid[ci] = cpid[i]
            n_open += 1
            cash -= alloc
        for q in range(slots):                        # 흐름: 추가(시가) → 추가분 손절 → 부분 익절 → 청산
            if not P_ON[q]:
                continue
            m = P_I[q, I_M]
            alloc = P_F[q, F_ALLOC]
            phi = P_F[q, F_PHI]
            an = phi * P_F[q, F_WA] * alloc
            if P_I[q, I_TA] == t:
                P_F[q, F_SA] = an / P_F[q, F_PA]
                P_F[q, F_CASH] -= an
                b, emp = basket(bs, bn, strat_cs, Op, C, 1, t, P_I[q, I_K], P_I[q, I_CI])
                bt += an / eq_prev * b
            if P_I[q, I_TS] == t:
                pr = P_F[q, F_SA] * P_F[q, F_PXS]
                P_F[q, F_CASH] += pr - cost_mult * cost_tab[m, t] * (pr if cost_on_sell else an)
                P_F[q, F_SA] = 0.0
            if P_I[q, I_TP] == t:
                tf = P_F[q, F_TF]
                pr = tf * (P_F[q, F_SB] + P_F[q, F_SA]) * P_F[q, F_PX1]
                P_F[q, F_CASH] += pr
                if cost_on_sell:
                    P_F[q, F_CASH] -= cost_mult * cost_tab[m, t] * pr
                P_F[q, F_SB] *= 1.0 - tf
                P_F[q, F_SA] *= 1.0 - tf
            if P_I[q, I_X] == t:
                pr = (P_F[q, F_SB] + P_F[q, F_SA]) * P_F[q, F_FIN]
                c_ = P_F[q, F_CASH] + pr
                if cost_on_sell:
                    c_ -= cost_mult * cost_tab[m, t] * pr
                else:
                    c_ -= cost_mult * cost_tab[m, t] * phi * (1.0 - P_F[q, F_WA]) * alloc
                    if P_I[q, I_TA] >= 0 and P_I[q, I_TS] < 0:
                        c_ -= cost_mult * cost_tab[m, t] * an
                cash += c_
                P_ON[q] = False
                ci = P_I[q, I_CI]
                held[ci] = False
                n_open -= 1
                rs = P_I[q, I_RS]
                rmode = int(PX[P_I[q, I_PAR], X_REENTRY])
                arm = P_I[q, I_RE] == 0 and ((rmode == 1 and (rs == R_STOP or rs == R_STOP_ENTRY or rs == R_FAILX))
                                             or (rmode == 2 and (rs == R_STOP or rs == R_STOP_ENTRY)))
                if arm:
                    if not W_ON[ci]:
                        act[na] = ci
                        na += 1
                    W_ON[ci] = True                   # 정정 6·9 재진입 감시
                    W_T[ci] = rmode
                    W_X[ci] = t
                    W_FI[ci] = -1
                    W_FMIN[ci] = np.inf
                    W_CROSS[ci] = False
                    if rmode == 2:                    # 이탈 구간 시작: 장중 기준 = 청산일, 종가 기준 = 이탈 종가일(전날)
                        b0 = t - 1 if PX[P_I[q, I_PAR], X_STOP_CLOSE] > 0.5 else t
                        for yy in range(max(b0, 0), t + 1):
                            if TR[ci, yy] and L[ci, yy] < W_FMIN[ci]:
                                W_FMIN[ci] = L[ci, yy]
                    for c_i in range(15):
                        W_I[ci, c_i] = P_I[q, c_i]
                    for c_f in range(18):
                        W_F[ci, c_f] = P_F[q, c_f]
        val = cash
        inv = 0.0
        for q in range(slots):
            if not P_ON[q]:
                continue
            c = C[P_I[q, I_CI], t]
            if c == c and c > 0:
                P_F[q, F_LAST] = c
            v = (P_F[q, F_SB] + P_F[q, F_SA]) * P_F[q, F_LAST]
            val += P_F[q, F_CASH] + v
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
