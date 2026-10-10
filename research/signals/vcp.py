"""VCP 재시험 — 연속 축소 수축(VCP) 검출 · 피벗 매수정지 시뮬 · 분기 단독 실적 가속 · 기반 수급.

사전등록 정본: ``research/logs/sepa_explore/VCP_RETEST.md`` — 5632879 + **정정 7dbbd0c** + **정정 2 aae4b8e**
(``REVIEW_VCP_RETEST.md`` 1·2단계, 모두 결과 보기 전). 이 파일은 그 §2~§7·정정 C2·C3·M1·M2·M3·M5·M7·M8·정정 2 를 옮긴 계산 기계다 — 수치는
:data:`VCP_PARAMS`·:data:`BUYSTOP_PARAMS`·:data:`E_PARAMS`·:data:`FLOW_PARAMS` 에 고정하고 탐색하지 않는다. 어느 팔이 좋은지는
러너(``research/experiments/vcp_retest_run.py``)가 숫자로만 적는다.

numba 는 선택이다(:mod:`research.signals.ga_fast` 와 같은 규약) — ``uv run --with numba …`` 면 커널이 JIT 되고, 없으면 **같은
소스**가 순수 파이썬으로 돈다(테스트는 두 경로 다). 배열은 :class:`research.signals.sepa._Book` 처럼 **종목×날짜**(전치)다.

**날짜 규약:** 셋업 행 ``s`` = 주문일 ``d`` 의 전날(``s = d − 1``). 셋업 조건은 전부 ``s`` 종가까지의 자료로 계산하고, 주문은
``d`` 장중 매수정지다. 트레이드 표의 ``event_day`` = ``s``, ``order_day`` = ``entry_day`` = ``d``, ``hold_days`` = 청산 행 − ``s``.

**사전등록 문구 → 구현 해석(모호한 곳은 전부 여기 적는다 — 결과 보기 전 결정):**

1. 창 = ``s−259..s`` (260 거래일, 정정 C1). **창 260일이 다 차지 않거나 창 안에 결측이 있으면 셋업 없음**(``no_data``).
   H0 = 창 최고 고가, 동률은 **늦은 봉**(정정 C2). 기반 길이 ``s − i0`` ∈ [15, 250] — θ 5% · T ≤ 6 · 단조 감소 때문에 실제 체결
   기반은 짧다(러너가 분포를 보고한다; 긴 기반 포함을 주장하지 않는다).
2. 지그재그(θ = 5%)는 ``i0`` 에서 고점 후보 = H0 로 시작한다. 봉마다 상태 전이는 하나뿐이고, 한 봉이 진행 방향의 새 극값과 반대쪽
   θ 반전을 함께 만들면 **극값 먼저, 반전 확인은 다음 봉부터**(정정 2 A-C1 — 두 상태 대칭, 그 봉의 극값을 버리지 않는다):
   - 오름(고점 후보) 상태: 고가 ≥ 후보 → 후보 갱신(동률 = 늦은 봉). 아니면 저가 ≤ 후보 × 0.95 → 고점 확정, 저점 후보 = 그 봉 저가.
   - 내림(저점 후보) 상태: 저가 ≤ 후보 → 후보 갱신(동률 = 늦은 봉). 아니면 고가 ≥ 후보 × 1.05 → 저점 확정, 고점 후보 = 그 봉 고가.
3. **피벗(정정 C2):** 지그재그가 오름 상태(마지막 확정점이 저점)로 끝나면 피벗 = 마지막 확정 저점 이후 진행 중 최고 고가(고점
   후보), 내림 상태(마지막 확정점이 고점)로 끝나면 그 고점. 수축 = 확정 쌍 (H_{k−1} − L_k)/H_{k−1} + 진행 중 마지막 수축
   (피벗 − L_last)/피벗, L_last = 피벗 다음날 ~ ``s`` 의 최저 저가. 중복 없음. 피벗 뒤 ``s − ip`` < 3 거래일이면 셋업 없음.
   문자 해석(늘 마지막 확정 고점)은 ``confirmed_only=True`` 비교 인자로만 둔다.
   **저점 상승(정정 2 A-M3):** 수축 저점 열 L_1, …, L_{T−1}, L_last 가 각각 L_k ≥ L_{k−1} × 0.99 (하락 계단 제외, ``low_rise``).
4. 감소: c_k < c_{k−1} **엄격**(마지막 수축 포함). 거래량 마름(정정 M7) = mean(vol[ip+1..s]) ≤ 0.8 × mean(vol[ip−49..ip])
   (**마지막 수축 시작 전 50일** = 피벗 봉을 포함한 50일, 수축 기간과 겹치지 않음). 정지일 거래량 0 은 0 으로 들어간다. NaN 이 있거나 비교 평균이 0 이면 마름 실패.
   (분할 직후 60 거래일 셋업 제외는 러너가 조정계수 패널로 건다.)
5. 매수정지: 주문가 = 피벗 + 1호가(``tick_size_vec(피벗)``), 손절가 = L_last − 1호가. 분할조정가라 격자 위가 아니어도 그 가격대
   호가단위. 주문 전 (피벗 − 손절가)/피벗 > 9.5% → 주문 없음. 그날 거래 불가 → 무효, 시가 ≥ 전일 종가 × :data:`sepa.LIMIT_UP_RATIO`
   → 버림, 시가 > 피벗 × 1.03 → 버림, 고가 < 주문가 → 미체결(피벗 소진 안 함). 체결가 = max(시가, 주문가), **진입가 = 체결가 × 1.001**.
   손절폭 d = (진입가 − 손절가)/진입가, d < 2% 면 손절가 = 진입가 × 0.98. 갭 체결로 d > 10% 는 그대로(``over10`` 기록).
   X2 칸(정정 M5·정정 2 B-M2)은 손절만 옛 규칙: d = clip(1 − 옛 저점 × 0.99 / 진입가, 3%, 10%), 손절가 = 진입가 × (1 − d).
   주 = 옛 저점 min(low[d−10..d−1])(주문 시점에 알려진 값), 기술 = 문구 그대로 min(low[d−10..d])(진입일 저가 포함) — 러너가 넘긴다.
   (주문 전 9.5% 거르기는 새 손절 기준 그대로 — 진입 집합이 V0 와 같아야 손절 규칙만의 차이가 된다.)
6. 청산(라이브 프로토콜 — :func:`research.signals.minervini_exits._simulate_one_v2` 기본값 + ``failed_breakout=True``,
   ``time_stop_days=None`` 의 하루 순서): 전날 결정 → 시가(돌파 실패·추적) / 손절선 터치(시가가 아래면 시가) / 2R 절반 익절(갭이면
   시가; 갭 익절일 저가 ≤ 본전선이면 그날 본전) / 120 거래일째 종가 / 종가 판단(익절 전 종가 < 피벗 → 다음 거래일 시가 전량
   ``failed_breakout`` — **진입일 종가 포함**(정정 M8), 익절 뒤 종가 < MA50 → 다음 거래일 시가 ``trail``). 익절 뒤 본전선 =
   **진입가 × 1.0**(라이브 ``strategy.py`` 기본 — ``BE_COST`` 0.5% 는 꺼진 ``be_r`` 전용). **매수정지 진입일의 2R 익절은 인정하지
   않는다**(정정 M8 — 순서 불명, 다음날부터). 손절·진입일 손절·본전 매도가에 −0.1% 미끄러짐(정정 MINOR — 대칭).
7. **진입일 손절 순서(정정 C3·정정 2 B-M1):** 한 번의 시뮬로 두 경로를 낸다. **진입일 종가 ≤ 손절가면 확정 손절**(종가는 체결
   뒤다 — 두 경로 모두 손절가 × 0.999). 모호 = 시가 ≤ 피벗 ∧ 진입일 저가 ≤ 손절가 ∧ 종가 > 손절가. 낙관 경로 = 모호한 날 진입일
   손절을 보지 않음(저가가 체결 전) — 이 경로가 트레이드의 기본 열(``ret``·``exit_*``)이다. 보수 경로 = 모호한 거래를 진입일
   손절가 × 0.999 로(``ret_stop0``). 모호하지 않은 거래는 두 경로가 같다(갭 체결의 진입일 저가 ≤ 손절가는 늘 손절).
   주 가정(분봉 비율 p 혼합)은 러너가 ``p × 보수 + (1−p) × 낙관`` 으로 만든다. 보유 중 거르기는 낙관 경로 청산일(더 늦은 쪽)로 건다.
8. 같은 피벗(종목 × 피벗 고가 행 ``ip``)은 첫 체결 하나만. 같은 종목 보유 중(신호 행 ``s`` < 청산 행)이면 새 셋업 무시 — 보유
   중 판정은 낙관 경로 청산일로 건다(혼합은 "낙관 경로의 거래 집합" 위의 혼합이다).
9. **NXT 절단(정정 2 B-M3):** :class:`Arrays` 의 ``CUT[종목]`` = 그 종목이 2026-09-14+ 미보정(NXT 오염) 행을 가지면 2026-09-14 행,
   아니면 패널 길이. 보유 경로는 ``CUT`` 전 마지막 거래 가능일(2026-09-11) 종가에서 ``data_end`` 로 끊는다(트레이드·널·대조 모두).
   진입 행 ≥ ``CUT`` 이면 ``skip_nxt_cut``.
10. **쐐기 시험(``WEDGE_TEST.md``)용 스위치 — 기본값 = VCP 동작 그대로:** ``fb_days``(돌파 실패 = 익절 전 연속 ``fb_days`` 거래 가능일
    종가 < 돌파 기준, 정지일은 세지도 끊지도 않는다; VCP 1, 쐐기 2), ``ref_order``(9.5% 사전 거르기·3% 갭의 기준을 피벗 대신 주문가로),
    ``be_r``(청산 v2 본전 이동 — 고가 ≥ 진입가 + be_r × (진입가 − 손절가) 인 날 다음 거래일부터 손절선 = 진입가 × 1.005; 하루 순서
    손절 > 본전 손절 > 2R 익절 > be_r 발동 기록 > 종가 판단, :func:`minervini_exits._simulate_one_v2` 와 같음 — 테스트로 대조).
    지그재그는 :func:`_zigzag_py` 하나를 :mod:`research.signals.wedge` 와 함께 쓴다.

실적(§6·정정 M2)·수급(§7·정정 M3)은 :func:`standalone_quarter_states`·:func:`e_accel_asof`·:class:`FlowCum` 설명에 적는다.
비용(``korea.cost_rt``)은 여기서 빼지 않는다 — 러너 몫.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals.minervini_exits import BE_COST
from research.signals.sepa import LIMIT_UP_RATIO, _Book
from swing_it.features.korea import tick_size_vec

try:  # numba 는 선택 — 없으면 같은 소스가 파이썬으로 돈다.
    import numba

    HAVE_NUMBA = True
    prange = numba.prange
except ImportError:  # pragma: no cover - CI 경로
    numba = None
    HAVE_NUMBA = False
    prange = range


def _jit(fn, parallel: bool = False):
    if not HAVE_NUMBA:
        return fn
    return numba.njit(cache=False, parallel=parallel, nogil=True)(fn)


# --- 사전등록 수치(VCP_RETEST.md + 정정 — 탐색하지 않는다) ---
VCP_PARAMS = {
    "win": 260,          # H0 창: d−1 까지 260 거래일(정정 C1)
    "base_min": 15,      # H0 위치 ≥ 15 거래일 앞
    "base_max": 250,     # H0 위치 ≤ 250 거래일 앞(정정 C1)
    "theta": 0.05,       # 지그재그 반전 5%
    "t_min": 2,          # 수축 수 T ≥ 2 (마지막 포함)
    "t_max": 6,          # T ≤ 6
    "c1_lo": 0.10,       # 첫 수축 ∈ [10%, 40%]
    "c1_hi": 0.40,
    "last_max": 0.10,    # 마지막 수축 ≤ 10%
    "pivot_min": 0.85,   # 피벗 ≥ 0.85 × H0
    "close_lo": 0.90,    # d−1 종가 ∈ [0.90, 1.0) × 피벗
    "vol_win": 50,       # 거래량 비교: 마지막 수축 시작 전 50일(정정 M7)
    "vol_max": 0.8,      # 마지막 수축 평균 거래량 ≤ 0.8 × 그 50일 평균
    "dry_min": 3,        # 피벗 뒤 ≥ 3 거래일(정정 C2)
    "confirmed_only": False,  # True = 문자 해석(늘 마지막 확정 고점) — 비교용
    "low_rise": 0.99,    # 저점 상승: L_k ≥ L_{k−1} × 0.99 (정정 2 A-M3). 0 = 끔(깔때기 기술용)
}
BUYSTOP_PARAMS = {
    "gap_max": 0.03,     # 시가 > 피벗 × 1.03 → 버림
    "slip": 0.001,       # 매수정지 체결가 +0.1%
    "sell_slip": 0.001,  # 손절·본전 매도가 −0.1%(정정 MINOR)
    "risk_pre_max": 0.095,  # 주문 전 (피벗 − 손절가)/피벗 > 9.5% → 주문 없음
    "d_cap": 0.10,       # 매수정지: 체결 뒤 d > 10% 는 기록만 / X1(시가 진입): d > 10% 면 안 산다
    "d_floor": 0.02,     # d < 2% → 손절가 = 진입가 × 0.98
    "take_r": 2.0,       # 2R 절반 익절
    "take_frac": 0.5,
    "max_hold": 120,     # 안전 상한(진입일 = 1일째)
    "be_mult": 1.0,      # 익절 뒤 본전선 = 진입가 × 1.0 (라이브 기본)
    "x1_chase": 0.05,    # X1 옛 시점: 시가 > 피벗 × 1.05 → 추격 금지
    "old_stop_mult": 0.99, "old_stop_floor": 0.03, "old_stop_cap": 0.10,   # X2 옛 손절
    "be_r": np.inf,      # 본전 이동(청산 v2 be_r): 고가 ≥ 진입가 + be_r × (진입가 − 손절가) 인 날 다음 거래일부터 손절선 = 진입가 × 1.005
    "sell_tick": False,  # True = 손절·본전 체결가 = min(시가, 손절선) − 1호가(쐐기 D 정정 1 M5; sell_slip 과 함께 곱한 뒤 뺀다)
    "fb_days": 1,        # 돌파 실패: 익절 전 연속 fb_days 일 종가 < 돌파 기준 → 다음날 시가(VCP = 1, 쐐기 = 2)
    "ref_order": False,  # 9.5% 사전 거르기·3% 갭의 기준: False = 피벗(VCP), True = 주문가(쐐기)
}

# --- 코드표 ---
(V_OK, V_NO_DATA, V_BASE, V_FEW, V_MANY, V_RECENT, V_C1, V_NOT_DECR, V_LOWER_LOW, V_LAST, V_PIVOT, V_CLOSE,
 V_DRY) = range(13)
VCP_STATUS = ("ok", "no_data", "base_len", "contractions<2", "contractions>6", "pivot<3d", "c1_range",
              "not_decreasing", "lower_low", "last>10%", "pivot<0.85H0", "close_range", "volume_dryup")
(S_OK, S_NO_NEXT, S_OVERLAP, S_PIVOT_DONE, S_RISK, S_HALT, S_LIMIT_UP, S_GAP, S_NO_FILL, S_BAD, S_CHASE,
 S_WIDE, S_BELOW, S_CUT) = range(14)
SIM_STATUS = ("ok", "skip_no_next_day", "skip_overlap", "skip_pivot_done", "skip_risk>9.5%", "skip_halt",
              "skip_limit_up", "skip_gap>3%", "no_fill", "skip_bad_input", "skip_chase>5%", "skip_d>10%",
              "skip_open<=stop", "skip_nxt_cut")
R_STOP, R_STOP_ENTRY, R_BE, R_TRAIL, R_FB, R_TIME, R_DELISTED, R_DATA_END = range(8)
REASONS = ("stop", "stop_entry", "breakeven", "trail", "failed_breakout", "time", "delisted", "data_end")


# ---------------------------------------------------------------------------
# 1. VCP 검출
# ---------------------------------------------------------------------------

def _zigzag_py(H, L, ci, i0, t, theta, hv, hi_, lv, li_):
    """지그재그(θ) — ``i0`` 고점 후보에서 ``t`` 까지. 확정 고점·저점을 ``hv/hi_``·``lv/li_`` 에 채우고
    ``(nh, nl, up, hc, ihc, lc, ilc)`` 를 돌려준다(모듈 설명 2 — 극값 먼저, 반전 확인은 다음 봉부터, 동률 = 늦은 봉).
    :mod:`research.signals.wedge` 도 같은 함수를 쓴다."""
    nh = 0
    nl = 0
    up = True
    hc = H[ci, i0]
    ihc = i0
    lc = 0.0
    ilc = -1
    for j in range(i0 + 1, t + 1):
        hj = H[ci, j]
        lj = L[ci, j]
        if up:
            if hj >= hc:
                hc = hj
                ihc = j
            elif lj <= hc * (1.0 - theta):
                hv[nh] = hc
                hi_[nh] = ihc
                nh += 1
                up = False
                lc = lj
                ilc = j
        else:
            if lj <= lc:                              # 극값 먼저(정정 2 A-C1) — 반전 확인은 다음 봉부터
                lc = lj
                ilc = j
            elif hj >= lc * (1.0 + theta):
                lv[nl] = lc
                li_[nl] = ilc
                nl += 1
                up = True
                hc = hj
                ihc = j
    return nh, nl, up, hc, ihc, lc, ilc


_zigzag = _jit(_zigzag_py)


def _vcp_one_py(H, L, C, V, ci, t, win, bmin, bmax, theta, tmin, tmax, c1lo, c1hi, lastmax, pivmin, closelo,
                volwin, volmax, drymin, confirmed_only, low_rise):
    """종목 ``ci`` 의 행 ``t``(= d−1) 까지 자료로 VCP 셋업 → ``(status, pivot, l_last, i0, ip, T, c1, c_last, vol_ratio, h0)``.

    규칙은 모듈 설명 1~4. ``status`` 는 처음 걸린 조건(:data:`VCP_STATUS`). 실패해도 그때까지 계산한 값은 채운다.
    """
    nan = np.nan
    s0 = t - win + 1
    if s0 < 0:
        return V_NO_DATA, nan, nan, -1, -1, 0, nan, nan, nan, nan
    i0 = -1
    h0 = -np.inf
    for j in range(s0, t + 1):
        hj = H[ci, j]
        if not (hj == hj and L[ci, j] == L[ci, j] and C[ci, j] == C[ci, j]):
            return V_NO_DATA, nan, nan, -1, -1, 0, nan, nan, nan, nan
        if hj >= h0:                                  # 동률 = 늦은 봉
            h0 = hj
            i0 = j
    base = t - i0
    if base < bmin or base > bmax:
        return V_BASE, nan, nan, i0, -1, 0, nan, nan, nan, h0

    cap = base + 3
    hv = np.empty(cap)
    hi_ = np.empty(cap, dtype=np.int64)
    lv = np.empty(cap)
    li_ = np.empty(cap, dtype=np.int64)
    nh, nl, up, hc, ihc, lc, ilc = _zigzag(H, L, ci, i0, t, theta, hv, hi_, lv, li_)
    if (not confirmed_only) and up and nl == nh and nh > 0:
        hv[nh] = hc                                   # 확정 저점 뒤 진행 중 고점 = 피벗
        hi_[nh] = ihc
        T = nh + 1
    else:
        T = nh
    if T < tmin:
        return V_FEW, nan, nan, i0, -1, T, nan, nan, nan, h0
    if T > tmax:
        return V_MANY, nan, nan, i0, -1, T, nan, nan, nan, h0
    ip = hi_[T - 1]
    piv = hv[T - 1]
    c1 = (hv[0] - lv[0]) / hv[0]
    if t - ip < drymin:
        return V_RECENT, piv, nan, i0, ip, T, c1, nan, nan, h0
    ll = np.inf
    for j in range(ip + 1, t + 1):
        if L[ci, j] < ll:
            ll = L[ci, j]
    clast = (piv - ll) / piv
    if c1 < c1lo or c1 > c1hi:
        return V_C1, piv, ll, i0, ip, T, c1, clast, nan, h0
    prev = c1
    for k in range(1, T - 1):
        ck = (hv[k] - lv[k]) / hv[k]
        if not ck < prev:
            return V_NOT_DECR, piv, ll, i0, ip, T, c1, clast, nan, h0
        prev = ck
    if not clast < prev:
        return V_NOT_DECR, piv, ll, i0, ip, T, c1, clast, nan, h0
    if low_rise > 0:
        for k in range(1, T):                         # 저점 열 lv[0..T−2], L_last
            lk = ll if k == T - 1 else lv[k]
            if lk < lv[k - 1] * low_rise:
                return V_LOWER_LOW, piv, ll, i0, ip, T, c1, clast, nan, h0
    if clast > lastmax:
        return V_LAST, piv, ll, i0, ip, T, c1, clast, nan, h0
    if piv < pivmin * h0:
        return V_PIVOT, piv, ll, i0, ip, T, c1, clast, nan, h0
    ct = C[ci, t]
    if not (ct < piv and ct >= closelo * piv):
        return V_CLOSE, piv, ll, i0, ip, T, c1, clast, nan, h0
    if ip - volwin + 1 < 0:
        return V_DRY, piv, ll, i0, ip, T, c1, clast, nan, h0
    vs = 0.0
    for j in range(ip + 1, t + 1):
        vs += V[ci, j]
    vb = 0.0
    for j in range(ip - volwin + 1, ip + 1):
        vb += V[ci, j]
    vr = nan
    if vb > 0 and vs == vs:
        vr = (vs / (t - ip)) / (vb / volwin)
    if not vr <= volmax:
        return V_DRY, piv, ll, i0, ip, T, c1, clast, vr, h0
    return V_OK, piv, ll, i0, ip, T, c1, clast, vr, h0


_vcp_one = _jit(_vcp_one_py)


def _vcp_batch_py(H, L, C, V, cis, ts, win, bmin, bmax, theta, tmin, tmax, c1lo, c1hi, lastmax, pivmin, closelo,
                  volwin, volmax, drymin, confirmed_only, low_rise, st, piv, ll, i0, ip, nT, c1, cl, vr, h0):
    m = len(cis)
    for i in prange(m):
        r = _vcp_one(H, L, C, V, cis[i], ts[i], win, bmin, bmax, theta, tmin, tmax, c1lo, c1hi, lastmax, pivmin,
                     closelo, volwin, volmax, drymin, confirmed_only, low_rise)
        st[i] = r[0]
        piv[i] = r[1]
        ll[i] = r[2]
        i0[i] = r[3]
        ip[i] = r[4]
        nT[i] = r[5]
        c1[i] = r[6]
        cl[i] = r[7]
        vr[i] = r[8]
        h0[i] = r[9]


_vcp_batch = _jit(_vcp_batch_py, parallel=True)


def _arr(x: pd.DataFrame, like: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(x.reindex(index=like.index, columns=like.columns).to_numpy(float).T)


def vcp_scan_arrays(H: np.ndarray, L: np.ndarray, C: np.ndarray, V: np.ndarray, cis, ts,
                    params: "dict | None" = None) -> dict:
    """종목×날짜 배열에서 (종목 ``cis[i]``, 행 ``ts[i]``) 마다 VCP 셋업(행 ``ts`` 까지 자료). → 열 이름 → 배열 dict."""
    p = {**VCP_PARAMS, **(params or {})}
    m = len(cis)
    out = {"status": np.zeros(m, np.int64), "pivot": np.full(m, np.nan), "l_last": np.full(m, np.nan),
           "i0": np.full(m, -1, np.int64), "ip": np.full(m, -1, np.int64), "n_contr": np.zeros(m, np.int64),
           "c1": np.full(m, np.nan), "c_last": np.full(m, np.nan), "vol_ratio": np.full(m, np.nan),
           "h0": np.full(m, np.nan)}
    if m:
        _vcp_batch(H, L, C, V, np.asarray(cis, np.int64), np.asarray(ts, np.int64), int(p["win"]),
                   int(p["base_min"]), int(p["base_max"]), float(p["theta"]), int(p["t_min"]), int(p["t_max"]),
                   float(p["c1_lo"]), float(p["c1_hi"]), float(p["last_max"]), float(p["pivot_min"]),
                   float(p["close_lo"]), int(p["vol_win"]), float(p["vol_max"]), int(p["dry_min"]),
                   bool(p["confirmed_only"]), float(p["low_rise"]), out["status"], out["pivot"], out["l_last"], out["i0"], out["ip"],
                   out["n_contr"], out["c1"], out["c_last"], out["vol_ratio"], out["h0"])
    return out


def vcp_setups(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
               mask: "pd.DataFrame | None" = None, params: "dict | None" = None, *, valid_only: bool = False
               ) -> pd.DataFrame:
    """VCP 셋업 표 — 한 행 = (종목, 주문일 ``d``). 값은 전부 **신호 행 ``s = d − 1`` 종가까지** 자료다.

    ``mask`` = 신호 행 ``s`` 의 후보(bool 패널). 반환 열: ``code, event_day(s), order_day(d), sig_idx, day_idx(d), col_idx,
    valid, status, pivot, l_last, i0, ip, n_contr, c1, c_last, vol_ratio, h0, base_len``. 패널 끝 행(주문일 없음)은 뺀다.
    """
    H, L, C, V = _arr(high, close), _arr(low, close), _arr(close, close), _arr(volume, close)
    if mask is None:
        mk = np.ones((close.shape[0], close.shape[1]), dtype=bool)
    else:
        mk = mask.reindex(index=close.index, columns=close.columns).fillna(False).to_numpy(dtype=bool).copy()
    mk[-1, :] = False
    si, ci = np.nonzero(mk)
    r = vcp_scan_arrays(H, L, C, V, ci, si, params)
    df = pd.DataFrame({"code": close.columns[ci], "event_day": close.index[si], "order_day": close.index[si + 1],
                       "sig_idx": si, "day_idx": si + 1, "col_idx": ci, **r})
    df["valid"] = df["status"] == V_OK
    df["base_len"] = np.where(df["i0"] >= 0, df["sig_idx"] - df["i0"], -1)
    if valid_only:
        df = df[df["valid"]]
    return df.sort_values(["order_day", "code"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 청산 경로 + 매수정지·시가 진입 시뮬
# ---------------------------------------------------------------------------

_TICK_UP = np.array([2_000.0, 5_000.0, 20_000.0, 50_000.0, 200_000.0, 500_000.0])
_TICK_SZ = np.array([1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1_000.0])


def _tick_py(px):
    """가격 → 현행 호가단위(:func:`korea.tick_size_vec` 와 같은 표, 스칼라)."""
    for i in range(len(_TICK_UP)):
        if px < _TICK_UP[i]:
            return _TICK_SZ[i]
    return _TICK_SZ[len(_TICK_SZ) - 1]


_tick = _jit(_tick_py)


def _sell_px(px, ks, sell_tick):
    """손절·본전 체결가 = px × (1 − sell_slip) [− 1호가]."""
    v = px * ks
    if sell_tick:
        v = v - _tick(px)
    return v


_sell = _jit(_sell_px)


def _exit_path_py(Op, H, L, C, MA, TR, PL, CUT, ci, d, entry, stop_px, tgt, fb_line, skip0, p0, take_frac, max_hold,
                  be_line, sell_slip, fb_days, be_r, sell_tick):
    """진입 행 ``d``(오프셋 0) 이후 라이브 청산 → ``(reason, x, partial, part_x, px1, fin_px)``. 규칙은 모듈 설명 6·7.

    ``skip0`` = 진입일 손절을 보지 않는다(낙관 경로), ``p0`` = 진입일 2R 익절 인정(시가 진입만 — 매수정지는 False).
    손절·본전 체결가에는 ``× (1 − sell_slip)``.
    """
    nan = np.nan
    n = Op.shape[1]
    lim = min(n, CUT[ci])                             # NXT 절단(모듈 설명 9)
    Ln = lim - d
    k_last = max_hold - 1
    j_time = -1
    if k_last <= Ln - 1:
        for j in range(k_last, Ln):
            if TR[ci, d + j]:
                j_time = j
                break
    horizon = j_time if j_time >= 0 else Ln - 1
    sig_end = min(k_last, horizon + 1)
    ks = 1.0 - sell_slip
    partial = False
    be_from = -1
    be_trig = -1                                      # be_r 발동 오프셋(청산 v2 — 다음 거래일부터 진입가 × (1 + BE_COST))
    be_level = entry + be_r * (entry - stop_px) if be_r < 1e18 else np.inf
    be_stop = entry * (1.0 + BE_COST)
    part_x = -1
    px1 = nan
    pending = -1
    fb_run = 0
    last = 0
    for j in range(horizon + 1):
        t = d + j
        if not TR[ci, t]:
            continue
        last = j
        op = Op[ci, t]
        if pending == R_FB or pending == R_TRAIL:
            return pending, t, partial, part_x, px1, op
        if j == 0:
            if (not skip0) and L[ci, t] <= stop_px:
                return R_STOP_ENTRY, t, False, -1, nan, _sell(stop_px, ks, sell_tick)
        else:
            line = stop_px
            why = R_STOP
            if partial and j >= be_from:
                line = be_line
                why = R_BE
            if be_trig >= 0 and j > be_trig and be_stop > line:
                line = be_stop
                why = R_BE
            if L[ci, t] <= line or op <= line:
                return why, t, partial, part_x, px1, _sell(min(op, line), ks, sell_tick)
        if (not partial) and (j > 0 or p0) and (H[ci, t] >= tgt or op >= tgt):
            gap = op >= tgt
            px1 = op if gap else tgt
            partial = True
            be_from = j + 1
            part_x = t
            if gap and L[ci, t] <= be_line:
                return R_BE, t, True, t, px1, _sell(be_line, ks, sell_tick)
        if be_trig < 0 and (H[ci, t] >= be_level or op >= be_level):
            be_trig = j
        if j == j_time:
            return R_TIME, t, partial, part_x, px1, C[ci, t]
        if j >= sig_end:
            continue
        if not partial:
            if C[ci, t] < fb_line:                    # 돌파 실패 — 연속 fb_days 일(정지일은 세지도 끊지도 않는다)
                fb_run += 1
                if fb_run >= fb_days:
                    pending = R_FB
            else:
                fb_run = 0
        else:
            m = MA[ci, t]
            if m == m and C[ci, t] < m:
                pending = R_TRAIL
    reason = R_DATA_END if (lim < n or PL[ci]) else R_DELISTED
    return reason, d + last, partial, part_x, px1, C[ci, d + last]


_exit_path = _jit(_exit_path_py)


def _run_buystop_py(Op, H, L, C, MA, TR, PL, CUT, cis, ds, pivots, orders, stops_raw, pids, old_low, stop_mode, ref_order,
                    fb_days, be_r, sell_tick, gap_max,
                    slip, sell_slip, risk_max, d_floor, take_r, take_frac, max_hold, be_mult, lu_ratio, old_mult,
                    old_floor, old_cap, no_overlap, st, x, fill, entry, stop, dd, tgt, gapf, amb, reason, part,
                    part_x, px1, fin):
    """사건(주문일 순 정렬)을 차례로 — 보유 중·같은 피벗 거르기가 순서에 의존하므로 병렬로 돌리지 않는다.

    ``stop_mode`` 0 = 새 손절(L_last − 1호가, 2% 하한), 1 = X2 옛 손절(``old_low`` = min(low[d−10..d]), × 0.99, [3%, 10%]).
    결과 경로 = 낙관(모호한 날 진입일 손절 안 봄), ``amb`` = 모호 여부(보수 경로는 진입일 손절가 × (1 − sell_slip)).
    """
    n_st, n = Op.shape
    busy = np.full(n_st, -1, np.int64)
    done_pid = np.full(n_st, -(1 << 40), np.int64)
    for i in range(len(cis)):
        ci = cis[i]
        d = ds[i]
        if d >= n or d < 1:
            st[i] = S_NO_NEXT
            continue
        s = d - 1
        if no_overlap and s < busy[ci]:
            st[i] = S_OVERLAP
            continue
        if done_pid[ci] == pids[i]:
            st[i] = S_PIVOT_DONE
            continue
        P = pivots[i]
        od = orders[i]
        sr = stops_raw[i]
        if not (P == P and od == od and sr == sr and P > 0 and sr > 0):
            st[i] = S_BAD
            continue
        ref = od if ref_order else P                     # 9.5%·3% 갭 기준: VCP = 피벗, 쐐기 = 주문가
        if (ref - sr) / ref > risk_max:
            st[i] = S_RISK
            continue
        if d >= CUT[ci]:
            st[i] = S_CUT
            continue
        if not TR[ci, d]:
            st[i] = S_HALT
            continue
        op = Op[ci, d]
        prev = C[ci, s]
        if prev == prev and prev > 0 and op >= prev * lu_ratio:
            st[i] = S_LIMIT_UP
            continue
        if op > ref * (1.0 + gap_max):
            st[i] = S_GAP
            continue
        if H[ci, d] < od:
            st[i] = S_NO_FILL
            continue
        f = max(op, od)
        e = f * (1.0 + slip)
        if stop_mode == 1:
            ol = old_low[i]
            if not (ol == ol and ol > 0):
                st[i] = S_BAD
                continue
            w = 1.0 - ol * old_mult / e
            w = min(max(w, old_floor), old_cap)
            sp = e * (1.0 - w)
        else:
            sp = sr
            w = (e - sp) / e
            if w < d_floor:
                sp = e * (1.0 - d_floor)
                w = d_floor
        tg = e * (1.0 + take_r * w)
        open_ok = op <= P and C[ci, d] > sp               # 종가 ≤ 손절가 = 확정 진입일 손절(정정 2 B-M1)
        is_amb = open_ok and L[ci, d] <= sp
        r = _exit_path(Op, H, L, C, MA, TR, PL, CUT, ci, d, e, sp, tg, P, open_ok, False, take_frac, max_hold,
                       e * be_mult, sell_slip, fb_days, be_r, sell_tick)
        st[i] = S_OK
        fill[i] = f
        entry[i] = e
        stop[i] = sp
        dd[i] = w
        tgt[i] = tg
        gapf[i] = op >= od
        amb[i] = is_amb
        reason[i] = r[0]
        x[i] = r[1]
        part[i] = r[2]
        part_x[i] = r[3]
        px1[i] = r[4]
        fin[i] = r[5]
        busy[ci] = r[1]
        done_pid[ci] = pids[i]


_run_buystop = _jit(_run_buystop_py)


def _run_open_py(Op, H, L, C, MA, TR, PL, CUT, cis, ds, dds, take_r, take_frac, max_hold, be_mult, lu_ratio, sell_slip,
                 slip, fb_days, be_r, sell_tick, st, x, entry, reason, part, part_x, px1, fin):
    """대조: 행 ``d`` 시가 × (1 + slip) 진입, 손절폭 ``dds[i]``, 돌파 실패 피벗 = 진입가. 겹침 없음(독립 표본) — 병렬."""
    n = Op.shape[1]
    for i in prange(len(cis)):
        ci = cis[i]
        d = ds[i]
        if d >= n or d < 1:
            st[i] = S_NO_NEXT
            continue
        if d >= CUT[ci]:
            st[i] = S_CUT
            continue
        if not TR[ci, d]:
            st[i] = S_HALT
            continue
        op = Op[ci, d]
        prev = C[ci, d - 1]
        if prev == prev and prev > 0 and op >= prev * lu_ratio:
            st[i] = S_LIMIT_UP
            continue
        w = dds[i]
        if not (w == w and w > 0):
            st[i] = S_BAD
            continue
        e = op * (1.0 + slip)
        sp = e * (1.0 - w)
        tg = e * (1.0 + take_r * w)
        r = _exit_path(Op, H, L, C, MA, TR, PL, CUT, ci, d, e, sp, tg, e, False, True, take_frac, max_hold,
                       e * be_mult, sell_slip, fb_days, be_r, sell_tick)
        st[i] = S_OK
        entry[i] = e
        reason[i] = r[0]
        x[i] = r[1]
        part[i] = r[2]
        part_x[i] = r[3]
        px1[i] = r[4]
        fin[i] = r[5]


_run_open = _jit(_run_open_py, parallel=True)


def _run_x1_py(Op, H, L, C, MA, TR, PL, CUT, cis, ds, pivots, stops_raw, pids, chase, d_cap, d_floor, take_r, take_frac,
               max_hold, be_mult, lu_ratio, sell_slip, fb_days, be_r, sell_tick, st, x, entry, stop, dd, reason, part, part_x, px1, fin):
    """X1(정정 M5): 신호 행 ``ds[i]`` 종가 돌파(러너가 고른다) → 다음 거래일 행 시가 진입, 추격 5%·상한가·정지 버림, 손절 = 새 손절
    (L_last − 1호가, 2% 하한), d > 10% 면 안 산다. 같은 피벗 첫 신호 하나·보유 중 무시. 순차."""
    n_st, n = Op.shape
    busy = np.full(n_st, -1, np.int64)
    done_pid = np.full(n_st, -(1 << 40), np.int64)
    for i in range(len(cis)):
        ci = cis[i]
        s = ds[i]
        e_ = s + 1
        if e_ >= n:
            st[i] = S_NO_NEXT
            continue
        if s < busy[ci]:
            st[i] = S_OVERLAP
            continue
        if done_pid[ci] == pids[i]:
            st[i] = S_PIVOT_DONE
            continue
        done_pid[ci] = pids[i]                        # 옛 시점: 피벗당 첫 신호 하나(체결 여부와 무관)
        P = pivots[i]
        sr = stops_raw[i]
        if e_ >= CUT[ci]:
            st[i] = S_CUT
            continue
        if not TR[ci, e_]:
            st[i] = S_HALT
            continue
        op = Op[ci, e_]
        prev = C[ci, s]
        if prev == prev and prev > 0 and op >= prev * lu_ratio:
            st[i] = S_LIMIT_UP
            continue
        if op > P * (1.0 + chase):
            st[i] = S_CHASE
            continue
        if op <= sr:
            st[i] = S_BELOW
            continue
        w = (op - sr) / op
        sp = sr
        if w > d_cap:
            st[i] = S_WIDE
            continue
        if w < d_floor:
            sp = op * (1.0 - d_floor)
            w = d_floor
        tg = op * (1.0 + take_r * w)
        r = _exit_path(Op, H, L, C, MA, TR, PL, CUT, ci, e_, op, sp, tg, P, False, True, take_frac, max_hold,
                       op * be_mult, sell_slip, fb_days, be_r, sell_tick)
        st[i] = S_OK
        entry[i] = op
        stop[i] = sp
        dd[i] = w
        reason[i] = r[0]
        x[i] = r[1]
        part[i] = r[2]
        part_x[i] = r[3]
        px1[i] = r[4]
        fin[i] = r[5]
        busy[ci] = r[1]


_run_x1 = _jit(_run_x1_py)


class Arrays:
    """시뮬 배열 한 벌(종목×날짜) — :class:`sepa._Book` 에 MA50·마지막 행 존재 여부를 더한 것."""

    def __init__(self, o, h, lo, c, ma50, volume):
        b = _Book(o, h, lo, c, ma50, volume)
        self.days, self.codes = b.days, b.codes
        self.O, self.H, self.L, self.C, self.MA = b.O, b.H, b.L, b.C, b.MA
        self.TR = np.ascontiguousarray(b.tradable)
        self.PL = np.ascontiguousarray(b.present[:, -1])
        self.CUT = np.full(len(self.codes), len(self.days), np.int64)   # NXT 절단 행(모듈 설명 9) — 러너가 채운다

    def set_cut(self, codes, cut_day) -> int:
        """``codes`` 종목의 보유 경로를 ``cut_day`` 행 전에서 끊는다(그 행부터 못 씀). 반환 = 걸린 종목 수."""
        r = int(self.days.searchsorted(pd.Timestamp(cut_day)))
        ci = self.codes.get_indexer(pd.Index(list(codes)))
        ci = ci[ci >= 0]
        self.CUT[ci] = r
        return int(len(ci))


def _ret_cols(entry, part, px1, fin, take_frac):
    h2 = fin / entry - 1.0
    h1 = np.where(part, px1 / entry - 1.0, h2)
    ret = np.where(part, take_frac * h1 + (1.0 - take_frac) * h2, h2)
    return ret, h1, h2


def _p(params):
    return {**BUYSTOP_PARAMS, **(params or {})}


def run_buystop(A: Arrays, cis, ds, pivots, l_last, pids, *, no_overlap: bool = True, old_low=None,
                params: "dict | None" = None) -> dict:
    """정렬된 사건 배열 → 결과 배열 dict(``status`` = :data:`SIM_STATUS` 코드). 주문가·손절가는 여기서 호가로 만든다.
    ``old_low`` 를 주면 X2 옛 손절. 결과의 ``ret`` 은 낙관 경로, ``ret_stop0`` 은 모호 거래의 보수 경로(아니면 ``ret`` 과 같다)."""
    p = _p(params)
    m = len(cis)
    pivots = np.asarray(pivots, float)
    l_last = np.asarray(l_last, float)
    orders = pivots + tick_size_vec(pivots)
    stops_raw = l_last - tick_size_vec(l_last)
    mode = 0 if old_low is None else 1
    ol = np.full(m, np.nan) if old_low is None else np.asarray(old_low, float)
    o = {"status": np.zeros(m, np.int64), "x": np.full(m, -1, np.int64), "fill": np.full(m, np.nan),
         "entry": np.full(m, np.nan), "stop": np.full(m, np.nan), "dd": np.full(m, np.nan), "tgt": np.full(m, np.nan),
         "gap_fill": np.zeros(m, np.bool_), "amb": np.zeros(m, np.bool_), "reason": np.full(m, -1, np.int64),
         "part": np.zeros(m, np.bool_), "part_x": np.full(m, -1, np.int64), "px1": np.full(m, np.nan),
         "fin": np.full(m, np.nan)}
    if m:
        _run_buystop(A.O, A.H, A.L, A.C, A.MA, A.TR, A.PL, A.CUT, np.asarray(cis, np.int64), np.asarray(ds, np.int64),
                     pivots, orders, stops_raw, np.asarray(pids, np.int64), ol, mode, bool(p["ref_order"]),
                     int(p["fb_days"]), float(p["be_r"]), bool(p["sell_tick"]), float(p["gap_max"]),
                     float(p["slip"]), float(p["sell_slip"]), float(p["risk_pre_max"]), float(p["d_floor"]),
                     float(p["take_r"]), float(p["take_frac"]), int(p["max_hold"]), float(p["be_mult"]),
                     float(LIMIT_UP_RATIO), float(p["old_stop_mult"]), float(p["old_stop_floor"]),
                     float(p["old_stop_cap"]), bool(no_overlap), o["status"], o["x"], o["fill"], o["entry"],
                     o["stop"], o["dd"], o["tgt"], o["gap_fill"], o["amb"], o["reason"], o["part"], o["part_x"],
                     o["px1"], o["fin"])
    o["order"] = orders
    o["stop_raw"] = stops_raw
    _attach_rets(o, p)
    s0 = o["stop"] * (1.0 - p["sell_slip"]) - (tick_size_vec(o["stop"]) if p["sell_tick"] else 0.0)
    o["ret_stop0"] = np.where(o["amb"], s0 / o["entry"] - 1.0, o["ret"])
    return o


def _attach_rets(o: dict, p: dict) -> None:
    m = len(o["status"])
    ok = o["status"] == S_OK
    ret, h1, h2 = (np.full(m, np.nan) for _ in range(3))
    if ok.any():
        ret[ok], h1[ok], h2[ok] = _ret_cols(o["entry"][ok], o["part"][ok], o["px1"][ok], o["fin"][ok], p["take_frac"])
    o.update(ret=ret, ret_half1=h1, ret_half2=h2)


def run_open_entry(A: Arrays, cis, ds, dds, params: "dict | None" = None) -> dict:
    """대조(모멘텀 정합): 행 ``ds`` 시가 × 1.001 진입(정정 2 — 매수 슬리피지 대칭), 손절폭 ``dds``, 돌파 실패 피벗 = 진입가, 같은
    청산(진입일 익절 인정 — 시가 진입은 순서가 모호하지 않다, 손절 매도 −0.1% 대칭)."""
    p = _p(params)
    m = len(cis)
    o = {"status": np.zeros(m, np.int64), "x": np.full(m, -1, np.int64), "entry": np.full(m, np.nan),
         "reason": np.full(m, -1, np.int64), "part": np.zeros(m, np.bool_), "part_x": np.full(m, -1, np.int64),
         "px1": np.full(m, np.nan), "fin": np.full(m, np.nan)}
    if m:
        _run_open(A.O, A.H, A.L, A.C, A.MA, A.TR, A.PL, A.CUT, np.asarray(cis, np.int64), np.asarray(ds, np.int64),
                  np.asarray(dds, float), float(p["take_r"]), float(p["take_frac"]), int(p["max_hold"]),
                  float(p["be_mult"]), float(LIMIT_UP_RATIO), float(p["sell_slip"]), float(p["slip"]), int(p["fb_days"]), float(p["be_r"]), bool(p["sell_tick"]),
                  o["status"], o["x"], o["entry"],
                  o["reason"], o["part"], o["part_x"], o["px1"], o["fin"])
    _attach_rets(o, p)
    return o


def run_x1(A: Arrays, cis, sig_rows, pivots, l_last, pids, params: "dict | None" = None) -> dict:
    """X1 칸 — 정렬된 (신호 행, 종목) 종가 돌파 사건 → 다음날 시가 진입·새 손절. :func:`_run_x1_py` 설명."""
    p = _p(params)
    m = len(cis)
    l_last = np.asarray(l_last, float)
    stops_raw = l_last - tick_size_vec(l_last)
    o = {"status": np.zeros(m, np.int64), "x": np.full(m, -1, np.int64), "entry": np.full(m, np.nan),
         "stop": np.full(m, np.nan), "dd": np.full(m, np.nan), "reason": np.full(m, -1, np.int64),
         "part": np.zeros(m, np.bool_), "part_x": np.full(m, -1, np.int64), "px1": np.full(m, np.nan),
         "fin": np.full(m, np.nan)}
    if m:
        _run_x1(A.O, A.H, A.L, A.C, A.MA, A.TR, A.PL, A.CUT, np.asarray(cis, np.int64), np.asarray(sig_rows, np.int64),
                np.asarray(pivots, float), stops_raw, np.asarray(pids, np.int64), float(p["x1_chase"]),
                float(p["d_cap"]), float(p["d_floor"]), float(p["take_r"]), float(p["take_frac"]), int(p["max_hold"]),
                float(p["be_mult"]), float(LIMIT_UP_RATIO), float(p["sell_slip"]), int(p["fb_days"]), float(p["be_r"]),
                bool(p["sell_tick"]), o["status"], o["x"],
                o["entry"],
                o["stop"], o["dd"], o["reason"], o["part"], o["part_x"], o["px1"], o["fin"])
    _attach_rets(o, p)
    return o


def buystop_entry_check(A: Arrays, cis, ds, pivots, l_last, params: "dict | None" = None) -> dict:
    """체결 여부·진입가·손절폭만(청산 없이, 벡터) — 널 후보 거르기용. 규칙은 :func:`_run_buystop_py` 의 주문~체결 부분과 같다
    (겹침·같은 피벗 거르기는 빼고). → ``status, entry, dd, stop, open_le_pivot, stop0_confirmed(진입일 종가 ≤ 손절가), amb`` —
    모호 정의는 시뮬과 같은 식(시가 ≤ 피벗 ∧ 저가 ≤ 손절가 ∧ 종가 > 손절가)."""
    p = _p(params)
    cis = np.asarray(cis, np.int64)
    ds = np.asarray(ds, np.int64)
    P = np.asarray(pivots, float)
    ll = np.asarray(l_last, float)
    od = P + tick_size_vec(P)
    sr = ll - tick_size_vec(ll)
    n = A.O.shape[1]
    m = len(cis)
    st = np.full(m, S_OK, np.int64)
    dsafe = np.clip(ds, 1, n - 1)
    op, hi, prev = A.O[cis, dsafe], A.H[cis, dsafe], A.C[cis, dsafe - 1]
    tr = A.TR[cis, dsafe] & (ds < A.CUT[cis])
    with np.errstate(invalid="ignore", divide="ignore"):
        bad = ~(np.isfinite(P) & np.isfinite(sr) & (P > 0) & (sr > 0))
        ref = od if p["ref_order"] else P
        risk = (ref - sr) / ref > p["risk_pre_max"]
        lu = np.isfinite(prev) & (prev > 0) & (op >= prev * LIMIT_UP_RATIO)
        gap = op > ref * (1.0 + p["gap_max"])
        nofill = hi < od
    for code, mk in ((S_NO_FILL, nofill), (S_GAP, gap), (S_LIMIT_UP, lu), (S_HALT, ~tr), (S_RISK, risk),
                     (S_BAD, bad), (S_NO_NEXT, (ds >= n) | (ds < 1))):
        st[mk] = code
    e = np.maximum(op, od) * (1.0 + p["slip"])
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.maximum((e - sr) / e, p["d_floor"])
    ok = st == S_OK
    with np.errstate(invalid="ignore", divide="ignore"):
        stop = np.where((e - sr) / e < p["d_floor"], e * (1.0 - p["d_floor"]), sr)
        cl, lw = A.C[cis, dsafe], A.L[cis, dsafe]
        confirmed = ok & (cl <= stop)
        amb = ok & (op <= P) & (lw <= stop) & (cl > stop)
    return {"status": st, "entry": np.where(ok, e, np.nan), "dd": np.where(ok, w, np.nan),
            "stop": np.where(ok, stop, np.nan), "open_le_pivot": ok & (op <= P), "stop0_confirmed": confirmed,
            "amb": amb}


TRADE_COLS = ["code", "event_day", "order_day", "entry_day", "exit_day", "fill_px", "entry_px", "exit_px", "ret",
              "exit_reason", "hold_days", "amb", "ret_stop0", "market", "pivot", "l_last", "order_px", "stop_px", "d",
              "over10", "gap_fill", "target_px", "partial_hit", "partial_day", "partial_px", "ret_half1", "ret_half2",
              "R"]


def _trade_frame(A: Arrays, ev: pd.DataFrame, r: dict, entry_rows: np.ndarray, sig_rows: np.ndarray,
                 d_cap: float = BUYSTOP_PARAMS["d_cap"]) -> pd.DataFrame:
    ok = r["status"] == S_OK
    days = A.days
    t = ev.loc[ok].copy()
    er, sr_ = entry_rows[ok], sig_rows[ok]
    x = r["x"][ok]
    px = r["part_x"][ok]
    t["event_day"] = days[sr_]
    t["order_day"] = days[er]
    t["entry_day"] = days[er]
    t["exit_day"] = days[x]
    t["entry_px"] = r["entry"][ok]
    t["exit_px"] = r["fin"][ok]
    t["ret"] = r["ret"][ok]
    t["exit_reason"] = np.asarray(REASONS, dtype=object)[r["reason"][ok]]
    t["hold_days"] = x - sr_
    if "stop" in r:
        t["stop_px"] = r["stop"][ok]
    if "dd" in r:
        t["d"] = r["dd"][ok]
        t["over10"] = r["dd"][ok] > d_cap + 1e-12
        t["R"] = t["ret"] / t["d"]
    t["partial_hit"] = r["part"][ok]
    t["partial_day"] = pd.DatetimeIndex([days[k] if k >= 0 else pd.NaT for k in px])
    t["partial_px"] = r["px1"][ok]
    t["ret_half1"] = r["ret_half1"][ok]
    t["ret_half2"] = r["ret_half2"][ok]
    t["exit_row"] = x
    t["entry_row"] = er
    return t


def simulate_buystop(A: Arrays, ev: pd.DataFrame, *, no_overlap: bool = True, old_stop: bool = False,
                     params: "dict | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """셋업 표 → ``(트레이드, 버림)``. ``ev`` 열: ``code, day_idx(주문일 행), pivot, l_last, ip(피벗 위치)`` (+ X2 면 ``old_low``)
    + 그 밖의 열은 트레이드 표로 따라간다. 정렬 = (주문일, 종목). 트레이드 열 ``ret``·``exit_*`` = 낙관 경로, ``amb``·``ret_stop0``
    = 진입일 손절 모호 여부·보수 경로 수익, ``stop0_day`` = 보수 경로 청산일(= 진입일)."""
    col_of = pd.Series(np.arange(len(A.codes)), index=A.codes)
    ev = ev.copy()
    ev["_c"] = col_of.reindex(ev["code"]).to_numpy()
    ev = ev[ev["_c"].notna()].copy()
    ev["_c"] = ev["_c"].astype(np.int64)
    ev = ev.sort_values(["day_idx", "code"], kind="mergesort").reset_index(drop=True)
    r = run_buystop(A, ev["_c"].to_numpy(), ev["day_idx"].to_numpy(), ev["pivot"].to_numpy(float),
                    ev["l_last"].to_numpy(float), ev["ip"].to_numpy(), no_overlap=no_overlap,
                    old_low=ev["old_low"].to_numpy(float) if old_stop else None, params=params)
    ok = r["status"] == S_OK
    di = ev["day_idx"].to_numpy()
    sk = ev.loc[~ok, ["code", "day_idx"]].copy()
    sk["skip_reason"] = np.asarray(SIM_STATUS, dtype=object)[r["status"][~ok]]
    t = _trade_frame(A, ev, r, di, di - 1, _p(params)["d_cap"])
    t["fill_px"] = r["fill"][ok]
    t["order_px"] = r["order"][ok]
    t["gap_fill"] = r["gap_fill"][ok]
    t["target_px"] = r["tgt"][ok]
    t["amb"] = r["amb"][ok]
    t["ret_stop0"] = r["ret_stop0"][ok]
    t = t.drop(columns=["_c"])
    if "market" not in t.columns:
        t["market"] = np.nan
    lead = [c for c in TRADE_COLS if c in t.columns]
    t = t[lead + [c for c in t.columns if c not in lead]]
    return t.reset_index(drop=True), sk.reset_index(drop=True)


def simulate_x1(A: Arrays, ev: pd.DataFrame, params: "dict | None" = None) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """X1 — ``ev`` 열 ``code, sig_row(종가 돌파 신호 행), pivot, l_last, ip``. 진입 = 신호 행 + 1 시가."""
    col_of = pd.Series(np.arange(len(A.codes)), index=A.codes)
    ev = ev.copy()
    ev["_c"] = col_of.reindex(ev["code"]).to_numpy().astype(np.int64)
    ev = ev.sort_values(["sig_row", "code"], kind="mergesort").reset_index(drop=True)
    sr_ = ev["sig_row"].to_numpy()
    r = run_x1(A, ev["_c"].to_numpy(), sr_, ev["pivot"].to_numpy(float), ev["l_last"].to_numpy(float),
               ev["ip"].to_numpy(), params=params)
    ok = r["status"] == S_OK
    sk = ev.loc[~ok, ["code", "sig_row"]].copy()
    sk["skip_reason"] = np.asarray(SIM_STATUS, dtype=object)[r["status"][~ok]]
    t = _trade_frame(A, ev, r, sr_ + 1, sr_, _p(params)["d_cap"])
    t["amb"] = False
    t["ret_stop0"] = t["ret"]
    t["fill_px"] = t["entry_px"]
    t["gap_fill"] = False
    return t.drop(columns=["_c"]).reset_index(drop=True), sk.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 같은 메뉴 무작위 셋업(널) — 피벗 = 직전 k 일 최고 고가, 손절 기준 = 직전 j 일 최저 저가
# ---------------------------------------------------------------------------

def _null_levels_py(H, L, cis, ss, ks, js, piv, ip, ll):
    for i in prange(len(cis)):
        ci = cis[i]
        s = ss[i]
        k = ks[i]
        j = js[i]
        best = -np.inf
        bi = -1
        ok = s - k + 1 >= 0 and s - j + 1 >= 0
        if ok:
            for q in range(s - k + 1, s + 1):
                h = H[ci, q]
                if not h == h:
                    ok = False
                    break
                if h >= best:                         # 동률 = 늦은 봉(VCP 와 같은 규약)
                    best = h
                    bi = q
        lo = np.inf
        if ok:
            for q in range(s - j + 1, s + 1):
                v = L[ci, q]
                if not v == v:
                    ok = False
                    break
                if v < lo:
                    lo = v
        if ok:
            piv[i] = best
            ip[i] = bi
            ll[i] = lo
        else:
            piv[i] = np.nan
            ip[i] = -1
            ll[i] = np.nan


_null_levels = _jit(_null_levels_py, parallel=True)
#: 직렬판 — fork 한 작업자 안에서는 numba 스레드 풀을 쓰지 않는다(부모가 띄운 스레드 풀은 fork 뒤 안전하지 않다).
_null_levels_serial = _jit(_null_levels_py)


def null_levels(A: Arrays, cis, ss, ks, js, *, parallel: bool = True) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """(종목, 신호 행 ``s``, k, j) → (피벗 = max high[s−k+1..s], 그 위치(동률 늦은 봉), 손절 기준 = min low[s−j+1..s])."""
    m = len(cis)
    piv, ip, ll = np.full(m, np.nan), np.full(m, -1, np.int64), np.full(m, np.nan)
    if m:
        (_null_levels if parallel else _null_levels_serial)(
            A.H, A.L, np.asarray(cis, np.int64), np.asarray(ss, np.int64), np.asarray(ks, np.int64),
            np.asarray(js, np.int64), piv, ip, ll)
    return piv, ip, ll


# ---------------------------------------------------------------------------
# 4. 실적 — 분기 단독 순이익(Q4 = 연간 − 9M) · E 가속 (§6 · 정정 M2)
# ---------------------------------------------------------------------------

E_PARAMS = {"min_yoy": 0.25, "min_rev_yoy": 0.10}     # 정정 M2: 140일 문 삭제
E_STATE_COLS = ("code", "t", "period", "avail", "netinc", "netinc_prior", "yoy", "rev_yoy", "prev_period",
                "prev_avail", "prev_netinc", "prev_netinc_prior", "yoy_prev")


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace("-", "").str[:8], format="%Y%m%d", errors="coerce")


def _yoy(cur: float, prior: float) -> float:
    if not (np.isfinite(cur) and np.isfinite(prior)) or prior == 0:
        return np.nan
    return (cur - prior) / abs(prior)


def _standalone(vis: dict, q: int) -> "tuple[float, float, float, float, object]":
    """분기 서수 ``q`` 의 단독 (순이익, 전년 동기 순이익, 매출, 전년 동기 매출, avail). Q1~Q3 = 그 행, Q4 = 연간 행 − 같은 회계연도
    Q1~Q3 행(값마다; 전년 동기도 **그 행들의** ``_prior`` 비교 수치끼리). 필요한 행이 그날 안 보이거나 값이 NaN 이면 NaN."""
    nan = np.nan
    r = vis.get(q)
    if r is None:
        return nan, nan, nan, nan, None
    if q % 4 != 3:
        return r[0], r[1], r[2], r[3], r[4]
    parts = [vis.get(q - k) for k in (3, 2, 1)]
    if any(x is None for x in parts):
        return nan, nan, nan, nan, r[4]
    out = []
    for f in range(4):
        v = r[f] - sum(x[f] for x in parts)
        out.append(float(v) if np.isfinite(v) else nan)
    return out[0], out[1], out[2], out[3], r[4]


def _pname(q: int) -> str:
    return f"{q // 4}Q{q % 4 + 1}"


def standalone_quarter_states(ea: pd.DataFrame, *, avail_col: str = "avail_date",
                              knowledge_col: "str | None" = "knowledge_date") -> pd.DataFrame:
    """실적 전 버전(``read_earnings(all_versions=True)``) → 종목별 "상태가 바뀌는 순간" 표(:data:`E_STATE_COLS`).

    **이중 시간축:** 한 버전은 ``max(avail, knowledge)`` 부터 보이고, 분기마다 그때까지 보인 버전 중 ``knowledge`` 가 가장 늦은 것을
    쓴다(:mod:`swing_it.features.earnings_leg` 와 같은 규약). **최신 분기 = 그날 보이는 가장 늦은 분기** — 단독 값·YoY 가 계산 불가여도
    앞 분기로 밀리지 않는다(정정 M2 — 그날 E 는 모름 → 거짓). 직전 분기 = 최신 − 1, 그날 알던 버전으로 같은 식.

    - 단독 순이익: Q1~Q3 = 그 행 그대로(DART 3개월), **Q4 = 연간 − (Q1+Q2+Q3)** — 같은 회계연도 세 분기가 그날 보일 때만.
    - 전년 동기: 그 행들의 ``netinc_prior``(당해 보고서 전기 비교 수치)를 같은 식으로. 매출도 같다.
    - YoY = (당기 − 전기)/|전기|, 전기 0·NaN 이면 NaN. ``yoy_prev`` 는 직전 분기 전년 동기 > 0 일 때만 유한(정정 M2).
    """
    df = ea.copy()
    df["code"] = df["code"].astype(str)
    df["_avail"] = _to_dt(df[avail_col])
    df["_know"] = _to_dt(df[knowledge_col]) if knowledge_col is not None else df["_avail"]
    df = df.dropna(subset=["_avail", "_know"])
    df["_vf"] = df[["_avail", "_know"]].max(axis=1)
    p = df["period"].astype(str)
    df["_q"] = p.str[:4].astype(int) * 4 + p.str[-1].astype(int) - 1
    for c in ("netinc", "netinc_prior", "revenue", "revenue_prior"):
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values(["code", "_vf", "_know", "_q"], kind="mergesort")
    rows = []
    codes = df["code"].to_numpy()
    vf = df["_vf"].to_numpy()
    kn = df["_know"].to_numpy()
    qq = df["_q"].to_numpy()
    av = df["_avail"].to_numpy()
    vals = df[["netinc", "netinc_prior", "revenue", "revenue_prior"]].to_numpy(float)
    n = len(df)
    i = 0
    while i < n:
        code = codes[i]
        vis: dict = {}
        know: dict = {}
        while i < n and codes[i] == code:
            t = vf[i]
            while i < n and codes[i] == code and vf[i] == t:
                q = int(qq[i])
                if q not in know or kn[i] >= know[q]:
                    know[q] = kn[i]
                    vis[q] = (vals[i, 0], vals[i, 1], vals[i, 2], vals[i, 3], av[i])
                i += 1
            latest = max(vis)
            ni, pr, rv, rp, avail = _standalone(vis, latest)
            pni, ppr, _, _, pav = _standalone(vis, latest - 1)
            yp = _yoy(pni, ppr) if (np.isfinite(ppr) and ppr > 0) else np.nan
            rows.append((code, t, _pname(latest), avail, ni, pr, _yoy(ni, pr), _yoy(rv, rp), _pname(latest - 1), pav,
                         pni, ppr, yp))
    out = pd.DataFrame(rows, columns=list(E_STATE_COLS))
    for c in ("t", "avail", "prev_avail"):
        out[c] = pd.to_datetime(out[c])
    return out


def e_accel_asof(states: pd.DataFrame, codes, dates, params: "dict | None" = None) -> pd.DataFrame:
    """(종목, 날짜) 쌍마다 그날(그날 포함) 알던 최신 상태 → ``e_known, e_accel`` + 값 열. 순서는 입력 순서.

    E 가속 = 최신 분기 단독 순이익 > 0 ∧ 전년 동기 > 0 ∧ YoY ≥ 25% ∧ 직전 분기 YoY 유한(전년 동기 > 0) ∧ 최신 YoY > 직전 YoY ∧
    최신 분기 매출 YoY ≥ 10%. ``e_known`` = 그날 상태가 있고 최신 분기 단독 순이익·전기가 유한. ``e_age`` 는 기술(문 아님).
    """
    p = {**E_PARAMS, **(params or {})}
    q = pd.DataFrame({"code": pd.Series(codes, dtype=str).to_numpy(),
                      "date": pd.to_datetime(pd.Series(dates)).astype("datetime64[ns]").to_numpy(),
                      "_i": np.arange(len(codes))})
    cols = ["e_known", "e_accel", "e_period", "e_avail", "e_netinc", "e_netinc_prior", "e_yoy", "e_yoy_prev",
            "e_rev_yoy", "e_age", "e_prev_avail", "e_prev_netinc", "e_prev_netinc_prior"]
    if len(q) == 0 or states.empty:
        out = pd.DataFrame(np.nan, index=range(len(q)), columns=cols)
        out["e_known"] = False
        out["e_accel"] = False
        return out
    st = states.assign(t=states["t"].astype("datetime64[ns]"), avail=states["avail"].astype("datetime64[ns]"),
                       prev_avail=states["prev_avail"].astype("datetime64[ns]")).sort_values("t")
    m = pd.merge_asof(q.sort_values("date"), st, left_on="date", right_on="t", by="code", direction="backward")
    m = m.sort_values("_i").reset_index(drop=True)
    age = (m["date"] - m["avail"]).dt.days
    ni, pr = m["netinc"].astype(float), m["netinc_prior"].astype(float)
    y, yp, ry = m["yoy"].astype(float), m["yoy_prev"].astype(float), m["rev_yoy"].astype(float)
    known = np.isfinite(ni) & np.isfinite(pr)
    acc = (known & (ni > 0) & (pr > 0) & (y >= p["min_yoy"]) & np.isfinite(yp) & (y > yp)
           & (ry >= p["min_rev_yoy"]))
    return pd.DataFrame({"e_known": known.to_numpy(bool), "e_accel": acc.fillna(False).to_numpy(bool),
                         "e_period": m["period"].to_numpy(), "e_avail": m["avail"].to_numpy(),
                         "e_netinc": ni.to_numpy(), "e_netinc_prior": pr.to_numpy(), "e_yoy": y.to_numpy(),
                         "e_yoy_prev": yp.to_numpy(), "e_rev_yoy": ry.to_numpy(), "e_age": age.to_numpy(float),
                         "e_prev_avail": m["prev_avail"].to_numpy(),
                         "e_prev_netinc": m["prev_netinc"].astype(float).to_numpy(),
                         "e_prev_netinc_prior": m["prev_netinc_prior"].astype(float).to_numpy()})


def eps_accel(e: pd.DataFrame, sh_now, sh_yago, sh_prev, sh_prev_yago, params: "dict | None" = None) -> np.ndarray:
    """주당 기준 E 가속(기술 칸, 정정 M2). 주식수 = 각 분기 ``avail`` 시점 as-of 와 그 1년 전 as-of(호출부가 조회). 순이익 조건은
    :func:`e_accel_asof` 와 같고 YoY 만 주당 순이익으로 다시 잰다(매출 조건은 그대로)."""
    p = {**E_PARAMS, **(params or {})}
    with np.errstate(invalid="ignore", divide="ignore"):
        c, pr = e["e_netinc"] / sh_now, e["e_netinc_prior"] / sh_yago
        pc, ppr = e["e_prev_netinc"] / sh_prev, e["e_prev_netinc_prior"] / sh_prev_yago
        y = (c - pr) / np.abs(pr)
        yp = np.where(ppr > 0, (pc - ppr) / np.abs(ppr), np.nan)
    ok = (np.isfinite(c) & np.isfinite(pr) & (c > 0) & (pr > 0) & (y >= p["min_yoy"]) & np.isfinite(yp) & (y > yp)
          & (e["e_rev_yoy"] >= p["min_rev_yoy"]))
    return np.asarray(ok, bool)


# ---------------------------------------------------------------------------
# 5. 수급 — 기반 기간 외국인+기관 누적 순매수 / 거래량 (§7 · 정정 M3)
# ---------------------------------------------------------------------------

FLOW_PARAMS = {"min_cover": 0.80}


class FlowCum:
    """``F_base`` 용 누적합(가격 달력 × 종목). 입력 = ``flow_leg.flow_wide`` 결과(외국인·기관·``acc_trde_qty``).

    - ``weight=None``(기술): Σ(외국인+기관) / Σ``acc_trde_qty`` — 셋 다 알려진 날만 더한다.
    - ``weight=거래대금 패널``(주, 정정 M3 — ``flow_leg`` 관례): 하루 비율 (외국인+기관)/``acc_trde_qty`` 의 거래대금 가중평균.
      거래량 0 ∧ 순매수 0 인 날은 알려진 0, 거래량 0 인데 순매수 ≠ 0 은 모름. 주식수 기준 단절에 불변.
    알려진 날 < ``min_cover`` × 창 길이면 모름(NaN).
    """

    def __init__(self, foreign: pd.DataFrame, institution: pd.DataFrame, volume: pd.DataFrame,
                 weight: "pd.DataFrame | None" = None):
        f = foreign.to_numpy(float)
        i_ = institution.reindex_like(foreign).to_numpy(float)
        v = volume.reindex_like(foreign).to_numpy(float)
        net = f + i_
        z = np.zeros((1, f.shape[1]))
        if weight is None:
            known = np.isfinite(net) & np.isfinite(v)
            num, den = np.where(known, net, 0.0), np.where(known, v, 0.0)
        else:
            w = weight.reindex_like(foreign).to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                daily = np.where(v > 0, net / np.where(v > 0, v, 1.0), np.nan)
            daily = np.where((v == 0) & (net == 0), 0.0, daily)
            known = np.isfinite(daily) & np.isfinite(w)
            num, den = np.where(known, daily * w, 0.0), np.where(known, w, 0.0)
        self.num = np.vstack([z, np.cumsum(num, axis=0)])
        self.den = np.vstack([z, np.cumsum(den, axis=0)])
        self.cnt = np.vstack([z, np.cumsum(known.astype(float), axis=0)])
        self.index, self.columns = foreign.index, foreign.columns

    def f_base(self, ci, lo, hi, min_cover: float = FLOW_PARAMS["min_cover"]) -> np.ndarray:
        """행 ``lo+1 .. hi`` (양끝 포함 — 사전등록 i0+1 ~ d−1) 의 비율. 알려진 날 < ``min_cover`` × 길이면 NaN."""
        ci, lo, hi = (np.asarray(a, np.int64) for a in (ci, lo, hi))
        ok = (lo >= 0) & (hi > lo) & (ci >= 0)
        lo_, hi_, ci_ = np.where(ok, lo, 0), np.where(ok, hi, 0), np.where(ok, ci, 0)
        a, b = lo_ + 1, hi_ + 1                     # 누적 배열은 한 줄 밀렸다: 행 r 까지 합 = cum[r+1]
        num = self.num[b, ci_] - self.num[a, ci_]
        den = self.den[b, ci_] - self.den[a, ci_]
        cnt = self.cnt[b, ci_] - self.cnt[a, ci_]
        length = (hi_ - lo_).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / np.where(den > 0, den, np.nan)
        return np.where(ok & (cnt >= min_cover * length) & (den > 0), r, np.nan)
