"""패턴 GA 사건 표 — (θ × 천장 창 × 진입 기준선) 격자마다 수렴 패턴의 **첫 관통일** 사건과 d−1 서술자.

사전등록 정본: ``research/logs/sepa_explore/PATTERN_GA.md`` — aeaf8e5 + 정정 0 ab612dc + **정정 1 72fab71**
(``REVIEW_PATTERN_GA.md`` "정정 1 초안" 12항목 + MINOR 1~12 를 그대로 채택 — 그 리뷰 절이 정본) + **정정 2 c483edb**(급등 뒤 쉼 맥락)
+ **정정 3 2154f77**(천장 종류·H0 = 첫 점·바닥 진입·위생 보강), 모두 결과 보기 전. 이 파일은 사건 표(설정과 무관한 고정 표)만
만든다. 거르기·체결·청산·포트폴리오는 :mod:`research.signals.pattern_fast`, 유전자·적합도·널·바는 러너
(``research/experiments/pattern_ga_run.py``). numba 는 선택(:mod:`research.signals.vcp` 규약). 배열은 **종목×날짜**(전치).

**날짜 규약:** 셋업 행 ``s = d − 1`` 종가까지의 자료로 모양·맥락을 정하고, 주문은 ``d`` 장중 매수정지(기준선 + 1호가).

**사전등록 문구 → 구현 해석(모호한 곳은 전부 여기 — 결과 보기 전 결정):**

1. 천장 H0 = ``s−cw+1..s`` 창 최고 고가(동률 = 늦은 봉), ``cw`` ∈ :data:`CEIL_WINS`. 창 안 결측이면 패턴 없음.
2. 지그재그 = :func:`vcp._zigzag_py` (θ ∈ :data:`THETAS`, 극값 먼저·반전 확인은 다음 봉부터, 동률 = 늦은 봉, i0 의 H0 에서 시작).
   첫 확정 고점은 늘 H0 다(창 최고라 그 뒤 고가는 H0 미만).
3. **두 패턴 종류(정정 3 ``h0_in_pattern``) — 한 (종목, s) 는 정확히 하나에 든다:**
   - ``h0in = 0`` (H0 는 패턴 밖, 정정 1 C3-vi): 확정 고점 ≥ 3. D = 마지막 확정 고점, B = 그 앞 확정 고점, C = B–D 사이 확정 저점,
     A = B 앞 확정 저점(패턴 첫 저점). 상단선 = (B, D), 하단선 = (C, E). 패턴 시작 = A.
   - ``h0in = 1`` (H0 = 패턴 첫 고점, ``WEDGE_D_TEST.md`` §2 와 같은 연속 조건): 확정 고점이 정확히 2(H0, D) — H0 와 D 사이 반전점은
     확정 저점 C 하나뿐. 상단선 = (H0, D), 하단선 = (C, E). 패턴 시작 = H0, 패턴 첫 저점 = C.
   - 공통: **E = D 뒤 ~ s 최저 저가(진행 중 저점 허용, 동률 = 늦은 봉)**. 로그가격 직선, 행 번호가 x. 모양(기울기 부호·수렴)은
     거르지 않는다 — 유전자가 고른다.
4. **패턴 식별 = (종목, i0, 행(D))** — E 가 내려가도 같은 패턴. 확정 반전점은 s 가 늘어도 바뀌지 않는다(i0 에서 순차). 체결은 패턴당
   하나(바닥 반전은 E 마다 사건이 따로 있을 수 있어 포트폴리오가 같은 패턴 두 번째 체결을 막는다).
5. **기준선(d):** 추세선 = U(d), 수평 = D, H0 = H0, **바닥 반전(정정 3)** = E 봉 고가. 주문가 = 기준선 + 1호가.
6. **첫 관통(정정 1 M5 · 정정 3):** 관통 = 거래 가능일 고가 ≥ 주문가(선 값 + 1호가, 그날 선 값). 패턴 식별·기준선마다 첫 관통일 d
   하나만 후보 — 패턴이 정의된 뒤의 관통은 물론, **정의 전(D 확정 전)의 관통도 셈에 넣는다**: s 에서 D 뒤(바닥 반전은 E 뒤) ~ s 에
   이미 관통한 봉이 있으면 그 패턴은 소진(정정 3: 첫 관통 판정과 저점 뒤 대기가 겹쳐 두 번째 관통을 사는 경로 차단). 그날 위생·
   맥락 거르기가 실패하면 그 패턴은 끝. 거르기(RS·breadth·단계·실적·모양 유전자·갭·손절 상한)도 전부 이 행의 d−1 값에서만
   (:mod:`pattern_fast`). ``gap_max`` 초과·상한가 시가로 안 산 날도 소진. 정지일(거래 불가)은 관통이 아니다.
7. **고정 위생(정정 1 C3-3 · 정정 3):** 연속 반전점 간격 ≥ 3 거래일(``h0in`` 0: A→B·B→C·C→D·D→E, 1: H0→C·C→D·D→E), 패턴 길이
   (시작 → s) ≥ 15 거래일, 무결성 = 시작 ~ E 종가 ≤ 상단선 × 1.01 ∧ 시작 ~ s 종가 ≥ 하단선 × 0.99 (리뷰 C3-3 "종가 ±1%"),
   **d−1 종가 ≤ 주문가 × 0.995**(정정 3 — 사실상 시가 매수 차단), 바닥 반전은 d − 행(E) ≤ 5. E 뒤 ~ s 상단 쪽은 6 이 맡는다
   (추세선 기준선이면 그 구간 고가 < U + 1호가라 종가 ±1% 를 자동 만족 — 수평·H0·바닥 기준선은 상단 추세선을 먼저 넘는 것을 허용).
8. **기본 맥락(가장 느슨한 ``stage`` — 모든 단계가 포함):** s 에서 유니버스(위생 ∧ ADV20 ≥ 10억) ∧ MA50 > MA150 > MA200 ∧ 분할 직후
   60 거래일 아님. 첫 관통일에 실패해도 소진(6).
9. **정규화 모양 좌표(정정 1 C3-1):** L = s − 패턴 시작, h = ln(max 고가 / min 저가)[시작..s], ŝ = 기울기 × L / h (상단·하단),
   폭 비 ρ = (ln U − ln 하단)(s) / (ln U − ln 하단)(시작) (< 1 좁아짐; 시작 폭 ≤ 0 이면 NaN). A 닿음 = A / 하단선(A) − 1
   (MINOR 6 — 하단선 두 저점 직전 저점; ``h0in`` 1 은 저점이 둘뿐이라 NaN → ``touch_lo`` = 3 에서 탈락).
10. 천장 거리 = **상단선(d)** / H0 (기준선과 무관 — 원문 "상단선(d) / H0"). 깊이 = (H0 − min 저가[i0..s]) / H0. 거래량 마름(MINOR 3)
    = mean(vol[D+1..s]) / mean(vol[i0−70..i0−21]) (H0 전 20일을 뺀 50일). 정정 2·3: H0 전 상승률(창 w ∈ {20, 40, 60}) = 저가 기준
    H0 / min(저가[i0−w..i0]) − 1 하나, 쉼 거리 = 패턴 첫 저점 행 − i0.
11. **수평 저항 수(정정 3 ``ceil_kind``):** 천장 창 안 확정 고점(창 시작부터 i0 전까지 같은 θ 지그재그로 확정된 고점 + H0 + H0 뒤 확정
    고점) 중 H0 × 0.97 이상인 수. ``ceil_kind`` = 수평 저항이면 ≥ 2 를 요구, 단일 고점 H0 면 요구 없음(러너).
12. **호가(2단계 리뷰 MINOR 1):** 1호가는 **그날의 표** — 2023-01-25 전은 옛 표(시장별: 1천 1 · 5천 5 · 1만 10 · 5만 50 · 그 위
    코스피 10만 100 · 50만 500 · 1,000 / 코스닥 100), 그 뒤는 통합 표(:func:`korea.tick_size_vec`). 주가가 분할 조정가라 격자 자체가
    깨져 있어 격자 올림·내림 정렬은 하지 않는다(선 값 + 그날 1호가). 손절·본전 체결 −1호가는 :mod:`pattern_fast`(쐐기 D 와 같다).
13. 손절 후보(정정 1 MINOR 4): E − 1호가, 하단선(d) − 1호가. 하단선 손절은 하단선(d) ≥ 종가(s) 면 무효(:mod:`pattern_fast`).
    바닥 반전 진입은 손절 = E − 1호가 고정, 익절 전 돌파 실패 규칙 없음(정정 3). 정정 5 손절 여유용으로 ATR20(d−1)(``atr_s``)을 싣는다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.signals import vcp as V
from research.signals.vcp import _jit, prange
from swing_it.features.korea import tick_size_vec

# --- 사전등록 격자·위생(PATTERN_GA.md 정정 1·3 — 탐색하지 않는다) ---
THETAS = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
CEIL_WINS = (60, 120, 260)
BASES = ("trend", "horiz", "h0", "bottom")   # 진입 기준선: 상단 추세선 · 마지막 확정 고점 수평선 · H0 · 바닥 반전(E 봉 고가)
B_TREND, B_HORIZ, B_H0, B_BOTTOM = range(4)
RUNUP_WINS = (20, 40, 60)                    # 정정 2
HYGIENE = {"gap_min": 3, "span_min": 15, "integ_tol": 0.01, "close_room": 0.995, "bottom_age": 5,
           "dry_skip": 20, "dry_win": 50, "res_tol": 0.03}
#: 모양 섞은 널(정정 1 C1-2b)이 사건끼리 같은 달 안에서 **행 단위로 함께** 재배정하는 열 — 진입일·가격·손절·맥락(RS·breadth·단계·
#: 실적·정정 2 급등·쉼)·청산 열은 건드리지 않는다. 정정 3 의 패턴 종류·수평 저항 수도 모양 서술자라 함께 섞는다.
SHAPE_COLS = ("s_up", "s_lo", "rho", "touch", "span", "ceil", "depth", "dry", "n_res", "h0in")

(ST_OK, ST_NO_DATA, ST_FEW, ST_NO_E) = range(4)
(H_OK, H_SPACING, H_SPAN, H_INTEG_UP, H_INTEG_LO, H_CLOSE, H_BOTTOM_AGE, H_CONTEXT) = range(8)
HYG_STATUS = ("ok", "spacing<3", "span<15", "integrity_upper", "integrity_lower", "close>order*0.995", "bottom_age>5",
              "context(d-1)")

#: 사건 표 실수 열(``F``) — 이름 → 열 번호.
F_COLS = ("level", "order", "h0", "A", "B", "Cl", "D", "E", "su", "sl", "s_up", "s_lo", "rho", "touch", "span", "hgt",
          "ceil", "depth", "dry", "runup20", "runup40", "runup60", "rest_gap", "lower_d", "u_d", "hE", "n_res", "h0in",
          "stopE_raw", "stopL_raw", "open_d", "high_d", "low_d", "close_d", "close_s", "adv20", "rs", "breadth",
          "above50", "tt_s", "tt_h0", "e_yoy", "e_yoy_prev", "market", "strat", "month", "atr_s",
          "e_q", "e_q_prev",             # FWM M1: 최신 분기·전년 같은 분기 단독 순이익(EPS 전년 동기 대비 상승 필수 — 러너가 채운다)
          "mkt_br200", "mkt_nhnl", "mkt_bos",   # FWM M7: 그 종목 시장의 d−1 국면(MA200 위 종목 비율, 신고가 비율, 돌파 성공률 20)
          "ma50_up")                            # 진화 2세대: d−1 MA50 > 20 거래일 전 MA50(러너가 채운다)
FI = {k: i for i, k in enumerate(F_COLS)}
#: 사건 표 정수 열(``I``) — iA = 패턴 첫 저점(정정 2 의 A), iS = 패턴 시작(``h0in`` 0: A, 1: H0).
I_COLS = ("d", "ci", "i0", "iA", "iB", "iC", "iD", "iE", "iS", "table")
II = {k: i for i, k in enumerate(I_COLS)}


def table_id(ti: int, wi: int, bi: int) -> int:
    """(θ 번호, 천장 창 번호, 기준선 번호) → 표 번호."""
    return (ti * len(CEIL_WINS) + wi) * len(BASES) + bi


N_TABLES = len(THETAS) * len(CEIL_WINS) * len(BASES)


def table_key(tid: int) -> "tuple[float, int, str]":
    bi = tid % len(BASES)
    wi = (tid // len(BASES)) % len(CEIL_WINS)
    ti = tid // (len(BASES) * len(CEIL_WINS))
    return THETAS[ti], CEIL_WINS[wi], BASES[bi]


# ---------------------------------------------------------------------------
# 0. 호가단위(numba) — 통합 표(``korea.tick_size_vec`` 와 같다) + 2023-01-25 전 옛 표(날짜별)
# ---------------------------------------------------------------------------

TICK_REFORM_DAY = "2023-01-25"

def _tick_py(p):
    if p < 2000.0:
        return 1.0
    if p < 5000.0:
        return 5.0
    if p < 20000.0:
        return 10.0
    if p < 50000.0:
        return 50.0
    if p < 200000.0:
        return 100.0
    if p < 500000.0:
        return 500.0
    return 1000.0


_tick = _jit(_tick_py)


def _tick_dated_py(p, row, mkt, reform):
    """행 ``row`` 의 호가단위 — ``row ≥ reform``(2023-01-25 행) 이면 통합 표, 아니면 옛 표(``mkt`` 1 = 코스닥)."""
    if row >= reform:
        return _tick(p)
    if p < 1000.0:
        return 1.0
    if p < 5000.0:
        return 5.0
    if p < 10000.0:
        return 10.0
    if p < 50000.0:
        return 50.0
    if mkt == 1:
        return 100.0
    if p < 100000.0:
        return 100.0
    if p < 500000.0:
        return 500.0
    return 1000.0


tick_dated = _jit(_tick_dated_py)


def tick_dated_vec(p, rows, mkt, reform: int) -> np.ndarray:
    """벡터판 :func:`_tick_dated_py` (NaN·0 이하 → NaN)."""
    p = np.asarray(p, float)
    rows = np.broadcast_to(np.asarray(rows), p.shape)
    mkt = np.broadcast_to(np.asarray(mkt), p.shape)
    new = tick_size_vec(p)
    pz = np.where(np.isfinite(p), p, 0.0)
    old = np.select([pz < 1000, pz < 5000, pz < 10000, pz < 50000, mkt == 1, pz < 100000, pz < 500000],
                    [1.0, 5.0, 10.0, 50.0, 100.0, 100.0, 500.0], 1000.0)
    out = np.where(rows >= reform, new, old)
    return np.where(np.isfinite(p) & (p > 0), out, np.nan)


# ---------------------------------------------------------------------------
# 1. 구조(한 행) — H0·지그재그·패턴 점
# ---------------------------------------------------------------------------

def _struct_py(H, L, C, ci, s, theta, cw, hv, hi_, lv, li_):
    """종목 ``ci`` 행 ``s`` 까지 → ``(상태, i0, h0, nh, iE, E)``. 반전점은 버퍼에. 상태 = :data:`ST_OK` 면 패턴이 정의된다
    (확정 고점 ≥ 2 — 2 면 ``h0in`` 1, ≥ 3 이면 0)."""
    s0 = s - cw + 1
    if s0 < 0:
        return ST_NO_DATA, -1, np.nan, 0, -1, np.nan
    i0 = -1
    h0 = -np.inf
    for j in range(s0, s + 1):
        hj = H[ci, j]
        if not (hj == hj and L[ci, j] == L[ci, j] and C[ci, j] == C[ci, j]):
            return ST_NO_DATA, -1, np.nan, 0, -1, np.nan
        if hj >= h0:
            h0 = hj
            i0 = j
    nh, nl, up, hc, ihc, lc, ilc = V._zigzag(H, L, ci, i0, s, theta, hv, hi_, lv, li_)
    if nh < 2:
        return ST_FEW, i0, h0, nh, -1, np.nan
    iD = hi_[nh - 1]
    E = np.inf
    iE = -1
    for j in range(iD + 1, s + 1):                    # E = D 뒤 최저 저가(진행 중, 동률 = 늦은 봉)
        if L[ci, j] <= E:
            E = L[ci, j]
            iE = j
    if iE < 0:
        return ST_NO_E, i0, h0, nh, -1, np.nan
    return ST_OK, i0, h0, nh, iE, E


_struct = _jit(_struct_py)


def _lines(hv, hi_, lv, li_, nh, iE, E):
    """→ (iU0, U0, iD, D, iC, Cl, su, sl): 상단선 = (U0, D), 하단선 = (Cl, E)."""
    iD = hi_[nh - 1]
    D = hv[nh - 1]
    iU0 = hi_[nh - 2]                                 # nh == 2 면 H0(h0in 1), 아니면 B
    U0 = hv[nh - 2]
    iC = li_[nh - 2]
    Cl = lv[nh - 2]
    su = (math.log(D) - math.log(U0)) / (iD - iU0)
    sl = (math.log(E) - math.log(Cl)) / (iE - iC)
    return iU0, U0, iD, D, iC, Cl, su, sl


_lines_j = _jit(_lines)


def _scan_py(H, L, C, MK, reform, cis, ss, theta, cw, st, i0o, iDo, iEo, lvt, Do, h0o, hEo, pre):
    """행마다 구조 → 첫 관통 탐색 재료(상태·i0·행(D)·행(E)·U(d)·D·H0·E 봉 고가) + 기준선별 '이미 관통' 비트 ``pre``
    (1 추세선: D 뒤 ~ s 고가 ≥ U(j) + 1호가, 2 수평: D 뒤 ~ s 고가 ≥ D + 1호가, 8 바닥: E 뒤 ~ s 고가 ≥ E 봉 고가 + 1호가). 병렬."""
    cap = cw + 3
    for k in prange(len(cis)):
        ci = cis[k]
        s = ss[k]
        hv = np.empty(cap)
        hi_ = np.empty(cap, dtype=np.int64)
        lv = np.empty(cap)
        li_ = np.empty(cap, dtype=np.int64)
        r = _struct(H, L, C, ci, s, theta, cw, hv, hi_, lv, li_)
        st[k] = r[0]
        i0o[k] = r[1]
        h0o[k] = r[2]
        if r[0] != ST_OK:
            continue
        nh = r[3]
        iE = r[4]
        E = r[5]
        iU0, U0, iD, D, iC, Cl, su, sl = _lines_j(hv, hi_, lv, li_, nh, iE, E)
        iDo[k] = iD
        iEo[k] = iE
        Do[k] = D
        lnD = math.log(D)
        lvt[k] = math.exp(lnD + su * (s + 1 - iD))
        hE = H[ci, iE]
        hEo[k] = hE
        b = 0
        mk = MK[ci]
        for j in range(iD + 1, s + 1):                # 관통 = 그날 선 값 + 그날 1호가(날짜별 표)
            hj = H[ci, j]
            u = math.exp(lnD + su * (j - iD))
            if hj >= u + tick_dated(u, j, mk, reform):
                b |= 1
            if hj >= D + tick_dated(D, j, mk, reform):
                b |= 2
        for j in range(iE + 1, s + 1):
            if H[ci, j] >= hE + tick_dated(hE, j, mk, reform):
                b |= 8
        pre[k] = b


_scan = _jit(_scan_py, parallel=True)


# ---------------------------------------------------------------------------
# 2. 서술자(첫 관통 행만) — 위생 · 정규화 모양 좌표 · 맥락 재료
# ---------------------------------------------------------------------------

def _desc_py(H, L, C, Vv, cis, ss, bases, theta, cw, gap_min, span_min, itol, b_age, dskip, dwin, rtol, hst, oi, of):
    """첫 관통 행마다 위생 상태 ``hst``(:data:`HYG_STATUS`, 종가 여유·맥락은 호출부)·정수 ``oi``(i0, iA, iB, iC, iD, iE, iS)·실수
    ``of``(:data:`F_COLS` 의 level … h0in). 규칙은 모듈 설명 3·7·9·10·11. 실패해도 계산한 값은 채운다. 병렬."""
    cap = cw + 3
    for k in prange(len(cis)):
        ci = cis[k]
        s = ss[k]
        d = s + 1
        hv = np.empty(cap)
        hi_ = np.empty(cap, dtype=np.int64)
        lv = np.empty(cap)
        li_ = np.empty(cap, dtype=np.int64)
        r = _struct(H, L, C, ci, s, theta, cw, hv, hi_, lv, li_)
        if r[0] != ST_OK:                             # 호출부가 ST_OK 행만 넘긴다 — 방어
            hst[k] = H_SPACING
            continue
        i0 = r[1]
        h0 = r[2]
        nh = r[3]
        iE = r[4]
        E = r[5]
        iU0, U0, iD, D, iC, Cl, su, sl = _lines_j(hv, hi_, lv, li_, nh, iE, E)
        h0in = nh == 2
        if h0in:
            iA = iC                                   # 패턴 첫 저점 = C (A 없음)
            A = np.nan
            iS = i0
        else:
            iA = li_[nh - 3]
            A = lv[nh - 3]
            iS = iA
        oi[k, 0] = i0
        oi[k, 1] = iA
        oi[k, 2] = iU0
        oi[k, 3] = iC
        oi[k, 4] = iD
        oi[k, 5] = iE
        oi[k, 6] = iS
        lnD = math.log(D)
        lnE = math.log(E)
        b = bases[k]
        ud = math.exp(lnD + su * (d - iD))
        hE = H[ci, iE]
        if b == 0:
            level = ud
        elif b == 1:
            level = D
        elif b == 2:
            level = h0
        else:
            level = hE
        of[k, 0] = level
        of[k, 2] = h0
        of[k, 3] = A
        of[k, 4] = U0
        of[k, 5] = Cl
        of[k, 6] = D
        of[k, 7] = E
        of[k, 8] = su
        of[k, 9] = sl
        span = s - iS
        mx = -np.inf
        mn = np.inf
        for j in range(iS, s + 1):
            if H[ci, j] > mx:
                mx = H[ci, j]
            if L[ci, j] < mn:
                mn = L[ci, j]
        hgt = math.log(mx / mn) if (mn > 0 and mx > mn) else np.nan
        of[k, 14] = span
        of[k, 15] = hgt
        if hgt == hgt and hgt > 0:
            of[k, 10] = su * span / hgt
            of[k, 11] = sl * span / hgt
        w0 = (lnD + su * (iS - iD)) - (lnE + sl * (iS - iE))
        w1 = (lnD + su * (s - iD)) - (lnE + sl * (s - iE))
        if w0 > 0:
            of[k, 12] = w1 / w0
        if not h0in:
            of[k, 13] = A / math.exp(lnE + sl * (iA - iE)) - 1.0
        of[k, 16] = ud / h0
        mn0 = np.inf
        for j in range(i0, s + 1):
            if L[ci, j] < mn0:
                mn0 = L[ci, j]
        of[k, 17] = (h0 - mn0) / h0
        b0 = i0 - dskip - dwin
        if b0 >= 0 and s > iD:
            vs = 0.0
            for j in range(iD + 1, s + 1):
                vs += Vv[ci, j]
            vb = 0.0
            for j in range(b0, b0 + dwin):
                vb += Vv[ci, j]
            if vb > 0 and vs == vs:
                of[k, 18] = (vs / (s - iD)) / (vb / dwin)
        for q in range(3):
            w = 20 * (q + 1)
            if i0 - w >= 0:
                m = np.inf
                for j in range(i0 - w, i0 + 1):
                    if L[ci, j] < m:
                        m = L[ci, j]
                if m > 0:
                    of[k, 19 + q] = h0 / m - 1.0
        of[k, 22] = iA - i0
        of[k, 23] = math.exp(lnE + sl * (d - iE))
        of[k, 24] = ud
        of[k, 25] = hE
        # 수평 저항 수(모듈 설명 11): 창 시작 ~ i0 전 지그재그 확정 고점 + H0 + H0 뒤 확정 고점 중 ≥ H0 × (1 − rtol)
        nres = 0
        for q in range(nh):
            if hv[q] >= h0 * (1.0 - rtol):
                nres += 1
        s0 = s - cw + 1
        if i0 - 1 > s0:
            pv = np.empty(cap)
            pi_ = np.empty(cap, dtype=np.int64)
            qv = np.empty(cap)
            qi_ = np.empty(cap, dtype=np.int64)
            ph = V._zigzag(H, L, ci, s0, i0 - 1, theta, pv, pi_, qv, qi_)
            for q in range(ph[0]):
                if pv[q] >= h0 * (1.0 - rtol):
                    nres += 1
        of[k, 26] = nres
        of[k, 27] = 1.0 if h0in else 0.0
        # --- 위생(모듈 설명 7) ---
        if h0in:
            sp_bad = iC - i0 < gap_min or iD - iC < gap_min or iE - iD < gap_min
        else:
            sp_bad = iU0 - iA < gap_min or iC - iU0 < gap_min or iD - iC < gap_min or iE - iD < gap_min
        if sp_bad:
            hst[k] = H_SPACING
            continue
        if span < span_min:
            hst[k] = H_SPAN
            continue
        bad = False
        for j in range(iS, iE + 1):
            if C[ci, j] > math.exp(lnD + su * (j - iD)) * (1.0 + itol):
                bad = True
                break
        if bad:
            hst[k] = H_INTEG_UP
            continue
        for j in range(iS, s + 1):
            if C[ci, j] < math.exp(lnE + sl * (j - iE)) * (1.0 - itol):
                bad = True
                break
        if bad:
            hst[k] = H_INTEG_LO
            continue
        if b == 3 and d - iE > b_age:
            hst[k] = H_BOTTOM_AGE
            continue
        hst[k] = H_OK


_desc = _jit(_desc_py, parallel=True)


# ---------------------------------------------------------------------------
# 3. 표 만들기
# ---------------------------------------------------------------------------

_PRE_BIT = (1, 2, 0, 8)


def base_level(base: int, lvt, Dv, h0, hE) -> np.ndarray:
    return (lvt, Dv, h0, hE)[base]


def first_crossings(H, TR, cis, ss, st, i0, iD, iE, lvt, Dv, h0, hE, pre, base: int, mkt=None, reform: int = 0) -> np.ndarray:
    """기준선 ``base`` 의 첫 관통 행 번호(입력 = (종목, s) 오름차순). 관통 = 구조 정의 ∧ d 거래 가능 ∧ 고가(d) ≥ 기준선(d) + 1호가
    ∧ 그 패턴이 s 까지 이미 관통하지 않음(``pre`` 비트 — 정의 전 관통 포함). 식별 (종목, i0, 행(D))마다 첫 행 하나(H0 = (종목, i0)).
    바닥 반전은 기준선이 E 봉 고가라 E 가 내려가면 새 기준선이다 — 식별 (종목, i0, 행(D), 행(E)) 마다 첫 관통(같은 E 봉 고가를
    두 번째로 넘는 것은 막고, 새 최저가가 나오면 다시 대기 — ``WEDGE_D_TEST.md`` §3 "D 가 옮겨 간다"). 같은 패턴 체결은 하나
    (포트폴리오가 (종목, i0, 행(D)) 로 건다)."""
    lvl = base_level(base, lvt, Dv, h0, hE)
    d = ss + 1
    ok = st == ST_OK
    if _PRE_BIT[base]:
        ok &= (pre & _PRE_BIT[base]) == 0
    with np.errstate(invalid="ignore"):
        order = lvl + (tick_size_vec(lvl) if mkt is None else tick_dated_vec(lvl, d, mkt[cis], reform))
        hd = H[cis, d]
        cross = ok & TR[cis, d] & np.isfinite(hd) & (hd >= order)
    idx = np.flatnonzero(cross)
    if not len(idx):
        return idx
    z = np.zeros(len(idx), np.int64)
    key = pd.DataFrame({"c": cis[idx], "i0": i0[idx], "iD": iD[idx] if base != B_H0 else z,
                        "iE": iE[idx] if base == B_BOTTOM else z})
    return idx[~key.duplicated(keep="first").to_numpy()]


def scan_rows(valid: np.ndarray, s_lo: int, s_hi: int) -> "tuple[np.ndarray, np.ndarray]":
    """``valid`` (종목×날짜 bool — 고가·저가·종가 유한) 에서 s ∈ [s_lo, s_hi] 행 → (종목, s), (종목, s) 오름차순."""
    m = np.zeros_like(valid)
    m[:, s_lo:s_hi + 1] = valid[:, s_lo:s_hi + 1]
    ci, s = np.nonzero(m)
    return ci.astype(np.int64), s.astype(np.int64)


def build_tables(H, L, C, Vv, Op, TR, ctx: dict, *, d_lo: int, d_hi: int, thetas=THETAS, ceil_wins=CEIL_WINS,
                 log=None, hygiene: "dict | None" = None) -> dict:
    """사건 표 전부(:data:`N_TABLES` 개를 한 배열에 이어 붙임, 표 안 정렬 = (d, ADV20 내림, 종목)).

    ``ctx`` (전부 종목×날짜): ``base``(bool — 모듈 설명 8 의 기본 맥락, s 행), ``above50``·``tt``(bool), ``rs``·``breadth``·``adv20``
    (실수), ``strat``(정수 — 모멘텀 정합 층, −1 = 없음), ``market``(종목별 0/1), ``month``(날짜별 정수). 주문일 d ∈ [d_lo, d_hi].
    반환: ``F``(:data:`F_COLS`), ``I``(:data:`I_COLS`), ``off``(표 번호 → 시작, 길이 N_TABLES+1), ``funnel``(표별 첫 관통·위생 건수)."""
    n_d = H.shape[1]
    valid = np.isfinite(H) & np.isfinite(L) & np.isfinite(C)
    s_lo = max(0, d_lo - 1 - max(CEIL_WINS) - 5)          # 사건 구간 앞에서 시작한 패턴의 관통도 잡는다(소진 판정)
    s_hi = min(d_hi - 1, n_d - 2)
    cis, ss = scan_rows(valid, s_lo, s_hi)
    MK = np.asarray(ctx["market"], np.int64)
    reform = int(ctx.get("tick_reform_row", 0))
    Fs, Is, funnel = [], [], {}
    off = [0]
    hy = {**HYGIENE, **(hygiene or {})}             # FWM M3: 변형별 위생(기본 = HYGIENE)
    for ti, th in enumerate(THETAS):
        for wi, cw in enumerate(CEIL_WINS):
            if th not in thetas or cw not in ceil_wins:
                for _ in BASES:
                    off.append(off[-1])
                continue
            m = len(cis)
            st = np.zeros(m, np.int64)
            i0 = np.full(m, -1, np.int64)
            iD = np.full(m, -1, np.int64)
            iE = np.full(m, -1, np.int64)
            pre = np.zeros(m, np.int64)
            lvt, Dv, h0, hE = (np.full(m, np.nan) for _ in range(4))
            _scan(H, L, C, MK, reform, cis, ss, float(th), int(cw), st, i0, iD, iE, lvt, Dv, h0, hE, pre)
            for bi, bname in enumerate(BASES):
                tid = table_id(ti, wi, bi)
                fc = first_crossings(H, TR, cis, ss, st, i0, iD, iE, lvt, Dv, h0, hE, pre, bi, MK, reform)
                dd = ss[fc] + 1
                fc = fc[(dd >= d_lo) & (dd <= d_hi)]
                k = len(fc)
                c_, s_ = cis[fc], ss[fc]
                hst = np.zeros(k, np.int64)
                oi = np.full((k, 7), -1, np.int64)
                of = np.full((k, len(F_COLS)), np.nan)
                if k:
                    _desc(H, L, C, Vv, c_, s_, np.full(k, bi, np.int64), float(th), int(cw), int(hy["gap_min"]),
                          int(hy["span_min"]), float(hy["integ_tol"]), int(hy["bottom_age"]), int(hy["dry_skip"]),
                          int(hy["dry_win"]), float(hy["res_tol"]), hst, oi, of)
                lvl = of[:, FI["level"]]
                d_all = s_ + 1
                with np.errstate(invalid="ignore"):
                    of[:, FI["order"]] = lvl + tick_dated_vec(lvl, d_all, MK[c_], reform)
                    room = C[c_, s_] <= of[:, FI["order"]] * hy["close_room"]
                hst = np.where((hst == H_OK) & ~room, H_CLOSE, hst)
                ctx_ok = ctx["base"][c_, s_] if k else np.zeros(0, bool)
                hst = np.where((hst == H_OK) & ~ctx_ok, H_CONTEXT, hst)
                h0in_ = of[:, FI["h0in"]] == 1             # 2단계 MINOR 5: 간격 탈락 중 D→E 만 짧은 것(소진 규칙 몫)
                i0_, iA_, iU_, iC_, iD_, iE_ = (oi[:, j] for j in range(6))
                fixed_bad = np.where(h0in_, ((iC_ - i0_) < hy["gap_min"]) | ((iD_ - iC_) < hy["gap_min"]),
                                     ((iU_ - iA_) < hy["gap_min"]) | ((iC_ - iU_) < hy["gap_min"]) | ((iD_ - iC_) < hy["gap_min"]))
                de_only = (hst == H_SPACING) & ((iE_ - iD_) < hy["gap_min"]) & ~fixed_bad
                funnel[(th, cw, bname)] = {"first_cross": int(k), "spacing_DE_only": int(de_only.sum()),
                                           **{HYG_STATUS[q]: int((hst == q).sum()) for q in range(len(HYG_STATUS))}}
                keep = hst == H_OK
                c_, s_, oi, of = c_[keep], s_[keep], oi[keep], of[keep]
                d_ = s_ + 1
                E_ = of[:, FI["E"]]
                lo_d = of[:, FI["lower_d"]]
                of[:, FI["stopE_raw"]] = E_ - tick_dated_vec(E_, d_, MK[c_], reform)
                with np.errstate(invalid="ignore"):
                    of[:, FI["stopL_raw"]] = lo_d - tick_dated_vec(np.where(lo_d > 0, lo_d, np.nan), d_, MK[c_], reform)
                for nm, arr, row in (("open_d", Op, d_), ("high_d", H, d_), ("low_d", L, d_), ("close_d", C, d_),
                                     ("close_s", C, s_)):
                    of[:, FI[nm]] = arr[c_, row]
                for nm in ("adv20", "rs", "breadth"):
                    of[:, FI[nm]] = ctx[nm][c_, s_]
                of[:, FI["above50"]] = ctx["above50"][c_, s_]
                of[:, FI["tt_s"]] = ctx["tt"][c_, s_]
                of[:, FI["tt_h0"]] = ctx["tt"][c_, oi[:, 0]]
                of[:, FI["market"]] = ctx["market"][c_]
                of[:, FI["strat"]] = ctx["strat"][c_, s_]
                of[:, FI["month"]] = ctx["month"][d_]
                if "atr" in ctx:                              # 정정 5 손절 여유(ATR20, d−1)
                    of[:, FI["atr_s"]] = ctx["atr"][c_, s_]
                I_ = np.column_stack([d_, c_, oi, np.full(len(c_), tid)]).astype(np.int64)
                o = np.lexsort((c_, -np.nan_to_num(of[:, FI["adv20"]], nan=-1.0), d_))
                Fs.append(of[o])
                Is.append(I_[o])
                off.append(off[-1] + len(o))
            if log:
                log(f"사건 표 θ {th:.0%} · 창 {cw}: " + " · ".join(
                    f"{b} {funnel[(th, cw, b)]['ok']:,}" for b in BASES))
    F = np.vstack(Fs) if Fs else np.zeros((0, len(F_COLS)))
    I_ = np.vstack(Is) if Is else np.zeros((0, len(I_COLS)), np.int64)
    return {"F": np.ascontiguousarray(F), "I": np.ascontiguousarray(I_), "off": np.asarray(off, np.int64), "funnel": funnel,
            "n_scan_rows": int(len(cis))}


def shuffle_shape(F: np.ndarray, off: np.ndarray, rep: int, seed: int = 20261012) -> np.ndarray:
    """모양 섞은 널(정정 1 C1-2b): 표마다 같은 달(주문일 d 의 달) 안에서 사건끼리 :data:`SHAPE_COLS` 묶음을 **행 단위로 함께**
    무작위 재배정한 ``F`` 사본. 진입일·가격·손절·맥락(RS·breadth·단계·실적·정정 2 급등·쉼)·청산 재료는 그대로."""
    G = F.copy()
    cols = np.array([FI[c] for c in SHAPE_COLS])
    rng = np.random.default_rng([seed, rep])
    mo = F[:, FI["month"]].astype(np.int64)
    for t in range(len(off) - 1):
        a, b = int(off[t]), int(off[t + 1])
        if b - a < 2:
            continue
        m = mo[a:b]
        order = np.argsort(m, kind="stable")
        bounds = np.flatnonzero(np.diff(m[order])) + 1
        src = np.arange(a, b)[order]
        for grp in np.split(src, bounds):
            if len(grp) > 1:
                G[np.ix_(grp, cols)] = F[np.ix_(rng.permutation(grp), cols)]
    return G


def table_slice(off: np.ndarray, tid: int) -> "tuple[int, int]":
    return int(off[tid]), int(off[tid + 1])


def pattern_points(H: np.ndarray, L: np.ndarray, C: np.ndarray, ci: int, s: int, theta: float, cw: int) -> dict:
    """차트용 — H0·반전점(A·B·C·D 이름 — ``h0in`` 1 은 H0 가 상단선 첫 점)·E·상단선/하단선 끝점(d 까지). 성과 정보 없음."""
    cap = cw + 3
    hv, hi_ = np.empty(cap), np.empty(cap, dtype=np.int64)
    lv, li_ = np.empty(cap), np.empty(cap, dtype=np.int64)
    r = _struct_py(H, L, C, ci, s, theta, cw, hv, hi_, lv, li_)
    out = {"status": int(r[0]), "i0": int(r[1]), "h0": float(r[2])}
    if r[0] != ST_OK:
        return out
    nh, iE, E = int(r[3]), int(r[4]), float(r[5])
    iU0, U0, iD, D, iC, Cl, su, sl = _lines(hv, hi_, lv, li_, nh, iE, E)
    pts = [{"row": int(r[1]), "price": float(r[2]), "kind": "H0"}]
    for q in range(nh - 1):
        lk = {nh - 3: "A", nh - 2: "C"}.get(q, "l")
        hk = {nh - 3: "B", nh - 2: "D"}.get(q, "h")
        pts.append({"row": int(li_[q]), "price": float(lv[q]), "kind": lk})
        pts.append({"row": int(hi_[q + 1]), "price": float(hv[q + 1]), "kind": hk})
    pts.append({"row": iE, "price": E, "kind": "E"})
    iS = int(r[1]) if nh == 2 else int(li_[nh - 3])
    out.update({"points": pts, "h0in": bool(nh == 2),
                "upper": {"from": {"row": int(iU0), "price": float(U0)},
                          "to": {"row": s + 1, "price": math.exp(math.log(D) + su * (s + 1 - iD))}},
                "lower": {"from": {"row": iS, "price": math.exp(math.log(E) + sl * (iS - iE))},
                          "to": {"row": s + 1, "price": math.exp(math.log(E) + sl * (s + 1 - iE))}}})
    return out
