"""미너비니 청산 v2 가속 커널 — 본전 이동·부분 익절 비율·강세 매도·피벗 허용폭·연속 이탈일 (설계 ``MINERVINI_EXITS_V2.md``).

설계 정본: ``research/logs/sepa_explore/MINERVINI_EXITS_V2.md`` (2e47f59, 결과 보기 전 고정). 이 파일은 **계산 기계**만 둔다
(어느 칸이 좋은지는 러너 ``research/experiments/exits_v2_run.py`` 가 리포트한다).

:mod:`research.signals.ga_fast` 의 ``_sim_one`` (실행 중인 GA2 가 쓰는 커널)은 **건드리지 않고** 여기에 하루씩 도는 새 커널을 둔다.
v2 유전자를 전부 끔(``be_r = ∞``·``take_frac = ½``·강세 매도 끔·``fb_tol = 0``·``fb_days = 1``)으로 두면 ``ga_fast._sim_one`` 과
트레이드 단위로 **비트 일치**한다(테스트 — 실데이터 확인은 러너 로그). 같은 정의가 :func:`research.signals.minervini_exits.simulate_minervini`
(v2 키워드)와 :class:`swing_it.live.strategy.PositionState` 에 있다 — 3중 패리티는 ``tests/test_live_genes.py``.

**하루 순서(거래 가능일 ``j`` — 정지일은 아무것도 안 하고 날짜만 센다, 진입일 = 오프셋 0):**

1. 전날 종가 결정 실행(시가): 돌파 실패·추적 → 남은 전량 시가 청산(끝). 강세 매도 → 남은 수량의 절반을 시가에 팔고 계속.
2. 손절(장중): 손절선 = 기술적 손절가, 본전이 살아 있으면 본전선(익절 뒤 본전 = 진입가, ``be_r`` 본전 = 진입가 × (1 + 0.5%)
   — 둘 다면 높은 쪽 = ``be_r`` 본전). 저가 ≤ 손절선 또는 시가 ≤ 손절선 → 남은 전량 ``min(시가, 손절선)``. 사유 = 본전선이면
   ``breakeven``, 아니면 ``stop``. (설계 우선순위 "손절 > 본전 손절" — 본전이 살아 있으면 손절선이 본전선으로 **바뀐다**: 본전선 >
   진입가 > 기술적 손절가라 같은 날 둘 다 닿으면 가격은 본전선을 먼저 지난다.)
3. 부분 익절(장중, 아직 안 했으면): 고가 ≥ 목표가(진입가 × (1 + take_r·d)) 또는 시가 ≥ 목표가 → 남은 수량 × ``take_frac`` 를 목표가
   (갭 상승이면 시가)에. 익절 뒤 본전(진입가)은 **다음 거래 가능일부터**; 예외 — 갭 상승 익절일에 저가 ≤ 진입가면 그날 진입가에
   나머지 전량(``breakeven``, 원전 규칙 7 그대로).
4. ``be_r`` 발동: 고가(또는 시가) ≥ 진입가 × (1 + be_r·d) 인 첫날 → **다음 거래 가능일부터** 본전선 = 진입가 × 1.005
   (발동일 당일엔 손절선이 그대로다 — 갭 예외 없음).
5. 종가: 시간 손절(켜져 있으면 — v2 는 끔) → 최대 보유(``max_hold`` 일째 이후 첫 거래 가능일 종가에 남은 전부) → 다음날 시가 결정
   (1 ~ max_hold−1 일째 종가에서만, 하나만 — 우선순위 **강세 매도 > 돌파 실패 > 추적**):

   - 강세 매도(한 트레이드에 **한 번**): 종가 ≥ 직전 행 종가 × (1 + x) 또는 종가 ≥ MA50 × (1 + y). 직전 행 종가·MA50 이 NaN 이면
     그 조건은 거짓.
   - 돌파 실패(익절 전에만): 종가 < 피벗 × (1 − fb_tol) 인 거래 가능일이 ``fb_days`` 번 **연속**(정지일은 세지도 끊지도 않는다,
     종가 ≥ 기준이면 0 으로) → 남은 전량.
   - 추적: 종가 < 추적 MA → 남은 전량. 익절 뒤에만(익절이 없으면 — ``take_r = ∞`` — 전량 추적, 원전 규칙 13).

   같은 종가에 강세 매도가 이기면 돌파 실패·추적은 그날 결정되지 않고 다음 종가에 다시 본다(연속 이탈 횟수는 계속 센다).
6. 다음 거래 가능일이 지평 밖이면 결정은 체결되지 않고 종료 청산(최대 보유 종가 / 데이터 끝 ``data_end`` / 폐지 ``delisted``).

**수량:** 부분 익절·강세 매도는 둘 다 **남은 수량** 대비 비율(라이브 ``PositionState._partial_qty`` 와 같다). 트레이드 수익 =
Σ (판 비율 × (체결가/진입가 − 1)) 를 시간 순으로 더한 것 — 옛 ``0.5·(h1 + h2)`` 와 비트 단위로 같다(×0.5 는 정확한 연산).
``R = ret / d``.

numba 는 선택(``uv run --with numba``). 없으면 같은 소스가 순수 파이썬으로 돈다. ``이름_py`` = 순수 파이썬, ``이름`` = JIT 판.
"""

from __future__ import annotations

import numpy as np

from research.signals import ga_fast as gf
from research.signals.ga_fast import (
    R_BREAKEVEN,
    R_DATA_END,
    R_DELISTED,
    R_FAILED,
    R_STOP,
    R_TIME,
    R_TIME_STOP,
    R_TRAIL,
    ST_BELOW_LOW,
    ST_LIMIT_UP,
    ST_NO_CONTRACTION,
    ST_NO_NEXT,
    ST_NO_OPEN,
    ST_OK,
)

HAVE_NUMBA = gf.HAVE_NUMBA
prange = gf.prange

#: ``be_r`` 본전선 = 진입가 × (1 + BE_COST) — 설계 "진입가(+비용 0.5%)". 라이브 :data:`swing_it.live.config.BE_COST` 와 같은 값(테스트).
BE_COST = 0.005

# 플래그 비트(트레이드 서술용)
F_PARTIAL, F_STRENGTH, F_BE_TRIG = 1, 2, 4

#: 강세 매도 단계(설계 표 순서) → (하루 상승률 x, MA50 이격 y). NaN = 그 조건 없음.
STRENGTH_LEVELS = ((np.nan, np.nan), (0.15, np.nan), (np.nan, 0.30), (0.15, 0.30))
STRENGTH_NAMES = ("끔", "x15%", "y30%", "x15%|y30%")


def _jit(fn, parallel: bool = False):
    return gf._jit(fn, parallel=parallel)


def _first_tr_py(TR, ci, e, lo, hi):
    """오프셋 [lo, hi] 에서 첫 거래 가능 오프셋, 없으면 −1 (``ga_fast._first_tr`` 와 같은 식, 이 모듈 사본)."""
    for j in range(lo, hi + 1):
        if TR[ci, e + j]:
            return j
    return -1


def _sim_one_v2_py(Op, H, L, C, MA, MA50, TR, last_present, ci, d, c_low, stop_floor, stop_cap, stop_mult, take_r,
                   take_frac, be_r, ss_x, ss_y, fb, fb_tol, fb_days, pivot, ts_days, max_hold, has_trail, lu_ratio,
                   lu_skip):
    """한 사건 → ``(status, x_abs, entry, dd, reason, e1_abs, e1_frac, e1_px, e2_abs, e2_frac, e2_px, fin_frac, fin_px, flags)``.

    ``e1``·``e2`` = 마지막 청산 전의 부분 매도(부분 익절·강세 매도, 시간 순, 없으면 abs −1). 비율은 **최초 수량** 대비(남은 수량 대비
    비율을 곱해 둔 값). ``ts_days ≤ 0`` = 시간 손절 끔, ``take_r = inf`` = 익절 없음, ``be_r = inf`` = 본전 이동 끔,
    ``ss_x``·``ss_y`` NaN = 그 강세 조건 없음, ``fb`` False 이거나 ``pivot`` NaN = 돌파 실패 없음. 규칙은 모듈 설명 1~6.
    """
    nan = np.nan
    n = Op.shape[1]
    e = d + 1
    if e >= n:
        return ST_NO_NEXT, -1, nan, nan, -1, -1, 0.0, nan, -1, 0.0, nan, 0.0, nan, 0
    if not TR[ci, e]:
        return ST_NO_OPEN, -1, nan, nan, -1, -1, 0.0, nan, -1, 0.0, nan, 0.0, nan, 0
    entry = Op[ci, e]
    prev = C[ci, d]
    if lu_skip and np.isfinite(prev) and prev > 0 and entry >= prev * lu_ratio:
        return ST_LIMIT_UP, -1, nan, nan, -1, -1, 0.0, nan, -1, 0.0, nan, 0.0, nan, 0
    if not np.isfinite(c_low):
        return ST_NO_CONTRACTION, -1, nan, nan, -1, -1, 0.0, nan, -1, 0.0, nan, 0.0, nan, 0
    if entry <= c_low:
        return ST_BELOW_LOW, -1, nan, nan, -1, -1, 0.0, nan, -1, 0.0, nan, 0.0, nan, 0

    dd = 1.0 - c_low * stop_mult / entry
    dd = min(max(dd, stop_floor), stop_cap)
    stop_px = entry * (1.0 - dd)
    tgt = entry * (1.0 + take_r * dd)
    partial_on = np.isfinite(take_r)
    be_on = np.isfinite(be_r)
    be_lvl = entry * (1.0 + be_r * dd) if be_on else nan
    be_px = entry * (1.0 + BE_COST)
    ssx_on = np.isfinite(ss_x)
    ssy_on = np.isfinite(ss_y)
    fb_on = fb and np.isfinite(pivot)
    fb_px = pivot * (1.0 - fb_tol) if fb_on else nan

    Ln = n - e
    k_last = max_hold - 1
    j_time = _first_tr_v(TR, ci, e, k_last, Ln - 1) if k_last <= Ln - 1 else -1
    horizon = j_time if j_time >= 0 else Ln - 1
    sig_n = min(k_last, horizon + 1)

    f = 1.0                       # 남은 비율(최초 수량 대비)
    e1_abs, e1_frac, e1_px = -1, 0.0, nan
    e2_abs, e2_frac, e2_px = -1, 0.0, nan
    n_ev = 0
    flags = 0
    pend = 0                      # 0 없음 · 1 강세 매도(절반) · 2 돌파 실패(전량) · 3 추적(전량)
    partial_done = False
    post_be_from = -1             # 익절 뒤 본전(진입가)이 걸리는 첫 오프셋
    be_trig = -1                  # be_r 발동 오프셋(본전선은 그 다음 거래 가능일부터)
    ss_done = False
    fb_cnt = 0
    ts_checked = False
    last_tr = 0
    for j in range(horizon + 1):
        if not TR[ci, e + j]:
            continue
        last_tr = j
        o = Op[ci, e + j]
        h = H[ci, e + j]
        lw = L[ci, e + j]
        c = C[ci, e + j]
        # 1. 전날 종가 결정 → 시가
        if pend == 2 or pend == 3:
            return (ST_OK, e + j, entry, dd, R_FAILED if pend == 2 else R_TRAIL, e1_abs, e1_frac, e1_px, e2_abs, e2_frac,
                    e2_px, f, o, flags)
        if pend == 1:
            q = f * 0.5
            f = f - q
            if n_ev == 0:
                e1_abs, e1_frac, e1_px = e + j, q, o
            else:
                e2_abs, e2_frac, e2_px = e + j, q, o
            n_ev += 1
            flags |= F_STRENGTH
            pend = 0
        # 2. 손절선(본전이 살아 있으면 본전선)
        lvl = stop_px
        is_be = False
        if partial_done and j >= post_be_from:
            lvl = entry
            is_be = True
        if be_trig >= 0 and j > be_trig:
            lvl = be_px
            is_be = True
        if lw <= lvl or o <= lvl:
            return (ST_OK, e + j, entry, dd, R_BREAKEVEN if is_be else R_STOP, e1_abs, e1_frac, e1_px, e2_abs, e2_frac,
                    e2_px, f, min(o, lvl), flags)
        # 3. 부분 익절
        if partial_on and not partial_done and (h >= tgt or o >= tgt):
            gap = o >= tgt
            px = o if gap else tgt
            q = f * take_frac
            f = f - q
            if n_ev == 0:
                e1_abs, e1_frac, e1_px = e + j, q, px
            else:
                e2_abs, e2_frac, e2_px = e + j, q, px
            n_ev += 1
            flags |= F_PARTIAL
            partial_done = True
            post_be_from = j + 1
            if gap and lw <= entry:
                return (ST_OK, e + j, entry, dd, R_BREAKEVEN, e1_abs, e1_frac, e1_px, e2_abs, e2_frac, e2_px, f, entry,
                        flags)
        # 4. be_r 발동(다음 거래 가능일부터)
        if be_on and be_trig < 0 and (h >= be_lvl or o >= be_lvl):
            be_trig = j
            flags |= F_BE_TRIG
        # 5. 종가
        if ts_days > 0 and not partial_done and not ts_checked and j >= ts_days - 1:
            ts_checked = True
            if c <= entry:
                return (ST_OK, e + j, entry, dd, R_TIME_STOP, e1_abs, e1_frac, e1_px, e2_abs, e2_frac, e2_px, f, c,
                        flags)
        if j == j_time:
            return ST_OK, e + j, entry, dd, R_TIME, e1_abs, e1_frac, e1_px, e2_abs, e2_frac, e2_px, f, c, flags
        if j < sig_n:
            ss = False
            if not ss_done and (ssx_on or ssy_on):
                pc_ = C[ci, e + j - 1]
                if ssx_on and np.isfinite(pc_) and c >= pc_ * (1.0 + ss_x):
                    ss = True
                m50 = MA50[ci, e + j]
                if ssy_on and np.isfinite(m50) and c >= m50 * (1.0 + ss_y):
                    ss = True
            fbk = False
            if fb_on and not partial_done:
                if c < fb_px:
                    fb_cnt += 1
                else:
                    fb_cnt = 0
                fbk = fb_cnt >= fb_days
            tr = False
            if has_trail and (partial_done or not partial_on):
                ma = MA[ci, e + j]
                if np.isfinite(ma) and c < ma:
                    tr = True
            if ss:
                pend = 1
                ss_done = True
            elif fbk:
                pend = 2
            elif tr:
                pend = 3
    # 6. 지평 끝(최대 보유 행이 없음 = 데이터 끝·폐지)
    reason = R_DATA_END if last_present[ci] else R_DELISTED
    return (ST_OK, e + last_tr, entry, dd, reason, e1_abs, e1_frac, e1_px, e2_abs, e2_frac, e2_px, f,
            C[ci, e + last_tr], flags)


def _sim_pairs_v2_py(Op, H, L, C, MA, MA50, TR, last_present, pc, pd_, pclow, ppiv, stop_floor, stop_cap, stop_mult,
                     take_r, take_frac, be_r, ss_x, ss_y, fb, fb_tol, fb_days, ts_days, max_hold, has_trail, lu_ratio,
                     lu_skip):
    """쌍 배열 전부를 시뮬(병렬).

    Returns: ``status(int8), x_abs(int32), ret(f64), dd(f64), reason(int8), ev_abs(m×2 int32), ev_frac(m×2), ev_ret(m×2),
    fin_frac, fin_ret, flags(int8)``. ``ev_ret``·``fin_ret`` = 그 매도분의 (체결가/진입가 − 1). ``ret`` = Σ 비율 × 수익(시간 순).
    """
    m = pc.shape[0]
    status = np.empty(m, np.int8)
    x_abs = np.full(m, -1, np.int32)
    ret = np.full(m, np.nan)
    ddo = np.full(m, np.nan)
    reason = np.full(m, -1, np.int8)
    ev_abs = np.full((m, 2), -1, np.int32)
    ev_frac = np.zeros((m, 2))
    ev_ret = np.full((m, 2), np.nan)
    fin_frac = np.zeros(m)
    fin_ret = np.full(m, np.nan)
    flags = np.zeros(m, np.int8)
    for i in prange(m):
        (st, x, ent, dd, rs, a1, f1, p1, a2, f2, p2, ff, pf, fl) = _sim_one_v2(
            Op, H, L, C, MA, MA50, TR, last_present, pc[i], pd_[i], pclow[i], stop_floor, stop_cap, stop_mult, take_r,
            take_frac, be_r, ss_x, ss_y, fb, fb_tol, fb_days, ppiv[i], ts_days, max_hold, has_trail, lu_ratio, lu_skip)
        status[i] = st
        if st == ST_OK:
            acc = 0.0
            if a1 >= 0:
                r1 = p1 / ent - 1.0
                acc += f1 * r1
                ev_abs[i, 0] = a1
                ev_frac[i, 0] = f1
                ev_ret[i, 0] = r1
            if a2 >= 0:
                r2 = p2 / ent - 1.0
                acc += f2 * r2
                ev_abs[i, 1] = a2
                ev_frac[i, 1] = f2
                ev_ret[i, 1] = r2
            rf = pf / ent - 1.0
            acc += ff * rf
            ret[i] = acc
            x_abs[i] = x
            ddo[i] = dd
            reason[i] = rs
            fin_frac[i] = ff
            fin_ret[i] = rf
            flags[i] = fl
    return status, x_abs, ret, ddo, reason, ev_abs, ev_frac, ev_ret, fin_frac, fin_ret, flags


# ---------------------------------------------------------------------------
# 자본 제약 포트폴리오(GA2 처리량 엔진 ``ga_fast._portfolio_py`` 의 v2 판 — 매도분이 여러 개, 절단 없음)
# ---------------------------------------------------------------------------

def _portfolio_v2_py(cd_pd, cd_pc, cd_fill, cd_ord, cd_rel, cd_rs, cd_x, cd_ev_abs, cd_ev_frac, cd_ev_ret, cd_fin_frac,
                     cd_fin_ret, cd_ctl, cd_cost, day_lo, day_hi, Op, C, t0, t1, slots, slot_w):
    """후보(신호 쌍, 신호일 → 종목 순 정렬 — ``day_lo[t]..day_hi[t]`` 가 신호일 ``t`` 후보) → 날짜별 포트폴리오 정합 초과.

    ``ga_fast._portfolio_py`` 와 같은 규칙(슬롯 ``slots`` × 자산 ``slot_w``, rel 오름차순(NaN 뒤) → RS 내림차순 → 종목 순, 보유 중
    종목 제외, 미체결 주문은 그날 슬롯만 비움, 날별 손익 = 남은 비율 × (종가 − 직전 표시가)/진입가, 매도일은 체결가, 대조는 보유일
    일할, 비용은 마지막 청산일, **짝 없는 포지션은 손익·비용 0**)에 매도분이 둘까지(``cd_ev_*``) 있는 것만 다르다. 절단(마크투마켓)은
    없다 — 전 구간 창(시험 1) 전용. ``cd_ctl`` = 대조 평균 수익(NaN = 짝 없음), ``cd_cost`` = 청산일 비용률.
    반환: ``(daily[t0:t1], opened 후보 번호, n_orders, n_unmatched)``.
    """
    T = t1 - t0
    daily = np.zeros(T)
    s_i = np.full(slots, -1, np.int64)
    s_lm = np.zeros(slots)
    s_fr = np.zeros(slots)
    s_cd = np.zeros(slots)
    s_m = np.zeros(slots, np.bool_)
    opened = np.empty(cd_pd.shape[0] + 1, np.int64)
    n_open = 0
    n_ord = 0
    n_unm = 0
    cand = np.empty(4096, np.int64)
    for t in range(t0, t1):
        acc = 0.0
        for s in range(slots):
            i = s_i[s]
            if i < 0:
                continue
            c = cd_pc[i]
            e = cd_pd[i] + 1
            if t < e:
                continue
            ent = Op[c, e]
            contrib = 0.0
            for k in range(2):
                if cd_ev_abs[i, k] == t:
                    contrib += cd_ev_frac[i, k] * (cd_ev_ret[i, k] - s_lm[s])
                    s_fr[s] -= cd_ev_frac[i, k]
            xe = cd_x[i]
            if xe == t:
                contrib += cd_fin_frac[i] * (cd_fin_ret[i] - s_lm[s])
            else:
                cl = C[c, t]
                if np.isfinite(cl):
                    m = cl / ent - 1.0
                    contrib += s_fr[s] * (m - s_lm[s])
                    s_lm[s] = m
            if s_m[s]:
                acc += slot_w * (contrib - s_cd[s])
                if xe == t:
                    acc -= slot_w * cd_cost[i]
            if xe == t:
                s_i[s] = -1
        daily[t - t0] = acc
        a = day_lo[t]
        b = day_hi[t]
        if b <= a:
            continue
        free = 0
        for s in range(slots):
            if s_i[s] < 0:
                free += 1
        if free == 0:
            continue
        nc = 0
        for i in range(a, b):
            if not cd_ord[i]:
                continue
            held = False
            for s in range(slots):
                if s_i[s] >= 0 and cd_pc[s_i[s]] == cd_pc[i]:
                    held = True
                    break
            if held:
                continue
            if nc < cand.shape[0]:
                cand[nc] = i
                nc += 1
        k_take = min(free, nc)
        for r in range(k_take):
            best = -1
            for q in range(nc):
                i = cand[q]
                if i < 0:
                    continue
                if best < 0:
                    best = q
                    continue
                j = cand[best]
                ri = cd_rel[i] if np.isfinite(cd_rel[i]) else np.inf
                rj = cd_rel[j] if np.isfinite(cd_rel[j]) else np.inf
                if ri < rj:
                    best = q
                elif ri == rj:
                    si = cd_rs[i] if np.isfinite(cd_rs[i]) else -np.inf
                    sj = cd_rs[j] if np.isfinite(cd_rs[j]) else -np.inf
                    if si > sj or (si == sj and cd_pc[i] < cd_pc[j]):
                        best = q
            i = cand[best]
            cand[best] = -1
            n_ord += 1
            if cd_fill[i]:
                for s in range(slots):
                    if s_i[s] < 0:
                        s_i[s] = i
                        s_lm[s] = 0.0
                        s_fr[s] = 1.0
                        s_m[s] = np.isfinite(cd_ctl[i])
                        if s_m[s]:
                            s_cd[s] = cd_ctl[i] / (cd_x[i] - cd_pd[i])
                        else:
                            s_cd[s] = 0.0
                            n_unm += 1
                        break
                opened[n_open] = i
                n_open += 1
    return daily, opened[:n_open], n_ord, n_unm


# --- 내보내기: 이름_py = 순수 파이썬, 이름 = numba(있으면) ---
_first_tr_v = _jit(_first_tr_py)
_sim_one_v2 = _jit(_sim_one_v2_py)
sim_pairs_v2 = _jit(_sim_pairs_v2_py, parallel=True)
portfolio_v2 = _jit(_portfolio_v2_py)

#: 순수 파이썬 경로 테스트가 JIT 판을 되감을 보조 함수 이름(리뷰 A-M2 — ``_py`` 커널이 부르는 전역 이름).
HELPERS = ("_first_tr_v", "_sim_one_v2")


# ---------------------------------------------------------------------------
# 파이썬 쪽 묶음 — 사건 표 → 엔진 트레이드 (패리티 테스트·러너 공용)
# ---------------------------------------------------------------------------

#: v2 청산 인자 기본값 = 현 라이브 프로토콜(본전 이동 끔·절반 익절·강세 매도 끔·피벗 허용폭 0·1일 이탈) + ``ga_fast.EXIT_PREREG``.
EXIT_V2_DEFAULT = {"stop_floor": 0.03, "stop_cap": 0.10, "stop_mult": 0.99, "take_r": 2.0, "take_frac": 0.5,
                   "be_r": np.inf, "ss_x": np.nan, "ss_y": np.nan, "fb": True, "fb_tol": 0.0, "fb_days": 1,
                   "ts_days": 0, "max_hold": 120, "has_trail": True, "lu_skip": True}


def sim_book_v2(Op, H, L, C, MA, MA50, TR, last_present, pc, pd_, pclow, params: dict, *, pivot=None, py: bool = False):
    """쌍 배열 시뮬 — ``params`` 는 :data:`EXIT_V2_DEFAULT` 모양(빠진 키는 기본값). ``py`` = 순수 파이썬 경로(테스트)."""
    from research.signals.sepa import LIMIT_UP_RATIO  # noqa: PLC0415

    p = {**EXIT_V2_DEFAULT, **params}
    f = _sim_pairs_v2_py if py else sim_pairs_v2
    piv = np.full(len(pc), np.nan) if pivot is None else np.ascontiguousarray(pivot, np.float64)
    ts = p["ts_days"]
    return f(Op, H, L, C, MA, MA50, TR, last_present, np.asarray(pc, np.int64), np.asarray(pd_, np.int64),
             np.asarray(pclow, np.float64), piv, float(p["stop_floor"]), float(p["stop_cap"]), float(p["stop_mult"]),
             float(p["take_r"]), float(p["take_frac"]), float(p["be_r"]), float(p["ss_x"]), float(p["ss_y"]),
             bool(p["fb"]), float(p["fb_tol"]), int(p["fb_days"]), int(ts) if ts else 0, int(p["max_hold"]),
             bool(p["has_trail"]), float(LIMIT_UP_RATIO), bool(p["lu_skip"]))


def engine_events_v2(book, events, params: dict, *, ma50=None, chase: float = 0.05, no_overlap: bool = True,
                     py: bool = False):
    """사건 표(``code, event_day, pivot, contraction_low``) → v2 엔진 트레이드 표(``simulate_minervini`` v2 비교용).

    ``book`` = :class:`research.signals.sepa._Book` (``book.MA`` = 추적 MA). ``ma50`` = 강세 매도 MA50 배열(종목×날짜, 없으면 NaN —
    MA50 조건이 꺼진 칸에선 안 쓴다). 정렬·겹침 제거는 ``ga_fast.engine_events`` 와 같다.
    """
    import pandas as pd  # noqa: PLC0415

    col_of = pd.Series(np.arange(len(book.codes)), index=book.codes)
    ev = events[events["code"].isin(book.codes) & events["event_day"].isin(book.days)].copy()
    ev["sim_d"] = book.days.get_indexer(ev["event_day"])
    ev["sim_c"] = col_of.reindex(ev["code"]).to_numpy()
    ev = ev.drop_duplicates(["code", "event_day"]).sort_values(["sim_d", "code"]).reset_index(drop=True)
    pc, pd_ = ev["sim_c"].to_numpy(np.int64), ev["sim_d"].to_numpy(np.int64)
    last_present = np.ascontiguousarray(book.present[:, -1])
    m50 = np.full(book.C.shape, np.nan) if ma50 is None else np.ascontiguousarray(ma50, np.float64)
    st, x, ret, dd, rs, ea, ef, er, ff, fr, fl = sim_book_v2(
        book.O, book.H, book.L, book.C, book.MA, m50, book.tradable, last_present, pc, pd_,
        ev["contraction_low"].to_numpy(float), params, pivot=ev["pivot"].to_numpy(float), py=py)
    n = len(book.days)
    ent_open = np.where(pd_ + 1 < n, book.O[pc, np.minimum(pd_ + 1, n - 1)], np.nan)
    ok = (st == ST_OK) & gf.chase_ok(ent_open, ev["pivot"].to_numpy(float), chase)
    if no_overlap:
        g = gf._greedy_py if py else gf.greedy
        sel = g(ok, pc, pd_, x.astype(np.int64), len(book.codes), 0, len(ev))
    else:
        sel = np.flatnonzero(ok)
    return pd.DataFrame({"code": ev["code"].to_numpy()[sel], "event_day": ev["event_day"].to_numpy()[sel],
                         "entry_day": book.days[pd_[sel] + 1], "exit_day": book.days[x[sel]], "ret": ret[sel],
                         "exit_reason": [gf.REASON_NAMES[k] for k in rs[sel]], "hold_days": x[sel] - pd_[sel],
                         "d": dd[sel], "partial_hit": (fl[sel] & F_PARTIAL) > 0,
                         "strength_hit": (fl[sel] & F_STRENGTH) > 0, "be_moved": (fl[sel] & F_BE_TRIG) > 0,
                         "fin_frac": ff[sel]})
