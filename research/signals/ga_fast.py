"""GA·CMA-ES 문턱 탐색용 가속 커널 — 미너비니 청산 사전계산·마스크·겹침 제거·부트스트랩·포트폴리오.

설계 정본: ``research/logs/sepa_explore/GA.md`` (bc29047) · ``GA2.md`` (421b01a → 162103a). 이 파일은 **계산 기계**만 둔다
(어떤 config 가 좋은지는 러너 ``research/experiments/ga_run.py`` 가 리포트한다).

numba 는 프로젝트 의존성이 아니다. ``uv run --with numba …`` 로 돌리면 커널이 JIT 되고, numba 가 없으면 **같은 소스**가
순수 파이썬으로 돈다(테스트가 두 경로를 다 본다 — CI 는 numba 없이 파이썬 경로만). 그래서 커널은 numba 가 받는
부분집합(스칼라·numpy 배열·루프)으로만 쓴다. 각 커널은 ``이름_py``(순수 파이썬) 와 ``이름``(numba 면 JIT, 아니면 같은 함수)
두 이름으로 내보낸다.

**청산 시뮬(:func:`sim_pairs`)** 은 :func:`research.signals.minervini_exits._simulate_one` 을 한 줄씩 옮긴 것이다
(규칙 1~11 그대로, 패리티 테스트가 트레이드 단위로 대조). GA2 의 청산 유전자 확장 두 가지만 더했다:

- ``trail`` 없음(``has_trail=False``) — 나머지 절반은 본전·최대 보유·데이터 끝으로만 나간다.
- ``take_r = ∞`` (익절 없음) ∧ ``trail`` 있음 — 원전 규칙 8 의 추적이 "나머지 절반" 이 없으므로 **전량 추적**으로 읽는다:
  1 ~ max_hold−1 일째 종가 < MA → 다음 거래 가능일 시가 전량 청산(:func:`research.signals.sepa._simulate_one` 의 MA50 청산과
  같은 식, 같은 날 손절보다 시가 청산이 먼저). 시간 손절은 추적 청산일보다 앞설 때만. 익절 없음 ∧ 추적 없음이면 손절·시간 손절·
  최대 보유만. (실행 전 결정 — ``GA2_RESULTS.md`` 구현 메모에 적는다.)
- ``fixed_stop`` (NaN = 기술적 손절) — :func:`sepa.simulate_sepa` 패리티 테스트용(고정 손절폭).

추격 금지(시가 > 피벗 × (1+chase))는 피벗이 config(피벗 창)에 따라 달라서 시뮬 밖에서 비트로 건다 — 시뮬 결과는 피벗과 무관하다
(추격 금지는 버림 여부만 바꾼다). 버림 사유의 **검사 순서**는 원본과 다르지만(추격이 마지막) 버림 여부는 같다.

**결정적 난수:** 부트스트랩·대조 추출은 정수 해시(lowbias32 변형, 곱셈 상수 < 2³¹ — int64 곱이 넘치지 않는다)로 뽑는다.
numba 와 순수 파이썬이 **비트 단위로 같은** 표본을 낸다(테스트). ``swing_it.validation.optimization._boot_lower`` 와 같은 모양
(부트스트랩 평균의 2.5% 백분위, 선형 보간)이지만 난수열이 달라 숫자는 같지 않다 — 통계적으로 가깝다(테스트).
"""

from __future__ import annotations

import numpy as np

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


# --- 코드표 ---
ST_OK, ST_NO_NEXT, ST_NO_OPEN, ST_LIMIT_UP, ST_NO_CONTRACTION, ST_BELOW_LOW = 0, 1, 2, 3, 4, 5
STATUS_NAMES = ("ok", "skip_no_next_day", "skip_no_open", "skip_limit_up", "skip_no_contraction", "skip_below_low")
R_STOP, R_BREAKEVEN, R_TRAIL, R_TIME_STOP, R_TIME, R_DELISTED, R_DATA_END, R_FAILED = 0, 1, 2, 3, 4, 5, 6, 7
REASON_NAMES = ("stop", "breakeven", "trail", "time_stop", "time", "delisted", "data_end", "failed_breakout")

M32 = 0xFFFFFFFF
SLOTS = 10            # 포트폴리오 슬롯(종목당 자산 10%)
SLOT_W = 0.1


# ---------------------------------------------------------------------------
# 1. 미너비니 청산 시뮬 (사전계산)
# ---------------------------------------------------------------------------

def _first_tr(TR, ci, e, lo, hi):
    """오프셋 [lo, hi] 에서 첫 거래 가능 오프셋, 없으면 −1."""
    for j in range(lo, hi + 1):
        if TR[ci, e + j]:
            return j
    return -1


def _sim_one_py(Op, H, L, C, MA, TR, last_present, ci, d, c_low, stop_floor, stop_cap, stop_mult, fixed_stop,
                take_r, ts_days, max_hold, has_trail, lu_ratio, lu_skip, pivot, fb):
    """한 사건 → ``(status, x_abs, entry, px1, px2, dd, partial_abs, reason)``. 규칙은 모듈 설명·``minervini_exits`` 1~12.

    배열은 종목×날짜(``_Book`` 전치). ``ts_days ≤ 0`` = 시간 손절 끔, ``take_r = inf`` = 익절 없음.
    ``fb`` = 돌파 실패(규칙 12 — ``minervini_exits`` ``failed_breakout``): 익절 전 1 ~ max_hold−1 일째 종가 < ``pivot`` 인 첫날(그날
    장중 손절·익절·시간 손절보다 앞선 날만) → 다음 거래 가능일 시가 전량. 익절 없음 ∧ 전량 추적과 겹치면 청산일이 이른 쪽(같은 날이면 돌파 실패).
    """
    nan = np.nan
    n = Op.shape[1]
    e = d + 1
    if e >= n:
        return ST_NO_NEXT, -1, nan, nan, nan, nan, -1, -1
    if not TR[ci, e]:
        return ST_NO_OPEN, -1, nan, nan, nan, nan, -1, -1
    entry = Op[ci, e]
    prev = C[ci, d]
    if lu_skip and np.isfinite(prev) and prev > 0 and entry >= prev * lu_ratio:
        return ST_LIMIT_UP, -1, nan, nan, nan, nan, -1, -1
    if not np.isfinite(c_low):
        return ST_NO_CONTRACTION, -1, nan, nan, nan, nan, -1, -1
    if entry <= c_low:
        return ST_BELOW_LOW, -1, nan, nan, nan, nan, -1, -1

    if np.isnan(fixed_stop):
        dd = 1.0 - c_low * stop_mult / entry
        dd = min(max(dd, stop_floor), stop_cap)
    else:
        dd = fixed_stop
    stop_px = entry * (1.0 - dd)
    tgt = entry * (1.0 + take_r * dd)
    partial_on = np.isfinite(take_r)

    Ln = n - e
    k_last = max_hold - 1
    j_time = _first_tr(TR, ci, e, k_last, Ln - 1) if k_last <= Ln - 1 else -1
    horizon = j_time if j_time >= 0 else Ln - 1
    last_tr = 0
    for j in range(horizon, -1, -1):
        if TR[ci, e + j]:
            last_tr = j
            break
    sig_n = min(k_last, horizon + 1)

    # --- 1단계: 전량 — 손절·익절 첫날(같은 날 둘 다면 손절) ---
    j_s = -1
    j_p = -1
    for j in range(horizon + 1):
        if not TR[ci, e + j]:
            continue
        op = Op[ci, e + j]
        if L[ci, e + j] <= stop_px or op <= stop_px:
            j_s = j
            break
        if partial_on and (H[ci, e + j] >= tgt or op >= tgt):
            j_p = j
            break
    # 익절 없음 ∧ 추적: 전량 추적(sepa MA50 청산 식)
    j_t1 = -1
    if (not partial_on) and has_trail:
        for j in range(sig_n):
            if TR[ci, e + j] and C[ci, e + j] < MA[ci, e + j]:
                j_t1 = _first_tr(TR, ci, e, j + 1, horizon) if j + 1 <= horizon else -1
                break
    j_ts = -1
    if ts_days > 0 and ts_days - 1 <= horizon:
        cand = _first_tr(TR, ci, e, ts_days - 1, horizon)
        if (cand >= 0 and (j_s < 0 or cand < j_s) and (j_p < 0 or cand < j_p) and (j_t1 < 0 or cand < j_t1)
                and C[ci, e + cand] <= entry):
            j_ts = cand
    # 돌파 실패(규칙 12)
    if fb and np.isfinite(pivot):
        lim = sig_n
        if j_s >= 0 and j_s < lim:
            lim = j_s
        if j_p >= 0 and j_p < lim:
            lim = j_p
        if j_ts >= 0 and j_ts < lim:
            lim = j_ts
        k_fb = -1
        for j in range(lim):
            if TR[ci, e + j] and C[ci, e + j] < pivot:
                k_fb = j
                break
        if k_fb >= 0:
            j_fb = _first_tr(TR, ci, e, k_fb + 1, horizon) if k_fb + 1 <= horizon else -1
            if j_t1 >= 0 and (j_fb < 0 or j_t1 < j_fb):
                px = Op[ci, e + j_t1]
                return ST_OK, e + j_t1, entry, px, px, dd, -1, R_TRAIL
            if j_fb >= 0:
                px = Op[ci, e + j_fb]
                return ST_OK, e + j_fb, entry, px, px, dd, -1, R_FAILED
            if j_time >= 0:
                px = C[ci, e + j_time]
                return ST_OK, e + j_time, entry, px, px, dd, -1, R_TIME
            px = C[ci, e + last_tr]
            return ST_OK, e + last_tr, entry, px, px, dd, -1, (R_DATA_END if last_present[ci] else R_DELISTED)

    if j_ts >= 0:
        px = C[ci, e + j_ts]
        return ST_OK, e + j_ts, entry, px, px, dd, -1, R_TIME_STOP
    if j_p < 0:
        if j_s >= 0 and (j_t1 < 0 or j_s < j_t1):
            px = min(Op[ci, e + j_s], stop_px)
            return ST_OK, e + j_s, entry, px, px, dd, -1, R_STOP
        if j_t1 >= 0:
            px = Op[ci, e + j_t1]
            return ST_OK, e + j_t1, entry, px, px, dd, -1, R_TRAIL
        if j_time >= 0:
            px = C[ci, e + j_time]
            return ST_OK, e + j_time, entry, px, px, dd, -1, R_TIME
        px = C[ci, e + last_tr]
        return ST_OK, e + last_tr, entry, px, px, dd, -1, (R_DATA_END if last_present[ci] else R_DELISTED)

    # --- 2단계: 절반 익절 후 나머지 절반 ---
    gap_up = Op[ci, e + j_p] >= tgt
    px1 = Op[ci, e + j_p] if gap_up else tgt
    j_b = -1
    b_px = nan
    if gap_up and L[ci, e + j_p] <= entry:
        j_b = j_p
        b_px = entry
    else:
        for j in range(j_p + 1, horizon + 1):
            if TR[ci, e + j] and (L[ci, e + j] <= entry or Op[ci, e + j] <= entry):
                j_b = j
                b_px = min(Op[ci, e + j], entry)
                break
    j_t = -1
    if has_trail and j_p < sig_n and j_b != j_p:
        for j in range(j_p, sig_n):
            if TR[ci, e + j] and C[ci, e + j] < MA[ci, e + j]:
                j_t = _first_tr(TR, ci, e, j + 1, horizon) if j + 1 <= horizon else -1
                break
    if j_t >= 0 and (j_b < 0 or j_t <= j_b):
        return ST_OK, e + j_t, entry, px1, Op[ci, e + j_t], dd, e + j_p, R_TRAIL
    if j_b >= 0:
        return ST_OK, e + j_b, entry, px1, b_px, dd, e + j_p, R_BREAKEVEN
    if j_time >= 0:
        return ST_OK, e + j_time, entry, px1, C[ci, e + j_time], dd, e + j_p, R_TIME
    reason = R_DATA_END if last_present[ci] else R_DELISTED
    return ST_OK, e + last_tr, entry, px1, C[ci, e + last_tr], dd, e + j_p, reason


def _sim_pairs_py(Op, H, L, C, MA, TR, last_present, pc, pd_, pclow, stop_floor, stop_cap, stop_mult, fixed_stop,
                  take_r, ts_days, max_hold, has_trail, lu_ratio, lu_skip, ppiv, fb):
    """쌍 배열(종목 열 ``pc``·신호일 행 ``pd_``·수축 저점 ``pclow``) 전부를 시뮬(병렬).

    Returns: ``status(int8), x_abs(int32), ret(f64), r1(f64), r2(f64), dd(f32), partial_abs(int32), reason(int8)``.
    ``ret = 0.5·((px1/진입−1) + (px2/진입−1))`` — 원본과 같은 연산 순서(패리티 1e−9).
    """
    m = pc.shape[0]
    status = np.empty(m, np.int8)
    x_abs = np.full(m, -1, np.int32)
    ret = np.full(m, np.nan)
    r1 = np.full(m, np.nan)
    r2 = np.full(m, np.nan)
    ddo = np.full(m, np.nan, np.float32)
    pab = np.full(m, -1, np.int32)
    reason = np.full(m, -1, np.int8)
    for i in prange(m):
        st, x, ent, px1, px2, dd, pa, rs = _sim_one(Op, H, L, C, MA, TR, last_present, pc[i], pd_[i], pclow[i],
                                                    stop_floor, stop_cap, stop_mult, fixed_stop, take_r, ts_days,
                                                    max_hold, has_trail, lu_ratio, lu_skip, ppiv[i], fb)
        status[i] = st
        if st == ST_OK:
            h1 = px1 / ent - 1.0
            h2 = px2 / ent - 1.0
            ret[i] = 0.5 * (h1 + h2)
            r1[i] = h1
            r2[i] = h2
            ddo[i] = dd
            x_abs[i] = x
            pab[i] = pa
            reason[i] = rs
    return status, x_abs, ret, r1, r2, ddo, pab, reason


# ---------------------------------------------------------------------------
# 2. 같은 종목 겹침 제거 (sepa / minervini_exits 의 _run 과 같은 규칙)
# ---------------------------------------------------------------------------

def _greedy_py(ok, pc, pd_, x_abs, n_codes, i0, i1):
    """[i0, i1) 쌍(신호일 → 종목 순 정렬) 중 ``ok`` 인 것을 진입일 순 탐욕으로 — 보유 중(신호일 < 그 종목 청산 행) 신호 버림.

    청산 행 당일 신호는 허용(원본 ``d < busy_until`` 과 같다). 반환: 고른 쌍 번호(오름차순).
    """
    busy = np.full(n_codes, -1, np.int64)
    out = np.empty(i1 - i0, np.int64)
    k = 0
    for i in range(i0, i1):
        if not ok[i]:
            continue
        c = pc[i]
        if pd_[i] < busy[c]:
            continue
        busy[c] = x_abs[i]
        out[k] = i
        k += 1
    return out[:k]


# ---------------------------------------------------------------------------
# 3. 결정적 해시 난수 · 부트스트랩 하단
# ---------------------------------------------------------------------------

def _h32_py(x):
    """32비트 정수 해시(lowbias32 변형, 곱셈 상수 < 2³¹ 라 int64 곱이 넘치지 않는다)."""
    x = x & M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x5BD1E995) & M32
    x ^= x >> 16
    return x


def _pct_sorted_py(s, q):
    """정렬된 배열의 백분위(``np.percentile`` 기본 선형 보간과 같다)."""
    n = s.shape[0]
    pos = q / 100.0 * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def _boot_means_py(x, n_boot, seed):
    """표본 ``x`` 의 부트스트랩 평균 ``n_boot`` 개(복원 추출, 해시 난수)."""
    n = x.shape[0]
    out = np.empty(n_boot)
    hs = _h32(seed * 0x2545F491 + 0x632BE5AB)
    for b in prange(n_boot):
        hb = _h32(hs ^ _h32(b + 0x165667B1))
        s = 0.0
        for i in range(n):
            u = _h32(hb ^ _h32(i * 3 + 1))
            s += x[(u * n) >> 32]
        out[b] = s / n
    return out


def _boot_ratio_py(num, den, n_boot, seed):
    """비율 추정량 Σnum/Σden 의 부트스트랩(트레이드 단위 복원 추출 — 1일당 초과)."""
    n = num.shape[0]
    out = np.empty(n_boot)
    hs = _h32(seed * 0x2545F491 + 0x632BE5AB)
    for b in prange(n_boot):
        hb = _h32(hs ^ _h32(b + 0x165667B1))
        sn = 0.0
        sd = 0.0
        for i in range(n):
            u = _h32(hb ^ _h32(i * 3 + 1))
            k = (u * n) >> 32
            sn += num[k]
            sd += den[k]
        out[b] = sn / sd if sd > 0 else np.nan
    return out


def _block_boot_py(monthly, block, n_boot, seed):
    """월 수익열의 이동 블록 부트스트랩(블록 ``block`` 개월) → 연환산 평균(×12) 표본."""
    M = monthly.shape[0]
    out = np.empty(n_boot)
    nstart = M - block + 1
    nb = (M + block - 1) // block
    hs = _h32(seed * 0x2545F491 + 0x1B873593)
    for b in prange(n_boot):
        hb = _h32(hs ^ _h32(b + 0x27D4EB2F))
        s = 0.0
        cnt = 0
        for k in range(nb):
            u = _h32(hb ^ _h32(k * 3 + 1))
            st = (u * nstart) >> 32
            for j in range(block):
                if cnt < M:
                    s += monthly[st + j]
                    cnt += 1
        out[b] = s / M * 12.0
    return out


# ---------------------------------------------------------------------------
# 4. 정합 대조 추출 (쌍마다 고정 — config 와 무관)
# ---------------------------------------------------------------------------

def _draw_controls_py(strat, ret, members, mem_strat, n_per, seed):
    """쌍 ``i`` 마다 같은 층(``strat[i]``)의 풀 쌍 ``n_per`` 개를 자기 제외·복원 추출 → 대조 평균 수익(없으면 NaN).

    ``members`` = 풀 쌍 번호를 (층, 번호) 순으로 정렬, ``mem_strat`` = 그 층. 난수는 (쌍 번호, 추출 번호) 해시라
    어느 부분집합을 고르든 그 쌍의 대조는 같다("대조 고정").
    """
    m = strat.shape[0]
    out = np.full(m, np.nan)
    nm = members.shape[0]
    hs = _h32(seed * 0x2545F491 + 0x85EBCA6B)
    for i in prange(m):
        s = strat[i]
        if s < 0:
            continue
        lo = np.searchsorted(mem_strat, s, side="left")
        hi = np.searchsorted(mem_strat, s, side="right")
        cnt = hi - lo
        if cnt <= 0:
            continue
        sp = -1
        if lo < nm:
            k = lo + np.searchsorted(members[lo:hi], i)
            if k < hi and members[k] == i:
                sp = k - lo
        eff = cnt - (1 if sp >= 0 else 0)
        if eff <= 0:
            continue
        hi_ = _h32(hs ^ _h32(i * 2 + 1))
        acc = 0.0
        for j in range(n_per):
            u = _h32(hi_ ^ _h32(j * 7 + 3))
            k = (u * eff) >> 32
            if sp >= 0 and k >= sp:
                k += 1
            acc += ret[members[lo + k]]
        out[i] = acc / n_per
    return out


# ---------------------------------------------------------------------------
# 5. 진입 마스크 (GA1 = RELAX 공간 엄격 / GA2 = 끔 끝 포함)
# ---------------------------------------------------------------------------

def _mask_ga1_py(i0, i1, tt, rs, brk_bits, vol, vavg, pead, rel, breadth, band, rs_min, w_idx, vol_mult, pead_min,
                 rel_max, breadth_min, band_sel, extra_ok):
    """GA1(RELAX 공간): 추세 템플릿 ∧ RS ≥ ∧ 피벗 돌파(종가 > 피벗_w ∧ 평균_w > 0 ∧ 거래량 ≥ vm × 평균_w — ``pivot_breakout``
    과 같은 식) ∧ PEAD ≥ ∧ rel ≤ ∧ breadth >.

    NaN 은 전부 통과 안 함(``convergence_run.arm_mask``·``e_pass`` 와 같다). ``band_sel`` −1 = 전체, 0 중소형, 1 유동.
    ``extra_ok`` 는 추가 조건(F=1 등, 없으면 전부 True). 체결 여부(status·추격)는 여기서 보지 않는다.
    """
    out = np.zeros(i1 - i0, np.bool_)
    bit = 1 << w_idx
    for i in prange(i0, i1):
        if not tt[i] or not extra_ok[i]:
            continue
        if band_sel >= 0 and band[i] != band_sel:
            continue
        if not (rs[i] >= rs_min):
            continue
        if (brk_bits[i] & bit) == 0 or not (vavg[i, w_idx] > 0 and vol[i] >= vol_mult * vavg[i, w_idx]):
            continue
        if not (pead[i] >= pead_min) or not (rel[i] <= rel_max) or not (breadth[i] > breadth_min):
            continue
        out[i - i0] = True
    return out


def _mask_ga2_py(i0, i1, nh, al, stack, slope_bits, rs, brk_bits, vol, vavg, pead, rel, breadth, band, g, band_sel,
                 extra_ok, eps_yoy, eps_acc, rev_yoy):
    """GA2 진입: 유전자 벡터 ``g`` (러너 :data:`G2_*` 순서, 끔 = NaN 또는 0). 끔인 조건은 NaN 도 통과(조건 자체가 없다).

    g = [near_high, above_low, ma_stack, slope_idx(0 끔 / 1·2·3 = 10·20·40일 비트), rs_min, pivot_w_idx(−1 = 돌파 없음),
    vol_mult, breadth_min, pead_min, rel_max, eps_yoy_min, eps_accel(0/1), rev_yoy_min] — 연속 유전자 끔 = NaN.
    거래량 배수의 평균 창 = 피벗 창(돌파 없음이면 50일, ``vavg`` 의 50일 열 = 3).
    """
    out = np.zeros(i1 - i0, np.bool_)
    nh_x, al_x, stk, slp, rs_x, pw, vm, br_x, pe_x, rel_x = g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9]
    eps_x, acc_on, rev_x = g[10], g[11] > 0, g[12]
    k_stack = int(stk)
    slope_bit = (1 << (int(slp) - 1)) if slp > 0 else 0
    pw_i = int(pw)
    v_col = pw_i if pw_i >= 0 else 3
    for i in prange(i0, i1):
        if not extra_ok[i]:
            continue
        if band_sel >= 0 and band[i] != band_sel:
            continue
        if not np.isnan(nh_x) and not (nh[i] >= nh_x):
            continue
        if not np.isnan(al_x) and not (al[i] >= al_x):
            continue
        if k_stack > 0 and stack[i] < k_stack:
            continue
        if slope_bit != 0 and (slope_bits[i] & slope_bit) == 0:
            continue
        if not np.isnan(rs_x) and not (rs[i] >= rs_x):
            continue
        if pw_i >= 0 and (brk_bits[i] & (1 << pw_i)) == 0:
            continue
        if not np.isnan(vm) and not (vavg[i, v_col] > 0 and vol[i] >= vm * vavg[i, v_col]):
            continue
        if not np.isnan(br_x) and not (breadth[i] > br_x):
            continue
        if not np.isnan(pe_x) and not (pead[i] >= pe_x):
            continue
        if not np.isnan(rel_x) and not (rel[i] <= rel_x):
            continue
        if not np.isnan(eps_x) and not (eps_yoy[i] >= eps_x):
            continue
        if acc_on and eps_acc[i] == 0:
            continue
        if not np.isnan(rev_x) and not (rev_yoy[i] >= rev_x):
            continue
        out[i - i0] = True
    return out


# ---------------------------------------------------------------------------
# 6. 절단(마크투마켓) · 대조 인덱스 · 트레이드 초과 · 자본 제약 포트폴리오
# ---------------------------------------------------------------------------

def _draw_idx_py(strat, members, mem_strat, n_per, seed):
    """:func:`_draw_controls_py` 와 같은 추출(같은 해시·같은 순서)을 평균 대신 **대조 쌍 번호**(m × n_per, 없으면 −1)로.

    대조를 번호로 두면 학습창 끝 절단(마크투마켓)을 대조에도 같은 날 걸 수 있다.
    """
    m = strat.shape[0]
    out = np.full((m, n_per), -1, np.int32)
    nm = members.shape[0]
    hs = _h32(seed * 0x2545F491 + 0x85EBCA6B)
    for i in prange(m):
        s = strat[i]
        if s < 0:
            continue
        lo = np.searchsorted(mem_strat, s, side="left")
        hi = np.searchsorted(mem_strat, s, side="right")
        cnt = hi - lo
        if cnt <= 0:
            continue
        sp = -1
        if lo < nm:
            k = lo + np.searchsorted(members[lo:hi], i)
            if k < hi and members[k] == i:
                sp = k - lo
        eff = cnt - (1 if sp >= 0 else 0)
        if eff <= 0:
            continue
        hi_ = _h32(hs ^ _h32(i * 2 + 1))
        for j in range(n_per):
            u = _h32(hi_ ^ _h32(j * 7 + 3))
            k = (u * eff) >> 32
            if sp >= 0 and k >= sp:
                k += 1
            out[i, j] = members[lo + k]
    return out


def _mark_py(i, cut, pc, pd_, x_abs, pab, ret, r2, Op, C):
    """쌍 ``i`` 트레이드를 ``cut`` 행 **직전 거래일 종가**에서 마크투마켓으로 끊는다 → ``(ret, x, r2, pab)``.

    청산 행 < cut 이면 그대로. 아니면 청산 = cut−1 행, 남은 포지션(절반 익절이 cut 전이면 절반) 가격 = cut−1 행 이하 마지막 유한 종가
    (진입일까지 거슬러, 없으면 진입가). cut 이후 가격은 쓰지 않는다. 절반 익절이 cut 이후면 익절 없음으로 본다.
    """
    x = int(x_abs[i])
    if x < cut:
        return float(ret[i]), x, float(r2[i]), int(pab[i])
    c = pc[i]
    e = pd_[i] + 1
    t_last = int(cut) - 1
    if t_last < e:
        return float(ret[i]), x, float(r2[i]), int(pab[i])
    ent = Op[c, e]
    m = 0.0
    for t in range(t_last, e - 1, -1):
        cl = C[c, t]
        if np.isfinite(cl):
            m = cl / ent - 1.0
            break
    pa = int(pab[i])
    if pa >= 0 and pa < cut:
        r1 = 2.0 * float(ret[i]) - float(r2[i])
        return 0.5 * (r1 + m), t_last, m, pa
    return m, t_last, m, -1


def _trade_excess_py(sel, cutv, pc, pd_, x_abs, pab, ret, r2, Op, C, cidx, cost_tab, market):
    """선택 트레이드(쌍 번호) → (초과, 보유일, 비용, 수익, 청산 행) — 트레이드·대조 모두 그 트레이드의 ``cutv`` 에서 마크투마켓.

    초과 = 트레이드 수익 − 대조 n_per 개 수익 평균(대조가 하나도 없으면 NaN). 비용 = ``cost_tab[청산 행, 시장]``.
    """
    n = sel.shape[0]
    ex = np.full(n, np.nan)
    hold = np.empty(n)
    cost = np.empty(n)
    rr = np.empty(n)
    xx = np.empty(n, np.int64)
    n_per = cidx.shape[1]
    for k in prange(n):
        i = sel[k]
        cut = cutv[i]
        rt, xe, _, _ = _mark(i, cut, pc, pd_, x_abs, pab, ret, r2, Op, C)
        rr[k] = rt
        xx[k] = xe
        hold[k] = xe - pd_[i]
        mk = market[i] if market[i] >= 0 else 0
        cost[k] = cost_tab[xe, mk]
        if cidx[i, 0] < 0:
            continue
        acc = 0.0
        for j in range(n_per):
            cr, _, _, _ = _mark(cidx[i, j], cut, pc, pd_, x_abs, pab, ret, r2, Op, C)
            acc += cr
        ex[k] = rt - acc / n_per
    return ex, hold, cost, rr, xx


def _portfolio_py(sig, fill, orderable, pc, pd_, day_start, t0, t1, prio_rel, prio_rs, x_abs, pab, ret, r2, Op, C,
                  cidx, cost_tab, market, cutv, slots, slot_w):
    """슬롯 ``slots`` 개 × 자산 ``slot_w`` 일 루프 → 날짜별 포트폴리오 정합 초과(``t0 ≤ t < t1``).

    - 신호일 ``t`` 의 신호(``sig``, 사건 조건 통과) 중 ``orderable`` 이고 보유 중이 아닌 것을 rel 오름차순(NaN 뒤) → RS 내림차순 →
      종목 순으로 빈 슬롯 수만큼 주문. 체결(``fill`` — 시뮬 체결 ∧ 추격 통과)이면 ``t+1`` 시가 진입, 아니면 그 슬롯은 그날 빈다.
    - 포지션은 그 쌍의 ``cutv`` 직전 거래일 종가에서 마크투마켓으로 끊는다(:func:`_mark_py` — 학습창 끝·폴드 경계). 대조도 같은 날 끊는다.
    - 보유 중 날별 손익 = 남은 비율 × (그날 종가 − 직전 표시가)/진입가, 절반 익절일·청산일은 체결가로(합 = 트레이드 수익).
      진입 비중은 진입 때 고정(자산 1 대비 ``slot_w``, 복리·재조정 없음) — 일별 합 = Σ ``slot_w`` × (수익 − 대조 − 비용).
    - 대조는 (끊은) 대조 평균 수익을 보유일로 나눠 보유일마다 뺀다(일할). 비용은 청산일에.
    - **짝 없는 포지션(대조 없음)은 손익·비용 모두 0** — 슬롯만 차지한다.
    반환: ``(daily[t0:t1], opened 쌍 번호, n_orders, n_unmatched)`` — n_orders = 슬롯을 받은 주문 수(체결 여부 무관).
    """
    T = t1 - t0
    daily = np.zeros(T)
    s_i = np.full(slots, -1, np.int64)
    s_lm = np.zeros(slots)
    s_fr = np.zeros(slots)
    s_x = np.zeros(slots, np.int64)
    s_pa = np.zeros(slots, np.int64)
    s_r1 = np.zeros(slots)
    s_r2 = np.zeros(slots)
    s_cd = np.zeros(slots)
    s_cost = np.zeros(slots)
    s_m = np.zeros(slots, np.bool_)
    opened = np.empty(day_start[t1] - day_start[t0] + 1, np.int64)
    n_open = 0
    n_ord = 0
    n_unm = 0
    n_per = cidx.shape[1]
    cand = np.empty(4096, np.int64)
    for t in range(t0, t1):
        acc = 0.0
        for s in range(slots):
            i = s_i[s]
            if i < 0:
                continue
            c = pc[i]
            e = pd_[i] + 1
            if t < e:
                continue
            ent = Op[c, e]
            contrib = 0.0
            if s_pa[s] == t:
                contrib += 0.5 * (s_r1[s] - s_lm[s])
                s_fr[s] = 0.5
            if s_x[s] == t:
                contrib += s_fr[s] * (s_r2[s] - s_lm[s])
            else:
                cl = C[c, t]
                if np.isfinite(cl):
                    m = cl / ent - 1.0
                    contrib += s_fr[s] * (m - s_lm[s])
                    s_lm[s] = m
            if s_m[s]:
                acc += slot_w * (contrib - s_cd[s])
                if s_x[s] == t:
                    acc -= slot_w * s_cost[s]
            if s_x[s] == t:
                s_i[s] = -1
        daily[t - t0] = acc
        a = day_start[t]
        b = day_start[t + 1]
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
            if not sig[i] or not orderable[i]:
                continue
            held = False
            for s in range(slots):
                if s_i[s] >= 0 and pc[s_i[s]] == pc[i]:
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
                ri = prio_rel[i] if np.isfinite(prio_rel[i]) else np.inf
                rj = prio_rel[j] if np.isfinite(prio_rel[j]) else np.inf
                if ri < rj:
                    best = q
                elif ri == rj:
                    si = prio_rs[i] if np.isfinite(prio_rs[i]) else -np.inf
                    sj = prio_rs[j] if np.isfinite(prio_rs[j]) else -np.inf
                    if si > sj or (si == sj and pc[i] < pc[j]):
                        best = q
            i = cand[best]
            cand[best] = -1
            n_ord += 1
            if fill[i]:
                cut = cutv[i]
                rt, xe, r2e, pae = _mark(i, cut, pc, pd_, x_abs, pab, ret, r2, Op, C)
                for s in range(slots):
                    if s_i[s] < 0:
                        s_i[s] = i
                        s_lm[s] = 0.0
                        s_fr[s] = 1.0
                        s_x[s] = xe
                        s_pa[s] = pae
                        s_r2[s] = r2e
                        s_r1[s] = 2.0 * rt - r2e
                        mk = market[i] if market[i] >= 0 else 0
                        s_cost[s] = cost_tab[xe, mk]
                        s_m[s] = cidx[i, 0] >= 0
                        if s_m[s]:
                            acc_c = 0.0
                            for j in range(n_per):
                                cr, _, _, _ = _mark(cidx[i, j], cut, pc, pd_, x_abs, pab, ret, r2, Op, C)
                                acc_c += cr
                            s_cd[s] = acc_c / n_per / (xe - pd_[i])
                        else:
                            s_cd[s] = 0.0
                            n_unm += 1
                        break
                opened[n_open] = i
                n_open += 1
    return daily, opened[:n_open], n_ord, n_unm


# --- 내보내기: 이름_py = 순수 파이썬, 이름 = numba(있으면) ---
_first_tr = _jit(_first_tr)
_h32 = _jit(_h32_py)
_pct_sorted = _jit(_pct_sorted_py)
_sim_one = _jit(_sim_one_py)
sim_pairs = _jit(_sim_pairs_py, parallel=True)
greedy = _jit(_greedy_py)
boot_means = _jit(_boot_means_py, parallel=True)
boot_ratio = _jit(_boot_ratio_py, parallel=True)
block_boot = _jit(_block_boot_py, parallel=True)
draw_controls = _jit(_draw_controls_py, parallel=True)
draw_idx = _jit(_draw_idx_py, parallel=True)
_mark = _jit(_mark_py)
trade_excess = _jit(_trade_excess_py, parallel=True)
mask_ga1 = _jit(_mask_ga1_py, parallel=True)
mask_ga2 = _jit(_mask_ga2_py, parallel=True)
portfolio = _jit(_portfolio_py)


def boot_lower(x: np.ndarray, *, n_boot: int = 500, seed: int = 0, q: float = 2.5, py: bool = False) -> float:
    """부트스트랩 평균의 ``q`` 백분위(``_boot_lower`` 와 같은 모양, 표본 < 5 면 −1.0)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    if len(x) < 5:
        return -1.0
    f = _boot_means_py if py else boot_means
    return float(np.percentile(f(x, n_boot, seed), q))


def boot_lower_ratio(num: np.ndarray, den: np.ndarray, *, n_boot: int = 500, seed: int = 0, q: float = 2.5,
                     py: bool = False) -> float:
    """Σnum/Σden 부트스트랩의 ``q`` 백분위(표본 < 5 면 −1.0)."""
    num = np.ascontiguousarray(num, dtype=np.float64)
    den = np.ascontiguousarray(den, dtype=np.float64)
    if len(num) < 5:
        return -1.0
    f = _boot_ratio_py if py else boot_ratio
    v = f(num, den, n_boot, seed)
    return float(np.nanpercentile(v, q))


def floor_penalty(n: int, floor: int) -> float:
    """``make_objective`` 와 같은 거래수 하한 페널티(단조 → 유효 구간으로 오르게)."""
    return -1.0 + n / floor * 1e-3


def stratum_order(strat: np.ndarray) -> np.ndarray:
    """층이 유효한 쌍 번호 전부를 (층, 번호) 순으로 — 한 번 정렬해 두고 풀마다 거르기만 한다."""
    idx = np.flatnonzero(strat >= 0).astype(np.int64)
    return np.ascontiguousarray(idx[np.lexsort((idx, strat[idx]))])


def stratum_index(strat: np.ndarray, ok: np.ndarray, order: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """대조 풀(``ok`` ∧ 층 유효) 쌍 번호를 (층, 번호) 순으로 → ``(members, mem_strat)``. ``order`` = :func:`stratum_order`."""
    if order is None:
        order = stratum_order(strat)
    members = np.ascontiguousarray(order[ok[order]])
    return members, np.ascontiguousarray(strat[members])


# ---------------------------------------------------------------------------
# 7. 파이썬 쪽 묶음 — 사건 표 → 엔진 트레이드 (패리티 테스트·러너 공용)
# ---------------------------------------------------------------------------

#: 미너비니 원전 청산(사전등록 기본값) — :data:`minervini_exits.PREREG_MINERVINI` 와 같은 값, 엔진 인자 이름으로.
EXIT_PREREG = {"stop_floor": 0.03, "stop_cap": 0.10, "stop_mult": 0.99, "fixed_stop": np.nan, "take_r": 2.0,
               "ts_days": 20, "max_hold": 120, "has_trail": True, "lu_skip": True, "fb": False}


def chase_ok(entry_open: np.ndarray, pivot: np.ndarray, chase: float) -> np.ndarray:
    """추격 금지 통과(원본 식 ``not (isfinite(pivot) and entry > pivot × (1+chase))``). ``chase = inf`` 면 전부 통과."""
    if not np.isfinite(chase):
        return np.ones(len(entry_open), bool)
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(pivot) & (entry_open > pivot * (1.0 + chase))
    return ~bad


def sim_book(Op, H, L, C, MA, TR, last_present, pc, pd_, pclow, params: dict, *, py: bool = False, pivot=None):
    """쌍 배열 시뮬 — ``params`` 는 :data:`EXIT_PREREG` 모양. ``py`` = 순수 파이썬 경로(테스트). ``pivot`` = 돌파 실패 기준 피벗(쌍별)."""
    from research.signals.sepa import LIMIT_UP_RATIO  # noqa: PLC0415

    f = _sim_pairs_py if py else sim_pairs
    ts = params["ts_days"]
    piv = np.full(len(pc), np.nan) if pivot is None else np.ascontiguousarray(pivot, np.float64)
    return f(Op, H, L, C, MA, TR, last_present, np.asarray(pc, np.int64), np.asarray(pd_, np.int64),
             np.asarray(pclow, np.float64), float(params["stop_floor"]), float(params["stop_cap"]),
             float(params["stop_mult"]), float(params["fixed_stop"]), float(params["take_r"]),
             int(ts) if ts is not None else 0, int(params["max_hold"]), bool(params["has_trail"]),
             float(LIMIT_UP_RATIO), bool(params["lu_skip"]), piv, bool(params.get("fb", False)))


def engine_events(book, events, params: dict, *, chase: float = 0.05, no_overlap: bool = True, py: bool = False):
    """사건 표(``code, event_day, pivot, contraction_low``) → 엔진 트레이드 표(``simulate_minervini`` 비교용).

    ``book`` 은 :class:`research.signals.sepa._Book`. 정렬·중복 제거는 원본 ``_run`` 과 같다(신호일 → 종목 코드 순).
    """
    import pandas as pd  # noqa: PLC0415

    col_of = pd.Series(np.arange(len(book.codes)), index=book.codes)
    ev = events[events["code"].isin(book.codes) & events["event_day"].isin(book.days)].copy()
    ev["sim_d"] = book.days.get_indexer(ev["event_day"])
    ev["sim_c"] = col_of.reindex(ev["code"]).to_numpy()
    ev = ev.drop_duplicates(["code", "event_day"]).sort_values(["sim_d", "code"]).reset_index(drop=True)
    pc, pd_ = ev["sim_c"].to_numpy(np.int64), ev["sim_d"].to_numpy(np.int64)
    last_present = np.ascontiguousarray(book.present[:, -1])
    st, x, ret, r1, r2, dd, pab, rs = sim_book(book.O, book.H, book.L, book.C, book.MA, book.tradable, last_present,
                                               pc, pd_, ev["contraction_low"].to_numpy(float), params, py=py,
                                               pivot=ev["pivot"].to_numpy(float))
    n = len(book.days)
    ent_open = np.where(pd_ + 1 < n, book.O[pc, np.minimum(pd_ + 1, n - 1)], np.nan)
    ok = (st == ST_OK) & chase_ok(ent_open, ev["pivot"].to_numpy(float), chase)
    if no_overlap:
        g = _greedy_py if py else greedy
        sel = g(ok, pc, pd_, x.astype(np.int64), len(book.codes), 0, len(ev))
    else:
        sel = np.flatnonzero(ok)
    out = pd.DataFrame({"code": ev["code"].to_numpy()[sel], "event_day": ev["event_day"].to_numpy()[sel],
                        "entry_day": book.days[pd_[sel] + 1], "exit_day": book.days[x[sel]], "ret": ret[sel],
                        "exit_reason": [REASON_NAMES[k] for k in rs[sel]], "hold_days": x[sel] - pd_[sel],
                        "d": dd[sel].astype(float), "partial_hit": pab[sel] >= 0})
    return out
