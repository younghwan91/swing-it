"""패턴 GA 학습창 적합도 — ``Engine.evaluate`` 의 파이썬 접착을 njit 한 번 호출로.

``Engine.candidates`` (표 거르기 → F1 하락 쐐기 거르기 → 체결 재료 모으기) → ``Engine.run`` (포트폴리오) →
``Engine.penalty`` (체결 하한 벌점) → ``Engine.monthly`` + ``block_boot`` + ``np.percentile`` (학습창 하단) 을 한
커널에서 한다. 숫자는 :meth:`Engine.evaluate_ref` 와 **비트 단위로 같다**(``tests/test_pattern_eval.py``·실캐시
대조). 커널 자체(``filter_events``·``portfolio``)는 :mod:`pattern_fast` 의 것을 그대로 부른다 — 여기서 바뀌는 건
배열 복사·``astype``·``concatenate``·dict·``np.unique``·``np.percentile`` 같은 접착뿐이다.

- 벌점의 연도 평균은 ``np.mean(list)`` 와 같은 덧셈 순서(numpy 짝 합 — 8 미만 순차, 8 이상 8 갈래)로 낸다 —
  창의 연도 수 < 128 만 받는다(그 이상은 numpy 가 재귀로 나눈다).
- 월 합·블록 부트스트랩·백분위는 ``krx_quant_core.research.evolve.kernels`` 의 njit 판(``*_nb``)을 쓴다 —
  ``ga_fast.block_boot`` + ``np.percentile`` 과 비트 동일(코어 시험).
- :func:`screen_kernel` = 건수만 선별(재시작 풀)을 덩어리 하나에 njit 한 번. 행마다 먼저 :func:`_prunable` 이 사건을 **연도
  조각**으로(싼 해부터) 거르며 연도 하한 ``ny[y] ≥ FLOOR_YEAR·f``·총 하한 ``n ≥ ffloor`` 의 상한(진입은 사건일에만,
  재진입은 원 트레이드 하나에 많아야 하나)을 대조해 실현 불가가 드러나면 그 행을 버린다(``n = -1``). 버린 행은
  :meth:`Engine.evaluate` 로도 실현 불가이고, 살아남은 행은 원래 계산 그대로다 — 선별 결과(실현 가능 행·진입 수)는 행마다
  부르는 것과 같다(``tests/test_pattern_eval.py``·실캐시 대조).
- numba 가 없으면 같은 소스가 파이썬으로 돈다(느리지만 같은 숫자). ``_jit`` 은 :mod:`vcp` 규약(``cache=False``) —
  numba 디스크 캐시는 다른 파일의 피호출 커널이 바뀌어도 무효화하지 않는다.
"""

from __future__ import annotations

import numpy as np
from krx_quant_core.research.evolve.kernels import block_boot_lower_nb, monthly_sums_nb

from research.signals import pattern_fast as PF
from research.signals.pattern_events import FI, II
from research.signals.vcp import _jit

__all__ = ["OUT_NAMES", "evaluate_kernel", "screen_kernel", "year_mean_np"]

_F_SUP, _F_SLO, _F_H0 = FI["s_up"], FI["s_lo"], FI["h0"]
_F_STRAT, _F_MKT, _F_U, _F_SU, _F_E, _F_SL = FI["strat"], FI["market"], FI["u_d"], FI["su"], FI["E"], FI["sl"]
_I_D, _I_CI, _I_I0, _I_ID, _I_IE = II["d"], II["ci"], II["i0"], II["iD"], II["iE"]
_N_Q, _N_X, _X_DYN, _X_RE = PF.N_Q, PF.N_X, PF.X_DYN, PF.X_REENTRY

#: ``out`` 칸 이름(순서).
OUT_NAMES = ("n", "n_pass", "pen", "feasible", "fit", "lower")


def _year_mean_py(v, m):
    """``np.mean(list(v[:m]))`` 과 같은 값(m < 128). numpy 짝 합: m < 8 이면 0 부터 순차, 아니면 8 갈래 누산
    (``r[j] += v[i+j]``) → ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`` → 나머지 순차. m = 0 이면 0.0(벌점 관례)."""
    if m == 0:
        return 0.0
    if m < 8:
        s = 0.0
        for i in range(m):
            s += v[i]
        return s / m
    r0, r1, r2, r3, r4, r5, r6, r7 = v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
    i = 8
    while i < m - (m % 8):
        r0 += v[i]
        r1 += v[i + 1]
        r2 += v[i + 2]
        r3 += v[i + 3]
        r4 += v[i + 4]
        r5 += v[i + 5]
        r6 += v[i + 6]
        r7 += v[i + 7]
        i += 8
    s = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < m:
        s += v[i]
        i += 1
    return s / m


year_mean_np = _jit(_year_mean_py)


def _violated_py(cnt, need, re_on, ffloor):
    """연도별 통과 상한 ``cnt`` 로 실현 가능(``ny[y] ≥ need[y]``·``n ≥ ffloor``)이 불가능한가. 진입은 사건일에만 일어나므로
    원 트레이드 ``ny[y] ≤ cnt[y]``, 재진입(원 트레이드 하나에 많아야 하나, 그 트레이드 진입 뒤)까지 ``ny[y] ≤ 2·cnt[y] + Σ_{y'<y} cnt``."""
    before = 0.0
    for y in range(cnt.shape[0]):
        c = cnt[y]
        if need[y] > 0:
            b = 2.0 * c + before if re_on else c
            if b < need[y] - 1e-9:
                return True
        before += c
    return (2.0 if re_on else 1.0) * before < ffloor - 1e-9


_violated = _jit(_violated_py)


def _prunable_py(F, Iv, aa, bb, gv, lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0, b_buf, fw_on, fw_lo,
                 re_on, year_of, n_years, yrow0, ffloor, yidx, yfrac, floor_year):
    """건수만 선별 가지치기 — 사건을 연도 조각으로 나눠(재진입 끔이면 사건이 적은 해부터, 켬이면 날짜 순) 거르며, 거른 해는 통과 수,
    아직 안 거른 해는 사건 수를 상한으로 :func:`_violated` 를 본다. 버퍼는 덮어쓴다(살아남은 행은 원래 순서로 다시 거른다)."""
    raw = np.zeros(n_years)
    slo = np.zeros(n_years, np.int64)
    shi = np.zeros(n_years, np.int64)
    k0 = aa
    while k0 < bb:
        y = year_of[Iv[k0, _I_D]]
        lim = yrow0[y + 1]
        lo, hi = k0 + 1, bb                           # 다음 해 첫 사건(사건은 표 안에서 날짜 순)
        while lo < hi:
            mid = (lo + hi) >> 1
            if Iv[mid, _I_D] < lim:
                lo = mid + 1
            else:
                hi = mid
        raw[y] = lo - k0
        slo[y] = k0
        shi[y] = lo
        k0 = lo
    need = np.zeros(n_years)
    for i in range(yfrac.shape[0]):
        if yfrac[i] > 0:
            need[yidx[i]] = floor_year * yfrac[i]
    cnt = raw.copy()
    done = np.zeros(n_years, np.bool_)
    while True:
        if _violated(cnt, need, re_on, ffloor):
            return True
        y = -1                                        # 재진입 켬: 날짜 순(앞 해 상한이 줄어야 뒤 해가 묶인다) · 끔: 사건이 적은 해부터
        for yy in range(n_years):
            if done[yy] or raw[yy] == 0:
                continue
            if y < 0 or (not re_on and (raw[yy] < raw[y] or (raw[yy] == raw[y] and need[yy] > need[y]))):
                y = yy
        if y < 0:
            return False
        done[y] = True
        ns = PF.filter_events(F, slo[y], shi[y], gv, lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0, b_buf)
        mk = 0
        for j in range(ns):
            if fw_on:
                ev = b_idx[j]
                su_ = F[ev, _F_SUP]
                sl_ = F[ev, _F_SLO]
                if not ((su_ < 0.0) and (sl_ < fw_lo) and (su_ < sl_)):
                    continue
            mk += 1
        cnt[y] = mk


_prunable = _jit(_prunable_py)


def _eval_py(F, Fo, Iv, aa, bb, gv, lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0, b_buf,
             fw_on, fw_lo, xv, Op, H, L, C, MA20, MA50, ATR, TR, LIM, bs, bn, strat_cs, cost_tab,
             cost_mult, pmix, t_lo, t_end, slots, slot_w, year_of, n_years, reform,
             month_of, ffloor, yidx, yfrac, floor_year, lam, counts_only, block, n_boot, seed, q, prune, yrow0, out):
    """학습창 적합도 한 번. ``out`` (길이 6, :data:`OUT_NAMES`) 을 채운다. ``counts_only`` 면 fit·lower 는 NaN.
    ``prune`` (``counts_only`` 일 때만) = :func:`_prunable` 이 실현 불가를 보이면 ``n = n_pass = -1``·``feasible = 0`` 으로 끝낸다
    (``yrow0[y]`` = 연도 번호 ``y`` 의 첫 날 행)."""
    if prune and _prunable(F, Iv, aa, bb, gv, lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0, b_buf, fw_on,
                           fw_lo, int(xv[_X_RE]) != 0, year_of, n_years, yrow0, ffloor, yidx, yfrac, floor_year):
        out[0] = -1
        out[1] = -1
        out[2] = np.nan
        out[3] = 0.0
        out[4] = np.nan
        out[5] = np.nan
        return
    n = PF.filter_events(F, aa, bb, gv, lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0,
                         b_buf) if bb > aa else 0
    keep = np.ones(n, np.bool_)
    if fw_on and n > 0:
        for j in range(n):
            ev = b_idx[j]
            su_ = F[ev, _F_SUP]
            sl_ = F[ev, _F_SLO]
            keep[j] = (su_ < 0.0) and (sl_ < fw_lo) and (su_ < sl_)
    m = 0
    for j in range(n):
        if keep[j]:
            m += 1
    out[4] = np.nan
    out[5] = np.nan
    ny = np.zeros(n_years, np.int64)
    if counts_only and m == 0:
        n_taken = 0
    else:
        cd = np.empty(m, np.int64)
        cci = np.empty(m, np.int64)
        ce = np.empty(m)
        csp = np.empty(m)
        cfb = np.empty(m)
        ch0 = np.empty(m)
        cu = np.empty(m)
        csu = np.empty(m)
        cE = np.empty(m)
        ciE = np.empty(m, np.int64)
        csl = np.empty(m)
        cbuf = np.empty(m)
        camb = np.empty(m, np.bool_)
        cskip0 = np.empty(m, np.bool_)
        cs0 = np.empty(m)
        cstrat = np.empty(m, np.int64)
        cmkt = np.empty(m, np.int64)
        cpar = np.zeros(m, np.int64)
        cev = np.empty(m, np.int64)
        cpid = np.empty(m, np.int64)
        k = 0
        for j in range(n):
            if not keep[j]:
                continue
            ev = b_idx[j]
            cev[k] = ev
            ce[k] = b_e[j]
            csp[k] = b_sp[j]
            cfb[k] = b_fb[j]
            camb[k] = b_amb[j]
            cskip0[k] = b_skip0[j]
            cs0[k] = b_s0[j]
            cbuf[k] = b_buf[j]
            cd[k] = Iv[ev, _I_D]
            cci[k] = Iv[ev, _I_CI]
            ch0[k] = F[ev, _F_H0]
            cstrat[k] = np.int64(Fo[ev, _F_STRAT])
            cmkt[k] = np.int64(Fo[ev, _F_MKT])
            cpid[k] = Iv[ev, _I_I0] * 100_000 + Iv[ev, _I_ID]
            cu[k] = Fo[ev, _F_U]
            csu[k] = Fo[ev, _F_SU]
            cE[k] = Fo[ev, _F_E]
            ciE[k] = Iv[ev, _I_IE]
            csl[k] = Fo[ev, _F_SL]
            k += 1
        PX = np.empty((2, _N_X))
        for c in range(_N_X):
            PX[0, c] = xv[c]
            PX[1, c] = xv[c]
        PX[1, _X_DYN] = 0.0
        nL = t_end - t_lo
        ret = np.zeros(nL)
        bench = np.zeros(nL)
        expo = np.zeros(nL)
        rec = np.zeros((2 * max(m, 1), _N_Q))
        res = PF.portfolio(cd, cci, ce, csp, cfb, ch0, cu, csu, cE, ciE, csl, cbuf, camb, cskip0, cs0,
                           cstrat, cmkt, cpar, cev, cpid, PX, Op, H, L, C, MA20, MA50, ATR, TR, LIM, bs,
                           bn, strat_cs, cost_tab, cost_mult, pmix, t_lo, t_end, slots, slot_w, ret,
                           bench, expo, rec, year_of, ny, reform, False, lu_ratio, 1)
        n_taken = res[0]
    # --- 벌점(Engine.penalty) ---
    pen_t = max(0.0, ffloor - n_taken) / ffloor if ffloor > 0 else 0.0
    py = np.empty(yfrac.shape[0])
    mp = 0
    for i in range(yfrac.shape[0]):
        f = yfrac[i]
        if f > 0:
            py[mp] = max(0.0, floor_year * f - ny[yidx[i]]) / (floor_year * f)
            mp += 1
    pen_y = year_mean_np(py, mp)
    pen = lam * (pen_t + pen_y)
    out[0] = n_taken
    out[1] = m
    out[2] = pen
    out[3] = 1.0 if (pen_t == 0 and pen_y == 0) else 0.0
    if counts_only:
        return
    # --- 학습창 하단(월 합 → 블록 부트스트랩 → 2.5 백분위) ---
    exc = ret - bench
    sums = np.empty(exc.shape[0])
    ids = np.empty(exc.shape[0], np.int64)
    km = monthly_sums_nb(exc, month_of, t_lo, sums, ids)
    lower = block_boot_lower_nb(sums[:km], block, n_boot, seed, q) if km >= 6 else -1.0
    out[5] = lower
    out[4] = lower - pen


evaluate_kernel = _jit(_eval_py)


def _screen_py(F, Fo, Iv, AA, BB, GV, lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0, b_buf, fw_on, FWLO,
               XV, Op, H, L, C, MA20, MA50, ATR, TR, LIM, bs, bn, strat_cs, cost_tab, cost_mult, pmix, t_lo, t_end,
               slots, slot_w, year_of, n_years, reform, month_of, ffloor, yidx, yfrac, floor_year, lam, yrow0,
               feas, n_out):
    """건수만 선별 덩어리 — 행 ``r`` 마다 :func:`evaluate_kernel` (``counts_only``·``prune``). ``feas[r]`` = 실현 가능,
    ``n_out[r]`` = 진입 수(버린 행은 −1)."""
    out = np.empty(6)
    for r in range(AA.shape[0]):
        evaluate_kernel(F, Fo, Iv, AA[r], BB[r], GV[r], lu_ratio, b_idx, b_e, b_sp, b_fb, b_amb, b_skip0, b_s0, b_buf,
                        fw_on, FWLO[r], XV[r], Op, H, L, C, MA20, MA50, ATR, TR, LIM, bs, bn, strat_cs, cost_tab,
                        cost_mult, pmix, t_lo, t_end, slots, slot_w, year_of, n_years, reform, month_of, ffloor,
                        yidx, yfrac, floor_year, lam, True, 3, 0, 0, 2.5, True, yrow0, out)
        feas[r] = out[3] > 0.5
        n_out[r] = np.int64(out[0])


screen_kernel = _jit(_screen_py)
