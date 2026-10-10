"""메타 세대 2 가속 커널 — 상한가 마감(LU) 판정 · 청산 규칙 A(보유 중 LU → 다음날 시가 매도) · 사건 연구 B 경로.

설계 정본: ``research/logs/sepa_explore/META_GEN2.md`` (f8151d6, 결과 보기 전 고정). 이 파일은 **계산 기계**만 둔다
(어느 쪽이 이득인지는 러너 ``research/experiments/meta_gen2_run.py`` 가 리포트한다).

**LU 판정(:func:`lu_close`):** 거래 가능일 ∧ 원 종가 ≥ 직전 유한 원 종가 × (1.30 − 0.007) — ``korea.is_limit_up_open`` 과 같은
비율 판정을 종가에. 같은 종목 연속 상한가는 첫날만 사건(:func:`lu_first` — 직전 **거래 가능일**이 LU 가 아닌 날).

**청산 규칙 A(:func:`apply_lu_exit`):** 청산 v2 커널(``exits_v2_fast.sim_book_v2``)이 낸 트레이드 경로를 뒤에서 고친다. 고칠 수 있는
이유: v2 의 결정(손절·익절·돌파 실패·추적·최대 보유)은 **가격만 보고** 남은 수량을 보지 않는다 — 수량은 결과에 곱해질 뿐이다.

- 보유 중 종가 = 진입일 ``e`` ~ 청산일 ``x`` 의 거래 가능일 ``t`` 중 ``t < x``, 또는 ``t = x`` 이면서 청산이 **종가 청산**(최대 보유·시간
  손절·데이터 끝·폐지)인 날. 그 첫 LU 종가 ``t*`` 다음 첫 거래 가능일 ``k`` 시가에 판다(시가가 상한가여도 판다 — 매수 대기가 쌓인 자리라
  매도는 체결된다). ``k`` 가 없으면(패널 끝·폐지) 그대로 둔다.
- **A (전량):** ``t*`` 까지의 매도분은 그대로, 나머지 전부 ``k`` 시가(사유 :data:`R_LU`). 다른 청산보다 우선 — ``t* = x`` 의 최대 보유 종가
  청산도 ``k`` 시가로 바뀐다.
- **A½ (절반, 트레이드당 한 번):** ``k`` 시가에 그때 남은 수량의 절반, 나머지는 원래 경로 그대로(뒤 매도분·최종 비율 × ½). 원래 최종
  청산이 ``k`` 시가 값과 같으면(돌파 실패·추적 시가 청산, 갭 손절) 바뀌는 것이 없다. ``t* = x`` 종가 청산과 겹치면 절반은 그 종가(원래
  규칙), 절반은 ``k`` 시가.
- 매도분 칸은 둘(``ev_*``)이다. 넘치면(현 청산 P1 은 익절 한 번뿐이라 생기지 않는다) 그 쌍은 고치지 않고 ``fired = −1`` 로 표시한다.

**사건 연구 B 경로(:func:`hold_paths`):** 신호일 ``d`` 종가 → ``d+1`` 시가 매수(정지면 버림, 시가 ≥ 전일 종가 × :data:`LU_RATIO` 면 상한가
체결 불가로 버림), 손절 = 진입가 × (1 − stop) (장중 저가·시가 ≤ 손절가 → ``min(시가, 손절가)``, 진입일 포함), 보유 ``h`` 거래일 = 진입일을
1일째로 ``e + h − 1`` 행 종가(그 행이 정지·하한가 잠김이면 처음 팔 수 있는 행 종가). **하한가로 잠긴 날(``meta_fast.locked_down``)은 손절도
청산도 못 한다**(메타 세대 1 정정 M4 와 같다). ``e + h − 1`` 이 패널 밖이면 NaN(미완), 그 전에 시세가 끊기면 마지막 유한 종가(사유 끝).

numba 는 선택(``uv run --with numba``). 없으면 같은 소스가 순수 파이썬으로 돈다. ``이름_py`` = 순수 파이썬, ``이름`` = JIT 판.
"""

from __future__ import annotations

import numpy as np

from research.signals import ga_fast as gf
from research.signals import meta_fast as mf

HAVE_NUMBA = gf.HAVE_NUMBA
prange = gf.prange

#: A 규칙 청산 사유(ga_fast·meta_fast 코드 뒤).
R_LU = 12
REASON_EXTRA = {R_LU: "limit_up_next_open"}
#: 종가에 일어나는 청산(그날 종가까지 보유 중이다).
CLOSE_EXITS = (gf.R_TIME, gf.R_TIME_STOP, gf.R_DATA_END, gf.R_DELISTED)
#: 상한가 판정 비율(``korea.LIMIT_UP_REL_TOL`` 0.7%p — ``sepa.LIMIT_UP_RATIO`` 와 같은 값).
LU_RATIO = 1.30 - mf.LIMIT_TOL
#: B 경로 청산 사유.
B_TIME, B_STOP, B_END = 0, 1, 2
B_REASONS = ("time", "stop", "end")


def _jit(fn, parallel: bool = False):
    return gf._jit(fn, parallel=parallel)


# ---------------------------------------------------------------------------
# 1. LU 판정 (파이썬 — 벡터)
# ---------------------------------------------------------------------------

def lu_close(raw_close: np.ndarray, TR: np.ndarray, ratio: float = LU_RATIO) -> np.ndarray:
    """상한가 마감(종목 × 날): 거래 가능 ∧ 원 종가 ≥ 직전 유한 원 종가 × ``ratio``. 직전 종가가 없거나 ≤ 0 이면 거짓."""
    raw = np.asarray(raw_close, float)
    prev = np.full_like(raw, np.nan)
    prev[:, 1:] = raw[:, :-1]
    prev = mf.pd_ffill_rows(prev)
    with np.errstate(invalid="ignore"):
        return np.asarray(TR, bool) & np.isfinite(raw) & (prev > 0) & (raw >= prev * ratio)


def lu_first(lu: np.ndarray, TR: np.ndarray) -> np.ndarray:
    """연속 상한가의 첫날: LU ∧ 직전 거래 가능일이 LU 가 아님(첫 거래 가능일의 LU 는 첫날)."""
    lu = np.asarray(lu, bool)
    tr = np.asarray(TR, bool)
    n, m = lu.shape
    idx = np.where(tr, np.arange(m)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    prev = np.full((n, m), -1, np.int64)
    prev[:, 1:] = idx[:, :-1]
    rows = np.arange(n)[:, None]
    prev_lu = np.where(prev >= 0, lu[rows, np.maximum(prev, 0)], False)
    return lu & ~prev_lu


# ---------------------------------------------------------------------------
# 2. 청산 규칙 A — v2 경로 후처리
# ---------------------------------------------------------------------------

def _apply_lu_exit_py(pc, pd_, x_abs, reason, ev_abs, ev_frac, ev_ret, fin_frac, fin_ret, Op, TR, LUC, half):
    """쌍 배열 → A(또는 ``half`` 면 A½)를 건 ``(x, reason, ev_abs, ev_frac, ev_ret, fin_frac, fin_ret, fired)``.

    ``fired`` = 0 LU 없음(또는 팔 날 없음) · 1 규칙이 경로를 바꿈 · 2 걸렸지만 값이 같음(A½ 이 원래 시가 청산과 겹침) · −1 매도분 칸 넘침(그대로 둠).
    규칙은 모듈 설명. 입력 배열은 고치지 않는다.
    """
    m = pc.shape[0]
    n = Op.shape[1]
    xo = x_abs.copy()
    ro = reason.copy()
    ea = ev_abs.copy()
    ef = ev_frac.copy()
    er = ev_ret.copy()
    ff = fin_frac.copy()
    fr = fin_ret.copy()
    fired = np.zeros(m, np.int8)
    for i in prange(m):
        x = x_abs[i]
        if x < 0:
            continue
        c = pc[i]
        e = pd_[i] + 1
        rs = reason[i]
        close_exit = rs == gf.R_TIME or rs == gf.R_TIME_STOP or rs == gf.R_DATA_END or rs == gf.R_DELISTED
        ts = -1
        for t in range(e, x + 1):
            if LUC[c, t] and (t < x or close_exit):
                ts = t
                break
        if ts < 0:
            continue
        k = -1
        for t in range(ts + 1, n):
            if TR[c, t]:
                k = t
                break
        if k < 0:
            continue
        pk = Op[c, k] / Op[c, e] - 1.0
        # t* 까지의 매도분(시간 순)과 그 뒤 매도분
        na = np.full(4, -1, np.int64)
        nf = np.zeros(4)
        nr = np.full(4, np.nan)
        nb = 0
        rem = 1.0
        for s in range(2):
            a = ev_abs[i, s]
            if a >= 0 and a <= ts:
                na[nb] = a
                nf[nb] = ev_frac[i, s]
                nr[nb] = ev_ret[i, s]
                nb += 1
                rem -= ev_frac[i, s]
        if not half:
            x_new, r_new, f_fin, r_fin = k, R_LU, rem, pk
            fl = 1
        else:
            q = 0.5 * rem
            if ts == x:
                na[nb] = x
                nf[nb] = q
                nr[nb] = fin_ret[i]
                nb += 1
                x_new, r_new, f_fin, r_fin = k, R_LU, q, pk
                fl = 1
            elif x == k and fin_ret[i] == pk:
                fired[i] = 2
                continue
            else:
                na[nb] = k
                nf[nb] = q
                nr[nb] = pk
                nb += 1
                for s in range(2):
                    a = ev_abs[i, s]
                    if a > ts:
                        na[nb] = a
                        nf[nb] = 0.5 * ev_frac[i, s]
                        nr[nb] = ev_ret[i, s]
                        nb += 1
                x_new, r_new, f_fin, r_fin = x, rs, 0.5 * fin_frac[i], fin_ret[i]
                fl = 1
        if nb > 2:
            fired[i] = -1
            continue
        for s in range(2):
            if s < nb:
                ea[i, s] = na[s]
                ef[i, s] = nf[s]
                er[i, s] = nr[s]
            else:
                ea[i, s] = -1
                ef[i, s] = 0.0
                er[i, s] = np.nan
        xo[i] = x_new
        ro[i] = r_new
        ff[i] = f_fin
        fr[i] = r_fin
        fired[i] = fl
    return xo, ro, ea, ef, er, ff, fr, fired


# ---------------------------------------------------------------------------
# 3. 사건 연구 B — 다음날 시가 매수 · h 거래일 종가 · 손절
# ---------------------------------------------------------------------------

def _hold_paths_py(pc, pd_, Op, H, L, C, TR, LD, hs, stop, lu_ratio):
    """쌍 배열 → ``(fill, ret[m × len(hs)], x_abs[m × len(hs)], why[m × len(hs)], lu_open)``. 규칙은 모듈 설명.

    ``fill`` = 진입 체결(다음날 거래 가능 ∧ 시가 유한 > 0 ∧ 상한가 시가 아님), ``lu_open`` = 상한가 시가로 버린 쌍. 미체결·미완은 ret NaN.
    """
    m = pc.shape[0]
    n = Op.shape[1]
    nh = hs.shape[0]
    fill = np.zeros(m, np.bool_)
    luo = np.zeros(m, np.bool_)
    ret = np.full((m, nh), np.nan)
    xab = np.full((m, nh), -1, np.int64)
    why = np.full((m, nh), -1, np.int8)
    for i in prange(m):
        c = pc[i]
        d = pd_[i]
        e = d + 1
        if e >= n or not TR[c, e]:
            continue
        ent = Op[c, e]
        if not (np.isfinite(ent) and ent > 0):
            continue
        prev = C[c, d]
        if np.isfinite(prev) and prev > 0 and ent >= prev * lu_ratio:
            luo[i] = True
            continue
        fill[i] = True
        stp = ent * (1.0 - stop)
        for q in range(nh):
            due = e + hs[q] - 1
            if due >= n:
                continue
            px = np.nan
            xr = -1
            wy = -1
            last = -1
            for t in range(e, n):
                if not TR[c, t]:
                    continue
                last = t
                if LD[c, t]:
                    continue
                o = Op[c, t]
                if L[c, t] <= stp or o <= stp:
                    px = min(o, stp)
                    xr = t
                    wy = B_STOP
                    break
                if t >= due:
                    px = C[c, t]
                    xr = t
                    wy = B_TIME
                    break
            if xr < 0:
                if last < 0:
                    continue
                # 팔 수 있는 날이 다시 오지 않음 — 마지막 거래 가능일 종가(정지·폐지·잠김이 패널 끝까지)
                px = C[c, last]
                xr = last
                wy = B_END
            ret[i, q] = px / ent - 1.0
            xab[i, q] = xr
            why[i, q] = wy
    return fill, ret, xab, why, luo


# --- 내보내기: 이름_py = 순수 파이썬, 이름 = numba(있으면) ---
apply_lu_exit = _jit(_apply_lu_exit_py, parallel=True)
hold_paths = _jit(_hold_paths_py, parallel=True)


def lu_exit(pc, pd_, ex: dict, Op, TR, LUC, *, half: bool, py: bool = False) -> dict:
    """청산 캐시 dict(``x, reason, ea, ef, er, ff, fr``) → A(``half`` 면 A½)를 건 같은 모양 dict + ``fired``."""
    f = _apply_lu_exit_py if py else apply_lu_exit
    x, r, ea, ef, er, ff, fr, fired = f(np.asarray(pc, np.int64), np.asarray(pd_, np.int64), ex["x"], ex["reason"], ex["ea"],
                                        ex["ef"], ex["er"], ex["ff"], ex["fr"], Op, TR, LUC, bool(half))
    return {"x": x, "reason": r, "ea": ea, "ef": ef, "er": er, "ff": ff, "fr": fr, "fired": fired}


def b_paths(pc, pd_, Op, H, L, C, TR, LD, hs, stop: float, *, py: bool = False):
    """:func:`hold_paths` 감싸기(지평 배열·비율 고정)."""
    f = _hold_paths_py if py else hold_paths
    return f(np.asarray(pc, np.int64), np.asarray(pd_, np.int64), Op, H, L, C, TR, LD, np.asarray(hs, np.int64), float(stop),
             float(LU_RATIO))
