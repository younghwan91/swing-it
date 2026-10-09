"""미너비니 원전 청산 프로토콜 — VCP 수축 피처 + 기술적 손절·2R 절반 익절·본전·MA 추적·시간 손절 시뮬레이터.

사전등록 정본은 ``research/logs/sepa_explore/MINERVINI_EXITS.md`` (36b7601, 결과 보기 전 확정)이다. 이 파일은 그
프로토콜을 **그대로** 옮긴 것이고 수치는 탐색하지 않는다 — 인자 기본값이 사전등록 값이고, 민감도 칸(고정 8% 손절·
익절 3R·추적 MA20·시간 손절 끔)만 인자로 바꾼다.

체결 규약은 :mod:`research.signals.sepa` 와 같다: 신호일 ``d`` 종가까지로 판단, ``d+1`` 시가 진입(그날 정지·시세
없음이면 버림), 추격 금지(시가 > 피벗 × 1.05), 상한가 시가 버림(:data:`sepa.LIMIT_UP_RATIO`), 정지일(거래량 0)은
진입도 청산도 못 함, 데이터가 끊기면 마지막 거래 가능일 종가(``delisted`` / ``data_end``), 같은 종목 보유 중 재진입
없음. 진입일 = 1일째(오프셋 0), ``hold_days`` = 청산 행 − 신호 행.

**사전등록 문구 → 구현 해석(모호한 곳은 전부 여기 적는다):**

1. VCP 깊이 = ``max(high[t−10..t−1]) / min(low[t−10..t−1]) − 1`` (돌파일 ``t`` 제외 10거래일).
2. VCP 거래량 = ``mean(vol[t−10..t−1]) / mean(vol[t−60..t−11])`` — 문구 "그 10일 평균 거래량 < **직전 50일** 평균의
   0.8" 의 "직전" 을 **그 10일 창의 직전** 50일로 읽는다(창이 겹치지 않아야 "수축 vs 그 전" 비교가 된다).
   정지일 거래량 0 은 0 으로 평균에 들어간다(:func:`swing_it.features.minervini.pivot_breakout` 와 같다).
3. 수축 저점 = ``min(low[t−10..t])`` — 사전등록 "t−10..t 저가 최소" 그대로 **돌파일 포함** 11거래일. (VCP 깊이 창과
   하루 다르다 — 문구가 다르게 적혀 있어 그대로 따른다.)
4. 손절폭 ``d = clip(1 − 수축저점×0.99 / 진입가, 3%, 10%)``, 손절가는 **자른 뒤 다시** ``진입가 × (1 − d)`` 로 둔다
   (자르기가 손절가를 실제로 옮긴다 — 하한 3% 면 기술적 지점보다 아래, 상한 10% 면 위로).
5. 진입 시가 ≤ 수축 저점이면 버린다(``skip_below_low``). 수축 저점이 정의 안 되면(이력 부족) ``skip_no_contraction``.
   고정 손절 민감도(``fixed_stop``)에서도 같은 거르기를 둔다 — 칸끼리 **진입 집합이 같아야** 손절 규칙만의 차이가 된다.
6. 익절 전(전량): 손절가 터치 → 손절(갭 하락이면 시가). 고가 ≥ 목표가(진입가 × (1 + take_r·d)) → 절반 익절(목표가,
   갭 상승이면 시가). **같은 날 둘 다 닿으면 손절 먼저**(사전등록 문구 그대로 — 갭 상승 시가로 목표를 넘긴 날도 저가가
   손절가에 닿았으면 손절). 익절 전에는 MA 추적 청산이 **없다**(사전등록 2번은 "나머지 절반" 규칙).
7. 본전 손절은 익절 **다음 거래 가능일**부터 건다 — 일봉으로는 익절일 안에서 고가·저가 순서를 모르고, 진입일엔
   시가 = 진입가라 저가 ≤ 진입가가 항상 참이어서 같은 날 검사를 하면 진입일 익절이 전부 본전으로 퇴화한다.
   예외: 갭 상승 시가 익절일은 시가가 먼저인 게 확정이라 그날 저가 ≤ 진입가면 그날 본전(진입가)으로 청산한다.
   이후 날은 저가 ≤ 진입가 → ``min(시가, 진입가)``.
8. 나머지 절반 추적: 익절일 포함 이후 거래 가능일 종가 < 추적 MA → 다음 거래 가능일 시가(``trail``). 다음날 시가가
   본전 아래여도 추적 청산이 먼저 결정됐으므로 사유는 ``trail``(체결가는 같은 시가).
9. 시간 손절: ``time_stop_days`` 일째(정지면 다음 거래 가능일) 종가에, 그날까지(그날 포함) 익절이 안 났고 손절도 안
   났으며 종가 ≤ 진입가면 그 종가에 전량 청산. 한 번만 본다(20일 이후 다시 보지 않는다).
10. 최대 보유 ``max_hold`` 일째 종가(정지면 다음 거래 가능일 종가) — 남은 포지션 전부. 추적 신호는 sepa 와 같이
    1 ~ max_hold−1 일째 종가에서만.
11. 트레이드 수익 = 두 절반 수익의 평균. 익절이 안 났으면 두 절반이 같은 가격에 함께 나가 ``ret_half1 = ret_half2``.
    ``R = ret / d``.
13. (선택, ``take_r = inf`` — 익절 없음, GA2 청산 유전자) 추적 MA 가 있으면 **전량 추적**: 1 ~ max_hold−1 일째 종가 < MA → 다음
    거래 가능일 시가 전량(``trail``; ``ga_fast._sim_one_py`` 의 ``j_t1`` 과 같은 식). 같은 날 손절보다 시가 청산이 먼저(손절은 엄격히
    이른 날만). 시간 손절은 추적 청산일보다 앞설 때만, 돌파 실패와 겹치면 청산일이 이른 쪽(같은 날이면 돌파 실패). ``trail_ma``
    패널이 전부 NaN 이면 추적 없음.
12. (선택, ``failed_breakout=True`` — 라이브 설계 §6 2026-10-10, 기본 **False** 로 옛 결과 재현) 돌파 실패: 익절 전
    거래 가능일 ``k`` (1 ~ max_hold−1 일째) 종가 < 피벗(사건의 ``pivot`` = 직전 50일 최고 고가) → **다음 거래 가능일 시가**
    전량(``failed_breakout``). 그날 장중 손절·익절이 먼저 났으면 그쪽. 시간 손절과 같은 날 결정이면 시간 손절(즉시 체결)이
    먼저. 다음날 시가가 손절·목표를 건드려도 시가 청산이 먼저(추적 청산과 같은 규칙 8). 다음 거래 가능일이 지평 밖이면 종료 청산.

14. (선택, **청산 v2** — ``MINERVINI_EXITS_V2.md`` 2e47f59, 키워드 기본값 = 옛 동작) 본전 이동 ``be_r``(고가 ≥ 진입가 × (1 + be_r·d)
    인 날의 **다음 거래 가능일부터** 손절선 = 진입가 × 1.005), 부분 익절 비율 ``take_frac``(남은 수량 대비), 강세 매도
    ``strength_ret``·``strength_ma_ext``(종가 ≥ 직전 행 종가 × (1+x) 또는 종가 ≥ MA50 × (1+y) → 다음 거래 가능일 시가에 남은 수량의
    절반, 한 번), 피벗 허용폭 ``fb_tol``(종가 < 피벗 × (1 − tol)), 연속 이탈일 ``fb_days``(정지일은 세지도 끊지도 않는다).
    하루 순서·같은 날 우선순위(손절 > 본전 손절 > 부분 익절 > 강세 매도 > 돌파 실패 > 추적)는
    :mod:`research.signals.exits_v2_fast` 모듈 설명 1~6 과 같다. 하나라도 기본값이 아니면 하루씩 도는 :func:`_simulate_one_v2`
    를 쓴다(``exits_v2=True`` 면 기본값에서도 — 옛 경로와 같은 결과인지 테스트용). 트레이드 수익 = Σ 판 비율 × 매도분 수익(시간 순),
    ``ret_half1`` = 부분 익절분 수익(없으면 마지막 청산분), ``ret_half2`` = 마지막 청산분 수익, ``R = ret / d``.

비용(``korea.cost_rt``)은 여기서 빼지 않는다 — 러너 몫(사전등록: 분할 청산도 총 비용 같음).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.signals.sepa import LIMIT_UP_RATIO, SKIP_COLS, TRADE_COLS, _Book
from swing_it.features.minervini import pivot_level

# --- 사전등록 수치(MINERVINI_EXITS.md — 탐색하지 않는다) ---
PREREG_MINERVINI = {
    "vcp_window": 10,        # 수축 창(돌파 직전 10거래일)
    "vcp_vol_window": 50,    # 거래량 비교 창(수축 창 직전 50거래일)
    "vcp_max_depth": 0.12,   # 수축 깊이 ≤ 12%
    "vcp_max_vol_ratio": 0.8,  # 수축 거래량 < 0.8 × 직전 50일
    "stop_mult": 0.99,       # 손절가 = 수축 저점 × 0.99
    "stop_floor": 0.03,      # 손절폭 하한(갭 노이즈)
    "stop_cap": 0.10,        # 손절폭 상한(원전 절대 한도)
    "take_r": 2.0,           # 절반 익절 = 진입가 × (1 + 2d)
    "time_stop_days": 20,    # 20거래일째 종가 시간 손절
    "max_hold": 120,         # 최대 보유(2분기)
    "chase": 0.05,
    "limit_up_skip": True,
}

MX_EXTRA_COLS = ["d", "stop_px", "target_px", "contraction_low", "partial_hit", "partial_day", "partial_px",
                 "ret_half1", "ret_half2", "R"]
MX_TRADE_COLS = TRADE_COLS + MX_EXTRA_COLS
#: 청산 v2 경로에서만 붙는 열(강세 매도 체결일·가격, 본전 이동 발동 여부, 마지막 청산 비율).
MX_V2_COLS = ["strength_hit", "strength_day", "strength_px", "be_moved", "final_frac"]
#: ``be_r`` 본전선 = 진입가 × (1 + BE_COST) — ``exits_v2_fast.BE_COST``·``swing_it.live.config.BE_COST`` 와 같은 값(테스트).
BE_COST = 0.005
#: 청산 v2 키워드 기본값(= 옛 동작).
V2_DEFAULTS = {"take_frac": 0.5, "be_r": np.inf, "strength_ret": None, "strength_ma_ext": None, "fb_tol": 0.0,
               "fb_days": 1}


# --- VCP 수축 피처 ---

def vcp_contraction(high: pd.DataFrame, low: pd.DataFrame, volume: pd.DataFrame, *, window: int = 10,
                    vol_window: int = 50) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """VCP 수축 → ``(깊이 패널, 거래량 비 패널)`` (날짜×종목, ``t`` 행 = ``t−1`` 종가까지 — 돌파일 자신은 안 들어간다).

    깊이 = ``max(high[t−w..t−1]) / min(low[t−w..t−1]) − 1``.
    거래량 비 = ``mean(vol[t−w..t−1]) / mean(vol[t−w−v..t−w−1])`` (수축 창 **직전** ``v`` 일, 겹치지 않음).
    창이 꽉 차야 값이 있다. 분모 평균이 0(전부 정지)이면 NaN.
    """
    hi = high.shift(1).rolling(window, min_periods=window).max()
    lo = low.shift(1).rolling(window, min_periods=window).min()
    depth = hi / lo.where(lo > 0) - 1.0
    v_recent = volume.shift(1).rolling(window, min_periods=window).mean()
    v_prior = volume.shift(window + 1).rolling(vol_window, min_periods=vol_window).mean()
    ratio = v_recent / v_prior.where(v_prior > 0)
    return depth, ratio


def vcp_ok(depth: pd.DataFrame, ratio: pd.DataFrame, *, max_depth: float = 0.12,
           max_vol_ratio: float = 0.8) -> pd.DataFrame:
    """VCP 문 = 깊이 ≤ 12% ∧ 거래량 비 < 0.8. NaN 은 거짓."""
    return (depth <= max_depth) & (ratio < max_vol_ratio)


def contraction_low_level(low: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """수축 저점 = ``min(low[t−w..t])`` — 돌파일 ``t`` **포함** ``w+1`` 거래일(사전등록 "t−10..t 저가 최소")."""
    return low.rolling(window + 1, min_periods=window + 1).min()


# --- 시뮬레이터 ---

def _first(mask: np.ndarray) -> "int | None":
    if mask.size == 0:
        return None
    i = int(mask.argmax())
    return i if mask[i] else None


def _first_from(tr: np.ndarray, k: int, hi: int) -> "int | None":
    """오프셋 ``k`` 이상 ``hi`` 이하에서 첫 거래 가능 오프셋."""
    if k > hi:
        return None
    j = _first(tr[k:hi + 1])
    return None if j is None else k + j


def _simulate_one(b: _Book, ci: int, d: int, pivot: float, c_low: float, *, stop_floor: float, stop_cap: float,
                  stop_mult: float, fixed_stop: "float | None", take_r: float, time_stop_days: "int | None",
                  max_hold: int, chase: "float | None", limit_up_skip: bool, failed_breakout: bool = False) -> dict:
    """한 사건 → 결과 dict. 건너뛰면 ``{"skip": 사유}``. 규칙은 모듈 docstring 1~11."""
    n = len(b.days)
    e = d + 1
    if e >= n:
        return {"skip": "skip_no_next_day"}
    tr = b.tradable[ci, e:]
    if not tr[0]:
        return {"skip": "skip_no_open"}
    op, hi, lw, cl, ma = b.O[ci, e:], b.H[ci, e:], b.L[ci, e:], b.C[ci, e:], b.MA[ci, e:]
    entry = op[0]
    if chase is not None and np.isfinite(pivot) and entry > pivot * (1.0 + chase):
        return {"skip": "skip_chase"}
    prev = b.C[ci, d]
    if limit_up_skip and np.isfinite(prev) and prev > 0 and entry >= prev * LIMIT_UP_RATIO:
        return {"skip": "skip_limit_up"}
    if not np.isfinite(c_low):
        return {"skip": "skip_no_contraction"}
    if entry <= c_low:
        return {"skip": "skip_below_low"}

    if fixed_stop is None:
        dd = float(np.clip(1.0 - c_low * stop_mult / entry, stop_floor, stop_cap))
    else:
        dd = float(fixed_stop)
    stop_px = entry * (1.0 - dd)
    tgt = entry * (1.0 + take_r * dd)

    # 시간 청산 행(max_hold 일째 이후 첫 거래 가능일) — sepa 와 같다.
    L = len(tr)
    k_last = max_hold - 1
    j_time = _first_from(tr, k_last, L - 1)
    horizon = j_time if j_time is not None else L - 1
    w = slice(0, horizon + 1)
    w_tr = tr[w]
    last_tr = int(np.flatnonzero(tr[: horizon + 1])[-1])

    def end_exit() -> "tuple[str, int, float]":
        if j_time is not None:
            return "time", j_time, cl[j_time]
        reason = "data_end" if b.present[ci, n - 1] else "delisted"
        return reason, last_tr, cl[last_tr]

    # --- 1단계: 전량 ---
    partial_on = np.isfinite(take_r)
    j_s = _first(w_tr & ((lw[w] <= stop_px) | (op[w] <= stop_px)))
    j_p = _first(w_tr & ((hi[w] >= tgt) | (op[w] >= tgt))) if partial_on else None
    if j_s is not None and j_p is not None and j_s <= j_p:
        j_p = None                                   # 같은 날이면 손절 먼저
    k_last_sig = min(k_last, horizon + 1)
    j_t1 = None                                      # 규칙 13: 익절 없음 ∧ 추적 → 전량 추적
    if not partial_on:
        with np.errstate(invalid="ignore"):
            sig1 = tr[:k_last_sig] & (cl[:k_last_sig] < ma[:k_last_sig])
        k1 = _first(sig1)
        if k1 is not None:
            j_t1 = _first_from(tr, k1 + 1, horizon)
    j_ts = None
    if time_stop_days is not None:
        cand = _first_from(tr, time_stop_days - 1, horizon)
        if (cand is not None and (j_s is None or cand < j_s) and (j_p is None or cand < j_p)
                and (j_t1 is None or cand < j_t1) and cl[cand] <= entry):
            j_ts = cand

    # 돌파 실패(규칙 12): 익절·손절 전 종가 < 피벗 인 첫 거래 가능일 k → 다음 거래 가능일 시가. 신호는 1 ~ max_hold−1 일째.
    j_fb, j_fb_sig = None, None
    if failed_breakout and np.isfinite(pivot):
        sig_n1 = min(k_last, horizon + 1)
        with np.errstate(invalid="ignore"):
            fb = tr[:sig_n1] & (cl[:sig_n1] < pivot)
        if j_s is not None:
            fb[j_s:] = False                              # 그날 손절이 먼저(장중)
        if j_p is not None:
            fb[j_p:] = False                              # 그날 익절이 먼저(장중)
        if j_ts is not None:
            fb[j_ts:] = False                             # 시간 손절(종가 즉시)이 먼저
        k = _first(fb)
        if k is not None:
            j_fb_sig = k
            j_fb = _first_from(tr, k + 1, horizon)

    out = {"entry": entry, "d": dd, "stop_px": stop_px, "target_px": tgt, "partial_hit": False,
           "partial_j": None, "partial_px": np.nan}
    if j_fb_sig is not None:
        if j_t1 is not None and (j_fb is None or j_t1 < j_fb):
            reason, x, px = "trail", j_t1, op[j_t1]
        elif j_fb is not None:
            reason, x, px = "failed_breakout", j_fb, op[j_fb]
        else:
            reason, x, px = end_exit()
    elif j_ts is not None:
        reason, x, px = "time_stop", j_ts, cl[j_ts]
    elif j_p is None:
        if j_s is not None and (j_t1 is None or j_s < j_t1):
            reason, x, px = "stop", j_s, min(op[j_s], stop_px)
        elif j_t1 is not None:
            reason, x, px = "trail", j_t1, op[j_t1]
        else:
            reason, x, px = end_exit()
    else:
        reason = None
    if reason is not None:
        return {**out, "reason": reason, "x": e + x, "px1": px, "px2": px}

    # --- 2단계: 절반 익절 후 나머지 절반 ---
    gap_up = op[j_p] >= tgt
    px1 = op[j_p] if gap_up else tgt
    out.update(partial_hit=True, partial_j=e + j_p, partial_px=px1)

    j_b, b_px = None, np.nan
    if gap_up and lw[j_p] <= entry:                  # 시가가 먼저 → 그날 본전 체결이 확정
        j_b, b_px = j_p, entry
    else:
        s = slice(j_p + 1, horizon + 1)
        jb = _first(tr[s] & ((lw[s] <= entry) | (op[s] <= entry)))
        if jb is not None:
            j_b = j_p + 1 + jb
            b_px = min(op[j_b], entry)

    j_t = None
    sig_n = min(k_last, horizon + 1)                 # 추적 신호는 1 ~ max_hold−1 일째 종가
    if j_p < sig_n and not (j_b == j_p):
        with np.errstate(invalid="ignore"):
            sig = tr[j_p:sig_n] & (cl[j_p:sig_n] < ma[j_p:sig_n])
        js = _first(sig)
        if js is not None:
            j_t = _first_from(tr, j_p + js + 1, horizon)

    if j_t is not None and (j_b is None or j_t <= j_b):
        reason, x, px2 = "trail", j_t, op[j_t]
    elif j_b is not None:
        reason, x, px2 = "breakeven", j_b, b_px
    else:
        reason, x, px2 = end_exit()
    return {**out, "reason": reason, "x": e + x, "px1": px1, "px2": px2}


def _simulate_one_v2(b: _Book, ci: int, d: int, pivot: float, c_low: float, *, stop_floor: float, stop_cap: float,
                     stop_mult: float, fixed_stop: "float | None", take_r: float, time_stop_days: "int | None",
                     max_hold: int, chase: "float | None", limit_up_skip: bool, failed_breakout: bool = False,
                     take_frac: float = 0.5, be_r: float = np.inf, strength_ret: "float | None" = None,
                     strength_ma_ext: "float | None" = None, fb_tol: float = 0.0, fb_days: int = 1) -> dict:
    """청산 v2(규칙 14) — 한 사건을 하루씩 상태 전이로. 건너뛰기 규칙은 :func:`_simulate_one` 과 같다.

    반환: ``{"skip": 사유}`` 또는 ``entry, d, stop_px, target_px, lots(마지막 청산 전 매도분 [(행, 최초 대비 비율, 가격, 종류)]),
    reason, x(마지막 청산 행), fin_px, fin_frac, be_moved``.
    """
    n = len(b.days)
    e = d + 1
    if e >= n:
        return {"skip": "skip_no_next_day"}
    tr = b.tradable[ci, e:]
    if not tr[0]:
        return {"skip": "skip_no_open"}
    op, hi, lw, cl, ma = b.O[ci, e:], b.H[ci, e:], b.L[ci, e:], b.C[ci, e:], b.MA[ci, e:]
    m50 = b.SMA50[ci, e:] if getattr(b, "SMA50", None) is not None else np.full(len(tr), np.nan)
    prev_cl = b.C[ci, d:n - 1]                        # 오프셋 j 의 직전 행 종가
    entry = op[0]
    if chase is not None and np.isfinite(pivot) and entry > pivot * (1.0 + chase):
        return {"skip": "skip_chase"}
    prev = b.C[ci, d]
    if limit_up_skip and np.isfinite(prev) and prev > 0 and entry >= prev * LIMIT_UP_RATIO:
        return {"skip": "skip_limit_up"}
    if not np.isfinite(c_low):
        return {"skip": "skip_no_contraction"}
    if entry <= c_low:
        return {"skip": "skip_below_low"}
    if fixed_stop is None:
        dd = float(np.clip(1.0 - c_low * stop_mult / entry, stop_floor, stop_cap))
    else:
        dd = float(fixed_stop)
    stop_px = entry * (1.0 - dd)
    tgt = entry * (1.0 + take_r * dd)
    use_partial = bool(np.isfinite(take_r))
    use_fb = bool(failed_breakout and np.isfinite(pivot))
    fb_line = pivot * (1.0 - fb_tol) if use_fb else np.nan
    be_line = entry * (1.0 + be_r * dd) if np.isfinite(be_r) else np.inf
    be_stop = entry * (1.0 + BE_COST)

    L = len(tr)
    k_last = max_hold - 1
    j_time = _first_from(tr, k_last, L - 1)
    horizon = j_time if j_time is not None else L - 1
    sig_end = min(k_last, horizon + 1)                # 다음날 시가 결정은 1 ~ max_hold−1 일째 종가에서만

    st = {"rem": 1.0, "lots": [], "partial": False, "be_from": None, "be_trig": None, "ss_used": False, "fb_run": 0,
          "ts_seen": False, "pending": None, "last": 0}

    def done(reason: str, j: int, px: float) -> dict:
        return {"entry": entry, "d": dd, "stop_px": stop_px, "target_px": tgt, "lots": st["lots"], "reason": reason,
                "x": e + j, "fin_px": px, "fin_frac": st["rem"], "be_moved": st["be_trig"] is not None}

    def sell(j: int, share: float, px: float, kind: str) -> None:
        q = st["rem"] * share
        st["rem"] = st["rem"] - q
        st["lots"].append((e + j, q, px, kind))

    for j in range(horizon + 1):
        if not tr[j]:
            continue
        st["last"] = j
        # (1) 전날 결정 → 시가
        if st["pending"] in ("failed_breakout", "trail"):
            return done(st["pending"], j, op[j])
        if st["pending"] == "strength":
            sell(j, 0.5, op[j], "strength")
            st["pending"] = None
        # (2) 손절선: 기술적 → 익절 뒤 본전(진입가) → be_r 본전(진입가 × 1.005)
        line, why = stop_px, "stop"
        if st["be_from"] is not None and j >= st["be_from"]:
            line, why = entry, "breakeven"
        if st["be_trig"] is not None and j > st["be_trig"]:
            line, why = be_stop, "breakeven"
        if lw[j] <= line or op[j] <= line:
            return done(why, j, min(op[j], line))
        # (3) 부분 익절
        if use_partial and not st["partial"] and (hi[j] >= tgt or op[j] >= tgt):
            gap = op[j] >= tgt
            sell(j, take_frac, op[j] if gap else tgt, "partial")
            st["partial"], st["be_from"] = True, j + 1
            if gap and lw[j] <= entry:
                return done("breakeven", j, entry)
        # (4) be_r 발동
        if st["be_trig"] is None and (hi[j] >= be_line or op[j] >= be_line):
            st["be_trig"] = j
        # (5) 종가
        if time_stop_days is not None and not st["partial"] and not st["ts_seen"] and j >= time_stop_days - 1:
            st["ts_seen"] = True
            if cl[j] <= entry:
                return done("time_stop", j, cl[j])
        if j_time is not None and j == j_time:
            return done("time", j, cl[j])
        if j >= sig_end:
            continue
        strong = False
        if not st["ss_used"]:
            pc_ = prev_cl[j]
            if strength_ret is not None and np.isfinite(pc_) and cl[j] >= pc_ * (1.0 + strength_ret):
                strong = True
            if strength_ma_ext is not None and np.isfinite(m50[j]) and cl[j] >= m50[j] * (1.0 + strength_ma_ext):
                strong = True
        failed = False
        if use_fb and not st["partial"]:
            st["fb_run"] = st["fb_run"] + 1 if cl[j] < fb_line else 0
            failed = st["fb_run"] >= fb_days
        trail = bool((st["partial"] or not use_partial) and np.isfinite(ma[j]) and cl[j] < ma[j])
        if strong:
            st["pending"], st["ss_used"] = "strength", True
        elif failed:
            st["pending"] = "failed_breakout"
        elif trail:
            st["pending"] = "trail"
    reason = "data_end" if b.present[ci, n - 1] else "delisted"
    return done(reason, st["last"], cl[st["last"]])


def _run(b: _Book, ev: pd.DataFrame, *, no_overlap: bool, rules: dict, v2: bool = False
         ) -> "tuple[pd.DataFrame, pd.DataFrame]":
    col_of = pd.Series(np.arange(len(b.codes)), index=b.codes)
    ev = ev[ev["code"].isin(b.codes) & ev["event_day"].isin(b.days)].copy()
    ev["sim_d"] = b.days.get_indexer(ev["event_day"])
    ev["sim_c"] = col_of.reindex(ev["code"]).to_numpy()
    ev = ev.drop_duplicates(["code", "event_day"]).sort_values(["sim_d", "code"])
    busy_until: dict[int, int] = {}
    trades, skips = [], []
    skip_extra = ("sim_d", "sim_c", "day_idx", "col_idx", "contraction_low")
    extra = [x for x in ev.columns if x not in MX_TRADE_COLS + MX_V2_COLS and x not in skip_extra]
    for row in ev.to_dict("records"):
        ci, d = int(row["sim_c"]), int(row["sim_d"])
        base = {"code": row["code"], "event_day": row["event_day"], "market": row.get("market", np.nan),
                "pivot": row.get("pivot", np.nan)}
        if no_overlap and d < busy_until.get(ci, -1):
            skips.append({**base, "skip_reason": "skip_overlap"})
            continue
        c_low = float(row["contraction_low"])
        r = (_simulate_one_v2 if v2 else _simulate_one)(b, ci, d, float(base["pivot"]), c_low, **rules)
        if "skip" in r:
            skips.append({**base, "skip_reason": r["skip"]})
            continue
        x = r["x"]
        if v2:
            busy_until[ci] = x
            trades.append({**base, **_v2_row(b, d, x, c_low, r), **{k: row[k] for k in extra}})
            continue
        busy_until[ci] = x
        ent = r["entry"]
        h1, h2 = r["px1"] / ent - 1.0, r["px2"] / ent - 1.0
        ret = 0.5 * (h1 + h2)
        trades.append({**base, "entry_day": b.days[d + 1], "exit_day": b.days[x], "entry_px": ent,
                       "exit_px": r["px2"], "ret": ret, "exit_reason": r["reason"], "hold_days": x - d,
                       "d": r["d"], "stop_px": r["stop_px"], "target_px": r["target_px"], "contraction_low": c_low,
                       "partial_hit": r["partial_hit"],
                       "partial_day": b.days[r["partial_j"]] if r["partial_j"] is not None else pd.NaT,
                       "partial_px": r["partial_px"], "ret_half1": h1, "ret_half2": h2, "R": ret / r["d"],
                       **{k: row[k] for k in extra}})
    cols = MX_TRADE_COLS + (MX_V2_COLS if v2 else []) + extra
    t = pd.DataFrame(trades, columns=cols) if trades else pd.DataFrame(columns=cols)
    s = pd.DataFrame(skips, columns=SKIP_COLS) if skips else pd.DataFrame(columns=SKIP_COLS)
    return t, s


def _v2_row(b: _Book, d: int, x: int, c_low: float, r: dict) -> dict:
    """v2 결과 → 트레이드 행. 수익 = Σ 비율 × 매도분 수익(시간 순, 마지막 청산분 포함) — ``exits_v2_fast`` 와 같은 순서."""
    ent = r["entry"]
    ret = 0.0
    part = None
    strong = None
    for j_abs, q, px, kind in r["lots"]:
        ret += q * (px / ent - 1.0)
        if kind == "partial":
            part = (j_abs, px)
        else:
            strong = (j_abs, px)
    h2 = r["fin_px"] / ent - 1.0
    ret += r["fin_frac"] * h2
    h1 = part[1] / ent - 1.0 if part is not None else h2
    return {"entry_day": b.days[d + 1], "exit_day": b.days[x], "entry_px": ent, "exit_px": r["fin_px"], "ret": ret,
            "exit_reason": r["reason"], "hold_days": x - d, "d": r["d"], "stop_px": r["stop_px"],
            "target_px": r["target_px"], "contraction_low": c_low, "partial_hit": part is not None,
            "partial_day": b.days[part[0]] if part is not None else pd.NaT,
            "partial_px": part[1] if part is not None else np.nan, "ret_half1": h1, "ret_half2": h2,
            "R": ret / r["d"], "strength_hit": strong is not None,
            "strength_day": b.days[strong[0]] if strong is not None else pd.NaT,
            "strength_px": strong[1] if strong is not None else np.nan, "be_moved": r["be_moved"],
            "final_frac": r["fin_frac"]}


def _pick(panel: pd.DataFrame, like: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    """사건 (event_day, code) 자리의 패널 값. 패널 밖은 NaN."""
    arr = panel.reindex(index=like.index, columns=like.columns).to_numpy(float)
    di = like.index.get_indexer(ev["event_day"])
    ci = like.columns.get_indexer(ev["code"])
    ok = (di >= 0) & (ci >= 0)
    return np.where(ok, arr[np.where(ok, di, 0), np.where(ok, ci, 0)], np.nan)


def simulate_minervini(o: pd.DataFrame, h: pd.DataFrame, lo: pd.DataFrame, c: pd.DataFrame, trail_ma: pd.DataFrame,
                       events: pd.DataFrame, *, volume: "pd.DataFrame | None" = None,
                       contraction_low: "pd.DataFrame | None" = None, window: int = 10,
                       stop_floor: float = 0.03, stop_cap: float = 0.10, stop_mult: float = 0.99,
                       fixed_stop: "float | None" = None, take_r: float = 2.0,
                       time_stop_days: "int | None" = 20, max_hold: int = 120, chase: "float | None" = 0.05,
                       limit_up_skip: bool = True, no_overlap: bool = True, pivot_window: int = 50,
                       return_skipped: bool = False, failed_breakout: bool = False, take_frac: float = 0.5,
                       be_r: float = np.inf, strength_ret: "float | None" = None,
                       strength_ma_ext: "float | None" = None, fb_tol: float = 0.0, fb_days: int = 1,
                       strength_ma: "pd.DataFrame | None" = None, exits_v2: "bool | None" = None):
    """사건 → 미너비니 원전 청산 트레이드(총수익, 비용 전). 규칙은 모듈 docstring 1~11.

    Args:
        o, h, lo, c: 날짜×종목 분할조정 OHLC.
        trail_ma: 나머지 절반 추적 MA 패널(기본 프로토콜 = MA50, 민감도 = MA20 패널을 넘긴다).
        events: ``code, event_day`` (+ ``pivot``, ``contraction_low``, ``market``, 그 밖의 열은 트레이드 표로 따라간다).
            ``pivot`` 이 없으면 직전 ``pivot_window`` 일 최고 고가, ``contraction_low`` 가 없으면 ``contraction_low``
            패널(없으면 :func:`contraction_low_level` ``(lo, window)``)의 사건일 값.
        volume: 거래량 패널 — 거래량 0 인 날(정지)을 체결 불가로 본다. **실데이터에선 반드시 넘긴다.**
        fixed_stop: 주면 기술적 손절 대신 고정 손절폭(민감도 "손절 고정 8%" = 0.08). 수축 저점 거르기는 그대로.
        take_r: 절반 익절 배수(민감도 3R = 3.0). ``inf`` = 익절 없음(규칙 13 — 추적 MA 가 있으면 전량 추적).
        time_stop_days: ``None`` 이면 시간 손절 끔(민감도).
        failed_breakout: True 면 규칙 12(익절 전 종가 < 피벗 → 다음 거래 가능일 시가). 기본 False — 사전등록 결과 재현.
            라이브 프로토콜(설계 §6) = ``failed_breakout=True, time_stop_days=None``.
        take_frac, be_r, strength_ret, strength_ma_ext, fb_tol, fb_days: 청산 v2(규칙 14). 기본값 = 옛 동작.
            ``strength_ma_ext`` 의 MA50 은 ``strength_ma`` 패널(없으면 ``moving_average(c, 50)``).
        exits_v2: None = v2 키워드가 하나라도 기본값이 아니면 v2 경로, True = 늘 v2(옛 경로와 같은지 테스트), False = 옛 경로.
        no_overlap: 같은 종목 보유 중(신호일 종가 기준, 나머지 절반 포함) 사건을 버린다. 대조에선 끈다.
        return_skipped: True 면 ``(trades, skipped)``.

    Returns:
        :data:`sepa.TRADE_COLS` + ``d, stop_px, target_px, contraction_low, partial_hit, partial_day, partial_px,
        ret_half1, ret_half2, R`` (+ 사건 열). ``exit_reason`` ∈ {stop, breakeven, trail, time_stop, time, delisted,
        data_end} — 마지막 청산(익절이 났으면 나머지 절반)의 사유. ``exit_day``·``exit_px``·``hold_days`` 도 마지막 청산.
    """
    ev = events.copy()
    if "pivot" not in ev.columns:
        ev["pivot"] = _pick(pivot_level(h.reindex_like(c), pivot_window), c, ev)
    if "contraction_low" not in ev.columns:
        cl_panel = contraction_low if contraction_low is not None else contraction_low_level(lo.reindex_like(c), window)
        ev["contraction_low"] = _pick(cl_panel, c, ev)
    b = _Book(o, h, lo, c, trail_ma, volume)
    rules = {"stop_floor": stop_floor, "stop_cap": stop_cap, "stop_mult": stop_mult, "fixed_stop": fixed_stop,
             "take_r": take_r, "time_stop_days": time_stop_days, "max_hold": max_hold, "chase": chase,
             "limit_up_skip": limit_up_skip, "failed_breakout": failed_breakout}
    v2kw = {"take_frac": take_frac, "be_r": be_r, "strength_ret": strength_ret, "strength_ma_ext": strength_ma_ext,
            "fb_tol": fb_tol, "fb_days": fb_days}
    v2 = exits_v2 if exits_v2 is not None else any(v2kw[k] != V2_DEFAULTS[k] for k in v2kw)
    if v2:
        rules.update(v2kw)
        if strength_ma_ext is not None:
            from swing_it.features.technical import moving_average  # noqa: PLC0415

            sma = strength_ma if strength_ma is not None else moving_average(c, 50)
            b.SMA50 = np.ascontiguousarray(sma.reindex(index=c.index, columns=c.columns).to_numpy(float).T)
    t, s = _run(b, ev, no_overlap=no_overlap, rules=rules, v2=v2)
    return (t, s) if return_skipped else t


def control_minervini(o: pd.DataFrame, h: pd.DataFrame, lo: pd.DataFrame, c: pd.DataFrame, trail_ma: pd.DataFrame,
                      pairs: pd.DataFrame, *, volume: "pd.DataFrame | None" = None,
                      market: "pd.Series | None" = None, pivot_window: int = 50, window: int = 10,
                      no_overlap: bool = False, **rules):
    """임의 (종목, 신호일) 쌍에 **같은 프로토콜**을 적용 — 정합 대조용(사전등록 "대조 종목에도 같은 프로토콜").

    대조의 손절가 = **자기** 수축 저점(``min(low[t−10..t])``) × 0.99, [3%, 10%] 로 자름. 피벗 = 직전 50일 최고 고가
    (:func:`sepa.control_trades` 와 같은 정의 — 추격 금지가 같은 식으로 걸린다). 자기 저점 아래 시가면 버린다
    (사건과 같은 규칙). 대조끼리는 독립 표본이라 기본 ``no_overlap=False``. ``pairs`` 에 ``pivot``·``contraction_low``
    가 있으면 그 값을 쓴다. ``rules`` 는 :func:`simulate_minervini` 의 나머지 키워드.
    """
    ev = pairs.copy()
    if market is not None and "market" not in ev.columns:
        ev["market"] = market.reindex(ev["code"]).to_numpy()
    return simulate_minervini(o, h, lo, c, trail_ma, ev, volume=volume, pivot_window=pivot_window, window=window,
                              no_overlap=no_overlap, **rules)
