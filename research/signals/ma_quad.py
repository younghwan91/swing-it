"""q차식 롤링 외삽 — 로그 종가 패널의 마지막 N개에 최소제곱 q차식을 맞춰 t+1..t+k 를 외삽한다.

s = −N+1..0 (0 = t) 기준 설계행렬의 정규방정식은 (N, q) 로만 정해지므로 외삽 가중치 W(k×N)를 한 번 구하고
종목별 슬라이딩 창에 점곱한다. 창에 NaN 이 하나라도 있으면 그 칸은 NaN(0 으로 채우지 않는다).
numba 는 선택이다(:mod:`research.signals.vcp` 와 같은 규약) — 없으면 같은 소스가 파이썬으로 돈다.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

try:
    import numba

    def _jit(fn):
        return numba.njit(cache=False, nogil=True)(fn)

except ImportError:  # pragma: no cover

    def _jit(fn):
        return fn


def extrapolation_weights(N: int, k: int, q: int) -> np.ndarray:
    """(k, N) — 창 값 y(s=−N+1..0) 에 곱하면 s=1..k 의 q차식 외삽값."""
    if q < 0 or N < q + 1 or k < 1:
        raise ValueError(f"잘못된 (N, k, q) = ({N}, {k}, {q})")
    s = np.arange(-N + 1, 1, dtype=np.float64) / N  # 조건수 완화(공간 불변)
    h = np.arange(1, k + 1, dtype=np.float64) / N
    A = np.vander(s, q + 1, increasing=True)  # (N, q+1)
    V = np.vander(h, q + 1, increasing=True)  # (k, q+1)
    pinv = np.linalg.lstsq(A, np.eye(N), rcond=None)[0]  # (q+1, N)
    return V @ pinv


@_jit
def _roll_dot(y, W, N, out):
    n_codes, n_days = y.shape
    k = W.shape[0]
    for i in range(n_codes):
        for t in range(N - 1, n_days):
            ok = True
            for j in range(N):
                if not np.isfinite(y[i, t - N + 1 + j]):
                    ok = False
                    break
            if not ok:
                continue
            for h in range(k):
                acc = 0.0
                for j in range(N):
                    acc += W[h, j] * y[i, t - N + 1 + j]
                out[i, t, h] = acc


def poly_forecast(y: np.ndarray, N: int, k: int, q: int) -> np.ndarray:
    """(n_codes, n_days, k) float32 — ``[i, t, h-1]`` = t 까지 N개로 맞춘 q차식의 t+h 값.

    창에 비유한 값(NaN/inf)이 있거나 t < N−1 이면 NaN. q=0 은 y[i, t] 반복(N 무관, y[i, t] 가 유한하면 값).
    (N, k, q) 가 잘못이면 ValueError(q=0 포함).
    """
    if k < 1 or q < 0 or N < q + 1:
        raise ValueError(f"잘못된 (N, k, q) = ({N}, {k}, {q})")
    y = np.ascontiguousarray(y, dtype=np.float64)
    n_codes, n_days = y.shape
    if q == 0:
        out = np.repeat(y[:, :, None], k, axis=2)
        return out.astype(np.float32)
    out = np.full((n_codes, n_days, k), np.nan, dtype=np.float64)
    _roll_dot(y, np.ascontiguousarray(extrapolation_weights(N, k, q)), N, out)
    return out.astype(np.float32)


def rolling_sigma(y: np.ndarray, w: int = 60) -> np.ndarray:
    """(n_codes, n_days) float64 — u 까지(포함) w 개 로그 수익(diff)의 표본 표준편차(ddof=1), 결측 있으면 NaN."""
    y = np.asarray(y, dtype=np.float64)
    r = np.full_like(y, np.nan)
    r[:, 1:] = np.diff(y, axis=1)
    n_codes, n_days = y.shape
    out = np.full((n_codes, n_days), np.nan)
    if n_days < w + 1:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    for a in range(0, n_codes, 64):  # 종목 덩어리 — (덩어리, 일, w) 임시 배열 메모리 제한
        win = sliding_window_view(r[a : a + 64], w, axis=1)
        out[a : a + 64, w - 1 :] = win.std(axis=2, ddof=1)  # NaN 이 끼면 NaN 전파
    return out


def forecast_z(y: np.ndarray, fc: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """(n_codes, n_days, k) float32 — ``z[i,t,h-1] = (y[i,t+h] − fc[i,t,h-1]) / sigma[i,t]``.

    t+h ≥ n_days, 입력에 비유한 값, sigma ≤ 0 이면 NaN.
    """
    y = np.asarray(y, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n_codes, n_days, k = fc.shape
    z = np.full((n_codes, n_days, k), np.nan, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok_s = np.isfinite(sigma) & (sigma > 0)
        for h in range(1, k + 1):
            if h >= n_days:
                break
            err = (y[:, h:] - fc[:, : n_days - h, h - 1].astype(np.float64)) / sigma[:, : n_days - h]
            err[~ok_s[:, : n_days - h]] = np.nan
            z[:, : n_days - h, h - 1] = np.where(np.isfinite(err), err, np.nan)
    return z


def month_starts(days: pd.DatetimeIndex) -> np.ndarray:
    """각 달 첫 거래일의 행 번호."""
    ym = np.asarray(days.year) * 12 + np.asarray(days.month)
    return np.flatnonzero(np.r_[True, ym[1:] != ym[:-1]])


def month_pool(
    days: pd.DatetimeIndex,
    z: np.ndarray,
    member: np.ndarray,
    t_m: int,
    k: int,
    years: int = 3,
    min_dates: int = 120,
    return_origins: bool = False,
):
    """t_m 달의 정규화 오차 풀 (P, k) float32. 원점 날짜가 min_dates 미만이면 None.

    원점 (i, u): ``days[u] >= days[t_m] − years년``, ``u ≤ t_m − k``(u+k 가 t_m 이후면 미래 누출),
    member[i,u], z[i,u,:] 전부 유한. return_origins=True 면 ``(pool, origins)`` — origins 는 (P,2) int64 (코드 행, u).
    """
    lo_date = days[t_m] - pd.DateOffset(years=years)
    u0 = int(days.searchsorted(lo_date, side="left"))
    u1 = t_m - k
    if u1 < u0:
        return None
    zz = z[:, u0 : u1 + 1, :]
    ok = member[:, u0 : u1 + 1] & np.isfinite(zz).all(axis=2)
    if int((ok.any(axis=0)).sum()) < min_dates:
        return None
    ii, uu = np.nonzero(ok)
    pool = zz[ii, uu, :].astype(np.float32)
    if return_origins:
        return pool, np.stack([ii, uu + u0], axis=1).astype(np.int64)
    return pool


def month_pools(days, z, member, k, years: int = 3, min_dates: int = 120):
    """달마다 ``(t_m, pool)`` 을 내는 제너레이터(자격 있는 달만). 전 달 풀을 한꺼번에 들지 않는다."""
    for t_m in month_starts(days):
        pool = month_pool(days, z, member, int(t_m), k, years, min_dates)
        if pool is not None:
            yield int(t_m), pool


def draw_index(code: str, t_m_date: pd.Timestamp, q: int, N: int, k: int, P: int, M: int = 200) -> np.ndarray:
    """(M,) int64 — 풀 [0, P) 에서의 복원추출 색인. 키 (코드, 연월, q, N, k) 로 결정적."""
    key = f"{code}|{t_m_date:%Y-%m}|{q}|{N}|{k}"
    seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(seed).integers(0, P, size=M).astype(np.int64)


# ---------------------------------------------------------------- 경로 → 이평선 → 확률

_MA_WINDOWS = (5, 20, 50, 60, 120, 150, 200)
_PROB_KEYS = ("p_align", "p_up_5_20", "p_up_20_60", "p_up_60_120")


def _rolling_mean(C: np.ndarray, w: int) -> np.ndarray:
    """창 안에 NaN 이 있으면 NaN 인 단순 이동평균(pandas rolling(w, min_periods=w).mean())."""
    return pd.DataFrame(C.T).rolling(w, min_periods=w).mean().to_numpy().T


def ma_state(C: np.ndarray) -> dict:
    """종가 패널 (n_codes, n_days) → 이평선·정배열·묶음 하단·d·200일 거름·미너비니 템플릿.

    정배열 = MA5 > MA20 > MA60 > MA120(엄격). NaN 이 있으면 False. 전부 당일 t 포함.
    """
    C = np.asarray(C, dtype=np.float64)
    ma = {w: _rolling_mean(C, w) for w in _MA_WINDOWS}
    with np.errstate(invalid="ignore"):
        aligned = (ma[5] > ma[20]) & (ma[20] > ma[60]) & (ma[60] > ma[120])
        bundle_low = np.minimum(np.minimum(ma[20], ma[60]), ma[120])
        d = 1.0 - bundle_low / C
        prev200 = np.full_like(C, np.nan)
        prev200[:, 20:] = ma[200][:, :-20]
        f200 = (C > ma[200]) & (ma[200] > prev200)
        lo = pd.DataFrame(C.T).rolling(250, min_periods=250).min().to_numpy().T
        hi = pd.DataFrame(C.T).rolling(250, min_periods=250).max().to_numpy().T
        tmpl = (
            (C > ma[150])
            & (ma[150] > ma[200])
            & (ma[50] > ma[150])
            & (C > ma[50])
            & (C >= 1.30 * lo)
            & (C >= 0.75 * hi)
        )
    return {"ma": ma, "aligned": aligned, "bundle_low": bundle_low, "d": d, "f200": f200, "tmpl": tmpl}


@_jit
def _event_kernel(C, fc_log, sigma, pool, idx, ci, s, k, out):
    n_ev, M = idx.shape
    n_days = C.shape[1]
    wins = np.array([5, 20, 60, 120])
    ma = np.empty(4)
    path = np.empty(k)
    for j in range(n_ev):
        c = ci[j]
        t = s[j]
        sg = sigma[c, t]
        ok = np.isfinite(sg) and t + k - 119 >= 0 and t < n_days
        if ok:
            for h in range(k):
                if not np.isfinite(fc_log[c, t, h]):
                    ok = False
        if ok:
            # 알려진 구간 C[t+k-119..t] (k<120 일 때) 이 전부 유한해야 한다
            for u in range(max(t + k - 119, 0), t + 1):
                if not np.isfinite(C[c, u]):
                    ok = False
                    break
        if not ok:
            for m in range(4):
                out[m, j] = np.nan
            continue
        known = np.zeros(4)
        for wi in range(4):
            w = wins[wi]
            acc = 0.0
            for u in range(t + k - w + 1, t + 1):
                acc += C[c, u]
            known[wi] = acc
        cnt = np.zeros(4)
        for m in range(M):
            row = idx[j, m]
            for h in range(k):
                path[h] = np.exp(fc_log[c, t, h] + sg * pool[row, h])
            for wi in range(4):
                w = wins[wi]
                acc = known[wi]
                for h in range(max(1, k - w + 1), k + 1):
                    acc += path[h - 1]
                ma[wi] = acc / w
            if ma[0] > ma[1] and ma[1] > ma[2] and ma[2] > ma[3]:
                cnt[0] += 1
            if ma[0] > ma[1]:
                cnt[1] += 1
            if ma[1] > ma[2]:
                cnt[2] += 1
            if ma[2] > ma[3]:
                cnt[3] += 1
        for m in range(4):
            out[m, j] = cnt[m] / M


def event_probs(C, fc_log, sigma, pool, idx, ci, s, k: int) -> dict[str, np.ndarray]:
    """사건 (ci, s) 마다 M 개 경로에서 t+k 의 정배열·단기>장기 비율. 입력이 비유한이면 NaN."""
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    idx = np.asarray(idx)
    pool = np.asarray(pool)
    assert np.isfinite(pool).all(), "pool 에 비유한 값"
    assert pool.shape[1] >= k, "pool 열 < k"
    assert idx.shape[0] == len(ci) == len(s), "idx/ci/s 길이 불일치"
    assert len(ci) == 0 or (idx.min() >= 0 and idx.max() < len(pool)), "idx 범위 밖"
    assert len(ci) == 0 or (s.min() >= 0 and s.max() < np.shape(C)[1] and ci.min() >= 0 and ci.max() < np.shape(C)[0]), "ci/s 범위 밖"
    out = np.empty((4, len(ci)), dtype=np.float64)
    _event_kernel(
        np.ascontiguousarray(C, dtype=np.float64),
        np.ascontiguousarray(fc_log, dtype=np.float64),
        np.ascontiguousarray(sigma, dtype=np.float64),
        np.ascontiguousarray(pool, dtype=np.float64),
        np.ascontiguousarray(idx, dtype=np.int64),
        np.asarray(ci, dtype=np.int64),
        np.asarray(s, dtype=np.int64),
        int(k),
        out,
    )
    return {key: out[i] for i, key in enumerate(_PROB_KEYS)}


def realized(C, ci, s, k: int) -> dict[str, np.ndarray]:
    """t+k 의 실제 MA 상태(0/1). s+k ≥ n_days 이거나 MA 가 NaN 이면 NaN."""
    C = np.asarray(C, dtype=np.float64)
    ci = np.asarray(ci, dtype=np.int64)
    e = np.asarray(s, dtype=np.int64) + k
    n = len(ci)
    valid = (e < C.shape[1]) & (e - 119 >= 0)
    ee = np.where(valid, e, 119)
    mas = []
    for w in (5, 20, 60, 120):
        cols = ee[:, None] - np.arange(w)[None, :]
        mas.append(C[ci[:, None], cols].mean(axis=1))  # NaN 전파
    m5, m20, m60, m120 = mas
    bad = ~valid | np.isnan(m5 + m20 + m60 + m120)
    res = {
        "p_align": ((m5 > m20) & (m20 > m60) & (m60 > m120)),
        "p_up_5_20": m5 > m20,
        "p_up_20_60": m20 > m60,
        "p_up_60_120": m60 > m120,
    }
    if n == 0:
        return {key: np.empty(0) for key in _PROB_KEYS}
    return {key: np.where(bad, np.nan, v.astype(np.float64)) for key, v in res.items()}


# ---------------------------------------------------------------- 교차 확률(골든 HMM 관문 0) — 위 함수들은 그대로

@_jit
def _cross_kernel(C, ci, s, fcl, sg, pool, idx, k, short, long, dead, out):
    """사건 j 마다 M 경로 중 ∃h∈1..k: MA_short(t+h) > MA_long(t+h) 비율(``dead`` ≠ 0 이면 MA_short < MA_long — 데드).

    경로 종가 path[h] = exp(fcl[j, h−1] + sg[j]·pool[idx[j, m], h−1]). MA_w(t+h) = (알려진 C[t+h−w+1..t] 합 + 경로 1..h 중 창 안) / w.
    알려진 쪽 최대 창은 h=1 의 long−1 칸 C[t−long+2..t] — 그 칸이 하나라도 비유한이거나 sg·fcl 이 비유한이면 NaN.
    """
    n_ev, M = idx.shape
    n_days = C.shape[1]
    ks = np.zeros(long)             # ks[n] = C[t−n+1..t] 합(n = 0..long−1)
    pc = np.zeros(k + 1)            # pc[h] = 경로 1..h 합
    for j in range(n_ev):
        c = ci[j]
        t = s[j]
        ok = t - long + 2 >= 0 and t < n_days and np.isfinite(sg[j])
        if ok:
            for h in range(k):
                if not np.isfinite(fcl[j, h]):
                    ok = False
                    break
        if ok:
            acc = 0.0
            ks[0] = 0.0
            for n in range(1, long):
                v = C[c, t - n + 1]
                if not np.isfinite(v):
                    ok = False
                    break
                acc += v
                ks[n] = acc
        if not ok:
            out[j] = np.nan
            continue
        cnt = 0
        for m in range(M):
            row = idx[j, m]
            pc[0] = 0.0
            for h in range(1, k + 1):
                pc[h] = pc[h - 1] + np.exp(fcl[j, h - 1] + sg[j] * pool[row, h - 1])
            hit = False
            for h in range(1, k + 1):
                if h < short:
                    a = (ks[short - h] + pc[h]) / short
                else:
                    a = (pc[h] - pc[h - short]) / short
                if h < long:
                    b = (ks[long - h] + pc[h]) / long
                else:
                    b = (pc[h] - pc[h - long]) / long
                if (dead == 0 and a > b) or (dead != 0 and a < b):
                    hit = True
                    break
            if hit:
                cnt += 1
        out[j] = cnt / M


def _check_windows(k: int, short: int, long: int) -> None:
    if k < 1 or short < 1 or long <= short:
        raise ValueError(f"잘못된 (k, short, long) = ({k}, {short}, {long})")


def cross_kernel_events(C, ci, s, fcl, sg, pool, idx, k: int, short: int = 50, long: int = 200, dead: bool = False) -> np.ndarray:
    """사건별 입력판 — ``fcl`` (n_ev, k) 경로 기준 로그 가격, ``sg`` (n_ev,) 배율. :func:`cross_probs` 와 HMM 경로가 함께 쓴다.

    ``dead=True`` 는 부등호만 반대(∃h: MA_short < MA_long, 엄격) — golden_bt 의 P_d. 기본 False = 예전 그대로."""
    _check_windows(k, short, long)
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    idx = np.asarray(idx)
    pool = np.asarray(pool)
    fcl = np.asarray(fcl)
    assert np.isfinite(pool).all(), "pool 에 비유한 값"
    assert pool.ndim == 2 and pool.shape[1] >= k, "pool 열 < k"
    assert idx.shape[0] == len(ci) == len(s) == len(fcl) == len(sg), "idx/ci/s/fcl/sg 길이 불일치"
    assert fcl.ndim == 2 and fcl.shape[1] >= k, "fcl 열 < k"
    assert len(ci) == 0 or (idx.min() >= 0 and idx.max() < len(pool)), "idx 범위 밖"
    assert len(ci) == 0 or (s.min() >= 0 and s.max() < np.shape(C)[1] and ci.min() >= 0 and ci.max() < np.shape(C)[0]), "ci/s 범위 밖"
    out = np.empty(len(ci), dtype=np.float64)
    _cross_kernel(
        np.ascontiguousarray(C, dtype=np.float64), ci, s,
        np.ascontiguousarray(fcl[:, :k], dtype=np.float64), np.ascontiguousarray(sg, dtype=np.float64),
        np.ascontiguousarray(pool[:, :k], dtype=np.float64), np.ascontiguousarray(idx, dtype=np.int64),
        int(k), int(short), int(long), int(bool(dead)), out,
    )
    return out


def cross_probs(C, fc_log, sigma, pool, idx, ci, s, k: int, short: int = 50, long: int = 200, dead: bool = False) -> np.ndarray:
    """(n_ev,) — 사건 (ci, s) 마다 M 경로에서 P(∃h≤k: MA_short(t+h) > MA_long(t+h))(엄격). 입력이 비유한이면 NaN.

    경로 = exp(fc_log[c, t, h−1] + sigma[c, t]·pool[idx[j, m], h−1]) — :func:`event_probs` 와 같은 규약, 키만 교차.
    """
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    fc_log = np.asarray(fc_log)
    sigma = np.asarray(sigma)
    assert fc_log.shape[2] >= k, "fc_log 열 < k"
    assert len(ci) == 0 or (s.min() >= 0 and s.max() < np.shape(C)[1] and ci.min() >= 0 and ci.max() < np.shape(C)[0]), "ci/s 범위 밖"
    return cross_kernel_events(C, ci, s, fc_log[ci, s, :k], sigma[ci, s], pool, idx, k, short, long, dead=dead)


def cross_realized(C, ci, s, k: int, short: int = 50, long: int = 200, ma: "dict | None" = None, dead: bool = False) -> np.ndarray:
    """(n_ev,) 0/1 — 실제 ∃h∈1..k: MA_short(t+h) > MA_long(t+h)(엄격, pandas rolling 평균 = :func:`ma_state` 와 같은 정의).

    s+k ≥ n_days 이거나 h=1..k 의 MA 중 하나라도 NaN(창 C[t−long+2..t+k] 에 NaN)이면 NaN. ``ma`` = {short: 패널, long: 패널} 재사용(선택).
    """
    _check_windows(k, short, long)
    C = np.asarray(C, dtype=np.float64)
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    if len(ci) == 0:
        return np.empty(0)
    if ma is None:
        ma = {short: _rolling_mean(C, short), long: _rolling_mean(C, long)}
    a_, b_ = ma[short], ma[long]
    n_days = C.shape[1]
    e = s[:, None] + np.arange(1, k + 1)[None, :]
    valid = (e[:, -1] < n_days) & (s >= 0)
    ee = np.where(valid[:, None], e, 0)
    a = a_[ci[:, None], ee]
    b = b_[ci[:, None], ee]
    bad = ~valid | np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1)
    with np.errstate(invalid="ignore"):
        hit = ((a < b) if dead else (a > b)).any(axis=1)
    return np.where(bad, np.nan, hit.astype(np.float64))
