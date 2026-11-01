"""가격 시계열 기업행동(액면분할·무상증자) 조정 — ``daily_bars`` → ``daily_bars_adjusted``.

## 무엇이 문제인가

`daily_bars` 는 원 일봉이 아니라 **키움 수정주가**(upd_stkpc_tp=1)다. 키움 수정주가는 내려받은 시점까지의
기업행동만 반영한다. 그런데 이 표는 여러 번에 나눠 받은 덩어리를 이어 붙인 것이라, 덩어리 경계에서
**"앞 덩어리를 받은 뒤 일어난 기업행동의 누적계수"만큼 계단(이음매)** 이 생긴다. 이음매는 몇 날짜에
몰린다(2019-02-27·2021-07-29·2024-01-08·2016-12-28 — 서로 무관한 수백 종목이 같은 날 점프).
예: 258830 2021-07-29 24,500 → 4,900(원주가는 그대로 24,500), 그 계단 = 10-20 의 5:1 분할.

옛 판(~2026-11)은 `daily_bars` 만 보고 "±30% 를 넘고 3거래일 유지되는 점프"를 분할로 봤다. 그래서

- 30% 안쪽 이음매는 그대로 통과했다(2016-09~2026-09 실측 이음매 174칸 중 106칸이 20~30%).
- 변동이 큰 구간의 이음매는 3일 유지 검사에서 떨어졌다.
- 진짜 −30% 하한가(수정가 반올림으로 비율 0.69998, 001380 2025-03-26)를 분할로 읽어 그날 수익률을 0 으로,
  앞 이력 전체를 ×0.7 했다(23칸).

## 무엇을 하는가 (원주가 대조)

같은 날 **두 수익률**을 비교한다: `daily_bars` 수익률 r_db 와 원주가 수익률 r_raw
(`daily_bars_raw.close`, 없으면 `supply_demand.close` — 키움 행은 `daily_bars_raw` 와 겹치는 날 100% 같다,
네이버 폐지 백필 행은 네이버 원 종가). 비율 q = r_db / r_raw 가 1 이면 그날 수정 기준이 그대로다.
q ≠ 1(계단)은 둘 중 하나다:

1. **기업행동** — 원주가가 기준가 재설정으로 뛰었고 수정주가는 이를 이미 반영했다 → 그대로 둔다.
2. **이음매**(또는 수정주가 한 날 오류) — 원주가는 이어지는데 수정주가만 뛰었다 → 그날 앞 이력 전체에
   q 를 곱해 그날 수익률을 r_raw 로 만든다(표준 back-adjust).

원주가만으로는 30% 안쪽 기업행동(권리락 −16% 등)과 30% 안쪽 이음매를 가를 수 없다. 심판은
`supply_demand.flu_rt`(키움 등락률, **기준가 대비**)다: m = 1 + flu_rt/1e4 는 기준가 재설정을 뺀 그날 시장
수익률이라, r_db 와 r_raw 중 m 에 가까운 쪽이 맞다. `flu_rt` 를 수익률 자체로 쓰지는 않는다(2026-09-14~
종가가 KRX 애프터마켓 통합값이라 기준가와 기준이 다르다 — 둘 중 고르는 데만 쓴다).

- 계단이 없는데(q = 1) 원주가가 ±30% 가격제한폭 밖으로 뛰었으면 수정주가도 같이 뛴 것이다 — 매일 덧붙인
  최근 구간에서 수정주가가 아직 반영 못 한 기업행동이다. m 이 그 점프를 설명하지 못하면(기준가 재설정) 그날
  수익률을 m 으로 맞춘다. m 도 같이 뛰었으면 진짜 움직임이다(정리매매처럼 가격제한이 없는 날) → 그대로.
- 심판이 없는 날(`supply_demand` 가 2017-01-02 부터라 그 전, 네이버 폐지 백필): 원주가가 제한폭 밖이고 수정주가가
  안이면 기업행동, 반대면 이음매. 둘 다 제한폭 안인 계단은 **같은 날 ``CLUSTER_MIN`` 종목 이상**에서 나올 때만
  이음매로 본다 — 실제 기업행동은 그렇게 몰리지 않는다. 아니면 그대로 둔다.
- **원주가가 없는 날**(`daily_bars_raw`·`supply_demand` 둘 다 없는 행): 옛 규칙(±30% 초과 + 3일 유지)으로
  돌아간다. 원주가 없이는 이음매와 진짜 움직임을 가를 수 없어서다. 이 경로만 옛 오탐·누락을 그대로 가진다.
  `raw_close` 열이 없는 입력(옛 호출)도 전부 이 경로다.

- **2026-09-14~(KRX 애프터마켓)**: 원 종가는 20:00 통합가, flu_rt 기준가는 전일 정규장 종가라 둘이 그 세션 등락만큼
  다르다(최대 35% 실측). 그래서 원주가 비로 제한폭을 재지 않고(0035S0 2026-09-30 상한가가 원주가 비 +43.6%),
  기준가 재설정은 "기준가 / 전일 원 종가"가 전일 flu_rt 로 본 가능 범위 밖일 때만 인정한다(``AFTERMARKET_FROM`` 주석).
- **원주가 표의 마지막 날은 쓰지 않는다**(``read_inputs``): 그날 19:00 스냅숏이라 확정 봉과 다르다(2026-10-30 실측
  1,813/2,588 종목, 최대 11.1%) — 쓰면 마지막 날에 가짜 이음매가 생겨 그 종목 이력 전체에 배수가 곱해진다.
  그날은 `supply_demand.close`(daily_finalize 가 확정치로 덮는다)로 대신한다.

`daily_bars` 는 수정 뒤 정수로 반올림돼 있어 작은 값(수정 종가 100원 등)에서 하루 수익률이 1% 가까이
흔들린다. 계단 문턱은 그 반올림 폭보다 크게 잡는다(``_step_tol``).

OHLC 전부에 같은 배수를 곱한다. 거래량은 기본 그대로(``adjust_volume=True`` 면 배수로 나눈다).

이 모듈은 quant-airflow 의 daily/weekly_price_adjust·daily_finalize DAG 가 swing-it 작업 트리를 PYTHONPATH 로
직접 실행한다 — **numpy·pandas 밖을 import 하지 않는다**(2026-09-14 16:55 런이 ``krx_quant_core`` import 로 죽었다).

CLI: ``python -m swing_it.price_adjust --db <DSN>`` 진단 리포트, ``--rebuild-db`` 로 표 전체 재계산.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# 한국 일일 가격제한 ±30%. krx_quant_core.market.limits.LIMIT_RATE 와 같은 값이지만 import 하지 않는다(위 docstring).
LIMIT = 0.30
DOWN = 1 - LIMIT  # 0.70 — 옛 규칙(원주가 없는 날)의 분할 후보 문턱
UP = 1 / DOWN  # ≈1.4286 (역분할/기준상향)
PERSIST_DAYS = 3  # 옛 규칙: 새 레벨이 유지되어야 분할로 인정(일시 스파이크 배제)
PERSIST_TOL = 0.25

# 원주가 수익률이 "시장에서 가능한 움직임"인 범위 [0.70, 1.30]. 상·하한가는 기준가 × (1 ± 0.3) 을 기준가 쪽으로
# 틱 반올림하므로 정확히 이 안에 든다 — BAND_EPS 는 부동소수 여유.
BAND_LO, BAND_HI, BAND_EPS = 1 - LIMIT, 1 + LIMIT, 0.002
STEP_TOL = 0.005  # q 가 이만큼(+ 반올림 폭) 벗어나야 계단으로 본다
ROUND_K = 0.6  # 수정가 정수 반올림 폭: |q−1| 오차 ≤ 0.5/db_t + 0.5/db_{t−1} 에 여유
EVENT_TOL = 0.02  # q=1·원주가 제한폭 밖일 때 기준가가 전일 원 종가와 이만큼 다르면 기준가 재설정(미반영 기업행동)
# 2026-09-14~ 원 종가는 KRX 애프터마켓(16:00~20:00) 끝 가격이라 기준가(전일 정규장 종가 R)와 그 세션 등락만큼 다르다 —
# 10월 실측 최대 35%(087010 10-07), 475830 2026-09-17 은 정규장 53,400 · 통합 35,500. 그래서 전일이 이날 이후면
# "기준가 / 전일 원 종가"(gap = R/I) 가 1 이 아니어도 기업행동이 아니다. 대신 **가능 범위**가 있다: 정규장 종가 R 과
# 통합 종가 I 둘 다 전일 기준가 × [0.7, 1.3] 안이므로 gap ∈ [0.7/m_{t−1}, 1.3/m_{t−1}](m_{t−1} = 전일 flu_rt 비).
# 그 밖이면 기준가 재설정이다. 전일 flu_rt 가 없거나 전일이 제한 없는 날(상장 첫날 등)이면 [0.7/1.3, 1.3/0.7].
AFTERMARKET_FROM = "2026-09-14"
GAP_EPS = 0.005  # 가능 범위의 틱 반올림 여유
CLUSTER_MIN = 10  # 심판 없는 제한폭 안 계단이 같은 날 이 종목 수 이상이면 수집 이음매 날


def _legacy_events(close: np.ndarray, allowed: np.ndarray) -> list[tuple[int, float]]:
    """옛 규칙: ±30% 초과 + 앞뒤 3일 레벨 유지면 분할. ``allowed[t]`` 인 날만 후보(원주가 없는 날)."""
    n = len(close)
    out: list[tuple[int, float]] = []
    if n <= PERSIST_DAYS + 1:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = close[1:] / close[:-1]
    cand = np.where(np.isfinite(ratio) & ((ratio < DOWN) | (ratio > UP)) & allowed[1:])[0] + 1
    for t in cand:
        if t >= n - PERSIST_DAYS or not (close[t - 1] > 0):
            continue
        # 새 레벨(c1)이 이후에 유지되고 AND 이전 레벨(c0)이 그 앞 며칠과 일관되어야
        # 진짜 분할. 하루만 튀는 스파이크는 되돌림 지점에서 c0(스파이크)이 직전과 어긋나 걸러짐.
        fwd = close[t + 1:t + 1 + PERSIST_DAYS]
        fwd = fwd[np.isfinite(fwd)]
        bwd = close[max(0, t - 1 - PERSIST_DAYS):t - 1]
        bwd = bwd[np.isfinite(bwd)]
        fwd_ok = fwd.size > 0 and abs(np.median(fwd) / close[t] - 1) < PERSIST_TOL
        bwd_ok = bwd.size == 0 or abs(np.median(bwd) / close[t - 1] - 1) < PERSIST_TOL
        if fwd_ok and bwd_ok:
            out.append((int(t), float(ratio[t - 1])))
    return out


def _in_band(r: np.ndarray, eps: np.ndarray | float) -> np.ndarray:
    return (r >= BAND_LO - eps) & (r <= BAND_HI + eps)


def _step_tol(db: np.ndarray, pdb: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return STEP_TOL + ROUND_K * (1 / np.abs(db) + 1 / np.abs(pdb))


def _classify(close: np.ndarray, raw: np.ndarray, ref: np.ndarray,
              aftermarket: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """한 종목(시간순)의 날마다 조정 배수를 고른다. ``aftermarket[t]`` = 전일(t−1)이 AFTERMARKET_FROM 이후.

    반환 (factor, kind, amb_q):
      factor[t] — 그날 **앞** 이력에 곱할 배수(1 = 그대로). kind[t] — 진단 라벨(빈 문자열 = 해당 없음).
      amb_q[t]  — 심판 없는 제한폭 안 계단의 q(아니면 NaN). 이음매 날 판정(CLUSTER_MIN)은 종목을 모아 본 뒤 한다.
    """
    n = len(close)
    factor = np.ones(n)
    kind = np.full(n, "", dtype=object)
    amb_q = np.full(n, np.nan)
    if n < 2:
        return factor, kind, amb_q
    c0, c1 = close[:-1], close[1:]
    w0, w1 = raw[:-1], raw[1:]
    m = ref[1:]
    m_prev = ref[:-1]
    am = np.zeros(n - 1, dtype=bool) if aftermarket is None else np.asarray(aftermarket, bool)[1:]
    # 애프터마켓 뒤에는 원주가 비가 ±30% 를 넘을 수 있다 — 심판 없는 날의 제한폭을 그만큼 넓힌다
    widen = np.where(am, BAND_HI / BAND_LO, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # 전일이 가격제한 없는 날(신규 상장 첫날 — 0035S0 2026-09-29 +50%, 정리매매)이면 전일 정규장 종가도 제한이 없다
        mp = np.where(np.isfinite(m_prev) & _in_band(m_prev, BAND_EPS), m_prev, np.nan)
        gap_lo = np.where(am, np.where(np.isfinite(mp), BAND_LO / mp, BAND_LO / BAND_HI) - GAP_EPS, 1 - EVENT_TOL)
        gap_hi = np.where(am, np.where(np.isfinite(mp), BAND_HI / mp, BAND_HI / BAND_LO) + GAP_EPS, 1 + EVENT_TOL)
        r_db = c1 / c0
        r_raw = w1 / w0
        q = r_db / r_raw
        tol = _step_tol(c1, c0)
        ok_db = np.isfinite(r_db) & (c0 > 0) & (c1 > 0)
        has_raw = ok_db & np.isfinite(r_raw) & (w0 > 0) & (w1 > 0)
        has_ref = has_raw & np.isfinite(m) & (m > 0)
        step = has_raw & (np.abs(q - 1) > tol)
        raw_in = (r_raw >= BAND_LO / widen - BAND_EPS) & (r_raw <= BAND_HI * widen + BAND_EPS)
        db_in = (r_db >= BAND_LO / widen - tol) & (r_db <= BAND_HI * widen + tol)
        gap = r_raw / m  # 기준가 / 전일 원 종가
        d_db = np.abs(np.log(r_db / m))  # 심판까지 로그 거리
        d_raw = np.abs(np.log(r_raw / m))

    f1 = np.ones(n - 1)
    k1 = np.full(n - 1, "", dtype=object)
    a1 = np.full(n - 1, np.nan)

    # 1) 계단 + 심판: m 에 가까운 쪽이 맞다. 애프터마켓 뒤에는 각 가설이 함의하는 R/I(이음매: gap, 기업행동:
    #    r_db/m)가 가능 범위 안인지 먼저 보고, 한쪽만 가능하면 그쪽이다.
    s_ref = step & has_ref
    with np.errstate(invalid="ignore"):
        seam_ok = (gap >= gap_lo) & (gap <= gap_hi)
        ev_gap = r_db / m
        ev_ok = (ev_gap >= gap_lo) & (ev_gap <= gap_hi)
    nearest = d_raw < d_db
    seam = s_ref & np.where(am, seam_ok & (~ev_ok | nearest), nearest)
    f1[seam] = q[seam]
    k1[seam] = "seam"
    k1[s_ref & ~seam] = "event"
    # 2) 계단, 심판 없음: 제한폭으로 가른다
    s_nr = step & ~has_ref
    k1[s_nr & ~raw_in & db_in] = "event"
    seam2 = s_nr & raw_in & ~db_in
    f1[seam2] = q[seam2]
    k1[seam2] = "seam"
    amb = s_nr & raw_in & db_in
    a1[amb] = q[amb]
    k1[amb | (s_nr & ~raw_in & ~db_in)] = "ambiguous"
    # 3) 계단 없음(q≈1)인데 원주가가 크게 뛰었다 — 수정주가가 아직 반영 못 한 기업행동인가?
    #    애프터마켓 전: 원주가 비가 제한폭 밖이고 기준가가 전일 원 종가와 다르면(기준가 재설정) 기업행동.
    #    애프터마켓 뒤: 원주가 비로는 제한폭을 못 잰다(0035S0 2026-09-30 상한가가 +43.6%) — gap 이 가능 범위 밖일 때만.
    flat_jump = has_raw & ~step & np.where(am, True, ~raw_in)
    with np.errstate(invalid="ignore", divide="ignore"):
        unrefl = flat_jump & has_ref & ((gap < gap_lo) | (gap > gap_hi)) & _in_band(m, BAND_EPS)
        f1[unrefl] = (r_db / m)[unrefl]  # 그날 수익률 = m(기준가 대비 시장 수익률)
    flat_jump &= ~am | ~raw_in
    k1[unrefl] = "unreflected"
    k1[flat_jump & ~unrefl] = "genuine_jump"

    factor[1:], kind[1:], amb_q[1:] = f1, k1, a1

    # 4) 원주가 없는 날 — 옛 규칙
    allowed = np.zeros(n, dtype=bool)
    allowed[1:] = ok_db & ~has_raw
    for t, r in _legacy_events(close, allowed):
        factor[t] = r
        kind[t] = "legacy_split"
    return factor, kind, amb_q


def _segments(codes: np.ndarray) -> list[tuple[int, int]]:
    bounds = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    return list(zip(np.r_[0, bounds].tolist(), np.r_[bounds, len(codes)].tolist()))


def _plan(df: pd.DataFrame) -> pd.DataFrame:
    """종목·날짜순 작업표: 날마다 ``_factor``(그날 앞 이력에 곱할 배수)·``_kind``, 원래 행 위치 ``_row``."""
    n = len(df)

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.full(n, np.nan)
        return pd.to_numeric(df[name], errors="coerce").to_numpy(float)

    work = pd.DataFrame({
        "code": df["code"].astype(str).to_numpy(),
        "date": df["date"].astype(str).to_numpy(),
        "close": col("close"),
        "raw": col("raw_close"),
        "ref": 1 + col("flu_rt") / 1e4,
        "_row": np.arange(n),
    }).sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    close, raw, ref = (work[c].to_numpy(float) for c in ("close", "raw", "ref"))
    am = work["date"].to_numpy() >= AFTERMARKET_FROM  # 행 t 의 날짜 — _classify 는 [1:] 와 짝지은 전일로 본다
    am_prev = np.r_[False, am[:-1]]
    fac = np.ones(n)
    kind = np.full(n, "", dtype=object)
    amb = np.full(n, np.nan)
    for s, e in _segments(work["code"].to_numpy()):
        fac[s:e], kind[s:e], amb[s:e] = _classify(close[s:e], raw[s:e], ref[s:e], am_prev[s:e])
    # 심판 없는 제한폭 안 계단: 같은 날 CLUSTER_MIN 종목 이상이면 수집 이음매 날
    is_amb = np.isfinite(amb)
    if is_amb.any():
        dates = work["date"].to_numpy()
        counts = pd.Series(dates[is_amb]).value_counts()
        hit = is_amb & np.isin(dates, counts.index[counts >= CLUSTER_MIN].to_numpy())
        fac[hit] = amb[hit]
        kind[hit] = "seam_cluster"
    work["_factor"] = fac
    work["_kind"] = kind
    return work


def _back_factors(work: pd.DataFrame) -> np.ndarray:
    """``_factor`` → 행마다 누적 배수 = 같은 종목의 **뒤** 날짜 배수 전부의 곱(표준 back-adjust)."""
    f = work["_factor"].to_numpy(float)
    out = np.ones(len(f))
    for s, e in _segments(work["code"].to_numpy()):
        seg = f[s:e]
        if (seg != 1).any():
            out[s:e - 1] = np.cumprod(seg[::-1])[::-1][1:]  # out[i] = prod(seg[i+1:])
    return out


def adjust_prices(df: pd.DataFrame, *, cols=("open", "high", "low", "close"),
                  adjust_volume: bool = False) -> pd.DataFrame:
    """long-format(code,date,OHLC[,volume][,raw_close][,flu_rt]) 가격을 종목별로 백조정해 반환.

    ``raw_close``(그날 원 종가)·``flu_rt``(키움 등락률 ×100) 열이 있으면 원주가 대조로 이음매를 찾고
    (모듈 docstring), 없으면 옛 ±30% 규칙만 쓴다. 입력을 변경하지 않고 조정된 사본을 돌려준다.
    `date` 오름차순 정렬을 가정하지 않는다.
    """
    out = df.copy()
    present = [c for c in cols if c in out.columns]
    # 정수 가격 컬럼에 실수 배수를 할당하므로 먼저 float 캐스팅(dtype 경고 방지)
    for c in present:
        out[c] = out[c].astype(float)
    if adjust_volume and "volume" in out.columns:
        out["volume"] = out["volume"].astype(float)
    if out.empty:
        return out
    work = _plan(out)
    fac = np.empty(len(out))
    fac[work["_row"].to_numpy()] = _back_factors(work)
    for c in present:
        out[c] = out[c].to_numpy(float) * fac
    if adjust_volume and "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy(float) / fac
    return out


def diagnose(df: pd.DataFrame) -> pd.DataFrame:
    """조정이 들어간 날(code,date,ratio,kind) — 진단용. ratio = 그날 앞 이력에 곱한 배수.

    kind: seam·seam_cluster(수집 이음매), unreflected(수정주가가 아직 반영 못 한 기업행동),
    legacy_split(원주가 없는 날의 옛 ±30% 규칙).
    """
    if df.empty:
        return pd.DataFrame(columns=["code", "date", "ratio", "kind"])
    work = _plan(df)
    hit = work[work["_factor"] != 1]
    return pd.DataFrame({
        "code": hit["code"].to_numpy(),
        "date": hit["date"].to_numpy(),
        "ratio": np.round(hit["_factor"].to_numpy(float), 3),
        "kind": hit["_kind"].to_numpy(),
    })


def _has_table(con: Any, name: str) -> bool:
    from .storage import _is_pg

    if _is_pg(con):
        cur = con.cursor()
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
        return bool(cur.fetchone()[0])
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


INPUT_COLS = "d.code,d.date,d.open,d.high,d.low,d.close,d.volume,d.trade_value,d.source"


def read_inputs(con: Any, *, cols: str = INPUT_COLS,
                date_from: str | None = None, date_to: str | None = None) -> pd.DataFrame:
    """``daily_bars`` + 원 종가 ``raw_close`` + 키움 등락률 ``flu_rt`` — ``adjust_prices`` 입력.

    raw_close = ``daily_bars_raw.close``(마지막 날 19:00 스냅숏은 빼고), 없으면 ``supply_demand.close``. flu_rt 는 키움 행만(네이버 백필은 NULL).
    ``daily_bars_raw`` 가 없는 DB(sqlite)는 ``supply_demand`` 만 본다. ``date_from``/``date_to``(포함)는 비교
    스크립트가 나눠 읽을 때만 쓴다 — 운영 재계산은 종목 전체 이력이 필요하다.
    """
    from .storage import _is_pg

    pg = _is_pg(con)
    ph = "%s" if pg else "?"
    has_raw = _has_table(con, "daily_bars_raw")
    # 날짜 조건은 세 표 모두에 건다 — d 에만 걸면 조인 상대 표를 매번 통째로 읽는다(연도별 조회가 60s 를 넘었다)
    rng, rng_params = "", []
    if date_from:
        rng += f" AND {{t}}.date >= {ph}"
        rng_params.append(date_from)
    if date_to:
        rng += f" AND {{t}}.date <= {ph}"
        rng_params.append(date_to)
    params: list[str] = []
    raw_join, raw_expr = "", "s.close"
    if has_raw:
        # 원주가 표의 마지막 날은 그날 19:00 스냅숏이다(daily_bars_raw DAG — 애프터마켓이 20:00 까지라 최종 봉과 다르다,
        # 2026-10-30 실측 1,813/2,588 종목, 최대 11.1%). 다음 날 19:00 에 최근 5일을 다시 받아야 확정이므로 마지막 날은
        # 원주가로 쓰지 않는다 — 그날은 supply_demand.close(daily_finalize 가 확정치로 덮는다)로 대신한다.
        raw_join = ("LEFT JOIN daily_bars_raw r ON r.code = d.code AND r.date = d.date"
                    " AND r.date < (SELECT max(date) FROM daily_bars_raw)" + rng.format(t="r") + " ")
        raw_expr = "COALESCE(r.close, s.close)"
        params += rng_params
    sd_join = "LEFT JOIN supply_demand s ON s.code = d.code AND s.date = d.date" + rng.format(t="s")
    params += rng_params
    where = (" WHERE TRUE" + rng.format(t="d")) if rng else ""
    params += rng_params
    like = "'kiwoom%%'" if (pg and params) else "'kiwoom%'"  # psycopg2 는 인자가 있을 때만 % 를 해석한다
    sql = (
        f"SELECT {cols}, {raw_expr} AS raw_close, "
        f"CASE WHEN s.source LIKE {like} THEN s.flu_rt END AS flu_rt "
        f"FROM daily_bars d {raw_join}{sd_join}{where}"
    )
    df = pd.read_sql(sql, con, params=params or None)
    df["date"] = df["date"].astype(str)
    return df


def rebuild_adjusted_table(con: Any, *, adjust_volume: bool = False) -> int:
    """Recompute back-adjusted OHLC for ALL of ``daily_bars`` and upsert into
    ``daily_bars_adjusted``.

    Back-adjustment must see a code's *entire* history to place each factor
    correctly (a seam detected today changes the adjustment applied to every
    earlier date for that code) — so this always recomputes from scratch over
    the full table rather than incrementally, and re-upserts every row
    (existing (code,date) rows are overwritten via the natural-key upsert).
    """
    from .storage import ADJUSTED_BAR_COLUMNS, upsert_daily_bars_adjusted

    # source 도 함께 읽어 그대로 전파한다 — 백테스트가 읽는 건 조정가 테이블이라,
    # "이 행의 trade_value 는 close*volume 근사치(폐지 종목 백필)"라는 사실이 여기
    # 없으면 ADV 문턱을 다루는 코드가 그걸 알 방법이 없다.
    df = read_inputs(con)
    adjusted = adjust_prices(df, adjust_volume=adjust_volume)
    records = list(adjusted[ADJUSTED_BAR_COLUMNS].itertuples(index=False, name=None))
    return upsert_daily_bars_adjusted(con, records)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="기업행동 조정 진단 리포트 / DB 조정가 테이블 재생성")
    ap.add_argument("--db", default=None)
    ap.add_argument("--rebuild-db", action="store_true",
                    help="진단만 하지 않고 daily_bars_adjusted 테이블을 전체 재계산해 upsert")
    args = ap.parse_args()
    from swing_it.storage import connect, db_default
    con = connect(args.db or db_default())

    if args.rebuild_db:
        n = rebuild_adjusted_table(con)
        print(f"daily_bars_adjusted 재생성 완료: {n}행 upsert")
        con.close()
        return 0

    df = read_inputs(con, cols="d.code,d.date,d.close")
    con.close()
    ev = diagnose(df)
    print(f"조정 지점 {len(ev)}건, 고유종목 {ev['code'].nunique() if len(ev) else 0}개")
    if len(ev):
        print("종류별:\n" + ev["kind"].value_counts().to_string())
        print("상위 날짜:\n" + ev["date"].value_counts().head(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
