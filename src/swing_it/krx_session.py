"""KRX 정규장 일봉 — 2026-09-14 부터 ``daily_bars`` 에 섞인 NXT 야간 체결을 걷어낸다.

**결함(2026-10-10 확인):** 2026-09-14 부터 키움 일봉 수집이 KRX+NXT 통합 시세를 받아 ``daily_bars.close`` 가 그날 마지막 체결
(NXT 19:59~20:00)이 됐다. 예: 005930 2026-09-17 일봉 종가 256,000 = 분봉 19:59, KRX 종가(15:30 단일가) 252,500.
분봉이 있는 종목 기준 10-08 하루에 1,351 중 1,182 종목의 종가가 KRX 종가와 다르다. ``daily_bars_adjusted`` 도 같은 값을 물려받는다.
고가·저가·거래량·거래대금도 NXT 시간대를 포함한다. 시가(09:00)는 정상이다.

(2026-10-08~09 에 "오진" 이라 적었던 진단 둘 — 분봉 15:30 봉이 공식 종가와 다르다는 설명, 일봉 = 수급 테이블 종가 일치 — 은 틀렸다.
수급 테이블도 같은 키움 피드라 일치가 아무것도 증명하지 않고, 분봉 ``ts`` 는 시간대 없는 KST 라 변환을 한 번 더 걸면 비교가 어긋난다.)

**고치는 법:** ``minute_bars``(KST naive) 의 09:00~15:30 봉으로 정규장 OHLCV 를 다시 만들고, 조정 테이블에는 같은 날 조정계수
(조정 종가 / 원 종가 — 두 값 모두 같은 오염을 담아 비율은 유효)를 곱해 넣는다. 분봉이 없는 종목·날짜는 원값을 두고 ``krx_patched=False`` 로 표시한다
(분봉은 하루 ~1,400~1,800 종목 — 유동 종목 위주라 NXT 거래 종목 대부분을 덮는다).

정본 수정은 수집기(quant-airflow, KRX 전용 시세 요청)에서 해야 한다 — 이 모듈은 그 전까지 swing-it 의 읽는 쪽 보정이다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: 일봉에 NXT 가 섞이기 시작한 첫 거래일.
NXT_START = "2026-09-14"
#: 정규장 분봉 구간(KST, 양 끝 포함). 종가 단일가는 보통 15:30 봉이지만 **변동성 완화장치로 종가 단일가가 연장되면 15:32 에 찍힌다**
#: (리뷰 `REVIEW_META_GEN2.md` N1 — 15:30 컷이면 그날 종가가 15:19 연속매매가가 됐다). NXT 애프터마켓 첫 봉은 15:35 라 15:32 컷은 그것을 넣지 않는다.
SESSION_OPEN = "09:00:00"
SESSION_CLOSE = "15:32:00"


def session_bars_from_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    """분봉(code, ts, open, high, low, close, volume, trade_value[원]) → 정규장 일봉(code, date, open, high, low, close,
    volume, trade_value[백만원]). ``ts`` 는 KST naive. 09:00~15:32 밖의 봉은 버린다. 종가 = 15:32 이하 마지막 봉의 종가(연장된 종가 단일가 포함)."""
    m = minutes.copy()
    m["ts"] = pd.to_datetime(m["ts"])
    t = m["ts"].dt.strftime("%H:%M:%S")
    m = m[(t >= SESSION_OPEN) & (t <= SESSION_CLOSE)]
    if m.empty:
        return pd.DataFrame(columns=["code", "date", "open", "high", "low", "close", "volume", "trade_value"])
    m = m.sort_values(["code", "ts"])
    m["date"] = m["ts"].dt.normalize()
    g = m.groupby(["code", "date"], sort=False)
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), trade_value=("trade_value", "sum")).reset_index()
    out["trade_value"] = out["trade_value"] / 1e6
    return out


def read_session_bars(con: Any, *, start: str = NXT_START, end: "str | None" = None) -> pd.DataFrame:
    """DB 분봉 → 정규장 일봉. ``start``~``end``(포함) 날짜."""
    q = ("SELECT code, ts, open, high, low, close, volume, trade_value FROM minute_bars "
         "WHERE ts >= %(lo)s AND ts < %(hi)s AND ts::time BETWEEN %(o)s AND %(c)s")
    hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else "2100-01-01"
    m = pd.read_sql_query(q, con, params={"lo": start, "hi": hi, "o": SESSION_OPEN, "c": SESSION_CLOSE})
    return session_bars_from_minutes(m)


def read_session_bars_agg(con: Any, *, start: str = NXT_START, end: "str | None" = None) -> pd.DataFrame:
    """:func:`read_session_bars` 와 같은 결과를 **DB 에서 집계**해 받는다(Postgres). 분봉을 pandas 로 끌어오면 하루 ~1,800 종목 ×
    391 분 = 70만 행, 한 달이면 1,300만 행이라 RSS 가 수 GB 로 뛴다(2026-10-10 실측) — 긴 구간·메모리 상한이 있는 작업은 이쪽.
    시가 = 첫 봉 open, 종가 = 15:32 이하 마지막 봉 close, 고저 = max/min, 거래량·거래대금 = 합(거래대금은 백만원)."""
    q = ("SELECT code, ts::date AS date, (array_agg(open ORDER BY ts))[1] AS open, max(high) AS high, min(low) AS low, "
         "(array_agg(close ORDER BY ts DESC))[1] AS close, sum(volume) AS volume, sum(trade_value) / 1e6 AS trade_value "
         "FROM minute_bars WHERE ts >= %(lo)s AND ts < %(hi)s AND ts::time BETWEEN %(o)s AND %(c)s "
         "GROUP BY code, ts::date ORDER BY code, date")
    hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else "2100-01-01"
    out = pd.read_sql_query(q, con, params={"lo": start, "hi": hi, "o": SESSION_OPEN, "c": SESSION_CLOSE})
    out["date"] = pd.to_datetime(out["date"])
    for col in ("open", "high", "low", "close", "volume", "trade_value"):
        out[col] = out[col].astype(float)
    return out


def patch_prices(prices: pd.DataFrame, session: pd.DataFrame, *, since: str = NXT_START,
                 adjusted: bool = True) -> pd.DataFrame:
    """롱 가격표(code, date, open, high, low, close, volume, trade_value …)의 ``since`` 이후 행을 정규장 값으로 바꾼다.

    ``adjusted=True`` 면 그 행의 조정계수 = 행의 close / 원 일봉 close 를 알 수 없으므로 **행 자신의 open 비율**을 쓴다:
    시가는 오염되지 않았으므로 factor = 행 open / 정규장 open. 가격 4개에 factor 를 곱하고 거래량·거래대금은 정규장 값 그대로.
    정규장 값이 없는 행은 그대로 두고 ``krx_patched=False``. 반환은 사본, 열 ``krx_patched`` 추가.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    # DB 일봉의 가격·거래량은 정수 열로 오기도 한다 — 정규장 값(실수·조정계수 곱)을 대입하려면 실수여야 한다.
    for col in ("open", "high", "low", "close", "volume", "trade_value"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["krx_patched"] = False
    if session.empty:
        return df
    s = session.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index(["code", "date"])
    late = df["date"] >= pd.Timestamp(since)
    key = pd.MultiIndex.from_arrays([df.loc[late, "code"], df.loc[late, "date"]])
    hit = key.isin(s.index)
    idx = df.index[late][hit]
    sv = s.reindex(key[hit])
    if adjusted:
        so = sv["open"].to_numpy()
        fac = np.where(so > 0, df.loc[idx, "open"].to_numpy(dtype=float) / np.where(so > 0, so, np.nan), np.nan)
    else:
        fac = np.ones(len(idx))
    ok = np.isfinite(fac)
    idx, sv, fac = idx[ok], sv[ok], fac[ok]
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df.loc[idx, col] = sv[col].to_numpy() * fac
    for col in ("volume", "trade_value"):
        if col in df.columns:
            df.loc[idx, col] = sv[col].to_numpy()
    df.loc[idx, "krx_patched"] = True
    return df


# ── 원(미수정) 종가 ───────────────────────────────────────────────────────────────────────────────────────────────
# **결함(2026-10-12 검증 29차):** ``daily_bars.close`` 는 이름과 달리 원 종가가 아니다 — 이후 액면분할·병합을 소급 반영한다
# (005930 2017-03-31 41,200 원 vs 실제 2,060,000 원, 000670 2024-07-31 29,626 vs 314,500, 병합 종목은 5·10 배 위로).
# 이것에 그 시점 주식수를 곱한 시총은 분할 전 대형주를 수십 분의 1 로, 병합 전 소형주를 수 배로 만든다(28세대 VOID, leader21 시총 상위 200).
# 원 종가 = ``supply_demand.close``(키움·네이버, 2017~ 거의 전 종목; 키움은 부호가 붙을 수 있어 절댓값). 없을 때만 일봉으로 돌아가고
# NXT 이후(``NXT_START``~)는 분봉 정규장 종가(원 가격)가 이긴다. 정본 수정은 quant-airflow(원 일봉 테이블) 소관.
RAW_CLOSE_CANARY = ("005930", "2017-03-31", 1_000_000.0)   # 50:1 분할 전 — 원 종가는 200만 원대


def combine_raw_close(sd: pd.DataFrame, bars: pd.DataFrame, session: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """(code, date, close) 세 원천을 우선순위(분봉 정규장 > 수급 종가 > 일봉)로 합친다. 반환 열 code, date, close, raw_src."""
    parts = []
    for df, src in ((session, "session"), (sd, "supply_demand"), (bars, "daily_bars")):
        if df is None or df.empty:
            continue
        x = df[["code", "date", "close"]].copy()
        x["date"] = pd.to_datetime(x["date"])
        x["close"] = x["close"].astype(float).abs()
        x = x[x["close"] > 0]
        x["raw_src"] = src
        parts.append(x)
    if not parts:
        return pd.DataFrame(columns=["code", "date", "close", "raw_src"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["code", "date"], keep="first")
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def check_raw_close_canary(df: pd.DataFrame) -> None:
    """원 종가 표가 소급 수정 가격으로 돌아갔으면 시끄럽게 실패한다(카나리아 행이 범위 안에 있을 때만)."""
    code, day, floor = RAW_CLOSE_CANARY
    hit = df[(df["code"] == code) & (pd.to_datetime(df["date"]) == pd.Timestamp(day))]
    if not hit.empty and float(hit["close"].iloc[0]) < floor:
        raise AssertionError(f"원 종가 카나리아 실패: {code} {day} close={hit['close'].iloc[0]} < {floor} — 소급 수정 가격을 읽었다")


def read_raw_close(con: Any, *, start: str, end: "str | None" = None) -> pd.DataFrame:
    """원(미수정) 종가 정문 — 시총(원 종가 × 그 시점 주식수)은 반드시 이것으로 만든다. Postgres."""
    hi = end or "2100-01-01"
    p = {"lo": start, "hi": hi}
    sd = pd.read_sql_query("SELECT code, date, close FROM supply_demand WHERE date >= %(lo)s AND date <= %(hi)s "
                           "AND close IS NOT NULL", con, params=p)
    bars = pd.read_sql_query("SELECT code, date, close FROM daily_bars WHERE date >= %(lo)s AND date <= %(hi)s", con, params=p)
    sess = read_session_bars_agg(con, start=max(start, NXT_START), end=end) if hi >= NXT_START else None
    out = combine_raw_close(sd, bars, sess)
    check_raw_close_canary(out)
    return out
