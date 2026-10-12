"""KRX 정규장 일봉 — 2026-09-14 부터 ``daily_bars`` 에 섞인 **KRX 애프터마켓**(16:00~20:00 연속매매) 체결을 걷어낸다.

**결함(2026-10-10 확인, 2026-10-12 정체 정정):** 2026-09-14 부터 키움 일봉·분봉(ka10081/ka10080)이 KRX 애프터마켓
(16:00~20:00 연속매매, ETF·ETN 제외) 체결을 포함해 준다. 그래서 ``daily_bars.close`` 가 그날 마지막 체결(~19:59)이 됐다.
예: 005930 2026-09-17 일봉 종가 256,000 = 분봉 19:59, KRX 정규장 종가(15:30 단일가) 252,500. 처음엔 NXT 야간 체결로 잘못
알았다(이름 ``NXT_START`` 는 그 흔적 — 호환용 별칭으로 남긴다). ``daily_bars_adjusted``·``daily_bars_raw``·``supply_demand.close``·
``acc_trde_qty`` 도 같은 피드라 함께 오염됐다. 고가·저가·거래량·거래대금도 애프터마켓을 포함한다. 시가(09:00)는 정상이다.

(2026-10-08~09 에 "오진" 이라 적었던 진단 둘 — 분봉 15:30 봉이 공식 종가와 다르다는 설명, 일봉 = 수급 테이블 종가 일치 — 은 틀렸다.
수급 테이블도 같은 키움 피드라 일치가 아무것도 증명하지 않고, 분봉 ``ts`` 는 시간대 없는 KST 라 변환을 한 번 더 걸면 비교가 어긋난다.)

**고치는 법 1 — 분봉:** ``minute_bars``(KST naive) 의 09:00~15:32 봉으로 정규장 OHLCV 를 다시 만들고, 조정 테이블에는 같은 날 조정계수
(행 시가 / 정규장 시가 — 시가는 오염되지 않았다)를 곱해 넣는다. 분봉이 없는 종목·날짜는 원값을 두고 ``krx_patched=False`` 로 표시한다
(분봉은 하루 ~1,400 종목뿐이다 — 나머지 ~1,200 종목은 아래 항등식으로만 종가를 되살린다).

**고치는 법 2 — 정규장 종가 항등식:** ``supply_demand`` 의 ``flu_rt``(등락률 ×100)는 그날 **기준가**(= 전 거래일 정규장 종가) 대비다.
그래서 정규장 종가(t) = 호가 반올림(|close(t+1)| / (1 + flu_rt(t+1)/1e4)). 2026-09-14~10-08 분봉 있는 25,303건 중 99.98% 정확
(어긋나는 것은 권리락·배당락처럼 기준가가 전일 종가와 다른 날). 마지막 거래일은 t+1 이 없어 쓸 수 없다. :func:`regular_close_from_supply_demand`.

정본 수정은 수집기(quant-airflow, 정규장 전용 시세 요청)에서 해야 한다 — 이 모듈은 그 전까지 swing-it 의 읽는 쪽 보정이다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: 일봉에 KRX 애프터마켓(16:00~20:00)이 섞이기 시작한 첫 거래일.
AFTERMARKET_START = "2026-09-14"
#: 옛 이름(처음에 NXT 로 잘못 알았다). 기존 호출부 호환용 별칭 — 값은 같다.
NXT_START = AFTERMARKET_START
#: 정규장 분봉 구간(KST, 양 끝 포함). 종가 단일가는 보통 15:30 봉이지만 **변동성 완화장치로 종가 단일가가 연장되면 15:32 에 찍힌다**
#: (리뷰 `REVIEW_META_GEN2.md` N1 — 15:30 컷이면 그날 종가가 15:19 연속매매가가 됐다). 분봉의 15:35 봉은 09-14 전에도 있던 봉이고
#: (애프터마켓 첫 봉이 아니다 — 옛 주석의 오류), 애프터마켓 봉은 16:00~19:59 다. 15:32 컷은 둘 다 넣지 않는다.
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


def read_session_bars(con: Any, *, start: str = AFTERMARKET_START, end: "str | None" = None) -> pd.DataFrame:
    """DB 분봉 → 정규장 일봉. ``start``~``end``(포함) 날짜."""
    q = ("SELECT code, ts, open, high, low, close, volume, trade_value FROM minute_bars "
         "WHERE ts >= %(lo)s AND ts < %(hi)s AND ts::time BETWEEN %(o)s AND %(c)s")
    hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else "2100-01-01"
    m = pd.read_sql_query(q, con, params={"lo": start, "hi": hi, "o": SESSION_OPEN, "c": SESSION_CLOSE})
    return session_bars_from_minutes(m)


def read_session_bars_agg(con: Any, *, start: str = AFTERMARKET_START, end: "str | None" = None) -> pd.DataFrame:
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


def patch_prices(prices: pd.DataFrame, session: pd.DataFrame, *, since: str = AFTERMARKET_START,
                 adjusted: bool = True) -> pd.DataFrame:
    """롱 가격표(code, date, open, high, low, close, volume, trade_value …)의 ``since`` 이후 행을 정규장 값으로 바꾼다.

    ``adjusted=True`` 면 그 행의 조정계수 = 행의 close / 원 일봉 close 를 알 수 없으므로 **행 자신의 open 비율**을 쓴다:
    시가는 오염되지 않았으므로 factor = 행 open / 정규장 open. 가격 4개에 factor 를 곱하고 거래량·거래대금은 정규장 값 그대로.
    정규장 값이 없는 행은 그대로 두고 ``krx_patched=False``. 반환은 사본, 열 ``krx_patched`` 추가.

    **멱등:** 입력에 이미 ``krx_patched`` 열이 있으면(정문 :func:`swing_it.storage.read_prices` 의 정규장 모드 출력 등) True 행은
    건드리지 않는다 — 조정계수를 다시 구하면 (so·fac)/so ≠ fac 로 끝자리가 흔들리기 때문이다. 없으면 예전과 같다.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    # DB 일봉의 가격·거래량은 정수 열로 오기도 한다 — 정규장 값(실수·조정계수 곱)을 대입하려면 실수여야 한다.
    for col in ("open", "high", "low", "close", "volume", "trade_value"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    prior = df["krx_patched"].fillna(False).to_numpy(bool) if "krx_patched" in df.columns else None
    df["krx_patched"] = False if prior is None else prior
    if session.empty:
        return df
    s = session.copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s.set_index(["code", "date"])
    late = df["date"] >= pd.Timestamp(since)
    if prior is not None:
        late &= ~prior
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


def read_session_bars_any(con: Any, *, start: str, end: "str | None" = None) -> "pd.DataFrame | None":
    """정규장 일봉(분봉 집계) — Postgres 는 :func:`read_session_bars_agg`, sqlite 는 ``minute_bars`` 표가 있으면 그것으로,
    없으면 ``None``(보정할 원천이 없다 — 부르는 쪽이 에러로 바꾼다)."""
    import sqlite3  # noqa: PLC0415

    if not isinstance(con, sqlite3.Connection):
        return read_session_bars_agg(con, start=start, end=end)
    if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='minute_bars'").fetchone() is None:
        return None
    hi = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else "2100-01-01"
    m = pd.read_sql_query("SELECT code, ts, open, high, low, close, volume, trade_value FROM minute_bars WHERE ts >= ? AND ts < ?",
                          con, params=(start, hi))
    return session_bars_from_minutes(m)


def patch_wide(panels: "dict[str, pd.DataFrame]", session: pd.DataFrame, *, since: str = AFTERMARKET_START,
               adjusted: bool = True) -> dict:
    """와이드 패널(날짜 × 종목) 묶음의 ``since`` 이후 행을 :func:`patch_prices` 로 정규장 값으로 바꿔 **제자리에** 꽂는다.

    ``adjusted=True`` 면 ``panels["open"]`` 이 있어야 한다(조정계수 = 행 시가 / 정규장 시가). 반환 = {rows, patched, unpatched}
    (``since`` 이후 값이 하나라도 있는 (날짜, 종목) 수 기준).
    """
    cols = [c for c in ("open", "high", "low", "close", "volume", "trade_value") if c in panels]
    if not cols:
        return {"rows": 0, "patched": 0, "unpatched": 0}
    days = panels[cols[0]].index
    late = days >= pd.Timestamp(since)
    if not late.any():
        return {"rows": 0, "patched": 0, "unpatched": 0}
    if adjusted and "open" not in panels:
        raise ValueError("조정 가격 패널을 보정하려면 open 패널이 필요하다(조정계수 = 행 시가 / 정규장 시가)")
    long = pd.concat({c: panels[c].loc[late].stack(future_stack=True) for c in cols}, axis=1)
    long = long.dropna(how="all").rename_axis(["date", "code"]).reset_index()
    patched = patch_prices(long, session, since=since, adjusted=adjusted)
    for c in cols:
        w = patched.pivot(index="date", columns="code", values=c)
        panels[c].loc[w.index, w.columns] = w.to_numpy()
    n, k = int(len(patched)), int(patched["krx_patched"].sum())
    return {"rows": n, "patched": k, "unpatched": n - k}


# ── 정규장 종가 항등식(supply_demand) ──────────────────────────────────────────────────────────────────────────────
#: 항등식 값이 같은 날 원 종가(애프터마켓 포함)와 이만큼 넘게 어긋나면 버린다 — 분할·병합으로 기준가가 조정된 날(t+1)이다.
IDENTITY_MAX_JUMP = 1.35


def regular_close_from_supply_demand(sd: pd.DataFrame, *, max_jump: float = IDENTITY_MAX_JUMP) -> pd.DataFrame:
    """``supply_demand``(code, date, close, flu_rt) → 정규장 종가(t) = 호가 반올림(|close(t+1)| / (1 + flu_rt(t+1)/1e4)).

    ``flu_rt``(등락률 ×100)는 그날 기준가(= 전 거래일 정규장 종가) 대비라, 다음 행이 전 행의 정규장 종가를 알려 준다. 종목의 **마지막 행**
    (t+1 없음)은 값이 없다. 기준가가 전일 종가와 다른 날(권리락·배당락)은 틀리고(2026-09-14~ 0.02%), 분할·병합은 원 종가 대비
    ``max_jump`` 배 밖이라 버린다. 호가표는 2023-01-25 통합 표(:func:`swing_it.features.korea.tick_size_vec`) — 그 전 날짜에 쓰면 안 된다.
    반환 열 code, date(datetime64), close.
    """
    from swing_it.features.korea import tick_size_vec  # noqa: PLC0415

    x = sd[["code", "date", "close", "flu_rt"]].copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    c = x["close"].astype(float).abs()
    g = x["code"]
    c1 = c.groupby(g).shift(-1)
    f1 = x["flu_rt"].astype(float).groupby(g).shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = (c1 / (1.0 + f1 / 1e4)).to_numpy(float)
        t = tick_size_vec(base)
        reg = np.round(base / t) * t
        ratio = reg / c.to_numpy(float)
    ok = np.isfinite(reg) & (reg > 0) & np.isfinite(ratio) & (ratio <= max_jump) & (ratio >= 1.0 / max_jump)
    return pd.DataFrame({"code": x["code"].to_numpy()[ok], "date": x["date"].to_numpy()[ok], "close": reg[ok]})


def read_regular_close(con: Any, *, start: str, end: "str | None" = None) -> pd.DataFrame:
    """``start``~``end`` 의 항등식 정규장 종가(:func:`regular_close_from_supply_demand`). t+1 행을 얻으려 ``end`` 뒤 한 달을 더 읽는다."""
    import sqlite3  # noqa: PLC0415

    ph = "?" if isinstance(con, sqlite3.Connection) else "%s"
    hi = (pd.Timestamp(end) + pd.Timedelta(days=31)).strftime("%Y-%m-%d") if end else "2100-01-01"
    sd = pd.read_sql_query(f"SELECT code, date, close, flu_rt FROM supply_demand WHERE date >= {ph} AND date <= {ph} "
                           "AND close IS NOT NULL AND flu_rt IS NOT NULL", con, params=(start, hi))
    out = regular_close_from_supply_demand(sd)
    out = out[out["date"] >= pd.Timestamp(start)]
    if end:
        out = out[out["date"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)


# ── 원(미수정) 종가 ───────────────────────────────────────────────────────────────────────────────────────────────
# **결함(2026-10-12 검증 29차):** ``daily_bars.close`` 는 이름과 달리 원 종가가 아니다 — 이후 액면분할·병합을 소급 반영한다
# (005930 2017-03-31 41,200 원 vs 실제 2,060,000 원, 000670 2024-07-31 29,626 vs 314,500, 병합 종목은 5·10 배 위로).
# 이것에 그 시점 주식수를 곱한 시총은 분할 전 대형주를 수십 분의 1 로, 병합 전 소형주를 수 배로 만든다(28세대 VOID, leader21 시총 상위 200).
# 원 종가 = ``supply_demand.close``(키움·네이버, 2017~ 거의 전 종목; 키움은 부호가 붙을 수 있어 절댓값). 없을 때만 일봉으로 돌아가고
# 애프터마켓 이후(``AFTERMARKET_START``~)는 분봉 정규장 종가(원 가격)가 이긴다(``aftermarket_fix=True`` 면 항등식이 그다음). 정본 수정은 quant-airflow(원 일봉 테이블) 소관.
RAW_CLOSE_CANARY = ("005930", "2017-03-31", 1_000_000.0)   # 50:1 분할 전 — 원 종가는 200만 원대


def combine_raw_close(sd: pd.DataFrame, bars: pd.DataFrame, session: "pd.DataFrame | None" = None, *,
                      identity: "pd.DataFrame | None" = None, sd_until: "str | None" = None) -> pd.DataFrame:
    """(code, date, close) 원천을 우선순위로 합친다. 반환 열 code, date, close, raw_src.

    기본(옛 동작, 모의매매 팔이 쓴다): 분봉 정규장 > 수급 종가 > 일봉. ``identity``(:func:`read_regular_close`)·``sd_until`` 을 주면
    분봉 정규장 > **항등식** > 수급 종가(``sd_until`` **전** 날짜만 — 그 뒤 수급 종가는 애프터마켓이 섞였다) > 일봉.
    """
    parts = []
    if sd is not None and sd_until is not None and not sd.empty:
        sd = sd[pd.to_datetime(sd["date"]) < pd.Timestamp(sd_until)]
    for df, src in ((session, "session"), (identity, "identity"), (sd, "supply_demand"), (bars, "daily_bars")):
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


def read_raw_close(con: Any, *, start: str, end: "str | None" = None, aftermarket_fix: bool = False) -> pd.DataFrame:
    """원(미수정) 종가 정문 — 시총(원 종가 × 그 시점 주식수)은 반드시 이것으로 만든다. Postgres.

    ``aftermarket_fix=False``(기본, 옛 동작 — 모의매매 팔 leader21 이 쓴다, 바꾸지 않는다): 분봉 정규장 > 수급 종가 > 일봉. 분봉이 없는
    ~1,200 종목은 2026-09-14 이후 애프터마켓이 섞인 수급 종가를 쓴다. ``True``: 분봉 > 항등식(:func:`read_regular_close`) >
    수급 종가(09-14 전만) > 일봉 — 새 코드는 이쪽. 마지막 거래일처럼 항등식이 없는 분봉 없는 종목은 일봉(오염)으로 떨어지고
    ``raw_src='daily_bars'`` 로 남는다.
    """
    hi = end or "2100-01-01"
    p = {"lo": start, "hi": hi}
    sd = pd.read_sql_query("SELECT code, date, close FROM supply_demand WHERE date >= %(lo)s AND date <= %(hi)s "
                           "AND close IS NOT NULL", con, params=p)
    from swing_it.storage import read_prices  # noqa: PLC0415 — storage 가 이 모듈을 쓰므로 순환을 피한다
    # 정문(가드레일 (g)). combine_raw_close 가 (code, date) 로 중복 제거·정렬하므로 행 순서는 결과에 영향이 없다.
    bars = read_prices(con, session="raw", cols=("code", "date", "close"), table="daily_bars", start=start, end=end)
    sess = read_session_bars_agg(con, start=max(start, AFTERMARKET_START), end=end) if hi >= AFTERMARKET_START else None
    if aftermarket_fix:
        ident = read_regular_close(con, start=max(start, AFTERMARKET_START), end=end) if hi >= AFTERMARKET_START else None
        out = combine_raw_close(sd, bars, sess, identity=ident, sd_until=AFTERMARKET_START)
    else:
        out = combine_raw_close(sd, bars, sess)
    check_raw_close_canary(out)
    return out


def split_consistent_cap(raw: pd.DataFrame, adj: pd.DataFrame, shares: pd.DataFrame, *, asof: str = "date",
                         raw_src: "pd.DataFrame | None" = None) -> "tuple[pd.DataFrame, int]":
    """시총 = 원 종가 × **분할 일관 주식수**(정정 G28-4·5, 검증 30차 D2 방식).

    주식수 표는 주로 연말 DART 행이라 분할·병합 뒤 수개월 늦게 바뀐다 — 원 종가 × 그 주식수는 그 사이 분할 배수만큼 틀린다(삼성전자 2018-05~12
    시총 순위 36~38). 분할 사건 = 원 가격 일 변동 |log| > 0.4(가격제한 ±30% 밖) 인데 연속 수정 가격(``adj``)과 |log| > 0.3 어긋난 날, 배수 =
    수정 변동 / 원 변동. 누적 배수 F 로 주식수 행(그 행 ``date`` 시점)을 사건 기준으로 옮긴 뒤 ``asof`` 열부터 앞으로 채우고 F_t 를 곱한다.
    ``raw_src`` 를 주면 원천이 바뀌는 날(supply_demand ↔ 일봉 대체)은 사건으로 치지 않는다. (f = daily_bars/원 종가 로 배수를 직접 구하는 첫 판은
    daily_bars 가 수집 구간마다 따로 소급돼 있어 분할을 놓쳤다 — 점검 위반 1,013.)

    Args:
        raw: 원 종가 넓은 표(날짜 × 종목), :func:`read_raw_close`.
        adj: 같은 색인·열의 연속 수정 종가(daily_bars_adjusted).
        shares: 긴 표 code, date, shares_outstanding(+ ``asof`` 열).
    Returns:
        (시총 넓은 표, 점검 위반 수 = 원 가격 일 변동 30% 안쪽인데 시총이 3배 넘게 뛴 종목-일).
    """
    rf = raw.ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = (rf / rf.shift(1)).to_numpy(float)
        ra = (adj / adj.shift(1)).to_numpy(float)
        ev = np.isfinite(rr) & np.isfinite(ra) & (np.abs(np.log(rr)) > 0.4) & (np.abs(np.log(rr / ra)) > 0.3)
    if raw_src is not None:
        sd = (raw_src.reindex(index=raw.index, columns=raw.columns) == "supply_demand").to_numpy()
        ev &= sd & np.vstack([np.zeros((1, sd.shape[1]), bool), sd[:-1]])
    f = np.where(ev, ra / rr, 1.0)
    F = np.cumprod(f, axis=0)
    days, codes = raw.index, raw.columns
    sh = shares[shares["shares_outstanding"] > 0].copy()
    sh["date"] = pd.to_datetime(sh["date"])
    sh["k"] = pd.to_datetime(sh[asof]).fillna(sh["date"]) if asof in sh else sh["date"]
    pos = days.searchsorted(sh["date"].to_numpy(), side="right") - 1
    ci = codes.get_indexer(sh["code"])
    ok = (pos >= 0) & (ci >= 0)
    fb = np.ones(len(sh))
    fb[ok] = F[pos[ok], ci[ok]]
    sh["s0"] = sh["shares_outstanding"].to_numpy(float) / fb
    S = sh.pivot_table(index="k", columns="code", values="s0", aggfunc="last")
    S = S.reindex(index=S.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
    cap = raw * S * F
    with np.errstate(divide="ignore", invalid="ignore"):
        jump = (cap / cap.shift(1)).abs()
        r1 = (raw / raw.shift(1)).abs()
    bad = int((((jump > 3) | (jump < 1 / 3)) & (r1 < 1.3) & (r1 > 1 / 1.3)).sum().sum())
    return cap, bad


def factor_cap(raw: pd.DataFrame, adj: pd.DataFrame, shares: pd.DataFrame, *, asof: str = "date") -> "tuple[pd.DataFrame, pd.DataFrame]":
    """시총 = 원 종가 × 분할 일관 주식수, 누적계수를 **원 일봉에서 재구성한 수정 가격**(daily_bars_adj_v2 = 원 × 누적 조정계수)에서 직접 읽는다.

    F = adj / raw(앞으로 채움). 주식수 행은 그 행 ``date`` 의 F 로 나눠 사건 기준으로 옮긴 뒤 ``asof`` 열부터 앞으로 채우고 F_t 를 곱한다 —
    :func:`split_consistent_cap` 과 같은 식이지만 점프 검출이 없다. 이음매 있는 daily_bars_adjusted 에 쓰면 안 된다(첫 판 위반 1,013, 정정 G28-5).

    Returns:
        (시총 넓은 표, F 넓은 표).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (adj / raw).where(lambda x: np.isfinite(x) & (x > 0)).ffill()
    days, codes = raw.index, raw.columns
    Fv = F.reindex(columns=codes).to_numpy(float)
    sh = shares[shares["shares_outstanding"] > 0].copy()
    sh["date"] = pd.to_datetime(sh["date"])
    sh["k"] = pd.to_datetime(sh[asof]).fillna(sh["date"]) if asof in sh else sh["date"]
    pos = days.searchsorted(sh["date"].to_numpy(), side="right") - 1
    ci = codes.get_indexer(sh["code"])
    ok = (pos >= 0) & (ci >= 0)
    # 행 날짜에 F 없음(데이터 시작 전·상장 전) → 그 종목의 첫 유한 F. v2 의 F 는 최신일이 1 이라 1.0 대체는 틀린다(검증 29 C1:
    # 042700 시총 1/5, 053110 20배).
    first = F.reindex(columns=codes).bfill().iloc[0].to_numpy(float)
    fb = np.full(len(sh), np.nan)
    fb[ok] = Fv[pos[ok], ci[ok]]
    pre = (ci >= 0) & ~np.isfinite(fb)
    fb[pre] = first[ci[pre]]
    fb = np.where(np.isfinite(fb), fb, 1.0)                    # 그 종목에 F 가 아예 없음 → 시총도 NaN 이라 영향 없음
    sh["s0"] = sh["shares_outstanding"].to_numpy(float) / fb
    S = sh.pivot_table(index="k", columns="code", values="s0", aggfunc="last")
    S = S.reindex(index=S.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
    return raw * S * F, F
