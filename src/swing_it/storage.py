"""Storage layer (read side) — sqlite or Postgres/TimescaleDB.

Collectors (and their write-side upsert helpers) now live in
quant-airflow/collectors/storage.py — an intentionally independent copy,
not a shared package. This module keeps only what strategies/features read:
``connect()``/``default_db_path()`` (also used for the local sqlite dev
fallback) and ``market_cap_asof()``/``market_cap_asof_bulk()``.
``connect()`` dispatches on the connection string: a
``postgresql://``/``postgres://`` DSN opens Postgres (psycopg2, imported
lazily so sqlite-only use never needs it installed); anything else opens a
local sqlite file exactly as before.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

_PG_PREFIXES = ("postgresql://", "postgres://")


def load_env_db() -> str:
    """**명시적 opt-in** — CWD 의 ``.env`` 에서 KR_QUANT_DB 를 읽어 환경에 실어준다.

    러너마다 복붙돼 있던 .env 파서의 유일한 본체다. 부르는 쪽이 명시적으로 부를 때만
    동작한다 — :func:`db_default` 는 이걸 부르지 않는다. 그래야 KR_QUANT_DB 를
    export 하지 않은 셸이 여전히 로컬 sqlite 폴백을 **보장**받는다(공유 운영 DB 로
    조용히 붙어버리면 안 된다).

    값을 ``os.environ`` 에 되실어 이후 ``db_default()`` 와 하위 프로세스가 본다.
    설정된 게 없으면 ``""``.
    """
    v = os.environ.get("KR_QUANT_DB")
    if v:
        return v
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("KR_QUANT_DB"):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                if v:
                    os.environ["KR_QUANT_DB"] = v
                    return v
    return ""


def db_default() -> str:
    """CLI ``--db`` flag default: ``KR_QUANT_DB`` env var if set, else local sqlite.

    Lets an analysis-only checkout point at the shared TimescaleDB once (via
    ``.env``/``export KR_QUANT_DB=postgresql://...``) instead of passing
    ``--db`` on every command. ``.env`` 을 스스로 읽지는 않는다 — :func:`load_env_db`
    를 명시적으로 부른 호출자만 그 경로를 탄다.
    """
    return os.environ.get("KR_QUANT_DB") or str(default_db_path())

# ka10059 (투자자기관별종목별) net-buy fields → DB columns.
# Order matters: it defines the column order for ``supply_demand`` inserts.
INVESTOR_COLUMNS: dict[str, str] = {
    "individual": "ind_invsr",   # 개인
    "foreign_": "frgnr_invsr",   # 외국인
    "institution": "orgn",       # 기관계
    "fnnc_invt": "fnnc_invt",    # 금융투자
    "insrnc": "insrnc",          # 보험
    "invtrt": "invtrt",          # 투신
    "bank": "bank",              # 은행
    "penfnd_etc": "penfnd_etc",  # 연기금 등
    "samo_fund": "samo_fund",    # 사모펀드
    "natn": "natn",              # 국가
    "etc_corp": "etc_corp",      # 기타법인
}

_INVESTOR_COL_DDL = ",\n            ".join(f"{c} INTEGER" for c in INVESTOR_COLUMNS)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS stocks (
    code   TEXT PRIMARY KEY,
    name   TEXT,
    market TEXT,
    sector TEXT,
    kind   TEXT
);
CREATE TABLE IF NOT EXISTS supply_demand (
    code         TEXT NOT NULL,
    date         TEXT NOT NULL,
    close        INTEGER,
    flu_rt       REAL,
    acc_trde_qty INTEGER,
    {_INVESTOR_COL_DDL},
    -- kiwoom = 11개 분류 전체. naver = 폐지 종목 부분 백필(기관·외국인 순매매만,
    -- 나머지는 NULL — 0 은 '순매매 없음'이고 NULL 은 '모름'이라 구분해야 한다).
    -- 외국인 정의도 다르다: 네이버 값은 4개 분류 합이 0이 되도록 맞춘 수치다.
    source       TEXT NOT NULL DEFAULT 'kiwoom',
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_sd_date ON supply_demand(date);
-- source: 'kiwoom' = 상장 종목(ka10081, trade_value 는 보고된 거래대금),
--         'naver'  = 상장폐지 종목 백필(siseJson, trade_value 는 close*volume/1e6
--                    근사 — 실측 오차 중앙값 0.7%, p95 3.6%).
-- 폐지 종목이 이 테이블에 들어오는 이유는 생존편향 때문이다: 수집 소스가 현재 상장
-- 종목만 돌려주므로, 그냥 두면 백테스트가 살아남은 회사만 보고 성적을 잰다.
CREATE TABLE IF NOT EXISTS daily_bars (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        INTEGER,
    high        INTEGER,
    low         INTEGER,
    close       INTEGER,
    volume      INTEGER,
    trade_value INTEGER,
    source      TEXT NOT NULL DEFAULT 'kiwoom',
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_db_date ON daily_bars(date);
CREATE TABLE IF NOT EXISTS short_selling (
    code            TEXT NOT NULL,
    date            TEXT NOT NULL,
    close           INTEGER,
    volume          INTEGER,
    short_qty       INTEGER,   -- 당일 공매도 수량 (shrts_qty)
    short_balance   INTEGER,   -- 공매도 잔고 수량 (ovr_shrts_qty)
    short_ratio     REAL,      -- 공매도 비중 % (trde_wght)
    short_avg_price INTEGER,   -- 공매도 평균가 (shrts_avg_pric)
    short_value     INTEGER,   -- 공매도 거래대금 (shrts_trde_prica)
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_ss_date ON short_selling(date);
CREATE TABLE IF NOT EXISTS credit_balance (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    close       INTEGER,
    new_qty     INTEGER,   -- 신규 신용매수 (new)
    repay_qty   INTEGER,   -- 상환 (rpya)
    balance_qty INTEGER,   -- 신용잔고 수량 (remn)
    balance_amt INTEGER,   -- 신용잔고 금액 (amt)
    balance_rt  REAL,      -- 신용잔고율 % (remn_rt)
    credit_rt   REAL,      -- 신용비율 % (shr_rt)
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_cb_date ON credit_balance(date);
CREATE TABLE IF NOT EXISTS sector_index (
    code        TEXT NOT NULL,  -- 업종코드 (001=KOSPI 종합, 101=KOSDAQ 종합 등)
    name        TEXT,
    date        TEXT NOT NULL,
    open        INTEGER,
    high        INTEGER,
    low         INTEGER,
    close       INTEGER,
    volume      INTEGER,
    trade_value INTEGER,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_si_date ON sector_index(date);
CREATE TABLE IF NOT EXISTS shares_outstanding_history (
    code               TEXT NOT NULL,
    date               TEXT NOT NULL,   -- 기준일(보고서 기준일). as-of 키가 아니다 — dart 행은 이보다 중앙값 82일 늦게 알려졌다
    shares_outstanding INTEGER,  -- sqlite INTEGER is dynamically 64-bit already;
    source             TEXT,     -- kiwoom·dart·krx
    knowledge_date     TEXT,     -- 알게 된 날(dart 행). kiwoom 행은 NULL — as-of 키 = coalesce(knowledge_date, date), read_shares
    PRIMARY KEY (code, date)     -- Postgres side (init_timescale.sql) must use BIGINT, not INTEGER(32bit) — 삼성전자(58억주) overflows it
);
CREATE INDEX IF NOT EXISTS idx_sh_date ON shares_outstanding_history(date);
-- 정정공시는 기존 행을 덮어쓰지 않고 새 버전으로 쌓인다(PK에 knowledge_date 포함).
-- 그래서 (code, period)당 행이 여럿일 수 있다 — 읽는 쪽은 반드시 as-of로 한 버전만
-- 골라야 하며, 그냥 SELECT 하면 패널이 조용히 중복된다. swing_it.storage.read_earnings()
-- 를 쓸 것.
CREATE TABLE IF NOT EXISTS earnings (
    code            TEXT NOT NULL,
    period          TEXT NOT NULL,   -- e.g. '2020Q1'
    avail_date      TEXT,            -- lookahead-safe availability date (period-end + filing lag)
    knowledge_date  TEXT NOT NULL,   -- 이 값을 알게 된 날(수집일) — 정정공시는 새 행
    netinc          REAL,
    netinc_prior    REAL,
    revenue         REAL,
    revenue_prior   REAL,
    op_income       REAL,
    op_income_prior REAL,
    PRIMARY KEY (code, period, knowledge_date)
);
CREATE INDEX IF NOT EXISTS idx_earnings_avail_date ON earnings(avail_date);
CREATE INDEX IF NOT EXISTS idx_earnings_asof ON earnings(code, period, knowledge_date DESC);
-- DART 공시 목록(quant-airflow). kind = periodic(정기보고서)·prelim(잠정실적)·pl_change(손익구조 변경).
-- 실적을 **실제로 알게 된 날**은 earnings.avail_date(제출 마감일 고정값)가 아니라 여기 rcept_dt 다 — read_earnings(with_known=True).
CREATE TABLE IF NOT EXISTS earnings_disclosures (
    code      TEXT NOT NULL,
    period    TEXT NOT NULL,
    kind      TEXT NOT NULL,
    rcept_dt  TEXT NOT NULL,
    rcept_no  TEXT,
    report_nm TEXT
);
CREATE TABLE IF NOT EXISTS consensus (
    code         TEXT NOT NULL,
    date         TEXT NOT NULL,   -- 스냅샷 수집일 (오늘)
    target_mean  REAL,            -- 목표주가 평균
    recomm_mean  REAL,            -- 투자의견 평균 (1~5, 5=강력매수)
    base_date    TEXT,            -- 컨센서스 기준일(네이버 createDate)
    fwd_eps      REAL,            -- 향후 컨센서스 EPS
    prev_eps     REAL,            -- 직전 확정 EPS
    est_year     TEXT,            -- fwd_eps가 가리키는 연도(예: '202612')
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_consensus_date ON consensus(date);
CREATE TABLE IF NOT EXISTS daily_bars_adjusted (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,   -- price_adjust.adjust_prices()의 back-adjust 배수 적용 후이므로
    high        REAL,   -- daily_bars(원자료, INTEGER)와 달리 REAL — 분할비율이 실수라 정수로
    low         REAL,   -- 안 떨어짐(예: 1주→4주 분할이면 종가가 1/4배가 됨)
    close       REAL,
    volume      INTEGER,      -- 기본은 미조정 원본 거래량 그대로(adjust_volume=False)
    trade_value INTEGER,      -- 거래대금은 가격조정과 무관(가격×수량이 아니라 원 보고값)
    source      TEXT NOT NULL DEFAULT 'kiwoom',  -- daily_bars.source 전파(근사 거래대금 식별)
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_dba_date ON daily_bars_adjusted(date);
CREATE TABLE IF NOT EXISTS delisted_stocks (
    code            TEXT NOT NULL,
    name            TEXT,
    market          TEXT,
    last_trade_date TEXT,   -- daily_bars 기준 마지막 거래일(상장폐지일 근사), 이력 없으면 NULL
    PRIMARY KEY (code)
);
"""


def default_db_path() -> Path:
    """Default DB location: ``<repo>/data/swing_it.db`` (gitignored)."""
    return Path(__file__).resolve().parents[2] / "data" / "swing_it.db"


def connect(db_path: str | Path | None = None) -> Any:
    """Open a connection with row access.

    ``db_path`` starting with ``postgresql://``/``postgres://`` opens Postgres
    (e.g. TimescaleDB) via psycopg2. Anything else is treated as a sqlite file
    path (default: ``<repo>/data/swing_it.db``, dirs created as needed).
    """
    if isinstance(db_path, str) and db_path.startswith(_PG_PREFIXES):
        import psycopg2  # noqa: PLC0415 — optional dep, only needed for this path

        con = psycopg2.connect(db_path)
        # Schema (tables, hypertables, compression policy) is provisioned by
        # quant-airflow/sql/init_timescale.sql, not here — init_db() only
        # applies to the sqlite path.
        return con

    path = Path(db_path) if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    init_db(con)
    return con


# 기존 sqlite 파일에 나중에 추가된 컬럼 — {테이블: [(컬럼, DDL 조각)]}.
# ``CREATE TABLE IF NOT EXISTS`` 는 이미 있는 테이블을 고치지 않으므로, 마이그레이션
# 이전에 만들어진 로컬 DB 는 컬럼이 없는 채로 남는다. 그 상태에서 새 인덱스
# (idx_earnings_asof 는 knowledge_date 를 참조)를 만들면 init_db 가 통째로 죽는다 —
# Postgres 없이 쓰는 경로가 막힌다(실측). Postgres 쪽은 sql/migrations/ 가 담당한다.
_SQLITE_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "earnings": [("knowledge_date", "TEXT")],
    "daily_bars": [("source", "TEXT NOT NULL DEFAULT 'kiwoom'")],
    "daily_bars_adjusted": [("source", "TEXT NOT NULL DEFAULT 'kiwoom'")],
    "shares_outstanding_history": [("source", "TEXT"), ("knowledge_date", "TEXT")],
}


def _add_missing_columns(con: sqlite3.Connection) -> None:
    """기존 파일에 빠진 컬럼을 채운다. 새 파일에는 no-op."""
    for table, cols in _SQLITE_ADDED_COLUMNS.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}  # noqa: S608 — 모듈 상수
        if not have:
            continue                      # 테이블 자체가 없다 = 새 DB, SCHEMA 가 만든다
        for name, ddl in cols:
            if name not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")  # noqa: S608 — 모듈 상수
    con.commit()


def init_db(con: sqlite3.Connection) -> None:
    _add_missing_columns(con)
    con.executescript(SCHEMA)
    con.commit()


def _is_pg(con: Any) -> bool:
    return not isinstance(con, sqlite3.Connection)


# swing_it.price_adjust.rebuild_adjusted_table() writes daily_bars_adjusted —
# the one write path kept here (not moved to quant-airflow) because
# price_adjust.py's split-detection logic is imported in-process by the backtest
# strategies, so the module as a whole stays in swing-it;
# weekly_price_adjust.py's DAG task still invokes it via
# `python -m swing_it.price_adjust --rebuild-db` (PYTHONPATH-based, no pip
# install needed) rather than through quant-airflow/collectors.
DAILY_BAR_COLUMNS: list[str] = [
    "code", "date", "open", "high", "low", "close", "volume", "trade_value",
]


def _upsert(
    con: Any,
    table: str,
    cols: list[str],
    records: list[tuple],
    *,
    pk_cols: tuple[str, ...] = ("code", "date"),
) -> int:
    if not records:
        return 0
    if _is_pg(con):
        # psycopg2 의 executemany 는 행마다 INSERT 문을 한 번씩 왕복시킨다 —
        # rebuild_adjusted_table() 처럼 570만 행을 upsert 하면 몇 시간이 걸린다
        # (실측: 40분 넘게 돌고도 진행 중). execute_values 는 한 문장에 여러 VALUES
        # 튜플을 실어 보내 같은 일을 수십 배 빠르게 끝낸다. 수집기 쪽
        # (quant-airflow/collectors/storage.py)은 이미 이 방식이다.
        import psycopg2.extras

        update_cols = [c for c in cols if c not in pk_cols]
        set_clause = ",".join(f"{c}=EXCLUDED.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table}({','.join(cols)}) VALUES %s "
            f"ON CONFLICT ({','.join(pk_cols)}) DO UPDATE SET {set_clause}"
        )
        template = "(" + ",".join(["%s"] * len(cols)) + ")"
        try:
            with con.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, records, template=template, page_size=1000)
        except Exception:
            con.rollback()
            raise
    else:
        placeholders = ",".join(["?"] * len(cols))
        sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({placeholders})"
        con.executemany(sql, records)
    con.commit()
    return len(records)


# 조정가 테이블은 원본의 source 를 그대로 실어 나른다 — 백테스트가 읽는 건 이쪽이라,
# "이 행의 trade_value 는 근사치"라는 사실이 여기 없으면 소비되는 자리에서 알 수 없다.
ADJUSTED_BAR_COLUMNS: list[str] = [*DAILY_BAR_COLUMNS, "source"]


def upsert_daily_bars_adjusted(con: Any, records: list[tuple]) -> int:
    """Insert/replace daily_bars_adjusted rows (tuples ordered by ADJUSTED_BAR_COLUMNS)."""
    return _upsert(con, "daily_bars_adjusted", ADJUSTED_BAR_COLUMNS, records)


_EARNINGS_READ_COLS = (
    "code", "period", "avail_date", "netinc", "netinc_prior",
    "revenue", "revenue_prior", "op_income", "op_income_prior",
)


def read_earnings(con: Any, *, asof: str | None = None, cols: "tuple[str, ...] | None" = None,
                  all_versions: bool = False, avail_from: "str | None" = None, parse_dates: bool = False,
                  with_known: bool = False, calendar: "Any | None" = None):
    """Earnings as a long frame with **one row per (code, period)** — as-of ``asof``.

    ``earnings`` is versioned: a DART restatement lands as a new row keyed by
    ``knowledge_date`` rather than overwriting the original. A plain
    ``SELECT * FROM earnings`` therefore returns several rows per (code, period)
    once restatements exist, which silently duplicates every downstream panel.
    This picks the newest version whose ``knowledge_date <= asof``.

    Args:
        con: Open connection (sqlite3 or psycopg2).
        asof: Ceiling on ``knowledge_date`` — a single instant. ``None`` means
            latest-known.
        cols: Columns to return (default :data:`_EARNINGS_READ_COLS`).
        all_versions: 접지 말고 모든 버전을 그대로 준다(``knowledge_date`` 포함).

            **백테스트는 이쪽을 써야 한다.** 스칼라 ``asof`` 는 "한 시점 기준"만
            표현할 수 있는데, 백테스트가 필요한 건 "각 바 t 시점 기준"이다. 버전
            선택은 날짜별로 달라져야 하므로 읽는 단계가 아니라
            :func:`swing_it.features.fundamentals.earnings_yoy_panel` 의
            ``knowledge_col`` 로 넘겨 as-of 조인에 함께 태운다.

            ``asof`` 없이 접어서 쓰면(기본값) 최신 정정본이 과거 날짜 셀에 들어가
            조용한 룩어헤드가 된다 — 터지지 않고 성과만 좋아지는 종류다.
        avail_from: ``all_versions`` 와 함께 — ``avail_date >= avail_from`` 행만(SQL 에서 자른다).
        parse_dates: ``avail_date`` 를 datetime64 로(``pd.read_sql(parse_dates=[...])`` 와 같은 경로).
        with_known: **실적 as-of 열 두 개**를 덧붙인다(:func:`_attach_known_dates`) —
            ``known_date`` = max(avail_date, 그 분기 정기보고서 첫 rcept_dt) **다음 거래일**(값을 쓸 수 있는 첫날),
            ``event_date`` = 그 분기 공시(잠정·정정·정기 아무거나) 첫 rcept_dt 다음 거래일(서프라이즈·PEAD 사건일).
            ``avail_date`` 는 제출 마감일 고정값이라 정기보고서 17.7% 가 그보다 늦게 나왔다 — as-of 키로 쓰면 미래 정보다
            (정정 G19-1). 2019-02-28 전에는 정기보고서 날짜가 없어 known 은 avail_date 로 돌아가고 event_date 는 NaT.
        calendar: ``with_known`` 의 거래일 달력(DatetimeIndex). 없으면 ``daily_bars_adjusted`` 의 날짜.

    Returns a pandas DataFrame.
    """
    import pandas as pd

    if with_known:
        want = tuple(cols or _EARNINGS_READ_COLS)
        need = tuple(c for c in ("code", "period", "avail_date") if c not in want)
        df = read_earnings(con, asof=asof, cols=want + need, all_versions=all_versions, avail_from=avail_from,
                           parse_dates=parse_dates)
        df = _attach_known_dates(con, df, calendar=calendar)
        return df.drop(columns=list(need)) if need else df
    use = tuple(cols or _EARNINGS_READ_COLS)
    sel = ", ".join(use)
    pdates = ["avail_date"] if parse_dates and "avail_date" in use else None
    if all_versions:
        if asof is not None:
            raise ValueError("all_versions 와 asof 는 함께 쓸 수 없다 — "
                             "버전 선택을 날짜별로 하려고 전부 넘기는 것이다")
        if avail_from is not None:
            return pd.read_sql_query(f"SELECT {sel} FROM earnings WHERE avail_date >= {_ph(con)}",  # noqa: S608 — 모듈 상수
                                     con, params=(avail_from,), parse_dates=pdates)
        return pd.read_sql_query(f"SELECT {sel} FROM earnings", con, parse_dates=pdates)  # noqa: S608 — 모듈 상수
    if avail_from is not None:
        raise ValueError("avail_from 은 all_versions=True 와 함께만 쓴다")
    if _is_pg(con):
        where = "WHERE knowledge_date <= %s " if asof else ""
        sql = (  # noqa: S608 — column names come from a module constant, not user input
            f"SELECT DISTINCT ON (code, period) {sel} FROM earnings {where}"
            "ORDER BY code, period, knowledge_date DESC"
        )
        params: tuple = (asof,) if asof else ()
    else:
        # sqlite has no DISTINCT ON — pick the max knowledge_date per key instead.
        cap = "AND x.knowledge_date <= ? " if asof else ""
        sql = (  # noqa: S608 — same
            f"SELECT {sel} FROM earnings e WHERE e.knowledge_date = ("
            "SELECT MAX(x.knowledge_date) FROM earnings x "
            f"WHERE x.code = e.code AND x.period = e.period {cap})"
        )
        params = (asof,) if asof else ()
    return pd.read_sql_query(sql, con, params=params or None, parse_dates=pdates)


def _attach_known_dates(con: Any, df, *, calendar=None):
    """``known_date``·``event_date`` 열(:func:`read_earnings` ``with_known``). 규칙은 :mod:`swing_it.features.eps_asof` 와 같다."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    disc = pd.read_sql_query("SELECT code, period, kind, rcept_dt FROM earnings_disclosures", con)
    disc["rcept_dt"] = pd.to_datetime(disc["rcept_dt"].astype(str))
    per = (disc[disc["kind"] == "periodic"].groupby(["code", "period"])["rcept_dt"].min().rename("_per"))
    ev = disc.groupby(["code", "period"])["rcept_dt"].min().rename("_ev")
    if calendar is None:
        cur = con.cursor()
        cur.execute(f"SELECT DISTINCT date FROM {PRICE_TABLE} ORDER BY date")  # noqa: S608 — 모듈 상수
        calendar = [r[0] for r in cur.fetchall()]
    days = pd.DatetimeIndex(pd.to_datetime([str(d)[:10] for d in calendar])).sort_values()
    out = df.copy()
    key = pd.MultiIndex.from_arrays([out["code"].astype(str), out["period"].astype(str)])
    per_dt = per.reindex(key).to_numpy()
    ev_dt = ev.reindex(key).to_numpy()
    avail = pd.to_datetime(out["avail_date"].astype(str).replace({"None": None, "nan": None, "NaT": None})).to_numpy()
    known = pd.DataFrame({"a": avail, "p": per_dt}).max(axis=1).to_numpy()

    def nxt(x):
        x = np.asarray(x, dtype="datetime64[ns]")
        ok = ~np.isnat(x)
        i = days.searchsorted(x[ok], side="right")
        res = np.full(len(x), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        inside = i < len(days)
        res[np.flatnonzero(ok)[inside]] = days.to_numpy()[i[inside]]
        return res

    out["known_date"] = nxt(known)
    out["event_date"] = nxt(ev_dt)
    return out


PRICE_TABLE = "daily_bars_adjusted"   # 분할조정. 백테스트는 원자료(daily_bars) 금지.
_PRICE_READ_COLS = ("code", "date", "open", "high", "low", "close", "volume", "trade_value")

#: 가격 정문의 세션 모드. ``"regular"``(기본) = 2026-09-14~ 행을 KRX 정규장 값으로 보정(:mod:`swing_it.krx_session`, 분봉 09:00~15:32),
#: ``"raw"`` = 테이블 값 그대로(애프터마켓 16:00~20:00 체결 포함). 사전등록 재현·자체 보정 경로는 ``"raw"`` 를 **명시**한다.
SESSIONS = ("regular", "raw")
#: 보정 대상 가격 테이블 → 조정 테이블인가(분봉 조정계수 = 행 시가 / 정규장 시가, 항등식 배율 = 행 종가 / 수급 종가).
#: 이 밖의 표(``daily_bars_raw``·``daily_bars_adj_v2`` 등)는 보정하지 않는다 — ``session="regular"`` 를 명시하면 :class:`ValueError`.
_SESSION_TABLES = {"daily_bars": False, "daily_bars_adjusted": True}
_SESSION_COLS = ("open", "high", "low", "close", "volume", "trade_value")

#: 정규장 모드에서 **고칠 수 없는 값**의 처리 — 세 정문(:func:`read_prices`·:func:`read_prices_wide`·:func:`read_supply_demand`) 공통.
#: ``"nan"``(기본) = NaN 으로 비운다(오염값을 남기지 않는다). ``"raw"`` = 테이블 원값(애프터마켓 포함)을 그대로 두고 ``krx_src`` 로 표시한다.
#: 정책 하나로 통일했다(2026-10-13 리뷰 MAJOR 1 — 전에는 가격은 원값, 수급은 NaN 이라 두 정문이 조용히 달랐다).
UNPATCHED = ("nan", "raw")
#: 행별 보정 원천(``krx_src`` 열, 정규장 모드에서 2026-09-14~ 행이 있을 때만 붙는다):
#:   ``"pre"`` 09-14 전(보정 불필요) · ``"session"`` 분봉 정규장(모든 열) · ``"identity"`` 정규장 종가 항등식(종가·등락률만 — 고가·저가·
#:   거래량·거래대금은 고칠 원천이 없어 :data:`UNPATCHED` 정책) · ``"none"`` 원천 없음(마지막 거래일의 분봉 없는 종목 등, 모든 오염 열이 정책).
#: 시가는 오염되지 않아 언제나 원값이다. ``krx_patched`` = 그 행의 요청 열이 전부 정규장 값인가.
KRX_SRC = ("pre", "session", "identity", "none")


class SessionUnavailable(RuntimeError):
    """정규장 보정이 필요한데 할 수 없다(sqlite 에 분봉 표 없음, strict 모드의 미보정 행)."""


def _check_session(session: "str | None", unpatched: str = "nan") -> None:
    if session is not None and session not in SESSIONS:
        raise ValueError(f"session 은 {SESSIONS} 중 하나 — {session!r}")
    if unpatched not in UNPATCHED:
        raise ValueError(f"unpatched 는 {UNPATCHED} 중 하나 — {unpatched!r}")


def _resolve_session(session: "str | None", table: str) -> str:
    """``session=None``(기본) → 보정 대상 표면 ``"regular"``, 아니면 ``"raw"``(경고 — 그 표는 보정 경로가 없다).
    보정 대상이 아닌 표에 ``"regular"`` 를 **명시**하면 조용히 원값을 주지 않고 :class:`ValueError`."""
    if table in _SESSION_TABLES:
        return session or "regular"
    if session == "regular":
        raise ValueError(f"{table} 은 정규장 보정 대상이 아니다({tuple(_SESSION_TABLES)} 만) — session='regular' 로 읽으면 애프터마켓이 섞인 "
                         "원값이 그대로 나온다. 원값이면 session='raw' 를 명시하라.")
    if session is None:
        import warnings  # noqa: PLC0415

        warnings.warn(f"{table} 은 정규장 보정 대상이 아니라 원값(session='raw')으로 읽는다 — session='raw' 를 명시하라.", stacklevel=3)
    return "raw"


def _session_needs_open(table: str, use: "tuple[str, ...]", session: str, end: "str | None") -> bool:
    """조정 테이블을 정규장으로 보정하려면 조정계수용 open 이 필요하다 — 요청에 없으면 안에서 더 읽고 버린다."""
    from swing_it.krx_session import AFTERMARKET_START  # noqa: PLC0415

    return (session == "regular" and _SESSION_TABLES.get(table, False) and "open" not in use
            and any(c in _SESSION_COLS for c in use) and (end is None or end >= AFTERMARKET_START))


def _apply_regular_session(con: Any, df, *, table: str, strict: bool, unpatched: str = "nan"):
    """긴 가격표의 2026-09-14~ 행을 정규장 값으로. 결과 ``attrs["krx_session"]`` 에 {rows, patched, unpatched, by_src, policy}.
    그 날짜 행이 없으면 **입력을 그대로**(dtype 도) 돌려준다 — 그래서 09-14 전 창은 ``"raw"`` 와 바이트 같다.

    원천 순서: ① 분봉 정규장(:func:`swing_it.krx_session.patch_prices` — 모든 열) ② 분봉이 없으면 종가만 정규장 종가 항등식
    (:func:`swing_it.krx_session.read_regular_close`; 조정표는 × 행 종가 / |수급 종가| — 같은 애프터마켓 마지막 체결의 조정 배율)
    ③ 그래도 못 고친 값(②의 고가·저가·거래량·거래대금, 원천 없는 행의 모든 오염 열)은 ``unpatched`` 정책(:data:`UNPATCHED`).
    보정한 경우 ``krx_patched``·``krx_src``(:data:`KRX_SRC`) 열이 붙고 가격 열은 float 가 된다(``date`` 열은 입력 그대로)."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from swing_it.krx_session import AFTERMARKET_START, patch_prices, read_regular_close, read_session_bars_any  # noqa: PLC0415

    info = {"session": "regular", "since": AFTERMARKET_START, "rows": 0, "patched": 0, "unpatched": 0}
    if (table not in _SESSION_TABLES or not {"code", "date"} <= set(df.columns)
            or not any(c in df.columns for c in _SESSION_COLS)):
        df.attrs["krx_session"] = info
        return df
    d = pd.to_datetime(df["date"])
    late = (d >= pd.Timestamp(AFTERMARKET_START)).to_numpy()
    if not late.any():
        df.attrs["krx_session"] = info
        return df
    lo, hi = d[late].min().strftime("%Y-%m-%d"), d[late].max().strftime("%Y-%m-%d")
    sess = read_session_bars_any(con, start=lo, end=hi)
    if sess is None:
        raise SessionUnavailable(
            f"{table} 에 {AFTERMARKET_START} 이후 행 {int(late.sum())}개가 있는데 이 DB 에 minute_bars 가 없어 정규장으로 보정할 수 "
            "없다(그 날짜 일봉에는 KRX 애프터마켓 16:00~20:00 체결이 섞였다). 원값이 필요하면 session='raw' 로 명시하라.")
    adjusted = _SESSION_TABLES[table]
    out = patch_prices(df, sess, since=AFTERMARKET_START, adjusted=adjusted)
    out["date"] = df["date"].to_numpy()                    # patch_prices 가 datetime 으로 바꾼 날짜 열을 입력 모양으로 되돌린다
    src = np.where(late, "none", "pre").astype(object)
    src[late & out["krx_patched"].to_numpy(bool)] = "session"
    need = src == "none"
    if "close" in out.columns and need.any():
        ident = read_regular_close(con, start=lo, end=hi, with_raw=True)
        if len(ident):
            i = ident.assign(code=ident["code"].astype(str), date=pd.to_datetime(ident["date"])).set_index(["code", "date"])
            key = pd.MultiIndex.from_arrays([out.loc[need, "code"].astype(str).to_numpy(), d[need].to_numpy()])
            ic = i["close"].reindex(key).to_numpy(float)
            if adjusted:                                   # 조정 배율 = 행 종가 / |수급 종가|(둘 다 같은 애프터마켓 마지막 체결)
                rc = i["sd_close"].reindex(key).to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ic = ic * (out.loc[need, "close"].to_numpy(float) / np.where(rc > 0, rc, np.nan))
            ok = np.isfinite(ic) & (ic > 0)
            rows = np.flatnonzero(need)[ok]
            out.iloc[rows, out.columns.get_loc("close")] = ic[ok]
            src[rows] = "identity"
    dirty = [c for c in _SESSION_COLS if c in out.columns and c != "open"]   # 시가는 오염되지 않았다
    rest = [c for c in dirty if c != "close"]
    if unpatched == "nan":
        for c in dirty:
            m = (src == "none") | ((src == "identity") & (c != "close"))
            if m.any():
                out.loc[m, c] = np.nan
    out["krx_patched"] = (src == "session") | ((src == "identity") & (not rest))
    out["krx_src"] = src
    n, k = int(late.sum()), int(out.loc[late, "krx_patched"].sum())
    info.update(rows=n, patched=k, unpatched=n - k, policy=unpatched,
                by_src={x: int((src == x).sum()) for x in ("session", "identity", "none")})
    out.attrs["krx_session"] = info
    if strict and n > k:
        raise SessionUnavailable(f"{table} {AFTERMARKET_START} 이후 {n}행 중 {n - k}행은 정규장 값으로 다 고치지 못했다(strict; 원천별 "
                                 f"{info['by_src']} — 항등식 행은 종가만 고친다)")
    return out


def read_prices(con: Any, *, cols: "tuple[str, ...] | None" = None,
                table: str = PRICE_TABLE, require_delisted: bool = True,
                start: "str | None" = None, end: "str | None" = None,
                parse_dates: bool = False, session: "str | None" = None, strict: bool = False, unpatched: str = "nan"):
    """백테스트용 가격 패널 — **상장폐지 종목 포함**을 로딩 시점에 검사한다.

    생존편향은 "가장 큰 숨은 인플레이터"인데(GUARDRAILS §3), 검증 스택은 넘겨받은
    트레이드만 믿으므로 유니버스가 이미 생존자만 담고 있으면 **어떤 게이트도 그걸
    못 잡는다**(§4 공백 2). 잡을 수 있는 유일한 지점이 데이터를 읽는 이 자리다.

    Args:
        con: 열린 연결.
        cols: 반환 컬럼(기본 :data:`_PRICE_READ_COLS`).
        table: 가격 테이블. 기본 분할조정 테이블 — 백테스트가 원자료를 읽으면
            액면분할이 수익률로 둔갑한다.
        require_delisted: DB 에 폐지 종목 시세가 있는데 이 조회 결과엔 없으면
            ``AssertionError``. 유니버스를 좁히는 WHERE 를 붙였거나 폐지 백필이
            안 돌았다는 뜻이다. 폐지 시세가 DB 에 아예 없으면(신규 DB 등) 통과시킨다 —
            없는 걸 요구할 수는 없고, 그 경우는 백필 자체가 선행 과제다.
        start, end: ``YYYY-MM-DD``(양 끝 포함) — **날짜로만** 자른다(종목은 자르지 않는다). 주면 폐지 검사도 그 창 기준
            (:func:`_assert_window_has_delisted`). ``trade_value`` 를 요청하면 단위 카나리아(:func:`_check_trade_value_units`,
            백만원이 아니면 :class:`TradeValueUnitError`)가 돈다(값은 바꾸지 않는다).
        parse_dates: ``date`` 열을 datetime64 로(``pd.read_sql(parse_dates=["date"])`` 와 같은 경로).
        session: ``"regular"``(``None`` 기본 = 보정 대상 표면 이것) — 2026-09-14~ 행(KRX 애프터마켓 16:00~20:00 체결이 섞였다)을 정규장
            값으로 보정한다(:func:`_apply_regular_session`: 분봉 > 종가 항등식 > ``unpatched`` 정책; 수는 ``df.attrs["krx_session"]``,
            보정했으면 ``krx_patched``·``krx_src`` 열). 그 날짜가 없는 창은 ``"raw"`` 와 바이트 같다. sqlite 에 분봉 표가 없는데 그 날짜가
            있으면 :class:`SessionUnavailable`. ``"raw"`` — 테이블 값 그대로(사전등록 재현·자체 보정 경로가 명시). 보정 대상이 아닌 표
            (``daily_bars_raw``·``daily_bars_adj_v2``)에 ``"regular"`` 를 명시하면 :class:`ValueError`, 기본값이면 경고 뒤 원값.
        strict: 정규장 모드에서 요청 열을 다 고치지 못한 행이 하나라도 있으면 :class:`SessionUnavailable`.
        unpatched: 정규장 모드에서 고칠 원천이 없는 값 — ``"nan"``(기본, 비운다) · ``"raw"``(원값 유지, ``krx_src`` 로 표시). :data:`UNPATCHED`.

    Returns: pandas DataFrame.
    """
    import pandas as pd

    _check_session(session, unpatched)
    session = _resolve_session(session, table)
    use = tuple(cols or _PRICE_READ_COLS)
    extra_open = _session_needs_open(table, use, session, end)
    sel = ", ".join(use + (("open",) if extra_open else ()))
    ph = _ph(con)
    where, params = [], []
    if start is not None:
        where.append(f"date >= {ph}")
        params.append(start)
    if end is not None:
        where.append(f"date <= {ph}")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    df = pd.read_sql_query(f"SELECT {sel} FROM {table}{clause}", con,  # noqa: S608 — 컬럼·테이블은 모듈 상수, 날짜는 바인딩
                           params=tuple(params) or None,
                           parse_dates=["date"] if parse_dates and "date" in use else None)
    if require_delisted:
        if where and "code" in df.columns:
            _assert_window_has_delisted(con, set(df["code"].astype(str)), table=table,
                                        start=start or "0000-01-01", end=end or "9999-12-31")
        elif not where:
            _assert_universe_has_delisted(con, df, table=table)
    _check_trade_value_units(con, df, table=table, start=start, end=end)
    if session == "regular":
        df = _apply_regular_session(con, df, table=table, strict=strict, unpatched=unpatched)
        if extra_open:
            attrs = dict(df.attrs)
            df = df.drop(columns="open")
            df.attrs.update(attrs)
    return df


#: 거래대금 단위 카나리아 — 표본의 trade_value / (close × volume / 1e6) 중앙값이 이 밖이면 trade_value 가 백만원 단위가 아니다
#: (원 단위면 ~1e6, 억 단위면 ~0.01). 조정 테이블은 분할 전 행의 비율이 분할 배수만큼 크지만 중앙값은 1 근처다.
TV_UNIT_BAND = (0.5, 2.0)
#: 카나리아가 보는 날짜 수 — 창 안에 고르게 + **마지막 날짜들**(최근 적재분만 단위가 바뀐 경우를 잡는다). 날짜마다 중앙값을 따로 잰다.
_TV_SAMPLE_DAYS, _TV_RECENT_DAYS, _TV_MIN_ROWS = 12, 3, 20


class TradeValueUnitError(AssertionError):
    """trade_value 가 백만원 단위가 아니다(수집·적재 단위 결함)."""


def _tv_sample_dates(dates) -> list:
    """정렬된 고유 날짜에서 고르게 :data:`_TV_SAMPLE_DAYS` 개 + 마지막 :data:`_TV_RECENT_DAYS` 개."""
    import numpy as np  # noqa: PLC0415

    u = sorted(set(dates))
    if len(u) <= 5 * _TV_SAMPLE_DAYS:
        return u                                    # 짧은 창(작은 픽스처 포함)은 전부
    pick = {u[i] for i in np.linspace(0, len(u) - 1, min(_TV_SAMPLE_DAYS, len(u))).astype(int)}
    return sorted(pick | set(u[-_TV_RECENT_DAYS:]))


def _check_trade_value_units(con: Any, df, *, table: str, start: "str | None", end: "str | None") -> None:
    """``trade_value`` 를 읽을 때 단위 카나리아. 값이 바뀌지 않는다 — 통과하거나 :class:`TradeValueUnitError`.

    창(``start``~``end``) 안 날짜를 고르게 + 마지막 날짜들을 골라 **날짜마다** 중앙값을 잰다(행이 :data:`_TV_MIN_ROWS` 미만인 날은 건너뛴다).
    프레임에 close·volume 이 있으면 그것으로, 없으면 같은 날짜들의 표본 질의로 잰다. (예전 대체 질의는 ``ORDER`` 없는 ``LIMIT 5000`` 이라
    표본이 창 어디서 오는지 몰랐고, 최근 행만 단위가 바뀐 경우를 놓쳤다 — 2026-10-13 리뷰.)"""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if "trade_value" not in df.columns or table not in _SESSION_TABLES:
        return
    if "date" not in df.columns or not len(df):
        return                                  # 날짜 없는 요청 — 판단할 창이 없다
    uniq = {str(u)[:10]: u for u in df["date"].unique()}   # 고유 날짜만 문자열로(수백만 행 astype(str) 을 피한다)
    days = _tv_sample_dates(uniq)
    if {"close", "volume"} <= set(df.columns):
        sel = df["date"].isin([uniq[d] for d in days]).to_numpy()
        x = df.loc[sel, ["date", "close", "volume", "trade_value"]]
        x = x.assign(day=x.pop("date").map({uniq[d]: d for d in days}))
    else:
        ph = _ph(con)
        x = pd.read_sql_query(f"SELECT date, close, volume, trade_value FROM {table} WHERE trade_value > 0 AND volume > 0 AND close > 0 "  # noqa: S608
                              f"AND date IN ({', '.join([ph] * len(days))})", con, params=tuple(days))
        x["day"] = x.pop("date").astype(str).str[:10]
    c, v, t = (pd.to_numeric(x[k], errors="coerce").abs().to_numpy(float) for k in ("close", "volume", "trade_value"))
    ok = np.isfinite(c) & np.isfinite(v) & np.isfinite(t) & (c > 0) & (v > 0) & (t > 0)
    if ok.sum() < _TV_MIN_ROWS:
        return                                  # 판단할 표본이 없다(빈 창·작은 픽스처)
    r = pd.Series(t[ok] / (c[ok] * v[ok] / 1e6)).groupby(x["day"].to_numpy()[ok])
    med = r.median()[r.size() >= _TV_MIN_ROWS]
    if med.empty:
        med = pd.Series({"(전체)": float(np.median(t[ok] / (c[ok] * v[ok] / 1e6)))})
    lo, hi = TV_UNIT_BAND
    bad = med[(med < lo) | (med > hi)]
    if len(bad):
        raise TradeValueUnitError(
            f"{table}.trade_value 단위 카나리아 실패: 날짜별 trade_value / (close × volume / 1e6) 중앙값이 {lo}~{hi} 밖 — "
            f"{', '.join(f'{d} {m:.4g}' for d, m in bad.items())} (표본 {len(med)}일). trade_value 는 백만원 단위여야 한다"
            f"(ADV 10억 = 1,000) — 적재 단위가 바뀌었는지 확인하라.")


def read_code_bars(con: Any, code: str, *, start: str, end: str,
                   cols: "tuple[str, ...]" = ("date", "volume"), table: str = "daily_bars") -> list[tuple]:
    """한 종목의 ``start``~``end``(포함) 일봉 행 튜플 — 차트·카드용. **유니버스 로더가 아니다**(종목 하나)."""
    ph = _ph(con)
    cur = con.cursor()
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE code={ph} AND date BETWEEN {ph} AND {ph}",  # noqa: S608 — 모듈 상수
                (code, start, end))
    return list(cur.fetchall())


_PRICE_WIDE_COLS = ("open", "high", "low", "close", "volume")


def trading_window(con: Any, *, end: "str | None", rows: int, table: str = PRICE_TABLE) -> "tuple[str, str] | None":
    """``end``(없으면 테이블 최신일) 까지 마지막 ``rows`` 거래일의 ``(시작일, 끝일)``(YYYY-MM-DD). 날짜만 — 유니버스를 정하지 않는다.

    라이브 그날치 패널(:func:`swing_it.live.features.load_inputs` ``last_rows``)이 전 기간을 읽지 않으려고 쓴다(통합 검토 M6).
    """
    ph = _ph(con)
    if end is None:
        cur = con.cursor()
        cur.execute(f"SELECT MAX(date) FROM {table}")  # noqa: S608 — 테이블은 모듈 상수
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        end = str(_as_date(row[0]))
    cur = con.cursor()
    cur.execute(f"SELECT DISTINCT date FROM {table} WHERE date <= {ph} ORDER BY date DESC LIMIT {ph}",  # noqa: S608
                (end, int(rows)))
    days = [_as_date(r[0]) for r in cur.fetchall()]
    if not days:
        return None
    return str(min(days)), str(max(days))


def _assert_window_has_delisted(con: Any, codes: "set[str]", *, table: str, start: str, end: str) -> None:
    """날짜 창 읽기용 폐지 포함 검사 — 그 창에 시세가 있는 폐지 종목(source='naver')이 결과에도 있어야 한다(없으면 요구하지 않는다)."""
    ph = _ph(con)
    cur = con.cursor()
    cur.execute(f"SELECT DISTINCT code FROM {table} WHERE source = {ph} AND date >= {ph} AND date <= {ph}",  # noqa: S608
                ("naver", start, end))
    expected = {str(r[0]) for r in cur.fetchall()}
    if expected and not (expected & set(codes)):
        raise AssertionError(f"{table} {start}~{end} 창에 폐지 종목 시세 {len(expected)}개가 있는데 결과엔 하나도 없다 — "
                             "생존편향. 유니버스를 좁히는 조건을 걷어낼 것")


class PricePanels(dict):
    """:func:`read_prices_wide` 의 정규장 모드 반환 — ``{col: 와이드 패널}`` dict 그대로에 보정 통계 ``krx_session`` 속성."""

    krx_session: dict


def read_prices_wide(con: Any, *, cols: "tuple[str, ...]" = _PRICE_WIDE_COLS, table: str = PRICE_TABLE,
                     require_delisted: bool = True, batch: int = 250_000,
                     start: "str | None" = None, end: "str | None" = None,
                     session: "str | None" = None, strict: bool = False, unpatched: str = "nan"):
    """:func:`read_prices` 의 **메모리 절약판** — 긴 프레임을 만들지 않고 날짜×종목 와이드 패널을 바로 채운다.

    ``session``·``strict``·``unpatched`` 는 :func:`read_prices` 와 같다(같은 :func:`_apply_regular_session` 을 09-14~ 칸에만 건다).
    ``"regular"``(기본)면 2026-09-14~ 칸을 정규장 값으로 제자리 보정하고 :class:`PricePanels`(``.krx_session`` = {rows, patched,
    unpatched, by_src, policy})를 돌려준다. ``"raw"`` 는 예전처럼 dict.
    """
    import pandas as pd  # noqa: PLC0415

    from swing_it.krx_session import AFTERMARKET_START  # noqa: PLC0415

    _check_session(session, unpatched)
    session = _resolve_session(session, table)
    use = tuple(cols)
    extra_open = _session_needs_open(table, use, session, end)
    panels = _read_prices_wide_raw(con, cols=use + (("open",) if extra_open else ()), table=table,
                                   require_delisted=require_delisted, batch=batch, start=start, end=end)
    if session == "raw":
        return panels
    out = PricePanels(panels)
    info = {"session": "regular", "since": AFTERMARKET_START, "rows": 0, "patched": 0, "unpatched": 0}
    days = next(iter(panels.values())).index if panels else pd.DatetimeIndex([])
    late = days >= pd.Timestamp(AFTERMARKET_START)
    cols_ = [c for c in _SESSION_COLS if c in out]
    if table in _SESSION_TABLES and late.any() and cols_:
        # 긴 정문과 **같은 함수**로 고친다(분봉 > 항등식 > 정책) — 09-14~ 칸만 길게 펴서 넘기고 제자리에 꽂는다.
        long = pd.concat({c: out[c].loc[late].stack(future_stack=True) for c in cols_}, axis=1)
        long = long.dropna(how="all").rename_axis(["date", "code"]).reset_index()
        fixed = _apply_regular_session(con, long, table=table, strict=strict, unpatched=unpatched)
        for c in cols_:
            w = fixed.pivot(index="date", columns="code", values=c)
            out[c].loc[w.index, w.columns] = w.to_numpy()
        info.update(fixed.attrs["krx_session"])
    if extra_open:
        del out["open"]
    out.krx_session = info
    return out


def _read_prices_wide_raw(con: Any, *, cols: "tuple[str, ...]", table: str, require_delisted: bool, batch: int,
                          start: "str | None", end: "str | None"):
    """:func:`read_prices_wide` 의 원값 본체(세션 보정 전).

    같은 정문 규칙이다: WHERE 없이 전 종목·전 기간을 읽고 폐지 종목 포함을 검사한다(``require_delisted``). 다른 점은
    모양뿐 — Postgres 서버측 커서로 ``batch`` 행씩 받아 미리 잡아 둔 ``(거래일 × 종목)`` float 배열에 꽂는다. 580만 행
    긴 프레임은 pandas 객체열 때문에 RSS 가 7GB 를 넘지만(2026-10-10 실측), 이 경로는 열 5개 기준 ~0.5GB 다.
    sqlite(로컬 개발)는 :func:`read_prices` 로 읽어 피벗한다(작은 DB 전제).

    ``start``/``end``(YYYY-MM-DD, 포함)를 주면 **날짜로만** 자른다(종목은 자르지 않는다 — 그 창의 전 종목, 폐지 포함). 폐지 검사도
    그 창 기준(:func:`_assert_window_has_delisted`). 라이브 그날치 패널이 전 기간 580만 행을 읽지 않게 하려는 것이다(통합 검토 M6).

    Returns: ``{col: DataFrame(index=날짜 DatetimeIndex, columns=종목 Index)}`` — 없는 (날짜, 종목) 은 NaN.
    """
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    bounded = start is not None or end is not None
    lo, hi = (start or "0000-01-01"), (end or "9999-12-31")
    if not _is_pg(con):
        if bounded:
            ph = _ph(con)
            sel = ", ".join(cols)
            df = pd.read_sql_query(f"SELECT code, date, {sel} FROM {table} WHERE date >= {ph} AND date <= {ph}",  # noqa: S608
                                   con, params=(lo, hi))
            if require_delisted:
                _assert_window_has_delisted(con, set(df["code"].astype(str)), table=table, start=lo, end=hi)
        else:
            df = read_prices(con, cols=("code", "date", *cols), table=table, require_delisted=require_delisted, session="raw")
        df["date"] = pd.to_datetime(df["date"])
        return {c: df.pivot(index="date", columns="code", values=c).astype(float) for c in cols}
    where = " WHERE date >= %s AND date <= %s" if bounded else ""
    params = (lo, hi) if bounded else ()
    with con.cursor() as cur:
        cur.execute(f"SELECT DISTINCT date FROM {table}{where} ORDER BY date", params)  # noqa: S608 — 테이블은 모듈 상수
        days = pd.DatetimeIndex([r[0] for r in cur.fetchall()])
        cur.execute(f"SELECT DISTINCT code FROM {table}{where} ORDER BY code", params)  # noqa: S608
        codes = pd.Index([str(r[0]) for r in cur.fetchall()])
    dmap = {d: i for i, d in enumerate(days.date)}
    cmap = {c: i for i, c in enumerate(codes)}
    arrays = {c: np.full((len(days), len(codes)), np.nan) for c in cols}
    sel = ", ".join(cols)
    with con.cursor(name=f"swing_it_prices_{os.getpid()}") as scur:
        scur.itersize = batch
        scur.execute(f"SELECT code, date, {sel} FROM {table}{where}", params)  # noqa: S608 — 컬럼·테이블은 모듈 상수
        while True:
            rows = scur.fetchmany(batch)
            if not rows:
                break
            cs, ds, *vals = zip(*rows)
            ci = np.fromiter((cmap[str(c)] for c in cs), dtype=np.int64, count=len(cs))
            di = np.fromiter((dmap[d] for d in ds), dtype=np.int64, count=len(ds))
            for c, v in zip(cols, vals):
                arrays[c][di, ci] = np.array(v, dtype=float)
    con.rollback()  # 서버측 커서는 트랜잭션 안에서만 산다 — 읽기 전용이라 되돌려 닫는다
    if require_delisted and bounded:
        _assert_window_has_delisted(con, set(map(str, codes)), table=table, start=lo, end=hi)
    elif require_delisted:
        _assert_universe_has_delisted(con, pd.DataFrame({"code": codes}), table=table)
    return {c: pd.DataFrame(arrays[c], index=days, columns=codes) for c in cols}


# ── 자료 커버리지 ─────────────────────────────────────────────────────────────────────────────────────────────
#: 표(또는 표·종류)마다 자료가 있는 첫날(2026-10-12 DB 실측). 이 전 구간을 요청하는 정문은 :func:`require_coverage` 로 막는다 —
#: 2020~23 설계 창에서 뉴스·공매도·분봉·공식 지수는 **없다**(조용히 빈 값·대용으로 채우면 거짓 통과가 난다; 재구축 대용 지수가
#: −4%/년 어긋나 거짓 통과를 낸 적 있다). ``earnings_disclosures.periodic`` 전에는 실적 known_date 가 avail_date 로 돌아간다.
COVERAGE: dict[str, str] = {
    "news_articles": "2023-03-15",
    "short_selling": "2024-05-09",
    "minute_bars": "2025-08-01",
    "sector_index": "2024-01-26",
    "earnings_disclosures.periodic": "2019-02-28",
}


class CoverageError(ValueError):
    """요청 구간이 표의 자료 시작일보다 앞선다."""


def require_coverage(table: str, start: "str | None") -> None:
    """``start``(None = 처음부터)가 :data:`COVERAGE` 의 시작일보다 앞서면 :class:`CoverageError`."""
    first = COVERAGE[table]
    if start is None or str(start)[:10] < first:
        raise CoverageError(f"{table} 자료는 {first} 부터다 — 요청 시작 {start or '처음'} 은 그 전이다. 그 전 구간은 값이 없다"
                            f"(대용을 쓰려면 명시하라).")


#: 공식 지수 표(``sector_index``)의 가격 열은 **×100 정수**다(001 KOSPI 2,500.12 → 250012).
_INDEX_PRICE_COLS = ("open", "high", "low", "close")


def read_index(con: Any, codes: "str | list[str] | tuple[str, ...] | None" = None, *,
               cols: "tuple[str, ...]" = ("code", "date", "close"), start: "str | None" = None, end: "str | None" = None,
               proxy: "Any | None" = None, with_level: bool = True, order_by_date: bool = False, parse_dates: bool = False):
    """업종·시장 지수 정문(``sector_index``; 001 = KOSPI, 101 = KOSDAQ). 직접 SQL 은 린트 (o) 위반.

    가격 열(open·high·low·close)은 표 그대로 **×100 정수**로 주고, ``with_level`` 이면 ``level = close / 100``(실수)을 덧붙인다.
    비교·수익률을 옛 숫자와 비트 단위로 맞춰야 하면 정수 열을 쓴다(정수/100 은 끝자리가 달라질 수 있다).

    **커버리지:** 공식 지수는 2024-01-26 부터다(:data:`COVERAGE`). ``start`` 가 없거나 그 전이면 ``proxy`` 를 명시해야 한다:
      - ``pd.Series``(날짜 → 대용 수준, 코드 하나일 때): 첫 공식일 앞쪽을 대용으로 잇는다. 대용은 **첫 공식일 수준에 맞춰** 배율을
        곱한다(``is_proxy`` 열). 대용 자체의 품질은 부르는 쪽 책임 — 재구축 대용이 −4%/년 어긋나 거짓 통과를 낸 적 있다.
      - ``str``: 부르는 쪽이 그 전 구간을 스스로 다룬다는 선언과 이유(예: "시총 가중 대용 — leader21.build"). 값은 공식 구간만.
    """
    import pandas as pd  # noqa: PLC0415

    if proxy is None:
        require_coverage("sector_index", start)
    use = tuple(cols)
    if not all(c in ("code", "name", "date", "open", "high", "low", "close", "volume", "trade_value") for c in use):
        raise ValueError(f"sector_index 열이 아니다: {use}")
    ph = _ph(con)
    where, params = [], []
    if isinstance(codes, str):
        where.append(f"code = {ph}")
        params.append(codes)
    elif codes is not None:
        where.append(f"code IN ({', '.join([ph] * len(codes))})")
        params.extend(codes)
    if start is not None:
        where.append(f"date >= {ph}")
        params.append(start)
    if end is not None:
        where.append(f"date <= {ph}")
        params.append(end)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    df = pd.read_sql_query(f"SELECT {', '.join(use)} FROM sector_index{clause}" + (" ORDER BY date" if order_by_date else ""),  # noqa: S608
                           con, params=tuple(params) or None, parse_dates=["date"] if parse_dates and "date" in use else None)
    if with_level and "close" in df.columns:
        df["level"] = df["close"].astype(float) / 100.0
    if isinstance(proxy, pd.Series):
        if isinstance(codes, (list, tuple)) and len(codes) != 1 or codes is None or "date" not in df.columns:
            raise ValueError("대용 Series 는 코드 하나·date 열이 있을 때만 잇는다")
        d = pd.to_datetime(df["date"])
        first = d.min()
        px = proxy.copy()
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()
        base = px.loc[:first].dropna()
        if base.empty or not with_level:
            raise ValueError("대용이 첫 공식일까지 값이 없거나 with_level=False — 수준을 맞출 수 없다")
        scale = float(df.loc[d == first, "level"].iloc[0]) / float(base.iloc[-1])
        pre = px.loc[px.index < first]
        if start is not None:
            pre = pre.loc[pre.index >= pd.Timestamp(start)]
        head = pd.DataFrame({"date": pre.index if parse_dates else [x.date() for x in pre.index], "level": pre.to_numpy(float) * scale})
        if "code" in use:
            head["code"] = codes if isinstance(codes, str) else codes[0]
        df["is_proxy"] = False
        head["is_proxy"] = True
        df = pd.concat([head, df], ignore_index=True)[list(df.columns)]
    return df


SUPPLY_TABLE = "supply_demand"
_SUPPLY_READ_COLS = ("code", "date", "individual", "foreign_", "institution")
#: 기관 세부 8종. 키움 소스에만 있고 네이버 폐지 백필에는 없다.
INSTITUTION_DETAIL_COLS = ("fnnc_invt", "insrnc", "invtrt", "bank",
                           "penfnd_etc", "samo_fund", "natn")


#: 네이버 폐지 백필이 채우는 컬럼. 나머지는 전부 NULL 이다.
NAVER_SUPPLY_COLS = ("code", "date", "close", "flu_rt", "acc_trde_qty",
                     "foreign_", "institution", "source")
#: 키움 소스에만 있는 컬럼 — 이걸 쓰면 유니버스가 현재 상장분으로 좁아진다.
KIWOOM_ONLY_SUPPLY_COLS = ("individual", "etc_corp") + INSTITUTION_DETAIL_COLS


def read_supply_demand(con: Any, *, cols: "tuple[str, ...] | None" = None,
                       start: "str | None" = None, end: "str | None" = None,
                       sources: "tuple[str, ...] | None" = None,
                       require_delisted: bool = True,
                       allow_individual_survivorship: bool = False,
                       session: str = "regular", strict: bool = False, parse_dates: bool = False, unpatched: str = "nan"):
    """수급 패널 정문 — 가격의 :func:`read_prices` 에 대응하는 자리.

    이 테이블에는 오랫동안 정문이 없었다. 그래서 소비자들이 ``supply_demand JOIN
    stocks`` 로 읽었는데 ``stocks`` 는 **현재 상장** 마스터라 WHERE 절이 없어도 폐지
    종목이 구조적으로 빠졌고, 검증 스택은 넘겨받은 표본만 보므로 모든 게이트가 초록인
    채로 성적만 부풀었다(GUARDRAILS §4 공백 2). 그 코드는 2026-08-16 에 삭제됐고,
    이 함수가 그 자리를 대신한다.

    **소스별로 컬럼이 반쪽이다.** 키움(현재 상장)은 11개 분류를 다 주지만, 네이버
    폐지 백필은 **외국인·기관 순매매량만** 준다 — ``individual`` 이 NULL 이다. 따라서
    ``individual`` 을 쓰면서 폐지 종목을 함께 담는 것은 **현재 데이터로 불가능**하고,
    ``individual IS NOT NULL`` 로 거르는 순간 조용히 생존편향이 들어온다. 그 선택을
    호출자가 **명시**하게 만드는 것이 이 함수의 존재 이유다.

    Args:
        con: 열린 연결.
        cols: 반환 컬럼(기본 :data:`_SUPPLY_READ_COLS`). 기관 세부는
            :data:`INSTITUTION_DETAIL_COLS` 를 붙여 요청한다.
        start: 조회 시작일(``YYYY-MM-DD``, 포함). ``None`` 이면 처음부터.
        end: 조회 종료일(포함). ``None`` 이면 끝까지.
        require_delisted: 조회 **구간 안에** 수급이 존재하는 폐지 종목이 결과에도
            담겼는지 검사한다. 구간을 좁히면 기대치도 그 구간으로 좁혀지므로,
            "오늘 하루" 같은 시점 조회에서 헛되이 터지지 않는다.
        allow_individual_survivorship: ``individual`` 을 요청하면서 폐지 종목 커버리지를
            포기한다고 **명시**할 때 ``True``. 명시 없이 둘을 함께 요구하면
            ``AssertionError`` 로 막는다 — 조용한 생존편향보다 시끄러운 실패가 낫다.
        session: ``"regular"``(기본) — ``close``·``flu_rt``·``acc_trde_qty`` 를 요청했고 2026-09-14~ 행이 있으면 그 행을 KRX 정규장 값으로
            바꾼다(:func:`_apply_regular_supply`): 종가 = 분봉 정규장 > 항등식(:func:`swing_it.krx_session.regular_close_from_supply_demand`),
            등락률 = 그 종가 / 그날 기준가, 거래량 = 분봉 정규장 합. 못 고친 값은 ``unpatched`` 정책(기본 **NaN** — 가격 정문과 같은
            :data:`UNPATCHED`), 수는 ``df.attrs["krx_session"]``, 행별 원천은 ``krx_src``(:data:`KRX_SRC`). 정규장 모드의 ``close`` 는
            **부호 없는 가격**이다(전 행 절댓값 — 키움 원천의 등락 부호는 ``session="raw"`` 에만 남는다; 2026-10-13 DB 에는 음수 0 건).
            ``"raw"`` — 테이블 값 그대로(사전등록 재현·자체 보정 경로가 명시). 순매매 수량 열은 손대지 않는다.
        strict: 정규장 모드에서 못 고친 행이 있으면 :class:`SessionUnavailable`.
        unpatched: :func:`read_prices` 와 같다(``"nan"`` 기본 · ``"raw"``).
        parse_dates: ``date`` 를 datetime64 로(``pd.read_sql(parse_dates=["date"])`` 와 같은 경로).

    Returns: pandas DataFrame.
    """
    import pandas as pd

    _check_session(session, unpatched)
    use = tuple(cols or _SUPPLY_READ_COLS)
    kiwoom_only = tuple(c for c in use if c in KIWOOM_ONLY_SUPPLY_COLS)
    if kiwoom_only and require_delisted and not allow_individual_survivorship:
        raise AssertionError(
            f"{', '.join(kiwoom_only)} 는 키움 소스에만 있고 네이버 폐지 백필에는 "
            "NULL 이라, 폐지 종목을 함께 담을 수 없다. 더 고약한 건 **지표마다 "
            "유니버스가 달라진다**는 점이다 — institution 으로 집계하면 폐지분이 들어오고 "
            "individual 로 집계하면 같은 종목이 NULL 로 빠져서, 두 막대를 나란히 그리면 "
            "분모가 다른 걸 비교하게 된다. 개인·기관세부가 꼭 필요하면 "
            "sources=('kiwoom',) 로 유니버스를 명시적으로 좁히고 "
            "allow_individual_survivorship=True 로 생존편향을 감수한다고 적어라."
        )

    ph = "%s" if _is_pg(con) else "?"
    where, params = [], []
    if start is not None:
        where.append(f"date >= {ph}")
        params.append(start)
    if end is not None:
        where.append(f"date <= {ph}")
        params.append(end)
    if sources is not None:
        where.append(f"source IN ({', '.join([ph] * len(sources))})")
        params.extend(sources)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    affected = session == "regular" and any(c in use for c in _SUPPLY_SESSION_COLS) and (end is None or end >= _aftermarket_start())
    extra = tuple(c for c in ("code", "date", "close", "flu_rt") if affected and c not in use)
    sel = ", ".join(use + extra)
    df = pd.read_sql_query(  # noqa: S608 — 컬럼은 모듈 상수, 날짜는 파라미터 바인딩
        f"SELECT {sel} FROM {SUPPLY_TABLE}{clause}", con, params=tuple(params) or None,
        parse_dates=["date"] if parse_dates and "date" in use else None)
    if require_delisted:
        _assert_supply_has_delisted(con, df, start=start, end=end)
    if affected:
        df = _apply_regular_supply(con, df, use=use, strict=strict, unpatched=unpatched)
        if extra:
            attrs = dict(df.attrs)
            df = df.drop(columns=list(extra))
            df.attrs.update(attrs)
    if session == "regular" and "close" in use and len(df) and (df["close"].dropna() < 0).any():
        attrs = dict(df.attrs)
        df["close"] = df["close"].abs()                     # 한 규약: 정규장 모드 종가 = 부호 없는 가격(보정 행은 원래 부호가 없다)
        df.attrs.update(attrs)
    return df


#: 2026-09-14~ KRX 애프터마켓이 섞인 수급 열(순매매 수량 열은 정규장 밖 체결을 따로 싣지 않아 대상이 아니다 — 확인 필요 시 별도 조사).
_SUPPLY_SESSION_COLS = ("close", "flu_rt", "acc_trde_qty")


def _aftermarket_start() -> str:
    from swing_it.krx_session import AFTERMARKET_START  # noqa: PLC0415

    return AFTERMARKET_START


def _apply_regular_supply(con: Any, df, *, use: "tuple[str, ...]", strict: bool, unpatched: str = "nan"):
    """수급표의 2026-09-14~ 행 close·flu_rt·acc_trde_qty 를 정규장 값으로(못 고치면 ``unpatched`` 정책). 그 날짜가 없으면 입력 그대로.
    가격 정문(:func:`_apply_regular_session`)과 같은 원천 순서·정책·표시 열(``krx_patched``·``krx_src``)이다."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from swing_it.krx_session import AFTERMARKET_START, read_regular_close, read_session_bars_any  # noqa: PLC0415

    info = {"session": "regular", "since": AFTERMARKET_START, "rows": 0, "patched": 0, "unpatched": 0}
    d = pd.to_datetime(df["date"])
    late = (d >= pd.Timestamp(AFTERMARKET_START)).to_numpy()
    if not late.any():
        df.attrs["krx_session"] = info
        return df
    lo, hi = d[late].min().strftime("%Y-%m-%d"), d[late].max().strftime("%Y-%m-%d")
    sess = read_session_bars_any(con, start=lo, end=hi)
    if sess is None:
        raise SessionUnavailable(
            f"{SUPPLY_TABLE} 에 {AFTERMARKET_START} 이후 행 {int(late.sum())}개가 있는데 이 DB 에 minute_bars 가 없어 close·flu_rt·"
            "acc_trde_qty 를 정규장으로 보정할 수 없다(KRX 애프터마켓 16:00~20:00 체결이 섞였다). 원값이 필요하면 session='raw'.")
    ident = read_regular_close(con, start=lo, end=hi)
    out = df.copy()
    key = pd.MultiIndex.from_arrays([out.loc[late, "code"].astype(str).to_numpy(), d[late].to_numpy()])
    s = sess.assign(code=sess["code"].astype(str), date=pd.to_datetime(sess["date"])).set_index(["code", "date"])
    i = ident.assign(code=ident["code"].astype(str)).set_index(["code", "date"])
    s_close = s["close"].reindex(key).to_numpy(float) if len(s) else np.full(len(key), np.nan)
    i_close = i["close"].reindex(key).to_numpy(float) if len(i) else np.full(len(key), np.nan)
    rc = np.where(np.isfinite(s_close), s_close, i_close)
    ok = np.ones(len(key), bool)
    keep = unpatched == "raw"                              # 못 고친 값을 원값으로 둘지(정책)
    for c in ("close", "flu_rt", "acc_trde_qty"):
        if c in out.columns:
            out[c] = out[c].astype(float)
    if "close" in use:
        raw_close = np.abs(df.loc[late, "close"].to_numpy(float))
        out.loc[late, "close"] = np.where(np.isfinite(rc) | (not keep), rc, raw_close)
        ok &= np.isfinite(rc)
    if "flu_rt" in use:
        raw_c = np.abs(df.loc[late, "close"].to_numpy(float))
        raw_f = df.loc[late, "flu_rt"].to_numpy(float)
        base = raw_c / (1.0 + raw_f / 1e4)                                                # 그날 기준가(= 전 거래일 정규장 종가)
        with np.errstate(divide="ignore", invalid="ignore"):
            fr = (rc / base - 1.0) * 1e4
        out.loc[late, "flu_rt"] = np.where(np.isfinite(fr) | (not keep), fr, raw_f)
        ok &= np.isfinite(fr)
    if "acc_trde_qty" in use:
        q = s["volume"].reindex(key).to_numpy(float) if len(s) else np.full(len(key), np.nan)
        out.loc[late, "acc_trde_qty"] = np.where(np.isfinite(q) | (not keep), q, df.loc[late, "acc_trde_qty"].to_numpy(float))
        ok &= np.isfinite(q)
    out["krx_patched"] = False
    out.loc[late, "krx_patched"] = ok
    src = np.where(late, "none", "pre").astype(object)
    src[np.flatnonzero(late)[np.isfinite(i_close)]] = "identity"
    src[np.flatnonzero(late)[np.isfinite(s_close)]] = "session"
    out["krx_src"] = src
    n, k = int(late.sum()), int(ok.sum())
    info.update(rows=n, patched=k, unpatched=n - k, policy=unpatched,
                by_src={x: int((src == x).sum()) for x in ("session", "identity", "none")})
    out.attrs["krx_session"] = info
    if strict and n > k:
        raise SessionUnavailable(f"{SUPPLY_TABLE} {AFTERMARKET_START} 이후 {n}행 중 {n - k}행을 정규장으로 고치지 못했다(strict)")
    return out


def _assert_supply_has_delisted(con: Any, df, *, start, end) -> None:
    """조회 구간에 수급이 있는 폐지 종목이 결과에도 담겼는지 확인."""
    ph = "%s" if _is_pg(con) else "?"
    where, params = ["source = " + ph], ["naver"]
    if start is not None:
        where.append(f"date >= {ph}")
        params.append(start)
    if end is not None:
        where.append(f"date <= {ph}")
        params.append(end)
    sql = (f"SELECT DISTINCT code FROM {SUPPLY_TABLE} "  # noqa: S608 — 모듈 상수
           f"WHERE {' AND '.join(where)}")
    if _is_pg(con):
        with con.cursor() as cur:
            cur.execute(sql, tuple(params))
            expected = {r[0] for r in cur.fetchall()}
    else:
        expected = {r[0] for r in con.execute(sql, tuple(params)).fetchall()}
    if not expected:
        return  # 그 구간에 폐지 종목 수급이 아예 없다 — 요구할 수 없다.
    if "code" not in getattr(df, "columns", []):
        return  # code 를 안 뽑았으면 검사할 수단이 없다.
    if not (expected & set(df["code"].astype(str))):
        raise AssertionError(
            f"{SUPPLY_TABLE} 조회 구간에 폐지 종목 수급 {len(expected)}개가 있는데 "
            f"결과엔 하나도 없다 — stocks(현재 상장 마스터) 조인이나 "
            f"individual IS NOT NULL 필터가 걸렸을 수 있다. 의도한 것이면 "
            f"require_delisted=False 로 명시하라."
        )


def _assert_universe_has_delisted(con: Any, df, *, table: str) -> None:
    """조회 결과가 폐지 종목을 담고 있는지 확인(DB 에 있는 경우에 한해)."""
    ph = "%s" if _is_pg(con) else "?"
    sql = f"SELECT count(DISTINCT code) FROM {table} WHERE source = {ph}"  # noqa: S608 — 모듈 상수
    if _is_pg(con):
        with con.cursor() as cur:
            cur.execute(sql, ("naver",))
            in_db = cur.fetchone()[0]
    else:
        in_db = con.execute(sql, ("naver",)).fetchone()[0]
    if not in_db:
        return                      # DB 에 폐지 시세 자체가 없음 — 백필이 선행 과제
    if "code" not in df.columns:
        return                      # code 를 안 뽑은 조회는 유니버스 판정 대상이 아님
    loaded = set(df["code"].astype(str).unique())
    ph2 = "%s" if _is_pg(con) else "?"
    sql2 = f"SELECT DISTINCT code FROM {table} WHERE source = {ph2}"  # noqa: S608 — 모듈 상수
    if _is_pg(con):
        with con.cursor() as cur:
            cur.execute(sql2, ("naver",))
            delisted = {r[0] for r in cur.fetchall()}
    else:
        delisted = {r[0] for r in con.execute(sql2, ("naver",)).fetchall()}
    if not (delisted & loaded):
        raise AssertionError(
            f"{table} 에 상장폐지 종목 {in_db}개가 있는데 로딩된 유니버스에 하나도 없다 — "
            f"생존편향이 들어간다. 유니버스를 좁히는 WHERE 를 걷어내거나, "
            f"의도한 것이면 read_prices(require_delisted=False) 로 명시하라."
        )


#: 백필 완료 후 허용 잔여. 상류가 영구 결측 종목에 ``backfill_markers`` 를 찍어주므로
#: 완료 상태의 기대값은 0 이다. 마커가 없는 구버전 DB 를 위해 여유만 둔다.
SHARES_RESIDUAL_OK = 10


def _ph(con: Any) -> str:
    return "%s" if _is_pg(con) else "?"


def _as_date(v: Any):
    import datetime as _dt  # noqa: PLC0415

    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    return _dt.date.fromisoformat(str(v)[:10])


def last_trading_date_before(con: Any, before: str) -> Any:
    """``before``(YYYY-MM-DD) **직전** DB 거래일(``daily_bars`` 날짜 집합 기준). 없으면 ``None``.

    유니버스 로더가 아니다 — 날짜 하나만 돌려준다(라이브 ``signal`` 의 ``--asof`` 재생·직전 거래일 판정). 가격·종목은
    :func:`read_prices` 로만 읽는다(규칙 (g)); 이 정문 안에 두는 이유도 그것이다.
    """
    cur = con.cursor()
    cur.execute(f"SELECT MAX(date) FROM daily_bars WHERE date < {_ph(con)}", (before,))
    row = cur.fetchone()
    return _as_date(row[0]) if row else None


def reference_daily_bars(con: Any, code: str, *, end: str, limit: int = 60) -> list[tuple]:
    """한 종목의 일봉 ``(date, open, high, low, close)`` 최신순 ``limit`` 개(``date ≤ end``) — 기준 종목·재생 시세용.

    유니버스 로더가 아니다(종목 하나, 날짜·보조 가격). 2026-09-14 이후 ``daily_bars`` 종가는 KRX 애프터마켓(16:00~20:00) 체결이 섞였으므로 호출부는
    **날짜**에만 기대고 가격은 보조로만 쓴다(:class:`swing_it.live.brokers.DBQuoteProvider`).
    """
    cur = con.cursor()
    ph = _ph(con)
    cur.execute(f"SELECT date, open, high, low, close FROM daily_bars WHERE code = {ph} AND date <= {ph} "
                f"ORDER BY date DESC LIMIT {ph}", (code, end, int(limit)))
    return [(_as_date(r[0]), r[1], r[2], r[3], r[4]) for r in cur.fetchall()]


def last_close_before(con: Any, code: str, before: str) -> "float | None":
    """한 종목의 ``before`` 직전 거래일 종가(재생 시세의 장전 현재가·전일종가 대용). 유니버스 로더가 아니다."""
    cur = con.cursor()
    ph = _ph(con)
    cur.execute(f"SELECT close FROM daily_bars WHERE code = {ph} AND date < {ph} ORDER BY date DESC LIMIT 1",
                (code, before))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def shares_backfill_pending(con: Any) -> int:
    """과거 상장주식수가 아직 없는 **상장** 종목 수. 완료되면 60 근처로 수렴한다.

    시총을 분모로 쓰는 연구는 이 값을 먼저 봐야 한다. 백필이 도는 중이면 유니버스가
    **시대별로 다르게** 좁혀지고, 그 상태의 판정은 신호가 아니라 처리 진행률을 잰다.
    실제로 그렇게 한 번 무효가 났다 — 분모가 5종목만 풀리던 구간에서 배터리가
    +1.057R·승률 73% 를 냈고, 유니버스를 고른 건 신호가 아니라 데이터 존재 여부였다.

    커버리지 숫자가 아니라 **그 데이터로 값이 나오는 종목**을 센다(§4-2 의 교훈 —
    커버리지는 초록인데 한 행도 안 쓰였던 사고).

    Returns: 미처리 종목 수. :data:`SHARES_RESIDUAL_OK` 이하면 정상 종료로 본다.
    """
    cur = con.cursor()
    # 상류(quant-airflow)가 "재조회해도 안 나오는" 종목에 backfill_markers 를 찍는다.
    # 그걸 빼야 **"아직 안 한 일" 과 "영구히 못 할 일" 이 구분된다** — 안 빼면 완료
    # 상태에서도 잔여가 남아 게이트가 영원히 중단된다(실측 106건).
    has_marker = False
    try:
        cur.execute("SELECT to_regclass('backfill_markers')")
        has_marker = cur.fetchone()[0] is not None
    except Exception:  # noqa: BLE001 — 구버전 DB·sqlite 는 마커 테이블이 없다
        con.rollback() if hasattr(con, "rollback") else None
    marker = ("AND NOT EXISTS (SELECT 1 FROM backfill_markers k WHERE k.code = b.code)"
              if has_marker else "")
    cur.execute(f"""
        SELECT count(*) FROM (
          SELECT b.code FROM daily_bars b
          WHERE b.source <> 'naver'
            AND b.date BETWEEN '2016-01-01' AND '2025-12-31'
          GROUP BY b.code
          HAVING NOT EXISTS (
            SELECT 1 FROM shares_outstanding_history s
            WHERE s.code = b.code AND s.date < '2026-01-01')
          {marker}
        ) t""")  # noqa: S608 — 리터럴 상수만, 사용자 입력 없음
    return int(cur.fetchone()[0])


# ── 상장주식수 정문 ───────────────────────────────────────────────────────────────────────────────────────────
# 표 ``shares_outstanding_history`` 의 함정(2026-10-12 측정): dart 행은 knowledge_date(알게 된 날)가 date(기준일)보다 중앙값 82일 늦다 —
# date 로 as-of 하면 석 달 미래 정보다. kiwoom 행은 knowledge_date 가 NULL(수집일 = 기준일). 그래서 as-of 키는
# ``coalesce(knowledge_date, date)`` 다. 0 이하 주식수는 결측이다(상장 종목이 0주일 수 없다 — market_cap_asof 주석 참조).
_SHARES_COLS = ("code", "date", "shares_outstanding", "knowledge_date")
_SHARES_DATE_COLS = ("date", "knowledge_date", "orig_rcept_dt", "asof")


def read_shares(con: Any, *, cols: "tuple[str, ...]" = _SHARES_COLS, codes: "list[str] | None" = None,
                with_asof: bool = False, parse_dates: bool = False, include_nonpositive: bool = False):
    """상장주식수 정문 — 긴 표. 직접 SELECT 는 린트 (k) 위반.

    Args:
        cols: 반환 열(표의 열 이름). 기본 code, date, shares_outstanding, knowledge_date.
        codes: 주면 그 종목만(SQL 에서 자른다).
        with_asof: ``asof = coalesce(knowledge_date, date)`` 열을 덧붙인다 — **as-of 조인은 이 열로** 한다(date 는 기준일이라
            dart 행은 석 달 미래 정보다).
        parse_dates: 날짜 열(date·knowledge_date·asof 등)을 datetime64 로(``pd.read_sql(parse_dates=...)`` 와 같은 경로).
        include_nonpositive: 0 이하 주식수 행도 준다(기본은 뺀다 — 결측이다).
    """
    import re as _re  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    use = tuple(cols)
    if not all(_re.fullmatch(r"[a-z_][a-z0-9_]*", c) for c in use):
        raise ValueError(f"열 이름이 아니다: {use}")
    sel = ", ".join(use) + (", coalesce(knowledge_date, date) AS asof" if with_asof else "")
    ph = _ph(con)
    where, params = [], []
    if not include_nonpositive:
        where.append("shares_outstanding > 0")
    if codes is not None:
        codes = [str(c) for c in codes]
        if _is_pg(con):
            where.append(f"code = ANY({ph})")
            params.append(codes)
        else:
            where.append(f"code IN ({', '.join([ph] * len(codes)) or 'NULL'})")
            params.extend(codes)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    out_cols = use + (("asof",) if with_asof else ())
    pdates = [c for c in out_cols if c in _SHARES_DATE_COLS] if parse_dates else None
    return pd.read_sql_query(f"SELECT {sel} FROM shares_outstanding_history{clause}",  # noqa: S608 — 열 이름은 위에서 검사
                             con, params=tuple(params) or None, parse_dates=pdates or None)


def market_cap_asof(con: Any, code: str, date: str, *, legacy: bool = False) -> int | float | None:
    """``code`` 의 ``date`` 시총 = **원 종가** × 그날까지 **알려진** 주식수(``coalesce(knowledge_date, date) <= date``).

    :func:`market_cap_asof_bulk` 한 줄판. ``legacy=True`` 는 옛 정의(``daily_bars.close`` — 이후 분할·병합이 소급 반영된 가격 —
    × ``date`` 기준 주식수)로, 그 숫자를 재현해야 하는 호출부만 명시한다. 값이 없으면 ``None``.
    """
    if legacy:
        return _market_cap_asof_legacy(con, code, date)
    import pandas as pd  # noqa: PLC0415

    v = market_cap_asof_bulk(con, pd.DataFrame({"code": [code], "date": [date]})).iloc[0]
    return None if pd.isna(v) else float(v)


def market_cap_asof_bulk(con: Any, df: Any, *, legacy: bool = False) -> Any:
    """여러 ``(code, date)`` 의 시총 = 원 종가 × 그날까지 알려진 주식수. ``legacy=True`` 는 옛 정의(아래 :func:`_market_cap_asof_bulk_legacy`).

    - 원 종가: 그 날짜의 ``supply_demand.close``(절댓값; 키움 피드의 원 가격) — 없으면 ``daily_bars.close``. 2026-09-14~ 은 분봉 정규장
      종가 > 정규장 항등식(:mod:`swing_it.krx_session`) > 일봉(애프터마켓 오염 — 마지막 거래일·분봉 없는 종목만).
      ``daily_bars.close`` 는 이후 분할·병합이 소급 반영된 가격이라(005930 2017 41,200 vs 실제 206만) 옛 정의는 분할 전 대형주를
      수십 분의 1 로 만들었다.
    - 주식수: ``asof = coalesce(knowledge_date, date) <= date`` 인 가장 최근 행(dart 행은 기준일보다 석 달쯤 늦게 알려진다), 0 이하 제외.

    **남은 결함(거울상):** 주식수 행이 주로 연말 DART 라 분할·병합 뒤 수개월 늦게 바뀐다 — 원 종가 × 그 주식수는 그 사이 분할
    배수만큼 틀린다. 패널 전체가 있으면 :func:`swing_it.krx_session.split_consistent_cap` 이 정답이다(정정 G28-4).

    Returns: ``df.index`` 에 맞춘 ``pandas.Series``(없으면 NaN).
    """
    if legacy:
        return _market_cap_asof_bulk_legacy(con, df)
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from swing_it.krx_session import AFTERMARKET_START, combine_raw_close, read_regular_close, read_session_bars_any  # noqa: PLC0415

    if df.empty:
        return pd.Series([], dtype=float, index=df.index)
    codes = sorted({str(c) for c in df["code"]})
    dates = pd.to_datetime(df["date"].astype(str))
    lo, hi = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
    ph = _ph(con)
    if _is_pg(con):
        cw, cp = f"code = ANY({ph})", [codes]
    else:
        cw, cp = f"code IN ({', '.join([ph] * len(codes))})", list(codes)
    sd = pd.read_sql_query(f"SELECT code, date, close FROM supply_demand WHERE {cw} AND date >= {ph} AND date <= {ph} "  # noqa: S608
                           "AND close IS NOT NULL", con, params=(*cp, lo, hi))
    bars = pd.read_sql_query(f"SELECT code, date, close FROM daily_bars WHERE {cw} AND date >= {ph} AND date <= {ph}",  # noqa: S608
                             con, params=(*cp, lo, hi))
    sess = ident = None
    if hi >= AFTERMARKET_START:
        a = max(lo, AFTERMARKET_START)
        sess = read_session_bars_any(con, start=a, end=hi)          # sqlite 에 분봉이 없으면 None — 항등식으로 간다
        ident = read_regular_close(con, start=a, end=hi)
        if sess is not None:
            sess = sess[sess["code"].astype(str).isin(codes)]
        ident = ident[ident["code"].astype(str).isin(codes)]
    for x in (sd, bars):
        x["code"] = x["code"].astype(str)
    rc = combine_raw_close(sd, bars, sess, identity=ident, sd_until=AFTERMARKET_START)
    sh = read_shares(con, cols=("code", "shares_outstanding"), codes=codes, with_asof=True)
    if rc.empty or sh.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)
    sh["code"] = sh["code"].astype(str)
    sh["asof"] = pd.to_datetime(sh["asof"].astype(str))
    sh = sh.sort_values(["asof", "code"], kind="mergesort")
    q = pd.DataFrame({"code": df["code"].astype(str).to_numpy(), "date": dates.to_numpy(), "_row": np.arange(len(df))})
    q = q.merge(rc[["code", "date", "close"]], on=["code", "date"], how="left").sort_values(["date", "code"], kind="mergesort")
    m = pd.merge_asof(q, sh[["code", "asof", "shares_outstanding"]], left_on="date", right_on="asof", by="code",
                      direction="backward").sort_values("_row")
    return pd.Series((m["close"].astype(float) * m["shares_outstanding"].astype(float)).to_numpy(), index=df.index)


def _market_cap_asof_legacy(con: Any, code: str, date: str) -> int | float | None:
    """**옛 정의**(``market_cap_asof(..., legacy=True)``) — ``daily_bars.close``(소급 수정 가격) × ``date`` 기준 주식수. 재현용.

Market cap for ``code`` on exactly ``date``: close * shares outstanding.

    ``close`` must exist for that exact date (no guessing). Shares outstanding
    is looked up as the most recent row with ``date <= date`` — never a row
    dated after the given date — to avoid lookahead bias (e.g. a stock split
    recorded later must not be applied retroactively to an earlier market cap).
    Returns ``None`` if either lookup is empty, or if shares outstanding is
    non-positive (treated as missing — a listed name cannot have 0 shares).
    """
    ph = "%s" if _is_pg(con) else "?"
    cur = con.cursor()
    cur.execute(
        f"SELECT close FROM daily_bars WHERE code={ph} AND date={ph}",
        (code, date),
    )
    bar_row = cur.fetchone()
    if bar_row is None:
        return None
    close = bar_row[0]

    cur.execute(
        f"SELECT shares_outstanding FROM shares_outstanding_history "
        f"WHERE code={ph} AND date<={ph} ORDER BY date DESC LIMIT 1",
        (code, date),
    )
    shares_row = cur.fetchone()
    if shares_row is None:
        return None
    shares = shares_row[0]

    # 0 주식수는 "진짜 0"이 아니라 결측이다 — 상장돼 있으면 0 일 수 없다. 가드가 None 만
    # 보면 close*0 = 0 이 정상 시총으로 나가고, 이걸 분모로 쓰는 소비자에서 inf 가 난다.
    # 게다가 조회가 `date <= 조회일 ORDER BY date DESC LIMIT 1` 이라 0 행이 한 번 들어오면
    # 그 종목의 최신 점이 되어 이후 모든 조회를 이긴다. 상류(quant-airflow collectors)가
    # 2026-08-27 에 막았지만 이 테이블에는 kiwoom·dart·krx + 수동 백필이 함께 쓴다.
    if close is None or shares is None or shares <= 0:
        return None
    return close * shares


def _market_cap_asof_bulk_legacy(con: Any, df: Any) -> Any:
    """**옛 정의**(``market_cap_asof_bulk(..., legacy=True)``) — 재현용.

Vectorized :func:`market_cap_asof` for many ``(code, date)`` rows at once.

    Byte-identical semantics to the per-row function — close comes from
    ``daily_bars`` on the exact date (NOT any close column the caller's
    frame may already carry, e.g. ``supply_demand.close``: that is a
    different, not-necessarily-equal price series in this DB — verified by
    spot-checking real rows, e.g. code 118000 on 2026-03-25 has
    ``daily_bars.close=2410`` vs ``supply_demand.close=241``). Shares
    outstanding is the most recent row with ``date <= date`` (lookahead-safe,
    same rule as the per-row SQL ``date<=? ORDER BY date DESC LIMIT 1``).

    This issues 2 bulk queries total instead of 2 queries per row —
    ``market_cap_asof`` called in a per-row Python loop over a multi-year
    dataset means millions of round trips to Postgres.

    Args:
        df: Must have ``code`` and ``date`` columns.

    Returns:
        A ``pandas.Series`` aligned with ``df.index``: ``market_cap`` per
        row, ``NaN`` where close or shares-outstanding is unavailable
        (mirrors ``market_cap_asof`` returning ``None``).
    """
    import pandas as pd  # noqa: PLC0415 — pandas is a strategy-layer dep, not storage's

    if df.empty:
        return pd.Series([], dtype=float, index=df.index)

    codes = df["code"].unique().tolist()
    ph = "%s" if _is_pg(con) else "?"
    placeholders = ",".join([ph] * len(codes))
    bars = pd.read_sql_query(
        f"SELECT code, date, close FROM daily_bars WHERE code IN ({placeholders})",
        con,
        params=codes,
    )
    bars["date"] = bars["date"].astype(str)

    shares = pd.read_sql_query(
        f"SELECT code, date, shares_outstanding FROM shares_outstanding_history "
        f"WHERE code IN ({placeholders})",
        con,
        params=codes,
    )
    if bars.empty or shares.empty:
        return pd.Series([float("nan")] * len(df), index=df.index)

    shares["date"] = pd.to_datetime(shares["date"].astype(str))
    # merge_asof with `by=` still requires the `on` column sorted globally
    # (not just within each `by` group) — sort by date first, code second.
    shares = shares.sort_values(["date", "code"])

    d = df[["code"]].reset_index(drop=True).copy()
    d["date_str"] = df["date"].astype(str).to_numpy()
    d["_row"] = d.index
    d = d.merge(bars, left_on=["code", "date_str"], right_on=["code", "date"], how="left")
    d["date"] = pd.to_datetime(d["date_str"])
    d_sorted = d.sort_values(["date", "code"])

    merged = pd.merge_asof(
        d_sorted,
        shares[["code", "date", "shares_outstanding"]],
        on="date",
        by="code",
        direction="backward",
    )
    merged = merged.sort_values("_row")
    # per-row 와 동일 semantics: 주식수가 0 이하면 결측으로 떨어뜨린다(더 오래된 유효
    # 행으로 대체하지 않는다 — 그러면 두 함수의 결과가 갈린다).
    merged.loc[merged["shares_outstanding"] <= 0, "shares_outstanding"] = float("nan")
    market_cap = (merged["close"] * merged["shares_outstanding"]).to_numpy()
    return pd.Series(market_cap, index=df.index)
