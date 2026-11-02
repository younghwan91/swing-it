"""ML 수익률 예측 — 입력 패널(DB 읽기 전용). PREREG §2·§3·§4, DESIGN §2.

:func:`build_panels` 가 END(인자, ≤ 2023-12-28)까지의 원천만 읽어 code × date 패널을 만든다. 절단 동일성 관문(0단계 a)은
END 를 결정일로 바꿔 이 함수를 다시 부른다 — 그래서 모든 읽기가 END 인자로 잘린다.

- 가격·시총·ADV·업종·스타일: core ``load_risk_inputs`` → PO-3.3 이음매 가드(``pead_optimizer_po4.guarded``, B 와 같다) → c0 ``exposures``.
- 거래량: core ``load_daily_bars``(로더와 같은 표·같은 캐시) — 정지일 라벨(§4)용.
- 상장주식수: 로더와 같은 식(core ``_asof_shares``·``_roll_events``) — F5·F8 분모.
- 수급·보유율: ``storage.read_investor_trend`` 하나(§3 MINOR a). 리포트: ``storage.read_research_reports``.
- 실적: c0 ``yoy_signal(end, fallback=False)``(F1). 공시: ``earnings_disclosures`` 분기별 첫 접수일(F2).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ml_forecast_common import LOAD_START, assert_window

V4_SHRINK, V4_HALF_LIFE, V4_NW, V4_GROUPS = 0.1, 90, 2, 10  # build_risk_model 기본 + v4 수축
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
STMT_TIMEOUT = "600s"


def _db():
    """읽기 전용 + statement_timeout 연결(psycopg2). core 로더에도 이 연결을 넘긴다."""
    from swing_it.storage import connect, db_default

    con = connect(db_default())
    con.set_session(readonly=True, autocommit=True)
    with con.cursor() as cur:
        cur.execute(f"SET statement_timeout='{STMT_TIMEOUT}'")
    return con


def _pivot(df: pd.DataFrame, col: str, codes, dates) -> np.ndarray:
    out = np.full((len(codes), len(dates)), np.nan)
    if df.empty:
        return out
    i = pd.Index([str(c) for c in codes]).get_indexer(df["code"].astype(str))
    t = pd.DatetimeIndex(dates).get_indexer(pd.to_datetime(df["date"]))
    ok = (i >= 0) & (t >= 0)
    out[i[ok], t[ok]] = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)[ok]
    return out


def fingerprint_range(lo, hi) -> list:
    """core ``_fingerprint``(daily_bars_adjusted, [lo, hi])."""
    from krx_quant_core.data.daily import _fingerprint

    con = _db()
    try:
        return _fingerprint(con, "daily_bars_adjusted", pd.Timestamp(lo).date(), pd.Timestamp(hi).date())
    finally:
        con.close()


def _shares(con, codes, dates, end: str) -> np.ndarray:
    """로더(core ``_load``)와 같은 상장주식수 패널 — knowledge_date 다음 거래일부터, 분할·병합 굴림."""
    from krx_quant_core.data.db import fetch_frame, sql_literal
    from krx_quant_core.data.risk_inputs import SHARE_EVENT_KINDS, _asof_shares, _roll_events

    b = pd.Timestamp(end).date()
    sh = fetch_frame(
        con,
        f"SELECT code, date, shares_outstanding, knowledge_date FROM shares_outstanding_history "
        f"WHERE knowledge_date <= {sql_literal(b)}",
        ["code", "date", "shares_outstanding", "knowledge_date"],
        dtypes={"code": str, "shares_outstanding": float}, timestamps=["date", "knowledge_date"])
    codes = np.asarray([str(c) for c in codes])
    shares, ref = _asof_shares(sh, codes, dates)
    kinds = ", ".join(f"'{k}'" for k in SHARE_EVENT_KINDS)
    ev = fetch_frame(
        con,
        f"SELECT code, date, upd_rt FROM corporate_actions WHERE kind IN ({kinds}) "
        f"AND upd_rt <> 0 AND upd_rt > -100 AND date <= {sql_literal(b)}",
        ["code", "date", "upd_rt"], dtypes={"code": str, "upd_rt": float}, timestamps=["date"])
    return _roll_events(shares, ref, ev, codes, dates)


def _first_disclosures(con, end: str) -> pd.DataFrame:
    """분기별 첫 공시일 min(rcept_dt) — prelim·pl_change·periodic(F2). rcept_dt ≤ END 만."""
    with con.cursor() as cur:
        cur.execute("SELECT code, period, min(rcept_dt) FROM earnings_disclosures "
                    "WHERE kind IN ('prelim', 'pl_change', 'periodic') AND rcept_dt <= %s GROUP BY code, period",
                    (str(pd.Timestamp(end).date()),))
        df = pd.DataFrame(cur.fetchall(), columns=["code", "period", "first"])
    df["code"] = df["code"].astype(str)
    df["first"] = pd.to_datetime(df["first"])
    assert df.empty or df["first"].max() <= pd.Timestamp(end)
    return df


def _delisted_last(con, codes, dates) -> np.ndarray:
    """폐지 종목의 마지막 거래일 열 번호(창 끝 뒤 폐지·상장은 큰 값)."""
    with con.cursor() as cur:
        cur.execute("SELECT code, last_trade_date FROM delisted_stocks")
        rows = cur.fetchall()
    last = {str(c): pd.Timestamp(d) for c, d in rows if d is not None and _ISO.match(str(d))}
    out = np.full(len(codes), 10**9, dtype=np.int64)
    didx = pd.DatetimeIndex(dates)
    for i, c in enumerate(codes):
        d = last.get(str(c))
        if d is not None:
            j = int(didx.searchsorted(d, side="right")) - 1
            if j < len(didx) - 1:  # 창 끝 이후 폐지는 창 안 폐지가 아니다
                out[i] = j
    return out


def delisted_ever(codes) -> np.ndarray:
    """``delisted_stocks`` 에 있는 종목(언제 폐지됐든) — 0단계 (c) 결측률 비교용."""
    con = _db()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT code FROM delisted_stocks")
            s = {str(r[0]) for r in cur.fetchall()}
    finally:
        con.close()
    return np.array([str(c) in s for c in codes])


def code_hash() -> str:
    """패널을 만드는 코드의 해시 — 캐시 이름에 넣어 코드를 고치면 옛 패널을 못 읽게 한다(정정 ML-2)."""
    import swing_it.features.fundamentals as fund

    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in (Path(__file__).resolve(), here / "pead_optimizer_c0.py", here / "pead_optimizer_po4.py",
              here / "pead_optimizer_seam.py", Path(fund.__file__).resolve()):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def build_panels(end: str, cache: Path | None = None, log=print) -> dict:
    """END 까지의 패널. 캐시 이름에 END·표 지문이 들어간다(표가 바뀌면 다시 만든다)."""
    import pead_optimizer_po4 as P4
    from krx_quant_core.data.daily import load_daily_bars
    from krx_quant_core.data.risk_inputs import STYLE_NAMES, load_risk_inputs, style_panels
    from pead_optimizer_c0 import exposures, yoy_signal
    from swing_it.storage import read_investor_trend, read_research_reports

    w = assert_window(end, "eval")
    end = w.end
    t0 = time.time()
    fp = P4.fingerprint("daily_bars_adjusted", end)
    key = hashlib.sha256(json.dumps([end, fp], default=str).encode()).hexdigest()[:12]
    path = None if cache is None else Path(cache) / f"panels_{end}_fp{key}_code{code_hash()}.pkl"
    if path is not None and path.exists():
        P = pickle.load(open(path, "rb"))
        log(f"패널 캐시 {path.name}")
        return P

    con = _db()
    try:
        x0 = load_risk_inputs(LOAD_START, end, conn=con)
        assert x0.dates[-1] <= pd.Timestamp(end), "로더가 END 뒤 날짜를 냈다"
        x, seam, seam_info = P4.guarded(x0, end)
        codes = np.asarray([str(c) for c in x.codes])
        dates = pd.DatetimeIndex(x.dates)
        log(f"  로더 {len(codes)}종목 × {len(dates)}일 ({dates[0].date()}~{dates[-1].date()}) {time.time() - t0:.0f}s")
        bars = load_daily_bars(dates[0].date(), pd.Timestamp(end).date(), adjusted=True, conn=con)
        bars = bars[bars["code"].astype(str).isin(set(codes))]
        vol = _pivot(bars.assign(code=bars["code"].astype(str)), "volume", codes, dates)
        del bars
        shares = _shares(con, codes, dates, end)
        disc = _first_disclosures(con, end)
        delist_last = _delisted_last(con, codes, dates)
        lo = str(dates[0].date())
        it = read_investor_trend(con, start=lo, end=end)
        assert it.empty or pd.to_datetime(it["date"]).max() <= pd.Timestamp(end)
        rep = read_research_reports(con, end=end)
        assert rep.empty or pd.to_datetime(rep["write_date"]).max() <= pd.Timestamp(end)
    finally:
        con.close()
    fnet = _pivot(it, "foreign_net", codes, dates)
    inet = _pivot(it, "institution_net", codes, dates)
    hold = _pivot(it, "foreign_hold_ratio", codes, dates)
    it_rows = len(it)
    del it
    yoy, sig_info = yoy_signal(codes, dates, end=end, fallback=False)

    sector = x.sector
    has_sec = np.vectorize(lambda s: s is not None, otypes=[bool])(sector)
    msk = has_sec & np.isfinite(x.cap) & (x.cap > 0) & np.isfinite(x.close)
    raw = style_panels(x)
    Xs = exposures(x, raw, msk)
    first = int(np.argmax(has_sec.any(axis=0)))
    rep = rep.assign(code=rep["code"].astype(str), write_date=pd.to_datetime(rep["write_date"]))
    P = {
        "end": end, "codes": codes, "dates": dates, "first": first,
        "close": x.close, "ret": x.ret, "cap": x.cap, "adv": x.adv_value, "tv": x.trade_value,
        "vol": vol, "msk": msk, "lab": sector, "Xs": {k: Xs[k] for k in STYLE_NAMES},
        "shares": shares, "fnet": fnet, "inet": inet, "hold": hold, "yoy": yoy,
        "disc": disc, "reports": rep[["research_id", "code", "broker", "write_date", "goal_price"]],
        "delist_last": delist_last, "seam": seam,
        "info": {"seam": {k: seam_info[k] for k in ("seam_cells", "seam_cells_by_source", "seam_codes")},
                 "signal": sig_info, "investor_trend_rows": int(it_rows), "reports_rows": int(len(rep)),
                 "disclosure_periods": int(len(disc)), "fingerprint": fp},
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(P, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  패널 끝 {time.time() - t0:.0f}s")
    return P


# ── v4 추정 유니버스·잔차 ──────────────────────────────────────────────────────


def v4_factor_returns(P: dict, mask: np.ndarray | None = None):
    """core ``factor_returns`` — 2019-01-02(WICS 첫날) 열부터. ``mask=None`` 이면 v4 마스크."""
    from krx_quant_core.risk.factor_model import factor_returns

    sl = slice(P["first"], None)
    m = P["msk"] if mask is None else mask
    return factor_returns(P["ret"][:, sl], {k: v[:, sl] for k, v in P["Xs"].items()}, P["lab"][:, sl],
                          P["cap"][:, sl], m[:, sl])


def v4_ok(P: dict, fr, a_full: int) -> np.ndarray:
    """``build_risk_model(…, asof=a, spec_shrink=0.1)`` 의 ``ok``(반환 종목) 을 같은 식으로(F·D 만 계산, 모델은 안 만든다).

    ``fr`` 은 :func:`v4_factor_returns` 결과(열 번호는 first 부터). 0단계에서 build_risk_model 과 비트 대조한다.
    """
    from krx_quant_core.risk.factor_model import _dummies, ewma_cov, specific_var

    a = a_full - P["first"]
    F = ewma_cov(fr.f[: a + 1], half_life=V4_HALF_LIFE, nw_lags=V4_NW)
    cap_a = P["cap"][:, a_full]
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.where(cap_a > 0, np.log(cap_a), np.nan)
    D = specific_var(fr.u[:, : a + 1], half_life=V4_HALF_LIFE, shrink=V4_SHRINK, size=size, n_groups=V4_GROUPS)
    names = fr.names
    G = len(names) - 1 - len(P["Xs"])
    N = len(P["codes"])
    dum = _dummies(P["lab"][:, a_full], names[1: 1 + G])
    X = np.hstack([np.ones((N, 1)), dum, np.stack([v[:, a_full] for v in P["Xs"].values()], axis=1)])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(D) & P["msk"][:, a_full]
    if G:
        ok &= X[:, 1: 1 + G].sum(axis=1) == 1.0
    dead = ~np.isfinite(np.diag(F))
    if dead.any():
        ok &= ~(X[:, dead] != 0).any(axis=1)
    return ok


def build_risk_model_codes(P: dict, a_full: int) -> np.ndarray:
    """대조용 — core ``build_risk_model`` 이 내는 종목(행 번호)."""
    from krx_quant_core.risk.factor_model import build_risk_model

    sl = slice(P["first"], None)
    m = build_risk_model(P["ret"][:, sl], {k: v[:, sl] for k, v in P["Xs"].items()}, P["lab"][:, sl],
                         P["cap"][:, sl], P["msk"][:, sl], a_full - P["first"], codes=np.arange(len(P["codes"])),
                         spec_shrink=V4_SHRINK, spec_vra_half_life=None)
    return np.asarray(m.codes, dtype=int)


def universe(P: dict, fr, days: list[int]) -> dict[int, np.ndarray]:
    """U(t) = v4 추정 유니버스 ∧ v4 유니버스 안 시총 상위 400 ∧ ADV ≥ 50억(정정 ML-1 ①). 행 번호(오름차순)."""
    from ml_forecast_common import ADV_MIN_WON, CAP_TOP

    out = {}
    for t in days:
        ok = v4_ok(P, fr, t)
        idx = np.flatnonzero(ok)
        top = idx[np.argsort(-P["cap"][idx, t], kind="stable")[:CAP_TOP]]
        liq = P["adv"][top, t] >= ADV_MIN_WON
        out[t] = np.sort(top[liq])
    return out


def label_mask(P: dict) -> np.ndarray:
    """mask_y[:, s−1] = v4 마스크[:, s−1] ∧ 거래량_s > 0 (PREREG §4 MAJOR-5). 마지막 열은 거짓."""
    m = np.zeros_like(P["msk"])
    with np.errstate(invalid="ignore"):
        traded = P["vol"][:, 1:] > 0
    m[:, :-1] = P["msk"][:, :-1] & traded
    return m
