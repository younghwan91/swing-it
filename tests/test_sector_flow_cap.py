"""sector_flow 의 기간말·월말 섹터 시총이 원래 경로(날짜마다 DB 조회)와 바이트 동일한지.

빠른 길은 ``attach_daily_cap`` 이 이미 붙인 ``cap`` 을 다시 쓴다. 그 값은 같은
``market_cap_asof_bulk`` 를 같은 (code, date) 로 부른 결과라 같아야 한다 — 여기서 그 계약을 지킨다.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swing_it.storage import connect, market_cap_asof_bulk

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("sector_flow", REPO / "scripts" / "sector_flow.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sf = _load()


def _old_sector_cap_monthly(con, df, dates):
    """원래 구현(9611cf4) — 월마다 DB 를 따로 조회한다."""
    months: dict[str, str] = {}
    for d in dates:
        months[d[:7]] = d
    codes = df[["code", "sector", "market"]].drop_duplicates("code")
    out: dict = {}
    for ym, last in sorted(months.items()):
        q = codes[["code"]].copy()
        q["date"] = last
        q["cap"] = market_cap_asof_bulk(con, q).to_numpy() / sf.EOK
        m = q.merge(codes, on="code").dropna(subset=["cap"])
        for (mk, sec), g in m.groupby(["market", "sector"]):
            out.setdefault(mk, {}).setdefault(sec, {})[ym] = round(float(g["cap"].sum()), 1)
    return out


def _build(tmp_path, seed):
    """엣지 케이스가 든 합성 DB 와 load() 모양의 df(code·date·sector·market)."""
    rng = np.random.default_rng(seed)
    con = connect(str(tmp_path / f"t{seed}.db"))
    days = [d.date().isoformat() for d in pd.bdate_range("2025-01-02", periods=150)]
    gap = days[70][:7]
    days = [d for d in days if d[:7] != gap]               # 한 달이 통째로 빈다
    codes = [f"{i:06d}" for i in range(40)]
    sectors = ["반도체", "금융", "제약", "화학"]
    con.executemany(
        "INSERT INTO stocks(code,name,market,sector,kind) VALUES (?,?,?,?,?)",
        [(c, c, "거래소" if i % 2 else "코스닥", sectors[i % 4], "stock")
         for i, c in enumerate(codes)],
    )
    bars, sh, rows = [], [], []
    for i, c in enumerate(codes):
        kind = i % 8                     # 0 주식수 없음 · 1 마지막 0주 · 2 미래 주식수만 · 3 거래정지 · 4 상폐
        last_day = days[int(rng.integers(30, len(days) - 5))] if kind == 4 else days[-1]
        for d in days:
            if d > last_day or (kind == 3 and rng.random() < 0.3):
                continue
            close = None if rng.random() < 0.01 else int(rng.integers(500, 300_000))
            bars.append((c, d, close, close, close, close, 1, 1, "kiwoom"))
            rows.append((c, d))
        if kind == 0:
            continue
        sdays = sorted(set(rng.choice(days, size=int(rng.integers(1, 6)), replace=False)))
        if kind == 2:
            sdays = [days[-10]]
        for j, d in enumerate(sdays):
            v = 0 if kind == 1 and j == len(sdays) - 1 else int(rng.integers(10**6, 9 * 10**8))
            sh.append((c, d, v))
    con.executemany(
        "INSERT INTO daily_bars(code,date,open,high,low,close,volume,trade_value,source)"
        " VALUES (?,?,?,?,?,?,?,?,?)", bars)
    con.executemany(
        "INSERT INTO shares_outstanding_history(code,date,shares_outstanding) VALUES (?,?,?)", sh)
    con.commit()
    meta = pd.DataFrame({"code": codes, "sector": [sectors[i % 4] for i in range(40)],
                         "market": ["거래소" if i % 2 else "코스닥" for i in range(40)]})
    df = pd.DataFrame(rows, columns=["code", "date"]).merge(meta, on="code")
    return con, df, sorted(set(df["date"]))


@pytest.mark.parametrize("seed", range(4))
def test_sector_cap_reuse_matches_db_path(tmp_path, seed):
    con, df, dates = _build(tmp_path, seed)
    with_cap = sf.attach_daily_cap(con, df)
    for last in (dates[-1], dates[len(dates) // 2], "2099-01-01"):
        fast = sf.sector_cap(con, with_cap, last)
        slow = sf.sector_cap(con, df, last)               # cap 열이 없으면 원래대로 DB 조회
        pd.testing.assert_frame_equal(fast.reset_index(drop=True), slow.reset_index(drop=True),
                                      check_exact=True)


@pytest.mark.parametrize("seed", range(4))
def test_sector_cap_monthly_matches_original(tmp_path, seed):
    con, df, dates = _build(tmp_path, seed)
    with_cap = sf.attach_daily_cap(con, df)
    for ds in (dates, dates[: len(dates) // 2], dates[-1:], []):
        want = json.dumps(_old_sector_cap_monthly(con, df, ds), ensure_ascii=False)
        assert json.dumps(sf.sector_cap_monthly(con, with_cap, ds), ensure_ascii=False) == want
        assert json.dumps(sf.sector_cap_monthly(con, df, ds), ensure_ascii=False) == want
