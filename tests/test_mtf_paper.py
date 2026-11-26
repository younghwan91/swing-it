"""전진 팔 mtf30_fwd(``research/paper/mtf_paper.py``) — 합성 자료, DB 없음.

1. 시작일 전 진입은 장부에 없다.
2. 앞부분 안정성: 자료를 d 까지만 준 장부에서 d 까지 청산된 거래는 뒤 자료를 더 준 장부와 같다(표류 0), d 까지 진입한 거래 집합도 같다.
   음성 대조: d 전 분봉을 바꾸면 표류가 잡힌다.
3. 덧붙이기 전용: 같은 asof 는 두 번 쓰지 않는다.
4. 분봉 완결 날 판정.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")

from research.paper import mtf_paper as P  # noqa: E402
from tests.test_mtf_transition import synth  # noqa: E402

TOPN = 12
START = "2026-10-01"


@pytest.fixture(scope="module")
def data():
    m15, dc, tv, codes = synth(seed=7, n_codes=30)
    q = pd.Series(np.exp(np.cumsum(np.random.default_rng(3).normal(0.002, 0.01, len(dc)))) * 800, index=dc.index)
    return m15, dc, tv, codes, q


def _book(data, cut=None, m15=None):
    m15_, dc, tv, codes, q = data
    m15 = m15_ if m15 is None else m15
    if cut is not None:
        c = pd.Timestamp(cut)
        m15, dc, tv, q = m15[m15.ts < c + pd.Timedelta(days=1)], dc[dc.index <= c], tv[tv.index <= c], q[q.index <= c]
    mkt = np.zeros(len(codes), np.int64)
    return P.book(m15, dc, tv, codes, q, mkt, lambda ud: np.full((len(ud), 2), 0.00415), fwd_start=START, topn=TOPN)


@pytest.fixture(scope="module")
def full(data):
    return _book(data)


def test_has_trades_and_none_before_start(full):
    for k in ("main", "shadow"):
        rows = P.trade_rows(full, k)
        assert rows, k
        assert min(r["entry"] for r in rows) >= START
    assert len(P.trade_rows(full, "shadow")) >= len(P.trade_rows(full, "main"))
    assert all(r["reg_on"] for r in P.trade_rows(full, "main"))


@pytest.mark.parametrize("cut", ["2026-10-14", "2026-10-29", "2026-11-06"])
def test_prefix_stability(data, full, cut):
    part = _book(data, cut)
    for k in ("main", "shadow"):
        a, b = P.trade_rows(part, k), P.trade_rows(full, k)
        assert P.drift(a, b, cut) == [], (k, cut)
        hi = str(pd.Timestamp(cut) + pd.Timedelta(days=1))
        assert {(r["code"], r["entry"]) for r in a} == {(r["code"], r["entry"]) for r in b if r["entry"] < hi}, (k, cut)
        sa = {(r["code"], r["entry"]): r["size"] for r in a}
        sb = {(r["code"], r["entry"]): r["size"] for r in b}
        assert all(abs(sa[x] - sb[x]) <= 1 for x in sa), (k, cut)


def test_drift_detects_changed_history(data, full):
    m15 = data[0].copy()
    rows = P.trade_rows(full, "shadow")
    done = [r for r in rows if r["realized"]]
    assert done
    r0 = done[0]
    hit = (m15.code == r0["code"]) & (m15.ts > pd.Timestamp(r0["entry"])) & (m15.ts <= pd.Timestamp(r0["exit"]))
    for c in ("open", "high", "low", "close"):
        m15.loc[hit, c] *= 1.07
    alt = _book(data, m15=m15)
    assert P.drift(rows, P.trade_rows(alt, "shadow"))


def test_append_only(tmp_path):
    p = tmp_path / "l.jsonl"
    P.append_row(p, {"arm": P.ARM, "asof": "2026-11-17", "x": 1})
    with pytest.raises(ValueError):
        P.append_row(p, {"arm": P.ARM, "asof": "2026-11-17", "x": 2})
    P.append_row(p, {"arm": P.ARM, "asof": "2026-11-18", "x": 3})
    assert [r["x"] for r in P.read_ledger(p)] == [1, 3]


def test_complete_days():
    d1, d2 = pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-17")
    rows = [("A", d1 + pd.Timedelta(hours=15)), ("B", d1 + pd.Timedelta(hours=15)), ("A", d2 + pd.Timedelta(hours=15)),
            ("B", d2 + pd.Timedelta(hours=14))]
    m = pd.DataFrame(rows, columns=["code", "ts"])
    assert P.complete_days(m, {d1: {"A", "B"}, d2: {"A", "B"}}) == [d1]
    assert P.complete_days(m, {d1: {"A", "B"}, d2: {"A", "B"}}, cover_min=0.5) == [d1, d2]


def test_drift_catches_added_and_open_changes():
    a = [{"code": "A", "entry": "2026-11-17 10:00:00", "realized": False, "dist": 0.03, "size": 1e6, "exit": None, "why": "미청산", "ret": 0.01}]
    b = [dict(a[0], dist=0.05), {"code": "B", "entry": "2026-11-17 11:00:00", "realized": False, "dist": 0.02, "size": 1e6, "exit": None,
                                 "why": "미청산", "ret": 0.0}]
    w = {x["what"] for x in P.drift(a, b, "2026-11-17")}
    assert w == {"보유 바뀜", "추가"}
    assert P.drift(a, b[1:2] + [{**b[1], "code": "C", "entry": "2026-11-18 09:30:00"}], "2026-11-17")[0]["what"] == "사라짐"
    c = [dict(a[0], realized=True, exit="2026-11-18 09:00:00", why="손절", ret=-0.03)]
    assert P.newly_realized(a, c) == c and P.newly_realized(c, c) == []


def test_rows_ok():
    days = pd.bdate_range("2026-11-09", "2026-11-16")
    m = pd.DataFrame({"code": "A", "ts": np.repeat(days, [26, 26, 26, 26, 26, 10])})
    assert P.rows_ok(m, days[4]) and not P.rows_ok(m, days[5])


def test_union_asof_drops_later_members():
    """--asof 백필: asof 뒤에야 거래대금 상위에 든 종목은 패널에서 빠진다(그날 아침 실행은 몰랐다)."""
    idx = pd.bdate_range("2026-11-16", "2026-11-20")
    mem = pd.DataFrame({"A": [True] * 5, "B": [False, False, False, True, True], "C": [False] * 5}, index=idx)
    m15 = pd.DataFrame({"code": ["A", "B", "A", "B"], "ts": [idx[0], idx[0], idx[3], idx[3]]})
    D = {"mem": mem, "union": ["A", "B"], "m15": m15}
    d = P.union_asof(D, idx[2])
    assert d["union"] == ["A"] and set(d["m15"]["code"]) == {"A"}
    assert P.union_asof(D, idx[3])["union"] == ["A", "B"]
