"""price_adjust: 기업행동 백조정 유틸 테스트."""

import pandas as pd

from swing_it import storage
from swing_it.price_adjust import adjust_prices, diagnose, rebuild_adjusted_table


def _series(closes, code="A"):
    n = len(closes)
    return pd.DataFrame({
        "code": [code] * n,
        "date": [f"2021-01-{i + 1:02d}" for i in range(n)],
        "open": closes, "high": closes, "low": closes, "close": closes,
    })


def test_4to1_split_becomes_continuous():
    # 100까지 상승 후 4:1 분할(→25)로 이후 25~30 유지
    closes = [80, 90, 100, 25, 26, 27, 28, 29]
    df = _series(closes)
    adj = adjust_prices(df)["close"].tolist()
    # 분할 이전 구간은 1/4로 백조정되어 연속(100→25)이어야
    assert abs(adj[2] / adj[3] - 1) < 0.2, adj
    assert abs(adj[0] - 20) < 1e-6 and abs(adj[2] - 25) < 1e-6, adj
    # 분할 이후 구간은 그대로
    assert adj[-1] == 29


def test_transient_spike_not_adjusted():
    # 하루만 튀고 되돌아오는 데이터 스파이크는 분할로 보지 않는다
    closes = [100, 101, 300, 102, 103, 104, 105, 106]
    df = _series(closes)
    adj = adjust_prices(df)["close"].tolist()
    assert adj == [float(x) for x in closes], adj


def test_normal_moves_untouched():
    closes = [100, 105, 110, 108, 112, 115, 113, 117]
    df = _series(closes)
    adj = adjust_prices(df)["close"].tolist()
    assert adj == [float(x) for x in closes]


def test_diagnose_flags_split_date():
    df = _series([80, 90, 100, 25, 26, 27, 28, 29])
    ev = diagnose(df)
    assert len(ev) == 1
    assert ev.iloc[0]["date"] == "2021-01-04"  # 100→25 지점
    assert abs(ev.iloc[0]["ratio"] - 0.25) < 0.01


def test_ohlc_scaled_together():
    n = 8
    df = pd.DataFrame({
        "code": ["A"] * n,
        "date": [f"2021-01-{i + 1:02d}" for i in range(n)],
        "open": [80, 90, 100, 25, 26, 27, 28, 29],
        "high": [82, 92, 102, 26, 27, 28, 29, 30],
        "low": [78, 88, 98, 24, 25, 26, 27, 28],
        "close": [80, 90, 100, 25, 26, 27, 28, 29],
    })
    adj = adjust_prices(df)
    # 분할 이전 high/low 도 동일 배수(0.25)로 조정
    assert abs(adj.iloc[2]["high"] - 25.5) < 1e-6
    assert abs(adj.iloc[2]["low"] - 24.5) < 1e-6


def test_rebuild_adjusted_table_writes_back_adjusted_rows_to_db():
    con = storage.connect(":memory:")
    con.executemany(
        "INSERT INTO daily_bars(code,date,open,high,low,close,volume,trade_value) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            ("005930", "2021-01-01", 80, 82, 78, 80, 1000, 80000),
            ("005930", "2021-01-02", 90, 92, 88, 90, 1000, 90000),
            ("005930", "2021-01-03", 100, 102, 98, 100, 1000, 100000),
            ("005930", "2021-01-04", 25, 26, 24, 25, 4000, 100000),  # 4:1 split day
            ("005930", "2021-01-05", 26, 27, 25, 26, 4000, 104000),
            ("005930", "2021-01-06", 27, 28, 26, 27, 4000, 108000),
            ("005930", "2021-01-07", 28, 29, 27, 28, 4000, 112000),
        ],
    )
    con.commit()

    n = rebuild_adjusted_table(con)
    assert n == 7

    import pandas as pd
    adj = pd.read_sql_query(
        "SELECT * FROM daily_bars_adjusted ORDER BY date", con)
    # pre-split day is back-adjusted to the post-split scale (~25), not the raw 100
    assert abs(adj.iloc[2]["close"] - 25.0) < 0.5
    # post-split rows are unchanged
    assert abs(adj.iloc[3]["close"] - 25.0) < 1e-6
    # volume/trade_value are untouched (adjust_volume defaults to False)
    assert adj.iloc[0]["volume"] == 1000
    assert adj.iloc[0]["trade_value"] == 80000
    con.close()


def test_rebuild_propagates_source_to_adjusted_table():
    """조정가 테이블이 daily_bars.source 를 그대로 물려받는지.

    끊기면 조용히 DEFAULT 'kiwoom' 으로 채워져, 폐지 종목의 근사 거래대금이
    실측치처럼 보인다 — ADV 문턱이 전적으로 trade_value 로 돌기 때문에 유니버스
    편입까지 영향이 가는데 되짚을 단서가 사라진다.
    """
    from swing_it.price_adjust import rebuild_adjusted_table
    from swing_it.storage import DAILY_BAR_COLUMNS, connect

    con = connect(":memory:")
    cols = [*DAILY_BAR_COLUMNS, "source"]
    ph = ",".join(["?"] * len(cols))
    rows = [
        ("K", "2020-01-02", 100, 110, 90, 100, 10, 1, "kiwoom"),
        ("K", "2020-01-03", 100, 110, 90, 100, 10, 1, "kiwoom"),
        ("N", "2020-01-02", 50, 55, 45, 50, 20, 2, "naver"),
        ("N", "2020-01-03", 50, 55, 45, 50, 20, 2, "naver"),
    ]
    con.executemany(
        f"INSERT INTO daily_bars({','.join(cols)}) VALUES({ph})", rows)  # noqa: S608 — 모듈 상수
    con.commit()

    rebuild_adjusted_table(con)

    got = dict(con.execute(
        "SELECT DISTINCT code, source FROM daily_bars_adjusted").fetchall())
    assert got == {"K": "kiwoom", "N": "naver"}
    con.close()


# ---------------------------------------------------------------------------
# 원주가 대조(raw_close·flu_rt) — 수집 이음매·하한가·기업행동 구분
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from swing_it import price_adjust as pa  # noqa: E402


def _frame(db, raw=None, flu=None, code="A", start=0, year=2021):
    """db(수정주가)·raw(원주가)·flu(등락률 ×100) 종가로 입력 프레임. None 원소는 결측."""
    n = len(db)
    d = {
        "code": [code] * n,
        "date": [f"{year}-{1 + (start + i) // 28:02d}-{1 + (start + i) % 28:02d}" for i in range(n)],
        "open": db, "high": db, "low": db, "close": db,
    }
    if raw is not None:
        d["raw_close"] = [np.nan if x is None else x for x in raw]
    if flu is not None:
        d["flu_rt"] = [np.nan if x is None else x for x in flu]
    return pd.DataFrame(d)


def _flu(raw, base=None):
    """원주가와 기준가로 키움 flu_rt(bp). base[t] 없으면 전일 원 종가."""
    out = [None]
    for t in range(1, len(raw)):
        b = raw[t - 1] if base is None or base[t] is None else base[t]
        out.append(round((raw[t] / b - 1) * 1e4, 2))
    return out


def _rets(s):
    a = np.asarray(s, float)
    return a[1:] / a[:-1]


def test_canary_025620_seam_below_db_scale_jump():
    # 025620 2021-07-29: daily_bars 1,780 → 29,179(×16.4), 원주가 1,780 → 1,810(+1.7%). 이음매.
    raw = [1850, 1775, 1810, 1780, 1810, 1880, 2040, 2185]
    L = [1.0] * 4 + [16.1209] * 4
    db = [round(r * x) for r, x in zip(raw, L)]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw)))["close"]
    r = _rets(adj)[3]
    assert abs(r - 1) < 0.3 and abs(r - 1810 / 1780) < 1e-3, r
    assert adj.iloc[-1] == db[-1]  # 마지막 날은 그대로(뒤 기준)


def test_canary_258830_seam_then_true_split():
    # 258830: 2021-07-29 이음매(daily_bars 24,500 → 4,900, 원주가 그대로) + 2021-10-20 진짜 5:1 분할(원주가 35,000 →
    # 6,600, daily_bars 7,000 → 6,600 은 이미 수정). 결과: 이음매 날 |r|<0.3, 분할 날 수익률 = 수정주가의 −5.7%.
    raw = [22650, 24500, 24500, 28800, 35000, 35000, 6600, 7010]
    L = [1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0]
    db = [r * x for r, x in zip(raw, L)]
    base = [None] * 6 + [7000, None]  # 분할 날 기준가 = 35,000 × 0.2
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw, base)))["close"].tolist()
    r = _rets(adj)
    assert abs(r[1] - 1) < 0.3 and abs(r[1] - 1.0) < 1e-9, r  # 이음매 날: 원주가 수익률(0%)
    assert abs(r[5] - 6600 / 7000) < 1e-9, r  # 분할 날: 수정주가 수익률 그대로
    assert abs(adj[0] - 22650 * 0.2) < 1e-6  # 이음매 앞도 분할 기준으로 내려온다


def test_canary_008830_seam_raw_up_3pct():
    # 008830 2019-02-27: daily_bars 28,850 → 2,975(−89.7%), 원주가 +3.1%.
    raw = [28750, 29100, 29100, 28850, 29750, 32600, 37500, 39200]
    L = [1.0] * 4 + [0.1] * 4
    db = [r * x for r, x in zip(raw, L)]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw)))["close"]
    r = _rets(adj)[3]
    assert abs(r - 1) < 0.3 and abs(r - 29750 / 28850) < 1e-9, r


def test_canary_001380_limit_down_is_not_a_split():
    # 001380 2025-03-26: 원주가 3,500 → 2,450 정확히 −30% 하한가. daily_bars 는 뒤 병합(×2.0037)이 들어간
    # 수정가 7,013 → 4,909(비율 0.69998)라 옛 규칙이 분할로 읽어 수익률 0·앞 이력 ×0.7 을 만들었다.
    raw = [3405, 3285, 3565, 3500, 2450, 2255, 2245, 2135]
    db = [6822, 6582, 7143, 7013, 4909, 4518, 4498, 4278]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw)))["close"].tolist()
    assert abs(_rets(adj)[3] - 0.70) < 1e-3
    assert adj == [float(x) for x in db]  # 아무것도 곱하지 않는다
    # 원주가가 없어 옛 규칙으로 가도 반올림 폭만큼 내린 문턱(DOWN − _step_tol)이라 분할로 읽지 않는다
    legacy = pa.adjust_prices(_frame(db))["close"].tolist()
    assert abs(_rets(legacy)[3] - 0.70) < 1e-3


def test_canary_086900_limit_down_without_referee():
    # 086900 2020-04-20 류: −30% 하한가인데 심판(flu_rt)이 없는 날도 원주가가 제한폭 안이면 진짜 움직임.
    raw = [10000, 10100, 10000, 7000, 7100, 7200, 7300]
    db = [x * 0.5 for x in raw]
    adj = pa.adjust_prices(_frame(db, raw))["close"].tolist()
    assert abs(_rets(adj)[2] - 0.70) < 1e-9
    assert adj == [float(x) for x in db]


def test_true_split_already_adjusted_is_kept():
    # 258830 2021-10-20 류: 원주가가 분할로 뛰고 수정주가는 이어진다 — 그대로 둔다(심판 有/無 둘 다).
    raw = [35000, 35000, 35000, 6600, 7010, 7500, 7840]
    db = [7000, 7000, 7000, 6600, 7010, 7500, 7840]
    base = [None] * 3 + [7000, None, None, None]
    for flu in (_flu(raw, base), None):
        adj = pa.adjust_prices(_frame(db, raw, flu))["close"].tolist()
        assert adj == [float(x) for x in db], flu


def test_unreflected_split_in_daily_appended_region_is_adjusted():
    # 매일 덧붙인 최근 구간: 수정주가도 원주가처럼 분할로 뛴다(q=1). 심판이 기준가 재설정을 보여 주면 조정하고,
    # 그날 수익률은 기준가 대비 시장 수익률(m)이 된다.
    raw = [35000, 35500, 35000, 6600, 7010, 7500, 7840]
    base = [None] * 3 + [7000, None, None, None]
    adj = pa.adjust_prices(_frame(list(raw), raw, _flu(raw, base)))["close"].tolist()
    r = _rets(adj)
    assert abs(r[2] - 6600 / 7000) < 1e-6, r
    assert abs(adj[0] - 7000) < 0.01  # flu_rt 는 bp 소수 둘째 자리까지
    ev = pa.diagnose(_frame(list(raw), raw, _flu(raw, base)))
    assert ev["kind"].tolist() == ["unreflected"]


def test_unlimited_genuine_crash_is_kept():
    # 정리매매처럼 가격제한이 없는 날의 −70% — 원주가·수정주가·심판이 모두 같이 뛰면 진짜 움직임.
    raw = [1000, 990, 300, 280, 250, 240, 230]
    adj = pa.adjust_prices(_frame(list(raw), raw, _flu(raw)))["close"].tolist()
    assert adj == [float(x) for x in raw]
    # 옛 규칙은 같은 하락을 분할로 지운다
    assert pa.adjust_prices(_frame(list(raw)))["close"].tolist() != [float(x) for x in raw]


def test_sub30_seam_is_removed():
    # 옛 규칙의 30% 문턱 아래(이음매 22%) — 원주가가 이어지면 이음매로 지운다.
    raw = [10000, 10100, 10050, 10200, 10150, 10300, 10250, 10400]
    L = [1.0] * 4 + [0.78] * 4
    db = [r * x for r, x in zip(raw, L)]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw)))["close"].tolist()
    np.testing.assert_allclose(_rets(adj), _rets(raw), rtol=1e-12)
    ev = pa.diagnose(_frame(db, raw, _flu(raw)))
    assert len(ev) == 1 and ev.iloc[0]["kind"] == "seam" and abs(ev.iloc[0]["ratio"] - 0.78) < 1e-3


def test_seam_in_volatile_run_is_removed():
    # 이음매(×1.6) 앞뒤가 ±25% 로 출렁이면 옛 3일 유지 검사가 떨어뜨렸다. 원주가 대조는 날마다 판단한다.
    raw = [10000, 12800, 9700, 12400, 9400, 12000, 9100, 11700]
    L = [1.0] * 4 + [1.6] * 4
    db = [r * x for r, x in zip(raw, L)]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw)))["close"].tolist()
    np.testing.assert_allclose(_rets(adj), _rets(raw), rtol=1e-12)
    legacy = pa.adjust_prices(_frame(db))["close"].tolist()
    assert abs(_rets(legacy)[3] / _rets(raw)[3] - 1) > 0.5  # 옛 규칙은 못 지운다


def test_small_corporate_action_is_kept_with_referee():
    # 권리락 −16%(30% 안쪽 기업행동): 원주가는 기준가 재설정으로 빠지고 수정주가는 이어진다. 심판이 수정주가 편.
    raw = [10000, 10100, 10000, 8500, 8600, 8700, 8800]
    db = [8400, 8484, 8400, 8500, 8600, 8700, 8800]  # 앞 이력 × 0.84 (수정 반영)
    base = [None] * 3 + [8400, None, None, None]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw, base)))["close"].tolist()
    assert adj == [float(x) for x in db]


def test_small_step_without_referee_kept_unless_clustered():
    # 심판 없는 날 제한폭 안 계단: 한 종목이면 기업행동일 수 있어 그대로, 같은 날 CLUSTER_MIN 종목 이상이면 이음매.
    raw = [10000, 10100, 10050, 10200, 10150, 10300, 10250, 10400]
    L = [1.0] * 4 + [0.8] * 4
    db = [r * x for r, x in zip(raw, L)]
    one = pa.adjust_prices(_frame(db, raw))["close"].tolist()
    assert one == [float(x) for x in db]
    many = pd.concat([_frame(db, raw, code=f"C{i:02d}") for i in range(pa.CLUSTER_MIN)], ignore_index=True)
    adj = pa.adjust_prices(many)
    for _, g in adj.groupby("code"):
        np.testing.assert_allclose(_rets(g.sort_values("date")["close"]), _rets(raw), rtol=1e-12)
    assert set(pa.diagnose(many)["kind"]) == {"seam_cluster"}


def test_rounding_noise_on_small_adjusted_prices_untouched():
    # 수정가가 작으면(≈50원) 정수 반올림만으로 하루 1% 가까이 어긋난다 — 계단이 아니다.
    raw = [50000, 50600, 50300, 50900, 50500, 51200, 50800]
    db = [round(r / 1000) for r in raw]  # 50, 51, 50, 51, 50, 51, 51
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw)))["close"].tolist()
    assert adj == [float(x) for x in db]


def test_legacy_rule_only_where_raw_missing():
    # 앞쪽은 원주가 없음(옛 규칙: 4:1 분할 탐지), 뒤쪽은 원주가 있음(하한가 −30% 는 그대로).
    db = [80, 90, 100, 25, 26, 27, 28, 29, 20.3, 21, 22, 23]
    raw = [None] * 7 + [29, 20.3, 21, 22, 23]
    adj = pa.adjust_prices(_frame(db, raw, _flu([x or 1 for x in raw])))["close"].tolist()
    r = _rets(adj)
    assert abs(r[2] - 1) < 1e-9  # 옛 규칙이 분할을 지움
    assert abs(r[7] - 0.70) < 1e-9  # 원주가 구간의 하한가는 살아 있다
    kinds = pa.diagnose(_frame(db, raw, _flu([x or 1 for x in raw])))["kind"].tolist()
    assert kinds == ["legacy_split"]


def test_unsorted_multi_code_input_and_volume():
    raw = [10000, 10100, 10050, 10200, 10150]
    db = [r * x for r, x in zip(raw, [1, 1, 0.5, 0.5, 0.5])]
    a = _frame(db, raw, _flu(raw), code="A")
    b = _frame([5.0] * 5, [5.0] * 5, [0.0] * 5, code="B")
    df = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=0)
    df["volume"] = 100
    out = pa.adjust_prices(df, adjust_volume=True)
    A = out[out.code == "A"].sort_values("date")
    np.testing.assert_allclose(_rets(A["close"]), _rets(raw), rtol=1e-12)
    assert A["volume"].iloc[0] == 200  # 앞 이력 가격 ×0.5 → 거래량 ×2
    assert (out[out.code == "B"]["close"] == 5.0).all()
    assert "raw_close" in out.columns  # 열 구성은 입력 그대로


def test_rebuild_reads_raw_from_supply_demand_on_sqlite():
    # sqlite 에는 daily_bars_raw 가 없다 — supply_demand.close·flu_rt 로 이음매를 찾아 지운다.
    con = storage.connect(":memory:")
    raw = [10000, 10100, 10050, 10200, 10150, 10300]
    L = [1.0, 1.0, 0.8, 0.8, 0.8, 0.8]
    flu = _flu(raw)
    for i, (r, x) in enumerate(zip(raw, L)):
        d = f"2021-01-{i + 1:02d}"
        c = round(r * x)
        con.execute("INSERT INTO daily_bars(code,date,open,high,low,close,volume,trade_value) VALUES (?,?,?,?,?,?,?,?)",
                    ("A", d, c, c, c, c, 10, 1))
        con.execute("INSERT INTO supply_demand(code,date,close,flu_rt,source) VALUES (?,?,?,?,?)",
                    ("A", d, r, flu[i], "kiwoom"))
    con.commit()
    assert rebuild_adjusted_table(con) == len(raw)
    adj = pd.read_sql_query("SELECT close FROM daily_bars_adjusted WHERE code='A' ORDER BY date", con)["close"]
    np.testing.assert_allclose(_rets(adj), _rets(raw), rtol=1e-3)
    con.close()


def test_module_imports_only_numpy_pandas():
    # quant-airflow DAG 가 PYTHONPATH 로 직접 실행한다 — 의존성이 늘면 운영 런이 죽는다(2026-09-14).
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(pa.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert mods <= {"__future__", "typing", "numpy", "pandas", "argparse", "swing_it"}, mods


# --- 2026-09-14~ 애프터마켓: 원 종가 = 20:00 통합가, flu_rt 기준가 = 전일 정규장 종가 -------------------------------
def _am_frame(db, raw, flu, code="A"):
    # 10월(애프터마켓 뒤) 날짜. 첫 행은 기준 없음
    return _frame(db, raw, flu, code=code, start=9 * 28, year=2026)


def test_aftermarket_upper_limit_day_is_kept():
    # 0035S0 2026-09-29(상장 첫날 +50%) → 09-30 상한가: 원 종가 비 +43.6% 는 전일 애프터마켓 하락 탓이다(전일 정규장
    # 29,900·통합 27,050, 기준가 = 29,900). 기업행동이 아니다 — 수정주가·원주가 수익률 그대로.
    raw = [18000, 27050, 38850, 30900, 31750]
    flu = [None, 5028, 2993, -2046, 32]
    adj = pa.adjust_prices(_am_frame(list(raw), raw, flu))["close"].tolist()
    assert adj == [float(x) for x in raw]


def test_aftermarket_session_drop_is_not_a_base_reset():
    # 475830 2026-09-17: 정규장 53,400(+5.3%) → 애프터마켓 35,500(하한가). 다음 날 기준가 53,400 이 전일 원 종가의 1.5배라도
    # 전일 flu(−30%) 로 본 가능 범위 [0.7/0.7, 1.3/0.7] 안이라 기준가 재설정이 아니다.
    raw = [50700, 50800, 35500, 37400, 26550]
    flu = [None, 20, -2998, -2996, -2901]
    adj = pa.adjust_prices(_am_frame(list(raw), raw, flu))["close"].tolist()
    assert adj == [float(x) for x in raw]


def test_aftermarket_seam_from_partial_rewrite_is_removed():
    # 117670 2026-10-06: 16:00 수집 15일 창이 10-21 권리락(−50%) 수정분을 10-06 부터만 다시 써 daily_bars 자체에
    # 계단(×0.5)이 생겼다. 원주가는 이어지고 심판도 원주가 편 → 이음매로 지운다. 권리락 날은 그대로.
    raw = [10200, 10420, 10310, 10500, 10400, 5300, 5400]
    db = [10200, 10420, 5155, 5250, 5200, 5300, 5400]
    base = [None] * 5 + [5200, None]
    flu = _flu(raw, base)
    flu[2] = 118  # 기준가 = 전일 정규장(애프터마켓과 조금 다름)
    adj = pa.adjust_prices(_am_frame(db, raw, flu))["close"].tolist()
    r = _rets(adj)
    assert abs(r[1] - 10310 / 10420) < 1e-9, r
    assert abs(r[4] - 5300 / 5200) < 1e-9, r


def test_base_reset_missing_from_events_is_kept():
    # 298000 2026-06-15 류: 키움 이벤트에 없는 기준가 재설정(×2). 수정주가는 이어지고(−29%) 원주가만 ×1.41 —
    # 심판(flu_rt = 수정주가 수익률)이 수정주가 편이라 그대로 둔다.
    raw = [10000, 10100, 10050, 14210, 14300]
    db = [20000, 20200, 20100, 14210, 14300]
    base = [None] * 3 + [20100, None]
    adj = pa.adjust_prices(_frame(db, raw, _flu(raw, base)))["close"].tolist()
    assert adj == [float(x) for x in db]


def test_raw_snapshot_on_last_raw_day_is_not_used():
    # daily_bars_raw 의 마지막 날은 19:00 스냅숏(애프터마켓 20:00 마감 전) — 확정 봉과 다르다. 그날은
    # supply_demand.close(확정)로 대신해 가짜 이음매를 만들지 않는다.
    con = storage.connect(":memory:")
    con.execute("CREATE TABLE daily_bars_raw (code TEXT, date TEXT, close INTEGER, PRIMARY KEY (code, date))")
    closes = [10000, 10100, 10050, 10200]
    for i, c in enumerate(closes):
        d = f"2026-10-{i + 26:02d}"
        con.execute("INSERT INTO daily_bars(code,date,open,high,low,close,volume,trade_value) VALUES (?,?,?,?,?,?,?,?)",
                    ("A", d, c, c, c, c, 10, 1))
        con.execute("INSERT INTO supply_demand(code,date,close,flu_rt,source) VALUES (?,?,?,?,?)",
                    ("A", d, c, None if i == 0 else round((c / closes[i - 1] - 1) * 1e4, 2), "kiwoom"))
        snap = 11000 if i == len(closes) - 1 else c  # 마지막 날 스냅숏은 확정치와 다르다
        con.execute("INSERT INTO daily_bars_raw(code,date,close) VALUES (?,?,?)", ("A", d, snap))
    con.commit()
    df = pa.read_inputs(con)
    assert df.sort_values("date")["raw_close"].tolist() == closes
    rebuild_adjusted_table(con)
    adj = pd.read_sql_query("SELECT close FROM daily_bars_adjusted ORDER BY date", con)["close"].tolist()
    assert adj == [float(x) for x in closes]
    con.close()


# --- 리뷰(PR #12) 반영 ---------------------------------------------------------------------------------------------
def test_canary_076610_seam_and_capital_reduction_same_day():
    # 076610 2024-01-08: 수집 이음매 날에 감자(upd_rt 436.54%)가 겹쳤다. r_db 4.21·r_raw 4.87 둘 다 심판(m 0.907)과
    # 어긋난다 — 가까운 쪽을 고르면 +321% 가 남는다. 둘 다 틀리면 그날 수익률 = m.
    raw = [300, 305, 312, 1518, 1500, 1490]
    db = [x * 1315 / 4.214744 / 312 for x in raw[:3]] + [1315, 1300, 1290]  # 이음매 날 r_db = 4.214744
    flu = _flu(raw)
    flu[3] = -932  # m = 0.9068
    adj = pa.adjust_prices(_frame(db, raw, flu))["close"].tolist()
    assert abs(_rets(adj)[2] - 0.9068) < 1e-4, _rets(adj)
    assert pa.diagnose(_frame(db, raw, flu))["kind"].tolist() == ["both_off"]


def test_canary_139050_unrefereed_consolidation_goes_to_legacy():
    # 139050 2023-06-28 네이버 폐지 백필: 426 → 6,570(+1442%, 상장주식수 12.77:1 병합). 심판이 없고 원주가·수정주가가
    # 같이 뛰어 q=1 — ×2 를 넘으면 옛 규칙(3일 유지)으로 분할 처리한다.
    raw = [430, 428, 426, 6570, 6600, 6550, 6500, 6520]
    adj = pa.adjust_prices(_frame(list(raw), raw))["close"].tolist()
    assert abs(_rets(adj)[2] - 1) < 0.3


def test_spac_relisting_jump_without_referee_is_kept():
    # 366330 2023-06-02 류: 거래정지(2,200 고정) 뒤 스팩 합병 재상장 첫날 +55.7% — 시가가 평가가격의 50~200% 에서 정해져
    # 진짜로 나온다. ×2 안이라 그대로 둔다(v2 와 같다).
    raw = [2200, 2200, 2200, 3425, 3570, 4380, 4090]
    adj = pa.adjust_prices(_frame(list(raw), raw))["close"].tolist()
    assert adj == [float(x) for x in raw]


def test_canary_138360_referee_outside_limit_is_not_used():
    # 138360 2023-01-27: flu_rt −52.9%(가격제한 없는 날). 그런 m 은 심판으로 쓰지 않고 제한폭 규칙으로 — 수정주가 그대로.
    raw = [4000, 4075, 4480, 2110, 2120, 2130]
    db = [8000, 8150, 2240, 2110, 2120, 2130]
    db[2] = 2110 / 0.941964
    db[1] = db[2] / (4480 / 4075)
    db[0] = db[1] / (4075 / 4000)
    flu = _flu(raw)
    flu[3] = -5290
    adj = pa.adjust_prices(_frame(db, raw, flu))["close"].tolist()
    assert abs(_rets(adj)[2] - 0.941964) < 1e-6


def test_aftermarket_small_seam_paired_with_later_event():
    # 207940: 15일 재기록 창이 10-02 권리락(−0.77%) 수정분을 09-17 부터만 덮었다 → 09-17 이음매 ×0.9923, 10-02 에 그
    # 역수 계단. 짝이 있으면 앞은 이음매(원주가 수익률), 뒤는 기업행동(수정주가 그대로).
    raw = [1405000, 1404000, 1399000, 1403000, 1428000, 1366000, 1318000]
    L = [1.0, 0.9923, 0.9923, 0.9923, 0.9923, 1.0, 1.0]
    db = [r * x for r, x in zip(raw, L)]
    base = [None] * 5 + [1428000 * 0.9923, None]
    flu = _flu(raw, base)
    adj = pa.adjust_prices(_am_frame(db, raw, flu))["close"].tolist()
    r = _rets(adj)
    assert abs(r[0] - 1404000 / 1405000) < 1e-9, r
    assert abs(r[4] - (1366000 / (1428000 * 0.9923))) < 1e-9, r


def test_aftermarket_small_unpaired_step_is_left_alone():
    # 애프터마켓 뒤 짝 없는 작은 계단(3%)은 심판 잡음보다 작아 가를 수 없다 — 이력 전체를 다시 곱하지 않는다.
    raw = [10000, 10100, 10300, 10250, 10400]
    db = [10000, 10100, 10300 * 0.97, 10250 * 0.97, 10400 * 0.97]
    flu = [None, 100, 150, -50, 146]
    out = pa.adjust_prices(_am_frame(db, raw, flu))["close"].tolist()
    assert out == [float(x) for x in db]
    assert pa.diagnose(_am_frame(db, raw, flu)).empty


def test_read_inputs_prefers_supply_demand_then_raw_table():
    # 확정치(supply_demand.close)가 먼저, 없는 날만 daily_bars_raw.close.
    con = storage.connect(":memory:")
    con.execute("CREATE TABLE daily_bars_raw (code TEXT, date TEXT, close INTEGER, PRIMARY KEY (code, date))")
    for d, c, sd, raw in [("2026-10-28", 100, None, 100), ("2026-10-29", 101, 101, 101), ("2026-10-30", 102, 102, 110)]:
        con.execute("INSERT INTO daily_bars(code,date,open,high,low,close,volume,trade_value) VALUES (?,?,?,?,?,?,?,?)",
                    ("A", d, c, c, c, c, 1, 1))
        if sd is not None:
            con.execute("INSERT INTO supply_demand(code,date,close,flu_rt,source) VALUES (?,?,?,?,?)",
                        ("A", d, sd, 0.0, "kiwoom"))
        con.execute("INSERT INTO daily_bars_raw(code,date,close) VALUES (?,?,?)", ("A", d, raw))
    con.commit()
    got = pa.read_inputs(con).sort_values("date")["raw_close"].tolist()
    assert got == [100, 101, 102]
    con.close()


def test_canary_056730_capital_reduction_after_suspension_keeps_db():
    # 056730 2017-01-13: 오랜 거래정지 뒤 감자(6.0035:1) 재상장. 주주 수익률 5,000/(578×6.0035) = 1.441 = r_db 이고
    # m(1.086)은 평가가격 기준가 대비라 주주 수익률이 아니다. r_db 가 ×2 안이므로 both_off 로 보지 않는다.
    raw = [578, 578, 578, 5000, 5050, 5100]
    db = [578 * 6.0035, 578 * 6.0035, 578 * 6.0035, 5000, 5050, 5100]
    flu = _flu(raw)
    flu[3] = 858
    adj = pa.adjust_prices(_frame(db, raw, flu))["close"].tolist()
    assert abs(_rets(adj)[2] - 5000 / (578 * 6.0035)) < 1e-9
    assert adj == [float(x) for x in db]


def test_transfer_listing_day_is_not_an_unreflected_event():
    # 107640 2024-06-24 류: 코넥스 → 코스닥 이전상장 첫날. 기준가가 공모가(30,000)로 재설정돼 m = +13.3% 지만 주주가 본
    # 움직임은 원 종가 55,000 → 34,000. 앞 20일 거래가 얇으면(평균 < 그날 1%) 기업행동으로 보지 않는다.
    n_pre = 25
    raw = [55000] * n_pre + [34000, 33000, 32500]
    flu = [None] + [0.0] * (n_pre - 1) + [1333.0, round((33000 / 34000 - 1) * 1e4, 2), round((32500 / 33000 - 1) * 1e4, 2)]
    df = _frame(list(raw), raw, flu)
    df["volume"] = [2000] * n_pre + [3_000_000, 900_000, 500_000]
    adj = pa.adjust_prices(df)["close"].tolist()
    assert adj == [float(x) for x in raw]
    # 거래가 두꺼웠다면 같은 모양은 미반영 기업행동이다
    df["volume"] = [1_000_000] * len(raw)
    assert pa.diagnose(df)["kind"].tolist() == ["unreflected"]


def test_benchmark_etf_raw_only_last_day_snapshot_dropped():
    # supply_demand 행이 없는 ETF(069500 등)는 원주가 표뿐 — 그 표의 마지막 날 19:00 스냅숏은 쓰지 않는다.
    con = storage.connect(":memory:")
    con.execute("CREATE TABLE daily_bars_raw (code TEXT, date TEXT, close INTEGER, PRIMARY KEY (code, date))")
    for d, c, raw in [("2026-10-28", 100, 100), ("2026-10-29", 101, 101), ("2026-10-30", 102, 110)]:
        con.execute("INSERT INTO daily_bars(code,date,open,high,low,close,volume,trade_value) VALUES (?,?,?,?,?,?,?,?)",
                    ("069500", d, c, c, c, c, 1, 1))
        con.execute("INSERT INTO daily_bars_raw(code,date,close) VALUES (?,?,?)", ("069500", d, raw))
    con.commit()
    got = pa.read_inputs(con).sort_values("date")["raw_close"].tolist()
    assert got[:2] == [100, 101] and np.isnan(got[2])
    con.close()
