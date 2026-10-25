"""토스 뉴스 아카이브 → 15:10 경계 종목×일 배열 테스트."""

import json

import pandas as pd
import pytest

from research.combined.news import news_arrays

DAYS = pd.DatetimeIndex(["2023-03-16", "2023-03-17", "2023-03-20"])  # 목·금·월


def _write(d, code, rows):
    with open(d / f"{code}.jsonl", "w") as f:
        for i, (ts, title) in enumerate(rows):
            f.write(
                json.dumps({"id": i, "title": title, "createdAt": ts, "code": code})
                + "\n"
            )


def test_cutoff_and_holiday_roll(tmp_path):
    _write(
        tmp_path,
        "000001",
        [
            ("2023-03-17T15:09:00", "[특징주] A"),
            ("2023-03-17T15:11:00", "[특징주] B"),
            ("2023-03-18T10:00:00", "[특징주] C"),
        ],
    )
    r = news_arrays(["000001"], DAYS, tmp_path)
    assert r["n_feat"].dtype == "uint16"
    assert r["n_feat"][0].tolist() == [0, 1, 2]
    assert r["n_all"][0].tolist() == [0, 1, 2]


def test_unknown_differs_from_zero(tmp_path):
    _write(tmp_path, "000001", [("2023-03-17T09:00:00", "평범")])
    r = news_arrays(["000001", "999999"], DAYS, tmp_path)
    assert not r["news_known"][1].any()
    assert r["news_known"][0].tolist() == [False, True, True]
    assert r["n_all"][0].tolist() == [0, 1, 0]


def test_dedup_id_code(tmp_path):
    with open(tmp_path / "000001.jsonl", "w") as f:
        for _ in range(2):
            f.write(
                json.dumps(
                    {
                        "id": 7,
                        "title": "x",
                        "createdAt": "2023-03-17T09:00:00",
                        "code": "000001",
                    }
                )
                + "\n"
            )
    r = news_arrays(["000001"], DAYS, tmp_path)
    assert r["n_all"][0].tolist() == [0, 1, 0]


def test_cutoff_exact_second(tmp_path):
    """15:10:00 은 당일, 15:10:01 은 다음 거래일. (side left/right 뒤집으면 실패)"""
    _write(
        tmp_path,
        "000001",
        [("2023-03-17T15:10:00", "a"), ("2023-03-17T15:10:01", "b")],
    )
    r = news_arrays(["000001"], DAYS, tmp_path)
    assert r["n_all"][0].tolist() == [0, 1, 1]


def test_outside_window_counted_nowhere(tmp_path):
    """t=0 하한(전 달력일 15:10, 초과) 이하·마지막 15:10 초과는 어디에도 안 센다. (lo0 제거 시 실패)"""
    _write(
        tmp_path,
        "000001",
        [
            ("2023-03-15T15:09:59", "x"),  # 하한 이전
            ("2023-03-15T15:10:00", "x"),  # 하한 경계(제외)
            ("2023-03-15T15:10:01", "in"),  # t=0
            ("2023-03-20T15:10:00", "in"),  # 마지막 경계(포함)
            ("2023-03-20T15:10:01", "x"),  # 마지막 이후
            ("2023-03-25T10:00:00", "x"),
        ],
    )
    r = news_arrays(["000001"], DAYS, tmp_path)
    assert r["n_all"][0].tolist() == [1, 0, 1]
    assert int(r["n_all"].sum()) == 2


def test_known_respects_start_date_and_first_article(tmp_path):
    """2023-03-15 전은 파일에 옛 기사가 있어도 False(date_ok 제거 시 실패).
    늦게 시작한 아카이브는 첫 기사 전 False, 후 True."""
    days = pd.bdate_range("2023-03-13", "2023-03-20")  # 13,14,15,16,17,20
    _write(tmp_path, "000001", [("2023-03-01T10:00:00", "old")])
    _write(tmp_path, "000002", [("2023-03-16T10:00:00", "late")])
    r = news_arrays(["000001", "000002"], days, tmp_path)
    assert r["news_known"][0].tolist() == [False, False, True, True, True, True]
    assert r["news_known"][1].tolist() == [False, False, False, True, True, True]


def test_null_id_rows_not_collapsed(tmp_path):
    rows = [
        {"title": "a", "createdAt": "2023-03-17T09:00:00"},
        {"title": "b", "createdAt": "2023-03-17T09:01:00"},
        {"title": "b", "createdAt": "2023-03-17T09:01:00"},  # 완전 동일 → 1
    ]
    with open(tmp_path / "000001.jsonl", "w") as f:
        f.write("\n".join(json.dumps(x) for x in rows) + "\n")
    r = news_arrays(["000001"], DAYS, tmp_path)
    assert r["n_all"][0].tolist() == [0, 2, 0]


def test_tz_aware_raises(tmp_path):
    _write(tmp_path, "000001", [("2023-03-17T09:00:00+09:00", "a")])
    with pytest.raises(ValueError, match="tz-aware"):
        news_arrays(["000001"], DAYS, tmp_path)


# ---------------------------------------------------------------- 정정 0 C1: 뉴스 구간 uni 끄기·누수 검사·신선도

import numpy as np  # noqa: E402

from research.combined import news as N  # noqa: E402

WIN = pd.bdate_range("2023-01-02", periods=400)                 # 2023-03-15 앞뒤를 걸친다
W0 = int(np.searchsorted(WIN, N.NEWS_START))


def _panel(seed=0):
    """생존 8(구간 첫날부터 known, 기사 잦음) · 폐지 8(파일 없음 = known 없음) · 잘린 page_cap 8(첫 기사가 구간 중간 — 미래 기사량이 정한 시작)."""
    rng = np.random.default_rng(seed)
    n, t = 24, len(WIN)
    known = np.zeros((n, t), bool)
    known[:8, W0:] = True
    known[16:, W0 + 150:] = True
    n_all = np.where(known, rng.poisson(2.0, (n, t)), 0).astype(np.uint16)
    dl = np.r_[np.zeros(8, bool), np.ones(8, bool), np.zeros(8, bool)]
    capped = np.r_[np.zeros(16, bool), np.ones(8, bool)]
    return n_all, known, np.ones((n, t), bool), dl, capped


def _done(d, code, status, since="2023-03-16"):
    (d / f"{code}.done").write_text(json.dumps({"status": status, "rows": 9999, "since": since}))


def test_news_excluded_is_a_build_time_code_set(tmp_path):
    _write(tmp_path, "000001", [("2023-03-16T09:00:00", "a")])                  # 정상
    _done(tmp_path, "000001", "done")
    _write(tmp_path, "0007C0", [("2023-03-16T09:00:00", "e")])                  # .done 없음 — 파일 없음과 같이 뺀다(M-b)
    _write(tmp_path, "000002", [("2024-08-01T09:00:00", "b")])                  # page_cap, 잘림(첫 기사 늦음)
    _done(tmp_path, "000002", "page_cap")
    _write(tmp_path, "000003", [("2023-03-16T11:00:00", "c")])                  # page_cap 이지만 수집 시작일에 닿음
    _done(tmp_path, "000003", "page_cap")
    _write(tmp_path, "000004", [("2024-08-01T09:00:00", "d")])                  # 늦게 시작해도 page_cap 아님(상장 등) — 빼지 않는다
    _done(tmp_path, "000004", "done")
    codes = ["000001", "000002", "000003", "000004", "999999", "0007C0"]
    fr = {}
    ex, det = N.news_excluded(codes, tmp_path, fr)
    assert ex.tolist() == [False, True, False, False, True, True]
    assert det == {"nofile": ["999999"], "no_done": ["0007C0"], "page_cap_late": ["000002"], "page_cap_kept": ["000003"]}
    (tmp_path / "999999.jsonl").write_text("")                                  # 아카이브가 자라도(전진) 얼린 분류가 이긴다
    _done(tmp_path, "999999", "done")
    assert N.news_excluded(codes, tmp_path, fr)[0].tolist() == ex.tolist()
    assert N.news_excluded(codes, tmp_path)[0][4] == False  # noqa: E712 — 얼린 분류 없이 다시 재면 바뀐다(그래서 얼린다)


def test_uni_off_whole_window_not_time_dependent():
    n_all, known, uni, dl, capped = _panel()
    ex = dl | capped
    u2, n_off = N.news_uni_off(uni, ex, WIN)
    assert u2[:, :W0].all()                                                   # 뉴스 구간 전은 그대로
    assert not u2[ex, W0:].any() and u2[~ex, W0:].all()                      # 제외 종목은 구간 **전체**(첫 기사 뒤도) 끈다
    assert n_off == int((uni & ~u2).sum()) == int(ex.sum()) * (len(WIN) - W0)


def test_news_leak_check_fires_without_exclusion_and_passes_with_it():
    n_all, known, uni, dl, capped = _panel()
    with pytest.raises(ValueError, match="known delisted.*known capped"):
        N.check_news_leak(n_all, known, uni, WIN, dl, capped=capped)        # 위반 주입: 제외를 안 하면 "0 = 모름" 이 폐지·page_cap 표지
    only_dead, _ = N.news_uni_off(uni, dl, WIN)
    with pytest.raises(ValueError, match="known capped"):
        N.check_news_leak(n_all, known, only_dead, WIN, dl, capped=capped)  # page_cap 제외만 빠져도 잡힌다
    u2, _ = N.news_uni_off(uni, dl | capped, WIN)
    out = N.check_news_leak(n_all, known, u2, WIN, dl, capped=capped)
    assert out["known_share_delisted"] is None and out["known_cells_delisted"] == 0   # 빈 집단은 비교하지 않는다(기록)
    assert out["known_share_capped"] is None and out["known_share_alive"] == 1.0


def test_news_leak_check_zero60_and_late_groups():
    """늦게 시작한 종목(상장 전 uni 없음)이 시작 뒤 기사가 드물면(60일 무기사 잦음) 이른 시작과 달라 실패한다(page_cap 표시 없이도)."""
    n_all, known, uni, dl, capped = _panel()
    u2, _ = N.news_uni_off(uni, dl, WIN)
    u2[16:, :W0 + 150] = False                                                # 상장 전: uni 없음 → known 몫은 같다
    sparse = n_all.copy()
    sparse[16:] = 0
    sparse[16:, W0 + 150] = 1                                                 # 첫 기사 하나 뒤로 무기사
    with pytest.raises(ValueError, match="zero60 late"):
        N.check_news_leak(sparse, known, u2, WIN, dl)
    out = N.check_news_leak(n_all, known, u2, WIN, dl)
    assert out["known_share_late"] == out["known_share_early"] == 1.0 and out["zero60_share_late"] < 0.05
    G = N.news_groups(known, WIN, dl)
    assert G["late"][16:].all() and G["early"][:8].all() and not (G["late"] | G["early"])[8:16].any()


def test_zero60_window_counts_only_fully_known_windows():
    n = np.zeros((1, 100), np.uint16)
    n[0, 10] = 1
    k = np.zeros((1, 100), bool)
    k[0, 5:] = True
    z, full = N._zero60(n, k)
    assert not full[0, :64].any() and full[0, 64:].all()                     # 창 첫 칸(t-59)이 known 이어야
    assert not z[0, 64:70].any() and z[0, 70:].all()                          # t=10 기사는 t=69 까지 창 안


def test_archive_fresh_requires_article_at_or_after_cutoff(tmp_path):
    _write(tmp_path, "000001", [("2026-10-22T15:09:00", "a")])
    _write(tmp_path, "000002", [("2026-10-21T09:00:00", "b")])
    assert not N.archive_fresh(tmp_path, "2026-10-22")
    _write(tmp_path, "000003", [("2026-10-22T15:10:00", "c")])
    assert N.archive_fresh(tmp_path, "2026-10-22") and not N.archive_fresh(tmp_path, "2026-10-23")
    assert N.archive_max_ts(tmp_path) == pd.Timestamp("2026-10-22 15:10")


# ---------------------------------------------------------------- 정정 0-2 M-b: 수집 대상(수집기와 같은 정의)

def test_select_collect_universe_matches_collector_rule():
    rows = [(f"{i:06d}", 1000.0 - i) for i in range(400)]
    rows += [("069500", 5000.0), ("000500", 1000.0 - 500)]
    u = N.select_collect_universe(rows)
    assert len(u) == 299 and "069500" not in u and u[0] == "000000"                  # 069500 이 상위 300 안 → 299
    tie = [("000002", 7.0), ("000001", 7.0), ("000003", 9.0)]
    assert N.select_collect_universe(tie) == ["000003", "000001", "000002"]          # 동률은 코드 오름차순
    rows2 = [(f"{i:06d}", 1000.0 - i) for i in range(400)]
    assert len(N.select_collect_universe(rows2)) == 300                               # 069500 없으면 300


def test_collect_universes_use_previous_trading_day_and_positive_value():
    tv = pd.DataFrame({"date": pd.to_datetime(["2023-03-14"] * 3 + ["2023-03-15"] * 3),
                       "code": ["A", "B", "C", "A", "B", "C"], "trade_value": [3.0, 2.0, 0.0, 1.0, 5.0, 9.0]})
    old = N.COLLECT_SIZE
    try:
        N.COLLECT_SIZE = 2
        u = N.collect_universes(tv, pd.DatetimeIndex(["2023-03-15", "2023-03-16", "2023-03-18"]))
    finally:
        N.COLLECT_SIZE = old
    assert u == {"2023-03-15": ["A", "B"], "2023-03-16": ["C", "B"], "2023-03-18": ["C", "B"]}   # 그날 값은 쓰지 않는다
    with pytest.raises(ValueError, match="앞 거래일"):
        N.collect_universes(tv, pd.DatetimeIndex(["2023-03-14"]))


def test_news_known_only_on_collected_days(tmp_path):
    _write(tmp_path, "000001", [("2023-03-16T09:00:00", "[특징주] a"), ("2023-03-17T09:00:00", "[특징주] b"), ("2023-03-20T09:00:00", "c")])
    days = pd.DatetimeIndex(["2023-03-14", "2023-03-16", "2023-03-17", "2023-03-20"])
    uv = {"2023-03-14": ["000001"], "2023-03-16": ["000001"], "2023-03-17": ["000002"], "2023-03-20": ["000001"]}
    cm = N.collection_mask(["000001", "000002"], days, uv)
    assert cm.tolist() == [[False, True, False, True], [False, False, True, False]]   # 구간 전(03-14)은 대상이어도 False
    full = N.news_arrays(["000001", "000002"], days, tmp_path)
    r = N.news_arrays(["000001", "000002"], days, tmp_path, collect=cm)
    assert full["news_known"][0].tolist() == [False, True, True, True]
    assert r["news_known"][0].tolist() == [False, True, False, True]                  # 03-17 은 수집 대상 밖 → 모름
    assert r["n_feat"][0].tolist() == [0, 1, 0, 0] and full["n_feat"][0].tolist() == [0, 1, 1, 0]   # 모름 칸 개수 0
    assert not r["news_known"][1].any()                                               # 대상이어도 파일 없으면 모름
    with pytest.raises(ValueError, match="모양"):
        N.apply_collect(full, cm[:, :2])


def test_collector_complete_reads_ledger(tmp_path):
    led = tmp_path / "ledger.jsonl"
    assert not N.collector_complete(led, "2026-10-23")
    led.write_text(json.dumps({"status": "failed", "days": {"2026-10-23": {"complete": True}}}) + "\n"
                   + json.dumps({"status": "ok", "days": {"2026-10-22": {"complete": True}, "2026-10-23": {"complete": False}}}) + "\n")
    assert N.collector_complete(led, "2026-10-22") and not N.collector_complete(led, "2026-10-23")   # 실패 실행·미완은 안 센다
