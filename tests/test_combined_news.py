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
