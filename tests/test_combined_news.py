"""토스 뉴스 아카이브 → 15:10 경계 종목×일 배열 테스트."""

import json

import pandas as pd

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
