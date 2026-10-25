"""토스 뉴스 아카이브(daytrade-it) → 15:10 경계 종목×일 개수 배열."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 정규식 출처: ~/git/daytrade-it/scripts/eval/autoresearch/prepare.py (원문 그대로)
FEAT_RE = re.compile(r"특징주")
EXCL_RE = re.compile(r"단독")
MOVE_RE = re.compile(
    r"급등|급락|상한가|하한가|신고가|신저가|강세|약세|껑충|폭등|폭락|들썩|치솟|곤두박질|↑|↓|%\s*(상승|하락|올라|내려)"
)

NEWS_START = pd.Timestamp(
    "2023-03-15"
)  # 이 날 전은 아카이브 커버리지 밖 → news_known False
CUTOFF = pd.Timedelta(hours=15, minutes=10)
_KEYS = ("n_all", "n_feat", "n_move", "n_excl")


def archive_dir() -> Path:
    p = Path.home() / "git/daytrade-it/data/toss_universe_42m"
    if not p.is_dir():
        raise SystemExit(f"뉴스 아카이브 없음: {p}")
    return p


def _read(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            t = a.get("title") or ""
            rows.append(
                (
                    a.get("id"),
                    a.get("createdAt"),
                    bool(FEAT_RE.search(t)),
                    bool(EXCL_RE.search(t)),
                    bool(MOVE_RE.search(t)),
                )
            )
    df = pd.DataFrame(rows, columns=["id", "ts", "feat", "excl", "move"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df.dropna(subset=["ts"]).drop_duplicates("id")  # (id, code): 파일이 종목별


def news_arrays(codes: list[str], days: pd.DatetimeIndex, archive: Path) -> dict:
    """기사는 days[t-1] 15:10 < createdAt <= days[t] 15:10 이면 t 에 센다.

    t=0 은 하한을 days[0]-1달력일 15:10 으로 둔다. 휴장일·15:10 뒤 기사는 다음 거래일로 간다.
    news_known[i,t]: 파일이 있고 첫 기사 ≤ days[t] 15:10 이며 days[t] >= NEWS_START.
    """
    N, T = len(codes), len(days)
    cut = (pd.DatetimeIndex(days).normalize() + CUTOFF).values
    lo0 = cut[0] - np.timedelta64(1, "D")
    out = {k: np.zeros((N, T), np.uint16) for k in _KEYS}
    known = np.zeros((N, T), bool)
    date_ok = pd.DatetimeIndex(days).normalize() >= NEWS_START
    for i, c in enumerate(codes):
        p = Path(archive) / f"{c}.jsonl"
        if not p.exists():
            continue
        df = _read(p)
        if df.empty:
            continue
        ts = df["ts"].values
        known[i] = (ts.min() <= cut) & date_ok
        idx = np.searchsorted(cut, ts, side="left")  # 최소 t: ts <= cut[t]
        ok = (idx < T) & (ts > lo0)
        for k, mask in zip(
            _KEYS,
            (
                np.ones(len(df), bool),
                df["feat"].values,
                df["move"].values,
                df["excl"].values,
            ),
        ):
            sel = ok & mask
            out[k][i] = np.minimum(np.bincount(idx[sel], minlength=T), 65535)
    out["news_known"] = known
    return out
