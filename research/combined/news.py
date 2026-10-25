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
                    t,
                    a.get("createdAt"),
                    bool(FEAT_RE.search(t)),
                    bool(EXCL_RE.search(t)),
                    bool(MOVE_RE.search(t)),
                )
            )
    df = pd.DataFrame(rows, columns=["id", "title", "ts", "feat", "excl", "move"])
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].dt.tz is not None:
        raise ValueError(f"createdAt 이 tz-aware: {path} (tz-naive 로컬시각 가정)")
    df = df.dropna(subset=["ts"])
    # (id, code) 중복 제거: 파일이 종목별이라 id 만. id 가 없는 행은 (title, ts) 로 따로 본다.
    nid = df["id"].isna()
    keep = ~(df["id"].duplicated() & ~nid) & ~(df.duplicated(["title", "ts"]) & nid)
    return df[keep]


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


# ---------------------------------------------------------------- 뉴스 구간 종목군·누수 검사(정정 0, C1)

ZERO_WIN = 60                 # 무기사 창(거래일)
NEWS_LEAK_TOL = {"known": 0.05, "zero60": 0.05}
# known 0.05: 잘리지 않은 아카이브에서도 늦게 시작한 종목(상장·드문 기사)은 첫 기사 전 known False 라 이른 시작보다 낮다(인과적,
# 2026-10-25 캐시 B 0.975 vs 0.992). 잘린 page_cap(첫 기사 전 수개월 known False — 리뷰 실측 ~0.5)·파일 없음(0)은 이 폭을 크게 넘는다.
NEWS_LEAK_MIN_CELLS = 250     # 이보다 적은 집단은 비교하지 않는다(값은 기록)
CAP_START = pd.Timestamp("2023-03-16")   # 수집 시작일(.done "since") — page_cap 종목의 첫 기사가 이 날보다 늦으면 잘린 아카이브
LATE_GRACE = pd.Timedelta(days=14)       # 첫 기사가 구간 시작 + 14일 뒤면 "늦은 시작" 집단(검사용 묶음일 뿐, uni 와 무관)


def news_excluded(codes: "list[str]", archive: Path) -> "tuple[np.ndarray, dict]":
    """뉴스 구간에서 통째로 뺄 종목(bool[N]) — 빌드 때 정해지는 상수 집합이다(시점에 따라 바뀌지 않는다).

    (i) 아카이브 파일이 없는 종목(2026-09 stocks 목록 밖 = 그 사이 폐지 등), (ii) page_cap 종목(``<code>.done`` status) 중 첫 기사 날짜가
    ``CAP_START`` 보다 늦은 것 — Toss API 가 종목당 최근 1만 건만 주어 거꾸로 잘렸고(재수집 불가), 그 시작일은 미래 기사량이 정한다.
    "첫 기사 전만 끄기"는 끄는 기간이 미래 기사량에 달려 여전히 누수라, 두 집합 모두 구간 **전체**를 끈다(정정 0 C1, 생존편향 공개).
    """
    excl = np.zeros(len(codes), bool)
    det = {"nofile": [], "page_cap_late": [], "page_cap_kept": []}
    for i, c in enumerate(codes):
        p = Path(archive) / f"{c}.jsonl"
        if not p.exists():
            excl[i] = True
            det["nofile"].append(c)
            continue
        d = Path(archive) / f"{c}.done"
        if d.exists() and json.loads(d.read_text(encoding="utf-8")).get("status") == "page_cap":
            df = _read(p)
            first = df["ts"].min() if len(df) else None
            if first is None or first.normalize() > CAP_START:
                excl[i] = True
                det["page_cap_late"].append(c)
            else:
                det["page_cap_kept"].append(c)
    return excl, det


def news_uni_off(uni: np.ndarray, excluded: np.ndarray, days: pd.DatetimeIndex) -> "tuple[np.ndarray, int]":
    """뉴스 구간(days ≥ NEWS_START) 전체에서 ``excluded`` 종목의 uni 를 끈다 → (새 uni, 끈 칸 수). 구간 전은 그대로."""
    win = (pd.DatetimeIndex(days).normalize() >= NEWS_START)[None, :]
    off = np.asarray(uni, bool) & win & np.asarray(excluded, bool)[:, None]
    return np.asarray(uni, bool) & ~off, int(off.sum())


def _zero60(n_all: np.ndarray, news_known: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(t 까지 60거래일 기사 합 == 0, 그 창 전체가 커버리지 안인 칸). news_known 은 종목마다 한 번 켜지면 유지된다."""
    n = np.asarray(n_all, np.int64)
    cs = np.cumsum(n, axis=1)
    s = cs.copy()
    s[:, ZERO_WIN:] = cs[:, ZERO_WIN:] - cs[:, :-ZERO_WIN]
    full = np.zeros(n.shape, bool)
    full[:, ZERO_WIN - 1:] = np.asarray(news_known, bool)[:, :n.shape[1] - ZERO_WIN + 1]
    return s == 0, full


def news_groups(news_known: np.ndarray, days: pd.DatetimeIndex, delisted: np.ndarray, capped: "np.ndarray | None" = None) -> dict:
    """비교 집단(bool[N]): 폐지·생존 / 늦은 시작(첫 기사가 구간 시작 + LATE_GRACE 뒤)·이른 시작 / (주면) page_cap 전부·그 밖
    (제외 뒤에는 page_cap 집단에 남은 칸 = 잘리지 않은 page_cap 뿐 — 제외가 빠지면 잘린 종목의 known False 가 이 쌍에서 드러난다)."""
    d = np.asarray(delisted, bool)
    days = pd.DatetimeIndex(days).normalize()
    k = np.asarray(news_known, bool)
    win = np.asarray(days >= NEWS_START)
    has = (k & win[None, :]).any(1)
    early_ix = np.asarray(days <= NEWS_START + LATE_GRACE) & win
    early = (k & early_ix[None, :]).any(1)
    G = {"delisted": d, "alive": ~d, "late": has & ~early, "early": early}
    if capped is not None:
        G.update(capped=np.asarray(capped, bool), uncapped=~np.asarray(capped, bool))
    return G


def check_news_leak(n_all: np.ndarray, news_known: np.ndarray, uni: np.ndarray, days: pd.DatetimeIndex, delisted: np.ndarray,
                    capped: "np.ndarray | None" = None, tol: "dict | None" = None, min_cells: int = NEWS_LEAK_MIN_CELLS) -> dict:
    """뉴스 구간 uni 칸에서 폐지/생존, 늦은/이른 시작, (주면) 남은 page_cap/그 밖 집단의 news_known 몫·60일 무기사 몫 차이가
    허용치(``NEWS_LEAK_TOL``) 안이어야 한다(아니면 ValueError).

    재는 것은 "개수 0 또는 모름"이 미래 폐지·미래 기사량(page_cap 역방향 수집)을 알리는지다. 60일 무기사는 창 전체가 커버리지 안인 칸만 센다.
    집단 칸이 ``min_cells`` 보다 적으면 그 쌍은 비교하지 않고 None 으로 남긴다.
    """
    tol = {**NEWS_LEAK_TOL, **(tol or {})}
    win = (pd.DatetimeIndex(days).normalize() >= NEWS_START)[None, :]
    u = np.asarray(uni, bool) & win
    known = np.asarray(news_known, bool)
    z, full = _zero60(n_all, known)
    G = news_groups(known, days, delisted, capped)
    pairs = [("delisted", "alive"), ("late", "early")] + ([("capped", "uncapped")] if capped is not None else [])
    out, bad = {}, []
    for a, b in pairs:
        for m, cell, val in (("known", u, known), ("zero60", u & full, z)):
            sh = {}
            for g in (a, b):
                c = cell & G[g][:, None]
                out[f"{m}_cells_{g}"] = int(c.sum())
                sh[g] = float(val[c].mean()) if c.sum() >= min_cells else None
                out[f"{m}_share_{g}"] = sh[g]
            if sh[a] is not None and sh[b] is not None and abs(sh[a] - sh[b]) >= tol[m]:
                bad.append(f"{m} {a} {sh[a]:.4f} vs {b} {sh[b]:.4f}(허용 {tol[m]})")
    if bad:
        raise ValueError("뉴스 누수 검사 실패 — " + "; ".join(bad))
    return out


def archive_max_ts(archive: Path) -> "pd.Timestamp | None":
    """아카이브 전체의 최대 createdAt(tz-naive KST). 전진 기록 전 신선도 확인용."""
    best = None
    for p in Path(archive).glob("*.jsonl"):
        df = _read(p)
        if len(df):
            m = df["ts"].max()
            best = m if best is None or m > best else best
    return best


def archive_fresh(archive: Path, day) -> bool:
    """전진 기록 조건(PREREG 정정 0, M6b): 아카이브 최대 createdAt ≥ day 15:10 이어야 그날 신호를 기록한다."""
    m = archive_max_ts(archive)
    return m is not None and m >= pd.Timestamp(day).normalize() + CUTOFF
