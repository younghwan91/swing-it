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


def news_arrays(codes: list[str], days: pd.DatetimeIndex, archive: Path, collect: "np.ndarray | None" = None) -> dict:
    """기사는 days[t-1] 15:10 < createdAt <= days[t] 15:10 이면 t 에 센다.

    t=0 은 하한을 days[0]-1달력일 15:10 으로 둔다. 휴장일·15:10 뒤 기사는 다음 거래일로 간다.
    news_known[i,t]: 파일이 있고 첫 기사 ≤ days[t] 15:10 이며 days[t] >= NEWS_START.
    ``collect``(bool[N,T], 정정 0-2 M-b): 그 (종목, 날)이 증분 수집기의 그날 수집 대상(``collection_mask``)일 때만 known — 아니면 known False·
    개수 0(모름). 백필 아카이브는 전 종목을 모았지만 전진 수집기는 상위 300 만 모으므로 백테스트도 같은 시점 규칙으로 가린다.
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
    return out if collect is None else apply_collect(out, collect)


def apply_collect(news: dict, collect: np.ndarray) -> dict:
    """수집 대상 가림(새 dict): 대상 밖 (종목, 날)은 known False·개수 0."""
    cm = np.asarray(collect, bool)
    if cm.shape != news["news_known"].shape:
        raise ValueError(f"collect 모양 {cm.shape} ≠ {news['news_known'].shape}")
    out = {k: np.where(cm, news[k], 0).astype(np.uint16) for k in _KEYS}
    out["news_known"] = news["news_known"] & cm
    return out


# ---------------------------------------------------------------- 수집 대상(증분 수집기와 같은 정의, 정정 0-2 M-b)

COLLECT_SIZE = 300            # daytrade-it application/news/universe.UNIVERSE_SIZE
COLLECT_DROP = "069500"       # toss_incremental.MARKET — select_universe 가 항상 붙인 뒤 수집기가 뺀다


def select_collect_universe(day_rows: "list[tuple[str, float]]") -> "list[str]":
    """수집기 ``select_universe`` + ``c != MARKET`` 그대로: 거래대금 내림차순(동률 코드 오름차순) 상위 300, 그중 069500 은 뺀다
    (069500 이 상위 300 안이면 299 종목)."""
    ranked = sorted(day_rows, key=lambda r: (-r[1], r[0]))
    return [c for c, _ in ranked[:COLLECT_SIZE] if c != COLLECT_DROP]


def collect_universes(tv: pd.DataFrame, days) -> "dict[str, list[str]]":
    """{날(YYYY-MM-DD): 그날 수집 대상} — 수집기 ``toss_incremental.universes`` 와 같은 정의.

    ``tv`` = ``daily_bars_adjusted`` 의 (date, code, trade_value) 긴 표(원값, session="raw" — 수집기는 테이블 값을 그대로 읽는다;
    단위는 백만원이지만 순위만 쓰므로 무관). trade_value > 0 행만, 그날 **직전 거래일**(표에 행이 있는 날 중 d 보다 앞선 마지막 날)의 순위.
    """
    t = tv[tv["trade_value"] > 0]
    dates = pd.to_datetime(t["date"]).dt.normalize()
    by_day = {d: list(zip(g["code"].astype(str), g["trade_value"].astype(float), strict=True))
              for d, g in t.assign(date=dates).groupby("date")}
    traded = np.array(sorted(by_day), dtype="datetime64[ns]")
    out = {}
    for d in pd.DatetimeIndex(days).normalize():
        k = int(np.searchsorted(traded, d.to_datetime64(), side="left"))
        if k == 0:
            raise ValueError(f"{d.date()} 앞 거래일 자료가 없다 — 수집 대상을 만들 수 없다")
        out[str(d.date())] = select_collect_universe(by_day[pd.Timestamp(traded[k - 1])])
    return out


def collection_mask(codes: "list[str]", days: pd.DatetimeIndex, universes: dict) -> np.ndarray:
    """bool[N,T]: 뉴스 구간(≥ NEWS_START)의 그날 수집 대상이면 True. 구간 전과 표에 없는 날은 False."""
    pos = {c: i for i, c in enumerate(codes)}
    m = np.zeros((len(codes), len(days)), bool)
    for t, d in enumerate(pd.DatetimeIndex(days).normalize()):
        if d < NEWS_START:
            continue
        for c in universes.get(str(d.date()), ()):
            i = pos.get(c)
            if i is not None:
                m[i, t] = True
    return m


# ---------------------------------------------------------------- 뉴스 구간 종목군·누수 검사(정정 0, C1)

ZERO_WIN = 60                 # 무기사 창(거래일)
NEWS_LEAK_TOL = {"known": 0.05, "zero60": 0.05}
# known 0.05: 잘리지 않은 아카이브에서도 늦게 시작한 종목(상장·드문 기사)은 첫 기사 전 known False 라 이른 시작보다 낮다(인과적,
# 2026-10-25 캐시 B 0.975 vs 0.992). 잘린 page_cap(첫 기사 전 수개월 known False — 리뷰 실측 ~0.5)·파일 없음(0)은 이 폭을 크게 넘는다.
NEWS_LEAK_MIN_CELLS = 250     # 이보다 적은 집단은 비교하지 않는다(값은 기록)
EXCLUDED_STATUS = ("nofile", "no_done", "page_cap_late")
CAP_START = pd.Timestamp("2023-03-16")   # 수집 시작일(.done "since") — page_cap 종목의 첫 기사가 이 날보다 늦으면 잘린 아카이브
LATE_GRACE = pd.Timedelta(days=14)       # 첫 기사가 구간 시작 + 14일 뒤면 "늦은 시작" 집단(검사용 묶음일 뿐, uni 와 무관)


def news_excluded(codes: "list[str]", archive: Path, frozen: "dict | None" = None) -> "tuple[np.ndarray, dict]":
    """뉴스 구간에서 통째로 뺄 종목(bool[N]) — 빌드 때 정해지는 상수 집합이다(시점에 따라 바뀌지 않는다).

    (i) 아카이브 파일이 없는 종목(2026-09 stocks 목록 밖 = 그 사이 폐지 등), (ii) page_cap 종목(``<code>.done`` status) 중 첫 기사 날짜가
    ``CAP_START`` 보다 늦은 것 — Toss API 가 종목당 최근 1만 건만 주어 거꾸로 잘렸고(재수집 불가), 그 시작일은 미래 기사량이 정한다.
    "첫 기사 전만 끄기"는 끄는 기간이 미래 기사량에 달려 여전히 누수라, 두 집합 모두 구간 **전체**를 끈다(정정 0 C1, 생존편향 공개).
    (iii) 파일은 있지만 ``.done`` 이 없는 종목(백필 완료 기록 없음 — 정정 0-2 M-b, 파일 없음과 같이 처리).
    ``frozen`` = {"status": {코드: 분류}} — 있으면 그 코드는 얼린 분류를 쓰고(전진 빌더가 아카이브가 자라도 같은 집합을 쓴다), 없는 코드만 계산해 더한다.
    분류: ok · page_cap_kept · nofile · no_done · page_cap_late(앞 둘 말고는 제외).
    """
    st = frozen.setdefault("status", {}) if frozen is not None else {}
    for c in codes:
        if c in st:
            continue
        p, d = Path(archive) / f"{c}.jsonl", Path(archive) / f"{c}.done"
        if not p.exists():
            st[c] = "nofile"
        elif not d.exists():
            st[c] = "no_done"
        elif json.loads(d.read_text(encoding="utf-8")).get("status") == "page_cap":
            df = _read(p)
            first = df["ts"].min() if len(df) else None
            st[c] = "page_cap_late" if first is None or first.normalize() > CAP_START else "page_cap_kept"
        else:
            st[c] = "ok"
    det = {k: [c for c in codes if st[c] == k] for k in ("nofile", "no_done", "page_cap_late", "page_cap_kept")}
    excl = np.array([st[c] in EXCLUDED_STATUS for c in codes], bool)
    return excl, det


def news_uni_off(uni: np.ndarray, excluded: np.ndarray, days: pd.DatetimeIndex) -> "tuple[np.ndarray, int]":
    """뉴스 구간(days ≥ NEWS_START) 전체에서 ``excluded`` 종목의 uni 를 끈다 → (새 uni, 끈 칸 수). 구간 전은 그대로."""
    win = (pd.DatetimeIndex(days).normalize() >= NEWS_START)[None, :]
    off = np.asarray(uni, bool) & win & np.asarray(excluded, bool)[:, None]
    return np.asarray(uni, bool) & ~off, int(off.sum())


def _rsum(a: np.ndarray) -> np.ndarray:
    cs = np.cumsum(np.asarray(a, np.int64), axis=1)
    s = cs.copy()
    s[:, ZERO_WIN:] = cs[:, ZERO_WIN:] - cs[:, :-ZERO_WIN]
    return s


def _zero60(n_all: np.ndarray, news_known: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(t 까지 60거래일 기사 합 == 0, 그 60일이 **모두** known 인 칸). 수집 대상 가림으로 known 이 켜졌다 꺼졌다 하므로 창 전체를 센다."""
    k = np.asarray(news_known, bool)
    full = _rsum(k) == ZERO_WIN
    full[:, :ZERO_WIN - 1] = False
    return _rsum(n_all) == 0, full


def news_groups(news_known: np.ndarray, days: pd.DatetimeIndex, delisted: np.ndarray, capped: "np.ndarray | None" = None) -> dict:
    """비교 집단(bool[N]): 폐지·생존 / 늦은 시작(첫 기사가 구간 시작 + LATE_GRACE 뒤)·이른 시작 / (주면) page_cap 전부·그 밖
    (제외 뒤에는 page_cap 집단에 남은 칸 = 잘리지 않은 page_cap 뿐 — 제외가 빠지면 잘린 종목의 known False 가 이 쌍에서 드러난다).
    ``news_known`` 은 수집 대상 가림 **전** 아카이브 known 을 준다 — 시작 시점 집단은 아카이브의 성질이다."""
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
                    capped: "np.ndarray | None" = None, tol: "dict | None" = None, min_cells: int = NEWS_LEAK_MIN_CELLS,
                    collect: "np.ndarray | None" = None, archive_known: "np.ndarray | None" = None) -> dict:
    """뉴스 구간 uni 칸에서 폐지/생존, 늦은/이른 시작, (주면) 남은 page_cap/그 밖 집단의 news_known 몫·60일 무기사 몫 차이가
    허용치(``NEWS_LEAK_TOL``) 안이어야 한다(아니면 ValueError).

    재는 것은 "개수 0 또는 모름"이 미래 폐지·미래 기사량(page_cap 역방향 수집)을 알리는지다. 60일 무기사는 창 전체가 커버리지 안인 칸만 센다.
    집단 칸이 ``min_cells`` 보다 적으면 그 쌍은 비교하지 않고 None 으로 남긴다.
    ``collect`` 가 있으면 그날 수집 대상 칸만 본다(수집 대상 여부는 전일 거래대금 — V 의 tv 로 이미 보이는 인과적 정보다).
    ``archive_known`` = 가림 전 아카이브 known(늦은/이른 시작 집단을 아카이브 시작일로 나눈다). 없으면 news_known.
    """
    tol = {**NEWS_LEAK_TOL, **(tol or {})}
    win = (pd.DatetimeIndex(days).normalize() >= NEWS_START)[None, :]
    u = np.asarray(uni, bool) & win
    if collect is not None:
        u = u & np.asarray(collect, bool)
    known = np.asarray(news_known, bool)
    z, full = _zero60(n_all, known)
    G = news_groups(known if archive_known is None else archive_known, days, delisted, capped)
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


def collector_complete(ledger: Path, day) -> bool:
    """전진 기록 조건(정정 0-2 M-b): 증분 수집기 원장(``toss_universe_incr/ledger.jsonl``)에 그날이 complete 로 적힌 성공 실행이 있어야 한다
    (그날 수집 대상 전부를 마감 뒤 실행에서 모았다 — 수집기 ``last_complete_day`` 와 같은 읽기)."""
    p, d = Path(ledger), str(pd.Timestamp(day).date())
    if not p.exists():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok" and (r.get("days") or {}).get(d, {}).get("complete"):
                return True
    return False
