"""관문 0 보정·기술 지표와 관문 판정 (PREREG.md 관문 0 문턱 그대로).

순수 numpy/pandas. 입력은 NaN 없는 1차원 배열(호출자가 NaN 사건을 미리 뺀다).
`cell_verdict` 는 JSON 직렬화 가능한 파이썬 기본형만 돌려준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- 사전등록 문턱 (PREREG.md 관문 0) -------------------------------------
HIGH_EDGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
HIGH_MIN_N = 300
HIGH_TOL = 0.05
ECE_ABS = 0.02
ECE_REL = 0.2          # × 기준율
BSS_MIN = 0.01
YEARS = tuple(range(2017, 2024))
YEARS_MIN_POS = 5
GATE_MIN_PASS = 12
GATE_MIN_BLOCK = 8


def _edges(edges) -> np.ndarray:
    # linspace 의 0.6000000000000001 같은 부동소수 오차가 p=0.6 을 아래 구간에 넣지 않게 반올림
    return np.round(np.asarray(edges, dtype=float), 12)


def _check(*arrs):
    n = len(arrs[0])
    for a in arrs:
        if len(a) != n:
            raise ValueError("길이가 다르다")
        if np.issubdtype(np.asarray(a).dtype, np.floating) and np.isnan(a).any():
            raise ValueError("NaN 이 있다")


def reliability(p, y, edges=np.linspace(0, 1, 11)) -> pd.DataFrame:
    """구간 [lo, hi), 마지막은 [., 1.0] 닫힘. 빈 구간은 n=0, 평균 NaN."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    _check(p, y)
    e = _edges(edges)
    nb = len(e) - 1
    idx = np.searchsorted(e, p, side="right") - 1
    idx = np.where(p == e[-1], nb - 1, idx)
    rows = []
    for b in range(nb):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "lo": float(e[b]), "hi": float(e[b + 1]), "n": n,
            "p_mean": float(p[m].mean()) if n else float("nan"),
            "y_rate": float(y[m].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows, columns=["lo", "hi", "n", "p_mean", "y_rate"])


def ece(p, y) -> float:
    r = reliability(p, y)
    n = r["n"].to_numpy()
    if n.sum() == 0:
        return float("nan")
    gap = np.abs(r["p_mean"] - r["y_rate"]).fillna(0.0).to_numpy()
    return float((n * gap).sum() / n.sum())


def brier(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def _bss(pA, pB, y) -> float:
    bb = brier(pB, y)
    if bb == 0.0:
        return float("nan")
    return float(1.0 - brier(pA, y) / bb)


def bss_ci(pA, pB, y, cluster, n_boot=2000, seed=0):
    """BSS 점추정과 군집 복원 추출 95% 구간 (점, 2.5%, 97.5%)."""
    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)
    y = np.asarray(y, dtype=float)
    cluster = np.asarray(cluster)
    _check(pA, pB, y, cluster)
    point = _bss(pA, pB, y)
    _, inv = np.unique(cluster, return_inverse=True)
    g = int(inv.max()) + 1 if len(inv) else 0
    if g == 0:
        return float("nan"), float("nan"), float("nan")
    sa = np.bincount(inv, weights=(pA - y) ** 2, minlength=g)
    sb = np.bincount(inv, weights=(pB - y) ** 2, minlength=g)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, g, size=(n_boot, g))
    ba = sa[pick].sum(axis=1)
    bb = sb[pick].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.where(bb > 0, 1.0 - ba / bb, np.nan)
    if np.isnan(boot).all():
        return point, float("nan"), float("nan")
    lo, hi = np.nanquantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def _skill_ok(bss: float, bss_lo: float) -> bool:
    return bool(bss >= BSS_MIN and bss_lo > 0)  # NaN 이면 False


def _quarter(month) -> np.ndarray:
    month = np.asarray(month, dtype=np.int64)
    return (month // 100) * 10 + ((month % 100) - 1) // 3 + 1


def cell_verdict(pA, pB, y, month, year, years=YEARS) -> dict:
    pA = np.asarray(pA, dtype=float)
    pB = np.asarray(pB, dtype=float)
    y = np.asarray(y, dtype=float)
    month = np.asarray(month)
    year = np.asarray(year)
    _check(pA, pB, y, month, year)

    base = float(y.mean()) if len(y) else float("nan")
    e = ece(pA, y) if len(y) else float("nan")

    hb = reliability(pA, y, edges=HIGH_EDGES)
    high_bins = [
        {"lo": float(r.lo), "hi": float(r.hi), "n": int(r.n),
         "p_mean": float(r.p_mean), "y_rate": float(r.y_rate)}
        for r in hb.itertuples()
    ]
    big = [b for b in high_bins if b["n"] >= HIGH_MIN_N]
    calib_ok = bool(
        len(big) >= 1
        and all(abs(b["p_mean"] - b["y_rate"]) <= HIGH_TOL for b in big)
        and e <= ECE_ABS and e <= ECE_REL * base
    )

    bss, lo, hi = bss_ci(pA, pB, y, month)
    skill_ok = _skill_ok(bss, lo)
    _, lo_q, _ = bss_ci(pA, pB, y, _quarter(month))

    years_pos = 0
    for yr in years:
        m = year == yr
        if not m.any():
            continue
        b = _bss(pA[m], pB[m], y[m])
        if b > 0:  # NaN > 0 은 False
            years_pos += 1
    years_ok = bool(years_pos >= YEARS_MIN_POS)

    return {
        "calib_ok": calib_ok, "skill_ok": bool(skill_ok), "years_ok": years_ok,
        "pass": bool(calib_ok and skill_ok and years_ok),
        "ece": float(e), "base_rate": base,
        "bss": float(bss), "bss_lo": float(lo), "bss_hi": float(hi),
        "bss_lo_q": float(lo_q), "years_pos": int(years_pos),
        "n": int(len(y)), "high_bins": high_bins,
    }


def gate_verdict(passed, Ns=(10, 15, 20, 30, 40), ks=(3, 5, 7, 10)) -> dict:
    grid = {(n, k) for n in Ns for k in ks}
    if set(passed) != grid:
        raise ValueError("칸 키가 격자와 다르다")
    ok = {c for c in grid if bool(passed[c])}
    seen: set = set()
    largest = 0
    for start in ok:
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            n, k = stack.pop()
            size += 1
            i, j = Ns.index(n), ks.index(k)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < len(Ns) and 0 <= b < len(ks):
                    c = (Ns[a], ks[b])
                    if c in ok and c not in seen:
                        seen.add(c)
                        stack.append(c)
        largest = max(largest, size)
    return {
        "pass": bool(len(ok) >= GATE_MIN_PASS and largest >= GATE_MIN_BLOCK),
        "n_pass": int(len(ok)), "largest_block": int(largest),
    }
