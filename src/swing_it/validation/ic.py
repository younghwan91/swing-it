"""4단계 알파 검증 — 일별 IC·분위 초과·요약·IC 널·폴드 IC. 합격 바는 없다(GUARDRAILS §8).

**순위 IC 는 평균 정보량을 과대 표시한다(ML S2: 순위 .041 대 평균 .014).** 순위는 꼬리를 지우는데, 롱온리는 평균만 수확한다. 그래서 순위 IC 만
보고하지 않는다. :func:`daily_mean_ic` 와 :func:`quintile_excess` 의 Q5−U, 그리고 2배 왕복 비용을 함께 보고한다.

**폐지의 한계.** 창 안에서 가격이 끊긴 종목은 마지막 관측 종가로 잰다. 폐지 뒤 수익을 0% 로 치는 것이고 정리매매·상장폐지 손실은 들어가지 않는다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from swing_it.diagnostics.hac import nw_t_rows
from swing_it.engine.panels import ffill_axis1
from swing_it.features.universe import RANK_BAND, RANK_BAND_ADV_FLOOR, rank_band_eligible
from swing_it.validation.keys import keyed_rng

IC_CLIP = (0.0, 0.10)
MIN_NAMES = 20
WINSOR = (0.01, 0.99)  # 날짜별 횡단면 분위
N_IC_PERM = 300


def _check_window(lo: "int | None", h: int) -> None:
    if lo is not None and lo < 0:
        raise ValueError(f"lo 는 0 이상이어야 한다 — 음수 번호는 패널 끝을 가리킨다: {lo}")
    if h < 1:
        raise ValueError(f"h 는 1 이상이어야 한다: {h}")


def _labelled_days(close, adv, cap, sig, lo, hi, h, adv_floor, cap_rank):
    """(t, 종목 번호, 라벨 수익) — t 신호 대 C[t+1+h]/C[t+1]−1. 적격 = U_t 없는 M0 의 시총 순위 띠(PO-6), 유한 라벨 20종목 이상인 날만."""
    _check_window(lo, h)
    nN, nD = close.shape
    cff = ffill_axis1(close)
    allu = np.ones(nN, bool)
    for t in range(lo, min(hi, nD - h - 2) + 1):
        el = rank_band_eligible(t, sig, adv, cap, allu, adv_floor=adv_floor, cap_rank=cap_rank)
        ret = cff[el, t + 1 + h] / close[el, t + 1] - 1.0
        ok = np.isfinite(ret)
        if ok.sum() >= MIN_NAMES:
            yield t, el[ok], ret[ok]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def _winsor(x: np.ndarray) -> np.ndarray:
    return np.clip(x, *np.quantile(x, WINSOR))


def daily_rank_ic(close, adv, cap, sig, lo: int, hi: int, h: int, *, adv_floor: float = RANK_BAND_ADV_FLOOR,
                  cap_rank: tuple[int, int] = RANK_BAND):
    """일별 순위 IC(``rank_ic`` 정의: t 신호 대 C[t+1+h]/C[t+1]−1, Spearman, 20종목 이상). 적격 = U_t 없는 M0 의 시총 순위 띠(PO-6).

    PO-7 MINOR 2 — 창 안에서 가격이 끊긴 종목(폐지·정지)은 마지막 관측 종가로 잰다(빼면 생존자 필터). C[t+1] 이 없으면 뺀다.
    """
    days, ics = [], []
    for t, el, ret in _labelled_days(close, adv, cap, sig, lo, hi, h, adv_floor, cap_rank):
        ic = _corr(stats.rankdata(sig[el, t]), stats.rankdata(ret))
        if ic == ic:
            ics.append(ic)
            days.append(t)
    return np.array(days, dtype=int), np.array(ics)


def daily_mean_ic(close, adv, cap, sig, lo: int, hi: int, h: int, *, adv_floor: float = RANK_BAND_ADV_FLOOR,
                  cap_rank: tuple[int, int] = RANK_BAND) -> dict:
    """일별 평균 IC — rank(신호) 대 **원수익**의 Pearson, 그리고 수익을 날마다 1·99 분위로 윈저한 값. 표본은 :func:`daily_rank_ic` 와 같다."""
    days, raw, wins = [], [], []
    for t, el, ret in _labelled_days(close, adv, cap, sig, lo, hi, h, adv_floor, cap_rank):
        a = stats.rankdata(sig[el, t])
        ic = _corr(a, ret)
        if ic == ic:
            days.append(t)
            raw.append(ic)
            wins.append(_corr(a, _winsor(ret)))
    return {"days": np.array(days, dtype=int), "ic": np.array(raw), "ic_winsor": np.array(wins)}


def quintile_excess(close, adv, cap, sig, lo: int, hi: int, h: int, *, adv_floor: float = RANK_BAND_ADV_FLOOR,
                    cap_rank: tuple[int, int] = RANK_BAND) -> dict:
    """일별 Q5−U·Q1−U (U = 그날 적격 전체). 평균·윈저 평균·중앙값 차. 신호 상·하위 20%, 비용 전."""
    keys = ("q5_mean", "q1_mean", "q5_winsor", "q1_winsor", "q5_median", "q1_median")
    out: dict = {k: [] for k in keys}
    days = []
    for t, el, ret in _labelled_days(close, adv, cap, sig, lo, hi, h, adv_floor, cap_rank):
        order = np.argsort(sig[el, t], kind="stable")
        k = ret.size // 5
        lo_i, hi_i = order[:k], order[-k:]
        w = _winsor(ret)
        days.append(t)
        for tag, idx in (("q5", hi_i), ("q1", lo_i)):
            out[f"{tag}_mean"].append(float(ret[idx].mean() - ret.mean()))
            out[f"{tag}_winsor"].append(float(w[idx].mean() - w.mean()))
            out[f"{tag}_median"].append(float(np.median(ret[idx]) - np.median(ret)))
    return {"days": np.array(days, dtype=int), **{k: np.array(v) for k, v in out.items()}}


def ic_summary(days: np.ndarray, ics: np.ndarray, h: int) -> dict:
    """평균, Newey-West t(lag = h, 겹치는 라벨), 비겹침 ICIR(앞에서부터 h 거래일 이상 떨어진 날만, 평균/표준편차, 연환산 안 함)."""
    _check_window(None, h)
    days, ics = np.asarray(days), np.asarray(ics, dtype=np.float64)
    if days.size != ics.size:
        raise ValueError(f"days({days.size})·ics({ics.size}) 길이가 다르다")
    if ics.size == 0:
        return {"n_days": 0, "mean": float("nan"), "nw_t": float("nan"), "icir_nonoverlap": float("nan"), "n_nonoverlap": 0}
    keep, last = [], None
    for i, d in enumerate(days):
        if last is None or d - last >= h:
            keep.append(i)
            last = d
    sel = ics[keep]
    icir = float(sel.mean() / sel.std(ddof=1)) if sel.size > 1 and sel.std(ddof=1) > 0 else float("nan")
    nw = float(nw_t_rows(ics[None, :], lag=h)[0]) if ics.size > h + 1 else float("nan")
    return {"n_days": int(ics.size), "mean": float(ics.mean()), "nw_t": nw, "icir_nonoverlap": icir, "n_nonoverlap": int(sel.size)}


def ic_permutation_null(close, adv, cap, sig, lo: int, hi: int, h: int, *, n: int = N_IC_PERM, adv_floor: float = RANK_BAND_ADV_FLOOR,
                        cap_rank: tuple[int, int] = RANK_BAND, seed_key: str = "ic-perm") -> dict:
    """신호를 종목끼리 섞은 일평균 IC 분포(``rank``·``mean``, 길이 n). 적격은 진짜 신호로 정한다. 시드 ``keyed_rng(seed_key, r)``.

    π 는 신호의 베타·변동성·사이즈 상관을 지운다. 기울기가 있는 신호는 이 널을 그냥 넘는다(:mod:`swing_it.book.nulls`) — 팩터 회귀와 함께 쓴다.
    """
    labelled = list(_labelled_days(close, adv, cap, sig, lo, hi, h, adv_floor, cap_rank))
    out = {"rank": np.full(n, np.nan), "mean": np.full(n, np.nan)}
    for r in range(n):
        pi = keyed_rng(seed_key, r).permutation(sig.shape[0])
        rank, mean = [], []
        for t, el, ret in labelled:
            vals = sig[pi[el], t]
            ok = np.isfinite(vals)
            if ok.sum() < MIN_NAMES:
                continue
            a = stats.rankdata(vals[ok])
            rank.append(_corr(a, stats.rankdata(ret[ok])))
            mean.append(_corr(a, ret[ok]))
        if rank:
            out["rank"][r], out["mean"][r] = np.nanmean(rank), np.nanmean(mean)
    return out


def fold_ic(days: np.ndarray, ics: np.ndarray, fold_start: int, h: int, *, clip: tuple[float, float] = IC_CLIP,
            detail: bool = False):
    """IC_k = 폴드 시작 (h+2) 거래일 앞까지 누적한 일별 IC 평균(라벨 겹침 퍼지), 범위를 잘라 쓴다.

    ⚠️ Grinold α(``alpha_vec``)의 입력으로 쓰면 α 크기를 그대로 정한다. 순위 IC 를 넣으면 평균 정보량보다 큰 α 를 믿게 된다(모듈 설명).
    ``detail`` 이면 ``{"ic": 잘린 값, "raw": 잘리기 전 평균, "n": 표본 수}`` 를 돌려준다. 잘려서 0 이 된 것과 표본이 없는 것은 다르다.
    """
    _check_window(None, h)
    sel = ics[days <= fold_start - (h + 2)]
    ic = float(np.clip(sel.mean(), *clip)) if sel.size else 0.0
    if detail:
        return {"ic": ic, "raw": float(sel.mean()) if sel.size else float("nan"), "n": int(sel.size)}
    return ic
