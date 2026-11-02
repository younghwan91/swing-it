"""4단계 알파 검증 — 일별 순위 IC 와 폴드 IC. 합격 바는 없다(GUARDRAILS §8)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from swing_it.engine.panels import ffill_axis1
from swing_it.features.universe import RANK_BAND, RANK_BAND_ADV_FLOOR, rank_band_eligible

IC_CLIP = (0.0, 0.10)


def daily_rank_ic(close, adv, cap, sig, lo: int, hi: int, h: int, *, adv_floor: float = RANK_BAND_ADV_FLOOR,
                  cap_rank: tuple[int, int] = RANK_BAND):
    """일별 순위 IC(``rank_ic`` 정의: t 신호 대 C[t+1+h]/C[t+1]−1, Spearman, 20종목 이상). 적격 = U_t 없는 M0 의 시총 순위 띠(PO-6).

    PO-7 MINOR 2 — 창 안에서 가격이 끊긴 종목(폐지·정지)은 마지막 관측 종가로 잰다(빼면 생존자 필터). C[t+1] 이 없으면 뺀다.
    """
    nN, nD = close.shape
    cff = ffill_axis1(close)
    allu = np.ones(nN, bool)
    days, ics = [], []
    for t in range(lo, min(hi, nD - h - 2) + 1):
        el = rank_band_eligible(t, sig, adv, cap, allu, adv_floor=adv_floor, cap_rank=cap_rank)
        ret = cff[el, t + 1 + h] / close[el, t + 1] - 1.0
        ok = np.isfinite(ret)
        if ok.sum() < 20:
            continue
        a = stats.rankdata(sig[el[ok], t])
        b = stats.rankdata(ret[ok])
        if a.std() > 0 and b.std() > 0:
            ics.append(float(np.corrcoef(a, b)[0, 1]))
            days.append(t)
    return np.array(days, dtype=int), np.array(ics)


def fold_ic(days: np.ndarray, ics: np.ndarray, fold_start: int, h: int, *, clip: tuple[float, float] = IC_CLIP) -> float:
    """IC_k = 폴드 시작 (h+2) 거래일 앞까지 누적한 일별 IC 평균(라벨 겹침 퍼지), 범위를 잘라 쓴다."""
    sel = ics[days <= fold_start - (h + 2)]
    if sel.size == 0:
        return 0.0
    return float(np.clip(sel.mean(), *clip))
