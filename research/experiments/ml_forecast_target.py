"""ML 수익률 예측 — 타깃 y_{i,t} = Σ_{s=t+2}^{t+21} u_{i,s} (PREREG §4).

- u 는 core ``factor_returns(R, X_style, WICS, cap, mask_y)`` 의 특이수익(:func:`ml_forecast_data.label_mask`).
- 거래량_s = 0 인 날은 u_s = NaN → 합에서 0. 재개일(거래량 > 0)은 남는다.
- 거래량_{t+1} = 0(또는 행 없음)이면 진입 불가 — IC·학습에서 뺀다(:func:`tradable`).
- 창 안 가격 행(정지일 포함)이 15일 미만이면 NaN. 단 창 안에서 폐지된 종목은 관측된 날까지 합하고 나머지 0(현금)으로 남긴다.
"""

from __future__ import annotations

import numpy as np
from ml_forecast_common import HOLD, LABEL_LAST, MIN_PRICE_DAYS

REPORT_KEYS = ("rows", "rows_finite", "halt_days", "rows_with_halt", "rows_nan_15", "rows_delist_in_window",
               "rows_delist_kept_by_exception", "rows_entry_excluded", "rows_seam")


def tradable(vol: np.ndarray, t: int) -> np.ndarray:
    """t+1 종가 체결 가능(거래량_{t+1} > 0). 행이 없으면(NaN) 불가."""
    with np.errstate(invalid="ignore"):
        return np.asarray(vol[:, t + 1] > 0)


def label_at(u_full: np.ndarray, vol: np.ndarray, close: np.ndarray, delist_last: np.ndarray, t: int):
    """한 날짜의 y(N,)와 행별 보조 배열(정지 여부·가격일·창 안 폐지)."""
    lo, hi = t + 2, t + LABEL_LAST + 1
    assert hi - lo == HOLD
    assert hi <= u_full.shape[1], "라벨 창이 패널 끝을 넘는다"
    u = u_full[:, lo:hi].copy()
    with np.errstate(invalid="ignore"):
        halt = vol[:, lo:hi] == 0
    u[halt] = np.nan
    s = np.where(np.isfinite(u), u, 0.0).sum(axis=1)
    price_days = np.isfinite(close[:, lo:hi]).sum(axis=1)
    dl = (delist_last >= t + 1) & (delist_last < t + LABEL_LAST)
    ok = ((price_days >= MIN_PRICE_DAYS) | dl) & ((price_days > 0) | dl)
    y = np.where(ok, s, np.nan)
    return y, halt, price_days, dl


def labels(u_full, vol, close, delist_last, U: dict[int, np.ndarray], seam=None) -> tuple[dict[int, np.ndarray], dict]:
    """날짜 t → (N,) y. 보고 개수(PREREG §4)는 U(t) 행에서 센다."""
    y = {}
    rep = dict.fromkeys(REPORT_KEYS, 0)
    for t, idx in U.items():
        yt, halt, pdays, dl = label_at(u_full, vol, close, delist_last, t)
        y[t] = yt
        tr = tradable(vol, t)[idx]
        fin = np.isfinite(yt[idx])
        rep["rows"] += len(idx)
        rep["rows_finite"] += int((fin & tr).sum())
        rep["halt_days"] += int(halt[idx][tr].sum())
        rep["rows_with_halt"] += int((halt[idx].any(axis=1) & fin & tr).sum())
        rep["rows_nan_15"] += int((~fin & tr).sum())
        rep["rows_delist_in_window"] += int((dl[idx] & tr).sum())
        rep["rows_delist_kept_by_exception"] += int((dl[idx] & (pdays[idx] < MIN_PRICE_DAYS) & tr).sum())
        rep["rows_entry_excluded"] += int((~tr).sum())
        if seam is not None:
            lo, hi = t + 2, t + LABEL_LAST + 1
            rep["rows_seam"] += int((seam[idx, lo:hi].any(axis=1) & fin & tr).sum())
    return y, rep
