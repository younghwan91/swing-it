"""``daily_bars_adjusted`` 이음매 가드 — PEAD 최적화 정정 PO-2.1 (numpy·pandas 만).

수정 종가 비 ρ_adj 와 원 종가 비 ρ_raw 가 20% 넘게 어긋나는데 그 사이에 ``corporate_actions`` 사건이 없으면
수정주가 표의 이음매로 보고 그 수익 셀을 버린다(NaN). 예: 025620 2021-07-29 수정 1780→29179, 원 1780→1810.

정정 PO-3.3: 두 번째 원천(``supply_demand``)은 거래정지 중 종가가 멈춰 있어 재개일 실제 움직임을 이음매로 오판한다
(221610 −13.6% 등). ``secondary_min_abs_ret`` 를 주면 두 번째 이후 원천 근거의 셀은 |수정 수익| 이 그 값을 넘을 때만
이음매로 보고, 나머지는 판정 불가로 남긴다. 기본 ``None`` 은 PO-2 동작이다.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

THRESHOLD = 0.2


def seam_guard(
    codes: np.ndarray,
    dates: pd.DatetimeIndex,
    ret: np.ndarray,
    close: np.ndarray,
    raw_sources: list[np.ndarray],
    events: pd.DataFrame,
    thr: float = THRESHOLD,
    secondary_min_abs_ret: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(ret 고친 것, close 고친 것, 이음매 마스크, 보고) 를 낸다.

    - ``ret[:, d]`` = ``close[:, d] / close[:, d-1] − 1``(로더 정의). ``raw_sources`` 는 우선순위 순 원 종가 패널들이다 —
      두 날 모두 값이 있는 첫 원천 하나로 ρ_raw 를 낸다(원천을 섞지 않는다). 없으면 판정 불가로 두고 셀을 남긴다.
    - ``events`` 는 (code, date) — 종류 불문. 열 d⁻ < date ≤ d 인 사건이 있으면 셀 (i, d) 는 설명된 것으로 남긴다.
    - close 는 이음매 날의 수정 비를 걷어내 이어 붙인다(그 날 수익 0 과 같다) — 수준을 쓰는 모멘텀용.
    """
    ret = np.asarray(ret, dtype=np.float64)
    N, T = ret.shape
    rho_adj = 1.0 + ret
    rho_raw = np.full((N, T), np.nan)
    src_id = np.full((N, T), -1, dtype=np.int8)
    for k, src in enumerate(raw_sources):
        src = np.asarray(src, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            pair = src[:, 1:] / src[:, :-1]
        ok = np.isfinite(pair) & (pair > 0) & (src[:, 1:] > 0) & (src[:, :-1] > 0)
        fill = np.isnan(rho_raw[:, 1:]) & ok
        sub = rho_raw[:, 1:]
        sub[fill] = pair[fill]
        src_id[:, 1:][fill] = k
    explained = np.zeros((N, T), dtype=bool)
    if len(events):
        cidx = pd.Index([str(c) for c in codes])
        ii = cidx.get_indexer(events["code"].astype(str))
        kk = dates.searchsorted(pd.to_datetime(events["date"]).to_numpy(), side="left")
        keep = (ii >= 0) & (kk >= 1) & (kk < T)
        explained[ii[keep], kk[keep]] = True
    fin = np.isfinite(rho_adj) & (rho_adj > 0)
    det = fin & np.isfinite(rho_raw)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.abs(rho_adj / rho_raw - 1.0)
    big = det & (dev > thr)
    cand = big & ~explained
    if secondary_min_abs_ret is None:
        seam = cand
        kept_secondary = np.zeros_like(cand)
    else:
        seam = cand & ((src_id == 0) | (np.abs(ret) > secondary_min_abs_ret))
        kept_secondary = cand & ~seam
    ret_f = ret.copy()
    ret_f[seam] = np.nan
    jump = np.where(seam, np.log(np.where(seam, rho_adj, 1.0)), 0.0)
    close_f = np.asarray(close, dtype=np.float64) * np.exp(-np.cumsum(jump, axis=1))
    und = fin & ~np.isfinite(rho_raw)
    ii, tt = np.nonzero(seam)
    by_date = Counter(str(dates[t].date()) for t in tt)
    by_code = Counter(str(codes[i]) for i in ii)
    cells = [{"code": str(codes[i]), "date": str(dates[t].date()), "ret_adj": float(ret[i, t]),
              "ret_raw": float(rho_raw[i, t] - 1.0)} for i, t in zip(ii, tt)]
    cells.sort(key=lambda c: -abs(c["ret_adj"] - c["ret_raw"]))
    info = {
        "threshold": thr,
        "cells_finite_ret": int(fin.sum()),
        "cells_determinable": int(det.sum()),
        "cells_undeterminable": int(und.sum()),
        "cells_undeterminable_abs_ret_gt_0.3": int((und & (np.abs(ret) > 0.3)).sum()),
        "cells_big_dev_explained_kept": int((big & explained).sum()),
        "seam_cells": int(seam.sum()),
        # 어느 쪽이 튀었나(보고만): 수정 쪽 |수익|>0.3 이면 수정표 이음매, 원 쪽이면 원 종가 오류일 수 있다
        "seam_cells_adj_abs_ret_gt_0.3": int((seam & (np.abs(ret) > 0.3)).sum()),
        "seam_cells_raw_abs_ret_gt_0.3": int((seam & (np.abs(rho_raw - 1.0) > 0.3)).sum()),
        "seam_codes": len(by_code),
        "seam_top_dates": by_date.most_common(10),
        "seam_top_codes": by_code.most_common(10),
        "seam_cells_largest": cells[:30],
        "secondary_min_abs_ret": secondary_min_abs_ret,
        "seam_cells_by_source": [int((seam & (src_id == k)).sum()) for k in range(len(raw_sources))],
        "candidates_by_source": [int((cand & (src_id == k)).sum()) for k in range(len(raw_sources))],
        "kept_secondary_undeterminable": int(kept_secondary.sum()),
        "kept_secondary_abs_ret_gt_0.1": int((kept_secondary & (np.abs(ret) > 0.1)).sum()),
    }
    return ret_f, close_f, seam, info
