"""smallcap 거래 재계산 — 패널·엔진을 거치지 않고 원 DB 행으로 표본 체결을 다시 낸다(검증). 탐색 절반 표본만."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE = ("smallcap", Stage.ALPHA_VALIDATION)
N_SAMPLE = 40


def recompute(rows: pd.DataFrame, calendar: pd.DatetimeIndex, e_day, last_day_end) -> dict:
    """한 종목의 v2·네이버 행으로 체결 하나. 날짜는 달력 위치로 잡는다."""
    g = rows.set_index("date").reindex(calendar)
    c, h, l_, v = g.close.to_numpy(float), g.high.to_numpy(float), g.low.to_numpy(float), g.volume.to_numpy(float)
    tv = (g.ncl * g.nvol / 1e6).to_numpy(float)
    traded = np.nan_to_num(v) > 0
    e = int(np.searchsorted(calendar.values, np.datetime64(e_day)))
    tr_idx = np.flatnonzero(traded)
    last = int(tr_idx[-1])
    tail = set(tr_idx[-L.TAIL_ROWS:].tolist()) if last < len(calendar) - 1 else set()
    cff = pd.Series(c).ffill().to_numpy()
    pos = 1.0                                   # 진입일 종가 대비 값(이음매: 이익 0)
    tp_e = (h[e] + l_[e] + c[e]) / 3.0 if np.isfinite(h[e]) and h[e] > 0 and np.isfinite(l_[e]) and l_[e] > 0 else c[e]
    base = c[e] / tp_e
    limit_change = np.datetime64("2015-06-15")
    frac, gross, x, selling, locked = 1.0, 0.0, None, False, False
    for k in range(e + 1, len(calendar)):
        if k > last:
            break
        if np.isfinite(c[k]):
            r = cff[k] / cff[k - 1] - 1.0
            newlim = calendar.values[k] >= limit_change
            seam = traded[k] and ((not traded[k - 1]) or abs(r) > (0.305 if newlim else 0.155)) and k not in tail
            pos *= 1.0 + (min(r, 0.0) if seam else r)
        if selling and traded[k]:
            thr = 0.29 if calendar.values[k] >= limit_change else 0.14
            r = cff[k] / cff[k - 1] - 1.0
            if k not in tail and r <= -thr + 1e-9:
                locked = True
            elif np.isfinite(tv[k]) and tv[k] > 0:
                val = base * pos
                ok = np.isfinite(h[k]) and h[k] > 0 and np.isfinite(l_[k]) and l_[k] > 0
                px = c[k] if not ok else (l_[k] if locked else (h[k] + l_[k] + c[k]) / 3.0)
                locked = False
                take = min(frac, 0.05 * tv[k] * 1e6 / (L.ORDER_WON * max(val, 1e-9)))
                gross += take * val * px / c[k]
                frac -= take
                x = k
                if frac <= 1e-12:
                    break
        if not selling and traded[k]:
            w = np.where(traded[max(0, k - 50):k], c[max(0, k - 50):k], np.nan)
            if np.isfinite(w).sum() >= L.WIN_MIN and c[k] < np.nanmin(w):
                selling = True
    if frac > 1e-12:
        gross += frac * base * pos
        x = min(last, len(calendar) - 1)
    return {"gross": gross - 1.0, "x": calendar[x]}


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='120s'")
    cache = Path(os.environ.get("SMALLCAP_CACHE", "/tmp"))
    obj = pickle.load(open(cache / "smallcap_s4_fills.pkl", "rb"))
    codes = obj["codes"]
    L.assert_explore(codes)
    p, _ = L.load_explore_panel(con)
    cal = pd.DatetimeIndex(p.dates)
    rng = np.random.default_rng(7)
    bad = 0
    tot = 0
    for name in ("HVNM", "MOM"):
        f = obj["book"][name]
        for r in f.iloc[rng.choice(len(f), size=min(N_SAMPLE, len(f)), replace=False)].itertuples():
            code = codes[r.c]
            rows = pd.read_sql_query(
                "select v.code,v.date,v.high,v.low,v.close,v.volume,n.close as ncl,n.volume as nvol from daily_bars_adj_v2 v "
                "left join naver_investor_trend n on n.code=v.code and n.date=v.date where v.code=%s and v.date between %s and %s",
                con, params=(code, L.DATA_START, L.DATA_END))
            rows["date"] = pd.to_datetime(rows.date)
            for col in ("close", "high", "low"):
                rows.loc[rows[col] <= 0, col] = np.nan
            out = recompute(rows, cal, cal[r.e], cal[-1])
            tot += 1
            same = abs(out["gross"] - r.gross) < 1e-6 and out["x"] == cal[r.x]
            if not same:
                bad += 1
                print("불일치", name, code, str(cal[r.e])[:10], r.gross, out["gross"], str(cal[r.x])[:10], str(out["x"])[:10])
    print(f"재계산 {tot}건, 불일치 {bad}건")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
