"""thinband 8단계 — 격자일 U 의 왕복 비용 분포(C8.md). 체결 전 추정치이고 북 수익은 계산하지 않는다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402
import thinband_lib as B  # noqa: E402

PIPELINE = ("thinband", Stage.COSTS)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "thinband"


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, _, _ = B.load_panel(con)
    lines = ["# thinband 8단계 — 비용 (2026-11-02, 북 수익 미계산)", "", "격자일 U·U2 종목의 왕복 1배 비용(주문 1,500만원, bp). 판정은 2배다. 호가표는 날짜별.", "",
             "| 대역 | 시기 | 5% | 50% | 95% | 호가 하한이 폴백을 넘는 비중 |", "|---|---|---|---|---|---|"]
    for name, band in (("U", L.BAND_U), ("U2", L.BAND_U2)):
        u = p.universe(band)
        out = {"~2023-01": [], "2023-01~": []}
        over = {"~2023-01": [], "2023-01~": []}
        for a, _ in B.periods_of(p.sig_t0, p.T):
            d = p.day_py[a]
            key = "2023-01~" if d >= L.TICK_REFORM else "~2023-01"
            for i in np.flatnonzero(u[:, a]):
                sig = p.SIG20[i, a]
                sig = min(sig if np.isfinite(sig) else L.VOL_FALLBACK, 1.0)
                adv, raw, mk = p.ADV[i, a], p.RAW[i, a], p.market[i]
                c = L.cost_one_way("BUY", d, mk, adv, sig, raw) + L.cost_one_way("SELL", d, mk, adv, sig, raw)
                out[key].append(c * 1e4)
                fb = next(hs for hi, hs in L.FALLBACK_HS if adv < hi)
                over[key].append(L.half_spread_bp(adv, raw, mk, d) > fb)
        for key, v in out.items():
            if v:
                q = np.quantile(v, [.05, .5, .95]).round(0)
                lines.append(f"| {name} | {key} | {q[0]:.0f} | {q[1]:.0f} | {q[2]:.0f} | {np.mean(over[key]):.1%} |")
    lines += ["", "인용 스프레드 대조는 2026 값이다(smallcap: 중앙 18.4·11.9bp, p90 28·26bp). **개혁 전 호가의 검증이 아니다. 2배가 유일한 완충이다.** y=1 은 자리표시값이다. (H+L+C)/3 는 VWAP 대비 +3.9bp 로 치우침이 작다(리뷰어 측정)."]
    (OUT / "C8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), pd.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
