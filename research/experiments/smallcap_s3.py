"""smallcap 3단계 — 사전등록 신호 구현 확인(S.md). 탐색 절반 연도별 사건 수만 낸다. 사건 뒤 수익은 계산하지 않는다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE = ("smallcap", Stage.ALPHA)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "smallcap"


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, _ = L.load_explore_panel(con)
    L.assert_explore(p.codes)
    us = {"U": p.universe(L.BAND_U), "U2": p.universe(L.BAND_U2)}
    lines = ["# smallcap 3단계 — 신호 구현 (탐색 절반, 2026-11-02, 수익 미계산)", "",
             "신호는 PREREG §4·§6 그대로다. 탐색 없음. 코드: `research/experiments/smallcap_lib.py` `hvnm_events`·`mom_events`, 손계산 시험 `tests/test_smallcap_lib.py`.", ""]
    yrs = None
    rows = []
    for sig, fn in (("HVNM", L.hvnm_events), ("MOM", L.mom_events)):
        for un, u in us.items():
            ev = fn(p, u)
            y = [str(p.dates[t])[:4] for t in ev.t]
            import pandas as pd
            c = pd.Series(y).value_counts().sort_index()
            yrs = sorted(c.index)
            rows.append((f"{sig}·{un}", len(ev), ev.c.nunique(), c))
    lines += ["| 신호·대역 | 사건 | 종목 | " + " | ".join(yrs) + " |", "|---|---|---|" + "---|" * len(yrs)]
    for name, n, nc, c in rows:
        lines.append(f"| {name} | {n:,} | {nc:,} | " + " | ".join(f"{int(c.get(y, 0)):,}" for y in yrs) + " |")
    (OUT / "S.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
