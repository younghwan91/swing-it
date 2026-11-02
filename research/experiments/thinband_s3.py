"""thinband 3단계 — 무작위 선택 구현과 격자 수(S.md). 수익을 계산하지 않는다."""

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

PIPELINE = ("thinband", Stage.ALPHA)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "thinband"


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, _, _ = B.load_panel(con)
    per = B.periods_of(p.sig_t0, p.T)
    u, u2 = p.universe(L.BAND_U), p.universe(L.BAND_U2)
    rows = [f"| {str(p.dates[a])[:10]} | {b - a} | {int(u[:, a].sum())} | {int(u2[:, a].sum())} |" for a, b in per]
    lines = ["# thinband 3단계 — 신호 없음, 무작위 선택 구현 (2026-11-02, 수익 미계산)", "",
             "선택은 `thinband_lib.simulate`(시드 고정 난수), 손계산 시험 `tests/test_thinband_lib.py`. 발굴 없음.", "",
             f"기간 {len(per)}. 첫 격자 {str(p.dates[per[0][0]])[:10]}, 끝 {str(p.dates[per[-1][1]])[:10]}.", "",
             "| 격자일 | 행 | U 종목 | U2 종목 |", "|---|---|---|---|", *rows]
    (OUT / "S.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7]), np.__name__, pd.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
