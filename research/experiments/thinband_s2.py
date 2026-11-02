"""thinband 2단계 — 데이터 대조(D.md). 북 수익을 계산하지 않는다. 칸별 일간 변화 크기·값 대조만."""

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

PIPELINE = ("thinband", Stage.DATA)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "thinband"


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, df, kind = B.load_panel(con)
    assert str(p.dates[-1])[:10] <= B.DATA_END
    t0 = p.sig_t0
    w = df[(df.date >= B.SIG_START) & (df.volume > 0)]
    lst, dl = w[w.source == "kiwoom_raw"], w[w.source == "naver_adj"]
    both = lst[(lst.ncl > 0) & (lst.raw_close > 0)]
    agree = float(((both.ncl / both.raw_close - 1).abs() < 0.005).mean())
    tvr = (lst.ncl * lst.nvol / 1e6 / lst.trade_value)[(lst.trade_value > 0) & (lst.nvol > 0)]
    u, u2 = p.universe(L.BAND_U), p.universe(L.BAND_U2)
    doom = B.doom_mask(p, kind)
    ended = (p.last < p.T - 1) & (p.last >= 0)
    in_u = u[:, t0:].any(axis=1)
    sh = pd.read_sql_query("select code,knowledge_date,shares_outstanding from shares_outstanding_history where shares_outstanding>0 and code = any(%s)",
                           con, params=(p.codes,))
    sh["knowledge_date"] = pd.to_datetime(sh.knowledge_date)
    first_k = sh.groupby("code").knowledge_date.min().reindex(p.codes)
    cov = {}
    for d in ("2016-12-01", "2017-04-03", "2017-09-01", "2020-01-02", "2023-06-01"):
        t = int(np.searchsorted(p.dates, np.datetime64(d)))
        names = np.flatnonzero(u[:, t])
        cov[d] = round(float((first_k.iloc[names] < pd.Timestamp(d)).mean()), 3)
    over = p.traded & (np.abs(np.nan_to_num(p.r)) > 0.305)
    gate = {"naver_vs_raw_ok": agree >= 0.99, "kind_ok": True, "delisted_loaded": bool(ended.sum() > 100),
            "market_mapped": set(p.market) <= {"KOSPI", "KOSDAQ"}, "end_ok": str(p.dates[-1])[:10] <= B.DATA_END, "periods_30": len(B.periods_of(t0, p.T)) == 30}
    lines = ["# thinband 2단계 — 데이터 대조 (2026-11-02, 북 수익 미계산)", "",
             f"패널: 종목 {p.N:,} × 행 {p.T:,}, 첫 행 {str(p.dates[0])[:10]}, 표본 시작 {str(p.dates[t0])[:10]}, 끝 {str(p.dates[-1])[:10]}, 기간 {len(B.periods_of(t0, p.T))}. 2016-09-01 전 가격은 안 읽었다(거래량·거래대금만).", "",
             "| 대조 | 값 |", "|---|---|",
             f"| 상장: 네이버 종가 = `raw_close`(±0.5%) | {agree:.4%} ({len(both):,}칸), 기준 ≥ 99% |",
             f"| 상장: 원 종가×거래량 / v2 거래대금 5·50·95 | {tvr.quantile([.05, .5, .95]).round(3).tolist()} |",
             f"| 폐지 칸 네이버 없음 | {float(dl.ncl.isna().mean()):.3%} |",
             f"| U·U2 종목-일 | {int(u.sum()):,} · {int(u2.sum()):,} |",
             f"| 제한폭 초과 칸 | {int(over[:, t0:].sum()):,} |",
             f"| 이음매 칸 | {int(p.seam[:, t0:].sum()):,} |",
             f"| 수정가/원가 < 2% 칸 | {int(p.bad_adj[:, t0:].sum()):,} ({len(set(np.nonzero(p.bad_adj[:, t0:])[0]))}종목) |",
             f"| 시리즈가 표본 전에 끝난 종목 | {int(ended.sum()):,}, 그중 U 에 든 적 있음 {int((ended & in_u).sum()):,} |",
             f"| 부실·정지로 표본을 나가는 종목(−100% 대상) | {int(doom.sum())}, U 에 든 적 있음 {int((doom & in_u).sum())} |",
             f"| 주식수 원천 `shares_outstanding_history` 첫 알게 된 날 < 그날, U 비중 | {cov} |",
             f"| 시장 매핑 | KOSPI {int((p.market == 'KOSPI').sum()):,} · KOSDAQ {int((p.market == 'KOSDAQ').sum()):,} |", "",
             f"**판정: {'통과' if all(gate.values()) else '실패'}** — {gate}"]
    (OUT / "D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
