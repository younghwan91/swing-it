"""smallcap 2단계 — 탐색 절반 데이터 대조(D.md). 사건·전략 수익을 계산하지 않는다. 대조에는 칸별 일간 변화 크기만 쓴다."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from swing_it.pipeline import Stage, require_stage
from swing_it.storage import connect

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smallcap_lib as L  # noqa: E402

PIPELINE = ("smallcap", Stage.DATA)
OUT = Path(__file__).resolve().parents[2] / "research" / "logs" / "smallcap"


def main() -> int:
    require_stage(*PIPELINE)
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True, autocommit=True)
    con.cursor().execute("set statement_timeout='180s'")
    p, df = L.load_explore_panel(con)
    L.assert_explore(p.codes)
    w = df[(df.date >= L.SIG_START) & (df.volume > 0)].copy()
    lst, dl = w[w.source == "kiwoom_raw"], w[w.source == "naver_adj"]
    both = lst[(lst.ncl > 0) & (lst.raw_close > 0)]
    agree = float(((both.ncl / both.raw_close - 1).abs() < 0.005).mean())
    tvr = (lst.ncl * lst.nvol / 1e6 / lst.trade_value)[(lst.trade_value > 0) & (lst.nvol > 0)]
    # v2 대 원 거래대금: 폐지 대역 칸
    d2 = df.sort_values(["code", "date"]).copy()
    d2["tv_raw"] = (d2.ncl * d2.nvol / 1e6).where(d2.volume > 0)
    d2["tv_v2"] = d2.trade_value.where(d2.volume > 0)
    for col in ("tv_raw", "tv_v2"):
        d2["adv_" + col] = d2.groupby("code")[col].transform(lambda s: s.shift(1).rolling(20, min_periods=15).mean())
    z = d2[(d2.date >= L.SIG_START) & (d2.volume > 0) & (d2.source == "naver_adj")]
    in_v2 = (z.adv_tv_v2 >= L.BAND_U[0]) & (z.adv_tv_v2 < L.BAND_U[1])
    in_raw = (z.adv_tv_raw >= L.BAND_U[0]) & (z.adv_tv_raw < L.BAND_U[1])
    u, u2 = p.universe(L.BAND_U), p.universe(L.BAND_U2)
    t0 = p.sig_t0
    over = p.traded & (np.abs(np.nan_to_num(p.r)) > np.where(p.is_new_limit, 0.305, 0.155))
    ended = (p.last < p.T - 1) & (p.last >= 0)
    gate = {"naver_vs_raw_ok": agree >= 0.99, "explore_only": all(L.half_of(c) == 0 for c in p.codes),
            "delisted_loaded": bool(ended.sum() > 100), "market_mapped": set(p.market) <= {"KOSPI", "KOSDAQ"}}
    lines = [
        "# smallcap 2단계 — 데이터 대조 (탐색 절반만, 2026-11-02, 전략 수익 미계산)", "",
        f"패널: 종목 {p.N:,} × 거래일 {p.T:,}, {str(p.dates[0])[:10]}~{str(p.dates[-1])[:10]}, 잠금 절반 0종목(단언).", "",
        "| 대조 | 값 | 기준 |", "|---|---|---|",
        f"| 상장: 네이버 종가 = `raw_close`(±0.5%) | {agree:.4%} ({len(both):,}칸) | ≥ 99% |",
        f"| 상장: 원 종가×거래량 / v2 거래대금, 5·50·95 백분위 | {tvr.quantile([.05, .5, .95]).round(3).tolist()} | 기록 |",
        f"| 폐지 칸: 네이버 값 없음 | {float(dl.ncl.isna().mean()):.3%} | U 밖 처리 |",
        f"| 폐지 대역 칸: v2 기준 대역 {int(in_v2.sum()):,} · 원 기준 대역 {int(in_raw.sum()):,} · v2 만 {int((in_v2 & ~in_raw).sum()):,} · 원 만 {int((in_raw & ~in_v2).sum()):,} | | C1 근거 |",
        f"| 제한폭 초과 칸(시그널 구간) | {int(over[:, t0:].sum()):,} | 이음매 처리 |",
        f"| 이음매 칸(재개·초과, 꼬리 제외) | {int(p.seam[:, t0:].sum()):,} | 이익 0·손실 유지, 벤치 0 |",
        f"| 수정 사건 칸(수정가/원가 계단 > 1%) | {int(p.action[:, t0:].sum()):,} ({len(set(np.nonzero(p.action[:, t0:])[0])):,}종목) | 사건에서 제외 |",
        f"| 시리즈가 표본 전에 끝난 종목 | {int(ended.sum()):,} (부실 분류 {int((p.distress & ended).sum()):,}) | 꼬리 {L.TAIL_ROWS}행 제한폭 잠김 없음 |",
        f"| 꼬리 칸 | {int(p.tail.sum()):,} | |",
        f"| 수정가/원가 < 2% 칸(자료 오류, U 밖) | {int(p.bad_adj[:, t0:].sum()):,} ({len(set(np.nonzero(p.bad_adj[:, t0:])[0])):,}종목) | 정정 SC-2 |",
        f"| 시장 매핑 | KOSPI {int((p.market == 'KOSPI').sum()):,} · KOSDAQ {int((p.market == 'KOSDAQ').sum()):,} | 단언 |",
        f"| U(t) 종목-일 / U2(t) | {int(u.sum()):,} / {int(u2.sum()):,} | |",
        f"| 대역 칸 고가·저가 없음(대표가 = 종가) | {float((~np.isfinite(p.H) | ~np.isfinite(p.L))[u].mean()):.4%} | 기록 |", "",
        f"**판정: {'통과' if all(gate.values()) else '실패'}** — {gate}",
    ]
    (OUT / "D.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
