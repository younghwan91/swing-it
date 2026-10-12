"""22세대 주도주 보유형 진화(META_EVOLUTION.md 22세대 — 결과 전 고정): 21세대 골격(격자 215의 θ·N·청산·손절·추가) 위에 국장 노출·국면·종목 순위 팔.

격자 = 동시 보유 {10, 20, 30} × 국면 {200일선(21세대), 200일선 ∨ 50일선 회복, 시장 폭 ≥ 40%} × 순위 거르기 {RS, RS ∧ 강한 업종(WICS 상위 30%),
RS ∧ 실적 YoY ≥ 25%} × 종목군 {시총 상위 200, 전체} = 54. 절차·널·기준선·바는 21세대와 같다(leader21.run). 21세대 기준선 = 격자 215 재실행.

사용: ``uv run --with numba --extra pg python research/experiments/leader22.py --cache <v13> --out <dir> --workers 4`` (KR_QUANT_DB)
"""
from __future__ import annotations

import argparse
import sys

sys.path[:0] = [".", "src"]

import research.experiments.leader21 as L  # noqa: E402

FIXED = {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "pyramid": 1, "regime": 1}
GRID = {"max_pos": (10, 20, 30), "regime_kind": ("k200", "k200r50", "breadth"), "rank": ("rs", "rs_sector", "rs_eps"), "univ": ("top200", "all")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    L.run(GRID, ap.parse_args(), extras=True, fixed=FIXED)


if __name__ == "__main__":
    main()
