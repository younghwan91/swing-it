"""README 히어로 이미지 — 순수 노이즈가 "6폴드 중 5폴드 양수"를 얼마나 자주 통과하는가.

`research/experiments/prop_gate.py` 의 랜덤 음성대조(``random_entry_control``)와 같은 절차를
numpy 로 재현한다: 진입일과 수익을 합성 풀(600 트레이드, ``_synthetic_setup(seed=1, edge=0.02)``)
에서 서로 독립으로 복원추출해 신호를 파괴하고, 고정 6폴드(``FOLDS``)의 TEST 구간별 평균 R 부호를
센다. seed=7 의 첫 50 draw 는 README §8 이 인용하는 실행과 같은 수열이다(카운트·p95 일치 확인).

실행(레포 루트): ``uv run python scripts/make_hero.py``
→ docs/images/hero-light.png · docs/images/hero-dark.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from hero_style import caption, render_both  # noqa: E402
from prop_gate import COSTS, _synthetic_setup  # noqa: E402
from swing_it.validation.optimization import TRAIN_HI  # noqa: E402
from swing_it.validation.walkforward import FOLDS, entry_mask  # noqa: E402

N_DRAWS = 5000
SEED = 7            # prop_gate.main() 의 음성대조와 같은 seed → 첫 50 draw 동일
STOP = 0.10
COST = COSTS[0]     # 46 bp 왕복(기준 비용)
PASS_K = 5          # "6폴드 중 5폴드 양수"
GENERATED = "2026-10-19"


def simulate() -> tuple[np.ndarray, np.ndarray, int]:
    """draw 별 (양수 폴드 수, OOS 기대값 R) 과 폴드 수."""
    entry, pool = _synthetic_setup(seed=1, edge=0.02)
    rng = np.random.default_rng(SEED)
    n = len(pool)
    k_pos = np.empty(N_DRAWS, int)
    oos_exp = np.empty(N_DRAWS, float)
    for i in range(N_DRAWS):
        ed = rng.choice(entry, size=n, replace=True)
        rr = rng.choice(pool, size=n, replace=True)   # 진입과 독립 → 신호 파괴
        k = 0
        for f in FOLDS:
            r = rr[entry_mask(ed, f.test_lo, f.test_hi)]
            k += int(len(r) > 0 and ((r - COST) / STOP).mean() > 0)
        k_pos[i] = k
        oos_exp[i] = ((rr[entry_mask(ed, lo=TRAIN_HI)] - COST) / STOP).mean()
    return k_pos, oos_exp, len(FOLDS)


def draw(fig, axes, p):
    ax1, ax2 = axes
    k_pos, oos_exp, n_folds = simulate()
    pass_frac = (k_pos >= PASS_K).mean()
    p95 = float(np.percentile(oos_exp, 95))

    # --- 왼쪽: 양수 폴드 수 분포 -------------------------------------------
    ks = np.arange(n_folds + 1)
    share = np.array([(k_pos == k).mean() for k in ks]) * 100
    colors = [p["amber"] if k >= PASS_K else p["blue"] for k in ks]
    ax1.bar(ks, share, width=0.62, color=colors, linewidth=0)
    ax1.set_xticks(ks)
    ax1.set_xlabel(f"Test folds with positive mean R (out of {n_folds})")
    ax1.set_ylabel("Share of signal-free strategies (%)")
    ax1.set_ylim(0, share.max() * 1.32)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("Fold-count bar: how often noise clears it", fontsize=12.5, fontweight="normal", pad=10)
    # 통과 영역 직접 라벨(숫자는 이 하나만)
    y_lab = share[PASS_K:].max() + share.max() * 0.06
    ax1.annotate(
        f"{pass_frac:.0%} of pure noise\npasses {PASS_K} of {n_folds}",
        xy=((PASS_K + n_folds) / 2, y_lab), xytext=(0, 0), textcoords="offset points",
        ha="center", va="bottom", color=p["ink"], fontsize=12, fontweight="bold",
    )
    ax1.axvspan(PASS_K - 0.5, n_folds + 0.5, color=p["amber_wash"], lw=0, zorder=0)

    # --- 오른쪽: 실제 기준 — 자기 랜덤 버전의 분포와 95 백분위 ---------------
    ax2.hist(oos_exp, bins=40, color=p["blue"], linewidth=0)
    ax2.axvline(p95, color=p["ink"], linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Out-of-sample expectancy (R) of the random version")
    ax2.set_ylabel("Strategies")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("The bar actually used: beat your own random version", fontsize=12.5, fontweight="normal", pad=10)
    ymax = ax2.get_ylim()[1]
    ax2.set_ylim(0, ymax * 1.18)
    ax2.annotate("null 95th percentile = the bar.\nA real strategy must land right of it.",
                 xy=(p95, ymax * 1.14), xytext=(-7, 0), textcoords="offset points",
                 ha="right", va="top", color=p["ink"], fontsize=10.5)

    fig.suptitle("Pure noise passes “5 of 6 folds positive” about half the time",
                 x=0.0, ha="left", fontsize=16, fontweight="bold", color=p["ink"], y=1.03)
    fig.tight_layout(w_pad=3.0)
    caption(
        fig,
        f"Simulation, {N_DRAWS:,} signal-free strategies (seed {SEED}), entry dates and returns resampled from a "
        f"600-trade synthetic pool with no entry signal — prop_gate.py random_entry_control.\n"
        f"6 rolling test folds 2020–2026, cost 46 bp, stop 10%. First 50 draws = README run. {GENERATED}.",
        p,
    )


if __name__ == "__main__":
    paths = render_both(draw, out_dir=str(ROOT / "docs" / "images"), name="hero", figsize=(12, 5),
                        ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
    for pth in paths:
        print(pth)
