"""배포 준비도 REPORTER — 판정기(verdict machine)가 아니라 신호 보고기.

이 모듈은 배포 준비도를 **숫자·분포**로 보고한다: OOS 기대값 R 의 부트스트랩
신뢰구간, 워크포워드 폴드 일관성 COUNT, 괴물 의존도, 최장 연패, (비용 스윕이
주어지면) 엣지가 죽는 비용. 그게 전부다.

이 모듈은 의도적으로 PASS/FAIL 을 내지 않으며 "monster<50%" 같은 임계값을
하드코딩하지 않는다. 확률적(stochastic) 데이터에 결정론적 판정을 박는 것 자체가
그 자체로 과최적 규칙(Principle 1: stochastic, not deterministic 위반)이기 때문이다.
이 신호들을 읽고 배포 여부를 판단하는 것은 연구자의 몫이다 — 이 함수의 몫이 아니다.

모든 입력은 순수 배열/매핑(R-멀티플, 진입일, 폴드별 기대값, 비용→기대값 곡선).
research/ 로부터 아무것도 import 하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# 다중검정 보정(Deflated Sharpe·PSR·t-haircut)과 기대값 부트스트랩 CI 의 **정본은
# krx-quant-core** 다(`krx_quant_core.stats.sharpe`). 세 레포가 같은 N 에 같은 SR0 를
# 내야 비교가 성립하므로 여기서 다시 계산하지 않는다. 옛 이름(`_bootstrap_ci` 등)은
# research/·tests 가 import 하므로 별칭으로 남긴다.
#
#   Deflated Sharpe Ratio — Bailey & López de Prado (2014), JPM 40(5), 94–107.
#   t-haircut — Harvey & Liu (2014), "…and the Cross-Section of Expected Returns".
#
# 이식하며 바뀐 한 가지: 퇴화 판정이 `std == 0` → `std < 1e-12`. 0.3 같은 상수열이
# 반올림 잔차(std≈1e-17)로 Sharpe ~5e15 를 내던 버그가 고쳐졌다(이제 NaN).
from krx_quant_core.stats.sharpe import bootstrap_mean_ci as _bootstrap_ci
from krx_quant_core.stats.sharpe import deflated_sharpe
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample as _deflation_block
from krx_quant_core.stats.sharpe import expected_max_sharpe_h0 as _expected_max_sharpe_h0
from krx_quant_core.stats.sharpe import probabilistic_sharpe as _prob_sharpe
from krx_quant_core.stats.sharpe import t_haircut as _t_haircut

from .fragility import max_loss_streak, monster_share

__all__ = [
    "gate_report",
    "deflated_sharpe",
    # 하위호환 별칭 — 구현은 krx_quant_core.stats.sharpe
    "_bootstrap_ci",
    "_deflation_block",
    "_expected_max_sharpe_h0",
    "_prob_sharpe",
    "_t_haircut",
]

NAN = float("nan")


def _cost_edge_dies(cost_curve: Mapping | Sequence) -> float | None:
    """엣지가 죽는 비용 — 기대값 R 이 처음으로 ≤0 이 되는 최소 비용.

    반환값 셋을 구분한다: 비용(죽는 지점) · ``None``(전 구간 생존) ·
    ``nan``(잴 수 없음 — 유효한 관측이 하나도 없다). 마지막을 ``None`` 과 묶으면
    빈 표본이 "생존" 으로 보고된다.

    ``cost_curve``: {비용: 기대값R} 매핑 또는 (비용, 기대값R) 시퀀스. 비용 오름차순으로 검사.
    """
    items = sorted(cost_curve.items()) if isinstance(cost_curve, Mapping) else sorted(cost_curve)
    seen = False
    for cost, exp in items:
        # NaN 은 "안 죽었다" 가 아니라 "잴 수 없었다" 다. `NaN <= 0` 이 False 라
        # 그냥 두면 **표본이 빈 곡선이 '전 구간 생존' 으로 출력된다** — 유니버스가
        # 비어 폴드가 n=0 인 상황에서 정확히 그 모양이 나온다.
        if exp != exp:          # NaN
            continue
        seen = True
        if exp <= 0:
            return float(cost)
    return None if seen else float("nan")


def gate_report(
    oos_R: np.ndarray,
    *,
    n_trials: int | None = None,
    fold_expectancies: Sequence | None = None,
    entry: np.ndarray | None = None,
    monster_k: int = 5,
    cost_curve: Mapping | Sequence | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """배포 준비도 신호를 구조화된 dict 로 보고한다 — PASS/FAIL 없음, 임계값 없음.

    이 함수는 REPORTER 다. bool 판정 키를 담지 않으며 임계값을 하드코딩하지 않는다.
    반환된 신호를 해석해 배포를 결정하는 것은 연구자의 판단이다 (모듈 docstring 참조).

    Args:
        oos_R: OOS 개별 트레이드 R-멀티플 배열.
        n_trials: 시도한 config 수(옵션). 주어지면 deflation 블록(Deflated Sharpe +
            Harvey-Liu t-haircut)을 계산해 선택편향을 숫자로 깎는다. None 이면 블록 생략
            (기본 동작 불변).
        fold_expectancies: 워크포워드 폴드별 기대값 R 시퀀스(옵션). 주어지면
            fold_consistency 에 폴드 수·양(+)폴드 COUNT 를 보고(판정 아님).
        entry: 진입일 배열(옵션) — 최장 연패를 시간순으로 계산할 때 사용.
        monster_k: 괴물 의존도 상위 k.
        cost_curve: {비용: 기대값R} 매핑/시퀀스(옵션) — 엣지가 죽는 비용 보고.
        n_boot, seed, ci: 기대값 부트스트랩 신뢰구간 파라미터.

    Returns:
        dict: n, expectancy_R, expectancy_ci=(lo, hi), monster_share, max_loss_streak,
        (n_trials 주면) deflation={n_trials, observed_sharpe, expected_max_sharpe_h0,
        deflated_sharpe, prob_sharpe_gt0, prob_deflated_sharpe, t_haircut},
        (fold_expectancies 주면) fold_consistency={n_folds, n_positive, fold_expectancies},
        (cost_curve 주면) cost_edge_dies. bool 판정 키는 포함하지 않는다.
    """
    R = np.asarray(oos_R, float)
    R = R[np.isfinite(R)]
    n = int(len(R))
    report: dict = {
        "n": n,
        "expectancy_R": float(R.mean()) if n else NAN,
        "expectancy_ci": _bootstrap_ci(R, n_boot=n_boot, seed=seed, ci=ci),
        "monster_share": monster_share(R, k=monster_k),
        "max_loss_streak": max_loss_streak(oos_R, entry=entry),
    }
    if n_trials is not None and n_trials >= 1:
        report["deflation"] = _deflation_block(R, n_trials)
    if fold_expectancies is not None:
        fe = np.asarray(fold_expectancies, float)
        report["fold_consistency"] = {
            "n_folds": int(len(fe)),
            "n_positive": int((fe > 0).sum()),
            "fold_expectancies": [float(x) for x in fe],
        }
    if cost_curve is not None:
        report["cost_edge_dies"] = _cost_edge_dies(cost_curve)
    return report
