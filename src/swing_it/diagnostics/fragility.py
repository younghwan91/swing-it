"""취약성 진단 — ``krx_quant_core.stats.fragility`` 재수출.

괴물 의존도·최장 연패·꼬리제거 민감도·중앙값·승리조건부 분포의 구현은 krx-quant-core
로 옮겼다(수치 동일). 개별 트레이드 R-멀티플 분포 렌즈라는 성격은 그대로다 — 포트폴리오
프레이밍(슬롯·동시보유·연환산·회전율)은 없다. 여기서 다시 구현하지 말 것.
"""

from __future__ import annotations

from krx_quant_core.stats.fragility import (
    fragility_report,
    max_loss_streak,
    median_trade,
    monster_share,
    tail_removal,
    win_conditional,
)

__all__ = [
    "fragility_report",
    "max_loss_streak",
    "median_trade",
    "monster_share",
    "tail_removal",
    "win_conditional",
]
