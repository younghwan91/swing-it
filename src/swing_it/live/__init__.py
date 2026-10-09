"""스윙 라이브 — 수렴 × 수급 × 실적 × 미너비니 청산 (설계: ``docs/superpowers/specs/2026-10-10-swing-live.md``).

**백테스트 = 라이브 = 같은 코드.** 진입 판정·주문 계획·포지션 상태 전이는 :mod:`.strategy` 하나에만 있고,
포트폴리오 백테스트(:mod:`.backtest`)와 데몬(``runtime``)이 둘 다 그것을 부른다. 연구 시뮬레이터
(``research/signals/minervini_exits.simulate_minervini``)와는 ``tests/test_live_parity.py`` 로 묶는다.

경계 규칙: 이 패키지는 ``research/`` 를 import 하지 않는다(``scripts/check_guardrails.py``).

- :mod:`.config` — :class:`~.config.StrategyConfig` · :class:`~.config.RiskConfig` (사전등록 CONVERGENCE A 수치)
- :mod:`.strategy` — ``scan`` · ``plan_entry`` · ``PositionState`` · ``Action`` (순수 로직)
- :mod:`.sizing` — 위험 1% 사이징, 10% 상한, 슬롯·현금 제약, 후보 순위
- :mod:`.features` — DB → 그날치 신호 피처(연구 경로와 같은 플래그)
- :mod:`.backtest` — 일 루프 포트폴리오 백테스트(같은 ``strategy`` 를 부른다)
"""
