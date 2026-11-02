"""파이프라인 11단계 — 순서가 곧 규칙이다. 표준 라이브러리만 쓴다(`scripts/check_guardrails.py` 가 읽는다)."""

from __future__ import annotations

from enum import IntEnum

_KOREAN = {
    1: "유니버스",
    2: "데이터",
    3: "알파 발굴",
    4: "알파 검증(IC/ICIR)",
    5: "ML 수익률 예측",
    6: "팩터 위험모델",
    7: "포트폴리오 최적화",
    8: "거래비용 모델",
    9: "백테스트",
    10: "주문 실행",
    11: "성과 기여·모니터링",
}


class Stage(IntEnum):
    UNIVERSE = 1
    DATA = 2
    ALPHA = 3
    ALPHA_VALIDATION = 4
    FORECAST = 5
    RISK_MODEL = 6
    PORTFOLIO = 7
    COSTS = 8
    BACKTEST = 9
    EXECUTION = 10
    ATTRIBUTION = 11

    @property
    def ko(self) -> str:
        return _KOREAN[int(self)]

    @property
    def label(self) -> str:
        return f"{int(self)}단계 {self.ko}"

    @classmethod
    def parse(cls, value: "Stage | int | str") -> "Stage":
        """번호·영문 이름을 받는다. 모르는 값은 ``ValueError``."""
        if isinstance(value, Stage):
            return value
        if isinstance(value, str) and not value.strip().isdigit():
            try:
                return cls[value.strip().upper()]
            except KeyError:
                raise ValueError(f"모르는 단계: {value!r}") from None
        try:
            return cls(int(value))
        except (TypeError, ValueError):
            raise ValueError(f"모르는 단계: {value!r}") from None
