"""금액 단위 — 거래대금 문턱을 맨 숫자로 쓰지 않게 한다(린트 (n)).

``daily_bars``·``daily_bars_adjusted`` 의 ``trade_value`` 는 **백만원** 단위다(ADV 10억 = 1,000). 맨 숫자 문턱은 단위를 몰라도
그럴듯해 보여서 한 자리 틀려도 아무도 못 본다 — 실제로 4f6ef61 이 문턱을 원 단위로 배선했고, ``features/universe.py`` 의 주석은
10,000 을 "10억" 이라 적었다(실제 100억). 문턱은 이 함수들로 쓴다: ``ADV_FLOOR = eok(200)`` → 20,000.0(백만원).

반환은 모두 **백만원** float(정확한 정수배라 옛 리터럴과 비트 단위로 같다 — ``eok(200) == 20000.0``).
"""

from __future__ import annotations

#: 1억 원 = 100 백만원.
_EOK = 100.0
#: 1조 원 = 1,000,000 백만원.
_JO = 1_000_000.0


def eok(n: float) -> float:
    """억 원 → 백만원(``trade_value`` 단위). ``eok(10) == 1000.0``(10억)."""
    return n * _EOK


def jo(n: float) -> float:
    """조 원 → 백만원. ``jo(1) == 1_000_000.0``."""
    return n * _JO


def mil(n: float) -> float:
    """백만원 그대로(이미 백만원으로 적은 값이라는 표시). ``mil(500) == 500.0``."""
    return float(n)


def won_to_mil(won: float) -> float:
    """원 → 백만원."""
    return won / 1e6
