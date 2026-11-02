"""2단계 데이터 — 정문(``storage.read_prices``·``read_earnings``)으로 읽고 지식 시점 보증을 붙인다.

유니버스로 SQL 을 좁히지 않는다. 좁히면 ``read_prices`` 의 폐지 종목 검사가 우회된다. 마스크는 3단계에서 신호에 건다.
"""

from __future__ import annotations

import json
from typing import Any

from swing_it.pipeline.contracts import STAGE_CONTRACTS, DataPanels, UniverseMask, expect
from swing_it.pipeline.gate import require_stage
from swing_it.pipeline.stages import Stage
from swing_it.storage import _EARNINGS_READ_COLS, read_earnings, read_prices

STAGE = Stage.DATA
CONTRACT = STAGE_CONTRACTS[STAGE]
REFERENCES = (
    "src/swing_it/storage.py::read_prices",
    "src/swing_it/storage.py::read_earnings",
    "research/experiments/pead_optimizer_b.py::fingerprint_range",
)


def _fingerprint(prices: Any) -> str:
    """행 수·최대 날짜·Σround(close×100)·Σvolume — 단계 B 의 표 지문과 같은 항목(``fingerprint_range``)."""
    close = prices["close"].astype(float)
    vol = prices["volume"].astype(float).sum() if "volume" in prices else None
    return json.dumps([len(prices), str(prices["date"].max()), int((close * 100).round().sum()),
                       None if vol is None else int(vol)])


def run(study: str, con: Any, universe: UniverseMask, *, end: str, aftermarket_handled: bool = False) -> DataPanels:
    """``end`` 까지의 가격과 실적 전 버전을 읽는다. 실적은 접지 않는다 — 버전 선택은 날짜별로 3단계가 한다."""
    require_stage(study, STAGE)
    expect(universe, UniverseMask, "universe")
    prices = read_prices(con)
    prices = prices.assign(date=prices["date"].astype(str))
    prices = prices[prices["date"] <= end].reset_index(drop=True)
    earnings = read_earnings(con, cols=(*_EARNINGS_READ_COLS, "knowledge_date"), all_versions=True, until=end)
    return DataPanels(prices=prices, earnings=earnings, universe=universe, end=end,
                      fingerprint=_fingerprint(prices), aftermarket_handled=aftermarket_handled)
