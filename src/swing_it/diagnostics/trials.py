"""다중검정 원장 — 한 알파에 **실제로 시도한 config 수**를 세는 장치.

구현(지문·중복 제거·손상 줄 무시·append-only JSONL)은 ``krx_quant_core.stats.trials``
에 있다. 코어는 설치된 패키지라 원장 위치를 추정할 수 없어 ``logs_dir`` 을 필수로
받는다. 이 모듈은 swing-it 의 기본 위치 ``research/logs`` 만 채워 넘기는 얇은 래퍼라,
``prop_gate(config=...)`` 등 기존 호출부는 바뀌지 않는다.

Deflated Sharpe / t-haircut(``gate_report``)의 시행 수 ``N`` 을 사람이 손으로 적으면
규율이 아니라 부탁이 된다(GUARDRAILS §4 공백 5). 게이트가 config 를 원장에 적고 N 은
원장에서 **읽는다**. 원장 위치: ``research/logs/<label>/TRIALS.jsonl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from krx_quant_core.stats import trials as _core
from krx_quant_core.stats.trials import LEDGER_NAME, config_fingerprint

__all__ = [
    "LEDGER_NAME",
    "config_fingerprint",
    "count_trials",
    "ledger_path",
    "read_trials",
    "record_trial",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _logs_dir(logs_dir: "Path | str | None") -> Path:
    # 호출 시점에 _repo_root 를 찾는다 — 테스트가 monkeypatch 로 바꿀 수 있어야 한다.
    return Path(logs_dir) if logs_dir is not None else _repo_root() / "research" / "logs"


def ledger_path(label: str, *, logs_dir: "Path | str | None" = None) -> Path:
    """``research/logs/<label>/TRIALS.jsonl`` 경로 (파일이 없어도 경로만 반환)."""
    return _core.ledger_path(label, logs_dir=_logs_dir(logs_dir))


def record_trial(label: str, config: dict, *, note: str | None = None,
                 logs_dir: "Path | str | None" = None) -> str:
    """config 를 원장에 append 하고 지문을 반환. 같은 지문이 이미 있으면 다시 안 적는다."""
    return _core.record_trial(label, config, note=note, logs_dir=_logs_dir(logs_dir))


def count_trials(label: str, *, logs_dir: "Path | str | None" = None) -> int:
    """이 알파에 기록된 **서로 다른** config 수. 원장이 없으면 0."""
    return _core.count_trials(label, logs_dir=_logs_dir(logs_dir))


def read_trials(label: str, *, logs_dir: "Path | str | None" = None) -> list[dict[str, Any]]:
    """원장 전체를 순서대로 반환(사람이 읽거나 문서에 싣기 위한 것)."""
    return _core.read_trials(label, logs_dir=_logs_dir(logs_dir))
