"""게이트 실행 기록 — krx-quant-core ``start_run`` 얇은 래퍼.

``prop_gate(config=...)`` 가 돌 때마다 ``research/runs/<label>/RUNS.jsonl`` 에 git sha·config·진입일
범위·시행 수·핵심 숫자를 남긴다. 시행 원장은 swing-it 이 쌓아 온 ``research/logs`` 를 그대로 이어
쓴다(``trials_dir``) — 같은 config 지문이면 N 이 늘지 않으니 :func:`~.trials.record_trial` 과 겹쳐도
중복 계산이 없다.

기록은 **simnode 의 git 체크아웃에서만** 남긴다. 백테스트는 simnode 에서만 돈다는 규칙의 기록판이라,
CI·개발 머신에서 도는 합성 테스트는 기록 없이 통과해야 하기 때문이다. 미커밋 상태는 막지 않고
``dirty`` 로 적는다(게이트 탐색 중 흔한 상태).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from krx_quant_core.runtime import DataSpec, start_run
from krx_quant_core.runtime.host import BACKTEST_HOST

from . import trials as _trials

__all__ = ["record_gate_run"]


def _scalars(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if isinstance(v, (bool, int, float, str, type(None)))}


def record_gate_run(
    label: str,
    config: dict,
    *,
    entry_range: tuple[str | None, str | None],
    result: dict[str, Any],
    logs_dir: Path | str | None = None,
) -> str | None:
    """기록한 run_id, 기록 조건이 아니면(simnode 아님·git 체크아웃 아님) ``None``."""
    if socket.gethostname() != BACKTEST_HOST:
        return None
    repo = _trials._repo_root()  # 테스트가 monkeypatch 하는 같은 함수 — 원장과 같은 레포를 본다
    if not (Path(repo) / ".git").exists():
        return None
    lo, hi = entry_range
    data = DataSpec(str(lo)[:10], str(hi)[:10], "gate") if lo and hi else None
    trials_dir = Path(logs_dir).resolve() if logs_dir is not None else _trials._logs_dir(None)
    with start_run(
        label, config, repo_root=repo, data=data, trials_dir=trials_dir, allow_dirty=True
    ) as run:
        run.log_result(_scalars(result))
    return run.run_id
