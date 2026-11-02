"""북의 포크 풀. 작업 입력은 ``initializer`` 로 넘긴다 — 포크라 복사·피클이 없고, 부모에 전역 상태가 남지 않는다."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from typing import Any, Callable


def fork_pool(workers: int, initializer: Callable[..., None], initargs: tuple[Any, ...]):
    """``initializer(*initargs)`` 를 마친 포크 작업자 풀. ``OMP_NUM_THREADS`` 가 없으면 작업자마다 스레드를 띄워 경합하므로 경고한다(CLAUDE.md 실행 환경 함정)."""
    if "OMP_NUM_THREADS" not in os.environ:
        warnings.warn("OMP_NUM_THREADS 가 없다 — 포크 작업자마다 스레드를 띄워 경합한다. 1 로 띄운다", RuntimeWarning, stacklevel=2)
    return mp.get_context("fork").Pool(workers, initializer=initializer, initargs=initargs)
