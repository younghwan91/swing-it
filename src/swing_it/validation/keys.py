"""대상·시나리오 이름으로 정하는 난수 — 실행 순서·작업자 수와 상관없이 같은 값이 나온다."""

from __future__ import annotations

import zlib

import numpy as np

SEED = 20261101


def keyed_rng(name: str, *extra: int, seed: int = SEED) -> np.random.Generator:
    """이름 + 정수 키 시드(PEAD 최적화 PO-1.7)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(zlib.crc32(name.encode()), *extra)))
