"""krx-quant-core 의 미배포 레이어를 부를 때만 import 한다.

swing-it 기본 핀(0.7.0)에는 위험모델·최적화·충격비용·기여가 없다. 모듈 머리에서 끌어오면 그 핀에서 import 부터 죽는다.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class CoreLayerMissing(ImportError):
    """krx-quant-core 에 그 레이어가 없다."""


def core_layer(module: str) -> ModuleType:
    """미배포 core 레이어를 import 한다. 없으면 무엇을 해야 하는지 말한다."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("krx_quant_core"):
            raise                                   # 그 레이어 안의 다른 의존성 문제는 가리지 않는다
        raise CoreLayerMissing(
            f"{module} 가 없다. swing-it 기본 핀 krx-quant-core==0.7.0 에는 위험모델·최적화·충격비용이 없고, core main 에만 있다"
            f"(미배포 — PyPI 배포는 사용자 지시가 필요하다). 지금 쓰려면 러너를 "
            f'`uv run --with "krx-quant-core[portfolio,db,fast] @ git+file://<core 체크아웃>@<커밋>"` 으로 돌린다.'
        ) from e
