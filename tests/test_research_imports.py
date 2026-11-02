"""research/·scripts/·examples/ 의 모든 모듈이 import 되는지.

이 스위트는 `src/swing_it` 라이브러리만 덮고 있어서, 라이브러리에서 심볼을 지워도
그걸 import 하던 리서치 러너는 아무도 안 깨뜨렸다 — ruff 는 크로스모듈 심볼을 풀지
않으므로 린트도 통과한다. 실제로 미너비니 제거 때 `fundamentals._yoy_vec` 이 이 경로로
사라져 pead_gate·pead_concentrated_gate·pead_refinement·prop_feasibility 4개가
동시에 깨졌고, 전체 테스트는 초록이었다.

각 모듈을 실제로 exec 해서 import 시점 오류만 잡는다(러너 본문은 `main()` 뒤에 있어
실행되지 않는다). DB·네트워크 불요.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DIRS = ("research", "scripts", "examples")
# 러너들은 서로를 top-level 이름으로 import 한다(from prop_swing_common import ...).
EXTRA_PATH = [str(REPO / "research" / "experiments"), str(REPO / "research" / "signals")]
# krx-quant-core portfolio-layer(PR #19, 미머지)를 git 참조로 설치한 환경에서만 import 되는 러너.
# dev 환경에는 그 모듈이 없다 — 그 모듈의 부재만 건너뛰고, 다른 import 오류는 그대로 실패시킨다.
NEEDS_PORTFOLIO_LAYER = {"pead_optimizer_c0.py", "pead_optimizer_po2.py", "pead_optimizer_po3.py",
                         "pead_optimizer_po4.py", "pead_optimizer_b.py", "pead_optimizer_b_report.py",
                         "pead_optimizer_b_feasibility.py"}
PORTFOLIO_LAYER_MODULES = ("krx_quant_core.data.risk_inputs", "krx_quant_core.risk", "krx_quant_core.portfolio")


def _modules() -> list[Path]:
    out: list[Path] = []
    for d in DIRS:
        out += sorted(p for p in (REPO / d).rglob("*.py") if "__pycache__" not in p.parts)
    return out


def _exec(path: Path):
    """모듈 하나를 exec 한다. 일반 import 와 같이 **exec 전에** sys.modules 에 올리고, 끝나면 내린다.

    안 올리면 `from __future__ import annotations` + `@dataclass` 가 `sys.modules[cls.__module__]` 를 None 으로 읽어
    AttributeError 로 죽는다 — pead_optimizer_c0·b 가 이 시험에서만(러너는 정상) 그렇게 실패했다.
    """
    spec = importlib.util.spec_from_file_location(f"_smoke_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass  # argparse 를 모듈 레벨에서 도는 스크립트
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_exec_handles_postponed_annotations_dataclass(tmp_path: Path):
    """회귀(주입): 위 실패 모양의 모듈이 `_exec` 로 import 된다."""
    f = tmp_path / "dc_mod.py"
    f.write_text("from __future__ import annotations\nfrom dataclasses import dataclass\n\n\n"
                 "@dataclass(frozen=True)\nclass A:\n    x: int = 1\n")
    assert _exec(f).A().x == 1


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(REPO)))
def test_module_imports(path: Path):
    added = [p for p in EXTRA_PATH if p not in sys.path]
    sys.path[:0] = added
    try:
        _exec(path)
    except ModuleNotFoundError as e:
        if path.name in NEEDS_PORTFOLIO_LAYER and (e.name or "").startswith(PORTFOLIO_LAYER_MODULES):
            pytest.skip(f"portfolio-layer 필요: {e.name}")
        pytest.fail(f"{path.relative_to(REPO)} import 실패: {type(e).__name__}: {e}")
    except (ImportError, AttributeError, NameError) as e:
        pytest.fail(f"{path.relative_to(REPO)} import 실패: {type(e).__name__}: {e}")
    finally:
        for p in added:
            sys.path.remove(p)
