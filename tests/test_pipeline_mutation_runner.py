"""변이 시험 도구 자신의 시험 — 못 잡는 변이·돌지 않은 변이를 정말 실패로 내는가."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_SOURCE = "def f(x):\n    return x + 1\n"
_TEST = "import m\n\ndef test_f():\n    assert m.f(1) == 2\n\ndef test_blind():\n    assert True\n"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("check_pipeline_mutations", REPO / "scripts" / "check_pipeline_mutations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _project(tmp_path: Path, mutation: str) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "m.py").write_text(_SOURCE, encoding="utf-8")
    (tmp_path / "tests" / "test_m.py").write_text(_TEST, encoding="utf-8")
    spec = tmp_path / "muts.toml"
    spec.write_text(mutation, encoding="utf-8")
    return spec


_CAUGHT = '''
[[mutation]]
id = "off-by-one"
guards = ["tests/test_m.py::test_f"]
file = "src/m.py"
find = "x + 1"
replace = "x + 2"
'''


def test_caught_mutation_passes_and_source_is_restored(runner, tmp_path):
    spec = _project(tmp_path, _CAUGHT)
    assert runner.run_mutations(tmp_path, spec, say=lambda _: None) == 0
    assert (tmp_path / "src" / "m.py").read_text(encoding="utf-8") == _SOURCE


def test_blind_guard_is_a_failure(runner, tmp_path):
    spec = _project(tmp_path, _CAUGHT.replace("test_f", "test_blind"))
    assert runner.run_mutations(tmp_path, spec, say=lambda _: None) == 1
    assert (tmp_path / "src" / "m.py").read_text(encoding="utf-8") == _SOURCE


@pytest.mark.parametrize("old,new", [
    ('find = "x + 1"', 'find = "not there"'),          # 앵커 없음 — 돌지 않은 변이는 실패다
    ('find = "x + 1"', 'find = "x"'),                  # 앵커가 여러 번
    ('file = "src/m.py"', 'file = "src/gone.py"'),     # 파일 없음
    ('guards = ["tests/test_m.py::test_f"]\n', ""),    # 선언 오류
])
def test_unrun_mutation_is_a_failure_not_a_warning(runner, tmp_path, old, new):
    spec = _project(tmp_path, _CAUGHT.replace(old, new))
    assert runner.run_mutations(tmp_path, spec, say=lambda _: None) >= 1


def test_syntax_error_mutation_does_not_count_as_caught(runner, tmp_path):
    """수집 오류로 빨개진 것을 잡힌 것으로 치면 아무 변이나 통과한다."""
    spec = _project(tmp_path, _CAUGHT.replace('replace = "x + 2"', 'replace = "x +"'))
    assert runner.run_mutations(tmp_path, spec, say=lambda _: None) == 1


def test_empty_spec_is_a_failure(runner, tmp_path):
    assert runner.run_mutations(tmp_path, _project(tmp_path, ""), say=lambda _: None) == 1


def test_red_baseline_is_a_failure(runner, tmp_path):
    """늘 빨간 시험은 아무것도 증명하지 않는다."""
    spec = _project(tmp_path, _CAUGHT)
    (tmp_path / "tests" / "test_m.py").write_text("def test_f():\n    assert False\n", encoding="utf-8")
    assert runner.run_mutations(tmp_path, spec, say=lambda _: None) == 1


def test_every_declared_mutation_targets_a_real_anchor(runner):
    """앵커가 썩으면 CI 단계에서 실패하지만, 여기서 먼저 빠르게 본다."""
    import tomllib

    for m in tomllib.loads((REPO / runner.SPEC).read_text(encoding="utf-8"))["mutation"]:
        assert (REPO / m["file"]).read_text(encoding="utf-8").count(m["find"]) == 1, m["id"]
