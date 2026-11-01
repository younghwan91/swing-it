"""PO-4.1 — 신창 러너가 END 를 굽은 함수·상수를 쓰지 않는지(AST). DB·portfolio-layer 불요."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "research" / "experiments" / "pead_optimizer_po4.py"
# END(2023-12-28)나 구창 α 키("OB")를 안에 굽고 있는 함수 — 인자 판으로 바꿔 써야 한다.
BAKED = {("P2", "raw_closes_and_events"), ("P2", "yoy_signal_strict"), ("P2", "prep"), ("P2", "_ob_task"),
         ("P3", "solve_ob"), ("P3", "evaluate"), ("P2", "END"), ("P3", "END")}


def _tree():
    return ast.parse(SRC.read_text())


def test_no_end_constant_import():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pead_optimizer"):
            names = {a.name for a in node.names}
            assert not names & {"END", "START", "GRID_START", "GRID_LAST"}, f"{node.module} 에서 {names} import"


def test_no_baked_end_functions():
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert (node.value.id, node.attr) not in BAKED, f"{node.value.id}.{node.attr} 는 END·키를 굽고 있다"


def test_injected_violation_is_caught():
    bad = ast.parse("import pead_optimizer_po2 as P2\nP2.yoy_signal_strict(1, 2)\n")
    hits = [n for n in ast.walk(bad) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and (n.value.id, n.attr) in BAKED]
    assert hits
