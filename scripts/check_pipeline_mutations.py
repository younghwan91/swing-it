#!/usr/bin/env python
"""파이프라인 가드의 변이 시험 — 가드를 없앴을 때 선언한 시험이 정말 빨개지는가.

통과만 보는 시험은 규칙이 죽어도 초록이다(GUARDRAILS §4.5). ``tests/pipeline_mutations.toml`` 의 변이를 하나씩 소스에 넣고,
선언한 시험이 실패하는지 본다. 끝나면 원본 바이트로 되돌린다. 앵커가 없으면 경고가 아니라 실패다 — 돌지 않은 변이는 아무것도
증명하지 않는다.

    [[mutation]]
    id = "짧은-이름"
    why = "이 가드가 없으면 무슨 일이 생기는가"
    guards = ["tests/test_x.py::test_y"]
    file = "src/x.py"
    find = "정확히 한 번 나오는 문자열"
    replace = "바꿀 문자열"

실행: ``python scripts/check_pipeline_mutations.py`` (CI 단계, 약 1분). exit 0 = 전부 잡힘.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC = "tests/pipeline_mutations.toml"


def _pytest(root: Path, guards: list[str], python: str) -> int:
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *guards],
                          cwd=root, env=env, capture_output=True, text=True, timeout=600).returncode


def run_mutations(root: Path, spec: Path, *, python: str = sys.executable, say=print) -> int:
    """변이를 전부 돌리고 실패 수를 돌려준다. 선언 오류·돌지 않은 변이도 실패로 센다."""
    mutations = tomllib.loads(spec.read_text(encoding="utf-8")).get("mutation", [])
    if not mutations:
        say(f"{spec} 에 [[mutation]] 이 없다")
        return 1
    guards = sorted({g for m in mutations for g in m.get("guards", [])})
    if _pytest(root, guards, python) != 0:
        say("변이 전에 시험이 이미 빨갛다 — 늘 빨간 시험은 아무것도 증명하지 않는다")
        return 1
    bad = 0
    for m in mutations:
        mid = m.get("id", "?")
        if not m.get("guards") or not m.get("file") or "find" not in m or "replace" not in m:
            say(f"[{mid}] 선언 오류: guards·file·find·replace 가 필요하다")
            bad += 1
            continue
        target = root / m["file"]
        if not target.is_file():
            say(f"[{mid}] 파일이 없다: {m['file']}")
            bad += 1
            continue
        original = target.read_bytes()
        text = original.decode("utf-8")
        count = text.count(m["find"])
        if count != 1:
            say(f"[{mid}] 앵커가 {m['file']} 에서 {count}번 나온다 — 정확히 1번이어야 한다")
            bad += 1
            continue
        try:
            target.write_bytes(text.replace(m["find"], m["replace"]).encode("utf-8"))
            code = _pytest(root, m["guards"], python)
        finally:
            target.write_bytes(original)
        if code == 1:
            say(f"[{mid}] 잡힘")
        else:
            why = "시험이 초록이다 — 이 가드를 못 지킨다" if code == 0 else f"시험이 실패가 아니라 오류로 끝났다(pytest {code})"
            say(f"[{mid}] {why}: {m.get('why', '')}")
            bad += 1
    say(f"변이 {len(mutations)}개, 못 잡음·오류 {bad}개")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, default=REPO / SPEC)
    args = ap.parse_args()
    return 1 if run_mutations(REPO, args.spec) else 0


if __name__ == "__main__":
    sys.exit(main())
