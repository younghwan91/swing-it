"""가드레일 린트 자체의 회귀 테스트 — 린트를 검사하는 린트.

이 파일이 없어서 실제로 일이 났다. 규칙 (e)가 **파일 단위**로 구현돼 있어서, 파일
어딘가에 ``config=`` 가 한 번만 있으면 같은 파일의 나머지 게이트 호출이 전부 면제됐다.
그 결과 민감도 스윕 18~27칸이 다중검정 원장을 통째로 우회했고 Deflated Sharpe 의
haircut 이 0 이 됐는데도 CI 는 초록이었다. **막으려던 실패 모드를 린트가 못 보고 있었고,
린트를 검사하는 것이 아무것도 없어서 아무도 몰랐다.**

그래서 여기서는 "위반 없는 트리에서 통과한다"만 보지 않는다 — 각 규칙에 대해 **위반을
주입하면 실제로 실패하는지**를 확인한다. 통과만 보는 테스트는 규칙이 죽어도 초록이다.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_guardrails", REPO / "scripts" / "check_guardrails.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lint():
    return _load()


def test_repo_is_currently_clean(lint):
    """현재 트리는 모든 규칙을 통과한다."""
    assert lint.main() == 0


# --- (k) 파이프라인 순서 ------------------------------------------------------------
#
# 러너가 study·단계를 선언하고 데이터를 읽기 전에 게이트를 부르는가. 위반을 주입해 실제로 잡는지 본다.

_HEAD = 'from swing_it.pipeline import Stage, require_stage\nPIPELINE = ("x", Stage.BACKTEST)\n'
_GATE = "    require_stage(*PIPELINE)\n"
_MAIN = 'if __name__ == "__main__":\n    main()\n'

_RUNNER_OK = """
import argparse
from swing_it.pipeline import require_stage, Stage
from swing_it.storage import connect

PIPELINE = ("toy", Stage.BACKTEST)

def main() -> int:
    \"\"\"돈다.\"\"\"
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    args = ap.parse_args()
    require_stage(*PIPELINE)
    connect(args.db)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def _tree(lint, tmp_path, monkeypatch):
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path / "research" / "experiments")
    monkeypatch.setattr(lint, "LOGS", tmp_path / "research" / "logs")
    for d in (lint.EXPERIMENTS, tmp_path / "research" / "paper", lint.LOGS, tmp_path / "scripts"):
        d.mkdir(parents=True)


def _rule_k(lint, tmp_path, monkeypatch, source, *, where="research/experiments", name="new_runner.py"):
    _tree(lint, tmp_path, monkeypatch)
    (tmp_path / where).mkdir(parents=True, exist_ok=True)
    (tmp_path / where / name).write_text(source, encoding="utf-8")
    return lint.check_pipeline_declared()


@pytest.mark.parametrize("src", [
    "from krx_quant_core.execution.kiwoom_broker import KiwoomBroker\n",
    "import krx_quant_core.execution.orders as o\n",
    "from swing_it.live.brokers import make\n",
    "from swing_it.execution import runner\n",
    'API = "kt10000"\n',
    'def f(c):\n    return c.post({"api-id": "kt10003"})\n',
])
def test_order_path_needs_the_execution_gate(lint, tmp_path, monkeypatch, src):
    """주문 코드를 게이트 없이 둘 수 없다. 게이트가 있어도 부르는 쪽이 안 부르면 그만이다(fcs sw-live 가 그렇다)."""
    _tree(lint, tmp_path, monkeypatch)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "orders.py").write_text(src, encoding="utf-8")
    out = lint.check_order_path_gated()
    assert len(out) == 1 and "require_execution_approval" in out[0], out
    (tmp_path / "src" / "orders.py").write_text(
        src + "from swing_it.pipeline import require_execution_approval\n\ndef run(s):\n    require_execution_approval(s, amount_krw=1)\n", encoding="utf-8")
    assert lint.check_order_path_gated() == []


def test_order_rule_sees_through_import_aliases(lint, tmp_path, monkeypatch):
    """별칭을 안 풀면 ``send_order as so`` 가 주문이 아니고, ``require_execution_approval as ok`` 가 게이트가 아니다."""
    _tree(lint, tmp_path, monkeypatch)
    (tmp_path / "src").mkdir()
    orders = tmp_path / "src" / "orders.py"
    orders.write_text("from broker import send_order as so\n\ndef run():\n    so()\n", encoding="utf-8")
    assert len(lint.check_order_path_gated()) == 1
    orders.write_text("from broker import send_order as so\nfrom swing_it.pipeline import require_execution_approval as ok\n\n"
                      "def run(s):\n    ok(s, amount_krw=1)\n    so()\n", encoding="utf-8")
    assert lint.check_order_path_gated() == []


def test_order_rule_ignores_a_mention_in_a_docstring(lint, tmp_path, monkeypatch):
    _tree(lint, tmp_path, monkeypatch)
    (tmp_path / "scripts" / "x.py").write_text('"""kt10000 은 쓰지 않는다."""\n# kt10001\n', encoding="utf-8")
    assert lint.check_order_path_gated() == []


def test_rule_k_accepts_a_declared_runner(lint, tmp_path, monkeypatch):
    assert _rule_k(lint, tmp_path, monkeypatch, _RUNNER_OK) == []


@pytest.mark.parametrize("where", ["research/experiments", "research/paper"])
def test_rule_k_catches_a_new_runner_without_declaration(lint, tmp_path, monkeypatch, where):
    src = """
from swing_it.storage import connect

def main():
    connect("x")

if __name__ == "__main__":
    main()
"""
    out = _rule_k(lint, tmp_path, monkeypatch, src, where=where)
    assert len(out) == 1 and "[pipeline]" in out[0] and "PIPELINE =" in out[0], out


def test_rule_k_catches_a_file_with_no_entry_point_and_no_declaration(lint, tmp_path, monkeypatch):
    """진입점을 빼는 것으로 규칙을 피하지 못한다."""
    assert len(_rule_k(lint, tmp_path, monkeypatch, "x = 1\n")) == 1


# 호출로 본다 — 위치가 아니다.

@pytest.mark.parametrize("where", ["scripts", "research/signals", "research/combined/deep"])
@pytest.mark.parametrize("src", [
    "from swing_it.storage import read_prices\n\ndef f(con):\n    return read_prices(con)\n",
    "from swing_it.storage import read_prices as rp\n\ndef f(con):\n    return rp(con)\n",
    "import swing_it.storage as st\n\ndef f(con):\n    return st.read_earnings(con)\n",
    "import pandas as pd\n\ndef f(con):\n    return pd.read_sql_query('x', con)\n",
    "from swing_it.strategies.pead import staggered_backtest\n\ndef f():\n    return staggered_backtest(1, 2)\n",
    "from x import simulate\n\ndef f():\n    return simulate()\n",
    "from x import simulate as s\n\ndef f():\n    return s()\n",
    "from x import my_backtest_v2\n\ndef f():\n    return my_backtest_v2()\n",
], ids=["read_prices", "alias", "module_alias", "read_sql", "backtest", "simulate", "simulate_alias", "any_backtest"])
def test_rule_k_catches_data_calls_outside_the_runner_folders(lint, tmp_path, monkeypatch, where, src):
    out = _rule_k(lint, tmp_path, monkeypatch, src, where=where)
    assert len(out) == 1 and "선언이 없다" in out[0], out


def test_rule_k_requires_end_on_read_prices_in_a_declared_runner(lint, tmp_path, monkeypatch):
    """옛 러너는 면제 목록에 있고 ``end=None`` 으로 표 전체를 읽는다. 새 러너가 2026-09-14~ 오염을 읽지 않게 ``end=`` 를 요구한다."""
    bad = _RUNNER_OK.replace('    connect(args.db)\n', '    read_prices(connect(args.db))\n').replace(
        "from swing_it.storage import connect", "from swing_it.storage import connect, read_prices")
    assert bad != _RUNNER_OK
    out = _rule_k(lint, tmp_path, monkeypatch, bad)
    assert len(out) == 1 and "end=" in out[0], out
    good = bad.replace("read_prices(connect(args.db))", 'read_prices(connect(args.db), end="2023-12-28")')
    assert _rule_k(lint, tmp_path / "ok", monkeypatch, good) == []


def test_rule_k_ignores_files_that_touch_no_data(lint, tmp_path, monkeypatch):
    src = "import json\n\ndef f(x):\n    return json.dumps(x)\n"
    assert _rule_k(lint, tmp_path, monkeypatch, src, where="scripts") == []


def test_rule_k_accepts_a_library_declaration_outside_the_runner_folders(lint, tmp_path, monkeypatch):
    src = ('PIPELINE_LIBRARY = "신호 로더"\nfrom swing_it.storage import read_prices\n\n'
           "def load(con):\n    return read_prices(con)\n")
    assert _rule_k(lint, tmp_path, monkeypatch, src, where="research/signals") == []


def test_rule_k_checks_a_declaration_even_when_no_data_call_is_visible(lint, tmp_path, monkeypatch):
    """스스로 러너라고 선언했으면 게이트도 있어야 한다."""
    out = _rule_k(lint, tmp_path, monkeypatch, 'PIPELINE = ("x", 9)\n', where="scripts")
    assert len(out) == 1 and "진입점" in out[0], out


@pytest.mark.parametrize("old,new,why", [
    ("    require_stage(*PIPELINE)\n", "", "게이트 호출 없음"),
    ("    require_stage(*PIPELINE)\n    connect(args.db)\n", "    connect(args.db)\n    require_stage(*PIPELINE)\n",
     "데이터를 먼저 읽음"),
    ("    require_stage(*PIPELINE)\n", "    if args.db:\n        require_stage(*PIPELINE)\n", "조건문 안"),
    ("    require_stage(*PIPELINE)\n", "    try:\n        require_stage(*PIPELINE)\n    except Exception:\n        pass\n",
     "try 로 삼킴"),
    ("    require_stage(*PIPELINE)\n", '    require_stage("other", 1)\n', "선언과 다른 인자"),
    ("    require_stage(*PIPELINE)\n", '    require_stage(*PIPELINE, root="/tmp")\n', "원장 바꿔치기"),
    ('PIPELINE = ("toy", Stage.BACKTEST)\n', "", "선언 없음"),
    ('PIPELINE = ("toy", Stage.BACKTEST)\n', 'PIPELINE = ("toy", 12)\n', "없는 단계"),
    ('PIPELINE = ("toy", Stage.BACKTEST)\n', 'PIPELINE = ("toy", Stage.NOPE)\n', "없는 단계 이름"),
    ('PIPELINE = ("toy", Stage.BACKTEST)\n', 'PIPELINE = ("Bad/Name", 9)\n', "잘못된 study"),
    ('PIPELINE = ("toy", Stage.BACKTEST)\n', 'PIPELINE = make()\n', "리터럴 아님"),
    ('if __name__ == "__main__":\n    raise SystemExit(main())\n', "", "진입점 없음"),
], ids=lambda x: x if " " not in x and "\n" not in x else None)
def test_rule_k_catches_each_way_of_dodging_the_gate(lint, tmp_path, monkeypatch, old, new, why):
    assert old in _RUNNER_OK
    out = _rule_k(lint, tmp_path, monkeypatch, _RUNNER_OK.replace(old, new))
    assert len(out) == 1, f"{why}: {out}"


# 데이터가 게이트보다 먼저 도는 모양 — 리뷰 공격(M2)

@pytest.mark.parametrize("src", [
    _HEAD + "from swing_it.storage import read_prices, connect\nPRICES = read_prices(connect())\n"
    "def main():\n" + _GATE + _MAIN,
    _HEAD + "def main(df=load()):\n" + _GATE + _MAIN,
    _HEAD + "def main(*, df=read_prices(1)):\n" + _GATE + _MAIN,
    _HEAD + 'def main(df):\n' + _GATE + 'if __name__ == "__main__":\n    main(read_prices(connect()))\n',
    _HEAD + 'def main(df):\n' + _GATE + 'if __name__ == "__main__":\n    import sys\n    sys.exit(main(load()))\n',
    _HEAD + "def parse_args():\n    return read_prices(connect())\ndef main():\n    a = parse_args()\n" + _GATE + _MAIN,
    _HEAD + "def main():\n    print(read_prices(connect()))\n" + _GATE + _MAIN,
    _HEAD + 'def main():\n    ap.add_argument("--x", default=read_prices(c))\n' + _GATE + _MAIN,
    _HEAD + "class C:\n    PRICES = read_prices(1)\ndef main():\n" + _GATE + _MAIN,
    _HEAD + "def load():\n    return read_prices(1)\nCACHE = load()\ndef main():\n" + _GATE + _MAIN,
    _HEAD + "def deep():\n    return read_prices(1)\ndef load():\n    return deep()\nCACHE = load()\ndef main():\n" + _GATE + _MAIN,
    _HEAD + "try:\n    PRICES = read_prices(1)\nexcept Exception:\n    PRICES = None\ndef main():\n" + _GATE + _MAIN,
    _HEAD + "@cache(read_prices(1))\ndef main():\n" + _GATE + _MAIN,
    _HEAD + "import subprocess\ndef main():\n    subprocess.run(['x'])\n" + _GATE + _MAIN,
], ids=["module_level", "default_arg_unknown", "kw_default", "arg_of_main", "arg_of_main_in_exit", "user_parse_args",
        "print_read", "add_argument_default", "class_body", "local_at_import", "local_deep_at_import", "module_try",
        "decorator", "subprocess_run"])
def test_rule_k_catches_data_before_the_gate(lint, tmp_path, monkeypatch, src):
    assert len(_rule_k(lint, tmp_path, monkeypatch, src)) == 1, src


@pytest.mark.parametrize("src", [
    'PIPELINE_LIBRARY = "h"\nfrom swing_it.storage import read_prices\ndf = read_prices(connect())\n',
    'PIPELINE_LIBRARY = "h"\ndef main():\n    staggered_backtest()\nmain()\n',
    'PIPELINE_LIBRARY = "h"\ndef main():\n    pass\nif "__main__" == __name__:\n    main()\n',
    'PIPELINE_LIBRARY = "h"\nimport sys\ndef main():\n    pass\nif len(sys.argv) > 1:\n    main()\n',
    'PIPELINE_LIBRARY = "h"\ndef main():\n    pass\ntry:\n    if __name__ == "__main__":\n        main()\nexcept Exception:\n    pass\n',
    'PIPELINE_LIBRARY = "라이브러리인 척"\n\nif __name__ == "__main__":\n    print(1)\n',
    'PIPELINE_LIBRARY = "  "\n',
    "PIPELINE_LIBRARY = make()\n",
], ids=["toplevel_read", "unguarded_main", "inverted_guard", "argv_guard", "guard_in_try", "plain_guard", "blank", "not_literal"])
def test_rule_k_catches_a_false_library_declaration(lint, tmp_path, monkeypatch, src):
    assert len(_rule_k(lint, tmp_path, monkeypatch, src)) == 1, src


@pytest.mark.parametrize("src", [
    "def require_stage(*a):\n    pass\nPIPELINE = ('x', 9)\ndef main():\n" + _GATE + _MAIN,
    "from elsewhere import require_stage\nPIPELINE = ('x', 9)\ndef main():\n" + _GATE + _MAIN,
    "from swing_it.pipeline import require_stage, StageGateError\nPIPELINE = ('x', 9)\ndef main():\n" + _GATE
    + 'if __name__ == "__main__":\n    try:\n        main()\n    except StageGateError:\n        run_anyway()\n',
], ids=["local_fake", "imported_fake", "swallow_and_run"])
def test_rule_k_catches_a_fake_or_swallowed_gate(lint, tmp_path, monkeypatch, src):
    assert len(_rule_k(lint, tmp_path, monkeypatch, src)) == 1, src


@pytest.mark.parametrize("call", ["staggered_backtest as bt", "simulate", "backtest", "simulate as s", "prop_gate"])
def test_rule_k_catches_a_backtest_declared_as_an_early_stage(lint, tmp_path, monkeypatch, call):
    """1단계로 적으면 앞 단계가 없어 게이트가 비어 버린다. 별칭으로도 못 피한다."""
    local = call.split(" as ")[-1]
    src = (f"from swing_it.pipeline import require_stage\nfrom x import {call}\n"
           f'PIPELINE = ("toy", 1)\n\ndef main():\n    require_stage(*PIPELINE)\n    {local}()\n\n' + _MAIN)
    out = _rule_k(lint, tmp_path, monkeypatch, src)
    assert len(out) == 1 and "백테스트는 9단계" in out[0], out
    (lint.EXPERIMENTS / "new_runner.py").write_text(src.replace('("toy", 1)', '("toy", 9)'), encoding="utf-8")
    assert lint.check_pipeline_declared() == []


# 멀쩡한 러너를 막지 않는다 — 거짓 양성이 많으면 규칙이 안 살아남는다

@pytest.mark.parametrize("src", [
    _HEAD + "import time\ndef main():\n    t0 = time.time()\n" + _GATE + _MAIN,
    _HEAD + "import logging\ndef main():\n    logging.basicConfig(level=20)\n" + _GATE + _MAIN,
    _HEAD + "def main():\n    args = build_parser().parse_args()\n" + _GATE + _MAIN,
    _HEAD + 'import argparse, datetime\ndef main():\n    ap = argparse.ArgumentParser()\n'
    '    ap.add_argument("--end", default=str(datetime.date.today()))\n    args = ap.parse_args()\n' + _GATE + _MAIN,
    _HEAD + 'import argparse, pathlib\ndef main():\n    ap = argparse.ArgumentParser()\n'
    '    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("x"))\n    args = ap.parse_args()\n' + _GATE + _MAIN,
    _HEAD + 'def run():\n' + _GATE + 'if __name__ == "__main__":\n    run()\n',
    _HEAD + 'def _main():\n' + _GATE + 'if __name__ == "__main__":\n    _main()\n',
    _HEAD + 'def main():\n' + _GATE + 'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    _HEAD + 'def main(a):\n' + _GATE + 'if __name__ == "__main__":\n    main(parse_args())\n',
    _HEAD + 'def main(a):\n' + _GATE + 'if __name__ == "__main__":\n    a = parse_args()\n    main(a)\n',
    _HEAD + 'def main(a):\n' + _GATE + 'if __name__ == "__main__":\n    a = my_parse()\n    main(a)\n',
    _HEAD + 'import asyncio\nasync def main():\n' + _GATE + 'if __name__ == "__main__":\n    asyncio.run(main())\n',
    _HEAD + 'import fire\ndef main():\n' + _GATE + 'if __name__ == "__main__":\n    fire.Fire(main)\n',
    _HEAD + 'def main():\n    """doc"""\n' + _GATE + _MAIN,
    'from swing_it.pipeline import require_stage\nPIPELINE = ("x", 9)\ndef main():\n'
    "    require_stage(PIPELINE[0], PIPELINE[1])\n" + _MAIN,
    _HEAD + "def main():\n    st = require_stage(*PIPELINE)\n" + _MAIN,
    'import swing_it.pipeline as sp\nPIPELINE = ("x", 9)\ndef main():\n    sp.require_stage(*PIPELINE)\n' + _MAIN,
    'from swing_it.pipeline import gate\nPIPELINE = ("x", 9)\ndef main():\n    gate.require_stage(*PIPELINE)\n' + _MAIN,
    'from swing_it.pipeline import require_stage\nSTAGE = 9\nPIPELINE = ("x", STAGE)\ndef main():\n' + _GATE + _MAIN,
    _HEAD + "def helper(x):\n    return x + 1\ndef main():\n    y = helper(1)\n" + _GATE + _MAIN,
    _HEAD + "def main():\n    def log(m):\n        print(m)\n" + _GATE + _MAIN,
    _HEAD + "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Flags:\n    a: int = 1\n"
    "def main(flags=Flags()):\n" + _GATE + _MAIN,
    _HEAD + "import re\nPAT = re.compile('x')\ndef main():\n" + _GATE + _MAIN,
], ids=["time", "logging", "build_parser", "default_today", "default_path", "entry_run", "entry_private", "system_exit",
        "main_of_parse", "args_then_main", "custom_parse", "async_main", "fire", "docstring", "index_args", "keep_result",
        "namespace", "gate_module", "stage_constant", "clean_helper", "nested_def", "dataclass_default", "module_regex"])
def test_rule_k_does_not_flag_a_normal_runner(lint, tmp_path, monkeypatch, src):
    assert _rule_k(lint, tmp_path, monkeypatch, src) == [], src


# docs/PIPELINE.md "린트가 못 잡는 것" 과 같은 목록이다. 이 시험이 빨개지면 문서도 고친다.

@pytest.mark.parametrize("src", [
    _HEAD + "import m\ndef main():\n    df = m.CACHE\n    n = df.shape\n" + _GATE + _MAIN,
    _HEAD + "from m import parse_args\ndef main():\n    a = parse_args()\n" + _GATE + _MAIN,
    _HEAD + 'import m\nFN = getattr(m, "read_" + "prices")\ndef main():\n' + _GATE + _MAIN,
    _HEAD + "import m\nCACHE = m.load_everything()\ndef main():\n" + _GATE + _MAIN,
    'from swing_it.pipeline import require_stage\nPIPELINE = ("x", 1)\ndef main():\n' + _GATE
    + "    my_engine()\n" + _MAIN,
], ids=["attribute_access", "imported_parse_args", "getattr", "unknown_call_at_import", "wrong_stage_without_backtest"])
def test_rule_k_known_limits_are_the_documented_ones(lint, tmp_path, monkeypatch, src):
    assert _rule_k(lint, tmp_path, monkeypatch, src) == [], src


def test_rule_k_reports_unparsable_files(lint, tmp_path, monkeypatch):
    assert "파싱 실패" in _rule_k(lint, tmp_path, monkeypatch, "def (:\n")[0]


def test_rule_k_skips_init_and_grandfathered_files_only(lint, tmp_path, monkeypatch):
    _tree(lint, tmp_path, monkeypatch)
    (lint.EXPERIMENTS / "__init__.py").write_text("", encoding="utf-8")
    (lint.EXPERIMENTS / "pead_gate.py").write_text("x = 1\n", encoding="utf-8")        # 면제
    (lint.EXPERIMENTS / "pead_gate2.py").write_text("x = 1\n", encoding="utf-8")       # 비슷한 이름은 면제 아님
    (tmp_path / "research" / "paper" / "pead_paper.py").write_text("x = 1\n", encoding="utf-8")   # fcs 면제
    (tmp_path / "research" / "paper" / "new_arm.py").write_text("x = 1\n", encoding="utf-8")      # 새 팔은 면제 아님
    read = "from s import read_prices\nread_prices\ndef f():\n    return read_prices(1)\n"
    (tmp_path / "scripts" / "compare_price_adjust.py").write_text(read, encoding="utf-8")         # 면제
    (tmp_path / "scripts" / "new_tool.py").write_text(read, encoding="utf-8")                     # 면제 아님
    out = lint.check_pipeline_declared()
    assert [o.split(" — ")[0] for o in out] == [
        "[pipeline] research/experiments/pead_gate2.py", "[pipeline] research/paper/new_arm.py",
        "[pipeline] scripts/new_tool.py"]


def test_grandfather_lists_never_grow(lint):
    """면제 목록은 기준 브랜치 목록의 부분집합이다 — 줄어들 수는 있어도 늘어날 수 없다. 사본을 두지 않고 기준의 린트를 읽는다."""
    base = os.environ.get("PIPELINE_BASE_REF", "origin/main")
    shown = subprocess.run(("git", "show", f"{base}:scripts/check_guardrails.py"), cwd=REPO, capture_output=True, text=True, timeout=30)
    if shown.returncode != 0:
        if os.environ.get("CI"):
            pytest.fail(f"기준 {base} 가 없다 — CI 는 fetch-depth: 0 으로 받는다")
        pytest.skip(f"기준 {base} 가 없다")
    tree = ast.parse(shown.stdout)
    for name in ("PIPELINE_GRANDFATHER", "PIPELINE_GRANDFATHER_ELSEWHERE", "PIPELINE_LOGS_GRANDFATHER"):
        allowed = ast.literal_eval(lint._module_constant(tree, name).args[0])
        assert getattr(lint, name) <= allowed, (name, getattr(lint, name) - allowed)


def test_finished_ml_runners_are_grandfathered_but_their_log_folder_is_not(lint):
    """ML 러너는 게이트 전에 돌고 끝났다 — 끝난 러너는 고치지 않는다. 그래도 원장은 있어야 한다."""
    names = lint.PIPELINE_GRANDFATHER_ELSEWHERE
    assert {f"research/experiments/ml_forecast_{x}.py" for x in ("book", "ic", "stage0", "s1_diag")} <= names
    assert "ml_forecast" not in lint.PIPELINE_LOGS_GRANDFATHER


def test_new_ml_style_runner_is_not_grandfathered(lint, tmp_path, monkeypatch):
    """면제는 파일 이름 단위다 — ml_forecast_ 접두사로 새 파일을 끼워 넣지 못한다."""
    out = _rule_k(lint, tmp_path, monkeypatch, "x = 1\n", name="ml_forecast_new.py")
    assert len(out) == 1


def test_grandfather_entries_on_main_exist(lint):
    """지운 파일의 면제를 남겨 두면 같은 이름으로 새 러너를 면제 없이 넣을 수 있다."""
    assert [n for n in lint.PIPELINE_GRANDFATHER if not (lint.REPO / n).is_file()] == []


def test_grandfathered_logs_folders_exist_or_live_on_fcs(lint):
    fcs_only = {"live_swing", "sepa_confluence", "sepa_explore"}
    assert [n for n in lint.PIPELINE_LOGS_GRANDFATHER - fcs_only if not (lint.LOGS / n).is_dir()] == []


def test_new_logs_folder_requires_a_ledger(lint, tmp_path, monkeypatch):
    """연구 이름 세 개를 박아 두는 것이 아니다 — 면제 없는 새 폴더는 전부 원장이 있어야 한다."""
    _tree(lint, tmp_path, monkeypatch)
    assert lint.check_pipeline_ledgers() == []
    (lint.LOGS / "pead_concentrated").mkdir()           # 면제
    assert lint.check_pipeline_ledgers() == []
    for name in ("ml_forecast", "brand_new_study"):
        (lint.LOGS / name).mkdir()
    out = lint.check_pipeline_ledgers()
    assert len(out) == 2 and "brand_new_study" in out[0] and "ml_forecast" in out[1]


def _ledger_with_one_pass(lint):
    gate = lint._load_gate()
    ev = lint.REPO / "e.md"
    ev.write_text("증거\n", encoding="utf-8")
    gate.record_stage("toy", 1, "pass", evidence=["e.md"], reason="왜", root=lint.REPO)
    return gate, ev


def test_rule_k_catches_tampered_evidence(lint, tmp_path, monkeypatch):
    _tree(lint, tmp_path, monkeypatch)
    _, ev = _ledger_with_one_pass(lint)
    assert lint.check_pipeline_ledgers() == []
    ev.write_text("고쳐 쓴 증거\n", encoding="utf-8")
    out = lint.check_pipeline_ledgers()
    assert any("해시" in x for x in out), out


def test_rule_k_catches_a_hand_edited_ledger(lint, tmp_path, monkeypatch):
    _tree(lint, tmp_path, monkeypatch)
    gate, _ = _ledger_with_one_pass(lint)
    p = gate.ledger_path("toy", root=lint.REPO)
    p.write_text(p.read_text(encoding="utf-8").replace('"stage": 1', '"stage": 2'), encoding="utf-8")
    assert any("체인" in x for x in lint.check_pipeline_ledgers())


# 원장 잘라내기 — 체인이 못 잡는 끝 줄을 기준 브랜치와 비교해 잡는다(M3)

def _base_repo(lint, tmp_path, monkeypatch):
    """기준 커밋에 원장 두 줄을 넣고 refs/remotes/origin/main 으로 둔다."""
    import subprocess

    _tree(lint, tmp_path, monkeypatch)
    monkeypatch.delenv("PIPELINE_BASE_REF", raising=False)

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True,
                       env={**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1",
                            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"})

    git("init", "-q")
    git("config", "user.email", "t@example.com")       # 픽스처 레포 안에서만 — 린트의 커밋 신원 규칙이 이 레포를 본다
    gate = lint._load_gate()
    ev = tmp_path / "e.md"
    ev.write_text("증거\n", encoding="utf-8")
    gate.record_stage("toy", 1, "pass", evidence=["e.md"], reason="왜", root=tmp_path)
    gate.record_stage("toy", 2, "fail", reason="데이터 문제", root=tmp_path)
    git("add", "-A")
    git("commit", "-qm", "base", "--no-verify")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    return gate, gate.ledger_path("toy", root=tmp_path)


def test_untouched_or_appended_ledger_passes(lint, tmp_path, monkeypatch):
    gate, path = _base_repo(lint, tmp_path, monkeypatch)
    assert lint.check_pipeline_ledgers_append_only() == []
    gate.record_stage("toy", 2, "pass", evidence=["e.md"], reason="고침", root=tmp_path)
    assert lint.check_pipeline_ledgers_append_only() == []


def test_truncated_ledger_is_caught(lint, tmp_path, monkeypatch):
    """끝 줄 fail 을 지우면 체인은 멀쩡하고 게이트가 열린다 — 그래서 기준과 비교한다."""
    gate, path = _base_repo(lint, tmp_path, monkeypatch)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first + "\n", encoding="utf-8")
    assert gate.verify("toy", root=tmp_path) == [], "체인은 못 잡는다는 전제"
    out = lint.check_pipeline_ledgers_append_only()
    assert len(out) == 1 and "덧붙이기만" in out[0], out


def test_deleted_ledger_is_caught(lint, tmp_path, monkeypatch):
    _, path = _base_repo(lint, tmp_path, monkeypatch)
    path.unlink()
    out = lint.check_pipeline_ledgers_append_only()
    assert len(out) == 1 and "사라졌다" in out[0], out


def test_rewritten_then_appended_ledger_is_caught(lint, tmp_path, monkeypatch):
    """두 브랜치 머지에서 --theirs 로 푼 경우 — 줄 수가 늘어도 기준 줄이 그대로 앞에 없으면 잡는다."""
    gate, path = _base_repo(lint, tmp_path, monkeypatch)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first + "\n", encoding="utf-8")
    gate.record_stage("toy", 2, "pass", evidence=["e.md"], reason="다른 브랜치", root=tmp_path)
    assert len(lint.check_pipeline_ledgers_append_only()) == 1


def test_missing_base_fails_in_ci_but_not_locally(lint, tmp_path, monkeypatch):
    _base_repo(lint, tmp_path, monkeypatch)
    monkeypatch.setenv("PIPELINE_BASE_REF", "refs/remotes/origin/nope")
    monkeypatch.delenv("CI", raising=False)
    assert lint.check_pipeline_ledgers_append_only() == []
    monkeypatch.setenv("CI", "true")
    out = lint.check_pipeline_ledgers_append_only()
    assert len(out) == 1 and "fetch-depth" in out[0], out


def test_non_git_tree_is_not_checked(lint, tmp_path, monkeypatch):
    _tree(lint, tmp_path, monkeypatch)
    monkeypatch.setenv("CI", "true")
    assert lint.check_pipeline_ledgers_append_only() == []


@pytest.mark.parametrize("inject", ["runner", "evidence", "truncate"])
def test_main_is_wired_to_the_pipeline_rules(lint, tmp_path, monkeypatch, capsys, inject):
    """규칙 함수가 살아 있어도 main() 에서 안 부르면 CI 는 초록이다 — 끝까지 본다."""
    if inject == "truncate":
        _, path = _base_repo(lint, tmp_path, monkeypatch)
    else:
        _tree(lint, tmp_path, monkeypatch)
    monkeypatch.setattr(lint, "SRC", tmp_path / "src")
    (tmp_path / "src").mkdir()
    assert lint.main() == 0
    if inject == "runner":
        (lint.EXPERIMENTS / "new_runner.py").write_text("x = 1\n", encoding="utf-8")
    elif inject == "evidence":
        _, ev = _ledger_with_one_pass(lint)
        ev.write_text("고쳐 쓴 증거\n", encoding="utf-8")
    else:
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert lint.main() == 1
    assert "[pipeline]" in capsys.readouterr().out


def test_ci_runs_the_guardrail_lint_on_pull_requests():
    """이 린트가 CI 단계라는 전제가 깨지면 위 규칙은 전부 부탁이 된다."""
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pull_request" in ci
    assert "run: python scripts/check_guardrails.py" in ci


def _ci_jobs(ci: str, *names: str) -> list[str]:
    """ci.yml 을 잡 단위로 자른다(YAML 파서 없이 — 린트 시험은 표준 라이브러리만 쓴다)."""
    out = []
    for name in names:
        start = ci.index(f"\n  {name}:\n")
        following = re.search(r"\n  [a-z-]+:\n", ci[start + 1:])
        out.append(ci[start:start + 1 + following.start()] if following else ci[start:])
    return out


def test_ci_checks_out_the_base_for_the_ledger_check():
    """얕은 체크아웃에는 origin/main 이 없다. 없으면 잘라내기 검사가 CI 에서 실패한다."""
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for job in _ci_jobs(ci, "test", "pipeline-mutations"):
        assert "fetch-depth: 0" in job, job


def test_ci_has_no_shallow_git_command_in_a_run_step():
    """pre-push 의 ci-local 이 run: 을 공유 .git 에서 돌린다. --depth·--shallow 를 넣으면 이 머신의 저장소가 shallow 가 된다(2026-11-02 사고)."""
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert not re.search(r"git\s+(fetch|pull|clone)\b[^\n]*--(depth|shallow)", ci)


def test_ci_runs_the_pipeline_mutations():
    """verify_merge·mutations.toml 이 없어져도(PR #18) 가드 변이 시험은 CI 에서 돈다."""
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "run: python scripts/check_pipeline_mutations.py\n" in ci, "기본 핀 변이"
    assert "run: python scripts/check_pipeline_mutations.py --spec tests/pipeline_mutations_core.toml\n" in ci, "core 를 까는 잡의 변이"


# --- (e) 호출 지점 단위 원장 규칙 ------------------------------------------------
#
# 아래 세 케이스가 이 규칙의 핵심이다. 특히 `two_calls_one_config` 는 예전 파일 단위
# 구현이 놓쳤던 바로 그 모양이다.

_TWO_CALLS_ONE_CONFIG = '''
from prop_gate import prop_gate

def preregistered(a, b):
    return prop_gate(a, b, 0.04, label="x", config={"stop": 0.04}, log_dir="d")

def sweep(a, b):
    rows = []
    for s in (0.03, 0.04, 0.05):
        rows.append(prop_gate(a, b, s, label=f"sens_{s}", verbose=False))
    return rows
'''

_MULTILINE_CALL_NO_CONFIG = '''
from prop_swing_common import gate_sim

def sweep(a, b):
    return gate_sim(
        a, b, 0.04,
        "sens",
        log_dir="d",
    )
'''

_MULTILINE_CALL_WITH_CONFIG = '''
from prop_swing_common import gate_sim

def sweep(a, b):
    return gate_sim(
        a, b, 0.04,
        "sens",
        config={"stop": 0.04},
        log_dir="d",
    )
'''


def _run_rule_e(lint, tmp_path, monkeypatch, source: str) -> list[str]:
    gate = tmp_path / "fake_gate.py"
    gate.write_text(source, encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    return lint.check_gates_record_trials()


def test_rule_e_catches_second_call_without_config(lint, tmp_path, monkeypatch):
    """파일에 config= 가 이미 한 번 있어도, config 없는 다른 호출은 잡아야 한다.

    예전 파일 단위 구현이 통과시키던 정확한 모양이다 — 이 테스트가 그 회귀를 막는다.
    """
    out = _run_rule_e(lint, tmp_path, monkeypatch, _TWO_CALLS_ONE_CONFIG)
    assert len(out) == 1, out
    assert "prop_gate()" in out[0]
    assert ":10" in out[0]      # 사전등록 호출(5행)이 아니라 스윕 호출을 지목해야 한다


def test_rule_e_catches_multiline_call(lint, tmp_path, monkeypatch):
    """여러 줄로 쪼갠 호출도 잡아야 한다(정규식 구현이 못 보던 모양)."""
    out = _run_rule_e(lint, tmp_path, monkeypatch, _MULTILINE_CALL_NO_CONFIG)
    assert len(out) == 1, out
    assert "gate_sim()" in out[0]


def test_rule_e_accepts_multiline_call_with_config(lint, tmp_path, monkeypatch):
    """config= 를 제대로 넘긴 여러 줄 호출은 통과 — 거짓 양성이 없어야 규칙이 살아남는다."""
    assert _run_rule_e(lint, tmp_path, monkeypatch, _MULTILINE_CALL_WITH_CONFIG) == []


def test_rule_e_ignores_the_harness_definition(lint, tmp_path, monkeypatch):
    """하버스 자신의 def 안에서의 호출은 대상이 아니다(자기 자신을 고발하면 규칙이 죽는다)."""
    src = '''
def gate_sim(a, b, stop, label, *, config, log_dir):
    return prop_gate(a, b, stop, label=label, config=config, log_dir=log_dir)
'''
    assert _run_rule_e(lint, tmp_path, monkeypatch, src) == []


# --- (a) src → research 경계 ----------------------------------------------------

def test_rule_a_catches_research_import(lint, tmp_path, monkeypatch):
    (tmp_path / "leaky.py").write_text(
        "from research.signals import contrarian_retail\n", encoding="utf-8"
    )
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "SRC", tmp_path)
    out = lint.check_src_no_research_import()
    assert len(out) == 1 and "boundary" in out[0]


# --- (b) 게이트 → VERDICT -------------------------------------------------------

def test_rule_b_catches_gate_without_verdict(lint, tmp_path, monkeypatch):
    (tmp_path / "orphan_gate.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(lint, "LOGS", tmp_path / "logs")
    out = lint.check_gates_have_verdict()
    assert len(out) == 1 and "verdict" in out[0]


# --- (c) 하드코딩 판정 ----------------------------------------------------------

def test_rule_c_catches_literal_verdict(lint, tmp_path, monkeypatch):
    (tmp_path / "judgey.py").write_text('verdict = "PASS"\n', encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    out = lint.check_no_literal_verdict()
    assert len(out) == 1 and "reporter" in out[0]


def test_rule_c_ignores_prose_in_comments(lint, tmp_path, monkeypatch):
    (tmp_path / "prose.py").write_text(
        '# PASS/FAIL bool 도 두지 않는다\nx = 1\n', encoding="utf-8"
    )
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    assert lint.check_no_literal_verdict() == []


# --- (g2) 생존자 전용 조인 -------------------------------------------------------

_DELETED_LOADER_SHAPE = '''
QUERY = """
    SELECT sd.code, sd.date, sd.close, s.name, s.market
    FROM supply_demand sd
    JOIN stocks s ON s.code = sd.code
    WHERE sd.date >= ?
"""
'''


def test_rule_g2_catches_multiline_survivor_join(lint, tmp_path, monkeypatch):
    """2026-08-16 에 삭제한 로더 3개의 실제 모양 — 여러 줄 SQL 이라 한 줄 정규식은 못 봤다."""
    (tmp_path / "loader.py").write_text(_DELETED_LOADER_SHAPE, encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    out = lint.check_no_survivor_only_join()
    assert len(out) == 1, out
    assert "universe" in out[0] and "delisted_stocks" in out[0]


def test_rule_g2_allows_supply_demand_without_the_join(lint, tmp_path, monkeypatch):
    """폐지분을 배제하지 않는 supply_demand 읽기는 통과해야 한다(거짓 양성 방지)."""
    (tmp_path / "ok.py").write_text(
        'Q = "SELECT code, date, individual FROM supply_demand WHERE date >= ?"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(lint, "REPO", tmp_path)
    assert lint.check_no_survivor_only_join() == []


def test_rule_i_flags_a_foreign_commit_identity(tmp_path, monkeypatch):
    """(i) 커밋 신원이 레포 설정과 다르면 잡는가 — 위반을 실제로 주입해서 본다.

    통과만 보는 테스트는 규칙이 죽어도 초록이다(GUARDRAILS §5 의 교훈).
    """
    import subprocess

    import scripts.check_guardrails as lint

    repo = tmp_path / "r"
    repo.mkdir()

    def git(*a, **kw):
        return subprocess.run(("git", *a), cwd=repo, capture_output=True,
                              text=True, check=False, **kw)

    git("init", "-q")
    git("config", "core.hooksPath", "")  # 전역 커밋 신원 훅이 이 임시 레포까지 잡아 위반 주입용 커밋을 막는 걸 방지
    git("config", "user.name", "T")
    git("config", "user.email", "me@example.com")
    (repo / "f.txt").write_text("x", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "ok")
    monkeypatch.setattr(lint, "REPO", repo)
    assert lint.check_commit_identity() == [], "정상 커밋인데 잡혔다"

    # 위반 주입 — 다른 이메일로 커밋
    (repo / "f.txt").write_text("y", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=work@corp.com", "commit", "-q", "-m", "bad")
    out = lint.check_commit_identity()
    assert out and "identity" in out[0], f"위반을 못 잡았다: {out}"
    assert "work@corp.com" in out[0]


def test_scan_skips_nested_worktrees_but_still_catches_real_violations(lint, tmp_path):
    """회귀 — `.claude/worktrees` 는 에이전트용 **레포 복사본**이다.

    스캔에 넣으면 같은 위반이 두 번 세어지고, 남이 작업 중인 코드로 CI 가 깨진다.
    반대로 건너뛰기가 넓으면 진짜 위반까지 조용히 놓친다 — 둘 다 검사한다.
    """
    # 린트 자신이 이 파일도 스캔하므로 위반 문자열은 쪼개서 만든다.
    bad = 'q = "SELECT * FROM ' + "daily_bars" + '_adjusted WHERE code = 1"\n'
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "real.py").write_text(bad, encoding="utf-8")
    (tmp_path / ".claude" / "worktrees" / "a" / "pkg").mkdir(parents=True)
    (tmp_path / ".claude" / "worktrees" / "a" / "pkg" / "copy.py").write_text(
        bad, encoding="utf-8")

    found = [p.name for p in lint._iter_py(tmp_path)]
    assert "real.py" in found, "진짜 위반 파일을 스캔에서 빠뜨렸다"
    assert "copy.py" not in found, "워크트리 복사본까지 스캔했다"


def test_scan_is_not_silenced_by_an_ancestor_directory_name(lint, tmp_path):
    """회귀 — 건너뛰기를 **절대경로**로 판단하면, 레포가 `.claude/...` 아래
    놓였을 때 스캔이 통째로 비어 린트가 항상 통과한다."""
    root = tmp_path / ".claude" / "checkout"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "real.py").write_text("x = 1\n", encoding="utf-8")
    assert [p.name for p in lint._iter_py(root)] == ["real.py"]


def test_identity_allows_github_merge_commits_but_not_other_addresses(lint, tmp_path,
                                                                      monkeypatch):
    """회귀 — PR 을 웹에서 머지하면 GitHub 이 신원을 자기 주소로 찍는다.

    author 는 계정 프라이버시 주소(`…@users.noreply.github.com`), committer 는
    `noreply@github.com` 이다. 이걸 안 봐주면 **머지할 때마다 CI 가 빨개진다.**

    반대로 넓게 열면 규칙이 사라진다 — 이 규칙이 막으려는 것은 의도치 않은
    주소가 공개 이력에 박히는 것이다. 둘 다 검사한다.
    """
    import os
    import subprocess

    repo = tmp_path / "r"
    repo.mkdir()

    def git(*a, env_extra=None):
        env = {**os.environ, **(env_extra or {})}
        return subprocess.run(("git", *a), cwd=repo, capture_output=True,
                              text=True, check=False, env=env)

    git("init", "-q")
    git("config", "core.hooksPath", "")  # 전역 커밋 신원 훅이 이 임시 레포까지 잡아 위반 주입용 커밋을 막는 걸 방지
    git("config", "user.name", "T")
    git("config", "user.email", "me@example.com")
    (repo / "f.txt").write_text("x", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "ok")
    monkeypatch.setattr(lint, "REPO", repo)
    assert lint.check_commit_identity() == []

    # GitHub 웹 머지가 찍는 신원 — 통과해야 한다.
    (repo / "f.txt").write_text("y", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "Merge pull request #5",
        env_extra={"GIT_AUTHOR_EMAIL": "48156556+someone@users.noreply.github.com",
                   "GIT_COMMITTER_EMAIL": "noreply@github.com"})
    assert lint.check_commit_identity() == [], "GitHub 머지 커밋이 잡혔다"

    # 사내 메일은 **여전히** 잡혀야 한다 — 봐주기가 넓어지지 않았는가.
    (repo / "f.txt").write_text("z", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "bad",
        env_extra={"GIT_AUTHOR_EMAIL": "someone@corp.example.com",
                   "GIT_COMMITTER_EMAIL": "someone@corp.example.com"})
    out = lint.check_commit_identity()
    assert out and "corp.example.com" in out[0], f"사내 메일을 못 잡는다: {out}"


# (j) ML 예측 피처 원천 ------------------------------------------------------------


def _run_rule_j(lint, tmp_path, monkeypatch, name, source):
    monkeypatch.setattr(lint, "REPO", tmp_path)
    monkeypatch.setattr(lint, "EXPERIMENTS", tmp_path)
    (tmp_path / name).write_text(source, encoding="utf-8")
    return lint.check_ml_forecast_sources()


@pytest.mark.parametrize("src", [
    'cur.execute("SELECT code FROM credit_balance")\n',
    'cur.execute("SELECT * FROM short_selling")\n',
    'cur.execute("SELECT * FROM program_trading")\n',
    'cur.execute("SELECT abs(close) FROM supply_demand")\n',
    "from swing_it.storage import read_supply_demand\n",
    "from swing_it.storage import INSTITUTION_DETAIL_COLS\n",
    'cols = ("code", "invtrt")\n',
])
def test_rule_j_catches_forbidden_sources(lint, tmp_path, monkeypatch, src):
    out = _run_rule_j(lint, tmp_path, monkeypatch, "ml_forecast_x.py", src)
    assert len(out) == 1 and "[ml-source]" in out[0]


def test_rule_j_scope_and_comments(lint, tmp_path, monkeypatch):
    # 다른 러너는 대상이 아니고, 주석 줄은 세지 않는다
    assert _run_rule_j(lint, tmp_path, monkeypatch, "pead_x.py", 'q = "FROM supply_demand"\n') == []
    assert _run_rule_j(lint, tmp_path, monkeypatch, "ml_forecast_y.py", "# supply_demand 는 쓰지 않는다\n") == []
    assert _run_rule_j(lint, tmp_path, monkeypatch, "ml_forecast_z.py", 'read_investor_trend(con)\n') == []
