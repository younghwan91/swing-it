"""새 연구를 정말 이렇게 쓸 수 있는가 — ``examples/pipeline_rank_book.py`` 를 합성 데이터로 끝까지 돌린다.

선언 → 게이트 → 단계별 ``src`` 함수. 비용 경로는 core 충격 레이어가 있어야 해 ``research`` CI 잡에서 돈다(거기서는 건너뛰면 실패다).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swing_it.pipeline import Stage, StageGateError, gate, record_stage

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "examples" / "pipeline_rank_book.py"
STUDY = "example_rank_book"


@pytest.fixture
def example():
    spec = importlib.util.spec_from_file_location("pipeline_rank_book", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _open_ledger(root: Path) -> None:
    """이 북의 정직한 원장 — 5·6·7 은 일부러 안 쓰고, 4 는 합성이라 검증을 건너뛴 이유를 파일로 남긴다."""
    (root / "e.md").write_text("증거\n", encoding="utf-8")
    for s in (Stage.UNIVERSE, Stage.DATA, Stage.ALPHA):
        record_stage(STUDY, s, "pass", evidence=["e.md"], reason="합성 데이터 시연", root=root)
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "skip", evidence=["e.md"], reason="합성 데이터 — 검증할 신호가 아니다", root=root)
    for s in (Stage.FORECAST, Stage.RISK_MODEL, Stage.PORTFOLIO):
        record_stage(STUDY, s, "skip", reason="순위 북 — 안 쓴다", root=root)
    record_stage(STUDY, Stage.COSTS, "pass", evidence=["e.md"], reason="정가 비용", root=root)


def test_example_satisfies_the_lint():
    """시연이 규칙을 어기면 시연이 아니다. examples/ 는 규칙 밖이라 여기서 직접 묻는다."""
    spec = importlib.util.spec_from_file_location("check_guardrails", REPO / "scripts" / "check_guardrails.py")
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    assert lint._pipeline_problem(tree, lint._load_gate(), lint._Names(tree), lint._call_nodes(tree)) is None


def test_example_stops_at_the_gate_before_touching_data(example, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(example, "synthetic_inputs", lambda: pytest.fail("게이트 전에 데이터를 읽었다"))
    with pytest.raises(StageGateError, match="pending"):
        example.main(["--no-costs"])


def test_example_runs_on_the_default_pin_once_the_ledger_is_open(example, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    _open_ledger(tmp_path)
    assert example.main(["--no-costs"]) == 0
    assert "IR_2x" in capsys.readouterr().out


def test_example_stops_when_an_earlier_stage_failed(example, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    _open_ledger(tmp_path)
    record_stage(STUDY, Stage.ALPHA, "fail", reason="신호 재정의 필요", root=tmp_path)
    with pytest.raises(StageGateError):
        example.main(["--no-costs"])


def test_example_book_really_uses_each_stage(example):
    """이름만 부른 시연이 아닌가 — 스텝·적격·널·회귀가 실제 값을 낸다."""
    out = example.run(example.synthetic_inputs(), costs=False)
    assert out["violations"] == []
    assert len(out["recs"]) >= 10 and all(r.n_names > 0 for r in out["recs"][2:])
    assert out["summary"]["oos_steps"] >= 10 and np.isfinite(out["summary"]["IR_2x"])
    assert out["eligible_first"].size >= example.SPEC.min_names
    assert np.isfinite(out["null_ir"]).all() and out["alpha"]["n"] >= 10
    assert out["signal_info"]["rows_used"] > 0 and len(out["ic_days"]) > 0
    assert all(r.cost1 == 0.0 for r in out["recs"]), "--no-costs 경로"
    assert all(np.isnan(r.exante_20) for r in out["recs"]), "위험모델 없이 돈다 — 6단계 skip"


def test_example_pays_list_price_when_the_core_cost_layer_is_there(example):
    pytest.importorskip("krx_quant_core.costs.impact")
    out = example.run(example.synthetic_inputs(), costs=True)
    assert out["violations"] == []
    assert sum(r.cost1 for r in out["recs"]) > 0
    assert out["summary"]["EV_2x"] < out["summary"]["EV_1x"], "비용 2배가 더 깎는다"


def test_known_date_rule_holds_in_the_example_signal(example):
    """3단계 규칙: 공시 다음 거래일부터. 공시일 당일 패널에는 그 분기 값이 없다."""
    inputs = example.synthetic_inputs()
    from swing_it.features.earnings_asof import yoy_strict_wide

    dates = inputs["dates"][inputs["dates"] <= pd.Timestamp(example.END)]
    sig, _ = yoy_strict_wide(inputs["earnings"], inputs["periodic"], inputs["codes"], dates, example.END)
    first_known = np.flatnonzero(np.isfinite(sig[0]))[0]
    avail = pd.Timestamp("2020-03-30")
    assert dates[first_known] > avail and dates[first_known - 1] <= avail
