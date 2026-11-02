"""실제 원장 — 완료 조건 (c). 연구 3개가 **실제 상태 그대로** 있는가.

소급 기록한 줄은 체인 해시로 고정한다. 이 줄들이 지워지거나 고쳐지면 여기서 빨개진다(체인은 끝부분 잘라내기를 못 잡는다).
뒤에 새 줄이 붙는 것은 정상이다 — 그래서 상태는 고정한 줄까지로 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from swing_it.pipeline import Stage, StageGateError, gate, require_stage, verify

REPO = Path(__file__).resolve().parents[1]

#: 2026-11-02 소급 기록: (줄 수, 마지막 줄 sha). 원장을 새로 적으면 바뀐다 — 그것이 이 고정의 목적이다.
PINNED = {
    "pead_optimizer": (9, "9b4de43d86d6b4bfed6a02b3d34893fd203c2ade295903af63b16d0184daabbc"),
    "pead_forward": (8, "6ca3107fe6f756c5a741b0cb5f2eb4c75f11b58ba01d211f45b420bebe697384"),
}

#: 소급 기록 시점의 실제 상태. 근거는 줄마다의 reason·evidence 다.
REAL_STATUS = {
    "pead_optimizer": {
        Stage.UNIVERSE: "pass", Stage.DATA: "pass", Stage.ALPHA: "pass",
        Stage.ALPHA_VALIDATION: "fail",     # B_verified: SIG 실패, "판단 불가(신호 약함)"
        Stage.FORECAST: "skip",
        Stage.RISK_MODEL: "pass",           # A1_repaired
        Stage.PORTFOLIO: "pass", Stage.COSTS: "pass",
        Stage.BACKTEST: "fail",             # 단계 B 닫음, 전진 B 미개시
        Stage.EXECUTION: "pending", Stage.ATTRIBUTION: "pending",
    },
    "pead_forward": {
        Stage.UNIVERSE: "pass", Stage.DATA: "pass", Stage.ALPHA: "pass",
        Stage.ALPHA_VALIDATION: "skip",     # 표본 안 검증 미통과를 알고 전진에 맡김 — pass 가 아니다
        Stage.FORECAST: "skip", Stage.RISK_MODEL: "skip", Stage.PORTFOLIO: "skip",
        Stage.COSTS: "pass",
        Stage.BACKTEST: "pending",          # 전진 진행 중(2026-11-05~, fcs). 첫 판정 K=12 전에는 pass 가 없다
        Stage.EXECUTION: "pending", Stage.ATTRIBUTION: "pending",
    },
}


def _events(study: str) -> list[dict]:
    return gate.read_events(study, root=REPO)


def test_real_ledgers_verify():
    """체인·행 규칙·증거 해시 — 증거를 고쳤는데 원장이 모르면 여기서 빨개진다."""
    assert verify(root=REPO) == []
    assert set(PINNED) <= set(gate.list_studies(root=REPO))


@pytest.mark.parametrize("study", sorted(PINNED))
def test_backfilled_lines_are_pinned(study):
    n, sha = PINNED[study]
    events = _events(study)
    assert len(events) >= n, f"{study} 소급 줄이 지워졌다"
    assert events[n - 1]["sha"] == sha, f"{study} 소급 줄이 바뀌었다"
    assert all(e["backfill"] == gate.BACKFILL_CUTOFF for e in events[:n]), "소급 기록임이 줄에 남아야 한다"


@pytest.mark.parametrize("study", sorted(PINNED))
def test_real_status_as_backfilled(study):
    n, _ = PINNED[study]
    states = {st.stage: st.status for st in gate._states(_events(study)[:n], None)}
    assert states == REAL_STATUS[study]


def test_no_invented_pass_on_the_results_stages_of_a_closed_study():
    n, _ = PINNED["pead_optimizer"]
    passes = [e["stage"] for e in _events("pead_optimizer")[:n] if e["status"] == "pass"]
    assert not set(passes) & {int(s) for s in gate.RESULT_STAGES}, "B_verified 는 통과라고 하지 않았다"


def test_backfilled_evidence_points_at_the_studys_own_artifacts():
    for study, (n, _) in PINNED.items():
        for e in _events(study)[:n]:
            for ref in e["evidence"]:
                assert ref["path"].startswith("research/logs/pead_"), ref
                assert (REPO / ref["path"]).is_file(), ref


def test_failed_study_is_stuck_until_a_newer_pass():
    n, _ = PINNED["pead_optimizer"]
    if len(_events("pead_optimizer")) > n:
        pytest.skip("소급 기록 뒤 새 줄이 있다 — 그 상태는 원장이 말한다")
    for s in Stage:
        if s > Stage.ALPHA_VALIDATION:
            with pytest.raises(StageGateError):
                require_stage("pead_optimizer", s, root=REPO)


def test_forward_arm_is_open_at_backtest_and_shut_after_it():
    """전진은 진행 중이어야 하고(게이트가 막으면 안 된다), 판정 전에는 10단계가 열리면 안 된다."""
    n, _ = PINNED["pead_forward"]
    if len(_events("pead_forward")) > n:
        pytest.skip("소급 기록 뒤 새 줄이 있다 — 그 상태는 원장이 말한다")
    require_stage("pead_forward", Stage.BACKTEST, root=REPO)
    require_stage("pead_forward", Stage.ATTRIBUTION, root=REPO)     # 전진 모니터링은 판정을 기다리지 않는다
    with pytest.raises(StageGateError):
        require_stage("pead_forward", Stage.EXECUTION, root=REPO)


def test_ml_forecast_has_no_ledger_until_it_is_merged():
    """브랜치 younghwan91/ml-forecast-prereg 는 미머지다. main 에 있는 것만 적는다 — 머지 시 린트 (k) 가 원장을 요구한다."""
    if (REPO / "research/logs/ml_forecast").exists():
        assert gate.ledger_path("ml_forecast", root=REPO).exists()
        return
    assert not gate.ledger_path("ml_forecast", root=REPO).exists()
    with pytest.raises(StageGateError, match="pending"):
        require_stage("ml_forecast", Stage.DATA, root=REPO)


def test_ledger_lines_are_valid_json_with_the_spec_fields():
    required = {"study", "stage", "status", "evidence", "git_sha", "data_fingerprint", "reviewer", "reason", "ts"}
    for study in PINNED:
        for raw in gate.ledger_path(study, root=REPO).read_text(encoding="utf-8").splitlines():
            assert required <= set(json.loads(raw))
