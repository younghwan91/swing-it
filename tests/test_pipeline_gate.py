"""파이프라인 게이트 규칙 — 규칙마다 위반을 주입해 실제로 막히는지 본다.

통과만 보는 테스트는 규칙이 죽어도 초록이다(GUARDRAILS §4.5). 그래서 여기 테스트 대부분은 "거부되는가"다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from swing_it.pipeline import (
    LedgerCorrupt,
    RecordRejected,
    Stage,
    StageGateError,
    gate,
    record_stage,
    require_execution_approval,
    require_stage,
    study_states,
    verify,
)
from swing_it.pipeline.__main__ import main as cli

STUDY = "toy"
APPROVAL = {"date": "2026-11-01", "quote": "실주문 1주 파일럿 진행해"}


def _doc(root: Path, name: str, text: str = "내용\n") -> str:
    p = root / "research" / "logs" / STUDY / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p.relative_to(root).as_posix()


def _verified(root: Path) -> str:
    return _doc(root, "S_verified.md", "quant-reviewer 리뷰 반영\n")


def _pass(root: Path, stage: Stage, **kw) -> dict:
    kw.setdefault("evidence", [_doc(root, f"s{int(stage)}.md", f"{int(stage)}단계 증거\n")])
    if stage in gate.RESULT_STAGES:
        kw.setdefault("reviewer", _verified(root))
    if stage is Stage.EXECUTION:
        kw.setdefault("approval", APPROVAL)
        kw.setdefault("forward_evidence", _doc(root, "forward.md", "전진 결과\n"))
    return record_stage(STUDY, stage, "pass", root=root, **kw)


def _open_upto(root: Path, stage: Stage) -> None:
    """stage 앞까지 전부 pass — 건너뛰지 않고 정문으로 적는다."""
    for s in Stage:
        if s >= stage:
            break
        _pass(root, s)


def _state(root: Path, stage: Stage):
    return next(st for st in study_states(STUDY, root=root) if st.stage is stage)


# --- require_stage: 순서 ---------------------------------------------------------

def test_first_stage_needs_nothing(tmp_path):
    require_stage(STUDY, Stage.UNIVERSE, root=tmp_path)


def test_no_ledger_blocks_every_later_stage(tmp_path):
    """원장이 없으면 앞 단계 증거가 없는 것이다 — 조용히 통과시키지 않는다."""
    for s in list(Stage)[1:]:
        with pytest.raises(StageGateError, match="pending"):
            require_stage(STUDY, s, root=tmp_path)


def test_gate_names_every_blocker(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    with pytest.raises(StageGateError) as e:
        require_stage(STUDY, Stage.ALPHA_VALIDATION, root=tmp_path)
    assert [b.stage for b in e.value.blockers] == [Stage.DATA, Stage.ALPHA]
    assert "2단계" in str(e.value) and "3단계" in str(e.value)


def test_passing_in_order_opens_each_next_stage(tmp_path):
    for s in Stage:
        require_stage(STUDY, s, root=tmp_path)
        _pass(tmp_path, s)


def test_require_accepts_int_stage(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    require_stage(STUDY, 2, root=tmp_path)
    with pytest.raises(StageGateError):
        require_stage(STUDY, 3, root=tmp_path)


def test_unknown_stage_is_rejected(tmp_path):
    with pytest.raises(StageGateError):
        require_stage(STUDY, 12, root=tmp_path)


# --- fail ------------------------------------------------------------------------

def test_fail_blocks_everything_after_it(tmp_path):
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="IC 없음", root=tmp_path)
    for s in Stage:
        if s > Stage.ALPHA_VALIDATION:
            with pytest.raises(StageGateError, match="fail"):
                require_stage(STUDY, s, root=tmp_path)


def test_failed_stage_can_be_rerun(tmp_path):
    """fail 은 뒤를 막지 자기 자신을 막지 않는다 — 다시 돌려야 새 pass 를 얻는다."""
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="IC 없음", root=tmp_path)
    require_stage(STUDY, Stage.ALPHA_VALIDATION, root=tmp_path)


def test_only_a_newer_pass_of_that_stage_clears_a_fail(tmp_path):
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="IC 없음", root=tmp_path)
    with pytest.raises(StageGateError):
        require_stage(STUDY, Stage.FORECAST, root=tmp_path)
    _pass(tmp_path, Stage.ALPHA_VALIDATION)
    require_stage(STUDY, Stage.FORECAST, root=tmp_path)


def test_fail_needs_a_reason(tmp_path):
    with pytest.raises(RecordRejected, match="사유"):
        record_stage(STUDY, Stage.UNIVERSE, "fail", root=tmp_path)


def test_late_fail_is_recordable_even_when_earlier_stages_are_open_or_not(tmp_path):
    """늦게 드러난 실패를 적을 길이 있어야 한다 — fail 은 더 막는 쪽이라 순서를 안 본다."""
    record_stage(STUDY, Stage.BACKTEST, "fail", reason="닫음", root=tmp_path)
    assert _state(tmp_path, Stage.BACKTEST).status == "fail"


# --- skip ------------------------------------------------------------------------

@pytest.mark.parametrize("reason", ["", "   ", "\n"])
def test_skip_needs_a_non_empty_reason(tmp_path, reason):
    _open_upto(tmp_path, Stage.FORECAST)
    with pytest.raises(RecordRejected, match="사유"):
        record_stage(STUDY, Stage.FORECAST, "skip", reason=reason, root=tmp_path)


def test_skip_opens_the_stages_after_it(tmp_path):
    _open_upto(tmp_path, Stage.FORECAST)
    record_stage(STUDY, Stage.FORECAST, "skip", reason="ML 안 씀", root=tmp_path)
    require_stage(STUDY, Stage.RISK_MODEL, root=tmp_path)


@pytest.mark.parametrize("stage", [Stage.UNIVERSE, Stage.DATA, Stage.ALPHA, Stage.COSTS, Stage.BACKTEST])
def test_core_stages_cannot_be_skipped(tmp_path, stage):
    with pytest.raises(RecordRejected, match="건너뛸 수 없"):
        record_stage(STUDY, stage, "skip", reason="귀찮다", root=tmp_path, backfill="2026-11-02")


def test_rank_book_may_skip_alpha_ml_risk_and_optimizer(tmp_path):
    """기관이 아니다 — 순위 북은 4·5·6·7 을 일부러 건너뛸 수 있고 시장 중립 의무도 없다."""
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    for s in (Stage.ALPHA_VALIDATION, Stage.FORECAST, Stage.RISK_MODEL, Stage.PORTFOLIO):
        record_stage(STUDY, s, "skip", reason="순위 북", root=tmp_path)
    require_stage(STUDY, Stage.COSTS, root=tmp_path)


# --- pass ------------------------------------------------------------------------

def test_pass_needs_evidence(tmp_path):
    with pytest.raises(RecordRejected, match="증거"):
        record_stage(STUDY, Stage.UNIVERSE, "pass", root=tmp_path)


def test_evidence_must_exist(tmp_path):
    with pytest.raises(RecordRejected, match="없다"):
        record_stage(STUDY, Stage.UNIVERSE, "pass", evidence=["research/logs/toy/nope.md"], root=tmp_path)


@pytest.mark.parametrize("path", ["/etc/hostname", "../outside.md", "research/../../outside.md"])
def test_evidence_must_stay_inside_the_repo(tmp_path, path):
    with pytest.raises(RecordRejected, match="레포 안"):
        record_stage(STUDY, Stage.UNIVERSE, "pass", evidence=[path], root=tmp_path)


def test_pass_needs_earlier_stages_open(tmp_path):
    """앞 단계가 안 열렸는데 뒤 단계 pass 를 적을 수 없다."""
    ev = _doc(tmp_path, "x.md")
    with pytest.raises(RecordRejected, match="앞 단계"):
        record_stage(STUDY, Stage.COSTS, "pass", evidence=[ev], root=tmp_path)


def test_skip_needs_earlier_stages_open(tmp_path):
    with pytest.raises(RecordRejected, match="앞 단계"):
        record_stage(STUDY, Stage.FORECAST, "skip", reason="ML 안 씀", root=tmp_path)


@pytest.mark.parametrize("stage", sorted(gate.RESULT_STAGES))
def test_results_stage_pass_without_reviewer_is_rejected(tmp_path, stage):
    _open_upto(tmp_path, stage)
    ev = _doc(tmp_path, "result.md")
    with pytest.raises(RecordRejected, match="검증 문서"):
        record_stage(STUDY, stage, "pass", evidence=[ev], root=tmp_path)


@pytest.mark.parametrize("name", ["B.md", "VERDICT.md", "verified_notes.md", "B_verified.txt"])
def test_reviewer_must_be_a_verified_doc_by_name(tmp_path, name):
    """`B.md`(검증 전)를 리뷰 참조로 적으면 안 된다."""
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    ev = _doc(tmp_path, "result.md")
    with pytest.raises(RecordRejected, match="검증 문서"):
        record_stage(STUDY, Stage.ALPHA_VALIDATION, "pass", evidence=[ev],
                     reviewer=_doc(tmp_path, name, "quant-reviewer 리뷰\n"), root=tmp_path)


def test_reviewer_doc_must_exist(tmp_path):
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    ev = _doc(tmp_path, "result.md")
    with pytest.raises(RecordRejected, match="없다"):
        record_stage(STUDY, Stage.ALPHA_VALIDATION, "pass", evidence=[ev],
                     reviewer="research/logs/toy/gone_verified.md", root=tmp_path)


def test_reviewer_doc_must_show_a_review(tmp_path):
    """이름만 맞춘 빈 파일로 통과하지 못한다."""
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    ev = _doc(tmp_path, "result.md")
    with pytest.raises(RecordRejected, match="리뷰 흔적"):
        record_stage(STUDY, Stage.ALPHA_VALIDATION, "pass", evidence=[ev],
                     reviewer=_doc(tmp_path, "empty_verified.md", "\n"), root=tmp_path)


def test_non_results_stage_pass_needs_no_reviewer(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    assert _state(tmp_path, Stage.UNIVERSE).status == "pass"


def test_fail_on_results_stage_needs_no_reviewer(tmp_path):
    record_stage(STUDY, Stage.BACKTEST, "fail", reason="닫음", root=tmp_path)


# --- 10단계 주문 실행 --------------------------------------------------------------

@pytest.mark.parametrize("approval", [
    None, {}, {"date": "2026-11-01"}, {"quote": "진행해"}, {"date": "2026-11-01", "quote": "  "},
    {"date": "언젠가", "quote": "진행해"}, {"date": "2999-01-01", "quote": "진행해"},
])
def test_execution_pass_needs_a_real_approval_record(tmp_path, approval):
    _open_upto(tmp_path, Stage.EXECUTION)
    ev = _doc(tmp_path, "x.md")
    fwd = _doc(tmp_path, "forward.md")
    with pytest.raises(RecordRejected, match="승인"):
        record_stage(STUDY, Stage.EXECUTION, "pass", evidence=[ev], approval=approval,
                     forward_evidence=fwd, root=tmp_path)


def test_execution_pass_needs_forward_evidence(tmp_path):
    _open_upto(tmp_path, Stage.EXECUTION)
    ev = _doc(tmp_path, "x.md")
    with pytest.raises(RecordRejected, match="전진"):
        record_stage(STUDY, Stage.EXECUTION, "pass", evidence=[ev], approval=APPROVAL, root=tmp_path)


def test_execution_pass_cannot_be_backfilled(tmp_path):
    """승인은 추론하지 않는다 — 소급 기록 경로가 있으면 안 된다."""
    ev = _doc(tmp_path, "x.md")
    fwd = _doc(tmp_path, "forward.md")
    with pytest.raises(RecordRejected, match="소급"):
        record_stage(STUDY, Stage.EXECUTION, "pass", evidence=[ev], approval=APPROVAL,
                     forward_evidence=fwd, backfill="2026-11-02", root=tmp_path)


def test_approval_on_other_stages_is_rejected(tmp_path):
    """승인 필드는 10단계 전용 — 다른 단계 줄에 묻혀 있으면 승인으로 오해된다."""
    ev = _doc(tmp_path, "x.md")
    with pytest.raises(RecordRejected, match="10단계"):
        record_stage(STUDY, Stage.UNIVERSE, "pass", evidence=[ev], approval=APPROVAL, root=tmp_path)


def test_require_stage_10_alone_is_not_approval(tmp_path):
    """앞 단계가 다 열려도 주문 경로는 승인 기록이 있어야 한다."""
    _open_upto(tmp_path, Stage.EXECUTION)
    require_stage(STUDY, Stage.EXECUTION, root=tmp_path)
    with pytest.raises(StageGateError, match="승인"):
        require_execution_approval(STUDY, root=tmp_path)


def test_skipping_execution_is_not_approval(tmp_path):
    _open_upto(tmp_path, Stage.EXECUTION)
    record_stage(STUDY, Stage.EXECUTION, "skip", reason="페이퍼 북 — 주문 없음", root=tmp_path)
    require_stage(STUDY, Stage.ATTRIBUTION, root=tmp_path)      # 페이퍼 모니터링은 막지 않는다
    with pytest.raises(StageGateError, match="승인"):
        require_execution_approval(STUDY, root=tmp_path)


def test_recorded_approval_is_returned(tmp_path):
    _open_upto(tmp_path, Stage.EXECUTION)
    _pass(tmp_path, Stage.EXECUTION)
    assert require_execution_approval(STUDY, root=tmp_path) == APPROVAL


def test_approval_goes_stale_when_an_earlier_stage_is_redone(tmp_path):
    _open_upto(tmp_path, Stage.EXECUTION)
    _pass(tmp_path, Stage.EXECUTION)
    _pass(tmp_path, Stage.COSTS)
    with pytest.raises(StageGateError):
        require_execution_approval(STUDY, root=tmp_path)


# --- 되돌아가기 --------------------------------------------------------------------

def test_going_back_makes_later_passes_stale(tmp_path):
    _open_upto(tmp_path, Stage.COSTS)
    _pass(tmp_path, Stage.DATA)
    states = {st.stage: st.status for st in study_states(STUDY, root=tmp_path)}
    assert states[Stage.DATA] == "pass"
    assert states[Stage.UNIVERSE] == "pass"
    assert all(states[s] == "stale" for s in (Stage.ALPHA, Stage.ALPHA_VALIDATION, Stage.FORECAST,
                                              Stage.RISK_MODEL, Stage.PORTFOLIO))


def test_stale_stage_blocks_and_must_be_re_evidenced(tmp_path):
    _open_upto(tmp_path, Stage.FORECAST)
    _pass(tmp_path, Stage.UNIVERSE)
    with pytest.raises(StageGateError, match="stale"):
        require_stage(STUDY, Stage.FORECAST, root=tmp_path)
    for s in (Stage.DATA, Stage.ALPHA, Stage.ALPHA_VALIDATION):
        _pass(tmp_path, s)
    require_stage(STUDY, Stage.FORECAST, root=tmp_path)


def test_going_back_keeps_a_later_fail_as_fail(tmp_path):
    """fail 은 그 단계의 새 pass 로만 풀린다 — stale 로 흐려지지 않는다."""
    _open_upto(tmp_path, Stage.BACKTEST)
    record_stage(STUDY, Stage.BACKTEST, "fail", reason="엣지 없음", root=tmp_path)
    _pass(tmp_path, Stage.ALPHA)
    assert _state(tmp_path, Stage.BACKTEST).status == "fail"


def test_going_back_keeps_a_later_skip(tmp_path):
    """스펙은 pass 만 무효화한다 — 일부러 안 쓴 단계의 결정은 증거에 묶여 있지 않다."""
    _open_upto(tmp_path, Stage.FORECAST)
    record_stage(STUDY, Stage.FORECAST, "skip", reason="ML 안 씀", root=tmp_path)
    _pass(tmp_path, Stage.DATA)
    assert _state(tmp_path, Stage.FORECAST).status == "skip"


def test_a_fail_on_an_earlier_stage_also_invalidates_later_passes(tmp_path):
    _open_upto(tmp_path, Stage.COSTS)
    record_stage(STUDY, Stage.ALPHA, "fail", reason="신호 재정의", root=tmp_path)
    assert _state(tmp_path, Stage.RISK_MODEL).status == "stale"


# --- 증거 무결성 -------------------------------------------------------------------

def test_tampered_evidence_goes_stale_and_blocks(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    _doc(tmp_path, "s1.md", "고쳐 쓴 증거\n")
    assert _state(tmp_path, Stage.UNIVERSE).status == "stale"
    with pytest.raises(StageGateError, match="해시"):
        require_stage(STUDY, Stage.DATA, root=tmp_path)
    assert any("해시" in p for p in verify(STUDY, root=tmp_path))


def test_missing_evidence_goes_stale_and_blocks(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    (tmp_path / "research/logs/toy/s1.md").unlink()
    assert _state(tmp_path, Stage.UNIVERSE).status == "stale"
    with pytest.raises(StageGateError, match="없다"):
        require_stage(STUDY, Stage.DATA, root=tmp_path)
    assert any("없다" in p for p in verify(STUDY, root=tmp_path))


def test_tampered_reviewer_doc_goes_stale(tmp_path):
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    _pass(tmp_path, Stage.ALPHA_VALIDATION)
    _doc(tmp_path, "S_verified.md", "리뷰 고쳐 씀\n")
    assert _state(tmp_path, Stage.ALPHA_VALIDATION).status == "stale"
    with pytest.raises(StageGateError):
        require_stage(STUDY, Stage.FORECAST, root=tmp_path)


def test_tampered_forward_evidence_goes_stale(tmp_path):
    _open_upto(tmp_path, Stage.EXECUTION)
    _pass(tmp_path, Stage.EXECUTION)
    _doc(tmp_path, "forward.md", "전진 결과 고침\n")
    assert _state(tmp_path, Stage.EXECUTION).status == "stale"
    with pytest.raises(StageGateError):
        require_execution_approval(STUDY, root=tmp_path)


def test_replacing_a_stages_evidence_only_rechecks_the_latest_event(tmp_path):
    """옛 줄의 증거는 대체됐으니 다시 보지 않는다 — 새 줄이 그 단계를 대변한다."""
    _pass(tmp_path, Stage.UNIVERSE)
    _doc(tmp_path, "s1.md", "고쳐 쓴 증거\n")
    _pass(tmp_path, Stage.UNIVERSE, evidence=[_doc(tmp_path, "s1b.md", "새 증거\n")])
    assert verify(STUDY, root=tmp_path) == []
    require_stage(STUDY, Stage.DATA, root=tmp_path)


def test_failed_stage_with_tampered_evidence_stays_fail_but_verify_reports_it(tmp_path):
    ev = _doc(tmp_path, "s4.md")
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", evidence=[ev], reason="IC 없음", root=tmp_path)
    _doc(tmp_path, "s4.md", "실패 증거를 고침\n")
    assert _state(tmp_path, Stage.ALPHA_VALIDATION).status == "fail"
    assert any("해시" in p for p in verify(STUDY, root=tmp_path))


def test_going_back_is_not_a_verify_problem(tmp_path):
    """되돌아가기는 허용된 절차다 — CI 를 빨갛게 만들지 않는다."""
    _open_upto(tmp_path, Stage.COSTS)
    _pass(tmp_path, Stage.DATA)
    assert verify(STUDY, root=tmp_path) == []


# --- 원장 무결성 -------------------------------------------------------------------

def _ledger(root: Path) -> Path:
    return gate.ledger_path(STUDY, root=root)


def test_ledger_is_append_only(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    before = _ledger(tmp_path).read_bytes()
    _pass(tmp_path, Stage.DATA)
    assert _ledger(tmp_path).read_bytes().startswith(before)


def test_ledger_line_records_what_the_spec_asks(tmp_path):
    ev = _doc(tmp_path, "s1.md", "증거\n")
    e = record_stage(STUDY, Stage.UNIVERSE, "pass", evidence=[ev], reason="왜",
                     data_fingerprint="[1,2,3]", root=tmp_path)
    line = json.loads(_ledger(tmp_path).read_text(encoding="utf-8").splitlines()[0])
    assert line == e
    assert line["study"] == STUDY and line["stage"] == 1 and line["status"] == "pass"
    assert line["evidence"] == [{"path": ev, "sha256": gate.sha256_file(tmp_path / ev)}]
    assert line["data_fingerprint"] == "[1,2,3]" and line["reason"] == "왜"
    assert line["git_sha"] and line["ts"] and "reviewer" in line
    assert line["prev"] is None


def test_each_line_chains_to_the_previous_one(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    _pass(tmp_path, Stage.DATA)
    first, second = map(json.loads, _ledger(tmp_path).read_text(encoding="utf-8").splitlines())
    assert first["sha"] == gate.event_sha(first)
    assert second["prev"] == first["sha"]


@pytest.mark.parametrize("old,new", [('"fail"', '"pass"'), ("IC 없음", "IC 있음")])
@pytest.mark.parametrize("has_later_line", [True, False])
def test_hand_edited_ledger_is_rejected_not_ignored(tmp_path, old, new, has_later_line):
    """fail 한 줄을 pass 로 바꾸거나 사유를 고치면 게이트가 열리면 안 된다 — 마지막 줄이어도."""
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="IC 없음", root=tmp_path)
    if has_later_line:
        _pass(tmp_path, Stage.ALPHA_VALIDATION)
    p = _ledger(tmp_path)
    p.write_text(p.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    with pytest.raises(LedgerCorrupt, match="체인"):
        require_stage(STUDY, Stage.FORECAST, root=tmp_path)
    assert any("체인" in x for x in verify(STUDY, root=tmp_path))


def test_removing_a_middle_line_is_rejected(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    _pass(tmp_path, Stage.DATA)
    _pass(tmp_path, Stage.ALPHA)
    p = _ledger(tmp_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
    with pytest.raises(LedgerCorrupt, match="체인"):
        require_stage(STUDY, Stage.ALPHA, root=tmp_path)


def test_garbage_line_is_rejected_not_ignored(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    with _ledger(tmp_path).open("a", encoding="utf-8") as f:
        f.write("{broken\n")
    with pytest.raises(LedgerCorrupt):
        require_stage(STUDY, Stage.DATA, root=tmp_path)


def test_record_refuses_to_append_to_a_corrupt_ledger(tmp_path):
    _pass(tmp_path, Stage.UNIVERSE)
    with _ledger(tmp_path).open("a", encoding="utf-8") as f:
        f.write("{broken\n")
    with pytest.raises(LedgerCorrupt):
        _pass(tmp_path, Stage.DATA)


def _forge(root: Path, **fields) -> None:
    """체인을 맞춰 손으로 덧붙인다 — 체인 검사만으로는 못 잡는 위조."""
    p = _ledger(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    base = {"study": STUDY, "status": "pass", "evidence": [], "reviewer": None, "approval": None,
            "forward_evidence": None, "git_sha": "x", "data_fingerprint": None, "reason": "",
            "backfill": None, "ts": "2026-11-02T00:00:00+09:00",
            "prev": json.loads(lines[-1])["sha"] if lines else None}
    ev = {**base, **fields}
    ev["sha"] = gate.event_sha(ev)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")


def _forge_evidence(root: Path) -> list[dict]:
    ev = _doc(root, "forged.md")
    return [{"path": ev, "sha256": gate.sha256_file(root / ev)}]


def test_forged_line_without_reviewer_is_rejected(tmp_path):
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    _forge(tmp_path, stage=4, evidence=_forge_evidence(tmp_path))
    with pytest.raises(LedgerCorrupt, match="검증 문서"):
        require_stage(STUDY, Stage.FORECAST, root=tmp_path)


def test_forged_line_out_of_order_is_rejected(tmp_path):
    _forge(tmp_path, stage=8, evidence=_forge_evidence(tmp_path))
    with pytest.raises(LedgerCorrupt, match="앞 단계"):
        require_stage(STUDY, Stage.BACKTEST, root=tmp_path)


def test_forged_skip_without_reason_is_rejected(tmp_path):
    _open_upto(tmp_path, Stage.FORECAST)
    _forge(tmp_path, stage=5, status="skip")
    with pytest.raises(LedgerCorrupt, match="사유"):
        require_stage(STUDY, Stage.RISK_MODEL, root=tmp_path)


@pytest.mark.parametrize("fields,match", [
    ({"stage": "4"}, "정수"),
    ({"stage": 12}, "모르는 단계"),
    ({"stage": 1, "evidence": "x.md"}, "목록"),
    ({"stage": 1, "evidence": [{"path": "x.md"}]}, "모양"),
    ({"stage": 1, "evidence": [{"path": "x.md", "sha256": "0"}], "reviewer": "x_verified.md"}, "모양"),
])
def test_malformed_line_is_corrupt_not_a_crash(tmp_path, fields, match):
    """어긋난 줄이 KeyError·TypeError 로 죽으면 원인이 가려진다."""
    _forge(tmp_path, **fields)
    with pytest.raises(LedgerCorrupt, match=match):
        require_stage(STUDY, Stage.DATA, root=tmp_path)


def test_line_of_another_study_is_rejected(tmp_path):
    _forge(tmp_path, stage=1, study="other", evidence=_forge_evidence(tmp_path))
    with pytest.raises(LedgerCorrupt, match="연구"):
        require_stage(STUDY, Stage.DATA, root=tmp_path)


@pytest.mark.parametrize("bad", ["", "Bad", "a/b", "../x", "x y"])
def test_bad_study_name_is_rejected(tmp_path, bad):
    with pytest.raises(StageGateError, match="연구 이름"):
        require_stage(bad, Stage.UNIVERSE, root=tmp_path)


def test_bad_status_is_rejected(tmp_path):
    with pytest.raises(RecordRejected, match="status"):
        record_stage(STUDY, Stage.UNIVERSE, "stale", reason="x", root=tmp_path)


# --- 소급 기록 ---------------------------------------------------------------------

def test_backfill_records_out_of_order_but_does_not_open_the_gate(tmp_path):
    """옛 증거를 적을 수는 있어도 게이트는 여전히 앞 단계 fail 로 막힌다 — 우회로가 아니다."""
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="SIG 실패", root=tmp_path)
    ev = _doc(tmp_path, "risk.md")
    record_stage(STUDY, Stage.RISK_MODEL, "pass", evidence=[ev], backfill="2026-11-02", root=tmp_path)
    assert _state(tmp_path, Stage.RISK_MODEL).status == "pass"
    with pytest.raises(StageGateError):
        require_stage(STUDY, Stage.PORTFOLIO, root=tmp_path)


def test_backfill_only_covers_work_before_the_gate_existed(tmp_path):
    ev = _doc(tmp_path, "risk.md")
    late = (date.fromisoformat(gate.BACKFILL_CUTOFF) + timedelta(days=1)).isoformat()
    with pytest.raises(RecordRejected, match="소급"):
        record_stage(STUDY, Stage.RISK_MODEL, "pass", evidence=[ev], backfill=late, root=tmp_path)


def test_backfill_still_needs_the_stage_rules(tmp_path):
    ev = _doc(tmp_path, "b.md")
    with pytest.raises(RecordRejected, match="검증 문서"):
        record_stage(STUDY, Stage.BACKTEST, "pass", evidence=[ev], backfill="2026-11-02", root=tmp_path)


# --- CLI ---------------------------------------------------------------------------

def test_cli_record_status_verify_roundtrip(tmp_path, capsys):
    ev = _doc(tmp_path, "s1.md")
    root = ["--root", str(tmp_path)]
    assert cli(["record", STUDY, "1", "pass", "--evidence", ev, "--reason", "왜", *root]) == 0
    assert cli(["record", STUDY, "universe", "pass", "--evidence", ev, *root]) == 0
    capsys.readouterr()
    assert cli(["status", STUDY, *root]) == 0
    out = capsys.readouterr().out
    assert "유니버스" in out and "pass" in out and "pending" in out
    assert cli(["verify", STUDY, *root]) == 0


def test_cli_verify_fails_on_tampered_evidence(tmp_path, capsys):
    ev = _doc(tmp_path, "s1.md")
    root = ["--root", str(tmp_path)]
    cli(["record", STUDY, "1", "pass", "--evidence", ev, *root])
    _doc(tmp_path, "s1.md", "고침\n")
    assert cli(["verify", STUDY, *root]) == 1
    assert "해시" in capsys.readouterr().out


def test_cli_rejected_record_exits_nonzero_and_writes_nothing(tmp_path, capsys):
    root = ["--root", str(tmp_path)]
    assert cli(["record", STUDY, "1", "skip", *root]) == 1
    assert not _ledger(tmp_path).exists()


def test_cli_execution_needs_both_approval_flags(tmp_path):
    _open_upto(tmp_path, Stage.EXECUTION)
    ev = _doc(tmp_path, "x.md")
    fwd = _doc(tmp_path, "forward.md")
    root = ["--root", str(tmp_path)]
    assert cli(["record", STUDY, "10", "pass", "--evidence", ev, "--forward-evidence", fwd,
                "--approval-date", "2026-11-01", *root]) == 1
    assert cli(["record", STUDY, "10", "pass", "--evidence", ev, "--forward-evidence", fwd,
                "--approval-date", "2026-11-01", "--approval-quote", "진행해", *root]) == 0


def test_cli_status_marks_blocked_stages(tmp_path, capsys):
    _open_upto(tmp_path, Stage.ALPHA_VALIDATION)
    root = ["--root", str(tmp_path)]
    record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="SIG 실패", root=tmp_path)
    ev = _doc(tmp_path, "risk.md")
    record_stage(STUDY, Stage.RISK_MODEL, "pass", evidence=[ev], backfill="2026-11-02", root=tmp_path)
    capsys.readouterr()
    cli(["status", STUDY, *root])
    out = capsys.readouterr().out
    assert "4단계" in out and "SIG 실패" in out


def test_module_entry_point_runs_in_the_real_repo():
    """`python -m swing_it.pipeline` 이 실제로 뜨는가(패키지 __main__ 연결)."""
    r = subprocess.run([sys.executable, "-m", "swing_it.pipeline", "status"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_gate_stays_dependency_free():
    """`check_guardrails.py` 가 무의존으로 게이트를 읽는다 — numpy·pandas 가 끌려오면 그 약속이 깨진다."""
    code = ("import sys; import swing_it.pipeline; from swing_it.pipeline import gate, stages; "
            "bad = {'numpy', 'pandas', 'krx_quant_core'} & set(sys.modules); "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], timeout=60).returncode == 0


# --- 러너 끝까지 — 완료 조건 (a) -----------------------------------------------------

_RUNNER = '''
from swing_it.pipeline import require_stage

PIPELINE = ("toy", 5)
TOUCHED = []

def main():
    require_stage(*PIPELINE)
    TOUCHED.append("data")          # 여기가 데이터 접근 자리다

if __name__ == "__main__":
    main()
'''


def _load_runner(tmp_path):
    import importlib.util

    f = tmp_path / "runner_fixture.py"
    f.write_text(_RUNNER, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("runner_fixture", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ready(root):
    _open_upto(root, Stage.FORECAST)


@pytest.mark.parametrize("state,match", [
    ("no_ledger", "pending"),
    ("earlier_stage_failed", "fail"),
    ("earlier_stage_redone", "stale"),
    ("evidence_tampered", "해시"),
    ("evidence_missing", "없다"),
    ("reviewer_doc_tampered", "해시"),
    ("ledger_hand_edited", "체인"),
])
def test_runner_does_not_touch_data_when_earlier_evidence_is_not_good(tmp_path, monkeypatch, state, match):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    if state != "no_ledger":
        _ready(tmp_path)
    if state == "earlier_stage_failed":
        record_stage(STUDY, Stage.ALPHA_VALIDATION, "fail", reason="IC 없음", root=tmp_path)
    elif state == "earlier_stage_redone":
        _pass(tmp_path, Stage.DATA)
    elif state == "evidence_tampered":
        _doc(tmp_path, "s2.md", "고침\n")
    elif state == "evidence_missing":
        (tmp_path / "research/logs/toy/s2.md").unlink()
    elif state == "reviewer_doc_tampered":
        _doc(tmp_path, "S_verified.md", "리뷰 고침\n")
    elif state == "ledger_hand_edited":
        p = _ledger(tmp_path)
        p.write_text(p.read_text(encoding="utf-8").replace('"stage": 2', '"stage": 3', 1), encoding="utf-8")
    runner = _load_runner(tmp_path)
    with pytest.raises(StageGateError, match=match):
        runner.main()
    assert runner.TOUCHED == []


def test_runner_runs_once_earlier_stages_are_open(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    _ready(tmp_path)
    runner = _load_runner(tmp_path)
    runner.main()
    assert runner.TOUCHED == ["data"]
