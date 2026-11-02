"""단계 게이트 — 앞 단계 증거가 없으면 뒤 단계는 돌지 않는다.

연구마다 덧붙이기 전용 원장 ``research/logs/<study>/PIPELINE.jsonl`` 을 둔다. 한 줄이 단계 사건 하나다.
규칙의 이유와 범위는 ``docs/PIPELINE.md``, 결정 기록은 ``docs/superpowers/specs/2026-11-02-pipeline-gate.md``.

- 유효 상태는 ``pass``·``fail``·``skip``·``stale``·``pending`` 다. ``pass``·``skip`` 만 뒤를 연다.
- ``stale``: 앞 단계가 더 나중에 다시 적혔거나(되돌아가기), 증거 파일이 없거나 해시가 어긋났다.
- 원장은 해시 체인(`sha`·`prev`)이고, 읽을 때 행 규칙을 다시 검사한다. 손상은 조용히 건너뛰지 않고 막는다
  — ``TRIALS.jsonl`` 처럼 손상 줄을 무시하면 fail 한 줄이 사라져 게이트가 열린다.

게이트는 판정기가 아니다(GUARDRAILS §8). 기록의 순서와 증거 해시만 보고, 증거가 무엇을 말하는지는 읽지 않는다.
표준 라이브러리만 쓴다(`scripts/check_guardrails.py` 가 읽는다).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from swing_it.pipeline.stages import Stage

__all__ = [
    "BACKFILL_CUTOFF",
    "LEDGER_NAME",
    "NOT_SKIPPABLE",
    "RESULT_STAGES",
    "LedgerCorrupt",
    "RecordRejected",
    "StageGateError",
    "StageState",
    "event_sha",
    "ledger_path",
    "list_studies",
    "read_events",
    "record_stage",
    "require_execution_approval",
    "require_stage",
    "sha256_file",
    "sha256_text",
    "study_states",
    "verify",
]

LEDGER_NAME = "PIPELINE.jsonl"
STATUSES = ("pass", "fail", "skip")
OPEN_STATUSES = ("pass", "skip")
#: 결과를 주장하는 단계 — pass 에 검증 문서가 필요하다.
RESULT_STAGES = frozenset({Stage.ALPHA_VALIDATION, Stage.FORECAST, Stage.BACKTEST})
#: 일부러 안 써도 되는 단계와 달리, 없으면 판정 자체가 거짓이 되는 단계.
NOT_SKIPPABLE = frozenset({Stage.UNIVERSE, Stage.DATA, Stage.ALPHA, Stage.COSTS, Stage.BACKTEST})
#: 소급 기록(``backfill``)은 게이트가 생기기 전 작업에만 쓴다.
BACKFILL_CUTOFF = "2026-11-02"

_STUDY = re.compile(r"[a-z][a-z0-9_]*\Z")
_VERIFIED_DOC = re.compile(r"(.+_verified|REVIEW_.+)\.md\Z")
_REVIEW_MARK = re.compile(r"리뷰|review", re.IGNORECASE)
_KST = timezone(timedelta(hours=9))


class StageGateError(RuntimeError):
    """앞 단계가 열리지 않아 이 단계를 돌 수 없다. ``blockers`` 에 막은 단계들이 있다."""

    def __init__(self, message: str, blockers: "Iterable[StageState]" = ()) -> None:
        super().__init__(message)
        self.blockers = tuple(blockers)


class RecordRejected(StageGateError):
    """규칙에 어긋나는 기록이라 적지 않았다."""


class LedgerCorrupt(StageGateError):
    """원장이 손상됐다. 복구 전까지 이 연구의 모든 단계가 막힌다."""


@dataclass(frozen=True)
class StageState:
    """한 단계의 유효 상태. ``why`` 는 stale 등의 이유, ``event`` 는 그 단계의 최신 줄."""

    stage: Stage
    status: str
    event: "dict[str, Any] | None" = None
    why: tuple[str, ...] = ()

    @property
    def open(self) -> bool:
        return self.status in OPEN_STATUSES

    def describe(self) -> str:
        reason = (self.event or {}).get("reason") or ""
        detail = "; ".join(self.why) or reason
        return f"{self.stage.label}: {self.status}" + (f" — {detail}" if detail else "")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _root(root: "Path | str | None") -> Path:
    # 호출 시점에 찾는다 — 테스트가 monkeypatch 로 바꿀 수 있어야 한다(diagnostics.trials 와 같다).
    return Path(root) if root is not None else _repo_root()


def _check_study(study: str) -> str:
    if not isinstance(study, str) or not _STUDY.match(study):
        raise StageGateError(f"연구 이름은 소문자·숫자·밑줄만 쓴다: {study!r}")
    return study


def _parse_stage(stage: Any, *, exc: type[StageGateError] = StageGateError) -> Stage:
    try:
        return Stage.parse(stage)
    except ValueError as e:
        raise exc(str(e)) from None


def ledger_path(study: str, *, root: "Path | str | None" = None) -> Path:
    """``research/logs/<study>/PIPELINE.jsonl`` 경로(파일이 없어도 경로만 반환)."""
    return _root(root) / "research" / "logs" / _check_study(study) / LEDGER_NAME


def list_studies(*, root: "Path | str | None" = None) -> list[str]:
    """원장이 있는 연구 이름들."""
    return sorted(p.parent.name for p in (_root(root) / "research" / "logs").glob(f"*/{LEDGER_NAME}"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event_sha(ev: dict[str, Any]) -> str:
    """줄 내용의 해시(``sha`` 필드 제외). 다음 줄의 ``prev`` 가 이 값을 가리킨다."""
    body = {k: v for k, v in ev.items() if k != "sha"}
    return sha256_text(json.dumps(body, ensure_ascii=False, sort_keys=True))


# --- 행 규칙 — 기록할 때와 읽을 때 같은 함수를 쓴다 ----------------------------------------


def _approval_problem(approval: Any) -> "str | None":
    if not isinstance(approval, dict):
        return "사용자 승인 기록(날짜·인용)이 없다 — 추론하지 않는다"
    quote = approval.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        return "사용자 승인 인용문이 비었다 — 사용자의 말을 그대로 적는다"
    try:
        day = date.fromisoformat(approval.get("date", ""))
    except (TypeError, ValueError):
        return f"사용자 승인 날짜가 YYYY-MM-DD 가 아니다: {approval.get('date')!r}"
    if day > datetime.now(_KST).date():
        return f"사용자 승인 날짜가 미래다: {day}"
    return None


def _backfill_problem(backfill: Any, stage: Stage) -> "str | None":
    if backfill is None:
        return None
    if stage is Stage.EXECUTION:
        return "10단계는 소급 기록을 쓸 수 없다 — 승인은 추론하지 않는다"
    try:
        day = date.fromisoformat(backfill)
    except (TypeError, ValueError):
        return f"소급 기록 날짜가 YYYY-MM-DD 가 아니다: {backfill!r}"
    if day > date.fromisoformat(BACKFILL_CUTOFF):
        return f"소급 기록은 게이트 이전 작업(≤ {BACKFILL_CUTOFF})에만 쓴다: {day}"
    return None


def _event_problems(ev: dict[str, Any]) -> list[str]:
    """줄 하나만 보고 판단할 수 있는 규칙 위반들."""
    try:
        stage = Stage.parse(ev.get("stage"))
    except ValueError as e:
        return [str(e)]
    status = ev.get("status")
    if status not in STATUSES:
        return [f"status 는 {STATUSES} 중 하나여야 한다: {status!r}"]
    out: list[str] = []
    reason = ev.get("reason")
    has_reason = isinstance(reason, str) and bool(reason.strip())
    if status in ("skip", "fail") and not has_reason:
        out.append(f"{status} 에는 사유가 필요하다")
    if status == "skip" and stage in NOT_SKIPPABLE:
        out.append(f"{stage.label} 은 건너뛸 수 없다 — 없으면 판정 자체가 거짓이 된다")
    if status == "pass":
        if not ev.get("evidence"):
            out.append("pass 에는 증거 파일이 필요하다")
        if stage in RESULT_STAGES and not _is_verified_name((ev.get("reviewer") or {}).get("path")):
            out.append(f"{stage.label} 의 pass 에는 검증 문서(*_verified.md·REVIEW_*.md) 참조가 필요하다")
    if status == "pass" and stage is Stage.EXECUTION:
        problem = _approval_problem(ev.get("approval"))
        if problem:
            out.append(problem)
        if not ev.get("forward_evidence"):
            out.append("10단계 pass 에는 전진 페이퍼 결과 증거가 필요하다")
    if stage is not Stage.EXECUTION and (ev.get("approval") or ev.get("forward_evidence")):
        out.append("승인·전진 증거 필드는 10단계 전용이다")
    problem = _backfill_problem(ev.get("backfill"), stage)
    if problem:
        out.append(problem)
    return out


def _is_verified_name(path: Any) -> bool:
    return isinstance(path, str) and bool(_VERIFIED_DOC.match(PurePosixPath(path).name))


def _refs(ev: dict[str, Any]) -> list[dict[str, str]]:
    """이 줄이 해시로 묶은 파일 전부."""
    out = list(ev.get("evidence") or [])
    for key in ("reviewer", "forward_evidence"):
        if ev.get(key):
            out.append(ev[key])
    return out


def _file_problems(ev: dict[str, Any], root: Path) -> list[str]:
    out = []
    for ref in _refs(ev):
        path = root / ref["path"]
        if not path.is_file():
            out.append(f"증거 파일이 없다: {ref['path']}")
        elif sha256_file(path) != ref["sha256"]:
            out.append(f"증거 해시가 어긋난다: {ref['path']}")
    return out


# --- 상태 접기 ------------------------------------------------------------------------


def _states(events: list[dict[str, Any]], root: "Path | None") -> list[StageState]:
    """사건 열을 단계별 유효 상태로 접는다. ``root`` 가 ``None`` 이면 파일은 보지 않는다."""
    latest: dict[Stage, tuple[int, dict[str, Any]]] = {}
    for i, ev in enumerate(events):
        latest[Stage(ev["stage"])] = (i, ev)
    out = []
    for stage in Stage:
        if stage not in latest:
            out.append(StageState(stage, "pending"))
            continue
        i, ev = latest[stage]
        status, why = ev["status"], []
        if status == "pass":
            redone = sorted({Stage(e["stage"]) for e in events[i + 1:] if e["stage"] < stage})
            why += [f"{s.label} 이 더 나중에 다시 기록됐다" for s in redone]
        if root is not None:
            why += _file_problems(ev, root)
        # fail 은 그 단계의 새 pass 로만 풀린다 — stale 로 흐리지 않는다. 증거 문제는 why 와 verify 에 남는다.
        if why and status != "fail":
            status = "stale"
        out.append(StageState(stage, status, ev, tuple(why)))
    return out


def read_events(study: str, *, root: "Path | str | None" = None) -> list[dict[str, Any]]:
    """원장 전체를 순서대로 읽고 체인·행 규칙·순서를 검사한다. 어긋나면 :class:`LedgerCorrupt`."""
    path = ledger_path(study, root=root)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    prev: "str | None" = None
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        where = f"{path.name}:{n}"
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LedgerCorrupt(f"{study} 원장 손상 {where}: JSON 이 아니다 ({e})") from None
        if not isinstance(ev, dict):
            raise LedgerCorrupt(f"{study} 원장 손상 {where}: 사건 줄이 아니다")
        if ev.get("sha") != event_sha(ev):
            raise LedgerCorrupt(f"{study} 원장 손상 {where}: 줄 해시가 어긋난다(체인) — 손으로 고쳤다")
        if ev.get("prev") != prev:
            raise LedgerCorrupt(f"{study} 원장 손상 {where}: 체인이 끊겼다 — 줄이 빠졌거나 순서가 바뀌었다")
        if ev.get("study") != study:
            raise LedgerCorrupt(f"{study} 원장 손상 {where}: 다른 연구의 줄이다 ({ev.get('study')!r})")
        problems = _event_problems(ev)
        if problems:
            raise LedgerCorrupt(f"{study} 원장 손상 {where}: " + "; ".join(problems))
        stage = Stage(ev["stage"])
        if ev["status"] in OPEN_STATUSES and ev.get("backfill") is None:
            shut = [st for st in _states(events, None) if st.stage < stage and not st.open]
            if shut:
                names = ", ".join(st.stage.label for st in shut)
                raise LedgerCorrupt(f"{study} 원장 손상 {where}: 앞 단계({names})가 안 열렸는데 {ev['status']} 이 적혔다")
        events.append(ev)
        prev = ev["sha"]
    return events


def study_states(study: str, *, root: "Path | str | None" = None) -> list[StageState]:
    """단계 1~11 의 유효 상태. 증거 해시까지 다시 본다."""
    return _states(read_events(study, root=root), _root(root))


# --- 정문 ------------------------------------------------------------------------------


def require_stage(study: str, stage: "Stage | int | str", *, root: "Path | str | None" = None) -> list[StageState]:
    """``stage`` 를 돌려도 되는지 본다. 앞 단계 중 하나라도 pass·skip 이 아니면 :class:`StageGateError`.

    러너는 데이터를 읽기 전에 ``require_stage(*PIPELINE)`` 을 부른다. 자기 단계의 fail 은 막지 않는다 — 다시 돌려야
    새 pass 를 얻는다.
    """
    _check_study(study)
    target = _parse_stage(stage)
    states = study_states(study, root=root)
    blockers = [st for st in states if st.stage < target and not st.open]
    if blockers:
        lines = "\n".join("  - " + st.describe() for st in blockers)
        raise StageGateError(
            f"{study}: {target.label} 을 돌 수 없다. 앞 단계가 열리지 않았다:\n{lines}\n"
            f"상태: python -m swing_it.pipeline status {study} (docs/PIPELINE.md)",
            blockers,
        )
    return states


def require_execution_approval(study: str, *, root: "Path | str | None" = None) -> dict[str, str]:
    """주문 경로의 정문. 10단계 자체의 유효한 pass(사용자 승인·전진 증거)가 있어야 승인 기록을 돌려준다.

    ``require_stage(study, 10)`` 은 1~9 단계만 본다. 그것만으로는 승인 없이 주문이 나갈 수 있다.
    """
    states = require_stage(study, Stage.EXECUTION, root=root)
    state = next(st for st in states if st.stage is Stage.EXECUTION)
    if state.status != "pass":
        raise StageGateError(f"{study}: 주문 실행 승인이 없다 — {state.describe()}", [state])
    return dict(state.event["approval"])


def _ref(path: Any, root: Path) -> dict[str, str]:
    rel = PurePosixPath(str(path))
    if rel.is_absolute() or ".." in rel.parts:
        raise RecordRejected(f"증거 경로는 레포 안의 상대경로여야 한다: {path}")
    target = root / rel
    if not target.is_file():
        raise RecordRejected(f"증거 파일이 없다: {path}")
    return {"path": rel.as_posix(), "sha256": sha256_file(target)}


def _git_sha(root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() or "unknown"


def record_stage(
    study: str,
    stage: "Stage | int | str",
    status: str,
    *,
    evidence: Iterable[str] = (),
    reviewer: "str | None" = None,
    reason: str = "",
    data_fingerprint: "str | None" = None,
    approval: "dict[str, str] | None" = None,
    forward_evidence: "str | None" = None,
    backfill: "str | None" = None,
    root: "Path | str | None" = None,
) -> dict[str, Any]:
    """단계 사건 한 줄을 덧붙인다. 규칙에 어긋나면 적지 않고 :class:`RecordRejected`.

    pass·skip 은 앞 단계가 열려 있어야 적힌다. fail 은 언제든 적힌다(더 막는 쪽이고, 늦게 드러난 실패를 적을 길이
    있어야 한다). ``backfill=<날짜>`` 는 게이트 이전 작업의 순서 검사만 면제한다 — 적어도 게이트는 열리지 않는다.
    """
    _check_study(study)
    target = _parse_stage(stage, exc=RecordRejected)
    base = _root(root)
    events = read_events(study, root=base)      # 손상된 원장에는 덧붙이지 않는다
    if status not in STATUSES:
        raise RecordRejected(f"status 는 {STATUSES} 중 하나여야 한다: {status!r}")

    # 파일을 읽기 전에 행 규칙부터 본다 — 사유·승인 누락이 "파일 없음" 뒤에 숨지 않게.
    draft: dict[str, Any] = {
        "stage": int(target), "status": status, "reason": reason, "approval": approval, "backfill": backfill,
        "evidence": list(evidence),
        "reviewer": {"path": reviewer} if reviewer else None,
        "forward_evidence": {"path": forward_evidence} if forward_evidence else None,
    }
    problems = _event_problems(draft)
    if problems:
        raise RecordRejected(f"{study} {target.label} {status} 거부: " + "; ".join(problems))

    if status in OPEN_STATUSES and backfill is None:
        shut = [st for st in _states(events, base) if st.stage < target and not st.open]
        if shut:
            lines = ", ".join(st.describe() for st in shut)
            raise RecordRejected(f"{study} {target.label} {status} 거부: 앞 단계가 안 열렸다 — {lines}")

    refs = [_ref(p, base) for p in draft["evidence"]]
    review = _ref(reviewer, base) if reviewer else None
    if review is not None and not _REVIEW_MARK.search((base / review["path"]).read_text(encoding="utf-8")):
        raise RecordRejected(f"검증 문서에 리뷰 흔적이 없다: {review['path']}")
    forward = _ref(forward_evidence, base) if forward_evidence else None

    path = ledger_path(study, root=base)
    ev: dict[str, Any] = {
        "study": study,
        "stage": int(target),
        "stage_name": target.ko,
        "status": status,
        "evidence": refs,
        "reviewer": review,
        "approval": approval if approval else None,
        "forward_evidence": forward,
        "data_fingerprint": data_fingerprint,
        "reason": reason,
        "backfill": backfill,
        "git_sha": _git_sha(base),
        "ts": datetime.now(_KST).isoformat(timespec="seconds"),
        "prev": events[-1]["sha"] if events else None,
    }
    ev["sha"] = event_sha(ev)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
    return ev


def verify(study: "str | None" = None, *, root: "Path | str | None" = None) -> list[str]:
    """체인·행 규칙·최신 줄의 증거 해시를 다시 본다. 문제 문장 목록을 돌려준다(비었으면 정상).

    되돌아가기로 생긴 stale 은 문제가 아니다 — 허용된 절차다. 옛 줄의 증거는 대체됐으니 보지 않는다.
    """
    base = _root(root)
    out: list[str] = []
    for name in ([study] if study is not None else list_studies(root=base)):
        try:
            events = read_events(name, root=base)
        except LedgerCorrupt as e:
            out.append(str(e))
            continue
        for st in _states(events, base):
            if st.event is not None:
                out += [f"{name} {st.stage.label}: {p}" for p in _file_problems(st.event, base)]
    return out
