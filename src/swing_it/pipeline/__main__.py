"""``python -m swing_it.pipeline status|record|verify``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from swing_it.pipeline import gate
from swing_it.pipeline.gate import RecordRejected, StageGateError
from swing_it.pipeline.stages import Stage


def _print_status(study: str, root: "Path | None") -> None:
    states = gate.study_states(study, root=root)
    print(f"{study}  ({gate.ledger_path(study, root=root).relative_to(gate._root(root))})")
    for st in states:
        ev = st.event or {}
        day = (ev.get("ts") or "")[:10]
        note = "; ".join(st.why) or ev.get("reason") or ""
        shut = gate._blockers(states, st.stage)
        mark = f"[{int(shut[0].stage)}단계에서 막힘] " if shut else ""
        print(f"  {int(st.stage):>2} {st.stage.ko:<14} {st.status:<8} {day:<10} {mark}{note}".rstrip())


def _status(args: argparse.Namespace) -> int:
    studies = [args.study] if args.study else gate.list_studies(root=args.root)
    if not studies:
        print("원장이 없다.")
    for i, study in enumerate(studies):
        if i:
            print()
        _print_status(study, args.root)
    return 0


def _refuse_other_repo(args: argparse.Namespace) -> None:
    """지금 폴더가 다른 레포면 거부한다. 편집 설치는 다른 워크트리의 원장에 조용히 적는다."""
    if args.root is not None:
        return
    root = gate._root(None).resolve()
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return
    top = r.stdout.strip()
    if r.returncode == 0 and top and Path(top).resolve() != root:
        raise RecordRejected(f"지금 폴더의 레포({top})와 원장이 적힐 레포({root})가 다르다. "
                             "그 레포의 환경으로 돌리거나 --root 로 명시한다.")


def _record(args: argparse.Namespace) -> int:
    _refuse_other_repo(args)
    approval = None
    if args.approval_date is not None or args.approval_quote is not None:
        approval = {"date": args.approval_date, "quote": args.approval_quote}
    ev = gate.record_stage(
        args.study, args.stage, args.status,
        evidence=args.evidence, reviewer=args.reviewer, reason=args.reason,
        data_fingerprint=args.data_fingerprint, approval=approval,
        forward_evidence=args.forward_evidence, backfill=args.backfill, root=args.root,
    )
    path = gate.ledger_path(ev["study"], root=args.root)
    print(f"기록: {ev['study']} {Stage(ev['stage']).label} {ev['status']} → {path}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    problems = gate.verify(args.study, root=args.root)
    for p in problems:
        print(p)
    if problems:
        print(f"\n문제 {len(problems)}건 — 증거를 되돌리거나, 고친 증거로 그 단계를 다시 기록한다(docs/PIPELINE.md).")
        return 1
    print("원장·증거 해시 정상.")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m swing_it.pipeline", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=None, help="레포 루트(기본: 이 패키지가 든 레포)")

    p = sub.add_parser("status", parents=[common], help="단계별 상태")
    p.add_argument("study", nargs="?")
    p.set_defaults(func=_status)

    p = sub.add_parser("record", parents=[common], help="단계 사건 한 줄을 덧붙인다")
    p.add_argument("study")
    p.add_argument("stage", help="번호 또는 이름(예: 4, alpha_validation)")
    p.add_argument("status", choices=gate.STATUSES)
    p.add_argument("--evidence", nargs="*", default=[], help="레포 상대 경로")
    p.add_argument("--reviewer", help="검증 문서(*_verified.md·REVIEW_*.md)")
    p.add_argument("--reason", default="")
    p.add_argument("--data-fingerprint")
    p.add_argument("--approval-date", help="10단계: 사용자 승인 날짜 YYYY-MM-DD")
    p.add_argument("--approval-quote", help="10단계: 사용자의 말 그대로")
    p.add_argument("--forward-evidence", help="10단계: 전진 페이퍼 결과")
    p.add_argument("--backfill", help=f"게이트 이전 작업의 실제 날짜(≤ {gate.BACKFILL_CUTOFF})")
    p.set_defaults(func=_record)

    p = sub.add_parser("verify", parents=[common], help="체인·행 규칙·증거 해시 재검사(문제 있으면 1)")
    p.add_argument("study", nargs="?")
    p.set_defaults(func=_verify)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except StageGateError as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
