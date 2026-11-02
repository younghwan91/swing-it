#!/usr/bin/env python
"""가드레일 경계 린트 — 순수 파이썬(무의존), CI 스텝.

GUARDRAILS §5 로드맵 item 5 를 코드로 강제한다. CI 는 특정 연구가 가드레일을
'썼는지'는 못 막지만, 아래 세 가지 **구조적 경계**는 기계적으로 지킬 수 있다:

  (a) 경계 위반 — ``src/swing_it/`` 가 ``research/`` 를 import 하면 실패.
      (TEMPLATE 1단계: src 는 research 를 import 하지 않는다. 반대만 허용.)
  (b) 판정 유실 — ``research/experiments/*_gate.py`` 에 대응 VERDICT.md 가 없으면 실패.
      (정직한 부정 결과도 산출물. 게이트만 돌리고 로깅 안 하면 기록이 사라진다.)
  (c) 하드코딩 판정 — 실험이 리터럴 "PASS"/"FAIL" 판정 문자열을 값으로 박으면 실패.
      (리포터-not-판정기, GUARDRAILS §8. 하드코딩 합격선 자체가 결정론적 과최적.)
  (d) raw earnings SELECT — 정정공시 버전이 중복 행으로 새어나오는 걸 막는다.
  (e) 원장 우회 — prop_gate/gate_sim **호출 지점마다** config= 를 요구한다(다중검정 N).
  (f) 자체 배터리 — *_gate.py 가 공용 하버스를 안 쓰면 실패.
  (g) raw 가격 SELECT — 유니버스에서 폐지 종목이 조용히 빠지는 걸 막는다.
  (g2) 생존자 전용 조인 — ``supply_demand JOIN stocks`` 는 WHERE 없이도 폐지분을 떨군다.
  (h) 코드 없는 판정 — (b)의 역방향. VERDICT 에 러너가 등록·실재하거나 재현불가 선언 필요.
  (j) ML 예측 피처 원천 — ``ml_forecast_*.py`` 가 생존자 전용·원천이 갈린 표를 읽으면 실패.

**이 린트 자체는 tests/test_check_guardrails.py 가 검사한다.** 각 규칙에 위반을 주입해
실제로 실패하는지 확인한다 — 규칙이 죽어도 초록인 상태를 실제로 겪었기 때문이다((e) 참조).

  (k) 파이프라인 순서 — 데이터·백테스트를 부르는 파일은 study·단계를 선언하고 읽기 전에 게이트를 부른다(docs/PIPELINE.md).
      원장 체인·행 규칙·증거 해시, 기준 브랜치 원장의 덧붙이기 전용도 여기서 본다.

무의존·표준 라이브러리만. exit 0 = 위반 없음, exit 1 = 위반.
실행: ``python scripts/check_guardrails.py``
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "swing_it"
EXPERIMENTS = REPO / "research" / "experiments"
LOGS = REPO / "research" / "logs"

# --- (b) 게이트 → 로그 디렉터리 매핑 규약 --------------------------------------
# 기본 규약: ``<name>_gate.py`` → ``research/logs/<name>/VERDICT.md``.
# 아래는 그 규약의 명시적 예외다(현실 반영 — 하드코딩된 합격선이 아니라 규약 맵).
#
# EXEMPT: 알파가 아니라 공용 하버스/라이브러리라 자체 VERDICT 를 갖지 않는 게이트.
GATE_EXEMPT = {
    # prop_gate 는 세 프랍 셋업을 동일 잣대로 재는 공용 리포터 배터리(알파 아님).
    "prop_gate",
    # 생존편향 측정 스크립트. 알파를 심사하는 게 아니라 유니버스 보정이 기존 측정에
    # 미친 영향을 재는 리포터라, 트레이드 표본도 사전등록 config 도 없다.
    "survivorship_bias_gate",
}
# OVERRIDE: 로그 디렉터리 이름이 기본 규약과 다른 게이트(판정이 통합·개명된 경우).
GATE_LOG_OVERRIDE = {
    # 베이스 PEAD 게이트. 판정은 pead_concentrated/VERDICT.md 로 통합됨
    # (+ research/logs/PEAD_REFINEMENT_RESULTS.md).
    "pead_gate": "pead_concentrated",
}

# (a) src import 경계 위반 패턴 — ``import research`` / ``from research``.
_IMPORT_RESEARCH = re.compile(r"^\s*(?:import\s+research\b|from\s+research\b)")
# (c) 값으로 박힌 리터럴 판정 문자열(주석·독스트링 문구는 제외하려 따옴표를 요구).
_LITERAL_VERDICT = re.compile(r"""(?<![#])["'](PASS|FAIL|PASSED|FAILED)["']""")


#: 스캔에서 뺄 디렉터리. `.claude/worktrees` 는 에이전트용 **레포 복사본**이라
#: 그냥 두면 같은 위반이 두 번 세어지고, 남의 작업 중인 코드로 CI 가 깨진다.
_SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".claude", ".git"}


def _iter_py(root: Path):
    for p in sorted(root.rglob("*.py")):
        # **루트 기준** 상대경로로 판단한다. 절대경로를 보면 레포가
        # `.claude/...` 아래 놓였을 때 스캔이 통째로 조용히 비어버린다.
        if _SKIP_DIRS & set(p.relative_to(root).parts):
            continue
        yield p


def check_src_no_research_import() -> list[str]:
    """(a) src/swing_it 가 research 를 import 하면 위반."""
    out = []
    for p in _iter_py(SRC):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _IMPORT_RESEARCH.match(line):
                rel = p.relative_to(REPO)
                out.append(f"[boundary] {rel}:{i} — src must not import research: {line.strip()}")
    return out


def check_gates_have_verdict() -> list[str]:
    """(b) 각 *_gate.py 에 대응 VERDICT.md 가 있어야 한다(EXEMPT/OVERRIDE 규약 적용)."""
    out = []
    for p in sorted(EXPERIMENTS.glob("*_gate.py")):
        stem = p.stem                       # 예: pullback_prop_gate
        if stem in GATE_EXEMPT:
            continue
        log_dir = GATE_LOG_OVERRIDE.get(stem, stem[: -len("_gate")])  # 기본: _gate 제거
        verdict = LOGS / log_dir / "VERDICT.md"
        if not verdict.exists():
            rel = p.relative_to(REPO)
            out.append(
                f"[verdict] {rel} — 대응 VERDICT 없음: "
                f"research/logs/{log_dir}/VERDICT.md 를 만들거나 규약 맵을 갱신하라."
            )
    return out


def check_no_literal_verdict() -> list[str]:
    """(c) 실험이 리터럴 "PASS"/"FAIL" 판정 문자열을 값으로 박으면 위반.

    주석 라인은 통째로 제외한다(예: prop_gate 독스트링의 'PASS/FAIL bool 도 두지 않는다'
    같은 서술은 따옴표가 없어 애초에 매칭 안 되지만, 안전하게 주석도 건너뛴다).
    """
    out = []
    for p in _iter_py(EXPERIMENTS):
        for i, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0]     # 인라인 주석 제거
            if _LITERAL_VERDICT.search(line):
                rel = p.relative_to(REPO)
                out.append(
                    f"[reporter] {rel}:{i} — 리터럴 판정 문자열(리포터-not-판정기 위반): {raw.strip()}"
                )
    return out


# (d) earnings 는 정정공시가 새 행으로 쌓이는 버전 테이블이라(PK 에 knowledge_date),
# 그냥 SELECT 하면 (code, period)당 행이 여럿이 되어 하류 패널이 조용히 중복된다.
# storage.read_earnings() 가 as-of 로 한 버전만 고르는 유일한 정문이다.
# SELECT ... FROM earnings 한 줄 형태만 잡는다 — "from earnings drift" 같은 산문이
# 걸리면 규칙이 소음이 되어 아무도 안 본다.
_RAW_EARNINGS = re.compile(r"\bSELECT\b.*\bFROM\s+earnings\b", re.IGNORECASE)
_EARNINGS_READ_EXEMPT = {
    "src/swing_it/storage.py",          # 정문 자신 + 스키마 문자열
    "tests/test_earnings_asof.py",      # 정문의 테스트 — 버전이 실제로 쌓이는지 직접 확인해야 한다
}


def check_no_raw_earnings_select() -> list[str]:
    """(d) storage 밖에서 earnings 를 직접 SELECT 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _EARNINGS_READ_EXEMPT or rel.startswith(".venv/"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _RAW_EARNINGS.search(line):
                out.append(
                    f"[earnings] {rel}:{i} — earnings 를 직접 SELECT 했다. "
                    f"swing_it.storage.read_earnings(con, asof=...) 를 쓸 것 "
                    f"(정정공시 버전이 중복 행으로 새어나온다)."
                )
    return out


# (e) 원장을 남기는 진입점(prop_gate·gate_sim)은 **호출 지점마다** config= 를 받아야 한다.
# 안 넘기면 DSR/t-haircut 이 계산되지 않고, 두 번째 config 를 조용히 돌려도 아무도
# 못 잡는다(GUARDRAILS §4 공백 5). 이 레포에서 반복된 "기능은 있고 쓰는 쪽이 안 씀"을
# 코드로 막는다.
#
# 2026-08-16 재작성. 이전 구현은 **파일 단위**였다 — 파일 어딘가에 `config=` 가 한 번만
# 있으면 같은 파일의 나머지 prop_gate 호출이 전부 면제됐다. 그래서 실제로 다음이 CI 를
# 통과하고 있었다: 사전등록 1회는 config= 를 넘기고, 민감도 스윕 18~27칸은 config 없이
# gate_sim 을 거쳐 원장을 우회 → TRIALS.jsonl 1~2줄 → `n_trials <= 1` → **deflation 0**.
# 즉 린트가 막으려던 바로 그 실패 모드를 린트가 못 봤다. 이제 ast 로 호출 노드를 세고
# 파일 전체 텍스트가 아니라 그 호출의 키워드만 본다. 정규식은 다인자 줄바꿈 호출에서
# 괄호 범위를 못 잡으므로 쓰지 않는다.
_LEDGER_ENTRYPOINTS = ("prop_gate", "gate_sim")


def _called_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def check_gates_record_trials() -> list[str]:
    """(e) prop_gate/gate_sim 호출은 각각 config= 를 동반해야 한다(호출 지점 단위)."""
    out = []
    for p in _iter_py(EXPERIMENTS):
        if p.stem in GATE_EXEMPT:
            continue
        text = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:                      # 파싱 실패는 침묵시키지 않는다
            out.append(f"[trials] {p.relative_to(REPO)} — 파싱 실패: {exc}")
            continue
        # 하버스 자신의 정의부(def prop_gate / def gate_sim)는 대상이 아니다.
        defined_here = {
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in _LEDGER_ENTRYPOINTS
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called_name(node)
            if name not in _LEDGER_ENTRYPOINTS or name in defined_here:
                continue
            if any(kw.arg == "config" for kw in node.keywords):
                continue
            out.append(
                f"[trials] {p.relative_to(REPO)}:{node.lineno} — {name}() 에 config= 를 "
                f"안 넘겼다. 사전등록 config 를 넘겨야 다중검정 원장(TRIALS.jsonl)에 "
                f"시행이 남고 DSR 이 계산된다(민감도 격자 셀도 시행이다 — GUARDRAILS §6)."
            )
    return out


# (f) *_gate.py 는 공용 하버스(prop_gate)를 재사용해야 한다. 자체 배터리를 새로 짜면
# 음성대조·비용 2배 스트레스·손안댄창·R분포·fragility 중 무엇이 빠졌는지 아무도 모른다
# (GUARDRAILS §4 공백 6·7 — 실제로 pead_gate 가 그 상태였다). 동일 잣대 강제.
_USES_HARNESS = re.compile(r"\bprop_gate\s*\(")


def check_gates_use_shared_harness() -> list[str]:
    """(f) *_gate.py 는 prop_gate 하버스를 통과시켜야 한다."""
    out = []
    for p in sorted(EXPERIMENTS.glob("*_gate.py")):
        if p.stem in GATE_EXEMPT:
            continue
        if _USES_HARNESS.search(p.read_text(encoding="utf-8")):
            continue
        rel = p.relative_to(REPO)
        out.append(
            f"[harness] {rel} — prop_gate 를 안 쓴다. 자체 배터리를 짜면 음성대조·"
            f"비용2배·손안댄창·R분포·fragility 중 빠진 게 드러나지 않는다."
        )
    return out


# (g) 가격 테이블도 storage 정문(read_prices)으로만 읽는다. 직접 SELECT 하면 유니버스를
# 좁히는 WHERE 가 조용히 들어가고, 생존편향은 검증 스택이 못 잡는다 — walk-forward·
# 음성대조·fragility 는 전부 넘겨받은 트레이드만 보기 때문이다(GUARDRAILS §4 공백 2).
_RAW_PRICES = re.compile(r"\bSELECT\b.*\bFROM\s+(daily_bars_adjusted|daily_bars)\b", re.I)
_PRICE_READ_EXEMPT = {
    "src/swing_it/storage.py",        # 정문 자신 + 스키마 문자열
    "src/swing_it/price_adjust.py",   # 조정가 테이블을 *만드는* 쪽 — 정문의 상류
    # 2026-08-16: tests/ 디렉터리 일괄 면제를 파일 단위로 좁혔다. 일괄 면제는
    # 어떤 픽스처 빌더든 생존자 전용 유니버스를 무검사로 스위트에 들일 수 있게 했고,
    # 규칙 (d)가 테스트 파일을 하나씩 명시하는 것과도 비대칭이었다.
    "tests/test_read_prices_guard.py",  # 정문의 가드 자체를 검사 — 원자료를 봐야 한다
    "tests/test_price_adjust.py",       # 조정 로직 검사 — 조정 전/후를 직접 비교한다
}

# 상장 마스터(stocks)와의 INNER JOIN 은 **WHERE 절 없이도** 폐지 종목을 떨군다.
# stocks 는 현재 상장 종목만 담는 마스터이고 폐지분은 delisted_stocks 에 따로 있기
# 때문이다. 즉 지울 조건절이 없고 편향이 조인 자체에 박혀 있어, 읽는 사람 눈에 잘 안
# 띈다. 실제로 이 모양의 로더 3개가 src/ 안에서 생존자 전용 유니버스로 수익률을
# 계산하고 있었다(2026-08-16 에 구현째 삭제). 다시 들어오면 여기서 막는다.
#
# 여러 줄로 쪼갠 SQL 을 잡아야 하므로 줄 단위가 아니라 **파일 전체**에서 찾는다 —
# 규칙 (g)의 한 줄 정규식이 놓치던 지점이 정확히 이것이었다.
_SD_JOIN_STOCKS = re.compile(
    r"\bFROM\s+supply_demand\b[\s\S]{0,200}?\bJOIN\s+stocks\b", re.I
)
_JOIN_RULE_EXEMPT = {
    "scripts/check_guardrails.py",        # 규칙 자신(위 정규식·주석)
    "tests/test_check_guardrails.py",    # 규칙의 회귀 테스트 — 위반 모양을 픽스처로 담는다
}


def check_no_raw_price_select() -> list[str]:
    """(g) storage/price_adjust 밖에서 가격 테이블을 직접 SELECT 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _PRICE_READ_EXEMPT or rel.startswith(".venv/"):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if _RAW_PRICES.search(line):
                out.append(
                    f"[universe] {rel}:{i} — 가격 테이블을 직접 SELECT 했다. "
                    f"swing_it.storage.read_prices() 를 쓸 것 (상장폐지 종목 포함을 "
                    f"로딩 시점에 검사한다)."
                )
    return out


def check_no_survivor_only_join() -> list[str]:
    """(g2) supply_demand 를 상장 마스터(stocks)와 INNER JOIN 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        # 규칙 자신과 그 회귀 테스트는 위반 모양을 문자열로 들고 있어야 한다.
        if rel in _JOIN_RULE_EXEMPT or rel.startswith(".venv/"):
            continue
        m = _SD_JOIN_STOCKS.search(p.read_text(encoding="utf-8"))
        if m is None:
            continue
        line_no = p.read_text(encoding="utf-8")[: m.start()].count("\n") + 1
        out.append(
            f"[universe] {rel}:{line_no} — supply_demand 를 stocks 와 INNER JOIN 했다. "
            f"stocks 는 현재 상장 마스터라 이 조인만으로 폐지 종목이 유니버스에서 "
            f"통째로 빠진다(WHERE 절이 없어도 그렇다). 폐지분이 필요하면 "
            f"delisted_stocks 를 함께 읽거나 조인을 걷어내라."
        )
    return out


# (h) 규칙 (b)의 **역방향** — 모든 VERDICT 에는 그 숫자를 만든 러너가 있어야 한다.
#
# (b)는 "이 코드에 판정이 있나"만 묻는다. 그래서 러너가 삭제되면 판정만 홀로 남고
# 아무도 못 잡는다 — 실제로 `research/logs/minervini_prop/VERDICT.md` 가 601트레이드
# 사전등록 배터리 전문을 싣고 있는데 그걸 만든 코드는 레포에 없다. 재현 불가는 그 자체로
# 죄가 아니지만(기각된 알파의 구현을 지우는 건 정당하다) **선언되지 않은** 재현 불가는
# 독자를 속인다. 그래서 둘 중 하나를 요구한다: 등록된 러너가 실재하거나, VERDICT 가
# 재현 불가를 명시하거나.
VERDICT_RUNNER = {
    "pead_concentrated": "pead_concentrated_gate.py",
    "pullback_prop": "pullback_prop_gate.py",
    "survivorship_bias": "survivorship_bias_gate.py",
    # 게이트 접미사를 안 쓰는 러너도 등록하면 된다 — 규칙의 목적은 파일명 규약이
    # 아니라 "판정에 코드가 붙어 있는가"다.
    "contrarian_retail": "contrarian_validate.py",
    "regime_switch": "regime_switch.py",
    "inst_flow_accel": "inst_flow_accel_gate.py",
    "catalyst_flow_swing": "catalyst_flow_swing_gate.py",
    "news_persistence_swing": "news_persistence_swing.py",
}
# VERDICT 가 러너 없이 존재해도 되는 유일한 조건: 아래 문구로 재현 불가를 선언한다.
_IRREPRODUCIBLE_MARKER = "재현 불가 고지"


# (k) 파이프라인 순서(docs/PIPELINE.md) — 이미지의 11단계를 코드로 강제한다.
#
# 위치가 아니라 **호출**로 본다. research/experiments·research/paper 의 파일은 전부, 그 밖의 research/**·scripts/**
# 파일은 데이터 읽기·백테스트를 부르면 선언해야 한다. 러너는 PIPELINE = ("<study>", <단계>) 를 적고 데이터를 읽기 전에
# require_stage(*PIPELINE) 을 부른다. 러너가 아닌 파일은 PIPELINE_LIBRARY = "<사유>" 를 적는다. "진입점이 없으면 러너가
# 아니다"로 추정하면 진입점을 빼는 것으로 이 규칙을 피할 수 있어서, 선언을 요구하고 선언과 실제가 맞는지 본다.
#
# 면제 목록. **2026-11-02 에 있던 파일만 있고, 줄어들기만 한다** — tests/test_check_guardrails.py 가 고정한다.
# 재현이 바이트 동일해야 하는 옛 러너를 고치지 않으려고 둔다. 새 파일에는 면제가 없다.
PIPELINE_GRANDFATHER = frozenset({
    "research/experiments/catalyst_flow_swing_gate.py",
    "research/experiments/contrarian_bo.py",
    "research/experiments/contrarian_distribution.py",
    "research/experiments/contrarian_selective.py",
    "research/experiments/contrarian_validate.py",
    "research/experiments/inst_flow_accel_gate.py",
    "research/experiments/news_persistence_swing.py",
    "research/experiments/pead_concentrated_gate.py",
    "research/experiments/pead_gate.py",
    "research/experiments/pead_optimizer_b.py",
    "research/experiments/pead_optimizer_b_feasibility.py",
    "research/experiments/pead_optimizer_b_report.py",
    "research/experiments/pead_optimizer_c0.py",
    "research/experiments/pead_optimizer_po2.py",
    "research/experiments/pead_optimizer_po3.py",
    "research/experiments/pead_optimizer_po4.py",
    "research/experiments/pead_optimizer_repaired.py",
    "research/experiments/pead_optimizer_seam.py",
    "research/experiments/pead_refinement.py",
    "research/experiments/prop_feasibility.py",
    "research/experiments/prop_gate.py",
    "research/experiments/prop_swing_common.py",
    "research/experiments/pullback_prop_gate.py",
    "research/experiments/regime_switch.py",
    "research/experiments/slippage_check.py",
    "research/experiments/survivorship_bias_gate.py",
    "research/signals/catalyst_flow.py",
    "research/signals/contrarian_retail.py",
    "scripts/compare_price_adjust.py",
    "scripts/make_hero.py",
})
# 2026-11-02 main 밖에 있는 파일. 이 브랜치에 없는 것이 정상이다. 오늘 적어 두지 않으면 머지 때 "늘어나지 않는다"를 어기게 된다.
PIPELINE_GRANDFATHER_ELSEWHERE = frozenset({
    # feature-combined-strategy
    "research/autoresearch/prepare.py",
    "research/combined/data.py",
    "research/combined/prepare.py",
    "research/experiments/adapt18.py",
    "research/experiments/ana31.py",
    "research/experiments/convergence_posthoc.py",
    "research/experiments/convergence_run.py",
    "research/experiments/core25.py",
    "research/experiments/event_families_build.py",
    "research/experiments/event_sources_build.py",
    "research/experiments/evo_ablate.py",
    "research/experiments/evo_consensus.py",
    "research/experiments/evo_flow_coverage.py",
    "research/experiments/evo_judge.py",
    "research/experiments/evo_report.py",
    "research/experiments/exit_ablate.py",
    "research/experiments/exits_v2_run.py",
    "research/experiments/flow_layer.py",
    "research/experiments/ga2_post.py",
    "research/experiments/ga2_run.py",
    "research/experiments/ga_run.py",
    "research/experiments/indep30.py",
    "research/experiments/indep30_coverage.py",
    "research/experiments/index27.py",
    "research/experiments/index28.py",
    "research/experiments/leader21.py",
    "research/experiments/leader22.py",
    "research/experiments/leader23.py",
    "research/experiments/leader24.py",
    "research/experiments/leader26.py",
    "research/experiments/leaderM_desc.py",
    "research/experiments/live_portfolio_backtest.py",
    "research/experiments/market_mood_run.py",
    "research/experiments/meta20.py",
    "research/experiments/meta20_post.py",
    "research/experiments/meta_gen1_run.py",
    "research/experiments/meta_gen2_run.py",
    "research/experiments/meta_gen3_run.py",
    "research/experiments/meta_gen4_run.py",
    "research/experiments/meta_gen5_run.py",
    "research/experiments/meta_gen6_run.py",
    "research/experiments/minervini_exits_run.py",
    "research/experiments/mood_avail_run.py",
    "research/experiments/news_veto_posthoc.py",
    "research/experiments/pattern_ga_run.py",
    "research/experiments/pattern_ga_view.py",
    "research/experiments/pead19.py",
    "research/experiments/rawcap_rerun.py",
    "research/experiments/recheck29.py",
    "research/experiments/regime_panel_run.py",
    "research/experiments/relax_run.py",
    "research/experiments/rl32.py",
    "research/experiments/sepa_confluence_gate.py",
    "research/experiments/sepa_explore.py",
    "research/experiments/sepa_momentum.py",
    "research/experiments/source_scan.py",
    "research/experiments/vcp_retest_run.py",
    "research/experiments/wedge_d_run.py",
    "research/experiments/wedge_test_run.py",
    "research/paper/index_paper.py",
    "research/paper/leader_paper.py",
    "research/paper/pead_forward_calib.py",
    "research/paper/pead_paper.py",
    "research/signals/sepa.py",
    # younghwan91/ml-forecast-prereg — 게이트 전에 돌고 끝난 러너다. 끝난 러너는 고치지 않는다(--mode render 도 그대로 돈다).
    # 원장은 면제가 아니다: research/logs/ml_forecast 는 PIPELINE.jsonl 을 요구한다.
    "research/experiments/ml_forecast_book.py",
    "research/experiments/ml_forecast_common.py",
    "research/experiments/ml_forecast_data.py",
    "research/experiments/ml_forecast_features.py",
    "research/experiments/ml_forecast_ic.py",
    "research/experiments/ml_forecast_models.py",
    "research/experiments/ml_forecast_nulls.py",
    "research/experiments/ml_forecast_s1_diag.py",
    "research/experiments/ml_forecast_stage0.py",
    "research/experiments/ml_forecast_target.py",
    # PR #18 이 지우는 파일
    "scripts/sector_flow.py",
})
# 2026-11-02 에 있던 로그 폴더 중 닫힌 연구. 여기 없는 research/logs/<폴더> 는 PIPELINE.jsonl 이 있어야 한다.
PIPELINE_LOGS_GRANDFATHER = frozenset({
    "catalyst_flow_swing",
    "contrarian_retail",
    "inst_flow_accel",
    "live_swing",
    "minervini_prop",
    "news_persistence_swing",
    "pead_concentrated",
    "prop_feasibility",
    "prop_swing_search",
    "pullback_prop",
    "regime_switch",
    "sepa_confluence",
    "sepa_explore",
    "survivorship_bias",
})

#: 데이터를 읽는 호출 — research/**·scripts/** 에서 부르면 선언을 요구한다.
_DATA_READS = frozenset({
    "read_prices", "read_earnings", "read_supply_demand", "read_raw_close", "read_investor_trend",
    "read_research_reports", "read_sql", "read_sql_query", "read_sql_table",
})
#: 이 호출이 있으면 9단계 이상이다. 1단계로 선언하면 앞 단계가 없어 게이트가 비어 버린다.
_BACKTEST_NAMES = frozenset({"simulate", "prop_gate", "gate_sim", "run_recipe"})
#: 게이트 앞에 와도 되는 호출. 인자 파싱과 데이터에 닿지 않는 준비뿐이다. --help 는 게이트 없이 떠야 한다.
_BEFORE_GATE_OK = frozenset({
    "ArgumentParser", "add_argument", "add_argument_group", "add_mutually_exclusive_group", "add_subparsers",
    "add_parser", "set_defaults", "exit", "SystemExit", "print", "time", "perf_counter", "monotonic", "now", "today",
    "str", "int", "float", "bool", "len", "split", "strip", "Path", "basicConfig", "getLogger", "setLevel", "Fire",
})
#: 이름에 파싱이 드러나는 호출(build_parser·parse_args·my_parse). 이 이름의 함수가 다른 모듈에서 데이터를 읽으면 못 잡는다.
_PARSER_NAME = re.compile(r"pars|_args$|^args$")


def _load_gate():
    """이 스크립트 옆 src 의 게이트. 표준 라이브러리만 쓰니 패키지가 설치되지 않아도 돈다."""
    src = str(Path(__file__).resolve().parent.parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from swing_it.pipeline import gate  # noqa: PLC0415 — 이 규칙에서만 쓴다

    return gate


def _is_bt_call(name: "str | None") -> bool:
    return bool(name) and (name in _BACKTEST_NAMES or "backtest" in name)


def _is_data_call(name: "str | None") -> bool:
    return name in _DATA_READS or name == "connect" or _is_bt_call(name)


class _Names:
    """import 별칭을 풀어 호출 이름을 돌려준다. ``from x import read_prices as rp``·``rp2 = read_prices`` 를 본다."""

    def __init__(self, tree: ast.Module) -> None:
        self.imported: dict[str, tuple[str, str]] = {}      # 지역 이름 → (모듈, 원래 이름)
        self.modules: dict[str, str] = {}                   # 지역 이름 → 모듈
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    self.imported[a.asname or a.name] = (node.module or "", a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    self.modules[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        self.assigned: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.assigned[t.id] = node.value.id

    def name(self, call: ast.Call) -> "str | None":
        f = call.func
        if isinstance(f, ast.Attribute):
            return f.attr
        if isinstance(f, ast.Name):
            local = self.assigned.get(f.id, f.id)
            return self.imported.get(local, ("", local))[1]
        return None

    def is_pipeline_gate(self, call: ast.Call) -> bool:
        """정말 swing_it.pipeline 의 require_stage 인가. 같은 이름의 지역 함수는 게이트가 아니다."""
        f = call.func
        if isinstance(f, ast.Name):
            module, original = self.imported.get(f.id, ("", f.id))
            return original == "require_stage" and module.startswith("swing_it.pipeline")
        if isinstance(f, ast.Attribute) and f.attr == "require_stage":
            base = f.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                module = self.modules.get(base.id) or self.imported.get(base.id, ("", ""))[0]
                return module.startswith("swing_it")
        return False


def _is_main_guard(node: ast.AST) -> bool:
    """``__name__`` 을 비교하는 if — 어느 쪽에 적어도 같다."""
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
        return False
    sides = [node.test.left, *node.test.comparators]
    return any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)


def _import_time(tree: ast.Module):
    """import 할 때 실행되는 노드: 모듈·클래스 본문, 기본 인자, 데코레이터. 함수 본문과 __main__ 블록은 뺀다."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from node.args.defaults
            yield from (d for d in node.args.kw_defaults if d is not None)
            yield from node.decorator_list
        elif isinstance(node, ast.ClassDef):
            yield from node.decorator_list
            stack.extend(node.body)
        elif _is_main_guard(node):
            stack.extend(node.orelse)
        elif isinstance(node, (ast.stmt, ast.ExceptHandler)):
            for child in ast.iter_child_nodes(node):
                stack.append(child)
        else:
            yield node


def _calls_in(node: ast.AST):
    """노드 안에서 **이 자리에서 실행되는** 호출. 안쪽 함수 정의의 본문은 뺀다(기본 인자는 포함)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Call):
            yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(n.args.defaults)
            stack.extend(d for d in n.args.kw_defaults if d is not None)
            stack.extend(n.decorator_list)
        elif isinstance(n, ast.Lambda):
            stack.extend(n.args.defaults)
        else:
            stack.extend(ast.iter_child_nodes(n))


def _functions(tree: ast.Module) -> dict:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _reaches_data(body, funcs: dict, names: _Names, seen: "set | None" = None) -> "str | None":
    """이 문장들에서(지역 함수를 따라가며) 데이터·백테스트 호출에 닿으면 그 이름."""
    seen = set() if seen is None else seen
    for stmt in body:
        for call in _calls_in(stmt):
            name = names.name(call)
            if _is_data_call(name):
                return name
            if name in funcs and name not in seen:
                seen.add(name)
                hit = _reaches_data(funcs[name].body, funcs, names, seen)
                if hit:
                    return hit
    return None


def _default_arg_problem(tree: ast.Module, names: _Names) -> "str | None":
    """기본 인자는 import 때 돈다. ``def main(df=load())`` 의 ``load`` 는 이름으로 못 가리니 소문자 호출을 전부 막는다.

    대문자로 시작하는 호출(``Flags()``·``Path("x")``)과 인자 파싱·시각은 설정 값으로 보고 둔다.
    """
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        for default in (*fn.args.defaults, *(d for d in fn.args.kw_defaults if d is not None)):
            for call in _calls_in(default):
                name = names.name(call)
                if not (name and (name[0].isupper() or name in _BEFORE_GATE_OK)):
                    return f"기본 인자에서 {name} 를 부른다 — import 때 돌아 게이트보다 먼저다. 함수 안으로 옮긴다."
    return None


def _import_time_problem(tree: ast.Module, names: _Names) -> "str | None":
    """import 때 데이터를 읽거나 지역 함수로 데이터에 닿으면 게이트가 소용없다."""
    problem = _default_arg_problem(tree, names)
    if problem:
        return problem
    funcs = _functions(tree)
    for node in _import_time(tree):
        for call in _calls_in(node):
            name = names.name(call)
            if _is_data_call(name):
                return f"import 때(모듈 수준·기본 인자·데코레이터) {name} 를 부른다 — 게이트보다 먼저 돈다."
            if name in funcs and _reaches_data(funcs[name].body, funcs, names, {name}):
                return f"import 때 {name}() 를 부른다 — 그 안에서 데이터에 닿는다."
    return None


def _gate_expr(stmt: ast.stmt) -> "ast.Call | None":
    if isinstance(stmt, ast.Expr):
        value = stmt.value
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        value = stmt.value
    else:
        return None
    return value if isinstance(value, ast.Call) else None


def _is_gate_call(stmt: ast.stmt, names: _Names) -> bool:
    """``require_stage(*PIPELINE)`` 또는 ``PIPELINE[0], PIPELINE[1]`` 인 문장 하나. 조건문 안·다른 인자는 치지 않는다."""
    call = _gate_expr(stmt)
    if call is None or call.keywords or not names.is_pipeline_gate(call):
        return False
    args = call.args
    if len(args) == 1 and isinstance(args[0], ast.Starred):
        return isinstance(args[0].value, ast.Name) and args[0].value.id == "PIPELINE"
    return [ast.unparse(a) for a in args] == ["PIPELINE[0]", "PIPELINE[1]"]


def _before_gate_ok(call: ast.Call, name: "str | None") -> bool:
    if name is None:
        return False
    if name == "run":
        return ast.unparse(call.func) == "asyncio.run"
    return name in _BEFORE_GATE_OK or bool(_PARSER_NAME.search(name))


def _gate_comes_first(body, funcs: dict, names: _Names, seen: set) -> "bool | None":
    """문장을 앞에서부터 본다. 참 = 게이트에 먼저 닿는다. 거짓 = 그 전에 다른 호출이 있다. None = 게이트 없이 끝났다.

    문장 안의 호출을 전부 본다 — ``main(read_prices(c))`` 처럼 인자가 먼저 돈다. 지역 함수는 이름과 상관없이 따라간다.
    ``fire.Fire(main)`` 처럼 넘기기만 해도 부른 것으로 친다.
    """
    for stmt in body:
        if _is_gate_call(stmt, names):
            return True
        locals_used: list[str] = []
        for call in _calls_in(stmt):
            name = names.name(call)
            if name in funcs:
                locals_used.append(name)
            elif not _before_gate_ok(call, name):
                return False
            for arg in (*call.args, *(k.value for k in call.keywords)):
                if isinstance(arg, ast.Name) and arg.id in funcs:
                    locals_used.append(arg.id)
        found = False
        for name in locals_used:
            if name in seen:
                continue
            seen.add(name)
            result = _gate_comes_first(funcs[name].body, funcs, names, seen)
            if result is False:
                return False
            found = found or result is True
        if found:
            return True
    return None


def _stage_literal(node: ast.AST, tree: ast.Module):
    if isinstance(node, ast.Name):
        value = _module_constant(tree, node.id)
        if value is not None:
            node = value
    if isinstance(node, ast.Constant):
        return node.value
    return node.attr if isinstance(node, ast.Attribute) else None


def _module_constant(tree: ast.Module, name: str):
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    return None


def _has_main_guard(tree: ast.Module) -> bool:
    """모듈 수준 어디든(try·with 안 포함) ``__name__`` 비교가 있는가. 함수 본문은 보지 않는다."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if _is_main_guard(node):
            return True
        if isinstance(node, (ast.stmt, ast.ExceptHandler)) and not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))
    return False


def _library_runs_at_import(tree: ast.Module, names: _Names) -> bool:
    """라이브러리인 척하는 러너: import 때 지역 함수 main 을 부른다(sys.argv 조건 등)."""
    funcs = _functions(tree)
    return any(names.name(call) == "main" and "main" in funcs
               for node in _import_time(tree) for call in _calls_in(node))


def _pipeline_problem(tree: ast.Module, gate) -> "str | None":
    """파일 하나의 선언 위반 사유. 문제 없으면 ``None``."""
    names = _Names(tree)
    library = _module_constant(tree, "PIPELINE_LIBRARY")
    declared = _module_constant(tree, "PIPELINE")
    problem = _import_time_problem(tree, names)
    if library is not None:
        if not (isinstance(library, ast.Constant) and isinstance(library.value, str) and library.value.strip()):
            return "PIPELINE_LIBRARY 는 사유 문자열이어야 한다."
        if _has_main_guard(tree) or _library_runs_at_import(tree, names):
            return "PIPELINE_LIBRARY 인데 진입점이 있다 — 러너면 PIPELINE 을 선언한다."
        return problem
    if declared is None:
        return ('PIPELINE = ("<study>", <단계>) 선언이 없다. 러너가 아니면 PIPELINE_LIBRARY = "<사유>" 를 적는다. '
                "면제 목록은 늘리지 않는다.")
    if not (isinstance(declared, ast.Tuple) and len(declared.elts) == 2):
        return 'PIPELINE 은 ("<study>", <단계>) 튜플 리터럴이어야 한다.'
    study, stage = declared.elts
    if not (isinstance(study, ast.Constant) and isinstance(study.value, str) and gate._STUDY.match(study.value)):
        return "PIPELINE 의 study 는 소문자·숫자·밑줄 문자열이어야 한다."
    try:
        parsed = gate.Stage.parse(_stage_literal(stage, tree))
    except ValueError:
        return f"PIPELINE 의 단계가 1~11 또는 Stage.<이름> 이 아니다: {ast.unparse(stage)}"
    called = {names.name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    backtests = sorted(n for n in called if _is_bt_call(n))
    if parsed < gate.Stage.BACKTEST and backtests:
        return (f"{backtests} 를 부르는데 {parsed.label} 로 선언했다 — 백테스트는 9단계다. "
                "앞 단계로 적으면 게이트가 비어 버린다.")
    if problem:
        return problem
    entry = next((n.body for n in tree.body if _is_main_guard(n)), None)
    if entry is None:
        return "PIPELINE 을 선언했는데 __main__ 진입점이 없다 — 러너가 아니면 PIPELINE_LIBRARY 를 적는다."
    if _gate_comes_first(entry, _functions(tree), names, set()) is not True:
        return ("데이터를 읽기 전에 require_stage(*PIPELINE) 을 불러야 한다 — 진입점의 맨 위 문장으로, 조건문 없이, "
                "swing_it.pipeline 에서 import 한 것으로. 앞에는 인자 파싱만 온다.")
    return None


def _touches_data(tree: ast.Module) -> bool:
    names = _Names(tree)
    return any(n in _DATA_READS or _is_bt_call(n)
               for n in (names.name(c) for c in ast.walk(tree) if isinstance(c, ast.Call)))


def check_pipeline_declared() -> list[str]:
    """(k) research/experiments·research/paper 전부, 그 밖의 research/**·scripts/** 는 데이터·백테스트를 부르면 선언."""
    gate = _load_gate()
    out = []
    for root in (REPO / "research", REPO / "scripts"):
        if not root.is_dir():
            continue
        for p in _iter_py(root):
            rel = p.relative_to(REPO).as_posix()
            if p.name == "__init__.py" or rel in PIPELINE_GRANDFATHER or rel in PIPELINE_GRANDFATHER_ELSEWHERE:
                continue
            always = rel.startswith(("research/experiments/", "research/paper/"))
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as exc:                  # 파싱 실패는 침묵시키지 않는다
                out.append(f"[pipeline] {rel} — 파싱 실패: {exc}")
                continue
            if not always and not _touches_data(tree) and _module_constant(tree, "PIPELINE") is None:
                continue
            problem = _pipeline_problem(tree, gate)
            if problem:
                out.append(f"[pipeline] {rel} — {problem} (docs/PIPELINE.md)")
    return out


def check_pipeline_ledgers() -> list[str]:
    """(k) 원장 체인·행 규칙·최신 증거 해시. research/logs/<폴더> 는 면제가 아니면 원장이 있어야 한다."""
    gate = _load_gate()
    out = [f"[pipeline] {p}" for p in gate.verify(root=REPO)]
    if LOGS.is_dir():
        for d in sorted(p for p in LOGS.iterdir() if p.is_dir() and p.name != "__pycache__"):
            if d.name not in PIPELINE_LOGS_GRANDFATHER and not (d / gate.LEDGER_NAME).exists():
                out.append(f"[pipeline] research/logs/{d.name} 에 {gate.LEDGER_NAME} 가 없다 — "
                           f"실제 상태를 python -m swing_it.pipeline record 로 적는다(docs/PIPELINE.md).")
    return out


def check_pipeline_ledgers_append_only() -> list[str]:
    """(k) 기준(기본 origin/main)의 원장은 지금 원장의 앞부분이어야 한다.

    해시 체인은 끝 줄 잘라내기를 못 잡는다. fail 한 줄을 지우면 게이트가 열린다. CI 는 기준을 받아 두므로 기준이 없으면
    실패다(``CI`` 환경변수). 로컬에서 기준이 없으면 볼 수 없으니 넘어간다. 기준은 ``PIPELINE_BASE_REF`` 로 바꾼다.
    """
    import subprocess  # noqa: PLC0415 — 이 규칙에서만 쓴다

    base = os.environ.get("PIPELINE_BASE_REF", "origin/main")

    def _git(*args: str):
        return subprocess.run(("git", *args), cwd=REPO, capture_output=True, text=True, timeout=30)

    try:
        if _git("rev-parse", "--git-dir").returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        return []
    if _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode != 0:
        if os.environ.get("CI"):
            return [f"[pipeline] 기준 {base} 가 없어 원장 잘라내기를 볼 수 없다 — CI 는 fetch-depth: 0 으로 받는다(.github/workflows/ci.yml)."]
        return []
    out = []
    listed = _git("ls-tree", "-r", "--name-only", base, "research/logs").stdout.splitlines()
    for rel in (p for p in listed if p.endswith("/PIPELINE.jsonl")):
        path = REPO / rel
        if not path.exists():
            out.append(f"[pipeline] {rel} — {base} 에 있는 원장이 사라졌다. 원장은 덧붙이기만 한다.")
        elif not path.read_text(encoding="utf-8").startswith(_git("show", f"{base}:{rel}").stdout):
            out.append(f"[pipeline] {rel} — {base} 의 줄이 앞부분에 그대로 없다(잘라내기·고치기). 원장은 덧붙이기만 한다.")
    return out


# (i) 커밋 신원이 레포 설정과 다르면 실패.
#
# 커밋 시 `git -c user.email=...` 로 신원을 덮어쓰면 의도치 않은 주소가 공개
# 이력에 박히고, 이력은 되돌리기 어렵다. "푸시 전에 확인하는 습관" 은 부탁이지
# 규칙이 아니다(§0) — 레포 설정을 상속하게 두고 어긋나면 여기서 막는다.
def check_commit_identity() -> list[str]:
    """(i) 최근 커밋의 author/committer 이메일이 레포 설정과 같은가."""
    import subprocess  # noqa: PLC0415 — 이 규칙에서만 쓴다

    def _git(*args: str) -> str:
        try:
            return subprocess.run(("git", *args), cwd=REPO, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 — git 이 없거나 저장소가 아니면 검사 생략
            return ""

    def _ok_identity(addr: str, want: str) -> bool:
        """레포 설정이거나, **GitHub 이 스스로 찍는 주소**면 통과.

        PR 을 웹에서 머지하면 GitHub 이 author 를 계정의 프라이버시 주소
        (`…@users.noreply.github.com`), committer 를 `noreply@github.com` 으로
        찍는다. 사람이 신원을 덮어쓴 게 아니라 GitHub 이 만든 커밋이다.
        이걸 안 봐주면 **PR 을 머지할 때마다 CI 가 빨개진다.**

        허용 범위는 이 둘로 좁힌다 — 이 규칙이 막으려는 것은 의도치 않은
        주소(특히 사내 메일)가 공개 이력에 박히는 것이고, GitHub noreply 는
        구성상 그럴 수 없다. 임의 도메인을 열어주면 규칙이 사라진다.
        """
        return (addr == want
                or addr == "noreply@github.com"
                or addr.endswith("@users.noreply.github.com"))

    want = _git("config", "--get", "user.email")
    if not want:
        return []                     # 설정이 없으면 비교할 기준이 없다
    out = _git("log", "-30", "--format=%h %ae %ce")
    bad = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        sha, ae, ce = parts
        if _ok_identity(ae, want) and _ok_identity(ce, want):
            continue
        if True:
            bad.append(
                f"[identity] {sha} — 커밋 신원이 레포 설정과 다르다 "
                f"(author={ae} committer={ce}, 설정={want}). "
                f"`git -c user.email=...` 로 덮어쓰지 말 것 — 레포 설정을 그대로 쓴다."
            )
    return bad


def check_verdicts_have_runner() -> list[str]:
    """(h) 각 research/logs/<alpha>/VERDICT.md 에 러너가 등록·실재하거나 재현불가 선언이 있어야 한다."""
    out = []
    if not LOGS.is_dir():
        return out
    for verdict in sorted(LOGS.glob("*/VERDICT.md")):
        alpha = verdict.parent.name
        runner = VERDICT_RUNNER.get(alpha)
        if runner and (EXPERIMENTS / runner).exists():
            continue
        if _IRREPRODUCIBLE_MARKER in verdict.read_text(encoding="utf-8"):
            continue
        rel = verdict.relative_to(REPO)
        detail = (
            f"등록된 러너 {runner} 가 없다" if runner
            else "VERDICT_RUNNER 에 등록되지 않았다"
        )
        out.append(
            f"[orphan-verdict] {rel} — {detail}. 판정을 만든 러너를 등록하거나, "
            f"러너를 지웠다면 VERDICT 에 '{_IRREPRODUCIBLE_MARKER}'를 적어 "
            f"재현 불가를 명시하라(선언 없는 재현 불가는 독자를 속인다)."
        )
    return out


# (j) ML 예측(PREREG §3·정정 ML-1 ⑯): 피처는 폐지 종목까지 같은 정의인 원천만 쓴다. ``credit_balance``·
# ``short_selling``·``program_trading`` 백필은 상장 생존 종목뿐이고, ``supply_demand`` 는 상장=키움·폐지=네이버로 원천이
# 갈려 정의 차이가 폐지 여부와 상관된다(결측·정의가 미래 폐지와 상관되면 누출이다). 기관 세부 열도 폐지 종목은 NULL 이다.
_ML_FORBIDDEN = re.compile(
    r"\b(credit_balance|short_selling|program_trading|supply_demand|read_supply_demand"
    r"|INSTITUTION_DETAIL_COLS|fnnc_invt|insrnc|invtrt)\b")


def check_ml_forecast_sources() -> list[str]:
    """(j) research/experiments/ml_forecast_*.py 가 금지 원천을 언급하면 위반(주석 줄 제외)."""
    out = []
    if not EXPERIMENTS.is_dir():
        return out
    for p in sorted(EXPERIMENTS.glob("ml_forecast_*.py")):
        rel = p.relative_to(REPO).as_posix()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            m = _ML_FORBIDDEN.search(line)
            if m:
                out.append(
                    f"[ml-source] {rel}:{i} — {m.group(1)} 를 읽었다. ML 예측 피처는 naver_investor_trend·"
                    f"naver_research_reports 처럼 폐지 종목까지 같은 정의인 원천만 쓴다(PREREG §3).")
    return out


def main() -> int:
    violations: list[str] = []
    violations += check_src_no_research_import()
    violations += check_gates_have_verdict()
    violations += check_no_literal_verdict()
    violations += check_no_raw_earnings_select()
    violations += check_gates_record_trials()
    violations += check_gates_use_shared_harness()
    violations += check_no_raw_price_select()
    violations += check_no_survivor_only_join()
    violations += check_verdicts_have_runner()
    violations += check_pipeline_declared()
    violations += check_pipeline_ledgers()
    violations += check_pipeline_ledgers_append_only()
    violations += check_commit_identity()
    violations += check_ml_forecast_sources()

    if violations:
        print("가드레일 린트 실패 — 위반 %d건:\n" % len(violations))
        for v in violations:
            print("  " + v)
        print("\n(GUARDRAILS.md §5 — 경계·판정·리포터 규칙)")
        return 1

    print("가드레일 린트 통과 — 경계·VERDICT·리포터·파이프라인 위반 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
