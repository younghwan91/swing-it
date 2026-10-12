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
  (j) 애프터마켓 오염 열 — ``supply_demand`` 의 close·flu_rt·acc_trde_qty 를 정문 밖에서 직접 SELECT 하면 실패.
  (k) 주식수 as-of — ``shares_outstanding_history`` 를 정문(storage.read_shares) 밖에서 직접 SELECT 하면 실패.
  (l) 실적 as-of — ``avail_date`` 를 as-of 키로 쓰는 자리(``avail_col="avail_date"``·merge_asof 키)는 ``# legacy-asof: <이유>``
      표시가 있어야 하고, ``avail_col`` 매개변수는 기본값을 가질 수 없다.
  (m) 분봉 시간대 — ``minute_bars`` 가 나오는 파일에서 ``at time zone``·``tz_localize``·``tz_convert``·``utc=True`` 금지.
  (n) 거래대금 단위 — ``ADV_*``·``adv_floor``·``TV_FLOOR`` 류 이름에 맨 숫자 리터럴(0 제외)을 넣으면 실패(swing_it.units.eok 등).
  (o) 공식 지수 — ``sector_index`` 를 정문(storage.read_index) 밖에서 직접 SELECT·JOIN 하면 실패.

**이 린트 자체는 tests/test_check_guardrails.py 가 검사한다.** 각 규칙에 위반을 주입해
실제로 실패하는지 확인한다 — 규칙이 죽어도 초록인 상태를 실제로 겪었기 때문이다((e) 참조).

무의존·표준 라이브러리만. exit 0 = 위반 없음, exit 1 = 위반.
실행: ``python scripts/check_guardrails.py``
"""

from __future__ import annotations

import ast
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



# --- SQL 문자열 수집기((d)(g) 공용) ----------------------------------------------------
# 2026-10-12: (d)(g) 가 **한 줄 정규식**이라 SELECT 와 FROM 이 다른 줄이면 못 잡았다 — 삼중따옴표 SQL,
# 암묵 연결("select ..."\n "from daily_bars"), ``+`` 연결 모두 통과했다. 이제 ast 로 **문자열 식 하나**를 통째로
# 접어(f-string 의 치환부는 ``{}``) 그 안에서 찾는다. 주석은 원래 안 보이고, 독스트링·맨 문자열 문장은 산문이라 뺀다.

def _fold_str(node: ast.AST) -> "str | None":
    """문자열 식을 한 문자열로 접는다. 문자열이 아니면 None. f-string 치환·비문자열 피연산자는 ``{}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                       for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _fold_str(node.left), _fold_str(node.right)
        if a is None and b is None:
            return None
        return (a if a is not None else "{}") + (b if b is not None else "{}")
    return None


def _sql_strings(text: str) -> "list[tuple[int, str]]":
    """파일 안의 (시작 줄, 접은 문자열) — 가장 바깥 문자열 식만, 독스트링·맨 문자열 문장 제외. 파싱 실패면 ValueError."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(str(exc)) from exc
    inner: set[int] = set()
    prose: set[int] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.JoinedStr)):
            prose.add(id(n.value))                      # 독스트링·맨 문자열 문장 = 산문
        if isinstance(n, (ast.BinOp, ast.JoinedStr)):
            for c in ast.iter_child_nodes(n):
                inner.add(id(c))
            if isinstance(n, ast.JoinedStr):
                for c in ast.walk(n):
                    if c is not n:
                        inner.add(id(c))
    out = []
    for n in ast.walk(tree):
        if id(n) in inner or id(n) in prose:
            continue
        if isinstance(n, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            s = _fold_str(n)
            if s is not None:
                out.append((n.lineno, s))
    return out


def _sql_hits(text: str, pattern: "re.Pattern[str]") -> "list[tuple[int, str]]":
    """``pattern`` 에 맞는 SQL 문자열의 (줄, 맞은 조각). 줄 = 문자열 시작 줄 + 맞은 곳까지의 줄바꿈 수."""
    hits = []
    for line, s in _sql_strings(text):
        for m in pattern.finditer(s):
            hits.append((line + s[: m.start()].count("\n"), " ".join(m.group(0).split())[:80]))
    return hits


# (d) earnings 는 정정공시가 새 행으로 쌓이는 버전 테이블이라(PK 에 knowledge_date),
# 그냥 SELECT 하면 (code, period)당 행이 여럿이 되어 하류 패널이 조용히 중복된다.
# storage.read_earnings() 가 as-of 로 한 버전만 고르는 유일한 정문이다.
# **문자열 리터럴 안의** SELECT … FROM earnings 를 잡는다(여러 줄 SQL 포함, 위 수집기) — 주석·독스트링의
# "from earnings drift" 같은 산문은 대상이 아니다. 규칙이 소음이 되면 아무도 안 본다.
_RAW_EARNINGS = re.compile(r"\bSELECT\b[\s\S]*?\bFROM\s+earnings\b", re.IGNORECASE)
_EARNINGS_READ_EXEMPT = {
    "src/swing_it/storage.py",          # 정문 자신 + 스키마 문자열
    "tests/test_earnings_asof.py",      # 정문의 테스트 — 버전이 실제로 쌓이는지 직접 확인해야 한다
    "tests/test_check_guardrails.py",   # 린트의 회귀 테스트 — 위반 모양을 픽스처 문자열로 담는다
}


def _scan_sql(exempt: "set[str]", pattern: "re.Pattern[str]", tag: str, msg: str) -> list[str]:
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in exempt or rel.startswith(".venv/"):
            continue
        try:
            hits = _sql_hits(p.read_text(encoding="utf-8"), pattern)
        except ValueError as exc:                       # 파싱 실패는 침묵시키지 않는다
            out.append(f"[{tag}] {rel} — 파싱 실패: {exc}")
            continue
        out += [f"[{tag}] {rel}:{i} — {msg} ({frag})" for i, frag in hits]
    return out


def check_no_raw_earnings_select() -> list[str]:
    """(d) storage 밖에서 earnings 를 직접 SELECT 하면 위반."""
    return _scan_sql(_EARNINGS_READ_EXEMPT, _RAW_EARNINGS, "earnings",
                     "earnings 를 직접 SELECT 했다. swing_it.storage.read_earnings(con, asof=...) 를 쓸 것 "
                     "(정정공시 버전이 중복 행으로 새어나온다)")


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
# 2026-10-12: 여러 줄 SQL 도(위 수집기), ``JOIN daily_bars`` 도 잡는다.
# daily_bars_raw·daily_bars_adj_v2(재구성 원·수정 일봉)도 같은 가격표다 — 이름으로 박혀 있으면 잡는다(변수로 넘긴 표 이름은 못 본다).
_RAW_PRICES = re.compile(r"\bSELECT\b[\s\S]*?\b(?:FROM|JOIN)\s+(daily_bars(?:_adjusted|_raw|_adj_v2)?)\b", re.I)
_PRICE_READ_EXEMPT = {
    "src/swing_it/storage.py",        # 정문 자신 + 스키마 문자열
    "src/swing_it/price_adjust.py",   # 조정가 테이블을 *만드는* 쪽 — 정문의 상류
    # 2026-08-16: tests/ 디렉터리 일괄 면제를 파일 단위로 좁혔다. 일괄 면제는
    # 어떤 픽스처 빌더든 생존자 전용 유니버스를 무검사로 스위트에 들일 수 있게 했고,
    # 규칙 (d)가 테스트 파일을 하나씩 명시하는 것과도 비대칭이었다.
    "tests/test_read_prices_guard.py",  # 정문의 가드 자체를 검사 — 원자료를 봐야 한다
    "tests/test_check_guardrails.py",   # 린트의 회귀 테스트 — 위반 모양을 픽스처 문자열로 담는다
    "tests/test_price_adjust.py",       # 조정 로직 검사 — 조정 전/후를 직접 비교한다
    # 2026-08-27: 무결성 감사기. 이 규칙이 막으려는 것은 "유니버스를 조용히 좁히는
    # 로더"인데, 이 파일은 유니버스를 **만들지 않는다** — count(*) 로 중복·OHLC
    # 불변식·단위 정합만 센다. 수익률 시계열을 만드는 코드가 이 파일에 들어오면
    # 면제를 거둬야 한다.
    "scripts/verify_report.py",
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
    # 리포트 **검증기**. 이 규칙은 "심사에 쓰는 유니버스가 조용히 좁아지는 것"을
    # 막는 것이고, 여기 조인은 그 반대 목적이다 — `scripts/sector_flow.py` 가
    # 만든 값을 대조하려면 **그것과 같은 유니버스**여야 한다(그쪽은 pandas
    # `merge(how="inner")` 라 이 정규식에 안 걸린다). 다른 유니버스로 재계산하면
    # 전 칸이 불일치로 뜨고 검사가 무력해진다.
    #
    # 좁아진다는 사실 자체는 숨기지 않는다 — `layer_f` 가 그 폭을 **재서 화면에
    # 적는다**("네이버 폐지 백필이 유니버스 밖에 남긴 종목 수"). 관측 화면의
    # 한계이지 심사 성적이 아니므로 실패로 만들지 않고 수치로 남긴다.
    "scripts/verify_report.py",
}


def check_no_raw_price_select() -> list[str]:
    """(g) storage/price_adjust 밖에서 가격 테이블을 직접 SELECT 하면 위반."""
    return _scan_sql(_PRICE_READ_EXEMPT, _RAW_PRICES, "universe",
                     "가격 테이블을 직접 SELECT 했다. swing_it.storage.read_prices() 를 쓸 것 "
                     "(상장폐지 종목 포함을 로딩 시점에 검사한다)")


# (j) 2026-09-14 부터 supply_demand 의 close·flu_rt·acc_trde_qty 에 KRX 애프터마켓(16:00~20:00) 체결이 섞였다. 정문
# storage.read_supply_demand 가 기본으로 정규장 값으로 고치고(원값은 session="raw" 명시), 원 종가 정문은 krx_session.read_raw_close 다.
# 직접 SELECT 는 그 보정을 조용히 건너뛴다 — 가격표 (g) 와 같은 이유로 막는다. 순매매 수량 열만 읽는 SELECT 는 대상이 아니다.
_SD_SELECT = re.compile(r"\bSELECT\b(?P<sel>(?:(?!\bSELECT\b)[\s\S])*?)\bFROM\s+supply_demand\b", re.I)
_SD_TAINTED = re.compile(r"(?<![\w.])(?:\w+\.)?(close|flu_rt|acc_trde_qty)\b|\*", re.I)
_SD_COLS_EXEMPT = {
    "src/swing_it/storage.py",            # 정문 자신
    "src/swing_it/krx_session.py",        # 원 종가·항등식 정문(정규장 종가를 여기서 만든다)
    "scripts/check_guardrails.py",        # 규칙 자신
    "tests/test_check_guardrails.py",     # 규칙의 회귀 테스트 — 위반 모양을 픽스처로 담는다
    # 리포트 검증기 — sector_flow 가 만든 값(원값 고정, session="raw")을 같은 원천으로 재계산해 대조한다.
    # 보정본으로 재계산하면 전 칸이 불일치로 뜬다(위 (g2) 면제와 같은 이유).
    "scripts/verify_report.py",
}


def check_no_raw_supply_close() -> list[str]:
    """(j) 정문 밖에서 supply_demand 의 close·flu_rt·acc_trde_qty 를 직접 SELECT 하면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _SD_COLS_EXEMPT or rel.startswith(".venv/"):
            continue
        try:
            strings = _sql_strings(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            out.append(f"[aftermarket] {rel} — 파싱 실패: {exc}")
            continue
        for line, sql in strings:
            for m in _SD_SELECT.finditer(sql):
                bad = _SD_TAINTED.search(m.group("sel"))
                if bad is None:
                    continue
                out.append(
                    f"[aftermarket] {rel}:{line + sql[: m.start()].count(chr(10))} — supply_demand 의 "
                    f"{bad.group(0)} 를 직접 SELECT 했다. 2026-09-14~ 값에 KRX 애프터마켓(16:00~20:00) 체결이 섞였다 — "
                    f"swing_it.storage.read_supply_demand(...) 정문을 쓸 것(원값이 필요하면 session=\"raw\" 명시), "
                    f"원 종가는 swing_it.krx_session.read_raw_close."
                )
    return out


# (k) 상장주식수는 정문 storage.read_shares 로만 읽는다. dart 행은 기준일(date)보다 중앙값 82일 늦게 알려지는데(knowledge_date),
# 직접 SELECT 하는 코드는 date 로 as-of 를 걸어 석 달 미래 정보를 썼다(21~26세대 시총). 정문은 asof = coalesce(knowledge_date, date)
# 열을 주고 0 이하 주식수를 뺀다.
_RAW_SHARES = re.compile(r"\b(?:FROM|JOIN)\s+shares_outstanding_history\b", re.I)
_SHARES_READ_EXEMPT = {
    "src/swing_it/storage.py",            # 정문 자신 + 스키마 + shares_backfill_pending(존재 여부만 센다)
    "scripts/check_guardrails.py",        # 규칙 자신
    "tests/test_check_guardrails.py",     # 규칙의 회귀 테스트
    # 무결성 감사기 — 0 이하 행 수·커버리지를 원 표에서 센다(유니버스·시총을 만들지 않는다; (g) 면제와 같은 이유).
    "scripts/verify_report.py",
}


def check_no_raw_shares_select() -> list[str]:
    """(k) storage 밖에서 shares_outstanding_history 를 직접 SELECT·JOIN 하면 위반."""
    return _scan_sql(_SHARES_READ_EXEMPT, _RAW_SHARES, "shares",
                     "shares_outstanding_history 를 직접 읽었다. swing_it.storage.read_shares(con, with_asof=True) 를 쓸 것 — "
                     "as-of 키는 coalesce(knowledge_date, date)(dart 행은 기준일보다 석 달쯤 늦게 알려진다), 0 이하는 결측")


# (l) earnings.avail_date 는 제출 마감일 고정값이라 정기보고서 17.7% 가 그보다 늦게 공시됐다 — as-of 키로 쓰면 미래 정보다(정정 G19-1).
# 올바른 키는 read_earnings(with_known=True) 의 known_date(max(avail_date, 정기 첫 rcept_dt) 다음 거래일). 옛 숫자를 재현해야 하는
# 자리는 막지 않고 **표시**를 요구한다 — 표시 없는 새 사용이 조용히 들어오는 것을 막는 게 목적이다. 기본값 avail_col="avail_date" 는
# 표시 없이 옛 키를 퍼뜨리는 통로라 금지한다. tests/ 는 그 동작 자체를 검사하므로 (l1) 표시 요구에서 뺀다.
_LEGACY_ASOF = "# legacy-asof:"
_ASOF_EXEMPT = {
    "src/swing_it/storage.py",            # 정문(known_date 를 만든다)
    "src/swing_it/features/eps_asof.py",  # 같은 규칙의 배열판(max(avail_date, rcept_dt))
    "scripts/check_guardrails.py",
    "tests/test_check_guardrails.py",
}


def _is_str(node, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def check_avail_date_asof() -> list[str]:
    """(l) avail_date as-of 키 사용엔 # legacy-asof: 표시, avail_col 기본값 금지."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _ASOF_EXEMPT or rel.startswith(".venv/"):
            continue
        text = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            out.append(f"[asof] {rel} — 파싱 실패: {exc}")
            continue
        lines = text.splitlines()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                pos = a.posonlyargs + a.args
                pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
                for arg, d in pairs:
                    if arg.arg == "avail_col" and d is not None:
                        out.append(f"[asof] {rel}:{n.lineno} — {n.name}() 의 avail_col 에 기본값이 있다. 필수로 두고 호출부가 "
                                   f"고르게 하라(avail_date 는 미래 정보 — read_earnings(with_known=True) 의 known_date).")
            if rel.startswith("tests/") or not isinstance(n, ast.Call):
                continue
            kw = {k.arg: k.value for k in n.keywords if k.arg}
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            hit = _is_str(kw.get("avail_col"), "avail_date") or (
                name == "merge_asof" and any(_is_str(kw.get(k), "avail_date") for k in ("on", "left_on", "right_on")))
            if not hit:
                continue
            span = lines[max(n.lineno - 2, 0): n.end_lineno]
            if any(_LEGACY_ASOF in ln for ln in span):
                continue
            out.append(f"[asof] {rel}:{n.lineno} — avail_date 를 as-of 키로 썼다(제출 마감일 고정값 — 정기보고서 17.7% 가 더 늦게 "
                       f"공시). read_earnings(with_known=True) 의 known_date 를 쓰거나, 옛 숫자 재현이면 "
                       f"'{_LEGACY_ASOF} <이유>' 를 단다.")
    return out


# (m) minute_bars.ts 는 시간대 없는 KST 다. 여기에 at time zone·tz_localize·tz_convert·utc=True 를 걸면 9시간 어긋나는데,
# 실제로 그 실수로 정상 데이터를 "오진"이라 잘못 판정했다(krx_session 독스트링). 분봉을 다루는 파일에서는 시간대 변환 자체를 막는다.
_TZ_SQL = re.compile(r"\bat\s+time\s+zone\b", re.I)
_TZ_CALLS = {"tz_localize", "tz_convert"}
_TZ_EXEMPT = {"scripts/check_guardrails.py", "tests/test_check_guardrails.py"}


def check_minute_bars_naive_kst() -> list[str]:
    """(m) minute_bars 를 다루는 파일에서 시간대 변환을 쓰면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _TZ_EXEMPT or rel.startswith(".venv/"):
            continue
        text = p.read_text(encoding="utf-8")
        if "minute_bars" not in text:
            continue
        try:
            tree = ast.parse(text)
            strings = _sql_strings(text)
        except (SyntaxError, ValueError) as exc:
            out.append(f"[tz] {rel} — 파싱 실패: {exc}")
            continue
        bad: list[tuple[int, str]] = [(ln, "at time zone") for ln, sv in strings if _TZ_SQL.search(sv)]
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr in _TZ_CALLS:
                    bad.append((n.lineno, f.attr))
                for k in n.keywords:
                    if k.arg == "utc" and isinstance(k.value, ast.Constant) and k.value.value is True:
                        bad.append((n.lineno, "utc=True"))
        out += [f"[tz] {rel}:{ln} — minute_bars 를 다루는 파일에서 {what} 를 썼다. minute_bars.ts 는 시간대 없는 KST 다 — "
                f"변환하면 9시간 어긋난다(정규장 구간은 swing_it.krx_session.SESSION_OPEN·SESSION_CLOSE 로 자른다)."
                for ln, what in sorted(set(bad))]
    return out


# (n) trade_value 는 백만원 단위다(ADV 10억 = 1,000). 맨 숫자 문턱은 한 자리 틀려도 그럴듯해 보인다 — 4f6ef61 이 원 단위로
# 배선했고, features/universe.py 는 10,000 을 "10억" 이라 적었다(실제 100억). 거래대금 문턱 이름(토큰 adv·tv)에 숫자 리터럴을
# 넣는 자리(대입·기본값·키워드 인자·dict 키)를 막고 swing_it.units.eok(억)·jo(조)·mil(백만원) 을 쓰게 한다. 0 은 단위와 무관해 허용.
# window·days·rank 처럼 돈이 아닌 이름은 뺀다. tests/ 는 작은 가짜 단위를 쓰므로 대상이 아니다.
_TV_TOKENS = {"adv", "tv"}
_TV_NOT_MONEY = {"window", "win", "days", "day", "n", "rank", "frac", "pct", "q", "ratio", "mult", "rel", "z", "lag", "len",
                 "periods", "min_periods", "col", "cols", "key", "name", "band", "sample", "unit", "units"}
#: dict 키는 문턱 낱말(floor·min·max·lo·hi·cap·cut)이 함께 있을 때만 본다 — {"tv": 500.0} 같은 데이터 행(검증기 합성 표)은 문턱이 아니다.
_TV_THRESHOLD = {"floor", "min", "max", "lo", "hi", "cap", "cut", "thr", "threshold"}
_TV_EXEMPT = {
    "scripts/check_guardrails.py", "tests/test_check_guardrails.py", "src/swing_it/units.py",
    # kqflow TUI 는 리포트 값을 **억 원** 단위로 받는다(scripts/sector_flow.py 가 원 / 1e8). trade_value(백만원) 문턱이 아니다.
    "src/swing_it/tui/flow_view.py",
}


def _is_money_name(name: str) -> bool:
    toks = [t for t in re.split(r"[_\d]+", name.lower()) if t]
    return bool(_TV_TOKENS & set(toks)) and not (_TV_NOT_MONEY & set(toks))


def _bare_number(node) -> bool:
    """숫자 리터럴(부호·사칙연산 포함, 0 제외)인가."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value != 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _bare_number(node.operand)
    if isinstance(node, ast.BinOp):
        lits = [x for x in (node.left, node.right) if isinstance(x, (ast.Constant, ast.UnaryOp, ast.BinOp))]
        return len(lits) == 2 and all(_bare_number(x) or (isinstance(x, ast.Constant) and x.value == 0) for x in lits)
    if isinstance(node, ast.Tuple):
        return any(_bare_number(e) for e in node.elts)
    return False


def check_trade_value_units() -> list[str]:
    """(n) 거래대금 문턱 이름에 맨 숫자 리터럴을 넣으면 위반."""
    out = []
    for p in _iter_py(REPO):
        rel = p.relative_to(REPO).as_posix()
        if rel in _TV_EXEMPT or rel.startswith((".venv/", "tests/")):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            out.append(f"[units] {rel} — 파싱 실패: {exc}")
            continue
        hits: list[tuple[int, str]] = []
        for n in ast.walk(tree):
            pairs = []
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    elts = t.elts if isinstance(t, ast.Tuple) else [t]
                    vals = n.value.elts if isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple) else [n.value] * len(elts)
                    pairs += list(zip(elts, vals))
            elif isinstance(n, ast.AnnAssign) and n.value is not None:
                pairs.append((n.target, n.value))
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                pos = a.posonlyargs + a.args
                pairs += [(x, d) for x, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)]
                pairs += [(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            elif isinstance(n, ast.Call):
                pairs += [(k, k.value) for k in n.keywords if k.arg]
            elif isinstance(n, ast.Dict):
                pairs += [(k, v) for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and isinstance(k.value, str)
                          and _TV_THRESHOLD & set(re.split(r"[_\d]+", k.value.lower()))]
            for tgt, val in pairs:
                name = (tgt.id if isinstance(tgt, ast.Name) else tgt.attr if isinstance(tgt, ast.Attribute)
                        else tgt.arg if isinstance(tgt, (ast.arg, ast.keyword)) else tgt.value if isinstance(tgt, ast.Constant)
                        else None)
                if isinstance(name, str) and _is_money_name(name) and _bare_number(val):
                    hits.append((getattr(val, "lineno", getattr(n, "lineno", 0)), name))
        out += [f"[units] {rel}:{ln} — 거래대금 문턱 {name} 에 맨 숫자를 넣었다. trade_value 는 백만원 단위다 — "
                f"swing_it.units.eok(억)·jo(조)·mil(백만원) 으로 단위를 적어라(예: eok(200) = 20,000 = 200억)."
                for ln, name in sorted(set(hits))]
    return out


# (o) sector_index 는 2024-01-26 부터만 있고 값은 ×100 정수다. 직접 읽는 코드는 그 전 구간을 조용히 비우거나 대용으로 채웠다
# (재구축 대용이 −4%/년 어긋나 거짓 통과). 정문 storage.read_index 는 그 전 요청에 대용(proxy) 명시를 요구하고 level(=close/100)을 준다.
_RAW_INDEX = re.compile(r"\b(?:FROM|JOIN)\s+sector_index\b", re.I)
_INDEX_READ_EXEMPT = {"src/swing_it/storage.py", "scripts/check_guardrails.py", "tests/test_check_guardrails.py"}


def check_no_raw_index_select() -> list[str]:
    """(o) storage 밖에서 sector_index 를 직접 SELECT·JOIN 하면 위반."""
    return _scan_sql(_INDEX_READ_EXEMPT, _RAW_INDEX, "index",
                     "sector_index 를 직접 읽었다. swing_it.storage.read_index(con, codes, ...) 를 쓸 것 — 공식 지수는 "
                     "2024-01-26 부터라 그 전 구간은 proxy= 로 대용을 명시해야 하고, 값은 ×100 정수(level = close/100)")


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
    "sepa_confluence": "sepa_confluence_gate.py",
}
# VERDICT 가 러너 없이 존재해도 되는 유일한 조건: 아래 문구로 재현 불가를 선언한다.
_IRREPRODUCIBLE_MARKER = "재현 불가 고지"


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
    violations += check_no_raw_supply_close()
    violations += check_no_raw_shares_select()
    violations += check_avail_date_asof()
    violations += check_minute_bars_naive_kst()
    violations += check_trade_value_units()
    violations += check_no_raw_index_select()
    violations += check_verdicts_have_runner()
    violations += check_commit_identity()

    if violations:
        print("가드레일 린트 실패 — 위반 %d건:\n" % len(violations))
        for v in violations:
            print("  " + v)
        print("\n(GUARDRAILS.md §5 — 경계·판정·리포터 규칙)")
        return 1

    print("가드레일 린트 통과 — 경계·VERDICT·리포터 위반 없음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
