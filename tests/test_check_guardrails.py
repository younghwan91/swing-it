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

import importlib.util
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


# --- (d)(g) 여러 줄 SQL — 2026-10-12 강화 ------------------------------------------------
#
# 예전 (d)(g)는 한 줄 정규식이라 SELECT 와 FROM 이 다른 줄이면 통과했다. 아래 모양들을 주입해 실제로
# 잡히는지, 주석·독스트링 산문은 안 잡는지 본다.

_MULTILINE_PRICE_SQL = '''
import pandas as pd

def load(con):
    q = """
        SELECT code, date, close
        FROM daily_bars_adjusted
        WHERE date >= '2020-01-01'
    """
    a = pd.read_sql(q, con)
    b = pd.read_sql("select code, date, close "
                    "from daily_bars where date >= '2020-01-01'", con)
    c = pd.read_sql("select code, close " + "from daily_bars_raw", con)
    return a, b, c
'''

_MULTILINE_EARNINGS_SQL = '''
def load(con):
    return con.execute("""SELECT code, period, netinc
                          FROM earnings
                          WHERE avail_date >= '2018-01-01'""")
'''

_PROSE_ONLY = '''
"""모듈 독스트링: SELECT 는 정문으로 — FROM daily_bars 직접 읽기 금지, select * from earnings 도 금지."""
# SELECT code FROM daily_bars  ← 주석
def f():
    """함수 독스트링 SELECT x FROM earnings"""
    return 1
'''


def _scan(lint, tmp_path, monkeypatch, src: str, fn: str) -> list[str]:
    (tmp_path / "loader.py").write_text(src, encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    return getattr(lint, fn)()


def test_rule_g_catches_multiline_and_concatenated_sql(lint, tmp_path, monkeypatch):
    """삼중따옴표·암묵 연결·``+`` 연결 SQL 셋 다 잡는다(한 줄 정규식이 놓치던 모양)."""
    out = _scan(lint, tmp_path, monkeypatch, _MULTILINE_PRICE_SQL, "check_no_raw_price_select")
    assert len(out) == 3, out
    assert all("universe" in o for o in out)
    assert any(":6" in o for o in out), out           # 삼중따옴표 SQL 은 SELECT 가 있는 줄을 가리킨다


def test_rule_d_catches_multiline_earnings_select(lint, tmp_path, monkeypatch):
    out = _scan(lint, tmp_path, monkeypatch, _MULTILINE_EARNINGS_SQL, "check_no_raw_earnings_select")
    assert len(out) == 1 and "earnings" in out[0], out


def test_rules_d_g_ignore_comments_and_docstrings(lint, tmp_path, monkeypatch):
    """산문은 위반이 아니다 — 거짓 양성이 많으면 규칙이 소음이 되어 꺼진다."""
    assert _scan(lint, tmp_path, monkeypatch, _PROSE_ONLY, "check_no_raw_price_select") == []
    assert _scan(lint, tmp_path, monkeypatch, _PROSE_ONLY, "check_no_raw_earnings_select") == []


def test_rule_g_reports_unparseable_file(lint, tmp_path, monkeypatch):
    """파싱 실패를 침묵하면 그 파일의 위반이 통째로 사라진다."""
    out = _scan(lint, tmp_path, monkeypatch, "def broken(:\n", "check_no_raw_price_select")
    assert len(out) == 1 and "파싱 실패" in out[0]


# --- (j) supply_demand 애프터마켓 오염 열 ------------------------------------------------------------------

_SD_TAINTED_READS = '''
import pandas as pd

def a(con):
    return pd.read_sql("SELECT code, date, close FROM supply_demand WHERE date >= '2026-09-01'", con)

def b(con):
    return pd.read_sql("""
        SELECT s.code, s.date, s.flu_rt
        FROM supply_demand s
    """, con)

def c(con):
    return pd.read_sql("select code, date, acc_trde_qty as v " "from supply_demand", con)

def d(con):
    return pd.read_sql("SELECT * FROM supply_demand", con)
'''

_SD_CLEAN_READS = '''
def ok(con):
    con.execute("SELECT code, date, foreign_, institution FROM supply_demand WHERE date >= ?")
    con.execute("SELECT MAX(date) FROM supply_demand")
    con.execute("SELECT b.close FROM daily_bars_x b WHERE b.code IN (SELECT code FROM supply_demand)")
'''


def test_rule_j_catches_tainted_supply_columns(lint, tmp_path, monkeypatch):
    """close·flu_rt·acc_trde_qty·* 를 직접 SELECT 하면(여러 줄·별칭·연결 포함) 잡는다."""
    out = _scan(lint, tmp_path, monkeypatch, _SD_TAINTED_READS, "check_no_raw_supply_close")
    assert len(out) == 4, out
    assert all("aftermarket" in o and "read_supply_demand" in o for o in out)


def test_rule_j_allows_flow_only_and_date_reads(lint, tmp_path, monkeypatch):
    """순매매 열·날짜만 읽는 SELECT 와, 바깥 SELECT 의 close(다른 표)는 대상이 아니다 — 거짓 양성 방지."""
    assert _scan(lint, tmp_path, monkeypatch, _SD_CLEAN_READS, "check_no_raw_supply_close") == []


# --- (k) 상장주식수 정문 -----------------------------------------------------------------------------------

_RAW_SHARES_READS = '''
import pandas as pd

def a(con):
    return pd.read_sql("select code, date, shares_outstanding from shares_outstanding_history", con)

def b(con, has):
    return pd.read_sql("select code, date" + (", orig_rcept_dt" if has else "") + " from shares_outstanding_history", con)

def c(con):
    con.execute("""SELECT b.code FROM daily_x b
                   JOIN shares_outstanding_history h ON h.code = b.code""")
'''


def test_rule_k_catches_direct_shares_reads(lint, tmp_path, monkeypatch):
    """직접 SELECT·문자열 연결·JOIN 모두 잡는다(recheck29 의 + 연결 모양 포함)."""
    out = _scan(lint, tmp_path, monkeypatch, _RAW_SHARES_READS, "check_no_raw_shares_select")
    assert len(out) == 3, out
    assert all("read_shares" in o for o in out)


def test_rule_k_ignores_prose_and_metadata(lint, tmp_path, monkeypatch):
    src = '''
"""주식수는 shares_outstanding_history 에서 — FROM shares_outstanding_history 직접 읽기 금지(독스트링 산문)."""
def f(cur):
    cur.execute("select column_name from information_schema.columns where table_name = 'shares_outstanding_history'")
'''
    assert _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_shares_select") == []


# --- (l) 실적 as-of 키 ------------------------------------------------------------------------------------

_AVAIL_ASOF_USES = '''
import pandas as pd

def a(ea, dates):
    return earnings_yoy_panel(ea, dates, avail_col="avail_date")

def b(ev, ea):
    return pd.merge_asof(ev, ea, left_on="date", right_on="avail_date", by="code")

def c(ea, dates):
    return earnings_leg_panel(ea, dates,
                              avail_col="avail_date")

def leaky(ea, *, avail_col="avail_date"):
    return ea
'''

_AVAIL_ASOF_OK = '''
def a(ea, dates):
    return earnings_yoy_panel(ea, dates, avail_col="known_date")

def b(ea, dates):  # legacy-asof: 18세대 기록 재현
    return earnings_yoy_panel(ea, dates,
                              avail_col="avail_date")

def c(ea, dates):
    return earnings_leg_panel(ea, dates, avail_col="avail_date")  # legacy-asof: 라이브 엔진 사전등록 그대로

def d(ea, *, avail_col):
    return ea
'''


def test_rule_l_catches_unmarked_avail_date_asof(lint, tmp_path, monkeypatch):
    """표시 없는 avail_date as-of 키(키워드·merge_asof·여러 줄)와 avail_col 기본값을 잡는다."""
    out = _scan(lint, tmp_path, monkeypatch, _AVAIL_ASOF_USES, "check_avail_date_asof")
    assert len(out) == 4, out
    assert sum("기본값" in o for o in out) == 1
    assert all("[asof]" in o for o in out)


def test_rule_l_accepts_known_date_and_marked_legacy(lint, tmp_path, monkeypatch):
    assert _scan(lint, tmp_path, monkeypatch, _AVAIL_ASOF_OK, "check_avail_date_asof") == []


# --- (m) 분봉 시간대 ---------------------------------------------------------------------------------------

_TZ_BAD = '''
import pandas as pd

def a(con):
    return pd.read_sql("SELECT code, ts AT TIME ZONE 'Asia/Seoul' AS t FROM minute_bars", con)

def b(m):
    m["ts"] = pd.to_datetime(m["ts"], utc=True).dt.tz_convert("Asia/Seoul")
    return m["ts"].dt.tz_localize(None)
'''


def test_rule_m_catches_timezone_conversion_on_minute_bars(lint, tmp_path, monkeypatch):
    out = _scan(lint, tmp_path, monkeypatch, _TZ_BAD, "check_minute_bars_naive_kst")
    kinds = {o.split(" 를 썼다")[0].rsplit(" ", 1)[-1] for o in out}
    assert kinds == {"zone", "utc=True", "tz_convert", "tz_localize"}, out


def test_rule_m_ignores_files_without_minute_bars(lint, tmp_path, monkeypatch):
    """분봉과 무관한 파일의 시간대 처리(뉴스 발행 시각 등)는 대상이 아니다."""
    src = 'import pandas as pd\nx = pd.to_datetime(["2026-01-01"], utc=True).tz_convert("Asia/Seoul")\n'
    assert _scan(lint, tmp_path, monkeypatch, src, "check_minute_bars_naive_kst") == []


# --- (n) 거래대금 단위 -------------------------------------------------------------------------------------

_TV_BARE = '''
ADV_FLOOR = 20000.0
MIN_TV = 1_000.0
ADV_LO, ADV_HI = 5_000.0, eok(200)
CFG = {"adv_floor": 5000, "horizon": 20}
FUNNEL = None
ADV_FLOORS_FUNNEL = (5_000.0, 3_000.0)

def f(prices, *, adv_floor: float = 20000.0):
    return g(prices, adv_floor=10 * 100)
'''

_TV_OK = '''
from swing_it.units import eok
ADV_FLOOR = eok(200)
ADV_WINDOW = 20
TV_UNIT_BAND = (0.5, 2.0)
ROW = {"tv": 500.0, "inst": 1.0}
CFG = {"adv_floor": eok(50), "adv_window": 20}

def f(prices, *, adv_floor: float = 0.0, adv_window: int = 20):
    return g(prices, adv_floor=eok(10))
'''


def test_rule_n_catches_bare_trade_value_thresholds(lint, tmp_path, monkeypatch):
    """대입·튜플 대입·dict 문턱 키·기본값·키워드 인자·리터럴 곱 모두 잡는다."""
    out = _scan(lint, tmp_path, monkeypatch, _TV_BARE, "check_trade_value_units")
    names = sorted(o.split("문턱 ")[1].split(" ")[0] for o in out)
    assert names == ["ADV_FLOOR", "ADV_FLOORS_FUNNEL", "ADV_LO", "MIN_TV", "adv_floor", "adv_floor", "adv_floor"], out


def test_rule_n_accepts_helpers_zero_and_non_money_names(lint, tmp_path, monkeypatch):
    assert _scan(lint, tmp_path, monkeypatch, _TV_OK, "check_trade_value_units") == []


# --- (o) 공식 지수 ----------------------------------------------------------------------------------------

def test_rule_o_catches_direct_index_reads(lint, tmp_path, monkeypatch):
    src = '''
import pandas as pd
a = lambda con: pd.read_sql("select date, close from sector_index where code='001' order by date", con)
def b(cur):
    cur.execute("""SELECT s.code, s.date, i.close
                   FROM supply_demand s JOIN sector_index i ON i.date = s.date""")
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_index_select")
    assert len(out) == 2 and all("read_index" in o for o in out), out


def test_rule_o_ignores_prose(lint, tmp_path, monkeypatch):
    src = '"""공식 지수는 sector_index(2024-01-26~). FROM sector_index 직접 읽기 금지."""\nx = 1\n'
    assert _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_index_select") == []


# --- 2026-10-13 리뷰 MAJOR 2: % 포맷·.format() SQL ------------------------------------------------------------
#
# 예전 _fold_str 는 BinOp(Mod) 를 접지 못해 ``"SELECT … FROM daily_bars WHERE date >= '%s'" % d`` 가 (d)(g)(j)(k)(o) 를
# 통째로 빠져나갔다(왼쪽 문자열은 BinOp 의 자식이라 따로도 안 보였다). 규칙마다 % 포맷·.format() 위반을 주입한다.

_PCT_CASES = {
    "check_no_raw_earnings_select": ('''
def f(con, d):
    return con.execute("SELECT code, period, netinc FROM earnings WHERE avail_date >= '%s'" % d)
def g(con, d):
    return con.execute("SELECT code FROM earnings WHERE period = '{}'".format(d))
''', "[earnings]"),
    "check_no_raw_price_select": ('''
def f(con, d, cols):
    a = con.execute("""SELECT %s
                       FROM daily_bars_adjusted WHERE date >= '%s'""" % (cols, d))
    b = con.execute("SELECT code, close FROM daily_bars WHERE date >= '{d}'".format(d=d))
    return a, b
''', "[universe]"),
    "check_no_raw_supply_close": ('''
def f(con, d):
    a = con.execute("SELECT code, date, close FROM supply_demand WHERE date >= '%s'" % d)
    b = con.execute("SELECT code, {} , flu_rt FROM supply_demand".format(d))
    return a, b
''', "[aftermarket]"),
    "check_no_raw_shares_select": ('''
def f(con, d):
    a = con.execute("select code, shares_outstanding from shares_outstanding_history where date <= '%s'" % d)
    b = con.execute("select code from shares_outstanding_history where code = '{}'".format(d))
    return a, b
''', "[shares]"),
    "check_no_raw_index_select": ('''
def f(con, code):
    a = con.execute("select date, close from sector_index where code = '%s'" % code)
    b = con.execute("select date, close from sector_index where code = '{}'".format(code))
    return a, b
''', "[index]"),
}


@pytest.mark.parametrize("fn", sorted(_PCT_CASES))
def test_sql_rules_catch_percent_and_format_sql(lint, tmp_path, monkeypatch, fn):
    src, tag = _PCT_CASES[fn]
    out = _scan(lint, tmp_path, monkeypatch, src, fn)
    assert len(out) == 2 and all(tag in o for o in out), out


def test_sql_rules_see_string_operand_of_unfoldable_mod(lint, tmp_path, monkeypatch):
    """왼쪽이 문자열이 아닌 ``%``(``fmt % "… FROM daily_bars"``)는 접지 못하지만, 오른쪽 문자열은 따로 본다."""
    src = 'def f(fmt):\n    return fmt % "SELECT code FROM daily_bars"\n'
    out = _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_price_select")
    assert len(out) == 1, out


def test_percent_arithmetic_is_not_sql(lint, tmp_path, monkeypatch):
    """숫자 % 연산·산문 % 포맷은 위반이 아니다."""
    src = 'x = 7 % 3\nmsg = "%d rows from daily_bars cache" % 5\n'
    assert _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_price_select") == []


# --- 2026-10-13 리뷰 MINOR: 거짓 음성·거짓 양성 ---------------------------------------------------------------------

def test_rule_g_catches_schema_quoted_and_read_sql_table(lint, tmp_path, monkeypatch):
    src = '''
import pandas as pd
def f(con):
    a = pd.read_sql("SELECT code, close FROM public.daily_bars", con)
    b = pd.read_sql('SELECT code, close FROM "daily_bars_adjusted" WHERE code = 1', con)
    c = pd.read_sql_table("daily_bars_raw", con)
    d = pd.read_sql_table(table_name="daily_bars", con=con)
    return a, b, c, d
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_price_select")
    assert len(out) == 4, out


def test_rule_g_ignores_other_tables_and_read_sql_table_of_others(lint, tmp_path, monkeypatch):
    src = '''
import pandas as pd
def f(con):
    a = pd.read_sql("SELECT code FROM daily_bars_x", con)
    b = pd.read_sql_table("stocks", con)
    return a, b
'''
    assert _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_price_select") == []


def test_rule_j_catches_join_supply_demand(lint, tmp_path, monkeypatch):
    """``JOIN supply_demand s`` 로 붙여 s.close·s.flu_rt 를 읽는 모양(별칭·AS·별칭 없음·WHERE 에서 읽기)."""
    src = '''
def f(con):
    con.execute("""SELECT b.code, s.close FROM daily_bars_x b
                   JOIN supply_demand s ON s.code = b.code AND s.date = b.date""")
    con.execute("SELECT b.code FROM daily_x b LEFT JOIN supply_demand AS q USING (code, date) WHERE q.flu_rt > 0")
    con.execute("SELECT b.code, supply_demand.acc_trde_qty FROM daily_x b JOIN supply_demand ON supply_demand.code = b.code")
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_supply_close")
    assert len(out) == 3 and all("aftermarket" in o for o in out), out


def test_rule_j_allows_count_star_and_flow_only_join(lint, tmp_path, monkeypatch):
    src = '''
def f(con):
    con.execute("SELECT count(*) FROM supply_demand WHERE date >= ?")
    con.execute("SELECT COUNT(*), max(date) FROM supply_demand")
    con.execute("SELECT b.close, s.institution FROM daily_x b JOIN supply_demand s ON s.code = b.code")
'''
    assert _scan(lint, tmp_path, monkeypatch, src, "check_no_raw_supply_close") == []


def test_rule_l_catches_list_and_variable_keys(lint, tmp_path, monkeypatch):
    src = '''
import pandas as pd
KEY = "avail_date"

def a(ev, ea):
    return pd.merge_asof(ev, ea, left_on="date", right_on=["avail_date"], by="code")

def b(ev, ea):
    return pd.merge_asof(ev, ea, left_on="date", right_on=KEY, by="code")

def c(ea, dates):
    col = "avail_date"
    return earnings_yoy_panel(ea, dates, avail_col=col)
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_avail_date_asof")
    assert len(out) == 3 and all("[asof]" in o for o in out), out


def test_rule_l_marker_must_be_on_same_or_previous_line(lint, tmp_path, monkeypatch):
    """표시가 호출에서 멀리(몇 줄 위·여러 줄 호출 중간의 다른 인자 줄) 있으면 면제가 아니다."""
    src = '''
# legacy-asof: 이 표시는 너무 멀다

def a(ea, dates):
    return earnings_yoy_panel(ea, dates, avail_col="avail_date")

def b(ea, dates):
    return earnings_leg_panel(
        ea,  # legacy-asof: 키와 떨어진 다른 인자 줄
        dates,
        horizon=20,
        avail_col="avail_date",
    )
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_avail_date_asof")
    assert len(out) == 2, out


def test_rule_l_marker_on_key_line_inside_multiline_call_is_ok(lint, tmp_path, monkeypatch):
    src = '''
def b(ea, dates):
    return earnings_leg_panel(
        ea, dates,
        avail_col="avail_date",  # legacy-asof: 사전등록 재현
    )
'''
    assert _scan(lint, tmp_path, monkeypatch, src, "check_avail_date_asof") == []


def test_rule_m_covers_files_using_session_bar_gateway(lint, tmp_path, monkeypatch):
    """문자열 minute_bars 없이 분봉 정문(read_session_bars*)만 부르는 파일도 대상이다."""
    src = '''
import pandas as pd
from swing_it.krx_session import read_session_bars_agg
def f(con):
    s = read_session_bars_agg(con, start="2026-09-14")
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize("Asia/Seoul")
    return s
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_minute_bars_naive_kst")
    assert len(out) == 1 and "tz_localize" in out[0], out


def test_rule_n_catches_trade_value_names_and_comparisons(lint, tmp_path, monkeypatch):
    src = '''
MIN_TRADE_VALUE = 1000.0

def f(df, trade_value, x):
    g(df, min_trade_value=5_000)
    a = df[df["trade_value"] >= 1000]
    b = trade_value > 2_000.0
    c = 500 < x.adv20
    return a, b, c
'''
    out = _scan(lint, tmp_path, monkeypatch, src, "check_trade_value_units")
    names = sorted(o.split("문턱 ")[1].split(" ")[0] for o in out)
    assert names == ["MIN_TRADE_VALUE", "adv20", "min_trade_value", "trade_value", "trade_value"], out


def test_rule_n_ignores_lookback_zero_and_unit_helpers(lint, tmp_path, monkeypatch):
    src = '''
from swing_it.units import eok
TV_LOOKBACK = 60
ADV_LOOKBACK_DAYS = 20
_TV_MIN_ROWS = 20
MIN_TRADE_VALUE = eok(10)

def f(df):
    a = df[df["trade_value"] > 0]
    b = df["adv20"] >= eok(50)
    c = df["n_trades"] >= 30
    return a, b, c
'''
    assert _scan(lint, tmp_path, monkeypatch, src, "check_trade_value_units") == []


# --- (p) 모의매매·재현 경로의 정문 인자 명시 -------------------------------------------------------------------------

_PAPER_BAD = '''
from swing_it.storage import read_prices, read_supply_demand, market_cap_asof_bulk
from swing_it.krx_session import read_raw_close

def load(con, s):
    a = read_prices(con, cols=("code", "date", "close"), start="2016-08-01")
    b = read_supply_demand(con, cols=("code", "date", "institution"))
    c = read_raw_close(con, start="2016-08-01")
    d = market_cap_asof_bulk(con, a)
    e = read_prices(con, session=s["session"], start="2016-08-01", unpatched=s["unpatched"])   # 변수도 명시다
    f = read_raw_close(con, start="2016-08-01", aftermarket_fix=False)
    return a, b, c, d, e, f
'''


def _paper_tree(lint, tmp_path, monkeypatch, rel: str, src: str) -> list[str]:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
    monkeypatch.setattr(lint, "REPO", tmp_path)
    return lint.check_paper_explicit_kwargs()


@pytest.mark.parametrize("rel", ["research/paper/leader_paper.py", "research/paper/new_arm.py",
                                 "research/experiments/leader21.py", "research/experiments/recheck29.py",
                                 "research/experiments/index27.py", "research/experiments/index28.py"])
def test_rule_p_requires_explicit_gateway_args_in_paper_paths(lint, tmp_path, monkeypatch, rel):
    out = _paper_tree(lint, tmp_path, monkeypatch, rel, _PAPER_BAD)
    assert len(out) == 4 and all("[explicit]" in o for o in out), out
    assert sorted(o.split("() 에 ")[1].split(" 를 ")[0] for o in out) == [
        "aftermarket_fix=", "legacy=", "session=, unpatched=", "session=, unpatched="]


def test_rule_p_requires_unpatched_unless_session_is_literal_raw(lint, tmp_path, monkeypatch):
    """정규장 모드(리터럴 "regular"·변수)면 못 고친 값 정책(unpatched=)도 명시 — 리터럴 "raw" 면 그 인자는 동작이 없어 요구하지 않는다."""
    src = (
        "def load(con, arm):\n"
        "    a = read_prices(con, session=\"raw\", cols=(\"code\", \"date\", \"close\"))\n"
        "    b = read_prices(con, session=\"regular\", cols=(\"code\", \"date\", \"close\"))\n"
        "    c = read_prices_wide(con, session=arm[\"session\"], cols=(\"close\",))\n"
        "    d = read_supply_demand(con, session=\"regular\", unpatched=\"nan\")\n"
        "    return a, b, c, d\n"
    )
    out = _paper_tree(lint, tmp_path, monkeypatch, "research/paper/arm.py", src)
    assert [o.split(":")[1].split(" ")[0] for o in out] == ["3", "4"], out
    assert all("unpatched=" in o for o in out)


def test_rule_p_ignores_files_outside_paper_paths(lint, tmp_path, monkeypatch):
    """원장 경로가 아닌 연구 파일은 대상이 아니다(정문 기본값을 쓰는 것이 새 코드의 정석)."""
    assert _paper_tree(lint, tmp_path, monkeypatch, "research/experiments/other.py", _PAPER_BAD) == []


def test_rule_p_table_covers_gateway_behaviour_args(lint):
    """표에서 정문 인자가 빠지면(규칙이 조용히 약해지면) 실패한다."""
    for fn in ("read_prices", "read_prices_wide", "read_supply_demand"):
        assert "session" in lint.EXPLICIT_KWARGS[fn]
    assert "aftermarket_fix" in lint.EXPLICIT_KWARGS["read_raw_close"]
    for pat in ("research/paper/*.py", "research/experiments/leader21.py", "research/experiments/recheck29.py"):
        assert pat in lint.PAPER_PATHS


# --- 2026-10-13 리뷰 MAJOR 3: 규칙이 main() 에 배선됐는가 --------------------------------------------------------------
#
# 규칙 함수의 단위 테스트는 ``main()`` 에서 ``violations += check_x()`` 한 줄이 빠져도 초록이다(GUARDRAILS §4.5 실패 모드).
# 그래서 (1) ast 로 모든 최상위 check_* 가 main() 에서 불리는지 보고, (2) 규칙마다 위반 하나만 넣은 임시 레포 트리에서
# main() 이 실패하고 그 규칙의 표지가 출력에 나오는지 끝단으로 본다.

def test_every_check_function_is_called_from_main():
    import ast

    tree = ast.parse((REPO / "scripts" / "check_guardrails.py").read_text(encoding="utf-8"))
    checks = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("check_")}
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {c.func.id for c in ast.walk(main) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert len(checks) >= 17, checks
    assert checks - called == set(), f"main() 에 배선되지 않은 규칙: {sorted(checks - called)}"


_E2E = {
    # 규칙: ({상대경로: 내용}, 출력에 나와야 할 표지)
    "a": ({"src/swing_it/leaky.py": "from research.signals import contrarian_retail\n"}, "[boundary]"),
    "b": ({"research/experiments/orphan_gate.py": "from prop_gate import prop_gate\nprop_gate(1, config={})\n"}, "[verdict]"),
    "c": ({"research/experiments/judgey.py": 'verdict = "PASS"\n'}, "[reporter]"),
    "d": ({"loader.py": 'q = "SELECT code FROM earnings WHERE period = \'%s\'" % 1\n'}, "[earnings]"),
    "e": ({"research/experiments/sweep.py": "from prop_gate import prop_gate\nprop_gate(1, 2)\n"}, "[trials]"),
    "f": ({"research/experiments/lonely_gate.py": "x = 1\n",
           "research/logs/lonely/VERDICT.md": "재현 불가 고지 — 테스트\n"}, "[harness]"),
    "g": ({"loader.py": 'q = "SELECT code, close FROM daily_bars"\n'}, "가격 테이블을 직접"),
    "g2": ({"loader.py": 'q = """SELECT sd.code FROM supply_demand sd\n JOIN stocks s ON s.code = sd.code"""\n'}, "INNER JOIN"),
    "h": ({"research/logs/ghost/VERDICT.md": "판정만 있다\n"}, "[orphan-verdict]"),
    "j": ({"loader.py": 'q = "SELECT code, close FROM supply_demand"\n'}, "[aftermarket]"),
    "k": ({"loader.py": 'q = "SELECT code FROM shares_outstanding_history"\n'}, "[shares]"),
    "l": ({"loader.py": 'def f(ea, d):\n    return panel(ea, d, avail_col="avail_date")\n'}, "[asof]"),
    "m": ({"loader.py": 'q = "SELECT ts AT TIME ZONE \'UTC\' FROM minute_bars"\n'}, "[tz]"),
    "n": ({"loader.py": "ADV_FLOOR = 20000.0\n"}, "[units]"),
    "o": ({"loader.py": 'q = "SELECT close FROM sector_index"\n'}, "[index]"),
    "p": ({"research/paper/arm.py": "def f(con):\n    return read_prices(con)\n"}, "[explicit]"),
}


def _e2e_tree(lint, tmp_path, monkeypatch, files: dict) -> Path:
    root = tmp_path / "repo"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(lint, "REPO", root)
    monkeypatch.setattr(lint, "SRC", root / "src" / "swing_it")
    monkeypatch.setattr(lint, "EXPERIMENTS", root / "research" / "experiments")
    monkeypatch.setattr(lint, "LOGS", root / "research" / "logs")
    return root


def test_main_passes_on_an_empty_tree(lint, tmp_path, monkeypatch, capsys):
    """끝단 기준선 — 빈 트리는 통과(아래 실패가 환경 탓이 아니라 주입한 위반 탓임을 보장)."""
    _e2e_tree(lint, tmp_path, monkeypatch, {})
    assert lint.main() == 0, capsys.readouterr().out


@pytest.mark.parametrize("rule", sorted(_E2E))
def test_main_fails_on_each_injected_violation(lint, tmp_path, monkeypatch, capsys, rule):
    files, mark = _E2E[rule]
    _e2e_tree(lint, tmp_path, monkeypatch, files)
    rc = lint.main()
    out = capsys.readouterr().out
    assert rc != 0 and mark in out, f"({rule}) 위반을 넣었는데 main() 이 못 잡았다 — 배선 누락? rc={rc}\n{out}"


def test_main_fails_on_foreign_commit_identity(lint, tmp_path, monkeypatch, capsys):
    """(i) 도 main() 경로로 — 다른 이메일 커밋이 있는 임시 레포."""
    import subprocess

    root = _e2e_tree(lint, tmp_path, monkeypatch, {"f.txt": "x"})

    def git(*a):
        return subprocess.run(("git", *a), cwd=root, capture_output=True, text=True, check=False)

    git("init", "-q")
    git("config", "core.hooksPath", "")
    git("config", "user.name", "T")
    git("config", "user.email", "me@example.com")
    git("add", "-A")
    git("-c", "user.email=work@corp.com", "commit", "-q", "-m", "bad")
    assert lint.main() != 0
    assert "[identity]" in capsys.readouterr().out
