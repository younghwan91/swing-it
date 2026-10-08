"""기사 LLM 판정 원장(quant-airflow ``judges`` · ``article_judgments``, migration 016) 읽기·쓰기.

판정의 정본은 DB 다. ``data/eval/judge_runs/`` 의 JSONL·상태 파일은 **작업 상태**(재개·재시도 건너뛰기)일 뿐이다.

규칙(016):

- judge 이름 하나 = 모델·system 프롬프트·user 템플릿·파라미터 한 벌. 한 글자라도 다르면 새 이름 —
  :func:`register_judge` 는 같은 이름이 다른 정의로 이미 있으면 :class:`JudgeCollision` 을 던진다.
- (기사, 종목, judge) 당 한 행. **성공 행은 덮어쓰지 않는다** — 실패 행(output NULL)만 나중 성공이 바꾼다
  (:func:`write_judgments` 의 ``ON CONFLICT ... WHERE article_judgments.output IS NULL``).
- 각 레포는 자기가 등록한(owner_repo) judge 의 행만 쓴다. 옛 파일의 일회성 이전만 예외이고, 그 사실을
  judge ``description`` 에 적는다(``v2-haiku45`` — daytrade-it 소유, swing-it 이 2026-10-09 한 번 이전).

쌍 키 ``uuid|code`` 의 uuid 는 ``pair_key_article(news_articles.id)`` (uuid5) 라 거꾸로 풀 수 없다 —
:func:`key_map` 이 ``news_company_feed`` 에 있는 기사 id 전부로 역사전을 만든다. 못 푸는 키는 추측하지 않고
세어서 보고한다.

DSN 은 :func:`swing_it.storage.load_env_db` / ``KR_QUANT_DB`` 로만 읽고 어디에도 찍지 않는다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from swing_it.news.prompt import (
    _MAX_OUTPUT_TOKENS,
    _SYSTEM,
    _USER_TEMPLATE,
    ADDENDUM_R3,
    DEFAULT_MODEL,
    system_with_addendum,
)
from swing_it.news.sweep import pair_key_article

OWNER = "swing-it"


class JudgeCollision(RuntimeError):
    """같은 judge 이름이 다른 정의(모델·system·user 템플릿·파라미터)로 이미 등록돼 있다."""


@dataclass(frozen=True)
class JudgeSpec:
    judge: str
    model: str
    system_prompt: str
    user_template: str
    params: dict = field(hash=False)
    output_schema: str
    owner_repo: str
    description: str

    @property
    def system_sha256(self) -> str:
        return hashlib.sha256(self.system_prompt.encode()).hexdigest()

    def identity(self) -> tuple:
        """이름 충돌 판정에 쓰는 정의 — description·owner 는 빼고."""
        return (self.model, self.system_sha256, self.user_template, _canon(self.params))


def _canon(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


#: daytrade-it 운영 v2(Haiku 4.5). 판정 요청: system 평문, temperature 0, max_tokens 400.
V2_HAIKU45 = JudgeSpec(
    judge="v2-haiku45",
    model=DEFAULT_MODEL,
    system_prompt=_SYSTEM,
    user_template=_USER_TEMPLATE,
    params={"max_tokens": _MAX_OUTPUT_TOKENS, "temperature": 0},
    output_schema="v2",
    owner_repo="daytrade-it",
    description=(
        "daytrade-it 운영 v2 추출 프롬프트(anthropic_client._SYSTEM/_USER_TEMPLATE), Haiku 4.5. "
        "2026-10-09 swing-it 이 daytrade-it data/eval/cache/universe_v2.jsonl(종목 유니버스 백필 배치 판정, "
        "2026-09-15~17)과 universe_v2_errors.jsonl 을 한 번 이전했다(일회성 import — 이후 쓰기는 daytrade-it 만). "
        "judged_at = 그 쌍을 담은 배치의 제출 시각(judged_at_exact FALSE)."
    ),
)

#: 스윙 2호 판정(사전등록 수정 1): 운영 v2 + "\n\n" + r3 보충, Haiku 5.5, 생각 끔.
V2R3_HAIKU55 = JudgeSpec(
    judge="v2r3-haiku55",
    model="claude-haiku-5-5",
    system_prompt=system_with_addendum(_SYSTEM, ADDENDUM_R3.strip()),
    user_template=_USER_TEMPLATE,
    params={"max_tokens": _MAX_OUTPUT_TOKENS, "thinking": {"type": "disabled"}},
    output_schema="v2",
    owner_repo=OWNER,
    description=(
        "swing-it 스윙 2호(뉴스 지속성) 판정 — 운영 v2 system + '\\n\\n' + 보충 r3"
        "(research/logs/news_persistence_swing/prompt_r3.txt), Haiku 5.5, thinking disabled, temperature 없음"
        "(모델이 받지 않는다). system 은 cache_control 블록으로 보내지만 모델 입력은 같다. "
        "1단계(2025-09~2026-02) 배치 캐시는 2026-10-09 이전(judged_at = 배치 제출 시각, exact FALSE)."
    ),
)

#: 운영 v2 그대로 Haiku 5.5, 생각 끔 — 2026-10-08 재파일럿(500쌍, 2026-04) 판정. 사전등록 수정 1 의 근거 자료.
V2_HAIKU55 = JudgeSpec(
    judge="v2-haiku55",
    model="claude-haiku-5-5",
    system_prompt=_SYSTEM,
    user_template=_USER_TEMPLATE,
    params={"max_tokens": _MAX_OUTPUT_TOKENS, "thinking": {"type": "disabled"}},
    output_schema="v2",
    owner_repo=OWNER,
    description=(
        "운영 v2 프롬프트 그대로 Haiku 5.5, thinking disabled, temperature 없음. 2026-10-08 재파일럿"
        "(2026-04 겹치는 500쌍, Haiku 4.5 와의 일치도) 배치 캐시를 2026-10-09 이전(judged_at = 배치 제출 시각, "
        "exact FALSE). 첫 파일럿(생각 기본값)은 정의를 고정할 수 없어 이전하지 않았다."
    ),
)

KNOWN_JUDGES = {s.judge: s for s in (V2_HAIKU45, V2R3_HAIKU55, V2_HAIKU55)}


def spec_for_request(model: str, system: str, params: dict) -> JudgeSpec:
    """요청 정의(모델·평문 system·파라미터) → 등록된 judge. 없으면 새 이름을 정해 등록하라고 실패한다."""
    want = (model, hashlib.sha256(system.encode()).hexdigest(), _USER_TEMPLATE, _canon(params))
    for spec in KNOWN_JUDGES.values():
        if spec.identity() == want:
            return spec
    raise JudgeCollision(
        f"요청 정의(model={model}, system sha256={want[1][:12]}…, params={want[3]})에 맞는 judge 가 없다 — "
        "새 judge 이름을 정의(JudgeSpec)하고 등록한 뒤 판정하라"
    )


# --- DB ----------------------------------------------------------------------------------


def register_judge(con, spec: JudgeSpec) -> bool:
    """없으면 등록(True), 같은 정의로 있으면 그대로(False), 다른 정의로 있으면 :class:`JudgeCollision`."""
    cur = con.cursor()
    cur.execute(
        "SELECT model, system_sha256, user_template, params FROM judges WHERE judge = %s", (spec.judge,)
    )
    row = cur.fetchone()
    if row is not None:
        model, sha, user_template, params = row
        params = params if isinstance(params, dict) else json.loads(params)
        if (model, sha, user_template, _canon(params)) != spec.identity():
            raise JudgeCollision(f"judge {spec.judge!r} 가 다른 정의로 이미 등록돼 있다 — 새 이름을 써라")
        return False
    cur.execute(
        "INSERT INTO judges (judge, model, system_sha256, system_prompt, user_template, params, output_schema, "
        "owner_repo, description) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)",
        (spec.judge, spec.model, spec.system_sha256, spec.system_prompt, spec.user_template, _canon(spec.params),
         spec.output_schema, spec.owner_repo, spec.description),
    )
    con.commit()
    return True


_COLS = ("article_id", "code", "judge", "input_hash", "output", "error", "judged_at", "judged_at_exact", "batch_id",
         "input_tokens", "cache_creation_tokens", "cache_read_tokens", "output_tokens")

_UPSERT = (
    "INSERT INTO article_judgments (" + ", ".join(_COLS) + ") VALUES %s "
    "ON CONFLICT (article_id, code, judge) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLS[3:])
    # 성공 행은 그대로. 실패 행은 성공으로, 또는 더 나중 시도의 실패로만 바뀐다(같은 행 재이전은 no-op — 멱등).
    + " WHERE article_judgments.output IS NULL"
    " AND (EXCLUDED.output IS NOT NULL OR EXCLUDED.judged_at > article_judgments.judged_at)"
    " RETURNING (xmax = 0) AS inserted"
)


def dedupe_rows(rows: Iterable[dict]) -> list[dict]:
    """같은 (기사, 종목, judge) 가 한 번에 두 번 오면 한 줄로 — 성공이 실패를 이기고, 같은 종류면 나중 것.

    (한 INSERT ... ON CONFLICT DO UPDATE 문은 같은 행을 두 번 건드릴 수 없다.)
    """
    out: dict[tuple, dict] = {}
    for r in rows:
        pk = (r["article_id"], r["code"], r["judge"])
        prev = out.get(pk)
        if prev is None or not (prev.get("output") is not None and r.get("output") is None):
            out[pk] = r
    return list(out.values())


def write_judgments(con, rows: Iterable[dict], page_size: int = 2000) -> dict[str, int]:
    """원장에 upsert. -> {"inserted", "updated", "unchanged"} (unchanged = 이미 성공 행이라 건드리지 않음).

    ``rows``: :data:`_COLS` 키의 dict. ``output`` 은 dict(JSONB) 또는 None, ``error`` 는 str 또는 None —
    정확히 하나만 None. 실패하면 롤백하고 예외를 그대로 올린다(파일만 성공한 척하지 않는다).
    """
    from psycopg2.extras import Json, execute_values  # noqa: PLC0415 — pg 경로에서만

    rows = dedupe_rows(rows)
    for r in rows:
        if (r.get("output") is None) == (r.get("error") is None):
            raise ValueError(f"output/error 중 정확히 하나만 있어야 한다: {r['article_id']}|{r['code']}")
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    cur = con.cursor()
    try:
        for i in range(0, len(rows), page_size):
            part = rows[i : i + page_size]
            values = [
                tuple(Json(r[c]) if c == "output" and r.get(c) is not None else r.get(c) for c in _COLS)
                for r in part
            ]
            got = execute_values(cur, _UPSERT, values, page_size=page_size, fetch=True)
            ins = sum(1 for (flag,) in got if flag)
            counts["inserted"] += ins
            counts["updated"] += len(got) - ins
            counts["unchanged"] += len(part) - len(got)
        con.commit()
    except Exception:
        con.rollback()
        raise
    return counts


def key_map(con) -> dict[str, str]:
    """쌍 키의 uuid → news_articles.id (``news_company_feed`` 에 있는 기사 전부, 약 142만)."""
    cur = con.cursor()
    cur.execute("SELECT DISTINCT article_id FROM news_company_feed")
    return {pair_key_article(aid): aid for (aid,) in cur.fetchall()}


def resolve_keys(keys: Iterable[str], kmap: dict[str, str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """``uuid|code`` 키들 → ({key: (article_id, code)}, 못 푼 키 목록). 추측하지 않는다."""
    found: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    for key in keys:
        uuid, _, code = key.partition("|")
        aid = kmap.get(uuid)
        if aid is None or not code:
            missing.append(key)
        else:
            found[key] = (aid, code)
    return found, missing


def read_judgments(con, judge: str, keys: Iterable[str] | None = None, *, include_errors: bool = False) -> dict[str, dict]:
    """judge 의 판정 → {쌍 키 ``uuid|code``: 행}. 행 = output·error·input_hash·judged_at·judged_at_exact·batch_id.

    기본은 성공 행만. ``keys`` 를 주면 그 키만(없는 키는 결과에 없다).
    """
    cur = con.cursor()
    cur.execute(
        "SELECT article_id, code, input_hash, output, error, judged_at, judged_at_exact, batch_id "
        "FROM article_judgments WHERE judge = %s" + ("" if include_errors else " AND output IS NOT NULL"),
        (judge,),
    )
    want = set(keys) if keys is not None else None
    out: dict[str, dict] = {}
    for aid, code, h, output, error, at, exact, batch in cur.fetchall():
        key = f"{pair_key_article(aid)}|{code}"
        if want is not None and key not in want:
            continue
        out[key] = {"article_id": aid, "code": code, "input_hash": h,
                    "output": output if output is None or isinstance(output, dict) else json.loads(output),
                    "error": error, "judged_at": at, "judged_at_exact": exact, "batch_id": batch}
    return out


def judgment_row(article_id: str, code: str, judge: str, input_hash: str, *, output: dict | None = None,
                 error: str | None = None, judged_at: dt.datetime, judged_at_exact: bool,
                 batch_id: str | None = None, usage: dict | None = None) -> dict:
    """:func:`write_judgments` 한 줄. ``usage`` 는 API usage 그대로(input/cache_creation/cache_read/output)."""
    u = usage or {}

    def tok(k: str) -> int | None:
        return int(u[k]) if u.get(k) is not None else None

    return {
        "article_id": article_id, "code": code, "judge": judge, "input_hash": input_hash,
        "output": output, "error": error, "judged_at": judged_at, "judged_at_exact": judged_at_exact,
        "batch_id": batch_id, "input_tokens": tok("input_tokens"),
        "cache_creation_tokens": tok("cache_creation_input_tokens"),
        "cache_read_tokens": tok("cache_read_input_tokens"), "output_tokens": tok("output_tokens"),
    }


def connect_db():
    """``KR_QUANT_DB`` (없으면 ``.env``) 로 Postgres 연결. sqlite 폴백 없음 — 원장은 운영 DB 에만 있다."""
    from swing_it.storage import connect, load_env_db  # noqa: PLC0415

    dsn = load_env_db()
    if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("KR_QUANT_DB(Postgres DSN)가 없다 — 판정 원장은 quant-airflow DB 에만 있다")
    return connect(dsn)


class UnmappedKeys(RuntimeError):
    """쌍 키를 news_articles.id 로 풀 수 없다 — 원장에 쓰지 않는다(추측 금지)."""


class JudgmentLedger:
    """판정기가 배치 결과를 원장에 쓰는 창구. judge 는 처음 쓸 때 등록하고, 이 레포 소유 judge 만 쓴다.

    ``entries``: ``{"key", "input_hash", "output" | "error", "batch_id", "judged_at", "judged_at_exact", "usage"}``.
    키를 하나라도 못 풀거나 DB 쓰기가 실패하면 예외 — 호출자는 그 배치를 처리 완료로 표시하지 않는다.
    """

    def __init__(self, con, spec: JudgeSpec, kmap: dict[str, str] | None = None) -> None:
        if spec.owner_repo != OWNER:
            raise PermissionError(f"judge {spec.judge!r} 는 {spec.owner_repo} 소유 — swing-it 은 쓰지 않는다")
        self.con, self.spec = con, spec
        self._kmap = kmap
        self._registered = False

    def record(self, entries: list[dict]) -> dict[str, int]:
        if not entries:
            return {"inserted": 0, "updated": 0, "unchanged": 0}
        if self._kmap is None:
            self._kmap = key_map(self.con)
        resolved, missing = resolve_keys((e["key"] for e in entries), self._kmap)
        if missing:
            raise UnmappedKeys(f"{len(missing)} 개 쌍 키를 news_articles.id 로 못 푼다, 예: {missing[:3]}")
        if not self._registered:
            register_judge(self.con, self.spec)
            self._registered = True
        rows = [
            judgment_row(*resolved[e["key"]], self.spec.judge, e["input_hash"], output=e.get("output"),
                         error=e.get("error"), judged_at=e["judged_at"], judged_at_exact=e["judged_at_exact"],
                         batch_id=e.get("batch_id"), usage=e.get("usage"))
            for e in entries
        ]
        return write_judgments(self.con, rows)
