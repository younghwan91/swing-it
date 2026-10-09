"""GA2 → 라이브 인계 파일 ``live_candidate.json`` — 쓰기·읽기·검증 (REVIEW_INTEGRATION §2–3).

스키마(``schema = 1``)::

    {"schema": 1, "genes": {19개 (+ 수급 3개)}, "strategy": asdict(StrategyConfig.from_genes(genes)),
     "fitness": {...GA 가 적은 그대로...}, "window": {...}, "created": "YYYY-MM-DD", "valid_from": "YYYY-MM-DD",
     "code_hash": sha256(유전자를 해석하는 코드 — config·strategy·features·sizing·backtest 소스)[:16],
     "strategy_hash": sha256(canonical strategy dict)[:16]}

:func:`load_candidate` 는 (1) 모르는 키·빠진 키, (2) ``strategy`` ≠ ``from_genes(genes)`` (두 표현의 자기 일관성), (3) ``strategy_hash``,
(4) ``code_hash`` ≠ 지금 코드(유전자 해석이 바뀌었을 수 있다 — ``allow_code_mismatch=True`` 가 아니면 예외), (5) ``valid_from`` 형식을
검사한다. 유전자 값은 :meth:`StrategyConfig.from_genes` 가 다시 검사한다(대응 없는 유전자·허용 밖 단계 → 예외).

``ga2_run.py`` 가 쓰는 옛 모양(``config`` 키, ``"inf"`` 문자열)은 :func:`genes_from_legacy` 로 읽는다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from swing_it.live.config import DEFAULT_STRATEGY, FIELD_NAMES, StrategyConfig

SCHEMA = 1
REQUIRED = ("schema", "genes", "strategy", "fitness", "window", "created", "valid_from", "code_hash", "strategy_hash")
#: 유전자를 해석하는 코드 — 이 파일들이 바뀌면 code_hash 가 바뀐다.
CODE_FILES = ("config.py", "strategy.py", "features.py", "sizing.py", "backtest.py")


class CandidateError(ValueError):
    """인계 파일이 스키마·해시·자기 일관성 검사를 통과하지 못했다."""


def _canon(obj) -> str:
    def default(v):
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        raise TypeError(type(v).__name__)

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)


def _jsonable(obj):
    """inf → "inf" (JSON 은 Infinity 를 표준으로 못 쓴다)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and math.isinf(obj):
        return "inf"
    return obj


def code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in CODE_FILES:
        h.update(name.encode())
        h.update((here / name).read_bytes())
    return h.hexdigest()[:16]


def strategy_hash(cfg: StrategyConfig) -> str:
    return hashlib.sha256(_canon(asdict(cfg)).encode()).hexdigest()[:16]


def _normalize_strategy(d: dict) -> dict:
    """JSON 에서 읽은 strategy dict 의 "inf" → inf (비교용)."""
    out = {}
    for k, v in d.items():
        out[k] = math.inf if isinstance(v, str) and v.strip().lower() in ("inf", "infinity", "∞") else v
    return out


def build_candidate(genes: dict, *, fitness: dict, window: dict, valid_from: "str | dt.date",
                    created: "str | dt.date | None" = None, base: "StrategyConfig | None" = None) -> dict:
    """유전자 → 인계 파일 dict(쓰기 전 자기 일관성 포함). 유전자가 틀리면 :meth:`StrategyConfig.from_genes` 가 예외."""
    cfg = StrategyConfig.from_genes(genes, base=base)
    vf = dt.date.fromisoformat(str(valid_from)[:10])
    cr = dt.date.today() if created is None else dt.date.fromisoformat(str(created)[:10])
    return {"schema": SCHEMA, "genes": _jsonable(cfg.to_genes()), "strategy": _jsonable(asdict(cfg)),
            "fitness": _jsonable(fitness), "window": _jsonable(window), "created": cr.isoformat(),
            "valid_from": vf.isoformat(), "code_hash": code_hash(), "strategy_hash": strategy_hash(cfg)}


def write_candidate(path: "str | Path", genes: dict, **kw) -> dict:
    doc = build_candidate(genes, **kw)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def genes_from_legacy(doc: dict) -> dict:
    """``ga2_run.py`` 의 ``live_candidate.json``(``config`` 키, ``"inf"`` 문자열) → 유전자 dict."""
    src = doc.get("genes", doc.get("config"))
    if not isinstance(src, dict):
        raise CandidateError("genes/config 블록이 없다")
    return {k: (math.inf if isinstance(v, str) and v.lower() == "inf" else v) for k, v in src.items()}


def load_candidate(path: "str | Path", *, allow_code_mismatch: bool = False, base: "StrategyConfig | None" = None
                   ) -> "tuple[StrategyConfig, dict]":
    """인계 파일 → ``(StrategyConfig, 문서)``. 검사에 하나라도 걸리면 :class:`CandidateError`(기본값으로 조용히 돌지 않는다)."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CandidateError(f"{p}: 읽기 실패 — {exc}") from exc
    if not isinstance(doc, dict):
        raise CandidateError(f"{p}: JSON 객체가 아니다")
    missing = [k for k in REQUIRED if k not in doc]
    extra = [k for k in doc if k not in REQUIRED and k not in ("note", "source")]
    if missing or extra:
        raise CandidateError(f"{p}: 빠진 키 {missing} · 모르는 키 {extra}")
    if doc["schema"] != SCHEMA:
        raise CandidateError(f"{p}: schema {doc['schema']} ≠ {SCHEMA}")
    try:
        dt.date.fromisoformat(doc["valid_from"])
        dt.date.fromisoformat(doc["created"])
    except (TypeError, ValueError) as exc:
        raise CandidateError(f"{p}: 날짜 형식 — {exc}") from exc
    try:
        cfg = StrategyConfig.from_genes(genes_from_legacy({"genes": doc["genes"]}), base=base)
    except ValueError as exc:
        raise CandidateError(f"{p}: 유전자 — {exc}") from exc
    strat = doc["strategy"]
    if not isinstance(strat, dict) or set(strat) != set(FIELD_NAMES):
        raise CandidateError(f"{p}: strategy 블록 키가 StrategyConfig 필드와 다르다")
    if _normalize_strategy(strat) != _normalize_strategy(_jsonable(asdict(cfg))):
        diff = {k: (strat.get(k), asdict(cfg).get(k)) for k in FIELD_NAMES
                if _normalize_strategy({k: strat.get(k)})[k] != _normalize_strategy(_jsonable({k: asdict(cfg)[k]}))[k]}
        raise CandidateError(f"{p}: strategy ≠ from_genes(genes): {diff}")
    if doc["strategy_hash"] != strategy_hash(cfg):
        raise CandidateError(f"{p}: strategy_hash 불일치")
    if doc["code_hash"] != code_hash() and not allow_code_mismatch:
        raise CandidateError(f"{p}: code_hash {doc['code_hash']} ≠ 지금 코드 {code_hash()} — 유전자 해석 코드가 바뀌었다; "
                             "다시 만들거나 allow_code_mismatch=True")
    return cfg, doc


def default_candidate_doc(valid_from: "str | dt.date") -> dict:
    """코드 기본 설정을 인계 파일 모양으로(비교·테스트용)."""
    return build_candidate(DEFAULT_STRATEGY.to_genes(), fitness={}, window={}, valid_from=valid_from)
