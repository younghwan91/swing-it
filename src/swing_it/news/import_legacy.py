"""옛 배치 판정 캐시(JSONL) → 판정 원장(``article_judgments``) 일회성 이전. 멱등 — 다시 돌려도 같다.

    python -m swing_it.news.import_legacy --judge v2-haiku45 \\
        --cache ~/git/daytrade-it/data/eval/cache/universe_v2.jsonl \\
        --errors ~/git/daytrade-it/data/eval/cache/universe_v2_errors.jsonl \\
        --states ~/git/daytrade-it/data/eval/cache/universe_batch_state*.json
    python -m swing_it.news.import_legacy --judge v2r3-haiku55 --cache .../judgments.jsonl ... --dry-run

순서(하나라도 어긋나면 쓰지 않고 멈춘다):

1. **judge 정의 검증** — 캐시 행 중 무작위 ``--sample`` (기본 2,000, 최소 1,000; 0 = 전부) 행의 ``input_hash`` 를
   DB 기사(``news_articles`` 의 먼저 수집된 판) + 지금 종목 마스터 + judge 정의로 다시 계산한다. 다른 행은 원인을
   가린다: ``--pairs`` (판정 당시 쌍 파일)의 **그 행 입력**으로 같은 judge 정의가 해시를 재현하면 "원천 차이"
   (DB 가 다른 판의 요약을 갖고 있거나 — 이전 전부터 daily_news 가 넣은 판 — 종목명이 바뀌었다)이고, judge 정의는
   맞다. **judge 정의로 재현되지 않는 행이 하나라도 있으면 이전하지 않는다**(쌍 파일이 없어 원인을 못 가려도 멈춘다).
2. **키 풀기** — ``uuid|code`` → ``news_articles.id`` (:func:`~swing_it.news.judgments_db.key_map`). 못 푸는 키는 세어
   보고하고 쓰지 않는다.
3. **judged_at** — 그 쌍(같은 input_hash)을 담고, 그 쌍이 오류로 끝나지 않은 배치의 제출 시각(``--states`` 의
   ``batches[].submitted_at``·``custom_ids``·``input_hashes``). 결과 시각은 캐시에 없어 ``judged_at_exact = FALSE``
   (Batch API 결과는 제출 후 24시간 안). 어느 상태에도 없는 키는 캐시 파일 mtime, 역시 FALSE — 센다.
4. **오류 행** — 오류 파일 중 **판정 시도였던 것만**(스키마 위반·과부하·결과 없음). 크레딧 부족·인증 같은
   계정 거절(``invalid_request_error`` 등)은 모델이 한 번도 보지 않은 요청이라 판정 실패가 아니다 — 원장에 넣지
   않고 센다. 성공이 있는 키의 오류는 넣지 않는다(성공 행이 이긴다). 같은 키의 오류가 여럿이면 마지막 배치.

원장 규칙상 성공 행은 덮어쓰지 않으므로 재실행은 새 행만 넣는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import Counter
from pathlib import Path

from swing_it.news.judgments_db import (
    KNOWN_JUDGES,
    JudgeSpec,
    connect_db,
    judgment_row,
    key_map,
    register_judge,
    resolve_keys,
    write_judgments,
)
from swing_it.news.pairs import load_stocks
from swing_it.news.prompt import build_prompt, judgment_input_hash

#: judge.py ``NON_ATTEMPT_ERROR_TYPES`` 와 같은 집합 — 계정 거절은 판정 시도가 아니다.
from swing_it.news.judge import counts_as_attempt  # noqa: E402  (순환 없음: judge 는 이 모듈을 import 하지 않는다)

MIN_VERIFY = 1000


def _ts(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def load_jsonl(path: Path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def batch_index(state_paths: list[Path]) -> tuple[dict[str, list[tuple]], dict[str, dt.datetime]]:
    """(키 → [(submitted_at, batch_id, input_hash)], batch_id → submitted_at) — 상태 파일 전부(이력 포함)."""
    by_key: dict[str, list[tuple]] = {}
    submitted: dict[str, dt.datetime] = {}
    for p in state_paths:
        s = json.loads(Path(p).read_text())
        batches = list(s.get("batches", [])) + [b for h in s.get("history", []) for b in h.get("batches", [])]
        for b in batches:
            at = _ts(b["submitted_at"])
            submitted[b["id"]] = at
            hashes = b.get("input_hashes") or {}
            for cid, key in b["custom_ids"].items():
                by_key.setdefault(key, []).append((at, b["id"], hashes.get(cid)))
    return by_key, submitted


def attribute(key: str, input_hash: str, by_key: dict, errored: set[tuple[str, str]]) -> tuple | None:
    """성공 캐시 행을 만든 배치 — 같은 input_hash 로 보냈고 그 키가 오류로 끝나지 않은 배치 중 가장 이른 것."""
    cands = sorted(c for c in by_key.get(key, ()) if c[2] in (input_hash, None) and (key, c[1]) not in errored)
    return cands[0] if cands else None


def load_articles_by_id(con, ids: list[str]) -> dict[str, tuple[str, str]]:
    """기사 id → (제목, 요약) — 같은 id 의 판이 여럿이면 먼저 수집된 판(쌍 생성기와 같은 규칙)."""
    cur = con.cursor()
    best: dict[str, tuple] = {}
    for i in range(0, len(ids), 5000):
        cur.execute(
            "SELECT id, title, summary, collected_at, published_at FROM news_articles WHERE id = ANY(%s)",
            (ids[i : i + 5000],),
        )
        for aid, title, summary, coll, pub in cur.fetchall():
            rank = (coll or pub, pub)
            if aid not in best or rank < best[aid][0]:
                best[aid] = (rank, title or "", summary or "")
    return {aid: (v[1], v[2]) for aid, v in best.items()}


def recompute_hash(spec: JudgeSpec, code: str, title: str, summary: str, stocks: dict) -> str:
    """DB 기사 + 종목 마스터 + judge 정의로 그 쌍의 입력 해시(쌍 생성기 규칙: body = 요약, 없으면 제목)."""
    s = stocks.get(code)
    _, user = build_prompt(code=code, name=s.name if s else code, market=s.market if s else None,
                           sector=s.sector if s else None, title=title, body=summary or title, category=None)
    return judgment_input_hash(spec.model, spec.system_prompt, user)


def pair_hash(spec: JudgeSpec, p: dict) -> str:
    """판정 당시 쌍 행(그때의 제목·요약·종목명)으로 judge 정의의 입력 해시."""
    _, user = build_prompt(code=p["code"], name=p["name"], market=p.get("market"), sector=p.get("sector"),
                           title=p["title"], body=p.get("body") or "", category=p.get("category"))
    return judgment_input_hash(spec.model, spec.system_prompt, user)


def scan_pairs(paths: list[Path], keys: set[str]) -> dict[str, list[dict]]:
    """쌍 파일들에서 ``keys`` 의 행만(같은 키가 여러 파일에 있으면 전부)."""
    out: dict[str, list[dict]] = {}
    prefixes = {k[:36] for k in keys}
    for path in paths:
        with Path(path).open() as fh:
            for line in fh:
                if line[9:45] not in prefixes:  # '{"key": "<uuid>' — 빠른 사전 필터, 아래서 정확히 확인
                    continue
                p = json.loads(line)
                if p["key"] in keys:
                    out.setdefault(p["key"], []).append(p)
    return out


def verify(con, spec: JudgeSpec, rows: list[dict], resolved: dict, n: int, seed: int = 0,
           pairs: list[Path] | None = None) -> dict:
    """input_hash 재계산 — DB 기사로 일치한 수, 원천 차이로 설명된 수, judge 정의로 재현 안 된 키."""
    pool = [r for r in rows if r["key"] in resolved]
    pick = pool if n <= 0 or n >= len(pool) else random.Random(seed).sample(pool, n)
    arts = load_articles_by_id(con, sorted({resolved[r["key"]][0] for r in pick}))
    stocks = load_stocks(con)
    bad = []
    for r in pick:
        aid, code = resolved[r["key"]]
        title, summary = arts.get(aid, ("", ""))
        if recompute_hash(spec, code, title, summary, stocks) != r["input_hash"]:
            bad.append(r)
    explained, unexplained = [], []
    found = scan_pairs(pairs or [], {r["key"] for r in bad}) if bad else {}
    for r in bad:
        if any(pair_hash(spec, p) == r["input_hash"] for p in found.get(r["key"], ())):
            explained.append(r["key"])
        else:
            unexplained.append(r["key"])
    return {"sample": len(pick), "db_matched": len(pick) - len(bad), "source_diff_explained": len(explained),
            "unexplained": len(unexplained), "unexplained_keys": unexplained[:20],
            "explained_keys": explained[:20]}


def build_rows(spec: JudgeSpec, cache: list[dict], errors: list[dict], resolved: dict, by_key: dict,
               submitted: dict, fallback_at: dt.datetime, stats: Counter) -> list[dict]:
    errored = {(e["key"], e.get("batch_id")) for e in errors}
    ok_keys = set()
    out = []
    for r in cache:
        if r["key"] not in resolved:
            continue
        aid, code = resolved[r["key"]]
        b = attribute(r["key"], r["input_hash"], by_key, errored)
        if b is None:
            stats["ok_judged_at_mtime"] += 1
            at, batch = fallback_at, None
        else:
            stats["ok_judged_at_batch"] += 1
            at, batch = b[0], b[1]
        ok_keys.add(r["key"])
        out.append(judgment_row(aid, code, spec.judge, r["input_hash"], output=r["v2"], judged_at=at,
                                judged_at_exact=False, batch_id=batch))
    last_err: dict[str, dict] = {}
    for e in errors:
        if not counts_as_attempt(e):
            stats["error_not_an_attempt_skipped"] += 1
            continue
        if e["key"] in ok_keys:
            stats["error_superseded_by_success"] += 1
            continue
        if e["key"] not in resolved:
            stats["error_unmapped"] += 1
            continue
        prev = last_err.get(e["key"])
        at = submitted.get(e.get("batch_id"))
        if prev is None or (at or fallback_at) >= (submitted.get(prev.get("batch_id")) or fallback_at):
            last_err[e["key"]] = e
    for key, e in last_err.items():
        aid, code = resolved[key]
        at = submitted.get(e.get("batch_id"))
        stats["error_judged_at_batch" if at else "error_judged_at_mtime"] += 1
        h = next((c[2] for c in by_key.get(key, ()) if c[1] == e.get("batch_id") and c[2]), None)
        if h is None:
            stats["error_no_input_hash_skipped"] += 1
            continue
        detail = e.get("detail")
        out.append(judgment_row(aid, code, spec.judge, h, error=f"{e['reason']}: {json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail}",
                                judged_at=at or fallback_at, judged_at_exact=False, batch_id=e.get("batch_id")))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judge", required=True, choices=sorted(KNOWN_JUDGES))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--errors", type=Path, default=None)
    ap.add_argument("--states", type=Path, nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=2000, help=f"input_hash 검증 표본(최소 {MIN_VERIFY}, 0 = 전부)")
    ap.add_argument("--pairs", type=Path, nargs="*", default=[],
                    help="판정 당시 쌍 파일 — DB 기사와 해시가 다른 행의 원인을 가린다")
    ap.add_argument("--dry-run", action="store_true", help="검증·집계만, 쓰지 않는다")
    args = ap.parse_args(argv)
    spec = KNOWN_JUDGES[args.judge]

    cache = load_jsonl(args.cache)
    errors = load_jsonl(args.errors) if args.errors else []
    dup = len(cache) - len({r["key"] for r in cache})
    if dup:
        sys.exit(f"REFUSED: 캐시에 같은 키가 {dup} 번 더 있다 — 어느 판정을 넣을지 규칙이 필요하다")
    con = connect_db()
    kmap = key_map(con)
    resolved, unmapped = resolve_keys({r["key"] for r in cache} | {e["key"] for e in errors}, kmap)
    stats: Counter = Counter(cache_rows=len(cache), error_rows=len(errors), unmapped_keys=len(unmapped))
    stats["unmapped_ok_rows"] = sum(1 for r in cache if r["key"] not in resolved)

    n = 0 if args.sample <= 0 else max(MIN_VERIFY, args.sample)
    v = verify(con, spec, cache, resolved, n, pairs=args.pairs)
    print(f"[{spec.judge}] input_hash 검증: 표본 {v['sample']:,} · DB 기사로 일치 {v['db_matched']:,} · "
          f"원천 차이(당시 쌍 입력으로 재현) {v['source_diff_explained']:,} · 재현 안 됨 {v['unexplained']:,}", flush=True)
    if v["sample"] < min(MIN_VERIFY, len(cache) - stats["unmapped_ok_rows"]) or v["unexplained"]:
        print(json.dumps(v, ensure_ascii=False, indent=1))
        sys.exit("REFUSED: judge 정의가 캐시 입력을 100% 재현하지 못한다 — 이전하지 않는다")

    by_key, submitted = batch_index(args.states)
    mtime = dt.datetime.fromtimestamp(args.cache.stat().st_mtime, tz=dt.UTC)
    rows = build_rows(spec, cache, errors, resolved, by_key, submitted, mtime, stats)
    stats["rows_to_write"] = len(rows)
    if args.dry_run:
        print(json.dumps({"judge": spec.judge, "verify": v, **stats, "dry_run": True}, ensure_ascii=False, indent=1))
        return 0
    register_judge(con, spec)
    counts = write_judgments(con, rows)
    print(json.dumps({"judge": spec.judge, "verify": {k: v[k] for k in ("sample", "db_matched", "source_diff_explained")}, **stats, **counts,
                      "unmapped_examples": unmapped[:5]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
