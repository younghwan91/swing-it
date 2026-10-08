"""Semantic ground truth for the v2 extraction fields, and Haiku 5.5 prompt evaluation against it.

swing-it 스윙 2호(LLM 지속성) 판정 프롬프트를 정하는 단계. **정답은 주가가 아니라 기사 의미다** — 상위 모델
(Opus 5.5, 자세한 채점표 + 생각)이 기사 내용만 보고 매긴다. 수익 데이터는 이 파일 어디에도 없다.

    # 1) 400쌍 표본(2026-03~09, 연구 구간 밖) — 4.5 판정상 main 260 + 나머지 140, train/val/test
    python -m swing_it.news.labelset sample
    # 2) 교사 라벨(Opus 5.5)
    python -m swing_it.news.labelset teacher
    # 3) Haiku 5.5 프롬프트 변형 채점 (--split train 으로 고르고, val 은 마지막에 한 번, test 는 한 번만)
    python -m swing_it.news.labelset eval --variant v2 --split train

**출처:** daytrade-it ``scripts/eval/persistence_labelset.py`` (브랜치 ``research/news-persistence-judge`` @ ba0662a).
2026-10-08 의 실행 결과(``labelset/sample.jsonl``·``teacher.jsonl``·``runs/*.jsonl``·``runs/summary.jsonl``)는
``data/eval/persistence_swing/labelset/`` 에 사본으로 있다 — 사전등록 수정 1 의 표가 그 숫자다. 바뀐 것: 프롬프트·채점표를 :mod:`swing_it.news.prompt`
에서 읽고, ``sample`` 의 입력(daytrade-it 의 Haiku 4.5 판정 쌍·캐시 — swing-it 으로 옮기지 않았다)을 인자로 받는다.
표본은 이미 뽑혀 있으므로 ``sample`` 을 다시 돌릴 일은 없다(다시 돌리면 같은 seed 로 같은 표본이 나와야 한다).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from swing_it.news.prompt import (
    _SYSTEM,
    RUBRIC,
    SchemaError,
    build_prompt,
    parse_features,
)

API = "https://api.anthropic.com/v1/messages"
D = Path("data/eval/persistence_swing")
#: ``sample`` 의 입력 — daytrade-it 데이터(이 레포로 옮기지 않았다): 2026-03~09 운영 사전필터 쌍과 Haiku 4.5 판정.
_DT = Path.home() / "git" / "daytrade-it" / "data" / "eval"
PAIRS = _DT / "universe_pairs_n300_prefiltered.jsonl"
CACHE45 = _DT / "cache" / "universe_v2.jsonl"
LABELSET = D / "labelset"
SAMPLE = LABELSET / "sample.jsonl"
TEACHER = LABELSET / "teacher.jsonl"
#: 변형별 채점 결과 ``<variant>_<split>_r<rep>.jsonl`` 과 요약 ``summary.jsonl``.
RUNS = LABELSET / "runs"
FIELDS = ("subject", "persistence", "sentiment_direction", "freshness", "reports_price_move")

def _key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        sys.exit("ANTHROPIC_API_KEY 필요")
    return k


def _post(body: dict) -> dict:
    h = {"x-api-key": _key(), "anthropic-version": "2023-06-01", "content-type": "application/json"}
    for attempt in range(4):
        try:
            r = httpx.post(API, headers=h, json=body, timeout=180)
            if r.status_code in (429, 500, 502, 503, 529):
                raise httpx.HTTPStatusError("retry", request=r.request, response=r)
            return r.json()
        except (httpx.HTTPError, ValueError):
            if attempt == 3:
                raise
            import time

            time.sleep(5 * (attempt + 1))
    return {}


def _text(resp: dict) -> str:
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


def cmd_sample(args) -> None:
    v45 = {}
    for line in Path(args.cache45).open():
        r = json.loads(line)
        v45[r["key"]] = r["v2"]
    pairs = [json.loads(line) for line in Path(args.pairs).open()]
    pairs = [p for p in pairs if p["key"] in v45 and "2026-03" <= p["day"][:7] <= "2026-09"]
    rng = random.Random(args.seed)
    main = [p for p in pairs if v45[p["key"]]["subject"] == "main"]
    rest = [p for p in pairs if v45[p["key"]]["subject"] != "main"]
    pick = rng.sample(main, 260) + rng.sample(rest, 140)
    rng.shuffle(pick)
    D.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as fh:
        for i, p in enumerate(pick):
            split = "train" if i % 2 == 0 else ("val" if (i // 2) % 2 == 0 else "test")
            fh.write(
                json.dumps({**p, "split": split, "v45": v45[p["key"]]}, ensure_ascii=False) + "\n"
            )
    print(f"sample {len(pick)} -> {SAMPLE} (pairs in window {len(pairs)}, main {len(main)})")


def _teacher_one(p: dict) -> dict:
    _, user = build_prompt(
        code=p["code"],
        name=p["name"],
        market=p.get("market"),
        sector=p.get("sector"),
        title=p["title"],
        body=p.get("body") or "",
        category=p.get("category"),
    )
    system = (
        _SYSTEM
        + "\n\n"
        + RUBRIC
        + '\n\n스키마 필드에 더해 "rationale"(한두 문장, 한국어)을 넣어라.'
    )
    body = {
        "model": "claude-opus-5-5",
        "max_tokens": 8000,
        "system": system,
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": user}],
    }
    resp = _post(body)
    txt = _text(resp)
    out = {"key": p["key"], "usage": resp.get("usage", {}), "stop": resp.get("stop_reason")}
    try:
        start, end = txt.index("{"), txt.rindex("}") + 1
        raw = json.loads(txt[start:end])
        rationale = raw.pop("rationale", "")
        out["teacher"] = {**parse_features(raw).__dict__, "rationale": rationale}
    except (ValueError, SchemaError, KeyError) as exc:
        out["error"] = f"{type(exc).__name__}: {txt[:200]}"
    return out


def cmd_teacher(args) -> None:
    sample = [json.loads(line) for line in SAMPLE.open()]
    done = {json.loads(line)["key"] for line in TEACHER.open()} if TEACHER.exists() else set()
    todo = [p for p in sample if p["key"] not in done]
    usage = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex, TEACHER.open("a") as fh:
        for r in ex.map(_teacher_one, todo):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            for k in ("input_tokens", "output_tokens"):
                usage[k] += int(r.get("usage", {}).get(k, 0) or 0)
    usd = usage["input_tokens"] * 4 / 1e6 + usage["output_tokens"] * 20 / 1e6
    print(f"teacher labelled {len(todo)} · tokens {dict(usage)} · ~${usd:.2f}")


# --- Haiku 5.5 prompt variants -----------------------------------------------------------

V3_EXTRA = """

필드 정의 보충(엄격히 따른다):
- persistence: one_off = 단발 사건·시황·루머·단기 테마·실적 무관 소식·소규모 단건 계약.
  multi_quarter = 다년 공급계약·수주잔고·신제품/신약 판매·증설 가동·가격 인상처럼 여러 분기 매출에 반영.
  structural = 신시장 진입·사업 전환·대형 M&A·규제 체제 변화·핵심 기술 확보처럼 경쟁 지위가 장기적으로 바뀜.
- sentiment_direction 은 주가가 아니라 회사 *사업*에 대한 방향이다. 호재 사실(수주·증설·허가)이면 1.
- freshness: 사실을 처음 전하면 new. 해설·칼럼·여러 기사 재가공이면 repeat."""


# --- LLM 없는 기준선: 키워드 규칙 -----------------------------------------------------------
# LLM 이 이 규칙보다 확실히 나아야 LLM 을 쓸 이유가 있다. 규칙은 채점표 정의를 그대로 키워드로 옮겼다(정답 보고 고치지 않는다).
import re as _re  # noqa: E402 — 원문 배치 그대로

_PRICE = _re.compile(
    r"상한가|하한가|급등|급락|신고가|신저가|주가\s*[+-]?\d|강세|약세|순매수|순매도|시황|마감|출렁"
)
_STRUCT = _re.compile(
    r"인수|합병|M&A|지분\s*인수|경영권|신사업|진출|전환|플랫폼|독점|규제\s*완화|법\s*개정|핵심\s*기술|원천\s*기술"
)
_MULTIQ = _re.compile(
    r"공급\s*계약|수주|계약\s*체결|증설|양산|가동|출시|판매\s*개시|허가|승인|가격\s*인상|장기\s*계약|수주잔고|실적\s*개선"
)
_POS = _re.compile(r"수주|계약|증설|양산|출시|허가|승인|개선|최대|흑자|성장|확대|인수|진출|수혜")
_NEG = _re.compile(r"적자|감소|하락|취소|해지|소송|제재|리콜|부진|악화|중단|철회")


def rules_label(p: dict) -> dict:
    t = f"{p['title']} {p.get('body') or ''}"
    title = p["title"]
    name = p.get("name") or ""
    subject = (
        "main"
        if name and name in title[: max(12, len(name) + 6)]
        else ("partial" if name in title else "mention")
    )
    price = bool(_PRICE.search(title))
    pers = (
        "structural" if _STRUCT.search(t) else ("multi_quarter" if _MULTIQ.search(t) else "one_off")
    )
    pos, neg = bool(_POS.search(t)), bool(_NEG.search(t))
    sent = 0 if price or pos == neg else (1 if pos else -1)
    fresh = "repeat" if _re.search(r"칼럼|사설|기고|인터뷰|전망|분석|해설", title) else "new"
    return {
        "subject": subject,
        "persistence": pers,
        "sentiment_direction": sent,
        "freshness": fresh,
        "reports_price_move": price,
    }


PROMPTS = (
    D / "prompts"
)  # vN.txt = 운영 _SYSTEM 뒤에 붙는 보충(정의·예시). v2 = 보충 없음(운영 그대로)


def haiku_request(p: dict, variant: str) -> dict:
    system, user = build_prompt(
        code=p["code"],
        name=p["name"],
        market=p.get("market"),
        sector=p.get("sector"),
        title=p["title"],
        body=p.get("body") or "",
        category=p.get("category"),
    )
    if variant == "v3":
        system = system + V3_EXTRA
    elif variant != "v2":
        system = system + "\n\n" + (PROMPTS / f"{variant}.txt").read_text(encoding="utf-8").strip()
    return {
        "model": "claude-haiku-5-5",
        "max_tokens": 400,
        "system": system,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": user}],
    }


def _haiku_one(arg) -> tuple[str, dict | None]:
    p, variant = arg
    txt = _text(_post(haiku_request(p, variant)))
    try:
        return p["key"], parse_features(txt).__dict__
    except (SchemaError, ValueError):
        return p["key"], None


def cmd_eval(args) -> None:
    sample = {p["key"]: p for p in map(json.loads, SAMPLE.open())}
    teacher = {r["key"]: r["teacher"] for r in map(json.loads, TEACHER.open()) if "teacher" in r}
    keys = [k for k, p in sample.items() if p["split"] == args.split and k in teacher]
    if args.variant == "rules":
        summ = report(
            keys, teacher, {k: rules_label(sample[k]) for k in keys}, sample, f"rules {args.split}"
        )
        RUNS.mkdir(parents=True, exist_ok=True)
        with (RUNS / "summary.jsonl").open("a") as fh:
            fh.write(json.dumps({"variant": "rules", "split": args.split, "rep": 1, **summ}) + "\n")
        return
    out_path = RUNS / f"{args.variant}_{args.split}_r{args.rep}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = {r["key"]: r["v"] for r in map(json.loads, out_path.open())} if out_path.exists() else {}
    todo = [(sample[k], args.variant) for k in keys if k not in have]
    with ThreadPoolExecutor(max_workers=args.workers) as ex, out_path.open("a") as fh:
        for k, v in ex.map(_haiku_one, todo):
            have[k] = v
            fh.write(json.dumps({"key": k, "v": v}, ensure_ascii=False) + "\n")
    summ = report(keys, teacher, have, sample, f"haiku55 {args.variant} {args.split} r{args.rep}")
    with (RUNS / "summary.jsonl").open("a") as fh:
        fh.write(
            json.dumps({"variant": args.variant, "split": args.split, "rep": args.rep, **summ})
            + "\n"
        )


def is_event(v: dict) -> bool:
    """스윙 사건 정의(사전등록) — main · 호재 · 비반복 · 주가보도 아님."""
    return (
        v["subject"] == "main"
        and v["sentiment_direction"] == 1
        and v["freshness"] != "repeat"
        and not v["reports_price_move"]
    )


def report(keys, teacher, cand, sample, name) -> dict:
    ok = [k for k in keys if cand.get(k)]
    print(f"\n== {name} · n={len(keys)} parsed={len(ok)}")
    acc = {}
    for f in FIELDS:
        acc[f] = sum(cand[k][f] == teacher[k][f] for k in ok) / max(1, len(ok))
        b = sum(sample[k]["v45"][f] == teacher[k][f] for k in ok) / max(1, len(ok))
        print(f"  {f:20s} {acc[f]:6.1%}   (haiku45 {b:6.1%})")
    te = {k for k in ok if is_event(teacher[k])}
    ce = {k for k in ok if is_event(cand[k])}
    tp = len(te & ce)
    prec, rec = tp / max(1, len(ce)), tp / max(1, len(te))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    pers = sum(cand[k]["persistence"] == teacher[k]["persistence"] for k in te) / max(1, len(te))
    h45e = {k for k in ok if is_event(sample[k]["v45"])}
    tp45 = len(te & h45e)
    f1_45 = 2 * tp45 / max(1, len(te) + len(h45e))
    pers45 = sum(sample[k]["v45"]["persistence"] == teacher[k]["persistence"] for k in te) / max(
        1, len(te)
    )
    tm = [k for k in ok if teacher[k]["subject"] == "main"]

    def macro(get):
        f1s = []
        for c in ("one_off", "multi_quarter", "structural"):
            tp_ = sum(get(k) == c and teacher[k]["persistence"] == c for k in tm)
            pp = sum(get(k) == c for k in tm)
            gt = sum(teacher[k]["persistence"] == c for k in tm)
            pr_, rc_ = tp_ / max(1, pp), tp_ / max(1, gt)
            f1s.append(0.0 if tp_ == 0 else 2 * pr_ * rc_ / (pr_ + rc_))
        return sum(f1s) / 3, f1s

    pm, pm_c = macro(lambda k: cand[k]["persistence"])
    pm45, _ = macro(lambda k: sample[k]["v45"]["persistence"])
    print(
        f"  PERSIST macro-F1 | teacher main (n={len(tm)}): {pm:.3f} {[round(x, 2) for x in pm_c]}   haiku45 {pm45:.3f}"
    )
    print(
        f"  EVENT  F1 {f1:.3f} (P {prec:.2f} R {rec:.2f}, teacher {len(te)} · cand {len(ce)})   haiku45 F1 {f1_45:.3f}"
    )
    print(f"  PERSISTENCE | teacher events: {pers:.1%}   haiku45 {pers45:.1%}")
    print(
        "  teacher events persistence:",
        Counter(teacher[k]["persistence"] for k in te),
        "| cand:",
        Counter(cand[k]["persistence"] for k in te),
    )
    return {
        "n": len(keys),
        "parsed": len(ok),
        "event_f1": round(f1, 4),
        "pers_acc": round(pers, 4),
        "pers_macro_main": round(pm, 4),
        **{f"acc_{f}": round(v, 4) for f, v in acc.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--pairs", default=str(PAIRS), help="2026-03~09 쌍(daytrade-it 운영 사전필터)")
    s.add_argument("--cache45", default=str(CACHE45), help="그 쌍의 Haiku 4.5 판정 캐시")
    t = sub.add_parser("teacher")
    t.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("eval")
    e.add_argument("--variant", default="v2")
    e.add_argument("--rep", type=int, default=1)
    e.add_argument("--split", default="train", choices=("train", "val", "test"))
    e.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"sample": cmd_sample, "teacher": cmd_teacher, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
