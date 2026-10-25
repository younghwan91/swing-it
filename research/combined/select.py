"""세 레포 종합 autoresearch — 사후 후보 선택기(루프가 끝난 뒤 메인 세션이 한 번). 에이전트는 이 파일을 고치지도 돌리지도 않는다.

설계 "후보 고르기"(docs/superpowers/specs/2026-10-25-combined-autoresearch-design.md):
1. 풀(``pool``): 원장 RUNS 의 ``eligible`` ∧ ¬void 시도 중 keep 전부 ∪ score 상위 10, 최대 10(score 순). 같은 sha 는 한 번.
2. 거르개(``screen``, 탐색 절반·선택 10폴드 A·B): score > 0, 거래당 순기대값(2배) > 0, 상위 1% 거래 제거 > 0, 최대 기여 한 종목 제거 > 0,
   연 체결 ≥ 20, 같은 메뉴 무작위 파라미터 널 100회의 95 백분위 초과(v4 ``null=("params", seed)``), 원장 score 재현.
   팩터 NW t·DSR(N = 루프 고유 시행 + swing 280)은 기록만 한다.
3. 확인(``confirm``): 시대 C 탐색 절반 — 학습 창 2023-01~2024-12 고원 → 시험 2025-01-02~2026-09-12 한 번. sha 마다 평생 1회.
   폴드가 하나라 σ 가 없다 → 확인 score = 그 폴드의 선택 알파. 최대인 하나를 고른다(동점이면 train.py 줄 수 적은 쪽).
4. 잠금(``lockbox``): 고른 하나만 잠금 절반(half 1)에서 A·B 10폴드 + C 1폴드를 같은 절차로. 바 = 11폴드 score > 0 ∧ 거래당 순기대값(2배) > 0.
   평생 1회(sha 무관). 실패하면 다음 순위로 넘어가지 않는다 → chosen = null.
확인·잠금 원장(CONFIRM.jsonl·LOCKBOX.jsonl)은 작업 트리 + ~/.cache 사본 두 곳에 덧붙이고, 거부 검사는 **두 사본의 합집합**으로 한다
(한쪽을 지워도 다시 열리지 않는다). 결과는 원장에 쓴 뒤에만 출력한다.

사람 열람 바(v4 b8): ``--screen`` 은 풀·거르개만 돌리고(보류 구간을 열지 않는다) 통과한 train.py 를 scratch 에 꺼내 경로를 보여 준다.
사람이 그 파일들을 읽고(e104063 구조 재제출 여부 포함 — program·PREREG) ``--confirm --reviewed <sha16>,<sha16>…`` 로 통과 목록 **전부**를
정확히 적어야 확인·잠금이 열린다. 거르개 뒤 원장이 바뀌었으면 거부한다(--screen 다시).

사용(작업 트리 루트, 루프가 멈춘 뒤):
  uv run --with numba python research/combined/select.py --screen
  uv run --with numba --extra pg python research/combined/select.py --confirm --reviewed <sha16>[,<sha16>…]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from research.combined import prepare as P  # noqa: E402
from research.combined.data import ERAS_C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIRM, LOCKBOX, SCREEN = (os.path.join(HERE, f) for f in ("CONFIRM.jsonl", "LOCKBOX.jsonl", "SCREEN.jsonl"))
SELECTION = os.path.join(HERE, "SELECTION.json")
SCRATCH = os.path.expanduser("~/.cache/cc-swing/scratch/combined_select")   # 풀 commit 의 train.py(사람이 읽는 사본)
TRAIN_REL = "research/combined/train.py"
POOL_MAX, POOL_TOP = 10, 10
N_NULL, NULL_Q = 100, 0.95
MIN_TRADES_PER_YEAR = 20
N_SWING_TRIALS = 280                                                       # 같은 데이터의 swing 루프 시행(DSR N 에 더한다)
SEARCH_ERAS, CONFIRM_ERAS = ("A", "B"), ("C",)


# ---------------------------------------------------------------- 원장(두 사본)

def _rows_both(path: str) -> list:
    """작업 트리 + 사본 두 곳의 행 합집합 — 거부 검사용(``P._read_jsonl`` 은 긴 쪽 하나만 본다)."""
    out = []
    for p in (path, os.path.join(P.SHADOW, os.path.basename(path))):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out += [json.loads(x) for x in f if x.strip()]
    return out


def _sha_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


@contextmanager
def _eras(names: tuple):
    """walk_forward_c 가 도는 시대를 잠시 바꾼다(prepare.ERAS 는 루프 시대 A·B — 확인 C 는 여기서만)."""
    old = P.ERAS
    P.ERAS = {e: ERAS_C[e] for e in names}
    try:
        yield
    finally:
        P.ERAS = old


# ---------------------------------------------------------------- 적재·git

def load_search(era: str) -> dict:
    from research.combined.data import load
    return load(era, 0)


def load_locked(era: str) -> dict:
    """잠금 절반(half 1) — DB 에서 메모리로만. DB 로 다시 만든 탐색 절반이 캐시와 다르면 시점이 어긋난 것 → 거부(v4 final 과 같다)."""
    import psycopg2

    from research.combined.data import _slice, build_era
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    F = build_era(con, era)
    if not np.array_equal(_slice(F, 0)["C"], load_search(era)["C"], equal_nan=True):
        raise SystemExit(f"시대 {era}: DB 로 다시 만든 탐색 절반이 캐시와 다르다 — 캐시·매니페스트 재구축 뒤 다시(정정 문단)")
    return _slice(F, 1)


def _git(*a: str) -> str:
    p = subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)}: {p.stderr.strip()}")
    return p.stdout


def train_commits() -> dict:
    """train.py 내용 sha(앞 16) → 그 내용을 가진 commit. 루프는 버린 시도를 ``reset --hard`` 하므로 reflog 까지 본다(모든 워크트리)."""
    out = {}
    for c in _git("rev-list", "--all", "--reflog", "--", TRAIN_REL).split():
        try:
            blob = _git("show", f"{c}:{TRAIN_REL}")
        except RuntimeError:
            continue                                                        # 그 commit 에서 파일이 지워졌다
        out.setdefault(hashlib.sha256(blob.encode()).hexdigest()[:16], c)
    return out


def extract_train(commit: str, sha: str) -> str:
    """``git show <commit>:research/combined/train.py`` → scratch/<sha>/train.py. 내용 sha 가 다르면 거부."""
    d = os.path.join(SCRATCH, sha)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "train.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_git("show", f"{commit}:{TRAIN_REL}"))
    if _sha_of(p) != sha:
        raise SystemExit(f"commit {commit} 의 train.py sha {_sha_of(p)} ≠ 원장 {sha}")
    return p


# ---------------------------------------------------------------- 1. 풀

def pool(runs: list) -> list:
    """eligible ∧ ¬void 평가 행(참고 실행·accept 행 제외, sha 당 첫 행) 중 keep 전부 ∪ score 상위 10 → score 순 최대 10."""
    keep_sha = {r["train_sha"] for r in runs if r.get("accept")}
    rows, seen = [], set()
    for r in runs:
        sha = r.get("train_sha")
        if r.get("accept") or r.get("reference") or "folds" not in r or sha in seen:
            continue
        seen.add(sha)
        if r.get("eligible") is True and not r.get("void") and isinstance(r.get("score"), (int, float)) and math.isfinite(r["score"]):
            rows.append(r)
    rows.sort(key=lambda r: -r["score"])                                   # 안정 정렬 — 동점은 원장 순서
    keep = [r for r in rows if r.get("beats_best") is True or r["train_sha"] in keep_sha]
    union = {r["train_sha"]: r for r in [*keep, *rows[:POOL_TOP]]}
    return sorted(union.values(), key=lambda r: -r["score"])[:POOL_MAX]


# ---------------------------------------------------------------- 2. 거르개

def concentration(D_by_era: dict, res: dict, train_path: str) -> dict:
    return P.concentration(D_by_era, res, train_path)


def factor_record(D_by_era: dict, res: dict) -> dict:
    """팩터(KOSPI·KOSDAQ·MOM·SMB·LV) 회귀 절편과 NW t — 기록만(바 아님)."""
    try:
        F = pd.concat([P.factor_months(D_by_era[e]) for e in P.ERAS])
        a, t = P.factor_t(res["months"], F)
        return {"alpha_m": a, "t_nw": t}
    except Exception as e:  # noqa: BLE001 — 기록용 값이라 실패는 사유를 남기고 넘어간다
        return {"alpha_m": float("nan"), "t_nw": float("nan"), "error": repr(e)}


def dsr_record(res: dict, n_trials: int) -> dict:
    """월 선택 수익의 DSR(N = 루프 고유 시행 + swing 280) — 기록만(바 아님). krx_quant_core(v4 와 같은 구현)."""
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
    ms = [m for m in res["months"] if len(m)]
    if not ms:
        return {"n_trials": n_trials, "error": "월 수익 없음"}
    return {**deflated_sharpe_from_sample(pd.concat(ms)["sel"].to_numpy(), n_trials), "n_trials": n_trials}


def screen(rec: dict, D_by_era: dict, train_path: str, workers: int = 1, n_null: int = N_NULL) -> dict:
    """탐색 절반·선택 구간 거르개. 앞 바가 떨어지면 널은 돌리지 않는다(널 바 = False 로 기록)."""
    sha = rec["train_sha"]
    if _sha_of(train_path) != sha:
        raise SystemExit(f"train.py sha {_sha_of(train_path)} ≠ 후보 {sha}")
    with _eras(SEARCH_ERAS):
        res = P.walk_forward_c(D_by_era, workers, train_path)
        s = P.summarize_c(res)
        conc = concentration(D_by_era, res, train_path)
        bars = {"repro_ledger": bool(math.isclose(s["score"], rec["score"], rel_tol=1e-9, abs_tol=1e-12)),
                "score_pos": bool(s["score"] > 0 and not s["void"]),
                "ev_net_2x_pos": bool(s["ev_net_2x"] > 0),
                "top1pct_removed_pos": bool(conc["top1pct"] > 0),
                "top_stock_removed_pos": bool(conc["top_stock"] > 0),
                "trades_per_year_ge_20": bool(s["oos_trades_per_year"] >= MIN_TRADES_PER_YEAR),
                "beats_null_p95": False}
        nulls, nvoid = [], 0
        if all(v for k, v in bars.items() if k != "beats_null_p95"):
            for r in range(n_null):
                ns = P.summarize_c(P.walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], null=("params", r + 1)))
                nvoid += bool(ns["void"])
                if np.isfinite(ns["raw_score"]):
                    nulls.append(float(ns["raw_score"]))
            p95 = float(np.quantile(nulls, NULL_Q)) if nulls else float("nan")
            bars["beats_null_p95"] = bool(nulls and s["score"] > p95)
        else:
            p95 = float("nan")
        n_dsr = P.trial_state()["n"] + N_SWING_TRIALS
        out = {"train_sha": sha, "commit": rec.get("commit"), "pass": all(bars.values()), "bars": bars, "score": s["score"],
               "ledger_score": rec["score"], "void": s["void"], "ev_net_2x": s["ev_net_2x"], "payoff_ratio": s["payoff_ratio"],
               "oos_trades": s["oos_trades"], "trades_per_year": s["oos_trades_per_year"], "folds_pos": s["folds_pos"],
               "top1pct_removed": conc["top1pct"], "top_stock_removed": conc["top_stock"], "top10d_removed": conc.get("top10d"),
               "null_kind": "params", "null_n": len(nulls), "null_void": nvoid, "null_p95": p95,
               "null_median": float(np.median(nulls)) if nulls else float("nan"),
               "factor": factor_record(D_by_era, res), "dsr": dsr_record(res, n_dsr), "train_lines": _lines(train_path),
               "folds": [{k: f[k] for k in ("era", "test", "params", "alpha", "n")} for f in res["folds"]]}
    return out


# ---------------------------------------------------------------- 3. 확인(sha 당 1회)

def _ev(res: dict) -> "tuple[float, float, int]":
    tr = pd.concat(res["trades"]) if res["trades"] else pd.DataFrame(columns=["ret"])
    win, loss = tr.loc[tr["ret"] > 0, "ret"], tr.loc[tr["ret"] <= 0, "ret"]
    pr = float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan")
    return (float(tr["ret"].mean()) if len(tr) else float("nan")), pr, int(len(tr))


def confirm(sha: str, train_path: str, workers: int = 1, commit: "str | None" = None) -> dict:
    """시대 C 탐색 절반 — 학습 창 고원 → 2025-01-02~2026-09-12 시험 한 번. CONFIRM.jsonl(+ 사본)에 덧붙인 뒤 돌려준다."""
    if _rows_both(LOCKBOX):
        raise SystemExit("잠금 절반이 이미 열렸다 — 선택은 끝났다, 확인 구간을 더 열지 않는다")
    if any(r.get("train_sha") == sha for r in _rows_both(CONFIRM)):
        raise SystemExit(f"확인 구간은 {sha} 에 이미 열렸다(CONFIRM.jsonl) — 두 번 열지 않는다")
    if _sha_of(train_path) != sha:
        raise SystemExit(f"train.py sha {_sha_of(train_path)} ≠ {sha} — 사람이 읽은 그 파일만 연다")
    with _eras(CONFIRM_ERAS):
        res = P.walk_forward_c({"C": load_search("C")}, workers, train_path)
    f = res["folds"][0]
    ev, pr, n = _ev(res)
    alpha = float(f["alpha"])
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "commit": commit, "era": "C", "half": 0,
           "test": [str(x) for x in f["test"]], "params": f["params"], "alpha": alpha,
           "score": alpha if math.isfinite(alpha) else float("nan"), "excess": f.get("excess"), "n": n,
           "trades_per_year": n / max(float(f.get("years") or 0), 1e-9), "ev_net_2x": ev, "payoff_ratio": pr,
           "trade_key": f.get("trade_key"), "train_lines": _lines(train_path), "evaluator": P.PREREG_TAG}
    P._append(CONFIRM, row)                                                # 쓰고 나서 돌려준다(출력은 그 뒤)
    return row


# ---------------------------------------------------------------- 4. 잠금(평생 1회)

def lockbox(sha: str, train_path: str, workers: int = 1, commit: "str | None" = None) -> dict:
    """잠금 절반 — A·B 10폴드 + C 1폴드(같은 절차). 바 = 11폴드 score > 0 ∧ 거래당 순기대값(2배) > 0. LOCKBOX.jsonl(+ 사본)."""
    if _rows_both(LOCKBOX):
        raise SystemExit("잠금 절반은 이미 열렸다(LOCKBOX.jsonl) — 최대 1회, 다음 순위로 넘어가지 않는다")
    if not any(r.get("train_sha") == sha for r in _rows_both(CONFIRM)):
        raise SystemExit(f"{sha} 의 확인 기록이 없다 — 확인 구간 순위로 고른 것만 잠금을 연다")
    if _sha_of(train_path) != sha:
        raise SystemExit(f"train.py sha {_sha_of(train_path)} ≠ {sha} — 사람이 읽은 그 파일만 연다")
    D = {e: load_locked(e) for e in (*SEARCH_ERAS, *CONFIRM_ERAS)}
    with _eras(SEARCH_ERAS):
        ab = P.walk_forward_c({e: D[e] for e in SEARCH_ERAS}, workers, train_path)
        s_ab = P.summarize_c(ab)
    with _eras(CONFIRM_ERAS):
        c = P.walk_forward_c({e: D[e] for e in CONFIRM_ERAS}, workers, train_path)
    both = {"folds": ab["folds"] + c["folds"], "trades": ab["trades"] + c["trades"]}
    score = P.score_of([f["alpha"] for f in both["folds"]])
    ev, pr, n = _ev(both)
    bars = {"score_pos": bool(score > 0), "ev_net_2x_pos": bool(ev > 0)}
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "commit": commit, "half": 1, "score": score,
           "score_ab": P.score_of([f["alpha"] for f in ab["folds"]]), "alpha_c": c["folds"][0]["alpha"], "void_ab": s_ab["void"],
           "ev_net_2x": ev, "ev_net_2x_ab": _ev(ab)[0], "ev_net_2x_c": _ev(c)[0], "payoff_ratio": pr, "n": n,
           "bars": bars, "pass": all(bars.values()), "evaluator": P.PREREG_TAG,
           "folds": [{k: f[k] for k in ("era", "test", "params", "alpha", "n")} for f in both["folds"]]}
    P._append(LOCKBOX, row)
    return row


# ---------------------------------------------------------------- 진입점

def _need_loop_done() -> dict:
    st = P.trial_state()
    if not st["stop"]:
        raise SystemExit(f"루프가 아직 멈추지 않았다(고유 시행 {st['n']}, 마지막 keep 뒤 {st['since_keep']}) — 후보 고르기는 루프 뒤 한 번")
    return st


def run_screen(workers: int) -> dict:
    """풀 → 거르개(탐색 절반만). 통과한 train.py 를 scratch 에 꺼내 사람이 읽을 경로를 남긴다. 보류 구간은 열지 않는다."""
    P.integrity()
    st = _need_loop_done()
    runs = P._read_jsonl(P.RUNS)
    pl = pool(runs)
    commits = train_commits()
    D = {e: load_search(e) for e in SEARCH_ERAS}
    results, review = {}, []
    for r in pl:
        sha, commit = r["train_sha"], commits.get(r["train_sha"])
        if commit is None:
            results[sha] = {"train_sha": sha, "commit": None, "pass": False, "reason": "train.py 내용의 commit 을 찾지 못함(reflog 밖)"}
            continue
        path = extract_train(commit, sha)
        results[sha] = screen({**r, "commit": commit}, D, path, workers=workers)
        P.log(f"거르개 {sha} ({commit[:10]}): pass {results[sha]['pass']}")
        if results[sha]["pass"]:
            review.append({"train_sha": sha, "commit": commit, "path": path, "lines": _lines(path)})
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_trials": st["n"], "pool": [r["train_sha"] for r in pl],
           "pool_rows": [{k: r.get(k) for k in ("train_sha", "score", "beats_best", "ev_net_2x", "oos_trades_per_year", "conc")} for r in pl],
           "results": results, "survivors": [x["train_sha"] for x in review], "review": review, "evaluator": P.PREREG_TAG}
    P._append(SCREEN, rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    print("\n사람 열람: 아래 파일을 읽고(e104063 구조 재제출 여부 포함) 통과 목록 전부를 --reviewed 에 적는다.")
    for x in review:
        print(f"  {x['train_sha']}  {x['commit'][:10]}  {x['lines']}줄  {x['path']}")
    print(f"  → select.py --confirm --reviewed {','.join(rec['survivors'])}")
    return rec


def run_confirm(reviewed: "list[str]", workers: int) -> dict:
    """확인(통과 전부, sha 당 1회) → 확인 score 최대(동점이면 짧은 train.py) → 그 하나만 잠금 1회 → SELECTION.json."""
    P.integrity()
    if _rows_both(LOCKBOX):
        raise SystemExit("잠금 절반이 이미 열렸다 — 선택은 끝났다(SELECTION.json)")
    st = _need_loop_done()
    scr = P._read_jsonl(SCREEN)
    if not scr:
        raise SystemExit("거르개 기록이 없다 — 먼저 select.py --screen")
    scr = scr[-1]
    if [r["train_sha"] for r in pool(P._read_jsonl(P.RUNS))] != scr["pool"] or st["n"] != scr["n_trials"]:
        raise SystemExit("거르개 뒤 원장이 바뀌었다 — select.py --screen 을 다시 돌리고 다시 열람")
    survivors = scr["survivors"]
    if sorted(set(reviewed)) != sorted(survivors) or len(reviewed) != len(set(reviewed)):
        raise SystemExit(f"--reviewed {sorted(reviewed)} ≠ 거르개 통과 {sorted(survivors)} — 통과한 train.py 전부를 사람이 읽고 정확히 적는다(v4 b8)")
    sel = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "chosen": None, "frozen_params": None, "picked": None, "reasons": [],
           "reviewed": sorted(reviewed), "pool": scr["pool_rows"], "screen": scr["results"], "confirm": [], "lockbox": None,
           "evaluator": P.PREREG_TAG}
    if not survivors:
        sel["reasons"].append(f"거르개 통과 없음(풀 {len(scr['pool'])}) — 확인·잠금을 열지 않았다. 판정: 후보 없음")
        _write_selection(sel)
        return sel
    by = {x["train_sha"]: x for x in scr["review"]}
    done = {r["train_sha"]: r for r in _rows_both(CONFIRM)}
    paths = {}
    for sha in survivors:                                                  # 이어 하기: 이미 연 sha 는 기록을 읽기만 한다
        paths[sha] = extract_train(by[sha]["commit"], sha)
        sel["confirm"].append(done[sha] if sha in done else confirm(sha, paths[sha], workers, commit=by[sha]["commit"]))
    order = sorted(sel["confirm"], key=lambda r: (-(r["score"] if math.isfinite(r["score"]) else -math.inf), r["train_lines"]))
    top = order[0]
    sel["picked"] = {"train_sha": top["train_sha"], "commit": top["commit"], "confirm_score": top["score"], "train_lines": top["train_lines"]}
    sel["reasons"].append(f"확인 score 최대 {top['train_sha']}(동점이면 train.py 줄 수 적은 쪽) — 확인 {len(order)}개 중")
    lk = lockbox(top["train_sha"], paths[top["train_sha"]], workers, commit=top["commit"])
    sel["lockbox"] = lk
    if lk["pass"]:
        sel["chosen"] = {"train_sha": top["train_sha"], "commit": top["commit"]}
        sel["frozen_params"] = top["params"]
        sel["reasons"].append("잠금 절반 바 통과(11폴드 score > 0 ∧ 거래당 순기대값 2배 > 0) — frozen_params = 시대 C 학습 창(2023~2024) 고원 값")
    else:
        sel["reasons"].append(f"잠금 절반 바 실패 {lk['bars']} — 다음 순위로 넘어가지 않는다. 판정: 후보 없음")
    _write_selection(sel)
    return sel


def _write_selection(sel: dict) -> None:
    with open(SELECTION, "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: sel[k] for k in ("chosen", "frozen_params", "picked", "reasons")}, ensure_ascii=False, indent=1, default=str))


def main(argv: "list[str] | None" = None) -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--screen", action="store_true", help="풀 → 거르개(탐색 절반만, 보류 구간 안 엶)")
    g.add_argument("--confirm", action="store_true", help="확인(통과 전부) → 1위 잠금 1회 → SELECTION.json")
    ap.add_argument("--reviewed", default=None, help="--confirm: 사람이 읽은 거르개 통과 train sha(앞 16) 전부, 쉼표로")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    P._G["t0"] = time.time()
    if a.screen:
        run_screen(a.workers)
    else:
        if a.reviewed is None:
            raise SystemExit("--confirm 은 --reviewed 가 필요하다(거르개 통과 전부)")
        run_confirm([s.strip() for s in a.reviewed.split(",") if s.strip()], a.workers)


if __name__ == "__main__":
    main()
