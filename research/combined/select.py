"""세 레포 종합 autoresearch — 사후 후보 선택기(루프가 끝난 뒤 메인 세션이 한 번). 에이전트는 이 파일을 고치지도 돌리지도 않는다.

설계 "후보 고르기"(docs/superpowers/specs/2026-10-25-combined-autoresearch-design.md):
1. 풀(``pool``): 원장 RUNS 의 ``eligible`` ∧ ¬void ∧ commit 있음 ∧ ¬dirty 시도 중 keep 전부 ∪ score 상위 10, 최대 10(score 순). 같은 sha 는 한 번.
   dirty·commit 없는 행은 train.py 를 commit 에서 되살릴 수 없어 풀에서 빼고 사유를 남긴다(``pool_rejects``).
2. 거르개(``screen``, 탐색 절반·선택 10폴드 A·B): score > 0, 거래당 순기대값(2배) > 0, 상위 1% 거래 제거 > 0, 최대 기여 한 종목 제거 > 0,
   연 체결 ≥ 20, 원장 score 재현, **솎기 널**(같은 폴드 파라미터·같은 cfg, 후보를 시험 구간 월별 통과율 그대로 무작위로 — ``null=("thin", seed)``)
   300회의 95 백분위 초과(유한 널 ≥ 270). 같은 메뉴 무작위 파라미터 널 100회(``("params", seed)``)는 기록만(정정 0 M2).
   팩터 NW t·DSR(N = 루프 고유 시행 + swing 280)은 기록만 한다 — GUARDRAILS 팩터 t > 2 바는 적용하지 않고 판정은 전진이 한다.
3. 사람 열람: 거르개 통과 전부를 읽고 ``--reviewed`` 에 전부 적는다. 열거된 사유(``EXCLUDE_REASONS``)에 걸리면 ``--exclude sha:사유`` —
   제외는 확인 구간을 열기 전에 SCREEN 에 review 행으로 기록되고, 확인이 한 번이라도 열린 뒤에는 바꿀 수 없다(정정 0 M5).
4. 확인(``confirm``): 시대 C 탐색 절반 — 학습 창 2023-01~2024-12 고원 → 시험 2025-01-02~2026-09-12 한 번. sha 마다 평생 1회.
   폴드가 하나라 σ 가 없다 → 확인 score = 그 폴드의 선택 알파. **유효성 바**(체결 ≥ 연 20 × 시험 연수, 거래당 순기대값(2배) > 0,
   최대 기여 한 종목 제거 알파 > 0)를 넘은 것만 순위에 든다(정정 0 M3). 그중 최대인 하나(동점이면 train.py 줄 수 적은 쪽). 0개면 후보 없음.
5. 잠금(``lockbox``): 고른 하나만 잠금 절반(half 1)에서 A·B 10폴드 + C 1폴드를 같은 절차로. 바 = 11폴드 score > 0 ∧ 거래당 순기대값(2배) > 0
   ∧ 최대 기여 한 종목 제거 11폴드 score > 0. 잠금 절반은 태그 전에 만든 디스크 캐시(``data.LOCKED_CACHE`` — 루프 격리가 가린다)만 읽고
   MANIFEST sha 와 대조한다(DB 재구축 없음 — 정정 0 M4). 평생 1회(sha 무관). 실패하면 다음 순위로 넘어가지 않는다 → chosen = null.
6. ``frozen_params`` = 고른 후보의 확인 행 params = **탐색 절반**의 시대 C 학습 창(2023-01~2024-12) 고원 값(정정 판정 18). 잠금 절반에서
   다시 고른 C 폴드 파라미터는 LOCKBOX 행에 기록만 하고 전진 팔에 쓰지 않는다.

보류 구간 원장(CONFIRM.jsonl·LOCKBOX.jsonl)은 작업 트리 + ~/.cache 사본 두 곳에 덧붙인다. **열기 전에 기록한다**: 사전 검사 뒤
``status: "opened"`` 행을 먼저 쓰고 계산하며, 결과는 두 번째 행(``status: "result"``)이다. 계산 중 죽어도 그 시도는 소모된 것이다.
거부 검사는 두 사본의 합집합으로 한다(한쪽을 지워도 다시 열리지 않는다). 결과는 원장에 쓴 뒤에만 출력한다.

섬(정정 0-2): 원장은 모든 섬(combined-loop-1~4)이 함께 쓰는 사본 디렉터리의 RUNS 다. 풀 = 모든 섬의 keep ∪ 전역 score 상위 10(뜻은 그대로),
섬 이름은 SCREEN·SELECTION 에 남긴다.

사람 열람 바(v4 b8): ``--screen`` 은 풀·거르개만 돌리고(보류 구간을 열지 않는다) 통과한 train.py 를 RUNS 의 commit 에서 scratch 로 꺼내
경로를 보여 준다. 사람이 그 파일들을 읽고(e104063 구조 재제출 여부 포함 — program·PREREG) ``--confirm --reviewed <sha16>,<sha16>…`` 로
통과 목록 **전부**를 정확히 적어야 확인·잠금이 열린다. ``confirm``·``lockbox`` 함수도 마지막 거르개 통과·reviewed 집합을 스스로 검사한다
(import 로 우회 불가). 거르개 뒤 원장이 바뀌었으면 거부한다(--screen 다시). 실행할 train.py 는 내용 sha 를 확인한 읽기 전용 사본으로 고정하고,
로드한 전략의 ``sha`` 를 계산 앞뒤로 대조한다(TOCTOU).

사용(작업 트리 루트, 루프가 멈춘 뒤):
  uv run --with numba python research/combined/select.py --screen
  uv run --with numba python research/combined/select.py --confirm --reviewed <sha16>[,<sha16>…] [--exclude <sha16>:<사유>[,…]]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
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
N_NULL, NULL_Q, MIN_NULL_FINITE = 300, 0.95, 270                           # 솎기 널(바)
N_NULL_PARAMS = 100                                                        # 무작위 파라미터 널(기록만)
MIN_TRADES_PER_YEAR = 20
EXCLUDE_REASONS = {"e104063": "은퇴한 e104063 구조(시장 대비 수축 + 실적 신선도) 또는 그 파라미터 변형 재제출",
                   "neutral_input": "중립 입력(0·False)을 감지하거나 중립 입력에서 통과(pass-through)하지 않는 부품",
                   "date_code_target": "날짜·종목(배열 위치 포함)을 겨냥하는 규칙",
                   "news_param": "뉴스 부품에 숫자 파라미터(params 의 창 길이·개수 문턱 등)",
                   "cross_stock_uni_off": "uni 가 꺼진 종목의 교차 종목 정보 사용(동종·상관 종목 고르기 — 정정 0-2 판정 28)"}
N_SWING_TRIALS = 280                                                       # 같은 데이터의 swing 루프 시행(DSR N 에 더한다)
SEARCH = {e: ERAS_C[e] for e in ("A", "B")}                                # 선택 구간(루프와 같은 10폴드)
CONF = {"C": ERAS_C["C"]}                                                  # 확인 구간 1폴드


# ---------------------------------------------------------------- 원장(두 사본)

def _rows_both(path: str) -> list:
    """작업 트리 + 사본 두 곳의 행 합집합 — 거부 검사용(``P._read_jsonl`` 은 긴 쪽 하나만 본다)."""
    out = []
    for p in (path, os.path.join(P.SHADOW, os.path.basename(path))):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out += [json.loads(x) for x in f if x.strip()]
    return out


def _opened(path: str, sha: str) -> None:
    """보류 구간을 열기 직전의 증인 행 — 계산 전에 두 사본에 쓴다."""
    P._append(path, {"train_sha": sha, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "opened", "evaluator": P.PREREG_TAG})


def _results(path: str) -> dict:
    return {r["train_sha"]: r for r in _rows_both(path) if r.get("status") == "result"}


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


@contextmanager
def _pinned(train_path: str, sha: str):
    """내용 sha 를 확인한 train.py 를 새 임시 디렉터리에 읽기 전용으로 고정 → 로드한 전략 sha 대조(앞) → 경로 → 다시 대조(뒤)."""
    b = open(train_path, "rb").read()
    if _sha_bytes(b) != sha:
        raise SystemExit(f"train.py sha {_sha_bytes(b)} ≠ {sha} — 사람이 읽은 그 파일만 연다")
    os.makedirs(SCRATCH, exist_ok=True)
    d = tempfile.mkdtemp(prefix="pin_", dir=SCRATCH)
    p = os.path.join(d, "train.py")
    try:
        with open(p, "wb") as f:
            f.write(b)
        os.chmod(p, 0o400)
        if P.load_strategy(p)["sha"] != sha:
            raise SystemExit(f"로드한 전략 sha ≠ {sha} — 거부")
        yield p
        if P.load_strategy(p)["sha"] != sha:
            raise SystemExit(f"계산 중 train.py 가 바뀌었다(전략 sha ≠ {sha}) — 결과 무효")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 적재·git

def load_search(era: str) -> dict:
    from research.combined.data import load
    return load(era, 0)


def load_locked(era: str) -> dict:
    """잠금 절반(half 1) — 태그 전에 탐색 절반과 같은 빌드로 만든 디스크 캐시만 읽는다(DB 재구축 없음 — 시점 불일치로 거부될 위험 제거, M4).
    읽은 바이트의 sha256 이 MANIFEST ``locked/era_X_s1.pkl`` 과 같아야 한다."""
    from research.combined.data import load
    man = json.load(open(P.MANIFEST, encoding="utf-8"))
    key = f"locked/era_{era}_s1.pkl"
    if key not in man:
        raise SystemExit(f"MANIFEST 에 {key} 가 없다 — 잠금 캐시는 태그 전에 만들어 sha 를 적어야 한다")
    return load(era, 1, sha=man[key])


def extract_train(commit: str, sha: str) -> str:
    """``git show <commit>:research/combined/train.py`` → scratch/<sha>/train.py. 내용 sha 가 원장과 다르면 거부."""
    p = subprocess.run(["git", "-C", ROOT, "show", f"{commit}:{TRAIN_REL}"], capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"git show {commit}:{TRAIN_REL} 실패: {p.stderr.decode(errors='replace').strip()}")
    if _sha_bytes(p.stdout) != sha:
        raise SystemExit(f"commit {commit} 의 train.py sha {_sha_bytes(p.stdout)} ≠ 원장 {sha}")
    d = os.path.join(SCRATCH, sha)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "train.py")
    with open(out, "wb") as f:
        f.write(p.stdout)
    return out


# ---------------------------------------------------------------- 1. 풀

def _eval_rows(runs: list) -> list:
    """평가 행(accept·crash·참고 제외), sha 당 첫 행, eligible ∧ ¬void ∧ 유한 score."""
    rows, seen = [], set()
    for r in runs:
        sha = r.get("train_sha")
        if r.get("accept") or r.get("crash") or r.get("reference") or "folds" not in r or sha in seen:
            continue
        seen.add(sha)
        if r.get("eligible") is True and not r.get("void") and isinstance(r.get("score"), (int, float)) and math.isfinite(r["score"]):
            rows.append(r)
    return rows


def _reject_reason(r: dict) -> "str | None":
    if not r.get("commit"):
        return "commit 없음 — train.py 를 되살릴 수 없다"
    if r.get("dirty") is not False:
        return f"dirty={r.get('dirty')} — 평가한 train.py 가 commit 과 다를 수 있다"
    return None


def pool_rejects(runs: list) -> list:
    return [{"train_sha": r["train_sha"], "score": r["score"], "reason": why} for r in _eval_rows(runs) if (why := _reject_reason(r))]


def pool(runs: list) -> list:
    """자격 행 중 commit 있고 ¬dirty 인 것 → keep(beats_best) 전부 ∪ score 상위 10 → score 순 최대 10."""
    rows = sorted((r for r in _eval_rows(runs) if _reject_reason(r) is None), key=lambda r: -r["score"])   # 안정 정렬
    keep_sha = {r["train_sha"] for r in runs if r.get("accept")}
    keep = [r for r in rows if r.get("beats_best") is True or r["train_sha"] in keep_sha]
    union = {r["train_sha"]: r for r in [*keep, *rows[:POOL_TOP]]}
    return sorted(union.values(), key=lambda r: -r["score"])[:POOL_MAX]


# ---------------------------------------------------------------- 2. 거르개

def concentration(D_by_era: dict, res: dict, train_path: str, eras: dict) -> dict:
    return P.concentration(D_by_era, res, train_path, eras=eras)


def factor_record(D_by_era: dict, res: dict, eras: dict) -> dict:
    """팩터(KOSPI·KOSDAQ·MOM·SMB·LV) 회귀 절편과 NW t — 기록만(바 아님)."""
    try:
        F = pd.concat([P.factor_months(D_by_era[e]) for e in eras])
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


def screen(rec: dict, D_by_era: dict, train_path: str, workers: int = 1, n_null: int = N_NULL, n_null_params: int = N_NULL_PARAMS) -> dict:
    """탐색 절반·선택 구간(A·B) 거르개. 앞 바가 떨어지면 널은 돌리지 않는다(널 바 = False 로 기록)."""
    sha = rec["train_sha"]
    reasons = []
    with _pinned(train_path, sha) as tp:
        res = P.walk_forward_c(D_by_era, workers, tp, eras=SEARCH)
        s = P.summarize_c(res)
        conc = concentration(D_by_era, res, tp, SEARCH)
        bars = {"repro_ledger": bool(math.isclose(s["score"], rec["score"], rel_tol=1e-9, abs_tol=1e-12)),
                "score_pos": bool(s["score"] > 0 and not s["void"]),
                "ev_net_2x_pos": bool(s["ev_net_2x"] > 0),
                "top1pct_removed_pos": bool(conc["top1pct"] > 0),
                "top_stock_removed_pos": bool(conc["top_stock"] > 0),
                "trades_per_year_ge_20": bool(s["oos_trades_per_year"] >= MIN_TRADES_PER_YEAR),
                "beats_null_p95": False}
        nulls, nvoid, p95 = [], 0, float("nan")
        pn, pvoid = [], 0
        if all(v for k, v in bars.items() if k != "beats_null_p95"):
            min_finite = math.ceil(MIN_NULL_FINITE * n_null / N_NULL)
            for r in range(n_null):                                        # 바: 솎기 널(같은 파라미터·cfg, 월별 통과율 그대로 무작위 후보)
                ns = P.summarize_c(P.walk_forward_c(D_by_era, 1, tp, fixed=res["folds"], null=("thin", r + 1), eras=SEARCH))
                nvoid += bool(ns["void"])
                if np.isfinite(ns["raw_score"]):
                    nulls.append(float(ns["raw_score"]))
            if len(nulls) < min_finite:
                reasons.append(f"null 표본 부족({len(nulls)}/{n_null} 유한, 최소 {min_finite})")
            else:
                p95 = float(np.quantile(nulls, NULL_Q))
                bars["beats_null_p95"] = bool(s["score"] > p95)
            for r in range(n_null_params):                                 # 기록만: 같은 메뉴 무작위 파라미터
                ns = P.summarize_c(P.walk_forward_c(D_by_era, 1, tp, fixed=res["folds"], null=("params", r + 1), eras=SEARCH))
                pvoid += bool(ns["void"])
                if np.isfinite(ns["raw_score"]):
                    pn.append(float(ns["raw_score"]))
        else:
            reasons.append("앞 바 실패 — 널 생략")
        reasons += [f"바 실패: {k}" for k, v in bars.items() if not v and not (k == "beats_null_p95" and reasons)]
        out = {"train_sha": sha, "commit": rec.get("commit"), "pass": all(bars.values()), "bars": bars, "reasons": reasons,
               "score": s["score"], "ledger_score": rec["score"], "void": s["void"], "ev_net_2x": s["ev_net_2x"],
               "payoff_ratio": s["payoff_ratio"], "oos_trades": s["oos_trades"], "trades_per_year": s["oos_trades_per_year"],
               "folds_pos": s["folds_pos"], "top1pct_removed": conc["top1pct"], "top_stock_removed": conc["top_stock"],
               "top10d_removed": conc.get("top10d"), "null_kind": "thin", "null_n": len(nulls), "null_void": nvoid, "null_p95": p95,
               "null_median": float(np.median(nulls)) if nulls else float("nan"),
               "params_null": {"n": len(pn), "void": pvoid, "p95": float(np.quantile(pn, NULL_Q)) if pn else float("nan"),
                               "median": float(np.median(pn)) if pn else float("nan"), "bar": False},
               "factor_bar": "GUARDRAILS 팩터 t>2 바 미적용 — 기록만, 판정은 전진(PREREG 정정 0 M2)",
               "factor": factor_record(D_by_era, res, SEARCH), "dsr": dsr_record(res, P.trial_state()["n"] + N_SWING_TRIALS),
               "train_lines": _lines(tp), "folds": [{k: f[k] for k in ("era", "test", "params", "alpha", "n")} for f in res["folds"]]}
    return out


# ---------------------------------------------------------------- 보류 구간 공통 검사

def _current_screen() -> dict:
    """마지막 거르개 기록(review 행 제외) — 지금 원장의 풀·시행 수와 같아야 한다(다르면 --screen 다시)."""
    scr = [r for r in P._read_jsonl(SCREEN) if r.get("kind") != "review"]
    if not scr:
        raise SystemExit("거르개 기록이 없다 — 먼저 select.py --screen")
    scr = scr[-1]
    if [r["train_sha"] for r in pool(P._read_jsonl(P.RUNS))] != scr["pool"] or P.trial_state()["n"] != scr["n_trials"]:
        raise SystemExit("거르개 뒤 원장이 바뀌었다 — select.py --screen 을 다시 돌리고 다시 열람")
    return scr


def _current_review() -> "dict | None":
    """마지막 거르개 기록 뒤의 마지막 사람 열람(review) 행 — reviewed 목록과 제외(sha → 열거 사유)."""
    rows = P._read_jsonl(SCREEN)
    last_scr = max((i for i, r in enumerate(rows) if r.get("kind") != "review"), default=-1)
    rv = [r for r in rows[last_scr + 1:] if r.get("kind") == "review"]
    return rv[-1] if rv else None


def parse_exclude(items: "list[str]") -> dict:
    """``sha:사유`` 목록 → {sha: 사유}. 사유는 ``EXCLUDE_REASONS`` 열거값만, sha 는 한 번만."""
    out = {}
    for it in items:
        sha, sep, why = it.partition(":")
        sha, why = sha.strip(), why.strip()
        if not sep or why not in EXCLUDE_REASONS:
            raise SystemExit(f"--exclude {it!r}: 'sha:사유', 사유는 {sorted(EXCLUDE_REASONS)} 중 하나")
        if sha in out:
            raise SystemExit(f"--exclude: {sha} 가 두 번")
        out[sha] = why
    return out


def record_review(scr: dict, reviewed: "list[str]", excluded: dict) -> dict:
    """사람 열람 결과를 확인 구간을 열기 **전에** SCREEN 에 review 행으로 남긴다. 확인이 한 번이라도 열렸으면 마지막 review 와 같아야 한다
    (결과를 본 뒤 제외를 바꾸지 못한다)."""
    bad = sorted(set(excluded) - set(scr["survivors"]))
    if bad:
        raise SystemExit(f"--exclude {bad} 는 거르개 통과 목록에 없다")
    prev = _current_review()
    same = prev is not None and sorted(prev["reviewed"]) == sorted(reviewed) and prev["excluded"] == excluded
    if _rows_both(CONFIRM):
        if not same:
            raise SystemExit("확인 구간이 이미 열렸다 — 사람 열람(reviewed·exclude)을 바꿀 수 없다(SCREEN 의 마지막 review 그대로 다시)")
        return prev
    if same:
        return prev
    rec = {"kind": "review", "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "screen_at": scr.get("at"), "reviewed": sorted(reviewed),
           "excluded": dict(excluded), "excluded_text": {k: EXCLUDE_REASONS[v] for k, v in excluded.items()}, "evaluator": P.PREREG_TAG}
    P._append(SCREEN, rec)
    return rec


def _gate(sha: str, reviewed: set) -> dict:
    scr = _current_screen()
    if sha not in scr["survivors"]:
        raise SystemExit(f"{sha} 는 마지막 거르개 통과 목록에 없다 — 보류 구간을 열지 않는다")
    if sha not in reviewed:
        raise SystemExit(f"{sha} 는 --reviewed 에 없다 — 사람이 읽은 train.py 만 연다(v4 b8)")
    rv = _current_review()
    if rv is None or sha not in rv["reviewed"]:
        raise SystemExit(f"{sha}: 거르개 뒤 사람 열람(review) 기록이 없다 — 확인 구간을 열기 전에 record_review")
    if sha in rv["excluded"]:
        raise SystemExit(f"{sha} 는 사람 열람에서 제외됐다({rv['excluded'][sha]}) — 보류 구간을 열지 않는다")
    return scr


def top_stock_removed(D_by_era: dict, runs: list, train_path: str) -> "tuple[str | None, list]":
    """runs = [(res, eras), …] 의 모든 시험 거래에서 순수익 합 최대 종목 하나를 빼고, 같은 폴드 파라미터로 재실행한 폴드 alpha 목록(runs 순).
    재실행 신호가 원래와 다르면(상태 의존) alpha 는 nan(바 실패)."""
    trs = [t for res, _ in runs for t in res["trades"]]
    nf = sum(len(res["folds"]) for res, _ in runs)
    if not trs:
        return None, [float("nan")] * nf
    tr = pd.concat(trs)
    code = str(tr.groupby("code")["ret"].sum().idxmax())
    out = []
    try:
        for res, eras in runs:
            r = P.walk_forward_c({e: D_by_era[e] for e in eras}, 1, train_path, fixed=res["folds"], exclude=[code], eras=eras)
            P._same_signals(res, r, "최대 종목(선택기)")
            out += [float(f["alpha"]) for f in r["folds"]]
    except P.StateLeak:
        return code, [float("nan")] * nf
    return code, out


def _ev(res: dict) -> "tuple[float, float, int]":
    tr = pd.concat(res["trades"]) if res["trades"] else pd.DataFrame(columns=["ret"])
    win, loss = tr.loc[tr["ret"] > 0, "ret"], tr.loc[tr["ret"] <= 0, "ret"]
    pr = float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan")
    return (float(tr["ret"].mean()) if len(tr) else float("nan")), pr, int(len(tr))


# ---------------------------------------------------------------- 3. 확인(sha 당 1회)

def confirm(sha: str, train_path: str, reviewed: set, workers: int = 1, commit: "str | None" = None) -> dict:
    """시대 C 탐색 절반 — 학습 창 고원 → 2025-01-02~2026-09-12 시험 한 번. 열기 전 opened 행, 끝나면 result 행(CONFIRM.jsonl + 사본).
    유효성 바(순위 진입 조건, M3): 체결 ≥ 연 20 × 시험 연수 ∧ 거래당 순기대값(2배) > 0 ∧ 최대 기여 한 종목 제거 알파 > 0 → ``valid``."""
    if _rows_both(LOCKBOX):
        raise SystemExit("잠금 절반이 이미 열렸다 — 선택은 끝났다, 확인 구간을 더 열지 않는다")
    if any(r.get("train_sha") == sha for r in _rows_both(CONFIRM)):
        raise SystemExit(f"확인 구간은 {sha} 에 이미 열렸다(CONFIRM.jsonl) — 두 번 열지 않는다")
    _gate(sha, reviewed)
    with _pinned(train_path, sha) as tp:
        D = {"C": load_search("C")}
        _opened(CONFIRM, sha)
        res = P.walk_forward_c(D, workers, tp, eras=CONF)
        ts_code, ts_alpha = top_stock_removed(D, [(res, CONF)], tp)
        lines = _lines(tp)
    f = res["folds"][0]
    ev, pr, n = _ev(res)
    alpha = float(f["alpha"])
    years = float(f.get("years") or 0)
    bars = {"trades_ge_20_per_year": bool(n >= MIN_TRADES_PER_YEAR * years), "ev_net_2x_pos": bool(ev > 0),
            "top_stock_removed_pos": bool(ts_alpha[0] > 0)}
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "status": "result", "commit": commit, "era": "C", "half": 0,
           "test": [str(x) for x in f["test"]], "params": f["params"], "alpha": alpha,
           "score": alpha if math.isfinite(alpha) else float("nan"), "excess": f.get("excess"), "n": n, "years": years,
           "trades_per_year": n / max(years, 1e-9), "ev_net_2x": ev, "payoff_ratio": pr,
           "top_stock": ts_code, "top_stock_removed_alpha": ts_alpha[0], "bars": bars, "valid": all(bars.values()),
           "trade_key": f.get("trade_key"), "train_lines": lines, "evaluator": P.PREREG_TAG}
    P._append(CONFIRM, row)                                                # 쓰고 나서 돌려준다(출력은 그 뒤)
    return row


# ---------------------------------------------------------------- 4. 잠금(평생 1회)

def lockbox(sha: str, train_path: str, reviewed: set, workers: int = 1, commit: "str | None" = None) -> dict:
    """잠금 절반 — A·B 10폴드 + C 1폴드(같은 절차). 바 = 11폴드 score > 0 ∧ 거래당 순기대값(2배) > 0 ∧ 최대 기여 한 종목 제거 11폴드 score > 0.
    열기 전 opened 행(LOCKBOX.jsonl + 사본)."""
    if _rows_both(LOCKBOX):
        raise SystemExit("잠금 절반은 이미 열렸다(LOCKBOX.jsonl) — 최대 1회, 다음 순위로 넘어가지 않는다")
    if sha not in _results(CONFIRM):
        raise SystemExit(f"{sha} 의 확인 결과가 없다 — 확인 구간 순위로 고른 것만 잠금을 연다")
    _gate(sha, reviewed)
    with _pinned(train_path, sha) as tp:
        D = {e: load_locked(e) for e in (*SEARCH, *CONF)}                 # DB 불일치 거부는 opened 전(1회를 태우지 않는다)
        _opened(LOCKBOX, sha)
        ab = P.walk_forward_c({e: D[e] for e in SEARCH}, workers, tp, eras=SEARCH)
        c = P.walk_forward_c({e: D[e] for e in CONF}, workers, tp, eras=CONF)
        ts_code, ts_alpha = top_stock_removed(D, [(ab, SEARCH), (c, CONF)], tp)
    s_ab = P.summarize_c(ab)                                               # A·B 만(시대 C 에는 summarize 를 쓰지 않는다)
    both = {"folds": ab["folds"] + c["folds"], "trades": ab["trades"] + c["trades"]}
    score = P.score_of([f["alpha"] for f in both["folds"]])
    ts_score = P.score_of(ts_alpha)
    ev, pr, n = _ev(both)
    bars = {"score_pos": bool(score > 0), "ev_net_2x_pos": bool(ev > 0), "top_stock_removed_pos": bool(ts_score > 0)}
    row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": sha, "status": "result", "commit": commit, "half": 1, "score": score,
           "score_ab": P.score_of([f["alpha"] for f in ab["folds"]]), "alpha_c": c["folds"][0]["alpha"], "void_ab": s_ab["void"],
           "ev_net_2x": ev, "ev_net_2x_ab": _ev(ab)[0], "ev_net_2x_c": _ev(c)[0], "payoff_ratio": pr, "n": n,
           "top_stock": ts_code, "top_stock_removed_score": ts_score,
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
    pl, rej = pool(runs), pool_rejects(runs)
    D = {e: load_search(e) for e in SEARCH}
    results, review = {}, []
    for r in pl:
        sha = r["train_sha"]
        path = extract_train(r["commit"], sha)
        results[sha] = screen(r, D, path, workers=workers)
        P.log(f"거르개 {sha} ({r['commit']}): pass {results[sha]['pass']}")
        if results[sha]["pass"]:
            review.append({"train_sha": sha, "commit": r["commit"], "island": r.get("island", "main"), "path": path, "lines": _lines(path)})
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_trials": st["n"], "pool": [r["train_sha"] for r in pl], "pool_rejects": rej,
           "pool_rows": [{k: r.get(k) for k in ("train_sha", "commit", "island", "score", "beats_best", "ev_net_2x", "oos_trades_per_year",
                                                "conc")}
                         for r in pl],
           "results": results, "survivors": [x["train_sha"] for x in review], "review": review, "evaluator": P.PREREG_TAG}
    P._append(SCREEN, rec)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    print("\n사람 열람: 아래 파일을 읽고 통과 목록 전부를 --reviewed 에 적는다. 열거 사유에 걸리면 --exclude sha:사유"
          f"({', '.join(f'{k}={v}' for k, v in EXCLUDE_REASONS.items())}).")
    for x in review:
        print(f"  {x['train_sha']}  {x['commit']}  {x['lines']}줄  {x['path']}")
    print(f"  → select.py --confirm --reviewed {','.join(rec['survivors'])}")
    for x in review:
        print(f"  섬 {x['island']}: {x['train_sha']}")
    return rec


def run_confirm(reviewed: "list[str]", workers: int, exclude: "dict | None" = None) -> dict:
    """사람 열람 기록(제외 포함) → 확인(제외 뒤 통과 전부, sha 당 1회) → 유효성 바를 넘은 것 중 확인 score 최대(동점이면 짧은 train.py)
    → 그 하나만 잠금 1회 → SELECTION.json."""
    exclude = dict(exclude or {})
    P.integrity(check_locked=True)
    if _rows_both(LOCKBOX):
        raise SystemExit("잠금 절반이 이미 열렸다 — 선택은 끝났다(SELECTION.json)")
    _need_loop_done()
    scr = _current_screen()
    survivors = scr["survivors"]
    rv = set(reviewed)
    if sorted(rv) != sorted(survivors) or len(reviewed) != len(rv):
        raise SystemExit(f"--reviewed {sorted(reviewed)} ≠ 거르개 통과 {sorted(survivors)} — 통과한 train.py 전부를 사람이 읽고 정확히 적는다(v4 b8)")
    rec = record_review(scr, reviewed, exclude)                           # 확인 구간을 열기 전에 기록
    opened = {r["train_sha"] for r in _rows_both(CONFIRM)}
    done = _results(CONFIRM)
    stuck = sorted(opened - set(done))
    if stuck:
        raise SystemExit(f"확인 중단 — 사람 판단: {stuck} 은 열렸지만 결과가 없다(계산 중 중단). 그 sha 는 소모됐다 — 다시 돌리지 않는다")
    sel = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "chosen": None, "frozen_params": None, "picked": None,
           "reasons": [f"풀 제외(dirty·commit 없음): {x['train_sha']} — {x['reason']}" for x in scr.get("pool_rejects", [])],
           "reviewed": sorted(reviewed), "excluded": rec["excluded"], "pool": scr["pool_rows"], "screen": scr["results"], "confirm": [],
           "lockbox": None, "evaluator": P.PREREG_TAG}
    sel["reasons"] += [f"사람 열람 제외: {k} — {EXCLUDE_REASONS[v]}" for k, v in rec["excluded"].items()]
    todo = [x for x in survivors if x not in rec["excluded"]]
    if not todo:
        sel["reasons"].append(f"거르개 통과 없음 또는 전부 제외(풀 {len(scr['pool'])}, 통과 {len(survivors)}) — 확인·잠금을 열지 않았다. 판정: 후보 없음")
        _write_selection(sel)
        return sel
    by = {x["train_sha"]: x for x in scr["review"]}
    paths = {}
    for sha in todo:                                                       # 이어 하기: 결과가 있는 sha 는 기록을 읽기만 한다
        paths[sha] = extract_train(by[sha]["commit"], sha)
        sel["confirm"].append(done[sha] if sha in done else confirm(sha, paths[sha], rv, workers, commit=by[sha]["commit"]))
    valid = [r for r in sel["confirm"] if r.get("valid") is True]
    if not valid:
        sel["reasons"].append(f"확인 유효성 바(체결·거래당 순기대값 2배·최대 종목 제거 알파) 통과 없음({len(sel['confirm'])}개 확인) — "
                              "잠금을 열지 않았다. 판정: 후보 없음")
        _write_selection(sel)
        return sel
    order = sorted(valid, key=lambda r: (-(r["score"] if math.isfinite(r["score"]) else -math.inf), r["train_lines"]))
    top = order[0]
    scr_top = scr["results"].get(top["train_sha"], {})
    sel["picked"] = {"train_sha": top["train_sha"], "commit": top["commit"], "island": by[top["train_sha"]].get("island"),
                     "confirm_score": top["score"], "train_lines": top["train_lines"],
                     "params_null_p95": (scr_top.get("params_null") or {}).get("p95"), "thin_null_p95": scr_top.get("null_p95"),
                     "factor_t_nw": (scr_top.get("factor") or {}).get("t_nw")}     # 기록만(바 아님)
    sel["reasons"].append(f"확인 score 최대 {top['train_sha']}(동점이면 train.py 줄 수 적은 쪽) — 유효 {len(order)}개 / 확인 {len(sel['confirm'])}개 중")
    lk = lockbox(top["train_sha"], paths[top["train_sha"]], rv, workers, commit=top["commit"])
    sel["lockbox"] = lk
    if lk["pass"]:
        sel["chosen"] = {"train_sha": top["train_sha"], "commit": top["commit"]}
        sel["frozen_params"] = top["params"]
        sel["reasons"].append("잠금 절반 바 통과(11폴드 score > 0 ∧ 거래당 순기대값 2배 > 0 ∧ 최대 종목 제거 score > 0) — "
                              "frozen_params = 탐색 절반 시대 C 학습 창(2023~2024) 고원 값")
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
    ap.add_argument("--exclude", default="", help=f"--confirm: 사람 열람 제외 sha:사유(사유 {sorted(EXCLUDE_REASONS)}), 쉼표로")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    P._G["t0"] = time.time()
    if a.screen:
        run_screen(a.workers)
    else:
        if a.reviewed is None:
            raise SystemExit("--confirm 은 --reviewed 가 필요하다(거르개 통과 전부)")
        ex = parse_exclude([s for s in a.exclude.split(",") if s.strip()])
        run_confirm([s.strip() for s in a.reviewed.split(",") if s.strip()], a.workers, exclude=ex)


if __name__ == "__main__":
    main()
