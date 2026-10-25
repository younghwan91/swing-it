"""세 레포 종합 autoresearch 고정 평가기 — 진입점이자 심판. 에이전트는 이 파일을 고치지 않는다(program.md · PREREG.md).

swing v4 평가기(``research/autoresearch/prepare.py``, 태그 autoresearch-prereg-4)를 골격으로 쓴다. v4 의 순수 함수(전역을 읽지 않는 것)는
import 해서 그대로 쓰고(sample_params·plateau_pick·clamp_cfg·exit_line·fold_stats·summarize·factor_months·factor_t·half_of·sha_file),
v4 의 ERAS·_G·VIEW_KEYS·TRAIN·CACHE 를 읽는 것은 복사해 고쳤다. 설계: docs/superpowers/specs/2026-10-25-combined-autoresearch-design.md.

v4 와 다른 점(전부):
1. 데이터: 시대 A(2011~2015 시험 5폴드)·B(2020~2024 시험 5폴드) = 10폴드, ``research.combined.data`` 캐시(갭·뉴스 키 추가). 2016 폴드는 없다.
   확인 시대 C 는 루프가 보지 못한다(격리 프로세스에 바인드하지 않는다 — select.py 몫). 시대를 도는 함수는 ``eras=`` 를 받는다(기본 A·B).
2. V = v4 키 + 갭(gap·tickp·gap_event) + 뉴스(n_all·n_feat·n_move·n_excl·news_known). 미래정보 절단 검사도 새 키를 자른다.
3. 진입은 **신호일 종가**(``entry="close"``, leader_C 방식) — 평가기가 강제한다. exit_dist(진입일 청산선 거리)도 종가 대비.
4. ``simulate_c`` 에 ``mask``(cand 에서 끌 칸 — 상위 1% 거래 제거용, 신호 칸만 끄는 부드러운 제거).
5. 폴드마다 ``trade_key``(시험 체결 (종목 위치, 진입일 위치) 정렬 목록 sha1 앞 12자). 거래 표에 ci·ti 열, fold = 전체 폴드 번호(v4 는 시대 안 번호).
6. 쏠림 진단 셋(``concentration``): 상위 1% 거래 신호 칸 끄기 · 최대 기여 한 종목 빼기 · 일별 선택 알파 상위 10일 0. v4 의 top5 제거를 대신한다.
   상위 1% 거래 수 = ceil(1% × 시험 체결 수)(최소 1).
7. 부품 제거(``ablation``): 레포 키를 중립값으로 바꾸고 폴드 파라미터 고정 재실행 → 체결 대칭차 ≥ 5% ∧ 작동 폴드 점수 기여 > 0
   ∧ 제거 실행 작동 폴드 체결 ≥ 원래의 50% ∧ 폴드마다 ≥ 5(정정 0 M1 — 중립 입력에서 전략이 멈추면 자격 아님).
   작동 폴드 = scalp 10폴드 전부, daytrade 시험 연도 2023·2024. 대칭차·점수 모두 작동 폴드에서 잰다. ``eligible`` = 둘 다 ok.
   자격은 keep 과 무관하다(후보 고르기 select.py 의 거르개).
8. keep(``beats_c``): 폴드 조건이 "10 중 6 개선" → "체결이 바뀐 폴드(≥ 1) 중 alpha 개선 과반". top5 조건 → 쏠림 셋 각각 BEST 의 같은 값 − 0.01 이상.
   폴드 정렬(era·test)이 BEST 와 다르면 (False, "fold mismatch"). BEST = 마지막 keep: 원장의 마지막 beats_best 평가 행이 마지막 accept 보다
   뒤면 그 행, 아니면 BEST_HISTORY.jsonl(작업 트리 + 사본 중 긴 쪽) 마지막 행. 둘 다 없을 때만 "첫 유효 실행"(BEST.json 을 지워도 바가
   되돌아가지 않는다).
9. 시행 상한(``trial_state``): 고유 train sha 150, 또는 첫 keep(= beats_best True 평가 행 — --accept 를 건너뛰어도 센다) 뒤 고유 sha 30개 연속
   keep 없음. 상한이면 ``evaluate`` 는 계산·무결성 검사 전에 거부한다. 원장은 작업 트리 + ~/.cache 사본 중 긴 쪽(v4 와 같음).
   crash 도 원장에 crash 행으로 남겨 시행으로 센다(v4 는 세지 않았다). crash 때는 격리 실행의 stdout 을 버리고 stderr 끝 20줄만 보인다.
   성공 때 stderr 는 평가기 로그 줄([ …s])만, 학습 점 실패는 예외 종류만 보인다(train 이 스스로 잰 구간 성적을 흘리는 통로 차단).
10. 출력 줄: score·void·beats_best·beats_detail·lookahead_check·folds_pos·oos_trades·payoff_ratio·ev_net_2x·ev_net_1x·쏠림 셋·
    news_folds_score·ablation 둘·eligible·trials·stop. ev_net_1x 는 같은 폴드 파라미터로 비용 1배 재실행.
11. 미래정보 검사(``causality_c``)는 절단 하나·전체 신호 하나마다 **새 프로세스**(spawn, maxtasksperchild=1)에서 디스크 캐시(또는 넘겨받은
    배열)로 돈다 — 평가 프로세스에 숨긴 상태(모듈 속성 등)가 절단 계산으로 넘어가지 못한다. 그리고 새 프로세스의 전체 신호가 평가에 쓴 신호와
    해시로 같아야 한다(호출 횟수·숨긴 상태로 신호를 바꾸는 train 은 void). 쏠림·비용 1배 재실행 신호도 원래 실행과 같아야 한다.
    검사 파라미터 = 폴드에서 고른 값 + 학습 점 표본(폴드마다 12개 — 미래로 고원 선택을 조종하는 train 차단; v4 는 고른 값만).
    표본 시드 = sha256(train sha + 원장 사본 디렉터리의 비밀 소금)(격리가 가리고 stdin 으로만 넘긴다 — train 은 어느 점인지 모른다).
    표본 학습 점은 절단 검사에 더해, fork 작업자가 고원 선택에 실제로 쓴 신호 지문이 새 프로세스 신호 지문과 같아야 한다.
    signals 의 예외도 비교한다(같은 x 면 어디서나 같이 실패해야 한다). 학습 점의 cfg 예외는 평가 전체를 crash 로 끝낸다(시행으로 셈).
    cfg(x) 는 signals 를 부른 적 없는 새 프로세스에서만 계산해(``fresh_cfgs``) 학습 점·채점·모든 재실행에 쓴다. 평가 프로세스의 cfg 가
    그 값과 다르면 void(signals 가 숨긴 전체 기간 V 를 cfg 가 읽는 길 차단; v4 는 cfg 를 검사하지 않았다).
    signals 결과는 받는 즉시 평가기 소유 사본으로 만든다(``call_signals`` — 정확히 dict, 값은 정확히 수 dtype np.ndarray, 아니면 거부:
    학습 점은 오류, 시험 폴드는 crash). 지문·시뮬레이션은 사본만 쓰고, 사본을 뜬 뒤 그 시뮬레이션이 끝날 때까지 train 코드(cfg 포함)를
    부르지 않는다 — train 이 쥔 배열을 cfg 등에서 제자리로 고치는 길(리뷰 rr6) 차단. x 도 사본으로 넘긴다.
    실패 메시지는 "fail: " 로 시작한다. 계산 본체는 ``run_all``(격리 안 ``--inner`` 가 부른다 — 테스트 가능).
    워크포워드 학습 점 풀은 fork 그대로다: 부모는 그 시대의 전체 V 만 보므로 거기서 숨긴 상태는 train 이 이미 보는 것 이상을 주지 못하고,
    점수 실행(A → B 순)의 시대 A 는 B 를 보기 전에 끝난다. 그 뒤 재실행의 시대 간 오염은 위 신호 해시 비교가 잡는다.
12. 허용 목록(``lint_train``) = v4 + 가격 격자 통로 금지(정정 0-4: ``%``·divmod·mod/fmod/remainder/gcd/lcm/modf/frexp 전부, 가격 O·H·L·C 에
    거는 round/rint/floor/ceil/trunc) + 상태 은닉 금지: 속성 대입·삭제 전부, 모듈 이름(np·pd·math·…)에 뿌리를 둔 첨자 대입·삭제와
    변경 메서드 호출(append·update·setdefault·pop·… — ``np.sort`` 같은 모듈 함수 자체는 허용), 모듈 이름을 속성 접근 외 값으로 쓰기,
    ``print``, 모듈 최상위의 실행문(설명문·import·def/class·리터럴 대입만 — 적재 횟수를 세는 길 차단).
13. 격리: 캐시 디렉터리를 가리고 era_A·era_B 만 다시 보인다. 원장 사본(평가 전에 만들어 늘 가린다)·~/git/daytrade-it(뉴스 아카이브)·
    ~/git/quant-airflow·~/.config·~/.ssh·v4 캐시·v4 보관본·v4 원장 사본·잠금 절반 캐시를 가린다. 시간 초과 30분 = crash.
14. 무결성: 태그 ``PREREG_TAG``(정정 1 뒤 ``combined-prereg-2``), 감시 파일 = 결합 prepare·data·gaps·news·select + v4 감시 파일 전부, MANIFEST 에 캐시 A·B·C sha
    (+ 잠금 절반 A·B·C sha — select.py 만 대조). 평가마다 commit 을 ``refs/combined/<sha>/<commit>`` 으로 붙잡는다.
    ``--build`` 는 data.build_all 에 맡기고 MANIFEST 는 쓰지 않는다(``--manifest`` 따로). ``--final`` 은 없다.
15. 원장 기록: 평가 행에 git commit(앞 12자)·dirty(train.py 가 HEAD 와 다름), 폴드별 era·test·params·alpha·trade_key·n, 쏠림 셋, 부품 제거,
    eligible, 거래당 값을 남긴다(select.py 가 읽는다). BEST.json·BEST_HISTORY 도 사본에 쓴다.
17. 섬(island) 모형(정정 0-2, 사용자 요청 2026-10-25): 루프 에이전트 넷이 각자 워크트리·브랜치 ``combined-loop-1``~``-4`` 에서 돈다.
    섬 = 현재 git 브랜치 이름(정규식 ``^combined-loop-[1-4]$`` 아니면 평가 거부 — 단 기준선 train.py 는 섬 "main" 으로 평가된다:
    태그 전 스모크·참고). 공유(전역, 원장 사본 디렉터리 ``~/.cache/cc-swing/combined_ledger/``): RUNS 행(+ ``island``), 시행 수(모든 섬의
    고유 train sha), 상한 150·멈춤 규칙(첫 keep 뒤 모든 섬 합쳐 연속 30 무 keep)·DSR N. 섬마다: keep 바(BEST_<섬>.json, BEST_HISTORY 행의
    island). 모든 섬은 같은 기준선 BEST(섬 main 의 첫 유효 실행)에서 출발한다. 공유 JSONL 덧붙이기·BEST 읽고 쓰기는 사본 디렉터리
    ``ledger.lock`` 의 fcntl.flock 아래서, 상한 검사는 실행 직전과 결과 행을 덧붙이기 직전 두 번(151 번째가 끼어들지 못한다).
    섬의 기본 작업자 수 = max(1, (cpu − 1) // 4).
18. 메모리(정정 1, 결과 불변): 미래정보 검사 작업자는 x 마다 (cand 비트 묶음, cand 칸 score, 지문)만 돌려주고 부모는 절단 결과를 받는 즉시
    비교·폐기한다(옛 코드는 모든 절단의 전체 배열을 부모에 모아 시대마다 수십 GB). 재실행 시대가 끝나면 ``_G`` 의 D·S·V 를 놓고
    gc.collect()(train 모듈 전역 ↔ 함수 순환 참조), fork 전 gc.collect()·gc.freeze().
16. ``--reference``: leader_C 고정값(θ90·N20·MA100·hard 0.15·슬롯 10·불타기·국면, 종가 진입)을 기준선 train.py 로 두 시대에 한 번 잰다.
    결과는 REFERENCE.json(+ 사본)에만 쓰고 RUNS 행은 쓰지 않는다(시행 아님). 이미 있으면 거부, train.py 가 기준선 sha 가 아니면 거부.
    leader_C 의 종목군 ``univ: top200`` 은 V 에 시총 순위 키가 없어 표현하지 못한다(종목군은 평가기 uni 그대로) — 정확한 leader_C 가 아니다.

사용(작업 트리 루트):
  uv run python research/combined/prepare.py > run.log 2>&1          # 실험 한 번
  uv run python research/combined/prepare.py --accept                 # 직전 실행이 beats_best 면 BEST 갱신(keep)
  uv run python research/combined/prepare.py --reference              # leader_C 고정값 참고 한 번(메인 세션, 루프 전)
  uv run --extra pg python research/combined/prepare.py --build      # 캐시(data.build_all, KR_QUANT_DB)
  uv run python research/combined/prepare.py --manifest               # 코드·캐시 sha 매니페스트(사전등록 때)
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from research.autoresearch import prepare as V4  # noqa: E402
from research.autoresearch.prepare import (  # noqa: E402,F401 — select.py 가 여기서 가져간다
    clamp_cfg,
    exit_line,
    factor_months,
    factor_t,
    fold_stats,
    half_of,
    plateau_pick,
    sample_params,
    sha_file,
)
from research.combined.data import CACHE, ERAS_C, FROZEN, FROZEN_FILES, LOCKED_CACHE, REPO_KEYS, VIEW_KEYS_C, view_c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")
SHADOW = os.path.expanduser("~/.cache/cc-swing/combined_ledger")        # 원장 사본(작업 트리를 지워도 남는다)
RUNS, BEST, LAST = (os.path.join(HERE, f) for f in ("RUNS.jsonl", "BEST.json", "last_run.json"))
BEST_HISTORY = os.path.join(HERE, "BEST_HISTORY.jsonl")
REFERENCE = os.path.join(HERE, "REFERENCE.json")
MANIFEST = os.path.join(HERE, "MANIFEST.json")
PREREG_TAG = "combined-prereg-2"
GUARDED_FILES = ("research/combined/prepare.py", "research/combined/data.py", "research/combined/gaps.py", "research/combined/news.py",
                 "research/combined/select.py", *V4.GUARDED_FILES)
CACHE_ERAS = ("A", "B", "C")                                              # MANIFEST 대상(C 는 확인 — 루프는 못 본다)
ERAS = {e: ERAS_C[e] for e in ("A", "B")}                                 # 루프가 보는 시대
COST_MULT = V4.COST_MULT
SEED = 20261025
N_EVAL, DISPERSION, MIN_TRAIN_TRADES_PER_YEAR = V4.N_EVAL, V4.DISPERSION, V4.MIN_TRAIN_TRADES_PER_YEAR
KEEP_DSCORE, KEEP_CONC_SLACK = 0.005, 0.01
MAX_TRIALS, MAX_SINCE_KEEP = 150, 30
ABL_MIN_CHANGED = 0.05
ABL_MIN_KEEP_TRADES, ABL_MIN_FOLD_TRADES = 0.5, 5                         # 제거 실행 체결 ≥ 원래의 50% ∧ 작동 폴드마다 ≥ 5(정정 0 M1)
ISLAND_RE = re.compile(r"^combined-loop-[1-4]$")
SANDBOX_TIMEOUT_S = 1800                                                  # 격리 실행 시간 초과 = crash(시행으로 센다)
NEWS_YEARS = (2023, 2024)
TOP_DAYS, TOP_TRADE_FRAC = 10, 0.01
CONC_KEYS = ("top1pct", "top_stock", "top10d")
CRASH_TAIL = 20
TRAIN_CHECK_PER_FOLD = 12                                                 # 미래정보 검사에 넣는 학습 점(폴드마다, 비밀 시드)
MUTATORS = {"append", "extend", "insert", "update", "setdefault", "pop", "popitem", "clear", "remove", "add", "discard",
            "__setitem__", "sort", "reverse"}
REF_PARAMS = {"theta": 90, "N": 20, "ma_exit": 100, "hard": 0.15, "exit_kind": "ma", "low_w": 20}   # leader_paper ARMS["leader_C"]
BASELINE_TRAIN_SHA = "2ee5f46b32d26839"                                   # 기준선 train.py sha256 앞 16자(테스트가 대조)
_SANDBOX_HIDE_FIXED = tuple(os.path.expanduser(p) for p in (
    "~/git/daytrade-it", "~/git/quant-airflow", "~/.config", "~/.ssh", "~/.cache/cc-swing/scratch/autoresearch",
    "~/.cache/cc-swing/autoresearch_archive")) + (V4.SHADOW, LOCKED_CACHE, FROZEN)
_G: dict = {}


class StateLeak(RuntimeError):
    """같은 V·파라미터에서 신호가 달라졌다 — train 이 호출 사이 상태로 신호를 바꾼다."""


def log(msg: str) -> None:
    print(f"[{time.time() - _G.get('t0', time.time()):6.0f}s] {msg}", file=sys.stderr, flush=True)


def sandbox_hide() -> tuple:
    return (SHADOW, *_SANDBOX_HIDE_FIXED)


def score_of(al) -> float:
    """점수 = 폴드 alpha 평균 − 0.5σ(ddof 1). 유한값 2개 미만이면 nan."""
    a = np.asarray(al, float)
    a = a[np.isfinite(a)]
    return float(a.mean() - DISPERSION * a.std(ddof=1)) if len(a) >= 2 else float("nan")


# ---------------------------------------------------------------- train.py 검사(v4 + 상태 은닉 금지)

def lint_train(src: str) -> "list[str]":
    """v4 허용 목록 + 상태 은닉 금지 — 위반 목록(비면 통과)."""
    bad = V4.lint_train(src)
    tree = ast.parse(src)
    mods = {a.asname or a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    parent = {ch: n for n in ast.walk(tree) for ch in ast.iter_child_nodes(n)}

    def root(n):
        while isinstance(n, (ast.Attribute, ast.Subscript, ast.Call)):
            n = n.func if isinstance(n, ast.Call) else n.value
        return n

    def literal(v):
        if isinstance(v, ast.Constant):
            return True
        if isinstance(v, (ast.Tuple, ast.List, ast.Set)):
            return all(literal(e) for e in v.elts)
        if isinstance(v, ast.Dict):
            return all(k is None or literal(k) for k in v.keys) and all(literal(e) for e in v.values)
        return isinstance(v, ast.UnaryOp) and literal(v.operand)
    for i, st in enumerate(tree.body):                               # 모듈 최상위: 설명문·import·def/class·리터럴 대입만(적재 시 실행 금지)
        ok = (isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
              or (i == 0 and isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str))
              or (isinstance(st, ast.Assign) and literal(st.value))
              or (isinstance(st, ast.AnnAssign) and (st.value is None or literal(st.value))))
        if not ok:
            bad.append(f"line {st.lineno}: 모듈 최상위 {type(st).__name__} — 설명문·import·def/class·리터럴 대입만")
    bad += lint_price_lattice(src)
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS
                and not isinstance(n.func.value, ast.Name)                   # np.sort·np.append 같은 모듈 함수는 새 객체를 낸다
                and isinstance(root(n.func.value), ast.Name) and root(n.func.value).id in mods):
            bad.append(f"line {n.lineno}: 모듈 {root(n.func.value).id} 에 닿는 객체 변경 .{n.func.attr}()(상태 은닉 금지)")
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bad.append(f"line {n.lineno}: 속성 대입·삭제 .{n.attr}(상태 은닉 금지)")
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, (ast.Store, ast.Del)):
            r = root(n)
            if isinstance(r, ast.Name) and r.id in mods:
                bad.append(f"line {n.lineno}: 모듈 {r.id} 안 첨자 대입·삭제(상태 은닉 금지)")
        elif isinstance(n, ast.Name):
            if n.id == "print":
                bad.append(f"line {n.lineno}: print(출력 통로 금지)")
            elif n.id in mods and not (isinstance(parent.get(n), ast.Attribute) and parent[n].value is n):
                bad.append(f"line {n.lineno}: 모듈 이름 {n.id} 를 값으로 쓰기(속성 접근만 허용)")
    return bad


PRICE_SUBS = ("O", "H", "L", "C")
MOD_ATTRS = {"mod", "rmod", "fmod", "remainder", "gcd", "lcm", "modf", "frexp", "divmod"}
ROUND_ATTRS = {"round", "around", "rint", "floor", "ceil", "trunc", "fix"}


def _is_price_sub(n) -> bool:
    return (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in PRICE_SUBS)


def price_tainted(fn) -> set:
    """함수 안에서 가격 첨자(V["O"|"H"|"L"|"C"])를 담은 식으로 대입된 이름(한 단계 — 그 이름을 다시 거친 이름은 세지 않는다)."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None and any(_is_price_sub(x) for x in ast.walk(n.value)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                out |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
    return out


def has_price(expr, tainted: set) -> bool:
    return any(_is_price_sub(x) or (isinstance(x, ast.Name) and x.id in tainted) for x in ast.walk(expr))


def lint_price_lattice(src: str) -> "list[str]":
    """정정 0-4 금지(가격 수준·격자 통로 — 미래 수정 사건 표지): 나머지·최대공약수 계열 전부, 그리고 가격(V 의 O·H·L·C 첨자나 그것으로
    대입된 이름, 한 단계)에 거는 반올림·내림·올림."""
    bad = []
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]
    taint_of = {}
    for fn in fns:
        t = price_tainted(fn)
        for x in ast.walk(fn):
            taint_of.setdefault(id(x), set()).update(t)
    for n in ast.walk(tree):
        if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)) or (isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Mod)):
            bad.append(f"line {n.lineno}: % 연산 금지(가격 격자·끝자리 통로 — 정정 0-4)")
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        if name in MOD_ATTRS:
            bad.append(f"line {n.lineno}: {name}() 금지(나머지·최대공약수 계열 — 정정 0-4)")
        elif name in ROUND_ATTRS or (isinstance(f, ast.Name) and f.id == "round"):
            t = taint_of.get(id(n), set())
            args = list(n.args) + [k.value for k in n.keywords] + ([f.value] if isinstance(f, ast.Attribute) and not (
                isinstance(f.value, ast.Name) and f.value.id in {"np", "numpy", "math", "pd", "pandas"}) else [])
            if any(has_price(a, t) for a in args):
                bad.append(f"line {n.lineno}: 가격(O·H·L·C)에 {name}() 금지(가격 수준·격자 통로 — 정정 0-4)")
    return bad


def load_strategy(path: str = TRAIN) -> dict:
    """결합 허용 목록을 먼저 통과해야 v4 load_strategy(새 모듈 인스턴스)로 로드한다."""
    bad = lint_train(open(path, encoding="utf-8").read())
    if bad:
        raise SystemExit("train.py 허용 목록 위반 — 평가 거부:\n  " + "\n  ".join(bad))
    return V4.load_strategy(path)


# ---------------------------------------------------------------- 시뮬레이션

def neutral_view(D: dict, neutral: "str | None" = None) -> dict:
    """V(복사본). ``neutral`` 레포의 키를 중립값으로: 갭·호가 비율 0.0, 사건 갭 False, 뉴스 개수 0, news_known False(같은 dtype)."""
    V = view_c(D)
    if neutral is not None:
        if neutral not in REPO_KEYS:
            raise ValueError(f"neutral={neutral!r} 는 {tuple(REPO_KEYS)} 중 하나")
        for k in REPO_KEYS[neutral]:
            V[k] = np.zeros_like(V[k])
    return V


def simulate_c(D: dict, sig: dict, cfg: dict, start: str, end: str, exclude=(), mask=None, cost_mult: float = COST_MULT):
    """v4 simulate 와 같되 신호일 종가 진입. ``mask`` = cand 에서 끌 칸(bool[N,T])."""
    import research.experiments.leader21 as L
    cand = np.asarray(sig["cand"], bool)
    if cand.shape != D["C"].shape:
        raise ValueError("cand·score 는 (종목, 날짜) 모양이어야 한다")
    cand = cand & D["uni"] & D["TR"] & D["eps_up"]                                          # 가드: EPS 상승·유동성·거래 가능
    if mask is not None:
        cand = cand & ~np.asarray(mask, bool)
    score = np.where(cand, np.asarray(sig["score"], float), np.nan)
    c = clamp_cfg(cfg)
    D2 = {**D, "cand": cand, "rs": score, "brk": {0: np.ones_like(cand)}, "ma": {"x": exit_line(D, c["exit_kind"], c["exit_w"])}}
    ecfg = {"theta": -np.inf, "N": 0, "ma_exit": "x", "hard": c["hard"], "pyramid": c["pyramid"], "regime": c["regime"],
            "max_pos": c["max_pos"], "risk": c["risk"], "pos_cap": c["pos_cap"], "entry": "close", "exclude": list(exclude)}
    with np.errstate(invalid="ignore"):
        return L.simulate(D2, ecfg, cost_mult, end=end, start=start, paper=True, net_mtm=True)   # 정지일 매도 미룸 · 순가치


def own_signals(sig, shape) -> dict:
    """train 이 돌려준 신호 → 평가기 소유 사본(cand bool·score float, 새 메모리). 정확히 dict 이고 값은 정확히 수 dtype np.ndarray(shape)만
    받는다(dict·ndarray 하위 클래스·``__array__`` 객체·object dtype 거부 — 읽을 때마다 train 코드가 돌 길 차단). 지문·시뮬레이션은 이 사본만
    쓴다: train 이 쥔 배열을 뒤에(cfg 등) 제자리에서 고쳐도 평가에 닿지 않는다(리뷰 rr6)."""
    if type(sig) is not dict:
        raise TypeError(f"signals 는 dict 를 돌려야 한다({type(sig).__name__})")
    out = {}
    for k, dt in (("cand", bool), ("score", float)):
        v = dict.get(sig, k)
        if type(v) is not np.ndarray or v.dtype.kind not in "biuf" or v.shape != tuple(shape):
            raise TypeError(f"signals[{k!r}] 는 수 dtype np.ndarray{tuple(shape)} 여야 한다({type(v).__name__})")
        out[k] = np.array(v, dtype=dt, copy=True)
    return out


def call_signals(S: dict, V: dict, x: dict, shape: "tuple | None" = None) -> dict:
    """signals 호출의 유일한 길 — x 는 사본으로 넘기고(train 이 평가기 파라미터를 고치지 못한다) 결과는 곧바로 소유 사본으로.
    기대 모양은 train 코드가 돌기 **전에** 정한다(``shape`` 또는 호출 전 V["C"].shape) — signals 가 V["C"] 를 바꿔 끼워도 모양 검사를 못 속인다."""
    shape = tuple(V["C"].shape if shape is None else shape)
    return own_signals(S["signals"](V, dict(x)), shape)


def sig_hash(sig: dict) -> str:
    """신호 지문 — cand 와 cand 칸의 score."""
    c = np.ascontiguousarray(np.asarray(sig["cand"], bool))
    s = np.where(c, np.nan_to_num(np.asarray(sig["score"], float), nan=-1.5e300), 0.0)
    h = hashlib.sha1(c.tobytes())
    h.update(np.ascontiguousarray(s).tobytes())
    return h.hexdigest()


def daily_sel(D: dict, eq: pd.Series, lo: str, hi: str) -> np.ndarray:
    """[lo, hi) 일별 선택 알파(로그) — 합 = fold_stats 의 (ls − lm)."""
    ix = (eq.index >= lo) & (eq.index < hi)
    x = eq[ix].to_numpy(float)
    if len(x) < 2:
        return np.zeros(0)
    k = D["kx"][D["days"].get_indexer(eq.index[ix])]
    expo = np.asarray(eq.attrs["expo"], float)[ix]
    return np.log(x[1:] / x[:-1]) - np.log1p(expo[:-1] * (k[1:] / k[:-1] - 1))


def trade_key(tr: pd.DataFrame) -> str:
    pairs = sorted(zip(tr["ci"].astype(int).tolist(), tr["ti"].astype(int).tolist())) if len(tr) else []
    return hashlib.sha1(json.dumps(pairs).encode()).hexdigest()[:12]


_CFG_MEMO: dict = {}


def _cfg_task(args):
    """spawn 작업자: signals 를 한 번도 부르지 않은 새 프로세스·새 모듈에서 cfg(x) 만 계산한다."""
    train_path, xs = args
    S = load_strategy(train_path)
    return [S["cfg"](dict(x)) for x in xs]


def _xkey(x: dict) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def fresh_cfgs(train_path: str, xs: list) -> "list[dict]":
    """cfg(x) — signals 를 부른 적 없는 새 프로세스에서 계산한 값(train sha·x 로 기억). 채점·재실행은 이 값만 쓴다.
    signals 가 모듈 전역 등에 숨긴 전체 기간 V 를 cfg 가 읽는 길(리뷰 IMPORTANT A)을 막는다."""
    sha = hashlib.sha256(open(train_path, "rb").read()).hexdigest()[:16]
    need = list({_xkey(x): x for x in xs if (sha, _xkey(x)) not in _CFG_MEMO}.values())
    if need:
        with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
            out = pool.map(_cfg_task, [(train_path, need)])[0]
        for x, c in zip(need, out, strict=True):
            _CFG_MEMO[(sha, _xkey(x))] = dict(c)
    return [dict(_CFG_MEMO[(sha, _xkey(x))]) for x in xs]


def training_points(spec: dict, fi: int, ei: int) -> "tuple[list[dict], np.ndarray]":
    """폴드 fi(시대 순번 ei)의 학습 점 — 워크포워드와 미래정보 검사가 같은 점을 쓴다."""
    return sample_params(spec, N_EVAL, SEED + 101 * fi + 7 * ei)


def causality_seed(train_path: str = TRAIN) -> int:
    """학습 점 표본 시드 = sha256(train sha + 비밀 소금). 소금은 원장 사본 디렉터리에 한 번 만든다(격리가 가린다) — train 은 어느 점이
    검사될지 모른다."""
    os.makedirs(SHADOW, exist_ok=True)
    p = os.path.join(SHADOW, "causality_salt")
    if not os.path.exists(p):
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, os.urandom(16).hex().encode())
            os.close(fd)
        except FileExistsError:
            pass
    salt = open(p, encoding="utf-8").read().strip()
    sha = hashlib.sha256(open(train_path, "rb").read()).hexdigest()[:16]
    return int(hashlib.sha256((sha + salt).encode()).hexdigest()[:12], 16)


def check_points(spec: dict, n_folds: int, ei: int, seed: int, k: int = TRAIN_CHECK_PER_FOLD) -> "list[dict]":
    """미래정보·해시 검사에 넣을 학습 점 표본 — 폴드마다 k 개, 비밀 시드(고원 선택을 미래로 조종하는 train 차단)."""
    out = []
    for fi in range(n_folds):
        pts, _ = training_points(spec, fi, ei)
        idx = np.random.default_rng([seed, fi, ei]).choice(len(pts), size=min(k, len(pts)), replace=False)
        out += [pts[int(i)] for i in sorted(idx)]
    return out


def _signals_or_err(S: dict, V: dict, x: dict, shape: "tuple | None" = None):
    try:
        return call_signals(S, V, x, shape)
    except Exception:  # noqa: BLE001 — 오류(형식 거부 포함)도 비교 대상(같은 x 면 어디서나 같아야 한다)
        return None


def _train_point(args):
    fi, j, x, cfg = args
    D, S, lo, hi = _G["D"], _G["S"], _G["folds"][fi][0], _G["folds"][fi][1]
    sig = _signals_or_err(S, _G["V"], x, D["C"].shape)               # 소유 사본 — 지문·시뮬레이션 사이에 train 코드는 돌지 않는다
    h = "ERR" if sig is None else sig_hash(sig)                      # 고원 선택에 실제로 쓴 신호의 지문(새 프로세스 값과 비교)
    try:
        if sig is None:
            raise RuntimeError("signals 실패")
        eq, tr = simulate_c(D, sig, cfg, lo, hi)
        st = fold_stats(D, eq, lo, hi)
        n = int((tr["entry"] >= lo).sum()) if len(tr) else 0
        return fi, j, (st["alpha"] if n >= MIN_TRAIN_TRADES_PER_YEAR * st["years"] else -1.0), h
    except Exception as e:  # noqa: BLE001 — 한 점 실패는 그 점만 버린다
        log(f"학습 점 fold {fi} #{j} 실패: {type(e).__name__}")              # 메시지는 숨긴다(출력 통로 차단)
        return fi, j, -1.0, h


def walk_forward_c(D_by_era: dict, workers: int, train_path: str = TRAIN, fixed: "list | None" = None, null: "tuple | None" = None,
                   neutral: "str | None" = None, cost_mult: float = COST_MULT, exclude=(), mask: "dict | None" = None,
                   eras: "dict | None" = None) -> dict:
    """시대마다 새 모듈로 train.py 로드 → 폴드별 학습 구간 고원 선택 → 시험 한 번(v4 walk_forward 복사).
    ``fixed`` = 그 폴드 파라미터 그대로, ``null`` = ("rank"|"thin"|"params", seed), ``neutral`` = 레포 키 중립화,
    ``exclude`` = 뺄 종목코드, ``mask`` = {시대: bool[N,T]} cand 에서 끌 칸, ``eras`` = 돌 시대(기본 루프 A·B)."""
    if neutral is not None and neutral not in REPO_KEYS:
        raise ValueError(f"neutral={neutral!r}")
    eras = ERAS if eras is None else eras
    res = {"folds": [], "trades": [], "months": [], "daily": [], "sig_hash": [], "cfg_mismatch": [], "point_hash": {}}
    for ei, (era, E) in enumerate(eras.items()):
        D = D_by_era[era]
        S = load_strategy(train_path)
        _G.update(D=D, S=S, V=neutral_view(D, neutral), folds=E["folds"])
        rng = np.random.default_rng((null[1] if null else 0) * 7919 + ei)
        if fixed is not None:
            chosen = [f["params"] for f in fixed if f["era"] == era]
            if null and null[0] == "params":
                chosen = [sample_params(S["params"], 1, int(rng.integers(1 << 30)))[0][0] for _ in chosen]
        else:
            tasks, pts_all = [], {}
            for fi in range(len(E["folds"])):
                pts, U = training_points(S["params"], fi, ei)
                pts_all[fi] = (pts, U)
                tasks += [(fi, j, x) for j, x in enumerate(pts)]
            tasks = [(*t, c) for t, c in zip(tasks, fresh_cfgs(train_path, [t[2] for t in tasks]), strict=True)]
            _signals_or_err(S, _G["V"], tasks[0][2], D["C"].shape)   # 부모에서 train.py 캐시를 데운다(fork 공유, 결과는 버린다)
            gc.collect()                                             # 앞 재실행의 쓰레기를 작업자가 물려받지 않게
            gc.freeze()                                              # 작업자 gc 가 부모 객체 머리를 건드려 쓰기 시 복사가 깨지지 않게(정정 1)
            try:
                with mp.get_context("fork").Pool(max(1, workers)) as pool:
                    out = pool.map(_train_point, tasks, chunksize=1)
            finally:
                gc.unfreeze()
            chosen, ph = [], res["point_hash"].setdefault(era, {})
            for fi, j, _, h in out:                                  # 같은 x 가 두 폴드에 나오면 지문도 같아야 한다
                k = _xkey(pts_all[fi][0][j])
                ph[k] = h if ph.get(k, h) == h else "CONFLICT"
            for fi in range(len(E["folds"])):
                pts, U = pts_all[fi]
                y = np.array([v for f, _, v, _ in sorted(out, key=lambda r: (r[0], r[1])) if f == fi])
                chosen.append(pts[plateau_pick(U, y)])
        cfgs = fresh_cfgs(train_path, chosen)
        for fi, (_, lo, hi) in enumerate(E["folds"]):
            x, cfg = chosen[fi], cfgs[fi]
            sig = call_signals(S, _G["V"], x, D["C"].shape)          # 소유 사본 — 여기서 시뮬레이션 끝까지 train 코드를 부르지 않는다
            res["sig_hash"].append(sig_hash(sig))
            if null and null[0] == "rank":
                sig = {**sig, "score": rng.random(D["C"].shape)}
            elif null and null[0] == "thin":                         # 후보 솎기: 시험 구간 월별 통과율 그대로
                base = D["uni"] & D["TR"] & D["eps_up"]
                c = np.asarray(sig["cand"], bool) & base
                mon = D["days"].to_period("M")
                pr = np.zeros(len(mon))
                for mm in np.unique(mon):
                    j = mon == mm
                    pr[j] = c[:, j].sum() / max(base[:, j].sum(), 1)
                sig = {"cand": base & (rng.random(D["C"].shape) < pr[None, :]), "score": rng.random(D["C"].shape)}
            eq, tr = simulate_c(D, sig, cfg, lo, hi, exclude=exclude, mask=None if mask is None else mask.get(era),
                                cost_mult=cost_mult)
            tr = tr[(tr["entry"] >= lo) & (tr["entry"] < hi)] if len(tr) else tr
            if len(tr):                                              # 진입일 청산선까지 거리(종가 대비) — '좁은 실질 손절' 감시
                c = clamp_cfg(cfg)
                line = exit_line(D, c["exit_kind"], c["exit_w"])
                ci = pd.Index(D["codes"]).get_indexer(tr["code"])
                ti = D["days"].get_indexer(pd.to_datetime(tr["entry"]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    tr = tr.assign(ci=ci, ti=ti, exit_dist=1.0 - line[ci, ti] / D["C"][ci, ti])
            st = fold_stats(D, eq, lo, hi)
            gi = len(res["folds"])
            res["folds"].append({"era": era, "test": [lo, hi], "params": x, "alpha": st["alpha"], "excess": st["excess"],
                                 "years": st["years"], "n": int(len(tr)), "half": D.get("half_id", 0), "trade_key": trade_key(tr)})
            res["months"].append(st["m"])
            res["daily"].append(daily_sel(D, eq, lo, hi))
            if len(tr):
                res["trades"].append(tr.assign(era=era, fold=gi))
            if clamp_cfg(S["cfg"](dict(x))) != clamp_cfg(cfg):          # 시뮬레이션 뒤: signals 를 부른 뒤의 cfg 가 새 프로세스 cfg 와 다르면
                res["cfg_mismatch"].append(gi)                       # 상태 의존(채점은 새 프로세스 cfg)
        _release()                                                   # 시대 끝: 그 시대 모듈 인스턴스(train 캐시)·V 를 놓는다(정정 1)
        S = sig = eq = tr = None
        gc.collect()                                                 # 모듈 전역 ↔ 함수 순환 참조라 참조 수만으로는 안 풀린다
    return res


def _release() -> None:
    """``_G`` 에서 시대 상태(D·S·V·folds)를 뺀다 — 다음 재실행·다음 단계로 train 캐시와 V 사본이 쌓이지 않게(정정 1)."""
    for k in ("D", "S", "V", "folds"):
        _G.pop(k, None)


def _same_signals(res: dict, r: dict, what: str) -> None:
    if res.get("sig_hash") and r.get("sig_hash") != res["sig_hash"]:
        raise StateLeak(f"fail: {what} 재실행 신호가 원래 실행과 다르다(같은 V·파라미터) — 상태 의존 신호")


def active_folds(folds: list, repo: str) -> "list[int]":
    """그 레포 부품이 작동하는 폴드 — daytrade 는 시험 연도 2023·2024(뉴스 아카이브), scalp 는 전부."""
    if repo == "daytrade":
        return [i for i, f in enumerate(folds) if int(str(f["test"][0])[:4]) in NEWS_YEARS]
    return list(range(len(folds)))


def summarize_c(res: dict) -> dict:
    s = V4.summarize(res)                                            # void 규칙·손익비·거래당(비용 2배) = v4 그대로
    s["ev_net_2x"] = s["trade_ret_mean"]
    s["news_folds_score"] = score_of([res["folds"][i]["alpha"] for i in active_folds(res["folds"], "daytrade")])
    return s


def _alphas(res: dict, idx=None) -> list:
    return [f["alpha"] for i, f in enumerate(res["folds"]) if idx is None or i in idx]


def top10d_score(daily: list, years: list, k: int = TOP_DAYS) -> float:
    """시험 구간 일별 선택 알파를 폴드를 합쳐 상위 k 일 0 으로 → 폴드 alpha = 합 / 연수 → score."""
    lens = [len(d) for d in daily]
    pooled = np.concatenate(daily) if lens and sum(lens) else np.zeros(0)
    if len(pooled):
        pooled = pooled.copy()
        pooled[np.argsort(-pooled, kind="stable")[:k]] = 0.0
    parts = np.split(pooled, np.cumsum(lens)[:-1]) if lens else []
    return score_of([p.sum() / max(y, 1e-9) for p, y in zip(parts, years, strict=True)])


def concentration(D_by_era: dict, res: dict, train_path: str = TRAIN, eras: "dict | None" = None) -> dict:
    """쏠림 진단 셋 — 각각 제거 뒤 score(void 무관 원점수). 파라미터는 고른 값 그대로. 재실행 신호가 원래와 다르면 StateLeak."""
    eras = ERAS if eras is None else eras
    if not res["trades"]:
        return {k: float("nan") for k in CONC_KEYS}
    tr = pd.concat(res["trades"])
    top = tr.nlargest(max(1, math.ceil(TOP_TRADE_FRAC * len(tr))), "ret")
    mask = {e: np.zeros(D_by_era[e]["C"].shape, bool) for e in eras}
    for era, ci, ti in zip(top["era"], top["ci"], top["ti"], strict=True):
        mask[era][int(ci), int(ti)] = True                           # 종가 진입: 신호 칸 = 진입 칸
    r1 = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], mask=mask, eras=eras)
    top_code = tr.groupby("code")["ret"].sum().idxmax()             # 순수익 합 최대 종목
    r2 = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], exclude=[top_code], eras=eras)
    _same_signals(res, r1, "상위 1%")
    _same_signals(res, r2, "최대 종목")
    return {"top1pct": score_of(_alphas(r1)), "top_stock": score_of(_alphas(r2)),
            "top10d": top10d_score(res["daily"], [f["years"] for f in res["folds"]])}


def _tset(res: dict, idx: list) -> set:
    out = set()
    for tr in res["trades"]:
        tr = tr[tr["fold"].isin(idx)]
        out |= set(zip(tr["fold"].astype(int), tr["ci"].astype(int), tr["ti"].astype(int)))
    return out


def ablation(D_by_era: dict, res: dict, train_path: str = TRAIN, eras: "dict | None" = None) -> dict:
    """레포 부품 제거 — 중립 V 로 폴드 파라미터 고정 재실행. 작동 폴드에서 체결 대칭차 ≥ 5% ∧ 점수 기여 > 0 ∧
    제거 실행이 **여전히 거래한다**(작동 폴드 체결 합 ≥ 원래의 50% ∧ 폴드마다 ≥ 5)이면 ok.
    마지막 조건(정정 0 M1): 중립 입력에서 거래를 멈추는 부품(news_known 으로 cand 를 막는 문 등)은 "부품 끔"이 아니라 "전략 끔"이라
    dscore 가 공짜로 양수가 된다 — 그것을 자격으로 치지 않는다. 중립 입력 = 통과(pass-through)가 program 규칙이다."""
    out = {}
    for repo in REPO_KEYS:
        idx = active_folds(res["folds"], repo)
        r = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], neutral=repo, eras=eras)
        a, b = _tset(res, idx), _tset(r, idx)
        changed = len(a ^ b) / max(len(a), 1)
        dscore = score_of(_alphas(res, idx)) - score_of(_alphas(r, idx))
        n0 = sum(res["folds"][i]["n"] for i in idx)
        n1 = sum(r["folds"][i]["n"] for i in idx)
        fmin = min((r["folds"][i]["n"] for i in idx), default=0)
        trades_ok = bool(n1 >= ABL_MIN_KEEP_TRADES * n0 and fmin >= ABL_MIN_FOLD_TRADES)
        out[repo] = {"changed": float(changed), "dscore": float(dscore), "n_orig": int(n0), "n_removed": int(n1),
                     "min_fold_n_removed": int(fmin), "trades_ok": trades_ok,
                     "ok": bool(changed >= ABL_MIN_CHANGED and dscore > 0 and trades_ok)}
    out["eligible"] = all(out[r]["ok"] for r in REPO_KEYS)
    return out


# ---------------------------------------------------------------- 미래정보 검사(새 프로세스마다 한 절단)

def _cut(D: dict, cut: int, shuffle_seed: "int | None" = None) -> dict:
    T = D["C"].shape[1]
    perm = None
    if shuffle_seed is not None:
        perm = np.r_[np.arange(cut + 1), cut + 1 + np.random.default_rng(shuffle_seed).permutation(T - cut - 1)]
    out = {}
    for k in VIEW_KEYS_C + ("kx", "kq", "regime_on"):
        v = D[k]
        if perm is not None:
            out[k] = v[:, perm] if v.ndim == 2 else v[perm] if len(v) == T else v
        else:
            out[k] = v[:, :cut + 1] if v.ndim == 2 else v[:cut + 1] if len(v) == T else v
    if "vmult" in D:                                                  # 종목별 가격 배수(시간 축 없음) — 절단 V 도 같은 배수로(K1)
        out["vmult"] = np.array(D["vmult"], copy=True)
        out["mkt_real"] = np.array(D["mkt_real"], copy=True)          # 격자 반올림은 그 칸의 날짜·시장 호가표로(섞어도 칸과 날짜가 같이 움직인다)
        days = pd.DatetimeIndex(D["days"])
        out["days"] = days[perm] if perm is not None else days[:cut + 1]
    return out


def pack_signals(cand: np.ndarray, score: np.ndarray, h) -> tuple:
    """검사 작업자가 돌려주는 작은 꼴(정정 1) — ((모양, cand 비트 묶음), cand 칸 score(행 우선 순서), 지문). 비교에 쓰는 것은 cand 와
    cand 칸 score 뿐이라 전체 배열(x 마다 N×T float64)을 부모로 보내지 않는다(옛 꼴은 시대마다 수십 GB 를 부모에 쌓았다)."""
    c = np.ascontiguousarray(cand, dtype=bool)
    return (c.shape, np.packbits(c, axis=None)), np.asarray(score, float)[c], h


def unpack_cand(pk: tuple) -> np.ndarray:
    shape, bits = pk
    return np.unpackbits(bits, count=int(np.prod(shape))).astype(bool).reshape(shape)


def _cause_task(args):
    """spawn 작업자(작업 하나 = 프로세스 하나): 디스크 캐시(경로) 또는 넘겨받은 배열에서 V 를 만들고 train.py 를 새로 로드한다.
    결과는 x 마다 ``pack_signals`` 꼴(오류면 (None, None, "ERR"))."""
    era, src, cut, mode, xs, train_path = args
    if isinstance(src, str):
        from research.combined.data import require_vmult
        with open(src, "rb") as f:
            src = require_vmult(pickle.load(f))
    V = view_c(src) if mode == "full" else view_c(_cut(src, cut, shuffle_seed=cut if mode == "shuffle" else None))
    src = None                                                       # 시대 원본은 V 를 만든 뒤 놓는다(정정 1)
    shape = V["C"].shape                                             # train 코드가 돌기 전에
    S = load_strategy(train_path)
    out = []
    for x in xs:
        g = _signals_or_err(S, V, x, shape)
        if g is None:
            out.append((None, None, "ERR"))
        elif mode == "full":
            out.append(pack_signals(g["cand"], g["score"], sig_hash(g)))   # g = 소유 사본(call_signals) — 다음 signals 호출 전에 묶는다
        else:
            out.append(pack_signals(g["cand"][:, :cut + 1], g["score"][:, :cut + 1], None))
    return era, cut, mode, out


def _cut_fail(era, cut: int, mode: str, xs: list, full_parts: list, parts: list) -> "str | None":
    """절단 작업 하나를 전체 신호와 비교 — 첫 불일치 메시지(없으면 None). 옛 비교와 같다: cand 는 전체 cand 의 [:, :cut+1] 과 칸마다 같아야 하고,
    score 는 그 cand 칸에서 같아야 한다(cand 가 같을 때만 비교하므로 절단 쪽 cand 칸 = 전체 쪽 cand 칸)."""
    for x, (pa, va, hf), (pb, vb, ht) in zip(xs, full_parts, parts, strict=True):
        if hf == "ERR" or ht == "ERR":
            if hf != ht:
                return f"fail: era {era} cut {cut} {mode} params {x}: 오류 불일치(전체 {hf != 'ERR'} · 절단 {ht != 'ERR'})"
            continue
        ca_full = unpack_cand(pa)
        ca, cb = ca_full[:, :cut + 1], unpack_cand(pb)
        if ca.shape != cb.shape or (ca != cb).any():
            return f"fail: era {era} cut {cut} {mode} params {x}: cand 불일치 {int((ca != cb).sum()) if ca.shape == cb.shape else 'shape'}칸"
        sa = va[np.nonzero(ca_full)[1] <= cut]                       # = 전체 score[:, :cut+1][ca](행 우선 순서 그대로)
        if not np.allclose(sa, vb, rtol=1e-9, atol=1e-12, equal_nan=True):
            return f"fail: era {era} cut {cut} {mode} params {x}: score 불일치"
    return None


def causality_c(D_by_era: dict, res: dict, train_path: str = TRAIN, workers: int = 4, n_cut: int = 12,
                sources: "dict | None" = None, eras: "dict | None" = None, check_seed: "int | None" = None) -> "tuple[bool, str]":
    """허용 0, 시대마다 절단 n_cut 곳(시험 폴드 시작 + 무작위), 절단 · 절단 뒤 시간 섞기 두 방식.
    검사하는 파라미터 = 그 시대 폴드 파라미터 전부 + 학습 점 표본(폴드마다 ``TRAIN_CHECK_PER_FOLD``, 비밀 시드 ``check_seed`` — 없으면
    ``causality_seed``). ``res["point_hash"]`` 가 있으면 표본 학습 점의 fork 작업자 신호(고원 선택에 쓴 것)가 새 프로세스 신호와 같아야 한다.
    전체 신호도 절단 신호도 새 프로세스에서 만든다(평가 프로세스의 숨긴 상태 차단).
    ``sources`` = {시대: 캐시 pickle 경로} — 작업자가 디스크에서 읽는다. 없으면 시대 dict 전체를 작업마다 pickle 로 넘긴다(맞지만 느리고
    작업자 수만큼 메모리를 쓴다 — 큰 시대(select.py 의 C 등)는 경로가 있으면 넘긴다. 메모리에만 있는 잠금 절반은 경로 없이 넘겨도 된다).
    ``res["sig_hash"]`` 가 있으면 새 프로세스 전체 신호가 평가에 쓴 신호와 같아야 한다."""
    eras = ERAS if eras is None else eras
    spec = load_strategy(train_path)["params"]                       # 학습 점 표본용(get_strategy 만 — signals 는 부르지 않는다)
    seed = causality_seed(train_path) if check_seed is None else check_seed
    tasks, xs_by, sampled = [], {}, {}
    for ei, (era, E) in enumerate(eras.items()):
        D = D_by_era[era]
        src = sources[era] if sources else D
        T = D["C"].shape[1]
        xs = []
        sampled[era] = check_points(spec, len(E["folds"]), ei, seed)
        for x in [f["params"] for f in res["folds"] if f["era"] == era] + sampled[era]:
            if x not in xs:
                xs.append(x)
        xs_by[era] = xs
        tasks.append((era, src, -1, "full", xs, train_path))
        starts = [int(D["days"].searchsorted(pd.Timestamp(lo))) for _, lo, _ in E["folds"]]
        rnd = np.random.default_rng(SEED + len(starts)).integers(int(T * 0.3), T - 2, size=max(0, n_cut - len(starts)))
        for cut in sorted(set(starts + [int(c) for c in rnd])):
            for mode in ("trunc", "shuffle"):
                tasks.append((era, src, min(cut, T - 2), mode, xs, train_path))
    full, first_fail, n = {}, None, 0                                 # 흘려 비교(정정 1): 절단 결과는 받는 즉시 비교하고 버린다
    with mp.get_context("spawn").Pool(max(1, min(workers, len(tasks))), maxtasksperchild=1) as pool:
        for era, cut, mode, parts in pool.imap(_cause_task, tasks, chunksize=1):   # 작업 순서 그대로(시대마다 전체가 절단보다 먼저)
            if mode == "full":
                full[era] = parts
                continue
            n += 1
            if first_fail is None:
                first_fail = _cut_fail(era, cut, mode, xs_by[era], full[era], parts)
    if res.get("sig_hash"):
        for f, h in zip(res["folds"], res["sig_hash"], strict=True):
            if f["era"] in full and full[f["era"]][xs_by[f["era"]].index(f["params"])][2] != h:
                return False, f"fail: era {f['era']} params {f['params']}: 새 프로세스 신호 ≠ 평가 신호(상태 의존)"
    for era, ph in (res.get("point_hash") or {}).items():
        for x in sampled.get(era, []):
            k = _xkey(x)
            if k in ph and full[era][xs_by[era].index(x)][2] != ph[k]:
                return False, f"fail: era {era} 학습 점 {x}: 고원 선택에 쓴 신호 ≠ 새 프로세스 신호(상태 의존)"
    if first_fail is not None:                                       # 우선순위는 옛 코드와 같다: 신호 해시 → 학습 점 지문 → 첫 절단 불일치
        return False, first_fail
    return True, f"ok({n} 검사)"


# ---------------------------------------------------------------- 계산 본체(격리 안)

def run_all(D_by_era: dict, workers: int, train_path: str = TRAIN, sources: "dict | None" = None, eras: "dict | None" = None,
            check_seed: "int | None" = None) -> dict:
    S = load_strategy(train_path)
    res = walk_forward_c(D_by_era, workers, train_path, eras=eras)
    log("워크포워드 끝")
    s = summarize_c(res)
    ok, why = causality_c(D_by_era, res, train_path, workers=max(1, workers // 2), sources=sources, eras=eras, check_seed=check_seed)
    if ok and res.get("cfg_mismatch"):
        ok, why = False, f"fail: cfg 가 signals 뒤 상태에 따라 바뀐다(폴드 {res['cfg_mismatch']}) — 채점은 새 프로세스 cfg"
    log("미래정보 검사 끝")
    conc = {k: float("nan") for k in CONC_KEYS}
    abl = {"scalp": None, "daytrade": None, "eligible": False}
    s["ev_net_1x"] = float("nan")
    if ok and not s["void"]:
        try:
            conc = concentration(D_by_era, res, train_path, eras=eras)
            abl = ablation(D_by_era, res, train_path, eras=eras)
            c1 = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], cost_mult=1.0, eras=eras)
            _same_signals(res, c1, "비용 1배")
            s["ev_net_1x"] = float(pd.concat(c1["trades"])["ret"].mean()) if c1["trades"] else float("nan")
            log("쏠림·부품 제거·비용 1배 끝")
        except StateLeak as e:
            ok, why = False, str(e)
            conc = {k: float("nan") for k in CONC_KEYS}
            abl = {"scalp": None, "daytrade": None, "eligible": False}
    if not ok:
        s.update(void=True, score=-1.0)
    s["lookahead_check"] = why
    s["train_sha"] = S["sha"]
    return {"summary": s, "folds": [dict(f) for f in res["folds"]], "conc": conc, "ablation": abl}


def _sources(eras) -> dict:
    return {e: os.path.join(CACHE, f"era_{e}_s0.pkl") for e in eras}


def inner(workers: int) -> None:
    """bwrap 안에서 도는 계산(train.py 코드가 실행되는 유일한 곳) — 결과 JSON 한 줄을 stdout 에. 원장·BEST 는 바깥이 쓴다."""
    from research.combined.data import load
    _G["t0"] = time.time()
    seed = int(sys.stdin.readline().strip())                         # 비밀 표본 시드 — 바깥이 stdin 으로(격리 안에서는 소금을 못 읽는다)
    out = run_all({e: load(e) for e in ERAS}, workers, sources=_sources(ERAS), check_seed=seed)
    sys.stdout.write("@@RESULT@@" + json.dumps(out, ensure_ascii=False, default=str) + "\n")


def inner_reference(workers: int) -> None:
    """leader_C 고정값을 기준선 train.py 로 두 시대에 — 폴드 파라미터 고정(학습 없음)."""
    from research.combined.data import load
    _G["t0"] = time.time()
    D_by_era = {e: load(e) for e in ERAS}
    fixed = [{"era": e, "params": dict(REF_PARAMS)} for e in ERAS for _ in ERAS[e]["folds"]]
    res = walk_forward_c(D_by_era, workers, TRAIN, fixed=fixed)
    out = {"summary": summarize_c(res), "folds": [dict(f) for f in res["folds"]]}
    sys.stdout.write("@@RESULT@@" + json.dumps(out, ensure_ascii=False, default=str) + "\n")


def sandbox_cmd_c(args: "list[str]") -> "list[str]":
    """bwrap: 루트 읽기 전용, 네트워크·PID 분리, 결합 캐시 디렉터리는 빈 tmpfs 뒤 era_A·era_B 만 다시 보이게(era_C 는 안 보인다)."""
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/dev/shm",
           "--unshare-net", "--unshare-pid", "--die-with-parent", "--tmpfs", CACHE]
    for e in ("A", "B"):
        f = os.path.join(CACHE, f"era_{e}_s0.pkl")
        cmd += ["--ro-bind", f, f]
    for h in sandbox_hide():
        if os.path.isdir(h):
            cmd += ["--tmpfs", h]
    py = os.path.join(ROOT, ".venv", "bin", "python")
    return cmd + ["--chdir", ROOT, "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "NUMBA_NUM_THREADS", "1", "--unsetenv", "KR_QUANT_DB",
                  py, os.path.abspath(__file__), *args]


# ---------------------------------------------------------------- 원장·keep·상한(바깥, 신뢰)

_LOCK = {"depth": 0, "fd": None}


@contextmanager
def ledger_lock():
    """원장 사본 디렉터리 ``ledger.lock`` 의 배타 flock(같은 프로세스 안에서는 다시 들어갈 수 있다) — 섬 여럿이 공유 원장을 함께 쓴다."""
    if _LOCK["depth"] == 0:
        os.makedirs(SHADOW, exist_ok=True)
        fd = os.open(os.path.join(SHADOW, "ledger.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        _LOCK["fd"] = fd
    _LOCK["depth"] += 1
    try:
        yield
    finally:
        _LOCK["depth"] -= 1
        if _LOCK["depth"] == 0:
            fcntl.flock(_LOCK["fd"], fcntl.LOCK_UN)
            os.close(_LOCK["fd"])
            _LOCK["fd"] = None


def _append(path: str, rec: dict) -> None:
    os.makedirs(SHADOW, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with ledger_lock():
        for p in (path, os.path.join(SHADOW, os.path.basename(path))):
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)


def _read_jsonl(path: str) -> list:
    out = []
    for p in (path, os.path.join(SHADOW, os.path.basename(path))):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append([json.loads(x) for x in f if x.strip()])
    return max(out, key=len) if out else []


def _write_both(path: str, obj: dict) -> None:
    os.makedirs(SHADOW, exist_ok=True)
    with ledger_lock():
        for p in (path, os.path.join(SHADOW, os.path.basename(path))):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def current_branch() -> str:
    p = subprocess.run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def island_of(sha: str) -> str:
    """평가할 섬 — 브랜치가 ``combined-loop-[1-4]`` 면 그 이름, 아니면 기준선 train.py 일 때만 "main", 그 밖은 거부."""
    b = current_branch()
    if ISLAND_RE.match(b):
        return b
    if sha == BASELINE_TRAIN_SHA:
        return "main"
    raise SystemExit(f"브랜치 {b!r} 는 섬이 아니다(combined-loop-1~4) — 평가 거부(기준선 train.py 만 main 에서 평가된다)")


def check_island_path(island: str) -> None:
    """섬 ↔ 워크트리 경로 대응을 숨긴 원장 디렉터리 ``islands.json`` 에 첫 사용 때 기록하고 그 뒤로는 대조한다(정정 0-3 사소 4).
    같은 섬을 다른 워크트리에서, 또는 한 워크트리를 다른 섬 브랜치로 돌리면 거부한다. 섬 main(기준선)은 검사하지 않는다."""
    if island == "main":
        return
    p = os.path.join(SHADOW, "islands.json")
    with ledger_lock():
        m = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        if island in m and m[island] != ROOT:
            raise SystemExit(f"섬 {island} 은 워크트리 {m[island]} 에 묶여 있다(지금 {ROOT}) — 평가 거부")
        other = [k for k, v in m.items() if v == ROOT and k != island]
        if other:
            raise SystemExit(f"워크트리 {ROOT} 는 섬 {other[0]} 에 묶여 있다 — 브랜치를 바꿔 다른 섬으로 쓰지 않는다")
        if island not in m:
            m[island] = ROOT
            with open(p, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=1)


def _isl(r: dict) -> str:
    return r.get("island", "main")


def best_path(island: str) -> str:
    return BEST.replace(".json", f"_{island}.json")


def git_state() -> dict:
    """원장 행에 남길 git 상태 — HEAD 앞 12자, train.py 가 HEAD 와 다른지(추적 안 됨·스테이지 포함)."""
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    head = git("rev-parse", "HEAD")
    st = git("status", "--porcelain", "--", os.path.relpath(TRAIN, ROOT))
    return {"commit": head.stdout.strip()[:12] if head.returncode == 0 else None,
            "dirty": bool(st.stdout.strip()) if st.returncode == 0 else None}


def pin_ref(sha: str, gs: dict) -> "str | None":
    """평가한 commit 을 ``refs/combined/<train sha>/<commit 앞 12>`` 로 붙잡는다 — 루프가 discard 로 ``reset --hard`` 해도 gc 가 지우지 못해
    select.py 가 원장 commit 에서 train.py 를 되살릴 수 있다. dirty·commit 없음이면 붙잡지 않는다(그 행은 어차피 풀에서 빠진다)."""
    if not gs.get("commit") or gs.get("dirty") is not False:
        return None
    ref = f"refs/combined/{sha}/{gs['commit']}"
    p = subprocess.run(["git", "-C", ROOT, "update-ref", ref, gs["commit"]], capture_output=True, text=True)
    return ref if p.returncode == 0 else None


def trial_state() -> dict:
    """n = 평가된 고유 train sha 수(crash 포함, 모든 섬 합쳐). since_keep = 마지막 keep(어느 섬이든 beats_best True 평가 행) 뒤 새 sha 수.
    stop = n ≥ 150 ∨ (keep 있음 ∧ since_keep ≥ 30). ``shas`` = 센 sha 집합."""
    seen, since, kept = set(), 0, False
    for r in _read_jsonl(RUNS):
        sha = r.get("train_sha")
        if sha is None or not ("folds" in r or r.get("crash")):
            continue
        new = sha not in seen
        seen.add(sha)
        if r.get("beats_best") is True and new:
            kept, since = True, 0
        elif r.get("beats_best") is True:
            kept = True                                              # 이미 센 sha 의 keep 은 연속 무 keep 을 되돌리지 않는다(정정 0-3 사소 5)
        elif new and kept:
            since += 1
    n = len(seen)
    return {"n": n, "since_keep": since, "stop": bool(n >= MAX_TRIALS or (kept and since >= MAX_SINCE_KEEP)), "shas": seen}


def _best_of(island: str) -> "dict | None":
    rows = [r for r in _read_jsonl(RUNS) if _isl(r) == island]
    hist = [h for h in _read_jsonl(BEST_HISTORY) if _isl(h) == island]
    ib = max((i for i, r in enumerate(rows) if r.get("beats_best") is True and "folds" in r), default=None)
    ia = max((i for i, r in enumerate(rows) if r.get("accept")), default=None)
    if hist and (ib is None or (ia is not None and ia > ib)):
        return hist[-1]
    if ib is not None:
        return rows[ib]
    return hist[-1] if hist else None


def best_record(island: str = "main") -> "dict | None":
    """그 섬의 BEST 바 = 그 섬의 마지막 keep. 섬 안에서 마지막 beats_best 평가 행이 마지막 --accept 행보다 뒤면 그 평가 행(accept 안 된 keep —
    상한 규칙과 같은 keep 정의), 아니면 그 섬의 BEST_HISTORY(작업 트리·사본 중 긴 쪽) 마지막 행. 섬에 keep 이 없으면 기준선(섬 main)의 BEST
    — 모든 섬이 같은 기준선에서 출발한다. 그것도 없으면 None(첫 유효 실행)."""
    b = _best_of(island)
    if b is None and island != "main":
        b = _best_of("main")
    return b


def _fold_id(f: dict) -> tuple:
    return f.get("era"), [str(x) for x in (f.get("test") or [])]


def beats_c(s: dict, folds: list, conc: dict, island: str = "main") -> "tuple[bool, str]":
    """keep: 체결이 바뀐 폴드 ≥ 1, 그중 alpha 개선 과반, Δscore ≥ 0.005, 쏠림 셋 각각 ≥ (그 섬) BEST 의 같은 값 − 0.01."""
    if s["void"]:
        return False, "void"
    b = best_record(island)
    if b is None:
        return True, "첫 유효 실행"
    if len(folds) != len(b["folds"]) or any(_fold_id(f) != _fold_id(g) for f, g in zip(folds, b["folds"])):
        return False, "fold mismatch"
    changed = [(f, g) for f, g in zip(folds, b["folds"]) if f["trade_key"] != g["trade_key"]]
    better = sum(f["alpha"] > g["alpha"] for f, g in changed)
    dconc = {k: conc[k] - b["conc"][k] for k in CONC_KEYS}
    ok = (len(changed) >= 1 and better * 2 > len(changed) and s["score"] >= b["score"] + KEEP_DSCORE
          and all(conc[k] >= b["conc"][k] - KEEP_CONC_SLACK for k in CONC_KEYS))
    return bool(ok), (f"dscore {s['score'] - b['score']:+.4f} · better {better}/{len(changed)} 바뀐 폴드 · "
                      + " · ".join(f"d{k} {v:+.4f}" for k, v in dconc.items()))


def integrity(check_cache: bool = True, check_locked: bool = False) -> None:
    """태그·감시 파일·MANIFEST sha. ``check_locked`` = 잠금 절반 캐시 sha 까지(select.py 만 — 루프 평가마다 해시하지 않는다)."""
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    if git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}").returncode != 0:
        print(f"경고: 태그 {PREREG_TAG} 없음 — 사전등록 전 시험 실행", file=sys.stderr)
        return
    guarded = (*GUARDED_FILES, os.path.relpath(MANIFEST, ROOT))
    changed = git("log", "--format=%h %s", f"{PREREG_TAG}..HEAD", "--", *guarded).stdout.strip()
    dirty = git("status", "--porcelain", "--", *guarded).stdout.strip()
    if changed or dirty:
        raise SystemExit(f"평가기·엔진·데이터가 사전등록({PREREG_TAG}) 뒤 바뀌었다 — 평가 거부.\n{changed}\n{dirty}")
    man = json.load(open(MANIFEST))
    for p, h in man.items():
        if p.startswith("cache/"):
            if check_cache and sha_file(os.path.join(CACHE, p[6:])) != h:
                raise SystemExit(f"캐시 {p} sha 불일치 — 평가 거부")
        elif p.startswith("frozen/"):
            if check_cache and sha_file(os.path.join(FROZEN, p[7:])) != h:
                raise SystemExit(f"얼린 뉴스 파일 {p} sha 불일치 — 평가 거부")
        elif p.startswith("locked/"):
            if check_locked and sha_file(os.path.join(LOCKED_CACHE, p[7:])) != h:
                raise SystemExit(f"잠금 캐시 {p} sha 불일치 — 거부")
        elif p.startswith("ext/"):
            if sha_file(V4._ext_path(p[4:])) != h:
                raise SystemExit(f"{p} sha 불일치 — 평가 거부")
        elif sha_file(os.path.join(ROOT, p)) != h:
            raise SystemExit(f"{p} sha 불일치 — 평가 거부")


def write_manifest() -> None:
    missing = [p for p in GUARDED_FILES if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"감시 파일 없음 — 매니페스트 거부: {missing}")
    man = {p: sha_file(os.path.join(ROOT, p)) for p in GUARDED_FILES}
    man.update({f"cache/era_{e}_s0.pkl": sha_file(os.path.join(CACHE, f"era_{e}_s0.pkl")) for e in CACHE_ERAS})
    man.update({f"locked/era_{e}_s1.pkl": sha_file(os.path.join(LOCKED_CACHE, f"era_{e}_s1.pkl")) for e in CACHE_ERAS})
    man.update({f"frozen/{f}": sha_file(os.path.join(FROZEN, f)) for f in FROZEN_FILES})
    man.update({f"ext/{p}": sha_file(V4._ext_path(p)) for p in V4.EXT_GUARDED})
    json.dump(man, open(MANIFEST, "w"), indent=1)
    log(f"MANIFEST.json 기록 — {len(man)} 항목")


def _fmt(k: str, v) -> str:
    if isinstance(v, dict):
        v = f"changed {v['changed']:.4f} · dscore {v['dscore']:+.4f} · trades_ok {v.get('trades_ok')} · ok {v['ok']}"
    elif v is None:
        v = "void"
    return f"{k + ':':22s} {v:.6f}" if isinstance(v, float) else f"{k + ':':22s} {v}"


def _sandboxed(args: "list[str]", cur: str, stdin: str = "", island: str = "main"):
    """격리 실행 → 결과 dict. 실패·시간 초과(``SANDBOX_TIMEOUT_S``)면 crash 행을 원장에 남기고(시행으로 센다) stderr 끝 20줄만 보이고
    끝낸다(stdout 은 버린다)."""
    try:
        p = subprocess.run(sandbox_cmd_c(args), capture_output=True, text=True, input=stdin, timeout=SANDBOX_TIMEOUT_S)
        rc, err, line = p.returncode, p.stderr, [x for x in p.stdout.splitlines() if x.startswith("@@RESULT@@")]
    except subprocess.TimeoutExpired as e:
        err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc, line, err = "timeout", [], err + f"\n시간 초과({SANDBOX_TIMEOUT_S}s) — crash"
    if rc != 0 or not line:
        _crash_append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "crash": True, "returncode": rc,
                       **git_state(), "island": island, "evaluator": PREREG_TAG})
        print("---\nscore:                 crash\n" + "\n".join(err.splitlines()[-CRASH_TAIL:]))
        raise SystemExit(rc if isinstance(rc, int) and rc else 1)
    sys.stderr.write("\n".join(x for x in p.stderr.splitlines() if re.match(r"^\[\s*\d+s\] ", x) or x.startswith("경고")) + "\n")
    return json.loads(line[-1][len("@@RESULT@@"):])


def _crash_append(rec: dict) -> None:
    """crash·sha 불일치 행도 결과 행과 같은 자물쇠·상한 재검사 아래서 덧붙인다(정정 0-3 사소 3) — 실행 중 다른 섬이 상한을 채웠고 이 sha 가
    새것이면 기록하지 않는다(151 번째 crash 도 끼지 못한다)."""
    with ledger_lock():
        st = trial_state()
        if st["stop"] and rec["train_sha"] not in st["shas"]:
            print(_stop_msg(st) + " — 실행 중 다른 섬이 상한을 채웠다, crash 를 기록하지 않는다", file=sys.stderr)
            return
        _append(RUNS, rec)


def _stop_msg(st: dict) -> str:
    return f"상한 도달 — 평가 거부 (고유 시행 {st['n']}, 마지막 keep 뒤 {st['since_keep']})"


def evaluate(workers: int) -> dict:
    with ledger_lock():
        st = trial_state()                                            # 상한 검사가 맨 먼저(계산·무결성보다 앞)
    if st["stop"]:
        raise SystemExit(_stop_msg(st))
    cur = hashlib.sha256(open(TRAIN, "rb").read()).hexdigest()[:16]
    isl = island_of(cur)                                              # 섬이 아닌 브랜치는 기준선만
    check_island_path(isl)
    integrity()
    os.makedirs(SHADOW, exist_ok=True)                                # 있어야 격리에서 가린다
    t0 = time.time()
    pin_ref(cur, git_state())                                         # crash 여도 commit 을 붙잡는다
    r = _sandboxed(["--inner", "--workers", str(workers)], cur, stdin=f"{causality_seed(TRAIN)}\n", island=isl)
    s, folds, conc, abl = r["summary"], r["folds"], r["conc"], r["ablation"]
    if s["train_sha"] != cur:
        _crash_append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "crash": True, "why": "sha mismatch",
                       **git_state(), "island": isl, "evaluator": PREREG_TAG})
        raise SystemExit("격리 실행이 다른 train.py 를 평가했다 — 거부")
    with ledger_lock():                                               # 상한 재검사 + keep 판정 + 덧붙이기를 한 자물쇠 안에서
        st = trial_state()
        if st["stop"] and cur not in st["shas"]:
            raise SystemExit(_stop_msg(st) + " — 실행 중 다른 섬이 상한을 채웠다, 이 결과는 기록하지 않는다")
        s["beats_best"], s["beats_detail"] = beats_c(s, folds, conc, isl)
        s["island"] = isl
        _eval_append(s, folds, conc, abl, cur, isl, t0)
        st = trial_state()
    s["trials"], s["stop"] = st["n"], st["stop"]
    _report(s, folds, conc, abl)
    return s


def _eval_append(s: dict, folds: list, conc: dict, abl: dict, cur: str, isl: str, t0: float) -> None:
    s.update({f"{k}_removed": conc[k] for k in CONC_KEYS})
    s.update(ablation_scalp=abl["scalp"], ablation_daytrade=abl["daytrade"], eligible=abl["eligible"])
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, **git_state(), "island": isl, "score": s["score"],
           "raw_score": s["raw_score"],
           "void": s["void"], "beats_best": s["beats_best"], "lookahead": s["lookahead_check"], "conc": conc, "ablation": abl,
           "eligible": abl["eligible"], "oos_trades": s["oos_trades"], "oos_trades_per_year": s["oos_trades_per_year"],
           "payoff_ratio": s["payoff_ratio"], "ev_net_2x": s["ev_net_2x"], "ev_net_1x": s["ev_net_1x"], "news_folds_score": s["news_folds_score"],
           "folds": [{k: f[k] for k in ("era", "test", "params", "alpha", "trade_key", "n")} for f in folds],
           "evaluator": PREREG_TAG, "elapsed_s": round(time.time() - t0, 1)}
    _append(RUNS, rec)


def _report(s: dict, folds: list, conc: dict, abl: dict) -> None:
    json.dump({"summary": s, "folds": folds, "conc": conc, "ablation": abl}, open(LAST, "w"), ensure_ascii=False, indent=1, default=str)
    shown = ("score", "void", "beats_best", "beats_detail", "lookahead_check", "folds_pos", "oos_trades", "payoff_ratio", "ev_net_2x",
             "ev_net_1x", "top1pct_removed", "top_stock_removed", "top10d_removed", "news_folds_score", "ablation_scalp",
             "ablation_daytrade", "eligible", "island", "trials", "stop")
    print("---")                                                           # 폴드별 값은 숨긴다
    for k in shown:
        print(_fmt(k, s[k]))


def accept() -> None:
    """그 섬의 BEST 갱신 — 원장(RUNS)에서 그 섬의 마지막 평가 기록이 현재 train.py 의 beats_best=True 일 때만. BEST_<섬>.json·BEST_HISTORY
    (island 필드)는 사본에도. 읽고 쓰기는 원장 자물쇠 안에서."""
    integrity(check_cache=False)
    cur = hashlib.sha256(open(TRAIN, "rb").read()).hexdigest()[:16]
    isl = island_of(cur)
    check_island_path(isl)
    with ledger_lock():
        evals = [r for r in _read_jsonl(RUNS) if "folds" in r and _isl(r) == isl]
        if not evals or evals[-1]["train_sha"] != cur or not evals[-1]["beats_best"]:
            raise SystemExit(f"섬 {isl} 원장의 마지막 평가가 현재 train.py 의 beats_best=True 가 아니다 — 갱신 거부")
        r = evals[-1]
        best = {"score": r["score"], "conc": r["conc"], "train_sha": cur, "commit": r.get("commit"), "evaluator": r["evaluator"],
                "island": isl, "eligible": r["eligible"],
                "folds": [{k: f[k] for k in ("era", "test", "alpha", "trade_key")} for f in r["folds"]]}
        _write_both(best_path(isl), best)
        _append(RUNS, {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "accept": True, "island": isl, "score": r["score"]})
        _append(BEST_HISTORY, best)
    print(f"BEST 갱신(섬 {isl}): score {r['score']:.4f}")


def reference(workers: int) -> None:
    """leader_C 고정값 참고 — 한 번만, 기준선 train.py 로만. REFERENCE.json(+ 사본)에만 쓰고 RUNS 행은 쓰지 않는다(시행 아님)."""
    if any(os.path.exists(p) for p in (REFERENCE, os.path.join(SHADOW, os.path.basename(REFERENCE)))):
        raise SystemExit("REFERENCE.json 이 이미 있다 — 참고 실행은 한 번뿐")
    integrity()
    cur = hashlib.sha256(open(TRAIN, "rb").read()).hexdigest()[:16]
    if cur != BASELINE_TRAIN_SHA:
        raise SystemExit(f"train.py({cur}) 가 기준선({BASELINE_TRAIN_SHA})이 아니다 — 참고 실행은 기준선으로만")
    os.makedirs(SHADOW, exist_ok=True)
    p = subprocess.run(sandbox_cmd_c(["--inner-reference", "--workers", str(workers)]), capture_output=True, text=True)
    line = [x for x in p.stdout.splitlines() if x.startswith("@@RESULT@@")]
    if p.returncode != 0 or not line:
        print("\n".join(p.stderr.splitlines()[-CRASH_TAIL:]))
        raise SystemExit(p.returncode or 1)
    r = json.loads(line[-1][len("@@RESULT@@"):])
    _write_both(REFERENCE, {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, **git_state(), "params": REF_PARAMS,
                            "note": "leader_C 고정값(univ top200 은 표현 불가 — 평가기 uni)", "evaluator": PREREG_TAG, **r})
    print(f"REFERENCE 기록: {REFERENCE}")


def default_workers(branch: "str | None" = None) -> int:
    """섬 넷이 16코어를 나눠 쓴다 — 섬 브랜치면 max(1, (cpu − 1) // 4), 아니면 cpu − 1."""
    n = os.cpu_count() or 2
    return max(1, (n - 1) // 4) if ISLAND_RE.match(current_branch() if branch is None else branch) else max(1, n - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="캐시 A·B·C(data.build_all, KR_QUANT_DB)")
    ap.add_argument("--manifest", action="store_true", help="코드·캐시 sha 매니페스트(사전등록 때)")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--reference", action="store_true", help="leader_C 고정값 참고 한 번(시행 아님)")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--inner-reference", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--workers", type=int, default=None, help="기본: 섬 브랜치면 max(1,(cpu−1)//4), 아니면 cpu−1")
    a = ap.parse_args()
    if a.workers is None:
        a.workers = default_workers()
    _G["t0"] = time.time()
    if a.build:
        from research.combined.data import build_all
        build_all()
    elif a.manifest:
        write_manifest()
    elif a.accept:
        accept()
    elif a.reference:
        reference(a.workers)
    elif a.inner:
        inner(a.workers)
    elif a.inner_reference:
        inner_reference(a.workers)
    else:
        evaluate(a.workers)


if __name__ == "__main__":
    main()
