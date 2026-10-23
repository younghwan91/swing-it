"""autoresearch 고정 평가기 v2 — 진입점이자 심판. 에이전트는 이 파일을 고치지 않는다(program.md · PREREG.md).

karpathy/autoresearch 의 prepare.py 자리 + dietmarwo/autoresearch-trading 의 "구조는 LLM, 숫자는 최적화기" 분리.
``train.py`` 의 ``get_strategy()`` 가 후보·순위·파라미터 범위·엔진 설정(청산선은 메뉴)을 내면, 여기서 폴드마다 학습 구간에서
파라미터를 고르고(라틴 하이퍼큐브 48점 → 이웃 5점 중앙값 최대 = 고원) 시험 구간에서 한 번 잰다.

v2(검증 리뷰 C1~C4·M1~M8 반영, PREREG.md 정정 1):
- 진입점은 이 파일. train.py 는 AST 허용 목록(numpy·pandas·math 등, 파일·OS·동적 실행·더블 언더스코어·종목코드·연도 리터럴 금지)을 통과해야 로드된다.
- train.py 가 받는 V 에는 날짜·종목코드가 없다(사후 지식 차단). 배열은 복사본.
- 탐색 캐시는 탐색 절반만 디스크에 둔다. 잠금 절반은 ``--final`` 때 DB 에서 만들어 메모리에서만 쓴다.
- 미래정보 검사: 평가마다 절단일 3개에서 V[:, :cut+1] 로 새 모듈이 만든 신호와 전체 신호 앞부분을 비교, 다르면 void.
- 점수 = 시험 폴드 **노출을 맞춘 선택 알파**(전략 로그 성장 − 전날 노출 × KOSPI 일수익 로그 성장, 연율) 평균 − 0.5σ.
  쉬는 해·타이밍은 점수에 0 이다. KOSPI 보유 대비 초과수익은 보고 열·졸업 바 1.
- 청산선은 메뉴(MA w ∈ [20, 250], 어제까지 w일 최저 저가 w ∈ [20, 120]). 진입은 다음 날 시가만. 거래정지일 매도는 미룬다(엔진 paper 경로).
  보유는 청산 비용을 뺀 순가치로 잰다(엔진 net_mtm).
- keep 은 평가기가 판정한다: 직전 최고 대비 Δscore ≥ 0.005 ∧ 폴드 ≥ 6/10 개선 ∧ top5 제거 점수 악화 ≤ 0.01 → ``beats_best``.
  run.log 에는 요약만 나온다(폴드별 값은 숨긴다).
- 시행 원장 RUNS.jsonl(덧붙이기 — 작업 트리 + ~/.cache 사본), N = 고유 train sha 수. 잠금 사용 LOCKBOX.jsonl(같은 이중 기록).
- 무결성: 태그 뒤 평가기·엔진·의존 파일 변경 금지 + MANIFEST.json 의 sha256(코드·캐시) 대조.

사용(작업 트리 루트):
  uv run --extra pg python research/autoresearch/prepare.py --build      # 탐색 캐시(한 번, KR_QUANT_DB) → MANIFEST.json
  uv run python research/autoresearch/prepare.py > run.log 2>&1          # 실험 한 번
  uv run python research/autoresearch/prepare.py --accept                 # 직전 실행이 beats_best 면 BEST 갱신(keep 때)
  uv run --extra pg python research/autoresearch/prepare.py --final       # 졸업 심사 — 메인 세션만, 잠금 최대 3회
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")
CACHE = os.path.expanduser("~/.cache/cc-swing/scratch/autoresearch")
SHADOW = os.path.expanduser("~/.cache/cc-swing/autoresearch_ledger")      # 원장 사본(작업 트리를 지워도 남는다)
RUNS, LOCKBOX, BEST, LAST = (os.path.join(HERE, f) for f in ("RUNS.jsonl", "LOCKBOX.jsonl", "BEST.json", "last_run.json"))
MANIFEST = os.path.join(HERE, "MANIFEST.json")
PREREG_TAG = "autoresearch-prereg-2"
GUARDED_FILES = ("research/autoresearch/prepare.py", "research/experiments/leader21.py", "research/experiments/indep30.py",
                 "src/swing_it/features/eps_asof.py", "src/swing_it/features/korea.py", "src/swing_it/storage.py")
LOCKBOX_MAX = 3
COST_MULT = 2.0
SEED = 20261024
N_EVAL = 48
K_PLATEAU = 5
MIN_TRAIN_TRADES_PER_YEAR = 8
MIN_FOLD_TRADES = 5
MAX_THIN_FOLDS = 2
MIN_OOS_TRADES = 150
MIN_OOS_TRADES_PER_YEAR = 20
DISPERSION = 0.5
KEEP_DSCORE, KEEP_FOLDS, KEEP_TOP5_SLACK = 0.005, 6, 0.01
N_NULL = 100
ERAS = {
    "A": {"data": ("2008-01-01", "2016-08-31"), "eps": "earnings_dart_doc",
          "folds": [(f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(2011, 2016)] + [("2014-01-01", "2016-01-01", "2016-09-01")]},
    "B": {"data": ("2016-09-01", "2023-12-31"), "eps": "earnings",
          "folds": [(f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(2020, 2024)]},
}
CFG_BOUNDS = {"max_pos": (5, 20), "hard": (0.10, 0.30), "risk": (0.002, 0.01), "pos_cap": (0.05, 0.20)}
CFG_CHOICES = {"pyramid": (0, 1), "regime": (0, 1), "exit_kind": ("ma", "low")}
EXIT_W = {"ma": (20, 250), "low": (20, 120)}
CFG_DEFAULT = {"max_pos": 10, "hard": 0.15, "risk": 0.005, "pos_cap": 0.10, "pyramid": 1, "regime": 0, "exit_kind": "ma", "exit_w": 100}
VIEW_KEYS = ("O", "H", "L", "C", "tv", "adv", "cap", "eps_yoy", "uni", "TR")
_G: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - _G.get('t0', time.time()):6.0f}s] {msg}", file=sys.stderr, flush=True)


def half_of(code: str) -> int:
    """종목 절반 잠금 — 코드 sha256 첫 바이트 짝홀. 0 = 탐색, 1 = 잠금."""
    return hashlib.sha256(str(code).encode()).digest()[0] & 1


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- train.py 검사(C1·C3)

ALLOWED_IMPORTS = {"numpy", "pandas", "math", "functools", "itertools", "collections", "dataclasses", "typing", "__future__"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr", "input",
                "breakpoint", "memoryview", "help", "exit", "quit"}
BANNED_ATTR = re.compile(r"^(load|loads|save|savez|savez_compressed|savetxt|loadtxt|genfromtxt|fromfile|tofile|memmap|fromregex|DataSource|"
                         r"ctypeslib|f2py|testing|lib|os|sys|system|eval|query|setflags|read_\w+|to_(pickle|csv|parquet|sql|json|excel|hdf|feather|"
                         r"clipboard|html|latex|markdown|xml|stata|orc)|year|month|day|dayofweek|weekday|quarter|date|dates|days|codes|code)$")
YEAR_LIKE = range(1990, 2031)


def lint_train(src: str) -> "list[str]":
    """train.py 허용 목록 — 위반 목록(비면 통과). 완전한 샌드박스는 아니다: 우회 시도는 program.md 위반 = 결과 무효."""
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"line {node.lineno}: import {m}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            bad.append(f"line {node.lineno}: 이름 {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("__") or BANNED_ATTR.match(node.attr)):
            bad.append(f"line {node.lineno}: 속성 .{node.attr}")
        elif isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, str) and (re.search(r"\d{6}", v) or re.search(r"(19|20)\d{2}", v)):
                bad.append(f"line {node.lineno}: 종목코드·날짜 같은 문자열 {v!r}")
            if isinstance(v, int) and not isinstance(v, bool) and v in YEAR_LIKE:
                bad.append(f"line {node.lineno}: 연도 같은 정수 {v}")
    return bad


def load_strategy(path: str = TRAIN) -> dict:
    """train.py 를 **새 모듈 인스턴스**로 로드(캐시 공유 없음). 허용 목록 위반이면 거부."""
    src = open(path, encoding="utf-8").read()
    bad = lint_train(src)
    if bad:
        raise SystemExit("train.py 허용 목록 위반 — 평가 거부:\n  " + "\n  ".join(bad))
    spec = importlib.util.spec_from_file_location(f"ar_train_{time.time_ns()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    S = mod.get_strategy()
    for k in ("params", "signals", "cfg"):
        if k not in S:
            raise SystemExit(f"get_strategy() 에 {k!r} 가 없다")
    S["sha"] = hashlib.sha256(src.encode()).hexdigest()[:16]
    return S


# ---------------------------------------------------------------- 캐시

def build_era(con, era: str) -> dict:
    import research.experiments.indep30 as I30
    from swing_it.features.eps_asof import eps_ok, eps_yoy
    from swing_it.features.korea import cost_rt
    from swing_it.storage import read_index
    E = ERAS[era]
    D = I30.build(con, E["data"], E["eps"])
    days, codes = D["days"], D["codes"]
    D["lu"] = np.where(days < pd.Timestamp("2015-06-15"), 1.14, 1.29)           # 정정 30-1 M1
    tax = np.array([cost_rt(d, "KOSPI") for d in days])                         # 매도일 세율(KOSPI·KOSDAQ 같은 값 — 확인)
    D["cost"], D["mkt"] = np.vstack([tax, tax]), np.zeros(len(codes), np.int64)
    ci, s = np.nonzero(D["uni"] & D["TR"])
    up, yoy = np.zeros(D["C"].shape, bool), np.full(D["C"].shape, np.nan, np.float32)
    up[ci, s] = eps_ok(codes, days, ci, s, con, source=E["eps"])
    yoy[ci, s] = eps_yoy(codes, days, ci, s, con, source=E["eps"])
    kq = read_index(con, "101", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,
                    proxy="autoresearch: 공식 KOSDAQ(팩터 회귀용)").set_index("date")["close"]
    keep = ("days", "codes", "O", "H", "L", "C", "TR", "cost", "mkt", "hi20", "adv", "uni", "cap", "tv", "lu", "regime_on", "kx", "diag")
    out = {k: D[k] for k in keep}
    out.update(eps_up=up, eps_yoy=yoy, half=np.array([half_of(c) for c in codes], np.int8), era=era,
               kq=pd.Series(kq).reindex(days).ffill().to_numpy(float))
    out["kx"] = pd.Series(out["kx"]).reindex(days).ffill().to_numpy(float)
    for k in ("adv", "cap", "tv"):
        out[k] = np.asarray(out[k], np.float32)
    out["diag"] = {**D["diag"], "eps_up_share": float(up[ci, s].mean())}
    return out


def slice_half(D: dict, half: int) -> dict:
    sel = np.flatnonzero(D["half"] == half)
    n = len(D["codes"])
    out = {}
    for k, v in D.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == n:
            out[k] = v[sel]
        elif k in ("mkt", "half"):
            out[k] = v[sel]
        else:
            out[k] = v
    out["codes"] = [D["codes"][i] for i in sel]
    out["half_id"] = half
    return out


def build() -> None:
    import psycopg2
    os.makedirs(CACHE, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    for era in ERAS:
        D = slice_half(build_era(con, era), 0)                                  # 탐색 절반만 디스크에(C1)
        with open(os.path.join(CACHE, f"era_{era}_s0.pkl"), "wb") as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"era {era}: 탐색 절반 {len(D['codes'])} codes × {len(D['days'])} days")
    man = {p: sha_file(os.path.join(ROOT, p)) for p in GUARDED_FILES}
    man.update({f"cache/era_{e}_s0.pkl": sha_file(os.path.join(CACHE, f"era_{e}_s0.pkl")) for e in ERAS})
    json.dump(man, open(MANIFEST, "w"), indent=1)
    log(f"MANIFEST.json 기록 — {len(man)} 항목")


def load_search(era: str) -> dict:
    with open(os.path.join(CACHE, f"era_{era}_s0.pkl"), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- 무결성(C1·M1)

def integrity(check_cache: bool = True) -> None:
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    if git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}").returncode != 0:
        print(f"경고: 태그 {PREREG_TAG} 없음 — 사전등록 전 시험 실행", file=sys.stderr)
        return
    guarded = (*GUARDED_FILES, "research/autoresearch/MANIFEST.json")
    changed = git("log", "--format=%h %s", f"{PREREG_TAG}..HEAD", "--", *guarded).stdout.strip()
    dirty = git("status", "--porcelain", "--", *guarded).stdout.strip()
    if changed or dirty:
        raise SystemExit(f"평가기·엔진이 사전등록({PREREG_TAG}) 뒤 바뀌었다 — 평가 거부.\n{changed}\n{dirty}")
    man = json.load(open(MANIFEST))
    for p, h in man.items():
        if p.startswith("cache/"):
            if check_cache and sha_file(os.path.join(CACHE, p[6:])) != h:
                raise SystemExit(f"캐시 {p} sha 불일치 — 평가 거부")
        elif sha_file(os.path.join(ROOT, p)) != h:
            raise SystemExit(f"{p} sha 불일치 — 평가 거부")


def _append(path: str, rec: dict) -> None:
    os.makedirs(SHADOW, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
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


def n_trials() -> int:
    return len({r["train_sha"] for r in _read_jsonl(RUNS)})


# ---------------------------------------------------------------- 시뮬레이션

def view(D: dict) -> dict:
    """train.py 에 넘기는 묶음 — 복사본, 날짜·종목코드 없음(C3·M7)."""
    V = {k: np.array(D[k], copy=True) for k in VIEW_KEYS}
    V.update(kx=np.array(D["kx"], copy=True), kq=np.array(D["kq"], copy=True), regime_on=np.array(D["regime_on"], copy=True))
    return V


def clamp_cfg(cfg: dict) -> dict:
    out = dict(CFG_DEFAULT)
    for k, (lo, hi) in CFG_BOUNDS.items():
        if k in cfg:
            out[k] = float(np.clip(cfg[k], lo, hi)) if k != "max_pos" else int(np.clip(round(cfg[k]), lo, hi))
    for k, ch in CFG_CHOICES.items():
        if k in cfg:
            if cfg[k] not in ch:
                raise ValueError(f"cfg[{k!r}]={cfg[k]!r} 는 허용 밖: {ch}")
            out[k] = cfg[k]
    lo, hi = EXIT_W[out["exit_kind"]]
    out["exit_w"] = int(np.clip(round(cfg.get("exit_w", out["exit_w"])), lo, hi))
    return out


_EXIT: dict = {}


def exit_line(D: dict, kind: str, w: int) -> np.ndarray:
    """청산선 메뉴(M2) — 종가 < 선이면 다음 날 시가 청산. ma = w일 종가 평균, low = 어제까지 w일 최저 저가(구조선)."""
    key = (D["era"], D.get("half_id", 0), kind, w, D["C"].shape)
    if key not in _EXIT:
        if kind == "ma":
            _EXIT[key] = pd.DataFrame(D["C"].T).rolling(w, min_periods=w).mean().to_numpy().T
        else:
            _EXIT[key] = pd.DataFrame(D["L"].T).shift(1).rolling(w, min_periods=w).min().to_numpy().T
    return _EXIT[key]


def simulate(D: dict, sig: dict, cfg: dict, start: str, end: str, exclude=(), cost_mult: float = COST_MULT):
    import research.experiments.leader21 as L
    cand = np.asarray(sig["cand"], bool)
    if cand.shape != D["C"].shape:
        raise ValueError("cand·score 는 (종목, 날짜) 모양이어야 한다")
    cand = cand & D["uni"] & D["TR"] & D["eps_up"]                                          # 가드: EPS 상승·유동성·거래 가능
    score = np.where(cand, np.asarray(sig["score"], float), np.nan)
    c = clamp_cfg(cfg)
    D2 = {**D, "cand": cand, "rs": score, "brk": {0: np.ones_like(cand)}, "ma": {"x": exit_line(D, c["exit_kind"], c["exit_w"])}}
    ecfg = {"theta": -np.inf, "N": 0, "ma_exit": "x", "hard": c["hard"], "pyramid": c["pyramid"], "regime": c["regime"],
            "max_pos": c["max_pos"], "risk": c["risk"], "pos_cap": c["pos_cap"], "entry": "open", "exclude": list(exclude)}
    with np.errstate(invalid="ignore"):
        return L.simulate(D2, ecfg, cost_mult, end=end, start=start, paper=True, net_mtm=True)   # 정지일 매도 미룸(M3) · 순가치(M4)


def fold_stats(D: dict, eq: pd.Series, lo: str, hi: str) -> dict:
    """[lo, hi) — 노출 맞춘 선택 알파(점수)와 KOSPI 보유 대비 초과(보고) 연율 로그, 월 수익(팩터·DSR)."""
    ix = (eq.index >= lo) & (eq.index < hi)
    x = eq[ix]
    if len(x) < 2:
        return {"alpha": float("nan"), "excess": float("nan"), "years": 0.0, "m": pd.DataFrame()}
    di = D["days"].get_indexer(x.index)
    k, q = D["kx"][di], D["kq"][di]
    expo = np.asarray(eq.attrs["expo"], float)[ix]
    rk = k[1:] / k[:-1] - 1
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
    ls = float(np.log(x.iloc[-1] / x.iloc[0]))
    lm = float(np.log1p(expo[:-1] * rk).sum())
    m = pd.DataFrame({"s": x.to_numpy(), "k": k, "q": q}, index=x.index).resample("ME").last().pct_change().dropna()
    em = pd.Series(np.cumprod(np.r_[1.0, 1 + expo[:-1] * rk]), index=x.index).resample("ME").last().pct_change().dropna()
    m["sel"] = m["s"] - em.reindex(m.index).fillna(0.0)
    return {"alpha": (ls - lm) / yrs, "excess": (ls - float(np.log(k[-1] / k[0]))) / yrs, "years": yrs, "m": m}


def _train_point(args):
    fi, j, x = args
    D, S, lo, hi = _G["D"], _G["S"], _G["folds"][fi][0], _G["folds"][fi][1]
    try:
        eq, tr = simulate(D, S["signals"](_G["V"], x), S["cfg"](x), lo, hi)
        st = fold_stats(D, eq, lo, hi)
        n = int((tr["entry"] >= lo).sum()) if len(tr) else 0
        return fi, j, (st["alpha"] if n >= MIN_TRAIN_TRADES_PER_YEAR * st["years"] else -1.0)
    except Exception as e:  # noqa: BLE001 — 한 점 실패는 그 점만 버린다
        log(f"학습 점 fold {fi} #{j} 실패: {e!r}")
        return fi, j, -1.0


def sample_params(spec: dict, n: int, seed: int) -> "tuple[list[dict], np.ndarray]":
    names = list(spec)
    if not names:
        return [{}], np.zeros((1, 0))
    rng = np.random.default_rng(seed)                                # 라틴 하이퍼큐브
    U = (np.argsort(rng.random((len(names), n)), axis=1).T + rng.random((n, len(names)))) / n
    pts = []
    for u in U:
        x = {}
        for k, name in enumerate(names):
            s = spec[name]
            if s[0] == "choice":
                x[name] = s[1][min(int(u[k] * len(s[1])), len(s[1]) - 1)]
            else:
                v = s[1] + u[k] * (s[2] - s[1])
                x[name] = int(round(v)) if s[0] == "int" else float(v)
        pts.append(x)
    return pts, U


def plateau_pick(U: np.ndarray, y: np.ndarray, k: int = K_PLATEAU) -> int:
    """이웃 k 개(자기 포함) **중앙값**이 가장 높은 점 — 고원. 평균은 외딴 극값 하나가 이웃 손실을 덮었다(테스트)."""
    if U.shape[1] == 0 or len(y) <= k:
        return int(np.argmax(y))
    d = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(d, axis=1)[:, :k]
    return int(np.argmax(np.median(y[nb], axis=1)))


def walk_forward(D_by_era: dict, workers: int, train_path: str = TRAIN, fixed: "list | None" = None, null: "tuple | None" = None,
                 cost_mult: float = COST_MULT) -> dict:
    """시대마다 새 모듈로 train.py 로드 → 폴드별 학습 구간 고원 선택 → 시험 한 번. ``fixed`` = 그 폴드 파라미터 그대로.
    ``null`` = ("rank"|"thin"|"params", seed) — 졸업 심사 널."""
    res = {"folds": [], "trades": [], "months": []}
    for era, E in ERAS.items():
        D = D_by_era[era]
        S = load_strategy(train_path)
        _G.update(D=D, S=S, V=view(D), folds=E["folds"])
        rng = np.random.default_rng((null[1] if null else 0) * 7919 + (0 if era == "A" else 1))
        if fixed is not None:
            chosen = [f["params"] for f in fixed if f["era"] == era]
            if null and null[0] == "params":
                chosen = [sample_params(S["params"], 1, int(rng.integers(1 << 30)))[0][0] for _ in chosen]
        else:
            tasks, pts_all = [], {}
            for fi in range(len(E["folds"])):
                pts, U = sample_params(S["params"], N_EVAL, SEED + 101 * fi + (0 if era == "A" else 7))
                pts_all[fi] = (pts, U)
                tasks += [(fi, j, x) for j, x in enumerate(pts)]
            S["signals"](_G["V"], tasks[0][2])                       # 부모에서 train.py 캐시를 데운다(fork 공유)
            with mp.get_context("fork").Pool(workers) as pool:
                out = pool.map(_train_point, tasks, chunksize=1)
            chosen = []
            for fi in range(len(E["folds"])):
                pts, U = pts_all[fi]
                y = np.array([v for f, _, v in sorted(out, key=lambda r: (r[0], r[1])) if f == fi])
                chosen.append(pts[plateau_pick(U, y)])
        for fi, (_, lo, hi) in enumerate(E["folds"]):
            x = chosen[fi]
            sig = S["signals"](_G["V"], x)
            if null and null[0] == "rank":
                sig = {**sig, "score": rng.random(D["C"].shape)}
            elif null and null[0] == "thin":
                base = D["uni"] & D["TR"] & D["eps_up"]
                c = np.asarray(sig["cand"], bool) & base
                p = c.sum() / max(base.sum(), 1)
                sig = {"cand": base & (rng.random(D["C"].shape) < p), "score": rng.random(D["C"].shape)}
            eq, tr = simulate(D, sig, S["cfg"](x), lo, hi, cost_mult=cost_mult)
            tr = tr[(tr["entry"] >= lo) & (tr["entry"] < hi)] if len(tr) else tr
            st = fold_stats(D, eq, lo, hi)
            res["folds"].append({"era": era, "test": [lo, hi], "params": x, "alpha": st["alpha"], "excess": st["excess"],
                                 "years": st["years"], "n": int(len(tr)), "half": D.get("half_id", 0)})
            res["months"].append(st["m"])
            if len(tr):
                res["trades"].append(tr.assign(era=era, fold=fi))
    return res


def summarize(res: dict) -> dict:
    al = np.array([f["alpha"] for f in res["folds"]])
    ex = np.array([f["excess"] for f in res["folds"]])
    n = np.array([f["n"] for f in res["folds"]])
    tr = pd.concat(res["trades"]) if res["trades"] else pd.DataFrame(columns=["ret", "code"])
    yrs = sum(f["years"] for f in res["folds"])
    void = ((n < MIN_FOLD_TRADES).sum() > MAX_THIN_FOLDS or n.sum() < MIN_OOS_TRADES or n.sum() / max(yrs, 1e-9) < MIN_OOS_TRADES_PER_YEAR
            or not np.isfinite(al).all())
    win, loss = tr.loc[tr["ret"] > 0, "ret"], tr.loc[tr["ret"] <= 0, "ret"]
    era = np.array([f["era"] for f in res["folds"]])
    return {"score": float(al.mean() - DISPERSION * al.std(ddof=1)) if not void else -1.0, "void": bool(void),
            "alpha_mean": float(np.nanmean(al)), "alpha_std": float(np.nanstd(al, ddof=1)), "folds_pos": int((al > 0).sum()),
            "excess_vs_kospi_mean": float(np.nanmean(ex)),
            "eraA_alpha": float(np.nanmean(al[era == "A"])) if (era == "A").any() else float("nan"),
            "eraB_alpha": float(np.nanmean(al[era == "B"])) if (era == "B").any() else float("nan"),
            "oos_trades": int(n.sum()), "oos_trades_per_year": float(n.sum() / max(yrs, 1e-9)), "thin_folds": int((n < MIN_FOLD_TRADES).sum()),
            "trade_ret_mean": float(tr["ret"].mean()) if len(tr) else float("nan"),
            "payoff_ratio": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
            "win_rate": float((tr["ret"] > 0).mean()) if len(tr) else float("nan")}


def top_removed_score(D_by_era: dict, res: dict, k: int = 5, train_path: str = TRAIN) -> float:
    if not res["trades"]:
        return float("nan")
    top = pd.concat(res["trades"]).groupby("code")["ret"].sum().sort_values(ascending=False).index[:k].tolist()
    al = []
    for era in ERAS:
        D = D_by_era[era]
        S = load_strategy(train_path)
        V = view(D)
        for f in [f for f in res["folds"] if f["era"] == era]:
            lo, hi = f["test"]
            eq, _ = simulate(D, S["signals"](V, f["params"]), S["cfg"](f["params"]), lo, hi, exclude=top)
            al.append(fold_stats(D, eq, lo, hi)["alpha"])
    al = np.array(al)
    return float(al.mean() - DISPERSION * al.std(ddof=1))


def _cut(D: dict, cut: int) -> dict:
    T = D["C"].shape[1]
    out = {}
    for k in VIEW_KEYS + ("kx", "kq", "regime_on"):
        v = D[k]
        out[k] = v[:, :cut + 1] if v.ndim == 2 else v[:cut + 1] if len(v) == T else v
    return out


def causality(D_by_era: dict, res: dict, train_path: str = TRAIN) -> "tuple[bool, str]":
    """미래정보 검사(C2): 절단일 3개에서 V[:, :cut+1] 로 새 모듈이 만든 신호 == 전체 신호 앞부분."""
    for era in ERAS:
        D = D_by_era[era]
        T = D["C"].shape[1]
        x = [f for f in res["folds"] if f["era"] == era][-1]["params"]
        full = load_strategy(train_path)["signals"](view(D), x)
        for frac in (0.45, 0.7, 0.92):
            cut = int(T * frac)
            part = load_strategy(train_path)["signals"](view(_cut(D, cut)), x)
            a, b = np.asarray(full["cand"], bool)[:, :cut + 1], np.asarray(part["cand"], bool)
            if a.shape != b.shape or (a != b).mean() > 1e-6:
                return False, f"era {era} cut {cut}: cand 불일치"
            sa = np.asarray(full["score"], float)[:, :cut + 1][a]
            sb = np.asarray(part["score"], float)[a]
            if not np.allclose(sa, sb, rtol=1e-7, atol=1e-12, equal_nan=True):
                return False, f"era {era} cut {cut}: score 불일치"
    return True, "ok"


def beats(s: dict, folds: list, top5: float) -> "tuple[bool, str]":
    if s["void"]:
        return False, "void"
    if not os.path.exists(BEST):
        return True, "첫 유효 실행"
    b = json.load(open(BEST))
    better = sum(f["alpha"] > g["alpha"] for f, g in zip(folds, b["folds"], strict=True))
    ok = s["score"] >= b["score"] + KEEP_DSCORE and better >= KEEP_FOLDS and top5 >= b["top5"] - KEEP_TOP5_SLACK
    return bool(ok), f"dscore {s['score'] - b['score']:+.4f} · better_folds {better}/10 · dtop5 {top5 - b['top5']:+.4f}"


def evaluate(workers: int) -> dict:
    integrity()
    _G["t0"] = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    D_by_era = {e: load_search(e) for e in ERAS}
    S = load_strategy()
    res = walk_forward(D_by_era, workers)
    s = summarize(res)
    ok, why = causality(D_by_era, res)
    if not ok:
        s.update(void=True, score=-1.0)
    s["lookahead_check"] = why
    s["top5_removed_score"] = top_removed_score(D_by_era, res) if not s["void"] else float("nan")
    s["beats_best"], s["beats_detail"] = beats(s, res["folds"], s["top5_removed_score"])
    seen = {r["train_sha"] for r in _read_jsonl(RUNS)}
    s["n_trials"] = len(seen | {S["sha"]})
    s["train_sha"] = S["sha"]
    s["elapsed_s"] = round(time.time() - _G["t0"], 1)
    _append(RUNS, {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": S["sha"], "score": s["score"], "void": s["void"],
                   "beats_best": s["beats_best"], "lookahead": why})
    json.dump({"summary": s, "folds": res["folds"]}, open(LAST, "w"), ensure_ascii=False, indent=1, default=str)
    shown = ("score", "void", "beats_best", "beats_detail", "lookahead_check", "folds_pos", "oos_trades", "oos_trades_per_year", "thin_folds",
             "top5_removed_score", "payoff_ratio", "trade_ret_mean", "n_trials", "train_sha", "elapsed_s")   # 폴드별 값은 숨긴다(M8)
    print("---")
    for k in shown:
        v = s[k]
        print(f"{k + ':':22s} {v:.6f}" if isinstance(v, float) else f"{k + ':':22s} {v}")
    return s


def accept() -> None:
    integrity(check_cache=False)
    last = json.load(open(LAST))
    cur = load_strategy()["sha"]
    if last["summary"]["train_sha"] != cur or not last["summary"]["beats_best"]:
        raise SystemExit("직전 실행이 현재 train.py 의 beats_best=True 가 아니다 — 갱신 거부")
    json.dump({"score": last["summary"]["score"], "top5": last["summary"]["top5_removed_score"], "train_sha": cur,
               "folds": [{"era": f["era"], "test": f["test"], "alpha": f["alpha"]} for f in last["folds"]]}, open(BEST, "w"), indent=1)
    _append(RUNS, {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "accept": True, "score": last["summary"]["score"]})
    print(f"BEST 갱신: score {last['summary']['score']:.4f}")


# ---------------------------------------------------------------- 졸업 심사(--final, 메인 세션만)

def factor_t(months: list) -> "tuple[float, float]":
    """월 전략 수익을 KOSPI·KOSDAQ 월 수익에 회귀한 절편과 NW(lag 3) t."""
    m = pd.concat([x for x in months if len(x)])
    y, X = m["s"].to_numpy(), np.column_stack([np.ones(len(m)), m["k"].to_numpy(), m["q"].to_numpy()])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtX_inv = np.linalg.inv(X.T @ X)
    Sg = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, 4):
        G = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        Sg += (1 - lag / 4) * (G + G.T)
    V = XtX_inv @ Sg @ XtX_inv
    return float(b[0]), float(b[0] / np.sqrt(V[0, 0]))


def final(workers: int) -> None:
    import psycopg2
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
    integrity()
    used = len(_read_jsonl(LOCKBOX))
    if used >= LOCKBOX_MAX:
        raise SystemExit(f"잠금 절반은 이미 {used}회 열렸다(최대 {LOCKBOX_MAX}) — PREREG.md")
    if n_trials() == 0:
        raise SystemExit("RUNS.jsonl 이 비었다 — 루프를 돈 작업 트리에서 실행해야 시행 수 N 이 맞다")
    _G["t0"] = time.time()
    S = load_strategy()
    Ds = {e: load_search(e) for e in ERAS}
    search = walk_forward(Ds, workers)
    s0 = summarize(search)
    top5_s = top_removed_score(Ds, search)
    c1 = summarize(walk_forward(Ds, workers, fixed=search["folds"], cost_mult=1.0))
    nulls = {kind: [summarize(walk_forward(Ds, workers, fixed=search["folds"], null=(kind, r + 1)))["score"] for r in range(N_NULL)]
             for kind in ("rank", "thin", "params")}
    log("널 완료")
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    Dl = {e: slice_half(build_era(con, e), 1) for e in ERAS}                   # 잠금 절반 — 메모리에서만
    lock = walk_forward(Dl, workers)
    s1 = summarize(lock)
    top5_l = top_removed_score(Dl, lock)
    ms = pd.concat([m for m in search["months"] if len(m)])
    N = n_trials()
    dsr = deflated_sharpe_from_sample(ms["sel"].to_numpy(), N)
    a_s, t_s = factor_t(search["months"])
    a_l, t_l = factor_t(lock["months"])
    bars = {"b1_search_positive": s0["score"] > 0 and s0["eraA_alpha"] > 0 and s0["eraB_alpha"] > 0 and s0["excess_vs_kospi_mean"] > 0,
            "b2_beats_nulls_p95": all(s0["score"] > float(np.quantile(v, 0.95)) for v in nulls.values()),
            "b3_top5_removed_positive": top5_s > 0,
            "b4_lockbox": s1["score"] > 0 and s1["folds_pos"] >= 6 and not s1["void"],
            "b5_dsr_ge_095": float(dsr.get("prob_deflated_sharpe") or 0) >= 0.95,
            "b6_trades": s0["oos_trades_per_year"] >= 20,
            "b7_factor_t": t_s > 2.0}
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": S["sha"], "n_trials": N, "search": s0, "search_top5_removed": top5_s,
           "search_cost1x": c1, "nulls_p95": {k: float(np.quantile(v, 0.95)) for k, v in nulls.items()},
           "nulls_median": {k: float(np.median(v)) for k, v in nulls.items()}, "lockbox": s1, "lockbox_top5_removed": top5_l,
           "factor": {"search_alpha_m": a_s, "search_t_nw": t_s, "lockbox_alpha_m": a_l, "lockbox_t_nw": t_l}, "dsr_monthly_sel": dsr,
           "bars": bars, "pass": all(bars.values()), "folds_search": search["folds"], "folds_lockbox": lock["folds"]}
    _append(LOCKBOX, out)                                                        # 잠금 사용의 증인(덧붙이기)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    _G["t0"] = time.time()
    if a.build:
        build()
    elif a.final:
        final(a.workers)
    elif a.accept:
        accept()
    else:
        evaluate(a.workers)


if __name__ == "__main__":
    main()
