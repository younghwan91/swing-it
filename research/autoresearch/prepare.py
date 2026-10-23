"""autoresearch 고정 평가기 — 에이전트는 이 파일을 고치지 않는다(program.md · PREREG.md).

karpathy/autoresearch 의 prepare.py 자리 + dietmarwo/autoresearch-trading 의 "구조는 LLM, 숫자는 최적화기" 분리.
``train.py`` 의 ``get_strategy()`` 가 후보·순위·청산선·파라미터 범위를 내면, 여기서 폴드마다 학습 구간에서 파라미터를 고르고
(라틴 하이퍼큐브 48점 → 이웃 중앙값이 가장 높은 점 = 고원) 시험 구간에서 한 번 잰다. 체결·비용·상한가·손절 체결은 leader21 엔진 그대로다.

가드(코드로 강제 — 부탁 아님):
- 데이터: 2008-01 ~ 2016-08(시대 A, 실적 earnings_dart_doc) · 2016-09 ~ 2023-12(시대 B, 실적 earnings). 2024~ 는 읽지 않는다.
- 종목 절반 잠금: 코드 해시로 반을 잠근다. train.py 는 탐색 절반의 배열만 받는다. 잠금 절반은 ``--final`` 에서만, 최대 3회.
- 후보는 늘 ``uni(20일 거래대금 ≥ 10억) ∧ 거래 가능 ∧ EPS 상승(as-of)`` 와 AND — train.py 가 끌 수 없다.
- 엔진 설정은 허용 키·범위로 자른다: 하드 손절 ≥ 10%(좁은 손절 금지), 시간 손절 없음(엔진에 없다), 구조 손절·외부 순위 키 금지.
- 비용: 매도일 세율 + 수수료 + 스프레드(`korea.cost_rt`) × 2배. 상한가 배수 2015-06-15 전 1.14(정정 30-1 M1), 뒤 1.29.

사용:
  uv run --extra pg python research/autoresearch/prepare.py --build      # 캐시 만들기(한 번, KR_QUANT_DB)
  uv run python research/autoresearch/train.py > run.log 2>&1            # 실험 한 번(train.py 가 evaluate() 를 부른다)
  uv run python research/autoresearch/prepare.py --final                  # 졸업 심사(잠금 절반 · 널 · DSR) — 사람만, 최대 3회
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

CACHE = os.path.expanduser(os.environ.get("AR_CACHE", "~/.cache/cc-swing/scratch/autoresearch"))
HERE = os.path.dirname(os.path.abspath(__file__))
LOCKBOX_LOG = os.path.join(HERE, "LOCKBOX.jsonl")
LOCKBOX_MAX = 3
PREREG_TAG = "autoresearch-prereg-1"
COST_MULT = 2.0
SEED = 20261024
N_EVAL = 48                     # 폴드당 학습 구간 탐색 점 수(실험 시간 예산을 고정한다)
K_PLATEAU = 5                   # 고원 = 이웃 K 개 중앙값
MIN_TRAIN_TRADES_PER_YEAR = 8
MIN_FOLD_TRADES = 5
MAX_THIN_FOLDS = 2              # 체결 < 5 인 폴드(국면으로 쉰 해 등)는 2개까지 — 그보다 많으면 void
MIN_OOS_TRADES = 150
MIN_OOS_TRADES_PER_YEAR = 20
DISPERSION = 0.5
ERAS = {
    "A": {"data": ("2008-01-01", "2016-08-31"), "eps": "earnings_dart_doc",
          "folds": [(f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(2011, 2016)] + [("2014-01-01", "2016-01-01", "2016-09-01")]},
    "B": {"data": ("2016-09-01", "2023-12-31"), "eps": "earnings",
          "folds": [(f"{y - 2}-01-01", f"{y}-01-01", f"{y + 1}-01-01") for y in range(2020, 2024)]},
}
#: 엔진 설정 허용 범위(train.py 의 cfg 를 여기로 자른다). 그 밖의 키는 버린다.
CFG_BOUNDS = {"max_pos": (5, 20), "hard": (0.10, 0.30), "risk": (0.002, 0.01), "pos_cap": (0.05, 0.20)}
CFG_CHOICES = {"entry": ("open", "close"), "pyramid": (0, 1), "regime": (0, 1)}
CFG_DEFAULT = {"max_pos": 10, "hard": 0.15, "risk": 0.005, "pos_cap": 0.10, "entry": "close", "pyramid": 1, "regime": 0}
#: train.py 가 받는 배열(탐색 절반 종목 × 날짜). 횡단면 순위는 train.py 가 이 절반 안에서 직접 만든다.
VIEW_KEYS = ("O", "H", "L", "C", "tv", "adv", "cap", "eps_yoy", "uni", "TR")
_G: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - _G.get('t0', time.time()):6.0f}s] {msg}", file=sys.stderr, flush=True)


def half_of(code: str) -> int:
    """종목 절반 잠금 — 코드의 sha256 첫 바이트 짝홀. 0 = 탐색, 1 = 잠금."""
    return hashlib.sha256(str(code).encode()).digest()[0] & 1


# ---------------------------------------------------------------- 캐시 만들기(--build)

def build_era(con, era: str) -> dict:
    import research.experiments.indep30 as I30
    from swing_it.features.eps_asof import eps_ok, eps_yoy
    from swing_it.features.korea import cost_rt
    E = ERAS[era]
    D = I30.build(con, E["data"], E["eps"])
    days, codes = D["days"], D["codes"]
    D["lu"] = np.where(days < pd.Timestamp("2015-06-15"), 1.14, 1.29)           # 정정 30-1 M1(호가 단위 여유)
    tax = np.array([cost_rt(d, "KOSPI") for d in days])                         # 매도일 세율(KOSPI·KOSDAQ 같음 — 확인함)
    D["cost"], D["mkt"] = np.vstack([tax, tax]), np.zeros(len(codes), np.int64)
    ci, s = np.nonzero(D["uni"] & D["TR"])
    up, yoy = np.zeros(D["C"].shape, bool), np.full(D["C"].shape, np.nan, np.float32)
    up[ci, s] = eps_ok(codes, days, ci, s, con, source=E["eps"])
    yoy[ci, s] = eps_yoy(codes, days, ci, s, con, source=E["eps"])
    keep = ("days", "codes", "O", "H", "L", "C", "TR", "cost", "mkt", "hi20", "adv", "uni", "cap", "tv", "lu", "regime_on", "kx", "diag", "delisted")
    out = {k: D[k] for k in keep}
    out.update(eps_up=up, eps_yoy=yoy, half=np.array([half_of(c) for c in codes], np.int8), era=era)
    for k in ("adv", "cap", "tv"):
        out[k] = np.asarray(out[k], np.float32)
    out["diag"] = {**D["diag"], "eps_up_share": float(up[ci, s].mean()), "half1_share": float(out["half"].mean())}
    return out


def build() -> None:
    import psycopg2
    os.makedirs(CACHE, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    for era in ERAS:
        D = build_era(con, era)
        with open(os.path.join(CACHE, f"era_{era}.pkl"), "wb") as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"era {era}: {len(D['codes'])} codes × {len(D['days'])} days · {D['diag']}")


# ---------------------------------------------------------------- 평가

def load_era(era: str, half: int) -> dict:
    with open(os.path.join(CACHE, f"era_{era}.pkl"), "rb") as f:
        D = pickle.load(f)
    sel = np.flatnonzero(D["half"] == half)
    n = len(D["codes"])
    for k, v in list(D.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n and k != "mkt":
            D[k] = v[sel]
    D["mkt"] = D["mkt"][sel]
    D["codes"] = [D["codes"][i] for i in sel]
    D["kx"] = pd.Series(D["kx"]).reindex(D["days"]).ffill().to_numpy(float)
    return D


def view(D: dict) -> dict:
    """train.py 에 넘기는 읽기 전용 묶음(복사본 아님 — 쓰기 금지 플래그)."""
    V = {k: D[k] for k in VIEW_KEYS}
    for v in V.values():
        v.setflags(write=False)
    V.update(days=D["days"], codes=list(D["codes"]), kx=D["kx"], regime_on=D["regime_on"].copy(), era=D["era"])
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
    return out


def simulate(D: dict, sig: dict, cfg: dict, start: str, end: str, exclude=()) -> "tuple[pd.Series, pd.DataFrame]":
    import research.experiments.leader21 as L
    cand = np.asarray(sig["cand"], bool) & D["uni"] & D["TR"] & D["eps_up"]                 # 가드: EPS 상승·유동성·거래 가능 강제
    score = np.where(cand, np.asarray(sig["score"], float), np.nan)
    exit_line = np.asarray(sig["exit_line"], float)
    if exit_line.shape != D["C"].shape or cand.shape != D["C"].shape:
        raise ValueError("cand·score·exit_line 은 (종목, 날짜) 모양이어야 한다")
    D2 = {**D, "cand": cand, "rs": score, "brk": {0: np.ones_like(cand)}, "ma": {"x": exit_line}}
    c = clamp_cfg(cfg)
    ecfg = {"theta": -np.inf, "N": 0, "ma_exit": "x", "hard": c["hard"], "pyramid": c["pyramid"], "regime": c["regime"],
            "max_pos": c["max_pos"], "risk": c["risk"], "pos_cap": c["pos_cap"], "entry": c["entry"], "exclude": list(exclude)}
    with np.errstate(invalid="ignore"):
        return L.simulate(D2, ecfg, COST_MULT, end=end, start=start)


def excess_log(D: dict, eq: pd.Series, lo: str, hi: str) -> "tuple[float, float]":
    """구간 [lo, hi) 연율 초과 로그수익(전략 − 공식 KOSPI 보유)과 연수."""
    ix = (eq.index >= lo) & (eq.index < hi)
    x = eq[ix]
    if len(x) < 2:
        return float("nan"), 0.0
    k = D["kx"][D["days"].get_indexer(x.index)]
    yrs = max((x.index[-1] - x.index[0]).days / 365.25, 1e-9)
    return float((np.log(x.iloc[-1] / x.iloc[0]) - np.log(k[-1] / k[0])) / yrs), yrs


def _train_point(args) -> "tuple[int, int, float]":
    fi, j, x = args
    D, S = _G["D"], _G["S"]
    lo, hi = _G["folds"][fi][0], _G["folds"][fi][1]
    try:
        sig = S["signals"](_G["V"], x)
        eq, tr = simulate(D, sig, S["cfg"](x), lo, hi)
        ex, yrs = excess_log(D, eq, lo, hi)
        n = len(tr[tr["entry"] >= lo]) if len(tr) else 0
        return fi, j, (ex if n >= MIN_TRAIN_TRADES_PER_YEAR * yrs else -1.0)
    except Exception as e:  # noqa: BLE001 — 한 점의 실패는 그 점만 버린다(로그)
        log(f"train point fold {fi} #{j} 실패: {e!r}")
        return fi, j, -1.0


def sample_params(spec: dict, n: int, seed: int) -> "tuple[list[dict], np.ndarray]":
    names = list(spec)
    if not names:
        return [{}], np.zeros((1, 0))
    rng = np.random.default_rng(seed)                                # 라틴 하이퍼큐브(차원마다 n 칸에 한 점씩 — 의존성 없이 고른 덮기)
    U = (np.argsort(rng.random((len(names), n)), axis=1).T + rng.random((n, len(names)))) / n
    pts = []
    for u in U:
        x = {}
        for k, name in enumerate(names):
            s = spec[name]
            if s[0] == "choice":
                x[name] = s[1][min(int(u[k] * len(s[1])), len(s[1]) - 1)]
            else:
                lo, hi = s[1], s[2]
                v = lo + u[k] * (hi - lo)
                x[name] = int(round(v)) if s[0] == "int" else float(v)
        pts.append(x)
    return pts, U


def plateau_pick(U: np.ndarray, y: np.ndarray, k: int = K_PLATEAU) -> int:
    """이웃 k 개(정규화 공간 유클리드, 자기 포함) **중앙값**이 가장 높은 점 — 한 점 극값이 아니라 고원(CLAUDE.md). 평균은 외딴 극값 하나가 이웃 손실을 덮었다(테스트)."""
    if U.shape[1] == 0 or len(y) <= k:
        return int(np.argmax(y))
    d = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    nb = np.argsort(d, axis=1)[:, :k]
    return int(np.argmax(np.median(y[nb], axis=1)))


def load_strategy(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("ar_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    S = mod.get_strategy()
    for k in ("params", "signals", "cfg"):
        if k not in S:
            raise ValueError(f"get_strategy() 에 {k!r} 가 없다")
    return S


def walk_forward(S: dict, half: int, workers: int, rank_null_seed: "int | None" = None, fixed: "list | None" = None) -> dict:
    """시대 A·B 폴드마다 학습 구간 파라미터 선택 → 시험 구간 한 번. ``fixed`` 를 주면 선택 없이 그 파라미터로(널·잠금 재사용)."""
    res = {"folds": [], "trades": []}
    for era, E in ERAS.items():
        D = load_era(era, half)
        _G.update(D=D, S=S, V=view(D), folds=E["folds"])
        chosen = []
        if fixed is None:
            tasks, U_all = [], {}
            for fi in range(len(E["folds"])):
                pts, U = sample_params(S["params"], N_EVAL, SEED + 101 * fi + (0 if era == "A" else 7))
                U_all[fi] = (pts, U)
                tasks += [(fi, j, x) for j, x in enumerate(pts)]
            S["signals"](_G["V"], tasks[0][2])                       # 부모에서 train.py 캐시를 데운다(fork 로 작업자가 공유)
            with mp.get_context("fork").Pool(workers) as pool:
                out = pool.map(_train_point, tasks, chunksize=1)
            for fi in range(len(E["folds"])):
                pts, U = U_all[fi]
                y = np.array([v for f, _, v in sorted(out) if f == fi])
                chosen.append(pts[plateau_pick(U, y)])
        else:
            chosen = [f["params"] for f in fixed if f["era"] == era]
        for fi, (_, lo, hi) in enumerate(E["folds"]):
            x = chosen[fi]
            sig = S["signals"](_G["V"], x)
            if rank_null_seed is not None:                           # 널: 같은 후보, 순위만 무작위
                rng = np.random.default_rng(rank_null_seed * 1000 + fi + (0 if era == "A" else 500))
                sig = {**sig, "score": rng.random(D["C"].shape)}
            eq, tr = simulate(D, sig, S["cfg"](x), lo, hi)
            tr = tr[(tr["entry"] >= lo) & (tr["entry"] < hi)] if len(tr) else tr
            ex, yrs = excess_log(D, eq, lo, hi)
            m = pd.Series(eq).resample("ME").last()
            km = pd.Series(D["kx"][D["days"].get_indexer(eq.index)], index=eq.index).resample("ME").last()
            mret = (m.pct_change() - km.pct_change()).dropna()
            res["folds"].append({"era": era, "test": [lo, hi], "params": x, "excess": ex, "years": yrs, "n": int(len(tr)),
                                 "cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / max(yrs, 1e-9)) - 1),
                                 "kospi": float((D["kx"][D["days"].searchsorted(pd.Timestamp(hi)) - 1] / D["kx"][D["days"].searchsorted(pd.Timestamp(lo))]) ** (1 / max(yrs, 1e-9)) - 1),
                                 "mexcess": mret.tolist()})
            if len(tr):
                res["trades"].append(tr.assign(era=era, fold=fi))
    return res


def summarize(res: dict) -> dict:
    ex = np.array([f["excess"] for f in res["folds"]])
    n = np.array([f["n"] for f in res["folds"]])
    tr = pd.concat(res["trades"]) if res["trades"] else pd.DataFrame(columns=["ret", "code"])
    yrs = sum(f["years"] for f in res["folds"])
    void = ((n < MIN_FOLD_TRADES).sum() > MAX_THIN_FOLDS or n.sum() < MIN_OOS_TRADES or n.sum() / max(yrs, 1e-9) < MIN_OOS_TRADES_PER_YEAR
            or not np.isfinite(ex).all())
    score = float(ex.mean() - DISPERSION * ex.std(ddof=1)) if not void else -1.0
    return {"score": score, "void": bool(void), "fold_mean": float(np.nanmean(ex)), "fold_std": float(np.nanstd(ex, ddof=1)),
            "folds_pos": f"{int((ex > 0).sum())}/{len(ex)}",
            "eraA_mean": float(np.nanmean([f["excess"] for f in res["folds"] if f["era"] == "A"])),
            "eraB_mean": float(np.nanmean([f["excess"] for f in res["folds"] if f["era"] == "B"])),
            "oos_trades": int(n.sum()), "oos_trades_per_year": float(n.sum() / max(yrs, 1e-9)), "min_fold_trades": int(n.min()), "thin_folds": int((n < MIN_FOLD_TRADES).sum()),
            "trade_ret_mean": float(tr["ret"].mean()) if len(tr) else float("nan"),
            "payoff_ratio": float(tr.loc[tr["ret"] > 0, "ret"].mean() / -tr.loc[tr["ret"] <= 0, "ret"].mean()) if (tr["ret"] > 0).any() and (tr["ret"] <= 0).any() else float("nan"),
            "win_rate": float((tr["ret"] > 0).mean()) if len(tr) else float("nan"),
            "fold_excess": " ".join(f"{f['era']}{f['test'][0][:4]}:{f['excess']:+.3f}({f['n']})" for f in res["folds"])}


def top_removed_score(S: dict, res: dict, half: int, k: int = 5) -> float:
    """폴드 파라미터 그대로, 시험 구간 기여(ret) 상위 k 종목을 안 산 것으로 다시 돌린 점수(보고 열)."""
    tr = pd.concat(res["trades"]) if res["trades"] else None
    if tr is None:
        return float("nan")
    top = tr.groupby("code")["ret"].sum().sort_values(ascending=False).index[:k].tolist()
    ex = []
    for era, E in ERAS.items():
        D = load_era(era, half)
        V = view(D)
        for f in [f for f in res["folds"] if f["era"] == era]:
            lo, hi = f["test"]
            eq, _ = simulate(D, S["signals"](V, f["params"]), S["cfg"](f["params"]), lo, hi, exclude=top)
            ex.append(excess_log(D, eq, lo, hi)[0])
    ex = np.array(ex)
    return float(ex.mean() - DISPERSION * ex.std(ddof=1))


def n_trials() -> int:
    p = os.path.join(HERE, "results.tsv")
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def integrity() -> None:
    """평가기 무결성 — 사전등록 태그 뒤로 prepare.py 를 바꾼 커밋이나 커밋 안 된 변경이 있으면 실행하지 않는다."""
    import subprocess
    rel = os.path.relpath(os.path.abspath(__file__), ROOT)
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True)
    if git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}").returncode != 0:
        print(f"경고: 태그 {PREREG_TAG} 없음 — 사전등록 전 시험 실행", file=sys.stderr)
        return
    changed = git("log", "--format=%h %s", f"{PREREG_TAG}..HEAD", "--", rel).stdout.strip()
    dirty = git("status", "--porcelain", "--", rel).stdout.strip()
    if changed or dirty:
        raise SystemExit(f"prepare.py 가 사전등록({PREREG_TAG}) 뒤 바뀌었다 — 평가 거부.\n{changed}\n{dirty}")


def evaluate(train_path: str, workers: "int | None" = None) -> dict:
    """실험 한 번 — 탐색 절반에서 워크포워드 점수와 보고 열을 stdout 에 ``key: value`` 로 낸다."""
    integrity()
    _G["t0"] = time.time()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    S = load_strategy(train_path)
    res = walk_forward(S, half=0, workers=workers)
    s = summarize(res)
    s["top5_removed_score"] = top_removed_score(S, res, 0) if not s["void"] else float("nan")
    s["n_trials_before"] = n_trials()
    s["elapsed_s"] = round(time.time() - _G["t0"], 1)
    print("---")
    for k, v in s.items():
        print(f"{k + ':':22s} {v:.6f}" if isinstance(v, float) else f"{k + ':':22s} {v}")
    with open(os.path.join(HERE, "last_run.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": s, "folds": [{k: v for k, v in f_.items() if k != "mexcess"} for f_ in res["folds"]]}, f, ensure_ascii=False, indent=1, default=str)
    return s


# ---------------------------------------------------------------- 졸업 심사(--final, 사람만)

def final(train_path: str, workers: int, n_null: int = 100) -> None:
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample
    used = 0
    if os.path.exists(LOCKBOX_LOG):
        with open(LOCKBOX_LOG, encoding="utf-8") as f:
            used = sum(1 for _ in f)
    integrity()
    if used >= LOCKBOX_MAX:
        raise SystemExit(f"잠금 절반은 이미 {used}회 열렸다(최대 {LOCKBOX_MAX}) — PREREG.md")
    _G["t0"] = time.time()
    S = load_strategy(train_path)
    search = walk_forward(S, 0, workers)
    s0 = summarize(search)
    null = []
    for r in range(n_null):
        null.append(summarize(walk_forward(S, 0, workers, rank_null_seed=r + 1, fixed=search["folds"]))["score"])
    lock = walk_forward(S, 1, workers)
    s1 = summarize(lock)
    mex = np.concatenate([f["mexcess"] for f in search["folds"]])
    dsr = deflated_sharpe_from_sample(mex, max(n_trials(), 1))
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": hashlib.sha256(open(train_path, "rb").read()).hexdigest()[:12],
           "n_trials": n_trials(), "search": s0, "search_top5_removed": top_removed_score(S, search, 0),
           "null_rank_p95": float(np.quantile(null, 0.95)), "null_rank_median": float(np.median(null)),
           "lockbox": s1, "lockbox_top5_removed": top_removed_score(S, lock, 1), "dsr_monthly": dsr,
           "folds_search": [{k: v for k, v in f.items() if k != "mexcess"} for f in search["folds"]],
           "folds_lockbox": [{k: v for k, v in f.items() if k != "mexcess"} for f in lock["folds"]]}
    with open(LOCKBOX_LOG, "a", encoding="utf-8") as f:            # 덧붙이기 전용 — 잠금 사용 횟수의 증인
        f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--train", default=os.path.join(HERE, "train.py"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    _G["t0"] = time.time()
    if a.build:
        build()
    elif a.final:
        final(a.train, a.workers)
    else:
        evaluate(a.train, a.workers)


if __name__ == "__main__":
    main()
