"""세 레포 종합 autoresearch 고정 평가기 — 진입점이자 심판. 에이전트는 이 파일을 고치지 않는다(program.md · PREREG.md).

swing v4 평가기(``research/autoresearch/prepare.py``, 태그 autoresearch-prereg-4)를 골격으로 쓴다. v4 의 순수 함수(전역을 읽지 않는 것)는
import 해서 그대로 쓰고(lint_train·load_strategy·sample_params·plateau_pick·clamp_cfg·exit_line·fold_stats·summarize·factor_months·factor_t·
half_of·sha_file), v4 의 ERAS·_G·VIEW_KEYS·TRAIN·CACHE 를 읽는 것은 복사해 고쳤다. 설계: docs/superpowers/specs/2026-10-25-combined-autoresearch-design.md.

v4 와 다른 점(전부):
1. 데이터: 시대 A(2011~2015 시험 5폴드)·B(2020~2024 시험 5폴드) = 10폴드, ``research.combined.data`` 캐시(갭·뉴스 키 추가). 2016 폴드는 없다.
   확인 시대 C 는 루프가 보지 못한다(격리 프로세스에 바인드하지 않는다 — select.py 몫).
2. V = v4 키 + 갭(gap·tickp·gap_event) + 뉴스(n_all·n_feat·n_move·n_excl·news_known). 미래정보 절단 검사도 새 키를 자른다.
3. 진입은 **신호일 종가**(``entry="close"``, leader_C 방식) — 평가기가 강제한다. exit_dist(진입일 청산선 거리)도 종가 대비.
4. ``simulate_c`` 에 ``mask``(cand 에서 끌 칸 — 상위 1% 거래 제거용).
5. 폴드마다 ``trade_key``(시험 체결 (종목 위치, 진입일 위치) 정렬 목록 sha1 앞 12자). 거래 표에 ci·ti 열, fold = 전체 폴드 번호(v4 는 시대 안 번호).
6. 쏠림 진단 셋(``concentration``): 상위 1% 거래 신호 칸 끄기 · 최대 기여 한 종목 빼기 · 일별 선택 알파 상위 10일 0. v4 의 top5 제거를 대신한다.
7. 부품 제거(``ablation``): 레포 키를 중립값으로 바꾸고 폴드 파라미터 고정 재실행 → 체결 대칭차 ≥ 5% ∧ 작동 폴드 점수 기여 > 0.
   작동 폴드 = scalp 10폴드 전부, daytrade 시험 연도 2023·2024. 대칭차·점수 모두 작동 폴드에서 잰다. ``eligible`` = 둘 다 ok.
   자격은 keep 과 무관하다(후보 고르기 select.py 의 거르개).
8. keep(``beats_c``): 폴드 조건이 "10 중 6 개선" → "체결이 바뀐 폴드(≥ 1) 중 alpha 개선 과반". top5 조건 → 쏠림 셋 각각 BEST 의 같은 값 − 0.01 이상.
9. 시행 상한(``trial_state``): 고유 train sha 150, 또는 첫 keep(= --accept 기록) 뒤 고유 sha 30개 연속 keep 없음. 상한이면 ``evaluate`` 는
   계산·무결성 검사 전에 거부한다. 원장은 작업 트리 + ~/.cache 사본 중 긴 쪽(v4 와 같음).
10. 출력 줄: score·void·beats_best·beats_detail·lookahead_check·folds_pos·oos_trades·payoff_ratio·ev_net_2x·ev_net_1x·쏠림 셋·
    news_folds_score·ablation 둘·eligible·trials·stop. ev_net_1x 는 같은 폴드 파라미터로 비용 1배 재실행.
11. 미래정보 검사 실패 메시지는 "fail: " 로 시작한다(v4 는 접두 없음). 계산 본체는 ``run_all``(격리 안 ``--inner`` 가 부른다 — 테스트 가능).
12. 격리: 캐시 디렉터리를 가리고 era_A·era_B 만 다시 보인다. 원장 사본(평가 전에 만들어 늘 가린다)·~/git/daytrade-it(뉴스 아카이브)·
    ~/git/quant-airflow·~/.config·~/.ssh 를 가린다.
13. 무결성: 태그 ``combined-prereg-1``, 감시 파일에 결합 data·gaps·news·select 와 v4 평가기 추가, MANIFEST 에 캐시 A·B·C sha.
    ``--build`` 는 data.build_all 에 맡기고 MANIFEST 는 쓰지 않는다(``--manifest`` 따로 — select.py 가 생긴 뒤). ``--final`` 은 없다.
14. 원장 기록: 평가 행에 폴드별 era·test·params·alpha·trade_key·n, 쏠림 셋, 부품 제거, eligible, 거래당 값을 남긴다(select.py 가 읽는다).
15. 상위 1% 거래 수 = ceil(1% × 시험 체결 수)(최소 1). crash 는 원장에 남지 않아 시행으로 세지 않는다(v4 와 같음).

사용(작업 트리 루트):
  uv run python research/combined/prepare.py > run.log 2>&1          # 실험 한 번
  uv run python research/combined/prepare.py --accept                 # 직전 실행이 beats_best 면 BEST 갱신(keep)
  uv run --extra pg python research/combined/prepare.py --build      # 캐시(data.build_all, KR_QUANT_DB)
  uv run python research/combined/prepare.py --manifest               # 코드·캐시 sha 매니페스트(사전등록 때)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time

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
    lint_train,
    load_strategy,
    plateau_pick,
    sample_params,
    sha_file,
)
from research.combined.data import CACHE, ERAS_C, REPO_KEYS, VIEW_KEYS_C, view_c  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train.py")
SHADOW = os.path.expanduser("~/.cache/cc-swing/combined_ledger")        # 원장 사본(작업 트리를 지워도 남는다)
RUNS, BEST, LAST = (os.path.join(HERE, f) for f in ("RUNS.jsonl", "BEST.json", "last_run.json"))
MANIFEST = os.path.join(HERE, "MANIFEST.json")
PREREG_TAG = "combined-prereg-1"
GUARDED_FILES = ("research/combined/prepare.py", "research/combined/data.py", "research/combined/gaps.py", "research/combined/news.py",
                 "research/combined/select.py", "research/autoresearch/prepare.py", "research/experiments/leader21.py",
                 "research/experiments/indep30.py")
CACHE_ERAS = ("A", "B", "C")                                              # MANIFEST 대상(C 는 확인 — 루프는 못 본다)
ERAS = {e: ERAS_C[e] for e in ("A", "B")}                                 # 루프가 보는 시대
COST_MULT = V4.COST_MULT
SEED = 20261025
N_EVAL, DISPERSION, MIN_TRAIN_TRADES_PER_YEAR = V4.N_EVAL, V4.DISPERSION, V4.MIN_TRAIN_TRADES_PER_YEAR
KEEP_DSCORE, KEEP_CONC_SLACK = 0.005, 0.01
MAX_TRIALS, MAX_SINCE_KEEP = 150, 30
ABL_MIN_CHANGED = 0.05
NEWS_YEARS = (2023, 2024)
TOP_DAYS, TOP_TRADE_FRAC = 10, 0.01
CONC_KEYS = ("top1pct", "top_stock", "top10d")
_SANDBOX_HIDE_FIXED = tuple(os.path.expanduser(p) for p in ("~/git/daytrade-it", "~/git/quant-airflow", "~/.config", "~/.ssh"))
_G: dict = {}


def log(msg: str) -> None:
    print(f"[{time.time() - _G.get('t0', time.time()):6.0f}s] {msg}", file=sys.stderr, flush=True)


def sandbox_hide() -> tuple:
    return (SHADOW, *_SANDBOX_HIDE_FIXED)


def score_of(al) -> float:
    """점수 = 폴드 alpha 평균 − 0.5σ(ddof 1). 유한값 2개 미만이면 nan."""
    a = np.asarray(al, float)
    a = a[np.isfinite(a)]
    return float(a.mean() - DISPERSION * a.std(ddof=1)) if len(a) >= 2 else float("nan")


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


def _train_point(args):
    fi, j, x = args
    D, S, lo, hi = _G["D"], _G["S"], _G["folds"][fi][0], _G["folds"][fi][1]
    try:
        eq, tr = simulate_c(D, S["signals"](_G["V"], x), S["cfg"](x), lo, hi)
        st = fold_stats(D, eq, lo, hi)
        n = int((tr["entry"] >= lo).sum()) if len(tr) else 0
        return fi, j, (st["alpha"] if n >= MIN_TRAIN_TRADES_PER_YEAR * st["years"] else -1.0)
    except Exception as e:  # noqa: BLE001 — 한 점 실패는 그 점만 버린다
        log(f"학습 점 fold {fi} #{j} 실패: {e!r}")
        return fi, j, -1.0


def walk_forward_c(D_by_era: dict, workers: int, train_path: str = TRAIN, fixed: "list | None" = None, null: "tuple | None" = None,
                   neutral: "str | None" = None, cost_mult: float = COST_MULT, exclude=(), mask: "dict | None" = None) -> dict:
    """시대마다 새 모듈로 train.py 로드 → 폴드별 학습 구간 고원 선택 → 시험 한 번(v4 walk_forward 복사).
    ``fixed`` = 그 폴드 파라미터 그대로, ``null`` = ("rank"|"thin"|"params", seed), ``neutral`` = 레포 키 중립화,
    ``exclude`` = 뺄 종목코드, ``mask`` = {시대: bool[N,T]} cand 에서 끌 칸."""
    if neutral is not None and neutral not in REPO_KEYS:
        raise ValueError(f"neutral={neutral!r}")
    res = {"folds": [], "trades": [], "months": [], "daily": []}
    for ei, (era, E) in enumerate(ERAS.items()):
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
                pts, U = sample_params(S["params"], N_EVAL, SEED + 101 * fi + 7 * ei)
                pts_all[fi] = (pts, U)
                tasks += [(fi, j, x) for j, x in enumerate(pts)]
            S["signals"](_G["V"], tasks[0][2])                       # 부모에서 train.py 캐시를 데운다(fork 공유)
            with mp.get_context("fork").Pool(max(1, workers)) as pool:
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
            elif null and null[0] == "thin":                         # 후보 솎기: 시험 구간 월별 통과율 그대로
                base = D["uni"] & D["TR"] & D["eps_up"]
                c = np.asarray(sig["cand"], bool) & base
                mon = D["days"].to_period("M")
                pr = np.zeros(len(mon))
                for mm in np.unique(mon):
                    j = mon == mm
                    pr[j] = c[:, j].sum() / max(base[:, j].sum(), 1)
                sig = {"cand": base & (rng.random(D["C"].shape) < pr[None, :]), "score": rng.random(D["C"].shape)}
            eq, tr = simulate_c(D, sig, S["cfg"](x), lo, hi, exclude=exclude, mask=None if mask is None else mask.get(era),
                                cost_mult=cost_mult)
            tr = tr[(tr["entry"] >= lo) & (tr["entry"] < hi)] if len(tr) else tr
            if len(tr):                                              # 진입일 청산선까지 거리(종가 대비) — '좁은 실질 손절' 감시
                c = clamp_cfg(S["cfg"](x))
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
    return res


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


def concentration(D_by_era: dict, res: dict, train_path: str = TRAIN) -> dict:
    """쏠림 진단 셋 — 각각 제거 뒤 score(void 무관 원점수). 파라미터는 고른 값 그대로."""
    if not res["trades"]:
        return {k: float("nan") for k in CONC_KEYS}
    tr = pd.concat(res["trades"])
    top = tr.nlargest(max(1, math.ceil(TOP_TRADE_FRAC * len(tr))), "ret")
    mask = {e: np.zeros(D_by_era[e]["C"].shape, bool) for e in ERAS}
    for era, ci, ti in zip(top["era"], top["ci"], top["ti"], strict=True):
        mask[era][int(ci), int(ti)] = True                           # 종가 진입: 신호 칸 = 진입 칸
    r1 = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], mask=mask)
    top_code = tr.groupby("code")["ret"].sum().idxmax()             # 순수익 합 최대 종목
    r2 = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], exclude=[top_code])
    return {"top1pct": score_of(_alphas(r1)), "top_stock": score_of(_alphas(r2)),
            "top10d": top10d_score(res["daily"], [f["years"] for f in res["folds"]])}


def _tset(res: dict, idx: list) -> set:
    out = set()
    for tr in res["trades"]:
        tr = tr[tr["fold"].isin(idx)]
        out |= set(zip(tr["fold"].astype(int), tr["ci"].astype(int), tr["ti"].astype(int)))
    return out


def ablation(D_by_era: dict, res: dict, train_path: str = TRAIN) -> dict:
    """레포 부품 제거 — 중립 V 로 폴드 파라미터 고정 재실행. 작동 폴드에서 체결 대칭차 ≥ 5% ∧ 점수 기여 > 0 이면 ok."""
    out = {}
    for repo in REPO_KEYS:
        idx = active_folds(res["folds"], repo)
        r = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], neutral=repo)
        a, b = _tset(res, idx), _tset(r, idx)
        changed = len(a ^ b) / max(len(a), 1)
        dscore = score_of(_alphas(res, idx)) - score_of(_alphas(r, idx))
        out[repo] = {"changed": float(changed), "dscore": float(dscore), "ok": bool(changed >= ABL_MIN_CHANGED and dscore > 0)}
    out["eligible"] = all(out[r]["ok"] for r in REPO_KEYS)
    return out


# ---------------------------------------------------------------- 미래정보 검사(v4 causality — VIEW_KEYS_C·루프 시대)

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
    return out


def _sig_many(V: dict, xs: list, train_path: str) -> list:
    S = load_strategy(train_path)                                     # 새 모듈 — 캐시 공유 없음
    return [S["signals"](V, x) for x in xs]


def _cause_task(args):
    era, cut, mode, xs, train_path = args
    V = view_c(_cut(_G["DC"][era], cut, shuffle_seed=cut if mode == "shuffle" else None))
    return era, cut, mode, [(np.asarray(g["cand"], bool)[:, :cut + 1], np.asarray(g["score"], float)[:, :cut + 1])
                            for g in _sig_many(V, xs, train_path)]


def causality_c(D_by_era: dict, res: dict, train_path: str = TRAIN, workers: int = 4, n_cut: int = 12) -> "tuple[bool, str]":
    """허용 0, 시대마다 절단 n_cut 곳(시험 폴드 시작 + 무작위), 그 시대 폴드 파라미터 전부, 절단 · 절단 뒤 시간 섞기 두 방식."""
    _G["DC"] = D_by_era
    tasks, full = [], {}
    for era, E in ERAS.items():
        D = D_by_era[era]
        T = D["C"].shape[1]
        xs = []
        for f in res["folds"]:
            if f["era"] == era and f["params"] not in xs:
                xs.append(f["params"])
        full[era] = (xs, _sig_many(view_c(D), xs, train_path))
        starts = [int(D["days"].searchsorted(pd.Timestamp(lo))) for _, lo, _ in E["folds"]]
        rnd = np.random.default_rng(SEED + len(starts)).integers(int(T * 0.3), T - 2, size=max(0, n_cut - len(starts)))
        for cut in sorted(set(starts + [int(c) for c in rnd])):
            for mode in ("trunc", "shuffle"):
                tasks.append((era, min(cut, T - 2), mode, xs, train_path))
    with mp.get_context("fork").Pool(max(1, min(workers, len(tasks)))) as pool:
        outs = pool.map(_cause_task, tasks, chunksize=1)
    for era, cut, mode, parts in outs:
        xs, fulls = full[era]
        for x, g, (cb, sb) in zip(xs, fulls, parts, strict=True):
            ca = np.asarray(g["cand"], bool)[:, :cut + 1]
            if ca.shape != cb.shape or (ca != cb).any():
                return False, f"fail: era {era} cut {cut} {mode} params {x}: cand 불일치 {int((ca != cb).sum()) if ca.shape == cb.shape else 'shape'}칸"
            sa = np.asarray(g["score"], float)[:, :cut + 1][ca]
            if not np.allclose(sa, sb[ca], rtol=1e-9, atol=1e-12, equal_nan=True):
                return False, f"fail: era {era} cut {cut} {mode} params {x}: score 불일치"
    return True, f"ok({len(tasks)} 검사)"


# ---------------------------------------------------------------- 계산 본체(격리 안)

def run_all(D_by_era: dict, workers: int, train_path: str = TRAIN) -> dict:
    S = load_strategy(train_path)
    res = walk_forward_c(D_by_era, workers, train_path)
    log("워크포워드 끝")
    s = summarize_c(res)
    ok, why = causality_c(D_by_era, res, train_path, workers=max(1, workers // 2))
    log("미래정보 검사 끝")
    if not ok:
        s.update(void=True, score=-1.0)
    s["lookahead_check"] = why
    if s["void"]:
        conc = {k: float("nan") for k in CONC_KEYS}
        abl = {"scalp": None, "daytrade": None, "eligible": False}
        s["ev_net_1x"] = float("nan")
    else:
        conc = concentration(D_by_era, res, train_path)
        abl = ablation(D_by_era, res, train_path)
        c1 = walk_forward_c(D_by_era, 1, train_path, fixed=res["folds"], cost_mult=1.0)
        s["ev_net_1x"] = float(pd.concat(c1["trades"])["ret"].mean()) if c1["trades"] else float("nan")
        log("쏠림·부품 제거·비용 1배 끝")
    s["train_sha"] = S["sha"]
    return {"summary": s, "folds": [dict(f) for f in res["folds"]], "conc": conc, "ablation": abl}


def inner(workers: int) -> None:
    """bwrap 안에서 도는 계산(train.py 코드가 실행되는 유일한 곳) — 결과 JSON 한 줄을 stdout 에. 원장·BEST 는 바깥이 쓴다."""
    from research.combined.data import load
    _G["t0"] = time.time()
    out = run_all({e: load(e) for e in ERAS}, workers)
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


def trial_state() -> dict:
    """n = 평가된 고유 train sha 수. since_keep = 마지막 keep(--accept 기록) 뒤 새 sha 수. stop = n ≥ 150 ∨ (keep 있음 ∧ since_keep ≥ 30)."""
    seen, since, kept = set(), 0, False
    for r in _read_jsonl(RUNS):
        if r.get("accept"):
            kept, since = True, 0
            continue
        sha = r.get("train_sha")
        if sha is None or "folds" not in r or sha in seen:
            continue
        seen.add(sha)
        if kept:
            since += 1
    n = len(seen)
    return {"n": n, "since_keep": since, "stop": bool(n >= MAX_TRIALS or (kept and since >= MAX_SINCE_KEEP))}


def beats_c(s: dict, folds: list, conc: dict) -> "tuple[bool, str]":
    """keep: 체결이 바뀐 폴드 ≥ 1, 그중 alpha 개선 과반, Δscore ≥ 0.005, 쏠림 셋 각각 ≥ BEST 의 같은 값 − 0.01."""
    if s["void"]:
        return False, "void"
    if not os.path.exists(BEST):
        return True, "첫 유효 실행"
    b = json.load(open(BEST))
    changed = [(f, g) for f, g in zip(folds, b["folds"], strict=True) if f["trade_key"] != g["trade_key"]]
    better = sum(f["alpha"] > g["alpha"] for f, g in changed)
    dconc = {k: conc[k] - b["conc"][k] for k in CONC_KEYS}
    ok = (len(changed) >= 1 and better * 2 > len(changed) and s["score"] >= b["score"] + KEEP_DSCORE
          and all(conc[k] >= b["conc"][k] - KEEP_CONC_SLACK for k in CONC_KEYS))
    return bool(ok), (f"dscore {s['score'] - b['score']:+.4f} · better {better}/{len(changed)} 바뀐 폴드 · "
                      + " · ".join(f"d{k} {v:+.4f}" for k, v in dconc.items()))


def integrity(check_cache: bool = True) -> None:
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
    man.update({f"ext/{p}": sha_file(V4._ext_path(p)) for p in V4.EXT_GUARDED})
    json.dump(man, open(MANIFEST, "w"), indent=1)
    log(f"MANIFEST.json 기록 — {len(man)} 항목")


def _fmt(k: str, v) -> str:
    if isinstance(v, dict):
        v = f"changed {v['changed']:.4f} · dscore {v['dscore']:+.4f} · ok {v['ok']}"
    elif v is None:
        v = "void"
    return f"{k + ':':22s} {v:.6f}" if isinstance(v, float) else f"{k + ':':22s} {v}"


def evaluate(workers: int) -> dict:
    st = trial_state()                                                # 상한 검사가 맨 먼저(계산·무결성보다 앞)
    if st["stop"]:
        raise SystemExit(f"상한 도달 — 평가 거부 (고유 시행 {st['n']}, 마지막 keep 뒤 {st['since_keep']})")
    integrity()
    os.makedirs(SHADOW, exist_ok=True)                                # 있어야 격리에서 가린다
    t0 = time.time()
    p = subprocess.run(sandbox_cmd_c(["--inner", "--workers", str(workers)]), capture_output=True, text=True)
    sys.stderr.write(p.stderr[-4000:])
    line = [x for x in p.stdout.splitlines() if x.startswith("@@RESULT@@")]
    if p.returncode != 0 or not line:
        print("---\nscore:                 crash\n" + p.stdout[-2000:])
        raise SystemExit(p.returncode or 1)
    r = json.loads(line[-1][len("@@RESULT@@"):])
    s, folds, conc, abl = r["summary"], r["folds"], r["conc"], r["ablation"]
    cur = hashlib.sha256(open(TRAIN, "rb").read()).hexdigest()[:16]
    if s["train_sha"] != cur:
        raise SystemExit("격리 실행이 다른 train.py 를 평가했다 — 거부")
    s["beats_best"], s["beats_detail"] = beats_c(s, folds, conc)
    s.update({f"{k}_removed": conc[k] for k in CONC_KEYS})
    s.update(ablation_scalp=abl["scalp"], ablation_daytrade=abl["daytrade"], eligible=abl["eligible"])
    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "score": s["score"], "raw_score": s["raw_score"], "void": s["void"],
           "beats_best": s["beats_best"], "lookahead": s["lookahead_check"], "conc": conc, "ablation": abl, "eligible": abl["eligible"],
           "oos_trades": s["oos_trades"], "oos_trades_per_year": s["oos_trades_per_year"], "payoff_ratio": s["payoff_ratio"],
           "ev_net_2x": s["ev_net_2x"], "ev_net_1x": s["ev_net_1x"], "news_folds_score": s["news_folds_score"],
           "folds": [{k: f[k] for k in ("era", "test", "params", "alpha", "trade_key", "n")} for f in folds],
           "evaluator": PREREG_TAG, "elapsed_s": round(time.time() - t0, 1)}
    _append(RUNS, rec)
    st = trial_state()
    s["trials"], s["stop"] = st["n"], st["stop"]
    json.dump({"summary": s, "folds": folds, "conc": conc, "ablation": abl}, open(LAST, "w"), ensure_ascii=False, indent=1, default=str)
    shown = ("score", "void", "beats_best", "beats_detail", "lookahead_check", "folds_pos", "oos_trades", "payoff_ratio", "ev_net_2x",
             "ev_net_1x", "top1pct_removed", "top_stock_removed", "top10d_removed", "news_folds_score", "ablation_scalp",
             "ablation_daytrade", "eligible", "trials", "stop")
    print("---")                                                           # 폴드별 값은 숨긴다
    for k in shown:
        print(_fmt(k, s[k]))
    return s


def accept() -> None:
    """BEST 갱신 — 원장(RUNS)의 마지막 평가 기록이 현재 train.py 의 beats_best=True 일 때만."""
    integrity(check_cache=False)
    cur = hashlib.sha256(open(TRAIN, "rb").read()).hexdigest()[:16]
    evals = [r for r in _read_jsonl(RUNS) if "folds" in r]
    if not evals or evals[-1]["train_sha"] != cur or not evals[-1]["beats_best"]:
        raise SystemExit("원장의 마지막 평가가 현재 train.py 의 beats_best=True 가 아니다 — 갱신 거부")
    r = evals[-1]
    best = {"score": r["score"], "conc": r["conc"], "train_sha": cur, "evaluator": r["evaluator"], "eligible": r["eligible"],
            "folds": [{k: f[k] for k in ("era", "test", "alpha", "trade_key")} for f in r["folds"]]}
    json.dump(best, open(BEST, "w"), indent=1)
    _append(RUNS, {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_sha": cur, "accept": True, "score": r["score"]})
    _append(os.path.join(HERE, "BEST_HISTORY.jsonl"), best)
    print(f"BEST 갱신: score {r['score']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="캐시 A·B·C(data.build_all, KR_QUANT_DB)")
    ap.add_argument("--manifest", action="store_true", help="코드·캐시 sha 매니페스트(사전등록 때)")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    _G["t0"] = time.time()
    if a.build:
        from research.combined.data import build_all
        build_all()
    elif a.manifest:
        write_manifest()
    elif a.accept:
        accept()
    elif a.inner:
        inner(a.workers)
    else:
        evaluate(a.workers)


if __name__ == "__main__":
    main()
