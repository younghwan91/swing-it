"""32 RL 단계 — RL-1 후보 선택 정책(META_EVOLUTION.md '32 RL 단계 사전등록' + 보충 1, 결과 전 고정).

사건 = leader_P 후보 종목-일(슬롯 무관 전부), 보상 = 그 사건을 혼자 샀을 때 leader_P 규칙으로 끝낸 투입 R(비용 2배).
정책 = 기대 R 회귀(LightGBM, 하이퍼파라미터 고정) — 1 단계 RL(문맥 밴딧)의 탐욕 정책. 학습 = 2016-09 ~ 2026-08 전 사건.
평가는 이 파일에서 하지 않는다(30 독립 표본 · 전진). 진단 = 특성 중요도 · 5 겹 시간 블록 교차검증 상관.
사용: ``uv run --with numba --with lightgbm --extra pg python research/experiments/rl32.py --out <dir>``
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "src"]
import research.experiments.leader21 as L  # noqa: E402
import research.paper.leader_paper as P  # noqa: E402

ARM = "leader_P"
TRAIN = ("2016-09-01", "2026-08-31")
R_CLIP = (-2.0, 20.0)
LGB = {"objective": "regression", "num_leaves": 15, "min_data_in_leaf": 200, "learning_rate": 0.03, "feature_fraction": 0.8, "verbose": -1}
ROUNDS, SEEDS = 300, (0, 1, 2, 3, 4)
FEATS = ("rs", "d_ma50", "d_ma150", "d_ma200", "stop_dist", "range20", "tv_ratio50", "log_adv", "d_hi252", "d_lo252",
         "r20", "r60", "r120", "eps_yoy", "breadth", "regime_dist")


def build(con) -> dict:
    """모의매매 경로 그대로(leader_paper.load_db + leader21.build + 공식 KOSPI 200일선 국면 덮어쓰기, leader_P 정문 인자)."""
    price = P.ARM_PRICE[ARM]
    days, codes, O, H, L_, C, TR, cost, mkt = P.load_db(con, session=price["session"], unpatched=price["unpatched"])  # noqa: E741
    D = L.build(days, codes, O, H, L_, C, TR, cost, mkt, con, pit_shares=True, krx_patch=True, raw_close=True,
                session=price["session"], aftermarket_fix=price["aftermarket_fix"], unpatched=price["unpatched"])
    from swing_it.storage import read_index
    kx = read_index(con, "001", cols=("date", "close"), with_level=False, order_by_date=True, parse_dates=True,
                    proxy="시총 가중 대용(leader21.build regime_on) — 지수 200일선이 생긴 날부터만 덮어쓴다").set_index("date")["close"]
    kma = kx.rolling(200, min_periods=200).mean()
    for d_, v in (kx >= kma).where(kma.notna()).dropna().items():
        if d_ in days:
            D["regime_on"][days.get_loc(d_)] = bool(v)
    load_tv(D, con)
    return D


def load_tv(D: dict, con) -> None:
    """특성 tv_ratio50 용 일 거래대금(백만원) 패널 — 엔진 D 에 없어서 같은 정문 인자로 따로 읽는다."""
    from swing_it.storage import read_prices
    price = P.ARM_PRICE[ARM]
    tv = read_prices(con, session=price["session"], cols=("code", "date", "trade_value"), start="2016-08-01", parse_dates=True,
                     unpatched=price["unpatched"])
    D["tv"] = tv.pivot_table(index="date", columns="code", values="trade_value", aggfunc="last").reindex(
        index=D["days"], columns=D["codes"]).to_numpy(float).T


def cand_mask(D: dict, cfg: dict) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """leader21.simulate 의 leader_P 후보 마스크를 그대로(종가 진입 체결 가능 · 국면 켬 포함) → (mask, low10, sdist)."""
    C, L_ = D["C"], D["L"]
    cand = D["cand"] & D["brk"][cfg["N"]] & (D["rs"] >= cfg["theta"])
    prev20 = pd.DataFrame(C.T).shift(1).rolling(20, min_periods=20).max().to_numpy().T
    low10 = pd.DataFrame(L_.T).rolling(10, min_periods=10).min().to_numpy().T
    with np.errstate(invalid="ignore", divide="ignore"):
        cand = cand & (C <= prev20 * (1 + cfg["chase_max"]))
        sdist = 1.0 - low10 / C
        cand = cand & (sdist > 0) & (sdist <= cfg["stop_max"])
        pc = np.concatenate([np.full((C.shape[0], 1), np.nan), C[:, :-1]], axis=1)
        lu = np.isfinite(pc) & (C >= pc * L.LU)
    ok = cand & D["TR"] & np.isfinite(C) & (C > 0) & ~lu & D["regime_on"][None, :]
    return ok, low10, sdist


def single_sim(D: dict, cfg: dict, ci: int, t: int, stop: float, sd: float, cost_mult: float = 2.0) -> dict:
    """사건 하나를 혼자 산 결과 — leader21.simulate(leader_P)의 체결 규약을 한 종목으로(비중은 무관, 단위 금액 같음)."""
    O, L_, C, TR = D["O"], D["L"], D["C"], D["TR"]  # noqa: E741
    mae, hi20, reg = D["ma"][cfg["ma_exit"]], D["hi20"], D["regime_on"]
    pos_cap = cfg["pos_cap"]
    uw = min(pos_cap, cfg["risk"] / sd)
    mu = max(1, min(3, int(pos_cap / uw + 1e-9)))
    cl = C[ci, t]
    qty, cost_b, units, last = 1.0 / cl, 1.0, 1, cl
    p_sell, p_add, nan_run = None, False, 0
    T = C.shape[1]
    for u in range(t + 1, T):
        if p_sell is not None:
            px = O[ci, u] if np.isfinite(O[ci, u]) else last
            return _fin(D, ci, u, px, qty, cost_b, units, sd, t, p_sell, cost_mult)
        if p_add and np.isfinite(O[ci, u]) and TR[ci, u]:
            qty += 1.0 / O[ci, u]
            cost_b += 1.0
            units += 1
        p_add = False
        if np.isfinite(L_[ci, u]) and L_[ci, u] <= stop:
            px = min(O[ci, u], stop) if np.isfinite(O[ci, u]) else stop
            return _fin(D, ci, u, px, qty, cost_b, units, sd, t, "hard_stop", cost_mult)
        c_ = C[ci, u]
        if not np.isfinite(c_):
            nan_run += 1
            if nan_run >= 5:
                return _fin(D, ci, u, last, qty, cost_b, units, sd, t, "no_data", cost_mult)
            continue
        nan_run, last = 0, c_
        if np.isfinite(mae[ci, u]) and c_ < mae[ci, u]:
            p_sell = "ma_exit"
        elif cfg["pyramid"] and units < mu and c_ >= cl * (1 + 0.10 * units) and hi20[ci, u]:
            p_add = True
        if not reg[u]:
            p_sell, p_add = "regime", False
    r = _fin(D, ci, T - 1, last, qty, cost_b, units, sd, t, "end", cost_mult)
    r["censored"] = True
    return r


def _fin(D, ci, u, px, qty, cost_b, units, sd, t, why, cost_mult) -> dict:
    c_rt = D["cost"][D["mkt"][ci], min(u, D["cost"].shape[1] - 1)] * cost_mult
    ret = px * (1 - c_rt) * qty / cost_b - 1
    return {"exit_t": u, "why": why, "ret": float(ret), "units": units, "R": float(ret * units / sd), "censored": False}


def features(D: dict, ci: np.ndarray, t: np.ndarray, sdist: np.ndarray, con) -> pd.DataFrame:
    from swing_it.features.eps_asof import eps_yoy
    C, H, L_ = D["C"], D["H"], D["L"]
    ma = D["ma"]
    hi252 = L._roll(pd.DataFrame(H.T), 252, "max")
    lo252 = L._roll(pd.DataFrame(L_.T), 252, "min")
    hi20 = L._roll(pd.DataFrame(H.T), 20, "max")
    lo20 = L._roll(pd.DataFrame(L_.T), 20, "min")
    tv50 = L._roll(pd.DataFrame(D["tv"].T), 50, "mean")

    def back(k):
        with np.errstate(invalid="ignore", divide="ignore"):
            return C[ci, t] / C[ci, np.maximum(t - k, 0)] - 1

    above = (C > ma[200]) & D["uni"] & np.isfinite(ma[200])
    breadth = above.sum(0) / np.maximum((D["uni"] & np.isfinite(ma[200])).sum(0), 1)
    cwi = np.cumprod(1 + D["cw"])
    rdist = cwi / pd.Series(cwi).rolling(200, min_periods=200).mean().to_numpy() - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        f = pd.DataFrame({
            "rs": D["rs"][ci, t], "d_ma50": C[ci, t] / ma[50][ci, t] - 1, "d_ma150": C[ci, t] / ma[150][ci, t] - 1,
            "d_ma200": C[ci, t] / ma[200][ci, t] - 1, "stop_dist": sdist[ci, t], "range20": hi20[ci, t] / lo20[ci, t] - 1,
            "tv_ratio50": D["tv"][ci, t] / tv50[ci, t], "log_adv": np.log(np.maximum(D["adv"][ci, t], 1e-9)),
            "d_hi252": C[ci, t] / hi252[ci, t] - 1, "d_lo252": C[ci, t] / lo252[ci, t] - 1,
            "r20": back(20), "r60": back(60), "r120": back(120),
            "eps_yoy": eps_yoy(D["codes"], D["days"], ci, t, con), "breadth": breadth[t], "regime_dist": rdist[t]})
    return f


MODEL_DIR = "/home/young/.cache/cc-swing/models/rl32"      # 레포 밖(적합 모델 = 배포북 성격, 공개 레포 금지) — SHA256SUMS 로 고정


def score_array(D: dict, cfg: dict, con, model_dir: str = MODEL_DIR) -> np.ndarray:
    """leader_P 후보 칸마다 RL-1 예측 R(5 시드 평균), 나머지 NaN. 모델 파일 해시가 SHA256SUMS 와 다르면 멈춘다."""
    import hashlib

    import lightgbm as lgb
    want = dict(ln.split()[::-1] for ln in open(os.path.join(model_dir, "SHA256SUMS")) if ln.strip())
    for f, h in want.items():
        if hashlib.sha256(open(os.path.join(model_dir, f), "rb").read()).hexdigest() != h:
            raise SystemExit(f"RL-1 모델 해시 불일치: {f}")
    if "tv" not in D:
        load_tv(D, con)
    mask, _, sdist = cand_mask(D, cfg)
    ci, t = np.nonzero(mask)
    X = features(D, ci, t, sdist, con)[list(FEATS)].to_numpy(float)
    pr = np.mean([lgb.Booster(model_file=os.path.join(model_dir, f"model_seed{s}.txt")).predict(X) for s in SEEDS], axis=0)
    out = np.full(D["C"].shape, np.nan)
    out[ci, t] = pr
    return out


def main() -> None:
    import lightgbm as lgb
    import psycopg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    con = psycopg2.connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    D = build(con)
    cfg = P.ARMS[ARM]
    mask, low10, sdist = cand_mask(D, cfg)
    days = D["days"]
    lo, hi = days.searchsorted(pd.Timestamp(TRAIN[0])), days.searchsorted(pd.Timestamp(TRAIN[1]), side="right")
    mask[:, :lo] = False
    mask[:, hi:] = False
    ci, t = np.nonzero(mask)
    # 보충 1 ⑥: backtest 의 모든 leader_P 진입이 사건 집합에 있어야 한다
    _, tr = L.simulate(D, cfg, 2.0, end=str(days[-1].date()), start=TRAIN[0])
    tr = tr[(pd.to_datetime(tr["entry"]) >= TRAIN[0]) & (pd.to_datetime(tr["entry"]) <= TRAIN[1])]
    evset = {(D["codes"][i], str(days[s].date())) for i, s in zip(ci, t, strict=True)}
    miss = [(c, e) for c, e in zip(tr["code"], tr["entry"], strict=True) if (c, e) not in evset]
    diag = {"events": int(len(ci)), "backtest_entries": int(len(tr)), "missing_entries": len(miss), "missing_examples": miss[:10]}
    print(json.dumps(diag, ensure_ascii=False))
    if miss:
        json.dump(diag, open(os.path.join(a.out, "diag.json"), "w"), ensure_ascii=False, indent=1)
        sys.exit(3)
    rows = [single_sim(D, cfg, int(i), int(s), float(low10[i, s]), float(sdist[i, s])) for i, s in zip(ci, t, strict=True)]
    ev = pd.DataFrame(rows)
    ev.insert(0, "code", [D["codes"][i] for i in ci])
    ev.insert(1, "date", days[t])
    F = features(D, ci, t, sdist, con)
    ev = pd.concat([ev, F.reset_index(drop=True)], axis=1)
    ev["y"] = ev["R"].clip(*R_CLIP)
    ev.to_parquet(os.path.join(a.out, "events.parquet"))
    X, y = ev[list(FEATS)].to_numpy(float), ev["y"].to_numpy(float)
    # 진단: 5 겹 시간 블록 교차검증(판정 무관)
    blocks = np.array_split(np.argsort(ev["date"].to_numpy()), 5)
    cv = []
    for k, te in enumerate(blocks):
        trn = np.setdiff1d(np.arange(len(ev)), te)
        pr = np.mean([lgb.train({**LGB, "seed": s}, lgb.Dataset(X[trn], y[trn]), ROUNDS).predict(X[te]) for s in SEEDS], axis=0)
        cv.append({"block": k, "from": str(ev["date"].iloc[te].min().date()), "to": str(ev["date"].iloc[te].max().date()),
                   "n": int(len(te)), "pearson": float(np.corrcoef(pr, y[te])[0, 1]),
                   "spearman": float(pd.Series(pr).corr(pd.Series(y[te]), method="spearman"))})
    models = [lgb.train({**LGB, "seed": s}, lgb.Dataset(X, y, feature_name=list(FEATS)), ROUNDS) for s in SEEDS]
    for s, m in zip(SEEDS, models, strict=True):
        m.save_model(os.path.join(a.out, f"model_seed{s}.txt"))
    imp = pd.DataFrame({f"seed{s}": m.feature_importance("gain") for s, m in zip(SEEDS, models, strict=True)}, index=list(FEATS)).mean(1)
    out = {**diag, "n_train": int(len(ev)), "censored": int(ev["censored"].sum()), "R_mean": float(ev["R"].mean()),
           "R_clip_share": float(((ev["R"] < R_CLIP[0]) | (ev["R"] > R_CLIP[1])).mean()), "cv": cv,
           "importance_gain": (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict(), "feats": list(FEATS), "lgb": LGB}
    json.dump(out, open(os.path.join(a.out, "train_summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("n_train", "censored", "R_mean", "R_clip_share", "cv", "importance_gain")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
