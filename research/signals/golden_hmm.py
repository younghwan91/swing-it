"""MA50 → MA200 골든 확률 — 국면 전환 가우시안 HMM·**순방향 필터만**·경로 교차 확률 (사양 2026-11-14-golden-hmm-design §2).

관측 x_t = [z_t = r_t/σ_t, g_t = log(MA50/MA200), Δ5 g_t] (σ_t = t 까지(포함) 60 로그 수익 표준편차 — :func:`ma_quad.rolling_sigma`).
학습: hmmlearn ``GaussianHMM(covariance_type="full", random_state=seed, n_iter≤200)`` 을 S ∈ S_range 마다 맞추고 BIC 최소를 고른다.
관측은 학습 자료의 차원별 평균·표준편차로 표준화한 뒤 맞춘다(아핀 변환 — 우도·BIC 순위 불변; hmmlearn 의 min_covar=1e-3·공분산 사전
1e-2 가 작은 차원 Δ5g(분산 ~1e-4)를 덮지 않게). 상태 사후확률은 t 까지 관측만 쓰는 **순방향 필터**(:func:`forward_filter`) 로만 낸다 —
hmmlearn 의 평활(predict_proba·score_samples, 미래 관측 사용)은 이 모듈에서 부르지 않는다(tests 가 소스를 검사한다).

경로(:func:`hmm_cross_probs`): 사건마다 시드 blake2b(code|date|k|"hmm"), s_t ~ 필터 사후, s_{t+h} ~ 전이행렬,
z_{t+h} ~ N(μ_s[z], Σ_s[z,z])(원 단위 주변분포), 경로 종가 = C_t·exp(σ_t·Σ_{j≤h} z_j). MA_short·MA_long = 알려진 과거 종가 + 경로 종가
(:func:`ma_quad.cross_kernel_events` — q0 의 :func:`ma_quad.cross_probs` 와 같은 산수).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.signals import ma_quad as MQ
from research.signals.ma_quad import _jit

SIGMA_W = 60
SHORT, LONG = 50, 200
DG_LAG = 5
OBS_NAMES = ("z", "g", "dg")


# =============================================================================================== 관측

def observations(C: np.ndarray, sigma_w: int = SIGMA_W, short: int = SHORT, long: int = LONG, lag: int = DG_LAG):
    """종가 패널 (n_codes, n_days) → (z, g, dg) 각 (n_codes, n_days) float64. 전부 t 까지(포함)로만 정해진다(인과).

    z = r_t / σ_t(σ ≤ 0·비유한이면 NaN), g = log(MA50/MA200)(pandas rolling, 창에 NaN 이면 NaN), dg = g_t − g_{t−lag}.
    """
    C = np.asarray(C, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(C > 0, np.log(C), np.nan)
    r = np.full_like(y, np.nan)
    r[:, 1:] = np.diff(y, axis=1)
    sig = MQ.rolling_sigma(y, sigma_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(sig) & (sig > 0), r / sig, np.nan)
        g = np.log(MQ._rolling_mean(C, short) / MQ._rolling_mean(C, long))
    g = np.where(np.isfinite(g), g, np.nan)
    dg = np.full_like(g, np.nan)
    dg[:, lag:] = g[:, lag:] - g[:, :-lag]
    return z, g, dg


def stack_obs(z, g, dg) -> np.ndarray:
    """(n_codes, n_days, 3) — 차원 순서 OBS_NAMES."""
    return np.stack([z, g, dg], axis=-1)


def train_sequences(X: np.ndarray, U: np.ndarray, min_len: int = 60):
    """학습 수열 = 유니버스 U ∧ 관측 3차원 유한인 연속 구간 중 길이 ≥ min_len. (수열 목록, 정보)."""
    X = np.asarray(X, dtype=np.float64)
    ok = np.asarray(U, dtype=bool) & np.isfinite(X).all(axis=-1)
    seqs = []
    for c in range(ok.shape[0]):
        row = ok[c].astype(np.int8)
        d = np.diff(np.r_[0, row, 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= min_len:
                seqs.append(np.ascontiguousarray(X[c, a:b]))
    return seqs, {"n_seq": len(seqs), "n_obs": int(sum(len(q) for q in seqs)), "min_len": int(min_len)}


# =============================================================================================== 모형

@dataclass
class HMMParams:
    """표준화 공간의 가우시안 HMM 매개변수 + 표준화(loc, scale). 관측 x 원 단위 → (x − loc) / scale."""

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    loc: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        for f in ("startprob", "transmat", "means", "covars", "loc", "scale"):
            setattr(self, f, np.asarray(getattr(self, f), dtype=np.float64))
        S = len(self.startprob)
        assert self.transmat.shape == (S, S) and self.means.shape == (S, 3) and self.covars.shape == (S, 3, 3)
        assert np.allclose(self.transmat.sum(axis=1), 1.0) and np.isclose(self.startprob.sum(), 1.0)
        assert (self.scale > 0).all()

    @property
    def S(self) -> int:
        return len(self.startprob)

    def raw_means(self) -> np.ndarray:
        return self.loc + self.scale * self.means

    def raw_covars(self) -> np.ndarray:
        return self.covars * np.outer(self.scale, self.scale)[None]

    def to_hmmlearn(self):
        """같은 매개변수의 hmmlearn GaussianHMM(표준화 공간) — 시험 대조용."""
        from hmmlearn.hmm import GaussianHMM

        hm = GaussianHMM(n_components=self.S, covariance_type="full", init_params="", params="")
        hm.n_features = 3
        hm.startprob_, hm.transmat_, hm.means_, hm.covars_ = self.startprob, self.transmat, self.means, self.covars
        return hm

    def report(self) -> dict:
        """보고용(원 단위 상태별 μ·Σ, 전이행렬, 시작 확률)."""
        return {"S": self.S, "obs": OBS_NAMES, "startprob": self.startprob.tolist(), "transmat": self.transmat.tolist(),
                "means_raw": self.raw_means().tolist(), "covars_raw": self.raw_covars().tolist(),
                "z_marginal_mu_sd": z_marginals(self).tolist(), "loc": self.loc.tolist(), "scale": self.scale.tolist()}


def z_marginals(P: HMMParams) -> np.ndarray:
    """(S, 2) — 상태별 z 주변분포 (평균, 표준편차), 원 단위."""
    mu = P.loc[0] + P.scale[0] * P.means[:, 0]
    sd = P.scale[0] * np.sqrt(P.covars[:, 0, 0])
    return np.stack([mu, sd], axis=1)


_FIT: dict = {}


def _fit_init(Xs, lengths, seed, n_iter) -> None:
    _FIT.update(Xs=Xs, lengths=lengths, seed=seed, n_iter=n_iter)


def _fit_one(S: int):
    from hmmlearn.hmm import GaussianHMM

    Xs, lengths, seed, n_iter = _FIT["Xs"], _FIT["lengths"], _FIT["seed"], _FIT["n_iter"]
    t = time.time()
    hm = GaussianHMM(n_components=S, covariance_type="full", n_iter=n_iter, random_state=seed)
    hm.fit(Xs, lengths)
    bic = float(hm.bic(Xs, lengths))
    info = {"n_iter": int(hm.monitor_.iter), "converged": bool(hm.monitor_.converged),
            "loglik_std_space": float(hm.monitor_.history[-1]) if len(hm.monitor_.history) else float("nan"),
            "n_obs": int(len(Xs)), "n_seq": int(len(lengths)), "secs": time.time() - t}
    return S, (hm.startprob_, hm.transmat_, hm.means_, hm.covars_), bic, info


def fit_hmm(X_list, S_range=(1, 5), seed: int = 0, n_iter: int = 200, workers: int = 1) -> dict:
    """수열 목록(각 (T_i, 3) 원 단위) → S ∈ [S_range[0], S_range[1]] 마다 GaussianHMM(full) 적합, BIC 최소 S.

    돌려줌: ``S``·``bic_table`` {S: BIC(표준화 공간 — 순위는 원 단위와 같다)}·``model``(고른 S)·``models`` {S: HMMParams}·``fits``(반복 수 등).
    """
    if n_iter > 200:
        raise ValueError("n_iter ≤ 200 (사전등록)")
    X = np.concatenate([np.asarray(q, dtype=np.float64) for q in X_list], axis=0)
    if not np.isfinite(X).all():
        raise ValueError("학습 관측에 비유한 값")
    lengths = np.array([len(q) for q in X_list], dtype=np.int64)
    loc, scale = X.mean(axis=0), X.std(axis=0)
    Xs = (X - loc) / scale
    Ss = list(range(int(S_range[0]), int(S_range[1]) + 1))
    try:
        if workers > 1 and len(Ss) > 1:
            # spawn: fork 는 부모의 BLAS 스레드 풀 잠금을 물려받아 자식이 멈췄다(2026-11-14 시험에서 재현)
            with mp.get_context("spawn").Pool(min(workers, len(Ss)), initializer=_fit_init,
                                              initargs=(Xs, lengths, seed, n_iter)) as pool:
                res = pool.map(_fit_one, sorted(Ss, reverse=True))
        else:
            _fit_init(Xs, lengths, seed, n_iter)
            res = [_fit_one(S) for S in Ss]
    finally:
        _FIT.clear()
    models, bics, fits = {}, {}, {}
    for S, (sp, tm, mu, cv), bic, info in res:
        models[S] = HMMParams(startprob=sp, transmat=tm, means=mu, covars=cv, loc=loc, scale=scale)
        bics[S] = bic
        fits[S] = info
    best = min(Ss, key=lambda S: (bics[S], S))                 # 동률이면 작은 S
    return {"S": best, "bic_table": {S: bics[S] for S in Ss}, "model": models[best], "models": {S: models[S] for S in Ss},
            "fits": {S: fits[S] for S in Ss}, "loc": loc, "scale": scale}


# =============================================================================================== 순방향 필터

def log_emission(P: HMMParams, X: np.ndarray) -> np.ndarray:
    """(..., S) — 원 단위 관측 X (..., 3) 의 상태별 로그 밀도(표준화 공간, 상수 야코비안 생략 — 사후확률엔 무관). 비유한 행은 NaN."""
    X = np.asarray(X, dtype=np.float64)
    Xs = (X - P.loc) / P.scale
    out = np.empty(X.shape[:-1] + (P.S,), dtype=np.float64)
    for q in range(P.S):
        L = np.linalg.cholesky(P.covars[q])
        d = Xs - P.means[q]
        sol = np.linalg.solve(L, d.reshape(-1, 3).T).T.reshape(d.shape)
        out[..., q] = -0.5 * (sol ** 2).sum(axis=-1) - np.log(np.diag(L)).sum() - 1.5 * np.log(2 * np.pi)
    out[~np.isfinite(X).all(axis=-1)] = np.nan
    return out


def _step(alpha_prev, started, lb_t, P: HMMParams):
    """한 날의 필터 갱신(행 = 수열). 시작 전 = startprob, 관측 결측 = 예측만, 관측 있음 = 예측 × 방출 정규화."""
    S = P.S
    pred = np.zeros_like(alpha_prev)
    for q in range(S):                                            # 원소 곱 합(행끼리 독립 — BLAS 블록 무관)
        pred += alpha_prev[:, q : q + 1] * P.transmat[q][None, :]
    pred = np.where(started[:, None], pred, P.startprob[None, :])
    ok = np.isfinite(lb_t).all(axis=1)
    new = pred.copy()
    if ok.any():
        lb = lb_t[ok]
        w = pred[ok] * np.exp(lb - lb.max(axis=1, keepdims=True))
        tot = w.sum(axis=1, keepdims=True)
        good = (tot[:, 0] > 0) & np.isfinite(tot[:, 0])
        w[good] = w[good] / tot[good]
        w[~good] = pred[ok][~good]                                # 방출이 전부 밑넘침 — 예측 유지(드묾)
        new[ok] = w
    live = started | ok
    new[~live] = np.nan
    return new, live


def forward_filter_panel(P: HMMParams, X: np.ndarray) -> np.ndarray:
    """(n_codes, n_days, S) — 관측 패널 (n_codes, n_days, 3) 의 순방향 필터 사후 p(s_t | x_≤t). 첫 유한 관측 전은 NaN."""
    X = np.asarray(X, dtype=np.float64)
    lb = log_emission(P, X)
    n, T = X.shape[:2]
    post = np.full((n, T, P.S), np.nan)
    alpha = np.full((n, P.S), np.nan)
    started = np.zeros(n, dtype=bool)
    for t in range(T):
        alpha, started = _step(np.nan_to_num(alpha), started, lb[:, t], P)
        post[:, t] = alpha
    return post


def forward_filter(model: HMMParams, X: np.ndarray) -> np.ndarray:
    """(T, S) — 한 수열 (T, 3) 의 순방향 필터 사후. t 의 값은 X[:t+1] 만으로 정해진다(평활 아님)."""
    X = np.asarray(X, dtype=np.float64)
    return forward_filter_panel(model, X[None])[0]


# =============================================================================================== 경로 교차 확률

def _seed(code: str, date_str: str, k: int) -> int:
    key = f"{code}|{date_str}|{int(k)}|hmm"
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


def path_seed(code: str, date, k: int) -> int:
    """사건 시드 = blake2b(code|YYYY-MM-DD|k|"hmm") 8바이트(little)."""
    return _seed(code, f"{pd.Timestamp(date):%Y-%m-%d}", k)


@_jit
def _hmm_paths(p0, U0, U, Nz, cumA, mu, sd, out):
    """사건 j·경로 m 의 누적 z(원 단위) → out[j*M + m, h]. s_t ~ p0[j], s_{t+h} ~ 전이 cumA[s], z ~ N(mu[s], sd[s]²)."""
    n, M = U0.shape
    k = U.shape[2]
    S = p0.shape[1]
    for j in range(n):
        tot = 0.0
        for q in range(S):
            tot += p0[j, q]
        for m in range(M):
            u = U0[j, m] * tot
            acc = 0.0
            st = S - 1
            for q in range(S):
                acc += p0[j, q]
                if u < acc:
                    st = q
                    break
            cz = 0.0
            for h in range(k):
                u = U[j, m, h]
                nxt = S - 1
                for q in range(S):
                    if u < cumA[st, q]:
                        nxt = q
                        break
                st = nxt
                cz += mu[st] + sd[st] * Nz[j, m, h]
                out[j * M + m, h] = cz


def hmm_cross_probs(model: HMMParams, C, sigma, post, ci, s, k: int, short: int = SHORT, long: int = LONG, M: int = 200,
                    *, codes, days, chunk: int = 2048) -> np.ndarray:
    """(n_ev,) — 사건 (ci, s) 마다 HMM 경로 M 개 중 ∃h≤k: MA_short > MA_long 비율.

    ``post`` = :func:`forward_filter_panel` 결과(필터 사후만). 사후·σ·C_t·알려진 창 C[t−long+2..t] 중 비유한이 있으면 NaN.
    사건별 시드(:func:`path_seed`) — 사건 순서·덩어리 크기와 무관하다.
    """
    ci = np.asarray(ci, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    C = np.ascontiguousarray(C, dtype=np.float64)
    days = pd.DatetimeIndex(days)
    n = len(ci)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    assert s.min() >= 0 and s.max() < C.shape[1] and ci.min() >= 0 and ci.max() < C.shape[0], "ci/s 범위 밖"
    zm = z_marginals(model)
    mu, sd = np.ascontiguousarray(zm[:, 0]), np.ascontiguousarray(zm[:, 1])
    cumA = np.cumsum(model.transmat, axis=1)
    cumA[:, -1] = np.inf                                         # 반올림으로 1 에 못 미쳐도 마지막 상태가 받는다
    p0 = np.asarray(post[ci, s], dtype=np.float64)
    ok = np.isfinite(p0).all(axis=1) & (p0.sum(axis=1) > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        logc = np.where(C[ci, s] > 0, np.log(C[ci, s]), np.nan)
    sg = np.asarray(sigma, dtype=np.float64)[ci, s]
    dstr = days.strftime("%Y-%m-%d")
    for a in range(0, n, max(int(chunk), 1)):
        sel = np.arange(a, min(n, a + max(int(chunk), 1)))
        sel = sel[ok[sel]]
        if len(sel) == 0:
            continue
        U0 = np.empty((len(sel), M))
        U = np.empty((len(sel), M, k))
        Nz = np.empty((len(sel), M, k))
        for i, j in enumerate(sel):
            rng = np.random.default_rng(_seed(codes[ci[j]], dstr[s[j]], k))
            U0[i] = rng.random(M)
            U[i] = rng.random((M, k))
            Nz[i] = rng.standard_normal((M, k))
        pool = np.empty((len(sel) * M, k))
        _hmm_paths(np.ascontiguousarray(p0[sel]), U0, U, Nz, cumA, mu, sd, pool)
        idx = np.arange(len(sel) * M, dtype=np.int64).reshape(len(sel), M)
        fcl = np.repeat(logc[sel][:, None], k, axis=1)
        out[sel] = MQ.cross_kernel_events(C, ci[sel], s[sel], fcl, sg[sel], pool, idx, k, short, long)
    return out
