"""골든 HMM 모듈 — research.signals.golden_hmm (합성 자료만, DB 없음)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hmmlearn")

from research.signals import golden_hmm as GH  # noqa: E402
from research.signals import ma_quad as MQ  # noqa: E402

DAYS = pd.bdate_range("2010-01-04", periods=700)


def _prices(n_codes=3, n_days=700, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    rows = [np.log(100) + 0.15 * np.sin(2 * np.pi * t / (170 + 25 * i)) + np.cumsum(rng.normal(0, 0.015, n_days)) for i in range(n_codes)]
    return np.exp(np.array(rows))


def _params(S=2, seed=0):
    """손으로 정한 HMM(표준화 공간) — 상태별 평균·공분산이 뚜렷이 다르게."""
    rng = np.random.default_rng(seed)
    means = np.array([[0.8, 0.5, 0.3], [-0.8, -0.5, -0.3], [0.0, 1.0, -1.0]])[:S]
    covars = np.stack([np.eye(3) * (0.5 + 0.3 * i) + 0.1 for i in range(S)])
    A = np.full((S, S), 0.05 / max(S - 1, 1)) + np.eye(S) * (0.95 - 0.05 / max(S - 1, 1)) if S > 1 else np.ones((1, 1))
    A = A / A.sum(axis=1, keepdims=True)
    return GH.HMMParams(startprob=np.full(S, 1.0 / S), transmat=A, means=means, covars=covars,
                        loc=rng.normal(0, 0.1, 3), scale=np.array([1.0, 0.1, 0.02]))


def _obs_panel(C):
    z, g, dg = GH.observations(C)
    return np.stack([z, g, dg], axis=-1)            # (n_codes, n_days, 3)


# --------------------------------------------------------------------------- 관측

def test_observations_definition():
    C = _prices()
    z, g, dg = GH.observations(C)
    y = np.log(C)
    sig = MQ.rolling_sigma(y, 60)
    r = np.full_like(y, np.nan)
    r[:, 1:] = np.diff(y, axis=1)
    assert np.allclose(z, r / sig, equal_nan=True)
    ma50 = pd.DataFrame(C.T).rolling(50, min_periods=50).mean().to_numpy().T
    ma200 = pd.DataFrame(C.T).rolling(200, min_periods=200).mean().to_numpy().T
    gg = np.log(ma50 / ma200)
    assert np.allclose(g, gg, equal_nan=True)
    exp_dg = np.full_like(gg, np.nan)
    exp_dg[:, 5:] = gg[:, 5:] - gg[:, :-5]
    assert np.allclose(dg, exp_dg, equal_nan=True)
    assert np.isnan(g[:, :199]).all() and np.isfinite(g[:, 199:]).all()
    assert np.isnan(dg[:, :204]).all() and np.isfinite(dg[:, 204:]).all()


# --------------------------------------------------------------------------- 순방향 필터 (Review Focus 1)

def test_forward_filter_matches_hmmlearn_last_step():
    """t 의 필터 사후 = hmmlearn 이 X[:t+1] 만 보고 낸 마지막 줄(마지막 줄은 평활 = 필터)."""
    P = _params(3)
    rng = np.random.default_rng(1)
    X = P.loc + P.scale * rng.normal(size=(120, 3))
    post = GH.forward_filter(P, X)
    hm = P.to_hmmlearn()
    Xs = (X - P.loc) / P.scale
    for t in (0, 1, 7, 50, 119):
        ref = hm.predict_proba(Xs[: t + 1])[-1]
        assert np.allclose(post[t], ref, atol=1e-10), t
    assert np.allclose(post.sum(axis=1), 1.0)


def test_filter_ignores_future():
    """t 뒤 관측(그리고 t 뒤 가격)을 바꿔도 t 까지의 필터 사후·관측·교차 확률이 비트 동일."""
    P = _params(3)
    C = _prices(seed=2)
    t = 450
    C2 = C.copy()
    C2[:, t + 1 :] *= np.exp(np.random.default_rng(9).normal(0, 0.2, C2[:, t + 1 :].shape))
    Xa, Xb = _obs_panel(C), _obs_panel(C2)
    assert np.array_equal(Xa[:, : t + 1], Xb[:, : t + 1], equal_nan=True)
    assert not np.array_equal(Xa[:, t + 1 :], Xb[:, t + 1 :], equal_nan=True)
    pa, pb = GH.forward_filter_panel(P, Xa), GH.forward_filter_panel(P, Xb)
    assert np.array_equal(pa[:, : t + 1], pb[:, : t + 1], equal_nan=True)
    assert not np.array_equal(pa[:, t + 1 :], pb[:, t + 1 :], equal_nan=True)
    # 한 수열판도 같은 성질
    for c in range(3):
        assert np.array_equal(GH.forward_filter(P, Xa[c])[: t + 1], GH.forward_filter(P, Xb[c])[: t + 1], equal_nan=True)
    sa = MQ.rolling_sigma(np.log(C), 60)
    sb = MQ.rolling_sigma(np.log(C2), 60)
    ci = np.repeat(np.arange(3), t + 1 - 260)
    s = np.tile(np.arange(260, t + 1), 3)                     # t 까지의 사건 전부(t 포함)
    codes = ["000001", "000002", "000003"]
    kw = dict(ci=ci, s=s, k=10, codes=codes, days=DAYS)
    qa = GH.hmm_cross_probs(P, C, sa, pa, **kw)
    qb = GH.hmm_cross_probs(P, C2, sb, pb, **kw)
    assert np.isfinite(qa).all() and np.array_equal(qa, qb)
    mid = np.flatnonzero((qa > 0) & (qa < 1))
    assert len(mid) >= 10                                      # 0·1 이 아닌 사건이 있어야 시험이 된다
    # 사건마다 바로 그 t0 뒤를 흔든다(경계 누출 — 사후 t0+1·평활 — 이 모든 사건에서 걸리게)
    rng = np.random.default_rng(7)
    for j in mid[:: max(len(mid) // 15, 1)]:
        c0, t0 = int(ci[j]), int(s[j])
        C3 = C.copy()
        C3[:, t0 + 1 :] *= np.exp(rng.normal(0, 0.2, C3[:, t0 + 1 :].shape))
        p3 = GH.forward_filter_panel(P, _obs_panel(C3))
        s3 = MQ.rolling_sigma(np.log(C3), 60)
        q3 = GH.hmm_cross_probs(P, C3, s3, p3, np.array([c0]), np.array([t0]), 10, codes=codes, days=DAYS)
        assert q3[0] == qa[j], (c0, t0)
        assert not np.array_equal(p3[c0, t0 + 1], pa[c0, t0 + 1])
        # 검정력: 하루 뒤 사후(미래 관측 포함)를 넣으면 값이 달라지는 사건이 있다
    shifted = pb.copy()
    shifted[:, :-1] = pb[:, 1:]
    assert not np.array_equal(GH.hmm_cross_probs(P, C2, sb, shifted, **kw), qa)


def test_panel_filter_equals_per_series_and_handles_nan():
    P = _params(2)
    X = _obs_panel(_prices(seed=3))
    X[1, 300, 0] = np.nan                         # 한 칸 결측 → 그 날은 예측만(관측 갱신 없음)
    X[2, 250:260] = np.nan
    pp = GH.forward_filter_panel(P, X)
    for c in range(3):
        assert np.allclose(pp[c], GH.forward_filter(P, X[c]), atol=1e-12, equal_nan=True)
    # 결측 날 = 전날 사후 × 전이행렬
    assert np.allclose(pp[1, 300], pp[1, 299] @ P.transmat)
    # 첫 유한 관측 전은 NaN, 첫 유한 관측 날은 startprob 에서 시작
    first = int(np.flatnonzero(np.isfinite(X[0]).all(axis=1))[0])
    assert np.isnan(pp[0, :first]).all() and np.isfinite(pp[0, first:]).all()
    lb = GH.log_emission(P, X[0, first][None, :])[0]
    w = P.startprob * np.exp(lb - lb.max())
    assert np.allclose(pp[0, first], w / w.sum())


def test_module_never_smooths():
    """평활(미래 사용) API 를 모듈이 부르지 않는다 — predict_proba·score_samples·decode·predict."""
    src = Path(GH.__file__).read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    for bad in (".predict_proba(", ".score_samples(", ".decode(", ".predict("):
        assert bad not in code, bad


# --------------------------------------------------------------------------- 경로 교차 확률

def test_cross_probs_zero_sigma_match_realized():
    """σ=0 → 경로 종가 = C_t 상수. 확률 = 같은 상수 미래를 넣은 실현값(Review Focus 2 의 HMM 판)."""
    P = _params(2)
    C = _prices(seed=4)
    n = C.shape[1]
    sig = np.zeros(C.shape)
    post = GH.forward_filter_panel(P, _obs_panel(C))
    s = np.arange(260, n - 16, 4)
    ci = np.arange(len(s)) % 3
    codes = ["000001", "000002", "000003"]
    for k in (5, 10, 15):
        p = GH.hmm_cross_probs(P, C, sig, post, ci, s, k, codes=codes, days=pd.bdate_range("2010-01-04", periods=n))
        exp = np.empty(len(s))
        for j, (c, t) in enumerate(zip(ci, s)):
            Cf = C[c : c + 1].copy()
            Cf[0, t + 1 :] = C[c, t]                  # 미래 = 평평(σ=0 경로)
            exp[j] = MQ.cross_realized(Cf, np.array([0]), np.array([t]), k)[0]
        assert np.array_equal(p, exp), k
        assert 0 < exp.sum() < len(exp)


def test_hmm_cross_probs_seed_and_nan_rules():
    P = _params(3)
    C = _prices(seed=5)
    sig = MQ.rolling_sigma(np.log(C), 60)
    post = GH.forward_filter_panel(P, _obs_panel(C))
    codes = ["000001", "000002", "000003"]
    ci, s = np.array([0, 1, 0]), np.array([400, 400, 500])
    a = GH.hmm_cross_probs(P, C, sig, post, ci, s, 10, codes=codes, days=DAYS)
    b = GH.hmm_cross_probs(P, C, sig, post, ci, s, 10, codes=codes, days=DAYS, chunk=1)
    assert np.array_equal(a, b)                                     # 덩어리 크기 무관(사건별 시드)
    assert np.allclose(a * 200, np.round(a * 200)) and ((a >= 0) & (a <= 1)).all()
    # 사건 키 = code|date|k|"hmm": 순서를 바꿔도 사건별 값 같음
    r = GH.hmm_cross_probs(P, C, sig, post, ci[::-1], s[::-1], 10, codes=codes, days=DAYS)
    assert np.array_equal(r[::-1], a)
    assert GH.path_seed("000001", DAYS[400], 10) != GH.path_seed("000001", DAYS[400], 5)
    # 사후 NaN(첫 관측 전)·σ NaN·알려진 창 NaN → NaN
    post2 = post.copy()
    post2[0, 400] = np.nan
    sig2 = sig.copy()
    sig2[1, 400] = np.nan
    C2 = C.copy()
    C2[0, 450] = np.nan
    out = GH.hmm_cross_probs(P, C2, sig2, post2, np.array([0, 1, 0]), np.array([400, 400, 500]), 10, codes=codes, days=DAYS)
    assert np.isnan(out).all()


def test_s1_equals_gaussian_rw():
    """S=1 → z iid N(μ, σ²) 랜덤워크. 독립 난수로 만든 가우시안 풀 + cross_probs 와 통계 허용오차 안에서 같다(Review Focus 4)."""
    P1 = GH.HMMParams(startprob=np.ones(1), transmat=np.ones((1, 1)), means=np.array([[0.4, 0.0, 0.0]]),
                      covars=np.array([np.eye(3) * 1.0 + 0.2]), loc=np.array([0.05, 0.0, 0.0]), scale=np.array([0.9, 0.1, 0.02]))
    mu = P1.loc[0] + P1.scale[0] * P1.means[0, 0]
    sd = P1.scale[0] * np.sqrt(P1.covars[0, 0, 0])
    assert np.allclose(GH.z_marginals(P1), [[mu, sd]])
    C = _prices(seed=6)
    n = C.shape[1]
    sig = np.full(C.shape, 0.02)
    post = np.ones(C.shape + (1,))
    codes = ["000001", "000002", "000003"]
    k, M = 10, 4000
    # MA50 이 MA200 바로 아래인 사건들(확률이 0·1 이 아닌 자리)
    st = MQ.ma_state(C)
    gap = 1 - st["ma"][50] / st["ma"][200]
    ci, s = np.nonzero((gap > 0) & (gap < 0.03) & (np.arange(n) < n - k - 1)[None, :])
    ci, s = ci[::3][:80], s[::3][:80]
    p_hmm = GH.hmm_cross_probs(P1, C, sig, post, ci, s, k, M=M, codes=codes, days=DAYS)
    rng = np.random.default_rng(123)
    pool = np.cumsum(rng.normal(mu, sd, size=(M, k)), axis=1)
    fcl = np.repeat(np.log(C)[:, :, None], k, axis=2)
    idx = np.tile(np.arange(M), (len(ci), 1))
    p_rw = MQ.cross_probs(C, fcl, sig, pool, idx, ci, s, k)
    assert np.isfinite(p_hmm).all() and ((p_hmm > 0.02) & (p_hmm < 0.98)).sum() >= 8
    assert np.max(np.abs(p_hmm - p_rw)) < 0.05                       # 몬테카를로 오차(M=4000: 표준오차 ≤ 0.011)
    assert abs(np.mean(p_hmm - p_rw)) < 0.01


# --------------------------------------------------------------------------- 학습

def _sample_two_regime(n_seq=25, T=300, seed=0):
    rng = np.random.default_rng(seed)
    mus = np.array([[1.0, 0.5, -0.5], [-1.0, -0.5, 0.5]])
    covs = np.array([np.eye(3) * 0.3, np.eye(3) * 1.5])
    A = np.array([[0.97, 0.03], [0.04, 0.96]])
    out = []
    for _ in range(n_seq):
        st = rng.integers(2)
        x = np.empty((T, 3))
        for t in range(T):
            st = st if t == 0 else rng.choice(2, p=A[st])
            x[t] = rng.multivariate_normal(mus[st], covs[st])
        out.append(x * np.array([1.0, 0.1, 0.02]) + np.array([0.0, -0.05, 0.0]))   # 실제 관측 크기(g·Δg 작다)
    return out


def test_bic_picks_true_S():
    X = _sample_two_regime()
    fit = GH.fit_hmm(X, S_range=(1, 4), seed=0)
    assert fit["S"] == 2
    assert set(fit["bic_table"]) == {1, 2, 3, 4}
    assert min(fit["bic_table"], key=fit["bic_table"].get) == 2
    m = fit["model"]
    assert m.means.shape == (2, 3) and m.covars.shape == (2, 3, 3) and np.allclose(m.transmat.sum(axis=1), 1)
    assert set(fit["models"]) == {1, 2, 3, 4} and fit["models"][1].means.shape == (1, 3)
    for info in fit["fits"].values():
        assert info["n_iter"] <= 200 and info["n_obs"] == 25 * 300
    # 결정적(random_state=0) — 다중 스레드 BLAS 반올림(1e-14) 밖에서는 같다
    fit2 = GH.fit_hmm(X, S_range=(2, 2), seed=0)
    assert np.allclose(fit2["models"][2].transmat, fit["models"][2].transmat, rtol=1e-9, atol=1e-12)


def test_fit_parallel_equals_serial():
    X = _sample_two_regime(n_seq=6, T=200, seed=1)
    a = GH.fit_hmm(X, S_range=(1, 3), seed=0, workers=1)
    b = GH.fit_hmm(X, S_range=(1, 3), seed=0, workers=3)
    assert a["S"] == b["S"]                                        # 작업자 수는 BLAS 반올림(1e-14)만 바꾼다
    for S in (1, 2, 3):
        assert np.isclose(a["bic_table"][S], b["bic_table"][S], rtol=1e-10)
        assert np.allclose(a["models"][S].means, b["models"][S].means, rtol=1e-8, atol=1e-10)


def test_train_sequences_universe_runs():
    """학습 수열 = 유니버스 ∧ 관측 유한 연속 구간, 길이 ≥ min_len."""
    n_days = 300
    X = np.random.default_rng(0).normal(size=(2, n_days, 3))
    U = np.zeros((2, n_days), bool)
    U[0, 10:100] = True                          # 90 → 들어감
    U[0, 150:200] = True                         # 50 → 빠짐(< 60)
    U[1, 0:300] = True
    X[1, 120, 2] = np.nan                        # 관측 결측이 구간을 자른다: [0,120) 120 · [121,300) 179
    seqs, info = GH.train_sequences(X, U, min_len=60)
    lens = sorted(len(q) for q in seqs)
    assert lens == [90, 120, 179]
    assert all(np.isfinite(q).all() for q in seqs)
    assert info["n_seq"] == 3 and info["n_obs"] == 389
    assert info["t_first"] == 0 and info["t_last"] == 299        # 쓴 수열의 첫·끝 행(패널 끝이 아니라)
    U[1] = False
    _, info2 = GH.train_sequences(X, U, min_len=60)
    assert info2["t_first"] == 10 and info2["t_last"] == 99
    _, none = GH.train_sequences(X, np.zeros_like(U), min_len=60)
    assert none["n_seq"] == 0 and none["t_first"] is None and none["t_last"] is None


def test_fit_records_converged_and_cap_separately():
    """hmmlearn monitor_.converged 는 반복 상한에 닿아도 True — 수렴은 history 로 직접, 상한 도달은 따로 적는다."""
    X = _sample_two_regime(n_seq=6, T=200, seed=2)
    cap = GH.fit_hmm(X, S_range=(2, 2), seed=0, n_iter=2)["fits"][2]
    assert cap["n_iter"] == 2 and cap["hit_cap"] is True
    assert cap["converged"] is False                              # 2 반복에선 개선폭이 tol 보다 크다
    assert cap["delta_last"] >= cap["tol"]
    free = GH.fit_hmm(X, S_range=(1, 1), seed=0, n_iter=200)["fits"][1]
    assert free["converged"] is True and free["hit_cap"] is False and free["n_iter"] < 200
    assert free["delta_last"] < free["tol"]
