"""mtf_transition 누출 테스트 3종(PREREG ``research/logs/mtf_transition/PREREG.md`` '누출 점검') — 합성 자료, DB 없음.

1. 날짜 5개에서 이후 자료를 지운 사본으로 유니버스·up·사건·선택 설정이 같다.
2. 학습 선택이 시험 달 자료를 보지 않는다(시험 달부터 가격을 교란해도 그 폴드 학습 점수·선택이 같다) — 음성 대조: 다음 폴드는 바뀐다.
3. 일봉 청산 판정은 그날 정규장 종가까지만 쓴다(그날 뒤 일봉을 지워도 청산이 같다) — 음성 대조: 다음 날 행을 쓰면 바뀐다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")

from research.signals import mtf_transition as T  # noqa: E402

TOPN = 12
MONTHS = ["2026-09", "2026-10", "2026-11"]


def synth(seed: int = 0, n_codes: int = 24):
    rng = np.random.default_rng(seed)
    ddays = pd.bdate_range("2025-09-01", "2026-11-13")
    i0 = int(np.searchsorted(ddays, pd.Timestamp("2026-07-01")))
    codes = [f"{i:06d}" for i in range(n_codes)]
    mu = rng.normal(0.0012, 0.0015, n_codes)
    lr = rng.normal(mu[:, None], 0.02, (n_codes, len(ddays)))
    logc = np.log(10000) + np.cumsum(lr, axis=1)
    nb_ = 26
    tod = pd.to_timedelta(9 * 60 + 15 * np.arange(nb_), unit="min")
    rows = []
    for d in range(i0, len(ddays)):
        prev = np.exp(logc[:, d - 1])
        r = rng.normal(mu[:, None] / nb_, 0.02 / np.sqrt(nb_) * 1.3, (n_codes, nb_))
        gap = rng.normal(0, 0.004, n_codes)
        cl = prev[:, None] * np.exp(gap[:, None] + np.cumsum(r, axis=1))
        op = np.concatenate([(prev * np.exp(gap))[:, None], cl[:, :-1]], axis=1)
        hi = np.maximum(op, cl) * np.exp(np.abs(rng.normal(0, 0.002, cl.shape)))
        lo = np.minimum(op, cl) * np.exp(-np.abs(rng.normal(0, 0.002, cl.shape)))
        vol = rng.lognormal(8, 0.6, cl.shape)
        logc[:, d] = np.log(cl[:, -1])
        ts = ddays[d] + tod
        for i, c in enumerate(codes):
            rows.append(pd.DataFrame({"code": c, "ts": ts, "open": op[i], "high": hi[i], "low": lo[i], "close": cl[i], "volume": vol[i]}))
    m15 = pd.concat(rows, ignore_index=True)
    dc = pd.DataFrame(np.exp(logc).T, index=ddays, columns=codes)
    tv = dc * rng.lognormal(0, 0.5, dc.shape) * np.linspace(0.5, 2.0, n_codes)[None, :] / 100
    hole = (dc.index >= "2026-09-20")[:, None] & (rng.random(dc.shape) < 0.05)       # unpatched=NaN 칸 → 분봉 종가 채움 경로
    dc = dc.mask(hole)
    return m15, dc, tv, codes


def build(m15, dc, tv, codes):
    Ss = []
    for tf, mins in (("15m", 15), ("30m", 30), ("60m", 60)):
        m = m15 if mins == 15 else T.aggregate(m15, mins)
        Ss.append(T.build_tf(m, dc, tv, codes, tf, codes=codes, topn=TOPN))
    return Ss


def select_all(Ss):
    OCs, dayclose = T.prepare(Ss)
    costs = np.full((len(Ss[0]["udates"]), 2), 0.00415)
    mkt = np.zeros(len(Ss[0]["codes"]), np.int64)
    cands = T.all_candidates(Ss, OCs, costs, mkt, 2.0)
    return T.wf_select(cands, dayclose, MONTHS), OCs


@pytest.fixture(scope="module")
def data():
    return synth()


@pytest.fixture(scope="module")
def full(data):
    m15, dc, tv, codes = data
    Ss = build(m15, dc, tv, codes)
    W, OCs = select_all(Ss)
    return Ss, W, OCs


def _evset(S, hi, strict):
    """hi 앞에서 시작한 봉의 사건. strict 면 특징까지 — 잘린 사본의 마지막 봉은 '하루 마지막 봉' 표시가 달라지므로 hi 직전 봉은 뺀다."""
    ev = S["ev"]
    ts = S["days"][ev["t"]]
    ok = (ts < hi - pd.Timedelta(minutes=S["tf_min"])) if strict else (ts < hi)
    out = set()
    for k in np.flatnonzero(ok):
        key = (S["codes"][ev["ci"][k]], ts[k], bool(ev["u1"][k]), bool(ev["u2"][k]), bool(ev["u3"][k]))
        if strict:
            key += (bool(ev["spread"][k]), bool(ev["last"][k]), round(float(ev["s1"][k]), 9), round(float(np.nan_to_num(ev["relvol"][k])), 9))
        out.add(key)
    return out


def test_synth_has_events(full):
    Ss, W, _ = full
    for S in Ss:
        assert len(S["ev"]["t"]) >= 20, S["tf"]
    assert W["scores"].shape == (288, len(MONTHS))
    assert np.isfinite(W["scores"]).all() and (W["n_train"].sum(axis=0) > 0).all()


def test_truncation_invariance(data, full):
    """날짜 5개: 그날 한 시각(정시 — 세 봉 단위 모두 봉 경계) 전까지의 분봉 + 그날 전까지의 일봉만 남긴 사본에서
    유니버스·up·사건(그 시각 전 봉)·선택(그날 전에 시작한 폴드)이 같다."""
    m15, dc, tv, codes = data
    Ss, W, _ = full
    rng = np.random.default_rng(1)
    cuts = ["2026-08-20", "2026-09-10", "2026-10-02", "2026-10-21", "2026-11-04"]
    for kd in cuts:
        kday = pd.Timestamp(kd)
        hi = kday + pd.Timedelta(hours=int(rng.integers(10, 16)))
        Tt = build(m15[m15.ts < hi], dc[dc.index < kday], tv[tv.index < kday], codes)
        for F, X in zip(Ss, Tt):
            for d in X["udates"]:
                assert F["memday"][pd.Timestamp(d)] == X["memday"][pd.Timestamp(d)], (F["tf"], kd, d)
            nd = len(X["udates"])
            assert np.array_equal(F["DS"]["up_prev"][:, :, :nd], X["DS"]["up_prev"]), (F["tf"], kd)
            assert np.array_equal(F["DS"]["def_prev"][:, :nd], X["DS"]["def_prev"]), (F["tf"], kd)
            nb_ = len(X["days"])
            assert F["days"][:nb_].equals(X["days"])
            assert np.array_equal(F["MEM"][:, :nb_], X["MEM"])
            a, b = _evset(F, hi, False), _evset(X, hi, False)
            assert a == b, (F["tf"], kd, len(a ^ b))
            assert _evset(F, hi, True) == _evset(X, hi, True), (F["tf"], kd)
        folds = [f for f, mo in enumerate(MONTHS) if pd.Timestamp(mo + "-01") <= kday]
        if folds:
            Wt, _ = select_all(Tt)
            for f in folds:
                assert Wt["picks"][f] == W["picks"][f], (kd, MONTHS[f])
                assert np.array_equal(Wt["scores"][:, f], W["scores"][:, f]), (kd, MONTHS[f])


def _perturb(m15, dc, tv, start, seed=5):
    rng = np.random.default_rng(seed)
    m = m15.copy()
    sel = (m.ts >= start).to_numpy()
    f = np.exp(rng.normal(0, 0.03, int(sel.sum())))
    for col in ("open", "high", "low", "close"):
        m.loc[sel, col] = m.loc[sel, col].to_numpy() * f
    m.loc[sel, "volume"] = m.loc[sel, "volume"].to_numpy() * np.exp(rng.normal(0, 0.5, int(sel.sum())))
    rowsel = dc.index >= start
    dc2, tv2 = dc.copy(), tv.copy()
    dc2.loc[rowsel] = dc2.loc[rowsel].to_numpy() * np.exp(rng.normal(0, 0.03, dc2.loc[rowsel].shape))
    tv2.loc[rowsel] = tv2.loc[rowsel].to_numpy() * np.exp(rng.normal(0, 0.8, tv2.loc[rowsel].shape))
    return m, dc2, tv2


def test_selection_ignores_test_month(data, full):
    """폴드 2026-10: 10월 1일부터 분봉·일봉·거래대금을 교란해도 그 폴드의 288 학습 점수·선택이 같다. 다음 폴드(11월)는 달라져야 한다(음성 대조)."""
    m15, dc, tv, codes = data
    Ss, W, _ = full
    f = MONTHS.index("2026-10")
    m2, dc2, tv2 = _perturb(m15, dc, tv, "2026-10-01")
    Wp, _ = select_all(build(m2, dc2, tv2, codes))
    assert np.array_equal(Wp["scores"][:, f], W["scores"][:, f])
    assert np.array_equal(Wp["scores"][:, :f], W["scores"][:, :f])
    assert Wp["picks"][: f + 1] == W["picks"][: f + 1]
    assert not np.array_equal(Wp["scores"][:, f + 1], W["scores"][:, f + 1])      # 교란이 실제로 닿는다


def _paths_with(S, DS, OC, var, st):
    ev = S["ev"]
    o = OC[(var, st)]
    idx = np.flatnonzero(o["ok"])
    name = {v[0]: (v[1], v[2]) for v in T.exit_variants()}[var]
    S2 = {**S, "DS": DS}
    return idx, T.run_paths(S2, ev["ci"][idx], ev["t"][idx] + 1, o["stop"][idx], *name)


def test_daily_exit_uses_that_day_close_only(data, full):
    """일봉 청산(X2·X3·X4): 날 d 까지의 일봉만 남겨도(그 뒤 행 삭제) 청산 체결일 ≤ d+1 인 일봉 청산은 같은 봉·같은 사유.
    그날 행(same)은 d 당일 종가를 쓴다(전날 행이 아님). 음성 대조: 다음 날 행을 당겨 쓰면 청산이 달라진다."""
    m15, dc, tv, codes = data
    Ss, W, OCs = full
    S, OC = Ss[2], OCs[2]                                    # 60분봉
    ud = S["udates"]
    for k in (60, 70, 80, 90, len(ud) - 3):
        d = pd.Timestamp(ud[k])
        assert np.allclose(np.nan_to_num(S["DS"]["close_same"][:, k]), np.nan_to_num(S["DS"]["dc"].ffill().loc[d].to_numpy()))
        mclose = pd.DataFrame(S["C"][:, S["last_bar"]].T, index=pd.DatetimeIndex(ud), columns=codes)
        DSt = T.daily_state(dc[dc.index <= d], codes, ud[: k + 1], mclose.iloc[: k + 1])
        for key in ("close_same", "ma10_same", "ma20_same"):
            assert np.array_equal(np.nan_to_num(DSt[key]), np.nan_to_num(S["DS"][key][:, : k + 1]), equal_nan=True), key
        assert np.array_equal(DSt["up_same"], S["DS"]["up_same"][:, :, : k + 1])
        # 일봉 청산 경로: 잘린 일봉(뒤쪽 날은 마지막 행 반복 → 판정 불가 칸)으로 돌려도 체결일 ≤ d+1 이면 같다
        pad = {kk: (np.concatenate([DSt[kk], np.repeat(DSt[kk][..., -1:], len(ud) - k - 1, axis=-1)], axis=-1)
                    if kk in ("close_same", "ma10_same", "ma20_same", "up_same") else S["DS"][kk]) for kk in S["DS"]}
        n_cmp = 0
        for var in ("X2", "X3", "X4U1"):
            idx, rf = _paths_with(S, S["DS"], OC, var, "S1")
            _, rt = _paths_with(S, pad, OC, var, "S1")
            fd = np.where(rf["xb"] >= 0, S["dix"][np.maximum(rf["xb"], 0)], 10 ** 9)
            m = rf["realized"] & (rf["why"] >= 3) & (fd <= k + 1)
            n_cmp += int(m.sum())
            assert np.array_equal(rf["xb"][m], rt["xb"][m]) and np.array_equal(rf["why"][m], rt["why"][m]), (var, k)
        assert n_cmp > 0
    # 음성 대조: 같은 날 행 대신 다음 날 행(미래)을 쓰면 일봉 청산이 바뀐다
    leak = dict(S["DS"])
    for kk in ("close_same", "ma10_same", "ma20_same"):
        a = S["DS"][kk].copy()
        a[:, :-1] = a[:, 1:]
        leak[kk] = a
    idx, rf = _paths_with(S, S["DS"], OC, "X2", "S1")
    _, rl = _paths_with(S, leak, OC, "X2", "S1")
    assert not np.array_equal(rf["xb"], rl["xb"])


def test_daily_exit_fills_next_day_open(full):
    """X2 사유 청산은 판정일 다음 거래일의 봉 시가에 체결되고, 판정일(체결 전 마지막 거래일) 종가 < 그날 MA20 이다."""
    Ss, W, OCs = full
    S, OC = Ss[0], OCs[0]
    o = OC[("X2", "S1")]
    m = o["realized"] & (o["why"] == 3)
    assert m.sum() > 0
    for k in np.flatnonzero(m)[:50]:
        b = int(o["xb"][k])
        d_fill = S["dix"][b]
        assert np.isclose(o["xpx"][k], S["O"][S["ev"]["ci"][k], b])
        i = S["ev"]["ci"][k]
        assert b == S["first_bar"][d_fill]                   # 합성 자료엔 빈 봉·하한가가 없어 미룸이 없다
        assert S["DS"]["close_same"][i, d_fill - 1] < S["DS"]["ma20_same"][i, d_fill - 1]


def test_menu_and_neighbors():
    M = T.menu()
    assert len(M) == 288 and M[0] == {"tf": "15m", "up": "U1", "spread": False, "overnight": True, "exit": "X1", "stop": "S1"}
    assert all(len(T.neighbors(i)) == 11 for i in range(288))
    s = np.zeros(288)
    s[5] = 10.0
    nt = np.arange(288)
    assert T.select(s, nt) == int(np.argmax(nt))            # 이웃 중앙값이 전부 0 → 동률 → 거래 수 많은 쪽


# ============================================================================== 재등록 MT-2 (PREREG 27334ef)

COST = 0.00415


def synth_kosdaq(seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2025-01-02", "2026-11-13")
    return pd.Series(800 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, len(d)))), index=d)


def setup_v2(Ss, q):
    RG = T.regime_on(q, Ss[0]["udates"], T.u1_share(Ss[0]))
    costs = np.full((len(Ss[0]["udates"]), 2), COST)
    mkt = np.zeros(len(Ss[0]["codes"]), np.int64)
    for S in Ss:
        S["regime"] = RG["on"]
        T.add_floor_stop(S, costs, mkt)
    return RG, costs, mkt


def select_v2(Ss, q):
    RG, costs, mkt = setup_v2(Ss, q)
    OCs, dayclose = T.prepare(Ss, stops=("SF",))
    C = T.all_candidates(Ss, OCs, costs, mkt, 2.0, axes=T.AXES_V2)
    return T.wf_select(C, dayclose, MONTHS, key="Racct", drop_top=3, axes=T.AXES_V2), C, RG


@pytest.fixture(scope="module")
def full_v2(data):
    m15, dc, tv, codes = data
    Ss = build(m15, dc, tv, codes)
    q = synth_kosdaq()
    W, C, RG = select_v2(Ss, q)
    return Ss, W, C, RG, q


def test_menu_v2():
    M = T.menu(T.AXES_V2)
    assert len(M) == 288 and {m["stop"] for m in M} == {"SF"} and {m["regime"] for m in M} == {"ignore", "on"}
    assert all(len(T.neighbors(i, T.AXES_V2)) == 11 for i in range(288))
    assert list(M[0]) == ["tf", "up", "spread", "overnight", "exit", "stop", "regime"]


def test_r_acct_formula():
    """R_acct = ret × min(W_CAP/RISK, 1/거리) = 비중 × ret ÷ RISK(비중 = min(10%, 0.5%/거리))."""
    ret = np.array([0.10, 0.10, -0.02, 0.03])
    dist = np.array([0.01, 0.20, 0.05, 0.025])
    assert np.allclose(T.r_acct(ret, dist), [2.0, 0.5, -0.4, 0.6])
    w = np.minimum(T.W_CAP, T.RISK / dist)
    assert np.allclose(T.r_acct(ret, dist), w * ret / T.RISK)
    # 1차 R(손절 거리로 나눔)과는 거리 < 5% 에서 다르다 — 좁은 손절 거래의 R 이 20 배 상한으로 묶인다
    assert not np.allclose(T.r_acct(ret, dist), ret / dist)


def test_floor_dist_and_stop(full_v2):
    """손절 거리 = max(구조 거리, 왕복 비용 × 2 × 3, 2%), 손절가 = 진입가 × (1 − 거리). 구조가 진입 전에 깨졌으면 거래 없음."""
    assert np.allclose(T.floor_dist([0.005, 0.03, -0.01, 0.021], COST), [0.0249, 0.03, 0.0249, 0.0249])
    assert np.allclose(T.floor_dist([0.005], 0.0025), [0.02])                 # 비용 × 6 < 2% 면 2%
    Ss, W, C, RG, q = full_v2
    floor = max(COST * 6, 0.02)
    n_bind = n_floor = 0
    for S in Ss:
        ev = S["ev"]
        ent, s1, sf = ev["entry_px"], ev["s1"], ev["sf"]
        ok = np.isfinite(sf)
        with np.errstate(invalid="ignore"):
            assert not (ok & ~(s1 < ent)).any()                               # 구조 손절 ≥ 진입가 → NaN 유지
            assert (np.isfinite(s1) & np.isfinite(ent) & (s1 < ent) == ok).all()
        d = (ent[ok] - sf[ok]) / ent[ok]
        sd = (ent[ok] - s1[ok]) / ent[ok]
        assert (d >= floor - 1e-12).all()
        assert np.allclose(d, np.maximum(sd, floor))
        assert (sf[ok] <= s1[ok] + 1e-9).all()                                # 손절가는 넓히기만 한다
        n_bind += int((sd > floor).sum())
        n_floor += int((sd < floor).sum())
    assert n_bind > 0 and n_floor > 0                                         # 두 경우가 모두 합성 자료에 있다
    for c in C:
        if len(c["ci"]):
            assert (c["dist"] >= floor - 1e-12).all()
            assert np.allclose(c["stop"], c["entry_px"] * (1 - c["dist"]))
            assert np.allclose(c["Racct"], T.r_acct(c["ret"], c["dist"]))


def test_train_score_drops_top3():
    assert T.robust_sum([5, 3, 1, -2, -4], 3) == -6
    assert T.robust_sum([4, -1], 3) == -1                                    # 양수만 뺀다
    assert T.robust_sum([-1, -2], 3) == -3                                   # 손실을 빼서 점수가 오르지 않는다
    assert T.robust_sum([5, 3, 1], 0) == 9
    acc = {"realized": np.array([True, True, True, True, True, False]), "exit_ts": np.array([0, 1, 2, 3, 10 ** 12, 0]),
           "Racct": np.array([9.0, 1.0, -1.0, 2.0, 50.0, 100.0]), "R": np.zeros(6)}
    sc, nt = T.train_scores(acc, ["2026-09"], key="Racct", drop_top=3)
    assert sc[0] == -1.0 and nt[0] == 4                                       # 9 + 1 − 1 + 2 − (9 + 2 + 1); 미실현·시험 뒤 청산 제외


def test_regime_uses_prior_day_only(full_v2):
    """국면(d) = d 전 마지막 지수 행 종가 > MA50 ∧ d 유니버스 U1 비율(전날 일봉) ≥ 25%. d 와 그 뒤 지수를 지우거나 교란해도 d 까지 같다.
    음성 대조: 그날 지수를 전날 날짜로 당겨(미래 정보) 쓰면 국면이 바뀐다."""
    Ss, W, C, RG, q = full_v2
    ud = pd.DatetimeIndex(Ss[0]["udates"])
    sh = T.u1_share(Ss[0])
    assert np.isfinite(sh).all() and 0 < RG["on"].sum() < len(ud) and 0 < RG["idx_on"].sum() < len(ud)
    rng = np.random.default_rng(9)
    for k in (5, 20, 40, 60, len(ud) - 2):
        d = ud[k]
        qt = q[q.index < d]
        rt = T.regime_on(qt, ud[: k + 1], sh[: k + 1])
        assert np.array_equal(rt["on"], RG["on"][: k + 1]) and np.array_equal(rt["idx_on"], RG["idx_on"][: k + 1])
        qp = q.copy()
        qp[qp.index >= d] *= np.exp(rng.normal(0, 0.2, int((qp.index >= d).sum())))
        rp = T.regime_on(qp, ud, sh)
        assert np.array_equal(rp["on"][: k + 1], RG["on"][: k + 1])
        assert (rt["prev_date"] < ud[: k + 1].values).all()
    leak = T.regime_on(q.shift(-1).dropna(), ud, sh)                          # d−1 날짜에 d 종가 → 미래
    assert not np.array_equal(leak["idx_on"], RG["idx_on"])
    # U1 비율의 분자는 전날 행(up_prev) — 그날 행(up_same)을 쓰면 달라진다(음성 대조)
    S = Ss[0]
    same = (S["DS"]["up_same"][0] & S["MEMd"]).sum(0) / S["MEMd"].sum(0)
    assert not np.allclose(same, sh)


def test_regime_gene_blocks_off_day_entries(full_v2):
    Ss, W, C, RG, q = full_v2
    M = T.menu(T.AXES_V2)
    for j, c in enumerate(M):
        if c["regime"] != "on":
            continue
        k = M.index({**c, "regime": "ignore"})
        a, b = C[j], C[k]
        assert a["reg_on"].all()
        S = Ss[T.AXES_V2["tf"].index(c["tf"])]
        assert RG["on"][S["dix"][a["eb"]]].all()                             # 진입 체결일 국면 ON
        assert set(zip(a["ci"], a["t"])) == set(zip(b["ci"][b["reg_on"]], b["t"][b["reg_on"]]))
    assert any(len(C[k]["ci"]) > len(C[j]["ci"]) for j, k in
               ((j, M.index({**c, "regime": "ignore"})) for j, c in enumerate(M) if c["regime"] == "on"))


def test_v2_selection_ignores_test_month(data, full_v2):
    """MT-2 선택(R_acct − 상위 3, 국면, 하한): 10월 1일부터 분봉·일봉·거래대금·지수를 교란해도 폴드 ≤ 10월 점수·선택이 같다. 11월은 바뀐다(음성 대조)."""
    m15, dc, tv, codes = data
    Ss, W, C, RG, q = full_v2
    f = MONTHS.index("2026-10")
    m2, dc2, tv2 = _perturb(m15, dc, tv, "2026-10-01")
    q2 = q.copy()
    s = q2.index >= "2026-10-01"
    q2[s] *= np.exp(np.random.default_rng(4).normal(0, 0.05, int(s.sum())))
    Wp, _, _ = select_v2(build(m2, dc2, tv2, codes), q2)
    assert np.array_equal(Wp["scores"][:, : f + 1], W["scores"][:, : f + 1])
    assert Wp["picks"][: f + 1] == W["picks"][: f + 1]
    assert not np.array_equal(Wp["scores"][:, f + 1], W["scores"][:, f + 1])


def test_v2_truncation_invariance(data, full_v2):
    """날짜 2개 절단(분봉 그 시각 전, 일봉·지수 그날 전): 국면·U1 비율·하한 손절·MT-2 선택이 같다."""
    m15, dc, tv, codes = data
    Ss, W, C, RG, q = full_v2
    for kd, hh in (("2026-09-15", 13), ("2026-11-04", 11)):
        kday = pd.Timestamp(kd)
        hi = kday + pd.Timedelta(hours=hh)
        Tt = build(m15[m15.ts < hi], dc[dc.index < kday], tv[tv.index < kday], codes)
        Wt, _, RGt = select_v2(Tt, q[q.index < kday])
        nd = len(Tt[0]["udates"])
        assert np.array_equal(RGt["on"], RG["on"][:nd]) and np.allclose(RGt["share"], RG["share"][:nd])
        for F, X in zip(Ss, Tt):
            fk = {(F["codes"][i], F["days"][t]): v for i, t, v in zip(F["ev"]["ci"], F["ev"]["t"], F["ev"]["sf"])}
            for i, t, v in zip(X["ev"]["ci"], X["ev"]["t"], X["ev"]["sf"]):
                if X["days"][t] < hi - pd.Timedelta(minutes=X["tf_min"]):
                    a = fk[(X["codes"][i], X["days"][t])]
                    assert (np.isnan(a) and np.isnan(v)) or np.isclose(a, v), (X["tf"], kd)
        folds = [f for f, mo in enumerate(MONTHS) if pd.Timestamp(mo + "-01") <= kday]
        for f in folds:
            assert Wt["picks"][f] == W["picks"][f]
            assert np.array_equal(Wt["scores"][:, f], W["scores"][:, f])
