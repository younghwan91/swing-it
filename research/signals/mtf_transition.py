"""일봉 상승 중 아래 봉(15·30·60분) 역배열→정배열 전환 — 패널·사건·거래 경로·단일 계좌·워크포워드 선택. 순수 함수(DB 없음).

정본: ``research/logs/mtf_transition/PREREG.md`` (4fb9525; 재등록 MT-2 = 27334ef). 해석은 실행기 ``research/experiments/mtf_transition_wf.py`` 의
RULINGS(1차)·RULINGS_V2(MT-2). MT-2 부품(``AXES_V2``·``add_floor_stop``·``r_acct``·``u1_share``·``regime_on``, ``drop_top``/``key``/``axes`` 인자)은
기본값이 1차 동작 그대로라 1차 실행을 바꾸지 않는다.
탐색 구현(``~/.cache/cc-swing/scratch/golden_ev/{transition,topdown_intraday_pit}``)의 시점 유니버스·분할 보정·전환 사건·구조 손절을
레포로 옮긴 것이다(규약 그대로, 문체만 정리).

시간 규약: 봉 t 의 상태는 그 봉 종가까지, 일봉 상승 상태는 **전날까지**의 일봉 행, 진입 = 다음 봉 시가.
일봉 청산 판정은 그날 행(그날 정규장 종가까지) → 다음 날 첫 봉 시가.
"""
from __future__ import annotations

import itertools

try:  # numba 는 선택(:mod:`research.signals.vcp` 와 같은 규약) — 없으면 같은 소스가 파이썬으로 돈다.
    import numba

    def _njit(fn):
        return numba.njit(cache=False)(fn)

except ImportError:  # pragma: no cover - CI 경로

    def _njit(fn):
        return fn
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------- 상수

TOPN = 300                      # 시점 유니버스: d−1 까지 20거래일 평균 거래대금 상위
ADV_WINDOW, ADV_MIN_PERIODS = 20, 15        # 탐색 load.py/pit.py 규약(rolling 20, min_periods 15)
M_REV = 20                      # 직전 20봉 안 완전 역배열
GAP = 20                        # 같은 종목 20봉 안 재발은 첫 봉만(탐색 pit.dedupe — 사슬 규약)
WIN, WIN_MIN = 80, 72           # 탐색 trans.transitions 의 자료 유효성: 최근 80봉 중 72봉 이상 유효
SIG_W = 60                      # σ60 = 60봉 로그수익 표준편차
SPREAD_K = 0.5                  # MA 간격 거르개: (MA5 − MA60)/MA60 ≥ σ60 × 0.5
S2_K = 2.0                      # S2 손절 거리 하한 = σ60 × 2
LIMIT_LO, LIMIT_HI = np.log(0.70) - 0.005, np.log(1.30) + 0.005   # 봉 사이 가격제한폭 밖 = 기업행위 의심(maquad_tf_diag 규약)
LIM_UP, LIM_DN = 1.295, 0.705   # 시가가 전날 종가 대비 이 밖이면 상·하한가 시가로 본다(호가 반올림 여유)
FILL_FROM = "2026-09-14"        # 일봉 정문 unpatched=NaN 칸을 그날 분봉 마지막 종가로 채우는 구간(탐색 pit.signals 규약)
SLOTS, W_CAP, RISK, CASH0 = 10, 0.10, 0.005, 1e8
MIN_SIZE = 0.001                # 남은 현금이 자산의 0.1% 미만이면 매수 불가(RULINGS["cash"])
TF_MIN = {"15m": 15, "30m": 30, "60m": 60}
INF_TS = np.iinfo(np.int64).max // 4

AXES = {
    "tf": ("15m", "30m", "60m"),
    "up": ("U1", "U2", "U3"),
    "spread": (False, True),
    "overnight": (True, False),           # True = 하루 마지막 봉 사건 허용
    "exit": ("X1", "X2", "X3", "X4"),
    "stop": ("S1", "S2"),
}
# 재등록 MT-2(27334ef): stop 유전자 1개(SF = 구조 손절 + 거리 하한), regime 유전자(무시 / 국면 OFF 날 신규 진입 금지). 곱 288.
AXES_V2 = {
    "tf": ("15m", "30m", "60m"),
    "up": ("U1", "U2", "U3"),
    "spread": (False, True),
    "overnight": (True, False),
    "exit": ("X1", "X2", "X3", "X4"),
    "stop": ("SF",),
    "regime": ("ignore", "on"),
}
FLOOR_COST_MULT, FLOOR_COST_K, FLOOR_ABS = 2.0, 3.0, 0.02   # 거리 하한 = max(왕복 비용 × 2 × 3, 2%)
REGIME_MA, REGIME_U1_MIN = 50, 0.25                         # 국면 ON = 전날 KOSDAQ > MA50 ∧ 그날 유니버스 U1 비율 ≥ 25%
WHY = {0: "미청산", 1: "손절", 2: "X1 아래 봉 MA60 이탈", 3: "X2 일봉 MA20 이탈", 4: "X3 일봉 MA10 이탈", 5: "X4 일봉 상승 꺼짐"}


def menu(axes: dict = AXES) -> list[dict]:
    """288 설정, 순서 고정(PREREG 표기 순: tf × up × spread × overnight × exit × stop [× regime])."""
    return [dict(zip(axes, v)) for v in itertools.product(*axes.values())]


def _coords(axes: dict = AXES) -> list[tuple]:
    return list(itertools.product(*(range(len(v)) for v in axes.values())))


def neighbors(i: int, axes: dict = AXES) -> list[int]:
    """자기 + 유전자 하나만 다른 설정(모든 축을 범주로 — 다른 값 전부). 288 칸에서 자기 포함 11 개(1차·MT-2 모두)."""
    co = _coords(axes)
    me = co[i]
    out = [i]
    for j, c in enumerate(co):
        if sum(a != b for a, b in zip(c, me)) == 1:
            out.append(j)
    return out


def neighbor_medians(scores, axes: dict = AXES) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if len(s) != len(_coords(axes)):
        raise ValueError("점수 수가 메뉴와 다르다")
    s = np.where(np.isfinite(s), s, -np.inf)
    return np.array([np.median(s[neighbors(i, axes)]) for i in range(len(s))])


def select(scores, n_trades, axes: dict = AXES) -> int:
    """이웃 중앙값 최대. 동률이면 (그 설정 자신의) 학습 실현 거래 수 많은 쪽, 그래도 같으면 메뉴 순서 앞."""
    nm = neighbor_medians(scores, axes)
    best = nm.max()
    cand = np.flatnonzero(nm == best)
    nt = np.asarray(n_trades)[cand]
    return int(cand[np.flatnonzero(nt == nt.max())[0]])


# --------------------------------------------------------------------------------------------- 패널

def aggregate(m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """15분봉 → 30·60분봉(하루 안, 정시·반시 시작). 15:15 봉(종가 단일가 포함)은 30분이면 15:00, 60분이면 15:00 봉으로 들어간다."""
    x = m.copy()
    x["ts"] = x["ts"].dt.floor(f"{minutes}min")
    g = x.sort_values(["code", "ts"]).groupby(["code", "ts"], sort=True)
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                 volume=("volume", "sum")).reset_index()


def _panel(m: pd.DataFrame, codes: list) -> tuple:
    grid = pd.DatetimeIndex(np.sort(m["ts"].unique()))
    ci = pd.Index(codes).get_indexer(m["code"])
    ti = grid.get_indexer(m["ts"])
    if (ci < 0).any():
        raise ValueError("codes 밖 종목")
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        a = np.full((len(codes), len(grid)), np.nan)
        a[ci, ti] = m[col].to_numpy(dtype=float)
        out[col] = a
    return grid, out


def roll(a: np.ndarray, w: int, fn: str = "mean") -> np.ndarray:
    r = pd.DataFrame(a.T).rolling(w)
    return getattr(r, fn)().to_numpy().T


def rsum(a: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.nancumsum(a, axis=1)], axis=1)
    out = np.full(a.shape, np.nan)
    out[:, w - 1:] = cs[:, w:] - cs[:, :-w]
    return out


def ever_in(cond: np.ndarray, w: int) -> np.ndarray:
    """칸 t 에서 [t−w, t−1] 안에 cond 가 한 번이라도 참(과거만)."""
    cs = np.concatenate([np.zeros((cond.shape[0], 1)), np.cumsum(cond, axis=1)], axis=1)
    t = np.arange(cond.shape[1])
    lo = np.maximum(t - w, 0)
    return (cs[:, t] - cs[:, lo]) > 0


def dedupe(cond: np.ndarray, gap: int = GAP) -> tuple:
    ec, et = [], []
    for i in range(cond.shape[0]):
        ts = np.flatnonzero(cond[i])
        if len(ts) == 0:
            continue
        first = np.r_[True, np.diff(ts) > gap]
        ec += [i] * int(first.sum())
        et += list(ts[first])
    return np.asarray(ec, np.int64), np.asarray(et, np.int64)


def adjust_splits(Op, H, Lo, C, dix, last_bar, udates, codes, dc) -> dict:
    """분봉은 원 가격. 봉 사이 로그 변화가 가격제한폭 밖인 종목은 그 지점으로 구간을 나누고 구간마다 (일봉 정문 수정종가 / 분봉 일 종가)
    중앙값을 곱한다(제자리 수정) — 구간 안 상수배라 이평선 순서·수익률 불변(탐색 pit.signals adjust=True 그대로)."""
    n_c, n = C.shape
    mcl0 = pd.DataFrame(C[:, last_bar].T, index=pd.DatetimeIndex(udates), columns=codes)
    ratio = (dc.reindex(mcl0.index) / mcl0).to_numpy().T
    adj, dropped = [], []
    for i in range(n_c):
        f = np.flatnonzero(np.isfinite(C[i]))
        if len(f) < 2:
            continue
        d = np.diff(np.log(C[i, f]))
        j = np.flatnonzero((d < LIMIT_LO) | (d > LIMIT_HI))
        if len(j) == 0:
            continue
        cuts = np.r_[0, f[j + 1], n]
        for a, b in zip(cuts[:-1], cuts[1:]):
            dd = np.unique(dix[a:b])
            r = ratio[i, dd]
            r = r[np.isfinite(r)]
            fc = np.median(r) if len(r) else np.nan
            for A in (Op, H, Lo, C):
                A[i, a:b] = A[i, a:b] * fc if np.isfinite(fc) else np.nan
            if not np.isfinite(fc):
                dropped.append(codes[i])
        adj.append(codes[i])
    return {"adj_codes": adj, "dropped_seg": sorted(set(dropped))}


def daily_state(dc_all: pd.DataFrame, codes: list, udates, mclose: pd.DataFrame) -> dict:
    """일봉 상태. ``prev`` = 거래일 d 보다 엄격히 앞선 마지막 행(진입 거르개), ``same`` = d 이하 마지막 행(일봉 청산 판정 — 그날 종가까지).
    U1 = 종가 > MA20 > MA60 > MA120, U2 = U1 ∧ MA20 > 5일 전 MA20, U3 = U2 ∧ MA60 > 20일 전 MA60. DEF = 세 정의에 필요한 값이 모두 있음."""
    dc = dc_all.reindex(columns=codes)
    fill = dc.isna() & (dc.index >= FILL_FROM)[:, None]
    dc = dc.where(~fill, mclose.reindex(dc.index))
    x = dc.ffill()
    m10, m20, m60, m120 = (x.rolling(w).mean() for w in (10, 20, 60, 120))
    u1 = (x > m20) & (m20 > m60) & (m60 > m120)
    u2 = u1 & (m20 > m20.shift(5))
    u3 = u2 & (m60 > m60.shift(20))
    dfn = (m120.notna() & m20.shift(5).notna() & m60.shift(20).notna()).to_numpy()
    ddays = dc.index.values
    ud = np.asarray(udates, dtype="datetime64[ns]")
    rp = np.searchsorted(ddays, ud, side="left") - 1
    rs = np.searchsorted(ddays, ud, side="right") - 1
    if not ((rp >= 0).all() and (ddays[rp] < ud).all()):
        raise ValueError("일봉이 분봉 첫날 전부터 있어야 한다")
    U = np.stack([u.to_numpy() for u in (u1, u2, u3)])            # 3 × 날 × 종목
    up_prev = np.ascontiguousarray((U[:, rp] & dfn[rp][None]).transpose(0, 2, 1))   # 3 × 종목 × 거래일
    up_same = np.ascontiguousarray((U[:, rs] & dfn[rs][None]).transpose(0, 2, 1))
    return {"dc": dc, "up_prev": up_prev, "def_prev": np.ascontiguousarray(dfn[rp].T), "up_same": up_same,
            "close_same": np.ascontiguousarray(x.to_numpy()[rs].T), "ma10_same": np.ascontiguousarray(m10.to_numpy()[rs].T),
            "ma20_same": np.ascontiguousarray(m20.to_numpy()[rs].T), "n_fill": int((fill & mclose.reindex(dc.index).notna()).to_numpy().sum())}


def universe(tv_all: pd.DataFrame, elig: list, codes: list, udates, topn: int = TOPN) -> tuple:
    """거래일 d 의 대상 = d 보다 앞선 마지막 일봉 행까지의 20거래일 평균 거래대금 상위 ``topn``(보통주·KOSPI/KOSDAQ = elig)."""
    tv = tv_all[[c for c in elig if c in tv_all.columns]]
    adv = tv.rolling(ADV_WINDOW, min_periods=ADV_MIN_PERIODS).mean()
    rk = adv.rank(axis=1, ascending=False, method="first").to_numpy()
    ud = np.asarray(udates, dtype="datetime64[ns]")
    r = np.searchsorted(tv.index.values, ud, side="left") - 1
    if not (r >= 0).all():
        raise ValueError("거래대금이 분봉 첫날 전부터 있어야 한다")
    cols = np.asarray(tv.columns)
    memday = {pd.Timestamp(d): set(cols[rk[k] <= topn]) for d, k in zip(udates, r)}
    cidx = {c: i for i, c in enumerate(codes)}
    Md = np.zeros((len(codes), len(udates)), bool)
    for k, d in enumerate(udates):
        ii = [cidx[c] for c in memday[pd.Timestamp(d)] if c in cidx]
        Md[ii, k] = True
    return Md, memday


@_njit
def struct_stop(rev, Lf, ci, t, m):
    """전환 전 역배열 구간 최저가: t−1 부터 거슬러 가장 가까운 역배열 봉(m 봉 안) → 그 연속 역배열 구간 시작부터 t 까지 저가 최저(탐색 trans 그대로)."""
    out = np.full(len(t), np.nan)
    for e in range(len(t)):
        i, tt = ci[e], t[e]
        j = -1
        for k in range(tt - 1, max(tt - m - 1, -1), -1):
            if rev[i, k]:
                j = k
                break
        if j < 0:
            continue
        s = j
        while s - 1 >= 0 and rev[i, s - 1]:
            s -= 1
        lo = np.inf
        for k in range(s, tt + 1):
            if Lf[i, k] < lo:
                lo = Lf[i, k]
        out[e] = lo
    return out


def relvol(V: np.ndarray, days: pd.DatetimeIndex) -> np.ndarray:
    """그 봉 거래량 ÷ 같은 시각대 직전 20거래일 평균(결측 봉 = 0 거래량, 이력 5일 미만 NaN)."""
    out = np.full(V.shape, np.nan)
    tod = np.asarray(days.hour * 100 + days.minute)
    Vz = np.nan_to_num(V, nan=0.0)
    for s in np.unique(tod):
        idx = np.flatnonzero(tod == s)
        mu = pd.DataFrame(Vz[:, idx].T).rolling(20, min_periods=5).mean().shift(1).to_numpy().T
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, idx] = np.where(mu > 0, Vz[:, idx] / mu, np.nan)
    return out


def build_tf(m: pd.DataFrame, dc_all: pd.DataFrame, tv_all: pd.DataFrame, elig: list, tf: str, codes: list | None = None,
             topn: int = TOPN) -> dict:
    """한 봉 단위 패널 + 일봉 상태 + 유니버스 + 전환 사건(일봉 U1 ON·DEF 인 것만 — U2·U3 ⊂ U1)과 사건 특징."""
    codes = sorted(m["code"].unique()) if codes is None else list(codes)
    days, P = _panel(m, codes)
    Op, H, Lo, C, V = P["open"], P["high"], P["low"], P["close"], P["volume"]
    dead = ~(np.nan_to_num(V, nan=0.0) > 0)
    for A in (Op, H, Lo, C):
        A[dead] = np.nan
    date = np.asarray(days.normalize())
    udates = np.unique(date)
    dix = np.searchsorted(udates, date).astype(np.int64)
    last_bar = np.flatnonzero(np.r_[date[1:] != date[:-1], True])
    first_bar = np.r_[0, last_bar[:-1] + 1]
    dc0 = dc_all.reindex(columns=codes)
    info = adjust_splits(Op, H, Lo, C, dix, last_bar, udates, codes, dc0)
    fin = np.isfinite(C)
    Cf = pd.DataFrame(C.T).ffill().to_numpy().T
    mclose = pd.DataFrame(C[:, last_bar].T, index=pd.DatetimeIndex(udates), columns=codes)
    DS = daily_state(dc_all, codes, udates, mclose)
    MEMd, memday = universe(tv_all, elig, codes, udates, topn)
    MEM = MEMd[:, dix]
    ma = {w: roll(Cf, w) for w in (5, 20, 60)}
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.full(Cf.shape, np.nan)
        lr[:, 1:] = np.log(Cf[:, 1:] / Cf[:, :-1])
    sig = roll(lr, SIG_W, "std")
    win = rsum(fin.astype(float), WIN) >= WIN_MIN
    al = (ma[5] > ma[20]) & (ma[20] > ma[60])
    rev = (ma[5] < ma[20]) & (ma[20] < ma[60])
    prev_al = np.zeros_like(al)
    prev_al[:, 1:] = al[:, :-1]
    c0 = al & ~prev_al & ever_in(rev, M_REV) & fin & win
    ci, t = dedupe(c0)
    inm = MEM[ci, t]
    n_first = len(t)
    ci, t = ci[inm], t[inm]
    u_ok = DS["up_prev"][0][ci, dix[t]] & DS["def_prev"][ci, dix[t]]
    n_univ = len(t)
    ci, t = ci[u_ok], t[u_ok]
    Lf = np.where(np.isfinite(Lo), Lo, Cf)
    s1 = struct_stop(rev, Lf, ci, t, M_REV)
    # 전날 종가(상·하한가 시가 판정 기준) — 봉 b 의 날 직전 거래일 마지막 봉의 ffill 종가
    dcl = Cf[:, last_bar]                                                    # 종목 × 거래일(그날 마지막 봉 ffill 종가)
    prevc = np.full(C.shape, np.nan)
    ok = dix >= 1
    prevc[:, ok] = dcl[:, dix[ok] - 1]
    n = C.shape[1]
    nxt = np.minimum(t + 1, n - 1)
    epx = np.where(t + 1 < n, Op[ci, nxt], np.nan)
    with np.errstate(invalid="ignore"):
        lim_up_open = epx >= prevc[ci, nxt] * LIM_UP
        spread = (ma[5][ci, t] - ma[60][ci, t]) / ma[60][ci, t] >= SPREAD_K * sig[ci, t]
    is_last = np.zeros(n, bool)
    is_last[last_bar] = True
    rv = relvol(V, days)
    sg = sig[ci, t]
    with np.errstate(invalid="ignore"):
        s2 = np.minimum(s1, epx * (1.0 - S2_K * sg))
    ev = {"ci": ci, "t": t, "u1": DS["up_prev"][0][ci, dix[t]], "u2": DS["up_prev"][1][ci, dix[t]], "u3": DS["up_prev"][2][ci, dix[t]],
          "spread": np.asarray(spread, bool), "last": is_last[t], "relvol": rv[ci, t], "sigma": sg, "entry_px": epx,
          "s1": s1, "s2": s2,
          "entry_ok": np.isfinite(epx) & (epx > 0) & ~np.asarray(lim_up_open, bool)}
    ts_min = (days.values.astype("datetime64[m]").astype(np.int64)).astype(np.int64)
    return {"tf": tf, "tf_min": TF_MIN[tf], "days": days, "codes": codes, "O": Op, "H": H, "Lo": Lo, "C": C, "V": V, "Cf": Cf, "ma": ma,
            "sig": sig, "fin": fin, "win": win, "date": date, "udates": udates, "dix": dix, "last_bar": last_bar, "first_bar": first_bar,
            "is_last": is_last, "prevc": prevc, "dcl": dcl, "MEM": MEM, "MEMd": MEMd, "memday": memday, "ts_min": ts_min, "ev": ev, "DS": DS,
            "info": {**info, "n_cond": int(c0.sum()), "n_first": n_first, "n_univ": n_univ, "n_u1": int(len(t)), "n_fill": DS["n_fill"]}}


# --------------------------------------------------------------------------------------------- 거래 경로

@_njit
def paths(Op, H, L, C, MA60, dix, is_last, prevc, dclose, dma10, dma20, dup, ci, eb, stop, mode, upk, lim_dn):
    """거래 경로(계좌 무관). eb = 진입 봉(시가 체결), stop = 초기 손절가. mode 1 = X1(아래 봉 MA60 종가 이탈 → 다음 봉 시가),
    2 = X2(일봉 종가 < 일봉 MA20 → 다음 날 첫 봉 시가), 3 = X3(MA10), 4 = X4(그날 일봉 상승 상태[upk] 꺼짐 → 다음 날 첫 봉 시가).
    봉마다 순서: ① 대기 청산(시가, 하한가 시가면 다음 봉) ② 손절(저가 ≤ 손절 → min(손절, 시가); 시가가 하한가면 다음 봉 시가로 대기)
    ③ 종가 조건(X1, 또는 그날 마지막 봉이면 일봉 조건 — 일봉 행은 그날 것). 끝까지 안 나가면 미청산(마지막 유한 종가 평가).
    반환: 청산 봉, 청산가, 사유(WHY), 시가 체결 여부, 실현 여부, 평가가."""
    N = len(ci)
    n = Op.shape[1]
    xb = np.full(N, -1, np.int64)
    xpx = np.full(N, np.nan)
    why = np.zeros(N, np.int64)
    at_open = np.zeros(N, np.bool_)
    realized = np.zeros(N, np.bool_)
    mtm = np.full(N, np.nan)
    for e in range(N):
        i = ci[e]
        pend = 0
        lastc = np.nan
        for b in range(eb[e], n):
            o = Op[i, b]
            lim = prevc[i, b] * lim_dn
            if pend > 0:
                if np.isfinite(o) and not (o <= lim):
                    xb[e] = b
                    xpx[e] = o
                    why[e] = pend
                    at_open[e] = True
                    realized[e] = True
                    break
            elif np.isfinite(L[i, b]) and L[i, b] <= stop[e]:
                if np.isfinite(o) and o <= stop[e]:
                    if o <= lim:
                        pend = 1
                    else:
                        xb[e] = b
                        xpx[e] = o
                        why[e] = 1
                        at_open[e] = True
                        realized[e] = True
                        break
                else:
                    xb[e] = b
                    xpx[e] = stop[e]
                    why[e] = 1
                    realized[e] = True
                    break
            if np.isfinite(C[i, b]):
                lastc = C[i, b]
            if pend == 0:
                if mode == 1 and np.isfinite(C[i, b]) and np.isfinite(MA60[i, b]) and C[i, b] < MA60[i, b]:
                    pend = 2
                elif mode >= 2 and is_last[b]:
                    d = dix[b]
                    if mode == 2 and dclose[i, d] < dma20[i, d]:
                        pend = 3
                    elif mode == 3 and dclose[i, d] < dma10[i, d]:
                        pend = 4
                    elif mode == 4 and not dup[upk, i, d]:
                        pend = 5
        if not realized[e]:
            mtm[e] = lastc
    return xb, xpx, why, at_open, realized, mtm


def exit_variants() -> list[tuple]:
    """(이름, mode, upk): X1, X2, X3, X4·U1, X4·U2, X4·U3."""
    return [("X1", 1, 0), ("X2", 2, 0), ("X3", 3, 0), ("X4U1", 4, 0), ("X4U2", 4, 1), ("X4U3", 4, 2)]


def variant_of(cfg: dict) -> str:
    return cfg["exit"] if cfg["exit"] != "X4" else "X4" + cfg["up"]


def run_paths(S: dict, ci, eb, stop, mode: int, upk: int) -> dict:
    DS = S["DS"]
    out = paths(S["O"], S["H"], S["Lo"], S["C"], S["ma"][60], S["dix"], S["is_last"], S["prevc"], DS["close_same"],
                DS["ma10_same"], DS["ma20_same"], DS["up_same"], np.asarray(ci, np.int64), np.asarray(eb, np.int64),
                np.asarray(stop, float), mode, upk, LIM_DN)
    return dict(zip(("xb", "xpx", "why", "at_open", "realized", "mtm"), out))


def outcomes(S: dict, stops: tuple = ("S1", "S2")) -> dict:
    """사건마다 (청산 변형 × 손절) 경로. 키 (변형 이름, 손절 이름). 손절가 = ``ev[이름.lower()]``(S1·S2, MT-2 는 SF — add_floor_stop)."""
    ev = S["ev"]
    eb = ev["t"] + 1
    res = {}
    for st in stops:
        stop = ev[st.lower()]
        ok = ev["entry_ok"] & np.isfinite(stop) & (stop < ev["entry_px"])
        idx = np.flatnonzero(ok)
        for name, mode, upk in exit_variants():
            r = run_paths(S, ev["ci"][idx], eb[idx], stop[idx], mode, upk)
            N = len(eb)
            full = {"xb": np.full(N, -1, np.int64), "xpx": np.full(N, np.nan), "why": np.zeros(N, np.int64),
                    "at_open": np.zeros(N, bool), "realized": np.zeros(N, bool), "mtm": np.full(N, np.nan)}
            for k in r:
                full[k][idx] = r[k]
            full["ok"] = ok
            full["stop"] = stop
            res[(name, st)] = full
    return res


# --------------------------------------------------------------------------------------------- MT-2: 손절 하한·계좌 단위 R·국면

def floor_dist(struct_dist, rt_cost, cost_mult: float = FLOOR_COST_MULT, k: float = FLOOR_COST_K, abs_min: float = FLOOR_ABS):
    """MT-2 손절 거리 = max(구조 거리, 왕복 비용 × 2 × 3, 2%)."""
    return np.maximum(np.maximum(np.asarray(struct_dist, float), np.asarray(rt_cost, float) * cost_mult * k), abs_min)


def add_floor_stop(S: dict, costs: np.ndarray, mkt: np.ndarray) -> None:
    """``ev["sf"]`` = 진입가 × (1 − 하한 거리). 구조 손절 S1 이 없거나 진입가 이상(구조가 진입 전에 이미 깨짐)이면 NaN(1차 '손절가 ≥ 진입가면
    거래하지 않는다' 그대로). 왕복 비용은 진입 체결일 세율(RULINGS_V2["floor_cost"])."""
    ev = S["ev"]
    n = len(S["days"])
    eb = np.minimum(ev["t"] + 1, n - 1)
    eday = S["dix"][eb]
    ent, s1 = ev["entry_px"], ev["s1"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(ent) & (ent > 0) & np.isfinite(s1) & (s1 < ent)
        dist = floor_dist((ent - s1) / ent, costs[eday, mkt[ev["ci"]]])
        ev["sf"] = np.where(ok, ent * (1.0 - dist), np.nan)
    ev["sf_struct_binding"] = ok & np.isclose(ev["sf"], s1)


def r_acct(ret, dist):
    """계좌 단위 R = 수익률 × min(W_CAP/RISK, 1/거리) = (비중 × 수익률) ÷ RISK — 비중 = min(W_CAP, RISK/거리)."""
    return np.asarray(ret, float) * np.minimum(W_CAP / RISK, 1.0 / np.asarray(dist, float))


def u1_share(S: dict) -> np.ndarray:
    """거래일 d 의 유니버스(d−1 까지 거래대금) 중 일봉 U1 ON(전날 행 ∧ DEF) 비율. 분모 = 그날 유니버스이면서 패널에 있는 종목."""
    M = S["MEMd"]
    on = S["DS"]["up_prev"][0] & M
    den = M.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, on.sum(axis=0) / den, np.nan)


def regime_on(kosdaq: pd.Series, udates, share, ma: int = REGIME_MA, thr: float = REGIME_U1_MIN) -> dict:
    """국면 ON(거래일 d) = d 보다 엄격히 앞선 마지막 지수 행의 종가 > 그 행까지의 MA``ma``(min_periods = ma) ∧ share[d] ≥ thr."""
    q = pd.Series(np.asarray(kosdaq, float), index=pd.DatetimeIndex(kosdaq.index)).sort_index().dropna()
    m = q.rolling(ma, min_periods=ma).mean()
    ud = np.asarray(udates, dtype="datetime64[ns]")
    r = np.searchsorted(q.index.values, ud, side="left") - 1
    ok = r >= 0
    rr = np.maximum(r, 0)
    close_prev = np.where(ok, q.to_numpy()[rr], np.nan)
    ma_prev = np.where(ok, m.to_numpy()[rr], np.nan)
    with np.errstate(invalid="ignore"):
        idx_on = np.isfinite(ma_prev) & (close_prev > ma_prev)
        sh = np.asarray(share, float)
        on = idx_on & np.isfinite(sh) & (sh >= thr)
    return {"on": on, "idx_on": idx_on, "share": sh, "close_prev": close_prev, "ma_prev": ma_prev,
            "prev_date": np.where(ok, q.index.values[rr], np.datetime64("NaT", "ns"))}


# --------------------------------------------------------------------------------------------- 후보·계좌

def cost_table(udates, markets=("KOSPI", "KOSDAQ")) -> np.ndarray:
    """거래일 × 시장 왕복 비용률(korea.cost_rt, 매도일 세율)."""
    from swing_it.features.korea import cost_rt

    return np.array([[cost_rt(pd.Timestamp(d).date(), mk) for mk in markets] for d in udates])


def candidates(S: dict, OC: dict, cfg: dict, tfi: int, costs: np.ndarray, mkt: np.ndarray, cost_mult: float,
               cfg_id: int = -1) -> dict:
    """설정 하나의 후보 사건 → 계좌용 배열. 시각은 분 단위 정수(봉 시작). 청산 시각 = 청산 봉 시작(손절 체결은 그 봉 안 —
    자리는 다음 봉 시작부터 빈다), 시가 청산은 그 봉 시작부터 자리가 빈다."""
    ev = S["ev"]
    o = OC[(variant_of(cfg), cfg["stop"])]
    k = cfg["up"]
    m = o["ok"] & ev["u" + k[1]]
    if cfg["spread"]:
        m &= ev["spread"]
    if not cfg["overnight"]:
        m &= ~ev["last"]
    reg = S.get("regime")
    n = len(S["ts_min"])
    reg_ev = (reg[S["dix"][np.minimum(ev["t"] + 1, n - 1)]] if reg is not None else np.zeros(len(ev["t"]), bool))
    if cfg.get("regime", "ignore") == "on":
        if reg is None:
            raise ValueError("regime 유전자 'on' 인데 S['regime'] 이 없다")
        m &= reg_ev
    idx = np.flatnonzero(m)
    ts = S["ts_min"]
    eb = ev["t"][idx] + 1
    xb = o["xb"][idx]
    real = o["realized"][idx]
    ent = ev["entry_px"][idx]
    stop = o["stop"][idx]
    dist = (ent - stop) / ent
    xpx = np.where(real, o["xpx"][idx], o["mtm"][idx])
    xday = np.where(real, S["dix"][np.maximum(xb, 0)], len(S["udates"]) - 1)
    cst = costs[xday, mkt[ev["ci"][idx]]] * cost_mult
    ret = xpx / ent - 1.0 - cst
    exit_ts = np.where(real, ts[np.maximum(xb, 0)], INF_TS)
    free = np.where(real, np.where(o["at_open"][idx], exit_ts, exit_ts + S["tf_min"]), INF_TS)
    rv = ev["relvol"][idx]
    return {"ci": ev["ci"][idx], "t": ev["t"][idx], "eb": eb, "xb": xb, "entry_ts": ts[eb], "exit_ts": exit_ts, "free_ts": free,
            "eday": S["dix"][eb], "xday": xday, "entry_px": ent, "exit_px": xpx, "stop": stop, "dist": dist, "ret": ret,
            "R": ret / dist, "Racct": r_acct(ret, dist), "reg_on": reg_ev[idx], "realized": real, "why": o["why"][idx], "last": ev["last"][idx],
            "relvol": np.where(np.isfinite(rv), rv, -np.inf), "tfi": np.full(len(idx), tfi, np.int64),
            "cfg": np.full(len(idx), cfg_id, np.int64), "ev_idx": idx, "n_bars": np.full(len(idx), n, np.int64),
            "month": month_of(ts[eb])}


CAND_KEYS = ("ci", "t", "eb", "xb", "entry_ts", "exit_ts", "free_ts", "eday", "xday", "entry_px", "exit_px", "stop", "dist", "ret",
             "R", "Racct", "reg_on", "realized", "why", "last", "relvol", "tfi", "cfg", "ev_idx", "n_bars", "month")


INT_KEYS = ("ci", "t", "eb", "xb", "entry_ts", "exit_ts", "free_ts", "eday", "xday", "why", "tfi", "cfg", "ev_idx", "n_bars")


def concat(parts: list) -> dict:
    parts = [p for p in parts if len(p["ci"])]
    if not parts:
        return {k: np.zeros(0, dtype=object if k == "month" else (np.int64 if k in INT_KEYS else (bool if k in ("realized", "last", "reg_on") else float)))
                for k in CAND_KEYS}
    return {k: np.concatenate([p[k] for p in parts]) for k in CAND_KEYS}


def take(c: dict, mask) -> dict:
    return {k: v[mask] for k, v in c.items()}


def order(c: dict) -> np.ndarray:
    """같은 봉 후보는 상대 거래량 내림차순(동률 = 봉 단위·종목 순)."""
    return np.lexsort((c["ci"], c["tfi"], -c["relvol"], c["entry_ts"]))


@_njit
def _account(entry_ts, free_ts, code, eday, entry_px, tfi, ret, dist, dayclose, slots, w_cap, risk, cash0, min_size):
    N = len(entry_ts)
    taken = np.zeros(N, np.bool_)
    size = np.zeros(N)
    pos = np.full(slots, -1, np.int64)
    cash = cash0
    for k in range(N):
        for s in range(slots):
            j = pos[s]
            if j >= 0 and free_ts[j] <= entry_ts[k]:
                cash += size[j] * (1.0 + ret[j])
                pos[s] = -1
        held = False
        n_open = 0
        free_slot = -1
        E = cash
        for s in range(slots):
            j = pos[s]
            if j < 0:
                if free_slot < 0:
                    free_slot = s
                continue
            n_open += 1
            if code[j] == code[k]:
                held = True
            mk = 1.0
            if eday[j] < eday[k]:
                px = dayclose[tfi[j], code[j], eday[k] - 1]
                if np.isfinite(px):
                    mk = px / entry_px[j]
            E += size[j] * mk
        if held or n_open >= slots:
            continue
        w = min(w_cap, risk / dist[k])
        sz = min(w * E, cash)
        if sz < min_size * E:
            continue
        taken[k] = True
        size[k] = sz
        cash -= sz
        pos[free_slot] = k
    return taken, size


def account(c: dict, dayclose: np.ndarray, slots: int = SLOTS) -> dict:
    """단일 계좌(1억, 슬롯 10, 비중 = min(10%, 0.5% ÷ 손절 거리), 자산 = 현금 + 보유분 전날 종가 평가). 같은 종목 보유 중이면 무시."""
    if len(c["ci"]) == 0:
        return {**c, "size": np.zeros(0)}
    o = order(c)
    c = take(c, o)
    taken, size = _account(c["entry_ts"].astype(np.int64), c["free_ts"].astype(np.int64), c["ci"].astype(np.int64),
                           c["eday"].astype(np.int64), c["entry_px"], c["tfi"].astype(np.int64), c["ret"], c["dist"], dayclose,
                           slots, W_CAP, RISK, CASH0, MIN_SIZE)
    out = take(c, taken)
    out["size"] = size[taken]
    return out


def equity_curve(tr: dict, dayclose: np.ndarray, udates, d0: int, d1: int, costs: np.ndarray, mkt: np.ndarray,
                 cost_mult: float) -> pd.DataFrame:
    """일별 장 마감 자산·노출(거래일 d0..d1). 보유분은 그날 종가 평가, 마지막 날 미청산분은 청산 비용을 뺀 값."""
    nd = d1 - d0 + 1
    pnl = np.zeros(nd)
    val = np.zeros(nd)
    for k in range(len(tr["ci"])):
        i, f, sz, ep = int(tr["ci"][k]), int(tr["tfi"][k]), float(tr["size"][k]), float(tr["entry_px"][k])
        a = int(tr["eday"][k]) - d0
        b = (int(tr["xday"][k]) - d0) if tr["realized"][k] else nd
        px = dayclose[f, i, d0:d1 + 1]
        for d in range(max(a, 0), min(b, nd)):
            v = sz * px[d] / ep if np.isfinite(px[d]) else sz
            val[d] += v
            pnl[d] += v - sz
        if tr["realized"][k] and b < nd:
            pnl[b:] += sz * float(tr["ret"][k])
        elif not tr["realized"][k]:
            last = sz * px[-1] / ep if np.isfinite(px[-1]) else sz
            pnl[-1] -= last * costs[d1, mkt[i]] * cost_mult
    eq = CASH0 + pnl
    return pd.DataFrame({"equity": eq, "expo": val / eq}, index=pd.DatetimeIndex(udates[d0:d1 + 1]))


# --------------------------------------------------------------------------------------------- 워크포워드

def month_start_ts(month: str) -> int:
    return int(np.datetime64(pd.Timestamp(month + "-01"), "m").astype(np.int64))


def robust_sum(x, drop_top: int = 0) -> float:
    """합에서 큰 순 ``drop_top`` 건을 뺀다 — 단 양수인 것만 뺀다(RULINGS_V2["train_score"]: 손실 거래를 빼서 점수가 오르지 않게)."""
    x = np.asarray(x, float)
    if drop_top <= 0 or len(x) == 0:
        return float(x.sum())
    top = np.sort(x)[::-1][:drop_top]
    return float(x.sum() - top[top > 0].sum())


def train_scores(acc: dict, months: list, key: str = "R", drop_top: int = 0) -> tuple:
    """폴드 m: 학습 = 계좌 시작 ~ m−1 말. 그 전에 청산 체결된(청산 봉 시작 < m 첫날) 거래의 ``key`` 합(MT-2 = R_acct 합 − 상위 3건)·수.
    계좌가 인과적이라 전 기간 한 번 돌린 장부를 잘라도 학습 창만 돌린 장부와 같다(테스트 test_selection_ignores_test_month)."""
    sc, nt = [], []
    for mo in months:
        cut = month_start_ts(mo)
        m = acc["realized"] & (acc["exit_ts"] < cut)
        sc.append(robust_sum(acc[key][m], drop_top))
        nt.append(int(m.sum()))
    return np.array(sc), np.array(nt)


def month_of(ts_min) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(np.asarray(ts_min, dtype="datetime64[m]")).strftime("%Y-%m"))


def oos_stream(cands_by_cfg: list, picks, months: list) -> dict:
    """폴드 m 에 뽑힌 설정의 후보 중 진입이 m 달인 것만 모은다(장부 하나로 다시 돌릴 입력)."""
    parts = []
    for mo, p in zip(months, picks):
        c = cands_by_cfg[p]
        parts.append(take(c, c["month"] == mo))
    return concat(parts)


def cluster_boot(R, cluster, n_boot: int = 10000, seed: int = 0) -> tuple:
    """R 평균과 군집(진입 거래일) 복원 추출 95% 구간 — golden_bt_wf.cluster_boot 와 같은 식."""
    R = np.asarray(R, dtype=float)
    if len(R) == 0:
        return float("nan"), float("nan"), float("nan")
    _, inv = np.unique(np.asarray(cluster), return_inverse=True)
    g = int(inv.max()) + 1
    s = np.bincount(inv, weights=R, minlength=g)
    c = np.bincount(inv, minlength=g).astype(float)
    pick = np.random.default_rng(seed).integers(0, g, size=(n_boot, g))
    boot = s[pick].sum(axis=1) / c[pick].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(R.mean()), float(lo), float(hi)


def rstats(R) -> dict:
    R = np.asarray(R, dtype=float)
    if len(R) == 0:
        return {"n": 0}
    w, lo = R[R > 0], R[R <= 0]
    pr = float(w.mean() / abs(lo.mean())) if len(w) and len(lo) and lo.mean() < 0 else float("nan")
    return {"n": int(len(R)), "R_sum": float(R.sum()), "R_mean": float(R.mean()), "R_median": float(np.median(R)),
            "win_rate": float(len(w) / len(R)), "avg_win_R": float(w.mean()) if len(w) else float("nan"),
            "avg_loss_R": float(lo.mean()) if len(lo) else float("nan"), "payoff": pr,
            "breakeven_win_rate": float(1 / (1 + pr)) if np.isfinite(pr) else float("nan")}


# --------------------------------------------------------------------------------------------- 전체 흐름(실행기·테스트 공용)

def prepare(Ss: list, stops: tuple = ("S1", "S2")) -> tuple:
    """봉 단위별 사건 경로(청산 변형 × 손절)와 계좌 평가용 일 종가 더미(봉 단위 × 종목 × 거래일)."""
    for S in Ss[1:]:
        if not (np.array_equal(S["udates"], Ss[0]["udates"]) and S["codes"] == Ss[0]["codes"]):
            raise ValueError("봉 단위끼리 거래일·종목이 같아야 한다")
    OCs = [outcomes(S, stops) for S in Ss]
    dayclose = np.ascontiguousarray(np.stack([S["dcl"] for S in Ss]))
    return OCs, dayclose


def all_candidates(Ss: list, OCs: list, costs: np.ndarray, mkt: np.ndarray, cost_mult: float, axes: dict = AXES) -> list:
    tfs = [S["tf"] for S in Ss]
    return [candidates(Ss[tfs.index(c["tf"])], OCs[tfs.index(c["tf"])], c, tfs.index(c["tf"]), costs, mkt, cost_mult, j)
            for j, c in enumerate(menu(axes))]


def wf_select(cands: list, dayclose: np.ndarray, months: list, key: str = "R", drop_top: int = 0, axes: dict = AXES) -> dict:
    """설정마다 전 기간 단일 계좌 한 번 → 폴드별 학습 점수(그 달 전에 청산된 ``key`` 합, 상위 ``drop_top`` 건 제외) → 이웃 중앙값 최대."""
    sc = np.zeros((len(cands), len(months)))
    nt = np.zeros((len(cands), len(months)), np.int64)
    for j, c in enumerate(cands):
        acc = account(c, dayclose)
        sc[j], nt[j] = train_scores(acc, months, key, drop_top)
    picks, nmed = [], []
    for f in range(len(months)):
        p = select(sc[:, f], nt[:, f], axes)
        picks.append(p)
        nmed.append(float(neighbor_medians(sc[:, f], axes)[p]))
    return {"picks": picks, "scores": sc, "n_train": nt, "nbr_median": nmed}


def entry_pool(S: dict) -> np.ndarray:
    """널 ②의 무작위 진입 풀(봉 t 신호 → t+1 시가): 유니버스 ∧ 종가 유효 ∧ 자료 유효(80봉 중 72) ∧ 일봉 DEF ∧ 다음 봉 시가 있음 ∧ 상한가 시가 아님.
    일봉 상승 ON 은 짝 거래의 up 유전자로 따로 건다."""
    Op = S["O"]
    On = np.full(Op.shape, np.nan)
    On[:, :-1] = Op[:, 1:]
    pn = np.full(Op.shape, np.nan)
    pn[:, :-1] = S["prevc"][:, 1:]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(On) & (On > 0) & ~(On >= pn * LIM_UP)
    return S["MEM"] & S["fin"] & S["win"] & S["DS"]["def_prev"][:, S["dix"]] & ok


def random_entries(Ss: list, pools: list, tr: dict, rng, costs: np.ndarray, mkt: np.ndarray, cost_mult: float,
                   cache: dict | None = None, axes: dict = AXES) -> dict:
    """널 ② 한 번: 전략 OOS 계좌의 진입마다 같은 봉 단위·같은 신호 봉(같은 날·같은 시각대)에서 그 폴드 up 유전자 ON 인 아무 종목을 골라
    다음 봉 시가에 산다. 손절 거리 = 짝 거래의 손절 거리(%, RULINGS["null2_stop"]), 청산 = 짝 설정의 exit. 반환: 실현 R·R_acct 배열 등."""
    M = menu(axes)
    cache = {} if cache is None else cache
    pick = np.full(len(tr["ci"]), -1, np.int64)
    for k in range(len(tr["ci"])):
        f, t, cfg = int(tr["tfi"][k]), int(tr["t"][k]), M[int(tr["cfg"][k])]
        key = (f, t, cfg["up"])
        if key not in cache:
            S = Ss[f]
            u = S["DS"]["up_prev"][int(cfg["up"][1]) - 1][:, S["dix"][t]]
            cache[key] = np.flatnonzero(pools[f][:, t] & u)
        cand = cache[key]
        if len(cand):
            pick[k] = cand[rng.integers(len(cand))]
    R = np.full(len(pick), np.nan)
    Ra = np.full(len(pick), np.nan)
    real = np.zeros(len(pick), bool)
    names = {v[0]: (v[1], v[2]) for v in exit_variants()}
    var = np.array([variant_of(M[int(c)]) for c in tr["cfg"]])
    for f in range(len(Ss)):
        S = Ss[f]
        for vn in np.unique(var):
            sel = np.flatnonzero((tr["tfi"] == f) & (var == vn) & (pick >= 0))
            if len(sel) == 0:
                continue
            ci, eb = pick[sel], tr["t"][sel] + 1
            ent = S["O"][ci, eb]
            stop = ent * (1.0 - tr["dist"][sel])
            mode, upk = names[vn]
            r = run_paths(S, ci, eb, stop, mode, upk)
            xday = S["dix"][np.maximum(r["xb"], 0)]
            cst = costs[xday, mkt[ci]] * cost_mult
            ret = r["xpx"] / ent - 1.0 - cst
            R[sel] = ret / tr["dist"][sel]
            Ra[sel] = r_acct(ret, tr["dist"][sel])
            real[sel] = r["realized"]
    return {"code": pick, "R": R, "Racct": Ra, "realized": real}
