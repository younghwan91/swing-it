"""일봉 상승 중 아래 봉 전환 — 워크포워드 최적화 실행기. **오염 구간(2025-08~2026-11), 통과해도 전진 모의매매 팔 등록 자격까지.**

정본: ``research/logs/mtf_transition/PREREG.md`` (4fb9525). 순수 부품은 ``research/signals/mtf_transition.py``, 누출 테스트는
``tests/test_mtf_transition.py``. 해석은 아래 RULINGS(결과 보기 전 코드에 적은 것) — RESULTS.md 에 그대로 옮긴다.

사용(작업 폴더에서):
``set -a && . ~/git/quant-airflow/.env && set +a && NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --with numba --with matplotlib --extra pg --extra dev python research/experiments/mtf_transition_wf.py [--smoke]``
``--smoke`` 는 모양·건수·시간만 출력하고 결과 파일을 쓰지 않는다(널 3회).
``--v2`` 는 재등록 MT-2(PREREG 맨 끝, 27334ef): 계좌 단위 R·손절 거리 하한·학습 점수 상위 3건 제외·국면(regime) 유전자 — 결과는 ``*_v2`` 파일
(RESULTS_v2.md·results_v2.json·oos_trades_v2.csv·equity_v2.png·r_hist_v2.png·trade_grid_v2.png), TRIALS 한 줄(trial ``mtf_transition_v2``).
인자 없이 돌리면 1차(4fb9525) 그대로다.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]
from research.signals import mtf_transition as T  # noqa: E402

OUT = REPO / "research" / "logs" / "mtf_transition"
SCRATCH = Path.home() / ".cache" / "cc-swing" / "scratch" / "mtf_transition"
TAG = "오염 구간(2025-08~2026-11) — 통과해도 전진 모의매매 팔 등록 자격까지, 실거래 근거 아님"
PREREG = "research/logs/mtf_transition/PREREG.md (4fb9525)"
DATA = ("2025-08-01", "2026-11-13")
DAILY_START = "2024-06-01"
MONTHS = [f"2026-{m:02d}" for m in range(2, 11)]
COST_JUDGE = 2.0
MIN_TRADES = 100
N_NULL = 300
SEED = 20261116
INTRA_OPEN, INTRA_CLOSE_EXCL = "09:00:00", "15:33:00"
TFS = ("15m", "30m", "60m")

RULINGS = {
    "bars": "분봉은 DB 에서 15분 단위로 묶고(15:30~15:32 단일가 분봉은 15:29 로 옮겨 15:15 봉에 합침 — 탐색 load_tf 규약) 30·60분봉은 그 15분봉을 "
            "pandas 로 다시 묶는다. first/max/min/last/sum 묶음은 결합적이라 5분봉에서 묶은 것과 같다.",
    "dead_bar": "거래량 0(또는 결측) 봉은 시고저종 모두 결측(체결 없음). 진입 봉 시가가 결측이면 그 사건은 진입하지 않는다.",
    "universe": "탐색 topdown_intraday_pit 규약 그대로: d 보다 앞선 마지막 일봉 행까지 20거래일(min 15) 평균 거래대금 순위 ≤ 300, "
                "보통주·KOSPI/KOSDAQ(load_listing_meta excluded 없음). 패널 종목 = 2025-08 이후 한 번이라도 대상이 된 합집합.",
    "daily_fill": "일봉 정문 unpatched='nan' 칸(2026-09-14~)은 그날 분봉 마지막 봉 종가로 채운다(탐색 pit.signals 규약). 그날 행은 다음 날부터만 쓰인다.",
    "split_adjust": "분봉 봉 사이 로그 변화가 ±30% 가격제한폭 밖이면 분할·병합으로 보고 구간마다 (일봉 수정종가/분봉 일 종가) 중앙값을 곱한다(탐색 규약). "
                    "구간 안 상수배라 이평선 순서·수익률·R 은 불변이고, 일봉 결측 채움 값에만 닿는다.",
    "event_validity": "전환 사건에 탐색 trans.transitions 의 자료 유효성 조건(최근 80봉 중 72봉 이상 체결)을 그대로 건다 — PREREG 의 공개 탐색 숫자가 이 조건으로 나왔다. "
                      "20봉 재발 묶음은 탐색 pit.dedupe 사슬 규약(직전 조건 봉과 20봉 초과 떨어져야 새 사건), 묶은 뒤 그날 유니버스 안 것만.",
    "up_def": "일봉 상태 정의 가능(DEF) = MA120·5일 전 MA20·20일 전 MA60 모두 있음. 사건은 U1 ON ∧ DEF 만 남긴다(U2·U3 ⊂ U1).",
    "sigma": "σ60 = 아래 봉 ffill 종가의 봉 사이 로그수익(밤 넘김 포함) 60봉 표본표준편차(ddof 1), 전환 봉까지.",
    "s1": "S1 = 전환 봉 직전부터 거슬러 20봉 안 가장 가까운 역배열 봉이 속한 연속 역배열 구간의 시작 ~ 전환 봉까지 저가 최저(탐색 trans.struct_stop 그대로).",
    "s2": "S2 손절가 = min(S1, 진입가 × (1 − 2·σ60)) — 거리 하한을 진입가 대비 비율 2σ60 으로 읽는다.",
    "stop_fill": "손절: 봉 저가 ≤ 손절가면 그 봉에서 min(손절가, 시가). 시가가 손절가 아래이면서 하한가 시가(전날 종가 × 0.705 이하)면 다음 봉 시가로 미룬다. "
                 "손절은 진입 봉부터. 손절가 ≥ 진입가면 거래하지 않는다.",
    "limits": "상·하한가 시가 = 그 봉 시가 ≥ 전날 종가 × 1.295 / ≤ × 0.705(전날 종가 = 직전 거래일 마지막 봉 ffill 종가, 같은 보정 척도). "
              "상한가 시가 진입은 건너뛰고, 하한가 시가 청산은 체결될 때까지 다음 봉으로 미룬다.",
    "exit_order": "봉마다 ① 대기 청산(시가) ② 손절 ③ 종가 조건(X1 = 그 봉 종가 < 아래 봉 MA60 → 다음 봉 시가; X2~X4 = 그날 마지막 봉에서 그날 일봉 행 판정 → "
                  "다음 거래일 첫 체결 봉 시가). X4 의 상승 정의는 그 설정의 up 유전자. 진입 당일 종가부터 판정한다.",
    "daily_close": "일봉 청산 판정의 종가·MA 는 일봉 정문 수정종가(채움 포함, 정지일 ffill). 아래 봉 종가와 섞지 않는다.",
    "unrealized": "2026-11-13 마지막 봉까지 청산 체결이 없으면(청산 판정 뒤 체결 봉이 자료 밖인 것 포함) 미청산 — 실현 R 바에서 빼고 마지막 유한 종가로 따로 평가.",
    "cost": "왕복 비용 = korea.cost_rt(청산 체결일 세율, 종목 시장) × 배수, 손익에서 한 번 뺀다. 미청산은 마지막 날 세율.",
    "R": "R = (청산가/진입가 − 1 − 비용) ÷ 손절 거리(진입가 대비) — PREREG 식에서 비중·자산이 약분된 것.",
    "account": "단일 계좌 1억·슬롯 10. 같은 시각 체결 순서: 시가 청산 → 진입. 봉 안 손절로 빈 자리는 다음 봉 시작부터. 비중 = min(10%, 0.5% ÷ 손절 거리) × 자산, "
               "자산 = 현금 + 보유분(전날 마지막 봉 종가 평가, 오늘 산 것은 원가).",
    "cash": "매수 금액 = min(비중 × 자산, 현금). 그것이 자산의 0.1% 미만이면 매수하지 않는다(현금 소진).",
    "priority": "같은 봉 후보가 넘치면 전환 봉 상대 거래량(그 시각대 직전 20거래일 평균 거래량 대비, 이력 5일 미만이면 최하위) 내림차순, 동률은 봉 단위·종목 순.",
    "train_score": "폴드 m 학습 점수 = 2025-08 부터 돈 그 설정의 단일 계좌에서 청산 체결 봉이 m 첫날 전인 거래의 R 합(비용 2배). m 전에 못 끝난 거래는 뺀다(실현만). "
                   "계좌가 인과적이라 전 기간 한 번 돈 장부를 자른 것과 학습 창만 돈 것이 같다(테스트로 확인).",
    "neighbors": "이웃 = 자기 + 유전자 하나만 다른 설정(모든 유전자를 범주로 — tf 도 다른 두 값 모두), 288 칸에서 자기 포함 11 개의 중앙값. "
                 "동률이면 그 설정 자신의 학습 실현 거래 수 많은 쪽, 그래도 같으면 메뉴 순서 앞(golden_bt 관례).",
    "oos": "OOS = 폴드 m 에 뽑힌 설정의 진입이 m 달인 후보를 모아 2026-02 첫날 1억 새 계좌 하나로 다시 돌린 것(앞 달 보유분이 다음 달 자리를 잡는다). "
           "폴드(달) 소속 = 진입 체결 시각의 달.",
    "bar2": "거래일 군집 = 진입 체결 거래일. 복원 추출 10,000 회, 95% 구간 2.5% 분위가 하한.",
    "bar3": "널 ① = 폴드마다 288 중 무작위 설정(복원, 균등) → 같은 OOS 계좌 조립·실행 → 실현 R 합. 300 회, 95 백분위.",
    "bar4": "널 ② = 전략 OOS 계좌의 진입마다(미청산 포함, 같은 수) 같은 봉 단위·같은 신호 봉에서 [유니버스 ∧ 그 폴드 up 유전자 ON(전날까지) ∧ DEF ∧ 80봉 중 72 유효 ∧ "
            "다음 봉 시가 있음 ∧ 상한가 시가 아님] 종목을 균등 추출(전략 종목 포함 가능)해 다음 봉 시가에 산다. 손절 거리 = 짝 거래의 손절 거리(%) — "
            "무작위 봉에는 전환 전 역배열 구간이 없어 S1 이 정의되지 않으므로 짝의 거리(S2 하한 포함)를 그대로 씌운다. 청산 = 짝 설정의 exit(그 종목 자료). "
            "슬롯 재배정 없이 1:1 로 짝짓고(같은 수 유지), 비중 계산은 R 에서 약분된다. 실현 R 합, 300 회, 95 백분위.",
    "bar5": "한 종목씩 빼기 = 거래 제거(재시뮬 아님): 종목별 실현 R 합 최대 1 종목, 상위 5 종목의 거래를 빼고 실현 R 합.",
    "bar6": "단일 계좌 OOS 수익 = 2026-11-13 장 마감 자산(미청산은 종가 평가 − 청산 비용) ÷ 1억 − 1. 노출 맞춘 KOSDAQ = 평균 일별 노출(장 마감 보유 평가 ÷ 자산) × "
            "KOSDAQ 수익(2026-02 첫 거래일 전날 종가 → 2026-11-13 종가). 그림의 노출 맞춘 곡선은 전날 노출 × 그날 지수 수익의 일별 복리(표시용).",
    "bar7": "양수 OOS 달 = 폴드(진입 달)별 OOS 계좌 실현 R 합 > 0 인 달 수. 계좌 월 수익 양수 달 수는 보고만.",
    "betas": "월 수익(계좌 자산, 2026-02 ~ 2026-11 부분 월 포함) ~ KOSPI, ~ KOSDAQ 각각 단순 OLS 와 둘 다 넣은 OLS 의 기울기·절편 t(OLS 표준오차).",
    "overnight_report": "밤 넘김 별도 = 하루 마지막 봉 사건(다음 날 첫 봉 시가 진입) 거래와 그 밖 거래를 나눠 보고.",
}


# =============================================================================================== 자료

def _connect():
    from swing_it.storage import connect

    if not os.environ.get("KR_QUANT_DB"):
        raise SystemExit("KR_QUANT_DB 가 없다 — set -a; . ~/git/quant-airflow/.env; set +a")
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return con


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


_T0 = time.time()


def load() -> dict:
    """일봉 정문(종가·거래대금) + 합집합 종목 15분봉. 캐시 ``SCRATCH``(없으면 DB 에서 만든다, 읽기 전용)."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    fd, fm = SCRATCH / "daily.pkl", SCRATCH / "m15.pkl"
    if fd.exists() and fm.exists():
        d = pd.read_pickle(fd)
        d["m15"] = pd.read_pickle(fm)
        log(f"캐시 읽음 {SCRATCH}")
        return d
    from swing_it.features.korea import UNIVERSE_MARKETS, load_listing_meta
    from swing_it.storage import read_prices_wide

    con = _connect()
    pw = read_prices_wide(con, cols=("close", "trade_value"), start=DAILY_START, end=DATA[1], session="regular", unpatched="nan")
    dc, tv = pw["close"], pw["trade_value"]
    dc.index = pd.DatetimeIndex(dc.index)
    tv.index = pd.DatetimeIndex(tv.index)
    meta = load_listing_meta(con)
    ok = meta["excluded"].isna() & meta["market"].isin(UNIVERSE_MARKETS)
    elig = [c for c in tv.columns if c in set(meta.index[ok])]
    adv = tv[elig].rolling(T.ADV_WINDOW, min_periods=T.ADV_MIN_PERIODS).mean().shift(1)
    rk = adv.rank(axis=1, ascending=False, method="first")
    mem = (rk <= T.TOPN) & (adv.index >= DATA[0])[:, None]
    union = sorted(mem.columns[mem.any(axis=0)])
    log(f"일봉 {dc.shape}, 보통주·KOSPI/KOSDAQ {len(elig)}, 합집합 {len(union)}, krx_session {pw.krx_session}")
    q = ("SELECT code, time_bucket('15 minutes', CASE WHEN ts::time >= '15:30:00' THEN ts::date + time '15:29:00' ELSE ts END) AS ts, "
         "first(open, ts) AS open, max(high) AS high, min(low) AS low, last(close, ts) AS close, sum(volume)::float8 AS volume FROM minute_bars "
         "WHERE code = ANY(%(codes)s) AND ts >= %(lo)s AND ts < %(hi)s AND ts::time >= %(o)s AND ts::time < %(c)s GROUP BY 1, 2 ORDER BY 1, 2")
    hi = (pd.Timestamp(DATA[1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    for a in range(0, len(union), 20):
        parts.append(pd.read_sql_query(q, con, params={"codes": union[a:a + 20], "lo": DATA[0], "hi": hi, "o": INTRA_OPEN, "c": INTRA_CLOSE_EXCL}))
        if (a // 20) % 10 == 0:
            log(f"  15분봉 {min(a + 20, len(union))}/{len(union)}")
    m = pd.concat(parts, ignore_index=True)
    m["ts"] = pd.to_datetime(m["ts"])
    d = {"close": dc, "tv": tv, "meta": meta, "elig": elig, "union": union, "krx_session": dict(pw.krx_session)}
    pd.to_pickle(d, fd)
    pd.to_pickle(m, fm)
    d["m15"] = m
    log(f"15분봉 {len(m):,} 행, 종목 {m.code.nunique()}")
    return d


def load_index() -> pd.DataFrame:
    from swing_it.storage import read_index

    con = _connect()
    ix = read_index(con, ["001", "101"], cols=("code", "date", "close"), start="2025-07-01", end=DATA[1], with_level=True,
                    order_by_date=True, parse_dates=True)
    w = ix.pivot(index="date", columns="code", values="level").rename(columns={"001": "kospi", "101": "kosdaq"})
    w.index = pd.DatetimeIndex(w.index)
    return w


def build_all(D: dict) -> list:
    codes = sorted(D["m15"]["code"].unique())
    Ss = []
    for tf in TFS:
        t0 = time.time()
        m = D["m15"] if tf == "15m" else T.aggregate(D["m15"], T.TF_MIN[tf])
        S = T.build_tf(m, D["close"], D["tv"], D["elig"], tf, codes=codes)
        Ss.append(S)
        log(f"{tf}: 종목 {len(codes)} × 봉 {len(S['days']):,}, 거래일 {len(S['udates'])}, 사건(U1·DEF) {len(S['ev']['t']):,} "
            f"({time.time() - t0:.0f}s)")
    return Ss


# =============================================================================================== 통계 보조

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def _p95(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, 0.95)) if len(x) else float("nan")


def ols(y, X) -> dict:
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    n, p = X.shape
    if n <= p:
        return {"n": n}
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    V = (e @ e) / (n - p) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(V))
    return {"n": n, "alpha": float(b[0]), "t_alpha": float(b[0] / se[0]), "betas": [float(v) for v in b[1:]],
            "t_betas": [float(v) for v in b[1:] / se[1:]]}


def cfg_str(c: dict) -> str:
    return f"{c['tf']}·{c['up']}·{'간격' if c['spread'] else '간격없음'}·{'밤허용' if c['overnight'] else '밤제외'}·{c['exit']}·{c['stop']}"


def month_eq(eq: pd.Series, base: float, prev_day) -> pd.Series:
    s = pd.concat([pd.Series([base], index=pd.DatetimeIndex([prev_day])), eq])
    return s.resample("ME").last().pct_change().dropna()


# =============================================================================================== 누출 점검(실자료)

def leak_real(D: dict, Ss: list, W: dict, codes: list, costs: np.ndarray, mkt: np.ndarray, v2_kosdaq: pd.Series | None = None,
              RG: dict | None = None) -> dict:
    """실자료로 PREREG 누출 점검 두 가지를 다시 돈다: (a) 날짜 5개 절단 → 유니버스·up·사건·선택, (b) 폴드 2026-06 부터 가격 교란 → 그 폴드까지 선택.
    ``v2_kosdaq`` 를 주면 MT-2 선택(국면·손절 하한·R_acct)으로 돌고, 절단 사본의 국면(지수도 그날 전까지만)이 원본과 같은지도 본다."""
    rng = np.random.default_rng(SEED + 7)
    ud = pd.DatetimeIndex(Ss[0]["udates"])
    pool = np.flatnonzero((ud >= "2026-02-01") & (ud < "2026-11-13"))
    ks = np.sort(rng.choice(pool, 5, replace=False))
    out = {"truncation": []}

    def sel(Sx, q=None):
        if v2_kosdaq is not None:
            return select_v2(Sx, q, costs[: len(Sx[0]["udates"])], mkt)[0]
        OCx, dcx = T.prepare(Sx)
        return T.wf_select(T.all_candidates(Sx, OCx, costs[: len(Sx[0]["udates"])], mkt, COST_JUDGE), dcx, MONTHS)

    for k in ks:
        kday = ud[k]
        hi = kday + pd.Timedelta(hours=int(rng.integers(10, 16)))
        m = D["m15"][D["m15"].ts < hi]
        Tt = []
        for tf in TFS:
            mm = m if tf == "15m" else T.aggregate(m, T.TF_MIN[tf])
            Tt.append(T.build_tf(mm, D["close"][D["close"].index < kday], D["tv"][D["tv"].index < kday], D["elig"], tf, codes=codes))
        r = {"day": str(kday.date()), "hi": str(hi), "tf": {}}
        for F, X in zip(Ss, Tt):
            nd, nb_ = len(X["udates"]), len(X["days"])
            uni = sum(F["memday"][pd.Timestamp(d)] != X["memday"][pd.Timestamp(d)] for d in X["udates"])
            up = int((F["DS"]["up_prev"][:, :, :nd] != X["DS"]["up_prev"]).sum())
            mem = int((F["MEM"][:, :nb_] != X["MEM"]).sum())

            def evs(S, strict):
                ev = S["ev"]
                ts = S["days"][ev["t"]]
                ok = ts < (hi - pd.Timedelta(minutes=S["tf_min"]) if strict else hi)
                return {(ev["ci"][j], ts[j], bool(ev["u1"][j]), bool(ev["u2"][j]), bool(ev["u3"][j]))
                        + ((bool(ev["spread"][j]), bool(ev["last"][j]), round(float(ev["s1"][j] / S["Cf"][ev["ci"][j], ev["t"][j]]), 9)) if strict else ())
                        for j in np.flatnonzero(ok)}
            a, b = evs(F, False), evs(X, False)
            r["tf"][F["tf"]] = {"univ_days_diff": int(uni), "up_cells_diff": up, "mem_cells_diff": mem, "events": len(a),
                                "events_diff": len(a ^ b), "events_feat_diff": len(evs(F, True) ^ evs(X, True))}
        folds = [f for f, mo in enumerate(MONTHS) if pd.Timestamp(mo + "-01") <= kday]
        if v2_kosdaq is not None:
            qt = v2_kosdaq[v2_kosdaq.index < kday]
            Wt = sel(Tt, qt)
            nd = len(Tt[0]["udates"])
            rg_t = Tt[0]["_RG"]
            r["regime_days_diff"] = int((rg_t["on"] != RG["on"][:nd]).sum())
            r["u1_share_max_abs_diff"] = float(np.nanmax(np.abs(rg_t["share"] - RG["share"][:nd])))
            r["regime_days"] = nd
        else:
            Wt = sel(Tt)
        r["folds_checked"] = [MONTHS[f] for f in folds]
        r["picks_same"] = bool(all(Wt["picks"][f] == W["picks"][f] for f in folds))
        r["scores_same"] = bool(all(np.array_equal(Wt["scores"][:, f], W["scores"][:, f]) for f in folds))
        r["scores_max_abs_diff"] = float(max([np.max(np.abs(Wt["scores"][:, f] - W["scores"][:, f])) for f in folds] or [0.0]))
        out["truncation"].append(r)
        log(f"  누출(절단) {kday.date()} {hi:%H:%M}: 폴드 {len(folds)} 선택 같음 {r['picks_same']} 점수 같음 {r['scores_same']} "
            f"국면 다른 날 {r.get('regime_days_diff', '—')} · "
            + " ".join(f"{k_} 사건차 {v['events_diff']}/{v['events']} 특징차 {v['events_feat_diff']} 유니버스차 {v['univ_days_diff']} up차 {v['up_cells_diff']}"
                       for k_, v in r["tf"].items()))
    # (b) 교란
    start = "2026-06-01"
    prng = np.random.default_rng(SEED + 11)
    m = D["m15"].copy()
    s_ = (m.ts >= start).to_numpy()
    f_ = np.exp(prng.normal(0, 0.03, int(s_.sum())))
    for col in ("open", "high", "low", "close"):
        m.loc[s_, col] = m.loc[s_, col].to_numpy() * f_
    m.loc[s_, "volume"] = m.loc[s_, "volume"].to_numpy() * np.exp(prng.normal(0, 0.5, int(s_.sum())))
    dc2, tv2 = D["close"].copy(), D["tv"].copy()
    rs = dc2.index >= start
    dc2.loc[rs] = dc2.loc[rs].to_numpy() * np.exp(prng.normal(0, 0.03, dc2.loc[rs].shape))
    tv2.loc[rs] = tv2.loc[rs].to_numpy() * np.exp(prng.normal(0, 0.8, tv2.loc[rs].shape))
    Sp = [T.build_tf(m if tf == "15m" else T.aggregate(m, T.TF_MIN[tf]), dc2, tv2, D["elig"], tf, codes=codes) for tf in TFS]
    if v2_kosdaq is not None:
        q2 = v2_kosdaq.copy()
        qs = q2.index >= start
        q2.loc[qs] = q2.loc[qs].to_numpy() * np.exp(prng.normal(0, 0.03, int(qs.sum())))
        Wp = sel(Sp, q2)
    else:
        Wp = sel(Sp)
    f0 = MONTHS.index("2026-06")
    out["perturb"] = {"from": start, "folds_checked": MONTHS[: f0 + 1],
                      "scores_same": bool(np.array_equal(Wp["scores"][:, : f0 + 1], W["scores"][:, : f0 + 1])),
                      "scores_max_abs_diff": float(np.max(np.abs(Wp["scores"][:, : f0 + 1] - W["scores"][:, : f0 + 1]))),
                      "picks_same": bool(Wp["picks"][: f0 + 1] == W["picks"][: f0 + 1]),
                      "later_scores_changed": bool(not np.array_equal(Wp["scores"][:, f0 + 1:], W["scores"][:, f0 + 1:]))}
    log(f"  누출(교란 {start}~): {out['perturb']}")
    return out


# =============================================================================================== 실행

def run(smoke: bool = False) -> dict:
    n_null = 3 if smoke else N_NULL
    D = load()
    Ss = build_all(D)
    t0 = time.time()
    OCs, dayclose = T.prepare(Ss)
    log(f"경로 12 변형 × 3 봉 단위 ({time.time() - t0:.0f}s)")
    udates, codes = Ss[0]["udates"], Ss[0]["codes"]
    mk = D["meta"]["market"].reindex(codes)
    if not mk.isin(["KOSPI", "KOSDAQ"]).all():
        raise ValueError("패널 종목 시장이 KOSPI/KOSDAQ 밖")
    mkt = (mk == "KOSDAQ").astype(np.int64).to_numpy()
    costs = T.cost_table(udates)
    C2 = T.all_candidates(Ss, OCs, costs, mkt, COST_JUDGE)
    C1 = T.all_candidates(Ss, OCs, costs, mkt, 1.0)
    log(f"후보: 설정 288, 후보 수 중앙 {int(np.median([len(c['ci']) for c in C2]))} (최소 {min(len(c['ci']) for c in C2)}, 최대 {max(len(c['ci']) for c in C2)})")
    t0 = time.time()
    W = T.wf_select(C2, dayclose, MONTHS)
    log(f"워크포워드 선택 ({time.time() - t0:.1f}s)")
    picks = W["picks"]
    acc = T.account(T.oos_stream(C2, picks, MONTHS), dayclose)
    acc1 = T.account(T.oos_stream(C1, picks, MONTHS), dayclose)
    real = acc["realized"]
    log(f"OOS 계좌: 진입 {len(acc['ci'])}, 실현 {int(real.sum())}, 미청산 {int((~real).sum())}")
    # 널 ①
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    d0 = int(np.searchsorted(udates, np.datetime64("2026-02-01")))
    d1 = len(udates) - 1
    null1, null1_eq, null1_n = [], [], []
    for _ in range(n_null):
        p = rng.integers(0, len(C2), len(MONTHS))
        a = T.account(T.oos_stream(C2, p, MONTHS), dayclose)
        null1.append(float(a["R"][a["realized"]].sum()))
        null1_n.append(int(a["realized"].sum()))
        null1_eq.append(T.equity_curve(a, dayclose, udates, d0, d1, costs, mkt, COST_JUDGE)["equity"].to_numpy() / T.CASH0)
    log(f"널 ① {n_null} 회 ({time.time() - t0:.0f}s)")
    # 널 ②
    t0 = time.time()
    pools = [T.entry_pool(S) for S in Ss]
    cache: dict = {}
    rng2 = np.random.default_rng(SEED + 1)
    null2, null2_n, null2_R, null2_miss = [], [], [], []
    for _ in range(n_null):
        x = T.random_entries(Ss, pools, acc, rng2, costs, mkt, COST_JUDGE, cache)
        rr = x["R"][x["realized"]]
        null2.append(float(np.sum(rr)))
        null2_n.append(int(x["realized"].sum()))
        null2_miss.append(int((x["code"] < 0).sum()))
        null2_R.append(rr)
    log(f"널 ② {n_null} 회 ({time.time() - t0:.0f}s), 짝 못 찾음 최대 {max(null2_miss) if null2_miss else 0}")
    if smoke:
        log("스모크: 결과 계산·파일 쓰기 생략")
        return {}
    ix = load_index()
    eq = T.equity_curve(acc, dayclose, udates, d0, d1, costs, mkt, COST_JUDGE)
    eq1 = T.equity_curve(acc1, dayclose, udates, d0, d1, costs, mkt, 1.0)
    log("누출 점검(실자료) …")
    LK = leak_real(D, Ss, W, codes, costs, mkt)
    res = assemble(D, Ss, W, acc, acc1, eq, eq1, ix, null1, null1_n, null2, null2_n, null2_miss, udates, d0, codes, LK)
    figures(D, Ss, acc, eq, ix, null1_eq, null2_R, res, udates, d0)
    write(res)
    return res


def assemble(D, Ss, W, acc, acc1, eq, eq1, ix, null1, null1_n, null2, null2_n, null2_miss, udates, d0, codes, LK) -> dict:
    M = T.menu()
    picks = W["picks"]
    real = acc["realized"]
    Rr = acc["R"][real]
    names = D["meta"]["name"].to_dict()
    n_real = int(real.sum())
    bars = {"b1_n_realized": {"n": n_real, "min": MIN_TRADES, "void": bool(n_real < MIN_TRADES), "pass": bool(n_real >= MIN_TRADES)}}
    eday = np.asarray(udates)[acc["eday"][real]]
    mean, lo, hi = T.cluster_boot(Rr, eday, n_boot=10000, seed=SEED)
    bars["b2_cluster_boot"] = {"mean": mean, "lo": lo, "hi": hi, "n_clusters": int(len(np.unique(eday))), "pass": bool(np.isfinite(lo) and lo > 0)}
    rs = float(Rr.sum())
    bars["b3_null_random_config"] = {"R_sum": rs, "p95": _p95(null1), "median": float(np.median(null1)), "share_ge": float(np.mean(np.asarray(null1) >= rs)),
                                     "null_n_realized_median": float(np.median(null1_n)), "pass": bool(rs > _p95(null1))}
    bars["b4_null_random_entry"] = {"R_sum": rs, "p95": _p95(null2), "median": float(np.median(null2)), "share_ge": float(np.mean(np.asarray(null2) >= rs)),
                                    "null_n_realized_median": float(np.median(null2_n)), "unpaired_max": int(max(null2_miss)), "pass": bool(rs > _p95(null2))}
    by_code = pd.Series(Rr).groupby(acc["ci"][real]).sum().sort_values(ascending=False, kind="stable")
    top1, top5 = by_code.index[:1].tolist(), by_code.index[:5].tolist()
    ex1 = float(Rr[~np.isin(acc["ci"][real], top1)].sum())
    ex5 = float(Rr[~np.isin(acc["ci"][real], top5)].sum())
    bars["b5_leave_out"] = {"top1": [f"{codes[i]} {names.get(codes[i], '')} ({by_code[i]:+.1f}R)" for i in top1], "R_sum_ex_top1": ex1,
                            "top5": [f"{codes[i]} {names.get(codes[i], '')} ({by_code[i]:+.1f}R)" for i in top5], "R_sum_ex_top5": ex5,
                            "pass": bool(ex1 > 0 and ex5 > 0)}
    prev_day = pd.Timestamp(udates[d0 - 1])
    q = ix["kosdaq"]
    q0, q1 = float(q.loc[:prev_day].iloc[-1]), float(q.loc[: pd.Timestamp(udates[-1])].iloc[-1])
    acc_ret = float(eq["equity"].iloc[-1] / T.CASH0 - 1)
    avg_expo = float(eq["expo"].mean())
    km = avg_expo * (q1 / q0 - 1)
    bars["b6_vs_expo_kosdaq"] = {"account_return": acc_ret, "avg_exposure": avg_expo, "kosdaq_return": q1 / q0 - 1, "expo_matched_kosdaq": km,
                                 "pass": bool(acc_ret > km)}
    fold_R = []
    for mo in MONTHS:
        mm = real & (acc["month"] == mo)
        fold_R.append(float(acc["R"][mm].sum()))
    npos = int(sum(x > 0 for x in fold_R))
    bars["b7_pos_months"] = {"fold_R_sum": fold_R, "n_pos": npos, "need": 5, "pass": bool(npos >= 5)}
    bars["all_pass"] = bool(all(v["pass"] for k, v in bars.items() if k.startswith("b")))
    # 폴드 표
    folds = []
    for f, mo in enumerate(MONTHS):
        p = picks[f]
        mm = acc["month"] == mo
        folds.append({"month": mo, "pick": p, "cfg": M[p], "cfg_str": cfg_str(M[p]), "train_R": float(W["scores"][p, f]),
                      "train_nbr_median": W["nbr_median"][f], "train_n": int(W["n_train"][p, f]), "test_entries": int(mm.sum()),
                      "test_realized": int((mm & real).sum()), "test_R": fold_R[f], "test_unrealized": int((mm & ~real).sum())})
    freq = {g: {str(v): int(sum(M[p][g] == v for p in picks)) for v in vals} for g, vals in T.AXES.items()}
    # 보고 열
    rep = {"all": T.rstats(Rr)}
    rep["by_reason"] = {T.WHY[int(w)]: T.rstats(acc["R"][real & (acc["why"] == w)]) for w in np.unique(acc["why"][real])}
    rep["by_tf"] = {tf: T.rstats(acc["R"][real & (acc["tfi"] == i)]) for i, tf in enumerate(TFS) if (acc["tfi"] == i).any()}
    rep["overnight_event"] = {"last_bar_event": T.rstats(acc["R"][real & acc["last"]]), "intraday_event": T.rstats(acc["R"][real & ~acc["last"]])}
    r1 = acc1["realized"]
    rep["cost_1x"] = {"all": T.rstats(acc1["R"][r1]), "account_return": float(eq1["equity"].iloc[-1] / T.CASH0 - 1),
                      "n_entries": int(len(acc1["ci"]))}
    un = ~real
    rep["unrealized"] = {"n": int(un.sum()), "R_mtm_sum": float(acc["R"][un].sum()), "R_mtm_mean": float(acc["R"][un].mean()) if un.any() else None,
                         "codes": [f"{codes[i]} {names.get(codes[i], '')}" for i in acc["ci"][un]]}
    hold_days = acc["xday"][real] - acc["eday"][real]
    rep["hold_trading_days"] = {"median": float(np.median(hold_days)) if n_real else None, "mean": float(np.mean(hold_days)) if n_real else None}
    rep["n_codes"] = int(len(np.unique(acc["ci"][real])))
    sm = month_eq(eq["equity"], T.CASH0, prev_day)
    nxt, last = prev_day + pd.Timedelta(days=1), pd.Timestamp(udates[-1])
    kx = month_eq(ix["kospi"].loc[nxt:last], float(ix["kospi"].loc[:prev_day].iloc[-1]), prev_day)
    kq = month_eq(ix["kosdaq"].loc[nxt:last], q0, prev_day)
    j = pd.DataFrame({"s": sm, "k": kx, "q": kq}).dropna()
    rep["monthly"] = {"table": {str(i.date()): {"s": r.s, "kospi": r.k, "kosdaq": r.q} for i, r in j.iterrows()},
                      "n_pos_account_months": int((j["s"] > 0).sum()), "n_months": int(len(j)),
                      "ols_kospi": ols(j["s"], [j["k"]]), "ols_kosdaq": ols(j["s"], [j["q"]]), "ols_both": ols(j["s"], [j["k"], j["q"]])}
    rep["kospi_return"] = float(ix["kospi"].loc[: pd.Timestamp(udates[-1])].iloc[-1] / ix["kospi"].loc[:prev_day].iloc[-1] - 1)
    rep["max_drawdown"] = float((eq["equity"] / eq["equity"].cummax() - 1).min())
    rep["oos_trades"] = pd.DataFrame({
        "code": [codes[i] for i in acc["ci"]], "name": [names.get(codes[i], "") for i in acc["ci"]], "tf": [TFS[i] for i in acc["tfi"]],
        "cfg": acc["cfg"], "month": acc["month"], "signal": [str(Ss[f]["days"][t]) for f, t in zip(acc["tfi"], acc["t"])],
        "entry": [str(Ss[f]["days"][b]) for f, b in zip(acc["tfi"], acc["eb"])],
        "exit": [str(Ss[f]["days"][b]) if r else "" for f, b, r in zip(acc["tfi"], acc["xb"], acc["realized"])],
        "entry_px": acc["entry_px"], "exit_px": acc["exit_px"], "stop": acc["stop"], "dist": acc["dist"], "R": acc["R"],
        "realized": acc["realized"], "why": [T.WHY[int(w)] if r else "미청산" for w, r in zip(acc["why"], acc["realized"])],
        "last_bar_event": acc["last"], "size": acc["size"]}).to_dict("records")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    info = {tf: {k: v for k, v in S["info"].items() if not isinstance(v, list)} | {"n_adj_codes": len(S["info"]["adj_codes"]),
            "n_bars": len(S["days"])} for tf, S in zip(TFS, Ss)}
    return {"label": TAG, "prereg": PREREG, "git": git, "dirty": dirty, "data": DATA, "months": MONTHS, "n_null": len(null1),
            "krx_session": D["krx_session"], "info": info, "bars": bars, "folds": folds, "gene_freq": freq, "report": rep,
            "null1": null1, "null2": null2, "leak_real": LK, "rulings": RULINGS, "equity": {str(i.date()): [r.equity, r.expo] for i, r in eq.iterrows()},
            "selection_scores": {"scores": W["scores"], "nbr_median_of_pick": W["nbr_median"]}}


# =============================================================================================== 그림

def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in sorted(Path.home().glob(".local/share/fonts/noto-kr/NotoSansKR-*.otf")):
        font_manager.fontManager.addfont(str(fp))
    plt.rcParams["font.family"] = "Noto Sans KR"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def figures(D, Ss, acc, eq, ix, null1_eq, null2_R, res, udates, d0) -> None:
    plt = _style()
    OUT.mkdir(parents=True, exist_ok=True)
    b = res["bars"]
    # 1. 자산 곡선
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = eq.index
    band = np.array(null1_eq)
    ax.fill_between(x, np.percentile(band, 5, axis=0), np.percentile(band, 95, axis=0), color="#999999", alpha=0.25,
                    label=f"널 ① 무작위 설정 {len(band)} 회 5–95%")
    ax.plot(x, np.median(band, axis=0), color="#777777", lw=1, ls="--", label="널 ① 중앙")
    q = ix["kosdaq"].reindex(x).ffill()
    q0 = float(ix["kosdaq"].loc[: pd.Timestamp(udates[d0 - 1])].iloc[-1])
    rq = (q / q.shift(1)).fillna(q.iloc[0] / q0).to_numpy() - 1
    ex = np.r_[0.0, eq["expo"].to_numpy()[:-1]]
    ax.plot(x, np.cumprod(1 + ex * rq), color="#eb6834", lw=1.6, label=f"노출 맞춘 KOSDAQ(전날 노출 × 지수, 평균 노출 {b['b6_vs_expo_kosdaq']['avg_exposure']:.0%})")
    ax.plot(x, q / q0, color="#eb6834", lw=0.8, alpha=0.5, label="KOSDAQ")
    ax.plot(x, eq["equity"] / T.CASH0, color="#2a78d6", lw=2.2, label="전략 OOS 단일 계좌(비용 2배)")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_ylabel("자산(시작 = 1)")
    ax.set_title(f"일봉 상승 중 아래 봉 전환 — 워크포워드 OOS 단일 계좌 2026-02 ~ 2026-11-13 (검증 전)\n{TAG}", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=110)
    plt.close(fig)
    # 2. R 히스토그램
    real = acc["realized"]
    R = acc["R"][real]
    fig, ax = plt.subplots(figsize=(12, 6))
    lo_, hi_ = min(-3.0, np.floor(np.nanmin(R)) if len(R) else -3), max(6.0, np.ceil(np.nanpercentile(R, 99.5)) if len(R) else 6)
    bins = np.linspace(lo_, hi_, 61)
    ax.hist(np.clip(R, lo_, hi_), bins=bins, color="#2a78d6", alpha=0.8, label=f"전략 OOS 실현 R (n={len(R)}, 평균 {np.mean(R):+.3f})")
    pool = np.concatenate(null2_R) if null2_R else np.zeros(0)
    if len(pool):
        h, e = np.histogram(np.clip(pool, lo_, hi_), bins=bins)
        ax.step(e[:-1], h / max(len(null2_R), 1), where="post", color="#eb6834", lw=1.6,
                label=f"널 ② 무작위 진입 R(회당 평균 도수, 평균 {np.nanmean(pool):+.3f})")
    ax.axvline(0, color="k", lw=0.8)
    if len(R):
        ax.axvline(np.mean(R), color="#2a78d6", ls="--", lw=1)
    ax.set_xlabel("실현 R(비용 2배, 양 끝은 구간에 모음)")
    ax.set_ylabel("거래 수")
    ax.set_title(f"OOS 실현 R 분포 (검증 전) — {TAG}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "r_hist.png", dpi=110)
    plt.close(fig)
    # 3. 거래 예시 12건 — 실현 거래를 R 순으로 늘어놓고 등간격 12건(최악·최고 포함)
    names = D["meta"]["name"].to_dict()
    M = T.menu()
    idx = np.flatnonzero(real)
    if len(idx) == 0:
        return
    srt = idx[np.argsort(acc["R"][idx], kind="stable")]
    pick = srt[np.unique(np.linspace(0, len(srt) - 1, min(12, len(srt))).round().astype(int))]
    fig, axs = plt.subplots(3, 4, figsize=(30, 17))
    for a, k in zip(axs.ravel(), pick):
        f = int(acc["tfi"][k])
        S = Ss[f]
        i, t, eb, xb = int(acc["ci"][k]), int(acc["t"][k]), int(acc["eb"][k]), int(acc["xb"][k])
        n = len(S["days"])
        xs = np.arange(max(t - 60, 0), min(xb + 30, n - 1) + 1)
        a.plot(xs, S["Cf"][i, xs], color="k", lw=1.0, label=f"{S['tf']} 종가")
        for w, c in ((5, "#ff7f0e"), (20, "#2ca02c"), (60, "#1f77b4")):
            a.plot(xs, S["ma"][w][i, xs], color=c, lw=0.9, label=f"{S['tf']} MA{w}")
        a.step(xs, S["DS"]["ma20_same"][i, S["dix"][xs]], where="post", color="#9467bd", lw=1.5, label="일봉 MA20(그날 종가 기준)")
        a.axvline(t, color="gray", ls=":", lw=1)
        a.scatter([eb], [acc["entry_px"][k]], color="r", zorder=5, s=60, label="진입(다음 봉 시가)")
        a.hlines(acc["stop"][k], eb, xb, color="r", ls="--", lw=1.1, label="초기 손절")
        a.scatter([xb], [acc["exit_px"][k]], marker="v", color="b", zorder=5, s=60, label="청산")
        dd = S["days"][xs]
        tk = [jj for jj in range(len(xs)) if jj == 0 or dd[jj].date() != dd[jj - 1].date()]
        step = max(1, len(tk) // 8)
        a.set_xticks(xs[tk][::step])
        a.set_xticklabels([dd[jj].strftime("%m-%d") for jj in tk][::step], fontsize=8)
        c = M[int(acc["cfg"][k])]
        a.set_title(f"{S['codes'][i]} {names.get(S['codes'][i], '')} · {S['days'][t]:%Y-%m-%d %H:%M} 전환 · {cfg_str(c)}\n"
                    f"R {acc['R'][k]:+.2f} · {T.WHY[int(acc['why'][k])]} · 보유 {xb - eb + 1}봉 · 손절 거리 {acc['dist'][k]:.1%}", fontsize=10)
        a.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"OOS 실현 거래 12건(R 순위 등간격, 최악→최고) — 선택된 봉 종가 + MA5·20·60 + 일봉 MA20 계단 + 진입·청산·손절 (검증 전) · {TAG}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "trade_grid.png", dpi=72)
    plt.close(fig)
    res["grid_trades"] = [int(k) for k in pick]


# =============================================================================================== 쓰기

def _f(v, fmt="+.3f"):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, fmt)


def _ox(b):
    return "O" if b else "X"


def write(res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = res["report"].pop("oos_trades")
    pd.DataFrame(trades).to_csv(OUT / "oos_trades.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1, allow_nan=False))
    b, rep = res["bars"], res["report"]
    L = ["# 일봉 상승 중 아래 봉 전환 — 워크포워드 결과 (검증 전)", "",
         f"git `{res['git']}`{' (작업 트리 수정 있음)' if res['dirty'] else ''} · 사전등록 `{res['prereg']}` · 자료 {res['data'][0]} ~ {res['data'][1]} · "
         f"OOS 폴드 {res['months'][0]} ~ {res['months'][-1]}(9) · 비용 2배 판정 · 널 {res['n_null']} 회", "",
         f"**{TAG}.**", "",
         f"**바 전부:** {'통과' if b['all_pass'] else '미달'}{' (바 1 미달 — VOID, 판단 불가)' if b['b1_n_realized']['void'] else ''} → " + ("PREREG 대로 마지막 폴드 설정으로 전진 모의매매 새 장부 사전등록." if b["all_pass"]
                                                                   else "PREREG 대로 이 셋업(일봉 상승 중 아래 봉 전환) 종결. 같은 분봉 자료로 변형 재실행 금지."), "",
         "## 합격 바(OOS 단일 계좌, 비용 2배)", "", "| 바 | 값 | 기준 | 통과 |", "|---|---|---|---|"]
    L.append(f"| 1 실현 거래 | {b['b1_n_realized']['n']} | ≥ 100 (미달 VOID) | {_ox(b['b1_n_realized']['pass'])} |")
    x = b["b2_cluster_boot"]
    L.append(f"| 2 R 평균 [거래일 군집 95%] | {_f(x['mean'])} [{_f(x['lo'])}, {_f(x['hi'])}] (군집 {x['n_clusters']}) | 하한 > 0 | {_ox(x['pass'])} |")
    x = b["b3_null_random_config"]
    L.append(f"| 3 R 합 vs 무작위 설정 | {_f(x['R_sum'], '+.1f')} | > 널 ① p95 {_f(x['p95'], '+.1f')} (중앙 {_f(x['median'], '+.1f')}, 널 ≥ 전략 {x['share_ge']:.1%}) | {_ox(x['pass'])} |")
    x = b["b4_null_random_entry"]
    L.append(f"| 4 R 합 vs 무작위 진입 | {_f(x['R_sum'], '+.1f')} | > 널 ② p95 {_f(x['p95'], '+.1f')} (중앙 {_f(x['median'], '+.1f')}, 널 ≥ 전략 {x['share_ge']:.1%}) | {_ox(x['pass'])} |")
    x = b["b5_leave_out"]
    L.append(f"| 5 종목 빼기 R 합 | 최대 1 종목 뺀 {_f(x['R_sum_ex_top1'], '+.1f')} / 상위 5 뺀 {_f(x['R_sum_ex_top5'], '+.1f')} | 둘 다 > 0 | {_ox(x['pass'])} |")
    x = b["b6_vs_expo_kosdaq"]
    L.append(f"| 6 계좌 수익 vs 노출 맞춘 KOSDAQ | {x['account_return']:+.2%} vs {x['expo_matched_kosdaq']:+.2%} (평균 노출 {x['avg_exposure']:.1%} × KOSDAQ {x['kosdaq_return']:+.2%}) | > | {_ox(x['pass'])} |")
    x = b["b7_pos_months"]
    L.append(f"| 7 양수 OOS 달 | {x['n_pos']}/9 ({', '.join(_f(v, '+.1f') for v in x['fold_R_sum'])}) | ≥ 5/9 | {_ox(x['pass'])} |")
    L += ["", f"바 5 제거 종목: 최대 1 = {', '.join(b['b5_leave_out']['top1'])}; 상위 5 = {', '.join(b['b5_leave_out']['top5'])}", "",
          "## 폴드별 선택(학습 = 2025-08 ~ m−1 실현 R 합 비용 2배의 이웃 중앙값 최대)", "",
          "| 폴드 | 설정 # | tf | up | 간격 | 밤 | exit | stop | 학습 R 합 | 이웃 중앙값 | 학습 실현 수 | 시험 진입 | 시험 실현 | 시험 R 합 | 미청산 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in res["folds"]:
        c = f["cfg"]
        L.append(f"| {f['month']} | {f['pick']} | {c['tf']} | {c['up']} | {'있음' if c['spread'] else '없음'} | {'허용' if c['overnight'] else '제외'} | {c['exit']} | {c['stop']} | "
                 f"{f['train_R']:+.1f} | {f['train_nbr_median']:+.1f} | {f['train_n']} | {f['test_entries']} | {f['test_realized']} | {f['test_R']:+.1f} | {f['test_unrealized']} |")
    L += ["", "유전자별 선택 빈도(9 폴드): " + "; ".join(f"{g} " + ", ".join(f"{k} {v}" for k, v in d.items()) for g, d in res["gene_freq"].items()), "",
          "## 보고 열(판정 무관)", ""]
    a = rep["all"]
    if a.get("n"):
        L.append(f"- 전체 실현 {a['n']}건 · 종목 {rep['n_codes']} · R 합 {a['R_sum']:+.1f} · 평균 {a['R_mean']:+.3f} · 중앙 {a['R_median']:+.3f} · 승률 {a['win_rate']:.1%} · "
                 f"손익비 {_f(a['payoff'], '.2f')} · 본전 승률 {_f(a['breakeven_win_rate'], '.1%')} · "
                 f"평균 이익 {_f(a['avg_win_R'], '+.2f')}R / 평균 손실 {_f(a['avg_loss_R'], '+.2f')}R · 보유 거래일 중앙 {_f(rep['hold_trading_days']['median'], '.0f')}")
    c1 = rep["cost_1x"]
    if c1["all"].get("n"):
        L.append(f"- 비용 1배(같은 선택): 실현 {c1['all']['n']}건 · R 합 {c1['all']['R_sum']:+.1f} · 평균 {c1['all']['R_mean']:+.3f} · 계좌 수익 {c1['account_return']:+.2%}")
    L.append(f"- 계좌: 수익 {b['b6_vs_expo_kosdaq']['account_return']:+.2%}, 최대 낙폭 {rep['max_drawdown']:.1%}, KOSPI 같은 기간 {rep['kospi_return']:+.2%}, KOSDAQ {b['b6_vs_expo_kosdaq']['kosdaq_return']:+.2%}; "
             f"계좌 월 수익 양수 {rep['monthly']['n_pos_account_months']}/{rep['monthly']['n_months']}")
    u = rep["unrealized"]
    L.append(f"- 미청산(2026-11-13, 실현 바 제외): {u['n']}건, 평가 R 합 {_f(u['R_mtm_sum'], '+.1f')}, 평균 {_f(u['R_mtm_mean'], '+.3f')} — {', '.join(u['codes'])}")
    L += ["", "| 구분 | n | R 합 | R 평균 | 승률 | 손익비 | 본전 승률 |", "|---|---|---|---|---|---|---|"]

    def row(name, s):
        if not s.get("n"):
            return f"| {name} | 0 | — | — | — | — | — |"
        be = s["breakeven_win_rate"]
        return (f"| {name} | {s['n']} | {s['R_sum']:+.1f} | {s['R_mean']:+.3f} | {s['win_rate']:.1%} | {_f(s['payoff'], '.2f')} | "
                f"{_f(be, '.1%')} |")
    for k, s in rep["by_reason"].items():
        L.append(row(f"청산: {k}", s))
    for k, s in rep["by_tf"].items():
        L.append(row(f"봉: {k}", s))
    L.append(row("밤 넘김 사건(하루 마지막 봉)", rep["overnight_event"]["last_bar_event"]))
    L.append(row("장중 사건", rep["overnight_event"]["intraday_event"]))
    mo = rep["monthly"]
    L += ["", "월 수익(계좌) ~ 지수: " + "; ".join(
        f"{nm} β {', '.join(_f(v, '+.2f') for v in o.get('betas', []))} (t {', '.join(_f(v, '+.1f') for v in o.get('t_betas', []))}) · 절편 {_f(o.get('alpha'), '+.4f')} (t {_f(o.get('t_alpha'), '+.2f')}) n={o['n']}"
        for nm, o in (("KOSPI", mo["ols_kospi"]), ("KOSDAQ", mo["ols_kosdaq"]), ("KOSPI+KOSDAQ", mo["ols_both"]))), "",
        "| 월 | 계좌 | KOSPI | KOSDAQ |", "|---|---|---|---|"]
    for d, r in mo["table"].items():
        L.append(f"| {d[:7]} | {r['s']:+.2%} | {r['kospi']:+.2%} | {r['kosdaq']:+.2%} |")
    lk = res["leak_real"]
    L += ["", "## 누출 점검", "", "- 테스트 `tests/test_mtf_transition.py`(합성 자료): 날짜 절단 불변, 시험 달 교란 시 선택 불변(+ 다음 폴드는 바뀌는 음성 대조), "
          "일봉 청산은 그날 종가까지(+ 다음 날 행을 당겨 쓰면 바뀌는 음성 대조), X2 체결 = 다음 날 첫 봉 시가.", "- 실자료 절단(날짜 5개, 정시 절단):"]
    for r in lk["truncation"]:
        L.append(f"  - {r['day']} {r['hi'][11:16]} 전: 선택 같음 {r['picks_same']} · 학습 점수 비트 같음 {r['scores_same']}(최대 차 {r['scores_max_abs_diff']:.1e}) (폴드 {', '.join(r['folds_checked']) or '없음'}) · "
                 + "; ".join(f"{k} 사건 {v['events']} 중 다름 {v['events_diff']}, 특징 다름 {v['events_feat_diff']}, 유니버스 다른 날 {v['univ_days_diff']}, up 다른 칸 {v['up_cells_diff']}"
                             for k, v in r["tf"].items()))
    p = lk["perturb"]
    L.append(f"- 실자료 교란({p['from']} 부터 분봉·일봉·거래대금): 폴드 {p['folds_checked'][0]}~{p['folds_checked'][-1]} 학습 점수 비트 같음 {p['scores_same']}(최대 차 {p['scores_max_abs_diff']:.1e}) · 선택 같음 {p['picks_same']} · "
             f"뒤 폴드 점수 바뀜 {p['later_scores_changed']}(음성 대조)")
    L += ["", "## 그림", "", "- `equity.png` — OOS 단일 계좌 자산(비용 2배)·노출 맞춘 KOSDAQ·널 ① 띠", "- `r_hist.png` — 실현 R 분포와 널 ② 무작위 진입 R",
          "- `trade_grid.png` — OOS 실현 거래 12건(R 순위 등간격)", "", "## 격자 관찰", "", "(그림을 열어 본 뒤 적는다)", "",
          "## Ruling(결과 보기 전 실행기에 적은 해석)", ""]
    for k, v in res["rulings"].items():
        L.append(f"- Ruling `{k}`: {v}")
    L += ["", "## 재현", "", "```", "set -a && . ~/git/quant-airflow/.env && set +a && NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
          "  uv run --with numba --with matplotlib --extra pg --extra dev python research/experiments/mtf_transition_wf.py", "```",
          f"캐시 `{SCRATCH}`(없으면 DB 에서 만든다). 거래 목록 `oos_trades.csv`."]
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    line = {"trial": "mtf_transition", "date": time.strftime("%Y-%m-%d"), "git": res["git"], "dirty": res["dirty"], "prereg": res["prereg"],
            "config": {"data": res["data"], "months": res["months"], "menu_axes": {k: list(v) for k, v in T.AXES.items()}, "n_menu": len(T.menu()),
                       "topn": T.TOPN, "slots": T.SLOTS, "w_cap": T.W_CAP, "risk": T.RISK, "cost_judge": COST_JUDGE, "min_trades": MIN_TRADES,
                       "n_null": res["n_null"], "seed": SEED},
            "selected": [f["pick"] for f in res["folds"]], "n_trials": len(T.menu()),
            "verdict": {k: v["pass"] for k, v in res["bars"].items() if k.startswith("b")} | {"all_pass": res["bars"]["all_pass"]}}
    with open(OUT / "TRIALS.jsonl", "a") as fh:
        fh.write(json.dumps(_clean(line), ensure_ascii=False, allow_nan=False) + "\n")
    log(f"쓰기: {OUT}")


# =============================================================================================== 재등록 MT-2 (PREREG 27334ef)

PREREG_V2 = "research/logs/mtf_transition/PREREG.md 재등록 MT-2 (27334ef)"
TAG_V2 = ("오염 구간(2025-08~2026-11), 1차 결과·리뷰를 본 뒤 재등록 — 판정이 아니라 진단, 통과해도 전진 모의매매 팔 사전등록까지")
DROP_TOP = 3
IDX_START = "2025-01-02"

RULINGS_V2 = {
    "scope": "MT-2 가 바꾸는 다섯 가지 밖은 1차 RULINGS 그대로(사건·유니버스·체결·계좌·비용·워크포워드 창·이웃 정의).",
    "r_acct": "R_acct = (청산가/진입가 − 1 − 비용) × min(W_CAP/RISK, 1/거리) = 20 과 1/거리 중 작은 것 — 비중 min(10%, 0.5%/거리) 로 산 거래의 "
              "계좌 손익 ÷ (0.5% × 진입 시 자산). 거리는 하한을 적용한 손절 거리. 학습 점수·바 2·3·4·5·7·널 ①② 전부 이 단위.",
    "floor": "손절 거리 = max(구조 거리, 왕복 비용 × 2 × 3, 2%), 손절가 = 진입가 × (1 − 거리). 구조 거리 = (진입가 − S1)/진입가(S1 = 1차 Ruling s1). "
             "S1 이 없거나 S1 ≥ 진입가(구조가 진입 전에 깨짐)면 1차처럼 거래하지 않는다(하한으로 살리지 않는다). stop 유전자는 이것 하나(SF).",
    "floor_cost": "하한의 왕복 비용 = korea.cost_rt(진입 체결일 세율, 종목 시장) × 2 — 진입 때 청산일을 모르므로 진입일(cost_rt 설명의 근사). "
                  "비용 1배 보고도 같은 손절(×2 하한)을 써서 거래가 같다(2026 = 2.49%, 2025 = 2.19%).",
    "train_score": "학습 점수 = 학습 창에 청산 체결된 거래의 R_acct 합 − 큰 순 3건(그중 양수인 것만 뺀다; 3건 미만이면 있는 만큼). "
                   "'상위 3건'을 큰 이익 3건으로 읽는다 — 상위 3건이 손실이면 빼서 점수가 오르게 되므로 빼지 않는다. 이웃 중앙값·동률 규칙은 1차 그대로.",
    "regime": "국면 ON(거래일 d) = [d 보다 엄격히 앞선 마지막 KOSDAQ(정문 read_index '101' level) 행의 종가 > 그 행까지 50행 이동평균(min 50)] ∧ "
              "[그날 유니버스(d−1 까지 거래대금 상위 300, 패널 종목) 중 일봉 U1 ON(전날 행 ∧ DEF) 비율 ≥ 25%]. 둘 다 d 전 자료만.",
    "regime_gene": "regime 유전자: ignore(무시) / on(국면 OFF 날 신규 진입 금지). 날 = 진입 체결 봉의 거래일(하루 마지막 봉 사건은 다음 날 국면). "
                   "보유 중인 거래는 국면이 꺼져도 규칙대로 청산(국면은 진입만 막는다).",
    "menu": "메뉴 축 순서 tf × up × spread × overnight × exit × stop(SF) × regime = 288. 이웃 = 유전자 하나만 다른 설정, 자기 포함 11 개(1차와 같은 수).",
    "regime_bars": "국면 ON 부분 = OOS 계좌(전체 OOS 장부, 다시 돌리지 않음)의 진입 중 진입일 국면 ON 인 것. 바 1~7 을 그 부분만으로: "
                   "바 3 널 ① = 같은 무작위 설정 계좌의 국면 ON 진입분 R_acct 합, 바 4 널 ② = 국면 ON 짝 거래의 무작위 진입 R_acct 합, "
                   "바 6 = 그 부분 거래만의 장부(전체 계좌에서 정한 금액 그대로)의 손익 ÷ 1억 vs 그 부분의 평균 노출 × 지수, "
                   "바 7 = 폴드별 국면 ON 진입분 R_acct 합 > 0 인 달 수(국면 ON 진입이 없는 달은 양수가 아님). 국면 ON 실현 < 100 이면 '국면 조건부 판단 불가(VOID)'.",
    "bar6": "바 6 은 노출 맞춘 KOSPI·KOSDAQ 둘 다 넘어야 통과(MT-2 '둘 다'). 노출·지수 수익 정의는 1차 Ruling bar6 그대로.",
    "null2": "널 ② 는 1차 Ruling bar4 그대로(짝의 하한 적용 거리·짝 설정 exit), 점수만 R_acct. 짝 진입이 같은 신호 봉이라 국면도 짝과 같다.",
    "regime_report": "폴드 표의 국면 ON 비율 = 그 달 거래일 중 국면 ON 날의 몫(지수 조건·U1 비율 조건 각각도 보고). 국면 OFF 날 진입분은 보고만.",
}


def load_index_v2() -> pd.DataFrame:
    """KOSPI·KOSDAQ 정문(MA50 예열을 위해 2025-01 부터). 1차 load_index 와 같은 원천."""
    from swing_it.storage import read_index

    con = _connect()
    ix = read_index(con, ["001", "101"], cols=("code", "date", "close"), start=IDX_START, end=DATA[1], with_level=True,
                    order_by_date=True, parse_dates=True)
    w = ix.pivot(index="date", columns="code", values="level").rename(columns={"001": "kospi", "101": "kosdaq"})
    w.index = pd.DatetimeIndex(w.index)
    return w


def setup_v2(Ss: list, kosdaq: pd.Series, costs: np.ndarray, mkt: np.ndarray) -> dict:
    """국면(진입일 기준 ON/OFF)과 하한 손절 ``ev['sf']`` 를 봉 단위마다 붙인다(제자리)."""
    RG = T.regime_on(kosdaq, Ss[0]["udates"], T.u1_share(Ss[0]))
    for S in Ss:
        S["regime"] = RG["on"]
        S["_RG"] = RG
        T.add_floor_stop(S, costs, mkt)
    return RG


def select_v2(Ss: list, kosdaq: pd.Series, costs: np.ndarray, mkt: np.ndarray, cost_mult: float = COST_JUDGE) -> tuple:
    setup_v2(Ss, kosdaq, costs, mkt)
    OCs, dayclose = T.prepare(Ss, stops=("SF",))
    C = T.all_candidates(Ss, OCs, costs, mkt, cost_mult, axes=T.AXES_V2)
    return T.wf_select(C, dayclose, MONTHS, key="Racct", drop_top=DROP_TOP, axes=T.AXES_V2), C, OCs, dayclose


def run_v2(smoke: bool = False) -> dict:
    n_null = 3 if smoke else N_NULL
    D = load()
    Ss = build_all(D)
    udates, codes = Ss[0]["udates"], Ss[0]["codes"]
    mk = D["meta"]["market"].reindex(codes)
    if not mk.isin(["KOSPI", "KOSDAQ"]).all():
        raise ValueError("패널 종목 시장이 KOSPI/KOSDAQ 밖")
    mkt = (mk == "KOSDAQ").astype(np.int64).to_numpy()
    costs = T.cost_table(udates)
    ix = load_index_v2()
    t0 = time.time()
    W, C2, OCs, dayclose = select_v2(Ss, ix["kosdaq"], costs, mkt)
    RG = Ss[0]["_RG"]
    C1 = T.all_candidates(Ss, OCs, costs, mkt, 1.0, axes=T.AXES_V2)
    log(f"MT-2 경로·후보·선택 ({time.time() - t0:.0f}s): 국면 ON 날 {int(RG['on'].sum())}/{len(udates)}, "
        f"후보 수 중앙 {int(np.median([len(c['ci']) for c in C2]))} (최소 {min(len(c['ci']) for c in C2)}, 최대 {max(len(c['ci']) for c in C2)}), "
        + ", ".join(f"{S['tf']} SF 유효 사건 {int(np.isfinite(S['ev']['sf']).sum())}/{len(S['ev']['t'])} 구조 거리가 하한보다 큼 {int(S['ev']['sf_struct_binding'].sum())}"
                    for S in Ss))
    picks = W["picks"]
    acc = T.account(T.oos_stream(C2, picks, MONTHS), dayclose)
    acc1 = T.account(T.oos_stream(C1, picks, MONTHS), dayclose)
    real = acc["realized"]
    log(f"OOS 계좌: 진입 {len(acc['ci'])}, 실현 {int(real.sum())}, 미청산 {int((~real).sum())}, 국면 ON 진입 {int(acc['reg_on'].sum())} "
        f"(실현 {int((real & acc['reg_on']).sum())})")
    d0 = int(np.searchsorted(udates, np.datetime64("2026-02-01")))
    d1 = len(udates) - 1
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    null1 = {"all": [], "on": [], "n_all": [], "n_on": [], "eq": []}
    for _ in range(n_null):
        p = rng.integers(0, len(C2), len(MONTHS))
        a = T.account(T.oos_stream(C2, p, MONTHS), dayclose)
        r_ = a["realized"]
        null1["all"].append(float(a["Racct"][r_].sum()))
        null1["on"].append(float(a["Racct"][r_ & a["reg_on"]].sum()))
        null1["n_all"].append(int(r_.sum()))
        null1["n_on"].append(int((r_ & a["reg_on"]).sum()))
        null1["eq"].append(T.equity_curve(a, dayclose, udates, d0, d1, costs, mkt, COST_JUDGE)["equity"].to_numpy() / T.CASH0)
    log(f"널 ① {n_null} 회 ({time.time() - t0:.0f}s)")
    t0 = time.time()
    pools = [T.entry_pool(S) for S in Ss]
    cache: dict = {}
    rng2 = np.random.default_rng(SEED + 1)
    null2 = {"all": [], "on": [], "n_all": [], "n_on": [], "miss": [], "R_all": [], "R_on": []}
    for _ in range(n_null):
        x = T.random_entries(Ss, pools, acc, rng2, costs, mkt, COST_JUDGE, cache, axes=T.AXES_V2)
        ok, okon = x["realized"], x["realized"] & acc["reg_on"]
        null2["all"].append(float(np.sum(x["Racct"][ok])))
        null2["on"].append(float(np.sum(x["Racct"][okon])))
        null2["n_all"].append(int(ok.sum()))
        null2["n_on"].append(int(okon.sum()))
        null2["miss"].append(int((x["code"] < 0).sum()))
        null2["R_all"].append(x["Racct"][ok])
        null2["R_on"].append(x["Racct"][okon])
    log(f"널 ② {n_null} 회 ({time.time() - t0:.0f}s), 짝 못 찾음 최대 {max(null2['miss']) if null2['miss'] else 0}")
    if smoke:
        log("스모크: 결과 계산·파일 쓰기 생략")
        return {}
    eq = T.equity_curve(acc, dayclose, udates, d0, d1, costs, mkt, COST_JUDGE)
    eq1 = T.equity_curve(acc1, dayclose, udates, d0, d1, costs, mkt, 1.0)
    acc_on = T.take(acc, acc["reg_on"])
    eq_on = T.equity_curve(acc_on, dayclose, udates, d0, d1, costs, mkt, COST_JUDGE)
    log("누출 점검(실자료, MT-2 선택) …")
    LK = leak_real(D, Ss, W, codes, costs, mkt, v2_kosdaq=ix["kosdaq"], RG=RG)
    res = assemble_v2(D, Ss, W, acc, acc1, eq, eq1, eq_on, ix, RG, null1, null2, udates, d0, codes, LK)
    figures_v2(D, Ss, acc, eq, eq_on, ix, RG, null1, null2, res, udates, d0)
    write_v2(res)
    return res


def bars_v2(part: dict, eq_part: pd.DataFrame, n1: list, n1_n: list, n2: list, n2_n: list, n2_miss: list, ix: pd.DataFrame, udates, d0: int,
            codes: list, names: dict) -> dict:
    """바 1~7(MT-2 단위 R_acct). ``part`` = OOS 계좌 장부(전체 또는 국면 ON 부분)."""
    real = part["realized"]
    Rr = part["Racct"][real]
    n_real = int(real.sum())
    b = {"b1_n_realized": {"n": n_real, "min": MIN_TRADES, "void": bool(n_real < MIN_TRADES), "pass": bool(n_real >= MIN_TRADES)}}
    eday = np.asarray(udates)[part["eday"][real]]
    mean, lo, hi = T.cluster_boot(Rr, eday, n_boot=10000, seed=SEED)
    b["b2_cluster_boot"] = {"mean": mean, "lo": lo, "hi": hi, "n_clusters": int(len(np.unique(eday))), "pass": bool(np.isfinite(lo) and lo > 0)}
    rs = float(Rr.sum())
    b["b3_null_random_config"] = {"R_sum": rs, "p95": _p95(n1), "median": float(np.median(n1)), "share_ge": float(np.mean(np.asarray(n1) >= rs)),
                                  "null_n_realized_median": float(np.median(n1_n)), "pass": bool(rs > _p95(n1))}
    b["b4_null_random_entry"] = {"R_sum": rs, "p95": _p95(n2), "median": float(np.median(n2)), "share_ge": float(np.mean(np.asarray(n2) >= rs)),
                                 "null_n_realized_median": float(np.median(n2_n)), "unpaired_max": int(max(n2_miss)), "pass": bool(rs > _p95(n2))}
    by_code = pd.Series(Rr).groupby(part["ci"][real]).sum().sort_values(ascending=False, kind="stable")
    top1, top5 = by_code.index[:1].tolist(), by_code.index[:5].tolist()
    ex1 = float(Rr[~np.isin(part["ci"][real], top1)].sum())
    ex5 = float(Rr[~np.isin(part["ci"][real], top5)].sum())
    b["b5_leave_out"] = {"top1": [f"{codes[i]} {names.get(codes[i], '')} ({by_code[i]:+.1f}R)" for i in top1], "R_sum_ex_top1": ex1,
                         "top5": [f"{codes[i]} {names.get(codes[i], '')} ({by_code[i]:+.1f}R)" for i in top5], "R_sum_ex_top5": ex5,
                         "pass": bool(n_real > 0 and ex1 > 0 and ex5 > 0)}
    prev_day, last = pd.Timestamp(udates[d0 - 1]), pd.Timestamp(udates[-1])
    rets = {k: float(ix[k].loc[:last].iloc[-1] / ix[k].loc[:prev_day].iloc[-1] - 1) for k in ("kospi", "kosdaq")}
    acc_ret = float(eq_part["equity"].iloc[-1] / T.CASH0 - 1)
    avg_expo = float(eq_part["expo"].mean())
    b["b6_vs_expo_index"] = {"account_return": acc_ret, "avg_exposure": avg_expo, "kospi_return": rets["kospi"], "kosdaq_return": rets["kosdaq"],
                             "expo_matched_kospi": avg_expo * rets["kospi"], "expo_matched_kosdaq": avg_expo * rets["kosdaq"],
                             "pass_kospi": bool(acc_ret > avg_expo * rets["kospi"]), "pass_kosdaq": bool(acc_ret > avg_expo * rets["kosdaq"]),
                             "pass": bool(acc_ret > avg_expo * rets["kospi"] and acc_ret > avg_expo * rets["kosdaq"])}
    fold_R = [float(part["Racct"][real & (part["month"] == mo)].sum()) for mo in MONTHS]
    fold_n = [int((real & (part["month"] == mo)).sum()) for mo in MONTHS]
    npos = int(sum(x > 0 for x in fold_R))
    b["b7_pos_months"] = {"fold_R_sum": fold_R, "fold_n": fold_n, "n_pos": npos, "need": 5, "pass": bool(npos >= 5)}
    b["all_pass"] = bool(all(v["pass"] for k, v in b.items() if k.startswith("b")))
    return b


def _rep_block(part: dict, Ss: list, key: str = "Racct") -> dict:
    real = part["realized"]
    rep = {"all": T.rstats(part[key][real])}
    rep["by_reason"] = {T.WHY[int(w)]: T.rstats(part[key][real & (part["why"] == w)]) for w in np.unique(part["why"][real])}
    rep["by_tf"] = {tf: T.rstats(part[key][real & (part["tfi"] == i)]) for i, tf in enumerate(TFS) if (part["tfi"] == i).any()}
    rep["overnight_event"] = {"last_bar_event": T.rstats(part[key][real & part["last"]]), "intraday_event": T.rstats(part[key][real & ~part["last"]])}
    rep["by_regime"] = {"국면 ON 진입": T.rstats(part[key][real & part["reg_on"]]), "국면 OFF 진입": T.rstats(part[key][real & ~part["reg_on"]])}
    return rep


def assemble_v2(D, Ss, W, acc, acc1, eq, eq1, eq_on, ix, RG, null1, null2, udates, d0, codes, LK) -> dict:
    M = T.menu(T.AXES_V2)
    picks = W["picks"]
    names = D["meta"]["name"].to_dict()
    real = acc["realized"]
    acc_on = T.take(acc, acc["reg_on"])
    bars = bars_v2(acc, eq, null1["all"], null1["n_all"], null2["all"], null2["n_all"], null2["miss"], ix, udates, d0, codes, names)
    bars_on = bars_v2(acc_on, eq_on, null1["on"], null1["n_on"], null2["on"], null2["n_on"], null2["miss"], ix, udates, d0, codes, names)
    ud = pd.DatetimeIndex(udates)
    reg_m = pd.DataFrame({"on": RG["on"], "idx_on": RG["idx_on"], "u1_ok": RG["share"] >= T.REGIME_U1_MIN, "share": RG["share"]},
                         index=ud).groupby(ud.strftime("%Y-%m")).mean()
    reg_month = {mo: {k: float(v) for k, v in r.items()} | {"days": int((ud.strftime("%Y-%m") == mo).sum())} for mo, r in reg_m.iterrows()}
    # 폴드 표
    folds = []
    for f, mo in enumerate(MONTHS):
        p = picks[f]
        mm = acc["month"] == mo
        folds.append({"month": mo, "pick": p, "cfg": M[p], "cfg_str": cfg_str_v2(M[p]), "train_score": float(W["scores"][p, f]),
                      "train_nbr_median": W["nbr_median"][f], "train_n": int(W["n_train"][p, f]), "test_entries": int(mm.sum()),
                      "test_realized": int((mm & real).sum()), "test_Racct": float(acc["Racct"][mm & real].sum()),
                      "test_unrealized": int((mm & ~real).sum()), "test_entries_on": int((mm & acc["reg_on"]).sum()),
                      "test_Racct_on": float(acc["Racct"][mm & real & acc["reg_on"]].sum()), "regime_on_share": reg_month[mo]["on"],
                      "idx_on_share": reg_month[mo]["idx_on"], "u1_ok_share": reg_month[mo]["u1_ok"], "u1_share_mean": reg_month[mo]["share"]})
    freq = {g: {str(v): int(sum(M[p][g] == v for p in picks)) for v in vals} for g, vals in T.AXES_V2.items()}
    rep = _rep_block(acc, Ss)
    rep["on"] = _rep_block(acc_on, Ss)
    rep["legacy_R_per_stop"] = T.rstats(acc["R"][real])
    r1 = acc1["realized"]
    rep["cost_1x"] = {"all": T.rstats(acc1["Racct"][r1]), "on": T.rstats(acc1["Racct"][r1 & acc1["reg_on"]]),
                      "account_return": float(eq1["equity"].iloc[-1] / T.CASH0 - 1), "n_entries": int(len(acc1["ci"]))}
    un = ~real
    rep["unrealized"] = {"n": int(un.sum()), "Racct_mtm_sum": float(acc["Racct"][un].sum()),
                         "Racct_mtm_mean": float(acc["Racct"][un].mean()) if un.any() else None,
                         "codes": [f"{codes[i]} {names.get(codes[i], '')}" for i in acc["ci"][un]]}
    hold_days = acc["xday"][real] - acc["eday"][real]
    n_real = int(real.sum())
    rep["hold_trading_days"] = {"median": float(np.median(hold_days)) if n_real else None, "mean": float(np.mean(hold_days)) if n_real else None}
    rep["n_codes"] = int(len(np.unique(acc["ci"][real])))
    sdist = np.array([Ss[int(f)]["ev"]["s1"][int(e)] for f, e in zip(acc["tfi"], acc["ev_idx"])])
    sdist = (acc["entry_px"] - sdist) / acc["entry_px"]
    floor = ~np.isclose(acc["dist"], sdist)
    rep["stop_dist"] = {"quantiles": {str(q): float(np.quantile(acc["dist"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(acc["dist"]) else {},
                        "share_floor_binding": float(floor.mean()) if len(floor) else None, "n_floor_binding": int(floor.sum())}
    prev_day = pd.Timestamp(udates[d0 - 1])
    sm = month_eq(eq["equity"], T.CASH0, prev_day)
    nxt, last = prev_day + pd.Timedelta(days=1), pd.Timestamp(udates[-1])
    kx = month_eq(ix["kospi"].loc[nxt:last], float(ix["kospi"].loc[:prev_day].iloc[-1]), prev_day)
    kq = month_eq(ix["kosdaq"].loc[nxt:last], float(ix["kosdaq"].loc[:prev_day].iloc[-1]), prev_day)
    j = pd.DataFrame({"s": sm, "k": kx, "q": kq}).dropna()
    rep["monthly"] = {"table": {str(i.date()): {"s": r.s, "kospi": r.k, "kosdaq": r.q} for i, r in j.iterrows()},
                      "n_pos_account_months": int((j["s"] > 0).sum()), "n_months": int(len(j)),
                      "ols_kospi": ols(j["s"], [j["k"]]), "ols_kosdaq": ols(j["s"], [j["q"]]), "ols_both": ols(j["s"], [j["k"], j["q"]])}
    rep["max_drawdown"] = float((eq["equity"] / eq["equity"].cummax() - 1).min())
    sd = acc["entry_px"] * (1.0 - sdist)
    rep["oos_trades"] = pd.DataFrame({
        "code": [codes[i] for i in acc["ci"]], "name": [names.get(codes[i], "") for i in acc["ci"]], "tf": [TFS[i] for i in acc["tfi"]],
        "cfg": acc["cfg"], "cfg_str": [cfg_str_v2(M[int(c)]) for c in acc["cfg"]], "month": acc["month"],
        "signal": [str(Ss[f]["days"][t]) for f, t in zip(acc["tfi"], acc["t"])],
        "entry": [str(Ss[f]["days"][b]) for f, b in zip(acc["tfi"], acc["eb"])],
        "exit": [str(Ss[f]["days"][b]) if r else "" for f, b, r in zip(acc["tfi"], acc["xb"], acc["realized"])],
        "entry_px": acc["entry_px"], "exit_px": acc["exit_px"], "s1_struct": sd, "stop": acc["stop"], "dist": acc["dist"], "ret": acc["ret"],
        "R_acct": acc["Racct"], "R_per_stop": acc["R"], "regime_on": acc["reg_on"], "realized": acc["realized"],
        "why": [T.WHY[int(w)] if r else "미청산" for w, r in zip(acc["why"], acc["realized"])],
        "last_bar_event": acc["last"], "size": acc["size"]}).to_dict("records")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    info = {tf: {k: v for k, v in S["info"].items() if not isinstance(v, list)} | {"n_adj_codes": len(S["info"]["adj_codes"]),
            "n_bars": len(S["days"]), "n_sf_valid": int(np.isfinite(S["ev"]["sf"]).sum()), "n_struct_binding": int(S["ev"]["sf_struct_binding"].sum())}
            for tf, S in zip(TFS, Ss)}
    return {"label": TAG_V2, "prereg": PREREG_V2, "git": git, "dirty": dirty, "data": DATA, "months": MONTHS, "n_null": len(null1["all"]),
            "krx_session": D["krx_session"], "info": info, "bars": bars, "bars_regime_on": bars_on, "folds": folds, "regime_by_month": reg_month,
            "gene_freq": freq, "report": rep, "null1": {k: null1[k] for k in ("all", "on", "n_all", "n_on")},
            "null2": {k: null2[k] for k in ("all", "on", "n_all", "n_on", "miss")}, "leak_real": LK, "rulings": RULINGS, "rulings_v2": RULINGS_V2,
            "equity": {str(i.date()): [r.equity, r.expo] for i, r in eq.iterrows()},
            "equity_regime_on_part": {str(i.date()): [r.equity, r.expo] for i, r in eq_on.iterrows()},
            "regime_daily": {str(pd.Timestamp(d).date()): [bool(o), bool(a), _clean(float(s))] for d, o, a, s in zip(udates, RG["on"], RG["idx_on"], RG["share"])},
            "selection_scores": {"scores": W["scores"], "nbr_median_of_pick": W["nbr_median"]}}


def cfg_str_v2(c: dict) -> str:
    return (f"{c['tf']}·{c['up']}·{'간격' if c['spread'] else '간격없음'}·{'밤허용' if c['overnight'] else '밤제외'}·{c['exit']}·{c['stop']}·"
            f"{'국면ON만' if c['regime'] == 'on' else '국면무시'}")


def figures_v2(D, Ss, acc, eq, eq_on, ix, RG, null1, null2, res, udates, d0) -> None:
    plt = _style()
    OUT.mkdir(parents=True, exist_ok=True)
    b = res["bars"]["b6_vs_expo_index"]
    # 1. 자산 곡선
    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = eq.index
    band = np.array(null1["eq"])
    ax.fill_between(x, np.percentile(band, 5, axis=0), np.percentile(band, 95, axis=0), color="#999999", alpha=0.25,
                    label=f"널 ① 무작위 설정 {len(band)} 회 5–95%")
    ax.plot(x, np.median(band, axis=0), color="#777777", lw=1, ls="--", label="널 ① 중앙")
    ex = np.r_[0.0, eq["expo"].to_numpy()[:-1]]
    for k, col in (("kosdaq", "#eb6834"), ("kospi", "#2ca02c")):
        q = ix[k].reindex(x).ffill()
        q0 = float(ix[k].loc[: pd.Timestamp(udates[d0 - 1])].iloc[-1])
        rq = (q / q.shift(1)).fillna(q.iloc[0] / q0).to_numpy() - 1
        ax.plot(x, np.cumprod(1 + ex * rq), color=col, lw=1.6, label=f"노출 맞춘 {k.upper()}(전날 노출 × 지수, 평균 노출 {b['avg_exposure']:.0%})")
        ax.plot(x, q / q0, color=col, lw=0.8, alpha=0.45, label=k.upper())
    ax.plot(x, eq_on["equity"] / T.CASH0, color="#9467bd", lw=1.4, ls="-.", label="국면 ON 진입분만의 장부(같은 금액)")
    ax.plot(x, eq["equity"] / T.CASH0, color="#2a78d6", lw=2.2, label="전략 OOS 단일 계좌(비용 2배)")
    off = ~RG["on"][d0:]
    ymin, ymax = ax.get_ylim()
    ax.fill_between(x, ymin, ymax, where=off, color="#d62728", alpha=0.06, step="mid", label="국면 OFF 날")
    ax.set_ylim(ymin, ymax)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_ylabel("자산(시작 = 1)")
    ax.set_title(f"재등록 MT-2 — 워크포워드 OOS 단일 계좌 2026-02 ~ 2026-11-13 (진단, 검증 전)\n{TAG_V2}", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_v2.png", dpi=110)
    plt.close(fig)
    # 2. R_acct 히스토그램
    real = acc["realized"]
    R = acc["Racct"][real]
    Ron = acc["Racct"][real & acc["reg_on"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    lo_ = min(-3.0, np.floor(np.nanmin(R)) if len(R) else -3)
    hi_ = max(6.0, np.ceil(np.nanpercentile(R, 99.5)) if len(R) else 6)
    bins = np.linspace(lo_, hi_, 61)
    ax.hist(np.clip(R, lo_, hi_), bins=bins, color="#2a78d6", alpha=0.75, label=f"전략 OOS 실현 R_acct 전체 (n={len(R)}, 평균 {np.mean(R) if len(R) else np.nan:+.3f})")
    ax.hist(np.clip(Ron, lo_, hi_), bins=bins, color="#9467bd", alpha=0.6,
            label=f"그중 국면 ON 진입 (n={len(Ron)}, 평균 {np.mean(Ron) if len(Ron) else np.nan:+.3f})")
    pool = np.concatenate(null2["R_all"]) if null2["R_all"] else np.zeros(0)
    if len(pool):
        h, e = np.histogram(np.clip(pool, lo_, hi_), bins=bins)
        ax.step(e[:-1], h / max(len(null2["R_all"]), 1), where="post", color="#eb6834", lw=1.6,
                label=f"널 ② 무작위 진입 R_acct(회당 평균 도수, 평균 {np.nanmean(pool):+.3f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("실현 R_acct = 수익률 × min(20, 1/거리) (비용 2배, 양 끝은 구간에 모음)")
    ax.set_ylabel("거래 수")
    ax.set_title(f"MT-2 OOS 실현 R_acct 분포 (진단, 검증 전) — {TAG_V2}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "r_hist_v2.png", dpi=110)
    plt.close(fig)
    # 3. 거래 12건 — 실현 거래 R_acct 순 등간격, 봉·거래량·MA·일봉 MA·진입·청산·손절(구조선·하한)
    names = D["meta"]["name"].to_dict()
    M = T.menu(T.AXES_V2)
    idx = np.flatnonzero(real)
    if len(idx) == 0:
        return
    srt = idx[np.argsort(acc["Racct"][idx], kind="stable")]
    pick = srt[np.unique(np.linspace(0, len(srt) - 1, min(12, len(srt))).round().astype(int))]
    fig, axs = plt.subplots(3, 4, figsize=(30, 17))
    for a, k in zip(axs.ravel(), pick):
        f = int(acc["tfi"][k])
        S = Ss[f]
        i, t, eb, xb = int(acc["ci"][k]), int(acc["t"][k]), int(acc["eb"][k]), int(acc["xb"][k])
        n = len(S["days"])
        xs = np.arange(max(t - 60, 0), min(xb + 30, n - 1) + 1)
        o, h, lo, c, v = (S[kk][i, xs] for kk in ("O", "H", "Lo", "C", "V"))
        up = c >= o
        w = 0.8 if len(xs) < 250 else 0.5
        a.vlines(xs, lo, h, color=np.where(up, "#d62728", "#1f77b4"), lw=0.5)
        a.vlines(xs, np.minimum(o, c), np.maximum(o, c), color=np.where(up, "#d62728", "#1f77b4"), lw=max(1.0, 4 * w * 250 / max(len(xs), 1) / 4))
        for ww, col in ((5, "#ff7f0e"), (20, "#2ca02c"), (60, "#8c564b")):
            a.plot(xs, S["ma"][ww][i, xs], color=col, lw=0.9, label=f"{S['tf']} MA{ww}")
        a.step(xs, S["DS"]["ma20_same"][i, S["dix"][xs]], where="post", color="#9467bd", lw=1.4, label="일봉 MA20(그날 종가 기준)")
        a.step(xs, S["DS"]["ma10_same"][i, S["dix"][xs]], where="post", color="#e377c2", lw=1.0, ls="--", label="일봉 MA10")
        a.axvline(t, color="gray", ls=":", lw=1)
        a.scatter([eb], [acc["entry_px"][k]], color="r", zorder=5, s=70, marker="^", label="진입(다음 봉 시가)")
        s1v = Ss[f]["ev"]["s1"][int(acc["ev_idx"][k])]
        a.hlines(acc["stop"][k], eb, xb, color="r", ls="--", lw=1.2, label=f"손절(하한 적용) 거리 {acc['dist'][k]:.1%}")
        if not np.isclose(s1v, acc["stop"][k]):
            a.hlines(s1v, eb, xb, color="orange", ls=":", lw=1.0, label="구조선 S1(하한보다 좁음)")
        a.scatter([xb], [acc["exit_px"][k]], marker="v", color="b", zorder=5, s=70, label="청산")
        lo_y, hi_y = np.nanmin(lo), np.nanmax(h)
        lo_y = min(lo_y, acc["stop"][k])
        a.set_ylim(lo_y - (hi_y - lo_y) * 0.28, hi_y + (hi_y - lo_y) * 0.05)
        a2 = a.twinx()
        a2.bar(xs, np.nan_to_num(v), color="#7f7f7f", alpha=0.35, width=1.0)
        a2.set_ylim(0, np.nanmax(np.nan_to_num(v)) * 4.5 if np.nanmax(np.nan_to_num(v)) > 0 else 1)
        a2.set_yticks([])
        dd = S["days"][xs]
        tk = [jj for jj in range(len(xs)) if jj == 0 or dd[jj].date() != dd[jj - 1].date()]
        step = max(1, len(tk) // 8)
        a.set_xticks(xs[tk][::step])
        a.set_xticklabels([dd[jj].strftime("%m-%d") for jj in tk][::step], fontsize=8)
        cc = M[int(acc["cfg"][k])]
        a.set_title(f"{S['codes'][i]} {names.get(S['codes'][i], '')} · {S['days'][t]:%Y-%m-%d %H:%M} 전환 · {cfg_str_v2(cc)}\n"
                    f"R_acct {acc['Racct'][k]:+.2f} (수익률 {acc['ret'][k]:+.1%}) · {T.WHY[int(acc['why'][k])]} · 보유 {xb - eb + 1}봉 · "
                    f"국면 {'ON' if acc['reg_on'][k] else 'OFF'}", fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"MT-2 OOS 실현 거래 12건(R_acct 순위 등간격, 최악→최고) — 봉·거래량·MA5·20·60·일봉 MA20/MA10·진입·청산·손절 (진단, 검증 전) · {TAG_V2}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "trade_grid_v2.png", dpi=72)
    plt.close(fig)
    res["grid_trades"] = [int(k) for k in pick]


def _bar_rows(b: dict, label: str) -> list:
    L = [f"### {label}", "", "| 바 | 값 | 기준 | 통과 |", "|---|---|---|---|"]
    L.append(f"| 1 실현 거래 | {b['b1_n_realized']['n']} | ≥ 100 (미달 VOID) | {_ox(b['b1_n_realized']['pass'])} |")
    x = b["b2_cluster_boot"]
    L.append(f"| 2 R_acct 평균 [거래일 군집 95%] | {_f(x['mean'])} [{_f(x['lo'])}, {_f(x['hi'])}] (군집 {x['n_clusters']}) | 하한 > 0 | {_ox(x['pass'])} |")
    x = b["b3_null_random_config"]
    L.append(f"| 3 R_acct 합 vs 무작위 설정 | {_f(x['R_sum'], '+.1f')} | > 널 ① p95 {_f(x['p95'], '+.1f')} (중앙 {_f(x['median'], '+.1f')}, 널 ≥ 전략 {x['share_ge']:.1%}) | {_ox(x['pass'])} |")
    x = b["b4_null_random_entry"]
    L.append(f"| 4 R_acct 합 vs 무작위 진입 | {_f(x['R_sum'], '+.1f')} | > 널 ② p95 {_f(x['p95'], '+.1f')} (중앙 {_f(x['median'], '+.1f')}, 널 ≥ 전략 {x['share_ge']:.1%}) | {_ox(x['pass'])} |")
    x = b["b5_leave_out"]
    L.append(f"| 5 종목 빼기 R_acct 합 | 최대 1 종목 뺀 {_f(x['R_sum_ex_top1'], '+.1f')} / 상위 5 뺀 {_f(x['R_sum_ex_top5'], '+.1f')} | 둘 다 > 0 | {_ox(x['pass'])} |")
    x = b["b6_vs_expo_index"]
    L.append(f"| 6 계좌 수익 vs 노출 맞춘 지수 | {x['account_return']:+.2%} vs KOSPI {x['expo_matched_kospi']:+.2%} / KOSDAQ {x['expo_matched_kosdaq']:+.2%} "
             f"(평균 노출 {x['avg_exposure']:.1%} × KOSPI {x['kospi_return']:+.2%} / KOSDAQ {x['kosdaq_return']:+.2%}) | 둘 다 > | "
             f"{_ox(x['pass'])} (KOSPI {_ox(x['pass_kospi'])}, KOSDAQ {_ox(x['pass_kosdaq'])}) |")
    x = b["b7_pos_months"]
    L.append(f"| 7 양수 OOS 달 | {x['n_pos']}/9 ({', '.join(_f(v, '+.1f') for v in x['fold_R_sum'])}; 실현 {', '.join(str(v) for v in x['fold_n'])}) | ≥ 5/9 | {_ox(x['pass'])} |")
    L += ["", f"바 5 제거 종목: 최대 1 = {', '.join(b['b5_leave_out']['top1']) or '—'}; 상위 5 = {', '.join(b['b5_leave_out']['top5']) or '—'}", ""]
    return L


def _row(name, s):
    if not s.get("n"):
        return f"| {name} | 0 | — | — | — | — | — |"
    return (f"| {name} | {s['n']} | {s['R_sum']:+.1f} | {s['R_mean']:+.3f} | {s['win_rate']:.1%} | {_f(s['payoff'], '.2f')} | "
            f"{_f(s['breakeven_win_rate'], '.1%')} |")


def write_v2(res: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = res["report"].pop("oos_trades")
    pd.DataFrame(trades).to_csv(OUT / "oos_trades_v2.csv", index=False)
    (OUT / "results_v2.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1, allow_nan=False))
    b, bo, rep = res["bars"], res["bars_regime_on"], res["report"]
    void_on = bo["b1_n_realized"]["void"]
    L = ["# 재등록 MT-2 — 일봉 상승 중 아래 봉 전환 워크포워드 결과 (진단, 검증 전)", "",
         f"git `{res['git']}`{' (작업 트리 수정 있음)' if res['dirty'] else ''} · 사전등록 `{res['prereg']}` · 자료 {res['data'][0]} ~ {res['data'][1]} · "
         f"OOS 폴드 {res['months'][0]} ~ {res['months'][-1]}(9) · 비용 2배 판정 · 널 {res['n_null']} 회 · 단위 R_acct", "",
         f"**{TAG_V2}.** 1차 판정(종결, RESULTS.md·정정 MT-1)은 그대로 유지된다.", "",
         f"**전체 OOS 바 전부:** {'통과' if b['all_pass'] else '미달'}{' (바 1 미달 — VOID)' if b['b1_n_realized']['void'] else ''} · "
         f"**국면 ON OOS 바 전부:** {'국면 조건부 판단 불가(VOID — 국면 ON 실현 < 100)' if void_on else ('통과' if bo['all_pass'] else '미달')}", "",
         "## 합격 바(OOS 단일 계좌, 비용 2배, R_acct)", ""]
    L += _bar_rows(b, "전체 OOS")
    L += _bar_rows(bo, "국면 ON 날 진입분만" + (" — VOID(판단 불가, 숫자는 보고만)" if void_on else ""))
    L += ["## 폴드별 선택(학습 = 2025-08 ~ m−1, 점수 = 실현 R_acct 합 − 상위 3건, 이웃 중앙값 최대)", "",
          "| 폴드 | 설정 # | tf | up | 간격 | 밤 | exit | regime | 학습 점수 | 이웃 중앙값 | 학습 실현 수 | 시험 진입 | 시험 실현 | 시험 R_acct 합 | 미청산 | "
          "국면 ON 진입 | 국면 ON R_acct 합 | 그 달 국면 ON 날 | 지수 조건 | U1 ≥ 25% | U1 비율 평균 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in res["folds"]:
        c = f["cfg"]
        L.append(f"| {f['month']} | {f['pick']} | {c['tf']} | {c['up']} | {'있음' if c['spread'] else '없음'} | {'허용' if c['overnight'] else '제외'} | {c['exit']} | "
                 f"{c['regime']} | {f['train_score']:+.1f} | {f['train_nbr_median']:+.1f} | {f['train_n']} | {f['test_entries']} | {f['test_realized']} | "
                 f"{f['test_Racct']:+.1f} | {f['test_unrealized']} | {f['test_entries_on']} | {f['test_Racct_on']:+.1f} | {f['regime_on_share']:.0%} | "
                 f"{f['idx_on_share']:.0%} | {f['u1_ok_share']:.0%} | {f['u1_share_mean']:.0%} |")
    L += ["", "유전자별 선택 빈도(9 폴드): " + "; ".join(f"{g} " + ", ".join(f"{k} {v}" for k, v in d.items()) for g, d in res["gene_freq"].items()), "",
          "달별 국면 ON 비율(전 기간, 거래일 몫 · 지수 조건 · U1 ≥ 25% · U1 비율 평균):", "", "| 달 | 거래일 | 국면 ON | 지수 > MA50 | U1 ≥ 25% | U1 비율 평균 |", "|---|---|---|---|---|---|"]
    for mo, r in res["regime_by_month"].items():
        L.append(f"| {mo} | {r['days']} | {r['on']:.0%} | {r['idx_on']:.0%} | {r['u1_ok']:.0%} | {r['share']:.0%} |")
    L += ["", "## 보고 열(판정 무관, R_acct)", ""]
    for nm, rp in (("전체", rep), ("국면 ON 진입분", rep["on"])):
        a = rp["all"]
        if a.get("n"):
            L.append(f"- {nm}: 실현 {a['n']}건 · R_acct 합 {a['R_sum']:+.1f} · 평균 {a['R_mean']:+.3f} · 중앙 {a['R_median']:+.3f} · 승률 {a['win_rate']:.1%} · "
                     f"손익비 {_f(a['payoff'], '.2f')} · 본전 승률 {_f(a['breakeven_win_rate'], '.1%')} · 평균 이익 {_f(a['avg_win_R'], '+.2f')} / 평균 손실 {_f(a['avg_loss_R'], '+.2f')}")
        else:
            L.append(f"- {nm}: 실현 0건")
    L.append(f"- 종목 {rep['n_codes']} · 보유 거래일 중앙 {_f(rep['hold_trading_days']['median'], '.0f')}")
    lg = rep["legacy_R_per_stop"]
    if lg.get("n"):
        L.append(f"- 참고(1차 단위, 손절 거리로 나눈 R): R 합 {lg['R_sum']:+.1f} · 평균 {lg['R_mean']:+.3f}")
    sdq = rep["stop_dist"]
    if sdq["quantiles"]:
        L.append("- 손절 거리 분위(10/25/50/75/90%): " + " / ".join(f"{v:.1%}" for v in sdq["quantiles"].values())
                 + f" · 하한이 구조 거리보다 넓어 하한이 걸린 거래 {sdq['n_floor_binding']}건({sdq['share_floor_binding']:.0%})")
    c1 = rep["cost_1x"]
    if c1["all"].get("n"):
        L.append(f"- 비용 1배(같은 선택·같은 손절): 실현 {c1['all']['n']}건 · R_acct 합 {c1['all']['R_sum']:+.1f} · 평균 {c1['all']['R_mean']:+.3f} · "
                 f"국면 ON R_acct 합 {_f(c1['on'].get('R_sum'), '+.1f')} · 계좌 수익 {c1['account_return']:+.2%}")
    x = b["b6_vs_expo_index"]
    L.append(f"- 계좌: 수익 {x['account_return']:+.2%}, 최대 낙폭 {rep['max_drawdown']:.1%}, KOSPI 같은 기간 {x['kospi_return']:+.2%}, KOSDAQ {x['kosdaq_return']:+.2%}; "
             f"계좌 월 수익 양수 {rep['monthly']['n_pos_account_months']}/{rep['monthly']['n_months']}")
    u = rep["unrealized"]
    L.append(f"- 미청산(2026-11-13, 실현 바 제외): {u['n']}건, 평가 R_acct 합 {_f(u['Racct_mtm_sum'], '+.2f')}, 평균 {_f(u['Racct_mtm_mean'], '+.3f')} — {', '.join(u['codes']) or '없음'}")
    L += ["", "| 구분 | n | R_acct 합 | R_acct 평균 | 승률 | 손익비 | 본전 승률 |", "|---|---|---|---|---|---|---|"]
    for k, s_ in rep["by_regime"].items():
        L.append(_row(k, s_))
    for k, s_ in rep["by_reason"].items():
        L.append(_row(f"청산: {k}", s_))
    for k, s_ in rep["by_tf"].items():
        L.append(_row(f"봉: {k}", s_))
    L.append(_row("밤 넘김 사건(하루 마지막 봉)", rep["overnight_event"]["last_bar_event"]))
    L.append(_row("장중 사건", rep["overnight_event"]["intraday_event"]))
    mo = rep["monthly"]
    L += ["", "월 수익(계좌) ~ 지수: " + "; ".join(
        f"{nm} β {', '.join(_f(v, '+.2f') for v in o.get('betas', []))} (t {', '.join(_f(v, '+.1f') for v in o.get('t_betas', []))}) · 절편 {_f(o.get('alpha'), '+.4f')} (t {_f(o.get('t_alpha'), '+.2f')}) n={o['n']}"
        for nm, o in (("KOSPI", mo["ols_kospi"]), ("KOSDAQ", mo["ols_kosdaq"]), ("KOSPI+KOSDAQ", mo["ols_both"]))), "",
        "| 월 | 계좌 | KOSPI | KOSDAQ |", "|---|---|---|---|"]
    for d, r in mo["table"].items():
        L.append(f"| {d[:7]} | {r['s']:+.2%} | {r['kospi']:+.2%} | {r['kosdaq']:+.2%} |")
    lk = res["leak_real"]
    L += ["", "## 누출 점검", "", "- 테스트 `tests/test_mtf_transition.py`(합성 자료): 1차 누출 테스트 전부 + MT-2 — R_acct 식, 손절 하한, 학습 점수 상위 3건, "
          "국면이 전날 자료만 씀(절단 불변 + 그날 지수를 전날로 당기면 바뀌는 음성 대조), MT-2 선택이 시험 달 교란에 불변(+ 다음 폴드 바뀌는 음성 대조), MT-2 절단 불변.",
          "- 실자료 절단(날짜 5개, 정시 절단, 지수도 그날 전까지):"]
    for r in lk["truncation"]:
        L.append(f"  - {r['day']} {r['hi'][11:16]} 전: 선택 같음 {r['picks_same']} · 학습 점수 비트 같음 {r['scores_same']}(최대 차 {r['scores_max_abs_diff']:.1e}) · "
                 f"국면 다른 날 {r.get('regime_days_diff')}/{r.get('regime_days')} · U1 비율 최대 차 {r.get('u1_share_max_abs_diff', float('nan')):.1e} "
                 f"(폴드 {', '.join(r['folds_checked']) or '없음'}) · "
                 + "; ".join(f"{k} 사건 {v['events']} 중 다름 {v['events_diff']}, 특징 다름 {v['events_feat_diff']}, 유니버스 다른 날 {v['univ_days_diff']}, up 다른 칸 {v['up_cells_diff']}"
                             for k, v in r["tf"].items()))
    p = lk["perturb"]
    L.append(f"- 실자료 교란({p['from']} 부터 분봉·일봉·거래대금·KOSDAQ 지수): 폴드 {p['folds_checked'][0]}~{p['folds_checked'][-1]} 학습 점수 비트 같음 {p['scores_same']}"
             f"(최대 차 {p['scores_max_abs_diff']:.1e}) · 선택 같음 {p['picks_same']} · 뒤 폴드 점수 바뀜 {p['later_scores_changed']}(음성 대조)")
    L += ["", "## 그림", "", "- `equity_v2.png` — OOS 단일 계좌 자산(비용 2배)·국면 ON 진입분 장부·노출 맞춘 KOSPI·KOSDAQ·널 ① 띠·국면 OFF 날 음영",
          "- `r_hist_v2.png` — 실현 R_acct 분포(전체·국면 ON)와 널 ② 무작위 진입 R_acct",
          "- `trade_grid_v2.png` — OOS 실현 거래 12건(R_acct 순위 등간격): 봉·거래량·MA·일봉 MA20/MA10·진입·청산·손절(하한)·구조선", "",
          "## 격자 관찰", "", "(그림을 열어 본 뒤 적는다)", "",
          "## Ruling — MT-2(결과 보기 전 실행기에 적은 해석)", ""]
    for k, v in res["rulings_v2"].items():
        L.append(f"- Ruling `v2.{k}`: {v}")
    L += ["", "1차 Ruling 은 RESULTS.md 와 같다(`results_v2.json` 의 `rulings`).", "", "## 재현", "", "```",
          "set -a && . ~/git/quant-airflow/.env && set +a && NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
          "  uv run --with numba --with matplotlib --extra pg --extra dev python research/experiments/mtf_transition_wf.py --v2", "```",
          f"캐시 `{SCRATCH}`(1차와 같은 캐시). 거래 목록 `oos_trades_v2.csv`."]
    (OUT / "RESULTS_v2.md").write_text("\n".join(L) + "\n")
    line = {"trial": "mtf_transition_v2", "date": time.strftime("%Y-%m-%d"), "git": res["git"], "dirty": res["dirty"], "prereg": res["prereg"],
            "nature": "진단(1차 결과·리뷰를 본 뒤 재등록), 판정 아님",
            "config": {"data": res["data"], "months": res["months"], "menu_axes": {k: list(v) for k, v in T.AXES_V2.items()}, "n_menu": len(T.menu(T.AXES_V2)),
                       "topn": T.TOPN, "slots": T.SLOTS, "w_cap": T.W_CAP, "risk": T.RISK, "cost_judge": COST_JUDGE, "min_trades": MIN_TRADES,
                       "n_null": res["n_null"], "seed": SEED, "score": "R_acct sum - top3", "floor": [T.FLOOR_COST_MULT, T.FLOOR_COST_K, T.FLOOR_ABS],
                       "regime": {"kosdaq_ma": T.REGIME_MA, "u1_min": T.REGIME_U1_MIN}},
            "selected": [f["pick"] for f in res["folds"]], "n_trials": len(T.menu(T.AXES_V2)),
            "verdict": {k: v["pass"] for k, v in b.items() if k.startswith("b")} | {"all_pass": b["all_pass"]},
            "verdict_regime_on": ({"void": True} if void_on else {k: v["pass"] for k, v in bo.items() if k.startswith("b")} | {"all_pass": bo["all_pass"]})}
    with open(OUT / "TRIALS.jsonl", "a") as fh:
        fh.write(json.dumps(_clean(line), ensure_ascii=False, allow_nan=False) + "\n")
    log(f"쓰기: {OUT} (*_v2)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--v2", action="store_true", help="재등록 MT-2 (PREREG 27334ef)")
    a = ap.parse_args(argv)
    if a.v2:
        run_v2(smoke=a.smoke)
    else:
        run(smoke=a.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
