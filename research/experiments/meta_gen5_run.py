#!/usr/bin/env python
"""메타 세대 5 — 코어 규칙의 진짜 표본 밖 검증: 1997~2016 코스피. **리포터 — 판정 아님.**

설계: ``research/logs/sepa_explore/META_GEN5.md`` (a071ad1 — 결과 보기 전). 이 러너는 숫자를 ``META_GEN5_RESULTS.md`` 에 적고
미리 적은 판정 조건 (1)~(5)와의 **산술 비교**만 한다. 판정은 사람이 한다. 이 세대에서 새로 고르는 것은 없다(규칙 C0·C1f·C1m·C4 고정).

**자산:** 코스피 가격 지수(``research/data/krx_index_daily_yahoo.csv`` 의 ``KOSPI`` — Yahoo ^KS11, 네이버 공식값과 겹치는 2,996일 대조).
코스피200 은 1997~2016 원천이 없어 코스피로 대용한다(2017+ 두 지수 일 수익 상관은 ``krx_index_daily_naver.csv`` 로 함께 적는다).

**현금:** 판정 = **현금 0%**(시장 밖 이자 없음 — 타이밍에 가장 불리한 보수 가정; C4 의 현금 12개월 수익 문턱도 0). 기억값 CD91 줄
(:data:`CD91_MEM`, DB 밖)은 기술로만 적는다(현금 이자·C4 문턱·초과 Sharpe 무위험 모두 그 값).

**체결·지표·부트스트랩·널:** 메타 세대 4 러너(``meta_gen4_run``)를 그대로 import — d 종가 신호 → d+1 시가, 편도 0.07%, 보수 연 0.05%,
일별 경로 MDD·Calmar, 초과 Sharpe(현금 0 이면 원 Sharpe), 짝지은 달 블록(3개월, 참고 12개월) 부트스트랩, 마르코프 널(C1m·C4 는 달 단위 =
월말 전환) + 원형 이동 널. 배당 = 연 1.5% × 시장 밖 비율을 타이밍에서 깎은 고정 보정 줄(판정 (3)), 1.0%·2.0% 민감도는 참고.

**창:** 판정 = 표본 밖 1997-12-01 ~ 2016-12-31(분리 1997-12 ~ 2006-12 · 2007-01 ~ 2016-12 — 같은 계열을 자른 것). 기술 = 2017-01 ~ 2026-09
(세대 4 창, 2017-01 현금에서 새로 시작 — 단 지수가 코스피200 이 아니라 코스피) · 전체 1997-12 ~ 2026-09.

Run:  uv run python research/experiments/meta_gen5_run.py
      (DB 접속 없음 — CSV 만 읽는다.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import ga_run as g1  # noqa: E402
import meta_gen1_run as mg1  # noqa: E402
import meta_gen2_run as mg2  # noqa: E402
import meta_gen4_run as m4  # noqa: E402

from research.signals import meta_fast as mf  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
SUB = "meta_gen5"
RESULTS_MD = "META_GEN5_RESULTS.md"
DESIGN_DOC = "research/logs/sepa_explore/META_GEN5.md (a071ad1)"
INDEX_CSV = "research/data/krx_index_daily_yahoo.csv"
NAVER_CSV = "research/data/krx_index_daily_naver.csv"
CORE_SYM = "KOSPI"
GEN4_SUMMARY = "research/logs/sepa_explore/meta_gen4/summary.csv"

# --- 설계 수치(META_GEN5.md — 세대 4 그대로, 탐색하지 않는다) ---
COST_1W, FEE_YR = m4.COST_1W, m4.FEE_YR
DIV_YR = 0.015                    # 배당 고정 보정 줄: 연 1.5% × 시장 밖 비율(판정 (3))
DIV_SENS = (0.010, 0.020)         # 참고: 1997~2016 코스피 배당수익률 대체로 1~2%
MA_N = 200
DM_MONTHS = m4.DM_MONTHS
N_RAND, N_BOOT = m4.N_RAND, m4.N_BOOT
BLOCK_M, BLOCK_ALT = m4.BLOCK_M, m4.BLOCK_ALT
ROT_MIN = m4.ROT_MIN
SEED = 20261010
P_BOOT, MDD_GAIN, CAGR_LOSS = m4.P_BOOT, m4.MDD_GAIN, m4.CAGR_LOSS
#: CD91 연평균 근사(비율) — **기억값(DB 에 금리 테이블이 없다)**. 1997~2016 = 이 러너 작성 시 기억(한국은행 연평균 근사),
#: 2017+ = 메타 세대 4 ``CD91``(세대 3·1 리뷰어 값 그대로). 기술 줄에만 쓴다.
CD91_MEM = {1997: 0.1324, 1998: 0.1498, 1999: 0.0684, 2000: 0.0708, 2001: 0.0532, 2002: 0.0481, 2003: 0.0431, 2004: 0.0379,
            2005: 0.0365, 2006: 0.0448, 2007: 0.0516, 2008: 0.0549, 2009: 0.0263, 2010: 0.0267, 2011: 0.0344, 2012: 0.0330,
            2013: 0.0272, 2014: 0.0249, 2015: 0.0177, 2016: 0.0149, **m4.CD91}
W_LO, W_HI = "1997-12-01", "2016-12-31"
H1 = ("1997-12-01", "2006-12-31")
H2 = ("2007-01-01", "2016-12-31")
G4 = ("2017-01-01", "2026-09-30")
FULL_HI = "2026-09-30"
RULES = ("C1f", "C1m", "C4")
PROCS = ("C0",) + RULES
MONTHLY = ("C1m", "C4")           # 월말에만 전환 → 마르코프 널도 달 단위
CASHES = {"zero": "현금 0%(판정)", "cd91": "기억값 CD91(기술)"}
N_TOP = 5
#: 색 = 메타 세대 4 그림과 같은 참조 팔레트 순서(C1m 은 세대 4 의 4번째 색).
COLORS = {"C0": "#2a78d6", "C1f": "#eb6834", "C1m": "#eda100", "C4": "#1baf7a"}


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


_pct, _f = g1._pct, g1._f


# ---------------------------------------------------------------------------
# 순수 헬퍼 (tests/test_meta_gen5_helpers.py)
# ---------------------------------------------------------------------------

def ma_month_signal(close: np.ndarray, n: int, month_id: np.ndarray) -> np.ndarray:
    """C1m 월말 확인 이동평균 신호: 매월 마지막 거래일 d 에 종가_d > n일 이동평균_d(``meta_gen4_run.ma_signal`` 과 같은 정의)이면 참,
    다음 월말 전날까지 유지(→ 체결 하루 밀림으로 다음 달 첫 거래일 시가 ~ 그 달 마지막 거래일 보유). 첫 월말 전·이동평균 없는 월말은 거짓."""
    daily = m4.ma_signal(close, n)
    me = mg1.month_ends(np.asarray(month_id))
    x = np.zeros(len(daily), bool)
    for k, d in enumerate(me):
        nxt = me[k + 1] if k + 1 < len(me) else len(daily)
        x[d:nxt] = daily[d]
    return x


def top_drawdowns(eq: np.ndarray, k: int = N_TOP) -> "list[dict]":
    """자산 경로의 낙폭 에피소드(새 고점 사이 구간) 중 깊은 순 ``k`` 개: 고점 행 · 저점 행 · 회복 행(고점 이상 첫 행, 없으면 None) · 낙폭."""
    eq = np.asarray(eq, float)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    new_hi = np.flatnonzero(eq >= peak * (1.0 - 1e-12))
    out = []
    for a, b in zip(new_hi, np.r_[new_hi[1:], len(eq)], strict=True):
        if b - a > 1:
            t = int(a + np.argmin(dd[a:b]))
            out.append({"peak": int(a), "trough": t, "recover": int(b) if b < len(eq) else None, "dd": float(dd[t])})
    return sorted(out, key=lambda e: e["dd"])[:k]


def span_path(r: np.ndarray, p: np.ndarray, a: int, t: int) -> dict:
    """확장 자산 행 a(고점) → t(저점) 사이(= 날 수익 행 [a, t)) 한 계열의 누적 수익 · 그 안 최대 낙폭 · 평균 보유."""
    rr = np.asarray(r, float)[a:t]
    if not len(rr):
        return {"ret": 0.0, "dd": 0.0, "tim": np.nan}
    e = np.r_[1.0, np.cumprod(1.0 + rr)]
    return {"ret": float(e[-1] - 1.0), "dd": float(min((e / np.maximum.accumulate(e) - 1.0).min(), 0.0)),
            "tim": float(np.mean(np.asarray(p, float)[a:t]))}


def calendar_cagr(total: float, d_first: pd.Timestamp, d_last: pd.Timestamp) -> float:
    """달력 기준 CAGR(참고): (1 + 누적)^(365.25 / 일수) − 1, 일수 = 첫 행 직전 종가 ~ 마지막 종가 근사(첫날~끝날 + 1일)."""
    days = (pd.Timestamp(d_last) - pd.Timestamp(d_first)).days + 1
    return float((1.0 + total) ** (365.25 / days) - 1.0) if total > -1 and days > 0 else np.nan


# ---------------------------------------------------------------------------
# 세계
# ---------------------------------------------------------------------------

class World5:
    """코스피 일봉(Yahoo) · 현금(0 또는 기억값 CD91) · 달 번호 · 규칙 신호. ``meta_gen4_run`` 엔진을 그대로 쓴다."""

    def __init__(self, cash: str):
        x = pd.read_csv(INDEX_CSV, parse_dates=["date"])
        k = x[x["symbol"] == CORE_SYM].set_index("date").sort_index()
        self.dates = pd.DatetimeIndex(k.index)
        self.O, self.C = k["open"].to_numpy(float), k["close"].to_numpy(float)
        self.ov = np.r_[np.nan, self.O[1:] / self.C[:-1] - 1.0]
        self.iday = self.C / self.O - 1.0
        self.mon = mg2.month_ids(self.dates)
        self.cash = cash
        self.rate = np.zeros(len(self.dates)) if cash == "zero" else mg2.cash_daily(self.dates, CD91_MEM)
        self.x = {"C0": np.ones(len(self.dates), bool), "C1f": m4.ma_signal(self.C, MA_N),
                  "C1m": ma_month_signal(self.C, MA_N, self.mon), "C4": m4.dual_momentum_signal(self.C, self.rate, self.mon, DM_MONTHS)}

    def rows(self, lo: str, hi: str) -> "tuple[int, int]":
        return int(self.dates.searchsorted(pd.Timestamp(lo))), int(self.dates.searchsorted(pd.Timestamp(hi), "right"))

    def run(self, p: np.ndarray, i0: int, i1: int) -> np.ndarray:
        return m4.core_returns(self.ov[i0:i1], self.iday[i0:i1], self.rate[i0:i1], p, 0.0)

    def series(self, nm: str, lo: str, hi: str) -> dict:
        """창 첫날 직전 = 현금(p = 0) 에서 시작한 계열."""
        i0, i1 = self.rows(lo, hi)
        p = m4.held(self.x[nm], i0, i1)
        return {"a": i0, "p": p, "r": self.run(p, i0, i1)}

    def sl(self, s: dict, lo: str, hi: str) -> dict:
        """계열을 날짜 범위로 자른다(규칙 상태는 이어진다) → r·p·d·rate·i0(전체 행)."""
        d = self.dates[s["a"]:s["a"] + len(s["r"])]
        m = np.asarray((d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi)))
        j = np.flatnonzero(m)
        return {"r": s["r"][m], "p": s["p"][m], "d": d[m], "i0": s["a"] + int(j[0]), "rate": self.rate[s["a"] + j]}


def wstats(r: np.ndarray, p: np.ndarray, d: pd.DatetimeIndex, rate: np.ndarray) -> dict:
    """한 창: CAGR(252/T — 저장소 ``equity_stats``)·달력 CAGR(참고)·MDD·Calmar(일별)·초과 SR(현금 0 이면 원 SR)·원 SR·시장 안·전환
    + 배당 고정 보정 줄(1.5% — 판정 (3), 1.0·2.0% 참고)."""
    r, p = np.asarray(r, float), np.asarray(p, float)
    mon = mg2.month_ids(d)
    s = mf.equity_stats(r)
    s["cagr_cal"] = calendar_cagr(s["total"], d[0], d[-1])
    s["sharpe"] = float(m4.excess_sharpe(r, rate, mon)[0])
    s["sharpe_raw"] = float(m4.excess_sharpe(r, np.zeros(len(r)), mon)[0])
    pi, sw = m4.timing_stats(p)
    s.update({"tim": pi, "n_sw": int((p[1:] != p[:-1]).sum()), "sw_py": sw * (len(p) - 1) / (len(p) / 252.0) if len(p) > 1 else 0.0})
    for y in (DIV_YR,) + DIV_SENS:
        e = mf.equity_stats(m4.div_adjust(r, p, y))
        s[f"cagr_div{int(round(y * 1000))}"] = e["cagr"]
        if y == DIV_YR:
            s.update({"cagr_div": e["cagr"], "mdd_div": e["mdd"], "calmar_div": e["calmar"]})
    s.update({"start": str(d[0].date()), "end": str(d[-1].date())})
    return s


def index_corr() -> dict:
    """2017-01 ~ 2026-09 코스피 vs 코스피200(네이버) 일 종가 수익 상관 · 연 변동성 · 베타 · 누적 + Yahoo 코스피 vs 네이버 코스피 상관."""
    nv = pd.read_csv(NAVER_CSV, parse_dates=["date"])
    cl = nv.pivot_table(index="date", columns="symbol", values="close").sort_index()
    yh = pd.read_csv(INDEX_CSV, parse_dates=["date"])
    cl["YKOSPI"] = yh[yh["symbol"] == CORE_SYM].set_index("date")["close"]
    cl = cl[["KOSPI", "KPI200", "YKOSPI"]].dropna()
    r = cl.pct_change()
    r = r[(r.index >= pd.Timestamp(G4[0])) & (r.index <= pd.Timestamp(G4[1]))].dropna()
    sd = r.std() * np.sqrt(252.0)
    out = {"n": len(r), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
           "corr": float(r["KOSPI"].corr(r["KPI200"])), "corr_yahoo": float(r["YKOSPI"].corr(r["KOSPI"])),
           "vol_kospi": float(sd["KOSPI"]), "vol_k200": float(sd["KPI200"]),
           "beta": float(np.cov(r["KPI200"], r["KOSPI"])[0, 1] / r["KOSPI"].var()),
           "tot_kospi": float(np.prod(1 + r["KOSPI"]) - 1), "tot_k200": float(np.prod(1 + r["KPI200"]) - 1),
           "maxabs_yahoo": float((r["YKOSPI"] - r["KOSPI"]).abs().max())}
    return out


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def plot_equity(panels: "list[tuple[str, pd.DatetimeIndex, dict]]", path: str) -> "str | None":
    """창별 자산(로그, 위)·낙폭(아래). 색은 절차를 따른다(세대 4 그림과 같은 팔레트). 점선 = 2007-01(판정 분리 경계)·2017-01(표본 밖 끝).
    그림 글자는 영어(이 기계에 한글 글꼴이 없다)."""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: PLC0415
    except ImportError:
        return None
    plt.rcParams["axes.unicode_minus"] = False
    widths = [max(1.0, (d[-1] - d[0]).days / 365.25 / 10.0) for _, d, _ in panels]
    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * sum(widths), 6.6), sharex="col", facecolor="#fcfcfb",
                             gridspec_kw={"height_ratios": [2.2, 1.0], "width_ratios": widths})
    marks = [pd.Timestamp(H2[0]), pd.Timestamp(G4[0])]
    for j, (title, d, series) in enumerate(panels):
        a0, a1 = axes[0, j], axes[1, j]
        for nm, r in series.items():
            eq = np.cumprod(1.0 + r)
            a0.plot(d, eq, color=COLORS.get(nm, "#52514e"), lw=2.0 if nm == "C0" else 1.3, label=nm)
            a1.plot(d, eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1.0, color=COLORS.get(nm, "#52514e"), lw=1.1)
        a0.set_yscale("log")
        a0.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        a0.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
        a0.yaxis.set_minor_formatter(NullFormatter())
        a1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        a0.set_title(title, fontsize=10, color="#0b0b0b")
        a0.set_ylabel("equity (log)" if j == 0 else "", fontsize=9, color="#52514e")
        a1.set_ylabel("drawdown" if j == 0 else "", fontsize=9, color="#52514e")
        for ax in (a0, a1):
            ax.set_facecolor("#fcfcfb")
            for m in marks:
                if d[0] < m < d[-1]:
                    ax.axvline(m, color="#8a8984", lw=0.8, ls=(0, (4, 3)), zorder=0)
            ax.grid(True, color="#e6e5e0", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=8, colors="#52514e")
        a0.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("META GEN 5 - KOSPI price index, buy-and-hold vs fixed timing rules (cash 0%, costs incl.; "
                 "dashed = 2007-01 split, 2017-01 end of out-of-sample)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--nulls", type=int, default=N_RAND)
    ap.add_argument("--boots", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    debug = os.path.normpath(args.out) != os.path.normpath(OUT_DIR)
    if (args.nulls != N_RAND or args.boots != N_BOOT) and not debug:
        ap.error("축소 실행은 --out 을 탐색 디렉터리 밖으로 둘 때만")
    sub = os.path.join(args.out, SUB)
    os.makedirs(sub, exist_ok=True)
    t0 = time.time()
    WS = {c: World5(c) for c in CASHES}
    D = WS["zero"].dates
    _log(f"KOSPI {D[0].date()} ~ {D[-1].date()} ({len(D)}일)", t0)

    # --- 계열: base = 1997-12 ~ 2026-09(판정·분리·전체 창은 이것을 자른다), g4 = 2017-01 현금에서 새로 시작 ---
    base = {c: {nm: W.series(nm, W_LO, FULL_HI) for nm in PROCS} for c, W in WS.items()}
    g4 = {c: {nm: W.series(nm, *G4) for nm in PROCS} for c, W in WS.items()}
    WINS = {"oos": (W_LO, W_HI, "판정 창 1997-12 ~ 2016-12 (표본 밖)", base),
            "h1": (*H1, "분리 1997-12 ~ 2006-12", base), "h2": (*H2, "분리 2007-01 ~ 2016-12", base),
            "g4": (*G4, "기술: 2017-01 ~ 2026-09 (세대 4 창, 코스피 — 2017-01 현금 시작)", g4),
            "full": (W_LO, FULL_HI, "기술: 전체 1997-12 ~ 2026-09", base)}
    stats = {}
    for c, W in WS.items():
        for wk, (lo, hi, _, src) in WINS.items():
            for nm in PROCS:
                q = W.sl(src[c][nm], lo, hi)
                stats[(c, nm, wk)] = wstats(q["r"], q["p"], q["d"], q["rate"])
    _log("계열·창 통계 끝", t0)

    # --- 판정 산술(현금 0) + 같은 산술의 기억값 CD91 줄(참고) ---
    boots, nulls, rows = {}, {}, []
    for ci, (c, W) in enumerate(WS.items()):
        q0 = W.sl(base[c]["C0"], W_LO, W_HI)
        mon = mg2.month_ids(q0["d"])
        B3 = m4.MonthBoot(mon, BLOCK_M, args.boots, SEED + BLOCK_M)
        B12 = m4.MonthBoot(mon, BLOCK_ALT, args.boots, SEED + BLOCK_ALT)
        st0 = stats[(c, "C0", "oos")]
        cs0 = (stats[(c, "C0", "h1")]["calmar"], stats[(c, "C0", "h2")]["calmar"])
        for k, nm in enumerate(RULES):
            q = W.sl(base[c][nm], W_LO, W_HI)
            assert (q["d"] == q0["d"]).all()
            b3, b12 = m4.paired_boot(q["r"], q0["r"], q["rate"], B3), m4.paired_boot(q["r"], q0["r"], q["rate"], B12)
            boots[(c, nm)] = {"b3": b3, "b12": b12}
            i0, i1 = q["i0"], q["i0"] + len(q["r"])
            rng = np.random.default_rng([SEED, ci, k])
            Pm = (m4.markov_month_paths(args.nulls, q["p"], mon, rng) if nm in MONTHLY
                  else m4.markov_paths(args.nulls, len(q["p"]), *m4.timing_stats(q["p"]), rng)).astype(float)
            Pc = m4.circular_paths(q["p"], args.nulls, rng)
            nm_ = {}
            for kind, P in (("markov", Pm), ("circ", Pc)):
                m = m4.path_metrics(W.run(P, i0, i1), mon, q["rate"])
                m.update({"pi_mean": float(P.mean()), "s_mean": float((P[:, 1:] != P[:, :-1]).mean())})
                nm_[kind] = m
            pi, s = m4.timing_stats(q["p"])
            nulls[(c, nm)] = {**nm_, "pi": pi, "s": s, "unit": "달" if nm in MONTHLY else "날"}
            st = stats[(c, nm, "oos")]
            cs = (stats[(c, nm, "h1")]["calmar"], stats[(c, nm, "h2")]["calmar"])
            mk95, cc95 = m4.qs(nm_["markov"]["calmar"], 95), m4.qs(nm_["circ"]["calmar"], 95)
            both95 = float(np.max([mk95, cc95]))                     # (4) 두 널 모두 위 = 둘 중 큰 p95 위(NaN 이면 NaN)
            rows.append({"cash": c, "proc": nm, "sw_py": st["sw_py"], "n_params": 0, "split": cs, "split0": cs0,
                         "markov_p95": mk95, "circ_p95": cc95,
                         "markov_pct": mg1.null_pct(st["calmar"], nm_["markov"]["calmar"]),
                         "circ_pct": mg1.null_pct(st["calmar"], nm_["circ"]["calmar"]),
                         "crit": m4.criteria(st, st0, b3, both95, cs, cs0),
                         "crit_each": m4.criteria(st, st0, b3, both95, cs, cs0, joint=False),
                         "crit_b12": m4.criteria(st, st0, b12, both95, cs, cs0),
                         "c4_markov": bool(np.isfinite(mk95) and st["calmar"] > mk95),
                         "c4_circ": bool(np.isfinite(cc95) and st["calmar"] > cc95),
                         "c3_sens": {y: bool(st0["cagr"] - st[f"cagr_div{int(round(y * 1000))}"] <= CAGR_LOSS + 1e-12) for y in DIV_SENS}})
    _log("부트스트랩 · 무작위 타이밍 널 끝", t0)
    vrows = [r for r in rows if r["cash"] == "zero"]
    passed = [r["proc"] for r in vrows if r["crit"]["all"]]
    simp = [r["proc"] for r in m4.simplest_pass(vrows)]
    passed_cd = [r["proc"] for r in rows if r["cash"] == "cd91" and r["crit"]["all"]]

    # --- 낙폭 에피소드(현금 0): 절차마다 자기 top 5 + C0 top 5 동안 규칙들 ---
    W0 = WS["zero"]
    episodes, c0_eps = [], {}
    for wk in ("oos", "full", "g4"):
        lo, hi, _, src = WINS[wk]
        qs_ = {nm: W0.sl(src["zero"][nm], lo, hi) for nm in PROCS}
        dext = W0.dates[qs_["C0"]["i0"] - 1:qs_["C0"]["i0"] + len(qs_["C0"]["r"])]   # 행 0 = 창 직전 종가
        lab = lambda j: str(dext[j].date()) if j is not None else "미회복"  # noqa: E731
        for nm in PROCS:
            eq = np.r_[1.0, np.cumprod(1.0 + qs_[nm]["r"])]
            for rank, e in enumerate(top_drawdowns(eq, N_TOP), 1):
                episodes.append({"win": wk, "proc": nm, "rank": rank, "peak": lab(e["peak"]), "trough": lab(e["trough"]),
                                 "recover": lab(e["recover"]), "dd": e["dd"], "days_down": e["trough"] - e["peak"],
                                 "days_recover": (e["recover"] - e["trough"]) if e["recover"] is not None else None})
        eq0 = np.r_[1.0, np.cumprod(1.0 + qs_["C0"]["r"])]
        c0_eps[wk] = []
        for e in top_drawdowns(eq0, N_TOP):
            row = {"peak": lab(e["peak"]), "trough": lab(e["trough"]), "recover": lab(e["recover"]), "dd": e["dd"],
                   "days": e["trough"] - e["peak"]}
            for nm in RULES:
                row[nm] = span_path(qs_[nm]["r"], qs_[nm]["p"], e["peak"], e["trough"])
            c0_eps[wk].append(row)

    # --- 코스피 vs 코스피200 · 세대 4 숫자 ---
    corr = index_corr()
    g4ref = None
    if os.path.exists(GEN4_SUMMARY):
        s4 = pd.read_csv(GEN4_SUMMARY)
        g4ref = {r.proc: r for r in s4[s4["win"] == "full"].itertuples()}
    _log("에피소드 · 상관 끝", t0)

    # --- 산출물 ---
    pd.DataFrame([{"cash": c, "proc": nm, "win": wk, **{k: v for k, v in s.items() if not isinstance(v, (list, dict))}}
                  for (c, nm, wk), s in stats.items()]).to_csv(os.path.join(sub, "summary.csv"), index=False)
    pd.DataFrame([{"cash": r["cash"], "proc": r["proc"], **{f"crit_{k}": v for k, v in r["crit"].items()},
                   "crit_each_all": r["crit_each"]["all"], "crit_b12_all": r["crit_b12"]["all"], "c4_markov": r["c4_markov"],
                   "c4_circ": r["c4_circ"], "null_markov_calmar_p95": r["markov_p95"], "null_circ_calmar_p95": r["circ_p95"],
                   "null_markov_pct": r["markov_pct"], "null_circ_pct": r["circ_pct"],
                   "calmar_h1": r["split"][0], "calmar_h2": r["split"][1], "c0_calmar_h1": r["split0"][0], "c0_calmar_h2": r["split0"][1],
                   **{f"c3_div{int(round(y * 1000))}": v for y, v in r["c3_sens"].items()},
                   **{f"b3_{k}": v for k, v in boots[(r["cash"], r["proc"])]["b3"].items() if not isinstance(v, list)},
                   **{f"b12_{k}": v for k, v in boots[(r["cash"], r["proc"])]["b12"].items() if not isinstance(v, list)}}
                  for r in rows]).to_csv(os.path.join(sub, "decision.csv"), index=False)
    pd.DataFrame([{"cash": c, "proc": nm, "null": kind, "calmar": a, "sharpe": b, "mdd": m_, "cagr": g}
                  for (c, nm), n in nulls.items() for kind in ("markov", "circ")
                  for a, b, m_, g in zip(n[kind]["calmar"], n[kind]["sharpe"], n[kind]["mdd"], n[kind]["cagr"], strict=True)]).to_csv(
        os.path.join(sub, "null_random.csv"), index=False)
    pd.DataFrame(episodes).to_csv(os.path.join(sub, "episodes.csv"), index=False)
    pd.DataFrame([{"win": wk, **{k: v for k, v in e.items() if k not in RULES},
                   **{f"{nm}_{k}": e[nm][k] for nm in RULES for k in ("ret", "dd", "tim")}}
                  for wk, es in c0_eps.items() for e in es]).to_csv(os.path.join(sub, "c0_episodes.csv"), index=False)
    i0b = base["zero"]["C0"]["a"]
    nav = pd.DataFrame(index=D[i0b:i0b + len(base["zero"]["C0"]["r"])])
    for c in CASHES:
        for nm in PROCS:
            s = base[c][nm]
            nav[f"{nm} [{c}]"] = np.cumprod(1.0 + s["r"])
            if c == "zero":
                nav[f"{nm} 보유"] = s["p"]
    nav.to_csv(os.path.join(sub, "nav.csv"))
    yrows = []
    for c in CASHES:
        for nm in PROCS:
            s = base[c][nm]
            d = D[s["a"]:s["a"] + len(s["r"])]
            for y in sorted(set(d.year)):
                m = np.asarray(d.year == y)
                e = mf.equity_stats(s["r"][m])
                yrows.append({"cash": c, "proc": nm, "year": int(y), "ret": e["total"], "mdd": e["mdd"], "tim": float(np.mean(s["p"][m]))})
    ydf = pd.DataFrame(yrows)
    ydf.to_csv(os.path.join(sub, "peryear.csv"), index=False)

    def seg(src, lo, hi):
        out = {nm: W0.sl(src["zero"][nm], lo, hi) for nm in PROCS}
        return out["C0"]["d"], {nm: q["r"] for nm, q in out.items()}
    fig = plot_equity([("Out-of-sample verdict 1997-12 ~ 2016-12", *seg(base, W_LO, W_HI)),
                       ("Gen 4 window 2017-01 ~ 2026-09 (KOSPI, fresh start)", *seg(g4, *G4)),
                       ("Full 1997-12 ~ 2026-09 (descriptive)", *seg(base, W_LO, FULL_HI))], os.path.join(sub, "equity.png"))
    led = write_ledger_rows(sub)
    elapsed = time.time() - t0
    L = render(locals())
    with open(os.path.join(args.out, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n=== 메타 세대 5 — {elapsed:.0f}초 · 통과(1–5, 현금 0): {', '.join(passed) or '없음'} · "
          f"코어: {simp[0] if simp else 'C0'} · 참고 CD91 줄 통과: {', '.join(passed_cd) or '없음'} ===")
    print(f"→ {os.path.join(args.out, RESULTS_MD)}")
    return 0


def write_ledger_rows(sub: str) -> int:
    """원장 ``meta_gen5/TRIALS.jsonl`` — 절차 한 줄씩(결과 숫자는 넣지 않는다)."""
    path = os.path.join(sub, "TRIALS.jsonl")
    seen = g1.read_fingerprints(path)
    base = {"design": DESIGN_DOC, "asset": f"{CORE_SYM} 가격 지수(Yahoo ^KS11)", "window": f"{W_LO} ~ {W_HI}", "cost_1w": COST_1W,
            "fee_yr": FEE_YR, "div_yr": DIV_YR, "exec": "d 종가 신호 → d+1 시가", "cash": "0%(판정)", "sharpe": "원(현금 0)",
            "calmar": "일별 경로"}
    cfgs = [{"procedure": "C0"}, {"procedure": "C1f", "ma": MA_N}, {"procedure": "C1m", "ma": MA_N, "check": "월말"},
            {"procedure": "C4", "months": DM_MONTHS}]
    rows = [{"config": {**base, **c}, "note": "메타 세대 5 — 1997~2016 표본 밖"} for c in cfgs]
    return g1.write_ledger(path, rows, seen)


# ---------------------------------------------------------------------------
# 서식
# ---------------------------------------------------------------------------

_ck, _pp = m4._ck, m4._pp


def _q3(b: dict, key: str) -> str:
    return " · ".join(_f(q) for q in b[key])


def render(v: dict) -> "list[str]":
    args, stats, rows, boots, nulls, WINS = v["args"], v["stats"], v["rows"], v["boots"], v["nulls"], v["WINS"]
    passed, simp, passed_cd, corr, g4ref = v["passed"], v["simp"], v["passed_cd"], v["corr"], v["g4ref"]
    episodes, c0_eps, ydf = v["episodes"], v["c0_eps"], v["ydf"]
    D = v["D"]
    L = [f"# 메타 세대 5 — 코어 규칙의 표본 밖 검증 1997~2016(코스피) — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         "설계 `META_GEN5.md`(a071ad1, 결과 보기 전)를 계산한 숫자다. ✓/✗ 는 미리 적은 문턱과의 **산술 비교**이고 판정은 사람이 한다. "
         "규칙 C0·C1f·C1m·C4 는 세대 4 에서 고정된 것이고 이 세대에서 고른 것은 없다.", "",
         f"재현: `uv run python research/experiments/meta_gen5_run.py` · 실행 {v['elapsed']:.0f}초 · 무작위 타이밍 널 {args.nulls}벌 × 2종/규칙 · "
         f"부트스트랩 {args.boots}벌 · DB 접속 없음.", ""]
    if v["debug"]:
        L += [f"> **디버그 실행** — 출력 `{args.out}`. 측정으로 쓰지 않는다.", ""]

    L += ["## 먼저 읽을 것", "",
          f"- **자산이 세대 4 와 다르다:** 코스피 가격 지수(Yahoo ^KS11, `{INDEX_CSV}`), 코스피200 아님. 2017-01 ~ 2026-09 에서 두 지수 일 종가 수익 상관 "
          f"**{corr['corr']:.4f}**({corr['n']}일, 네이버), 연 변동성 코스피 {corr['vol_kospi'] * 100:.1f}% · 코스피200 {corr['vol_k200'] * 100:.1f}%, "
          f"코스피200 의 코스피 베타 {corr['beta']:.3f}, 같은 기간 누적 코스피 {_pct(corr['tot_kospi'])} · 코스피200 {_pct(corr['tot_k200'])}. "
          f"Yahoo 코스피 vs 네이버 코스피 일 수익 상관 {corr['corr_yahoo']:.6f}(최대 차 {corr['maxabs_yahoo'] * 100:.3f}%p).",
          "- **현금 0%(판정):** 시장 밖 구간은 이자를 받지 않는다. C4 의 '현금 12개월 수익' 문턱도 0(= 12개월 가격 수익 > 0). 원 Sharpe = 초과 Sharpe. "
          "기억값 CD91 줄은 아래 표의 기술 줄·참고 열이다(값은 '자료·가정' — **기억값, DB 밖**).",
          "- **배당 제외 가격 지수:** 판정 (3)은 연 1.5% × 시장 밖 비율을 타이밍 CAGR 에서 깎은 줄로 본다(C0 는 그대로).",
          "- **2000-12 까지 토요일 장(반일)이 Yahoo 자료에 없다** — 토요일 움직임은 다음 월요일 밤 수익에 들어간다(종가 계열은 이어짐). "
          "그래서 해마다 거래일이 241~253일이고, CAGR(252/T 저장소 관례)은 달력 CAGR 과 조금 다르다(두 열 모두 적는다).",
          "- **워밍업:** 자료 시작 1996-12-11. MA200 은 1997-10 부터 있고, C4 는 첫 12개월 비교가 1997-12-31 월말이라 **1997-12 한 달은 현금**(설계 창 그대로).",
          "- 분리 창(1997-12 ~ 2006-12 · 2007-01 ~ 2016-12)은 판정 창 계열을 자른 것(규칙 상태 이어짐). 2017~2026 창은 세대 4 처럼 2017-01-02 시가에 현금에서 새로 시작.", ""]

    # --- 판정 표 ---
    L += ["## 판정 산술 (META_GEN5.md '판정 구간과 바', 현금 0%)", "",
          "C0(코스피 매수보유) 대비 **모두**: (1) 짝지은 달 블록 부트스트랩(판정 창 일별 경로를 3개월 블록으로 재표본, "
          f"{args.boots}벌)에서 P(Sharpe > C0 **그리고** Calmar > C0) ≥ 0.8 — 세대 4 와 같은 엄격한 읽기(동시) · (2) MDD 가 10%p 이상 얕음 · "
          "(3) CAGR_C0 − CAGR(배당 1.5% 고정 보정 줄) ≤ 2%p · (4) Calmar > 무작위 타이밍 널 Calmar p95 — **마르코프(C1m·C4 는 월말 전환)와 원형 이동 둘 다** · "
          "(5) 1997-12 ~ 2006-12 · 2007-01 ~ 2016-12 **각각** Calmar ≥ C0.", "",
          "| 규칙 | (1) P(SR) · P(Calmar) · **P(둘 다)** | (2) ΔMDD | (3) CAGR_C0 − CAGR(배당 1.5% 보정) | "
          "(4) Calmar · 마르코프 p95 (백분위) · 원형 p95 (백분위) | (5) Calmar 97-06 · C0 / 07-16 · C0 | **통과(1–5)** | "
          "참고: (1) 각각 · (1) 12개월 블록 · CD91 줄로 전부 |", "|---|---|---|---|---|---|---|---|"]
    rcd = {r["proc"]: r for r in rows if r["cash"] == "cd91"}
    for r in (x for x in rows if x["cash"] == "zero"):
        nm = r["proc"]
        st, st0, b, c = stats[("zero", nm, "oos")], stats[("zero", "C0", "oos")], boots[("zero", nm)]["b3"], r["crit"]
        L.append(f"| **{nm}** | {b['p_sharpe']:.2f} · {b['p_calmar']:.2f} · **{b['p_both']:.2f}** {_ck(c['c1'])} | "
                 f"{_pp(st['mdd'] - st0['mdd'])} ({_pct(st['mdd'])} vs {_pct(st0['mdd'])}) {_ck(c['c2'])} | "
                 f"{_pp(st0['cagr'] - st['cagr_div'])} ({_pct(st0['cagr'])} − {_pct(st['cagr_div'])}) {_ck(c['c3'])} | "
                 f"{_f(st['calmar'])} · {_f(r['markov_p95'])} ({r['markov_pct'] * 100:.0f}%) {_ck(r['c4_markov'])} · "
                 f"{_f(r['circ_p95'])} ({r['circ_pct'] * 100:.0f}%) {_ck(r['c4_circ'])} → {_ck(c['c4'])} | "
                 f"{_f(r['split'][0])} · {_f(r['split0'][0])} / {_f(r['split'][1])} · {_f(r['split0'][1])} {_ck(c['c5'])} | "
                 f"**{_ck(c['all'])}** | {_ck(r['crit_each']['all'])} · {_ck(r['crit_b12']['all'])} · {_ck(rcd[nm]['crit']['all'])} |")
    L += ["", f"**산술 결과:** 통과(1–5) = {', '.join(passed) or '없음'}. "
          + (f"통과 중 전환/년 적은 순 = {', '.join(simp)}." if simp else ""), ""]
    if simp:
        L += [f"**적용되는 결론 규칙(설계 '결론 규칙' 첫째 줄):** 타이밍 규칙이 표본 밖 1997~2016 에서 통과 → 코어 = **{simp[0]}**"
              "(동률이면 전환 적은 것). 2017~2026(세대 4, 표본 내) 성과는 아래 기술 표, 2026-07 같은 빠른 급락에는 소용없다는 한계는 아래 에피소드 표와 함께 본다.", ""]
    else:
        L += ["**적용되는 결론 규칙(설계 '결론 규칙' 둘째 줄):** 아무것도 통과 못 함 → **코어 = 매수보유가 1997~2026 30년 두 독립 구간"
              "(1997~2016 표본 밖 · 2017~2026 세대 4)에서 확인됨.** 문구는 '이 30년 자료에서 매수보유를 버릴 근거 없음'.", ""]
    L += [f"참고 — 기억값 CD91 줄(현금 이자·C4 문턱·초과 SR 모두 CD91)로 같은 산술: 통과 = {', '.join(passed_cd) or '없음'}. 판정에 쓰지 않는다.", "",
          "(3) 배당 민감도(참고): " + " · ".join(
              f"{r['proc']} " + " / ".join(
                  f"{y * 100:.1f}% {_pp(stats[('zero', 'C0', 'oos')]['cagr'] - stats[('zero', r['proc'], 'oos')][f'cagr_div{int(round(y * 1000))}'])} "
                  f"{_ck(r['c3_sens'][y])}" for y in DIV_SENS) for r in rows if r["cash"] == "zero") + ".", ""]
    # --- 숫자 ---
    L += ["## 숫자 (비용·보수 포함, 가격 지수)", "",
          "CAGR = 날 수 기준(252/T — 저장소 `equity_stats`, 세대 4 와 같은 관례), 달력 CAGR = (1 + 누적)^(365.25/일수) − 1(참고). MDD = 일별 경로(초기 자산 1 을 첫 고점으로), "
          "Calmar = CAGR / |MDD|, SR = 월 복리 (초과)수익 평균/표준편차 × √12(현금 0 이면 원 SR). 시장 안 = 평균 보유, 전환/년 = 보유가 바뀐 날 / 년. "
          "CAGR 배당 1.5% = 날별 − 1.5%/252 × (1 − 보유)(C0 불변).", ""]
    for c, clab in CASHES.items():
        L += [f"### {clab}", ""]
        for wk, (lo, hi, lab, _) in WINS.items():
            L += [f"**{lab}**", "", "| 절차 | CAGR | 달력 CAGR | MDD | Calmar | SR | 원 SR | 시장 안 | 전환/년 (횟수) | CAGR 배당 1.5% | 기간 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for nm in PROCS:
                s = stats[(c, nm, wk)]
                L.append(f"| {nm} | {_pct(s['cagr'])} | {_pct(s['cagr_cal'])} | {_pct(s['mdd'])} | {_f(s['calmar'])} | {_f(s['sharpe'])} | "
                         f"{_f(s['sharpe_raw'])} | {s['tim'] * 100:.0f}% | {s['sw_py']:.2f} ({s['n_sw']}) | {_pct(s['cagr_div'])} | {s['start']} ~ {s['end']} |")
            L.append("")
    if g4ref is not None:
        L += ["**세대 4 숫자와 나란히(2017-01 ~ 2026-09, 기술):** 세대 4 = 코스피200 · CD91 현금 · 같은 엔진(`meta_gen4/summary.csv` 'full' 행), "
              "여기 = 코스피 · 기억값 CD91 줄.", "",
              "| 절차 | 세대 4 KOSPI200 CAGR · MDD · Calmar · 초과 SR | 이 러너 KOSPI CAGR · MDD · Calmar · 초과 SR |", "|---|---|---|"]
        for nm in PROCS:
            g = g4ref.get(nm)
            s = stats[("cd91", nm, "g4")]
            gs = "—(세대 4 에 고정 규칙 없음)" if g is None else f"{_pct(g.cagr)} · {_pct(g.mdd)} · {_f(g.calmar)} · {_f(g.sharpe)}"
            L.append(f"| {nm} | {gs} | {_pct(s['cagr'])} · {_pct(s['mdd'])} · {_f(s['calmar'])} · {_f(s['sharpe'])} |")
        L.append("")

    # --- 에피소드 ---
    L += ["## 최악 낙폭 에피소드 (현금 0%)", "",
          "에피소드 = 새 고점 사이 구간. 고점일 = 그 고점 종가 날(창 첫 행 직전 종가 = 시작점), 회복일 = 고점을 처음 다시 넘은 날('미회복' = 그 창 끝까지 못 넘음). 거래일 = 고점 → 저점.", ""]
    edf = pd.DataFrame(episodes)
    for wk in ("oos", "full", "g4"):
        L += [f"### {WINS[wk][2]} — 절차마다 깊은 순 {N_TOP}개", "", "| 절차 | # | 고점 → 저점 | 낙폭 | 거래일 하락 | 회복일 (거래일) |", "|---|---|---|---|---|---|"]
        for nm in PROCS:
            for e in edf[(edf["win"] == wk) & (edf["proc"] == nm)].itertuples():
                rec = "미회복" if e.recover == "미회복" else f"{e.recover} ({int(e.days_recover)})"
                L.append(f"| {nm} | {e.rank} | {e.peak} → {e.trough} | {_pct(e.dd)} | {e.days_down} | {rec} |")
        L += ["", f"**C0 의 깊은 {N_TOP}개 동안 규칙들**(칸 = 고점 종가 → 저점 종가 누적 수익 / 그 안 규칙 자신의 최대 낙폭 / 평균 보유):", "",
              "| C0 고점 → 저점 (회복) | 거래일 | C0 | " + " | ".join(RULES) + " |", "|---|---|---|" + "---|" * len(RULES)]
        for e in c0_eps[wk]:
            cells = [f"{_pct(e[nm]['ret'])} / {_pct(e[nm]['dd'])} / {e[nm]['tim'] * 100:.0f}%" for nm in RULES]
            L.append(f"| {e['peak']} → {e['trough']} ({e['recover']}) | {e['days']} | {_pct(e['dd'])} | " + " | ".join(cells) + " |")
        L.append("")

    # --- 널 ---
    L += ["## 무작위 타이밍 널 (판정 창, 규칙마다 2종 × " + f"{args.nulls}벌)", "",
          "**마르코프**: 판정 창 보유의 π·전환 빈도 s 에 맞춘 2-상태 체인(p10 = s/(2π), p01 = s/(2(1 − π)), 첫 상태 ~ Bernoulli(π)) — **C1m·C4 는 달 단위 체인**"
          "(월말에만 전환, 달 첫날 보유로 π·s 를 재고 날로 펼침), C1f 는 날 단위. **원형 이동**: 실제 보유 열을 오프셋 ∈ [60, T − 60) 거래일만큼 돌림. "
          "같은 엔진·비용·현금. 백분위 = 실제보다 작은 널 비율.", "",
          "| 현금 | 규칙 | 널 | 단위 | π 목표 · 널 실현 | 전환 빈도 목표 · 널 실현(날) | Calmar · p50 · p95 (백분위) | SR · p50 · p95 | MDD · p50 · p95 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, nm = r["cash"], r["proc"]
        s = stats[(c, nm, "oos")]
        for kind, lab in (("markov", "마르코프"), ("circ", "원형 이동")):
            n = nulls[(c, nm)][kind]
            L.append(f"| {c} | {nm} | {lab} | {nulls[(c, nm)]['unit'] if kind == 'markov' else '날'} | {nulls[(c, nm)]['pi']:.3f} · {n['pi_mean']:.3f} | "
                     f"{nulls[(c, nm)]['s']:.4f} · {n['s_mean']:.4f} | "
                     f"{_f(s['calmar'])} · {_f(m4.qs(n['calmar'], 50))} · {_f(m4.qs(n['calmar'], 95))} ({mg1.null_pct(s['calmar'], n['calmar']) * 100:.0f}%) | "
                     f"{_f(s['sharpe'])} · {_f(m4.qs(n['sharpe'], 50))} · {_f(m4.qs(n['sharpe'], 95))} | "
                     f"{_pct(s['mdd'])} · {_pct(m4.qs(n['mdd'], 50))} · {_pct(m4.qs(n['mdd'], 95))} |")
    L += ["", "## 부트스트랩 상세 (C0 대비, 판정 창, 일별 경로 달 블록, 짝지음·공통 난수)", "",
          "| 현금 | 규칙 | 블록 | 달 | P(SR) | ΔSR 5·50·95% | P(Calmar) | ΔCalmar 5·50·95% | P(둘 다) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for key, lab in (("b3", "3개월"), ("b12", "12개월")):
            b = boots[(r["cash"], r["proc"])][key]
            L.append(f"| {r['cash']} | {r['proc']} | {lab} | {b['months']} | {b['p_sharpe']:.3f} | {_q3(b, 'dsharpe')} | {b['p_calmar']:.3f} | "
                     f"{_q3(b, 'dcalmar')} | {b['p_both']:.3f} |")

    # --- 연도별 ---
    L += ["", "## 연도별 수익 (현금 0%, 그 해 보유 비율)", "", "| 해 | " + " | ".join(PROCS) + " |", "|---|" + "---|" * len(PROCS)]
    yz = ydf[ydf["cash"] == "zero"]
    for y in sorted(set(yz["year"])):
        cells = []
        for nm in PROCS:
            q = yz[(yz["proc"] == nm) & (yz["year"] == y)]
            cells.append("" if q.empty else f"{_pct(q['ret'].iloc[0])} ({q['tim'].iloc[0] * 100:.0f}%)")
        L.append(f"| {y} | " + " | ".join(cells) + " |")
    L += ["", "1997 = 12-01 ~ 12-30, 2026 = 01-02 ~ 09-30. 1997-12 ~ 2026-09 한 계열(2017+ 도 이어진 상태 — 위 '2017~2026 창'은 새로 시작한 별도 계열). "
          "가격 지수(배당 제외).", ""]

    # --- 자료 ---
    L += ["## 자료 · 가정", "",
          f"- **지수:** `{INDEX_CSV}` 의 `{CORE_SYM}`({D[0].date()} ~ {D[-1].date()}, {len(D)}일; 같은 파일의 KPI200 1행은 무시). 체결 = 지수 시가(ETF 시가 대용). "
          "네이버 공식값과 겹치는 2,996일 중앙 상대오차 2e-8 · 최대 0.58%(설계 문서).",
          "- **현금:** 판정 = 0%. 기술 줄 CD91 연평균 근사 " + " · ".join(f"{y} {r * 100:.2f}%" for y, r in sorted(CD91_MEM.items()))
          + " — 1997~2016 = **이 러너 작성 때의 기억값**(한국은행 연평균 근사, DB 밖), 2017+ = 세대 4 `CD91`(기억값). 날 금리 = 연 / 252.",
          f"- **비용:** 보유가 바뀐 날 시가에 편도 {COST_1W * 100:.2f}%, 보수 연 {FEE_YR * 100:.2f}% / 252 × 그날 보유. 창 첫날은 현금에서 진입(C0 도 진입 비용 한 번).",
          f"- **배당:** 고정 보정 줄 연 {DIV_YR * 100:.1f}% × 시장 밖 비율(판정 (3)), 민감도 {DIV_SENS[0] * 100:.1f}% · {DIV_SENS[1] * 100:.1f}%(참고).",
          "- **규칙:** C1f = d 종가 > 200일 단순 이동평균(d 포함) → d+1 시가 · C1m = 월말 종가 > 200일선 → 다음 달 첫 거래일 시가부터 그 달 끝까지 "
          "(`ma_month_signal`) · C4 = 월말 12개월 전 월말 대비 수익 > 같은 기간 현금 복리(현금 0 → > 0).",
          "- **부트스트랩:** 판정 창 일별 수익을 달 블록(비원형 이동 블록 3개월 · 참고 12개월)으로 재표본해 이어 붙인 일별 경로 → Calmar(일별 MDD)·SR. "
          "같은 창의 모든 비교에 같은 달 표본(`meta_gen4_run.MonthBoot`·`paired_boot`).",
          f"- **원장:** `{SUB}/TRIALS.jsonl` 새 줄 {v['led']}개. 산출: `{SUB}/` summary·decision·null_random·episodes·c0_episodes·nav·peryear.csv, `equity.png`.", ""]
    if v["fig"]:
        L += ["## 그림", "", f"![자산·낙폭]({SUB}/equity.png)", "",
              "왼쪽 = 판정 창 1997-12 ~ 2016-12, 가운데 = 2017-01 ~ 2026-09(코스피, 새로 시작), 오른쪽 = 전체 1997-12 ~ 2026-09. 위 = 자산(로그), 아래 = 낙폭. "
              "점선 = 2007-01(분리 경계) · 2017-01(표본 밖 끝). 현금 0%.", ""]
    return L


if __name__ == "__main__":
    raise SystemExit(main())
