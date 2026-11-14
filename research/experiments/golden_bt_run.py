"""골든 임박 자리 포지션 최적화 백테스트 실행기 — ``research/logs/golden_bt/PREREG.md``(07f7ce9, 결과 전 고정).

주 시험 2017-10 ~ 2023-12(워크포워드 폴드 2020·2021·2022·2023, 확장 학습 창, 평가 끝 2023-12-28, 비용 2배 판정) →
재현 2009-01 ~ 2016-08(주 시험 마지막 선택 설정 하나 그대로). 엔진 ``research/signals/golden_bt_engine.py``,
메뉴·널·바 ``research/signals/golden_bt_wf.py``. 해석은 아래 RULINGS 에 적는다.

자료(정문만 — CLAUDE.md 데이터 함정):
- 가격 ``daily_bars_adj_v2``(``read_prices_wide``, ``session="raw"`` — 두 창 모두 2026-09-14 애프터마켓 오염 전, 이 표는 regular 를 받지 않는다),
  주 창 2013-06-01 ~ 2023-12-31(q0 라이브러리 3년 예열), 재현 창 2007-11-22 ~ 2016-08-31(golden_hmm 판정 패널과 같음). 2024~ 는 읽지 않는다.
- 정지일(거래량 0·결측) 시가·종가 NaN(ma_quad MQ-2), pre_listing 칸 NaN, 0 이하 가격 NaN, 1,000원 규칙 = 원 종가 없으면 수정 종가.
- 실적 ``eps_ok`` — 주 창 ``earnings``(maquad_calib 규약), 재현 창 ``earnings_dart_doc``(indep30 규약).
- 시총 = 1,000원 규칙 가격 × 주식수 as-of(주 창 ``read_shares(with_asof=True)``, 재현 창 earnings_dart_doc.shares_common@knowledge_date).
- 지수 ``read_index`` 001·101(보고·팩터 회귀), 노출 맞춘 시장 = 시총 가중 시장 일수익(indep30.market_cw).

첫 실제 실행이 곧 판정 — 정본 ``research/logs/golden_bt/`` 로만, ``research/ src/ tests/`` 가 깨끗한 트리에서만, 결과 파일·원장 줄이
있으면 다시 돌지 않는다. 출력은 메모리에서 다 만든 뒤 JSON → MD → TRIALS(마지막). ``--smoke`` 는 모양·건수·시간만 출력한다
(수익·R·CAGR·바 없음 — 엔진 시간 탐침은 가짜 확률로 돌고 결과를 버린다).

사용(실제 실행은 컨트롤러가 한 번):
``set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
uv run --with numba --extra pg --extra research python research/experiments/golden_bt_run.py [--smoke] [--workers 16]``
"""
from __future__ import annotations

import os

# numba·BLAS 를 부르기 전에 — 다중 작업자 스레드 경합·fork 뒤 BLAS 잠금 방지(CLAUDE.md 실행 환경 함정)
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path[:0] = [".", "src"]
from research.experiments import maquad_calib as MQC  # noqa: E402
from research.signals import golden_bt_engine as E  # noqa: E402
from research.signals import golden_bt_wf as W  # noqa: E402
from research.signals import ma_quad as MQ  # noqa: E402
from swing_it.features.korea import MIN_HISTORY, universe_mask  # noqa: E402
from swing_it.units import eok  # noqa: E402

# --- 사전등록 고정값 ------------------------------------------------------------------------------------------
PRICE_TABLE = "daily_bars_adj_v2"
MAIN = {"name": "main", "data": ("2013-06-01", "2023-12-31"), "start": "2017-10-02", "end": "2023-12-28", "eps": "earnings"}
REP = {"name": "rep", "data": ("2007-11-22", "2016-08-31"), "start": "2009-01-02", "end": "2016-08-31", "eps": "earnings_dart_doc"}
FOLD_YEARS = (2020, 2021, 2022, 2023)
HALVES = (("2009-01-02", "2013-01-01"), ("2013-01-01", "2016-09-01"))
K = 10
M_PATHS = 200
SHORT, LONG = 50, 200
GAP_MAX = 0.05
D_MAX = 0.15
ADV_FLOOR = eok(10)            # ADV20 ≥ 10억(백만원 단위 1,000)
MEMBER_HIST_Q0 = 70            # q0 라이브러리 모집단 상장 ≥ 70일(ma_quad·golden_hmm q=0 그대로)
SIGMA_W = 60
POOL_YEARS = 3
POOL_MIN_DATES = 120
LU_CUT = "2015-06-15"          # 가격제한폭 ±15% → ±30%
LU = (1.14, 1.29)              # 상한가 시가 판정 배수(autoresearch 정정 30-1 M1 · leader21 LU)
INDEX_MIN_COVER = 0.95
SEED = 20261115
DSR_N = {"golden_bt_menu": 48, "autoresearch_unique_train_sha": 280, "ga_generations_upper": 32, "ma_cross_family_prior_gates": 2}
N_TRIALS = int(sum(DSR_N.values()))
SMOKE_PROB_MONTHS = 3

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "research" / "logs" / "golden_bt"
SCRATCH = Path.home() / ".cache" / "cc-swing" / "scratch" / "golden_bt"
OUT_JSON, OUT_MD, OUT_TRIALS = "backtest.json", "RESULTS.md", "TRIALS.jsonl"
TRIAL_NAME = "golden_bt"

RULINGS = {
    "session": "read_prices_wide(table=daily_bars_adj_v2, session='raw') — 두 창 모두 2026-09-14 애프터마켓 오염 전. 2024~ 는 요청하지 않는다"
               "(주 창 끝 2023-12-31, 주식수·실적 원천 행도 창 끝 뒤 as-of 는 쓰지 않는다). 2024~ 보고 열은 내지 않는다(조회하지 않음).",
    "suspension": "정지일(거래량 0·결측) 시가·종가·원 종가 NaN(MQ-2) — 그날 체결 불가, 판정 없음. 매도 대기 중 정지면 다음 체결 가능일 시가. "
                  "확률·사전 후보의 MA 는 창에 NaN 이 있으면 NaN 그대로(GB-1 에서도 바꾸지 않음).",
    "gb1_fallback": "정정 GB-1(컨트롤러, 결과 전): 골든 판정은 정지일을 건너뛴 MA50·MA200(유효 종가 마지막 50·200 개 평균, E.valid_ma)으로 한다. "
                    "보유 중 그날 판정 신호가 NaN 이면 — 골든 전(P_g NaN): 보유, 손절만. 골든 뒤(dead_p 의 P_d 또는 ma200 설정의 MA200 이 NaN): "
                    "종가 < 건너뛴 MA200 → 다음 시가(사유 ma200_fb). 신호가 유한하면 원래 규칙. 거래별 NaN 판정일 수(nan_dec)·대체 청산 수를 보고한다. "
                    "ma200 설정에도 같은 대체를 쓴다(그 설정의 신호가 정규 MA200 이라 같은 결함). 널 ② 의 골든(추가 매수 정지)도 건너뛴 MA.",
    "pool": "사전 후보 = 유니버스(위생·ADV20 ≥ 10억·상장 252일) ∧ EPS↑ ∧ 0 < 1 − MA50/MA200 ≤ 0.05 ∧ 0 < d = 1 − MA50/종가 ≤ 0.15, 판정일 창 안.",
    "probs": "P_g·P_d = golden_hmm q0 절차 그대로(poly_forecast(y,1,10,0)·σ60·month_pool 3년·원점 ≥ 120·u ≤ t_m−k·모집단 상장 ≥ 70·"
             "draw_index(code, 달, 0, 0, 10) M=200) → ma_quad.cross_probs(k=10, 50/200), P_d 는 dead=True(같은 경로·같은 색인, 부등호만 반대). "
             "계산 칸 = 창 안에 사전 후보가 한 번이라도 있었던 종목의 첫 후보일 ~ 창 끝 중 종가 유한 칸(보유 중 판정 전부 포함). 나머지 NaN.",
    "fill": "t 종가 판정 → t+1 시가. 상한가 시가 = 시가 ≥ 전일 종가 × (2015-06-15 전 1.14, 뒤 1.29). 시가 ≤ 손절선이면 사지 않는다. "
            "못 산 주문은 버린다(그날 다른 후보로 채우지 않는다).",
    "sizing_d": "비중의 d 는 체결 시가 기준 d_fill = 1 − 손절선/시가(손실 한도 0.25% 를 실제 진입가로 맞춘다). 사전 후보의 d ≤ 0.15 는 판정일 종가 기준, "
                "d_fill 상한으로 따로 거르지 않는다(등록에 없는 거름을 더하지 않는다).",
    "prob_topup": "prob: 기준 = min(5%, 0.25%/d_fill) 진입 때 고정. 골든 전 판정일마다 목표 = 기준 × P_g(t), 현재 비중 = 보유 가치/그날 종가 자산, "
                  "목표 > 현재면 차이를 다음 시가에 추가(진입과 같은 체결 조건, 전날 종가 자산 기준 금액, 현금 한도). 줄이지 않는다. "
                  "골든 뒤에는 추가하지 않는다(P_g 가 사실상 1 이 되는 구간 — 보수 쪽). 폴드 진입 마감 뒤에도 보유 포지션의 추가는 규칙대로.",
    "golden": "골든 = 진입 뒤 판정일 종가 기준 MA50 > MA200(엄격)이 처음 된 날. 그날 판정부터 골든 뒤 규칙(dead_p·ma200)이 적용되고 "
              "예측 붕괴·추가 매수는 멈춘다. 손절은 앞뒤 모두.",
    "exit_priority": "같은 날 사유가 겹치면 손절 > (골든 전) 붕괴 / (골든 뒤) 데드·MA200 — 붕괴와 데드는 골든 전후로 배타.",
    "nan_prob": "보유 중 그날 P 가 NaN 이면 붕괴·데드 판정을 건너뛴다(청산 사유 아님).",
    "slots": "슬롯 20 = 보유(매도 대기 포함) 수. 청산으로 빈 슬롯은 체결일 종가 판정부터 쓴다. 순위 P_g 내림차순, 같으면 종목 행 순. "
             "같은 종목 재진입은 청산 체결일 종가 판정부터(다음 시가).",
    "delist": "마지막 유한 종가 뒤(폐지·자료 끝)에는 그 다음 거래일에 마지막 가격으로 청산(사유 delist, 그날 비용).",
    "cost": "korea.cost_rt(청산일, 시장) × 비용 배수(판정 2배)를 매도 대금에서 뺀다 — 재현 창도 같은 함수(그 시기 세율). 시장 = 현재 상장 메타(PIT 아님).",
    "R": "R = 손익 ÷ (0.25% × 첫 매수 금액을 정한 전날 종가 자산). 추가 매수분 손익도 같은 분모.",
    "train_score": "학습 점수 = 학습 창 [2017-10-02, Y−1 말] 만 시뮬(빈 장부): 끝난 거래 + Y−1 마지막 종가 평가 보유분(비용 2배 포함)의 R 합. "
                   "Y 가격은 엔진에 들어가지 않는다.",
    "oos": "폴드 Y 시험 = 빈 장부, Y 첫 거래일 체결부터 Y 말 체결까지 새 진입, 보유는 진입 설정대로 2023-12-28 종가까지(끝 보유분 비용 포함 평가). "
           "폴드 밖 거래 = 네 폴드 거래를 이어 붙인 것(체결 수·R·바 ①②③⑥).",
    "oos_equity": "정정 GB-1(컨트롤러, 결과 전): 자산 기반 바(④ 팩터 알파·⑤ 연복리·⑦ DSR)와 월 팩터 회귀의 폴드 밖 자산 = 모든 폴드 밖 거래의 "
                  "(비중 × 그날 수익) 합(거래별 그날 손익 ÷ 첫 매수 때 자산, 청산일 비용 포함)을 2020 첫 거래일 ~ 2023-12-28 에 복리로 이은 것 "
                  "(W.oof_series) — 폴드 Y 진입은 Y+1 에도 자기 설정대로 기여한다. 총노출 = Σ 보유 가치 ÷ 첫 매수 때 자산(폴드가 겹치면 1 을 넘을 수 있다), "
                  "평균·최대를 보고하고 노출 맞춘 시장도 이 노출로. 재현은 단일 시뮬 자산 그대로.",
    "neighbors": "이웃 = 자기 + 한 축만 다른 설정: θ_in 은 순서 축 한 칸(0.3↔0.5↔0.7↔0.9), θ_out·sizing·exit(dead_p 0.5·dead_p 0.7·ma200) 은 "
                 "범주 축이라 다른 값 전부. 선택 = 이웃 학습 점수 중앙값 최대, 동률이면 메뉴 순서 앞.",
    "null1": "널 ① = 300 회, 회마다 폴드마다 48 중 무작위 설정(복원, 시드 고정) 의 폴드 밖 R 합(비용 2배)을 더한다. 재현은 회마다 설정 하나.",
    "null2": "널 ② = 300 회, 선택 설정의 체결된 진입마다 같은 판정일 같은 풀(사전 후보 ∧ P_g 유한 — P_g 문턱 없음)에서 무작위 종목(그날 안 비복원, "
             "체결 불가면 다음 후보, 풀이 모자라면 모자란 수 기록), 같은 체결·손절(판정일 MA50 고정)·비중(그 폴드 sizing, prob 는 골든 전 추가 포함) 규칙. "
             "보유 기간 = 선택 설정 폴드 밖 거래 중 손절 아닌 청산(붕괴·데드·MA200·폐지·끝)의 체결 간 거래일 수 분포에서 복원 추출, 그 날짜 시가 매도"
             "(정지면 미룸), 손절이 먼저면 손절. prob 에 무작위 종목 P_g = 0 이면 비중 0 체결(R 0, 진입 수에 셈). 거래 단위(자산 1 고정, 슬롯·현금 제약 없음 — 슬롯·현금 제약은 주 전략만 깎으므로 보수 쪽).",
    "bar2": "② 진입 연월 군집 복원 추출 2,000 회, 거래 R 평균의 2.5% 분위 > 0(ma_quad_calib.bss_ci 규약).",
    "bar3": "③ 종목별 R 합 상위 5 종목의 거래를 빼고 R 평균 > 0(거래 제거 — 재시뮬 아님).",
    "bar4": "④ 월 수익(첫 달 포함) ~ KOSPI + KOSDAQ + MOM(12-1) + SMB + LV + REV(1개월 반전, 직전 달 수익 하위 30% − 상위 30%) — "
            "팩터는 autoresearch.prepare.factor_months 를 그대로 쓰고 REV 만 같은 규약으로 더한다(prepare.py 는 MANIFEST sha 로 보호돼 고치지 않는다). "
            "'OLS·NW lag 3' 은 둘 다 t > 2 로 읽는다(보수 쪽).",
    "bar5": "⑤ 폴드 밖 자산 연복리 > 노출 맞춘 시장(전날 노출 × 시총 가중 시장 일수익 복리, 같은 날짜). 보고 열 KOSPI 보유 연복리.",
    "bar7": f"⑦ DSR = krx_quant_core deflated_sharpe_from_sample(폴드 밖 월 수익, N) 의 prob_deflated_sharpe > 0.9. N = {N_TRIALS} = "
            + " + ".join(f"{k} {v}" for k, v in DSR_N.items()) + " (GA 는 META_EVOLUTION 세대 28 + 30·32 단계의 상한 32). 월 수익 — 거래 R 은 겹침 상관을 무시해 낙관.",
    "void": "체결(폴드 밖 거래) < 100 → VOID(주 시험·재현 각각).",
    "rep": "재현 = 폴드 2023 의 선택 설정 하나를 2009-01-02 ~ 2016-08-31 에 그대로(빈 장부, 끝까지 진입, 2016-08-31 종가 평가). 바 ①②④⑤ + 두 반기"
           "(진입일 [2009-01-02, 2013-01-01)·[2013-01-01, 2016-09-01)) R 평균 > 0, VOID 같음. q0 라이브러리는 패널 시작 2007-11-22 에서 3년을 다 못 채운다"
           "(golden_hmm RULINGS q0_library 와 같음).",
    "cap": "시총은 SMB 팩터와 시장 가중에만. 주 창 shares_outstanding_history(asof = coalesce(knowledge_date, date)), 재현 창 earnings_dart_doc "
           "(indep30 규약) — 원 종가 × 주식수라 분할 직후 수개월 어긋날 수 있다(보고 품질, 판정 거래 R 과 무관).",
    "index": "KOSPI(001)·KOSDAQ(101)은 read_index(proxy=선언) — 창 안 거래일 95% 이상에 값이 없으면 멈춘다.",
}

_G: dict = {}
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# =============================================================================================== 자료

def _shares(con, days, codes, eps_source: str, data_end: str) -> np.ndarray:
    from swing_it.storage import read_earnings, read_shares

    end = pd.Timestamp(data_end)
    if eps_source == "earnings_dart_doc":
        sh = read_earnings(con, cols=("code", "period", "knowledge_date", "shares_common"), all_versions=True, parse_dates=True,
                           table="earnings_dart_doc")
        sh = sh[(sh["shares_common"] > 0) & (sh["shares_common"] <= 1e10) & (sh["knowledge_date"] <= end)]
        sh = sh.sort_values(["knowledge_date", "period"], kind="stable")
        shw = sh.pivot_table(index="knowledge_date", columns="code", values="shares_common", aggfunc="last")
    else:
        sh = read_shares(con, with_asof=True, parse_dates=True)
        sh = sh[(sh["shares_outstanding"] <= 1e10) & (sh["asof"] <= end)].sort_values("asof", kind="stable")
        shw = sh.pivot_table(index="asof", columns="code", values="shares_outstanding", aggfunc="last")
    shw.columns = [str(c) for c in shw.columns]
    shw = shw.reindex(index=shw.index.union(days)).sort_index().ffill().reindex(index=days, columns=codes)
    return np.array(shw.to_numpy(dtype=float).T, order="C")


def _index(con, code: str, data: tuple, days) -> pd.Series:
    from swing_it.storage import read_index

    df = read_index(con, code, cols=("date", "close"), start=data[0], end=data[1], with_level=True, order_by_date=True, parse_dates=True,
                    proxy="golden_bt: 공식 지수 행이 있는 날만 쓴다 — 창 안 커버리지 ≥ 95% 검사(RULINGS index)")
    return df.set_index("date")["level"].astype(float).reindex(pd.DatetimeIndex(days))


def load(con, win: dict) -> dict:
    """창 ``win`` 의 시가·종가·거래대금·1,000원 규칙 가격·시총·지수 (n_codes, n_days) + 메타. 지표 없음."""
    from swing_it.features.korea import load_listing_meta
    from swing_it.storage import read_prices, read_prices_wide

    data = win["data"]
    p = read_prices_wide(con, cols=("open", "close", "trade_value", "raw_close", "volume"), table=PRICE_TABLE, start=data[0], end=data[1],
                         session="raw")
    days = pd.DatetimeIndex(p["close"].index)
    codes = pd.Index([str(c) for c in p["close"].columns])

    def arr(c):
        return np.array(p[c].to_numpy(dtype=float).T, dtype=np.float64, order="C", copy=True)

    O, C, TV, RC, V = arr("open"), arr("close"), arr("trade_value"), arr("raw_close"), arr("volume")  # noqa: E741
    p = None
    C_src = C.copy()
    n_susp = MQC.mask_suspensions(C, V, RC, O)              # MQ-2: 정지일 시가·종가 NaN — 맨 먼저
    del V
    cs = read_prices(con, cols=("code", "date", "close_src"), table=PRICE_TABLE, start=data[0], end=data[1], session="raw", parse_dates=True)
    pl = cs.loc[cs["close_src"] == "pre_listing", ["code", "date"]]
    del cs
    di = days.get_indexer(pd.to_datetime(pl["date"]))
    cj = codes.get_indexer(pl["code"].astype(str))
    if (di < 0).any() or (cj < 0).any():
        raise RuntimeError("pre_listing 행이 와이드 패널 격자 밖이다")
    if np.isfinite(C_src[cj, di]).any():
        raise RuntimeError(f"pre_listing 칸 {int(np.isfinite(C_src[cj, di]).sum())} 개에 종가가 있다 — 자료가 바뀌었다, 멈춘다")
    del C_src
    for a in (C, TV, RC, O):
        a[cj, di] = np.nan
    n_nonpos = int((C <= 0).sum())
    C[C <= 0] = np.nan
    with np.errstate(invalid="ignore"):
        n_open_bad = int((np.isfinite(C) & ~(O > 0)).sum())
        O[~(O > 0) | ~np.isfinite(C)] = np.nan
    n_rc_fill = int((~np.isfinite(RC) & np.isfinite(C)).sum())
    PX = np.where(np.isfinite(RC), RC, C)
    meta = load_listing_meta(con)
    cap = PX * _shares(con, days, list(codes), win["eps"], data[1])
    kx, kq = _index(con, "001", data, days), _index(con, "101", data, days)
    info = {"table": PRICE_TABLE, "start": data[0], "end": data[1], "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
            "n_codes": len(codes), "n_days": len(days), "n_pre_listing_cells": len(pl), "n_suspension_cells_nan": n_susp,
            "n_close_nonpositive": n_nonpos, "n_open_missing_on_trading_day": n_open_bad, "n_price_rule_from_adj_close": n_rc_fill,
            "cap_finite_share": float(np.isfinite(cap[np.isfinite(C)]).mean()) if np.isfinite(C).any() else 0.0}
    return {"days": days, "codes": list(codes), "O": O, "C": C, "raw_close": PX, "tv": TV, "cap": cap, "meta": meta, "kx": kx, "kq": kq,
            "info": info, "win": win}


def check_window(days, win: dict) -> None:
    d = pd.DatetimeIndex(days)
    if d.min() < pd.Timestamp(win["data"][0]) or d.max() > pd.Timestamp(win["data"][1]) or d.max() >= pd.Timestamp("2024-01-01"):
        raise RuntimeError(f"{win['name']} 패널 {d.min().date()} ~ {d.max().date()} 이 {win['data']} 밖(또는 2024~)")


def _universe(X: dict, **kw) -> np.ndarray:
    idx = pd.Index(X["codes"])
    days = pd.DatetimeIndex(X["days"])
    u = universe_mask(pd.DataFrame(X["C"].T, index=days, columns=idx), pd.DataFrame(np.asarray(X["tv"], float).T, index=days, columns=idx),
                      X["meta"], raw_close=pd.DataFrame(np.asarray(X["raw_close"], float).T, index=days, columns=idx),
                      adv_floor=ADV_FLOOR, **kw)
    return np.array(u.to_numpy(dtype=bool).T, order="C", copy=True)


def sim_bounds(days, win: dict) -> tuple[int, int]:
    """체결 첫 행(≥ 1 — 그 전날이 첫 판정일)·마지막 행."""
    s, e = W.day_index(days, win["start"], "first"), W.day_index(days, win["end"], "last")
    if s < 1 or e <= s:
        raise RuntimeError(f"{win['name']}: 시뮬 구간 행 [{s}, {e}] 가 잘못")
    return s, e


def lu_array(days) -> np.ndarray:
    return np.where(pd.DatetimeIndex(days) < pd.Timestamp(LU_CUT), LU[0], LU[1])


def cost_table(days) -> np.ndarray:
    from swing_it.features.korea import cost_rt

    return np.array([[cost_rt(d, m) for d in pd.DatetimeIndex(days)] for m in ("KOSPI", "KOSDAQ")], dtype=np.float64)


def prepare(X: dict, eps_fn) -> dict:
    """상태·유니버스·q0 모집단·σ·MA50/200·사전 후보 풀(EPS↑ 는 기초 칸에만 ``eps_fn(ci, s)``)·비용·상한·시장."""
    X = dict(X)
    days = pd.DatetimeIndex(X["days"])
    C = np.ascontiguousarray(X["C"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(C > 0, np.log(C), np.nan)
    sigma = MQ.rolling_sigma(y, SIGMA_W)
    ma50, ma200 = MQ._rolling_mean(C, SHORT), MQ._rolling_mean(C, LONG)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = 1.0 - ma50 / ma200
        d = 1.0 - ma50 / C
    U = _universe(X)
    member = _universe(X, min_history=MEMBER_HIST_Q0)
    s0, e0 = sim_bounds(days, X["win"])
    in_win = np.zeros(len(days), bool)
    in_win[s0 - 1 : e0] = True                           # 판정일 = 체결 첫날 전날 ~ 마지막 날 전날
    with np.errstate(invalid="ignore"):
        base = U & (gap > 0) & (gap <= GAP_MAX) & (d > 0) & (d <= D_MAX) & in_win[None, :]
    ci, s = np.nonzero(base)
    eps = np.zeros_like(U)
    if len(ci):
        eps[ci, s] = np.asarray(eps_fn(ci, s), dtype=bool)
    mk = X["meta"]["market"].reindex(pd.Index(X["codes"]))
    X.update(days=days, C=C, y=y, sigma=sigma, ma50=ma50, ma200=ma200, ma50v=E.valid_ma(C, SHORT), ma200v=E.valid_ma(C, LONG), gap=gap, U=U, member=member, base=base, pool=base & eps,
             mkt=np.asarray(mk.eq("KOSDAQ").to_numpy(), dtype=np.int64), cost=cost_table(days), lu=lu_array(days),
             last_valid=E.last_valid_index(C), bounds=(s0, e0))
    return X


# =============================================================================================== 확률 패널

def prob_cells(X: dict) -> dict:
    """계산할 칸: 판정일 창 안에 사전 후보가 있었던 종목의 첫 후보일 ~ 시뮬 끝 중 종가 유한. {달 첫 행: (ci, s)}."""
    pool = X["pool"]
    s0, e0 = X["bounds"]
    days = X["days"]
    has = pool.any(axis=1)
    first = np.where(has, np.argmax(pool, axis=1), -1)
    T = pool.shape[1]
    cols = np.arange(T)
    want = has[:, None] & (cols[None, :] >= first[:, None]) & (cols[None, :] <= e0) & np.isfinite(X["C"])
    s, ci = np.nonzero(want.T)                                 # 날짜 순
    ms = MQ.month_starts(days)
    tm = ms[np.searchsorted(ms, s, side="right") - 1]
    out = {}
    for t_m in np.unique(tm):
        sel = tm == t_m
        out[int(t_m)] = (ci[sel].astype(np.int64), s[sel].astype(np.int64))
    return out


def _month_probs(X: dict, fc, z, t_m: int, ci, s) -> tuple:
    """한 달 칸들의 (P_g, P_d, 풀 크기) — golden_hmm q0_probs 와 같은 풀·같은 색인."""
    days, codes = X["days"], X["codes"]
    got = MQ.month_pool(days, z, X["member"], int(t_m), K, years=POOL_YEARS, min_dates=POOL_MIN_DATES)
    if got is None or len(got) == 0:
        nan = np.full(len(ci), np.nan)
        return nan, nan.copy(), 0
    draws = {int(c): MQ.draw_index(codes[c], days[t_m], 0, 0, K, len(got), M_PATHS) for c in np.unique(ci)}
    idx = np.stack([draws[int(c)] for c in ci])
    pg = MQ.cross_probs(X["C"], fc, X["sigma"], got, idx, ci, s, K, SHORT, LONG)
    pdv = MQ.cross_probs(X["C"], fc, X["sigma"], got, idx, ci, s, K, SHORT, LONG, dead=True)
    return pg, pdv, int(len(got))


def _prob_task(t_m):
    ci, s = _G["cells"][t_m]
    t = time.time()
    pg, pdv, P = _month_probs(_G["X"], _G["fc"], _G["z"], t_m, ci, s)
    return t_m, pg, pdv, P, time.time() - t


def _warmup() -> None:
    """numba 컴파일을 부모에서 한 번 — fork 자식이 물려받는다."""
    C = np.exp(np.log(np.linspace(100, 130, 300)))[None, :].repeat(2, axis=0)
    for dead in (False, True):
        MQ.cross_kernel_events(C, [0], [250], np.zeros((1, 3)), np.zeros(1), np.zeros((2, 3)), np.zeros((1, 2), np.int64), 3, 5, 20, dead=dead)


def prob_panels(X: dict, workers: int, months: "list | None" = None) -> dict:
    """P_g·P_d (n_codes, n_days) float32(계산 안 한 칸 NaN) + 달별 풀 크기·시간. ``months`` = 일부 달만(스모크)."""
    cells = prob_cells(X)
    todo = sorted(cells) if months is None else [m for m in months if m in cells]
    fc = MQ.poly_forecast(X["y"], 1, K, 0)
    z = MQ.forecast_z(X["y"], fc, X["sigma"])
    PG = np.full(X["C"].shape, np.nan, np.float32)
    PD = np.full(X["C"].shape, np.nan, np.float32)
    info = {"n_months": len(todo), "n_cells": int(sum(len(cells[m][0]) for m in todo)), "pool_P": {}, "secs": {}}
    _G.update(X=X, fc=fc, z=z, cells=cells)
    _warmup()
    try:
        if workers > 1 and len(todo) > 1:
            with mp.get_context("fork").Pool(min(workers, len(todo))) as pool:
                res = list(pool.imap_unordered(_prob_task, todo))
        else:
            res = [_prob_task(m) for m in todo]
    finally:
        _G.clear()
    for t_m, pg, pdv, P, secs in res:
        ci, s = cells[t_m]
        PG[ci, s] = pg
        PD[ci, s] = pdv
        key = f"{X['days'][t_m]:%Y-%m}"
        info["pool_P"][key] = P
        info["secs"][key] = round(secs, 2)
    info["n_nopool_months"] = int(sum(1 for v in info["pool_P"].values() if v == 0))
    info["n_finite_pg"] = int(np.isfinite(PG).sum())
    info["n_pool_cells"] = int(X["pool"].sum())
    info["n_pool_finite_pg"] = int((X["pool"] & np.isfinite(PG)).sum())
    return {"PG": PG, "PD": PD, "info": info}


def sim_data(X: dict, P: dict) -> dict:
    keys = ("days", "codes", "O", "C", "ma50", "ma200", "ma50v", "ma200v", "pool", "lu", "cost", "mkt", "last_valid")
    return {**{k: X[k] for k in keys}, "PG": P["PG"], "PD": P["PD"]}


# =============================================================================================== 병렬 시뮬

def _sim_task(a):
    tag, i, f, cfg, cm, s, e, ee = a
    t = time.time()
    tr, eq = E.simulate(_G["D"], cfg, cm, s, e, ee)
    return tag, i, f, cm, tr, eq, time.time() - t


def run_sims(D: dict, tasks: list, workers: int) -> list:
    _G["D"] = D
    try:
        if workers > 1 and len(tasks) > 1:
            with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
                return list(pool.imap_unordered(_sim_task, tasks, chunksize=1))
        return [_sim_task(a) for a in tasks]
    finally:
        _G.clear()


def _null2_task(a):
    tag, r = a
    rng = np.random.default_rng([SEED, 2, 0 if tag == "main" else 1, r])
    return tag, r, W.null2_rep(_G["D"][tag], _G["plan"][tag], _G["holds"][tag], rng)


def run_null2(Ds: dict, plans: dict, holds: dict, n: int, workers: int) -> dict:
    tasks = [(tag, r) for tag in plans for r in range(n)]
    _G.update(D=Ds, plan=plans, holds=holds)
    try:
        if workers > 1:
            with mp.get_context("fork").Pool(workers) as pool:
                res = list(pool.imap_unordered(_null2_task, tasks, chunksize=4))
        else:
            res = [_null2_task(a) for a in tasks]
    finally:
        _G.clear()
    out = {tag: [None] * n for tag in plans}
    for tag, r, v in res:
        out[tag][r] = v
    return out


# =============================================================================================== 워크포워드·재현

def walk_forward(D: dict, workers: int) -> dict:
    """48 × 4 학습 점수 → 고원 선택 → 48 × 4 폴드 밖(비용 2배; 널 ①) + 선택 설정 비용 1배. 지표 계산은 assemble 에서."""
    M = W.menu()
    s0, e0 = sim_bounds(D["days"], MAIN)
    folds = W.fold_windows(D["days"], FOLD_YEARS, MAIN["start"], MAIN["end"])
    if folds[0]["train"][0] != s0 or folds[-1]["test"][1] != e0:
        raise RuntimeError("폴드 경계가 시뮬 구간과 다르다")
    tasks = [("train", i, f, M[i], W.COST_JUDGE, fo["train"][0], fo["train"][1], None) for i in range(len(M)) for f, fo in enumerate(folds)]
    tasks += [("oos", i, f, M[i], W.COST_JUDGE, fo["test"][0], fo["test"][1], fo["entry_end"]) for i in range(len(M)) for f, fo in enumerate(folds)]
    t = time.time()
    res = run_sims(D, tasks, workers)
    log(f"주 시험 시뮬 {len(tasks)} 개 · {time.time() - t:.0f}s")
    score = np.zeros((len(M), len(folds)))
    rs = np.zeros((len(M), len(folds)))
    oos = {}
    for tag, i, f, _, tr, eq, _ in res:
        if tag == "train":
            score[i, f] = W.r_sum(tr)
        else:
            rs[i, f] = W.r_sum(tr)
            oos[(i, f)] = (tr, eq)
    sel = [W.select(score[:, f]) for f in range(len(folds))]
    c1 = run_sims(D, [("oos1", sel[f], f, M[sel[f]], 1.0, fo["test"][0], fo["test"][1], fo["entry_end"]) for f, fo in enumerate(folds)], workers)
    c1 = {f: (tr, eq) for _, _, f, _, tr, eq, _ in c1}
    return {"menu": M, "folds": folds, "score": score, "rsum": rs, "sel": sel, "oos": oos, "oos_cost1": c1,
            "nbr_median": [W.neighbor_medians(score[:, f]).tolist() for f in range(len(folds))]}


def oos_selected(wf: dict, D: dict, cost1: bool = False) -> tuple:
    """선택 설정의 폴드 밖 거래(열 fold·cfg·sizing 추가)와 정정 GB-1 폴드 밖 연속 자산(거래 합 — W.oof_series)."""
    trs = []
    for f, fo in enumerate(wf["folds"]):
        i = wf["sel"][f]
        tr, _ = wf["oos_cost1"][f] if cost1 else wf["oos"][(i, f)]
        trs.append(tr.assign(fold=fo["year"], cfg=i, sizing=wf["menu"][i]["sizing"]))
    tr = pd.concat(trs, ignore_index=True)
    return tr, W.oof_series(D, tr, wf["folds"][0]["test"][0], wf["folds"][-1]["test"][1])


def replicate(D: dict, cfg_i: int, workers: int) -> dict:
    M = W.menu()
    s, e = sim_bounds(D["days"], REP)
    tasks = [("rep", i, 0, M[i], W.COST_JUDGE, s, e, None) for i in range(len(M))] + [("rep1", cfg_i, 0, M[cfg_i], 1.0, s, e, None)]
    t = time.time()
    res = run_sims(D, tasks, workers)
    log(f"재현 시뮬 {len(tasks)} 개 · {time.time() - t:.0f}s")
    rs = np.zeros(len(M))
    out = {"cfg": cfg_i, "bounds": (s, e)}
    for tag, i, _, _, tr, eq, _ in res:
        if tag == "rep":
            rs[i] = W.r_sum(tr)
            if i == cfg_i:
                out["trades"], out["eq"] = tr.assign(cfg=i, sizing=M[i]["sizing"]), eq
        else:
            out["trades_cost1"], out["eq_cost1"] = tr, eq
    out["rsum_all"] = rs
    return out


def factor_data(X: dict) -> dict:
    return {"days": X["days"], "C": X["C"], "cap": X["cap"], "uni": X["U"]}


def market_cw(X: dict) -> pd.Series:
    from research.experiments.indep30 import market_cw as _mcw

    return pd.Series(_mcw(X["C"], X["cap"]), index=X["days"])


def _index_cover(X: dict, s: int, e: int) -> dict:
    out = {}
    for k in ("kx", "kq"):
        v = X[k].iloc[s : e + 1]
        out[k] = float(v.notna().mean())
    return out


def require_index(X: dict, s: int, e: int) -> None:
    cov = _index_cover(X, s, e)
    bad = {k: v for k, v in cov.items() if v < INDEX_MIN_COVER}
    if bad:
        raise RuntimeError(f"{X['win']['name']}: 지수 커버리지 {bad} < {INDEX_MIN_COVER} — 멈춘다")


def assemble(X1: dict, D1: dict, wf: dict, X2: dict, D2: dict, rep: dict, n2: dict, P1info: dict, P2info: dict) -> dict:
    """주 시험·재현 바와 보고 열. 널 ② 결과 ``n2`` = {main|rep: [회별 dict]}."""
    s1, e1 = sim_bounds(D1["days"], MAIN)
    tr, eq = oos_selected(wf, D1)
    tr1, eq1 = oos_selected(wf, D1, cost1=True)
    cw1 = market_cw(X1)
    mk_eq = W.expo_market(eq, cw1)
    m = W.monthly(eq, X1["kx"], X1["kq"])
    F1 = W.factor_months_rev(factor_data(X1))
    null1 = W.null1_folds(wf["rsum"], W.N_NULL, seed=SEED)
    null2 = np.array([r["R_sum"] for r in n2["main"]])
    fold_rs = [W.r_sum(tr[tr["fold"] == y]) for y in FOLD_YEARS]
    bars = W.bars_main(tr, fold_rs, eq, mk_eq, m, F1, null1, null2, N_TRIALS)
    kx = X1["kx"].reindex(eq.index).ffill()
    main = {
        "bars": bars,
        "selection": [{"year": fo["year"], "cfg_i": wf["sel"][f], "cfg": wf["menu"][wf["sel"][f]],
                       "train_window": [str(D1["days"][fo["train"][0]].date()), str(D1["days"][fo["train"][1]].date())],
                       "train_R_sum_selected": float(wf["score"][wf["sel"][f], f]),
                       "train_nbr_median_selected": float(wf["nbr_median"][f][wf["sel"][f]])} for f, fo in enumerate(wf["folds"])],
        "train_scores": wf["score"].tolist(), "oos_rsum_by_cfg_fold": wf["rsum"].tolist(),
        "report": W.report(D1, tr, e1),
        "cost": {"x2": {"R_sum": W.r_sum(tr), "n": int(len(tr)), "cagr": W.cagr(eq)},
                 "x1": {"R_sum": W.r_sum(tr1), "n": int(len(tr1)), "cagr": W.cagr(eq1)}},
        "exposure_mean": float(np.mean(eq.attrs["expo"])), "exposure_max": float(np.max(eq.attrs["expo"])),
        "kospi_cagr": W.cagr(kx.dropna()) if kx.notna().sum() > 1 else float("nan"),
        "yearly_return_x2": W.yearly_returns(eq), "fold_R_sum": dict(zip(FOLD_YEARS, fold_rs, strict=True)),
        "null1": _qs(null1), "null2": {**_qs(null2), "n_short_total": int(sum(r["n_short"] for r in n2["main"])),
                                       "n_fill_mean": float(np.mean([r["n_fill"] for r in n2["main"]]))},
        "null2_holds_src": W.null2_holds(tr)[1],
        "skips": _skips(wf["oos"], wf["sel"]),
        "index_cover": _index_cover(X1, s1, e1), "probs": P1info, "eval": [str(D1["days"][s1].date()), str(D1["days"][e1].date())],
    }
    s2, e2 = rep["bounds"]
    cw2 = market_cw(X2)
    eq2 = rep["eq"]
    mk2 = W.expo_market(eq2, cw2)
    m2 = W.monthly(eq2, X2["kx"], X2["kq"])
    F2 = W.factor_months_rev(factor_data(X2))
    rnull1 = W.null1_folds(rep["rsum_all"][:, None], W.N_NULL, seed=SEED + 1)
    rnull2 = np.array([r["R_sum"] for r in n2["rep"]])
    rbars = W.bars_rep(rep["trades"], eq2, mk2, m2, F2, rnull1, rnull2, HALVES)
    kx2 = X2["kx"].reindex(eq2.index).ffill()
    repo = {
        "bars": rbars, "cfg_i": rep["cfg"], "cfg": W.menu()[rep["cfg"]], "report": W.report(D2, rep["trades"], e2),
        "cost": {"x2": {"R_sum": W.r_sum(rep["trades"]), "n": int(len(rep["trades"])), "cagr": W.cagr(eq2)},
                 "x1": {"R_sum": W.r_sum(rep["trades_cost1"]), "n": int(len(rep["trades_cost1"])), "cagr": W.cagr(rep["eq_cost1"])}},
        "exposure_mean": float(np.mean(eq2.attrs["expo"])), "exposure_max": float(np.max(eq2.attrs["expo"])), "kospi_cagr": W.cagr(kx2.dropna()) if kx2.notna().sum() > 1 else float("nan"),
        "yearly_return_x2": W.yearly_returns(eq2), "rsum_by_cfg": rep["rsum_all"].tolist(),
        "null1": _qs(rnull1), "null2": {**_qs(rnull2), "n_short_total": int(sum(r["n_short"] for r in n2["rep"]))},
        "null2_holds_src": W.null2_holds(rep["trades"])[1],
        "skips": eq2.attrs["skips"], "index_cover": _index_cover(X2, s2, e2), "probs": P2info,
        "eval": [str(D2["days"][s2].date()), str(D2["days"][e2].date())],
    }
    verdict = {"main_pass": bool(bars["pass"]), "main_void": bool(bars["void"]), "rep_pass": bool(rbars["pass"]), "rep_void": bool(rbars["void"]),
               "both_pass": bool(bars["pass"] and rbars["pass"])}
    return {"verdict": verdict, "main": main, "rep": repo, "_trades": {"main": tr, "rep": rep["trades"]}, "_eq": {"main": eq, "rep": eq2}}


def _qs(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "p50": float(np.median(x)), "p95": float(np.quantile(x, 0.95)), "max": float(x.max())}


def _skips(oos: dict, sel: list) -> dict:
    out: dict = {}
    for f, i in enumerate(sel):
        for k, v in oos[(i, f)][1].attrs["skips"].items():
            out[k] = out.get(k, 0) + int(v)
    return out


def pipeline(X1: dict, D1: dict, X2: dict, D2: dict, P1info: dict, P2info: dict, workers: int) -> dict:
    """워크포워드 → 재현(폴드 2023 선택) → 널 ② → 바·보고."""
    wf = walk_forward(D1, workers)
    rep = replicate(D2, wf["sel"][-1], workers)
    tr_main, _ = oos_selected(wf, D1)
    _, e1 = sim_bounds(D1["days"], MAIN)
    plans = {"main": W.null2_plan(tr_main, D1["days"], e1), "rep": W.null2_plan(rep["trades"], D2["days"], rep["bounds"][1])}
    holds = {"main": W.null2_holds(tr_main)[0], "rep": W.null2_holds(rep["trades"])[0]}   # 각 시험의 선택 설정 거래(RULINGS null2)
    t = time.time()
    n2 = run_null2({"main": D1, "rep": D2}, plans, holds, W.N_NULL, workers)
    log(f"널 ② {2 * W.N_NULL} 회 · {time.time() - t:.0f}s")
    return assemble(X1, D1, wf, X2, D2, rep, n2, P1info, P2info)


# =============================================================================================== 출력

def _already(logs_dir) -> list[str]:
    logs_dir = Path(logs_dir)
    found = [n for n in (OUT_JSON, OUT_MD) if (logs_dir / n).exists()]
    t = logs_dir / OUT_TRIALS
    if t.exists():
        for line in t.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("trial") == TRIAL_NAME:
                found.append(f"{OUT_TRIALS}({TRIAL_NAME})")
                break
    return found


def guard_rerun(logs_dir=LOGS) -> None:
    found = _already(logs_dir)
    if found:
        raise SystemExit(f"{logs_dir}: {found} 가 있다 — 이 백테스트는 이미 돌았다(재실행 = 새 시도, 사전등록 필요)")


def write_outputs(logs_dir, json_text: str, md_text: str, trial: str) -> None:
    """JSON → MD → 원장(마지막). 이미 있으면 아무것도 쓰지 않고 멈춘다. 도중 실패는 쓴 목록과 함께 RuntimeError(재실행은 막힌다)."""
    logs_dir = Path(logs_dir)
    found = _already(logs_dir)
    if found:
        raise RuntimeError(f"{logs_dir}: {found} 가 이미 있다 — 출력을 쓰지 않는다(재실행 금지)")
    logs_dir.mkdir(parents=True, exist_ok=True)
    done: list[str] = []
    try:
        MQC._write_new(logs_dir / OUT_JSON, json_text)
        done.append(OUT_JSON)
        MQC._write_new(logs_dir / OUT_MD, md_text)
        done.append(OUT_MD)
        MQC.append_line(logs_dir / OUT_TRIALS, trial)
        done.append(OUT_TRIALS)
    except Exception as e:
        raise RuntimeError(f"출력 쓰기 실패 — 이미 쓴 것 {done}. 재실행은 막힌다 — {logs_dir} 를 손으로 확인하라") from e


require_clean_tree = MQC.require_clean_tree
git_sha = MQC.git_sha


def config() -> dict:
    return {"main": MAIN, "rep": REP, "fold_years": FOLD_YEARS, "halves": HALVES, "k": K, "M": M_PATHS, "short": SHORT, "long": LONG,
            "gap_max": GAP_MAX, "d_max": D_MAX, "adv_floor_mn": ADV_FLOOR, "universe_min_history": MIN_HISTORY, "member_hist_q0": MEMBER_HIST_Q0,
            "sigma_w": SIGMA_W, "pool_years": POOL_YEARS, "pool_min_dates": POOL_MIN_DATES, "lu_cut": LU_CUT, "lu": LU, "table": PRICE_TABLE,
            "slots": E.SLOTS, "w_cap": E.W_CAP, "risk": E.RISK, "cost_judge": W.COST_JUDGE, "min_trades": W.MIN_TRADES, "n_null": W.N_NULL,
            "seed": SEED, "dsr_n": DSR_N, "n_trials": N_TRIALS, "menu_axes": W.AXES,
            "prereg": "research/logs/golden_bt/PREREG.md (07f7ce9)"}


def _f(x, fmt="{:+.3f}") -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def _ox(b: dict) -> str:
    return "O" if b.get("pass") else "X"


def results_md(res: dict, meta: dict) -> str:
    v, mn, rp = res["verdict"], res["main"], res["rep"]
    b, rb = mn["bars"], rp["bars"]
    L = ["# 골든 임박 자리 포지션 백테스트 결과 (검증 전)", "",
         f"git `{meta['git']}` · 사전등록 `research/logs/golden_bt/PREREG.md`(07f7ce9) · 주 시험 {mn['eval'][0]} ~ {mn['eval'][1]}(폴드 "
         f"{', '.join(map(str, FOLD_YEARS))}) · 재현 {rp['eval'][0]} ~ {rp['eval'][1]} · 비용 2배 판정 · DSR N={N_TRIALS}", "",
         f"**주 시험:** {'VOID(체결 < 100)' if v['main_void'] else ('바 전부 통과' if v['main_pass'] else '바 미달')} · "
         f"**재현:** {'VOID(체결 < 100)' if v['rep_void'] else ('바 전부 통과' if v['rep_pass'] else '바 미달')} · "
         f"**둘 다:** {'통과 → 전진 모의매매 사전등록' if v['both_pass'] else '실패 → 이 셋업 종결(PREREG)'}", "",
         "## 주 시험 바(폴드 밖·비용 2배)", "", "| 바 | 값 | 기준 | 통과 |", "|---|---|---|---|",
         f"| 체결 | {b['n_trades']:,} | ≥ 100 | {'O' if not b['void'] else 'X'} |",
         f"| ① R 합 vs 널 | {_f(b['b1_nulls']['R_sum'], '{:+.1f}')} | > 널① p95 {_f(b['b1_nulls']['null1_p95'], '{:+.1f}')} ∧ 널② p95 "
         f"{_f(b['b1_nulls']['null2_p95'], '{:+.1f}')} | {_ox(b['b1_nulls'])} |",
         f"| ② R 평균 [월 군집 95%] | {_f(b['b2_cluster_boot']['mean'])} [{_f(b['b2_cluster_boot']['lo'])}, {_f(b['b2_cluster_boot']['hi'])}] "
         f"| 하한 > 0 | {_ox(b['b2_cluster_boot'])} |",
         f"| ③ 상위 5 종목 제거 R 평균 | {_f(b['b3_top5_removed']['mean'])} | > 0 | {_ox(b['b3_top5_removed'])} |",
         f"| ④ 팩터 알파(월) t OLS / NW | {_f(b['b4_factor_alpha']['alpha'], '{:+.4f}')} · {_f(b['b4_factor_alpha']['t_ols'], '{:+.2f}')} / "
         f"{_f(b['b4_factor_alpha']['t_nw'], '{:+.2f}')} (n={b['b4_factor_alpha']['n']}) | 둘 다 > 2 | {_ox(b['b4_factor_alpha'])} |",
         f"| ⑤ 연복리 vs 노출 맞춘 시장 | {_f(b['b5_cagr_vs_expo_market']['cagr'], '{:+.2%}')} vs "
         f"{_f(b['b5_cagr_vs_expo_market']['expo_market_cagr'], '{:+.2%}')} | > | {_ox(b['b5_cagr_vs_expo_market'])} |",
         f"| ⑥ 폴드 R 합 > 0 | {b['b6_folds']['n_pos']}/4 ({', '.join(_f(x, '{:+.1f}') for x in b['b6_folds']['fold_R_sum'])}) | ≥ 3/4 "
         f"| {_ox(b['b6_folds'])} |",
         f"| ⑦ DSR | {_f(b['b7_dsr'].get('prob_deflated_sharpe'), '{:.3f}')} (월 SR {_f(b['b7_dsr'].get('observed_sharpe'))}) | > 0.9 "
         f"| {_ox(b['b7_dsr'])} |", "",
         "## 폴드 선택(학습 = 이웃 중앙값 최대)", "", "| 폴드 | θ_in | θ_out | sizing | exit | θ_d | 학습 R 합 | 이웃 중앙값 | 시험 R 합 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in mn["selection"]:
        c = s["cfg"]
        L.append(f"| {s['year']} | {c['th_in']} | {c['th_out']} | {c['sizing']} | {c['exit']} | {c['th_d'] if c['th_d'] is not None else '—'} | "
                 f"{_f(s['train_R_sum_selected'], '{:+.1f}')} | {_f(s['train_nbr_median_selected'], '{:+.1f}')} | "
                 f"{_f(mn['fold_R_sum'][s['year']], '{:+.1f}')} |")
    L += ["", "## 재현 2009-01 ~ 2016-08(폴드 2023 선택 설정)", "", "| 바 | 값 | 기준 | 통과 |", "|---|---|---|---|",
          f"| 체결 | {rb['n_trades']:,} | ≥ 100 | {'O' if not rb['void'] else 'X'} |",
          f"| ① R 합 vs 널 | {_f(rb['b1_nulls']['R_sum'], '{:+.1f}')} | > 널① p95 {_f(rb['b1_nulls']['null1_p95'], '{:+.1f}')} ∧ 널② p95 "
          f"{_f(rb['b1_nulls']['null2_p95'], '{:+.1f}')} | {_ox(rb['b1_nulls'])} |",
          f"| ② R 평균 [월 군집 95%] | {_f(rb['b2_cluster_boot']['mean'])} [{_f(rb['b2_cluster_boot']['lo'])}, {_f(rb['b2_cluster_boot']['hi'])}] "
          f"| 하한 > 0 | {_ox(rb['b2_cluster_boot'])} |",
          f"| ④ 팩터 알파 t OLS / NW | {_f(rb['b4_factor_alpha']['t_ols'], '{:+.2f}')} / {_f(rb['b4_factor_alpha']['t_nw'], '{:+.2f}')} | 둘 다 > 2 "
          f"| {_ox(rb['b4_factor_alpha'])} |",
          f"| ⑤ 연복리 vs 노출 맞춘 시장 | {_f(rb['b5_cagr_vs_expo_market']['cagr'], '{:+.2%}')} vs "
          f"{_f(rb['b5_cagr_vs_expo_market']['expo_market_cagr'], '{:+.2%}')} | > | {_ox(rb['b5_cagr_vs_expo_market'])} |",
          "| 두 반기 R 평균 | " + " · ".join(f"{h['lo'][:4]}~ {_f(h['mean'])} (n={h['n']})" for h in rb["b_halves"]["halves"])
          + f" | 둘 다 > 0 | {_ox(rb['b_halves'])} |", "",
          "## 보고 열", ""]
    for name, part in (("주 시험", mn), ("재현", rp)):
        rr = part["report"]
        a = rr.get("all", {})
        L += [f"### {name}", "",
              f"- 기대값(R 평균) {_f(a.get('R_mean'))} · 손익비 {_f(a.get('payoff'), '{:.2f}')} · 평균 이익 {_f(a.get('avg_win_R'))} R · 평균 손실 "
              f"{_f(a.get('avg_loss_R'))} R · 승률(보고) {_f(a.get('win_share'), '{:.1%}')} · 체결 {a.get('n', 0):,}",
              f"- 비용 1배 R 합 {_f(part['cost']['x1']['R_sum'], '{:+.1f}')} / 2배 {_f(part['cost']['x2']['R_sum'], '{:+.1f}')} · 연복리 1배 "
              f"{_f(part['cost']['x1']['cagr'], '{:+.2%}')} / 2배 {_f(part['cost']['x2']['cagr'], '{:+.2%}')} · 평균 노출 {_f(part['exposure_mean'], '{:.1%}')} "
              f"· 최대 노출 {_f(part['exposure_max'], '{:.1%}')} · KOSPI 보유 연복리 {_f(part['kospi_cagr'], '{:+.2%}')}",
              f"- 확률 NaN 판정일이 있던 거래 {rr.get('nan_prob', {}).get('n_trades_any_nan_decision', 0):,} · 대체 청산(ma200_fb) "
              f"{rr.get('nan_prob', {}).get('n_fallback_exits', 0):,} (정정 GB-1)",
              f"- 털기율(손절 뒤 20일 안 진입가 회복) {_f(rr.get('shakeout', {}).get('rate'), '{:.1%}')} (손절 {rr.get('shakeout', {}).get('n_stop', 0)}) · "
              f"골든 도달 {_f(rr.get('golden_share'), '{:.1%}')} · 보유 중앙 {_f(rr.get('hold_days', {}).get('median'), '{:.0f}')} 일",
              f"- 널 ① p50/p95 {_f(part['null1'].get('p50'), '{:+.1f}')} / {_f(part['null1'].get('p95'), '{:+.1f}')} · 널 ② "
              f"{_f(part['null2'].get('p50'), '{:+.1f}')} / {_f(part['null2'].get('p95'), '{:+.1f}')} (모자란 진입 합 {part['null2'].get('n_short_total', 0)})", "",
              "| 청산 사유 | n | R 평균 | R 합 |", "|---|---|---|---|"]
        for k, s in rr.get("by_reason", {}).items():
            L.append(f"| {k} | {s['n']:,} | {_f(s.get('R_mean'))} | {_f(s.get('R_sum'), '{:+.1f}')} |")
        L += ["", "| d_fill 구간 | n | R 평균 |", "|---|---|---|"]
        for k, s in rr.get("by_d_fill", {}).items():
            L.append(f"| {k} | {s.get('n', 0):,} | {_f(s.get('R_mean'))} |")
        L += ["", "| 진입 연도 | n | R 합 | R 평균 | 자산 수익(2배) |", "|---|---|---|---|---|"]
        for y, s in rr.get("by_entry_year", {}).items():
            L.append(f"| {y} | {s['n']:,} | {_f(s.get('R_sum'), '{:+.1f}')} | {_f(s.get('R_mean'))} | "
                     f"{_f(part['yearly_return_x2'].get(y), '{:+.2%}')} |")
        L.append("")
    L += ["2024~: 조회하지 않음(오염 — 사전등록 보고 열은 선택, 생략).", "",
          "해석(RULINGS)은 `backtest.json` 의 `rulings` 에 있다. 거래 CSV(차트·검증 스웜용): `~/.cache/cc-swing/scratch/golden_bt/`."]
    return "\n".join(L) + "\n"


# =============================================================================================== 본체

def _connect():
    from swing_it.features.eps_asof import eps_ok
    from swing_it.storage import connect

    if not os.environ.get("KR_QUANT_DB"):
        raise SystemExit("KR_QUANT_DB 가 없다 — set -a; . ~/git/quant-airflow/.env; set +a")
    con = connect(os.environ["KR_QUANT_DB"])
    con.set_session(readonly=True)
    return con, eps_ok


def load_prepare(con, eps_ok, win: dict) -> dict:
    t = time.time()
    raw = load(con, win)
    check_window(raw["days"], win)
    i = raw["info"]
    log(f"{win['name']} load: 종목 {i['n_codes']:,} × 거래일 {i['n_days']:,} ({i['first_day']} ~ {i['last_day']}) · pre_listing {i['n_pre_listing_cells']:,} · "
        f"정지일 NaN {i['n_suspension_cells_nan']:,} · 시가 결측(거래일) {i['n_open_missing_on_trading_day']:,} · 시총 유한 {i['cap_finite_share']:.3f} · "
        f"{time.time() - t:.1f}s")
    t = time.time()
    codes, days = raw["codes"], raw["days"]
    X = prepare(raw, lambda ci, s: eps_ok(codes, days, ci, s, con, source=win["eps"]))
    log(f"{win['name']} prepare: U 종목-일 {int(X['U'].sum()):,} · 기초 칸 {int(X['base'].sum()):,} · 사전 후보 풀 {int(X['pool'].sum()):,} · "
        f"시뮬 행 {X['bounds']} · {time.time() - t:.1f}s")
    return X


def _fake_probs(X: dict) -> dict:
    """스모크 시간 탐침 전용 가짜 확률(결정적 균등 난수) — 결과는 버리고 출력하지 않는다."""
    rng = np.random.default_rng(0)
    return {"PG": rng.random(X["C"].shape, dtype=np.float32), "PD": rng.random(X["C"].shape, dtype=np.float32)}


def _smoke(Xs: dict, args) -> int:
    """모양·건수·시간만 — 수익·R·CAGR·바 없음. 확률은 달 몇 개만 실제로 계산(시간), 엔진은 가짜 확률로 시간만."""
    est = 0.0
    for name, X in Xs.items():
        cells = prob_cells(X)
        months = sorted(cells)
        n_cells = sum(len(v[0]) for v in cells.values())
        pick = [months[int(j)] for j in np.linspace(0, len(months) - 1, min(SMOKE_PROB_MONTHS, len(months)))] if months else []
        t = time.time()
        P = prob_panels(X, 1, months=pick)
        secs = time.time() - t
        done = P["info"]["n_cells"]
        per_cell = secs / max(done, 1)
        t_prob = per_cell * n_cells / max(args.workers, 1) + 30
        log(f"{name} 확률 칸: {n_cells:,} 칸 · {len(months)} 달 · 탐침 {len(pick)} 달 {done:,} 칸 {secs:.1f}s → 전체 ≈ {t_prob / 60:.1f} 분"
            f"(작업자 {args.workers}) · 탐침 풀 크기 {P['info']['pool_P']}")
        D = sim_data(X, _fake_probs(X))
        s, e = X["bounds"]
        t = time.time()
        tr, _ = E.simulate(D, W.menu()[0], 2.0, s, e)
        t_sim = time.time() - t
        n_sims = (48 * 8 + 4) if name == "main" else 49
        t = time.time()
        plan = W.null2_plan(tr.assign(sizing="eq"), D["days"], e)
        W.null2_rep(D, plan, np.maximum(tr["hold"].to_numpy(int), 1) if len(tr) else np.array([10]), np.random.default_rng(0))
        t_n2 = time.time() - t
        t_tot = t_prob + n_sims * t_sim / args.workers + W.N_NULL * t_n2 / args.workers
        log(f"{name} 엔진 탐침(가짜 확률, 전 구간 1 설정): {t_sim:.1f}s · 시뮬 {n_sims} 개 → ≈ {n_sims * t_sim / args.workers / 60:.1f} 분 · "
            f"널 ② 1 회 {t_n2:.1f}s → 300 회 ≈ {W.N_NULL * t_n2 / args.workers / 60:.1f} 분 · 지수 커버리지 "
            f"{ {k: round(v, 3) for k, v in _index_cover(X, s, e).items()} }")
        est += t_tot
    log(f"예상(실제 실행, 적재·준비 제외): ≈ {est / 60:.0f} 분 — 적재·준비 시간은 위 load/prepare 줄")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="모양·건수·시간만(지표 없음)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scratch", default=str(SCRATCH))
    ap.add_argument("--logs", default=str(LOGS))
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    scratch, logs = Path(args.scratch), Path(args.logs)
    if not args.smoke:
        if logs.resolve() != Path(LOGS).resolve():
            raise SystemExit(f"실제 실행은 정본 {LOGS} 로만 — --logs 는 바꾸지 않는다")
        require_clean_tree()
        guard_rerun(logs)
    scratch.mkdir(parents=True, exist_ok=True)
    gs = git_sha()
    con, eps_ok = _connect()
    try:
        X1 = load_prepare(con, eps_ok, MAIN)
        X2 = load_prepare(con, eps_ok, REP)
    finally:
        con.close()                                        # fork 전에 닫는다(자식은 DB 를 쓰지 않는다)
    if args.smoke:
        return _smoke({"main": X1, "rep": X2}, args)
    for X in (X1, X2):
        require_index(X, *X["bounds"])

    t = time.time()
    P1 = prob_panels(X1, args.workers)
    log(f"main 확률: 칸 {P1['info']['n_cells']:,} · 유한 P_g {P1['info']['n_finite_pg']:,} · 풀 없는 달 {P1['info']['n_nopool_months']} · {time.time() - t:.0f}s")
    t = time.time()
    P2 = prob_panels(X2, args.workers)
    log(f"rep 확률: 칸 {P2['info']['n_cells']:,} · 유한 P_g {P2['info']['n_finite_pg']:,} · 풀 없는 달 {P2['info']['n_nopool_months']} · {time.time() - t:.0f}s")
    D1, D2 = sim_data(X1, P1), sim_data(X2, P2)
    for tag, P in (("main", P1), ("rep", P2)):
        np.savez_compressed(scratch / f"probs_{tag}.npz", PG=P["PG"], PD=P["PD"])

    res = pipeline(X1, D1, X2, D2, P1["info"], P2["info"], args.workers)
    final = res["main"]["selection"][-1]["cfg_i"]
    for tag in ("main", "rep"):
        res["_trades"][tag].to_csv(scratch / f"trades_{tag}.csv", index=False)
        res["_eq"][tag].to_frame("equity_x2").assign(expo=res["_eq"][tag].attrs["expo"]).to_csv(scratch / f"equity_{tag}.csv")
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    meta = {"git": gs["sha"], "git_head": gs["sha"], "git_dirty": gs["dirty"], "git_clean_paths": MQC.WATCHED, "date": _dt.date.today().isoformat(),
            "data": {"main": X1["info"], "rep": X2["info"]}, "config": config(), "rulings": RULINGS}
    json_text = json.dumps(MQC._clean({**meta, **out}), allow_nan=False, ensure_ascii=False, indent=1) + "\n"
    md_text = results_md(out, meta)
    trial = MQC.trial_line({"trial": TRIAL_NAME, "date": meta["date"], "config": config(), "verdict": out["verdict"],
                            "selected": [s["cfg_i"] for s in out["main"]["selection"]], "rep_cfg": final, "n_trials": N_TRIALS, "git": gs["sha"]})
    write_outputs(logs, json_text, md_text, trial)
    log(f"완료 → {logs / OUT_JSON} · {logs / OUT_MD} · 원장 {logs / OUT_TRIALS} · 거래 CSV {scratch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
