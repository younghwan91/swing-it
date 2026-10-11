#!/usr/bin/env python
"""패턴 GA — 수렴 모양·천장·진입 기준선·돌파 실패·본전·확장 청산을 유전자로, CMAwM + BIPOP × 워크포워드. **리포터 — 판정 아님.**

사전등록: ``research/logs/sepa_explore/PATTERN_GA.md`` — aeaf8e5 + 정정 0 ab612dc + **정정 1 72fab71**(``REVIEW_PATTERN_GA.md`` 의
"정정 1 초안" 12항목 + MINOR 1~12 를 그대로 채택 — 그 리뷰 절이 정본) + 정정 2 c483edb(급등 뒤 쉼) + 정정 3 2154f77(천장 종류·H0 첫 점·
바닥 진입·위생), 모두 결과 보기 전. 숫자를 ``PATTERN_GA_RESULTS.md`` 에 적고 바 문턱과의 **산술 비교**만 한다.

- 사건 표: :mod:`research.signals.pattern_events` (θ 6 × 천장 창 3 × 기준선 4 = 72 표, 첫 관통일만, d−1 서술자) — 설정과 무관하게 한 번
  만들어 ``--cache`` 디렉터리에 .npy 로 두고 작업자가 memmap 으로 함께 읽는다.
- 평가: :mod:`research.signals.pattern_fast` (거르기 → 매수정지 체결 → 확장 청산 경로(들어간 트레이드만) → 10 슬롯 포트폴리오 − 종목별
  모멘텀 정합 바스켓). 적합도 = 학습창 일별 초과의 월합 3개월 블록 부트스트랩(500, 공통 난수 seed 0) 연환산 2.5% 하단 − 연속 벌점
  λ·[max(0, 12·연수 − n)/(12·연수) + 연평균 max(0, 4·연 비율 − n_y)/(4·연 비율)], λ = 0.20(리뷰 C3-4·M6-3). 비용 2×(주).
- 워크포워드: ``swing_it.validation.walkforward.FOLDS`` 6개(3년 학습 → 1년 시험, 마지막 시험 2025-01 ~ 사건 2026-08-31). 학습 트레이드 =
  진입 ∈ 학습창, 학습창 끝(test_lo 직전 거래일 종가) 시가평가(purge 금지). 폴드마다 CMAwM + BIPOP 시드 10 × 런당 10,000 평가.
  **절차 출력(M2)** = 시드 최적 10개를 새 부트스트랩 난수(seed :data:`REEVAL_SEED`)로 학습창에서 다시 평가한 1등. 표본 밖 = 폴드 출력
  config 를 시험 연도 신규 진입에만 쓰는 **하나의 연속 포트폴리오**(열린 포지션은 원 config 청산), 데이터 끝 2026-09-11 시가평가.
- 널: (2a) 폴드마다 균등 추출 풀(:data:`POOL_N`)에서 학습창 실현 가능(벌점 0) 설정을 폴드별 독립으로 뽑아 이어 붙인 OOS 300벌,
  (2b) 모양 서술자를 같은 달 안에서 섞은 사건 표 위에서 **절차 전체**(폴드 6 × 시드 10 × 10,000) 50벌. 기술: 같은 풀의 무작위 탐색
  best-of-2,000 300벌, 청산 전용 대조 300벌(GA 출력의 청산 유전자 고정, 진입·모양 무작위).
- 원장: ``research/logs/sepa_explore/pattern_ga/TRIALS.jsonl`` — 절차·널·민감도 칸과 설계 진단(건수만)을 결과 전에, 런 요약은 런이
  끝날 때마다(결과 문서보다 먼저).

Run:  uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/pattern_ga_run.py --feasibility-only
      uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/pattern_ga_run.py --workers 12
디버그: ... pattern_ga_run.py --debug --out <scratch>/pga_debug   (폴드 2 · 시드 2 · 예산 300 · 널 축소 — 콘솔은 건수·시간만)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from research.signals import event_families as EF  # noqa: E402
from research.signals import ga_fast as gf  # noqa: E402
from research.signals import pattern_events as PE  # noqa: E402
from research.signals import pattern_eval as PEV  # noqa: E402
from research.signals import pattern_fast as PF  # noqa: E402
from research.signals.sepa import LIMIT_UP_RATIO  # noqa: E402

OUT_DIR = "research/logs/sepa_explore"
LABEL = "pattern_ga"
RUN_LABEL = "pattern_ga_runs"           # 런 요약 원장(2단계 MINOR 10 — DSR 변형 N 에 안 센다)
SUB = "pattern_ga"
RESULTS_MD = "PATTERN_GA_RESULTS.md"
FEAS_MD = "FEASIBILITY.md"
PREREG_DOC = ("research/logs/sepa_explore/PATTERN_GA.md (aeaf8e5 + 정정 0 ab612dc + 정정 1 72fab71 = REVIEW_PATTERN_GA.md 정정 1 초안·"
              "MINOR 1~12 + 정정 2 c483edb + 정정 3 2154f77) + REVIEW_PATTERN_GA.md 2단계 필수 수정 1~8")
#: 설계 진단·사슬 원장 줄의 지문용(사전등록 문자열이 늘어도 같은 줄이 두 번 들어가지 않게 — 2단계 MINOR 10).
DESIGN_PREREG = ("research/logs/sepa_explore/PATTERN_GA.md (aeaf8e5 + 정정 0 ab612dc + 정정 1 72fab71 = REVIEW_PATTERN_GA.md 정정 1 초안·"
                 "MINOR 1~12 + 정정 2 c483edb + 정정 3 2154f77)")
DEFAULT_CACHE = os.environ.get("PATTERN_GA_CACHE", "/tmp/pattern_ga_cache")
CACHE_VERSION = "pga-v4"                # 사건 표 정의가 바뀌면 올린다 — 코드 해시도 대조한다(2단계 MINOR 11)

# --- 사전등록 수치(탐색하지 않는다) ---
EVENT_LO, EVENT_HI = "2017-09-15", "2026-08-31"
END_DAY = "2026-09-11"                 # 데이터 끝(NXT 절단) — 시가평가(M2-3)
ADV_FLOOR = 1_000.0                    # ADV20 ≥ 10억(백만원)
MINUTE_LO = "2025-08-01"
COST_MULT = 2.0
SEEDS = tuple(range(10))
BUDGET = 10_000
N_NULL_A, N_NULL_B, N_RS, N_EXIT_CTL = 300, 50, 300, 300
POOL_N = 100_000                       # 폴드당 균등 추출 풀 — 완전 평가(무작위 탐색 best-of-2,000 기술용, 실현 불가능 포함)
SCREEN_N = 6_000_000                   # 2단계 M1(정정 10: 3M→6M, 건수 투영 근거): 폴드당 건수만 선별 → 실현 가능 풀(널 2a)
SCREEN_CHUNK = 50_000
FEAS_MIN = 500                         # 2단계 M1: 폴드별 실현 가능 고유 config 하한 — 미달이면 2a ✗(판단 불가)
RESTART_SCREEN = 100_000               # 2단계 M2: BIPOP 재시작 평균 = 앞 10만 개 선별의 실현 가능 config(실제·2b 각자 자료)
RS_BEST_OF = 2_000
FIT_BOOT, FIT_SEED, REEVAL_SEED, CI_BOOT = 500, 0, 7_919, 2_000
CI_SEED = 20261018                     # 보고 CI 부트스트랩 — 적합도 공통 난수와 다른 시드(2단계 MINOR 12)
LAMBDA, FLOOR_PER_YEAR, FLOOR_YEAR = 0.20, 12.0, 4.0
OFF_ZONE = 0.02
POOL_SEED, NULL_A_SEED, RS_SEED, EXIT_CTL_SEED, NULLB_SEED = 20261013, 20261014, 20261015, 20261016, 20261012
CHART_SEED, CHART_N = 20261011, 12
SLIP_SENS = (0.003, 0.005)             # 매수만 · 매수·매도 모두(2단계 MINOR 7) 두 칸씩
TOPK = (1, 3, 5)                       # 상위 k 거래 제거 곡선(MINOR 4)
ORDER_SEED = 20261017
TOP_PBO, PBO_BLOCKS = 100, 12
FOLD_MIN_FILLS = 20                    # 바 4: 체결 < 20 폴드는 판단 불가
B6_N, B6_GL, B6_MARGIN, BE_BAND = 60, 2.0, 0.05, 0.005
DSR_STRUCTS = 12                       # 이 계열 검증 구조 수(리뷰 M3 — 10~12)
N_FEAS = 50_000                        # 실현 가능 비율 — 균등 추출 수(건수만 · 2a 선별 규모 예측용)
FLAT = 0.15                            # 모양 이름: |ŝ| ≤ 0.15 = 평평(MINOR 9, 결과 전 고정)
REGION_RUNUP, REGION_REST = 0.30, 20.0  # 정정 2 착상 영역 좌표

#: 변형(``PATTERN_GA_VARIANT`` — spawn 작업자도 같은 값을 읽도록 환경변수). "fw" = PATTERN_GA_FW.md:
#: 사건 모양을 하락 쐐기로 고정(F1), 불타기 필수(F2), 본전 이동 0.5·1R(F3), 계산 규모(F4). 기본("")은 PATTERN_GA.md 그대로.
#: "fwl" = PATTERN_GA_FWL.md: FW 전부 + 주도주 고정(EPS YoY ≥ 25% ∧ 가속, RS ≥ 70, 추세 템플릿) — 같은 데이터 세 번째 좁힘(탐색 전용).
#: "fwm" = PATTERN_GA_FWM.md: FWL 위 미너비니 정합(사용자 차트 지적) — EPS 직전 분기 대비 상승 필수, 손절폭 ≤ 5%(손절선을 끌어올림),
#: 짧고 날카로운 쐐기 허용(위생 간격 ≥ 1·길이 ≥ 5), stage 3/4(d−1 종가 > MA50). 탐색 전용.
VARIANT = os.environ.get("PATTERN_GA_VARIANT", "")
FAST_EVAL = os.environ.get("PATTERN_GA_FAST", "1") != "0"   # 0 이면 evaluate 가 원래 파이썬 경로(evaluate_ref)
FIT_MODE = os.environ.get("PATTERN_GA_FIT", "ann")   # 11세대 후보: "trade" = 거래당 슬롯 대비 순(2배) 부트스트랩 하한 − 벌점(사용자 목표 거래당 +5%)
CORE_BIPOP = os.environ.get("PATTERN_GA_CORE_BIPOP", "1") != "0"   # 0 이면 ga_run_one 이 cma_bipop_x0(cmaes 순정)
FW = VARIANT in ("fw", "fwl", "fwm")
FWL = VARIANT in ("fwl", "fwm")
FWM = VARIANT == "fwm"
REGIME_CSV = "research/logs/sepa_explore/regime/regime_panel.csv"   # FWM M7: 시장 국면 패널(d 종가까지 자료, 뒤쪽 창만)
VIEW_TOL = 20                          # 리뷰 FWM M7: 열람 제외 = 같은 종목 ± 20 거래일
STOP_MAX = 0.05                        # FWM M2(사용자 "−5% 이상 손실은 잘못 · 스탑로스를 거기에 건다")
HYGIENE_FWM = {"gap_min": 1, "span_min": 5}   # FWM M3: 대한해운 2021-01(H0 → 첫 저점 2일, 패턴 7~9일)이 지워지지 않게
if FWM:
    CACHE_VERSION = "pga-v5-fwm"
if FW:
    _T = VARIANT
    LABEL, RUN_LABEL, SUB, RESULTS_MD = f"pattern_ga_{_T}", f"pattern_ga_{_T}_runs", f"pattern_ga_{_T}", f"PATTERN_GA_{_T.upper()}_RESULTS.md"
    PREREG_DOC = DESIGN_PREREG = ("research/logs/sepa_explore/PATTERN_GA_FWM.md (PATTERN_GA_FWL.md 위 미너비니 정합 M1~M5 — 탐색 전용)" if FWM else
                                  "research/logs/sepa_explore/PATTERN_GA_FWL.md (PATTERN_GA_FW.md 위 주도주 고정 L1~L4 — 탐색 전용)" if FWL else
                                  "research/logs/sepa_explore/PATTERN_GA_FW.md (PATTERN_GA.md 정정 0~11 위 변형 F1~F5)")
    SEEDS = tuple(range(int(os.environ.get("PATTERN_GA_FW_SEEDS", "10"))))
    BUDGET = int(os.environ.get("PATTERN_GA_FW_BUDGET", "10000"))
    N_NULL_B = int(os.environ.get("PATTERN_GA_FW_NULL_B", "50"))
    SCREEN_N = int(os.environ.get("PATTERN_GA_FW_SCREEN", "300000000"))  # 리뷰 FW M1: 적응 선별 상한(폴드당)
    RESTART_SCREEN = 200_000 if FWM else 1_000_000                        # 리뷰 FW M2: 10만 → 100만 · 리뷰 FWM S2: FWM 20만(4배 보정 투영 ≈ 19h > 18h)
FW_FEAS_TARGET = 1_000                 # 리뷰 FW M1: 폴드마다 실현 가능 config 가 이 수에 닿는 가장 짧은 덩어리 접두(건수만 정지 규칙)
FW_Q_2A, FW_Q_2B = 99.0, 100.0         # 리뷰 FW M5: 사후 좁힘 다중성 — 2a p99 · 2b 50벌 전부 위
FW_PARENT_VIEWED = "research/logs/sepa_explore/pattern_ga_fw/parent_viewed_trades.csv"   # 리뷰 FW M5: 부모 열람 OOS 거래
FW_PARENT_SHA256 = "6e989c1948d8a10ce020a20cb7461f32fb4f6e6cd98949111ceaa170c49d6fdb"   # 리뷰 FW M10
if FWL:                                # L5(사용자 "수천번 매매할 필요 없다"): 연 12 → 6, 연도별 4 → 2 · 리뷰 FWL M6: 9/3 은 2a 를 막아 6/2 유지,
    FLOOR_PER_YEAR, FLOOR_YEAR = 6.0, 2.0   # 바 6 건수만 미달은 '하한 결합 — 판단 불가'(정정 FWL-2)
    FEAS_MIN = 300                     # 리뷰 FWL M1: 폴드 6 실현 가능 비율 ≈ 1.8e-6 — 상한 3억에서 기대 ≈ 540, 2a 하한 500 → 300
if os.environ.get("PATTERN_GA_FLOOR"):  # 12세대: 거래당 적합도가 거래를 줄이는 것 견제 — "연,연도별"(예 "10,3")
    FLOOR_PER_YEAR, FLOOR_YEAR = (float(v) for v in os.environ["PATTERN_GA_FLOOR"].split(","))
if FWL:                                # L4: 부모 186 + FW 중간 열람 155 의 합집합(종목, 주문일)
    FW_PARENT_VIEWED = "research/logs/sepa_explore/pattern_ga_fwl/parent_viewed_trades.csv"
    FW_PARENT_SHA256 = "fbb985092c458e0c358bba97770264b85677b2f75468692040612f94dcc8d1d7"
if FWM:                                # FWM M5: FWL 339 + FW 최종 실제 GA 99 의 합집합(종목, 주문일)
    FW_PARENT_VIEWED = "research/logs/sepa_explore/pattern_ga_fwm/parent_viewed_trades.csv"
    FW_PARENT_SHA256 = "d92bbd8a14461748957cfd57e04a4dcdeb51964fe8037f6fc3c89e7e78474ef1"


def fw_ok(su: float, sl: float, g: dict) -> bool:
    """FW F1 사건 모양: 상단선 하락 ∧ 하단선 < fw_lo ∧ 수렴(리뷰 FW M7 — 사례표도 같은 판정)."""
    return bool(su < 0.0 and sl < g["fw_lo"] and su < sl)

# --- 유전자(정정 0·1·2·3) ---
#: 연속 (이름, 하한, 상한, 끔 끝) — [0, 1] 정규화, 끔 끝에서 :data:`OFF_ZONE` 안이면 조건 끔.
CONT = (("rs_min", 0.0, 90.0, "lo"), ("breadth_min", 0.0, 0.7, "lo"),
        ("sup_c", -1.5, 1.0, None), ("sup_hw", 0.05, 1.5, "hi"), ("slo_c", -1.0, 1.5, None), ("slo_hw", 0.05, 1.5, "hi"),
        ("rho_max", 0.2, 1.5, "hi"), ("span_c", math.log(15.0), math.log(120.0), None), ("span_hw", 0.1, 1.5, "hi"),
        ("ceil_min", 0.70, 1.00, "lo"), ("depth_max", 0.15, 0.50, "hi"), ("dry_max", 0.5, 1.5, "hi"),
        ("runup_min", 0.0, 1.0, "lo"), ("rest_gap_max", 3.0, 120.0, "hi"),
        ("gap_max", 0.01, 0.05, None), ("stop_cap", 0.05, 0.12, None), ("fb_tol", 0.0, 0.03, None), ("trail_k", 2.0, 5.0, None),
        ("stop_buf", 0.0, 1.0, "lo"), ("stop_hard", 1.5, 3.0, None),                       # 정정 5
        ("climax_day", 0.10, 0.25, None), ("climax_5d", 0.20, 0.50, None), ("climax_r", 2.0, 6.0, None))   # 정정 4
if FW:                                 # FW F1(정정 FW-1, 사용자 "바운드를 유연하게"): 하단선 기울기 상한 유전자
    CONT = (*CONT, ("fw_lo", -0.15, 0.15, None))
if FWL:                                # L2: RS 하한 70~90(끔 없음)
    CONT = (("rs_min", 70.0, 90.0, None), *CONT[1:])
if FWM:                                # M3 짧은 쐐기: 길이 중심 하한 15 → 5 · M2: 손절 원값 거리 상한(소진)은 25%까지(손절선은 5% 로 끌어올림)
    CONT = tuple(("span_c", math.log(5.0), math.log(120.0), None) if c[0] == "span_c" else
                 ("stop_cap", 0.05, 0.25, None) if c[0] == "stop_cap" else c for c in CONT)
#: 명목형(M2-1) — 범주별 연속 로짓 [0, 1] → argmax(동률 = 앞).
#: FWL L3: stage = 추세 템플릿(d−1 전부 · H0 일 TT)만.
NOM = (("stage", (3, 4) if FWM else (2, 3) if FWL else (0, 1, 2, 3)), ("entry", PE.BASES), ("trail", ("ma20", "ma50", "chand")),
       ("pyramid", ("1r", "newhigh", "upper") if FW else ("off", "1r", "newhigh", "upper")),
       *((("mkt", ("off", "nhnl", "bos", "br200")),) if FWM else ()))   # FWM M7: 시장 국면(그 종목 시장, d−1)
#: 이산(순서형 — CMAwM 정수 축).
DISC = (("ceil_win", PE.CEIL_WINS), ("theta", PE.THETAS), ("runup_win", PE.RUNUP_WINS), ("touch_lo", (2, 3)),
        ("e_min", (None, 0.25, 0.5) if FWM else (0.25, 0.5) if FWL else (None, 0.0, 0.25, 0.5)),
        ("e_accel", (0, 1) if FWM else (1, 1) if FWL else (0, 1)),   # FWL L1: EPS 가속 필수 · FWM M1: 직전 분기 대비 상승은 늘, YoY 는 선택
        ("stop_anchor", ("E", "lower") if FWM else ("E", "lower", "dynamic")), ("fb_days", (1, 2, 3, 0)),
        ("be_r", (0.5, 1.0) if FW else (0.5, 1.0, 1.5, 2.0, math.inf)), ("take_r", (2.0, 3.0, 4.0, math.inf)), ("take_frac", (1 / 3, 1 / 2)),
        ("tighten", (0, 1)), ("ceil_kind", ("single", "resist")), ("h0_in_pattern", (0, 1)), ("climax", (0, 1)),
        ("stop_basis", ("intraday", "intraday") if FWM else ("intraday", "close")),   # FWM M2: 종가 손절은 −5% 를 넘긴다
        ("fail_exit", ("off", "prev_low", "entry_low")),
        ("reentry", ("off", "F", "line")), ("reentry_win", (1, 2, 3, 4, 5)))
#: 진화 모드 메뉴 덮어쓰기(``PATTERN_GA_MENU`` JSON — META_EVOLUTION.md): {"nom": {이름: [수준…]}, "disc": {이름: [수준…]},
#: "cont": {이름: [하한, 상한]}}. "inf" → ∞, null → 끔. 변형 메뉴 뒤에 적용한다(spawn 작업자도 같은 환경변수를 읽는다).
MENU_OVERRIDE = json.loads(os.environ.get("PATTERN_GA_MENU", "") or "{}")
if MENU_OVERRIDE:
    def _lv(v):
        return math.inf if v == "inf" else v
    _no, _do, _co = MENU_OVERRIDE.get("nom", {}), MENU_OVERRIDE.get("disc", {}), MENU_OVERRIDE.get("cont", {})
    def _dl(xs):                                       # 검증 MUST-FIX 3: CMAwM 이산 축은 선택지 ≥ 2 — 한 수준이면 복제(stop_basis 와 같은 방식)
        t_ = tuple(_lv(x) for x in xs)
        return t_ * 2 if len(t_) == 1 else t_
    #: 진화 1세대 새 유전자(덮어쓰기에 이름이 있을 때만 메뉴에 붙는다 — 기존 변형의 차원·재현 불변)
    EVO_GENES = {"trail_start": (0.0,), "trail_basis": ("intraday",), "stop_policy": ("clamp",),
                 "exit_safe": (0,), "be_basis": ("high",), "entry_q": (0,),
                 "stop_max": (0.05,), "climax_basis": ("intraday",),         # 7세대: 손절 상한 · 8세대: 급등 매도 판단
                 "flow_win": (20,), "indiv_sell": (0,), "e_age_min": (1,),  # 9세대: 수급 창 · 개인 순매도 요구 · 반전봉 뒤 확인 일수
                 "first_frac": (0.6667,), "pyr_close": (0,),
                 # 13세대(EVENT_FAMILIES.md 정정 E-1): 사건 가족 켬/끔 · P 신고가 창·진입 · C 종류·진입
                 "fam_w": (1,), "fam_p": (0,), "fam_c": (0,), "p_hiw": (60,), "p_entry": ("open",), "c_kind": ("both",),
                 "c_entry": ("open",), "lock_to": (None,)}                   # 13a: +1R 본전 발동 뒤 잠금 수준(R)                # 10세대: 첫 진입 비중 · +1R 추가 종가 확인             # 2세대: 추적 무장·불타기 안전 · 본전 발동 기준
    DISC = (*DISC, *((n, _dl(_do[n])) for n in EVO_GENES if n in _do))
    NOM = tuple((n, tuple(_lv(x) for x in _no[n]) if n in _no else v) for n, v in NOM)
    DISC = tuple((n, _dl(_do[n]) if n in _do else v) for n, v in DISC)
    def _off(n, c3):                                   # [하한, 상한] 은 기존 끔 끝 유지, [하한, 상한, null] 은 끔 없음(필수 조건)
        return _co[n][2] if len(_co[n]) >= 3 else c3
    CONT = tuple((c[0], float(_co[c[0]][0]), float(_co[c[0]][1]), _off(c[0], c[3])) if c[0] in _co else c for c in CONT)
    #: 진화 3세대 후보 연속 유전자(이름이 있을 때만 붙는다): d−1 ATR20/종가 상한, 위쪽 끝 = 끔
    EVO_CONT = {"atr_max": ("atr_max", 0.015, 0.06, "hi"), "flow_min": ("flow_min", -0.1, 0.3, "lo"),   # 9세대: 수급 하한(아래 끝 = 끔)
                "body_min": ("body_min", -0.02, 0.01, "lo"),           # 9세대: d−1 몸통 하한(아래 끝 = 끔)
                "blowoff_x": ("blowoff_x", 5.0, 20.0, "hi"),           # 10세대: 블로오프 H0 거래량 배수 문턱(위쪽 끝 = 끔)
                # 13세대 P·C 문턱(메뉴의 세 번째 원소 null = 끔 없음 — 필수 문턱)
                "p_vmult": ("p_vmult", 2.0, 8.0, None), "p_ret": ("p_ret", 0.02, 0.15, None), "p_clv": ("p_clv", 0.5, 1.0, None),
                "p_hold": ("p_hold", 0.0, 0.05, "hi"), "p_flow": ("p_flow", -0.1, 0.3, "lo"),
                "c_ret": ("c_ret", 0.0, 0.1, "lo"), "c_vmult": ("c_vmult", 1.0, 5.0, "lo"), "c_flow": ("c_flow", -0.1, 0.3, "lo")}
    CONT = (*CONT, *(EVO_CONT[n][:1] + (float(_co[n][0]), float(_co[n][1]), _off(n, EVO_CONT[n][3])) for n in EVO_CONT if n in _co))
    _unk = (set(_no) - {n for n, _ in NOM}) | (set(_do) - {n for n, _ in DISC}) | (set(_co) - {c[0] for c in CONT})
    if _unk:
        raise SystemExit(f"PATTERN_GA_MENU 모르는 유전자: {sorted(_unk)}")
N_LOGIT = sum(len(v) for _, v in NOM)
N_CONT = len(CONT) + N_LOGIT
DISC_SIZES = tuple(len(v) for _, v in DISC)
DIM = N_CONT + len(DISC)
GENE_NAMES = tuple(n for n, *_ in CONT) + tuple(n for n, _ in NOM) + tuple(n for n, _ in DISC)
STAGE_NAMES = ("MA 정렬만", "+ 종가 > MA50", "추세 템플릿 전부(d−1)", "H0 일 TT ∧ d−1 MA 정렬", "H0 일 TT ∧ d−1 종가 > MA50",
               "H0 일 TT ∧ d−1 종가 > MA50 ∧ MA50 상승")
#: 모양 유전자(M6 모양 비활성 짝·C1-2b 와 같은 묶음 — 정정 3 의 천장 종류·H0 첫 점 포함).
SHAPE_GENES = ("sup_c", "sup_hw", "slo_c", "slo_hw", "rho_max", "touch_lo", "span_c", "span_hw", "ceil_min", "depth_max",
               "dry_max", "ceil_kind", "h0_in_pattern")
#: 청산 유전자(M1-f 청산 전용 대조가 고정하는 묶음).
EXIT_GENES = ("fb_days", "fb_tol", "be_r", "take_r", "take_frac", "trail", "trail_k", "tighten", "pyramid", "climax", "climax_day",
              "climax_5d", "climax_r", "fail_exit", "reentry", "reentry_win")
#: 정정 4~7 착상 거래 — (이름, 해당 유전자 판정, 착상 연도, 시험창 포함). 그 해가 학습창(시험 포함이면 시험창도)에 든 폴드에서 그
#: 유전자를 고르면 '착상과 같은 데이터' 표시(정정 5·6·7 문구 · 정정 4 는 같은 취지 기술).
IDEA_FLAGS = (("정정 4 클라이맥스(유나이티드제약 2018-01)", lambda g: bool(g.get("climax")), (2018,), False),
              ("정정 5 종가 기준 손절(한화솔루션 2020-12)", lambda g: g.get("stop_basis") == "close", (2020,), False),
              ("정정 6 조기 청산·재진입·상단선 불타기(동서 2021-01)",
               lambda g: g.get("fail_exit") not in (None, "off") or g.get("reentry") == "F" or g.get("pyramid") == "upper", (2021,), True),
              ("정정 7 하단선 동적 손절(아이즈비전 2021-05 외)", lambda g: g.get("stop_anchor") == "dynamic", (2020, 2021), True))


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def _pct(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def _f(x, fmt: str = "{:.2f}") -> str:
    return "nan" if x is None or not (isinstance(x, (int, float, np.floating, np.integer)) and np.isfinite(x)) else fmt.format(x)


# ---------------------------------------------------------------------------
# 1. 부호화 · 복호화 (순수 — tests/test_pattern_ga.py)
# ---------------------------------------------------------------------------

def canonical(g: dict) -> dict:
    """조건부로 평평한 유전자를 ``None`` 으로(같은 표현형 = 같은 키 — GA2 D-m1 ii). H0 기준선 × 신고가 불타기는 금지(M1-c → 끔)."""
    g = dict(g)
    if g["sup_hw"] is None:
        g["sup_c"] = None
    if g["slo_hw"] is None:
        g["slo_c"] = None
    if g["span_hw"] is None:
        g["span_c"] = None
    if g["runup_min"] is None:
        g["runup_win"] = None
    if g["e_min"] is None:
        g["e_accel"] = None
    if g["take_r"] == math.inf:
        g["take_frac"] = None
    if g["trail"] != "chand":
        g["trail_k"] = None
        g["tighten"] = None
    if not g.get("climax"):
        g["climax_day"] = g["climax_5d"] = g["climax_r"] = None
    if g["entry"] in ("horiz", "h0") and g.get("stop_anchor") == "dynamic":   # 정정 7: 하단선 동적은 추세선 기준선만(그 밖 = 끔 → E)
        g["stop_anchor"] = "E"
    if g["entry"] == "h0" and g["pyramid"] == "newhigh":
        g["pyramid"] = "1r" if FW else "off"            # FW F2: 불타기 필수 — 금지 조합은 1R 불타기로
    if g["entry"] == "bottom":                       # 정정 3: 손절 = E 고정, 익절 전 돌파 실패 없음
        g["stop_anchor"] = None
        g["fb_days"] = None
        g["fb_tol"] = None
    elif g["fb_days"] == 0:
        g["fb_tol"] = None
    if g["entry"] != "bottom":                       # 정정 6: 반등 실패 조기 청산은 바닥 반전 진입에만
        g["fail_exit"] = None
    re_ = g.get("reentry")                           # 정정 6 F 재진입 = 바닥 반전만 · 정정 9 선 복귀 = 하단선 손절(고정·동적)만
    if (re_ == "F" and g["entry"] != "bottom") or (re_ == "line" and g.get("stop_anchor") not in ("lower", "dynamic")):
        g["reentry"] = "off"
    if g.get("reentry") != "line":
        g["reentry_win"] = None
    if g.get("stop_basis") != "close" and g.get("stop_anchor") != "dynamic":   # 재난 손절 = 종가 기준·하단선 동적에서만(정정 5·7)
        g["stop_hard"] = None
    if "fam_p" in g or "fam_c" in g:                 # 13세대: 가족 하나는 켠다 · 꺼진 가족의 문턱은 평평(None)
        if not (g.get("fam_w", 1) or g.get("fam_p") or g.get("fam_c")):
            g["fam_p"] = 1
        if not g.get("fam_p"):
            for k in ("p_vmult", "p_ret", "p_clv", "p_hold", "p_flow", "p_hiw", "p_entry"):
                if k in g:
                    g[k] = None
        elif g.get("p_entry") != "close" and "p_hold" in g:
            g["p_hold"] = None
        if not g.get("fam_c"):
            for k in ("c_ret", "c_vmult", "c_flow", "c_kind", "c_entry"):
                if k in g:
                    g[k] = None
    return g


def decode(x) -> dict:
    """CMA 벡터(연속 18 + 로짓 14 [0,1] + 이산 14 단계 인덱스) → 유전자 dict(:func:`canonical`)."""
    x = np.asarray(x, float)
    out: dict = {}
    for i, (name, lo, hi, off) in enumerate(CONT):
        u = float(np.clip(x[i], 0.0, 1.0))
        if (off == "lo" and u <= OFF_ZONE) or (off == "hi" and u >= 1.0 - OFF_ZONE):
            out[name] = None
        else:
            out[name] = round(lo + u * (hi - lo), 4)
    j = len(CONT)
    for name, levels in NOM:
        v = x[j:j + len(levels)]
        out[name] = levels[int(np.argmax(v))]
        j += len(levels)
    for q, (name, levels) in enumerate(DISC):
        k = int(np.clip(round(float(x[N_CONT + q])), 0, len(levels) - 1))
        out[name] = levels[k]
    return canonical(out)


def encode(g: dict) -> np.ndarray:
    """유전자 → CMA 벡터(끔 = 끔 끝 값, ``None`` 인 조건부 유전자 = 범위 가운데·첫 단계). :func:`decode` 의 역(4자리 반올림 안에서)."""
    v = []
    for name, lo, hi, off in CONT:
        val = g.get(name)
        if val is None:
            v.append(0.0 if off == "lo" else (1.0 if off == "hi" else 0.5))
        else:
            v.append((val - lo) / (hi - lo))
    for name, levels in NOM:
        oh = [0.0] * len(levels)
        oh[levels.index(g[name])] = 1.0
        v += oh
    for name, levels in DISC:
        val = g.get(name)
        v.append(float(levels.index(val)) if val in levels else 0.0)
    return np.array(v)


def gkey(g: dict) -> tuple:
    return tuple(g.get(k) for k in GENE_NAMES)


def ghash(g: dict) -> int:
    return int.from_bytes(hashlib.blake2b(repr(gkey(g)).encode(), digest_size=8).digest(), "little", signed=True)


def x0_feasible() -> np.ndarray:
    """CMA 첫 런 평균(리뷰 C3-4) — 모든 모양 구간·맥락 거르기 끔(실현 가능점), 나머지는 범위 가운데."""
    x = np.full(DIM, 0.5)
    for i, (name, lo, hi, off) in enumerate(CONT):
        if off == "lo":
            x[i] = OFF_ZONE / 2
        elif off == "hi":
            x[i] = 1.0 - OFF_ZONE / 2
    for q, (name, levels) in enumerate(DISC):
        x[N_CONT + q] = {"e_min": 0.0, "touch_lo": 0.0, "ceil_kind": 0.0, "climax": 0.0, "stop_basis": 0.0, "fail_exit": 0.0,
                         "reentry": 0.0, "stop_anchor": 0.0, "reentry_win": 0.0}.get(name, (len(levels) - 1) / 2.0)
    if MENU_OVERRIDE:                                 # 진화: 끔 없는 필수 연속 조건은 가장 느슨한 끝에서 시작(실현 가능성)
        loose = {"runup_min": 0.0, "dry_max": 1.0, "atr_max": 1.0, "depth_max": 1.0, "rs_min": 0.0}
        for i, (name, lo, hi, off) in enumerate(CONT):
            if off is None and name in loose and name in MENU_OVERRIDE.get("cont", {}):
                x[i] = loose[name]
    if FWL:                                           # FWL L6: 주도주 고정 안에서 가장 느슨한 실현 가능점(건수만 탐색 — 6폴드 모두 실현 가능)
        names = [n for n, *_ in CONT]
        x[names.index("rs_min")], x[names.index("fw_lo")] = 0.0, 1.0          # RS 70 · 하단선 < +0.15
        j = len(CONT)
        for name, levels in NOM:
            want = {"stage": 3, "entry": "bottom"}.get(name)
            if want is not None and want not in levels:   # 진화 메뉴 덮어쓰기: 가장 느슨한 수준이 빠졌으면 첫 수준(stage 4 등)
                want = levels[0]
            if want is not None:
                x[j:j + len(levels)] = 0.0
                x[j + levels.index(want)] = 1.0
            j += len(levels)
        for q, (name, levels) in enumerate(DISC):
            if name in ("theta", "ceil_win"):
                w_ = {"theta": 0.08, "ceil_win": 120}[name]
                x[N_CONT + q] = float(levels.index(w_) if w_ in levels else 0)
    return x


def gene_vec(g: dict, extra_slip: float = 0.0) -> np.ndarray:
    """유전자 → :mod:`pattern_fast` 거르기 벡터(``gv``). 끔 = NaN."""
    def nz(v):
        return np.nan if v is None else float(v)

    gv = np.full(PF.N_G, np.nan)
    gv[PF.G_STAGE] = g["stage"]
    gv[PF.G_RS] = nz(g["rs_min"])
    gv[PF.G_BR] = nz(g["breadth_min"])
    if g["sup_hw"] is not None:
        gv[PF.G_SUP_LO], gv[PF.G_SUP_HI] = g["sup_c"] - g["sup_hw"], g["sup_c"] + g["sup_hw"]
    if g["slo_hw"] is not None:
        gv[PF.G_SLO_LO], gv[PF.G_SLO_HI] = g["slo_c"] - g["slo_hw"], g["slo_c"] + g["slo_hw"]
    gv[PF.G_RHO] = nz(g["rho_max"])
    gv[PF.G_TOUCH] = g["touch_lo"]
    if g["span_hw"] is not None:
        gv[PF.G_SPAN_LO], gv[PF.G_SPAN_HI] = math.exp(g["span_c"] - g["span_hw"]), math.exp(g["span_c"] + g["span_hw"])
    gv[PF.G_CEIL] = nz(g["ceil_min"])
    gv[PF.G_DEPTH] = nz(g["depth_max"])
    gv[PF.G_DRY] = nz(g["dry_max"])
    gv[PF.G_RUNUP_COL] = PE.RUNUP_WINS.index(g["runup_win"]) if g["runup_win"] is not None else 0
    gv[PF.G_RUNUP] = nz(g["runup_min"])
    gv[PF.G_REST] = nz(g["rest_gap_max"])
    gv[PF.G_EMIN] = nz(g["e_min"])
    gv[PF.G_EACC] = float(g["e_accel"] or 0)
    gv[PF.G_GAP] = g["gap_max"]
    gv[PF.G_ANCHOR] = {"lower": 1.0, "dynamic": 2.0}.get(g["stop_anchor"], 0.0)
    gv[PF.G_CAP] = g["stop_cap"]
    gv[PF.G_FB_TOL] = nz(g["fb_tol"]) if g["fb_tol"] is not None else 0.0
    gv[PF.G_SLIP] = 0.001 + extra_slip
    gv[PF.G_DFLOOR] = 0.02
    gv[PF.G_BASE] = PE.BASES.index(g["entry"])
    gv[PF.G_H0IN] = float(g["h0_in_pattern"])
    gv[PF.G_CEILKIND] = 1.0 if g["ceil_kind"] == "resist" else 0.0
    gv[PF.G_EQ] = 1.0 if FWM else 0.0                # FWM M1: EPS 전년 같은 분기 대비 상승(흑자) 필수(정정 FWM-1 M9)
    gv[PF.G_SMAX] = float(g.get("stop_max") or STOP_MAX) if FWM else np.nan       # FWM M2 · 6세대 유전자
    gv[PF.G_MKT] = ("off", "nhnl", "bos", "br200", "nhnl_br200").index(g.get("mkt") or "off")   # FWM M7 · 5세대 nhnl_br200
    gv[PF.G_SPOL] = 1.0 if g.get("stop_policy") == "skip" else 0.0                 # 진화 1세대
    gv[PF.G_ATRMAX] = np.nan if g.get("atr_max") is None else float(g["atr_max"])   # 진화 3세대 후보
    gv[PF.G_ENTQ] = float(g.get("entry_q") or 0)                                      # 3세대
    gv[PF.G_FLMIN] = np.nan if g.get("flow_min") is None else float(g["flow_min"])     # 9세대 수급
    gv[PF.G_FLWIN] = 1.0 if int(g.get("flow_win") or 20) == 20 else 0.0
    gv[PF.G_INDSELL] = float(g.get("indiv_sell") or 0)
    gv[PF.G_BODYMIN] = np.nan if g.get("body_min") is None else float(g["body_min"])
    gv[PF.G_EAGEMIN] = float(g["e_age_min"]) if g.get("e_age_min") is not None else np.nan
    gv[PF.G_BLOWX] = np.nan if g.get("blowoff_x") is None else float(g["blowoff_x"])
    gv[PF.G_STOP_BUF] = g.get("stop_buf") or 0.0
    gv[PF.G_STOP_CLOSE] = 1.0 if g.get("stop_basis") == "close" else 0.0
    gv[PF.G_STOP_HARD] = g["stop_hard"] if g.get("stop_hard") is not None else np.nan
    return gv


def exit_vec(g: dict, extra_slip: float = 0.0, sell_extra: float = 0.0) -> np.ndarray:
    """유전자 → 청산 인자 벡터(``px``)."""
    px = np.zeros(PF.N_X)
    px[PF.X_TAKE_R] = g["take_r"]
    px[PF.X_TAKE_FRAC] = g["take_frac"] if g["take_frac"] is not None else 0.5
    px[PF.X_TRAIL] = ("ma20", "ma50", "chand").index(g["trail"])
    px[PF.X_TRAIL_K] = g["trail_k"] if g["trail_k"] is not None else 3.0
    px[PF.X_TIGHTEN] = float(g["tighten"] or 0)
    px[PF.X_PYR] = ("off", "1r", "newhigh", "upper").index(g["pyramid"])
    px[PF.X_BE_R] = g["be_r"]
    px[PF.X_FB_DAYS] = 0 if g["fb_days"] is None else g["fb_days"]
    px[PF.X_MAX_HOLD] = 120
    px[PF.X_FROM_ENTRY] = 1.0
    px[PF.X_SLIP] = 0.001 + extra_slip
    px[PF.X_SELL_SLIP] = 0.001
    px[PF.X_BE_MULT] = 1.0
    px[PF.X_TIGHT_R] = 3.0
    px[PF.X_PYR_R] = 1.0
    px[PF.X_WB] = float(g.get("first_frac") or PF.WB_PYR)   # 10세대: 첫 진입 비중(시험 매수 1/2 · 기존 2/3 · 1)
    px[PF.X_PYR_CLOSE] = float(g.get("pyr_close") or 0)
    def _nan(v):
        return np.nan if v is None else float(v)
    px[PF.X_ON_HOLD] = np.nan                         # 13세대(가족 행은 fam_exit 가 채운다)
    px[PF.X_LOCK_TO] = _nan(g.get("lock_to"))         # 13세대 청산 분석(exitlab): 끔 = NaN(기존과 비트 동일)
    px[PF.X_REV_R] = _nan(g.get("rev_r"))
    px[PF.X_REV_NEXT] = float(g.get("rev_next") or 0)
    px[PF.X_PLOW_R] = _nan(g.get("plow_r"))
    px[PF.X_CLOSE_CUT] = _nan(g.get("close_cut"))
    px[PF.X_SELL_TICK] = 1.0                     # 손절·본전 체결 −1호가(쐐기 D 와 같다)
    px[PF.X_SELL_EXTRA] = sell_extra
    px[PF.X_CLIMAX] = float(g.get("climax") or 0)
    px[PF.X_CLIMAX_DAY] = g.get("climax_day") or 0.15
    px[PF.X_CLIMAX_5D] = g.get("climax_5d") or 0.30
    px[PF.X_CLIMAX_R] = g.get("climax_r") or 4.0
    px[PF.X_STOP_CLOSE] = 1.0 if g.get("stop_basis") == "close" else 0.0
    px[PF.X_STOP_HARD] = g.get("stop_hard") or 2.0
    px[PF.X_FAIL_EXIT] = {"prev_low": 1, "entry_low": 2}.get(g.get("fail_exit"), 0)
    px[PF.X_REENTRY] = {"F": 1.0, "line": 2.0}.get(g.get("reentry"), 0.0)
    px[PF.X_REENTRY_WIN] = float(g.get("reentry_win") or 0)
    px[PF.X_DYN] = 1.0 if g.get("stop_anchor") == "dynamic" else 0.0
    px[PF.X_GAP] = g["gap_max"]
    px[PF.X_CAP] = g["stop_cap"]
    px[PF.X_BUF] = g.get("stop_buf") or 0.0
    px[PF.X_DFLOOR] = 0.02
    px[PF.X_SMAX] = float(g.get("stop_max") or STOP_MAX) if FWM else np.nan     # FWM M2: 재진입 손절에도 같은 상한
    px[PF.X_INTRA] = 1.0 if FWM else 0.0            # FWM M6: 장중 청산(알고리즘 매매)
    px[PF.X_TRAIL_START] = float(g.get("trail_start") or 0.0)   # 진화 1세대
    px[PF.X_TRAIL_CLOSE] = 1.0 if g.get("trail_basis") == "close" else 0.0
    px[PF.X_SAFE] = float(g.get("exit_safe") or 0)                 # 진화 2세대
    px[PF.X_BE_CLOSE] = 1.0 if g.get("be_basis") == "close" else 0.0
    px[PF.X_CLIMAX_CLOSE] = 1.0 if g.get("climax_basis") == "close" else 0.0   # 8세대
    px[PF.X_RISK] = STOP_MAX if FWM else np.nan     # 7세대(검증 MUST-FIX 9): FWM 은 늘 위험 기준 비중 — 손절 한 번의 슬롯 손실 ≤ 5%(갭 예외)
    return px


def table_of(g: dict) -> int:
    return PE.table_id(PE.THETAS.index(g["theta"]), PE.CEIL_WINS.index(g["ceil_win"]), PE.BASES.index(g["entry"]))


def shape_off(g: dict) -> dict:
    """모양 비활성 짝(M6-1): 모양 유전자만 끔(구간 전체), 나머지 그대로."""
    h = dict(g)
    h.update({"sup_c": None, "sup_hw": None, "slo_c": None, "slo_hw": None, "rho_max": None, "touch_lo": 2, "span_c": None,
              "span_hw": None, "ceil_min": None, "depth_max": None, "dry_max": None, "ceil_kind": "single"})
    return canonical(h)


def shape_name(cu, cl) -> str:
    """정규화 좌표 모양 이름(MINOR 9 — 결과 전 고정, |ŝ| ≤ :data:`FLAT` = 평평)."""
    if cu is None or cl is None or not (np.isfinite(cu) and np.isfinite(cl)):
        return "모양 끔(구간 없음)" if (cu is None and cl is None) else "한쪽만"
    up, lo = (0 if abs(cu) <= FLAT else (1 if cu > 0 else -1)), (0 if abs(cl) <= FLAT else (1 if cl > 0 else -1))
    if up == 0 and lo == 1:
        return "평평한 천장"
    if up == 1 and lo == 1:
        return "상승 쐐기" if cl > cu else "상승 채널/확산"
    if up == -1 and lo == -1:
        return "하락 쐐기" if cu < cl else "하락 채널/확산"
    if up == -1 and lo == 1:
        return "대칭 삼각"
    return "기타"


def config_shape(g: dict) -> str:
    return shape_name(g.get("sup_c") if g.get("sup_hw") is not None else None,
                      g.get("slo_c") if g.get("slo_hw") is not None else None)


def _falling_interval(g: dict) -> bool:
    """두 구간이 켜져 있고 전부 음수 ∧ 중심 ŝ_up < ŝ_lo (정정 1 C2 하락 쐐기 좌표의 구간판)."""
    if g.get("sup_hw") is None or g.get("slo_hw") is None:
        return False
    return (g["sup_c"] + g["sup_hw"] < 0) and (g["slo_c"] + g["slo_hw"] < 0) and g["sup_c"] < g["slo_c"]


def _rest_ok(g: dict) -> bool:
    return (g.get("runup_min") is not None and g["runup_min"] >= REGION_RUNUP and g.get("rest_gap_max") is not None
            and g["rest_gap_max"] <= REGION_REST)


def user_region(g: dict) -> dict:
    """사용자 착상 영역(결과 전 고정): W = 정정 1 C2(하락 쐐기 ∧ 추세선 ∧ fb 2) ∧ 정정 2(급등 ≥ 30% ∧ 쉼 ≤ 20일),
    WD = 정정 3(H0 = 첫 점 ∧ 바닥 반전) ∧ 하락 쐐기 ∧ 정정 2."""
    w = _falling_interval(g) and g["entry"] == "trend" and g.get("fb_days") == 2 and _rest_ok(g)
    wd = _falling_interval(g) and g["entry"] == "bottom" and g.get("h0_in_pattern") == 1 and _rest_ok(g)
    return {"W": bool(w), "WD": bool(wd), "any": bool(w or wd)}


def region_overlap(g: dict) -> dict:
    """2단계 M3 (2) 겹침: 모양 구간(끔 = 전체)이 하락 쐐기 사분면(ŝ_up < 0 ∧ ŝ_lo < 0)과 겹치고 기준선이 영역과 같다(W 추세선, WD 바닥)."""
    def neg(c, hw):
        return hw is None or (c - hw) < 0
    q = neg(g.get("sup_c"), g.get("sup_hw")) and neg(g.get("slo_c"), g.get("slo_hw"))
    return {"W": bool(q and g["entry"] == "trend"), "WD": bool(q and g["entry"] == "bottom"),
            "any": bool(q and g["entry"] in ("trend", "bottom"))}


def trade_region_share(E: "Engine", t: pd.DataFrame) -> float:
    """2단계 M3 (3) 트레이드 기준: OOS 체결 중 사건 자체가 착상 영역(:func:`event_region`)인 비율."""
    if t is None or t.empty:
        return np.nan
    ev = t["ev"].to_numpy(int)
    F = np.asarray(E.F_orig)[ev]
    w, wd = event_region(F, np.asarray(E.I)[ev, PE.II["table"]] % len(PE.BASES))
    return float(np.mean(w | wd))


def event_region(F: np.ndarray, base: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """사건별 착상 영역(건수 세기용, 가장 느슨한 창 — 60일 급등): W·WD 행 bool."""
    fi = PE.FI
    su, sl = F[:, fi["s_up"]], F[:, fi["s_lo"]]
    with np.errstate(invalid="ignore"):
        fall = (su < 0) & (sl < 0) & (su < sl) & (F[:, fi["runup60"]] >= REGION_RUNUP) & (F[:, fi["rest_gap"]] <= REGION_REST)
    w = fall & (base == PE.B_TREND)
    wd = fall & (base == PE.B_BOTTOM) & (F[:, fi["h0in"]] == 1)
    return w, wd


def median_config(cfgs: "list[dict]") -> "dict | None":
    """M2-2: 시드 최적이 모두 같은 모양 이름일 때만 — 연속은 중앙값, 이산은 단계 중앙값(짝수면 아래), 명목은 최빈(동률 = 앞)."""
    if len({config_shape(c) for c in cfgs}) != 1:
        return None
    xs = np.array([encode(c) for c in cfgs])
    med = np.empty(DIM)
    med[:len(CONT)] = np.median(xs[:, :len(CONT)], axis=0)
    j = len(CONT)
    for name, levels in NOM:
        cnt = np.zeros(len(levels))
        for c in cfgs:
            cnt[levels.index(c[name])] += 1
        oh = np.zeros(len(levels))
        oh[int(np.argmax(cnt))] = 1.0
        med[j:j + len(levels)] = oh
        j += len(levels)
    for q in range(len(DISC)):
        s = np.sort(xs[:, N_CONT + q])
        med[N_CONT + q] = s[(len(s) - 1) // 2]
    return decode(med)


def gene_str(g: dict) -> str:
    def c(v, fmt="{:.2f}"):
        return "끔" if v is None else fmt.format(v)

    def inf(v):
        return "끔" if v is None or v == math.inf else f"{v:g}"

    sup = "끔" if g["sup_hw"] is None else f"[{g['sup_c'] - g['sup_hw']:+.2f},{g['sup_c'] + g['sup_hw']:+.2f}]"
    slo = "끔" if g["slo_hw"] is None else f"[{g['slo_c'] - g['slo_hw']:+.2f},{g['slo_c'] + g['slo_hw']:+.2f}]"
    trk = "" if g["trail_k"] is None else f" k{g['trail_k']:.2f}"
    hard_s = "" if g.get("stop_hard") is None else f" 재난{g['stop_hard']:.2f}"
    clx = "" if not g.get("climax") else f" 클라이맥스(일{g['climax_day']:.0%}·5일{g['climax_5d']:.0%}·{g['climax_r']:.1f}R)"
    span = "끔" if g["span_hw"] is None else f"[{math.exp(g['span_c'] - g['span_hw']):.0f},{math.exp(g['span_c'] + g['span_hw']):.0f}]"
    fwl = f"FW하단<{g['fw_lo']:+.2f} " if g.get("fw_lo") is not None else ""      # 리뷰 FW M9
    return (f"{fwl}θ{g['theta']:.0%} 창{g['ceil_win']} {g['entry']} 단계{g['stage']} rs≥{c(g['rs_min'], '{:.0f}')} br≥{c(g['breadth_min'])} "
            f"ŝ↑{sup} ŝ↓{slo} ρ≤{c(g['rho_max'])} 닿음{g['touch_lo']} L{span} 천장≥{c(g['ceil_min'])} 깊이≤{c(g['depth_max'])} "
            f"마름≤{c(g['dry_max'])} 급등{g['runup_win'] or ''}≥{c(g['runup_min'])} 쉼≤{c(g['rest_gap_max'], '{:.0f}')} "
            f"천장{g['ceil_kind']} H0첫점{g['h0_in_pattern']} E≥{c(g['e_min'])}{'가속' if g['e_accel'] else ''} 갭≤{g['gap_max']:.3f} "
            f"손절{g['stop_anchor'] or 'E'}≤{g['stop_cap']:.3f} | fb{g['fb_days'] if g['fb_days'] is not None else '-'}·"
            f"{c(g['fb_tol'], '{:.3f}')} be{inf(g['be_r'])} 익절{inf(g['take_r'])}R×{c(g['take_frac'])} 추적{g['trail']}"
            f"{trk}{'조임' if g['tighten'] else ''} 불타기{g['pyramid']}{clx} 손절기준{g.get('stop_basis')}"
            f"{hard_s} 여유{c(g.get('stop_buf'))}ATR 조기청산{g.get('fail_exit') or '-'} 재진입{g.get('reentry') or '-'}{g.get('reentry_win') or ''}")


# ---------------------------------------------------------------------------
# 2. CMA-ES(CMAwM + BIPOP) — ``ga_run.cma_bipop`` + 실현 가능 첫 평균(리뷰 C3-4)
# ---------------------------------------------------------------------------

def cma_bipop_x0(fitness, n_cont: int, disc_sizes: "tuple[int, ...]", *, budget: int, seed: int, x0=None,
                 sigma0: float = 0.3, stall: bool = True, opt_cls=None, x_restart=None):
    """``ga_run.cma_bipop`` 과 같은 재시작 규칙(Hansen 2009 BIPOP, ``stall`` 정체 재시작) — 첫 런 평균만 ``x0``(실현 가능점).
    ``ask`` 는 (평가용 x, tell 용 연속 x) 를 돌려준다 — 적합도는 평가용으로 재고 **tell 에는 표본 연속 x** 를 넘긴다(GA2 D-m1 iv).
    ``opt_cls`` = 시험용 대역(기본 ``cmaes.CMAwM``). ``x_restart`` = 재시작 평균 후보(2단계 M2 — 그 학습창의 실현 가능 config;
    없으면 균등). ``fitness`` 가 (값, 진입 수) 를 돌려주면 재시작별 '진입 0건 표본 비율' 을 기록한다. 반환 (이력[(번호, x_eval, 값)],
    재시작 기록)."""
    if opt_cls is None:
        from cmaes import CMAwM as opt_cls  # noqa: PLC0415, N813 — 선택 의존성(uv run --with cmaes==0.13.1)
    dim = n_cont + len(disc_sizes)
    bounds = np.array([[0.0, 1.0]] * n_cont + [[0.0, float(k - 1)] for k in disc_sizes])
    steps = np.array([0.0] * n_cont + [1.0] * len(disc_sizes))
    scale = bounds[:, 1] - bounds[:, 0]
    cov0 = np.diag(scale ** 2)
    rng = np.random.default_rng(seed)
    lam0 = 4 + int(3 * math.log(dim))
    hist, restarts = [], []
    n_eval, large_used, small_used, large_pop = 0, 0, 0, lam0
    first = True
    while n_eval < budget:
        mean = bounds[:, 0] + rng.random(dim) * scale
        if x_restart is not None and len(x_restart):
            mean = np.clip(np.asarray(x_restart[int(rng.integers(len(x_restart)))], float), bounds[:, 0], bounds[:, 1])
        if first:
            regime, lam, sig = "first", lam0, sigma0
            if x0 is not None:
                mean = np.clip(np.asarray(x0, float), bounds[:, 0], bounds[:, 1])
            first = False
        elif small_used < large_used:
            u = rng.random()
            regime, lam, sig = "small", max(2, int(lam0 * (0.5 * large_pop / lam0) ** (u ** 2))), sigma0 * 10 ** (-2 * rng.random())
        else:
            large_pop *= 2
            regime, lam, sig = "large", large_pop, sigma0
        opt = opt_cls(mean=mean, sigma=sig, bounds=bounds, steps=steps, seed=int(rng.integers(2 ** 31 - 1)),
                      population_size=lam, cov=cov0.copy())
        used, gens, run_best, since, zero = 0, 0, -np.inf, 0, 0
        stall_n = 10 + math.ceil(30 * dim / lam)
        why = "budget"
        while n_eval < budget:
            if opt.should_stop():
                why = "should_stop"
                break
            if stall and since >= stall_n:
                why = "stall"
                break
            sols = []
            for _ in range(opt.population_size):
                if n_eval >= budget:
                    break
                x_eval, x_tell = opt.ask()
                fv = fitness(x_eval)
                if isinstance(fv, tuple):
                    v = float(fv[0])
                    zero += int(fv[1] == 0)
                else:
                    v = float(fv)
                hist.append((n_eval, x_eval.copy(), v))
                n_eval += 1
                used += 1
                sols.append((x_tell, -v))
            if len(sols) < opt.population_size:
                break
            opt.tell(sols)
            gens += 1
            g_best = -min(v for _, v in sols)
            if g_best > run_best + 1e-12:
                run_best, since = g_best, 0
            else:
                since += 1
        restarts.append({"regime": regime, "popsize": lam, "sigma": sig, "evals": used, "gens": gens, "why": why,
                         "zero_frac": zero / used if used else np.nan})
        if regime == "small":
            small_used += used
        else:
            large_used += used
    return hist, restarts


# ---------------------------------------------------------------------------
# 3. 엔진 — 캐시(memmap) 위 평가
# ---------------------------------------------------------------------------

ARR_NAMES = ("O", "H", "L", "C", "MA20", "MA50", "ATR", "TR", "LIM", "strat_cs", "bs", "bn", "cost_tab", "days_ns",
             "month_of", "year_of", "F", "I", "off", "codes")


class Win:
    """창: 진입 행 ``[d_lo, d_hi)`` (사건 구간과 교집합), 포트폴리오 날 ``[t_lo, t_end)`` (``t_end`` 직전 거래일 종가 시가평가)."""

    def __init__(self, E: "Engine", lo: str, hi: str, end: "str | None"):
        self.lo, self.hi, self.end = lo, hi, end
        self.d_lo = max(E.row(lo), E.row(EVENT_LO))
        self.d_hi = min(E.row(hi), int(E.days.searchsorted(pd.Timestamp(EVENT_HI), side="right")))
        self.t_lo = self.d_lo
        self.t_end = min(E.row(end), E.end_row + 1) if end else E.end_row + 1


class Engine:
    """캐시 디렉터리의 배열(memmap)과 평가 함수. ``F`` 는 :meth:`set_rep` 로 모양 섞은 사본과 바꿔 끼운다(2b)."""

    def __init__(self, cache: str, mmap: bool = True):
        mode = "r" if mmap else None
        for nm in ARR_NAMES:
            setattr(self, nm, np.load(os.path.join(cache, f"{nm}.npy"), mmap_mode=mode, allow_pickle=False))
        with open(os.path.join(cache, "meta.json"), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.days = pd.DatetimeIndex(np.asarray(self.days_ns).astype("datetime64[ns]"))
        self.end_row = int(self.days.searchsorted(pd.Timestamp(END_DAY), side="right") - 1)
        self.F_orig = self.F
        self._Fa = self._Foa = np.asarray(self.F)        # njit 은 memmap 대신 같은 버퍼의 ndarray 뷰를 받는다
        self._Ia = np.asarray(self.I)
        self.rep = None
        self.fw_off = False                              # 리뷰 FW M6: "F1 끔" 짝(같은 config, 하락 쐐기 거르기만 끔)
        self.pmix = float(self.meta.get("p", 0.5))
        self.n_years = int(np.max(self.year_of)) + 1
        self.reform = self.row(PE.TICK_REFORM_DAY)
        mx = int(np.max(np.diff(self.off))) if len(self.off) > 1 else 0
        self._buf = {k: np.empty(max(mx, 1), t) for k, t in (("idx", np.int64), ("e", float), ("sp", float), ("fb", float),
                                                               ("amb", np.bool_), ("skip0", np.bool_), ("s0", float), ("buf", float))}
        mo = np.asarray(self.month_of)
        self.month_year = np.zeros(int(mo.max()) + 1, np.int64)
        self.month_year[mo] = self.days.year.to_numpy()
        ydays = pd.Series(1, index=self.days).groupby(self.days.year).sum()
        self.year_len = {int(y): float(max(v, 245)) for y, v in ydays.items()}
        xf = os.path.join(cache, "XF.npy")                # 13세대 사건 가족 표(event_families_build.py)
        self.fam_on = os.path.exists(xf) and len(self.off) > PE.N_TABLES + EF.N_FAM_TABLES
        self.NF = 3 if self.fam_on else 1                 # 폴드당 청산 행: W · P · C
        if self.fam_on:
            self.XF = np.load(xf, allow_pickle=False)
            self.fam_lo = int(self.off[EF.TID_P])

    def row(self, date_s: str) -> int:
        return int(self.days.searchsorted(pd.Timestamp(date_s)))

    def set_rep(self, rep: "int | None") -> None:
        if rep == self.rep:
            return
        self.F = self.F_orig if rep is None else PE.shuffle_shape(np.asarray(self.F_orig), np.asarray(self.off), rep,
                                                                  seed=NULLB_SEED)
        self._Fa = np.asarray(self.F)
        self.rep = rep

    # --- 후보 ---
    def candidates(self, g: dict, w: Win, *, par: int = 0, extra_slip: float = 0.0, allow=None, F=None) -> dict:
        tid = table_of(g)
        a, b = PE.table_slice(self.off, tid)
        dcol = self.I[a:b, PE.II["d"]]
        aa = a + int(np.searchsorted(dcol, w.d_lo))
        bb = a + int(np.searchsorted(dcol, w.d_hi))
        B = self._buf
        F_ = self.F if F is None else F
        n = PF.filter_events(F_, aa, bb, gene_vec(g, extra_slip), float(LIMIT_UP_RATIO), B["idx"], B["e"], B["sp"],
                             B["fb"], B["amb"], B["skip0"], B["s0"], B["buf"]) if bb > aa else 0
        idx = B["idx"][:n].copy()
        keep = np.ones(n, bool) if allow is None else np.asarray(allow)[idx]
        if FW and n and not self.fw_off:                 # F1: 하락 쐐기(유연) — 상단선 하락 ∧ 하단선 < fw_lo ∧ 수렴, 현재 F(2b 면 섞은 서술자)
            su_, sl_ = np.asarray(F_[idx, PE.FI["s_up"]], float), np.asarray(F_[idx, PE.FI["s_lo"]], float)
            keep &= (su_ < 0.0) & (sl_ < g["fw_lo"]) & (su_ < sl_)
        I_, F = self.I, self.F
        out = {"ev": idx[keep], "e": B["e"][:n][keep].copy(), "sp": B["sp"][:n][keep].copy(), "fb": B["fb"][:n][keep].copy(),
               "amb": B["amb"][:n][keep].copy(), "skip0": B["skip0"][:n][keep].copy(), "s0": B["s0"][:n][keep].copy(),
               "buf": B["buf"][:n][keep].copy(),
               "n_pass": int(keep.sum())}
        ev = out["ev"]
        out.update(d=I_[ev, PE.II["d"]].astype(np.int64), ci=I_[ev, PE.II["ci"]].astype(np.int64),
                   h0=np.asarray(F[ev, PE.FI["h0"]], float), strat=np.asarray(self.F_orig[ev, PE.FI["strat"]]).astype(np.int64),
                   mkt=np.asarray(self.F_orig[ev, PE.FI["market"]]).astype(np.int64), par=np.full(len(ev), par, np.int64),
                   pid=(I_[ev, PE.II["i0"]].astype(np.int64) * 100_000 + I_[ev, PE.II["iD"]].astype(np.int64)),
                   u=np.asarray(self.F_orig[ev, PE.FI["u_d"]], float), su=np.asarray(self.F_orig[ev, PE.FI["su"]], float),
                   lE=np.asarray(self.F_orig[ev, PE.FI["E"]], float), liE=I_[ev, PE.II["iE"]].astype(np.int64),
                   lsl=np.asarray(self.F_orig[ev, PE.FI["sl"]], float))
        return out

    def fam_candidates(self, g: dict, w: Win, fam: str, *, par: int, extra_slip: float = 0.0) -> dict:
        """13세대 가족 P·C 후보(EVENT_FAMILIES.md 정정 E-1) — :meth:`candidates` 와 같은 열."""
        tid = EF.TID_P if fam == "P" else EF.TID_C
        a, b = PE.table_slice(self.off, tid)
        dcol = self.I[a:b, PE.II["d"]]
        aa = a + int(np.searchsorted(dcol, w.d_lo))
        bb = a + int(np.searchsorted(dcol, w.d_hi))
        F_ = self.F_orig
        rows = np.arange(aa, bb)
        if len(rows):
            m = EF.gene_mask(self.XF[rows - self.fam_lo], np.asarray(F_[aa:bb]), g, fam, PE.FI)
            rows = rows[m]
        entry = g.get("p_entry", "open") if fam == "P" else g.get("c_entry", "open")
        slip = float(exit_vec(g)[PF.X_SLIP]) + extra_slip
        mk = np.asarray(F_[rows, PE.FI["market"]]).astype(np.int64) if len(rows) else np.zeros(0, np.int64)
        fx = EF.fills(rows, np.asarray(self.I), PE.II, self.O, self.H, self.L, self.C, self.TR, entry, g, slip,
                      float(LIMIT_UP_RATIO), mk, int(self.reform)) if len(rows) else \
            {k: np.zeros(0, t) for k, t in (("rows", np.int64), ("d", np.int64), ("ci", np.int64), ("e", float), ("sp", float),
                                            ("amb", bool), ("skip0", bool), ("s0", float))}
        ev = fx["rows"]
        n = len(ev)
        Fo, I_ = self.F_orig, self.I
        return {"ev": ev, "e": fx["e"], "sp": fx["sp"], "fb": np.full(n, -np.inf), "amb": fx["amb"], "skip0": fx["skip0"],
                "s0": fx["s0"], "buf": np.zeros(n), "n_pass": n, "d": fx["d"], "ci": fx["ci"],
                "h0": np.asarray(Fo[ev, PE.FI["h0"]], float), "strat": np.asarray(Fo[ev, PE.FI["strat"]]).astype(np.int64),
                "mkt": np.asarray(Fo[ev, PE.FI["market"]]).astype(np.int64), "par": np.full(n, par, np.int64),
                "pid": I_[ev, PE.II["i0"]].astype(np.int64) * 100_000 + I_[ev, PE.II["iD"]].astype(np.int64),
                "u": np.asarray(Fo[ev, PE.FI["u_d"]], float), "su": np.zeros(n), "lE": np.asarray(Fo[ev, PE.FI["E"]], float),
                "liE": I_[ev, PE.II["iE"]].astype(np.int64), "lsl": np.zeros(n)}

    def fam_exit(self, g: dict, fam: str, extra_slip: float = 0.0, sell_extra: float = 0.0) -> np.ndarray:
        """가족 청산 행 — W 청산 유전자 공유, 돌파 실패·반등 실패·재진입·하단선 동적 끔, 종가 진입이면 X_ENTRY_CLOSE·X_ON_HOLD."""
        px = exit_vec(g, extra_slip, sell_extra)
        px[PF.X_FB_DAYS] = 0
        px[PF.X_FAIL_EXIT] = 0
        px[PF.X_REENTRY] = 0
        px[PF.X_DYN] = 0
        if fam == "P" and g.get("p_entry") == "close":
            px[PF.X_ENTRY_CLOSE] = 1.0
            ph = g.get("p_hold")
            px[PF.X_ON_HOLD] = np.nan if ph is None else float(ph)
        return px

    def cands_all(self, g: dict, w: Win, *, k: int = 0, extra_slip: float = 0.0, sell_extra: float = 0.0,
                  allow=None) -> "tuple[list[dict], list[np.ndarray]]":
        """폴드 ``k`` 의 (후보 묶음, 청산 행) — 가족 캐시가 없으면 W 하나(기존과 비트 동일). 청산 행 번호 = k·NF + (W 0 · P 1 · C 2)."""
        if not self.fam_on:
            return [self.candidates(g, w, par=k, extra_slip=extra_slip, allow=allow)], [exit_vec(g, extra_slip, sell_extra)]
        cs = []
        if g.get("fam_w", 1):
            cs.append(self.candidates(g, w, par=3 * k, extra_slip=extra_slip, allow=allow))
        if g.get("fam_c", 0):
            cs.append(self.fam_candidates(g, w, "C", par=3 * k + 2, extra_slip=extra_slip))
        if g.get("fam_p", 0):
            cs.append(self.fam_candidates(g, w, "P", par=3 * k + 1, extra_slip=extra_slip))
        pxs = [exit_vec(g, extra_slip, sell_extra), self.fam_exit(g, "P", extra_slip, sell_extra),
               self.fam_exit(g, "C", extra_slip, sell_extra)]
        return cs, pxs

    # --- 포트폴리오 ---
    def run(self, cands: "list[dict]", pxs: "list[np.ndarray]", t_lo: int, t_end: int, *, cost_mult: float = COST_MULT,
            pmix: "float | None" = None, order_seed: "int | None" = None, exclude=None, cost_on_sell: bool = False) -> dict:
        cat = {k: np.concatenate([c[k] for c in cands]) if cands else np.zeros(0) for k in
               ("d", "ci", "e", "sp", "fb", "h0", "u", "su", "lE", "liE", "lsl", "buf", "amb", "skip0", "s0", "strat", "mkt", "par",
                "ev", "pid")}
        if self.fam_on and len(cat["d"]):               # 13세대: 같은 날 W → C → P(묶음 순서), 날짜 순 안정 정렬(W 단독이면 항등)
            o = np.argsort(cat["d"], kind="stable")
            cat = {k: v[o] for k, v in cat.items()}
        if exclude is not None and len(cat["ev"]):
            m = ~np.isin(cat["ev"], np.asarray(list(exclude), np.int64))
            cat = {k: v[m] for k, v in cat.items()}
        if order_seed is not None and len(cat["d"]):
            rk = np.random.default_rng(order_seed).random(len(cat["d"]))
            o = np.lexsort((rk, cat["d"]))
            cat = {k: v[o] for k, v in cat.items()}
        n = len(cat["d"])
        L = t_end - t_lo
        ret, bench, expo = np.zeros(L), np.zeros(L), np.zeros(L)
        rec = np.zeros((2 * max(n, 1), PF.N_Q))         # 재진입은 원 거래당 최대 1 — 기록 수 ≤ 2 × 후보
        ny = np.zeros(self.n_years, np.int64)
        if not pxs:
            pxs = [np.zeros(PF.N_X)]
            pxs[0][PF.X_SMAX] = np.nan
        fixed = [p.copy() for p in pxs]                   # 재진입 트레이드용 행(하단선 동적 끔 — 재진입 손절은 고정)
        for p in fixed:
            p[PF.X_DYN] = 0.0
        PX = np.ascontiguousarray(np.vstack(pxs + fixed))
        res = PF.portfolio(cat["d"].astype(np.int64), cat["ci"].astype(np.int64), cat["e"].astype(float),
                           cat["sp"].astype(float), cat["fb"].astype(float), cat["h0"].astype(float), cat["u"].astype(float),
                           cat["su"].astype(float), cat["lE"].astype(float), cat["liE"].astype(np.int64), cat["lsl"].astype(float),
                           cat["buf"].astype(float),
                           cat["amb"].astype(np.bool_), cat["skip0"].astype(np.bool_), cat["s0"].astype(float),
                           cat["strat"].astype(np.int64),
                           cat["mkt"].astype(np.int64), cat["par"].astype(np.int64), cat["ev"].astype(np.int64),
                           cat["pid"].astype(np.int64), PX, self.O, self.H, self.L, self.C, self.MA20, self.MA50, self.ATR,
                           self.TR, self.LIM, self.bs, self.bn, self.strat_cs, self.cost_tab, float(cost_mult),
                           float(self.pmix if pmix is None else pmix), int(t_lo), int(t_end), PF.SLOTS, PF.SLOT_W, ret,
                           bench, expo, rec, self.year_of, ny, int(self.reform), bool(cost_on_sell), float(LIMIT_UP_RATIO),
                           len(pxs))
        n_taken = int(res[0])
        return {"ret": ret, "bench": bench, "exc": ret - bench, "expo": expo, "rec": rec[:n_taken], "n": n_taken, "ny": ny,
                "sk_slot": int(res[1]), "sk_held": int(res[2]), "unmatched": int(res[3]), "n_cand": n, "t_lo": t_lo,
                "t_end": t_end}

    def monthly(self, x: np.ndarray, t_lo: int) -> "tuple[np.ndarray, np.ndarray]":
        mo = np.asarray(self.month_of[t_lo:t_lo + len(x)])
        u, inv = np.unique(mo, return_inverse=True)
        return np.bincount(inv, weights=x), u

    def year_frac(self, w: Win) -> dict:
        """창의 연도별 비율(그해 거래일 중 창 안 거래일, ≤ 1) — 창마다 한 번 계산해 둔다."""
        if getattr(w, "_frac", None) is None:
            yrs = self.days[w.t_lo:w.t_end].year
            cnt = pd.Series(1, index=yrs).groupby(level=0).sum()
            w._frac = {int(y): min(1.0, v / self.year_len[int(y)]) for y, v in cnt.items()}
        return w._frac

    def penalty(self, r: dict, w: Win) -> "tuple[float, dict]":
        frac = self.year_frac(w)
        y0 = int(self.days[0].year)
        F_ = FLOOR_PER_YEAR * sum(frac.values())
        pen = max(0.0, F_ - r["n"]) / F_ if F_ > 0 else 0.0
        py = [max(0.0, FLOOR_YEAR * f - r["ny"][y - y0]) / (FLOOR_YEAR * f) for y, f in frac.items() if f > 0]
        pen_y = float(np.mean(py)) if py else 0.0
        return LAMBDA * (pen + pen_y), {"floor": F_, "years": frac, "pen_total": pen, "pen_year": pen_y,
                                        "feasible": bool(pen == 0 and pen_y == 0)}

    def evaluate(self, g: dict, w: Win, *, seed: int = FIT_SEED, counts_only: bool = False, allow=None,
                 want_monthly: bool = False) -> dict:
        """학습창 적합도 — :meth:`evaluate_fast` (njit 한 번 호출, :meth:`evaluate_ref` 와 비트 동일)가 받는 경우는
        그쪽, 아니면(``allow``·``want_monthly``·``PATTERN_GA_FAST=0``) 원래 경로."""
        if allow is None and not want_monthly and FAST_EVAL and (FIT_MODE != "trade" or counts_only) and not self.fam_on:
            return self.evaluate_fast(g, w, seed=seed, counts_only=counts_only)
        return self.evaluate_ref(g, w, seed=seed, counts_only=counts_only, allow=allow, want_monthly=want_monthly)

    def _pen_arrays(self, w: Win) -> "tuple[float, np.ndarray, np.ndarray]":
        """:meth:`penalty` 의 창별 상수(하한 합·연도 번호·연도 비율 — dict 순서 그대로). 창마다 한 번."""
        if getattr(w, "_pen", None) is None:
            frac = self.year_frac(w)
            if len(frac) >= 128:                          # pattern_eval.year_mean_np 는 < 128 만 numpy 와 같다
                raise ValueError(f"창의 연도 수 {len(frac)} ≥ 128")
            y0 = int(self.days[0].year)
            w._pen = (FLOOR_PER_YEAR * sum(frac.values()), np.array([y - y0 for y in frac], np.int64),
                      np.array(list(frac.values()), float))
        return w._pen

    def evaluate_fast(self, g: dict, w: Win, *, seed: int = FIT_SEED, counts_only: bool = False) -> dict:
        """:meth:`evaluate_ref` 와 같은 값을 :func:`research.signals.pattern_eval.evaluate_kernel` 한 번으로."""
        tid = table_of(g)
        a, b = PE.table_slice(self.off, tid)
        dcol = self.I[a:b, PE.II["d"]]
        aa = a + int(np.searchsorted(dcol, w.d_lo))
        bb = a + int(np.searchsorted(dcol, w.d_hi))
        ffloor, yidx, yfrac = self._pen_arrays(w)
        B = self._buf
        out = np.empty(6)
        PEV.evaluate_kernel(self._Fa, self._Foa, self._Ia, aa, bb, gene_vec(g), float(LIMIT_UP_RATIO), B["idx"], B["e"],
                            B["sp"], B["fb"], B["amb"], B["skip0"], B["s0"], B["buf"], bool(FW and not self.fw_off),
                            float(g["fw_lo"]) if FW else 0.0, exit_vec(g), self.O, self.H, self.L, self.C, self.MA20,
                            self.MA50, self.ATR, self.TR, self.LIM, self.bs, self.bn, self.strat_cs, self.cost_tab,
                            float(COST_MULT), float(self.pmix), int(w.t_lo), int(w.t_end), PF.SLOTS, PF.SLOT_W,
                            self.year_of, self.n_years, int(self.reform), self.month_of, float(ffloor), yidx, yfrac,
                            float(FLOOR_YEAR), float(LAMBDA), bool(counts_only), 3, FIT_BOOT, int(seed), 2.5, out)
        r = {"n": int(out[0]), "n_pass": int(out[1]), "pen": float(out[2]), "feasible": bool(out[3])}
        if counts_only:
            return r
        r.update(fit=float(out[4]), lower=float(out[5]))
        return r

    def evaluate_ref(self, g: dict, w: Win, *, seed: int = FIT_SEED, counts_only: bool = False, allow=None,
                     want_monthly: bool = False) -> dict:
        """학습창 적합도(모듈 설명). ``counts_only`` = 진입 수·벌점만(수익 숫자를 만들지 않는다 — 실현 가능 비율용)."""
        cs, pxs_ = self.cands_all(g, w, allow=allow)
        c = {"n_pass": sum(x["n_pass"] for x in cs)}
        if counts_only and c["n_pass"] == 0:              # 선별 지름길 — 진입 0건(벌점만)
            r0 = {"n": 0, "ny": np.zeros(self.n_years, np.int64)}
            pen, pinfo = self.penalty(r0, w)
            return {"n": 0, "n_pass": 0, "pen": pen, "feasible": pinfo["feasible"]}
        r = self.run(cs, pxs_, w.t_lo, w.t_end)
        pen, pinfo = self.penalty(r, w)
        out = {"n": r["n"], "n_pass": c["n_pass"], "pen": pen, "feasible": pinfo["feasible"]}
        if counts_only:
            return out
        if FIT_MODE == "trade":                      # 11세대 후보: 거래당 슬롯 대비 순(2배 비용)의 주문일 군집 부트스트랩 10% 분위
            rec = r["rec"]
            if len(rec) >= 3:
                wb_ = rec[:, PF.Q_WB]
                wa_ = np.where(rec[:, PF.Q_ADDED] > 0.5, 1.0 - wb_, 0.0)
                net2 = rec[:, PF.Q_PNL] - 2.0 * (rec[:, PF.Q_CX] * wb_ + rec[:, PF.Q_CADD] * wa_)
                xs = net2 * np.minimum(1.0, rec[:, PF.Q_ALLOC] / PF.SLOT_W)
                # 검증 13차: 주문일 군집 부트스트랩(같은 날 진입 묶음), 10% 분위 하한
                _, gi = np.unique(rec[:, PF.Q_D].astype(np.int64), return_inverse=True)
                ng = int(gi.max()) + 1
                gs = np.bincount(gi, weights=xs, minlength=ng)
                gc = np.bincount(gi, minlength=ng).astype(float)
                rng = np.random.default_rng(seed)
                pick = rng.integers(0, ng, size=(FIT_BOOT, ng))
                bm = gs[pick].sum(axis=1) / gc[pick].sum(axis=1)
                lower = float(np.percentile(bm, 10.0))
            else:
                lower = -1.0
            out.update(fit=lower - pen, lower=lower)
            if want_monthly:
                out["monthly"] = self.monthly(r["exc"], w.t_lo)[0]
            return out
        mo, _ = self.monthly(r["exc"], w.t_lo)
        lower = float(np.percentile(gf.block_boot(np.ascontiguousarray(mo), 3, FIT_BOOT, seed), 2.5)) if len(mo) >= 6 else -1.0
        out.update(fit=lower - pen, lower=lower)
        if want_monthly:
            out["monthly"] = mo
        return out

    def stitched(self, cfgs: "list[dict]", tests: "list[Win]", *, extra_slip: float = 0.0, sell_extra: float = 0.0,
                 cost_mult: float = COST_MULT, pmix: "float | None" = None, order_seed: "int | None" = None, exclude=None,
                 cost_on_sell: bool = False) -> dict:
        """폴드별 config → 이어 붙인 OOS 연속 포트폴리오(폴드 시험 창 진입만 그 config, 열린 포지션은 원 config 청산)."""
        cands, pxs = [], []
        for k, (c, w) in enumerate(zip(cfgs, tests, strict=True)):
            cs_, px_ = self.cands_all(c, w, k=k, extra_slip=extra_slip, sell_extra=sell_extra)
            cands += cs_
            pxs += px_
        t_lo, t_end = tests[0].t_lo, tests[-1].t_end
        r = self.run(cands, pxs, t_lo, t_end, cost_mult=cost_mult, pmix=pmix, order_seed=order_seed, exclude=exclude,
                     cost_on_sell=cost_on_sell)
        mo, mid = self.monthly(r["exc"], t_lo)
        r["monthly"], r["month_ids"] = mo, mid
        r["ann"] = float(mo.mean() * 12) if len(mo) else np.nan
        r["fold_ann"], r["fold_n"] = [], []
        starts = [int(self.month_of[min(w.d_lo, len(self.days) - 1)]) for w in tests] + [10 ** 9]
        for k, w in enumerate(tests):
            sel = (mid >= starts[k]) & (mid < starts[k + 1])
            r["fold_ann"].append(float(mo[sel].mean() * 12) if sel.any() else np.nan)
            r["fold_n"].append(int(((r["rec"][:, PF.Q_D] >= w.d_lo) & (r["rec"][:, PF.Q_D] < w.d_hi)).sum()) if len(r["rec"]) else 0)
        years = self.month_year[mid] if len(mid) else np.zeros(0, np.int64)
        def ann(m):
            return float(mo[m].mean() * 12) if m.any() else np.nan

        r["calm_ann"] = ann(np.isin(years, (2022, 2023, 2024)))
        r["ex_bundles_ann"] = ann(~np.isin(years, (2020, 2021, 2025, 2026)))
        r["ex2021_ann"] = ann(years != 2021)
        r["ex2020_21_ann"] = ann(~np.isin(years, (2020, 2021)))
        r["ex2025_26_ann"] = ann(~np.isin(years, (2025, 2026)))
        r["expo_mean"] = float(np.mean(r["expo"])) if len(r["expo"]) else np.nan
        r["years"] = years
        return r


def fold_windows(E: Engine, folds) -> "list[tuple[Win, Win]]":
    """FOLDS → (학습 창: 진입 [train_lo, train_hi), test_lo 직전 시가평가), (시험 창: 진입 [test_lo, test_hi) ∩ 사건 구간)."""
    out = []
    for f in folds:
        out.append((Win(E, f.train_lo, f.train_hi, f.test_lo), Win(E, f.test_lo, f.test_hi, None)))
    return out


# ---------------------------------------------------------------------------
# 4. 작업자(spawn — 캐시는 memmap 으로 공유)
# ---------------------------------------------------------------------------

_W: dict = {}


def worker_init(cache: str, folds_spec: list) -> None:
    if gf.HAVE_NUMBA:
        gf.numba.set_num_threads(1)
    E = Engine(cache)
    from swing_it.validation.walkforward import Fold  # noqa: PLC0415
    folds = [Fold(*f) for f in folds_spec]
    _W.update(E=E, wins=fold_windows(E, folds))


def ga_run_one(E: Engine, w: Win, seed: int, budget: int, *, top: int = 0, x_restart=None) -> dict:
    """CMA 한 런(학습창 w) → 최적·적합도·재시작(진입 0건 표본 비율 포함)·고유 config 수·해시·(선택) 학습 적합도 상위 config.
    ``x_restart`` = BIPOP 재시작 평균 후보(그 학습창의 실현 가능 config 벡터 — 2단계 M2)."""
    memo: dict = {}
    tt = time.time()
    if FWL and (x_restart is None or len(x_restart) < 10):   # 리뷰 FWL M2: 재시작 풀 < 10 이면 x0 에서 재시작(실제·2b 같은 규칙)
        x_restart = x0_feasible()[None, :]
    if CORE_BIPOP:                                            # 코어 0.8 bipop — cma_bipop_x0 과 이력·memo·재시작 비트 동일
        from krx_quant_core.research.evolve import bipop  # noqa: PLC0415

        def keyfn(X):
            gs = [decode(x) for x in X]
            return [gkey(g) for g in gs], gs

        def ev(gs):
            rs = [E.evaluate(g, w) for g in gs]
            return [r["fit"] for r in rs], [r["n"] for r in rs]

        res = bipop(ev, budget=budget, seed=seed, x0=x0_feasible(), x_restart=x_restart, keyfn=keyfn,
                    n_cont=N_CONT, disc_sizes=DISC_SIZES)
        memo = res.memo                                       # 키 → (적합도, 진입 수, g) — 처음 본 순서
        hv, rst = res.hist_values, res.restarts
        best = decode(res.hist_x[res.best_index])             # 원래와 같이 이력 x 를 다시 복호화
    else:
        def fn(x):
            g = decode(x)
            k = gkey(g)
            if k not in memo:
                r = E.evaluate(g, w)
                memo[k] = (r["fit"], r["n"], g)
            return memo[k][0], memo[k][1]

        hist, rst = cma_bipop_x0(fn, N_CONT, DISC_SIZES, budget=budget, seed=seed, x0=x0_feasible(), x_restart=x_restart)
        hv = np.array([h[2] for h in hist])
        best = decode(hist[int(np.argmax(hv))][1])
    fit, n, _ = memo[gkey(best)]
    curve = np.maximum.accumulate(hv)
    hist = hv
    out = {"best": best, "fit": fit, "n": n, "restarts": rst, "uniq": len(memo), "sec": time.time() - tt,
           "n_eval": len(hist), "curve": {m: float(curve[m - 1]) for m in (100, 1000, 3000, 10000, len(curve)) if m <= len(curve)}}
    out["hashes"] = np.array([ghash(v[2]) for v in memo.values()], np.int64)
    if top:
        vals = sorted(memo.values(), key=lambda v: -v[0])[:top]
        out["top"] = [(v[0], v[2]) for v in vals]
    return out


def ga_task(t: dict) -> dict:
    E = _W["E"]
    E.set_rep(t.get("rep"))
    tr, _ = _W["wins"][t["fold"]]
    r = ga_run_one(E, tr, t["seed"], t["budget"], top=t.get("top", 0), x_restart=t.get("x_restart"))
    r.update(fold=t["fold"], seed=t["seed"], rep=t.get("rep"))
    return r


def procedure(E: Engine, wins: "list[tuple[Win, Win]]", bests: "list[list[dict]]") -> dict:
    """M2: 폴드마다 시드 최적을 새 난수(:data:`REEVAL_SEED`)로 학습창 재평가 → 1등 = 절차 출력."""
    outs, re_fits = [], []
    for (tr, _), bk in zip(wins, bests, strict=True):
        f = [E.evaluate(b, tr, seed=REEVAL_SEED)["fit"] for b in bk]
        outs.append(bk[int(np.argmax(f))])
        re_fits.append(f)
    return {"cfgs": outs, "re_fits": re_fits}


def random_x(rng: np.random.Generator, n: int) -> np.ndarray:
    """같은 메뉴 균등 추출(연속·로짓 U[0,1], 이산 단계 균등)."""
    X = rng.random((n, DIM))
    for q, k in enumerate(DISC_SIZES):
        X[:, N_CONT + q] = rng.integers(0, k, size=n)
    return X


def screen_chunk(E: Engine, w: Win, fold: int, chunk: int, n: int) -> "tuple[np.ndarray, np.ndarray]":
    """건수만 선별(2단계 M1·M2): 균등 추출 덩어리(seed [POOL_SEED, 폴드, 덩어리]) 중 학습창 실현 가능(벌점 0) config 의 벡터·진입 수.
    수익 숫자는 만들지 않는다. 실제·2b 가 같은 덩어리 seed 를 쓴다(자료만 다르다)."""
    X = random_x(np.random.default_rng([POOL_SEED, fold, chunk]), n)
    keep, nn = [], []
    for i in range(n):
        r = E.evaluate(decode(X[i]), w, counts_only=True)
        if r["feasible"]:
            keep.append(i)
            nn.append(r["n"])
    return X[keep], np.asarray(nn, np.int64)


def restart_pool(E: Engine, w: Win, fold: int, n_screen: int) -> np.ndarray:
    """BIPOP 재시작 평균 후보 = 앞 ``n_screen`` 개 선별의 실현 가능 config(실제·2b 같은 규칙, 각자 자료). 비면 빈 배열(→ 균등)."""
    out = [screen_chunk(E, w, fold, c, min(SCREEN_CHUNK, n_screen - c * SCREEN_CHUNK))[0]
           for c in range(math.ceil(n_screen / SCREEN_CHUNK))]
    return np.vstack(out) if out else np.zeros((0, DIM))


def screen_task(t: dict) -> dict:
    E = _W["E"]
    E.set_rep(t.get("rep"))
    tr, _ = _W["wins"][t["fold"]]
    X, nn = screen_chunk(E, tr, t["fold"], t["chunk"], t["n"])
    return {"fold": t["fold"], "chunk": t["chunk"], "X": X, "n": nn}


def fw_extras(E: "Engine", cfgs: "list[dict]", tests: "list[Win]", st: dict) -> dict:
    """리뷰 FW M6: 노출·불타기 발동률·"F1 끔" 짝(같은 config, 하락 쐐기 거르기만 끔) — 실제와 2b 각 벌에 같은 정의."""
    rec = st["rec"]
    add = float(np.mean(rec[:, PF.Q_ADDED] > 0.5)) if len(rec) else np.nan
    E.fw_off = True
    try:
        off = E.stitched(cfgs, tests)
    finally:
        E.fw_off = False
    return {"expo": st["expo_mean"], "add_rate": add, "f1off_ann": off["ann"], "f1off_expo": off["expo_mean"],
            "f1_gain": st["ann"] - off["ann"]}


def fw_adaptive_screen(ex, nf_: int, cap_n: int, r_chunks: int, workers: int, t0: float) -> "tuple[dict, dict]":
    """리뷰 FW M1: 폴드마다 덩어리 0, 1, 2, … 를 선별해 실현 가능 config 누적이 :data:`FW_FEAS_TARGET` 에 닿는 **가장 짧은 접두**
    (최소 ``r_chunks`` 덩어리, 상한 ``cap_n``)만 쓴다. 정지는 건수만 보고, 접두는 완료 순서와 무관하다(결정적)."""
    cap = max(r_chunks, math.ceil(cap_n / SCREEN_CHUNK))
    got = {k: {} for k in range(nf_)}
    nxt = {k: 0 for k in range(nf_)}
    inflight: dict = {}

    def prefix(k):
        tot, c = 0, 0
        while c in got[k]:
            tot += len(got[k][c]["X"])
            c += 1
            if tot >= FW_FEAS_TARGET and c >= r_chunks:
                return c, tot, True
        return c, tot, False

    def wants(k):
        return not prefix(k)[2] and nxt[k] < cap

    per = max(2, (3 * workers) // nf_)
    last = time.time()
    while True:
        for k in range(nf_):
            while wants(k) and sum(1 for v in inflight.values() if v[0] == k) < per:
                c = nxt[k]
                inflight[ex.submit(screen_task, {"fold": k, "chunk": c, "n": SCREEN_CHUNK})] = (k, c)
                nxt[k] += 1
        if not inflight:
            break
        done, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
        for f in done:
            k, c = inflight.pop(f)
            got[k][c] = f.result()
        if time.time() - last > 600:
            last = time.time()
            _log("FW 적응 선별: " + " · ".join(f"폴드{k + 1} {prefix(k)[1]}/{FW_FEAS_TARGET}({nxt[k] * SCREEN_CHUNK / 1e6:.0f}M)"
                                             for k in range(nf_)), t0)
    scr, restart_x = {}, {}
    for k in range(nf_):
        c_end = prefix(k)[0] if prefix(k)[2] else len(got[k])
        keep = [got[k][c] for c in range(c_end) if c in got[k]]
        scr[k] = keep
        restart_x[k] = np.vstack([r["X"] for r in keep[:r_chunks]]) if keep else np.zeros((0, DIM))
    _log("FW 적응 선별 끝: " + " · ".join(f"폴드{k + 1} 실현 가능 {sum(len(r['X']) for r in scr[k]):,}개 / "
                                       f"{len(scr[k]) * SCREEN_CHUNK / 1e6:.1f}M" for k in range(nf_)), t0)
    return scr, restart_x


def null_b_task(t: dict) -> dict:
    """(2b) 모양 섞은 널 한 벌 = 섞은 사건 표 위에서 절차 전체(재시작 풀 선별 → 폴드 × 시드 × 예산 → 재평가 1등 → 이어 붙인 OOS)
    + 모양 끔 짝. 재시작 풀은 실제와 같은 규칙(앞 :data:`RESTART_SCREEN` 개 선별)을 섞은 자료에서 다시 센다."""
    E = _W["E"]
    E.set_rep(t["rep"])
    wins = _W["wins"]
    tt = time.time()
    bests, rsts, zf, n_pool = [], [], [], []
    for k, (tr, _) in enumerate(wins):
        xr = restart_pool(E, tr, k, t["restart_screen"])
        n_pool.append(len(xr))
        bk = []
        for s in t["seeds"]:
            r = ga_run_one(E, tr, int(1000 * (t["rep"] + 1) + s), t["budget"], x_restart=xr)
            bk.append(r["best"])
            rsts.append(len(r["restarts"]))
            zf += [x["zero_frac"] for x in r["restarts"][1:]]
        bests.append(bk)
    pr = procedure(E, wins, bests)
    tests = [te for _, te in wins]
    st = E.stitched(pr["cfgs"], tests)
    so = E.stitched([shape_off(c) for c in pr["cfgs"]], tests)
    fwx = fw_extras(E, pr["cfgs"], tests, st) if FW else {}
    cu = [c["sup_c"] for c in pr["cfgs"] if c["sup_hw"] is not None]
    cl = [c["slo_c"] for c in pr["cfgs"] if c["slo_hw"] is not None]
    E.set_rep(None)
    return {"rep": t["rep"], "ann": st["ann"], "calm_ann": st["calm_ann"], "n_oos": st["n"], "shape_off_ann": so["ann"],
            "shape_gain": st["ann"] - so["ann"], "sup_c_sd": float(np.std(cu)) if len(cu) > 1 else np.nan,
            "slo_c_sd": float(np.std(cl)) if len(cl) > 1 else np.nan, "restart_pool": n_pool,
            "restart_zero_frac": float(np.nanmean(zf)) if zf else np.nan,
            "shapes": [config_shape(c) for c in pr["cfgs"]], "restarts_mean": float(np.mean(rsts)), "sec": time.time() - tt, **fwx}


def pool_task(t: dict) -> dict:
    """무작위 탐색 기술(2단계 M1-4): 폴드 학습창에서 균등 추출 config(실현 불가능 포함 — 탐색 흉내)를 한 번씩 완전 평가 → (적합도, 진입 수)."""
    E = _W["E"]
    E.set_rep(None)
    tr, _ = _W["wins"][t["fold"]]
    X = random_x(np.random.default_rng([POOL_SEED + 1, t["fold"], t["chunk"]]), t["n"])
    fit, n = np.empty(t["n"]), np.empty(t["n"], np.int64)
    for i in range(t["n"]):
        r = E.evaluate(decode(X[i]), tr)
        fit[i] = r["fit"]
        n[i] = r["n"]
    return {"fold": t["fold"], "chunk": t["chunk"], "X": X, "fit": fit, "n": n}


def landscape_task(t: dict) -> list:
    """민감도 지형: 폴드 출력 config 에서 한 유전자씩 범위 전체 — 학습 적합도·시험 연도 연환산(기술)."""
    E = _W["E"]
    E.set_rep(None)
    tr, te = _W["wins"][t["fold"]]
    x0 = encode(t["base"])
    rows = []
    for gi in range(DIM):
        grid = np.linspace(0, 1, 6) if gi < N_CONT else np.arange(DISC_SIZES[gi - N_CONT])
        for u in grid:
            x = x0.copy()
            x[gi] = u
            g = decode(x)
            if gkey(g) == gkey(t["base"]) and not np.isclose(u, x0[gi]):
                continue
            r = E.evaluate(g, tr)
            st = E.stitched([g], [te])
            rows.append({"fold": t["fold"], "dim": gi, "u": float(u), "train_fit": r["fit"], "train_n": r["n"],
                         "oos_ann": st["ann"], "oos_n": st["n"], "is_base": bool(np.isclose(u, x0[gi]))})
    return rows


# ---------------------------------------------------------------------------
# 5. 캐시 만들기(DB — 한 번)
# ---------------------------------------------------------------------------

def build_cache(cache: str, t0: float, *, with_p: bool) -> dict:
    """DB → 가격·맥락·층·바스켓·비용 배열 + 사건 표 72개 → ``cache`` (.npy). 분봉 p 는 ``with_p`` 일 때만(전체 모드)."""
    import vcp_retest_run as R  # noqa: PLC0415
    from sepa_momentum import MOM_N_Q, day_market_median_half, day_market_quantiles, momentum_12_1  # noqa: PLC0415

    from research.signals import vcp as V  # noqa: PLC0415
    from swing_it.features.technical import moving_average  # noqa: PLC0415

    os.makedirs(cache, exist_ok=True)
    ctx = R.Ctx(t0)
    A, days, codes = ctx.A, ctx.days, ctx.codes
    p = ctx.p
    close = p.close.reindex(index=days, columns=codes)
    hi_, lo_ = p.high.reindex(index=days, columns=codes), p.low.reindex(index=days, columns=codes)
    m20 = moving_average(close, 20)
    m150, m200 = moving_average(close, 150).to_numpy(float), moving_average(close, 200).to_numpy(float)
    prev = close.shift(1).to_numpy(float)
    hv_, lv_ = hi_.to_numpy(float), lo_.to_numpy(float)
    with np.errstate(invalid="ignore"):
        tr_ = np.fmax(np.fmax(hv_ - lv_, np.abs(hv_ - prev)), np.abs(lv_ - prev))
    atr = pd.DataFrame(np.where(np.isfinite(hv_ - lv_), tr_, np.nan)).rolling(20, min_periods=20).mean()
    uni = ctx.universe(ADV_FLOOR)
    with np.errstate(invalid="ignore"):
        align = (ctx.ma50 > m150) & (m150 > m200)
    base = uni.to_numpy(bool) & align & ~ctx.split["recent"]
    mom = momentum_12_1(p.close)
    q = day_market_quantiles(mom, uni, p.market, n_q=MOM_N_Q).reindex(index=days, columns=codes).to_numpy(float)
    a2 = day_market_median_half(ctx.D.adv20, uni, p.market).reindex(index=days, columns=codes).to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(a2) & uni.to_numpy(bool)
    strat = np.where(ok, ctx.mkt[None, :] * 20 + np.nan_to_num(q).astype(np.int64) * 2 + np.nan_to_num(a2).astype(np.int64), -1)
    del mom
    _log(f"맥락·층 패널 — 기본 맥락 종목-일 {int(base.sum()):,}", t0)
    T = lambda x: np.ascontiguousarray(np.asarray(x).T)  # noqa: E731
    month_of = ((days.year - days[0].year) * 12 + days.month - 1).to_numpy().astype(np.int64)
    year_of = (days.year - days[0].year).to_numpy().astype(np.int64)
    ctxd = {"base": T(base), "above50": T(np.asarray(ctx.p.close.reindex(index=days, columns=codes).to_numpy(float) > ctx.ma50)),
            "tt": T(ctx.tt), "rs": T(ctx.rs), "breadth": T(ctx.br), "adv20": T(ctx.adv), "strat": T(strat),
            "market": ctx.mkt.astype(np.int64), "month": month_of,
            "tick_reform_row": int(days.searchsorted(pd.Timestamp(PE.TICK_REFORM_DAY))), "atr": T(atr.to_numpy(float))}
    d_lo = int(days.searchsorted(pd.Timestamp(EVENT_LO)))
    d_hi = int(days.searchsorted(pd.Timestamp(EVENT_HI), side="right") - 1)
    tb = PE.build_tables(A.H, A.L, A.C, ctx.VOL, A.O, A.TR, ctxd, d_lo=d_lo, d_hi=d_hi, log=lambda m: _log(m, t0),
                         hygiene=HYGIENE_FWM if FWM else None)
    F, I_ = tb["F"], tb["I"]
    ci, s = I_[:, PE.II["ci"]], I_[:, PE.II["d"]] - 1
    Ee = V.e_accel_asof(ctx.est, codes[ci], days[s])
    F[:, PE.FI["e_yoy"]] = Ee["e_yoy"].to_numpy(float)
    F[:, PE.FI["e_yoy_prev"]] = Ee["e_yoy_prev"].to_numpy(float)
    F[:, PE.FI["e_q"]] = Ee["e_netinc"].to_numpy(float)            # FWM M1: 최신 분기 단독 순이익(그날 알던 것)
    F[:, PE.FI["e_q_prev"]] = Ee["e_netinc_prior"].to_numpy(float)  # 정정 FWM-1 M9(사용자 확인): 전년 같은 분기 단독 순이익
    rg = pd.read_csv(REGIME_CSV, usecols=["date", "market", "breadth_ma200", "nh_nl_ratio", "breakout_success20"])   # FWM M7
    rg = rg[rg["market"].isin(["KOSPI", "KOSDAQ"])].assign(date=lambda x: pd.to_datetime(x["date"]))
    key = pd.DataFrame({"date": days[s], "market": np.where(ctx.mkt[ci] == 1, "KOSDAQ", "KOSPI")})
    mg = key.merge(rg, on=["date", "market"], how="left")
    F[:, PE.FI["mkt_br200"]] = mg["breadth_ma200"].to_numpy(float)   # 리뷰 FWM M6: 지수 대용(KOSDAQ 편향) 대신 MA200 위 비율
    F[:, PE.FI["mkt_nhnl"]] = mg["nh_nl_ratio"].to_numpy(float)
    F[:, PE.FI["mkt_bos"]] = mg["breakout_success20"].to_numpy(float)
    m50 = np.asarray(ctx.ma50, float)                              # 진화 2세대: d−1 MA50 > 20 거래일 전(날짜×종목)
    s20 = np.maximum(s - 20, 0)
    with np.errstate(invalid="ignore"):
        flows_meta = _fill_flows(F, ctx, days, codes, ci, s)         # 9세대 수급 열
        oo1, cc1 = np.asarray(A.O)[ci, s], np.asarray(A.C)[ci, s]     # 9세대: d−1 봉 몸통 · d − 행(E)
        F[:, PE.FI["body_s"]] = np.where(oo1 > 0, cc1 / np.where(oo1 > 0, oo1, 1.0) - 1.0, np.nan)
        vol_ = np.nan_to_num(np.asarray(ctx.VOL, float))            # 10세대: H0 거래량 배수(종목×날짜)
        cs_ = np.concatenate([np.zeros((vol_.shape[0], 1)), np.cumsum(vol_, axis=1)], axis=1)
        i0v = I_[:, PE.II["i0"]]
        lo_ = np.maximum(i0v - 20, 0)
        m20_ = (cs_[ci, i0v] - cs_[ci, lo_]) / np.maximum(i0v - lo_, 1)
        F[:, PE.FI["h0_volx"]] = np.where((i0v >= 20) & (m20_ > 0), vol_[ci, i0v] / np.where(m20_ > 0, m20_, 1.0), np.nan)
        iE_ = I_[:, PE.II["iE"]]
        F[:, PE.FI["e_age"]] = np.where(iE_ >= 0, (s + 1 - iE_).astype(float), np.nan)
        m20v = m20.to_numpy(float)                                   # 3세대: d−1 MA20 > 5 거래일 전
        s5 = np.maximum(s - 5, 0)
        F[:, PE.FI["ma20_up"]] = np.where(np.isfinite(m20v[s, ci]) & np.isfinite(m20v[s5, ci]), (m20v[s, ci] > m20v[s5, ci]).astype(float), np.nan)
        i0 = I_[:, PE.II["i0"]]                                      # 3세대: H0 봉 윗꼬리 = (고 − max(시, 종)) / (고 − 저)
        hh_, oo_, cc_, ll_ = (np.asarray(x)[ci, i0] for x in (A.H, A.O, A.C, A.L))
        rng_ = hh_ - ll_
        F[:, PE.FI["h0_wick"]] = np.where(rng_ > 0, (hh_ - np.maximum(oo_, cc_)) / np.where(rng_ > 0, rng_, 1.0), np.nan)
        F[:, PE.FI["ma50_up"]] = np.where(np.isfinite(m50[s, ci]) & np.isfinite(m50[s20, ci]), (m50[s, ci] > m50[s20, ci]).astype(float), np.nan)
    end_row = int(days.searchsorted(pd.Timestamp(END_DAY), side="right") - 1)
    LIM = np.minimum(A.CUT, end_row + 1).astype(np.int64)
    strat_cs = T(strat).astype(np.int64)
    bs, bn = PF.basket_tables(A.O, A.C, strat_cs, 40)
    arrs = {"O": A.O, "H": A.H, "L": A.L, "C": A.C, "MA20": T(m20.to_numpy(float)), "MA50": A.MA, "ATR": T(atr.to_numpy(float)),
            "TR": A.TR, "LIM": LIM, "strat_cs": strat_cs, "bs": bs, "bn": bn, "cost_tab": R.cost_table(days),
            "days_ns": days.values.astype("datetime64[ns]").astype(np.int64), "month_of": month_of, "year_of": year_of,
            "F": F, "I": I_, "off": tb["off"], "codes": np.asarray(codes.astype(str), dtype="<U12")}
    meta = {"version": CACHE_VERSION, "code_hash": code_hash(), "built": date.today().isoformat(), "adv_floor": ADV_FLOOR, "event": [EVENT_LO, EVENT_HI],
            "end_day": END_DAY, "n_events": int(len(F)), "n_scan_rows": tb["n_scan_rows"], "n_cut_stocks": ctx.n_cut_stocks,
            "funnel": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in tb["funnel"].items()}, "base_rows": int(base.sum()),
            "flows": flows_meta}
    if with_p:
        pinfo = R._estimate_p(ctx.con, A, uni, days, None, band_lo=0.90, min_age=0, params={"ref_order": True})
        meta.update(p=pinfo["p"], p_ci=list(pinfo.get("ci", (np.nan, np.nan))), p_n=pinfo["n_classified"],
                    p_counts=pinfo.get("counts", {}), p_window=pinfo.get("window"))
    write_cache(cache, arrs, meta)
    return meta


def _fill_flows(F, ctx, days, codes, ci, s) -> None:
    """9세대: supply_demand → d−1(행 s)까지 N일(5·20) (기관+외국인)·개인 순매수 수량 합 / 거래량 합. 그날 수급은 장 마감 뒤 확정 → d 주문에 쓴다.
    N일 중 80% 이상 자료가 있을 때만 값(아니면 NaN — 거르기는 '모르면 통과')."""
    q = pd.read_sql("SELECT code, date, individual, foreign_, institution, acc_trde_qty, source FROM supply_demand", ctx.con,
                    parse_dates=["date"])                       # (code, date) 는 기본키 — 중복 없음
    q["code"] = q["code"].astype(str)
    q = q[q["code"].isin(set(map(str, codes)))]
    def wide(col):
        return q.pivot_table(index="date", columns="code", values=col, aggfunc="last").reindex(index=days, columns=codes)
    smart = wide("foreign_").add(wide("institution"), fill_value=np.nan)
    indiv, vol = wide("individual"), wide("acc_trde_qty")
    for n_, cs, ci_ in ((5, "fl_smart5", "fl_indiv5"), (20, "fl_smart20", "fl_indiv20")):
        mp_ = int(0.8 * n_)
        v = vol.rolling(n_, min_periods=mp_).sum()
        v = v.where(v > 0)
        rs_ = (smart.rolling(n_, min_periods=mp_).sum() / v).to_numpy(float)
        ri_ = (indiv.rolling(n_, min_periods=mp_).sum() / v).to_numpy(float)
        F[:, PE.FI[cs]] = rs_[s, ci]
        F[:, PE.FI[ci_]] = ri_[s, ci]
    yr = q["date"].dt.year                                          # 검증 11차 MINOR: 캐시 meta 에 수급 자료 시점·피복 기록
    return {"max_date": str(q["date"].max().date()) if len(q) else None,
            "codes_by_year": {int(k): int(v) for k, v in q.groupby(yr)["code"].nunique().items()},
            "rows_by_source": {str(k): int(v) for k, v in q["source"].value_counts().items()}}


def write_cache(cache: str, arrs: dict, meta: dict) -> None:
    """배열(:data:`ARR_NAMES`)·메타를 ``cache`` 에 .npy·json 으로(작업자가 memmap 으로 읽는다)."""
    os.makedirs(cache, exist_ok=True)
    for k in ARR_NAMES:
        np.save(os.path.join(cache, f"{k}.npy"), np.ascontiguousarray(arrs[k]), allow_pickle=False)
    with open(os.path.join(cache, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)


def code_hash() -> str:
    """사건 표·평가 커널 소스 해시(2단계 MINOR 11) — 캐시 meta 와 대조해 낡은 표로 돌지 않게."""
    h = hashlib.sha256()
    with open(PE.__file__, "rb") as fh:               # 가속(2026-10-11): 캐시 내용은 사건 표(PE)와 캐시 빌더만 정한다 —
        h.update(fh.read())                           # 평가 커널(PF) 수정으로 캐시를 다시 만들지 않는다
    import inspect  # noqa: PLC0415
    h.update(inspect.getsource(build_cache).encode())
    h.update(inspect.getsource(PF.basket_tables).encode())          # 검증 MINOR: 캐시 빌더가 부르는 PF 함수
    return h.hexdigest()[:16]


def meta_hash(cache: str) -> str:
    with open(os.path.join(cache, "meta.json"), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def ensure_cache(cache: str, t0: float, *, with_p: bool) -> None:
    mp_ = os.path.join(cache, "meta.json")
    meta = json.load(open(mp_, encoding="utf-8")) if os.path.exists(mp_) else {}
    if "families" in meta and not (meta.get("version") == CACHE_VERSION and meta.get("code_hash") == code_hash()):
        raise SystemExit(f"사건 가족 캐시 {cache} 는 기준 캐시와 링크를 공유한다 — 재빌드 금지(event_families_build.py 로 다시 만든다)")
    if meta.get("version") == CACHE_VERSION and meta.get("code_hash") == code_hash() and all(os.path.exists(os.path.join(cache, f"{k}.npy")) for k in ARR_NAMES):
        if with_p and "p" not in meta:
            import vcp_retest_run as R  # noqa: PLC0415
            ctx = R.Ctx(t0)
            uni = ctx.universe(ADV_FLOOR)
            pinfo = R._estimate_p(ctx.con, ctx.A, uni, ctx.days, None, band_lo=0.90, min_age=0, params={"ref_order": True})
            meta.update(p=pinfo["p"], p_ci=list(pinfo.get("ci", (np.nan, np.nan))), p_n=pinfo["n_classified"],
                        p_counts=pinfo.get("counts", {}), p_window=pinfo.get("window"))
            with open(mp_, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=1, default=float)
        _log(f"캐시 재사용 {cache} (사건 {meta['n_events']:,})", t0)
        return
    build_cache(cache, t0, with_p=with_p)


# ---------------------------------------------------------------------------
# 6. 원장
# ---------------------------------------------------------------------------

def _record_design(out_dir: str) -> None:
    """MINOR 10·11 — 리뷰가 센 설계 진단(건수만)과 '차트 열람 → 정의 수정 → 같은 데이터 재시험' 사슬 한 줄."""
    from swing_it.diagnostics.trials import record_trial  # noqa: PLC0415
    for v_ in ("θ 3·5·8% × 천장 창 120·260 × 기준선(추세선·수평) 문법 사건 수", "무작위 설정 3,000 실현 가능성(사전등록 원 범위)"):
        record_trial(LABEL, {"prereg": DESIGN_PREREG, "design_diagnostic": v_, "evaluated": "건수만, 성과 미계산",
                             "source": "REVIEW_PATTERN_GA.md 1단계 0. 실현 가능성 실측"}, logs_dir=out_dir,
                     note=f"패턴 GA 설계 진단(건수만): {v_}")
    record_trial(LABEL, {"prereg": DESIGN_PREREG, "chain": "차트 열람 → 정의 수정 → 같은 데이터 재시험: VCP → 쐐기 → 쐐기 D → 패턴 GA",
                         "evaluated": "기록만"}, logs_dir=out_dir,
                 note="MINOR 10: 이 계열 차트 열람 → 정의 수정 → 같은 KRX 데이터 재시험 사슬(VCP → 쐐기 → 쐐기 D → 패턴 GA)")


def _proc_config(budget: int, seeds: tuple, fac_sha: str, pmix: float) -> dict:
    return {"prereg": PREREG_DOC, "universe": f"위생 ∧ ADV20 ≥ {ADV_FLOOR:,.0f} 백만원 ∧ d−1 MA50>MA150>MA200 ∧ 분할 60일 제외",
            "event": [EVENT_LO, EVENT_HI], "end_mark": END_DAY, "tables": f"θ {PE.THETAS} × 창 {PE.CEIL_WINS} × 기준선 {PE.BASES}",
            "hygiene": {**PE.HYGIENE, **(HYGIENE_FWM if FWM else {})}, "genes": {"cont": [list(c[:3]) for c in CONT], "nominal": [[n, list(map(str, v))] for n, v in NOM],
                                              "disc": [[n, list(map(str, v))] for n, v in DISC]},
            "fitness": f"10 슬롯 포트폴리오 − 종목별 모멘텀 정합 바스켓, 월합 3개월 블록 부트스트랩 {FIT_BOOT} 2.5% 하단 − "
                       f"λ{LAMBDA}·(연 {FLOOR_PER_YEAR:g} 하한·연도별 {FLOOR_YEAR:g}) 연속 벌점, 비용 {COST_MULT:g}배, p 혼합 {pmix:.3f}",
            "optimizer": f"CMAwM+BIPOP(cmaes 0.13.1) 시드 {len(seeds)} × {budget}, 첫 평균 = 실현 가능점, 재평가 seed {REEVAL_SEED}",
            "folds": "swing_it.validation.walkforward.FOLDS(6), 학습창 끝 시가평가", "factors_sha": fac_sha}


# ---------------------------------------------------------------------------
# 7. 실현 가능성(건수만) — FEASIBILITY.md
# ---------------------------------------------------------------------------

def feasibility(E: Engine, wins, folds, args, t0: float) -> dict:
    """사건 표 건수·폴드×연도 건수·맥락 거르기 공백·무작위 설정 실현 가능 비율·착상 영역 학습창 연 체결 수·분봉 커버리지.
    **수익 숫자는 만들지 않는다**(포트폴리오는 슬롯·보유 거르기의 진입 수를 세는 데만 — 수익 열은 버린다)."""
    F, I_, off = np.asarray(E.F), np.asarray(E.I), np.asarray(E.off)
    days = E.days
    yr = days[I_[:, PE.II["d"]]].year.to_numpy()
    tid = I_[:, PE.II["table"]]
    base = tid % len(PE.BASES)
    res: dict = {"tables": [], "years": sorted(set(yr.tolist()))}
    for t in range(PE.N_TABLES):
        a, b = PE.table_slice(off, t)
        th, cw, bn = PE.table_key(t)
        k = f"{th}|{cw}|{bn}"
        fu = E.meta["funnel"].get(k, {})
        cnt = pd.Series(yr[a:b]).value_counts().to_dict()
        res["tables"].append({"theta": th, "cw": cw, "base": bn, "n": b - a, "h0in": int((F[a:b, PE.FI["h0in"]] == 1).sum()),
                              "years": {int(y): int(v) for y, v in cnt.items()}, "funnel": fu})
    # 폴드 × 연도(학습·시험) — 기준 표(θ 5% · 창 260) 기준선별 + 맥락 거르기(breadth > 0.5, d−1 TT, H0 일 TT)
    ctx_rows = []
    for bi, bn in enumerate(PE.BASES):
        a, b = PE.table_slice(off, PE.table_id(PE.THETAS.index(0.05), PE.CEIL_WINS.index(260), bi))
        sub = F[a:b]
        y = yr[a:b]
        for nm, m in (("전부", np.ones(b - a, bool)), ("breadth ≥ 0.5", sub[:, PE.FI["breadth"]] >= 0.5),
                      ("d−1 추세 템플릿", sub[:, PE.FI["tt_s"]] > 0.5), ("H0 일 추세 템플릿", sub[:, PE.FI["tt_h0"]] > 0.5),
                      ("RS ≥ 70", sub[:, PE.FI["rs"]] >= 70)):
            ctx_rows.append({"base": bn, "filter": nm, **{int(yy): int(((y == yy) & m).sum()) for yy in res["years"]}})
    res["ctx_years"] = ctx_rows
    # 착상 영역(정정 1 C2 + 정정 2 + 정정 3) — 폴드 학습창 연 체결(통과 사건 · 슬롯 진입 수), 가장 느슨한 나머지 유전자
    rw, rwd = event_region(F, base)
    loose = {"rs_min": None, "breadth_min": None, "sup_c": None, "sup_hw": None, "slo_c": None, "slo_hw": None, "rho_max": None,
             "span_c": None, "span_hw": None, "ceil_min": None, "depth_max": None, "dry_max": None, "runup_min": None,
             "runup_win": None, "rest_gap_max": None, "gap_max": 0.05, "stop_cap": 0.12, "fb_tol": 0.0, "trail_k": None,
             "stage": 0, "trail": "ma50", "pyramid": "off", "touch_lo": 2, "e_min": None, "e_accel": None, "stop_anchor": "E",
             "fb_days": 2, "be_r": math.inf, "take_r": 2.0, "take_frac": 0.5, "tighten": None, "ceil_kind": "single",
             **({"fw_lo": 0.15} if FW else {})}                       # 리뷰 FW M8: 가장 느슨한 경계
    reg_rows = []
    for rname, mask, entry, h0in in (("W", rw, "trend", 0), ("W", rw, "trend", 1), ("WD", rwd, "bottom", 1)):
        for th in PE.THETAS:
            for cw in PE.CEIL_WINS:
                g = canonical({**loose, "theta": th, "ceil_win": cw, "entry": entry, "h0_in_pattern": h0in})
                row = {"region": rname, "h0in": h0in, "theta": th, "cw": cw}
                for k, (tr, _) in enumerate(wins):
                    r = E.evaluate(g, tr, counts_only=True, allow=mask)
                    yrs = sum(E.penalty({"n": 0, "ny": np.zeros(E.n_years, np.int64)}, tr)[1]["years"].values())
                    row[f"f{k}_pass_yr"] = r["n_pass"] / yrs
                    row[f"f{k}_taken_yr"] = r["n"] / yrs
                    row[f"f{k}_feasible"] = r["feasible"]
                reg_rows.append(row)
    res["region"] = reg_rows
    res["region_verdict"] = {}
    for rname in ("W", "WD"):
        rows = [r for r in reg_rows if r["region"] == rname]
        res["region_verdict"][rname] = [bool(any(r[f"f{k}_feasible"] for r in rows)) for k in range(len(wins))]
    # 무작위 설정 실현 가능 비율(같은 메뉴 균등 — 건수만)
    rng = np.random.default_rng([POOL_SEED, 999])
    X = random_x(rng, args.n_feas)
    t1 = time.time()
    rows = []
    for i in range(len(X)):
        g = decode(X[i])
        row = {"entry": g["entry"]}
        for k, (tr, _) in enumerate(wins):
            r = E.evaluate(g, tr, counts_only=True)
            row[f"f{k}_n"], row[f"f{k}_pass"], row[f"f{k}_feas"] = r["n"], r["n_pass"], r["feasible"]
        rows.append(row)
    sec = (time.time() - t1) / max(1, len(X) * len(wins))
    res["random"] = pd.DataFrame(rows)
    res["sec_per_eval_counts"] = sec
    # 평가 1회 시간(적합도 전부 — 부트스트랩 포함, 숫자는 버린다)
    t1 = time.time()
    m = min(300, len(X))
    for i in range(m):
        E.evaluate(decode(X[i]), wins[-1][0])
    res["sec_per_eval_fit"] = (time.time() - t1) / max(1, m)
    # 분봉 커버리지(10억~50억 · ≥ 50억, 2025-08+ 사건 (종목, d) 고유)
    res["minute"] = minute_coverage(E, args) if not args.no_db else {}
    return res


def minute_coverage(E: Engine, args) -> dict:
    """M7: 분봉 구간 사건(전 표 합집합 (종목, 주문일))의 ADV20 구간별 분봉 유무 건수."""
    from prop_swing_common import load_env_db  # noqa: PLC0415

    from swing_it.storage import connect, db_default  # noqa: PLC0415
    F, I_ = np.asarray(E.F), np.asarray(E.I)
    d = I_[:, PE.II["d"]]
    lo = E.row(MINUTE_LO)
    m = d >= lo
    u = pd.DataFrame({"ci": I_[m, PE.II["ci"]], "d": d[m], "adv": F[m, PE.FI["adv20"]]}).drop_duplicates(["ci", "d"]).reset_index(drop=True)
    u["band"] = np.where(u["adv"] >= 5_000, "≥50억", "10억~50억")
    load_env_db()
    con = connect(db_default())
    has = np.zeros(len(u), bool)
    codes = np.asarray(E.codes)
    q = "SELECT DISTINCT code FROM minute_bars WHERE ts >= %(lo)s AND ts < %(hi)s AND code = ANY(%(c)s)"
    for dd, gidx in u.groupby("d").groups.items():
        day = E.days[int(dd)]
        cs = [str(codes[c]) for c in u.loc[gidx, "ci"]]
        got = set(pd.read_sql_query(q, con, params={"lo": day.strftime("%Y-%m-%d"),
                                                     "hi": (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "c": cs})["code"].astype(str))
        has[np.asarray(gidx)] = [c in got for c in cs]
    u["has"] = has
    return {b: {"n": int(len(g)), "minute": int(g["has"].sum())} for b, g in u.groupby("band")}


def write_feasibility(path: str, res: dict, E: Engine, wins, args, elapsed: float, est: dict) -> None:
    yrs = res["years"]
    L = [f"# 패턴 GA — 실현 가능성(건수만, 성과 미계산) {date.today().isoformat()}", "",
         f"`pattern_ga_run.py --feasibility-only` ({PREREG_DOC}). 사건(주문일) {EVENT_LO}~{EVENT_HI}, 유니버스 ADV20 ≥ 10억. "
         "포트폴리오는 슬롯·보유 거르기의 **진입 수를 세는 데만** 썼고, 수익·청산 사유·손절 여부는 어디에도 적지 않는다. "
         f"실행 {elapsed / 60:.1f}분. 사건 표 {E.meta['n_events']:,}행(훑은 종목-일 {E.meta['n_scan_rows']:,}, 기본 맥락 종목-일 "
         f"{E.meta['base_rows']:,}).", "",
         "## 1. 사건 표 건수 (θ × 천장 창 × 기준선) — 첫 관통 → 위생 → 맥락", "",
         "| θ | 창 | 기준선 | 첫 관통 | 간격<3 (그중 D→E 만) | 길이<15 | 무결성 상/하 | 종가 여유 | 바닥 >5일 | 맥락(d−1) | **사건** | H0 첫 점 | " + " | ".join(str(y) for y in yrs) + " |",
         "|" + "---|" * (12 + len(yrs))]
    for t in res["tables"]:
        fu = t["funnel"]
        L.append(f"| {t['theta']:.0%} | {t['cw']} | {t['base']} | {fu.get('first_cross', 0):,} | {fu.get('spacing<3', 0):,} "
                 f"({fu.get('spacing_DE_only', 0):,}) | "
                 f"{fu.get('span<15', 0):,} | {fu.get('integrity_upper', 0):,}/{fu.get('integrity_lower', 0):,} | "
                 f"{fu.get('close>order*0.995', 0):,} | {fu.get('bottom_age>5', 0):,} | {fu.get('context(d-1)', 0):,} | "
                 f"**{t['n']:,}** | {t['h0in']:,} | " + " | ".join(f"{t['years'].get(y, 0):,}" for y in yrs) + " |")
    L += ["", "## 2. 맥락 거르기별 연도 건수(θ 5% · 창 260) — 폴드 공백 점검(쐐기 리뷰 F)", "",
          "| 기준선 | 거르기 | " + " | ".join(str(y) for y in yrs) + " |", "|" + "---|" * (2 + len(yrs))]
    for r in res["ctx_years"]:
        L.append(f"| {r['base']} | {r['filter']} | " + " | ".join(f"{r.get(y, 0):,}" for y in yrs) + " |")
    L += ["", "## 3. 사용자 착상 영역 — 폴드 학습창 연 체결 수(정정 1 C2 + 정정 2 + 정정 3, 쐐기 리뷰 F)", "",
          f"영역 W = 하락 쐐기(ŝ_up < 0 ∧ ŝ_lo < 0 ∧ ŝ_up < ŝ_lo) ∧ 상단 추세선 ∧ 60일 급등 ≥ {REGION_RUNUP:.0%} ∧ 쉼 ≤ {REGION_REST:.0f}일"
          "(fb 2 · H0 첫 점 0/1 둘 다), WD = 같은 모양·맥락 ∧ H0 = 첫 점 ∧ 바닥 반전. 나머지 유전자는 가장 느슨하게(단계 0, 거르기 끔, "
          "갭 5%, 손절 상한 12%, 라이브 청산 꼴). '통과' = 거르기·체결 조건 통과 사건(슬롯 전 상한), '진입' = 10 슬롯·보유 거르기 뒤. "
          f"**실현 가능** = 학습창 연 {FLOOR_PER_YEAR:g}건·연도별 {FLOOR_YEAR:g}건 하한(비율 보정) 통과.", "",
          "| 영역 | H0 첫 점 | θ | 창 | " + " | ".join(f"폴드{k + 1} 통과/진입 연" for k in range(len(wins))) + " |",
          "|" + "---|" * (4 + len(wins))]
    for r in res["region"]:
        L.append(f"| {r['region']} | {r['h0in']} | {r['theta']:.0%} | {r['cw']} | " + " | ".join(
            f"{r[f'f{k}_pass_yr']:.1f}/{r[f'f{k}_taken_yr']:.1f}{' ✓' if r[f'f{k}_feasible'] else ''}" for k in range(len(wins))) + " |")
    for rname, v in res["region_verdict"].items():
        bad = [k + 1 for k, ok in enumerate(v) if not ok]
        L.append("")
        L.append(f"- **영역 {rname}:** " + ("모든 폴드에서 실현 가능한 (θ, 창) 칸이 있다." if not bad else
                                          f"폴드 {bad} 는 어느 (θ, 창) 칸도 학습창 하한을 못 넘는다 → **그 폴드에서 이 영역은 이 데이터로 GA 판단 불가**"
                                          "(결과 전 표시 — GA 최적이 이 영역에 떨어질 수 없거나, 떨어져도 벌점 구역)."))
    rnd = res["random"]
    L += ["", f"## 4. 같은 메뉴 균등 무작위 설정 {len(rnd):,}개 — 실현 가능 비율(건수만)", "",
          "| 폴드(학습창) | 통과 사건 0 | 통과 사건 중앙/90% | 진입 중앙/90% | **실현 가능(벌점 0)** |", "|---|---|---|---|---|"]
    for k, (tr, _) in enumerate(wins):
        L.append(f"| {k + 1} ({tr.lo}~{tr.hi}) | {(rnd[f'f{k}_pass'] == 0).mean():.1%} | {rnd[f'f{k}_pass'].median():.0f}/"
                 f"{rnd[f'f{k}_pass'].quantile(0.9):.0f} | {rnd[f'f{k}_n'].median():.0f}/{rnd[f'f{k}_n'].quantile(0.9):.0f} | "
                 f"**{rnd[f'f{k}_feas'].mean():.1%}** |")
    L += ["", f"건수만 선별 {SCREEN_N:,}개/폴드의 기대 실현 가능 수(위 비율 × {SCREEN_N:,}): " + " · ".join(
        f"폴드{k + 1} {rnd[f'f{k}_feas'].mean() * SCREEN_N:,.0f}" for k in range(len(wins)))
          + f" (2a 하한 {FEAS_MIN} — 2단계 M1)"]
    short = [k + 1 for k in range(len(wins)) if rnd[f"f{k}_feas"].mean() * SCREEN_N < FEAS_MIN]
    if short:
        L += ["", f"**결과 전 경고:** 폴드 {short} 는 선별 상한 {SCREEN_N:,} 개로 실현 가능 {FEAS_MIN} 개에 못 미칠 것으로 예상된다(비율 추정 표본 "
              f"{len(rnd):,}개 — 잡음 큼). 그대로 돌리면 그 폴드 때문에 2a 가 '판단 불가 = ✗' 로 기록될 수 있다. 선별 규모를 바꾸는 것은 결과 전 결정이다."]
    by_e = rnd.groupby("entry")[[f"f{k}_feas" for k in range(len(wins))]].mean()
    L += ["", "기준선별 실현 가능 비율: " + " · ".join(f"{e} " + "/".join(f"{v:.0%}" for v in row) for e, row in by_e.iterrows())]
    mi = res.get("minute") or {}
    L += ["", f"## 5. 분봉 커버리지(M7 — {MINUTE_LO}+ 사건 (종목, 주문일) 고유, 전 표 합집합)", "",
          "| ADV20 구간 | 사건 | 분봉 있음 |", "|---|---|---|"]
    for b, v in mi.items():
        L.append(f"| {b} | {v['n']:,} | {v['minute']:,} ({v['minute'] / max(v['n'], 1):.1%}) |")
    L += ["", "분봉 없는 사건의 진입일 모호 처리: 분봉 p 는 분봉 있는 합성 사건으로 한 번 재고(전체 모드), 모든 모호 거래에 같은 p 혼합"
          "(보수 p=1 · 낙관 p=0 경계를 함께 보고).", "",
          "## 6. 시간 추정", "",
          f"- 평가 1회(건수만) {res['sec_per_eval_counts'] * 1e3:.2f} ms · 적합도 전부 {res['sec_per_eval_fit'] * 1e3:.2f} ms(단일 스레드, 폴드 6 학습창)",
          f"- 전체 실행: {est['lines']}", "",
          "## 공통", "",
          f"- NXT 미보정 종목 {E.meta.get('n_cut_stocks', 0):,}개 보유 경로 2026-09-11 절단, 전 종목 데이터 끝 {END_DAY} 시가평가.",
          "- 이 문서는 건수만 적는다. 판정은 사람이 한다.", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def estimate(sec: float, sec_cnt: float, workers: int, budget: int, seeds: int, n_null_b: int, pool_n: int) -> dict:
    ev_real = 6 * seeds * budget
    ev_null = n_null_b * ev_real
    ev_pool = 6 * pool_n
    ev_misc = 6 * DIM * 6 * 2 + 3 * (N_NULL_A + N_RS + N_EXIT_CTL) * 6 + 6 * seeds * (1 + TOP_PBO)
    cnt = 6 * SCREEN_N + n_null_b * 6 * RESTART_SCREEN       # 건수만 선별(2a 풀 · 2b 재시작 풀)
    tot = ev_real + ev_null + ev_pool + ev_misc
    h = (tot * sec + cnt * sec_cnt) / workers / 3600.0
    return {"evals": tot, "hours": h, "lines": f"완전 평가 ≈ {tot:,.0f}회(실제 {ev_real:,} · 모양 섞은 널 {ev_null:,} · 무작위 탐색 풀 {ev_pool:,} · "
                                               f"기타 {ev_misc:,}) × {sec * 1e3:.2f} ms + 건수만 선별 {cnt:,.0f}회 × {sec_cnt * 1e3:.2f} ms "
                                               f"/ 작업자 {workers} ≈ **{h:.1f}시간**(+ 캐시·분봉 p 적재·JIT)"}


# ---------------------------------------------------------------------------
# 8. 보고용 트레이드 표
# ---------------------------------------------------------------------------

def trade_frame(E: Engine, r: dict, cfgs: "list[dict]") -> pd.DataFrame:
    """포트폴리오 기록 → 트레이드 표(p 혼합 재료: ``ret``/``ret_stop0``·``net``/``net2``(1·2배)). 비율은 슬롯 배분 1 단위."""
    rec = r["rec"]
    if not len(rec):
        return pd.DataFrame()
    ev = rec[:, PF.Q_EV].astype(np.int64)
    F, I_ = np.asarray(E.F_orig), np.asarray(E.I)
    d, x = rec[:, PF.Q_D].astype(int), rec[:, PF.Q_X].astype(int)
    e, sp, wb = rec[:, PF.Q_E], rec[:, PF.Q_SP], rec[:, PF.Q_WB]
    added = rec[:, PF.Q_ADDED] > 0.5
    wa = np.where(added, 1.0 - wb, 0.0)
    c1 = rec[:, PF.Q_CX] * wb + rec[:, PF.Q_CADD] * wa
    mk_ = np.asarray(E.F_orig)[ev, PE.FI["market"]].astype(int)
    s0p = rec[:, PF.Q_S0]                                            # 진입일 장중 손절선(장중 기준 = 손절가, 종가 기준 = 재난선)
    s0 = s0p * 0.999 - PE.tick_dated_vec(s0p, d, mk_, E.reform)      # 보수 경로 진입일 손절 체결(−0.1% − 1호가)
    amb_ = rec[:, PF.Q_AMB] > 0.5
    ret0 = np.where(amb_, wb * (s0 / e - 1.0), rec[:, PF.Q_PNL])   # vcp 규약: 보수 경로는 모호 거래만(그 밖 = 낙관과 같다)
    t = pd.DataFrame({"ev": ev, "code": np.asarray(E.codes)[I_[ev, PE.II["ci"]]], "ci": I_[ev, PE.II["ci"]], "d_row": d, "x_row": x,
                      "event_day": E.days[d - 1], "order_day": E.days[d], "entry_day": E.days[d], "exit_day": E.days[x],
                      "fold": rec[:, PF.Q_PAR].astype(int) // E.NF, "entry_px": e, "stop_px": sp, "d": (e - sp) / e,
                      "ret": rec[:, PF.Q_PNL], "ret_stop0": ret0, "amb": rec[:, PF.Q_AMB] > 0.5,
                      "exit_reason": np.asarray(PF.REASONS, dtype=object)[rec[:, PF.Q_REASON].astype(int)], "s0": s0p,
                      "hold_days": x - (d - 1), "wb": wb, "added": added, "partial": rec[:, PF.Q_TP] >= 0,
                      "mfe_px": rec[:, PF.Q_MFE], "alloc": rec[:, PF.Q_ALLOC], "exit_open": rec[:, PF.Q_EXIT_OPEN] > 0.5,
                      "reentry": rec[:, PF.Q_RE] > 0.5,
                      "exit_px": rec[:, PF.Q_FIN], "partial_px": rec[:, PF.Q_PX1], "add_px": rec[:, PF.Q_PA]})
    t["net"] = t["ret"] - c1
    t["net2"] = t["ret"] - 2 * c1
    t["net_stop0"] = np.where(amb_, t["ret_stop0"] - rec[:, PF.Q_CD] * wb, t["net"])
    t["net2_stop0"] = np.where(amb_, t["ret_stop0"] - 2 * rec[:, PF.Q_CD] * wb, t["net2"])
    t["mfe_R"] = (t["mfe_px"] - e) / (e - sp)
    t["real_R_post"] = t["net"] / (wb * t["d"])       # 분모 = 호가·매수 미끄러짐 뒤 R(MINOR 3)
    t["order_px"] = F[ev, PE.FI["order"]]
    t["u_d"] = F[ev, PE.FI["u_d"]]
    t["order_vs_u"] = t["order_px"] / t["u_d"] - 1.0    # 2단계 MINOR 2: 주문가가 상단 추세선 위(이미 쐐기 밖 매수)
    t["above_u"] = t["order_vs_u"] > 0
    t["adv20"] = F[ev, PE.FI["adv20"]]
    t["s_up"], t["s_lo"] = F[ev, PE.FI["s_up"]], F[ev, PE.FI["s_lo"]]
    t["shape"] = [shape_name(a, b) for a, b in zip(t["s_up"], t["s_lo"], strict=True)]
    t["table"] = I_[ev, PE.II["table"]]
    t["base"] = [PE.BASES[k % len(PE.BASES)] if k < PE.N_TABLES else ("P" if k == EF.TID_P else "C") for k in t["table"]]
    t["fam"] = np.where(t["table"] == EF.TID_P, "P", np.where(t["table"] == EF.TID_C, "C", "W"))   # 13세대 가족
    t["entry_close"] = (t["fam"] == "P").to_numpy() & (d == I_[ev, PE.II["d"]] - 1)          # 종가 진입(주문 행 = s)
    return t


def pmix_net(t: pd.DataFrame, p: float, col: str = "net") -> pd.Series:
    """거래별 p 혼합 순 — 진입일 손절 보수 경로는 모호(``amb``) 거래에만 섞는다(:func:`vcp_retest_run.expand` 와 같은 정의).
    2026-10-11 발견: 보고용 혼합이 모든 거래에 보수 경로를 섞어 승률·손익비를 깎았다(바 6 판정 경로 ``R.expand`` 는 정상)."""
    return t[col].where(~t["amb"].astype(bool), p * t[f"{col}_stop0"] + (1 - p) * t[col])


def mfe_table(t: pd.DataFrame, p: float) -> dict:
    """M1-d: MFE 구간(<1, 1~2, 2~4, ≥4 R)별 실현 R 평균·되돌림, '+1R → 손실' 비율, 승자 포착률(기술) — 낙관 경로·p 혼합 순."""
    if t is None or t.empty:
        return {}
    mix = pmix_net(t, p)
    rr = mix / (t["wb"] * t["d"])
    b = pd.cut(t["mfe_R"], [-np.inf, 1, 2, 4, np.inf], right=False, labels=["<1R", "1~2R", "2~4R", "≥4R"])
    out = {"buckets": {str(k): {"n": int(len(g)), "real_R": float(rr[g.index].mean()),
                                "giveback_R": float((t.loc[g.index, "mfe_R"] - rr[g.index]).mean())}
                       for k, g in t.groupby(b, observed=True)},
           "plus1R_to_loss": float(((t["mfe_R"] >= 1) & (mix < 0)).mean())}
    win = mix > 0
    cap = (mix[win] / (t.loc[win, "wb"] * (t.loc[win, "mfe_px"] / t.loc[win, "entry_px"] - 1))).replace([np.inf, -np.inf], np.nan)
    out["winner_capture_median"] = float(cap.median()) if win.any() else np.nan
    return out


def pair_exits(E: Engine, t: pd.DataFrame, cfgs: "list[dict]", variant) -> dict:
    """같은 진입(OOS 트레이드)에 다른 청산을 붙인 짝 — ``variant(px, cfg) -> px``. 낙관 경로 총손익(배분 1 단위) 차이, 날짜군집 CI."""
    from krx_quant_core.stats.matched_null import cluster_bootstrap_ci  # noqa: PLC0415
    if t is None or t.empty:
        return {"n": 0}
    out = np.empty(PF.N_O)
    F = np.asarray(E.F_orig)
    a, b = [], []
    for r in t.itertuples(index=False):
        g = cfgs[int(r.fold)]
        px = exit_vec(g)
        fb = F[r.ev, PE.FI["level"]] * (1 - (g["fb_tol"] or 0.0)) if g["entry"] != "bottom" else -np.inf
        skip0 = bool(F[r.ev, PE.FI["open_d"]] < F[r.ev, PE.FI["order"]] and F[r.ev, PE.FI["close_d"]] > r.s0)
        args = path_args(E, F, r, g, fb, skip0)
        PF.exit_path(*args, px, out)
        a.append(out[PF.O_PNL])
        PF.exit_path(*args, variant(px.copy(), g), out)
        b.append(out[PF.O_PNL])
    diff = np.array(b) - np.array(a)
    lo, hi, _ = cluster_bootstrap_ci(diff, t["order_day"].dt.strftime("%Y%m%d").to_numpy(), n_boot=CI_BOOT, seed=0)
    return {"n": int(len(diff)), "a": float(np.mean(a)), "b": float(np.mean(b)), "diff": float(diff.mean()), "lo": lo, "hi": hi}


def variant_paths(E: Engine, t: pd.DataFrame, cfgs: "list[dict]", variant) -> pd.DataFrame:
    """같은 진입에 원래 청산과 ``variant`` 청산 경로를 각각 돌린 거래별 표(낙관 경로, 배분 1 단위 총손익·사유·청산가·MFE R)."""
    rows = []
    out = np.empty(PF.N_O)
    F = np.asarray(E.F_orig)
    for r in t.itertuples(index=False):
        g = cfgs[int(r.fold)]
        px = exit_vec(g)
        fb = F[r.ev, PE.FI["level"]] * (1 - (g["fb_tol"] or 0.0)) if g["entry"] != "bottom" else -np.inf
        skip0 = bool(F[r.ev, PE.FI["open_d"]] < F[r.ev, PE.FI["order"]] and F[r.ev, PE.FI["close_d"]] > r.s0)
        args = path_args(E, F, r, g, fb, skip0)
        row = {"order_day": r.order_day, "R": float(r.entry_px - r.stop_px), "e": float(r.entry_px)}
        for tag, p_ in (("a", px), ("b", variant(px.copy(), g))):
            PF.exit_path(*args, p_, out)
            row.update({f"{tag}_pnl": out[PF.O_PNL], f"{tag}_reason": PF.REASONS[int(out[PF.O_REASON])], f"{tag}_fin": out[PF.O_FIN],
                        f"{tag}_mfe_R": (out[PF.O_MFE] - r.entry_px) / (r.entry_px - r.stop_px)})
        rows.append(row)
    return pd.DataFrame(rows)


def climax_report(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> dict:
    """정정 4 보고: 클라이맥스 켠 폴드 거래 — 발동 거래의 포착률(실현 / 보유 중 최고), 발동 안 했을 때(같은 진입·나머지 청산) 짝 차이."""
    if t is None or t.empty:
        return {"n": 0}
    sub = t[[bool(cfgs[int(f)].get("climax")) for f in t["fold"]]]
    if sub.empty:
        return {"n": 0}
    vp = variant_paths(E, sub, cfgs, lambda px, g: _set(px, PF.X_CLIMAX, 0.0))
    fired = (vp["a_reason"] == "climax").to_numpy()
    cap = (vp["a_pnl"] / (sub["wb"].to_numpy() * (sub["mfe_px"].to_numpy() / sub["entry_px"].to_numpy() - 1)))[fired]
    d = (vp["a_pnl"] - vp["b_pnl"]).to_numpy()
    return {"n": int(len(sub)), "fired": int(fired.sum()), "capture_median": float(np.median(cap)) if fired.any() else np.nan,
            "diff_all": float(d.mean()), "diff_fired": float(d[fired].mean()) if fired.any() else np.nan}


def stop_basis_report(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> dict:
    """정정 5 보고: 종가 기준 폴드 거래를 장중 기준으로 바꾼 짝 — 살린 거래(장중이면 손절, 종가 기준 경로는 이후 +2R 도달) 수,
    키운 손실(둘 다 손절인데 종가 기준 체결가가 더 나쁨) 수·평균 차이(배분 1 단위)."""
    if t is None or t.empty:
        return {"n": 0}
    sub = t[[cfgs[int(f)].get("stop_basis") == "close" for f in t["fold"]]]
    if sub.empty:
        return {"n": 0}
    vp = variant_paths(E, sub, cfgs, lambda px, g: _set(px, PF.X_STOP_CLOSE, 0.0))
    intra_stop = vp["b_reason"].isin(["stop", "stop_entry"]).to_numpy()
    saved = intra_stop & (vp["a_mfe_R"] >= 2).to_numpy() & ~vp["a_reason"].isin(["stop", "stop_entry"]).to_numpy()
    both = intra_stop & vp["a_reason"].isin(["stop", "stop_entry"]).to_numpy()
    worse = both & (vp["a_fin"] < vp["b_fin"]).to_numpy()
    return {"n": int(len(sub)), "saved": int(saved.sum()), "worse": int(worse.sum()),
            "worse_mean": float((vp["a_pnl"] - vp["b_pnl"])[worse].mean()) if worse.any() else np.nan,
            "diff_all": float((vp["a_pnl"] - vp["b_pnl"]).mean())}


def idea_data_flags(cfgs: "list[dict]", wins) -> "list[list[str]]":
    """정정 4~7: 착상 거래 연도가 학습창(정정 6·7 은 시험창도)에 든 폴드에서 그 유전자를 골랐으면 '착상과 같은 데이터'."""
    out = []
    for c, (tr, te) in zip(cfgs, wins, strict=True):
        tr_y = set(range(pd.Timestamp(tr.lo).year, pd.Timestamp(tr.hi).year))
        te_y = set(range(pd.Timestamp(te.lo).year, max(pd.Timestamp(te.hi).year, pd.Timestamp(te.lo).year + 1)))
        out.append([nm for nm, pred, years, with_test in IDEA_FLAGS
                    if pred(c) and any(y in tr_y or (with_test and y in te_y) for y in years)])
    return out


def path_args(E: Engine, F: np.ndarray, r, g: dict, fb: float, skip0: bool) -> tuple:
    """트레이드 표 한 행 → :func:`pattern_fast._path_py` 인자(px·out 앞까지)."""
    ev = int(r.ev)
    I_ = np.asarray(E.I)
    lbuf = (g.get("stop_buf") or 0.0) * F[ev, PE.FI["atr_s"]] if g.get("stop_buf") else 0.0
    return (E.O, E.H, E.L, E.C, E.MA20, E.MA50, E.ATR, E.TR, E.LIM, int(r.ci), int(r.d_row), float(r.entry_px),
            float(r.stop_px), float(fb), float(F[ev, PE.FI["h0"]]), float(F[ev, PE.FI["u_d"]]), float(F[ev, PE.FI["su"]]),
            float(F[ev, PE.FI["E"]]), int(I_[ev, PE.II["iE"]]), float(F[ev, PE.FI["sl"]]), float(lbuf), skip0,
            int(F[ev, PE.FI["market"]]), int(E.reform))


def fail_exit_report(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> dict:
    """정정 6 보고: 반등 실패 조기 청산을 켠 폴드 거래(재진입 아닌 원 거래) — 끈 짝과 비교해 줄인 손실(끔이면 손실·켬이 덜 잃음)과
    잘라 낸 승자(끔이면 이익·켬이 덜 벌거나 잃음)."""
    if t is None or t.empty:
        return {"n": 0}
    sub = t[[cfgs[int(f)].get("fail_exit") not in (None, "off") for f in t["fold"]] & ~t["reentry"].to_numpy(bool)]
    if sub.empty:
        return {"n": 0}
    vp = variant_paths(E, sub, cfgs, lambda px, g: _set(px, PF.X_FAIL_EXIT, 0.0))
    a, b = vp["a_pnl"].to_numpy(), vp["b_pnl"].to_numpy()
    fired = (vp["a_reason"] == "fail_exit").to_numpy()
    cut_loss = fired & (b < 0) & (a > b)
    cut_win = fired & (b > 0) & (a < b)
    return {"n": int(len(sub)), "fired": int(fired.sum()), "cut_loss": int(cut_loss.sum()),
            "cut_loss_gain": float((a - b)[cut_loss].mean()) if cut_loss.any() else np.nan, "cut_win": int(cut_win.sum()),
            "cut_win_cost": float((a - b)[cut_win].mean()) if cut_win.any() else np.nan, "diff_all": float((a - b).mean())}


def reentry_report(t: pd.DataFrame, p: float) -> dict:
    """정정 6 보고: 재진입 거래만 따로(p 혼합 순 2배)."""
    if t is None or t.empty or not t["reentry"].any():
        return {"n": 0}
    x = t[t["reentry"]]
    mix = pmix_net(x, p, "net2")
    return {"n": int(len(x)), "net2_mean": float(mix.mean()), "win": float((mix > 0).mean()),
            "reasons": x["exit_reason"].value_counts().to_dict()}


def pyramid_kind_pairs(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> dict:
    """정정 6 보고: 불타기 종류별 짝 — 같은 진입(재진입 제외)에 불타기 끔 대비 각 종류(+1R·첫 신고가·상단선 돌파)의 낙관 경로 총손익 차이."""
    if t is None or t.empty:
        return {}
    sub = t[~t["reentry"].to_numpy(bool)]
    out = {}
    for nm, code in (("+1R", PF.PYR_1R), ("첫 신고가", PF.PYR_NEWHIGH), ("상단선 돌파", PF.PYR_UPPER)):
        rows = sub[[not (code == PF.PYR_NEWHIGH and cfgs[int(f)]["entry"] == "h0") for f in sub["fold"]]]
        if rows.empty:
            continue
        vp = variant_paths(E, rows, cfgs, lambda px, g, c=code: _set(_set(px, PF.X_PYR, float(c)), PF.X_WB, PF.WB_PYR))
        base = variant_paths(E, rows, cfgs, lambda px, g: _set(px, PF.X_PYR, float(PF.PYR_OFF)))
        out[nm] = {"n": int(len(rows)), "diff": float((vp["b_pnl"] - base["b_pnl"]).mean())}
    return out


def dynamic_stop_report(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> dict:
    """정정 7 보고: 하단선 동적 폴드 거래 — 하단선(d) 고정 손절 짝과 비교한 살린 거래(고정이면 손절, 동적 경로는 이후 +2R)·키운 손실
    (둘 다 손절인데 동적 체결가가 더 나쁨), 그리고 동적 손절 체결가의 진입가 대비 분포."""
    if t is None or t.empty:
        return {"n": 0}
    sub = t[[cfgs[int(f)].get("stop_anchor") == "dynamic" for f in t["fold"]]]
    if sub.empty:
        return {"n": 0}
    vp = variant_paths(E, sub, cfgs, lambda px, g: _set(px, PF.X_DYN, 0.0))
    fix_stop = vp["b_reason"].isin(["stop", "stop_entry"]).to_numpy()
    dyn_stop = vp["a_reason"].isin(["stop", "stop_entry"]).to_numpy()
    saved = fix_stop & ~dyn_stop & (vp["a_mfe_R"] >= 2).to_numpy()
    worse = fix_stop & dyn_stop & (vp["a_fin"] < vp["b_fin"]).to_numpy()
    rel = (vp["a_fin"] / vp["e"] - 1.0)[dyn_stop]
    return {"n": int(len(sub)), "saved": int(saved.sum()), "worse": int(worse.sum()),
            "worse_mean": float((vp["a_pnl"] - vp["b_pnl"])[worse].mean()) if worse.any() else np.nan,
            "stop_fill_q": {q: float(rel.quantile(q)) for q in (0.1, 0.5, 0.9)} if dyn_stop.any() else {}}


# --- 정정 8 사용자 판정 사례 재현표(보고만) ---
CASES = (("성창기업지주", "000180", "2021-01-18", "잘함"), ("조선선재", "120030", "2021-06-11", "잘함"),
         ("한화손해보험", "000370", "2021-06-15", "잘함"), ("콘텐트리중앙", "036420", "2021-06-24", "잘함"),
         ("클래시스", "214150", "2024-01-09", "잘함"), ("에스원", "012750", "2026-05-04", "잘함"),
         ("농심(VCP)", "004370", "2020-06-11", "잘못"), ("유나이티드제약", "033270", "2018-01-23", "잘못"),
         ("한화솔루션", "009830", "2020-12-07", "잘못"), ("동서", "026960", "2021-01-21", "잘못"),
         ("롯데하이마트", "071840", "2021-02-24", "잘못"), ("아이즈비전", "031310", "2021-05-06", "잘못"))
CASE_TOL = 5                                          # 사례 날짜 ± 5 거래일 안 가장 가까운 사건
WD_TRADES = ("research/logs/sepa_explore/wedge_d/adv1000/trades_WD.csv", "research/logs/sepa_explore/wedge_d/trades_WD.csv")


def diagnose(Frow: np.ndarray, gv: np.ndarray, lu_ratio: float = float(LIMIT_UP_RATIO)) -> str:
    """사건 한 행이 거르기에서 처음 걸린 조건 이름(:func:`pattern_fast._filter_py` 와 같은 순서 — 시험이 대조한다). 통과면 'ok'."""
    fi = PE.FI

    def nn(v):
        return v == v

    f = Frow
    if nn(gv[PF.G_H0IN]) and f[fi["h0in"]] != gv[PF.G_H0IN]:
        return "h0_in_pattern"
    if int(gv[PF.G_CEILKIND]) == 1 and not f[fi["n_res"]] >= 2:
        return "ceil_kind"
    st = int(gv[PF.G_STAGE])
    if (st == 1 and not f[fi["above50"]] > 0.5) or (st == 2 and not f[fi["tt_s"]] > 0.5) or (st == 3 and not f[fi["tt_h0"]] > 0.5):
        return "stage"
    for nm, col, lo in (("rs_min", "rs", gv[PF.G_RS]), ("breadth_min", "breadth", gv[PF.G_BR])):
        if nn(lo) and not f[fi[col]] >= lo:
            return nm
    for nm, col, a, b in (("s_up", "s_up", gv[PF.G_SUP_LO], gv[PF.G_SUP_HI]), ("s_lo", "s_lo", gv[PF.G_SLO_LO], gv[PF.G_SLO_HI]),
                          ("span", "span", gv[PF.G_SPAN_LO], gv[PF.G_SPAN_HI])):
        if nm != "span" and nn(b) and not (f[fi[col]] >= a and f[fi[col]] <= b):
            return nm
        if nm == "span":
            span_ab = (a, b)
    if nn(gv[PF.G_RHO]) and not f[fi["rho"]] <= gv[PF.G_RHO]:
        return "rho_max"
    if int(gv[PF.G_TOUCH]) == 3 and not abs(f[fi["touch"]]) <= 0.03:
        return "touch_lo"
    if nn(span_ab[1]) and not (f[fi["span"]] >= span_ab[0] and f[fi["span"]] <= span_ab[1]):
        return "span"
    if nn(gv[PF.G_CEIL]) and not f[fi["ceil"]] >= gv[PF.G_CEIL]:
        return "ceil_min"
    if nn(gv[PF.G_DEPTH]) and not f[fi["depth"]] <= gv[PF.G_DEPTH]:
        return "depth_max"
    if nn(gv[PF.G_DRY]) and not f[fi["dry"]] <= gv[PF.G_DRY]:
        return "dry_max"
    if nn(gv[PF.G_RUNUP]) and not f[PF.F_RUN20 + int(gv[PF.G_RUNUP_COL])] >= gv[PF.G_RUNUP]:
        return "runup_min"
    if nn(gv[PF.G_REST]) and not f[fi["rest_gap"]] <= gv[PF.G_REST]:
        return "rest_gap_max"
    if nn(gv[PF.G_EMIN]):
        if not f[fi["e_yoy"]] >= gv[PF.G_EMIN]:
            return "e_min"
        if gv[PF.G_EACC] > 0.5 and not (nn(f[fi["e_yoy_prev"]]) and f[fi["e_yoy"]] > f[fi["e_yoy_prev"]]):
            return "e_accel"
    buf = [np.empty(1, t) for t in (np.int64, float, float, float, np.bool_, np.bool_, float, float)]
    G1 = np.ascontiguousarray(np.asarray(f, float)[None, :])
    if PF.filter_events(G1, 0, 1, gv, lu_ratio, *buf) == 0:
        return "체결 조건(상한가·갭·손절 원값·손절 상한)"
    return "ok"


def case_table(E: Engine, cfgs: "list[dict]", wins) -> "tuple[pd.DataFrame, list]":
    """정정 8: 사례마다 폴드 출력 config 로 — 사건 표에 있나(± :data:`CASE_TOL` 거래일), 거르기 첫 탈락 조건, 통과면 그 config 청산과
    기준(라이브) 청산의 진입·청산·순(2배 비용, 낙관 경로)·R·포착률. 포트폴리오 슬롯은 보지 않는다(사례 단독). + 쐐기 D 거래(있으면)."""
    codes = pd.Index(np.asarray(E.codes))
    F, I_ = np.asarray(E.F_orig), np.asarray(E.I)
    wd = None
    for path in WD_TRADES:
        if os.path.exists(path):
            wd = pd.read_csv(path, dtype={"code": str}, parse_dates=["order_day", "exit_day"])
            break
    rows, charts = [], []
    out = np.empty(PF.N_O)
    for name, code, day, verdict in CASES:
        ci = int(codes.get_loc(code)) if code in codes else -1
        dd = E.row(day)
        base = {"case": name, "code": code, "date": day, "verdict": verdict}
        if wd is not None:
            w = wd[(wd["code"] == code) & ((wd["order_day"] - pd.Timestamp(day)).abs() <= pd.Timedelta(days=7))]
            base["wd"] = (f"{w['order_day'].iloc[0].date()} → {w['exit_day'].iloc[0].date()} {w['exit_reason'].iloc[0]} "
                          f"R {w['R'].iloc[0]:+.2f}") if len(w) else "쐐기 D 체결 없음"
        chart_done = False
        for k, g in enumerate(cfgs):
            r_ = {**base, "fold": k + 1}
            if ci < 0:
                rows.append({**r_, "status": "종목 없음"})
                continue
            a, b = PE.table_slice(E.off, table_of(g))
            m = np.flatnonzero((I_[a:b, PE.II["ci"]] == ci) & (np.abs(I_[a:b, PE.II["d"]] - dd) <= CASE_TOL))
            if not len(m):
                rows.append({**r_, "status": "사건 표 없음(그 θ·창·기준선에서 첫 관통·위생·맥락)"})
                continue
            ev = a + int(m[np.argmin(np.abs(I_[a + m, PE.II["d"]] - dd))])
            d = int(I_[ev, PE.II["d"]])
            gv = gene_vec(g)
            why = diagnose(F[ev], gv)
            if FW and why == "ok" and not fw_ok(float(F[ev, PE.FI["s_up"]]), float(F[ev, PE.FI["s_lo"]]), g):
                why = "FW 모양(하락 쐐기 계열 아님 — 리뷰 FW M7)"
            r_.update(event_day=str(E.days[d].date()), offset=d - dd)
            if why != "ok":
                rows.append({**r_, "status": f"거르기 탈락: {why}"})
                continue
            buf = [np.empty(1, t) for t in (np.int64, float, float, float, np.bool_, np.bool_, float, float)]
            PF.filter_events(np.ascontiguousarray(F[ev][None, :]), 0, 1, gv, float(LIMIT_UP_RATIO), *buf)
            e, sp, fb, skip0, lbuf = buf[1][0], buf[2][0], buf[3][0], bool(buf[5][0]), buf[7][0]
            m_ = int(F[ev, PE.FI["market"]])
            res = {}
            for tag, px in (("ga", exit_vec(g)), ("live", live_px(exit_vec(g), g))):
                PF.exit_path(E.O, E.H, E.L, E.C, E.MA20, E.MA50, E.ATR, E.TR, E.LIM, ci, d, e, sp, fb, float(F[ev, PE.FI["h0"]]),
                             float(F[ev, PE.FI["u_d"]]), float(F[ev, PE.FI["su"]]), float(F[ev, PE.FI["E"]]), int(I_[ev, PE.II["iE"]]),
                             float(F[ev, PE.FI["sl"]]), float(lbuf), skip0, m_, int(E.reform), px, out)
                wb = out[PF.O_WB]
                net2 = out[PF.O_PNL] - 2 * E.cost_tab[m_, int(out[PF.O_X])] * wb
                mfe = out[PF.O_MFE] / e - 1.0
                res.update({f"{tag}_exit": str(E.days[int(out[PF.O_X])].date()), f"{tag}_reason": PF.REASONS[int(out[PF.O_REASON])],
                            f"{tag}_net2": net2, f"{tag}_R": net2 / (wb * (e - sp) / e),
                            f"{tag}_capture": out[PF.O_PNL] / (wb * mfe) if mfe > 0 else np.nan})
            rows.append({**r_, "status": "포착", "entry_px": e, "stop_px": sp, **res})
            if not chart_done:
                th, cw, bn = PE.table_key(table_of(g))
                pt = PE.pattern_points(np.asarray(E.H), np.asarray(E.L), np.asarray(E.C), ci, d - 1, th, cw)
                lo, hi = max(0, pt.get("i0", d) - 20), min(len(E.days) - 1, d + 60)
                charts.append({"case": name, "code": code, "verdict": verdict, "fold": k + 1, "table": {"theta": th, "ceil_win": cw,
                               "entry": bn}, "order_day": str(E.days[d].date()), "entry_px": float(e), "stop_px": float(sp),
                               "points": [{"date": str(E.days[p["row"]].date()), "price": p["price"], "kind": p["kind"]}
                                          for p in pt.get("points", [])],
                               "upper": {kk: {"date": str(E.days[min(v["row"], len(E.days) - 1)].date()), "price": v["price"]}
                                         for kk, v in pt.get("upper", {}).items()},
                               "lower": {kk: {"date": str(E.days[min(v["row"], len(E.days) - 1)].date()), "price": v["price"]}
                                         for kk, v in pt.get("lower", {}).items()},
                               "bars": [{"date": str(E.days[j].date()), "o": float(E.O[ci, j]), "h": float(E.H[ci, j]),
                                         "l": float(E.L[ci, j]), "c": float(E.C[ci, j])} for j in range(lo, hi + 1)]})
                chart_done = True
    return pd.DataFrame(rows), charts


def _set(px: np.ndarray, i: int, v: float) -> np.ndarray:
    px[i] = v
    return px


def live_px(px: np.ndarray, g: dict) -> np.ndarray:
    p = PF.LIVE_X.copy()
    p[PF.X_SELL_TICK] = px[PF.X_SELL_TICK]
    p[PF.X_FB_DAYS] = 0 if g["entry"] == "bottom" else 1
    return p


def pyr_toggle(px: np.ndarray, g: dict) -> np.ndarray:
    px[PF.X_PYR] = PF.PYR_OFF if px[PF.X_PYR] != PF.PYR_OFF else PF.PYR_1R
    return px


# ---------------------------------------------------------------------------
# 9. 본체
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:  # noqa: C901 — 단계가 많은 리포터
    from swing_it.validation.walkforward import FOLDS  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="사건 표·배열 .npy 디렉터리(없으면 DB 에서 만든다)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--feasibility-only", action="store_true", help="건수만(수익 미계산) → pattern_ga/FEASIBILITY.md")
    ap.add_argument("--debug", action="store_true", help="축소 실행(폴드 2 · 시드 2 · 예산 300 · 널 축소) — --out 을 탐색 디렉터리 밖으로")
    ap.add_argument("--n-feas", type=int, default=N_FEAS)
    ap.add_argument("--no-db", action="store_true", help="실현 가능 모드에서 분봉 커버리지 조회 생략(캐시만)")
    ap.add_argument("--evolve", default="", help="진화 모드 후보 이름(META_EVOLUTION.md): 폴드 1~4 실제 GA 만, 널·봉인 폴드 없음")
    args = ap.parse_args(argv)
    if args.evolve:
        return run_evolve(args)
    out_dir = args.out
    debug = os.path.normpath(out_dir) != os.path.normpath(OUT_DIR)
    if args.debug and not debug:
        ap.error("--debug 는 --out 을 탐색 디렉터리 밖으로 둘 때만(원장 보호)")
    sub_dir = os.path.join(out_dir, SUB)
    os.makedirs(sub_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    t0 = time.time()
    fold_ids = list(range(len(FOLDS)))
    seeds, budget, n_a, n_b, n_rs, n_ec, pool_n = SEEDS, BUDGET, N_NULL_A, N_NULL_B, N_RS, N_EXIT_CTL, POOL_N
    screen = (SCREEN_N, RESTART_SCREEN, FEAS_MIN)
    if args.debug:
        fold_ids, seeds, budget, n_a, n_b, n_rs, n_ec, pool_n = [4, 5], (0, 1), 300, 20, 2, 20, 20, 400
        screen = (100_000, 50_000, 20)
        args.n_feas = min(args.n_feas, 60)
    folds = [FOLDS[k] for k in fold_ids]
    folds_spec = [tuple(f) for f in folds]
    ensure_cache(args.cache, t0, with_p=not args.feasibility_only)
    _record_design(out_dir)
    E = Engine(args.cache)
    wins = fold_windows(E, folds)
    _log(f"엔진 — 사건 {len(E.F):,} · 표 {PE.N_TABLES} · 폴드 {len(wins)}", t0)

    if args.feasibility_only:
        E.evaluate(decode(x0_feasible()), wins[-1][0])              # JIT 예열(시간에서 뺀다)
        res = feasibility(E, wins, folds, args, t0)
        est = estimate(res["sec_per_eval_fit"], res["sec_per_eval_counts"], args.workers, BUDGET, len(SEEDS), N_NULL_B, POOL_N)
        path = os.path.join(sub_dir, FEAS_MD)
        write_feasibility(path, res, E, wins, args, time.time() - t0, est)
        res["random"].to_csv(os.path.join(sub_dir, "feasibility_random.csv"), index=False)
        pd.DataFrame(res["region"]).to_csv(os.path.join(sub_dir, "feasibility_region.csv"), index=False)
        from swing_it.diagnostics.trials import record_trial  # noqa: PLC0415
        record_trial(LABEL, {"prereg": PREREG_DOC, "feasibility": "사건 표 건수·무작위 설정 실현 가능 비율·착상 영역 연 체결·분봉 커버리지",
                             "n_feas": int(args.n_feas), "evaluated": "건수만, 성과 미계산"}, logs_dir=out_dir,
                     note="패턴 GA 실현 가능성(건수만, 결과 전)")
        rnd = res["random"]
        print(f"사건 표 {E.meta['n_events']:,}행 · 무작위 실현 가능(폴드별) " + " · ".join(
            f"{rnd[f'f{k}_feas'].mean():.1%}" for k in range(len(wins))) + f" · 착상 영역 {res['region_verdict']}")
        print(est["lines"])
        print(f"→ {path}")
        return 0

    return run_full(E, wins, folds_spec, args, out_dir, sub_dir, seeds, budget, n_a, n_b, n_rs, n_ec, pool_n, debug, t0,
                    screen_n=screen[0], restart_screen=screen[1], feas_min=screen[2])


EVO_FOLDS = (0, 1, 2, 3)                # META_EVOLUTION.md: 설계 선택용 = 폴드 1~4 시험(2020~2023). 폴드 5·6(2024-01~) 봉인
EVO_LABEL, EVO_SUB = "pattern_ga_evo", "pattern_ga_evo"


def run_evolve(args) -> int:
    """진화 모드 — 후보 메뉴(``PATTERN_GA_MENU``)로 폴드 1~4 실제 GA(시드·예산 그대로) → 절차 출력 → 폴드 1~4 시험 이어 붙인 OOS 숫자만.
    널·민감도·봉인 폴드(5·6)는 계산하지 않는다. 원장 = ``pattern_ga_evo``(DSR 시행 수에 센다)."""
    from swing_it.diagnostics.trials import record_trial  # noqa: PLC0415
    from swing_it.validation.walkforward import FOLDS  # noqa: PLC0415

    t0 = time.time()
    folds_spec = [tuple(FOLDS[k]) for k in EVO_FOLDS]
    ensure_cache(args.cache, t0, with_p=True)
    E = Engine(args.cache)
    wins = fold_windows(E, [FOLDS[k] for k in EVO_FOLDS])
    out = os.path.join(args.out, EVO_SUB)
    os.makedirs(out, exist_ok=True)
    head = run_provenance(args.cache).get("git_head", "")
    offs_env = os.environ.get("PATTERN_GA_SEED_OFFSETS", "")       # 가속: 시드 집합 여럿을 한 프로세스에서(작업자·JIT·재시작 풀 한 번)
    offs = [int(x) for x in offs_env.split(",") if x.strip()] if offs_env else [int(os.environ.get("PATTERN_GA_SEED_OFFSET", "0"))]
    names = {o: (f"{args.evolve}_s{o}" if offs_env else args.evolve) for o in offs}
    all_seeds = {o: tuple(s + o for s in SEEDS) for o in offs}
    for o in offs:
        record_trial(EVO_LABEL, {"variant": VARIANT, "menu": MENU_OVERRIDE, "fit_mode": FIT_MODE, "floor": [FLOOR_PER_YEAR, FLOOR_YEAR], "candidate": names[o], "folds": list(EVO_FOLDS),
                                 "seeds": list(all_seeds[o]), "budget": BUDGET, "git_head": head, "role": "진화 후보(결과 전)"},
                     logs_dir=args.out, note=f"패턴 GA 진화 후보 {names[o]}")
    ctx = mp.get_context("spawn")
    r_chunks = max(1, math.ceil(RESTART_SCREEN / SCREEN_CHUNK))
    res = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=worker_init,
                                initargs=(args.cache, folds_spec)) as ex:
        sf = [ex.submit(screen_task, {"fold": k, "chunk": c, "n": SCREEN_CHUNK}) for k in range(len(wins)) for c in range(r_chunks)]
        rx = {k: {} for k in range(len(wins))}
        for f in cf.as_completed(sf):
            r = f.result()
            rx[r["fold"]][r["chunk"]] = r["X"]
        # 검증 MUST-FIX 2: 끝난 순서가 아니라 덩어리 번호 순서로 쌓는다(전체 러너와 같은 결정성)
        rx = {k: (np.vstack([v[c] for c in sorted(v)]) if v else np.zeros((0, DIM))) for k, v in rx.items()}
        _log("재시작 풀: " + " · ".join(f"폴드{k + 1} {len(v)}" for k, v in rx.items()), t0)
        seed_list = sorted({s for o in offs for s in all_seeds[o]})
        futs = [ex.submit(ga_task, {"fold": k, "seed": s, "budget": BUDGET, "x_restart": rx[k]}) for k in range(len(wins)) for s in seed_list]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            res[(r["fold"], r["seed"])] = r
            if i % 20 == 0:
                _log(f"진화 {args.evolve} 런 {i}/{len(futs)}", t0)
    # 검증 에이전트 MUST-FIX 1: 봉인 — 마지막 설계 선택 폴드 시험 끝(2024-01-01)에서 가격 경로·포트폴리오를 자른다(열린 포지션은 직전 종가 시가평가)
    # 절차(시드 최적 재평가)는 학습창에서만 평가하므로 봉인 자르기 전에 모두 끝낸다.
    prs = {o: procedure(E, wins, [[res[(k, s)]["best"] for s in all_seeds[o]] for k in range(len(wins))]) for o in offs}
    f_last = FOLDS[EVO_FOLDS[-1]]
    seal = E.row(f_last.test_hi)
    E.LIM = np.minimum(np.asarray(E.LIM), seal).astype(np.int64)
    E.end_row = seal - 1
    tests = [te for _, te in wins]
    tests[-1] = Win(E, f_last.test_lo, f_last.test_hi, f_last.test_hi)
    for o in offs:
        _evolve_write(E, prs[o]["cfgs"], tests, seal, out, args, names[o], all_seeds[o], head, t0)
    return 0


def _evolve_write(E, cfgs, tests, seal, out, args, name, evo_seeds, head, t0) -> None:
    """진화 후보 하나(시드 집합 하나)의 폴드 1~4 이어 붙인 OOS 숫자·거래 → ``<name>.json``·``<name>_trades.csv``·원장."""
    from swing_it.diagnostics.trials import record_trial  # noqa: PLC0415

    base = {"variant": VARIANT, "menu": MENU_OVERRIDE, "fit_mode": FIT_MODE, "floor": [FLOOR_PER_YEAR, FLOOR_YEAR], "candidate": name, "folds": list(EVO_FOLDS), "seeds": list(evo_seeds),
            "budget": BUDGET, "git_head": head}
    st = E.stitched(cfgs, tests)
    if len(st["rec"]) and int(np.max(st["rec"][:, PF.Q_X])) >= seal:
        raise SystemExit("봉인 위반: 청산 행이 봉인 경계를 넘는다")
    t = trade_frame(E, st, cfgs)
    p = E.pmix
    met = {"ann": st["ann"], "fold_ann": st["fold_ann"], "fold_n": st["fold_n"], "n": int(len(t)), "expo": st["expo_mean"]}
    if len(t):
        for tag, col in (("c1", "net"), ("c2", "net2")):
            x = pmix_net(t, p, col)
            w_, l_ = x > 0.005, x < -0.005
            met[tag] = {"win": float(w_.sum() / max(1, w_.sum() + l_.sum())), "mean": float(x.mean()),
                        "gl": float(x[w_].mean() / -x[l_].mean()) if w_.any() and l_.any() else float("nan")}
        xs = pmix_net(t, p, "net2") * (t["alloc"] / PF.SLOT_W).clip(upper=1.0)   # 6세대: 위험 기준 비중이면 슬롯 대비 손익(계좌 영향)
        met["c2_slot_mean"] = float(xs.mean())
        risk_slot = (t["alloc"] / PF.SLOT_W) * t["d"]                    # 검증 MUST-FIX 9: 손절 위험(슬롯 대비) ≤ 5% 단언
        if FWM and float(risk_slot.max()) > STOP_MAX + 1e-6:
            raise SystemExit(f"위험 기준 비중 위반: 슬롯 대비 손절 위험 최대 {float(risk_slot.max()):.4f}")
        met["c2_slot_min"] = float(xs.min())
        met["reasons"] = t["exit_reason"].value_counts().to_dict()
        met["loss55"] = int((pmix_net(t, p) < -0.055).sum())          # 검증 MINOR 2: 거래 순손익(추가분 포함) 기준
    met["cfgs"] = cfgs                                                  # 검증 MINOR 1: 원래 값(∞ 는 JSON Infinity)
    with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump({**base, "metrics": met, "sec": time.time() - t0}, fh, ensure_ascii=False, indent=1, default=float)
    t.to_csv(os.path.join(out, f"{name}_trades.csv"), index=False)
    record_trial(EVO_LABEL, {**base, "role": "진화 후보 결과", "ann": met["ann"], "n": met["n"]}, logs_dir=args.out,
                 note=f"패턴 GA 진화 후보 {name} 결과")
    _log(f"진화 {name} 끝 — 폴드 1~4 연환산 {met['ann']:+.2%} · 체결 {met['n']}", t0)


def run_full(E, wins, folds_spec, args, out_dir, sub_dir, seeds, budget, n_a, n_b, n_rs, n_ec, pool_n, debug, t0,
             screen_n: int = SCREEN_N, restart_screen: int = RESTART_SCREEN, feas_min: int = FEAS_MIN) -> int:
    """전체(또는 --debug) 실행 — 원장 → 건수 선별(재시작·2a 풀) → 실제 GA → 무작위 탐색 풀 → 널 2b → 후처리 → 결과 문서."""
    import vcp_retest_run as R  # noqa: PLC0415

    from krx_quant_core.stats.matched_null import cluster_bootstrap_ci  # noqa: PLC0415
    from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample  # noqa: PLC0415
    from swing_it.diagnostics.pbo import pbo_cscv  # noqa: PLC0415
    from swing_it.diagnostics.trials import count_trials, record_trial  # noqa: PLC0415

    timings: dict = {}
    if FW and not debug:                              # 리뷰 FW M10: 고정한 부모 열람 목록이 없거나 바뀌었으면 시작하지 않는다
        with open(FW_PARENT_VIEWED, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != FW_PARENT_SHA256:
                raise SystemExit(f"{FW_PARENT_VIEWED} 해시 불일치 — 사전등록 고정 목록과 다르다")
    fac_sha = R._sha(R.FACTORS_PKL)
    pmix = E.pmix
    base_cfg = {**_proc_config(budget, seeds, fac_sha, pmix), **run_provenance(args.cache)}
    record_trial(LABEL, {**base_cfg, "role": "절차(주) — 폴드별 시드 최적 재평가 1등, 이어 붙인 OOS"}, logs_dir=out_dir,
                 note="패턴 GA 절차(결과 전)")
    for nm, d_ in (("2a", f"실현 가능 조건부 같은 메뉴 무최적화 무작위 설정 {N_NULL_A}벌(폴드당 건수 선별 {SCREEN_N:,} → 실현 가능 ≥ "
                          f"{FEAS_MIN} 아니면 ✗, seed {NULL_A_SEED})"),
                   ("2b", f"모양 서술자 같은 달 섞음 × 절차 전체 {N_NULL_B}벌(재시작 풀 같은 규칙, seed {NULLB_SEED})"),
                   ("rs", f"무작위 탐색 best-of-{RS_BEST_OF} {N_RS}벌(기술, 균등 풀 {POOL_N:,}/폴드 실현 불가능 포함)"),
                   ("exit_ctl", f"청산 전용 대조 {N_EXIT_CTL}벌(기술)"), ("2a_entry", "2a′ 기준선 정합(기술)"),
                   ("2a_expo", "노출 정규화 2a(기술)")):
        record_trial(LABEL, {**base_cfg, "null": nm, "desc": d_}, logs_dir=out_dir, note=f"패턴 GA 널 {nm}(결과 전)")
    for cell in ("클라이맥스 짝(정정 4)", "손절 기준 짝(정정 5)", "조기 청산 짝(정정 6)", "재진입 거래(정정 6)", "불타기 종류별 짝(정정 6)",
                 "하단선 동적 손절 짝(정정 7)", "비용 1배", "보수 p=1", "낙관 p=0", "미끄러짐 매수 +0.3%", "미끄러짐 매수 +0.5%", "미끄러짐 매수·매도 +0.3%",
                 "미끄러짐 매수·매도 +0.5%", "비용 매도 금액 기준", "같은 날 무작위 순서", "상위 k(1·3·5) 거래 제외", "상위 5% 제외",
                 "2021 제외", "모양 끔 짝", "중앙값 config(조건부)", "기준 청산 짝", "불타기 짝"):
        record_trial(LABEL, {**base_cfg, "sensitivity": cell}, logs_dir=out_dir, note=f"패턴 GA 민감도·기술: {cell}")
    led_dir = out_dir
    nf_ = len(wins)

    res, scr, pools, nulls_b = {}, {k: [] for k in range(nf_)}, {k: [] for k in range(nf_)}, []
    hashes: set = set()
    ctx = mp.get_context("spawn")
    t1 = time.time()
    n_chunks = math.ceil(screen_n / SCREEN_CHUNK)
    r_chunks = math.ceil(restart_screen / SCREEN_CHUNK)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=worker_init,
                                initargs=(args.cache, folds_spec)) as ex:
        # 1) 건수 선별 — 재시작 풀 덩어리 먼저(실제 GA 가 기다린다), 나머지는 2a 풀
        if FW:                                       # 리뷰 FW M1·M2: 적응 선별(건수만 정지 규칙) — 재시작 풀 = 앞 r_chunks 덩어리
            scr, restart_x = fw_adaptive_screen(ex, nf_, screen_n, r_chunks, args.workers, t0)
            sfut = {}
        else:
            sfut = {ex.submit(screen_task, {"fold": k, "chunk": c, "n": min(SCREEN_CHUNK, screen_n - c * SCREEN_CHUNK)}): (k, c)
                    for c in range(n_chunks) for k in range(nf_)}
            restart_x = {k: [] for k in range(nf_)}
            pending = {fut for fut, (k, c) in sfut.items() if c < r_chunks}
            for f in cf.as_completed(pending):
                r = f.result()
                scr[r["fold"]].append(r)
                restart_x[r["fold"]].append(r["X"])
            restart_x = {k: (np.vstack(v) if v else np.zeros((0, DIM))) for k, v in restart_x.items()}
        timings["restart_screen"] = time.time() - t1
        _log("재시작 풀(실제): " + " · ".join(f"폴드{k + 1} {len(v)}" for k, v in restart_x.items()), t0)
        # 2) 실제 GA · 무작위 탐색 풀 · 2b
        tasks = [{"fold": k, "seed": s, "budget": budget, "top": TOP_PBO, "x_restart": restart_x[k]} for k in range(nf_)
                 for s in seeds]
        futs = {ex.submit(ga_task, t): t for t in tasks}
        per_chunk = 2_000 if not args.debug else 200
        pf = [ex.submit(pool_task, {"fold": k, "chunk": c, "n": min(per_chunk, pool_n - c * per_chunk)}) for k in range(nf_)
              for c in range(math.ceil(pool_n / per_chunk))]
        nfut = [ex.submit(null_b_task, {"rep": r, "seeds": seeds, "budget": budget, "restart_screen": restart_screen})
                for r in range(n_b)]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            res[(r["fold"], r["seed"])] = r
            hashes.update(r["hashes"].tolist())
            record_trial(RUN_LABEL, {"prereg": PREREG_DOC, "run": f"fold{r['fold']}/seed{r['seed']}", "budget": budget,
                                     "n_unique_configs_evaluated": r["uniq"], "best": {k: str(v) for k, v in r["best"].items()},
                                     "restarts": len(r["restarts"]), "git_head": base_cfg["git_head"]},
                         logs_dir=led_dir, note="패턴 GA 런 요약(별도 라벨 — DSR 변형 N 에 안 센다)")
            if i % 10 == 0 or i == len(futs):
                _log(f"실제 런 {i}/{len(futs)} · 재시작 {len(r['restarts'])} · {r['sec']:.0f}s", t0)
        timings["ga"] = time.time() - t1
        bests = [[res[(k, s)]["best"] for s in seeds] for k in range(nf_)]
        pr = procedure(E, wins, bests)
        lf = [ex.submit(landscape_task, {"fold": k, "base": c}) for k, c in enumerate(pr["cfgs"])]
        for f, (k, c) in sfut.items():
            if c >= r_chunks:
                scr[k].append(f.result())
        timings["screen"] = time.time() - t1
        for f in cf.as_completed(pf):
            r = f.result()
            pools[r["fold"]].append(r)
        timings["pool"] = time.time() - t1
        _log("선별·풀 끝", t0)
        for i, f in enumerate(cf.as_completed(nfut), 1):
            nulls_b.append(f.result())
            if i % 5 == 0 or i == len(nfut):
                _log(f"모양 섞은 널 {i}/{len(nfut)}", t0)
        timings["null_b"] = time.time() - t1
        land = [row for f in lf for row in f.result()]
        timings["landscape"] = time.time() - t1

    # --- 후처리(부모) ---
    t1 = time.time()
    tests = [te for _, te in wins]
    out_cfgs = pr["cfgs"]
    st = E.stitched(out_cfgs, tests)
    tr_df = trade_frame(E, st, out_cfgs)
    v = gf.block_boot(np.ascontiguousarray(st["monthly"]), 3, CI_BOOT, CI_SEED)
    ci_lo, ci_hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    out_feas = []                                    # 2단계 M6: 출력 config 의 학습창 진입·벌점·실현 가능
    for c, (tr, _) in zip(out_cfgs, wins, strict=True):
        r_ = E.evaluate(c, tr, counts_only=True)
        out_feas.append({"n": r_["n"], "pen": r_["pen"], "feasible": r_["feasible"]})
    # (2a) — 건수 선별 실현 가능 풀(고유 config), 하한 미달이면 ✗
    feas_pool = {}
    for k in range(nf_):
        X = np.vstack([c["X"] for c in scr[k]]) if scr[k] else np.zeros((0, DIM))
        gs, seen = [], set()
        for x in X:
            g = decode(x)
            kk = gkey(g)
            if kk not in seen:
                seen.add(kk)
                gs.append(g)
        feas_pool[k] = gs
    pool_feas = {k: len(v) for k, v in feas_pool.items()}
    null_a_ok = all(v >= feas_min for v in pool_feas.values())
    null_a, null_a_entry = pd.DataFrame(), pd.DataFrame()
    if all(v > 0 for v in pool_feas.values()):
        rng_a = np.random.default_rng(NULL_A_SEED)
        rows = []
        for _ in range(n_a):
            cfgs = [feas_pool[k][int(rng_a.integers(len(feas_pool[k])))] for k in range(nf_)]
            s_ = E.stitched(cfgs, tests)
            rows.append({"ann": s_["ann"], "calm_ann": s_["calm_ann"], "n": s_["n"], "expo": s_["expo_mean"],
                         "ann_per_expo": s_["ann"] / s_["expo_mean"] if s_["expo_mean"] > 0 else np.nan,
                         **base_counts(E, s_["rec"])})
        null_a = pd.DataFrame(rows)
        # 2a′ 기준선 정합(기술): 폴드마다 GA 출력과 같은 기준선의 실현 가능 config 만
        same = {k: [g for g in feas_pool[k] if g["entry"] == out_cfgs[k]["entry"]] for k in range(nf_)}
        if all(len(v) for v in same.values()):
            rng_b = np.random.default_rng(NULL_A_SEED + 1)
            null_a_entry = pd.DataFrame([{"ann": E.stitched([same[k][int(rng_b.integers(len(same[k])))] for k in range(nf_)],
                                                            tests)["ann"]} for _ in range(n_a)])
    pool_comp = pool_composition(feas_pool, out_cfgs)
    rs_rows = []
    P = {k: {"X": np.vstack([c["X"] for c in pools[k]]), "fit": np.concatenate([c["fit"] for c in pools[k]])} for k in pools}
    rng_r = np.random.default_rng(RS_SEED)
    for _ in range(n_rs):
        cfgs = []
        for k in range(nf_):
            idx = rng_r.choice(len(P[k]["fit"]), size=min(RS_BEST_OF, len(P[k]["fit"])), replace=False)
            cfgs.append(decode(P[k]["X"][idx[int(np.nanargmax(P[k]["fit"][idx]))]]))
        rs_rows.append({"ann": E.stitched(cfgs, tests)["ann"]})
    rs_df = pd.DataFrame(rs_rows)
    ec_df = pd.DataFrame()
    if all(pool_feas.values()):
        rng_e = np.random.default_rng(EXIT_CTL_SEED)
        ec_rows = []
        for _ in range(n_ec):
            cfgs = []
            for k in range(nf_):
                g = dict(feas_pool[k][int(rng_e.integers(len(feas_pool[k])))])
                g.update({x: out_cfgs[k][x] for x in EXIT_GENES})
                cfgs.append(canonical(g))
            ec_rows.append({"ann": E.stitched(cfgs, tests)["ann"]})
        ec_df = pd.DataFrame(ec_rows)
    nb = pd.DataFrame(nulls_b)
    timings["post_nulls"] = time.time() - t1

    # --- 기술: 모양 끔 짝·선택도, 중앙값, 민감도 ---
    so = E.stitched([shape_off(c) for c in out_cfgs], tests)
    fw_real = fw_extras(E, out_cfgs, tests, st) if FW else {}
    fw_parent = {}
    if FW and os.path.exists(FW_PARENT_VIEWED):        # 리뷰 FW M5: 부모 열람 OOS 거래(종목·주문일) 제외 민감도(사전등록)
        pv = pd.read_csv(FW_PARENT_VIEWED, dtype={"code": str})
        cidx = {c: i for i, c in enumerate(np.asarray(E.codes))}
        I_all = np.asarray(E.I)
        keys = {(cidx[c], E.row(d)) for c, d in zip(pv["code"], pv["order_day"], strict=True) if c in cidx}
        if FWM:                                       # 리뷰 FWM M7: 같은 종목 ± VIEW_TOL 거래일(같은 셋업의 다른 기준선·재진입까지)
            eci, ed = I_all[:, PE.II["ci"]], I_all[:, PE.II["d"]]
            mask = np.zeros(len(I_all), bool)
            for c_, r_ in keys:
                mask |= (eci == c_) & (np.abs(ed - r_) <= VIEW_TOL)
        else:
            mask = np.fromiter(((int(a_), int(b_)) in keys for a_, b_ in zip(I_all[:, PE.II["ci"]], I_all[:, PE.II["d"]])), bool,
                               count=len(I_all))
        ex_ev = np.flatnonzero(mask)
        s_ex = E.stitched(out_cfgs, tests, exclude=ex_ev)
        fw_parent = {"n_parent": len(pv), "n_events_excluded": int(len(ex_ev)), "ann": s_ex["ann"], "n": s_ex["n"],
                     "overlap_trades": int(np.isin(tr_df["ev"].to_numpy(), ex_ev).sum()) if len(tr_df) else 0, "_st": s_ex}
    sel_rows = []
    for k, (c, w) in enumerate(zip(out_cfgs, tests, strict=True)):
        a_, b_ = E.candidates(c, w), E.candidates(shape_off(c), w)
        sel_rows.append({"fold": k, "pass": a_["n_pass"], "pass_shape_off": b_["n_pass"],
                         "selectivity": a_["n_pass"] / b_["n_pass"] if b_["n_pass"] else np.nan})
    taken_so = set(so["rec"][:, PF.Q_EV].astype(int).tolist()) if len(so["rec"]) else set()
    overlap = float(np.mean([e in taken_so for e in tr_df["ev"]])) if len(tr_df) else np.nan
    meds = [median_config(bk) for bk in bests]
    med_ok = all(m is not None for m in meds) and all(E.evaluate(m, tr)["feasible"] for m, (tr, _) in zip(meds, wins, strict=True)) \
        if all(m is not None for m in meds) else False
    med_st = E.stitched(meds, tests) if med_ok else None
    sens = {"비용 1배": E.stitched(out_cfgs, tests, cost_mult=1.0), "보수 p=1": E.stitched(out_cfgs, tests, pmix=1.0),
            "낙관 p=0": E.stitched(out_cfgs, tests, pmix=0.0),
            **{f"미끄러짐 매수 +{x:.1%}": E.stitched(out_cfgs, tests, extra_slip=x) for x in SLIP_SENS},
            **{f"미끄러짐 매수·매도 +{x:.1%}": E.stitched(out_cfgs, tests, extra_slip=x, sell_extra=x) for x in SLIP_SENS},
            "비용 매도 금액 기준": E.stitched(out_cfgs, tests, cost_on_sell=True),
            "같은 날 무작위 순서": E.stitched(out_cfgs, tests, order_seed=ORDER_SEED)}
    if len(tr_df):
        contrib = (tr_df["ret"] * tr_df["alloc"]).to_numpy()
        order_c = np.argsort(-contrib)
        for k in TOPK:
            sens[f"상위 {k}거래 제외"] = E.stitched(out_cfgs, tests, exclude=tr_df["ev"].to_numpy()[order_c[:k]])
        sens["상위 5% 제외"] = E.stitched(out_cfgs, tests,
                                       exclude=tr_df["ev"].to_numpy()[order_c[:max(1, int(round(0.05 * len(tr_df))))]])
    mfe = {"GA": mfe_table(tr_df, pmix)}
    base_pair = pair_exits(E, tr_df, out_cfgs, live_px)
    pyr_pair = pair_exits(E, tr_df, out_cfgs, pyr_toggle)
    clx_rep = climax_report(E, tr_df, out_cfgs)
    sb_rep = stop_basis_report(E, tr_df, out_cfgs)
    fx_rep = fail_exit_report(E, tr_df, out_cfgs)
    re_rep = reentry_report(tr_df, pmix)
    pyr_kinds = pyramid_kind_pairs(E, tr_df, out_cfgs)
    dyn_rep = dynamic_stop_report(E, tr_df, out_cfgs)
    cases, case_charts = case_table(E, out_cfgs, wins)
    idea_flags = idea_data_flags(out_cfgs, wins)
    trade_ci = {}                                    # 2단계 MINOR 4: 거래당 순 2배(p 혼합) 주문일 군집 CI
    if len(tr_df) >= 3:
        mix2 = pmix_net(tr_df, pmix, "net2")
        lo_, hi_, _ = cluster_bootstrap_ci(mix2.to_numpy(), tr_df["order_day"].dt.strftime("%Y%m%d").to_numpy(),
                                          n_boot=CI_BOOT, seed=CI_SEED)
        trade_ci = {"mean": float(mix2.mean()), "lo": lo_, "hi": hi_}
    by_base = {"GA": base_counts(E, st["rec"]),
               "2a 중앙": {b: float(null_a[f"n_{b}"].median()) for b in PE.BASES} if len(null_a) else {}}
    above_u = (tr_df.groupby("base")["above_u"].agg(["mean", "size"]).to_dict("index") if len(tr_df) else {})
    F_fac = pd.read_pickle(R.FACTORS_PKL) if os.path.exists(R.FACTORS_PKL) else None
    dates = E.days[st["t_lo"]:st["t_end"]]
    fac = R.nw_factor(pd.Series(st["ret"], dates), F_fac[R.FACTOR_COLS].reindex(dates)) if F_fac is not None else {}
    fac1 = (R.nw_factor(pd.Series(sens["비용 1배"]["ret"], dates), F_fac[R.FACTOR_COLS].reindex(dates))
            if F_fac is not None else {})
    kospi_exc = np.nan
    if F_fac is not None:
        mk = F_fac["MKT_KOSPI_TVW"].reindex(dates).fillna(0.0).to_numpy()
        ex_k = st["ret"] - np.r_[0.0, st["expo"][:-1]] * mk
        mo_k, _ = E.monthly(ex_k, st["t_lo"])
        kospi_exc = float(mo_k.mean() * 12)
    n_prior = {lab: count_trials(lab, logs_dir=d) for lab, d in (*R.PRIOR_LEDGERS, ("vcp_retest", "research/logs/sepa_explore"),
                                                                  ("wedge", "research/logs/sepa_explore"),
                                                                  ("wedge_d", "research/logs/sepa_explore"),
                                                                  *((("pattern_ga", OUT_DIR),) if FW else ()),
                                                                  *((("pattern_ga_fw", OUT_DIR),) if FWL else ()))}   # 리뷰 FW M4
    n_led = count_trials(LABEL, logs_dir=led_dir)
    n_dsr_a = n_led + sum(n_prior.values())
    dsr = {}
    parent_uniq = 0
    if FW:                                            # 리뷰 FW M4: 부모 PATTERN_GA 런 원장의 평가 고유 config 합(겹침 무시 — 위쪽 경계)
        for sub_ in ("pattern_ga_runs", *(("pattern_ga_fw_runs",) if FWL else ())):
            pr_path = os.path.join(OUT_DIR, sub_, "TRIALS.jsonl")
            if os.path.exists(pr_path):
                with open(pr_path, encoding="utf-8") as fh:
                    parent_uniq += sum(int(json.loads(x)["config"].get("n_unique_configs_evaluated", 0)) for x in fh if x.strip())
    for tag, N in (("변형", n_dsr_a), ("구조", DSR_STRUCTS),
                   *((("부모·이번 고유 config", n_dsr_a + parent_uniq + len(hashes)),) if FW else ())):
        d_ = deflated_sharpe_from_sample(st["monthly"], N) if len(st["monthly"]) > 2 else None
        dsr[tag] = {"N": N, "prob": float(d_["prob_deflated_sharpe"]) if d_ else np.nan,
                    "sr": float(d_["observed_sharpe"]) if d_ else np.nan}
    pbo = []
    for k, (tr, _) in enumerate(wins):
        cand = {}
        for s in seeds:
            for fv, g in res[(k, s)].get("top", []):
                cand.setdefault(gkey(g), (fv, g))
        top = sorted(cand.values(), key=lambda v: -v[0])[:TOP_PBO]
        if len(top) >= 2:
            M = np.column_stack([E.evaluate(g, tr, want_monthly=True)["monthly"] for _, g in top])
            nbk = min(PBO_BLOCKS, (M.shape[0] // 2) * 2)
            pbo.append(pbo_cscv(M, n_blocks=nbk, metric="sharpe", std_floor=1e-6)["pbo"] if nbk >= 2 else np.nan)
        else:
            pbo.append(np.nan)
    ws = R.wstats(R.expand(tr_df, pmix), n_dsr_a) if len(tr_df) else {"n": 0}
    fw_b6 = {}
    if FW:                                            # 리뷰 FW M3: 손익비를 승률과 같은 부분집합(본전 제외)으로 · 본전 비율 · 불타기 끔 판
        def b6_stats(t_):
            if t_ is None or not len(t_):
                return {"n": 0}
            xx_ = R.expand(t_, pmix)
            ww_, n2_ = xx_["w"].to_numpy(float), xx_["net2"].to_numpy(float)
            wi, lo_m = n2_ >= BE_BAND, n2_ <= -BE_BAND
            glb = (float(np.average(n2_[wi], weights=ww_[wi]) / -np.average(n2_[lo_m], weights=ww_[lo_m]))
                   if wi.any() and lo_m.any() else np.nan)
            wr = float(ww_[wi].sum() / (ww_[wi].sum() + ww_[lo_m].sum())) if (wi | lo_m).any() else np.nan
            return {"n": int(len(t_)), "gl_ex_be": glb, "win_ex_be": wr, "be_share": float(ww_[~(wi | lo_m)].sum() / ww_.sum()),
                    "need": 1.0 / (1.0 + glb) + B6_MARGIN if np.isfinite(glb) else np.nan}
        np_cfgs = [dict(c, pyramid="off") for c in out_cfgs]
        st_np = E.stitched(np_cfgs, tests)
        fw_b6 = {"GA": b6_stats(tr_df), "불타기 끔(첫 단위만)": b6_stats(trade_frame(E, st_np, np_cfgs)), "np_ann": st_np["ann"]}
        if "_st" in fw_parent:                       # 리뷰 FWL M4: 열람 거래 제외판의 바 1·6
            s_ex = fw_parent.pop("_st")
            v_ex = gf.block_boot(np.ascontiguousarray(s_ex["monthly"]), 3, CI_BOOT, CI_SEED) if len(s_ex["monthly"]) else np.array([np.nan])
            fw_parent["ci_lo"] = float(np.nanpercentile(v_ex, 2.5))
            fw_parent["b6"] = b6_stats(trade_frame(E, s_ex, out_cfgs))
        n_fw_strict = int((tr_df["shape"] == "하락 쐐기").sum()) if len(tr_df) else 0
        fw_b6["strict_share"] = n_fw_strict / len(tr_df) if len(tr_df) else np.nan
    timings["post"] = time.time() - t1

    # --- 바 산술 ---
    def p95(s):
        return float(np.nanpercentile(s, 95)) if len(s) else np.nan

    w2 = np.nan                                       # 바 6 승률 = 2배 순 기준, 본전(|순| < 0.5%) 제외(쐐기 시험과 같은 처리)
    if ws.get("n"):
        xx = R.expand(tr_df, pmix)
        ww, n2 = xx["w"].to_numpy(float), xx["net2"].to_numpy(float)
        wn, ls = ww[n2 >= BE_BAND].sum(), ww[n2 <= -BE_BAND].sum()
        w2 = float(wn / (wn + ls)) if wn + ls > 0 else np.nan
    gl = ws.get("gl_ratio2", np.nan)
    q2a, q2b = (FW_Q_2A, FW_Q_2B) if FW else (95.0, 95.0)
    if FW:
        gl = fw_b6["GA"].get("gl_ex_be", np.nan)      # 리뷰 FW M3: 승률과 같은 부분집합
    be_need = 1.0 / (1.0 + gl) + B6_MARGIN if np.isfinite(gl) else np.nan
    fold_judge = [n >= FOLD_MIN_FILLS for n in st["fold_n"]]
    calm_folds = [k for k, w in enumerate(tests) if str(w.lo)[:4] in ("2022", "2023", "2024")]
    b4_judgeable = all(fold_judge[k] for k in calm_folds) and len(calm_folds) == 3
    bars = {"b1": bool(np.isfinite(st["ann"]) and st["ann"] > 0 and ci_lo > 0),
            "b2a": bool(null_a_ok and len(null_a) and st["ann"] > float(np.nanpercentile(null_a["ann"], q2a))),
            "b2b": bool(len(nb) and st["ann"] > float(np.nanpercentile(nb["ann"], q2b))),
            "b3": bool(np.isfinite(fac.get("alpha_t", np.nan)) and fac["alpha_t"] > 2.0),
            "b4": bool(b4_judgeable and np.isfinite(st["calm_ann"]) and st["calm_ann"] > 0
                       and np.isfinite(st["ex2020_21_ann"]) and st["ex2020_21_ann"] > 0
                       and np.isfinite(st["ex2025_26_ann"]) and st["ex2025_26_ann"] > 0),
            "b6": bool(ws.get("n", 0) >= B6_N and np.isfinite(gl) and gl >= B6_GL and np.isfinite(w2) and w2 > be_need)}
    bars["b2"] = bars["b2a"] and bars["b2b"]
    bars["all"] = all(bars[k] for k in ("b1", "b2", "b3", "b4", "b6"))
    if FW and fw_parent:                              # 리뷰 FWL M4: 열람 거래 제외판 바 1·6 (FWL 은 '탐색' 문구에 둘 다 필요)
        b6e = fw_parent.get("b6", {})
        bars["b1_ex"] = bool(np.isfinite(fw_parent["ann"]) and fw_parent["ann"] > 0 and fw_parent.get("ci_lo", -1.0) > 0)
        bars["b6_ex"] = bool(b6e.get("n", 0) >= B6_N and np.isfinite(b6e.get("gl_ex_be", np.nan)) and b6e["gl_ex_be"] >= B6_GL
                             and np.isfinite(b6e.get("win_ex_be", np.nan)) and b6e["win_ex_be"] > b6e["need"])
        if FWL:
            bars["all"] = bars["all"] and bars["b1_ex"] and bars["b6_ex"]
    b6_note = {}
    if FWL:                                           # 정정 FWL-2: 건수만 미달(손익비·승률은 통과)이면 '하한 결합 — 판단 불가'(통과 아님)
        b6_note["전체"] = ("하한 결합 — 판단 불가" if ws.get("n", 0) < B6_N and np.isfinite(gl) and gl >= B6_GL
                          and np.isfinite(w2) and w2 > be_need else "")
        e6_ = fw_parent.get("b6", {}) if fw_parent else {}
        b6_note["열람 제외"] = ("하한 결합 — 판단 불가" if e6_.get("n", 0) < B6_N and np.isfinite(e6_.get("gl_ex_be", np.nan))
                             and e6_["gl_ex_be"] >= B6_GL and e6_.get("win_ex_be", 0) > e6_.get("need", 1) else "")
    regions = [user_region(c) for c in out_cfgs]
    region_ovl = [region_overlap(c) for c in out_cfgs]
    region_trade = trade_region_share(E, tr_df)
    n_uniq = len(hashes)
    elapsed = time.time() - t0
    V = locals()
    L = render(V)
    if FW:
        L = L[:3] + fw_lines(V) + L[3:]
    with open(os.path.join(out_dir, RESULTS_MD), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    # --- 산출물 ---
    pd.DataFrame([{"fold": k, "seed": s, "train_fit": r["fit"], "train_n": r["n"], "uniq": r["uniq"], "sec": r["sec"],
                   "restarts": len(r["restarts"]), "why": ",".join(x["why"] for x in r["restarts"]),
                   "regimes": ",".join(x["regime"] for x in r["restarts"]),
                   "zero_frac": ",".join(f"{x['zero_frac']:.2f}" for x in r["restarts"]),
                   **{g: str(r["best"].get(g)) for g in GENE_NAMES}}
                  for (k, s), r in res.items()]).to_csv(os.path.join(sub_dir, "runs.csv"), index=False)
    null_a.to_csv(os.path.join(sub_dir, "null_2a.csv"), index=False)
    null_a_entry.to_csv(os.path.join(sub_dir, "null_2a_entry.csv"), index=False)
    nb.to_csv(os.path.join(sub_dir, "null_2b.csv"), index=False)
    rs_df.to_csv(os.path.join(sub_dir, "random_search.csv"), index=False)
    ec_df.to_csv(os.path.join(sub_dir, "exit_control.csv"), index=False)
    pd.DataFrame(land).to_csv(os.path.join(sub_dir, "landscape.csv"), index=False)
    if len(tr_df):
        tr_df.to_csv(os.path.join(sub_dir, "oos_trades.csv"), index=False)
    pd.DataFrame({"ret": st["ret"], "bench": st["bench"], "expo": st["expo"]}, index=dates).to_csv(
        os.path.join(sub_dir, "oos_daily.csv"))
    with open(os.path.join(sub_dir, "procedure.json"), "w", encoding="utf-8") as fh:
        json.dump({"folds": [list(f) for f in folds_spec], "configs": [{k: (None if v is None else str(v) if v == math.inf else v)
                                                                       for k, v in c.items()} for c in out_cfgs],
                   "reeval_fits": pr["re_fits"], "out_feasibility": out_feas, "regions": regions, "region_overlap": region_ovl,
                   "p": pmix, "provenance": run_provenance(args.cache)}, fh, ensure_ascii=False, indent=1, default=str)
    cases.to_csv(os.path.join(sub_dir, "cases.csv"), index=False)
    with open(os.path.join(sub_dir, "chart_points.json"), "w", encoding="utf-8") as fh:
        json.dump({"oos_random12": chart_points(E, tr_df, out_cfgs), "cases": case_charts}, fh, ensure_ascii=False, indent=1)
    print(f"\n=== 패턴 GA — {elapsed / 60:.1f}분 · 고유 config {n_uniq:,} · 원장 {n_led}줄 · 폴드 {nf_} · 시드 {len(seeds)} · 예산 {budget:,} ===")
    print("시간(s): " + " · ".join(f"{k} {v:.0f}" for k, v in timings.items()))
    print(f"OOS 진입 {st['n']} · 폴드별 {st['fold_n']} · 실현 가능 풀 {pool_feas} · 널 2a {len(null_a)} · 2b {len(nb)} · "
          f"무작위 탐색 {len(rs_df)} · 청산 대조 {len(ec_df)}")
    print(f"→ {os.path.join(out_dir, RESULTS_MD)}")
    return 0


def base_counts(E: Engine, rec: np.ndarray) -> dict:
    """기준선별 진입 수(2단계 MINOR 14)."""
    if not len(rec):
        return {f"n_{b}": 0 for b in PE.BASES}
    tb = np.asarray(E.I)[rec[:, PF.Q_EV].astype(int), PE.II["table"]] % len(PE.BASES)
    return {f"n_{b}": int((tb == i).sum()) for i, b in enumerate(PE.BASES)}


def pool_composition(feas_pool: dict, out_cfgs: "list[dict]") -> pd.DataFrame:
    """2단계 M1-2: 실현 가능 풀 구성(기준선·단계·θ·창·켠 모양 구간 수) — 폴드별 비율과 GA 출력 값."""
    rows = []
    for k, gs in feas_pool.items():
        if not gs:
            continue
        df = pd.DataFrame([{"entry": g["entry"], "stage": g["stage"], "theta": g["theta"], "ceil_win": g["ceil_win"],
                            "shape_on": sum(g[x] is not None for x in ("sup_hw", "slo_hw", "rho_max", "span_hw", "ceil_min",
                                                                       "depth_max", "dry_max"))} for g in gs])
        o = out_cfgs[k]
        o_on = sum(o[x] is not None for x in ("sup_hw", "slo_hw", "rho_max", "span_hw", "ceil_min", "depth_max", "dry_max"))
        for col, val in (("entry", o["entry"]), ("stage", o["stage"]), ("theta", o["theta"]), ("ceil_win", o["ceil_win"]),
                         ("shape_on", o_on)):
            rows.append({"fold": k, "item": col, "out": val,
                         "pool": " · ".join(f"{a} {b:.0%}" for a, b in df[col].value_counts(normalize=True).sort_index().items())})
    return pd.DataFrame(rows)


def run_provenance(cache: str) -> dict:
    """2단계 MINOR 10: git HEAD·작업 트리 깨끗함·캐시 meta 해시·코드 해시·numba·cmaes 버전."""
    import subprocess  # noqa: PLC0415

    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            return ""
    try:
        import cmaes  # noqa: PLC0415
        cv = cmaes.__version__
    except ImportError:
        cv = None
    return {"git_head": sh("git", "rev-parse", "HEAD"), "git_clean": sh("git", "status", "--porcelain") == "",
            "cache_meta_hash": meta_hash(cache) if os.path.exists(os.path.join(cache, "meta.json")) else None,
            "code_hash": code_hash(), "numba": gf.numba.__version__ if gf.HAVE_NUMBA else None, "cmaes": cv}


def chart_points(E: Engine, t: pd.DataFrame, cfgs: "list[dict]") -> list:
    """MINOR 10: OOS 트레이드 무작위 :data:`CHART_N` 건(seed :data:`CHART_SEED`) — 반전점·선·주문·체결·손절·청산·봉(그리기용)."""
    if t is None or t.empty:
        return []
    rng = np.random.default_rng(CHART_SEED)
    pick = rng.choice(len(t), size=min(CHART_N, len(t)), replace=False)
    out = []
    F = np.asarray(E.F_orig)
    for i in sorted(pick):
        r = t.iloc[i]
        ci, d = int(r["ci"]), int(r["d_row"])
        if int(r["table"]) >= PE.N_TABLES:            # 13세대 가족 사건(P·C) — 쐐기 반전점 없음, 신호 전 60봉
            th, cw, bn = None, None, str(r.get("fam", "P"))
            pt = {}
            lo = max(0, d - 60)
        else:
            th, cw, bn = PE.table_key(int(r["table"]))
            pt = PE.pattern_points(np.asarray(E.H), np.asarray(E.L), np.asarray(E.C), ci, d - 1, th, cw)
            lo = max(0, (pt.get("i0", d) if pt.get("i0", -1) >= 0 else d) - 20)
        hi = min(len(E.days) - 1, int(E.end_row), int(r["x_row"]) + 10)   # 검증 MUST-FIX 5: 봉인(잘린 end_row) 너머 봉을 그리지 않는다

        def dd(row):
            return str(E.days[min(int(row), len(E.days) - 1)].date())
        bars = [{"date": dd(j), "o": float(E.O[ci, j]), "h": float(E.H[ci, j]), "l": float(E.L[ci, j]), "c": float(E.C[ci, j])}
                for j in range(lo, hi + 1)]
        out.append({"code": str(r["code"]), "fold": int(r["fold"]), "table": {"theta": th, "ceil_win": cw, "entry": bn},
                    "signal_day": dd(d - 1), "order_day": dd(d), "shape": r["shape"],
                    "points": [{"date": dd(p["row"]), "price": p["price"], "kind": p["kind"]} for p in pt.get("points", [])],
                    "upper": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt.get("upper", {}).items()},
                    "lower": {k: {"date": dd(v["row"]), "price": v["price"]} for k, v in pt.get("lower", {}).items()},
                    "level": float(F[int(r["ev"]), PE.FI["level"]]), "order_px": float(F[int(r["ev"]), PE.FI["order"]]),
                    "entry_px": float(r["entry_px"]), "stop_px": float(r["stop_px"]), "u_d": float(r["u_d"]),
                    "order_vs_upper": float(r["order_vs_u"]), "order_above_upper": bool(r["above_u"]),
                    "exit": {"date": dd(r["x_row"]), "price": float(r["exit_px"]),
                             "reason": str(r["exit_reason"]), "ambiguous_entry_day": bool(r["amb"]),
                             "partial": bool(r["partial"]), "added": bool(r["added"])},
                    "bars": bars})
    return out


# ---------------------------------------------------------------------------
# 10. 서식
# ---------------------------------------------------------------------------

CLAIM_SCOPE = """> **전 바 통과가 확립하는 것:** 2020-01 ~ 2026-08 한국 ADV ≥ 10억에서, 이 연구 계열이 같은 기간을 본 뒤 설계한 패턴·청산 메뉴의 값을 직전 3년 학습창에서 매년 다시 고르는 **절차**가 남긴 의사-표본 밖 기록 하나다. 그 기록이 다음 셋을 넘었다: (a) 같은 메뉴에서 최적화 없이 뽑은 설정들의 95 백분위, (b) 모양 서술자를 섞은 데이터에서 같은 절차를 돌린 결과의 95 백분위, (c) 팩터 절편 t > 2. 이 데이터에서 모양 유전자가 맥락·청산 적재 이상을 더했다는 **약한** 증거다.
>
> **확립하지 않는 것:** (i) 특정 모양(하락 쐐기 등)이 유효하다는 것. 특히 그 모양이 사용자 착상 영역 안이면, 착상 자체가 OOS 연도를 본 데서 나왔으므로 독립 증거가 아니다. (ii) VCP·쐐기 시험의 판정을 바꾸는 것. (iii) 마지막 학습창 config 의 특정 값이 유효하다는 것. 그 config 는 표본 밖 기록이 없다. (iv) 2020~21·2025~26 두 상승 묶음 밖에서 통한다는 것(2022~24 조각을 따로 본다). (v) 이 계열 수십 개 변형·10여 개 검증 구조 뒤의 다중성을 넘는다는 것. (vi) 용량·미끄러짐(10억 종목 돌파 체결)을 넘는다는 것.
>
> **불통과가 확립하는 것:** "이 메뉴에서 매년 다시 고르는 절차는 의사-OOS 에서 같은 메뉴 기준선·모양 섞은 널·팩터를 넘지 못했다." 다른 시드·적합도·기준선의 ✓ 는 판정에 쓰지 않는다. 같은 데이터에서 메뉴를 고쳐 다시 돌리지 않는다.
>
> **허용하는 행동:** 통과해도 전진 페이퍼(고정 config, 페이퍼 판정 규칙 사전등록)까지만이다. 독립 표본(미국)에 같은 사전등록을 한 번 적용하는 것이 다음 증거다."""


def fw_lines(V: dict) -> "list[str]":
    """FW 머리 절(리뷰 FW M3·M5·M6·M9) — 결과 문서 맨 앞."""
    st, nb, b6, fr, fp, out_cfgs = V["st"], V["nb"], V["fw_b6"], V["fw_real"], V["fw_parent"], V["out_cfgs"]
    g = b6.get("GA", {})
    share = b6.get("strict_share", np.nan)
    label = "하락 쐐기" if np.isfinite(share) and share >= 0.5 else "하락 쐐기 계열(유연 경계 — 엄격 하락 쐐기 50% 미만)"
    L = ["## FW 머리 절 (리뷰 FW 필수 수정 반영)", "",
         *(["> **탐색 전용(FWL):** 같은 2020~2026 데이터에서 세 번째로 좁힌 메뉴(부모 → FW → 주도주 고정)다. 바를 다 넘어도 판정이 아니며, "
            "근거는 전진 페이퍼에서만 나온다.", ""] if FWL else []),
         f"> **다중성 미보정(리뷰 FW M5):** 이 메뉴는 부모 `PATTERN_GA` 의 표본 밖 매매·차트를 본 뒤 좁혔다(모양 계열·불타기 필수·본전 0.5/1R). "
         f"명목 p95 바는 보정되지 않은 값이라 판정 문턱을 **2a p{V['q2a']:g} · 2b {len(nb)}벌 최댓값 위**로 올렸다. 통과해도 '탐색' 이다.", "",
         f"- **바 6(사용자 기준, 리뷰 FW M3 — 손익비·승률 모두 본전 |순 2배| < {BE_BAND:.1%} 제외):** OOS {g.get('n', 0)}건 · "
         f"손익비 {_f(g.get('gl_ex_be', np.nan))} (≥ {B6_GL:g}) · 승률 {_f(g.get('win_ex_be', np.nan))} vs 필요 {_f(g.get('need', np.nan))} · "
         f"본전 비율 {_f(g.get('be_share', np.nan))} → {_m(V['bars']['b6'])}"]
    np_ = b6.get("불타기 끔(첫 단위만)", {})
    L += [f"- 불타기 끔(같은 config·첫 단위만, 기술): {np_.get('n', 0)}건 · 손익비 {_f(np_.get('gl_ex_be', np.nan))} · "
          f"승률 {_f(np_.get('win_ex_be', np.nan))} · 연환산 {_pct(b6.get('np_ann', np.nan))}",
          f"- **모양 표기(리뷰 FW M9):** OOS 거래 중 엄격 하락 쐐기(`shape_name`) 비율 {_f(share)} → 이 결과를 부르는 이름: **{label}**. "
          f"폴드별 fw_lo: " + " · ".join(f"{k + 1}: {c.get('fw_lo', np.nan):+.2f}" for k, c in enumerate(out_cfgs))]
    if len(nb) and fr:
        def dist(col):
            v = nb[col].to_numpy(float) if col in nb else np.array([])
            v = v[np.isfinite(v)]
            return f"2b 중앙 {_f(float(np.median(v)), '{:.3f}')} · p95 {_f(float(np.percentile(v, 95)), '{:.3f}')}" if len(v) else "2b 없음"
        L += ["- **노출·불타기·F1 끔 짝(리뷰 FW M6 — 실제 vs 2b 같은 정의):**",
              f"  - 평균 노출 {_f(fr['expo'], '{:.3f}')} ({dist('expo')})",
              f"  - 불타기 발동률 {_f(fr['add_rate'], '{:.3f}')} ({dist('add_rate')})",
              f"  - F1 끔 연환산 {_pct(fr['f1off_ann'])} · F1 이득 {_pct(fr['f1_gain'])} ({dist('f1_gain')})",
              f"  - 노출 1 단위당 연환산: 실제 {_pct(st['ann'] / fr['expo'] if fr['expo'] else np.nan)} · "
              f"2b 중앙 {_pct(float(np.nanmedian(nb['ann'] / nb['expo'])) if 'expo' in nb else np.nan)}"]
    if fp:
        L += [f"- **부모 열람 거래 제외(리뷰 FW M5, 사전등록 민감도):** 부모 OOS {fp['n_parent']}건과 같은 (종목, 주문일) 사건 "
              f"{fp['n_events_excluded']}개 제외 → 연환산 {_pct(fp['ann'])} ({fp['n']}건) · 이번 OOS 중 겹친 거래 {fp['overlap_trades']}건"]
        e6 = fp.get("b6", {})
        bb = V["bars"]
        L += ["", "| 바(리뷰 FWL M4) | 전체 | 열람 거래 제외 |", "|---|---|---|",
              f"| 1 연환산 · CI 하단 | {_pct(st['ann'])} · {_pct(V['ci_lo'])} {_m(bb['b1'])} | {_pct(fp['ann'])} · "
              f"{_pct(fp.get('ci_lo', np.nan))} {_m(bb.get('b1_ex', False))} |",
              f"| 6 n · 손익비 · 승률 vs 필요 | {g.get('n', 0)} · {_f(g.get('gl_ex_be', np.nan))} · {_f(g.get('win_ex_be', np.nan))} vs "
              f"{_f(g.get('need', np.nan))} {_m(bb['b6'])} | {e6.get('n', 0)} · {_f(e6.get('gl_ex_be', np.nan))} · "
              f"{_f(e6.get('win_ex_be', np.nan))} vs {_f(e6.get('need', np.nan))} {_m(bb.get('b6_ex', False))} |", ""]
        nt = V.get("b6_note") or {}
        if any(nt.values()):
            L += ["- 바 6 비고(정정 FWL-2): " + " · ".join(f"{k}: {v}" for k, v in nt.items() if v), ""]
    d3 = V["dsr"].get("부모·이번 고유 config", {})
    L += [f"- DSR(부모·이번 고유 config N {d3.get('N', 0):,}): {_f(d3.get('prob', np.nan))}", ""]
    return L


def _m(ok) -> str:
    return "✓" if ok else "✗"


def render(V: dict) -> "list[str]":  # noqa: C901
    st, bars, nb, null_a, ws = V["st"], V["bars"], V["nb"], V["null_a"], V["ws"]
    out_cfgs, wins, regions = V["out_cfgs"], V["wins"], V["regions"]
    L = [f"# 패턴 GA — 실행 결과 {date.today().isoformat()} (러너 자동 생성, 판정 아님)", "",
         f"사전등록 `{PREREG_DOC}` 의 공간·적합도·절차·널·합격 바를 그대로 계산한 숫자다. ✓/✗ 는 산술 비교일 뿐이다. "
         "재현: `uv run --with numba --with cmaes==0.13.1 --extra pg python research/experiments/pattern_ga_run.py`.", ""]
    if V["debug"]:
        L += [f"> **디버그 실행** — 폴드 {len(wins)} · 시드 {len(V['seeds'])} · 예산 {V['budget']} · 널 2a {len(null_a)} · 2b {len(nb)}, "
              f"출력 `{V['out_dir']}`. 측정으로 쓰지 않는다.", ""]
    def rl(rr):
        return " · ".join(f"폴드{k + 1} {'**' + ','.join(n for n in ('W', 'WD') if r[n]) + '**' if r['any'] else '밖'}"
                          for k, r in enumerate(rr))
    share = V["region_trade"]
    contam = bool(np.isfinite(share) and share >= 0.5)
    L += ["**착상 영역(정정 1 C2·정정 2·정정 3, 결과 전 좌표 — 2단계 M3 세 단계):**",
          f"1. config 영역(엄격): {rl(regions)}",
          f"2. config 겹침(모양 구간이 하락 쐐기 사분면과 겹침 ∧ 기준선 일치): {rl(V['region_ovl'])}",
          "4. 착상 거래 연도(정정 4~7 유전자 × 그 거래 연도가 든 학습·시험창): " + (" · ".join(
              f"폴드{k + 1} **{'·'.join(f)}: 착상과 같은 데이터**" for k, f in enumerate(V["idea_flags"]) if f) or "해당 없음"),
          f"3. 트레이드 기준(OOS 체결 중 사건 자체가 착상 영역): **{_f(share, '{:.0%}')}**" + (
              " → **착상 오염 가능 — 독립 표본 필요**(착상이 OOS 연도를 본 데서 나왔다)." if contam else "")
          + (" · 1 또는 2 에 든 폴드도 같은 표시를 붙여 읽는다." if any(r["any"] for r in regions + V["region_ovl"]) else ""), "",
          "## 주장 범위 (리뷰 정본 문구 — 정정 1 C2)", "", CLAIM_SCOPE, "",
          "판정 되살리기 금지: GA 최적이 VCP·쐐기·쐐기 D 정의의 이웃에 떨어져도 그 판정을 바꾸지 않는다. 신뢰 근거는 전진 페이퍼·독립 표본뿐이다.", "",
          "## 합격 바 산술 — 절차 출력(시드 최적 재평가 1등) 이어 붙인 OOS, 2배 비용, 정합 바스켓 대조", "",
          "| 바 | 문턱 | 값 | 충족 |", "|---|---|---|---|",
          f"| 1 | 연환산 초과 > 0 ∧ 월 블록 95% CI 하단 > 0 | {_pct(st['ann'])} [{_pct(V['ci_lo'])}, {_pct(V['ci_hi'])}] | {_m(bars['b1'])} |",
          f"| 2a | > 같은 메뉴 무최적화 실현 가능 설정 p{V['q2a']:g} (폴드별 실현 가능 고유 config ≥ {V['feas_min']} 아니면 판단 불가 = ✗) | "
          f"{_pct(st['ann'])} vs {_pct(float(np.nanpercentile(null_a['ann'], V['q2a'])) if len(null_a) else np.nan)} "
          f"(백분위 {_f(float((null_a['ann'] < st['ann']).mean()) if len(null_a) else np.nan)})"
          + ("" if V["null_a_ok"] else " · **실현 가능 풀 하한 미달 → 판단 불가**")
          + (" · **출력이 널 지지 밖**(실현 불가능 폴드 " + ",".join(str(k + 1) for k, o in enumerate(V["out_feas"]) if not o["feasible"]) + ")"
             if any(not o["feasible"] for o in V["out_feas"]) else "") + f" | {_m(bars['b2a'])} |",
          f"| 2b | > 모양 섞은 널(절차 전체) p{V['q2b']:g} | {_pct(st['ann'])} vs {_pct(float(np.nanpercentile(nb['ann'], V['q2b'])) if len(nb) else np.nan)} "
          f"(백분위 {_f(float((nb['ann'] < st['ann']).mean()) if len(nb) else np.nan)}) · 해석(2단계 M5): 통과는 '모양 열이 급등·쉼·손절 거리·"
          f"기준선 값에 이미 든 모양 정보 **이상**을 더했다' 는 뜻(섞지 않는 열에 모양 정보가 일부 남아 널이 더 세다 — 보수 쪽) | {_m(bars['b2b'])} |",
          f"| 3 | 팩터 α t > 2 (NW 5, {', '.join(['MKT_KOSPI_TVW', 'MKT_KOSDAQ_TVW', 'UMD', 'SMB_ADV', 'LOWVOL'])}) | "
          f"t {_f(V['fac'].get('alpha_t', np.nan))} · α {_pct(V['fac'].get('alpha_ann', np.nan))} (1배 t {_f(V['fac1'].get('alpha_t', np.nan))}) | {_m(bars['b3'])} |",
          f"| 4 | 2022~24 조각 > 0 ∧ 2020~21 뺀 값 > 0 ∧ 2025~26 뺀 값 > 0 ∧ 시험 2022·2023·2024 폴드 체결 ≥ {FOLD_MIN_FILLS}"
          f"(아니면 판단 불가 = ✗, 2단계 M4) | {_pct(st['calm_ann'])} · {_pct(st['ex2020_21_ann'])} · {_pct(st['ex2025_26_ann'])}"
          f"{'' if V['b4_judgeable'] else ' · **판단 불가**'} | {_m(bars['b4'])} |",
          f"| 6 | OOS 거래 ≥ {B6_N} ∧ 손익비(2배) ≥ {B6_GL:g} ∧ 승률(본전 제외) > 손익분기 + 5%p | n {ws.get('n', 0)} · "
          f"{_f(V['gl'])}{'(본전 제외 — 리뷰 FW M3)' if FW else ''} · {_f(V['w2'])} vs {_f(V['be_need'])} | {_m(bars['b6'])} |",
          f"| **전부** | 1·2(a∧b)·3·4·6{' + 열람 제외판 1·6(FWL)' if FWL else ''} | | **{_m(bars['all'])}** |", "",
          f"바 5(기술): 폴드 내 CSCV PBO {[_f(x) for x in V['pbo']]} (리뷰 M3 — 노이즈에서도 0.37~0.59, 과적합 부재의 증거 아님). WFE 삭제. "
          f"DSR(기술): N = 보고 변형 {V['dsr']['변형']['N']} → P {_f(V['dsr']['변형']['prob'], '{:.3f}')} · N = 검증 구조 {DSR_STRUCTS} → "
          f"P {_f(V['dsr']['구조']['prob'], '{:.3f}')}. 폴드 체결 < {FOLD_MIN_FILLS} = 판단 불가: "
          f"{[f'폴드{k + 1}' for k, ok in enumerate(V['fold_judge']) if not ok] or '없음'}.", "",
          "## 폴드별 절차 출력", "",
          "| 폴드 | 학습 | 시험 | 모양 이름 | 재평가 적합도(시드 최적 10) | 학습 진입 · 벌점 · 실현 가능(M6) | OOS 연환산 | OOS 체결 | "
          "착상 영역 엄격/겹침 | config |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (c, (tr, te)) in enumerate(zip(out_cfgs, wins, strict=True)):
        rf = V["pr"]["re_fits"][k]
        of_ = V["out_feas"][k]
        L.append(f"| {k + 1} | {tr.lo}~{tr.hi} | {te.lo}~{te.hi} | {config_shape(c)} | 최고 {_f(max(rf), '{:+.3f}')} · 중앙 "
                 f"{_f(float(np.median(rf)), '{:+.3f}')} | {of_['n']} · {_f(of_['pen'], '{:.3f}')} · {_m(of_['feasible'])} | "
                 f"{_pct(st['fold_ann'][k])} | {st['fold_n'][k]} | "
                 f"{','.join(n for n in ('W', 'WD') if regions[k][n]) or '밖'}/{','.join(n for n in ('W', 'WD') if V['region_ovl'][k][n]) or '밖'} "
                 f"| `{gene_str(c)}` |")
    shapes = pd.Series([config_shape(V["res"][(k, s)]["best"]) for k in range(len(wins)) for s in V["seeds"]])
    L += ["", "**모양 선택의 폴드·시드 분포(사전 예상 — 해석은 모양 섞은 널의 같은 분포와 함께, MINOR 12):** "
          + " · ".join(f"{k} {v}" for k, v in shapes.value_counts().items()),
          f"폴드 간 ŝ_up 중심 표준편차 {_f(float(np.std([c['sup_c'] for c in out_cfgs if c['sup_hw'] is not None])) if sum(c['sup_hw'] is not None for c in out_cfgs) > 1 else np.nan)}"
          f" · 널 2b 중앙 {_f(float(nb['sup_c_sd'].median()) if len(nb) else np.nan)}.",
          f"BIPOP 재시작(런당 평균) {np.mean([len(r['restarts']) for r in V['res'].values()]):.1f} — 0 인 런 "
          f"{sum(len(r['restarts']) <= 1 for r in V['res'].values())}개는 'BIPOP 아님'(M4-2). 고유 config {V['n_uniq']:,}. 재시작 평균 = 그 학습창 "
          f"실현 가능 풀(앞 {V['restart_screen']:,}개 선별: " + " · ".join(f"폴드{k + 1} {len(v)}" for k, v in V["restart_x"].items())
          + ") — 재시작 표본 중 진입 0건 비율 평균 "
          + _f(_mean_or_nan([x["zero_frac"] for r in V["res"].values() for x in r["restarts"][1:]]), "{:.0%}")
          + f" (널 2b {_f(float(nb['restart_zero_frac'].mean()) if len(nb) and 'restart_zero_frac' in nb else np.nan, '{:.0%}')}).", "",
          "## 널·대조", "",
          "| 분포 | 벌 | 중앙 | p95 | 실제 백분위 |", "|---|---|---|---|---|"]
    ape = st["ann"] / st["expo_mean"] if st["expo_mean"] > 0 else np.nan
    for nm, df, col, ref in (("(2a) 무최적화 실현 가능 설정", null_a, "ann", st["ann"]), ("(2b) 모양 섞은 널 — 절차 전체", nb, "ann", st["ann"]),
                             ("2a′ 기준선 정합(기술)", V["null_a_entry"], "ann", st["ann"]),
                             ("노출 정규화 2a = 연환산 초과 / 평균 노출(기술)", null_a, "ann_per_expo", ape),
                             ("무작위 탐색 best-of-2,000 — 실현 불가능 포함 균등 풀(기술)", V["rs_df"], "ann", st["ann"]),
                             ("청산 전용 대조(GA 청산 고정, 진입·모양 무작위 — 기술)", V["ec_df"], "ann", st["ann"])):
        s = df[col] if len(df) and col in df else pd.Series(dtype=float)
        L.append(f"| {nm} | {len(s)} | {_pct(float(s.median()) if len(s) else np.nan)} | {_pct(float(np.nanpercentile(s, 95)) if len(s) else np.nan)} | "
                 f"{_f(float((s < ref).mean()) if len(s) else np.nan)} |")
    L += ["", f"건수 선별(폴드당 균등 {V['screen_n']:,}, 수익 미계산) 실현 가능 고유 config: " + " · ".join(
        f"폴드{k + 1} {v:,}" for k, v in sorted(V["pool_feas"].items())) + f" (하한 {V['feas_min']}"
          + (" — 충족)." if V["null_a_ok"] else " — **미달: 2a 판단 불가 = ✗**)."), "",
          "실현 가능 풀 구성 vs GA 출력(2단계 M1-2):", "", "| 폴드 | 항목 | GA 출력 | 풀 비율 |", "|---|---|---|---|"]
    for r in V["pool_comp"].to_dict("records") if len(V["pool_comp"]) else []:
        L.append(f"| {r['fold'] + 1} | {r['item']} | {r['out']} | {r['pool']} |")
    L += ["", "기준선별 OOS 체결 수(MINOR 14): GA " + " · ".join(f"{b} {V['by_base']['GA'].get(f'n_{b}', 0)}" for b in PE.BASES)
          + " · 2a 벌 중앙 " + " · ".join(f"{b} {_f(V['by_base']['2a 중앙'].get(b, np.nan), '{:.0f}')}" for b in PE.BASES)]
    so = V["so"]
    gain = st["ann"] - so["ann"]
    L += ["", f"**모양 비활성 짝(M6-1):** 출력 {_pct(st['ann'])} − 모양 끔 {_pct(so['ann'])} = {_pct(gain)} · 널 2b 같은 차이 p95 "
          f"{_pct(float(np.nanpercentile(nb['shape_gain'], 95)) if len(nb) else np.nan)} (백분위 "
          f"{_f(float((nb['shape_gain'] < gain).mean()) if len(nb) else np.nan)}). 모양 선택도(통과 사건 / 모양 끔 통과): "
          + " · ".join(f"폴드{r['fold'] + 1} {_f(r['selectivity'])}" for r in V["sel_rows"])
          + f" · 출력 체결 중 모양 끔 config 도 산 비율 {_f(V['overlap'])}.", ""]
    tc = V["trade_ci"]
    L += ["## 트레이드·청산 (OOS, p 혼합, 2배 비용)", "",
          f"- 거래당 순 2배 평균 {_pct(tc.get('mean', np.nan))} [주문일 군집 95% CI {_pct(tc.get('lo', np.nan))}, {_pct(tc.get('hi', np.nan))}] (MINOR 4)",
          "- 주문가 > 상단 추세선(이미 쐐기 밖 매수, MINOR 2) 비율 — 기준선별: " + (" · ".join(
              f"{b} {v['mean']:.0%}/{int(v['size'])}" for b, v in V["above_u"].items()) or "—"),
          f"- n {ws.get('n', 0)} · 승/본전/패 {_f(ws.get('win', np.nan))}/{_f(ws.get('be', np.nan))}/{_f(ws.get('lossc', np.nan))} · "
          f"순 2배 평균 {_pct(ws.get('net2_mean', np.nan))} · 손익비(1배 {_f(ws.get('gl_ratio', np.nan))} · 2배 {_f(ws.get('gl_ratio2', np.nan))}) · "
          f"평균 손실 R {_f(ws.get('avg_loss_R', np.nan), '{:+.2f}')} · 보유 중앙 {_f(ws.get('hold_median', np.nan), '{:.0f}')}일 · 청산 {ws.get('exits', {})}",
          f"- 노출 평균 {_f(float(np.mean(st['expo'])) if len(st['expo']) else np.nan)} → 노출당 연환산 초과 "
          f"{_pct(st['ann'] / float(np.mean(st['expo'])) if len(st['expo']) and np.mean(st['expo']) > 0 else np.nan)} (M1-b)",
          f"- 코스피 TVW × 노출 대조 연환산 초과(원 정의 — 기술) {_pct(V['kospi_exc'])}", ""]
    mf = V["mfe"]["GA"]
    if mf:
        L += ["| MFE 구간 | n | 실현 R 평균 | 되돌림 R |", "|---|---|---|---|"]
        for k, v in mf["buckets"].items():
            L.append(f"| {k} | {v['n']} | {_f(v['real_R'], '{:+.2f}')} | {_f(v['giveback_R'], '{:.2f}')} |")
        L += ["", f"'+1R → 손실' 비율 {_f(mf['plus1R_to_loss'])} · 승자 포착률 중앙(기술 — 승자 조건부) {_f(mf['winner_capture_median'])}", ""]
    cr, sr_ = V["clx_rep"], V["sb_rep"]
    L += [f"- 클라이맥스(정정 4): 켠 폴드 거래 {cr.get('n', 0)} · 발동 {cr.get('fired', 0)} · 발동 거래 포착률 중앙 {_f(cr.get('capture_median', np.nan))} · "
          f"끔 대비 짝 차이 전체 {_pct(cr.get('diff_all', np.nan))} · 발동 거래 {_pct(cr.get('diff_fired', np.nan))}(낙관 경로, 배분 1 단위)",
          f"- 종가 기준 손절(정정 5): 종가 기준 폴드 거래 {sr_.get('n', 0)} · 살린 거래(장중이면 손절, 이후 +2R) {sr_.get('saved', 0)} · "
          f"키운 손실 {sr_.get('worse', 0)}건 평균 {_pct(sr_.get('worse_mean', np.nan))} · 장중 대비 짝 차이 {_pct(sr_.get('diff_all', np.nan))}"]
    fx, rr, dr = V["fx_rep"], V["re_rep"], V["dyn_rep"]
    L += [f"- 반등 실패 조기 청산(정정 6): 켠 폴드 원 거래 {fx.get('n', 0)} · 발동 {fx.get('fired', 0)} · 줄인 손실 {fx.get('cut_loss', 0)}건 "
          f"평균 {_pct(fx.get('cut_loss_gain', np.nan))} · 잘라 낸 승자 {fx.get('cut_win', 0)}건 평균 {_pct(fx.get('cut_win_cost', np.nan))} · "
          f"끔 대비 짝 차이 {_pct(fx.get('diff_all', np.nan))}",
          f"- 바닥 재진입(정정 6) 거래만: n {rr.get('n', 0)} · 순 2배 평균 {_pct(rr.get('net2_mean', np.nan))} · 승률 {_f(rr.get('win', np.nan))} · "
          f"청산 {rr.get('reasons', {})}",
          "- 불타기 종류별 짝(끔 대비, 정정 6): " + (" · ".join(f"{k} n {v['n']} {_pct(v['diff'])}" for k, v in V["pyr_kinds"].items()) or "—"),
          f"- 하단선 동적 손절(정정 7): 해당 거래 {dr.get('n', 0)} · 살린 거래 {dr.get('saved', 0)} · 키운 손실 {dr.get('worse', 0)}건 평균 "
          f"{_pct(dr.get('worse_mean', np.nan))} · 동적 손절 체결가/진입가 − 1 (10/50/90%) "
          + ("/".join(_pct(v) for v in dr.get("stop_fill_q", {}).values()) or "—")]
    for nm, pp in (("기준 청산 짝(라이브 프로토콜 − GA 청산, 같은 진입)", V["base_pair"]), ("불타기 짝(켬↔끔 바꿈 − 원래)", V["pyr_pair"])):
        if pp.get("n"):
            L.append(f"- {nm}: n {pp['n']} · 원래 {_pct(pp['a'])} → 바꿈 {_pct(pp['b'])} · 차이 {_pct(pp['diff'])} [{_pct(pp['lo'])}, {_pct(pp['hi'])}] "
                     "(낙관 경로 총손익, 배분 1 단위, 날짜군집 CI)")
    cs = V["cases"]
    L += ["", "## 사용자 판정 사례 재현(정정 8 — 보고만, 판정에 쓰지 않는다)", "",
          f"폴드별 출력 config 로 각 사례(± {CASE_TOL} 거래일 안 가장 가까운 사건)를 단독 시뮬(슬롯 무시, 낙관 경로, 2배 비용). "
          "칸 = 포착이면 'GA 청산 사유 R / 기준 청산 R · 포착률', 아니면 빠진 단계. 전체 열은 `pattern_ga/cases.csv`, 차트는 chart_points.json 의 cases.", "",
          "| 사례 | 판정 | 쐐기 D | " + " | ".join(f"폴드{k + 1}" for k in range(len(wins))) + " |", "|" + "---|" * (3 + len(wins))]
    for (nm, vd), g_ in (cs.groupby(["case", "verdict"], sort=False) if len(cs) else []):
        cells = []
        for _, r in g_.sort_values("fold").iterrows():
            if r["status"] == "포착":
                cells.append(f"{r['ga_reason']} {_f(r['ga_R'], '{:+.1f}')}R / {_f(r['live_R'], '{:+.1f}')}R · {_f(r['ga_capture'], '{:.0%}')}")
            else:
                cells.append(str(r["status"]).replace("|", "/"))
        L.append(f"| {nm} | {vd} | {g_['wd'].iloc[0] if 'wd' in g_ else '—'} | " + " | ".join(cells) + " |")
    L += ["", "## 민감도 (기술 — 판정에 쓰지 않는다)", "", "| 칸 | OOS 연환산 초과 | 2022~24 | 체결 |", "|---|---|---|---|"]
    for nm, s in V["sens"].items():
        L.append(f"| {nm} | {_pct(s['ann'])} | {_pct(s['calm_ann'])} | {s['n']} |")
    L.append(f"| 2021 제외 | {_pct(st['ex2021_ann'])} | | |")
    L.append("상위 k 거래 제외 곡선(k = 1·3·5)은 포트폴리오를 다시 돌린 값이다(단순 빼기 아님).")
    med = V["med_st"]
    L.append(f"| 중앙값 config(모든 시드 같은 모양 ∧ 실현 가능일 때만) | {_pct(med['ann']) if med else '정의 불가'} | "
             f"{_pct(med['calm_ann']) if med else ''} | {med['n'] if med else ''} |")
    adv = V["tr_df"]["adv20"] if len(V["tr_df"]) else pd.Series(dtype=float)
    al = V["tr_df"]["alloc"] if len(V["tr_df"]) else pd.Series(dtype=float)
    L += ["", "## 용량·체결 (M7)", "",
          f"- 개인 계좌 NAV 1억·3억에서 슬롯 금액 / ADV20: 중앙 {_f(float((al * 100 / adv).median()) if len(adv) else np.nan, '{:.2%}')} · "
          f"{_f(float((al * 300 / adv).median()) if len(adv) else np.nan, '{:.2%}')}, 90% {_f(float((al * 300 / adv).quantile(0.9)) if len(adv) else np.nan, '{:.2%}')}(3억).",
          f"- 분봉 p = {V['pmix']:.3f} ({E_meta(V, 'p_window')}, 판별 {E_meta(V, 'p_n')}) — '같은 정의로 다시' = VCP 재시험과 같은 합성 매수정지 사건"
          "(분봉 구간 2025-08+, ADV20 ≥ 10억 유니버스, 주문가 기준)으로 다시 잰 값이지 이 GA 사건 표에서 잰 값이 아니다(MINOR 9). p = 0·1 경계 칸을 함께 본다. "
          "10억~50억 분봉 커버리지는 `pattern_ga/FEASIBILITY.md`.",
          "- 호가: 2023-01-25 전 옛 표·뒤 통합 표(날짜별), 격자 정렬 없음(조정 주가), 손절·본전 체결 −1호가(MINOR 1). R 기준: stop_cap = (주문가 − 손절 원값)/주문가, "
          "2% 하한·real_R_post 분모 = (진입가 − 손절가)/진입가(호가·미끄러짐 뒤, 비용 전)(MINOR 3). 비용 = 매수 금액 기준(주) · 매도 금액 기준 칸(MINOR 8).", "",
          "## 구현 메모 (결과 전 결정)", "",
          "- 사건 표·위생·첫 관통·정규화 좌표 = `research/signals/pattern_events.py` 모듈 설명 1~12, 체결·청산·포트폴리오·대조 = "
          "`research/signals/pattern_fast.py` 모듈 설명. 유전자 부호화·벌점·창 = 이 러너 모듈 설명·상수.",
          "- 명목형(단계·기준선·추적·불타기)은 범주별 연속 로짓 → argmax(M2-1), 구간은 (중심, 반폭)(C3-2), 끔 = 끝 2% 구역. 조건부로 평평한 "
          "유전자는 키에서 `None`(같은 표현형 = 같은 평가).",
          f"- 원장 {V['n_led']}줄 · 이 계열 원장 합 {V['n_dsr_a']} · 실행 {V['elapsed'] / 60:.1f}분 (" + " · ".join(
              f"{k} {v:.0f}s" for k, v in V["timings"].items()) + ")",
          "- 민감도 지형(`pattern_ga/landscape.csv`)의 이웃 OOS 숫자는 기술이며 **판정·후속 선택에 쓰지 않는다**(MINOR 15).",
          "- OOS 트레이드 차트: `pattern_ga/chart_points.json` (seed 20261011 무작위 12건). **차트를 본 뒤의 정의 수정은 새 사전등록 · 이 데이터 "
          "재시험 금지**(MINOR 10). 이 계열 사슬: VCP → 쐐기 → 쐐기 D → 패턴 GA.", ""]
    return L


def _mean_or_nan(v) -> float:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(v.mean()) if len(v) else np.nan


def E_meta(V: dict, k: str):
    return V["E"].meta.get(k, "—")


if __name__ == "__main__":
    raise SystemExit(main())
