# 이평선 외삽 — 관문 1 백테스트(시도 1: X-dead) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 관문 0 통과 시, 사전등록 §6 + 정정 MQ-1·2·3 의 시도 1(X-dead)을 워크포워드 2020~23·비용 2배·널 ①② 로 판정한다.

**Architecture:** (1) `ma_quad.event_probs` 에 P_dead 키 추가 (2) 실행기 `maquad_panels.py` 가 (q, N, k) 마다 종목-일 확률 패널을 한 번 계산해 캐시 (3) 순수 엔진 `research/signals/maquad_engine.py` 가 설정 하나를 시뮬 (4) `research/signals/maquad_wf.py` 가 메뉴·워크포워드·널·바 (5) 실행기 `research/experiments/maquad_backtest.py`.

**Tech Stack:** Python 3.12, numpy, pandas, numba, pytest, multiprocessing(fork) 16 작업자.

**Spec:** `docs/superpowers/specs/2026-11-14-ma-quad-forecast-design.md` §6, `research/logs/ma_quad/PREREG.md`(관문 1 줄 + MQ-1·MQ-2·MQ-3).

## Global Constraints

- **관문 0 이 합격(`calib_results.json` 의 gate pass)일 때만 실제 실행.** 실행기는 시작 때 그 파일을 읽고 불합격이면 중단한다.
- 첫 실제 실행이 판정이다. 스모크·테스트는 수익·R·CAGR·바를 출력하지 않는다(모양·건수·시간만).
- 데이터·마스크·정지일 처리는 관문 0 실행기(`research/experiments/maquad_calib.py` 의 `load`·`mask_suspensions`·`prepare`)를 그대로 import 해서 쓴다. 고치지 않는다(관문 0 재현성).
- 엔진 구간 2017-10-02 ~ 2023-12-28(평가 끝 = 2023-12-28 종가). 폴드 Y ∈ {2020, 2021, 2022, 2023}.
- 체결: t 종가 판정 → t+1 시가. 상한가 시가(시가 ≥ 전일 종가 × 상한 배수 − 1호가; leader21 `lu` 규약) 매수 불가. 정지일(MQ-2 NaN) 매수·매도 불가 → 매도는 다음 거래 가능일 시가.
- 진입 후보: 사전 후보(유니버스 ∧ EPS↑ ∧ ¬정배열 ∧ 0 < d ≤ 0.10 ∧ f200) ∧ P_align ≥ θ_in ∧ 겹침(일봉 3쌍 P_up ≥ 0.5 수) ≥ o_min. 순위 = P_align(q) − P_align(q=0) 내림차순.
- 진입 재확인: d_fill = 1 − 손절선 / t+1 시가; d_fill ≤ 0 또는 > 0.10 → 사지 않음. 손절선 = 진입 판정일 묶음 하단, 고정.
- 비중 = min(0.05, 0.0025 / d_fill), 슬롯 20, 종목당 1 포지션, 불타기 없음, 현금 부족 시 남은 현금까지. 재진입: 청산 체결 다음 날부터.
- 청산(시도 1): ① 종가 < 손절선 → 다음 날 시가 ② 정배열(종가 기준) 실현 뒤: P_dead(쌍, k) ≥ θ_dead → 다음 날 시가 ③ 정배열 실현 전: P_align < θ_out → 다음 날 시가. 같은 날 여러 사유면 ① > ③ > ② 순으로 사유 기록. 최대 보유 없음.
- 비용: `korea.cost_rt(청산일, 시장)` × {1, 2}, 판정 2. 거래 수익 = 청산가/진입가 − 1 − 비용.
- 투입 R = 비중 × 수익 / 0.0025(자산 0.25% 단위).
- 메뉴: q {1, 2} × N {10,15,20,30,40} × k {3,5,7,10} × θ_in {0.5,0.6,0.7,0.8} × θ_out {0.2,0.3,0.4}(θ_out < θ_in) × o_min {0,1,2} × 쌍 {5/20, 20/60} × θ_dead {0.5,0.6,0.7} = 8,640.
- 선택: 학습 창(2017-10-02 ~ Y−1 말, 끝난 거래 + 보유분 Y−1 마지막 종가 평가)의 비용 2배 투입 R 합계의 이웃 중앙값(축마다 한 칸 차이 + 자기) 최대. 동률이면 메뉴 순서 앞.
- 널 ① = 폴드마다 무작위 설정(선택 없음) 300 조합, 시드 고정. 널 ② = MQ-3 정의 300 회.
- 바(비용 2배, 폴드 밖 이어 붙인 거래): ① R 합계 > 널 ①·② 95 백분위 ② 거래 R 평균 날짜 군집(진입 연월) 부트스트랩 95% 하한 > 0 ③ 상위 5 종목(종목별 R 합 기준) 제거 후 R 평균 > 0 ④ 월 수익 팩터 알파 t > 2(OLS·NW 둘 다, `autoresearch.prepare.factor_months`·`factor_t`) ⑤ 연복리 > 노출 맞춘 시장(전날 노출 × 시총 가중 시장 일수익) ∧ 보고 열 KOSPI 보유 ⑥ 폴드 ≥ 3/4 R 합 > 0. 체결 < 100 → VOID.
- 캐시 `~/.cache/cc-swing/scratch/ma_quad/` · `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1` · pidfile · 원장은 파이썬 덧붙이기 · 경로 지정 `git add` · 한국어 커밋 + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **보유 종목의 확률 결측** — 보유 중 그날 P 가 NaN(정지·창 결측)이면 ③·② 판정을 하지 않고 넘긴다(청산 사유 아님), ① 은 종가가 있으면 그대로 → Task 3 `test_nan_prob_does_not_exit`.
2. **같은 날 손절과 신규 진입 경쟁** — 청산 체결로 빈 슬롯은 그날 진입에 쓰지 않는다(청산·진입 모두 t+1 시가라 순서: 청산 먼저, 빈 슬롯은 다음 판정일부터) → `test_freed_slot_next_day`.
3. **갭 하락으로 손절가 아래 체결** — 시가가 손절선보다 한참 아래여도 시가 체결(손실 > 1R 허용) → `test_gap_below_stop_fills_at_open`.
4. **폴드 경계 누출** — 학습 창 성적이 Y 가격을 쓰지 않는다 → Task 4 `test_train_window_truncation`.
5. **평가 끝 보유분** — 2023-12-28 종가로 평가, 비용 포함 → Task 3 `test_data_end_mark`.

---

### Task 1: P_dead 키

**Files:** Modify `research/signals/ma_quad.py`, Test `tests/test_ma_quad.py`
**Interfaces:** Produces `event_probs(...)` 반환에 `p_dead_5_20`·`p_dead_20_60` = h = 1..k 중 한 번이라도 단기 MA < 장기 MA 인 경로 비율. `realized` 에도 같은 키(실제 경로).
- [ ] 테스트: `test_p_dead_zero_sigma_matches_realized`(σ=0·fc=실제 미래 → 같음), `test_p_dead_monotone_in_k`(같은 사건 k 늘리면 p_dead 비감소). 기존 테스트 전부 유지.
- [ ] 구현(같은 커널 안에서 h 마다 비교 누적) → 테스트 → 커밋 `research(ma_quad): P_dead(데드 예측) 키`

### Task 2: 확률 패널

**Files:** Create `research/experiments/maquad_panels.py`, Test `tests/test_maquad_panels.py`
**Interfaces:** Consumes `maquad_calib.load/prepare/cell_probs` 계열. Produces `build_panel(X, q, N, k) -> dict[str, np.ndarray]`(키 p_align·p_up_*·p_dead_*, (n_codes, n_days) float16, 정의 안 되는 칸 NaN; 대상 칸 = 2017-10-02 ~ 2023-12-28 중 위생 ∧ ADV ≥ 10억 이거나 아무 사전 후보였던 종목의 그 뒤 날짜 — 단순하게: 그 구간 C 유한 전 칸), `panel_path(q,N,k)`, CLI `--smoke`(50 종목).
- [ ] 테스트: 합성 패널에서 `build_panel` 값이 같은 (ci, s) 의 `cell_probs` 와 같음, 결측 칸 NaN.
- [ ] 구현(관문 0 의 월 루프 재사용, q=0 은 k 마다) → 스모크(시간·메모리 측정, 지표 없음) → 커밋 `research(ma_quad): 종목-일 확률 패널`

### Task 3: 엔진

**Files:** Create `research/signals/maquad_engine.py`, Test `tests/test_maquad_engine.py`
**Interfaces:** Produces `simulate(D: dict, cfg: dict, cost_mult: float, start: int, end: int, entry_end: int | None = None) -> tuple[pd.DataFrame, pd.Series]` — D 키: days, codes, O, C, lu, market, cand(사전 후보 bool), stop_line(묶음 하단), aligned, P(dict[(q,N,k)] → 패널 dict), P0(dict[k] → p_align). cfg 키 q,N,k,th_in,th_out,o_min,pair,th_dead. trades 열: code, entry_day, exit_day, entry_px, exit_px, w, d_fill, ret, R, reason(①·②·③·end), aligned_at. 반환 Series = 일별 자산(시작 1.0). `entry_end` 이후에는 진입 없음(폴드).
- [ ] 테스트(합성 2~5 종목, 손으로 계산한 기대값): `test_entry_next_open`, `test_skip_when_open_below_stop`, `test_skip_when_dfill_over_10pct`, `test_limit_up_open_no_buy`, `test_stop_exit_next_open`, `test_gap_below_stop_fills_at_open`, `test_collapse_exit_before_alignment_only`, `test_dead_exit_after_alignment_only`, `test_nan_prob_does_not_exit`, `test_suspension_delays_sell`, `test_weight_formula`, `test_slots_20_rank_by_edge`, `test_freed_slot_next_day`, `test_reentry_next_day`, `test_data_end_mark`, `test_cost_mult`.
- [ ] 구현(일 루프 파이썬 + numpy, 설정당 ≤ 1초 목표; 넘으면 numba) → 테스트 → 커밋 `research(ma_quad): 백테스트 엔진(시도 1 X-dead)`

### Task 4: 워크포워드·널·바

**Files:** Create `research/signals/maquad_wf.py`, Test `tests/test_maquad_wf.py`
**Interfaces:** Produces `menu() -> list[dict]`(8,640, 순서 고정), `neighbors(i) -> list[int]`, `train_score(trades, equity, train_end_day) -> float`(끝난 거래 + 보유분 평가, Y 가격 미사용), `select(scores) -> int`(이웃 중앙값 최대), `null1(...)`, `null2(...)`, `bars(oof_trades, oof_monthly, nulls, F) -> dict`(바 1~6 + VOID + 보고 열), `dsr(...)`.
- [ ] 테스트: `test_menu_size_8640`, `test_neighbors_axis_steps`, `test_train_window_truncation`(학습 끝 뒤 가격을 바꿔도 점수 같음), `test_select_plateau_over_spike`(고립 최대보다 고원 선택), `test_null2_matches_daily_counts`, `test_bars_void_under_100`, `test_top5_removal`.
- [ ] 구현 → 테스트 → 커밋 `research(ma_quad): 워크포워드·널·바(시도 1)`

### Task 5: 실행기

**Files:** Create `research/experiments/maquad_backtest.py`, Test `tests/test_maquad_backtest_runner.py`
**Interfaces:** CLI(`--smoke`, `--workers 16`), 관문 0 합격 확인, 패널 빌드/로드, 학습 창 점수 = 8,640 × 4 폴드(작업자 분배), 폴드 밖 시뮬, 널, 바. 출력 `research/logs/ma_quad/backtest_g1.json`(NaN→null)·`RESULTS_G1.md`(검증 전)·`TRIALS.jsonl` 1행·거래 CSV(`~/.cache/.../g1_trades.csv`, 차트·스웜용)·재실행 방지.
- [ ] 테스트: 관문 0 불합격이면 중단, 스모크가 지표를 출력하지 않음, JSON strict, TRIALS 한 줄 덧붙임.
- [ ] 전체 pytest·ruff·guardrails → 스모크(시간 추정) → 커밋 `research(ma_quad): 관문 1 실행기(결과 전)`

### Task 6: 실행·검증
- [ ] 실제 실행(pidfile) → `(검증 전)` 커밋 → swing-visualizer(자산 곡선·연도별·거래 R 분포·예시 거래 차트: 봉·거래량·MA·진입/청산·손절선·외삽 밴드) 게시
- [ ] quant-reviewer 1 + haiku-verifier 15(거래 6~8개씩 원 DB 재계산 — 규약: t 종가 판정·t+1 시가, 손절선 고정, d_fill 재확인, 상한가·정지일, 비용 2배, 평가 끝 2023-12-28) → 정정 → 같은 URL 갱신 → 판정 보고. 통과 시 관문 2(2009~2016 1회) 계획.
