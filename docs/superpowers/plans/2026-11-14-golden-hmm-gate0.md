# MA50→MA200 골든 확률(HMM) — 관문 0 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2009-01 ~ 2016-08 사전 후보에서 HMM 경로 모형의 P_golden(k) 이 무작위 기준 q0 보다 MA50 > MA200 교차를 잘 맞히는지 한 번 판정한다.

**Architecture:** `research/signals/golden_hmm.py`(관측·HMM 학습·**순방향 필터만**·경로 모의·교차 확률) + `ma_quad.py` 에 일반 교차 확률 함수 추가 + 실행기 `research/experiments/golden_hmm_gate0.py`(ma_quad 실행기의 load·정지일·짝짓기·재실행 방지·원장 패턴 재사용, 구간 인자화).

**Tech Stack:** Python 3.12, numpy, pandas, numba, `hmmlearn`(research extra 에 추가), pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-golden-hmm-design.md`, `research/logs/golden_hmm/PREREG.md`(4e09516).

## Global Constraints
- 첫 실제 실행이 판정. 스모크·테스트는 지표(BSS·Brier·ECE·확률·실현율) 출력 금지. 깨끗한 트리에서만, 정본 `research/logs/golden_hmm/` 로만 한 번.
- 판정 구간 데이터: `daily_bars_adj_v2`, 2007-11-22 ~ 2016-08-31, `session="raw"`(오염 2026-09-14~ 와 무관), 폐지 포함(naver_adj), O/H/L ≤ 0 → NaN, `close_src="pre_listing"` NaN, **거래량 0·결측 날 종가 NaN**(ma_quad MQ-2). 실적 `eps_asof.eps_ok(..., source="earnings_dart_doc")`. 1,000원 규칙: raw_close 없으면 수정 종가.
- HMM 학습 데이터: 같은 테이블 2016-09-01 ~ 2023-12-31(라벨 미사용; 수익·이평선만). 2024~ 사용 금지.
- 사건: 유니버스(`universe_mask(adv_floor=1000)`, 상장 252일) ∧ EPS↑ ∧ MA50 < MA200 ∧ 1 − MA50/MA200 ≤ 0.05 ∧ 2009-01-02 ≤ t ∧ t+k ≤ 2016-08-31(거래일). k ∈ {5, 10, 15}.
- 라벨 y = h ∈ 1..k 중 한 번이라도 MA50 > MA200(엄격). MA 창에 NaN 이면 사건·라벨 없음.
- HMM: 관측 x_t = [z_t = r_t/σ_t, g_t = log(MA50/MA200), Δ5 g_t], σ_t = t 까지 60 로그수익 표준편차. 가우시안 full 공분산, S ∈ {1..5} 중 학습 데이터 BIC 최소, `random_state=0`, n_iter ≤ 200. 학습 수열 = 유니버스 연속 구간(길이 ≥ 60), 표본이 크면 결정적 종목 표본으로 줄여도 됨(기록).
- **순방향 필터만**(t 까지 관측으로 상태 사후확률). `predict_proba`(평활, 미래 사용) 금지.
- 경로: M = 200, 시드 = blake2b(code|date|k|"hmm"). 상태 s_t ~ 필터 사후, s_{t+h} ~ 전이행렬, z_{t+h} ~ N(μ_s[z], Σ_s[z,z]), 가격 = C_t·exp(Σ z·σ_t). MA50·MA200 = 알려진 과거 종가 + 경로 종가.
- q0: ma_quad 절차 그대로(q=0, N 무관, 경험 라이브러리 3년·월 갱신·원점 ≥ 120, member min_history 70, M=200, draw_index 키 N=0) — 교차 확률만 새 함수로.
- 판정: `ma_quad_calib.cell_verdict(pA=hmm, pB=q0, y, month, year, years=tuple(range(2009, 2017)))` 단 연도 바 ≥ 6/8 → `years_ok` 를 이 실행기에서 `years_pos ≥ 6` 으로 계산(함수 기본 5 는 7년용). 관문 = k=10 합격 ∧ 3칸 중 ≥ 2 합격.
- 보고(판정 무관): S=1 가우시안 대 q0, 격차 구간(<1%·1~3%·3~5%), 연도별 BSS, M=1000 민감도(1% 표본), 선택된 S·BIC 표·상태별 μ/Σ, 짝짓기 탈락 사유.
- 출력 strict JSON(NaN→null) · RESULTS.md(검증 전) · TRIALS.jsonl 파이썬 덧붙이기(마지막에) · 재실행 방지. 캐시 `~/.cache/cc-swing/scratch/golden_hmm/`. `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`. 경로 지정 git add, 한국어 커밋 + `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus
1. 평활 사후확률 누출 → `test_filter_ignores_future`(t 뒤 관측을 바꿔도 t 의 필터 사후·확률 비트 동일).
2. 경로 이평선 산수 off-by-one → `test_cross_probs_zero_sigma_match_realized`(σ=0·실제 미래 → 확률 = 실현).
3. 라벨 끝 → `test_label_end_2016_08_31`.
4. S=1 이면 HMM 확률이 단일 가우시안 랜덤워크와 같은 분포 → `test_s1_equals_gaussian_rw`(통계 허용오차).
5. 학습 구간과 판정 구간 겹침 없음 → `test_train_window_disjoint`.

---

### Task 1: 교차 확률 일반 함수 (ma_quad)
**Files:** Modify `research/signals/ma_quad.py`, Test `tests/test_ma_quad.py`
**Produces:** `cross_probs(C, fc_log, sigma, pool, idx, ci, s, k, short=50, long=200) -> np.ndarray`(사건별 P(∃h≤k: MA_short > MA_long)), `cross_realized(C, ci, s, k, short=50, long=200) -> np.ndarray`(0/1, NaN 규칙 같음). 기존 함수 불변.
- [ ] 테스트(σ=0 일치, k≥long 경계, NaN) → 구현(numba) → 커밋 `research(golden_hmm): 이평선 교차 확률 함수`

### Task 2: HMM 모듈
**Files:** Create `research/signals/golden_hmm.py`, Test `tests/test_golden_hmm.py`; Modify `pyproject.toml`(research/dev extra 에 hmmlearn)
**Produces:** `observations(C) -> (z, g, dg)`, `fit_hmm(X_list, S_range=(1,5), seed=0) -> dict(S, bic_table, model)`, `forward_filter(model, X) -> (T, S) 필터 사후`, `hmm_cross_probs(model, C, sigma, post, ci, s, k, short=50, long=200, M=200) -> np.ndarray`.
- [ ] 테스트: Review Focus 1·2·4 + `test_bic_picks_true_S`(합성 2국면 자료에서 S=2) → 구현 → 커밋 `research(golden_hmm): 국면 전환 HMM·순방향 필터·경로 교차 확률`

### Task 3: 실행기
**Files:** Create `research/experiments/golden_hmm_gate0.py`, Test `tests/test_golden_hmm_runner.py`. `maquad_calib.load` 를 구간 인자로 감싸거나 같은 규약의 새 load(정지일 NaN·pre_listing·폐지 대체 가격 포함).
- [ ] 테스트: Review Focus 3·5, 짝짓기(hmm·q0 둘 다 유한한 사건만), strict JSON, TRIALS 한 줄, 재실행·더러운 트리 거부, 스모크 무지표.
- [ ] 전체 pytest·ruff·guardrails → 스모크(모양·건수·시간·HMM 학습 시간) → 커밋 `research(golden_hmm): 관문 0 실행기(결과 전)`

### Task 4: 실행·검증 (컨트롤러)
- [ ] 실제 실행 → `(검증 전)` 커밋 → 차트 게시 → quant-reviewer + haiku-verifier 12(사건·라벨 원 DB 재계산) → 정정 → 판정. 통과 시 관문 1 계획.
