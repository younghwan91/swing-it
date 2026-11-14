# 이평선 q차식 외삽 — 관문 0(확률 보정) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 종목-일마다 q차식(q=0·1·2) 외삽 + 경험 오차 라이브러리로 이평선 경로를 만들고, 2017~2023 사전 후보에서 P_align(k) 의 보정·기술(BSS 대 B0)을 재서 관문 0 을 판정한다.

**Architecture:** `research/signals/ma_quad.py` 에 순수 배열 함수(적합·라이브러리·경로 확률, numba)와 `research/signals/ma_quad_calib.py` 에 지표·판정 함수(순수 numpy)를 둔다.
실행기 `research/experiments/maquad_calib.py` 가 DB 를 읽어 사건을 만들고 (q, N, k) 칸을 작업자 16개로 나눠 돌린 뒤 JSON·RESULTS.md·TRIALS.jsonl 을 쓴다. 관문 1·2·봉 모듈은 관문 0 통과 뒤 별도 계획.

**Tech Stack:** Python 3.12, numpy, pandas, numba(`uv run --with numba`), pytest. DB = kr_quant(읽기 전용, psycopg2 `--extra pg`).

**Spec:** `docs/superpowers/specs/2026-11-14-ma-quad-forecast-design.md`(§4·§5·§14), 사전등록 `research/logs/ma_quad/PREREG.md`(커밋 7d8654e).

## Global Constraints

- 결과를 보기 전에 판정 코드를 끝내고 커밋한다. 스모크는 **모양·시간만** 출력한다(지표 출력 금지). 첫 실제 실행이 곧 판정이다.
- 가격: `storage.read_prices_wide(con, table="daily_bars_adj_v2", start="2013-06-01", end="2024-01-31", session="regular", unpatched="nan")`. `close_src == "pre_listing"` 제외, O/H/L ≤ 0 → NaN(indep30 `load` 규약).
- 사건 날짜 t: 2017-01-02 ≤ t 이고 t+k 거래일이 **2023-12-28 이하**(라벨이 2024 가격을 쓰지 않게 — 내 판단, 사양 §2 "2024~ 소모").
- 유니버스: `korea.universe_mask(close, trade_value, meta, raw_close=raw_close, adv_floor=1000)` (ADV20 ≥ 10억 = 1,000 백만원, min_history 252 기본). 호출 방식은 `research/experiments/convergence_run.py:337` 참고.
- EPS: `eps_asof.eps_ok(codes, days, ci, s, con, source="earnings")`.
- 라이브러리 모집단: 위생(보통주·원 종가 ≥ 1,000원) ∧ ADV20 ≥ 10억 ∧ 이력 ≥ N+60 — 상장 252일 조건은 걸지 않는다(`universe_mask(..., min_history=N+60)` 로 구현).
- 라이브러리 원점: t 가 속한 달의 첫 거래일 t_m 기준 u ∈ [t_m − 3 달력년, t_m − k 거래일], 원점 날짜 ≥ 120 거래일일 때만 신호 정의. (q, N, k) 마다 따로.
- σ_u = u 까지(포함) 60개 로그 수익의 표본 표준편차(ddof=1), 결측 있으면 NaN.
- 경로 M = 200, 시드 = `keyed`(code, 연월, q, N, k) 결정적; 같은 달 안 같은 종목은 같은 색인(공통 난수).
- 정배열 = MA5 > MA20 > MA60 > MA120(엄격 부등호). 묶음 하단 = min(MA20, MA60, MA120). d = 1 − 묶음 하단 / 종가.
- 칸 합격·관문 합격 문턱은 PREREG.md 숫자 그대로(아래 Task 4 테스트에 박는다).
- 작업자: `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`, 16 프로세스. 캐시는 `~/.cache/cc-swing/scratch/ma_quad/`(/tmp 금지, 만들기 전 `du -sh ~/.cache/cc-swing/scratch`).
- 커밋 메시지 한국어 conventional, 경로 지정 `git add`(`-A` 금지). push 없음.

## Review Focus

1. **라이브러리 미래 누출** — 원점 u 의 오차는 u+k 가격을 쓴다. u ≤ t_m − k 가 깨지면 보정이 공짜로 좋아진다 → Task 2 테스트 `test_pool_no_future`.
2. **결측 창** — 정지일(NaN 종가)이 적합 창·MA 창에 끼면 그 종목-일은 신호 없음(NaN), 0 으로 채우지 않는다 → Task 1 `test_nan_window_gives_nan`, Task 3 `test_nan_history_skips_event`.
3. **경계 동률** — P = 0.5 정확히, MA 동률(=)은 정배열 아님 → Task 3 `test_tie_is_not_aligned`, Task 4 `test_bin_edges_left_closed`.
4. **빈 구간** — 신뢰도 표에서 n < 300 구간은 판정에서 빠지고, P ≥ 0.5 구간이 전부 비면 보정 조건 실패(통과 아님) → Task 4 `test_calibration_fails_when_no_high_bins`.
5. **연도 BSS 의 빈 해** — 한 해의 사건이 0 이면 그 해는 "양수 아님"으로 센다 → Task 4 `test_empty_year_counts_as_not_positive`.

---

### Task 1: q차식 롤링 외삽

**Files:**
- Create: `research/signals/ma_quad.py`
- Test: `tests/test_ma_quad.py`

**Interfaces:**
- Produces: `poly_forecast(y: np.ndarray, N: int, k: int, q: int) -> np.ndarray` — `y` (n_codes, n_days) 로그 종가 float64, 반환 (n_codes, n_days, k) float32, `[i, t, h-1]` = t 까지 N개로 맞춘 q차식의 t+h 값. 창에 NaN 이 하나라도 있거나 t < N−1 이면 NaN. q=0 은 y[i, t] 반복(N 무관).
- Produces: `rolling_sigma(y: np.ndarray, w: int = 60) -> np.ndarray` (n_codes, n_days) float64.

- [ ] **Step 1: 실패하는 테스트**
```python
def test_quadratic_exact():      # y = 1 + 0.1 s + 0.01 s^2 를 정확히 외삽(오차 < 1e-5)
def test_line_exact_q1():        # 직선은 q=1·q=2 둘 다 정확
def test_q0_is_last_value():     # q=0 → 모든 h 에서 y[t]
def test_nan_window_gives_nan(): # 창 안 NaN 하나 → 그 t 부터 N−1 일 NaN
def test_matches_polyfit():      # 무작위 계열에서 np.polyfit(deg=q) 외삽과 1e-6 안
def test_rolling_sigma_ddof1():  # pandas diff().rolling(60).std() 와 같음
```
- [ ] **Step 2:** `uv run --with numba --extra dev pytest tests/test_ma_quad.py -q` → FAIL(ImportError)
- [ ] **Step 3: 구현** — s = −N+1..0 기준 정규방정식 계수는 N·q 로만 정해지므로 외삽 가중치 행렬 W(k×N)를 한 번 계산(`np.linalg.lstsq` 로 설계행렬 의사역 → 외삽 행 곱)하고, 종목별로 슬라이딩 창 점곱(numba). 
- [ ] **Step 4:** 같은 명령 → PASS
- [ ] **Step 5:** `git add research/signals/ma_quad.py tests/test_ma_quad.py && git commit -m "research(ma_quad): q차식 롤링 외삽·σ"`

### Task 2: 오차 라이브러리와 표본 추출

**Files:**
- Modify: `research/signals/ma_quad.py`
- Test: `tests/test_ma_quad.py`

**Interfaces:**
- Consumes: `poly_forecast`, `rolling_sigma`.
- Produces: `forecast_z(y, fc, sigma) -> np.ndarray` (n_codes, n_days, k) float32 = (y[t+h] − fc[t,h−1]) / sigma[t], 미래가 없거나 NaN 이면 NaN.
- Produces: `month_pools(days: pd.DatetimeIndex, z: np.ndarray, member: np.ndarray, k: int, years: int = 3, min_dates: int = 120) -> dict[int, np.ndarray]` — 키 = 달 첫 거래일의 행 번호 t_m, 값 = (P, k) float32 풀(member ∧ z 전부 유한인 원점, u ∈ [t_m − 3년, t_m − k]). 원점 날짜 < 120 이면 그 달 키 없음.
- Produces: `draw_index(code: str, t_m_date: pd.Timestamp, q: int, N: int, k: int, P: int, M: int = 200) -> np.ndarray` (M,) int64 — 결정적(`hashlib.blake2b` 키 → `np.random.default_rng`).

- [ ] **Step 1: 실패하는 테스트**
```python
def test_pool_no_future():          # 풀에 들어간 모든 원점 u 가 u + k <= t_m 을 만족(원점 행 번호를 같이 돌려받는 디버그 인자로 확인)
def test_pool_window_3y():          # t_m − 3년 이전 원점 없음
def test_pool_min_dates():          # 원점 날짜 119 → 키 없음, 120 → 있음
def test_draw_index_deterministic(): # 같은 인자 같은 결과, 연월 바꾸면 다름
def test_z_nan_tail():              # 마지막 k 일 z 는 NaN
```
- [ ] **Step 2:** pytest → FAIL
- [ ] **Step 3: 구현** (`month_pools(..., return_origins=False)` 디버그 인자 포함)
- [ ] **Step 4:** pytest → PASS
- [ ] **Step 5:** commit `research(ma_quad): 오차 라이브러리(월 갱신·3년·미래 누출 차단)`

### Task 3: 경로 → 이평선 → 확률

**Files:**
- Modify: `research/signals/ma_quad.py`
- Test: `tests/test_ma_quad.py`

**Interfaces:**
- Consumes: Task 1·2.
- Produces: `ma_state(C: np.ndarray) -> dict` = {"ma": {5,20,60,120: (n_codes,n_days)}, "aligned": bool, "bundle_low": float, "d": float}(MA 창에 NaN 있으면 NaN·aligned False).
- Produces: `event_probs(C: np.ndarray, fc_log: np.ndarray, sigma: np.ndarray, pool: np.ndarray, idx: np.ndarray, ci: np.ndarray, s: np.ndarray, k: int) -> dict[str, np.ndarray]`
  — 사건(ci[j], s[j])마다 경로 y(t+h) = fc_log[ci,s,h−1] + sigma[ci,s]·pool[idx[j]][h−1], 가격 = exp. MA_w(t+h) = (알려진 C[t+h−w+1 .. t] 합 + 경로 가격 합)/w.
  반환 키: `p_align`(t+k 정배열 비율), `p_up_5_20`·`p_up_20_60`·`p_up_60_120`(t+k 에 단기 > 장기). idx 는 (n_events, M).
- Produces: `realized(C, ci, s, k) -> dict` 같은 키의 0/1 실현값(t+k 의 실제 MA 상태).

- [ ] **Step 1: 실패하는 테스트**
```python
def test_zero_sigma_matches_realized():  # sigma=0, fc=실제 미래 로그가 → p_align == realized (전부 0/1)
def test_known_part_arithmetic():        # MA120(t+5) 계산이 pandas rolling(120) 에 미래 5개 붙인 값과 1e-9 안
def test_tie_is_not_aligned():           # MA5 == MA20 → aligned False
def test_nan_history_skips_event():      # 창 안 NaN → p_align NaN
def test_p_in_unit_interval():           # 무작위 입력에서 0 ≤ p ≤ 1, M=200 이면 p*200 정수
```
- [ ] **Step 2:** pytest → FAIL
- [ ] **Step 3: 구현** — numba 커널(사건 × 경로 × h, MA 는 알려진 부분 합을 사건마다 한 번 계산 후 경로 가격 누적).
- [ ] **Step 4:** pytest → PASS
- [ ] **Step 5:** commit `research(ma_quad): 경로 이평선·정배열 확률`

### Task 4: 보정·기술 지표와 관문 판정

**Files:**
- Create: `research/signals/ma_quad_calib.py`
- Test: `tests/test_ma_quad_calib.py`

**Interfaces:**
- Produces: `reliability(p, y, edges=np.linspace(0, 1, 11)) -> pd.DataFrame`(열 lo, hi, n, p_mean, y_rate; 구간 [lo, hi), 마지막은 [0.9, 1.0]).
- Produces: `ece(p, y) -> float`(10구간 n 가중 |p_mean − y_rate|), `brier(p, y) -> float`.
- Produces: `bss_ci(pA, pB, y, cluster: np.ndarray, n_boot=2000, seed=0) -> tuple[float, float, float]`(점, 2.5%, 97.5%; 군집 = 연월 정수, 군집 복원 추출).
- Produces: `cell_verdict(pA, pB, y, month: np.ndarray, year: np.ndarray) -> dict` — 키 `calib_ok, skill_ok, years_ok, pass, ece, base_rate, bss, bss_lo, bss_hi, years_pos, high_bins`.
  규칙(PREREG): 보정 = P_A ≥ 0.5 구간 edges [0.5,0.6,0.7,0.8,0.9,1.0] 중 n ≥ 300 인 구간마다 |p_mean − y_rate| ≤ 0.05 이고 그런 구간 ≥ 1, ECE ≤ 0.02 ∧ ECE ≤ 0.2 × 기준율;
  기술 = bss ≥ 0.01 ∧ bss_lo > 0; 연도 = 2017~2023 중 BSS > 0 인 해 ≥ 5(사건 0 인 해는 양수 아님).
- Produces: 보고용 `bss_ci(..., cluster=분기 정수)` 민감도를 `cell_verdict` 반환에 `bss_lo_q` 로 같이 넣는다(판정 무관).
- Produces: `gate_verdict(passed: dict[tuple[int, int], bool]) -> dict` — 키 (N, k), 이웃 = N 목록 또는 k 목록에서 한 칸 차이(4-이웃). `pass` = 합격 ≥ 12 ∧ 최대 연결 덩어리 ≥ 8; `n_pass`, `largest_block` 도 반환.

- [ ] **Step 1: 실패하는 테스트**
```python
def test_bin_edges_left_closed():                 # p=0.5 는 [0.5,0.6) 구간, p=1.0 은 마지막 구간
def test_perfect_calibration_passes():            # y ~ Bernoulli(p) 대표본 → calib_ok True
def test_calibration_fails_when_no_high_bins():   # p 전부 < 0.5 → calib_ok False
def test_bss_threshold_001():                     # bss 0.009 → skill_ok False
def test_empty_year_counts_as_not_positive():     # 2019 사건 0 → years_pos 에 안 셈
def test_gate_needs_block_of_8():                 # 합격 12칸이지만 덩어리 최대 7 → False; 덩어리 8 → True
def test_bss_ci_cluster_resampling():             # 같은 군집 복제 시 CI 폭이 군집 수 기준으로 넓어짐
```
- [ ] **Step 2:** `uv run --extra dev pytest tests/test_ma_quad_calib.py -q` → FAIL
- [ ] **Step 3: 구현**
- [ ] **Step 4:** → PASS
- [ ] **Step 5:** commit `research(ma_quad): 보정·BSS·관문 판정 함수(사전등록 문턱)`

### Task 5: 관문 0 실행기

**Files:**
- Create: `research/experiments/maquad_calib.py`
- Test: `tests/test_maquad_calib_runner.py`(sqlite 픽스처 또는 합성 패널로 `build_events`·`run_cell` 경로만)

**Interfaces:**
- Consumes: Task 1~4, `storage.read_prices_wide`, `korea.universe_mask`, `eps_asof.eps_ok`.
- Produces: `load(con) -> dict`(days, codes, C, raw_close, tv, meta), `build_events(X, k) -> (ci, s)`(사전 후보: 유니버스 ∧ EPS↑ ∧ ¬aligned ∧ 0 < d ≤ 0.10 ∧ **MQ-1: 종가 > MA200 ∧ MA200[t] > MA200[t−20]** ∧ 2017-01-02 ≤ t ∧ t+k ≤ 2023-12-28 거래일),
  `build_events_all(X, k)`(¬aligned 전부, 보고용), `run_cell(X, q, N, k) -> dict`(p, y, 연월, 연도 배열 저장 + `cell_verdict`), CLI `--smoke`(50 종목·2017 만, **모양·시간만 출력**), `--workers 16`, `--m-sens`(칸 q=2·N=20·k=5, 사건의 결정적 1% 표본에서 M=1,000 대 200 의 |Δp| 분포 — 보고만).
- 보고(판정 무관, MQ-1): 템플릿 충족 부분집합(종가 > MA150 ∧ MA150 > MA200 ∧ MA50 > MA150 ∧ 종가 > MA50 ∧ 종가 ≥ 250일 최저×1.30 ∧ 종가 ≥ 250일 최고×0.75)의 같은 표, 200일선 거름 없는 원래 사전 후보의 같은 표.
- 보고(판정 무관): 정배열 아닌 전체 종목-일(`build_events_all`)의 같은 표, 해상도 = P ≥ 0.5 구간 실현율 A·B0·B1 비교.
- 출력: `research/logs/ma_quad/calib_results.json`(칸별 verdict + reliability 표 + q=1 대 B1 BSS 보고 + 사건 수 + git SHA + 데이터 구간), `research/logs/ma_quad/RESULTS.md`(표: 20칸 q=2 판정, 관문 판정, q=1 보고, 쌍별 P_up 보고), `TRIALS.jsonl` 에 파이썬으로 1행 덧붙이기(`{"trial":"gate0", "config":{…}, "verdict":…, "git":…}`).
- 주봉 표는 이번 실행기에서 뺀다(보고 전용, 관문 1 계획에서 겹침과 함께 — 판정 무관).

- [ ] **Step 1: 실패하는 테스트** — `test_events_respect_label_end`(t+k > 2023-12-28 사건 없음), `test_events_are_preccandidates`(합성 패널에서 aligned 인 날 제외·d 범위), `test_run_cell_writes_arrays`(합성 패널 2 종목에서 키 존재).
- [ ] **Step 2:** pytest → FAIL
- [ ] **Step 3: 구현** — 칸은 (q, N, k) 로 작업자에 분배, q=0 은 k 마다 한 번(N 무관)만 계산해 공유. p 배열은 `~/.cache/cc-swing/scratch/ma_quad/` 에 npz.
- [ ] **Step 4:** pytest → PASS, 그리고 `set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --with numba --extra pg python research/experiments/maquad_calib.py --smoke` → 모양·칸당 시간만 출력, 전체 예상 시간 기록.
- [ ] **Step 5:** 린트 `uv run --extra dev ruff check src/ tests/ research/ scripts/ examples/` · `uv run python scripts/check_guardrails.py` 통과, 전체 `uv run --with numba --extra dev pytest -q` 통과 → commit `research(ma_quad): 관문 0 실행기(결과 전)`

### Task 6: 실행 · 검증 전 커밋 · 검증

- [ ] **Step 1:** 실제 실행(백그라운드, pidfile) — `... python research/experiments/maquad_calib.py --workers 16 & echo $! > ~/.cache/cc-swing/scratch/ma_quad/run.pid`. 첫 진행 줄로 남은 시간 재계산.
- [ ] **Step 2:** 결과 파일 커밋 `research(verdict): 이평선 외삽 관문 0 결과(검증 전)`. `swing-visualizer` 로 신뢰도 도표(칸별 A·B0·B1) + 예시 종목 차트(봉·거래량·MA·외삽선 10~90 밴드) → '검증 전' 게시.
- [ ] **Step 3:** 한 메시지에서 `quant-reviewer` 1 + `haiku-verifier` 12(각 사건 8개: 원 DB 로 t+k 정배열 실현값·d·EPS 후보 여부 재계산; 규약 — 정규장 종가, 엄격 부등호, 라벨 끝 2023-12-28, MA 창 NaN 이면 사건 아님)를 띄운다.
- [ ] **Step 4:** CRITICAL/MAJOR 반영 → `정정 MQ-<n>` 문단(PREREG.md 끝) + 커밋 → 같은 URL 갱신 → 판정 보고.
