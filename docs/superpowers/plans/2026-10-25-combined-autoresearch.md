# 세 레포 종합 오토리서치 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** swing v4 평가기를 바탕으로, 세 레포(swing·scalp·daytrade) 부품을 합친 구조를 고르는 고정 평가기 + 루프 + 사후 후보 선택기를 만들고, 후보가 나오면 전진 팔 `leader_X` 로 올린다.

**Architecture:** 새 디렉터리 `research/combined/`. 순수 도우미(AST 허용 목록, 표본 추출, 고원 선택, 청산선, 팩터 t, bwrap 명령)는 `research.autoresearch.prepare` 에서 import 만 하고 그 파일은 바꾸지 않는다(MANIFEST 로 고정). 데이터 구축(`data.py`·`gaps.py`·`news.py`), 루프 평가기(`prepare.py`), 루프 뒤 후보 선택(`select.py`)을 파일로 나눈다. 루프는 `train.py` 만 고친다.

**Tech Stack:** Python 3.12, numpy·pandas·numba(엔진), psycopg2(읽기 전용 DB), bwrap, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-25-combined-autoresearch-design.md` (fcs 커밋 040d32b). 실행자는 계획과 설계를 함께 읽는다.

## Global Constraints

- 작업 위치: fcs 에서 가지 친 브랜치 `combined-ar`, 워크트리 `~/git/.harness-wt/swing-it/combined-ar`(Orca `worktree create`; Orca CLI 가 안 되면 사용자에게 요청). fcs·`~/git/swing-it` 에서 직접 작업하지 않는다.
- `research/autoresearch/*`·`research/experiments/leader21.py`·`indep30.py`·기존 장부·기존 팔은 바꾸지 않는다.
- 푸시는 사용자가 말할 때만. origin 은 공개 — 캐시·장부 원본·`data/` 는 커밋하지 않는다.
- 비밀값: `set -a; . ~/git/quant-airflow/.env; set +a`. DB 연결은 `con.set_session(readonly=True)`.
- 테스트: `uv run --with numba --extra dev pytest -q tests/test_combined_*.py`. 린트: `uv run --extra dev ruff check research/combined tests/test_combined_*.py`. 가드레일: `uv run python scripts/check_guardrails.py`.
- numba 작업자: `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1`. 프로세스는 pidfile 로만 다룬다(`pkill -f` 금지).
- 커밋: 한국어 conventional(`research(prereg): …`, `feat(combined): …`), 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git add` 는 경로 지정.
- 구간: 시대 A 데이터 2008-01-01~2016-08-31, 시험 폴드 2011~2015(5). 시대 B 데이터 2016-09-01~2024-12-31, 시험 폴드 2020~2024(5). 확인 구간 C: 데이터 2016-09-01~2026-09-11, 학습 2023-01-01~2025-01-01, 시험 2025-01-02~2026-09-12(1).
- 종목 분할: `research.autoresearch.prepare.half_of`(무솔트, 짝 = 탐색).
- 비용 `cost_mult` 2.0(점수·바), 1.0 은 기록만. 진입 = 신호일 종가(`entry="close"`).
- 상수: `KEEP_DSCORE = 0.005`, `KEEP_CONC_SLACK = 0.01`, `MAX_TRIALS = 150`, `STOP_NO_KEEP = 30`, `ABL_MIN_CHANGE = 0.05`, `POOL_MAX = 10`, `N_NULL = 100`, `NEWS_CUTOFF = "15:10"`, `PREREG_TAG = "combined-prereg-1"`, `SEED = 20261025`.
- 뉴스 원천: `~/git/daytrade-it/data/toss_universe_42m/<code>.jsonl`(`createdAt`, `title`). 정규식은 daytrade `scripts/eval/autoresearch/prepare.py` 의 `FEAT_RE`·`EXCL_RE`·`MOVE_RE` 를 그대로 복사한다(import 하지 않는다 — 다른 레포).

## Review Focus

1. **아카이브에 없는 종목·날짜와 기사 0건을 구분해야 한다.** `news_known=False` 면 개수는 0 이지만 "모름"이다. → Task 2 `test_unknown_differs_from_zero`.
2. **15:10 뒤 기사와 휴장일 기사는 다음 거래일로 넘어가야 한다.** 신호일에 새면 일 단위 절단 검사로는 못 잡는다. → Task 2 `test_cutoff_and_holiday_roll`.
3. **분할일에 가짜 −50% 갭이 생기면 안 되고, 정지 뒤 재개일 갭은 NaN 이어야 한다.** 틱/가격은 원주가로 계산한다. → Task 1 `test_split_day_gap_and_reopen_nan`, `test_tickp_uses_raw_price`.
4. **확인 구간 캐시는 루프 샌드박스 안에서 읽히면 안 된다.** → Task 3 `test_confirm_cache_hidden_in_sandbox`.
5. **같은 train sha 재실행은 시행 수를 늘리지 않고, 상한 뒤에는 평가를 거부해야 한다.** → Task 4 `test_trial_cap_and_same_sha`.

---

### Task 1: 야간 갭 특성 (`gaps.py`)

**Files:**
- Create: `research/combined/__init__.py`, `research/combined/gaps.py`
- Test: `tests/test_combined_gaps.py`

**Interfaces:**
- Produces: `gap_arrays(O: np.ndarray, C: np.ndarray, raw_close: np.ndarray, reopen: np.ndarray, lu: np.ndarray) -> dict` → `{"gap": float32[N,T], "tickp": float32[N,T], "gap_event": bool[N,T]}`.
  - `gap[:, t] = O[:, t] / C[:, t-1] - 1`. 같은 수정가 계열끼리 나누므로 분할 불변이다. `reopen` 칸과 t=0 은 NaN.
  - `tickp[:, t] = krx_tick(raw_close[:, t-1]) / raw_close[:, t-1]`. 호가 단위는 `swing_it.features.korea` 의 기존 틱 함수(없으면 krx_quant_core 의 것)를 쓰고, 2023-01-25 호가 개편을 반영한다.
  - `gap_event = |gap| >= (lu[t] - 1) / 2`. 제한폭 절반 이상, 상승·하락 대칭.

- [ ] **Step 1: 실패 테스트 작성** — `test_split_day_gap_and_reopen_nan`: 2:1 분할을 수정가로 소급한 계열이면 분할일 `gap` ≈ 0 이고, `reopen[0, 5]=True` 면 `gap[0, 5]` 는 NaN. `test_tickp_uses_raw_price`: 수정가 60,905·원주가 3,160 이면 `tickp` = tick(3160)/3160(≈15.8bp)이어야 하고 1.6bp 가 아니다(scalp VERDICT §5 사례). `test_gap_event_symmetric`: lu 1.30 에서 +15%·−15% 둘 다 True, +14.9% 는 False.
- [ ] **Step 2: 실패 확인** — `uv run --with numba --extra dev pytest -q tests/test_combined_gaps.py` → ImportError.
- [ ] **Step 3: `gap_arrays` 구현**
- [ ] **Step 4: 통과 확인** — 같은 명령, 3 passed.
- [ ] **Step 5: 커밋** — `feat(combined): 야간 갭·원주가 틱/가격·사건 갭 배열`

### Task 2: 뉴스 배열 (`news.py`)

**Files:**
- Create: `research/combined/news.py`
- Test: `tests/test_combined_news.py`

**Interfaces:**
- Produces: `news_arrays(codes: list[str], days: pd.DatetimeIndex, archive: Path) -> dict` → `{"n_all","n_feat","n_move","n_excl": uint16[N,T], "news_known": bool[N,T]}`.
  - 기사는 `createdAt <= days[t] 15:10` 이고 `createdAt > days[t-1] 15:10` 이면 t 에 센다. 휴장일·15:10 뒤 기사는 다음 거래일로 간다. `(id, code)` 로 중복을 없앤다.
  - `news_known[i, t]` = 그 종목 아카이브의 첫 기사 시각 ≤ days[t] 15:10. 파일이 없으면 전부 False. 2023-03-15 전은 전부 False.
  - `archive_dir() -> Path` 는 `~/git/daytrade-it/data/toss_universe_42m` 를 찾고, 없으면 SystemExit.

- [ ] **Step 1: 실패 테스트 작성** — 임시 디렉터리에 jsonl 을 쓴다.
  - `test_cutoff_and_holiday_roll`: 금 15:09 `[특징주]` → 금 `n_feat` 1. 금 15:11 → 월 1. 토 10:00 → 월 1.
  - `test_unknown_differs_from_zero`: 파일 없는 종목은 `news_known` 전부 False. 있는 종목은 첫 기사 이후 True 이고, 기사 없는 날은 개수 0.
  - `test_dedup_id_code`: 같은 `(id, code)` 두 줄 → 1.
- [ ] **Step 2: 실패 확인** — `pytest -q tests/test_combined_news.py` → ImportError.
- [ ] **Step 3: `news_arrays`·`archive_dir` 구현** — 정규식 세 개는 daytrade 원문 그대로 모듈 상수로 둔다. 출처 주석 한 줄을 단다.
- [ ] **Step 4: 통과 확인** — 3 passed.
- [ ] **Step 5: 커밋** — `feat(combined): 토스 뉴스 아카이브 → 15:10 경계 종목×일 배열`

### Task 3: 시대 캐시 구축 (`data.py`)

**Files:**
- Create: `research/combined/data.py`
- Test: `tests/test_combined_data.py`

**Interfaces:**
- Consumes: `gap_arrays`(Task 1), `news_arrays`(Task 2), `research.autoresearch.prepare.build_era` 의 단계를 복사한 구축(그 함수는 `ERAS` 전역을 읽으므로 import 하지 않고 같은 단계를 `ERAS_C` 로 재현한다), `half_of`, `slice_half`.
- Produces:
  - `ERAS_C: dict` — 키 `"A"`·`"B"`·`"C"`. 각 값은 `{"data": (lo, hi), "eps": str, "folds": [(train_lo, test_lo, test_hi), …]}`. 값은 Global Constraints 의 구간 그대로이고, C 의 eps 는 `"earnings"`.
  - `CACHE = ~/.cache/cc-swing/scratch/combined`.
  - `build_all() -> None` — 시대마다 `era_{E}_s0.pkl`(탐색 절반)을 쓴다. 잠금 절반은 디스크에 쓰지 않는다.
  - `load(era: str, half: int = 0) -> dict` — half 1 은 DB 에서 메모리로 만든다(`select.py` 만 부른다).
  - `VIEW_KEYS_C = (*v4 VIEW_KEYS, "gap", "tickp", "gap_event", "n_all", "n_feat", "n_move", "n_excl", "news_known")`.
  - `REPO_KEYS = {"scalp": ("gap", "tickp", "gap_event"), "daytrade": ("n_all", "n_feat", "n_move", "n_excl", "news_known")}`.
    swing 은 엔진·청산 메뉴·EPS 강제가 늘 쓰이므로 제거 검사 대상이 아니다. 이 점을 PREREG 에 적는다.
- 가격은 정문으로만 읽는다: `read_prices(session="regular", unpatched="raw")`, 원주가는 `read_raw_close(aftermarket_fix=True)`. 린트 (p) 를 통과해야 한다.

- [ ] **Step 1: 실패 테스트 작성**
  - `test_eras_shape`: A 폴드 5개, B 폴드 5개, 마지막 B 시험 = 2024. C 폴드 1개, 시험 2025-01-02~2026-09-12. B 데이터 끝 = 2024-12-31.
  - `test_view_keys_have_no_dates_or_codes`: 장난감 D 로 `view_c(D)` 를 부르면 키가 `VIEW_KEYS_C ∪ {kx, kq, regime_on}` 과 같고, 값은 모두 ndarray 복사본이다.
  - `test_confirm_cache_hidden_in_sandbox`: `prepare.sandbox_cmd_c([...])` 결과에 `era_C_s0.pkl` 경로가 `--tmpfs` 로 가려져 있거나 바인드 목록에 없다.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: `ERAS_C`·`build_all`·`load`·`view_c` 구현** — `view_c` 는 Task 4 의 prepare 에 둬도 된다. 그럴 경우 테스트 import 를 맞춘다.
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 실제 구축 1회.** `set -a; . ~/git/quant-airflow/.env; set +a; NUMBA_NUM_THREADS=1 uv run --with numba --extra pg python -m research.combined.data --build`.
  - 확인: 진단(`news_known` 비율 — 2023·2024 의 uni 칸 중 True 몫, `gap` NaN 비율, `reopen` 칸 수)을 stderr 로 찍는다.
  - **점수는 계산하지 않는다.**
- [ ] **Step 6: 커밋** — `feat(combined): 시대 A·B(~2024)·확인 C 캐시 구축 — 탐색 절반만 디스크에`

### Task 4: 루프 평가기 (`prepare.py`)

**Files:**
- Create: `research/combined/prepare.py`, `research/combined/train.py`(기준선), `research/combined/program.md`
- Test: `tests/test_combined_guards.py`

**Interfaces:**
- Consumes: Task 3 의 `ERAS_C`(A·B 만 루프에), `load`, `REPO_KEYS`, `VIEW_KEYS_C`. `research.autoresearch.prepare` 의 `lint_train`, `sample_params`, `plateau_pick`, `exit_line`, `clamp_cfg`, `factor_months`, `factor_t`, `causality` 단계(시대 키·VIEW 를 인자로 받지 않으면 복사).
- Produces (`research/combined/prepare.py`):
  - `simulate_c(D, sig, cfg, start, end, exclude=(), mask=None, cost_mult=2.0)`
    - v4 `simulate` 와 같되 `entry="close"` 다.
    - `mask: bool[N,T] | None` 는 cand 에서 끌 칸이다. 상위 1% 거래 제거에 쓴다.
    - 강제 AND(uni ∧ TR ∧ eps_up)·`clamp_cfg`·`paper=True`·`net_mtm=True` 는 v4 와 같다.
    - void 규칙도 v4 그대로다: 체결 < 5 인 폴드 2개 초과, 시험 체결 < 150, 연 < 20.
  - `walk_forward_c(D_by_era, workers, train_path, fixed=None, null=None, neutral: str | None = None) -> dict`
    - `neutral` 은 `"scalp"|"daytrade"`. V 의 그 레포 키를 중립값으로 바꾼다: `gap`·`tickp` → 0.0, `gap_event` → False, 뉴스 개수 → 0, `news_known` → False.
    - 반환 `res["folds"]` 의 각 폴드에 `trade_key: str` 를 넣는다. 시험 체결 `(code_idx, entry_t)` 정렬 목록의 sha1 앞 12자다.
  - `concentration(D_by_era, res, train_path) -> {"top1pct": float, "top_stock": float, "top10d": float}`. 각 값은 제거 뒤 score 다.
    - `top1pct`: 순수익 상위 1%(최소 1건) 거래의 신호 칸을 `mask` 로 끄고 고른 파라미터 그대로 다시 돌린다.
    - `top_stock`: 순수익 합 최대 종목을 `exclude` 해서 다시 돌린다.
    - `top10d`: 다시 돌리지 않는다. 시험 구간 일별 선택 알파 상위 10일을 0 으로 두고 score 를 다시 낸다.
  - `ablation(D_by_era, res, train_path) -> {"scalp": {"changed": float, "dscore": float, "ok": bool}, "daytrade": {...}, "eligible": bool}`
    - `changed` 는 제거했을 때 체결 집합 대칭차 ÷ 원래 체결 수다.
    - `dscore` 는 원래 점수 − 제거 점수다. 부품이 작동하는 폴드만 쓴다. daytrade 는 2023·2024 폴드다.
    - `ok = changed >= 0.05 and dscore > 0`, `eligible = all ok`.
  - `beats_c(s, folds, conc) -> (bool, str)`
    - `changed = [f for f, g in zip(folds, best.folds) if f.trade_key != g.trade_key]` 로 체결이 바뀐 폴드를 모은다.
    - keep 조건: `len(changed) >= 1`, 바뀐 폴드 중 alpha 개선이 과반, `Δscore >= 0.005`, `conc` 세 값이 각각 BEST 의 같은 값 − 0.01 이상.
  - `trial_state() -> {"n": int, "since_keep": int, "stop": bool}`
    - 원장 `research/combined/RUNS.jsonl` 과 사본 `~/.cache/cc-swing/combined_ledger/` 를 읽는다. N 은 고유 train sha 수다.
    - `stop = n >= 150 or (첫 keep 이후 since_keep >= 30)`.
    - `stop` 이면 `evaluate` 는 "상한 도달 — 평가 거부"로 SystemExit 한다.
  - 출력 줄(run.log, 이 순서): `score: void: beats_best: beats_detail: lookahead_check: folds_pos: oos_trades: payoff_ratio: ev_net_2x: ev_net_1x: top1pct_removed: top_stock_removed: top10d_removed: news_folds_score: ablation_scalp: ablation_daytrade: eligible: trials: stop:`. 폴드별 값은 찍지 않는다.
  - CLI: `--build`(Task 3 위임), `--manifest`, `--accept`, `--inner`, `--workers`. `--final` 은 없다(선택은 `select.py`).
  - 무결성: `PREREG_TAG = "combined-prereg-1"`. `GUARDED_FILES` 는 `research/combined/{prepare,data,gaps,news,select}.py`, `research/autoresearch/prepare.py`, `research/experiments/leader21.py`, `research/experiments/indep30.py` 다. MANIFEST 에는 캐시 `era_A_s0`·`era_B_s0`·`era_C_s0` sha 와 krx_quant_core DSR 파일이 들어간다.
  - 샌드박스: v4 `sandbox_cmd` 에서 바인드를 `era_A_s0`·`era_B_s0` 둘로만 바꾸고, `era_C_s0`·원장 사본·daytrade 아카이브를 가린다.
- `train.py` 기준선은 v4 기준선 계약을 그대로 따른다. 차이는 셋이다.
  - 진입은 종가(평가기가 강제하고 train 은 정하지 않는다).
  - `signals` 는 tt ∧ N일 돌파 ∧ RS ≥ θ 이고, score 는 RS 다.
  - params 는 `theta (int 60~95)`, `N (choice 20,50,120)`, `ma_exit (int 50~200)`, `hard (float 0.10~0.25)`, `exit_kind`, `low_w` 다. 뉴스·갭은 쓰지 않는다. 따라서 기준선은 자격 없음이 정상이다.
- `program.md`: v4 program 을 바탕으로 다음을 바꾼다.
  - 출력 줄 grep 목록, keep 은 `beats_best`.
  - "세 레포 부품"의 뜻과 `eligible`(자격 없는 keep 도 keep 이다 — 다음 시도의 디딤돌이다).
  - 뉴스 부품에는 숫자 파라미터 금지(params 에 뉴스 관련 이름을 두지 않는다).
  - 금지 목록은 v4 그대로에 e104063 구조(시장 대비 수축 + 실적 신선도 조합) 재제출 금지를 더한다.
  - 상한 150 / 30.

- [ ] **Step 1: 실패 테스트 작성(위반 주입형 — GUARDRAILS §4.5)**
  - `test_entry_is_close`: 장난감 D 에서 `simulate_c` 거래의 진입가 == 신호일 C.
  - `test_ablation_flags_decorative_component`: train 이 `gap` 을 읽기만 하고 결과에 안 쓰면 `ablation["scalp"]["ok"] is False` 이고 `eligible is False`.
  - `test_ablation_neutral_values`: `neutral="daytrade"` 로 V 를 만들면 `news_known` 은 전부 False, 개수는 전부 0.
  - `test_keep_counts_changed_folds_only`: 2개 폴드만 바뀌고 둘 다 개선 + Δscore 0.006 이면 True. 바뀐 폴드 2개 중 1개만 개선이면 False.
  - `test_keep_concentration_vs_best`: `conc.top_stock` 이 BEST 대비 −0.011 이면 False, −0.009 이면 True.
  - `test_trial_cap_and_same_sha`: RUNS 에 같은 sha 2줄 → n 1. 고유 150 → stop, `evaluate` SystemExit. 첫 keep 뒤 30 연속 무 keep → stop.
  - `test_lookahead_injection_voids`: t+1 C 를 읽는 train → `lookahead_check` 실패, void.
  - `test_integrity_refuses_changed_guarded_file`: 태그 뒤 `gaps.py` 를 바꾸면 SystemExit. git 임시 저장소 픽스처를 쓴다.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: `prepare.py` 구현** — 기준선 `train.py`·`program.md` 도 같이 쓴다.
- [ ] **Step 4: 통과 확인** — `pytest -q tests/test_combined_*.py`, 그다음 전체 `uv run --with numba --extra dev pytest -q`(약 3분, 기존 실패가 늘지 않는지 본다).
- [ ] **Step 5: 스모크 1회(점수 미열람).** `uv run python research/combined/prepare.py > /tmp/…/smoke.log 2>&1`. 그다음 `void:`·`lookahead_check:`·`oos_trades:`·`trials:` 줄만 grep 한다. score 줄은 보지 않는다. 걸린 시간을 적는다.
- [ ] **Step 6: 커밋** — `feat(combined): 루프 평가기 — 종가 진입·부품 제거·쏠림 keep·시행 상한(결과 전)`

### Task 5: 사후 후보 선택기 (`select.py`)

**Files:**
- Create: `research/combined/select.py`
- Test: `tests/test_combined_select.py`

**Interfaces:**
- Consumes: Task 4 의 `walk_forward_c`, `concentration`, `ablation`, `trial_state`, 원장. Task 3 의 `load(era, half)`.
- Produces:
  - `pool(runs: list[dict]) -> list[dict]` — `eligible` ∧ `not void` 인 시도 중 keep 전부 ∪ score 상위 10, 최대 10(score 순).
  - `screen(rec, D_by_era, train_path) -> dict` — 거르개다.
    - score > 0, ev_net_2x > 0, top1pct_removed > 0, top_stock_removed > 0, 연 체결 ≥ 20.
    - score > 파라미터 무작위 널 100회의 95 백분위. v4 의 `null=("params", seed)` 경로를 쓴다.
    - 팩터 NW t 와 DSR(N = 고유 시행 + 280)은 기록만 한다.
  - `confirm(sha, train_path) -> dict` — 시대 C 탐색 절반, 학습 창 고원 → 2025-01-02~2026-09-12 시험 1회. `CONFIRM.jsonl` 에 덧붙인다(+ 사본). 같은 sha 를 다시 부르면 거부.
  - `lockbox(sha, train_path) -> dict` — 잠금 절반에서 A·B 10폴드 + C 1폴드를 같은 절차로 돌린다. 바는 `score > 0 ∧ ev_net_2x > 0`. `LOCKBOX.jsonl` 에 남기고, 두 번째 호출은 거부(최대 1회).
  - `main(--reviewed <sha16>)` — pool → screen → confirm(통과한 것 전부) → 확인 score 최대(동점이면 train.py 줄 수 적은 쪽) → lockbox. `SELECTION.json` 을 쓴다.
    - `SELECTION.json` 의 `chosen` 은 sha 또는 null, `frozen_params` 는 C 학습 창 고원 값, 사유도 함께 적는다.
    - `train.py` 는 `git show <commit>:research/combined/train.py` 로 scratch 에 꺼내 경로로 넘긴다. `--reviewed` 가 그 sha 와 다르면 거부한다(사람 열람 바, v4 b8).
    - e104063 구조 재제출 여부는 사람이 `--reviewed` 전에 확인한다(program·PREREG 에 적힌 대로).

- [ ] **Step 1: 실패 테스트 작성**
  - `test_pool_only_eligible_max10`.
  - `test_confirm_refuses_same_sha_twice`.
  - `test_lockbox_once`.
  - `test_no_fallback_after_lockbox_fail`: 1위가 잠금에서 실패하면 `chosen is None` 이고, 2위를 돌리지 않는다(호출 횟수 확인).
  - `test_reviewed_sha_required`.
- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현**
- [ ] **Step 4: 통과 확인**
- [ ] **Step 5: 커밋** — `feat(combined): 사후 후보 선택 — 풀·거르개(널)·확인 1회·잠금 1회(결과 전)`

### Task 6: 사전등록·리뷰·태그

**Files:**
- Create: `research/combined/PREREG.md`, `research/combined/MANIFEST.json`
- Modify: `.claude/settings.json`(deny 에 `research/combined/{prepare,data,gaps,news,select}.py`, `MANIFEST.json`, `PREREG.md`, `*.jsonl`, `BEST.json`, `SELECTION.json` 편집과 `Bash(git tag:*)` — v4 deny 방식 그대로), `.gitignore`(`research/combined/results.tsv`, `run.log`)

- [ ] **Step 1: `PREREG.md` 작성(결과 전).** 설계 문서의 정한 것 전부를 결과 전 문장으로 옮긴다.
  - 구간, 분할, 뉴스 규칙, 부품 메뉴와 강제, keep, 상한, 선택 규칙, 전진 바와 판정일(2027-01-29 중간 점검, 2027-04-30 판정, 40건 미만이면 최대 2027-10-29).
  - 공개 항목: 2024~26 재사용은 사용자 승인 예외라 증거가 아니다. swing 부품은 제거 검사 대상이 아니다. Task 3·4 스모크에서 본 것(진단·체결 수만).
- [ ] **Step 2: `python research/combined/prepare.py --manifest`** → MANIFEST.json.
- [ ] **Step 3: quant-reviewer 1회(점수 미열람)** — 평가기·데이터·선택기·PREREG 를 심사시킨다. CRITICAL·MAJOR 는 코드나 PREREG "정정 0"(태그 전이라 본문 수정 가능)으로 반영하고, 리뷰 원문 요약을 PREREG 끝에 둔다.
- [ ] **Step 4: 테스트 전부 통과 확인** → `--manifest` 다시 → 커밋 `research(prereg): 세 레포 종합 오토리서치 사전등록(결과 전)`.
- [ ] **Step 5: 태그** `git tag combined-prereg-1`(deny 가 막으면 사용자에게 `! git tag combined-prereg-1` 요청). 그다음 `prepare.py` 를 한 번 돌려 무결성 통과와 기준선 RUNS 1행을 확인한다.

### Task 7: 루프 실행

- [ ] **Step 1:** `combined-ar` 워크트리에서 Orca 터미널(또는 tmux 세션 `combined-loop`)로 claude 루프 에이전트를 띄운다. 프롬프트는 "research/combined/program.md 를 읽고 따른다"다.
- [ ] **Step 2:** 메인 세션은 진행을 기다린다(Monitor, `trials:` 줄). 실험 50·100·종료 시점에 `swing-visualizer` 로 점수 추이 차트를 '검증 전' 표시로 게시한다.
- [ ] **Step 3:** `stop: True` 가 되면 루프 세션을 끝낸다(pidfile·tmux 세션 이름으로만).

### Task 8: 후보 선택·검증·판정 문서

- [ ] **Step 1:** 메인 세션이 풀의 train.py 를 읽는다. e104063 구조 재제출이 아닌지 확인한 뒤 `select.py --reviewed <sha>` 로 실행한다.
- [ ] **Step 2:** quant-reviewer 1 + haiku-verifier 10~20(한 메시지에서 병렬)을 띄운다. 확인 구간 거래를 원 DB 로 재계산한다.
  - 프롬프트에 체결 규약을 적는다: 신호일 종가 진입, 다음 날 시가 청산, 정지일 매도 미룸, 2023-01-25 호가 개편, 하드 손절 체결 규칙.
- [ ] **Step 3:** `research/combined/VERDICT.md` 를 쓴다(scalp VERDICT 형식). 후보 또는 "후보 없음", 손익비·거래당 기대값, 쏠림, 공개 항목을 담는다. 리뷰 CRITICAL·MAJOR 를 반영한 뒤에만 보고한다. 차트(후보 거래 봉·거래량·MA·진입/청산·손절선)도 게시한다.
- [ ] **Step 4:** 커밋 `research(verdict): 종합 오토리서치 후보 선택`. 후보가 없으면 여기서 끝난다.

### Task 9: 전진 팔 `leader_X` (후보가 있을 때만)

**Files:**
- Create: `research/combined/frozen/leader_x.py`(고른 train.py 사본), `research/combined/live.py`, `research/combined/LEADER_X_PREREG.md`
- Modify: `research/paper/leader_paper.py`(ARMS·PAIR·ARM_LEDGER·DETAIL_ARMS 에 `leader_X` 추가, 기존 팔 경로 불변)
- Test: `tests/test_combined_live.py`

**Interfaces:**
- `live.py`
  - `signals_asof(con, asof: date) -> tuple[dict, dict]` — `data.py` 의 같은 구축 함수로 전 종목(두 절반 모두)·2023-01-01~asof 배열을 만들고, 얼린 train 의 `signals(V, frozen_params)` 를 돌린다. 반환은 `(D, sig)` 다.
  - 뉴스 아카이브가 asof 15:10 까지 갱신됐는지 확인하고, 아니면 그날 장부 행에 `news_stale=True` 를 적는다. 신호는 그대로 낸다(사전등록에 적는다).
- `leader_paper.py` 의 `leader_X` 는 이 D·sig 로 `simulate_c` 와 같은 엔진 설정(`entry="close"`, cfg 는 frozen)을 써서 장부 `research/paper/leader_x_ledger.jsonl` 에 행을 덧붙인다. 짝은 `leader_C_rs`.

- [ ] **Step 1: 실패 테스트** `test_live_matches_backtest_signals`: 같은 D 로 `signals_asof` 의 마지막 날 cand 가 백테스트 경로의 같은 날 cand 와 같아야 한다(피처 패리티).
- [ ] **Step 2~4:** 구현하고 통과를 확인한다. 기존 팔 장부 바이트가 그대로인지 확인한다(`--arm leader_C --asof <어제>` 를 드라이런한 출력이 장부 마지막 행과 같아야 한다).
- [ ] **Step 5:** `LEADER_X_PREREG.md` 에 다음을 적고, 첫 장부 행 전에 커밋한다.
  - 규칙 sha, frozen_params, 첫 기록일, 바 4개와 판정일(설계 그대로).
  - 커밋 `research(prereg): 전진 팔 leader_X 사전등록(결과 전)`.
- [ ] **Step 6:** `combined-ar` 를 fcs 에 머지한다(사용자 확인 뒤). 다음 날 07:43 크론 장부에 `leader_X` 행이 생겼는지 확인한다.
