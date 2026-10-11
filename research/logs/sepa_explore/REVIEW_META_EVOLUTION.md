# 메타 진화 검증 리뷰 (검증 에이전트 — 상시 동행)

## 1차: bea20af 코드·1세대 실행 스크립트 (2026-10-11 16:10, G1A 실행 중)

### MUST-FIX

1. **봉인 구간(2024-01~)이 진화 모드 OOS 에 섞인다 — `pattern_ga_run.py:1923-1924`(`run_evolve`의 `tests`·`E.stitched`)**
   - 원인: `fold_windows` 는 시험 창을 `Win(E, test_lo, test_hi, None)` 으로 만든다(`:852`). `end=None` 이면 `t_end = E.end_row + 1`(2026-09-11, `:661`)이다.
     전체 러너는 폴드 6 이 마지막이라 맞다. 진화 모드는 폴드 4(2023)가 마지막이라 포트폴리오 날이 2024-01 ~ 2026-09 까지 이어진다.
   - 결과 (a) 2023-12 진입 거래가 2024 가격으로 보유·청산된다. 최대 120일이고, 수익·청산 사유·`_trades.csv` 의 `exit_px`/`exit_day` 에 들어간다.
     META_EVOLUTION.md "폴드 5·6 을 계산하지 않는다" 위반이다.
     (b) `ann` 은 약 80개월 평균 ×12 다. 활동 48개월 + 이후 0 이 약 32개월이라 약 0.6배로 희석된다. 그래서 "러너 바 1 과 같은 값"이 아니다.
     `fold_ann[3]` 은 `sel = mid >= starts[3]` 에 2024~2026 이 다 들어가 약 0.27배로 희석된다.
     후보끼리 순위는 거의 보존된다(같은 배수). 다만 절대값과 폴드 4 값은 틀린다.
   - 고침 (`run_evolve`, `fold_windows` 직후·`stitched` 전, 메인 프로세스만 — GA 학습창은 폴드 4 학습 = 2020~2022 라 무영향):
     ```python
     seal = E.row(FOLDS[EVO_FOLDS[-1]].test_hi)            # 2024-01-01 행
     E.LIM = np.minimum(np.asarray(E.LIM), seal)            # 경로가 2023 마지막 종가에서 R_DATA_END 로 끝나게(rec·trades.csv 도 2024 가격 없음)
     E.end_row = seal - 1
     tests = [te for _, te in wins]
     f4 = FOLDS[EVO_FOLDS[-1]]; tests[-1] = Win(E, f4.test_lo, f4.test_hi, f4.test_hi)   # t_end = seal → 48개월
     ```
     `t_end` 만 자르면 안 된다. `rec` 는 진입 때 전체 경로로 채워져 `trades.csv` 에 2024 청산가가 남는다. 그래서 `LIM` 도 자른다.
   - G1A 는 이미 이 코드로 돈다. GA(학습창)는 유효하다. **stitched 재평가만 다시 하면 된다.** G1B~E 는 새 프로세스라 지금 고치면 반영된다.
     G1A 는 cfgs 로 재평가한다(MINOR 1 참고). 고치기 전에는 G1A `_trades.csv` 의 `exit_day ≥ 2024-01-01` 행과 `ann`·`fold_ann` 을 열지 않는다.

### MINOR

1. `pattern_ga_run.py:1936` `met["cfgs"]` 를 `str(v)` 로 저장한다. 그래서 숫자·inf 형이 사라져 재평가하려면 파싱해야 한다.
   원 dict 를 그대로 저장한다(`json.dump(default=float)`, inf 는 `Infinity`).
2. `pattern_ga_run.py:1935` `loss55` 는 첫 단위 `exit_px/entry_px` 만 본다. 그래서 불타기 추가분 손실(대양전기공업 −2.0% 사례 같은 것)이 빠진다.
   거래 순손익으로 센다: `(pmix_net(t, p, "net") < -0.055 * t["alloc"]…)`. 또는 기존 FWM 보고와 같은 정의를 쓴다.

### 확인(문제 없음)

- 메뉴 덮어쓰기: 단일 수준 NOM(`entry ["bottom"]`)은 argmax 0 이다. `"inf"` 는 ∞ 로 바뀐다. EVO_GENES 는 이름이 있을 때만 DISC 끝에 붙는다. 모르는 이름은 종료한다.
  θ 0.1 = `PE.THETAS` 0.10 이다. `pyramid "off"` 는 FW 에서도 `exit_vec` 인덱스 0 이다.
- 엔진: 장중 블록의 `hh` 는 전날까지 최고가다(갱신은 `:526`, 뒤에 있음). 그래서 미래를 보지 않는다.
  종가 블록은 당일 `hh` 를 쓰는데, 종가 시점에 알려진 값이라 괜찮다. `t_close` 는 장중 추적을 끄고 종가 추적을 켠다(배타).
- `stop_policy=skip`: `cap` 뒤, `dfloor` 앞이다. 판단 `e = max(시가, 주문가)` 는 체결 시점에 아는 값이다. 재진입은 끌어올림 고정이다(`:804`, `:878`).
- 1세대 스크립트 메뉴는 문서 표와 일치한다(G1A~E).
