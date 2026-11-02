# ML 수익률 예측 — 구현 설계 (PREREG 짝, 결과 전)

이 문서는 [`PREREG.md`](PREREG.md) 를 어떻게 구현하는지만 적는다. 바·구간·피처의 정의는 PREREG 가 정본이다. 둘이 다르면 PREREG 를 따른다.

## 1. 의존성

- **krx-quant-core 핀을 0.7.0 → 0.8.0 으로 올린다.** PR #19 가 머지된 판(`8286e3e`, `13c56a7` 포함)이다. 들어오는 것은 `portfolio.optimize`·`risk.factor_model`
  (`spec_vra_half_life` 포함)·`costs.impact`·`stats.sharpe.t_haircut` 이다. B 는 `portfolio-layer` 체크아웃으로 돌았다.
  핀을 올린 뒤 B 패리티(§4 S2-0)로 같은 숫자를 확인한다.
- **LightGBM 을 새 optional extra `ml = ["lightgbm==<구현 시 최신 4.x 고정>"]` 로 넣고, `uv.lock` 도 갱신한다.**
  옛 `ml_pead.py` 는 선언 없는 lightgbm 에 의존해 고아가 됐다. 같은 일을 막는다. 버전은 RUN.md 에도 적는다.
  LightGBM 은 버전이 바뀌면 결정성이 깨진다.
- 릿지는 numpy 닫힌 해로 푼다(sklearn 불필요). Spearman·정규 분위수는 기존 B 코드가 쓰는 scipy 를 쓴다.

## 2. 모듈

`src/` 는 `research/` 를 import 하지 않는다(린트). 일반 기능(정문 읽기)만 `src/` 에 두고, 이 연구의 조립은 `research/experiments/ml_forecast_*.py` 에 둔다.
러너 이름은 `*_gate.py` 가 아니다. `prop_gate` 하버스는 거래 R 기반이라 이 연구에 맞지 않는다. 규칙 (f) 대상도 아니다.

| 파일 | 내용 | 재사용 |
|---|---|---|
| `src/swing_it/storage.py` | 새 정문 둘. `read_investor_trend(con, cols, start, end)`(`naver_investor_trend`), `read_research_reports(con, start, end)`(`naver_research_reports`). 둘 다 폐지 종목 포함을 assert 한다(`_assert_supply_has_delisted` 와 같은 꼴). sqlite `SCHEMA` 에도 표를 넣는다(테스트용) | `read_supply_demand`·`read_earnings` 꼴 |
| `ml_forecast_common.py` | 상수 전부를 둔다: START·END(2023-12-28)·STEP 20·H 20·유니버스 문턱·피처 목록·EW 부호표·LightGBM 파라미터·IC₀·σ_target. `assert_window(end)` 는 END 초과나 ≥ 2024-01-01 이면 `ValueError` 를 낸다 | c0 `keyed_rng` |
| `ml_forecast_data.py` | `build_panels(end)`. core `load_risk_inputs` → `pead_optimizer_seam.seam_guard`(PO-3.3, 원 종가 원천은 B 와 같다) → c0 `exposures` 로 v4 표준화 스타일 → WICS 라벨 → core `factor_returns` 로 u. 수급·보유율·리포트는 정문으로 읽어 code×date 로 편다. c0 `yoy_signal` 은 END 를 인자로 받게 바꾼다(기본값은 그대로). 표 지문과 함께 scratchpad 피클로 캐시한다 | B `build_ctx` 의 캐시 꼴 |
| `ml_forecast_features.py` | F1~F15 를 순수 함수로 둔다(패널 → code×date). 날짜마다 U(t) 안에서 rank-gauss 로 바꾼다. 모든 함수가 "t 열은 t 까지의 열만 읽는다"를 지키고, 테스트가 검사한다 | — |
| `ml_forecast_target.py` | y_{i,t} = Σ_{s=t+2}^{t+21} u. 15일 규칙과 폐지 보존을 적용하고, 학습용 rank-gauss 를 낸다 | — |
| `ml_forecast_models.py` | `train_rows(T0)`(5거래일 간격, t+21 ≤ T0 퍼지), `fit_predict_{M,E,L,P}`. M 은 시드 1~5 평균이다. `num_threads=1`, 작업자 병렬 | — |
| `ml_forecast_stage0.py` | **라벨 없이** 돈다: 커버리지, 절단 재계산 동일성(무작위 결정일 5개), 폐지 대 상장 결측률. 출력은 `stage0.{json,md}` 다. u·y 는 계산하지 않는다 | — |
| `ml_forecast_ic.py` | 1단계. 원장 기록(민감도 → 1차 순) → OOS 예측 → 일별 IC → N2 1,000회 → N1 20회 → I-1~I-4 → `S1.{json,md}`(검증 전) | B 의 순열 생성 꼴 |
| `ml_forecast_book.py` | 2단계. 1단계를 통과한 후보만 돈다. 일반화한 B 하네스로 북 13개와 널 300×후보를 돌려 P-1~P-7·등급을 내고 `S2.{json,md}` 에 쓴다 | `pead_optimizer_b.simulate`·`match_lambda`·`trade_costs`·`holding_intervals`·`stationary_boot_dir`, `_report` 의 HAC 회귀·`alpha_hurdle`·`deflation` |
| `pead_optimizer_b.py`(수정) | 주입점을 더한다: `universe=`(적격 함수·벤치), `max_weight=`, `sigma_target=`(고정값이면 R 에 맞추지 않는다), `ic_const=`. **기본값 `None` 이면 지금 경로와 비트 동일**하다(`perf` 규약 — 커밋 메시지에 "B.json 재현 같음") | — |
| `ml_forecast_report.py` | md 렌더러. 판정 문장은 등급표에서 기계적으로 고른다 | `pead_optimizer_b_report` |

**원장 N:** `pead_optimizer_b_report.ledger_n` 에 `ml_forecast` 라벨과 상수 3(옛 ML-PEAD 2, RL-1 1)을 더한 `ledger_n_ml()` 을 만든다.
상수는 근거(커밋 SHA)를 주석에 둔다.

## 3. 테스트 (DB·네트워크 불요, 합성 패널)

| 테스트 | 무엇을 막나 |
|---|---|
| `test_ml_forecast_features.py` 절단 동일성 | 합성 패널을 t 에서 잘라 낸 피처 = 전체 패널 피처의 t 열(모든 F). 미래 열을 읽는 피처를 잡는다 |
| 같은 파일 알게 된 날 | 실적·리포트 값이 known 다음 거래일에 처음 나타나는지, 정정판이 무관한 과거를 바꾸지 않는지(`test_earnings_bitemporal` 꼴) |
| 같은 파일 F6 비중첩 | F6 가 (t−60, t−20] 만 쓰고 F5 창과 겹치지 않는지 |
| 같은 파일 EW 부호표 해시 | 부호·방향 피처 목록이 바뀌면 실패한다(`test_pattern_ga_fwm` 의 해시 회귀 꼴) |
| `test_ml_forecast_target.py` | 심은 팩터 모델에서 u 를 복원하는지, y 창이 정확히 t+2..t+21 인지, 15일 규칙, 창 안 폐지 종목이 남는지 |
| `test_ml_forecast_split.py` | 학습 행이 모두 t+21 ≤ T0 인지. `assert_window` 가 2023-12-29·2024-01-02 를 거부하는지(**봉인 구간 접근을 코드로 막는 테스트**) |
| `test_ml_forecast_models.py` 결정성 | 같은 시드면 예측 바이트가 같은지 |
| 같은 파일 누출 주입 | 피처 = y + 잡음을 넣으면 OOS IC > 0.5 가 나와야 한다(배선 확인). 라벨을 섞으면 \|IC\| 가 작아야 한다. 통과만 보는 테스트가 아니라 위반을 주입한다(GUARDRAILS §4.5) |
| `test_ml_forecast_null.py` | N2 순열이 예측 시계열을 보존하는지, 같은 키면 같은 π 인지, 분위수 정의 |
| `test_storage.py` 추가 | 새 정문이 폐지 종목이 빠진 결과에서 `AssertionError` 를 내는지 |
| `test_check_guardrails.py` 추가 | **새 린트 규칙 (h)** — `research/experiments/ml_forecast_*.py` 가 생존자 전용 표(`credit_balance`·`short_selling`·`program_trading`)나 기관 세부 열(`INSTITUTION_DETAIL_COLS`)을 읽으면 실패한다. 위반을 주입하는 테스트를 같이 둔다 |
| `test_pead_optimizer_b_inject.py` | 주입 인자가 `None` 이면 작은 합성 Ctx 에서 옛 경로와 스텝 수익이 비트 단위로 같은지 |

명령: `uv run --extra dev --extra ml pytest -q`, `ruff check ...`, `uv run python scripts/check_guardrails.py`.

## 4. 실행 순서와 커밋 (단계마다 커밋, 다음 단계 숫자를 보기 전)

1. **구현 + 테스트.** `feat(ml_forecast): …`. 데이터 실행 없음.
2. **S0 — 0단계(라벨 전).** 커버리지·절단·결측 관문 → `research(prereg): ML 0단계 관문(라벨 전)`.
   실패한 피처는 이 시점에 정정 ML-n 으로 빼고 공개한다.
3. **S1 — 1단계.**
   - 원장 기록 → 실행 → `S1.md`(검증 전) 커밋.
   - 같은 메시지 안에서 병렬 검증을 띄운다. `quant-reviewer` 1개와 `haiku-verifier` 12개다. 입자마다 (결정일, 종목) 5~10개를 원 DB 로 다시 계산한다:
     F3·F4·F5·F7 값, 실적 알게 된 날, t+1→t+21 원 수익.
     프롬프트에 규약을 미리 적는다: t+1 종가 체결, 이음매 가드, 알게 된 날 = 다음 거래일, 2023-01-25 호가 개편.
   - CRITICAL·MAJOR 를 반영한 뒤 `S1_verified.md` 를 쓴다.
4. **S2-0 — 하네스 패리티.**
   - core 0.8.0 + 주입 하네스로 B 상수 → `B.json` O·R 스텝 수익이 1e-10 안에 들어야 한다. 실패하면 VOID 다.
   - 널 3회로 시간을 재고 외삽을 `RUN.md` 에 적는다(PREREG §10. 벽시계 12시간을 넘으면 Modal 을 판단한다).
5. **S2 — 2단계.**
   - 원장 기록 → 실행 → `S2.md`(검증 전).
   - `quant-reviewer` 1개와 `haiku-verifier` 12개(B 와 같은 스텝 원장 재계산 항목)를 띄운다.
   - 리뷰를 반영한 뒤 `VERDICT.md` 를 쓴다. 러너를 등록하고, 맨 위에서 PREREG SHA 를 가리킨다.
6. **결과 공개.** 결과가 나오는 즉시 `swing-visualizer` 로 '검증 전' 페이지를 만든다. 그림은 일별 IC·누적 능동수익·연도별 막대·상위 종목 기여다.
   검증 뒤 같은 URL 을 갱신한다.
   - 공개 레포다. 적합된 모델·피처 중요도·북 성적표는 배포북 성격이라 **푸시하지 않는다**(`docs/alpha-research.md` §8). 사용자와 정한다.

## 5. krx-quant-core 재사용 지도

- `data.risk_inputs.load_risk_inputs`·`style_panels`·`STYLE_NAMES` — 가격·시총·ADV·업종·스타일 원천. 애프터마켓 거부와 폐지 assert 를 그대로 받는다.
- `risk.factor_model.factor_returns`(u) · `build_risk_model(spec_shrink=0.1, spec_vra_half_life=None)` = v4 의 D·Σ.
- `portfolio.optimize`·`Constraints` — B 와 같은 호출.
- `costs.impact.trade_cost_bp`·`costs.tax.sell_tax_rate`·`costs.model.DEFAULT_*` — 정가 비용.
- `stats.sharpe.t_haircut`·`deflated_sharpe_from_sample`, `stats.trials` — 다중검정.
- `backtest.crosssectional._trailing_adv` — ADV 정의를 B 와 맞춘다.

## 6. 위험 목록

누출 경로는 일곱 갈래로 본다.
- (1) `earnings.avail_date` 가 정기보고서 18% 에서 실제보다 이르다. max(avail, periodic rcept_dt) 다음 거래일로 막는다. 2019-02 이전 분기는 avail 로 돌아간다.
- (2) 정정판은 knowledge_date 이중 시간축으로 고른다.
- (3) 리포트 `write_date` 에는 시각이 없다. 그래서 다음 거래일부터 쓴다.
- (4) 수급의 T 당일 값은 잠정이고 이튿날 06:30 에 확정된다. 이력에는 확정치만 있지만, 확정은 t+1 종가 체결 전이다.
- (5) 네이버 보유율 이력은 2026-10 에 백필한 값이다. 사후 정정이 섞였는지는 모른다. 보고 한계로 적는다.
- (6) 타깃의 잔차화는 미래 노출을 쓰지만 라벨에만 들어간다.
- (7) 이 연구자는 2020~23 을 이미 여러 번 봤다(코로나 급락, 2023 이차전지 랠리). 그래서 rank-gauss 타깃과 P-5 단일 종목 상한을 둔다.

생존편향은 세 곳에서 막는다. 로더의 폐지 assert, 생존자 전용 원천 배제(린트 (h)), 결측률 관문이다. 폐지 수급은 기관·외국인만 있어서 F8 은 기관 **합계**를 쓴다.
`stocks.market` 은 현재 스냅샷이다. 그래서 이전 상장 종목은 반스프레드를 잘못 잡는다(B 와 같은 한계).
애프터마켓 혼입은 END 2023-12-28 로 구조적으로 피한다. 전진에는 정규장 경로가 먼저 필요하다.
수정주가 이음매는 PR #12·#14 로 수리된 표를 쓰고, 지문을 기록하고, PO-3.3 가드를 그대로 둔다. 다만 두 가지 오탐이 남는다:
- 심판 없는 병합. 014100 2021-03-15 에서 가드가 진짜 −5.5% 하루를 지웠다.
- 정지 뒤 재개일 점프.

그래서 가드가 걸린 셀 수와 그 셀이 타깃 상위·하위 1% 에 든 비율을 보고한다.
마지막으로 통계 쪽 위험이 있다. 학습 행은 20일 라벨이 겹쳐 실효 표본이 작다. 그래서 학습 쪽 숫자로는 추론하지 않는다. 그리고 LightGBM 버전 결정성은 핀으로 묶는다.
