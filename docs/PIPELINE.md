# 파이프라인 11단계

연구는 아래 순서로만 흐른다. 순서는 게이트와 CI 가 막는다. 부를 함수는 대부분 `src/` 에 있다. 예외는 5단계(연구 코드에만 있다)·10단계(없다)·11단계 보고서다. 지금 상태는 `python -m swing_it.pipeline status` 로 본다.

## 단계와 실제 함수

함수는 러너가 부르고 게이트도 러너가 부른다. 표는 `swing_it.pipeline.stages.STAGE_FUNCTIONS` 와 같고, 시험이 경로와 심볼의 실재를 본다.

| # | 단계 | 함수 | 위치 |
|---|---|---|---|
| 1 | 유니버스 | `rank_band_eligible`·`u_exclusion_causes` | `swing_it.features.universe` |
| 2 | 데이터 | `read_prices`·`read_earnings`·`read_periodic_filings` | `swing_it.storage`. PEAD 러너의 가격은 이 정문이 아니라 `pead_optimizer_po4.load_inputs` → core `load_risk_inputs`(`load_daily_bars`) 다 |
| 3 | 알파 발굴 | `yoy_strict_wide`·`load_yoy_strict` | `swing_it.features.earnings_asof` |
| 4 | 알파 검증(IC/ICIR) | `daily_rank_ic`·`daily_mean_ic`·`quintile_excess`·`ic_summary`·`ic_permutation_null`·`fold_ic` | `swing_it.validation.ic` |
| 5 | ML 수익률 예측 | `research/experiments/ml_forecast_models.py` | **`src` 에 없다.** 연구 코드에 남았다. `check_forecast` 는 있지만 부르는 러너가 아직 없다 |
| 6 | 팩터 위험모델 | `sub_model`·`active_risk`, 빌드는 `load_guarded`·`seam_guard`·`build_panels`·`build_models` | `swing_it.book.risk`, `swing_it.book.riskmodel`·`seam`(core `build_risk_model` 호출) |
| 7 | 포트폴리오 | 순위 북 `tranche_names`·`r_target`, 최적화 북 `o_target`·`match_lambda` | `swing_it.book.rank`, `swing_it.book.optimized` |
| 8 | 거래비용 | `trade_costs`·`linear_bp`·`trailing_vol` | `swing_it.book.costs` |
| 9 | 백테스트 | `simulate`·`check_invariants`·`book_summary`, `full_parity`, `permutation_nulls`, `ols_hac` | `swing_it.book.accounting`·`metrics`·`parity`·`nulls`, `swing_it.diagnostics.hac` |
| 10 | 주문 실행 | main 에 없음 — `docs/execution/DESIGN.md` | 먼저 dry-run. 실주문 경로의 첫 문장은 `require_execution_approval` 이다(린트 (l)). 아래 **게이트가 막지 못하는 주문** 참고 |
| 11 | 성과 기여·모니터링 | `attribution_report`, core `factor_attribution` | `research/experiments/pead_optimizer_b_report.py` |

북의 규칙은 `swing_it.book.spec.BookSpec` 으로 넘긴다. 기본값은 PEAD 단계 B 사전등록 값이다. 판정 바(`verdict`)는 연구마다 사전등록하므로 러너에 남는다.

`research/experiments/pead_optimizer_b.py` 는 위 함수를 부르는 러너가 됐다. **지문 `e66f309e07de` 표에서** 옮긴 뒤의 코드가 `B.json` 의 숫자를 1e-10 안에서 재현했다(`research/logs/pead_optimizer/B_modules_parity.json`, 재현 방법은 `tests/real_db/test_b_json_reproduction.py`).

**지금 표에서는 O 가 어긋난다.** 2026-11-02 16:55 재계산(052670 ×1500 등)으로 표 지문이 `400dfcc393ab` 가 됐다. R 경로는 그대로이고 O 값 약 334개가 움직인다(IR 최대 3.8e-6, 폴드 IR·왜도 수 e-5, 널 q95 2.2e-4, `n_dust` 정수 3). 재현 시험은 지문을 먼저 보고 "표 바뀜"과 "코드 회귀"를 가른다. 기본은 IC 절을 건너뛴다 — IC 패널이 준비 400일 때문에 약 2015-09 부터 읽어 2009~2016-08 잠금 대역에 걸린다. ML S2-0(`ml_forecast_book.py --mode parity`)은 `fingerprint_same_as_B` 가 거짓이 되어 표가 바뀐 지금은 `pass` 가 나오지 않는다. 코드 문제가 아니라 표 문제다.

**1단계 띠의 뜻.** 시총 21~100위는 "as-of 주식수가 있는 종목 안의" 순위다. 전 시장 순위가 아니다. 2021~22 측정에서 주식수 공백으로 평균 26.8종목이 순위에서 빠졌고 띠의 약 13% 가 전 시장 순위 띠와 달랐다. B 숫자는 이 정의로 재현한다. 메우려면 quant-airflow 의 `shares_outstanding_history` 백필이 필요하다(요청 사항). 스텝별 원인 분리는 `StepRec.u_excl_cap_missing`·`u_excl_model`·`cap_missing_in_u` 에 있다.

**2단계 가격 정문.** `read_prices(end=None)` 은 옛 호출 그대로 표 전체를 읽는다. 옛 러너의 숫자를 바꾸지 않으려는 기본값이다. `end` 를 주면 그날까지 (code, date) 순서로 읽고, 2026-09-14 이상이면 `CheckError` 다. 파이프라인을 선언한 새 러너는 린트 (k) 가 `end=` 를 요구한다. **`volume` 은 수정주가 표에서도 조정되지 않는다.** ADV 는 `trade_value`(백만원)만 쓴다. 실적 정정 이력은 2026-08 전에 없다. 코드는 버전을 처리하지만 그 전 값에 정정치가 섞인 정도는 측정하지 않았다(GUARDRAILS §4-3).

**3단계 신호의 기본과 재현.** `yoy_strict_wide` 의 기본값은 새 연구용이다. 정기보고서 요구는 2018Q4 부터(옛 2019Q1 은 end 2019-04~05 에서 절단 동일성이 깨진다), 정정본은 알게 된 다음 거래일부터, 같은 날 알려진 분기는 늦은 분기다. 단계 B·P4 재현 러너만 `REPRODUCE_B` 를 넘긴다. 옛 분기가 새 분기보다 늦게 알려진 역전(실측 3건)은 `tie_col` 로 풀리지 않는다. 값은 새 분기가 올 때까지 이어지고 나이 제한은 `max_age_days` 로 건다.

**4단계에서 보고할 것.** 순위 IC 는 평균 정보량을 과대 표시한다(ML S2: 순위 .041 대 평균 .014). 롱온리는 평균만 수확하므로 `daily_mean_ic`·`quintile_excess` 의 Q5−U 와 2배 왕복 비용을 함께 적는다. 요약은 `ic_summary`(NW lag = h, 비겹침 ICIR), 널은 `ic_permutation_null` 이다. 책 전체를 돌리는 `permutation_nulls` 는 9단계 널이고, 두 널 모두 π 가 신호의 베타·변동성·사이즈 상관을 지운다(기울기 입력에서 기각률 23~28%). 팩터 회귀와 함께 쓴다. 폐지 뒤 수익은 0% 로 잰다.

**5단계의 한계.** N1(라벨 섞기)은 학습 경로 스모크다. 누출 검사가 아니다. 주입한 누출 6종을 하나도 못 잡았다. 누출은 절단 동일성과 `check_forecast` 로 본다. ML 연구는 닫혔고 `check_forecast` 를 안 부른 채 끝났다. 새 예측 러너는 거래일 달력을 넘겨 부른다.

**7단계 알려진 결함 — 재사용 전에 고칠 것.** 얼린 칸(거래 불가 보유)이 표류로 종목 상한·업종 밴드를 넘으면 문제가 infeasible 이 되어 O 북이 그 스텝을 멈춘다(B 2022-04-06 k=33 `solver_fail`). 얼린 칸을 분리하면 O 숫자가 바뀌어 코드는 고치지 않았다. **사전등록 정정이 필요하다.** 새 연구는 이 수정, `BookSpec(min_names_after_nan=True, strict_lam_cause=True)` 를 함께 켠다. 둘 다 기본 끔은 B 재현 때문이다. `max_turnover` 는 Σ|Δw|(매수 + 매도 합)이고 편도가 아니다. O 재현은 cvxpy·clarabel 판에 달린다. `uv.lock` 에 없어 새 환경에서 5e-6 안팎 움직였다.

**8단계 비용의 한계.** KOSPI 반스프레드 5bp 는 2017~22 틱 하한 미만이다. 1배는 그 구간에서 낙관이고 2배가 덮는다(B 영향 상한 4.4bp). 2배는 세금까지 두 배다. ADV 는 core `_trailing_adv` 가 관측 1개로도 평균을 내므로 새 연구는 `trailing_adv(min_obs=ADV_MIN_OBS)` 를 쓴다(B 재현은 그대로). core 쪽 메모(이 PR 범위 밖): `costs/impact.py` 의 ADV 가드는 하한뿐이라 원을 다시 ×1e6 한 값이 통과한다. `costs/tax.py` 의 시행일은 2차 출처다.

**core 의존.** 기본 핀 `krx-quant-core==0.7.0` 에 없는 것은 부를 때 import 하고, 없으면 `CoreLayerMissing` 이 설치 방법을 말한다. `trade_costs` 의 충격비용, `o_target` 의 최적화기, 위험모델, `factor_attribution` 이 그렇다. 비용을 끈 순위 북(`Flags(costs=False)`)은 기본 핀에서 돈다. 나머지 시험은 CI `research` 잡이 core `8286e3e` 로 돌리고, 거기서는 건너뛰면 실패다.

## 경계 검사

`swing_it.pipeline.checks` 의 독립 함수다. 어느 하네스든 부를 수 있고, 어기면 `CheckError` 다. 합격 여부는 묻지 않는다.

| 함수 | 거부하는 것 |
|---|---|
| `check_universe(frame, min_per_day=1)` | 중복 (code, date), 날짜 아닌 값·결측, 코드 결측·빈 값, bool 아닌 적격 열, 전체 적격 0개, 적격이 `min_per_day` 미만인 날 |
| `check_panels(prices, earnings, end)` | `end` 뒤 가격, 종가 NaN·inf·0·음수, `end` 뒤에 안 실적 행, `end` ≥ 2026-09-14(애프터마켓 오염 — main 의 `read_prices` 는 정규장 값을 만들지 못한다) |
| `check_signal(sig)` | inf, 전부 NaN |
| `check_forecast(df, horizon, calendar)` | 학습 구간 안 예측, `train_end` 뒤 거래일 `horizon` 일(마지막 라벨일까지의 거래일 수) 안의 예측, 거래일이 아닌 날짜 |
| `check_weights(w, codes)` | 빈 배열, 숫자가 아닌 값, 음수, 합 초과, NaN, 코드 중복 |
| `check_attribution(frame, names)` | 기여 합과 total 의 차이 |

**보증하지 않는 것.** `check_panels` 는 알게 된 날 규칙을 모른다. 실적을 쓸 수 있는 첫날은 max(`avail_date`, 정기보고서 첫 접수일) 다음 거래일이고, 신호를 `yoy_strict_wide` 로 만들어야 지켜진다.

## 게이트 규칙

원장은 연구마다 `research/logs/<study>/PIPELINE.jsonl` 이다. 코드만 덧붙인다. 한 줄이 단계 사건 하나이고, 앞 줄과 해시로 묶인다.

- `require_stage(study, stage)` 는 앞 단계가 모두 pass·skip 이어야 통과한다. 아니면 `StageGateError` 다.
- **fail** 은 그 단계의 새 pass 전까지 뒤를 전부 막는다. skip 으로 덮을 수 없고, 같은 증거·같은 검증 문서로 뒤집을 수 없다.
- **skip** 은 4·5·6·7·10·11 만 할 수 있다. 1·2·3·8·9 는 없으면 판정이 거짓이 되므로 건너뛸 수 없다. **4단계 skip** 은 왜 건너뛰는지 적은 증거 파일이 필수이고, 앞 단계(신호)를 다시 적으면 stale 이 된다.
- **pass** 는 증거 파일이 필수이고, 파일의 sha256 을 줄에 적는다.
- **pass·skip·fail 모두 사유가 필수다.**
- **4·5·9 의 pass** 는 검증 문서(`*_verified.md`·`REVIEW_*.md`) 참조가 필수다.
- **10 의 pass** 는 사용자 승인 기록(날짜·사용자의 말 그대로·만료일·금액 상한, 필요하면 바구니 해시)과 전진 페이퍼 결과 증거가 필수다. 승인일은 9단계 pass 날 이후이고, 적는 날에서 14일 안이고, 만료는 승인일에서 92일 안이다. 소급 기록으로 적을 수 없다. 주문 경로는 `require_stage` 가 아니라 `require_execution_approval(study, amount_krw=…, basket_hash=…)` 을 부른다. 만료·금액·바구니가 어긋나면 막는다. 같은 승인을 그 범위 안에서 여러 번 쓰는 것은 막지 않는다 — 주문 원장이 없다.
- **stale** 은 뒤를 막는다. 앞 단계가 더 나중에 다시 적혔거나, 증거 파일이 없거나, 해시가 어긋난 경우다.
- 손으로 잘못 고친 원장은 건너뛰지 않고 막는다. **위조는 못 막는다.** 해시 체인에 키가 없어 체인을 맞춰 덧붙인 줄·날짜를 바꾼 `ts` 는 통과하고, 꼬리 삭제는 CI 의 `origin/main` 앞부분 대조로 머지된 줄만 잡는다. 실수와 망각을 막는 장치이지 보안 장치가 아니다.
- 소급 기록(`backfill`)은 2026-11-02 까지 적은 줄만 받는다. 그 뒤에 옛 날짜로 적을 수 없다.

pass·skip 은 앞 단계가 열려 있어야 적힌다. fail 은 언제든 적힌다.

**게이트가 막지 못하는 주문.** 게이트는 부른 코드만 막는다.

- main 에는 주문 경로가 없다. **fcs 브랜치에는 있다** — `src/swing_it/live/brokers.py`(`KiwoomBroker`)와 `cli.py`(`sw-live --execute --confirm-real`). 게이트를 부르지 않는다. 크론에는 없다. fcs 를 머지하면 린트 (l) 이 빨개진다.
- scalp-it·daytrade-it 의 주문 코드는 다른 레포라 이 게이트와 린트 밖이다.
- 판정 전 1주 실주문 파일럿은 **지금 합법 경로가 없다.** 10단계 pass 는 9단계 pass 를 요구하고, 전진 판정은 2027-10 무렵이다. 파일럿 전용 승인을 만들지는 사용자 결정이다(`docs/execution/DESIGN.md` §7).

**전진 중인 페이퍼 북.** 전진 판정은 1년 뒤다. 그동안 볼 수 있어야 하므로 두 가지를 정했다.

- 전진 일일 러너는 **9단계로 선언한다.** 전진 자체가 시험이고, 판정 전까지 9단계는 pending 이다.
- 기여·모니터링(**11단계**)은 9·10 이 pending 이어도 열린다. 1~8 은 열려 있어야 하고, 어디든 fail·stale 이면 막힌다. 10단계는 9단계 판정 없이 열리지 않는다.

**CI** 는 `scripts/check_guardrails.py` 규칙 (k) 로 다음을 본다.

- **선언.** `research/experiments`·`research/paper` 의 파일 전부, 그 밖의 `research/**`·`scripts/**` 는 데이터 읽기(`read_prices`·`read_earnings`·`read_sql*` 등)나 백테스트(`simulate`·`*backtest*`·`prop_gate` 등)를 부르면 선언해야 한다. 2026-11-02 면제 목록에 있는 파일만 빠진다. 목록은 줄어들기만 하고 테스트가 고정한다.
- **게이트 위치.** import 때(모듈 수준·기본 인자·데코레이터) 데이터를 읽으면 위반이다. 진입점에서 게이트 전에 올 수 있는 호출은 인자 파싱과 시각·로깅 준비뿐이다. import 별칭을 풀고, 지역 함수는 이름과 상관없이 따라간다.
- **로그 폴더.** `research/logs/<폴더>` 는 면제가 아니면 `PIPELINE.jsonl` 이 있어야 한다.
- **원장.** 체인·행 규칙·최신 증거 해시, 그리고 `origin/main` 의 원장이 지금 원장의 앞부분인지. 되돌아가기로 생긴 stale 은 정상이라 빨강이 아니다.
- **가드 자체.** `scripts/check_pipeline_mutations.py` 가 `tests/pipeline_mutations.toml` 의 변이를 넣어 시험이 정말 빨개지는지 본다.

## 새 연구를 더하는 법

`examples/pipeline_rank_book.py` 가 합성 데이터로 끝까지 도는 예다. 선언하고, 데이터를 읽기 전에 게이트를 부르고, 단계마다 위 표의 함수를 부른다. 게이트 앞에는 인자 파싱만 온다. 조건문 안 호출은 호출로 치지 않는다.

```python
from swing_it.book.accounting import check_invariants, simulate
from swing_it.book.metrics import book_summary
from swing_it.book.spec import BookSpec
from swing_it.pipeline import Stage, require_stage

PIPELINE = ("my_study", Stage.BACKTEST)
SPEC = BookSpec(top_n=5)

def main() -> int:
    args = parse_args()          # argparse 만
    require_stage(*PIPELINE)     # 앞 단계 증거가 없으면 여기서 멈춘다
    ctx = load_my_panels(args)
    recs = simulate(ctx, "R", spec=SPEC)
    ...

if __name__ == "__main__":
    raise SystemExit(main())
```

위험모델을 안 쓰는 순위 북은 `models_from_mask` 로 종목 집합만 넘긴다. 러너가 아닌 파일은 `PIPELINE_LIBRARY = "<무엇인지>"` 를 적는다. 그런 파일에는 진입점이 없어야 한다. 백테스트 함수를 부르는 러너는 9단계 앞으로 선언할 수 없다.

단계를 끝내면 적는다. `record` 는 어느 원장에 적었는지 출력하고, 지금 폴더가 다른 레포면 거부한다.

```bash
python -m swing_it.pipeline record my_study 1 pass --evidence research/logs/my_study/U.md --reason "…"
python -m swing_it.pipeline record my_study 4 pass --evidence research/logs/my_study/S.md \
    --reviewer research/logs/my_study/S_verified.md --reason "…"
python -m swing_it.pipeline record my_study 6 skip --reason "순위 북 — 위험모델 안 씀"
python -m swing_it.pipeline status my_study
python -m swing_it.pipeline verify my_study
```

## 되돌아가는 법

앞 단계를 다시 적으면 된다. 금지하지 않는다.

```bash
python -m swing_it.pipeline record my_study 3 pass --evidence research/logs/my_study/SIG2.md --reason "신호 재정의"
```

그 뒤 단계의 pass 와 4단계 skip 은 stale 이 되고, 순서대로 다시 증거를 대야 한다. 다른 skip·fail 은 그대로 남는다. 증거 파일을 고친 경우도 같다. 고친 파일로 그 단계를 다시 적는다.

원장이 머지에서 충돌하면 기준 브랜치 쪽 줄을 그대로 두고, 자기 줄은 `record` 로 다시 적는다. 체인이 이어져야 하고, CI 가 기준 줄이 앞에 그대로 있는지 본다.

## 일부러 강제하지 않는 것

- **증거의 질.** `README.md` 도 증거로 적을 수 있다. 다른 연구의 검증 문서도 참조할 수 있다. 더 조이지 않는다 — 공개 커밋에 이름이 남는 것이 비용이고, 나머지는 개인 연구의 자유다.
- **판정의 옳음.** 게이트는 순서와 해시만 본다. 검증 문서의 내용은 읽지 않는다. 거짓 pass 는 리뷰가 잡는다.
- **사용자 승인의 진위.** 날짜와 인용문의 존재만 본다.
- **시장 중립·벤치마크 의무.** 기관이 아니다. 순위 북은 6·7단계를 건너뛰어도 된다.
- **새 이름으로 다시 시작하기.** fail 을 피하려고 연구 이름을 바꾸는 것은 코드로 못 막는다. 다중검정 원장과 리뷰가 잡는다.
- **`data_fingerprint`.** 줄에 적을 뿐 다시 계산하지 않는다. 데이터가 바뀌어도 stale 이 되지 않는다.

## 린트가 못 잡는 것

`tests/test_check_guardrails.py::test_rule_k_known_limits_are_the_documented_ones` 가 이 목록을 고정한다.

- **호출이 아닌 접근.** `df = m.CACHE` 처럼 다른 모듈이 import 때 읽어 둔 값을 속성으로 가져오는 경우.
- **다른 모듈 안에서 읽는 경우.** 이름이 `parse_args` 인 외부 함수, `m.load_everything()` 처럼 이름으로 데이터 읽기를 알 수 없는 모듈 수준 호출. 지역 함수만 따라간다.
- **동적 호출.** `getattr`·`importlib`·`exec`.
- **커서로 직접 읽기.** `cur.execute("SELECT …")` 는 데이터 호출로 치지 않는다. 가격·실적은 규칙 (d)(g) 가 따로 본다.
- **선언한 단계의 진실성.** 백테스트 호출만 기계로 본다. 내 엔진을 부르면서 1단계로 적는 것은 리뷰가 잡는다.
- **면제 파일의 내용 변경, `examples/`.**
- **체인을 맞춘 위조.** 규칙을 지키면서 코드로 덧붙인 거짓 줄.

## 원장 문구 정정

원장은 고치지 않는다. 소급 기록의 사유 가운데 원문보다 넓게 읽히는 곳을 여기 적는다.

- `pead_optimizer` 2단계의 "c0 재현 66개 항목"·"PO-4 구창 재현"은 **수리 전** 표에서 한 것이다. 수리된 표의 확인은 `A1_repaired`·`C0_repaired` 다.
- `pead_forward` 7단계의 "시장 중립 의무는 없다"는 사전등록 §2.2 에 없는 말이다. 이 레포의 결정이다.
