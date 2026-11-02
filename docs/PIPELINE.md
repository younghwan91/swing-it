# 파이프라인 11단계

연구는 아래 순서로만 흐른다. 단계마다 모듈과 입출력 계약이 있고, 순서는 게이트와 CI 가 막는다. 지금 상태는 `python -m swing_it.pipeline status` 로 본다.

## 단계와 계약

코드는 `src/swing_it/pipeline/` 에 있다. 모듈은 기존 구현을 감싸기만 한다. 진입점은 모두 `run(study, ...)` 이고 첫 문장이 게이트다.

| # | 단계 | 모듈 | 받는 것 | 내는 것 | 위임 대상 |
|---|---|---|---|---|---|
| 1 | 유니버스 | `s01_universe` | 시총·ADV 패널 | `UniverseMask` | `features.universe.smallmid_universe` |
| 2 | 데이터 | `s02_data` | `UniverseMask`, DB 연결 | `DataPanels` | `storage.read_prices`·`read_earnings` |
| 3 | 알파 발굴 | `s03_alpha` | `DataPanels` | `Signal` | `features.fundamentals.earnings_yoy_panel` |
| 4 | 알파 검증(IC/ICIR) | `s04_alpha_validation` | `Signal`, `DataPanels` | `ICReport` | core `rank_ic` |
| 5 | ML 수익률 예측 | `s05_forecast` | `Signal`, `DataPanels` | `Forecast` | 구현 없음(`NotImplementedError`) |
| 6 | 팩터 위험모델 | `s06_risk_model` | 수익·노출 패널 | `RiskModelOutput` | core `build_risk_model` * |
| 7 | 포트폴리오 최적화 | `s07_portfolio` | 기대수익, `RiskModelOutput` | `TargetWeights` | core `optimize` * |
| 8 | 거래비용 모델 | `s08_costs` | `TargetWeights`, 직전 비중 | `CostEstimate` | core `trade_cost_bp` * |
| 9 | 백테스트 | `s09_backtest` | `Signal`, `DataPanels`, config | `BacktestResult` | `strategies.pead.staggered_backtest` + 다중검정 원장 |
| 10 | 주문 실행 | `s10_execution` | `TargetWeights` | `OrderPlan` | 구현 없음(`docs/execution/DESIGN.md`) |
| 11 | 성과 기여·모니터링 | `s11_attribution` | 비중·노출·팩터·특이 수익 | `Attribution` | core `factor_attribution` * |

\* swing-it 기본 핀 `krx-quant-core==0.7.0` 에 없다. 부를 때 import 하고, 없으면 `CoreLayerMissing` 이 설치 방법을 말한다.

**계약이 막는 것** (`contracts.py`, 어기면 `ContractError`)

- `DataPanels`: `end` 뒤에 안 값(`knowledge_date`·`avail_date`)과 미래 가격을 거부한다. `end` 가 2026-09-14 이후면 애프터마켓 처리를 명시해야 한다.
- `Forecast`: 학습 구간 안 예측을 거부한다.
- `TargetWeights`: 음수 비중과 합 1 초과를 거부한다.
- `BacktestResult`: 비용 기준(`gross`·`net_1x`·`net_2x`)과 원장의 시행 수가 있어야 한다.
- `OrderPlan`: `dry_run=True` 만 받는다.
- 어느 계약에도 합격 필드는 없다. 사람이 읽고 원장에 적는다(GUARDRAILS §8).

연구 하네스(`research/experiments/pead_optimizer_b.py` 등)는 옮기지 않았다. 옮기면 바이트 동일 재현이 깨지고, `src` 는 `research` 를 import 하지 않는다. 각 모듈의 `REFERENCES` 가 경로로 가리킨다.

## 게이트 규칙

원장은 연구마다 `research/logs/<study>/PIPELINE.jsonl` 이다. 코드만 덧붙인다. 한 줄이 단계 사건 하나이고, 앞 줄과 해시로 묶인다.

- `require_stage(study, stage)` 는 앞 단계가 모두 pass·skip 이어야 통과한다. 아니면 `StageGateError` 다.
- **fail** 은 그 단계의 새 pass 전까지 뒤를 전부 막는다. 사유가 필수다. skip 으로 덮을 수 없고, 같은 증거·같은 검증 문서로 뒤집을 수 없다.
- **skip** 은 사유가 필수다. 4·5·6·7·10·11 만 건너뛸 수 있다. 1·2·3·8·9 는 없으면 판정이 거짓이 되므로 건너뛸 수 없다.
- **pass** 는 증거 파일이 필수이고, 파일의 sha256 을 줄에 적는다.
- **4·5·9 의 pass** 는 검증 문서(`*_verified.md`·`REVIEW_*.md`) 참조가 필수다.
- **10 의 pass** 는 사용자 승인 기록(날짜 + 사용자의 말 그대로)과 전진 페이퍼 결과 증거가 필수다. 소급 기록으로 적을 수 없다. 주문 경로는 `require_stage` 가 아니라 `require_execution_approval` 을 부른다.
- **stale** 은 뒤를 막는다. 앞 단계가 더 나중에 다시 적혔거나, 증거 파일이 없거나, 해시가 어긋난 경우다.
- 손으로 고친 원장은 건너뛰지 않고 막는다.

pass·skip 은 앞 단계가 열려 있어야 적힌다. fail 은 언제든 적힌다.

**CI** 는 `scripts/check_guardrails.py` 규칙 (k) 로 두 가지를 본다.

- `research/experiments/*.py`·`research/paper/*.py` 의 선언. 2026-11-02 면제 목록(`PIPELINE_GRANDFATHER`·`PIPELINE_GRANDFATHER_FCS`)에 있는 파일만 빠진다. 목록은 줄어들기만 하고 테스트가 고정한다.
- 원장의 체인·행 규칙·최신 증거 해시. 되돌아가기로 생긴 stale 은 정상이라 빨강이 아니다.

## 새 연구를 더하는 법

러너 머리에 선언하고, 데이터를 읽기 전에 게이트를 부른다. 앞에는 인자 파싱만 온다. 조건문 안 호출은 호출로 치지 않는다.

```python
from swing_it.pipeline import Stage, require_stage

PIPELINE = ("my_study", Stage.BACKTEST)

def main() -> int:
    args = parse_args()          # argparse 만
    require_stage(*PIPELINE)     # 앞 단계 증거가 없으면 여기서 멈춘다
    ...
```

러너가 아닌 파일은 `PIPELINE_LIBRARY = "<무엇인지>"` 를 적는다. 그런 파일에는 `__main__` 진입점이 없어야 한다. 백테스트 함수를 부르는 러너는 9단계 앞으로 선언할 수 없다.

단계를 끝내면 적는다.

```bash
python -m swing_it.pipeline record my_study 1 pass --evidence research/logs/my_study/U.md --reason "…"
python -m swing_it.pipeline record my_study 4 pass --evidence research/logs/my_study/S.md \
    --reviewer research/logs/my_study/S_verified.md
python -m swing_it.pipeline record my_study 6 skip --reason "순위 북 — 위험모델 안 씀"
python -m swing_it.pipeline status my_study
python -m swing_it.pipeline verify my_study
```

## 되돌아가는 법

앞 단계를 다시 적으면 된다. 금지하지 않는다.

```bash
python -m swing_it.pipeline record my_study 3 pass --evidence research/logs/my_study/SIG2.md
```

그 뒤 단계의 pass 는 stale 이 되고, 순서대로 다시 증거를 대야 한다. skip·fail 은 그대로 남는다. 증거 파일을 고친 경우도 같다. 고친 파일로 그 단계를 다시 적는다.

## 일부러 강제하지 않는 것

- **판정의 옳음.** 게이트는 순서와 해시만 본다. 검증 문서의 내용은 읽지 않는다. 거짓 pass 는 리뷰가 잡는다.
- **사용자 승인의 진위.** 날짜와 인용문의 존재만 본다.
- **시장 중립·벤치마크 의무.** 기관이 아니다. 순위 북은 6·7단계를 건너뛰어도 된다.
- **새 이름으로 다시 시작하기.** fail 을 피하려고 연구 이름을 바꾸는 것은 코드로 못 막는다. 다중검정 원장과 리뷰가 잡는다.
- **선언한 단계의 진실성.** 백테스트 호출만 기계로 본다. 그 밖의 잘못된 단계 선언은 리뷰가 잡는다.
- **원장 끝 줄 잘라내기.** 체인은 못 잡는다. 지금 있는 줄은 `tests/test_pipeline_ledgers.py` 가 고정하고, 그 뒤는 git diff 로 본다.
- **면제 파일의 내용 변경, `scripts/`·`examples/`.** 규칙 밖이다.
