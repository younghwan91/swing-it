# 스윙 라이브 — 수렴 × 수급 × 실적 × 미너비니 청산 (설계, 2026-10-10)

사용자 지시(2026-10-10): "이걸 기반으로 매매 알고리즘을 라이브 트레이딩으로 짜라. 다른 트레이더 레포·코어 레포를 참조하라. 백테스트도 하라."
로드맵 §4 의 "실주문 경로는 사용자 지시 전엔 만들지 않는다" 는 이 지시로 풀린다. 다만 **기본은 페이퍼**다 — 이 전략은 아직 표본 밖 판정이 없다
(`research/logs/sepa_explore/CONVERGENCE.md` 실행 중). 실계좌는 `--execute --confirm-real` 두 플래그를 사람이 직접 줄 때만 나간다(daytrade-it 패턴).
크론은 이 문서 단계에서 설치하지 않는다(워크트리는 cron-install 대상도 아니다).

## 1. 원칙

- **백테스트 = 라이브 = 같은 코드.** 진입 판정·주문 계획·포지션 상태 전이는 `swing_it.live.strategy` 하나에만 있다. 포트폴리오 백테스트와 데몬이
  둘 다 그것을 부른다. 연구 시뮬레이터(`research/signals/minervini_exits.simulate_minervini`)와는 **패리티 테스트**로 묶는다(자본 무제한·겹침 금지에서
  트레이드별 진입·청산·수익이 일치).
- 주문은 지정가만(코어 검증 범위). 손절·익절은 클라이언트 감시(서버 스톱 없음). 매도는 킬스위치·가드가 막지 않는다(코어 OMS 규약).
- 상태는 재시작에 살아남는다: 포지션 장부(`PositionBook` JSONL) + 전략 원장(JSONL: 피벗·손절·목표·진입일·절반 익절 여부).

## 2. 전략 v0 (설정 `src/swing_it/live/config.py` — 사전등록 CONVERGENCE A 와 같은 정의)

- 유니버스: 위생 통과 ∧ **ADV20 ≥ 5,000 백만원(50억)**(두 구간 합).
- 진입 후보(신호일 d, d 종가까지): 추세 템플릿 ∧ RS ≥ 70 ∧ 피벗 돌파(직전 50일 고가) ∧ 거래량 ≥ 1.5×50일 ∧ breadth(시장) > 0.5 ∧ PEAD 순위 ≥ 0.70
  ∧ 상대 수렴 rel ≤ 1.0 ∧ 수급 F = 1.
- 주문 계획: 진입 지정가 = 호가 반올림(min(피벗 × 1.05, 상한가 − 1틱)) — 장전 동시호가에 내면 "시가 ≤ 지정가일 때만 시가 체결" = 백테스트 추격 금지.
  손절가 = 진입가 × (1 − d), d = clip(1 − 수렴저점×0.99/진입가, 3%, 10%) — 진입가는 체결 후 확정되므로 손절·목표는 **체결 시** 계산.
  시가 ≤ 수렴 저점이면 체결 직후 그날 청산(백테스트의 skip_below_low 와 같은 결과가 되도록 — 패리티 테스트로 확인할 차이는 문서화).
- 청산: 장중 저가 ≤ 손절 → 손절(갭이면 시가) / 고가 ≥ 진입×(1+2d) → 절반 익절 + 나머지 손절 본전 / 종가 < MA50 → 다음날 시가 매도 /
  20거래일째 종가 ≤ 진입가 & 익절 전 → 그날 종가(라이브: 15:20 종가 동시호가 지정가) / 120거래일 종가.
- 사이징: 주수 = floor(min(위험 1.0% × 자산 / (진입가 × d_est), 10% × 자산 / 진입가)), d_est = 신호일 기준 예상 손절폭. 최대 10종목.
  후보가 빈 슬롯보다 많으면 rel 오름차순(더 강한 수렴) → RS 내림차순.
- 위험 한도: 일 손실 한도(코어 KillSwitch) + **다일 낙폭 킬**(자산 고점 대비 −15% 에서 신규 진입 정지, 파일 `data/swing_STOP` 생성).

## 3. 모듈 (전부 `src/swing_it/live/`, 연구 코드 import 금지 — 경계 규칙)

| 파일 | 책임 |
|---|---|
| `config.py` | `StrategyConfig`(위 수치 전부, frozen dataclass), `RiskConfig` |
| `strategy.py` | `scan(panels, d, cfg) -> list[Candidate]`, `plan_entry(c, equity, prev_close, cfg) -> EntryPlan`, `PositionState`(상태 전이: `on_fill_open(price)`, `on_bar(open, high, low, close, ma50, day_no) -> list[Action]`, `on_quote(last) -> list[Action]`), `Action(kind: stop/partial/breakeven/trail_next_open/time_close/max_hold, qty_frac, price)` |
| `features.py` | 신호 피처를 DB 에서 그날치로 계산(`swing_it.features.*` 재사용 — minervini·earnings_leg·flow_leg·korea, rel) |
| `sizing.py` | 위 사이징, 슬롯·현금 제약 |
| `backtest.py` | 포트폴리오 백테스트: 일 루프, 현금·슬롯, 동시호가 체결 모형, 장중 손절/익절은 일봉 고저(같은 날 둘 다면 손절 먼저), 비용 `korea.cost_rt` |
| `state.py` | 전략 원장 JSONL + PositionBook 저널, 재시작 복원, 대사(`oms.reconcile`) |
| `runtime.py` | 작업: `signal`(07:35, `daily_finalize` 완료·최신일 확인) / `preopen`(08:30~08:55 지정가 매수·전일 결정 매도) / `open_check`(09:01 시가 확인·미체결 취소) / `monitor`(09:00~15:20 60초 REST 폴링, 손절·익절) / `close`(15:20 종가 동시호가 시간 손절) / `eod`(15:40 MA50·보유일 갱신, 다음날 매도 결정, 리포트) |
| `brokers.py` | 코어 `PaperBroker`(시가 체결 에뮬: 09:00 시가로 Quote 주입) / `KiwoomBroker(dry_run=...)` 배선, `OrderGuardConfig(price_band_pct≥0.06, max_qty_hard_cap 자산 기준)` |
| `cli.py` | `sw-live {signal,preopen,open-check,monitor,close,eod,status,reconcile} [--paper(기본)|--dry-run|--execute --confirm-real]` |

## 4. 운영 (설치하지 않음 — 사용자 결정)

simnode 메인 클론 `deploy/crontab.simnode` 에 넣을 줄을 문서로만 둔다. 키움 겹침 시간(06:30~07:30, 10:00~10:15, 16:00~16:50) 회피, 앱키 1세션
(웹소켓 안 씀, REST 폴링). 알림은 quant-airflow `collectors/alert.py`(Discord) 패턴. 계좌는 다른 레포와 공유 → 아침 보유 기준선, `own_orders_only=True`.

## 5. 검증

- 단위 테스트: 상태 전이 전 분기, 사이징, 동시호가 체결, 재시작 복원, 킬스위치·낙폭 킬, 가드.
- **패리티 테스트**: 같은 사건에서 `backtest`(자본 무제한) 트레이드 = `simulate_minervini` 트레이드.
- 페이크 브로커로 하루 전체 리허설(signal → preopen → open-check → monitor → close → eod) 통합 테스트.
- 독립 검증 에이전트가 이 코드도 감사한다(`research/logs/sepa_explore/AUDIT.md`).
