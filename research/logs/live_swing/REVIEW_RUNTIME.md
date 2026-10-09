# 스윙 라이브 런타임 검토 (REVIEW_RUNTIME) — 2026-10-10

검토 대상(워킹트리, 미커밋): `src/swing_it/live/{state,brokers,runtime,cli}.py` (L1·L4·L5 반영본, runtime.py 1378행),
`tests/test_live_{state,brokers,runtime}.py`(58 통과), `docs/live-swing-runbook.md`, `deploy/crontab.swing-live.example`.
방법: 코드 읽기 + 코어(`krx_quant_core.execution.*`, `risk.killswitch`) 대조 + **공격 테스트 11개**
(`scratchpad/review_runtime/test_adversarial.py`, 전부 재현됨 = 11 passed) + `minute_bars` 실측(손절 미끄러짐·가격 밴드).
키움 호출·주문·크론 설치 없음. 파일 수정 없음(이 문서만 생성).

**결론: PAPER 운용 GO(조건부).** 아래 M3·M4·M7 세 가지는 페이퍼에서도 조용히 하루를 날리는 모드라 **첫 주 운용 전 수정 권고**.
CRITICAL 두 건은 실계좌(live)·dry-run 경로에만 해당하며 페이퍼 판정에는 영향이 없다 — 다만 §6 "실계좌 전환" 전에 반드시 고쳐야 한다.

---

## 1. 안전(SAFETY)

### 실주문 경로 열거 — 두 플래그 없이는 못 연다 (확인)

`KiwoomBroker(dry_run=False)` 가 만들어지는 유일한 경로: `brokers.build_bundle` 413행 `dry_run=(mode != "live")`, 그리고
`(mode == "live") != bool(confirm_real)` 이면 `ValueError`(393–394). `mode="live"` 는 `cli.resolve_mode`(48–63) 가
`--execute and --confirm-real` 일 때만 돌려주고, 환경변수·설정 파일·스냅샷 `mode` 필드로는 못 바꾼다(스냅샷 모드는 **거부용** 이중 확인,
`state.StateStore.load` 292–293). `--ignore-clock` 은 CLI(57–58)와 `SwingRuntime.__init__`(334–335)에서 페이퍼 전용으로 막힌다.
`SWING_KIWOOM_IS_MOCK` 는 서버 선택뿐(`make_kiwoom_api` 370). 크론 예시는 전부 `--paper`. 우회 경로 없음. **PASS.**

### 가드 수학 (확인)

`entry_guard_config`(brokers 311–332): `max_qty = max_qty_hard_cap = ⌊자산×0.12/지정가⌋`, 밴드 6%, 총 20건. 자산은 전략 자본
기준(`_sizing_equity`), 가격은 지정가 ≥ 1원이라 상한이 터무니없어질 입력이 없다. 그 위 `_submit_entry` 1104–1107 명목 상한
(자산×10%×1.07). **PASS.** 다만:

- **m1** 가드의 일일 횟수 카운터는 `OrderGuard` 인스턴스(프로세스) 안에만 있다(`_make_ctx` 446). 크론 = 작업마다 새 프로세스라
  `max_orders_total=20` 은 preopen 한 번 안에서만 유효. 멱등성은 상태(`planned→entry_pending`)가 지키므로 실해는 없지만 "일일 20건" 문구는 사실이 아니다.
- **m2** 밴드 6% vs 지정가: 지정가 = min(피벗×1.05, 상한가−1틱) 이고 돌파일 종가 > 피벗이라 **위쪽 밴드는 절대 안 걸리고** 아래쪽만 걸린다.
  `daily_bars` 2024-01~2026-09 원시 돌파 사건 9,853건: 종가/피벗 > 1.05(지정가 < 전일종가 → 갭하락에만 체결) **41.4%**, > 1.117
  (지정가 < 전일종가×0.94 → **가드가 거부**) **15.3%**, > 1.167(10% 밴드) 7.7%. 백테스트도 그 사건은 11.7% 갭하락이 아니면 미체결이라
  패리티엔 무해하지만, 15% 의 후보가 `order_blocked` 로 끝난다 — 계획 단계에서 미리 `skipped(limit_below_band)` 로 떨구거나 밴드를 12% 로 두는 편이 로그가 깨끗하다.

### CRITICAL (live·dry-run 경로 — 페이퍼엔 해당 없음)

**C1. 주문 POST 와 스냅샷 저장 사이 크래시 → 같은 매수를 두 번 낸다.** `_submit_entry` 1112 `oms.buy` 가 POST 한 뒤 상태 전이
(1115–1117)는 메모리에서만 일어나고 `state.json` 은 `_run_locked` finally(481–484)에서 한 번 저장된다. SIGKILL·전원·OOM 이면 다음
실행에서 같은 종목이 아직 `planned` 라 다시 산다. 사건 원장(`store.event` 583)은 fsync 됐지만 복원엔 쓰지 않는다(state.py 11–12).
브로커 미체결(`open_orders`)로 "내 주문이 이미 걸려 있나" 확인하는 단계도 없다. 재현: `test_crash_between_post_and_state_save_duplicates_buy`
— 브로커에 같은 종목 매수 2건. 매도는 코어 `OrderManager._pending_sell_qty`(브로커 미체결 합산)가 2중 매도를 막지만 매수에는 그런 벽이 없다.
→ 고침: POST **전에** `entry_pending`+`oid` 를 저장(intent-log)하고, 시작 시 `open_orders()` 의 자기 종목 매수를 `own_ord_nos` 와 대조.

**C2. 취소-체결 경합 + 잔고 조회 지연 → 사용자 주식을 판다 / 전략 주식을 사용자 몫으로 넘긴다.** `_cancel` 600–606 은 취소가
거부되면(이미 체결) 잔고를 **한 번** 다시 읽고(`_fills_from_holdings`), 아직 반영이 안 됐으면 `canceled` 로 표시한다. 그 직후
`_work_exits` 876–888 은 같은 수량을 다시 낸다(코어 `_pending_sell_qty` 는 브로커에서 사라진 체결 주문을 못 본다). 사용자가 같은
종목을 들고 있으면 두 번째 매도가 **사용자 주식**으로 체결된다. 재현: `test_cancel_reject_with_lagging_holdings_resells_user_shares`
— 매도 2건 POST, 사용자 100주 소진, 경보 문구는 "계좌에 전략 몫이 모자람"(오진). 진입 쪽 대칭: 09:02 취소 거부 + 잔고 지연 →
`missed` 처리 → 뒤늦게 보이는 +100 은 L4 규칙으로 **사용자 몫**이 돼 영영 안 판다(`test_entry_cancel_reject_with_lagging_holdings_loses_position_to_user_bucket`,
경보는 warn 하나). kt00018 반영 지연이 몇 초라도 있으면 매 재가격(60초)마다 창이 열린다.
→ 고침: 취소 거부(rc≠0, "이미 체결" 계열)는 그 주문을 `fill_pending` 상태로 두고 **잔고가 설명할 때까지** 재발주 금지; 진입은 `missed`
전에 N초/2회 재조회. 근본적으로는 런북 §6 가 말하는 대로 `ka10075/ka10076` 검증 → 주문번호 기준 체결.

### MAJOR

**M1. 거부·차단된 매도가 조용하다 + 매 틱 재발주.** `_record_order` 584–586 은 `unknown` 에만 경보, `_submit_sell` 628 은 info
로그뿐. 거부된 손절은 `exit_intent` 가 남아 60초마다 새 주문을 POST 한다(`test_rejected_sell_is_silent_and_respammed_every_tick`:
5틱 = 5건 거부, error/warn 경보 0). 하루면 ~370건. 포지션은 "열림"으로 평가액에 계속 잡힌다.
→ `blocked/rejected` 매도 1회째 error 경보, 3회 연속이면 `exit_intent` 를 `needs_human` 으로 바꾸고 재발주 중단.

**M2. 같은 종목에 걸린 사용자 미체결 매도가 전략 손절을 막는다.** 코어 `OrderManager._pending_sell_qty` 는 `broker.open_orders()`
전부를 합산한다(`own_orders` 로 거르지 않음). 사용자가 같은 종목에 매도 주문을 걸어 두면 가용 = 장부 − 사용자 잔량 → "청산 수량이
보유 초과" 차단(`test_user_pending_sell_on_same_code_blocks_strategy_stop`). M1 과 합쳐져 **손절이 조용히 안 나간다.**
→ 런타임이 `oms.sell` 전에 예약 수량을 자기 `live_orders` 로만 계산하도록 코어 옵션 추가 요청, 또는 코어에 `own_orders` 필터.

**M3. 09:02 가짜 휴장 판정이 하루 종일 고정된다 (페이퍼 포함).** `_session_open` 506–513: `ka10081` 최신 봉 < 오늘이면
`sessions[today]="closed"` + 오늘 주문 전부 취소 + 계획 만료. 이후 모든 작업은 492–494 에서 그 값을 믿고 **아무것도 안 한다** —
보유 포지션이 손절가 아래로 가도 monitor·close 가 침묵(`test_false_holiday_at_open_check_is_sticky_and_leaves_positions_unmonitored`).
근거가 "quant-airflow 가 장중에 오늘 봉을 봤다"(2026-08-24, 10:05 관측)뿐이라 09:02 에 봉이 아직 없을 가능성을 배제 못 한다.
→ 달력(`holidays.txt`) 과 `ka10081` 이 **둘 다** 휴장이라야 취소; 하나만이면 경보 + 09:30 재확인. `closed` 판정도 그날 다음 작업에서 재검증.

**M4. `ka10001 open_pric` 이 없으면 모든 진입이 조용히 `missed`.** `_job_open_check` 1129–1130 은 `q.open` 이 None 이면 단일가를
먹이지 않고 넘어가 1136–1151 에서 장전 주문 취소 → `missed(open None)`. 경보 없음(`test_missing_open_price_marks_all_entries_missed_without_alert`).
`open_pric`·`base_pric` 은 미검증 필드(brokers 31–35). 페이퍼 첫날 이 상태면 **2주 페이퍼가 통째로 "진입 0"** 으로 끝나고 아무도 모른다.
→ `open is None` 이면 error 경보 + 09:02 `cur_prc` 를 대용(경고 표기)하거나 09:05 재시도. 첫 페이퍼 실행 전 §6-4 필드 확인을 **페이퍼 조건**으로 올릴 것.

**M5. signal 하루 결손 = 추적 청산 결손 (L5 결합).** `_decide_on_db_close` 1282–1315 는 배치의 `d` 종가 하나만 본다. D2 signal 이
안 돌면(daily_finalize 지연 → `_abort_signal` 950–959 가 1282 **앞에서** return, 크론 미실행, 시간창 밖 rc 2) D1 종가 < MA50 결정은
영영 평가되지 않고 D3 은 D2 종가만 본다(`test_missed_signal_job_skips_trail_exit`: 연구는 D2 시가 매도, 라이브는 보유 지속). 같은 날
자산 행·낙폭 킬·일 손실 킬도 평가되지 않는다. 런북 136행 "결과는 같다" 는 결손 없는 날에만 참.
→ `last_eod` 이후 **빠진 거래일 전부**를 DB 종가로 순서대로 `on_close` (배치에 `trail_closes` 가 이미 50개 있으므로 가능); abort 시에도
보유 종목 결정만은 수행(abort 사유가 "오늘 봉 포함" 이 아니면 d 는 쓸 수 있다).

**M6. 부분 체결 + 사용자 같은 날 매수 → 과잉 귀속 → 사용자 주식 매도.** L4 는 "살아 있는 주문의 잔량까지" 배정하므로 단일가 60주
부분 체결 + 사용자 50주 매수(+110)는 전략 주문에 100 이 가고 10 만 사용자 몫이 된다. 손절 때 100주를 팔아 사용자 40주가 함께 나간다
(`test_partial_fill_plus_user_buy_same_day_over_attributes`). 잔고 차이 방식의 본질적 한계 — 런북 69–72 행의 "구분 불가" 목록에
이 경우(과잉 귀속)도 추가하고, 실계좌 전제 조건(주문번호 체결)으로 못박을 것.

**M7. 크론 지연 → 시간창 밖 → 종료 코드 2 뿐, 경보 없음 (페이퍼 포함).** `_check_window` 366–374 / `run` 391–394 는 로그만.
크론 예시는 `--ignore-clock` 을 안 주므로 페이퍼도 시간창을 강제한다(의도대로) — preopen 이 09:00:00 에 뜨면 그날 진입 0 이 조용히 지나간다
(`test_late_cron_skips_day_silently_in_paper_too`). → rc 2 에 warn 경보.

**M8. 비밀 파일 커밋 위험.** 크론 예시(`SWL=... . ./.env.live`)와 런북은 레포 루트에 `.env.live` 를 만들라고 하는데 `.gitignore` 는
`.env` 한 줄뿐이라 `.env.live` 는 **추적 대상**이다. → `.env.*` 추가.

### MINOR

- m3 `_killswitch` 537–538: "직전 거래일 손실" = `equity_history` 마지막 두 행 차이 — 행 날짜 간격을 안 본다(결손 뒤엔 2일 변동).
- m4 세 파일(`state.json`·`book.jsonl`·`paper_broker.json`)은 트랜잭션이 아니다. `_apply_fill` 737 은 장부에 즉시 fsync, 상태는 끝에 저장.
  그 사이 크래시 → 장부엔 주식, 원장엔 포지션 없음 → 키움은 delta 0 이라 영영 `pos None` 고아(평가액엔 잡힘, 안 팔림). `reconcile` 의 "원장≠장부" 행이 유일한 단서.
  시작 시 "장부에 있고 원장에 없는 종목" error 경보 권고.
- m5 `reconcile` 은 장중에 못 돈다(monitor 가 잠금을 종일 쥐고, 120초 대기 후 rc 2). 크론 주석(15:50)만 가능 — 런북에 명시.
- m6 UNKNOWN + 주문번호 없음(타임아웃): `_cancel` 590–592 는 브로커 취소 없이 `expired`. 나중에 체결되면 L4 로 사용자 몫. 주문 시 error 경보는 있다(584).
- m7 `_auction_sell` 기준가가 `base_pric` 없을 때 `trail_closes[-1]`(DB **수정주가**) — 액면분할 직후엔 하한가 바닥이 틀린다(1035).
- m8 페이퍼 monitor 와 dry-run/live monitor 를 같이 돌리면 같은 앱키로 REST 폴링이 2배(10종목 × 2/분) — 키움 요청 한도·다른 데몬과 충돌. 한 모드만 운용 권고.
- m9 `FakeKiwoom`(tests 423–474)은 체결 즉시 잔고가 바뀌는 **즉시 일관성** 가짜라 C2 류 경합을 구조적으로 못 본다(아래 §6).

---

## 2. 상태·복구

- 원자적 저장(`atomic_write_json` tmp→fsync→rename→dir fsync, state.py 214–230), 원장 fsync append(206–211), 손상·모드 불일치·
  스냅샷 유실 거부(272–294) — **PASS**. `.bak` 는 직전 판 하나.
- 시계: 전부 tz-aware KST(`_now` 362–364, `OrderRec.ts` isoformat, 페이퍼 `arrival_ts` 직렬화). `PersistentPaperBroker` 는 `_last_seen_ts`
  를 저장하지 않지만 latency 0 이라 무해(첫 시세에 도착 고정). **PASS.**
- 잠금: `flock` 이라 크래시 뒤 낡은 잠금 없음. monitor 는 09:05~15:19 종일 보유 → close(15:21) 는 `monitor_until` 15:19 로 비켜간다(여유 ~1분; 10종목 REST 지연이면 close 가 120초 대기 → 통과).
- DB 다운(signal): 941–947 예외 → `signal_abort` + error 경보 + 보유 청산 계속. **PASS**(단 M5).
- 휴장 파일 없음: `ka10081` 로 대체(M3 참고). 주말 OK.
- 두 모드 공유: 디렉터리·잠금·모드별 정지 파일 분리, 전역 `swing_STOP` 만 공유 — 의도대로. 단 live 의 낙폭 킬 파일 = 전역 `swing_STOP`
  이라 사람이 다른 이유로 만든 전역 정지를 지우면 **낙폭 킬 ack** 로 읽힌다(`drawdown_check` 386–388). 런북 §4 에 한 줄 추가.

## 3. 체결 현실성 vs 백테스트

- **장전 동시호가 에뮬 = 백테스트 규칙**: 페이퍼 `feed_auction(bid=ask=시가)` → 코어 `ask ≤ 지정가 → ask 체결` ⟺ `EntryPlan.fills_at`
  (시가 ≤ 지정가 ∧ 시가 < 상한가; 지정가 ≤ 상한가−1틱이라 두 번째 조건 포함). **동일.** 실단일가는 부분 배정이 가능하나 `partial` 처리와 09:02 잔량 취소가 있다.
- **상한가 건너뛰기**: 지정가 = 상한가−1틱(strategy `entry_limit`) — OK.
- **손절 폴링 미끄러짐(실측, 아래 §7 표)**: 기대 −46bp(평균)/−15bp(중앙). 1.9% 는 그날 못 잡고 그 중 상당수는 안 잡히는 편이 유리했다(표본 특성).
- **시간 손절**: 15:20 현재가 판단 + 종가 단일가 매도(백테스트 = 종가) — 문서화됨. 종가 동시호가 매도 지정가 = 현재가×0.9 → 사실상 시장가.
- **NXT**: 6자리 코드 = KRX 라는 전제가 문서 기반 미검증(brokers 20–24). 틀리면 08:35 `base_pric`/`cur_prc` 가 NXT 프리마켓(08:00~08:50)
  값일 수 있고, 손절 판단 현재가가 NXT 체결일 수 있다. L5 가 eod 결정을 DB 로 옮긴 것은 옳다; 장중 판단가는 여전히 전제 위에 있다 — §6-4 를 **페이퍼 전** 로 당길 것(M4 와 같은 이유).
- **매도 지정가(L1)**: `_exit_limit` 890–903 — 1차 `shift_ticks(floor_tick(last), −1)`(9,153→9,140; 2,000→1,999 밴드 경계 OK; 1원이면 그대로),
  2차부터 `last×(1−0.5%(k−1))` 내림, 3% 상한, 하한가 바닥 `max(px, limit_down(prev_close))` 886. `test_exit_limit_ticks_and_limit_down_floor` **PASS.**
  단 `prev_close` 가 없으면 바닥이 없다(0.97×last 까지만이라 실해 없음).

## 4. 위험

- 낙폭 킬: 정지 파일 삭제 = ack → 다음 평가에서 고점 재설정(386–389) — 의도 명확, 테스트 있음. 평가 시점이 다음날 07:35 로 옮겨져 **하루 늦게** 걸린다(그날 진입 전이라 효과 동일).
- 일 손실 킬: 직전 평가일 자산 변동(평가손익 포함). 장중 손실은 다음 아침에만 반영 — 장전 진입 전략이라 충분. m3 참고.
- 현금: 계좌 예수금을 안 본다(런북 110행 명시). 거부 → `order_rejected` skip, 상태 일관. **PASS.** 단 거부가 조용하다(M1 은 매도; 매수 거부도 info 뿐 — warn 권고).
- 집중: 10×10% 는 전략 자본 기준의 가상 한도. 사용자 거래가 예수금을 쓰면 키움 거부로 자연 축소.
- 사용자가 전략 포지션을 수동 매도: 전략 매도 주문이 없으면 `error` 경보 + 장부 유지 + 기준선 음수(L4, 테스트 있음) → 전략은 계속 그 주식을 팔려 시도(M1 재발주 루프). 전략 매도가 걸려 있으면 **구분 불가**(런북 69–72 인정). 운영 규칙으로만 막는 상태.

## 5. 운영

- 크론 시각: signal 07:35 는 `daily_finalize` 06:30 + ~55분(07:25 전후) 뒤 — 여유 10분. 늦으면 abort+경보(안전 방향, M5 결손은 남음).
  10:00~10:15 180초, 16:00 전 종료 — 겹침 창 회피 OK. monitor 재기동 10분 간격 OK.
- 경보 사각: 거부·차단 주문(M1), `open None`(M4), rc 2 시간창(M7), 잠금 경합 rc 2, 크론 자체가 안 돈 날(데드맨 없음 — eod/signal info 경보 부재를 사람이 알아채야 한다).
  → `signal`·`eod` 가 매일 info 를 보내므로 "오늘 info 없음" 데드맨 하나면 된다.
- 로그 위생: 앱키·토큰을 찍는 코드 없음(`make_kiwoom_api`, `OrderResult.to_record`). `log.exception` 이 kiwoom_client 예외 문자열을 그대로 찍는 점만 주의. M8 `.env.live`.
- 런북: 모드·일정·안전·킬·복구·전환 단계는 충실. 빠진 것: (1) 롤백 시 "크론 줄을 지우면 보유 포지션이 무감시" — monitor/close 줄은 남기고
  `swing_STOP` 으로 진입만 막든지, 수동 청산 절차를 적을 것(101행); (2) m5 reconcile 시각 제약; (3) §4 전역 정지 파일 삭제 = 낙폭 ack.

## 6. 테스트

리허설 테스트(`test_live_runtime.py`)는 **의미 있다**: 작업마다 새 런타임(= 새 프로세스), 파일 영속, 페이퍼 단일가 규칙, 재시작 재가격,
기준선 보호, L4 네 시나리오를 실제 상태 파일로 검증한다. 동어반복이 아니다. 다만 `FakeKiwoom` 은 "체결 즉시 잔고 반영" 가짜라 **지연·경합**
(C2)·거부(M1)·외부 미체결(M2)·필드 결손(M4)·가짜 휴장(M3)·크래시(C1)·결손일(M5)을 전혀 못 본다. 위 11개 시나리오를 그대로 `tests/` 로 옮길 것을 권고
(`scratchpad/review_runtime/test_adversarial.py` — 프로젝트 하네스를 import 해 그대로 돈다).

---

## 7. 손절 미끄러짐 실측 (런북 §7 표에 기입할 값)

데이터: `minute_bars` 는 **2025-08-01~2026-10-08 만** 있다(2024 불가). 신호일 2025-08-01~2026-09-05, 원시 돌파 유니버스
(종가 > 직전 50일 고가 ∧ 거래량 ≥ 1.5×50일 ∧ ADV20 ≥ 50억; RS·PEAD·수렴 조건은 분봉 표본을 위해 뺌), 다음날 시가 체결(시가 ≤ 피벗×1.05),
손절 d = clip(1 − 수축저점×0.99/진입가, 3%, 10%). 2,324 진입 중 무작위 600 표본 → 분봉 있는 574, 20거래일 내 손절가 터치 353, 그 중 시가 갭(두 모형 모두 시가 체결) 45, **장중 터치 308**.
폴러 모형: 60초(또는 180초) 간격으로 분봉 종가를 `현재가` 로 보고 첫 `현재가 ≤ 손절가` 에서 체결(지정가 = 현재가 − 1틱 ≈ 체결가). KRX 09:00~15:19 분봉만.

| 항목 | 60초 폴링 | 180초 폴링 |
|---|---|---|
| 터치 당일 감지 / 다음날 이후 감지 / 창(20일) 안 미감지 | 87.0% / 11.0% / 1.9% | 85.4% / 12.3% / 2.3% |
| 감지 시점(터치 분봉 이후, 중앙값) | 1분 | 3분 |
| 체결가/손절가 − 1 (감지된 것, 평균 / 중앙) | **−46bp / −15.5bp** | −51bp / −21bp |
| 같은 분위 p25 / p10 / p75 | −42bp / −107bp / −6bp | −48bp / −125bp / −7bp |
| 당일 감지만 (평균 / 중앙) | −36bp / −14bp | −40bp / −20bp |
| 미감지 포지션의 창 끝 가격/손절가 − 1 (중앙) | +15.1%(표본 6) | +9.7%(표본 7) |

- 터치 분봉의 **종가 중앙값은 손절가 +4.9bp** — 저가만 닿고 되돌린 분이 절반이라, 60초 폴러는 그 분엔 못 보고 1분 뒤(중앙) 더 낮은 가격에서 본다. 이것이 미끄러짐의 주원인(호가·대기열은 포함 안 됨 → 실제는 이보다 몇 bp 더 나쁘다).
- 트레이드당 기대수익 영향: 손절 비율 ~60% × 46bp ≈ **−0.28%p/트레이드**(평균 기준; 중앙값 기준 −0.09%p). 감사 추정(30~100bp, −0.2~−0.6%p) 안.
- "첫 지정가 1분 내 체결 비율" 은 1분봉으로는 못 잰다(틱·호가 필요). 지정가 = 현재가 − 1틱은 하락 중 최우선 매수호가 이하라 갭이 아니면 체결 가정이 무리하지 않다 — **미측정**으로 두고 페이퍼 원장(`attempt` 분포)으로 채울 것.
- 민감도용: 백테스트 손절가에 −46bp(평균) / −107bp(p10) 를 붙여 두 줄로 보고하면 된다. 스크립트·행 데이터: `scratchpad/review_runtime/stop_poller.py`, `stop_poller_rows.csv`.

## 8. 페이퍼 GO / NO-GO (1차, 04:00 이전 코드 기준 — §9 로 대체됨)

**GO — 조건 3개를 먼저 고치고 시작.** (1) M4 `open None` 경보+대체, (2) M3 가짜 휴장 이중 확인, (3) M7 rc 2 경보. 이 셋은 페이퍼에서
"진입 0 / 무감시" 를 2주 동안 **조용히** 만들 수 있어 페이퍼 검증 자체를 무효로 만든다. 첫 페이퍼 날은 §6-4 필드 확인(`open_pric`·`base_pric`·6자리=KRX)을
사람이 1회 하고 시작한다. 실계좌 전환은 C1·C2·M1·M2·M6 해결(+ `ka10075/ka10076` 실호출 검증 → 주문번호 체결) 전에는 **NO-GO**.

---

## 9. 최종 재검토 — C1·C2·M1~M8 수정본 (runtime.py 1631행, 04:01 런북)

프로젝트 테스트 72 통과(`test_live_runtime_adversarial.py` 14개 포함). 재공격 테스트 10개(`scratchpad/review_runtime/test_final.py`, 전부 재현).
고쳐진 것: C1 POST 뒤 크래시(의도 → 브로커 주문 이어받음, 재발주 없음 — 검증), C2 "이미 체결" 경합(fill_pending 예약 — 프로젝트 테스트),
M1(3회 거부 → halted + error), M2(`_LedgerOMS` — 남의 미체결 무시, 매수 가드 불변 — 검증), M4(open 없음 → 경보·보류·09:05 대용),
M5(따라잡기 ≤ 15거래일 — 10일 결손에서 정확히 한 번 팔고 그 밖엔 안 판다 — 검증), M7(rc 2 경보), M8(`.gitignore`). 남은 것:

### (1) C1 — 의도 저장·`_resume_orders`

- 의도 저장 → POST 순서라 "POST 뒤 의도 미저장" 은 구조적으로 불가능하다(`_begin_order` 608–623 이 `store.save` 뒤 POST). 저장 자체가 실패하면 POST 안 함(fail-closed). **PASS.**
- **F1 (MAJOR, live)** `_resume_orders` 660–663 의 (종목·방향·수량·가격) 매칭은 **사용자가 같은 종목에 같은 수량·가격으로 걸어 둔 주문**을 우리 것으로
  이어받는다(`test_c1_resume_adopts_users_identical_order`: 우리 POST 는 안 나갔는데 `U000009` 를 `submitted` 로 채택, 우리 진입은 영영 안 나감,
  그 주문이 체결되면 사용자 주식이 전략 포지션이 된다). 키움엔 클라이언트 주문 ID 가 없어 완전한 해결은 없다 — 채택 조건을 "미체결 중 **우리가 모르는
  번호가 정확히 하나** ∧ 번호가 `own_ord_nos` 최댓값보다 크다(당일 계좌 순번)" 로 좁히고, 채택은 warn 이 아니라 **error + 사람 확인**으로.
- **F2 (MAJOR, live)** UNKNOWN **+ 주문번호 없음**(접수 뒤 응답 유실): `_resume_orders` 는 `intent` 만 본다(648). `_cancel` 700–702 는 번호가 없으면
  브로커 취소 없이 `expired`. 09:02 에 원장은 `missed` 로 닫지만 **브로커 주문은 하루 종일 살아 있다** → 11:00 가격이 내려오면 **추격 체결** →
  L4 규칙으로 사용자 몫(`test_c1_unknown_without_ord_no_is_never_adopted_nor_cancelled`). → `unknown ∧ ord_no None` 도 `intent` 와 같이 대조하고,
  대조 실패 시 09:02 에 **같은 종목의 알 수 없는 미체결 매수**가 있으면 error 경보(취소는 사람이).
- 같은 종목 동일 의도 2건 충돌: 재가격은 취소 뒤 새 가격(0.5% 간격 ≥ 1틱, 200원 이상)이라 같은 프로세스 안에서는 생기지 않는다. 전날 의도는
  `_resume_orders` 가 오늘 미체결과 대조 → 없음 → unknown(재발주 없음) → `_expire_stale` 만료. **PASS.**

### (2) C2 — `fill_pending`

- 이미-체결 경합: 예약 유지·재조회 3회·미체결 확인 → 재발주 없음. **PASS**(프로젝트 테스트).
- **F3 (MAJOR, live) 영구 교착 — 손절이 영영 안 나간다.** 취소 거부가 "이미 체결" 이 아닌 이유(사용자가 HTS 에서 우리 주문을 취소, 키움 취소 API 오류,
  휴장일 주문 소멸)면 잔고는 안 변하고 미체결에도 없어 `fill_pending` 으로 남는다. 그 뒤 **빠져나갈 길이 없다**: `_work_exits` 1035 가 건너뛰고,
  `_auction_sell` 1074 는 live sell 이 있다고 안 내고, `_pending_sell` 742 가 수량을 예약해 `_submit_sell` 가용 0, `_job_eod` 의 `_cancel` 은 697–699 에서
  즉시 return, 다음 날엔 `trade_date` 필터(843)로 잔고 배정 후보도 아니다. `test_c2_fill_pending_after_user_hts_cancel_has_no_exit_path`: 손절가 아래
  8,000원에서 **이틀 동안 매도 0건**, 유일한 흔적은 취소 시점 warn 하나. → `fill_pending` 만료 규칙: eod(또는 다음 preopen 기준선 뒤)에 미체결에도 없고
  잔고도 장부와 같으면 `expired` 로 풀고 error 경보; 운영 수단 `sw-live release-order <oid>`.
- "다음 잔고 차이에 먼저 배정": 같은 날은 안전(우선순위 845). 날이 바뀌면 후보에서 빠지는데 예약은 남는다 — 그것이 F3 의 교착 메커니즘.

### (3) M3 — 휴장 의심과 **진짜 휴장**

- `ka10001` 이 `open_pric` 을 비워 주는 경우(지연)엔 의도대로 동작한다(주문 유지·틱마다 재확인). **PASS.**
- **F4 (MAJOR, 페이퍼 포함)** `_quote` 는 `open_pric` 이 참이기만 하면 세션을 `open` 으로 확정한다(1095–1098). 달력에 없는 진짜 휴장일에 `ka10001` 이
  **직전 거래일 시가를 그대로**(캐시) 주면 — 거래소 휴장일에 기본정보 조회가 빈 값을 준다는 근거가 없다 — `holiday_suspect` 를 기록하고도 같은 작업 안에서
  `open` 으로 뒤집혀 `_resolve_open` 이 **어제 시가를 오늘 단일가**로 먹인다. 페이퍼: 휴장일에 어제 시가로 진입 체결 + `day_no` 전진
  (`test_m3_real_holiday_with_stale_open_confirms_session_and_fills_paper`). 실계좌: 주문은 브로커에 없으니 09:02 취소 거부 → `fill_pending` → F3 교착,
  eod 도 못 거둔다(`test_m3_real_holiday_live_orders_end_up_fill_pending`). 휴장일에 새 매도 POST 는 없다(세션 `open` 뒤 보유 포지션 갭 판단이 나오면 그 매도는
  나간다 — 거부 3회 → halted). → 시세로 세션을 확정하려면 **신선도 증거**가 필요하다: `open_pric ≠ DB 직전 거래일 시가` 또는 `ka10001` 의 일자·체결시각 필드
  (미검증), 아니면 `ka10081` 오늘 봉만 확정 근거로 두고 시세는 "의심 해제 후보" 로만. 그때까지 **`holidays.txt` 완비가 페이퍼의 전제 조건**이다.

### (4) M5 — 따라잡기

- 10거래일 결손: 결손 2일째 종가 < MA50 → 복귀일 시가 매도 의도 1건, 그 외 없음, 자산 행 날짜별 생성. **PASS**(`test_m5_catch_up_two_week_outage_sells_what_it_should`).
- **m10 (MINOR)** 배치는 `trail_ma + CATCHUP_DAYS(15)` 개 종가라 **15거래일(3주) 초과 결손**은 오래된 결손일의 MA 가 NaN → 그날의 추적 결정이 warn 한 줄과 함께
  사라진다(`test_m5_catch_up_beyond_catchup_days_starves_ma`). `n = trail_ma + max(CATCHUP_DAYS, last_eod 이후 일수)` 로 하면 없어진다. 결손 중 `day_no` 는 안 올라
  시간 손절·최대 보유가 결손 길이만큼 늦는다(문서화만).

### (5) `_LedgerOMS`

`_pending_sell_qty` 만 덮어써 자기 원장(`intent` 제외)으로 센다(120–130, 477–481). 매수 가드(하드캡·밴드·횟수)·킬·`_held_for_exit`(장부 우선) 전부
코어 그대로(`test_ledger_oms_keeps_core_buy_guard_and_only_counts_own_sells`). 코어가 브로커 미체결로 막던 "크래시 뒤 자기 매도 중복" 은 의도 저장 +
`_resume_orders` 가 대신한다. **PASS.**

### 최종 판정

**PAPER: GO.** 전제: (a) `data/live_swing/holidays.txt` 에 2026 KRX 휴장일 전부(F4 의 유일한 방어), (b) 첫날 §6-4 필드 확인(`open_pric`·`base_pric`·6자리=KRX)
사람이 1회, (c) 한 모드만 운용(앱키 공유), (d) "오늘 signal/eod info 없음" 데드맨. F4 수정(세션 확정에 신선도 증거)은 페이퍼 중 첫 패치로.

**LIVE 전환 전 남은 조건(전부 필수):** F1(채택 조건 좁히기 + error), F2(`unknown` 무번호 대조·09:02 미체결 경보), F3(`fill_pending` 만료·수동 해제),
F4(세션 신선도), `ka10075/ka10076` 실호출 검증 → `fills_verified=True` 주문번호 체결(M6 과잉 귀속·C2 류 잔고 추정 전부의 근본 해결), m10,
런북 §6 2주 페이퍼 + 1주 dry-run 결과, §3 공유 계좌 규칙(전략 종목 수동 매매 금지)의 사용자 동의.
