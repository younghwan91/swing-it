# 스윙 라이브 — 종단 통합 · GA→라이브 인계 검토 (REVIEW_INTEGRATION) — 2026-10-10

검토자 C (트레이딩 시스템 엔지니어 + 퀀트, 적대적). 대상: HEAD `28693d2`(뉴스 거부권·섀도·EPS 설정 포함) + 검토 중 워킹트리
(`src/swing_it/live/*`, `tests/test_live_*`, `deploy/crontab.swing-live.example`, `src/swing_it/krx_session.py`,
`research/experiments/{ga_run,ga2_run}.py`, `research/signals/ga_fast.py`). 검토 초반엔 `news_veto.py`·EPS 필드·`test_live_strategy.py`
EPS 테스트가 다른 두 에이전트의 **편집 중**이었다(첫 두 번의 테스트 실행에서 서로 다른 EPS 테스트가 실패 → `28693d2` 커밋 뒤 재실행 143 통과).
방법: 문서·코드 읽기 + **실DB 리허설**(페이퍼 모드, 실제 `SwingRuntime`·실제 신호 경로, 시세만 DB 정규장 분봉으로 만든 가짜 —
키움 호출 0, API 키 없음, 주문은 프로세스 밖으로 안 나감) + 실DB 30거래일 스캔 + 결손 실험. DB 읽기만. 파일 수정 없음(이 문서만).
스크립트: 세션 scratchpad `review_integration/{scan10.py, rehearse.py, f_lag.py}` + 결과 `scan10.json`, `run_1002/`, `run_multi/`.

**한 줄 결론.** *배관은 돈다* — 실DB 신호 → 계획 → 장전 지정가 → 09:00 체결 → 감시(손절 강제 틱 포함) → 종가 → 장 마감 → 다음날
signal 의 DB 종가 결정·자산 확정까지 3거래일 연속 27개 작업 전부 rc 0, 크래시 0. *그러나 "재최적화 결과를 파일로 쓰고 런타임이
읽는다" 는 설계 §(롤링 재적합)는 코드에 없다* — 설정 파일 입력 자체가 없고, GA2 19유전자 중 **8개가 라이브에 대응물이 없거나 의미가
다르다**(§2 표). 그리고 **수급·실적 테이블이 하루만 늦으면 후보가 조용히 0 이 된다**(§6 실험 — "후보 없음" info 한 줄뿐).
**페이퍼 GO 는 유지(선행 검토와 같음), 단 M1(수급·실적 신선도 abort)은 첫 주 전에 넣을 것. GA2 config 인계는 C1 해결 전엔 불가.**

---

## 1. 하루 리허설 — 실DB × 실런타임 × 가짜 시세 (체크리스트 1)

### 1.1 왜 "real CLI" 로 못 돌렸나 (M2)

`sw-live signal --paper --asof <과거일>` 은 두 가지 이유로 리허설이 **불가능**하다.

1. **키움 없이는 어떤 작업도 안 뜬다.** `SwingRuntime._make_ctx` → `self.bundle` → `build_bundle` → `make_kiwoom_api()` (`brokers.py`) —
   페이퍼도 `RestQuoteProvider` 를 만들려고 **토큰 발급 호출**을 한다(`KiwoomAPI(...)`). `signal` 은 시세가 한 번(`session_date`) 필요하긴 하지만
   DB 만으로 돌 수 있는 경로가 없다. 가짜 시세 공급자 옵션(`--quotes fake:<file>` 류)이 없다. 이번 검토의 규칙(키움 호출 금지)상 CLI 는 못 썼다.
2. **`--asof` 과거일은 signal 이 항상 중단한다.** `FeaturesSignalSource.load` 는 `signal_panels_for(con, None, cfg)` — 신호일 `d` 를 **DB 최신일로
   고정**하고, `d ≥ asof` 면 `RuntimeError("DB 최신일 ≥ 거래일 — 오늘 봉(잠정치)")`. 즉 리허설이 되는 `--asof` 는 "DB 최신일의 다음 거래일" 하나뿐이고,
   런북 "시간창 밖에서 리허설: `--ignore-clock --asof YYYY-MM-DD`" 는 그 하루에만 참이다. 재생(replay)이 안 되니 사고 뒤 재현도 안 된다.

그래서 같은 클래스·같은 작업 함수를 그대로 쓰되 (a) `FakeQuoteProvider` 를 **DB 정규장 분봉**(`krx_session.read_session_bars`)의 시가·고가·저가·종가로
채우고(`date=asof` 로 F4 신선도 증거 충족, 기준 종목 005930 일봉은 DB `daily_bars`), (b) `FeaturesSignalSource` 를 상속해 `signal_panels_for` 의 `d` 만
"asof 직전 DB 거래일" 로 고정했다(`ReplaySource`, 나머지 경로 동일 — NXT 보정 확인·`trail_closes`·거부권·섀도 전부 그대로). 시계는 주입(작업 시간창 강제 유지,
`ignore_clock=False`). 자본 1억, `holidays.txt` 는 레포 `data/live_swing/holidays.txt` 사본.

### 1.2 최근 30거래일 후보 수 (실DB, 라이브 피처 경로 `load_inputs(last_rows=330)` → `build_signal_panels` → `scan`)

적재 **125 s** + 피처 6 s (전 종목 2,747 · 330행). 유니버스(위생 ∧ ADV20 ≥ 50억) 386~410 종목/일. **분봉 미보정 유니버스 종목 0/일 (30일 전부)**,
F·PEAD 결측 0/일, rel 결측 ≤ 1/일.

| d | 유니버스 | 뼈대 S | S∧PEAD≥.7 | S∧PEAD∧rel≤1 | **후보(F=1)** | 거부권(원장 재사용/미판정/거부) |
|---|---|---|---|---|---|---|
| 08-25 ~ 09-22 (19일) | 386~410 | 0~3 (합 22) | 0~3 (합 10) | 0~1 (합 1) | **0** | — |
| 09-23 | 400 | 2 | 2 | 1 | **1** (417840) | 0 / 17 / 0 |
| 09-28, 09-29 | 399, 396 | 0, 1 | 0, 1 | 0 | 0 | — |
| 09-30 | 400 | 1 | 1 | 1 | **1** (417840) | 0 / 17 / 0 |
| 10-01 | 399 | 4 | 2 | 1 | **1** (096530) | 0 / 7 / 0 |
| 10-02, 10-06, 10-07 | 398, 394, 394 | 1, 1, 1 | 0, 0, 1 | 0 | 0 | — |
| 10-08 | 392 | 3 | 1 | 1 | **1** (417840) | 0 / 18 / 0 |

마지막 10거래일(09-23 ~ 10-08): 후보 있는 날 **4일, 각 1종목**(417840 ×3 — 같은 종목의 재돌파, 096530 ×1), 거부 0. 연 12.5건(REVIEW_BACKTEST)과 맞는 빈도.
거부권 **원장 재사용 0** — 연구 배치(`v2r3-haiku55`, 36,214행)는 전일 거래대금 상위 300 종목만 판정해 이 후보들의 기사는 하나도 없다. 라이브는 매 후보를 그날
새로 판정한다(7~18쌍/일, Haiku 5.5 — 비용 수십 원, 상한 500쌍·600 s 는 넉넉).

### 1.3 3거래일 연속 체인 (asof 2026-10-02 → 10-06 → 10-07, 상태 디렉터리 공유)

| asof | signal | preopen | open_check | monitor ×3 (09:06 시가 / 10:30 저가 / 14:00 종가) | close | eod | 자산 행(다음날 signal 확정) |
|---|---|---|---|---|---|---|---|
| 10-02 (d=10-01) | 후보 1 (096530) → planned, 지정가 37,500 × 268주 (= 자산 10.05%) | `order_intent` 저장 → 매수 submitted | 09:00 시가 37,300 ≤ 37,500 → **체결 37,300**, d=10%, 손절 33,570, 목표 44,760 | 저가 36,900 > 손절 → 무행동 | 매도 의도 없음 | 잠정 자산 99,892,800 | — |
| 10-06 (d=10-02) | **DB 종가 36,900 으로 돌파 실패·MA50 판정 → 없음**, 자산 행 10-02 = 99,892,800 (dd −0.11%) 확정, 후보 0 | 기준선·매도 없음 | 보유 day_no 2 | 무행동 | — | 잠정 99,986,600 | 10-02 행 |
| 10-07 (d=10-06) | 자산 행 10-06 확정, 후보 0 | — | day_no 3 | 무행동 | — | 잠정 100,428,800 | 10-06 행 |

27개 작업 전부 rc 0, 예외 0, 경보는 info 뿐(신호·장 마감). 원장 사건 순서: `equity, signal, baseline, order_intent, order, fill, position_open, eod | equity, signal, baseline, eod | …` — C1 의도 저장 → POST 순서 확인.
**손절 강제 틱 변형**(`run_1002`, 10:30 틱에 현재가 = 손절가 × 0.995 주입): `청산 결정 stop 268주(continuous)` → 지정가 = 현재가 − 1틱 → 페이퍼 즉시 체결 33,350 →
`position_close stop 실현 −1,065,409원`(−10.66%: d 10% + 1틱 + 비용) → close/eod 에서 재발주 없음, `status` 거부권 줄 `진입군 n=1 평균순수익 −10.66% 손절률 100%`. 정상.

관찰(결함 아님): 10-02 체결 뒤 같은 날 signal 자산 행은 10-01 것 그대로(설계대로 확정은 다음날). 섀도는 생성되지 않았다(거부 0) — 섀도 경로는 프로젝트 테스트
(`test_vetoed_candidate_becomes_shadow_position_without_orders` 등 4개)로만 확인했다.

### 1.4 리허설 중 실제로 난 사고

- **DB 연결 끊김 1회**: 첫 스캔에서 `read_prices` 도중 `psycopg2.OperationalError: server closed the connection unexpectedly`(GA 실행과 DB 를 공유하는 시간대).
  라이브에서 같은 일이 07:35 에 나면 `signal_abort` + error 경보 + "오늘 신규 진입 없음" — 재시도 없음(M6). 보유 종목의 종가 결정도 그날은 빠진다(다음날 따라잡기 M5 경로가 메운다).
- `load_inputs` 는 `read_prices(con)` 로 **전 기간 전 종목을 읽은 뒤** 메모리에서 300행으로 자른다(125 s 중 대부분). 신호 시간창(07:31~08:29)엔 들어가지만 DB 부하·끊김 노출이 크다.

## 2. GA2 → 라이브 인계: 유전자 → `StrategyConfig` 대응표 (체크리스트 2) — 가장 중요한 산출물

사실 관계: `live_candidate.json`(`ga2_run.py` 740~757행)은 `config: {유전자 19개}` 를 쓴다(끔 = `None`, 없음 = `"inf"`/`0`). 이를 읽는 코드는 **어디에도 없다** —
`cli.py` 는 `RuntimeConfig(...)` 에 strategy 를 안 넘기고(`RuntimeConfig.strategy = DEFAULT_STRATEGY`, `FeaturesSignalSource(cfg.strategy)`), `--strategy-file` 류 플래그가
없으며, `StrategyConfig(` 는 `config.py` 밖에서 호출되지 않는다. 즉 **런타임은 코드에 박힌 기본값으로만 돈다.**

| # | GA2 유전자 | 범위 / 끔 | `StrategyConfig` 필드 | 판정 | 비고(의미 차이) |
|---|---|---|---|---|---|
| 1 | `near_high` | [0.50, 0.95] / None | **없음** | 결손 | 라이브 `trend_template(c, high, low)` 는 `near_high=0.75` **고정 인자**(`features.build_signal_panels`). GA2 마스크는 추세 템플릿을 이 4개 유전자로 **분해**한다(`_mask_ga2_py`) — 라이브는 원전 템플릿 전체를 요구 |
| 2 | `above_low` | [1.00, 1.60] / None | **없음** | 결손 | `above_low=1.30` 고정 |
| 3 | `ma_stack` | {0,1,2,3} | **없음** | 결손 | 라이브 TT = `close>MA50>MA150>MA200` 전부(= 3 고정) |
| 4 | `ma200_slope_days` | {0,10,20,40} | **없음** | 결손 | `slope_lag=20` 고정 |
| 5 | `rs_min` | [0, 95] / None | `rs_min` | 대응 | 끔(None) → GA 는 RS NaN 도 통과, 라이브 `rs ≥ 0` 은 NaN 탈락 |
| 6 | `pivot_window` | {0,20,30,40,50,60} | `pivot_window` | 부분 | 20~60 은 1:1(피벗·거래량 평균 창·`plan.pivot`·돌파 실패 기준·chase 상한이 전부 같은 창을 따른다 — 일관). **0 = 돌파 요구 없음은 대응물 없음**(라이브는 `breakout` 필수) |
| 7 | `vol_mult` | [0, 2.5] / None | `vol_mult` | 대응 | 끔 → 라이브 0.0: `pivot_breakout` 이 `vol_avg > 0` 을 추가로 요구(GA 끔은 그 조건도 없음) — 무시 가능 |
| 8 | `breadth_min` | [0, 0.7] / None | `breadth_min` | 대응 | 둘 다 `>`; 끔 → 0.0 (NaN 차이는 5 와 같음) |
| 9 | `pead_min` | [0, 0.95] / None | `pead_rank_min` | 대응 | **이름 다름**; 끔 → GA 는 PEAD NaN(실적 없음) 통과, 라이브 0.0 은 NaN 탈락 |
| 10 | `rel_max` | [0.5, 4.0] / None | `rel_max` | 대응 | 끔 = None 둘 다, NaN 통과 둘 다 — 유일하게 끔 의미가 정확히 같은 연속 유전자 |
| 11 | `eps_yoy_min` | [0, 1.0] / None | `eps_yoy_min` (28693d2) | **정의 다름** | GA(`ga_run.py` 380~390행): 순이익을 **분기 말(`_pe`/`_pe0`) as-of 주식수**로 먼저 나눠 `earnings_leg_panel` 에 넣는다. 라이브 `features.earnings_panels`: `(netinc/netinc_prior) × (주식수_{d−365}/주식수_d)` — **신호일 d 와 d−1년 as-of 주식수**. 증자·분할 시점이 분기 말과 신호일 사이면 값이 다르다. 피처 패리티 테스트 없음 |
| 12 | `eps_accel` | {0,1} | `eps_accel` | 대응 | `yoy > yoy_prev`; GA 는 NaN→0(가속 아님), 라이브 NaN→`==1` 거짓 — 같다 |
| 13 | `rev_yoy_min` | [−0.2, 0.5] / None | `rev_yoy_min` | 대응 | 같은 `rev_yoy` 열 |
| 14 | `chase` | {0.02,0.05,0.10,∞} | `chase` | 대응 | ∞ → 라이브 지정가 = 상한가 − 1틱(`min(inf, …)`), 백테스트 `limit=inf` ∧ 상한가 비율 — GA 와 같은 집합 |
| 15 | `stop_cap` | {.06,.08,.10,.12,.15} | `stop_cap` | 대응 | `sizing.position_size` 의 위험 1% 항이 `stop_cap > 0.10` 에서 **처음으로 묶인다**(0.12·0.15 면 종목 비중 < 10%) — GA 포트폴리오 적합도는 비중 10% 고정이라 **사이징이 어긋난다** |
| 16 | `take_r` | {1.5,2,3,4,∞} | `take_r` | 부분 | 유한값 1:1. **∞(익절 없음)**: GA 커널은 "전량 MA 추적"(`ga_fast` 모듈 설명)인데 라이브 `PositionState` 는 추적을 **익절 뒤 단계(`half`)에서만** 본다 → `target_px = inf` 면 영영 `full` → MA 추적이 **한 번도 안 걸린다**. 의미 불일치 |
| 17 | `trail_ma` | {10,20,50,0=없음} | `trail_ma` | 부분 | 10·20·50 은 1:1(`moving_average(c, trail_ma)`, `_decide_on_db_close` 의 `n`, `SignalBatch` 길이 전부 cfg 를 따른다 — 단 패널 키·문서는 `ma50` 고정 이름). **0 = 추적 없음은 대응물 없음**(`rolling(0)` 오류) |
| 18 | `failed_breakout` | {0,1} | `failed_breakout` | 대응 | bool; 둘 다 "그 config 의 피벗 창" 기준, 1~max_hold−1 일째 종가 — 같다(패리티 테스트가 못박음) |
| 19 | `max_hold` | {20,60,120} | `max_hold` | 대응 | |
| — | 수급 시대 `smart_min` / `foreign_pos` / `persist_min` | [−.1,.2] / {0,1} / {0,8,10,12} | `require_flow` (bool) 만 | **결손** | 라이브 F = `smart>0 ∧ foreign>0` = (`smart_min=0, foreign_pos=1, persist_min=0`) 한 점만 표현 가능 |

라이브에만 있고 GA 에 없는 것(인계 시 그대로 둬야 하는 고정값): `adv_floor` 5,000(GA H5 와 같음), `contraction_window` 10, `stop_floor` .03, `stop_mult` .99, `partial_frac` .5,
`limit_up_ratio`, `rel_*` 창, `time_stop_days`(None — GA2 는 유전자에서 제거), **`news_veto`**(GA 는 거부권 없이 골랐다 — 인계 config 의 OOS 근거엔 거부권이 없다), 사이징·위험 전부.

의미가 다른 곳(유전자 값과 무관하게): (i) 본전 시점 — 일봉 `on_bar` 는 커널과 같고, 라이브 이벤트 경로 `on_quote` 는 익절 당일 나머지 절반에 **원래 손절이 그대로 산다**
(28693d2 "당일 손절 공백 수정" — 커널엔 없는 보호, 방향은 라이브 유리·문서화됨). (ii) 돌파 실패 판정일 — 둘 다 진입일(1일째) 종가부터, 같다. (iii) 수축 창 — 둘 다 10 고정.
(iv) `pivot_window ≠ 50` — 라이브는 `pivot` 패널·`plan.pivot`·`entry_limit` 이 전부 같은 창에서 나와 일관(확인).

**결론(C1):** 19개 중 1:1 은 11개(5·7·8·9·10·12·13·14·15·18·19 — 그중 5·8·9·15 는 끔/사이징 의미 차이 있음), 부분 3개(6·16·17), 정의 다름 1개(11), 결손 4개(1~4) + 수급 3개.
GA2 가 고른 설정을 "1:1 로 싣는다"(config.py 주석)는 **지금 참이 아니다.** 결손 4개는 `trend_template` 에 `near_high·above_low·stack_k·slope_lag` 인자를 열고 cfg 필드를 더하면
된다(함수는 이미 `near_high`·`above_low`·`slope_lag` 인자를 받는다 — `ma_stack` 만 새로). 16·17 은 `PositionState` 에 "익절 없음 ∧ 추적 = 전량 추적", `trail_ma=None` 분기가 필요하다.

## 3. 설정 버전·롤링 재적합 (체크리스트 3)

**현황: 없다.** 설정 파일 읽기 없음(§2), 스냅샷·원장에 설정 해시 없음(`LiveState` 에 `capital` 뿐), `PositionRec` 에 진입 당시 설정 없음. 더 나쁜 것(M3):
`SwingRuntime._strat` 은 **모든 작업에서 열린 포지션을 현재 전역 설정으로 다시 묶는다**(`PositionState.from_dict(pos.strat, self.cfg.strategy)`, runtime.py 1082행) 그리고
`_decide_on_db_close` 는 `self.cfg.strategy.trail_ma` 로 MA 를 센다. 첫 재적합이 `take_r`·`trail_ma`·`max_hold`·`failed_breakout`·`stop_cap` 을 바꾸는 순간, **옛 설정으로 들어간
포지션의 청산 규칙이 소급해서 바뀐다**(`stop_px` 만 체결 때 굳고 `target_px` 는 재계산되지 않지만 `max_hold_due`·`trail`·`failed_breakout` 은 cfg 를 매번 읽는다).

최소 설계(제안 — 이 문서에만 적는다):

- **파일.** `data/live_swing/strategy/<YYYY-MM-DD>_<sha8>.json` (재적합 작업이 쓴다) + `data/live_swing/<mode>/strategy.json` → 심볼릭 링크 또는 "현재" 포인터 파일. 내용:
  `{"schema": 1, "generated": iso, "valid_from": "YYYY-MM-DD"(≥ 다음 거래일), "source": {"path": "research/logs/sepa_explore/live_candidate.json", "sha256": …, "commit": …,
  "procedure": "GA2 port 시드 중앙값"}, "genes": {19개 그대로}, "strategy": {StrategyConfig 필드 전부}, "risk": {RiskConfig 전부}, "hash": sha256(canonical strategy+risk)}`.
- **로더.** `StrategyConfig.from_genes(genes) -> StrategyConfig` — §2 표를 코드로: 대응 없는 유전자가 끔이 아니면 **`ValueError`**(조용히 무시 금지), 이름 변환(`pead_min→pead_rank_min`,
  `trail_ma 0→None`, `"inf"→math.inf`, `0/1→bool`), 그리고 `load_strategy_file(path)` — 알 수 없는 키·빠진 키 거부, `strategy` 블록이 `from_genes(genes)` 와 **같아야** 통과(두 표현의 자기 일관성).
  CLI `--strategy-file`(기본 `<mode>/strategy.json`, 없으면 DEFAULT 로 돌되 배너에 "DEFAULT(코드 기본값)" 명시).
- **버전 고정.** 스냅샷 `LiveState.config_hash`·원장 모든 사건에 `cfg=<sha8>`; `PositionRec.cfg` 에 계획 시점 `strategy` dict 를 **복사** → `_strat` 은 `pos.cfg` 로 복원,
  `_decide_on_db_close` 도 `pos` 의 `trail_ma`. 새 설정은 `valid_from` 이후 **신규 계획에만**. `max_positions`·위험 한도만 전역.
- **적재 시 패리티 테스트**(재적합 작업이 파일을 쓰기 전에 돌리고 결과 해시를 파일에 적는다): (1) `genes → cfg → genes` 왕복 동일, (2) 고정 합성 패널 + 실DB 2025-01~ 사건에서
  `run_backtest(cfg, unlimited=True)` 트레이드 = `simulate_minervini(같은 프로토콜)` = **`ga_fast.engine_events(같은 g·ek)`**(설정을 고른 커널 자체) — 진입일·청산일·사유·수익 1e-9,
  (3) `features.build_signal_panels(cfg)` 의 d 행 vs GA 사전계산의 같은 (종목, d) 피처(특히 `eps_yoy`) 불일치 0. (2)(3) 중 하나라도 어긋나면 파일을 쓰지 않는다.
- 운영 규칙(런북에 적을 것): 설정 교체일엔 `sw-live status` 가 `설정 <sha8> (valid_from …) · 열린 포지션 중 옛 설정 n개` 를 찍는다.

## 4. `krx_session` 사용 (체크리스트 4)

- **커버리지(실측)**: 30거래일 × 유니버스 386~410 종목에서 분봉 미보정 **0**. 신호일 d 의 후보가 미보정이면 `보정 안 된 후보 제외` warn + 제외(확인). `krx_patched` 는
  "그날 한 종목이라도 보정됐나" 의 전역 플래그라 분봉 테이블 전체 결손만 잡는다(그 경우 abort — 올바름).
- **M4 — 보유 종목은 검사 안 한다.** `FeaturesSignalSource.load` 의 `trail_closes` 는 보정 패널의 `close[code]` 를 그대로 쓰고, `patched_codes` 대조는 **후보에만** 건다.
  보유 종목의 d 행이 미보정이면(거래정지·분봉 누락) 그 종가 = NXT 19:59 체결가로 **돌파 실패·MA50·최대 보유·자산 행·낙폭 킬**을 조용히 판정한다. 분봉은 상장 종목의 절반만 있어
  (하루 1,400~1,800) 유니버스 밖으로 밀린 보유 종목(ADV 급감)에서 실제로 생길 수 있다. → `_decide_on_db_close` 에 `code ∉ patched_codes` 면 그날 결정 보류 + warn(후보와 같은 규칙).
- 섀도도 같은 `trail_closes` 를 쓴다(같은 문제, 같은 고침).

## 5. 뉴스 거부권 + 섀도 (체크리스트 5, 코디네이터 추가 요청)

확인한 것(PASS):
- **PIT 15:30.** `apply_live_veto`: `hi = win[-1] 15:30 KST`(`CLOSE_CUTOFF`), `map_signal_days(mode="close")` = "published_at < d 15:30 → d, 그 뒤 → 다음 거래일" — 연구와 같은 함수.
  창은 거래일 기준(`window_days` 20·5, d 포함), 5일 창은 20일 창의 마지막 5일. 테스트 `test_count_news_windows_pit_and_unknown`. d+1 07:35 에 d 15:30 이전 기사만 — 전진 규칙 그대로.
- **비용·시간 가드.** `MAX_JUDGED_PER_DAY=500`, `TIME_BUDGET=600 s`, 요청 타임아웃 30 s, 넘으면 나머지 모름 + 경보(`info["alerts"]` → signal 의 warn 경보 `신호 원천 경고`, 테스트 2개).
  실측 후보 1종목/일 × 7~18쌍 → 수십 원. API 전면 장애 시 최악 = 18쌍 × 30 s = 9분 ≤ 600 s; 적재 131 s + 600 s = 12분 < 시간창 58분.
- **키.** `api_key_from_env`: 환경변수 → `../quant-airflow/.env` 직접 파싱(워크트리에서도 `parents[3].parent` = `~/git`). 키 문자열이 로그·경보·원장에 가는 경로 **없음**(grep: 유일한
  언급은 "ANTHROPIC_API_KEY 없음" 문자열). 오류 문자열은 `http <code>: resp.text[:200]`(API 오류 본문)·`type(exc).__name__` 뿐. `build_request` 는 본문만 만들고 헤더는 `messages_judge`
  클로저 안에만 있다.
- **모름 ≠ 거부.** `split_candidates`: 재료 없는 종목 = 진입; `count_news` 는 판정 없는 쌍을 `unknown` 으로 세고 `neg` 에 안 넣는다. 키 없음·DB 실패·원장 읽기 실패·판정 실패 전부
  "전 후보 모름 + 경보" 로 접힌다(`test_apply_live_veto_without_api_key…`, `…db_failure_never_blocks`). `FeaturesSignalSource._apply_veto` 가 한 번 더 감싼다(ImportError·예외 → 전 후보 진입).
  리허설에서도 키 없이 돌아 "후보 1 · 거부 0" 으로 진행됐다(모름 경로).
- **원장.** judge `v2r3-haiku55-live`, `LIVE_SPEC` 은 `V2R3_HAIKU55` 의 model/system/user/params 를 그대로(테스트 `test_live_spec_is_same_definition_as_research_judge`) →
  `pair_input_hash` 동일. `register_judge` 는 같은 이름·다른 정의면 `JudgeCollision`, 같은 정의면 멱등; `JudgmentLedger.record` 는 첫 `record` 에서만 등록(미판정 0 이면 등록 안 함 — 무해).
  행 쓰기는 `ON CONFLICT (article_id, code, judge) DO UPDATE … WHERE output IS NULL` — 실패 행만 덮어쓴다 → 재실행 멱등. 원장 쓰기 실패는 경보 + 그날 판정은 쓴다(다음날 재판정).
  재사용 순서 `v2r3-haiku55-live` → `v2r3-haiku55`(같은 정의)만, `v2-haiku45` 는 안 쓴다(맞다 — 다른 모델).
- **섀도 생애주기.** planned(signal) → 09:02 `_shadow_open`: 세션 `open` 확정 뒤에만, `plan.fills_at(open)`(실포지션과 같은 식) → open/`missed`; 전일 `planned` 는 `stale_plan` 만료;
  `below_low` 즉시 청산; 15:40 `_shadow_eod`: `on_bar(시가·고가·저가·현재가, MA = DB 49종가 + 오늘)` 진입일 포함(day_no 1 — 백테스트와 같음); "다음날 시가" 결정은 `exit_intent` 로
  남겨 다음 `_shadow_open` 의 `on_open` 이 `_pending_exit` 로 체결; 섀도 장부(`shadow_book.jsonl`)로 실비용. 실포지션과 코드 겹치면 섀도 안 만듦; live 모드에서도 주문 0(테스트).
  `trade_stats` 는 `status == closed ∧ entered_qty > 0` 만 세므로 `missed`·`expired` 는 양쪽 다 제외 — 비교 모집단이 "체결된 것끼리" 로 대칭.

문제:
- **M5 — 섀도와 실포지션의 종가 결정 경로가 다르다.** 실포지션: 다음날 07:35 DB **KRX 보정 종가**로 돌파 실패·MA50·최대 보유. 섀도: 15:40 `ka10001 cur_prc`(KRX 15:30 단일가라는
  검증 없음 — 감사 L5, 바로 그 이유로 실포지션은 안 쓰는 값)와 미검증 `high_pric/low_pric`. 거부군 vs 진입군은 **측정 방법이 다른 두 집단**의 비교가 된다. 섀도의 `trail_closes` 는
  signal 이 이미 받으므로(`held` 에 섀도 포함) 종가 결정만 실포지션과 같이 `_decide_on_db_close` 로 옮기고, 15:40 은 손절·익절(일중 고저)만 쓰면 대칭이 된다.
- **m1 — 합격 바 지표 불일치.** `NEWS_VETO.md` 사전등록: "거부군 평균 **정합 초과** < 진입군". `shadow_vs_real_line`·런북 6b: "평균 **순수익**". 정합 대조 없이 순수익으로 바꿨으면
  사전등록 문서를 고쳐야 한다(지금은 둘이 다르다).
- **m2 — 백테스트 거부 패널은 라이브와 다른 판정 모집단.** `veto_panel_from_ledger` 우선순위 `v2r3-haiku55 → v2-haiku45`(다른 모델 포함), 연구 배치는 거래대금 상위 300만 → 백테스트의
  "모름" 집합이 라이브와 다르다. "같은 규칙" 은 경계값만 같다.
- **m3 — 기사 조회 하한 `lo = win[0] − 4일`.** 연휴로 거래일 간격이 5일을 넘으면(예: 추석 09-23 → 09-28 은 5일, 설·추석+대체휴일 조합은 6일 가능) 첫 휴일에 난 기사가 조회 밖이다.
  10일로.
- **m4 — 섀도 `open_pric` 없음(M4 시나리오)엔 대용이 없다** → 그날 `planned` 로 남고 다음날 `stale_plan`. 실포지션은 09:05 이후 현재가 대용. 비대칭(작음).

## 6. 비밀·로그·경보 (체크리스트 6)

- 비밀: `.gitignore` 에 `.env`·`.env.*`·`data/`·`data/live_swing/**`. 크론 예시는 `../quant-airflow/.env` 를 source → `sw-live` 프로세스 환경에 KIWOOM·ANTHROPIC·DART·SHARADAR·DB 비밀 전부가
  실린다(필요 이상). `notify` 는 제목·본문 텍스트만 POST, `os.environ` 덤프 없음, 예외 문자열에 헤더 없음. 로그 파일 `data/live_swing/<mode>/logs/`·`cron.log` 는 레포 밖(gitignore). **PASS.**
  (작게: `ALERT_WEBHOOK_URL` 폴백은 다른 데몬과 같은 채널 — "swing-live" 접두로 구분된다.)
- **M1 — 조용한 0: 수급·실적 신선도 검사가 없다.** signal 의 중단 조건은 가격 최신일 = 직전 거래일, NXT 보정 여부 뿐. 실험(`f_lag.py`, d = 10-08, 유니버스 392):

  | 입력 | F=1 | F NaN | PEAD NaN | 후보 | F·PEAD 조건 끈 변형 |
  |---|---|---|---|---|---|
  | 정상 | 98 | 0 | 0 | 417840 | 417840 |
  | `supply_demand` 가 **하루** 늦음 | **0** | **392** | 0 | **없음** | 417840 |
  | `earnings` 비어 있음 | 98 | 0 | 392 | **없음** | 없음 |

  결과는 info 경보 "후보 없음" 한 줄 — 정상 상태(10거래일 중 6일이 후보 0)와 **구별 불가**. 질문의 "일주일 후보 0 경보" 는 기저율(후보 있는 날 ≈ 1/8)상 잡음이다; 맞는 감시는
  **원인 쪽**이다: signal 이 (a) `max(supply.date) == d`, (b) 유니버스의 F 결측 비율·PEAD 결측 비율 ≤ 10%(실측 0%), (c) `shares_outstanding_history` 최신일 ≤ 30일(EPS 켤 때), 아니면
  **error 경보 + 진입 중단**(보유 결정은 수행). 더해 주간 요약에 "뼈대 S 수" 를 넣으면(30일 중 22일 ≥ 1) 피처 파이프라인 고장을 사람이 본다.
- 데드맨: 런북이 전제로 적은 "오늘 signal/eod info 없음" 감시는 코드에 없다(외부 cron 감시 과제 — 명시만).
- `_check_window` rc 2 + warn(M7 반영) 확인. `signal_abort` 는 error. 상태 손상 rc 3 + error. **PASS.**

## 7. 테스트 (체크리스트 7)

- 전체: `tests/` **1,150 통과 · 3 건너뜀 · 1 실패**(32.6 s, 3코어) — 실패는 편집 중이던 `tests/test_live_strategy.py` EPS 테스트(첫 실행 `test_earnings_gates_off_by_default…`,
  둘째 실행 `test_eps_yoy_uses_pit_shares_and_falls_back` — 파일이 실행 사이에 바뀜). **`28693d2` 커밋 뒤 재실행: `test_live_{strategy,runtime,news_veto,parity,backtest}` 143 통과.**
  `test_ga_fast`·`test_ga_run_helpers` 38 통과 · 3 건너뜀(numba 없음).
- 시계·KST·휴일 의존: `tests/test_live_*` 에 `now_kst()`·`date.today()`·`datetime.now()` 직접 호출 **없음**(전부 `clock=` 주입, `SIGNAL_D` 상수). 휴장 파일은 tmp 에 생성. **플레이키 없음.**
- 환경 의존(주의): 첫 실패 `test_earnings_gates_off…` 는 `DataFrame.stack()` 의 NaN 처리가 **pandas 버전**에 따라 다른 데(이 venv 는 pandas 3.0.5 — NaN 행 유지) 기댄 단언이었다 —
  커밋본에서 바뀌었는지 확인할 것(지금은 통과). `test_pbo` 2.4 s·`test_tui_smoke` 2.1 s 가 가장 느리다.
- 공백: (1) GA2 config → `StrategyConfig` 로더·패리티 테스트(§3) 없음 — 로더가 없으니 당연. (2) `take_r=∞`·`trail_ma=0`·`pivot_window=0` 을 `StrategyConfig` 에 넣었을 때의 동작 테스트 없음.
  (3) 수급·실적 결손 → 후보 0 시나리오 테스트 없음(§6). (4) 보유 종목 미보정 종가 시나리오 없음(§4). (5) `--asof` 과거일 재생 테스트 없음(불가능하므로).

---

## 판정 목록

### CRITICAL

- **C1. GA2 → 라이브 1:1 로더가 없고, 있어도 19유전자 중 8개가 실리지 않는다**(§2). 설정 파일 입력 자체가 없고(`cli.py`·`RuntimeConfig.strategy = DEFAULT_STRATEGY`), 추세
  유전자 4개(`near_high·above_low·ma_stack·ma200_slope_days`)와 `pivot_window=0`·`trail_ma=0`·`take_r=∞`(전량 추적) 와 수급 3개는 대응물이 없거나 의미가 다르며, `eps_yoy` 는 **정의가
  다르다**(분기 말 주식수 vs 신호일 주식수). "롤링 재최적화" 철학의 전제(검증한 것 = 운용하는 것, REVIEW_GA2 퀀트 비판 1)가 코드에 없다. → §2 결손 해소 + §3 로더·패리티 테스트.
  GA2 결과가 나와도 **이 전에는 인계 불가.**

### MAJOR

- **M1. 수급·실적 테이블 하루 지연 → 후보 조용히 0**(§6 실험, F NaN 392/392). 신선도 abort 추가.
- **M2. 재생·리허설 불가**(§1.1): `d=None` 고정 + 키움 없이 어떤 작업도 못 뜸. `--asof` 재생(`signal_panels_for(d=asof 직전 거래일)`)과 `--quotes fake:<csv>`/오프라인 공급자를 CLI 에.
  페이퍼 2주의 사고 재현·사후 분석이 지금은 코드 밖 스크립트로만 된다.
- **M3. 열린 포지션이 전역 설정에 매번 다시 묶인다**(§3) — 첫 재적합에서 옛 포지션의 청산 규칙이 소급 변경. 포지션별 설정 고정 + 설정 해시.
- **M4. 보유 종목의 미보정(NXT) 종가가 검사 없이 결정·자산에 쓰인다**(§4). 실측 노출 0 이지만 유니버스 밖으로 밀린 보유 종목에서 생긴다.
- **M5. 섀도 종가 결정이 실포지션과 다른 가격·시각**(§5) — 거부군 vs 진입군 비교의 측정 비대칭. `trail_closes` 로 같은 경로로.
- **M6. 아침 적재가 전 기간 전 종목 `read_prices` + 재시도 없음**(§1.4, 125 s, 끊김 1회 실관측). SQL 에 날짜 하한 푸시다운(`last_rows` → `date ≥`) + 시간창 안 2회 재시도.
- **M7. `stop_cap > 0.10` 유전자값에서 사이징이 GA 와 어긋난다**(§2 #15): 위험 1% 항이 묶여 비중 < 10%. GA 포트폴리오 적합도는 비중 10% 고정 — 인계 시 `risk_per_trade = stop_cap × max_position_frac` 로 맞추거나 위험 항을 끈다(REVIEW_BACKTEST M5 와 같은 뿌리).

### MINOR

- m1 합격 바 지표(정합 초과 vs 순수익) 문서·코드 불일치(§5). m2 백테스트 거부 패널의 판정 모집단 ≠ 라이브(§5). m3 기사 조회 하한 4일 < 긴 연휴(§5). m4 섀도 `open_pric` 대용 없음(§5).
- m5 연속 유전자 "끔" 의 NaN 의미(GA 통과 / 라이브 탈락 — `rs_min`·`pead_rank_min`·`breadth_min`·`vol_mult`), §2. 로더에서 끔 → `None` 필드로 받아 조건을 **빼는** 쪽이 GA 와 같다.
- m6 `SignalBatch`·패널 키 `ma50`·문서가 "MA50" 고정 이름 — `trail_ma≠50` 이면 이름만 거짓(동작은 cfg 를 따른다).
- m7 `live_candidate.json` 의 `"inf"` 문자열·`0`=없음 규약이 `genes_off_meaning` 문자열에만 있다 — 로더 입력 규약은 스키마(버전 번호)로.
- m8 크론 환경에 불필요한 비밀(DART·SHARADAR·Airflow)까지 실린다 — `.env` 에서 KIWOOM·DB·ANTHROPIC·웹훅만 뽑는 래퍼.
- m9 "일주일 후보 0 경보" 는 기저율상 잡음 — 대신 M1 의 원인 감시 + 주간 뼈대 S 요약(§6).
- m10 `trade_stats` 의 진입군은 `state.history` 전부(거부권 켜기 전 포지션 포함) — 비교 창을 "거부권 운용 시작일 이후" 로 자를 것.

### 확인한 것(문제 없음)

실DB 3거래일 체인 27작업 rc 0(§1.3)·손절 강제 틱 체결·C1 의도 저장 순서·자산 행 다음날 확정·30일 유니버스 분봉 보정 100%·F/PEAD 결측 0·거부권 PIT 15:30·모름≠거부·
키 미노출·원장 멱등·섀도 생애주기·시계 주입 테스트·가드레일 린트 통과(`scripts/check_guardrails.py`)·`krx-quant-core==0.6.2` 고정·`pivot_window≠50` 일관성·`failed_breakout` 정의 일치.

---

## 재검증 — `7084b8f`(+ `4b1bc69`) + 검토 중 워킹트리 (2026-10-10 05:30~06:05)

대상: 커밋 `7084b8f`("GA→라이브 인계 …")·`4b1bc69`, 그리고 재검증 **도중** 다른 에이전트가 고친 워킹트리(`runtime.py`·`cli.py`·`brokers.py`·`features.py`
— 재생 시계 `SimClock`·신선도 `≤ d` 비교·인계 파일 라우팅·종료 코드 4; 이어서 `config.py`·`strategy.py` — 청산 v2 필드 5개, 기본값 = 현 동작).
CRITICAL/MAJOR 만 다시 본다. 방법은 같다(코드 읽기 + 실DB 재생, DB 읽기만, 키움·Anthropic 키를 **환경에서 지운 채**, 3코어). 파일 수정 없음(이 절만 덧붙임).

### 실 CLI 3거래일 재생 — `sw-live <job> --paper --asof <d> --quotes db`

스크립트 `scratchpad/review_integration/cli_rehearse.sh`: 거래일마다 `signal → preopen → open-check → monitor --once → close → eod → status`, 상태 디렉터리 공유,
`KIWOOM_APP_KEY/SECRET`·`ANTHROPIC_API_KEY` 를 `unset` 한 뒤 실행(키움이 한 번이라도 불렸으면 `make_kiwoom_api` 가 예외를 냈을 것이다).

| 실행 | 코드 | 결과 |
|---|---|---|
| 1차(05:31~05:39) | 커밋 `7084b8f` 그대로 | 21작업 전부 rc 0 이지만 **1·2일째 `signal` 이 매번 중단**: `입력 신선도 실패: supply_demand 최신일 2026-10-08 ≠ 신호일 2026-10-01` — 신선도 검사가 전역 `MAX(date)` 와 재생 신호일을 비교해 **과거일 재생은 항상 진입 0**. 또 중단이 rc 0 이라 크론이 못 본다. 2일째 `preopen` 부터 다른 에이전트의 워킹트리 수정이 들어와 3일째 `signal` 은 통과(혼합 버전 — 결과로 쓰지 않음). 원장 시각도 그 지점에서 벽시계(10-10 05:3x) → 재생 시각(10-06 08:35)으로 바뀐다(`SimClock` 도입). |
| **최종(05:43~06:01)** | 워킹트리(신선도 `≤ d`·`SimClock`·rc 4) | **21작업 전부 rc 0, 경보 error/warn 0, 예외 0.** 아래 표. |

최종 실행 원장(재생 시계, 발췌):

| asof | 사건 |
|---|---|
| 10-02 | 07:35 `config_changed`(— → `f3f6a99b`) · `equity` 10-01 1억 · `signal` 후보 [096530] 계획 1 → 08:35 `baseline` · `order_intent` 096530 268@37,500 → `order` submitted → 09:02 `fill` 268@**37,300**(DB 정규장 시가) · `position_open`(손절 33,570 · 목표 44,760 · `cfg_hash f3f6a99b`) → 15:40 `eod` 잠정 99,892,800 |
| 10-06 | 07:35 `equity` 10-02 확정 99,892,800(dd −0.11%) · `signal` 후보 0(보유 종목 DB 종가 결정 — 돌파 실패·MA 없음) → `baseline` → `eod` 잠정 99,986,600 |
| 10-07 | 07:35 `equity` 10-06 99,986,600 · `signal` 후보 0 → 08:35 **`config_changed` `f3f6a99b → b4728a9c`, `pinned_positions: ['096530']`** → `eod` 잠정 100,428,800 |

- 체결가·손절·목표·자산 행이 첫 검토의 하네스 리허설(§1.3)과 원 단위까지 같다 — 하네스가 실 CLI 경로와 같은 결과를 냈다는 확인이기도 하다.
- `signal` 소요 130~148 s(거의 전부 DB 적재) — 시간창 58분 안.
- **10-07 08:35 의 `config_changed` 는 재검증 도중 다른 에이전트가 `StrategyConfig` 에 청산 v2 필드 5개(기본 = 현 동작)를 넣었기 때문이다.** 실 운용에서
  일어날 "코드 갱신 → 설정 해시 변경" 을 우연히 실제로 재현했다: 열린 포지션은 계획 시점 설정(`f3f6a99b`)에 고정된 채 계속 관리됐고, 원장·경보에 남았다(M3 고침이 동작함).

### 항목별 재검증

| 원 판정 | 고침 | 재검증 | 남은 것 |
|---|---|---|---|
| **C1** GA2 → 라이브 1:1 로더 없음, 8개 유전자 결손·의미 다름 | `StrategyConfig.from_genes`/`to_genes`(19 + 수급 3, 모르는 유전자·단계 밖·범위 밖 → 예외), 추세 4성분 분해(`trend_mask`), `pivot_window=0`·`trail_ma=0`·`take_r=∞`(전량 추적, `_close_full`)·`chase=∞`, 끔(None) = NaN 통과(GA 마스크와 같음), `risk_per_trade = max_position_frac`(사이징 정합), `eps_yoy` 를 GA 커널 식(분기 말 분할 조정 주식수, `eps_adjusted_earnings`)으로 교체, `candidate.py`(스키마·`strategy_hash`·`code_hash`·자기 일관성), 3자 패리티 `tests/test_live_genes.py`(`run_backtest` = `simulate_minervini` = `ga_fast.engine_events`, 전부 끔·돌파 없음+추격 ∞ 포함) + EPS 패널 = `ga_run.eps_features` 테스트 | **해소.** 표 §2 의 결손·의미 차이가 전부 코드·테스트로 닫혔다. 커밋본의 `load_strategy_config` 는 `config.load_candidate`(없는 이름)를 찾아 인계 파일을 **관대한 경로로 읽었다**(스키마·해시 검사 우회) — 워킹트리에서 `CANDIDATE_MARKERS`/`genes` 가 있으면 `candidate.load_candidate` 로 강제하도록 고쳐짐(확인). | **MAJOR(인계 전): `ga2_run.py` 는 여전히 옛 모양(`config` 키, `"inf"` 문자열, `fitness` 가 문자열)으로 `live_candidate.json` 을 쓴다** → `--strategy-config` 에 그대로 주면 `CandidateError`(빠진 키) 로 **크게 실패**한다(올바른 실패). 변환(`write_candidate(path, genes_from_legacy(doc), fitness=…, window=…, valid_from=…)`)을 하는 명령·런북 절차가 없다. GA2 결과가 나오면 이 한 단계를 스크립트로 둘 것. |
| **M1** 수급·실적 지연 → 후보 조용히 0 | `_freshness`: 유니버스 F·PEAD 결측 ≤ 10%, 실적 패널 비어 있지 않음, `supply_demand` 최신일 = d. 실패면 보유 결정·자산·섀도는 하고 진입만 중단 + error | **해소(워킹트리 기준).** 커밋본은 전역 `MAX(date)` 비교라 재생을 막았다(위 1차 실행) — 워킹트리에서 `MAX(date) WHERE date ≤ d` 로 고쳐짐, 중단 시 **rc 4**(`RC_ENTRIES_ABORTED`) 로 크론이 구분한다. 최종 실행 3일 모두 통과(`freshness.ok`). | 워킹트리 rc 4 계약에 맞춰 **`test_live_runtime_adversarial.py` 2개가 아직 0 을 기대해 실패**(`test_stale_signal_data_still_settles_held_positions`·`test_f4_real_holiday_with_stale_open_does_not_fill_paper` — 둘 다 의도된 중단 경로, 동작 회귀 아님). 커밋 전 갱신 필요. |
| **M2** 재생·리허설 불가(키움 필수, `d=None` 고정) | `RestQuoteProvider` 지연 생성(첫 시세 전엔 키움 안 부름), `DBQuoteProvider`(minute_bars 정규장 봉, `date=asof`), `--quotes db --asof`, 신호일 = `asof` 직전 DB 거래일, 워킹트리에서 `SimClock` + 작업별 예정 시각(`REPLAY_JOB_TIMES`, `--clock`) | **해소.** 키·토큰 없이 실 CLI 로 3거래일 21작업 재생, 결과가 하네스와 일치. 재생 시계 없이는 매 작업이 그날 전체 봉을 봤을 것(커밋본) — 워킹트리에서 고쳐짐. | — |
| **M3** 열린 포지션이 전역 설정에 다시 묶임 | `PositionRec.cfg`/`cfg_hash`(계획 시 고정), `_cfg_of` 를 상태 복원·DB 종가 결정·섀도까지, `config_changed` 사건·경보, 원장 모든 사건에 `cfg=<8자리>`, `--strategy-config` | **해소** — 실제 설정 해시 변경(위 10-07)에서 고정이 동작했다. | MINOR: 고정 dict 에는 **나중에 추가된 필드가 없어** `_strategy_from_dict` 가 그 필드를 **지금 코드 기본값**으로 채운다. 이번 v2 필드는 기본값이 현 동작이라 무해했지만, 나중 커밋이 기본값을 바꾸면 옛 포지션이 조용히 새 규칙을 탄다. 또 필드 하나 추가할 때마다(값이 같아도) 해시가 바뀌어 `config_changed` 경보가 난다. → 고정 dict 에 없는 필드는 "그 필드가 생기기 전 동작" 값으로 채우는 이주표(또는 `StrategyConfig` 에 `schema_version`)를 둘 것. |
| **M4** 보유 종목 미보정 종가 무검사 | `_decide_on_db_close`·`_shadow_settle`: `code ∉ patched_codes` 면 결정 보류 + warn | **해소**(코드 확인; 실데이터 노출 0 이라 재생에선 발동 안 함 — 프로젝트 테스트가 덮음). | — |
| **M5** 섀도 종가 결정이 실포지션과 다른 가격·시각 | `_shadow_settle`: 다음날 `signal` 의 DB 보정 일봉으로 `on_bar`, `eod` 의 섀도 처리 제거, 09:02 시세 없을 때 DB 시가 진입 | **해소**(코드 확인; 재생 3일 거부 0 이라 섀도 미발생 — 프로젝트 테스트가 덮음). 상태 줄이 "순수익 기준 — 사전등록 지표 아님" 을 명시(m1 도 정리). | — |
| **M6** 아침 적재 전체 `read_prices` + 재시도 없음 | 3회 재시도(5·15 s), `storage.read_prices_wide`(서버측 커서, 메모리 절약) 추가 | **부분.** 재시도는 들어갔다. `read_prices_wide` 는 만들어졌지만 **`features.load_inputs` 가 아직 `read_prices` 를 부른다**(적재 130~148 s, 전 기간 전 종목) — 시간창 안이라 페이퍼엔 무해. | MINOR(성능): `load_inputs` 를 `read_prices_wide` 로. |
| **M7** `stop_cap > 0.10` 사이징 불일치 | `from_genes` 가 `risk_per_trade = max_position_frac` | **해소.** | — |

전체 테스트(커밋 `7084b8f`): **1,272 통과 · 6 건너뜀**(93 s, 3코어). 워킹트리(재검증 끝 시점) 라이브 테스트: 243 통과 · **2 실패**(위 rc 4 계약 — 테스트 미갱신).

### 최종 판정

**PAPER: GO** — 코드 기본 설정(`StrategyConfig()`, `--strategy-config` 없이)으로. 실 CLI 가 키움·API 키 없이 실DB 로 3거래일을 끝까지 돌았고(21작업 rc 0),
체결·손절·목표·자산이 독립 하네스와 일치하며, 수급·실적 지연은 이제 rc 4 + error 로 드러나고, 설정 교체는 열린 포지션을 건드리지 않는다는 것을
실제 해시 변경으로 확인했다. 전제(첫 페이퍼 날 전):

1. 워킹트리 수정(신선도 `≤ d`·`SimClock`·인계 파일 라우팅·rc 4)을 **커밋**하고, rc 4 를 기대하도록 adversarial 테스트 2개를 고친 뒤 전체 테스트 녹색.
2. 크론 감시가 rc 4 를 "신규 진입 중단" 으로 다루게(런북 §2 표에 이미 있음 — 크론 래퍼·데드맨이 0 만 정상으로 보게).
3. 선행 검토의 페이퍼 전제 그대로: `holidays.txt` 2026 완비, 첫날 `ka10001` 필드 사람 확인 1회, 한 모드만 운용.

**GA2 config 를 페이퍼에 싣는 것: 아직 NO-GO** — 로더·패리티는 준비됐지만 (a) `ga2_run.py` 출력 → 인계 스키마 변환 단계가 없고(지금은 크게 실패한다 — 올바르다),
(b) 고정 dict 의 새 필드 기본값 문제(M3 잔여)가 롤링 재적합의 소급 변경 구멍으로 남는다, (c) GA2 결과 자체가 아직 없다(`ga2/STATUS.txt` 실행 중).
**실계좌: NO-GO**(선행 REVIEW_RUNTIME §9 조건 그대로).
