# swing-it — Claude 작업 지침

한국 주식(KOSPI·KOSDAQ) 리서치 레포. main = kqflow TUI(`sw-flow`·`sw-ledger`) + 알파 심사 프레임워크(공개, MIT).
사용자의 실제 연구(스윙·미너비니·진화 GA·모의매매)는 대부분 **`feature-combined-strategy`**(이하 **fcs**,
워크트리 `~/git/swing-it-feature-combined-strategy`)에 있고 main 에 머지되지 않았다. 아래에서 "(fcs)" 표시 경로는 main 에 없다.

## 명령 (검증됨)
- 테스트: `uv run --extra dev pytest -q` — DB·네트워크 불요. fcs 는 `uv run --with numba --extra dev pytest -q`(약 3분).
- 린트: `uv run --extra dev ruff check src/ tests/ research/ scripts/ examples/` (CI 와 같은 범위)
- 가드레일 린트: `uv run python scripts/check_guardrails.py` (CI 단계). 규칙을 고치면 `tests/test_check_guardrails.py` 에
  위반을 주입하는 테스트를 같이 넣는다(통과만 보는 테스트는 규칙이 죽어도 초록이었다 — GUARDRAILS §4.5).
- TUI 수정 뒤: `uv run python scripts/verify_merge.py <워크트리>` (수 분).
- DB(TimescaleDB `kr_quant`, **읽기 전용**): `set -a; . ~/git/quant-airflow/.env; set +a` 뒤 `$KR_QUANT_DB`,
  psycopg2 는 `--extra pg`, 연결은 `con.set_session(readonly=True)`. 수집·스키마 변경은 quant-airflow 소관 — 거기에 요청한다.
- 모의매매 (fcs): `uv run --with numba --extra pg python research/paper/leader_paper.py` — 하루 한 번,
  DB 확정(06:30) 뒤 장 전 07:30~08:50. 세션 시작 때 장부 마지막 날짜를 보고 빠진 거래일이 있으면 돌린다.

## 브랜치·커밋
- 커밋은 브랜치까지만. push·머지·실주문은 사용자가 명시할 때만.
- **origin 은 PUBLIC 이다.** 배포북(재현 성적표·적합된 비중·유니버스 문턱처럼 따라 할 수 있는 레시피)은 공개 브랜치에 올리지 않는다
  (2026-08-27 결정, `docs/alpha-research.md` §8 끝 — 배포북을 공개 푸시했다가 회수·히스토리 재작성한 사고). fcs 를 main 에
  머지하기 전에 `docs/individual-trader-method.md`·`deploy/p1_satellite.json`·`research/paper/` 의 공개 여부를 사용자와 정한다.
- 메시지: 한국어 conventional — `research(prereg|evolve|paper|verdict): …`, `fix(…)`, `perf(…)`, `docs(…)`.
- `perf` 변경은 결과가 비트/바이트 동일함을 확인하고 메시지에 적는다(예 5074842 "결과 같음", da10c58 "바이트 동일").
- 미너비니 책 원본(저작권)·`.orca/` 는 커밋 금지. `.orca/` gitignore 는 main 계열에만 있으니 fcs 에서는 `git add -A` 대신 경로를 지정한다.

## 연구 규율 — 정본 `docs/GUARDRAILS.md` (새 실험 설계 전에 읽는다)
- **사전등록 순서:** 후보·절차·합격 바를 결과 보기 전 문서에 적고 `research(prereg): …(결과 전)` 커밋 → 실행 →
  `(검증 전)` 결과 커밋 → 검증 반영 `정정 G<세대>-<n>` 커밋. 결과를 본 뒤 사전등록 문단은 고치지 않는다 —
  고칠 일이 생기면 새 정정 문단을 쓰고, 결과를 이미 봤는지 공개한다.
- **봉인 구간 2024-01~ 은 20세대에서 소모됐다.** 2024~26 숫자는 "오염, 증거 아님"으로만 쓴다. 설계 선택은
  2017~2023(진화 폴드 1~4 = 2020~23), 깨끗한 확인은 전진 모의매매뿐이다.
- **널 = 같은 메뉴의 무작위 설정**(95 백분위) + 팩터 회귀 알파 t>2. ">0" 바나 순수 노이즈 널은 종목군 기울기로
  10~27% 가 그냥 통과한다(fcs `REVIEW_GA2_B.md`).
- **세대·실험 보고 기본 열:** 비용 1·2배(판정은 2배, 왕복 41.5bp 기준), 상위 5 종목 제거, 노출 맞춘 국면 타이밍
  시장 대조, 연도별, 체결 수, 직전 세대·KOSPI 대비(정정 G22-1).
- 기각된 정의를 같은 KRX 데이터로 파라미터만 바꿔 다시 돌리지 않는다(낚시). 다음은 독립 표본 또는 전진.
- "알파"는 실행 가능한 형태로만 주장한다. 롱온리, 매도 거래세가 포함된 비용, 자본 1~3억(fcs `docs/individual-trader-method.md`)의 시가 주문이 ADV 용량 안이어야 한다.
  손절 폭을 좁혀서 얻은 개선은 의심한다. 갭과 호가 때문에 손절가 그대로 체결되지 않기 때문이다. 실제로 좁은 손절은 대장주를 다 털렸다(21세대 동기).
- 체결 수가 하한에 못 미치면 VOID·판단 불가로 적는다. 거래를 줄여 거래당 숫자를 올리는 방향은 피한다 —
  사용자는 거래 수 × 거래당을 본다.
- 모의매매 팔(leader_A 등)은 도중에 규칙을 바꾸지 않는다. 바꾸려면 새 팔·새 장부로 사전등록한다.
- 원장(`TRIALS.jsonl`·`RUNS.jsonl`·`research/paper/*.jsonl`)은 코드만 덧붙인다. 손으로 고치지 않는다(`.claude/settings.json` 의 `Edit` deny —
  편집 도구·`sed`·`>>` 리다이렉트를 막고, 파이썬 스크립트의 덧붙이기는 막지 않는다).
- 가격·실적·주식수·지수는 정문(`storage.read_prices`·`read_earnings`·`read_shares`·`read_index`)으로 읽는다. 직접 SELECT 는
  린트 (d)(g)(j)(k)(o) 위반이다(2026-10-12 기준선 0건).

## 데이터 함정 (kr_quant) — 대부분 정문·린트가 강제한다(fcs `scripts/check_guardrails.py`)
- **KRX 애프터마켓 오염(2026-09-14~, 16:00~20:00; 처음엔 NXT 로 오인):** 일봉 고저종·거래량, 수급 close·flu_rt·acc_trde_qty. 시가만 정상.
  → 정문 `read_prices`·`read_prices_wide`·`read_supply_demand` 기본 `session="regular"` / 린트 (g)(j). 원값은 `session="raw"` 명시.
  판단: 미보정 행 수(`attrs["krx_session"]`)를 보고 `strict=True` 를 쓸지 정한다(분봉은 하루 ~1,400 종목뿐). 글로는 못 막았다(730def6).
- `minute_bars.ts` = 시간대 없는 KST → 린트 (m). 정규장 = `krx_session.SESSION_OPEN/CLOSE`(15:32). main 로드맵의 "오진" 문구는 틀렸다.
- 실적 as-of → `read_earnings(with_known=True)` 의 `known_date`·`event_date` / 린트 (l)(`avail_date` 키는 `# legacy-asof:`). 2019-02-28 전은 avail_date.
- `trade_value` = 백만원 → `swing_it.units.eok()` / 린트 (n) / `read_prices` 단위 카나리아. TUI 리포트 값은 억 단위.
- 주식수 → `read_shares(with_asof=True)` / 린트 (k). 지수 → `read_index`(×100 정수 + `level`, 2024-01-26 전은 `proxy=` 필수) / 린트 (o).
  대용 지수의 품질은 부르는 쪽 책임(재구축 대용 −4%/년 어긋남 → 거짓 통과). 커버리지 → `storage.COVERAGE`·`require_coverage`
  (뉴스 2023-03-15·공매도 2024-05-09 — 정문 없음, 직접 쓰면 시작일을 스스로 확인).
- **`daily_bars.close` 는 원 종가가 아니다**(분할 소급) → `krx_session.read_raw_close`(새 코드 `aftermarket_fix=True`)·
  `market_cap_asof(_bulk)`(옛 정의 `legacy=True`). 판단: 원 종가 × 연말 DART 주식수는 분할 뒤 수개월 틀린다(거울상) — 패널이면
  `split_consistent_cap`(`asof=` 필수). 21~26세대 설계 숫자는 시총 구성에 따라 5~15% 흔들린다(정정 G28-2·4).
- 판단(코드로 못 옮김): 수급 주체 열은 순매매 **수량**(× 종가는 근사), 네 주체 합 ≈ 0 이라 `etc_corp` 를 빼면 안 되고 `institution` 은
  합계 열을 그대로 쓴다. 0 은 "0 또는 미보고", 폐지 종목은 기관·외국인만(나머지 NULL). `stocks` 업종·시장은 현재 스냅샷이다.

## 실행 환경 함정
- numba 다중 작업자는 `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1` 환경변수가 필수다. 코드의 `set_num_threads(1)` 은 안 먹어서
  16코어에 스레드 190개가 경합했다(9.7초 → 174초). 긴 계산은 16코어를 작업자로 채운다(사용자 "시피유가 노는데?").
- **`pkill -f`·`pgrep -f "<문자열>"`·`ps | grep "[x]yz"` 금지** — 같은 문자열이 든 내 셸이 죽는다(3회 반복).
  띄울 때 `echo $! > <pidfile>`, 대기는 `kill -0 $(cat <pidfile>)`, 종료도 pidfile 로만 한다.
- 유전자 메뉴 튜플 줄 끝에 sed 로 `# 주석` 을 붙이지 않는다 — 뒤 항목이 주석에 삼켜졌다(2회). 메뉴를 편집한 뒤에는
  `tests/test_pattern_ga_fwm.py`(fcs, 변형별 DIM·GENE_NAMES 해시 회귀)를 돌린다.
- 큰 캐시는 세션 scratchpad 나 `~/of80/` 에 둔다. /tmp 는 할당량(EDQUOT)에 걸렸다 — 만들기 전에 `du` 로 확인한다.
- 긴 실행의 남은 시간은 첫 진행 줄을 실측해서 다시 계산한다(추정이 두 번 틀렸다).

## kqflow TUI·리포트 (main, `src/swing_it/tui/`·`scripts/sector_flow.py`)
- 화면은 수급 사실만 말한다. 리포트에는 구간 수익률 4개뿐이라 경로를 모른다. 그래서 눌림목·반전 **판정**을 점수화하는
  제안은 하지 않는다(`reversal_flags` 의 ⚠ 는 판정 불가 라벨이다).
- 비율 열은 분모 게이트를 같이 설계한다. 분모가 작으면 숫자 대신 결측을 낸다. 문턱은 목록 안 몫으로 잡는다.
  이 결함이 다섯 번 나왔다(`flow_view.py` 최근집중 주석).

## 사용자 매매 원칙 (규칙·유전자 설계 때)
- 수급 신뢰 순서: 기관 순매수만으로 충분하고 외국인 동반은 요구하지 않는다. 기관 안에서는 투신·연기금(`INST_DETAIL`)을 우선한다.
- 손실은 짧게, 수익은 길게. **시간 손절 금지** — 조건 손절(돌파 실패, 구조선 종가 이탈, MA 이탈)만 쓴다.
- 손실 한도 −5% 는 **비중으로** 건다. 손절 폭을 5% 로 좁히는 게 아니다. 16~20세대가 이를 좁은 % 손절로 옮기는 바람에
  2024~26 대장주를 다 털렸다(fcs META_EVOLUTION.md 21세대 동기).
- 이익 잠금·본전 이동·촘촘한 추적으로 승자를 일찍 자르면 원칙 위반이다(고점 반납도, 털기 손절도 사용자가 차트로 짚은 실수다).
- EPS 상승은 필수 조건이다. GA 유전자로 끄지 않는다. 장중 매매(장중 손절·매도)가 전제다.
- 정석 진입: 급등 뒤 **하락** 쐐기, H0 를 A 로 세고 바닥 D 에서 산다. 상승 쐐기는 오답이다.
- 미너비니 근거는 fcs `research/logs/sepa_explore/MINERVINI_RULES.md`(쪽 번호)와 책 그래프 `/home/young/minervini-graph/`
  (`graphify query`)에서 찾는다. 사용자는 "왠만하면 미너비니를 따라"라고 했다.
- 파라미터 하나를 고정하자는 제안을 반복하지 않는다. 고정·사전등록하는 것은 절차와 합격 바이고, 파라미터는 워크포워드로 굴린다.
  최적화는 확률적이다 — 한 경로의 극값·경계가 아니라 고원을 고르고, 거래별 R 분포로 본다.

## 보고·검증 방식 (사용자가 반복 요구한 것)
- **IMPORTANT — 결과는 차트와 함께, 나오는 즉시 보여준다.** 차트에는 봉·거래량·MA·진입/청산·손절선을 그린다.
  검증 전이라도 '검증 전' 표시로 바로 게시하고, 검증 뒤 같은 URL 을 갱신한다. 긴 실행 중에도 중간 결과를 보여준다.
  그림은 `swing-visualizer` 에이전트가 만들고, 세대 숫자는 `research/logs/sepa_explore/evolution_ledger.json`(fcs)에 쌓는다.
- 실험마다 시니어 퀀트 관점 리뷰(`quant-reviewer`)와 하이쿠 스웜(`haiku-verifier` 10~20개)을 함께 띄운다. 스웜은 거래·사건을
  원 DB 로 개별 재계산한다. 리뷰의 CRITICAL/MAJOR 를 반영하기 전에는 결론을 보고하지 않는다.
  스웜 프롬프트에는 그 엔진의 체결 규약(진입일 모호 처리, data_end, 손절 하한, 2023-01-25 호가 개편 등)을 미리 적는다.
  안 적으면 입자가 규약을 오류로 오판한다. 입자가 짚은 것은 캐시 배열로 직접 다시 확인한다.
- 세대를 보고할 때는 연대기 표(세대별 숫자, 직전 세대 대비, 시장 대비)를 같이 보인다. 개선 없이 세대만 쌓지 않는다.
  2~3세대가 나아지지 않으면 축을 바꾸거나 멈춤 규칙을 쓴다(사용자 "진화가 아니라 퇴화를 하고 있네").
- 질문으로 멈추지 않는다. 진단 → 계획 → 병렬 에이전트 → 통합까지 스스로 끝낸다. 먼저 기존 판정 문서를 읽어 닫힌 가설은 다시 열지 않는다.

## 문서 위치
- `docs/GUARDRAILS.md` — 연구 정석·완료 정의·사전등록 템플릿. 새 알파나 실험 전에 읽는다.
- `docs/alpha-research.md` · `research/logs/*/VERDICT.md` — 기존 판정 전문. 가설을 세우기 전에 기각 목록을 확인한다.
- `research/TEMPLATE.md` — 실험 파이프라인 규약(src 는 research 를 import 하지 않는다 등).
- (fcs) `research/logs/sepa_explore/META_EVOLUTION.md` — 진화 세대 절차와 세대별 사전등록·결과·정정. 새 세대 전에 마지막 2~3세대를 읽는다.
- (fcs) `research/logs/sepa_explore/` 의 `*_RESULTS.md`·`REVIEW_*.md` — 실험별 결과와 리뷰.
- (fcs) `docs/individual-trader-method.md` — 현재 완결 매매법(코어 지수 ETF + 위성).
- (fcs) `research/paper/` — 모의매매 장부(덧붙이기 전용, `*_VOID_*` 는 무효 보관본).
