# combined autoresearch — 세 레포(swing·scalp·daytrade) 부품을 합친 KRX 탐색 (사람이 고치는 파일)

swing v4 루프(`research/autoresearch/program.md`)와 같은 방식이다. 숫자 파라미터는 평가기가 폴드마다 고르고, 너는 **구조**만 바꾼다.
목표는 전진 모의매매에 올릴 후보 하나를 고를 **재료**를 만드는 것이다. 후보는 루프가 끝난 뒤 메인 세션이 `select.py` 로 고른다 — 마지막 keep 이 후보가 아니다.
사전등록·합격 바는 `PREREG.md`, 설계는 `docs/superpowers/specs/2026-10-25-combined-autoresearch-design.md` 다.
시작 전에 이 파일, `PREREG.md`, `prepare.py`(읽기만), `train.py` 를 읽는다.

## 파일
- `prepare.py` — 고정 평가기이자 진입점. 워크포워드(학습 2년 → 시험 1년, 시대 A 2011~2015 5폴드 · 시대 B 2020~2024 5폴드),
  학습 구간 라틴 하이퍼큐브 48점 → 고원 선택, 엔진 체결(**신호일 종가 진입**), 비용 2배, 미래정보 검사, 쏠림 진단, 부품 제거 검사, keep 판정,
  시행 상한. **고치지 않는다**(태그·sha 매니페스트로 검사 — 바뀌면 평가가 거부된다).
- `train.py` — 네가 고치는 유일한 파일. 계약·허용 목록·V 의 키는 파일 머리말에 있다(numpy·pandas·math·functools·itertools·collections 만).
- `results.tsv` — 네 메모(git 추적 안 함). 머리줄: `commit	score	beats_best	eligible	oos_trades	ev_net_2x	status	description`.
  시행 수의 정본은 평가기가 쓰는 `RUNS.jsonl` 이다.

## 실험 한 번
1. `train.py` 를 아이디어 하나로 고친다 → `git commit -m "exp: <한 줄>"` (`train.py` 만 add)
2. `uv run python research/combined/prepare.py > run.log 2>&1` (작업 트리 루트; 30분 넘으면 죽이고 crash)
3. `grep -E "^(score|void|beats_best|beats_detail|lookahead_check|folds_pos|oos_trades|payoff_ratio|ev_net_2x|ev_net_1x|top1pct_removed|top_stock_removed|top10d_removed|news_folds_score|ablation_scalp|ablation_daytrade|eligible|trials|stop):" run.log`
   비었으면 `tail -n 50 run.log` — 쉬운 오류면 고쳐 다시, 아니면 crash 로 적고 넘어간다. "허용 목록 위반" 이면 train.py 를 고친다.
   "상한 도달 — 평가 거부" 면 루프가 끝난 것이다(아래 '멈춤'). **crash 도 시행 하나로 센다**(원장에 crash 행). crash 때는 stderr 끝 20줄만 보인다.
4. `results.tsv` 에 한 줄 덧붙인다(모든 시도 — discard·crash 포함).
5. **keep 은 평가기가 판정한다:** `beats_best: True` 면 `uv run python research/combined/prepare.py --accept` 로 기준을 올리고 커밋을 남긴다.
   아니면 `git reset --hard HEAD~1`.
   `beats_best` = 직전 최고(BEST) 대비 Δscore ≥ 0.005 ∧ **체결이 바뀐 폴드(1개 이상) 중 과반에서 alpha 개선** ∧
   쏠림 셋(`top1pct_removed`·`top_stock_removed`·`top10d_removed`) 각각이 BEST 의 같은 값보다 0.01 넘게 나빠지지 않음.
   체결이 안 바뀐 폴드는 세지 않는다 — 2개 폴드(2023·2024)에서만 일하는 뉴스 부품도 keep 될 수 있다.
6. 다시 1.

## 점수와 출력 줄이 뜻하는 것
- score = 10개 시험 폴드의 **노출을 맞춘 선택 알파**(전략 로그 성장 − 전날 노출 × KOSPI 일수익 로그 성장, 연율)의 평균 − 0.5 × 표준편차. 비용 2배.
  쉬는 해·시장 타이밍은 0 이다. 폴드별 값은 일부러 보여주지 않는다. 음수에서 시작하는 것이 정상이다.
- `payoff_ratio`(손익비)·`ev_net_2x`·`ev_net_1x`(거래당 순기대값, 비용 2·1배)를 승률보다 먼저 본다.
- 쏠림 셋: 순수익 상위 1% 거래를 빼고 / 순수익 합이 가장 큰 한 종목을 빼고 / 일별 선택 알파 상위 10일을 0 으로 두고 다시 낸 score.
  상위 1% 제거는 그 거래의 **신호 칸 하나만** 끈다(부드러운 제거) — 같은 종목을 다음 날 다시 살 수 있다. 종목째 빼는 것은 top_stock 이다.
- `news_folds_score` = 뉴스가 있는 2023·2024 시험 폴드만의 score.
- `ablation_scalp`·`ablation_daytrade`: 그 레포 입력을 중립값(갭·호가 비율 0, 사건 갭 False, 뉴스 개수 0, news_known False)으로 바꾸고
  같은 폴드 파라미터로 다시 돌린 결과. `changed` = 체결 집합이 바뀐 비율, `dscore` = 원래 score − 제거 score(daytrade 는 2023·2024 폴드만).
  `ok` = changed ≥ 0.05 ∧ dscore > 0 ∧ `trades_ok`(제거 실행의 작동 폴드 체결 합 ≥ 원래의 50% ∧ 폴드마다 ≥ 5).
- **중립 입력 = 통과(pass-through).** 부품은 중립값(갭·tickp 0, 사건 갭 False, 뉴스 0, news_known False)을 받으면 아무것도 막거나 바꾸지 않아야 한다.
  `cand & news_known` 처럼 중립 입력에서 거래를 멈추는 문은 `trades_ok` 에 걸리고, 사람 열람에서 제외된다(`neutral_input`).
  중립 입력(전부 0·False)을 감지해 다르게 행동하는 것도 제외 사유다.

## "세 레포 부품"과 eligible
- **swing:** leader21 골격(추세 템플릿·RS·N일 돌파·청산 메뉴·하드 손절·불타기·국면), 수축·실적 신선도 같은 특성. EPS 상승은 평가기가 강제한다.
- **scalp:** 야간 갭 지속성(gap, 원주가 비용항 tickp 를 빼고, 창 w/2·w·2w)을 돌파 후보 안 순위 키나 거르개로. 제한폭 절반 이상 갭(gap_event)은 대칭으로 뺀다.
- **daytrade:** [특징주] 헤드라인 거부(n_feat, 신호일 전후 창), 기사 뒤 이미 반영됐는지(일봉 가격 이동) 검사. `news_known` 이 False 면 개수 0 은 "모름"이다.
  뉴스 구간(2023-03-15~)에서 아카이브가 없거나 잘린 종목은 평가기가 uni 를 꺼 둔다(PREREG 정정 0) — 그 종목은 그 구간에 후보가 아니다.
- `eligible: True` = scalp·daytrade 부품 제거 검사가 둘 다 ok(그 부품이 실제로 체결과 점수를 바꾼다). 장식으로 읽기만 하는 부품은 걸린다.
- **자격 없는 keep 도 keep 이다.** 자격은 keep 판정에 쓰이지 않는다 — 다음 시도의 디딤돌이다. 기준선(`train.py` 처음 상태)은 갭·뉴스를 안 써서 자격이 없다.
  다만 후보는 자격 있는 시도에서만 나오니, 세 레포 부품이 모두 일하는 구조로 가야 한다.

## 하지 말 것(어기면 결과 무효)
- `prepare.py`·`data.py`·`gaps.py`·`news.py`·`select.py`·엔진·데이터·의존성·캐시·원장(`RUNS.jsonl`·`BEST.json`·`MANIFEST.json`) 변경, 태그 조작,
  `select.py`·`--reference` 실행, 확인 구간(2025~)·잠금 절반 추론.
- `REFERENCE.json`(기준선 점수)·`RUNS.jsonl`·`BEST*.json`·`last_run.json`·캐시·원장 사본을 어떤 방법으로든 읽기(Bash 포함). 출력 줄만 본다.
- tickp 로 원주가를 되살려 수정 종가와 비교하기(미래 수정 계수 — 미래정보다, 사람 열람 제외 사유).
- 허용 목록 우회(문자열로 모듈 이름 만들기, 내장 함수 꺼내기 등), DB·파일·네트워크 접근.
- 호출 사이 상태 숨기기: 모듈(np·pd 등)에 속성·항목 넣기, 모듈 이름을 값으로 넘기기, `print`(허용 목록이 막는다). 같은 V·파라미터면
  언제 몇 번째로 불러도 같은 신호여야 한다 — 평가기가 새 프로세스의 신호와 비교하고, 다르면 void 다.
  **속성 대입은 전부 금지다**(`df.columns = …` 도 걸린다 — `df = df.set_axis(cols, axis=1)` 처럼 새 객체를 만든다).
- `cfg(x)` 는 x 만으로 정한다. 평가기는 signals 를 부른 적 없는 새 프로세스에서 cfg 를 계산해 그 값으로 채점하고, 평가 프로세스의 cfg 와
  다르면 void 다. **학습 점 하나라도 cfg 가 예외를 내면 평가 전체가 crash 로 끝나고 시행 하나로 센다**(signals 예외는 그 점만 버린다).
- `signals` 는 **정확히 dict** `{'cand': …, 'score': …}` 를 돌려주고, 두 값은 **정확히 `np.ndarray`**(bool·정수·실수 dtype, V['C'] 와 같은 모양)여야
  한다. DataFrame·하위 클래스·`__array__` 객체·object dtype 은 거부된다(학습 점은 그 점만 버리고, 시험 폴드는 평가 전체 crash — 시행으로 센다).
  평가기는 받는 즉시 복사해 그 사본만 쓰므로, 돌려준 배열이나 받은 x 를 뒤에서 고쳐도 채점에 닿지 않는다.
- 미래정보 검사는 고른 파라미터뿐 아니라 학습 점 표본(폴드마다 12개, 비밀 시드 — 어느 점인지 알 수 없다)의 신호도 본다. 표본 점은
  고원 선택 때 실제로 쓴 신호가 새 프로세스 신호와 같아야 한다. 같은 x 면 언제·어디서·몇 번째 호출이든 같은 신호(같은 예외)여야 한다.
- 모듈 최상위에는 설명문·import·def/class·리터럴 대입(`_C: dict = {}` 등)만 둔다. 계산·호출은 함수 안에서 한다.
  np·pd 등 모듈에 닿는 객체를 바꾸는 메서드(append·update·setdefault·pop·sort …)도 금지다(`np.sort(a)` 같은 함수 호출은 된다).
- 스스로 잰 구간 성적을 예외 메시지·출력으로 내보내기(성공 때 stderr 는 평가기 로그만, 학습 점 실패는 예외 종류만 보인다).
- 사후 지식: 아는 승자 종목·업종·연도를 겨냥하는 규칙(날짜·코드를 받지 않는다 — 배열 위치로 같은 짓을 하지 않는다).
- 미래 정보: t 신호는 t 종가(뉴스는 t 15:10)까지의 값만(평가기가 절단 재계산으로 검사 — 걸리면 void).
- 숫자 맞추기: 상수를 손으로 바꿔 점수를 올리지 않는다 — 숫자는 `params` 범위로 내고 평가기에 맡긴다. 범위를 한 점으로 좁히는 것도 손 맞추기다.
- **뉴스 부품에는 숫자 파라미터를 두지 않는다.** `params` 에 뉴스 관련 이름(창 길이·개수 문턱 등)을 두지 않는다 — 2023 폴드의 학습 창에는 뉴스가 없어
  고원 선택이 무의미하다. 창 길이 같은 것은 train.py 에 고정 규칙으로 쓴다.
- **시간 손절**·**이익 잠금·본전 이동·촘촘한 추적**·**좁은 손절**. 청산은 메뉴(MA w 20~250, 어제까지 w일 최저 저가 w 20~120)와
  하드 손절(≥ 10%)뿐이다. 손실 한도는 비중(`risk`)으로 건다.
- 거래를 줄여 거래당 숫자를 올리는 방향(체결 < 5 인 폴드 2개 초과, 전체 < 150, 연 < 20 이면 void).
- **은퇴한 e104063 구조(시장 대비 수축 + 실적 신선도 조합)와 그 파라미터 변형을 다시 내지 않는다.** 그 조합은 swing 루프에서 은퇴했다(PREREG 정정 3).

## 사용자 매매 원칙(설계 공간)
- EPS 상승은 필수다(평가기가 강제). 수익은 길게, 손실은 짧게. 미너비니를 웬만하면 따른다.
- 정석 진입: 급등 뒤 **하락** 쐐기(고점 H0 = A, 바닥 D 에서 매수). 상승 쐐기는 오답이다.
- 승률이 아니라 손익비·기대값을 본다. 승률이 높은데 손익비 < 1 이면 승자를 일찍 자르는 청산부터 의심한다.

## 이미 닫힌 것(같은 것을 파라미터만 바꿔 다시 내지 않는다)
- 추세 템플릿 + N일 돌파 + RS + EPS(= swing 기준선, 23세대 C): 2009~16 에서 KOSPI 보유보다 연 −1.4pp, 상위 5 종목 빼면 알파 없음.
- 거래대금 종목군·θ 70·슬롯 20(leader_M): 2009~16 연 −5.7%. 쫓지 않기 + 10일 저가 구조 손절(leader_P): 무작위 솎기 널 중앙보다 낮음.
- RL 순위(leader_P_RL): 무작위 점수와 같음. 200일선 국면 청산: 2009~16 에 보유 대비 −6.8pp/년.
- 11~18세대: 진입을 느슨하게 해 거래를 늘린 것 전부 거래당 ≤ 0. 좁은 % 손절: 대장주를 다 털림.
- swing 루프 280 시행: e104063(시장 대비 수축 + 실적 신선도) 은퇴 — 위 금지 목록.
- scalp: 야간 갭 선택만으로 매일 매매는 비용을 못 넘는다(선택 초과 +31bp 는 안정이나 비용 절반). 그러니 갭은 **스윙 돌파 안의 순위·거르개**로 쓴다.
- daytrade: 장중 구조 ~1,000 개, BEST 비용 뒤 −2.4bp, 한 종목이 알파 44%. 그러니 뉴스는 **거부·확인 거르개**로 쓴다.

## 멈춤
- 평가기가 시행 상한을 센다: **고유 train sha 150회**(crash 포함), 또는 **첫 keep 이후 고유 sha 30회 연속 keep 없음**
  (keep = `beats_best: True` 평가 — `--accept` 를 건너뛰어도 keep 으로 센다. BEST 바는 사본에도 있어 BEST.json 을 지워도 되돌아가지 않는다). 둘 중 먼저 오는 쪽에서
  `stop: True` 가 찍히고, 다음 평가는 "상한 도달 — 평가 거부"로 끝난다. 같은 train.py 를 다시 돌리는 것은 시행을 늘리지 않는다.
- 그 전에는 사람에게 묻지 않는다. 실험 사이에 말하지 않는다. 10 실험마다 한 번만:
```
=== 진행 (N 실험) ===
기준선 X → 최고 Y (쏠림 셋 a/b/c) · keep K / discard D / crash C / void V · 자격 있는 시도 E
배운 것: <한 줄>
```
아이디어가 떨어지면 `research/logs/sepa_explore/MINERVINI_RULES.md`, scalp-it `research/autoresearch/VERDICT.md`, daytrade-it
`docs/research/2026-10-24-autoresearch.md` 에서 새 구조를 찾는다. 후보 고르기(`select.py`)는 메인 세션이 한다 — 너는 하지 않는다.
