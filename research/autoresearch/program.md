# autoresearch — 가드 붙인 KRX 스윙 탐색 (사람이 고치는 파일)

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) 의 루프를 이 레포 가드레일 안에 넣었다. 숫자 파라미터는
[dietmarwo/autoresearch-trading](https://github.com/dietmarwo/autoresearch-trading) 처럼 평가기가 폴드마다 고르고, 너는 **구조**만 바꾼다.
사전등록·합격 바·잠금 규칙은 `PREREG.md` 다(정정 1 이 우선). 시작 전에 이 파일, `PREREG.md`, `prepare.py`(읽기만), `train.py` 를 읽는다.

## 파일
- `prepare.py` — 고정 평가기이자 진입점. 워크포워드(학습 2년 → 시험 1년, 시대 A 2011~2016-08 6폴드 · 시대 B 2020~2023 4폴드),
  학습 구간 라틴 하이퍼큐브 48점 → 고원 선택, 엔진 체결(다음 날 시가 진입), 비용 2배, 미래정보 검사, keep 판정. **고치지 않는다**
  (태그·sha 매니페스트로 검사 — 바뀌면 평가가 거부된다).
- `train.py` — 네가 고치는 유일한 파일. 계약·허용 목록은 파일 머리말에 있다(numpy·pandas·math·functools·itertools·collections 만).
- `results.tsv` — 네 메모(git 추적 안 함). 머리줄: `commit	score	beats_best	folds_pos	oos_trades	top5_removed	status	description`.
  시행 수의 정본은 평가기가 쓰는 `RUNS.jsonl` 이다.

## 실험 한 번
1. `train.py` 를 아이디어 하나로 고친다 → `git commit -m "exp: <한 줄>"` (`train.py` 만 add)
2. `uv run python research/autoresearch/prepare.py > run.log 2>&1` (작업 트리 루트; 15분 넘으면 죽이고 crash)
3. `grep -E "^(score|void|beats_best|beats_detail|lookahead_check|folds_pos|oos_trades|top5_removed_score|payoff_ratio):" run.log`
   비었으면 `tail -n 50 run.log` — 쉬운 오류면 고쳐 다시, 아니면 crash 로 적고 넘어간다. "허용 목록 위반" 이면 train.py 를 고친다.
4. `results.tsv` 에 한 줄 덧붙인다(모든 시도 — discard·crash 포함).
5. **keep 은 평가기가 판정한다:** `beats_best: True` 면 `uv run python research/autoresearch/prepare.py --accept` 로 기준을 올리고 커밋을 남긴다.
   아니면 `git reset --hard HEAD~1`. (`beats_best` = 직전 최고 대비 Δscore ≥ 0.005 ∧ 10폴드 중 ≥ 6 개선 ∧ top5 제거 점수 악화 ≤ 0.01.)
   같은 점수면 더 단순한 쪽이 낫다 — 줄을 지웠는데 score 차이가 0.002 안이면 단순한 쪽을 남겨도 된다(`--accept` 없이).
6. 다시 1.

## 점수가 뜻하는 것
score = 10개 시험 폴드의 **노출을 맞춘 선택 알파**(전략 로그 성장 − 전날 노출 × KOSPI 일수익 로그 성장, 연율)의 평균 − 0.5 × 표준편차. 비용 2배.
쉬는 해·시장 타이밍은 0 이다 — 점수를 올리는 길은 무엇을 사고 언제 파느냐(종목 선택·보유 관리)뿐이다. 폴드별 값은 일부러 보여주지 않는다.
지금까지 이 데이터의 모든 규칙은 시장 보유를 못 이겼다. 음수에서 시작하는 것이 정상이다.

## 하지 말 것(어기면 결과 무효)
- `prepare.py`·엔진·데이터·의존성·캐시·원장(`RUNS.jsonl`·`BEST.json`·`LOCKBOX.jsonl`·`MANIFEST.json`) 변경, `--final` 실행, 태그 조작.
- 허용 목록 우회(문자열로 모듈 이름 만들기, 내장 함수 꺼내기 등), DB·파일·네트워크 접근, 잠금 절반 추론.
- 사후 지식: 아는 승자 종목·업종·연도를 겨냥하는 규칙(날짜·코드를 받지 않는다 — 배열 위치로 같은 짓을 하지 않는다).
- 미래 정보: t 신호는 t 종가까지의 값만(평가기가 절단 재계산으로 검사 — 걸리면 void).
- 숫자 맞추기: 상수를 손으로 바꿔 점수를 올리지 않는다 — 숫자는 `params` 범위로 내고 평가기에 맡긴다. 범위를 한 점으로 좁히는 것도 손 맞추기다.
- **시간 손절**·**이익 잠금·본전 이동·촘촘한 추적**·**좁은 손절**. 청산은 메뉴(MA w 20~250, 어제까지 w일 최저 저가 w 20~120)와
  하드 손절(≥ 10%)뿐이다. 손실 한도는 비중(`risk`)으로 건다.
- 거래를 줄여 거래당 숫자를 올리는 방향(체결 < 5 인 폴드 2개 초과, 전체 < 150, 연 < 20 이면 void).

## 사용자 매매 원칙(설계 공간)
- EPS 상승은 필수다(평가기가 강제). 수익은 길게, 손실은 짧게. 미너비니를 웬만하면 따른다.
- 정석 진입: 급등 뒤 **하락** 쐐기(고점 H0 = A, 바닥 D 에서 매수). 상승 쐐기는 오답이다.
- 승률이 아니라 손익비·기대값을 본다. 승률이 높은데 손익비 < 1 이면 승자를 일찍 자르는 청산부터 의심한다.

## 이미 닫힌 것(같은 것을 파라미터만 바꿔 다시 내지 않는다)
- 추세 템플릿 + N일 돌파 + RS + EPS(= 기준선, 23세대 C): 2009~16 에서 KOSPI 보유보다 연 −1.4pp, 상위 5 종목 빼면 알파 없음.
- 거래대금 종목군·θ 70·슬롯 20(leader_M): 2009~16 연 −5.7%. 쫓지 않기 + 10일 저가 구조 손절(leader_P): 무작위 솎기 널 중앙보다 낮음.
- RL 순위(leader_P_RL): 무작위 점수와 같음. 200일선 국면 청산: 2009~16 에 보유 대비 −6.8pp/년.
- 11~18세대: 진입을 느슨하게 해 거래를 늘린 것 전부 거래당 ≤ 0. 좁은 % 손절: 대장주를 다 털림.
그러니 다른 **구조**를 시도하라: 진입 맥락(수축·거래량 마름·눌림의 모양), 순위(무엇을 먼저 사나), 청산선 메뉴 선택, 비중·불타기.

## 멈추지 않는다
루프가 시작되면 사람에게 묻지 않는다. 실험 사이에 말하지 않는다. 10 실험마다 한 번만:
```
=== 진행 (N 실험) ===
기준선 X → 최고 Y (top5 제거 Z) · keep K / discard D / crash C / void V
배운 것: <한 줄>
```
아이디어가 떨어지면 `research/logs/sepa_explore/MINERVINI_RULES.md`, `~/git/swing-it/data/minervini-graph/`(`graphify query`)에서 새 구조를 찾는다.
사람이 끊을 때까지 계속한다. 졸업 심사(`--final`)는 메인 세션이 한다 — 너는 하지 않는다.
