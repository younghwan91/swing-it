# 천장 아래 하락 쐐기 — 깔때기(건수만, 성과 미계산) 2026-10-10

`wedge_test_run.py --funnel-only` (research/logs/sepa_explore/WEDGE_TEST.md (7196cad + 정정 0 d1b7db8 + 정정 0b 0a911e2 + 정정 1 7e99bdb + 정정 2 bf62569)). 사건(주문일) 2017-09-15~2026-08-31. 시뮬은 진입 체결과 보유 중 거르기(그 거르기에 필요한 청산일)까지만 쓰고, 청산 사유·수익·손절 여부는 어디에도 적지 않는다. 실행 1.4분.

**사전 규칙:** W 체결 ≥ 80 ∧ 2017-09~2024-12 체결 ≥ 40 인 가장 높은 하한 → **넷 다 미달 → W VOID**(기술만).
<!-- adv_floor_rule: none · v0_void: true -->

| ADV20 하한(백만원) | W 체결 | 하위(≤2024-12) | 규칙 충족 | W1d | W-be1 | W-nc | W-tt | W-E | W-F | 쐐기 셋업 행(맥락 통과) | 셋업 후보 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5,000 | 6 | 6 | ✗ | 6 | 6 | 6 | 1 | 1 | 0 | 24 | 198,061 |
| 3,000 | 7 | 7 | ✗ | 7 | 7 | 7 | 1 | 1 | 0 | 28 | 246,687 |
| 2,000 | 11 | 11 | ✗ | 11 | 11 | 13 | 2 | 1 | 0 | 51 | 283,029 |
| 1,000 | 12 | 12 | ✗ | 12 | 12 | 16 | 2 | 1 | 0 | 55 | 334,913 |

## 하한별 깔때기

### ADV20 ≥ 5,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 198,203 − 분할 직후 60일 142 = 198,061
- 쐐기 첫 탈락: ok 31 · no_data 210 · base_len 92,003 · highs<3 6,236 · highs_not_falling 54,839 · lows_not_falling 30,737 · spacing 13,109 · c1_range 1 · not_decreasing 238 · last>15% 49 · slope 13 · A_touch 339 · depth 38 · E<2d 84 · already_broken 129 · integrity 1 · close_band 3 · volume_dryup 1
- 쐐기 통과 31 − H0 맥락(추세 템플릿 ∧ RS ≥ 70) 탈락 7 = **24** · d−1 추세 템플릿 3 · E 가속 6 · 수급 기간 0 (F 알려짐 0) · 널 풀(맥락 통과·쐐기 아님 · 깊이 띠) 53,809 · W-nc 셋업 행 28 · 진행 중 E 셋업 행 8 · H0 전 20일 급등 > 30% 셋업 행 15
- 팔 셋업 행: {'W': 24, 'W1d': 24, 'W-be1': 24, 'W-nc': 28, 'W-tt': 3, 'W-E': 6, 'W-F': 0}
- **W 체결 6** (하위 6) · 연도별 {2018: 1, 2021: 4, 2022: 1}
- W 체결 기반 길이(H0 → s, 거래일) 최소/25%/중앙/75%/최대: 30/44/59/75/120
- W 체결 쐐기 안 수축 수: {2: 6} · 체결가 → H0 거리 10/50/90%: +10.8%/+20.8%/+29.9%
- 돌파일 거래량/직전 50일 구간(0: < 1, 1: 1~1.5, 2: ≥ 1.5): {0: 3, 2: 3} · 갭 체결(시가 ≥ 주문가) 0 · 진행 중 E 체결 1
- H0 전 20일 급등률 10/50/90%: +18%/+47%/+72% · > 30% 4/6 · d−1 RS 10/50/90: 46/72/88
- 개인 체결 가능성: ADV20 > 1,000 백만원 6/6 · ADV20 중앙 9,791 백만원
- 주문·체결 결과(버림 사유): W: no_fill 16, skip_risk>9.5% 2 · W1d: no_fill 16, skip_risk>9.5% 2 · W-be1: no_fill 16, skip_risk>9.5% 2 · W-nc: no_fill 17, skip_risk>9.5% 5 · W-tt: no_fill 2 · W-E: no_fill 3, skip_risk>9.5% 2

### ADV20 ≥ 3,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 246,851 − 분할 직후 60일 164 = 246,687
- 쐐기 첫 탈락: ok 35 · no_data 266 · base_len 109,156 · highs<3 7,984 · highs_not_falling 71,475 · lows_not_falling 39,402 · spacing 17,101 · c1_range 1 · not_decreasing 340 · last>15% 75 · slope 23 · A_touch 424 · depth 76 · E<2d 132 · already_broken 189 · integrity 1 · close_band 4 · volume_dryup 3
- 쐐기 통과 35 − H0 맥락(추세 템플릿 ∧ RS ≥ 70) 탈락 7 = **28** · d−1 추세 템플릿 4 · E 가속 6 · 수급 기간 0 (F 알려짐 0) · 널 풀(맥락 통과·쐐기 아님 · 깊이 띠) 68,583 · W-nc 셋업 행 36 · 진행 중 E 셋업 행 9 · H0 전 20일 급등 > 30% 셋업 행 18
- 팔 셋업 행: {'W': 28, 'W1d': 28, 'W-be1': 28, 'W-nc': 36, 'W-tt': 4, 'W-E': 6, 'W-F': 0}
- **W 체결 7** (하위 7) · 연도별 {2018: 1, 2021: 5, 2022: 1}
- W 체결 기반 길이(H0 → s, 거래일) 최소/25%/중앙/75%/최대: 30/44/45/74/120
- W 체결 쐐기 안 수축 수: {2: 7} · 체결가 → H0 거리 10/50/90%: +11.5%/+21.1%/+32.2%
- 돌파일 거래량/직전 50일 구간(0: < 1, 1: 1~1.5, 2: ≥ 1.5): {0: 4, 2: 3} · 갭 체결(시가 ≥ 주문가) 1 · 진행 중 E 체결 1
- H0 전 20일 급등률 10/50/90%: +20%/+44%/+70% · > 30% 5/7 · d−1 RS 10/50/90: 47/65/88
- 개인 체결 가능성: ADV20 > 1,000 백만원 7/7 · ADV20 중앙 6,220 백만원
- 주문·체결 결과(버림 사유): W: no_fill 18, skip_overlap 1, skip_risk>9.5% 2 · W1d: no_fill 18, skip_overlap 1, skip_risk>9.5% 2 · W-be1: no_fill 18, skip_overlap 1, skip_risk>9.5% 2 · W-nc: no_fill 20, skip_overlap 1, skip_risk>9.5% 8 · W-tt: no_fill 3 · W-E: no_fill 3, skip_risk>9.5% 2

### ADV20 ≥ 2,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 283,203 − 분할 직후 60일 174 = 283,029
- 쐐기 첫 탈락: ok 65 · no_data 302 · base_len 119,603 · highs<3 9,543 · highs_not_falling 85,102 · lows_not_falling 46,467 · spacing 20,298 · c1_range 1 · not_decreasing 461 · last>15% 87 · slope 26 · A_touch 514 · depth 99 · E<2d 190 · already_broken 258 · integrity 1 · close_band 6 · volume_dryup 6
- 쐐기 통과 65 − H0 맥락(추세 템플릿 ∧ RS ≥ 70) 탈락 14 = **51** · d−1 추세 템플릿 7 · E 가속 6 · 수급 기간 0 (F 알려짐 0) · 널 풀(맥락 통과·쐐기 아님 · 깊이 띠) 80,219 · W-nc 셋업 행 69 · 진행 중 E 셋업 행 24 · H0 전 20일 급등 > 30% 셋업 행 25
- 팔 셋업 행: {'W': 51, 'W1d': 51, 'W-be1': 51, 'W-nc': 69, 'W-tt': 7, 'W-E': 6, 'W-F': 0}
- **W 체결 11** (하위 11) · 연도별 {2018: 1, 2020: 2, 2021: 7, 2022: 1}
- W 체결 기반 길이(H0 → s, 거래일) 최소/25%/중앙/75%/최대: 30/44/65/74/120
- W 체결 쐐기 안 수축 수: {2: 11} · 체결가 → H0 거리 10/50/90%: +14.1%/+23.9%/+37.9%
- 돌파일 거래량/직전 50일 구간(0: < 1, 1: 1~1.5, 2: ≥ 1.5): {0: 6, 1: 2, 2: 3} · 갭 체결(시가 ≥ 주문가) 1 · 진행 중 E 체결 4
- H0 전 20일 급등률 10/50/90%: +15%/+44%/+64% · > 30% 7/11 · d−1 RS 10/50/90: 51/65/90
- 개인 체결 가능성: ADV20 > 1,000 백만원 11/11 · ADV20 중앙 5,082 백만원
- 주문·체결 결과(버림 사유): W: no_fill 30, skip_overlap 5, skip_pivot_done 2, skip_risk>9.5% 3 · W1d: no_fill 30, skip_overlap 3, skip_pivot_done 4, skip_risk>9.5% 3 · W-be1: no_fill 30, skip_overlap 5, skip_pivot_done 2, skip_risk>9.5% 3 · W-nc: no_fill 37, skip_gap>3% 1, skip_overlap 5, skip_pivot_done 2, skip_risk>9.5% 11 · W-tt: no_fill 5 · W-E: no_fill 3, skip_risk>9.5% 2

### ADV20 ≥ 1,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 335,099 − 분할 직후 60일 186 = 334,913
- 쐐기 첫 탈락: ok 80 · no_data 338 · base_len 133,943 · highs<3 11,748 · highs_not_falling 105,186 · lows_not_falling 56,826 · spacing 24,435 · c1_range 1 · not_decreasing 624 · last>15% 128 · slope 37 · A_touch 749 · depth 147 · E<2d 280 · already_broken 368 · integrity 2 · close_band 6 · volume_dryup 15
- 쐐기 통과 80 − H0 맥락(추세 템플릿 ∧ RS ≥ 70) 탈락 25 = **55** · d−1 추세 템플릿 7 · E 가속 6 · 수급 기간 0 (F 알려짐 0) · 널 풀(맥락 통과·쐐기 아님 · 깊이 띠) 95,994 · W-nc 셋업 행 77 · 진행 중 E 셋업 행 25 · H0 전 20일 급등 > 30% 셋업 행 25
- 팔 셋업 행: {'W': 55, 'W1d': 55, 'W-be1': 55, 'W-nc': 77, 'W-tt': 7, 'W-E': 6, 'W-F': 0}
- **W 체결 12** (하위 12) · 연도별 {2018: 2, 2020: 2, 2021: 7, 2022: 1}
- W 체결 기반 길이(H0 → s, 거래일) 최소/25%/중앙/75%/최대: 30/45/64/74/120
- W 체결 쐐기 안 수축 수: {2: 12} · 체결가 → H0 거리 10/50/90%: +14.4%/+22.5%/+37.7%
- 돌파일 거래량/직전 50일 구간(0: < 1, 1: 1~1.5, 2: ≥ 1.5): {0: 6, 1: 2, 2: 4} · 갭 체결(시가 ≥ 주문가) 1 · 진행 중 E 체결 4
- H0 전 20일 급등률 10/50/90%: +16%/+43%/+64% · > 30% 7/12 · d−1 RS 10/50/90: 51/68/90
- 개인 체결 가능성: ADV20 > 1,000 백만원 12/12 · ADV20 중앙 4,068 백만원
- 주문·체결 결과(버림 사유): W: no_fill 33, skip_overlap 5, skip_pivot_done 2, skip_risk>9.5% 3 · W1d: no_fill 33, skip_overlap 3, skip_pivot_done 4, skip_risk>9.5% 3 · W-be1: no_fill 33, skip_overlap 5, skip_pivot_done 2, skip_risk>9.5% 3 · W-nc: no_fill 40, skip_gap>3% 1, skip_overlap 5, skip_pivot_done 2, skip_risk>9.5% 13 · W-tt: no_fill 5 · W-E: no_fill 3, skip_risk>9.5% 2

## 공통

- NXT 미보정 행이 있는 종목 1,302개 — 보유 경로는 2026-09-11 에서 `data_end` 로 끊는다.
- 차트용 반전점: `wedge/chart_points.json` (하한별 W 체결 전부, 기하만).
- 판정은 사람이 한다. 이 문서는 사전 규칙의 산술만 적는다.

