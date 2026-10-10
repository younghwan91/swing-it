# 쐐기 D(바닥 진입) — 깔때기(건수만, 성과 미계산) 2026-10-10

`wedge_d_run.py --funnel-only` (research/logs/sepa_explore/WEDGE_D_TEST.md (2154f77 + 정정 1 b008fce, 탐색 표시)). 사건(주문일) 2017-09-15~2026-08-31. 시뮬은 진입 체결과 보유 중 거르기(그 거르기에 필요한 청산일)까지만 쓰고, 청산 사유·수익·손절 여부는 어디에도 적지 않는다. 실행 1.5분. 사슬: VCP(결과 차트) → 쐐기(결과 차트 12건) → 쐐기 D(진입·청산 동시 변경, 사용자 지시 예외 · 탐색).

**사전 규칙:** WD 체결 ≥ 80 ∧ 2017-09~2024-12 체결 ≥ 40 인 가장 높은 하한 → **넷 다 미달 → WD VOID**(기술만).
<!-- adv_floor_rule: none · v0_void: true -->

| ADV20 하한(백만원) | WD 체결 | 하위(≤2024-12) | 규칙 충족 | WD 열람 겹침(진입일) | WD-be1 | WD-up | WD-nc | WD-nocap | WD 셋업 행(맥락 통과) | 셋업 후보 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5,000 | 10 | 10 | ✗ | 0 | 10 | 5 | 15 | 10 | 57 | 198,061 |
| 3,000 | 19 | 16 | ✗ | 0 | 19 | 9 | 25 | 19 | 75 | 246,687 |
| 2,000 | 24 | 21 | ✗ | 0 | 24 | 9 | 31 | 24 | 85 | 283,029 |
| 1,000 | 25 | 22 | ✗ | 0 | 25 | 10 | 35 | 25 | 91 | 334,913 |

## 하한별 깔때기

### ADV20 ≥ 5,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 198,203 − 분할 직후 60일 142 = 198,061
- 첫 탈락: ok 67 · no_data 210 · base_len 92,003 · not_four_points 101,169 · C>A*0.99 431 · D>B*0.99 1,761 · spacing 1,425 · c1_range 29 · not_decreasing 536 · c2>15% 86 · slope 17 · integrity 194 · volume_dryup 75 · d-iD>5 16 · consumed_D 42
- 모양·진입 통과 67 − H0 맥락 탈락 10 = **57** · WD-up 셋업 행 51 · WD-nc 81 · 널 풀(맥락 통과·WD 아님) 74,314 · 팔 셋업 행 {'WD': 57, 'WD-be1': 57, 'WD-up': 51, 'WD-nc': 81, 'WD-nocap': 57}
- **WD 체결 10** (하위 10) · 연도별 {2019: 1, 2020: 1, 2021: 5, 2022: 1, 2023: 1, 2024: 1}
- 열람 창(`research/logs/sepa_explore/wedge_d/viewed_windows.csv` ± 60 거래일)과 진입일이 겹치는 체결(깔때기 모드 — 보유 끝은 안 봄): WD 0 · WD-be1 0 · WD-up 0 · WD-nc 0 · WD-nocap 0
- 기반 길이(H0 → s) 10/50/90: 18/22/31 · 체결가의 D 대비 10/50/90: +2.1%/+3.7%/+5.7% · D → 체결 일수 10/50/90: 1/1/2 · 체결가 → H0 거리 10/50/90: +11.8%/+13.8%/+16.0%
- H0 전 상승률(저가 기준) 10/50/90 — 20일 +16%/+30%/+35% · 40일 +28%/+36%/+56% · 60일 +30%/+39%/+81%
- D 확정(5% 반등) 체결 1 · 갭 체결 1 · 체결일 거래량/50일 구간 {0: 9, 1: 1} · ADV20 중앙 10,750 백만원
- 주문·체결 결과(버림 사유): WD: no_fill 35, skip_overlap 1, skip_pivot_done 8, skip_risk>9.5% 3 · WD-be1: no_fill 35, skip_overlap 1, skip_pivot_done 8, skip_risk>9.5% 3 · WD-up: no_fill 41, skip_overlap 3, skip_risk>9.5% 2 · WD-nc: no_fill 49, skip_overlap 1, skip_pivot_done 13, skip_risk>9.5% 3 · WD-nocap: no_fill 35, skip_overlap 1, skip_pivot_done 8, skip_risk>9.5% 3

### ADV20 ≥ 3,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 246,851 − 분할 직후 60일 164 = 246,687
- 첫 탈락: ok 91 · no_data 266 · base_len 109,156 · not_four_points 131,330 · C>A*0.99 510 · D>B*0.99 2,314 · spacing 1,792 · c1_range 39 · not_decreasing 619 · c2>15% 96 · slope 17 · integrity 283 · volume_dryup 77 · d-iD>5 28 · consumed_D 69
- 모양·진입 통과 91 − H0 맥락 탈락 16 = **75** · WD-up 셋업 행 69 · WD-nc 107 · 널 풀(맥락 통과·WD 아님) 93,767 · 팔 셋업 행 {'WD': 75, 'WD-be1': 75, 'WD-up': 69, 'WD-nc': 107, 'WD-nocap': 75}
- **WD 체결 19** (하위 16) · 연도별 {2017: 1, 2019: 2, 2020: 1, 2021: 9, 2022: 1, 2023: 1, 2024: 1, 2025: 2, 2026: 1}
- 열람 창(`research/logs/sepa_explore/wedge_d/viewed_windows.csv` ± 60 거래일)과 진입일이 겹치는 체결(깔때기 모드 — 보유 끝은 안 봄): WD 0 · WD-be1 0 · WD-up 0 · WD-nc 0 · WD-nocap 0
- 기반 길이(H0 → s) 10/50/90: 16/20/27 · 체결가의 D 대비 10/50/90: +2.0%/+3.4%/+4.7% · D → 체결 일수 10/50/90: 1/1/2 · 체결가 → H0 거리 10/50/90: +11.5%/+14.8%/+22.6%
- H0 전 상승률(저가 기준) 10/50/90 — 20일 +14%/+30%/+49% · 40일 +27%/+33%/+61% · 60일 +28%/+46%/+81%
- D 확정(5% 반등) 체결 1 · 갭 체결 3 · 체결일 거래량/50일 구간 {0: 16, 1: 2, 2: 1} · ADV20 중앙 5,174 백만원
- 주문·체결 결과(버림 사유): WD: no_fill 39, skip_overlap 5, skip_pivot_done 9, skip_risk>9.5% 3 · WD-be1: no_fill 39, skip_overlap 5, skip_pivot_done 9, skip_risk>9.5% 3 · WD-up: no_fill 54, skip_overlap 3, skip_pivot_done 1, skip_risk>9.5% 2 · WD-nc: no_fill 57, skip_overlap 5, skip_pivot_done 17, skip_risk>9.5% 3 · WD-nocap: no_fill 39, skip_overlap 5, skip_pivot_done 9, skip_risk>9.5% 3

### ADV20 ≥ 2,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 283,203 − 분할 직후 60일 174 = 283,029
- 첫 탈락: ok 105 · no_data 302 · base_len 119,603 · not_four_points 156,069 · C>A*0.99 574 · D>B*0.99 2,774 · spacing 2,131 · c1_range 59 · not_decreasing 720 · c2>15% 122 · slope 23 · integrity 355 · volume_dryup 81 · d-iD>5 28 · consumed_D 83
- 모양·진입 통과 105 − H0 맥락 탈락 20 = **85** · WD-up 셋업 행 75 · WD-nc 120 · 널 풀(맥락 통과·WD 아님) 108,745 · 팔 셋업 행 {'WD': 85, 'WD-be1': 85, 'WD-up': 75, 'WD-nc': 120, 'WD-nocap': 85}
- **WD 체결 24** (하위 21) · 연도별 {2017: 1, 2018: 1, 2019: 2, 2020: 1, 2021: 13, 2022: 1, 2023: 1, 2024: 1, 2025: 2, 2026: 1}
- 열람 창(`research/logs/sepa_explore/wedge_d/viewed_windows.csv` ± 60 거래일)과 진입일이 겹치는 체결(깔때기 모드 — 보유 끝은 안 봄): WD 0 · WD-be1 0 · WD-up 0 · WD-nc 0 · WD-nocap 0
- 기반 길이(H0 → s) 10/50/90: 15/20/28 · 체결가의 D 대비 10/50/90: +2.1%/+3.5%/+5.6% · D → 체결 일수 10/50/90: 1/1/2 · 체결가 → H0 거리 10/50/90: +11.4%/+15.2%/+25.5%
- H0 전 상승률(저가 기준) 10/50/90 — 20일 +15%/+31%/+48% · 40일 +28%/+36%/+59% · 60일 +30%/+50%/+79%
- D 확정(5% 반등) 체결 1 · 갭 체결 4 · 체결일 거래량/50일 구간 {0: 20, 1: 3, 2: 1} · ADV20 중앙 4,565 백만원
- 주문·체결 결과(버림 사유): WD: no_fill 43, skip_overlap 6, skip_pivot_done 9, skip_risk>9.5% 3 · WD-be1: no_fill 43, skip_overlap 6, skip_pivot_done 9, skip_risk>9.5% 3 · WD-up: no_fill 60, skip_overlap 3, skip_pivot_done 1, skip_risk>9.5% 2 · WD-nc: no_fill 61, skip_overlap 6, skip_pivot_done 19, skip_risk>9.5% 3 · WD-nocap: no_fill 43, skip_overlap 6, skip_pivot_done 9, skip_risk>9.5% 3

### ADV20 ≥ 1,000 백만원

- 셋업 후보(신호 행 s: H5 ∧ MA50>MA150>MA200 ∧ MA200 상승 ∧ breadth > 0.5) 335,099 − 분할 직후 60일 186 = 334,913
- 첫 탈락: ok 119 · no_data 338 · base_len 133,943 · not_four_points 191,986 · C>A*0.99 677 · D>B*0.99 3,455 · spacing 2,611 · c1_range 68 · not_decreasing 872 · c2>15% 137 · slope 38 · integrity 444 · volume_dryup 83 · d-iD>5 44 · consumed_D 98
- 모양·진입 통과 119 − H0 맥락 탈락 28 = **91** · WD-up 셋업 행 87 · WD-nc 144 · 널 풀(맥락 통과·WD 아님) 128,957 · 팔 셋업 행 {'WD': 91, 'WD-be1': 91, 'WD-up': 87, 'WD-nc': 144, 'WD-nocap': 91}
- **WD 체결 25** (하위 22) · 연도별 {2017: 1, 2018: 1, 2019: 2, 2020: 1, 2021: 14, 2022: 1, 2023: 1, 2024: 1, 2025: 2, 2026: 1}
- 열람 창(`research/logs/sepa_explore/wedge_d/viewed_windows.csv` ± 60 거래일)과 진입일이 겹치는 체결(깔때기 모드 — 보유 끝은 안 봄): WD 0 · WD-be1 0 · WD-up 0 · WD-nc 0 · WD-nocap 0
- 기반 길이(H0 → s) 10/50/90: 15/20/28 · 체결가의 D 대비 10/50/90: +2.1%/+3.5%/+5.6% · D → 체결 일수 10/50/90: 1/1/2 · 체결가 → H0 거리 10/50/90: +11.4%/+15.6%/+24.9%
- H0 전 상승률(저가 기준) 10/50/90 — 20일 +15%/+30%/+48% · 40일 +28%/+39%/+58% · 60일 +31%/+48%/+78%
- D 확정(5% 반등) 체결 1 · 갭 체결 5 · 체결일 거래량/50일 구간 {0: 21, 1: 3, 2: 1} · ADV20 중앙 4,436 백만원
- 주문·체결 결과(버림 사유): WD: no_fill 48, skip_overlap 6, skip_pivot_done 9, skip_risk>9.5% 3 · WD-be1: no_fill 48, skip_overlap 6, skip_pivot_done 9, skip_risk>9.5% 3 · WD-up: no_fill 70, skip_overlap 3, skip_pivot_done 1, skip_risk>9.5% 3 · WD-nc: no_fill 79, skip_overlap 7, skip_pivot_done 19, skip_risk>9.5% 4 · WD-nocap: no_fill 48, skip_overlap 6, skip_pivot_done 9, skip_risk>9.5% 3

## 공통

- NXT 미보정 행이 있는 종목 1,302개 — 보유 경로는 2026-09-11 에서 `data_end` 로 끊는다.
- 차트 JSON: `wedge_d/chart_points.json` (하한별 WD 체결, 봉은 H0 − 60 ~ 주문일 — 깔때기 모드는 청산·이후 봉 없음).
- 정정 1(b008fce) 반영: D ≤ 0.99·B, 같은 (A, C) 두 번째 관통 차단, 손절 체결 = 손절가 − 1호가, WD-up 같은 청산, WD-nocap 팔, 열람 제외 표시.
- 탐색 표시: 판정은 사람이, 이 문서는 사전 규칙의 산술만.

