# 단계 B 패리티 검사 (PO-1.4) — 코드 검사, 성과 숫자 없음

core `13c56a7`, END 2023-12-28, 표 daily_bars_adjusted(지금 표 — PR #12 전). 격자 2019-08-01~2023-12-15 (55일).
하네스 패리티 모드(지연 0·무비용·강제 매도 끔·현금 없음, `start_index` = 첫 격자일, `min_names=20`)와 core `staggered_tranche_backtest(top_n=10, cap_rank=(20,100), horizon=60, step=20, adv_floor=20000, delisting_exit=True)` 의 스텝별 비교.
**book·bench 값 자체는 출력·저장하지 않았다**(B 는 수리된 표 뒤에 연다, PO-5.2).

| 항목 | 값 |
|---|---|
| 스텝 수(하네스/core) | 54/54 (같은 날짜 True) |
| 기간 | 2019-08-01 ~ 2023-11-17 |
| book 최대 절대차 | 0.000e+00 |
| bench 최대 절대차 | 2.776e-17 |
| NaN 불일치 | 0 |
| n_universe 같음 | True |
| 적격 집합 같음(격자 전부) | True |
| ADV 바이트 같음(core `_trailing_adv`) | True |
| 바 | 1e-10 |

**패리티: 통과.**

- 표 지문 시작=끝: True [2877771, '2023-12-28', 6675565091165, 2316557639282].
- 격자 모델이 PO-3 캐시(v4)와 공통 53개 비트 동일: True.
- 이음매 가드: {'seam_cells': 185, 'seam_cells_by_source': [183, 2]}. 시장 구분: {'market_unknown': 0, 'delisted_with_last_date': 489}.
- 수리된 표(PR #12)가 배포되면 같은 검사를 다시 통과해야 B 를 돌린다(PO-5.2-4).

_40.8초_
