# c0-repro-check — 수리 **전** 표에서 c0 경로가 PO2.json 을 다시 내는지 (PO-7.1 권고, 2026-11-02)

- `pead_optimizer_repaired.py --mode c0-repro-check`, swing-it `24bde8b`, core `13c56a7`, 새 캐시. 반대 관문(지문 = A1.json 수리 전) 통과. 구창(≤ 2023-12-28)만 읽었다.
- 결과: PO2.json 대비 비교 66개 항목 전부 소수 5자리 일치(최대 차 0), `repro_exact_PO2` = True, 124초. 수리 뒤 `--mode c0` 는 같은 코드 경로다.
- 아래 본문 제목의 "수리된" 은 러너 렌더러 문구다 — 이 실행은 수리 전 표다. 출력 이름은 C0_repaired 와 겹치지 않게 C0_repro_check 로 옮겼다.

---

# 강건성 재실행 C0_repaired — PO-2 러너 C0 부분, 수리된 `daily_bars_adjusted`(PR #12), 보고만 (PO-2.1-6·PO-5.2)

배포일 None, swing-it `24bde8b`, core `13c56a7`. END 2023-12-28, v3, PO-2 이음매 가드(같은 코드).
지문(2019-01-01~2026-09-11) [4591382, '2026-09-11', 10568817328205, 3149933269582], (2019-01-01~2023-12-28) [2877771, '2023-12-28', 6675565091165, 2316557639282].

| 묶음 | 항목 | PO2.json | 수리 뒤 | 차 |
|---|---|---|---|---|
| R0_noguard | random | 0.97428 | 0.97428 | 0.00000 |
| R0_noguard | capw | 1.00232 | 1.00232 | 0.00000 |
| R0_noguard | style:size | 1.01243 | 1.01243 | 0.00000 |
| R0_noguard | style:beta | 0.98684 | 0.98684 | 0.00000 |
| R0_noguard | style:momentum | 0.90762 | 0.90762 | 0.00000 |
| R0_noguard | style:resid_vol | 0.87297 | 0.87297 | 0.00000 |
| R0_noguard | style:liquidity | 0.83363 | 0.83363 | 0.00000 |
| R0_noguard | style:value | 0.90532 | 0.90532 | 0.00000 |
| R0_guard | random | 0.97010 | 0.97010 | 0.00000 |
| R0_guard | capw | 1.00286 | 1.00286 | 0.00000 |
| R0_guard | style:size | 1.01323 | 1.01323 | 0.00000 |
| R0_guard | style:beta | 0.99288 | 0.99288 | 0.00000 |
| R0_guard | style:momentum | 0.91213 | 0.91213 | 0.00000 |
| R0_guard | style:resid_vol | 0.87405 | 0.87405 | 0.00000 |
| R0_guard | style:liquidity | 0.82778 | 0.82778 | 0.00000 |
| R0_guard | style:value | 0.91076 | 0.91076 | 0.00000 |
| tails | CV_med | 0.32099 | 0.32099 | 0.00000 |
| tails | CV_split | 0.23259 | 0.23259 | 0.00000 |
| tails | nu_u | 4.50000 | 4.50000 | 0.00000 |
| tails | kurt_u | 13.83146 | 13.83146 | 0.00000 |
| tails | nu_f | 9.33000 | 9.33000 | 0.00000 |
| tails | kurt_f | 1.12570 | 1.12570 | 0.00000 |
| sigma_target | value | 0.03802 | 0.03802 | 0.00000 |
| sigma_target | n | 51 | 51 | 0 |
| sigma_target | q10 | 0.03181 | 0.03181 | 0.00000 |
| sigma_target | q90 | 0.04602 | 0.04602 | 0.00000 |
| OB_band | ok | 10363 | 10363 | 0 |
| OB_band | lam_fail | 37 | 37 | 0 |
| OB_band | solver_fail | 0 | 0 | 0 |
| realized | S:size | 1.01323 | 1.01323 | 0.00000 |
| realized | S:beta | 0.99288 | 0.99288 | 0.00000 |
| realized | S:momentum | 0.91213 | 0.91213 | 0.00000 |
| realized | S:resid_vol | 0.87405 | 0.87405 | 0.00000 |
| realized | S:liquidity | 0.82778 | 0.82778 | 0.00000 |
| realized | S:value | 0.91076 | 0.91076 | 0.00000 |
| realized | P5 | 0.95697 | 0.95697 | 0.00000 |
| realized | P50 | 0.95725 | 0.95725 | 0.00000 |
| realized | P10B | 1.04808 | 1.04808 | 0.00000 |
| realized | HV | 0.86060 | 0.86060 | 0.00000 |
| realized | SM | 0.99573 | 0.99573 | 0.00000 |
| realized | IP:G10 | 0.83500 | 0.83500 | 0.00000 |
| realized | IP:G15 | 1.11338 | 1.11338 | 0.00000 |
| realized | IP:G20 | 0.86321 | 0.86321 | 0.00000 |
| realized | IP:G25 | 0.81824 | 0.81824 | 0.00000 |
| realized | IP:G30 | 0.87253 | 0.87253 | 0.00000 |
| realized | IP:G35 | 1.09258 | 1.09258 | 0.00000 |
| realized | IP:G40 | 0.89360 | 0.89360 | 0.00000 |
| realized | IP:G45 | 1.13835 | 1.13835 | 0.00000 |
| realized | IP:G50 | 1.05669 | 1.05669 | 0.00000 |
| realized | IP:G55 | 1.29401 | 1.29401 | 0.00000 |
| realized | IE:G10 | 0.74737 | 0.74737 | 0.00000 |
| realized | IE:G15 | 0.90594 | 0.90594 | 0.00000 |
| realized | IE:G20 | 0.95574 | 0.95574 | 0.00000 |
| realized | IE:G25 | 0.77279 | 0.77279 | 0.00000 |
| realized | IE:G30 | 1.07925 | 1.07925 | 0.00000 |
| realized | IE:G35 | 1.07759 | 1.07759 | 0.00000 |
| realized | IE:G40 | 0.90345 | 0.90345 | 0.00000 |
| realized | IE:G45 | 1.00234 | 1.00234 | 0.00000 |
| realized | IE:G50 | 1.18905 | 1.18905 | 0.00000 |
| realized | IE:G55 | 1.19695 | 1.19695 | 0.00000 |
| realized | OB21:band | 1.05905 | 1.05905 | 0.00000 |
| realized | OB5:band | 1.09973 | 1.09973 | 0.00000 |
| seam_cells | seam_cells | 362 | 362 | 0 |
| seam_cells | seam_codes | 262 | 262 | 0 |
| nan_to_zero | seam | 112 | 112 | 0 |
| nan_to_zero | other | 431 | 431 | 0 |

실현 바이어스 최대 절대 차 0.00000. 판정이 아니다(C0 는 보정 단계).

