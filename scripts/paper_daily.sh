#!/bin/bash
# 전진 모의매매 하루 한 번(평일 07:43, DB 확정 06:30 뒤 장 전) — 팔마다 장부에 덧붙이고 장부만 이 브랜치에 커밋한다.
# 커밋까지 하는 이유: 10-13~10-23 장부가 커밋 전에 워크트리째 지워져 사라졌다(2026-10-24 복구, 행에 backfilled=true).
# leader_P_RL 은 빠진다 — RL-1 기각(30 독립 표본) 뒤 모델 디렉터리(~/.cache/cc-swing/models/rl32)도 없다.
set -u
cd "$(dirname "$0")/.."
set -a; . ~/git/quant-airflow/.env; set +a
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
UV=/home/young/.local/bin/uv
for arm in leader_A leader_C leader_C20 leader_A_rs leader_C_rs leader_C20_rs leader_M leader_P leader_M_ana; do
    echo "== $(date '+%F %T') $arm"
    $UV run --with numba --extra pg python research/paper/leader_paper.py --arm "$arm" | grep -E "^(기록|이미 기록됨|시작 전)" || echo "실패: $arm"
done
# PEAD 순위 북 전진 팔(research/paper/PEAD_FORWARD_PREREG.md) — 위 leader 팔 뒤에 따로 돈다. 실패·시간 초과(15분)여도 leader 장부와
# 아래 커밋은 그대로 간다(set -e 없음, 오류는 이 로그로). 행 쓰기는 덧붙이기 전용이라 실패한 날은 --asof 로 채운다.
echo "== $(date '+%F %T') pead_rank_F1"
# 정정 F1-3: pipefail 이라 파이썬이 죽거나 부실한 결정일 행을 거부하면(종료 1) grep 이 다른 줄을 건져도 실패로 남는다. grep 이 감시·실패 줄도 남긴다.
# "이미 기록됨(재실행)" 은 오늘 쓴 행, "이미 기록됨(새 거래일 없음)" 은 휴장 아니면 DB 지연이다. 가격이 지수보다 뒤처지면 pead_paper 가 실패: 로 찍고,
# 지수까지 멈춘 경우는 quant-airflow 헬스체크(report_paper_ledger, 11:35·18:10)가 장부 날짜·그 뒤 틱으로 잡는다 — 이 스크립트가 안 돌아도 잡힌다.
( set -o pipefail; timeout 900 $UV run --extra pg python -W ignore research/paper/pead_paper.py | grep -E "^(기록|이미 기록됨|시작 전|감시|실패)" ) || echo "실패: pead_rank_F1"
git add research/paper/*_ledger.jsonl
git diff --cached --quiet || git commit -q -m "research(paper): $(date +%F) 장부(cron)" && echo "커밋 $(git rev-parse --short HEAD)"
