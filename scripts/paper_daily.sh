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
git add research/paper/*_ledger.jsonl
git diff --cached --quiet || git commit -q -m "research(paper): $(date +%F) 장부(cron)" && echo "커밋 $(git rev-parse --short HEAD)"
