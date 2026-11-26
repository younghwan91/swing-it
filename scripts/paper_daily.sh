#!/bin/bash
# 전진 모의매매 하루 한 번(평일 07:43, DB 확정 06:30 뒤 장 전) — 팔마다 장부에 덧붙이고 장부만 이 브랜치에 커밋한다.
# 커밋까지 하는 이유: 10-13~10-23 장부가 커밋 전에 워크트리째 지워져 사라졌다(2026-10-24 복구, 행에 backfilled=true).
# 커밋 뒤 origin 에도 올린다: 11-21 워크트리가 다시 지워졌다 — 이번엔 11-20 커밋이 push 돼 있어 살았지만, 로컬 커밋만으로는
# 워크트리 삭제를 못 견딘다(2026-11-27 복구, 행에 backfilled=true).
# leader_P_RL 은 빠진다 — RL-1 기각(30 독립 표본) 뒤 모델 디렉터리(~/.cache/cc-swing/models/rl32)도 없다.
set -u
cd "$(dirname "$0")/.."
set -a; . ~/git/quant-airflow/.env; set +a
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
UV=/home/young/.local/bin/uv
# 겹친 실행 방지(수동 실행과 cron 이 같은 장부에 덧붙이거나 git 색인을 같이 잡지 않게). 잠금은 프로세스가 끝나면 풀린다.
mkdir -p "$HOME/.cache/cc-swing"
exec 9>"$HOME/.cache/cc-swing/paper_daily.lock"
flock -n 9 || { echo "실패: 이미 다른 paper_daily 가 돌고 있다 — 이번 실행은 건너뜀"; exit 1; }
FAILS=()
# 팔 하나 실행: 이름, 시간 제한(초), 명령. pipefail 이라 파이썬이 죽으면 grep 이 다른 줄을 건져도 실패로 남는다(정정 F1-3 과 같은 규칙).
# 시간 제한이 없으면 한 팔이 멈췄을 때 뒤 팔과 장부 커밋이 모두 막힌다.
run_arm() {
    local name=$1 lim=$2
    shift 2
    echo "== $(date '+%F %T') $name"
    ( set -o pipefail; timeout "$lim" "$@" | grep -E "^(기록|이미 기록됨|시작 전|감시|실패)" ) || { echo "실패: $name"; FAILS+=("$name"); }
}
# leader 9팔을 한 프로세스로 — 가격 설정(원값·정문 보정) 두 묶음마다 자료를 한 번만 읽는다(P3: 9번 따로 1228초 → 269초,
# 2026-11-19 임시 장부로 팔별 행이 run_at 빼고 바이트 동일 확인). 한 팔이 죽으면 그 팔만 "실패: <팔>" 을 찍고 나머지는 계속 간다.
# -W ignore 는 pandas 경고만 지운다 — 예외 추적은 그대로 로그에 남는다.
LEADER_ARMS=leader_A,leader_C,leader_C20,leader_A_rs,leader_C_rs,leader_C20_rs,leader_M,leader_P,leader_M_ana
run_arm leader 1800 $UV run --with numba --extra pg python -W ignore research/paper/leader_paper.py --arms "$LEADER_ARMS"
# PEAD 순위 북 전진 팔(research/paper/PEAD_FORWARD_PREREG.md) — 위 leader 팔 뒤에 따로 돈다. 실패·시간 초과(15분)여도 leader 장부와
# 아래 커밋은 그대로 간다(set -e 없음, 오류는 이 로그로). 행 쓰기는 덧붙이기 전용이라 실패한 날은 --asof 로 채운다.
# 정정 F1-3: pipefail 이라 파이썬이 죽거나 부실한 결정일 행을 거부하면(종료 1) grep 이 다른 줄을 건져도 실패로 남는다. grep 이 감시·실패 줄도 남긴다.
# "이미 기록됨(재실행)" 은 오늘 쓴 행, "이미 기록됨(새 거래일 없음)" 은 휴장 아니면 DB 지연이다. 가격이 지수보다 뒤처지면 pead_paper 가 실패: 로 찍고,
# 지수까지 멈춘 경우는 quant-airflow 헬스체크(report_paper_ledger, 11:35·18:10)가 장부 날짜·그 뒤 틱으로 잡는다 — 이 스크립트가 안 돌아도 잡힌다.
run_arm pead_rank_F1 900 $UV run --extra pg python -W ignore research/paper/pead_paper.py
# 일봉 상승 중 30분봉 전환 전진 팔(research/paper/MTF30_FORWARD_PREREG.md) — 전날 분봉으로 계좌를 처음부터 다시 돌려 한 줄 덧붙인다(약 30초).
run_arm mtf30_fwd 1200 $UV run --with numba --extra pg python -W ignore research/paper/mtf_paper.py
# 장부 경로만 커밋한다(git commit -- 경로). 예전처럼 색인 전체를 커밋하면 그 시각 이 워크트리에 누가 스테이징해 둔 연구 파일까지
# "장부(cron)" 커밋에 섞여 들어간다. 다른 스테이징은 그대로 남는다.
LEDGERS=(research/paper/*_ledger.jsonl)
git add -- "${LEDGERS[@]}"
if git diff --cached --quiet -- "${LEDGERS[@]}"; then
    echo "커밋 없음: 장부 변화 없음"
# 메시지는 영어 — 전역 commit-msg 훅(2026-11-24~)이 한글 메시지를 거부한다.
elif git commit -q -m "research(paper): $(date +%F) ledger (cron)" -- "${LEDGERS[@]}"; then
    echo "커밋 $(git rev-parse --short HEAD)"
else
    echo "실패: 장부 커밋"
    FAILS+=("commit")
fi
# origin/feature-combined-strategy 에 올린다(이 브랜치는 장부 브랜치 — PR 없이 바로). 커밋이 없어도 앞선 날 push 가 실패해
# 남은 커밋이 있으면 올린다. 장부 밖 파일을 건드린 커밋이 섞여 있으면 사람 몫이라 올리지 않고 실패로 남긴다(공개 origin).
# SKIP_CI_LOCAL: 전역 pre-push 의 로컬 CI(장부만 바뀐 커밋엔 무의미, 아침 실행을 늘린다). 신원 검사는 그대로 돈다.
BR=feature-combined-strategy
if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BR" ]; then
    echo "실패: 장부 push — 이 워크트리가 $BR 이 아니다($(git rev-parse --abbrev-ref HEAD))"
    FAILS+=("push")
elif ! timeout 60 git fetch -q origin "$BR"; then
    echo "실패: 장부 push — origin fetch 실패"
    FAILS+=("push")
elif [ -n "$(git rev-list "origin/$BR..HEAD")" ]; then
    other=$(git diff --name-only "origin/$BR...HEAD" | grep -vE '^research/paper/[^/]+_ledger\.jsonl$')
    if [ -n "$other" ]; then
        echo "실패: 장부 push 보류 — origin 에 없는 커밋이 장부 밖 파일을 바꾼다: $(echo $other | head -c 200)"
        FAILS+=("push")
    elif SKIP_CI_LOCAL=1 PUSH_SYNC_SKIP=1 timeout 180 git push -q origin "HEAD:refs/heads/$BR"; then
        echo "push origin/$BR $(git rev-parse --short HEAD)"
    else
        echo "실패: 장부 push (origin 이 앞서 있으면 사람이 합친다)"
        FAILS+=("push")
    fi
fi
echo "요약: $(date '+%F %T') 실패 ${#FAILS[@]}건${FAILS[*]:+ — ${FAILS[*]}}"
