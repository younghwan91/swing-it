#!/usr/bin/env bash
# sw-live 크론 래퍼 — 종료 코드를 경보 수준으로 바꾼다(코드는 docs/live-swing-runbook.md §2 "종료 코드").
#
#   deploy/sw-live-cron.sh <job> [sw-live 인자...]      예) deploy/sw-live-cron.sh signal --paper
#
# 0 = 정상(조용히 끝)            2 = warn  (건너뜀: 시간창 밖·잠금 경합·인자 오류)
# 4 = error (신규 진입 중단 — 보유 결정은 수행)   1 = error (작업 오류)   3 = error (상태 손상)
#
# 런타임도 같은 사건에 자기 경보를 보낸다. 이 래퍼는 **종료 코드 자체**를 감시 채널에 남기는 마지막 그물이다(런타임이 경보 전에
# 죽었거나, 경보 채널 설정이 빠졌어도 크론 로그·quant-airflow 경보 로그엔 남는다). 경보 실패는 무시하고 원래 종료 코드를 돌려준다.
#
# 환경: SW_LIVE_BIN(기본: 이 레포 .venv/bin/sw-live), ALERT_CMD(기본: quant-airflow collectors/alert.py — 웹훅 없으면 로그만).
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
SW_LIVE_BIN="${SW_LIVE_BIN:-$here/.venv/bin/sw-live}"
ALERT_CMD="${ALERT_CMD:-python3 $here/../quant-airflow/collectors/alert.py}"
job="${1:-?}"

"$SW_LIVE_BIN" "$@"
rc=$?
case "$rc" in
  0) exit 0 ;;
  2) level=warn;  what="건너뜀(시간창 밖·잠금 경합·인자 오류)" ;;
  4) level=error; what="신규 진입 중단(입력 신선도·DB·최신일·NXT 보정 실패) — 보유 결정·자산은 수행" ;;
  3) level=error; what="상태 손상 — 작업 중단, 사람 확인" ;;
  1) level=error; what="작업 오류" ;;
  *) level=error; what="알 수 없는 종료 코드" ;;
esac
echo "[sw-live-cron] $(date '+%F %T') job=$job rc=$rc level=$level $what"
# shellcheck disable=SC2086 — ALERT_CMD 는 명령+인자 문자열이다
$ALERT_CMD "$level" "sw-live $job rc=$rc: $what" </dev/null || true
exit "$rc"
