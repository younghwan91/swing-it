# 프로세스 그림 (archify)

- `paper-daily.workflow.html` — 매일 모의매매 파이프라인(`scripts/paper_daily.sh`)과 발견한 문제점 P1~P5. 원본 `paper-daily.workflow.json`, 근거 줄은 리비전 a87cbe4 에 고정.

## 다시 그리기
스킬은 커밋하지 않는다(18MB). `skills-lock.json` 의 버전으로 이 워크트리에 설치한다:

```bash
npx -y skills@latest add https://archify.si --skill archify --agent claude-code --copy --yes
node .claude/skills/archify/bin/archify.mjs doctor
```

이 머신(Ubuntu)은 AppArmor 가 Chromium 샌드박스를 막는다. 브라우저 점검은 `--no-sandbox` 래퍼를 `ARCHIFY_CHROME` 으로 준다:

```bash
printf '#!/bin/sh\nexec ~/.cache/ms-playwright/chromium-1140/chrome-linux/chrome --no-sandbox "$@"\n' > /tmp/chrome-ns && chmod +x /tmp/chrome-ns
ARCHIFY_CHROME=/tmp/chrome-ns node .claude/skills/archify/bin/archify.mjs finalize workflow docs/process/paper-daily.workflow.json \
  docs/process/paper-daily.workflow.html --repo-root . --quality showcase --out-dir docs/process/review-N --json
```
