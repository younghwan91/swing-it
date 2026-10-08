"""뉴스 LLM 판정 — 스윙 2호(뉴스 지속성) 의 쌍 생성·판정·의미 정답지 도구.

2026-10-08 daytrade-it 연구 브랜치(``research/news-persistence-judge``)에서 이식했다. swing-it 이 자기 판정을 소유한다.

    prompt     운영 v2 추출 프롬프트(바이트 동일 사본)·스키마 파서·입력 해시·스윙 보충
    prefilter  판정 전 결정론 필터(roundup·팬아웃·지수 시황·스윙 태그) — 전부 파라미터
    sweep      쌍 키(uuid5)·유니버스 선택·장중 순회 도달 재현
    pairs      quant-airflow DB(토스 종목별 피드)에서 쌍 파일을 만든다      ``python -m swing_it.news.pairs``
    judge      Message Batches 판정기(예산·재개·캐싱·실측 usage)          ``python -m swing_it.news.judge``
    labelset   의미 정답지(교사 라벨)와 프롬프트 변형 채점                  ``python -m swing_it.news.labelset``

데이터(쌍 파일·판정 캐시·정답지)는 ``data/eval/`` (gitignore) 에 둔다. 수익 데이터는 이 패키지 어디에도 없다.
"""
