"""월간 리밸런스 북 — 트랜치 순위 북(R)과 최적화 북(O)의 회계. 7·8·9단계의 실제 구현이다.

PEAD 단계 B(``research/experiments/pead_optimizer_b.py``)에서 옮겼고, 그 러너는 이제 여기를 부른다. 옮긴 뒤에도 ``B.json`` 을 1e-10 안에서
재현한다(``tests/real_db/test_b_json_reproduction.py``). 기본값(:class:`~swing_it.book.spec.BookSpec`)은 그 사전등록 값이다.

- :mod:`~swing_it.book.spec` 규칙 한 벌. :mod:`~swing_it.book.context` 패널 묶음·스텝 기록.
- :mod:`~swing_it.book.rank` 순위 북. :mod:`~swing_it.book.optimized` 최적화 북(core 지연 import).
- :mod:`~swing_it.book.costs` 정가 비용. :mod:`~swing_it.book.risk` 위험모델 부분집합.
- :mod:`~swing_it.book.accounting` 회계 엔진 ``simulate``. :mod:`~swing_it.book.metrics` 요약. :mod:`~swing_it.book.nulls` 순열 널.
- :mod:`~swing_it.book.parity` core 와의 패리티.

판정은 없다. 바는 연구마다 사전등록하고 러너에 둔다.
"""
