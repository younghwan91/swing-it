"""``research/logs/pead_optimizer/B.json`` 재현 — 옮긴 구현이 실데이터에서 판정 숫자를 전부 그대로 내는가.

실제 DB 와 core ``13c56a7``(B 를 만든 판)이 필요해 CI 에서는 돌지 않는다. 약 7분. ``research`` 잡의 건너뛰기 금지 범위 밖이다.

    set -a; . ~/git/quant-airflow/.env; set +a
    SWING_IT_B_CACHE=<레포 밖> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg --extra dev \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>@13c56a7" \\
      python -m pytest -q tests/real_db/test_b_json_reproduction.py

``SWING_IT_B_OUT`` 을 주면 대조 결과를 그 경로에 JSON 으로 쓴다(성과 숫자 없음, 개수와 최대 차만).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-10


def compare(got, want, path: str = "", bad: "list | None" = None, stat: "list | None" = None):
    """깊은 대조. 숫자는 ``TOL``, NaN 끼리는 같다, 문자열·불·None 은 정확히 같아야 한다. 어긋난 곳 목록과 (개수, 최대 차)."""
    bad = [] if bad is None else bad
    stat = [0, 0.0] if stat is None else stat
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(got) | set(want)):
            if k not in got or k not in want:
                bad.append((f"{path}/{k}", "key", k in got, k in want))
            else:
                compare(got[k], want[k], f"{path}/{k}", bad, stat)
    elif isinstance(want, list) and isinstance(got, list):
        if len(got) != len(want):
            bad.append((path, "len", len(got), len(want)))
        else:
            for i, (g, w) in enumerate(zip(got, want)):
                compare(g, w, f"{path}[{i}]", bad, stat)
    elif any(isinstance(v, (bool, str)) or v is None for v in (got, want)):
        if got != want:
            bad.append((path, "value", got, want))
    else:
        stat[0] += 1
        g, w = float(got), float(want)
        if not (math.isnan(g) and math.isnan(w)):
            d = abs(g - w)
            stat[1] = max(stat[1], d if d == d else math.inf)
            if not d <= TOL:
                bad.append((path, "num", g, w))
    return bad, stat


def test_compare_catches_each_kind_of_difference():
    """대조기가 죽어 있으면 이 재현은 아무것도 증명하지 않는다."""
    assert compare({"a": 1.0}, {"a": 1.0})[0] == []
    assert compare({"a": 1.0}, {"a": 1.0 + 1e-9})[0]
    assert compare({"a": float("nan")}, {"a": float("nan")})[0] == []
    assert compare({"a": float("nan")}, {"a": 1.0})[0]
    assert compare({"a": "x"}, {"a": "y"})[0]
    assert compare({"a": 1.0}, {"b": 1.0})[0]
    assert compare([1.0], [1.0, 2.0])[0]
    assert compare({"a": None}, {"a": 1.0})[0]


@pytest.mark.skipif(not os.environ.get("SWING_IT_B_CACHE"), reason="실제 DB·core 13c56a7 필요 — SWING_IT_B_CACHE 를 준다")
def test_promoted_code_reproduces_b_json():
    sys.path.insert(0, str(REPO / "research" / "experiments"))
    import pead_optimizer_b as B

    t0 = time.time()
    cache = Path(os.environ["SWING_IT_B_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)

    class Args:
        core_ref = "13c56a7"
        workers = int(os.environ.get("SWING_IT_B_WORKERS", "12"))

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    ctx, info, x = B.build_ctx(cache, None, Args.workers, log, core_ref=Args.core_ref)
    pc = B.full_parity(ctx)
    assert pc["pass"], pc
    res, viol, steps = B.compute_b(Args, ctx, info, x, pc, log)
    assert viol == []
    res["invariants"] = {"violations": 0, "steps_checked": steps}
    got = json.loads(json.dumps(res, ensure_ascii=False, default=float))
    full = json.loads((REPO / "research/logs/pead_optimizer/B.json").read_text(encoding="utf-8"))
    bad, (n, worst) = compare(got, {k: full[k] for k in got})
    out = {"compared_numbers": n, "max_abs_diff": worst, "tol": TOL, "n_mismatch": len(bad), "core_ref": Args.core_ref,
           "sections": sorted(got), "elapsed_s": time.time() - t0,
           "note": "B.json 의 cells·ic·invariants·mode·parity·primary 전부를 대조했다. code·fingerprints·elapsed_s 는 실행마다 달라 뺐다."}
    if os.environ.get("SWING_IT_B_OUT"):
        Path(os.environ["SWING_IT_B_OUT"]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    assert bad == [], bad[:20]
