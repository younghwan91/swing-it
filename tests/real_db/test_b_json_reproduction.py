"""``research/logs/pead_optimizer/B.json`` 재현 — 옮긴 구현이 실데이터에서 판정 숫자를 전부 그대로 내는가.

실제 DB 와 core ``13c56a7``(B 를 만든 판)이 필요해 CI 에서는 돌지 않는다. 약 7분. ``research`` 잡의 건너뛰기 금지 범위 밖이다.

    set -a; . ~/git/quant-airflow/.env; set +a
    SWING_IT_B_CACHE=<레포 밖> OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg --extra dev \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>@13c56a7" \\
      python -m pytest -q tests/real_db/test_b_json_reproduction.py

``SWING_IT_B_OUT`` 을 주면 대조 결과를 그 경로에 JSON 으로 쓴다(성과 숫자 없음, 개수와 최대 차만).

**표 지문을 먼저 본다.** B.json 은 지문 ``e66f309e07de`` 표에서 나왔다. 2026-11-02 16:55 재계산(PR #14, 052670 ×1500 등)으로 지문이 ``400dfcc393ab`` 가 됐다.
모델이 움직여 R 은 그대로이고 O 만 어긋난다 — :func:`reproduction_verdict` 가 "표 바뀜"과 "코드 회귀"를 가른다.

**IC 절은 기본으로 건너뛴다.** IC 패널은 ``load_risk_inputs(2016-11-01)`` 인데 준비 400일이 붙어 약 2015-09 부터 읽는다. 2009-01~2016-08 잠금 대역 안이다.
그래서 기본은 B.json 의 ``ic_by_year`` 를 입력으로 쓰고 ``ic`` 절을 대조하지 않는다. ``SWING_IT_B_FULL_IC=1`` 은 잠금 대역을 읽으니 smallcap 판정 전에는 쓰지 않는다.

**솔버 판.** ``uv.lock`` 에 cvxpy·clarabel 이 없어 새 환경에서 O 가 5e-6 안팎 움직인다. 결과 JSON 에 판을 적는다.

**코드 회귀의 진짜 검사.** B.json 대조는 표·솔버 차이를 함께 본다. ``SWING_IT_B_BASELINE=<옛 트리 출력 JSON>`` 을 주면 같은 표·같은 환경에서 옛 코드가 낸 값과 맞대어 하나라도 다르면 실패한다.
옛 출력은 ``SWING_IT_B_GOT`` 으로 만든다. 성과 숫자가 들어 있으니 레포 밖에 둔다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-10
SECTIONS = ("cells", "ic", "invariants", "mode", "parity", "primary")  # code·fingerprints·elapsed_s 는 실행마다 다르다
FP_OF_B_JSON = "e66f309e07de"
#: 지문별 알려진 O 차이 폭(B.json 은 지문 e66f309e07de). R 은 어느 표에서도 그대로다.
#: 400dfcc393ab 의 IR 차이는 솔버 환경에 달렸다: 한 환경 3.83e-6(검증 9단계 bmodels·docenv), 다른 환경 5.65e-6(cvxpy 1.9.3, clarabel 0.11.1, 둘 다 uv.lock 에 없다).
#: 5.65e-6 쪽은 옮기기 전 코드(03fee83)도 같은 표·같은 환경에서 830개 값이 전부 같게 냈다 — 코드가 만든 차이가 아니다. 폭은 진짜 회귀 검사가 아니라 안전망이다.
#: 코드 회귀는 ``SWING_IT_B_BASELINE`` 으로 옛 트리 출력과 맞대어 본다.
KNOWN_TABLE_BOUNDS = {"400dfcc393ab": {"ir": 6e-6, "count": 5, "other": 5e-4}}
_O_PATH = re.compile(r"/O/|^/cells/|^/primary/(rho|boot_dIR|report/f_boot|verdict/(dIR_|exante_risk_ratio|null_q95_O|realized_active_risk_ratio))")
_IR_PATH = re.compile(r"/O/IR_\dx$|/verdict/dIR_\dx$")
_COUNT_PATH = re.compile(r"/n_[a-z_]+$")


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


def compare_sections(got: dict, want: dict, *, skip: tuple = ()) -> tuple:
    """:data:`SECTIONS` 전부를 대조한다. ``got`` 에 없는 절은 어긋남이다. ``skip`` 은 일부러 안 본 절이고 결과에 적는다."""
    bad: list = []
    stat = [0, 0.0]
    for sec in SECTIONS:
        if sec in skip:
            continue
        if sec not in got:
            bad.append((f"/{sec}", "section", False, True))
        else:
            compare(got[sec], want[sec], f"/{sec}", bad, stat)
    return bad, stat


def fp_short(fp) -> str:
    """모델 캐시 이름과 같은 표 지문 해시(``pead_optimizer_b.model_cache_name``)."""
    return hashlib.sha256(json.dumps(fp, default=str).encode()).hexdigest()[:12]


def reproduction_verdict(bad: list, *, fp_short: str) -> dict:
    """어긋난 목록을 판정한다. "같음"·"표 바뀜"(R 그대로, O 는 알려진 폭 안)·"코드 회귀"·"모르는 표". O 차이는 숨기지 않고 개수·최대를 낸다."""
    nums = [b for b in bad if b[1] == "num"]
    o = [b for b in nums if _O_PATH.search(b[0])]
    diff = lambda b: abs(float(b[2]) - float(b[3]))  # noqa: E731
    ir = [diff(b) for b in o if _IR_PATH.search(b[0])]
    out = {"n_bad": len(bad), "n_O": len(o), "max_O": max(map(diff, o), default=0.0), "max_IR": max(ir, default=0.0), "fingerprint": fp_short,
           "worst_O": sorted(((b[0], diff(b)) for b in o), key=lambda x: -x[1])[:5]}
    if not bad:
        return {**out, "kind": "같음"}
    if fp_short == FP_OF_B_JSON:
        return {**out, "kind": "코드 회귀", "why": "표가 B.json 과 같은데 어긋난다"}
    non_o = [b[0] for b in bad if not (b[1] == "num" and _O_PATH.search(b[0]))]
    if non_o:
        return {**out, "kind": "코드 회귀", "why": f"O 밖이 어긋난다: {non_o[:5]}"}
    if fp_short not in KNOWN_TABLE_BOUNDS:
        return {**out, "kind": "모르는 표", "why": "이 지문의 O 폭을 잰 적이 없다. 사람이 보고 KNOWN_TABLE_BOUNDS 에 적는다"}
    lim = KNOWN_TABLE_BOUNDS[fp_short]
    over = [b[0] for b in o if diff(b) > (lim["ir"] if _IR_PATH.search(b[0]) else lim["count"] if _COUNT_PATH.search(b[0]) else lim["other"])]
    if over:
        return {**out, "kind": "코드 회귀", "why": f"표 변화의 알려진 폭을 넘었다: {over[:5]}"}
    return {**out, "kind": "표 바뀜", "why": f"지문 {FP_OF_B_JSON} → {fp_short}"}


def test_compare_sections_catches_a_missing_top_level_section():
    """옛 대조는 ``{k: full[k] for k in got}`` 라 ``got`` 에서 빠진 절을 통째로 못 봤다."""
    want = {s: {"a": 1.0} for s in SECTIONS} | {"code": "x"}
    got = {s: {"a": 1.0} for s in SECTIONS}
    assert compare_sections(got, want)[0] == []
    partial = {k: v for k, v in got.items() if k != "primary"}
    assert ("/primary", "section", False, True) in compare_sections(partial, want)[0]
    assert compare_sections(partial, want, skip=("primary",))[0] == []


def _bad(path, g=0.0, w=1e-7, kind="num"):
    return (path, kind, g, w)


def test_verdict_tells_a_changed_table_from_a_code_regression():
    """표가 그대로인데 어긋나면 코드 회귀다. 표가 바뀌었으면 R 은 그대로여야 하고 O 는 알려진 폭 안이어야 한다."""
    same, fp = "e66f309e07de", "400dfcc393ab"
    assert reproduction_verdict([], fp_short=same)["kind"] == "같음"
    assert reproduction_verdict([_bad("/primary/O/IR_2x")], fp_short=same)["kind"] == "코드 회귀"
    assert reproduction_verdict([], fp_short=fp)["kind"] == "같음"
    ok = reproduction_verdict([_bad("/primary/O/IR_2x", 1.0, 1.0 + 1e-6), _bad("/cells/c/O/n_dust", 3, 0), _bad("/primary/verdict/null_q95_O", 1.0, 1.0 + 2e-4)], fp_short=fp)
    assert ok["kind"] == "표 바뀜" and ok["n_O"] == 3 and ok["max_O"] == pytest.approx(3.0) and ok["max_IR"] == pytest.approx(1e-6)
    assert reproduction_verdict([_bad("/primary/R/IR_2x", 1.0, 1.0 + 1e-6)], fp_short=fp)["kind"] == "코드 회귀", "R 은 표 변화와 무관하게 그대로다"
    assert reproduction_verdict([_bad("/primary/O/IR_2x", 1.0, 1.0 + 1e-4)], fp_short=fp)["kind"] == "코드 회귀", "알려진 폭을 넘으면 표로 설명되지 않는다"
    assert reproduction_verdict([_bad("/primary/verdict/verdict", "a", "b", "value")], fp_short=fp)["kind"] == "코드 회귀", "라벨은 표 변화로 바뀌면 안 된다"
    assert reproduction_verdict([_bad("/primary/O/IR_2x")], fp_short="000000000000")["kind"] == "모르는 표", "모르는 표의 폭은 아무도 모른다"


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

    import pead_optimizer_po4 as P4

    t0 = time.time()
    cache = Path(os.environ["SWING_IT_B_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    full = json.loads((REPO / "research/logs/pead_optimizer/B.json").read_text(encoding="utf-8"))
    fp = fp_short(P4.fingerprint("daily_bars_adjusted", B.END_B))
    print(f"표 지문 {fp} (B.json {FP_OF_B_JSON}) — {'같은 표' if fp == FP_OF_B_JSON else '표가 바뀌었다. O 는 어긋날 수 있고 R 은 어긋나면 안 된다'}", flush=True)
    assert fp_short(full["fingerprints"]["daily_bars_adjusted_b_range"]) == FP_OF_B_JSON

    class Args:
        core_ref = "13c56a7"
        workers = int(os.environ.get("SWING_IT_B_WORKERS", "12"))

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    skip: tuple = ()
    if not os.environ.get("SWING_IT_B_FULL_IC"):        # 잠금 대역을 읽지 않는다(모듈 설명)
        ic = full["ic"]
        ic_by = {int(h): {int(y): float(v) for y, v in d.items()} for h, d in ic["ic_by_year"].items()}
        ic_info = {**ic, "zero_folds": {int(k): v for k, v in ic["zero_folds"].items()}, "n_days": {int(k): v for k, v in ic["n_days"].items()}}
        B.ic_by_year_for = lambda args, _log: (ic_by, ic_info)
        skip = ("ic",)

    ctx, info, x = B.build_ctx(cache, None, Args.workers, log, core_ref=Args.core_ref)
    pc = B.full_parity(ctx)
    assert pc["pass"], pc
    res, viol, steps = B.compute_b(Args, ctx, info, x, pc, log)
    assert viol == []
    res["invariants"] = {"violations": 0, "steps_checked": steps}
    got = json.loads(json.dumps(res, ensure_ascii=False, default=float))
    bad, (n, worst) = compare_sections(got, full, skip=skip)
    v = reproduction_verdict(bad, fp_short=fp)
    versions = {p: importlib.metadata.version(p) for p in ("cvxpy", "clarabel", "numpy", "scipy")}
    out = {"compared_numbers": n, "max_abs_diff": worst, "tol": TOL, "n_mismatch": len(bad), "core_ref": Args.core_ref,
           "sections_compared": [s for s in SECTIONS if s not in skip], "sections_skipped": list(skip), "elapsed_s": time.time() - t0,
           "verdict": v, "R_mismatch": [b[0] for b in bad if "/R/" in b[0]], "versions": versions,
           "note": "code·fingerprints·elapsed_s 는 실행마다 달라 뺐다. ic 절을 건너뛰면 IC 는 B.json 의 ic_by_year 를 입력으로 쓴 것이다."}
    if os.environ.get("SWING_IT_B_GOT"):            # 코드끼리 대조할 때만 — 성과 숫자가 들어 있으니 레포 밖에 둔다
        Path(os.environ["SWING_IT_B_GOT"]).write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
    base_bad: list = []
    if os.environ.get("SWING_IT_B_BASELINE"):
        base = json.loads(Path(os.environ["SWING_IT_B_BASELINE"]).read_text(encoding="utf-8"))
        base_bad, (base_n, base_worst) = compare_sections(got, base, skip=skip)
        out["vs_baseline"] = {"path": Path(os.environ["SWING_IT_B_BASELINE"]).name, "compared_numbers": base_n, "n_mismatch": len(base_bad), "max_abs_diff": base_worst}
    if os.environ.get("SWING_IT_B_OUT"):
        Path(os.environ["SWING_IT_B_OUT"]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    assert base_bad == [], base_bad[:20]
    assert v["kind"] in ("같음", "표 바뀜"), (v, bad[:20])
