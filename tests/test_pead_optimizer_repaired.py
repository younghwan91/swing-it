"""PO-5.2 강건성 재실행 러너 — 관문·분기·비교·렌더(합성·기존 JSON 만, DB 불요, portfolio-layer 필요)."""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("krx_quant_core.portfolio")
EXP = Path(__file__).resolve().parents[1] / "research" / "experiments"
LOGS = Path(__file__).resolve().parents[1] / "research" / "logs" / "pead_optimizer"
sys.path.insert(0, str(EXP))
import pead_optimizer_po4 as P4  # noqa: E402
import pead_optimizer_repaired as RP  # noqa: E402

A1 = json.loads((LOGS / "A1.json").read_text())
FP_PRE = A1["fingerprint_before"]["daily_bars_adjusted"]


def test_guard_refuses_pre_repair_table():
    with pytest.raises(SystemExit, match="아직 배포되지 않았다"):
        RP.repair_guard(list(FP_PRE), FP_PRE)
    RP.repair_guard([FP_PRE[0] + 1, FP_PRE[1], FP_PRE[2] - 7, FP_PRE[3]], FP_PRE)  # 수리 뒤 — 통과
    with pytest.raises(SystemExit, match="최대 날짜"):
        RP.repair_guard([FP_PRE[0], "2026-09-10", FP_PRE[2] + 1, FP_PRE[3]], FP_PRE)
    with pytest.raises(SystemExit, match="모양"):
        RP.repair_guard([1, 2], FP_PRE)
    # repro-check 는 반대 — 수리 전 표에서만
    RP.repair_guard(list(FP_PRE), FP_PRE, expect_repaired=False)
    with pytest.raises(SystemExit, match="수리 전 표에서만"):
        RP.repair_guard([FP_PRE[0] + 1, FP_PRE[1], FP_PRE[2], FP_PRE[3]], FP_PRE, expect_repaired=False)


def test_out_guard_never_overwrites_originals(tmp_path):
    with pytest.raises(AssertionError):
        RP.out_guard("A1", tmp_path)
    with pytest.raises(AssertionError):
        RP.out_guard("C0", tmp_path)
    RP.out_guard("A1_repaired", tmp_path)
    assert not set(f"{s}.json" for s in RP.STEMS) & set(RP.PROTECTED)


def test_out_guard_once_only(monkeypatch, tmp_path):
    monkeypatch.setattr(RP, "OUT", tmp_path)
    (tmp_path / "A1_repaired.json").write_text("{}")
    with pytest.raises(SystemExit, match="한 번만"):
        RP.out_guard("A1_repaired", tmp_path)
    RP.out_guard("C0_repaired", tmp_path)


@pytest.mark.parametrize("new,label,opens", [("통과", None, True), ("실패", "취약", False), ("VOID", "취약", False)])
def test_branch_po52(new, label, opens):
    b = RP.repaired_branch(new, "통과")
    assert b["label"] == label and b["B_opens"] is opens and b["changed"] is (new != "통과")
    assert ("전진" in b["next"]) is (not opens)


def test_compare_po2_identity_and_perturbation():
    po2 = json.loads((LOGS / "PO2.json").read_text())
    same = copy.deepcopy(po2)
    cmp = RP.compare_po2(same, po2)
    assert RP.repro_exact(cmp) and cmp["max_abs_diff_realized"] == 0
    assert "OB21:band" in cmp["realized"] and not any(k.endswith(":noband") for k in cmp["realized"])
    moved = copy.deepcopy(po2)
    moved["sigma_target"]["value"] += 1e-4
    assert not RP.repro_exact(RP.compare_po2(moved, po2))
    md = RP.render_c0_repaired({"compare_PO2": cmp, "fingerprint_full": [1], "fingerprint_old": [2], "deployed_on": "2026-11-10",
                                "swing_it_sha": "x", "core_ref": "13c56a7"})
    assert "C0_repaired" in md and "| sigma_target | value |" in md


def test_render_a1_repaired_header_and_no_original_review():
    r = copy.deepcopy(A1)
    r["repaired"] = {**RP.repaired_branch("실패", "통과"), "deployed_on": "2026-11-10", "swing_it_sha": "abc",
                     "fingerprint_pre_repair": FP_PRE, "sigma_target_fixed": 0.04013, "sigma_target_rebuilt": 0.0405}
    md = RP.render_a1_repaired(r)
    assert md.startswith("# 강건성 재실행 A1_repaired") and '라벨 "취약"' in md
    assert "독립 리뷰 반영 (정정" not in md  # 원래 A1 리뷰 절은 넣지 않는다


def test_render_a1_original_unchanged_by_review_param():
    """po4.render_a1 에 ``review`` 인자를 더했다 — 원래 A1.md 는 바이트 그대로 나와야 한다."""
    assert P4.render_a1(A1) == (LOGS / "A1.md").read_text()


def test_a1_mode_patches_recheck_used_by_run_new():
    """a1 은 ``po4.run_new`` 를 그대로 부르고 시작 재확인만 바꾼다 — run_new 가 모듈 전역 ``recheck_old`` 를 부르는지 본다."""
    tree = ast.parse((EXP / "pead_optimizer_po4.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_new")
    called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "recheck_old" in called
    src = (EXP / "pead_optimizer_repaired.py").read_text()
    assert "P4.recheck_old = recheck_old_report_only(holder)" in src
    assert '"OB-new"' in (EXP / "pead_optimizer_po4.py").read_text()  # 신창 α 키 고정은 run_new 안에 있다
