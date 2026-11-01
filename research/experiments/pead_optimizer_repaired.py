"""PEAD 최적화 정정 PO-5.2 — PR #12(``daily_bars_adjusted`` 이음매 수리) 배포 뒤 강건성 재실행. 정정 PO-7 에서 구현 고정.

수리된 표로 **한 번씩** 돈다. 출력은 ``C0_repaired.*``·``A1_repaired.*`` 뿐이다 — ``A1.json``·``C0.json``·``PO2.json`` 을 덮지 않는다.

- ``--mode c0`` (PO-2.1-6): PO-2 러너의 C0 부분(가드 끈 R0·가드 켠 R0·σ_target·OB(밴드 0.10)·구창 실현 바이어스·CV_s·ν_u·ν_f)을
  **같은 코드·같은 정의**(v3, PO-2 이음매 가드 — ``secondary_min_abs_ret`` 없음, END 2023-12-28)로 다시 내고 ``PO2.json`` 과의 차이를 보고만 한다.
  가드 끈 R0 가 설계 §6 과 0.005 넘게 어긋나도 멈추지 않는다(표가 바뀐 것이 이 실행의 전제다) — 보고만.
- ``--mode a1`` (PO-5.2-2): ``pead_optimizer_po4.run_new`` 를 **그대로** 부른다. 위험모델(v4 구창·신창)·B 유니버스·OB·실현 수익을 새 캐시에서
  처음부터 다시 만들고, σ_target(``PO4_repro.json`` 0.040130…)·밴드(``PO3.json``)·α 난수 키 ``"OB-new"``·판정 규칙(PO-3.6·PO-4.3)은 고정이다.
  바꾸는 곳은 하나 — 시작 때의 구창 재확인(``recheck_old``)이 수리된 표에서는 맞을 수 없으므로 멈추지 않고 보고만 하며, 수리된 표로 다시 낸
  σ_target 을 같이 적는다(보고만). 그 뒤 PO-5.2-3 분기: 판정이 ``통과`` 면 라벨 없음·B 를 연다, 아니면(실패·VOID) **"취약"**·v4 는 전진 A·
  B 는 역사 구간에서 열지 않는다.
- ``--mode c0-repro-check``: **수리 전 표에서만** 돈다(반대 관문). c0 경로가 ``PO2.json`` 을 소수 5자리로 다시 내는지 레포 밖(``--out``)에만
  쓴다 — 수리 뒤 실행 전에 코드 경로를 확인하는 용도. 신창은 읽지 않는다.

관문: ``daily_bars_adjusted`` 2019-01-01~2026-09-11 지문(PO-4.2 정의)이 ``A1.json`` 의 ``fingerprint_before`` 와 **같으면** 수리가 배포되지
않은 것이므로 c0·a1 은 돌지 않는다. 출력 파일이 이미 있어도 돌지 않는다(한 번만).

    set -a; . ~/git/quant-airflow/.env; set +a
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 uv run --extra pg \\
      --with "krx-quant-core[portfolio,db,fast] @ git+file://<core>/portfolio-layer@13c56a7" \\
      python research/experiments/pead_optimizer_repaired.py --mode c0 --deployed-on YYYY-MM-DD --cache <scratch>/c0rep
    ... --mode a1 --deployed-on YYYY-MM-DD --cache <scratch>/a1rep --po3-cache <scratch>/po3cache
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "logs" / "pead_optimizer"
FULL_END = "2026-09-11"
OLD_END = "2023-12-28"
STEMS = ("C0_repaired", "A1_repaired")
PROTECTED = ("A1.json", "A1.md", "C0.json", "C0.md", "PO2.json", "PO2.md", "PO3.json", "PO4_repro.json")
LABEL_FRAGILE = "취약"


# ── 관문(순수 함수 — 시험 가능) ──────────────────────────────────────────────


def repair_guard(fp_now: list, fp_pre: list, *, expect_repaired: bool = True) -> None:
    """수리 배포 여부 관문. ``fp_pre`` = ``A1.json`` 의 ``fingerprint_before.daily_bars_adjusted``(수리 전)."""
    if not (isinstance(fp_now, (list, tuple)) and len(fp_now) == 4):
        raise SystemExit(f"지문 모양이 이상하다: {fp_now!r}")
    if str(fp_now[1]) != FULL_END:
        raise SystemExit(f"지문 최대 날짜 {fp_now[1]} ≠ {FULL_END} — 표가 잘렸거나 범위가 다르다")
    same = list(map(str, fp_now)) == list(map(str, fp_pre))
    if expect_repaired and same:
        raise SystemExit("daily_bars_adjusted 지문이 A1.json(수리 전)과 같다 — PR #12 가 아직 배포되지 않았다. 돌지 않는다(PO-5.2-1)")
    if not expect_repaired and not same:
        raise SystemExit("repro-check 는 수리 전 표에서만 돈다 — 지문이 A1.json 과 다르다")


def out_guard(stem: str, outdir: Path) -> None:
    """출력은 ``C0_repaired``·``A1_repaired`` 만, 레포 로그에서는 한 번만."""
    assert stem in STEMS, stem
    assert f"{stem}.json" not in PROTECTED
    if outdir.resolve() == OUT.resolve() and (outdir / f"{stem}.json").exists():
        raise SystemExit(f"{stem}.json 이 이미 있다 — 강건성 재실행은 한 번만 한다(PO-5.2)")


def repaired_branch(verdict_repaired: str, verdict_original: str) -> dict:
    """PO-5.2-3 — 기계적 분기. VOID 도 '바뀜'으로 센다."""
    ok = verdict_repaired == "통과"
    return {"verdict_original": verdict_original, "verdict_repaired": verdict_repaired,
            "changed": verdict_repaired != verdict_original, "label": None if ok else LABEL_FRAGILE,
            "B_opens": ok,
            "next": ("라벨 없음. B 를 연다 — 수리된 표로 패리티를 다시 통과한 뒤(PO-5.2-4)." if ok else
                     "A 결과에 \"취약\" 라벨. v4 판정은 전진 A(PO-5.4)로 넘긴다. B 는 역사 구간(2020~2023)에서 열지 않는다.")}


def diff_table(before: dict, after: dict, keys) -> dict:
    """``before``·``after`` 의 같은 키 숫자 차(after − before). 없는 키는 None."""
    out = {}
    for k in keys:
        b, a = before.get(k), after.get(k)
        d = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        out[k] = {"before": b, "after": a, "diff": d}
    return out


def git_sha() -> str:
    r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip()


# ── c0 (PO-2 러너 C0 부분, 같은 정의) ──────────────────────────────────────────


def c0_part(cache: Path, workers: int, log) -> dict:  # pragma: no cover - DB
    import pead_optimizer_po2 as P2
    import pead_optimizer_po3 as P3
    import pead_optimizer_po4 as P4
    from krx_quant_core.backtest.crosssectional import _trailing_adv
    from krx_quant_core.risk.factor_model import factor_returns
    from pead_optimizer_c0 import N_OB
    from pead_optimizer_seam import seam_guard

    x0 = P4.load_inputs(OLD_END)
    raw_src, events = P4.raw_closes_and_events(x0.codes, x0.dates, OLD_END)
    rf, cf, seam, seam_info = seam_guard(x0.codes, x0.dates, x0.ret, x0.close, raw_src, events)  # PO-2 가드(같은 코드)
    del raw_src
    x = dataclasses.replace(x0, ret=rf, close=cf)
    c_raw = P4.prep(x0, *P4.OLD_ASOF, OLD_END)
    m_raw = P3.build_models(c_raw, c_raw["asofs"], workers, cache / "models_noguard.pkl", log)
    r0_noguard = P2.r0_values(c_raw, m_raw)
    del c_raw, m_raw
    c = P4.prep(x, *P4.OLD_ASOF, OLD_END)
    R, cap, lab, dates, asofs, sectors, first = (c["R"], c["cap"], c["lab"], c["dates"], c["asofs"], c["sectors"], c["first"])
    nN, nD = R.shape
    sig_all, sig_info = P4.yoy_signal_strict(x.codes, x.dates, OLD_END)
    adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
    sig, adv = sig_all[:, first:], adv_all[:, first:]
    t_0 = int(np.searchsorted(dates, np.datetime64(P4.OLD_GRID[0])))
    grid = [t for t in range(t_0, nD, 20) if dates[t] <= pd.Timestamp(P4.OLD_GRID[1])]
    models = P3.build_models(c, sorted(set(asofs) | set(grid)), workers, cache / "models_guard_v3.pkl", log)
    r0_guard = P2.r0_values(c, models)
    st, st_rows = P3.sigma_target_of(models, grid, sig, adv, cap, nN)
    bu = P3.b_universe(models, asofs, sig, adv, cap, nN)
    ob = P4.solve_ob(models, asofs, bu, lab, st, workers, cache / f"ob_band_{st:.6f}.pkl", "OB")
    ob_stat = {k: int(sum(ob[n][1][k] for n in ob)) for k in ("ok", "lam_fail", "solver_fail", "solves")}
    U = factor_returns(R, c["Xs"], lab, cap, c["msk"]).u
    ev = P4.evaluate(c, models, bu, ob, seam[:, first:], U, "")
    tails = P4.tails(ev["spec_z"])
    nu_f, k_f, n_f = P4.nu_f_of(ev["Zr"], sectors)
    realized = dict(ev["realized"])
    realized["OB21:band"] = realized.pop("OB21")
    realized["OB5:band"] = realized.pop("OB5")
    return {"window": [str(dates[asofs[0]].date()), str(dates[asofs[-1]].date())], "n_asof": len(asofs),
            "data_end": str(dates[-1].date()), "seam": seam_info, "signal": sig_info,
            "nan_to_zero_cells": ev["nan_to_zero"], "R0_noguard": r0_noguard, "R0_guard": r0_guard,
            "tails": {**tails, "nu_f": nu_f, "kurt_f": k_f, "n_f": n_f},
            "sigma_target": {"value": st, "n": len(st_rows), "q10": float(np.quantile(st_rows, 0.1)),
                             "q90": float(np.quantile(st_rows, 0.9))},
            "OB": {"band": ob_stat}, "ob_ok_share": ob_stat["ok"] / (N_OB * len(asofs)),
            "B_universe_size": [int(min(len(v) for v in bu.values())), int(max(len(v) for v in bu.values()))],
            "old_realized_report_only": {k: v for k, v in realized.items() if k not in ("capw", "LO5", "LO50", "KOSPI")}}


def compare_po2(new: dict, po2: dict) -> dict:
    """``PO2.json`` 대비 차이(보고만)."""
    out = {"R0_noguard": diff_table(po2["R0_noguard"]["values"], new["R0_noguard"]["values"], po2["R0_noguard"]["values"]),
           "R0_guard": diff_table(po2["R0_guard"]["values"], new["R0_guard"]["values"], po2["R0_guard"]["values"]),
           "tails": diff_table(po2["tails"], new["tails"], ("CV_med", "CV_split", "nu_u", "kurt_u", "nu_f", "kurt_f")),
           "sigma_target": diff_table(po2["sigma_target"], new["sigma_target"], ("value", "n", "q10", "q90")),
           "OB_band": diff_table(po2["OB"]["band"], new["OB"]["band"], ("ok", "lam_fail", "solver_fail")),
           "realized": diff_table(po2["old_realized_report_only"], new["old_realized_report_only"],
                                  [k for k in po2["old_realized_report_only"] if not k.endswith(":noband")]),
           "seam_cells": diff_table(po2["seam"], new["seam"], ("seam_cells", "seam_codes")),
           "nan_to_zero": diff_table(po2["nan_to_zero_cells"], new["nan_to_zero_cells"], ("seam", "other"))}
    out["max_abs_diff_realized"] = max((abs(v["diff"]) for v in out["realized"].values() if v["diff"] is not None), default=None)
    return out


def repro_exact(cmp: dict) -> bool:
    """c0-repro-check — 소수 5자리(PO2.json 반올림)로 전부 같으면 참."""
    for blk in cmp.values():
        if not isinstance(blk, dict):
            continue
        for v in blk.values():
            if isinstance(v, dict) and v.get("diff") is not None and abs(v["diff"]) > 6e-6:
                return False
    return True


# ── a1 (po4.run_new 그대로, 시작 재확인만 보고만) ──────────────────────────────


def recheck_old_report_only(holder: dict):  # pragma: no cover - DB
    """``po4.recheck_old`` 대신 — 수리된 표에서 구창 R0·v4 모델 비트는 맞을 수 없다. 멈추지 않고 보고만, σ_target 을 다시 낸다(보고만)."""
    import pead_optimizer_po2 as P2
    import pead_optimizer_po3 as P3
    import pead_optimizer_po4 as P4
    from krx_quant_core.backtest.crosssectional import _trailing_adv

    def recheck(args, cache: Path, po3, log) -> dict:
        x0 = P4.load_inputs(OLD_END)
        c_raw = P4.prep(x0, *P4.OLD_ASOF, OLD_END)
        m_raw = P3.build_models(c_raw, c_raw["asofs"], args.workers, cache / "old_models_noguard.pkl", log)
        r0 = P2.r0_values(c_raw, m_raw)
        r0_ok, r0_d = P4.r0_matches_po3(r0, po3)
        del c_raw, m_raw
        x, _, seam_info = P4.guarded(x0, OLD_END)
        c = P4.prep(x, *P4.OLD_ASOF, OLD_END)
        dates, first, nN = c["dates"], c["first"], c["R"].shape[0]
        t_0 = int(np.searchsorted(dates, np.datetime64(P4.OLD_GRID[0])))
        grid = [t for t in range(t_0, len(dates), 20) if dates[t] <= pd.Timestamp(P4.OLD_GRID[1])]
        models = P3.build_models(c, sorted(set(c["asofs"]) | set(grid)), args.workers, cache / "old_models_v4.pkl", log,
                                 vra=P4.V4["vra"], shrink=P4.V4["shrink"])
        keys = [c["asofs"][i] for i in P4.RECHECK_N]
        import pickle

        old = pickle.load(open(Path(args.po3_cache) / "models_vraoff_s0.1.pkl", "rb"))
        bit = P4.models_bit_same(old, models, keys)
        sig_all, _ = P4.yoy_signal_strict(x.codes, x.dates, OLD_END)
        adv_all = _trailing_adv(x.trade_value / 1e6, len(x.dates), 20, x.close)
        st, rows = P3.sigma_target_of(models, grid, sig_all[:, first:], adv_all[:, first:], c["cap"], nN)
        rc = {"R0": r0["values"], "R0_diff_vs_PO3": r0_d, "R0_match": r0_ok, "v4_models_checked": P4.RECHECK_N,
              "v4_models_bit_same": bit, "ok": True, "report_only": True,
              "note": "수리된 표 — R0·모델 비트 불일치는 예상된 것(PO-5.2), 멈추지 않는다",
              "old_seam_cells": seam_info["seam_cells"], "sigma_target_rebuilt": st, "sigma_target_rebuilt_n": len(rows)}
        holder["recheck"] = rc
        log(f"구창(수리된 표) R0 PO3 일치 {r0_ok}, v4 모델 비트 {bit}, σ_target 재계산 {st:.6f}(보고만)")
        return rc

    return recheck


def render_a1_repaired(r: dict) -> str:
    import pead_optimizer_po4 as P4

    rp = r.get("repaired", {})
    lab = f"라벨 \"{rp['label']}\"" if rp.get("label") else "라벨 없음"
    head = ["# 강건성 재실행 A1_repaired — 수리된 `daily_bars_adjusted`(PR #12) 로 한 번, 보고만 (PO-5.2)", "",
            f"- 배포일 {rp.get('deployed_on')}, swing-it `{rp.get('swing_it_sha')}`, core `{r.get('core_ref')}`.",
            f"- 지문(2019-01-01~{FULL_END}): 수리 전(A1.json) {rp.get('fingerprint_pre_repair')} → 지금 {r.get('fingerprint_before', {}).get('daily_bars_adjusted')}.",
            f"- 고정: σ_target {rp.get('sigma_target_fixed')}, 밴드 `PO3.json`, α 키 `OB-new`. 수리된 표로 다시 낸 σ_target(보고만): "
            f"{rp.get('sigma_target_rebuilt')}.",
            f"- **판정: 원래 {rp.get('verdict_original')} → 재실행 {rp.get('verdict_repaired')} — {lab}.** {rp.get('next')}",
            "- 아래 본문은 `po4.render_a1` 그대로다(A1 독립 리뷰 절은 원래 A1 에 대한 것이라 뺐다).", "", "---", ""]
    body = P4.render_a1(r, review=False)
    return "\n".join(head) + body


def render_c0_repaired(r: dict) -> str:
    cmp = r["compare_PO2"]
    L = ["# 강건성 재실행 C0_repaired — PO-2 러너 C0 부분, 수리된 `daily_bars_adjusted`(PR #12), 보고만 (PO-2.1-6·PO-5.2)", "",
         f"배포일 {r.get('deployed_on')}, swing-it `{r.get('swing_it_sha')}`, core `{r.get('core_ref')}`. END {OLD_END}, v3, PO-2 이음매 가드(같은 코드).",
         f"지문(2019-01-01~{FULL_END}) {r['fingerprint_full']}, (2019-01-01~{OLD_END}) {r['fingerprint_old']}.", "",
         "| 묶음 | 항목 | PO2.json | 수리 뒤 | 차 |", "|---|---|---|---|---|"]

    def fmt(v):
        return "—" if v is None else (f"{v:.5f}" if isinstance(v, float) else str(v))

    for blk, d in cmp.items():
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            L.append(f"| {blk} | {k} | {fmt(v['before'])} | {fmt(v['after'])} | {fmt(v['diff'])} |")
    L += ["", f"실현 바이어스 최대 절대 차 {fmt(cmp.get('max_abs_diff_realized'))}. 판정이 아니다(C0 는 보정 단계).", ""]
    return "\n".join(L) + "\n"


def _rnd(o):
    import pead_optimizer_po4 as P4

    return P4.rnd(o)


def main() -> int:  # pragma: no cover - DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("c0", "a1", "c0-repro-check"), required=True)
    ap.add_argument("--deployed-on", default="", help="PR #12 배포 날짜(c0·a1 필수)")
    ap.add_argument("--cache", required=True, help="레포 밖, 새 디렉터리")
    ap.add_argument("--po3-cache", default="", help="a1: PO-3 캐시(구창 v4 비트 대조 — 보고만)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="", help="c0-repro-check 출력 디렉터리(레포 밖)")
    args = ap.parse_args()
    import pead_optimizer_po4 as P4

    repo = Path(__file__).resolve().parents[2]
    cache = Path(args.cache)
    assert repo not in cache.resolve().parents, "캐시는 레포 밖에 둔다"
    assert not cache.exists() or not any(cache.iterdir()), "캐시는 새 디렉터리여야 한다(처음부터 다시 만든다)"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    a1 = json.loads((OUT / "A1.json").read_text())
    fp_pre = a1["fingerprint_before"]["daily_bars_adjusted"]
    fp_full0 = P4.fingerprint("daily_bars_adjusted", FULL_END)
    if args.mode == "c0-repro-check":
        repair_guard(fp_full0, fp_pre, expect_repaired=False)
        assert args.out and repo not in Path(args.out).resolve().parents, "repro-check 는 레포 밖에만 쓴다"
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        assert args.deployed_on, "--deployed-on 필요(PO-5.2-1)"
        repair_guard(fp_full0, fp_pre)
        outdir = OUT
    stem = "C0_repaired" if args.mode.startswith("c0") else "A1_repaired"
    out_guard(stem, outdir)

    if args.mode.startswith("c0"):
        fp_old0 = P4.fingerprint("daily_bars_adjusted", OLD_END)
        new = c0_part(cache, args.workers, log)
        fp_full1 = P4.fingerprint("daily_bars_adjusted", FULL_END)
        fp_old1 = P4.fingerprint("daily_bars_adjusted", OLD_END)
        if fp_full1 != fp_full0 or fp_old1 != fp_old0:
            raise SystemExit("실행 중 표 지문이 바뀌었다 — 결과를 쓰지 않는다(PO-4.2)")
        po2 = json.loads((OUT / "PO2.json").read_text())
        res = {"mode": args.mode, "deployed_on": args.deployed_on or None, "swing_it_sha": git_sha(), "core_ref": "13c56a7",
               "fingerprint_full": fp_full0, "fingerprint_old": fp_old0, "fingerprint_pre_repair_A1": fp_pre, **new}
        res = json.loads(json.dumps(_rnd(res), ensure_ascii=False))
        res["compare_PO2"] = compare_po2(res, po2)
        if args.mode == "c0-repro-check":
            res["repro_exact_PO2"] = repro_exact(res["compare_PO2"])
            log(f"PO2.json 재현(5자리) {res['repro_exact_PO2']}")
        res["elapsed_s"] = time.time() - t0
        (outdir / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        (outdir / f"{stem}.md").write_text(render_c0_repaired(res))
        return 0 if (args.mode == "c0" or res["repro_exact_PO2"]) else 1

    # a1 — po4.run_new 그대로
    assert args.po3_cache, "--po3-cache 필요"
    holder: dict = {}
    P4.recheck_old = recheck_old_report_only(holder)
    rep = json.loads((OUT / "PO4_repro.json").read_text())
    ns = argparse.Namespace(mode="new", end=FULL_END, asof_from="2024-01-01", asof_to="2026-07-31", n_asof=31,
                            cache=str(cache), po3_cache=args.po3_cache, workers=args.workers, stem=stem,
                            prereg_sha=a1["prereg"].get("PO-4", ""), repro_sha=a1["prereg"].get("PO-4_repro", ""))
    P4._T0[0] = t0
    code = P4.run_new(ns, cache, log)
    r = json.loads((OUT / f"{stem}.json").read_text())
    if "STOP" not in r:
        br = repaired_branch(r["verdict"], a1["verdict"])
        r["repaired"] = {**br, "deployed_on": args.deployed_on, "swing_it_sha": git_sha(), "fingerprint_pre_repair": fp_pre,
                         "fingerprint_differs_from_pre": r["fingerprint_before"]["daily_bars_adjusted"] != fp_pre,
                         "sigma_target_fixed": float(rep["sigma_target_repr"]),
                         "sigma_target_rebuilt": holder.get("recheck", {}).get("sigma_target_rebuilt"),
                         "OB21_original": a1["OB21"], "OB5_original": a1["OB5"]}
        r = json.loads(json.dumps(_rnd(r), ensure_ascii=False))
        (OUT / f"{stem}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        log(f"강건성 재실행: {br}")
    (OUT / f"{stem}.md").write_text(render_a1_repaired(r) if "STOP" not in r else P4.render_a1(r))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
