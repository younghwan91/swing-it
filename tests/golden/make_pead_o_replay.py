"""``pead_o_solver_replay.npz`` 를 만든 스크립트 — 기록용이다. **다시 만들지 않는다.**

옮긴 뒤 코드의 ``03fee83`` 트리에서 실제 솔버를 돌려 응답을 호출 순서대로 녹화한다. 지금 코드로 다시 만들면 그 코드끼리 비교하게 되어 아무것도 증명하지 못한다.
같은 런의 O 출력이 ``pead_b_golden.json`` 의 O 절과 비트까지 같은지도 본다(cvxpy 1.9.3·clarabel 0.11.1 에서 같았다). 다르면 골든 절을 고쳐야 하니 실패한다.

    cd <03fee83 트리>; uv run --python 3.12 --extra dev --extra ml \\
      --with "krx-quant-core[portfolio] @ git+file://<core>@8286e3e" python <이 파일> <03fee83 트리> <골든 JSON> <출력 npz>
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = ("2020-01-01", "2023-12-31")
FOLD_YEARS = (2020, 2021, 2022, 2023)
IC = {y: 0.05 for y in range(2017, 2026)}


def _same_bits(got, want) -> bool:
    return json.dumps(got, default=float, sort_keys=True) == json.dumps(want, default=float, sort_keys=True)


def main() -> int:
    tree, golden_path, out_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    sys.path[:0] = [str(tree / "tests"), str(HERE.parent)]  # book_fixtures 는 그 트리 것, solver_replay 는 이 레포 것

    from book_fixtures import synthetic_ctx
    from make_pead_b_golden import steps
    from solver_replay import read_records, recorder, to_arrays

    import swing_it
    from swing_it.book.accounting import check_invariants, r_sigma_series, simulate
    from swing_it.book.metrics import book_summary, oos, step_returns
    from swing_it.book.nulls import permutation_nulls, stationary_boot_dir
    from swing_it.book.spec import PRIMARY

    assert str(tree) in swing_it.__file__, swing_it.__file__
    opt = importlib.import_module("krx_quant_core.portfolio.optimize")     # 패키지가 같은 이름 함수를 내보내 ``import … as`` 는 함수를 준다
    real = opt.optimize
    golden = json.loads(golden_path.read_text(encoding="utf-8"))["seeds"]
    arrays: dict = {}
    differs: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        def recording(key):
            path = Path(tmp) / key
            opt.optimize = recorder(real, path)
            return path

        for seed in ("0", "3"):
            ctx = synthetic_ctx(int(seed))
            sigma = r_sigma_series(ctx)
            path = recording(f"o{seed}")
            recs = simulate(ctx, "O", ic_by_year=IC, sigma_r=sigma)
            arrays |= to_arrays(f"o{seed}", read_records(path))
            s = golden[seed]
            got = {"o": steps(recs, contrib=seed == "0"), "summary_o": json.loads(json.dumps(book_summary(recs, EVAL, FOLD_YEARS), default=float)),
                   "invariants_o": check_invariants(recs, "O")}
            if seed == "0":     # 널·부트는 첫 시드만 시험한다
                path = recording("nulls0")
                nulls = permutation_nulls(ctx, PRIMARY, IC, sigma, workers=1, n=2, window=EVAL)
                arrays |= to_arrays("nulls0", read_records(path))
                got["nulls_O"], s["nulls_O"] = [float(v) for v in nulls["O"]], s["nulls"]["O"]
                a_o = oos(step_returns(recs, 2.0), EVAL)["a"].to_numpy()
                a_r = oos(step_returns(simulate(ctx, "R"), 2.0), EVAL)["a"].to_numpy()
                got["boot"] = [float(v) for v in stationary_boot_dir(a_o, a_r, n=50)]
            differs += [f"시드 {seed} {k}" for k, v in got.items() if not _same_bits(v, s[k])]

    print({p: importlib.metadata.version(p) for p in ("cvxpy", "clarabel", "numpy", "scipy", "krx-quant-core")},
          {k: int(v.size) for k, v in arrays.items() if k.endswith(".status")})
    if differs:
        print("골든과 다른 절:", differs)
        return 1
    np.savez_compressed(out_path, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
