"""옮긴 구현이 옮기기 전 숫자를 그대로 내는가 — 합성 데이터의 ``golden/pead_b_golden.json`` 과 대조한다.

그 파일은 ``src/swing_it/book/`` 으로 옮기기 **전** 코드가 만들었다(``golden/make_pead_b_golden.py``). 실데이터 ``B.json`` 대조는
``tests/real_db/test_b_json_reproduction.py`` 다. 허용오차는 전부 1e-9 다.

**O 는 솔버를 재생한다.** 솔버 값은 판과 기계에 따라 움직여(CI 에서 λ 1.6e-4, 널 IR 5.5e-3) 폭으로 재면 넓어질 뿐이고 무엇을 재는지도 흐려진다. 그래서 옮긴 직후 ``03fee83`` 트리에서
녹화한 응답(``golden/pead_o_solver_replay.npz``, ``golden/make_pead_o_replay.py``)을 돌려주고 솔버 **밖**(λ 이분, 먼지, 회계, 요약, 널, 부트)만 잰다. 입력 요약은 호출마다 맞댄다(:mod:`solver_replay`).
실제 솔버는 값을 핀하지 않고 한 번만 돌려 불변식과 상태를 본다. 재생 시험이 진짜 잡는지는 ``tests/pipeline_mutations_core.toml`` 이 CI ``research`` 잡에서 증명한다.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book_fixtures import synthetic_ctx  # noqa: E402
from solver_replay import Replay  # noqa: E402

from swing_it.book.accounting import check_invariants, r_sigma_series, simulate  # noqa: E402
from swing_it.book.metrics import book_summary, oos, step_returns  # noqa: E402
from swing_it.book.nulls import permutation_nulls, stationary_boot_dir  # noqa: E402
from swing_it.book.parity import full_parity, parity_book, parity_check  # noqa: E402
from swing_it.book.spec import PRIMARY, Flags  # noqa: E402
from swing_it.diagnostics.hac import nw_t_rows, ols_hac  # noqa: E402
from swing_it.validation.ic import daily_rank_ic, fold_ic  # noqa: E402
from swing_it.validation.keys import keyed_rng  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden" / "pead_b_golden.json").read_text(encoding="utf-8"))["seeds"]
REPLAY = Path(__file__).parent / "golden" / "pead_o_solver_replay.npz"
SEEDS = sorted(GOLDEN)
EVAL = ("2020-01-01", "2023-12-31")
FOLD_YEARS = (2020, 2021, 2022, 2023)
IC = {y: 0.05 for y in range(2017, 2026)}
STEP_FIELDS = ("k", "t", "date", "gross", "cost1", "bench", "turnover", "exante_20", "factor_share", "status", "lam", "lam_hit",
               "est_cost", "n_names", "n_forced", "n_delist", "n_frozen", "n_untradeable", "u_excl", "perm_nan_share", "f", "f_next",
               "lam_cause", "n_tranches", "cash", "min_w", "max_w", "max_w_frozen", "sector_dev", "sector_tol", "turn_bound", "n_dust",
               "dust_mass", "delist_mass", "n_buy_limit_up")


def same(got, want, tol, path=""):
    """깊은 대조. 숫자는 ``tol``, NaN·None 은 같게 본다(JSON 왕복에서 NaN 이 None 으로 적혔다)."""
    if isinstance(want, dict):
        assert set(got) == set(want), path
        for k in want:
            same(got[k], want[k], tol, f"{path}/{k}")
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            same(g, w, tol, f"{path}[{i}]")
    elif isinstance(want, bool) or isinstance(want, str):
        assert got == want, path
    elif want is None or (isinstance(want, float) and math.isnan(want)):
        assert got is None or (isinstance(got, (float, np.floating)) and math.isnan(got)), f"{path}: {got}"
    else:
        assert got is not None and abs(float(got) - float(want)) <= tol, f"{path}: {got} != {want}"


def rows(recs, *, contrib):
    out = []
    for r in recs:
        row = {k: getattr(r, k) for k in STEP_FIELDS}
        row["alarm_top3"] = [[str(n), float(e)] for n, e in r.alarm_top3]
        if contrib:
            row["contrib"] = [list(c) for c in r.contrib]
        out.append(row)
    return out


def _needs_costs():
    pytest.importorskip("krx_quant_core.costs.impact")


def _needs_optimizer():
    _needs_costs()
    pytest.importorskip("krx_quant_core.portfolio")


@pytest.fixture
def replay_solver(monkeypatch):
    """``replay_solver(스트림)`` — core ``optimize`` 를 그 녹화로 바꾼다. 시험 끝에 녹화를 다 썼는지 본다."""
    _needs_optimizer()
    opt = importlib.import_module("krx_quant_core.portfolio.optimize")   # 패키지가 같은 이름 함수를 내보내 ``import … as`` 는 함수를 준다
    used = []

    def use(key):
        with np.load(REPLAY) as data:
            r = Replay(data, key, opt.OptResult)
        monkeypatch.setattr(opt, "optimize", r)
        used.append(r)
        return r

    yield use
    for r in used:
        r.assert_used_up()


# --- 기본 핀(0.7.0)에서 도는 것 ---------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_rank_book_without_costs_matches_golden(seed):
    recs = simulate(synthetic_ctx(int(seed)), "R", flags=Flags(costs=False))
    same(rows(recs, contrib=seed == "0"), GOLDEN[seed]["r_no_costs"], 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_parity_books_match_golden(seed):
    ctx = synthetic_ctx(int(seed))
    for lag, key in ((0, "parity_lag0"), (1, "parity_lag1")):
        got = json.loads(parity_book(ctx, lag).to_json(orient="records"))
        same(got, GOLDEN[seed][key], 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_core_parity_matches_golden(seed):
    ctx = synthetic_ctx(int(seed))
    want = GOLDEN[seed]["parity_check"]
    got = parity_check(ctx)
    assert got["pass"] is want["pass"] is True
    assert got["steps_harness"] == want["steps_harness"]
    assert full_parity(ctx)["pass"] is True


@pytest.mark.parametrize("seed", SEEDS)
def test_daily_ic_matches_golden(seed):
    ctx = synthetic_ctx(int(seed))
    days, ics = daily_rank_ic(ctx.C, ctx.adv, ctx.cap, ctx.sig, 40, ctx.C.shape[1] - 1, 20)
    assert [int(d) for d in days] == GOLDEN[seed]["daily_ic_days"]
    same([float(v) for v in ics], GOLDEN[seed]["daily_ic"], 1e-9)
    same(fold_ic(days, ics, int(days[len(days) // 2]), 20), GOLDEN[seed]["fold_ic"], 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_small_helpers_match_golden(seed):
    rng = np.random.default_rng(int(seed))
    y, x = rng.normal(0, 1, 60), rng.normal(0, 1, (60, 3))
    same(ols_hac(y, x), GOLDEN[seed]["ols_hac"], 1e-9)
    same([float(v) for v in nw_t_rows(rng.normal(0, 1, (4, 30)))], GOLDEN[seed]["nw_t_rows"], 1e-9)
    same([float(v) for v in keyed_rng("golden", int(seed)).random(3)], GOLDEN[seed]["keyed_rng"], 0.0)


# --- core 충격비용·최적화기가 있어야 도는 것(research CI 잡 — 건너뛰면 실패) ---------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_rank_book_with_costs_matches_golden(seed):
    _needs_costs()
    ctx = synthetic_ctx(int(seed))
    recs = simulate(ctx, "R")
    same(rows(recs, contrib=seed == "0"), GOLDEN[seed]["r"], 1e-9)
    same(json.loads(json.dumps(book_summary(recs, EVAL, FOLD_YEARS), default=float)), GOLDEN[seed]["summary_r"], 1e-9)
    assert check_invariants(recs, "R") == GOLDEN[seed]["invariants_r"]
    same({str(k): v for k, v in r_sigma_series(ctx).items()}, GOLDEN[seed]["sigma"], 1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_optimized_book_matches_golden(seed, replay_solver):
    ctx = synthetic_ctx(int(seed))
    sigma = r_sigma_series(ctx)
    replay_solver(f"o{seed}")
    recs = simulate(ctx, "O", ic_by_year=IC, sigma_r=sigma)
    same(rows(recs, contrib=seed == "0"), GOLDEN[seed]["o"], 1e-9)
    same(json.loads(json.dumps(book_summary(recs, EVAL, FOLD_YEARS), default=float)), GOLDEN[seed]["summary_o"], 1e-9)
    assert check_invariants(recs, "O") == GOLDEN[seed]["invariants_o"]


def test_nulls_and_boot_match_golden(replay_solver):
    seed = SEEDS[0]
    ctx = synthetic_ctx(int(seed))
    sigma = r_sigma_series(ctx)
    replay_solver("nulls0")
    nulls = permutation_nulls(ctx, PRIMARY, IC, sigma, workers=1, n=2, window=EVAL)    # 재생은 호출 순서에 묶인다. 작업자 하나
    want = GOLDEN[seed]["nulls"]
    same([float(v) for v in nulls["R"]], want["R"], 1e-9)
    same([float(v) for v in nulls["O"]], want["O"], 1e-9)
    assert nulls["violations"] == want["violations"]
    replay_solver(f"o{seed}")
    a_o = oos(step_returns(simulate(ctx, "O", ic_by_year=IC, sigma_r=sigma), 2.0), EVAL)["a"].to_numpy()
    a_r = oos(step_returns(simulate(ctx, "R"), 2.0), EVAL)["a"].to_numpy()
    same([float(v) for v in stationary_boot_dir(a_o, a_r, n=50)], GOLDEN[seed]["boot"], 1e-9)


def test_replay_rejects_a_changed_input_or_an_unused_recording():
    """재생이 입력을 안 보면 O 코드가 바뀌어도 녹화된 답을 그대로 받아 초록이 된다."""
    class Con:
        budget = 1.0

    def result(**kw):
        return kw

    data = {"k.summary": np.array([[2.0, 2, 3.0, 5.0, 0.5, 1.0]]), "k.weights": np.array([0.5, 0.5]), "k.ends": np.array([2]),
            "k.status": np.array(["optimal"]), "k.est_cost": np.array([0.01])}
    alpha, w_prev = np.array([1.0, 2.0]), np.array([0.25, 0.25])
    r = Replay(data, "k", result)
    with pytest.raises(AssertionError, match="호출 0번"):
        r.assert_used_up()
    assert r(alpha, None, w_prev, risk_aversion=2.0, constraints=Con)["weights"].tolist() == [0.5, 0.5]
    r.assert_used_up()
    with pytest.raises(AssertionError, match="많다"):
        r(alpha, None, w_prev, risk_aversion=2.0, constraints=Con)
    with pytest.raises(AssertionError, match="lam"):
        Replay(data, "k", result)(alpha, None, w_prev, risk_aversion=2.1, constraints=Con)
    with pytest.raises(AssertionError, match="alpha_sum"):
        Replay(data, "k", result)(alpha + 1e-6, None, w_prev, risk_aversion=2.0, constraints=Con)


# --- 실제 솔버(값은 핀하지 않는다 — 판과 기계에 따라 움직인다) ------------------------------------

def test_real_solver_keeps_the_invariants():
    _needs_optimizer()
    pytest.importorskip("cvxpy")
    ctx = synthetic_ctx(int(SEEDS[0]))
    recs = simulate(ctx, "O", ic_by_year=IC, sigma_r=r_sigma_series(ctx))
    assert recs
    assert check_invariants(recs, "O") == []
    assert {r.status for r in recs} == {"optimal"}
