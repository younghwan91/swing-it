"""단계 모듈과 계약 — 게이트를 건너뛸 수 없고, 계약을 어기면 만들어지지 않는다.

core 레이어(위험모델·최적화·충격비용·기여)는 기본 핀 0.7.0 에 없다. 그 경로의 실제 위임 시험은 core main 으로 돌릴 때만 뜬다:
``PYTHONPATH=<core>/src uv run --with cvxpy --extra dev pytest tests/test_pipeline_stages.py``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swing_it.pipeline import Stage, StageGateError, gate, record_stage
from swing_it.pipeline.contracts import (
    STAGE_CONTRACTS,
    Attribution,
    BacktestResult,
    ContractError,
    CoreLayerMissing,
    CostEstimate,
    DataPanels,
    Forecast,
    ICReport,
    OrderPlan,
    RiskModelOutput,
    Signal,
    TargetWeights,
    UniverseMask,
    core_layer,
)
from swing_it.storage import connect

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "swing_it" / "pipeline"
STUDY = "toy"

MODULES = {
    Stage.UNIVERSE: "s01_universe",
    Stage.DATA: "s02_data",
    Stage.ALPHA: "s03_alpha",
    Stage.ALPHA_VALIDATION: "s04_alpha_validation",
    Stage.FORECAST: "s05_forecast",
    Stage.RISK_MODEL: "s06_risk_model",
    Stage.PORTFOLIO: "s07_portfolio",
    Stage.COSTS: "s08_costs",
    Stage.BACKTEST: "s09_backtest",
    Stage.EXECUTION: "s10_execution",
    Stage.ATTRIBUTION: "s11_attribution",
}


def _module(stage: Stage):
    return importlib.import_module(f"swing_it.pipeline.{MODULES[stage]}")


# --- 구조 --------------------------------------------------------------------------

def test_the_eleven_stages_are_the_users_process_in_order():
    assert [s.ko for s in Stage] == [
        "유니버스", "데이터", "알파 발굴", "알파 검증(IC/ICIR)", "ML 수익률 예측", "팩터 위험모델",
        "포트폴리오 최적화", "거래비용 모델", "백테스트", "주문 실행", "성과 기여·모니터링",
    ]
    assert [int(s) for s in Stage] == list(range(1, 12))


def test_every_stage_has_a_module_and_a_contract():
    assert set(STAGE_CONTRACTS) == set(Stage) == set(MODULES)
    for stage in Stage:
        m = _module(stage)
        assert m.STAGE is stage
        assert m.CONTRACT is STAGE_CONTRACTS[stage]


@pytest.mark.parametrize("stage", list(Stage), ids=lambda s: s.name)
def test_run_declares_the_stage_output_type(stage):
    """모듈의 반환 표기가 계약표와 같은가 — 표가 코드와 따로 놀지 않게."""
    tree = ast.parse((PKG / f"{MODULES[stage]}.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert ast.unparse(fn.returns) == STAGE_CONTRACTS[stage].gives.__name__


@pytest.mark.parametrize("stage", list(Stage), ids=lambda s: s.name)
def test_run_opens_with_the_gate(stage):
    """모든 단계 모듈의 ``run`` 첫 문장이 ``require_stage(study, STAGE)`` 다 — 러너 밖에서 불러도 순서를 건너뛰지 못한다."""
    tree = ast.parse((PKG / f"{MODULES[stage]}.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    first = ast.unparse(body[0])
    assert first == "require_stage(study, STAGE)", f"{MODULES[stage]}.run 첫 문장: {first}"
    assert list(inspect.signature(_module(stage).run).parameters)[0] == "study"


@pytest.mark.parametrize("stage", [s for s in Stage if s is not Stage.UNIVERSE], ids=lambda s: s.name)
def test_run_is_blocked_before_it_touches_anything(stage, tmp_path, monkeypatch):
    """앞 단계 증거가 없으면 입력을 건드리기 전에 막힌다 — 넘긴 값이 쓰이면 ``_Bomb`` 이 터진다."""

    class _Bomb:
        def __getattr__(self, name):
            raise AssertionError(f"게이트 전에 입력을 건드렸다: {name}")

    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    run = _module(stage).run
    args = [p for p in list(inspect.signature(run).parameters.values())[1:]]
    pos = [_Bomb() for p in args if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is p.empty]
    kw = {p.name: _Bomb() for p in args if p.kind is p.KEYWORD_ONLY and p.default is p.empty}
    with pytest.raises(StageGateError):
        run(STUDY, *pos, **kw)


def _resolve_reference(ref: str) -> None:
    path, _, symbol = ref.partition("::")
    assert (REPO / path).is_file(), f"참조 경로가 없다: {path}"
    if symbol:
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert symbol in names, f"{path} 에 {symbol} 가 없다"


@pytest.mark.parametrize("stage", list(Stage), ids=lambda s: s.name)
def test_references_to_existing_code_are_real(stage):
    """연구 하네스는 경로 문자열로만 가리킨다(src → research import 금지). 그 경로가 썩지 않게 한다."""
    for ref in _module(stage).REFERENCES:
        _resolve_reference(ref)


@pytest.mark.parametrize("ref", ["research/experiments/nope.py", "src/swing_it/storage.py::nope"])
def test_reference_check_catches_a_rotten_pointer(ref):
    with pytest.raises(AssertionError):
        _resolve_reference(ref)


def test_pipeline_package_has_no_order_path():
    """주문·키움 호출·로그인을 하지 않는다 — 코드에 그 경로 자체가 없다(독스트링 설명은 제외하고 import·호출을 본다)."""
    banned = ("kiwoom", "krx_quant_core.execution", "httpx", "requests")
    for p in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
                mods = [ast.unparse(node)]
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "core_layer":
                mods = [ast.unparse(node)]
            assert not [m for m in mods if any(b in m for b in banned)], f"{p.name}: {mods}"


def test_src_stays_free_of_research_imports():
    """경계(린트 (a))를 importlib 로 돌아가지 않는다."""
    for p in PKG.glob("*.py"):
        assert "research." not in "".join(
            ast.unparse(n) for n in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
            if isinstance(n, (ast.Import, ast.ImportFrom)))


def test_price_adjust_does_not_depend_on_the_pipeline():
    """``price_adjust.py`` 는 Airflow 라이브 마운트로 돈다. 나머지 import 범위는 ``tests/test_price_adjust.py`` 가 본다."""
    tree = ast.parse((REPO / "src/swing_it/price_adjust.py").read_text(encoding="utf-8"))
    mods = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not [m for m in mods if "pipeline" in m]


# --- 구현 없는 단계 -----------------------------------------------------------------

def _open_upto(root: Path, stage: Stage) -> None:
    (root / "e.md").write_text("증거\n", encoding="utf-8")
    (root / "e_verified.md").write_text("quant-reviewer 리뷰\n", encoding="utf-8")
    for s in Stage:
        if s >= stage:
            break
        kw = {"reviewer": "e_verified.md"} if s in gate.RESULT_STAGES else {}
        record_stage(STUDY, s, "pass", evidence=["e.md"], root=root, **kw)


@pytest.mark.parametrize("stage,hint", [(Stage.FORECAST, "ml-forecast-prereg"), (Stage.EXECUTION, "docs/execution/DESIGN.md")])
def test_unimplemented_stage_points_to_where_it_lives(stage, hint, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    _open_upto(tmp_path, stage)
    with pytest.raises(NotImplementedError, match=hint):
        _module(stage).run(STUDY, *[None] * (len(inspect.signature(_module(stage).run).parameters) - 1))


# --- core 레이어 지연 import --------------------------------------------------------

@pytest.mark.parametrize("module", ["krx_quant_core.risk.factor_model", "krx_quant_core.portfolio",
                                    "krx_quant_core.costs.impact", "krx_quant_core.risk.attribution"])
def test_missing_core_layer_says_what_to_do(module, monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _raise_missing(module))   # 0.7.0 이든 main 이든 "없는 상태"
    with pytest.raises(CoreLayerMissing, match=r"0\.7\.0.*(미배포)"):
        core_layer(module)


def _raise_missing(module: str):
    def _f(name, *a, **k):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return _f


def test_core_layer_does_not_hide_a_different_missing_dependency(monkeypatch):
    """cvxpy 같은 다른 의존성 문제를 "core 가 없다"로 둔갑시키지 않는다."""
    def _f(name, *a, **k):
        raise ModuleNotFoundError("No module named 'cvxpy'", name="cvxpy")
    monkeypatch.setattr(importlib, "import_module", _f)
    with pytest.raises(ModuleNotFoundError) as e:
        core_layer("krx_quant_core.portfolio")
    assert not isinstance(e.value, CoreLayerMissing)


@pytest.mark.parametrize("stage,module", [(Stage.RISK_MODEL, "krx_quant_core.risk.factor_model"),
                                          (Stage.PORTFOLIO, "krx_quant_core.portfolio"),
                                          (Stage.COSTS, "krx_quant_core.costs.impact"),
                                          (Stage.ATTRIBUTION, "krx_quant_core.risk.attribution")])
def test_core_layers_are_imported_lazily(stage, module):
    """기본 핀 0.7.0 에서 ``swing_it.pipeline.sNN`` import 는 되어야 한다 — 모듈 머리에서 끌어오면 안 된다."""
    tree = ast.parse((PKG / f"{MODULES[stage]}.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert module not in {getattr(n, "module", None) for n in top}
    assert any(module in ast.unparse(c) for c in ast.walk(tree) if isinstance(c, ast.Call))


# --- 계약: 위반을 주입한다 -----------------------------------------------------------

def _universe() -> UniverseMask:
    frame = pd.DataFrame({"code": ["A", "B"], "date": ["2020-01-02", "2020-01-02"], "eligible": [True, False]})
    return UniverseMask(frame=frame, definition="d", survivorship="s")


def _prices(end="2020-01-03"):
    return pd.DataFrame({"code": ["A", "A"], "date": ["2020-01-02", end], "close": [1.0, 2.0]})


def _earnings(**kw):
    base = {"code": ["A"], "period": ["2019Q3"], "avail_date": ["2019-11-14"], "knowledge_date": ["2019-11-14"]}
    return pd.DataFrame({**base, **kw})


def _panels(**kw):
    args = {"prices": _prices(), "earnings": _earnings(), "universe": _universe(), "end": "2020-01-03",
            "fingerprint": "[1]"}
    return DataPanels(**{**args, **kw})


def test_valid_contracts_construct():
    assert _panels().trading_dates() == ["2020-01-02", "2020-01-03"]


@pytest.mark.parametrize("build,match", [
    (lambda: UniverseMask(_universe().frame.assign(eligible=[1, 0]), "d", "s"), "bool"),
    (lambda: UniverseMask(_universe().frame.iloc[:0], "d", "s"), "비었다"),
    (lambda: UniverseMask(_universe().frame.assign(eligible=[False, False]), "d", "s"), "적격"),
    (lambda: UniverseMask(_universe().frame.assign(code=["A", "A"]), "d", "s"), "중복"),
    (lambda: UniverseMask(_universe().frame, "d", "  "), "survivorship"),
    (lambda: _panels(prices=_prices("2020-02-01")), "미래"),
    (lambda: _panels(earnings=_earnings(knowledge_date=["2020-06-01"])), "몰랐던"),
    (lambda: _panels(earnings=_earnings(avail_date=["2020-06-01"])), "몰랐던"),
    (lambda: _panels(earnings=_earnings(knowledge_date=[None])), "결측"),
    (lambda: _panels(earnings=_earnings().drop(columns="knowledge_date")), "열이 없다"),
    (lambda: _panels(end="2026-09-14", prices=_prices("2026-09-14")), "애프터마켓"),
    (lambda: _panels(universe=_universe().frame), "UniverseMask"),
    (lambda: Signal(pd.DataFrame({"code": ["A"], "date": ["d"], "signal": [np.inf]}), "n", "k"), "inf"),
    (lambda: Signal(pd.DataFrame({"code": ["A"], "date": ["d"], "signal": [np.nan]}), "n", "k"), "유효한"),
    (lambda: ICReport("n", 40, 0.1, 2.0, 0, 0.5), "잰 날"),
    (lambda: ICReport("n", 0, 0.1, 2.0, 10, 0.5), "horizon"),
    (lambda: Forecast(pd.DataFrame({"code": ["A"], "date": ["2020-01-02"], "forecast": [0.1],
                                    "train_end": ["2020-01-02"]}), "m", 20), "누출"),
    (lambda: Forecast(pd.DataFrame({"code": ["A"], "date": ["2020-01-03"], "forecast": [np.nan],
                                    "train_end": ["2020-01-02"]}), "m", 20), "NaN"),
    (lambda: RiskModelOutput("2020-01-02", object(), "v4"), "없다"),
    (lambda: TargetWeights("d", np.array(["A"]), np.array([-0.1]), "m"), "음수"),
    (lambda: TargetWeights("d", np.array(["A", "B"]), np.array([0.8, 0.8]), "m"), "합"),
    (lambda: TargetWeights("d", np.array(["A", "A"]), np.array([0.1, 0.1]), "m"), "중복"),
    (lambda: TargetWeights("d", np.array(["A"]), np.array([0.1, 0.1]), "m"), "길이"),
    (lambda: CostEstimate(date(2020, 1, 2), np.array(["A"]), np.array([-1.0]), 0.0), "음수"),
    (lambda: BacktestResult("l", pd.DataFrame(), {}, "gross", 1), "비었다"),
    (lambda: BacktestResult("l", pd.DataFrame({"x": [1]}), {}, "넷", 1), "cost_basis"),
    (lambda: BacktestResult("l", pd.DataFrame({"x": [1]}), {}, "gross", 0), "시행"),
    (lambda: OrderPlan("d", ("A",), ("BUY",), (1,), (100.0,), dry_run=False), "dry_run"),
    (lambda: OrderPlan("d", ("A",), ("HOLD",), (1,), (100.0,)), "side"),
    (lambda: OrderPlan("d", ("A",), ("BUY",), (0,), (100.0,)), "수량"),
    (lambda: OrderPlan("d", ("A",), ("BUY",), (1,), (0.0,)), "지정가"),
    (lambda: Attribution(pd.DataFrame({"date": ["d"], "f": [0.1], "specific": [0.1], "total": [0.3]}), ("f",)), "기여 합"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_contract_violation_is_refused(build, match):
    with pytest.raises(ContractError, match=match):
        build()


def test_aftermarket_must_be_acknowledged_not_ignored():
    p = _panels(end="2026-09-14", prices=_prices("2026-09-14"), aftermarket_handled=True)
    assert p.aftermarket_handled


def test_no_contract_carries_a_verdict_field():
    """리포터-not-판정기(GUARDRAILS §8) — 합격 필드가 끼어들지 않는다."""
    for c in STAGE_CONTRACTS.values():
        names = {f.lower() for f in c.gives.__dataclass_fields__}
        assert not names & {"passed", "verdict", "ok", "accepted", "go"}, c.gives


def test_stage_expects_contract_objects_not_raw_frames(tmp_path, monkeypatch):
    """앞 단계 산출물 대신 날것을 넘기면 안 받는다."""
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    _open_upto(tmp_path, Stage.ALPHA)
    with pytest.raises(ContractError, match="DataPanels"):
        _module(Stage.ALPHA).run(STUDY, pd.DataFrame())


# --- 실제 위임: 합성 데이터로 1→2→3→4→9 를 끝까지 흘린다 --------------------------------

N, T = 40, 240


def _seed_db(path: Path, end_days: pd.DatetimeIndex) -> None:
    rng = np.random.default_rng(7)
    con = connect(path)
    rows = []
    for i in range(N):
        close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, T)))
        code = f"{i:06d}"
        rows += [(code, d.strftime("%Y-%m-%d"), c, c, c, c, 10, 500000.0, "kiwoom") for d, c in zip(end_days, close)]
    con.executemany("INSERT INTO daily_bars_adjusted(code,date,open,high,low,close,volume,trade_value,source) "
                    "VALUES(?,?,?,?,?,?,?,?,?)", rows)
    eas = []
    for i in range(N):
        code = f"{i:06d}"
        eas.append((code, "2019Q3", "2019-11-14", 100.0 + i, 80.0, "2019-11-14"))
        # 정정: 2020-03-01 에 안 값 — 그 전 날짜에 새어 들어오면 안 된다
        eas.append((code, "2019Q3", "2019-11-14", 999.0, 80.0, "2020-03-01"))
    con.executemany("INSERT INTO earnings(code,period,avail_date,netinc,netinc_prior,knowledge_date) "
                    "VALUES(?,?,?,?,?,?)", eas)
    con.commit()
    con.close()


@pytest.fixture
def flow(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    days = pd.bdate_range("2020-01-02", periods=T)
    db = tmp_path / "t.db"
    _seed_db(db, days)
    con = connect(db)
    cap = pd.DataFrame([{"code": f"{i:06d}", "date": d.strftime("%Y-%m-%d"), "market_cap": 1e12 - i}
                        for i in range(N) for d in days])
    adv = cap[["code", "date"]].assign(adv=30000.0)
    yield tmp_path, con, cap, adv, days
    con.close()


def _advance(root: Path, stage: Stage, *, status="pass") -> None:
    (root / "e.md").write_text("증거\n", encoding="utf-8")
    (root / "e_verified.md").write_text("quant-reviewer 리뷰\n", encoding="utf-8")
    kw = {"reviewer": "e_verified.md"} if stage in gate.RESULT_STAGES else {}
    record_stage(STUDY, stage, status, evidence=["e.md"], root=root, **kw)


def test_flow_through_implemented_stages_on_the_default_pin(flow):
    from swing_it.pipeline import s01_universe, s02_data, s03_alpha, s04_alpha_validation, s09_backtest

    root, con, cap, adv, days = flow
    universe = s01_universe.run(STUDY, cap, adv, cap_rank=(0, 30), adv_floor=20000.0,
                                definition="시총 1~30위 ∧ ADV≥200억", survivorship="합성 — 폐지 없음")
    assert universe.frame["eligible"].any()
    _advance(root, Stage.UNIVERSE)

    panels = s02_data.run(STUDY, con, universe, end=days[-1].strftime("%Y-%m-%d"))
    assert panels.prices["date"].max() == days[-1].strftime("%Y-%m-%d")
    assert len(panels.earnings) == 2 * N, "실적은 전 버전을 접지 않고 받는다"
    assert json.loads(panels.fingerprint)[:2] == [N * T, days[-1].strftime("%Y-%m-%d")]
    _advance(root, Stage.DATA)

    signal = s03_alpha.run(STUDY, panels)
    wide = signal.frame.set_index(["code", "date"])["signal"]
    assert wide[("000000", "2020-01-02")] == pytest.approx((100 - 80) / 80), "정정 전 값"
    assert wide[("000000", days[-1].strftime("%Y-%m-%d"))] == pytest.approx((999 - 80) / 80), "정정 후 값"
    assert np.isnan(wide[(f"{N - 1:06d}", "2020-01-02")]), "유니버스 밖은 신호 없음"
    _advance(root, Stage.ALPHA)

    ic = s04_alpha_validation.run(STUDY, signal, panels, horizon=5, adv_floor=0.0, start_index=5)
    assert ic.n_days > 0 and ic.signal_name == "pead_yoy"
    _advance(root, Stage.ALPHA_VALIDATION)

    for s in (Stage.FORECAST, Stage.RISK_MODEL, Stage.PORTFOLIO):
        record_stage(STUDY, s, "skip", reason="순위 북", root=root)
    _advance(root, Stage.COSTS)

    config = {"horizon": 20, "step": 10, "top_n": 5, "adv_floor": 0.0, "start_index": 5, "min_names": 5}
    logs = root / "logs"
    bt = s09_backtest.run(STUDY, signal, panels, config=config, logs_dir=logs)
    assert bt.cost_basis == "gross" and bt.n_trials == 1 and not bt.periods.empty
    assert (logs / STUDY / "TRIALS.jsonl").exists(), "백테스트는 다중검정 원장을 우회하지 못한다"


def test_universe_stage_runs_with_no_ledger(flow):
    from swing_it.pipeline import s01_universe

    _, _, cap, adv, _ = flow
    assert s01_universe.run(STUDY, cap, adv, cap_rank=(0, 30), adv_floor=20000.0, definition="d", survivorship="s")


def test_data_stage_cuts_at_end_and_keeps_the_fingerprint_honest(flow):
    from swing_it.pipeline import s01_universe, s02_data

    root, con, cap, adv, days = flow
    universe = s01_universe.run(STUDY, cap, adv, cap_rank=(0, 30), adv_floor=20000.0, definition="d", survivorship="s")
    _advance(root, Stage.UNIVERSE)
    end = days[100].strftime("%Y-%m-%d")
    panels = s02_data.run(STUDY, con, universe, end=end)
    assert panels.prices["date"].max() == end
    assert panels.earnings["knowledge_date"].max() <= end


# --- core main 이 있을 때만: 실제 위임 --------------------------------------------------

def test_risk_portfolio_cost_attribution_delegate_to_core(tmp_path, monkeypatch):
    fm = pytest.importorskip("krx_quant_core.risk.factor_model")
    pytest.importorskip("krx_quant_core.portfolio")
    pytest.importorskip("krx_quant_core.costs.impact")
    pytest.importorskip("krx_quant_core.risk.attribution")
    pytest.importorskip("cvxpy")
    from swing_it.pipeline import s06_risk_model, s07_portfolio, s08_costs, s11_attribution

    monkeypatch.setattr(gate, "_repo_root", lambda: tmp_path)
    _open_upto(tmp_path, Stage.RISK_MODEL)
    rng = np.random.default_rng(3)
    n, t = 40, 200
    R = rng.normal(0, 0.02, (n, t))
    cap = np.full((n, t), 1e12) * (1 + rng.random((n, 1)))
    mask = np.ones((n, t), dtype=bool)
    labels = np.array([f"S{i % 4}" for i in range(n)])
    risk = s06_risk_model.run(STUDY, R, None, labels, cap, mask, t - 1, asof="2020-12-31", label="toy",
                              codes=np.array([f"{i:06d}" for i in range(n)]))
    assert isinstance(risk.model, fm.RiskModel)

    record_stage(STUDY, Stage.RISK_MODEL, "pass", evidence=["e.md"], root=tmp_path)
    w0 = np.zeros(len(risk.model.codes))
    target = s07_portfolio.run(STUDY, rng.normal(0, 0.01, len(w0)), risk, w0, risk_aversion=10.0)
    assert target.status == "optimal" and target.weights.sum() == pytest.approx(1.0, abs=1e-6)

    record_stage(STUDY, Stage.PORTFOLIO, "pass", evidence=["e.md"], root=tmp_path)
    cost = s08_costs.run(STUDY, target, w0, aum=3e8, adv_value=np.full(len(w0), 2e10), daily_vol=np.full(len(w0), 0.02),
                         market="KOSPI", trade_date=date(2023, 6, 1), half_spread_bp=5.0)
    double = s08_costs.run(STUDY, target, w0, aum=3e8, adv_value=np.full(len(w0), 2e10), daily_vol=np.full(len(w0), 0.02),
                           market="KOSPI", trade_date=date(2023, 6, 1), half_spread_bp=5.0, multiplier=2.0)
    assert cost.nav_cost > 0 and double.nav_cost == pytest.approx(2 * cost.nav_cost)

    record_stage(STUDY, Stage.COSTS, "pass", evidence=["e.md"], root=tmp_path)
    (tmp_path / "e_verified.md").write_text("quant-reviewer 리뷰\n", encoding="utf-8")
    record_stage(STUDY, Stage.BACKTEST, "pass", evidence=["e.md"], reviewer="e_verified.md", root=tmp_path)
    (tmp_path / "fwd.md").write_text("전진\n", encoding="utf-8")
    record_stage(STUDY, Stage.EXECUTION, "skip", reason="페이퍼", root=tmp_path)
    T_, N_, K_ = 3, 2, 2
    frame = s11_attribution.run(STUDY, rng.random((N_, T_)), rng.random((T_, N_, K_)), rng.random((T_, K_)),
                                rng.random((N_, T_)), names=["a", "b"], dates=["d1", "d2", "d3"])
    assert list(frame.frame.columns) == ["date", "a", "b", "specific", "total"]
