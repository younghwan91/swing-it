"""경계 검사·단계 표·지연 import — 위반을 주입해 실제로 거부하는지 본다."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swing_it.core_layers import CoreLayerMissing, core_layer
from swing_it.pipeline.checks import (
    CheckError,
    check_attribution,
    check_forecast,
    check_panels,
    check_signal,
    check_universe,
    check_weights,
)
from swing_it.pipeline.stages import STAGE_FUNCTIONS, Stage

REPO = Path(__file__).resolve().parents[1]


def _universe(**kw):
    return pd.DataFrame({"code": ["A", "B"], "date": ["2020-01-02", "2020-01-02"], "eligible": [True, False], **kw})


def _prices(end="2020-01-03"):
    return pd.DataFrame({"code": ["A", "A"], "date": ["2020-01-02", end], "close": [1.0, 2.0]})


def _earnings(**kw):
    return pd.DataFrame({"code": ["A"], "period": ["2019Q3"], "avail_date": ["2019-11-14"], "knowledge_date": ["2019-11-14"], **kw})


def _forecast(date="2020-03-03", train_end="2020-01-02", forecast=0.1):
    return pd.DataFrame({"code": ["A"], "date": [date], "forecast": [forecast], "train_end": [train_end]})


def test_valid_inputs_pass():
    check_universe(_universe())
    check_panels(_prices(), _earnings(), "2020-01-03")
    check_signal(np.array([[1.0, np.nan]]))
    check_forecast(_forecast(), 20)
    check_weights([0.5, 0.5], ["A", "B"])
    check_attribution(pd.DataFrame({"date": ["d"], "f": [0.1], "specific": [0.1], "total": [0.2]}), ("f",))


@pytest.mark.parametrize("call,match", [
    (lambda: check_universe(_universe(eligible=[1, 0])), "bool"),
    (lambda: check_universe(_universe().iloc[:0]), "비었다"),
    (lambda: check_universe(_universe(eligible=[False, False])), "적격"),
    (lambda: check_universe(_universe(code=["A", "A"])), "중복"),
    (lambda: check_universe(_universe(date=["2020-13-45", "2020-01-02"])), "날짜"),
    (lambda: check_universe(_universe(date=[None, "2020-01-02"])), "날짜"),
    (lambda: check_universe(_universe(code=[None, "B"])), "코드"),
    (lambda: check_universe(_universe(code=["", "B"])), "코드"),
    (lambda: check_universe(pd.DataFrame({"code": ["A", "B"], "date": ["2020-01-02", "2020-01-03"], "eligible": [True, False]})), "2020-01-03"),
    (lambda: check_panels(_prices("2020-02-01"), _earnings(), "2020-01-03"), "미래"),
    (lambda: check_panels(_prices(), _earnings(knowledge_date=["2020-06-01"]), "2020-01-03"), "몰랐던"),
    (lambda: check_panels(_prices(), _earnings(avail_date=["2020-06-01"]), "2020-01-03"), "몰랐던"),
    (lambda: check_panels(_prices(), _earnings(knowledge_date=[None]), "2020-01-03"), "결측"),
    (lambda: check_panels(_prices(), _earnings().drop(columns="knowledge_date"), "2020-01-03"), "열이 없다"),
    (lambda: check_panels(_prices("2026-09-14"), _earnings(), "2026-09-14"), "애프터마켓"),
    (lambda: check_panels(_prices().assign(close=[np.nan, 2.0]), _earnings(), "2020-01-03"), "close"),
    (lambda: check_panels(_prices().assign(close=[0.0, 2.0]), _earnings(), "2020-01-03"), "close"),
    (lambda: check_panels(_prices().assign(close=[-1.0, 2.0]), _earnings(), "2020-01-03"), "close"),
    (lambda: check_panels(_prices().assign(close=[np.inf, 2.0]), _earnings(), "2020-01-03"), "close"),
    (lambda: check_signal(np.array([[np.inf]])), "inf"),
    (lambda: check_signal(np.array([[np.nan]])), "유효한"),
    (lambda: check_signal(np.array([1.0])), "2차원"),
    (lambda: check_forecast(_forecast(date="2020-01-02"), 20), "누출"),
    (lambda: check_forecast(_forecast(date="2020-01-03"), 20), "겹친다"),
    (lambda: check_forecast(_forecast(forecast=np.nan), 20), "NaN"),
    (lambda: check_forecast(_forecast(), 0), "horizon"),
    (lambda: check_weights([-0.1]), "음수"),
    (lambda: check_weights([0.8, 0.8]), "합"),
    (lambda: check_weights([np.nan]), "NaN"),
    (lambda: check_weights([0.1, 0.1], ["A", "A"]), "중복"),
    (lambda: check_weights([0.1, 0.1], ["A"]), "길이"),
    (lambda: check_attribution(pd.DataFrame({"date": ["d"], "f": [0.1], "specific": [0.1], "total": [0.3]}), ("f",)), "기여 합"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_violation_is_refused(call, match):
    with pytest.raises(CheckError, match=match):
        call()


def test_aftermarket_window_has_no_way_around():
    """"처리했다"고 말하는 인자는 없다 — main 의 read_prices 는 정규장 값을 만들지 못한다."""
    assert "handled" not in check_panels.__code__.co_varnames


def test_a_day_without_eligible_names_can_be_declared():
    """날짜별 0개는 기본 거부다. 예열 구간처럼 의도한 빈 날은 호출자가 말한다."""
    frame = pd.DataFrame({"code": ["A", "B"], "date": ["2020-01-02", "2020-01-03"], "eligible": [True, False]})
    check_universe(frame, min_per_day=0)


# --- 단계 표 --------------------------------------------------------------------------

def test_the_eleven_stages_are_the_users_process_in_order():
    assert [s.ko for s in Stage] == [
        "유니버스", "데이터", "알파 발굴", "알파 검증(IC/ICIR)", "ML 수익률 예측", "팩터 위험모델",
        "포트폴리오 최적화", "거래비용 모델", "백테스트", "주문 실행", "성과 기여·모니터링",
    ]
    assert [int(s) for s in Stage] == list(range(1, 12))


def _resolve(ref: str) -> None:
    path, _, symbol = ref.partition("::")
    assert (REPO / path).is_file(), f"경로가 없다: {path}"
    if symbol:
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert symbol in names, f"{path} 에 {symbol} 가 없다"


@pytest.mark.parametrize("stage", list(Stage), ids=lambda s: s.name)
def test_stage_table_points_at_real_functions(stage):
    assert STAGE_FUNCTIONS[stage]
    for ref in STAGE_FUNCTIONS[stage]:
        if ref == "research/experiments/ml_forecast_models.py" and not (REPO / ref).exists():
            continue                                        # ML 브랜치가 머지되기 전
        _resolve(ref)


@pytest.mark.parametrize("ref", ["research/experiments/nope.py", "src/swing_it/storage.py::nope"])
def test_table_check_catches_a_rotten_pointer(ref):
    with pytest.raises(AssertionError):
        _resolve(ref)


@pytest.mark.parametrize("stage", list(Stage), ids=lambda s: s.name)
def test_the_doc_table_matches_the_code_table(stage):
    doc = (REPO / "docs" / "PIPELINE.md").read_text(encoding="utf-8").splitlines()
    row = next(line for line in doc if line.startswith(f"| {int(stage)} | "))
    for ref in STAGE_FUNCTIONS[stage]:
        symbol = ref.partition("::")[2] or Path(ref).name
        assert symbol in row, f"docs/PIPELINE.md {int(stage)}단계 줄에 {symbol} 가 없다"


# --- 모듈 경계 ------------------------------------------------------------------------

BOOK_FILES = sorted((REPO / "src" / "swing_it" / "book").glob("*.py")) + sorted((REPO / "src" / "swing_it" / "pipeline").glob("*.py"))


def _imports(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append(node.module or "")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "core_layer":
            out.append(ast.unparse(node))
    return out


@pytest.mark.parametrize("path", BOOK_FILES, ids=lambda p: p.name)
def test_no_order_path_and_no_research_import(path):
    """주문·키움 호출을 하지 않고, src 는 research 를 부르지 않는다(린트 (a) 를 importlib 로 돌아가지 않는다)."""
    banned = ("kiwoom", "krx_quant_core.execution", "httpx", "requests", "research")
    assert not [m for m in _imports(path) if any(b in m for b in banned)]


def test_price_adjust_does_not_depend_on_the_new_packages():
    """``price_adjust.py`` 는 Airflow 라이브 마운트로 돈다. 나머지 import 범위는 ``tests/test_price_adjust.py`` 가 본다."""
    mods = _imports(REPO / "src" / "swing_it" / "price_adjust.py")
    assert not [m for m in mods if "pipeline" in m or "book" in m]


@pytest.mark.parametrize("name", ["accounting", "context", "costs", "metrics", "nulls", "optimized", "parity", "rank", "risk", "spec"])
def test_book_modules_import_on_the_default_pin(name):
    """기본 핀 0.7.0 에서도 import 된다 — 충격비용·최적화기는 부를 때 import 한다."""
    importlib.import_module(f"swing_it.book.{name}")


def test_gate_does_not_pull_the_book_in():
    """린트가 게이트를 읽는다. 게이트가 무거워지면 린트가 무의존이 아니게 된다(test_pipeline_gate 가 sys.modules 로도 본다)."""
    assert not [m for m in _imports(REPO / "src" / "swing_it" / "pipeline" / "gate.py") if m.startswith("swing_it.book")]


# --- core 레이어 지연 import ------------------------------------------------------------

@pytest.mark.parametrize("module", ["krx_quant_core.portfolio.optimize", "krx_quant_core.costs.impact"])
def test_missing_core_layer_says_what_to_do(module, monkeypatch):
    def _missing(name, *a, **k):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    with pytest.raises(CoreLayerMissing, match=r"0\.7\.0.*미배포"):
        core_layer(module)


def test_core_layer_does_not_hide_a_different_missing_dependency(monkeypatch):
    """cvxpy 같은 다른 의존성 문제를 "core 가 없다"로 둔갑시키지 않는다."""
    def _missing(name, *a, **k):
        raise ModuleNotFoundError("No module named 'cvxpy'", name="cvxpy")

    monkeypatch.setattr(importlib, "import_module", _missing)
    with pytest.raises(ModuleNotFoundError) as e:
        core_layer("krx_quant_core.portfolio.optimize")
    assert not isinstance(e.value, CoreLayerMissing)


def test_trade_costs_on_a_missing_layer_is_a_clear_error_not_a_zero(monkeypatch):
    """비용 레이어가 없다고 비용 0 으로 조용히 돌지 않는다."""
    from swing_it.book import costs

    monkeypatch.setattr(costs, "core_layer", lambda m: (_ for _ in ()).throw(CoreLayerMissing(m)))
    with pytest.raises(CoreLayerMissing):
        costs.trade_costs(np.array([0.1]), np.array([1e10]), np.array([0.02]), np.array(["KOSPI"], dtype=object),
                          pd.Timestamp("2023-06-01").date(), np.array([5.0]), 3e8)
    assert costs.trade_costs(np.array([0.0]), np.array([1e10]), np.array([0.02]), np.array(["KOSPI"], dtype=object),
                             pd.Timestamp("2023-06-01").date(), np.array([5.0]), 3e8)[0] == 0.0
