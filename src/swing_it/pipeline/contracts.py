"""단계별 입출력 계약 — 각 단계가 무엇을 받고 무엇을 내는가.

계약 객체는 만들어질 때 스스로 검사한다(``__post_init__``). 어긋나면 :class:`ContractError` 다. 판정 필드는 없다
— 리포터-not-판정기(GUARDRAILS §8)이고, 통과 여부는 사람이 읽고 원장에 적는다.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from swing_it.pipeline.stages import Stage

#: 2026-09-14~ 일봉 고가·저가·종가·거래량에 애프터마켓 체결이 섞였다(레포 CLAUDE.md "데이터 함정").
AFTERMARKET_FROM = "2026-09-14"


class ContractError(ValueError):
    """단계 입출력이 계약과 다르다."""


class CoreLayerMissing(ImportError):
    """krx-quant-core 에 그 레이어가 없다. swing-it 기본 핀(0.7.0)에는 위험모델·최적화·충격비용이 없다."""


def core_layer(module: str) -> ModuleType:
    """미배포 core 레이어를 부를 때만 import 한다. 없으면 무엇을 해야 하는지 말한다."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("krx_quant_core"):
            raise                                   # 그 레이어 안의 다른 의존성 문제는 가리지 않는다
        raise CoreLayerMissing(
            f"{module} 가 없다. swing-it 기본 핀 krx-quant-core==0.7.0 에는 위험모델·최적화·충격비용이 없고, core main 에만 있다"
            f"(미배포 — PyPI 배포는 사용자 지시가 필요하다). 지금 쓰려면 러너를 "
            f'`uv run --with "krx-quant-core[portfolio,db,fast] @ git+file://<core 체크아웃>@<커밋>"` 으로 돌린다'
            f"(research/experiments/pead_optimizer_b.py 머리 참고)."
        ) from e


def expect(value: Any, kind: "type | tuple[type, ...]", name: str) -> Any:
    """앞 단계 산출물을 받는 자리의 형 검사 — 앞 단계 없이 날것을 넘기지 못하게 한다."""
    if not isinstance(value, kind):
        want = kind.__name__ if isinstance(kind, type) else "·".join(k.__name__ for k in kind)
        raise ContractError(f"{name} 은 {want} 여야 한다: {type(value).__name__}")
    return value


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} 이 비었다")


def _long_frame(frame: Any, cols: Sequence[str], name: str, *, key: Sequence[str] = ("code", "date")) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"{name} 은 DataFrame 이어야 한다: {type(frame).__name__}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ContractError(f"{name} 에 열이 없다: {missing}")
    if frame.empty:
        raise ContractError(f"{name} 이 비었다 — 빈 응답을 성공으로 넘기지 않는다")
    if key and frame.duplicated(list(key)).any():
        raise ContractError(f"{name} 의 {tuple(key)} 가 중복이다 — 정정 버전이 새어 나왔을 수 있다")


def _day(values: pd.Series, name: str) -> pd.Series:
    days = pd.to_datetime(values.astype(str), errors="coerce", format="mixed")
    if days.isna().any():
        raise ContractError(f"{name} 에 날짜가 아닌 값·결측이 있다")
    return days


def _finite(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ContractError(f"{name} 에 NaN·inf 가 있다")
    return arr


@dataclass(frozen=True)
class UniverseMask:
    """1단계. ``frame``: 긴 형식 code·date·eligible(bool), (code, date) 하나씩.

    ``survivorship``: 폐지 종목을 어떻게 다뤘는지 적는다. 비우면 안 된다 — 비워 두면 생존편향이 조용히 들어온다.
    """

    frame: pd.DataFrame
    definition: str
    survivorship: str

    def __post_init__(self) -> None:
        _long_frame(self.frame, ("code", "date", "eligible"), "UniverseMask.frame")
        if self.frame["eligible"].dtype != bool:
            raise ContractError(f"eligible 은 bool 이어야 한다: {self.frame['eligible'].dtype}")
        if not self.frame["eligible"].any():
            raise ContractError("적격 종목이 하나도 없다")
        _text(self.definition, "definition")
        _text(self.survivorship, "survivorship")


@dataclass(frozen=True)
class DataPanels:
    """2단계. 지식 시점(knowledge date) 보증이 붙은 패널.

    - ``earnings`` 는 접지 않은 전 버전이고, 모든 행이 ``knowledge_col``·``avail_date`` 를 가지며 ``end`` 이하다.
    - ``prices`` 는 ``end`` 이하이고 (code, date) 하나씩이다.
    - ``end`` 가 애프터마켓 오염 구간이면 ``aftermarket_handled`` 를 명시해야 한다.
    """

    prices: pd.DataFrame
    earnings: pd.DataFrame
    universe: UniverseMask
    end: str
    fingerprint: str
    knowledge_col: str = "knowledge_date"
    aftermarket_handled: bool = False

    def __post_init__(self) -> None:
        expect(self.universe, UniverseMask, "universe")
        _text(self.fingerprint, "fingerprint")
        _long_frame(self.prices, ("code", "date", "close"), "DataPanels.prices")
        end = _day(pd.Series([self.end]), "end").iloc[0]
        if _day(self.prices["date"], "prices.date").max() > end:
            raise ContractError(f"prices 에 end({self.end}) 뒤 날짜가 있다 — 미래 데이터")
        _long_frame(self.earnings, ("code", "period", "avail_date", self.knowledge_col),
                    "DataPanels.earnings", key=())
        for col in ("avail_date", self.knowledge_col):
            if _day(self.earnings[col], f"earnings.{col}").max() > end:
                raise ContractError(f"earnings.{col} 에 end({self.end}) 뒤 날짜가 있다 — 그때 몰랐던 값")
        if self.end >= AFTERMARKET_FROM and not self.aftermarket_handled:
            raise ContractError(f"end {self.end} 는 애프터마켓 오염 구간({AFTERMARKET_FROM}~)이다. "
                                "정규장 값으로 고쳤다면 aftermarket_handled=True 로 명시한다")

    def trading_dates(self) -> list[str]:
        return sorted(self.prices["date"].astype(str).unique())


@dataclass(frozen=True)
class Signal:
    """3단계. ``frame``: 긴 형식 code·date·signal. NaN 은 그날 신호 없음이다(유니버스 밖 포함). ``known_at``: 언제 안 값인가."""

    frame: pd.DataFrame
    name: str
    known_at: str

    def __post_init__(self) -> None:
        _long_frame(self.frame, ("code", "date", "signal"), "Signal.frame")
        _text(self.name, "name")
        _text(self.known_at, "known_at")
        if np.isinf(self.frame["signal"].to_numpy(dtype=np.float64)).any():
            raise ContractError("signal 에 inf 가 있다")
        if self.frame["signal"].notna().sum() == 0:
            raise ContractError("유효한 신호가 하나도 없다")


@dataclass(frozen=True)
class ICReport:
    """4단계. 숫자만 보고한다. ``icir`` 은 IC 일별 열을 주는 core 에서만 나온다(기본 핀 0.7.0 은 ``None``)."""

    signal_name: str
    horizon: int
    ic_mean: float
    ic_nw_t: float
    n_days: int
    frac_positive: float
    regimes: tuple[Mapping[str, Any], ...] = ()
    icir: "float | None" = None

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ContractError(f"horizon 은 양수여야 한다: {self.horizon}")
        if self.n_days <= 0:
            raise ContractError("IC 를 잰 날이 없다 — 표본 부족을 숫자로 넘기지 않는다")
        _finite(self.ic_mean, "ic_mean")


@dataclass(frozen=True)
class Forecast:
    """5단계. ``frame``: 긴 형식 code·date·forecast·train_end. 모든 행이 ``date > train_end`` — 표본 안 예측은 안 받는다."""

    frame: pd.DataFrame
    model: str
    horizon: int

    def __post_init__(self) -> None:
        _long_frame(self.frame, ("code", "date", "forecast", "train_end"), "Forecast.frame")
        _text(self.model, "model")
        _finite(self.frame["forecast"], "forecast")
        if (_day(self.frame["date"], "date") <= _day(self.frame["train_end"], "train_end")).any():
            raise ContractError("학습 구간 안 예측이 있다 — 누출")


@dataclass(frozen=True)
class RiskModelOutput:
    """6단계. ``model``: core ``RiskModel`` 또는 같은 모양(``X``·``F``·``D``·``names``·``codes``, NaN 없음)."""

    asof: str
    model: Any
    label: str

    def __post_init__(self) -> None:
        _text(self.asof, "asof")
        _text(self.label, "label")
        for attr in ("X", "F", "D"):
            if not hasattr(self.model, attr):
                raise ContractError(f"위험모델에 {attr} 가 없다")
            _finite(getattr(self.model, attr), f"model.{attr}")


@dataclass(frozen=True)
class TargetWeights:
    """7단계. 롱온리, 합 ≤ 1(나머지는 현금). ``status`` 는 최적화기 상태이고 ``optimal`` 이 아니면 거래하지 않는다."""

    asof: str
    codes: np.ndarray
    weights: np.ndarray
    method: str
    status: str = "optimal"

    def __post_init__(self) -> None:
        w = _finite(self.weights, "weights")
        if w.ndim != 1 or len(self.codes) != len(w):
            raise ContractError(f"codes({len(self.codes)})·weights({w.shape}) 길이가 다르다")
        if len(set(np.asarray(self.codes).tolist())) != len(w):
            raise ContractError("codes 가 중복이다")
        if (w < -1e-9).any():
            raise ContractError("음수 비중 — 실매매 북은 롱온리다")
        if w.sum() > 1 + 1e-6:
            raise ContractError(f"비중 합 {w.sum():.6f} > 1")
        _text(self.method, "method")


@dataclass(frozen=True)
class CostEstimate:
    """8단계. 정가 편도 비용(bp)과 NAV 대비 비용. 판정은 ``multiplier=2`` 다(레포 CLAUDE.md)."""

    trade_date: Any
    codes: np.ndarray
    one_way_bp: np.ndarray
    nav_cost: float
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        bp = _finite(self.one_way_bp, "one_way_bp")
        if (bp < 0).any():
            raise ContractError("비용이 음수다")
        if len(self.codes) != len(bp):
            raise ContractError("codes·one_way_bp 길이가 다르다")
        if not self.multiplier > 0:
            raise ContractError(f"multiplier 는 양수여야 한다: {self.multiplier}")
        _finite(self.nav_cost, "nav_cost")


#: 9단계 결과가 비용을 뗐는지. 총수익(gross)을 순수익처럼 읽는 실수를 막으려고 이름을 강제한다.
COST_BASES = ("gross", "net_1x", "net_2x")


@dataclass(frozen=True)
class BacktestResult:
    """9단계. 사전등록 config 한 개의 결과와 그 연구의 시행 수(다중검정 원장 ``n_trials``)."""

    label: str
    periods: pd.DataFrame
    summary: Mapping[str, Any]
    cost_basis: str
    n_trials: int
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.label, "label")
        if not isinstance(self.periods, pd.DataFrame) or self.periods.empty:
            raise ContractError("periods 가 비었다 — 빈 응답을 성공으로 넘기지 않는다")
        if self.cost_basis not in COST_BASES:
            raise ContractError(f"cost_basis 는 {COST_BASES} 중 하나여야 한다: {self.cost_basis!r}")
        if self.n_trials < 1:
            raise ContractError("시행 수가 원장에 없다 — 다중검정 N 을 손으로 적지 않는다")


@dataclass(frozen=True)
class OrderPlan:
    """10단계. 주문 **계획**이다. 구현 전까지 ``dry_run=True`` 만 받는다(``docs/execution/DESIGN.md`` §4)."""

    asof: str
    codes: tuple[str, ...]
    sides: tuple[str, ...]
    shares: tuple[int, ...]
    limit_prices: tuple[float, ...]
    dry_run: bool = True

    def __post_init__(self) -> None:
        if self.dry_run is not True:
            raise ContractError("주문 계획은 dry_run 만 받는다 — 실주문 경로는 없다(docs/execution/DESIGN.md)")
        n = len(self.codes)
        if not n == len(self.sides) == len(self.shares) == len(self.limit_prices):
            raise ContractError("codes·sides·shares·limit_prices 길이가 다르다")
        if set(self.sides) - {"BUY", "SELL"}:
            raise ContractError(f"side 는 BUY·SELL 이어야 한다: {sorted(set(self.sides))}")
        if any(not isinstance(q, int) or q <= 0 for q in self.shares):
            raise ContractError("주문 수량은 양의 정수여야 한다")
        if any(not p > 0 for p in self.limit_prices):
            raise ContractError("지정가는 양수여야 한다 — 시장가는 쓰지 않는다")


@dataclass(frozen=True)
class Attribution:
    """11단계. ``frame``: date·<팩터…>·specific·total. 날짜마다 ``total = Σ 팩터 + specific``."""

    frame: pd.DataFrame
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        cols = ("date", *self.names, "specific", "total")
        _long_frame(self.frame, cols, "Attribution.frame", key=("date",))
        parts = self.frame[[*self.names, "specific"]].to_numpy(dtype=np.float64).sum(axis=1)
        if not np.allclose(parts, self.frame["total"].to_numpy(dtype=np.float64), atol=1e-9, equal_nan=True):
            raise ContractError("기여 합이 total 과 다르다 — 설명 안 된 몫이 숨었다")


@dataclass(frozen=True)
class StageContract:
    """한 단계의 입출력. ``takes`` 는 앞 단계 산출물 중 이 단계가 받는 것이다."""

    stage: Stage
    takes: tuple[type, ...]
    gives: type


#: 11단계 전체의 입출력 — ``docs/PIPELINE.md`` 표와 같다.
STAGE_CONTRACTS: dict[Stage, StageContract] = {
    c.stage: c
    for c in (
        StageContract(Stage.UNIVERSE, (), UniverseMask),
        StageContract(Stage.DATA, (UniverseMask,), DataPanels),
        StageContract(Stage.ALPHA, (DataPanels,), Signal),
        StageContract(Stage.ALPHA_VALIDATION, (Signal, DataPanels), ICReport),
        StageContract(Stage.FORECAST, (Signal, DataPanels), Forecast),
        StageContract(Stage.RISK_MODEL, (DataPanels,), RiskModelOutput),
        StageContract(Stage.PORTFOLIO, (RiskModelOutput,), TargetWeights),
        StageContract(Stage.COSTS, (TargetWeights,), CostEstimate),
        StageContract(Stage.BACKTEST, (Signal, DataPanels), BacktestResult),
        StageContract(Stage.EXECUTION, (TargetWeights,), OrderPlan),
        StageContract(Stage.ATTRIBUTION, (BacktestResult,), Attribution),
    )
}
