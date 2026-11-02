"""파이프라인 11단계 — 순서가 곧 규칙이다. 표준 라이브러리만 쓴다(`scripts/check_guardrails.py` 가 읽는다)."""

from __future__ import annotations

from enum import IntEnum

_KOREAN = {
    1: "유니버스",
    2: "데이터",
    3: "알파 발굴",
    4: "알파 검증(IC/ICIR)",
    5: "ML 수익률 예측",
    6: "팩터 위험모델",
    7: "포트폴리오 최적화",
    8: "거래비용 모델",
    9: "백테스트",
    10: "주문 실행",
    11: "성과 기여·모니터링",
}


class Stage(IntEnum):
    UNIVERSE = 1
    DATA = 2
    ALPHA = 3
    ALPHA_VALIDATION = 4
    FORECAST = 5
    RISK_MODEL = 6
    PORTFOLIO = 7
    COSTS = 8
    BACKTEST = 9
    EXECUTION = 10
    ATTRIBUTION = 11

    @property
    def ko(self) -> str:
        return _KOREAN[int(self)]

    @property
    def label(self) -> str:
        return f"{int(self)}단계 {self.ko}"

    @classmethod
    def parse(cls, value: "Stage | int | str") -> "Stage":
        """번호·영문 이름을 받는다. 모르는 값은 ``ValueError``."""
        if isinstance(value, Stage):
            return value
        if isinstance(value, str) and not value.strip().isdigit():
            try:
                return cls[value.strip().upper()]
            except KeyError:
                raise ValueError(f"모르는 단계: {value!r}") from None
        try:
            return cls(int(value))
        except (TypeError, ValueError):
            raise ValueError(f"모르는 단계: {value!r}") from None


#: 단계 → 실제 함수. ``docs/PIPELINE.md`` 표와 같고, 시험이 경로·심볼의 실재를 본다. 부르는 쪽은 러너다 — 게이트는 러너가 부른다.
STAGE_FUNCTIONS: dict[Stage, tuple[str, ...]] = {
    Stage.UNIVERSE: ("src/swing_it/features/universe.py::rank_band_eligible", "src/swing_it/features/universe.py::u_exclusion_causes"),
    Stage.DATA: ("src/swing_it/storage.py::read_prices", "src/swing_it/storage.py::read_earnings",
                 "src/swing_it/storage.py::read_periodic_filings"),
    Stage.ALPHA: ("src/swing_it/features/earnings_asof.py::yoy_strict_wide", "src/swing_it/features/earnings_asof.py::load_yoy_strict"),
    Stage.ALPHA_VALIDATION: ("src/swing_it/validation/ic.py::daily_rank_ic", "src/swing_it/validation/ic.py::daily_mean_ic",
                             "src/swing_it/validation/ic.py::quintile_excess", "src/swing_it/validation/ic.py::ic_summary",
                             "src/swing_it/validation/ic.py::ic_permutation_null", "src/swing_it/validation/ic.py::fold_ic"),
    Stage.FORECAST: ("research/experiments/ml_forecast_models.py", ),
    Stage.RISK_MODEL: ("src/swing_it/book/risk.py::sub_model", "src/swing_it/book/risk.py::active_risk",
                       "src/swing_it/book/riskmodel.py::build_models", "src/swing_it/book/riskmodel.py::build_panels",
                       "src/swing_it/book/riskmodel.py::load_guarded", "src/swing_it/book/seam.py::seam_guard"),
    Stage.PORTFOLIO: ("src/swing_it/book/rank.py::tranche_names", "src/swing_it/book/rank.py::r_target",
                      "src/swing_it/book/optimized.py::o_target", "src/swing_it/book/optimized.py::match_lambda"),
    Stage.COSTS: ("src/swing_it/book/costs.py::trade_costs", "src/swing_it/book/costs.py::linear_bp",
                  "src/swing_it/book/costs.py::trailing_vol"),
    Stage.BACKTEST: ("src/swing_it/book/accounting.py::simulate", "src/swing_it/book/accounting.py::check_invariants",
                     "src/swing_it/book/metrics.py::book_summary", "src/swing_it/book/parity.py::full_parity",
                     "src/swing_it/book/nulls.py::permutation_nulls",
                     "src/swing_it/diagnostics/hac.py::ols_hac"),
    Stage.EXECUTION: ("docs/execution/DESIGN.md", ),
    Stage.ATTRIBUTION: ("research/experiments/pead_optimizer_b_report.py::attribution_report", ),
}
