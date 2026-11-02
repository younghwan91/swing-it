"""북이 읽는 패널 묶음과 스텝 기록."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from krx_quant_core.market.codes import Market

from swing_it.book.costs import HALF_SPREAD_BP, trailing_vol
from swing_it.engine.panels import ffill_axis1


@dataclass
class Ctx:
    """``code × date`` 패널(위험모델 좌표, 열 = 거래일). 금액 단위는 이름에 적었다.

    ``models``: t → (idx, X, F, D, names), 일 단위. 위험모델을 안 쓰는 순위 북은 :func:`models_from_mask` 로 종목 집합만 넘긴다.
    """

    dates: pd.DatetimeIndex
    C: np.ndarray  # 수정 종가(이음매 가드 뒤)
    tv_won: np.ndarray  # 일 거래대금(원)
    adv: np.ndarray  # t 제외 20일 평균 거래대금(백만원, core _trailing_adv)
    cap: np.ndarray
    sig: np.ndarray  # 진짜 YoY 신호
    ret: np.ndarray  # 일수익(가드 뒤)
    lab: np.ndarray  # WICS 섹터
    market: np.ndarray  # "KOSPI"/"KOSDAQ"
    grid: list[int]
    models: dict  # t → (idx, X, F, D, names)
    delist_after: np.ndarray  # 마지막 거래일 열 번호(폐지 아님 = 큰 수)
    Cff: np.ndarray = field(default=None)
    hs_bp: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.Cff is None:
            self.Cff = ffill_axis1(self.C)
        if self.hs_bp is None:
            # 표기는 Market.parse 가 푼다(한글·소문자). 모르는 값은 예외다. KONEX 만 정가 값이 없어 비용 큰 쪽(KOSDAQ)으로 둔다
            self.hs_bp = np.array([HALF_SPREAD_BP.get(Market.parse(m).value, HALF_SPREAD_BP["KOSDAQ"]) for m in self.market])
        self._u = {}
        self._vol = {}

    def in_u(self, t: int) -> np.ndarray:
        if t not in self._u:
            self._u[t] = index_mask(self.C.shape[0], self.models[t][0])
        return self._u[t]

    def vol(self, t: int) -> np.ndarray:
        """``trailing_vol(ret, t)`` 캐시 — 널이 같은 격자일을 수백 번 다시 잰다. 읽기 전용이다."""
        if t not in self._vol:
            v = trailing_vol(self.ret, t)
            v.flags.writeable = False
            self._vol[t] = v
        return self._vol[t]


def models_from_mask(mask: np.ndarray, grid: list[int]) -> dict:
    """위험모델 없는 순위 북의 ``Ctx.models`` — 격자일마다 U_t 종목만 적는다. 사전 위험 칸은 NaN 으로 남는다."""
    return {t: (np.flatnonzero(mask[:, t]), None, None, None, []) for t in grid}


def index_mask(n: int, idx) -> np.ndarray:
    """행 번호 목록 → 길이 ``n`` 불리언."""
    m = np.zeros(n, bool)
    m[np.asarray(idx, dtype=int)] = True
    return m


def delisted(ctx: Ctx, f: int) -> np.ndarray:
    """f 에 폐지로 본다 = ``delisted_stocks.last_trade_date`` 가 f 전 ∧ f 에 가격 없음(코드 재사용 방어, PO-6)."""
    return (ctx.delist_after < f) & ~np.isfinite(ctx.C[:, f])


@dataclass
class StepRec:
    k: int
    t: int
    date: str
    gross: float  # G_k − 1 (비용 전, 체결 뒤 → 다음 체결 전)
    cost1: float  # 체결일 비용(1배, NAV 대비)
    bench: float
    turnover: float
    exante_20: float  # 사전 20일 능동위험(목표 비중, 그 날 모델)
    factor_share: float
    status: str = "optimal"
    lam: float = float("nan")
    lam_hit: bool = True
    est_cost: float = float("nan")
    n_names: int = 0
    n_forced: int = 0
    n_delist: int = 0
    n_frozen: int = 0
    n_untradeable: int = 0
    u_excl: float = float("nan")
    u_excl_cap_missing: float = float("nan")  # u_excl 중 시총 없음(주식수 공백일 수 있다, u_exclusion_causes)
    u_excl_model: float = float("nan")  # u_excl 중 시총은 있는데 위험모델 밖
    cap_missing_in_u: float = float("nan")  # U_t 안인데 시총이 없어 띠에서 빠진 비율
    perm_nan_share: float = float("nan")
    contrib: list = field(default_factory=list)  # (종목, 체결 뒤 비중, 종목 수익, 1배 비용)
    # PO-7 — 회계 창·불변식·보고
    f: int = -1  # 체결일 열
    f_next: int = -1  # 다음 체결일 열(스텝 끝)
    lam_cause: str = ""
    lam_statuses: dict = field(default_factory=dict)  # 실패 풀이의 status 별 횟수(보고만)
    n_buy_outside: int = 0  # O: 해가 no_buy 종목을 산 수(불변식)
    max_part: float = float("nan")  # O: 해의 max |Δw|·AUM/ADV(불변식)
    n_tranches: int = 0  # R: 활성 트랜치 수
    cash: float = 0.0  # 체결 뒤 현금 비중
    min_w: float = 0.0
    max_w: float = 0.0  # PO-9: 거래 가능·안 얼린 종목 중 최대(체결 뒤)
    max_w_frozen: float = 0.0  # 얼린 보유(거래 불가) 중 최대 — 표류로 상한을 넘을 수 있다(보고만)
    sector_dev: float = float("nan")  # O: max_g |Σw_g − b_g| (체결 뒤)
    sector_tol: float = float("nan")
    turn_bound: float = float("nan")  # O(구축 뒤): 2/3 + 강제 매도·폐지·표류·재비례 몫
    n_dust: int = 0  # O: 해 직후 0 으로 지운 먼지 비중 수
    dust_mass: float = 0.0  # O: 지운 먼지 비중 합
    delist_mass: float = 0.0  # 폐지 현금화 비중(비용은 있고 회전에는 안 든다)
    n_buy_limit_up: int = 0  # 체결일 종가가 상한가(+29% 이상)인 매수 수
    alarm_top3: list = field(default_factory=list)  # 팩터 몫 > 20% 인 날의 큰 노출 셋 (이름, 노출)
