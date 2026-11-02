"""북의 규칙 한 벌. 기본값은 PEAD 단계 B 사전등록 값이다(``research/logs/pead_optimizer/PREREG.md`` §2·PO-1.6)."""

from __future__ import annotations

from dataclasses import dataclass

from swing_it.features.universe import RANK_BAND, RANK_BAND_ADV_FLOOR


@dataclass(frozen=True)
class BookSpec:
    """리밸런스 간격·보유 지평·트랜치·유니버스·최적화 제약. 새 연구는 값을 바꿔 넘긴다."""

    step: int = 20                  # 리밸런스 간격(거래일)
    horizon: int = 60               # 트랜치 보유 지평 — 활성 트랜치 수 = horizon // step
    top_n: int = 10                 # 트랜치 종목 수
    min_names: int = 20             # 적격이 이보다 적으면 그 트랜치는 없다
    adv_floor: float = RANK_BAND_ADV_FLOOR
    cap_rank: tuple[int, int] = RANK_BAND
    max_weight: float = 0.10        # O 종목 상한
    max_turnover: float = 2.0 / 3.0  # O 편도 회전율 상한
    max_participation: float = 0.05  # O 참여율 상한

    @property
    def n_tranches(self) -> int:
        return self.horizon // self.step


DEFAULT_SPEC = BookSpec()


@dataclass(frozen=True)
class Cell:
    """최적화 설정 한 칸: α 지평 × 업종 밴드 × AUM. 기본값 = PEAD 단계 B 1차 (60, 0.10, 3억)."""

    h_alpha: int = 60
    band: float = 0.10
    aum: float = 3e8

    @property
    def name(self) -> str:
        return f"H{self.h_alpha}_band{self.band:.2f}_aum{self.aum:.0e}"


PRIMARY = Cell()


@dataclass(frozen=True)
class Flags:
    """실제 규칙 스위치(PO-1.4 ①~⑥). 기본 = 실제 규칙 전부 켬. 전부 끈 것이 패리티 모드다(:func:`swing_it.book.parity.parity_book`)."""

    lag: int = 1  # ① t+1 종가 체결
    drift: bool = True  # ② 스텝 사이 매수 보유 표류(값 회계)
    forced_sale: bool = True  # ③ 위험모델 이탈 강제 매도
    delist_cash: bool = True  # ④ 폐지 대금 현금 → 다음 리밸런스 재투자
    no_trade: bool = True  # ⑤ 체결일 거래량 0·가격 없음 → 거래 불가
    costs: bool = True  # ⑥ 비용(1배로 기록, 2배는 배수)
