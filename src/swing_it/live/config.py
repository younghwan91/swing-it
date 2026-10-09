"""스윙 라이브 전략 v0 설정 — 사전등록 CONVERGENCE A 와 같은 정의(설계 §2).

수치 출처(탐색하지 않는다):

- 진입: ``research/logs/sepa_explore/CONVERGENCE.md`` 팔 A — 뼈대 S(추세 템플릿 ∧ RS ≥ 70 ∧ 피벗 돌파 ∧ 거래량 1.5×)
  ∧ breadth > 0.5 ∧ PEAD 순위 ≥ 0.70 ∧ 상대 수렴 rel ≤ 1.0 ∧ 수급 F = 1.
  **유니버스만 다르다:** 라이브는 ADV20 ≥ 5,000 백만원(50억) 한 구간(사전등록 U_s 50~200억 ∪ 이미 본 ≥200억).
  PEAD 순위 모집단(그날 위생 통과 ∧ ADV20 ≥ 5,000)은 사전등록과 같다 — 라이브 유니버스 = 그 모집단.
- 청산: ``research/signals/minervini_exits.py`` 프로토콜 — 수축 저점 × 0.99 손절 [3%, 10%], 2R 절반 익절 + 본전,
  나머지 MA50 추적(다음날 시가), 120거래일 최대 보유(안전 상한). **설계 §6:** 시간 손절 대신 돌파 실패(익절 전 종가 < 피벗 →
  다음날 시가) — 연구 시뮬레이터의 ``failed_breakout=True, time_stop_days=None`` 과 같다.
- 사이징·위험: 설계 §2 — 위험 1% / 10% 상한 / 10종목 / 낙폭 −15% 신규 진입 정지.

둘 다 frozen dataclass 라 원장·리포트에 ``dataclasses.asdict`` 로 그대로 적을 수 있다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: 상한가 시가 판정 허용폭 — :data:`swing_it.features.korea.LIMIT_UP_REL_TOL` 과 같은 값(0.7%p).
#: 여기 사본을 두는 이유: config 는 무의존(다른 에이전트·CLI 가 가볍게 import)이어야 한다.
#: 같은 값임은 ``tests/test_live_strategy.py`` 가 못박는다.
LIMIT_UP_REL_TOL = 0.007


@dataclass(frozen=True)
class StrategyConfig:
    """전략 v0 수치 전부(설계 §2). 기본값 = 사전등록 A + 라이브 유니버스(ADV20 ≥ 50억).

    단위: 금액·거래대금은 백만원(``adv_floor``) — 가격은 원. 비율은 소수(0.05 = 5%).
    """

    # --- 유니버스(korea.universe_mask 위생 + ADV 하한) ---
    adv_floor: float = 5_000.0          # ADV20 ≥ 5,000 백만원(50억) — 상한 없음(두 구간 합)
    adv_window: int = 20
    min_price: float = 1_000.0          # 원자료 종가(daily_bars.close) 기준
    min_history: int = 252

    # --- 뼈대 S(sepa.PREREG_SKELETON) ---
    rs_min: float = 70.0
    breadth_min: float = 0.5            # 그 종목 시장의 MA50 위 비율 > 0.5
    pivot_window: int = 50              # 피벗 = 직전 50일 최고 고가
    vol_mult: float = 1.5               # 돌파일 거래량 ≥ 1.5 × 직전 50일 평균

    # --- 확인 층 ---
    pead_rank_min: float = 0.70         # PEAD 순위 ≥ 0.70 (모집단 = 그날 위생 ∧ ADV20 ≥ adv_floor)
    rel_max: "float | None" = 1.0       # 상대 수렴 rel ≤ 1.0. None = 조건 끔(변형용)
    require_flow: bool = True           # 수급 F = 1. False = 조건 끔(F 가 없는 2024-02-19 이전 백테스트 변형)

    # --- 상대 수렴 rel(CONVERGENCE.md 정의) ---
    rel_window: int = 10                # 분자: t−10..t−1 고저폭
    rel_med_window: int = 250           # 분모: 그 값의 250일 이동 중앙값
    rel_min_periods: int = 120
    rel_gap: int = 11                   # 분모 창 끝 = t−11

    # --- 진입 체결(sepa 규약) ---
    chase: float = 0.05                 # 지정가 상한 = 피벗 × 1.05 (= 백테스트 추격 금지)
    limit_up_ratio: float = 1.30 - LIMIT_UP_REL_TOL   # 분할조정 패널의 상한가 시가 판정(sepa.LIMIT_UP_RATIO)

    # --- 청산(minervini_exits.PREREG_MINERVINI) ---
    contraction_window: int = 10        # 수축 저점 = min(low[t−10..t]) (돌파일 포함 11일)
    stop_mult: float = 0.99             # 손절 기준 = 수축 저점 × 0.99
    stop_floor: float = 0.03            # 손절폭 하한
    stop_cap: float = 0.10              # 손절폭 상한
    take_r: float = 2.0                 # 절반 익절 = 진입가 × (1 + 2d)
    partial_frac: float = 0.5           # 익절 비율(보유 수량 대비)
    trail_ma: int = 50                  # 나머지 추적 MA
    #: 돌파 실패(설계 §6, 2026-10-10 사용자 지시 — 미너비니 원전): 절반 익절 전 어느 날이든 **종가 < 피벗**(돌파 기준가 =
    #: 직전 50일 최고 고가, 계획의 ``pivot``) 이면 다음 거래일 시가에 전량 청산(MA50 추적과 같은 경로: d 종가 결정 → d+1 시가).
    failed_breakout: bool = True
    #: 시간 손절(20거래일째 종가 ≤ 진입가) — 설계 §6 으로 **폐지**(기본 None). 옛 프로토콜 재현용으로만 숫자를 준다.
    time_stop_days: "int | None" = None
    max_hold: int = 120                 # 최대 보유(진입일 = 1일째) — 안전 상한(신호 아님)

    # --- 백테스트 체결 모형(라이브 주문엔 쓰지 않는다) ---
    #: 손절·본전 체결 미끄러짐(bp). 일봉 백테스트는 손절가·진입가에 정확히 판다고 보지만, 라이브는 60초 REST 폴링으로
    #: 터치를 보고 나서 주문을 내므로 그보다 불리하게 체결된다(감사 AUDIT.md). 백테스트는 stop·breakeven 매도가에
    #: ``× (1 − bps/1e4)`` 를 곱해 현금·손익에 반영한다(트레이드 ``ret`` 은 연구와 같은 미끄러짐 전 총수익, ``net_ret`` 에 반영).
    #: 패리티 모드(``unlimited``)에선 0.
    stop_slippage_bps: float = 50.0

    # --- 사이징(설계 §2) ---
    #: 주수 = floor(min(risk × 자산 / (기준가 × d_est), max_position_frac × 자산 / 기준가)). **d_est ≤ stop_cap = 10% 이므로
    #: 위험 1% 항은 늘 10% 상한보다 크다 — 실질 사이징은 "종목당 자산 10% 고정, 손절 폭 d(거의 10%)" 다(REVIEW_BACKTEST M5).
    #: 위험 항은 stop_cap 을 올리거나 risk 를 낮추는 변형에서만 묶인다. 코드는 두 항을 그대로 둔다.
    risk_per_trade: float = 0.01        # 위험 = 자산 × 1% (현 수치에선 묶이지 않는다)
    max_position_frac: float = 0.10     # 종목당 자산 10% 상한 (실질 사이징)
    max_positions: int = 10             # 최대 동시 보유 종목

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RiskConfig:
    """위험 한도(설계 §2 "위험 한도"). 일 손실은 코어 ``KillSwitch``, 다일 낙폭은 이 레포의 파일 킬.

    - ``daily_loss_limit_pct``: 코어 ``KillSwitchConfig.daily_loss_limit_pct`` 로 넘길 값(자산 대비, 실현 손익 기준).
    - ``drawdown_kill``: 자산 고점 대비 이 비율 이상 빠지면 **신규 진입 정지**(보유 청산은 계속 — 매도는 막지 않는다).
      정지는 ``stop_file`` 을 만들어 표시한다.
    - 재개 규칙(REVIEW_BACKTEST M4): 킬 뒤 ``kill_resume_days`` 거래일이 지나거나 자산이 고점 × (1 − ``kill_resume_recovery``)
      이상으로 회복하면 — **둘 중 먼저 오는 것** — 신규 진입을 다시 허용하고 고점을 그때 자산으로 다시 잰다("둘 중 늦은 것" 으로
      두면 신규 진입이 없는 동안 자산이 평평해 영원히 재개되지 않는다). 백테스트는 이 규칙을 그대로 돌리고(``dd_kill`` 켠 변형),
      **라이브는 사람이 ``stop_file`` 을 지워야 풀린다**(자동 재개 없음 — 규칙은 사람이 지울 때의 기준이다).
    - ``max_qty_hard_cap_frac``: 주문 가드 수량 상한 = 자산 × 이 비율 / 가격(사이징 10% 상한 위 여유).
    - ``price_band_pct``: 주문 가드 가격 밴드(지정가가 기준가 대비 이만큼 넘게 벗어나면 거부) — 피벗×1.05 지정가가
      전일종가 대비 5% 넘게 위일 수 있어 ≥ 0.06.
    """

    daily_loss_limit_pct: float = 0.03
    drawdown_kill: float = 0.15
    kill_resume_days: int = 60          # 킬 뒤 이만큼 거래일이 지나면 재개(먼저 오는 것)
    kill_resume_recovery: float = 0.05  # 또는 자산이 고점 × (1 − 0.05) 이상 회복하면 재개
    stop_file: str = "data/swing_STOP"
    max_qty_hard_cap_frac: float = 0.12
    price_band_pct: float = 0.06

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_STRATEGY = StrategyConfig()
DEFAULT_RISK = RiskConfig()
