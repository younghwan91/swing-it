"""스윙 라이브 전략 v0 설정 — 사전등록 CONVERGENCE A 와 같은 정의(설계 §2) + GA2 유전자 19개의 1:1 대응(§3 인계).

수치 출처(탐색하지 않는다):

- 진입: ``research/logs/sepa_explore/CONVERGENCE.md`` 팔 A — 뼈대 S(추세 템플릿 ∧ RS ≥ 70 ∧ 피벗 돌파 ∧ 거래량 1.5×)
  ∧ breadth > 0.5 ∧ PEAD 순위 ≥ 0.70 ∧ 상대 수렴 rel ≤ 1.0 ∧ 수급 F = 1.
  **유니버스만 다르다:** 라이브는 ADV20 ≥ 5,000 백만원(50억) 한 구간(사전등록 U_s 50~200억 ∪ 이미 본 ≥200억).
  PEAD 순위 모집단(그날 위생 통과 ∧ ADV20 ≥ 5,000)은 사전등록과 같다 — 라이브 유니버스 = 그 모집단.
- 청산: ``research/signals/minervini_exits.py`` 프로토콜 — 수축 저점 × 0.99 손절 [3%, 10%], 2R 절반 익절 + 본전,
  나머지 MA50 추적(다음날 시가), 120거래일 최대 보유(안전 상한). **설계 §6:** 시간 손절 대신 돌파 실패(익절 전 종가 < 피벗 →
  다음날 시가) — 연구 시뮬레이터의 ``failed_breakout=True, time_stop_days=None`` 과 같다.
- 사이징·위험: 설계 §2 — 위험 1% / 10% 상한 / 10종목 / 낙폭 −15% 신규 진입 정지.

**GA2 유전자 대응(REVIEW_INTEGRATION §2, 전부 1:1):** :meth:`StrategyConfig.from_genes` / :meth:`StrategyConfig.to_genes`.
추세 템플릿은 GA2 처럼 4개 성분(``near_high``·``above_low``·``ma_stack``·``ma200_slope_days``)으로 분해돼 있고 기본값이 원전
템플릿(0.75·1.30·3·20)이다. 연속 문턱은 ``None`` = 끔(NaN 도 통과 — GA 마스크와 같다). ``pivot_window = 0`` = 돌파 요구 없음
(피벗은 직전 50일 고가 — 추격·돌파 실패 기준), ``take_r = inf`` = 익절 없음(추적은 전량, 1일째부터), ``trail_ma = 0`` = 추적 없음.
수급 유전자(``smart_min``·``foreign_pos``·``persist_min``)를 하나라도 주면 ``require_flow`` 의 F 규칙 대신 그 조건을 쓴다
(기존 F = smart > 0 ∧ foreign > 0 은 ``smart_min=0, foreign_pos=True, persist_min=0`` 과 경계 smart = 0 한 점만 다르다).

**청산 v2 유전자(``MINERVINI_EXITS_V2.md`` 2e47f59, 선택 — 기본값 = 현 동작):** ``be_r``(본전 이동 배수, ∞ = 끔)·``take_frac``
(→ ``partial_frac``)·``strength_sell``(강세 매도 단계 0~3 → ``strength_ret``·``strength_ma_ext``)·``fb_tol``·``fb_days``. 19개 유전자와
따로 :data:`EXIT_V2_GENE_FIELDS` 에 두고, 하나라도 주면 다섯 개 다 있어야 한다. 그때 ``take_r`` 은 v2 단계(1.0 포함)도 허용한다.
피벗 허용폭의 "끔" 은 ``failed_breakout = 0`` 이다.

둘 다 frozen dataclass 라 원장·리포트에 ``dataclasses.asdict`` 로 그대로 적을 수 있다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from swing_it.units import eok

#: 상한가 시가 판정 허용폭 — :data:`swing_it.features.korea.LIMIT_UP_REL_TOL` 과 같은 값(0.7%p).
#: 여기 사본을 두는 이유: config 는 무의존(다른 에이전트·CLI 가 가볍게 import)이어야 한다.
#: 같은 값임은 ``tests/test_live_strategy.py`` 가 못박는다.
LIMIT_UP_REL_TOL = 0.007
#: 본전 이동(``be_r``) 손절선 = 진입가 × (1 + BE_COST) — 설계 "진입가(+비용 0.5%)". 연구 커널(``exits_v2_fast.BE_COST``)과 같은 값(테스트).
BE_COST = 0.005

#: GA2 유전자(``research/experiments/ga2_run.py`` G2_CONT + G2_DISC 순서) → :class:`StrategyConfig` 필드. 19개.
GENE_FIELDS: dict[str, str] = {
    "near_high": "near_high", "above_low": "above_low", "rs_min": "rs_min", "vol_mult": "vol_mult",
    "breadth_min": "breadth_min", "pead_min": "pead_rank_min", "rel_max": "rel_max", "eps_yoy_min": "eps_yoy_min",
    "rev_yoy_min": "rev_yoy_min", "eps_accel": "eps_accel", "ma_stack": "ma_stack",
    "ma200_slope_days": "ma200_slope_days", "pivot_window": "pivot_window", "chase": "chase", "stop_cap": "stop_cap",
    "take_r": "take_r", "trail_ma": "trail_ma", "failed_breakout": "failed_breakout", "max_hold": "max_hold",
}
#: 수급 시대 유전자(ga2_run FLOW_CONT + FLOW_DISC) → 필드. 주면 ``require_flow=False`` 로 바꾸고 이 조건을 쓴다.
FLOW_GENE_FIELDS: dict[str, str] = {"smart_min": "smart_min", "foreign_pos": "foreign_pos", "persist_min": "persist_min"}
#: 이산 유전자의 허용 단계(ga2_run G2_DISC·FLOW_DISC 그대로).
GENE_LEVELS: dict[str, tuple] = {
    "eps_accel": (0, 1), "ma_stack": (0, 1, 2, 3), "ma200_slope_days": (0, 10, 20, 40),
    "pivot_window": (0, 20, 30, 40, 50, 60), "chase": (0.02, 0.05, 0.10, math.inf),
    "stop_cap": (0.06, 0.08, 0.10, 0.12, 0.15), "take_r": (1.5, 2.0, 3.0, 4.0, math.inf),
    "trail_ma": (10, 20, 50, 0), "failed_breakout": (0, 1), "max_hold": (20, 60, 120),
    "foreign_pos": (0, 1), "persist_min": (0, 8, 10, 12),
}
#: 청산 v2 유전자(MINERVINI_EXITS_V2.md) → 필드. ``strength_sell`` 은 단계 코드 → (``strength_ret``, ``strength_ma_ext``) 두 필드.
EXIT_V2_GENE_FIELDS: dict[str, str] = {"be_r": "be_r", "take_frac": "partial_frac", "strength_sell": "strength_sell",
                                       "fb_tol": "fb_tol", "fb_days": "fb_days"}
#: 청산 v2 단계(설계 표 그대로). ``take_r`` 은 v2 유전자가 있을 때 1.0 도 허용한다.
EXIT_V2_LEVELS: dict[str, tuple] = {
    "be_r": (0.5, 1.0, 1.5, 2.0, math.inf), "take_frac": (1 / 3, 1 / 2), "strength_sell": (0, 1, 2, 3),
    "fb_tol": (0.0, 0.02, 0.05), "fb_days": (1, 2), "take_r": (1.0, 1.5, 2.0, 3.0, 4.0, math.inf),
}
#: 강세 매도 단계 → (하루 상승률 x, MA50 이격 y). None = 그 조건 없음. 0 = 끔.
STRENGTH_SELL: dict[int, tuple] = {0: (None, None), 1: (0.15, None), 2: (None, 0.30), 3: (0.15, 0.30)}
#: 연속 유전자의 범위(ga2_run G2_CONT·FLOW_CONT) — 범위 밖이면 거부.
GENE_RANGES: dict[str, tuple[float, float]] = {
    "near_high": (0.50, 0.95), "above_low": (1.00, 1.60), "rs_min": (0.0, 95.0), "vol_mult": (0.0, 2.5),
    "breadth_min": (0.0, 0.7), "pead_min": (0.0, 0.95), "rel_max": (0.5, 4.0), "eps_yoy_min": (0.0, 1.0),
    "rev_yoy_min": (-0.2, 0.5), "smart_min": (-0.10, 0.20),
}


def _num(name: str, v):
    """유전자 JSON 값 → 숫자. ``"inf"`` 문자열은 ∞. None 은 그대로."""
    if v is None:
        return None
    if isinstance(v, str):
        if v.strip().lower() in ("inf", "∞", "infinity"):
            return math.inf
        raise ValueError(f"유전자 {name}: 숫자가 아니다 {v!r}")
    if isinstance(v, bool):
        return int(v)
    return v


@dataclass(frozen=True)
class StrategyConfig:
    """전략 v0 수치 전부(설계 §2). 기본값 = 사전등록 A + 라이브 유니버스(ADV20 ≥ 50억).

    단위: 금액·거래대금은 백만원(``adv_floor``) — 가격은 원. 비율은 소수(0.05 = 5%).
    ``None`` 인 문턱은 **끔**(그 조건이 없고 NaN 도 통과) — GA2 마스크(``ga_fast._mask_ga2_py``)와 같은 의미.
    """

    # --- 유니버스(korea.universe_mask 위생 + ADV 하한) ---
    adv_floor: float = eok(50)          # ADV20 ≥ 5,000 백만원(50억) — 상한 없음(두 구간 합)
    adv_window: int = 20
    min_price: float = 1_000.0          # 원자료 종가(daily_bars.close) 기준
    min_history: int = 252

    # --- 추세 템플릿(GA2 4성분, 기본 = 원전 전체: close ≥ 0.75×252일 고가 ∧ ≥ 1.30×저가 ∧ MA 정렬 3 ∧ MA200 > 20일 전) ---
    near_high: "float | None" = 0.75    # close ≥ x × 252일 장중 고가(당일 포함). None = 끔
    above_low: "float | None" = 1.30    # close ≥ x × 252일 장중 저가. None = 끔
    ma_stack: int = 3                   # close>MA50, MA50>MA150, MA150>MA200 중 참인 개수 ≥ k. 0 = 끔
    ma200_slope_days: int = 20          # MA200 > MA200[t−k]. 0 = 끔 (단계 10/20/40)

    # --- 뼈대 S(sepa.PREREG_SKELETON) ---
    rs_min: "float | None" = 70.0       # RS 등급 ≥. None = 끔
    breadth_min: "float | None" = 0.5   # 그 종목 시장의 MA50 위 비율 >. None = 끔
    pivot_window: int = 50              # 피벗 = 직전 w 일 최고 고가, 돌파 = 종가 > 피벗. 0 = 돌파 요구 없음(피벗은 50일)
    vol_mult: "float | None" = 1.5      # 돌파일 거래량 ≥ x × 직전 w 일 평균(w = pivot_window, 0 이면 50). None = 끔

    # --- 확인 층 ---
    pead_rank_min: "float | None" = 0.70   # PEAD 순위 ≥ (모집단 = 그날 위생 ∧ ADV20 ≥ adv_floor). None = 끔
    rel_max: "float | None" = 1.0       # 상대 수렴 rel ≤. None = 끔
    require_flow: bool = True           # 수급 F = 1 (smart > 0 ∧ foreign > 0). 수급 유전자를 주면 False 로 바꾼다
    smart_min: "float | None" = None    # smart_ratio_20 ≥ (GA 수급 유전자). None = 안 씀
    foreign_pos: "bool | None" = None   # foreign_ratio_20 > 0 요구. None = 안 씀
    persist_min: int = 0                # smart_pos_days_20 ≥. 0 = 안 씀

    # --- 상대 수렴 rel(CONVERGENCE.md 정의) ---
    rel_window: int = 10                # 분자: t−10..t−1 고저폭
    rel_med_window: int = 250           # 분모: 그 값의 250일 이동 중앙값
    rel_min_periods: int = 120
    rel_gap: int = 11                   # 분모 창 끝 = t−11

    # --- 실적 문(GA2.md 유전자 — 전부 기본 끔 = 현 동작 그대로) ---
    #: 최신 분기 **주당** 순이익 YoY 하한 — GA 커널 정의: 순이익/주식수(분기 말 as-of, 분할 조정) 를 ``earnings_leg_panel`` 에 넣어
    #: 계산(:func:`swing_it.live.features.eps_adjusted_earnings`). 주식수가 없는 분기는 순이익 YoY 로 대체(``eps_from_netinc``).
    eps_yoy_min: "float | None" = None
    eps_accel: bool = False             # 가속: 이번 분기 EPS YoY > 직전 분기 EPS YoY(둘 다 유한)
    rev_yoy_min: "float | None" = None  # 매출 YoY 하한

    # --- 뉴스 거부권(설계 §7, NEWS_VETO.md) ---
    #: 신호 작업에서 후보의 최근 20거래일 악재(5일 ≥ 1 또는 20일 ≥ 2)를 거부한다(:mod:`swing_it.live.news_veto`). 모름은 거부 안 함.
    #: 백테스트는 ``run_backtest(veto=패널)`` 로 과거 원장(2025-09~)에서만 적용한다. GA 는 거부권 없이 골랐다 — 인계 때 그대로 둔다.
    news_veto: bool = True

    # --- 진입 체결(sepa 규약) ---
    chase: float = 0.05                 # 지정가 상한 = 피벗 × (1+chase) (= 백테스트 추격 금지). inf = 추격 금지 없음
    limit_up_ratio: float = 1.30 - LIMIT_UP_REL_TOL   # 분할조정 패널의 상한가 시가 판정(sepa.LIMIT_UP_RATIO)

    # --- 청산(minervini_exits.PREREG_MINERVINI + GA2 청산 유전자) ---
    contraction_window: int = 10        # 수축 저점 = min(low[t−10..t]) (돌파일 포함 11일)
    stop_mult: float = 0.99             # 손절 기준 = 수축 저점 × 0.99
    stop_floor: float = 0.03            # 손절폭 하한
    stop_cap: float = 0.10              # 손절폭 상한
    take_r: float = 2.0                 # 절반 익절 = 진입가 × (1 + take_r·d). inf = 익절 없음(추적은 전량·1일째부터)
    partial_frac: float = 0.5           # 익절 비율(보유 수량 대비) — v2 유전자 take_frac
    trail_ma: int = 50                  # 추적 MA(나머지 절반, 익절 없음이면 전량). 0 = 추적 없음
    #: 본전 이동(v2): 고가 ≥ 진입가 × (1 + be_r·d) 인 날의 **다음 거래일부터** 손절선 = 진입가 × (1 + BE_COST). inf = 끔(현 동작 —
    #: 익절 뒤 본전 = 진입가만).
    be_r: float = math.inf
    #: 강세 매도(v2, 한 트레이드에 한 번): 종가 ≥ 전일 종가 × (1 + strength_ret) 또는 종가 ≥ MA50 × (1 + strength_ma_ext) →
    #: 다음 거래일 시가에 남은 수량의 절반. None = 그 조건 없음(둘 다 None = 끔).
    strength_ret: "float | None" = None
    strength_ma_ext: "float | None" = None
    #: 돌파 실패 피벗 허용폭(v2): 종가 < 피벗 × (1 − fb_tol). 0 = 현 동작.
    fb_tol: float = 0.0
    #: 돌파 실패 연속 이탈일(v2): 그 조건이 거래일 fb_days 번 연속이면 다음 거래일 시가. 1 = 현 동작.
    fb_days: int = 1
    #: 돌파 실패(설계 §6, 2026-10-10 사용자 지시 — 미너비니 원전): 절반 익절 전 어느 날이든 **종가 < 피벗**(돌파 기준가 =
    #: 직전 pivot_window 일 최고 고가, 계획의 ``pivot``) 이면 다음 거래일 시가에 전량 청산(MA50 추적과 같은 경로: d 종가 결정 → d+1 시가).
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
    #: 위험 항은 stop_cap 을 올리거나 risk 를 낮추는 변형에서만 묶인다. GA 포트폴리오 적합도는 종목당 10% 고정이라
    #: :meth:`from_genes` 는 ``risk_per_trade = max_position_frac`` 로 둬서(위험 항이 절대 묶이지 않게) 사이징을 맞춘다.
    risk_per_trade: float = 0.01        # 위험 = 자산 × 1% (현 수치에선 묶이지 않는다)
    max_position_frac: float = 0.10     # 종목당 자산 10% 상한 (실질 사이징)
    max_positions: int = 10             # 최대 동시 보유 종목

    def to_dict(self) -> dict:
        return asdict(self)

    # --- GA2 유전자 ↔ 설정 ---
    @property
    def partial_on(self) -> bool:
        return math.isfinite(self.take_r)

    @property
    def has_trail(self) -> bool:
        return self.trail_ma > 0

    @property
    def strength_on(self) -> bool:
        return self.strength_ret is not None or self.strength_ma_ext is not None

    @property
    def exits_v2_default(self) -> bool:
        """청산 v2 필드가 전부 현 동작 값인가(그러면 :meth:`to_genes` 가 v2 유전자를 넣지 않는다)."""
        return (math.isinf(self.be_r) and self.partial_frac == 0.5 and not self.strength_on and self.fb_tol == 0.0
                and self.fb_days == 1)

    @property
    def effective_pivot_window(self) -> int:
        """피벗·거래량 평균 창 — 돌파 요구가 없으면(0) 50일(GA 커널 ``v_col = 3``·``w_idx = W50`` 과 같다)."""
        return self.pivot_window if self.pivot_window > 0 else 50

    @classmethod
    def from_genes(cls, genes: dict, *, base: "StrategyConfig | None" = None) -> "StrategyConfig":
        """GA2 유전자 dict(``live_candidate.json`` 의 ``genes``/``config``) → 설정. **모르는 유전자·허용 밖 단계는 예외**(조용히 무시 금지).

        19개 유전자 전부 있어야 한다. 수급 유전자 3개는 선택(하나라도 있으면 셋 다 있어야 하고 ``require_flow=False``).
        ``base`` 위에 덮는다(기본 :data:`DEFAULT_STRATEGY` — 유니버스·rel 창·수축 창·사이징·거부권은 유전자 밖 고정값).
        사이징: ``risk_per_trade = max_position_frac`` (GA 포트폴리오 적합도의 슬롯 10% 고정과 같게 — 위험 항이 안 묶인다).
        """
        base = DEFAULT_STRATEGY if base is None else base
        unknown = set(genes) - set(GENE_FIELDS) - set(FLOW_GENE_FIELDS) - set(EXIT_V2_GENE_FIELDS)
        if unknown:
            raise ValueError(f"대응 없는 유전자: {sorted(unknown)}")
        missing = set(GENE_FIELDS) - set(genes)
        if missing:
            raise ValueError(f"빠진 유전자: {sorted(missing)}")
        flow_given = set(genes) & set(FLOW_GENE_FIELDS)
        if flow_given and flow_given != set(FLOW_GENE_FIELDS):
            raise ValueError(f"수급 유전자는 셋 다 있어야 한다: 있음 {sorted(flow_given)}")
        v2_given = set(genes) & set(EXIT_V2_GENE_FIELDS)
        if v2_given and v2_given != set(EXIT_V2_GENE_FIELDS):
            raise ValueError(f"청산 v2 유전자는 다섯 개 다 있어야 한다: 있음 {sorted(v2_given)}")
        levels = {**GENE_LEVELS, **(EXIT_V2_LEVELS if v2_given else {})}
        kw: dict = {}
        for gene, fld in {**GENE_FIELDS, **FLOW_GENE_FIELDS, **EXIT_V2_GENE_FIELDS}.items():
            if gene not in genes:
                continue
            v = _num(gene, genes[gene])
            if gene in levels:
                if v is None or v not in levels[gene]:
                    raise ValueError(f"유전자 {gene}: 허용 단계 {levels[gene]} 밖 {genes[gene]!r}")
                if gene == "strength_sell":
                    kw["strength_ret"], kw["strength_ma_ext"] = STRENGTH_SELL[int(v)]
                    continue
                if gene == "fb_days":
                    kw[fld] = int(v)
                    continue
                if gene in ("eps_accel", "failed_breakout", "foreign_pos"):
                    v = bool(v)
                elif gene in ("ma_stack", "ma200_slope_days", "pivot_window", "trail_ma", "max_hold", "persist_min"):
                    v = int(v)
                else:
                    v = float(v)
            elif gene in GENE_RANGES:
                if v is not None:
                    lo, hi = GENE_RANGES[gene]
                    if not (lo - 1e-9 <= float(v) <= hi + 1e-9):
                        raise ValueError(f"유전자 {gene}: 범위 [{lo}, {hi}] 밖 {v!r}")
                    v = float(v)
            else:
                raise ValueError(f"유전자 {gene}: 단계·범위 정의가 없다")
            kw[fld] = v
        if flow_given:
            kw["require_flow"] = False
        kw["risk_per_trade"] = base.max_position_frac
        from dataclasses import replace  # noqa: PLC0415

        return replace(base, **kw)

    def to_genes(self, *, flow: "bool | None" = None) -> dict:
        """설정 → GA2 유전자 dict(:meth:`from_genes` 의 역 — 왕복 동일). ``flow`` None 이면 수급 유전자가 설정돼 있을 때만 넣는다.

        청산 v2 유전자는 v2 필드가 현 동작 값이 아닐 때만 넣는다(강세 매도 조합이 :data:`STRENGTH_SELL` 단계가 아니면 예외).
        """
        out: dict = {}
        for gene, fld in GENE_FIELDS.items():
            v = getattr(self, fld)
            out[gene] = int(v) if gene in ("eps_accel", "failed_breakout") else v
        use_flow = (self.smart_min is not None or self.foreign_pos is not None or self.persist_min) if flow is None \
            else flow
        if use_flow:
            out["smart_min"] = self.smart_min
            out["foreign_pos"] = int(bool(self.foreign_pos))
            out["persist_min"] = int(self.persist_min)
        if not self.exits_v2_default:
            code = [k for k, v in STRENGTH_SELL.items() if v == (self.strength_ret, self.strength_ma_ext)]
            if not code:
                raise ValueError(f"강세 매도 ({self.strength_ret}, {self.strength_ma_ext}) 는 유전자 단계가 아니다")
            out.update({"be_r": self.be_r, "take_frac": self.partial_frac, "strength_sell": code[0],
                        "fb_tol": self.fb_tol, "fb_days": int(self.fb_days)})
        return out

    def describe_genes(self) -> list[tuple[str, str, object]]:
        """(유전자, 필드, 값) 표 — 리포트용."""
        rows = [(g, f, getattr(self, f)) for g, f in {**GENE_FIELDS, **FLOW_GENE_FIELDS}.items()]
        return rows + [("strength_sell", "strength_ret·strength_ma_ext", (self.strength_ret, self.strength_ma_ext))
                       if g == "strength_sell" else (g, f, getattr(self, f)) for g, f in EXIT_V2_GENE_FIELDS.items()]


FIELD_NAMES = tuple(f.name for f in fields(StrategyConfig))


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


# ---------------------------------------------------------------------------
# 포지션별 설정 고정(라이브 런타임 · 통합 검토 M3 재검증)
# ---------------------------------------------------------------------------
#
# 라이브는 열린 포지션마다 **진입 시점의 StrategyConfig 전체 dict** 를 원장에 고정하고, 청산 판단 때 그 dict 로 설정을 다시 만든다
# (``swing_it.live.runtime.SwingRuntime._cfg_of``). 나중에 필드를 **추가**하면 옛 dict 엔 그 키가 없다 — 그때 현재 코드 기본값을 넣으면
# 기본값이 "켜짐" 인 새 규칙이 열린 포지션에 소급 적용된다. 그래서:
#
# 1. :data:`PIN_BASELINE_FIELDS` — 고정 기능이 들어온 날(2026-10-10)의 필드 이름. **손으로 적은 상수**다(자동으로 늘지 않는다).
# 2. 그 뒤에 추가하는 필드는 반드시 :data:`LEGACY_DEFAULTS` 에 "그 필드가 없던 때의 동작을 내는 값"을 적는다 — 보통 규칙을 끄는 값.
#    고정 dict 에 없는 필드는 이 값으로 채운다. 적지 않으면 :func:`pinned_config` 가 예외를 낸다(조용히 기본값을 넣지 않는다).
#    ``tests/test_live_runtime.py::test_every_field_after_pin_baseline_has_legacy_default`` 가 이 규칙을 못박는다.
# 3. 필드를 **지우면** 고정 dict 에 남은 키는 버리고 경고 목록으로 돌려준다(동작이 바뀌었다는 뜻 — 런타임이 경보한다).

PIN_BASELINE_FIELDS: frozenset[str] = frozenset((
    "adv_floor", "adv_window", "min_price", "min_history", "near_high", "above_low", "ma_stack", "ma200_slope_days",
    "rs_min", "breadth_min", "pivot_window", "vol_mult", "pead_rank_min", "rel_max", "require_flow", "smart_min",
    "foreign_pos", "persist_min", "rel_window", "rel_med_window", "rel_min_periods", "rel_gap", "eps_yoy_min",
    "eps_accel", "rev_yoy_min", "news_veto", "chase", "limit_up_ratio", "contraction_window", "stop_mult", "stop_floor",
    "stop_cap", "take_r", "partial_frac", "trail_ma", "be_r", "strength_ret", "strength_ma_ext", "fb_tol", "fb_days",
    "failed_breakout", "time_stop_days", "max_hold", "stop_slippage_bps", "risk_per_trade", "max_position_frac",
    "max_positions",
))

#: 필드 이름 → 그 필드가 **없던 시절의 동작**을 내는 값(위 2). 고정 기능 이후 추가된 필드만 여기 온다.
LEGACY_DEFAULTS: dict[str, object] = {}


def pinned_config(pinned: dict, *, legacy: "dict | None" = None) -> "tuple[StrategyConfig, list[str]]":
    """고정 dict → ``(StrategyConfig, 경고 목록)``. 없는 필드는 :data:`LEGACY_DEFAULTS` 로(없으면 ``ValueError``), 지워진 필드는 버린다.

    ``"inf"`` 문자열은 ``math.inf`` 로 읽는다(JSON 직렬화 규약). **현재 코드 기본값은 절대 쓰지 않는다** — 고정의 뜻이 그것이다.
    """
    import math  # noqa: PLC0415
    from dataclasses import fields as _fields  # noqa: PLC0415

    legacy = LEGACY_DEFAULTS if legacy is None else legacy
    names = [f.name for f in _fields(StrategyConfig)]
    warnings: list[str] = []
    dropped = sorted(set(pinned) - set(names))
    if dropped:
        warnings.append(f"고정 설정의 필드가 코드에서 사라졌다(버림): {dropped}")
    kw: dict = {}
    missing_no_legacy = []
    for n in names:
        if n in pinned:
            v = pinned[n]
        elif n in legacy:
            v = legacy[n]
            warnings.append(f"고정 설정에 없는 필드 {n} ← LEGACY_DEFAULTS {v!r}")
        else:
            missing_no_legacy.append(n)
            continue
        if isinstance(v, str) and v.strip().lower() in ("inf", "infinity", "+inf"):
            v = math.inf
        kw[n] = v
    if missing_no_legacy:
        raise ValueError(f"고정 설정에 없는 필드 {missing_no_legacy} 의 LEGACY_DEFAULTS 가 없다 — config.py 에 적을 것"
                         "(현재 기본값으로 채우면 열린 포지션 규칙이 소급해 바뀐다)")
    return StrategyConfig(**kw), warnings
