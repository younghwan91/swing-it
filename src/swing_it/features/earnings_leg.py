"""실적 다리(E) 패널 — SEPA 합류 설계의 오닐 C 축이자 확신 층의 PEAD 축.

설계: ``docs/superpowers/specs/2026-10-09-sepa-confluence.md`` §3.

- **E 실적 다리:** 그날 알려진 최신 분기(정정 반영 이중 시간축) 순이익 YoY ≥ +25% ∧ 당기 순이익 > 0
  ∧ 공시(``avail_date``) 후 ≤ 90일.
- **가속(확신 층):** 최신 분기 YoY > 직전 분기 YoY (둘 다 유한).
- **PEAD 랭크:** 그날 YoY 를 가진 종목 사이의 횡단면 백분위 — 검증된 신호의 랭킹 그대로.

**YoY 정의는 검증된 PEAD 와 같다**(``research/experiments/pead_refinement.load_data`` — ``pead_gate`` 가 그걸
쓴다): ``yoy = (netinc - netinc_prior) / |netinc_prior|`` (:func:`swing_it.features.fundamentals._yoy_vec`),
``netinc_prior`` 가 0·결측이면 NaN 이고 **NaN 인 행은 패널에 들어가기 전에 버린다**(``dropna(subset=["yoy"])``).
그래서 최신 분기 YoY 가 계산 불가면 *그 전 분기*가 "최신" 으로 남는다(그만큼 ``age`` 가 커져 E 의 90일 문에서
걸러진다). 전년 동기 적자(``prior < 0``)도 버리지 않는다 — 분모가 ``|prior|`` 라 적자 축소·흑자 전환은 양의
YoY 다(예: −100 → +10 은 +1.1, −100 → −50 은 +0.5). 적자 지속 쪽은 E 의 "당기 순이익 > 0" 이 거른다.

**시점 안전(이중 시간축, GUARDRAILS §4-3):** 어느 분기가 공시됐나(``avail_date``)와 그 분기의 어느 버전을
그때 알고 있었나(``knowledge_date``)는 독립인 축이다. 한 버전은 ``max(avail, knowledge)`` 부터 보이고,
날짜 ``d`` 에서 분기마다 그때까지 보인 버전 중 ``knowledge_date`` 가 가장 늦은 것을 쓴다. "최신 분기" 는
보이는 분기 중 가장 늦은 분기다 — 옛 분기의 정정본이 최신 분기를 밀어내지 않는다. ``age`` 는 *정정일이 아니라
그 분기의 공시일*부터 잰다(정정으로 신선도가 리셋되면 드리프트 창이 왜곡된다). 이 규약은
:func:`swing_it.features.fundamentals.earnings_yoy_panel` (``knowledge_col``) 과 같고, 그 함수와 ``yoy``·
``age_days`` 가 일치함을 ``tests/test_earnings_leg.py`` 가 못박는다.

**왜 ``earnings_yoy_panel`` 을 그대로 부르지 않나.** 그 함수는 값 하나만 실어 나른다 — 매출·영업이익 YoY·
당기 순이익을 *같은 행*에서, 직전 분기 YoY 를 *그날 알던 버전*으로 함께 붙이려면 상태를 행 단위로 들고 가야
한다. 또 종목별 ``merge_asof`` 루프는 3,000종목 × 정정 구간 수만큼 돈다. 여기서는 "상태가 바뀌는 순간"
(버전이 보이기 시작하는 날)만 계산하고 거래일로 펼친다 — 같은 규약을 한 번의 벡터 연산으로.

**Q4 는 연간이다(데이터 규약).** ``earnings`` 의 Q1~Q3 은 DART 분/반기보고서 ``thstrm_amount`` (3개월),
``Q4`` 는 사업보고서의 **연간** 금액이다(연간 − 9M 으로 Q4 단독을 만들지 않는다). 검증된 PEAD 도 이 행을
그대로 썼으므로 여기서도 그대로 둔다. 그래서 4~5월 초(최신 = Q4)의 YoY 는 연간 YoY 이고, Q1 공시 직후의
가속 비교는 "Q1 3개월 YoY vs 전년 연간 YoY" 다 — 해석할 때 주의.

**avail_date 는 실제 공시일이 아니다.** 수집기가 기말 + 45일(분기)·+90일(연간)로 찍는 보수적 상한이라
같은 분기의 모든 종목이 같은 ``avail_date`` 를 갖는다(5/15·8/14·11/14·3/31). 따라서 ``age`` 도 날짜마다
전 종목이 거의 같고, "공시 후 ≤ 90일" 은 사실상 달력 문이다 — Q3(11/14) 이후 90일인 2/12 를 넘기면 Q4(3/31)
가 보일 때까지 E 가 전 종목에서 꺼진다.

Pure DataFrame in → DataFrame out. DB 는 호출부가 ``storage.read_earnings(con, all_versions=True)`` 로 읽는다
(접어 읽으면 최신 정정본이 과거 날짜에 들어가 조용한 룩어헤드가 된다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import _yoy_vec

#: E 실적 다리 — 최신 분기 순이익 YoY 하한(오닐 C: +25%).
E_MIN_YOY = 0.25
#: E 실적 다리 — 공시(``avail_date``) 후 최대 달력일.
E_MAX_AGE_DAYS = 90

#: :func:`earnings_leg_panel` 출력 열(순서 고정).
LEG_COLS = ("code", "date", "period", "yoy", "yoy_prev", "rev_yoy", "op_yoy",
            "netinc", "netinc_pos", "age_days", "age_tdays")


def _to_dt(s: pd.Series) -> pd.Series:
    """``YYYYMMDD``·``YYYY-MM-DD``·date 를 모두 datetime 으로."""
    return pd.to_datetime(s.astype(str).str.replace("-", "").str[:8], format="%Y%m%d", errors="coerce")


def period_ordinal(period: pd.Series) -> pd.Series:
    """``"2024Q3"`` → ``2024*4 + 2``. 연속 분기는 정확히 1 차이(2023Q4 → 2024Q1 도 1)."""
    p = period.astype(str)
    return p.str[:4].astype(int) * 4 + p.str[-1].astype(int) - 1


def _prepare(earnings: pd.DataFrame, avail_col: str, knowledge_col: str | None) -> pd.DataFrame:
    """버전 행을 정규화하고 YoY 를 붙인다 — PEAD 규칙대로 순이익 YoY 가 NaN 인 행은 버린다."""
    ea = earnings.copy()
    ea["code"] = ea["code"].astype(str)
    ea["_avail"] = _to_dt(ea[avail_col])
    ea["_know"] = _to_dt(ea[knowledge_col]) if knowledge_col is not None else ea["_avail"]
    ea["yoy"] = _yoy_vec(ea["netinc"], ea["netinc_prior"])
    for out, cur, prior in (("rev_yoy", "revenue", "revenue_prior"),
                            ("op_yoy", "op_income", "op_income_prior")):
        ea[out] = _yoy_vec(ea[cur], ea[prior]) if {cur, prior} <= set(ea.columns) else np.nan
    # 검증된 PEAD 와 같은 자리에서 같은 기준으로 버린다(pead_refinement.load_data 의 dropna(subset=["yoy"])).
    ea = ea.dropna(subset=["_avail", "_know", "yoy"])
    # 이 버전을 실제로 쓸 수 있게 된 날. 늦게 수집된 공시(_know > _avail)도 여기서 걸린다.
    ea["_valid_from"] = ea[["_avail", "_know"]].max(axis=1)
    ea["_pord"] = period_ordinal(ea["period"])
    # 같은 (종목, 분기) 에서 같은 날 보이기 시작하는 버전이 여럿이면 가장 늦게 안 것 하나만.
    ea = (ea.sort_values(["code", "_pord", "_valid_from", "_know"])
            .drop_duplicates(subset=["code", "_pord", "_valid_from"], keep="last"))
    return ea[["code", "period", "_pord", "_avail", "_valid_from", "netinc",
               "yoy", "rev_yoy", "op_yoy"]].reset_index(drop=True)


def _states_at_events(ea: pd.DataFrame) -> pd.DataFrame:
    """종목별 "상태가 바뀌는 순간"(어떤 버전이 보이기 시작하는 날)마다 최신·직전 분기 상태."""
    ea = ea.sort_values(["code", "_valid_from", "_pord"]).reset_index(drop=True)
    # 그 순간까지 보인 분기 중 최신 — 정정본이 나중에 보여도 옛 분기면 최신을 바꾸지 못한다.
    ea["_latest"] = ea.groupby("code", sort=False)["_pord"].cummax()
    ev = (ea.drop_duplicates(subset=["code", "_valid_from"], keep="last")
            [["code", "_valid_from", "_latest"]].rename(columns={"_valid_from": "t"}))
    ev["_prev"] = ev["_latest"] - 1

    right = ea[["code", "_pord", "_valid_from", "period", "_avail", "netinc",
                "yoy", "rev_yoy", "op_yoy"]].sort_values("_valid_from")
    ev = ev.sort_values("t")
    # (종목, 분기) 의 버전을 그 순간 기준 as-of 로 고른다 — 이중 시간축의 버전 축.
    cur = pd.merge_asof(
        ev.rename(columns={"_latest": "_pord"}), right,
        left_on="t", right_on="_valid_from", by=["code", "_pord"], direction="backward")
    prev = pd.merge_asof(
        ev.rename(columns={"_prev": "_pord"})[["code", "_pord", "t"]],
        right[["code", "_pord", "_valid_from", "yoy"]].rename(columns={"yoy": "yoy_prev"}),
        left_on="t", right_on="_valid_from", by=["code", "_pord"], direction="backward")
    cur["yoy_prev"] = prev["yoy_prev"].to_numpy()  # merge_asof 는 왼쪽 순서를 지킨다 — 둘 다 ev 순서
    return cur.sort_values(["code", "t"]).reset_index(drop=True)


def earnings_leg_panel(
    earnings: pd.DataFrame,
    trading_dates: list[str],
    *,
    avail_col: str = "avail_date",
    knowledge_col: str | None = "knowledge_date",
) -> pd.DataFrame:
    """시점 안전 (code, date) 롱 패널 — 그날 알던 최신 분기의 실적 상태.

    Args:
        earnings: ``read_earnings(con, all_versions=True)`` 모양 — ``code``·``period``(``YYYYQn``)·
            ``avail_col``·``netinc``·``netinc_prior``, 있으면 ``revenue``·``revenue_prior``·``op_income``·
            ``op_income_prior``, 그리고 ``knowledge_col``.
        trading_dates: 거래일(``YYYY-MM-DD``). 보통 ``sorted(prices["date"].unique())``.
        avail_col: 공시(가용)일 열.
        knowledge_col: 정정공시 버전 열. ``None`` 이면 입력에 버전이 하나뿐이고 공시일에 알았다고 본다.

    Returns:
        :data:`LEG_COLS` 열의 롱 프레임. 종목의 첫 공시가 보이기 전 날짜는 **행이 없다**(= 모름; 3,000종목
        × 2,500일을 NaN 으로 채우지 않는다). 열:

        - ``period``: 최신 분기(순이익 YoY 가 계산되는 것 중). ``yoy``: 그 분기 순이익 YoY.
        - ``yoy_prev``: *바로 앞 분기*(``period`` − 1)의 순이익 YoY, 그날 알던 버전. 앞 분기가 없거나
          YoY 계산 불가면 NaN(더 옛 분기로 건너뛰지 않는다).
        - ``rev_yoy``·``op_yoy``: 같은 행의 매출·영업이익 YoY(같은 공식, 계산 불가면 NaN).
        - ``netinc``·``netinc_pos``: 같은 행의 당기 순이익과 ``> 0`` 여부.
        - ``age_days``: ``date − avail_date`` 달력일. ``age_tdays``: ``avail_date`` 당일(휴일이면 다음
          거래일)을 0 으로 센 거래일 수.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(sorted(trading_dates))))
    ea = _prepare(earnings, avail_col, knowledge_col)
    if ea.empty or len(dates) == 0:
        return pd.DataFrame(columns=list(LEG_COLS))
    ev = _states_at_events(ea)

    d64 = dates.to_numpy()
    start = np.searchsorted(d64, ev["t"].to_numpy(), side="left")
    # 다음 상태가 시작되기 전까지가 이 상태의 구간. 거래일 사이에 상태가 둘 바뀌면 앞의 것은 길이 0.
    nxt = np.append(start[1:], len(d64))
    last_of_code = np.append(ev["code"].to_numpy()[1:] != ev["code"].to_numpy()[:-1], True)
    end = np.where(last_of_code, len(d64), nxt)
    count = np.maximum(end - start, 0)

    rep = np.repeat(np.arange(len(ev)), count)
    offs = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    di = np.repeat(start, count) + offs

    out = ev.iloc[rep][["code", "period", "yoy", "yoy_prev", "rev_yoy", "op_yoy", "netinc"]]
    out = out.reset_index(drop=True)
    avail = ev["_avail"].to_numpy()[rep]
    out.insert(1, "date", np.asarray(dates.strftime("%Y-%m-%d"))[di])
    out["netinc_pos"] = out["netinc"] > 0
    out["age_days"] = ((d64[di] - avail) // np.timedelta64(1, "D")).astype(int)
    out["age_tdays"] = di - np.searchsorted(d64, avail, side="left")
    return out[list(LEG_COLS)]


def e_flag(panel: pd.DataFrame, *, min_yoy: float = E_MIN_YOY,
           max_age_days: int = E_MAX_AGE_DAYS) -> pd.Series:
    """E 실적 다리: ``yoy ≥ min_yoy`` ∧ ``netinc > 0`` ∧ ``age_days ≤ max_age_days``. NaN 은 False."""
    return (panel["yoy"] >= min_yoy) & panel["netinc_pos"].astype(bool) & (panel["age_days"] <= max_age_days)


def accel_flag(panel: pd.DataFrame) -> pd.Series:
    """가속: 최신 분기 YoY > 직전 분기 YoY, 둘 다 유한할 때만 True(NaN 비교는 False)."""
    y, p = panel["yoy"], panel["yoy_prev"]
    return np.isfinite(y) & np.isfinite(p) & (y > p)


def pead_rank(panel: pd.DataFrame, *, eligible: pd.Series | None = None) -> pd.Series:
    """날짜별 YoY 횡단면 백분위(0, 1] — 값을 가진 종목 사이에서만(동점은 평균 순위).

    Args:
        panel: :func:`earnings_leg_panel` 결과.
        eligible: ``panel`` 과 같은 인덱스의 bool. 주면 그 행만 순위에 넣는다(예: 유동성 문을 통과한
            유니버스 — 검증된 PEAD 는 ADV 문 안에서 순위를 매겼다). 나머지 행은 NaN.
    """
    y = panel["yoy"] if eligible is None else panel["yoy"].where(eligible.astype(bool))
    return y.groupby(panel["date"]).rank(pct=True)


def add_earnings_flags(panel: pd.DataFrame, *, eligible: pd.Series | None = None) -> pd.DataFrame:
    """``e_flag``·``accel``·``pead_rank`` 열을 붙인 사본(설계서 고정 수치)."""
    return panel.assign(e_flag=e_flag(panel), accel=accel_flag(panel),
                        pead_rank=pead_rank(panel, eligible=eligible))
