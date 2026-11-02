"""단계 경계 검사 — 어느 하네스든 부를 수 있는 독립 함수. 어기면 :class:`CheckError` 다.

실제 러너는 배열과 프레임을 그대로 쓰므로 계약 객체로 감싸지 않는다. 합격 여부는 묻지 않는다 — 형태와 시점만 본다(GUARDRAILS §8).
검사하는 것만 적었다. **알게 된 날 규칙은 여기서 보증하지 않는다** — 신호를 :mod:`swing_it.features.earnings_asof` 로 만든다.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

#: 2026-09-14~ 일봉 고가·저가·종가·거래량에 애프터마켓 체결이 섞였다(레포 CLAUDE.md "데이터 함정").
#: main 의 ``read_prices`` 에는 정규장 경로가 없어 이 구간을 받지 않는다.
AFTERMARKET_FROM = "2026-09-14"


class CheckError(ValueError):
    """단계 입출력이 약속한 모양이 아니다."""


def _long_frame(frame: Any, cols: Sequence[str], name: str, *, key: Sequence[str] = ("code", "date")) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise CheckError(f"{name} 은 DataFrame 이어야 한다: {type(frame).__name__}")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise CheckError(f"{name} 에 열이 없다: {missing}")
    if frame.empty:
        raise CheckError(f"{name} 이 비었다 — 빈 응답을 성공으로 넘기지 않는다")
    if key and frame.duplicated(list(key)).any():
        raise CheckError(f"{name} 의 {tuple(key)} 가 중복이다 — 정정 버전이 새어 나왔을 수 있다")


def _day(values: pd.Series, name: str) -> pd.Series:
    days = pd.to_datetime(values.astype(str), errors="coerce", format="mixed")
    if days.isna().any():
        raise CheckError(f"{name} 에 날짜가 아닌 값·결측이 있다")
    return days


def check_price_end(end: Any) -> pd.Timestamp:
    """가격 상한이 날짜이고 애프터마켓 오염 구간 전인지 본다. 정문 ``read_prices`` 와 :func:`check_panels` 가 같이 쓴다."""
    end_ts = _day(pd.Series([end]), "end").iloc[0]
    if end_ts >= pd.Timestamp(AFTERMARKET_FROM):
        raise CheckError(f"end {end} 는 애프터마켓 오염 구간({AFTERMARKET_FROM}~)이다. "
                         "main 의 read_prices 는 정규장 값을 만들지 못한다 — 이 구간은 받지 않는다")
    return end_ts


def check_universe(frame: pd.DataFrame, *, min_per_day: int = 1) -> None:
    """1단계. 긴 형식 code·date·eligible(bool), (code, date) 하나씩, 날짜·코드가 유효하고, 날마다 적격이 ``min_per_day`` 개 이상이다.

    예열 구간처럼 의도한 빈 날이 있으면 ``min_per_day=0`` 으로 말한다. 그래도 전체에 적격이 하나도 없으면 거부한다.
    """
    _long_frame(frame, ("code", "date", "eligible"), "universe")
    if frame["eligible"].dtype != bool:
        raise CheckError(f"eligible 은 bool 이어야 한다: {frame['eligible'].dtype}")
    codes = frame["code"]
    if codes.isna().any() or (codes.astype(str).str.strip() == "").any():
        raise CheckError("universe 의 코드에 결측·빈 값이 있다")
    days = _day(frame["date"], "universe.date")
    if not frame["eligible"].any():
        raise CheckError("적격 종목이 하나도 없다")
    per_day = frame["eligible"].groupby(days).sum()
    short = per_day[per_day < min_per_day]
    if len(short):
        raise CheckError(f"적격이 {min_per_day}개 미만인 날이 {len(short)}일 있다(첫날 {short.index[0].date()}) — "
                         "의도한 빈 날이면 min_per_day 로 말한다")


def check_panels(prices: pd.DataFrame, earnings: pd.DataFrame, end: str, *, knowledge_col: str = "knowledge_date") -> None:
    """2단계. ``end`` 뒤 가격·``end`` 뒤에 안 실적 행, 양의 유한수가 아닌 종가를 거부한다. ``end`` 는 애프터마켓 오염 구간 전이어야 한다."""
    _long_frame(prices, ("code", "date", "close"), "prices")
    end_ts = check_price_end(end)
    close = prices["close"].to_numpy(dtype=np.float64)
    if not (np.isfinite(close) & (close > 0)).all():
        raise CheckError("prices.close 에 NaN·inf·0·음수가 있다 — 없는 날은 행이 없어야 한다")
    if _day(prices["date"], "prices.date").max() > end_ts:
        raise CheckError(f"prices 에 end({end}) 뒤 날짜가 있다 — 미래 데이터")
    _long_frame(earnings, ("code", "period", "avail_date", knowledge_col), "earnings", key=())
    for col in ("avail_date", knowledge_col):
        if _day(earnings[col], f"earnings.{col}").max() > end_ts:
            raise CheckError(f"earnings.{col} 에 end({end}) 뒤 날짜가 있다 — 그때 몰랐던 값")


def check_signal(sig: np.ndarray) -> None:
    """3단계. code × date 배열. NaN 은 그날 신호 없음이고, inf 와 전부 NaN 은 거부한다."""
    arr = np.asarray(sig, dtype=np.float64)
    if arr.ndim != 2:
        raise CheckError(f"신호는 code × date 2차원이어야 한다: {arr.shape}")
    if np.isinf(arr).any():
        raise CheckError("신호에 inf 가 있다")
    if not np.isfinite(arr).any():
        raise CheckError("유효한 신호가 하나도 없다")


def check_forecast(df: pd.DataFrame, horizon: int, calendar: Any) -> None:
    """5단계. 긴 형식 code·date·forecast·train_end. ``train_end`` = 그 예측을 만든 모델의 마지막 학습 행 날짜.

    ``horizon`` = 학습 행 날짜에서 **마지막 라벨일까지의 거래일 수**(ML: y = Σ_{s=t+2}^{t+21} 이면 21). ``date`` 는 ``train_end`` 에서 거래일로
    ``horizon`` 일 이상 떨어져야 그 라벨이 다 끝나 있다. ``calendar`` 는 거래일 전체(단조 증가)이고, 두 날짜 열은 그 안에 있어야 한다.
    평일로 세면 휴장일이 낀 겹침을 통과시킨다(실제 달력에서 겹침의 57.7%).
    """
    _long_frame(df, ("code", "date", "forecast", "train_end"), "forecast")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise CheckError(f"horizon 은 양의 정수여야 한다: {horizon}")
    cal = pd.DatetimeIndex(pd.to_datetime(list(calendar))) if len(calendar) else pd.DatetimeIndex([])
    if cal.empty or not cal.is_monotonic_increasing or cal.has_duplicates:
        raise CheckError("calendar 는 비어 있지 않고 중복 없이 단조 증가하는 거래일이어야 한다")
    if not np.isfinite(df["forecast"].to_numpy(dtype=np.float64)).all():
        raise CheckError("forecast 에 NaN·inf 가 있다")
    date = pd.DatetimeIndex(_day(df["date"], "date"))
    train_end = pd.DatetimeIndex(_day(df["train_end"], "train_end"))
    at_date, at_end = cal.get_indexer(date), cal.get_indexer(train_end)
    if (at_date < 0).any() or (at_end < 0).any():
        raise CheckError("date·train_end 에 calendar 의 거래일이 아닌 날이 있다")
    if (at_date <= at_end).any():
        raise CheckError("학습 구간 안 예측이 있다 — 누출")
    if (at_date - at_end < horizon).any():
        raise CheckError(f"학습 라벨이 예측일과 겹친다 — train_end 뒤 거래일 {horizon}일 안의 예측이 있다")


def check_weights(w: Any, codes: Any = None, *, budget: float = 1.0, tol: float = 1e-6) -> None:
    """7단계. 롱온리, 합 ≤ ``budget``(나머지는 현금), NaN·inf 없음. ``codes`` 를 주면 길이·중복을 본다."""
    try:
        arr = np.asarray(w, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise CheckError(f"비중은 숫자여야 한다: {e}") from None
    if arr.ndim != 1:
        raise CheckError(f"비중은 1차원이어야 한다: {arr.shape}")
    if arr.size == 0:
        raise CheckError("비중이 비었다 — 전부 현금이면 0 인 칸을 넘긴다")
    if not np.isfinite(arr).all():
        raise CheckError("비중에 NaN·inf 가 있다")
    if (arr < -tol).any():
        raise CheckError("음수 비중 — 실매매 북은 롱온리다")
    if arr.sum() > budget + tol:
        raise CheckError(f"비중 합 {arr.sum():.6f} > {budget}")
    if codes is not None:
        codes = list(codes)
        if len(codes) != arr.size:
            raise CheckError(f"codes({len(codes)})·weights({arr.size}) 길이가 다르다")
        if len(set(codes)) != arr.size:
            raise CheckError("codes 가 중복이다")


def check_attribution(frame: pd.DataFrame, names: Sequence[str], actual: Any, *, atol: float = 1e-9, actual_atol: float = 1e-9) -> None:
    """11단계. date·<팩터…>·specific·total, 날짜마다 total = Σ 팩터 + specific, 그리고 total 이 ``actual``(실제 수익, 같은 순서)과 맞는가.

    앞의 항등식은 core 가 total 을 그렇게 만들어 늘 참이다. 설명 안 된 몫이 숨는 곳은 total 과 실제 수익 사이다. 일 단리 합과 복리 스텝 수익을 맞대면
    차이가 있으니 ``actual_atol`` 을 넉넉히 주되 부호 있는 차를 함께 보고한다(``attribution_report(detail=True)``). NaN·inf 는 거부한다.
    """
    _long_frame(frame, ("date", *names, "specific", "total"), "attribution", key=("date",))
    cells = frame[[*names, "specific"]].to_numpy(dtype=np.float64)
    total = frame["total"].to_numpy(dtype=np.float64)
    if not (np.isfinite(cells).all() and np.isfinite(total).all()):
        raise CheckError("attribution 에 NaN·inf 가 있다 — 그 몫이 합에서 조용히 빠진다")
    if not np.allclose(cells.sum(axis=1), total, rtol=0.0, atol=atol):
        raise CheckError("기여 합이 total 과 다르다 — 설명 안 된 몫이 숨었다")
    act = np.asarray(actual, dtype=np.float64)
    if act.shape != total.shape:
        raise CheckError(f"actual 길이 {act.shape} 가 attribution {total.shape} 과 다르다")
    if not np.isfinite(act).all():
        raise CheckError("actual 에 NaN·inf 가 있다")
    worst = float(np.abs(total - act).max())
    if worst > actual_atol:
        raise CheckError(f"귀속 합이 실제 수익과 다르다 — 최대 차 {worst:.6f} > {actual_atol}")
