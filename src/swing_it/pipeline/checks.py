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
    """2단계. ``end`` 뒤 가격·``end`` 뒤에 안 실적 행을 거부한다. ``end`` 는 애프터마켓 오염 구간 전이어야 한다."""
    _long_frame(prices, ("code", "date", "close"), "prices")
    end_ts = _day(pd.Series([end]), "end").iloc[0]
    if end_ts >= pd.Timestamp(AFTERMARKET_FROM):
        raise CheckError(f"end {end} 는 애프터마켓 오염 구간({AFTERMARKET_FROM}~)이다. "
                         "main 의 read_prices 는 정규장 값을 만들지 못한다 — 이 구간은 받지 않는다")
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


def check_forecast(df: pd.DataFrame, horizon: int) -> None:
    """5단계. 긴 형식 code·date·forecast·train_end. ``train_end`` = 그 예측을 만든 모델의 마지막 학습 행 날짜.

    학습 라벨은 ``horizon`` 거래일 앞을 본다. 그래서 ``date`` 는 ``train_end`` 에서 평일 ``horizon`` 일 이상 떨어져야 한다.
    KRX 달력을 모르니 평일로 센다 — 휴장일이 끼면 놓칠 수 있다(겹침을 잘못 막지는 않는다).
    """
    _long_frame(df, ("code", "date", "forecast", "train_end"), "forecast")
    if horizon <= 0:
        raise CheckError(f"horizon 은 양수여야 한다: {horizon}")
    if not np.isfinite(df["forecast"].to_numpy(dtype=np.float64)).all():
        raise CheckError("forecast 에 NaN·inf 가 있다")
    date = _day(df["date"], "date").to_numpy("datetime64[D]")
    train_end = _day(df["train_end"], "train_end").to_numpy("datetime64[D]")
    if (date <= train_end).any():
        raise CheckError("학습 구간 안 예측이 있다 — 누출")
    if (np.busday_count(train_end, date) < horizon).any():
        raise CheckError(f"학습 라벨이 예측일과 겹친다 — train_end 뒤 평일 {horizon}일 안의 예측이 있다")


def check_weights(w: Any, codes: Any = None, *, budget: float = 1.0, tol: float = 1e-6) -> None:
    """7단계. 롱온리, 합 ≤ ``budget``(나머지는 현금), NaN·inf 없음. ``codes`` 를 주면 길이·중복을 본다."""
    arr = np.asarray(w, dtype=np.float64)
    if arr.ndim != 1:
        raise CheckError(f"비중은 1차원이어야 한다: {arr.shape}")
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


def check_attribution(frame: pd.DataFrame, names: Sequence[str]) -> None:
    """11단계. date·<팩터…>·specific·total. 날짜마다 total = Σ 팩터 + specific — 설명 안 된 몫이 숨지 않는다."""
    _long_frame(frame, ("date", *names, "specific", "total"), "attribution", key=("date",))
    parts = frame[[*names, "specific"]].to_numpy(dtype=np.float64).sum(axis=1)
    if not np.allclose(parts, frame["total"].to_numpy(dtype=np.float64), atol=1e-9, equal_nan=True):
        raise CheckError("기여 합이 total 과 다르다 — 설명 안 된 몫이 숨었다")
