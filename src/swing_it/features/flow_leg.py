"""SEPA 합류의 F(수급) 다리 — 외국인+기관 20일 매집 확인 층.

설계: ``docs/superpowers/specs/2026-10-09-sepa-confluence.md`` §3 "F 수급".

    F = (외국인+기관 20일 순매수 / 20일 거래량 > 0) ∧ (외국인 20일 순매수 > 0)

외국인을 따로 보는 이유는 MULTI_ALPHA 부록이다(외국인이 전 지평선 최대 초과 = 스마트머니, 기관은 단기 역).
이 층은 *확인* 이지 단독 알파가 아니다 — 수급 *가속*(``inst_flow_accel``)은 이미 기각됐고 여기서 다시 재지 않는다.

**입력은 반드시** :func:`swing_it.storage.read_supply_demand` 정문 결과다(``foreign_``·``institution``·
``acc_trde_qty`` — 셋 다 네이버 폐지 백필에도 있어 폐지 종목을 담는다). 개인·기관세부는 쓰지 않는다.

**분모는 ``supply_demand.acc_trde_qty`` 다 — ``daily_bars(_adjusted).volume`` 이 아니다(2026-10-09 실측).**
같은 행의 순매수 수량과 ``acc_trde_qty`` 는 같은 주식수 기준이다(전 행에서 ``|foreign_| ≤ acc_trde_qty``).
그런데 그 기준은 *수집 시점의 수정주가 기준* 이라 ``daily_bars`` 원 거래량과 다를 수 있다(키움 2024 행의 ~20%,
배수 최대 ~5배) — 그 경우 ``|순매수| > daily_bars.volume`` 인 행이 실제로 나온다. ``daily_bars_adjusted.volume``
은 **조정되지 않은** 원 거래량이라(종가만 조정) 역시 맞지 않는다. 그래서 하루 비율 ``순매수/acc_trde_qty`` 는
분할·병합과 무관하다(분자·분모가 같은 행·같은 기준).

**하지만 20일 *합* 은 기준이 섞이면 무관하지 않다.** 수집 뒤에 원자료가 이어 붙어서 ``supply_demand`` 시계열
안에 조정 안 된 이벤트(기준 단절)가 있다(실측 949건·730종목, 2026 에 381건 집중). 단절을 낀 창에서
``Σ순매수/Σ거래량`` 은 단절 전 날들을 배수만큼 가볍게/무겁게 센다 — 부호도 바뀔 수 있다. 그래서 두 방식을 둔다:

- ``weight=None`` (설계서 문자 그대로): ``Σ순매수 / Σacc_trde_qty``. 단절 창에서 가중이 틀어진다.
- ``weight=거래대금 패널``: 하루 비율 ``순매수/acc_trde_qty`` 를 그날 거래대금으로 가중평균한다. 거래대금은
  주식수 기준이 없으므로(원 단위) 단절을 낀 창에서도 기준 불변이다. 단절이 없고 가격이 일정하면 앞 방식과 같다.

어느 쪽을 판정 config 로 쓸지는 사전등록(VERDICT.md)에서 고른다 — 이 모듈은 둘 다 계산할 뿐이다.

**NULL 은 모름, 0 은 없음.** 창 20일 중 하루라도 외국인·기관·거래량(·가중) 중 하나가 모르면(NULL 이거나
**행 자체가 없으면**) 결과는 NaN 이다. 0 으로 채우지 않는다. 지금 DB 의 ``foreign_``·``institution`` 에는
NULL 이 없다 — "모름" 은 거의 전부 **행 없음**(2017~2023 은 연 ~650 종목만 수집)에서 온다. 그래서 패널은
반드시 거래일 달력(가격 패널의 index)으로 reindex 해야 한다. 수급 행만으로 피벗하면 빠진 날이 창에서
조용히 건너뛰어져 20일 창이 21일·22일 창이 된다.
⚠️ 필드값 0 은 "관망" 과 "미보고" 가 구분되지 않는다(수집기 ``to_int`` 가 파싱 실패를 0 으로 준다 —
inst_flow_accel 사전등록에 적힌 한계). 0 을 그대로 0 으로 쓴다; 결측으로 바꾸는 것은 사후 선택이다.

**시점:** 행 t 의 값은 t−19..t 거래일을 쓴다(t 포함). 투자자별 확정치는 t 장 마감 뒤에 나오므로 **t+1 시가
진입**(설계서 규칙)에서는 룩어헤드가 아니다. t 종가 진입에 쓰려면 호출자가 ``shift(1)`` 해야 한다.

**중첩관측(GUARDRAILS §3):** 이 모듈은 20일 창 하나만 만든다. ML 피처로 5·20·60 같은 끝이 같은 창을 나란히
쓰려면 겹치지 않게 차분해서 쓴다. Sias(2004) 대로 수급에서 보는 양은 부호보다 **크기** 다 — F 는 설계서가
정한 부호 플래그이고, 크기는 ``smart_ratio_20`` 연속값으로 남긴다.

Pure DataFrame in → DataFrame out. 넓은 패널은 index=날짜, columns=종목.
"""

from __future__ import annotations

import pandas as pd

WINDOW = 20  # 설계서 고정값 — 탐색하지 않는다.


def flow_wide(sd: pd.DataFrame, *, index: "pd.Index | None" = None,
              columns: "pd.Index | None" = None) -> dict[str, pd.DataFrame]:
    """정문 결과(long) → ``foreign``·``institution``·``volume`` 넓은 패널(날짜×종목).

    ``index`` 는 **거래일 달력**(보통 가격 패널의 index)을 넘긴다 — 수급 행이 없는 (날짜, 종목) 이 NaN(모름)으로
    남아야 창 완전성 검사가 의미를 갖는다. ``None`` 이면 수급 행에 나온 날짜만 쓴다(테스트·탐색용).
    ``(code, date)`` 는 PK 라 ``pivot`` 이 중복에서 시끄럽게 실패한다(``pivot_table`` 처럼 조용히 합치지 않는다).
    """
    df = sd[["code", "date", "foreign_", "institution", "acc_trde_qty"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    out = {}
    for name, col in (("foreign", "foreign_"), ("institution", "institution"), ("volume", "acc_trde_qty")):
        p = df.pivot(index="date", columns="code", values=col).sort_index().astype(float)
        if index is not None or columns is not None:
            p = p.reindex(index=pd.DatetimeIndex(index) if index is not None else p.index,
                          columns=columns if columns is not None else p.columns)
        out[name] = p
    return out


def _full(known: pd.DataFrame, window: int) -> pd.DataFrame:
    """창 ``window`` 일이 전부 참인가(첫 ``window−1`` 행은 거짓)."""
    return known.astype(float).rolling(window, min_periods=window).sum().eq(window)


def _window_ratio(num: pd.DataFrame, volume: pd.DataFrame, window: int,
                  weight: "pd.DataFrame | None") -> pd.DataFrame:
    """창 비율. ``weight`` 없으면 Σnum/Σvolume, 있으면 하루 비율(num/volume)의 가중평균. 모름 하루라도 → NaN."""
    vol = volume.reindex_like(num)
    if weight is None:
        n = num.rolling(window, min_periods=window).sum()
        d = vol.rolling(window, min_periods=window).sum()
        return n / d.where(d > 0)
    w = weight.reindex_like(num)
    daily = num / vol.where(vol > 0)
    # 거래 없는 날(거래량 0 ∧ 순매수 0)은 "알려진 0" — 가중 0 으로 들어간다. 거래량 0 인데 순매수 ≠ 0 은 모순 → 모름.
    daily = daily.mask(vol.eq(0) & num.eq(0), 0.0)
    n = (daily * w).rolling(window, min_periods=window).sum()
    d = w.rolling(window, min_periods=window).sum()
    return n / d.where(d > 0)


def flow_known(foreign: pd.DataFrame, institution: pd.DataFrame, volume: pd.DataFrame, *,
               window: int = WINDOW, weight: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """창 ``window`` 일 동안 외국인·기관·거래량(·가중)이 **전부** 알려졌는가(bool 패널).

    데이터 *존재* 마스크다 — 창 전체가 거래정지(Σ거래량 0)면 known 이어도 비율은 NaN 이다.
    """
    k = foreign.notna() & institution.reindex_like(foreign).notna() & volume.reindex_like(foreign).notna()
    if weight is not None:
        k &= weight.reindex_like(foreign).notna()
    return _full(k, window)


def smart_ratio(foreign: pd.DataFrame, institution: pd.DataFrame, volume: pd.DataFrame, *,
                window: int = WINDOW, weight: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """``smart_ratio_20`` — (외국인+기관) 20일 순매수 / 20일 거래량. 하나라도 모르면 NaN(합도 모름)."""
    net = foreign + institution.reindex_like(foreign)  # flows.smart_money_net 과 같은 NaN 전파
    return _window_ratio(net, volume, window, weight)


def foreign_net(foreign: pd.DataFrame, *, window: int = WINDOW) -> pd.DataFrame:
    """``foreign_net_20`` — 외국인 20일 순매수 **수량**(주) 합. 하루라도 모르면 NaN.

    ⚠️ 수량 합이라 종목 사이·기준 단절을 낀 창에서는 크기를 비교하지 않는다 — 부호만 쓴다.
    """
    return foreign.rolling(window, min_periods=window).sum()


def foreign_ratio(foreign: pd.DataFrame, volume: pd.DataFrame, *, window: int = WINDOW,
                  weight: "pd.DataFrame | None" = None) -> pd.DataFrame:
    """``foreign_ratio_20`` — 외국인 20일 순매수 / 20일 거래량. ``weight=None`` 이면 부호가 ``foreign_net`` 과 같다."""
    return _window_ratio(foreign, volume, window, weight)


def f_flag(smart: pd.DataFrame, foreign: pd.DataFrame) -> pd.DataFrame:
    """F 플래그 = smart > 0 ∧ foreign > 0 → float 패널(1.0 / 0.0 / **NaN = 모름**).

    ``foreign`` 은 ``foreign_net`` 이든 ``foreign_ratio`` 든 부호만 쓴다. bool 로 주지 않는 이유: bool 은
    NaN 을 담지 못해 "모름" 이 "거짓" 으로 접힌다 — 수급이 없는 종목을 "수급 나쁨" 으로 세는 함정이다.
    알려진 행만 고르려면 ``flag.notna()`` 를 마스크로 쓴다.
    """
    fo = foreign.reindex_like(smart)
    flag = ((smart > 0) & (fo > 0)).astype(float)
    return flag.where(smart.notna() & fo.notna())


def smart_persistence(foreign: pd.DataFrame, institution: pd.DataFrame, *,
                      window: int = WINDOW) -> pd.DataFrame:
    """최근 ``window`` 일 중 외국인+기관 순매수 > 0 인 날 수(0..window). 하루라도 모르면 NaN. ML 피처용.

    하루 부호만 세므로 분할·기준 단절과 무관하다. 순매수 0 인 날(관망 또는 미보고)은 > 0 이 아니다.
    """
    net = foreign + institution.reindex_like(foreign)
    pos = (net > 0).astype(float).where(net.notna())
    return pos.rolling(window, min_periods=window).sum()


def flow_leg(sd: pd.DataFrame, *, index: "pd.Index | None" = None, columns: "pd.Index | None" = None,
             weight: "pd.DataFrame | None" = None, window: int = WINDOW) -> dict[str, pd.DataFrame]:
    """정문 결과 → F 다리 패널 한 벌.

    Returns: ``flow_known``(bool), ``smart_ratio_20``, ``foreign_net_20``(주), ``foreign_ratio_20``,
    ``F``(1/0/NaN), ``smart_pos_days_20``. ``weight`` 를 주면(거래대금 패널) 두 비율이 기준 불변 가중평균이 된다.
    """
    w = flow_wide(sd, index=index, columns=columns)
    f, i, v = w["foreign"], w["institution"], w["volume"]
    wt = weight.reindex_like(f) if weight is not None else None
    sr = smart_ratio(f, i, v, window=window, weight=wt)
    fr = foreign_ratio(f, v, window=window, weight=wt)
    return {
        "flow_known": flow_known(f, i, v, window=window, weight=wt),
        "smart_ratio_20": sr,
        "foreign_net_20": foreign_net(f, window=window),
        "foreign_ratio_20": fr,
        "F": f_flag(sr, fr),
        "smart_pos_days_20": smart_persistence(f, i, window=window),
    }
