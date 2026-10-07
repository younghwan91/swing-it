"""수급 패널 — 스윙 신호의 수급 축.

2026-08-16 에 옛 flow 모듈이 지워진 뒤(생존편향 경로였다) 처음 다시 두는 수급 피처다. 입력은
**반드시** :func:`swing_it.storage.read_supply_demand` 정문 결과다 — 외국인·기관은 네이버 폐지 백필에도
있어 상장폐지 종목을 담을 수 있다. 개인·기관세부는 쓰지 않는다(키움 전용 → 생존편향).

**단위:** ``supply_demand`` 투자자 열은 **순매수 수량(주)** 이다(금액 아님, 분할 미조정). 그래서 같은 날
거래량으로 나눈 비율로만 비교한다 — 분할 전후·종목 사이 비교가 수량 그대로면 의미가 없다.

**NULL 은 모름, 0 은 없음** — 외국인·기관 중 하나라도 NULL 이면 결과도 NaN 이다(0 으로 채우지 않는다).
"""

from __future__ import annotations

import pandas as pd


def smart_money_net(sd: pd.DataFrame) -> pd.DataFrame:
    """외국인+기관 순매수 수량 패널(날짜×종목). 어느 한쪽이 NULL 이면 NaN.

    ``sd`` 는 ``code``·``date``·``foreign_``·``institution`` 열을 가진 정문 결과.
    """
    df = sd[["code", "date", "foreign_", "institution"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    net = df["foreign_"] + df["institution"]  # NaN 전파 — 한쪽 모름이면 합도 모름
    df = df.assign(net=net)
    # pivot_table 은 전부 NaN 인 행을 지운다 — "모름" 이 "행 없음" 이 되면 안 되므로 pivot 을 쓴다((code,date) 는 PK).
    return df.pivot(index="date", columns="code", values="net").sort_index()


def net_to_volume(net: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """순매수 수량 / 같은 날 거래량. 거래량 0·결측이면 NaN. 두 패널은 같은 축으로 정렬해 넣는다."""
    vol = volume.reindex_like(net)
    return net / vol.where(vol > 0)
