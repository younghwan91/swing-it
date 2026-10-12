"""한국 위생 패널 — 스윙 3호(SEPA 합류)의 유니버스·시장·섹터 주도(K)·상한가·비용.

설계서 ``docs/superpowers/specs/2026-10-09-sepa-confluence.md`` §3 "유니버스(한국 위생)", "K 섹터 주도",
"진입/청산(한국)" 과 §1 비용 행을 코드로 옮긴다. 수치는 전부 사전등록 상수다(탐색 금지).

**규약(:mod:`swing_it.features.technical` 과 같음):** 입력·출력은 날짜×종목 와이드 패널. 날짜 ``d`` 행의
값은 ``d`` 종가까지로 계산된다 — 전략은 ``d+1`` 시가에 진입한다. 가격은 분할조정 패널(정문
:func:`swing_it.storage.read_prices` — 폐지 종목 포함)이어야 한다.

**메타데이터 출처와 한계(2026-10-09 실측, 건수는 ``research/logs/sepa_confluence/coverage_korea.md``):**

- 종목명·시장: 현재 마스터 ``stocks``(거래소/코스닥) + 폐지 마스터 ``delisted_stocks``(유가증권/코스닥/코넥스).
  둘 다에 있는 코드(2026 년 폐지 30건)는 ``stocks`` 를 쓴다 — 두 표의 이름·시장이 전부 일치했다.
- **시장은 시점정합(PIT)이 아니다.** DB 에 상장시장 이력이 없다(이전상장 — 코스닥→유가증권 — 전의 날짜도
  지금 시장으로 분류된다). 비용에는 영향이 없다: 거래세+농특세 **합계**가 2016~2026 모든 구간에서 두 시장이
  같다(:func:`cost_schedule`). 영향은 시장별 breadth 국면과 시장 정합 대조에만 있다.
- 섹터: ``stocks.sector`` (벤더 업종 26종 + 빈 문자열) — **현재 스냅샷 하나뿐이고 폐지 종목엔 없다.**
  ``delisted_stocks`` 에 섹터 컬럼이 없고, ``theme_members`` 는 2026-08-23 부터, ``sector_index`` 는
  2024-01-26 부터라 과거 소속을 복원할 출처가 없다. 그래서 K 는 섹터를 아는 종목에서만 정의한다(NA ≠ False).
- 우선주: 코드 끝자리가 ``'0'`` 이 아니면 우선주(종류주식)다. 가격 테이블의 비-0 코드 24개를 전수 대조했더니
  이름이 전부 ``…우``/``…우B`` 였다(현재 상장 우선주는 가격 테이블에 아예 없다). 이름 규칙(``…우$``)은 **걸지
  않는다** — ``연우``·``성우`` 같은 보통주(끝자리 0)를 잘못 뺀다(실측).
- 스팩·리츠·투자회사: 이름으로 뺀다. ``…스팩…``, ``…N호``(선박투자·해외자원·스팩 — 예 ``동북아10호``,
  ``하나니켈1호``, ``케이프이에스제4호``), ``…리츠``(``블리츠웨이엔터테인먼트`` 같은 부분일치를 피하려고 끝맺음만).
  ETF·ETN 은 가격 테이블에 없다(이름 대조 0건) — 규칙은 남겨 둔다.

Pure DataFrame in → DataFrame out (DB 를 읽는 것은 :func:`load_listing_meta` 하나).
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from swing_it.features.technical import average_trade_value, simple_return
from swing_it.units import eok

# --- 사전등록 상수(설계서 §3) — 실험마다 바꾸지 않는다 ---
ADV_WINDOW = 20             # ADV20
ADV_FLOOR = eok(200)        # 20일 평균 거래대금 ≥ 200억 (백만원 단위: 20,000 = 200억)
MIN_PRICE = 1_000.0         # 가격 ≥ 1,000원 (거래 가능한 원 단위 가격 — raw_close 권장, universe_mask 참고)
MIN_HISTORY = 252           # 상장 후 252거래일 이상
UNIVERSE_MARKETS = ("KOSPI", "KOSDAQ")   # 코넥스 제외
SECTOR_WINDOW = 60          # K: 섹터 60일 수익률
SECTOR_TOP_FRAC = 1.0 / 3.0  # K: 섹터 횡단면 상위 1/3
MIN_SECTOR_MEMBERS = 3      # K: 그날 유니버스 소속 3종목 미만 섹터는 순위에서 뺀다(1종목 섹터 = 종목 수익률)

#: 레포 기준 왕복 비용 41.5bp 의 세금 외 부분(설계서 §1, ``catalyst_flow_swing_gate.COST_RT``):
#: 수수료 3bp(0.015% × 2 — ``krx_quant_core.costs.DEFAULT_COMMISSION_RATE`` 와 같다) + 스프레드 18.5bp.
#: 스프레드는 cross-repo synthesis 의 가정치다(호가로 보정한 값 아님).
COMMISSION_RT = 0.0003
SPREAD_RT = 0.00185

_MARKET_ALIASES = {
    "KOSPI": "KOSPI", "거래소": "KOSPI", "유가증권": "KOSPI", "코스피": "KOSPI",
    "KOSDAQ": "KOSDAQ", "코스닥": "KOSDAQ",
    "KONEX": "KONEX", "코넥스": "KONEX",
}

# 이름 기반 제외. 순서 = 사유 우선순위(스팩이 "…N호" 로도 끝나므로 스팩을 먼저 본다).
_EXCLUDE_NAME_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("spac", re.compile(r"스팩|기업인수목적")),
    ("reit", re.compile(r"리츠$|REIT", re.I)),
    ("fund", re.compile(r"\d+호$|인프라투융자|맥쿼리인프라|선박투자|투자회사")),
    ("etf_etn", re.compile(r"^(KODEX|TIGER|KBSTAR|RISE|ACE|ARIRANG|HANARO|KOSEF|SOL|PLUS|KIWOOM|TIMEFOLIO)\s|ETN|ETF")),
)


# --- 메타데이터 ---

def normalize_market(value: Any) -> "str | None":
    """``거래소``/``유가증권``/``코스닥``/``코넥스``/영문 → ``KOSPI``/``KOSDAQ``/``KONEX``. 모르면 ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return _MARKET_ALIASES.get(str(value).strip().upper())


def exclusion_reason(code: str, name: "str | None") -> "str | None":
    """보통주가 아니면 그 사유(``preferred``/``spac``/``reit``/``fund``/``etf_etn``/``unknown``), 보통주면 ``None``.

    코드 끝자리 ≠ ``'0'`` 이 1차 규칙이다(KRX 표준코드 6번째 자리 = 종류: 0 보통, 5·7·9·K… 우선).
    새 형식 영숫자 코드(``0155E0`` 등)도 끝자리 규칙이 같다. 이름이 없으면 판별할 수 없어 ``unknown``.
    """
    code = str(code)
    if not code or code[-1] != "0":
        return "preferred"
    if name is None or (isinstance(name, float) and math.isnan(name)) or not str(name).strip():
        return "unknown"
    nm = str(name).strip()
    for reason, pat in _EXCLUDE_NAME_RULES:
        if pat.search(nm):
            return reason
    return None


def load_listing_meta(con: Any) -> pd.DataFrame:
    """``stocks`` + ``delisted_stocks`` → 코드별 ``name``·``market``(정규화)·``sector``·``delisted``·``excluded``.

    둘 다에 있는 코드는 ``stocks`` 행을 쓴다(현재 상장 우선; 실측상 이름·시장이 같다). ``delisted`` 는
    ``delisted_stocks`` 에 있는지, ``sector`` 는 빈 문자열을 NaN 으로 바꾼 현재 스냅샷이다.
    가격 테이블이 아니므로 정문(read_prices) 대상이 아니다 — 여기서는 유니버스를 좁히지 않는다.
    """
    cur = pd.read_sql_query("SELECT code, name, market, sector FROM stocks", con)
    dl = pd.read_sql_query("SELECT code, name, market FROM delisted_stocks", con)
    cur["src"] = "stocks"
    dl["src"] = "delisted_stocks"
    dl["sector"] = np.nan
    meta = pd.concat([dl, cur], ignore_index=True).drop_duplicates("code", keep="last")
    meta["delisted"] = meta["code"].isin(set(dl["code"]))
    meta["market"] = meta["market"].map(normalize_market)
    meta["sector"] = meta["sector"].where(meta["sector"].fillna("").str.strip() != "")
    meta["excluded"] = [exclusion_reason(c, n) for c, n in zip(meta["code"], meta["name"])]
    return meta.set_index("code")[["name", "market", "sector", "delisted", "excluded", "src"]].sort_index()


def market_of_code(meta: pd.DataFrame, codes: "pd.Index | None" = None) -> pd.Series:
    """code → ``KOSPI``/``KOSDAQ``/``KONEX``(모르면 NaN). **PIT 아님** — 모듈 설명의 이전상장 한계 참고."""
    s = meta["market"]
    return s if codes is None else s.reindex(codes)


def sector_of_code(meta: pd.DataFrame, codes: "pd.Index | None" = None) -> pd.Series:
    """code → 벤더 업종(현재 스냅샷, 폐지 종목·미분류는 NaN). **PIT 아님 + 생존편향** — 모듈 설명 참고."""
    s = meta["sector"]
    return s if codes is None else s.reindex(codes)


# --- 유니버스 ---

def listing_age(close: pd.DataFrame) -> pd.DataFrame:
    """패널 안에서 첫 유효 종가 이후 경과 거래일(첫날 = 1). 거래정지일도 센다(상장 상태는 이어진다).

    ⚠️ 패널이 시작하는 날(2016-09-09) 이전에 상장한 종목도 그날부터 센다 — 좌측 절단. 그래서
    유니버스는 패널 시작 후 252거래일(≈2017-09) 전엔 비어 있다. 뼈대(MA200·252일 고저)도 어차피 그만큼 필요하다.
    """
    pos = pd.Series(np.arange(1, len(close) + 1, dtype=float), index=close.index)
    first = close.notna().cumsum().gt(0)
    first_pos = close.notna().idxmax()                       # 각 종목 첫 유효일(전부 NaN 이면 첫날 — 아래 마스크로 지운다)
    start = pos.reindex(first_pos.to_numpy()).to_numpy() - 1.0
    age = pd.DataFrame(np.subtract.outer(pos.to_numpy(), start), index=close.index, columns=close.columns)
    return age.where(first)


def universe_mask(close: pd.DataFrame, trade_value: pd.DataFrame, meta: pd.DataFrame, *,
                  raw_close: "pd.DataFrame | None" = None,
                  adv_floor: float = ADV_FLOOR, adv_window: int = ADV_WINDOW,
                  adv_cap: "float | None" = None,
                  min_price: float = MIN_PRICE, min_history: int = MIN_HISTORY,
                  markets: tuple[str, ...] = UNIVERSE_MARKETS) -> pd.DataFrame:
    """한국 위생 유니버스(날짜×종목 bool). 전부 ``d`` 종가까지로 정해진다.

    조건(전부 AND):

    1. 보통주 — :func:`exclusion_reason` 이 ``None``(우선주·스팩·리츠·투자회사·ETF/ETN·이름 미상 제외).
    2. 시장 ∈ ``markets``(:func:`market_of_code`, PIT 아님). 메타에 없는 코드는 제외.
    3. ``d`` 에 거래됨(종가 유효).
    4. ADV20 ≥ ``adv_floor`` — ``d`` 포함 직전 ``adv_window`` 행 평균 거래대금(백만원), 창이 다 차야 값이 있다
       (:func:`~swing_it.features.technical.average_trade_value`). 거래정지일(NaN)이 창에 끼면 그날은 빠진다.
       폐지 종목(``source='naver'``)의 거래대금은 ``close×volume`` 근사다(중앙 오차 0.7%).
       ``adv_cap`` 을 주면 ADV20 < ``adv_cap`` 도 건다(구간 유니버스 — 예 중소형 5,000 ≤ ADV20 < 20,000,
       ``research/logs/sepa_explore/CONVERGENCE.md``). 기본 ``None`` = 상한 없음(기존 동작 그대로).
    5. 가격 ≥ ``min_price`` — **``raw_close`` 를 주면 그걸로**, 없으면 ``close``(분할조정)로 본다.
       1,000원 규칙은 실제 호가 가격에 대한 것이라 원자료가 맞다: 뒤에 액면분할한 종목은 조정가가 과거를
       깎아 내려 "1,000원 미만" 으로 잘못 빠지고, 병합(역분할)은 반대로 잘못 남는다. 원자료
       (``read_prices(con, table="daily_bars")``)도 수집 배치마다 조정 기준이 달랐던 이력이 있어
       (``price_adjust`` 모듈 설명) 완전한 원가는 아니다 — 그래도 "그날 화면에 찍힌 가격" 에 가장 가깝다.
    6. 상장 후 ``min_history`` 거래일 이상(:func:`listing_age`).

    Args:
        close: 분할조정 종가.
        trade_value: 거래대금(백만원) — 같은 모양.
        meta: :func:`load_listing_meta` 결과(``code`` 인덱스, ``market``·``excluded`` 컬럼).
        raw_close: 원자료 종가(선택). 1,000원 규칙에만 쓴다.
        adv_cap: ADV20 상한(미만, 백만원). ``None`` 이면 상한 없음.

    Returns:
        ``close`` 와 같은 모양의 bool 패널.
    """
    codes = close.columns
    excl = meta["excluded"].reindex(codes)
    mk = meta["market"].reindex(codes)
    common = (excl.isna() & meta.index.to_series().reindex(codes).notna() & mk.isin(markets)).to_numpy()

    tv = trade_value.reindex(index=close.index, columns=codes)
    adv = average_trade_value(tv, adv_window)
    px = close if raw_close is None else raw_close.reindex(index=close.index, columns=codes)
    ok = (close.notna()
          & (adv >= adv_floor)
          & (px >= min_price)
          & (listing_age(close) >= min_history))
    if adv_cap is not None:
        ok &= adv < adv_cap
    return ok & pd.Series(common, index=codes)


# --- K 섹터 주도 ---

def sector_returns(close: pd.DataFrame, sector_map: pd.Series, *, universe: "pd.DataFrame | None" = None,
                   window: int = SECTOR_WINDOW, min_members: int = MIN_SECTOR_MEMBERS) -> pd.DataFrame:
    """날짜×섹터 동일가중 ``window`` 일 수익률 = 그날 유니버스 소속·섹터 확인 종목의 ``window`` 일 단순수익률 평균.

    ``sector_index`` 를 쓰지 않는다 — 2024-01-26 부터라 과거를 못 덮는다. 구성 종목은 **그날** 유니버스
    (``d`` 종가까지로 정해진다)라 시점 안전하다. 그날 유효 구성 종목이 ``min_members`` 미만이면 NaN.
    """
    r = simple_return(close, window)
    if universe is not None:
        r = r.where(universe.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool))
    sec = sector_map.reindex(close.columns)
    known = sec.notna().to_numpy()
    r = r.loc[:, known]
    keys = sec[known].to_numpy()
    mean = r.T.groupby(keys).mean().T
    cnt = r.T.groupby(keys).count().T
    return mean.where(cnt >= min_members)


def sector_leadership(close: pd.DataFrame, sector_map: pd.Series, *, universe: "pd.DataFrame | None" = None,
                      window: int = SECTOR_WINDOW, top_frac: float = SECTOR_TOP_FRAC,
                      min_members: int = MIN_SECTOR_MEMBERS) -> pd.DataFrame:
    """K 층: 종목의 섹터가 그날 섹터 횡단면 상위 ``top_frac`` 인가 → 날짜×종목 nullable bool(``boolean``).

    순위 = :func:`sector_returns` 의 내림차순(동률은 높은 쪽으로, ``method='min'``). 그날 순위가 매겨진 섹터
    수가 ``n`` 이면 순위 ≤ ``n × top_frac`` 이 주도 섹터다(``n=26`` → 상위 8개). 순위가 매겨진 섹터가 3개 미만인
    날은 전부 NA.

    **NA ≠ False(설계서 F 와 같은 원칙):** 섹터를 모르는 종목(폐지 종목 대부분·미분류)과 그날 순위에서 빠진
    섹터(구성 < ``min_members``)의 종목은 ``<NA>`` 다. 확신 점수에서 NA 를 0 으로 세면 폐지 종목이 체계적으로
    K=0 이 된다 — 소비자가 정한다.

    종목 자신이 유니버스 밖이어도 섹터 값은 붙는다(섹터 수익률 구성에만 유니버스를 쓴다).
    """
    sr = sector_returns(close, sector_map, universe=universe, window=window, min_members=min_members)
    n = sr.notna().sum(axis=1)
    rank = sr.rank(axis=1, ascending=False, method="min")
    lead = (rank <= n.to_numpy()[:, None] * top_frac).astype(float).where(sr.notna())
    lead.loc[n < 3] = np.nan
    sec = sector_map.reindex(close.columns)
    out = lead.reindex(columns=sec.to_numpy())            # 섹터 NaN → 전부 NaN 컬럼
    out.columns = close.columns
    out.index = close.index
    return out.astype("boolean")


# --- 상한가 ---

#: KRX 호가단위(2023-01-25 통합 표). 정본은 ``kiwoom_client.tick_size`` — 벡터 연산용 사본이며
#: ``krx_quant_core.market.limits.limit_up_price`` 와의 일치를 테스트가 검사한다.
_TICK_UPPER = np.array([2_000, 5_000, 20_000, 50_000, 200_000, 500_000], dtype=float)
_TICK_SIZE = np.array([1, 5, 10, 50, 100, 500, 1_000], dtype=float)

#: 상한가 시가 판정 허용폭(전일종가 대비). 상한가 = 전일종가×1.3 을 그 가격대 호가로 **내림**한 값이라
#: 1.3 보다 최대 한 호가 낮다. 1,000원 이상에서 한 호가는 상한가의 0.5% 이하(옛 2023-01 이전 표 포함)
#: → 전일종가 대비 0.65%p 이하. 0.7%p 로 잡아 조정가·옛 호가표에서도 상한가 시가를 놓치지 않게 한다.
LIMIT_UP_REL_TOL = 0.007


def tick_size_vec(price: "pd.DataFrame | np.ndarray") -> np.ndarray:
    """가격 → 호가단위(현행 표, float 배열). 0 이하·NaN 은 NaN."""
    p = np.asarray(price, dtype=float)
    t = _TICK_SIZE[np.searchsorted(_TICK_UPPER, np.where(np.isfinite(p), p, 0.0), side="right")]
    return np.where(np.isfinite(p) & (p > 0), t, np.nan)


def limit_up_price(prev_close: "pd.DataFrame | pd.Series | np.ndarray") -> Any:
    """전일종가 → 상한가(현행 호가표, ±30% — 2015-06-15 이후). ``krx_quant_core.market.limits.limit_up_price``
    의 벡터판(``1.3`` 리터럴·``+1e-9`` 까지 같은 식). 입력과 같은 모양으로 돌려준다."""
    raw = np.asarray(prev_close, dtype=float) * 1.3
    tick = tick_size_vec(raw)
    out = np.floor(raw / tick + 1e-9) * tick
    if isinstance(prev_close, pd.DataFrame):
        return pd.DataFrame(out, index=prev_close.index, columns=prev_close.columns)
    if isinstance(prev_close, pd.Series):
        return pd.Series(out, index=prev_close.index)
    return out


def is_limit_up_open(open_: Any, prev_close: Any, *, exact: bool = False,
                     rel_tol: float = LIMIT_UP_REL_TOL) -> Any:
    """시가가 상한가인가(→ 설계서: 체결 불가로 버림). 기본은 비율 판정, ``exact=True`` 는 호가 판정.

    - 기본: ``open / prev_close ≥ 1.30 − rel_tol``. 분할조정가(호가 격자가 깨진다)와 2023-01-25 이전 호가표에서도
      성립하는 판정이다. 상한가 바로 아래(+29.3% 이상) 시가도 잡히지만, 그 시가는 이미 추격 금지(피벗×1.05)
      한참 위라 진입 규칙상 결과가 같다.
    - ``exact=True``: ``open ≥ limit_up_price(prev_close)``. 원자료·현행 호가표(2023-01-25+)에서만 정확하다.

    기준가격이 전일종가가 아닌 날(권리락·배당락·신규상장 등)은 다루지 않는다(``krx_quant_core`` 와 같음).
    NaN·0 이하 입력은 False.
    """
    o = np.asarray(open_, dtype=float)
    p = np.asarray(prev_close, dtype=float)
    valid = np.isfinite(o) & np.isfinite(p) & (p > 0) & (o > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if exact:
            hit = o >= np.asarray(limit_up_price(p), dtype=float)
        else:
            hit = o / np.where(p > 0, p, np.nan) >= 1.30 - rel_tol
    res = hit & valid
    like = open_ if isinstance(open_, (pd.DataFrame, pd.Series)) else None
    if isinstance(like, pd.DataFrame):
        return pd.DataFrame(res, index=like.index, columns=like.columns)
    if isinstance(like, pd.Series):
        return pd.Series(res, index=like.index)
    return res


# --- 비용 ---

def _as_date(on: Any) -> date:
    return pd.Timestamp(on).date()


def cost_rt(on: Any, market: Any = "KOSPI", *, commission_rt: float = COMMISSION_RT,
            spread_rt: float = SPREAD_RT) -> float:
    """왕복 비용률 = 매도세(일자별, ``krx_quant_core.costs.sell_tax_rate``) + 수수료 + 스프레드.

    ``on`` 은 **매도일**(세율은 매도 시점 기준)이다 — 보유가 연초를 넘으면 진입일이 아니라 청산일 세율이 붙는다.
    청산일을 모르면 진입일로 근사한다. 2026 세율에서 41.5bp(레포 기준선)와 같다. 2배 스트레스는 호출부가 ×2.
    세율 출처·신뢰도는 ``krx_quant_core/costs/tax.py`` 설명(2026 인상은 보도 확인, 이전 단계는 2차 출처).
    """
    from krx_quant_core.costs import sell_tax_rate  # noqa: PLC0415 — 비용 쪽만 코어에 의존

    return float(sell_tax_rate(_as_date(on), market)) + commission_rt + spread_rt


def cost_schedule(markets: tuple[str, ...] = UNIVERSE_MARKETS, *, commission_rt: float = COMMISSION_RT,
                  spread_rt: float = SPREAD_RT) -> pd.DataFrame:
    """세율 시행일별 왕복 비용표(bp) — ``krx_quant_core.costs.TAX_SCHEDULE`` 를 그대로 펼친다(세율을 여기서 만들지 않는다)."""
    from krx_quant_core.costs import TAX_SCHEDULE, tax_rate  # noqa: PLC0415

    rows = []
    for period in TAX_SCHEDULE:
        eff = period.effective
        for m in markets:
            tr = tax_rate(eff, m)
            tax = float(tr.total)
            rows.append({
                "effective": None if eff == date.min else eff.isoformat(),
                "market": m,
                "securities_tax_bp": float(tr.securities_tax) * 1e4,
                "rural_special_tax_bp": float(tr.rural_special_tax) * 1e4,
                "tax_bp": tax * 1e4,
                "commission_bp": commission_rt * 1e4,
                "spread_bp": spread_rt * 1e4,
                "round_trip_bp": (tax + commission_rt + spread_rt) * 1e4,
            })
    return pd.DataFrame(rows)
