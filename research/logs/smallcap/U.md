# smallcap 1단계 — 유니버스 커버리지 (2026-11-02, 결과 전)

수익률·시가·고가·저가·종가는 읽지 않았다. 거래대금·거래량·행 수·NULL 여부만 셌다. 후보 규칙의 사건 수도 세지 않았다(사건 정의에 수익률이 들어가기 때문이다). 쿼리 전문은 끝에 있다.

## 유니버스 U(t) — PREREG §3

보통주(코드 끝 0)·이름으로 스팩·리츠·ETF류 제외·`pre_listing` 제외, 그날 거래량 > 0, ADV20(t−20..t−1, 관측 ≥ 15, 거래량 0 인 날 제외) ∈ [3억, 20억). `half_of`(코드 sha256 첫 바이트 짝홀): 1 = 잠금 절반, 0 = 탐색 절반.

## U(t) 종목-일·종목 수·폐지 비중 (2009-01-02 ~ 2016-08-31)

"폐지" = `daily_bars_adj_v2.source='naver_adj'`(2026-10 기준 비상장 전부 — 표본 안에서 폐지되지 않은 종목도 들어 있다).

| 연도 | 잠금 종목-일 | 잠금 종목 | 잠금 폐지 비중 | 탐색 종목-일 | 탐색 종목 | 탐색 폐지 비중 |
|---|---|---|---|---|---|---|
| 2009 | 68,371 | 699 | 27.3% | 61,506 | 658 | 27.8% |
| 2010 | 71,696 | 715 | 22.7% | 66,257 | 687 | 23.6% |
| 2011 | 66,970 | 700 | 19.6% | 64,963 | 645 | 21.6% |
| 2012 | 73,564 | 722 | 15.6% | 70,034 | 697 | 17.5% |
| 2013 | 74,211 | 669 | 13.4% | 74,450 | 671 | 16.9% |
| 2014 | 75,940 | 685 | 11.3% | 77,468 | 711 | 15.0% |
| 2015 | 80,435 | 769 | 11.2% | 84,106 | 801 | 12.7% |
| 2016(~08) | 58,086 | 739 | 10.3% | 61,052 | 724 | 11.1% |
| 합계 | 569,273 | | | 559,836 | | |

하루 U(t) 종목 수(5·50·95 백분위): 잠금 256·295·358, 탐색 234·289·374.

## ADV20 구간별 종목-일 (전 종목, 두 절반 합)

| ADV20 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016(~08) |
|---|---|---|---|---|---|---|---|---|
| 1.5~3억 | 45,725 | 52,244 | 50,947 | 50,926 | 49,909 | 49,888 | 48,276 | 29,941 |
| **3~7.5억** | 65,014 | 70,473 | 69,094 | 72,290 | 73,840 | 75,662 | 77,161 | 53,002 |
| **7.5~20억** | 64,863 | 67,480 | 62,839 | 71,308 | 74,821 | 77,746 | 87,380 | 66,136 |
| 20~50억 | 46,722 | 47,861 | 46,075 | 56,095 | 55,243 | 55,239 | 72,150 | 53,112 |
| 50~200억 | 47,752 | 44,947 | 44,055 | 47,855 | 44,830 | 43,037 | 66,186 | 45,189 |
| 200억~ | 28,264 | 24,380 | 25,501 | 20,347 | 15,160 | 14,752 | 25,085 | 14,229 |

## 생존편향·원천

| 항목 | 값 |
|---|---|
| `daily_bars_adj_v2` | 2007-11~ 행 있음. 2009~2016-08 연 상장 1,263~1,698종목·폐지 383~705종목. 폐지 쪽 `raw_close` 는 전부 NULL |
| KIND 2009-01~2016-08 폐지 대비 v2 | 504건 중 502 (2015·2016 각 1건 없음) |
| `delisted_stocks.last_trade_date` | 4,213행 중 489행만 값이 있고 이 구간은 0건 — 대조에 못 쓴다. 메모리의 "KIND 연 5~20% 누락"은 여기서 재확인 불가 |
| 폐지 쪽 원 종가 | v2 에 없음. `naver_investor_trend.close` 가 U(t) 폐지 종목-일의 99.98%(193,624 중 193,579)를 덮는다. 값 대조는 2단계 |
| 시장 구분(현재 스냅샷) | U(t) 종목-일 거래소 374,979 · 유가증권 53,327 · 코스닥 700,786 · 코넥스 17 |
| 지수 001·101 | 2006~2016 거래일 전부(연 245~253행) |
| `earnings_dart_doc` | 2007~2015 69,612행(연 1,670~2,093종목). 파서 바 97.89% ≥ 95% 통과. **2016 없음**. 이 연구는 쓰지 않는다(PREREG §2) |

## 호가·분봉 (비용 보정용)

| 표 | 범위 | 최근 한 주 종목 | 2026-08-31 주 U 대역 759종목과 겹침 |
|---|---|---|---|
| `minute_bars` | 2025-08-01~ | 2,527 (10-19 주) | 688 (91%) |
| `quotes` | 2026-08-24~ | 466 | 94 (12%) |
| `ticks` | 2026-08-24~ | 462 | 94 (12%) |
| `quote_events` | 2026-09-16~ | 466 | 그 주 0 (시작 전) |

2009~2016 호가는 없다. 겹치는 94종목은 scalp-it 수집 대상이라 무작위 표본이 아니다.

## 쿼리 전문 (읽기 전용, `set statement_timeout` 60~180초)

```sql
select column_name,data_type from information_schema.columns where table_name=%s order by ordinal_position  -- daily_bars_adj_v2, naver_investor_trend, delisted_stocks, kind_delistings, minute_bars, quotes, quote_events, ticks, sector_index, earnings_dart_doc, stocks
select code,date,trade_value,volume,source,close_src,raw_close is not null as has_raw from daily_bars_adj_v2 where date between '2007-11-01' and '2016-08-31'
select code,name,kind from stocks
select code,name,null::text as kind from delisted_stocks
select code,name,null::text as kind from kind_delistings
select code,delist_date from kind_delistings where delist_date between '2009-01-01' and '2016-08-31'
select code,last_trade_date from delisted_stocks
select distinct code from daily_bars_adj_v2 where source='naver_adj'
select code,date from naver_investor_trend where date between '2009-01-01' and '2016-08-31' and close is not null
select code,market from stocks
select code,market from delisted_stocks
select code,extract(year from date)::int y,count(*) n from sector_index where code in ('001','101') and date<'2017-01-01' group by 1,2 order by 1,2
select left(period,4) y,count(*) n,count(distinct code) codes from earnings_dart_doc group by 1 order by 1
select min(ts) a,max(ts) b from minute_bars  -- quotes, quote_events, ticks 도 같다
select count(distinct code) n from minute_bars where ts>=%s and ts<%s  -- 2023-06-05, 2024-06-03, 2025-06-09, 2026-08-03, 2026-10-19 부터 5일
select count(distinct code) n from {quotes,quote_events,ticks} where ts>=%s and ts<%s  -- 2026-06-01, 2026-08-03, 2026-10-19 부터 5일
select code,avg(trade_value) adv,count(*) n from daily_bars_adjusted where date between '2026-08-12' and '2026-09-11' group by code
select distinct code from {minute_bars,quotes,quote_events,ticks} where ts>='2026-08-31' and ts<'2026-09-05'
select last_trade_date from delisted_stocks where last_trade_date is not null limit 3
select count(*), count(last_trade_date) from delisted_stocks
select count(*) from delisted_stocks where replace(last_trade_date,'-','') between '20090101' and '20160831'
select count(distinct d.code), count(distinct d.code) filter (where exists (select 1 from daily_bars_adj_v2 v where v.code=d.code and v.source='naver_adj' and v.date<='2016-08-31')) from delisted_stocks d where replace(last_trade_date,'-','') between '20090101' and '20160831'
```
