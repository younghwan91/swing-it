# smallcap 1단계 — 유니버스 커버리지 (2026-11-02, 결과 전. 정정 SC-1 로 다시 셈)

수익률·시가·고가·저가·종가는 읽지 않았다. 원 종가×원 거래량·거래량·행 수·NULL 여부만 셌다. 두 절반을 세는 것은 이 파일뿐이고 2·3·4·8단계 러너는 탐색 절반만 읽는다. 재현: `research/experiments/smallcap_s1.py`.

**첫 판과 달라진 점(C1).** 첫 판의 거래대금은 폐지 종목에서 v2 근사값이었다. 리뷰어 실측으로 탐색 절반 폐지 대역 칸 103,273 중 16,272 가 가짜이고 25,334 가 빠졌다. 이제 **모든 종목**이 `naver_investor_trend` 원 종가 × 원 거래량이고 v2 로 돌아가지 않는다. 값이 없으면 U 밖이다.

## 유니버스

보통주(코드 끝 0)·`name_filter.csv` 제외·`pre_listing` 제외, 그날 거래량 > 0, 그날 원 거래대금 있음, ADV20(t−20..t−1, 관측 ≥ 15, 거래량 0 인 날 제외). U(t) = [3억, 20억), U2(t) = [20억, 100억)(§6 지도 대조). 절반 = 코드 sha256 첫 바이트 짝홀(`smallcap_lib.half_of`, 시험 벡터 있음). 폐지 = `source='naver_adj'`(2026-10 기준 비상장 전부).

### U(t) ADV 3~20억

| 연도 | 잠금 종목-일 | 잠금 종목 | 잠금 폐지 비중 | 탐색 종목-일 | 탐색 종목 | 탐색 폐지 비중 |
|---|---|---|---|---|---|---|
| 2009 | 71,992 | 719 | 30.7% | 65,912 | 673 | 32.5% |
| 2010 | 73,723 | 736 | 24.5% | 67,963 | 700 | 25.4% |
| 2011 | 68,458 | 703 | 21.1% | 65,436 | 653 | 21.8% |
| 2012 | 75,199 | 731 | 17.3% | 72,320 | 706 | 19.7% |
| 2013 | 76,051 | 677 | 15.2% | 75,553 | 675 | 17.7% |
| 2014 | 76,606 | 691 | 11.9% | 77,680 | 716 | 15.0% |
| 2015 | 81,511 | 773 | 12.4% | 85,103 | 807 | 13.5% |
| 2016 | 59,109 | 746 | 11.9% | 61,492 | 729 | 11.5% |
| 합계 | 582,649 | | | 571,459 | | |

하루 종목 수(5·50·95 백분위): 잠금 [264, 302, 365], 탐색 [240, 296, 376]

### U2(t) ADV 20~100억

| 연도 | 잠금 종목-일 | 잠금 종목 | 잠금 폐지 비중 | 탐색 종목-일 | 탐색 종목 | 탐색 폐지 비중 |
|---|---|---|---|---|---|---|
| 2009 | 36,189 | 473 | 29.6% | 33,463 | 461 | 25.5% |
| 2010 | 37,346 | 484 | 25.1% | 35,269 | 456 | 23.3% |
| 2011 | 37,441 | 491 | 17.8% | 32,057 | 444 | 16.9% |
| 2012 | 43,765 | 522 | 17.2% | 40,976 | 504 | 15.0% |
| 2013 | 41,506 | 457 | 15.4% | 41,447 | 472 | 14.2% |
| 2014 | 39,162 | 464 | 13.9% | 41,686 | 510 | 14.6% |
| 2015 | 55,294 | 632 | 13.0% | 57,790 | 663 | 15.3% |
| 2016 | 41,445 | 626 | 14.3% | 40,167 | 611 | 14.4% |
| 합계 | 332,148 | | | 322,855 | | |

하루 종목 수(5·50·95 백분위): 잠금 [121, 166, 256], 탐색 [111, 159, 262]

### ADV20 구간별 종목-일 (두 절반 합)

| ADV20(백만원) | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 |
|---|---|---|---|---|---|---|---|---|
| 150~300 | 49,516 | 55,352 | 53,030 | 52,295 | 51,130 | 51,391 | 48,669 | 29,911 |
| 300~750 | 69,754 | 74,449 | 71,509 | 75,266 | 76,514 | 76,876 | 78,266 | 53,143 |
| 750~2000 | 68,150 | 67,237 | 62,385 | 72,253 | 75,090 | 77,410 | 88,348 | 67,458 |
| 2000~5000 | 45,083 | 46,480 | 44,830 | 55,350 | 54,491 | 54,984 | 72,939 | 54,362 |
| 5000~10000 | 24,569 | 26,135 | 24,668 | 29,391 | 28,462 | 25,864 | 40,145 | 27,250 |
| 10000~20000 | 17,330 | 16,491 | 17,147 | 16,447 | 15,614 | 16,340 | 26,004 | 17,490 |
| 20000~ | 21,620 | 20,030 | 23,629 | 18,215 | 13,663 | 14,353 | 23,218 | 12,777 |

거래가 있는 v2 칸 중 네이버 값이 없어 U 밖으로 나가는 비중: {'kiwoom_raw': 0.00484, 'naver_adj': 0.01848}

이름 거르개 제외 68종목(`name_filter.csv`). 메리츠 포함 여부: False.

**정정 SC-2:** 위 표에 더해 수정가/원가 < 2% 칸은 U 밖이다(탐색 절반 1종목 484칸, 2단계 `D.md`). 이 파일은 OHLC 를 읽지 않아 그 칸을 빼지 않고 센다.

## 생존편향·원천

| 항목 | 값 |
|---|---|
| KIND 2009-01~2016-08 폐지 대비 v2 | 504건 중 502 (2015·2016 각 1건 없음) |
| `delisted_stocks.last_trade_date` | 4,213행 중 489행만 값이 있고 이 구간은 0건 — 대조에 못 쓴다 |
| 지수 001·101 | 2006~2016 거래일 전부(연 245~253행) |
| `earnings_dart_doc` | 2007~2015 (파서 바 97.89% 통과). 신호에 안 쓴다. 주식수만 시총 팩터에 쓴다(as-of, 접수 다음날) |

## 호가·분봉 (비용 대조용)

| 표 | 범위 | 2026-08-31 주 대역 759종목과 겹침 |
|---|---|---|
| `minute_bars` | 2025-08-01~ | 688 (91%) |
| `quotes`·`ticks` | 2026-08-24~ | 94 (12%, scalp-it 수집 대상이라 편향 표본) |
| `quote_events` | 2026-09-16~ | 그 주 0 |

## 쿼리 전문 (읽기 전용, `set statement_timeout` 60~180초 — 앞쪽은 첫 판 조회, 그 값은 정정으로 대체됐다)

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
-- 정정 SC-1 재집계(research/experiments/smallcap_s1.py)
SELECT code, date, volume, source, close_src FROM daily_bars_adj_v2   -- read_prices 정문, 판다스에서 기간·보통주 거름
select code,name,market from stocks
select code,name,market from delisted_stocks
select code,name,reason from kind_delistings
select code,date,close*volume/1e6 as tv from naver_investor_trend where date between '2007-11-01' and '2016-08-31' and close is not null
```
