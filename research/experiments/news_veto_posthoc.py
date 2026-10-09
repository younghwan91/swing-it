"""사후 관찰(판정 아님): 뼈대 S 트레이드(2025-09+)에서 진입 직전 악재 뉴스 유무별 성적 — `NEWS_VETO.md`."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from swing_it.news.daily_leg import build_daily, load_trading_days
from swing_it.storage import connect, db_default, load_env_db

KST = dt.timezone(dt.timedelta(hours=9))


def main() -> int:
    load_env_db()
    con = connect(db_default())
    days = load_trading_days(con)
    daily = build_daily(con, lo=dt.datetime(2025, 8, 1, tzinfo=KST), hi=dt.datetime(2026, 10, 1, tzinfo=KST),
                        mode="close", trading_days=days)
    daily["date"] = pd.to_datetime(daily["date"])
    t = pd.read_csv("research/logs/sepa_confluence/s_trades.csv", dtype={"code": str})
    t["event_day"] = pd.to_datetime(t["event_day"])
    t = t[t["event_day"] >= "2025-09-01"].copy()

    def neg_in(code: str, d: pd.Timestamp, k: int) -> tuple[int, int]:
        lo = d - pd.tseries.offsets.BDay(k)
        s = daily[(daily.code == code) & (daily.date > lo) & (daily.date <= d)]
        return int(s["n_neg"].sum()), int(s["n_articles"].sum())

    for k in (5, 20):
        t[f"neg{k}"], t[f"art{k}"] = zip(*[neg_in(c, d, k) for c, d in zip(t.code, t.event_day)], strict=True)

    def summ(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"n": len(g), "excess": g.excess.mean(), "net": g.net.mean(),
                          "win": (g.net > 0).mean(), "stop": (g.exit_reason == "stop").mean()})

    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    print("S 트레이드 2025-09+:", len(t))
    for k in (5, 20):
        print(f"\n직전 {k}거래일 악재 유무")
        print(t.groupby(t[f"neg{k}"] > 0).apply(summ))
    print("\n악재 건수별(20일)")
    print(t.groupby(np.minimum(t.neg20, 3)).apply(summ))
    print("\n판정 기사 없음 vs 있음(20일)")
    print(t.groupby(t.art20 > 0).apply(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
