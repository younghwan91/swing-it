"""탐색(판정 아님): 미너비니 청산 'VCP 끔' 339 트레이드 — 손절폭 d(수렴 정도)·상대 수렴·수급 F·뉴스 N 별 손익 모양."""
import os, sys, time
import numpy as np, pandas as pd
ROOT = "/home/young/git/swing-it-feature-combined-strategy"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/research/experiments"); os.chdir(ROOT)
from prop_swing_common import load_env_db
load_env_db()
from sepa_confluence_gate import Data
from swing_it.storage import connect, db_default
from swing_it.news.daily_leg import build_daily, n_flag_panel, n_flag_values
con = connect(db_default()); D = Data(con, time.time()); p = D.p
t = pd.read_csv("research/logs/sepa_explore/minervini_exits_trades.csv", dtype={"code": str})
t = t[t["cell"] == "VCP 끔"].copy()
t["event_day"] = pd.to_datetime(t["event_day"])
di = p.days.get_indexer(t["event_day"]); ci = p.close.columns.get_indexer(t["code"])
t["F"] = D.pick(D.flow["F"], di, ci)
# 상대 수렴: 직전 10일 고저폭 / 그 종목 직전 250일의 10일 고저폭 중앙값
rng10 = p.high.rolling(10).max().shift(1) / p.low.rolling(10).min().shift(1) - 1
rel = rng10 / rng10.rolling(250, min_periods=120).median().shift(10)
t["rng10"] = D.pick(rng10, di, ci); t["rel"] = D.pick(rel, di, ci)
import datetime as dt
nw = t[t["event_day"] >= "2025-09-01"]
daily = build_daily(con, lo=dt.datetime(2025, 6, 1), hi=dt.datetime(2026, 10, 1), mode="close", trading_days=[d.date() for d in p.days])
npan = n_flag_panel(daily, sorted({d.date() for d in nw["event_day"]}), codes=set(nw["code"]))
npan["N"] = n_flag_values(npan)
npan["event_day"] = pd.to_datetime(npan["date"])
t = t.merge(npan[["code", "event_day", "N"]], on=["code", "event_day"], how="left")
def summ(g):
    loss = g["net"][g["net"] < 0]; win = g["net"][g["net"] > 0]
    return pd.Series({"n": len(g), "mean": g["net"].mean(), "median": g["net"].median(), "win": (g["net"] > 0).mean(),
                      "stop%": (g["exit_reason"] == "stop").mean(), "avg_loss": loss.mean(), "avg_gain": win.mean(),
                      "G/L": win.mean() / -loss.mean() if len(loss) and len(win) else np.nan,
                      "worst10%": g["net"].quantile(0.1), "R": g["R"].mean()})
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
print("전체"); print(summ(t).to_frame().T)
print("\n손절폭 d 3분위"); print(t.groupby(pd.qcut(t["d"].rank(method="first"), 3, labels=["좁음", "중간", "넓음"])).apply(summ))
print("\n상대 수렴 rel(직전10일폭/평소) 3분위"); print(t.dropna(subset=["rel"]).groupby(pd.qcut(t["rel"].dropna(), 3, labels=["강수렴", "중간", "확대"])).apply(summ))
print("\n수급 F (2024-02-19+)"); s = t[t["event_day"] >= "2024-02-19"]; print(s.groupby("F", dropna=False).apply(summ))
print("\n뉴스 N (2025-09+)"); s = t[t["event_day"] >= "2025-09-01"]; print(s.groupby("N", dropna=False).apply(summ))
print("\nF=1 ∧ 강수렴(rel 하위 1/3), 2024-02-19+"); q = t["rel"].quantile(1/3)
s = t[(t["event_day"] >= "2024-02-19")]; print(s.groupby([s["F"], s["rel"] <= q]).apply(summ))
