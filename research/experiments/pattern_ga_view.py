"""패턴 GA 열람 도구 — 런 원장(시드 최적)에서 절차 출력·표본 밖 매매를 다시 세우고, 차트 페이지(거래량 포함)와 기술 표를 만든다.

판정에 쓰지 않는다(기술). 러너(:mod:`pattern_ga_run`)의 ``procedure`` → ``stitched`` → ``trade_frame`` → ``chart_points`` 를 그대로 쓴다.
변형은 ``--variant`` 로 고르고, 러너 import 전에 ``PATTERN_GA_VARIANT`` 를 정한다(모듈 상수가 import 시점에 정해진다).

추가 기술(사전등록 ``PATTERN_GA_FWM.md`` 정정 FWM-1 M5·M7·M7 기사 민감도):

- **−5% 초과 손실**: 첫 단위 가격 손익(청산가/진입가 − 1) < −5.5%. 원인은 하나로, 진입일 손절 → 시가 갭(청산이 시가) → 추가분(불타기 있음) → 기타 순.
- **열람 표시**: 열람 목록(종목, 주문일)과 같은 종목 ± ``VIEW_TOL`` 거래일이면 ``viewed``.
- **기사(2023-04-01+ 주문, 판정 자료 끝 2026-02-27)**: 그 종목(판정 행의 ``code``) 기사의 LLM 판정(``article_judgments`` judge ``v2r3-haiku55``, subject main) 중
  주문일 전 7일 ~ 주문일 08:30 KST 에 발행된 것 — 부정(sentiment_direction < 0) 있음/없음. 판정 모델이 사후 지식을 가졌을 수 있다(한계).

사용: ``uv run --with numba --extra pg python research/experiments/pattern_ga_view.py --variant fwm --cache <캐시> --ledger
research/logs/sepa_explore/pattern_ga_fwm_runs/TRIALS.jsonl --out <디렉터리> --title "…" --h1 "…" --lede "…"``
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_ga_view_template.html")
NEWS_JUDGE, NEWS_FROM, NEWS_TO = "v2r3-haiku55", "2023-04-01", "2026-02-27"
LOSS5 = -0.055
US_CSV = "research/logs/sepa_explore/regime/us_overnight.csv"   # yfinance 일봉 종가(2017~), 미국 날짜 — 한국 주문일 d 전 마지막 미국 거래일
US_DROP = -0.01                                                  # 기술 분해 문턱(결과 전 고정): 전날 밤 나스닥 선물 −1% 이하


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ledger", default="", help="런 원장 TRIALS.jsonl(시드 최적 'best' 줄)")
    ap.add_argument("--evo-json", default="", help="진화 후보 결과 JSON(pattern_ga_evo/<이름>.json) — 그 설정으로 폴드 1~4·봉인 그대로")
    ap.add_argument("--git-prefix", default="", help="이 git 앞자리 줄만(재시작 전 원장 섞임 방지)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="패턴 GA 결과")
    ap.add_argument("--h1", default="패턴 GA — 표본 밖 매매")
    ap.add_argument("--lede", default="")
    ap.add_argument("--no-news", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = _args()
    evo = None
    if a.evo_json:                                    # 진화 후보: 메뉴 덮어쓰기를 import 전에 같게 둔다
        evo = json.load(open(a.evo_json, encoding="utf-8"))
        a.variant = evo["variant"]
        os.environ["PATTERN_GA_MENU"] = json.dumps(evo["menu"])
    os.environ["PATTERN_GA_VARIANT"] = a.variant
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    sys.path[:0] = [".", "src"]
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    import research.experiments.pattern_ga_run as G
    from swing_it import storage
    from swing_it.validation.walkforward import FOLDS

    os.makedirs(a.out, exist_ok=True)
    cont = {n for n, *_ in G.CONT}
    levels = {n: v for n, v in G.NOM} | {n: v for n, v in G.DISC}

    def parse(b):
        g = {}
        for k, s in b.items():
            if k in cont:
                g[k] = None if s == "None" else float(s)
            elif k in levels:
                m = [lv for lv in levels[k] if str(lv) == s]
                if not m and s != "None":
                    m = [lv for lv in levels[k] if isinstance(lv, float) and math.isclose(lv, float(s))]
                g[k] = m[0] if m else None
        return g

    E = G.Engine(a.cache)
    if evo is not None:                               # 폴드 1~4 · 봉인(run_evolve 와 같은 자르기)
        wins = G.fold_windows(E, [FOLDS[k] for k in G.EVO_FOLDS])
        cfgs = evo["metrics"]["cfgs"]
        f_last = FOLDS[G.EVO_FOLDS[-1]]
        seal = E.row(f_last.test_hi)
        E.LIM = np.minimum(np.asarray(E.LIM), seal).astype(np.int64)
        E.end_row = seal - 1
        tests = [te for _, te in wins]
        tests[-1] = G.Win(E, f_last.test_lo, f_last.test_hi, f_last.test_hi)
    else:
        runs = {}
        for line in open(a.ledger, encoding="utf-8"):
            d = json.loads(line)["config"]
            if "run" in d and d.get("git_head", "").startswith(a.git_prefix):
                runs[d["run"]] = parse(d["best"])
        wins = G.fold_windows(E, FOLDS)
        need = [f"fold{k}/seed{s}" for k in range(len(wins)) for s in G.SEEDS]
        miss = [r for r in need if r not in runs]
        if miss:
            raise SystemExit(f"원장에 런 {len(miss)}개 없음(예: {miss[:3]}) — 실제 GA 가 끝나지 않았다")
        bests = [[runs[f"fold{k}/seed{s}"] for s in G.SEEDS] for k in range(len(wins))]
        cfgs = G.procedure(E, wins, bests)["cfgs"]
        tests = [te for _, te in wins]
    st = E.stitched(cfgs, tests)
    if evo is not None and abs(st["ann"] - evo["metrics"]["ann"]) > 1e-9:
        raise SystemExit(f"진화 결과 재현 실패: {st['ann']} vs {evo['metrics']['ann']}")
    t = G.trade_frame(E, st, cfgs)
    G.CHART_N = len(t)
    pts = G.chart_points(E, t, cfgs)
    if evo is not None:                               # 검증 MUST-FIX 5: 봉인 너머 봉이 페이지에 들어가면 중단
        last_bar = max((b["date"] for c in pts for b in c["bars"]), default="")
        if last_bar >= str(FOLDS[G.EVO_FOLDS[-1]].test_hi)[:10]:
            raise SystemExit(f"봉인 누출: 차트 봉 {last_bar}")
    p = E.pmix
    t["mix"] = G.pmix_net(t, p)
    t["R"] = t["mix"] / (t["wb"] * t["d"])
    t["px_ret"] = t["exit_px"] / t["entry_px"] - 1.0
    t["loss5"] = t["px_ret"] < LOSS5
    cause = np.where(t["x_row"] == t["d_row"], "진입일", np.where(t["exit_open"], "시가 갭", np.where(t["added"], "추가분", "기타")))
    t["loss5_cause"] = np.where(t["loss5"], cause, "")

    # 열람 표시
    t["viewed"] = False
    if G.FW and os.path.exists(G.FW_PARENT_VIEWED):
        pv = pd.read_csv(G.FW_PARENT_VIEWED, dtype={"code": str})
        tol = getattr(G, "VIEW_TOL", 0) if G.FWM else 0
        vk = {}
        for c, d in zip(pv["code"], pv["order_day"], strict=True):
            vk.setdefault(c, []).append(E.row(d))
        t["viewed"] = [any(abs(int(r) - v) <= tol for v in vk.get(c, ())) for c, r in zip(t["code"], t["d_row"], strict=True)]

    con = storage.connect(storage.db_default())
    cur = con.cursor()
    codes = sorted(set(t["code"]))
    cur.execute("SELECT code,name,market FROM stocks WHERE code = ANY(%s)", (codes,))
    nm = {c: (n, m) for c, n, m in cur.fetchall()}

    # 전날 밤 미국장(기술): 주문일 d 보다 앞선 마지막 미국 거래일의 종가 수익률(미국 장 마감 = 한국 05~06시, d 장 전에 안다)
    t["us_nq"], t["us_spx"], t["us_sox"] = np.nan, np.nan, np.nan
    if os.path.exists(US_CSV):
        us = pd.read_csv(US_CSV, parse_dates=["us_date"]).set_index("us_date")
        ur = us[["nq", "spx", "sox"]].dropna(how="all").pct_change()
        pos = ur.index.searchsorted(t["order_day"].to_numpy(), side="left") - 1
        ok = pos >= 1
        for col, src in (("us_nq", "nq"), ("us_spx", "spx"), ("us_sox", "sox")):
            vals = np.full(len(t), np.nan)
            vals[ok] = ur[src].to_numpy()[pos[ok]]
            t[col] = vals
        if "ewy_open" in us.columns:                  # 야간선물 무료 대용: EWY 미국 장중(시가→종가, 한국 22:30~05:00) 수익률
            ew = (us["ewy_close"] / us["ewy_open"] - 1.0).dropna()
            p2 = ew.index.searchsorted(t["order_day"].to_numpy(), side="left") - 1
            v2 = np.full(len(t), np.nan)
            v2[p2 >= 0] = ew.to_numpy()[p2[p2 >= 0]]
            t["us_ewy_oc"] = v2

    # 기사 오버레이(기술)
    t["news_neg"], t["news_pos"], t["news_n"] = np.nan, np.nan, np.nan
    if not a.no_news:
        sel = t[(t["order_day"] >= pd.Timestamp(NEWS_FROM)) & (t["order_day"] <= pd.Timestamp(NEWS_TO))]
        for i, r in sel.iterrows():
            lo = (r["order_day"] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            hi = r["order_day"].strftime("%Y-%m-%d") + " 08:30:00+09"
            cur.execute("""SELECT (j.output->>'sentiment_direction')::int FROM article_judgments j
                           JOIN news_articles n ON n.id = j.article_id
                           WHERE j.judge = %s AND j.code = %s AND n.published_at >= %s AND n.published_at < %s
                             AND j.output->>'subject' = 'main'""", (NEWS_JUDGE, r["code"], lo, hi))
            dirs = [x[0] for x in cur.fetchall() if x[0] is not None]
            t.at[i, "news_n"], t.at[i, "news_neg"], t.at[i, "news_pos"] = len(dirs), float(any(x < 0 for x in dirs)), float(any(x > 0 for x in dirs))

    def vol(code, d0, d1):
        cur.execute("SELECT date, volume FROM daily_bars WHERE code=%s AND date BETWEEN %s AND %s", (code, d0, d1))
        return {str(x): float(v) for x, v in cur.fetchall()}

    f = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x  # noqa: E731
    cards = []
    for c in pts:
        r = t[(t["code"] == c["code"]) & (t["order_day"].dt.strftime("%Y-%m-%d") == c["order_day"])].iloc[0]
        ci = int(r["ci"])
        v = vol(c["code"], c["bars"][0]["date"], c["bars"][-1]["date"])
        bars = []
        for b in c["bars"]:
            jx = E.days.searchsorted(pd.Timestamp(b["date"]))
            m50 = float(E.MA50[ci, jx])
            bars.append([b["date"], b["o"], b["h"], b["l"], b["c"], m50 if np.isfinite(m50) and m50 > 0 else None, v.get(b["date"])])
        n_, m_ = nm.get(c["code"], (c["code"], ""))
        cards.append(dict({k: x for k, x in c.items() if k != "bars"}, name=n_, market=m_, bars=bars, entry=str(r["entry_day"])[:10],
                          net=float(r["mix"]), R=float(r["R"]), reason=r["exit_reason"], hold=int(r["hold_days"]), d=float(r["d"]),
                          partial_px=f(float(r["partial_px"])) if r["partial"] else None, add_px=f(float(r["add_px"])) if r["added"] else None,
                          reentry=bool(r["reentry"]), mfe_R=float(r["mfe_R"]), loss5=bool(r["loss5"]), viewed=bool(r["viewed"])))
    rows = [dict(name=nm.get(r["code"], (r["code"], ""))[0], code=r["code"], fold=int(r["fold"]), order=str(r["order_day"])[:10],
                 exit=str(r["exit_day"])[:10], reason=r["exit_reason"], shape=r["shape"], d=float(r["d"]), net=float(r["mix"]), R=float(r["R"]),
                 mfe=float(r["mfe_R"]), hold=int(r["hold_days"]), added=bool(r["added"]), partial=bool(r["partial"])) for _, r in t.iterrows()]

    def stats(x):
        w, lz = x > 0.005, x < -0.005
        n_w, n_l = int(w.sum()), int(lz.sum())
        gl = float(x[w].mean() / -x[lz].mean()) if n_w and n_l else float("nan")
        return {"n": int(len(x)), "win": n_w / (n_w + n_l) if n_w + n_l else float("nan"), "gl": gl, "mean": float(x.mean()) if len(x) else float("nan")}

    x2 = t["net2"].where(~t["amb"].astype(bool), p * t["net2_stop0"] + (1 - p) * t["net2"])
    s_all, s2 = stats(t["mix"]), stats(x2)
    s_ex = stats(t.loc[~t["viewed"], "mix"])
    news = {}
    if not a.no_news:
        nn = t[t["news_n"].notna()]
        news = {"n": int(len(nn)), "with_news": int((nn["news_n"] > 0).sum()),
                "neg": stats(nn.loc[nn["news_neg"] > 0, "mix"]), "no_neg": stats(nn.loc[~(nn["news_neg"] > 0), "mix"])}
    l5 = t[t["loss5"]]
    usd = t["us_nq"] <= US_DROP
    us_split = {"down": stats(t.loc[usd, "mix"]), "rest": stats(t.loc[~usd & t["us_nq"].notna(), "mix"])}
    if "us_ewy_oc" in t.columns:                      # 기술 분해 문턱(결과 전 고정): EWY 장중 −1% 이하
        ewd = t["us_ewy_oc"] <= US_DROP
        us_split["ewy_down"] = stats(t.loc[ewd, "mix"])
        us_split["ewy_rest"] = stats(t.loc[~ewd & t["us_ewy_oc"].notna(), "mix"])
    summ = dict(n=len(t), win=s_all["win"], avgwin=float(t["mix"][t["mix"] > 0.005].mean()), avgloss=float(t["mix"][t["mix"] < -0.005].mean()),
                mean=s_all["mean"], Rmean=float(t["R"].mean()), ann=st["ann"], expo=float(np.mean(st["expo"])), fold_ann=st["fold_ann"],
                fold_n=st["fold_n"], folds=[[str(tr.lo), str(tr.hi), str(te.lo), str(te.hi)] for tr, te in wins],
                shapes=[G.config_shape(c) for c in cfgs],
                cfgs=[{k: (None if v is None else (str(v) if not isinstance(v, (int, float)) else v)) for k, v in c.items()} for c in cfgs],
                reasons=t["exit_reason"].value_counts().to_dict(), shape_counts=t["shape"].value_counts().to_dict(),
                add_rate=float(t["added"].mean()), cost2=s2, ex_viewed=s_ex, n_viewed=int(t["viewed"].sum()), news=news,
                us=us_split, loss5={"n": int(len(l5)), "causes": l5["loss5_cause"].value_counts().to_dict()},
                below50=float(np.mean([float(np.asarray(E.C)[int(r["ci"]), int(r["d_row"]) - 1] < np.asarray(E.MA50)[int(r["ci"]), int(r["d_row"]) - 1])
                                       for _, r in t.iterrows()])) if len(t) else float("nan"))
    t.drop(columns=[c for c in ("table",) if c in t.columns]).to_csv(os.path.join(a.out, "oos_trades.csv"), index=False)
    json.dump({"summary": summ}, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1, default=float)
    summ["notes"] = notes(summ)
    data = json.dumps(dict(summary=summ, cards=cards, rows=rows), ensure_ascii=False, separators=(",", ":"), default=float)
    h = open(TEMPLATE, encoding="utf-8").read()
    h = h.replace("/*__DATA__*/null", data).replace("__TITLE__", a.title).replace("__H1__", a.h1).replace("__LEDE__", a.lede).replace("__N__", str(len(t)))
    open(os.path.join(a.out, "page.html"), "w", encoding="utf-8").write(h)
    print(json.dumps({k: summ[k] for k in ("n", "win", "mean", "ann", "cost2", "ex_viewed", "n_viewed", "loss5", "news", "us", "below50")},
                     ensure_ascii=False, default=float), file=sys.stderr)


def _p(x, d=0):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def _sp(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:+.{d}f}%"


def notes(s: dict) -> list:
    """페이지 '먼저 보실 점' — 숫자만(판정 문구 없음)."""
    gl = s["avgwin"] / -s["avgloss"] if s["avgloss"] and s["avgloss"] < 0 else float("nan")
    out = [f"<b>거래 단위: 승률(본전 제외) {_p(s['win'])}, 손익비 {gl:.2f} : 1, 거래당 {_sp(s['mean'])}</b>(비용 1배). "
           f"비용 2배: 승률 {_p(s['cost2']['win'])}, 손익비 {s['cost2']['gl']:.2f}, 거래당 {_sp(s['cost2']['mean'])}.",
           f"<b>계좌 연환산(같은 모멘텀 종목 바구니 대비 초과, 비용 2배) {_sp(s['ann'])}</b>, 평균 노출 {_p(s['expo'])}.",
           f"<b>첫 단위 −5.5% 넘는 손실 {s['loss5']['n']}건</b> — 원인: " +
           (" · ".join(f"{k} {v}건" for k, v in s["loss5"]["causes"].items()) or "없음") + ".",
           f"전날 종가가 MA50 아래인 매수 {_p(s['below50'])}.",
           f"이미 차트로 본 거래(같은 종목 ±20거래일) {s['n_viewed']}건을 빼면: {s['ex_viewed']['n']}건, 승률 {_p(s['ex_viewed']['win'])}, "
           f"손익비 {s['ex_viewed']['gl']:.2f}, 거래당 {_sp(s['ex_viewed']['mean'])}."]
    r = s["reasons"]
    out.append("청산: " + " · ".join(f"{k} {v}건" for k, v in r.items()) + f". 불타기 발동 {_p(s['add_rate'])}.")
    u = s.get("us") or {}
    if u.get("down", {}).get("n") is not None:
        out.append(f"전날 밤 나스닥 선물 −1% 이하인 날 주문 {u['down']['n']}건: 승률 {_p(u['down']['win'])}, 거래당 {_sp(u['down']['mean'])} · "
                   f"나머지 {u['rest']['n']}건: 승률 {_p(u['rest']['win'])}, 거래당 {_sp(u['rest']['mean'])} (기술).")
    if u.get("ewy_down", {}).get("n") is not None:
        out.append(f"야간 대용(EWY 미국 장중) −1% 이하인 밤 다음 주문 {u['ewy_down']['n']}건: 거래당 {_sp(u['ewy_down']['mean'])} · "
                   f"나머지 {u['ewy_rest']['n']}건: 거래당 {_sp(u['ewy_rest']['mean'])} (기술).")
    nw = s.get("news") or {}
    if nw.get("n"):
        out.append(f"기사(2023-04 ~ 2026-02 주문 {nw['n']}건, 주문 전 7일 기사 있음 {nw['with_news']}건): 부정 기사 있음 {nw['neg']['n']}건 거래당 "
                   f"{_sp(nw['neg']['mean'])} · 없음 {nw['no_neg']['n']}건 거래당 {_sp(nw['no_neg']['mean'])} (기술, 판정 모델 사후 지식 가능).")
    return out


if __name__ == "__main__":
    main()
