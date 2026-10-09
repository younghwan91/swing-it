#!/usr/bin/env python
"""시황 기사 기분 지수 판정 실행 — 적응 층 B (``research/logs/sepa_explore/ADAPTIVE.md`` §1-B, 설계 고정 후 자료 준비).

:mod:`swing_it.news.market_mood` 로 (1) 신호일별 시황 기사 ≤ 10건을 고르고 (2) Haiku 5.5 로 판정해 원장(``article_judgments``,
judge ``market-mood-haiku55``)에 쓰고 (3) 일별 평균·20일 EMA 를 CSV 로, 요약을 ``MOOD_PILOT.md`` 에 적는다.
**수익·트레이드 결과는 계산하지 않는다** — 표의 KOSPI 일수익률은 판정 품질을 눈으로 볼 때 옆에 두는 기술 열이지 검정이 아니다.

두 모드: ``--mode messages`` (파일럿 — 비배치, 즉시 결과) · ``--mode batch`` (전 구간 — Message Batches, 50% 할인, 재개 가능).
비용 가드: 추정이 ``--hard-cap-usd`` 를 넘으면 아무것도 보내지 않는다. 키가 없으면 선택·추정까지만 하고 멈춘다(``--dry-run`` 과 같다).

Run:
  파일럿   .venv/bin/python research/experiments/market_mood_run.py --start 2026-07-01 --end 2026-09-26 --mode messages \\
             --budget-usd 1 --hard-cap-usd 2 --section 파일럿
  전 구간  ... --start 2023-03-15 --end 2026-10-09 --mode batch --hard-cap-usd 12 --section "전 구간" --run full
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swing_it.news import market_mood as mm  # noqa: E402
from swing_it.storage import connect, db_default, load_env_db  # noqa: E402

OUT_DIR = Path("research/logs/sepa_explore/regime")
PANEL_CSV = OUT_DIR / "regime_panel.csv"
FULL_LO, FULL_HI = "2023-03-15", "2026-10-09"   # 기사 DB 시작 ~ 현재(전 구간 비용 추정 창)


def trading_days(con) -> list[dt.date]:
    """거래일 달력 — 국면 패널 CSV 가 있으면 그 날짜(= daily_bars_adjusted 거래일), 없으면 정문으로 읽는다."""
    if PANEL_CSV.exists():
        d = pd.read_csv(PANEL_CSV, usecols=["date"])["date"].unique()
        return sorted(pd.to_datetime(d).date)
    from swing_it.news.daily_leg import load_trading_days  # noqa: PLC0415

    return load_trading_days(con)


def kospi_returns(con) -> pd.Series:
    """sector_index 001 종가 일수익률(기술 열) — 2024-01-26 부터."""
    cur = con.cursor()
    cur.execute("SELECT date, close FROM sector_index WHERE code = '001' ORDER BY date")
    s = pd.Series({pd.Timestamp(d): float(c) for d, c in cur.fetchall()}).sort_index()
    return s.pct_change()


def select_window(con, days: list[dt.date], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lo = dt.datetime.combine(pd.Timestamp(start).date() - dt.timedelta(days=4), dt.time(0), tzinfo=mm.KST)
    hi = dt.datetime.combine(pd.Timestamp(end).date(), dt.time(15, 30), tzinfo=mm.KST)
    arts = mm.load_market_articles(con, lo, hi)
    sel = mm.select_market_articles(arts, days)
    d0, d1 = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    sel = sel[(sel["signal_day"] >= d0) & (sel["signal_day"] <= d1)].reset_index(drop=True)
    return arts, sel


def md_table(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for i, r in df.iterrows():
        cells = ["" if pd.isna(v) else (fmt.format(v) if isinstance(v, float) else str(v)) for v in r]
        rows.append("| " + " | ".join([str(getattr(i, "date", lambda: i)())] + cells) + " |")
    return "\n".join([head, sep, *rows])


def full_history_estimate(con, days: list[dt.date], per_request_usd: "float | None") -> dict:
    """전 구간(2023-03-15~) 선택 건수와 비용 추정 — 글자수 추정과 파일럿 실측 단가 둘 다."""
    _, sel = select_window(con, days, FULL_LO, FULL_HI)
    recs = sel.to_dict("records")
    est_batch = mm.estimate_usd(recs, batch=True, cache_hit_rate=0.5)
    est_msg = mm.estimate_usd(recs, batch=False)
    out = {"requests": len(sel), "days": int(sel["signal_day"].nunique()), "est_batch_usd": est_batch["usd"],
           "est_messages_usd": est_msg["usd"]}
    if per_request_usd is not None:
        out["measured_rate_messages_usd"] = per_request_usd * len(sel)
        out["measured_rate_batch_usd"] = per_request_usd * len(sel) * 0.5
    return out


def write_section(path: Path, section: str, body: str) -> None:
    """``## {section}`` 절을 교체(없으면 추가). 파일이 없으면 제목부터 쓴다."""
    header = "# LLM 시장 기분 지수 — 판정 실행 기록 (자료 준비, 결과 분석 아님)\n\n"
    text = path.read_text(encoding="utf-8") if path.exists() else header
    marker = f"## {section}\n"
    if marker in text:
        pre, rest = text.split(marker, 1)
        nxt = rest.find("\n## ")
        text = pre + marker + body + ("\n" + rest[nxt + 1:] if nxt >= 0 else "")
    else:
        text = text.rstrip("\n") + "\n\n" + marker + body
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--mode", choices=("messages", "batch"), default="messages")
    ap.add_argument("--budget-usd", type=float, default=1.0, help="비배치: 실측 누계가 넘으면 멈춘다")
    ap.add_argument("--hard-cap-usd", type=float, default=2.0, help="추정이 넘으면 아무것도 보내지 않는다")
    ap.add_argument("--section", default="파일럿", help="MOOD_PILOT.md 절 제목")
    ap.add_argument("--run", default="pilot", help="배치 상태 파일 이름(judge_runs/market-mood-haiku55/<run>.state.json)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--table-days", type=int, default=20)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load_env_db()
    con = connect(db_default())
    days = trading_days(con)
    arts, sel = select_window(con, days, args.start, args.end)
    per_day = sel.groupby("signal_day").size()
    n_days = len([d for d in days if pd.Timestamp(args.start).date() <= d <= pd.Timestamp(args.end).date()])
    print(f"candidates {len(arts):,} → market titles {int(arts['title'].map(mm.is_market_title).sum()):,} → "
          f"selected {len(sel):,} over {len(per_day)}/{n_days} trading days "
          f"(per day min {per_day.min() if len(per_day) else 0}, median {per_day.median() if len(per_day) else 0})",
          flush=True)
    already = {k: v for k, v in mm.existing_outputs(con).items() if k in set(sel["key"])}
    key = mm.api_key()
    dry = args.dry_run or key is None
    if key is None:
        print("ANTHROPIC_API_KEY 없음(환경변수·../quant-airflow/.env) — 선택·추정까지만 하고 API 는 부르지 않는다")
    info: dict
    if args.mode == "messages":
        if dry:
            todo = [r for r in sel.to_dict("records") if r["key"] not in already]
            info = {"n_selected": len(sel), "n_reused": len(already), "n_todo": len(todo),
                    "estimate": mm.estimate_usd(todo, batch=False), "n_judged": 0, "n_ok": 0, "n_error": 0,
                    "usage": {}, "measured_usd": 0.0, "dry_run": True}
            outputs = dict(already)
            print(f"DRY RUN: to judge {len(todo)} estimate ${info['estimate']['usd']:.4f}")
        else:
            ledger = mm.ledger_for(con, sel["id"])
            outputs, info = mm.judge_articles(sel, mm.messages_judge(key), ledger, budget_usd=args.budget_usd,
                                              hard_cap_usd=args.hard_cap_usd, already=already)
    else:
        import httpx  # noqa: PLC0415

        rd = mm.run_dir()
        rd.mkdir(parents=True, exist_ok=True)
        ledger = None if dry else mm.ledger_for(con, sel["id"])
        with httpx.Client() as client:
            judge = mm.MarketBatchJudge(client=client, api_key=key, model=mm.MODEL, cache=rd / "judgments.jsonl",
                                        errors=rd / "errors.jsonl", state=rd / f"{args.run}.state.json",
                                        budget_usd=args.hard_cap_usd, ledger=ledger)
            outputs, info = judge.run_articles(sel, hard_cap_usd=args.hard_cap_usd, already=already, dry_run=dry)
        info["dry_run"] = dry
    daily = mm.daily_mood(sel, outputs)
    win_days = [d for d in days if pd.Timestamp(args.start).date() <= d <= pd.Timestamp(args.end).date()]
    idx = mm.mood_index(daily, win_days)
    kr = kospi_returns(con)
    idx["kospi_ret"] = kr.reindex(idx.index)
    tag = "mood_daily" if args.mode == "batch" else "mood_pilot_daily"
    idx.to_csv(out / f"{tag}.csv", float_format="%.4f")
    # 이 집계의 행 d 는 세션 d−1 의 마감 시황이다(REVIEW_REGIME_DATA.md M1) — 신호용은 mood_avail_run.py 의 mood_avail_daily.csv.
    csv = out / f"{tag}.csv"
    csv.write_text("# 세션 d−1 묘사, 신호에 쓰지 말 것 — 신호용은 mood_avail_daily.csv (REVIEW_REGIME_DATA.md M1)\n"
                   + csv.read_text(encoding="utf-8"), encoding="utf-8")
    n_judged = info.get("n_judged", info.get("n_ok", 0) + info.get("n_error", 0))
    valid = (info["n_ok"] / n_judged) if n_judged else float("nan")
    per_req = (info["measured_usd"] / n_judged) if n_judged and info["measured_usd"] else None
    if args.mode == "batch" and per_req is not None:
        per_req *= 2  # 배치 단가 → 비배치 환산(아래 추정은 둘 다 적는다)
    full = full_history_estimate(con, days, per_req)
    con.close()

    # --- MD ---
    judged_days = int((daily["n_judged"] > 0).sum()) if len(daily) else 0
    year_cov = (idx.assign(y=idx.index.year).groupby("y")["n_judged"].apply(lambda s: float((s > 0).mean())).round(3)
                if len(idx) else pd.Series(dtype=float))
    cols = ["n_judged", *mm.FIELDS, "mood_score", "kospi_ret"]
    tail = idx[cols].dropna(subset=["market_direction"]).tail(args.table_days).copy()
    tail["kospi_ret"] = tail["kospi_ret"] * 100
    tail = tail.rename(columns={"n_judged": "n", "market_direction": "dir", "risk_appetite": "risk",
                                "foreign_flow_tone": "frgn", "breadth_tone": "brdth", "event_risk": "event",
                                "mood_score": "score", "kospi_ret": "KOSPI %"})
    usage = info.get("usage", {})
    body = "\n".join([
        f"실행 {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · 모드 {args.mode} · 구간 {args.start}~{args.end} · "
        f"`research/experiments/market_mood_run.py` · 소요 {time.time() - t0:.0f}s" + (" · **DRY RUN**" if dry else ""),
        "",
        f"- 후보 기사(DB 제목 패턴) {len(arts):,} → 종목 기사 태그 제외·중복 제거·일 10건 → **선택 {len(sel):,}건 / "
        f"거래일 {len(per_day)}/{n_days}** (일 최소 {per_day.min() if len(per_day) else 0}, 중앙값 "
        f"{per_day.median() if len(per_day) else 0:.0f}, 최대 {per_day.max() if len(per_day) else 0})",
        f"- 판정: 재사용 {info.get('n_reused', 0)} · 새로 {n_judged} (성공 {info.get('n_ok', 0)}, 실패 {info.get('n_error', 0)}"
        f"{', 미판정 ' + str(info['n_unjudged']) if info.get('n_unjudged') else ''}) · 유효 JSON 비율 "
        f"{'' if n_judged == 0 else f'{valid:.1%}'} · 판정된 거래일 {judged_days}",
        f"- 비용: 추정 ${info['estimate']['usd']:.4f} (글자수 기반, 보수적) · **실측 ${info['measured_usd']:.4f}** "
        f"(usage: " + ", ".join(f"{k}={v:,}" for k, v in usage.items()) + ")"
        + (f" · 배치 {[b['id'] for b in info.get('batches', [])]}" if info.get("batches") else ""),
        "- 연도별 판정 거래일 비율: " + ", ".join(f"{y}: {v:.2f}" for y, v in year_cov.items()),
        f"- 원장: judge `{mm.JUDGE}`, code `{mm.CODE}` — {json.dumps(info.get('ledger'), ensure_ascii=False)}",
        f"- 전 구간({FULL_LO}~{FULL_HI}) 추정: 선택 {full['requests']:,}건 / {full['days']:,}일 → 글자수 추정 배치 "
        f"${full['est_batch_usd']:.2f} · 비배치 ${full['est_messages_usd']:.2f}"
        + (f" · **실측 단가 환산 배치 ${full['measured_rate_batch_usd']:.2f} · 비배치 ${full['measured_rate_messages_usd']:.2f}**"
           if "measured_rate_batch_usd" in full else ""),
        f"- 산출: `{tag}.csv` (거래일 × 일별 평균 5필드 + mood_score + 20일 EMA + KOSPI 일수익률[기술 열])",
        "",
        f"### 마지막 {len(tail)}거래일 일별 평균 (기술 — 검정 아님; KOSPI % 는 sector_index 001 종가 일수익률)",
        "",
        md_table(tail.rename_axis("date")),
        "",
    ])
    write_section(out / "MOOD_PILOT.md", args.section, body)
    print(json.dumps({k: v for k, v in info.items() if k != "batches"}, ensure_ascii=False, default=str, indent=1))
    print(f"full-history estimate: {full}")
    print(f"wrote {out / f'{tag}.csv'} and MOOD_PILOT.md [{args.section}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
