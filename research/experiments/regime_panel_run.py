#!/usr/bin/env python
"""시장 국면 패널 생성 — 적응 층 A·C (``research/logs/sepa_explore/ADAPTIVE.md`` §1, 설계 고정 후 자료 준비).

:mod:`swing_it.features.regime` 로 날짜×시장(KOSPI·KOSDAQ·ALL) 패널을 만들어 저장한다. **수익·트레이드 결과는 계산하지
않는다**(자료 준비 단계 — 국면 신호의 단독 가치는 ADAPTIVE.md §3 절차로 따로 잰다).

산출물(``research/logs/sepa_explore/regime/``):
- ``regime_panel.csv`` (gitignore — *.csv) · ``regime_panel.npz`` (같은 내용, 압축) · ``REGIME_PANEL.md`` (연도별 커버리지·보정 통계).

자원: GA 가 12코어를 쓰는 동안 돌리므로 BLAS 스레드 2개(아래 환경변수)·메모리 ≤ 4GB(긴 가격 프레임은 열별 피벗 뒤 바로 버린다).

Run:  OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python research/experiments/regime_panel_run.py [--start 2016-09-09] [--end ...]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from swing_it.features.regime import (  # noqa: E402
    FEATURES,
    INDEX_ATV_WINDOW,
    INDEX_CAP,
    INDEX_MIN_CORR,
    MARKETS,
    MIN_INDEX_N,
    OFFICIAL_SINCE,
    RET_LIMIT,
    RW_QUANTILE,
    RW_SEED,
    RW_SIMS,
    build_regime_panel,
    coverage_by_year,
    index_sanity,
    load_regime_inputs,
    rw_threshold_table,
    tv_weighted_index,
)
from swing_it.storage import connect, db_default, load_env_db  # noqa: E402

OUT_DIR = Path("research/logs/sepa_explore/regime")


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def save_npz(panel: pd.DataFrame, path: Path) -> None:
    """긴 프레임 → npz(날짜 int64[ns]·시장 코드·피처 행렬 float64) — pyarrow 없이도 읽는 압축본."""
    np.savez_compressed(
        path,
        date=pd.to_datetime(panel["date"]).to_numpy().astype("datetime64[ns]").astype(np.int64),
        market=panel["market"].map({m: i for i, m in enumerate(MARKETS)}).to_numpy(dtype=np.int8),
        markets=np.array(MARKETS),
        features=np.array(FEATURES),
        values=panel[list(FEATURES)].to_numpy(dtype=float),
    )


def md_table(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    """DataFrame → 마크다운 표(tabulate 없이). index 가 첫 열."""
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = []
    for i, r in df.iterrows():
        cells = ["" if pd.isna(v) else (str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() and abs(v) >= 1
                 else fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)) for v in r]
        rows.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join([head, sep, *rows])


def official_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    """2024-01-26 이후 연도별 대용·공식 누적 수익과 일별 상관(시장별) — 대조 기술(수익 분석 아님: 지수 대용이 지수를 재현하는지)."""
    rows = []
    for m in ("KOSPI", "KOSDAQ"):
        p = panel[panel["market"] == m].set_index("date")[["index_ret", "official_ret"]].dropna()
        p = p[p.index >= pd.Timestamp(OFFICIAL_SINCE)]
        for y, g in p.groupby(p.index.year):
            rows.append({"market_year": f"{m} {y}", "n": len(g), "corr": g["index_ret"].corr(g["official_ret"]),
                         "proxy_cum_%": 100 * ((1 + g["index_ret"]).prod() - 1),
                         "official_cum_%": 100 * ((1 + g["official_ret"]).prod() - 1)})
    return pd.DataFrame(rows).set_index("market_year")


#: 진단용 대안 가중(창, 상한) — 고정된 판(20일·10%)이 공식 지수를 얼마나 따라가는지 옆에 둔다. 판정이 아니라 기술.
VARIANTS = ((INDEX_ATV_WINDOW, INDEX_CAP), (20, 1.0), (60, INDEX_CAP), (120, INDEX_CAP))


def variant_table(inp, official: pd.DataFrame) -> pd.DataFrame:
    """대안 가중별 2024-01-26+ 공식 지수 대조(상관·누적) — 지수 재현 품질 기술(수익 분석 아님)."""
    off = official.pct_change(fill_method=None)
    rows = []
    for w, cap in VARIANTS:
        r = tv_weighted_index(inp.close, inp.trade_value, inp.market, window=w, cap=cap)["index_ret"]
        row = {"variant": f"ATV{w} cap {cap:.0%}" + (" (고정 판)" if (w, cap) == VARIANTS[0] else "")}
        for m in ("KOSPI", "KOSDAQ"):
            d = pd.concat([r[m], off[m]], axis=1, keys=["p", "o"]).dropna()
            d = d[d.index >= pd.Timestamp(OFFICIAL_SINCE)]
            row[f"{m} corr"] = d["p"].corr(d["o"])
            row[f"{m} cum % (공식)"] = f"{100 * ((1 + d['p']).prod() - 1):+.1f} ({100 * ((1 + d['o']).prod() - 1):+.1f})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def write_summary(panel: pd.DataFrame, info: dict, path: Path, *, elapsed: float, sanity: pd.DataFrame,
                  variants: "pd.DataFrame | None" = None) -> None:
    cov = coverage_by_year(panel)
    last = panel[panel["market"] == "ALL"].tail(1).iloc[0]
    spec_rows = {}
    for m in MARKETS:
        a = panel[panel["market"] == m]
        for pre, name in (("idx", "지수 대용"), ("br", "breadth_ma50")):
            n_spec = int(a[f"{pre}_share"].notna().sum())
            n_val = int(a[f"{pre}_period"].notna().sum())
            spec_rows[f"{m} {name}"] = {"주기도 계산일": n_spec, "유효일": n_val,
                                        "유효 %": 100 * n_val / max(n_spec, 1),
                                        "share 중앙값": float(a[f"{pre}_share"].median()),
                                        "share 최대": float(a[f"{pre}_share"].max())}
    spec_tab = pd.DataFrame.from_dict(spec_rows, orient="index").rename_axis("계열")
    rw = rw_threshold_table()
    san = sanity.copy()
    for c in ("ok_weight", "ok_ret", "ok_corr"):
        san[c] = san[c].map({True: "충족", False: "**미달**"})
    lines = [
        "# 시장 국면 패널 — 생성 요약 (자료 준비, 결과 분석 아님)",
        "",
        f"생성: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · 소요 {elapsed:.0f}s · 최대 RSS {_rss_gb():.2f} GB · "
        f"`research/experiments/regime_panel_run.py`",
        "",
        f"- 행: {len(panel):,} (날짜 {panel['date'].nunique():,} × 시장 {len(MARKETS)}), 구간 {panel['date'].min().date()} ~ "
        f"{panel['date'].max().date()}",
        f"- 2026-09-14~ 정규장 보정(krx_session): 행 {info.get('nxt_rows', 0):,} 중 {info.get('nxt_patched', 0):,} 보정",
        "- 정의: `src/swing_it/features/regime.py` 모듈 독스트링(전부 d 종가까지 자료, 뒤쪽 창만).",
        "- `flow20` 은 2024-02 이후만(그 전은 수급 수집이 연 ~650 종목). `official_*` 은 sector_index 001/101 이 있는 "
        "2024-01-26 이후(`official_vs_ma200` 은 + 200일).",
        "",
        "## 지수 대용 재구축 (2026-10-10, 리뷰 `REVIEW_REGIME_DATA.md` C1)",
        "",
        "- **이전 판(주식수 × 조정 종가 시총가중)은 틀렸다.** `daily_bars` 의 \"원\" 가격 일부가 이미 분할조정돼 있고 DART 주식수 일부가 "
        "단위 오류(×1000)라, 삼성전자가 2017-04~2019-03 KOSPI 가중 ~0.8%(실제 ~25%), 태광산업이 2022-02~03 KOSPI 41%, KOSDAQ 소형주 몇 개가 "
        "최대 43~61% 를 가졌다(KOSDAQ 대용 2020 +114%, 공식 +44.6%). 이전 요약의 \"삼성전자 가중을 고쳤다\" 는 **사실이 아니었다** — "
        "주식수 환산은 `daily_bars` 가 진짜 원가격일 때만 맞는데 삼성전자 2017 행은 이미 조정돼 있었다. ±31% 절단도 그 종목의 수익만 지우고 "
        "가중은 남겨 증상(−73% 하루)만 가렸다. 그 판의 `index_*`·`rv20`·`idx_*` 는 이 파일·npz 에서 **뺐다**(같은 이름이 새 정의로 바뀌었다).",
        f"- **새 판:** 종목 가중 = 직전 {INDEX_ATV_WINDOW}거래일 평균 거래대금(d−1 까지, 주식수 불필요), 시장마다 단일 종목 ≤ {INDEX_CAP:.0%} "
        f"상한·재분배, 그날 |r| > {RET_LIMIT:.0%} 종목은 수익·가중 둘 다 제외. 대형주 상한 때문에 공식(유동시총) 지수보다 "
        "삼성전자·SK하이닉스 비중이 낮다 — 국면 신호로만 쓴다.",
        f"- **위생 바(테스트 `test_committed_panel_meets_index_sanity_bars` 가 이 npz 로 고정):** 매일 최대 단일 가중 ≤ {INDEX_CAP:.0%}, "
        f"|일수익률| ≤ {RET_LIMIT:.0%}, {OFFICIAL_SINCE}+ 공식 지수 일수익률과 상관 ≥ {INDEX_MIN_CORR}.",
        "",
        md_table(san.rename_axis("market"), fmt="{:.4f}"),
        "",
        "연도별 대조(공식 지수가 DB 에 있는 2024-01-26 이후만; 2017~2023 은 DB 에 공식 지수가 없어 대조하지 못했다):",
        "",
        md_table(official_by_year(panel), fmt="{:.3f}"),
        "",
        "**읽는 법:** 고정 판은 바(상관 ≥ 0.95)를 **간신히** 넘는다(위 표). 10% 상한이 삼성전자·SK하이닉스를 눌러 KOSPI 대용이 공식보다 덜 오르고, "
        "20일 거래대금 가중이 KOSDAQ 의 과열 테마주를 따라가 KOSDAQ 대용이 공식보다 크게 덜 오른다(연 KOSDAQ 2025 상관 0.88). "
        "국면 신호(200일선 대비·변동성·부호)로는 쓸 수 있지만 \"지수 수준\" 으로 읽으면 안 된다. 대안 가중 진단(같은 코드, 창·상한만 바꿈 — "
        "판을 바꾸려면 결과 보기 전에 사전등록에서 고른다):",
        "",
        md_table(variants, fmt="{:.4f}") if variants is not None else "(진단 생략)",
        "",
        "## 스펙트럼 유효성 (리뷰 M4)",
        "",
        f"- `*_period` 는 `*_share` 가 같은 길이 무작위 보행 {RW_SIMS:,}벌의 {RW_QUANTILE:.0%} 분위(시드 {RW_SEED}) **초과**인 날만 값, "
        "아니면 NaN. `*_share`·`*_haar*` 는 그대로 둔다(Haar 비율은 무작위 보행 귀무와 0.005 안 — 리뷰 M4, 정보 없음으로 본다).",
        "- 창 길이별 문턱: " + ", ".join(f"{w}일 {v:.4f}" for w, v in rw.items()),
        "- 512일 문턱이 낮은 것은 대역 상한(120일)보다 긴 주기 쪽에 무작위 보행 파워가 몰려 대역 안 봉우리의 몫이 작아지기 때문이다.",
        "",
        md_table(spec_tab, fmt="{:.3f}"),
        "",
        "무작위 보행이면 유효 % ≈ 5 다. 실제 계열의 `share` 는 대체로 무작위 보행보다 **덜** 한 봉우리에 몰려 있어(중앙값 ~0.65 vs 문턱 0.94) "
        "유효일이 거의 없다(KOSDAQ 지수 대용만 무작위 보행 기대치 수준인 ~5%). 지배 주기를 잴 근거가 없다는 리뷰 M4 와 같은 결론이다. "
        "적응 방식 4 는 거의 늘 기본 척도로 돌아간다"
        "(ADAPTIVE §7: 효과 없을 가능성 높음).",
        "",
        "## 주의(자료 사건·커버리지)",
        "",
        f"- 지수 대용은 거래대금 20일 창이 찬 뒤(패널 시작 + 21거래일)부터, `index_n` < {MIN_INDEX_N} 인 날은 NaN. "
        "`index_vs_ma200` 은 그 뒤 200일, 스펙트럼(256일 창)은 그 뒤 256일부터 값이 있다.",
        "- `idx_sign_ema20` = sign(`index_ret`) 의 20일 EMA — 기분 지수(B)가 넘어야 할 기준선(리뷰 M3).",
        "- 시장 구분은 현재 스냅샷(이전상장 종목은 이전 전 역사도 새 시장 — 리뷰 m2). 정지 종목·스팩이 breadth 모집단의 2~4%(m1).",
        "- 2026-09-14~ 는 분봉 정규장 집계로 보정(`krx_session.read_session_bars_agg`). 분봉이 없는 종목·날짜는 NXT 섞인 값 그대로"
        "(위 '보정' 수치 — d 20:00 에 정해지는 값이라 누출은 아니지만 정의가 섞인다, m8).",
        "- `breakout_success20` 는 시장 수준의 돌파 성공률(피벗 50일·거래량 1.5배, 5일 뒤 종가 > 피벗)이지 트레이드 결과가 아니다 — "
        "d 행은 d 까지 결과가 확정된 돌파(b ≤ d−5)만 센다.",
        "",
        "## 연도별 커버리지 (ALL 시장, 유효값 비율)",
        "",
        md_table(cov.T.rename_axis("feature")),
        "",
        "## 마지막 행 (ALL)",
        "",
        "| 피처 | 값 |", "|---|---|",
    ]
    for f in FEATURES:
        v = last[f]
        lines.append(f"| {f} | {'' if pd.isna(v) else f'{v:.4g}'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    load_env_db()
    con = connect(db_default())
    inp = load_regime_inputs(con, start=args.start, end=args.end)
    con.close()
    print(f"loaded: {inp.close.shape} days×codes, RSS {_rss_gb():.2f} GB, {time.time() - t0:.0f}s", flush=True)
    panel = build_regime_panel(inp)
    print(f"panel: {panel.shape}, RSS {_rss_gb():.2f} GB, {time.time() - t0:.0f}s", flush=True)
    sanity = index_sanity(panel)
    print(sanity.to_string(), flush=True)
    panel.to_csv(out / "regime_panel.csv", index=False, float_format="%.6g")
    save_npz(panel, out / "regime_panel.npz")
    variants = variant_table(inp, inp.official) if inp.official is not None else None
    write_summary(panel, inp.patch_info or {}, out / "REGIME_PANEL.md", elapsed=time.time() - t0, sanity=sanity,
                  variants=variants)
    print(f"wrote {out}/regime_panel.csv (+npz, REGIME_PANEL.md)")
    bad = sanity[~(sanity["ok_weight"] & sanity["ok_ret"] & sanity["ok_corr"])]
    if len(bad):
        print(f"지수 대용 위생 바 미달: {list(bad.index)} — 산출물은 썼지만 적응 입력으로 쓰지 말 것", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
