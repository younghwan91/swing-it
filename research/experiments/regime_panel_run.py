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

from swing_it.features.regime import FEATURES, MARKETS, build_regime_panel, coverage_by_year, load_regime_inputs  # noqa: E402
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
        cells = ["" if pd.isna(v) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)) for v in r]
        rows.append("| " + " | ".join([str(i)] + cells) + " |")
    return "\n".join([head, sep, *rows])


def write_summary(panel: pd.DataFrame, info: dict, path: Path, *, elapsed: float) -> None:
    cov = coverage_by_year(panel)
    last = panel[panel["market"] == "ALL"].tail(1).iloc[0]
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
        "- `flow20` 은 2024-02 이후만(그 전은 수급 수집이 연 ~650 종목). `official_vs_ma200` 은 sector_index 001/101 이 있는 "
        "2024-01-26 이후 + 200일.",
        "",
        "## 주의(자료 사건·커버리지)",
        "",
        "- **지수 대용은 2017-03-30 부터**: 주식수 기록(DART, 연 1회 12-31 기준)이 다음 해 3월 말에야 알려진다(knowledge_date). "
        "그 전엔 가중치를 가진 종목이 한두 개라 `index_n < 100` 인 날은 NaN 으로 둔다(자리 채우지 않음 — PIT). "
        "`index_vs_ma200` 은 그 뒤 200일(2018-01-24~), 스펙트럼(256일 창)은 2018-04~ 부터 값이 있다.",
        "- **주식수는 조정가 기준으로 환산**(`shares_asof(ratio=원종가/조정종가)`): 기록은 당시 분할 기준, 조정 종가는 오늘 기준이라 "
        "그대로 곱하면 2017 삼성전자 가중이 50배 작아지고 역분할 종목(900120)이 시장의 54% 가 된다 — 실측 후 고쳤다.",
        "- **주식수 > 1e10 은 수집 오류로 NaN**(214310 = 7.4e13, 130740 = 9.5e12 등). **|일수익률| > 31% 는 자료 사건으로 지수·분산에서 제외**"
        "(가격제한폭 밖 — 조정 누락·감자). 그 전엔 2024-12-12 지수 대용이 −73% 였다.",
        "- 대조(2024-01-26~): 대용 `index_vs_ma200` 과 공식 KOSPI(`official_vs_ma200`) 상관 0.9995(KOSPI 행). 레벨은 전 상장 시총가중이라 "
        "공식 지수(유동주식·대형주 비중)와 수준은 다르다 — 국면 신호로만 쓴다.",
        "- 2026-09-14~ 는 분봉 정규장 집계로 보정(`krx_session.read_session_bars_agg`, DB 집계 — 분봉 1,300만 행을 pandas 로 안 끌어온다). "
        "분봉이 없는 종목·날짜는 NXT 섞인 값 그대로(위 '보정' 수치 참고).",
        "- `breakout_success20` 는 시장 수준의 돌파 성공률(피벗 50일·거래량 1.5배, 5일 뒤 종가 > 피벗)이지 트레이드 결과가 아니다 — "
        "d 행은 d 까지 결과가 확정된 돌파(b ≤ d−5)만 센다.",
        "",
        "## 연도별 커버리지 (ALL 시장, 유효값 비율)",
        "",
        md_table(cov.T.rename_axis("feature")),
        "",
        "KOSPI 행의 `official_vs_ma200`(sector_index 001) 커버리지: "
        + ", ".join(f"{y}: {v:.2f}" for y, v in coverage_by_year(panel, ("official_vs_ma200",), market="KOSPI")
                    ["official_vs_ma200"].items()),
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
    panel.to_csv(out / "regime_panel.csv", index=False, float_format="%.6g")
    save_npz(panel, out / "regime_panel.npz")
    write_summary(panel, inp.patch_info or {}, out / "REGIME_PANEL.md", elapsed=time.time() - t0)
    print(f"wrote {out}/regime_panel.csv (+npz, REGIME_PANEL.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
