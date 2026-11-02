"""6단계 위험모델 빌드 — 원천 대조(이음매)·패널·격자별 모델과 입력 해시 캐시. core 위험모델 층은 부를 때 import 한다.

숫자를 만드는 것은 core ``build_risk_model`` 이다. 여기는 PEAD 단계 B 가 그 함수에 넘긴 입력을 만드는 경로를 옮긴 것이고, 옮기기 전 캐시 모델과
비트 동일함을 실데이터로 확인했다(``research/logs/pead_optimizer/B_modules_parity.json``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from swing_it.book.seam import seam_guard
from swing_it.core_layers import core_layer
from swing_it.storage import read_seam_sources

#: PO-3.3 — 두 번째 원천 근거의 이음매는 |수정 수익| 이 이 값을 넘을 때만 본다.
SECONDARY_MIN_ABS_RET = 0.3
MIN_EXPOSURE_NAMES = 30

_CTX: dict = {}


def load_guarded(con: Any, x0, end: str, *, secondary_min_abs_ret: float = SECONDARY_MIN_ABS_RET):
    """core ``load_risk_inputs`` 결과에 이음매 가드를 건다. ``(x, seam, info)`` — x 의 ret·close 는 가드 뒤 값이다."""
    raw_src, events = read_seam_sources(con, x0.codes, x0.dates, end)
    rf, cf, seam, info = seam_guard(x0.codes, x0.dates, x0.ret, x0.close, raw_src, events, secondary_min_abs_ret=secondary_min_abs_ret)
    return dataclasses.replace(x0, ret=rf, close=cf), seam, info


def exposures(x, raw: dict, mask: np.ndarray) -> dict:
    """스타일 원값 → 날짜별 시총 가중 표준화 노출. 유효 종목 30개 미만인 날은 NaN, 그 밖의 유효 종목 결측은 0."""
    standardize = core_layer("krx_quant_core.risk.factor_model").standardize
    out = {}
    for k, v in raw.items():
        e = np.full_like(v, np.nan)
        for t in range(v.shape[1]):
            m = mask[:, t]
            if m.sum() < MIN_EXPOSURE_NAMES:
                continue
            col = standardize(v[:, t], x.cap[:, t], m)
            col[m & ~np.isfinite(col)] = 0.0
            e[:, t] = col
        out[k] = e
    return out


def build_panels(x) -> dict:
    """:func:`build_models` 입력 묶음. 업종이 처음 나오는 날(``first``)부터 자른다. 마스크 = 업종 있음 ∧ 시총 > 0 ∧ 종가 유한."""
    style_panels = core_layer("krx_quant_core.data.risk_inputs").style_panels
    sector = x.sector
    has_sec = np.not_equal(sector, None).astype(bool)      # 객체 배열 칸별. NaN 은 None 이 아니다(옛 ``s is not None`` 과 같다)
    mask = has_sec & np.isfinite(x.cap) & (x.cap > 0) & np.isfinite(x.close)
    raw = style_panels(x)
    xall = exposures(x, raw, mask)
    first = int(np.argmax(has_sec.any(axis=0)))
    sl = slice(first, None)
    return {"R": x.ret[:, sl], "cap": x.cap[:, sl], "msk": mask[:, sl], "lab": sector[:, sl],
            "Xs": {k: v[:, sl] for k, v in xall.items()}, "dates": x.dates[sl], "first": first, "raw": raw}


def _hash_array(h, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    h.update(f"{arr.shape}".encode())
    if arr.dtype == object:
        codes, uniques = pd.factorize(arr.ravel(), use_na_sentinel=True)
        h.update(json.dumps([str(u) for u in uniques]).encode())
        arr = codes
    h.update(str(arr.dtype).encode())
    h.update(np.ascontiguousarray(arr))


def model_key(c: dict, need: Iterable[int], *, vra: "float | None" = None, shrink: float = 0.3) -> str:
    """모델을 정하는 입력 전부의 해시 — 격자·설정·수익·시총·마스크·업종·노출. 다른 항목(날짜 표시 등)은 보지 않는다."""
    h = hashlib.sha256()
    h.update(json.dumps({"need": sorted(int(t) for t in need), "vra": vra, "shrink": shrink}).encode())
    for k in ("R", "cap", "msk", "lab"):
        _hash_array(h, c[k])
    for k in sorted(c["Xs"]):
        h.update(k.encode())
        _hash_array(h, c["Xs"][k])
    return h.hexdigest()[:16]


def _build(a):
    build_risk_model = core_layer("krx_quant_core.risk.factor_model").build_risk_model
    c = _CTX
    m = build_risk_model(c["R"], c["Xs"], c["lab"], c["cap"], c["msk"], a, codes=np.arange(c["R"].shape[0]),
                         spec_shrink=c["shrink"], spec_vra_half_life=c["vra"])
    return a, (m.codes.astype(int), m.X, m.F, m.D, list(m.names))


def build_models(c: dict, need, workers: int, path: Path, log: Callable[[str], None], *, vra=None, shrink: float = 0.3) -> dict:
    """격자일마다 ``t → (행 번호, X, F, D, 이름)`` 일 단위 모델. 캐시 파일은 ``<path 이름>_in<입력 해시>`` 다.

    옛 구현(``pead_optimizer_po3.build_models``)은 경로가 있으면 설정을 안 보고 읽었다 — 다른 ``shrink`` 로 불러도 옛 모델을 돌려줬다.
    입력 해시는 core 코드를 모른다. core 판이 바뀌면 경로에 그 판을 적는다(``pead_optimizer_b.model_cache_name``).
    """
    path = Path(path)
    keyed = path.with_name(f"{path.stem}_in{model_key(c, need, vra=vra, shrink=shrink)}{path.suffix}")
    if keyed.exists():
        with keyed.open("rb") as f:
            return pickle.load(f)
    _CTX.clear()
    _CTX.update({k: c[k] for k in ("R", "Xs", "lab", "cap", "msk")})
    _CTX.update({"vra": vra, "shrink": shrink})
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            models = dict(pool.imap_unordered(_build, need, chunksize=2))
    finally:
        _CTX.clear()
    with keyed.open("wb") as f:
        pickle.dump(models, f)
    log(f"  모델 {len(models)}개 → {keyed.name}")
    return models


def as_risk_model(models: dict, t: int):
    """튜플 모델 → core ``RiskModel`` 과 행 번호. 최적화기·검증이 받는 모양이다."""
    idx, X, F, D, names = models[t]
    return core_layer("krx_quant_core.risk.factor_model").RiskModel(codes=idx, names=names, X=X, F=F, D=D), idx
