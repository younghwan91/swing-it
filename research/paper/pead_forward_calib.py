"""전진 팔 pead_rank_F1 순차 경계 보정(사전등록 PEAD_FORWARD_PREREG.md §3) — 전진 행이 하나도 없을 때 한 번 돈다.

입력은 과거 재현(``pead_paper.py --replicate``, 2020~23 OOS 48 스텝 a_k(2배) — 단계 B 와 같은 규칙의 재현, 전진 데이터 아님)뿐이다.

1. f_boot(PO-1.3 절차 그대로): 평균 뺀 a_k 를 원형 블록(3) 부트스트랩으로 길이 36 경로 300,000 개, 경로마다 max_K t_K/모양_K.
   기각 확률 ≤ 0.10 인 0.005 격자 최소 배수. 최종 경계 = OBF 모양(2.36·1.67·1.36) × max(1.30, f_boot)(정정 F1-1 m1, 처음 1.19).
2. 그 경계의 귀무 α(한쪽): 부트스트랩·iid 정규·t₄·AR(1) φ=0.18(재현 1차 자기상관)·0.3 (각 200,000). 배수 1.19 에서의 α 도 같이 적는다.
   민감도(보고만): 2023 오른쪽 꼬리를 뺀 2020~22(36 스텝)만으로 낸 f_boot.
3. 무용 중단(비구속) 문턱에서 귀무·대립의 중단 확률과 경계 통과 누적 확률(검정력, 연 IR 0.3·0.59·1.0).

사용: ``uv run python research/paper/pead_forward_calib.py`` → ``research/paper/pead_forward_calib.json``.
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path[:0] = [".", "src"]

import research.paper.pead_paper as PP  # noqa: E402

HERE = Path(__file__).resolve().parent
REPL = HERE / "pead_forward_replication.json"
OUT = HERE / "pead_forward_calib.json"
SEED = 20261102
F_FLOOR, F_GRID, ALPHA = 1.30, 0.005, 0.10   # 정정 F1-1 m1: 1.19 → 1.30(재현 AR(1) φ≈0.18 정규에서 α 0.123 → 0.0995)
N_FBOOT, BLOCK = 300_000, 3
N_SIM = 200_000
FUTILITY = {24: 0.0}                 # 비구속 무용 중단: t_24 < 0 (2년 뒤 EV 점추정 ≤ 0 쪽)
ANN = math.sqrt(252.0 / PP.STEP)


def rng_of(key: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(zlib.crc32(key.encode()),)))


def nw_t_rows(X: np.ndarray, lag: int = PP.NW_LAG) -> np.ndarray:
    """행마다 :func:`pead_paper.nw_t`(= core ``newey_west_t``)."""
    n = X.shape[1]
    mu = X.mean(axis=1)
    d = X - mu[:, None]
    var = (d * d).sum(axis=1) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * (d[:, k:] * d[:, :-k]).sum(axis=1) / n
    se = np.sqrt(np.clip(var, 0.0, None) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, mu / se, np.nan)


def t_at_looks(paths: np.ndarray) -> np.ndarray:
    return np.stack([nw_t_rows(paths[:, :K]) for K in PP.LOOKS], axis=1)


def boot_paths(x: np.ndarray, n: int, key: str, L: int = 36) -> np.ndarray:
    T = x.size
    nb = int(math.ceil(L / BLOCK))
    st = rng_of(key).integers(0, T, size=(n, nb))
    idx = ((st[:, :, None] + np.arange(BLOCK)[None, None, :]) % T).reshape(n, -1)[:, :L]
    return x[idx]


def f_boot(x0: np.ndarray) -> dict:
    x = x0 - x0.mean()
    tl = t_at_looks(boot_paths(x, N_FBOOT, "fboot"))
    ratios = np.max(np.where(np.isfinite(tl), tl / np.array(PP.OBF_SHAPE), -np.inf), axis=1)
    f = math.floor(np.quantile(ratios, 1.0 - ALPHA) / F_GRID) * F_GRID
    while np.mean(ratios > f) > ALPHA:
        f = round(f + F_GRID, 10)
    while f - F_GRID > 0 and np.mean(ratios > f - F_GRID) <= ALPHA:
        f = round(f - F_GRID, 10)
    mult = max(F_FLOOR, f)
    return {"T_in_sample": int(x.size), "n": N_FBOOT, "block": BLOCK, "f_boot": float(f), "alpha_at_f_boot": float(np.mean(ratios > f)),
            "alpha_at_1.19": float(np.mean(ratios > F_FLOOR)), "multiplier": float(mult),
            "boundaries": [round(c * mult, 3) for c in PP.OBF_SHAPE]}


def sequential(tl: np.ndarray, b: np.ndarray, futility: dict | None) -> dict:
    """경로별 순차 판정 → 시점별 누적 기각·무용 중단 확률. 무용 중단은 그 시점 기각이 아닐 때만(경계 먼저 본다)."""
    n = tl.shape[0]
    alive = np.ones(n, dtype=bool)
    rej_cum, fut_cum = [], []
    rej = np.zeros(n, dtype=bool)
    fut = np.zeros(n, dtype=bool)
    for j, K in enumerate(PP.LOOKS):
        r = alive & (tl[:, j] >= b[j])
        rej |= r
        alive &= ~r
        if futility and K in futility:
            s = alive & (tl[:, j] < futility[K])
            fut |= s
            alive &= ~s
        rej_cum.append(float(rej.mean()))
        fut_cum.append(float(fut.mean()))
    return {"reject_cum": rej_cum, "futility_cum": fut_cum}


def main() -> int:
    rep = json.loads(REPL.read_text(encoding="utf-8"))
    a = np.asarray(rep["a_2x"], dtype=np.float64)
    fb = f_boot(a)
    b = np.array(fb["boundaries"])
    sd = float(a.std(ddof=1))
    x = a - a.mean()
    res = {"input": {"file": REPL.name, "n": int(a.size), "EV_2x": float(a.mean()), "sd_2x": sd, "IR_2x": float(a.mean() / sd * ANN),
                     "nw_t_2x": PP.nw_t(a), "ac1": float(np.corrcoef(a[1:], a[:-1])[0, 1])},
           "obf_shape": list(PP.OBF_SHAPE), "looks": list(PP.LOOKS), "f_boot": fb, "boundaries": fb["boundaries"],
           "futility": {str(k): v for k, v in FUTILITY.items()}, "null": {}, "power": {}}
    rng = rng_of("null")
    dgps = {"boot": boot_paths(x, N_SIM, "null-boot"), "iid_normal": rng.normal(size=(N_SIM, 36)),
            "t4": rng.standard_t(4, size=(N_SIM, 36))}
    for phi in (0.18, 0.3):
        e = rng_of(f"ar1-{phi}").normal(size=(N_SIM, 36))
        ar = np.empty_like(e)
        ar[:, 0] = e[:, 0] / math.sqrt(1 - phi * phi)
        for i in range(1, 36):
            ar[:, i] = phi * ar[:, i - 1] + e[:, i]
        dgps[f"ar1_{phi}"] = ar
    b119 = np.array([round(c * max(1.19, fb["f_boot"]), 3) for c in PP.OBF_SHAPE])
    for name, P in dgps.items():
        tl = t_at_looks(P)
        res["null"][name] = {"no_futility": sequential(tl, b, None), "with_futility": sequential(tl, b, FUTILITY),
                             "at_multiplier_1.19_no_futility": sequential(tl, b119, None)}
    res["f_boot_2020_2022_only"] = f_boot(a[: 36])                   # 보고만 — 경계는 위 값
    # 대립: 평균 뺀 부트스트랩 경로에 δ = 연 IR / √(252/20) × sd 를 더한다(iid 정규도 같이)
    for ir in (0.3, 0.59, 1.0):
        delta = ir / ANN * sd
        Pb = boot_paths(x, N_SIM, f"alt-boot-{ir}") + delta
        Pn = rng_of(f"alt-n-{ir}").normal(size=(N_SIM, 36)) * sd + delta
        res["power"][f"IR_{ir}"] = {"boot": sequential(t_at_looks(Pb), b, FUTILITY), "iid_normal": sequential(t_at_looks(Pn), b, FUTILITY)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
