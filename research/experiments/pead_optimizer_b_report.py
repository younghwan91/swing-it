"""PEAD 최적화 단계 B — 보고만 항목과 ``B.md`` 렌더러(정정 PO-7, B 결과 전에 구현·고정).

``pead_optimizer_b.run_b`` 가 부른다. 판정 바(SIG·B-NI-1~5·VOID)는 ``pead_optimizer_b.verdict`` 이고, 여기 것은 바가 아니다 —
단 "PEAD 북 자격"(§2)은 판정문에 붙는 라벨 규칙이라 여기서 기계적으로 정한다.

- 팩터 회귀 알파(PO-7.4): 스텝 순수익(2배·1배) ~ 1 + KOSPI + KOSDAQ + UMD(12−1) + SMB + LOWVOL, Newey-West lag 2 HAC t.
- PEAD 북 자격: 두 북 모두 알파 HAC t ≤ 문턱 = max(2.0, Bonferroni 양측 α 0.05 를 원장 N 으로 나눈 z) 이면
  "구성은 정했지만 PEAD 는 팩터 노출".
- KOSPI 대비, ``factor_attribution`` 특이 몫, Deflated Sharpe·t-haircut(원장 N), PO-1.3 f_boot(전진 순차 경계 최종 확인).

숫자만 낸다(GUARDRAILS §8). 실데이터로는 ``--mode run`` 안에서만 돈다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from krx_quant_core.risk.attribution import factor_attribution
from krx_quant_core.stats.sharpe import deflated_sharpe_from_sample, t_haircut

from swing_it.diagnostics.hac import NW_LAG, nw_t_rows, ols_hac
from swing_it.pipeline.checks import check_attribution

__all__ = ["NW_LAG", "ols_hac"]  # 옮긴 구현을 이 이름으로 부르는 러너가 있다

# run_b 가 실행 전에 확인한다(PO-7 M4 — 빠진 항목이 있으면 B 를 돌리지 않는다).
IMPLEMENTED = {"factor_alpha": True, "pead_qualification": True, "vs_kospi": True, "attribution": True,
               "deflation": True, "f_boot": True, "render": True}
REQUIRED = tuple(IMPLEMENTED)

FACTOR_NAMES = ("KOSPI", "KOSDAQ", "UMD", "SMB", "LOWVOL")
ALPHA_T_FLOOR = 2.0  # GUARDRAILS 팩터 알파 t>2
META_GEN3_TRIALS = 13  # fcs META_GEN3 — 원장 파일이 이 브랜치에 없어 상수로 적는다(PREREG §4)
LEDGERS = ("pead_optimizer", "pead_concentrated")
LS_Q = 0.30  # 롱숏 팩터: 상·하위 30%
MOM_LOOKBACK, MOM_SKIP = 252, 21
OBF_SHAPE = (2.36, 1.67, 1.36)  # PO-1.3 O'Brien–Fleming 모양(K = 12·24·36)
OBF_LOOKS = (12, 24, 36)
F_FLOOR = 1.19
F_GRID = 0.005
N_FBOOT, FBOOT_BLOCK, FBOOT_ALPHA = 300_000, 3, 0.10
LABEL_FIRST = ("위험모델 v4 — D·최적화 바이어스만 신창 OB 로 검증, F·업종 미검증. "
               "F·스타일도 검정하지 않았다. 신창 보고만 값은 과소예측 쪽으로 기운다 — S:size 1.332·SM 1.274·S:momentum 1.186·P10B 1.089 "
               "(구창 v4 1.018·0.999·0.909·0.998).")
LABEL_T1 = "t+1 정지 정보 사용 — 체결일(t+1) 거래 불가 여부를 결정 문제의 no_trade 로 쓴다(t+1 종가 체결 규약의 일부, PO-7 MINOR 3)."
LABEL_MARKET = ("시장 구분(반스프레드 5·15bp)은 `stocks` 현재 스냅샷이다. 2020~23 에 KOSDAQ 이던 이전 상장 종목은 반스프레드가 "
                "작게 잡힌다(이전 상장 이력 원천 없음, PO-7 MINOR 1).")
LABEL_QUAL = "구성은 정했지만 PEAD 는 팩터 노출"
FOLD_NOTE = ("GUARDRAILS §2: 2020/21 은 TRAIN 내부 → 클린 아님. IC 는 폴드마다 누적·퍼지로 학습했지만 PEAD 신호 자체가 2016~2026 연구로 "
             "골라졌다 — 어느 폴드도 신호 선택에 대해 깨끗하지 않다(§2).")


def ledger_n(count_trials=None, extra: int = META_GEN3_TRIALS) -> dict:
    """원장 N = N_B(``pead_optimizer``) + ``pead_concentrated`` + 메타 세대 3(13, 상수)(PREREG §4)."""
    if count_trials is None:
        from swing_it.diagnostics.trials import count_trials
    parts = {lab: int(count_trials(lab)) for lab in LEDGERS}
    parts["META_GEN3(상수)"] = int(extra)
    return {"N": int(sum(parts.values())), "parts": parts}


# ML 예측 원장 상수(PREREG §9·정정 ML-4). 근거는 그 정정 문단의 커밋에 있다.
ML_OLD_ML_PEAD_AND_RL = 3  # 옛 ML-PEAD IC·매매 목표 2개 + 32세대 RL-1 LightGBM 1개
ML_UNRUN_E_SENSITIVITY = 2  # 계획했지만 E 가 진입 실패라 돌리지 않은 민감도(ML-4 ⑧ — N 을 줄이지 않는다)
ML_EVOLUTION_GENERATIONS = 20  # 진화 세대, 세대당 1로 센 하한
ML_REVIEW_IC_LOOKS = 31  # S1 리뷰어 추가 IC 관찰(ML-4 ⑥) — 넓은 N 에만


def ledger_n_ml(count_trials=None) -> dict:
    """ML 예측의 좁은·넓은 원장 N(PREREG §9). 라벨 값이고 판정에 쓰지 않는다."""
    if count_trials is None:
        from swing_it.diagnostics.trials import count_trials
    narrow = {lab: int(count_trials(lab)) for lab in ("ml_forecast", "pead_optimizer", "pead_concentrated")}
    narrow.update({"META_GEN3(상수)": META_GEN3_TRIALS, "옛 ML-PEAD·RL-1(상수)": ML_OLD_ML_PEAD_AND_RL,
                   "돌리지 않은 E 민감도(상수)": ML_UNRUN_E_SENSITIVITY})
    wide = {lab: int(count_trials(lab)) for lab in ("inst_flow_accel", "catalyst_flow_swing", "pullback_prop")}
    wide.update({"진화 세대(상수)": ML_EVOLUTION_GENERATIONS, "S1 리뷰어 관찰(상수)": ML_REVIEW_IC_LOOKS})
    n = int(sum(narrow.values()))
    return {"narrow": n, "wide": n + int(sum(wide.values())), "narrow_parts": narrow, "wide_extra_parts": wide}


def alpha_hurdle(n_trials: int) -> dict:
    """문턱 = max(2.0, Bonferroni 양측 α=0.05/N 의 z) — core ``t_haircut``."""
    th = t_haircut(n_trials)
    return {"N": int(n_trials), "bonferroni_t": th["adjusted_hurdle_t"], "hurdle": max(ALPHA_T_FLOOR, th["adjusted_hurdle_t"]),
            "rule": "max(2.0, z_{1−0.05/(2N)})"}


def pead_qualification(t_O: float, t_R: float, hurdle: float) -> dict:
    """§2 "PEAD 북 자격" — 두 북 모두 알파 HAC t 가 문턱을 못 넘으면 라벨을 붙인다(전진 팔에도)."""
    o_ok = bool(np.isfinite(t_O) and t_O > hurdle)
    r_ok = bool(np.isfinite(t_R) and t_R > hurdle)
    return {"O_over": o_ok, "R_over": r_ok, "label": None if (o_ok or r_ok) else LABEL_QUAL}


# ── 스텝 창·팩터 수익 ──────────────────────────────────────────────────────────


def compound(daily_ret: np.ndarray, f: int, f_next: int) -> tuple[float, int]:
    """(f, f_next] 일수익 복리. 결측 날은 0 으로 두고 그 수를 돌려준다."""
    r = np.asarray(daily_ret[f + 1:f_next + 1], dtype=np.float64)
    miss = int((~np.isfinite(r)).sum())
    return float(np.prod(1.0 + np.nan_to_num(r, nan=0.0)) - 1.0), miss


def hold_return(Cff: np.ndarray, C: np.ndarray, names: np.ndarray, f: int, f_next: int) -> float:
    """동일가중 매수 보유 f → f_next(벤치와 같은 규약: f 에 가격 있는 종목, 폐지는 마지막 종가)."""
    names = names[np.isfinite(C[names, f])]
    if names.size == 0:
        return float("nan")
    return float(np.mean(Cff[names, f_next] / C[names, f] - 1.0))


def long_short(score: np.ndarray, pool: np.ndarray, Cff, C, f: int, f_next: int, q: float = LS_Q) -> float:
    """``pool`` 안 ``score`` 상위 q − 하위 q, 동일가중 매수 보유. score 가 큰 쪽이 롱이다."""
    ok = pool[np.isfinite(score[pool]) & np.isfinite(C[pool, f])]
    if ok.size < 10:
        return float("nan")
    order = ok[np.argsort(-score[ok], kind="stable")]
    m = max(1, int(math.floor(q * ok.size)))
    return hold_return(Cff, C, order[:m], f, f_next) - hold_return(Cff, C, order[-m:], f, f_next)


def factor_step_returns(ctx, recs, kospi_ret: np.ndarray, kosdaq_ret: np.ndarray, trailing_vol, adv_floor: float) -> pd.DataFrame:
    """스텝마다(체결일 f → 다음 체결일) 다섯 팩터 수익(PO-7.4 정의).

    - KOSPI·KOSDAQ: `sector_index` 001·101 일수익 복리.
    - 풀: 결정일 t 의 위험모델 종목(U_t) ∧ ADV20 ≥ 200억(B 와 같은 유동성 문턱, 신호 조건 없음).
    - UMD: C_ff[t−21]/C_ff[t−252] − 1 상위 30% − 하위 30%. SMB: 시총 하위 30% − 상위 30%. LOWVOL: 63일 변동성 하위 30% − 상위 30%.
    - 모두 결정일 t 정보로 정렬, f → f_next 동일가중 매수 보유(폐지 = 마지막 종가).
    """
    rows = []
    for r in recs:
        t, f, fn = r.t, r.f, r.f_next
        pool = np.flatnonzero(ctx.in_u(t) & (ctx.adv[:, t] >= adv_floor))
        mom = (ctx.Cff[:, t - MOM_SKIP] / ctx.Cff[:, t - MOM_LOOKBACK] - 1.0) if t >= MOM_LOOKBACK else np.full(ctx.C.shape[0], np.nan)
        kp, mk = compound(kospi_ret, f, fn)
        kq, mq = compound(kosdaq_ret, f, fn)
        rows.append({"t": t, "date": r.date, "KOSPI": kp, "KOSDAQ": kq,
                     "UMD": long_short(mom, pool, ctx.Cff, ctx.C, f, fn),
                     "SMB": long_short(-ctx.cap[:, t], pool, ctx.Cff, ctx.C, f, fn),
                     "LOWVOL": long_short(-trailing_vol(ctx.ret, t), pool, ctx.Cff, ctx.C, f, fn),
                     "index_missing_days": mk + mq})
    return pd.DataFrame(rows)


def factor_alpha_report(net: pd.DataFrame, fac: pd.DataFrame) -> dict:
    """``net`` (t, net) 과 팩터 표를 t 로 붙여 회귀. 연환산 알파 = 스텝 알파 × 252/20."""
    m = net.merge(fac, on="t", how="inner")
    r = ols_hac(m["net"].to_numpy(), m[list(FACTOR_NAMES)].to_numpy())
    r["alpha_ann"] = r["alpha"] * 252.0 / 20.0
    r["factors"] = list(FACTOR_NAMES)
    r["note"] = "원 수익(현금 금리 미차감) — 절편에 rf·(1−Σβ_지수) 가 섞인다"
    return r


# ── KOSPI 대비·Deflated Sharpe ────────────────────────────────────────────────


def vs_kospi(net: pd.DataFrame, fac: pd.DataFrame) -> dict:
    m = net.merge(fac[["t", "KOSPI"]], on="t", how="inner")
    ex = m["net"] - m["KOSPI"]
    sd = float(ex.std(ddof=1)) if len(ex) > 1 else float("nan")
    by = {str(y): float(g.mean()) for y, g in ex.groupby(pd.to_datetime(m["date"]).dt.year)}
    return {"n": int(len(ex)), "excess_mean": float(ex.mean()), "excess_IR": float(ex.mean() / sd * math.sqrt(252 / 20)) if sd > 0 else float("nan"),
            "cum_book": float(np.prod(1 + m["net"]) - 1), "cum_kospi": float(np.prod(1 + m["KOSPI"]) - 1),
            "excess_mean_by_year": by}


def deflation(net, n_trials: int) -> dict:
    """절대 순수익(스텝)의 Deflated Sharpe·t-haircut(PREREG §4, N = 원장 N)."""
    return deflated_sharpe_from_sample(np.asarray(net, dtype=np.float64), n_trials)


# ── factor_attribution 특이 몫 ────────────────────────────────────────────────


def _exposure_slot(prep: dict, names: list, rows: np.ndarray, d: int) -> np.ndarray:
    """``factor_returns`` 가 날짜 d 수익에 쓴 노출(d−1 열): [market, 업종 더미, 스타일]."""
    lab = prep["lab"][rows, d - 1]
    X = np.zeros((rows.size, len(names)))
    for j, nm in enumerate(names):
        if nm == "market":
            X[:, j] = 1.0
        elif nm in prep["Xs"]:
            X[:, j] = prep["Xs"][nm][rows, d - 1]
        else:
            X[:, j] = (lab == nm).astype(float)
    return X


def step_active_weights(ctx, r, el: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """스텝 r 안의 날짜별 보유 비중(체결 뒤 비중의 매수 보유 표류). 반환 (rows, days, W[rows, days]) — 북 − 벤치."""
    held = [(i, w) for i, w, _, _ in r.contrib if w > 0]
    hi = np.array([i for i, _ in held], dtype=int)
    hw = np.array([w for _, w in held], dtype=float)
    bel = el[np.isfinite(ctx.C[el, r.f])]
    rows = np.union1d(hi, bel).astype(int)
    pos = {int(c): k for k, c in enumerate(rows)}
    days = np.arange(r.f + 1, r.f_next + 1)
    W = np.zeros((rows.size, days.size))
    cash = max(r.cash, 0.0)
    for j, d in enumerate(days):
        if hi.size:
            v = hw * ctx.Cff[hi, d - 1] / ctx.Cff[hi, r.f]
            nav = v.sum() + cash
            for c, x in zip(hi, v / nav):
                W[pos[int(c)], j] += x
        if bel.size:
            vb = ctx.Cff[bel, d - 1] / ctx.C[bel, r.f]
            vb = vb / vb.sum()
            for c, x in zip(bel, vb):
                W[pos[int(c)], j] -= x
    return rows, days, W


def attribution_report(ctx, recs, fr, prep, eligible_of, *, detail: bool = False, actual_atol: float = 0.05) -> dict:
    """스텝 능동수익(비용 전)을 core ``factor_attribution`` 으로 팩터·특이로 나눈다.

    - 날짜 d 비중 = 체결 뒤 비중의 표류(북) − 벤치 표류, 노출 = d−1 열, f·u = ``factor_returns`` (가드 뒤 수익, v4 와 같은 X).
    - 쥔 종목의 u 또는 노출이 그날 결측이면 그 종목-날은 비중 0 으로 두고 센다(정지·마스크 밖 — 그날 수익 없음).
    - 몫: 분산 몫 = var(s)/(var(φ)+var(s)) (공분산 따로), 평균 몫 = mean(s)/mean(φ+s). 묶음별 평균(시장·업종·스타일).
    - ``detail``(새 연구용, 기본 끔은 B.json 재현): 스텝마다 ``check_attribution`` 으로 귀속 합을 실제 수익과 대조(``actual_atol``)하고, 부호 있는 평균 차와
      실제 수익 기준 특이 몫을 더한다. 평균 몫의 분모는 귀속 합이라 실제 수익 기준과 다르다(B: 0.896 대 0.806).

    **백테스트 전용이다.** 전진 모니터링 경로는 아직 없다(``docs/PIPELINE.md`` 11단계).
    """
    names = list(fr.names)
    groups = {"market": [n for n in names if n == "market"], "sector": [n for n in names if n not in prep["Xs"] and n != "market"],
              "style": [n for n in names if n in prep["Xs"]]}
    phi, spec, gross_act, tot_sum, zeroed = [], [], [], [], 0
    frames = []
    gsum = {g: [] for g in groups}
    for r in recs:
        if r.f_next <= r.f:
            continue
        el = eligible_of(r.t)
        rows, days, W = step_active_weights(ctx, r, el)
        Xd = np.stack([_exposure_slot(prep, names, rows, d) for d in days])  # [T, n, K]
        U = fr.u[rows][:, days]
        bad = ~np.isfinite(U) | ~np.isfinite(Xd).all(axis=2).T
        zeroed += int((bad & (W != 0)).sum())
        Wc = np.where(bad, 0.0, W)
        Xc = np.where(np.isfinite(Xd), Xd, 0.0)
        Uc = np.where(np.isfinite(U), U, 0.0)
        F = np.nan_to_num(fr.f[days], nan=0.0)
        a = factor_attribution(Wc, Xc, F, Uc, names=names, dates=list(days))
        phi.append(float(a[names].to_numpy().sum()))
        spec.append(float(a["specific"].sum()))
        tot_sum.append(float(a["total"].sum()))
        for g, cols in groups.items():
            gsum[g].append(float(a[cols].to_numpy().sum()) if cols else 0.0)
        gross_act.append(r.gross - r.bench)
        if detail:
            frames.append({"date": r.date, **{n: float(a[n].sum()) for n in names}, "specific": spec[-1], "total": tot_sum[-1]})
    phi, spec = np.array(phi), np.array(spec)
    vs, vf = float(np.var(spec, ddof=1)), float(np.var(phi, ddof=1))
    tot = phi + spec
    extra = {}
    if detail:
        check_attribution(pd.DataFrame(frames), names, gross_act, atol=1e-9, actual_atol=actual_atol)
        act = float(np.mean(gross_act))
        extra = {"mean_actual": act, "signed_gap_mean": float(np.mean(np.array(tot_sum) - np.array(gross_act))),
                 "specific_share_of_actual": float(spec.mean() / act) if act != 0 else float("nan")}
    return {**extra, "n_steps": int(phi.size), "specific_share_var": vs / (vs + vf) if vs + vf > 0 else float("nan"),
            "cov_phi_spec": float(np.cov(phi, spec, ddof=1)[0, 1]) if phi.size > 1 else float("nan"),
            "mean_factor": float(phi.mean()), "mean_specific": float(spec.mean()),
            "specific_share_mean": float(spec.mean() / tot.mean()) if tot.mean() != 0 else float("nan"),
            "mean_by_group": {g: float(np.mean(v)) for g, v in gsum.items()},
            "zeroed_name_days": zeroed,
            "check_mean_abs_gap": float(np.mean(np.abs(np.array(tot_sum) - np.array(gross_act)))) if gross_act else float("nan"),
            "check_note": "일 귀속 합(단리) − 스텝 능동수익(복리) 의 평균 절대 차 — 복리·현금·결측 날 몫"}


# ── PO-1.3 f_boot ─────────────────────────────────────────────────────────────


def max_ratio(paths: np.ndarray, looks=OBF_LOOKS, shape=OBF_SHAPE) -> np.ndarray:
    """경로마다 max_K t_K / shape_K. 배수 f 에서 기각 ⇔ 이 값 > f."""
    out = np.full(paths.shape[0], -np.inf)
    for K, c in zip(looks, shape):
        t = nw_t_rows(paths[:, :K])
        out = np.maximum(out, np.where(np.isfinite(t), t / c, -np.inf))
    return out


def f_boot(d, n: int = N_FBOOT, block: int = FBOOT_BLOCK, seed_key: str = "fboot", chunk: int = 50_000) -> dict:
    """PO-1.3 — d_k(2배, 2020~23) 평균 뺀 계열의 원형 블록 부트스트랩(블록 3) 길이 36, n 회. α ≤ 0.10 인 최소 배수(0.005 간격).

    최종 경계 = OBF 모양 × max(1.19, f_boot). 올리기만 한다.
    """
    from pead_optimizer_c0 import keyed_rng

    x = np.asarray(d, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    T, L = x.size, OBF_LOOKS[-1]
    nb = int(math.ceil(L / block))
    rng = keyed_rng(seed_key)
    ratios = np.empty(n)
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        st = rng.integers(0, T, size=(m, nb))
        idx = ((st[:, :, None] + np.arange(block)[None, None, :]) % T).reshape(m, -1)[:, :L]
        ratios[s:s + m] = max_ratio(x[idx])
    f = math.floor(np.quantile(ratios, 1.0 - FBOOT_ALPHA) / F_GRID) * F_GRID
    while np.mean(ratios > f) > FBOOT_ALPHA:
        f = round(f + F_GRID, 10)
    while f - F_GRID > 0 and np.mean(ratios > f - F_GRID) <= FBOOT_ALPHA:
        f = round(f - F_GRID, 10)
    mult = max(F_FLOOR, f)
    return {"T_in_sample": int(T), "n": int(n), "block": block, "f_boot": float(f), "alpha_at_f_boot": float(np.mean(ratios > f)),
            "alpha_at_1.19": float(np.mean(ratios > F_FLOOR)), "multiplier": float(mult),
            "boundaries": [round(c * mult, 4) for c in OBF_SHAPE], "looks": list(OBF_LOOKS)}


# ── B.md ──────────────────────────────────────────────────────────────────────


def _f(v, n=3):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{n}f}"


def render_b(res: dict) -> str:
    """B.json → B.md. 숫자는 JSON 에서만 온다. 첫 줄 라벨은 PO-3.6·PO-5.3 그대로."""
    L = [f"# 단계 B — 현행 순위 북 R 대 최적화 북 O (2020~2023, {res.get('status_label', '검증 전')})", "",
         f"**{LABEL_FIRST}**", "", f"- {LABEL_T1}", f"- {LABEL_MARKET}", ""]
    if "STOP" in res:
        return "\n".join(L + [f"**멈춤: {res['STOP']}**"]) + "\n"
    p = res["primary"]
    v = p["verdict"]
    L += [f"사전등록 PREREG §2·PO-1~PO-7, core `{res['code']['core_ref']}`, swing-it `{res['code']['swing_it_sha']}`, "
          f"표 지문 {res['fingerprints']['daily_bars_adjusted_b_range']}(A1_repaired 와 같음 {res['fingerprints']['same_as_A1_repaired']}).", "",
          "## 판정 (§2·PO-1.6·PO-5 기계적 적용)", "", f"**{v['verdict']}**" + (f" — 실패 {', '.join(v['failed'])}" if v["failed"] else ""), "",
          "| 바 | 값 | 통과 |", "|---|---|---|",
          f"| SIG: IR(R)(2배) > R 널 95% | {_f(p['R']['IR_2x'])} vs {_f(v['null_q95_R'])} | {v['bars']['SIG']} |",
          f"| B-NI-1: ΔIR(2배) ≥ −x (x={_f(v['x'])}) | {_f(v['dIR_2x'])} | {v['bars']['B-NI-1']} |",
          f"| B-NI-2: ΔIR(1배) ≥ −x | {_f(v['dIR_1x'])} | {v['bars']['B-NI-2']} |",
          f"| B-NI-3: sd(a_O) ≤ 1.25·sd(a_R) (2배) | {_f(p['O']['sd_a_2x'], 4)} vs {_f(p['R']['sd_a_2x'], 4)} | {v['bars']['B-NI-3']} |",
          f"| B-NI-4: IR(O)(2배) > O 널 95% | {_f(p['O']['IR_2x'])} vs {_f(v['null_q95_O'])} | {v['bars']['B-NI-4']} |",
          f"| B-NI-5: EV(O)(2배) > 0 | {_f(p['O']['EV_2x'], 5)} | {v['bars']['B-NI-5']} |", "",
          f"- B-NI-1: {v['B-NI-1_note']}.", f"- B-NI-3: {v['B-NI-3_label']}.",
          f"- VOID 조건(PO-8 — λ 실패는 solver·gap·down 원인만): {v['void']}.",
          f"- λ 실패 원인 up 비율 {_f(v.get('lam_up_share'))}"
          + (f" — **{v['risk_label']}**. 실현 능동위험 비 sd(a_O)/sd(a_R)(2배) {_f(v.get('realized_active_risk_ratio_2x'))}, "
             f"사전 비 {_f(v.get('exante_risk_ratio'))}." if v.get("risk_label") else "."),
          f"- IC_k = 0 폴드: {v.get('ic_zero_label') or '없음'}.", f"- ΔIR 부트스트랩 5·50·95%: {', '.join(_f(x) for x in p['boot_dIR_q05_50_95'])}, ρ(a_O, a_R) {_f(p['rho'])}.", ""]
    rep = p["report"]
    q = rep["qualification"]
    L += ["## PEAD 북 자격 (§2, PO-7.4)", "",
          f"문턱 {_f(rep['hurdle']['hurdle'], 2)} = {rep['hurdle']['rule']}, 원장 N = {rep['ledger']['N']} {rep['ledger']['parts']}.", "",
          "| 북 | 알파/스텝(2배) | 연환산 | HAC t | β KOSPI·KOSDAQ·UMD·SMB·LOWVOL | R² | n |", "|---|---|---|---|---|---|---|"]
    for b in ("O", "R"):
        fa = rep["factor_alpha"][b]["2x"]
        L.append(f"| {b} | {_f(fa['alpha'], 5)} | {_f(fa['alpha_ann'], 4)} | {_f(fa['t_alpha'], 2)} | "
                 f"{'·'.join(_f(x, 2) for x in fa['beta'])} | {_f(fa['r2'], 2)} | {fa['n']} |")
    L += ["", f"**{q['label'] or '라벨 없음(한 북 이상이 문턱을 넘었다)'}**", ""]
    L += ["## 보고만", "", "| 항목 | O | R |", "|---|---|---|"]
    for key, nm, n in (("IR_1x", "IR(1배)", 3), ("EV_1x", "EV(1배)", 5), ("step_payoff_2x", "스텝 손익비(2배)", 2),
                       ("interval_payoff_2x", "보유 구간 손익비(2배)", 2), ("interval_EV_drop_top5_2x", "상위 5 구간 제거 EV", 5),
                       ("interval_skew_2x", "구간 왜도", 2), ("turnover_mean", "회전율(평균)", 3), ("cost1_mean", "비용 1배(평균)", 5),
                       ("names_mean", "보유 종목 수", 1), ("realized_over_exante", "실현/사전 능동위험", 3)):
        L.append(f"| {nm} | {_f(p['O'].get(key), n)} | {_f(p['R'].get(key), n)} |")
    for b in ("O", "R"):
        L.append(f"- 폴드별 IR(2배) {b}: {p[b]['folds_IR_2x']}")
    L += [f"- {FOLD_NOTE}"]
    for b in ("O", "R"):
        k, dfl, at = rep["vs_kospi"][b], rep["deflation"][b], rep["attribution"][b]
        L += [f"- {b} KOSPI 대비(2배 순수익 − KOSPI): 스텝 평균 {_f(k['excess_mean'], 5)}, IR {_f(k['excess_IR'])}, 누적 {_f(k['cum_book'])} vs {_f(k['cum_kospi'])}.",
              f"- {b} Deflated Sharpe(N={dfl['n_trials']}): 관측 {_f(dfl['observed_sharpe'])}, SR0 {_f(dfl['expected_max_sharpe_h0'])}, "
              f"PSR(>SR0) {_f(dfl['prob_deflated_sharpe'])}, t 문턱 {_f(dfl['t_haircut']['adjusted_hurdle_t'], 2)}.",
              f"- {b} factor_attribution: 특이 분산 몫 {_f(at['specific_share_var'])}, 평균 팩터·특이 {_f(at['mean_factor'], 5)}·{_f(at['mean_specific'], 5)}, "
              f"묶음 {at['mean_by_group']}, 0 처리한 종목-날 {at['zeroed_name_days']}, 점검 차 {_f(at['check_mean_abs_gap'], 5)}."]
    o = p["O"]
    broke = " — 그 날 \"F·업종은 O 에 무관\" 전제가 깨졌다" if o["factor_alarm_n"] else ""
    L += [f"- O 팩터 몫 10·50·90% {o['factor_share_q10_50_90']}, 경보(>20%) {o['factor_alarm_n']}회 {o['factor_alarm_dates'][:10]}"
          f"{broke}. 큰 노출 셋: {o.get('factor_alarm_top3', [])[:5]}. R 경보 {p['R']['factor_alarm_n']}회(보고만).",
          f"- λ 실패 원인(O): {o.get('lam_causes')}. optimal 비율 {_f(o['optimal_share'])}, λ 실패 {_f(o['lam_fail_share'])}.",
          f"- 상한가 체결 매수 수: O {o.get('n_buy_limit_up')}, R {p['R'].get('n_buy_limit_up')}. 먼지 제거 O {o.get('n_dust')}.",
          f"- 불변식 위반 0(위반이 있으면 이 파일을 쓰지 않는다). 검사 스텝 {res['invariants']['steps_checked']}.", ""]
    fb = rep.get("f_boot")
    if fb:
        L += ["## 전진 순차 경계 최종 확인 (PO-1.3)", "",
              f"d_k(2배) {fb['T_in_sample']}스텝, 원형 블록 {fb['block']}, {fb['n']}회: f_boot {fb['f_boot']:.3f}(α {fb['alpha_at_f_boot']:.4f}), "
              f"1.19 에서 α {fb['alpha_at_1.19']:.4f}. 최종 배수 {fb['multiplier']:.3f} → 경계 {fb['boundaries']} (K = {fb['looks']}).", ""]
    L += ["## 민감도 6칸 (원장 기록, 승자 선택 금지)", "", "| 칸 | IR(2배) | EV(2배) | optimal | λ 실패 | 그중 up |", "|---|---|---|---|---|---|"]
    for nm, d in res["cells"].items():
        L.append(f"| {nm} | {_f(d['O']['IR_2x'])} | {_f(d['O']['EV_2x'], 5)} | {_f(d['O']['optimal_share'])} | "
                 f"{_f(d['O']['lam_fail_share'])} | {_f(d['O'].get('lam_up_share'))} |")
    L += ["", "## 실제 규칙과의 차이 (PO-1.4, R, 누적)", "", "| 규칙 | 스텝 | 차 합 | 차 평균 | 최대 | 사건 |", "|---|---|---|---|---|---|"]
    for t in p["toggles"]:
        L.append(f"| {t['rule']} | {t['steps']} | {_f(t['diff_sum'], 5)} | {_f(t['diff_mean'], 5)} | {_f(t['diff_max_abs'], 5)} | {t['events']} |")
    L += ["", f"_{_f(res.get('elapsed_s'), 0)}초_"]
    return "\n".join(L) + "\n"
