"""메타 세대 1 가속 커널 — 절대수익(복리) 포트폴리오 · 시가평가 절단(매도분 여러 개) · 월 모멘텀(P2) 경로.

설계 정본: ``research/logs/sepa_explore/META_GEN1.md`` (결과 보기 전 고정). 이 파일은 **계산 기계**만 둔다
(어느 절차가 좋은지는 러너 ``research/experiments/meta_gen1_run.py`` 가 리포트한다).

:mod:`research.signals.ga_fast` 의 처리량 엔진(``_portfolio_py`` — 슬롯 10 × 자산 10% 고정, 정합 초과)과
:mod:`research.signals.exits_v2_fast` 의 v2 포트폴리오(매도분 둘, 절단 없음)를 합친 판이다. 세 가지만 더했다.

1. **절대수익 시계열(복리):** 진입 비중 = 신호일 종가 기준 **그때 자산의** ``slot_w`` (원장 원금이 아니라 현재 자산 — 복리). 날별 손익 =
   Σ 명목 × (남은 비율 × Δ종가/진입가, 매도일은 체결가), 비용은 마지막 청산일에 명목 × 비용률(GA2 와 같은 날·같은 크기). 날별 수익률 =
   손익 / 전일 자산. 현금 제약은 두지 않는다 — 손실 종목을 새 종목으로 바꾸면 총 노출이 자산을 조금 넘을 수 있어 **최대 총 노출**을
   같이 돌려준다(리포트에 적는다).
2. **정합 초과 시계열(GA2 와 같은 식):** 진입 비중 10% 고정(원금 1 대비), 대조 평균 수익을 보유일로 일할해 매일 빼고, 비용은 청산일에,
   짝 없는 포지션은 0. 슬롯 상한·조기 해제가 꺼져 있으면 ``ga_fast.portfolio`` 와 같은 숫자(테스트).
3. **날별 슬롯 상한 ``cap[t]``** (노출 조절 — 신규 진입만 막고 보유는 팔지 않는다)과 **조기 해제 표시 ``early``** (월 리밸런스: 신호일
   종가에 이미 정해진 "다음날 시가 매도" 포지션은 그날 주문의 빈 슬롯으로 센다 — P2 전용, P1 계열은 GA2 그대로 끔).

**시가평가 절단(:func:`_mark_ev_py`):** 학습창 끝·폴드 경계 ``cut`` 행 직전 거래일 종가에서 남은 수량을 닫는다(``ga_fast._mark_py`` 의
매도분 여러 개 판 — 절단 전 매도분은 그대로, 절단 뒤 매도분은 없앰). 대조도 같은 날 끊는다.

**P2 경로(:func:`_p2_trades_py`):** 리밸런스 신호일 ``rb_d[k]`` (매월 마지막 거래일 종가) 상위 ``N[k]`` 종목 → 다음날 시가 진입,
개별 손절 = 진입가 × (1 − stop) (장중 저가·시가 ≤ 손절가면 ``min(시가, 손절가)``), 이후 리밸런스에서 상위 ``N`` 밖이면 그 다음날(첫 거래
가능일) 시가에 전량, 안이면 계속 보유(거래 없음). 시세가 끊기면 마지막 거래 가능일 종가. 대조(:func:`_p2_ctl_py`)는 같은 날 같은 층 쌍을
**같은 진입일 시가 → 트레이드와 같은 청산일**(트레이드가 시가 청산이면 시가, 아니면 종가)로 들고, 자기 −15% 손절도 건다.

numba 는 선택(``uv run --with numba``). 없으면 같은 소스가 순수 파이썬으로 돈다. ``이름_py`` = 순수 파이썬, ``이름`` = JIT 판.
"""

from __future__ import annotations

import numpy as np

from research.signals import ga_fast as gf

HAVE_NUMBA = gf.HAVE_NUMBA
prange = gf.prange

#: P2 청산 사유(ga_fast 사유 코드 뒤에 붙인다).
R_REBAL, R_MARK = 10, 11
#: 사유 이름(리포트용) — ga_fast.REASON_NAMES 확장.
REASON_EXTRA = {R_REBAL: "rebalance", R_MARK: "mark"}


def _jit(fn, parallel: bool = False):
    return gf._jit(fn, parallel=parallel)


# ---------------------------------------------------------------------------
# 1. 시가평가 절단 — 매도분 둘 + 최종
# ---------------------------------------------------------------------------

def _mark_ev_py(i, cut, pc, pd_, x_abs, ev_abs, ev_frac, ev_ret, fin_frac, fin_ret, Op, C):
    """쌍 ``i`` 를 ``cut`` 행 직전 거래일 종가에서 끊는다 → ``(x, a0, f0, r0, a1, f1, r1, ff, fr)``.

    청산 행 < cut 이면 그대로. 아니면 청산 = cut−1 행, 절단 전 매도분은 남기고 절단 이후 매도분은 없앤다(남은 비율에 합친다),
    남은 비율의 가격 = cut−1 행 이하 마지막 유한 종가(진입일까지 거슬러, 없으면 진입가). ``ga_fast._mark_py`` 와 같은 규칙.
    """
    x = int(x_abs[i])
    a0 = int(ev_abs[i, 0])
    a1 = int(ev_abs[i, 1])
    f0 = ev_frac[i, 0]
    f1 = ev_frac[i, 1]
    r0 = ev_ret[i, 0]
    r1 = ev_ret[i, 1]
    if x < cut:
        return x, a0, f0, r0, a1, f1, r1, fin_frac[i], fin_ret[i]
    c = pc[i]
    e = pd_[i] + 1
    t_last = int(cut) - 1
    if t_last < e:
        return x, a0, f0, r0, a1, f1, r1, fin_frac[i], fin_ret[i]
    ent = Op[c, e]
    m = 0.0
    for t in range(t_last, e - 1, -1):
        cl = C[c, t]
        if np.isfinite(cl):
            m = cl / ent - 1.0
            break
    rem = 1.0
    if a0 >= 0 and a0 < cut:
        rem -= f0
    else:
        a0, f0, r0 = -1, 0.0, np.nan
    if a1 >= 0 and a1 < cut:
        rem -= f1
    else:
        a1, f1, r1 = -1, 0.0, np.nan
    return t_last, a0, f0, r0, a1, f1, r1, rem, m


def _total_py(a0, f0, r0, a1, f1, r1, ff, fr):
    """매도분 합 = Σ 비율 × 수익(시간 순)."""
    acc = 0.0
    if a0 >= 0:
        acc += f0 * r0
    if a1 >= 0:
        acc += f1 * r1
    return acc + ff * fr


def _cands_py(idx, cutv, pc, pd_, x_abs, ev_abs, ev_frac, ev_ret, fin_frac, fin_ret, Op, C, cidx, cost_tab, market,
              cost_mult):
    """후보 쌍 ``idx`` (신호일 → 종목 순) 각각을 자기 절단 ``cutv[k]`` 로 끊은 매도분·대조 평균·비용 → 압축 배열.

    대조 = ``cidx[i]`` 의 쌍들(같은 청산 배열, 같은 절단) 평균 총수익, 짝 없음이면 NaN. 비용 = ``cost_tab[청산 행, 시장] × cost_mult``.
    체결 안 된 쌍(청산 행 < 0)은 x = −1.
    """
    n = idx.shape[0]
    x = np.full(n, -1, np.int64)
    ea = np.full((n, 2), -1, np.int64)
    ef = np.zeros((n, 2))
    er = np.full((n, 2), np.nan)
    ff = np.zeros(n)
    fr = np.full(n, np.nan)
    tot = np.full(n, np.nan)
    ctl = np.full(n, np.nan)
    cost = np.zeros(n)
    n_per = cidx.shape[1]
    for k in prange(n):
        i = idx[k]
        if x_abs[i] < 0:
            continue
        cut = cutv[k]
        xe, a0, f0, r0, a1, f1, r1, f_, r_ = _mark_ev(i, cut, pc, pd_, x_abs, ev_abs, ev_frac, ev_ret, fin_frac, fin_ret,
                                                     Op, C)
        x[k] = xe
        ea[k, 0] = a0
        ea[k, 1] = a1
        ef[k, 0] = f0
        ef[k, 1] = f1
        er[k, 0] = r0
        er[k, 1] = r1
        ff[k] = f_
        fr[k] = r_
        tot[k] = _total(a0, f0, r0, a1, f1, r1, f_, r_)
        mk = market[i] if market[i] >= 0 else 0
        cost[k] = cost_tab[xe, mk] * cost_mult
        if cidx[i, 0] < 0:
            continue
        acc = 0.0
        for j in range(n_per):
            q = cidx[i, j]
            ya, b0, g0, s0, b1, g1, s1, gf_, sf_ = _mark_ev(q, cut, pc, pd_, x_abs, ev_abs, ev_frac, ev_ret, fin_frac,
                                                           fin_ret, Op, C)
            acc += _total(b0, g0, s0, b1, g1, s1, gf_, sf_)
        ctl[k] = acc / n_per
    return x, ea, ef, er, ff, fr, tot, ctl, cost


# ---------------------------------------------------------------------------
# 2. 포트폴리오 — 복리 절대수익 + GA2 정합 초과, 날별 슬롯 상한, 조기 해제
# ---------------------------------------------------------------------------

def _portfolio_meta_py(cd_pd, cd_pc, cd_fill, cd_p1, cd_p2, cd_x, cd_ea, cd_ef, cd_er, cd_ff, cd_fr, cd_ctl, cd_cost,
                       cd_early, day_lo, day_hi, cap, Op, C, t0, t1, slot_w, n_arr):
    """후보(신호 ∧ 주문 가능, 신호일 → 종목 순 — ``day_lo[t]..day_hi[t]`` 가 신호일 ``t`` 후보) → 날별 (절대수익, 정합 초과, 총 노출).

    - 주문(``ga_fast._portfolio_py`` 와 같다): 신호일 ``t`` 에 빈 슬롯 수만큼 ``cd_p1`` 오름차순(NaN 뒤) → ``cd_p2`` 내림차순(NaN 뒤) →
      종목 순. 보유 중 종목 제외. 체결이 아니면 그 슬롯은 그날 빈다(주문 수에는 센다).
    - 빈 슬롯 = ``cap[t]`` − (보유 중 − ``cd_early`` 이고 ``t+1`` 에 청산하는 것). ``n_arr`` = 슬롯 배열 길이(상한의 2배 이상).
    - 절대수익: 명목 = ``slot_w`` × 신호일 종가 자산. 날별 수익률 = 그날 손익 / 전일 자산(비용은 청산일 명목 × 비용률).
    - 정합 초과: 진입 비중 ``slot_w`` 고정, 대조 일할 차감, 짝 없음 0 — GA2 처리량 적합도와 같은 시계열.
    반환: ``(raw[t0:t1], excess[t0:t1], gross[t0:t1], opened 후보 번호, n_orders, n_unmatched)``.
    """
    T = t1 - t0
    raw = np.zeros(T)
    exc = np.zeros(T)
    gross = np.zeros(T)
    s_i = np.full(n_arr, -1, np.int64)
    s_lm = np.zeros(n_arr)
    s_fr = np.zeros(n_arr)
    s_cd = np.zeros(n_arr)
    s_not = np.zeros(n_arr)
    s_m = np.zeros(n_arr, np.bool_)
    opened = np.empty(cd_pd.shape[0] + 1, np.int64)
    n_open = 0
    n_ord = 0
    n_unm = 0
    eq = 1.0
    cand = np.empty(4096, np.int64)
    for t in range(t0, t1):
        pnl = 0.0
        acc = 0.0
        val = 0.0
        for s in range(n_arr):
            i = s_i[s]
            if i < 0:
                continue
            c = cd_pc[i]
            e = cd_pd[i] + 1
            if t < e:
                continue
            ent = Op[c, e]
            contrib = 0.0
            for k in range(2):
                if cd_ea[i, k] == t:
                    contrib += cd_ef[i, k] * (cd_er[i, k] - s_lm[s])
                    s_fr[s] -= cd_ef[i, k]
            xe = cd_x[i]
            if xe == t:
                contrib += cd_ff[i] * (cd_fr[i] - s_lm[s])
            else:
                cl = C[c, t]
                if np.isfinite(cl):
                    m = cl / ent - 1.0
                    contrib += s_fr[s] * (m - s_lm[s])
                    s_lm[s] = m
            pnl += s_not[s] * contrib
            if s_m[s]:
                acc += slot_w * (contrib - s_cd[s])
            if xe == t:
                pnl -= s_not[s] * cd_cost[i]
                if s_m[s]:
                    acc -= slot_w * cd_cost[i]
                s_i[s] = -1
            else:
                val += s_not[s] * s_fr[s] * (1.0 + s_lm[s])
        raw[t - t0] = pnl / eq
        eq += pnl
        exc[t - t0] = acc
        gross[t - t0] = val / eq if eq > 0 else np.nan
        a = day_lo[t]
        b = day_hi[t]
        if b <= a:
            continue
        n_act = 0
        for s in range(n_arr):
            i = s_i[s]
            if i >= 0 and not (cd_early[i] and cd_x[i] == t + 1):
                n_act += 1
        free = cap[t] - n_act
        if free <= 0:
            continue
        nc = 0
        for i in range(a, b):
            held = False
            for s in range(n_arr):
                if s_i[s] >= 0 and cd_pc[s_i[s]] == cd_pc[i]:
                    held = True
                    break
            if held:
                continue
            if nc < cand.shape[0]:
                cand[nc] = i
                nc += 1
        k_take = min(free, nc)
        for r in range(k_take):
            best = -1
            for q in range(nc):
                i = cand[q]
                if i < 0:
                    continue
                if best < 0:
                    best = q
                    continue
                j = cand[best]
                ri = cd_p1[i] if np.isfinite(cd_p1[i]) else np.inf
                rj = cd_p1[j] if np.isfinite(cd_p1[j]) else np.inf
                if ri < rj:
                    best = q
                elif ri == rj:
                    si = cd_p2[i] if np.isfinite(cd_p2[i]) else -np.inf
                    sj = cd_p2[j] if np.isfinite(cd_p2[j]) else -np.inf
                    if si > sj or (si == sj and cd_pc[i] < cd_pc[j]):
                        best = q
            i = cand[best]
            cand[best] = -1
            n_ord += 1
            if cd_fill[i] and cd_x[i] >= 0:
                for s in range(n_arr):
                    if s_i[s] < 0:
                        s_i[s] = i
                        s_lm[s] = 0.0
                        s_fr[s] = 1.0
                        s_not[s] = slot_w * eq
                        s_m[s] = np.isfinite(cd_ctl[i])
                        if s_m[s]:
                            s_cd[s] = cd_ctl[i] / (cd_x[i] - cd_pd[i])
                        else:
                            s_cd[s] = 0.0
                            n_unm += 1
                        break
                opened[n_open] = i
                n_open += 1
    return raw, exc, gross, opened[:n_open], n_ord, n_unm


# ---------------------------------------------------------------------------
# 3. P2 — 월 리밸런스 모멘텀 경로와 대조
# ---------------------------------------------------------------------------

def _p2_trades_py(sel_pair, sel_k, pc, pd_, rb_d, N, rank_kc, Op, H, L, C, TR, LD, last_tr, last_present, fill, cutv, stop):
    """리밸런스 ``sel_k[q]`` 에서 고른 쌍 ``sel_pair[q]`` 의 경로 → ``(x, 수익, 사유, early, 시가 청산)``.

    계획 청산 = ``k`` 뒤 첫 리밸런스 ``kk`` 중 ``rank_kc[kk, 종목] ≥ N[kk]`` (상위 밖 — 국면 게이트면 ``N = 0`` 이라 전부)인 것의
    신호일 다음 첫 거래 가능일 시가. 그 전에 장중 손절(저가·시가 ≤ 진입가 × (1 − stop) → ``min(시가, 손절가)``). 절단 ``cutv[q]`` 직전
    거래일까지 안 나가면 그 종가로 시가평가(사유 mark). 시세가 끊기면(마지막 거래 가능일 < 지평) 그날 종가(delisted / data_end).
    ``early`` = 계획 청산일이 정확히 리밸런스 신호일 + 1(그 신호일 주문에서 슬롯이 빈다). 체결 안 됨이면 x = −1.
    **하한가로 잠긴 날(``LD``)은 팔 수 없다** — 손절·리밸런스 매도 모두 다음 잠기지 않은 거래 가능일 시가로 밀린다(설계 정정 M4).
    """
    n = sel_pair.shape[0]
    n_days = Op.shape[1]
    K = rb_d.shape[0]
    x = np.full(n, -1, np.int64)
    ret = np.full(n, np.nan)
    reason = np.full(n, -1, np.int8)
    early = np.zeros(n, np.bool_)
    xopen = np.zeros(n, np.bool_)
    for q in prange(n):
        i = sel_pair[q]
        if not fill[i]:
            continue
        k = sel_k[q]
        c = pc[i]
        e = pd_[i] + 1
        ent = Op[c, e]
        stp = ent * (1.0 - stop)
        plan = -1
        plan_d = -1
        for kk in range(k + 1, K):
            if rank_kc[kk, c] >= N[kk]:
                plan_d = rb_d[kk] + 1
                for t in range(plan_d, n_days):
                    if TR[c, t] and not LD[c, t]:
                        plan = t
                        break
                break
        hi = min(int(cutv[q]) - 1, n_days - 1)
        lt = last_tr[c]
        done = False
        for t in range(e, hi + 1):
            if t > lt:
                break
            if t == plan:
                x[q] = t
                ret[q] = Op[c, t] / ent - 1.0
                reason[q] = R_REBAL
                early[q] = t == plan_d
                xopen[q] = True
                done = True
                break
            if TR[c, t] and not LD[c, t]:
                op = Op[c, t]
                if L[c, t] <= stp or op <= stp:
                    x[q] = t
                    ret[q] = min(op, stp) / ent - 1.0
                    reason[q] = gf.R_STOP
                    done = True
                    break
        if done:
            continue
        if lt < hi:
            x[q] = lt
            ret[q] = C[c, lt] / ent - 1.0
            reason[q] = gf.R_DATA_END if last_present[c] else gf.R_DELISTED
            continue
        m = 0.0
        for t in range(hi, e - 1, -1):
            cl = C[c, t]
            if np.isfinite(cl):
                m = cl / ent - 1.0
                break
        x[q] = hi
        ret[q] = m
        reason[q] = R_MARK
    return x, ret, reason, early, xopen


def _p2_ctl_py(sel_pair, x, xopen, cidx, pc, pd_, Op, L, C, TR, LD, stop):
    """P2 트레이드별 대조 평균 수익 — 대조 쌍(같은 날·같은 층, 자기 제외)을 같은 진입일 시가에 사서 트레이드 청산일 ``x[q]``
    (트레이드가 시가 청산이면 그날 시가, 아니면 종가 — 거래 불가면 마지막 유한 종가)까지 들고, 자기 −stop 손절도 건다(하한가 잠긴 날은
    못 팔아 다음 거래일로 밀린다 — 트레이드 청산일을 넘으면 그날 종가 평가). 짝 없음 NaN."""
    n = sel_pair.shape[0]
    out = np.full(n, np.nan)
    n_per = cidx.shape[1]
    for q in prange(n):
        i = sel_pair[q]
        xe = x[q]
        if xe < 0 or cidx[i, 0] < 0:
            continue
        acc = 0.0
        for j in range(n_per):
            qi = cidx[i, j]
            c2 = pc[qi]
            e = pd_[qi] + 1
            ent = Op[c2, e]
            stp = ent * (1.0 - stop)
            px = np.nan
            last = ent
            for t in range(e, xe + 1):
                if TR[c2, t] and not LD[c2, t]:
                    op = Op[c2, t]
                    if t == xe and xopen[q]:
                        px = op
                        break
                    if L[c2, t] <= stp or op <= stp:
                        px = min(op, stp)
                        break
                cl = C[c2, t]
                if np.isfinite(cl):
                    last = cl
            if np.isnan(px):
                px = last
            acc += px / ent - 1.0
        out[q] = acc / n_per
    return out


# --- 내보내기: 이름_py = 순수 파이썬, 이름 = numba(있으면) ---
_mark_ev = _jit(_mark_ev_py)
_total = _jit(_total_py)
cands = _jit(_cands_py, parallel=True)
portfolio_meta = _jit(_portfolio_meta_py)
p2_trades = _jit(_p2_trades_py, parallel=True)
p2_ctl = _jit(_p2_ctl_py, parallel=True)

#: 순수 파이썬 경로 테스트가 JIT 판을 되감을 보조 함수 이름(``_py`` 커널이 부르는 전역 이름).
HELPERS = ("_mark_ev", "_total")


# ---------------------------------------------------------------------------
# 4. 파이썬 쪽 도우미 — 성적·월 블록 셔플·등가중 보유
# ---------------------------------------------------------------------------

def day_bounds(cd_pd: np.ndarray, n_days: int) -> "tuple[np.ndarray, np.ndarray]":
    """후보 신호일 배열(오름차순) → 날별 후보 범위 ``(day_lo, day_hi)``."""
    d = np.arange(n_days)
    return (np.searchsorted(cd_pd, d, "left").astype(np.int64), np.searchsorted(cd_pd, d, "right").astype(np.int64))


def month_block_perm(month_of_day: np.ndarray, lo: int, hi: int, rng: np.random.Generator) -> np.ndarray:
    """날 행 [lo, hi) 를 달 블록 단위로 섞은 행 순서(길이 hi − lo). 달 안의 날 순서는 그대로, 달 순서만 무작위.

    가짜 국면 신호 = ``series[lo:hi][perm − lo]`` 를 원래 날 자리에 다시 놓는다(달 길이가 달라 경계는 밀린다 — 블록 셔플의 보통 모양).
    """
    m = month_of_day[lo:hi]
    starts = np.r_[0, np.flatnonzero(np.diff(m)) + 1]
    ends = np.r_[starts[1:], len(m)]
    order = rng.permutation(len(starts))
    return np.concatenate([np.arange(starts[j], ends[j]) for j in order]) + lo


def equity_stats(r: np.ndarray, ppy: int = 252) -> dict:
    """날별 수익률 → CAGR(기간 수 연환산 — ``krx_quant_core`` ``cagr``)·MDD(초기 자산 1 을 첫 고점으로)·Calmar(= CAGR / |MDD|)·누적."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"cagr": np.nan, "mdd": np.nan, "calmar": np.nan, "total": np.nan, "n": 0}
    eq = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    cagr = float(eq[-1] ** (ppy / r.size) - 1.0) if eq[-1] > 0 else -1.0
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {"cagr": cagr, "mdd": mdd, "calmar": calmar, "total": float(eq[-1] - 1.0), "n": int(r.size)}


def ew_daily(C: np.ndarray, member_days: np.ndarray, member_codes: np.ndarray, t_from: int, t_to: int,
             limit: float = 0.35) -> np.ndarray:
    """유니버스 등가중 **매일 재조정** 보유 → 날별 수익률 ``[t_from, t_to)``.

    날 ``t`` 수익 = 전일(``t−1``) 유니버스 종목들의 종가 수익 평균. 둘 중 하나라도 NaN 이거나 |수익| > ``limit`` (가격제한 ±30% 밖 =
    조정 누락 같은 자료 사건)인 종목은 그날 평균에서 뺀다. ``member_days``·``member_codes`` = 유니버스 (신호일, 종목) 쌍(신호일 오름차순).
    """
    out = np.full(t_to - t_from, np.nan)
    b = np.searchsorted(member_days, np.arange(C.shape[1] + 1))
    for t in range(t_from, t_to):
        cs = member_codes[b[t - 1]:b[t]]
        if len(cs) == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r = C[cs, t] / C[cs, t - 1] - 1.0
        r = r[np.isfinite(r) & (np.abs(r) <= limit)]
        out[t - t_from] = r.mean() if len(r) else 0.0
    return out


def ew_monthly(C: np.ndarray, member_days: np.ndarray, member_codes: np.ndarray, rb_days: np.ndarray, t_from: int,
               t_to: int, limit: float = 0.35) -> np.ndarray:
    """유니버스 등가중 **월 재조정** 보유 → 날별 수익률 ``[t_from, t_to)``.

    리밸런스 신호일 ``rb_days`` (매월 마지막 거래일) 종가에 그날 유니버스 종목을 같은 금액으로 사서 다음 리밸런스까지 둔다(비중 표류).
    종가 NaN(정지·폐지)은 직전 가격 유지(수익 0), |날 수익| > ``limit`` 는 0 으로(자료 사건). 첫 리밸런스 전 날은 NaN.
    """
    out = np.full(t_to - t_from, np.nan)
    b = np.searchsorted(member_days, np.arange(C.shape[1] + 1))
    rbs = [int(d) for d in rb_days if d < t_to]
    for j, d in enumerate(rbs):
        nxt = rbs[j + 1] if j + 1 < len(rbs) else t_to - 1
        if nxt < t_from:
            continue
        cs = member_codes[b[d]:b[d + 1]]
        if len(cs) == 0:
            continue
        # NaN 이 끼면 직전 유한 종가 대비로 — 정지일 수익 0, 재개일에 몰아서
        px = C[cs, d:nxt + 1].copy()
        for k in range(1, px.shape[1]):
            bad = ~np.isfinite(px[:, k])
            px[bad, k] = px[bad, k - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = px[:, 1:] / px[:, :-1] - 1.0
        r = np.where(np.isfinite(r) & (np.abs(r) <= limit), r, 0.0)
        v = np.cumprod(1.0 + r, axis=1)
        v_prev = np.concatenate([np.ones((len(cs), 1)), v[:, :-1]], axis=1)
        port = (v_prev * r).sum(axis=0) / v_prev.sum(axis=0)
        for k, t in enumerate(range(d + 1, nxt + 1)):
            if t_from <= t < t_to:
                out[t - t_from] = port[k]
    return out


# ---------------------------------------------------------------------------
# 5. 설계 정정(REVIEW_META_GEN1 반영) — 잠긴 하한가·깨진 조정가·Calmar 부트스트랩
# ---------------------------------------------------------------------------

#: 가격제한폭 ±30% 의 판정 허용폭(``korea.LIMIT_UP_REL_TOL`` 과 같은 0.7%p).
LIMIT_TOL = 0.007


def locked_down(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, TR: np.ndarray,  # noqa: E741
                tol: float = LIMIT_TOL) -> np.ndarray:
    """하한가로 **잠긴** 날(종목 × 날): 거래 가능 ∧ 고가 = 저가(한 가격) ∧ 종가 ≤ 직전 유한 종가 × (0.70 + tol). 그날은 팔 수 없다."""
    prev = np.full_like(C, np.nan)
    prev[:, 1:] = C[:, :-1]
    prev = pd_ffill_rows(prev)
    with np.errstate(invalid="ignore"):
        return np.asarray(TR, bool) & (H <= L * (1.0 + 1e-6)) & (C <= prev * (0.70 + tol))


def pd_ffill_rows(a: np.ndarray) -> np.ndarray:
    """행(종목)마다 NaN 을 직전 유한값으로(앞쪽 NaN 은 그대로)."""
    n, m = a.shape
    idx = np.where(np.isfinite(a), np.arange(m)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(n)[:, None], idx]


def clean_jumps(O: np.ndarray, H: np.ndarray, L: np.ndarray, C: np.ndarray, limit: float = 0.31):  # noqa: E741
    """조정 일봉의 |종가 수익| > ``limit`` (가격제한 ±30% 밖 = 조정 누락 같은 자료 사건)를 **그날 수익 0** 으로 지운 가격.

    직전 유한 종가 대비 비율 f 가 범위 밖이면 그날부터 뒤 가격(시·고·저·종) 전부를 f 로 나눈다 — 그 종목 그날 수익만 빠지고 앞뒤 수익은
    그대로다(설계 정정 M4 "그 종목 그날 제외"). 반환: (O, H, L, C, 지운 날 bool 종목 × 날).
    """
    prev = np.full_like(C, np.nan)
    prev[:, 1:] = C[:, :-1]
    prev = pd_ffill_rows(prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = C / prev
        bad = np.isfinite(f) & (np.abs(f - 1.0) > limit)
    adj = np.where(bad, f, 1.0)
    cum = np.cumprod(adj, axis=1)
    return O / cum, H / cum, L / cum, C / cum, bad


def calmar_paths(paths: np.ndarray, ppy: int, mdd_floor: float = 0.01) -> np.ndarray:
    """수익 경로들(행 = 표본) → Calmar = CAGR / max(|MDD|, mdd_floor) (초기 자산 1 을 첫 고점으로)."""
    eq = np.cumprod(1.0 + paths, axis=1)
    eq = np.concatenate([np.ones((eq.shape[0], 1)), eq], axis=1)
    mdd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    with np.errstate(invalid="ignore"):
        cagr = np.where(eq[:, -1] > 0, eq[:, -1] ** (ppy / paths.shape[1]) - 1.0, -1.0)
    return cagr / np.maximum(np.abs(mdd), mdd_floor)


def calmar_boot_lower(monthly: np.ndarray, *, block: int = 3, n_boot: int = 500, seed: int = 0, q: float = 2.5) -> float:
    """학습창 절대 포트폴리오 월 수익의 이동 블록 부트스트랩(블록 3개월, 500회, 비원형 — ``ga_fast.block_boot`` 와 같은 모양) Calmar 의
    ``q`` 백분위(설계 정정 C1 — 국면 게이트·노출 (a, b) 선택 적합도). MDD 하한 1% (낙폭이 없으면 분모가 0). 월 < 4 면 −∞."""
    m = np.asarray(monthly, float)
    M = len(m)
    if M < 4:
        return -np.inf
    rng = np.random.default_rng(seed)
    nb = -(-M // block)
    starts = rng.integers(0, M - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :M]
    return float(np.percentile(calmar_paths(m[idx], 12), q))


def stationary_idx(T: int, block: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """정상(stationary) 블록 부트스트랩 날 번호(n × T, 원형) — 평균 블록 길이 ``block`` (리뷰어 ``calmar_boot.py`` 와 같은 과정, 벡터화)."""
    new = rng.random((n, T)) < 1.0 / block
    new[:, 0] = True
    start = rng.integers(0, T, size=(n, T))
    pos = np.where(new, np.arange(T)[None, :], 0)
    np.maximum.accumulate(pos, axis=1, out=pos)
    s0 = np.take_along_axis(start, pos, axis=1)
    return (s0 + np.arange(T)[None, :] - pos) % T


def paired_calmar_boot(a: np.ndarray, b: np.ndarray, *, block: float = 63, n: int = 2000, seed: int = 0) -> dict:
    """같은 날 번호로 두 일별 수익을 재표본 → Calmar 차이(a − b) 90% CI·P(차이 > 0), CAGR 차이도(리뷰 M3, 252일 연환산, MDD 하한 없음)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    out_c, out_g = [], []
    for k in range(0, n, 250):
        ix = stationary_idx(len(a), block, min(250, n - k), rng)
        ca = calmar_paths(a[ix], 252, mdd_floor=1e-9)
        cb = calmar_paths(b[ix], 252, mdd_floor=1e-9)
        out_c.append(ca - cb)
        ga = np.prod(1 + a[ix], axis=1) ** (252 / len(a)) - np.prod(1 + b[ix], axis=1) ** (252 / len(b))
        out_g.append(ga)
    dc, dg = np.concatenate(out_c), np.concatenate(out_g)
    return {"dcalmar_lo": float(np.percentile(dc, 5)), "dcalmar_med": float(np.median(dc)),
            "dcalmar_hi": float(np.percentile(dc, 95)), "p_calmar": float((dc > 0).mean()),
            "dcagr_lo": float(np.percentile(dg, 5)), "dcagr_hi": float(np.percentile(dg, 95)), "p_cagr": float((dg > 0).mean())}
