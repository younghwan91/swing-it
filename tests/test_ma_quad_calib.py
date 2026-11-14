"""관문 0 보정·기술·관문 판정 (사전등록 문턱 그대로)."""
import numpy as np
import pytest

from research.signals import ma_quad_calib as C

NS = (10, 15, 20, 30, 40)
KS = (3, 5, 7, 10)


def test_bin_edges_left_closed():
    p = np.array([0.5, 0.4999, 1.0, 0.9, 0.0])
    y = np.zeros(5)
    r = C.reliability(p, y)
    assert r.loc[r.lo == 0.5, "n"].iloc[0] == 1          # 0.5 -> [0.5,0.6)
    assert r.loc[np.isclose(r.lo, 0.9), "n"].iloc[0] == 2  # 0.9 와 1.0
    assert r.loc[r.lo == 0.4, "n"].iloc[0] == 1
    assert r.n.sum() == 5
    assert np.isnan(r.loc[r.lo == 0.6, "p_mean"].iloc[0])


def _months(n):
    years = 2017 + (np.arange(n) % 7)
    mon = 1 + (np.arange(n) // 7) % 12
    return years * 100 + mon, years


def test_perfect_calibration_passes():
    rng = np.random.default_rng(1)
    n = 60000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.random(n) < p).astype(float)
    pB = np.full(n, 0.5)
    month, year = _months(n)
    v = C.cell_verdict(p, pB, y, month, year)
    assert v["calib_ok"] is True
    assert v["skill_ok"] is True


def test_calibration_fails_when_no_high_bins():
    rng = np.random.default_rng(2)
    n = 5000
    p = rng.uniform(0.0, 0.49, n)
    y = (rng.random(n) < p).astype(float)
    month, year = _months(n)
    v = C.cell_verdict(p, np.full(n, y.mean()), y, month, year)
    assert v["calib_ok"] is False
    assert v["pass"] is False


def test_high_bin_miscalibrated_fails():
    n = 4000
    p = np.full(n, 0.75)
    y = np.zeros(n)
    y[: int(n * 0.6)] = 1  # 0.15 차이
    month, year = _months(n)
    v = C.cell_verdict(p, np.full(n, 0.6), y, month, year)
    assert v["calib_ok"] is False


def test_small_high_bin_ignored_but_needs_one():
    n = 1000
    p = np.full(n, 0.75)
    y = np.zeros(n)
    y[:750] = 1
    p2 = np.concatenate([p, np.full(100, 0.95)])
    y2 = np.concatenate([y, np.zeros(100)])  # n=100<300 이므로 무시
    # ECE 가 문턱을 넘지 않도록 큰 저확률 덩어리를 같이 둔다
    m, yr = _months(len(p2))
    v = C.cell_verdict(p2, np.full(len(p2), 0.7), y2, m, yr)
    hb = {round(b["lo"], 1): b for b in v["high_bins"]}
    assert hb[0.9]["n"] == 100
    # 0.7 구간만 판정 대상: 오차 0
    assert abs(hb[0.7]["p_mean"] - hb[0.7]["y_rate"]) <= 0.05


def _bss_case(target):
    """pB=0.5 상수, Brier(pB)=0.25. pA 로 BSS=target 을 만든다."""
    n = 10000
    y = np.zeros(n)
    y[: n // 2] = 1
    # pA = 0.5 + e*(2y-1): Brier = (0.5-e)^2
    e = 0.5 - np.sqrt(0.25 * (1 - target))
    pA = 0.5 + e * (2 * y - 1)
    return pA, np.full(n, 0.5), y


def test_bss_threshold_001():
    pA, pB, y = _bss_case(0.009)
    assert C.brier(pA, y) == pytest.approx(0.25 * (1 - 0.009), rel=1e-9)
    m, yr = _months(len(y))
    v = C.cell_verdict(pA, pB, y, m, yr)
    assert v["bss"] == pytest.approx(0.009, abs=1e-9)
    assert v["skill_ok"] is False
    pA, pB, y = _bss_case(0.011)
    v = C.cell_verdict(pA, pB, y, m, yr)
    assert v["skill_ok"] is True


def test_bss_exactly_001_passes():
    # 경계: bss >= 0.01 (등호 포함)
    # 임의 pA 로 정확히 0.01 을 만들기 어려우니 내부 판정 함수로 직접
    assert C._skill_ok(0.01, 1e-9) is True
    assert C._skill_ok(0.01, 0.0) is False
    assert C._skill_ok(0.0099999, 0.5) is False


def test_brier_b_zero_gives_nan_skill_fail():
    n = 100
    y = np.ones(n)
    pB = np.ones(n)
    pA = np.full(n, 0.9)
    m, yr = _months(n)
    v = C.cell_verdict(pA, pB, y, m, yr)
    assert np.isnan(v["bss"])
    assert v["skill_ok"] is False


def test_empty_year_counts_as_not_positive():
    pA, pB, y = _bss_case(0.2)
    n = len(y)
    year = 2017 + (np.arange(n) % 7)
    month = year * 100 + 1
    keep = year != 2019
    v = C.cell_verdict(pA[keep], pB[keep], y[keep], month[keep], year[keep])
    assert v["years_pos"] == 6
    assert v["years_ok"] is True
    # 2018, 2019 모두 비면 5 → 여전히 ok, 세 해 비면 4 → 실패
    keep = ~np.isin(year, [2018, 2019, 2020])
    v = C.cell_verdict(pA[keep], pB[keep], y[keep], month[keep], year[keep])
    assert v["years_pos"] == 4
    assert v["years_ok"] is False


def test_nan_and_length_rejected():
    p = np.array([0.5, np.nan])
    with pytest.raises(ValueError):
        C.cell_verdict(p, np.array([0.5, 0.5]), np.array([0.0, 1.0]),
                       np.array([201701, 201701]), np.array([2017, 2017]))
    with pytest.raises(ValueError):
        C.cell_verdict(np.array([0.5]), np.array([0.5, 0.5]), np.array([0.0]),
                       np.array([201701]), np.array([2017]))


def test_verdict_is_json_plain():
    import json
    rng = np.random.default_rng(3)
    n = 3000
    p = rng.uniform(0, 1, n)
    y = (rng.random(n) < p).astype(float)
    m, yr = _months(n)
    v = C.cell_verdict(p, np.full(n, 0.5), y, m, yr)
    assert "bss_lo_q" in v
    json.dumps(v)  # numpy 타입이 남아 있으면 TypeError


def test_bss_ci_cluster_resampling():
    rng = np.random.default_rng(4)
    nc, per = 20, 50
    cl = np.repeat(np.arange(nc), per)
    y = (rng.random(nc * per) < 0.5).astype(float)
    sign = np.where(rng.random(len(y)) < 0.6, 1.0, -1.0)  # 60% 맞는 신호 -> 군집마다 BSS 가 다르다
    pA = 0.5 + 0.1 * sign * (2 * y - 1)
    pB = np.full(len(y), 0.5)
    pt, lo, hi = C.bss_ci(pA, pB, y, cl, n_boot=500)
    # 군집 안 복제(10배)는 군집 수가 같으므로 CI 폭이 거의 같아야 한다
    pt2, lo2, hi2 = C.bss_ci(np.repeat(pA, 10), np.repeat(pB, 10), np.repeat(y, 10),
                             np.repeat(cl, 10), n_boot=500)
    assert pt == pytest.approx(pt2)
    assert (hi2 - lo2) == pytest.approx(hi - lo, rel=0.05)
    # 군집 수가 적으면 넓다
    few = cl < 3
    _, lo3, hi3 = C.bss_ci(pA[few], pB[few], y[few], cl[few], n_boot=500)
    assert (hi3 - lo3) > (hi - lo)
    # 결정적
    assert C.bss_ci(pA, pB, y, cl, n_boot=200) == C.bss_ci(pA, pB, y, cl, n_boot=200)


def _grid(cells):
    return {(n, k): (n, k) in cells for n in NS for k in KS}


def test_gate_needs_block_of_8():
    # 열 3개(N=10,15,20)×k 4 = 12 칸 → 덩어리 12
    cells = {(n, k) for n in NS[:3] for k in KS}
    g = C.gate_verdict(_grid(cells))
    assert g["n_pass"] == 12 and g["largest_block"] == 12 and g["pass"] is True
    # 체스판 10 칸: 서로 이웃하지 않아 덩어리 1
    cells = {(n, k) for i, n in enumerate(NS) for j, k in enumerate(KS) if (i + j) % 2 == 0}
    assert len(cells) == 10
    g = C.gate_verdict(_grid(cells))
    assert g["pass"] is False and g["largest_block"] == 1
    # 12칸: 7짜리 덩어리 + 분리된 5칸
    block7 = {(10, 3), (10, 5), (10, 7), (10, 10), (15, 3), (15, 5), (15, 7)}
    rest = {(30, 3), (30, 5), (30, 7), (30, 10), (40, 5)}
    g = C.gate_verdict(_grid(block7 | rest))
    assert g["n_pass"] == 12 and g["largest_block"] == 7 and g["pass"] is False
    # 덩어리 8 이 되도록 (15,10) 추가 대신 rest 일부를 교체: 13칸
    g = C.gate_verdict(_grid(block7 | {(15, 10)} | rest))
    assert g["largest_block"] == 8
    assert g["n_pass"] == 13 and g["pass"] is True


def test_gate_needs_12_cells():
    cells = {(n, k) for n in NS[:2] for k in KS} | {(20, 3), (20, 5), (20, 7)}
    g = C.gate_verdict(_grid(cells))
    assert g["n_pass"] == 11 and g["largest_block"] == 11 and g["pass"] is False


def test_gate_key_mismatch_raises():
    d = _grid(set())
    d.pop((10, 3))
    with pytest.raises(ValueError):
        C.gate_verdict(d)


# --- 경계 등호·입력 범위 (Task 4 리뷰 minor) -------------------------------------------

def _mix(p_hi, n_hi, ones_hi, extra=()):
    """높은 구간 덩어리 + ECE 를 묽히는 잘 보정된 저확률 덩어리(p=0.15, n=40000, 실현 0.15)."""
    parts = [(p_hi, n_hi, ones_hi), (0.15, 40000, 6000), *extra]
    p = np.concatenate([np.full(n, v) for v, n, _ in parts])
    y = np.concatenate([np.r_[np.ones(o), np.zeros(n - o)] for _, n, o in parts])
    m, yr = _months(len(p))
    return p, y, m, yr


def test_high_gap_exactly_005_passes():
    p, y, m, yr = _mix(0.75, 10000, 7000)      # 0.75 − 0.70 = 0.05(부동소수 0.05000000000000004)
    v = C.cell_verdict(p, np.full(len(p), y.mean()), y, m, yr)
    assert v["calib_ok"] is True
    p, y, m, yr = _mix(0.75, 10000, 6999)      # 0.0501
    v = C.cell_verdict(p, np.full(len(p), y.mean()), y, m, yr)
    assert v["calib_ok"] is False


def test_ece_exactly_002_passes():
    n = 10000
    for ones, ok in ((7300, True), (7299, False)):  # 0.75 − 0.73 = 0.02 / 0.0201
        p = np.full(n, 0.75)
        y = np.r_[np.ones(ones), np.zeros(n - ones)]
        m, yr = _months(n)
        v = C.cell_verdict(p, np.full(n, 0.5), y, m, yr)
        assert v["calib_ok"] is ok, (ones, v["ece"])


def test_high_bin_n_299_ignored_300_counted():
    for n_bad, ok in ((299, True), (300, False)):
        # 잘 맞는 0.7 구간(n=1000) + 크게 어긋난 0.9 구간(p=0.95, 실현 0) n_bad
        p, y, m, yr = _mix(0.75, 1000, 750, extra=((0.95, n_bad, 0),))
        v = C.cell_verdict(p, np.full(len(p), y.mean()), y, m, yr)
        hb = {round(b["lo"], 1): b for b in v["high_bins"]}
        assert hb[0.9]["n"] == n_bad
        assert v["ece"] <= 0.02
        assert v["calib_ok"] is ok, n_bad


def test_prob_and_label_range_rejected():
    m, yr = np.array([201701, 201701]), np.array([2017, 2017])
    with pytest.raises(ValueError):
        C.cell_verdict(np.array([0.5, 1.01]), np.array([0.5, 0.5]), np.array([0.0, 1.0]), m, yr)
    with pytest.raises(ValueError):
        C.cell_verdict(np.array([0.5, 0.5]), np.array([-0.01, 0.5]), np.array([0.0, 1.0]), m, yr)
    with pytest.raises(ValueError):
        C.cell_verdict(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.0, 0.5]), m, yr)
    with pytest.raises(ValueError):
        C.reliability(np.array([1.5]), np.array([1.0]))
    with pytest.raises(ValueError):
        C.brier(np.array([0.5]), np.array([2.0]))
