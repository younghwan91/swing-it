"""``scripts/verify_report.py`` — 검사 **자체**가 맞는지.

일일 리포트 검증이 2026-09~10 의 27일 중 25일 빨갰다. 그중 H 층 두 줄은 화면의
결함이 아니라 검사의 가정이었다:

* ``G·End 는 끝`` — 커서를 3 에 두고 G 뒤에 ``> 3`` 인지를 봤다. 관문을 통과한
  섹터가 적은 날에는 전 종목(t)이 0~2행, 종합이 0행이라 커서가 **정확히 끝에
  가 있는데도** "끝으로 안 간다" 고 했다(09-07: 2행 목록에서 arow=1).
* ``커서가 범위 안에`` — 목록 크기와 무관하게 커서 3 에서 출발해, 0행인 전 종목
  화면에서 (설계대로) 안 듣는 ``s``·``r`` 뒤에 "arow=3 (행 0)" 이라고 했다.
  범위를 벗어난 것은 키가 아니라 검사가 만든 출발점이었다.

거짓 경보가 매일 뜨면 진짜 경보가 그 소음에 묻힌다. 그래서 두 방향을 다 본다 —
**짧은 목록에서 조용한가**, 그리고 **끝 키가 정말 끝에 못 가면 여전히 무는가**.
DB·네트워크 없이 합성 페이로드만 쓴다.
"""
from __future__ import annotations

import curses
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

END_CHECK = "g·Home 은 처음"
RANGE_CHECK = "커서가 범위 안에"


def _load():
    spec = importlib.util.spec_from_file_location(
        "verify_report", REPO / "scripts" / "verify_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flow(n_names: int, gsec: float | None = 0.6) -> dict:
    """섹터 하나·종목 ``n_names`` 개 — 전 종목(t) 목록이 **짧은** 날.

    ``combined`` 는 비운다 — G 가 나오는 창이 2개 미만인 날(2026-10-07 이 그랬다)
    ``sector_numbers`` 가 종합 축을 아예 안 싣는다.
    """
    names = {}
    for i in range(n_names):
        inst = 1000.0 - 100.0 * i
        names[f"{i:06d}"] = {
            "name": f"종목{i}", "sector": "S", "market": "거래소", "cap": 100000.0,
            "win": {"5": {"inst": inst * (0.7 - 0.1 * i), "forgn": 0.0,
                          "indiv": 0.0, "etc": 0.0, "tv": 1000.0,
                          "invtrt": 100.0 + 50 * i, "penfnd_etc": 60.0 - 10 * i,
                          "ret": 0.0},
                    "20": {"inst": inst, "forgn": 0.0, "indiv": 0.0, "etc": 0.0,
                           "tv": 4000.0 + 500 * i, "invtrt": 600.0 - 100 * i,
                           "penfnd_etc": 300.0 - 40 * i, "ret": 0.0}}}
    block = {"from": "2026-08-01", "to": "2026-08-28", "k": 1.0, "b": 0.0, "t": 2.0,
             "rows": [{"sector": "S", "n_all": n_names, "thin": False,
                       "inst": 1000.0, "forgn": 0.0, "indiv": 0.0, "etc": 0.0,
                       "cap": 100000.0, "accel": 1.0, "ret": 0.0, "a_idx": 1.0,
                       "cap_idx": 100000.0, "G": gsec, "G_pass": gsec is not None,
                       "pct1y": {}, "spark": {}, "top": {}}]}
    return {"asof": "2026-08-28", "finalized": True,
            "dates": ["2026-08-01", "2026-08-28"], "names": names,
            "blocks": {f"{w}|{m}": block for w in (5, 20, 60, 120)
                       for m in ("전체", "거래소", "코스닥")},
            "combined": {}}


def _ledger() -> dict:
    """원장 쪽 합성 페이로드 — ``test_tui_ledger_view`` 의 픽스처와 같은 모양.

    섹터가 **4개**다 — 원장 표가 4행이라 예전 검사(`> 3`)는 원장 쪽에서도 같은
    거짓 경보를 냈다("ledger/원장 G 가 끝으로 안 간다 (row=3)"). 실제 리포트는
    27섹터라 안 드러났을 뿐이다. 섹터를 늘리면 H 층 폭 전수가 수십 배 느려진다.
    """
    from swing_it.tui.ledger_view import ACTOR_KEYS

    n = 40
    dates = [f"2026-0{1 + i // 28}-{1 + i % 28:02d}" for i in range(n)]
    secs = ["전기/전자", "운송장비/부품", "부동산", "출판/매체복제"]
    mkts = ["거래소", "코스닥"]

    def cell(seed: int):
        out = {k: [round((-1) ** (i + j) * (seed + i * (j + 1)) * 1.5, 2)
                   for i in range(n)] for j, k in enumerate(ACTOR_KEYS)}
        out["tv"] = [1000.0] * n
        return out

    return {"dates": dates, "sectors": secs, "markets": mkts,
            "flows": {m: {s: cell(1 + i * 7 + len(m)) for i, s in enumerate(secs)}
                      for m in mkts},
            "cap": {m: {s: 10000.0 * (i + 1) for i, s in enumerate(secs)}
                    for m in mkts},
            "n_by_sector": {m: {s: 50 for s in secs} for m in mkts},
            "n_names": 100, "finalized": True}


def _h_fails(D: dict) -> list[str]:
    vr = _load()
    vr.FAILS.clear()
    vr.layer_h(D, _ledger())
    return list(vr.FAILS)


@pytest.mark.parametrize("n_names", [2, 3])
def test_end_key_check_accepts_the_true_last_row_of_a_short_list(n_names):
    """회귀 — 2~3행 목록에서 G 가 끝(1·2행)에 갔는데 검사가 `> 3` 을 요구했다.

    종합이 0행인 것도 같이 태운다(10-07 의 `flow/종합 … row=0`).
    주입: 검사를 예전 `v <= 3` 판정으로 되돌리면 실패한다.
    """
    from swing_it.tui.flow_view import State

    st = State(_flow(n_names))
    st.allv = True
    assert 0 < len(st.all_picks()) <= 3, "픽스처가 짧은 목록을 못 만든다 — 검사가 헛돈다"
    fails = _h_fails(_flow(n_names))
    assert not [f for f in fails if END_CHECK in f], fails


def test_cursor_range_check_starts_from_a_state_the_app_can_reach():
    """회귀 — 2행 목록에 커서 3 을 박고 시작해 "arow=3 (행 2)" 라고 했다.

    주입: 출발 커서를 다시 목록과 무관한 3 으로 두면 실패한다.
    """
    fails = _h_fails(_flow(2))
    assert not [f for f in fails if RANGE_CHECK in f], fails


def test_end_key_check_says_which_screens_it_could_not_see(capsys):
    """관문 전멸 — 어느 구간·시장에서도 목록이 없으면 **못 봤다고 적는다.**

    빈 목록에서 "0 이면 맞다" 로 조용히 통과하면 검사는 아무것도 안 본 것이다.
    """
    vr = _load()
    vr.FAILS.clear()
    vr.layer_h(_flow(3, gsec=None), _ledger())
    assert not [f for f in vr.FAILS if END_CHECK in f or RANGE_CHECK in f], vr.FAILS
    line = [ln for ln in capsys.readouterr().out.splitlines() if END_CHECK in ln][0]
    assert "못 본 화면" in line and "전 종목(t)" in line and "종합" in line, line


def test_end_key_check_still_bites_when_end_stops_short(monkeypatch, capsys):
    """반대 방향 — 끝 키가 **한 칸 모자라게** 서면 검사가 여전히 빨개야 한다.

    `> 3` 을 지운 대가로 검사가 이빨을 잃으면 안 된다. 짧은 목록(3행)에서
    G 가 끝-1 에 서는 결함은 예전 검사로는 **원리적으로** 못 잡던 것이다.
    """
    from swing_it.tui import flow_app as FA

    real = FA.handle_key

    def short(st, k, *a, **kw):
        out = real(st, k, *a, **kw)
        if st.allv and k in (ord("G"), curses.KEY_END):
            st.arow = max(st.arow - 1, 0)
        return out

    monkeypatch.setattr(FA, "handle_key", short)
    fails = _h_fails(_flow(3))
    assert [f for f in fails if END_CHECK in f], (
        "끝 키가 끝에 못 가는데 검사가 통과했다: " + repr(fails))
    assert "끝=2" in capsys.readouterr().out
