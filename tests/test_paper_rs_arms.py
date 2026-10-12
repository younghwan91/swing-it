"""모의매매 쌍 팔 *_rs(META_EVOLUTION.md '전진 팔 *_rs 사전등록', 2026-10-13) 설정 표 회귀.

쌍 팔은 짝 팔과 cfg 가 같고 정문 애프터마켓 보정 인자(session·aftermarket_fix·unpatched)만 다르다. 기존 팔의 정문 인자·장부는 고정이다.
unpatched 는 '*_rs 보충 — 정문 보정 범위 확대(첫 기록 전)'에서 더했다 — 기존 팔은 session="raw" 라 그 인자가 동작하지 않는다.
"""
from __future__ import annotations

import research.paper.index_paper as IP
import research.paper.leader_paper as LP

EXISTING = ("leader_A", "leader_C", "leader_C20")


def test_rs_arm_cfg_equals_pair():
    assert LP.RS_PAIR == {"leader_A_rs": "leader_A", "leader_C_rs": "leader_C", "leader_C20_rs": "leader_C20"}
    for rs, base in LP.RS_PAIR.items():
        assert LP.ARMS[rs] == LP.ARMS[base]
        assert LP.ARMS[rs] is not LP.ARMS[base]                     # 사본 — 한쪽을 고쳐도 다른 쪽이 따라 바뀌지 않는다


def test_existing_arms_price_and_ledger_fixed():
    assert LP.PRICE_RAW == {"session": "raw", "aftermarket_fix": False, "unpatched": "nan"}
    for arm in EXISTING:
        assert arm not in LP.ARM_PRICE and arm not in LP.ARM_START
    assert LP.LEDGER == "research/paper/leader21_ledger.jsonl"
    assert LP.ARM_LEDGER["leader_C20"] == "research/paper/leader_c20_ledger.jsonl"
    assert IP.ARMS["index27"] == {"ledger": "research/paper/index27_ledger.jsonl", "session": "raw", "aftermarket_fix": False,
                                  "unpatched": "nan"}


def test_rs_arms_only_price_args_differ_and_ledgers_new():
    old_ledgers = {LP.LEDGER, LP.ARM_LEDGER["leader_C20"], IP.LEDGER}
    for rs in LP.RS_PAIR:
        assert LP.ARM_PRICE[rs] == {"session": "regular", "aftermarket_fix": True, "unpatched": "raw"}
        assert LP.ARM_START[rs] == "2026-10-13"
        assert LP.ARM_LEDGER[rs] not in old_ledgers
    rs = IP.ARMS["index27_rs"]
    assert (rs["session"], rs["aftermarket_fix"], rs["unpatched"], rs["start"]) == ("regular", True, "raw", "2026-10-13")
    assert rs["ledger"] not in old_ledgers
