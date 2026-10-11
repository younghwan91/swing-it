"""패턴 GA-FW 변형(``PATTERN_GA_VARIANT=fw``) — 모듈 상수가 import 시점에 정해지므로 별도 프로세스에서 확인(리뷰 FW MINOR)."""
import os
import subprocess
import sys

CODE = r"""
import math, sys
sys.path[:0] = ['.', 'src']
import numpy as np
import research.experiments.pattern_ga_run as G
assert G.FW and G.SUB == 'pattern_ga_fw' and G.RESTART_SCREEN == 1_000_000
assert dict(G.NOM)['pyramid'] == ('1r', 'newhigh', 'upper') and dict(G.DISC)['be_r'] == (0.5, 1.0)
assert 'fw_lo' in G.GENE_NAMES
X = G.random_x(np.random.default_rng(0), 3000)
gs = [G.decode(x) for x in X]
assert all(g['pyramid'] != 'off' for g in gs) and all(g['be_r'] in (0.5, 1.0) for g in gs)
assert all(-0.15 <= g['fw_lo'] <= 0.15 for g in gs)
g = gs[0]
assert abs(G.decode(G.encode(g))['fw_lo'] - g['fw_lo']) < 1e-3
h = dict(g, entry='h0', pyramid='newhigh'); assert G.canonical(h)['pyramid'] == '1r'
g2 = dict(g, fw_lo=0.0)
assert G.fw_ok(-0.5, -0.2, g2) and not G.fw_ok(0.1, 0.3, g2) and not G.fw_ok(-0.5, 0.1, g2) and not G.fw_ok(-0.1, -0.3, g2)
assert 'FW하단<' in G.gene_str(g)
print('ok')
"""


def test_fw_variant_menu():
    env = {**os.environ, "PATTERN_GA_VARIANT": "fw"}
    r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok")


def test_default_variant_unchanged():
    env = {k: v for k, v in os.environ.items() if k != "PATTERN_GA_VARIANT"}
    code = ("import sys; sys.path[:0]=['.','src']; import research.experiments.pattern_ga_run as G; "
            "assert not G.FW and 'fw_lo' not in G.GENE_NAMES and dict(G.NOM)['pyramid'][0] == 'off' and G.RESTART_SCREEN == 100_000; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_fwl_variant_menu():
    code = r"""
import sys
sys.path[:0] = ['.', 'src']
import numpy as np
import research.experiments.pattern_ga_run as G
assert G.FW and G.FWL and G.SUB == 'pattern_ga_fwl' and G.FLOOR_PER_YEAR == 9.0 and G.FLOOR_YEAR == 3.0 and G.FEAS_MIN == 300
gs = [G.decode(x) for x in G.random_x(np.random.default_rng(0), 3000)]
assert all(g['e_min'] in (0.25, 0.5) and g['e_accel'] == 1 for g in gs)
assert all(70.0 <= g['rs_min'] <= 90.0 for g in gs) and all(g['stage'] in (2, 3) for g in gs)
assert all(g['pyramid'] != 'off' and -0.15 <= g['fw_lo'] <= 0.15 for g in gs)
print('ok')
"""
    env = {**os.environ, "PATTERN_GA_VARIANT": "fwl"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
