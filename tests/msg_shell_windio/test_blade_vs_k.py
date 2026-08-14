"""Unit test: the IEA-22 verifier demo (in-memory Blade vs committed .K).

The demo examples/OpenSG_shell/windio/7_verify_blade_timo_vs_K.py IS the
test: it recomputes every committed props/iea22_rXXXX.K station fully in
memory (pynumad.sg_homo.timo) and asserts the beam stiffness AND mass
matrices agree within tolerance.  Here it runs as a subprocess so a
verification failure (nonzero exit) fails the suite with the demo's own
per-station log attached.

Run:  pytest tests/msg_shell_windio/test_blade_vs_k.py -q
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEMO_DIR = os.path.join(ROOT, "examples", "OpenSG_shell", "windio")
DEMO = "7_verify_blade_timo_vs_K.py"


def test_blade_timo_vs_vabs_k():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep \
        + env.get("PYTHONPATH", "")
    env.pop("FE_JAX_CORE", None)
    p = subprocess.run([sys.executable, DEMO], cwd=DEMO_DIR, env=env,
                       capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, \
        "verifier failed:\n" + p.stdout[-3000:] + p.stderr[-2000:]
    assert "FAIL" not in p.stdout
