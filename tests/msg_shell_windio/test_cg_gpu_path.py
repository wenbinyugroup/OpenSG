"""Parity test for the GPU-compatible vmapped projected-CG homo path.

solver="cg" replaces every direct factorization with a device-resident,
jit + vmap-over-load-cases matrix-free CG (sparse_projected_cg): identical
code runs batched on CPU and parallel on GPU.  Verified parity vs the direct
solvers (2026-08-10, CPU): beam RHC V0+V1 2.4e-7, TPMS shell_sg3d 1.5e-8,
plate RHC 1.5e-6 (its 1e-6 CG tolerance class).

On a CPU-only machine CG is MUCH slower than PARDISO (beam 81 s vs 3.3 s),
so this test is gated:  OPENSG_CG_TESTS=1 pytest tests/msg_shell_windio -q
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENSG_CG_TESTS") != "1",
    reason="CPU-slow GPU path; set OPENSG_CG_TESTS=1 to run")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RHC = os.path.join(ROOT, "examples", "OpenSG-solid",
                   "7_get_beam_props_from_SG", "RHC_SW_2UC_45.yaml")


def test_beam_cg_matches_direct(tmp_path):
    import jax.numpy as jnp
    from opensg_solid.sg_homo import plate_homo_2d

    mp = jnp.array([
        (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
        (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
        (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
    ang = jnp.array([45.0, -45.0, 0.0])
    kw = dict(material_param=mp, angles=ang, n_model=1, plot=False,
              workdir=str(tmp_path))
    Cd = np.asarray(plate_homo_2d(RHC, **kw)["C_eff"])
    Cc = np.asarray(plate_homo_2d(RHC, solver="cg", **kw)["C_eff"])
    cut = np.abs(Cd).max() / 1e3
    m = np.abs(Cd) > cut
    rel = np.abs(Cc[m] / Cd[m] - 1.0).max()
    assert rel < 1e-5, "beam cg vs direct: max rel %.2e" % rel
