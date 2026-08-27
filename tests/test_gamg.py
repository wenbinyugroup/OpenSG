"""test_gamg.py -- --solver gamg (iter 4, sg_gamg) law equivalence.

The PETSc GAMG-preconditioned CG backend must reproduce the direct
factorization's law on the same SG: classical (6x6 ABD) and
shear-refined (8x8 ABDG, exercising the make_constrained_solver ladder
twin) both gate at 1e-8 relative.  The SG is a small checked-in
example (the 2UC 45-deg laminate 1-D SG), run from a tmp copy so the
.out/.npz sidecars never dirty the example folder.  Skips cleanly when
petsc4py is not installed (the optional [gamg] extra).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("petsc4py")

_YAML = (Path(__file__).resolve().parents[1] / "examples" /
         "OpenSG-solid" / "3_get_plate_props_from_2D_SG" /
         "Plate_1D_SG_2UC_45.yaml")


@pytest.mark.parametrize("refined", [0, 1], ids=["classical", "refined"])
def test_gamg_matches_direct(refined, tmp_path):
    """In:  refined 0 (classical ABD) | 1 (shear-refined ABDG ladder)
    Out: -- (asserts gamg law == direct law to 1e-8 relative)."""
    from opensg_solid.sg_homo import plate_homo_2d

    p = tmp_path / _YAML.name
    shutil.copy(_YAML, p)
    r_d = plate_homo_2d(str(p), refined=refined, solver="direct",
                        recovery=False, plot=False)
    r_g = plate_homo_2d(str(p), refined=refined, solver="gamg",
                        recovery=False, plot=False)
    law_d = np.asarray(r_d["law"], float)
    law_g = np.asarray(r_g["law"], float)
    assert law_d.shape == law_g.shape
    scale = float(np.abs(law_d).max())
    assert np.abs(law_g - law_d).max() / scale <= 1e-8
