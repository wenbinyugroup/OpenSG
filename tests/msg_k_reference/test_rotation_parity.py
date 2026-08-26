"""ONE rotation convention across the package -- the 2026-08-25 sign
unification's guard.

THE BUG THIS PINS.  Until 2026-08-25 the general SG engine's R_Sig
(sg_materials.rotate_C_matrix / _rotate_C_np, the map under every yaml
`angle:` block) was the rm_plate_1D rotation_6x6 at the OPPOSITE angle
sign: the same ply angle built MIRROR laminates on the two paths.  On
the [45/0/-30/90] stack below the general-vs-rm_plate_1D ABD disagreed
at rel 1.4e-1 (A16-type couplings as exact sign mirrors, e.g.
A6[0,5] = +5.5611e5 vs -5.5611e5); sign-insensitive layups (0/90,
[45/-45]s A-matrix) hid it from every other test.  The package
convention is now the VABS/OpenSG-TW lineage everywhere: `angle: a` is
the fiber at +a, right-handed about the layer normal y3 -- the same
sense as the VABS .sg and SwiftComp .sc layer tables the io writers
copy the angle into VERBATIM.

Two gates:
  kernel  -- _rotate_C_np(C, t) == rotation_6x6(t) C rotation_6x6(t)^T
             exactly (the two modules share one matrix again);
  ABD     -- a [45/0/-30/90] 1-D through-thickness SG fed to the
             general engine as type-1 `angle:` blocks must reproduce
             rm_plate_1D's MSG ABD to 1e-10 relative, and the NEGATED
             angles must NOT (the mirror is a real, detected
             difference -- the test has teeth).

Fast: one 4-element 1-D homogenization.

Run:  pytest tests/msg_k_reference/test_rotation_parity.py -q
(env opensg_2_0)
"""
import numpy as np

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_mesh import laminate_to_sg
from opensg_solid.sg_materials import _rotate_C_np
from opensg_solid.rm_plate_1D.msg_materials import (compute_ABD_matrix,
                                                    build_stiffness_6x6,
                                                    rotation_6x6)

E = [108000.0, 8000.0, 8000.0]
G = [4000.0, 4000.0, 3000.0]
NU = [0.32, 0.32, 0.3]
ENG9 = E + G + NU                  # yaml order E1 E2 E3 G12 G13 G23 v12 v13 v23
DB = {"ply": {"E": E, "G": G, "nu": NU}}
ANGLES = [45.0, 0.0, -30.0, 90.0]
THICK = [0.4, 0.3, 0.2, 0.1]       # asymmetric: every ABD block sign-sensitive


def test_rotation_kernel_is_the_rm_plate_rotation():
    C = build_stiffness_6x6(E, G, NU)
    for t in (37.5, -12.25, 90.0):
        R = rotation_6x6(t)
        want = R @ C @ R.T
        got = _rotate_C_np(C, t)
        assert np.abs(got - want).max() <= 1e-13 * np.abs(want).max(), t


def _general_abd(angles):
    """The [45/0/-30/90] laminate through the GENERAL engine's `angle:`
    block ingestion (type-1 constants + angle -- NOT the pre-rotated
    type-2 route laminate_to_sg emits, which never exercised the bug)."""
    sc = laminate_to_sg(THICK, [0.0] * len(THICK), ["ply"] * len(THICK), DB)
    sc["materials"] = {k + 1: {"type": 1, "engineering": list(ENG9),
                               "angle": float(angles[k])}
                       for k in range(len(THICK))}
    r = plate_homo_2d(sc, n_model=2, refined=0, plot=False)
    return np.asarray(r["C_eff"], float)


def test_angle_block_abd_matches_rm_plate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    want, _ = compute_ABD_matrix(THICK, ANGLES, ["ply"] * len(THICK), DB)
    got = _general_abd(ANGLES)
    scale = np.abs(want).max()
    assert np.abs(got - want).max() <= 1e-10 * scale

    # the mirror is a DETECTED difference, not test slack: negating the
    # angles (the pre-unification reading of the same yaml) must miss by
    # orders more than the gate
    mirror = _general_abd([-a for a in ANGLES])
    assert np.abs(mirror - want).max() > 1e-3 * scale
