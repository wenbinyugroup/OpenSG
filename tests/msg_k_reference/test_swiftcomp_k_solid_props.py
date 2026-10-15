"""Gate: msg-solid 3-D SG homogenization vs the SwiftComp `.K` reference.

The coverage that did not exist.  The msg-shell beam route has been gated
against VABS `.K` files for a while; the msg-solid 3-D route -- the one
whose SG measure changed (omega for a 3-D SG is the node BOUNDING-BOX
volume, the periodic unit cell, NOT the summed material volume) -- had no
automated `.K` gate at all, so nothing stopped that normalization from
silently regressing by the relative density.

Reference: SwiftComp's own `.K` for the same two TPMS (Schwarz-P) unit
cells, vendored beside the tracked SwiftComp decks --
    examples/OpenSG-solid/6_get_solid_props_from_3D_SG/
        sample_1/Sample_1.sc  (546k linear tets, relative density 0.300000)
        sample_2/Sample_2.sc  (192k linear tets, relative density 0.085694)
        sample_{1,2}/Sample_{1,2}.sc.k
The .sc goes straight into plate_homo_2d the way sample_{1,2}/
run_sample_{1,2}.py feed it (n_model 3, periodic, material_param); the
converted yaml/msh land in tmp_path.  (The example folder's Sample_?.yaml
is a gitignored conversion artifact, so a fresh clone has none.)
Both cells are aluminium E = 69 GPa, nu = 0.3, periodic in all three
directions; the material is passed in Pa while the `.sc.k` is in MPa, so
the unit convention is stated to the verifier rather than assumed.

Tolerance: MEASURED worst relative difference is 4.9e-8 (Sample_1, C22)
and 4.2e-8 (Sample_2, C55) -- both at the 8-significant-digit print
quantization of the `.K` itself, i.e. the two codes agree to every digit
SwiftComp prints.  The gate is 1e-6, ~20x the measured worst (headroom
for print rounding and solver-thread reassociation) and still ~7 orders
of magnitude tighter than the relative-density regression it exists to
catch.  Each case additionally asserts that the gate REJECTS that exact
regression -- a gate that cannot fail is worthless.

Marked `slow` (Sample_1 ~76 s, Sample_2 ~23 s, .sc read + yaml/msh conversion
included; ~100 s for the pair on msg at 4 threads),
so the fast suite deselects them the same way it deselects the other
slow-marked tests in this repo -- with `-m "not slow"`; pytest has no
default marker filter here, so a bare run DOES include them.

Run:  pytest tests/msg_k_reference -q -m "not slow"   fast: reader only
      pytest tests/msg_k_reference -q -m slow         these two gates
      pytest tests/msg_k_reference -q                 everything (~100 s)
(env opensg_2_0)
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
S6 = os.path.join(ROOT, "examples", "OpenSG-solid",
                  "6_get_solid_props_from_3D_SG")

RTOL = 1.0e-6            # see the module docstring: ~20x the measured worst

# sample, relative density (material volume / bounding-box volume) = the
# factor the OLD omega convention would have multiplied the law by
CASES = [("1", 0.300000), ("2", 0.085694)]

TERMS = ["C11", "C12", "C13", "C22", "C23", "C33", "C44", "C55", "C66"]

# the decks' aluminium in Pa (the .sc embeds 69000 MPa, 0.3), as the
# sample runners pass it
E_PA, NU = 69.0e9, 0.30
G_PA = E_PA / (2 * (1 + NU))


@pytest.mark.slow
@pytest.mark.parametrize("s,rho_rel", CASES, ids=["Sample_1", "Sample_2"])
def test_solid_3d_props_vs_swiftcomp_k(tmp_path, s, rho_rel):
    import jax.numpy as jnp
    from opensg_solid.sg_homo import plate_homo_2d
    from opensg_solid.io.k_file import compare_to_K, read_solid_k

    sc = os.path.join(S6, "sample_%s" % s, "Sample_%s.sc" % s)
    kref = os.path.join(S6, "sample_%s" % s, "Sample_%s.sc.k" % s)
    assert os.path.exists(sc) and os.path.exists(kref)

    # the SwiftComp deck itself, run as run_sample_<s>.py runs it (the .K
    # digit-parity mode); workdir keeps the converted .yaml/.msh and the
    # .out out of the example folder
    r = plate_homo_2d(sc, material_param=jnp.array([(E_PA, E_PA, E_PA,
                                                     G_PA, G_PA, G_PA,
                                                     NU, NU, NU)]),
                      angles=jnp.array([0.0]), n_model=3,
                      workdir=str(tmp_path), plot=False, boundary="periodic")
    assert r["n_model"] == 3 and r["boundary"] == "periodic"

    # the Pa material passed above is the deck's own one, not a substitute
    mats = r["sc"]["materials"]
    assert len(mats) == 1 and mats[1]["type"] == 0
    assert abs(mats[1]["E"] * 1.0e6 / E_PA - 1.0) < 1e-12
    assert abs(mats[1]["nu"] - NU) < 1e-12

    C = np.asarray(r["C_eff"])
    assert np.abs(C - C.T).max() < 1e-9 * np.abs(C).max()

    # ---- THE GATE: all nine independent terms, Pa vs the .sc.k's MPa
    rep = compare_to_K(C, kref, "swiftcomp_solid", ref_unit="MPa",
                       value_unit="Pa", terms=TERMS, symmetrize=True,
                       rtol=RTOL, label="msg-solid Sample_%s" % s)
    assert rep["passed"], "\n" + rep["table"]
    assert rep["worst"]["rel_diff"] < RTOL

    # ---- and the normalization named directly, so a failure above is
    # diagnosed rather than just reported: omega is the node bounding box,
    # and these cells are the unit cube, so it is 1 -- NOT the material
    # volume (which would be the relative density, 0.300 / 0.0857)
    nodes = np.asarray(r["sc"]["nodes"], float)
    bbox = float(np.prod(nodes.max(0) - nodes.min(0)))
    assert abs(r["omega"] / bbox - 1.0) < 1e-12, (
        "omega %.10g is not the node bounding-box volume %.10g -- the 3-D "
        "solid SG measure regressed" % (r["omega"], bbox))

    # ---- the engineering constants OpenSG prints must match too
    got = read_solid_k(str(tmp_path / ("Sample_%s.out" % s)))["constants"]
    want = read_solid_k(kref)["constants"]
    for k in want:
        scale = 1.0e-6 if k[0] in "EG" else 1.0      # E/G in Pa, nu neither
        assert abs(got[k] * scale / want[k] - 1.0) < 1e-6, (
            "%s: OpenSG %.8g vs SwiftComp %.8g" % (k, got[k] * scale,
                                                   want[k]))

    # ---- NEGATIVE CONTROL 1: the omega bug.  C is exactly linear in
    # 1/omega, so dividing by the material volume instead of the cell
    # volume scales the whole law by the relative density.
    bug = compare_to_K(C * rho_rel, kref, "swiftcomp_solid",
                       ref_unit="MPa", value_unit="Pa", terms=TERMS,
                       symmetrize=True, rtol=RTOL,
                       label="omega = material volume (seeded regression)")
    assert not bug["passed"] and bug["n_fail"] == len(TERMS)
    assert abs(bug["worst"]["rel_diff"] - (1.0 - rho_rel)) < 1e-3

    # ---- NEGATIVE CONTROL 2: a Voigt index permutation.  A Schwarz-P
    # cell is cubic-symmetric, so swapping two shears (or two normals) is
    # physically invisible and no verifier could see it; the catchable
    # mis-convention is an off-by-one that mixes a NORMAL with a SHEAR
    # slot, [11 22 33 23 13 12] read as [11 22 23 33 13 12].
    P = [0, 1, 3, 2, 4, 5]
    perm = compare_to_K(C[np.ix_(P, P)], kref, "swiftcomp_solid",
                        ref_unit="MPa", value_unit="Pa", terms=TERMS,
                        symmetrize=True, rtol=RTOL,
                        label="Voigt 33<->23 swap (seeded regression)")
    assert not perm["passed"] and perm["worst"]["rel_diff"] > 0.5

    # ---- and a 0.01 % drift, two orders below the coarsest thing anyone
    # would call a modeling difference, must still be caught
    drift = compare_to_K(C * 1.0001, kref, "swiftcomp_solid",
                         ref_unit="MPa", value_unit="Pa", terms=TERMS,
                         symmetrize=True, rtol=RTOL, label="0.01 % drift")
    assert not drift["passed"] and drift["n_fail"] == len(TERMS)
