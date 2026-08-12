"""The shell SEGMENT/RING element family is 4-node only -- and the reason is the
missing OPERATORS, not the transverse-shear scheme.

Two things are locked down here.

(1) THE GUARD.  assemble_segment_indep / assemble_constraint /
    build_C_Psi_segment6 / sg_mesh.extract must refuse a triangle (and a mixed
    tri+quad table) with a message that names the cause.  Before
    require_quad_mesh existed, an all-triangle segment yaml ran through boundary
    extraction and BOTH boundary-ring solves and then died several hundred lines
    later on a bare
        ValueError: cannot reshape array of size 17280 into shape (960,24)
    while a MIXED yaml died even earlier inside numpy on
        setting an array element with a sequence ... inhomogeneous shape.
    Neither named the element type, and `opensg <yaml>` reaches both: cli.mesh_kind
    routes ANY 3-or-4-node mesh with n_model 1 to segment_timo_from_3dyaml.

(2) THE DECISION RECORD.  The documented reason for shear='full' on the segment
    is that Dvorkin-Bathe tying "aliases the algebraic drilling shear".  That is
    true for a WARPED quad and false for a triangle, and both halves are asserted
    here so the assessment is not re-litigated from the docstring alone:
      * on a triangle the frame is constant (affine element), so every MITC3
        tying point shares the Gauss point's (a1,a2,n) and the tied rotation
        block stays inside span{a1.om, a2.om} -- zero drilling content;
      * on a warped quad the tying points carry a MOVED frame, so the tied
        rotation block acquires an n-component -- real drilling content, growing
        with the warp, and absent on a planar quad of any shape.

Run:  pytest src/opensg_shell/tests/test_segment_tri_guard.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri3_kernels import (ALU_ISO_DE, ALU_ISO_G, AX, CROSS, Q_GENERIC,  # noqa: E402
                          SHIFT_GENERIC, TIE_MITC3, TRI_GPTS, TRI_NAMES,
                          mitc3_shear_batch, solid_fluct_ops_tri_batch,
                          tri_batch, tri_frame_batch)
from opensg_shell.sg_assembly import (_shear_batch, _surf_frame_batch,  # noqa: E402
                                      _tie_rows_batch, assemble_constraint,
                                      assemble_segment_indep,
                                      build_C_Psi_segment6,
                                      quad_ops_indep_batch, require_quad_mesh)

NDOF6 = 6
GP = 1.0 / np.sqrt(3.0)


# ----------------------------------------------------------------- helpers ---
def _quad_batch(X):
    """One quad in the generic 3-D pose, with its own normal as the e3 reference.

    In:  X (4,3) float corner coordinates in the z = 0 plane.
    Out: Xe (1,4,3), e3g (1,3) in GLOBAL component order."""
    Xe = (np.asarray(X, float) @ Q_GENERIC.T + SHIFT_GENERIC)[None, :, :]
    c = _surf_frame_batch(Xe, np.zeros((1, 3)), 0.0, 0.0, CROSS, AX)[4]
    e3g = np.zeros((1, 3))
    e3g[0, AX], e3g[0, CROSS[0]], e3g[0, CROSS[1]] = c["y1"][0], c["y2"][0], c["y3"][0]
    return Xe, e3g


def _normal(c):
    """Gauss-point normal in the (ax, cross0, cross1) order the operators use."""
    return np.array([c["y1"][0], c["y2"][0], c["y3"][0]])


def _drilling_content(BG, nrm, nnode):
    """max over (row, node) of |c_a . n| / |c_a| for the ROTATION block of BG.

    c_a = dBG/d(om_a) is the row's sensitivity to the nodal rotation vector.
    Zero means the row cannot see the drilling component n.om_a."""
    worst = 0.0
    for row in range(2):
        for a in range(nnode):
            c = BG[0, row, NDOF6 * a + 3: NDOF6 * a + 6]
            nc = float(np.linalg.norm(c))
            if nc > 1e-30:
                worst = max(worst, abs(float(c @ nrm)) / nc)
    return worst


def _one_quad_strip():
    """A single 4-node element and the arguments the segment assembler wants."""
    nodes = np.array([[0.0, 1.0, 0.0], [0.3, 1.0, 0.0],
                      [0.3, 0.7, 0.7], [0.0, 0.7, 0.7]])
    quads = np.array([[0, 1, 2, 3]], int)
    e3s = np.array([[0.0, 0.6, 0.8]])
    return (nodes, quads, np.zeros(1, int), e3s,
            {0: np.asarray(ALU_ISO_DE, float)}, {0: np.asarray(ALU_ISO_G, float)},
            np.zeros(1))


# ================================================================= the guard ==
def test_require_quad_mesh_passes_a_genuine_quad_table_through():
    q = [[0, 1, 2, 3], [1, 4, 5, 2]]
    out = require_quad_mesh(q, "unit test")
    assert out.shape == (2, 4) and out.dtype.kind == "i"
    assert np.array_equal(out, np.asarray(q))


@pytest.mark.parametrize("k,frag", [(3, "3-NODE"), (2, "2-NODE"), (8, "8-NODE")])
def test_require_quad_mesh_names_the_arity(k, frag):
    with pytest.raises(ValueError) as ei:
        require_quad_mesh(np.zeros((5, k), int), "unit test")
    msg = str(ei.value)
    assert frag in msg and "4-node only" in msg
    assert "tri3_in_the_segment_route.md" in msg


def test_require_quad_mesh_rejects_a_mixed_arity_table():
    with pytest.raises(ValueError, match="MIXED-ARITY"):
        require_quad_mesh([[0, 1, 2, 3], [1, 4, 5]], "unit test")


def test_require_quad_mesh_message_says_the_shear_scheme_is_not_the_blocker():
    """The guard must carry the assessment, or the next reader re-runs it."""
    with pytest.raises(ValueError) as ei:
        require_quad_mesh(np.zeros((4, 3), int), "unit test")
    msg = str(ei.value)
    assert "OPERATOR gap" in msg          # the real cause
    assert "MITC3 is sound here" in msg   # the ruled-out cause
    assert "Gamma_e" in msg and "Gamma_l" in msg


def test_segment_assembler_accepts_quads():
    """The guard must be inert on the production path."""
    nodes, quads, sd, e3s, D_by, G_by, k22 = _one_quad_strip()
    out = assemble_segment_indep(nodes, quads, sd, e3s, D_by, G_by, k22,
                                 CROSS, AX)
    Dhh, Dhe, Dee = out[0], out[1], out[2]
    assert Dhh.shape == (24, 24) and Dhe.shape == (24, 4) and Dee.shape == (4, 4)
    assert np.all(np.isfinite(Dhh))
    assert np.allclose(Dhh, Dhh.T, atol=1e-6 * np.abs(Dhh).max())


@pytest.mark.parametrize("fn", ["assemble_segment_indep", "assemble_constraint",
                                "build_C_Psi_segment6"])
def test_segment_entry_points_reject_triangles(fn):
    """Each 4-node-only entry point fails EARLY and by name -- not on a reshape."""
    nodes, quads, sd, e3s, D_by, G_by, k22 = _one_quad_strip()
    tris = quads[:, :3]                              # a genuine 3-node table
    with pytest.raises(ValueError) as ei:
        if fn == "assemble_segment_indep":
            assemble_segment_indep(nodes, tris, sd, e3s, D_by, G_by, k22, CROSS, AX)
        elif fn == "assemble_constraint":
            assemble_constraint(nodes, tris, sd, e3s, k22, CROSS, AX)
        else:
            build_C_Psi_segment6(nodes, tris, CROSS)
    msg = str(ei.value)
    assert fn in msg and "3-NODE" in msg
    assert "reshape" not in msg              # the old bare-numpy failure


def test_extract_rejects_a_mixed_tri_quad_segment_yaml(tmp_path):
    """sg_mesh.extract stores seg_cells as ONE rectangular array; a mixed table
    used to raise numpy's 'inhomogeneous shape' from deep inside the bundle."""
    import yaml as _yaml
    from opensg_shell.sg_mesh import extract

    p = tmp_path / "mixed_seg.yaml"
    _yaml.safe_dump({"nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.5, 0.0]],
                     "elements": [[1, 2, 3, 4], [2, 5, 3]]},
                    open(p, "w"), default_flow_style=None, sort_keys=False)
    with pytest.raises(ValueError) as ei:
        extract(str(p), str(tmp_path / "b.npz"), write_yaml=False, plot=False)
    msg = str(ei.value)
    assert "sg_mesh.extract" in msg and "MIXED-ARITY" in msg
    assert "inhomogeneous" not in msg


def test_extract_rejects_an_all_triangle_segment_yaml(tmp_path):
    """The all-tri case used to survive extraction AND both ring solves before
    dying on 'cannot reshape ... into (ne,24)'.  It must fail at the door."""
    import yaml as _yaml
    from opensg_shell.sg_mesh import extract

    p = tmp_path / "tri_seg.yaml"
    _yaml.safe_dump({"nodes": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                               [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                     "elements": [[1, 2, 3], [1, 3, 4]]},
                    open(p, "w"), default_flow_style=None, sort_keys=False)
    with pytest.raises(ValueError) as ei:
        extract(str(p), str(tmp_path / "b.npz"), write_yaml=False, plot=False)
    assert "sg_mesh.extract" in str(ei.value) and "3-NODE" in str(ei.value)


# ================================================ the decision record: MITC3 ==
def test_tri3_frame_is_constant_so_every_mitc3_tie_shares_the_gauss_frame():
    """The affine map is why the aliasing mechanism cannot fire on a triangle."""
    names, Xe, e3e = tri_batch()
    keys = ("x11", "x21", "x31", "x12", "x22", "x32", "y1", "y2", "y3")
    r, s, _ = TRI_GPTS[0]
    cg = tri_frame_batch(Xe, e3e, r, s, CROSS, AX)[4]
    for (tr, ts) in TIE_MITC3:
        ct = tri_frame_batch(Xe, e3e, tr, ts, CROSS, AX)[4]
        drift = max(float(np.max(np.abs(ct[k] - cg[k]))) for k in keys)
        assert drift == 0.0, "frame drift %.3e at tying point (%g,%g)" % (drift, tr, ts)


@pytest.mark.parametrize("i,name", list(enumerate(TRI_NAMES)))
def test_mitc3_carries_no_drilling_content_on_a_triangle(i, name):
    """max |c_a.n|/|c_a| stays at machine zero under MITC3, exactly as under
    'full' -- the aliasing objection recorded for the quad does NOT transfer."""
    _, Xe, e3e = tri_batch()
    Xe, e3e = Xe[i:i + 1], e3e[i:i + 1]
    r, s, _ = TRI_GPTS[0]
    _, Bg, _, _, Gm = solid_fluct_ops_tri_batch(Xe, e3e, r, s, CROSS, AX)
    nrm = _normal(tri_frame_batch(Xe, e3e, r, s, CROSS, AX)[4])
    tie = [solid_fluct_ops_tri_batch(Xe, e3e, tr, ts, CROSS, AX)[1]
           for (tr, ts) in TIE_MITC3]
    BGt = mitc3_shear_batch(Gm, tie[0], tie[1], tie[2], r, s)
    assert _drilling_content(Bg, nrm, 3) < 1e-14, name
    assert _drilling_content(BGt, nrm, 3) < 1e-14, name


def test_mitc3_really_does_change_the_rotation_interpolation():
    """Guard against the above passing for the trivial reason that MITC3 is a
    no-op: the tied rotation block must differ substantially from the untied one
    while STILL carrying no drilling."""
    _, Xe, e3e = tri_batch()
    r, s, _ = TRI_GPTS[0]
    _, Bg, _, _, Gm = solid_fluct_ops_tri_batch(Xe, e3e, r, s, CROSS, AX)
    tie = [solid_fluct_ops_tri_batch(Xe, e3e, tr, ts, CROSS, AX)[1]
           for (tr, ts) in TIE_MITC3]
    BGt = mitc3_shear_batch(Gm, tie[0], tie[1], tie[2], r, s)
    rot = np.concatenate([np.arange(NDOF6 * a + 3, NDOF6 * a + 6) for a in range(3)])
    d = np.abs(BGt[:, :, rot] - Bg[:, :, rot]).max(axis=(1, 2))
    ref = np.abs(Bg[:, :, rot]).max(axis=(1, 2))
    assert np.all(d / ref > 0.1), "MITC3 rotation block barely moves: %s" % (d / ref)


# ================================ the decision record: the quad objection IS real
QUADS_PLANAR = {
    "parallelogram": [(0., 0., 0.), (1., .2, 0.), (1.3, 1.2, 0.), (.3, 1., 0.)],
    "trapezoid": [(0., 0., 0.), (2., 0., 0.), (1.4, 1., 0.), (.2, 1., 0.)],
}


@pytest.mark.parametrize("name", sorted(QUADS_PLANAR))
@pytest.mark.parametrize("scheme", ["mitc4_g23", "mitc4_wonly", "mitc4_both"])
def test_tying_a_planar_quad_carries_no_drilling_content(name, scheme):
    """A PLANAR quad has one normal everywhere, so tying is clean whatever its
    shape -- the aliasing is not caused by om3 being independent."""
    Xe, e3g = _quad_batch(QUADS_PLANAR[name])
    BGh = quad_ops_indep_batch(Xe, e3g, -GP, -GP, CROSS, AX)[4]
    nrm = _normal(_surf_frame_batch(Xe, e3g, -GP, -GP, CROSS, AX)[4])
    BGt = _shear_batch(-GP, -GP, scheme, BGh, _tie_rows_batch(Xe, e3g, CROSS, AX))
    assert _drilling_content(BGt, nrm, 4) < 1e-12


@pytest.mark.parametrize("lift,floor", [(0.15, 1e-2), (0.50, 5e-2)])
def test_tying_a_WARPED_quad_does_alias_the_drilling(lift, floor):
    """The documented objection, isolated: warp moves the frame between the
    tying points and the Gauss point, and drilling leaks into the shear row.
    'full' and 'mitc4_wonly' (rotation columns kept fully integrated) stay clean,
    which is what makes warp -- not the independent om3 -- the cause."""
    Xe, e3g = _quad_batch([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., lift)])
    BGh = quad_ops_indep_batch(Xe, e3g, -GP, -GP, CROSS, AX)[4]
    nrm = _normal(_surf_frame_batch(Xe, e3g, -GP, -GP, CROSS, AX)[4])
    tie = _tie_rows_batch(Xe, e3g, CROSS, AX)
    assert _drilling_content(BGh, nrm, 4) < 1e-12                       # untied
    assert _drilling_content(_shear_batch(-GP, -GP, "mitc4_wonly", BGh, tie),
                             nrm, 4) < 1e-12
    for scheme in ("mitc4_g23", "mitc4_both"):
        got = _drilling_content(_shear_batch(-GP, -GP, scheme, BGh, tie), nrm, 4)
        assert got > floor, "%s aliased only %.3e at warp %.2f" % (scheme, got, lift)
