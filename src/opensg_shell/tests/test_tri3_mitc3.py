"""Element-level validation of the NATIVE 3-node triangle (tri3) and its MITC3
assumed transverse-shear field.

Covered
  MITC3 identity      the shipped assumed field reproduces an INDEPENDENTLY
                      written Lee & Bathe (2004) formula bit-for-bit, and it is
                      demonstrably NOT edge-averaging / linear interpolation
  rigid body          translation + infinitesimal rotation -> zero in B, Bg, Dr
  patch tests         constant membrane strain; constant membrane strain PLUS
                      constant curvature with identically zero transverse shear
                      (the Kirchhoff state: quadratic deflection, linear rotation);
                      constant transverse shear
  locking mechanism   the displacement-based ('full') shear of that Kirchhoff
                      state is O(h*kappa) and grows with distortion, while MITC3
                      annihilates it to machine zero
  spatial isotropy    the 18x18 stiffness is invariant under cyclic renumbering
                      of the corners and under a rigid rotation of the element
                      in space (in-plane isotropic wall law)
  frame objectivity   WITH the drilling treatment on -- for the tri3 AND for the
                      quad.  This was a strict xfail while the drilling was a
                      PENALTY on a residual row divided by C33 = n.e_3 (the
                      element stiffness then changed by 1.8e-4 relative under a
                      rigid rotation); it passes at ~1e-16 now that omega_3 is
                      pinned by an element-wise LAGRANGE MULTIPLIER on the
                      undivided, frame-covariant row
  zero-energy modes   9 unconstrained (6 rigid + 3 drilling); ONE element-mean
                      multiplier removes exactly one of the three, so a single
                      element keeps 8 -- on a patch, where the multipliers
                      outnumber the nodal drilling modes, exactly the 6 rigid
                      modes survive and they span the rigid subspace
  locking curve       simply-supported and clamped square plate, uniform
                      pressure, t/L = 1e-2 .. 1e-4: 'full' collapses like t^2,
                      MITC3 stays thickness-independent

Run:  pytest src/opensg_shell/tests/test_tri3_mitc3.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tri3_kernels import (ALU_ISO_DE, ALU_ISO_G, AX, CROSS, Q_GENERIC,
                          SHIFT_GENERIC, TRI_GPTS, TRI_NAMES, TIE_MITC3,
                          const_shear_state, dof_rotation, frames,
                          kirchhoff_state, mitc3_shear_batch,
                          patch_constrained_spectrum,
                          plate_center_deflection, quad_Ke, quad_batch,
                          rigid_basis, rigid_basis_n, rigid_state,
                          shear_rows, solid_fluct_ops_tri_batch,
                          square_quad_mesh, square_tri_mesh, tri_Ke,
                          tri_batch, tri_frame_batch)

SCHEMES = ("full", "mitc3")
# machine-precision gate: the error is normalized by |B|_max * |u|_max, the
# natural size of the matrix-vector product, so 1e-13 is ~450 * eps
MACH = 1.0e-13


def _ops(Xe, e3e, r, s):
    return solid_fluct_ops_tri_batch(Xe, e3e, r, s, CROSS, AX)


def _scale(M, u):
    return np.abs(M).max() * np.abs(u).max()


# ============================================================ MITC3 identity =
def test_mitc3_matches_an_independent_lee_bathe_formula():
    """The shipped tying must agree bit-for-bit with a formula written from the
    paper, at interior AND corner parent points."""
    _, Xe, e3e = tri_batch()
    G11, G12, G21, G22 = frames(Xe, e3e)[4]
    g = [_ops(Xe, e3e, tr, ts)[1] for (tr, ts) in TIE_MITC3]

    def cov(x):
        return (G11[:, None] * x[:, 0] + G21[:, None] * x[:, 1],
                G12[:, None] * x[:, 0] + G22[:, None] * x[:, 1])

    rA, sA = cov(g[0])
    rB, sB = cov(g[1])
    rC, sC = cov(g[2])
    c = (rC - rA) - (sC - sB)
    det = (G11 * G22 - G12 * G21)[:, None]
    for (r, s) in [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3),
                   (0.0, 0.0), (1.0, 0.0), (0.3, 0.45)]:
        ert, est = rA + c * s, sB - c * r
        ref = np.stack([(G22[:, None] * ert - G21[:, None] * est) / det,
                        (-G12[:, None] * ert + G11[:, None] * est) / det], 1)
        got = mitc3_shear_batch((G11, G12, G21, G22), g[0], g[1], g[2], r, s)
        assert np.abs(got - ref).max() == 0.0, (r, s)


def test_mitc3_is_not_edge_averaging_or_linear_interpolation():
    """The coupling constant c is what makes MITC3 more than two independent 1-D
    ties; on a distorted element it must move the answer by a large fraction."""
    _, Xe, e3e = tri_batch()
    G11, G12, G21, G22 = Gm = frames(Xe, e3e)[4]
    g = [_ops(Xe, e3e, tr, ts)[1] for (tr, ts) in TIE_MITC3]

    def cov(x):
        return (G11[:, None] * x[:, 0] + G21[:, None] * x[:, 1],
                G12[:, None] * x[:, 0] + G22[:, None] * x[:, 1])

    rA, sA = cov(g[0])
    rB, sB = cov(g[1])
    rC, sC = cov(g[2])
    c = (rC - rA) - (sC - sB)
    det = (G11 * G22 - G12 * G21)[:, None]
    assert np.abs(c).max() > 1.0                      # the correction is live
    r, s = 1 / 6, 1 / 6
    got = mitc3_shear_batch(Gm, g[0], g[1], g[2], r, s)
    # (a) drop c  ->  two uncoupled 1-D ties  (b) plain average of the three ties
    ert, est = rA + 0.0 * s, sB - 0.0 * r
    nc = np.stack([(G22[:, None] * ert - G21[:, None] * est) / det,
                   (-G12[:, None] * ert + G11[:, None] * est) / det], 1)
    avg = (g[0] + g[1] + g[2]) / 3.0
    for other in (nc, avg):
        rel = np.abs(got - other).max() / np.abs(got).max()
        assert rel > 0.1, rel


def test_tri_scheme_mapping():
    from opensg_shell.sg_assembly import tri_scheme_for
    assert tri_scheme_for("full") == "full"
    for k in ("mitc", "mitc3", "mitc4_both", "mitc4_g23", "mitc4_wonly"):
        assert tri_scheme_for(k) == "mitc3"
    with pytest.raises(ValueError):
        tri_scheme_for("nope")


# ================================================================ patch tests =
@pytest.mark.parametrize("scheme", SCHEMES)
def test_rigid_body_motion_is_strain_free(scheme):
    """Translation + infinitesimal rotation -> zero membrane, curvature, shear
    AND zero drilling residual, on every reference shape."""
    _, Xe, e3e = tri_batch()
    u = rigid_state(Xe, [0.5, -1.3, 2.2], [0.013, -0.021, 0.007])
    for (r, s, _w) in TRI_GPTS:
        B, _Bgu, Dr, _dJ, Gm = _ops(Xe, e3e, r, s)
        Bg = shear_rows(Xe, e3e, Gm, r, s, scheme)
        for M in (B, Bg):
            assert np.abs(np.einsum('eib,eb->ei', M, u)).max() <= MACH * _scale(M, u)
        assert np.abs(np.einsum('eb,eb->e', Dr, u)).max() <= MACH * _scale(Dr, u)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constant_membrane_strain_patch(scheme):
    """Pure in-plane linear displacement: rows [e11 e22 2e12] exact, everything
    else zero."""
    _, Xe, e3e = tri_batch()
    tgt = dict(e11=1.3e-3, e22=-0.7e-3, g12=2.1e-3, k11=0.0, k22=0.0, k12x2=0.0)
    u = kirchhoff_state(Xe, e3e, **tgt)
    ref = np.array([tgt["e11"], tgt["e22"], tgt["g12"], 0.0, 0.0, 0.0])
    for (r, s, _w) in TRI_GPTS:
        B, _Bgu, _Dr, _dJ, Gm = _ops(Xe, e3e, r, s)
        Bg = shear_rows(Xe, e3e, Gm, r, s, scheme)
        assert np.abs(np.einsum('eib,eb->ei', B, u) - ref).max() <= MACH * _scale(B, u)
        assert np.abs(np.einsum('eib,eb->ei', Bg, u)).max() <= MACH * _scale(Bg, u)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constant_membrane_plus_constant_curvature_patch(scheme):
    """THE patch test: all six rows of B reproduce the imposed constants to
    machine precision, at every Gauss point, on every distorted shape, under both
    shear schemes, with a rigid body motion superposed."""
    _, Xe, e3e = tri_batch()
    tgt = dict(e11=1.3e-3, e22=-0.7e-3, g12=2.1e-3,
               k11=0.9e-2, k22=-1.7e-2, k12x2=2.6e-2)
    ref = np.array([tgt["e11"], tgt["e22"], tgt["g12"],
                    tgt["k11"], tgt["k22"], tgt["k12x2"]])
    u = kirchhoff_state(Xe, e3e, w_rigid=[0.5, -1.3, 2.2],
                        th_rigid=[0.013, -0.021, 0.007], **tgt)
    for (r, s, _w) in TRI_GPTS:
        B = _ops(Xe, e3e, r, s)[0]
        err = np.abs(np.einsum('eib,eb->ei', B, u) - ref)
        assert err.max() <= MACH * _scale(B, u), (scheme, r, s, err.max())


def test_mitc3_annihilates_the_kirchhoff_shear_and_full_does_not():
    """The Kirchhoff (constant-curvature, zero-shear) state has a QUADRATIC
    deflection, which a linear tri3 cannot interpolate; the displacement-based
    shear of its nodal interpolant is therefore spurious and O(h*kappa).  MITC3
    must tie it to zero -- that is the locking cure, at element level."""
    names, Xe, e3e = tri_batch()
    tgt = dict(e11=1.3e-3, e22=-0.7e-3, g12=2.1e-3,
               k11=0.9e-2, k22=-1.7e-2, k12x2=2.6e-2)
    u = kirchhoff_state(Xe, e3e, **tgt)
    worst = {"full": np.zeros(len(names)), "mitc3": np.zeros(len(names))}
    scale = 0.0
    for scheme in SCHEMES:
        for (r, s, _w) in TRI_GPTS:
            Gm = _ops(Xe, e3e, r, s)[4]
            Bg = shear_rows(Xe, e3e, Gm, r, s, scheme)
            scale = max(scale, _scale(Bg, u))
            worst[scheme] = np.maximum(worst[scheme],
                                       np.abs(np.einsum('eib,eb->ei', Bg, u)).max(1))
    assert worst["full"].min() > 1.0e-3, worst["full"]      # spurious, every shape
    assert worst["mitc3"].max() <= MACH * scale, worst["mitc3"]   # annihilated
    assert worst["full"].max() / max(worst["mitc3"].max(), 1e-300) > 1.0e10


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constant_transverse_shear_patch(scheme):
    """A constant transverse shear state must pass through the tying unchanged
    (the MITC3 consistency condition) and generate no membrane/curvature."""
    _, Xe, e3e = tri_batch()
    g13, g23 = 3.7e-3, -1.9e-3
    u = const_shear_state(Xe, e3e, g13, g23)
    for (r, s, _w) in TRI_GPTS:
        B, _Bgu, _Dr, _dJ, Gm = _ops(Xe, e3e, r, s)
        Bg = shear_rows(Xe, e3e, Gm, r, s, scheme)
        got = np.einsum('eib,eb->ei', Bg, u)
        assert np.abs(got - np.array([g13, g23])).max() <= MACH * _scale(Bg, u)
        assert np.abs(np.einsum('eib,eb->ei', B, u)).max() <= MACH * _scale(B, u)


# ============================================================ spatial isotropy =
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("perm", [(1, 2, 0), (2, 0, 1)])
def test_element_stiffness_invariant_under_cyclic_renumbering(scheme, perm):
    """The element frame is built from corner 1, yet with an in-plane isotropic
    wall law the 18x18 must not know which corner is numbered 1 (Lee & Bathe's
    isotropy requirement)."""
    _, Xe, e3e = tri_batch(tilt=False)
    K0 = tri_Ke(Xe, e3e, ALU_ISO_DE, ALU_ISO_G, scheme, drill="lagrange")
    ix = np.concatenate([np.arange(6 * p, 6 * p + 6) for p in perm] + [[18]])
    Kp = tri_Ke(Xe[:, list(perm), :], e3e, ALU_ISO_DE, ALU_ISO_G, scheme,
                drill="lagrange")
    rel = (np.abs(Kp - K0[:, ix][:, :, ix]).max(axis=(1, 2))
           / np.abs(K0).max(axis=(1, 2)))
    assert rel.max() < 1.0e-12, dict(zip(TRI_NAMES, rel))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_element_stiffness_invariant_under_rigid_rotation(scheme):
    """Rotate the element (and its reference normal) rigidly in space: with the
    DOFs rotated to match, the stiffness must be unchanged.  Drilling treatment
    OFF here; the test below repeats it WITH the multiplier."""
    _, X0, e30 = tri_batch(tilt=False)
    K0 = tri_Ke(X0, e30, ALU_ISO_DE, ALU_ISO_G, scheme)
    Xr = np.einsum('ij,eaj->eai', Q_GENERIC, X0) + SHIFT_GENERIC
    e3r = np.tile(Q_GENERIC @ np.array([0.0, 0.0, 1.0]), (X0.shape[0], 1))
    Kr = tri_Ke(Xr, e3r, ALU_ISO_DE, ALU_ISO_G, scheme)
    T = dof_rotation(Q_GENERIC)
    rel = (np.abs(np.einsum('ia,eab,jb->eij', T, K0, T) - Kr).max(axis=(1, 2))
           / np.abs(K0).max(axis=(1, 2)))
    assert rel.max() < 1.0e-12, dict(zip(TRI_NAMES, rel))


def test_drilling_row_is_frame_covariant():
    """The shipped drilling residual is the MULTIPLIED-THROUGH row
        DR = om.n - 1/2 [ d(w.a2)/dx1 - d(w.a1)/dx2 ] ,
    which is exactly frame-covariant: DR(rotated) == T DR(unrotated).

    It used to be divided by C33 = n.e_3, and THAT is what broke objectivity --
    the second assertion shows the divided row would move by O(1) under the
    same rotation (the historical 1.8e-4 stiffness drift came from it, weighted
    by the penalty)."""
    _, X0, e30 = tri_batch(tilt=False)
    Xr = np.einsum('ij,eaj->eai', Q_GENERIC, X0) + SHIFT_GENERIC
    e3r = np.tile(Q_GENERIC @ np.array([0.0, 0.0, 1.0]), (X0.shape[0], 1))
    r = s = 1.0 / 3.0
    Dr0 = _ops(X0, e30, r, s)[2]
    Drr = _ops(Xr, e3r, r, s)[2]
    c0 = tri_frame_batch(X0, e30, r, s, CROSS, AX)[4]["y3"]
    cr = tri_frame_batch(Xr, e3r, r, s, CROSS, AX)[4]["y3"]
    assert np.abs(np.abs(c0) - 1.0).max() < 1e-14        # flat, axis-aligned
    assert 0.1 < np.abs(cr).max() < 0.99                 # tilted: C33 != 1
    T = dof_rotation(Q_GENERIC)
    rhs = np.einsum('ab,eb->ea', T, Dr0)
    scale = np.abs(rhs).max()
    assert np.abs(Drr - rhs).max() < 1.0e-13 * scale     # as shipped: covariant
    # the OLD divided row, rebuilt here, is not
    assert np.abs(Drr / cr[:, None]
                  - np.einsum('ab,eb->ea', T, Dr0 / c0[:, None])).max() \
        > 0.1 * scale


@pytest.mark.parametrize("elem", ["tri3", "quad"])
def test_element_matrix_invariant_under_rigid_rotation_WITH_drilling(elem):
    """THE frame-objectivity gate, drilling ON.  Was a strict xfail while the
    drilling was a penalty on the 1/C33-divided row (relative drift 1.8e-4);
    with the element-wise Lagrange multiplier on the undivided row the bordered
    element matrix is objective to machine precision, for the triangle and for
    the quad alike."""
    if elem == "tri3":
        names, X0, e30 = tri_batch(tilt=False)
        Ke = lambda X, e: tri_Ke(X, e, ALU_ISO_DE, ALU_ISO_G,   # noqa: E731
                                 "mitc3", drill="lagrange")
        nd = 18
    else:
        names, X0, e30 = quad_batch(tilt=False)
        Ke = lambda X, e: quad_Ke(X, e, ALU_ISO_DE, ALU_ISO_G,  # noqa: E731
                                  "mitc", drill="lagrange")
        nd = 24
    K0 = Ke(X0, e30)
    Xr = np.einsum('ij,eaj->eai', Q_GENERIC, X0) + SHIFT_GENERIC
    e3r = np.einsum('ij,ej->ei', Q_GENERIC, e30)
    Kr = Ke(Xr, e3r)
    T = dof_rotation(Q_GENERIC, ndof=nd, lagrange=True)
    rel = (np.abs(np.einsum('ia,eab,jb->eij', T, K0, T) - Kr).max(axis=(1, 2))
           / np.abs(K0).max(axis=(1, 2)))
    assert rel.max() < 1.0e-12, dict(zip(names, rel))


# ========================================================== zero-energy modes =
@pytest.mark.parametrize("scheme", SCHEMES)
def test_zero_energy_modes_without_drilling(scheme):
    """B and Bg alone leave a 9-dimensional kernel: the 6 rigid modes plus 3
    drilling modes (the nodal om.n, which no strain row sees on a flat element)."""
    _, Xe, e3e = tri_batch(tilt=False)
    for j, name in enumerate(TRI_NAMES):
        K = tri_Ke(Xe[j:j + 1], e3e[j:j + 1], ALU_ISO_DE, ALU_ISO_G,
                   scheme)[0]
        K = 0.5 * (K + K.T)
        ev = np.linalg.eigvalsh(K)
        assert int((np.abs(ev) < 1.0e-12 * ev.max()).sum()) == 9, (name, ev[:12])


@pytest.mark.parametrize("elem,nv", [("tri3", 3), ("quad", 4)])
def test_one_element_mean_multiplier_removes_exactly_one_drilling_mode(elem, nv):
    """COUNTING, on one element.  A lone element carries 3 (tri) or 4 (planar
    quad) nodal drilling modes on top of the 6 rigid ones -- fewer on a WARPED
    quad, whose moving frame already sees three of them.  One element-MEAN
    multiplier is ONE linear condition, so it removes exactly one, whatever the
    shape: constrained = unconstrained - 1.

    That is the inf-sup arithmetic of the element-constant multiplier space, not
    a defect -- see the two patch tests for what it means on a mesh."""
    if elem == "tri3":
        names, Xe, e3e = tri_batch(tilt=False)
        args = (ALU_ISO_DE, ALU_ISO_G, "mitc3")
        K0 = tri_Ke(Xe, e3e, *args)
        Kb = tri_Ke(Xe, e3e, *args, drill="lagrange")
    else:
        names, Xe, e3e = quad_batch(tilt=False)
        args = (ALU_ISO_DE, ALU_ISO_G, "mitc")
        K0 = quad_Ke(Xe, e3e, *args)
        Kb = quad_Ke(Xe, e3e, *args, drill="lagrange")
    m = 6 * nv
    for j, name in enumerate(names):
        ev0 = np.linalg.eigvalsh(0.5 * (K0[j] + K0[j].T))
        n0 = int((np.abs(ev0) < 1.0e-12 * ev0.max()).sum())
        g = Kb[j, :m, m]
        Z = np.linalg.svd(g[None, :], full_matrices=True)[2][1:].T   # null(g)
        Kc = Z.T @ (0.5 * (Kb[j, :m, :m] + Kb[j, :m, :m].T)) @ Z
        ev = np.linalg.eigvalsh(Kc)
        n1 = int((np.abs(ev) < 1.0e-12 * ev.max()).sum())
        assert n0 >= 7 and n1 == n0 - 1, (name, n0, n1, ev[:12] / ev.max())


@pytest.mark.parametrize("mesh", ["tri", "quad"])
@pytest.mark.parametrize("n", [3, 4])
def test_curved_patch_keeps_exactly_the_six_rigid_modes(mesh, n):
    """THE zero-energy gate.  On a CURVED (non-coplanar) patch -- what a 3-D
    shell SG actually is -- the element-wise drilling multiplier leaves EXACTLY
    6 zero modes and they span exactly the rigid-body set: no spurious drilling
    mode survives, no physical mode is killed.  Triangles and quads alike."""
    nodes, conn = (square_tri_mesh if mesh == "tri" else square_quad_mesh)(1.0, n)
    nodes = nodes.copy()
    nodes[:, 2] = 0.35 * (nodes[:, 0]**2 + 0.7 * nodes[:, 1]**2)   # paraboloid
    r = patch_constrained_spectrum(nodes, conn, ALU_ISO_DE, ALU_ISO_G)
    ev, Z = r["ev"], r["Z"]
    assert r["nzero"] == 6, (r["nzero"], ev[:10] / ev.max())
    assert ev[6] / ev.max() > 1.0e-9, ev[:10] / ev.max()      # clean gap
    V = np.linalg.eigh(Z.T @ r["K"] @ Z)[1][:, :6]
    Rq = np.linalg.qr(rigid_basis_n(nodes))[0]
    sv = np.linalg.svd(Rq.T @ (Z @ V), compute_uv=False)
    assert sv.min() > 1.0 - 1.0e-8, sv                        # same 6-D space


@pytest.mark.parametrize("mesh,extra", [("tri", 2), ("quad", None)])
def test_flat_patch_keeps_pure_drilling_modes_and_they_are_harmless(mesh, extra):
    """The one place the element-constant multiplier is deficient: a FLAT patch,
    where every element normal is the same and the nodal drilling om.n is a
    genuine zero-energy field.  The multipliers then act on that field through
    the element-node incidence matrix, and

      * a triangulation leaves its 2 THREE-COLOURING modes (one value per
        colour class, summing to zero on every triangle) -- 2 whatever the
        refinement;
      * a quad mesh has only ~1 element per node, so it leaves V - ne of them
        (the classical Q1/P0 checkerboard family).

    They are HARMLESS to a homogenization: every surviving mode is a pure nodal
    drilling field (zero translation, rotation along the normal), so it carries
    no strain, no energy AND no macro coupling -- D_eff is invariant under it.
    They only make the KKT singular, which is why ring_indep factors its (small)
    saddle point densely.  Asserted here so a future change of multiplier space
    has to face the number."""
    n = 4
    nodes, conn = (square_tri_mesh if mesh == "tri" else square_quad_mesh)(1.0, n)
    r = patch_constrained_spectrum(nodes, conn, ALU_ISO_DE, ALU_ISO_G)
    want = 6 + (extra if extra is not None
                else len(nodes) - r["n_drill"])
    assert r["nzero"] == want, (r["nzero"], want, r["ev"][:18] / r["ev"].max())
    Z = r["Z"]
    V = np.linalg.eigh(Z.T @ r["K"] @ Z)[1][:, :r["nzero"]]
    M = Z @ V                                        # zero modes, full dofs
    # every zero mode lies in span(rigid modes, nodal drilling om3)
    D = np.zeros((r["ndof"], len(nodes)))
    D[5::6, :] = np.eye(len(nodes))                  # flat patch: n = e3
    B = np.linalg.qr(np.hstack([rigid_basis_n(nodes), D]))[0]
    leak = np.abs(M - B @ (B.T @ M)).max()
    assert leak < 1.0e-9, leak


# =============================================================== locking curve =
def test_full_scheme_locks_and_mitc3_does_not():
    """THE headline.  Simply-supported square plate, uniform pressure, fixed
    16x16x2 triangle mesh, thickness swept over two decades.  Normalized by the
    Kirchhoff analytic centre deflection, the displacement-based element must
    collapse like t^2 while MITC3 stays put."""
    n = 16
    got = {}
    for scheme in SCHEMES:
        got[scheme] = [plate_center_deflection(1.0, tl, n, scheme, "ss")
                       for tl in (1e-2, 1e-3, 1e-4)]
        got[scheme] = [w / wk for (w, wk) in got[scheme]]
    f, m = got["full"], got["mitc3"]
    assert f[0] < 0.10 and f[1] < 0.005 and f[2] < 1e-3, f      # already locked
    assert f[2] / f[0] < 0.01, f                                # collapses ~ t^2
    assert min(m) > 0.97 and max(m) < 1.03, m                   # accurate
    assert (max(m) - min(m)) < 0.01, m                          # t-independent
    assert m[2] / f[2] > 1.0e3, (m, f)                          # cure is decisive


@pytest.mark.parametrize("bc,lo", [("ss", 0.97), ("clamped", 0.92)])
def test_mitc3_is_thickness_independent(bc, lo):
    """MITC3 accuracy must not degrade as t/L -> 1e-4, for both supports."""
    n = 16
    m = [plate_center_deflection(1.0, tl, n, "mitc3", bc)
         for tl in (1e-2, 1e-3, 1e-4)]
    m = [w / wk for (w, wk) in m]
    assert min(m) > lo and max(m) < 1.03, (bc, m)
    assert (max(m) - min(m)) < 0.05, (bc, m)


def test_mitc3_converges_under_refinement():
    """Mesh refinement at the thinnest plate must approach the Kirchhoff value
    monotonically -- the 'full' element cannot (it locks harder than it converges)."""
    tl = 1.0e-4
    norm = lambda sch, n: (lambda p: p[0] / p[1])(          # noqa: E731
        plate_center_deflection(1.0, tl, n, sch, "ss"))
    m = [norm("mitc3", n) for n in (4, 8, 16)]
    assert m[0] < m[1] < m[2] < 1.0, m
    assert m[2] > 0.97, m
    f = [norm("full", n) for n in (4, 8, 16)]
    assert max(f) < 1.0e-3, f


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
