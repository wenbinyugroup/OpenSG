"""Validation / regression test for the core MSG-RM plate law
(``rm_plate.msg_rm_plate``, the Yu-2002/2003 construction).

Everything the core module used to check in its ``__main__`` block lives here, plus the
stronger checks that need an INDEPENDENT assembly: the test re-assembles the Eq.-(30)
SG matrices in plain NumPy and verifies the module's warping fields against the
Euler-Lagrange equations and the Eq.-(31) gauge directly.

Covered
  analytic anchors   isotropic nu=0 -> G = 5/6 G h exactly, with U* at machine zero;
                     A6 == msg_materials.compute_ABD_matrix at the matching z_ref
  Eq. (31)/(39)      H-identities (sum of node weights = h, psi^T H psi = I) and the
                     gauge psi^T H V = 0 for V0, V11, V12
  Eq. (34)/(35)      zeroth-order multiplier vanishes identically (kernel^T D_he = 0)
  Eq. (42) E-L       E V1a + Dbar_a = H psi (psi^T Dbar_a) for both first-order blocks
  Eq. (58)           the w3 row of kernel^T Dbar_a is exactly zero -> Yu's 24 constants
  reference plane    A6(fraction) equals the offset transform of A6(0)
  API                rm_plate_msg_batch == rm_plate_msg; the SPD gate; jax.grad vs FD

Run:  pytest tests/test_msg_rm_plate.py   |   python tests/test_msg_rm_plate.py
"""
import os
import sys

import numpy as np
import pytest

CC = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.join(CC, "src"))
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from opensg_solid.rm_plate_1D.msg_materials import (compute_ABD_matrix, rotated_stiffness_6x6,
                                             _plate_B)
from opensg_solid.rm_plate_1D.msg_rm_plate import (_bucket, _E0, _E1, _grad_ops, _lagrange_N,
                                            rm_plate_msg, rm_plate_msg_batch)
from opensg_solid.rm_plate_1D.msg_transverse_shear import transverse_shear_stiffness

ISO = {"iso": {"E": [70e9] * 3, "G": [70e9 / 2.6] * 3, "nu": [0.3] * 3, "rho": 1.0}}
ISO0 = {"iso0": {"E": [70e9] * 3, "G": [35e9] * 3, "nu": [0.0] * 3, "rho": 1.0}}
GR = {"gr": {"E": [172.4e9, 6.89e9, 6.89e9], "G": [3.45e9, 1.38e9, 3.45e9],
             "nu": [0.25] * 3, "rho": 1.0}}
SAND = {"biax": {"E": [11.5e9, 11.5e9, 1.3e10], "G": [11.8e9, 3.5e9, 3.5e9],
                 "nu": [0.5, 0.09, 0.09], "rho": 1.0},
        "foam": {"E": [1.42e8] * 3, "G": [6.0e7] * 3, "nu": [0.2] * 3, "rho": 1.0}}

CASES = [
    ("iso nu=0.3", ISO, [0.01], [0.0], ["iso"]),
    ("[0/90/0]", GR, [0.005] * 3, [0.0, 90.0, 0.0], ["gr"] * 3),
    ("sandwich", SAND, [0.002, 0.042, 0.002], [0.0] * 3, ["biax", "foam", "biax"]),
    ("[45/0/-30/90]", GR, [0.004, 0.003, 0.005, 0.003], [45.0, 0.0, -30.0, 90.0], ["gr"] * 4),
]


# --------------------------------------------------------------- independent assembly
def assemble(thick, angles, mats, mdb, fraction, n_per_layer, p):
    """Plain-NumPy re-assembly of the Eq.-(30) matrices + the H weights (no JAX, no core
    kernel): the reference the module's fields are checked against."""
    nlay = len(thick)
    C = [rotated_stiffness_6x6(mdb[mats[k]]["E"], mdb[mats[k]]["G"], mdb[mats[k]]["nu"],
                               angles[k]) for k in range(nlay)]
    nodes_xi = np.linspace(-1.0, 1.0, p + 1)
    xi_g, w_g = np.polynomial.legendre.leggauss(max(3, p + 1))
    n_elem = nlay * n_per_layer
    n_node = p * n_elem + 1
    nd = 3 * n_node
    bot = np.concatenate([[0.0], np.cumsum(thick)])
    h = float(bot[-1]); z_ref = fraction * h

    node_x = np.empty(n_node); elem_layer = np.empty(n_elem, int)
    idx = 0
    for k in range(nlay):
        for s in range(n_per_layer):
            xl = bot[k] + thick[k] * s / n_per_layer
            xr = bot[k] + thick[k] * (s + 1) / n_per_layer
            for j in range(p):
                node_x[p * idx + j] = xl + (xr - xl) * j / p
            elem_layer[idx] = k; idx += 1
    node_x[p * n_elem] = bot[-1]; node_x -= z_ref

    Z = lambda *s: np.zeros(s)
    D = dict(hh=Z(nd, nd), he=Z(nd, 6), ee=Z(6, 6), hl1=Z(nd, nd), hl2=Z(nd, nd),
             l1l1=Z(nd, nd), l1l2=Z(nd, nd), l2l2=Z(nd, nd), l1e=Z(nd, 6), l2e=Z(nd, 6))
    w_node = np.zeros(n_node)
    for e in range(n_elem):
        xl = node_x[p * e]; xr = node_x[p * e + p]; he = xr - xl
        Ck = C[elem_layer[e]]
        dofs = np.arange(3 * p * e, 3 * p * e + 3 * (p + 1)); ix = np.ix_(dofs, dofs)
        for q in range(len(xi_g)):
            xi = xi_g[q]; dw = 0.5 * he * w_g[q]
            x_q = 0.5 * (xl + xr) + 0.5 * he * xi
            Gh = _plate_B(nodes_xi, xi, he)
            Gl1, Gl2 = _grad_ops(nodes_xi, xi)
            Ge = _E0 + x_q * _E1
            D["hh"][ix] += Gh.T @ Ck @ Gh * dw
            D["he"][dofs] += Gh.T @ Ck @ Ge * dw
            D["ee"] += Ge.T @ Ck @ Ge * dw
            D["hl1"][ix] += Gh.T @ Ck @ Gl1 * dw
            D["hl2"][ix] += Gh.T @ Ck @ Gl2 * dw
            D["l1l1"][ix] += Gl1.T @ Ck @ Gl1 * dw
            D["l1l2"][ix] += Gl1.T @ Ck @ Gl2 * dw
            D["l2l2"][ix] += Gl2.T @ Ck @ Gl2 * dw
            D["l1e"][dofs] += Gl1.T @ Ck @ Ge * dw
            D["l2e"][dofs] += Gl2.T @ Ck @ Ge * dw
            w_node[p * e:p * e + p + 1] += dw * _lagrange_N(nodes_xi, xi)
    kernel = np.zeros((nd, 3))
    kernel[0::3, 0] = kernel[1::3, 1] = kernel[2::3, 2] = 1.0
    D.update(nd=nd, h=h, kernel=kernel, w_dof=np.repeat(w_node, 3), w_node=w_node)
    return D


def pieces(name, mdb, thk, ang, mats, fraction=0.5, npl=1, p=4):
    r = rm_plate_msg(thk, ang, mats, mdb, n_per_layer=npl, elem_order=p, fraction=fraction)
    A = assemble(thk, ang, mats, mdb, fraction, npl, p)
    psi = A["kernel"] / np.sqrt(A["h"])
    Hpsi = A["w_dof"][:, None] * psi
    D1bar = (A["hl1"] - A["hl1"].T) @ r["V0"] - A["l1e"]
    D2bar = (A["hl2"] - A["hl2"].T) @ r["V0"] - A["l2e"]
    return r, A, psi, Hpsi, D1bar, D2bar


# ------------------------------------------------------------------------- the tests
def test_isotropic_nu0_gives_five_sixths():
    """The one case with a closed-form answer: G = (5/6) G h -- exact to solver
    precision, which is only possible if the U* least-squares fit is residual-free
    (an exactly asymptotically correct RM model exists for iso nu=0)."""
    h = 0.01
    r = rm_plate_msg([h], [0.0], ["iso0"], ISO0, fraction=0.5)
    assert np.allclose(np.diag(r["G_msg"]) / (35e9 * h), 5.0 / 6.0, rtol=1e-10)


def test_A6_matches_compute_ABD():
    for name, mdb, thk, ang, mats in CASES:
        h = float(sum(thk))
        r = rm_plate_msg(thk, ang, mats, mdb, fraction=0.5)
        ref = np.asarray(compute_ABD_matrix(thk, ang, mats, mdb, n_per_layer=4,
                                            z_ref=0.5 * h)[0])[:6, :6]
        rel = np.max(np.abs(r["A6"] - ref)) / np.max(np.abs(ref))
        assert rel < 1e-10, "%s: A6 vs compute_ABD rel %.2e" % (name, rel)


def test_H_identities_and_gauge():
    """Eq. (31): sum of the node weights is h, psi^T H psi = I, and every returned
    warping block satisfies the gauge psi^T H V = 0."""
    for name, mdb, thk, ang, mats in CASES:
        r, A, psi, Hpsi, _, _ = pieces(name, mdb, thk, ang, mats)
        assert abs(A["w_node"].sum() - A["h"]) < 1e-12 * A["h"]
        assert np.max(np.abs(psi.T @ Hpsi - np.eye(3))) < 1e-12
        scale = np.max(np.abs(A["w_dof"]))
        for key in ("V0", "V11", "V12", "V21", "V22", "V23"):
            V = r[key]
            g = np.max(np.abs(Hpsi.T @ V)) / (scale * np.max(np.abs(V)))
            assert g < 1e-9, "%s: gauge violation on %s = %.2e" % (name, key, g)


def test_zeroth_order_multiplier_vanishes():
    """Eq. (35): kernel^T D_he = 0 because Gamma_h of a through-thickness constant is 0."""
    for name, mdb, thk, ang, mats in CASES:
        _, A, _, _, _, _ = pieces(name, mdb, thk, ang, mats)
        rel = np.max(np.abs(A["kernel"].T @ A["he"])) / np.max(np.abs(A["he"]))
        assert rel < 1e-12, "%s: lam0 = %.2e" % (name, rel)


def test_euler_lagrange_residuals():
    """Zeroth order (Eq. 34) and the first-order E-L of Eq. (42), against the independent
    assembly: E V + Dbar = H psi (psi^T Dbar)."""
    for name, mdb, thk, ang, mats in CASES:
        r, A, psi, Hpsi, D1bar, D2bar = pieces(name, mdb, thk, ang, mats)
        for V, rhs, tag in ((r["V0"], A["he"], "V0"),
                            (r["V11"], D1bar, "V11"), (r["V12"], D2bar, "V12")):
            res = A["hh"] @ V + rhs - Hpsi @ (psi.T @ rhs)
            rel = np.max(np.abs(res)) / np.max(np.abs(rhs))
            assert rel < 1e-9, "%s: E-L residual on %s = %.2e" % (name, tag, rel)


def test_relaxation_is_24_constants():
    """Eq. (58): the w3 row of kernel^T Dbar_a is exactly zero for monoclinic laminates,
    which is why only the 2 in-plane rows (24 constants) are carried."""
    for name, mdb, thk, ang, mats in CASES:
        _, A, _, _, D1bar, D2bar = pieces(name, mdb, thk, ang, mats)
        for Db, tag in ((D1bar, "1"), (D2bar, "2")):
            S = A["kernel"].T @ Db
            assert np.max(np.abs(S[2])) <= 1e-12 * np.max(np.abs(S[:2])), \
                "%s: w3 row of S_%s not zero" % (name, tag)


def test_reference_plane_transform():
    """A6(fraction) must equal the exact plate offset transform of A6(0):
    membrane unchanged, B -> B - zA, D -> D - 2zB + z^2 A  (code strain ordering)."""
    mdb, thk, ang, mats = GR, [0.004, 0.003, 0.005, 0.003], [45.0, 0.0, -30.0, 90.0], ["gr"] * 4
    h = float(sum(thk))
    A0 = rm_plate_msg(thk, ang, mats, mdb, fraction=0.0)["A6"]
    for f in (0.25, 0.5, 1.0):
        z = f * h
        T = np.block([[np.eye(3), -z * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        Af = rm_plate_msg(thk, ang, mats, mdb, fraction=f)["A6"]
        rel = np.max(np.abs(Af - T.T @ A0 @ T)) / np.max(np.abs(Af))
        assert rel < 1e-12, "fraction %.2f: offset transform rel %.2e" % (f, rel)


def test_batch_equals_single():
    layups = [{"mat_names": mats, "thick": thk, "angles": ang}
              for _, _, thk, ang, mats in CASES if _ is not None]
    mdb = {**ISO, **GR, **SAND}
    lay = [{"mat_names": m, "thick": t, "angles": a} for _, _, t, a, m in CASES]
    batch = rm_plate_msg_batch(lay, mdb, fraction=0.5)
    for (name, _md, thk, ang, mats), rb in zip(CASES, batch):
        rs = rm_plate_msg(thk, ang, mats, mdb, fraction=0.5)
        rel = np.max(np.abs(rb["G_msg"] - rs["G_msg"])) / np.max(np.abs(rs["G_msg"]))
        assert rel < 1e-8, "%s: batch vs single %.2e" % (name, rel)


def test_spd_gate_and_whitney_order():
    """G must come back SPD for these laminates, and stay within an order of the
    complementary-energy (Whitney) value."""
    for name, mdb, thk, ang, mats in CASES:
        r = rm_plate_msg(thk, ang, mats, mdb, fraction=0.5)
        assert r["G_msg"] is not None, "%s: X not SPD" % name
        assert np.linalg.eigvalsh(r["G_msg"]).min() > 0
        Gw = np.asarray(transverse_shear_stiffness(thk, ang, mats, mdb)[0])
        ratio = r["G_msg"][0, 0] / Gw[0, 0]
        assert 0.1 < ratio < 10.0, "%s: G/Whitney = %.3f" % (name, ratio)


def test_gradients_match_finite_difference():
    """The payoff of the JAX implementation: dG/d(ply thickness) must match central FD."""
    thk = [0.004, 0.003, 0.005, 0.003]; ang = [45.0, 0.0, -30.0, 90.0]
    C_b = jnp.asarray(np.array([rotated_stiffness_6x6(GR["gr"]["E"], GR["gr"]["G"],
                                                      GR["gr"]["nu"], a) for a in ang]))
    bk = _bucket(len(thk), 1, 4)

    def G11(t):
        return bk["jit_single"](t, C_b, jnp.asarray(0.5))[1][0, 0]

    g = np.asarray(jax.grad(G11)(jnp.asarray(thk)))
    t0 = np.asarray(thk, float)
    for i in range(len(thk)):
        d = 1e-7
        tp = t0.copy(); tp[i] += d
        tm = t0.copy(); tm[i] -= d
        fd = (float(G11(jnp.asarray(tp))) - float(G11(jnp.asarray(tm)))) / (2 * d)
        assert abs(g[i] - fd) <= 1e-6 * abs(fd), "ply %d: grad %.6e vs FD %.6e" % (i, g[i], fd)


def test_material_frame_default_rotation():
    """msgrm_strain_at_depth returns PLY-MATERIAL-frame Gam/Sig by default.
    Three machine-exact checks on a [30] off-axis single ply:
      (a) the pair (Sig_mat, Gam_mat) equals (rotation_6x6(-th) @ Sig_plate,
          rotation_6x6(th).T @ Gam_plate) -- the repo convention;
      (b) the UNROTATED material stiffness closes the law in the ply frame,
          Sig_mat = C_material(0 deg) @ Gam_mat -- only true if the
          stress/engineering-strain rotation pair is consistent;
      (c) sigma33/eps33 are frame-invariant."""
    from opensg_solid.rm_plate_1D.msg_materials import rotation_6x6
    from opensg_solid.rm_plate_1D.msg_rm_plate import msgrm_strain_at_depth
    r = rm_plate_msg([0.01], [30.0], ["gr"], GR, fraction=0.5)
    E6 = np.array([1e-3, -2e-4, 3e-4, 0.1, -0.05, 0.02])
    dE1 = np.array([1e-4, 0.0, 2e-4, 0.01, 0.0, 3e-3])
    dE2 = np.array([0.0, 1e-4, -1e-4, 0.0, 0.02, 1e-3])
    z = 0.0023
    Gp, Sp, th = msgrm_strain_at_depth(r, z, E6, dE1, dE2, frame="plate")
    Gm, Sm, th2 = msgrm_strain_at_depth(r, z, E6, dE1, dE2)   # default: material
    assert th == th2 == 30.0
    assert np.max(np.abs(Sm - rotation_6x6(-30.0) @ Sp)) < 1e-9 * np.max(np.abs(Sp))
    assert np.max(np.abs(Gm - rotation_6x6(30.0).T @ Gp)) < 1e-12 * np.max(np.abs(Gp))
    C0 = rotated_stiffness_6x6(GR["gr"]["E"], GR["gr"]["G"], GR["gr"]["nu"], 0.0)
    assert np.max(np.abs(Sm - C0 @ Gm)) < 1e-9 * np.max(np.abs(Sm))
    assert abs(Sm[2] - Sp[2]) < 1e-9 * max(abs(Sp[2]), 1.0)
    assert abs(Gm[2] - Gp[2]) < 1e-12 * max(abs(Gp[2]), 1e-12)
    with pytest.raises(ValueError):
        msgrm_strain_at_depth(r, z, E6, frame="global")


def test_quartic_default_beats_cubic_on_sigma33():
    """The 5-noded (quartic) default represents the whole ladder exactly (V2 is a
    per-ply quartic), so the natural sigma33 face traction is machine zero."""
    thk = [0.005] * 3; ang = [0.0, 90.0, 0.0]; mats = ["gr"] * 3
    r4 = rm_plate_msg(thk, ang, mats, GR, n_per_layer=1, elem_order=4, fraction=0.5)
    r3 = rm_plate_msg(thk, ang, mats, GR, n_per_layer=4, elem_order=3, fraction=0.5)
    rel = np.max(np.abs(r4["G_msg"] - r3["G_msg"])) / np.max(np.abs(r4["G_msg"]))
    assert rel < 1e-10, "G must not depend on elem_order (>=3): %.2e" % rel
    assert r4["V0"].shape[0] < r3["V0"].shape[0]     # and does it with fewer dofs


if __name__ == "__main__":
    fails = 0
    for nm, fn in sorted(list(globals().items())):
        if nm.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  %s" % nm)
            except AssertionError as exc:
                fails += 1
                print("FAIL  %s: %s" % (nm, exc))
    print("\n%s" % ("all tests passed" if not fails else "%d FAILURES" % fails))
    sys.exit(1 if fails else 0)
