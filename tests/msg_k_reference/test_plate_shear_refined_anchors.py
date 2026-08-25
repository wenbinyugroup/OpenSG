"""Analytical CORRECTNESS anchors for the shear-refined plate machinery.

test_plate_ladder_general proves cross-dimension PARITY (1-D, 2-D and
3-D SGs agree); this file pins ABSOLUTE correctness against closed
forms.  The interval engine supports p1..p4 elements (sg_mesh.
_cell_basis, nodes/elem 2..5); with p >= 3 the exact homogeneous-plate
warpings (cubic in x3, quartic for the load ladder) lie IN the FE
space, so the Galerkin solution IS the exact solution and every anchor
lands at machine precision.  p1 ladders converge to the same numbers
at O(h^2) (iso nu=0: k = 1.0 at nz=2 -> 0.8340 at nz=32 -> 5/6).

Measured on the server (2026-08-25, opensg_2_0 + PYTHONPATH src):
  iso nu=0, p3 nz=2:     k = G11/(G h) = 5/6 to 2.7e-16 rel
  iso nu=0.3 (p3/p4):    k = 0.88053740014523, mesh/order-independent
                         to 4e-14 (nu-dependent, NOT 5/6)
  hex8 vs p1 ladder:     G parity 3.0e-14 rel at nz=32
  qt6-only sigma33:      exactly LINEAR -q(1+zeta)/2, rms 1.5e-15 --
      the uniform-q load column alone carries only the pressure
      diffusion; the equilibrium CUBIC -q(2+3z-z^3)/4 needs the
      consistent second-derivative state dE11[3] = +q/D11 (plate
      equilibrium D11 k11,11 = q, POSITIVE sign), then rms vs cubic
      = 1.7e-15 (nu=0, p4) / 4.2e-15 (nu=0.3, p4)
  sigma13 (dE1[3]=Q1/D0): parabola rms 1.2e-15, RAW amplitude
      integral sigma13 dz = Q1 to 0.0 -- 1.1e-15 (NO Q-consistency
      rescale in the call: the guard the AR5 benchmark hangs on)
  V2L inertness:         bit-level 0.0 for qt6 = [q,0,0,0,0,0]
"""
import numpy as np
import pytest

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields

E0, H = 1000.0, 1.0


def sg_1d_p(nz, nu, p):
    """1-D interval ladder of order p.

    In:  nz elements over [-H/2, H/2]; nu Poisson; p basis order 1..4
         (gmsh/basix line node order: the two END nodes first, then
         the interior nodes equispaced -- sg_mesh._cell_basis).
    Out: SG dict for plate_homo_2d."""
    ze = np.linspace(-H / 2, H / 2, nz + 1)
    zs = list(ze)
    cells = []
    for k in range(nz):
        c = [k, k + 1]
        for j in range(1, p):
            zs.append(ze[k] + (ze[k + 1] - ze[k]) * j / p)
            c.append(len(zs) - 1)
        cells.append(c)
    nodes = np.zeros((len(zs), 3))
    nodes[:, 0] = zs
    return {"dim": 1, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(nz, int),
            "materials": {1: {"type": 0, "E": E0, "nu": nu}},
            "scale": 1.0}


def sg_3d_hex(nz, nu, nx=2, ny=2, w=0.25):
    """Trilinear hex slab of the same homogeneous plate.

    In:  nz layers over [-H/2, H/2]; nu; nx x ny in-plane fill of a
         w x w cell.
    Out: SG dict for plate_homo_2d."""
    z = np.linspace(-H / 2, H / 2, nz + 1)
    xs = np.linspace(0, w, nx + 1)
    ys = np.linspace(0, w, ny + 1)
    nid = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i  # noqa: E731
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[nid(i, j, k)] = (xs[i], ys[j], z[k])
    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cells.append([nid(i, j, k), nid(i + 1, j, k),
                              nid(i + 1, j + 1, k), nid(i, j + 1, k),
                              nid(i, j, k + 1), nid(i + 1, j, k + 1),
                              nid(i + 1, j + 1, k + 1),
                              nid(i, j + 1, k + 1)])
    return {"dim": 3, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int),
            "materials": {1: {"type": 0, "E": E0, "nu": nu}},
            "scale": 1.0}


def gauss_profile(r, col, **ff):
    """In:  r homogenization dict; col SwiftComp stress column
         (xx yy zz yz xz xy); ff dehom_fields keyword drives.
    Out: (zq, s) -- flattened Gauss x3 coordinates and stress."""
    _, Sig, _ = dehom_fields(r, np.zeros(6), **ff)
    xe = np.asarray(r["x_end"])
    zq = np.einsum("qn,en->eq", np.asarray(r["phi_qn"]),
                   xe[:, :, r["n_sg"] - 1])
    return zq.ravel(), np.asarray(Sig)[:, :, col].ravel()


def gauss_dv(r):
    """In:  r homogenization dict (single-batch SG).
    Out: (E, Q) Gauss measures detJ * W -- integral f dV =
         (f * gauss_dv).sum(); divide by r['omega'] for the
         per-unit-in-plane-area thickness integral."""
    J = np.einsum("end,qnp->eqdp", np.asarray(r["x_end"]),
                  np.asarray(r["dphi_dxi_qnp"]))
    detJ = (np.abs(np.linalg.det(J)) if r["n_sg"] > 1
            else np.abs(J[..., 0, 0]))
    return detJ * np.asarray(r["W_q"])[None, :]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    import os
    os.chdir(tmp_path_factory.mktemp("anchors"))
    return {"p3_nu0": plate_homo_2d(sg_1d_p(2, 0.0, 3), refined=1),
            "p3_nu03": plate_homo_2d(sg_1d_p(8, 0.3, 3), refined=1),
            "p1_nu03": plate_homo_2d(sg_1d_p(32, 0.3, 1), refined=1),
            "hex_nu03": plate_homo_2d(sg_3d_hex(32, 0.3), refined=1),
            "p4_nu0": plate_homo_2d(sg_1d_p(4, 0.0, 4), refined=1),
            "p4_nu03": plate_homo_2d(sg_1d_p(4, 0.3, 4), refined=1)}


def test_g_iso_nu0_exact(runs):
    """Single-layer isotropic nu=0 plate, 1-D SG (p3, exact-in-space
    warping): G11 = G22 = (5/6) G h EXACTLY -- the textbook shear
    correction, no LS/energy fudge (measured 2.7e-16 rel)."""
    G = np.asarray(runs["p3_nu0"]["G_msg"])
    Gh = (E0 / 2.0) * H                      # iso nu=0: G = E/2
    ref = (5.0 / 6.0) * Gh
    assert abs(G[0, 0] - ref) <= 1e-10 * ref
    assert abs(G[1, 1] - ref) <= 1e-10 * ref
    assert abs(G[0, 1]) <= 1e-10 * ref
    assert abs(G[1, 0]) <= 1e-10 * ref


def test_g_iso_nu03_converged(runs):
    """Isotropic nu=0.3: the energy-consistent shear correction k =
    G11/(G h) is NU-DEPENDENT, not 5/6 -- that is the point.  The p3
    nz=8 ladder is converged (measured k = 0.88053740014523, identical
    to p4 and to nz=4/16 within 4e-14; the p1 ladder walks to the same
    number at O(h^2): 0.8899 at nz=8, 0.88057 at nz=128)."""
    G = np.asarray(runs["p3_nu03"]["G_msg"])
    k = G[0, 0] / ((E0 / 2.6) * H)           # G = E/(2(1+nu))
    assert 0.878 <= k <= 0.884
    assert abs(G[0, 0] - G[1, 1]) <= 1e-10 * G[0, 0]


def test_g_parity_3d(runs):
    """The nu=0.3 plate meshed as a trilinear hex slab must reproduce
    the p1 interval ladder's G at the SAME nz=32 discretization --
    absolute anchor of the 3-D generalization (measured 3.0e-14
    rel)."""
    G1 = np.asarray(runs["p1_nu03"]["G_msg"])
    G3 = np.asarray(runs["hex_nu03"]["G_msg"])
    assert np.abs(G3 - G1).max() <= 1e-8 * np.abs(G1).max()


def test_sigma33_pressure_cubic(runs):
    """Unit uniform top pressure: the qt6 load column ALONE carries
    exactly the linear diffusion profile -q(1+zeta)/2 (zeta = 2 x3/h,
    loaded face at zeta=+1); adding the equilibrium-consistent
    second-derivative state dE11[3] = +q/D11 (D11 k11,11 = q) turns it
    into the elasticity CUBIC -q(2 + 3 zeta - zeta^3)/4 -- face values
    -q (top) and 0 (bottom).  p4 elements hold the quartic warping
    exactly: measured rms 1.5e-15 (linear), 1.7e-15 (cubic, nu=0),
    4.2e-15 (cubic, nu=0.3); tolerances pinned at ~3x measured.  The
    zeta(1 - zeta^2) interior correction is PURE second-order-chain
    content -- a factor error in the V2/V2L machinery moves it 2x and
    fails this test outright."""
    q = 1.0
    qt6 = np.array([q, 0, 0, 0, 0, 0])
    r = runs["p4_nu0"]
    z, s_lin = gauss_profile(r, 2, qt6=qt6)
    zeta = 2.0 * z / H
    lin = -q * (1.0 + zeta) / 2.0
    assert float(np.sqrt(np.mean((s_lin - lin) ** 2))) <= 5e-15

    D0 = E0 * H ** 3 / 12.0                  # D11 at nu=0
    z, s = gauss_profile(r, 2, qt6=qt6,
                         dE11=np.array([0, 0, 0, q / D0, 0, 0]))
    zeta = 2.0 * z / H
    cub = -q * (2.0 + 3.0 * zeta - zeta ** 3) / 4.0
    assert float(np.sqrt(np.mean((s - cub) ** 2))) <= 6e-15
    p = np.polyfit(z, s, 3)                  # exact cubic through the
    assert abs(np.polyval(p, H / 2) + q) <= 1e-13   # Gauss cloud
    assert abs(np.polyval(p, -H / 2)) <= 1e-13

    D11 = E0 * H ** 3 / (12.0 * (1.0 - 0.3 ** 2))
    z, s = gauss_profile(runs["p4_nu03"], 2, qt6=qt6,
                         dE11=np.array([0, 0, 0, q / D11, 0, 0]))
    zeta = 2.0 * z / H
    cub = -q * (2.0 + 3.0 * zeta - zeta ** 3) / 4.0
    assert float(np.sqrt(np.mean((s - cub) ** 2))) <= 1.5e-14


def test_sigma13_parabolic(runs):
    """Pure Q1 drive (nu=0 so dE1 = [0,0,0, Q1/D0, 0,0] with no
    Poisson coupling): sigma13(x3) = 1.5 Q1/h (1 - zeta^2), and the
    RAW thickness integral of sigma13 must equal Q1 -- NO Q-consistency
    rescale is requested in the call, so this pins the Eq. 63 chain's
    absolute amplitude (the guard the AR5 junction question hangs on).
    Measured: rms 1.2e-15, max 2.8e-15, |I - Q1| = 0.0 (p3 nz=2);
    sigma23 stays silent at 1e-32."""
    Q1 = 1.0
    D0 = E0 * H ** 3 / 12.0
    r = runs["p3_nu0"]
    _, Sig, _ = dehom_fields(r, np.zeros(6),
                             dE1=np.array([0, 0, 0, Q1 / D0, 0, 0]))
    xe = np.asarray(r["x_end"])
    zq = np.einsum("qn,en->eq", np.asarray(r["phi_qn"]), xe[:, :, 0])
    s13 = np.asarray(Sig)[:, :, 4]           # xz slot, SwiftComp order
    ana = 1.5 * Q1 / H * (1.0 - (2.0 * zq / H) ** 2)
    assert float(np.sqrt(np.mean((s13 - ana) ** 2))) <= 5e-15
    assert float(np.abs(s13 - ana).max()) <= 1e-14
    I = float((s13 * gauss_dv(r)).sum() / r["omega"])
    assert abs(I - Q1) <= 1e-13              # the amplitude guard
    assert float(np.abs(np.asarray(Sig)[:, :, 3]).max()) <= 1e-12


def test_v2l_inert_uniform_q(runs):
    """qt6 = [q,0,0,0,0,0]: the V2L quintet multiplies the five zero
    q-derivatives, so zeroing the stored V2Lt columns must not move a
    single bit of the recovery (measured 0.0 on Sig/Gam/U)."""
    q6 = np.array([1.0, 0, 0, 0, 0, 0])
    r = runs["p4_nu0"]
    Ga, Sa, Ua = dehom_fields(r, np.zeros(6), qt6=q6)
    r2 = dict(r)
    r2["V2Lt"] = np.zeros_like(np.asarray(r["V2Lt"]))
    Gb, Sb, Ub = dehom_fields(r2, np.zeros(6), qt6=q6)
    assert np.abs(np.asarray(Sa) - np.asarray(Sb)).max() <= 1e-14
    assert np.abs(np.asarray(Ga) - np.asarray(Gb)).max() <= 1e-14
    assert np.abs(np.asarray(Ua) - np.asarray(Ub)).max() <= 1e-14
