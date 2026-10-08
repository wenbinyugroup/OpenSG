"""The IN-PLANE channel of the shear-refined plate recovery on a 3-D SG.

test_plate_shear_refined_anchors pins sigma33 and sigma13 on a 1-D SG;
test_plate_ladder_general pins the 8x8 law and the sigma33/q-column
profiles across dimensions.  Neither can touch the IN-PLANE rows of
Gamma_h, and that is structural, not an oversight:

    sg_dehom._elem_chain pads the warping gradient by dimension --
        n_sg == 1:  dphi_dx -> [0, 0, d/dx0]
        n_sg == 2:  dphi_dx -> [0, d/dx0, d/dx1]
        n_sg == 3:  used as-is
    so rows 0 (11), 1 (22) and 5 (12) of Gamma_h are IDENTICALLY ZERO on
    a 1-D SG and only half-live on a 2-D one.  A laminate meshed as a
    3-D slab does not help either: its warping is a function of y3
    alone, so the in-plane derivatives of the FE field are machine-zero
    there too.  Every existing anchor is therefore blind to a factor
    error in the n_sg == 3 in-plane branch of the recovery -- exactly
    the branch a lattice SG leans on.

This file closes that gap three ways:

  1. ABSOLUTE closed form.  A homogeneous isotropic block meshed as a
     3-D SG has the exact plane-stress answer sigma11 = Q11 (e11 + y3
     k11) + Q12 (e22 + y3 k22), Q11 = E/(1-nu^2) -- no reference
     implementation needed, and it is the in-plane AMPLITUDE itself.
     At nu = 0 the exact warping is zero, so every macro state lands at
     machine precision on any mesh; at nu = 0.3 the exact warping is
     linear in y3 for a MEMBRANE state (in the trilinear space, still
     machine-exact) and quadratic for a BENDING one (NOT in the p1
     space -- the element dV-mean is the exact quantity there, and the
     pointwise field is O(h) off, measured 2.3% at nz = 8).

  2. PROBED banks.  blank_ladder zeroes every stored warping column and
     one bank is set back to a field whose gradient is known in closed
     form (w_i = A_ij y_j, exactly representable in any FE space), so
     each refined kernel -- _v2_batch, _v266_batch's D and T chains,
     _vq_batch -- is made to return its Gamma_h/Gamma_l ALONE and is
     checked against the constant symmetric gradient of A.  This is the
     only construction in the suite that drives the in-plane rows of
     Gamma_h at n_sg == 3, and it is absolute.

  3. PARITY + LINEARITY + the row split, so an n_sg-dependent branch
     error, a broken ablation ladder or a silent edit to the Eq. 66 row
     split all fail loudly.

Measured on the server (2026-08-31, opensg_2_0, JAX_ENABLE_X64=1);
tolerances are pinned several orders above these:
  nu=0 hex8  3-D, general 6-state:  rel 6.4e-16  (abs 1.5e-12 on 2295)
  nu=0 tet4  3-D, general 6-state:  rel 7.8e-16
  A6 vs analytic ABD, nu=0 3-D:     rel 2.3e-16
  nu=0.3 hex8 3-D membrane:         rel 2.9e-15
  nu=0.3 hex8 3-D bending:          element dV-mean rel 2.5e-16, but
      POINTWISE 2.3e-2 -- the p1 warping-space error, which is exactly
      why the mean is the anchor and the pointwise value only a guard
  linear-field probe, all 10 banks x n_sg 1/2/3:  max 1.4e-15 absolute
      (seam-free elements 8/8 at 1-D, 8/16 at 2-D, 16/36 at 3-D)
  1-D vs 2-D vs 3-D layer-mean parity, all drives: rel 1.9e-14

Mutation-checked (2026-08-31): seven injected defects, each caught.  The
headline one -- Gamma_h's in-plane rows scaled by 0.915 for n_sg == 3
ONLY, i.e. the reported "right shape, ~8.5% short in amplitude"
signature -- is caught HERE and by nothing else: all 12 tests of
test_plate_ladder_general and test_plate_shear_refined_anchors still
pass with it in place.  Same for a 3-D SG wrongly padded like a 1-D one,
a mis-scaled classical Ge tilt, a scrambled Gamma_l row map and a
silently dropped detilt.
"""
import numpy as np
import pytest

from opensg_solid.sg_homo import plate_homo_2d, _detilt_cols_2d
from opensg_solid.sg_dehom import dehom_fields

E0, NU, H, NZ = 1000.0, 0.3, 1.0, 8
W = 0.25                                  # in-plane cell size


# --------------------------------------------------------------- meshes
def mat(nu):
    """In:  nu float.  Out: the one-material dict of an isotropic SG."""
    return {1: {"type": 0, "E": E0, "nu": nu}}


def sg_1d(nu=NU, nz=NZ):
    """In:  nu; nz elements over [-H/2, H/2].
    Out: SG dict -- the p1 interval stack (the anchor dimension)."""
    z = np.linspace(-H / 2, H / 2, nz + 1)
    nodes = np.zeros((nz + 1, 3))
    nodes[:, 0] = z
    return {"dim": 1, "nodes": nodes,
            "cells": [[k, k + 1] for k in range(nz)],
            "mat_id": np.ones(nz, int), "materials": mat(nu),
            "scale": 1.0}


def sg_2d(nu=NU, nz=NZ, nx=2):
    """In:  nu; nz through-thickness layers; nx in-plane divisions.
    Out: SG dict -- the quad4 strip of the same plate."""
    z = np.linspace(-H / 2, H / 2, nz + 1)
    xs = np.linspace(0, W, nx + 1)
    nid = lambda i, k: k * (nx + 1) + i                      # noqa: E731
    nodes = np.zeros(((nx + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for i in range(nx + 1):
            nodes[nid(i, k), :2] = (xs[i], z[k])
    cells = [[nid(i, k), nid(i + 1, k), nid(i + 1, k + 1),
              nid(i, k + 1)] for k in range(nz) for i in range(nx)]
    return {"dim": 2, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int), "materials": mat(nu),
            "scale": 1.0}


def _hex_cells(nx, ny, nz, nid):
    """In:  nx/ny/nz counts; nid(i, j, k) node numbering callable.
    Out: list of hex8 connectivities in the gmsh corner order."""
    return [[nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k),
             nid(i, j + 1, k), nid(i, j, k + 1), nid(i + 1, j, k + 1),
             nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
            for k in range(nz) for j in range(ny) for i in range(nx)]


def sg_3d(nu=NU, nz=NZ, nx=2, ny=2, tet=False):
    """In:  nu; nz through-thickness layers; nx x ny in-plane fill of a
         W x W cell; tet -- split every hex into the 6-tet fan.
    Out: SG dict -- the 3-D slab of the same plate (thickness on x3)."""
    z = np.linspace(-H / 2, H / 2, nz + 1)
    xs, ys = np.linspace(0, W, nx + 1), np.linspace(0, W, ny + 1)
    nid = lambda i, j, k: (k * (ny + 1) + j) * (nx + 1) + i  # noqa: E731
    nodes = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes[nid(i, j, k)] = (xs[i], ys[j], z[k])
    hexes = _hex_cells(nx, ny, nz, nid)
    if not tet:
        cells = hexes
    else:
        T6 = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
              (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
        cells = [[h[a], h[b], h[c], h[d]] for h in hexes
                 for a, b, c, d in T6]
    return {"dim": 3, "nodes": nodes, "cells": cells,
            "mat_id": np.ones(len(cells), int), "materials": mat(nu),
            "scale": 1.0}


# -------------------------------------------------------------- helpers
def gauss_z(r):
    """In:  r homogenization dict (single batch).
    Out: (E, Q) physical THICKNESS coordinate of every Gauss point (the
         last SG coordinate, the one Ge tilts with)."""
    xe = np.asarray(r["x_end"])
    return np.einsum("qn,en->eq", np.asarray(r["phi_qn"]),
                     xe[:, :, r["n_sg"] - 1])


def gauss_dv(r):
    """In:  r homogenization dict (single batch).
    Out: (E, Q) Gauss measure detJ * W -- integral f dV = (f * dV).sum()."""
    J = np.einsum("end,qnp->eqdp", np.asarray(r["x_end"]),
                  np.asarray(r["dphi_dxi_qnp"]))
    detJ = (np.abs(np.linalg.det(J)) if r["n_sg"] > 1
            else np.abs(J[..., 0, 0]))
    return detJ * np.asarray(r["W_q"])[None, :]


def analytic_plate(E, nu, y3, eps6):
    """Exact local state of a HOMOGENEOUS isotropic plate SG under a
    macro plate strain -- the plane-stress answer the SG warping
    reproduces exactly (sigma33 = sigma13 = sigma23 = 0 identically).

    In:  E, nu floats; y3 (n,) thickness coordinate; eps6 (6,) the macro
         measures [e11, e22, 2e12, k11, k22, 2k12].
    Out: (Gam (n, 6), Sig (n, 6)) in SwiftComp order (xx yy zz yz xz xy)."""
    y3 = np.asarray(y3, float).ravel()
    e11 = eps6[0] + y3 * eps6[3]
    e22 = eps6[1] + y3 * eps6[4]
    g12 = eps6[2] + y3 * eps6[5]
    Q, G = E / (1.0 - nu * nu), E / (2.0 * (1.0 + nu))
    Gam = np.zeros((y3.size, 6))
    Gam[:, 0], Gam[:, 1] = e11, e22
    Gam[:, 2] = -nu / (1.0 - nu) * (e11 + e22)
    Gam[:, 5] = g12
    Sig = np.zeros((y3.size, 6))
    Sig[:, 0] = Q * (e11 + nu * e22)
    Sig[:, 1] = Q * (e22 + nu * e11)
    Sig[:, 5] = G * g12
    return Gam, Sig


def analytic_abd(E, nu, h):
    """In:  E, nu, h.  Out: (6, 6) the closed-form plate ABD of a
    homogeneous isotropic plate about its mid-surface, in the engine's
    [N11 N22 N12 M11 M22 M12] x [e11 e22 2e12 k11 k22 2k12] order."""
    Q, G = E / (1.0 - nu * nu), E / (2.0 * (1.0 + nu))
    q3 = np.array([[Q, nu * Q, 0.0], [nu * Q, Q, 0.0], [0.0, 0.0, G]])
    A6 = np.zeros((6, 6))
    A6[:3, :3] = h * q3
    A6[3:, 3:] = h ** 3 / 12.0 * q3
    return A6


def layer_means(r, F, nz=NZ):
    """Thickness-binned means -- the dimension-independent comparison
    handle (Gauss clouds of a 1-D, 2-D and 3-D mesh do not share point
    positions, but they share the layer partition).

    In:  r homogenization dict; F (E, Q, 6) a recovered field; nz layers.
    Out: (nz, 6) the plain mean of F over the Gauss points of each
         thickness layer of [-H/2, H/2]."""
    z = gauss_z(r).ravel()
    S = np.asarray(F).reshape(-1, 6)
    edge = np.linspace(-H / 2, H / 2, nz + 1)
    b = np.clip(np.digitize(z, edge) - 1, 0, nz - 1)
    return np.array([S[b == k].mean(axis=0) for k in range(nz)])


_LADDER_KEYS = ("V0_ladder", "V11", "V12", "V11bar", "V12bar",
                "V11barD", "V12barD", "V21", "V22", "V23",
                "V21t", "V22t", "V23t", "V1Lt", "V2Lt", "V1Lb", "V2Lb")


def blank_ladder(r):
    """In:  r refined-plate dict.
    Out: a shallow copy with V0 AND every stored ladder/load column
         zeroed (shapes kept).  Set one bank back to a chosen field and
         the recovery returns that bank's chain ALONE -- the probe
         harness the Gamma_h/Gamma_l anchors below are built on."""
    r2 = dict(r)
    r2["V0"] = np.zeros_like(np.asarray(r["V0"], float))
    for k in _LADDER_KEYS:
        if r2.get(k) is not None:
            r2[k] = np.zeros_like(np.asarray(r2[k], float))
    return r2


def reduced_coords(r):
    """In:  r homogenization dict (single batch).
    Out: (n_node, 3) coordinate carried by every reduced (periodic-
         master) node, ZERO-PADDED so an n_sg-dim SG occupies the LAST
         n_sg slots of (x, y, z) -- the same padding the recovery
         kernels apply to the gradients."""
    cells = np.asarray(r["periodic_cells_en"])
    xe = np.asarray(r["x_end"])
    n_sg = int(r["n_sg"])
    X = np.zeros((int(cells.max()) + 1, 3))
    X[cells.ravel(), 3 - n_sg:] = xe.reshape(-1, n_sg)
    return X


def clean_elements(r, X3):
    """Elements whose node coordinates AGREE with the reduced-node table
    -- i.e. that carry no periodic image.  A field defined by a formula
    on X3 is exactly that formula on these elements, so its FE gradient
    is the formula's gradient; on a seam element the shared dof holds
    the master's value at the slave's position and the field is (by
    construction) discontinuous.

    In:  r homogenization dict; X3 (n_node, 3) from reduced_coords.
    Out: (m,) int element indices."""
    cells = np.asarray(r["periodic_cells_en"])
    xe = np.asarray(r["x_end"])
    n_sg = int(r["n_sg"])
    d = np.abs(X3[cells][:, :, 3 - n_sg:] - xe).max(axis=(1, 2))
    return np.nonzero(d < 1e-12)[0]


def linear_field(r, A):
    """The nodal warping w_i(y) = A_ij y_j on the reduced dof space.

    In:  r homogenization dict; A (3, 3) gradient matrix.
    Out: (w_flat (n_unique,), Adj (3, 3), keep (m,) clean elements).
         Adj is A with the UNRESOLVED coordinate columns struck out
         (a d-dim SG has no derivative along the padded slots), i.e.
         the gradient the recovery must return."""
    X3 = reduced_coords(r)
    n_sg = int(r["n_sg"])
    Adj = np.zeros((3, 3))
    Adj[:, 3 - n_sg:] = np.asarray(A, float)[:, 3 - n_sg:]
    w = X3 @ Adj.T                                   # (n_node, 3)
    return w.ravel(), Adj, clean_elements(r, X3)


def gamma_of(Adj):
    """In:  Adj (3, 3) a constant displacement gradient dw_i/dy_j.
    Out: (6,) the SwiftComp-order engineering strain Gamma_h w --
         [w1,1  w2,2  w3,3  w3,2+w2,3  w3,1+w1,3  w2,1+w1,2]."""
    return np.array([Adj[0, 0], Adj[1, 1], Adj[2, 2],
                     Adj[2, 1] + Adj[1, 2], Adj[2, 0] + Adj[0, 2],
                     Adj[1, 0] + Adj[0, 1]])


def interp_nodal(r, w_flat):
    """In:  r homogenization dict; w_flat (n_unique,) nodal 3-vector
         field, node-major.
    Out: (E, Q, 3) the field VALUE at the Gauss points -- what the
         Gamma_l1/Gamma_l2 operators consume."""
    cells = np.asarray(r["periodic_cells_en"])
    w = np.asarray(w_flat, float).reshape(-1, 3)[cells]
    return np.einsum("qn,end->eqd", np.asarray(r["phi_qn"]), w)


def first_moment_weights(r):
    """In:  r homogenization dict (single batch).
    Out: (n_node,) the EXACT first moments int(y3 N_a dV) scattered onto
         the reduced nodes -- the measure _detilt_cols_2d projects with
         (sg_assembly.plate_ladder_element_blocks' wyN)."""
    cells = np.asarray(r["periodic_cells_en"])
    phi = np.asarray(r["phi_qn"])
    wy_eqn = phi[None, :, :] * (gauss_z(r) * gauss_dv(r))[:, :, None]
    out = np.zeros(int(cells.max()) + 1)
    np.add.at(out, cells.ravel(),
              wy_eqn.sum(axis=1).ravel())
    return out


# UNIT-scale macro states (the suite's convention: unit drives, so a
# recovered stress is O(E) and a tolerance reads as a relative error).
E6_GEN = np.array([1.3, -0.7, 0.9, 2.1, -1.1, 0.6])
E6_MEM = np.array([1.3, -0.7, 0.9, 0.0, 0.0, 0.0])
E6_BEN = np.array([0.0, 0.0, 0.0, 2.1, -1.1, 0.6])
DE1 = np.array([0.0, 0, 0, 1.7, 0, 0])
DE2 = np.array([0.0, 0, 0, 0, -0.9, 0])
DE11 = np.array([0.0, 0, 0, 1.3, 0, 0])
DE12 = np.array([0.0, 0, 0, 0, 0, 0.5])
DE22 = np.array([0.0, 0, 0, 0, -0.8, 0])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The homogenizations every test in this module shares (refined, so
    the V2 chains and load columns are stored at every dimension)."""
    import os
    os.chdir(tmp_path_factory.mktemp("inplane3d"))
    return {
        "1d": plate_homo_2d(sg_1d(), refined=1),
        "2d": plate_homo_2d(sg_2d(), refined=1),
        "3d": plate_homo_2d(sg_3d(), refined=1),
        "3d_nu0": plate_homo_2d(sg_3d(nu=0.0), refined=1),
        "3d_tet_nu0": plate_homo_2d(sg_3d(nu=0.0, nz=4, tet=True),
                                    refined=1),
        "3d_wide": plate_homo_2d(sg_3d(nz=4, nx=3, ny=3), refined=1),
    }


# ===================================================================== 1
# ABSOLUTE closed-form anchors on a 3-D SG -- the in-plane AMPLITUDE
# =====================================================================
@pytest.mark.parametrize("tag", ["3d_nu0", "3d_tet_nu0"])
def test_analytic_inplane_stress_3d_nu0(runs, tag):
    """A homogeneous nu = 0 block meshed as a 3-D SG: the exact warping
    is IDENTICALLY ZERO (no Poisson contraction to represent), so the
    recovery must return the closed-form plate state POINTWISE on any
    mesh and at any element order -- hex8 and the tet4 fan alike.

    This is the absolute in-plane amplitude anchor: sigma11 = E (e11 +
    y3 k11) and sigma22 = E (e22 + y3 k22) with NO reference
    implementation in the loop, on the n_sg == 3 branch that no 1-D
    anchor can reach.  Anything that multiplies the in-plane recovery by
    a wrong factor fails here."""
    r = runs[tag]
    z = gauss_z(r).ravel()
    Gam, Sig, _ = dehom_fields(r, E6_GEN)
    Ga, Sa = analytic_plate(E0, 0.0, z, E6_GEN)
    scale = np.abs(Sa).max()
    assert scale > 1.0                              # a real signal
    assert np.abs(np.asarray(Sig).reshape(-1, 6) - Sa).max() \
        <= 1e-10 * scale
    assert np.abs(np.asarray(Gam).reshape(-1, 6) - Ga).max() \
        <= 1e-10 * np.abs(Ga).max()


def test_analytic_abd_3d(runs):
    """The same block's 6x6 plate law, closed form.  A6 is what the
    in-plane recovery is calibrated against, so pin it in the same
    place: at nu = 0 the warping space is exact and BOTH the membrane
    and the bending blocks land on h Q and h^3/12 Q."""
    for key in ("A6_ladder", "C_eff"):
        A6 = np.asarray(runs["3d_nu0"][key], float)
        ref = analytic_abd(E0, 0.0, H)
        assert np.abs(A6 - ref).max() <= 1e-10 * np.abs(ref).max(), key


def test_analytic_membrane_stress_3d_poisson(runs):
    """nu = 0.3, MEMBRANE macro state: the exact warping w3 = -nu/(1-nu)
    y3 (e11 + e22) is LINEAR in y3, hence inside the trilinear hex
    space, so the Galerkin recovery is again exact POINTWISE.  Brings
    the Poisson coupling into the in-plane amplitude (sigma11 = Q11 e11
    + Q12 e22) -- the nu = 0 anchor above cannot see a mis-scaled Q12."""
    r = runs["3d"]
    z = gauss_z(r).ravel()
    Gam, Sig, _ = dehom_fields(r, E6_MEM)
    Ga, Sa = analytic_plate(E0, NU, z, E6_MEM)
    assert np.abs(Sa[:, 1]).max() > 1.0             # Q12 really loaded
    assert np.abs(np.asarray(Sig).reshape(-1, 6) - Sa).max() \
        <= 1e-10 * np.abs(Sa).max()
    assert np.abs(np.asarray(Gam).reshape(-1, 6) - Ga).max() \
        <= 1e-10 * np.abs(Ga).max()


def test_analytic_bending_stress_3d_poisson(runs):
    """nu = 0.3, BENDING macro state.  The exact w3 is QUADRATIC in y3
    and a trilinear hex cannot hold it, so eps33 is the element-wise L2
    projection of the exact (constant per layer) and the POINTWISE
    in-plane stress carries an O(h) warping-space error -- measured 2.3%
    at nz = 8, and it is a discretization error, not a formulation one.

    The element dV-MEAN is the exact quantity: the projection error is
    odd about the element midplane, so it integrates out and the mean
    must equal the closed form at the element centroid to machine
    precision.  Both are asserted -- the mean tightly (the anchor), the
    pointwise loosely (a guard that the error stays O(h) and does not
    hide a factor)."""
    r = runs["3d"]
    z, dV = gauss_z(r), gauss_dv(r)
    _, Sig, _ = dehom_fields(r, E6_BEN)
    Sig = np.asarray(Sig)
    _, Sa = analytic_plate(E0, NU, z.ravel(), E6_BEN)
    scale = np.abs(Sa).max()
    assert scale > 1.0
    assert np.abs(Sig.reshape(-1, 6) - Sa).max() <= 0.05 * scale

    zc = (z * dV).sum(axis=1) / dV.sum(axis=1)      # element centroids
    _, Sc = analytic_plate(E0, NU, zc, E6_BEN)
    Sm = (Sig * dV[:, :, None]).sum(axis=1) / dV.sum(axis=1)[:, None]
    assert np.abs(Sm - Sc).max() <= 1e-10 * scale


# ===================================================================== 2
# DIMENSIONAL PARITY -- 1-D, 2-D and 3-D SGs of the SAME plate
# =====================================================================
_DRIVES = {
    "bending": dict(epsilon_bar=E6_BEN),
    "membrane": dict(epsilon_bar=E6_MEM),
    "pressure": dict(epsilon_bar=np.zeros(6),
                     qt6=np.array([1.0, 0, 0, 0, 0, 0])),
    "first_deriv": dict(epsilon_bar=np.zeros(6), dE1=DE1, dE2=DE2),
    "second_deriv": dict(epsilon_bar=np.zeros(6), dE11=DE11,
                         dE12=DE12, dE22=DE22),
}


@pytest.mark.parametrize("drive", sorted(_DRIVES))
def test_recovered_stress_dimensional_parity(runs, drive):
    """ONE homogeneous plate, meshed as a 1-D interval stack, a 2-D quad
    strip and a 3-D hex slab, must return the SAME recovered stress at
    the same y3 -- for the macro state, the load column AND both
    derivative ladders.  ALL SIX components, not just sigma33.

    The 1-D column is the validated dimension (rm_plate_1D parity), so
    any n_sg-dependent branch error in the recovery chain -- a gradient
    padding that drops a row, a bank contracted with the wrong driver,
    a factor applied only when n_sg == 3 -- shows up as a 2-D or 3-D
    departure from it here.  Compared as thickness-layer means, the one
    handle the three Gauss clouds share."""
    kw = dict(_DRIVES[drive])
    eps = kw.pop("epsilon_bar")
    ref = layer_means(runs["1d"], dehom_fields(runs["1d"], eps, **kw)[1])
    scale = np.abs(ref).max()
    assert scale > 1e-6, "drive %s is inert on the 1-D SG" % drive
    for tag in ("2d", "3d"):
        r = runs[tag]
        got = layer_means(r, dehom_fields(r, eps, **kw)[1])
        assert np.abs(got - ref).max() <= 1e-10 * scale, (drive, tag)


def test_inplane_rows_are_carried_not_dropped(runs):
    """The parity above would also be satisfied if the in-plane rows
    were zero everywhere.  They are not: under the bending state the
    3-D SG must carry a real sigma11/sigma22/sigma12, and the 33/23/13
    rows must stay at plane-stress zero.  A guard against a vacuous
    parity pass."""
    _, Sig, _ = dehom_fields(runs["3d"], E6_GEN)
    S = np.abs(np.asarray(Sig).reshape(-1, 6))
    for row in (0, 1, 5):
        assert S[:, row].max() > 0.1 * S.max(), row
    for row in (2, 3, 4):
        assert S[:, row].max() <= 0.05 * S.max(), row


# ===================================================================== 3
# The DETILTED chain's IN-PLANE rows, and every other refined bank
# =====================================================================
_A_PROBE = np.array([[0.31, -0.47, 0.72],
                     [0.58, 0.19, -0.36],
                     [-0.24, 0.65, 0.43]])

_ALL6 = (0, 1, 2, 3, 4, 5)
_BANKS = [
    # (bank key, the dehom_fields driver that contracts it, the driver
    #  slot to load, the output rows that bank's chain owns)
    ("V11bar", "dE1", 0, _ALL6),          # Eq. 63 first order
    ("V12bar", "dE2", 1, _ALL6),
    ("V21", "dE11", 3, (0, 1, 5)),        # Eq. 64-66 D chain: in-plane
    ("V22", "dE12", 5, (0, 1, 5)),
    ("V23", "dE22", 4, (0, 1, 5)),
    ("V21t", "dE11", 3, (2, 3, 4)),       # T chain: 33 / 23 / 13
    ("V22t", "dE12", 5, (2, 3, 4)),
    ("V23t", "dE22", 4, (2, 3, 4)),
    ("V1Lt", "qt6", 0, _ALL6),            # the pressure load ladder
    ("V1Lb", "qb6", 0, _ALL6),
]


@pytest.mark.parametrize("tag", ["1d", "2d", "3d_wide"])
@pytest.mark.parametrize("bank,drv,slot,rows", _BANKS,
                         ids=[b[0] for b in _BANKS])
def test_gamma_h_of_every_bank_is_the_exact_gradient(runs, tag, bank,
                                                     drv, slot, rows):
    """THE in-plane anchor.  Every stored warping column is zeroed, ONE
    bank is set to the exactly-representable field w_i = A_ij y_j, and
    the recovery is driven so that bank alone is contracted.  Whatever
    that bank's chain does to Gamma_h must then be the CONSTANT
    symmetric gradient of A -- known in closed form, no reference
    implementation.

    On a 3-D SG this drives rows 0 (11), 1 (22) and 5 (12) of Gamma_h
    with genuinely nonzero in-plane derivatives, which no laminate SG at
    any dimension can do (their warping is a function of y3 alone) and
    which the padding makes structurally impossible at n_sg == 1.  The
    D-chain banks V21/V22/V23 are checked on the in-plane rows they own
    and the T-chain banks V21t/V22t/V23t on rows 33/23/13, so the
    detilted chain's in-plane content is asserted directly.

    Run at n_sg = 1, 2 and 3 so a branch that is right in one dimension
    and wrong in another cannot hide."""
    r = runs[tag]
    if r.get(bank) is None:
        pytest.skip("%s carries no %s" % (tag, bank))
    w, Adj, keep = linear_field(r, _A_PROBE)
    assert keep.size, "no seam-free element on %s" % tag

    r2 = blank_ladder(r)
    col = np.zeros_like(np.asarray(r[bank], float))
    if col.ndim == 1:                    # (n_unique,) load column
        col = w.copy()
    else:                                # (n_unique, 6): one slot only
        col[:, slot] = w
    r2[bank] = col
    kw = {drv: np.eye(6)[slot]}
    dGam = np.asarray(dehom_fields(r2, np.zeros(6), **kw)[0])[keep]

    ref = gamma_of(Adj)
    scale = max(np.abs(ref).max(), 1.0)
    for row in range(6):
        want = ref[row] if row in rows else 0.0
        assert np.abs(dGam[:, :, row] - want).max() <= 1e-9 * scale, row
    if int(r["n_sg"]) == 3:              # the channel under test is live
        assert abs(ref[0]) > 0.1 and abs(ref[1]) > 0.1


@pytest.mark.parametrize("tag", ["1d", "2d", "3d_wide"])
def test_gamma_l_value_operators_row_map(runs, tag):
    """The OTHER half of the refined chain: Gamma_l1/Gamma_l2 consume
    warping VALUES, not gradients, so their row map is dimension-
    independent and absolutely checkable --

        Gamma_l1 g = [g1, 0, 0, 0, g3, g2]   (e11 <- w1, 2g13 <- w3,
                                              2g12 <- w2)
        Gamma_l2 g = [0, g2, 0, g3, 0, g1]   (e22 <- w2, 2g23 <- w3,
                                              2g12 <- w1)

    Driving V0_ladder alone (every other bank blank) makes the Eq. 63
    term exactly Gamma_l1(V0 dE1) + Gamma_l2(V0 dE2), so the recovered
    strain is the interpolated nodal field routed through those rows.
    Pins rm_plate_1D._grad_ops on the general SG."""
    r = runs[tag]
    w, _Adj, _keep = linear_field(r, _A_PROBE)
    u = interp_nodal(r, w)                            # (E, Q, 3)
    assert np.abs(u).max() > 1e-6

    for slot, kw in ((1, dict(dE1=np.eye(6)[0])),
                     (2, dict(dE2=np.eye(6)[0]))):
        r2 = blank_ladder(r)
        r2["V0_ladder"] = np.outer(w, np.eye(6)[0])
        G = np.asarray(dehom_fields(r2, np.zeros(6), **kw)[0])
        want = np.zeros_like(G)
        if slot == 1:
            want[..., 0], want[..., 4], want[..., 5] = (u[..., 0],
                                                        u[..., 2],
                                                        u[..., 1])
        else:
            want[..., 1], want[..., 3], want[..., 5] = (u[..., 1],
                                                        u[..., 2],
                                                        u[..., 0])
        assert np.abs(G - want).max() <= 1e-10 * np.abs(u).max(), slot


def test_refined_gamma_h_matches_the_classical_kernel(runs):
    """_elem_chain (refined) and _element_dehomo_kernel (classical) must
    build the SAME Gamma_h.  Feed one arbitrary nodal field through both
    -- as the classical V0 column and as the refined V11bar column --
    and the strains must agree to machine precision once the classical
    path's macro Ge term is subtracted.

    The classical kernel is the validated one (it is what produces the
    ABD every external cross-check pins), so this makes it the reference
    for the refined chain's gradient operator INCLUDING the in-plane
    rows at n_sg == 3.  A random field is used deliberately: it has
    nonzero derivatives in every direction, so no row can pass by being
    silently zero."""
    r = runs["3d_wide"]
    rng = np.random.default_rng(20260831)
    w = rng.standard_normal(np.asarray(r["V0"]).shape[0])

    ra = dict(r)
    ra["V0"] = np.outer(w, np.eye(6)[0])
    Ga = np.asarray(dehom_fields(ra, np.eye(6)[0])[0])
    Ga = Ga - np.array([1.0, 0, 0, 0, 0, 0])          # strip Ge e1

    rb = blank_ladder(r)
    rb["V11bar"] = np.outer(w, np.eye(6)[0])
    Gb = np.asarray(dehom_fields(rb, np.zeros(6), dE1=np.eye(6)[0])[0])

    scale = np.abs(Ga).max()
    assert scale > 1.0
    assert np.abs(Ga - Gb).max() <= 1e-10 * scale
    for row in (0, 1, 5):                             # in-plane, live
        assert np.abs(Ga[..., row]).max() > 0.01 * scale, row


def test_detilt_projection_properties():
    """_detilt_cols_2d as a pure operator -- the Eq. 64 D-chain source
    that has no assertion anywhere else.  It must (a) leave the w3
    component untouched, (b) annihilate the exact first moment
    int(y3 W dV) of the two in-plane components, (c) be idempotent (it
    is an L2 projection), and (d) fix any block that is already
    tilt-free.  The lumped-weight signature must still be refused."""
    rng = np.random.default_rng(11)
    n = 21
    y = np.linspace(-0.5, 0.5, n)
    wy = rng.uniform(0.5, 1.5, n) * y                # a valid moment set
    cols = rng.standard_normal((3 * n, 6))

    D = _detilt_cols_2d(cols, y, wy)
    Wc, Wd = cols.reshape(-1, 3, 6), D.reshape(-1, 3, 6)
    assert np.abs(Wd[:, 2, :] - Wc[:, 2, :]).max() == 0.0
    for comp in (0, 1):
        m1 = (wy[:, None] * Wd[:, comp, :]).sum(axis=0)
        assert np.abs(m1).max() <= 1e-10 * np.abs(cols).max()
    assert np.abs(_detilt_cols_2d(D, y, wy) - D).max() \
        <= 1e-12 * np.abs(D).max()
    assert np.abs(Wd - Wc).max() > 1e-3               # it did something
    with pytest.raises(TypeError, match="EXACT first-moment"):
        _detilt_cols_2d(cols, y, wy, np.ones(n))


def test_v11barD_is_the_detilted_v11bar_3d(runs):
    """On the 3-D SG itself: the stored V11barD/V12barD must be V11bar/
    V12bar with the thickness-linear content projected out of the two
    IN-PLANE components and w3 left alone.  Checked against the exact
    int(y3 N_a dV) moments rebuilt here from the run's own quadrature,
    so a change of detilt MEASURE has to be deliberate."""
    r = runs["3d"]
    wy = first_moment_weights(r)
    y = reduced_coords(r)[:, 2]
    for raw, det in (("V11bar", "V11barD"), ("V12bar", "V12barD")):
        A = np.asarray(r[raw], float).reshape(-1, 3, 6)
        B = np.asarray(r[det], float).reshape(-1, 3, 6)
        assert np.abs(B[:, 2, :] - A[:, 2, :]).max() == 0.0, det
        assert np.abs(B[:, :2, :] - A[:, :2, :]).max() > 0.0, det
        for comp in (0, 1):
            m1 = (wy[:, None] * B[:, comp, :]).sum(axis=0)
            assert np.abs(m1).max() <= 1e-8 * np.abs(
                (wy[:, None] * A[:, comp, :]).sum(axis=0)).max() \
                + 1e-12 * np.abs(A).max(), (det, comp)
        assert np.abs(_detilt_cols_2d(
            np.asarray(r[det], float), y, wy)
            - np.asarray(r[det], float)).max() <= 1e-10 * np.abs(A).max()


# ===================================================================== 4
# LINEARITY -- the additivity the stage ladder / ablation relies on
# =====================================================================
_FAMILIES = {
    "macro": dict(epsilon_bar=E6_GEN),
    "first": dict(epsilon_bar=np.zeros(6), dE1=DE1, dE2=DE2,
                  dE11=np.zeros(6)),
    "second": dict(epsilon_bar=np.zeros(6), dE11=DE11, dE12=DE12,
                   dE22=DE22),
    "load": dict(epsilon_bar=np.zeros(6),
                 qt6=np.array([1.0, 0.3, -0.2, 0.5, -0.1, 0.4]),
                 dE11=np.zeros(6)),
}


def _call(r, kw):
    """In: r; kw a _FAMILIES entry.  Out: (E, Q, 6) recovered stress."""
    kw = dict(kw)
    return np.asarray(dehom_fields(r, kw.pop("epsilon_bar"), **kw)[1])


def test_recovery_is_linear_in_every_driver_family(runs):
    """The ablation that characterised the lattice discrepancy ("V0 only
    -> + load column -> + first deriv -> + second deriv") is only
    meaningful if the recovery is LINEAR in its drivers and the families
    SUPERPOSE.  Both are asserted here, on the 3-D SG:

      scaling:      S(alpha d) = alpha S(d) for each family
      superposition: S(macro + first + second + load) = the sum of the
                     four, with the second-order path forced on
                     throughout (passing any dE1x routes _v266_batch
                     instead of _v2_batch, and that switch must not
                     change the answer when the second drivers are zero)

    A stage ladder run against a non-additive chain measures nothing, so
    this gates the diagnostic method itself."""
    r = runs["3d"]
    for name, kw in _FAMILIES.items():
        base = _call(r, kw)
        assert np.abs(base).max() > 1e-9, name
        scaled = dict(kw)
        for k, v in scaled.items():
            scaled[k] = np.asarray(v, float) * -2.5
        got = _call(r, scaled)
        assert np.abs(got + 2.5 * base).max() <= 1e-9 * \
            np.abs(base).max(), name

    allkw = {"epsilon_bar": np.zeros(6)}
    total = np.zeros_like(_call(r, _FAMILIES["macro"]))
    for kw in _FAMILIES.values():
        total = total + _call(r, kw)
        for k, v in kw.items():
            allkw[k] = np.asarray(allkw.get(k, np.zeros(6)), float) \
                + np.asarray(v, float)
    got = _call(r, allkw)
    assert np.abs(got - total).max() <= 1e-9 * np.abs(total).max()


def test_second_order_path_collapses_to_first_order(runs):
    """With ZERO second derivatives the Eq. 64-66 chain must reproduce
    the Eq. 63 first-order term bit for bit -- the row split becomes a
    no-op and _v266_batch degenerates to _v2_batch.  This is what lets
    the stage ladder attribute a change to the second-derivative terms
    alone."""
    r = runs["3d"]
    a = np.asarray(dehom_fields(r, E6_GEN, dE1=DE1, dE2=DE2)[1])
    b = np.asarray(dehom_fields(r, E6_GEN, dE1=DE1, dE2=DE2,
                                dE11=np.zeros(6))[1])
    assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max()


# ===================================================================== 5
# The Eq. 66 ROW SPLIT is a CHOICE -- pin it so a change is deliberate
# =====================================================================
def test_eq66_row_split_is_exactly_rows_2_to_5(runs):
    """sg_dehom._v266_batch sends rows 2:5 (33, 23, 13) to the TILTED
    chain and keeps rows 0/1/5 (11, 22, 12) on the DETILTED one:

        dSig = SigD.at[:, :, 2:5].set(SigT[:, :, 2:5])

    That split is a modelling choice, not an identity, and it is under
    active question (using the detilted chain for ALL rows was measured
    to improve sigma13 and sigma33 on a lattice SG while leaving the
    in-plane rows bit-identical).  This test states the CURRENT contract
    so any edit to it is a deliberate act with a failing test attached,
    not a silent drift:

      * zeroing the T-chain sources V21t/V22t/V23t moves rows 2/3/4 ONLY
      * zeroing the D-chain sources V21/V22/V23 moves rows 0/1/5 ONLY

    Both directions are asserted, so neither a widened nor a narrowed
    slice survives."""
    r = runs["3d"]
    kw = dict(dE11=DE11, dE12=DE12, dE22=DE22)
    base = np.asarray(dehom_fields(r, E6_GEN, **kw)[1])
    scale = np.abs(base).max()

    for keys, moved, still in ((("V21t", "V22t", "V23t"), (2, 3, 4),
                                (0, 1, 5)),
                               (("V21", "V22", "V23"), (0, 1, 5),
                                (2, 3, 4))):
        r2 = dict(r)
        for k in keys:
            r2[k] = np.zeros_like(np.asarray(r[k], float))
        got = np.asarray(dehom_fields(r2, E6_GEN, **kw)[1])
        for row in still:
            assert np.abs(got[..., row] - base[..., row]).max() == 0.0, \
                (keys, row)
        assert max(np.abs(got[..., row] - base[..., row]).max()
                   for row in moved) > 1e-9 * scale, keys


def test_row_split_chains_really_differ(runs):
    """The split only means something if the two chains disagree: the
    D-chain's Gamma_l uses the DETILTED V11barD/V12barD and the T-chain
    the raw V11bar/V12bar.  Assert the recovered in-plane stress
    actually changes when the D chain is forced onto the tilted columns
    -- otherwise the test above would be pinning a distinction without
    a difference."""
    r = runs["3d"]
    kw = dict(dE11=DE11, dE22=DE22)
    base = np.asarray(dehom_fields(r, E6_GEN, **kw)[1])
    r2 = dict(r)
    r2["V11barD"], r2["V12barD"] = r["V11bar"], r["V12bar"]
    got = np.asarray(dehom_fields(r2, E6_GEN, **kw)[1])
    d = np.abs(got - base)
    assert max(d[..., row].max() for row in (0, 1, 5)) \
        > 1e-9 * np.abs(base).max()
    for row in (2, 3, 4):                     # T chain never saw them
        assert d[..., row].max() == 0.0, row
