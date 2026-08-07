"""solid_props.py -- equivalent 3-D SOLID properties from the shell cross-section SG.

THE formulation is the author's "MSG shell solid properties" document (OneDrive
`opensg_shell solid properties/MSG_shell_solid_properties.pdf`, consolidated from the
handwritten `Final formulation 2`): the same RM shell SG and fluctuation operators as
the Timoshenko ring, homogenized into classical 3-D elasticity.

    eps_shell = Gamma_e * ebar + Gamma_h * w
    ebar = [G11, G22, G33, 2G23, 2G13, 2G12]   (1 = beam axis, 2/3 = cross axes)
    w    = [w1, w2, w3 | om1, om2, om3]

Gamma_h is EXACTLY the fluctuation operator already assembled by the Timoshenko code
(segment_indep.quad_ops_indep_batch: BDh membrane/curvature rows, BGh shear rows,
including the axial d/dy1 columns, which vanish on the axially-constant strip).
Gamma_e is the document's 8x6 macro operator (Section 4); its rows are implemented
verbatim in solid_macro_ops_batch below, with the three derivation-audited
corrections (2026-08-06, confirmed by the author): the 2*eps12 off-diagonal buckets
carry the pair-sum ONCE (see derivation_2eps12.pdf -- the outer 2 belongs to the
diagonal entries only), and the 2*Gamma_23 buckets of the shear rows carry
X21*C33 / X22*C33 (the page-6 expansion coefficients; the undeformed-state test and
the 2-D solid reference both confirm).  The drilling rotation om3 has
NO solid-strain term (the document's page-10 cancellation); its multiplied-through
residual  g = C33*om3 + C31*om1 + C32*om2 - 1/2(X_i2 w_i;1 - X_i1 w_i;2) = 0  is the
code's existing DR row with a ZERO macro column, enforced by the same element-constant
Lagrange multipliers as the Timoshenko model.

FE process (document Section 6): one KKT solve for V0 with the 4-mode rigid kernel;
D_eff = Dee + V0^T Dhe (6x6); C3D = D_eff / w_SG.  No V1 step.
"""
import numpy as np

from .segment_indep import (assemble_segment_indep, assemble_constraint,
                            _surf_frame_batch, _tie_rows_batch, _shear_batch,
                            quad_ops_indep_batch, NDOF6)

GBAR_ORDER = ("strains [e11 e22 e33 2e23 2e13 2e12] -> stiffness C11..C66  "
              "(1=beam axis, 2/3=cross-section axes)")


def solid_macro_ops_batch(Xe, e3e, xi, eta, cross, ax):
    """The document's Gamma_e at (xi,eta): BDe6 (ne,6,6) on the membrane+curvature
    rows [e11,e22,2e12,K11,K22,K12+K21], BGe6 (ne,2,6) on the shear rows
    [2g13,2g23]; dA (ne,).  Entries verbatim from Eq. (17) of the document."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    ne = Xe.shape[0]
    X11, X21, X31 = c["x11"], c["x21"], c["x31"]        # X_{i1} = x_{i;1}
    X12, X22, X32 = c["x12"], c["x22"], c["x32"]        # X_{i2} = x_{i;2}
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]           # C_{3i}
    z = np.zeros(ne)

    BDe6 = np.stack([
        np.stack([X11**2, X21**2, X31**2,
                  X31*X21, X11*X31, X11*X21], 1),
        np.stack([X12**2, X22**2, X32**2,
                  X32*X22, X32*X12, X12*X22], 1),
        np.stack([2*X11*X12, 2*X22*X21, 2*X31*X32,
                  X22*X31 + X32*X21, X12*X31 + X32*X11,
                  X12*X21 + X11*X22], 1),
        np.stack([z, z, z, z, z, z], 1),                # K11: no solid-strain content
        np.stack([z, z, z, z, z, z], 1),                # K22
        np.stack([z, z, z, z, z, z], 1),                # K12+K21
    ], axis=1)
    # Shear rows: the array C3i*Xja is NOT symmetric, so the eb-consistent
    # representation of the raw gamma-vector rows (for a symmetric macro state
    # ub_i,j = G_ij) carries HALF the pair-sum on each 2G_ij column, diagonals
    # single.  Full pair-sums double the drive and clash with the drilling-tied
    # omega_3 under the axial-shear warping mechanism (see the axial-shear note:
    # the mechanism u_total = rigid rotation about e3 must be zero-energy).
    BGe6 = np.stack([
        np.stack([C31*X11, C32*X21, C33*X31,
                  0.5*(X31*C32 + X21*C33), 0.5*(X31*C31 + X11*C33),
                  0.5*(X21*C31 + X11*C32)], 1),         # 2g13 row
        np.stack([C31*X12, C32*X22, C33*X32,
                  0.5*(X32*C32 + X22*C33), 0.5*(X32*C31 + X12*C33),
                  0.5*(X22*C31 + X12*C32)], 1),         # 2g23 row
    ], axis=1)
    return BDe6, BGe6, dA


def assemble_solid_macro(nodes, quads, subdom, e3s, D_by, G_by, cross, ax,
                         dof_map=None, shear="mitc4_g23"):
    """Dhe6 (ndof,6), Dee6 (6,6): the solid-macro blocks, mirroring
    assemble_segment_indep's quadrature, wall-law lookup, tied-shear scheme and
    dof_map.  The drilling residual has NO solid-strain column, so no multiplier or
    penalty cross terms arise here."""
    if dof_map is None:
        dof_map = np.arange(len(nodes))
    dof_map = np.asarray(dof_map, int)
    nodes = np.asarray(nodes, float); quads = np.asarray(quads, int)
    ne = len(quads)
    Nn = int(np.max(dof_map)) + 1; ndof = NDOF6 * Nn
    Xe = nodes[quads]; e3e = np.asarray(e3s, float)
    sd = np.asarray(subdom, int)
    keys = sorted(set(int(s) for s in sd))
    Darr = np.stack([np.asarray(D_by[k], float) for k in keys])
    Garr = np.stack([np.asarray(G_by[k], float) for k in keys])
    pos = {k: i for i, k in enumerate(keys)}
    sdi = np.array([pos[int(s)] for s in sd])
    De = Darr[sdi]; Gm = Garr[sdi]

    g = (NDOF6 * dof_map[quads])[:, :, None] + np.arange(NDOF6)[None, None, :]
    g = g.reshape(ne, 24)
    tie = None if shear == "full" else _tie_rows_batch(Xe, e3e, cross, ax)

    Ehe = np.zeros((ne, 24, 6)); Dee = np.zeros((6, 6))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        _, BDh, _, _, BGh, _, _, _, _, dA = quad_ops_indep_batch(
            Xe, e3e, xi, eta, cross, ax)
        BDe6, BGe6, _ = solid_macro_ops_batch(Xe, e3e, xi, eta, cross, ax)
        BGt = BGh if shear == "full" else _shear_batch(xi, eta, shear, BGh, tie)
        w = dA[:, None, None]
        DBe = np.einsum('eij,ejb->eib', De, BDe6)
        GBe = np.einsum('eij,ejb->eib', Gm, BGe6)
        Ehe += w * (np.einsum('eia,eib->eab', BDh, DBe)
                    + np.einsum('eia,eib->eab', BGt, GBe))
        Dee += np.einsum('e,eab->ab', dA,
                         np.einsum('eia,eib->eab', BDe6, DBe)
                         + np.einsum('eia,eib->eab', BGe6, GBe))
    Dhe = np.zeros((ndof, 6))
    np.add.at(Dhe, g.reshape(-1), Ehe.reshape(-1, 6))
    return Dhe, Dee


def ring_solid(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross, h=None,
               shear="mitc4_g23", lam_space="elem", return_fields=False,
               periodic=False):
    """Equivalent-solid homogenization of the ring SG.  Returns D_eff (6,6), the
    contour-integrated stiffness per unit axial length on GBAR_ORDER; with
    return_fields=True also the 6-column warping V0.

    Mirrors ring_indep: one-quad-deep prismatic strip, top row DOF-mapped onto the
    bottom, element-constant drilling Lagrange multipliers (zero macro column),
    rigid-kernel KKT, single V0 solve, D_eff = Dee + V0^T Dhe.  No V1.

    periodic=True additionally ties opposite bounding-box edges of the contour
    through `periodic_map.shell_periodic_assembly_map` -- the shell analogue of
    the solid side's mesh_to_periodic_sparse_assembly_map.  The tie rides in the
    assembly map (element connectivity re-pointed at master nodes), so no
    constraint rows are added, and the kernel drops the in-plane rotation,
    keeping the 3 translations.  Without it an isolated cross-section is a FREE
    SG, whose equivalent-solid stiffness is rank one by construction: every
    macro strain except Gamma_11 is cancelled at zero energy by an affine
    fluctuation, which only periodicity forbids."""
    from .fe_jax.msg_rm_timo import build_C_Psi
    from scipy.linalg import lu_factor, lu_solve

    m = len(rx)
    if h is None:
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])
    if periodic:
        from .periodic_multiscale import mesh_to_periodic_sparse_assembly_map
        # feed one "cell" per node so the returned reduced connectivity IS the
        # node -> master map the assemblers take as dof_map
        rc, _ = mesh_to_periodic_sparse_assembly_map(
            m, np.arange(m)[:, None], rx[:, cross], 3, NDOF6)
        node_master = np.asarray(rc, int).ravel()
    else:
        node_master = np.arange(m)
    dof_map = np.concatenate([node_master, node_master])
    quads = np.array([[a, b, m + b, m + a] for a, b in rcells], dtype=int)
    e3q = np.asarray(re3)

    Dhh, _, _, _, _, _ = assemble_segment_indep(
        nodes, quads, rsub, e3q, D_by, G_by, np.asarray(k22_edge), cross, ax,
        kg_e=None, pen=0.0, dof_map=dof_map, shear=shear)
    Gc, _, _ = assemble_constraint(nodes, quads, rsub, e3q, np.asarray(k22_edge),
                                   cross, ax, dof_map=dof_map, lam_space=lam_space)
    Dhe6, Dee6 = assemble_solid_macro(nodes, quads, rsub, e3q, D_by, G_by,
                                      cross, ax, dof_map=dof_map, shear=shear)
    Dhh = np.asarray(Dhh) / h; Gc = Gc / h
    Dhe6 = Dhe6 / h; Dee6 = Dee6 / h

    M = Dhh.shape[0]; P = Gc.shape[0]
    C5, Psi5 = build_C_Psi(rx[:, cross], rcells, p=1)
    # rigid kernel: 3 translations + the in-plane rotation for a free SG; tying
    # opposite edges removes the rotation, so a periodic cell carries only 3
    nk = 3 if periodic else 4
    C6 = np.zeros((nk, M))
    for n in range(m):
        s = NDOF6 * node_master[n]
        C6[:, s:s + 5] += C5[:nk, 5 * n:5 * n + 5]

    naug = M + P
    A = np.zeros((naug + nk, naug + nk))
    A[:M, :M] = Dhh; A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
    A[:M, naug:] = C6.T; A[naug:, :M] = C6
    R0 = np.zeros((naug + nk, 6)); R0[:M] = -Dhe6         # zero macro drilling column
    # The kernel rows pin the rigid modes, but [Dhh; Gc] also has a null
    # direction per node along the drilling rotation (om3 enters no strain row
    # and the element-constant multipliers do not span it).  Those directions
    # carry no load -- Dhe6 is orthogonal to them -- so the minimum-norm
    # least-squares solution is the physical one, whereas an LU factorization
    # of the rank-deficient KKT returns garbage (it produced a negative C44).
    V0 = np.linalg.lstsq(A, R0, rcond=None)[0][:naug]
    Deff = Dee6 + V0[:M].T @ Dhe6
    Deff = 0.5 * (Deff + Deff.T)
    if return_fields:
        return Deff, np.asarray(V0[:M])
    return Deff


def elastic_constants(C3D):
    """The 9 engineering constants from the compliance S = inv(C3D):
    E1,E2,E3 = 1/S11,1/S22,1/S33; G23,G13,G12 = 1/S44,1/S55,1/S66;
    nu12 = -S12*E1, nu13 = -S13*E1, nu23 = -S23*E2.  Also returns cond(C3D)."""
    C = np.asarray(C3D, float)
    S = np.linalg.inv(C)
    out = {
        "E1": 1.0 / S[0, 0], "E2": 1.0 / S[1, 1], "E3": 1.0 / S[2, 2],
        "G23": 1.0 / S[3, 3], "G13": 1.0 / S[4, 4], "G12": 1.0 / S[5, 5],
        "nu12": -S[0, 1] / S[0, 0], "nu13": -S[0, 2] / S[0, 0],
        "nu23": -S[1, 2] / S[1, 1],
        "cond": float(np.linalg.cond(C)),
    }
    return out, S


def build_solid_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source="msg",
                       cell_area=None, periodic=False):
    """Load the shell yaml exactly as build_rm_bundle does (same reference logic,
    same MSG wall transverse-shear upgrade), run ring_solid, and package:

        {"C3D", "D_eff", "cell_area", "area_source", "V0", geometry..., "order"}

    C3D = D_eff / cell_area.  cell_area=None -> convex-hull area of the contour."""
    import yaml as _yaml
    from .oml_ring import load_ring_ref

    d = _yaml.safe_load(open(shell_yaml))
    if ref is None:
        ref = d.get("reference", "center")
    R = load_ring_ref(shell_yaml, ref)
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    G_by = list(R["G_by"])
    if g_source == "msg":
        from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
        from .emit_abd import material_db_from_yaml
        _mdb = material_db_from_yaml(d["materials"])
        for si, sec in enumerate(d["sections"]):
            _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
            _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                               [p[0] for p in _pl], _mdb, fraction=frac)
            if _rr["G_msg"] is not None:
                G_by[si] = np.asarray(_rr["G_msg"])
    Deff, V0 = ring_solid(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], G_by,
                          R["k22"], R["ax"], R["cross"], shear=shear,
                          lam_space="elem", return_fields=True, periodic=periodic)
    area_source = "user"
    if cell_area is None:
        from scipy.spatial import ConvexHull
        cell_area = float(ConvexHull(R["rx"][:, R["cross"]]).volume)
        area_source = "hull"
    return {"C3D": Deff / float(cell_area), "D_eff": Deff,
            "cell_area": float(cell_area), "area_source": area_source,
            "V0": V0, "rx3": np.asarray(R["rx"]),
            "red_cells": np.asarray(R["cells"]), "rsub": np.asarray(R["rsub"]),
            "re3": np.asarray(R["re3"]), "k22": np.asarray(R["k22"]),
            "ax": int(R["ax"]), "cross": list(R["cross"]), "ref": ref,
            "g_source": g_source, "order": GBAR_ORDER}
