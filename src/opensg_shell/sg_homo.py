"""ALL homogenization drivers of the msg-shell package: boundary ring ->
Timoshenko 6x6 (constrained 6-DOF independent-drilling element), the RM
homogenization bundle feeding the two-step dehom (sg_dehom), cross-section SG ->
equivalent 3-D solid, 3-D shell SG -> C3D, and the aperiodic/tapered segment ->
Timoshenko flow.
Merged from: run_ring_indep.py, solid_props.py, shell_sg3d.py, segment_taper.py,
dehom_rm.py (build_rm_bundle and _strip only).

CIRCULARITY NOTE: sg_mesh imports ring_indep from this module, so sg_homo must
NOT import sg_mesh at module level; its sg_mesh imports (load_ring_ref, extract)
are deferred INSIDE the functions that use them (build_solid_bundle,
build_rm_bundle, segment_timo_from_3dyaml), following the deferred-import
pattern segment_taper and shell_sg3d already used.

run_ring_indep.py -- 6-DOF (independent omega_3, element-lambda Lagrange) BOUNDARY RING
---------------------------------------------------------------------------------------
solver + shear-scheme comparison against the validated 5-DOF MITC ring and the 3-D solid.

Ring = one-quad-deep prismatic strip with the top node row DOF-mapped onto the bottom row
(exactly ring_general's construction), assembled with the constrained 6-DOF element; the
drilling constraint is enforced by element-constant Lagrange multipliers and the rigid
modes by the standard KKT rows.  All matrices are per-unit-length (/h).

Public entry point: ring_indep (constrained 6-DOF ring solve).

solid_props.py -- equivalent 3-D SOLID properties from the shell cross-section SG
---------------------------------------------------------------------------------
The same RM shell SG and fluctuation operators as the Timoshenko ring,
homogenized into classical 3-D elasticity.  Public entry points:
build_solid_bundle (yaml -> C3D bundle) and ring_solid (assembled KKT solve).

    eps_shell = Gamma_e * ebar + Gamma_h * w
    ebar = [G11, G22, G33, 2G23, 2G13, 2G12]   (1 = beam axis, 2/3 = cross axes)
    w    = [w1, w2, w3 | om1, om2, om3]

FE process: one KKT solve for V0 with the 4-mode rigid kernel;
D_eff = Dee + V0^T Dhe (6x6); C3D = D_eff / w_SG.  No V1 step.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THE solid_props SECTION   (ne = #elements, m = #contour
# nodes, nred = #master nodes after the periodic tie, ndof = 6*nred, NDOF6 = 6)
# ----------------------------------------------------------------------------
# geometry / frames
#   rx        (m, 3) float     contour nodes (cross-section plane, axial = 0)
#   rcells    (ne, 2) int      line-element connectivity (0-based)
#   rsub      (ne,) int        section/layup id per element
#   re3       (ne, 3) float    wall normal e3 per element
#   h         float            strip depth along the axial direction
#   nodes     (2m, 3) float    strip nodes = [rx ; rx + h*ez]
#   quads     (ne, 4) int      strip quad connectivity [a, b, m+b, m+a]
#   Xe        (ne, 4, 3) float quad corner coordinates per element
#   e3e       (ne, 3) float    = re3
#   xi, eta   float            Gauss-point coordinates on the quad
#   cross     (2,) list int    indices of the two cross-section axes
#   ax        int              index of the axial direction
#   N         (4,) float       bilinear shape functions at (xi, eta)
#   D1, D2    (ne, 4) float    dN/dzeta1, dN/dzeta2 (curvilinear derivatives)
#   c         dict of (ne,)    direction cosines: x11..x32 = X_{i,alpha};
#                              y1, y2, y3 = C_{3i}
#   xi1, xi2  (ne, 3) float    X_{i1}, X_{i2} stacked
#   yv        (ne, 3) float    C_{3i} stacked
#   den       (ne,) float      C33 divisor of DR (1 where |C33| < 1e-8)
#   dA        (ne,) float      area weight at the Gauss point
# operators
#   B -> BDh  (ne, 6, 24)      Gamma_h membrane+curvature rows (slots w | om)
#   Bg -> BGh (ne, 2, 24)      Gamma_h transverse-shear rows
#   Dr -> DRh (ne, 24)         drilling residual row DR (divided by C33)
#   BDe6      (ne, 6, 6)       Gamma_e membrane+curvature rows on ebar
#   BGe6      (ne, 2, 6)       Gamma_e transverse-shear rows on ebar
#   tie       dict of (ne,1,24) g23 Dvorkin-Bathe tying rows (only g23 is tied
#                              for a cross-section; g13 stays at Gauss)
# material laws
#   D_by      list (6, 6)      wall ABD per section (about the reference line)
#   G_by      list (2, 2)      wall transverse-shear law per section
#   De, Gm    (ne,6,6)/(ne,2,2) per-element laws (D_by/G_by gathered by rsub)
#   k22_edge  (ne,) float      edge curvature factor (loader output; unused here)
# assembly
#   node_master (m,) int       node -> master map (periodic tie; identity free)
#   dof_map   (2m,) int        strip dof map = [node_master ; node_master]
#   g         (ne, 24) int     global dof columns per element
#   Ehh, Ehe  (ne,24,24)/(ne,24,6)  per-element blocks before scatter
#   Dhh       (ndof, ndof)     int Gamma_h^T K Gamma_h dA
#   Dhe       (ndof, 6)        int Gamma_h^T K Gamma_e dA
#   Dee       (6, 6)           int Gamma_e^T K Gamma_e dA
#   Gc        (ne, ndof)       int_e DR dA rows (element-constant multipliers)
#   C5        (4, 5m)          rigid-mode rows from build_C_Psi (5 dof/node)
#   C6        (nk, ndof)       kernel rows on 6-dof slots; nk = 3 periodic
#                              (translations), 4 free (+ in-plane rotation)
# solve / outputs
#   A         (naug+nk, naug+nk)  KKT matrix, naug = ndof + ne
#   V0        (ndof, 6)        fluctuation per unit macro strain (min-norm lsq)
#   Deff      (6, 6)           Dee + V0^T Dhe   (per unit axial length)
#   C3D       (6, 6)           Deff / cell_area, order = GBAR_ORDER
#   cell_area float            normalizing area (convex hull if not given)
# ----------------------------------------------------------------------------

shell_sg3d.py -- 3-D shell SG: equivalent solid properties of a shell-element
-----------------------------------------------------------------------------
structure gene (TPMS-class cells).  boundary="periodic" (default: all three
directions tied) or "aperiodic" (boundary solution w = 0 mapped onto the
bounding-box nodes; single-cell kinematic upper bound).

Reuses the msg_shell operators unchanged -- solid_fluct_ops_batch (Gamma_h),
solid_macro_ops_batch (Gamma_e) and the per-element frames are geometry-
general; what changes vs a cross-section run is only the environment:
  * general shell facets in 3-D, no prismatic strip.  3-node TRIANGLES are
    first-class NATIVE elements: solid_fluct_ops_tri_batch /
    solid_macro_ops_tri_batch on the area coordinates [L1,L2,L3], 18 DOF, the
    degree-2 3-point rule, and MITC3 assumed transverse shear (Lee & Bathe
    2004).  Quads and triangles are assembled as two SEPARATE batches --
    (ne4,24,24) and (ne3,18,18) -- scattered into one global system; a
    triangle is never padded to a quad.  The old collapsed-quad encoding
    [n1 n2 n3 n3] is REJECTED by check_element_edges: it reproduced the CST
    exactly, but its zero-length edge carried a MITC tying point and returned
    a silent NaN;
  * periodicity through the sparse assembly map on the FULL 3-D coordinates
    (all opposite faces, edges and corners -- the 3-D SG default);
  * sparse assembly (scipy) -- 3-D SGs are too large for dense Dhh;
  * drilling om3 by penalty on the SAME element-constant residual the
    cross-section route enforces with multipliers;
  * kernel = 3 translations (Lagrange border);
  * JUNCTION handling: shell edges shared by >2 elements are junction lines
    (a smooth TPMS has none -- every edge interior, count reported); when
    present they take the in-code hex-element micro treatment (3-D analog of
    the quad junction remesh) -- detection is wired, the hex micro follows
    the same dC = E_solid_mini - E_shell_mini construction.

Writes <yaml base>_C3D.out with the solve time (OpenSG default) AND
<yaml base>_ABDG.out -- the step-1 wall plate laws (8x8 Reissner-Mindlin
ABDG + compliance, one block per section), the same file and the same
emitter (write_abdg_out) the cross-section routes produce.

segment_taper.py -- Aperiodic-boundary shell segment homogenization (tapered or prismatic)
------------------------------------------------------------------------------------------
The msg-shell counterpart of OpenSG-FEniCSx's boundary/segment flow
(SolidBounMesh + compute_timo_boun -> compute_stiffness(l_submesh, r_submesh)):

  1. the segment's two END CROSS-SECTIONS are extracted topologically from the
     3-D shell yaml (free-edge components) and written as separate 1-D yamls
     (`sg_mesh.extract`);
  2. each boundary ring is solved on its own (`ring_indep`) -> the ring
     Timoshenko 6x6 and the warping fields V0, V1 (six DOFs per node,
     drilling omega_3 included);
  3. the ring fields are MAPPED onto the segment's boundary nodes (node2seg)
     and imposed as Dirichlet data replacing the axial periodicity;
  4. the segment Timoshenko 6x6 follows from the first-order (V1)
     finalization, with the energy divided by the segment length L.

For a PRISMATIC segment the result must reproduce the boundary ring's own
6x6 -- the seg==ring identity is the standing verification.

dehom_rm.py (build_rm_bundle, _strip) -- RM homogenization bundle
-----------------------------------------------------------------
build_rm_bundle homogenizes with the RM ring (ring_indep) and packages
everything the RM-consistent two-step dehomogenization (sg_dehom) needs;
_strip reconstructs the one-quad-deep prismatic strip mesh EXACTLY as
ring_indep does.
"""
import os, sys, json
import io
import contextlib
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import yaml
import yaml as _yaml

from .sg_assembly import (_surf_frame_batch, _shear_batch, NDOF6,
                          tri_frame_batch, mitc3_shear_batch, tri_scheme_for,
                          check_element_edges, TRI_GPTS, TIE_MITC3, TRI_NDOF)
from .sg_materials import _material_by_section
from .sg_periodicity import mesh_to_periodic_sparse_assembly_map
from .fe_jax.msg_hermite import solve_tw_from_yaml          # layup_db / material_db by name


# ===================== from run_ring_indep.py =====================

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
BENCH = os.path.join(REPO, "examples", "data", "benchmark")


def ring_indep(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross, h=None,
               shear="mitc4_g23", lam_space="elem", return_fields=False):
    """Constrained 6-DOF ring SG.  Returns the ring Timoshenko C6 (6,6); with
    return_fields=True also the zeroth/first-order warping fields V0, V1 (6m x 4,
    multiplier rows stripped) for the segment Dirichlet transfer -- including the
    drilling omega_3 boundary values.

    PRODUCTION shear scheme (ring): 'mitc4_g23' -- tie ONLY gamma_23.  Under span
    invariance gamma_13 carries no fluctuation gradient (it is algebraic in the
    directors), so only gamma_23 pairs a differentiated displacement with rotations."""
    import jax.numpy as jnp
    from .sg_assembly import (assemble_segment_indep, assemble_constraint, NDOF6)
    from .fe_jax.msg_rm_timo import build_C_Psi
    from .fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

    m = len(rx)
    if h is None:
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])
    dof_map = np.concatenate([np.arange(m), np.arange(m)])
    quads = np.array([[a, b, m + b, m + a] for a, b in rcells], dtype=int)
    e3q = np.asarray(re3)

    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, rsub, e3q, D_by, G_by, np.asarray(k22_edge), cross, ax,
        kg_e=None, pen=0.0, dof_map=dof_map, shear=shear)
    Gc, Gl, Ge = assemble_constraint(nodes, quads, rsub, e3q, np.asarray(k22_edge),
                                     cross, ax, dof_map=dof_map, lam_space=lam_space)
    Dhh, Dhe, Dhl, Dll, Dle = [np.asarray(A) / h for A in (Dhh, Dhe, Dhl, Dll, Dle)]
    Dee = np.asarray(Dee) / h
    Gc, Gl, Ge = Gc / h, Gl / h, Ge / h

    M = Dhh.shape[0]; P = Gc.shape[0]
    # 5-DOF rigid kernel/constraints on the contour, embedded into 6 DOF (om3 rigid-free)
    C5, Psi5 = build_C_Psi(rx[:, cross], rcells, p=1)          # (4,5m), (5m,4)
    C6 = np.zeros((4, M)); Psi6 = np.zeros((M, 4))
    for n in range(m):
        C6[:, 6 * n:6 * n + 5] = C5[:, 5 * n:5 * n + 5]
        Psi6[6 * n:6 * n + 5, :] = Psi5[5 * n:5 * n + 5, :]
    Psi6[3::6, 3] *= -1.0                                      # validated-kernel om1 sign flip

    naug = M + P
    # thin (naug, 4) blocks only -- the naug^2 zero-padded Dhh_a/Dhl_a/Dll_a of the
    # earlier dense route cost more in allocation/copies than the factorization itself
    # (the ring KKT is ~0.5-1 % dense).
    Dhe_a = np.vstack([Dhe, Ge])
    Dle_a = np.vstack([Dle, np.zeros((P, 4))])
    Psi_a = np.vstack([Psi6, np.zeros((P, 4))])
    Dc_a = np.vstack([C6.T, np.zeros((P, 4))])

    # dense KKT [[Dhh, Gc^T, C6^T], [Gc, 0, 0], [C6, 0, 0]] -- LAPACK-factorized once
    # for the V0 AND V1 solves, assembled DIRECTLY from the blocks (no naug^2 padded
    # intermediates).  Dense LU is kept deliberately: the saddle point carries a
    # near-null drilling/kernel mode, and a sparse factorization resolves it along a
    # different direction (~1e-3 of max in the null couplings) that would leak into
    # the V0/V1 recovery fields.
    from scipy.linalg import lu_factor, lu_solve
    A = np.zeros((naug + 4, naug + 4))
    A[:M, :M] = Dhh
    A[:M, M:naug] = Gc.T; A[M:naug, :M] = Gc
    A[:M, naug:] = C6.T; A[naug:, :M] = C6
    Alu = lu_factor(A)
    R0 = np.zeros((naug + 4, 4)); R0[:naug] = -Dhe_a
    V0 = lu_solve(Alu, R0)[:naug]
    Deff = Dee + V0.T @ Dhe_a
    # V1 RHS projected onto range(Dhh): the block form of msg_solver.prepare_v1_rhs
    # (Dhl_a/Dll_a act on the first M rows only; Gl couples the multiplier rows)
    V0m, V0p = V0[:M], V0[M:naug]
    DhlV0 = np.vstack([Dhl @ V0m, Gl @ V0m])
    DhlTV0Dle = np.vstack([Dhl.T @ V0m + Gl.T @ V0p, np.zeros((P, 4))]) + Dle_a
    V0DllV0 = V0m.T @ (Dll @ V0m)
    b_unproj = DhlV0 - DhlTV0Dle
    bb = Dc_a @ (np.linalg.inv(Psi_a.T @ Dc_a) @ (Psi_a.T @ b_unproj)) - b_unproj
    R1 = np.zeros((naug + 4, 4)); R1[:naug] = bb
    V1 = lu_solve(Alu, R1)[:naug]
    C6r, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(Deff),
        jnp.array(V0DllV0), jnp.array(DhlV0), jnp.array(DhlTV0Dle),
        jnp.array(Psi_a), jnp.array(Dc_a))
    C6r = np.asarray(C6r)
    C6r = 0.5 * (C6r + C6r.T)
    if return_fields:
        return C6r, np.asarray(V0[:M]), np.asarray(V1[:M])
    return C6r


# ===================== from solid_props.py =====================

GBAR_ORDER = ("strains [e11 e22 e33 2e23 2e13 2e12] -> stiffness C11..C66  "
              "(1=beam axis, 2/3=cross-section axes)")


def solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax):
    """Gamma_h for the SOLID route (standalone -- no Gamma_l, no beam macro).
    Returns BDh (ne,6,24), BGh (ne,2,24), DRh (ne,24), dA (ne,).
    DOF slots per node: 0:3 = w_i, 3:6 = om_i."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    ne = Xe.shape[0]
    xi1 = np.stack([c["x11"], c["x21"], c["x31"]], axis=1)     # X_{i1}
    xi2 = np.stack([c["x12"], c["x22"], c["x32"]], axis=1)     # X_{i2}
    yv = np.stack([c["y1"], c["y2"], c["y3"]], axis=1)         # C_{3i}

    B = np.zeros((ne, 6, 4, NDOF6))
    B[:, 0, :, 0:3] = D1[:, :, None] * xi1[:, None, :]                  # eps11 = X_i1 w_i,1
    B[:, 1, :, 0:3] = D2[:, :, None] * xi2[:, None, :]                  # eps22 = X_i2 w_i,2
    B[:, 2, :, 0:3] = (D2[:, :, None] * xi1[:, None, :]
                       + D1[:, :, None] * xi2[:, None, :])              # 2eps12 = X_i1 w_i,2 + X_i2 w_i,1
    B[:, 3, :, 3:6] = D1[:, :, None] * xi2[:, None, :]                  # K11 = X_i2 om_i,1
    B[:, 4, :, 3:6] = -D2[:, :, None] * xi1[:, None, :]                 # K22 = -X_i1 om_i,2
    B[:, 5, :, 3:6] = (D2[:, :, None] * xi2[:, None, :]
                       - D1[:, :, None] * xi1[:, None, :])              # K12+K21 = X_i2 om_i,2 - X_i1 om_i,1

    Bg = np.zeros((ne, 2, 4, NDOF6))
    Bg[:, 0, :, 0:3] = D1[:, :, None] * yv[:, None, :]                  # 2g13 = C_3i w_i,1 ...
    Bg[:, 0, :, 3:6] = N[None, :, None] * xi2[:, None, :]               # ... + X_i2 om_i
    Bg[:, 1, :, 0:3] = D2[:, :, None] * yv[:, None, :]                  # 2g23 = C_3i w_i,2 ...
    Bg[:, 1, :, 3:6] = N[None, :, None] * (-xi1)[:, None, :]            # ... - X_i1 om_i

    # DR = om3 + [ C31 om1 + C32 om2 - 1/2 (X_i2 w_i,1 - X_i1 w_i,2) ] / C33
    # (C33 -> 0: multiplied-through form kept)
    den = np.where(np.abs(yv[:, 2]) > 1e-8, yv[:, 2], 1.0)
    Dr = np.zeros((ne, 4, NDOF6))
    Dr[:, :, 0:3] = -0.5 * (D1[:, :, None] * xi2[:, None, :]
                            - D2[:, :, None] * xi1[:, None, :]) \
        / den[:, None, None]                    # -1/2 (X_i2 w_i,1 - X_i1 w_i,2)/C33
    Dr[:, :, 3:6] = (N[None, :, None] * yv[:, None, :]
                     / den[:, None, None])      # om3 + (C31 om1 + C32 om2)/C33
    return (B.reshape(ne, 6, 24), Bg.reshape(ne, 2, 24),
            Dr.reshape(ne, 24), dA)


def _tie_rows_solid(Xe, e3e, cross, ax):
    """Dvorkin-Bathe MITC tying rows for the solid route: only g23 is tied
    for a cross-section (g13 stays at the Gauss value).

    In:  Xe: (ne, 4, 3) float quad corner coordinates per element;
         e3e: (ne, 3) float wall normal per element;
         cross: (2,) int indices of the two cross-section axes;
         ax: int index of the axial direction.
    Out: dict {"g23m", "g23p"}: (ne, 1, 24) float g23 Gamma_h rows sampled
         at tying points (0, -1) and (0, +1)."""
    f = lambda xi, eta: solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)[1]
    return {"g23m": f(0.0, -1.0)[:, 1:2, :], "g23p": f(0.0, 1.0)[:, 1:2, :]}


def _tie_rows_solid4(Xe, e3e, cross, ax):
    """MITC4 tying rows for the SURFACE (3-D shell SG) route: BOTH covariant
    shear rows are tied, the standard Dvorkin-Bathe field.

    In:  Xe (ne,4,3), e3e (ne,3), cross (2,) int, ax int -- as _tie_rows_solid.
    Out: dict {"g13m","g13p","g23m","g23p"}, each (ne,1,24): row 0 at
         (-1,0)/(+1,0) and row 1 at (0,-1)/(0,+1).
    Unlike the cross-section ring (where 'mitc4_g23' ties only gamma_23 because
    gamma_13 is algebraic in the drilling rotation on a flat wall), a closed 3-D
    shell SG has no privileged axis, so both rows are assumed."""
    f = lambda xi, eta: solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)[1]
    return {"g13m": f(-1.0, 0.0)[:, 0:1, :], "g13p": f(1.0, 0.0)[:, 0:1, :],
            "g23m": f(0.0, -1.0)[:, 1:2, :], "g23p": f(0.0, 1.0)[:, 1:2, :]}


def solid_fluct_ops_tri_batch(Xe, e3e, r, s, cross, ax):
    """Gamma_h of the NATIVE 3-node triangle -- the tri3 twin of
    solid_fluct_ops_batch, same strain rows, same DOF layout, 18 columns.

    In:  Xe: (ne,3,3) float triangle corner coordinates;
         e3e: (ne,3) float wall normal per element;
         r, s: float parent coordinates on the unit triangle;
         cross: (2,) int cross-section axes; ax: int axial axis.
    Out: B (ne,6,18) rows [eps11 eps22 2eps12 K11 K22 2K12];
         Bg (ne,2,18) rows [2eps13 2eps23] (UNTIED, displacement-based);
         Dr (ne,18) drilling residual; dA (ne,) = |g_r x g_s| (multiply by the
         quadrature weight); Gm the covariant metric for the MITC3 tying.
    DOF slots per node: 0:3 = w_i, 3:6 = om_i, exactly as the quad."""
    N, D1, D2, dA, c, Gm = tri_frame_batch(Xe, e3e, r, s, cross, ax)
    ne = Xe.shape[0]
    xi1 = np.stack([c["x11"], c["x21"], c["x31"]], axis=1)     # X_{i1}
    xi2 = np.stack([c["x12"], c["x22"], c["x32"]], axis=1)     # X_{i2}
    yv = np.stack([c["y1"], c["y2"], c["y3"]], axis=1)         # C_{3i}

    B = np.zeros((ne, 6, 3, NDOF6))
    B[:, 0, :, 0:3] = D1[:, :, None] * xi1[:, None, :]                  # eps11 = X_i1 w_i,1
    B[:, 1, :, 0:3] = D2[:, :, None] * xi2[:, None, :]                  # eps22 = X_i2 w_i,2
    B[:, 2, :, 0:3] = (D2[:, :, None] * xi1[:, None, :]
                       + D1[:, :, None] * xi2[:, None, :])              # 2eps12
    B[:, 3, :, 3:6] = D1[:, :, None] * xi2[:, None, :]                  # K11 = X_i2 om_i,1
    B[:, 4, :, 3:6] = -D2[:, :, None] * xi1[:, None, :]                 # K22 = -X_i1 om_i,2
    B[:, 5, :, 3:6] = (D2[:, :, None] * xi2[:, None, :]
                       - D1[:, :, None] * xi1[:, None, :])              # K12+K21

    Bg = np.zeros((ne, 2, 3, NDOF6))
    Bg[:, 0, :, 0:3] = D1[:, :, None] * yv[:, None, :]                  # 2g13 = C_3i w_i,1 ...
    Bg[:, 0, :, 3:6] = N[None, :, None] * xi2[:, None, :]               # ... + X_i2 om_i
    Bg[:, 1, :, 0:3] = D2[:, :, None] * yv[:, None, :]                  # 2g23 = C_3i w_i,2 ...
    Bg[:, 1, :, 3:6] = N[None, :, None] * (-xi1)[:, None, :]            # ... - X_i1 om_i

    den = np.where(np.abs(yv[:, 2]) > 1e-8, yv[:, 2], 1.0)
    Dr = np.zeros((ne, 3, NDOF6))
    Dr[:, :, 0:3] = -0.5 * (D1[:, :, None] * xi2[:, None, :]
                            - D2[:, :, None] * xi1[:, None, :]) \
        / den[:, None, None]
    Dr[:, :, 3:6] = (N[None, :, None] * yv[:, None, :]
                     / den[:, None, None])
    return (B.reshape(ne, 6, TRI_NDOF), Bg.reshape(ne, 2, TRI_NDOF),
            Dr.reshape(ne, TRI_NDOF), dA, Gm)


def solid_macro_ops_tri_batch(Xe, e3e, r, s, cross, ax):
    """Gamma_e of the native triangle -- identical algebra to
    solid_macro_ops_batch, evaluated on the tri3 frame.

    In:  Xe (ne,3,3), e3e (ne,3), r, s float, cross (2,) int, ax int.
    Out: BDe6 (ne,6,6), BGe6 (ne,2,6), dA (ne,).
    The direction cosines are constant over an affine triangle, so these rows
    do not actually depend on (r, s)."""
    N, D1, D2, dA, c, Gm = tri_frame_batch(Xe, e3e, r, s, cross, ax)
    ne = Xe.shape[0]
    X11, X21, X31 = c["x11"], c["x21"], c["x31"]
    X12, X22, X32 = c["x12"], c["x22"], c["x32"]
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]
    z = np.zeros(ne)
    BDe6 = np.stack([
        np.stack([X11**2, X21**2, X31**2,
                  X31*X21, X11*X31, X11*X21], 1),
        np.stack([X12**2, X22**2, X32**2,
                  X32*X22, X32*X12, X12*X22], 1),
        np.stack([2*X11*X12, 2*X22*X21, 2*X31*X32,
                  X22*X31 + X32*X21, X12*X31 + X32*X11,
                  X12*X21 + X11*X22], 1),
        np.stack([z, z, z, z, z, z], 1),
        np.stack([z, z, z, z, z, z], 1),
        np.stack([z, z, z, z, z, z], 1),
    ], axis=1)
    BGe6 = np.stack([
        np.stack([C31*X11, C32*X21, C33*X31,
                  0.5*(X31*C32 + X21*C33), 0.5*(X31*C31 + X11*C33),
                  0.5*(X21*C31 + X11*C32)], 1),
        np.stack([C31*X12, C32*X22, C33*X32,
                  0.5*(X32*C32 + X22*C33), 0.5*(X32*C31 + X12*C33),
                  0.5*(X22*C31 + X12*C32)], 1),
    ], axis=1)
    return BDe6, BGe6, dA


def tri_shear_rows(Xe, e3e, cross, ax, r, s, Bg_gauss, Gm, scheme):
    """Transverse-shear rows of the tri3 element at (r, s) under `scheme`.

    In:  Xe (ne,3,3), e3e (ne,3), cross (2,) int, ax int;
         r, s: float parent coordinates;
         Bg_gauss: (ne,2,18) untied rows already evaluated at (r, s);
         Gm: covariant metric from tri_frame_batch;
         scheme: 'full' (displacement-based) or 'mitc3' (assumed field).
    Out: (ne,2,18) shear rows.
    Convenience wrapper; the production loop hoists the three tying-point
    evaluations out of the quadrature loop instead (they are (r,s)-independent)."""
    if scheme == "full":
        return Bg_gauss
    g = [solid_fluct_ops_tri_batch(Xe, e3e, tr, ts, cross, ax)[1]
         for (tr, ts) in TIE_MITC3]
    return mitc3_shear_batch(Gm, g[0], g[1], g[2], r, s)


def solid_macro_enn_batch(Xe, e3e, xi, eta, cross, ax):
    """The e_nn completion row of Gamma_e (through-thickness normal strain):
        e_nn = C_3i G_ij C_3j
    -> row [C31^2, C32^2, C33^2, C32*C33, C31*C33, C31*C32]  (ne, 6).
    Its Gamma_h partner is zero on a midline (no thickness-stretch DOF)."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]
    return np.stack([C31**2, C32**2, C33**2,
                     C32*C33, C31*C33, C31*C32], 1), dA


def solid_macro_ops_batch(Xe, e3e, xi, eta, cross, ax):
    """The document's Gamma_e at (xi,eta): BDe6 (ne,6,6) on the membrane+curvature
    rows [e11,e22,2e12,K11,K22,K12+K21], BGe6 (ne,2,6) on the shear rows
    [2g13,2g23]; dA (ne,).  Entries verbatim from Eq. (17) of the document."""
    N, D1, D2, dA, c = _surf_frame_batch(Xe, e3e, xi, eta, cross, ax)
    ne = Xe.shape[0]
    X11, X21, X31 = c["x11"], c["x21"], c["x31"]        # X_{i1}
    X12, X22, X32 = c["x12"], c["x22"], c["x32"]        # X_{i2}
    C31, C32, C33 = c["y1"], c["y2"], c["y3"]           # C_{3i}
    z = np.zeros(ne)

    # Gamma_e = Gamma_h on  w_i -> G_ij y_j ,  om_i -> 0
    # columns: [ G11  G22  G33  2G23  2G13  2G12 ]
    BDe6 = np.stack([
        # eps11 = X_i1 G_ij X_j1
        np.stack([X11**2, X21**2, X31**2,
                  X31*X21, X11*X31, X11*X21], 1),
        # eps22 = X_i2 G_ij X_j2
        np.stack([X12**2, X22**2, X32**2,
                  X32*X22, X32*X12, X12*X22], 1),
        # 2eps12 = X_i1 G_ij X_j2 + X_i2 G_ij X_j1
        np.stack([2*X11*X12, 2*X22*X21, 2*X31*X32,
                  X22*X31 + X32*X21, X12*X31 + X32*X11,
                  X12*X21 + X11*X22], 1),
        np.stack([z, z, z, z, z, z], 1),                # K11     (om -> 0)
        np.stack([z, z, z, z, z, z], 1),                # K22     (om -> 0)
        np.stack([z, z, z, z, z, z], 1),                # K12+K21 (om -> 0)
    ], axis=1)
    # G_ij is the TENSOR macro strain: off-diagonals = engineering/2, so the
    # engineering-shear columns carry HALF the pair-sum.  This makes the rows
    # literally Gamma_h on (w_i -> G_ij y_j, om_i -> 0); the om-embedding that
    # would restore full pair-sums is inadmissible (its fiber tilt jumps
    # between wall families at junctions while nodal om is single-valued).
    BGe6 = np.stack([
        # 2g13 = C_3i G_ij X_j1
        np.stack([C31*X11, C32*X21, C33*X31,
                  0.5*(X31*C32 + X21*C33), 0.5*(X31*C31 + X11*C33),
                  0.5*(X21*C31 + X11*C32)], 1),
        # 2g23 = C_3i G_ij X_j2
        np.stack([C31*X12, C32*X22, C33*X32,
                  0.5*(X32*C32 + X22*C33), 0.5*(X32*C31 + X12*C33),
                  0.5*(X22*C31 + X12*C32)], 1),
    ], axis=1)
    return BDe6, BGe6, dA


def _C_from_eng(E, G, nu):
    """6x6 stiffness from engineering constants; Voigt [11,22,33,23,13,12],
    G = [G12,G13,G23], nu = [nu12,nu13,nu23]."""
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1/E[0], 1/E[1], 1/E[2]
    S[0, 1] = S[1, 0] = -nu[0]/E[0]
    S[0, 2] = S[2, 0] = -nu[1]/E[0]
    S[1, 2] = S[2, 1] = -nu[2]/E[1]
    S[3, 3], S[4, 4], S[5, 5] = 1/G[2], 1/G[1], 1/G[0]
    return np.linalg.inv(S)


def _rot_inplane(C, th_deg):
    """Rotate a 6x6 Voigt stiffness by th about axis 3 (ply angle about the
    wall normal, 1 -> 2 ccw)."""
    th = np.radians(th_deg)
    c, s = np.cos(th), np.sin(th)
    cs = c*s
    R = np.array([[c**2, s**2, 0, 0, 0, 2*cs],
                  [s**2, c**2, 0, 0, 0, -2*cs],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, c, -s, 0],
                  [0, 0, 0, s, c, 0],
                  [-cs, cs, 0, 0, 0, c**2 - s**2]])
    return R @ C @ R.T


def wall_solid_law(sections, materials):
    """Per-section through-thickness-integrated UN-CONDENSED wall law
    sum_k t_k * C_ply(theta_k), wall Voigt [11,22,nn,2n,1n,12].

    In:  sections: list of yaml section dicts, each with "layup" =
         [[material_name, thickness, angle_deg], ...];
         materials: list of yaml material dicts with E/G/nu (possibly under
         an "elastic" key).
    Out: list of (6, 6) float, one t*C per section.
    This is the law the e_nn completion contracts with."""
    mats = {str(m["name"]): m for m in materials}
    out = []
    for sec in sections:
        C = np.zeros((6, 6))
        for p in sec["layup"]:
            m = mats[str(p[0])]
            el = m["elastic"] if "elastic" in m else m
            Cp = _C_from_eng([float(v) for v in el["E"]],
                             [float(v) for v in el["G"]],
                             [float(v) for v in el["nu"]])
            C += float(p[1]) * _rot_inplane(Cp, float(p[2]))
        out.append(C)
    return out


_VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _voigt_rotate(C, Q):
    """C' = Q C Q^T on the 4th-order tensor; C engineering Voigt
    [11,22,33,23,13,12] (stiffness entries equal tensor components)."""
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_IDX):
        for J, (k, l) in enumerate(_VOIGT_IDX):
            C4[i, j, k, l] = C4[j, i, k, l] = C4[i, j, l, k] \
                = C4[j, i, l, k] = C[I, J]
    C4r = np.einsum('ai,bj,ck,dl,ijkl->abcd', Q, Q, Q, Q, C4)
    Cr = np.zeros((6, 6))
    for I, (i, j) in enumerate(_VOIGT_IDX):
        for J, (k, l) in enumerate(_VOIGT_IDX):
            Cr[I, J] = C4r[i, j, k, l]
    return Cr


def junction_inventory(rx, cells, rsub, cross, ang_tol_deg=1.0):
    """Junction nodes of the midline mesh: a node shared by walls with
    distinct tangents.  Per junction: walls = [(section, tangent2d, weight)],
    weight = 1 if the wall runs THROUGH the node (elements on both sides,
    counted overlap length t_other), 1/2 if it ENDS there (counted t_other/2)."""
    ctol = np.cos(np.radians(ang_tol_deg))
    adj = {}
    for e, (n1, n2) in enumerate(cells):
        adj.setdefault(int(n1), []).append(e)
        adj.setdefault(int(n2), []).append(e)
    out = []
    for nd, elist in adj.items():
        if len(elist) < 2:
            continue
        groups = []
        for e in elist:
            n1, n2 = cells[e]
            other = int(n2) if int(n1) == nd else int(n1)
            v = np.asarray(rx[other])[cross] - np.asarray(rx[nd])[cross]
            d = v/np.linalg.norm(v)
            side = 1
            if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0):
                d, side = -d, -1
            for g in groups:
                if float(np.dot(g["dir"], d)) > ctol:
                    g["sides"].add(side)
                    g["secs"].add(int(rsub[e]))
                    break
            else:
                groups.append({"dir": d, "sides": {side},
                               "secs": {int(rsub[e])}})
        if len(groups) < 2:
            continue
        walls = [(sorted(g["secs"])[0], g["dir"],
                  1.0 if len(g["sides"]) == 2 else 0.5) for g in groups]
        out.append({"node": int(nd), "walls": walls})
    return out


def _wall_normal_integrand(d2, ABD, G):
    """Per-unit-length Dee integrand of a wall, NORMAL (1..3) block, section
    frame.  Nonzero Gamma_e rows on the normal columns for a cross-section
    wall (tangent (0,c,s), normal (0,-s,c)):
        e11 -> [1, 0, 0]          x A11
        e22 -> [0, c^2, s^2]      x A22 (+A12 cross)
        2g23 -> [0, -sc, cs]      x G22   (diagonal columns, unhalved)"""
    c, s = float(d2[0]), float(d2[1])
    r1 = np.array([1.0, 0.0, 0.0])
    r2 = np.array([0.0, c*c, s*s])
    r3 = np.array([0.0, -s*c, c*s])
    A11, A12, A22 = ABD[0, 0], ABD[0, 1], ABD[1, 1]
    W = (A11*np.outer(r1, r1) + A12*(np.outer(r1, r2) + np.outer(r2, r1))
         + A22*np.outer(r2, r2) + G[1, 1]*np.outer(r3, r3))
    return W


def junction_census_correction(inv, Cw_by, t_by, D_by, G_by):
    """Junction census correction to Dee: on each wall-overlap block
    A_j = t_A t_B / sin(theta) swap the walls' condensed law for the full
    un-condensed 3-D law,
        dDee = A_j C_j  -  sum_i w_i (t_other/sin) W_i,     normal block only,
    C_j = mean of the two walls' per-volume un-condensed laws rotated to the
    section frame (Q columns: axial, tangent, normal).

    In:  inv: junction_inventory output, [{"node": int, "walls":
         [(section, tangent2d, weight)]}];
         Cw_by: list of (6, 6) float wall t*C per section (wall_solid_law);
         t_by: list of float total wall thickness per section;
         D_by: list of (6, 6) float wall ABD per section;
         G_by: list of (2, 2) float wall transverse-shear law per section.
    Out: dD (6, 6) float additive Deff correction; only the normal (1..3)
         block is nonzero."""
    dD = np.zeros((6, 6))
    for J in inv:
        ws = J["walls"]
        for iA in range(len(ws)):
            for iB in range(iA + 1, len(ws)):
                (sA, dA, wA), (sB, dB, wB) = ws[iA], ws[iB]
                tA, tB = t_by[sA], t_by[sB]
                sinth = abs(dA[0]*dB[1] - dA[1]*dB[0])
                if sinth < 1e-6:
                    continue
                Aj = tA*tB/sinth
                Cj = np.zeros((6, 6))
                for sec, d2, tk in ((sA, dA, tA), (sB, dB, tB)):
                    c, s = float(d2[0]), float(d2[1])
                    Q = np.array([[1.0, 0.0, 0.0],
                                  [0.0, c, -s],
                                  [0.0, s, c]])
                    Cj += 0.5*_voigt_rotate(Cw_by[sec]/tk, Q)
                blk = Aj*Cj[:3, :3] \
                    - wA*(tB/sinth)*_wall_normal_integrand(dA, D_by[sA],
                                                           G_by[sA]) \
                    - wB*(tA/sinth)*_wall_normal_integrand(dB, D_by[sB],
                                                           G_by[sB])
                dD[:3, :3] += blk
    return dD


def junction_inventory_merged(rx, cells, rsub, re3, cross, master):
    """Junctions of the PERIODICALLY REDUCED midline mesh: lattice-coincident
    nodes merge (a ring's four L-corners become ONE X-crossing of the merged
    walls), and coincident periodic partner walls appear as stacked members
    [(sec, mirrored)] of one wall family, ordered bottom (-n) to top;
    mirrored = the partner's frame is the 180-deg-about-axial image
    (e3 . n_canonical < 0), i.e. its ply angles negate in the merged frame."""
    rx = np.asarray(rx, float)
    cells = np.asarray(cells, int)
    adj = {}
    for e, (n1, n2) in enumerate(cells):
        adj.setdefault(int(master[n1]), []).append((e, int(n1)))
        adj.setdefault(int(master[n2]), []).append((e, int(n2)))
    ctol = np.cos(np.radians(1.0))
    out = []
    for mnode, elist in adj.items():
        if len(elist) < 2:
            continue
        groups = []
        for e, nd in elist:
            n1, n2 = cells[e]
            other = int(n2) if int(n1) == nd else int(n1)
            v = rx[other][cross] - rx[nd][cross]
            dcan = v/np.linalg.norm(v)
            side = 1
            if dcan[0] < -1e-9 or (abs(dcan[0]) < 1e-9 and dcan[1] < 0):
                dcan, side = -dcan, -1
            ncan = np.array([-dcan[1], dcan[0]])
            dn = float(np.dot(np.asarray(re3[e], float)[cross], ncan))
            for g in groups:
                if abs(float(np.dot(g["dir"], dcan))) > ctol:
                    g["mem"].append((side, int(rsub[e]), dn))
                    # a member node with this family on BOTH sides = the wall
                    # runs THROUGH its own node (owns the junction block)
                    if (nd, -side) in g["ndside"]:
                        g["thru"] = True
                    g["ndside"].add((nd, side))
                    break
            else:
                groups.append({"dir": dcan,
                               "mem": [(side, int(rsub[e]), dn)],
                               "ndside": {(nd, side)}, "thru": False})
        if len(groups) < 2:
            continue
        walls = []
        for g in groups:
            sides = {m[0] for m in g["mem"]}
            w = 1.0 if len(sides) == 2 else 0.5
            sd = 1 if 1 in sides else -1
            stack = tuple((sec, dn < 0) for (s, sec, dn) in
                          sorted([m for m in g["mem"] if m[0] == sd],
                                 key=lambda m: -m[2]))
            walls.append({"dir": g["dir"], "weight": w, "stack": stack,
                          "thru": bool(g.get("thru", False))})
        out.append({"node": int(mnode), "walls": walls})
    return out


def assemble_solid_macro(nodes, quads, subdom, e3s, D_by, G_by, cross, ax,
                         dof_map=None, shear="mitc4_g23", Cw_by=None):
    """Dhe6 (ndof,6), Dee6 (6,6): the solid-macro blocks, mirroring
    assemble_segment_indep's quadrature, wall-law lookup, tied-shear scheme and
    dof_map.  The drilling residual has NO solid-strain column, so no multiplier or
    penalty cross terms arise here.

    Cw_by (list of 6x6 wall t*C per section, from wall_solid_law) activates the
    e_nn COMPLETION in Dee: the membrane normal block [e11,e22,2e12] switches
    from the plane-stress A to the un-condensed t*C block on
    [e11,e22,e_nn,2e12], with the e_nn macro row C_3i G_ij C_3j.  Dee's normal
    block then equals A_mat*C exactly (the solid's).  Dhe is unchanged (the
    e_nn row has no midline fluctuation partner).  Cw_by=None reproduces the
    previous behaviour identically."""
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
    tie = None if shear == "full" else _tie_rows_solid(Xe, e3e, cross, ax)

    if Cw_by is not None:
        Cw_arr = np.stack([np.asarray(Cw_by[k], float) for k in keys])
        # wall-law sub-block on [e11, e22, e_nn, 2e12] (wall Voigt 0,1,2,5)
        Cn4e = Cw_arr[sdi][:, [0, 1, 2, 5]][:, :, [0, 1, 2, 5]]
        A3e = De[:, 0:3, 0:3]                  # plane-stress block it replaces

    Ehe = np.zeros((ne, 24, 6)); Dee = np.zeros((6, 6))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        BDh, BGh, _, dA = solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)
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
        if Cw_by is not None:
            # e_nn COMPLETION of Dee: swap the membrane normal block
            # [e11,e22,2e12]xA  ->  [e11,e22,e_nn,2e12] x (t*C)
            Benn, _ = solid_macro_enn_batch(Xe, e3e, xi, eta, cross, ax)
            Bn4 = np.stack([BDe6[:, 0, :], BDe6[:, 1, :],
                            Benn, BDe6[:, 2, :]], axis=1)      # (ne,4,6)
            Bm3 = BDe6[:, 0:3, :]                              # (ne,3,6)
            Dee += np.einsum('e,eab->ab', dA,
                             np.einsum('eia,eij,ejb->eab', Bn4, Cn4e, Bn4)
                             - np.einsum('eia,eij,ejb->eab', Bm3, A3e, Bm3))
    Dhe = np.zeros((ndof, 6))
    np.add.at(Dhe, g.reshape(-1), Ehe.reshape(-1, 6))
    return Dhe, Dee


def assemble_solid_strip(nodes, quads, subdom, e3s, D_by, G_by, cross, ax,
                         dof_map=None, shear="mitc4_g23"):
    """Dhh (ndof,ndof) and Gc (ne,ndof) for the SOLID route -- standalone.

    Carries only what the zeroth-order solid theory needs: Gamma_h, the
    element-constant drilling rows, and dA.  No Gamma_l / V1 ladder, no beam
    macro columns, no drilling penalty (om3 is enforced exactly by the
    multipliers, not by a penalty).

    Gc row per element e:  int_e DR dA = 0,  with
        DR = C_3i om_i - 1/2 (X_i2 w_i,1 - X_i1 w_i,2)
    i.e. the drilling residual is written on the SOLID-side om3 through C_3i,
    with NO macro-strain column (the macro contributions cancel identically)."""
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
    tie = None if shear == "full" else _tie_rows_solid(Xe, e3e, cross, ax)

    Ehh = np.zeros((ne, 24, 24)); Gel = np.zeros((ne, 24))
    gpv = 1.0 / np.sqrt(3.0)
    for (xi, eta) in [(-gpv, -gpv), (gpv, -gpv), (gpv, gpv), (-gpv, gpv)]:
        BDh, BGh, DRh, dA = solid_fluct_ops_batch(Xe, e3e, xi, eta, cross, ax)
        BGt = BGh if shear == "full" else _shear_batch(xi, eta, shear, BGh, tie)
        w = dA[:, None, None]
        DB = np.einsum('eij,ejb->eib', De, BDh)
        GB = np.einsum('eij,ejb->eib', Gm, BGt)
        Ehh += w * (np.einsum('eia,eib->eab', BDh, DB)
                    + np.einsum('eia,eib->eab', BGt, GB))
        Gel += DRh * dA[:, None]

    Dhh = np.zeros((ndof, ndof))
    np.add.at(Dhh, (g[:, :, None], g[:, None, :]), Ehh)
    Gc = np.zeros((ne, ndof))
    np.add.at(Gc, (np.arange(ne)[:, None], g), Gel)
    return Dhh, Gc


def ring_solid(rx, rcells, rsub, re3, D_by, G_by, k22_edge, ax, cross, h=None,
               shear="mitc4_g23", lam_space="elem", return_fields=False,
               periodic=True, Cw_by=None, Dhh_extra=None):
    """Equivalent-solid homogenization of the ring SG.  Returns D_eff (6,6), the
    contour-integrated stiffness per unit axial length on GBAR_ORDER; with
    return_fields=True also the 6-column warping V0.

    Mirrors ring_indep: one-quad-deep prismatic strip, top row DOF-mapped onto the
    bottom, element-constant drilling Lagrange multipliers (zero macro column),
    rigid-kernel KKT, single V0 solve, D_eff = Dee + V0^T Dhe.  No V1.

    periodic=True (THE DEFAULT -- periodicity is always part of the msg_shell
    solid-property route) ties opposite bounding-box faces of the SG through
    `periodic_multiscale.mesh_to_periodic_sparse_assembly_map`, the shell mirror
    of the solid side's map: for a 2-D SG both in-plane directions are tied, for
    a 3-D SG all three, and the repeated dof_map[dof_map] resolves the edge and
    corner chains, so opposite faces, edges AND corners are all periodic.  The
    tie rides in the assembly map (element connectivity re-pointed at master
    nodes), so no constraint rows are added, and the kernel drops the in-plane
    rotation, keeping the 3 translations.

    periodic=False leaves the SG FREE, whose equivalent-solid stiffness is rank
    one by construction: every macro strain except Gamma_11 is cancelled at zero
    energy by an affine fluctuation, which only periodicity forbids.  Kept for
    diagnostics only."""
    from .fe_jax.msg_rm_timo import build_C_Psi
    from scipy.linalg import lu_factor, lu_solve

    m = len(rx)
    if h is None:
        h = float(np.mean(np.linalg.norm(rx[rcells[:, 1]] - rx[rcells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx, rx + h * ez])
    if periodic:
        from .sg_periodicity import mesh_to_periodic_sparse_assembly_map
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

    Dhh, Gc = assemble_solid_strip(nodes, quads, rsub, e3q, D_by, G_by,
                                   cross, ax, dof_map=dof_map, shear=shear)
    Dhe6, Dee6 = assemble_solid_macro(nodes, quads, rsub, e3q, D_by, G_by,
                                      cross, ax, dof_map=dof_map, shear=shear,
                                      Cw_by=Cw_by)
    Dhh = np.asarray(Dhh) / h; Gc = Gc / h
    Dhe6 = Dhe6 / h; Dee6 = Dee6 / h
    if Dhh_extra is not None:
        # caller-supplied fluctuation stiffening (e.g. rigid junction blocks);
        # acts on w only, so Dee/Dhe and every drive channel are untouched
        Dhh = Dhh + Dhh_extra

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
    # lstsq min-norm is load-bearing: the drilling null-space makes an LU
    # solve of the rank-deficient KKT unreliable (do NOT swap for a direct solve).
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


def write_abdg_out(out_path, sections, D_by, G_by):
    """Write each section's wall plate law in the SwiftComp .K layout: the
    8x8 Reissner-Mindlin ABDG = [[A B 0; B^T D 0; 0 0 G]] stiffness and its
    compliance per section, no engineering constants.

    In:  out_path str; sections list of yaml section dicts; D_by, G_by
         per-section ABD (6, 6) and transverse-shear G (2, 2)
    Out: the .out file at out_path; returns out_path (str)."""
    from opensg_solid.sg_homo import _knum
    with open(out_path, "w") as f:
        f.write(" OpenSG msg-shell wall plate laws, one block per section\n"
                " rows [eps11 eps22 2eps12 | K11 K22 K12+K21 | 2g13 2g23]\n")
        for si, sec in enumerate(sections):
            name = sec.get("elementSet", "section_%d" % si)
            lay = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
            ABDG = np.zeros((8, 8))
            ABDG[:6, :6] = np.asarray(D_by[si], float)
            ABDG[6:, 6:] = np.asarray(G_by[si], float)
            try:
                S = np.linalg.inv(ABDG)
            except np.linalg.LinAlgError:               # degenerate wall law: report, don't abort
                S = np.linalg.pinv(ABDG)
            f.write("\n section %d: %s  layup %s\n" % (si, name, lay))
            f.write(" The Effective Reissner-Mindlin Plate Stiffness Matrix\n"
                    " --------------------------------------------\n")
            for i in range(8):
                f.write("".join(_knum(ABDG[i, j]) for j in range(8)) + "\n")
            f.write("\n The Effective Reissner-Mindlin Plate Compliance"
                    " Matrix\n"
                    " --------------------------------------------\n")
            for i in range(8):
                f.write("".join(_knum(S[i, j]) for j in range(8)) + "\n")
    return out_path


def build_solid_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source=None,
                       cell_area=None, periodic=True, junction=None):
    """Load the shell yaml exactly as build_rm_bundle does (same reference logic,
    same MSG wall transverse-shear upgrade), run ring_solid, and package:

        {"C3D", "D_eff", "cell_area", "area_source", "V0", geometry..., "order"}

    C3D = D_eff / cell_area.  cell_area=None -> convex-hull area of the contour.
    ``g_source`` is accepted and ignored (MSG is the only wall-G route) --
    see sg_materials.check_g_source."""
    import time as _time
    import yaml as _yaml
    from .sg_mesh import load_ring_ref
    from .sg_materials import check_g_source

    check_g_source(g_source, "build_solid_bundle")
    _t0 = _time.perf_counter()
    d = _yaml.safe_load(open(shell_yaml))
    if ref is None:
        ref = d.get("reference", "center")
    R = load_ring_ref(shell_yaml, ref)
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    G_by = list(R["G_by"])
    # wall transverse-shear G: the MSG (Yu-2002 LS) construction, the only route
    from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
    from .sg_materials import material_db_from_yaml
    _mdb = material_db_from_yaml(d["materials"])
    for si, sec in enumerate(d["sections"]):
        _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
        _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl],
                           [p[0] for p in _pl], _mdb, fraction=frac)
        if _rr["G_msg"] is not None:
            G_by[si] = np.asarray(_rr["G_msg"])
    import os as _os
    write_abdg_out(_os.path.splitext(shell_yaml)[0] + "_ABDG.out",
                   d["sections"], R["D_by"], G_by)
    # e_nn channel NOT included: the condensed ABD already carries the
    # pointwise thickness relaxation; freezing e_nn in Gamma_e without a
    # Gamma_h partner double-counts that energy.
    Deff, V0 = ring_solid(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], G_by,
                          R["k22"], R["ax"], R["cross"], shear=shear,
                          lam_space="elem", return_fields=True, periodic=periodic)
    jinfo = None
    if junction == "census":
        # sigma_nn is blocked on the t_A x t_B wall-overlap blocks: swap the
        # condensed wall law for the full 3-D law there (normal block only;
        # junction shear relaxes with the joint and is left to the walls)
        t_by = [sum(float(p[1]) for p in sec["layup"]) for sec in d["sections"]]
        Cw_by = wall_solid_law(d["sections"], d["materials"])
        inv = junction_inventory(R["rx"], R["cells"], R["rsub"], R["cross"])
        dD = junction_census_correction(inv, Cw_by, t_by, R["D_by"], G_by)
        Deff = Deff + dD
        jinfo = {"n_junctions": len(inv), "inventory": inv, "dDee": dD}
    elif junction in ("micro", "microcell"):
        # level-2: per-junction-type local 2-D corner micro-solve (general
        # laminate, dC = E_solid_patch - E_shell_patch, mirror-line lattice
        # environment).  Inventory on the PERIODICALLY REDUCED mesh: a ring's
        # four L-corners merge into one X-crossing of the stacked 2t walls
        # (the mirrored partner's plies negate: m45 -> antisymmetric [+-45]).
        from .sg_junction import corner_micro_law
        from .sg_periodicity import mesh_to_periodic_sparse_assembly_map
        rxJ = np.asarray(R["rx"])
        mJ = len(rxJ)
        if periodic:
            rc, _ = mesh_to_periodic_sparse_assembly_map(
                mJ, np.arange(mJ)[:, None], rxJ[:, R["cross"]], 3, NDOF6)
            masterJ = np.asarray(rc, int).ravel()
        else:
            masterJ = np.arange(mJ)
        inv = junction_inventory_merged(rxJ, R["cells"], R["rsub"], R["re3"],
                                        R["cross"], masterJ)
        dD = np.zeros((6, 6))
        cache = {}
        for J in inv:
            ws = sorted(J["walls"],
                        key=lambda w: (-w["weight"], -float(w["thru"])))
            if len(ws) != 2:
                raise NotImplementedError(">2 wall families at a junction")
            A, B = ws
            if abs(float(np.dot(A["dir"], B["dir"]))) > 0.05:
                raise NotImplementedError("non-orthogonal junction")
            topo = {(1.0, 1.0): "X", (1.0, 0.5): "T",
                    (0.5, 0.5): "L"}[(A["weight"], B["weight"])]
            # block fill: the family running THROUGH its own member node owns
            # the block (I-beam flange); merged L-corners (neither) -> mitre4
            fill = "A" if (A["thru"] or topo != "X") else \
                   ("mitre4" if not B["thru"] else "A")
            key = (topo, A["stack"], B["stack"], fill, junction)
            if key not in cache:
                if junction == "microcell" and topo == "X":
                    from .sg_junction import microcell_law
                    cache[key] = microcell_law(list(A["stack"]),
                                               list(B["stack"]),
                                               d["sections"], d["materials"],
                                               R["D_by"], G_by, fill=fill)
                else:
                    cache[key] = corner_micro_law(topo, list(A["stack"]),
                                                  list(B["stack"]),
                                                  d["sections"],
                                                  d["materials"],
                                                  fill=fill)
            dCloc, jinf = cache[key]
            c, s = float(A["dir"][0]), float(A["dir"][1])
            Q = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            d6 = np.zeros((6, 6))
            d6[:3, :3] = dCloc
            dD += _voigt_rotate(d6, Q)
            # scaled D44 (in-plane shear) merge-census correction: both mini
            # D44s are wall-bending dominated (~1/span), so the mini
            # difference transfers by Lm/span when the two cross spans agree
            if junction == "microcell" and "Lm" in jinf:
                ext = rxJ[:, R["cross"]].max(0) - rxJ[:, R["cross"]].min(0)
                if abs(ext[0] - ext[1]) < 0.05*max(ext):
                    dD[3, 3] += (jinf["D_solid_mini"][3, 3]
                                 - jinf["D_shell_mini"][3, 3]) \
                        * jinf["Lm"]/float(ext[0])
        Deff = Deff + dD
        jinfo = {"n_junctions": len(inv), "inventory": inv, "dDee": dD,
                 "types": sorted(cache.keys())}
    area_source = "user"
    if cell_area is None:
        from scipy.spatial import ConvexHull
        cell_area = float(ConvexHull(R["rx"][:, R["cross"]]).volume)
        area_source = "hull"
    solve_time = _time.perf_counter() - _t0
    # every OpenSG run writes its timed .out by default
    _C = Deff / float(cell_area)
    from opensg_solid.sg_homo import write_sc_K
    write_sc_K(_os.path.splitext(shell_yaml)[0] + "_C3D.out", _C,
               solve_time=solve_time,
               model="msg-shell equivalent 3D solid (cross-section SG),"
                     " omega %.8g (%s)" % (float(cell_area), area_source),
               name="Cauchy Continuum")
    return {"C3D": Deff / float(cell_area), "D_eff": Deff,
            "solve_time": solve_time,
            "cell_area": float(cell_area), "area_source": area_source,
            "junction": jinfo, "V0": V0, "rx3": np.asarray(R["rx"]),
            "red_cells": np.asarray(R["cells"]), "rsub": np.asarray(R["rsub"]),
            "re3": np.asarray(R["re3"]), "k22": np.asarray(R["k22"]),
            "ax": int(R["ax"]), "cross": list(R["cross"]), "ref": ref,
            "g_source": "msg", "order": GBAR_ORDER}


# ===================== from shell_sg3d.py =====================

_G = 1.0/np.sqrt(3.0)
_GPTS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]
_ONE3 = 1.0/3.0                      # triangle centroid, for the nodal area weights


def _sparse_solve(A, R, sym=False):
    """Multi-RHS sparse direct solve: PARDISO (multithreaded MKL) when available,
    SuperLU fallback.  SuperLU is single-threaded CPU; PARDISO uses all cores and
    is the same direct solver opensg_solid defaults to.  (The device-resident/GPU
    alternative in this stack is the matrix-free EBE Chebyshev-CG of fe_jax, as in
    opensg_solid.plate_homo_2d(solver="cg") -- iterative, no factorization at all.)

    sym=True: PARDISO real-symmetric-indefinite (mtype=-2) on the upper triangle
    (~25-40 % faster factor; the KKT multiplier block gets its explicit zero
    diagonal), guarded by a 1e-8 relative-residual check with fallback.

    In:
        A: (n, n) scipy sparse matrix (any format).
        R: (n, k) dense right-hand sides.
        sym: bool, the matrix is symmetric (periodic KKT / pinned stiffness).
    Out:
        (n, k) solution array.
    """
    R = np.asarray(R, float)
    if sym:
        try:
            from pypardiso import PyPardisoSolver
            Ac = sp.csr_matrix(A)
            Au = sp.triu(Ac + sp.diags(np.zeros(Ac.shape[0])), format="csr")
            slv = PyPardisoSolver(mtype=-2)
            X = np.asarray(slv.solve(Au, R)).reshape(R.shape)
            slv.free_memory(everything=True)
            bn = np.linalg.norm(R)
            if bn == 0.0 or np.linalg.norm(Ac @ X - R) <= 1e-8 * bn:
                return X
        except Exception:
            pass                                 # any failure -> general path
    try:
        import pypardiso
        X = pypardiso.spsolve(sp.csr_matrix(A), R)
        return X.reshape(R.shape)
    except ImportError:
        return spla.splu(sp.csc_matrix(A)).solve(R)


def shell_sg3d(yaml_path, omega=None, drill_pen=1.0e-3, g_source=None,
               solver="direct",
               boundary=None, shear="mitc"):
    """Equivalent 3-D solid stiffness of a 3-D shell SG.

    MIXED MESH.  The surface may hold 3-node triangles and 4-node quads at once.
    They are assembled as two SEPARATE BATCHES -- (ne3,18,18) tri3 blocks and
    (ne4,24,24) quad blocks -- scattered into the same global system; a triangle
    is NEVER padded to a quad.  The old collapsed-quad triangle [n1 n2 n3 n3] is
    rejected by check_element_edges (it made the MITC tying point on the
    zero-length edge singular and returned a silent NaN).

    shear   : transverse-shear treatment, the shared vocabulary.
      * "mitc" (default) -- assumed transverse shear: MITC4 (Dvorkin-Bathe, both
        covariant rows tied) on a quad, MITC3 (Lee & Bathe 2004, the three
        edge-midpoint ties with the coupling correction) on a triangle.  A
        triangle gets MITC3 wherever a quad gets MITC4; see tri_scheme_for.
      * "full" -- displacement-based shear at the quadrature points, the legacy
        (locking-prone) treatment this route used before the tri3 element
        existed.  Kept for ablation and for reproducing older numbers.

    boundary = "periodic" (the default) or "aperiodic".  Left as None it is
    read from the yaml header, where the OPTIONAL `aperiodic: 1` key is the
    boundary opt-out -- the same key opensg_solid.plate_homo_2d reads, so the
    two CLIs share one header contract; an explicit argument always wins.
      * "periodic"  -- warping fluctuation w periodic: opposite faces, edges
        and corners tied through the sparse assembly map; rigid translations
        removed by 3 area-weighted Lagrange rows.
      * "aperiodic" -- the BOUNDARY-SOLUTION treatment: for a unit cell the
        boundary solution is the macro (affine) field itself, and mapping it
        onto the boundary nodes prescribes zero TRANSLATIONAL fluctuation
        (w1 = w2 = w3 = 0, Dirichlet) on every node of the bounding-box
        faces; the rotational DOFs stay natural (free).  Translations are
        what the macro solid strain prescribes, and clamping them forbids
        the affine fluctuations that make a FREE aperiodic SG rank-one
        (see Rules/periodicity_in_solid_props.md) and fixes the rigid
        modes -- no Lagrange border.  Clamping the ROTATIONS too would
        over-stiffen the cut edges (measured: +12 %% -> +7 %% on C11 by
        freeing them).  Kinematic (Dirichlet) homogenization is an upper
        bound on one cell; it converges to "periodic" as the SG grows
        (boundary layer ~ 1/N) -- that convergence is the acceptance check.

    Variables
    ---------
    omega   : SG measure the homogenized law is divided by = the VOLUME the
              equivalent continuum occupies, i.e. the node BOUNDING-BOX
              volume ptp(y1)*ptp(y2)*ptp(y3), measured from the mesh by
              default -- the same convention opensg_solid uses, and the same
              cell the periodic assembly map ties (opposite box faces).  So
              the returned C3D, the CLI print and <base>_C3D.out are one and
              the same matrix.  An explicit argument (or the yaml `omega:`
              header key, which the CLI passes here) overrides it -- e.g.
              omega = the midsurface SURFACE AREA, also returned in the result
              dict as "surface_area", for a per-unit-area wall law.
    De, Gm  : laminate 6x6 (center-ref) and transverse-shear 2x2
    uniq,inv: master-node set and node->master map (identity if aperiodic)
    K, Dhe, Dee : fluctuation/macro energy blocks of
              2U = w^T K w + 2 w^T Dhe ebar + ebar^T Dee ebar
    V0      : minimizing fluctuation per macro strain column ebar;
              Deff = Dee + V0^T Dhe  (valid for both boundary treatments)
    g_source: accepted and IGNORED -- the wall transverse-shear block G is
              always the MSG (Yu-2002 LS) construction; see
              sg_materials.check_g_source
    """
    from .sg_materials import check_g_source
    check_g_source(g_source, "shell_sg3d")
    t0 = time.perf_counter()
    try:                                     # libyaml C loader: the pure-python parse
        from yaml import CSafeLoader as _YL3  # of a big 3-D SG mesh costs ~25 s alone
    except ImportError:
        from yaml import SafeLoader as _YL3
    d = _yaml.load(open(yaml_path), Loader=_YL3)
    if boundary is None:                     # yaml header: `aperiodic: 1` is
        _ap = d.get("aperiodic")             # the opt-out, absent = periodic
        boundary = "aperiodic" if _ap is not None and int(_ap) else "periodic"
    if boundary not in ("periodic", "aperiodic"):
        raise ValueError("boundary must be 'periodic' or 'aperiodic', got %r"
                         % (boundary,))
    row = lambda r: " ".join(str(x) for x in
                             (r if isinstance(r, list) else [r])).split()
    if shear not in ("mitc", "full"):
        raise ValueError("shell_sg3d shear must be 'mitc' or 'full', got %r"
                         % (shear,))
    tri_scheme = tri_scheme_for(shear)                 # 'mitc' -> 'mitc3'
    nd = np.array([[float(v) for v in row(r)][:3] for r in d["nodes"]])
    el = [[int(v) for v in row(r)] for r in d["elements"]]
    # MIXED MESH: a 3-node entry is a NATIVE triangle (tri3 + MITC3), a 4-node
    # entry a genuine quad (MITC4).  The collapsed-quad triangle is gone.
    nv = np.array([len(e) for e in el], int)
    if not np.all((nv == 3) | (nv == 4)):
        bad = int(np.argmax((nv != 3) & (nv != 4)))
        raise ValueError("shell_sg3d: element %d has %d nodes; a 3-D shell SG "
                         "holds 3-node triangles and/or 4-node quads only"
                         % (bad, nv[bad]))
    itri = np.nonzero(nv == 3)[0]
    iquad = np.nonzero(nv == 4)[0]
    el3 = (np.array([el[i] for i in itri], int) - 1) if len(itri) \
        else np.zeros((0, 3), int)
    el4 = (np.array([el[i] for i in iquad], int) - 1) if len(iquad) \
        else np.zeros((0, 4), int)
    # the guard that makes the silent-NaN collapsed-quad path unreachable
    check_element_edges(nd, el3, tag="triangle", ids=itri)
    check_element_edges(nd, el4, tag="quad", ids=iquad)
    ori = np.array(d["elementOrientations"], float)
    e33, e34 = ori[itri, 6:9], ori[iquad, 6:9]
    nn, ne, ne3, ne4 = len(nd), len(el), len(el3), len(el4)

    # STEP 1 of the two-step homogenization: every layup -> its wall plate law.
    # Done for ALL sections (not just the one the SG solve uses) so the emitted
    # _ABDG.out is the complete step-1 record, exactly as the cross-section
    # routes build_rm_bundle / build_solid_bundle write it.
    D_by, G_by = _material_by_section(d["sections"], d["materials"],
                                      center_ref=True)
    # wall transverse-shear G: the MSG (Yu-2002 LS) construction, the only route
    from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
    from .sg_materials import material_db_from_yaml
    _mdb = material_db_from_yaml(d["materials"])
    G_by = [np.asarray(G_by[si], float).reshape(2, 2)
            for si in range(len(d["sections"]))]
    for si, sec in enumerate(d["sections"]):
        pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
        rr = rm_plate_msg([p[1] for p in pl], [p[2] for p in pl],
                          [p[0] for p in pl], _mdb, fraction=0.5)
        if rr["G_msg"] is not None:
            G_by[si] = np.asarray(rr["G_msg"], float).reshape(2, 2)
    # step 1 on disk, same emitter and same layout as the cross-section routes
    write_abdg_out(os.path.splitext(yaml_path)[0] + "_ABDG.out",
                   d["sections"], D_by, G_by)
    # STEP 2 uses ONE wall law for the whole surface: section 0.  A per-element
    # section id (`sets:`/`rsub`) is not yet plumbed through this route, so a
    # multi-section 3-D shell SG would silently run on layup 0 -- refuse it
    # rather than return a wrong number (the ABDG above still lists them all).
    if len(d["sections"]) > 1:
        raise NotImplementedError(
            "shell_sg3d homogenizes a 3-D shell SG with ONE wall law; this yaml"
            " declares %d sections (%s).  The per-element section map is not"
            " wired into this route yet -- split the SG, or use one section."
            % (len(d["sections"]),
               ", ".join(str(s.get("elementSet", i))
                         for i, s in enumerate(d["sections"]))))
    De = np.asarray(D_by[0] if not isinstance(D_by, dict) else D_by[0],
                    float).reshape(6, 6)
    Gm = np.asarray(G_by[0], float).reshape(2, 2)

    # junction lines: shell edges shared by more than two elements
    cnt = Counter()
    for e in el:
        for k in range(len(e)):
            cnt[tuple(sorted((e[k], e[(k+1) % len(e)])))] += 1
    n_junc_edges = sum(1 for v in cnt.values() if v > 2)

    if boundary == "periodic":
        # periodic map on the FULL 3-D coordinates: all faces/edges/corners
        rc, _ = mesh_to_periodic_sparse_assembly_map(nn,
                                                     np.arange(nn)[:, None],
                                                     nd, 3, NDOF6)
        master = np.asarray(rc, int).ravel()
        uniq, inv = np.unique(master, return_inverse=True)
    else:                        # aperiodic: every node its own master
        uniq = inv = np.arange(nn)
    ndof = NDOF6*len(uniq)

    Xe4 = nd[el4]
    Xe3 = nd[el3]
    gd4 = (NDOF6*inv[el4][:, :, None]
           + np.arange(NDOF6)[None, None, :]).reshape(ne4, 24).astype(np.int32)
    gd3 = (NDOF6*inv[el3][:, :, None]
           + np.arange(NDOF6)[None, None, :]).reshape(ne3, TRI_NDOF).astype(np.int32)
    Dhe = np.zeros((ndof, 6))
    Dee = np.zeros((6, 6))
    A11 = De[0, 0]
    A_surf = 0.0
    Krow, Kcol, Kval = [], [], []

    # ---------------- batch 1: genuine QUADS (bilinear, 2x2 Gauss, MITC4) ---------
    if ne4:
        tie4 = None if shear == "full" else _tie_rows_solid4(Xe4, e34, [1, 2], 0)
        Ke_acc = np.zeros((ne4, 24, 24))
        Fe_acc = np.zeros((ne4, 24, 6))
        for xi, eta in _GPTS:
            B, Bg, Dr, dA = solid_fluct_ops_batch(Xe4, e34, xi, eta, [1, 2], 0)
            BDe6, BGe6, _ = solid_macro_ops_batch(Xe4, e34, xi, eta, [1, 2], 0)
            if tie4 is not None:
                Bg = _shear_batch(xi, eta, "mitc4_both", Bg, tie4)
            w = dA[:, None, None]
            DB = np.einsum('ij,ejb->eib', De, B)*w
            GB = np.einsum('ij,ejb->eib', Gm, Bg)*w
            DBe = np.einsum('ij,ejb->eib', De, BDe6)*w
            GBe = np.einsum('ij,ejb->eib', Gm, BGe6)*w
            Ke_acc += np.einsum('eia,eib->eab', B, DB) \
                + np.einsum('eia,eib->eab', Bg, GB) \
                + (drill_pen*A11)*(dA[:, None, None]*Dr[:, :, None]*Dr[:, None, :])
            Fe_acc += np.einsum('eia,eib->eab', B, DBe) \
                + np.einsum('eia,eib->eab', Bg, GBe)
            Dee += np.einsum('eia,eib->ab', BDe6, DBe) \
                + np.einsum('eia,eib->ab', BGe6, GBe)
            A_surf += float(dA.sum())
        np.add.at(Dhe, gd4.ravel(), Fe_acc.reshape(-1, 6))
        Krow.append(np.repeat(gd4, 24, 1).ravel())
        Kcol.append(np.tile(gd4, (1, 24)).ravel())
        Kval.append(Ke_acc.ravel())

    # ---------------- batch 2: native TRIANGLES (tri3, 3-point rule, MITC3) -------
    # a separate (ne3,18,18) batch scattered into the SAME global system -- no
    # padding to 24 DOF, no per-element Python loop
    if ne3:
        tie3 = None
        if tri_scheme == "mitc3":
            tie3 = [solid_fluct_ops_tri_batch(Xe3, e33, tr, ts, [1, 2], 0)[1]
                    for (tr, ts) in TIE_MITC3]
        Ke_acc = np.zeros((ne3, TRI_NDOF, TRI_NDOF))
        Fe_acc = np.zeros((ne3, TRI_NDOF, 6))
        for (r_, s_, wq) in TRI_GPTS:
            B, Bg, Dr, dJ, Gmet = solid_fluct_ops_tri_batch(Xe3, e33, r_, s_,
                                                            [1, 2], 0)
            BDe6, BGe6, _ = solid_macro_ops_tri_batch(Xe3, e33, r_, s_, [1, 2], 0)
            if tie3 is not None:
                Bg = mitc3_shear_batch(Gmet, tie3[0], tie3[1], tie3[2], r_, s_)
            dA = wq*dJ                       # rule weight x |g_r x g_s|
            w = dA[:, None, None]
            DB = np.einsum('ij,ejb->eib', De, B)*w
            GB = np.einsum('ij,ejb->eib', Gm, Bg)*w
            DBe = np.einsum('ij,ejb->eib', De, BDe6)*w
            GBe = np.einsum('ij,ejb->eib', Gm, BGe6)*w
            Ke_acc += np.einsum('eia,eib->eab', B, DB) \
                + np.einsum('eia,eib->eab', Bg, GB) \
                + (drill_pen*A11)*(dA[:, None, None]*Dr[:, :, None]*Dr[:, None, :])
            Fe_acc += np.einsum('eia,eib->eab', B, DBe) \
                + np.einsum('eia,eib->eab', Bg, GBe)
            Dee += np.einsum('eia,eib->ab', BDe6, DBe) \
                + np.einsum('eia,eib->ab', BGe6, GBe)
            A_surf += float(dA.sum())
        np.add.at(Dhe, gd3.ravel(), Fe_acc.reshape(-1, 6))
        Krow.append(np.repeat(gd3, TRI_NDOF, 1).ravel())
        Kcol.append(np.tile(gd3, (1, TRI_NDOF)).ravel())
        Kval.append(Ke_acc.ravel())

    K = sp.csr_matrix((np.concatenate(Kval),
                       (np.concatenate(Krow), np.concatenate(Kcol))),
                      shape=(ndof, ndof))

    if boundary == "periodic":
        # kernel: the 3 rigid translations, area-weighted Lagrange rows.  Any
        # strictly positive nodal weighting removes the same kernel and leaves
        # Deff untouched (a translation is annihilated by B, Bg and DR, so
        # V0^T Dhe does not see it) -- both element families just contribute
        # their own area.
        wA = np.zeros(len(uniq))
        if ne4:
            _, _, _, dA0 = solid_fluct_ops_batch(Xe4, e34, 0.0, 0.0, [1, 2], 0)
            np.add.at(wA, inv[el4].ravel(), np.repeat(dA0, 4))
        if ne3:
            dA0 = solid_fluct_ops_tri_batch(Xe3, e33, _ONE3, _ONE3,
                                            [1, 2], 0)[3]
            np.add.at(wA, inv[el3].ravel(), np.repeat(dA0/6.0, 3))
        Cc = sp.lil_matrix((3, ndof))
        for k in range(3):
            Cc[k, k::NDOF6] = wA
        if solver == "cg":
            # device-resident GPU path: the 3 area-weighted translation rows
            # become a projection and ALL 6 macro load cases solve at once
            # (vmapped matrix-free CG; see sparse_projected_cg)
            from opensg_solid.sg_assembly import sparse_projected_cg
            V0 = sparse_projected_cg(K, np.asarray(Cc.todense()), -Dhe, NDOF6)
        else:
            A = sp.bmat([[K, Cc.T], [Cc, None]], format="csc")
            R = np.zeros((ndof + 3, 6))
            R[:ndof] = -Dhe
            V0 = _sparse_solve(A, R, sym=True)[:ndof]
        n_bnd = 0
    else:
        if solver == "cg":
            raise ValueError("shell_sg3d solver='cg' supports the periodic "
                             "boundary (the SwiftComp-parity default)")
        # boundary solution mapped to boundary nodes: zero translational
        # fluctuation on the bounding-box faces (rotations stay natural);
        # solve the free interior
        box0, box1 = nd.min(0), nd.max(0)
        tol = 1e-6*float(np.max(box1 - box0))
        onb = ((np.abs(nd - box0) < tol) | (np.abs(nd - box1) < tol)).any(1)
        bnodes = np.where(onb)[0]
        n_bnd = len(bnodes)
        bd = (NDOF6*bnodes[:, None] + np.arange(3)[None, :]).ravel()
        free = np.setdiff1d(np.arange(ndof), bd)
        V0 = np.zeros((ndof, 6))
        V0[free] = _sparse_solve(K[free][:, free].tocsc(), -Dhe[free], sym=True)
    Deff = Dee + V0.T @ Dhe
    Deff = 0.5*(Deff + Deff.T)
    # SG measure: the VOLUME the equivalent continuum occupies = the node
    # bounding box, the same convention opensg_solid uses.  It is also the
    # periodic cell this route ties (the assembly map pairs opposite box
    # faces), so C3D, the CLI print and the .out are now ONE number.
    V_cell = float(np.prod(nd.max(0) - nd.min(0)))
    if omega is None:
        omega = V_cell                       # SG measure = unit-cell volume
    C3D = Deff/float(omega)
    solve_time = time.perf_counter() - t0

    from opensg_solid.sg_homo import write_sc_K
    # the .out follows SwiftComp's normalization (per unit-cell volume) so its
    # moduli compare directly with solid .K files; with the default omega the
    # returned C3D is that same matrix
    bc_txt = ("periodic in 3 dirs" if boundary == "periodic"
              else "aperiodic: w=0 on %d boundary nodes" % n_bnd)
    el_txt = ("%d tri3/%s" % (ne3, "MITC3" if tri_scheme == "mitc3" else "full")
              if ne3 else "")
    el_txt += (", " if (ne3 and ne4) else "")
    el_txt += ("%d quad4/%s" % (ne4, "MITC4" if shear == "mitc" else "full")
               if ne4 else "")
    write_sc_K(os.path.splitext(yaml_path)[0] + "_C3D.out", Deff/V_cell,
               solve_time=solve_time,
               model="msg-shell equivalent 3D solid (3-D shell SG, %d nodes,"
                     " %d elems [%s], %d junction edges, %s,"
                     " per unit cell %.6g)"
                     % (nn, ne, el_txt, n_junc_edges, bc_txt, V_cell),
               name="Cauchy Continuum")
    return {"C3D": C3D, "D_eff": Deff, "solve_time": solve_time,
            "n_junction_edges": n_junc_edges, "ndof": ndof,
            "boundary": boundary, "n_boundary_nodes": n_bnd,
            "omega": float(omega), "cell_volume": V_cell,
            "surface_area": float(A_surf),
            "n_tri": ne3, "n_quad": ne4, "shear": shear,
            "tri_shear": tri_scheme}


# ===================== from segment_taper.py =====================

NDOF6 = 6            # u1 u2 u3 theta_1 theta_2 omega_3 (independent drilling)
NEPS = 4             # macro strain columns: eps_11, kappa_1, kappa_2, kappa_3


def _augment_drilling(Dhh, Dhe, Dhl, Dll, Dle, Gc, Gl, Ge):
    """Append the element-constant drilling Lagrange multipliers as a
    saddle-point block.

    The drilling rotation omega_3 is an independent 6th DOF; objectivity is
    restored by the element-wise constraint  g_e(V) = 0  enforced with one
    multiplier per element:

        [ Dhh  Gc^T ] [ V ]   [ -Dhe ]
        [ Gc    0   ] [ mu] = [ -Ge  ]        (zeroth order)

    Variables
    ---------
    Dhh, Dhe, Dhl, Dll, Dle : warping/strain energy blocks, 2U =
        V'^T Dll V' + 2 V'^T (Dhl^T V + Dle eps) + V^T Dhh V + 2 V^T Dhe eps
        + eps^T Dee eps   (V' = d V/d x_axis; MSG shell energy functional)
    Gc, Gl, Ge : constraint rows paired with (V, V', eps) respectively
    M, P       : warping DOF count, multiplier count
    returns    : (Dhh_a, Dhe_a, Dhl_a, Dll_a, Dle_a) on the augmented
                 space of size naug = M + P
    """
    M, P = Dhh.shape[0], Gc.shape[0]
    naug = M + P
    Dhh_a = np.zeros((naug, naug))
    Dhh_a[:M, :M] = Dhh
    Dhh_a[:M, M:] = Gc.T
    Dhh_a[M:, :M] = Gc
    Dhe_a = np.zeros((naug, NEPS)); Dhe_a[:M] = Dhe; Dhe_a[M:] = Ge
    Dhl_a = np.zeros((naug, naug)); Dhl_a[:M, :M] = Dhl; Dhl_a[M:, :M] = Gl
    Dll_a = np.zeros((naug, naug)); Dll_a[:M, :M] = Dll
    Dle_a = np.zeros((naug, NEPS)); Dle_a[:M] = Dle
    return Dhh_a, Dhe_a, Dhl_a, Dll_a, Dle_a


def _boundary_dirichlet(rings, b, key):
    """Map a boundary-ring warping field onto segment boundary DOFs.

    Replaces the periodic boundary condition of the ring problem: on each
    end cross-section the segment warping is PRESCRIBED to the separately
    solved ring solution,

        V_seg(x_boundary node, :) = V_ring(ring node, :)     (all 6 DOFs)

    Variables
    ---------
    rings : {"L"/"R": {"V0"/"V1": (6m x 4) ring field}} from ring_indep
    b     : the boundary bundle (npz) -- b["L_node2seg"][i] is the segment
            node index of ring node i (likewise "R_")
    key   : "V0" or "V1"
    V     : ring field reshaped (m_ring, 6, 4)
    bd    : constrained global DOF indices, 6*seg_node + component
    bv    : prescribed values, one (4,) row per constrained DOF
    """
    bd, bv = [], []
    for side in ("L", "R"):
        V = rings[side][key].reshape(-1, NDOF6, NEPS)
        for i, sn in enumerate(np.asarray(b["%s_node2seg" % side])):
            for c in range(NDOF6):
                bd.append(NDOF6*int(sn) + c)
                bv.append(V[i, c, :])
    return np.array(bd), np.array(bv, float)


def segment_timo_from_3dyaml(seg_yaml, workdir=None, lam_space="elem",
                             shear="full", write_boundary_yamls=True,
                             return_full=False):
    """Timoshenko 6x6 of a shell SEGMENT with aperiodic (Dirichlet) ends.

    Zeroth order (Euler-Bernoulli):
        Dhh_a V0 = -Dhe_a   with  V0 = V0_ring  on both end sections
        A_EB    = (Dee + V0^T Dhe_a) / L                       (4x4)
    First order (generalized Timoshenko, Yu et al.):
        Dhh_a V1 = b(V0)    with  V1 = V1_ring  on both end sections
        (b and the auxiliary blocks V0^T Dll V0, Dhl V0, Dhl^T V0 + Dle are
         formed by prepare_v1_rhs; finalize_v1_and_compute_deff performs the
         energy transformation (A_EB, B, C, D) -> S 6x6, each block / L)
    One LU factorization of the constrained Dhh_a serves both solves.

    Variables
    ---------
    seg_yaml  : 3-D shell segment yaml (nodes, quads, sections, materials)
    workdir   : output folder (default: alongside seg_yaml)
    lam_space : drilling-multiplier space; "elem" = element-constant
    shear     : transverse-shear integration for the SEGMENT operators;
                "full" is the production default (MITC tying aliases the
                drilling content on flat-walled/webbed sections)
    ax, cross : axial coordinate index and the two cross-section indices
    D_by, G_by: per-section laminate stiffness / shear blocks (center ref)
    k22_e,kg_e: element hoop-curvature and geometric-curvature corrections
    rings     : per-side ring results {C6, V0, V1} solved SEPARATELY
    Lz        : segment length = extent of nodes along the axis
    S6        : symmetrized segment Timoshenko 6x6
    returns   : dict(S6, C6L, C6R, L, solve_time [, V0, V1, rings, npz])
                and writes `<yaml>_Timo.out` (SwiftComp layout, timed)
    """
    import jax.numpy as jnp
    from .sg_mesh import extract
    from .sg_assembly import (dirichlet_solve, dirichlet_factor,
                                  dirichlet_solve_fac, compute_k22,
                                  compute_kg)
    from .sg_assembly import (assemble_segment_indep, assemble_constraint,
                                build_C_Psi_segment6)
    from .sg_materials import _material_by_section
    from .fe_jax.msg_solver import prepare_v1_rhs, finalize_v1_and_compute_deff

    t0 = time.perf_counter()
    base = os.path.splitext(seg_yaml)[0]
    if workdir is not None:
        base = os.path.join(workdir, os.path.basename(base))
    npz = base + "_boundaries.npz"
    # 1. boundary extraction: free-edge components -> 1-D yamls + node2seg
    with contextlib.redirect_stdout(io.StringIO()):
        extract(seg_yaml, npz, write_yaml=write_boundary_yamls)
    b = np.load(npz, allow_pickle=True)
    ax = int(b["axis"])
    cross = tuple(j for j in range(3) if j != ax)
    nodes = np.asarray(b["seg_x"])
    quads = np.asarray(b["seg_cells"])
    sd = np.asarray(b["seg_subdom"])
    e1s, e2s, e3s = (np.asarray(b["seg_e1"]), np.asarray(b["seg_e2"]),
                     np.asarray(b["seg_e3"]))
    D_by, G_by = _material_by_section(json.loads(str(b["sections"])),
                                      json.loads(str(b["materials"])),
                                      center_ref=True)
    cents = nodes[quads].mean(1)
    k22_e = compute_k22(cents, e2s, e3s, quads)
    kg_e = compute_kg(cents, e1s, e2s, e3s, quads)

    # 2. boundary rings solved SEPARATELY -> C6, V0, V1 per side
    rings = {}
    for side in ("L", "R"):
        rx = np.asarray(b["%s_x" % side])
        rc = np.asarray(b["%s_cells" % side])
        rs = np.asarray(b["%s_subdom" % side])
        re3 = np.asarray(b["%s_e3" % side])
        kr = compute_k22(rx[rc].mean(1), np.asarray(b["%s_e2" % side]),
                         re3, rc)
        C6r, V0r, V1r = ring_indep(rx, rc, rs, re3, D_by, G_by, kr, ax,
                                   list(cross), lam_space=lam_space,
                                   return_fields=True)
        rings[side] = dict(C6=C6r, V0=V0r, V1=V1r)

    # 3. segment operators + drilling constraint (saddle point, pen=0)
    Dhh, Dhe, Dee, Dhl, Dll, Dle = assemble_segment_indep(
        nodes, quads, sd, e3s, D_by, G_by, k22_e, cross, ax, kg_e=kg_e,
        pen=0.0, shear=shear)
    Gc, Gl, Ge = assemble_constraint(nodes, quads, sd, e3s, k22_e, cross, ax,
                                     kg_e=kg_e, lam_space=lam_space)
    Dhh, Dhe, Dhl, Dll, Dle = map(np.asarray, (Dhh, Dhe, Dhl, Dll, Dle))
    M = Dhh.shape[0]
    Dhh_a, Dhe_a, Dhl_a, Dll_a, Dle_a = _augment_drilling(
        Dhh, Dhe, Dhl, Dll, Dle, np.asarray(Gc), np.asarray(Gl),
        np.asarray(Ge))
    naug = Dhh_a.shape[0]
    # rigid-body projector C and macro mode shapes Psi on the augmented space
    C, Psi = build_C_Psi_segment6(nodes, quads, cross)
    Psi[3::6, 3] *= -1.0             # torsion mode sign convention
    Psi_a = np.zeros((naug, NEPS)); Psi_a[:M] = Psi
    Dc_a = np.zeros((naug, NEPS)); Dc_a[:M] = C.T

    # 4a. zeroth order:  Dhh_a V0 = -Dhe_a,  V0 = V0_ring on the ends
    bd0, bv0 = _boundary_dirichlet(rings, b, "V0")
    fac = dirichlet_factor(Dhh_a, bd0)
    V0 = dirichlet_solve_fac(fac, -Dhe_a, bd0, bv0)
    Lz = float(nodes[:, ax].max() - nodes[:, ax].min())
    EB = (np.asarray(Dee) + V0.T @ Dhe_a)/Lz           # A_EB (4x4), per length
    # 4b. first order:  Dhh_a V1 = b(V0),  V1 = V1_ring on the ends
    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        jnp.array(V0), jnp.array(Dhl_a), jnp.array(Dll_a), jnp.array(Dle_a),
        jnp.array(Psi_a), jnp.array(Dc_a))
    bd1, bv1 = _boundary_dirichlet(rings, b, "V1")
    V1 = (dirichlet_solve_fac(fac, np.asarray(bb), bd1, bv1)
          if np.array_equal(bd0, bd1) else
          dirichlet_solve(Dhh_a, np.asarray(bb), bd1, bv1))
    # generalized-Timoshenko transformation; energy blocks per unit length
    S6, *_ = finalize_v1_and_compute_deff(
        jnp.array(V1), jnp.array(V0), jnp.array(EB),
        jnp.array(np.asarray(V0DllV0)/Lz), jnp.array(np.asarray(DhlV0)/Lz),
        jnp.array(np.asarray(DhlTV0Dle)/Lz), jnp.array(Psi_a),
        jnp.array(Dc_a))
    S6 = 0.5*(np.asarray(S6) + np.asarray(S6).T)
    solve_time = time.perf_counter() - t0

    from opensg_solid.sg_homo import write_sc_K
    write_sc_K(base + "_Timo.out", S6, solve_time=solve_time,
               model="msg-shell tapered/aperiodic segment"
                     " (boundary V0/V1 Dirichlet, L=%.6g)" % Lz,
               constants=False, name="Timoshenko Beam")
    out = dict(S6=S6, C6L=rings["L"]["C6"], C6R=rings["R"]["C6"], L=Lz,
               solve_time=solve_time)
    if return_full:
        out.update(V0=np.asarray(V0[:M]), V1=np.asarray(V1[:M]),
                   rings=rings, npz=npz)
    return out


# ===================== from dehom_rm.py (build_rm_bundle, _strip) =====================

def _strip(rx3, cells, ax):
    """Reconstruct the one-quad-deep prismatic strip mesh EXACTLY as ring_indep does.

    In:
        rx3: (m,3) float, ring node coordinates.
        cells: (n_el,2) int, ring line-element connectivity.
        ax: int, beam-axis index (0/1/2) the strip is extruded along.
    Out:
        nodes: (2m,3) float, strip node coordinates (ring + extruded copy).
        quads: (n_el,4) int, quad connectivity [a, b, m+b, m+a].
        h: float, extrusion length (mean ring element length).
    """
    rx3 = np.asarray(rx3, float); m = len(rx3)
    h = float(np.mean(np.linalg.norm(rx3[cells[:, 1]] - rx3[cells[:, 0]], axis=1)))
    ez = np.zeros(3); ez[ax] = 1.0
    nodes = np.vstack([rx3, rx3 + h * ez])
    quads = np.array([[a, b, m + b, m + a] for a, b in cells], dtype=int)
    return nodes, quads, h


def build_rm_bundle(shell_yaml, ref=None, shear="mitc4_g23", g_source=None):
    """Homogenize with the RM ring and package everything the two-step dehom needs.

    ``ref=None`` reads the reference surface from the yaml's ``reference`` field -- the single
    source of truth, set when the 1-D yaml is created (absent -> "center" = mid-surface) -- so
    homogenization and dehom follow the same reference; pass an explicit ``ref`` only to override.

    In:
        shell_yaml: str, path to the 1-D shell section yaml.
        ref: None | "center" | "oml" | "oml_flip" | "iml", reference-surface override.
        shear: str, RM transverse-shear tying scheme passed to ring_indep.
        g_source: DEPRECATED, accepted and ignored.  The wall transverse-shear block
            G is always the MSG (Yu-2002 least-squares) projection -- the only route;
            see sg_materials.check_g_source.
    Out:
        dict bundle: "Timo" (6,6) RM Timoshenko matrix; "solve_time" float (the
        value written into <base>_Timo.out); "V0"/"V1" (6m,4) warping modes;
        "corners" (m,2) section contour coords; "red_cells" (n_el,2) connectivity;
        "rx3" (m,3), "re3" (n_el,3), "k22" (n_el,) ring geometry; "ax" int beam axis;
        "cross" axis pair; "strip" (nodes, quads, h); "layup_per_elem" list of layup
        names per element; "layup_db"/"material_db" plate-SG databases (by-name,
        geometry-free); "frac" float; "ref" str; "g_source" str (always "msg").
    """
    import time as _time
    from .sg_mesh import load_ring_ref
    from .sg_materials import check_g_source
    check_g_source(g_source, "build_rm_bundle")
    _t0 = _time.perf_counter()
    try:
        from yaml import CSafeLoader as _YL
    except ImportError:
        from yaml import SafeLoader as _YL
    d = yaml.load(open(shell_yaml), Loader=_YL)
    if ref is None:                                       # single source of truth: yaml records its ref
        ref = d.get("reference", "center")                # (set at 1-D-yaml creation; absent -> center)
    R = load_ring_ref(shell_yaml, ref)
    # Single reference decision: ring laminate ref, plate-SG z_ref, layup_db frac, emitted ABD,
    # and the recovery depth conversion in stress_at_points ALL follow ``frac``.
    frac = {"center": 0.5, "oml": 0.0, "oml_flip": 1.0, "iml": 1.0}.get(ref, 0.0)
    # G_msg is reference-independent, but the SG carries the recovery warping, so its z_ref
    # must sit at the chosen reference surface.
    # R["G_by"] is a per-section DICT {si: (2,2)} -- materialize the per-section list of
    # VALUES (list(dict) yields the integer keys; latent until a section keeps its
    # loader G because the msg replacement below is skipped on a non-SPD U*-fit).
    G_by = [np.asarray(R["G_by"][si], float) for si in range(len(d["sections"]))]
    # wall transverse-shear G: the MSG (Yu-2002 LS) construction, the only route
    from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
    from .sg_materials import material_db_from_yaml
    _mdb = material_db_from_yaml(d["materials"])
    for si, sec in enumerate(d["sections"]):
        _pl = [[str(p[0]), float(p[1]), float(p[2])] for p in sec["layup"]]
        _h = sum(p[1] for p in _pl)
        _rr = rm_plate_msg([p[1] for p in _pl], [p[2] for p in _pl], [p[0] for p in _pl],
                           _mdb, fraction=frac)
        # guard the borderline-SPD U*-fit: accept the MSG G only when it is a
        # finite SPD 2x2 (ev_min ~ 0 flips run-to-run on thin tip laminates);
        # otherwise keep the energy-consistent G from the ring loader.
        _Gm = _rr["G_msg"]
        if _Gm is not None:
            _Gm = np.asarray(_Gm, float)
            if (_Gm.shape == (2, 2) and np.all(np.isfinite(_Gm))
                    and np.linalg.det(_Gm) > 0.0 and _Gm[0, 0] > 0.0):
                G_by[si] = _Gm
    import os as _os2
    write_abdg_out(_os2.path.splitext(shell_yaml)[0] + "_ABDG.out",
                   d["sections"], R["D_by"], G_by)
    C6, V0, V1 = ring_indep(R["rx"], R["cells"], R["rsub"], R["re3"], R["D_by"], G_by,
                            R["k22"], R["ax"], R["cross"], shear=shear, lam_space="elem",
                            return_fields=True)
    C6 = 0.5 * (C6 + C6.T)
    nodes, quads, h = _strip(R["rx"], R["cells"], R["ax"])

    sec_names = [s["elementSet"] for s in d["sections"]]
    layup_per_elem = [sec_names[int(si)] for si in R["rsub"]]
    # by-name plate layup_db + material_db are GEOMETRY-FREE loader dicts -- read them
    # directly (the KL solve_tw_from_yaml previously used here spent ~5 s/station
    # solving the whole Hermite ring just to surface these two dicts).
    from .fe_jax.msg_mesh import load_yaml as _load_sg_yaml
    _, _, _mat_db_kl, _layup_db_kl, _ = _load_sg_yaml(shell_yaml)
    # emit the per-station ABD yaml at the SAME reference (once, cached) for dehom + shell buckling
    try:
        import os as _os
        from .sg_materials import emit_station_abd
        _tag = _os.path.splitext(_os.path.basename(shell_yaml))[0]
        _ay = _os.path.join(_os.path.dirname(shell_yaml) or ".", "abd", _tag + "_abd.yaml")
        if not _os.path.exists(_ay):
            emit_station_abd(shell_yaml, _ay, station=_tag,
                             ref="mid" if ref == "center" else "oml")
    except Exception:
        # intentional best-effort: ABD yaml emission failure must not abort the bundle build
        pass
    # OpenSG default: SwiftComp-format timed .out for the beam model too
    from opensg_solid.sg_homo import write_sc_K
    _solve_time = _time.perf_counter() - _t0
    write_sc_K(_os2.path.splitext(shell_yaml)[0] + "_Timo.out", C6,
               solve_time=_solve_time,
               model="msg-shell beam model"
                     " [ext sh2 sh3 twist bend2 bend3]",
               constants=False, name="Timoshenko Beam")
    return {"Timo": C6, "solve_time": _solve_time,
            "V0": np.asarray(V0), "V1": np.asarray(V1),
            "corners": R["rx"][:, R["cross"]], "red_cells": np.asarray(R["cells"]),
            "rx3": np.asarray(R["rx"]), "re3": np.asarray(R["re3"]), "k22": np.asarray(R["k22"]),
            "ax": int(R["ax"]), "cross": list(R["cross"]), "strip": (nodes, quads, h),
            "layup_per_elem": layup_per_elem, "layup_db": _layup_db_kl,
            "material_db": _mat_db_kl, "frac": frac, "ref": ref,
            "g_source": "msg"}
