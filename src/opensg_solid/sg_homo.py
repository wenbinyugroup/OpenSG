"""sg_homo.py -- homogenization drivers of the general SG engine: the
SSDM periodic-CG pipeline (n_model 2/3 and the shared EB 4-mode beam
masks), the Beam_solid Timoshenko/KKT beam path (n_model 1), and the
public plate_homo_2d.  Element kernels / assembly / solvers live in
sg_assembly.py, materials in sg_materials.py, input in sg_mesh.py,
recovery in sg_dehom.py.  The beam KKT path needs n_sg >= 2 (the shared
EB kernels still cover the 1-D-SG beam via
full_homogenization_pipeline).

Voigt/order conventions: 3-D strain/stress order is the SwiftComp
(xx, yy, zz, yz, xz, xy) with y3 = the plate thickness direction being
the LAST SG coordinate; the plate law rows are
[N11 N22 N12 M11 M22 M12], the beam law rows
[eps11 gam12 gam13 kappa1 kappa2 kappa3] (Timoshenko).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE  (axis suffixes: e element, n elem-node,
# q quad point, d spatial dim, p parametric dim, H macro mode)
# ----------------------------------------------------------------------------
# inputs
#   sc_path, workdir    the .sc/.yaml input; where <base>.yaml/.msh/.png go
#   material_param      (n_mat, 9) engineering override or None
#   angles              (n_mat,) deg per material or None
#   n_model             1 beam (Timo/KKT), 2 plate, 3 solid
#   elem_rotation       (E, 9) per-element DCs or None (beam only)
# working (both paths)
#   sc, n_sg, base      parsed SG dict, its dimension, the output basename
#   points              (V, d) nodes;  cells (E, N);  cell_domain_ids (E,)
#   fe_type, xi_qp, W_q, phi_qn, dphi_dxi_qnp   basis/quadrature (basix)
#   x_end               (E, N, d);  V = #nodes
#   periodic_cells_en   (E, N) master connectivity;  dof_map_np (V*3,)
# working (SSDM paths, n_model 2/3)
#   solver              "direct" (csr + pardiso, default) | "cg" (SSDM)
#   shear_refined, plot RM G ladder switch (plate only); mesh-PNG switch
#   fe_hi, phi_hi, dphi_hi, W_hi   degree-2p+2 quadrature (l-blocks)
#   lad                 plate_shear_ladder dict -> r["G_msg"]/["ABDG"]/
#                       ["A6_ladder"]/["X_shear"]/["Ustar_rel"]
#   D_hh..D_l2e, w_dof, kernel, Hpsi, KKT, scl   ladder assembly (csr)
#   D1bar, D2bar, V11, V12, H11/H12/H22, S1, S2  the first-order ladder
#   _D1_SG, _D2_SG, blocks/units/Amat/X/G        the U* LS reduction
#   C_m, C_ess          (n_mat, 6, 6) / (E, 6, 6) stiffness tables
#   unique_dofs, n_unique   the solve size (#dofs);  u_0_g_full zero seed
#   Dhe, J_euu, inv_blocks, A_op/M_op/cheb_M   CG pipeline internals
#   dof_map, rows/cols/data, keep, A_csr, RHS  _homo_direct assembly
#   V0_matrix           (n_unique, H) fluctuations;  D1, D_bar (H, H)
# working (beam KKT path, n_model 1)
#   reduced_cells, num_unique, x_unique, N_primal   compressed periodics
#   C_asm, C_stress     (E, 6, 6) assembly / recovery stiffness
#   Dhh..Dle, Dee, omega, dehomo_data   sg_assembly.assemble_system_matrices
#   Psi, Dc             (N_primal, 4) rigid-body ops
#   RHS_V0, V0, D1_V0, A_augmented      the KKT V0 solve
#   C_eb                (4, 4) EB stiffness = (Dee + D1_V0)/omega
#   bb, DhlV0, DhlTV0Dle, V0DllV0       l-chain RHS pieces (Eq. 100)
#   R_aug, V_aug, V1s_raw, V1s          the second KKT solve (Eq. 85)
#   B_tim, C_tim, Q_base/Q_tim, Ginv/G_tim, Y_tim, A_tim
#                       the Timoshenko reduction blocks
#   er                  (E, 9) elem_rotation (identity tile when None)
# outputs
#   C_eff               (6, 6) macro law (Timo 6x6 for beam); C_eff_EB (4, 4)
#   C_timo / Deff_srt   (6, 6) [eps11 gam12 gam13 kap1 kap2 kap3]
#   r                   the result dict -- everything sg_dehom needs rides
#                       along (V0/V1s, dehomo_data, C_*, connectivity)
# ----------------------------------------------------------------------------
"""
import os
from functools import partial
from typing import Any, Dict, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from fe_jax.basis_quadrature import (FiniteElementType, ElementFamily,
                                     LagrangeVariant, QuadratureType,
                                     get_quadrature,
                                     eval_basis_and_derivatives)
from fe_jax.setup import (mesh_to_jax,
                          mesh_to_periodic_sparse_assembly_map)

from opensg_solid.sg_assembly import (
    _sparse_direct_solve, apply_block_precond, apply_chebyshev_precond,
    assemble_rigid_body_ops, assemble_system_matrices,
    calculate_RHS_and_Ke_batch_periodic, compress_periodic_cells_jax,
    compute_block_inv_diag, compute_homogenized_constants,
    ebe_jacobian_product_periodic, estimate_max_eigenvalue,
    plate_ladder_element_blocks, solve_fluctuation_field)
from opensg_solid.sg_materials import (build_material_C,
                                       get_heterogeneous_C_matrix)
from opensg_solid.sg_mesh import _cell_basis, load_sg_input, plot_sg_mesh


@partial(jax.jit, static_argnames=['n_unique', 'n_model', 'n_sg'])
def full_homogenization_pipeline(
    x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, C_ess,
    periodic_cells, unique_dofs, n_unique, n_model, n_sg
):
    """SSDM Chebyshev-preconditioned CG homogenization: solve the periodic
    fluctuation columns and reduce to the effective macro law (selectable
    as solver="cg").  The eigen estimate is HOISTED out of the per-column
    solve: it never reads the column values, so one estimate serves all
    columns bitwise-identically.

    In:  x_end (E, N, d) element node coords; u_0_g (V*3,) zero seed;
         dphi_dxi_qnp/phi_qn/W_q quadrature basis derivatives, values,
         weights; C_ess (E, 6, 6) per-element stiffness; periodic_cells
         (E, N) master connectivity; unique_dofs (n_unique,) global dof
         ids; n_unique int solve size; n_model 1 beam / 2 plate /
         3 solid; n_sg SG dimension
    Out: C_eff (H, H) effective stiffness; V0_matrix (n_unique, H)
         fluctuation columns; omega float SG measure."""
    Dhe, J_euu = calculate_RHS_and_Ke_batch_periodic(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
        u_0_g[unique_dofs], n_model, n_sg)
    inv_blocks = compute_block_inv_diag(J_euu, periodic_cells, n_unique)

    A_op = jax.tree_util.Partial(
        ebe_jacobian_product_periodic, J_euu, periodic_cells, n_unique)
    M_op = jax.tree_util.Partial(
        apply_block_precond, inv_blocks, n_unique)
    estimated_eig_max = estimate_max_eigenvalue(A_op, M_op, -Dhe.T[0],
                                                num_iters=15)
    estimated_eig_min = estimated_eig_max / 25.0
    cheb_M = jax.tree_util.Partial(
        apply_chebyshev_precond, inv_blocks, n_unique,
        estimated_eig_max, estimated_eig_min, A_op, 4)

    def solve_inner(b_col):
        res, _ = jax.scipy.sparse.linalg.cg(A_op, b_col, M=cheb_M, tol=1e-6)
        return res

    # vmap over the macro load cases: all V0 columns solve SIMULTANEOUSLY
    # (batched matvecs on CPU, parallel on GPU; lax.map was sequential)
    V0_matrix = jax.vmap(solve_inner)(-Dhe.T).T
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, Dhe)
    D_bar, omega = compute_homogenized_constants(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, n_model, n_sg)
    C_eff = (D_bar + D1) / omega
    return C_eff, V0_matrix, omega


def _homo_direct(x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, C_ess,
                 periodic_cells, unique_dofs, n_unique, n_model, n_sg,
                 bdofs=None):
    """Direct-sparse variant of full_homogenization_pipeline (the
    default): SAME element kernels and the SAME pinned-first-node system
    the EBE operator applies (rows 0:3 -> identity, zero RHS), assembled
    into one csr and factorized once for all columns (pypardiso, scipy
    SuperLU fallback).  Digit-safe swap: C_eff is stationary in V0, so
    solver error enters second order.
    bdofs (aperiodic mode): global DOFs pinned to ZERO fluctuation
    (w = 0 Dirichlet on the boundary nodes) INSTEAD of the first-node
    pin -- the connectivity is then the raw mesh, not the periodic map.

    In:  x_end (E, N, d) element node coords; u_0_g (V*3,) zero seed;
         dphi_dxi_qnp/phi_qn/W_q quadrature basis tables; C_ess
         (E, 6, 6) per-element stiffness; periodic_cells (E, N)
         connectivity; unique_dofs (n_unique,) global dof ids; n_unique
         int solve size; n_model 1 beam / 2 plate / 3 solid; n_sg SG
         dimension; bdofs (n_b,) int global DOFs or None (periodic)
    Out: C_eff (H, H) effective stiffness; V0_matrix (n_unique, H)
         fluctuation columns; omega float SG measure."""
    Dhe, J_euu = calculate_RHS_and_Ke_batch_periodic(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells,
        u_0_g[unique_dofs], n_model, n_sg)
    E_elem, n_ed, _ = J_euu.shape
    N_nodes = n_ed // 3
    dof_map = ((np.asarray(periodic_cells, dtype=np.int64) * 3)
               .reshape(E_elem, N_nodes, 1)
               + np.arange(3)).reshape(E_elem, n_ed).astype(np.int32)
    rows = np.repeat(dof_map, n_ed, axis=1).ravel()
    cols = np.tile(dof_map, (1, n_ed)).ravel()
    data = np.asarray(J_euu).ravel()
    if bdofs is None:
        keep = rows >= 3                # pinned rows 0:3 -> unit diagonal
        pin = np.arange(3, dtype=np.int32)
    else:                               # pinned rows = boundary DOFs
        pinmask = np.zeros(n_unique, bool)
        pinmask[np.asarray(bdofs, np.int64)] = True
        keep = ~pinmask[rows]
        pin = np.where(pinmask)[0].astype(np.int32)
    rows = np.concatenate([rows[keep], pin])
    cols = np.concatenate([cols[keep], pin])
    data = np.concatenate([data[keep], np.ones(len(pin))])
    A_csr = csr_matrix((data, (rows, cols)), shape=(n_unique, n_unique))
    RHS = -np.asarray(Dhe)              # rows 0:3 already zeroed
    if bdofs is not None:
        RHS[pin] = 0.0                  # w = 0 on the boundary nodes
    V0_matrix = jnp.asarray(_sparse_direct_solve(A_csr, RHS, sym=True))
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, Dhe)
    D_bar, omega = compute_homogenized_constants(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, n_model, n_sg)
    C_eff = (D_bar + D1) / omega
    return C_eff, V0_matrix, omega


# ---------------- RM shear-refined plate law (from rm_plate_1D.msg_rm_plate)
# D_1, D_2 (Yu Eq. 51) selectors and the U* LS cutoff -- identical to
# msg_rm_plate (_LS_RCOND imported so the two stay in lockstep).
from opensg_solid.rm_plate_1D.msg_rm_plate import _LS_RCOND

_D1_SG = np.zeros((6, 2)); _D1_SG[3, 0] = 1.0; _D1_SG[5, 1] = 1.0
_D2_SG = np.zeros((6, 2)); _D2_SG[4, 1] = 1.0; _D2_SG[5, 0] = 1.0


def _rm_ls_reduction(A6, H11, H12, H22, S1, S2):
    """Yu Eqs. (57)-(61): the 78-equation / 27-unknown U* least squares
    -> X (2x2 shear compliance), G = X^-1.  NumPy port of the reduction
    inside msg_rm_plate._bucket.single (same column equilibration,
    truncated-SVD minimum-norm solve, SPD gate); it consumes only
    assembled quantities, so it is SG-dimension-agnostic.

    In:  A6 (6, 6) plate law; H11/H12/H22 (6, 6) second-order energy
         blocks; S1/S2 (2, 6) constraint couplings
    Out: G (2, 2) shear stiffness (inv(X)); X (2, 2) shear compliance;
         ev_min float min eigenvalue of X (SPD gate); Ustar_rel float
         relative LS residual; c1, c2 (2, 6) the kernel-translation
         constants of the LS solution -- the V1bar tilt columns of the
         Eq. 63 recovery (msg_rm_plate Eq. 58)."""
    H11 = 0.5 * (H11 + H11.T)
    H22 = 0.5 * (H22 + H22.T)
    H_tt = np.block([[H11, H12], [H12.T, H22]])
    AD1 = A6 @ _D1_SG
    AD2 = A6 @ _D2_SG

    def blocks(X, c1, c2):
        Bs = H11 + AD1 @ X @ AD1.T + c1.T @ S1 + S1.T @ c1
        Cs = H12 + AD1 @ X @ AD2.T + c1.T @ S2 + S1.T @ c2
        Ds = H22 + AD2 @ X @ AD2.T + c2.T @ S2 + S2.T @ c2
        return np.block([[Bs, Cs], [Cs.T, Ds]])

    M0 = blocks(np.zeros((2, 2)), np.zeros((2, 6)), np.zeros((2, 6)))
    b0 = -M0.ravel()
    cols = []
    for j in range(27):
        pj = np.zeros(27); pj[j] = 1.0
        Xj = np.array([[pj[0], pj[1]], [pj[1], pj[2]]])
        cols.append(blocks(Xj, pj[3:15].reshape(2, 6),
                           pj[15:27].reshape(2, 6)).ravel() + b0)
    Amat = np.stack(cols, axis=1)
    cs = np.linalg.norm(Amat, axis=0)
    cs = np.where(cs == 0, 1.0, cs)
    U, sig, Vt = np.linalg.svd(Amat / cs, full_matrices=False)
    ok = sig > _LS_RCOND * sig[0]
    sig_inv = np.where(ok, 1.0 / np.where(ok, sig, 1.0), 0.0)
    sol = (Vt.T @ (sig_inv * (U.T @ b0))) / cs
    X = np.array([[sol[0], sol[1]], [sol[1], sol[2]]])
    c1 = sol[3:15].reshape(2, 6)
    c2 = sol[15:27].reshape(2, 6)
    res = blocks(X, c1, c2)
    Ustar_rel = np.linalg.norm(res) / (np.linalg.norm(H_tt) + 1e-30)
    ev_min = float(np.linalg.eigvalsh(X).min())
    G = np.linalg.inv(X)
    return G, X, ev_min, Ustar_rel, c1, c2


def _detilt_cols_2d(cols, y2_n, wA_n):
    """The 2-D analog of rm_plate_1D._detilt_inplane: project the TILT
    (thickness-linear content) out of the IN-PLANE components of a
    warping-column block; w3 keeps its tilt.  The 1-D trapezoid line
    integrals become MATERIAL-AREA-weighted nodal sums (the same lumped
    areas the <w> = 0 constraint uses), which reduces to the 1-D form
    on a uniform through-thickness line.

    In:  cols (n_unique, 6); y2_n (n_nodes,) reduced-node thickness
         coordinate; wA_n (n_nodes,) lumped nodal material areas
    Out: (n_unique, 6) detilted copy."""
    W = np.asarray(cols).reshape(-1, 3, 6).copy()
    z2 = float((wA_n * y2_n * y2_n).sum())
    for comp in (0, 1):
        m1 = (wA_n[:, None] * y2_n[:, None] * W[:, comp, :]).sum(axis=0)
        W[:, comp, :] -= np.outer(y2_n, m1 / z2)
    return W.reshape(-1, 6)


def _plate_face_loads_3d(pts, cells, yf, ftol):
    """Consistent nodal loads of a UNIT pressure on one thickness face
    (y3 = yf) of a 3-D plate SG -- the 2-D analog of the trapezoid edge
    weights: the face of a plate SG is FLAT, so its facets are planar
    and the loads are exact area integrals of the shape functions.

    Facets are found GEOMETRICALLY (an element's nodes lying on the
    plane), never through topology tables, so gmsh-vs-basix corner
    ordering cannot mislabel them: tet4 -> 3 on-plane nodes (linear
    triangle, A/3 each), hex8 -> 4 (bilinear quad, 2x2 Gauss consistent
    load), tet10 -> 6 (straight P2 triangle: corners 0, midsides A/3;
    corner/midside split by the nodes-0..3-are-corners convention both
    gmsh and basix share).

    In:  pts (V, 3) SG coordinates; cells (E, N) int; yf float the face
         plane; ftol float the on-plane tolerance
    Out: (nid (n,) int node ids, wgt (n,) float loads) -- repeated node
         ids allowed, the caller scatter-adds."""
    on = np.abs(pts[:, 2] - yf) < ftol
    onc = on[np.asarray(cells)]
    N = cells.shape[1]
    need = {4: 3, 8: 4, 10: 6}.get(N)
    if need is None:
        raise ValueError("3-D plate face loads support tet4/hex8/tet10,"
                         " got %d-node elements" % N)
    nid_l, wgt_l = [], []
    for e in np.nonzero(onc.sum(axis=1) == need)[0]:
        f = np.asarray(cells[e])[onc[e]]
        p = pts[f][:, :2]
        if N == 4:
            A = 0.5 * abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                          - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
            nid_l.append(f)
            wgt_l.append(np.full(3, A / 3.0))
        elif N == 8:
            c = p.mean(axis=0)
            k = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
            f, p = f[k], p[k]
            g = 1.0 / np.sqrt(3.0)
            xi = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
            sn = np.array([[-1.0, -1.0], [1.0, -1.0],
                           [1.0, 1.0], [-1.0, 1.0]])
            w = np.zeros(4)
            for q in range(4):
                Nq = 0.25 * (1 + sn[:, 0] * xi[q, 0]) \
                    * (1 + sn[:, 1] * xi[q, 1])
                dN = 0.25 * np.column_stack(
                    [sn[:, 0] * (1 + sn[:, 1] * xi[q, 1]),
                     sn[:, 1] * (1 + sn[:, 0] * xi[q, 0])])
                J = dN.T @ p
                w += Nq * abs(np.linalg.det(J))
            nid_l.append(f)
            wgt_l.append(w)
        else:                                   # tet10 P2 facet
            corner = np.asarray(cells[e])[:4]
            fc = f[np.isin(f, corner)]
            fm = f[~np.isin(f, corner)]
            pc = pts[fc][:, :2]
            A = 0.5 * abs((pc[1, 0] - pc[0, 0]) * (pc[2, 1] - pc[0, 1])
                          - (pc[2, 0] - pc[0, 0]) * (pc[1, 1] - pc[0, 1]))
            nid_l.append(fm)
            wgt_l.append(np.full(len(fm), A / 3.0))
    if not nid_l:
        raise ValueError("no element facet lies on the y3 = %g face --"
                         " wrong thickness axis or tolerance?" % yf)
    return np.concatenate(nid_l), np.concatenate(wgt_l)


def plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess,
                       reduced_cells, n_unique, n_sg, omega,
                       f_faces=None, node_y2=None):
    """The RM transverse-shear block G (2x2) of a plate SG via the
    first-order warping ladder -- rm_plate_1D.msg_rm_plate's Eq. 30-61
    flow on the general SG assembly.

    DERIVATION STATUS.  n_sg=1: this IS Yu (2005) / msg_rm_plate --
    gated digit-tight on the 1-D anchors (iso nu=0 -> 5/6 G h, the test
    laminates).  n_sg=2/3: the SwiftComp-style extension -- the same
    energy functional with Gamma_h carrying the resolved-dimension
    derivatives, periodic master-slave coupling, and the <w> = 0
    average constraint replacing the 1-D node constraint; gated on the
    laminate-as-strip equivalence (a laminate meshed as a 2-D/3-D SG
    must reproduce its 1-D G under refinement).  NOT yet validated
    against an external reference for SGs heterogeneous IN-PLANE
    (e.g. the RHC honeycomb) -- SwiftComp cross-check is the open item.
    The Eq. 63 FIRST-ORDER refined recovery IS ported (sg_dehom.
    _v2_batch, driven by the .ff strain derivatives; gated on the
    homogeneous-plate sigma13 parabola, 1-D and 2-D SG).  The
    second-order Eq. 64-66 stage (V2x chains, tilt/detilt row split)
    and the qt6/qb6 pressure load ladder are ALSO ported for 2-D
    single-batch SGs (this function's node_y2/f_faces inputs;
    sg_dehom._v266_batch/_vq_batch) -- gated on Pagano [0/90/0] s=4
    vs exact elasticity and on flat/iso/sandwich closed-form loads.

    In:  x_end/dphi_hi/phi_hi/W_hi -- geometry + a quadrature exact to
         degree 2p+2; C_ess (E, 6, 6); reduced_cells (E, N); n_unique
         dofs; n_sg; omega -- the IN-PLANE SG measure (1-D SG: 1).
         Every per-element argument may also be a PER-BATCH LIST (one
         entry per element type of a mixed SG, all over the SHARED
         reduced dof space): the element blocks run per batch and
         scatter-add into the same global csr/dense objects, so the RM
         math below never sees the batching (the fe_jax ElementBatch /
         shell_sg3d tri+quad doctrine).
         f_faces OPTIONAL (n_unique, 2) -- consistent nodal loads of a
         UNIT face pressure on the [top, bottom] SG face (already
         /omega, signs top -, bottom +, exactly the msg_rm_plate Lt/Lb
         convention).  When given, the PRESSURE-DRIVEN warping ladder
         (Yu Eqs. 29/45/64, the rm_plate_1D load columns) is solved on
         the same constrained system and returned: V1Lt/V1Lb
         (n_unique,) first-order load columns and V2Lt/V2Lb
         (n_unique, 5) quintets for the (q,1 q,2 q,11 q,12 q,22)
         drivers.  sg_dehom consumes them for the qt6/qb6 recovery.
    Out: dict A6 (6, 6), G_msg (2, 2, None unless X SPD), X, ev_min,
         Ustar_rel, V0/V11/V12/V11bar/V12bar (n_unique, 6)
         [+ V1Lt/V2Lt/V1Lb/V2Lb when f_faces is given]."""
    from opensg_solid.sg_mixed import ladder_blocks   # assembly layer
    blk = ladder_blocks(x_end, dphi_hi, phi_hi, W_hi, C_ess,
                        reduced_cells, n_unique, n_sg, omega)
    D_hh, D_hl1, D_hl2 = blk["D_hh"], blk["D_hl1"], blk["D_hl2"]
    D_l11, D_l12, D_l22 = blk["D_l11"], blk["D_l12"], blk["D_l22"]
    D_he, D_l1e, D_l2e = blk["D_he"], blk["D_l1e"], blk["D_l2e"]
    D_ee, w_dof = blk["D_ee"], blk["w_dof"]

    kernel = np.zeros((n_unique, 3))
    kernel[0::3, 0] = kernel[1::3, 1] = kernel[2::3, 2] = 1.0
    # <w> = 0 via the KKT rows (psi normalization immaterial; scl
    # balances the blocks as in msg_rm_plate)
    scl = float(np.max(np.abs(D_hh.diagonal())))
    Hpsi = w_dof[:, None] * kernel
    from scipy.sparse import bmat as _bmat
    KKT = _bmat([[D_hh, scl * csr_matrix(Hpsi)],
                 [scl * csr_matrix(Hpsi.T), None]], format="csr")

    def solve_constrained(rhs):
        aug = np.vstack([-rhs, np.zeros((3, rhs.shape[1]))])
        return _sparse_direct_solve(KKT, aug, sym=True)[:n_unique]

    V0 = solve_constrained(D_he)
    A6 = D_ee + V0.T @ D_he

    D1bar = (D_hl1 @ V0 - D_hl1.T @ V0) - D_l1e
    D2bar = (D_hl2 @ V0 - D_hl2.T @ V0) - D_l2e
    V1 = solve_constrained(np.concatenate([D1bar, D2bar], axis=1))
    V11, V12 = V1[:, :6], V1[:, 6:]

    H11 = V0.T @ (D_l11 @ V0) + D1bar.T @ V11
    H12 = V0.T @ (D_l12 @ V0) + 0.5 * (D1bar.T @ V12 + V11.T @ D2bar)
    H22 = V0.T @ (D_l22 @ V0) + D2bar.T @ V12
    S1 = (kernel.T @ D1bar)[:2]
    S2 = (kernel.T @ D2bar)[:2]

    G, X, ev_min, Ustar_rel, c1, c2 = _rm_ls_reduction(A6, H11, H12,
                                                       H22, S1, S2)
    # the Eq. 63 recovery columns: V1bar = V1 + kernel c_a with the
    # kernel constants of the LS solution IN-PLANE only (w3 keeps its
    # gauge) -- msg_rm_plate Eq. 58; raw V1bar feeds Gamma_h (the tilt
    # carries the mean shear), the ladder V0 feeds Gamma_l1/Gamma_l2
    c1_ks = np.zeros((3, 6)); c1_ks[:2] = c1
    c2_ks = np.zeros((3, 6)); c2_ks[:2] = c2
    out = {"A6": A6, "G_msg": (G if ev_min > 0 else None), "X": X,
           "ev_min": ev_min, "Ustar_rel": float(Ustar_rel),
           "V0": V0, "V11": V11, "V12": V12,
           "V11bar": V11 + kernel @ c1_ks,
           "V12bar": V12 + kernel @ c2_ks}
    if node_y2 is not None:
        # ---- second-order warping V2 (Eq. 64), msg_rm_plate lines
        # 308-330 ported term for term.  V2 energy is O((h/l)^4): A6/G
        # untouched; the columns exist purely for the Eq. 65-66
        # recovery.  TWO variants -- the source must MATCH the
        # recovery's Gamma_l columns: D-chain (detilted) feeds the
        # in-plane stress rows, T-chain (tilted) the 33/23/13 rows.
        wA = np.asarray(blk["w_dof"])[0::3]
        V11bar, V12bar = out["V11bar"], out["V12bar"]
        V11barD = _detilt_cols_2d(V11bar, node_y2, wA)
        V12barD = _detilt_cols_2d(V12bar, node_y2, wA)
        AH1 = lambda M: D_hl1 @ M - D_hl1.T @ M          # noqa: E731
        AH2 = lambda M: D_hl2 @ M - D_hl2.T @ M          # noqa: E731
        D21D = AH1(V11barD) - D_l11 @ V0
        D22D = (AH1(V12barD) + AH2(V11barD)
                - (D_l12 @ V0 + D_l12.T @ V0))
        D23D = AH2(V12barD) - D_l22 @ V0
        D21T = AH1(V11bar) - D_l11 @ V0
        D22T = (AH1(V12bar) + AH2(V11bar)
                - (D_l12 @ V0 + D_l12.T @ V0))
        D23T = AH2(V12bar) - D_l22 @ V0
        V2 = solve_constrained(np.concatenate(
            [D21D, D22D, D23D, D21T, D22T, D23T], axis=1))
        out["V11barD"], out["V12barD"] = V11barD, V12barD
        out["V21"], out["V22"], out["V23"] = (V2[:, :6], V2[:, 6:12],
                                              V2[:, 12:18])
        out["V21t"], out["V22t"], out["V23t"] = (V2[:, 18:24],
                                                 V2[:, 24:30],
                                                 V2[:, 30:36])
    if f_faces is not None:
        # ---- the pressure-driven load ladder (rm_plate_1D lines
        # 332-352, ported term for term).  The KKT <w> = 0 rows absorb
        # the net face force exactly as the 1-D node constraint does,
        # so the pure-Neumann columns are well posed.
        V1L = solve_constrained(np.asarray(f_faces, float))
        V1Lt, V1Lb = V1L[:, 0], V1L[:, 1]

        def v2l(vl):
            """the five second-order load columns of one face
            (q,1 q,2 q,11 q,12 q,22 drivers) -- msg_rm_plate v2l"""
            return solve_constrained(np.stack(
                [(D_hl1 @ vl - D_hl1.T @ vl),
                 (D_hl2 @ vl - D_hl2.T @ vl),
                 -(D_l11 @ vl),
                 -((D_l12 @ vl) + (D_l12.T @ vl)),
                 -(D_l22 @ vl)], axis=1))

        out["V1Lt"], out["V2Lt"] = V1Lt, v2l(V1Lt)
        out["V1Lb"], out["V2Lb"] = V1Lb, v2l(V1Lb)
    return out


# ------------------------------------- beam Timoshenko/KKT drivers (n_model=1)
@jax.jit
def prepare_v1_rhs(V0, Dhl, Dll, Dle_dense, Psi, Dc):
    """RHS of the l-chain V1s solve, projected per Eq. 100 so it is
    orthogonal to the rigid-body kernel.

    In:  V0 (N_primal, 4) fluctuation field of the EB solve; Dhl/Dll
         (N_primal, N_primal) sparse system blocks; Dle_dense
         (N_primal, 4); Psi/Dc (N_primal, 4) rigid-body ops
    Out: bb (N_primal, 4) projected RHS; DhlV0 (N_primal, 4);
         DhlTV0Dle (N_primal, 4) = Dhl.T @ V0 + Dle; V0DllV0 (4, 4)."""
    DhlV0 = Dhl @ V0
    V0DllV0 = V0.T @ (Dll @ V0)
    DhlTV0Dle = Dhl.T @ V0 + Dle_dense
    b_unproj = DhlV0 - DhlTV0Dle

    inv_Dc_T_Psi = jnp.linalg.inv(Psi.T @ Dc)
    tmp_corr_b = inv_Dc_T_Psi @ (Psi.T @ b_unproj)
    bb = (Dc @ tmp_corr_b) - b_unproj
    return bb, DhlV0, DhlTV0Dle, V0DllV0


@jax.jit
def finalize_v1_and_compute_deff(V1s_raw, V0, D_eff, V0DllV0, DhlV0,
                                 DhlTV0Dle, Psi, Dc):
    """Project V1s per Eq. 85 and reduce to the 6x6 Timoshenko stiffness
    [eps11 gam12 gam13 kap1 kap2 kap3].  Q_base maps the 2 shears into
    the 4 classical modes.

    In:  V1s_raw (N_primal, 4) unprojected l-chain solution; V0
         (N_primal, 4) EB fluctuations; D_eff (4, 4) EB stiffness from
         the V0 solve; V0DllV0 (4, 4); DhlV0/DhlTV0Dle (N_primal, 4)
         from prepare_v1_rhs; Psi/Dc (N_primal, 4) rigid-body ops
    Out: Deff_srt (6, 6) Timoshenko stiffness; B_tim (4, 4) coupling
         block; C_tim (4, 4) symmetrized second-order block; V1s
         (N_primal, 4) projected fluctuations."""
    inv_DcT_Psi = jnp.linalg.inv(Dc.T @ Psi)
    tmp_corr_v1 = inv_DcT_Psi @ (Dc.T @ V1s_raw)
    V1s = V1s_raw - (Psi @ tmp_corr_v1)

    Ainv = jnp.linalg.inv(D_eff)

    B_tim = DhlTV0Dle.T @ V0
    C_tim_unsym = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_tim_unsym + C_tim_unsym.T)

    Q_base = jnp.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [1.0, 0.0]
    ], dtype=jnp.float64)
    Q_tim = Ainv @ Q_base

    Ginv = Q_tim.T @ (C_tim - B_tim.T @ Ainv @ B_tim) @ Q_tim
    G_tim = jnp.linalg.inv(Ginv)

    Y_tim = B_tim.T @ Q_tim @ G_tim
    A_tim = D_eff + Y_tim @ Ginv @ Y_tim.T

    Deff_srt = jnp.zeros((6, 6), dtype=jnp.float64)
    Deff_srt = Deff_srt.at[0, 3:6].set(A_tim[0, 1:4])
    Deff_srt = Deff_srt.at[0, 1:3].set(Y_tim[0, :])
    Deff_srt = Deff_srt.at[0, 0].set(A_tim[0, 0])
    Deff_srt = Deff_srt.at[3:6, 3:6].set(A_tim[1:4, 1:4])
    Deff_srt = Deff_srt.at[3:6, 1:3].set(Y_tim[1:4, :])
    Deff_srt = Deff_srt.at[3:6, 0].set(A_tim[1:4, 0])
    Deff_srt = Deff_srt.at[1:3, 1:3].set(G_tim)
    Deff_srt = Deff_srt.at[1:3, 3:6].set(Y_tim.T[:, 1:4])
    Deff_srt = Deff_srt.at[1:3, 0].set(Y_tim.T[:, 0])
    return Deff_srt, B_tim, C_tim, V1s


_TRI_GP = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])   # degree-2 exact
_QUAD_GP = np.array([(a, b) for a in (-1/np.sqrt(3), 1/np.sqrt(3))
                     for b in (-1/np.sqrt(3), 1/np.sqrt(3))])


def mass_matrix_2d(sc, density):
    """6x6 beam mass matrix of a 2-D solid SG (VABS frame, section origin).

    Straight-sided corner-node integration (subparametric: exact for tri3/quad4,
    the geometric approximation of curved tri6/quad8 edges is negligible for a
    mass integral): per element  m = int rho dA,  S2/S3 = int rho x dA,
    I = int rho x x dA, assembled as

        [[ m    0    0    0    S3  -S2 ]
         [ 0    m    0   -S3   0    0  ]
         [ 0    0    m    S2   0    0  ]
         [ 0   -S3   S2  I22+I33  0  0 ]
         [ S3   0    0    0    I22 -I23]
         [-S2   0    0    0   -I23  I33]]

    In:  sc dict (load_sg_input): nodes (V, 3) with (x2, x3) in cols 0:2,
         cells, mat_id (E,) 1-based;
         density: (n_mat,) per material in mat-id order, or {mat_id: rho}.
    Out: (M, info): M (6, 6); info dict mpus / mass_center (2,) /
         i22 / i33 / i23 about the section origin."""
    nd = np.asarray(sc["nodes"], float)[:, :2]
    mats = np.asarray(sc["mat_id"], int)
    if isinstance(density, dict):
        rho_of = {int(k): float(v) for k, v in density.items()}
    else:
        dens = np.asarray(density, float).ravel()
        rho_of = {i + 1: float(dens[i]) for i in range(len(dens))}
    m = S2 = S3 = I22 = I33 = I23 = 0.0
    for c, mid in zip(sc["cells"], mats):
        rho = rho_of[int(mid)]
        nn = len(c)
        if nn in (3, 6):
            X = nd[np.asarray(c[:3], int)]
            J2 = abs((X[1, 0] - X[0, 0]) * (X[2, 1] - X[0, 1])
                     - (X[2, 0] - X[0, 0]) * (X[1, 1] - X[0, 1]))
            for (l1, l2) in _TRI_GP:
                w = J2 / 6.0
                x = X[0] + l1 * (X[1] - X[0]) + l2 * (X[2] - X[0])
                m += rho * w
                S2 += rho * w * x[0]; S3 += rho * w * x[1]
                I22 += rho * w * x[1] ** 2; I33 += rho * w * x[0] ** 2
                I23 += rho * w * x[0] * x[1]
        elif nn in (4, 8, 9):
            X = nd[np.asarray(c[:4], int)]
            for (xi, eta) in _QUAD_GP:
                Nx = 0.25 * np.array([-(1 - eta), (1 - eta),
                                      (1 + eta), -(1 + eta)])
                Ne = 0.25 * np.array([-(1 - xi), -(1 + xi),
                                      (1 + xi), (1 - xi)])
                J = np.array([Nx @ X, Ne @ X])
                w = abs(np.linalg.det(J))
                Nf = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                      (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                x = Nf @ X
                m += rho * w
                S2 += rho * w * x[0]; S3 += rho * w * x[1]
                I22 += rho * w * x[1] ** 2; I33 += rho * w * x[0] ** 2
                I23 += rho * w * x[0] * x[1]
        else:
            raise ValueError("mass_matrix_2d: unsupported 2-D element with "
                             "%d nodes" % nn)
    M = np.array([[m,   0,   0,   0,        S3,  -S2],
                  [0,   m,   0,  -S3,       0,    0],
                  [0,   0,   m,   S2,       0,    0],
                  [0,  -S3,  S2,  I22 + I33, 0,   0],
                  [S3,  0,   0,   0,        I22, -I23],
                  [-S2, 0,   0,   0,       -I23,  I33]])
    info = dict(mpus=m, mass_center=np.array([S2 / m, S3 / m]) if m else
                np.zeros(2), i22=I22, i33=I33, i23=I23)
    return M, info


def _beam_homo_kkt(sc, n_sg, points, cells, x_end, phi_qn, dphi_dxi_qnp,
                   W_q, dof_map_np, material_param, angles, elem_rotation,
                   solver="direct"):
    """The n_model=1 beam driver: V0 EB solve under the 4 rigid-body
    Lagrange constraints, the l-chain V1s solve reusing the factorized
    KKT matrix, then the 6x6 Timoshenko reduction.

    solver="cg" replaces BOTH KKT factorizations with the device-resident
    projected CG (sparse_projected_cg): the rigid-body rows become a
    projection and the 4 V0 + 4 V1 load cases are vmapped -- the
    GPU-compatible path (identical digits by V0/V1 stationarity).

    In:  sc parsed SG dict; n_sg SG dimension (>= 2); points (V, d)
         nodes; cells (E, N) connectivity; x_end (E, N, d) element node
         coords; phi_qn/dphi_dxi_qnp/W_q quadrature basis tables;
         dof_map_np (V*3,) periodic dof map; material_param (n_mat, 9)
         engineering override or None; angles (n_mat,) deg or None;
         elem_rotation (E, 9) per-element DCs or None; solver str
    Out: the plate_homo_2d r dict -- C_eff (6, 6) Timoshenko, C_eff_EB
         (4, 4) classical, V0/V1s, dehomo_data, C_ess/C_stress,
         elem_rotation (identity tile when None), connectivity, omega,
         sc; everything the beam dehom needs rides along."""
    V = points.shape[0]
    cells_np = np.asarray(cells, np.uint64)
    reduced_cells, num_unique, x_unique = compress_periodic_cells_jax(
        V, cells_np, dof_map_np, x_end, ndof_per_node=3)
    N_primal = num_unique * 3

    C_asm, C_stress = get_heterogeneous_C_matrix(sc, material_param,
                                                 angles, elem_rotation)
    (Dhh, Dhe, Dee, Dll, Dhl, Dle, omega,
     dehomo_data) = assemble_system_matrices(
        x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_cells, N_primal,
        C_asm, 1, n_sg)
    Psi, Dc = assemble_rigid_body_ops(x_unique, x_end, dphi_dxi_qnp,
                                      phi_qn, W_q, reduced_cells,
                                      N_primal, n_sg)

    RHS_V0 = -np.array(Dhe.todense())
    if solver == "cg":
        from .sg_assembly import sparse_projected_cg
        Crows = np.asarray(Dc, float).T               # (4, N) rigid-body rows
        V0 = jnp.asarray(sparse_projected_cg(Dhh, Crows, RHS_V0, 3))
        D1_V0 = jnp.einsum("ni,nj->ij", V0, -jnp.asarray(RHS_V0))
    else:
        V0, D1_V0, A_augmented = solve_fluctuation_field(Dhh, RHS_V0, Dc, 1)
    C_eb = (Dee + D1_V0) / omega

    bb, DhlV0, DhlTV0Dle, V0DllV0 = prepare_v1_rhs(
        V0, Dhl, Dll, jnp.array(Dle.todense()), Psi, Dc)
    if solver == "cg":
        from .sg_assembly import sparse_projected_cg
        V1s_raw = jnp.asarray(sparse_projected_cg(Dhh, Crows,
                                                  np.asarray(bb), 3))
    else:
        R_aug = np.concatenate([np.array(bb),
                                np.zeros((4, bb.shape[1]))], axis=0)
        V_aug = _sparse_direct_solve(A_augmented, R_aug, sym=True)
        V1s_raw = jnp.array(V_aug[:N_primal, :])
    C_timo, _B_tim, _C_tim, V1s = finalize_v1_and_compute_deff(
        V1s_raw, V0, C_eb, V0DllV0, DhlV0, DhlTV0Dle, Psi, Dc)

    E = x_end.shape[0]
    if elem_rotation is None:
        er = jnp.tile(jnp.array([1., 0., 0., 0., 1., 0., 0., 0., 1.]),
                      (E, 1))
    else:
        er = jnp.asarray(elem_rotation, float).reshape(E, 9)

    return {"C_eff": np.asarray(C_timo), "C_eff_EB": np.asarray(C_eb),
            "V0": V0, "V1s": V1s, "dehomo_data": dehomo_data,
            "reduced_periodic_cells": reduced_cells,
            "C_ess": C_asm, "C_stress": C_stress, "elem_rotation": er,
            "x_end": x_end, "phi_qn": phi_qn,
            "dphi_dxi_qnp": dphi_dxi_qnp, "W_q": W_q,
            "n_sg": n_sg, "n_model": 1, "omega": float(omega), "sc": sc}


def _to_basix_order(cells, n_sg, nn):
    """gmsh -> basix order for quad4, hex8 and tet10; tri3, tet4 and the
    interval degrees coincide (the interval only because sc_to_yaml swaps
    the raw .sc 5-node order [end,end,25%,75%,50%] at read time).

    tet10: gmsh lists the 6 midsides on edges (12, 23, 13, 14, 34, 24)
    after the 4 corners; basix tetrahedron-P2 orders its edge DOFs
    (34, 24, 23, 14, 13, 12) -- the permutation below.  Without it det J
    changes sign inside every element and nothing raises.

    In:  cells (E, nn) connectivity; n_sg SG dimension; nn nodes/elem
    Out: (E, nn) reordered connectivity (unchanged unless quad4/hex8/
         tet10)."""
    if n_sg == 2 and nn == 4:
        return cells[:, jnp.array([0, 1, 3, 2])]
    if n_sg == 3 and nn == 8:
        return cells[:, jnp.array([0, 1, 3, 2, 4, 5, 7, 6])]
    if n_sg == 3 and nn == 10:
        return cells[:, jnp.array([0, 1, 2, 3, 8, 9, 5, 7, 6, 4])]
    return cells


# ------------------------------------------------------------- the public API
def _knum(v):
    """Format one number in the SwiftComp .K fixed-width style.

    In:  v float
    Out: str, 19-char right-justified '%.7E' mantissa with a signed
         3-digit exponent (e.g. '     1.2345678E+003')."""
    m, e = ("%.7E" % v).split("E")
    return "%19s" % ("%sE%s%03d" % (m, e[0], abs(int(e))))


def write_sc_K(path, C, solve_time=None, model="", constants=True, name="",
               mass=None):
    """Write an effective stiffness in the SwiftComp .K format (as a .out
    file): stiffness, compliance and (3-D solid law only) the orthotropic-
    approximated engineering constants; OpenSG banner at the top, 'Time
    taken' at the bottom.  Units follow the input; solid Voigt order
    [11 22 33 23 13 12], engineering shears.

    In:  path str; C (n, n) effective stiffness; solve_time float | None;
         model str banner text; constants bool (True: 3-D solid only);
         name str matrix title infix -- it MUST be the console law title
         of the macro model, so that ' The Effective <name> Stiffness
         Matrix' in the file reads the same as the terminal:
         'Timoshenko Beam' (beam 6x6), 'Euler-Bernoulli Beam' (beam 4x4),
         'Classical Plate' (ABD 6x6), 'Reissner-Mindlin' (ABDG 8x8),
         'Cauchy Continuum' (3-D solid 6x6);  '' -> 'The Effective
         Stiffness Matrix' (no macro law names itself that -- do not use);
         mass (6, 6) | None -- beam mass matrix, written VABS .K style
         ('The 6X6 Mass Matrix') after the compliance
    Out: the .out file at path; returns None."""
    C = np.asarray(C, float)
    S = np.linalg.inv(C)
    n = C.shape[0]
    infix = (name + " ") if name else ""
    with open(path, "w") as f:
        f.write(" OpenSG %s\n\n" % model)
        f.write(" The Effective %sStiffness Matrix\n"
                " --------------------------------------------\n" % infix)
        for i in range(n):
            f.write("".join(_knum(C[i, j]) for j in range(n)) + "\n")
        f.write("\n The Effective %sCompliance Matrix\n"
                " --------------------------------------------\n" % infix)
        for i in range(n):
            f.write("".join(_knum(S[i, j]) for j in range(n)) + "\n")
        if mass is not None:
            Mm = np.asarray(mass, float)
            f.write("\n The 6X6 Mass Matrix\n"
                    " ========================================================\n\n")
            for i in range(6):
                f.write("".join(_knum(Mm[i, j]) for j in range(6)) + "\n")
        if not constants:
            if solve_time is not None:
                f.write("\n Time taken: %.2f sec\n" % solve_time)
            return
        f.write("\n The Engineering Constants (Approximated as Orthotropic)\n"
                " ----------------------------------------------------------\n")
        for lbl, v in (("E1 ", 1/S[0, 0]), ("E2 ", 1/S[1, 1]),
                       ("E3 ", 1/S[2, 2]), ("G12", 1/S[5, 5]),
                       ("G13", 1/S[4, 4]), ("G23", 1/S[3, 3]),
                       ("nu12", -S[0, 1]/S[0, 0]),
                       ("nu13", -S[0, 2]/S[0, 0]),
                       ("nu23", -S[1, 2]/S[1, 1])):
            f.write("  %-4s=%s\n" % (lbl, _knum(v)))
        if solve_time is not None:
            f.write("\n Time taken: %.2f sec\n" % solve_time)


def plate_homo_2d(sc_path: str,                         # the .sc/.yaml input
                    material_param=None,                # (n_mat, 9) override
                    angles: Optional[Sequence[float]] = None,   # deg/material
                    n_model: Optional[int] = None,      # 1 beam, 2 plate, 3 3D
                    refined: Optional[int] = None,      # 0 classical, 1 shear-refined
                    workdir: Optional[str] = None,      # where .yaml/.msh go
                    elem_rotation=None,                 # (E, 9) per-element DCs
                    solver: str = "direct",             # "direct" | "cg"
                    shear_refined: bool = False,        # legacy alias of model=1
                    plot: bool = True,                  # <base>_mesh.png if absent
                    boundary: Optional[str] = None,     # 'aperiodic'|'periodic'
                    density=None,                       # (n_mat,) beam mass rho
                    omega: Optional[float] = None,      # user SG measure (3-D solid)
                    q_reaction: Optional[str] = None    # 'uniform' | 'tau'
                    ) -> Dict[str, Any]:
    """Homogenize ONE structure gene (1-D/2-D/3-D .sc) to the macro law.

    Out: dict r -- C_eff (6, 6) np [N11 N22 N12 M11 M22 M12] (the plate
    ABD for n_model=2; the Timoshenko 6x6 for n_model=1, with the EB 4x4
    in C_eff_EB), the fluctuation fields, C_ess, x_end, phi_qn,
    dphi_dxi_qnp, W_q, the periodic connectivity, n_sg, n_model, omega,
    sc (the parsed .sc dict).  Everything the dehom needs rides along --
    homogenize once, recover as often as needed (the msg_rm_plate rule).
    r["law"] / r["law_title"] carry the law SELECTED by `refined` (the
    one the .out reports), so a driver needs no branching to print it.
    refined: 0 = classical (plate ABD 6x6 / beam EB 4x4), 1 =
    shear-refined (plate ABDG 8x8 / beam Timoshenko 6x6); None = the
    legacy default (beam Timoshenko; plate classical unless
    shear_refined=True); ignored for n_model=3, whose 6x6 solid law is
    the single option.
    A yaml SG may carry the analysis request in its own header (leading
    scalar keys, see sg_mesh.read_yaml_header): `n_model:` gives the
    macro model (1 beam, 2 plate, 3 solid), `refined:` the 0/1 switch
    and the OPTIONAL `aperiodic: 1` the boundary opt-out (omit the key
    for a periodic SG -- periodic is the default at every dimension);
    those fill any argument left at None, so a self-describing
    file runs as plate_homo_2d("file.yaml") with no arguments --
    explicit arguments always win over the header.  Without header or
    argument the legacy defaults apply (plate; refined as above).
    n_model=1 routes through the Beam_solid KKT engine (n_sg >= 2);
    elem_rotation is consumed by every model (beam, plate and solid).
    solver: "direct" (default; one sparse factorization for all columns)
    or "cg" (the verbatim SSDM Chebyshev-CG pipeline) -- both produce
    the same digits, see _homo_direct.
    The plate model=1 route runs the RM first-order warping ladder
    (plate_shear_ladder) -> r["G_msg"] (2x2), r["ABDG"] (8x8
    [[A6, 0], [0, G]]), r["A6_ladder"]; derivation status per SG
    dimension: see plate_shear_ladder.  `shear_refined=True` is the
    legacy spelling of model=1 (plate only).
    sc_path may also be an ALREADY-PARSED sc dict (laminate_to_sg).
    boundary: 'periodic' (default; ties opposite faces/edges/corners --
    the SwiftComp-parity mode) or 'aperiodic' (explicit request only):
    the boundary-solution treatment mapping the macro/affine field onto
    the boundary, i.e. ZERO fluctuation (w = 0 Dirichlet) on every
    bounding-box-face node.  Aperiodic forbids the rank-one affine fields
    of a free SG, fixes the rigid modes, and is a kinematic upper bound
    on a single cell (stiffer than periodic).
    omega: the USER SG measure, overriding the one measured from the mesh
    (3-D solid macro model only; the yaml header key `omega:` fills it in,
    and an explicit argument wins over the header).  The measured default
    for a 3-D SG is the node bounding box -- the periodic unit cell; pass
    `omega` only when the equivalent continuum occupies something else.
    q_reaction: how the pressure LOAD COLUMN balances the net face force
    (refined 2-D plate SG only).  NOT a yaml key: the yaml describes the
    structure alone, and this choice belongs to the LOADING side -- it
    rides in the `.ff` (`q_reaction: tau`), which the cli reads and
    passes here for a dehomogenization run.  The load column is a
    pure-Neumann cell solve, so the applied face force must be reacted
    somewhere inside the cell:
      'uniform' (DEFAULT, and every result before this key existed) lets
          the <w> = 0 KKT rows absorb it -- mechanically a uniform body
          force over the material area.  Exact for a cell that is
          HOMOGENEOUS IN-PLANE (a laminate), where the reaction is
          uniform anyway; that is the regime the ladder was gated on.
      'tau' reacts it along the cell's OWN transverse-shear path: the
          weight is sigma_xz under a unit Q1, taken from this SG's Eq. 63
          first-order recovery (integral gated to 1).  For an
          in-plane-heterogeneous cell (sandwich core, stiffened panel)
          the spanwise shear really does flow through the webs, and the
          uniform reaction under-loads the face-sheet bays: on the
          HC_pm45 honeycomb it damps the face-sheet bay bending ~20 %
          (compare/four_way/bay_load_check.png).  Costs one extra ladder
          factorization; A6/G/ABDG are unchanged (the load columns feed
          recovery only)."""
    from .sg_mesh import read_yaml_header
    _hdr = read_yaml_header(sc_path) if isinstance(sc_path, str) else {}
    if omega is None and _hdr.get("omega") is not None:
        omega = float(_hdr["omega"])
    omega_user = None if omega is None else float(omega)
    del omega                    # below, `omega` is the MEASURED measure
    if n_model is None:
        n_model = int(_hdr.get("n_model", 2))
    if n_model not in (1, 2, 3):
        raise ValueError("n_model must be 1 (beam), 2 (plate) or 3 (3-D"
                         " solid), got %r" % n_model)
    if shear_refined and n_model != 2:
        raise ValueError("shear_refined applies to the plate model "
                         "(n_model=2)")
    if boundary is None and _hdr.get("aperiodic") is not None:
        # `aperiodic: 1` in the yaml header is the explicit opt-out;
        # absent (or 0) means periodic -- the default for every SG
        boundary = "aperiodic" if int(_hdr["aperiodic"]) else "periodic"
    if refined is None and _hdr.get("refined") is not None:
        refined = int(_hdr["refined"])
    if refined is None:
        refined = 1 if (n_model == 1 or shear_refined) else 0
    elif refined not in (0, 1):
        raise ValueError("refined must be 0 (classical: plate ABD / beam"
                         " EB) or 1 (shear-refined: plate ABDG / beam"
                         " Timoshenko)")
    shear_refined = (n_model == 2 and refined == 1)
    if isinstance(sc_path, dict):
        base = os.path.join(workdir if workdir is not None else ".",
                            "laminate_sg")
    else:
        base = os.path.splitext(sc_path)[0]
        if workdir is not None:
            base = os.path.join(workdir, os.path.basename(base))
    import time as _time
    _t0 = _time.perf_counter()
    sc = load_sg_input(sc_path, base)
    n_sg = sc["dim"]

    def _emit_plot():
        # visualization stays OUT of the timed span.  Same mesh -> same
        # figure, so it is drawn only when the PNG is MISSING or OLDER
        # than the yaml it depicts: a re-converted SG (new mesh, new
        # material binding) must not keep showing the previous picture.
        png = base + "_mesh.png"
        if not plot:
            return
        stale = True
        if os.path.exists(png):
            stale = (isinstance(sc_path, str) and os.path.exists(sc_path)
                     and os.path.getmtime(png) < os.path.getmtime(sc_path))
        if stale:
            plot_sg_mesh(sc, png, msh_path=base + ".msh")

    nn = sorted({len(c) for c in sc["cells"]})
    mixed = len(nn) != 1
    if mixed and n_model == 1:
        raise ValueError("mixed element types: the beam (n_model 1) "
                         "KKT route is single-batch; split the mesh or "
                         "use one element type: %s nodes/elem" % nn)
    if mixed and solver == "cg":
        raise ValueError("mixed element types need solver='direct' "
                         "(the EBE/Chebyshev CG operators are "
                         "single-batch): %s nodes/elem" % nn)
    points = jnp.array(np.asarray(sc["nodes"], float)[:, 0:n_sg])
    cell_domain_ids = jnp.array(np.asarray(sc["mat_id"], int) - 1)
    V = points.shape[0]

    from opensg_solid.sg_mixed import (attach_periodic_maps,
                                       build_batches, fe_tables,
                                       homo_direct_batched)
    if not mixed:
        cells = jnp.array(np.asarray(sc["cells"], np.uint64))
        cells = _to_basix_order(cells, n_sg, nn[0])
        phi_qn, dphi_dxi_qnp, W_q = fe_tables(n_sg, nn[0])
        x_end = mesh_to_jax(vertices=points, cells=cells)
    else:
        # per-element-type batches over the SHARED nodes-x-3 dof space:
        # the batching layer lives in sg_mixed; element ORDER within a
        # batch follows the global mesh order, so mat_id/C_ess split by
        # the same index lists
        bat = build_batches(sc, points, n_sg)

    # boundary defaults to periodic on every route; aperiodic only on request
    if boundary is None:
        boundary = "periodic"
    if boundary == "periodic":
        # the periodic NODE reduction inside the map is built from the
        # points alone (cells only route the reduced ids), so per-batch
        # calls share one consistent reduced numbering
        if not mixed:
            periodic_cells_en, dof_map_np = \
                mesh_to_periodic_sparse_assembly_map(
                    V, cells, points, n_model, atol=1e-6)
        else:
            dof_map_np = attach_periodic_maps(bat, V, points, n_model,
                                              boundary)
        bdofs = None
    else:
        if n_model == 1 or solver == "cg":
            raise ValueError("boundary='aperiodic' supports the plate/solid "
                             "models with solver='direct'")
        # boundary solution mapped to the boundary nodes: zero fluctuation
        # (w = 0, Dirichlet) on every bounding-box-face node of the SG
        if not mixed:
            periodic_cells_en, dof_map_np = cells, np.arange(V*3)
        else:
            dof_map_np = attach_periodic_maps(bat, V, points, n_model,
                                              boundary)
        pts = np.asarray(points)
        box0, box1 = pts.min(0), pts.max(0)
        tol = 1e-6*float(np.max(box1 - box0))
        onb = ((np.abs(pts - box0) < tol) | (np.abs(pts - box1) < tol)).any(1)
        bdofs = (3*np.where(onb)[0][:, None] + np.arange(3)).ravel()

    if n_model == 1:
        r = _beam_homo_kkt(sc, n_sg, points, cells, x_end, phi_qn,
                           dphi_dxi_qnp, W_q, dof_map_np,
                           material_param, angles, elem_rotation,
                           solver=solver)
        r["solve_time"] = _time.perf_counter() - _t0
        # 6x6 mass matrix (VABS .K style): rho per material from `density`,
        # falling back to the material blocks' own `density:` key (the
        # user-authored yaml route) and then to the .sc T/rho line (aux[1])
        if density is None:
            aux_rho = [float(m.get("density", m.get("aux", [0, 0])[1]))
                       for _, m in sorted(sc["materials"].items())]
            density = aux_rho if any(r_ > 0 for r_ in aux_rho) else None
        M6 = None
        if density is not None:
            M6, minfo = mass_matrix_2d(sc, np.asarray(density, float))
            r["Mass"] = M6; r["mass_info"] = minfo
        if refined == 0:
            r["law"] = np.asarray(r["C_eff_EB"])
            r["law_title"] = ("Euler-Bernoulli Beam Stiffness Matrix  "
                              "[eps11 kappa1 kappa2 kappa3]")
            _nm = "Euler-Bernoulli Beam"
        else:
            r["law"] = np.asarray(r["C_eff"])
            r["law_title"] = ("Timoshenko Beam Stiffness Matrix  "
                              "[eps11 gam12 gam13 kappa1 kappa2 kappa3]")
            _nm = "Timoshenko Beam"
        write_sc_K(base + ".out", r["law"],
                   solve_time=r["solve_time"],
                   model="msg-solid beam model, omega %.8g"
                         % float(r["omega"]),
                   constants=False, name=_nm, mass=M6)
        _emit_plot()
        return r

    # per-element frames must be applied for plate/solid too, not only beam;
    # elem_rotation rows must be in the solver's global order -- see
    # sg_materials.elem_rotation_from_yaml.
    C_ess, _ = get_heterogeneous_C_matrix(sc, material_param, angles,
                                          elem_rotation)
    if mixed:
        C_all = jnp.asarray(C_ess)
        for b in bat:
            b["C_ess"] = C_all[jnp.asarray(b["idx"])]

    unique_dofs = jnp.unique(dof_map_np)
    n_unique = len(unique_dofs)

    u_0_g_full = jnp.zeros(shape=(V * 3))
    if mixed:
        C_eff, V0_matrix, omega = homo_direct_batched(
            bat, u_0_g_full, unique_dofs, n_unique, points, n_model,
            n_sg, bdofs=bdofs)
    elif solver == "direct":
        C_eff, V0_matrix, omega = _homo_direct(
            x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q, C_ess,
            periodic_cells_en, unique_dofs, n_unique, n_model, n_sg,
            bdofs=bdofs)
    else:
        C_eff, V0_matrix, omega = full_homogenization_pipeline(
            x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q, C_ess,
            periodic_cells_en, unique_dofs, n_unique, n_model, n_sg)

    if omega_user is not None and n_model == 3:
        # the user measure WINS over the measured one; C_eff is exactly
        # linear in 1/omega, so rescaling here is the same as having
        # divided by omega_user inside the assembly
        C_eff = np.asarray(C_eff) * (float(omega) / omega_user)
        omega = omega_user

    if not mixed:
        r = {"C_eff": np.asarray(C_eff), "V0": V0_matrix, "C_ess": C_ess,
             "x_end": x_end, "phi_qn": phi_qn,
             "dphi_dxi_qnp": dphi_dxi_qnp,
             "W_q": W_q, "periodic_cells_en": periodic_cells_en,
             "n_sg": n_sg, "n_model": n_model, "omega": float(omega),
             "boundary": boundary,
             "n_boundary_nodes": 0 if bdofs is None else len(bdofs)//3,
             "sc": sc}
    else:
        # the per-mesh arrays become PER-BATCH LISTS (same batch order
        # everywhere); sg_dehom loops them and concatenates flat clouds
        r = {"C_eff": np.asarray(C_eff), "V0": V0_matrix,
             "C_ess": [b["C_ess"] for b in bat],
             "x_end": [b["x_end"] for b in bat],
             "phi_qn": [b["phi_qn"] for b in bat],
             "dphi_dxi_qnp": [b["dphi_dxi_qnp"] for b in bat],
             "W_q": [b["W_q"] for b in bat],
             "periodic_cells_en": [b["periodic_cells"] for b in bat],
             "batch_nn": [b["nn"] for b in bat],
             "batch_idx": [np.asarray(b["idx"]) for b in bat],
             "n_sg": n_sg, "n_model": n_model, "omega": float(omega),
             "boundary": boundary,
             "n_boundary_nodes": 0 if bdofs is None else len(bdofs)//3,
             "sc": sc}

    if shear_refined:
        # (the model/shear_refined guard lives at the top of the function)
        # the l-blocks integrate basis VALUE products -> degree 2p+2
        #
        # consistent nodal loads of a UNIT face pressure on the top and
        # bottom SG faces, for the qt6/qb6 load-recovery ladder.  Signs
        # follow msg_rm_plate's Lt/Lb (-1 top w3, +1 bottom w3): a
        # positive pressure pushes INTO each face.  /omega matches the
        # ladder_blocks normalization.  Single-batch SGs of ANY
        # dimension: 1-D faces are the two end NODES (weight 1, omega =
        # 1 -- exactly rm_plate_1D's Lt/Lb), 2-D faces are the top and
        # bottom EDGES (trapezoid weights = the exact consistent load
        # of linear edges on a flat face), 3-D faces are the two
        # thickness FACES (_plate_face_loads_3d facet integrals).
        # Periodic in-plane nodes scatter to one master dof, which is
        # exactly the tiled-face integral.
        f_faces = None
        node_y2 = None
        if not mixed:
            pts2 = np.asarray(points)[:, :n_sg]
            red_of = np.full(pts2.shape[0], -1, dtype=np.int64)
            red_of[np.asarray(cells, dtype=np.int64).ravel()] = \
                np.asarray(periodic_cells_en).ravel()
            # reduced-node THICKNESS coordinate (always the LAST SG
            # coordinate) for the V2 detilt -- periodic partners share
            # it, so scatter order is immaterial
            thick = pts2[:, n_sg - 1]
            node_y2 = np.zeros(n_unique // 3)
            node_y2[red_of] = thick
            ftol = 1e-6 * float(max(np.ptp(pts2[:, k])
                                    for k in range(n_sg)))
            f_faces = np.zeros((n_unique, 2))
            # SIGNS: solve_constrained negates its rhs internally, so
            # the stored columns are the NEGATIVE of the physical load
            # (top +, bottom -); calibrated against the flat-laminate
            # closed-form sigma33 profile (-q at the top face, 0 at
            # the bottom).
            for col, (yf, sgn) in enumerate(
                    ((thick.max(), 1.0), (thick.min(), -1.0))):
                if n_sg == 3:
                    nid, wgt = _plate_face_loads_3d(
                        pts2, np.asarray(cells, dtype=np.int64), yf,
                        ftol)
                else:
                    nid = np.where(np.abs(thick - yf) < ftol)[0]
                    if n_sg == 1:
                        wgt = np.ones(len(nid))     # the face is a node
                    else:
                        nid = nid[np.argsort(pts2[nid, 0])]
                        seg = np.diff(pts2[nid, 0])
                        wgt = np.zeros(len(nid))
                        wgt[:-1] += 0.5 * seg
                        wgt[1:] += 0.5 * seg
                np.add.at(f_faces[:, col], 3 * red_of[nid] + 2,
                          sgn * wgt / float(omega))
        if not mixed:
            phi_hi, dphi_hi, W_hi = fe_tables(n_sg, nn[0], hi=True)
            lad = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess,
                                     periodic_cells_en, n_unique, n_sg,
                                     float(omega), f_faces=f_faces,
                                     node_y2=node_y2)
        else:
            hi = [fe_tables(n_sg, b["nn"], hi=True) for b in bat]
            lad = plate_shear_ladder(
                [b["x_end"] for b in bat],
                [h[1] for h in hi], [h[0] for h in hi],
                [h[2] for h in hi],
                [b["C_ess"] for b in bat],
                [b["periodic_cells"] for b in bat],
                n_unique, n_sg, float(omega))
        r["G_msg"] = lad["G_msg"]
        r["X_shear"] = lad["X"]
        r["A6_ladder"] = lad["A6"]
        r["Ustar_rel"] = lad["Ustar_rel"]
        # the ladder triple, in ITS OWN <w> = 0 gauge -- what the Eq. 63
        # refined recovery consumes (sg_dehom).  The classical r["V0"]
        # (pinned-node gauge) cannot feed the Gamma_l VALUE operators.
        r["V0_ladder"] = lad["V0"]
        r["V11"] = lad["V11"]
        r["V12"] = lad["V12"]
        r["V11bar"] = lad["V11bar"]
        r["V12bar"] = lad["V12bar"]
        # the pressure-driven load columns (qt6/qb6 recovery) and the
        # second-order Eq. 64 chains, when the SG shape supports them
        # (2-D, single batch)
        for k in ("V1Lt", "V2Lt", "V1Lb", "V2Lb", "V11barD", "V12barD",
                  "V21", "V22", "V23", "V21t", "V22t", "V23t"):
            if k in lad:
                r[k] = lad[k]
        # optional tau reaction of the load columns -- see the q_reaction
        # docstring (the flag arrives from the .ff via the cli, never
        # from the yaml).  A6/G above are already final: the load columns
        # feed the recovery only, so this re-solves just that block.
        if q_reaction is None:
            q_reaction = "uniform"
        q_reaction = str(q_reaction).strip().lower()
        if q_reaction not in ("uniform", "tau"):
            raise ValueError("q_reaction must be 'uniform' (the <w> = 0"
                             " KKT reaction) or 'tau' (react along the"
                             " cell's shear path), got %r" % q_reaction)
        r["q_reaction"] = q_reaction
        if (q_reaction == "tau" and f_faces is not None
                and lad.get("V1Lt") is not None):
            from .sg_dehom import _v2_batch
            dE1u = np.linalg.solve(np.asarray(r["C_eff"], float),
                                   np.array([0.0, 0, 0, 1.0, 0, 0]))
            _, dSig = _v2_batch(periodic_cells_en,
                                jnp.asarray(lad["V0"]),
                                jnp.asarray(lad["V11bar"]),
                                jnp.asarray(lad["V12bar"]),
                                x_end, C_ess, dphi_dxi_qnp, phi_qn,
                                jnp.asarray(dE1u), jnp.zeros(6), n_sg)
            tau_e = np.asarray(dSig).mean(axis=1)[:, 4]      # sigma_xz
            if n_sg == 2 and nn[0] == 4:
                # the original quad4 shoelace route, kept verbatim so
                # existing 2-D results stay digit-identical
                cyc = np.asarray(cells, dtype=np.int64)[:, [0, 1, 3, 2]]
                xq, yq = pts2[cyc][:, :, 0], pts2[cyc][:, :, 1]
                area_e = 0.5 * np.abs(
                    (xq * np.roll(yq, -1, axis=1)
                     - np.roll(xq, -1, axis=1) * yq).sum(axis=1))
                scat = cyc
            else:
                # any other single-batch SG: the element measure from
                # the quadrature itself (|det J| . W), the same measure
                # the assembly integrates with
                Je = np.einsum("end,qnp->eqdp", np.asarray(x_end),
                               np.asarray(dphi_dxi_qnp))
                if n_sg == 1:
                    dJ = np.abs(Je[..., 0, 0])
                else:
                    dJ = np.abs(np.linalg.det(Je))
                area_e = dJ @ np.asarray(W_q)
                scat = np.asarray(cells, dtype=np.int64)
            I_tau = float((tau_e * area_e).sum() / float(omega))
            tau_w = np.zeros(n_unique // 3)
            np.add.at(tau_w, red_of[scat.ravel()],
                      np.repeat(tau_e * area_e / scat.shape[1],
                                scat.shape[1]))
            fv = f_faces.copy()
            for col in (0, 1):
                fv[2::3, col] -= (fv[2::3, col].sum()
                                  * tau_w / tau_w.sum())
            lad_q = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi,
                                       C_ess, periodic_cells_en,
                                       n_unique, n_sg, float(omega),
                                       f_faces=fv, node_y2=node_y2)
            for k in ("V1Lt", "V2Lt", "V1Lb", "V2Lb"):
                r[k] = lad_q[k]
            print("q_reaction: tau -- load columns re-reacted along the"
                  " cell's shear path (integral sigma_xz under unit Q1"
                  " = %.6f, target 1)" % I_tau)
        elif q_reaction == "tau":
            print("note: q_reaction: tau needs a single-batch plate SG"
                  " with the load ladder (mixed SGs do not carry it"
                  " yet) -- keeping the uniform reaction")
            r["q_reaction"] = "uniform"
        r["ABDG"] = None
        if lad["G_msg"] is not None:
            ABDG = np.zeros((8, 8))
            ABDG[:6, :6] = lad["A6"]
            ABDG[6:, 6:] = lad["G_msg"]
            r["ABDG"] = ABDG
    # every OpenSG run writes its timed .out by default (constants: 3-D only)
    r["solve_time"] = _time.perf_counter() - _t0
    _mdl = {2: "msg-solid plate model",
            3: "msg-solid 3D elastic model"}.get(n_model, "msg-solid")
    _bc = ("periodic" if r.get("boundary", "periodic") == "periodic"
           else "aperiodic: w=0 on %d boundary nodes"
                % r["n_boundary_nodes"])
    if n_model == 2 and shear_refined and r.get("ABDG") is not None:
        r["law"] = np.asarray(r["ABDG"])
        r["law_title"] = ("Reissner-Mindlin Stiffness Matrix  "
                          "[N11 N22 N12 M11 M22 M12 Q1 Q2]")
        write_sc_K(base + ".out", r["law"],
                   solve_time=r["solve_time"],
                   model="%s, omega %.8g, %s" % (_mdl, float(r["omega"]),
                                                 _bc),
                   constants=False, name="Reissner-Mindlin")
    else:
        r["law"] = np.asarray(r["C_eff"])
        r["law_title"] = ("Classical Plate Stiffness Matrix  "
                          "[N11 N22 N12 M11 M22 M12]" if n_model == 2 else
                          "Cauchy Continuum Stiffness Matrix  "
                          "[11 22 33 23 13 12]")
        write_sc_K(base + ".out", r["law"],
                   solve_time=r["solve_time"],
                   model="%s, omega %.8g, %s" % (_mdl, float(r["omega"]),
                                                 _bc),
                   constants=(n_model == 3),
                   name="Classical Plate" if n_model == 2
                        else "Cauchy Continuum")
    _emit_plot()
    return r
