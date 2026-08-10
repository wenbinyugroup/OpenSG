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
         relative LS residual."""
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
    return G, X, ev_min, Ustar_rel


def plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess,
                       reduced_cells, n_unique, n_sg, omega):
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
    The shear-refined RECOVERY (Eq. 63-66 ladder) is not ported --
    plate dehom stays the classical Kirchhoff-mode recovery.

    In:  x_end/dphi_hi/phi_hi/W_hi -- geometry + a quadrature exact to
         degree 2p+2; C_ess (E, 6, 6); reduced_cells (E, N); n_unique
         dofs; n_sg; omega -- the IN-PLANE SG measure (1-D SG: 1)
    Out: dict A6 (6, 6), G_msg (2, 2, None unless X SPD), X, ev_min,
         Ustar_rel, V0/V11/V12 (n_unique, 6)."""
    out = plate_ladder_element_blocks(x_end, dphi_hi, phi_hi, W_hi,
                                      jnp.asarray(C_ess), n_sg)
    (hh_b, he_b, ee_b, hl1_b, hl2_b, l11_b, l12_b, l22_b,
     l1e_b, l2e_b, wN_b) = [np.asarray(o) for o in out]

    E_elem, n_ed = he_b.shape[0], he_b.shape[1]
    N_nodes = n_ed // 3
    dof_map = ((np.asarray(reduced_cells, dtype=np.int64) * 3)
               .reshape(E_elem, N_nodes, 1)
               + np.arange(3)).reshape(E_elem, n_ed)
    rows = np.repeat(dof_map, n_ed, axis=1).ravel()
    cols = np.tile(dof_map, (1, n_ed)).ravel()

    def nn(Be):
        return csr_matrix((Be.ravel() / omega, (rows, cols)),
                          shape=(n_unique, n_unique))

    def ns(Be):
        M = np.zeros((n_unique, 6))
        np.add.at(M, dof_map.ravel(),
                  Be.reshape(-1, 6) / omega)
        return M

    D_hh = nn(hh_b); D_hl1 = nn(hl1_b); D_hl2 = nn(hl2_b)
    D_l11 = nn(l11_b); D_l12 = nn(l12_b); D_l22 = nn(l22_b)
    D_he = ns(he_b); D_l1e = ns(l1e_b); D_l2e = ns(l2e_b)
    D_ee = ee_b.sum(axis=0) / omega

    w_node = np.zeros(n_unique // 3)
    np.add.at(w_node, np.asarray(reduced_cells,
                                 dtype=np.int64).ravel(), wN_b.ravel())
    w_dof = np.repeat(w_node, 3)

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

    G, X, ev_min, Ustar_rel = _rm_ls_reduction(A6, H11, H12, H22, S1, S2)
    return {"A6": A6, "G_msg": (G if ev_min > 0 else None), "X": X,
            "ev_min": ev_min, "Ustar_rel": float(Ustar_rel),
            "V0": V0, "V11": V11, "V12": V12}


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
    """gmsh/.sc -> basix (tensor) vertex order for quad4 and hex8; tri,
    tet and interval orders coincide.

    In:  cells (E, nn) connectivity; n_sg SG dimension; nn nodes/elem
    Out: (E, nn) reordered connectivity (unchanged unless quad4/hex8)."""
    if n_sg == 2 and nn == 4:
        return cells[:, jnp.array([0, 1, 3, 2])]
    if n_sg == 3 and nn == 8:
        return cells[:, jnp.array([0, 1, 3, 2, 4, 5, 7, 6])]
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
         name str matrix title infix -- '' -> 'The Effective Stiffness
         Matrix'; 'Timoshenko' (beam 6x6), 'Classical Plate' (ABD 6x6),
         'Reissner-Mindlin Plate' (ABDG 8x8);
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
                    n_model: int = 2,                   # 1 beam, 2 plate, 3 3D
                    workdir: Optional[str] = None,      # where .yaml/.msh go
                    elem_rotation=None,                 # (E, 9) per-element DCs
                    solver: str = "direct",             # "direct" | "cg"
                    shear_refined: bool = False,        # + RM G 2x2 (n_model=2)
                    plot: bool = True,                  # write <base>_mesh.png
                    boundary: Optional[str] = None,     # 'aperiodic'|'periodic'
                    density=None                        # (n_mat,) beam mass rho
                    ) -> Dict[str, Any]:
    """Homogenize ONE structure gene (1-D/2-D/3-D .sc) to the macro law.

    Out: dict r -- C_eff (6, 6) np [N11 N22 N12 M11 M22 M12] (the plate
    ABD for n_model=2; the Timoshenko 6x6 for n_model=1, with the EB 4x4
    in C_eff_EB), the fluctuation fields, C_ess, x_end, phi_qn,
    dphi_dxi_qnp, W_q, the periodic connectivity, n_sg, n_model, omega,
    sc (the parsed .sc dict).  Everything the dehom needs rides along --
    homogenize once, recover as often as needed (the msg_rm_plate rule).
    n_model=1 routes through the Beam_solid KKT engine (n_sg >= 2);
    elem_rotation is consumed by every model (beam, plate and solid).
    solver: "direct" (default; one sparse factorization for all columns)
    or "cg" (the verbatim SSDM Chebyshev-CG pipeline) -- both produce
    the same digits, see _homo_direct.
    shear_refined=True (plate only) additionally runs the RM first-order
    warping ladder (plate_shear_ladder) -> r["G_msg"] (2x2), r["ABDG"]
    (8x8 [[A6, 0], [0, G]]), r["A6_ladder"]; derivation status per SG
    dimension: see plate_shear_ladder.
    sc_path may also be an ALREADY-PARSED sc dict (laminate_to_sg).
    boundary: 'periodic' (default; ties opposite faces/edges/corners --
    the SwiftComp-parity mode) or 'aperiodic' (explicit request only):
    the boundary-solution treatment mapping the macro/affine field onto
    the boundary, i.e. ZERO fluctuation (w = 0 Dirichlet) on every
    bounding-box-face node.  Aperiodic forbids the rank-one affine fields
    of a free SG, fixes the rigid modes, and is a kinematic upper bound
    on a single cell (stiffer than periodic)."""
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
    if plot:
        plot_sg_mesh(sc, base + "_mesh.png", msh_path=base + ".msh")

    nn = sorted({len(c) for c in sc["cells"]})
    if len(nn) != 1:
        raise ValueError("mixed element types in one SG are unsupported "
                         "(single homogeneous batch): %s nodes/elem" % nn)
    points = jnp.array(np.asarray(sc["nodes"], float)[:, 0:n_sg])
    cells = jnp.array(np.asarray(sc["cells"], np.uint64))
    cells = _to_basix_order(cells, n_sg, nn[0])
    cell_domain_ids = jnp.array(np.asarray(sc["mat_id"], int) - 1)

    ctype, bdeg = _cell_basis(n_sg, nn[0])
    # enum member names differ across fe_jax vintages (default vs Default)
    _qt = getattr(QuadratureType, "default", None) or getattr(
        QuadratureType, "Default")
    fe_type = FiniteElementType(
        cell_type=ctype, family=ElementFamily.P, basis_degree=bdeg,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=_qt, quadrature_degree=6 if bdeg > 1 else 2)
    xi_qp, W_q = get_quadrature(fe_type=fe_type)
    phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type,
                                                      xi_qp=xi_qp)
    x_end = mesh_to_jax(vertices=points, cells=cells)
    V = points.shape[0]

    # boundary defaults to periodic on every route; aperiodic only on request
    if boundary is None:
        boundary = "periodic"
    if boundary == "periodic":
        periodic_cells_en, dof_map_np = mesh_to_periodic_sparse_assembly_map(
            V, cells, points, n_model, atol=1e-6)
        bdofs = None
    else:
        if n_model == 1 or solver == "cg":
            raise ValueError("boundary='aperiodic' supports the plate/solid "
                             "models with solver='direct'")
        # boundary solution mapped to the boundary nodes: zero fluctuation
        # (w = 0, Dirichlet) on every bounding-box-face node of the SG
        periodic_cells_en, dof_map_np = cells, np.arange(V*3)
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
        # falling back to the .sc material blocks' T/rho line (aux[1])
        if density is None:
            aux_rho = [float(sc["materials"][mid].get("aux", [0, 0])[1])
                       for mid in sorted(sc["materials"])]
            density = aux_rho if any(r_ > 0 for r_ in aux_rho) else None
        M6 = None
        if density is not None:
            M6, minfo = mass_matrix_2d(sc, np.asarray(density, float))
            r["Mass"] = M6; r["mass_info"] = minfo
        write_sc_K(base + ".out", np.asarray(r["C_eff"]),
                   solve_time=r["solve_time"],
                   model="msg-solid beam model, omega %.8g"
                         % float(r["omega"]),
                   constants=False, name="Timoshenko", mass=M6)
        return r

    # per-element frames must be applied for plate/solid too, not only beam;
    # elem_rotation rows must be in the solver's global order -- see
    # sg_materials.elem_rotation_from_yaml.
    C_ess, _ = get_heterogeneous_C_matrix(sc, material_param, angles,
                                          elem_rotation)

    unique_dofs = jnp.unique(dof_map_np)
    n_unique = len(unique_dofs)

    u_0_g_full = jnp.zeros(shape=(V * 3))
    if solver == "direct":
        C_eff, V0_matrix, omega = _homo_direct(
            x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q, C_ess,
            periodic_cells_en, unique_dofs, n_unique, n_model, n_sg,
            bdofs=bdofs)
    else:
        C_eff, V0_matrix, omega = full_homogenization_pipeline(
            x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q, C_ess,
            periodic_cells_en, unique_dofs, n_unique, n_model, n_sg)

    r = {"C_eff": np.asarray(C_eff), "V0": V0_matrix, "C_ess": C_ess,
         "x_end": x_end, "phi_qn": phi_qn, "dphi_dxi_qnp": dphi_dxi_qnp,
         "W_q": W_q, "periodic_cells_en": periodic_cells_en,
         "n_sg": n_sg, "n_model": n_model, "omega": float(omega),
         "boundary": boundary,
         "n_boundary_nodes": 0 if bdofs is None else len(bdofs)//3,
         "sc": sc}

    if shear_refined:
        if n_model != 2:
            raise ValueError("shear_refined applies to the plate model "
                             "(n_model=2)")
        # the l-blocks integrate basis VALUE products -> degree 2p+2
        fe_hi = FiniteElementType(
            cell_type=ctype, family=ElementFamily.P, basis_degree=bdeg,
            lagrange_variant=LagrangeVariant.equispaced,
            quadrature_type=_qt, quadrature_degree=2 * bdeg + 2)
        xi_hi, W_hi = get_quadrature(fe_type=fe_hi)
        phi_hi, dphi_hi = eval_basis_and_derivatives(fe_type=fe_hi,
                                                     xi_qp=xi_hi)
        lad = plate_shear_ladder(x_end, dphi_hi, phi_hi, W_hi, C_ess,
                                 periodic_cells_en, n_unique, n_sg,
                                 float(omega))
        r["G_msg"] = lad["G_msg"]
        r["X_shear"] = lad["X"]
        r["A6_ladder"] = lad["A6"]
        r["Ustar_rel"] = lad["Ustar_rel"]
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
        write_sc_K(base + ".out", np.asarray(r["ABDG"]),
                   solve_time=r["solve_time"],
                   model="%s, omega %.8g, %s" % (_mdl, float(r["omega"]),
                                                 _bc),
                   constants=False, name="Reissner-Mindlin Plate")
    else:
        write_sc_K(base + ".out", np.asarray(r["C_eff"]),
                   solve_time=r["solve_time"],
                   model="%s, omega %.8g, %s" % (_mdl, float(r["omega"]),
                                                 _bc),
                   constants=(n_model == 3),
                   name="Classical Plate" if n_model == 2 else "")
    return r
