"""sg_assembly.py -- element kernels, periodic assembly, preconditioners
and linear solvers of the general SG engine.

PROVENANCE (plate/solid, n_model 2/3).  _element_residual_single_case,
calculate_RHS_and_Ke_batch_periodic, ebe_jacobian_product_periodic,
apply_block_precond, compute_block_inv_diag, estimate_max_eigenvalue,
apply_chebyshev_precond, compute_homogenized_constants are VERBATIM
from the working SSDM plate model (Dehom_plate_plots_SSDM.py) --
globals lifted into arguments, nothing else.

PROVENANCE (beam, n_model 1).  assemble_system_matrices (energy_hh/he/ee
+ the l-chain energy_ll/hl/le, sparse Dhh),
build_lagrange_constraint_matrix, compress_periodic_cells_jax,
assemble_rigid_body_ops, solve_fluctuation_field (augmented KKT) are
lifted from the OneDrive Beam_solid.py (Claude_code/Beam_solid.py;
adaptations noted per function: repo material interface upstream,
pypardiso with scipy SuperLU fallback, (y2, y3) = the LAST two SG
coordinates in build_lagrange_constraint_matrix).

THE FE CORE.  The periodic maps and scatter/gather transforms come from
the legacy fe_jax core, resolved by opensg_solid/__init__ via
$FE_JAX_CORE (default ~/OpenSG_2.0).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE  (axis suffixes: e element, n elem-node,
# q quad point, d spatial dim, p parametric dim, s Voigt-6, H macro mode)
# ----------------------------------------------------------------------------
# inputs
#   x_end               (E, N, d) element-node coordinates
#   dphi_dxi_qnp        (Q, N, p) parametric basis gradients
#   phi_qn              (Q, N) basis values;  W_q (Q,) quadrature weights
#   C_ess / C_in        (E, 6, 6) / (6, 6) per-element stiffness
#   periodic_cells, reduced_periodic_cells
#                       (E, N) master-node (periodic) connectivity
#   u_g_flat            (n_unique,) unraveled dof vector (zero seed)
#   n_unique / N_primal the solve size: unique periodic dofs (3/node)
#   dof_map             (V*3,) periodic dof map (compress input)
#   n_model, n_sg       static: 1 beam / 2 plate / 3 solid; SG dimension
# working (SSDM kernels)
#   J_qdp, G_qpd        (Q, d, p) Jacobian and inverse; det_J_q (Q,)
#   dphi_dx_qnd/_3D     (Q, N, d->3) physical gradients, zero-padded
#   dx, dy, dz          (Q, N) gradient rows;  eps_qs, stress (Q, 6)
#   mask_ones_*/mask_y2/mask_y3_*   the macro-mode masks building Ge
#   Ge                  (Q, 6, H) macro strain map (H = 4 beam / 6 else)
#   R_*, R_nodes_cases  element residuals;  K_elem (N*3, N*3) via jacfwd
#   rhs_matrix / Dhe    (n_unique, H) case RHS;  J_uu_batch (E, N*3, N*3)
#   inv_blocks          (nodes, 3, 3) block-Jacobi inverse diagonals
#   eig_max, eig_min, theta   power-method bounds + Chebyshev shifts
#   D_bar, omega        (H, H) direct Ge^T C Ge integral; SG measure
# working (beam KKT)
#   D_hh_e..D_le_e      element bilinear forms from AD of the energies
#   Dhh, Dll, Dhl       sparse COO (N_primal, N_primal); Dhe, Dle
#                       (N_primal, 4); Dee (4, 4)
#   dehomo_data         {dphi_dx (E,Q,N,d), Ge (E,Q,6,H), dV_q (E,Q)}
#   Psi, Dc             (N_primal, 4) rigid-body kernel + constraint matrix
#   C_ett, C_assembled  build_lagrange_constraint_matrix element/global rows
#   aug_row/col/data    COO streams of the augmented KKT matrix
#   A_augmented         csr (N_primal+4, N_primal+4); R_aug the padded RHS
#   x_unique            (n_masters, d) unique-master coordinates
# outputs
#   V0                  (N_primal, cases) fluctuation solution
#   D1                  (cases, cases) = -(V0^T RHS)
# ----------------------------------------------------------------------------
"""
from functools import partial

import numpy as np

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from scipy.sparse import csr_matrix

from fe_jax.setup import (transform_global_unraveled_to_element_node,
                          transform_element_node_to_global_unraveled_sum)


# ------------------------------------------------- element kernels (VERBATIM)
@jax.jit
def _element_residual_single_case(
    u_nd: jnp.ndarray, x_nd: jnp.ndarray, dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray, W_q: jnp.ndarray, C_ss: jnp.ndarray,
    epsilon_bar: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
):
    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    G_qpd = jnp.linalg.inv(J_qdp)
    det_J_q = jnp.linalg.det(J_qdp)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

    Q, N, _ = dphi_dx_qnd.shape
    zeros = jnp.zeros((Q, N), dtype=dphi_dx_qnd.dtype)
    if x_nd.shape[1] == 1:      # 1D SG
        dphi_dx_3D = jnp.stack([zeros, zeros, dphi_dx_qnd[..., 0]], axis=-1)
    elif x_nd.shape[1] == 2:    # 2D SG
        dphi_dx_3D = jnp.stack([zeros, dphi_dx_qnd[..., 0],
                                dphi_dx_qnd[..., 1]], axis=-1)
    else:
        dphi_dx_3D = dphi_dx_qnd
    dx, dy, dz = dphi_dx_3D[..., 0], dphi_dx_3D[..., 1], dphi_dx_3D[..., 2]
    u_x, u_y, u_z = u_nd[:, 0], u_nd[:, 1], u_nd[:, 2]
    eps_xx, eps_yy, eps_zz = dx @ u_x, dy @ u_y, dz @ u_z
    eps_yz = (dy @ u_z) + (dz @ u_y)
    eps_xz = (dx @ u_z) + (dz @ u_x)
    eps_xy = (dx @ u_y) + (dy @ u_x)

    eps_qs = jnp.stack([eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy],
                       axis=-1) + epsilon_bar
    stress = jnp.einsum("ij, qj, q, q -> qi", C_ss, eps_qs, det_J_q, W_q)

    R_x = jnp.sum(stress[:, 0, None] * dx + stress[:, 5, None] * dy
                  + stress[:, 4, None] * dz, axis=0)
    R_y = jnp.sum(stress[:, 5, None] * dx + stress[:, 1, None] * dy
                  + stress[:, 3, None] * dz, axis=0)
    R_z = jnp.sum(stress[:, 4, None] * dx + stress[:, 3, None] * dy
                  + stress[:, 2, None] * dz, axis=0)
    return jnp.stack([R_x, R_y, R_z], axis=-1)


@partial(jax.jit, static_argnames=['n_model', 'n_sg'])
def calculate_RHS_and_Ke_batch_periodic(
    x_end: jnp.ndarray, dphi_dxi_qnp: jnp.ndarray, phi_qn: jnp.ndarray,
    W_q: jnp.ndarray, C_ess: jnp.ndarray, periodic_cells,
    u_g_flat: jnp.ndarray, n_model: int, n_sg: int
):
    if n_model == 1:
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
    elif n_model == 2:
        mask_ones_2d = (jnp.zeros((6, 6)).at[0, 0].set(1.0)
                        .at[1, 1].set(1.0).at[5, 2].set(1.0))
        mask_y3_2d = (jnp.zeros((6, 6)).at[0, 3].set(1.0)
                      .at[1, 4].set(1.0).at[5, 5].set(1.0))

    u_end = transform_global_unraveled_to_element_node(periodic_cells,
                                                       u_g_flat, U=3)
    N = x_end.shape[1]
    U = 3

    def element_process_all_cases_and_Ke(u_nd, x_nd, C_ss):
        x_q = jnp.dot(phi_qn, x_nd)
        if n_model == 3:
            Ge = jnp.eye(6)
            run_cases = jax.vmap(_element_residual_single_case,
                                 in_axes=(None, None, None, None, None,
                                          None, 1))
        elif n_model == 2:
            y3_q = x_q[:, n_sg - 1]
            Ge = (mask_ones_2d[None, :, :]
                  + mask_y3_2d[None, :, :] * y3_q[:, None, None])
            run_cases = jax.vmap(_element_residual_single_case,
                                 in_axes=(None, None, None, None, None,
                                          None, 2))
        elif n_model == 1:
            y3_q = x_q[:, n_sg - 1]
            y2_q = (jnp.zeros_like(x_q[:, 0]) if n_sg == 1
                    else x_q[:, n_sg - 2])
            Ge = (mask_ones_1d[None, :, :]
                  + mask_y2[None, :, :] * y2_q[:, None, None]
                  + mask_y3_1d[None, :, :] * y3_q[:, None, None])
            run_cases = jax.vmap(_element_residual_single_case,
                                 in_axes=(None, None, None, None, None,
                                          None, 2))

        R_nodes_cases = run_cases(u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q,
                                  C_ss, Ge)

        def single_case_for_jac(u_flat):
            u_nd_t = u_flat.reshape(N, U)
            dummy_eps = jnp.zeros(6)
            R_single = _element_residual_single_case(
                u_nd_t, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, dummy_eps)
            return R_single.ravel()

        K_elem = jax.jacfwd(single_case_for_jac)(u_nd.ravel())
        return R_nodes_cases, K_elem

    batch_processor = jax.vmap(element_process_all_cases_and_Ke,
                               in_axes=(0, 0, 0))
    R_end_cases, J_uu_batch = batch_processor(u_end, x_end, C_ess)

    R_cases_first = jnp.transpose(R_end_cases, (1, 0, 2, 3))

    def assemble_one_case(R_one_case_EN3):
        return transform_element_node_to_global_unraveled_sum(
            periodic_cells=periodic_cells, v_en=R_one_case_EN3,
            num_nodes=u_g_flat.shape[0] // 3)

    R_global_cases = jax.vmap(assemble_one_case)(R_cases_first)
    num_cases = R_global_cases.shape[0]
    rhs_matrix = R_global_cases.reshape(num_cases, -1).T
    rhs_matrix = rhs_matrix.at[0:3, :].set(0.0)
    return rhs_matrix, J_uu_batch


def ebe_jacobian_product_periodic(J_uu, periodic_cells, n_unique_u,
                                  z_u: jnp.ndarray):
    z_u_enu = transform_global_unraveled_to_element_node(periodic_cells, z_u)
    _, N, U = z_u_enu.shape
    z_u_local = z_u_enu.reshape(-1, N * U)
    f_u_local_flat = jnp.einsum('eij,ej->ei', J_uu, z_u_local)
    f_u_local = f_u_local_flat.reshape(-1, N, U)
    f_u_global_full = transform_element_node_to_global_unraveled_sum(
        periodic_cells=periodic_cells, v_en=f_u_local,
        num_nodes=n_unique_u // U)
    f_u_global_full = f_u_global_full.at[0:3].set(z_u[0:3])
    return f_u_global_full


@partial(jax.jit, static_argnames=["n_unique_u"])
def apply_block_precond(inv_blocks, n_unique_u, x_reduced):
    x_nodes = x_reduced.reshape(-1, 3)
    precond_nodes = jnp.einsum('nij,nj->ni', inv_blocks, x_nodes)
    return precond_nodes.ravel()


@partial(jax.jit, static_argnames=["n_unique"])
def compute_block_inv_diag(J_uu, periodic_cells, n_unique):
    E, n_u, _ = J_uu.shape
    N = n_u // 3
    J_uu_reshaped = J_uu.reshape(E, N, 3, N, 3)
    local_3x3_blocks = jnp.diagonal(J_uu_reshaped, axis1=1, axis2=3)
    local_3x3_blocks = jnp.moveaxis(local_3x3_blocks, -1, 1)
    num_unique = n_unique // 3
    global_blocks = jnp.zeros((num_unique, 3, 3))
    global_blocks = global_blocks.at[periodic_cells].add(local_3x3_blocks)
    I_3x3 = jnp.eye(3)[None, :, :]
    global_blocks_safe = global_blocks + I_3x3 * 1e-8
    return jnp.linalg.inv(global_blocks_safe)


def estimate_max_eigenvalue(A_op, M_op, b_col, num_iters=15):
    """Power method on M^-1 A for the Chebyshev bounds."""
    v = jnp.ones_like(b_col)
    v = v / jnp.linalg.norm(v)

    def power_step(i, state):
        v_current, current_lam = state
        w = M_op(A_op(v_current))
        lam_new = jnp.vdot(v_current, w)
        v_new = w / (jnp.linalg.norm(w) + 1e-12)
        return (v_new, lam_new)

    _, lambda_max = jax.lax.fori_loop(0, num_iters, power_step, (v, 0.0))
    return lambda_max * 1.05


def apply_chebyshev_precond(inv_blocks, n_unique_u, eig_max, eig_min, A_op,
                            degree, x_reduced):
    d = (eig_max + eig_min) / 2.0
    c = (eig_max - eig_min) / 2.0
    i = jnp.arange(1, degree + 1)
    theta = d + c * jnp.cos(jnp.pi * (2 * i - 1) / (2 * degree))

    def step(z, theta_i):
        r = x_reduced - A_op(z)
        z_update = apply_block_precond(inv_blocks, n_unique_u, r)
        z_new = z + z_update / theta_i
        return z_new, None

    z0 = jnp.zeros_like(x_reduced)
    z_final, _ = jax.lax.scan(step, z0, theta)
    return z_final


@partial(jax.jit, static_argnames=['n_model', 'n_sg'])
def compute_homogenized_constants(
    x_end: jnp.ndarray, dphi_dxi_qnp: jnp.ndarray, phi_qn: jnp.ndarray,
    W_q: jnp.ndarray, C_ess: jnp.ndarray, n_model: int, n_sg: int
):
    if n_model == 3:
        def _get_vol(x_nd):
            J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
            return jnp.sum(jnp.linalg.det(J_qdp) * W_q)
        elem_vols = jax.vmap(_get_vol)(x_end)
        omega = jnp.sum(elem_vols)
    elif n_model == 2:
        if n_sg == 3:
            omega = jnp.ptp(x_end[..., 0]) * jnp.ptp(x_end[..., 1])
        elif n_sg == 2:
            omega = jnp.ptp(x_end[..., 0])
        else:
            omega = 1.0
    elif n_model == 1:
        omega = jnp.ptp(x_end[..., 0]) if n_sg == 3 else 1.0
    else:
        omega = 1.0

    if n_model == 1:
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
    elif n_model == 2:
        mask_ones_2d = (jnp.zeros((6, 6)).at[0, 0].set(1.0)
                        .at[1, 1].set(1.0).at[5, 2].set(1.0))
        mask_y3_2d = (jnp.zeros((6, 6)).at[0, 3].set(1.0)
                      .at[1, 4].set(1.0).at[5, 5].set(1.0))

    def compute_D_elem(x_nd, C_ss):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        if n_sg > 1:
            detJ_q = jnp.abs(jnp.linalg.det(J_qdp))
        else:
            detJ_q = jnp.abs(J_qdp[..., 0, 0])
        dV_q = detJ_q * W_q
        x_q = jnp.dot(phi_qn, x_nd)
        if n_model == 3:
            Ge = jnp.eye(6)
            D_elem = jnp.einsum('ji,jk,kl,q->il', Ge, C_ss, Ge, dV_q)
        elif n_model == 2:
            y3_q = x_q[:, n_sg - 1]
            Ge = (mask_ones_2d[None, :, :]
                  + mask_y3_2d[None, :, :] * y3_q[:, None, None])
            D_elem = jnp.einsum('qji,jk,qkl,q->il', Ge, C_ss, Ge, dV_q)
        elif n_model == 1:
            y3_q = x_q[:, n_sg - 1]
            y2_q = (jnp.zeros_like(x_q[:, 0]) if n_sg == 1
                    else x_q[:, n_sg - 2])
            Ge = (mask_ones_1d[None, :, :]
                  + mask_y2[None, :, :] * y2_q[:, None, None]
                  + mask_y3_1d[None, :, :] * y3_q[:, None, None])
            D_elem = jnp.einsum('qji,jk,qkl,q->il', Ge, C_ss, Ge, dV_q)
        return D_elem

    D_elem_batch = jax.vmap(compute_D_elem)(x_end, C_ess)
    D_bar = jnp.sum(D_elem_batch, axis=0)
    return D_bar, omega


# ------------------------- beam Timoshenko/KKT assembly (from Beam_solid.py)
@partial(jax.jit, static_argnames=['N_primal', 'n_model', 'n_sg'])
def assemble_system_matrices(
    x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_periodic_cells, N_primal,
    C_ess, n_model, n_sg
):
    """Energy-form assembly (Beam_solid.py): sparse Dhh/Dhe + dense Dee,
    and for n_model=1 the beam l-chain Dll/Dhl/Dle.  All bilinear forms
    come from AD of the element energies; the kinematic masks match the
    SSDM kernels above (same 4 EB macro modes for n_model=1)."""
    E_elem, N_nodes = x_end.shape[0], x_end.shape[1]

    if n_model == 1:
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
    else:
        mask_ones_2d = (jnp.zeros((6, 6)).at[0, 0].set(1.0)
                        .at[1, 1].set(1.0).at[5, 2].set(1.0))
        mask_y3_2d = (jnp.zeros((6, 6)).at[0, 3].set(1.0)
                      .at[1, 4].set(1.0).at[5, 5].set(1.0))

    def get_element_matrices(x_nd, C_in):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        det_J_q = jnp.abs(jnp.linalg.det(J_qdp))
        dV_q = det_J_q * W_q

        inv_J = jnp.linalg.inv(J_qdp)
        dphi_dx = jnp.einsum("qpd,qnp->qnd", inv_J, dphi_dxi_qnp)

        x_q = jnp.einsum("qn,nd->qd", phi_qn, x_nd)
        Q = x_q.shape[0]
        C_q = jnp.repeat(C_in[None, :, :], Q, axis=0) if C_in.ndim == 2 \
            else C_in

        if n_model == 3:
            Ge = jnp.repeat(jnp.eye(6)[None, ...], Q, axis=0)
        elif n_model == 2:
            y3_q = x_q[:, n_sg - 1]
            Ge = (mask_ones_2d[None, ...]
                  + mask_y3_2d[None, ...] * y3_q[:, None, None])
        else:
            if n_sg == 2:
                y2_q = x_q[:, 0]
                y3_q = x_q[:, 1]
            else:
                y2_q = jnp.zeros_like(x_q[:, 0])
                y3_q = x_q[:, 0]
            Ge = (mask_ones_1d[None, ...]
                  + mask_y2[None, ...] * y2_q[:, None, None]
                  + mask_y3_1d[None, ...] * y3_q[:, None, None])

        def compute_eps_h(uh_flat):
            uh = uh_flat.reshape(N_nodes, 3)
            zeros = jnp.zeros_like(dphi_dx[..., 0])
            D = dphi_dx.shape[-1]
            if D == 1:
                grad_phi = jnp.stack([zeros, zeros, dphi_dx[..., 0]],
                                     axis=-1)
            elif D == 2:
                grad_phi = jnp.stack([zeros, dphi_dx[..., 0],
                                      dphi_dx[..., 1]], axis=-1)
            else:
                grad_phi = dphi_dx

            grad_u = jnp.einsum("ni,qnj->qij", uh, grad_phi)
            return jnp.stack([
                grad_u[:, 0, 0], grad_u[:, 1, 1], grad_u[:, 2, 2],
                grad_u[:, 1, 2] + grad_u[:, 2, 1],
                grad_u[:, 0, 2] + grad_u[:, 2, 0],
                grad_u[:, 0, 1] + grad_u[:, 1, 0]
            ], axis=1)

        def compute_eps_l(ul_flat):
            # d/dx1 of the warping: eps11 <- u1, 2eps13 <- u3, 2eps12 <- u2
            ul = ul_flat.reshape(N_nodes, 3)
            u_q = jnp.einsum("qn,ni->qi", phi_qn, ul)
            zeros = jnp.zeros_like(u_q[:, 0])
            return jnp.stack([u_q[:, 0], zeros, zeros, zeros, u_q[:, 2],
                              u_q[:, 1]], axis=1)

        def compute_eps_e(ue_flat):
            return jnp.einsum("qip,p->qi", Ge, ue_flat)

        Ndofs = N_nodes * 3
        Ndofs_e = Ge.shape[-1]
        zeros_u, zeros_e = jnp.zeros(Ndofs), jnp.zeros(Ndofs_e)

        def energy_hh(uh_flat):
            eps_h = compute_eps_h(uh_flat)
            return 0.5 * jnp.einsum("qi,qij,qj,q->", eps_h, C_q, eps_h,
                                    dV_q)

        def energy_he(uh_flat, ue_flat):
            eps_h = compute_eps_h(uh_flat)
            eps_e = compute_eps_e(ue_flat)
            return jnp.einsum("qi,qij,qj,q->", eps_h, C_q, eps_e, dV_q)

        def energy_ee(ue_flat):
            eps_e = compute_eps_e(ue_flat)
            return 0.5 * jnp.einsum("qi,qij,qj,q->", eps_e, C_q, eps_e,
                                    dV_q)

        D_hh_e = jax.hessian(energy_hh)(zeros_u)
        D_he_e = jax.jacfwd(jax.grad(energy_he, argnums=0),
                            argnums=1)(zeros_u, zeros_e)
        D_ee_e = jax.hessian(energy_ee)(zeros_e)

        if n_model == 1:
            def energy_ll(ul_flat):
                eps_l = compute_eps_l(ul_flat)
                return 0.5 * jnp.einsum("qi,qij,qj,q->", eps_l, C_q,
                                        eps_l, dV_q)

            def energy_hl(uh_flat, ul_flat):
                eps_h = compute_eps_h(uh_flat)
                eps_l = compute_eps_l(ul_flat)
                return jnp.einsum("qi,qij,qj,q->", eps_h, C_q, eps_l,
                                  dV_q)

            def energy_le(ul_flat, ue_flat):
                eps_l = compute_eps_l(ul_flat)
                eps_e = compute_eps_e(ue_flat)
                return jnp.einsum("qi,qij,qj,q->", eps_l, C_q, eps_e,
                                  dV_q)

            D_ll_e = jax.hessian(energy_ll)(zeros_u)
            D_hl_e = jax.jacfwd(jax.grad(energy_hl, argnums=0),
                                argnums=1)(zeros_u, zeros_u)
            D_le_e = jax.jacfwd(jax.grad(energy_le, argnums=0),
                                argnums=1)(zeros_u, zeros_e)

            return (D_hh_e, D_he_e, D_ee_e, D_ll_e, D_hl_e, D_le_e,
                    dphi_dx, Ge, dV_q)

        # the l-operators exist only for the beam macro model
        return D_hh_e, D_he_e, D_ee_e, dphi_dx, Ge, dV_q

    vmap_out = jax.vmap(get_element_matrices, in_axes=(0, 0))(x_end, C_ess)

    dof_map = ((reduced_periodic_cells * 3).reshape(E_elem, N_nodes, 1)
               + jnp.array([0, 1, 2])).reshape(E_elem, N_nodes * 3)
    rows_sq = jnp.repeat(dof_map, N_nodes * 3, axis=1).ravel()
    cols_sq = jnp.tile(dof_map, (1, N_nodes * 3)).ravel()

    D_hh_b, D_he_b, D_ee_b = vmap_out[0], vmap_out[1], vmap_out[2]

    Ndofs_e = D_he_b.shape[-1]
    rows_rect = jnp.repeat(dof_map, Ndofs_e, axis=1).ravel()
    cols_rect = jnp.tile(jnp.arange(Ndofs_e),
                         (E_elem, N_nodes * 3)).ravel()

    Dhh = jsparse.COO((D_hh_b.ravel(), rows_sq, cols_sq),
                      shape=(N_primal, N_primal))
    Dhe = jsparse.COO((D_he_b.ravel(), rows_rect, cols_rect),
                      shape=(N_primal, Ndofs_e))
    Dee = jnp.sum(D_ee_b, axis=0)

    dehomo_data = {
        "dphi_dx": vmap_out[-3],    # (E, Q, N, D)
        "Ge": vmap_out[-2],         # (E, Q, 6, Ndofs_e)
        "dV_q": vmap_out[-1]        # (E, Q)
    }

    if n_model == 3:
        omega = jnp.sum(dehomo_data["dV_q"])
    elif n_model == 2:
        if n_sg == 3:
            omega = jnp.ptp(x_end[..., 0]) * jnp.ptp(x_end[..., 1])
        elif n_sg == 2:
            omega = jnp.ptp(x_end[..., 0])
        else:
            omega = 1.0
    elif n_model == 1:
        omega = jnp.ptp(x_end[..., 0]) if n_sg == 3 else 1.0
    else:
        omega = 1.0

    if n_model == 1:
        D_ll_b, D_hl_b, D_le_b = vmap_out[3], vmap_out[4], vmap_out[5]
        Dll = jsparse.COO((D_ll_b.ravel(), rows_sq, cols_sq),
                          shape=(N_primal, N_primal))
        Dhl = jsparse.COO((D_hl_b.ravel(), rows_sq, cols_sq),
                          shape=(N_primal, N_primal))
        Dle = jsparse.COO((D_le_b.ravel(), rows_rect, cols_rect),
                          shape=(N_primal, Ndofs_e))

        return (Dhh._sort_indices(), Dhe._sort_indices(), Dee,
                Dll._sort_indices(), Dhl._sort_indices(),
                Dle._sort_indices(), omega, dehomo_data)

    return (Dhh._sort_indices(), Dhe._sort_indices(), Dee, omega,
            dehomo_data)


@partial(jax.jit, static_argnames=['N_primal_dofs'])
def build_lagrange_constraint_matrix(
    x_end, dphi_dxi_qnp, phi_qn, W_q, reduced_periodic_cells, N_primal_dofs
):
    """The 4 macroscopic constraint rows [u1, u2, u3, -z u2 + y u3]
    integrated over the SG (Beam_solid.py).  (y2, y3) are the LAST two SG
    coordinates (Beam_solid assumed 3 columns; n_sg >= 2 required).  The
    routed beam path constrains with assemble_rigid_body_ops' Dc, as
    Beam_solid's driver did -- this builder is kept importable."""
    E, N_nodes = x_end.shape[0], x_end.shape[1]

    J_eqdp = jnp.einsum("end,qnp->eqdp", x_end, dphi_dxi_qnp)
    det_J_eq = jnp.linalg.det(J_eqdp)
    dV_eq = det_J_eq * W_q

    int_phi_en = jnp.einsum("qn,eq->en", phi_qn, dV_eq)

    x_eq = jnp.einsum("qn,end->eqd", phi_qn, x_end)
    y_eq, z_eq = x_eq[..., -2], x_eq[..., -1]

    int_y_phi_en = jnp.einsum("eq,qn,eq->en", y_eq, phi_qn, dV_eq)
    int_z_phi_en = jnp.einsum("eq,qn,eq->en", z_eq, phi_qn, dV_eq)

    zeros_en = jnp.zeros_like(int_phi_en)
    r0 = jnp.stack([int_phi_en, zeros_en, zeros_en],
                   axis=-1).reshape(E, N_nodes * 3)
    r1 = jnp.stack([zeros_en, int_phi_en, zeros_en],
                   axis=-1).reshape(E, N_nodes * 3)
    r2 = jnp.stack([zeros_en, zeros_en, int_phi_en],
                   axis=-1).reshape(E, N_nodes * 3)
    r3 = jnp.stack([zeros_en, -int_z_phi_en, int_y_phi_en],
                   axis=-1).reshape(E, N_nodes * 3)
    C_ett = jnp.stack([r0, r1, r2, r3], axis=1)

    dof_map_enu = jnp.stack([
        reduced_periodic_cells * 3 + 0,
        reduced_periodic_cells * 3 + 1,
        reduced_periodic_cells * 3 + 2
    ], axis=-1).reshape(E, N_nodes * 3)

    C_assembled = jnp.zeros((4, N_primal_dofs))
    C_assembled = C_assembled.at[:, dof_map_enu].add(
        C_ett.transpose(1, 0, 2))
    return C_assembled


def compress_periodic_cells_jax(V, cells, dof_map, x_end,
                                ndof_per_node=3):
    """Master-node compression of the periodic dof map (Beam_solid.py):
    the reduced element connectivity, the unique-master count, and the
    unique-master coordinates (assemble_rigid_body_ops needs them)."""
    flat_x = x_end.reshape(-1, x_end.shape[-1])
    flat_cells = cells.reshape(-1)

    global_points = jnp.zeros((V, x_end.shape[-1]))
    global_points = global_points.at[flat_cells].set(flat_x)

    master_nodes = dof_map[::ndof_per_node] // ndof_per_node

    unique_masters = jnp.unique(master_nodes)
    num_unique = int(unique_masters.shape[0])

    full_to_reduced = jnp.full(V, -1, dtype=jnp.int32)
    full_to_reduced = full_to_reduced.at[unique_masters].set(
        jnp.arange(num_unique, dtype=jnp.int32))

    reduced_periodic_cells = full_to_reduced[master_nodes[cells]]
    x_unique = global_points[unique_masters]
    return reduced_periodic_cells, num_unique, x_unique


# ------------------- RM plate first-order ladder blocks (Yu Eq. 30 set,
# generalized from rm_plate_1D.msg_rm_plate to a 1-D/2-D/3-D SG)
_E0_PLATE = np.zeros((6, 6)); _E0_PLATE[0, 0] = _E0_PLATE[1, 1] = _E0_PLATE[5, 2] = 1.0
_E1_PLATE = np.zeros((6, 6)); _E1_PLATE[0, 3] = _E1_PLATE[1, 4] = _E1_PLATE[5, 5] = 1.0
_jE0_PLATE = jnp.asarray(_E0_PLATE)
_jE1_PLATE = jnp.asarray(_E1_PLATE)


@partial(jax.jit, static_argnames=['n_sg'])
def plate_ladder_element_blocks(x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess,
                                n_sg):
    """Element blocks of the RM first-order warping ladder on a general
    SG: hh, he, ee, hl1, hl2, l1l1, l1l2, l2l2, l1e, l2e (+ the <.>
    node weights).  Gamma_h = the SG strain operator (resolved dims,
    zero-padded exactly like the SSDM kernels); Gamma_l1/Gamma_l2 =
    the in-plane-derivative shift operators with the rm_plate_1D
    _grad_ops rows (w,1: e11<-w1, 2g13<-w3, 2g12<-w2 | w,2: e22<-w2,
    2g23<-w3, 2g12<-w1); Gamma_eps = E0 + y3 E1 (the plate masks).
    Pass a quadrature exact to degree 2p+2 -- the l-blocks integrate
    basis VALUE products, one degree above the classical set."""
    def one(x_nd, C):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        G_qpd = jnp.linalg.inv(J_qdp)
        detJ = (jnp.abs(jnp.linalg.det(J_qdp)) if n_sg > 1
                else jnp.abs(J_qdp[..., 0, 0]))
        dV = detJ * W_q
        dphi = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

        Q, N = phi_qn.shape
        z = jnp.zeros((Q, N))
        if n_sg == 1:
            d3 = jnp.stack([z, z, dphi[..., 0]], axis=-1)
        elif n_sg == 2:
            d3 = jnp.stack([z, dphi[..., 0], dphi[..., 1]], axis=-1)
        else:
            d3 = dphi
        dx, dy, dz_ = d3[..., 0], d3[..., 1], d3[..., 2]

        Bh = jnp.zeros((Q, 6, N, 3))
        Bh = Bh.at[:, 0, :, 0].set(dx).at[:, 1, :, 1].set(dy)
        Bh = Bh.at[:, 2, :, 2].set(dz_)
        Bh = Bh.at[:, 3, :, 1].set(dz_).at[:, 3, :, 2].set(dy)
        Bh = Bh.at[:, 4, :, 0].set(dz_).at[:, 4, :, 2].set(dx)
        Bh = Bh.at[:, 5, :, 0].set(dy).at[:, 5, :, 1].set(dx)
        Bh = Bh.reshape(Q, 6, N * 3)

        P = phi_qn
        Bl1 = jnp.zeros((Q, 6, N, 3))
        Bl1 = Bl1.at[:, 0, :, 0].set(P).at[:, 4, :, 2].set(P)
        Bl1 = Bl1.at[:, 5, :, 1].set(P).reshape(Q, 6, N * 3)
        Bl2 = jnp.zeros((Q, 6, N, 3))
        Bl2 = Bl2.at[:, 1, :, 1].set(P).at[:, 3, :, 2].set(P)
        Bl2 = Bl2.at[:, 5, :, 0].set(P).reshape(Q, 6, N * 3)

        x_q = jnp.dot(phi_qn, x_nd)
        y3 = x_q[:, n_sg - 1]
        Ge = _jE0_PLATE[None] + _jE1_PLATE[None] * y3[:, None, None]

        def bil(Ba, Bb):
            return jnp.einsum("qim,ij,qjn,q->mn", Ba, C, Bb, dV)

        def bil_s(Ba):
            return jnp.einsum("qim,ij,qjs,q->ms", Ba, C, Ge, dV)

        ee = jnp.einsum("qis,ij,qjt,q->st", Ge, C, Ge, dV)
        wN = jnp.einsum("qn,q->n", phi_qn, dV)
        return (bil(Bh, Bh), bil_s(Bh), ee, bil(Bh, Bl1), bil(Bh, Bl2),
                bil(Bl1, Bl1), bil(Bl1, Bl2), bil(Bl2, Bl2),
                bil_s(Bl1), bil_s(Bl2), wN)

    return jax.vmap(one, in_axes=(0, 0))(x_end, C_ess)


_direct_spsolve = None


def _sparse_direct_solve(A_csr, B):
    """Direct sparse solve A X = B (dense multi-RHS).  pypardiso when
    importable (the Beam_solid solver), scipy SuperLU otherwise."""
    global _direct_spsolve
    if _direct_spsolve is None:
        try:
            from pypardiso import spsolve as _direct
        except ImportError:
            from scipy.sparse.linalg import spsolve as _direct
        _direct_spsolve = _direct
    return np.asarray(_direct_spsolve(A_csr, B))


def solve_fluctuation_field(Dhh_sparse, RHS_dense, Dc_matrix, n_model):
    """The fluctuation solve (Beam_solid.py): for n_model=1 the AUGMENTED
    KKT system [[Dhh, s Dc], [s Dc^T, 0]] (s = 1e8 scales the constraint
    rows to the stiffness magnitude), otherwise the plain system; rows
    with no entries get a unit diagonal + zero RHS (periodic slaves).
    Out: V0 (N, cases), D1 = -(V0^T RHS), and the factorizable csr
    A_augmented (reused for the V1s solve)."""
    R_array = np.asarray(RHS_dense)
    actual_N = R_array.shape[0]
    num_cases = R_array.shape[1]

    if not hasattr(Dhh_sparse, 'row'):
        Dhh_sparse = Dhh_sparse.tocoo()

    dhh_data = np.asarray(Dhh_sparse.data)
    dhh_row = np.asarray(Dhh_sparse.row)
    dhh_col = np.asarray(Dhh_sparse.col)

    if int(n_model) == 1 and Dc_matrix is not None:
        Dc_dense = np.asarray(Dc_matrix)
        if Dc_dense.shape[0] == 4 and Dc_dense.shape[1] == actual_N:
            Dc_dense = Dc_dense.T

        num_constraints = Dc_dense.shape[1]
        Dc_scaled = Dc_dense * 1e8

        bl_row, bl_col = np.where(Dc_scaled.T)
        bl_data = Dc_scaled.T[bl_row, bl_col]
        bl_row += actual_N

        tr_row, tr_col = np.where(Dc_scaled)
        tr_data = Dc_scaled[tr_row, tr_col]
        tr_col += actual_N

        aug_row = np.concatenate([dhh_row, tr_row, bl_row])
        aug_col = np.concatenate([dhh_col, tr_col, bl_col])
        aug_data = np.concatenate([dhh_data, tr_data, bl_data])

        total_N = actual_N + num_constraints

        zero_block = np.zeros((num_constraints, num_cases),
                              dtype=R_array.dtype)
        R_aug = np.vstack([R_array, zero_block])
    else:
        aug_row, aug_col, aug_data = dhh_row, dhh_col, dhh_data
        total_N = actual_N
        R_aug = R_array

    row_counts = np.bincount(aug_row, minlength=total_N)
    empty_rows = np.where(row_counts == 0)[0]
    if len(empty_rows) > 0:
        aug_row = np.concatenate([aug_row, empty_rows])
        aug_col = np.concatenate([aug_col, empty_rows])
        aug_data = np.concatenate([aug_data,
                                   np.ones_like(empty_rows,
                                                dtype=aug_data.dtype)])
        R_aug[empty_rows, :] = 0.0

    A_augmented = csr_matrix((aug_data, (aug_row, aug_col)),
                             shape=(total_N, total_N))
    if A_augmented.shape[0] != R_aug.shape[0]:
        raise RuntimeError("dimension mismatch: A %s vs R %s"
                           % (A_augmented.shape, R_aug.shape))

    V_aug = _sparse_direct_solve(A_augmented, R_aug)

    V0 = V_aug[:actual_N, :]
    D1 = -(V0.T @ R_array)
    return jnp.array(V0), jnp.array(D1), A_augmented


@partial(jax.jit, static_argnames=['N_primal', 'n_sg'])
def assemble_rigid_body_ops(x_unique, x_end, dphi_dxi_qnp, phi_qn, W_q,
                            reduced_periodic_cells, N_primal, n_sg):
    """The rigid-body kernel Psi (3 translations + twist [0, -z, y]) and
    the 4-row constraint matrix Dc (Beam_solid.py); Dc's twist row uses
    the GRADIENT weights int dphi/dz, -int dphi/dy.  n_sg >= 2."""
    dim = x_unique.shape[1]
    y_coords = jnp.zeros_like(x_unique[:, 0]) if dim == 1 \
        else x_unique[:, dim - 2]
    z_coords = x_unique[:, dim - 1]

    num_nodes = N_primal // 3

    psi_0 = jnp.tile(jnp.array([1., 0., 0.]), num_nodes)
    psi_1 = jnp.tile(jnp.array([0., 1., 0.]), num_nodes)
    psi_2 = jnp.tile(jnp.array([0., 0., 1.]), num_nodes)

    psi_3 = jnp.zeros(N_primal)
    psi_3 = psi_3.at[1::3].set(-z_coords)
    psi_3 = psi_3.at[2::3].set(y_coords)

    Psi = jnp.column_stack([psi_0, psi_1, psi_2, psi_3])

    E_elem, N_nodes = x_end.shape[0], x_end.shape[1]

    def element_Dc(x_nd):
        J = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        det_J = jnp.abs(jnp.linalg.det(J))
        dV = det_J * W_q
        inv_J = jnp.linalg.inv(J)
        dphi_dx = jnp.einsum("qpd,qnp->qnd", inv_J, dphi_dxi_qnp)

        idx_y = 0 if dphi_dx.shape[-1] == 2 else 1
        idx_z = 1 if dphi_dx.shape[-1] == 2 else 2

        int_phi = jnp.einsum("qn,q->n", phi_qn, dV)
        int_dphi_dy = jnp.einsum("qn,q->n", dphi_dx[..., idx_y], dV)
        int_dphi_dz = jnp.einsum("qn,q->n", dphi_dx[..., idx_z], dV)

        c_e = jnp.zeros((4, N_nodes * 3))
        c_e = c_e.at[0, 0::3].set(int_phi)
        c_e = c_e.at[1, 1::3].set(int_phi)
        c_e = c_e.at[2, 2::3].set(int_phi)
        c_e = c_e.at[3, 1::3].set(int_dphi_dz)
        c_e = c_e.at[3, 2::3].set(-int_dphi_dy)
        return c_e

    c_batches = jax.vmap(element_Dc)(x_end)

    dof_map = ((reduced_periodic_cells * 3).reshape(E_elem, N_nodes, 1)
               + jnp.array([0, 1, 2])).reshape(E_elem, N_nodes * 3)

    Dc_T = jnp.zeros((4, N_primal)).at[:, dof_map.ravel()].add(
        c_batches.transpose(1, 0, 2).reshape(4, -1))
    return Psi, Dc_T.T
