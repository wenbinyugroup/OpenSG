"""sg_dehom.py -- dehomogenization of the general SG engine: the SSDM
Gauss-point recovery kernel (n_model 2/3), the generalized-Timoshenko
beam recovery chain (n_model 1), the public dehom_fields /
plate_dehom_2d entries, and the Gauss exports.

The recovery kernels are verbatim ports of validated references; the
batched SSDM recovery is wrapped in a module-level jit so repeated
dehom calls in one process (and, via the persistent cache, across
processes) do not re-trace.

Output order everywhere: SwiftComp (xx, yy, zz, yz, xz, xy).  The
OpenSG dehom FILES <prefix>.SM/.EM/.U follow the rm_plate_1D
rm_dehom.write_field layout instead -- printed order 11 22 33 12 13 23
via the SGIDX reorder ON WRITE (storage order never changes), rows
x y z + components with the SG coordinates in the last n_sg slots.
dehom_fields is the one-pass entry (strain + stress + fluctuation
displacement); plate_dehom_2d is a two-output view of it.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE  (axis suffixes: e element, n elem-node,
# q quad point, d spatial dim, s Voigt-6, H macro mode)
# ----------------------------------------------------------------------------
# inputs
#   r                   the plate_homo_2d result dict
#   epsilon_bar         (6,) macro plate strain [e11 e22 2e12 k11 k22 2k12]
#                       (n_model 2/3) or the beam state st [x1 ext, shear2,
#                       shear3, twist, bend2, bend3] (n_model 1)
# working (SSDM kernel, n_model 2/3)
#   V0_ndH              (N, 3, H) element fluctuation modes;  x_nd (N, d)
#   C_ss                (6, 6);  J_qdp, G_qpd, dphi_dx_qnd  geometry
#   dx_qn, dy_qn, dz_qn (Q, N) gradient rows
#   Gamma_h_S_V0_qsH    (Q, 6, H) fluctuation strain modes
#   Ge_sH               (Q, 6, H) macro map;  Strain_Concentration_qsH
# working (beam chain, n_model 1)
#   Comp_srt            (6, 6) = C_eff^-1;  st_m (4,) classical part
#   F_1d, F1, F2, F3    the successive-derivative ladder
#   R1, R2, R3          (6, 6) recovery/equilibrium operators
#   gamma1..3 (2,), st_cl1/2 (4,), st_m_final (4,)   corrected states
#   w_1, w1s_1, w1s_2, w_2   (N_primal,) fluctuation displacement vectors
#   dehomo_data         {dphi_dx (E,Q,N,d), Ge (E,Q,6,4), dV_q (E,Q)}
#   dof_map             (E, N*3) dof routing;  uh_1/ul_1/ul_2/uh_2 (E, N, 3)
#   eps_Eb, eps_Timo, st_3D_global, st_3D_elem   (E, Q, 6) strain builds
#   Rsig_batch          (E, 6, 6) Voigt rotations;  C_q (E, Q, 6, 6)
# outputs
#   Gamma_qs / Gam      (E, Q, 6) strain;  Sigma_qs / Sig (E, Q, 6) stress
#   U_eqd               (E, Q, 3) FLUCTUATION-ONLY displacement at the
#                       Gauss points (2/3: V0 @ eps; beam: w_1 + w1s_1)
#   coords, xyz         (E*Q, d) Gauss coordinates; padded x y z columns
#   SGIDX, ORDER        SwiftComp storage -> 11 22 33 12 13 23 print map
#   <prefix>.txt/.vtk   the Gauss table + point-cloud exports
#   <prefix>.SM/.EM/.U  the OpenSG dehom files (rm_dehom layout)
# ----------------------------------------------------------------------------
"""
from functools import partial
from typing import Any, Dict, Optional, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from fe_jax.setup import transform_global_unraveled_to_element_node


# --------------------------------------------------------------- dehom kernel
def _element_dehomo_kernel(
    V0_ndH: jnp.ndarray, x_nd: jnp.ndarray, C_ss: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray, phi_qn: jnp.ndarray,
    epsilon_bar_H: jnp.ndarray, n_model: int, n_sg: int
):
    """Single-element Gauss-point recovery: local strain
    (Gamma_h V0 + Ge) @ epsilon_bar and its stress C @ strain.

    In:
        V0_ndH: (N, 3, H) element fluctuation modes, H = 6 for
            plate/solid, 4 for beam macro modes.
        x_nd: (N, d) element node coordinates, d = SG dimension.
        C_ss: (6, 6) element stiffness.
        dphi_dxi_qnp: (Q, N, p) parametric basis gradients.
        phi_qn: (Q, N) basis values.
        epsilon_bar_H: (H,) macro strain state.
        n_model: 1 beam / 2 plate / 3 solid.
        n_sg: SG spatial dimension.
    Out:
        Gamma_qs: (Q, 6) local strain, SwiftComp order.
        Sigma_qs: (Q, 6) local stress, SwiftComp order.
    Gradients are zero-padded so a d-dim SG occupies the LAST d slots
    of (x, y, z).
    """
    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    G_qpd = jnp.linalg.inv(J_qdp)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

    Q, N, _ = dphi_dx_qnd.shape
    zeros_qn = jnp.zeros((Q, N), dtype=dphi_dx_qnd.dtype)
    if x_nd.shape[1] == 1:
        dphi_dx_qnd = jnp.stack([zeros_qn, zeros_qn, dphi_dx_qnd[..., 0]],
                                axis=-1)
    elif x_nd.shape[1] == 2:
        dphi_dx_qnd = jnp.stack([zeros_qn, dphi_dx_qnd[..., 0],
                                 dphi_dx_qnd[..., 1]], axis=-1)

    dx_qn, dy_qn, dz_qn = (dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1],
                           dphi_dx_qnd[..., 2])
    V0_x_nH, V0_y_nH, V0_z_nH = (V0_ndH[:, 0, :], V0_ndH[:, 1, :],
                                 V0_ndH[:, 2, :])
    eps_xx_fluct_qH = dx_qn @ V0_x_nH
    eps_yy_fluct_qH = dy_qn @ V0_y_nH
    eps_zz_fluct_qH = dz_qn @ V0_z_nH
    eps_yz_fluct_qH = (dy_qn @ V0_z_nH) + (dz_qn @ V0_y_nH)
    eps_xz_fluct_qH = (dx_qn @ V0_z_nH) + (dz_qn @ V0_x_nH)
    eps_xy_fluct_qH = (dx_qn @ V0_y_nH) + (dy_qn @ V0_x_nH)
    Gamma_h_S_V0_qsH = jnp.stack([
        eps_xx_fluct_qH, eps_yy_fluct_qH, eps_zz_fluct_qH,
        eps_yz_fluct_qH, eps_xz_fluct_qH, eps_xy_fluct_qH], axis=1)

    x_qd = jnp.dot(phi_qn, x_nd)
    if n_model == 3:
        Ge_sH = jnp.eye(6)
        Strain_Concentration_qsH = Gamma_h_S_V0_qsH + Ge_sH[None, :, :]
    elif n_model == 2:
        mask_ones_2d = (jnp.zeros((6, 6)).at[0, 0].set(1.0)
                        .at[1, 1].set(1.0).at[5, 2].set(1.0))
        mask_y3_2d = (jnp.zeros((6, 6)).at[0, 3].set(1.0)
                      .at[1, 4].set(1.0).at[5, 5].set(1.0))
        y3_q = x_qd[:, n_sg - 1]
        Ge_sH = (mask_ones_2d[None, :, :]
                 + mask_y3_2d[None, :, :] * y3_q[:, None, None])
        Strain_Concentration_qsH = Gamma_h_S_V0_qsH + Ge_sH
    else:
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
        y3_q = x_qd[:, n_sg - 1]
        y2_q = (jnp.zeros_like(x_qd[:, 0]) if n_sg == 1
                else x_qd[:, n_sg - 2])
        Ge_sH = (mask_ones_1d[None, :, :]
                 + mask_y2[None, :, :] * y2_q[:, None, None]
                 + mask_y3_1d[None, :, :] * y3_q[:, None, None])
        Strain_Concentration_qsH = Gamma_h_S_V0_qsH + Ge_sH

    Gamma_qs = jnp.einsum("qij,j->qi", Strain_Concentration_qsH,
                          epsilon_bar_H)
    Sigma_qs = jnp.einsum("ij, qj -> qi", C_ss, Gamma_qs)
    return Gamma_qs, Sigma_qs


@partial(jax.jit, static_argnames=["n_sg"])
def _v2_batch(periodic_cells_en, V0lad, V11bar, V12bar, x_end, C_ess,
              dphi_dxi_qnp, phi_qn, dE1, dE2, n_sg):
    """The Eq. 63 first-order refined ADDITION to the plate recovery:

        dGam = Gamma_h (V11bar dE1 + V12bar dE2)
             + Gamma_l1 (V0lad dE1) + Gamma_l2 (V0lad dE2)

    with the strain-derivative drivers dE1/dE2 (d eps / d x_a, the
    plate measures [e11 e22 2e12 k11 k22 2k12],a) and the ladder's own
    <w> = 0-gauged warping triple -- Gamma_l consumes warping VALUES,
    so the pinned-node classical V0 gauge cannot be substituted.
    Gamma_l1 places the value components per the rm_plate_1D _grad_ops
    rows (e11 <- w1, 2g13 <- w3, 2g12 <- w2), Gamma_l2 (e22 <- w2,
    2g23 <- w3, 2g12 <- w1); Gamma_h is the padded SG strain operator
    of the classical kernel.  Zero drivers -> zero addition (the
    recovery is linear in its drivers).

    In:
        periodic_cells_en: (E, N) periodic connectivity.
        V0lad, V11bar, V12bar: (n_unique, 6) ladder warping columns.
        x_end: (E, N, d); C_ess: (E, 6, 6).
        dphi_dxi_qnp: (Q, N, p); phi_qn: (Q, N).
        dE1, dE2: (6,) strain derivatives.
        n_sg: static SG dimension.
    Out:
        dGam_eqs, dSig_eqs: (E, Q, 6), SwiftComp order
        (xx, yy, zz, yz, xz, xy)."""
    gather = jax.vmap(transform_global_unraveled_to_element_node,
                      in_axes=(None, 1, None), out_axes=-1)
    # element warping fields, contracted with the drivers up front
    V0_endH = gather(periodic_cells_en, V0lad, 3)      # (E, N, 3, 6)
    V11_endH = gather(periodic_cells_en, V11bar, 3)
    V12_endH = gather(periodic_cells_en, V12bar, 3)
    wh_end = (jnp.einsum("ends,s->end", V11_endH, dE1)
              + jnp.einsum("ends,s->end", V12_endH, dE2))
    wl1_end = jnp.einsum("ends,s->end", V0_endH, dE1)
    wl2_end = jnp.einsum("ends,s->end", V0_endH, dE2)
    return jax.vmap(_elem_chain, in_axes=(0, 0, 0, 0, 0, None, None))(
        wh_end, wl1_end, wl2_end, x_end, C_ess, dphi_dxi_qnp, phi_qn)


def _elem_chain(wh_nd, wl1_nd, wl2_nd, x_nd, C_ss, dphi_dxi_qnp,
                phi_qn):
    """ONE element of a refined-recovery chain: Gamma_h of the
    contracted GRADIENT-use field wh + Gamma_l1/Gamma_l2 of the
    contracted VALUE-use fields wl1/wl2, then stress.  Shared by the
    Eq. 63 first-order (_v2_batch), the qt6/qb6 load ladder
    (_vq_batch) and the Eq. 64-66 chains (_v266_batch) -- the callers
    differ only in HOW the fields are contracted from their columns.

    In:  wh_nd/wl1_nd/wl2_nd (N, 3); x_nd (N, d); C_ss (6, 6);
         dphi_dxi_qnp (Q, N, p); phi_qn (Q, N)
    Out: dGam_qs, dSig_qs (Q, 6), SwiftComp order."""
    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    G_qpd = jnp.linalg.inv(J_qdp)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    Q, N, _ = dphi_dx_qnd.shape
    z_qn = jnp.zeros((Q, N), dtype=dphi_dx_qnd.dtype)
    if x_nd.shape[1] == 1:
        dphi_dx_qnd = jnp.stack(
            [z_qn, z_qn, dphi_dx_qnd[..., 0]], axis=-1)
    elif x_nd.shape[1] == 2:
        dphi_dx_qnd = jnp.stack(
            [z_qn, dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1]], axis=-1)
    dx_qn, dy_qn, dz_qn = (dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1],
                           dphi_dx_qnd[..., 2])
    gx, gy, gz = wh_nd[:, 0], wh_nd[:, 1], wh_nd[:, 2]
    Gh_qs = jnp.stack([
        dx_qn @ gx, dy_qn @ gy, dz_qn @ gz,
        (dy_qn @ gz) + (dz_qn @ gy),
        (dx_qn @ gz) + (dz_qn @ gx),
        (dx_qn @ gy) + (dy_qn @ gx)], axis=1)
    u1_qd = jnp.einsum("qn,nd->qd", phi_qn, wl1_nd)
    u2_qd = jnp.einsum("qn,nd->qd", phi_qn, wl2_nd)
    z_q = jnp.zeros_like(u1_qd[:, 0])
    Gl_qs = jnp.stack([
        u1_qd[:, 0], u2_qd[:, 1], z_q,
        u2_qd[:, 2], u1_qd[:, 2],
        u1_qd[:, 1] + u2_qd[:, 0]], axis=1)
    dGam_qs = Gh_qs + Gl_qs
    dSig_qs = jnp.einsum("ij,qj->qi", C_ss, dGam_qs)
    return dGam_qs, dSig_qs


@partial(jax.jit, static_argnames=["n_sg"])
def _v266_batch(periodic_cells_en, V0lad, V11bar, V12bar, V11barD,
                V12barD, V21, V22, V23, V21t, V22t, V23t,
                x_end, C_ess, dphi_dxi_qnp, phi_qn,
                dE1, dE2, d11, d12, d22, n_sg):
    """The SECOND-ORDER Eq. 64-66 refined recovery -- the two-chain
    kernel of rm_plate_1D._warp_terms / msgrm_strain_at_depth on the
    general SG assembly:

      chain A (in-plane stress rows):
        w   = V11bar dE1 + V12bar dE2 + V21 d11 + V22 d12 + V23 d22
        g1  = V0 dE1 + V11barA d11 + V12barA d12
        g2  = V0 dE2 + V11barA d12 + V12barA d22
      chain B (rows 33/23/13):
        w_t = ... V21t/V22t/V23t ...;  g_at with the RAW V1bar columns

    THE TWO CHAINS ARE NOW FED IDENTICAL COLUMNS (2026-09-01, the TPMS
    correction).  Chain A used to take DETILTED columns -- an implicit
    stand-in for the R -> eps conversion, applied to the in-plane rows
    only.  Yu Eq. 50 now does that conversion explicitly, ON THE
    DRIVERS, before this kernel is reached, so detilting here as well
    would apply it twice.  The caller passes the raw V1bar/V2t bank
    into both slots, which makes the row split below a no-op and leaves
    this kernel's arithmetic untouched; the structure is kept only so
    it still reads against Yu's text.  Do NOT reintroduce a detilted
    bank in the A slots.

      dGam/dSig rows (11, 22, 12) from chain A, rows 2:5
      (33, 23, 13 -- the same slice in this kernel's SwiftComp order)
      from the T-chain, exactly msg_rm_plate lines 624-630.

    With zero second-order drivers both chains collapse to the Eq. 63
    first-order term and the split is a no-op (matches _v2_batch).
    The element kernel is _v2_batch's `one` body verbatim.

    In:  periodic_cells_en (E, N); the ladder/V2 column blocks
         (n_unique, 6) each; x_end (E, N, d); C_ess (E, 6, 6);
         dphi_dxi_qnp (Q, N, p); phi_qn (Q, N); dE1/dE2/d11/d12/d22
         (6,) drivers; n_sg static.
    Out: dGam_eqs, dSig_eqs (E, Q, 6), SwiftComp order."""
    gather = jax.vmap(transform_global_unraveled_to_element_node,
                      in_axes=(None, 1, None), out_axes=-1)
    g = {k: gather(periodic_cells_en, v, 3) for k, v in
         (("V0", V0lad), ("b1", V11bar), ("b2", V12bar),
          ("d1", V11barD), ("d2", V12barD),
          ("21", V21), ("22", V22), ("23", V23),
          ("21t", V21t), ("22t", V22t), ("23t", V23t))}
    con = lambda G_, v: jnp.einsum("ends,s->end", G_, v)   # noqa: E731
    whD = (con(g["b1"], dE1) + con(g["b2"], dE2) + con(g["21"], d11)
           + con(g["22"], d12) + con(g["23"], d22))
    g1D = con(g["V0"], dE1) + con(g["d1"], d11) + con(g["d2"], d12)
    g2D = con(g["V0"], dE2) + con(g["d1"], d12) + con(g["d2"], d22)
    whT = (con(g["b1"], dE1) + con(g["b2"], dE2) + con(g["21t"], d11)
           + con(g["22t"], d12) + con(g["23t"], d22))
    g1T = con(g["V0"], dE1) + con(g["b1"], d11) + con(g["b2"], d12)
    g2T = con(g["V0"], dE2) + con(g["b1"], d12) + con(g["b2"], d22)

    run = jax.vmap(_elem_chain, in_axes=(0, 0, 0, 0, 0, None, None))
    GamD, SigD = run(whD, g1D, g2D, x_end, C_ess, dphi_dxi_qnp, phi_qn)
    GamT, SigT = run(whT, g1T, g2T, x_end, C_ess, dphi_dxi_qnp, phi_qn)
    # ROW SPLIT: rows 2:5 (33, 23, 13) from chain B.  A NO-OP whenever
    # one bank is passed into both slots -- what the corrected
    # composition does; see this function's docstring.
    dGam = GamD.at[:, :, 2:5].set(GamT[:, :, 2:5])
    dSig = SigD.at[:, :, 2:5].set(SigT[:, :, 2:5])
    return dGam, dSig


@partial(jax.jit, static_argnames=["n_sg"])
def _vq_batch(periodic_cells_en, V1L, V2L, x_end, C_ess,
              dphi_dxi_qnp, phi_qn, q6, n_sg):
    """The PRESSURE-DRIVEN recovery addition of ONE face -- the qt6/qb6
    term of rm_plate_1D._warp_terms, on the general SG assembly:

        dGam = Gamma_h (V1L q + V2L [q,1 q,2 q,11 q,12 q,22])
             + Gamma_l1 (V1L q,1) + Gamma_l2 (V1L q,2)

    with q6 = [q, q,1, q,2, q,11, q,12, q,22] of that face's pressure.
    For a UNIFORM pressure only the Gamma_h V1L q term is nonzero.  The
    element kernel is _v2_batch's `one` body verbatim (kept in sync by
    hand -- the gated function is not refactored).

    In:
        periodic_cells_en: (E, N) periodic connectivity.
        V1L: (n_unique,) that face's first-order load column.
        V2L: (n_unique, 5) its second-order quintet.
        x_end: (E, N, d); C_ess: (E, 6, 6).
        dphi_dxi_qnp: (Q, N, p); phi_qn: (Q, N).
        q6: (6,) the face pressure and its in-plane derivatives.
        n_sg: static SG dimension.
    Out:
        dGam_eqs, dSig_eqs: (E, Q, 6), SwiftComp order."""
    gather = jax.vmap(transform_global_unraveled_to_element_node,
                      in_axes=(None, 1, None), out_axes=-1)
    VL_end = gather(periodic_cells_en, V1L[:, None], 3)[..., 0]
    V2_end5 = gather(periodic_cells_en, V2L, 3)
    wh_end = VL_end * q6[0] + jnp.einsum("ends,s->end", V2_end5, q6[1:])
    wl1_end = VL_end * q6[1]
    wl2_end = VL_end * q6[2]
    return jax.vmap(_elem_chain, in_axes=(0, 0, 0, 0, 0, None, None))(
        wh_end, wl1_end, wl2_end, x_end, C_ess, dphi_dxi_qnp, phi_qn)


@partial(jax.jit, static_argnames=["n_model", "n_sg"])
def _dehom_batch(periodic_cells_en, V0, x_end, C_ess, dphi_dxi_qnp,
                 phi_qn, epsilon_bar, n_model, n_sg):
    """All-element SSDM recovery (vmap of _element_dehomo_kernel),
    jitted once per (shape, n_model, n_sg); epsilon_bar is traced, so
    new load cases reuse the compiled kernel.

    In:
        periodic_cells_en: (E, N) periodic connectivity.
        V0: (n_unique, H) fluctuation solution.
        x_end: (E, N, d) element-node coordinates.
        C_ess: (E, 6, 6) per-element stiffness.
        dphi_dxi_qnp: (Q, N, p) parametric basis gradients.
        phi_qn: (Q, N) basis values.
        epsilon_bar: (H,) macro strain state.
        n_model, n_sg: static ints (1/2/3 model; SG dimension).
    Out:
        Gamma_eqs: (E, Q, 6) strain, SwiftComp order.
        Sigma_eqs: (E, Q, 6) stress, SwiftComp order.
    """
    get_element_V0 = jax.vmap(transform_global_unraveled_to_element_node,
                              in_axes=(None, 1, None), out_axes=-1)
    V0_endH = get_element_V0(periodic_cells_en, V0, 3)
    batched = jax.vmap(_element_dehomo_kernel,
                       in_axes=(0, 0, 0, None, None, None, None, None))
    return batched(V0_endH, x_end, C_ess, dphi_dxi_qnp, phi_qn,
                   epsilon_bar, n_model, n_sg)


# ---------------------------- beam recovery chain (from Beam_solid.py)
@jax.jit
def build_recovery_matrix(st):
    """6x6 equilibrium operator of the recovery chain from the 6 state
    components (Beam_solid.py)."""
    s0, s1, s2, s3, s4, s5 = st[0], st[1], st[2], st[3], st[4], st[5]

    R_top_left = jnp.array([
        [0., s5, -s4],
        [-s5, 0., s3],
        [s4, -s3, 0.]
    ])
    R_bottom_left = jnp.array([
        [0., s2, -s1],
        [-s2, 0., s0],
        [s1, -s0, 0.]
    ])
    R_top_right = jnp.zeros((3, 3))
    return jnp.block([
        [R_top_left, R_top_right],
        [R_bottom_left, R_top_left]
    ])


@jax.jit
def compute_fluctuations_gpu(Deff_srt, st, V0, V1):
    """Successive-derivative generalized-Timoshenko recovery ladder
    (product-rule chain F' = R F seeded with F_1d = Comp @ st): builds
    the shear-corrected states and warping displacement fields for
    beam (n_model 1) dehomogenization.

    In:
        Deff_srt: (6, 6) effective Timoshenko stiffness.
        st: (6,) sectional STRAIN [eps11, 2g12, 2g13, k1, k2, k3].
        V0: (N_primal, 4) zeroth-order warping modes.
        V1: (N_primal, 4) first-order warping modes.
    Out:
        w_1: (N_primal,) zeroth-order warping, V0 @ st_m_final.
        w1s_1: (N_primal,) first-order warping, V1 @ st_cl1_final.
        w1s_2: (N_primal,) first-order warping at the
            second-derivative state, V1 @ st_cl2_final.
        w_2: (N_primal,) zeroth-order warping at the first-derivative
            state, V0 @ st_cl1_final.
        st_m_final: (4,) shear-corrected classical state
            [eps11, k1, k2, k3].
    Constraint: with the compliance seed the transverse-shear entries
    st[1], st[2] are recovery-inert -- drive only states whose stress
    rides the classical channels (extension, twist, bending).
    """
    Comp_srt = jnp.linalg.inv(Deff_srt)
    st_m = jnp.array([st[0], st[3], st[4], st[5]], dtype=jnp.float64)

    F_1d = Comp_srt @ st
    st_plus_1 = st.at[0].add(1.0)
    R1 = build_recovery_matrix(st_plus_1)
    F1 = R1 @ F_1d

    st_Tim1 = Comp_srt @ F1
    st_cl1 = jnp.array([st_Tim1[0], st_Tim1[3], st_Tim1[4], st_Tim1[5]])
    gamma1 = jnp.array([st_Tim1[1], st_Tim1[2]])

    R2 = build_recovery_matrix(st_Tim1)
    F2 = (R1 @ F1) + (R2 @ F_1d)

    st_Tim2 = Comp_srt @ F2
    st_cl2 = jnp.array([st_Tim2[0], st_Tim2[3], st_Tim2[4], st_Tim2[5]])
    gamma2 = jnp.array([st_Tim2[1], st_Tim2[2]])

    R3 = build_recovery_matrix(st_Tim2)
    F3 = 2.0 * (R2 @ F1) + (R3 @ F_1d) + (R1 @ F2)

    st_Tim3 = Comp_srt @ F3
    gamma3 = jnp.array([st_Tim3[1], st_Tim3[2]])

    Q = jnp.array([[0., 0.], [0., 0.], [0., -1.], [1., 0.]],
                  dtype=jnp.float64)
    st_m_final = st_m + (Q @ gamma1)
    st_cl1_final = st_cl1 + (Q @ gamma2)
    st_cl2_final = st_cl2 + (Q @ gamma3)

    w_1 = V0 @ st_m_final
    w1s_1 = V1 @ st_cl1_final
    w1s_2 = V1 @ st_cl2_final
    w_2 = V0 @ st_cl1_final
    return w_1, w1s_1, w1s_2, w_2, st_m_final


@jax.jit
def get_Rsig_matrix(R_flat):
    """Voigt stress-transformation 6x6 from a flattened (9,) DC matrix
    (Beam_solid.py; the FEniCS R_sig convention, note the internal R.T)."""
    R = R_flat.reshape(3, 3)
    R = R.T
    b11, b12, b13 = R[0, 0], R[1, 0], R[2, 0]
    b21, b22, b23 = R[0, 1], R[1, 1], R[2, 1]
    b31, b32, b33 = R[0, 2], R[1, 2], R[2, 2]
    return jnp.array([
        [b11*b11, b12*b12, b13*b13, 2*b12*b13, 2*b11*b13, 2*b11*b12],
        [b21*b21, b22*b22, b23*b23, 2*b22*b23, 2*b21*b23, 2*b21*b22],
        [b31*b31, b32*b32, b33*b33, 2*b32*b33, 2*b31*b33, 2*b31*b32],
        [b21*b31, b22*b32, b23*b33, b23*b32 + b22*b33, b23*b31 + b21*b33,
         b22*b31 + b21*b32],
        [b11*b31, b12*b32, b13*b33, b13*b32 + b12*b33, b13*b31 + b11*b33,
         b12*b31 + b11*b32],
        [b11*b21, b12*b22, b13*b23, b13*b22 + b12*b23, b13*b21 + b11*b23,
         b12*b21 + b11*b22]
    ])


@partial(jax.jit, static_argnames=['n_model', 'n_sg'])
def recover_gauss_point_fields(
    w_1, w1s_1, w1s_2, w_2, st_m_final,
    dehomo_data, C_ess1, reduced_periodic_cells, phi_qn, x_end,
    elem_rotation, n_model, n_sg
):
    """Gauss-point strain/stress recovery of the beam chain from the
    warping fields of compute_fluctuations_gpu.

    In:
        w_1, w1s_1, w1s_2, w_2: (N_primal,) warping displacement
            fields (compute_fluctuations_gpu outputs).
        st_m_final: (4,) shear-corrected classical state.
        dehomo_data: dict {dphi_dx (E, Q, N, D), Ge (E, Q, 6, H),
            dV_q (E, Q)} from assembly.
        C_ess1: (E, 6, 6) or (E, Q, 6, 6) per-element
            PRE-elem-rotation stiffness.
        reduced_periodic_cells: (E, N) reduced connectivity.
        phi_qn: (Q, N) basis values.
        x_end: (E, N, d) element-node coordinates.
        elem_rotation: (E, 9) flattened per-element DC matrices.
        n_model, n_sg: static ints.
    Out:
        x_q: (E, Q, d) Gauss-point coordinates.
        st_3D_elem: (E, Q, 6) strain rotated into the element frame.
        stress_3D_local: (E, Q, 6) stress = C_ess1 @ strain.
    Stress comes out in the frame the strain was rotated into
    (identity elem_rotation -> the global SG frame, matching the
    plate dehom).
    """
    dphi_dx = dehomo_data["dphi_dx"]
    Ge = dehomo_data["Ge"]
    E_elem, Q, N_nodes, D = dphi_dx.shape

    x_q = jnp.einsum("qn,end->eqd", phi_qn, x_end)
    dof_map = ((reduced_periodic_cells * 3).reshape(E_elem, N_nodes, 1)
               + jnp.array([0, 1, 2])).reshape(E_elem, N_nodes * 3)

    uh_1 = w_1[dof_map].reshape(E_elem, N_nodes, 3)
    if n_model == 1:
        ul_1 = w1s_1[dof_map].reshape(E_elem, N_nodes, 3)
        ul_2 = w1s_2[dof_map].reshape(E_elem, N_nodes, 3)
        uh_2 = w_2[dof_map].reshape(E_elem, N_nodes, 3)

    zeros = jnp.zeros_like(dphi_dx[..., 0])
    grad_phi = jnp.stack([zeros, dphi_dx[..., 0], dphi_dx[..., 1]],
                         axis=-1) if D == 2 else dphi_dx

    def compute_eps_h_batch(uh):
        grad_u = jnp.einsum("eni,eqnj->eqij", uh, grad_phi)
        return jnp.stack([grad_u[..., 0, 0], grad_u[..., 1, 1],
                          grad_u[..., 2, 2],
                          grad_u[..., 1, 2] + grad_u[..., 2, 1],
                          grad_u[..., 0, 2] + grad_u[..., 2, 0],
                          grad_u[..., 0, 1] + grad_u[..., 1, 0]], axis=-1)

    def compute_eps_l_batch(ul):
        u_q = jnp.einsum("qn,eni->eqi", phi_qn, ul)
        z_q = jnp.zeros_like(u_q[..., 0])
        return jnp.stack([u_q[..., 0], z_q, z_q, z_q, u_q[..., 2],
                          u_q[..., 1]], axis=-1)

    def compute_eps_e_batch(ue):
        return jnp.einsum("eqip,p->eqi", Ge, ue)

    eps_Eb = compute_eps_h_batch(uh_1) + compute_eps_e_batch(st_m_final)
    eps_Timo = (compute_eps_h_batch(ul_1) + compute_eps_l_batch(uh_2)
                + compute_eps_l_batch(ul_2)) if n_model == 1 \
        else jnp.zeros_like(eps_Eb)

    st_3D_global = eps_Eb + eps_Timo

    Rsig_batch = jax.vmap(get_Rsig_matrix)(elem_rotation)
    st_3D_elem = jnp.einsum("eji,eqj->eqi", Rsig_batch, st_3D_global)

    C_q = jnp.repeat(C_ess1[:, None, :, :], Q, axis=1) \
        if C_ess1.ndim == 3 else C_ess1
    stress_3D_local = jnp.einsum("eqij,eqj->eqi", C_q, st_3D_elem)

    return x_q, st_3D_elem, stress_3D_local


# --------- the OpenSG dehom file convention (rm_plate_1D.rm_dehom):
# printed component order 11 22 33 12 13 23, pulled from the SwiftComp
# storage (xx yy zz yz xz xy) via SGIDX -- reorder happens ON WRITE ONLY
SGIDX = {"11": 0, "22": 1, "33": 2, "12": 5, "13": 4, "23": 3}
ORDER = ("11", "22", "33", "12", "13", "23")


def _u_at_gauss(u_flat, cells_en, phi_qn):
    """(E, Q, 3) displacement at the Gauss points from an unraveled
    nodal vector on the (periodic-reduced) node space."""
    u_en = jnp.asarray(u_flat).reshape(-1, 3)[jnp.asarray(cells_en)]
    return np.asarray(jnp.einsum("qn,end->eqd", phi_qn, u_en))


# ------------------------------------------------------------- the public API
def dehom_fields(r: Dict[str, Any],
                 epsilon_bar: Sequence[float],
                 dE1: Sequence[float] = None,
                 dE2: Sequence[float] = None,
                 Q: Sequence[float] = None,
                 qt6: Sequence[float] = None,
                 qb6: Sequence[float] = None,
                 dE11: Sequence[float] = None,
                 dE12: Sequence[float] = None,
                 dE22: Sequence[float] = None,
                 dEhi: Dict[str, Sequence[float]] = None):
    """One recovery pass: the Gauss strain/stress AND the fluctuation
    displacement (nothing recomputed -- plate_dehom_2d is a view of the
    same pass).

    In:  r (plate_homo_2d dict); epsilon_bar as plate_dehom_2d;
         dE1/dE2 (6,) OPTIONAL in-plane derivatives of the plate
         measures (d eps / d x1, d x2) -- when given (plate, refined
         run) the Eq. 63 first-order refined term is ADDED via the
         stored shear-ladder warpings (V0_ladder/V11bar/V12bar); with
         both None the output is the classical recovery, digit-
         identical to before;
         Q (2,) OPTIONAL transverse shear resultants [Q1 Q2] -- when
         given, the recovered sigma_13/sigma_23 profiles are RESCALED
         per SG so their thickness-integrated force (1/omega
         integral sigma_a3 dV) equals Q_a exactly (the rm_dehom
         Q-consistency rule: profile shape from the gradients, carried
         force from the plate solution); skipped for a channel whose
         integral is < 1e-3 of max|Q|.
         qt6/qb6 (6,) OPTIONAL = [q, q,1, q,2, q,11, q,12, q,22] of the
         TOP/BOTTOM face pressure (q positive pushing INTO the face) --
         the rm_plate_1D load-ladder recovery on the 2-D SG: adds the
         pressure-driven local field (sigma33 face content, sigma22
         load content) that no resultant state carries.  Needs the
         V1Lt/V2Lt columns from a refined 2-D plate homogenization.
         dE11/dE12/dE22 (6,) OPTIONAL second in-plane derivatives of
         the plate measures (E,11 E,12 E,22) -- when any is given the
         SECOND-ORDER Eq. 64-66 recovery replaces the first-order term:
         the two V2 chains, fed the SAME raw V1bar/V2t bank since the
         2026-09-01 TPMS correction (Yu Eq. 50 converts R -> eps on the
         drivers instead, so the old in-plane detilt would double it).
         Needs V21..V23t from a refined 2-D plate homogenization.
    Out: (Gamma_eqs, Sigma_eqs, U_eqd) -- (E, Q, 6), (E, Q, 6),
         (E, Q, 3) np.  U is the FLUCTUATION-ONLY displacement at the
         Gauss points (no macro contribution exists for a unit-state SG
         recovery -- the single-station rm_dehom precedent): n_model
         2/3 = V0 @ epsilon_bar (+ V11 dE1 + V12 dE2 on the refined
         path, mean-zero columns); n_model 1 = the recovery chain's
         zeroth + first-order warping fields (w_1 + w1s_1)."""
    if r["n_model"] == 1:
        st = jnp.asarray(epsilon_bar, float)
        if st.shape != (6,):
            raise ValueError("n_model=1 dehom takes the 6-component beam"
                             " state [x1 ext, shear2, shear3, twist,"
                             " bend2, bend3]")
        w_1, w1s_1, w1s_2, w_2, st_m_final = compute_fluctuations_gpu(
            jnp.asarray(r["C_eff"]), st, r["V0"], r["V1s"])
        _, Gam, Sig = recover_gauss_point_fields(
            w_1, w1s_1, w1s_2, w_2, st_m_final, r["dehomo_data"],
            r["C_stress"], r["reduced_periodic_cells"], r["phi_qn"],
            r["x_end"], r["elem_rotation"], 1, r["n_sg"])
        U = _u_at_gauss(w_1 + w1s_1, r["reduced_periodic_cells"],
                        r["phi_qn"])
        return np.asarray(Gam), np.asarray(Sig), U

    second = (dE11 is not None or dE12 is not None or dE22 is not None)
    refined = (dE1 is not None or dE2 is not None or second)
    if refined and r.get("V11bar") is None:
        raise ValueError(
            "strain-derivative recovery needs the shear ladder: run the"
            " homogenization with refined: 1 (plate) so V0_ladder/"
            "V11bar/V12bar are stored")
    if second and r.get("V21") is None:
        raise ValueError(
            "second-order (Eq. 64-66) recovery needs the V2 chains:"
            " a refined single-batch plate homogenization (any SG"
            " dimension) stores V21..V23t; mixed SGs do not carry them"
            " yet")
    if refined:
        d1 = jnp.zeros(6) if dE1 is None else jnp.asarray(dE1, float)
        d2 = jnp.zeros(6) if dE2 is None else jnp.asarray(dE2, float)
    if second:
        d11 = jnp.zeros(6) if dE11 is None else jnp.asarray(dE11, float)
        d12 = jnp.zeros(6) if dE12 is None else jnp.asarray(dE12, float)
        d22 = jnp.zeros(6) if dE22 is None else jnp.asarray(dE22, float)
        # ---- Yu Eq. 50 in DRIVER space (2026-09-01) -----------------
        # The macro state handed in is the REISSNER measure R; Gamma_eps
        # consumes the CLASSICAL measure eps.  Eq. 50 converts them,
        #     eps = R - D_a gamma,a
        # and Eq. 54 -- the equilibrium relation Yu uses to eliminate the
        # strain-measure derivatives -- supplies gamma,a:
        #     G gamma + F_c = D1^T A R,1 + D2^T A R,2 + m
        #  => gamma,a = G^-1 (D1^T A R,1a + D2^T A R,2a)      (m = 0)
        # R,1a and R,2a ARE the second-derivative drivers, so nothing new
        # is needed: no transverse-shear measure, no Q column.  The same
        # correction on the FIRST-derivative drivers would need third
        # derivatives and is one order higher, so it is dropped.
        #
        # This REPLACES the detilt, which existed only as an implicit
        # stand-in for the same conversion (rm_plate_1D._detilt_inplane
        # docstring).  Doing both double-counts; doing neither is worse
        # still.  Measured on the Pagano [0/90/0] s=4 gate against the
        # statically determinate moment (Q1 = 0, so every stage above V0
        # must add ZERO moment): the second-order stage injected 12.31 %
        # of the applied M11 with the detilt, and 0.01 % with this.
        from opensg_solid.sg_homo import (      # call-time: no cycle
            _D1_SG, _D2_SG)
        _A6 = r.get("A6_ladder")
        _G = r.get("G_msg")
        if _A6 is not None and _G is not None:
            A6 = jnp.asarray(_A6, float)[:6, :6]
            Gi = jnp.linalg.inv(jnp.asarray(_G, float))
            D1, D2 = jnp.asarray(_D1_SG), jnp.asarray(_D2_SG)
            g1 = Gi @ (D1.T @ A6 @ d11 + D2.T @ A6 @ d12)
            g2 = Gi @ (D1.T @ A6 @ d12 + D2.T @ A6 @ d22)
            epsilon_bar = (jnp.asarray(epsilon_bar, float)
                           - (D1 @ g1 + D2 @ g2))
            # ---- the SAME conversion on the DERIVATIVE drivers ------
            # Eqs. 63/64/66 are written in the classical measure
            # throughout -- V1 = V11 eps,1 + V12 eps,2 and
            # V2 = V21 eps,11 + V22 eps,12 + V23 eps,22 -- so leaving
            # d1/d2 and d11/d12/d22 as the Reissner R,a / R,ab is a
            # measure mismatch, not a higher-order remainder.  It was
            # dropped on the grounds of being "one order higher", but on
            # this SG the fourth derivative measures ~0.93 of the second
            # (converged to 1.9% between subdiv 3 and 5), so that
            # argument does not hold here.
            #   c,b    = G^-1 (D1^T A R,1b   + D2^T A R,2b)
            #   c,b..  = G^-1 (D1^T A R,1b.. + D2^T A R,2b..)
            #   eps,X  = R,X - (D1 c,1X + D2 c,2X)
            # Mixed partials commute, so a family is looked up by its
            # SORTED multi-index.  c is built from R throughout (Eq. 54
            # is a relation on R), so the raw drivers feed it -- only
            # the drivers handed onward are converted, and the order
            # below matters: epsilon_bar above already consumed the
            # unconverted d11/d12/d22, which is correct.
            if dEhi:
                def _hi(ix):
                    return jnp.asarray(dEhi["".join(sorted(ix))], float)

                def _cd(b, rest):
                    """c,b differentiated by the multi-index `rest`."""
                    return Gi @ (D1.T @ A6 @ _hi("1" + b + rest)
                                 + D2.T @ A6 @ _hi("2" + b + rest))

                _h3 = all(k in dEhi for k in ("111", "112", "122",
                                              "222"))
                _h4 = all(k in dEhi for k in ("1111", "1112", "1122",
                                              "1222", "2222"))
                if _h3:                       # eps,a  (drives V1)
                    d1 = d1 - (D1 @ _cd("1", "1") + D2 @ _cd("2", "1"))
                    d2 = d2 - (D1 @ _cd("1", "2") + D2 @ _cd("2", "2"))
                if _h4:                       # eps,ab (drives V2)
                    d11 = d11 - (D1 @ _cd("1", "11")
                                 + D2 @ _cd("2", "11"))
                    d12 = d12 - (D1 @ _cd("1", "12")
                                 + D2 @ _cd("2", "12"))
                    d22 = d22 - (D1 @ _cd("1", "22")
                                 + D2 @ _cd("2", "22"))
                print(" Eq. 50 on the derivative drivers: first-order"
                      " %s, second-order %s"
                      % ("converted" if _h3 else "SKIPPED (need d3eps)",
                         "converted" if _h4 else "SKIPPED (need d4eps)"))
        else:
            print(" WARNING: Eq. 50 conversion SKIPPED (no A6_ladder/"
                  "G_msg) -- NOT the corrected composition.  The"
                  " second-order in-plane rows fall back to the"
                  " detilt's implicit conversion, which the 2026-09-01"
                  " TPMS correction replaced; rerun from a refined"
                  " (refined: 1) single-batch plate homogenization to"
                  " get the Eq. 50 route.")
    qpairs = [(q6, v1, v2) for q6, v1, v2
              in ((qt6, "V1Lt", "V2Lt"), (qb6, "V1Lb", "V2Lb"))
              if q6 is not None]
    if qpairs and r.get("V1Lt") is None:
        raise ValueError(
            "face-pressure recovery (qt6/qb6) needs the load ladder:"
            " a refined (refined: 1) single-batch plate homogenization"
            " (any SG dimension) stores V1Lt/V2Lt/V1Lb/V2Lb; mixed SGs"
            " do not carry them yet")

    # a MIXED SG stores the per-mesh arrays as PER-BATCH LISTS (one per
    # element type, shared dof space) -- run every kernel per batch; a
    # single-batch run keeps its (E, Q, .) shapes, a mixed run returns
    # FLAT (P, .) Gauss clouds (concatenated in batch order, matching
    # gauss_coords)
    mixed = isinstance(r["x_end"], (list, tuple))
    _l = lambda v: v if isinstance(v, (list, tuple)) else [v]  # noqa: E731
    B = list(zip(_l(r["periodic_cells_en"]), _l(r["x_end"]),
                 _l(r["C_ess"]), _l(r["dphi_dxi_qnp"]), _l(r["phi_qn"]),
                 _l(r["W_q"])))

    Gam_b, Sig_b, dV_b = [], [], []
    for cells_b, x_b, C_b, dp_b, ph_b, W_b in B:
        Gam, Sig = _dehom_batch(cells_b, r["V0"], x_b, C_b, dp_b, ph_b,
                                jnp.asarray(epsilon_bar, float),
                                r["n_model"], r["n_sg"])
        Gam, Sig = np.asarray(Gam), np.asarray(Sig)
        if second:
            # Eq. 50 is now applied to the DRIVERS above, so the
            # detilted columns must NOT be used as well -- the RAW
            # V1bar/V2 bank feeds every row (the tilt/detilt row split
            # collapses).  Passing the raw columns in both slots keeps
            # the kernel unchanged.
            dGam, dSig = _v266_batch(
                cells_b, jnp.asarray(r["V0_ladder"]),
                jnp.asarray(r["V11bar"]), jnp.asarray(r["V12bar"]),
                jnp.asarray(r["V11bar"]), jnp.asarray(r["V12bar"]),
                jnp.asarray(r["V21t"]), jnp.asarray(r["V22t"]),
                jnp.asarray(r["V23t"]), jnp.asarray(r["V21t"]),
                jnp.asarray(r["V22t"]), jnp.asarray(r["V23t"]),
                x_b, C_b, dp_b, ph_b, d1, d2, d11, d12, d22,
                r["n_sg"])
            Gam = Gam + np.asarray(dGam)
            Sig = Sig + np.asarray(dSig)
        elif refined:
            dGam, dSig = _v2_batch(
                cells_b, jnp.asarray(r["V0_ladder"]),
                jnp.asarray(r["V11bar"]), jnp.asarray(r["V12bar"]),
                x_b, C_b, dp_b, ph_b, d1, d2, r["n_sg"])
            Gam = Gam + np.asarray(dGam)
            Sig = Sig + np.asarray(dSig)
        for q6, kv1, kv2 in qpairs:
            dGam, dSig = _vq_batch(
                cells_b, jnp.asarray(r[kv1]), jnp.asarray(r[kv2]),
                x_b, C_b, dp_b, ph_b, jnp.asarray(q6, float),
                r["n_sg"])
            Gam = Gam + np.asarray(dGam)
            Sig = Sig + np.asarray(dSig)
        if Q is not None and refined:
            J = jnp.einsum("end,qnp->eqdp", jnp.asarray(x_b), dp_b)
            detJ = (jnp.abs(jnp.linalg.det(J)) if r["n_sg"] > 1
                    else jnp.abs(J[..., 0, 0]))
            dV_b.append(np.asarray(detJ * W_b[None, :]))
        Gam_b.append(Gam)
        Sig_b.append(Sig)

    if Q is not None and refined:
        # Q-consistency: ONE global integral over ALL batches -- the
        # carried force is a property of the whole SG, so a single
        # factor rescales slot 4 (xz -> Q1) / slot 3 (yz -> Q2)
        Qv = np.asarray(Q, float).reshape(2)
        qmax = max(abs(Qv[0]), abs(Qv[1]), 1e-30)
        for qi, slot in ((0, 4), (1, 3)):
            I = sum(float((S[:, :, slot] * dV).sum())
                    for S, dV in zip(Sig_b, dV_b)) / r["omega"]
            if abs(I) > 1e-3 * qmax:
                for S in Sig_b:
                    S[:, :, slot] *= Qv[qi] / I

    u_flat = jnp.asarray(r["V0"]) @ jnp.asarray(epsilon_bar, float)
    if refined:
        u_flat = (jnp.asarray(r["V0_ladder"])
                  @ jnp.asarray(epsilon_bar, float)
                  + jnp.asarray(r["V11"]) @ d1
                  + jnp.asarray(r["V12"]) @ d2)
    if second:
        # the physical (tilted) V2 warping, as in _warp_terms' w_t.
        # DELIBERATE deviation from msgrm_warping_at_depth (which uses
        # the untilted V21/V22/V23): affects the .U export only, never
        # stress/strain.
        u_flat = (u_flat + jnp.asarray(r["V21t"]) @ d11
                  + jnp.asarray(r["V22t"]) @ d12
                  + jnp.asarray(r["V23t"]) @ d22)
    for q6, kv1, kv2 in qpairs:
        qv = jnp.asarray(q6, float)
        u_flat = (u_flat + jnp.asarray(r[kv1]) * qv[0]
                  + jnp.asarray(r[kv2]) @ qv[1:])
    U_b = [_u_at_gauss(u_flat, cells_b, ph_b)
           for (cells_b, _x, _C, _dp, ph_b, _W) in B]

    if not mixed:
        return Gam_b[0], Sig_b[0], U_b[0]
    return (np.concatenate([g.reshape(-1, 6) for g in Gam_b]),
            np.concatenate([s.reshape(-1, 6) for s in Sig_b]),
            np.concatenate([u.reshape(-1, 3) for u in U_b]))


def load_column_F(r: Dict[str, Any],
                  q_top: float = 0.0,
                  q_bot: float = 0.0,
                  dq_top: Sequence[float] = (0.0, 0.0),
                  dq_bot: Sequence[float] = (0.0, 0.0)):
    """The load-related constitutive term F, straight from Yu's Eq. 47.

    Yu, Hodges & Volovoi, Comput. Struct. 81 (2003) 439-454.  The
    Reissner-like energy is 2 Pi_R = R^T A R + gamma^T G gamma + 2 R^T F
    (Eq. 61), so the plate law carries one term more than a pure
    stiffness:

        {N; M} = [ABD] {eps; kappa} + F

    with, from Eq. 47,

        F = V0^T L - 1/2 (D1^T V1L,1 + V11^T L,1
                          + D2^T V1L,2 + V12^T L,2)
        P = V1L^T L                     (quadratic in the load; Yu drops
                                         it from Eq. 61 because it does
                                         not affect the 2-D equations)

    and L itself, from the line after Eq. 28,

        L = S+^T s - S-^T b - <S^T phi>

    the consistent nodal load of the top traction s, the bottom traction
    b and the body force phi.  That vector is assembled during
    homogenization and exported as r["L_faces"], one column per face.

    HOW THE STATION ENTERS.  L and V1L are LINEAR in the applied load, so
    for a pressure field q(x1, x2) on a face

        L(x) = q(x) L_u        V1L(x) = q(x) V1L_u
        L,a  = q,a  L_u        V1L,a  = q,a  V1L_u

    and Eq. 47 separates into SG-level blocks times station-level scalars:

        F = q F_unit - 1/2 ( q,1 G1_unit + q,2 G2_unit )

        F_unit  = V0^T L_u                          (6, 2)
        G1_unit = D1^T V1L_u + V11^T L_u            (6, 2)
        G2_unit = D2^T V1L_u + V12^T L_u            (6, 2)

    per face, summed over faces, PER UNIT WIDTH of the plate (L_u is
    assembled already divided by the SG measure omega, the same
    normalization as the ABD; until 2026-09-08 this function multiplied
    by omega once more and was omega times too large on every SG with
    omega != 1).  The three blocks are formed once in the
    homogenization (D1, D2 are the Eq. 43-44 right-hand sides of the
    V11/V12 solves) and the station supplies only q, q,1 and q,2 -- so a
    sinusoidal load costs nothing more than a uniform one.  For a uniform
    pressure q,a = 0 and the whole bracket drops: F = q F_unit.

    WHY THIS REPLACES AN INTEGRATION.  F was previously obtained by
    running a full dehomogenization at ZERO macro strain -- so the
    recovered field is the load column alone -- and then integrating
    sigma and sigma*z over the cell.  That costs a recovery solve and
    only ever yields F for the one load it was run with.  Eq. 47 gives
    the same number as a matrix product.  Verified on the Schwarz-P
    rho = 0.3 cell at q = 7946.1 Pa, uniform:

        integrated   F_N = (-567.876, -567.876)  F_M = (-90.908, -90.908)
        Eq. 47       F_N = (-567.876, -567.876)  F_M = (-90.923, -90.923)
        ratio             1.000000                    1.00016

    Yu's own validation (Sec. 6.1) is a SINUSOIDAL split load
    s3 = b3 = (p0/2) sin(pi x1/L), where q,1 is live and the G1 block
    matters; that is why it is carried here and not dropped.

    In:  r -- a plate_homo_2d dict carrying "F_unit" (present whenever
             the run was shear-refined with recovery); q_top, q_bot
             float [Pa] the pressures on the two faces at this station
             (positive pushes into the face, the deck convention);
             dq_top, dq_bot -- (q,1, q,2) [Pa/m] of each face's pressure
             at this station, default (0, 0) = uniform
    Out: {"F" (6,) [F_N11 F_N22 F_N12 F_M11 F_M22 F_M12] in N/m and N,
          "F_N" (3,), "F_M" (3,), "P" float}

    Raises KeyError when r has no "F_unit" -- rerun the homogenization
    with recovery enabled."""
    for k in ("F_unit", "G1_unit", "G2_unit", "L_faces"):
        if k not in r:
            raise KeyError("this homogenization carries no %s; rerun"
                           " plate_homo_2d with recovery=True so the"
                           " load columns are assembled" % k)
    # NO omega factor here (fixed 2026-09-08).  L_faces is assembled per
    # unit face pressure ALREADY divided by omega (see
    # compute_fluctuations_gpu: "f_faces ... already /omega"), so V0^T L
    # is the resultant PER UNIT WIDTH -- the same normalization as the
    # ABD (C_eff = (D_bar + D1)/omega).  Multiplying by omega made F
    # omega times too large on every SG whose measure is not 1: on the
    # HC honeycomb (omega = 7.5256 mm) F_N was 7.5x the integrated
    # zero-strain resultant, and under a x100 geometric scaling F grew
    # x1e4 instead of x100.  The TPMS unit cube (omega = 1) hid it.
    # Gate: Final_pipeline_metric/scripts/gate_load_column_omega.py.
    Fu = np.asarray(r["F_unit"], float)
    G1 = np.asarray(r["G1_unit"], float)
    G2 = np.asarray(r["G2_unit"], float)
    q = np.array([q_top, q_bot], float)
    q1 = np.array([dq_top[0], dq_bot[0]], float)
    q2 = np.array([dq_top[1], dq_bot[1]], float)
    F = Fu @ q - 0.5 * (G1 @ q1 + G2 @ q2)
    # P = V1L^T L, quadratic in the load, reported for completeness
    # (same per-unit-width normalization)
    L = np.asarray(r["L_faces"], float)
    V1L = (q_top * np.asarray(r["V1Lt"], float)
           + q_bot * np.asarray(r["V1Lb"], float))
    P = float(V1L @ (L @ q))
    return {"F": F, "F_N": F[:3], "F_M": F[3:], "P": P}


def plate_dehom_2d(r: Dict[str, Any],
                   epsilon_bar: Sequence[float]):
    """Recover the local 3-D strain/stress at every element Gauss point.

    In:  r (plate_homo_2d dict); epsilon_bar (6,) the MACRO plate
         strain [e11, e22, 2e12, k11, k22, 2k12] for n_model=2/3, or
         for n_model=1 the beam state st [x1 ext, shear2, shear3,
         twist, bend2, bend3] of the Timoshenko recovery chain
         (Beam_solid.py convention; see compute_fluctuations_gpu for
         which states are valid recovery drivers)
    Out: (Gamma_eqs, Sigma_eqs) -- (E, Q, 6) np each, SwiftComp order
         (xx, yy, zz, yz, xz, xy)."""
    Gam, Sig, _ = dehom_fields(r, epsilon_bar)
    return Gam, Sig


def gauss_coords(r: Dict[str, Any]) -> np.ndarray:
    """(P, n_sg) physical coordinates of every recovery point (a mixed
    SG concatenates its per-batch clouds in the same batch order as
    dehom_fields)."""
    if isinstance(r["x_end"], (list, tuple)):
        return np.concatenate([
            np.asarray(jnp.einsum('qn, end -> eqd', ph, x))
            .reshape(-1, r["n_sg"])
            for ph, x in zip(r["phi_qn"], r["x_end"])])
    g = jnp.einsum('qn, end -> eqd', r["phi_qn"], r["x_end"])
    return np.asarray(g).reshape(-1, r["n_sg"])


def material_frame_fields(Gam, Sig, r):
    """Rotate the dehom Gauss clouds from the SG-global frame into each
    element's PLY MATERIAL frame (SwiftComp storage xx yy zz yz xz xy).

    The block `angle: a` bakes the ply stiffness INTO the SG frame
    (sg_materials), so the recovery output is SG-global; the material
    frame is recovered by rotating into the fiber frame,
    theta = +a about the SG THICKNESS axis (the last SG coordinate).
    The sign tracks the unified VABS-sign convention of 2026-08-25
    (OpenSG `angle: a` == Abaqus `*Orientation ..., 3, a`); the pm45
    3-D material-frame gate (ply RMS 2.6-6.6 %, the model error, vs
    46-264 % with the wrong sign) was run under the pre-unification
    mirror PAIR (engine -a, deck `3, -a`) -- both sides flip together,
    so the gate parity holds for a deck regenerated from the same
    yaml.

    Strain columns are engineering (gamma = 2 eps); the shear pair
    (yz, xz) rotates as a vector so only the (xx, yy, xy) row needs the
    half/double bookkeeping, done inline below.

    A yaml with per-element frames (elementOrientations) is REFUSED:
    there the material frame composes with the element frame and that
    path is not gated -- run --global instead.

    In:  Gam, Sig -- (E, Q, 6) single-batch or flat (P, 6) mixed clouds
         (dehom_fields output); r -- the plate_homo_2d dict ("sc" for
         mat_id/materials, "batch_idx" for the mixed order).
    Out: (Gam_m, Sig_m) same shapes, material frame.  The fluctuation
         displacement is NOT rotated (a per-element vector rotation of
         a continuous field is meaningless) -- .U stays SG-global."""
    sc = r["sc"]
    if sc.get("elementOrientations") is not None:
        raise ValueError(
            "--material is gated only for block-angle yamls; this SG"
            " carries per-element frames (elementOrientations) -- use"
            " --global")
    ang = {int(k): float(m.get("angle", 0.0) or 0.0)
           for k, m in sc["materials"].items()}
    a_e = np.array([ang[int(m_)] for m_ in np.asarray(sc["mat_id"], int)])

    Gs, Ss = np.asarray(Gam, float), np.asarray(Sig, float)
    if Ss.ndim == 2:                      # mixed: flat, per-batch order
        th = np.concatenate([
            np.repeat(a_e[idx], ph.shape[0])
            for idx, ph in zip(r["batch_idx"], r["phi_qn"])])
    else:                                 # single batch: element order
        th = np.repeat(a_e, Ss.shape[1]).reshape(Ss.shape[:2])
    c = np.cos(np.radians(th))
    s = np.sin(np.radians(th))

    def rot(F, eng):
        """storage (xx yy zz yz xz xy).  eng flags ENGINEERING shear
        columns (gamma = 2 eps): the xy column enters the normal rows
        halved (k) and the (yy - xx) term enters the xy row doubled (m);
        the (yz, xz) pair rotates as a plain vector either way, the
        factor 2 cancels."""
        xx, yy, zz = F[..., 0], F[..., 1], F[..., 2]
        yz, xz, xy = F[..., 3], F[..., 4], F[..., 5]
        k = 1.0 if eng else 2.0
        m = 2.0 if eng else 1.0
        R = np.empty_like(F)
        R[..., 0] = c*c*xx + s*s*yy + k*c*s*xy
        R[..., 1] = s*s*xx + c*c*yy - k*c*s*xy
        R[..., 2] = zz
        R[..., 3] = c*yz - s*xz
        R[..., 4] = s*yz + c*xz
        R[..., 5] = m*c*s*(yy - xx) + (c*c - s*s)*xy
        return R

    return rot(Gs, True), rot(Ss, False)


def _write_field(path, name, unit, F, xyz, lattice_note, title):
    """The rm_plate_1D .SM/.EM/.U text layout (rm_dehom.write_field):
    rows x y z + components, printed order 11 22 33 12 13 23 (the SGIDX
    reorder happens HERE -- storage stays SwiftComp xx yy zz yz xz xy),
    or U1 U2 U3 for a 3-column field."""
    F = np.asarray(F).reshape(-1, F.shape[-1])
    if F.shape[-1] == 6:
        cols = ["%s%s[%s]" % (name, c, unit) for c in ORDER]
        F = F[:, [SGIDX[c] for c in ORDER]]
    else:
        cols = ["U1[m]", "U2[m]", "U3[m]"]
    with open(path, "w") as f:
        f.write("# %s\n# lattice: %s; z = 0 is the reference surface\n"
                % (title, lattice_note))
        f.write("# %12s %14s %14s" % ("x[m]", "y[m]", "z[m]")
                + "".join(" %14s" % c for c in cols) + "\n")
        np.savetxt(f, np.hstack([xyz, F]), fmt="%14.6e", delimiter=" ")


def _vtk_corner_cell(n_sg, N):
    """VTK linear cell (type id, corner index list into the element's
    basix-ordered nodes) for an n_sg-dim N-node element.  Higher-order
    elements degrade to their CORNERS -- the painted field is
    element-constant, so nothing is lost visually.  quad4/hex8 undo the
    _to_basix_order tensor swap (self-inverse permutations); every other
    supported type keeps gmsh order with corners first.

    In:  n_sg int SG dimension; N int nodes per element.
    Out: (vtk_type, [corner indices]) | None when unmapped."""
    if n_sg == 1 and N >= 2:
        return 3, [0, 1]                          # VTK_LINE
    if n_sg == 2 and N in (3, 6):
        return 5, [0, 1, 2]                       # VTK_TRIANGLE
    if n_sg == 2 and N in (4, 9):
        return 9, [0, 1, 3, 2]                    # VTK_QUAD (de-basix)
    if n_sg == 3 and N in (4, 10):
        return 10, [0, 1, 2, 3]                   # VTK_TETRA
    if n_sg == 3 and N == 8:
        return 12, [0, 1, 3, 2, 4, 5, 7, 6]       # VTK_HEXAHEDRON
    return None


def export_gauss(r, Gamma_eqs, Sigma_eqs, prefix="gauss_results",
                 U_eqd=None, frame=None):
    """The .txt table + rendered .vtk of the recovered Gauss fields,
    PLUS the OpenSG dehom files <prefix>.SM (stress) / .EM (strain) and
    -- when U_eqd from dehom_fields is given -- .U (fluctuation
    displacement).

    The .vtk is the EXPLODED parent-mesh rendering: every element is
    emitted with its OWN copy of its corner nodes and carries the
    per-element MEAN of its Q Gauss values, so ParaView draws a filled
    contour that stays discontinuous across ply/material boundaries
    (never node-averaged -- junction bleed).  Element-constant is the
    honest rendering of Gauss data on the SG mesh: a bilinear
    GP-to-corner extrapolation overshoots ~50x the element's own spread
    on thin plies.  The raw one-row-per-Gauss-point cloud stays in the
    .txt; an element type without a VTK map falls back to the old
    point-cloud .vtk with a printed note.

    File component order is the rm_plate_1D convention (11 22 33 12 13
    23; units follow the SG input system); the (x, y, z) columns carry
    the SG coordinates in the LAST n_sg slots (y3 = thickness/last SG
    coordinate -> the z column), leading slots zero.  The .vtk (like
    the .SM/.EM) is in whatever FRAME the caller recovered -- the
    `frame` string is stamped into its title line."""
    coords = gauss_coords(r)
    strains = np.asarray(Gamma_eqs).reshape(-1, 6)
    stresses = np.asarray(Sigma_eqs).reshape(-1, 6)
    d = coords.shape[1]
    num_pts = strains.shape[0]

    gp = np.arange(1, num_pts + 1).reshape(-1, 1)
    combined = np.hstack((gp, coords, strains, stresses))
    coord_h = "\t".join("Coord_%s" % c for c in "XYZ"[:d])
    header = ("GP_Index\t" + coord_h +
              "\tEps_xx\tEps_yy\tEps_zz\tEps_yz\tEps_xz\tEps_xy"
              "\tSig_xx\tSig_yy\tSig_zz\tSig_yz\tSig_xz\tSig_xy")
    np.savetxt(prefix + ".txt", combined,
               fmt="\t".join(['%d'] + ['%.6e'] * (d + 12)),
               header=header, comments='', delimiter='\t')

    names = ["xx", "yy", "zz", "yz", "xz", "xy"]

    def _cloud_vtk():
        """The pre-2026-08 point-cloud .vtk -- the fallback for an
        element type _vtk_corner_cell does not map."""
        coords3 = np.hstack((coords, np.zeros((num_pts, 3 - d)))) \
            if d < 3 else coords
        with open(prefix + ".vtk", "w") as f:
            f.write("# vtk DataFile Version 3.0\nGauss Point Voigt Data"
                    " (%s)\nASCII\nDATASET UNSTRUCTURED_GRID\n"
                    % (frame or "frame unspecified"))
            f.write("POINTS %d float\n" % num_pts)
            np.savetxt(f, coords3, fmt='%.6e')
            f.write("\nCELLS %d %d\n" % (num_pts, num_pts * 2))
            np.savetxt(f, np.column_stack(
                (np.ones(num_pts, dtype=int),
                 np.arange(num_pts, dtype=int))), fmt='%d')
            f.write("\nCELL_TYPES %d\n" % num_pts)
            np.savetxt(f, np.ones(num_pts, dtype=int), fmt='%d')
            f.write("\nPOINT_DATA %d\n" % num_pts)
            for i, nm in enumerate(names):
                f.write("SCALARS Eps_%s float 1\nLOOKUP_TABLE default\n"
                        % nm)
                np.savetxt(f, strains[:, i], fmt='%.6e')
            for i, nm in enumerate(names):
                f.write("SCALARS Sig_%s float 1\nLOOKUP_TABLE default\n"
                        % nm)
                np.savetxt(f, stresses[:, i], fmt='%.6e')

    # ---- the exploded parent-mesh .vtk (per-batch: a mixed SG's flat
    # clouds are consumed in the same batch order dehom_fields wrote)
    _l = lambda v: v if isinstance(v, (list, tuple)) else [v]  # noqa: E731
    xbs = [np.asarray(x) for x in _l(r["x_end"])]
    Qs = [ph.shape[0] for ph in _l(r["phi_qn"])]
    maps = [_vtk_corner_cell(r["n_sg"], x.shape[1]) for x in xbs]
    if any(m is None for m in maps):
        bad = [x.shape[1] for x, m in zip(xbs, maps) if m is None]
        print("export_gauss: no VTK cell map for n_sg=%d, %s-node"
              " elements -- point-cloud .vtk written instead"
              % (r["n_sg"], bad))
        _cloud_vtk()
    else:
        mat_e = None
        if r.get("sc") is not None and "mat_id" in r["sc"]:
            mids = np.asarray(r["sc"]["mat_id"], int)
            mat_e = ([mids[np.asarray(i)] for i in r["batch_idx"]]
                     if "batch_idx" in r else [mids])
        pts, conn, ctyp, e_val, e_mat = [], [], [], [], []
        off = row = 0
        for bi, (x_b, Q, (vt, corner)) in enumerate(zip(xbs, Qs, maps)):
            Eb, _N, dd = x_b.shape
            k = len(corner)
            xc = np.zeros((Eb, k, 3))
            xc[:, :, :dd] = x_b[:, corner, :]     # SG coords lead, like
            pts.append(xc.reshape(-1, 3))          # the old cloud .vtk
            base = off + k * np.arange(Eb)[:, None]
            conn.append(np.hstack([np.full((Eb, 1), k, int),
                                   base + np.arange(k)]))
            ctyp.append(np.full(Eb, vt, int))
            e_val.append(np.hstack([
                strains[row:row + Eb * Q].reshape(Eb, Q, 6).mean(1),
                stresses[row:row + Eb * Q].reshape(Eb, Q, 6).mean(1)]))
            if mat_e is not None:
                e_mat.append(mat_e[bi].astype(float))
            off += Eb * k
            row += Eb * Q
        pts = np.vstack(pts)
        ev = np.vstack(e_val)
        E_tot = ev.shape[0]
        with open(prefix + ".vtk", "w") as f:
            f.write("# vtk DataFile Version 3.0\n"
                    "OpenSG dehom, exploded SG mesh, per-element Gauss"
                    " mean (%s)\nASCII\nDATASET UNSTRUCTURED_GRID\n"
                    % (frame or "frame unspecified"))
            f.write("POINTS %d float\n" % len(pts))
            np.savetxt(f, pts, fmt='%.8e')
            sz = sum(c.size for c in conn)
            f.write("\nCELLS %d %d\n" % (E_tot, sz))
            for c in conn:
                np.savetxt(f, c, fmt='%d')
            f.write("\nCELL_TYPES %d\n" % E_tot)
            np.savetxt(f, np.concatenate(ctyp), fmt='%d')
            f.write("\nCELL_DATA %d\n" % E_tot)
            for i, nm in enumerate(names):
                f.write("SCALARS Eps_%s float 1\nLOOKUP_TABLE default\n"
                        % nm)
                np.savetxt(f, ev[:, i], fmt='%.6e')
            for i, nm in enumerate(names):
                f.write("SCALARS Sig_%s float 1\nLOOKUP_TABLE default\n"
                        % nm)
                np.savetxt(f, ev[:, 6 + i], fmt='%.6e')
            if mat_e is not None:
                f.write("SCALARS mat_id float 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, np.concatenate(e_mat), fmt='%.1f')

    # the OpenSG dehom files: SG coords into the LAST n_sg of (x, y, z)
    xyz = np.zeros((num_pts, 3))
    xyz[:, 3 - d:] = coords
    model = {1: "beam", 2: "plate", 3: "solid"}.get(r["n_model"], "?")
    note = ("element Gauss points (%d), SwiftComp storage reordered to "
            "11 22 33 12 13 23 on write" % num_pts)
    if frame:
        # the frame tag makes the files self-describing: material-frame
        # and SG-global .SM are byte-identical in layout otherwise
        note += "; frame: %s" % frame
    # frame IN THE FILENAME for stress/strain: the DEFAULT (material
    # frame) keeps the bare .SM/.EM names; a --global run writes
    # _global.SM/_global.EM so a global-frame file can never be
    # mistaken for the usual material-frame output.  .U keeps its name
    # in both modes -- displacement is never rotated (always SG-global).
    ftag = ("_global" if frame and "global" in str(frame).lower()
            else "")
    _write_field(prefix + ftag + ".SM", "S", "Pa", stresses, xyz, note,
                 "%s %s dehom stress (units follow the SG input system)"
                 % (prefix, model))
    _write_field(prefix + ftag + ".EM", "E", "-", strains, xyz, note,
                 "%s %s dehom strain" % (prefix, model))
    wrote_u = ""
    if U_eqd is not None:
        _write_field(prefix + ".U", "U", "m", np.asarray(U_eqd), xyz,
                     note, "%s %s dehom FLUCTUATION-ONLY displacement "
                     "(no macro contribution at a unit-state SG "
                     "recovery; beam = w0 + w1s warping)"
                     % (prefix, model))
        wrote_u = " / .U"

    # ---- per-ELEMENT export: the Gauss mean at the element centroid.
    # This is the elemental quantity that pairs with a CENTROIDAL 3-D
    # FEA stress (one value per element, piecewise constant) -- pairing
    # Gauss points across two codes is not meaningful, element means
    # are.  Same _write_field layout, so downstream readers are shared.
    def _batches(F, w):
        out = []
        for b in (F if isinstance(F, (list, tuple)) else [F]):
            a = np.asarray(b)
            out.append(a.reshape(a.shape[0], -1, w))
        return out

    row = 0
    Sc, Uc, Xc = [], [], []
    ub = (_batches(U_eqd, 3) if U_eqd is not None else None)
    for bi, Sb in enumerate(_batches(Sigma_eqs, 6)):
        Eb, Qb = Sb.shape[:2]
        Sc.append(Sb.mean(axis=1))
        Xc.append(xyz[row:row + Eb * Qb].reshape(Eb, Qb, 3).mean(axis=1))
        if ub is not None:
            Uc.append(ub[bi].mean(axis=1))
        row += Eb * Qb
    note_e = (note + "; ELEMENTAL: unweighted mean of the element's Q"
              " Gauss values at the centroid (Q = quadrature count, 24"
              " for tet10 at degree 6 -- NOT 4)")
    _write_field(prefix + "_elemental" + ftag + ".SM", "S", "Pa",
                 np.vstack(Sc), np.vstack(Xc), note_e,
                 "%s %s dehom stress, per-element mean" % (prefix, model))
    if ub is not None:
        _write_field(prefix + "_elemental.U", "U", "m", np.vstack(Uc),
                     np.vstack(Xc), note_e,
                     "%s %s dehom displacement, per-element mean"
                     % (prefix, model))
    print("export_gauss: %d points -> %s.txt / .vtk / .SM / .EM%s"
          % (num_pts, prefix, wrote_u))
