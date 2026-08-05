"""sg_dehom.py -- dehomogenization of the general SG engine, split out
of rm_plate_2D.py: the SSDM Gauss-point recovery kernel (n_model 2/3),
the Beam_solid.py Timoshenko recovery chain (n_model 1), the public
plate_dehom_2d, and the Gauss exports.

PROVENANCE.  _element_dehomo_kernel is VERBATIM from the SSDM plate
model; build_recovery_matrix / compute_fluctuations_gpu /
get_Rsig_matrix / recover_gauss_point_fields are the Beam_solid.py
lifts (recover_gauss_point_fields additionally returns the recovered
strain -- the plate_dehom_2d/export_gauss contract).  The batched SSDM
recovery is wrapped in a module-level jit so repeated dehom calls in
one process (and, via the persistent cache, across processes) do not
re-trace.

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
from typing import Any, Dict, Sequence

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


@partial(jax.jit, static_argnames=["n_model", "n_sg"])
def _dehom_batch(periodic_cells_en, V0, x_end, C_ess, dphi_dxi_qnp,
                 phi_qn, epsilon_bar, n_model, n_sg):
    """All-element SSDM recovery, jitted once per (shape, n_model, n_sg);
    epsilon_bar is traced, so new load cases reuse the compiled kernel."""
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
    """Successive-derivative generalized-Timoshenko recovery ladder,
    VERBATIM from Beam_solid.py (the WORKING reference -- its core
    computation is kept untouched by decision).

    st is the sectional STRAIN 6-vector [eps11, 2g12, 2g13, k1, k2, k3]:
    the zeroth-order warping and Gamma_eps use st_m = [eps11, k1, k2,
    k3] directly, and the recovered stress integrates back to
    Deff @ st on the classical channels (RHC closures: ext 7e-4,
    twist 3e-7, bend2 9e-10).

    USAGE (what the as-lifted chain supports): drive the recovery with
    states whose stress rides the CLASSICAL channels (extension, twist,
    bending, and their combinations).  The transverse-shear entries
    st[1], st[2] are recovery-inert here: the ladder is the VABS
    refined-theory product-rule chain (F' = R(eps+e1) F,
    F'' = R F' + R' F, ..., eps^(n) = Comp @ F^(n), with
    build_recovery_matrix the intrinsic-equilibrium operator), but it
    is SEEDED with F_1d = Comp @ st -- a compliance-scaled vector
    (~1e-9 of Deff @ st), so the derivative states that would carry the
    shear stress (the Q = dM/dx bending gradient) are negligible and a
    pure-shear st recovers ~zero stress (measured).  The refined-theory
    seed would be the force state Deff @ st; that deviation is
    DOCUMENTED, not applied."""
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
    """Gauss-point strain/stress recovery of the beam chain
    (Beam_solid.py; lifted with the strain added to the returns).
    C_ess1 = the per-ELEMENT pre-elem-rotation stiffness -- stress comes
    out in the element frame the strain was rotated into (identity
    elem_rotation -> the global SG frame, matching the plate dehom)."""
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
                 epsilon_bar: Sequence[float]):
    """One recovery pass: the Gauss strain/stress AND the fluctuation
    displacement (nothing recomputed -- plate_dehom_2d is a view of the
    same pass).

    In:  r (plate_homo_2d dict); epsilon_bar as plate_dehom_2d
    Out: (Gamma_eqs, Sigma_eqs, U_eqd) -- (E, Q, 6), (E, Q, 6),
         (E, Q, 3) np.  U is the FLUCTUATION-ONLY displacement at the
         Gauss points (no macro contribution exists for a unit-state SG
         recovery -- the single-station rm_dehom precedent): n_model
         2/3 = V0 @ epsilon_bar; n_model 1 = the recovery chain's
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

    Gam, Sig = _dehom_batch(r["periodic_cells_en"], r["V0"], r["x_end"],
                            r["C_ess"], r["dphi_dxi_qnp"], r["phi_qn"],
                            jnp.asarray(epsilon_bar, float),
                            r["n_model"], r["n_sg"])
    U = _u_at_gauss(jnp.asarray(r["V0"])
                    @ jnp.asarray(epsilon_bar, float),
                    r["periodic_cells_en"], r["phi_qn"])
    return np.asarray(Gam), np.asarray(Sig), U


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
    """(E*Q, n_sg) physical coordinates of every recovery point."""
    g = jnp.einsum('qn, end -> eqd', r["phi_qn"], r["x_end"])
    return np.asarray(g).reshape(-1, r["n_sg"])


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


def export_gauss(r, Gamma_eqs, Sigma_eqs, prefix="gauss_results",
                 U_eqd=None):
    """The .txt table + point-cloud .vtk of the recovered Gauss fields
    (verbatim export of the SSDM script, parameterized), PLUS the
    OpenSG dehom files <prefix>.SM (stress) / .EM (strain) and -- when
    U_eqd from dehom_fields is given -- .U (fluctuation displacement).
    File component order is the rm_plate_1D convention (11 22 33 12 13
    23; units follow the SG input system); the (x, y, z) columns carry
    the SG coordinates in the LAST n_sg slots (y3 = thickness/last SG
    coordinate -> the z column), leading slots zero."""
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

    coords3 = np.hstack((coords, np.zeros((num_pts, 3 - d)))) \
        if d < 3 else coords
    names = ["xx", "yy", "zz", "yz", "xz", "xy"]
    with open(prefix + ".vtk", "w") as f:
        f.write("# vtk DataFile Version 3.0\nGauss Point Voigt Data\n"
                "ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS %d float\n" % num_pts)
        np.savetxt(f, coords3, fmt='%.6e')
        f.write("\nCELLS %d %d\n" % (num_pts, num_pts * 2))
        np.savetxt(f, np.column_stack((np.ones(num_pts, dtype=int),
                                       np.arange(num_pts, dtype=int))),
                   fmt='%d')
        f.write("\nCELL_TYPES %d\n" % num_pts)
        np.savetxt(f, np.ones(num_pts, dtype=int), fmt='%d')
        f.write("\nPOINT_DATA %d\n" % num_pts)
        for i, nm in enumerate(names):
            f.write("SCALARS Eps_%s float 1\nLOOKUP_TABLE default\n" % nm)
            np.savetxt(f, strains[:, i], fmt='%.6e')
        for i, nm in enumerate(names):
            f.write("SCALARS Sig_%s float 1\nLOOKUP_TABLE default\n" % nm)
            np.savetxt(f, stresses[:, i], fmt='%.6e')

    # the OpenSG dehom files: SG coords into the LAST n_sg of (x, y, z)
    xyz = np.zeros((num_pts, 3))
    xyz[:, 3 - d:] = coords
    model = {1: "beam", 2: "plate", 3: "solid"}.get(r["n_model"], "?")
    note = ("element Gauss points (%d), SwiftComp storage reordered to "
            "11 22 33 12 13 23 on write" % num_pts)
    _write_field(prefix + ".SM", "S", "Pa", stresses, xyz, note,
                 "%s %s dehom stress (units follow the SG input system)"
                 % (prefix, model))
    _write_field(prefix + ".EM", "E", "-", strains, xyz, note,
                 "%s %s dehom strain" % (prefix, model))
    wrote_u = ""
    if U_eqd is not None:
        _write_field(prefix + ".U", "U", "m", np.asarray(U_eqd), xyz,
                     note, "%s %s dehom FLUCTUATION-ONLY displacement "
                     "(no macro contribution at a unit-state SG "
                     "recovery; beam = w0 + w1s warping)"
                     % (prefix, model))
        wrote_u = " / .U"
    print("export_gauss: %d points -> %s.txt / .vtk / .SM / .EM%s"
          % (num_pts, prefix, wrote_u))
