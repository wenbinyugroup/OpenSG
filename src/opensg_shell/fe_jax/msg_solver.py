"""
Shared FEM infrastructure for the MSG thin-walled (TW) Timoshenko solve:
Gauss quadrature, element geometry, the KKT fluctuation solve, and the
Timoshenko V1 / 6x6 assembly.  Element-agnostic — used by the Hermite C1
pipeline (``msg_hermite``).  The KKT system is solved with pypardiso
(Intel MKL PARDISO).
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
import pypardiso

jax.config.update('jax_default_matmul_precision', 'highest')
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)


def gauss_legendre_01(n_pts):
    """Gauss-Legendre quadrature points and weights on [0, 1]."""
    xi_ref, w_ref = np.polynomial.legendre.leggauss(n_pts)
    return jnp.array(0.5 * (xi_ref + 1.0)), jnp.array(0.5 * w_ref)


def compute_element_geometry(nodes, cells):
    """Arc-length L and tangent (xdot2, xdot3) per element.

    Uses the two end-corner nodes (first and last columns), so it works for
    both 3-node quadratic cells and 2-node Hermite cells.
    """
    n1, n2 = cells[:, 0], cells[:, -1]
    dy2 = nodes[n2, 0] - nodes[n1, 0]
    dy3 = nodes[n2, 1] - nodes[n1, 1]
    L_e = np.sqrt(dy2**2 + dy3**2)
    return jnp.array(L_e), jnp.array(dy2 / L_e), jnp.array(dy3 / L_e)


# =============================================================================
# KKT Solver
# =============================================================================

def assemble_kkt(Dhh_sparse, Dc_matrix):
    """Build the augmented KKT matrix [Dhh  Dc; Dc^T  0] (constraints scaled by 1e8).

    Returns (A_aug CSR, N primal dofs, n_con, emp) where `emp` are empty rows that were
    given a unit diagonal (their RHS must be zeroed by the caller)."""
    if not hasattr(Dhh_sparse, 'row'):
        Dhh_sparse = Dhh_sparse.tocoo()
    dhh_data = np.asarray(Dhh_sparse.data)
    dhh_row  = np.asarray(Dhh_sparse.row)
    dhh_col  = np.asarray(Dhh_sparse.col)
    N = Dhh_sparse.shape[0]

    Dc = np.asarray(Dc_matrix)
    if Dc.shape[0] == 4 and Dc.shape[1] == N:
        Dc = Dc.T
    n_con = Dc.shape[1]
    Dc_sc = Dc * 1e8

    bl_r, bl_c = np.where(Dc_sc.T);   bl_r = bl_r + N
    tr_r, tr_c = np.where(Dc_sc);     tr_c = tr_c + N

    aug_row  = np.concatenate([dhh_row, tr_r, bl_r])
    aug_col  = np.concatenate([dhh_col, tr_c, bl_c])
    aug_data = np.concatenate([dhh_data, Dc_sc.T[bl_r-N, bl_c-N],
                                          Dc_sc[tr_r, tr_c-N]])
    total = N + n_con
    cnt = np.bincount(aug_row, minlength=total)
    emp = np.where(cnt == 0)[0]
    if len(emp):
        aug_row  = np.concatenate([aug_row, emp])
        aug_col  = np.concatenate([aug_col, emp])
        aug_data = np.concatenate([aug_data, np.ones(len(emp), dtype=aug_data.dtype)])
    A_aug = csr_matrix((aug_data, (aug_row, aug_col)), shape=(total, total))
    return A_aug, N, n_con, emp


def solve_fluctuation_field(Dhh_sparse, RHS_dense, Dc_matrix):
    """Solve KKT system [Dhh  Dc; Dc^T  0][V0; lam] = [RHS; 0].

    Returns
    -------
    V0          : (N_primal, n_cases) fluctuation field
    D1          : (n_cases, n_cases) = -(V0^T @ RHS)
    A_augmented : scipy CSR of the full KKT system (reused for Timoshenko RHS)
    """
    R = np.asarray(RHS_dense)
    N, n_cases = R.shape
    A_aug, N, n_con, emp = assemble_kkt(Dhh_sparse, Dc_matrix)
    R_aug = np.vstack([R, np.zeros((n_con, n_cases))])
    if len(emp):
        R_aug[emp] = 0.0
    V_aug = pypardiso.spsolve(A_aug, R_aug)

    V0 = V_aug[:N, :]
    D1 = -(V0.T @ R)
    return jnp.array(V0), jnp.array(D1), A_aug


# =============================================================================
# Timoshenko enhancement (V1 RHS + 6x6 assembly)
# =============================================================================

@jax.jit
def prepare_v1_rhs(V0, Dhl, Dll, Dle_dense, Psi, Dc):
    """Compute RHS for the V1 (shear-warping) solve, projected onto range(Dhh)."""
    DhlV0 = Dhl @ V0
    V0DllV0 = V0.T @ (Dll @ V0)
    DhlTV0Dle = Dhl.T @ V0 + Dle_dense
    b_unproj = DhlV0 - DhlTV0Dle

    tmp = jnp.linalg.inv(Psi.T @ Dc) @ (Psi.T @ b_unproj)
    bb = (Dc @ tmp) - b_unproj
    return bb, DhlV0, DhlTV0Dle, V0DllV0


@jax.jit
def finalize_v1_and_compute_deff(V1s_raw, V0, D_eff, V0DllV0,
                                  DhlV0, DhlTV0Dle, Psi, Dc):
    """Project V1s and assemble the sorted 6x6 Timoshenko stiffness.

    Output ordering: [EA, GA12, GA13, GJ, EI2, EI3].
    """
    tmp = jnp.linalg.inv(Dc.T @ Psi) @ (Dc.T @ V1s_raw)
    V1s = V1s_raw - (Psi @ tmp)

    # D_eff is a symmetric stiffness by construction (Hessian of the EB energy).  A
    # Dirichlet-constrained TAPER segment solve produces a small (2-6%) antisymmetric
    # ARTIFACT in D_eff (V0 is fixed at the boundary rings, so Dee+V0.T@Dhe is not the
    # exact symmetric Schur complement).  Ainv=inv(D_eff) would carry that asymmetry
    # into Q_tim -> the ANTISYMMETRIC shear-bending coupling (C26/C35), inflating it by
    # up to +78% on anisotropic tapers while leaving the symmetric terms untouched.
    # Enforcing the physical invariant here is a no-op for prismatic/symmetric D_eff.
    D_eff = 0.5 * (D_eff + D_eff.T)
    Ainv = jnp.linalg.inv(D_eff)
    B_tim = DhlTV0Dle.T @ V0

    C_unsym = V0DllV0 + V1s.T @ (DhlV0 + DhlTV0Dle)
    C_tim = 0.5 * (C_unsym + C_unsym.T)

    Q_base = jnp.array([[0., 0.], [0., 0.], [0., -1.], [1., 0.]])
    Q_tim = Ainv @ Q_base
    Ginv = Q_tim.T @ (C_tim - B_tim.T @ Ainv @ B_tim) @ Q_tim
    G_tim = jnp.linalg.inv(Ginv)
    Y_tim = B_tim.T @ Q_tim @ G_tim
    A_tim = D_eff + Y_tim @ Ginv @ Y_tim.T

    S = jnp.zeros((6, 6))
    S = S.at[0, 0].set(A_tim[0, 0])
    S = S.at[0, 1:3].set(Y_tim[0, :])
    S = S.at[0, 3:6].set(A_tim[0, 1:4])
    S = S.at[1:3, 0].set(Y_tim.T[:, 0])
    S = S.at[1:3, 1:3].set(G_tim)
    S = S.at[1:3, 3:6].set(Y_tim.T[:, 1:4])
    S = S.at[3:6, 0].set(A_tim[1:4, 0])
    S = S.at[3:6, 1:3].set(Y_tim[1:4, :])
    S = S.at[3:6, 3:6].set(A_tim[1:4, 1:4])
    return S, B_tim, C_tim, V1s
