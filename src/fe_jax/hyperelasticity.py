import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from .utils import rank2_tensor_to_voigt, rank2_voigt_to_tensor


@jax.jit
def st_venant_kirchhoff(F_qdd: jnp.ndarray, material_params_qm: jnp.ndarray):
    """
    A constitutive relation for an isotropic St. Venant-Kirchhoff hyperelastic solid. Identical to linear elasticity, except using nonlinear strain.
    First argument is deformation gradient, rather than strain, since other hyperelastic models are not naturally in terms of even nonlinear strain.
    """
    E = material_params_qm[..., 0]
    nu = material_params_qm[..., 1]
    G = 0.5 * E / (1.0 + nu)
    zero = jnp.zeros_like(nu)
    if F_qdd.shape[2] == 1:  # 1D
        C_qss = E.transpose((1, 0))[:, jnp.newaxis]
    elif F_qdd.shape[2] == 2:  # 2D
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, zero],
                    [-nu / E, 1.0 / E, zero],
                    [zero, zero, 1.0 / G],
                ]
            ).transpose((2, 0, 1))
        )
    elif F_qdd.shape[2] == 3:  # 3D
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, -nu / E, zero, zero, zero],
                    [-nu / E, 1.0 / E, -nu / E, zero, zero, zero],
                    [-nu / E, -nu / E, 1.0 / E, zero, zero, zero],
                    [zero, zero, zero, 1.0 / G, zero, zero],
                    [zero, zero, zero, zero, 1.0 / G, zero],
                    [zero, zero, zero, zero, zero, 1.0 / G],
                ]
            ).transpose((2, 0, 1))
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")
    eps_qdd = 0.5 * (
        jnp.einsum("qdi,qdj -> qij", F_qdd, F_qdd)
        - jnp.eye(F_qdd.shape[2])[jnp.newaxis, :, :]
    )
    stress_qdd = rank2_voigt_to_tensor(
        jnp.einsum("qsi,qi->qs", C_qss, rank2_tensor_to_voigt(eps_qdd))
    )
    return stress_qdd


@jax.jit
def mooney_rivlin(F_qdd: jnp.ndarray, material_params_qm: jnp.ndarray):
    """
    A constitutive model for Mooney-Rivlin hyperelasticity, assumes plane strain/linear strain for the lower dimensional cases,
    essentially assuming a block form of the deformation gradient with the relevant dimensions as provided by the argument F_qdd in the upper left,
    and 1s along the remaining diagonal.
    TODO:
    Since we only want the upper left block of the stress as well, we can essentially work only on that block,
    which means that the computation is independent of dimension. This may not be valid, perhaps confirm how 2D hyperelasticity is "usually" done
    """
    C1_q = material_params_qm[..., 0]
    C2_q = material_params_qm[..., 1]
    D1_q = material_params_qm[..., 2]

    if F_qdd.shape[2] <= 3:
        J_q = jnp.linalg.det(F_qdd)
        Jm13_q = J_q ** (-1 / 3)
        Fbar_qdd = jnp.einsum("q,qdj -> qdj", Jm13_q, F_qdd)
        Bbar_qdd = jnp.einsum("qdj,qdk -> qjk", Fbar_qdd, Fbar_qdd)
        I1bar_q = jnp.einsum("qdd->q", Bbar_qdd)
        I2bar_q = 0.5 * (I1bar_q**2 - jnp.einsum("qij,qji -> q", Bbar_qdd, Bbar_qdd))
        BbarFbar_qdd = jnp.einsum("qjd,qdk -> qjk", Bbar_qdd, Fbar_qdd)
        Fbarinv_qdd = jnp.linalg.inv(Fbar_qdd).transpose((0, 2, 1))
        FinvTcoefficients_q = 2 / D1_q * Jm13_q ** (-2) * (J_q - 1) - 2 / 3 * Jm13_q * (
            C1_q * I1bar_q + 2 * C2_q * I2bar_q
        )
        Fcoefficients_q = 2 * Jm13_q * (C1_q + I1bar_q * C2_q)
        BFcoefficients_q = 2 * Jm13_q * C2_q
        stress_qdd = (
            jnp.einsum("q,qdj -> qjd", FinvTcoefficients_q, Fbarinv_qdd)
            + jnp.einsum("q,qdj->qdj", Fcoefficients_q, Fbar_qdd)
            + jnp.einsum("q,qdj->qdj", BFcoefficients_q, BbarFbar_qdd)
        )
    else:
        raise RuntimeError("Deformation Gradient must be at most 3D")
    return stress_qdd

@jax.jit
def hyperelasticity_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_qm: jnp.ndarray,
    constitutive_model: Callable,
):
    """
    Residual function that computes the residual for the weak form corresponding to hyperelasticity


    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    dphi_dxi_qnp  : derivative of basis functions in parametric coordinate system at
                    quadrature points, ndarray[float, (Q, N, P)]
    W_q           : quadrature weights, ndarray[float, (Q,)]
    mat_params_qm : material parameters, ndarray[float, (Q, M)]
    constitutive_relation : constitutive stress-strain relation, arguments
                  (F_qdd: jnp.ndarray, material_params_qm: jnp.ndarray) where F_qdd is the deformation gradient, dx/dX

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """

    D = u_nd.shape[1]
    P = dphi_dxi_qnp.shape[2]
    assert P == D
    # Formulation assumes solid elements otherwise a different approach is needed (i.e. shells)

    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    F_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd + x_nd)
    stress_qdd = constitutive_model(F_qdd=F_qdd, material_params_qm=material_params_qm)

    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd)
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)

    return R_nd
