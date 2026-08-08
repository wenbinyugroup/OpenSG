import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable

from .utils import rank2_tensor_to_voigt, rank2_voigt_to_tensor, is_required


@jax.jit
def elastic_isotropic(eps_qdd: jnp.ndarray, material_params_qm: jnp.ndarray):
    """
    A constitive relation for a linear elastic isotropic material.

    Parameters
    ----------
    eps_qdd       : infinitesimal strain tensor, ndarray[float, (Q, D, D)]
    mat_params_qm : material parameters, ndarray[float, (Q, M)]

    Returns
    -------
    stress_qdd  : stress tensor, ndarray[float, (Q, D, D)]
    """

    E = material_params_qm[..., 0]
    nu = material_params_qm[..., 1]
    G = 0.5 * E / (1.0 + nu)
    zero = jnp.zeros_like(nu)
    if eps_qdd.shape[2] == 1:  # 1D
        C_qss = E.transpose((1, 0))[:, jnp.newaxis]
    elif eps_qdd.shape[2] == 2:  # 2D
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E, -nu / E, zero],
                    [-nu / E, 1.0 / E, zero],
                    [zero, zero, 1.0 / G],
                ]
            ).transpose((2, 0, 1))
        )
    elif eps_qdd.shape[2] == 3:  # 3D
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
   # unit_strain=jnp.array([1, 0, 0, 0, 0, 0]) 
   # unit_strain=jnp.eye(N)
   

    stress_qdd = rank2_voigt_to_tensor(
      # jnp.einsum("qsi,qi->qs", C_qss, unit_strain + rank2_tensor_to_voigt(eps_qdd))
      jnp.einsum("qsi,qi->qs", C_qss, rank2_tensor_to_voigt(eps_qdd))
    )
    
    return stress_qdd, jnp.zeros(shape=(material_params_qm.shape[0], 0)) # no internal state


@jax.jit
def elastic_orthotropic(
    eps_qdd: jnp.ndarray, material_params_qm: jnp.ndarray
):
    """
    A constitive relation for a linear elastic isotropic material.

    Parameters
    ----------
    eps_qdd       : infinitesimal strain tensor, ndarray[float, (Q, D, D)]
    mat_params_qm : material parameters, ndarray[float, (Q, M)]

    Returns
    -------
    stress_qdd  : stress tensor, ndarray[float, (Q, D, D)]
    """
    zero = jnp.zeros(shape=(material_params_qm.shape[0:1]))
    if eps_qdd.shape[2] == 1:  # 1D
        assert (
            material_params_qm.shape[-1] == 1
        ), f"Orthotropic elasticity in 1D requires 1 material parameter, received {material_params_qm.shape[-1]}"

        E = material_params_qm[..., 0]
        C_qss = E.transpose((1, 0))[:, jnp.newaxis]

    elif eps_qdd.shape[2] == 2:  # 2D
        assert (
            material_params_qm.shape[-1] == 4
        ), f"Orthotropic elasticity in 2D requires 4 material parameters, received {material_params_qm.shape[-1]}"

        E_xx = material_params_qm[..., 0]
        E_yy = material_params_qm[..., 1]
        nu_xy = material_params_qm[..., 2]
        G_xy = material_params_qm[..., 3]

        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E_xx, -nu_xy / E_xx, zero],
                    [-nu_xy / E_xx, 1.0 / E_yy, zero],
                    [zero, zero, 1.0 / G_xy],
                ]
            ).transpose((2, 0, 1))
        )

    elif eps_qdd.shape[2] == 3:  # 3D
        assert (
            material_params_qm.shape[-1] == 9
        ), f"Orthotropic elasticity in 3D requires 9 material parameters, received {material_params_qm.shape[-1]}"

        E_xx = material_params_qm[..., 0]
        E_yy = material_params_qm[..., 1]
        E_zz = material_params_qm[..., 2]
        nu_xy = material_params_qm[..., 3]
        nu_yz = material_params_qm[..., 4]
        nu_xz = material_params_qm[..., 5]
        G_xy = material_params_qm[..., 6]
        G_yz = material_params_qm[..., 7]
        G_xz = material_params_qm[..., 8]

        # Note: inv could be avoided if it is a bottleneck, see:
        # https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/hooke_orthotropic.cfm
        C_qss = jnp.linalg.inv(
            jnp.array(
                [
                    [1.0 / E_xx, -nu_xy / E_xx, -nu_xz / E_xx, zero, zero, zero],
                    [-nu_xy / E_xx, 1.0 / E_yy, -nu_yz / E_yy, zero, zero, zero],
                    [-nu_xz / E_xx, -nu_yz / E_yy, 1.0 / E_zz, zero, zero, zero],
                    [zero, zero, zero, 1.0 / G_yz, zero, zero],
                    [zero, zero, zero, zero, 1.0 / G_xz, zero],
                    [zero, zero, zero, zero, zero, 1.0 / G_xy],
                ]
            ).transpose((2, 0, 1))
        )
    else:
        raise RuntimeError("Strain must be 1D, 2D or 3D to compute stress.")

    stress_qdd = rank2_voigt_to_tensor(
        jnp.einsum("qsi,qi->qs", C_qss, rank2_tensor_to_voigt(eps_qdd))
    )
    return stress_qdd, jnp.zeros(shape=(material_params_qm.shape[0], 0)) # no internal state


@jax.jit
def linear_elasticity_residual(
    u_nd: jnp.ndarray,
    x_nd: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    W_q: jnp.ndarray,
    material_params_qm: jnp.ndarray,
    internal_state_qi: jnp.ndarray,
    constitutive_model: Callable,
):
    """
    Residual function that computes the residual for the weak form corresponding to linear
    elasticity.

    Parameters
    ----------
    u_nd          : solution vector, ndarray[float, (N, D)]
    x_nd          : coordinates, ndarray[float, (N, D)]
    dphi_dxi_qnp  : derivative of basis functions in parametric coordinate system at
                    quadrature points, ndarray[float, (Q, N, P)]
    W_q           : quadrature weights, ndarray[float, (Q,)]
    mat_params_qm : material parameters, ndarray[float, (Q, M)]
    constitutive_relation : constitutive stress-strain relation, function with arguments
                  (eps_qdd: jnp.ndarray, material_params_qm: jnp.ndarray)

    Returns
    -------
    R_nd  : residual vector, ndarray[float, (N, D)]
    """
    
    D = u_nd.shape[1]
    P = dphi_dxi_qnp.shape[2]
    
    # assert P == D # P is not necessarily equal to D for MSG
    
    # Formulation assumes solid elements otherwise a different approach is needed (i.e. shells)

    J_qpd = jnp.einsum("nd,qnp->qpd", x_nd, dphi_dxi_qnp)

    G_qpd = jnp.linalg.inv(J_qpd).transpose(0, 2, 1)
    det_J_q = jnp.linalg.det(J_qpd)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)

    du_dx_qdd = jnp.einsum("qnd,ni->qid", dphi_dx_qnd, u_nd)

    #eps_qdd = 0.5 * (du_dx_qdd + du_dx_qdd.transpose((0, 2, 1)))
    du_dx_qdd_padded = jnp.pad(du_dx_qdd, ((0,0), (0,0), (0,1)))
    eps_qdd = 0.5 * (du_dx_qdd_padded + du_dx_qdd_padded.transpose((0, 2, 1))) # for MSG, we need a 3x3 strain tensor
    
    constitutive_args = {}

    if is_required(constitutive_model, "eps_qdd"):
        constitutive_args["eps_qdd"] = eps_qdd

    if is_required(constitutive_model, "material_params_qm"):
        constitutive_args["material_params_qm"] = material_params_qm

    if is_required(constitutive_model, "internal_state_qi"):
        constitutive_args["internal_state_qi"] = internal_state_qi

    stress_qdd, new_internal_state_qi = constitutive_model(**constitutive_args)
    
    grad_dphi_dx_stress_qnd = jnp.einsum("qni,qid->qnd", dphi_dx_qnd, stress_qdd[:,0:2,:])
    
    det_JxW_q = jnp.einsum("q,q->q", det_J_q, W_q)
    R_nd = jnp.einsum("qnd,q->nd", grad_dphi_dx_stress_qnd, det_JxW_q)
    
    return R_nd, new_internal_state_qi
