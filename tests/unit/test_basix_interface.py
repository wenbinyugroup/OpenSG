from helper import *

# Baseline taken from: https://people.math.sc.edu/Burkardt/classes/cg_2007/cg_lab_fem_basis_triangle.pdf

def triangle_basis_p1(xi_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates basis functions at a parametric points (xi, eta).

    Parameters
    ----------
    xi_qp  : dense 2d-array with shape (N_qp, N_xi)

    Returns
    -------
    phi_qp : 2d-array with shape (N_qp, N_n)
    """
    return jnp.array([xi_qp[:, 0], xi_qp[:, 1], 1.0 - xi_qp[:, 0] - xi_qp[:, 1]]).T


def triangle_basis_p1_d_xi(xi_qp: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluates the derivatives of the basis functions with respect to the parametric
    coordinates at given parametric points (xi, eta).

    Parameters
    ----------
    xi_qp  : dense 2d-array with shape (N_qp, N_xi)

    Returns
    -------
    dphi_dxi_qp : 2d-array with shape (N_qp, N_n, N_xi)
    """
    return jnp.array([[[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]] * xi_qp.shape[0])


def get_triangle_gauss_quadrature_1():
    """
    Returns the array describing the 1-point Gauss quadrature rule for a triangle.

    Returns
    -------
    2d-array (N_qp, N_xi+1)
    ```
       [ [xi_0, eta_0, weight_0] ]
    ```
    """
    return jnp.array([[1.0 / 3.0, 1.0 / 3.0, 0.5]])


def test_triangle_p1_basis_and_quadrature():
    fe_type = FiniteElementType(
        cell_type=CellType.triangle,
        family=ElementFamily.P,
        basis_degree=1,
        lagrange_variant=LagrangeVariant.equispaced,
        quadrature_type=QuadratureType.Default,
        quadrature_degree=3,
    )

    # Expected sizes
    Q = 6
    N = 3
    P = 2

    xi, W = get_quadrature(fe_type=fe_type)
    assert xi.shape == (Q, P)
    assert W.shape == (Q,)
    assert np.isclose(np.sum(W), 0.5)

    phi, dphi_dxi = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi)
    assert phi.shape == (Q, N)
    assert dphi_dxi.shape == (Q, N, P)

    # Note: the ordering is different between:
    # - triangle_basis_p1: https://people.math.sc.edu/Burkardt/classes/cg_2007/cg_lab_fem_basis_triangle.pdf
    # - basix: https://defelement.org/elements/examples/triangle-lagrange-equispaced-1.html
    # So, the index permutes them to match
    phi_test = triangle_basis_p1(xi_qp=xi)[:, [2, 0, 1]]
    assert np.isclose(phi, phi_test).all()
    dphi_dxi_test = triangle_basis_p1_d_xi(xi_qp=xi)[:, [2, 0, 1], :]
    assert np.isclose(dphi_dxi, dphi_dxi_test).all()
