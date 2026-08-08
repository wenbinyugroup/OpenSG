import numpy as np

from basix import (
    CellType,
    ElementFamily,
    LagrangeVariant,
    QuadratureType,
    PolysetType,
    create_element,
    make_quadrature,
)

from dataclasses import dataclass
from typing import Any


@dataclass
class FiniteElementType:
    """
    Defines the properties of a finite element formulation used to compute the basis
    functions and quadratures, see eval_basis_and_derivatives() and get_quadrature().
    """
    cell_type: CellType
    family: ElementFamily
    basis_degree: int
    lagrange_variant: LagrangeVariant
    quadrature_type: QuadratureType
    quadrature_degree: int


def eval_basis_and_derivatives(
    fe_type: FiniteElementType, xi_qp: np.ndarray[Any, np.dtype[np.float64]]
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """
    Evaluates basis functions and its first derivatives at the given parametric points.
    Uses basis functions from the basix library, see https://github.com/FEniCS/basix for numbering
    documentation.

    Parameters
    ----------
    fe_type : defines the type of finite element for which to evaluate the basis functions
    xi_qp   : dense 2d-array with shape (Q, P)

    Returns
    -------
    phi      : 2d-array with shape (Q, N)
    dphi_dxi : 3d-array with shape (Q, N, P)
    """

    e = create_element(
        family=fe_type.family,
        celltype=fe_type.cell_type,
        degree=fe_type.basis_degree,
        lagrange_variant=fe_type.lagrange_variant,
    )
    r = np.array(e.tabulate(n=1, x=xi_qp))  #e.tabulate(degree, points): (derivatives, points, basis functions, value dimension)

    return (r[0, :, :, 0], r[1:, :, :, 0].transpose((1, 2, 0))) # (phi_qn, dphi_dxi_qnp)


def get_quadrature(
    fe_type: FiniteElementType,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """
    Get quadrature scheme (points and weights) for a given finite element type. Uses quadrature
    from the basix library. For documentation, refer to:
    https://docs.fenicsproject.org/basix/main/python/demo/demo_quadrature.py.html.

    Parameters
    ----------
    fe_type : defines the type of finite element the quadrature is for

    Returns
    -------
    xi : 2d-array with shape (Q, P)
    W  : 1d-array with shape (Q,)
    """
    quad_points, weights = make_quadrature(
        cell=fe_type.cell_type,
        degree=fe_type.quadrature_degree,
        rule=fe_type.quadrature_type,
    )
    return (np.array(quad_points), np.array(weights))
