from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from typing import Union
from dolfinx.fem import form, petsc, Function, locate_dofs_topological
from ufl import (
    TrialFunction,
    TestFunction,
    rhs,
    as_tensor,
    dot,
    SpatialCoordinate,
    Measure,
    grad,
    as_vector,
    Jacobian,
    sqrt,
    cross,
)

from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix

from mpi4py import MPI
import ufl
from contextlib import ExitStack

from opensg.utils.shared import local_frame_1D


def generate_boundary_markers(xmin, xmax):
    """Generate functions to mark left and right boundaries of a mesh.

    Parameters
    ----------
    xmin : float
        Minimum x-coordinate of the mesh
    xmax : float
        Maximum x-coordinate of the mesh

    Returns
    -------
    tuple
        Two functions that return True for points on the left and right
        boundaries respectively
    """

    def is_left_boundary(x):
        return np.isclose(x[0], xmin)

    def is_right_boundary(x):
        return np.isclose(x[0], xmax)

    return is_left_boundary, is_right_boundary


def eps(vector):
    """Compute strain tensor and strain vector from displacement gradient.

    Parameters
    ----------
    vector : ufl.Coefficient
        Displacement vector field

    Returns
    -------
    tuple
        (strain_tensor, strain_vector) where:
        - strain_tensor is a 3x3 tensor
        - strain_vector is a 6-component vector [ε11, ε22, ε33, γ23, γ13, γ12]
    """
    E1 = ufl.as_vector([0, 0, vector[2].dx(0), (vector[1].dx(0)), (vector[0].dx(0)), 0])
    ans = as_tensor(
        [
            (E1[0], 0.5 * E1[5], 0.5 * E1[4]),
            (0.5 * E1[5], E1[1], 0.5 * E1[3]),
            (0.5 * E1[4], 0.5 * E1[3], E1[2]),
        ]
    )
    return ans, E1


### local reference frames
def local_frame(mesh):
    """Compute local orthonormal frame for shell elements.

    This function computes an orthonormal frame (e1, e2, e3) at each point
    of the mesh, where:
    - e1 is the first tangent vector
    - e2 is the second tangent vector
    - e3 is the normal vector

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        Mesh object

    Returns
    -------
    tuple
        Three ufl.Vector objects (e1, e2, e3) forming the local frame
    """
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])  # tangential direction
    t2 = as_vector([t[0, 1], t[1, 1], t[2, 1]])
    e3 = -cross(
        t1, t2
    )  # outward  normal direction (default)  # default mean when e3 = cross(t1, t2)
    e3 /= sqrt(dot(e3, e3))
    e1 = t2 / sqrt(dot(t2, t2))  # 1- direction axis
    e2 = cross(e3, e1)
    e2 /= sqrt(dot(e2, e2))  # 2- direction  -circumferential (default not rh rule)
    return e1, e2, e3


def deri(e):
    """Compute derivatives and curvatures of a local frame.

    This function computes the derivatives of the local frame vectors and
    the associated curvatures of the shell element.

    Parameters
    ----------
    e : tuple
        Tuple of three ufl.Vector objects (e1, e2, e3) forming the local frame

    Returns
    -------
    tuple
        Six scalar values representing the curvatures:
        (k11, k12, k21, k22, k13, k23)
    """
    e1, e2, e3 = e[0], e[1], e[2]
    a1_1 = dot(e1, grad(e1))
    a1_2 = dot(e2, grad(e1))  # directional derivative of e2 along e1 direction.
    a2_1 = dot(e1, grad(e2))
    a2_2 = dot(e2, grad(e2))
    a3_1 = dot(e1, grad(e3))
    a3_2 = dot(e2, grad(e3))

    # Initial Curvatures (Shell Element)
    k11 = dot(a3_1, e1)
    k12 = dot(a3_1, e2)
    k21 = dot(a3_2, e1)
    k22 = dot(a3_2, e2)
    k13 = dot(a1_1, e2)
    k23 = dot(a1_2, e2)
    return k11, k12, k21, k22, k13, k23


def local_grad(ee, q):
    """Compute directional derivative along a vector.

    Parameters
    ----------
    ee : ufl.Vector
        Direction vector
    q : ufl.Coefficient
        Scalar or vector field

    Returns
    -------
    ufl.Coefficient
        Directional derivative of q along ee
    """
    return dot(ee, grad(q))


def ddot(w, d1):
    """Compute dot product of two 3D vectors.

    Parameters
    ----------
    w : ufl.Vector
        First vector
    d1 : ufl.Vector
        Second vector

    Returns
    -------
    ufl.Coefficient
        Scalar product w·d1
    """
    return d1[0] * w[0] + d1[1] * w[1] + d1[2] * w[2]


############################################################################
# We need four matrices for our energy form/weak form
# EB requires only (Gamma_h and Gamma_e) matrix
# Timo needs (Gamma_l and Gamma_d) matrix. Note in paper, Gamma_l mentioend, but that does not contribute to Timoshenko like model.
# For input, local frame (e), spatial coordinate x and dx is required as Gamma_h for left boun/right boun/main qaud mesh would be different.
# Gamma_h*w column matrix
def gamma_h(e, x, w):
    """Compute the gamma_h operator for MSG-Shell formulations.

    This function computes the gamma_h operator. It represents the strain
    measures in terms of the displacement field.

    Parameters
    ----------
    e : tuple
        Local frame vectors (e1, e2, e3)
    x : ufl.SpatialCoordinate
        Spatial coordinates
    w : ufl.Coefficient
        Test or trial function

    Returns
    -------
    ufl.Tensor
        6-component vector representing the strain measures
    """
    k11, k12, k21, k22, k13, k23 = deri(e)  # extracting initial curvatures
    x11, x21, x31 = (
        local_grad(e[0], x[0]),
        local_grad(e[0], x[1]),
        local_grad(e[0], x[2]),
    )
    x12, x22, x32 = (
        local_grad(e[1], x[0]),
        local_grad(e[1], x[1]),
        local_grad(e[1], x[2]),
    )
    y1, y2, y3 = local_grad(e[2], x[0]), local_grad(e[2], x[1]), local_grad(e[2], x[2])
    d1 = [x11, x21, x31]
    d2 = [x12, x22, x32]
    d3 = [y1, y2, y3]

    d11 = as_vector([-k11 * d3[ii] + k13 * d2[ii] for ii in range(3)])
    d22 = as_vector([-k22 * d3[ii] - k23 * d1[ii] for ii in range(3)])
    d12 = as_vector([-k21 * d3[ii] + k23 * d2[ii] for ii in range(3)])
    d21 = as_vector([-k12 * d3[ii] - k13 * d1[ii] for ii in range(3)])

    w_d1 = [local_grad(e[0], w[i]) for i in range(3)]
    w_d2 = [local_grad(e[1], w[i]) for i in range(3)]
    w_d11 = [local_grad(e[0], w_d1[i]) for i in range(3)]
    w_d22 = [local_grad(e[1], w_d2[i]) for i in range(3)]

    w_d12 = [local_grad(e[1], w_d1[ii]) for ii in range(3)]
    w_d21 = [local_grad(e[0], w_d2[ii]) for ii in range(3)]
    w_11 = [local_grad(d11, w[ii]) for ii in range(3)]
    w_22 = [local_grad(d22, w[ii]) for ii in range(3)]
    w_12 = [local_grad(d12, w[ii]) for ii in range(3)]
    w_21 = [local_grad(d21, w[ii]) for ii in range(3)]

    G1 = ddot(w_d1, d1)
    G2 = ddot(w_d2, d2)
    G3 = ddot(w_d1, d2) + ddot(w_d2, d1)
    G4 = (
        -k11 * G1
        - k12 * 0.5 * G3
        - ddot(w_d11, d3)
        + k13 * ddot(w_d2, d3)
        - ddot(w_11, d3)
    )
    G5 = (
        -k22 * G2
        - k21 * 0.5 * G3
        - ddot(w_d22, d3)
        - k23 * ddot(w_d1, d3)
        - ddot(w_22, d3)
    )
    G6 = (
        -(k11 + k22) * 0.5 * G3
        - k12 * G2
        - k21 * G1
        + k23 * ddot(w_d2, d3)
        - k13 * ddot(w_d1, d3)
        - ddot(w_d12, d3)
        - ddot(w_d21, d3)
        - ddot(w_12, d3)
        - ddot(w_21, d3)
    )

    E1 = as_tensor([G1, G2, G3, G4, G5, G6])
    return E1


def gamma_l(e, x, w):
    """Compute the gamma_l operator for MSG-Shell formulations.

    This function computes the gamma_l operator. It represents the strain
    measures in terms of the displacement field for the Timoshenko model.

    Parameters
    ----------
    e : tuple
        Local frame vectors (e1, e2, e3)
    x : ufl.SpatialCoordinate
        Spatial coordinates
    w : ufl.Coefficient
        Test or trial function

    Returns
    -------
    ufl.Tensor
        6-component vector representing the strain measures
    """
    y1, y2, y3 = (
        x[2],
        x[0],
        x[1],
    )  # In MSG-Shell formulations, y1 should be the beam axis & (y2,y3) as cross-sectional coordinates)
    # In mesh data, z coordinates are
    k11, k12, k21, k22, k13, k23 = deri(e)
    x11, x21, x31 = (
        local_grad(e[0], x[0]),
        local_grad(e[0], x[1]),
        local_grad(e[0], x[2]),
    )
    x12, x22, x32 = (
        local_grad(e[1], x[0]),
        local_grad(e[1], x[1]),
        local_grad(e[1], x[2]),
    )
    y1, y2, y3 = local_grad(e[2], x[0]), local_grad(e[2], x[1]), local_grad(e[2], x[2])
    d1 = [x11, x21, x31]
    d2 = [x12, x22, x32]
    d3 = [y1, y2, y3]
    dd1, dd2, dd3 = as_vector(d1), as_vector(d2), as_vector(d3)
    # wd=[w[i].dx(0) for i in range(3)] # dx(0) is for w'

    # Gamma_l*w' column matrix     (w' defined by same function space as test/trail function
    #                                are basis functions which are same for both w and w')
    w_d1 = as_vector([local_grad(dd1, w[ii]) for ii in range(3)])
    w_d2 = as_vector([local_grad(dd2, w[ii]) for ii in range(3)])

    L1, L2 = x11 * (ddot(d1, w)), x12 * (ddot(d2, w))
    L3 = ddot(x11 * dd2 + x12 * dd1, w)
    L4 = -2 * x11 * ddot(w_d1, d3) + ddot(
        k11 * (y1 * dd3 - x11 * dd1) - 0.5 * k12 * (x12 * dd1 + x11 * dd2), w
    )
    L5 = -2 * x12 * ddot(w_d2, d3) + ddot(
        k22 * (y1 * dd3 - x12 * dd2) - 0.5 * k21 * (x12 * dd1 + x11 * dd2), w
    )
    L6 = -2 * ddot(x11 * w_d2 + x12 * w_d1, d3) + ddot(
        k12 * (y1 * dd3 - x12 * dd2)
        + k21 * (y1 * dd3 - x11 * dd1)
        - 0.5 * (k11 + k22) * (x12 * dd1 + x11 * dd2),
        w,
    )
    return as_tensor([L1, L2, L3, L4, L5, L6])


def gamma_e(e, x):
    """Compute the gamma_e operator for MSG-Shell formulations.

    This function computes the gamma_e operator.  It represents the strain
    measures in terms of the spatial coordinates.

    Parameters
    ----------
    e : tuple
        Local frame vectors (e1, e2, e3)
    x : ufl.SpatialCoordinate
        Spatial coordinates

    Returns
    -------
    ufl.Tensor
        6x4 matrix representing the strain measures
    """
    k11, k12, k21, k22, k13, k23 = deri(e)
    x11, x21, x31 = (
        local_grad(e[0], x[0]),
        local_grad(e[0], x[1]),
        local_grad(e[0], x[2]),
    )
    x12, x22, x32 = (
        local_grad(e[1], x[0]),
        local_grad(e[1], x[1]),
        local_grad(e[1], x[2]),
    )
    y1, y2, y3 = local_grad(e[2], x[0]), local_grad(e[2], x[1]), local_grad(e[2], x[2])
    d1 = [x11, x21, x31]
    d2 = [x12, x22, x32]
    d3 = [y1, y2, y3]
    Rn = x[1] * (x11 * x32 + x12 * x31) - x[2] * (x11 * x22 + x12 * x21)

    E41 = -k11 * (x11**2 - y1**2) - k12 * x11 * x12
    E51 = -k21 * x11 * x12 - k22 * (x12**2 - y1**2)
    E61 = (
        -k11 * x11 * x12
        - k12 * (x12**2 - y1**2)
        - k22 * x11 * x12
        - k21 * (x11**2 - y1**2)
    )

    E42 = (
        k11 * (y1 * (x[1] * y3 - x[2] * y2) + x11 * (x[2] * x21 - x[1] * x31))
        + k12 * 0.5 * (-Rn)
        + 2 * x11 * (y2 * x31 - y3 * x21)
    )
    E52 = (
        k22 * (y1 * (x[1] * y3 - x[2] * y2) + x12 * (x[2] * x22 - x[1] * x32))
        + k21 * 0.5 * (-Rn)
        + 2 * x12 * (y2 * x32 - y3 * x22)
    )

    E62 = (
        (k11 + k22) * 0.5 * (-Rn)
        + (k12 + k21) * (y1 * (x[1] * y3 - x[2] * y2))
        + k12 * x12 * (x[2] * x22 - x[1] * x32)
        + 2 * (y2 * (x12 * x31 + x11 * x32) - y3 * (x12 * x21 + x11 * x22))
        + k21 * x11 * (x[2] * x21 - x[1] * x31)
    )

    E43 = (
        k11 * x[2] * (y1**2 - x11**2)
        - k12 * x11 * x12 * x[2]
        + x11 * (y3 * x11 - 2 * y1 * x31)
    )
    E53 = (
        k22 * x[2] * (y1**2 - x12**2)
        - k21 * x11 * x12 * x[2]
        + x12 * (y3 * x12 - 2 * y1 * x32)
    )
    E63 = (
        -(k11 + k22) * x[2] * x11 * x12
        + x[2] * (k12 * (y1**2 - x12**2) + k21 * (y1**2 - x11**2))
        - 2 * y1 * (x12 * x31 + x11 * x32)
        + 2 * y3 * x11 * x12
    )

    E44 = (
        k11 * x[1] * (-(y1**2) + x11**2)
        + k12 * x11 * x12 * x[1]
        + x11 * (-y2 * x11 + 2 * y1 * x21)
    )
    E54 = (
        k22 * x[1] * (-(y1**2) + x12**2)
        + k21 * x11 * x12 * x[1]
        + x12 * (-y2 * x12 + 2 * y1 * x22)
    )
    E64 = (
        (k11 + k22) * x11 * x12 * x[1]
        + x[1] * (k12(-(y1**2) + x12**2) + k21 * (-(y1**2) + x11**2))
        + 2 * y1 * (x12 * x21 + x11 * x22)
        - 2 * y2 * x11 * x12
    )

    return as_tensor(
        [
            (x11**2, x11 * (x[1] * x31 - x[2] * x21), x[2] * x11**2, -x[1] * x11**2),
            (x12**2, x12 * (x[1] * x32 - x[2] * x22), x[2] * x12**2, -x[1] * x12**2),
            (2 * x11 * x12, Rn, 2 * x11 * x12 * x[2], -2 * x11 * x12 * x[1]),
            (E41, E42, E43, E44),
            (E51, E52, E53, E54),
            (E61, E62, E63, E64),
        ]
    )


def gamma_d(e, x):
    """Compute the gamma_d operator for MSG-Shell formulations.

    This function computes the gamma_d operator. It represents additional
    strain measures needed for the Timoshenko model.

    Parameters
    ----------
    e : tuple
        Local frame vectors (e1, e2, e3)
    x : ufl.SpatialCoordinate
        Spatial coordinates

    Returns
    -------
    ufl.Tensor
        6x4 matrix representing additional strain measures
    """
    x11, x21, x31 = (
        local_grad(e[0], x[0]),
        local_grad(e[0], x[1]),
        local_grad(e[0], x[2]),
    )
    x12, x22, x32 = (
        local_grad(e[1], x[0]),
        local_grad(e[1], x[1]),
        local_grad(e[1], x[2]),
    )
    y1, y2, y3 = local_grad(e[2], x[0]), local_grad(e[2], x[1]), local_grad(e[2], x[2])
    O = ufl.as_vector((0, 0, 0, 0))
    R = ufl.as_vector((-y1, -y3 * x[1] + y2 * x[2], -x[2] * y1, x[1] * y1))
    ret_val = as_tensor([O, O, O, x11 * x11 * R, x12 * x12 * R, 2 * x11 * x12 * R])
    return ret_val


def local_boun(mesh_l, frame, subdomains_l):
    """Set up function spaces and measures for boundary analysis.

    This function creates the necessary function spaces and measures
    for analyzing the boundary regions of a shell segment.

    Parameters
    ----------
    mesh_l : dolfinx.mesh.Mesh
        Mesh for the boundary region
    frame : tuple
        Local frame vectors (e1, e2, e3)
    subdomains_l : dolfinx.mesh.MeshTags
        Tags identifying different regions in the boundary mesh

    Returns
    -------
    tuple
        Contains:
        - e: interpolated local frame
        - V_l: function space
        - dv: trial function
        - v: test function
        - x: spatial coordinates
        - dx: measure
    """
    V_l = dolfinx.fem.functionspace(
        mesh_l, basix.ufl.element("CG", mesh_l.topology.cell_name(), 2, shape=(3,))
    )
    le1, le2, le3 = frame
    e1l, e2l, e3l = Function(V_l), Function(V_l), Function(V_l)

    fexpr1 = dolfinx.fem.Expression(
        le1, V_l.element.interpolation_points(), comm=MPI.COMM_WORLD
    )
    e1l.interpolate(fexpr1)

    fexpr2 = dolfinx.fem.Expression(
        le2, V_l.element.interpolation_points(), comm=MPI.COMM_WORLD
    )
    e2l.interpolate(fexpr2)

    fexpr3 = dolfinx.fem.Expression(
        le3, V_l.element.interpolation_points(), comm=MPI.COMM_WORLD
    )
    e3l.interpolate(fexpr3)

    frame = [e1l, e2l, e3l]
    dv = TrialFunction(V_l)
    v = TestFunction(V_l)
    x = SpatialCoordinate(mesh_l)
    dx = Measure("dx")(domain=mesh_l, subdomain_data=subdomains_l)

    return frame, V_l, dv, v, x, dx


# def A_mat(ABD, e_l, x_l, dx_l, nullspace_l, v_l, dvl, nphases):
#     """Assembly matrix

#     Parameters
#     ----------
#     ABD : _type_
#         _description_
#     e_l : _type_
#         _description_
#     x_l : _type_
#         _description_
#     dx_l : _type_
#         _description_
#     nullspace_l : _type_
#         _description_
#     v_l : _type_
#         _description_
#     dvl : _type_
#         _description_
#     nphases : _type_
#         _description_

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     F2 = sum([dot(dot(as_tensor(ABD[i]),gamma_h_abd(e_l,x_l,dvl)), gamma_h_abd(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)])
#     A_l = assemble_matrix(form(F2))
#     A_l.assemble()
#     A_l.setNullSpace(nullspace_l)
#     return A_l


def solve_eb_boundary(ABD, meshdata):
    """Solve the Euler-Bernoulli beam problem on a boundary.

    This function solves the Euler-Bernoulli beam equations on a
    boundary region using the provided ABD matrices.

    Parameters
    ----------
    ABD : list
        List of ABD matrices for each material
    meshdata : dict
        Dictionary containing mesh data for the boundary

    Returns
    -------
    tuple
        Contains:
        - D_eff: effective stiffness matrix
        - V0: fluctuation functions
    """

    # assert len(ABD) == nphases
    nphases = len(ABD)

    mesh = meshdata["mesh"]
    # frame = meshdata["frame"]
    frame = local_frame_1D(mesh)
    subdomains = meshdata["subdomains"]
    nullspace = meshdata["nullspace"]
    # For applying bc, we only require solved fluctuating functions (V0) as input to bc.
    e, V, dv, v_, x, dx = local_boun(mesh, frame, subdomains)
    mesh.topology.create_connectivity(1, 1)
    V0 = initialize_array(V)[0]
    A = A_mat(ABD, e, x, dx, compute_nullspace(V), v_, dv, nphases)
    for p in range(4):
        Eps = gamma_e(e, x)[:, p]
        F2 = sum(
            [
                dot(dot(as_tensor(ABD[i]), Eps), gamma_h_abd(e, x, v_)) * dx(i)
                for i in range(nphases)
            ]
        )
        r_he = form(rhs(F2))
        F = petsc.assemble_vector(r_he)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        nullspace.remove(F)
        w = solve_ksp(A, F, V)
        V0[:, p] = w.x.array[:]
    return V0


# NOTE: different definition between timo and eb scripts
# def initialize_array(V_l):
#     xxx = 3*len(np.arange(*V_l.dofmap.index_map.local_range))  # total dofs
#     V0 = np.zeros((xxx,4))
#     Dle = np.zeros((xxx,4))
#     Dhe = np.zeros((xxx,4))
#     Dhd = np.zeros((xxx,4))
#     Dld = np.zeros((xxx,4))
#     D_ed = np.zeros((4,4))
#     D_dd = np.zeros((4,4))
#     D_ee = np.zeros((4,4))
#     V1s = np.zeros((xxx,4))
#     return V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s


def initialize_array(V):
    """Initialize arrays for storing computation results.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Function space determining the size of arrays

    Returns
    -------
    tuple
        Contains initialized numpy arrays for:
        - V0: fluctuation functions
        - Dle, Dhe: strain matrices
        - D_ee: stiffness matrix
        - V1s: additional fluctuation functions
    """
    ndofs = 3 * len(np.arange(*V.dofmap.index_map.local_range))  # total dofs of mesh
    V0 = np.zeros((ndofs, 4))
    Dle = np.zeros((ndofs, 4))
    Dhe = np.zeros((ndofs, 4))
    D_ee = np.zeros((4, 4))
    V1s = np.zeros((ndofs, 4))
    return V0, Dle, Dhe, D_ee, V1s


def dof_mapping_quad(V, v2a, V_l, w_ll, boundary_facets_left, entity_mapl):
    """Map degrees of freedom between boundary and main meshes.

    This function maps the solution from a boundary mesh to the
    corresponding degrees of freedom in the main mesh.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
        Function space on main mesh
    v2a : dolfinx.fem.Function
        Function to store mapped values
    V_l : dolfinx.fem.FunctionSpace
        Function space on boundary mesh
    w_ll : dolfinx.fem.Function
        Solution on boundary mesh
    boundary_facets_left : array
        Array of boundary facet indices
    entity_mapl : array
        Mapping between boundary and main mesh entities

    Returns
    -------
    dolfinx.fem.Function
        Updated v2a with mapped values
    """
    deg = 2
    dof_S2L = []
    for i, xx in enumerate(entity_mapl):
        # For a unique facet obtain dofs
        dofs = locate_dofs_topological(
            V, 1, np.array([xx])
        )  # WB segment mesh facets dofs
        dofs_left = locate_dofs_topological(
            V_l, 1, np.array([boundary_facets_left[i]])
        )  # Boundary mesh facets dofs

        for k in range(deg + 1):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.x.array[3 * dofs[k] + j] = w_ll[
                        3 * dofs_left[k] + j
                    ]  # store boundary solution of fluctuating functions
    return v2a


def tangential_projection(u, n):
    """Project a vector field onto the tangent space.

    Parameters
    ----------
    u : ufl.Coefficient
        Vector field to project
    n : ufl.FacetNormal
        Normal vector field

    Returns
    -------
    ufl.Coefficient
        Projected vector field
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


def facet_vector_approximation(
    V: dolfinx.fem.FunctionSpace,
    mt: Union[dolfinx.mesh.MeshTags, None] = None,
    mt_id: Union[int, None] = None,
    tangent: bool = False,
    interior: bool = False,
    jit_options: Union[dict, None] = None,
    form_compiler_options: Union[dict, None] = None,
) -> dolfinx.fem.Function:
    jit_options = jit_options if jit_options is not None else {}
    form_compiler_options = (
        form_compiler_options if form_compiler_options is not None else {}
    )

    comm = V.mesh.comm  # MPI Communicator
    n = ufl.FacetNormal(V.mesh)  # UFL representation of mesh facet normal
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)  # Trial and test functions

    # Create interior facet integral measure
    dS = (
        ufl.dS(domain=V.mesh)
        if mt is None
        else ufl.dS(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    )

    # If tangent==True, the right-hand side of the problem should be a tangential projection of the facet normal vector.

    c = dolfinx.fem.Constant(
        V.mesh, (1.0, 1.0, 1.0)
    )  # Vector to tangentially project the facet normal vectors on

    a = (ufl.inner(u("+"), v("+")) + ufl.inner(u("-"), v("-"))) * dS
    L = (
        ufl.inner(tangential_projection(c, n("+")), v("+")) * dS
        + ufl.inner(tangential_projection(c, n("-")), v("-")) * dS
    )
    # If tangent==false the right-hand side is simply the facet normal vector.
    a = (ufl.inner(u("+"), v("+")) + ufl.inner(u("-"), v("-"))) * dS
    L = (ufl.inner(n("+"), v("+")) + ufl.inner(n("-"), v("-"))) * dS

    # Find all boundary dofs, which are the dofs where we want to solve for the facet vector approximation.
    # Start by assembling test functions integrated over the boundary integral measure.
    ones = dolfinx.fem.Constant(
        V.mesh, dolfinx.default_scalar_type((1,) * V.mesh.geometry.dim)
    )  # A vector of ones

    local_val = dolfinx.fem.form((ufl.dot(ones, v("+")) + ufl.dot(ones, v("-"))) * dS)
    local_vec = dolfinx.fem.assemble_vector(local_val)

    # For the dofs that do not lie on the boundary of the mesh the assembled vector has value zero.
    # Extract these dofs and use them to deactivate the corresponding block in the linear system we will solve.
    bdry_dofs_zero_val = np.flatnonzero(np.isclose(local_vec.array, 0))
    deac_blocks = np.unique(bdry_dofs_zero_val // V.dofmap.bs).astype(np.int32)

    # Create sparsity pattern by manipulating the blocks to be deactivated and set
    # a zero Dirichlet boundary condition for these dofs.
    bilinear_form = dolfinx.fem.form(
        a, jit_options=jit_options, form_compiler_options=form_compiler_options
    )
    pattern = dolfinx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()
    u_0 = dolfinx.fem.Function(V)
    u_0.x.array[:] = 0
    bc_deac = dolfinx.fem.dirichletbc(u_0, deac_blocks)

    # Create the matrix
    A = dolfinx.cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(bilinear_form._cpp_object)
    form_consts = dolfinx.cpp.fem.pack_constants(bilinear_form._cpp_object)
    dolfinx.fem.petsc.assemble_matrix(
        A, bilinear_form, constants=form_consts, coeffs=form_coeffs, bcs=[bc_deac]
    )

    # Insert the diagonal with the deactivated blocks.
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.petsc.insert_diagonal(
            A=A,
            V=bilinear_form.function_spaces[0],
            bcs=[bc_deac._cpp_object],
            diagonal=1.0,
        )
    A.assemble()

    # Assemble the linear form and the right-hand side vector.
    linear_form = dolfinx.fem.form(
        L, jit_options=jit_options, form_compiler_options=form_compiler_options
    )
    b = dolfinx.fem.petsc.assemble_vector(linear_form)

    # Apply lifting to the right-hand side vector and set boundary conditions.
    dolfinx.fem.petsc.apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc_deac])

    # Setup a linear solver using the Conjugate Gradient method.
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)

    # Solve the linear system and perform ghost update.
    nh = dolfinx.fem.Function(V)  # Function for the facet vector approximation
    solver.solve(b, nh.x.petsc_vec)
    nh.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )

    # Normalize the vectors to get the unit facet normal/tangent vector.
    nh_norm = ufl.sqrt(ufl.inner(nh, nh))  # Norm of facet vector
    cond_norm = ufl.conditional(
        ufl.gt(nh_norm, 1e-10), nh_norm, 1.0
    )  # Avoid division by zero
    nh_norm_vec = ufl.as_vector(
        (nh[0] / cond_norm, nh[1] / cond_norm, nh[2] / cond_norm)
    )

    nh_normalized = dolfinx.fem.Expression(
        nh_norm_vec, V.element.interpolation_points()
    )

    n_out = dolfinx.fem.Function(V)
    n_out.interpolate(nh_normalized)

    return n_out
