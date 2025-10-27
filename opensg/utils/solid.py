"""
Created on Tue Jun 17 20:08:04 2025

@author: bagla0
"""

from typing import List, Tuple
from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from dolfinx.fem import form, Function, locate_dofs_topological, assemble_scalar
from ufl import (
    TrialFunction,
    TestFunction,
    as_tensor,
    dot,
    SpatialCoordinate,
    Measure,
    as_vector,
)
import ufl
#import petsc4py.PETSc
from petsc4py import PETSc
from slepc4py import SLEPc


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
        return np.isclose(x[0], xmin, atol=0.01)

    def is_right_boundary(x):
        return np.isclose(x[0], xmax, atol=0.01)

    return is_left_boundary, is_right_boundary


def Rsig(frame):
    """Generate rotation matrix in UFL form.
    
    Parameters
    ----------
    frame : array-like
        Local coordinate frame
        
    Returns
    -------
    ufl.Tensor
        Rotation matrix tensor
    """
    b11, b12, b13 = frame[0][0], frame[1][0], frame[2][0]
    b21, b22, b23 = frame[0][1], frame[1][1], frame[2][1]
    b31, b32, b33 = frame[0][2], frame[1][2], frame[2][2]

    return as_tensor(
        [
            (
                b11 * b11,
                b12 * b12,
                b13 * b13,
                2 * b12 * b13,
                2 * b11 * b13,
                2 * b11 * b12,
            ),
            (
                b21 * b21,
                b22 * b22,
                b23 * b23,
                2 * b22 * b23,
                2 * b21 * b23,
                2 * b21 * b22,
            ),
            (
                b31 * b31,
                b32 * b32,
                b33 * b33,
                2 * b32 * b33,
                2 * b31 * b33,
                2 * b31 * b32,
            ),
            (
                b21 * b31,
                b22 * b32,
                b23 * b33,
                b23 * b32 + b22 * b33,
                b23 * b31 + b21 * b33,
                b22 * b31 + b21 * b32,
            ),
            (
                b11 * b31,
                b12 * b32,
                b13 * b33,
                b13 * b32 + b12 * b33,
                b13 * b31 + b11 * b33,
                b12 * b31 + b11 * b32,
            ),
            (
                b11 * b21,
                b12 * b22,
                b13 * b23,
                b13 * b22 + b12 * b23,
                b13 * b21 + b11 * b23,
                b12 * b21 + b11 * b22,
            ),
        ]
    )


def C(i, frame, material_parameters):  # Stiffness matrix
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = material_parameters[i]
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1], S[0, 2] = -v12 / E1, -v13 / E1
    S[1, 0], S[1, 2] = -v12 / E1, -v23 / E2
    S[2, 0], S[2, 1] = -v13 / E1, -v23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
    CC = as_tensor(np.linalg.inv(S))
    R_sig = Rsig(frame)
    return dot(dot(R_sig, CC), R_sig.T)


def stress_output(mat_param, mesh, stress_3D, points):
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

    points = np.array(points, dtype=mesh.geometry.x.dtype)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(
        bb_tree, points
    )
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, potential_colliding_cells, points
    )
    points_on_proc = []
    cells = []
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
    cells = np.array(cells, dtype=np.int32)
    # print(cells)
    return stress_3D.eval(points_on_proc, cells)

def CC(mat_param):
    CC = []
    for i in range(len(mat_param)):
        E1, E2, E3, G12, G13, G23, v12, v13, v23 = mat_param[i]
        S = np.zeros((6, 6))
        S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
        S[0, 1], S[0, 2] = -v12 / E1, -v13 / E1
        S[1, 0], S[1, 2] = -v12 / E1, -v23 / E2
        S[2, 0], S[2, 1] = -v13 / E1, -v23 / E2
        S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
        CC.append(np.linalg.inv(S))
    return CC

def get_mass_solid(meshdata, density, Taper=False):  # Mass matrix
    nphases=max(meshdata["subdomains"].values)+1
    x, dx = (
        ufl.SpatialCoordinate(meshdata["mesh"]),
        ufl.Measure("dx")(
            domain=meshdata["mesh"], subdomain_data=meshdata["subdomains"]
        ),
    )
    coord=meshdata["mesh"].geometry.x
    L=1
    if Taper:
        L = max(coord[:, 0]) - min(coord[:, 0])
        
    mu = assemble_scalar(form(sum([density[i] * dx(i) for i in range(nphases)])))
    xm2 = (1 / mu) * assemble_scalar(
        form(sum([x[1] * density[i] * dx(i) for i in range(nphases)]))
    )
    xm3 = (1 / mu) * assemble_scalar(
        form(sum([x[2] * density[i] * dx(i) for i in range(nphases)]))
    )
    i22 = assemble_scalar(
        form(sum([(x[2] ** 2) * density[i] * dx(i) for i in range(nphases)]))
    )
    i33 = assemble_scalar(
        form(sum([(x[1] ** 2) * density[i] * dx(i) for i in range(nphases)]))
    )
    i23 = assemble_scalar(
        form(sum([x[1] * x[2] * density[i] * dx(i) for i in range(nphases)]))
    )
    return (1/L)*np.array(
        [
            (mu, 0, 0, 0, mu * xm3, -mu * xm2),
            (0, mu, 0, -mu * xm3, 0, 0),
            (0, 0, mu, mu * xm2, 0, 0),
            (0, -mu * xm3, mu * xm2, i22 + i33, 0, 0),
            (mu * xm3, 0, 0, 0, i22, i23),
            (-mu * xm2, 0, 0, 0, i23, i33),
        ]
    )


def gamma_e(x):
    """Compute gamma_e matrix.

    Parameters
    ----------
    x : array-like
        Coordinate point

    Returns
    -------
    ufl.Tensor
        Gamma_e matrix tensor
    """
    gamma_e = as_tensor(
        [
            (1, 0, x[2], -x[1]),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, x[1], 0, 0),
            (0, -x[2], 0, 0),
        ]
    )

    return gamma_e


def gamma_h(dx, v, dim):
    aa, b = 1, 2
    if dim == 2:
        E1 = as_vector(
            [
                0,
                v[1].dx(aa),
                v[2].dx(b),
                v[1].dx(b) + v[2].dx(aa),
                v[0].dx(b),
                v[0].dx(aa),
            ]
        )
    elif dim == 3:
        E1 = as_vector(
            [
                v[0].dx(0),
                v[1].dx(aa),
                v[2].dx(b),
                v[1].dx(b) + v[2].dx(aa),
                v[0].dx(b) + v[b].dx(0),
                v[0].dx(aa) + v[aa].dx(0),
            ]
        )

    return E1


def gamma_l(v):
    # e,x required as element can be of left/right boundary or quad mesh
    ret_val = as_vector([v[0], 0, 0, 0, v[2], v[1]])
    return ret_val


def local_boun(mesh, frame, subdomains):
    V = dolfinx.fem.functionspace(
        mesh, basix.ufl.element("CG", mesh.topology.cell_name(), 1, shape=(3,))
    )
    le1, le2, le3 = frame
    e1l, e2l, e3l = Function(V), Function(V), Function(V)

    fexpr1 = dolfinx.fem.Expression(
        le1, V.element.interpolation_points, comm=MPI.COMM_WORLD
    )
    e1l.interpolate(fexpr1)

    fexpr2 = dolfinx.fem.Expression(
        le2, V.element.interpolation_points, comm=MPI.COMM_WORLD
    )
    e2l.interpolate(fexpr2)

    fexpr3 = dolfinx.fem.Expression(
        le3, V.element.interpolation_points, comm=MPI.COMM_WORLD
    )
    e3l.interpolate(fexpr3)

    frame = [e1l, e2l, e3l]
    dv = TrialFunction(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)
    dx = Measure("dx")(domain=mesh, subdomain_data=subdomains)

    return frame, V, dv, v, x, dx


def initialize_array(V):
    xxx = 3 * V.dofmap.index_map.local_range[1]  # total dofs
    V0 = np.zeros((xxx, 4))
    Dle = np.zeros((xxx, 4))
    Dhe = np.zeros((xxx, 4))
    Dee = np.zeros((4, 4))
    V1s = np.zeros((xxx, 4))
    return V0, Dle, Dhe, Dee, V1s


def dof_mapping_quad(V, v2a, V_l, w_ll, boundary_facets_left, entity_mapl):
    """dof mapping makes solved unknown value w_l(Function(V_l)) assigned to v2a (Function(V)).
    The boundary of wind blade mesh is a 1D curve. The facet/edge number is obtained from cell to edge connectivity (conn3) showed in subdomain subroutine.
    The same facet/edge number of extracted mesh_l (submesh) is obtaine din entity_mapl (gloabl mesh number). refer how submesh was generated.
    Therefore, once identifying the edge number being same for global(mesh)&boundary mesh(mesh_l), we equate the dofs and store w_l to v2a.
    The dofs can be verified by comparing the coordinates of local and global dofs if required.

    Parameters
    ----------
    V : _type_
        _description_
    v2a : _type_
        _description_
    V_l : _type_
        _description_
    w_ll : 1D array (len(4))
        Fluctuating function data for case p
    boundary_facets_left : _type_
        _description_
    entity_mapl : _type_
        _description_


    Returns
    -------
    _type_
        _description_
    """
    dof_S2L = []
    for i, xx in enumerate(entity_mapl):
        dofs = locate_dofs_topological(V, 2, np.array([xx]))
        dofs_left = locate_dofs_topological(V_l, 2, np.array([boundary_facets_left[i]]))

        for k in range(len(dofs)):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.x.array[3 * dofs[k] + j] = w_ll[
                        3 * dofs_left[k] + j
                    ]  # store boundary solution of fluctuating functions
    return v2a


def recov(st):
    recov = np.zeros((6, 6))
    recov[0, 1], recov[0, 2] = st[5], -st[4]
    recov[1, 0], recov[1, 2] = -st[5], st[3]
    recov[2, 0], recov[2, 1] = st[4], -st[3]
    recov[3:6, 3:6] = recov[0:3, 0:3]

    recov[3, 1], recov[3, 2] = st[2], -st[1]
    recov[4, 0], recov[4, 2] = -st[2], st[0]
    recov[5, 0], recov[5, 1] = st[1], -st[0]

    return recov


def EPS_get_spectrum(
    EPS: SLEPc.EPS, V: dolfinx.fem.FunctionSpace
) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]:
    """Retrieve eigenvalues and eigenfunctions from SLEPc EPS object.
    Parameters
    ----------
    EPS
       The SLEPc solver
    V
       The function space
    Returns
    -------
        Tuple consisting of: List of complex converted eigenvalues,
         lists of converted eigenvectors (real part) and (imaginary part)
    """
    # Get results in lists
    eigval = list()
    eigvec_r = list()
    eigvec_i = list()
    for i in range(EPS.getConverged()):
        vr = dolfinx.fem.Function(V)
        vi = dolfinx.fem.Function(V)

        eigval.append(EPS.getEigenpair(i, vr.x.petsc_vec, vi.x.petsc_vec))
        eigvec_r.append(vr)
        eigvec_i.append(vi)  # Sort by increasing magnitude
    idx = np.argsort(np.abs(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)


def solve_GEP_shiftinvert(
    A: PETSc.Mat,
    B: PETSc.Mat,
    problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GHEP,
    solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
    nev: int = 2,
    tol: float = 1e-6,
    max_it: int = 1000,
    target: float = 5,
    shift: float = 1,
) -> SLEPc.EPS:
    """
     Solve generalized eigenvalue problem A*x=lambda*B*x using shift-and-invert
     as spectral transform method.
     Parameters
     ----------
     A
        The matrix A
     B
        The matrix B
     problem_type
        The problem type, for options see: https://bit.ly/3gM5pth
    solver:
        Solver type, for options see: https://bit.ly/35LDcMG
     nev
         Number of requested eigenvalues.
     tol
        Tolerance for slepc solver
     max_it
        Maximum number of iterations.
     target
        Target eigenvalue. Also used for sorting.
     shift
        Shift 'sigma' used in shift-and-invert.
    interval
        A tuple (min_val, max_val) to restrict the search to a specific
        interval on the real axis. To search for only positive eigenvalues,
        you could use `interval=(1e-9, PETSc.DECIMAL_MAX)`.
        This feature is supported by solvers like KRYLOVSCHUR.
    
    Returns
    -------
    EPS
    The SLEPc solver
    """

    # Build an Eigenvalue Problem Solver object
    EPS = SLEPc.EPS()
    EPS.create(comm=MPI.COMM_WORLD)
    EPS.setOperators(A, B)
    EPS.setProblemType(problem_type)

    # set the number of eigenvalues requested
    EPS.setDimensions(nev=nev, ncv=25)
    # Set solver
    EPS.setType(solver)
    # set eigenvalues of interest
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    # EPS.setTarget(target)  # sorting
    #  EPS.setInterval(1e-3, 1e3)

    # set tolerance and max iterations
    EPS.setTolerances(tol=tol, max_it=max_it)
    # Set up shift-and-invert
    # Only work if 'whichEigenpairs' is 'TARGET_XX'
    # Restrict the computed eigenvalues to a specific interval.
    # This is the key change to filter for only positive eigenvalues.

    #
    ST = EPS.getST()
    #  ST.setType(SLEPc.ST.Type.SINVERT)
    #  ST.setShift(shift)

    ST.getKSP().setType("preonly")
    ST.getKSP().getPC().setType("lu")
    ST.getKSP().getPC().setFactorSolverType("mumps")
    EPS.setST(ST)
    # set monitor
    #  it_skip = 1

    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    return EPS

def eigen_eps(u):
    """Compute strain tensor from displacement field.
    
    Parameters
    ----------
    u : dolfinx.fem.Function
        Displacement field
        
    Returns
    -------
    ufl.Expr
        Strain Vector
    """
    eps=as_vector([
        u[0].dx(0),
        u[1].dx(1),
        u[2].dx(2),
        (u[1].dx(2)+u[2].dx(1)),
        (u[0].dx(2)+u[2].dx(0)),
        (u[0].dx(1)+u[1].dx(0))
        ])
    return eps

