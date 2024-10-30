# Initialization of libraries
from mpi4py import MPI
import numpy as np
import dolfinx
import basix
from dolfinx.fem import form, petsc, Function, functionspace
from ufl import TrialFunction, TestFunction, inner, lhs, rhs
import petsc4py.PETSc
from mpi4py import MPI
import ufl
from contextlib import ExitStack


def ksp_solve(A, F, V):
    """Krylov Subspace Solver for Aw = F

    Parameters
    ----------
    A : array
        stiffness matrix
    F : array
        Load or force vector
    V : function space
        _description_

    Returns
    -------
    array
        solution vector (displacement field)
    """
    w = Function(V)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(
        icntl=25, ival=0
    )  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F, w.vector)
    w.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD
    )
    ksp.destroy()
    return w

def nullspace(V):
    """Compute nullspace to restrict Rigid body motions

    Constructs a translational null space for the vector-valued function space V
    and ensures that it is properly orthonormalized.

    Parameters
    ----------
    V : functionspace
        _description_

    Returns
    -------
    NullSpace
        Nullspace of V
    """
    # extract the Index Map from the Function Space
    index_map = V.dofmap.index_map

    # initialize nullspace basis with petsc vectors
    nullspace_basis = [
        dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs)
        for _ in range(4)
    ]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]

    # identify the degrees of freedom indices for each subspace (x, y, z)
    dofs = [V.sub(i).dofmap.list for i in range(3)]

    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0

    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)

    return petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)


def ABD_mat(ii, thick, nlay, angle, mat_names, material_database):
    """Compute the ABD matrix for a composite layup structure

    Constructs a local stiffness matrix for a composite laminate
    by assembling the contributions from multiple layers.

    Parameters
    ----------
    ii : _type_
        _description_
    thick : _type_
        _description_
    nlay : _type_
        _description_
    material_parameters : _type_
        _description_
    matid : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    ## 1D mesh generation
    # initialization
    deg = 2
    cell = ufl.Cell("interval")
    elem = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
    domain = ufl.Mesh(elem)

    # nodes (1D SG)
    th, s = [0], 0  # Reference starting point
    for k in thick[ii]:
        s = s + k  # Add the thickness of each layer
        th.append(s)
    points = np.array(th)
    
    # elements
    cell = []
    for k in range(nlay[ii]):
        cell.append([k, k + 1])
    cellss = np.array(cell)
    
    # Create mesh object
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, points, domain)
    num_cells = dom.topology.index_map(dom.topology.dim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    
    #
    # Note: For a layup like layup1: containing 3 layers:
    # we defined 1D mesh with 3 elements and make each element a seperate subdomain(where integration is computed by *dx(i)
    # for subdomain i) using cells-[0,1,2]
    
    # assigning each element as subdomain
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells)
    x, dx = ufl.SpatialCoordinate(dom), ufl.Measure("dx")(domain=dom, subdomain_data=subdomain)
    gamma_e = create_gamma_e(x)

    nphases = len(cells)

    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3,)))
    u, v = TrialFunction(V), TestFunction(V)
    
    # Weak form of energy
    # dx is for integration
    F2 = sum(
        [inner(sigma(
            vector=u, material_parameters=material_database[mat_names[ii][i]], 
            theta=angle[ii][i], Eps=gamma_e[:,0])[0], 
        eps(v)[0]) * dx(i) for i in range(nphases)]
    )
    
    # lhs gives left hand side of weak form : coeff matrix here
    A = petsc.assemble_matrix(form(lhs(F2)))  
    A.assemble()
    null = nullspace(V)
    A.setNullSpace(null)  # Set the nullspace
    xx = 3 * V.dofmap.index_map.local_range[1]  # total dofs
    
    # Initialization
    V0, Dhe, D_ee = np.zeros((xx, 6)), np.zeros((xx, 6)), np.zeros((6, 6))

    # Assembly
    for p in range(6):
        Eps = gamma_e[:, p]
        
        # weak form
        F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0]) * dx(i) for i in range(nphases)])
        
        # rhs is used for getting right hand side of Aw = F; (which is F here)
        F = petsc.assemble_vector(form(rhs(F2)))
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(F)  # Orthogonalize F to the null space of A^T
        w = ksp_solve(A, F, V)
        Dhe[:, p] = F[:]  # Dhe matrix formation
        V0[:, p] = w.vector[:]  # V0 matrix formation
        
    D1 = np.matmul(V0.T, -Dhe)  # Additional information matrix

    # Scalar assembly for each term of D_ee matrix
    for s in range(6):  
        for k in range(6):
            # Scalar assembly
            f = dolfinx.fem.form(sum([Dee(i)[s, k] * dx(i) for i in range(nphases)]))  
            D_ee[s, k] = dolfinx.fem.assemble_scalar(f)
            
    D_eff = D_ee + D1
    
    return D_eff

def create_gamma_e(x):
    """_summary_

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    gamma_e = ufl.as_tensor([
        (1, 0, 0, x[0], 0, 0),
        (0, 1, 0, 0, x[0], 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, x[0]),
    ])
    
    return gamma_e

def R_sig(C, t):
    """Compute rotation matrix

    Parameters
    ----------
    C : _type_
        _description_
    t : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    th = np.deg2rad(t)
    c, s, cs = np.cos(th), np.sin(th), np.cos(th) * np.sin(th)
    R_Sig = np.array(
        [
            (c**2, s**2, 0, 0, 0, -2 * cs),
            (s**2, c**2, 0, 0, 0, 2 * cs),
            (0, 0, 1, 0, 0, 0),
            (0, 0, 0, c, s, 0),
            (0, 0, 0, -s, c, 0),
            (cs, -cs, 0, 0, 0, c**2 - s**2),
        ]
    )
    return np.matmul(np.matmul(R_Sig, C), R_Sig.transpose())

def Dee(i):
    """ Simplifying gamma_e.T*C*gamma_e

    Parameters
    ----------
    i : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    C = sigma(u, i, Eps)[1]
    x0 = x[0]
    return ufl.as_tensor(
        [
            (C[0, 0], C[0, 1], C[0, 5], x0 * C[0, 0], x0 * C[0, 1], x0 * C[0, 5]),
            (C[1, 0], C[1, 1], C[1, 5], x0 * C[1, 0], x0 * C[1, 1], x0 * C[1, 5]),
            (C[5, 0], C[5, 1], C[5, 5], x0 * C[5, 0], x0 * C[5, 1], x0 * C[5, 5]),
            (
                x0 * C[0, 0],
                x0 * C[0, 1],
                x0 * C[0, 5],
                x0 * x0 * C[0, 0],
                x0 * x0 * C[0, 1],
                x0 * x0 * C[0, 5],
            ),
            (
                x0 * C[1, 0],
                x0 * C[1, 1],
                x0 * C[1, 5],
                x0 * x0 * C[1, 0],
                x0 * x0 * C[1, 1],
                x0 * x0 * C[1, 5],
            ),
            (
                x0 * C[5, 0],
                x0 * C[5, 1],
                x0 * C[5, 5],
                x0 * x0 * C[5, 0],
                x0 * x0 * C[5, 1],
                x0 * x0 * C[5, 5],
            ),
        ]
    )

def sigma(vector, material_parameters, theta, Eps):
    E1, E2, E3 = material_parameters["E"]
    G12, G13, G23 = material_parameters["G"]
    v12, v13, v23 = material_parameters["nu"]
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1], S[0, 2] = -v12 / E1, -v13 / E1
    S[1, 0], S[1, 2] = -v12 / E1, -v23 / E2
    S[2, 0], S[2, 1] = -v13 / E1, -v23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
    C = np.linalg.inv(S)
    C = R_sig(C, theta)
    s1 = ufl.dot(ufl.as_tensor(C), eps(vector)[1] + Eps)
    return ufl.as_tensor([(s1[0], s1[5], s1[4]), (s1[5], s1[1], s1[3]), (s1[4], s1[3], s1[2])]), C
    
def eps(vector):  # (Gamma_h * w)
    E1 = ufl.as_vector([0, 0, vector[2].dx(0), (vector[1].dx(0)), (vector[0].dx(0)), 0])
    ans = ufl.as_tensor([
        (E1[0], 0.5 * E1[5], 0.5 * E1[4]),
        (0.5 * E1[5], E1[1], 0.5 * E1[3]),
        (0.5 * E1[4], 0.5 * E1[3], E1[2]),
        ])
    return ans, E1
