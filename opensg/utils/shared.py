from ufl import CellDiameter, FacetNormal, Jacobian, Measure, as_vector, avg, cross, div, dot, grad, inner, jump, sqrt
import dolfinx
import petsc4py
import numpy as np
from dolfinx.fem import Function
import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

def compute_nullspace(V, ABD=False):
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

    # Get geometric dim
    gdim = 3

    # Set dimension of nullspace
    # if ABD:
    dim = 3
    # else:
    #     dim = 4

    # Create list of vectors for null space
    nullspace_basis = [
        dolfinx.la.vector(V.dofmap.index_map, bs=V.dofmap.index_map_bs, dtype=petsc4py.PETSc.ScalarType)  # type: ignore
        for i in range(dim)
    ]

    basis = [b.array for b in nullspace_basis]
    dofs = [V.sub(i).dofmap.list.reshape(-1) for i in range(gdim)]

    # Build translational null space basis
    for i in range(gdim):
        basis[i][dofs[i]] = 1.0
    # Build rotational null space basis
    # if not ABD:
    #     xx = V.tabulate_dof_coordinates()
    #     dofs_block = V.dofmap.list.reshape(-1)
    #     x2, x3 = xx[dofs_block, 1], xx[dofs_block, 2]
    #     basis[3][dofs[1]] = -x3
    #     basis[3][dofs[2]] = x2
    #     for b in nullspace_basis:
    #         b.scatter_forward()

    dolfinx.la.orthonormalize(nullspace_basis)
    local_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    basis_petsc = [
        petsc4py.PETSc.Vec().createWithArray(x[:local_size], bsize=gdim, comm=V.mesh.comm)  # type: ignore
        for x in basis
    ]
    return petsc4py.PETSc.NullSpace().create(comm=V.mesh.comm, vectors=basis_petsc)  # type: ignore

def solve_ksp(A, F, V):
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
    Function
        solution function (displacement field)
    """

    w = Function(V)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    mat = ksp.getPC().getFactorMatrix()
    mat.setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    mat.setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    mat.setMumpsIcntl(icntl=4, ival=1) 
    mat.setMumpsIcntl(icntl=14, ival=80)
    mat.setMumpsIcntl(icntl=7, ival=7)
    mat.setMumpsIcntl(icntl=1, ival=1e-6)
    # Enable solver monitoring

#     PETSc.Options().setValue("ksp_monitor", "")  # Equivalent to "ksp_monitor": None in petsc_options
    ksp.setFromOptions()
    ksp.solve(F, w.x.petsc_vec)  # Solve scaled system

    w.x.petsc_vec.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    return w

def deri_constraint(dvl, v_l, mesh, nh):
    """Compute derivative constraints for boundary conditions.

    This function computes constraints on derivatives that are
    needed for proper boundary condition enforcement.

    Parameters
    ----------
    dvl : ufl.Coefficient
        Trial function
    v_l : ufl.Coefficient
        Test function
    mesh : dolfinx.mesh.Mesh
        Mesh object
    nh : float
        Constraint parameter

    Returns
    -------
    ufl.Form
        Form representing the derivative constraints
    """
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    h_avg = (h('+') + h('-')) / 2.0
    dS=Measure('dS')(domain=mesh)
    if mesh.topology.dim==1:
        alpha=1e10
        nn= - inner(avg(div(grad(dvl))), jump(grad(v_l), n))*dS \
            - inner(jump(grad(dvl), n), avg(div(grad(v_l))))*dS \
            + (alpha/h_avg**2)*inner(jump(grad(dvl),n), jump(grad(v_l),n))*dS
        return nn
    elif mesh.topology.dim==2:
        alpha=1e10
        beta=4e5
        nn= - inner(avg(div(grad(dvl))), jump(grad(v_l), n))*dS \
            - inner(jump(grad(dvl), n), avg(div(grad(v_l))))*dS \
            + (alpha/h_avg**2)*inner(jump(grad(dvl),n), jump(grad(v_l),n))*dS
        tt=- inner(avg(div(grad(dvl))), jump(grad(v_l), nh))*dS \
           - inner(jump(grad(dvl), nh), avg(div(grad(v_l))))*dS \
           + (beta/h_avg**2)*inner(jump(grad(dvl),nh), jump(grad(v_l),nh))*dS
        return nn+tt


def local_frame_1D(mesh):
    """Compute local orthonormal frame for 1D curved elements.

    This function computes an orthonormal frame (e1, e2, e3) at each point
    of a 1D curved mesh, where:
    - e1 is aligned with the global x-axis
    - e2 is the tangent vector
    - e3 is computed to complete the right-handed system

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        1D mesh object

    Returns
    -------
    tuple
        Three ufl.Vector objects (e1, e2, e3) forming the local frame
    """
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
    e2=  t1/ sqrt(dot(t1, t1))
    e1=  as_vector([1,0,0]) # Right Lay up
    e3= cross(e1,e2)
    e3=  e3/ sqrt(dot(e3, e3))
    return e1, e2, e3
