########### EB Model General (dolfinx) ###################
############ Solid Elements  #########################
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
from mpi4py import MPI
import numpy as np
import meshio
import dolfinx
from dolfinx.fem import form, petsc, Function, functionspace, locate_dofs_topological, apply_lifting, set_bc
from ufl import Jacobian, as_vector, dot, cross,sqrt, conditional, replace, as_matrix,FacetNormal
from ufl import lt,SpatialCoordinate, as_tensor,  Measure
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, dx, dot,eq
import petsc4py.PETSc
from contextlib import ExitStack
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from typing import Dict
import ufl
import basix
from scipy.sparse import csr_matrix
#domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5,5, dolfinx.mesh.CellType.triangle)
domain, subdomains, boundaries = gmshio.read_from_msh("solid_cylinder_2D.msh", MPI.COMM_WORLD,0, gdim=3)

# GELCOAT-1
E1,E2,E3=10, 10, 10
G12,G13,G23= 3.846, 3.846, 3.846
v12,v13,v23 = 0.3,0.3,0.3

E11,E21,E31=10.3e10,10.3e10, 10.3e10
G121,G131,G231= 7.293e9, 7.293e9, 7.293e9
v121,v131,v231 = 0.3,0.3,0.3
material_parameters=np.array([(E1,E2,E3,G12,G13,G23,v12,v13,v23),(E11,E21,E31,G121,G131,G231,v121,v131,v231)])
nphases = len(material_parameters)

x=SpatialCoordinate(domain)
dx=Measure('dx')(domain=domain, subdomain_data=subdomains)
Eps= as_vector((1,0,0,0,0,0))

def eps(v,r):
    if r=='G_h':
        E1=as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    elif r=='G_l':
        E1=as_vector([v[0],0,0,0, v[2],v[1]])
    return E1

def C(i):
    [E1,E2,E3,G12,G13,G23,v12,v13,v23]= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
    return as_tensor(np.linalg.inv(S))

def sigma(v, i,Eps):
    s1= dot(C(i),eps(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),C

def sigma_gl(v, i, Eps):
    s1= dot(C(i),gamma_l(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])])

V= dolfinx.fem.functionspace(domain, basix.ufl.element(
    "CG", domain.topology.cell_name(), 1, shape=(3, )))

dv= ufl.TrialFunction(V)
v_ = ufl.TestFunction(V)
dx=Measure('dx')(domain=domain, subdomain_data=subdomains)
points=domain.geometry.x
x = ufl.SpatialCoordinate(domain)
a,b=0,1

Eps2=ufl.as_tensor([(1,0,x[b],-x[a]),
                (0,0,0,0),
                (0,0,0,0),
                (0,0,0,0),
               (0,x[a],0,0),
               (0,-x[b],0,0)])

xxx=3*V.dofmap.index_map.local_range[1]
Dle=np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
Dhd=np.zeros((xxx,4))
Dld=np.zeros((xxx,4))
V0=np.zeros((xxx,4))
D_ed=np.zeros((4,4))
D_dd=np.zeros((4,4))
D_ee=np.zeros((4,4))  
V1s=np.zeros((xxx,4))


# In[3]:


a_he=sum([dot(dot(C(i),eps(dv,'G_h')),eps(v_,'G_h'))*dx(i) for i in range(nphases)])
for p in range(4):
    Eps=Eps2[:,p]
    r_he=sum([dot(dot(C(i),Eps),eps(v_,'G_h'))*dx(i) for i in range(nphases)])
    F_l = petsc.assemble_vector(form(r_he))
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    A_l=assemble_matrix(form(a_he))
    A_l.assemble()
    w_l=Function(V)
    
    # Nullspace implement
    index_map = V.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(4)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

    # Dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list for i in range(3)]

    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0

    # Build rotational null space basis
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list
    x1, x2, x3 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    basis[3][dofs[0]] = -x2
    basis[3][dofs[1]] = x1
    
    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)

    nullspace_l = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)

    # Set the nullspace
    A_l.setNullSpace(nullspace_l)    

    nullspace_l.remove(F_l) # nullspace_l
    # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A_l) # A_l
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F_l, w_l.vector)
    w_l.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    Dhe[:,p]=  F_l[:]
    V0[:,p]= w_l.vector[:]  
  #  print('Computed','Dhe',(p+1)*25,'%')
    
D1=np.matmul(V0.T,-Dhe)   
x = SpatialCoordinate(domain)
def Dee(i):
    Cc=C(i)
    x2,x3=x[a],x[b]
    return as_tensor([(Cc[0,0], Cc[0,4]*x2-Cc[0,5]*x3,Cc[0,0]*x3,-Cc[0,0]*x2),
                      (Cc[4,0]*x2-Cc[5,0]*x3, x2*(Cc[4,4]*x2-Cc[5,4]*x3)-x3*(Cc[4,5]*x2-Cc[5,5]*x3),x3*(Cc[4,0]*x2-Cc[5,0]*x3),-x2*(Cc[4,0]*x2-Cc[5,0]*x3)),
                      (Cc[0,0]*x3,  x3*(Cc[0,4]*x2-Cc[0,5]*x3), Cc[0,0]*x3**2, -Cc[0,0]*x2*x3),
                      (-Cc[0,0]*x2, -x2*(Cc[0,4]*x2-Cc[0,5]*x3),  -Cc[0,0]*x2*x3, Cc[0,0]*x2**2)])
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([Dee(i)[s,k]*dx(i) for i in range(nphases)]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(D_eff)
