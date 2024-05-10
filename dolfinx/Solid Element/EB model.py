########## Euler Bernouli Model (Solid Element) ###################### 

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
from mpi4py import MPI
import numpy as np
import meshio
import dolfinx
from dolfinx.fem import VectorFunctionSpace, form, petsc, Function, FunctionSpace, locate_dofs_topological, apply_lifting, set_bc
from ufl import Jacobian, as_vector, dot, cross,sqrt, conditional, replace, as_matrix,FacetNormal
from ufl import lt,SpatialCoordinate, as_tensor, VectorElement, FiniteElement, MixedElement, Measure
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, dx, dot,eq
import petsc4py.PETSc
from contextlib import ExitStack
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from typing import Dict
import basix.ufl

# 2D mesh
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1,1, dolfinx.mesh.CellType.triangle)
# 3D mesh
#domain = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [1, 1, 1]], [1, 1, 1], dolfinx.mesh.CellType.hexahedron)

# Import External Mesh
#domain, subdomains, boundaries = gmshio.read_from_msh("3D_SG_2elemaxial.msh", MPI.COMM_WORLD,0, gdim=3)

from dolfinx.fem import VectorFunctionSpace, form, petsc, Function
import ufl
import numpy as np
from contextlib import ExitStack
import petsc4py.PETSc


# GELCOAT-1
E1,E2,E3=3.4400E+09, 3.4400E+09, 3.4400E+09
G12,G13,G23= 1.3230E+09, 1.3230E+09, 1.3230E+09
v12,v13,v23 = 0.3,0.3,0.3
material_parameters=[(E1,E2,E3,G12,G13,G23,v12,v13,v23)]
nphases = len(material_parameters)
x=ufl.SpatialCoordinate(domain)
Eps= ufl.as_vector((1,0,0,0,0,0))

def eps(v):
    E1=ufl.as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    return ufl.as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def sigma(v,i,Eps):     
        E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
        S=np.zeros((6,6))
        S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
        S[0,1], S[0,2]= -v12/E1, -v13/E1
        S[1,0], S[1,2]= -v12/E1, -v23/E2
        S[2,0], S[2,1]= -v13/E1, -v23/E2
        S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
        C=ufl.as_tensor(np.linalg.inv(S))
        s1= ufl.dot(C,eps(v)[1]+Eps)
        return ufl.as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]), C
V = VectorFunctionSpace(domain, ("Lagrange", 1),dim=3)
#V = dolfinx.fem.functionspace(domain, basix.ufl.element(
 #   "Lagrange", domain.topology.cell_name(), 1, shape=(3, ), gdim=3))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx = ufl.Measure('dx')(domain=domain)
points=domain.geometry.x
x = ufl.SpatialCoordinate(domain)

tdim=domain.topology.dim
num_cells=domain.topology.index_map(tdim).size_local

if tdim==2:
    a,b=0,1
elif tdim==3:
    a,b=1,2

Eps2T=ufl.as_tensor([(1,0,0,0,0,0),
                (0,0,0,0,x[a],-x[b]),
                (x[b],0,0,0,0,0),
                (-x[a],0,0,0,0,0)]) 

Eps2=ufl.as_tensor([(1,0,x[b],-x[a]),
                (0,0,0,0),
                (0,0,0,0),
                (0,0,0,0),
               (0,x[a],0,0),
               (0,-x[b],0,0)])

F2 = sum([ufl.inner(sigma(u, 0, Eps)[0], eps(v)[0])*dx]) 
A=  petsc.assemble_matrix(form(ufl.lhs(F2)))
A.assemble()
F = petsc.assemble_vector(form(ufl.rhs(F2)))
F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)


# In[24]:


# Create list of vectors for null space
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
x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
basis[3][dofs[1]] = -x2
basis[3][dofs[2]] = x1
# Create vector space basis and orthogonalize
dolfinx.la.orthonormalize(nullspace_basis)

nullspace = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
assert nullspace.test(A)
# Set the nullspace
A.setNullSpace(nullspace)
# Orthogonalize F to the null space of A^T
nullspace.remove(F)


# In[25]:


xx=len(points)*3
# Omega
omega=dolfinx.fem.assemble_scalar(form(sum([1*dx])))
V0 = np.zeros((xx,4))
Dhe=np.zeros((xx,4))
D_ee=np.zeros((4,4))

# Assembly
for p in range(4):
    Eps=Eps2[:,p] 
    F2 = sum([ufl.inner(sigma(u, 0, Eps)[0], eps(v)[0])*dx]) 
    F = petsc.assemble_vector(form(ufl.rhs(F2)))
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    w = Function(V)
    # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F, w.vector)
    w.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    Dhe[:,p]= F[:] # Dhe matrix formation
    V0[:,p]= w.vector[:] # V0 matrix formation
    
D1=np.matmul(V0.T,-Dhe)    
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([ufl.dot(Eps2T,ufl.dot(sigma(u,0,Eps)[1],Eps2))[s,k]*dx]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1 
D_eff=D_eff/omega
print(D_eff)
#print('Euler Bernoulli Stiffness Matrix')
#print(Deff_srt)
Comp_srt=np.linalg.inv(D_eff)                           







