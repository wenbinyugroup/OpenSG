
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh, locate_entities
from mpi4py import MPI
import numpy as np
import meshio
import dolfinx
import basix
from dolfinx.fem import form, petsc, Function, functionspace, locate_dofs_topological, apply_lifting, set_bc
from ufl import Jacobian, as_vector, dot, cross,sqrt, conditional, replace, as_matrix,FacetNormal
from ufl import lt,SpatialCoordinate, as_tensor, Measure, Mesh
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, dx, dot, eq, grad
from dolfinx import fem
import petsc4py.PETSc
from contextlib import ExitStack
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from typing import Dict
import scipy
from scipy.sparse import csr_matrix
import ufl



material_parameters=np.array([ (1.000000e+01,  1.000000e+01, 1.000000e+01,1.000000e+00, 1.000000e+00, 1.000000e+00,0.3,0.3,0.3),
  (1.030000e+10,  1.030000e+10,  1.030000e+10,8.000000e+09,8.000000e+09,8.000000e+09,0.3,0.3,0.3),  
(1.030000e+10,  1.030000e+10,  1.030000e+10,8.000000e+09,8.000000e+09,8.000000e+09,0.3,0.3,0.3),
(3.700000e+10,  9.000000e+09, 9.000000e+09, 4.000000e+09,  4.000000e+09, 4.000000e+09,0.28,0.28,0.28),
 (1.000000e+07, 1.000000e+07, 1.000000e+07,  2.000000e+05,2.000000e+05,  2.000000e+05,0.3,0.3,0.3),
 (20e6,1.45e6,1.45e6,1e6,1e6,0.48e6,0.3,0.3,0.49)])       


# In[53]:


deg = 2
cell = ufl.Cell("interval")
elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
domain = ufl.Mesh(elem)

# Nodes (1D SG)
#th,s=[(0,0,0)],0 # Reference-------- 0
#for k in thick[ii]:
   # s=s+float(k)
 #   th.append((0,0,s) )
th=[0, 0.000381, 0.0008910000000000001, 0.010431]
# Elements  

cellss=[(0,1),(1,2),(2,3)]
nphases = len(cellss)

angle=[0,0,20]
matid=[1,2,3]
# Create 1D SG mesh
dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, th, domain)

# Subdomain data (MeshTags)
cell_map = dom.topology.index_map(dom.topology.dim)
num_cells_on_process = cell_map.size_local + cell_map.num_ghosts
cells = np.arange(num_cells_on_process, dtype=np.int32)
subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells) # assigning each element as subdomain

dx = Measure('dx')(domain=dom, subdomain_data=subdomain)
########################################################
x = SpatialCoordinate(dom)

Eps2=as_tensor([(1,0,0,x[0],0,0),
                  (0,1,0,0,x[0],0),
                  (0,0,0,0,0,0),
                  (0,0,0,0,0,0),
                  (0,0,0,0,0,0),
                  (0,0,1,0,0,x[0])])  # Gamma_e matrix


Eps= Eps2[:,0]

def eps(v): # (Gamma_h * w)
    E1= as_vector([0,0,v[2].dx(0),v[1].dx(0),v[0].dx(0),0])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def R_sig(C,t): # Rotation matrix
    th=np.deg2rad(t)
    c,s,cs=np.cos(th),np.sin(th),np.cos(th)*np.sin(th)
    R_Sig= np.array([(c**2, s**2, 0,0,0,-2*cs),
               (s**2, c**2, 0,0,0,2*cs),
               (0,0,1,0,0,0),
               (0,0,0,c,s,0),
               (0,0,0,-s,c,0),
               (cs,-cs,0,0,0,c**2-s**2)])
    return np.matmul(np.matmul(R_Sig,C),R_Sig.transpose())

def sigma(v,i,Eps):     
        E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[i]-1]
        S=np.zeros((6,6))
        S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
        S[0,1], S[0,2]= -v12/E1, -v13/E1
        S[1,0], S[1,2]= -v12/E1, -v23/E2
        S[2,0], S[2,1]= -v13/E1, -v23/E2
        S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
        C=np.linalg.inv(S)
        th=angle[i] # ii denotes the layup id
        C= R_sig(C,th) 
        s1= dot(as_tensor(C),eps(v)[1]+Eps)
        return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]), C
    
# Creating FE function Space
V = functionspace(dom, basix.ufl.element("CG", "interval", 2, shape=(3, )))
u = TrialFunction(V)
v = TestFunction(V)
Eps= Eps2[:,0]
F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0])*dx(i) for i in range(nphases)]) # Weak form of energy
A=  petsc.assemble_matrix(form(lhs(F2)))
A.assemble()
F = petsc.assemble_vector(form(rhs(F2)))
F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)

#################################################################
################## Create list of vectors for null space##############################
#################################################################

index_map = V.dofmap.index_map
nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(3)]

with ExitStack() as stack:
    vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
    basis = [np.asarray(x) for x in vec_local]

# Dof indices for each subspace (x, y and z dofs)
dofs = [V.sub(i).dofmap.list for i in range(3)]

# Build translational null space basis
for i in range(3):
    basis[i][dofs[i]] = 1.0

# Create vector space basis and orthogonalize

dolfinx.la.orthonormalize(nullspace_basis)
nullspace = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
#assert nullspace.test(A)
# Set the nullspace
A.setNullSpace(nullspace)
# Orthogonalize F to the null space of A^T
nullspace.remove(F)

xx = 3*V.dofmap.index_map.local_range[1] # total dofs

# Initialization
V0 = np.zeros((xx,6))
Dhe=np.zeros((xx,6))
D_ee=np.zeros((6,6))
# Assembly
for p in range(6):
    Eps=Eps2[:,p] 
    F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0])*dx(i) for i in range(nphases)]) # weak form
    F = petsc.assemble_vector(form(rhs(F2)))
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
x=SpatialCoordinate(dom)
# Getting Eps.T*C*Eps in D_eff calculation
def Dee(i):
        C=sigma(u, i, Eps)[1]
        x0=x[0]
        return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
                      (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
                      (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
                      (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
                      (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
                     (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])

for s in range(6):
    for k in range(6): 
        f=dolfinx.fem.form(sum([Dee(i)[s,k]*dx(i) for i in range(nphases)]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1 
np.set_printoptions(linewidth=np.inf)
print(D_eff)
