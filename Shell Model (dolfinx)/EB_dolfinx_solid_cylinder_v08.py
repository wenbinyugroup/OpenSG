###############     OpenSG         ############################
########### Euler Bernoulli Model (dolfinx) ###################
############ Test For Shell Cylinder  #########################
###############     OpenSG         ############################
########### Euler Bernoulli Model (dolfinx) ###################
############ Test For Shell Cylinder  #########################
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
#domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10,10, mesh.CellType.triangle)
domain, subdomains, boundaries = gmshio.read_from_msh("2D_SG_1Dshellvalidate_5radius.msh", MPI.COMM_WORLD,0, gdim=3)

# GELCOAT-1
E1,E2,E3=3.4400E+03, 3.4400E+03, 3.4400E+03
G12,G13,G23= 1.3230E+03, 1.3230E+03, 1.3230E+03
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
V= dolfinx.fem.functionspace(domain, basix.ufl.element(
    "CG", domain.topology.cell_name(), 1, shape=(3, )))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
dx = ufl.Measure('dx')(domain=domain)
points=domain.geometry.x
x = ufl.SpatialCoordinate(domain)
a,b=0,1

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
#################################################################
################## Create list of vectors for null space##############################
#################################################################
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
x = ufl.SpatialCoordinate(domain)
def Dee(i):
    C=sigma(u,i,Eps)[1]
    x2,x3=x[a],x[b]
    return as_tensor([(C[0,0], C[0,4]*x2-C[0,5]*x3,C[0,0]*x3,-C[0,0]*x2),
                      (C[4,0]*x2-C[5,0]*x3, x2*(C[4,4]*x2-C[5,4]*x3)-x3*(C[4,5]*x2-C[5,5]*x3),   x3*(C[4,0]*x2-C[5,0]*x3),-x2*(C[4,0]*x2-C[5,0]*x3)),
                      (C[0,0]*x3,  x3*(C[0,4]*x2-C[0,5]*x3), C[0,0]*x3**2, -C[0,0]*x2*x3),
                      (-C[0,0]*x2, -x2*(C[0,4]*x2-C[0,5]*x3),  -C[0,0]*x2*x3, C[0,0]*x2**2)])
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(Dee(0)[s,k]*dx)
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1 
np.set_printoptions(linewidth=np.inf)
print(np.around(D_eff)) 

# Optional
## The above dolfinx is verified with OpenSG's legacy-dolfin code.
## The snippet can be found in below code only for reference:

###############################################################Legacy-dolfin####################################
##
################################################################################################################
# -----------------------------mesh import-------------------------------
from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
from scipy.interpolate import griddata
%matplotlib inline
plt.style.use('classic')
import time
import scipy

# ** User Input- .msh file name**
fname= "2D_SG_1Dshellvalidate_5radius"
import subprocess
subprocess.check_output('dolfin-convert  '+ fname+ '.msh  '+ fname +'.xml', shell=True)
mesh = Mesh(fname + ".xml")
#----------------------------- Material Data Input-----------------------------

# **The sequence of material initialization is corresponding to subdomain (physical_region) sequence**

# GELCOAT-1
E1,E2,E3=3.4400E+03, 3.4400E+03, 3.4400E+03
G12,G13,G23= 1.3230E+03, 1.3230E+03, 1.3230E+03
v12,v13,v23 = 0.3,0.3,0.3
material_parameters=[(E1,E2,E3,G12,G13,G23,v12,v13,v23)]
nphases = len(material_parameters)

## -----------------------------Functions-----------------------------

x = SpatialCoordinate(mesh)
Eps2T=as_tensor([(1,0,0,0,0,0),
                (0,0,0,0,x[0],-x[1]),
                (x[1],0,0,0,0,0),
                (-x[0],0,0,0,0,0)]) 

Eps2=Eps2T.T

def eps(v):
    E1=as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def sigma(v, i,Eps):
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
    C=as_tensor(np.linalg.inv(S))
    s1= dot(C,eps(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),C,s1

# -----------------------------FE Function Space-----------------------------
Ve = VectorElement("CG", mesh.ufl_cell(), 2,dim=3) 
Re = VectorElement("R", mesh.ufl_cell(), 0,dim=4)
W = FunctionSpace(mesh, MixedElement([Ve, Re]))
V = FunctionSpace(mesh, Ve)

v_,lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
Eps= as_vector((1,0,0,0,0,0))
dx = Measure('dx')(domain=mesh)

c1=lamb_[0]*dv[0]+lamb_[1]*dv[1]+lamb_[2]*dv[2]
c2=dlamb[0]*v_[0]+dlamb[1]*v_[1]+dlamb[2]*v_[2]
c3=lamb_[3]*(dv[2].dx(0)-dv[1].dx(1))+dlamb[3]*(v_[2].dx(0)-v_[1].dx(1))

F2 = sum([inner(sigma(dv, 0, Eps)[0], eps(v_)[0])*dx])+(c1+c2+c3)*dx 
a2=lhs(F2)
xx=len(assemble(rhs(F2)).get_local())
# Omega
omega=assemble(sum([Constant(1)*dx]))

# -----------------------------Assembled matrices-----------------------------
# --------------Getting Dhe, V0 

V0 = np.zeros((xx,4))
Dhe=np.zeros((xx,4))
D_ee=np.zeros((4,4))

for p in range(4):   
        Eps=Eps2[:,p] 
        F2 = sum([inner(sigma(dv, 0, Eps)[0], eps(v_)[0])*dx])               
        w2 = Function(W) 
        L2=rhs(F2)
        solve(a2 == L2, w2,[])       
        Dhe[:,p]= assemble(L2).get_local() # Dhe matrix formation
        V0[:,p]= (w2.vector().get_local()) # V0 matrix formation

for s in range(4):
    for k in range(4): 
        D_ee[s,k]=assemble(sum([dot(Eps2T,dot(sigma(dv,0,Eps)[1],Eps2))[s,k]*dx]))
        
D1=np.matmul(V0.T,-Dhe)
D_eff= D_ee + D1 
D_eff=D_eff/omega
#-------------------------Printing Output Data-------------------------------
print('  ')  
print('Stiffness Matrix')
np.set_printoptions(linewidth=np.inf)
print(np.around(D_eff))
########################
########################
