###############     OpenSG         ############################
########### Timo Model (dolfinx) ###################
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
domain, subdomains, boundaries = gmshio.read_from_msh("SG40_cut.msh", MPI.COMM_WORLD,0, gdim=3)

# GELCOAT-1
E1,E2,E3=3.4400E+03, 3.4400E+03, 3.4400E+03
G12,G13,G23= 1.3230E+03, 1.3230E+03, 1.3230E+03
v12,v13,v23 = 0.3,0.3,0.3
material_parameters=[(E1,E2,E3,G12,G13,G23,v12,v13,v23)]
nphases = len(material_parameters)
x=SpatialCoordinate(domain)
Eps= as_vector((1,0,0,0,0,0))

def eps(v,r):
    if r=='G_h':
        E1=as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    elif r=='G_l':
        E1=as_vector([v[0],0,0,0, v[2],v[1]])
    return E1

def C(i):
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
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

a_he=dot(dot(C(0),eps(dv,'G_h')),eps(v_,'G_h'))*dx
for p in range(4):
    Eps=Eps2[:,p]
    r_he=dot(dot(C(0),Eps),eps(v_,'G_h'))*dx
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
    assert nullspace_l.test(A_l)
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
    Cc=C(0)
    x2,x3=x[a],x[b]
    return as_tensor([(Cc[0,0], Cc[0,4]*x2-Cc[0,5]*x3,Cc[0,0]*x3,-Cc[0,0]*x2),
                      (Cc[4,0]*x2-Cc[5,0]*x3, x2*(Cc[4,4]*x2-Cc[5,4]*x3)-x3*(Cc[4,5]*x2-Cc[5,5]*x3),x3*(Cc[4,0]*x2-Cc[5,0]*x3),-x2*(Cc[4,0]*x2-Cc[5,0]*x3)),
                      (Cc[0,0]*x3,  x3*(Cc[0,4]*x2-Cc[0,5]*x3), Cc[0,0]*x3**2, -Cc[0,0]*x2*x3),
                      (-Cc[0,0]*x2, -x2*(Cc[0,4]*x2-Cc[0,5]*x3),  -Cc[0,0]*x2*x3, Cc[0,0]*x2**2)])
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(Dee(0)[s,k]*dx)
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(D_eff)

def gamma_h(v):
    E1=as_vector([0,v[1].dx(0),v[2].dx(1),0.5*(v[1].dx(1)+v[2].dx(0)),0.5*(v[0].dx(1)),0.5*(v[0].dx(0))])
    return E1

def sigma(v, i,Eps):
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
    C=as_tensor(np.linalg.inv(S))
    s1= dot(C,eps2(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),C

def gamma_l(v):
    E1= as_vector([v[0],0,0,0, v[2],v[1]])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def sigma_gl(v, i, Eps):
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
    C=as_tensor(np.linalg.inv(S))
    s1= dot(C,gamma_l(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])])
def eps2(v):
    E1=as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

F1=sum([inner(sigma_gl(dv,0,Eps),gamma_l(v_)[0])*dx])
a1=lhs(F1)
Dll=assemble_matrix(form(a1))
Dll.assemble()
ai, aj, av=Dll.getValuesCSR()
Dll=csr_matrix((av, aj, ai)).toarray()
for p in range(4):
    
        Eps=Eps2[:,p] 
        F1=sum([inner(sigma_gl(dv,0,Eps),gamma_l(v_)[0])*dx])
        L1 = rhs(F1)          
        Dle[:,p]= petsc.assemble_vector(form(L1))[:]

F_dhl=sum([inner(sigma(dv,0,Eps)[0],gamma_l(v_)[0])*dx]) 
a3=lhs(F_dhl)
Dhl=assemble_matrix(form(a3))
Dhl.assemble()
ai, aj, av=Dhl.getValuesCSR()
Dhl=csr_matrix((av, aj, ai)).toarray()


#DhlTV0
DhlV0= np.matmul(Dhl.T,V0)

#DhlTV0Dle
DhlTV0Dle= np.matmul(Dhl, V0)+ Dle

#V0DllV0
V0DllV0= np.matmul(np.matmul(V0.T,Dll),V0)

# V1s
bb=DhlTV0Dle-DhlV0

for p in range(4):
    w_l=Function(V)
    
    F=petsc4py.PETSc.Vec().createWithArray(bb[:,p],comm=MPI.COMM_WORLD)
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_l.remove(F)

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
    ksp.solve(F, w_l.vector)
    w_l.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    V1s[:,p]= w_l.vector[:]  
# Ainv
Ainv= np.linalg.inv(D_eff)

# B_tim
B_tim= np.matmul(DhlTV0Dle.T,V0)

# C_tim
C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)
C_tim=0.5*(C_tim+C_tim.T)

# D_tim
D_tim= np.matmul(DhlTV0Dle.T, V1s)

# Ginv
Q_tim=np.matmul(Ainv,np.array([(0,0),(0,0),(0,-1),(1,0)]))
Ginv= np.matmul(np.matmul(Q_tim.T,(C_tim-np.matmul(np.matmul(B_tim.T,Ainv),B_tim))),Q_tim)

G_tim=np.linalg.inv(Ginv)

Y_tim= np.matmul(np.matmul(B_tim.T,Q_tim),G_tim)
A_tim= D_eff + np.matmul(np.matmul(Y_tim,Ginv),Y_tim.T)

# Deff_srt

D=np.zeros((6,6))

D[4:6,4:6]=G_tim
D[0:4,4:6]=Y_tim
D[4:6,0:4]=Y_tim.T
D[0:4,0:4]=A_tim

Deff_srt=np.zeros((6,6))
Deff_srt[0,3:6]=A_tim[0,1:4]
Deff_srt[0,1:3]=Y_tim[0,:]
Deff_srt[0,0]=A_tim[0,0]

Deff_srt[3:6,3:6]=A_tim[1:4,1:4]
Deff_srt[3:6,1:3]=Y_tim[1:4,:]
Deff_srt[3:6,0]=A_tim[1:4,0]

Deff_srt[1:3,1:3]=G_tim
Deff_srt[1:3,3:6]=Y_tim.T[:,1:4]
Deff_srt[1:3,0]=Y_tim.T[:,0]
Deff_srt
print('  ')  
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(Deff_srt)) 
