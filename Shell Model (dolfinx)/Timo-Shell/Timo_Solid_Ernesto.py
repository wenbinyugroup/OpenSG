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
domain, subdomains, boundaries = gmshio.read_from_msh("solid_cylinder_2D.msh", MPI.COMM_WORLD,0, gdim=3)

# GELCOAT-1
E1,E2,E3=3.4400E+09, 3.4400E+09, 3.4400E+09
G12,G13,G23= 1.3230E+09, 1.3230E+09, 1.3230E+09
v12,v13,v23 = 0.3,0.3,0.3
material_parameters=[(E1,E2,E3,G12,G13,G23,v12,v13,v23)]
nphases = len(material_parameters)
x=SpatialCoordinate(domain)
Eps= as_vector((1,0,0,0,0,0))

# Gamma_h*w and Gamma_l*w' column matrix
def eps(v,r):
    if r=='G_h':
        E1=as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    elif r=='G_l':
        E1=as_vector([v[0],0,0,0, v[2],v[1]])
    return E1

def C(i): # Stiffness matrix 
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
# Gamma_e matrix
Eps2=ufl.as_tensor([(1,0,x[b],-x[a]),
                (0,0,0,0),
                (0,0,0,0),
                (0,0,0,0),
               (0,x[a],0,0),
               (0,-x[b],0,0)])
#Initialize
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

# --------------Getting Dle

Dle=np.zeros((xx,4))
for p in range(4):
        Eps=Eps2[:,p] 
        F1=sum([inner(sigma_gl(dv,0,Eps),gamma_l(v_)[0])*dx])
        a1=lhs(F1)+(c1+c2+c3)*dx
        L1 = rhs(F1)      
        w1 = Function(W) 
        solve(a1 == L1, w1,[])       
        Dle[:,p]= -assemble(L1).get_local() # Dhe matrix formation
        
# --------------Getting Dhl
Eps=Eps2[:,0] 
F_dhl=sum([inner(sigma(dv,0,Eps)[0],gamma_l(v_)[0])*dx]) #Dlh
C=sigma(dv, 0, Eps)[1]

a3=dot(gamma_l(v_)[1],(C*eps(dv)[1]))*dx +(c1+c2+c3)*dx
L3 = rhs(F_dhl)
w3 = Function(W) 
solve(a3 == L3, w3,[])  

x3=len(assemble(L3).get_local()) 

A3=assemble(a3)
ai, aj, av= as_backend_type(A3).mat().getValuesCSR()
Dhl=csr_matrix((av, aj, ai))
Dhl=Dhl.toarray()

#DhlTV0
DhlV0= np.matmul(Dhl.T,V0)

#DhlTV0Dle
DhlTV0Dle= np.matmul(Dhl, V0)+ Dle

#V0DllV0
V0DllV0= np.matmul(np.matmul(V0.T,Dll),V0)

# V1s
b=DhlTV0Dle-DhlV0
V1s=scipy.sparse.linalg.spsolve(Dhh_csr, b, permc_spec=None, use_umfpack=True)

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
Ginv= np.matmul(np.matmul(Q_tim.T,C_tim-np.matmul(np.matmul(B_tim.T,Ainv),B_tim)),Q_tim)
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

#Com=np.linalg.inv(Deff_srt)
#SC_loc=Com[2,3]/Com[3,3], Com[1,3]/Com[3,3]
#print(Com)
#print('SC_loc',SC_loc)
print('  ')  
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(Deff_srt))