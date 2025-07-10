
#-----------------------------OpenSG------------------------------
# -------------------2D/3D Timoshenko Model Uasing Periodic Boundary Conditions--------------
#
# This code is developed by Multiscale Structural Mechanics laboratory 
# at Purdue University under the supervision of Prof. Wenbin Yu. This 
# code is intended for general purpose usage for performing accurate
# and precise structural analysis. The code is based on Mechanics of 
# Structure Genome (MSG) theory to perform homogenization and 
# dehomogenization of complex composite laminates like wind blades.
#
# -----------------------------mesh import-------------------------------
from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('classic')
import time
import scipy
# **Input- .msh file name**
#
#fname= 

import subprocess
subprocess.check_output('dolfin-convert  '+ 'SG40_3D'+ '.msh  '+ 'SG40_3D' +'.xml', shell=True)
mesh = Mesh('SG40_3D' + ".xml")
subdomains = MeshFunction("size_t", mesh, 'SG40_3D' + "_physical_region.xml")

# -------------------------------Periodic Boundary Conditions-------------------------------
vertices = np.array([[0.1,0,0],   # 1: Right (x+)
                     [0.0,0,0]])   # 4: left   (x-)   

# ** Taking any geometric periodic points on x+ and x- face respectively based on beam model (3D SG)**

class PeriodicBoundary(SubDomain):
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[0,:]-self.vv[1,:] # first vector generating periodicity
        
    def inside(self, x, on_boundary):
        # faces
        left = near(x[0],self.vv[1,0]) 
        return bool((left) and on_boundary)

    def map(self, x, y):
        """ Mapping the right boundary to left """  
        # faces
        right = near(x[0],self.vv[0,0])
        if right:
            y[0] = x[0] - (self.a1[0])
            y[1] = x[1] - (self.a1[1])
            y[2] = x[2] - (self.a1[2])
        else: 
            y[0] = x[0] 
            y[1] = x[1] 
            y[2] = x[2]  
#-------------------Periodic Boundary Conditions End---------------

#----------------------------- Material Data Input-----------------------------

# **The sequence of material initialization is corresponding to subdomain (physical_region) sequence**

# GELCOAT-1
E1,E2,E3=3.4400E+09, 3.4400E+09, 3.4400E+09
G12,G13,G23= 1.3230E+09, 1.3230E+09, 1.3230E+09
v12,v13,v23 = 0.3,0.3,0.3
material_parameters=[(E1,E2,E3,G12,G13,G23,v12,v13,v23)]

# Gtriax-2
E1,E2,E3= 2.8700E+10, 1.6600E+10, 1.6700E+10
G12,G13,G23= 8.4000E+09, 3.4900E+09, 3.4900E+09
v12,v13,v23 = 0.50, 0.17, 0.0
material_parameters.append((E1,E2,E3,G12,G13,G23,v12,v13,v23))

# GUNI- 3
E1,E2,E3= 4.4600E+10, 1.7000E+10, 1.6700E+10
G12,G13,G23= 3.2700E+09, 3.5000E+09, 3.4800E+09
v12,v13,v23 = 0.26,0.26,0.35
material_parameters.append((E1,E2,E3,G12,G13,G23,v12,v13,v23))

# FOAM- 4
E1,E2,E3= 1.2920E+08, 1.2920E+08, 1.2920E+08
G12,G13,G23= 4.8947E+07, 4.8947E+07, 4.8947E+07
v12,v13,v23 = 0.32,0.32,0.32
material_parameters.append((E1,E2,E3,G12,G13,G23,v12,v13,v23))

# CarbonUD 5
E1,E2,E3= 1.1450E+11, 8.3900E+09, 8.3900E+09
G12,G13,G23= 5.9900E+09,5.9900E+09, 5.9900E+09
v12,v13,v23 = 0.27,0.27,0.27
material_parameters.append((E1,E2,E3,G12,G13,G23,v12,v13,v23))

nphases = len(material_parameters)

## -----------------------------Functions-----------------------------
#
dim=len(mesh.coordinates()[0])
if dim==2:
    a,b=0,1
elif dim==3:
    a,b=1,2
x = SpatialCoordinate(mesh)
Eps2T=as_tensor([(1,0,0,0,0,0),
                (0,0,0,0,x[a],-x[b]),
                (x[b],0,0,0,0,0),
                (-x[a],0,0,0,0,0)]) 

Eps2=as_tensor([(1,0,x[b],-x[a]),
                (0,0,0,0),
                (0,0,0,0),
                (0,0,0,0),
               (0,x[a],0,0),
               (0,-x[b],0,0)])

## -----------------------------Functions-----------------------------

def eps(v):
    E1=as_vector([0,v[1].dx(a),v[2].dx(b),v[1].dx(b)+v[2].dx(a),v[0].dx(b),v[0].dx(a)])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def gamma_h(v):
    E1=as_vector([0,v[1].dx(a),v[2].dx(b),v[1].dx(b)+v[2].dx(a),v[0].dx(b),v[0].dx(a)])
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
    s1= dot(C,eps(v)[1]+Eps)
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

# -----------------------------FE Function Space-----------------------------

Ve = VectorElement("CG", mesh.ufl_cell(), 1,dim=3) 
Re = VectorElement("R", mesh.ufl_cell(), 0,dim=4)
if dim==2:
    W = FunctionSpace(mesh, MixedElement([Ve, Re]))
elif dim==3:
    W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10))
V = FunctionSpace(mesh, Ve)

v_,lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
Eps= as_vector((1,0,0,0,0,0))
dx = Measure('dx')(domain=mesh, subdomain_data=subdomains)

c1=lamb_[0]*dv[0]+lamb_[1]*dv[1]+lamb_[2]*dv[2]
c2=dlamb[0]*v_[0]+dlamb[1]*v_[1]+dlamb[2]*v_[2]
if dim==2:
    c3=lamb_[3]*(dv[1].dx(1)-dv[2].dx(0))+dlamb[3]*(v_[1].dx(1)-v_[2].dx(0))
elif dim==3:
    c3=lamb_[3]*(dv[2].dx(1)-dv[1].dx(2))+dlamb[3]*(v_[2].dx(1)-v_[1].dx(2))

F2 = sum([inner(sigma(dv, i, Eps)[0], eps(v_)[0])*dx(i) for i in range(nphases)])    
a2=lhs(F2)+(c1+c2+c3)*dx
xx=len(assemble(rhs(F2)).get_local())
# Omega
omega=assemble(sum([Constant(1)*dx(i)for i in range(nphases)]))

# -----------------------------Assembled matrices-----------------------------

F1=sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx(i) for i in range(nphases)])
a1=lhs(F1)+(c1+c2+c3)*dx

A1=assemble(a1)
from scipy.sparse import csr_matrix
ai, aj, av= as_backend_type(A1).mat().getValuesCSR()
Dll=csr_matrix((av, aj, ai))
Dll=Dll.toarray()

xx=len(Dll)

# --------------Getting Dle

Dle=np.zeros((xx,4))
for p in range(4):
    
        Eps=Eps2[:,p] 
        F1=sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx(i) for i in range(nphases)])
        
        a1=lhs(F1)+(c1+c2+c3)*dx
        L1 = rhs(F1)      
        w1 = Function(W) 
        solve(a1 == L1, w1,[])       
        Dle[:,p]= -assemble(L1).get_local() # Dhe matrix formation
        
# --------------Getting Dhl
Eps=Eps2[:,0] 
F_dhl=sum([inner(sigma(dv,i,Eps)[0],gamma_l(v_)[0])*dx(i) for i in range(nphases)]) 

a3=lhs(F_dhl)+(c1+c2+c3)*dx
L3 = rhs(F_dhl)
w3 = Function(W) 
solve(a3 == L3, w3,[])  

x3=len(assemble(L3).get_local()) 

A3=assemble(a3)
ai, aj, av= as_backend_type(A3).mat().getValuesCSR()
Dhl=csr_matrix((av, aj, ai))
Dhl=Dhl.toarray()

# --------------Getting Dhe, V0 

V0 = np.zeros((xx,4))
Dhe=np.zeros((xx,4))

for p in range(4):   
        Eps=Eps2[:,p] 
        F2 = sum([inner(sigma(dv, i, Eps)[0], eps(v_)[0])*dx(i) for i in range(nphases)])   
        a2=lhs(F2)+(c1+c2+c3)*dx
        L2 = rhs(F2)      
        w2 = Function(W) 
        solve(a2 == L2, w2,[])       
        Dhe[:,p]= assemble(L2).get_local() # Dhe matrix formation
        V0[:,p]= (w2.vector().get_local()) # V0 matrix formation
        
# --------------Getting Dhh/E

A2=assemble(a2)
ai, aj, av= as_backend_type(A2).mat().getValuesCSR()
Dhh_csr=csr_matrix((av, aj, ai))
Dhh=Dhh_csr.toarray()   

# --------------Getting Dee
D_ee=np.zeros((4,4))
for s in range(4):
    for k in range(4): 
        D_ee[s,k]=assemble(sum([dot(Eps2T,dot(sigma(dv,i,Eps)[1],Eps2))[s,k]*dx(i) for i in range(nphases)]))
        
D1=np.matmul(V0.T,-Dhe)
D_eff= D_ee + D1 # Effective Stiffness Matrix
D_eff=D_eff/omega
D_eff

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
B_tim=B_tim/omega

# C_tim
C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)
C_tim=0.5*(C_tim+C_tim.T)
C_tim=C_tim/omega

# D_tim
D_tim= np.matmul(DhlTV0Dle.T, V1s)
D_tim=D_tim/omega

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
#print('Stiffness Matrix')
#print(Deff_srt)

Comp_srt=np.linalg.inv(Deff_srt)
#print('Compliance Matrix')
#print(Comp_srt)

np.set_printoptions(precision=4) 
print(np.around(Deff_srt))

