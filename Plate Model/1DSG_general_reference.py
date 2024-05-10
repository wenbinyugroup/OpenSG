################# Plate Model (1D SG)##################
# legacy_dolfin

# -----------------------------mesh import-------------------------------
from __future__ import print_function
from dolfin import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from scipy.interpolate import griddata
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('classic')
import time
import meshio
from dolfin import *
import ufl
n_layer=4

mesh = IntervalMesh(n_layer,-4,0)
plot(mesh)


# Assign subdomain

#CompiledSubDomain('x[0] < -0.5 + DOLFIN_EPS').mark(subdomains, 0)
#CompiledSubDomain('x[0] > 0.5 - DOLFIN_EPS').mark(subdomains, 0)
#sub_domain_0=SubMesh(mesh,subdomains,0)
#sub_domain_1=SubMesh(mesh,subdomains,1)


# Input: layer#, angle, # Material ID
data=[[0,20,0],
      [1,45,1],
      [2,45,1],
      [3,20,0]]

# subdomain data
tdim = mesh.topology().dim()
subdomains = MeshFunction('size_t', mesh, tdim,1)
subdomains.array()[:]=[0,1,1,0]
#print(subdomains.array())

# Orientation DG0 Function
degree=0
O=VectorElement("DG", mesh.ufl_cell(), degree,dim=1)
VO = FunctionSpace(mesh, O)
vo=Function(VO)
vo.vector()[:]=[20,45,45,20] # can be taken from external data ()

VO.tabulate_dof_coordinates() # Element numbering starts from bottom to reference(zero)

# Material data

material_parameters=[]

material_parameters.append([142500000.0, 142500000.0, 142500000.0, 54000000.0, 54000000.0, 54000000.0, 0.3194, 0.3194, 0.3194])

material_parameters.append([43700000000.0, 16500000000.0, 15450000000.0, 3265000000.0, 3495000000.0, 3480000000.0, 0.262, 0.264, 0.35])

nphases = len(material_parameters)

x = SpatialCoordinate(mesh)
Eps2P=as_tensor([(1,0,0,x[0],0,0),
                  (0,1,0,0,x[0],0),
                  (0,0,0,0,0,0),
                  (0,0,0,0,0,0),
                  (0,0,0,0,0,0),
                  (0,0,1,0,0,x[0])]) 

Eps2PT=as_tensor([(1,0,0,0,0,0),
                 (0,1,0,0,0,0),
                 (0,0,0,0,0,1),
                 (x[0],0,0,0,0,0),
                 (0,x[0],0,0,0,0),
                 (0,0,0,0,0,x[0])])

def eps(v):
    E1= as_vector([0,0,v[2].dx(0),(v[1].dx(0)),(v[0].dx(0)),0])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def C_new(C,th):
   # th=np.deg2rad(theta)           # vo[0] act as th
    c,s,cs=cos(th),sin(th),cos(th)*sin(th)
    R_sig=np.array([(c**2, s**2, 0,0,0,2*cs),
                   (s**2, c**2, 0,0,0,-2*cs),
                   (0,0,1,0,0,0),
                   (0,0,0,c,-s,0),
                   (0,0,0,s,c,0),
                   (-cs,cs,0,0,0,c**2-s**2)])
    return dot(dot(as_tensor(R_sig),C),as_tensor(R_sig.T))

def sigma(v, i,Eps):  
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
    C=as_tensor(np.linalg.inv(S))
    C=C_new(C,vo[0]) 
    s1= dot(C,eps(v)[1]+Eps)      
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),C,s1         

Ve = VectorElement("CG", mesh.ufl_cell(), 1,dim=3)
Re = VectorElement("R", mesh.ufl_cell(), 0,dim=3)
W = FunctionSpace(mesh, MixedElement([Ve, Re]))
V = FunctionSpace(mesh, Ve)

v_,lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
dx = Measure('dx')(domain=mesh, subdomain_data=subdomains)

c=dot(lamb_,dv)+ dot(dlamb,v_)
xx=len(mesh.coordinates())*3+3


V0 = np.zeros((xx,6))
Dhe = np.zeros((xx,6))
D_ee=np.zeros((6,6))
for p in range(6):
        Eps=Eps2P[:,p] 
      #  F = sum([inner(sigma(dv, i, Eps)[0], (eps_t(v_)[0] + Eps2T_ten(Eps)))*dx(i) for i in range(nphases)])
        F = sum([inner(sigma(dv, i, Eps)[0], eps(v_)[0])*dx(i) for i in range(nphases)])
        a=lhs(F)+ c*dx
        L = rhs(F)      
        w = Function(W) 
        solve(a == L, w,[])       
        V0[:,p]= (w.vector().get_local()) # V0 matrix formation
        Dhe[:,p]= assemble(L).get_local() # Dhe matrix formation
        
D1=np.matmul(V0.T,-Dhe)
for s in range(6):
    for k in range(6): 
        D_ee[s,k]=assemble(sum([dot(Eps2PT,dot(sigma(dv, i, Eps)[1],Eps2P))[s,k]*dx(i) for i in range(nphases)]))
D_eff= D_ee + D1 # Effective Stiffness Matrix 

#-------------------------Printing Output Data-------------------------------
print('  ')  
print('Plate Stiffness Matrix')
np.set_printoptions(linewidth=np.inf)
with np.printoptions(precision=5, suppress=True):
    print(D_eff)
Com=np.linalg.inv(D_eff) 
print('  ') 
print('Compliance Matrix')
with np.printoptions(precision=10, suppress=True):
    print(Com)
