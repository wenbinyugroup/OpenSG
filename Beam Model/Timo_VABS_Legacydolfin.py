#!/usr/bin/env python
# coding: utf-8

# In[35]:


# -----------------------------OpenSG------------------------------
# -------------------Timoshenko Beam Model using 2DSG--------------
#
# This code is developed by multiscale structural analysis laboratory 
# at Purdue University under the supervision of Prof. Wenbin Yu. This 
# code is intended for general purpose usage for performing accurate
# and precise structural analysis. The code is based on Mechanics of 
# Structure Genome (MSG) theory to perform homogenization and 
# dehomogenization. 
#
# -----------------------------mesh import-------------------------------
from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from scipy.interpolate import griddata
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('classic')
import time
import scipy

# ** User Input- .msh file name**
fname= "2D_Ankit_cyl"
import subprocess
subprocess.check_output('dolfin-convert  '+ fname+ '.msh  '+ fname +'.xml', shell=True)
mesh = Mesh(fname + ".xml")
#subdomains = MeshFunction("size_t", mesh, fname + "_physical_region.xml")
#facets = MeshFunction("size_t", mesh, fname + "_facet_region.xml")



# In[47]:


#----------------------------- Material Data Input-----------------------------

# **The sequence of material initialization is corresponding to subdomain (physical_region) sequence**

# Carbon Uni
E1,E2,E3=3.700000e+10,    9.000000e+09,    9.000000e+09
G12,G13,G23= 4.000000e+09,    4.000000e+09,    4.000000e+09
v12,v13,v23 = 0.28,0.28,0.28
material_parameters=[(E1,E2,E3,G12,G13,G23,v12,v13,v23)]

nphases = len(material_parameters)
data=np.loadtxt('Ankit_Cyl_Orien.txt', delimiter=',', skiprows=0, dtype=str)
O=VectorElement("DG", mesh.ufl_cell(), 0,dim=3)
VO = FunctionSpace(mesh, O)
E1,E2,E3=Function(VO),Function(VO),Function(VO)

for i in range(mesh.num_cells()):
    d=data[i].split()[1:]
    a1=np.array([float(d[0]),float(d[1]),float(d[2])])
    b1=np.array([float(d[3]),float(d[4]),float(d[5])])
    c1=np.array([float(d[6]),float(d[7]),float(d[8])])
    EE1=a1-c1/np.linalg.norm(a1-c1)
    EE3=np.cross(EE1,(b1-c1))
    EE3=EE3/np.linalg.norm(EE3)
    EE2=np.cross(EE3,EE1)
    EE2=EE2/np.linalg.norm(EE2)
    for k in range(3):
        E1.vector()[3*i+k]=EE1[k]
        E2.vector()[3*i+k]=EE2[k]
        E3.vector()[3*i+k]=EE3[k]


# In[ ]:





# In[90]:


# Direction cosine matrix
dc_matrix=as_tensor([(E1[0],E2[0],E3[0]),(E1[1],E2[1],E3[1]),(E1[2],E2[2],E3[2])])
lay=-45
c,s=np.cos(np.deg2rad(lay)),np.sin(np.deg2rad(lay))
b11,b12,b13=c,s,0
b21,b22,b23=-s,c,0
b31,b32,b33=0,0,1
dc_matrix=dc_matrix*as_tensor([(c,s,0),(-s,c,0),(0,0,1)])
b11,b12,b13=dc_matrix[0,0],dc_matrix[0,1],dc_matrix[0,2]
b21,b22,b23=dc_matrix[1,0],dc_matrix[1,1],dc_matrix[1,2]
b31,b32,b33=dc_matrix[2,0],dc_matrix[2,1],dc_matrix[2,2]

R_sig=as_tensor([(b11*b11, b12*b12, b13*b13, 2*b12*b13, 2*b11*b13,2* b11*b12),
             (b21*b21, b22*b22, b23*b23, 2*b22*b23, 2*b21*b23, 2*b21*b22),
             (b31*b31, b32*b32, b33*b33, 2*b32*b33, 2*b31*b33, 2*b31*b32),
             (b21*b31, b22*b32, b23*b33, b23*b32+b22*b33, b23*b31+b21*b33, b22*b31+b21*b32),
             (b11*b31, b12*b32, b13*b33, b13*b32+b12*b33, b13*b31+b11*b33, b12*b31+b11*b32),
             (b11*b21, b12*b22, b13*b23, b13*b22+b12*b23, b13*b21+b11*b23, b12*b21+b11*b22)])

def C(i):
    [E1,E2,E3,G12,G13,G23,v12,v13,v23]= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12   
    C=as_tensor(np.linalg.inv(S))
    C1=dot(dot(R_sig,C),R_sig.T) 
    return C1


# In[91]:


## -----------------------------Functions-----------------------------

x = SpatialCoordinate(mesh)
Eps2T=as_tensor([(1,0,0,0,0,0),
                (0,0,0,0,x[0],-x[1]),
                (x[1],0,0,0,0,0),
                (-x[0],0,0,0,0,0)]) 

Eps2=as_tensor([(1,0,x[1],-x[0]),
                (0,0,0,0),
                (0,0,0,0),
                (0,0,0,0),
               (0,x[0],0,0),
               (0,-x[1],0,0)])

def eps(v):
    E1=as_vector([0,v[1].dx(0),v[2].dx(1),v[1].dx(1)+v[2].dx(0),v[0].dx(1),v[0].dx(0)])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def gamma_h(v):
    E1=as_vector([0,v[1].dx(0),v[2].dx(1),0.5*(v[1].dx(1)+v[2].dx(0)),0.5*(v[0].dx(1)),0.5*(v[0].dx(0))])
    return E1

def sigma(v, i,Eps):   
    s1= dot(C(i),eps(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),C(i)

def gamma_l(v):
    E1= as_vector([v[0],0,0,0, v[2],v[1]])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def sigma_gl(v, i, Eps):  
    s1= dot(C(i),gamma_l(v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])])

# -----------------------------FE Function Space-----------------------------

Ve = VectorElement("CG", mesh.ufl_cell(), 1,dim=3) 
Re = VectorElement("R", mesh.ufl_cell(), 0,dim=4)
W = FunctionSpace(mesh, MixedElement([Ve, Re]))
V = FunctionSpace(mesh, Ve)

v_,lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
Eps= as_vector((1,0,0,0,0,0))
dx = Measure('dx')(domain=mesh)

c1=lamb_[0]*dv[0]+lamb_[1]*dv[1]+lamb_[2]*dv[2]
c2=dlamb[0]*v_[0]+dlamb[1]*v_[1]+dlamb[2]*v_[2]
c3=lamb_[3]*(dv[1].dx(1)-dv[2].dx(0))+dlamb[3]*(v_[1].dx(1)-v_[2].dx(0))

# Omega
#omega=assemble(sum([Constant(1)*dx(i)for i in range(nphases)]))
omega=1
# -----------------------------Assembled matrices-----------------------------
i=0
F1=sum([inner(sigma_gl(dv,0,Eps),gamma_l(v_)[0])*dx])
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
        F1=sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx])
        
       # a1=lhs(F1)+(c1+c2+c3)*dx
        L1 = rhs(F1)      
      #  w1 = Function(W) 
      #  solve(a1 == L1, w1,[])       
        Dle[:,p]= -assemble(L1).get_local() # Dhe matrix formation
        
# --------------Getting Dhl
Eps=Eps2[:,0] 
F_dhl=sum([inner(sigma(dv,i,Eps)[0],gamma_l(v_)[0])*dx]) 

a3=lhs(F_dhl)+(c1+c2+c3)*dx
L3 = rhs(F_dhl)
#w3 = Function(W) 
#solve(a3 == L3, w3,[])  

x3=len(assemble(L3).get_local()) 
A3=assemble(a3)
ai, aj, av= as_backend_type(A3).mat().getValuesCSR()
Dhl=csr_matrix((av, aj, ai))
Dhl=Dhl.toarray()


# In[93]:


# --------------Getting Dhe, V0 

V0 = np.zeros((xx,4))
Dhe=np.zeros((xx,4))

for p in range(4):   
        Eps=Eps2[:,p] 
        F2 = sum([inner(sigma(dv, 0, Eps)[0], eps(v_)[0])*dx])   
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
        D_ee[s,k]=assemble(sum([dot(Eps2T,dot(sigma(dv,0,Eps)[1],Eps2))[s,k]*dx]))
        
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
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse.linalg
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
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(Deff_srt))  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




