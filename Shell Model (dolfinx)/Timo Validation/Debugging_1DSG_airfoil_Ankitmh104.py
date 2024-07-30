#!/usr/bin/env python
# coding: utf-8

# In[17]:


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


# In[18]:


cell = ufl.Cell("interval")
elem= basix.ufl.element("S", "interval", 1, shape=(3, ))
domain = ufl.Mesh(elem)

dd=['l12','l23','l34','h34','h23','h12']
data=np.loadtxt('datapoints_mh104.txt', delimiter=',', skiprows=0, dtype=str)
sub,c,pt=[],0,[]

for i,p in enumerate(data):
    points=p.split()  
    if points[0] in dd:
        c=c+1
    sub.append(c)
    pt.append([0,1.9*(float(points[1])-0.25),1.9*float(points[2])])
    
points=np.array(pt)
nnode=len(points)
cell=[]

for k in range(nnode-1):
        cell.append([k,k+1])   
elem = np.array(cell)

elem=np.concatenate((elem, np.array([[nnode-1, 0]])))

sub=np.array(sub,dtype=np.int32)

sub[sub == 4] = 2
sub[sub == 5] = 1
sub[sub == 6] = 0


# In[19]:


# Create 1D SG mesh
dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, elem, points, domain)
# Put new subdomains using original cell index 
lnn=[]
o_cell_idx =  dom.topology.original_cell_index
for k in o_cell_idx:
    lnn.append(sub[k])
lnn=np.array(lnn,dtype=np.int32)   


# In[20]:


# Subdomain data (MeshTags)
cell_map = dom.topology.index_map(dom.topology.dim)
num_cells_on_process = cell_map.size_local + cell_map.num_ghosts
cells_l = np.arange(nnode, dtype=np.int32)
subdomains = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells_l, lnn)


# In[21]:


import pyvista
pyvista.start_xvfb()
u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(dom,1)
grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
grid.cell_data["Marker"] = subdomains.values[:]
grid.set_active_scalars("Marker")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid, show_edges=True)
u_plotter.add_point_labels(dom.geometry.x,subdomains.values[:])

u_plotter.view_yz() # x is beam axis
u_plotter.show()


# In[22]:


# Generate ABD matrix (Plate model)
def ABD_mat(ii):
    deg = 2
    cell = ufl.Cell("interval")
    elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th,s=[(0,0,0)],0 # Reference-------- 0
    for k in thick[ii]:
        s=s+k
        th.append((0,0,s) )
    points = np.array(th) 

    # Elements  
    cell=[]
    for k in range(nlay[ii]):
        cell.append([k,k+1])   
    cellss = np.array(cell)
    
    # Create 1D SG mesh
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, points, domain)

    # Subdomain data (MeshTags)
    cell_map = dom.topology.index_map(dom.topology.dim)
    num_cells_on_process = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells_on_process, dtype=np.int32)
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells) # assigning each element as subdomain
    
    dx = Measure('dx')(domain=dom, subdomain_data=subdomain)
    ########################################################
    x = SpatialCoordinate(dom)

    Eps2=as_tensor([(1,0,0,x[2],0,0),
                      (0,1,0,0,x[2],0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,1,0,0,x[2])])  # Gamma_e matrix

    nphases = nlay[ii]
    Eps= Eps2[:,0]

    def eps(v): # (Gamma_h * w)
        E1= as_vector([0,0,v[2].dx(2),(v[1].dx(2)),(v[0].dx(2)),0])
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
            E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[ii][i]]
            S=np.zeros((6,6))
            S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
            S[0,1], S[0,2]= -v12/E1, -v13/E1
            S[1,0], S[1,2]= -v12/E1, -v23/E2
            S[2,0], S[2,1]= -v13/E1, -v23/E2
            S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
            C=np.linalg.inv(S)
            th=angle[ii][i] # ii denotes the layup id
            C= R_sig(C,th) 
            s1= dot(as_tensor(C),eps(v)[1]+Eps)
            return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]), C
        
    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3, )))
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

    # Omega
    omega=dolfinx.fem.assemble_scalar(form(sum([1*dx]))) # volume (length in 1D SG) 
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
            x0=x[2]
            return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
                          (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
                          (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
                          (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
                          (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
                          (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])
    for s in range(6):
        for k in range(6): 
            f=fem.form(sum([Dee(i)[s,k]*dx(i) for i in range(nphases)]))
            D_ee[s,k]=fem.assemble_scalar(f)

    D_eff= D_ee + D1 
    return(D_eff)


# In[23]:


material_parameters=np.array([ (1.000000e+01,  1.000000e+01, 1.000000e+01,1.000000e+00, 1.000000e+00, 1.000000e+00,0.3,0.3,0.3),
  (1.030000e+10,  1.030000e+10,  1.030000e+10,8.000000e+09,8.000000e+09,8.000000e+09,0.3,0.3,0.3),  
(1.030000e+10,  1.030000e+10,  1.030000e+10,8.000000e+09,8.000000e+09,8.000000e+09,0.3,0.3,0.3),
(3.700000e+10,  9.000000e+09, 9.000000e+09, 4.000000e+09,  4.000000e+09, 4.000000e+09,0.28,0.28,0.28),
 (1.000000e+07, 1.000000e+07, 1.000000e+07,  2.000000e+05,2.000000e+05,  2.000000e+05,0.3,0.3,0.3) ])          
           


# In[24]:


thick=np.array([(0.000381,0.00051,0.00053*18),
(0.000381,0.00051,0.00053*33),
(0.000381,0.00051,0.00053*17,0.00053*38,0.003125,0.00053*37,0.00053*16),
(0.000381,0.00051,0.00053*17,0.003125,0.00053*16) ])   


# In[25]:


angle=np.array([(0,0,20),
                (0,0,20),
                (0,0,20,30,0,30,20),
                (0,0,20,0,0)])


# In[26]:


matid=np.array([(0,1,2),(0,1,2),(0,1,2,3,4,3,2),(0,1,2,4,2)])
nphases=max(subdomains.values)+1
nlay=[]
for i in matid:
    nlay.append(len(i))
nphases=len(nlay)


# In[27]:


# Store ABD matrices for layup (as list)
ABD_=[] 
for i in range(nphases):
    ABD_.append(ABD_mat(i))
print('Computed ABD matrix')

def ABD_matrix(i):
    return(as_tensor(ABD_[i]))


# In[28]:


# Local Frame for Boundary (should be same as quad mesh local frame)
def local_frame_1D(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangential direction
    e2=  t1/ sqrt(dot(t1, t1)) 
    e1=  as_vector([-1,0,0])                    # beam-axis direction 
    e3= cross(e1,e2)
    return e1, e2, e3

def deri(e): # derivatives of local frame (Refer: Prof: Yu thesis)
    #a3,1
    e1,e2,e3=e[0],e[1],e[2]
    a1_1=dot(e1,grad(e1))
    a1_2=dot(e2,grad(e1))
    a2_1=dot(e1,grad(e2))
    a2_2=dot(e2,grad(e2))
    a3_1=dot(e1,grad(e3))
    a3_2=dot(e2,grad(e3))
    
    # Initial Curvatures (Shell Element)
    k11= dot(a3_1,e1)
    k12= dot(a3_1,e2)
    k21= dot(a3_2,e1)
    k22= dot(a3_2,e2)
    k13= dot(a1_1,e2)
    k23= dot(a1_2,e2)
    return k11,k12,k21,k22,k13,k23

def local_grad(ee,q):
    return dot(ee,grad(q))

def ddot(w,d1):
    return (d1[0]*w[0]+d1[1]*w[1]+d1[2]*w[2])
    
# Gamma_h*w column matrix
def gamma_h(e,x,w):    
    # e,x required as element can be of left/right boundary or quad mesh
    k11,k12,k21,k22,k13,k23= deri(e) # extracting initial curvatures

    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1=[x11,x21,x31]
    d2=[x12,x22,x32]
    d3=[y1,y2,y3]
    # 
    x111,x211,x311=-k11*y1+k13*x12, -k11*y2+k13*x22, -k11*y3+k13*x32 # e1;1-- [x11,x21,x31];1
    x121,x221,x321=-k12*y1-k13*x11, -k12*y2-k13*x21, -k12*y3-k13*x31 # e2;1
    d11, d21= as_vector([x111,x211,x311]), as_vector([x121,x221,x321])
    
    x112,x212,x312=-k21*y1+k23*x12, -k21*y2+k23*x22, -k21*y3+k23*x32 # e2;1
    x122,x222,x322=-k22*y1-k23*x11, -k22*y2-k23*x21, -k22*y3-k23*x31 # e2;2    
    d12, d22= as_vector([x112,x212,x312]), as_vector([x122,x222,x322])
    # Direction derivative :
    # grad(w1)=[dw1/dx1, dw1/dx2, dw1/dx3 ]
    # w;1=dw/d(e1) = e1.grad(w1)  =  local_grad(e1,w1)

    w_d1=[local_grad(e[0],w[0]),local_grad(e[0],w[1]),local_grad(e[0],w[2])]   # w;1
    w_d2= [local_grad(e[1],w[0]),local_grad(e[1],w[1]),local_grad(e[1],w[2])]  # w;2
    w_d11=[local_grad(e[0],w_d1[0]),local_grad(e[0],w_d1[1]),local_grad(e[0],w_d1[2])] # (w;1);1 -- local grad of w;1 along e1
    w_d22=[local_grad(e[1],w_d2[0]),local_grad(e[1],w_d2[1]),local_grad(e[1],w_d2[2])] # (w;2);2

    w_d12=local_grad(e[1],w_d1[0]), local_grad(e[1],w_d1[1]), local_grad(e[1],w_d1[2]) # (w;1);2
    w_d21=local_grad(e[0],w_d2[0]), local_grad(e[0],w_d2[1]), local_grad(e[0],w_d2[2]) # (w;2);1
    
    w_11=[local_grad(d11,w[0]),local_grad(d11,w[1]),local_grad(d11,w[2])] # w;11 --local grad along e1;1 direction
    w_22=[local_grad(d22,w[0]),local_grad(d22,w[1]),local_grad(d22,w[2])] # w;22
    w_12=[local_grad(d12,w[0]),local_grad(d12,w[1]),local_grad(d12,w[2])] # w;12
    w_21=[local_grad(d21,w[0]),local_grad(d21,w[1]),local_grad(d21,w[2])] # w;21
    
    # Gamma_h*w column matrix 
    G1=ddot(w_d1,d1)
    G2=ddot(w_d2,d2)
    G3=ddot(w_d1,d2)+ddot(w_d2,d1)
    G4=-k11*G1-k12*0.5*G3-ddot(w_d11,d3)+k13*ddot(w_d2,d3)-ddot(w_d11,d3)-ddot(w_11,d3)
    G5=-k22*G2-k21*0.5*G3-ddot(w_d22,d3)-k23*ddot(w_d1,d3)-ddot(w_d22,d3)-ddot(w_22,d3)
    G6=-(k11+k22)*0.5*G3-k12*G2-k21*G1+k23*ddot(w_d2,d3)-k13*ddot(w_d1,d3)-ddot(w_d12,d3)-ddot(w_d21,d3) \
       -ddot(w_12,d3)-ddot(w_21,d3)
    
    E1= as_vector([G1,G2,G3,G4,G5,G6])
    return E1
    
# Gamma_e matrix   
def gamma_e(e,x):
    k11,k12,k21,k22,k13,k23= deri(e)
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1=[x11,x21,x31]
    d2=[x12,x22,x32]
    d3=[y1,y2,y3]
    Rn= x[1]*(x11*x32+x12*x31)-x[2]*(x11*x22+x12*x21)

    E41=-k11*(x11**2-y1**2)-k12*x11*x12
    E51=-k21*x11*x12-k22*(x12**2-y1**2)
    E61=-k11*x11*x12-k12*(x12**2-y1**2)-k22*x11*x12-k21*(x11**2-y1**2)

    E42=k11*(y1*(x[1]*y3-x[2]*y2)+x11*(x[2]*x21-x[1]*x31))+k12*0.5*(-Rn)+2*x11*(y2*x31-y3*x21)
    E52=k22*(y1*(x[1]*y3-x[2]*y2)+x12*(x[2]*x22-x[1]*x32))+k21*0.5*(-Rn)+2*x12*(y2*x32-y3*x22)     

    E62=(k11+k22)*0.5*(-Rn)+(k12+k21)*(y1*(x[1]*y3-x[2]*y2))+k12*x12*(x[2]*x22-x[1]*x32)\
        +2*(y2*(x12*x31+x11*x32)-y3*(x12*x21+x11*x22))+k21*x11*(x[2]*x21-x[1]*x31)

    E43= k11*x[2]*(y1**2-x11**2)-k12*x11*x12*x[2]+x11*(y3*x11-2*y1*x31)
    E53= k22*x[2]*(y1**2-x12**2)-k21*x11*x12*x[2]+x12*(y3*x12-2*y1*x32)
    E63= -(k11+k22)*x[2]*x11*x12+x[2]*(k12*(y1**2-x12**2)+k21*(y1**2-x11**2))-2*y1*(x12*x31+x11*x32)+2*y3*x11*x12

    E44= k11*x[1]*(-y1**2+x11**2)+k12*x11*x12*x[1]+x11*(-y2*x11+2*y1*x21)
    E54= k22*x[1]*(-y1**2+x12**2)+k21*x11*x12*x[1]+x12*(-y2*x12+2*y1*x22)
    E64= (k11+k22)*x11*x12*x[1]+x[1]*(k12(-y1**2+x12**2)+k21*(-y1**2+x11**2))+2*y1*(x12*x21+x11*x22)-2*y2*x11*x12

    return as_tensor([(x11**2, x11*(x[1]*x31-x[2]*x21),  x[2]*x11**2, -x[1]*x11**2),
                    (x12**2, x12*(x[1]*x32-x[2]*x22),  x[2]*x12**2, -x[1]*x12**2),
                    (2*x11*x12,Rn, 2*x11*x12*x[2], -2*x11*x12*x[1]),
                    (E41,E42,E43,E44),
                    (E51,E52,E53,E54), 
                    (E61,E62,E63,E64)])
    


# In[35]:


# Optional arguments
options = ["-O1", "-O2", "-O3", "-Ofast"]
cffi_options = [options[3]]
jit_options = {"cffi_extra_compile_args": cffi_options,
                                "cffi_libraries": ["m"]}

deg=2
#-----------------Finite Element function Space-------------------------------------------
# Local Frame for mesh 
V = dolfinx.fem.functionspace(dom, basix.ufl.element(
    "S", dom.topology.cell_name(), deg, shape=(3, )))
frame = local_frame_1D(dom) 
ve1, ve2, ve3= frame
e1,e2,e3=Function(V), Function(V), Function(V)

fexpr1=dolfinx.fem.Expression(ve1,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1.interpolate(fexpr1)  # ve1 replace by EE1

fexpr2=dolfinx.fem.Expression(ve2,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2.interpolate(fexpr2)  # ve2 replace by EE1

fexpr3=dolfinx.fem.Expression(ve3,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3.interpolate(fexpr3) # ve3 replace by EE1
e=[e1,e2,e3]
dv = TrialFunction(V)
v_ = TestFunction(V)
x, dx=SpatialCoordinate(dom), Measure('dx')(domain=dom, subdomain_data=subdomains)


# In[36]:


# Left Nullspace
def nullspace(V):
    index_map = V.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(4)]
    
    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]
    
    # Dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list for i in range(3)]
    
    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0
    
    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)
    
    return petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
    
nullspace=nullspace(V)
F2 = sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
a2=form(F2)
A=assemble_matrix(a2)
A.assemble()
A.setNullSpace(nullspace)

xxx= 3*V.dofmap.index_map.local_range[1]
V0=np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
D_ee=np.zeros((4,4))


# In[37]:


for p in range(4):
    Eps=gamma_e(e,x)[:,p]
    F2= sum([dot(dot(ABD_matrix(i),Eps+gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
    r_he=form(rhs(F2))
    F_l = petsc.assemble_vector(r_he)
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    w_l=Function(V)
    
    nullspace.remove(F_l) # nullspace_l
    # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A) # A
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
    V0[:,p]=w_l.vector[:]
    Dhe[:,p]=F_l[:]
D1=np.matmul(V0.T,-Dhe) 

for s in range(4):
    for k in range(4): 
        f=sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)])
        D_ee[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))
D_eff= D_ee + D1

#-------------------------Printing Output Data-------------------------------
print('  ')  
print('EB Stiffness Matrix \n')
np.set_printoptions(precision=4)
print(D_eff) 

