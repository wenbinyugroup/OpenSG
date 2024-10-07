## Timoshenko Beam model for shell elements (1D curved elements)

# Initialization of libraries
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
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
mesh_l, subdomains, boundaries = gmshio.read_from_msh("1D_R413_4layup_tt.msh", MPI.COMM_WORLD,0, gdim=3)


# For total 368 nodes of 1D circle, assigning each quadrant in clockwise manner a layup id.
# This process is for this specific example (In general we would be getting it from yaml data)
dd=[92,184,277]
c,sub=0,[]
for i in range(367):
    if i in dd:
        c=c+1
    sub.append(c)
lnn=np.array(sub,dtype=np.int32)  
num_cells= mesh_l.topology.index_map(mesh_l.topology.dim).size_local 
cells_l = np.arange(num_cells, dtype=np.int32)
subdomains = dolfinx.mesh.meshtags(mesh_l, mesh_l.topology.dim, cells_l, lnn)

# solving linear system of equations: A-coeff matrix; F force vector;w_l unknown vector; Solving A.w_l = F
def ksp_solve(A,F,V):
    w = Function(V)
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
    return w
    
# Assigning nullspace to restrict Rigid body motions
def nullspace(V):
    index_map = V.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(3)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]
    dofs = [V.sub(i).dofmap.list for i in range(3)]      # Dof indices for each subspace (x, y and z dofs)
    for i in range(3):                                   # Build translational null space basis
        basis[i][dofs[i]] = 1.0
    dolfinx.la.orthonormalize(nullspace_basis)           # Create vector space basis and orthogonalize
    return petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)


# Generate ABD matrix (Plate Stiffness matrix) - 6X6 matrix used as an input material information for shell elements (for each layup)
def ABD_mat(ii):
    # 1D mesh generation  
    deg = 2
    cell = ufl.Cell("interval")
    elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th,s=[0],0 # Reference-------- 0
    for k in thick[ii]:
        s=s+k           # Outward normal
        th.append(s)
    points = np.array(th) 
    # Elements  
    cell=[]
    for k in range(nlay[ii]):
        cell.append([k,k+1])   
    cellss = np.array(cell)
    
    # Create 1D SG mesh
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, points, domain) # mesh generated 
    num_cells= dom.topology.index_map(dom.topology.dim).size_local 
    cells = np.arange(num_cells, dtype=np.int32)
    #
    # Note: For a layup like layup1: containing 3 layers:
    # we defined 1D mesh with 3 elements and make each element a seperate subdomain(where integration is computed by *dx(i) 
    #for subdomain i) using cells-[0,1,2] 
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells) # assigning each element as subdomain
    x,dx = SpatialCoordinate(dom),Measure('dx')(domain=dom, subdomain_data=subdomain) 
    gamma_e=as_tensor([(1,0,0,x[0],0,0),       # Gamma_e matrix
                      (0,1,0,0,x[0],0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,1,0,0,x[0])])  

    nphases = len(cells)
    def eps(v): # (Gamma_h * w)
        E1= as_vector([0,0,v[2].dx(0),(v[1].dx(0)),(v[0].dx(0)),0])
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
            theta=angle[ii][i] # ii denotes the layup id
            C= R_sig(C,theta) 
            s1= dot(as_tensor(C),eps(v)[1]+Eps)
            return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]), C
        
    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3, )))
    u,v = TrialFunction(V), TestFunction(V)
    F2 = sum([inner(sigma(u, i, gamma_e[:,0])[0], eps(v)[0])*dx(i) for i in range(nphases)]) # Weak form of energy;*dx is for integartion
    A=  petsc.assemble_matrix(form(lhs(F2)))  # lhs gives left hand side of weak form : coeff matrix here
    A.assemble()
    null = nullspace(V)
    A.setNullSpace(null)      # Set the nullspace
    xx = 3*V.dofmap.index_map.local_range[1] # total dofs
    # Initialization
    V0, Dhe, D_ee = np.zeros((xx,6)), np.zeros((xx,6)), np.zeros((6,6))

    # Assembly
    for p in range(6):
        Eps=gamma_e[:,p] 
        F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0])*dx(i) for i in range(nphases)]) # weak form
        F = petsc.assemble_vector(form(rhs(F2)))  # rhs is used for getting right hand side of A w = F; (which is F here)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(F)                        # Orthogonalize F to the null space of A^T
        w=ksp_solve(A,F,V)
        Dhe[:,p]= F[:]                        # Dhe matrix formation
        V0[:,p]= w.vector[:]                  # V0 matrix formation
    D1=np.matmul(V0.T,-Dhe)                   # Additional information matrix
    
    def Dee(i):                         # Simplifying gamma_e.T*C*gamma_e 
            C=sigma(u, i, Eps)[1]
            x0=x[0]
            return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
                          (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
                          (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
                          (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
                          (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
                         (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])
    
    for s in range(6): # Scalar assembly for each term of D_ee matrix 
        for k in range(6): 
            f=dolfinx.fem.form(sum([Dee(i)[s,k]*dx(i) for i in range(nphases)])) # Scalar assembly
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

    D_eff= D_ee + D1 
    return(D_eff) 
###################################
 
material_parameters=np.array([(3.700000e+10,  9.000000e+09, 9.000000e+09, 4.000000e+09, 4.000000e+09, 4.000000e+09,0.28,0.28,0.28), 
(4.500000e+10,  6.000000e+09, 6.000000e+09, 3.000000e+09, 3.000000e+09, 3.000000e+09,0.30,0.30,0.30),
(1.03000e+10,  3.000000e+09, 3.000000e+09, 1.000000e+09, 1.000000e+09, 1.000000e+09,0.30,0.30,0.30) ])   

# Defining layup information (taken from yaml file)
# Information for 4 layers used:
thick=np.array([(0.2*0.00053,0.2*0.0004,0.2*0.00035*18),
(0.2*0.00053,0.2*0.00035,0.2*0.0004*17,0.2*0.00035*17,0.2*0.00053),
(0.2*0.00053,0.2*0.00035,0.2*0.00035*17,0.2*0.00053*17,0.2*0.0004),
(0.2*0.00053,0.2*0.0004,0.2*0.00035*17,0.2*0.00053,0.2*0.00053*16)])   

angle=np.array([(30,70,-45),
                (-20,0,30,40,50),
                (0,0,30,-40,10),
                (10,20,-30,20,-10)]) 
                
matid=np.array([(0,2,1),(0,1,2,1,0),(0,1,1,0,2),(0,2,1,0,0)]) # material id


nphases=max(subdomains.values)+1
nlay=[]
for i in angle:
    nlay.append(len(i))
print('nphases',nphases)    

# Store ABD matrices for layup (as list)
ABD_=[] 
for i in range(nphases):
    ABD_.append(ABD_mat(i))
print('Computed ABD matrix')

def ABD_matrix(i):
    return(as_tensor(ABD_[i])) 

########################################################
# The above part was only initial input seperate from timoshenko beam code. 
# Seperate ABD matrix can be given as input instead of solving within this code (or called as seperate function)

######################################## Finite Element Implemenatation#############################
# Defining functions for Part 1: Euler-Bernoulli stiffness matrix

def local_frame_1D(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # [x2-x1,y2-y1,z2-z1] tangent vector for each element. 
    # Direction of tangenet vector about axis (1,0,0) depends on how we defined the mesh elements. 
    # For given example, final node-initial node --> clockwise direction about (1,0,0) axis.

    e2=  t1/ sqrt(dot(t1, t1))  # converting to unit vector
    e1=  as_vector([1,0,0]) # Beam axis in x direction (global)
    e3= cross(e1,e2)    # normal vector
    e3=  e3/ sqrt(dot(e3, e3))   
    return e1, e2, e3

# Obtaining the curvature of curved elements  
def deri(e): # Input local element frame 
    #a3,1
    e1,e2,e3=e[0],e[1],e[2]
    e1_1=dot(e1,grad(e1))     # Directional derivative of e1 w.r.t. e2  --> d(e1)/d(e2) 
    e1_2=dot(e2,grad(e1))     # grad (.)- is 3X1 vector --> [d(.)/dx ,d(.)/dy,d(.)/dz]                                           
    e2_1=dot(e1,grad(e2))     # e2_1 shows derivative of e2 vector w.r.t. e1 vector
    e2_2=dot(e2,grad(e2))
    e3_1=dot(e1,grad(e3))
    e3_2=dot(e2,grad(e3))
    
    # Initial Curvatures     # 
    k11= dot(e3_1,e1)
    k12= dot(e3_1,e2)
    k21= dot(e3_2,e1)
    k22= dot(e3_2,e2)
    k13= dot(e1_1,e2)
    k23= dot(e1_2,e2)
    return k11,k12,k21,k22,k13,k23

def local_grad(ee,q):
    return dot(ee,grad(q))
    
def ddot(w,d1):
    return (d1[0]*w[0]+d1[1]*w[1]+d1[2]*w[2])

# Defining FE function space for Serendipidity element

V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
    "S", mesh_l.topology.cell_name(), 2, shape=(3, ))) 
    
# shape is dof for each node; # deg 2 has 3 nodes per element
frame_l = local_frame_1D(mesh_l)    
le1, le2, le3= frame_l

# We need to interpolate the ufl-based local_frame(le1,le2,le3) to Finite Element Space(Serendipidity Element) 
e1l,e2l,e3l=Function(V_l), Function(V_l), Function(V_l)

fexpr1=dolfinx.fem.Expression(le1,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1l.interpolate(fexpr1) 

fexpr2=dolfinx.fem.Expression(le2,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2l.interpolate(fexpr2) 

fexpr3=dolfinx.fem.Expression(le3,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3l.interpolate(fexpr3) 

# Whenever we define Function(V_l), we can view the data using: print(e1l.vector[:]) ; arranged based on dof numbering 

x, dx=SpatialCoordinate(mesh_l), Measure('dx')(domain=mesh_l, subdomain_data=subdomains) 
e=[e1l,e2l,e3l]   

dv, v_= TrialFunction(V_l), TestFunction(V_l)
xxx=3*len(np.arange(*V_l.dofmap.index_map.local_range))  # total dofs for Functionspace V_l

# Initialize np matrices
V0 = np.zeros((xxx,4))
D_ee=np.zeros((4,4))  
Dhe = np.zeros((xxx,4))
mesh_l.topology.create_connectivity(1, 1)

# Gamma_e matrix (6x4) 
k11,k12,k21,k22,k13,k23= deri(e)
x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])      # Cab rotation matrix (xyz frame to local element)
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

gamma_e= as_tensor([(x11**2, x11*(x[1]*x31-x[2]*x21),  x[2]*x11**2, -x[1]*x11**2),
                        (x12**2, x12*(x[1]*x32-x[2]*x22),  x[2]*x12**2, -x[1]*x12**2),
                        (2*x11*x12,Rn, 2*x11*x12*x[2], -2*x11*x12*x[1]),
                        (E41,E42,E43,E44),
                        (E51,E52,E53,E54), 
                        (E61,E62,E63,E64)])

# (gamma_h.w) vector (6X1)

def gamma_h(e,x,w):
        d11=as_vector([-k11*d3[ii]+k13*d2[ii] for ii in range(3)])
        d22=as_vector([-k22*d3[ii]-k23*d1[ii] for ii in range(3)])
        d12=as_vector([-k21*d3[ii]+k23*d2[ii] for ii in range(3)])
        d21=as_vector([-k12*d3[ii]-k13*d1[ii] for ii in range(3)])
        
        w_d1=[local_grad(e[0],w[i]) for i in range(3)]
        w_d2= [local_grad(e[1],w[i]) for i in range(3)]
        w_d11=[local_grad(e[0],w_d1[i]) for i in range(3)]
        w_d22=[local_grad(e[1],w_d2[i]) for i in range(3)]

        w_d12=[local_grad(e[1],w_d1[ii]) for ii in range(3)]
        w_d21=[local_grad(e[0],w_d2[ii]) for ii in range(3)]
        w_11=[local_grad(d11,w[ii]) for ii in range(3)]
        w_22=[local_grad(d22,w[ii]) for ii in range(3)]
        w_12=[local_grad(d12,w[ii]) for ii in range(3)]
        w_21=[local_grad(d21,w[ii]) for ii in range(3)]

        G1=ddot(w_d1,d1)
        G2=ddot(w_d2,d2)
        G3=ddot(w_d1,d2)+ddot(w_d2,d1)
        G4=-k11*G1-k12*0.5*G3-ddot(w_d11,d3)+k13*ddot(w_d2,d3)
        G5=-k22*G2-k21*0.5*G3-ddot(w_d22,d3)-k23*ddot(w_d1,d3)
        G6=-(k11+k22)*0.5*G3-k12*G2-k21*G1+k23*ddot(w_d2,d3)-k13*ddot(w_d1,d3)-ddot(w_d12,d3)-ddot(w_d21,d3)

        E1= as_tensor([G1,G2,G3,G4,G5,G6])
        return E1
        
# Weak form for energy formulation
F2 = sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
a2=form(F2)
A_l=assemble_matrix(a2)
A_l.assemble() # coefficient matrix generation
nullspace_l = nullspace(V_l)  
A_l.setNullSpace(nullspace_l)     # Assigning nullspace to remove rigid body motions

# Solving 4 weak forms to obtain Euler-Bernoulli matrix (Intermediate step) of dim- 4X4
# The below loop gives solution of unknown fluctuating functions (zeroth order approximation): V0

# Mathematically, it solve the Euler-Lagrange equations generated after minimation (analytically)
for p in range(4):
    Eps=gamma_e[:,p]
    F2 = sum([dot(dot(ABD_matrix(i),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
    r_he=form(rhs(F2))
    F_l = petsc.assemble_vector(r_he)
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_l.remove(F_l)
    w_l=ksp_solve(A_l,F_l,V_l)
    Dhe[:,p]=  F_l[:]
    V0[:,p]= w_l.vector[:]  
    
D1=np.matmul(V0.T,-Dhe)   # this 4X4 matrix is also called additional information matrix

# Solving for 4X4 matrix from gamma_e and ABD matrix by scalar assembly
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([dot(dot(gamma_e.T,ABD_matrix(i)),gamma_e)[s,k]*dx(i) for i in range(nphases)]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
  
D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)

print('\n EB Stiffness Matrix \n')
np.set_printoptions(precision=4)
print(np.around(D_eff))  


# Part 2: Timoshenko Stiffness matrix
dd1,dd2,dd3=as_vector(d1), as_vector(d2), as_vector(d3)

# (Gamma_l*w') - 6X1 vector
def gamma_l(e,x,w):
        w_d1=as_vector([local_grad(dd1,w[ii]) for ii in range(3)])
        w_d2= as_vector([local_grad(dd2,w[ii]) for ii in range(3)])

        L1,L2=x11*(ddot(d1,w)),x12*(ddot(d2,w))
        L3=ddot(x11*dd2+x12*dd1,w)
        L4=-2*x11*ddot(w_d1,d3)+ddot(k11*(y1*dd3-x11*dd1)-0.5*k12*(x12*dd1+x11*dd2),w)
        L5=-2*x12*ddot(w_d2,d3)+ddot(k22*(y1*dd3-x12*dd2)-0.5*k21*(x12*dd1+x11*dd2),w)
        L6=-2*ddot(x11*w_d2+x12*w_d1,d3)+ddot(k12*(y1*dd3-x12*dd2)+k21*(y1*dd3-x11*dd1)-0.5*(k11+k22)*(x12*dd1+x11*dd2),w)
        E1=  as_tensor([L1,L2,L3,L4,L5,L6])
        return E1

# Defining gamma_d matrix (6X4)     
O=as_vector((0,0,0,0))
R=as_vector((-y1,-y3*x[1]+y2*x[2],-x[2]*y1,x[1]*y1))
gamma_d= as_tensor([O,O,O,x11*x11*R,x12*x12*R,2*x11*x12*R])

# Initialize
Dll=np.zeros((xxx,xxx))
Dle=np.zeros((xxx,4))
Dhd=np.zeros((xxx,4))
Dld=np.zeros((xxx,4))
D_ed=np.zeros((4,4))
D_dd=np.zeros((4,4)) 
V1s=np.zeros((xxx,4))
######### Obtaining intermediate matrices after global assembly

F1=sum([dot(dot(ABD_matrix(i),gamma_l(e,x,v_)),gamma_l(e,x,dv))*dx(i) for i in range(nphases)]) # Weak form 
a1=form(F1)
Dll=assemble_matrix(a1)
Dll.assemble()
ai, aj, av=Dll.getValuesCSR()
Dll=csr_matrix((av, aj, ai)).toarray()  # Converting csr matrix to np array

for p in range(4):
        Eps=gamma_e[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
        Dle[:,p]= petsc.assemble_vector(form(F1))[:]    
    
        Eps=gamma_d[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_h(e,x,v_))*dx(i) for i in range(nphases)])      
        Dhd[:,p]= petsc.assemble_vector(form(F1))[:]
    
        Eps=gamma_d[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])      
        Dld[:,p]= petsc.assemble_vector(form(F1))[:]
    
F_dhl=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
a3=form(F_dhl)
Dhl=assemble_matrix(a3)
Dhl.assemble()
ai, aj, av=Dhl.getValuesCSR()
Dhl=csr_matrix((av, aj, ai)).toarray()     # Converting csr matrix to np array

# Solving 4X4 matrix after scalar assembly
for s in range(4):
    for k in range(4):    
        f=dolfinx.fem.form(sum([dot(dot(gamma_e.T,ABD_matrix(i)),gamma_d)[s,k]*dx(i) for i in range(nphases)]))
        D_ed[s,k]=dolfinx.fem.assemble_scalar(f)

        f=dolfinx.fem.form(sum([dot(dot(gamma_d.T,ABD_matrix(i)),gamma_d)[s,k]*dx(i) for i in range(nphases)]))
        D_dd[s,k]=dolfinx.fem.assemble_scalar(f) 
    
#DhlTV0
DhlV0=np.matmul(Dhl.T,V0)

#DhlTV0Dle
DhlTV0Dle=np.matmul(Dhl,V0)+Dle

#V0DllV0
V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)

# V1s
b=DhlTV0Dle-DhlV0-Dhd

# Solving unknown fluctuating functions (first order approximation) V1s
for p in range(4):
    F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_l.remove(F)
    w_l=ksp_solve(A_l,F,V_l)
    V1s[:,p]= w_l.vector[:]  
    
# Ainv
Ainv=np.linalg.inv(D_eff)

# B_tim
B_tim= np.matmul(DhlTV0Dle.T,V0)+ D_ed+ np.matmul(V0.T,Dhd)

# C_tim
C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)+2*np.matmul(V0.T,Dld)+ D_dd+np.matmul(V1s.T,Dhd)
C_tim=0.5*(C_tim+C_tim.T)

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

print('\n Timoshenko Stiffness Matrix \n')
np.set_printoptions(precision=4)
print(np.around(Deff_srt)) 
