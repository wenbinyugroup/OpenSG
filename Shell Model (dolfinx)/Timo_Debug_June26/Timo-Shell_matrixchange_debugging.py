### Update for dolfinx latest v0.8
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
mesh, subdomains, boundaries = gmshio.read_from_msh("cylshell_1152elem.msh", MPI.COMM_WORLD,0, gdim=3)

pp=mesh.geometry.x
x_min,x_max=min(pp[:,0]), max(pp[:,0])

def left(x):
    return np.isclose(x[0], x_min)
def right(x):
    return np.isclose(x[0], x_max)

tdim=mesh.topology.dim
fdim = tdim - 1
facets_left = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim,
                                       marker=left)
facets_right = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim,
                                       marker=right)

mesh_r, entity_mapr, vertex_mapr, geom_mapr = create_submesh(mesh, fdim, facets_right)
mesh_l, entity_mapl, vertex_mapl, geom_mapl = create_submesh(mesh, fdim, facets_left)

import pyvista
pyvista.start_xvfb()
u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(mesh_l,fdim)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_yz()
#u_plotter.show()
#########################################################################################

ABD=as_tensor([[ 7.56043956e+08,  2.26813187e+08,  0.00000000e+00, 9.31322575e-10,  9.31322575e-10,  0.00000000e+00],
              [ 2.26813187e+08,  7.56043956e+08,  0.00000000e+00, 9.31322575e-10,  9.31322575e-10,  0.00000000e+00],
             [ 0.00000000e+00,  0.00000000e+00,  2.64600000e+08, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.39698386e-09, -1.39698386e-09,  0.00000000e+00, 2.52014652e+06,  7.56043956e+05,  0.00000000e+00],
            [-1.39698386e-09, -1.39698386e-09,  0.00000000e+00, 7.56043956e+05,  2.52014652e+06,  0.00000000e+00],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  8.82000000e+05]])

def local_frame_1D(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
    e1=  t1/ sqrt(dot(t1, t1))
    e2=  as_vector([1,0,0])
    e3= cross(e1,e2)
    return e1, e2, e3

def deri(e):
    #a3,1
    e1,e2,e3=e[0],e[1],e[2]
    a1_1=dot(e1,grad(e1))
    a1_2=dot(e2,grad(e1))
    a2_1=dot(e1,grad(e2))
    a2_2=dot(e2,grad(e2))
    a3_1=dot(e1,grad(e3))
    a3_2=dot(e2,grad(e3))
    
    # Initial Curvatures
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


deg=2
# Left Boundary

V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
    "S", mesh_l.topology.cell_name(), deg, shape=(3, )))
w_l = Function(V_l)

frame_l = local_frame_1D(mesh_l)
le1, le2, le3= frame_l

e1l,e2l,e3l=Function(V_l), Function(V_l), Function(V_l)

fexpr1=dolfinx.fem.Expression(le1,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1l.interpolate(fexpr1) 

fexpr2=dolfinx.fem.Expression(le2,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2l.interpolate(fexpr2) 

fexpr3=dolfinx.fem.Expression(le3,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3l.interpolate(fexpr3) 

x, dx=SpatialCoordinate(mesh_l), Measure('dx')(domain=mesh_l)

e=[e1l,e2l,e3l]

dv = TrialFunction(V_l)
v_= TestFunction(V_l)
local_range = V_l.dofmap.index_map.local_range
dofss = np.arange(*local_range)
xxx=3*len(dofss)

V0 = np.zeros((xxx,4))
Dle=np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
Dhd=np.zeros((xxx,4))
Dld=np.zeros((xxx,4))
V0=np.zeros((xxx,4))
D_ed=np.zeros((4,4))
D_dd=np.zeros((4,4))
D_ee=np.zeros((4,4))  
V1s=np.zeros((xxx,4))
mesh_l.topology.create_connectivity(1, 1)


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

Eps2= as_tensor([(x11**2, x11*(x[1]*x31-x[2]*x21),  x[2]*x11**2, -x[1]*x11**2),
                        (x12**2, x12*(x[1]*x32-x[2]*x22),  x[2]*x12**2, -x[1]*x12**2),
                        (2*x11*x12,Rn, 2*x11*x12*x[2], -2*x11*x12*x[1]),
                        (E41,E42,E43,E44),
                        (E51,E52,E53,E54), 
                        (E61,E62,E63,E64)])

O=as_vector((0,0,0,0))
R=as_vector((-y1,-y3*x[1]+y2*x[2],-x[2]*y1,x[1]*y1))
Eps2d= as_tensor([O,O,O,x11*x11*R,x12*x12*R,2*x11*x12*R])

def eps2(e,x,dx,w):
    
        k11,k12,k21,k22,k13,k23= deri(e)
        x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
        x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
        y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
        d1=[x11,x21,x31]
        d2=[x12,x22,x32]
        d3=[y1,y2,y3]
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
        G4=-k11*G1-k12*0.5*G3-ddot(w_d11,d3)+k13*ddot(w_d2,d3)-ddot(w_d11,d3)-ddot(w_11,d3)
        G5=-k22*G2-k21*0.5*G3-ddot(w_d22,d3)-k23*ddot(w_d1,d3)-ddot(w_d22,d3)-ddot(w_22,d3)
        G6=-(k11+k22)*0.5*G3-k12*G2-k21*G1+k23*ddot(w_d2,d3)-k13*ddot(w_d1,d3) \
           -ddot(w_d12,d3)-ddot(w_d21,d3)-ddot(w_12,d3)-ddot(w_21,d3)

        E1= as_vector([G1,G2,G3,0.5*G4,0.5*G5,0.5*G6])
        return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1
   # E1=as_vector([0,v[1].dx(0),v[2].dx(1),0.5*(v[1].dx(1)+v[2].dx(0)),0.5*(v[0].dx(1)),0.5*(v[0].dx(0))])
    #return E1
# gamma_h--> eps2[1]
def sigma(e,x,dx,v, i,Eps):
    s1= dot(ABD, eps2(e,x,dx,v)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])])

def gamma_l(e,x,dx,w):
        k11,k12,k21,k22,k13,k23= deri(e)
        x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
        x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
        y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
        d1=[x11,x21,x31]
        d2=[x12,x22,x32]
        d3=[y1,y2,y3]
        dd1,dd2=as_vector(d1), as_vector(d2)
        w_d1=[local_grad(dd1,w[0]),local_grad(dd1,w[1]),local_grad(dd1,w[2])]
        w_d2= [local_grad(dd2,w[0]),local_grad(dd2,w[1]),local_grad(dd2,w[2])]

        L1,L2=x11*(ddot(d1,w)),x12*(ddot(d2,w))
        L3=ddot([2*x11*x12,(x12*x21+x11*x22),(x12*x31+x11*x32)],w)
        L4=-2*x11*ddot(w_d1,d3)

        L5=ddot([(k22*(y1**2-x12**2)),
                  (k22*(y1*y2-x12*x22)),
                  (k22*(y1*y3-x12*x32))],w)-2*x12*ddot(w_d2,d3)
        L6=ddot([(-k22*x11*x12),
                 (-0.5*k22*(x21*x12+x11*x22)),
                 (-0.5*k22*(x31*x12+x11*x32))],w)-2*(x11*ddot(w_d2,d3)+x12*ddot(w_d1,d3))
                
        E1=  as_vector([L1,L2,L3,0.5*L4,0.5*L5,0.5*L6])
        return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1
 #   E1= as_vector([v[0],0,0,0, v[2],v[1]])
   # return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def sigma_gl(e,x,dx,w, i, Eps):
    s1= dot(ABD,gamma_l(e,x,dx,w)[1]+Eps)
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])])

Eps=Eps2[:,0]
F1=sum([inner(sigma_gl(e,x,dx,dv, 0, Eps),gamma_l(e,x,dx,v_)[0])*dx])
a1=lhs(F1)
Dll=assemble_matrix(form(a1))
Dll.assemble()
ai, aj, av=Dll.getValuesCSR()
Dll=csr_matrix((av, aj, ai)).toarray()
for p in range(4):
    
        Eps=Eps2[:,p] 
        F1=sum([inner(sigma_gl(e,x,dx,dv,0,Eps),gamma_l(e,x,dx,v_)[0])*dx])
        L1 = rhs(F1)          
        Dle[:,p]= petsc.assemble_vector(form(L1))[:]
    
        Eps=Eps2d[:,p] 
        F1=sum([inner(sigma_gl(e,x,dx,dv,0,Eps),eps2(e,x,dx,v_)[0])*dx])
        L1 = rhs(F1)          
        Dhd[:,p]= petsc.assemble_vector(form(L1))[:]
    
        Eps=Eps2d[:,p] 
        F1=sum([inner(sigma_gl(e,x,dx,dv,0,Eps),gamma_l(e,x,dx,v_)[0])*dx])
        L1 = rhs(F1)          
        Dld[:,p]= petsc.assemble_vector(form(L1))[:]

F_dhl=sum([inner(sigma(e,x,dx,dv,0,Eps),gamma_l(e,x,dx,v_)[0])*dx]) 
a3=lhs(F_dhl)
Dhl=assemble_matrix(form(a3))
Dhl.assemble()
ai, aj, av=Dhl.getValuesCSR()
Dhl=csr_matrix((av, aj, ai)).toarray()
    
F2 = sum([inner(sigma(e,x,dx,dv, 0, Eps), eps2(e,x,dx,v_)[0])*dx])   
a2=lhs(F2)
A_l=assemble_matrix(form(a2))
A_l.assemble()
for p in range(4):
    Eps=Eps2[:,p]
    F2 = sum([inner(sigma(e,x,dx,dv, 0, Eps), eps2(e,x,dx,v_)[0])*dx])   
    r_he=rhs(F2)
    F_l = petsc.assemble_vector(form(r_he))
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)

    w_l=Function(V_l)
    # Nullspace implement

    index_map = V_l.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V_l.dofmap.index_map_bs) for i in range(4)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]

    # Dof indices for each subspace (x, y and z dofs)
    dofs = [V_l.sub(i).dofmap.list for i in range(3)]

    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0
    # Build rotational null space basis
   # xx = V_l.tabulate_dof_coordinates()
   # dofs_block = V_l.dofmap.list
  #  x1, x2, x3 = xx[dofs_block, 0], xx[dofs_block, 1], xx[dofs_block, 2]
  #  basis[3][dofs[1]] = -x3
   # basis[3][dofs[2]] = x2

    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)

    nullspace_l = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)

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
    print('Computed','Dhe',(p+1)*25,'%')
    
D1=np.matmul(V0.T,-Dhe)   
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([dot(dot(Eps2.T,ABD),Eps2)[s,k]*dx]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

        f=dolfinx.fem.form(sum([dot(dot(Eps2.T,ABD),Eps2d)[s,k]*dx]))
        D_ed[s,k]=dolfinx.fem.assemble_scalar(f)

        f=dolfinx.fem.form(sum([dot(dot(Eps2d.T,ABD),Eps2)[s,k]*dx]))
        D_dd[s,k]=dolfinx.fem.assemble_scalar(f)  
    print('Computed',(s+1)*25,'%')
    
D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)
D_eff=D_eff
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(D_eff))  


#DhlTV0
DhlV0=np.matmul(Dhl,V0)

#DhlTV0Dle
DhlTV0Dle=np.matmul(Dhl.T,V0)+Dle

#V0DllV0
V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)

# V1s
b=DhlTV0Dle-DhlV0-Dhd
for p in range(4):
    w_l=Function(V_l)
    F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_l.remove(F)
    
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
# Ainv
Ainv=np.linalg.inv(D_eff)

# B_tim
B_tim= np.matmul(DhlTV0Dle.T,V0)+ D_ed+ np.matmul(V0.T,Dhd)

# C_tim
C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)+2*np.matmul(V0.T,Dld)+ D_dd+np.matmul(V1s.T,Dhd)
C_tim=0.5*(C_tim+C_tim.T)

# D_tim
D_tim= np.matmul((np.matmul(Dhl.T,V0)+Dle).T,V1s)

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
