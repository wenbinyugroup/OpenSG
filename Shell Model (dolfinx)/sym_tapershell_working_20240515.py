from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
from mpi4py import MPI
import numpy as np
import meshio
import dolfinx
from dolfinx.fem import VectorFunctionSpace, form, petsc, Function, FunctionSpace, locate_dofs_topological, apply_lifting, set_bc
from ufl import Jacobian, as_vector, dot, cross,sqrt, conditional, replace, as_matrix,FacetNormal
from ufl import lt,SpatialCoordinate, as_tensor, VectorElement, FiniteElement, MixedElement, Measure
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, dx, dot, eq, grad
import petsc4py.PETSc
from contextlib import ExitStack
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from typing import Dict
import scipy
from scipy.sparse import csr_matrix
mesh, subdomains, boundaries = gmshio.read_from_msh("bar_urc_shell_xaxis.msh", MPI.COMM_WORLD,0, gdim=3)

#cylshell_1152elem               3Dshell_tapered      bar_urc_shell_xaxis
#mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1,1, dolfinx.mesh.CellType.quadrilateral)
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
u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(mesh,tdim)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_yz()
u_plotter.show()
#########################################################################################

def local_frame(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
    t2 = as_vector([t[0, 1], t[1, 1], t[2, 1]])
    e3 = cross(t1, t2)
    e3 /= sqrt(dot(e3, e3))
    e1=  t2/ sqrt(dot(t2, t2))
    e2 = cross(e3, e1)
    e2 /= sqrt(dot(e2, e2))
    return e1, e2, e3

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

# 4 layer outer reference--
# Obtained for E= 3.44e+03, v= 0.3 (Isotropic) using 1D SG isotropic under plate model 
# Can also be evaulated with (h=1)
# A = E h/ (1-v^2)   [ 1    v   0;  v  1  0; 0   0  (1-v)/2]; B=0;
# D=  A (h^2/12);

ABD=as_tensor([( 3.78022e+03,  1.13407e+03,  0.00000e+00, 0, 0,  0.00000e+00   ),
 ( 1.13407e+03,  3.78022e+03,  0.00000e+00, 0, 0,  0.00000e+00   ),
 ( 0.00000e+00,  0.00000e+00,  1.32308e+03,  0.00000e+00,  0.00000e+00, 0),
 (0, 0,  0.00000e+00,  3.15018e+02,  9.45055e+01,  0.00000e+00 ),
 (0, 0,  0.00000e+00,  9.45055e+01,  3.15018e+02,  0.00000e+00 ),
 (0.00000e+00,  0.00000e+00, 0,  0.00000e+00,  0.00000e+00,  1.10256e+02)])

def local_grad(ee,q):
    return dot(ee,grad(q))

def ddot(w,d1):
    return (d1[0]*w[0]+d1[1]*w[1]+d1[2]*w[2])

def sigma(v,i,Eps,e,x):     
    s1= dot(ABD,eps(e,x,v)+Eps)
    return s1 

def eps(e,x,w):    
    k11,k12,k21,k22,k13,k23= deri(e)
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1=[x11,x21,x31]
    d2=[x12,x22,x32]
    d3=[y1,y2,y3]
    
    w_d1=[local_grad(e[0],w[0]),local_grad(e[0],w[1]),local_grad(e[0],w[2])]
    w_d2= [local_grad(e[1],w[0]),local_grad(e[1],w[1]),local_grad(e[1],w[2])]
    w_d11=[local_grad(e[0],w_d1[0]),local_grad(e[0],w_d1[1]),local_grad(e[0],w_d1[2])]
    w_d22=[local_grad(e[1],w_d2[0]),local_grad(e[1],w_d2[1]),local_grad(e[1],w_d2[2])]

    w_d12=local_grad(e[1],w_d1[0]), local_grad(e[1],w_d1[1]), local_grad(e[1],w_d1[2])
    w_d21=local_grad(e[0],w_d2[0]), local_grad(e[0],w_d2[1]), local_grad(e[0],w_d2[2])
    
    G1=ddot(w_d1,d1)
    G2=ddot(w_d2,d2)
    G3=ddot(w_d1,d2)+ddot(w_d2,d1)
    G4=-k11*G1-k12*0.5*G3+k13*ddot(w_d2,d3)-ddot(w_d11,d3)
    G5=-k22*G2-k21*0.5*G3-k23*ddot(w_d1,d3)-ddot(w_d22,d3)
    G6=-(k11+k22)*0.5*G3-k12*G2-k21*G1+k23*ddot(w_d2,d3)-k13*ddot(w_d1,d3)-ddot(w_d12,d3)-ddot(w_d21,d3)
    
    E1= as_vector([G1,G2,G3,G4,G5,G6])
    return E1

def Eps2(e,x):
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

options = ["-O1", "-O2", "-O3", "-Ofast"]
cffi_options = [options[3]]
jit_options = {"cffi_extra_compile_args": cffi_options,
                                "cffi_libraries": ["m"]}

deg=2
#-----------------Finite Element function Space-------------------------------------------
Ve = VectorElement("S", mesh.ufl_cell(), deg)
V = dolfinx.fem.FunctionSpace(mesh, Ve)
frame = local_frame(mesh)
ve1, ve2, ve3= frame
e1,e2,e3=Function(V), Function(V), Function(V)

fexpr1=dolfinx.fem.Expression(ve1,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1.interpolate(fexpr1) 

fexpr2=dolfinx.fem.Expression(ve2,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2.interpolate(fexpr2) 

fexpr3=dolfinx.fem.Expression(ve3,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3.interpolate(fexpr3)

dv = TrialFunction(V)
v_ = TestFunction(V)
deg=2
# Left Boundary

Ve_l = VectorElement("S", mesh_l.ufl_cell(), deg)
V_l = FunctionSpace(mesh_l, Ve_l)

w_l = Function(V_l)
# Right Boundary

Ve_r = VectorElement("S", mesh_r.ufl_cell(), deg)
V_r = FunctionSpace(mesh_r, Ve_r)
w_r = Function(V_r)
frame_l = local_frame_1D(mesh_l)
le1, le2, le3= frame_l

frame_r = local_frame_1D(mesh_r)
re1, re2, re3= frame_r

e1r,e2r,e3r=Function(V_r), Function(V_r), Function(V_r)
e1l,e2l,e3l=Function(V_l), Function(V_l), Function(V_l)

# Interpolating local frame in FE space for evaluating curvatures.

fexpr1=dolfinx.fem.Expression(le1,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1l.interpolate(fexpr1) 

fexpr2=dolfinx.fem.Expression(le2,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2l.interpolate(fexpr2) 

fexpr3=dolfinx.fem.Expression(le3,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3l.interpolate(fexpr3) 

fexpr1=dolfinx.fem.Expression(re1,V_r.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1r.interpolate(fexpr1) 

fexpr2=dolfinx.fem.Expression(re2,V_r.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2r.interpolate(fexpr2) 

fexpr3=dolfinx.fem.Expression(re3,V_r.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3r.interpolate(fexpr3) 

mesh.topology.create_connectivity(1, 2)
            
#boundary_facets= exterior_facet_indices(mesh.topology)
boundary_dofs = locate_dofs_topological(V, fdim, [entity_mapr,entity_mapl])

v2a=Function(V)
e=[e1,e2,e3]
x, dx= SpatialCoordinate(mesh), Measure('dx')(domain=mesh)
Eps=Eps2(e,x)[:,0] 
F2 = dot(sigma(dv, 0, Eps,e,x), eps(e,x,v_))*dx
# bc 
bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
# Assembled Coefficient matrix (without bc implementation), used in final Dhe computation
a= form(lhs(F2))
B=assemble_matrix(a)
B.assemble()
ai, aj, av=B.getValuesCSR()
BB=csr_matrix((av, aj, ai))
BB=BB.toarray()
#-------------------------------------------------- 
# Coefficient matrix with bc implemented and removed truncation error.
A=assemble_matrix(a,[bc])
A.assemble()
ai, aj, av=A.getValuesCSR()
AA=csr_matrix((av, aj, ai))
AA=AA.toarray()
avg=np.trace(AA)/len(AA)       
for i,x in enumerate(av):
    if x==1:
        av[i]=avg      
AA_csr=scipy.sparse.csr_matrix((av, aj, ai))
AAA=AA_csr.toarray() 
AA=scipy.sparse.csr_matrix(AAA) # To be used in final assembly


############# Left Nullspace ###################
dvl = TrialFunction(V_l)
v_l= TestFunction(V_l)
x=SpatialCoordinate(mesh_l)
e_l=[e1l,e2l,e3l]
dx = Measure('dx')(domain=mesh_l)
#
Eps= Eps2(e_l,x)[:,0]
F2 = dot(sigma(dvl, 0, Eps,e_l,x), eps(e_l,x,v_l))*dx
A_l=assemble_matrix(form(lhs(F2), jit_options=jit_options))
A_l.assemble()

# Nullspace implement

index_map = V_l.dofmap.index_map
nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V_l.dofmap.index_map_bs) for i in range(4)]

with ExitStack() as stack:
    vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
    basis = [np.asarray(x) for x in vec_local]
    
# Dof indices for each subspace (x, y and z dofs)
dofs = [V_l.sub(i).dofmap.list for i in range(3)]

# Build translational null space basis
for i in range(3):
    basis[i][dofs[i]] = 1.0
    
# Create vector space basis and orthogonalize
dolfinx.la.orthonormalize(nullspace_basis)

nullspace_l = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
assert nullspace_l.test(A_l)
# Set the nullspace
A_l.setNullSpace(nullspace_l)

############ Right Nullspace ###################
dvr = TrialFunction(V_r)
v_r= TestFunction(V_r)
x=SpatialCoordinate(mesh_r)
e_r=[e1r,e2r,e3r]
dx = Measure('dx')(domain=mesh_r)
#
Eps= Eps2(e_r,x)[:,0]
F2 = dot(sigma(dvr, 0, Eps,e_r,x), eps(e_r,x,v_r))*dx
A_r=assemble_matrix(form(lhs(F2), jit_options=jit_options))
A_r.assemble()

# Nullspace implement

index_map = V_r.dofmap.index_map
nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V_r.dofmap.index_map_bs) for i in range(4)]

with ExitStack() as stack:
    vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
    basis = [np.asarray(x) for x in vec_local]
    
# Dof indices for each subspace (x, y and z dofs)
dofs = [V_r.sub(i).dofmap.list for i in range(3)]

# Build translational null space basis
for i in range(3):
    basis[i][dofs[i]] = 1.0
    # For degree 2, only translational null space can be implemented. As getting coordinates of mid curved edges is not usual for rotation nullspace.
    # boundary can be given degree 1 (all 4 nullspace involved), and shell elements can be solved using degree 2 elements.
# Create vector space basis and orthogonalize
dolfinx.la.orthonormalize(nullspace_basis)

nullspace_r = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
assert nullspace_r.test(A_r)
# Set the nullspace
A_r.setNullSpace(nullspace_r)

local_range = V.dofmap.index_map.local_range
dofss = np.arange(*local_range)
xxx=3*len(dofss)
V0 = np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
D_ee=np.zeros((4,4))

mesh.topology.create_connectivity(1, 2)
mesh_l.topology.create_connectivity(0, 1)
mesh_r.topology.create_connectivity(0, 1)
# Assembly
for p in range(4):
    # Left 
    v2a=Function(V)
    e=[e1l,e2l,e3l]
    x, dx=SpatialCoordinate(mesh_l), Measure('dx')(domain=mesh_l)
    Eps=Eps2(e,x)[:,p] 
    F2 = dot(sigma(dvl, 0, Eps,e,x), eps(e,x,v_l))*dx
    F_l = petsc.assemble_vector(form(rhs(F2),  jit_options=jit_options))
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_l.remove(F_l)
    
    a,L= form(lhs(F2)), form(rhs(F2))
    
  # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A_l)
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
    #dof_mapping
    
    dof_S2L=[]
    boundary_facets = dolfinx.mesh.locate_entities(mesh, fdim, left)
    boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
    for i,xx in enumerate(boundary_facets):
        dofs = dolfinx.fem.locate_dofs_topological(V,1, entity_mapl[i])
        dofs_left= dolfinx.fem.locate_dofs_topological(V_l,1, boundary_facets_left[i])

        for k in range(deg+1):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                l=3*dofs_left[k],3*dofs_left[k]+1, 3*dofs_left[k]+2
                m=3*dofs[k],3*dofs[k]+1,3*dofs[k]+2
                v2a.vector[m]=w_l.vector[l]
    # Right
    e=[e1r,e2r,e3r]
    x, dx=SpatialCoordinate(mesh_r), Measure('dx')(domain=mesh_r)
    Eps=Eps2(e,x)[:,p] 
    F2 = dot(sigma(dvr, 0, Eps,e,x), eps(e,x,v_r))*dx
    F_r = petsc.assemble_vector(form(rhs(F2),  jit_options=jit_options))
    F_r.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_r.remove(F_r)
    a,L= form(lhs(F2)), form(rhs(F2))

  # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A_r)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F_r, w_r.vector)
    w_r.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    
    #dof_mapping
    dof_S2R=[]
    boundary_facets = dolfinx.mesh.locate_entities(mesh, fdim, right)
    boundary_facets_right= dolfinx.mesh.locate_entities(mesh_r, fdim, right)
    for i,xx in enumerate(boundary_facets):
        dofs = dolfinx.fem.locate_dofs_topological(V,1, entity_mapr[i])
        dofs_right= dolfinx.fem.locate_dofs_topological(V_r,1, boundary_facets_right[i])

        for k in range(deg+1):
            if dofs[k] not in dof_S2R:
                dof_S2R.append(dofs[k])
                r=3*dofs_right[k],3*dofs_right[k]+1, 3*dofs_right[k]+2
                m=3*dofs[k],3*dofs[k]+1,3*dofs[k]+2
                v2a.vector[m]=w_r.vector[r]  

    # ------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------
    w=Function(V)
    e=[e1,e2,e3]
    x, dx=SpatialCoordinate(mesh), Measure('dx')(domain=mesh)
    Eps=Eps2(e,x)[:,p] 
    F2 = dot(sigma(dv, 0, Eps,e,x), eps(e,x,v_))*dx
    # bc 
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    a,b= form(lhs(F2)), form(rhs(F2))
#------------------------------------------------------------------------------------- 

    F = petsc.assemble_vector(b)
    apply_lifting(F, [a], [[bc]])
    set_bc(F, [bc])
    FF=F[:]
  # Truncation Error removal for Dhe vector
    for i in boundary_dofs:
        for k in range(3):
            FF[3*i+k]=avg*F[3*i+k]

    V1s= scipy.sparse.linalg.spsolve(AA, FF, permc_spec=None, use_umfpack=True)
    Dhe[:,p]=  np.matmul(BB,V1s) # BB matrix is symmetric coefficient matrix without bc modifcation.
    V0[:,p]= V1s # V0 matrix formation  
    print(p) 
D1=np.matmul(V0.T,-Dhe) 

for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(dot(dot(Eps2(e,x).T,ABD),Eps2(e,x))[s,k]*dx)
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1
# Omega
omega=dolfinx.fem.assemble_scalar(form(sum([1*dx])))
D_eff=D_eff/(x_max-x_min)

#-------------------------Printing Output Data-------------------------------
print('  ')  
print('Stiffness Matrix')
np.set_printoptions(linewidth=np.inf)
print(np.around(D_eff))  

#Optional
############################################################################
############################################################################
################## EB Model of Boundary#####################################
############################################################################
############################################################################
# Left boundary solve
local_range = V_l.dofmap.index_map.local_range
dofs = np.arange(*local_range)
xxx=3*len(dofs)
V0l = np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
D_ee=np.zeros((4,4))
for p in range(4):
    e=[e1l,e2l,e3l]
    x, dx=SpatialCoordinate(mesh_l), Measure('dx')(domain=mesh_l)
    Eps=Eps2(e,x)[:,p] 
    F2 = dot(sigma(dvl, 0, Eps,e,x), eps(e,x,v_l))*dx
    F_l = petsc.assemble_vector(form(rhs(F2),  jit_options=jit_options))
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_l.remove(F_l)
    a,L= form(lhs(F2)), form(rhs(F2))

    # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A_l)
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
    
    Dhe[:,p]= F_l[:] # Dhe matrix formation
    V0l[:,p]= w_l.vector[:] # V0 matrix formation
D1=np.matmul(V0l.T,-Dhe) 
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(dot(dot(Eps2(e,x).T,ABD),Eps2(e,x))[s,k]*dx)
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1
# Omega
omega=dolfinx.fem.assemble_scalar(form(sum([1*dx])))
D_eff=D_eff
np.set_printoptions(linewidth=np.inf)
print(np.around(D_eff))  
################################################################################################
# Right boundary solve
local_range = V_r.dofmap.index_map.local_range
dofs = np.arange(*local_range)
xxx=3*len(dofs)
V0r = np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
D_ee=np.zeros((4,4))
for p in range(4):
    e=[e1r,e2r,e3r]
    x, dx=SpatialCoordinate(mesh_r), Measure('dx')(domain=mesh_r)
    Eps=Eps2(e,x)[:,p] 
    F2 = dot(sigma(dvr, 0, Eps,e,x), eps(e,x,v_r))*dx
    F_r = petsc.assemble_vector(form(rhs(F2),  jit_options=jit_options))
    F_r.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    nullspace_r.remove(F_r)
    a,L= form(lhs(F2)), form(rhs(F2))

    # ksp solve
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A_r)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    ksp.getPC().getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F_r, w_r.vector)
    w_r.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    
    Dhe[:,p]= F_r[:] # Dhe matrix formation
    V0r[:,p]= w_r.vector[:] # V0 matrix formation
D1=np.matmul(V0r.T,-Dhe) 
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(dot(dot(Eps2(e,x).T,ABD),Eps2(e,x))[s,k]*dx)
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1
# Omega
omega=dolfinx.fem.assemble_scalar(form(sum([1*dx])))
D_eff=D_eff
np.set_printoptions(linewidth=np.inf)
print(np.around(D_eff)) 
