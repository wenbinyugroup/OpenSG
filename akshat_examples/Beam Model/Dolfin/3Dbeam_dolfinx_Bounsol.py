
###############     OpenSG Solid Patch  Test       ############################
########### 3D Timo Beam Using Boundary Solutions ###################


import meshio# Read the Gmsh file

mesh = meshio.read("SG40_3D_x.msh")# Write volume mesh (tetrahedra)
tetra_cells = mesh.cells_dict.get("tetra", None)
if tetra_cells is not None:
    tetra_mesh = meshio.Mesh(
        points=mesh.points,
        cells={"tetra": tetra_cells},
        cell_data={"gmsh:physical": [mesh.cell_data_dict["gmsh:physical"]["tetra"]]}
    )
    meshio.write("mesh.xdmf", tetra_mesh)# Write facet mesh (triangles for boundaries)
    
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh
from mpi4py import MPI
import dolfinx.fem as fem
import numpy as np

# Read the volume mesh
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    subdomains = xdmf.read_meshtags(mesh, name="Grid")


from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix 
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
import meshio
import dolfinx
from dolfinx.fem import form, petsc, Function, functionspace, locate_dofs_topological, apply_lifting, set_bc, assemble_scalar
from ufl import Jacobian, as_vector, dot, cross,sqrt, conditional, replace, as_matrix,FacetNormal
from ufl import lt,SpatialCoordinate, as_tensor,  Measure
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, dx, dot,eq, cos, sin
import petsc4py.PETSc
from contextlib import ExitStack
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from typing import Dict
import ufl
import basix
from scipy.sparse import csr_matrix, linalg
import numpy as np
import yaml
from yaml import CLoader as cLd
import scipy
import gmsh
## Define input parameters
import pyvista 


#gmsh.initialize()
#gmsh.option.setNumber("General.Terminal", 0)    # mesh read output will not be printed
#mesh, subdomains, boundaries = gmshio.read_from_msh("SG40_3D_x.msh", MPI.COMM_SELF,rank=0, gdim=3) 
o_cell_idx=mesh.topology.original_cell_index  # Original cell Index from mesh file


#lnn=[]
#for k in o_cell_idx:
#     lnn.append(0)
#lnn=np.array(lnn,dtype=np.int32)    

cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells = cell_map.size_local + cell_map.num_ghosts
#cells = np.arange(num_cells, dtype=np.int32)
#subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, lnn)


#import pyvista
pyvista.start_xvfb()
u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(mesh,mesh.topology.dim)
grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
grid.cell_data["Marker"] = subdomains.values[:]
grid.set_active_scalars("Marker")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid, show_edges=True)

u_plotter.show_axes()
#u_plotter.add_axes_at_origin()
#u_plotter.view_yz() # z is beam axis
u_plotter.show()


pp=mesh.geometry.x
x_min,x_max=min(pp[:,0]), max(pp[:,0])
L=x_max-x_min
#pp[:,0]=pp[:,0]-0.5*(x_min+x_max)
#x_min,x_max=min(pp[:,0]), max(pp[:,0])

fdim=2
def left(x):
    return np.isclose(x[0], x_min,atol=0.01)
def right(x):
    return np.isclose(x[0], x_max,atol=0.01)

tdim=mesh.topology.dim
fdim = tdim - 1
facets_left = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim,
                                       marker=left)
facets_right = dolfinx.mesh.locate_entities_boundary(mesh, dim=fdim,
                                       marker=right)

mesh_l, entity_mapl, vertex_mapl, geom_mapl = create_submesh(mesh, fdim, facets_left)
mesh_r, entity_mapr, vertex_mapr, geom_mapr = create_submesh(mesh, fdim, facets_right)


deg=1
def local_boun(mesh,subdomains): 
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "CG", mesh.topology.cell_name(), deg, shape=(3, )))  #** Polynomial Order=1
        
    return V, TrialFunction(V), TestFunction(V),SpatialCoordinate(mesh), \
          Measure('dx')(domain=mesh, subdomain_data=subdomains)

nphases = max(subdomains.values)+1

mesh.topology.create_connectivity(3,2)
cell_of_face_mesh=mesh.topology.connectivity(3,2)
el_face=4



# Cell to Face connectivity
conn3=[]
for i in range(num_cells):
    c=[]
    for k in range(el_face):
        c.append((cell_of_face_mesh.array[el_face*i+k]))
    conn3.append(c) 
conn=np.ndarray.flatten(np.array(conn3))

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


def subdomains_boun(mesh_l,left,entity_mapl):
    VV_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
        "DG", mesh_l.topology.cell_name(), 0, shape=(3, )))
    th,El1,El2,Nl=Function(VV_l),Function(VV_l),Function(VV_l),Function(VV_l)
    sub_L=[]
    boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
    for i,xx in enumerate(entity_mapl):
             idx=int(np.where(conn==xx)[0]/el_face)   
             sub_L.append(subdomains.values[idx])
             
    sub_L= np.array(sub_L,dtype=np.int32)
    num_cells_l = mesh_l.topology.index_map(mesh_l.topology.dim).size_local 
    cells_l = np.arange(num_cells_l, dtype=np.int32)
    subdomains_l = dolfinx.mesh.meshtags(mesh_l, mesh_l.topology.dim, cells_l, sub_L)
    return subdomains_l,  boundary_facets_left
  
subdomains_l, boundary_facets_left=subdomains_boun(mesh_l,left,entity_mapl) # generating boundary submesh
subdomains_r, boundary_facets_right=subdomains_boun(mesh_r,right,entity_mapr)


def C(i):  # Stiffness matrix
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12 
    CC=as_tensor(np.linalg.inv(S))
    return CC  
#

def ksp_solve(A,F,V): # Linear Solver 
    w = Function(V)
    ksp = petsc4py.PETSc.KSP()
    ksp.create(comm=MPI.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFactorSetUpSolverType()
    mat = ksp.getPC().getFactorMatrix()
    mat.setMumpsIcntl(icntl=24, ival=1)  # detect null pivots
    mat.setMumpsIcntl(icntl=25, ival=0)  # do not compute null space again
    ksp.setFromOptions()
    ksp.solve(F, w.vector)  # Solve scaled system

    w.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    return w.vector[:]

def nullspace(V): # Apply constraints to restrict Rigid body motion
    index_map = V.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(6)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]
    dofs = [V.sub(i).dofmap.list for i in range(3)]      # Dof indices for each subspace (x, y and z dofs)
    for i in range(3):                                   # Build translational null space basis
        basis[i][dofs[i]] = 1.0
        
    xx = V.tabulate_dof_coordinates()
    xx = xx.reshape((-1, 3))
    
    for i in range(len(xx)):  # Build twist nullspace
        basis[3][3*i+1]=-xx[i,2]
        basis[3][3*i+2]=xx[i,1] 
        
    # Build rotational null space basis
  #  xx = V.tabulate_dof_coordinates()
 #   dofs_block = V.dofmap.list
 #   x1, x2, x3 = xx[dofs_block, 0], xx[dofs_block, 1], xx[dofs_block, 2]
 #   basis[3][dofs[1]] = -x3
 #   basis[3][dofs[2]] = x2

    dolfinx.la.orthonormalize(nullspace_basis)           # Create vector space basis and orthogonalize
    # Convert nullspace_l to a dense matrix N
    N = []
    for n in nullspace_basis:
        with n.localForm() as n_local:
            N.append(np.asarray(n_local))
    N = np.column_stack(N)  # Shape: (n_dofs, 6)
    N=N[:,0:4]
    return N, nullspace_basis

def gamma_h(dx,v): 
        return as_vector([0,v[1].dx(1),v[2].dx(2),v[1].dx(2)+v[2].dx(1),v[0].dx(2),v[0].dx(1)])

def gamma_l(v): 
    return as_vector([v[0],0,0,0, v[2],v[1]])  

def gamma_e(x):   
    return as_tensor([(1,0,x[2],-x[1]),
                (0,0,0,0),
                (0,0,0,0),
                (0,0,0,0),
               (0,x[1],0,0),
               (0,-x[2],0,0)])

def A_mat(dx_l,v_l,dvl):  # Boundary Coefficient matrix D_hh
    F2 = sum([dot(dot(C(i),gamma_h(dx_l,dvl)),gamma_h(dx_l,v_l))*dx_l(i) for i in range(nphases)])       
    A_l=assemble_matrix(form(F2))
    A_l.assemble()
    return A_l

def initialize_array(V): 
    xxx=3*V.dofmap.index_map.local_range[1]  # total dofs 
    V0 = np.zeros((xxx,4))
    Dle=np.zeros((xxx,4))
    Dhe=np.zeros((xxx,4)) 
    Dee=np.zeros((4,4)) 
    V1s=np.zeros((xxx,4))
    return V0,Dle,Dhe,Dee,V1s

def timo_boun(mesh_l,subdomains_l): # Compute Boundary Timoshenko Solutions
    V_l, dv, v_, x, dx=local_boun(mesh_l,subdomains_l)   
    
    mesh_l.topology.create_connectivity(2, 2)
    V0,Dle,Dhe,Dee,V1s=initialize_array(V_l)
    
    N, nullspace_basis=nullspace(V_l)
    nullspace_b=petsc4py.PETSc.NullSpace().create(vectors=nullspace_basis)
    A_l=A_mat(dx,v_,dv) 
    A_l.setNullSpace(nullspace_b)
 
    # Boundary Euler-Bernoulli 
    for i in range(4):
        F2 = sum([dot(dot(C(ii),gamma_e(x)[:,i]),gamma_h(dx,v_))*dx(ii) for ii in range(nphases)])  
        r_he=form(rhs(F2))
        F = petsc.assemble_vector(r_he)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        nullspace_b.remove(F)
     #   F_array = F_l[:]
     #   for n in nullspace_basis:
      #      with n.localForm() as n_local:
      #          n_array = np.asarray(n_local)
       #         n_array = n_array
      #          F_array -= np.dot(F_array, n_array) * n_array
        Dhe[:,i]=F[:]   
     #   F=petsc4py.PETSc.Vec().createWithArray(F_array,comm=MPI.COMM_WORLD)
     #   F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
        V0[:,i]= ksp_solve(A_l, F, V_l)
  
    V0_csr=csr_matrix(V0) 
    D1=V0_csr.T.dot(csr_matrix(-Dhe))  

    def Dee_(i):
        Cc=C(i)
        x2,x3=x[1],x[2]
        return as_tensor([(Cc[0,0], Cc[0,4]*x2-Cc[0,5]*x3,Cc[0,0]*x3,-Cc[0,0]*x2),
                          (Cc[4,0]*x2-Cc[5,0]*x3, x2*(Cc[4,4]*x2-Cc[5,4]*x3)-x3*(Cc[4,5]*x2-Cc[5,5]*x3),x3*(Cc[4,0]*x2-Cc[5,0]*x3),-x2*(Cc[4,0]*x2-Cc[5,0]*x3)),
                          (Cc[0,0]*x3,  x3*(Cc[0,4]*x2-Cc[0,5]*x3), Cc[0,0]*x3**2, -Cc[0,0]*x2*x3),
                          (-Cc[0,0]*x2, -x2*(Cc[0,4]*x2-Cc[0,5]*x3),  -Cc[0,0]*x2*x3, Cc[0,0]*x2**2)])
    for s in range(4):
        for k in range(4): 
            f=dolfinx.fem.form(sum([Dee_(i)[s,k]*dx(i) for i in range(nphases)]))
            Dee[s,k]=dolfinx.fem.assemble_scalar(f)
    D_eff= Dee + D1 # Effective Stiffness Matrix (EB)
    
    # Boundary timoshenko matrix
    F1=sum([dot(dot(C(i),gamma_l(dv)),gamma_l(v_))*dx(i) for i in range(nphases)])
    a1=form(F1)
    Dll=assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av=Dll.getValuesCSR()
    Dll=csr_matrix((av, aj, ai))    
    
    for p in range(4):
            F1=sum([dot(dot(C(i),gamma_e(x)[:,p]),gamma_l(v_))*dx(i) for i in range(nphases)])
            Dle[:,p]= petsc.assemble_vector(form(F1))[:]      
        
    F_dhl=sum([dot(dot(C(i),gamma_h(dx,dv)),gamma_l(v_))*dx(i) for i in range(nphases)]) 
    a3=form(F_dhl) 
    Dhl=assemble_matrix(a3)
    Dhl.assemble()   
    ai, aj, av=Dhl.getValuesCSR()
    Dhl=csr_matrix((av, aj, ai))
        
    Dle_csr =csr_matrix(Dle)
    
    #DhlV0=np.matmul(Dhl.T,V0)
    DhlV0=Dhl.T.dot(V0_csr) 
    
    #DhlTV0Dle=np.matmul(Dhl,V0)+Dle
    DhlTV0Dle=Dhl.dot(V0_csr)+Dle_csr
    
    #V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)
    V0DllV0=(V0_csr.T.dot(Dll)).dot(V0_csr)
    
    # V1s
    bb=(DhlTV0Dle-DhlV0).toarray()
    
    for i in range(4):
        F_array = bb[:,i]
     #   for n in nullspace_basis:
     #       with n.localForm() as n_local:
     #           n_array = np.asarray(n_local)
     #           n_array = n_array
     #           F_array -= np.dot(F_array, n_array) * n_array
        F=petsc4py.PETSc.Vec().createWithArray(F_array,comm=MPI.COMM_WORLD)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
        nullspace_b.remove(F)
        V1s[:,i]= ksp_solve(A_l, F, V_l)
     
    V1s_csr=csr_matrix(V1s)    
    # Ainv
    Ainv=np.linalg.inv(D_eff).astype(np.float64)
    
    # B_tim
    B_tim=DhlTV0Dle.T.dot(V0_csr)
    B_tim=B_tim.toarray().astype(np.float64)
    
    # C_tim
    C_tim= V0DllV0 + V1s_csr.T.dot(DhlV0 + DhlTV0Dle) 
    C_tim=0.5*(C_tim+C_tim.T)
    C_tim=C_tim.toarray().astype(np.float64)
    
    # Ginv
    Q_tim=np.matmul(Ainv,np.array([(0,0),(0,0),(0,-1),(1,0)])).astype(np.float64)
    Ginv= np.matmul(np.matmul(Q_tim.T,(C_tim-np.matmul(np.matmul(B_tim.T,Ainv),B_tim))),Q_tim).astype(np.float64)
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
    Deff_srt[3:6,0]=A_tim[1:4,0].flatten()
    
    Deff_srt[1:3,1:3]=G_tim
    Deff_srt[1:3,3:6]=Y_tim.T[:,1:4]
    Deff_srt[1:3,0]=Y_tim.T[:,0].flatten()

    return np.around(D_eff),np.around(Deff_srt),V0, V1s,Dhe,bb,A_l,C_tim,B_tim,Dee

DEB_l,Deff_l,V0_l,V1_l,bl1,bbl2,A_l,C_timl,B_timl,Dee_l=timo_boun(mesh_l,subdomains_l) 

#******************************TIMO BOUNDARY SOLVE*******************************
print('Left Timo \n')
np.set_printoptions(precision=4) 
print(Deff_l) 


DEB_r,Deff_r,V0_r,V1_r,br2,br1,A_r,C_timr,B_timr, D1r=timo_boun(mesh_r,subdomains_r)
print('Right Timo \n')
np.set_printoptions(precision=4) 
print(Deff_r) 


V_l, dvl, v_l, x_l, dx_l=local_boun(mesh_l,subdomains_l)
V_r, dvr, v_r, x_r, dx_r=local_boun(mesh_r,subdomains_r)  
V, dv,  v_,  x, dx=local_boun(mesh,subdomains)  

def dof_mapping_quad(v2a,V_l,w_ll,boundary_facets_left,entity_mapl):
    dof_S2L=[] 
    for i,xx in enumerate(entity_mapl):
        dofs = locate_dofs_topological(V,2, np.array([xx]))
        dofs_left= locate_dofs_topological(V_l,2, np.array([boundary_facets_left[i]]))

        for k in range(len(dofs)): # Quadratic lagrange has 9 dofs for quadrilateral face.  
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.vector[3*dofs[k]+j]=w_ll[3*dofs_left[k]+j] # store boundary solution of fluctuating functions
    return v2a,dof_S2L

#******************TAPER INITIALIZE***************************************

F2 = sum([(dot(dot(C(i),gamma_h(dx,dv)),gamma_h(dx,v_))+dot(dv,v_))*dx(i) for i in range(nphases)])   
a= form(F2)

N, nullspace_basis=nullspace(V)

# bc applied
boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((entity_mapr,entity_mapl), axis=0))
v2a=Function(V)
bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
A=assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
A.assemble()

nullspace_b=petsc4py.PETSc.NullSpace().create(vectors=nullspace_basis)
#assert nullspace_b.test(A)

A.setNullSpace(nullspace_b)   
V0,Dle,Dhe,Dee,V1s=initialize_array(V)
mesh.topology.create_connectivity(2, 3)
mesh_l.topology.create_connectivity(2, 2)
mesh_r.topology.create_connectivity(2, 2)

def Dee_(i):
    Cc=C(i)
    x2,x3=x[1],x[2]
    return as_tensor([(Cc[0,0], Cc[0,4]*x2-Cc[0,5]*x3,Cc[0,0]*x3,-Cc[0,0]*x2),
                      (Cc[4,0]*x2-Cc[5,0]*x3, x2*(Cc[4,4]*x2-Cc[5,4]*x3)-x3*(Cc[4,5]*x2-Cc[5,5]*x3),x3*(Cc[4,0]*x2-Cc[5,0]*x3),-x2*(Cc[4,0]*x2-Cc[5,0]*x3)),
                      (Cc[0,0]*x3,  x3*(Cc[0,4]*x2-Cc[0,5]*x3), Cc[0,0]*x3**2, -Cc[0,0]*x2*x3),
                      (-Cc[0,0]*x2, -x2*(Cc[0,4]*x2-Cc[0,5]*x3),  -Cc[0,0]*x2*x3, Cc[0,0]*x2**2)])

# ***************TAPER EULER-BERNOULLI**********************************          
for p in range(4): # 4 load cases meaning 
   # Boundary 
   v2a=Function(V) 
   v2a,dofs_l=dof_mapping_quad(v2a,V_l,V0_l[:,p],boundary_facets_left,entity_mapl)
   v2a,dofs_r=dof_mapping_quad(v2a,V_r,V0_r[:,p],boundary_facets_right,entity_mapr)  
   
   F2=-sum([dot(dot(C(i),gamma_e(x)[:,p]),gamma_h(dx,v_))*dx(i) for i in range(nphases)]) 
   bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
   F = petsc.assemble_vector(form(F2)) 
   F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
   nullspace_b.remove(F)

   Dhe[:,p]=F        
   apply_lifting(F, [a], [bc]) # apply bc to rhs vector (Dhe) based on known fluc solutions
   set_bc(F, bc) 
   V0[:,p]=ksp_solve(A,F,V)

V0_csr=csr_matrix(V0)    
D1=V0_csr.T.dot(csr_matrix(-Dhe)).astype(np.float64)
for s in range(4):
   for k in range(4): 
       f=dolfinx.fem.form(sum([Dee_(i)[s,k]*dx(i) for i in range(nphases)]))
       Dee[s,k]=dolfinx.fem.assemble_scalar(f)
       
D_eff= (Dee.astype(np.float64) + D1)/L         
D_eff=0.5*(D_eff+D_eff.T)


#*****************TAPER TIMOSHENKO**********************************
F1=sum([dot(dot(C(i),gamma_l(dv)),gamma_l(v_))*dx(i) for i in range(nphases)])
a1=form(F1)
Dll=assemble_matrix(a1)
Dll.assemble()

ai, aj, av=Dll.getValuesCSR()
Dll=csr_matrix((av, aj, ai))   


# In[34]:


for p in range(4):
        F1=sum([dot(dot(C(i),gamma_e(x)[:,p]),gamma_l(v_))*dx(i) for i in range(nphases)])
        Dle[:,p]= petsc.assemble_vector(form(F1))[:]

F_dhl=sum([dot(dot(C(i),gamma_h(dx,v_)),gamma_l(dv))*dx(i) for i in range(nphases)]) 
a3=form(F_dhl)
Dhl=assemble_matrix(a3)
Dhl.assemble()

ai, aj, av=Dhl.getValuesCSR()
Dhl=csr_matrix((av, aj, ai))

Dle_csr =csr_matrix(Dle)

#DhlV0=np.matmul(Dhl.T,V0)
DhlV0=Dhl.T.dot(V0_csr) 

#DhlTV0Dle=np.matmul(Dhl,V0)+Dle
DhlTV0Dle=Dhl.dot(V0_csr)+Dle_csr

#V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)
V0DllV0=(V0_csr.T.dot(Dll)).dot(V0_csr)

# B_tim
B_tim=DhlTV0Dle.T.dot(V0_csr)
B_tim=B_tim.toarray().astype(np.float64)/L


# V1s
b=(DhlTV0Dle-DhlV0).toarray()

for p in range(4): # 4 load cases meaning 
    # Boundary 
    v2a=Function(V) 
    v2a,dofs_l=dof_mapping_quad(v2a,V_l,V1_l[:,p],boundary_facets_left, entity_mapl)
    v2a,dofs_r=dof_mapping_quad(v2a,V_r,V1_r[:,p],boundary_facets_right,entity_mapr)  
    bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
    F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)   
    nullspace_b.remove(F)
    apply_lifting(F, [a], [bc]) # apply bc to rhs vector (Dhe) based on known fluc solutions
    set_bc(F, bc)   

    V1s[:,p]=ksp_solve(A,F,V)
    
V1s_csr=csr_matrix(V1s)
L=(x_max-x_min)


C_tim=V0DllV0 + V1s_csr.T.dot(DhlV0 + DhlTV0Dle) 
C_tim=0.5*(C_tim+C_tim.T)

C_tim=C_tim.toarray()/L

# Ainv 
Ainv=np.linalg.inv(D_eff)

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
Deff_srt[3:6,0]=A_tim[1:4,0].flatten()

Deff_srt[1:3,1:3]=G_tim
Deff_srt[1:3,3:6]=Y_tim.T[:,1:4]
Deff_srt[1:3,0]=Y_tim.T[:,0].flatten()

print('\n Solid Taper Timo \n')

np.set_printoptions(precision=4)
print(np.around(Deff_srt))  

