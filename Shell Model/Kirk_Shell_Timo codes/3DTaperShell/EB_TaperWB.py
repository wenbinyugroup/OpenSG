

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
import scipy.sparse.linalg

import yaml
from yaml import CLoader as cLd

## Updated: Define input parameters
meshYaml = 'bar_urc_shell_mesh.yaml'  ## the name of the yaml file containing the whole blade mesh
secInd = [1] ## the index of the spanwise section you want
mshFile = 'SG_shell.msh'

## Read the mesh yaml file
inFile = open(meshYaml,'r')
meshData = yaml.load(inFile,Loader=cLd)
inFile.close()

## Extract the mesh for the section
nodes = meshData['nodes']
numNds = len(nodes)
elements = meshData['elements']
numEls = len(elements)

ndNewLabs = -1*np.ones(numNds,dtype=int)
elNewLabs = -1*np.ones(numEls,dtype=int)
elLayID = -1*np.ones(numEls,dtype=int)

layCt = 0
for es in meshData['sets']['element']:
    nmLst = es['name'].split('_')
    if(len(nmLst) > 2):
        if(int(nmLst[1]) in secInd):
            for eli in es['labels']:
                elNewLabs[eli] = 1
                elLayID[eli] = layCt
                for nd in elements[eli]:
                    if(nd > -1):
                        ndNewLabs[nd] = 1
            layCt += 1

eLab = 1
for i, e in enumerate(elNewLabs):
    if(e == 1):
        elNewLabs[i] = eLab
        eLab += 1

ndLab = 1
for i, n in enumerate(ndNewLabs):
    if(n == 1):
        ndNewLabs[i] = ndLab
        ndLab += 1

## Write .msh file
outFile = open(mshFile,'w')

outFile.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
newNumNds = np.max(ndNewLabs)
outFile.write(str(newNumNds) + '\n')

for i, nd in enumerate(nodes):
    lab = ndNewLabs[i]
    if(lab > -1):
        ln = [str(lab),str(nd[2]),str(nd[0]),str(nd[1])]
      #  ln = [str(lab),str(nd[0]),str(nd[1]),str(nd[2])]
        outFile.write(' '.join(ln) + '\n')

outFile.write('$EndNodes\n$Elements\n')

newNumEls = np.max(elNewLabs)
outFile.write(str(newNumEls) + '\n')

for i, el in enumerate(elements):
    lab = elNewLabs[i]
    if(lab > -1):
        ln = [str(lab)]
        if(el[3] == -1):
            ln.append('2')
        else:
            ln.append('3')
        ln.append('2')
        ln.append(str(elLayID[i]+1))
        ln.append(str(elLayID[i]+1))
        for nd in el:
            if(nd > -1):
                ln.append(str(ndNewLabs[nd]))
        outFile.write(' '.join(ln) + '\n')
outFile.write('$EndElements\n')

outFile.close()
mesh, subdomains, boundaries = gmshio.read_from_msh("SG_shell.msh", MPI.COMM_WORLD,0, gdim=3)
lnn=subdomains.values[:]-1
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local 
cells = np.arange(num_cells, dtype=np.int32)
subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, np.array(lnn,dtype=np.int32))
o_cell_idx= mesh.topology.original_cell_index

# Material_parameters
material_parameters,eProps=[],[]
matDic = dict()
for i, m in enumerate(meshData['materials']):
    matDic[m['name']] = i
    el = m['elastic']
    eProps = el['E']
    eProps.extend(el['G'])
    eProps.extend(el['nu'])
    material_parameters.append(eProps) 
material_parameters[0]=material_parameters[5]    
# Load Layup Data 
matid, thick,angle,nlay=[],[],[],[]
for sec in meshData['sections']:
    nmLst = sec['elementSet'].split('_')
    if(len(nmLst) > 2):
        m,t,an=[],[],[]
        if(int(nmLst[1]) in secInd):
            layup = sec['layup']
            nlay.append(len(layup))
            for l in layup:
                m.append(matDic[l[0]])     
            matid.append(m)
            for l in layup:
                t.append(l[1])
            thick.append(t)
            for l in layup:
                an.append(l[2])
            angle.append(an) 


# Local Orientation (DG0 function)
VV = dolfinx.fem.functionspace(mesh, basix.ufl.element(
    "DG", mesh.topology.cell_name(), 0, shape=(3, )))
EE1,EE2,N=Function(VV),Function(VV),Function(VV) 

o_cell_idx=mesh.topology.original_cell_index  # Original cell Index from mesh file
orien=[]
for i, eo in enumerate(meshData['elementOrientations']):
    elab = elNewLabs[i]
    if(elNewLabs[i] > -1):
        o=[]
        for k in range(9):
            o.append(eo[k])
        orien.append(o)

# Store orientation for each element
for k,ii in enumerate(o_cell_idx):
# Storing data to DG0 functions 
    EE2.vector[3*k],EE2.vector[3*k+1],EE2.vector[3*k+2]=orien[ii][5],orien[ii][3],orien[ii][4]  # e2
    N.vector[3*k],N.vector[3*k+1],N.vector[3*k+2]=orien[ii][8],orien[ii][6],orien[ii][7]   #  e3 
    EE1.vector[3*k], EE1.vector[3*k+1],EE1.vector[3*k+2]=orien[ii][2],orien[ii][0],orien[ii][1]  # e1    outward normal 
frame=[EE1,EE2,N]

# Geometry Extraction 
pp=mesh.geometry.x                 # point data
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
#
# generating cell to edge connectivity to map the subdomains (layup data) from quad mesh to boundary.
mesh.topology.create_connectivity(2,1)  # (quad mesh topology, boundary(1D) mesh topology)
cell_of_facet_mesh=mesh.topology.connectivity(2,1)
# Cell to Edge connectivity
conn3=[]
for i in range(num_cells):
    c=[]
    for k in range(4):
        c.append((cell_of_facet_mesh.array[4*i+k])) # 4 is used as number of edges ina  quad element
    conn3.append(c)
conn=np.ndarray.flatten(np.array(conn3))
# Left Boundary
def subdomains_boun(mesh_l,left,entity_mapl):
    VV_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
        "DG", mesh_l.topology.cell_name(), 0, shape=(3, )))
    El1,El2,Nl=Function(VV_l),Function(VV_l),Function(VV_l)
    sub_L=[]
    boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
    for i,xx in enumerate(entity_mapl):
             idx=int(np.where(conn==xx)[0]/4)   # 4 is for number of nodes in quad element
             sub_L.append(subdomains.values[idx])
             for j in range(3):
                        El1.vector[3*i+j]=EE1.vector[3*idx+j]
                        El2.vector[3*i+j]=EE2.vector[3*idx+j]
                        Nl.vector[3*i+j] =N.vector[3*idx+j]
    frame=El1,El2,Nl                    
    sub_L= np.array(sub_L,dtype=np.int32)
    num_cells_l = mesh_l.topology.index_map(mesh_l.topology.dim).size_local 
    cells_l = np.arange(num_cells_l, dtype=np.int32)
    subdomains_l = dolfinx.mesh.meshtags(mesh_l, mesh_l.topology.dim, cells_l, sub_L)
    return subdomains_l, frame,boundary_facets_left
    # Mapping the orinetation data from quad mesh to boundary. The alternative is to use local_frame_1D(mesh_l).
    # Either of both can be used in local_boun subroutine 
subdomains_l, frame_l,boundary_facets_left=subdomains_boun(mesh_l,left,entity_mapl) # generating boundary submesh
subdomains_r, frame_r,boundary_facets_right=subdomains_boun(mesh_r,right,entity_mapr)


# Generate ABD matrix (Plate model)
def ABD_mat(ii):
    deg = 2
    cell = ufl.Cell("interval")
    elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th,s=[0],0 # Reference-------- 0
    for k in thick[ii]:
        s=s-k           # Inward normal in orien provided by yaml file
        th.append(s)
    points = np.array(th) 
    # Elements  
    cell=[]
    for k in range(nlay[ii]):
        cell.append([k,k+1])   
    cellss = np.array(cell)
    
    # Create 1D SG mesh
    dom = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cellss, points, domain)
    num_cells= dom.topology.index_map(dom.topology.dim).size_local 
    cells = np.arange(num_cells, dtype=np.int32)
    subdomain = dolfinx.mesh.meshtags(dom, dom.topology.dim, cells, cells) # assigning each element as subdomain
    x,dx = SpatialCoordinate(dom),Measure('dx')(domain=dom, subdomain_data=subdomain)

    Eps2=as_tensor([(1,0,0,x[0],0,0),       # Gamma_e matrix
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
    F2 = sum([inner(sigma(u, i, Eps2[:,0])[0], eps(v)[0])*dx(i) for i in range(nphases)]) # Weak form of energy
    A=  petsc.assemble_matrix(form(lhs(F2)))
    A.assemble()
    null = nullspace(V)
    A.setNullSpace(null)      # Set the nullspace
    xx = 3*V.dofmap.index_map.local_range[1] # total dofs
    # Initialization
    V0, Dhe, D_ee = np.zeros((xx,6)), np.zeros((xx,6)), np.zeros((6,6))

    # Assembly
    for p in range(6):
        Eps=Eps2[:,p] 
        F2 = sum([inner(sigma(u, i, Eps)[0], eps(v)[0])*dx(i) for i in range(nphases)]) # weak form
        F = petsc.assemble_vector(form(rhs(F2)))
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(F)                        # Orthogonalize F to the null space of A^T
        w=ksp_solve(A,F,V)
        Dhe[:,p]= F[:]                        # Dhe matrix formation
        V0[:,p]= w.vector[:]                  # V0 matrix formation
    D1=np.matmul(V0.T,-Dhe)  
    
    def Dee(i):                         # Getting Eps.T*C*Eps in D_eff calculation
            C=sigma(u, i, Eps)[1]
            x0=x[0]
            return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
                          (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
                          (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
                          (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
                          (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
                         (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])

    for s in range(6):
        for k in range(6): 
            f=dolfinx.fem.form(sum([Dee(i)[s,k]*dx(i) for i in range(nphases)])) # Scalar assembly
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

    D_eff= D_ee + D1 
    return(D_eff)


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

def nullspace(V):
    index_map = V.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(4)]
    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]
    dofs = [V.sub(i).dofmap.list for i in range(3)]      # Dof indices for each subspace (x, y and z dofs)
    for i in range(3):                                   # Build translational null space basis
        basis[i][dofs[i]] = 1.0
    
    dolfinx.la.orthonormalize(nullspace_basis)           # Create vector space basis and orthogonalize
    return petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)

# Store ABD matrices for layup (as list)
nphases=max(subdomains.values[:])+1
ABD_=[] 
for i in range(nphases):
    ABD_.append(ABD_mat(i))
print('Computed',nphases,'ABD matrix')

def ABD_matrix(i):
    return(as_tensor(ABD_[i]))
########################################## ABD matrix generation ends here ###########################
##
####################################Finite Element Implementation ##########################################
#Local Frame for Boundary (should be same as quad mesh local frame)
# Local frame from OpenSG  (wind blade quad mesh)
def local_frame(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangential curvilinear direction -1
    t2 = as_vector([t[0, 1], t[1, 1], t[2, 1]]) # tangential curvilinear direction -2
    e3 = cross(t1, t2) # normal direction 
    e3 /= sqrt(dot(e3, e3))
    e1=  t2/ sqrt(dot(t2, t2)) # 1- direction
    e2 = cross(e3, e1)
    e2 /= sqrt(dot(e2, e2)) # 2- direction
    return e1, e2, e3
    # Note: we don't require local_frame when we are using input orientation stored in EE1, EE2,N.
# On comparison of local_frame(mesh) and given dc matrix (EE1,EE2,N), the difference is former is clockwise about beam axis, while later is ccw about beam axis
# for circumferential tangent direction. local_frame(mesh)-[e1,e2,e3] can be interpolated to DG (deg 0) with shape 3 to compare both data.
# manually generated local frame of 1D curved mesh
def local_frame_1D_manual(mesh_l):
    coord=mesh_l.geometry.x
    V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
        "DG", mesh_l.topology.cell_name(), 0, shape=(3, )))   
    c_to_v = mesh_l.topology.connectivity(mesh_l.topology.dim, 0)  # cell_to_vertex connectivity
    E2=Function(V_l) 
    for cell in range(num_cells_l):
        vertices = c_to_v.links(cell)
        dofs = V_l.dofmap.cell_dofs(cell) 
        n=coord[vertices[1]]-coord[vertices[0]] #          [x2-x1,y2-y1,z2-z1]
        E2.vector[3*cell:3*cell+3]=n/np.linalg.norm(n) 
    e1= as_vector([1,0,0]) # Global x-axis 
    e3=cross(e1,E2) 
    return e1,E2,e3
    
def local_frame_1D(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangent vector of 1D curved mesh
    e2=  t1/ sqrt(dot(t1, t1))
    e1=  as_vector([1,0,0]) # Right Lay up
    e3= cross(e1,e2)
    e3=  e3/ sqrt(dot(e3, e3)) 
    return e1, e2, e3
# Note grad(e2) will give derivatives of e2 in (x,yz,) global mesh frame. But, to obtain derivatives along local curvoilinear coordinates, 
# we use directional derivatives.
def deri(e): # derivatives of local frame (Refer: Prof: Yu thesis)
    #a3,1
    e1,e2,e3=e[0],e[1],e[2]
    a1_1=dot(e1,grad(e1))
    a1_2=dot(e2,grad(e1))   # directional derivative of e2 along e1 direction.
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

 ############################################################################
# We need four matrices for our energy form/weak form 
# EB requires only (Gamma_h and Gamma_e) matrix
# Timo needs (Gamma_l and Gamma_d) matrix. Note in paper, Gamma_l mentioend, but that does not contribute to Timoshenko like model.
# For input, local frame (e), spatial coordinate x and dx is required as Gamma_h for left boun/right boun/main qaud mesh would be different.
# Gamma_h*w column matrix
def gamma_h(e,x,w):    
    k11,k12,k21,k22,k13,k23= deri(e) # extracting initial curvatures
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
    G4=-k11*G1-k12*0.5*G3-ddot(w_d11,d3)+k13*ddot(w_d2,d3)-ddot(w_11,d3)
    G5=-k22*G2-k21*0.5*G3-ddot(w_d22,d3)-k23*ddot(w_d1,d3)-ddot(w_22,d3)
    G6=-(k11+k22)*0.5*G3-k12*G2-k21*G1+k23*ddot(w_d2,d3)-k13*ddot(w_d1,d3) \
       -ddot(w_d12,d3)-ddot(w_d21,d3)-ddot(w_12,d3)-ddot(w_21,d3)

    E1= as_tensor([G1,G2,G3,G4,G5,G6])
    return E1

def gamma_l(e,x,w): 
# e,x required as element can be of left/right boundary or quad mesh
    y1,y2,y3=x[2],x[0],x[1]  # In MSG-Shell formulations, y1 should be the beam axis & (y2,y3) as cross-sectional coordinates)
    # In mesh data, z coordinates are
    k11,k12,k21,k22,k13,k23= deri(e)
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1=[x11,x21,x31]
    d2=[x12,x22,x32]
    d3=[y1,y2,y3]
    dd1,dd2,dd3=as_vector(d1), as_vector(d2), as_vector(d3)
   # wd=[w[i].dx(0) for i in range(3)] # dx(0) is for w'
    
    # Gamma_l*w' column matrix     (w' defined by same function space as test/trail function
    #                                are basis functions which are same for both w and w')
    w_d1=as_vector([local_grad(dd1,w[ii]) for ii in range(3)])
    w_d2= as_vector([local_grad(dd2,w[ii]) for ii in range(3)])

    L1,L2=x11*(ddot(d1,w)),x12*(ddot(d2,w))
    L3=ddot(x11*dd2+x12*dd1,w)
    L4=-2*x11*ddot(w_d1,d3)+ddot(k11*(y1*dd3-x11*dd1)-0.5*k12*(x12*dd1+x11*dd2),w)
    L5=-2*x12*ddot(w_d2,d3)+ddot(k22*(y1*dd3-x12*dd2)-0.5*k21*(x12*dd1+x11*dd2),w)
    L6=-2*ddot(x11*w_d2+x12*w_d1,d3)+ddot(k12*(y1*dd3-x12*dd2)+k21*(y1*dd3-x11*dd1)-0.5*(k11+k22)*(x12*dd1+x11*dd2),w)
    return  as_tensor([L1,L2,L3,L4,L5,L6])
    
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

def gamma_d(e,x):
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    O=as_vector((0,0,0,0))
    R=as_vector((-y1,-y3*x[1]+y2*x[2],-x[2]*y1,x[1]*y1))
    return as_tensor([O,O,O,x11*x11*R,x12*x12*R,2*x11*x12*R])
    
# Visualization
import pyvista
pyvista.start_xvfb()
u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(mesh,mesh.topology.dim)
grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
grid.cell_data["Marker"] = subdomains.values[:]
grid.set_active_scalars("Marker")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(grid)
u_plotter.show_axes()
#u_plotter.view_xy() # z is beam axis
u_plotter.show()

# Obtaining preprocessing mesh terms for boundary as well as main wb (quad mesh).
# not to confuse with l here, it applies for right boundary as well by giving appropriate arguments
def local_boun(mesh_l,frame_l,subdomains_l):
    V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
        "S", mesh_l.topology.cell_name(), 2, shape=(3, )))
    le1, le2, le3 = frame_l
    e1l,e2l,e3l=Function(V_l), Function(V_l), Function(V_l)
    
    fexpr1=dolfinx.fem.Expression(le1,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
    e1l.interpolate(fexpr1) 
    
    fexpr2=dolfinx.fem.Expression(le2,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
    e2l.interpolate(fexpr2) 
    
    fexpr3=dolfinx.fem.Expression(le3,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
    e3l.interpolate(fexpr3) 
        
    return [e1l,e2l,e3l], V_l, TrialFunction(V_l), TestFunction(V_l),SpatialCoordinate(mesh_l), \
          Measure('dx')(domain=mesh_l, subdomain_data=subdomains_l)
    
def A_mat(e_l,x_l,dx_l,nullspace_l,v_l,dvl):
    F2 = sum([dot(dot(ABD_matrix(i),gamma_h(e_l,x_l,dvl)), gamma_h(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)])   
    A_l=assemble_matrix(form(F2))
    A_l.assemble()
    A_l.setNullSpace(nullspace_l) 
    return A_l

def solve_boun(mesh_l,frame_l,subdomains_l): # For applying bc, we only require solved fluctuating functions (V0) as input to bc.
    e, V_l, dv, v_, x, dx=local_boun(mesh_l,frame_l,subdomains_l)          
    mesh_l.topology.create_connectivity(1, 1)
    V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s=initialize_array(V_l)
    A_l=A_mat(e,x,dx,nullspace(V_l),v_,dv)

    for p in range(4):
        Eps=gamma_e(e,x)[:,p]
        F2 = sum([dot(dot(ABD_matrix(i),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
        r_he=form(rhs(F2))
        F_l = petsc.assemble_vector(r_he)
        F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        nullspace_l.remove(F_l)
        w_l=ksp_solve(A_l,F_l,V_l)
        V0[:,p]= w_l.vector[:] 
    return V0
    
def initialize_array(V_l):
    xxx=3*len(np.arange(*V_l.dofmap.index_map.local_range))  # total dofs 
    V0 = np.zeros((xxx,4))
    Dle=np.zeros((xxx,4))
    Dhe=np.zeros((xxx,4))
    Dhd=np.zeros((xxx,4))
    Dld=np.zeros((xxx,4))
    D_ed=np.zeros((4,4))
    D_dd=np.zeros((4,4))
    D_ee=np.zeros((4,4))  
    V1s=np.zeros((xxx,4))
    return V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s
    
def timo_boun(mesh_l,subdomains_l,frame_l):

    e, V_l, dv, v_, x, dx=local_boun(mesh_l,frame_l,subdomains_l)          
    mesh_l.topology.create_connectivity(1, 1)
    V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s=initialize_array(V_l)
    A_l=A_mat(e,x,dx,nullspace(V_l),v_,dv)
    
    for p in range(4):
        Eps=gamma_e(e,x)[:,p]
        F2 = sum([dot(dot(ABD_matrix(i),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
        r_he=form(rhs(F2))
        F_l = petsc.assemble_vector(r_he)
        F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        nullspace_l.remove(F_l)
        w_l=ksp_solve(A_l,F_l,V_l)
        Dhe[:,p]=  F_l[:]
        V0[:,p]= w_l.vector[:]  
    D1=np.matmul(V0.T,-Dhe)   
    for s in range(4):
        for k in range(4): 
            f=dolfinx.fem.form(sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
        
    D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)
    print('EB Computed')
    F1=sum([dot(dot(ABD_matrix(i),gamma_l(e,x,v_)),gamma_l(e,x,dv))*dx(i) for i in range(nphases)])
    a1=form(F1)
    Dll=assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av=Dll.getValuesCSR()
    Dll=csr_matrix((av, aj, ai)).toarray()
    
    for p in range(4):
            Eps=gamma_e(e,x)[:,p] 
            F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
            Dle[:,p]= petsc.assemble_vector(form(F1))[:]
        
            Eps=gamma_d(e,x)[:,p] 
            F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_h(e,x,v_))*dx(i) for i in range(nphases)])      
            Dhd[:,p]= petsc.assemble_vector(form(F1))[:]
        
            Eps=gamma_d(e,x)[:,p] 
            F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])      
            Dld[:,p]= petsc.assemble_vector(form(F1))[:]
        
    F_dhl=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
    a3=form(F_dhl)
    Dhl=assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av=Dhl.getValuesCSR()
    Dhl=csr_matrix((av, aj, ai)).toarray()
    for s in range(4):
        for k in range(4):    
            f=dolfinx.fem.form(sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_d(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_ed[s,k]=dolfinx.fem.assemble_scalar(f)
    
            f=dolfinx.fem.form(sum([dot(dot(gamma_d(e,x).T,ABD_matrix(i)),gamma_d(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_dd[s,k]=dolfinx.fem.assemble_scalar(f) 
        
    #DhlTV0
    DhlV0=np.matmul(Dhl.T,V0)
    
    #DhlTV0Dle
    DhlTV0Dle=np.matmul(Dhl,V0)+Dle
    
    #V0DllV0
    V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)
    
    # V1s
    b=DhlTV0Dle-DhlV0-Dhd
    
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
    
    return np.around(D_eff),np.around(Deff_srt)

# Initialize terms
e_l, V_l, dvl, v_l, x_l, dx_l=local_boun(mesh_l,frame_l ,subdomains_l)
e_r, V_r, dvr, v_r, x_r, dx_r=local_boun(mesh_r,frame_r ,subdomains_r)
nullspace_l, nullspace_r= nullspace(V_l),nullspace(V_r)
A_l,A_r=A_mat(e_l,x_l,dx_l,nullspace_l,v_l,dvl),A_mat(e_r,x_r,dx_r,nullspace_r,v_r,dvr)

# dof mapping makes solved unknown value w_l(Function(V_l)) assigned to v2a (Function(V)). 
# The boundary of wind blade mesh is a 1D curve. The facet/edge number is obtained from cell to edge connectivity (conn3) showed in subdomain subroutine.
# The same facet/edge number of extracted mesh_l (submesh) is obtaine din entity_mapl (gloabl mesh number). refer how submesh was generated.
#Therefore, once identifying the edge number being same for global(mesh)&boundary mesh(mesh_l), we equate the dofs and store w_l to v2a.
# The dofs can be verified by comparing the coordinates of local and global dofs if required. 
def dof_mapping_quad(v2a,V_l,w_ll,boundary_facets_left,entity_mapl):
    dof_S2L=[]
    for i,xx in enumerate(entity_mapl):
        dofs = locate_dofs_topological(V,1, np.array([xx]))
        dofs_left= locate_dofs_topological(V_l,1, np.array([boundary_facets_left[i]]))
        
        for k in range(deg+1):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.vector[3*dofs[k]+j]=w_ll[3*dofs_left[k]+j] # store boundary solution of fluctuating functions
    return v2a
V0_l,V0_r=solve_boun(mesh_l,local_frame_1D(mesh_l),subdomains_l),solve_boun(mesh_r,local_frame_1D(mesh_l),subdomains_r)
# The local_frame_l(mesh_l) can be replaced with frame_l, if we want to use mapped orientation from given direction cosine matrix (orien mesh data-yaml)

# Quad mesh
e,  V, dv,  v_,  x,    dx=local_boun(mesh,frame,subdomains)
V0,Dle,Dhe,Dhd,Dld,D_ed,D_dd,D_ee,V1s=initialize_array(V)
mesh.topology.create_connectivity(1, 2)
mesh_l.topology.create_connectivity(1, 1)
mesh_r.topology.create_connectivity(1, 1)

# Obtaining coefficient matrix AA and BB with and without bc applied.
# Note: bc is applied at boundary dofs. We define v2a containing all dofs of entire wind blade.
boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((entity_mapr,entity_mapl), axis=0))
F2=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])  
v2a=Function(V) # b default, v2a has zero value for all. 
bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs) # This shows only boundary_dofs are taken for v2a under bc, which are zero (known) as input.
a= form(F2)
B=assemble_matrix(a)   # Obtain coefficient matrix without BC applied: BB
B.assemble()
ai, aj, av=B.getValuesCSR()
BB=csr_matrix((av, aj, ai))
BB=BB.toarray()  
 
A=assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
A.assemble()
ai, aj, av=A.getValuesCSR()
AA=csr_matrix((av, aj, ai))
AA=AA.toarray()
avg=np.trace(AA)/len(AA)     
# 
for i,xx in enumerate(av):
    if xx==1:
        av[i]=avg        # averaging is done so that all terms are of same order. Note after appliying bc at [A=assemble_matrix(a,[bc])], the dofs of 
                         # coefficientmatrix has only 1 replaced at that dofs. 
AA_csr=csr_matrix((av, aj, ai))
AAA=AA_csr.toarray() 
AA=scipy.sparse.csr_matrix(AAA) 

# Assembly
# Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).
for p in range(4): # 4 load cases meaning 
    # Boundary 
    v2a=Function(V)
    v2a=dof_mapping_quad(v2a,V_l,V0_l[:,p],boundary_facets_left,entity_mapl) 
    v2a=dof_mapping_quad(v2a,V_r,V0_r[:,p],boundary_facets_right,entity_mapr)  
    
    # quad mesh
    F2=sum([dot(dot(ABD_matrix(i),gamma_e(e,x)[:,p]), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])  
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    F = petsc.assemble_vector(form(rhs(F2)))
    apply_lifting(F, [a], [[bc]]) # apply bc to rhs vector (Dhe)
    set_bc(F, [bc])
    for i in boundary_dofs:
        for k in range(3):
            F[3*i+k]=avg*F[3*i+k] # normalize small terms
            
    V0[:,p]=  scipy.sparse.linalg.spsolve(AA, F, permc_spec=None, use_umfpack=True) # obtain sol: E* V1s = b*
    Dhe[:,p]= scipy.sparse.csr_array(BB).dot(V0[:,p])
    
D1=np.matmul(V0.T,-Dhe) 
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
L=(x_max-x_min)
D_eff= D_ee + D1
D_eff=D_eff/L  # L is divided because of 3D shell mesh and corresponding beam length need to divided.
#--------------------------------------Printing Output Data---------------------------------------
print('  ')  
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(D_eff)) 

# We can also obtain the boundary timo 6X6 matrix as below
np.set_printoptions(precision=4)
print(timo_boun(mesh_r,subdomains_r,local_frame_1D(mesh_r))[1]) # Left Boundary timo

np.set_printoptions(precision=4)
print(timo_boun(mesh_l,subdomains_l,local_frame_1D(mesh_l))[1]) # Right Boundary timo
