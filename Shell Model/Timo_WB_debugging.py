#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


#############EB Shell model including orientation and Taper ##################
### Update for dolfinx latest v0.8
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
import scipy.sparse.linalg
import yaml
from yaml import CLoader as cLd

## Updated: Define input parameters

meshYaml = 'bar_urc_shell_mesh.yaml'  ## the name of the yaml file containing the whole blade mesh
secInd = np.linspace(1,1,1) ## the index of the spanwise section you want
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
        outFile.write(' '.join(ln) + '\n')

outFile.write('$EndNodes\n$Elements\n')

newNumEls = np.max(elNewLabs)
outFile.write(str(newNumEls) + '\n')
layupp=[]
for i, el in enumerate(elements):
    lab = elNewLabs[i]
    if(lab > -1):
        ln = [str(lab)]
        if(el[3] == -1):
            ln.append('2')
        else:
            ln.append('3')
        ln.append('2')
        ln.append('2')
        ln.append(str(elLayID[i]))
        layupp.append(int(elLayID[i]))
        for nd in el:
            if(nd > -1):
                ln.append(str(ndNewLabs[nd]))
        outFile.write(' '.join(ln) + '\n')
outFile.write('$EndElements\n')

outFile.close()
mesh, subdomains, boundaries = gmshio.read_from_msh("SG_shell.msh", MPI.COMM_WORLD,0, gdim=3)

o_cell_idx =  mesh.topology.original_cell_index
lnn=[]
for k in o_cell_idx:
    lnn.append(layupp[k])
lnn=np.array(lnn,dtype=np.int32)    

cell_map = mesh.topology.index_map(mesh.topology.dim)
num_cells_on_process = cell_map.size_local + cell_map.num_ghosts
cells = np.arange(num_cells_on_process, dtype=np.int32)
subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, lnn)
#for i in subdomains.values:
 #   print(i)

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

o_cell_id=mesh.topology.original_cell_index  # Original cell Index from mesh file

orien=[]
for i, eo in enumerate(meshData['elementOrientations']):
    elab = elNewLabs[i]
    if(elNewLabs[i] > -1):
        o=[]
        for k in range(9):
            o.append(eo[k])
        orien.append(o)
        
# Store orientation for each element
for k,ii in enumerate(o_cell_id):
    a1=np.array(orien[ii][0:3])   
    b1=np.array(orien[ii][3:6])
    c1=np.array(orien[ii][6:9]) 
    # Storing data to DG0 functions 
    EE1.vector[3*k],EE1.vector[3*k+1],EE1.vector[3*k+2]=a1   # e1
    EE2.vector[3*k],EE2.vector[3*k+1],EE2.vector[3*k+2]=b1  #  e2 
    N.vector[3*k], N.vector[3*k+1],N.vector[3*k+2]=c1        # e3 normal same as +x3 direction based on ABD.

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

# Generate ABD matrix (Plate model)

def ABD_mat(ii):
    deg = 2
    cell = ufl.Cell("interval")
    elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th,s=[(0,0,0)],0 # Reference-------- 0
    for k in thick[ii]:
        s=s-k
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
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V.dofmap.index_map_bs) for i in range(4)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [np.asarray(x) for x in vec_local]

    # Dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list for i in range(3)]

    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0

    # Build rotational null space basis
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    basis[3][dofs[1]] = -x2
    basis[3][dofs[2]] = x1
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

nphases=len(nlay)
# Store ABD matrices for layup (as list)
ABD_=[] 
for i in range(nphases):
    ABD_.append(ABD_mat(i))
print('Computed ABD matrix')

def ABD_matrix(i):
    return(as_tensor(ABD_[i]))


# In[3]:


# Local frame from OpenSG 
def local_frame(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangential direction 
    t2 = as_vector([t[0, 1], t[1, 1], t[2, 1]]) 
    e3 = cross(t1, t2) # normal direction 
    e3 /= sqrt(dot(e3, e3))
    e1=  t2/ sqrt(dot(t2, t2)) # 1- direction
    e2 = cross(e3, e1)
    e2 /= sqrt(dot(e2, e2)) # 2- direction
    return e1, e2, e3

# Local Frame for Boundary (should be same as quad mesh local frame)
def local_frame_1D(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangential direction
    e1=  t1/ sqrt(dot(t1, t1))
    e2=  as_vector([1,0,0])                    # beam-axis direction 
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

# 4 layer outer reference--
# Obtained for E= 3.44e+03, v= 0.3 (Isotropic) using 1D SG isotropic under plate model 
# Can also be evaulated with (h=1)
# A = E h/ (1-v^2)   [ 1    v   0;  v  1  0; 0   0  (1-v)/2]; B=0;
# D=  A (h^2/12);

def local_grad(ee,q):
    return dot(ee,grad(q))

def ddot(w,d1):
    return (d1[0]*w[0]+d1[1]*w[1]+d1[2]*w[2])
    
# Gamma_h*w column matrix
def gamma_h(e,x,w):    
    # e,x required as element can be of left/right boundary or quad mesh
    k11,k12,k21,k22,k13,k23= deri(e) # extracting initial curvatures
    # Generating C_ab matrix 
    #C_ab:[ x11   x21   x31
    #       x12   x22   x32
    #       y1    y2    y3 ]
    # Direction derivative (about Local Frame e1,e2,e3)
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
    
# Gamma_l*w' column matrix
def gamma_l(e,x,w): 
# e,x required as element can be of left/right boundary or quad mesh
    k11,k12,k21,k22,k13,k23= deri(e)
    x11,x21,x31= local_grad(e[0],x[0]), local_grad(e[0],x[1]), local_grad(e[0],x[2])
    x12,x22,x32= local_grad(e[1],x[0]), local_grad(e[1],x[1]), local_grad(e[1],x[2])
    y1,y2,y3= local_grad(e[2],x[0]), local_grad(e[2],x[1]), local_grad(e[2],x[2])
    d1=[x11,x21,x31]
    d2=[x12,x22,x32]
    d3=[y1,y2,y3]
   # wd=[w[i].dx(0) for i in range(3)] # dx(0) is for w'
    
    # Gamma_l*w' column matrix     (w' defined by same function space as test/trail function
    #                                are basis functions which are same for both w and w')
    L1=x11*(ddot(d1,w))   
    L2=x12*(ddot(d2,w)) 
    L3=ddot([2*x11*x12,(x12*x21+x11*x22),(x12*x31+x11*x32)],w)
    L4=ddot([(k11*(y1**2-x11**2)-k12*x11*x12), 
             (k11*(y1*y2-x11*x21)-0.5*k12*(x12*x21+x11*x22)), 
              (k11*(y1*y3-x11*x31)-k12*0.5*(x12*x31+x11*x32))],w)
    L5=ddot([(-k21*x11*x12+k22*(y1**2-x12**2)),
              (-k21*0.5*(x12*x21+x11*x22)+k22*(y1*y2-x12*x22)),
              (-k21*0.5*(x12*x31+x11*x32)+k22*(y1*y3-x12*x32))],w)
    L6=ddot([(k21*(y1**2-x11**2)-(k11+k22)*x11*x12+k12*(y1**2-x12**2)),
             (k21*(y1*y2-x11*x21)-0.5*(k11+k22)*(x21*x12+x11*x22)+k12*(y1*y2-x12*x22)),
             (k21*(y1*y3-x11*x31)-0.5*(k11+k22)*(x31*x12+x11*x32)+k12*(y1*y3-x12*x32))],w)
    return as_vector([L1,L2,L3,L4,L5,L6])

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
        R=as_vector((-y1,-y3*x[1]+y2*x[2],-x[2],x[1]))

        return as_tensor([O,O,O,x11*x11*R,x12*x12*R,x11*x12*R])


# In[4]:


# Optional arguments
options = ["-O1", "-O2", "-O3", "-Ofast"]
cffi_options = [options[3]]
jit_options = {"cffi_extra_compile_args": cffi_options,
                                "cffi_libraries": ["m"]}

deg=1
#-----------------Finite Element function Space-------------------------------------------
# Local Frame for mesh 
V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
    "S", mesh.topology.cell_name(), deg, shape=(3, )))
frame = local_frame(mesh) 
ve1, ve2, ve3= frame
e1,e2,e3=Function(V), Function(V), Function(V)

fexpr1=dolfinx.fem.Expression(ve1,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e1.interpolate(fexpr1)  # ve1 replace by EE1

fexpr2=dolfinx.fem.Expression(ve2,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e2.interpolate(fexpr2)  # ve2 replace by EE1

fexpr3=dolfinx.fem.Expression(ve3,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
e3.interpolate(fexpr3) # ve3 replace by EE1

dv = TrialFunction(V)
v_ = TestFunction(V)

# Left Boundary

V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
    "S", mesh_l.topology.cell_name(), deg, shape=(3, )))
w_l = Function(V_l)
dvl = TrialFunction(V_l)
v_l = TestFunction(V_l)
# Right Boundary

V_r = dolfinx.fem.functionspace(mesh_r, basix.ufl.element(
    "S", mesh_r.topology.cell_name(), deg, shape=(3, )))
dvr = TrialFunction(V_r)
v_r = TestFunction(V_r)

w_r = Function(V_r)
frame_l = local_frame_1D(mesh_l)
le1, le2, le3= frame_l

frame_r = local_frame_1D(mesh_r)
re1, re2, re3= frame_r

e1r,e2r,e3r=Function(V_r), Function(V_r), Function(V_r)
e1l,e2l,e3l=Function(V_l), Function(V_l), Function(V_l)
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

# Cell-Facet connectivity 
# Subdomain mapping to boundary mesh

mesh.topology.create_connectivity(2,1)
cell_of_facet_mesh=mesh.topology.connectivity(2,1)
# Cell to Edge connectivity
nelem=int(len(cell_of_facet_mesh.array)/4)
conn3=[]
for i in range(nelem):
    c=[]
    for k in range(4):
        c.append((cell_of_facet_mesh.array[4*i+k]))
    conn3.append(c)

#Left face
sub_L=[]
boundary_facets = dolfinx.mesh.locate_entities(mesh, fdim, left)
boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
for i,xx in enumerate(boundary_facets_left):
    
        for ii,k in enumerate(conn3):
            if entity_mapl[i] in k:
                sub_L.append(subdomains.values[ii])

sub_L= np.array(sub_L,dtype=np.int32)
cell_map_l = mesh_l.topology.index_map(mesh_l.topology.dim)
num_cells_on_process_l = cell_map_l.size_local + cell_map_l.num_ghosts
cells_l = np.arange(num_cells_on_process_l, dtype=np.int32)
subdomains_l = dolfinx.mesh.meshtags(mesh_l, mesh_l.topology.dim, cells_l, sub_L)

#Right face
sub_R=[]
boundary_facets = dolfinx.mesh.locate_entities(mesh, fdim, right)
boundary_facets_right= dolfinx.mesh.locate_entities(mesh_r, fdim, right)
for i,xx in enumerate(boundary_facets_right):
        for ii,k in enumerate(conn3):
            if entity_mapr[i] in k:
                sub_R.append(subdomains.values[ii])
                
sub_R= np.array(sub_R,dtype=np.int32)

cell_map_r = mesh_r.topology.index_map(mesh_r.topology.dim)
num_cells_on_process_r = cell_map_r.size_local + cell_map_r.num_ghosts
cells_r = np.arange(num_cells_on_process_r, dtype=np.int32)
subdomains_r = dolfinx.mesh.meshtags(mesh_r, mesh_r.topology.dim, cells_r, sub_R)

# In[155]:
mesh.topology.create_connectivity(1, 2)
            
boundary_facets= exterior_facet_indices(mesh.topology)
boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((entity_mapr,entity_mapl), axis=0))
e_l,e_r=[e1l,e2l,e3l], [e1r,e2r,e3r]
x_l,dx_l = SpatialCoordinate(mesh_l),Measure('dx')(domain=mesh_l, subdomain_data=subdomains_l)
x_r,dx_r = SpatialCoordinate(mesh_r),Measure('dx')(domain=mesh_r, subdomain_data=subdomains_r)
e=[e1,e2,e3]
x, dx=SpatialCoordinate(mesh), Measure('dx')(domain=mesh, subdomain_data=subdomains)


# In[5]:


v2a=Function(V)
F2 = sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
# bc 
bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)

# Obtain coefficient matrix without BC applied: BB
ahh= form(lhs(F2))
B=assemble_matrix(ahh)
B.assemble()
ai, aj, av=B.getValuesCSR()
BB=csr_matrix((av, aj, ai))
BB=BB.toarray()  
#-------------------------------------------------- 
# Obtain coefficient matrix with BC applied: AA
A=assemble_matrix(ahh,[bc])
A.assemble()
ai, aj, av=A.getValuesCSR()
AA=csr_matrix((av, aj, ai))
AA=AA.toarray()
avg=np.trace(AA)/len(AA)     
# 
for i,xx in enumerate(av):
    if xx==1:
        av[i]=avg      
AA_csr=scipy.sparse.csr_matrix((av, aj, ai))
AAA=AA_csr.toarray() 
AA=scipy.sparse.csr_matrix(AAA) 


# In[6]:


# Left Nullspace
def nullspace(V_l):
    index_map = V.dofmap.index_map
    nullspace_basis = [dolfinx.la.create_petsc_vector(index_map, V_l.dofmap.index_map_bs) for i in range(4)]
    
    with ExitStack() as stack:
        vec_local = [stack.enter_context(xx.localForm()) for xx in nullspace_basis]
        basis = [np.asarray(xx) for xx in vec_local]
    
    # Dof indices for each subspace (x, y and z dofs)
    dofs = [V_l.sub(i).dofmap.list for i in range(3)]
    
    # Build translational null space basis
    for i in range(3):
        basis[i][dofs[i]] = 1.0
    
    # Create vector space basis and orthogonalize
    dolfinx.la.orthonormalize(nullspace_basis)
    
    return petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
    
nullspace_l,nullspace_r=nullspace(V_l), nullspace(V_r)   

def A(e,x,dx,dv,v_,null):
    F2 = sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
    a2=form(F2)
    A_l=assemble_matrix(a2)
    A_l.assemble()
    A_l.setNullSpace(null)
    return A_l

A_l,A_r=A(e_l,x_l,dx_l,dvl,v_l,nullspace_l), A(e_r,x_r,dx_r,dvr,v_r,nullspace_r)
V0_l=np.zeros((3*V_l.dofmap.index_map.local_range[1],4))
V0_r=np.zeros((3*V_r.dofmap.index_map.local_range[1],4))                    


# In[7]:


def boun_V0(e,x,dx,dv,v_,V,A,p): 

    xxx= 3*V.dofmap.index_map.local_range[1]
    Eps=gamma_e(e,x)[:,p]
 #   F2 = sum([dot(dot(ABD_matrix(i),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
    F2= sum([dot(dot(ABD_matrix(i),Eps+gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
    r_he=form(rhs(F2))
    F_l = petsc.assemble_vector(r_he)
    F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    w_l=Function(V)
    
    nullspace_l.remove(F_l) # nullspace_l
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
    return w_l,F_l
    
mesh_l.topology.create_connectivity(1, 1)
mesh_r.topology.create_connectivity(1, 1)
global_facets_l,global_facets_r  = locate_entities(mesh, fdim, left), locate_entities(mesh, fdim, right)
local_facets_l,local_facets_r = locate_entities(mesh_l, fdim, left), locate_entities(mesh_r, fdim, right)

def dof_mapping(v2a,w_l,V_l,boundary_facets,boundary_facets_left):
        dof_S2L=[]
        for i,xx in enumerate(boundary_facets):
            dofs = locate_dofs_topological(V,1, np.array([xx]))
            dofs_left= locate_dofs_topological(V_l,1, np.array([boundary_facets_left[i]]))
    
            for k in range(deg+1):
                if dofs[k] not in dof_S2L:
                    dof_S2L.append(dofs[k])
                    l=3*dofs_left[k],3*dofs_left[k]+1, 3*dofs_left[k]+2
                    m=3*dofs[k],3*dofs[k]+1,3*dofs[k]+2
                    v2a.vector[m]=w_l.vector[l] # store boundary solution of fluctuating functions
        return v2a


# In[8]:


# Assembly
xxx=3*V.dofmap.index_map.local_range[1]
V0 = np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
D_ee=np.zeros((4,4))
#
Dle=np.zeros((xxx,4))
Dhd=np.zeros((xxx,4))
Dld=np.zeros((xxx,4))
D_ed=np.zeros((4,4))
D_dd=np.zeros((4,4))
V1s=np.zeros((xxx,4))

for p in range(4): # 4 load cases meaning 
    v2a=Function(V)
    # Left Boundary 
    w_l=boun_V0(e_l,x_l,dx_l,dvl,v_l,V_l,A_l,p)[0]
    V0_l[:,p]=w_l.vector[:]
    v2a=dof_mapping(v2a,w_l,V_l,global_facets_l,local_facets_l)

    # Right Boundary
    w_r=boun_V0(e_r,x_r,dx_r,dvr,v_r,V_r,A_r,p)[0]
    V0_r[:,p]=w_r.vector[:]
    v2a=dof_mapping(v2a,w_r,V_r,global_facets_r,local_facets_r)
    
    # Compute Quad mesh
    w=Function(V)
    Eps=gamma_e(e,x)[:,p] 
    F2 = sum([dot(dot(ABD_matrix(i),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)]) 
    # bc 
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    b= form(-F2)
#------------------------------------------------------------------------------------- 
    F = petsc.assemble_vector(b)
    apply_lifting(F, [ahh], [[bc]]) # apply bc to rhs vector (Dhe) based on known fluc solutions
    set_bc(F, [bc])
    FF=F[:]
    for i in boundary_dofs:
        for k in range(3):
            FF[3*i+k]=avg*F[3*i+k] # normalize small terms

    V0[:,p]= scipy.sparse.linalg.spsolve(AA, FF, permc_spec=None, use_umfpack=True) # obtain sol: E* V1s = b*
    Dhe[:,p]=  np.matmul(BB,V0[:,p]) # obtain b : E * V1s
    print('Computed EB:',25*(p+1),'%') 
D1=np.matmul(V0.T,-Dhe) 

for s in range(4):
    for k in range(4): 
        f=sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)])
        D_ee[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

        f=sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_d(e,x))[s,k]*dx(i) for i in range(nphases)])
        D_ed[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

        f=sum([dot(dot(gamma_d(e,x).T,ABD_matrix(i)),gamma_d(e,x))[s,k]*dx(i) for i in range(nphases)])
        D_dd[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))
        
D_eff= D_ee + D1
D_eff=D_eff/(x_max-x_min)
#-------------------------Printing Output Data-------------------------------
print('  ')  
print('EB Stiffness Matrix \n')
np.set_printoptions(precision=4)
print(np.around(D_eff))  

# Timoshenko beam Model begins

def boun_V1s(V_l,A_l,b_l,p):
    w_ll=Function(V_l)
    F=petsc4py.PETSc.Vec().createWithArray(b_l[:,p],comm=MPI.COMM_WORLD)
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    
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
    ksp.solve(F, w_ll.vector)
    w_ll.vector.ghostUpdate(
        addv=petsc4py.PETSc.InsertMode.INSERT, mode=petsc4py.PETSc.ScatterMode.FORWARD)
    ksp.destroy()
    return w_ll
    
def Timo_Matrix(e,x,dx,dv,v_,V,A,V0):
    xxx= 3*V.dofmap.index_map.local_range[1]
    Dle_l=np.zeros((xxx,4))
    Dhe_l=np.zeros((xxx,4))
    Dhd_l=np.zeros((xxx,4))

    F1=sum([dot(dot(ABD_matrix(i),gamma_l(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
    a1=form(F1)
    Dll_l=assemble_matrix(a1)
    Dll_l.assemble()
    ai, aj, av=Dll_l.getValuesCSR()
    Dll_l=csr_matrix((av, aj, ai)).toarray()
    for p in range(4):
            Eps=gamma_e(e,x)[:,p] 
            F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
            L1 = form(F1)   
            Dle_l[:,p]= petsc.assemble_vector(L1)[:]
            
            Eps=gamma_d(e,x)[:,p] 
            F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
            L1 = form(F1)          
            Dhd_l[:,p]= petsc.assemble_vector(L1)[:]

        
    F_dhl_l=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
    a3=form(F_dhl_l)
    Dhl_l=assemble_matrix(a3)
    Dhl_l.assemble()
    ai, aj, av=Dhl_l.getValuesCSR()
    Dhl_l=csr_matrix((av, aj, ai)).toarray()
    
    #DhlTV0
    DhlV0_l=np.matmul(Dhl_l.T,V0)
    
    #DhlTV0Dle
    DhlTV0Dle_l=np.matmul(Dhl_l,V0)+Dle_l
    
    #V0DllV0
    V0DllV0_l=np.matmul(np.matmul(V0.T,Dll_l),V0)
    
    # V1s
    b_l=DhlTV0Dle_l-DhlV0_l-Dhd_l
    return b_l
# Timo matrices assembly

Fll=sum([dot(dot(ABD_matrix(i),gamma_l(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
a1=form(Fll)
Dll=assemble_matrix(a1)
Dll.assemble()
ai, aj, av=Dll.getValuesCSR()
Dll=csr_matrix((av, aj, ai))
Dll=Dll.toarray()    

F_dhl=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
a3=form(F_dhl)
Dhl=assemble_matrix(a3)
Dhl.assemble()
ai, aj, av=Dhl.getValuesCSR()
Dhl=csr_matrix((av, aj, ai))
Dhl=Dhl.toarray()  

# Timo Matrices
for p in range(4):
    Eps=gamma_e(e,x)[:,p] 
    F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
    L1 = form(F1) 
    Dle[:,p]= petsc.assemble_vector(L1)[:] 
    
    Eps=gamma_d(e,x)[:,p] 
    F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
    L1 = form(F1) 
    Dhd[:,p]= petsc.assemble_vector(L1)[:]    

    Eps=gamma_d(e,x)[:,p] 
    F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])
    L1 = form(F1) 
    Dld[:,p]= petsc.assemble_vector(L1)[:]    
    
#DhlTV0
DhlV0=np.matmul(Dhl.T,V0)

#DhlTV0Dle
DhlTV0Dle=np.matmul(Dhl,V0)+Dle

#V0DllV0
V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)

# V1s
b=DhlTV0Dle-DhlV0-Dhd
b_l=Timo_Matrix(e_l,x_l,dx_l,dvl,v_l,V_l,A_l,V0_l)
b_r=Timo_Matrix(e_r,x_r,dx_r,dvr,v_r,V_r,A_r,V0_r)

# Assembly of unit load
for p in range(4): # 4 load cases meaning 
    v2a=Function(V)
    # Left Boundary 
    w_ll=boun_V1s(V_l,A_l,b_l,p)
    v2a=dof_mapping(v2a,w_ll,V_l,global_facets_l,local_facets_l)

    # Right Boundary
    w_rr=boun_V1s(V_r,A_r,b_r,p)
    v2a=dof_mapping(v2a,w_rr,V_r,global_facets_r,local_facets_r)
    
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    apply_lifting(F, [ahh], [[bc]]) # apply bc to rhs vector (-Dhe) based on known fluc solutions
    set_bc(F, [bc])
    FF=F[:]
    for i in boundary_dofs:
        for k in range(3):
            FF[3*i+k]=avg*F[3*i+k] # normalize small terms

    V1s[:,p]= scipy.sparse.linalg.spsolve(AA, FF, permc_spec=None, use_umfpack=True) # obtain sol: E* V1s = b*
    print('Computed V1s:',25*(p+1),'%') 

# Ainv
Ainv=np.linalg.inv(D_eff)

# B_tim
B_tim= np.matmul(DhlTV0Dle.T,V0)+ D_ed+ np.matmul(V0.T,Dhd)
B_tim=B_tim/(x_max-x_min)

# C_tim
C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle+Dhd)+D_dd+2*np.matmul(V0.T,Dld)
C_tim=0.5*(C_tim+C_tim.T)
C_tim=C_tim/(x_max-x_min)

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

print(' ')
print('Timoshenko Stiffness Matrix \n')

np.set_printoptions(precision=4)
print(np.around(Deff_srt))  


# In[ ]:





# In[ ]:


# Left Boundary Timo check
xxx= 3*V_l.dofmap.index_map.local_range[1]
Dhe_l=np.zeros((xxx,4))
D_ee_l=np.zeros((4,4))
D_ed_l=np.zeros((4,4))
D_dd_l=np.zeros((4,4))
for p in range(4): # 4 load cases meaning 
    v2a=Function(V)
    # Left Boundary 
    w_l,F_l=boun_V0(e_l,x_l,dx_l,dvl,v_l,V_l,A_l,p)
    V0_l[:,p],Dhe_l[:,p]=w_l.vector[:],F_l[:]
D1=np.matmul(V0_l.T,-Dhe_l) 

for s in range(4):
    for k in range(4): 
        f=sum([dot(dot(gamma_e(e_l,x_l).T,ABD_matrix(i)),gamma_e(e_l,x_l))[s,k]*dx_l(i) for i in range(nphases)])
        D_ee_l[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

        f=sum([dot(dot(gamma_d(e_l,x_l).T,ABD_matrix(i)),gamma_e(e_l,x_l))[s,k]*dx_l(i) for i in range(nphases)])
        D_ed_l[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

        f=sum([dot(dot(gamma_d(e_l,x_l).T,ABD_matrix(i)),gamma_d(e_l,x_l))[s,k]*dx_l(i) for i in range(nphases)])
        D_dd_l[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))        
D_eff= D_ee_l + D1
#-------------------------Printing Output Data-------------------------------
print('  ')  
print('Left Boundary: EB Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(D_eff))  
################################################################
# Left Boundary Timo

V1s_l=np.zeros((xxx,4))
Dhe_l=np.zeros((xxx,4))
Dhd_l=np.zeros((xxx,4))
Dld_l=np.zeros((xxx,4))
Dle_l=np.zeros((xxx,4))
F1=sum([dot(dot(ABD_matrix(i),gamma_l(e_l,x_l,v_l)),gamma_l(e_l,x_l,dvl))*dx_l(i) for i in range(nphases)])
a1=form(F1)
Dll_l=assemble_matrix(a1)
Dll_l.assemble()
ai, aj, av=Dll_l.getValuesCSR()
Dll_l=csr_matrix((av, aj, ai)).toarray()
for p in range(4):
        Eps=gamma_e(e_l,x_l)[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)])
        L1 = form(F1)   
        Dle_l[:,p]= petsc.assemble_vector(L1)[:]
    
        Eps=gamma_d(e_l,x_l)[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_h(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)])
        L1 = form(F1)         
        Dhd_l[:,p]= petsc.assemble_vector(L1)[:]

        Eps=gamma_d(e_l,x_l)[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)])
        L1 = form(F1)         
        Dld_l[:,p]= petsc.assemble_vector(L1)[:]
    
F_dhl_l=sum([dot(dot(ABD_matrix(i),gamma_h(e_l,x_l,dvl)),gamma_l(e_l,x_l,v_l))*dx_l(i) for i in range(nphases)]) 
a3=form(F_dhl_l)
Dhl_l=assemble_matrix(a3)
Dhl_l.assemble()
ai, aj, av=Dhl_l.getValuesCSR()
Dhl_l=csr_matrix((av, aj, ai)).toarray()

#DhlTV0
DhlV0_l=np.matmul(Dhl_l.T,V0_l)

#DhlTV0Dle
DhlTV0Dle_l=np.matmul(Dhl_l,V0_l)+Dle_l

#V0DllV0
V0DllV0_l=np.matmul(np.matmul(V0_l.T,Dll_l),V0_l)

# V1s
b_l=(DhlTV0Dle_l-DhlV0_l-Dhd_l)
for p in range(4): # 4 load cases meaning 
    v2a=Function(V)
    # Left Boundary 
    w_ll=boun_V1s(V_l,A_l,b_l,p)
    V1s_l[:,p]=w_ll.vector[:]
# Ainv
Ainv=np.linalg.inv(D_eff)

# B_tim
B_tim= np.matmul(DhlTV0Dle_l.T,V0_l)+ np.matmul(V0_l.T,Dhd_l)+D_ed_l

# C_tim
C_tim= V0DllV0_l + np.matmul(V1s_l.T,DhlV0_l + DhlTV0Dle_l)+np.matmul(V1s_l.T,Dhd_l)+D_dd_l+2*np.matmul(V0_l.T,Dld_l)
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

print('Left Boundary: Timoshenko Stiffness Matrix \n')

np.set_printoptions(precision=4)
print(np.around(Deff_srt))    


# In[ ]:


# Right Boundary Timo check
xxx= 3*V_r.dofmap.index_map.local_range[1]
Dhe_r=np.zeros((xxx,4))
D_ee_r=np.zeros((4,4))
D_dd_r=np.zeros((4,4))
D_ed_r=np.zeros((4,4))
for p in range(4): # 4 load cases meaning 
    v2a=Function(V)
    # Left Boundary 
    w_r,F_r=boun_V0(e_r,x_r,dx_r,dvr,v_r,V_r,A_r,p)
    V0_r[:,p],Dhe_r[:,p]=w_r.vector[:],F_r[:]
    
D1=np.matmul(V0_r.T,-Dhe_r) 

for s in range(4):
    for k in range(4): 
        f=sum([dot(dot(gamma_e(e_r,x_r).T,ABD_matrix(i)),gamma_e(e_r,x_r))[s,k]*dx_r(i) for i in range(nphases)])
        D_ee_r[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

        f=sum([dot(dot(gamma_d(e_r,x_r).T,ABD_matrix(i)),gamma_d(e_r,x_r))[s,k]*dx_r(i) for i in range(nphases)])
        D_dd_r[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

        f=sum([dot(dot(gamma_d(e_r,x_r).T,ABD_matrix(i)),gamma_e(e_r,x_r))[s,k]*dx_r(i) for i in range(nphases)])
        D_ed_r[s,k]=dolfinx.fem.assemble_scalar(dolfinx.fem.form(f))

D_eff_r= D_ee_r + D1
#-------------------------Printing Output Data-------------------------------
print('  ')  
print('Right Boundary: EB Stiffness Matrix \n')
np.set_printoptions(precision=4)
print(np.around(D_eff_r))  
################################################################
# Left Boundary Timo

V1s_r=np.zeros((xxx,4))
Dhe_r=np.zeros((xxx,4))
Dhd_r=np.zeros((xxx,4))
Dld_r=np.zeros((xxx,4))
Dle_r=np.zeros((xxx,4))
F1=sum([dot(dot(ABD_matrix(i),gamma_l(e_r,x_r,v_r)),gamma_l(e_r,x_r,dvr))*dx_r(i) for i in range(nphases)])
a1=form(F1)
Dll_r=assemble_matrix(a1)
Dll_r.assemble()
ai, aj, av=Dll_r.getValuesCSR()
Dll_r=csr_matrix((av, aj, ai)).toarray()
for p in range(4):
        Eps=gamma_e(e_r,x_r)[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e_r,x_r,v_r))*dx_r(i) for i in range(nphases)])
        L1 = form(F1)   
        Dle_r[:,p]= petsc.assemble_vector(L1)[:]
    
        Eps=gamma_d(e_r,x_r)[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_h(e_r,x_r,v_r))*dx_r(i) for i in range(nphases)])
        L1 = form(F1)         
        Dhd_r[:,p]= petsc.assemble_vector(L1)[:]

        Eps=gamma_d(e_r,x_r)[:,p] 
        F1=sum([dot(dot(ABD_matrix(i),Eps),gamma_l(e_r,x_r,v_r))*dx_r(i) for i in range(nphases)])
        L1 = form(F1)         
        Dld_r[:,p]= petsc.assemble_vector(L1)[:]
    
F_dhl_r=sum([dot(dot(ABD_matrix(i),gamma_h(e_r,x_r,dvr)),gamma_l(e_r,x_r,v_r))*dx_r(i) for i in range(nphases)]) 
a3=form(F_dhl_r)
Dhl_r=assemble_matrix(a3)
Dhl_r.assemble()
ai, aj, av=Dhl_r.getValuesCSR()
Dhl_r=csr_matrix((av, aj, ai)).toarray()

#DhlTV0
DhlV0_r=np.matmul(Dhl_r.T,V0_r)

#DhlTV0Dle
DhlTV0Dle_r=np.matmul(Dhl_r,V0_r)+Dle_r

#V0DllV0
V0DllV0_r=np.matmul(np.matmul(V0_r.T,Dll_r),V0_r)

# V1s
b_r=DhlTV0Dle_r-DhlV0_r-Dhd_r
for p in range(4): # 4 load cases meaning 
    v2a=Function(V)
    # Left Boundary 
    w_rr=boun_V1s(V_r,A_r,b_r,p)
    V1s_r[:,p]=w_rr.vector[:]
# Ainv
Ainv=np.linalg.inv(D_eff_r)

# B_tim
B_tim= np.matmul(DhlTV0Dle_r.T,V0_r)+ np.matmul(V0_r.T,Dhd_r)+D_ed_r

# C_tim
C_tim= V0DllV0_r + np.matmul(V1s_r.T,DhlV0_r + DhlTV0Dle_r)+np.matmul(V1s_r.T,Dhd_r)+D_dd_r+2*np.matmul(V0_r.T,Dld_r)
C_tim=0.5*(C_tim+C_tim.T)

# Ginv
Q_tim=np.matmul(Ainv,np.array([(0,0),(0,0),(0,-1),(1,0)]))
Ginv= np.matmul(np.matmul(Q_tim.T,(C_tim-np.matmul(np.matmul(B_tim.T,Ainv),B_tim))),Q_tim)
G_tim=np.linalg.inv(Ginv)
Y_tim= np.matmul(np.matmul(B_tim.T,Q_tim),G_tim)
A_tim= D_eff_r + np.matmul(np.matmul(Y_tim,Ginv),Y_tim.T)

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

print('Right Boundary: Timoshenko Stiffness Matrix \n')

np.set_printoptions(precision=4)
print(np.around(Deff_srt))


# In[ ]:





# In[ ]:





# In[ ]:




