
#############EB Shell model including orientation and Taper ##################

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


# In[144]:


import yaml
from yaml import CLoader as cLd

## Updated: Define input parameters

meshYaml = 'bar_urc_shell_mesh.yaml'  ## the name of the yaml file containing the whole blade mesh
secInd = np.linspace(1,2,2) ## the index of the spanwise section you want
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
        for nd in el:
            if(nd > -1):
                ln.append(str(ndNewLabs[nd]))
        outFile.write(' '.join(ln) + '\n')
outFile.write('$EndElements\n')

outFile.close()
mesh, subdomains, boundaries = gmshio.read_from_msh("SG_shell.msh", MPI.COMM_WORLD,0, gdim=3)


# In[145]:


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


# In[146]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[147]:


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


# In[ ]:





# In[148]:


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


# In[149]:


# Store ABD matrices for layup (as list)
ABD_=[] 
for i in range(nphases):
    ABD_.append(ABD_mat(i))
print('Computed ABD matrix')

def ABD_matrix(i):
    return(as_tensor(ABD_[i]))


# In[ ]:





# In[ ]:





# In[150]:


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


# In[151]:


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

def sigma(v,i,Eps,e,x):     
    s1= dot(ABD_matrix(i),eps(e,x,v)+Eps)
    return s1 
    
# Gamma_h*w column matrix
def eps(e,x,w):    
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
def gamma_l(e,x,dx,w): 
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
    
# Gamma_e matrix   
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


# In[152]:


# Checking curvature or local frame:
#V = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
  #  "S", mesh_l.topology.cell_name(), deg, shape=(1, )))

#ee=Function(V)

#fexpr1=dolfinx.fem.Expression(e1[0],V.element.interpolation_points(), comm=MPI.COMM_WORLD)
#ee.interpolate(fexpr1) 
#ee.vector[:]


# In[153]:


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
# Right Boundary

V_r = dolfinx.fem.functionspace(mesh_r, basix.ufl.element(
    "S", mesh_r.topology.cell_name(), deg, shape=(3, )))

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


# In[154]:


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

v2a=Function(V)
e=[e1,e2,e3]
x, dx= SpatialCoordinate(mesh), Measure('dx')(domain=mesh, subdomain_data=subdomains)
Eps=Eps2(e,x)[:,0] 
F2 = sum([dot(sigma(dv, i, Eps,e,x), eps(e,x,v_))*dx(i) for i in range(nphases)])
# bc 
bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
# Obtain coefficient matrix without BC applied: BB
a= form(lhs(F2))
B=assemble_matrix(a)
B.assemble()
ai, aj, av=B.getValuesCSR()
BB=csr_matrix((av, aj, ai))
BB=BB.toarray()  
#-------------------------------------------------- 
# Obtain coefficient matrix with BC applied: AA
A=assemble_matrix(a,[bc])
A.assemble()
ai, aj, av=A.getValuesCSR()
AA=csr_matrix((av, aj, ai))
AA=AA.toarray()
avg=np.trace(AA)/len(AA)     
# 
for i,x in enumerate(av):
    if x==1:
        av[i]=avg      
AA_csr=scipy.sparse.csr_matrix((av, aj, ai))
AAA=AA_csr.toarray() 
AA=scipy.sparse.csr_matrix(AAA) 


############# Left Nullspace ###################
dvl = TrialFunction(V_l)
v_l= TestFunction(V_l)
x=SpatialCoordinate(mesh_l)
e_l=[e1l,e2l,e3l]
dx = Measure('dx')(domain=mesh_l, subdomain_data=subdomains_l)
#
Eps= Eps2(e_l,x)[:,0]
F2 = sum([dot(sigma(dvl, 0, Eps,e_l,x), eps(e_l,x,v_l))*dx(i) for i in range(nphases)])
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
dx = Measure('dx')(domain=mesh_r, subdomain_data=subdomains_r)
#
Eps= Eps2(e_r,x)[:,0]
F2 = sum([dot(sigma(dvr, 0, Eps,e_r,x), eps(e_r,x,v_r))*dx(i) for i in range(nphases)])
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
    
# Create vector space basis and orthogonalize
dolfinx.la.orthonormalize(nullspace_basis)

nullspace_r = petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
assert nullspace_r.test(A_r)
# Set the nullspace
A_r.setNullSpace(nullspace_r)

xxx=3*V.dofmap.index_map.local_range[1]
V0 = np.zeros((xxx,4))
Dhe=np.zeros((xxx,4))
D_ee=np.zeros((4,4))

mesh.topology.create_connectivity(1, 2)
mesh_l.topology.create_connectivity(1, 1)
mesh_r.topology.create_connectivity(1, 1)


# In[156]:


# Assembly
for p in range(4): # 4 load cases meaning 
    # Left Boundary 
    v2a=Function(V)
    e=[e1l,e2l,e3l]
    x, dx=SpatialCoordinate(mesh_l), Measure('dx')(domain=mesh_l, subdomain_data=subdomains_l)
    Eps=Eps2(e,x)[:,p] 
    F2 = sum([dot(sigma(dvl, i, Eps,e,x), eps(e,x,v_l))*dx(i) for i in range(nphases)])
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
    #dof_mapping (left)
    dof_S2L=[]
    boundary_facets = dolfinx.mesh.locate_entities(mesh, fdim, left)
    boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
    
    for i,xx in enumerate(boundary_facets):
        dofs = locate_dofs_topological(V,1, np.array([entity_mapl[i]]))
        dofs_left= locate_dofs_topological(V_l,1, np.array([boundary_facets_left[i]]))

        for k in range(deg+1):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                l=3*dofs_left[k],3*dofs_left[k]+1, 3*dofs_left[k]+2
                m=3*dofs[k],3*dofs[k]+1,3*dofs[k]+2
                v2a.vector[m]=w_l.vector[l] # store boundary solution of fluctuating functions
    ##################################################################################                
    # Right
    e=[e1r,e2r,e3r]
    x, dx=SpatialCoordinate(mesh_r), Measure('dx')(domain=mesh_r, subdomain_data=subdomains_r)
    Eps=Eps2(e,x)[:,p] 
    F2 = sum([dot(sigma(dvr, i, Eps,e,x), eps(e,x,v_r))*dx(i) for i in range(nphases)])
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
    
    #dof_mapping (right)
    dof_S2R=[]
    boundary_facets = dolfinx.mesh.locate_entities(mesh, fdim, right)
    boundary_facets_right= dolfinx.mesh.locate_entities(mesh_r, fdim, right)
    for i,xx in enumerate(boundary_facets):
        dofs = dolfinx.fem.locate_dofs_topological(V,1, np.array([entity_mapr[i]]))
        dofs_right= dolfinx.fem.locate_dofs_topological(V_r,1, np.array([boundary_facets_right[i]]))

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
    x, dx=SpatialCoordinate(mesh), Measure('dx')(domain=mesh, subdomain_data=subdomains)
    Eps=Eps2(e,x)[:,p] 
    F2 = sum([dot(sigma(dv, i, Eps,e,x), eps(e,x,v_))*dx(i) for i in range(nphases)])
    
    # bc 
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    a,b= form(lhs(F2)), form(rhs(F2))
#------------------------------------------------------------------------------------- 
    F = petsc.assemble_vector(b)
    apply_lifting(F, [a], [[bc]]) # apply bc to rhs vector (Dhe) based on known fluc solutions
    set_bc(F, [bc])
    FF=F[:]
    for i in boundary_dofs:
        for k in range(3):
            FF[3*i+k]=avg*F[3*i+k] # normalize small terms

    V1s= scipy.sparse.linalg.spsolve(AA, FF, permc_spec=None, use_umfpack=True) # obtain sol: E* V1s = b*
    Dhe[:,p]=  np.matmul(BB,V1s) # obtain b : E * V1s
    V0[:,p]= V1s # V0 matrix formation  
    print('Computed:',25*(p+1),'%') 
D1=np.matmul(V0.T,-Dhe) 

for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([dot(dot(Eps2(e,x).T,ABD_matrix(i)),Eps2(e,x))[s,k]*dx(i) for i in range(nphases)]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)

D_eff= D_ee + D1
D_eff=D_eff/(x_max-x_min)

#-------------------------Printing Output Data-------------------------------
print('  ')  
print('Stiffness Matrix')
np.set_printoptions(precision=4)
print(np.around(D_eff))  





