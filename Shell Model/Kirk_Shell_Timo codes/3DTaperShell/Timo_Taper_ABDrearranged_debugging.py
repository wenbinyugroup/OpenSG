
######################Timoshenko Beam Model (Tapered Shell Wind Blade Segment)###################
#Input: .yaml File
#Output: Timoshenko Stiffness Matrix:
# a) Boundary Timoshenko Stiffness (Deff_l/Deff_r)
# b) Entire Blade Segment Timoshenko Stiffness (Deff_srt)
# Byproduct and intermediate step of computation as:
# c) Boundary Euler-Bernoulli Stiffness (D_effEB_l/D_effEB_r)
# d) Entire Blade Segment Euler-Bernoulli  Stiffness (D_eff)

# Input variables/data: 
# section_id= Constant           ! Segment_id of wind blade to be computed
# mesh= mesh data of entire wind blade (WB) segment 
# subdomains= layup id of mesh elements ( a.k.a. physical domain of mesh)
# boundaries= facet id of boundary elements of WB mesh  ! Not useful in computation
# x_min,x_max= minimum/maximum x (beam axis) coordinate to define boundary mesh
# tdim,fdim= mesh topology/facet dimension !   For quad mesh: tdim=2, fdim=1
# facets_left/right= Left/Right boundary facet id of WB mesh (In WB numbering) ! useful in generating boundary mesh
# mesh_r/mesh_l= left and right boundary mesh ( Generated in dolfinx as submesh)
# entity_mapl/mapr= facet id of left/right boudnary (In boundary mesh numbering)
# num_cells = number of quadrilateral mesh elements
# o_cell_idx = Original mesh element connectivity (SG_Shell.msh). Note: dolfinx does the renumbering of vertex and mesh connectivity
# VV = functionspace (Discontinuous galerkin) to store orientation data
# frame ([EE1,EE2,N]) = Local Orientation frame (from .yaml) stored as ufl function
# ! [EE1,EE2,N]: EE1-> curvilinear tangent direction (along beam axis), EE2-> Circumferential tangent Direction  (CCW about  +x (beam) axis), N -> Inward normal direction
# material_parameters = Stores anisotropic material properties of layups ! Same for all WB Segments


# nphases = number of layups in the WB segment
# nlay =  [nphases,1]                ! number of layers in each layup of WB segment
# matid = [nphases,nlay,1]             ! Contains thickness data of layup id- nlay
# thick = [nphases,nlay,1]
# conn3 = [num_cells,4]              ! facet connecticty matrix for each element
# subdomains_l = layup id of left boundary mesh ! Output for subdomains.values[:] arranged in boundary dofs
# frame_l = Local orientation frame (orthogonal) for each left boundary element
# ![E1l,E2l,Nl]: E1l-> beam axis (+x dir), E2l-> Circumferential tangent Direction  (CCW about  +x (beam) axis), E3l -> Inward normal direction
# boundary_facets_left =  [num_facets_left_boundary,1]     !facet id connectivity of boundary mesh (In mesh_l based numbering)
# D_effEB_l/D_effEB_r =  Euler-Bernoulli matrix of left/right boundaryy of WB Segment
# Deff_l/Deff_r = Timoshenko Stiffness matrix of left/right boundary of WB Segment


# ndofs_WBmesh = [3*len(np.arange(*V.dofmap.index_map.local_range)),1]  ! total dofs of WB segment mesh (Note: V=> functionspace of WB mesh)
# V0_l/V0_r=  [ndofs_leftmesh,4]  ! Fluctuating function solution after solving 4 load cases (for EB Stiffness/a.k.a zeroth order) 
# V1_l/V1_r=  [ndofs_leftmesh,4]  ! Fluctuating function solution after solving 4 load cases (for Timo Stiffness/ a.k.a first order) 
# e_l = Local orientation Frame for left boundary mesh interpolated in V_l (Serendipidity) functionspace
# e_r = Local orientation Frame for right boundary mesh interpolated in V_r (Serendipidity) functionspace
# e =  Local orientation Frame for WB Segment mesh interpolated in V (Serendipidity) functionspace
# V_l/V_r = functionspace of left/right boundary mesh (UFL)
# V= functionspace for defining Wb segment mesh (UFL)
# dvl,v_l= Trial and Test Functions for left boundary (V_l) (UFL)
# dvr,v_r= Trial and Test Functions for right boundary (V_r) (UFL)
# dv,v_= Trial and Test Functions for WB mesh (V) (UFL)
# x_l,x_r = Spatial Coordinate Data for left/right boundary (UFL)
# dx_l = Measure for left boundary mesh with subdomains_l assigned 
# dx = Measure for WB mesh with subdomains assigned 
# ! Used in defining weak form *dx(i) mean integration over mesh elements with layup id  i.
# nullspace_l = contain nullspace vector for used as constraint to block Rigid body motion when solving variation form (ksp solver) over left boundary mesh.
# ndofs_leftmesh=  [3*len(np.arange(*V.dofmap.index_map.local_range)),1]  ! total dofs of left boundary mesh 
# A_l= [ndofs_leftmesh,ndofs_leftmesh]   !Global Assembly Coefficient matrix (in ufl form) for left boundary 
# ! Can print matrix as array by A_l[:,:]
# ! Used in solving 4 load cases where A_l w_l = F_l ( similar to Ax=b) 
# ! Unknown w_l is stored in V0_l[:,p] for load case p
# A = [ndofs_WBmesh,ndofs_WBmesh]    !Global Assembly matrix for WB mesh 
# a= Bilinear weak form for WB mesh ! a(dv,v_) 
# F2= Linear weak form for WB mesh  ! F2(v_) 
# Can compare with weak form to solve as ! a(dv,v_) w = F2(v_), 
# where w.vector[ndofs_WBmesh,1] is unknown dofs value stored in V0[ndofs_WBmesh,4] 
# F= [ndofs_WBmesh,1]     ! Global Assembled Right hand vector for EB case
# b =[ndofs_WBmesh, 4] ! Equivalent to Right hand vector F for Timo case
# v0 = Stores the solved V0[ndofs_WBmesh,p] in ufl function form for each load case
# v2a = UFL function defined over V (functionspace_WB mesh) ! Output by v2a.vector[:]
# bc= dirichilet bc (known ufl function storing boundary solutions, dofs of boundary need to be constrained) 
# Dhe = [ndofs_WBmesh,4] ! < gamma_h.T ABD gamma_e>
# Dle = [ndofs_WBmesh,4] ! < gamma_l.T ABD gamma_e>
# Dll = [ndofs_WBmesh,ndofs_WBmesh] ! < gamma_l.T ABD gamma_l>
# Dhl = [ndofs_WBmesh,ndofs_WBmesh] ! < gamma_h.T ABD gamma_l>
# D_ee = [4,4]    ! < gamma_e.T ABD gamma_e>
# D1= [4,4]      ! < V0.T -Dhe> ! Information matrix from fluctuating function data 
# V0= [ndofs_WBmesh, 4] ! Solve [A w =F] ! Fluctuating function after solving 4 load cases on WB mesh (EB)
# V1s = [ndofs_WBmesh, 4] !Solve [A w =b] ! Fluctuating function after solving 4 load cases on WB mesh (Timo)
# D_eff=[4,4] ! Euler-Bernoulli Stiffness matrix of entire WB segment 
# Deff_srt=[6,6] ! Timoshenko Stiffness matrix of entire WB segment 

# Note: Dhe, Dle, Dll, Dhl, Dee for Boundary Timo Solving, (dofs_leftmesh/dofs_rightmesh) replaces dofs_WBmesh 


### Compatible with dolfinx latest v0.8
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh,CellType, GhostMode
from mpi4py import MPI
import numpy as np 
import meshio
import dolfinx
import ufl
from dolfinx.fem import form, petsc, Function, functionspace, locate_dofs_topological, apply_lifting, set_bc
from ufl import Jacobian, as_vector, dot, cross,sqrt, conditional, replace, as_matrix,FacetNormal
from ufl import lt,SpatialCoordinate, as_tensor, Measure, Mesh,CellDiameter, FacetNormal, avg, div, dS, dx, grad, inner
from ufl import TrialFunction, TestFunction, inner, lhs, rhs, dx, dot, eq, grad, jump, pi, sin
import petsc4py.PETSc
from contextlib import ExitStack
from dolfinx.io import gmshio
from mpi4py import MPI
from pathlib import Path
from typing import Dict
import scipy
from scipy.sparse import csr_matrix
import scipy.sparse.linalg                                        
import basix
import yaml
from yaml import CLoader as cLd
import basix.ufl
import dolfinx as dfx
from petsc4py import PETSc
import gmsh
## Updated: Define input parameters
section_id=[1] ## the index of the spanwise sections you want inside braces
meshYaml = 'BAR_URC_numEl_52.yaml'  ## the name of the yaml file containing the whole blade mesh
secInd = section_id 
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


# Material Data Extarct
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
combined_rows = list(zip(thick, matid, angle))


# In[2]:


ii = 0
unique_rows=[]
for es in meshData['sets']['element']:
    nmLst = es['name'].split('_')
    if(len(nmLst) > 2):
        if(int(nmLst[1]) in secInd):
            if combined_rows[ii] not in unique_rows:
             #   layCt+=1
                unique_rows.append(combined_rows[ii])

            lay_num=unique_rows.index(combined_rows[ii])    
            for eli in es['labels']:
                elNewLabs[eli] = 1
                elLayID[eli] = lay_num
                for nd in elements[eli]:
                    if(nd > -1):
                        ndNewLabs[nd] = 1
            ii+=1
       #      layCt += 1


# In[3]:


thick,matid,angle=list(zip(*unique_rows))[0], list(zip(*unique_rows))[1], list(zip(*unique_rows))[2]


# In[4]:


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
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)    # mesh read output will not be printed
mesh, subdomains, boundaries = gmshio.read_from_msh("SG_shell.msh", MPI.COMM_WORLD,0, gdim=3)
lnn=subdomains.values[:]-1
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local 
cells = np.arange(num_cells, dtype=np.int32)
subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, np.array(lnn,dtype=np.int32))
o_cell_idx= mesh.topology.original_cell_index


# Local Orientation (DG0 function) of quad mesh element (from yaml data)
#VV = dolfinx.fem.functionspace(mesh, basix.ufl.element(
#    "DG", mesh.topology.cell_name(), 0, shape=(3, )))
#EE1,EE2,N=Function(VV),Function(VV),Function(VV) 

#o_cell_idx=mesh.topology.original_cell_index  # Original cell Index from mesh file
#orien=[]
#for i, eo in enumerate(meshData['elementOrientations']):
   # elab = elNewLabs[i]
   # if(elNewLabs[i] > -1):
     #   o=[]
      #  for k in range(9):
      #      o.append(eo[k])
     #   orien.append(o)

# Store orientation for each element
#for k,ii in enumerate(o_cell_idx):
# Storing data to DG0 functions 
  #  EE2.vector[3*k],EE2.vector[3*k+1],EE2.vector[3*k+2]=orien[ii][5],orien[ii][3],orien[ii][4]  # e2
  #  N.vector[3*k],N.vector[3*k+1],N.vector[3*k+2]=orien[ii][8],orien[ii][6],orien[ii][7]   #  e3 
  #  EE1.vector[3*k], EE1.vector[3*k+1],EE1.vector[3*k+2]=orien[ii][2],orien[ii][0],orien[ii][1]  # e1    outward normal 
#frame=[EE1,EE2,N]

# Geometry Extraction 
pp=mesh.geometry.x                 # point data
x_min,x_max=min(pp[:,0]), max(pp[:,0])

def left(x):
    return np.isclose(x[0], x_min,atol=0.05) #************Update added tolerance term*******************
def right(x):
    return np.isclose(x[0], x_max,atol=0.05)#************Update added tolerance term*******************

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
    """
    Map the subdomains and local orientation farme from WB Segment mesh elements to boundary mesh
    
    Parameters:
        mesh_l: Left boundary mesh
        left:  Left/Right boundary marker
        entity_mapl: Left facets id of WB segment mesh
    Returns:
        subdomains_l: Left boundary mapped subdomains
        frame_l: Left Boundary local orientation frame (orthogonal)
        boundary_facets_left: Facets id of Left Boundary
    """    
   # VV_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
   #     "DG", mesh_l.topology.cell_name(), 0, shape=(3, )))
   # El1,El2,Nl=Function(VV_l),Function(VV_l),Function(VV_l)
    sub_L=[]
    boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
    for i,xx in enumerate(entity_mapl):
             idx=int(np.where(conn==xx)[0]/4)   # 4 is for number of nodes in quad element
             sub_L.append(subdomains.values[idx])
          #   for j in range(3):
           #             El1.vector[3*i+j]=EE1.vector[3*idx+j]
         #               El2.vector[3*i+j]=EE2.vector[3*idx+j]
         #               Nl.vector[3*i+j] =N.vector[3*idx+j]
   # frame=El1,El2,Nl                    
    sub_L= np.array(sub_L,dtype=np.int32)
    num_cells_l = mesh_l.topology.index_map(mesh_l.topology.dim).size_local 
    cells_l = np.arange(num_cells_l, dtype=np.int32)
    subdomains_l = dolfinx.mesh.meshtags(mesh_l, mesh_l.topology.dim, cells_l, sub_L)
    return subdomains_l, boundary_facets_left
    # Mapping the orinetation data from quad mesh to boundary. The alternative is to use local_frame_1D(mesh_l).
    # Either of both can be used in local_boun subroutine 
subdomains_l, boundary_facets_left=subdomains_boun(mesh_l,left,entity_mapl) # generating boundary submesh
subdomains_r, boundary_facets_right=subdomains_boun(mesh_r,right,entity_mapr)

# Generate ABD matrix (Plate model)
def ABD_mat(ii):
    """
    MSG based Kirchoff plate stiffness matrix of composite laminates
    
    Parameters:
        ii: Layup id

    Returns:
        ABD: [6,6] ! Plate Stiffness Matrix

    """  
    deg = 2
    cell = ufl.Cell("interval")
    elem= basix.ufl.element("Lagrange", "interval", 1, shape=(3, ))
    domain = ufl.Mesh(elem)

    # Nodes (1D SG)
    th,s=[0],0 # Reference-------- 0
    for k in thick[ii]:
        s=s+k           # Inward normal in orien provided by yaml file
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
    x,dx = SpatialCoordinate(dom),Measure('dx')(domain=dom, subdomain_data=subdomain, metadata={"quadrature_degree": 4})

    gamma_e=as_tensor([(1,0,0,x[0],0,0),       # Gamma_e matrix
                      (0,1,0,0,x[0],0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,0,0,0,0),
                      (0,0,1,0,0,x[0])])  

    nphases = len(cells)
    def gamma_h(v): # (Gamma_h * w)
        E1= as_vector([0,0,v[2].dx(0),(v[1].dx(0)),(v[0].dx(0)),0])
        return E1

    def R_sig(C,t): # Rotation matrix
        """
        Performs rotation from local material frame to global frame
        
        Parameters:
            C: [6,6] numpy array  ! Stiffness matrix
            t: constant           ! rotation angle
            
        Returns:
            C': Rotated Stiffness matrix
        """
        th=np.deg2rad(t)
        c,s,cs=np.cos(th),np.sin(th),np.cos(th)*np.sin(th)
        R_Sig= np.array([(c**2, s**2, 0,0,0,-2*cs),
                   (s**2, c**2, 0,0,0,2*cs),
                   (0,0,1,0,0,0),
                   (0,0,0,c,s,0),
                   (0,0,0,-s,c,0),
                   (cs,-cs,0,0,0,c**2-s**2)])
        return np.matmul(np.matmul(R_Sig,C),R_Sig.transpose())

    def Stiff_mat(i):     
        """
        Compute the [6,6] Stiffness matrix using material elastic constants
        
        Parameters:
            i: Material parameters id/ matid
            
        Returns:
            C: [6,6] Stiffness Matrix
        """
        E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[matid[ii][i]]
        S=np.zeros((6,6))
        S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
        S[0,1], S[0,2]= -v12/E1, -v13/E1
        S[1,0], S[1,2]= -v12/E1, -v23/E2
        S[2,0], S[2,1]= -v13/E1, -v23/E2
        S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12    
        C=np.linalg.inv(S)
        theta=angle[ii][i] # ii denotes the layup id
        C= as_tensor(R_sig(C,theta)) 
        return  C
        
    # Creating FE function Space
    V = functionspace(dom, basix.ufl.element("CG", "interval", deg, shape=(3, )))
    dv,v_ = TrialFunction(V), TestFunction(V)
    F2 = sum([dot(dot(Stiff_mat(i),gamma_h(dv)),gamma_h(v_))*dx(i) for i in range(nphases)]) # Weak form of energy(load vec)
    A=  petsc.assemble_matrix(form(F2))
    A.assemble()
    null = nullspace(V)
    A.setNullSpace(null)      # Set the nullspace
    ndofs = 3*V.dofmap.index_map.local_range[1] # total dofs
    # Initialization
    V0, Dhe, D_ee = np.zeros((ndofs,6)), np.zeros((ndofs,6)), np.zeros((6,6))

    # Assembly ! 6 Load Cases
    for p in range(6):
        # Right Hand Side vector in weak form (F2(v_))
        # < gamma_e[:,p].T Stiffness_Matrix gamma_h>
        F2= -sum([dot(dot(Stiff_mat(i),gamma_e[:,p]),gamma_h(v_))*dx(i) for i in range(nphases)]) 
        F = petsc.assemble_vector(form(F2))
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        null.remove(F)                        # Orthogonalize F to the null space of A^T
        Dhe[:,p]= F[:]                        # Dhe matrix formation
        V0[:,p]= ksp_solve(A,F,V)[0]          # Solved Fluctuating Functions
    D1=np.matmul(V0.T,-Dhe)  
    
    def Dee(i):  
        """
        Performs < gamma_e.T Stiffness_matrix gamma_e > and give simplified form
        
        Parameters:
            i: matid 
     
        Returns:
            Dee: [6,6] ufl tensor 
        """
        C=Stiff_mat(i)
        x0=x[0]
        return as_tensor([(C[0,0],C[0,1],C[0,5],x0*C[0,0],x0*C[0,1],x0*C[0,5]),
                      (C[1,0],C[1,1],C[1,5],x0*C[1,0],x0*C[1,1],x0*C[1,5]),
                      (C[5,0],C[5,1],C[5,5],x0*C[5,0],x0*C[5,1],x0*C[5,5]),
                      (x0*C[0,0],x0*C[0,1],x0*C[0,5],x0*x0*C[0,0],x0*x0*C[0,1],x0*x0*C[0,5]),
                      (x0*C[1,0],x0*C[1,1],x0*C[1,5],x0*x0*C[1,0],x0*x0*C[1,1],x0*x0*C[1,5]),
                     (x0*C[5,0],x0*C[5,1],x0*C[5,5],x0*x0*C[5,0],x0*x0*C[5,1],x0*x0*C[5,5])])
    # Scalar assembly
    for s in range(6):
        for k in range(6): 
            f=dolfinx.fem.form(sum([Dee(i)[s,k]*dx(i) for i in range(nphases)])) 
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)   # D_ee [6,6]  

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
    return w.vector[:],w

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
nphases=max(subdomains.values)+1
ABD_=[] 
for i in range(nphases):
    ABD_.append(ABD_mat(i))
print('Computed',nphases,'ABD matrix')

def ABD_matrix(i):
    return(as_tensor(ABD_[i]))
########################################## ABD matrix generation ends here ###########################
##
####################################Finite Element Implementation ##########################################
# Update: Removed local frame mesh from here as, for development of openSG, we go with yaml orientation
# Note grad(e2) will give derivatives of e2 in (x,yz,) global mesh frame. But, to obtain derivatives along local curvoilinear coordinates, 
# we use directional derivatives.
deg=2 # degree of Serendipidity finite elements
def local_frame(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]]) # tangential direction 
    t2 = as_vector([t[0, 1], t[1, 1], t[2, 1]]) 
    e3 = -cross(t1, t2) # outward  normal direction (default)  # default mean when e3 = cross(t1, t2)
    e3 /= sqrt(dot(e3, e3))
    e1=  t2/ sqrt(dot(t2, t2)) # 1- direction axis
    e2 = cross(e3, e1)
    e2 /= sqrt(dot(e2, e2)) # 2- direction  -circumferential (default not rh rule)
    return e1, e2, e3
    
def local_frame_1D(mesh): 
    t = Jacobian(mesh)
    t1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
    e2=  t1/ sqrt(dot(t1, t1))
    e1=  as_vector([1,0,0]) # Right Lay up
    e3= cross(e1,e2)
    e3=  e3/ sqrt(dot(e3, e3)) 
    return e1, e2, e3

def deri(e): # derivatives of local frame (Refer: Prof: Yu thesis)
    
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
    """
    Generate (gamma_h*w) vector (ufl)  
    
    Parameters:
        e: Local Orientation Frame in S functionspace (obtained from localboun)
        x: Spatial Coordinate
        w: Trial/Test Function   (i.e. dv/v_)
    Returns:
        gamma_h [6,1] (ufl tensor): Equivalent to gamma_h vector (ufl) showed in MSG-Shell formulations
    
    """
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
    """
    Generate (gamma_l*w') tensor (ufl)  where w' related with MSG Shell formulation
    showing derivative with same functionspace as dv/v_.
    
    Parameters:
        e: Local Orientation Frame in S functionspace (obtained from localboun)
        x: Spatial Coordinate
        w: Trial/Test Function   (i.e. dv/v_)
    Returns:
        gamma_l [6,1] (ufl tensor): Equivalent to gamma_l vector (ufl) showed in MSG-Shell formulations
    
    """
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
    w_d1= as_vector([local_grad(dd1,w[ii]) for ii in range(3)])
    w_d2= as_vector([local_grad(dd2,w[ii]) for ii in range(3)])

    L1,L2=x11*(ddot(d1,w)),x12*(ddot(d2,w))
    L3=ddot(x11*dd2+x12*dd1,w)
    L4=-2*x11*ddot(w_d1,d3)+ddot(k11*(y1*dd3-x11*dd1)-0.5*k12*(x12*dd1+x11*dd2),w)
    L5=-2*x12*ddot(w_d2,d3)+ddot(k22*(y1*dd3-x12*dd2)-0.5*k21*(x12*dd1+x11*dd2),w)
    L6=-2*ddot(x11*w_d2+x12*w_d1,d3)+ddot(k12*(y1*dd3-x12*dd2)+k21*(y1*dd3-x11*dd1)-0.5*(k11+k22)*(x12*dd1+x11*dd2),w)
    return  as_tensor([L1,L2,L3,L4,L5,L6])
    
# Gamma_e matrix   
def gamma_e(e,x):
    """
    Generate gamma_e tensor (ufl) whose columns are used as load cases.  
    
    Parameters:
        e: Local Orientation Frame in S functionspace (obtained from localboun)
        x: Spatial Coordinate
    Returns:
        gamma_e [6,4] (ufl tensor): Equivalent to gamma_e tensor (ufl) showed in MSG-Shell formulations
    
    """
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
# Update Removing gamma_d matrix to reduce computational load as it is not affecting significantly in final stiffness output

# Obtaining preprocessing mesh terms for boundary as well as main wb (quad mesh).
# not to confuse with l here, it applies for right boundary as well by giving appropriate arguments

#************Update generating tangent facet vector (nh) manually************************
#************Used existing code which needs simplification***************************
#************For now, output of nh is required in der_constraint(dv,v_,mesh) code************

def tangential_projection(u: ufl.Coefficient, n: ufl.FacetNormal) -> ufl.Coefficient:
    """
    See for instance:
    https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u

def facet_vector_approximation(V: dfx.fem.FunctionSpace,
                               mt: dfx.mesh.MeshTags | None = None,
                               mt_id: int | None = None,
                               tangent: bool = False,
                               interior: bool = False,
                               jit_options: dict | None = None,
                               form_compiler_options: dict | None = None) -> dfx.fem.Function:
 
    jit_options = jit_options if jit_options is not None else {}
    form_compiler_options = form_compiler_options if form_compiler_options is not None else {}

    comm  = V.mesh.comm # MPI Communicator
    n     = ufl.FacetNormal(V.mesh) # UFL representation of mesh facet normal
    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V) # Trial and test functions

    # Create interior facet integral measure
    dS = ufl.dS(domain=V.mesh) if mt is None else ufl.dS(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)
    
    # If tangent==True, the right-hand side of the problem should be a tangential projection of the facet normal vector.


    c = dfx.fem.Constant(V.mesh, (1.0, 1.0, 1.0)) # Vector to tangentially project the facet normal vectors on

    a = (ufl.inner(u('+'), v('+')) + ufl.inner(u('-'), v('-'))) * dS
    L = ufl.inner(tangential_projection(c, n('+')), v('+')) * dS \
        + ufl.inner(tangential_projection(c, n('-')), v('-')) * dS
    # If tangent==false the right-hand side is simply the facet normal vector.
    a = (ufl.inner(u('+'), v('+')) + ufl.inner(u('-'), v('-'))) * dS
    L = (ufl.inner(n('+'), v('+')) + ufl.inner(n('-'), v('-'))) * dS

    # Find all boundary dofs, which are the dofs where we want to solve for the facet vector approximation.
    # Start by assembling test functions integrated over the boundary integral measure.
    ones = dfx.fem.Constant(V.mesh, dfx.default_scalar_type((1,) * V.mesh.geometry.dim)) # A vector of ones

    local_val = dfx.fem.form((ufl.dot(ones, v('+')) + ufl.dot(ones, v('-')))*dS)
    local_vec = dfx.fem.assemble_vector(local_val)

    # For the dofs that do not lie on the boundary of the mesh the assembled vector has value zero.
    # Extract these dofs and use them to deactivate the corresponding block in the linear system we will solve.
    bdry_dofs_zero_val  = np.flatnonzero(np.isclose(local_vec.array, 0))
    deac_blocks = np.unique(bdry_dofs_zero_val // V.dofmap.bs).astype(np.int32)

    # Create sparsity pattern by manipulating the blocks to be deactivated and set
    # a zero Dirichlet boundary condition for these dofs.
    bilinear_form = dfx.fem.form(a, jit_options=jit_options,
                                 form_compiler_options=form_compiler_options)
    pattern = dfx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()
    u_0 = dfx.fem.Function(V)
    u_0.vector.set(0)
    bc_deac = dfx.fem.dirichletbc(u_0, deac_blocks)

    # Create the matrix
    A = dfx.cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dfx.cpp.fem.pack_coefficients(bilinear_form._cpp_object)
    form_consts = dfx.cpp.fem.pack_constants(bilinear_form._cpp_object)
    dfx.fem.petsc.assemble_matrix(A, bilinear_form, constants=form_consts, coeffs=form_coeffs, bcs=[bc_deac])

    # Insert the diagonal with the deactivated blocks.
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dfx.cpp.fem.petsc.insert_diagonal(A=A, V=bilinear_form.function_spaces[0], bcs=[bc_deac._cpp_object], diagonal=1.0)
    A.assemble()

    # Assemble the linear form and the right-hand side vector.
    linear_form = dfx.fem.form(L, jit_options=jit_options,
                               form_compiler_options=form_compiler_options)
    b = dfx.fem.petsc.assemble_vector(linear_form)


    # Apply lifting to the right-hand side vector and set boundary conditions.
    dfx.fem.petsc.apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dfx.fem.petsc.set_bc(b, [bc_deac])

    # Setup a linear solver using the Conjugate Gradient method.
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)

    # Solve the linear system and perform ghost update.
    nh    = dfx.fem.Function(V)     # Function for the facet vector approximation
    solver.solve(b, nh.vector)
    nh.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Normalize the vectors to get the unit facet normal/tangent vector.
    nh_norm = ufl.sqrt(ufl.inner(nh, nh)) # Norm of facet vector
    cond_norm = ufl.conditional(ufl.gt(nh_norm, 1e-10), nh_norm, 1.0) # Avoid division by zero
    nh_norm_vec = ufl.as_vector((nh[0]/cond_norm, nh[1]/cond_norm, nh[2]/cond_norm))

    nh_normalized = dfx.fem.Expression(nh_norm_vec, V.element.interpolation_points())
    
    n_out = dfx.fem.Function(V)
    n_out.interpolate(nh_normalized)

    return n_out

# Case flags
all_facets   = False # Approximate all facet vectors if True, subset if False
tangent_flag = True # Approximate normals if False, approximate tangents if True
interior     = True  # Set to True if the facets are internal (e.g. an interface between two domains)
                     # Set to False if the facets are on the mesh boundary

dim = 3 # Spatial dimension of mesh
  
DEFAULT = 2
SUBSET  = 3

# Mark the interior facets lying on an interface inside the square/cube.
# NOTE: this is a pretty ad-hoc way of marking an interface.
def locator(x):
        """ Marker function that returns True if the x-coordinate is between xmin and xmax. """
        return np.logical_and(x[0] > x_min, x[0] > x_min)

# Create necessary topological entities of the mesh
mesh.topology.create_entities(mesh.topology.dim-1)
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

# Create an array facet_marker which contains the facet marker value for all facets in the mesh.
facet_dim = mesh.topology.dim-1 # Topological dimension of facets
num_facets   = mesh.topology.index_map(facet_dim).size_local + mesh.topology.index_map(facet_dim).num_ghosts # Total number of facets in mesh
facet_marker = np.full(num_facets, DEFAULT, dtype=np.int32) # Default facet marker value is 2

subset=np.linspace(1,num_facets,num_facets,dtype=int)-1
n1=np.setdiff1d(subset, entity_mapl)
subset_facets=np.setdiff1d(n1, entity_mapr)
#subset_facets = dolfinx.mesh.locate_entities(mesh, facet_dim, locator) # Get subset of facets to be marked
facet_marker[subset_facets] = SUBSET # Fill facet marker array with the value SUBSET
facet_tags = dolfinx.mesh.meshtags(mesh, facet_dim, np.arange(num_facets, dtype=np.int32), facet_marker) # Create facet meshtags
ft_id = SUBSET # Set the meshtags id for which we will approximate the facet vectors
# Create a DG1 space for the facet vectors to be approximated.
DG1   = basix.ufl.element(family="Lagrange", cell=mesh.basix_cell(), degree=1, discontinuous=True, shape=(mesh.geometry.dim,))
space = dolfinx.fem.functionspace(mesh=mesh, element=DG1)
# Compute the facet vector approximation (Tangent) 
nh = facet_vector_approximation(V=space, mt=facet_tags, mt_id=ft_id, interior=interior, tangent=tangent_flag) 

#**************Facet tangent vector generation ends******************
#*************We can look forward to optimize once code is finalized***************

def local_boun(mesh_l,frame,subdomains_l):
    """
    Interpolate the local orientation frame to function space .i.e. from DG space to S space
    
    Parameters:
        mesh_l: Left Boundary mesh
        frame_l: Left Boundary local orientation frame
        subdomains_l: layup distribution subdomain of left boundary mesh (obtained after mapping)
    Returns:
        [e1l,e2l,e3l]: S functionspace interpolated frame
        V_l: Left Boundary functionspace (interval mesh)
        TrialFunction(V_l), TestFunction(V_l)
        x,dx: Spatial Coordinate and integration measure 
    """
    V_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
        "S", mesh_l.topology.cell_name(), 2, shape=(3, )))
    le1, le2, le3 = frame
    e1l,e2l,e3l=Function(V_l), Function(V_l), Function(V_l)
    
    fexpr1=dolfinx.fem.Expression(le1,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
    e1l.interpolate(fexpr1) 
    
    fexpr2=dolfinx.fem.Expression(le2,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
    e2l.interpolate(fexpr2) 
    
    fexpr3=dolfinx.fem.Expression(le3,V_l.element.interpolation_points(), comm=MPI.COMM_WORLD)
    e3l.interpolate(fexpr3) 
        
    return [e1l,e2l,e3l], V_l, TrialFunction(V_l), TestFunction(V_l),SpatialCoordinate(mesh_l), \
          Measure('dx')(domain=mesh_l, subdomain_data=subdomains_l)
    


# In[ ]:





# In[10]:


def deri_constraint(dvl,v_l,mesh):
    """
    Generate C0-Interior Penalty term for bilinear weak form to constrain normal and tangent derivative along the interface
    Boundary mesh only requires normal derivative constraint, and WB mesh requires both tangent and normal derivative constraint
    Parameters:
        dv,v_: Trail and test function (either WB segment/boundary)
        mesh: Input mesh (either WB segment/boundary)
        
    Returns:
        nn: bilinear normal derivative constraint terms (boundary mesh)
        nn+tt: bilinear constraint terms for both normal and tangent boundary (WB surface mesh).
    """
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    h_avg = (h('+') + h('-')) / 2.0
    dS=Measure('dS')(domain=mesh)
    if mesh.topology.dim==1:
        alpha=1e10
        nn= - inner(avg(div(grad(dvl))), jump(grad(v_l), n))*dS \
            - inner(jump(grad(dvl), n), avg(div(grad(v_l))))*dS \
            + (alpha/h_avg**2)*inner(jump(grad(dvl),n), jump(grad(v_l),n))*dS
        return nn
    elif mesh.topology.dim==2: 
        alpha=1e10
        beta=4e5
        nn= - inner(avg(div(grad(dvl))), jump(grad(v_l), n))*dS \
            - inner(jump(grad(dvl), n), avg(div(grad(v_l))))*dS \
            + (alpha/h_avg**2)*inner(jump(grad(dvl),n), jump(grad(v_l),n))*dS
        tt=- inner(avg(div(grad(dvl))), jump(grad(v_l), nh))*dS \
           - inner(jump(grad(dvl), nh), avg(div(grad(v_l))))*dS \
           + (beta/h_avg**2)*inner(jump(grad(dvl),nh), jump(grad(v_l),nh))*dS
        return nn+tt  
        
# Removed A_mat
# Removing the solve boundary function as same output is now extracted from timo_boun function

def initialize_array(V):
    """
    Output initialized numpy arrays
    
    Parameters:
        V: functionspace
        
    Returns:
        V0,Dle,Dhe,D_ee,V1s: numpy arrays
    """
    ndofs=3*len(np.arange(*V.dofmap.index_map.local_range))  # total dofs of mesh
    V0 = np.zeros((ndofs,4))
    Dle=np.zeros((ndofs,4))
    Dhe=np.zeros((ndofs,4))
    D_ee=np.zeros((4,4))  
    V1s=np.zeros((ndofs,4)) 
    return V0,Dle,Dhe,D_ee,V1s
    
def timo_boun(mesh_l,subdomains_l,frame_l):
    """
    Solve EB and Timo model for boundary mesh. The output flcutuating functions V0 and V1s are used dirichilet boundary constraints
    
    Parameters:
        mesh_l: Left Boundary mesh
        subdomains_l:  Left Boundary subdomains (layup information)
        frame_l: local orientation frame (DG space)
    Returns:
        D_eff (np array): [4,4] !Boundary EB Stiffness matrix
        Deff_srt (np array):[6,6] !Boundary Timoshenko Stiffness matrix 
        V0 (np array): [ndofs_leftmesh,4] !Boundary fluctuating function solutions after solving load cases (useful in WB segment EB)
        V1s (np array):[ndofs_leftmesh,4] !Boundary fluctuating function solutions after solving load cases (useful in WB segment Timoshenko Stiffness)
    """
    
    e, V_l, dv, v_, x, dx=local_boun(mesh_l,frame_l,subdomains_l)          
    mesh_l.topology.create_connectivity(1, 1)
    V0,Dle,Dhe,D_ee,V1s=initialize_array(V_l)
    nullspace_l=nullspace(V_l)
 #   A_l=A_mat(e,x,dx,nullspace(V_l),v_,dv,mesh_l)
    # Compute A_mat
    F2 = sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
    ff=  deri_constraint(dv,v_,mesh_l) 
    A_l=assemble_matrix(form(F2+ff))
    A_l.assemble()
    A_l.setNullSpace(nullspace_l) 
    
    for p in range(4):
        Eps=gamma_e(e,x)[:,p]
        F2 = sum([dot(dot(ABD_matrix(i),Eps), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])   
        r_he=form(rhs(F2))
        F_l = petsc.assemble_vector(r_he)
        F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
        nullspace_l.remove(F_l)
        Dhe[:,p]=  petsc.assemble_vector(r_he)[:]
        V0[:,p]= ksp_solve(A_l,F_l,V_l)[0]  

    D1=np.matmul(V0.T,-Dhe)   
    for s in range(4):
        for k in range(4): 
            f=dolfinx.fem.form(sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)]))
            D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
        
    D_eff= D_ee + D1 # Effective Stiffness Matrix (EB)
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
        
    F_dhl=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
    ff=deri_constraint(dv,v_,mesh_l)
    a3=form(F_dhl+ff)
    Dhl=assemble_matrix(a3)
    Dhl.assemble()
    ai, aj, av=Dhl.getValuesCSR()
    Dhl=csr_matrix((av, aj, ai)).toarray()
        
    #DhlTV0
    DhlV0=np.matmul(Dhl.T,V0)
    
    #DhlTV0Dle
    DhlTV0Dle=np.matmul(Dhl,V0)+Dle
    
    #V0DllV0
    V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)
    
    # V1s  ****Updated from previous version as for solving boundary V1s, we can directly use (A_l V1s=b),  and solve for V1s
    b=DhlTV0Dle-DhlV0
    ai, aj, av=A_l.getValuesCSR()  
    A_l=csr_matrix((av, aj, ai))
    V1s=scipy.sparse.linalg.spsolve(A_l, b, permc_spec=None, use_umfpack=True)    
    
    # Ainv
    Ainv=np.linalg.inv(D_eff)
    
    # B_tim
    B_tim= np.matmul(DhlTV0Dle.T,V0)
    
    # C_tim
    C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)
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
    
    return np.around(D_eff),np.around(Deff_srt),V0, V1s

# Initialize terms
e_l, V_l, dvl, v_l, x_l, dx_l=local_boun(mesh_l,local_frame_1D(mesh_l),subdomains_l)
e_r, V_r, dvr, v_r, x_r, dx_r=local_boun(mesh_r,local_frame_1D(mesh_r) ,subdomains_r)

# dof mapping makes solved unknown value w_l(Function(V_l)) assigned to v2a (Function(V)). 
# The boundary of wind blade mesh is a 1D curve. The facet/edge number is obtained from cell to edge connectivity (conn3) showed in subdomain subroutine.
# The same facet/edge number of extracted mesh_l (submesh) is obtaine din entity_mapl (gloabl mesh number). refer how submesh was generated.
#Therefore, once identifying the edge number being same for global(mesh)&boundary mesh(mesh_l), we equate the dofs and store w_l to v2a.
# The dofs can be verified by comparing the coordinates of local and global dofs if required. 
def dof_mapping_quad(v2a,V_l,w_ll,boundary_facets_left,entity_mapl):
    """
    Map boundary solutions (V0_l and V1_l) to WB segment mesh functions to use a dirichilet boundary constraint
    
    Parameters:
        v2a : ufl function to store boundary solutions
        V_l:  Left boundary functionspace
        w_ll[ndofs_leftmesh,1]: numpy column vector containing columns of V0_l (i.e. V0_l[:,p]) 
        boundary_facets_left[num_facets_left,1]: Left Boundary facets id (numbering in left boundary mesh)
        entity_mapl[num_facets_left,1]:  Left Boundary facets id (numbering in Wb Segment mesh)
    Returns:
        v2a: Modified WB mesh function containg V0_l data at boundary dofs 
    """
    dof_S2L=[]
    for i,xx in enumerate(entity_mapl):
        # For a unique facet obtain dofs 
        dofs = locate_dofs_topological(V,1, np.array([xx]))  # WB segment mesh facets dofs
        dofs_left= locate_dofs_topological(V_l,1, np.array([boundary_facets_left[i]])) # Boundary mesh facets dofs
        
        for k in range(deg+1):
            if dofs[k] not in dof_S2L:
                dof_S2L.append(dofs[k])
                for j in range(3):
                    v2a.vector[3*dofs[k]+j]=w_ll[3*dofs_left[k]+j] # store boundary solution of fluctuating functions
    return v2a
#Removed solve boun
#V0_l,V0_r=solve_boun(mesh_l,local_frame_1D(mesh_l),subdomains_l),solve_boun(mesh_r,local_frame_1D(mesh_l),subdomains_r)
D_effEB_l,Deff_l,V0_l,V1_l=timo_boun(mesh_l,subdomains_l,local_frame_1D(mesh_l))
D_effEB_r,Deff_r,V0_r,V1_r=timo_boun(mesh_r,subdomains_r,local_frame_1D(mesh_r))

#***************** Wb Segment (surface mesh) computation begins************************
e,  V, dv,  v_,  x,    dx=local_boun(mesh,local_frame(mesh),subdomains)
V0,Dle,Dhe,D_ee,V1s=initialize_array(V)
mesh.topology.create_connectivity(1, 2)
mesh_l.topology.create_connectivity(1, 1)
mesh_r.topology.create_connectivity(1, 1)

# Obtaining coefficient matrix AA and BB with and without bc applied.
# Note: bc is applied at boundary dofs. We define v2a containing all dofs of entire wind blade.

F2=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])
ff=deri_constraint(dv,v_,mesh)
a= form(F2+ff)
B=assemble_matrix(form(F2))   # Obtain coefficient matrix without BC applied: BB
B.assemble()
ai, aj, av=B.getValuesCSR()
BB=csr_matrix((av, aj, ai))
# bc applied
boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((entity_mapr,entity_mapl), axis=0))
v2a=Function(V)
bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
A=assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
A.assemble() 

# Assembly
# Running for 4 different F vector. However, F has bc applied to it where, stored known values of v2a is provided for each loop (from boun solve).
# Assembly  
for p in range(4): # 4 load cases meaning 
    # Boundary 
    v2a=Function(V) 
    v2a=dof_mapping_quad(v2a,V_l,V0_l[:,p],boundary_facets_left,entity_mapl)
    v2a=dof_mapping_quad(v2a,V_r,V0_r[:,p],boundary_facets_right,entity_mapr)  
    
    F2=-sum([dot(dot(ABD_matrix(i),gamma_e(e,x)[:,p]), gamma_h(e,x,v_))*dx(i) for i in range(nphases)])  
    bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
    F = petsc.assemble_vector(form(F2))
    apply_lifting(F, [a], [bc]) # apply bc to rhs vector (Dhe) based on known fluc solutions
    set_bc(F, bc)         
    V0[:,p],v=ksp_solve(A,F,V)  
  #  Dhe[:,p]= petsc.assemble_vector(form(F2))
    Dhe[:,p]=scipy.sparse.csr_array(BB).dot(V0[:,p])
    
D1=np.matmul(V0.T,-Dhe) 
for s in range(4):
    for k in range(4): 
        f=dolfinx.fem.form(sum([dot(dot(gamma_e(e,x).T,ABD_matrix(i)),gamma_e(e,x))[s,k]*dx(i) for i in range(nphases)]))
        D_ee[s,k]=dolfinx.fem.assemble_scalar(f)
L=(x_max-x_min)
D_eff= (D_ee + D1)/L

print('\n EB Tapered Stiffness \n')
np.set_printoptions(precision=4)
print(np.around(D_eff)) 


# In[11]:


##################Timoshenko Stiffness Matrix for WB segment begins###################################
# Process is similar to Timoshenko boundary implemented over WB segment mesh
F1=sum([dot(dot(ABD_matrix(i),gamma_l(e,x,v_)),gamma_l(e,x,dv))*dx(i) for i in range(nphases)])
a1=form(F1)
Dll=assemble_matrix(a1)
Dll.assemble()
ai, aj, av=Dll.getValuesCSR()  
Dll=csr_matrix((av, aj, ai)).toarray()

#Dhl
F_dhl=sum([dot(dot(ABD_matrix(i),gamma_h(e,x,dv)),gamma_l(e,x,v_))*dx(i) for i in range(nphases)]) 
a3=form(F_dhl)
Dhl=assemble_matrix(a3)
Dhl.assemble()
ai, aj, av=Dhl.getValuesCSR()
Dhl=csr_matrix((av, aj, ai)).toarray() 

for p in range(4):
        F1=sum([dot(dot(ABD_matrix(i),gamma_e(e,x)[:,p]),gamma_l(e,x,v_))*dx(i) for i in range(nphases)])  
        Dle[:,p]= petsc.assemble_vector(form(F1))[:]

#DhlTV0
DhlV0=np.matmul(Dhl.T,V0)

#DhlTV0Dle
DhlTV0Dle=np.matmul(Dhl,V0)+Dle

#V0DllV0
V0DllV0=np.matmul(np.matmul(V0.T,Dll),V0)

# V1s
b=DhlTV0Dle-DhlV0

# B_tim
B_tim= np.matmul(DhlTV0Dle.T,V0)
B_tim=B_tim/L

# Assembly
# For Wb mesh (surface elements, for solving (A V1s = b), directly cannot be computed as we require dirichilet bc)
for p in range(4): # 4 load cases meaning 
    # Boundary 
    v2a=Function(V) 
    v2a=dof_mapping_quad(v2a,V_l,V1_l[:,p],boundary_facets_left,entity_mapl)
    v2a=dof_mapping_quad(v2a,V_r,V1_r[:,p],boundary_facets_right,entity_mapr)  
    bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
    
    # quad mesh
    F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD) # Converting b[:,p] numpy vector to petsc array
    F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    apply_lifting(F, [a], [bc]) 
    set_bc(F, bc)          
    V1s[:,p]=ksp_solve(A,F,V)[0]

# C_tim
C_tim= V0DllV0 + np.matmul(V1s.T,DhlV0 + DhlTV0Dle)
C_tim=0.5*(C_tim+C_tim.T)
C_tim=C_tim/L
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
Deff_srt[3:6,0]=A_tim[1:4,0]

Deff_srt[1:3,1:3]=G_tim
Deff_srt[1:3,3:6]=Y_tim.T[:,1:4]
Deff_srt[1:3,0]=Y_tim.T[:,0]

print('\n Timo Stiffness Matrix for WB Segment \n')
np.set_printoptions(precision=4)
print(np.around(Deff_srt)) 


print('\n Timoshenko Stiffness (Left Boundary) \n')
np.set_printoptions(precision=4)
print(np.around(Deff_l)) 


# Compare Unique ABD matrices
for ii,AB in enumerate(ABD_):
    np.set_printoptions(precision=4)
    print('\n',ii,'\n')
    print(np.around(AB))
