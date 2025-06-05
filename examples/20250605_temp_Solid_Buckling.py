
###############     OpenSG Patch         ############################
########### Tapered Timo Buckling Model (Solid Elements) ###################

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix 
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
import meshio
import dolfinx
from dolfinx.fem import dirichletbc,form, petsc, Function, functionspace, locate_dofs_topological, apply_lifting, set_bc, assemble_scalar
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
from petsc4py import PETSc
from slepc4py import SLEPc
from eigenvalue_solver import solve_GEP_shiftinvert, EPS_get_spectrum
import pyvista 

def write_beamdyn_files(beam_stiff, beam_inertia, radial_stations,file_name_prepend):
    # extension='K'
    # if station_list is None or len(station_list) == 0:
    #     station_list = list(range(len(geometry.ispan)))

    # radial_stations=geometry.ispan/geometry.ispan[-1]
    # radial_stations=radial_stations[station_list]

    if round(radial_stations[-1],2) ==1.0:
        radial_stations[-1]=1.0
    else:
        raise ValueError('The last radial station should be 1.0')
    if round(radial_stations[0],2) ==0.0:
        radial_stations[0]=0.0
    else:
        raise ValueError('The first radial station should be 0.0')
    
    if len(beam_stiff) != len(beam_inertia) and len(beam_stiff) != len(radial_stations):
        raise ValueError(f'\nThere are {len(beam_stiff)} stiffnesses\nThere are {len(beam_inertia)} inertias\nThere are {len(radial_stations)} radial stations \nThese need to be equal.')

    mu=[0.00257593, 0.0017469,  0.0017469,  0.0017469,  0.00257593, 0.0017469]

    beam_stiff,beam_inertia=transformMatrixToBeamDyn(beam_stiff,beam_inertia)
    _=write_beamdyn_prop('.', file_name_prepend, radial_stations, beam_stiff, beam_inertia, mu)
    
    return 
def write_beamdyn_prop(folder, wt_name, radial_stations, beam_stiff, beam_inertia, mu):
    n_pts = len(radial_stations)

        
    propFileName= 'bd_props_'+wt_name + '.inp'
    
    
    file = open(folder +'/'+propFileName, 'w')
    file.write(' ------- BEAMDYN V1.00.* INDIVIDUAL BLADE INPUT FILE --------------------------\n')
    file.write(' Test Format 1\n')
    file.write(' ---------------------- BLADE PARAMETERS --------------------------------------\n')
    file.write('%u   station_total    - Number of blade input stations (-)\n' % (n_pts))
    file.write(' 1   damp_type        - Damping type: 0: no damping; 1: damped\n')
    file.write('  ---------------------- DAMPING COEFFICIENT------------------------------------\n')
    file.write('   mu1        mu2        mu3        mu4        mu5        mu6\n')
    file.write('   (-)        (-)        (-)        (-)        (-)        (-)\n')
    file.write('\t %.5e \t %.5e \t %.5e \t %.5e \t %.5e \t %.5e\n' % (mu[0], mu[1], mu[2], mu[3], mu[4], mu[5])) 
    file.write(' ---------------------- DISTRIBUTED PROPERTIES---------------------------------\n')
    
    for i in range(n_pts):
        file.write('\t %.6f \n' % (radial_stations[i]))
        # write stiffness matrices
        for j in range(6):
            file.write('\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n' % (
            beam_stiff[i, j, 0], beam_stiff[i, j, 1], beam_stiff[i, j, 2], beam_stiff[i, j, 3], beam_stiff[i, j, 4],
            beam_stiff[i, j, 5]))
        file.write('\n')

        # write inertia properties
        for j in range(6):
            file.write('\t %.16e \t %.16e \t %.16e \t %.16e \t %.16e \t %.16e\n' % (
            beam_inertia[i, j, 0], beam_inertia[i, j, 1], beam_inertia[i, j, 2], beam_inertia[i, j, 3],
            beam_inertia[i, j, 4], beam_inertia[i, j, 5]))
        file.write('\n')
        # ToDO: check correct translation of stiffness and mass matrices from VABS and anbax !!!
    file.close()

    print('Finished writing BeamDyn_Blade File')

    return propFileName
def transformMatrixToBeamDyn(beam_stiff,beam_inertia):
    beamDynData={}

    B = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])  # NEW transformation matrix
    T = np.dot(np.identity(3), np.linalg.inv(B))
    
    nStations, _,_=np.shape(beam_stiff)

    for i_station in range(nStations):
        beam_stiff[i_station,:,:]=trsf_sixbysix(beam_stiff[i_station,:,:], T)
     #   beam_inertia[i_station,:,:]=trsf_sixbysix(beam_inertia[i_station,:,:], T)
   
    return(beam_stiff,beam_inertia)
def trsf_sixbysix(M, T):
    """
    Transform six-by-six compliance/stiffness matrix. 
    change of reference frame in engineering (or Voigt) notation.
    
    Parameters
    ----------
    M : np.ndarray
        6x6 Siffness or Mass Matrix
    T : np.ndarray
        Transformation Matrix
        
    Returns
    ----------
    res : np.ndarray
        Transformed 6x6 matrix
    """

    TS_1 = np.dot(np.dot(T.T, M[0:3, 0:3]), T)
    TS_2 = np.dot(np.dot(T.T, M[3:6, 0:3]), T)
    TS_3 = np.dot(np.dot(T.T, M[0:3, 3:6]), T)
    TS_4 = np.dot(np.dot(T.T, M[3:6, 3:6]), T)

    tmp_1 = np.vstack((TS_1, TS_2))
    tmp_2 = np.vstack((TS_3, TS_4))
    res = np.hstack((tmp_1, tmp_2))
    return res


left_timo,right_timo, taper_timo=[],[],[]
left_mass,right_mass, taper_mass=[],[],[]
blade_length = 100
left_origin,right_origin, taper_origin=[],[],[]  

for segment in np.linspace(0,5,6):
    # ****************OBTAIN MESH DATA FROM YAML***********************
    
    meshYaml = 'bar_urc_npl_1_ar_5-segment_'+ str(int(segment)) +'.yaml'  ## the name of the yaml file containing the whole blade mesh
    #   meshYaml = 'new_tapered_solid_dcm.yaml'
    mshFile = 'SG_solid.msh'
    oriFile = 'solid.orientation'
    ## Read the mesh yaml file
    
    inFile = open(meshYaml,'r')
    meshData = yaml.load(inFile,Loader=cLd)
    inFile.close()
    ## Extract the mesh for the section
    nodes = meshData['nodes']
    elements = meshData['elements']
    elem=[]

    ## Write .msh file
    outFile = open(mshFile,'w')
    
    outFile.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
    outFile.write(str(len(nodes)) + '\n')
    points=[]
    for i, nd in enumerate(nodes):
            ndd=nd[0].split()
            ln = [str(i+1),str(ndd[2]),str(ndd[0]),str(ndd[1])] # Making x-axis as beam axis
            outFile.write(' '.join(ln) + '\n')
            points.append(ln)
    outFile.write('$EndNodes\n$Elements\n')
    
    newNumEls = len(elements)
    outFile.write(str(newNumEls) + '\n')
                
    for j,eli in enumerate(elements):
                
                ln = [str(j+1)]
                ln.append('5')
                ln.append('2')
                ln.append(str(3))   
                ln.append(str(3))    
                ell=eli[0].split()
                for n in ell:
                    ln.append(n)
                outFile.write(' '.join(ln) + '\n')
    outFile.write('$EndElements\n')
       
    outFile.close()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)    # mesh read output will not be printed
    mesh, subdomains, boundaries = gmshio.read_from_msh("SG_solid.msh", MPI.COMM_SELF,rank=0, gdim=3) 
    o_cell_idx=mesh.topology.original_cell_index  # Original cell Index from mesh file

    layCt=-1
    mat_name=[]
    elLayID=np.zeros((newNumEls))
    for es in meshData['sets']['element']:
        if es['labels'] is not None:
            mat_name.append(es['name'])
            layCt += 1
            for eli in es['labels']:
                elLayID[eli-1]=layCt
    lnn=[]
    for k in o_cell_idx:
         lnn.append(elLayID[k])
    lnn=np.array(lnn,dtype=np.int32)    
    
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, lnn)
    #########################################################################################
    material_parameters, density=[], []
    
    mat_names=[material['name'] for material in meshData['materials']]

    for mat in mat_name:
        es=meshData['materials'][mat_names.index(mat)]
        material_parameters.append(np.array((np.array(es['E']),np.array(es['G']),es['nu'])).flatten())
        density.append(es['rho'])  
    
    # ************Shifting origin to mid-length along axis***************************
    # ************Origin on beam axis*************************************
    
    # **********Store orientation for each element********************
    VV = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "DG", mesh.topology.cell_name(), 0, shape=(3, )))
    EE1,EE2,N=Function(VV),Function(VV),Function(VV) 
    
    orien=[]
    for i, eo in enumerate(meshData['elementOrientations']):
            o=[]
            for k in range(9):
                o.append(eo[k])
            orien.append(o)
    
    for k,ii in enumerate(o_cell_idx):
        a2= np.array([(orien[ii][2],orien[ii][0],orien[ii][1]),
                      (orien[ii][5],orien[ii][3],orien[ii][4]),
                      (orien[ii][8],orien[ii][6],orien[ii][7])])
        
    # Storing data to DG0 functions 
 
        EE2.vector[3*k],EE2.vector[3*k+1],EE2.vector[3*k+2]=a2[1,0],a2[1,1],a2[1,2]   # e2
        N.vector[3*k],N.vector[3*k+1],N.vector[3*k+2]=a2[2,0],a2[2,1],a2[2,2]   #  e3 
        EE1.vector[3*k], EE1.vector[3*k+1],EE1.vector[3*k+2]=a2[0,0],a2[0,1],a2[0,2]   # e1    
        
    frame=[EE1,EE2,N]
    
    pp=mesh.geometry.x
    
    # ************Shifting origin to mid-length along axis***************************
    # ************Origin on beam axis*************************************
    
    x_min,x_max=min(pp[:,0]), max(pp[:,0])
    L=x_max-x_min
    mean=0.5*(x_min+x_max)  # Left origin for taper segments

    left_origin.append(float(x_min)/blade_length),right_origin.append(float(x_max)/blade_length),taper_origin.append(float(mean)/blade_length)
    pp[:,0]=pp[:,0]-mean
    x_min,x_max=min(pp[:,0]), max(pp[:,0])
    
    # ***********GENERATE BOUNDARY MESH************************
    print('Segment',str(int(segment)),'[Mid-Origin: ',str(mean),'] [',str(num_cells),']'+ '\n')
    
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

    # **********Store orientation for each element********************
    VV = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "DG", mesh.topology.cell_name(), 0, shape=(3, )))
    EE1,EE2,N=Function(VV),Function(VV),Function(VV) 
    
    o_cell_idx=mesh.topology.original_cell_index  # Original cell Index from mesh file
    orien=[]
    for i, eo in enumerate(meshData['elementOrientations']):
            o=[]
            for k in range(9):
                o.append(eo[k])
            orien.append(o)
    
    for k,ii in enumerate(o_cell_idx):
        a2= np.array([(orien[ii][2],orien[ii][0],orien[ii][1]),
                      (orien[ii][5],orien[ii][3],orien[ii][4]),
                      (orien[ii][8],orien[ii][6],orien[ii][7])])
        
    # Storing data to DG0 functions 
 
        EE2.vector[3*k],EE2.vector[3*k+1],EE2.vector[3*k+2]=orien[ii][5],orien[ii][3],orien[ii][4]   # e2
        N.vector[3*k],N.vector[3*k+1],N.vector[3*k+2]=orien[ii][8],orien[ii][6],orien[ii][7]   #  e3 
        EE1.vector[3*k], EE1.vector[3*k+1],EE1.vector[3*k+2]=orien[ii][2],orien[ii][0],orien[ii][1]   # e1    
        
    frame=[EE1,EE2,N]
    
    # Boundary
    # **********************MAP layup data to boundary*************************
    import math
    V= dolfinx.fem.functionspace(mesh, basix.ufl.element(
    "DG", mesh.topology.cell_name(), 0, shape=(3, )))
    E1,E2,E3=Function(V),Function(V),Function(V)
        
    for i,ii in enumerate(o_cell_idx):
        ang=math.atan2(-orien[ii][4], orien[ii][3])
        c,s=cos(ang),sin(ang)
        E1.vector[3*i:3*i+3]=1,0,0
        E2.vector[3*i:3*i+3]=0,c,s
        E3.vector[3*i:3*i+3]=0,-s,c


    nphases = max(subdomains.values)+1
    mesh.topology.create_connectivity(3,2)
    cell_of_face_mesh=mesh.topology.connectivity(3,2)
    # Cell to Face connectivity
    conn3=[]
    for i in range(num_cells):
        c=[]
        for k in range(6):
            c.append((cell_of_face_mesh.array[6*i+k]))
        conn3.append(c) 
    conn=np.ndarray.flatten(np.array(conn3))
    
    def subdomains_boun(mesh_l,left,entity_mapl):
        VV_l = dolfinx.fem.functionspace(mesh_l, basix.ufl.element(
            "DG", mesh_l.topology.cell_name(), 0, shape=(3, )))
        th,El1,El2,Nl=Function(VV_l),Function(VV_l),Function(VV_l),Function(VV_l)
        sub_L=[]
        boundary_facets_left= dolfinx.mesh.locate_entities(mesh_l, fdim, left)
        for i,xx in enumerate(entity_mapl):
                 idx=int(np.where(conn==xx)[0]/6)   
                 sub_L.append(subdomains.values[idx])
             #    th.vector[3*i],th.vector[3*i+1],th.vector[3*i+2]=orien[idx][2],orien[idx][0],orien[idx][1]
                 for j in range(3):
                            El1.vector[3*i+j]=frame[0].vector[3*idx+j]
                            El2.vector[3*i+j]=frame[1].vector[3*idx+j]
                            Nl.vector[3*i+j]=frame[2].vector[3*idx+j]
    
        frame_l=El1,El2,Nl                  
        sub_L= np.array(sub_L,dtype=np.int32)
        num_cells_l = mesh_l.topology.index_map(mesh_l.topology.dim).size_local 
        cells_l = np.arange(num_cells_l, dtype=np.int32)
        subdomains_l = dolfinx.mesh.meshtags(mesh_l, mesh_l.topology.dim, cells_l, sub_L)
        return subdomains_l, frame_l, boundary_facets_left
      
    subdomains_l, frame_l,boundary_facets_left=subdomains_boun(mesh_l,left,entity_mapl) # generating boundary submesh
    subdomains_r, frame_r,boundary_facets_right=subdomains_boun(mesh_r,right,entity_mapr)

    # Direction cosine matrix
    dc_matrix=as_tensor([(frame[0][0],frame[1][0],frame[2][0]),(frame[0][1],frame[1][1],frame[2][1]),(frame[0][2],frame[1][2],frame[2][2])])   

    def Rsig(dc_matrix):   # ROTATION MATRIX IN UFL Form 
        b11,b12,b13=dc_matrix[0,0],dc_matrix[0,1],dc_matrix[0,2]
        b21,b22,b23=dc_matrix[1,0],dc_matrix[1,1],dc_matrix[1,2]
        b31,b32,b33=dc_matrix[2,0],dc_matrix[2,1],dc_matrix[2,2]
        
        return as_tensor([(b11*b11, b12*b12, b13*b13, 2*b12*b13, 2*b11*b13,2* b11*b12),
                     (b21*b21, b22*b22, b23*b23, 2*b22*b23, 2*b21*b23, 2*b21*b22),
                     (b31*b31, b32*b32, b33*b33, 2*b32*b33, 2*b31*b33, 2*b31*b32),
                     (b21*b31, b22*b32, b23*b33, b23*b32+b22*b33, b23*b31+b21*b33, b22*b31+b21*b32),
                     (b11*b31, b12*b32, b13*b33, b13*b32+b12*b33, b13*b31+b11*b33, b12*b31+b11*b32),
                     (b11*b21, b12*b22, b13*b23, b13*b22+b12*b23, b13*b21+b11*b23, b12*b21+b11*b22)])
    
    def C(i,dc_matrix):  # Stiffness matrix
        E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
        S=np.zeros((6,6))
        S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
        S[0,1], S[0,2]= -v12/E1, -v13/E1
        S[1,0], S[1,2]= -v12/E1, -v23/E2
        S[2,0], S[2,1]= -v13/E1, -v23/E2
        S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12 
        CC=as_tensor(np.linalg.inv(S))
        R_sig=Rsig(dc_matrix)
        return dot(dot(R_sig,CC),R_sig.T) 
    #
    def mass_boun(x,dx): # Mass matrix
        mu= assemble_scalar(form(sum([density[i]*dx(i) for i in range(nphases)])))
        xm2=(1/mu)*assemble_scalar(form(sum([x[1]*density[i]*dx(i) for i in range(nphases)])))
        xm3=(1/mu)*assemble_scalar(form(sum([x[2]*density[i]*dx(i) for i in range(nphases)])))
        i22=assemble_scalar(form(sum([(x[2]**2)*density[i]*dx(i) for i in range(nphases)])))
        i33=assemble_scalar(form(sum([(x[1]**2)*density[i]*dx(i) for i in range(nphases)])))    
        i23=assemble_scalar(form(sum([x[1]*x[2]*density[i]*dx(i) for i in range(nphases)])))
        return np.array([(mu,0,0,0,mu*xm3,-mu*xm2),
                          (0,mu,0,-mu*xm3,0,0),
                          (0,0,mu,mu*xm2,0,0),
                          (0,-mu*xm3, mu*xm2, i22+i33, 0,0),
                          (mu*xm3, 0,0,0,i22,i23),
                          (-mu*xm2,0,0,0,i23,i33)])  
    
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
        aa,b=1,2
        if dim==2:
            E1=as_vector([0,v[1].dx(aa),v[2].dx(b),v[1].dx(b)+v[2].dx(aa),v[0].dx(b),v[0].dx(aa)])
        elif dim==3:
            E1=as_vector([v[0].dx(0),v[1].dx(aa),v[2].dx(b),v[1].dx(b)+v[2].dx(aa),v[0].dx(b)+v[b].dx(0),v[0].dx(aa)+v[aa].dx(0)])
    
        return E1
    
    def gamma_l(v): 
        return as_vector([v[0],0,0,0, v[2],v[1]])  
        
    def local_boun(mesh,frame,subdomains): 
        V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
            "CG", mesh.topology.cell_name(), deg, shape=(3, )))  #** Polynomial Order=1
        le1, le2, le3 = frame
        e1l,e2l,e3l=Function(V), Function(V), Function(V)
        
        fexpr1=dolfinx.fem.Expression(le1,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
        e1l.interpolate(fexpr1) 
        
        fexpr2=dolfinx.fem.Expression(le2,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
        e2l.interpolate(fexpr2) 
        
        fexpr3=dolfinx.fem.Expression(le3,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
        e3l.interpolate(fexpr3) 
            
        return [e1l,e2l,e3l], V, TrialFunction(V), TestFunction(V),SpatialCoordinate(mesh), \
              Measure('dx')(domain=mesh, subdomain_data=subdomains)   
    
    def gamma_e(x):   
        return as_tensor([(1,0,x[2],-x[1]),
                    (0,0,0,0),
                    (0,0,0,0),
                    (0,0,0,0),
                   (0,x[1],0,0),
                   (0,-x[2],0,0)])
    
    def A_mat(dx_l,v_l,dvl,dc_matrix_l):  # Boundary Coefficient matrix D_hh
        F2 = sum([dot(dot(C(i,dc_matrix_l),gamma_h(dx_l,dvl)),gamma_h(dx_l,v_l))*dx_l(i) for i in range(nphases)])       
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
    
    def apply_null_A(A_l,nullspace_basis):
        # Convert nullspace_l to a dense matrix N
        N = []
        for n in nullspace_basis:
            with n.localForm() as n_local:
                N.append(np.asarray(n_local))
        N = np.column_stack(N)  # Shape: (n_dofs, 6)
        N=N[:,0:4]
        
        # Create N N^T as a PETSc matrix
        n_dofs = A_l.getSize()[0]
        NNT = petsc4py.PETSc.Mat().create(comm=MPI.COMM_WORLD)
        NNT.setSizes([n_dofs, n_dofs])
        NNT.setType('aij')  # Sparse, since N N^T has structure
        NNT.setPreallocationNNZ(6)  # Each row has contributions from 6 RBMs
        NNT.setUp()
        
        # Create NNT with proper preallocation
        n_dofs = A_l.getSize()[0]
        NNT = petsc4py.PETSc.Mat().create(comm=MPI.COMM_WORLD)
        NNT.setSizes([n_dofs, n_dofs])
        NNT.setType('aij')
        # Preallocate: N N^T could have up to n_dofs nonzeros per row (dense-like)
        NNT.setPreallocationNNZ(n_dofs)  # Conservative: assumes dense
        # Alternative: Estimate nonzeros based on RBM sparsity (e.g., 100 if sparse)
        # NNT.setPreallocationNNZ(100)
        NNT.setUp()
        
        # Populate N N^T
        row_start, row_end = NNT.getOwnershipRange()
        for i in range(row_start, row_end):
            for j in range(n_dofs):
                val = np.dot(N[i, :], N[j, :])
                if abs(val) > 1e-12:  # Only set significant nonzeros
                    NNT.setValue(i, j, val, addv=petsc4py.PETSc.InsertMode.INSERT)
        
        NNT.assemble()
        
        # Scale and add
        lambda_val = 1e10
        NNT.scale(lambda_val)
        A_prime = A_l.copy()
        A_prime.axpy(1.0, NNT)
        A_prime.assemble()
        return A_prime
    
    def timo_boun(mesh_l,subdomains_l,fr): # Compute Boundary Timoshenko Solutions
        e, V_l, dv, v_, x, dx=local_boun(mesh_l,fr,subdomains_l)   
        dc_matrix_l=as_tensor([(fr[0][0],fr[1][0],fr[2][0]),(fr[0][1],fr[1][1],fr[2][1]),(fr[0][2],fr[1][2],fr[2][2])]) 
        mesh_l.topology.create_connectivity(2, 2)
        V0,Dle,Dhe,Dee,V1s=initialize_array(V_l)
        
        N, nullspace_basis=nullspace(V_l)
        nullspace_b=petsc4py.PETSc.NullSpace().create(vectors=nullspace_basis)
        A_l=A_mat(dx,v_,dv,dc_matrix_l) 
        A_l.setNullSpace(nullspace_b)
     #   A_prime=apply_null_A(A_l,nullspace_basis)  
        
        # Boundary Euler-Bernoulli 
        for i in range(4):
            F2 = sum([dot(dot(C(ii,dc_matrix_l),gamma_e(x)[:,i]),gamma_h(dx,v_))*dx(ii) for ii in range(nphases)])  
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
            Cc=C(i,dc_matrix_l)
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
        F1=sum([dot(dot(C(i,dc_matrix_l),gamma_l(dv)),gamma_l(v_))*dx(i) for i in range(nphases)])
        a1=form(F1)
        Dll=assemble_matrix(a1)
        Dll.assemble()
        ai, aj, av=Dll.getValuesCSR()
        Dll=csr_matrix((av, aj, ai))    
        
        for p in range(4):
                F1=sum([dot(dot(C(i,dc_matrix_l),gamma_e(x)[:,p]),gamma_l(v_))*dx(i) for i in range(nphases)])
                Dle[:,p]= petsc.assemble_vector(form(F1))[:]      
            
        F_dhl=sum([dot(dot(C(i,dc_matrix_l),gamma_h(dx,dv)),gamma_l(v_))*dx(i) for i in range(nphases)]) 
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
        
        return np.around(D_eff),np.around(Deff_srt),V0, V1s,Dhe,bb,A_l,C_tim,B_tim   
    dim=2
    deg=1
    DEB_l,Deff_l,V0_l,V1_l,bl1,bbl2,A_l,C_timl,B_timl=timo_boun(mesh_l,subdomains_l,frame_l) 
    left_timo.append(Deff_l)
    #******************************TIMO BOUNDARY SOLVE*******************************
    print('Left Timo \n')
    np.set_printoptions(precision=4) 
    print(Deff_l/1e4) 
    
    
    DEB_r,Deff_r,V0_r,V1_r,br2,br1,A_r,C_timr,B_timr=timo_boun(mesh_r,subdomains_r,frame_r)
    right_timo.append(Deff_r)
    print('Right Timo \n')
    np.set_printoptions(precision=4) 
    print(Deff_r) 
        
    e_l, V_l, dvl, v_l, x_l, dx_l=local_boun(mesh_l,frame_l,subdomains_l)
    e_r, V_r, dvr, v_r, x_r, dx_r=local_boun(mesh_r,frame_r,subdomains_r)  
    e,  V, dv,  v_,  x, dx=local_boun(mesh,frame,subdomains)  

    # ******************MASS MATRIX**********************************
    left_mass.append(mass_boun(x_l,dx_l))
    right_mass.append(mass_boun(x_r,dx_r))    
    taper_mass.append((1/L)*mass_boun(x,dx))
    dim=3
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
    
 
    F2 = sum([(dot(dot(C(i, dc_matrix),gamma_h(dx,dv)),gamma_h(dx,v_))+dot(dv,v_))*dx(i) for i in range(nphases)])   
    a= form(F2)
    
    N, nullspace_basis=nullspace(V)
    
    # bc applied
    boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((entity_mapr,entity_mapl), axis=0))
    v2a=Function(V)
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    A=assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
    A.assemble()
 #   A=apply_null_A(A,nullspace_basis)
    nullspace_b=petsc4py.PETSc.NullSpace().create(vectors=nullspace_basis)
 #   assert nullspace_b.test(A)

 #   A.setNullSpace(nullspace_b)   
    V0,Dle,Dhe,Dee,V1s=initialize_array(V)
    mesh.topology.create_connectivity(2, 3)
    mesh_l.topology.create_connectivity(2, 2)
    mesh_r.topology.create_connectivity(2, 2)
    
    def Dee_(i):
        Cc=C(i,dc_matrix)
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
       
       F2=-sum([dot(dot(C(i,dc_matrix),gamma_e(x)[:,p]),gamma_h(dx,v_))*dx(i) for i in range(nphases)]) 
       bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
       F = petsc.assemble_vector(form(F2)) 
       F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
    #   nullspace_b.remove(F)
    #   F_array = F[:]
    #   F_array = F.getArray()
     #  NTF = np.dot(N.T, F_array)  # N^T F
     #  F_array -= np.dot(N, NTF)  # Assumes N is orthonormal; see note below
     #  F.setArray(F_array)
     #  F.assemble()
    #   F=petsc4py.PETSc.Vec().createWithArray(F_array,comm=MPI.COMM_WORLD)
    #   F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE) 
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
 #   Eps=gamma_e(x)[:,0] 
 #   F1=sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx(i) for i in range(nphases)])
    F1=sum([dot(dot(C(i,dc_matrix),gamma_l(dv)),gamma_l(v_))*dx(i) for i in range(nphases)])
    a1=form(F1)
    Dll=assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av=Dll.getValuesCSR()
    Dll=csr_matrix((av, aj, ai))    
    
    for p in range(4):
           # Eps=gamma_e(x)[:,p] 
          #  F1=-sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx(i) for i in range(nphases)])
            F1=sum([dot(dot(C(i,dc_matrix),gamma_e(x)[:,p]),gamma_l(v_))*dx(i) for i in range(nphases)])
            Dle[:,p]= petsc.assemble_vector(form(F1))[:]
            
 #   F_dhl=sum([inner(sigma(dv,i,Eps,dx)[0],gamma_l(v_)[0])*dx(i) for i in range(nphases)])
    F_dhl=sum([dot(dot(C(i,dc_matrix),gamma_h(dx,dv)),gamma_l(v_))*dx(i) for i in range(nphases)])    
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
    b=(DhlTV0Dle-DhlV0).toarray()
  #  A=assemble_matrix(a)  # Obtain coefficient matrix with BC applied: AA
  #  A.assemble()
  #  A.setNullSpace(nullspace_b)
    
    for p in range(4): # 4 load cases meaning 
        # Boundary 
        v2a=Function(V) 
        v2a,dofs_l=dof_mapping_quad(v2a,V_l,V1_l[:,p],boundary_facets_left, entity_mapl)
        v2a,dofs_r=dof_mapping_quad(v2a,V_r,V1_r[:,p],boundary_facets_right,entity_mapr)  
        bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
        F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)   
        nullspace_b.remove(F)
     #   F_array = b[:,p]
     #   F_array = F[:]
     #   F_array = F.getArray()
     #   NTF = np.dot(N.T, F_array)  # N^T F
     #   F_array -= np.dot(N, NTF)  # Assumes N is orthonormal; see note below
     #   F.setArray(F_array)
     #   F.assemble()
    #    F=petsc4py.PETSc.Vec().createWithArray(F_array,comm=MPI.COMM_WORLD)
     #   F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)       
        apply_lifting(F, [a], [bc]) # apply bc to rhs vector (Dhe) based on known fluc solutions
        set_bc(F, bc)   
 
        V1s[:,p]=ksp_solve(A,F,V)
        
    V1s_csr=csr_matrix(V1s)
    B_tim=DhlTV0Dle.T.dot(V0_csr) #np.matmul(V0.T,Dhd)
    B_tim=B_tim.toarray()/L
    
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
    
    taper_timo.append(Deff_srt)
    print('\n Solid Taper Timo \n')
    
    np.set_printoptions(precision=4)
    print(np.around(Deff_srt))  


#********************************
######## Dehomogenization Begins ###############
#*********************************

area=assemble_scalar(form(1*dx_r))   # Right Boundary area where traction load applied

N0=1*area    # Input Load - 1N/m^2 traction at right end

# Input Global Strain
Comp_srt=np.linalg.inv(Deff_srt)
rf=[-N0,0,0,0,0,0] # VABS convention of reaction force
FF=np.array((rf[0],rf[1],rf[2],rf[3],rf[4],rf[5]))
st=np.matmul(Comp_srt,FF)   # Timoshenko beam strain 

st_m=np.array((st[0],st[3],st[4],st[5]), dtype=np.float64) # Euler-Bernoulli beam strain
                            
def recov(st):
    recov=np.zeros((6,6))
    recov[0,1],recov[0,2]=st[5],-st[4]
    recov[1,0],recov[1,2]=-st[5],st[3]
    recov[2,0],recov[2,1]=st[4],-st[3]
    recov[3:6,3:6]=recov[0:3,0:3]
    
    recov[3,1],recov[3,2]=st[2],-st[1]
    recov[4,0],recov[4,2]=-st[2],st[0]
    recov[5,0],recov[5,1]=st[1],-st[0]
    
    return recov
#
# Strain derivatives
# First derivative 

F_1d=np.matmul(Deff_srt,st)
R1=recov(st)
F1= np.matmul(R1,F_1d)
st_Tim1=np.matmul(Comp_srt,F1)

#
st_cl1=np.array([st_Tim1[0],st_Tim1[3],st_Tim1[4],st_Tim1[5]])
gamma1=np.array([st_Tim1[1],st_Tim1[2]])

# Second Derivative
R2=recov(st_Tim1)
F2=np.matmul(R1,F1)+np.matmul(R2,F_1d)
st_Tim2=np.matmul(Comp_srt,F2)

#
st_cl2=np.array([st_Tim2[0],st_Tim2[3],st_Tim2[4],st_Tim2[5]])
gamma2=np.array([st_Tim2[1],st_Tim2[2]])

# Using Warping Function data (displacements)
a1=np.matmul(V0,st_m) # disp_fluctuations 
a2=np.matmul(V1s,st_cl1) # disp_fluctuation from srt
a3=np.matmul(V1s,st_cl2) 
a4=np.matmul(V0,st_cl1)

V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "CG", mesh.topology.cell_name(), 1, shape=(3, )))  # Should be same degree 1 as homogenization degree

w_1=Function(V) # disp_fluctuations
w1s_1=Function(V) # disp_fluctuation from srt
w1s_2=Function(V)
w_2=Function(V) 

for i in range(len(V0[:,0])):
    w_1.vector[i],w1s_1.vector[i],w1s_2.vector[i],w_2.vector[i]=a1[i],a2[i],a3[i],a4[i] 

# 3D strain recovery
st_Eb=gamma_h(dx,w_1)+dot(gamma_e(x),as_vector((st_m)))
st_Timo=gamma_h(dx,w1s_1)+ gamma_l(w_2)+gamma_l(w1s_2)
st_3D=st_Eb+st_Timo                                   # Recovered 3D strain

V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "CG", mesh.topology.cell_name(), 1, shape=(6, ))) 
strain_3D=Function(V)            

fexpr1=dolfinx.fem.Expression(st_3D,V.element.interpolation_points(), comm=MPI.COMM_WORLD)
strain_3D.interpolate(fexpr1)  # Interpolate 3D strain to ufl function

CC=[]
for i in range(nphases):
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12 
    CC.append(np.linalg.inv(S))

stress_3D_dehom=Function(V) 
fexpr1=dolfinx.fem.Expression(dot(as_tensor(CC[0]),strain_3D),V.element.interpolation_points(), comm=MPI.COMM_WORLD)
stress_3D_dehom.interpolate(fexpr1) # Recovered 3D stress for isotropic 

V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "CG", mesh.topology.cell_name(),1, shape=(3, ))) 

u_ = TestFunction(V)
du = TrialFunction(V) # For displacement

def epsilon(u): 
    E1=as_vector([u[0].dx(0),u[1].dx(1),u[2].dx(2),(u[1].dx(2)+u[2].dx(1)),(u[0].dx(2)+u[2].dx(0)),(u[0].dx(1)+u[1].dx(0))])
    return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1

def sigma(u,i):  # v is strain variable (ufl)
    E1,E2,E3,G12,G13,G23,v12,v13,v23= material_parameters[i]
    S=np.zeros((6,6))
    S[0,0], S[1,1], S[2,2]=1/E1, 1/E2, 1/E3
    S[0,1], S[0,2]= -v12/E1, -v13/E1
    S[1,0], S[1,2]= -v12/E1, -v23/E2
    S[2,0], S[2,1]= -v13/E1, -v23/E2
    S[3,3], S[4,4], S[5,5]= 1/G23, 1/G13, 1/G12 
    s1=dot(as_tensor(CC[i]),epsilon(u)[1])
    return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),s1


# Dirichlet boundary conditions: Clamped and simply supported
fdim = mesh.topology.dim - 1
u_L = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
bcs = [
    dirichletbc(u_L, locate_dofs_topological(V, fdim, entity_mapl), V),
    dirichletbc(
        dolfinx.default_scalar_type(0),
        locate_dofs_topological(V.sub(1), fdim, entity_mapr),
        V.sub(1),
    ),
    dirichletbc(
        dolfinx.default_scalar_type(0),
        locate_dofs_topological(V.sub(2), fdim, entity_mapr),
        V.sub(2),
    )
] 

# Linear Elasticity Bilinear Form
a = sum([dot(sigma(du,i)[1],epsilon(u_)[1])*dx(i) for i in range(nphases)])
# Stiffness matrix
K = assemble_matrix(form(a), bcs=bcs, diagonal=1)
K.assemble() 

def sigma_prestress(i):
    s_pre=dot(as_tensor(CC[i]),strain_3D)
    return as_tensor([(s_pre[0],s_pre[5],s_pre[4]),
                      (s_pre[5],s_pre[1],s_pre[3]),
                      (s_pre[4],s_pre[3],s_pre[2])]) 

kgform = -sum([inner(sigma_prestress(i),grad(du).T*grad(u_))*dx(i) for i in range(nphases)])
KG = assemble_matrix(form(kgform), bcs=bcs, diagonal=0)
KG.assemble()    # epsilon(du) and grad(du) both are same  

from eigenvalue_solver import solve_GEP_shiftinvert, EPS_get_spectrum

# Requested number of eigenvalues
N_eig = 6

# Solve eigenvalue problem
eigensolver = solve_GEP_shiftinvert(
    K,
    KG,
    problem_type=SLEPc.EPS.ProblemType.GHIEP,
    solver=SLEPc.EPS.Type.KRYLOVSCHUR,
    nev=N_eig,
    tol=1e-6,
    shift=1e3,
)

# Extract eigenpairs
(eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(eigensolver, V) 

import pyvista
pyvista.start_xvfb()
pyvista.set_jupyter_backend("static")

# Grid for the mesh
tdim = mesh.topology.dim
mesh_topology, mesh_cell_types, mesh_geometry = dolfinx.plot.vtk_mesh(mesh, tdim)
mesh_grid = pyvista.UnstructuredGrid(mesh_topology, mesh_cell_types, mesh_geometry)

# Grid for functions (2nd order elements)
u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)

# Plot the first 3 eigenmodes
#pl = pyvista.Plotter(shape=(1, 1))

for i in range(4):
    #    pl.subplot(1 , 0)
        pl = pyvista.Plotter(shape=(1, 1))
        eigenmode = f"eigenmode_{i:02}"
        pl.add_text(
            f"Eigenmode {i+1}",
            font_size=12,
        )
        eigen_vector = eigvec_r[i]
        u_grid[eigenmode] = eigen_vector.x.array.reshape(
            u_geometry.shape[0], V.dofmap.index_map_bs
        )
        #pl.add_mesh(mesh_grid, style="wireframe")
        pl.add_mesh(u_grid.warp_by_vector(eigenmode, factor=40), show_scalar_bar=False)
        pl.view_isometric()

  
        # Save the plot as a PNG
        pl.screenshot(f"eigenmode_{i+1:02}.png")
        pl.show() 
        # Close the plotter to free memory
        pl.close()
