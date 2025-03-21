#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############     OpenSG         ############################
########### Tapered Timo Model (Solid Elements) ###################

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem, assemble_matrix 
from dolfinx.mesh import locate_entities_boundary, exterior_facet_indices, create_submesh
from mpi4py import MPI
import numpy as np
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
import time
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
        beam_inertia[i_station,:,:]=trsf_sixbysix(beam_inertia[i_station,:,:], T)
   
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

# In[ ]:

# for segment in range(2):
for segment in np.linspace(0,27,28):  
    start_time = time.perf_counter()  # Start timer
    # ****************OBTAIN MESH DATA FROM YAML***********************
    meshYaml = '/Users/ecamare/myprojects/se_project/verification/solid/mesh/bar_urc_npl_1_ar_5-segment_'+ str(int(segment)) +'.yaml'  ## the name of the yaml file containing the whole blade mesh
    mshFile = 'SG_solid.msh'
    oriFile = 'solid.orientation'
    ## Read the mesh yaml file
    
    inFile = open(meshYaml,'r')
    meshData = yaml.load(inFile,Loader=cLd)
    inFile.close()
    ## Extract the mesh for the section
    nodes = meshData['nodes']
    numNds = len(nodes)
    elements = meshData['elements']
    numEls = len(elements)
    elLayID = np.ones(numEls,dtype=int)
    elLayID=[]
    mat_name=[]
    layCt=-1
    for es in meshData['sets']['element']:
        if es['labels'] is not None:
            mat_name.append(es['name'])
            layCt += 1
            for eli in es['labels']:
                elLayID.append(layCt)
                
    material_parameters, density=[], []
    for es in meshData['materials']:
        if es['name'] in mat_name:
            material_parameters.append(np.array((np.array(es['E']),np.array(es['G']),es['nu'])).flatten())
            density.append(es['rho'])
            
    ## Write .msh file
    outFile = open(mshFile,'w')
    
    outFile.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n')
    outFile.write(str(len(nodes)) + '\n')
    
    for i, nd in enumerate(nodes):
            ndd=nd[0].split()
            ln = [str(i+1),str(ndd[2]),str(ndd[0]),str(ndd[1])] # Making x-axis as beam axis
            outFile.write(' '.join(ln) + '\n')
    
    outFile.write('$EndNodes\n$Elements\n')
    
    newNumEls = len(elements)
    outFile.write(str(newNumEls) + '\n')
    
    for i, el in enumerate(elements):
            ln = [str(i+1)]
            ln.append('5')
            ln.append('2')
            ln.append(str(elLayID[i]+1))   
            ln.append(str(elLayID[i]+1))    
            ell=el[0].split()
            for n in ell:
                ln.append(n)
            outFile.write(' '.join(ln) + '\n')
    outFile.write('$EndElements\n')
    
    outFile.close()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)    # mesh read output will not be printed
    mesh, subdomains, boundaries = gmshio.read_from_msh("SG_solid.msh", MPI.COMM_WORLD,0, gdim=3) 
    
    lnn=[]
    #for k in o_cell_idx:
       #  lnn.append(elLayID[k])
       #  lnn.append(0)
    lnn=np.array(subdomains.values[:]-1,dtype=np.int32)    
    
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    subdomains = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, lnn)
    
    pp=mesh.geometry.x
    
    # ************Shifting origin to mid-length along axis***************************
    # ************Origin on beam axis*************************************
    
    x_min,x_max=min(pp[:,0]), max(pp[:,0])
    left_origin.append(float(x_min)/blade_length),right_origin.append(float(x_max)/blade_length),taper_origin.append(0.5*(float(x_max)+float(x_min))/blade_length)
    L=x_max-x_min
    pp[:,0]=pp[:,0]-0.5*(x_max+x_min)
    x_min,x_max=min(pp[:,0]), max(pp[:,0])

     

    print('Segment',str(int(segment)),'['+str(num_cells),'elements] \n')
    # ***********GENERATE BOUNDARY MESH************************
    
    fdim=2
    def left(x):
        return np.isclose(x[0], x_min,atol=0.02)
    def right(x):
        return np.isclose(x[0], x_max,atol=0.02)
    
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
    # Storing data to DG0 functions 
        EE2.vector[3*k],EE2.vector[3*k+1],EE2.vector[3*k+2]=orien[ii][5],orien[ii][3],orien[ii][4]   # e2
        N.vector[3*k],N.vector[3*k+1],N.vector[3*k+2]=orien[ii][8],orien[ii][6],orien[ii][7]   #  e3 
        EE1.vector[3*k], EE1.vector[3*k+1],EE1.vector[3*k+2]=orien[ii][2],orien[ii][0],orien[ii][1]  # e1    
    frame=[EE1,EE2,N]
    
    
    # Boundary
    
    # **********************MAP layup data to boundary*************************
    
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
                            El1.x.array[3*i+j]=frame[0].x.array[3*idx+j]
                            El2.x.array[3*i+j]=frame[1].x.array[3*idx+j]
                            Nl.x.array[3*i+j] =frame[2].x.array[3*idx+j]
    
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
    
    #pyvista.start_xvfb()
   # u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(mesh_l,mesh_l.topology.dim)
   # grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
   # grid.cell_data["Marker"] = subdomains_l.values[:]
   # grid.set_active_scalars("Marker")
   # u_plotter = pyvista.Plotter()
   # u_plotter.add_mesh(grid, show_edges=True)
   # points=mesh.geometry.x 
    #u_plotter.show_axes()
    #u_plotter.add_axes_at_origin()
   # u_plotter.view_yz() # z is beam axis
   # u_plotter.show() 
    
   # pyvista.start_xvfb()
   # u_topology, u_cell_types, u_geometry=dolfinx.plot.vtk_mesh(mesh,mesh.topology.dim)
   # grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
   # grid.cell_data["Marker"] = subdomains.values[:]
   # grid.set_active_scalars("Marker")
   # u_plotter = pyvista.Plotter()
   # u_plotter.add_mesh(grid, show_edges=True)
   # points=mesh.geometry.x 
    #u_plotter.show_axes()
    #u_plotter.add_axes_at_origin()
    #u_plotter.view_yz() # z is beam axis
  #  u_plotter.show() 
    #def Rsig(dc_matrix,th): 
       # lay=th   #-45
       # c,s=np.cos(np.deg2rad(lay)),np.sin(np.deg2rad(lay))
       # b11,b12,b13=c,s,0
       # b21,b22,b23=-s,c,0
       # b31,b32,b33=0,0,1
      #  dc_matrix=dc_matrix*as_tensor([(c,s,0),(-s,c,0),(0,0,1)])
    
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
        return w.vector[:],w
    
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
        return petsc4py.PETSc.NullSpace().create(nullspace_basis, comm=MPI.COMM_WORLD)
    
    
    def gamma_h(dx,v): 
            return as_vector([0,v[1].dx(1),v[2].dx(2),v[1].dx(2)+v[2].dx(1),v[0].dx(2),v[0].dx(1)])
    
    def gamma_l(v): 
        return as_vector([v[0],0,0,0, v[2],v[1]])  
        
    def local_boun(mesh,frame,subdomains): 
        V = dolfinx.fem.functionspace(mesh, basix.ufl.element(
            "CG", mesh.topology.cell_name(), 1, shape=(3, )))  #** Polynomial Order=1
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

    # Initialize terms                                        
    e_l, V_l, dvl, v_l, x_l, dx_l=local_boun(mesh_l,frame_l,subdomains_l)
    e_r, V_r, dvr, v_r, x_r, dx_r=local_boun(mesh_r,frame_r,subdomains_r)
    e,  V, dv,  v_,  x, dx=local_boun(mesh,frame,subdomains)  
    
    
    def A_mat(dx_l,nullspace_l,v_l,dvl,dc_matrix_l):  # Boundary Coefficient matrix D_hh
        F2 = sum([dot(dot(C(i,dc_matrix_l),gamma_h(dx_l,dvl)),gamma_h(dx_l,v_l))*dx_l(i) for i in range(nphases)])       
        A_l=assemble_matrix(form(F2))
        A_l.assemble()
        B=assemble_matrix(form(F2))
        B.assemble()
        A_l.setNullSpace(nullspace_l) 
        return A_l,B
    
    def initialize_array(V): 
        xxx=3*V.dofmap.index_map.local_range[1]  # total dofs 
        V0 = np.zeros((xxx,4))
        Dle=np.zeros((xxx,4))
        Dhe=np.zeros((xxx,4)) 
        Dee=np.zeros((4,4)) 
        V1s=np.zeros((xxx,4))
        return V0,Dle,Dhe,Dee,V1s
    
    def timo_boun(mesh_l,subdomains_l,fr): # Compute Boundary Timoshenko Solutions
        e, V_l, dv, v_, x, dx=local_boun(mesh_l,fr,subdomains_l)   
        dc_matrix_l=as_tensor([(fr[0][0],fr[1][0],fr[2][0]),(fr[0][1],fr[1][1],fr[2][1]),(fr[0][2],fr[1][2],fr[2][2])])   
        mesh_l.topology.create_connectivity(2, 2)
        V0,Dle,Dhe,Dee,V1s=initialize_array(V_l)
        nullspace_l=nullspace(V_l)
        A_l, B=A_mat(dx,nullspace_l,v_,dv,dc_matrix_l) 
        v0=[]
        
        # Boundary Euler-Bernoulli 
        for p in range(4):
            F2 = sum([dot(dot(C(i,dc_matrix_l),gamma_e(x)[:,p]),gamma_h(dx,v_))*dx(i) for i in range(nphases)])  
            r_he=form(F2)
            F_l = petsc.assemble_vector(r_he)
            F_l.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
           # nullspace_l.remove(F_l)
            Dhe[:,p]=  F_l[:]
            V0[:,p],v= ksp_solve(A_l,F_l,V_l)  
            v0.append(v)
        
      #  ai, aj, av=A_l.getValuesCSR()  
      #  A_l=csr_matrix((av, aj, ai))
        
        ai, aj, av=A_l.getValuesCSR()  
        A_l=csr_matrix((av, aj, ai))
        epsilon = scipy.sparse.linalg.eigsh(A_l, k=1, which='LA')[0]/10**8
        # small regularization parameter  for ill-conditioned coeff matrix
        A_reg = A_l + epsilon * np.eye(A_l.shape[0])
        
        V0=scipy.sparse.linalg.spsolve(csr_matrix(A_reg), Dhe, permc_spec=None, use_umfpack=True)      
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
                Dle[:,p]= petsc.assemble_vector(form(rhs(F1)))[:]      
            
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
        
        for p in range(4): # 4 load cases meaning 
            F=petsc4py.PETSc.Vec().createWithArray(bb[:,p],comm=MPI.COMM_WORLD)
            F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)        
          #  nullspace_l.remove(F)
            Dhe[:,p]=F
            
        V1s=scipy.sparse.linalg.spsolve(csr_matrix(A_reg), Dhe, permc_spec=None, use_umfpack=True)     
        
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
        
        return np.around(D_eff),np.around(Deff_srt),V0, V1s,epsilon[0]
        
    # ******************MASS MATRIX**********************************
    print('\n Left Mass Matrix \n')
    np.set_printoptions(precision=4)    
    print(mass_boun(x_l,dx_l))
    
    D_effEB_l,Deff_l,V0_l,V1_l,ep1=timo_boun(mesh_l,subdomains_l,frame_l) 
    left_timo.append(Deff_l)
    #******************************TIMO BOUNDARY SOLVE*******************************
    print('\n Left Timo \n')
    np.set_printoptions(precision=4) 
    print(Deff_l) 
    
    D_effEB_r,Deff_r,V0_r,V1_r,ep2=timo_boun(mesh_r,subdomains_r,frame_r)
    right_timo.append(Deff_r)
    print('Right Timo \n')
    np.set_printoptions(precision=4) 
    print(np.around(Deff_r)) 
    
    t1 = time.perf_counter()
    print('\n Computed time for boundary:', str(t1-start_time))

# *************TAPER STARTS*************
    print('\n Taper Mass Matrix \n')
    np.set_printoptions(precision=4)    
    print((1/L)*mass_boun(x,dx))
    
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
    
    def gamma_l(v):
        E1= as_vector([v[0],0,0,0, v[2],v[1]])
        return as_tensor([(E1[0],0.5*E1[5],0.5*E1[4]),(0.5*E1[5],E1[1],0.5*E1[3]),(0.5*E1[4],0.5*E1[3],E1[2])]),E1
    
    def sigma(v, i,Eps,dx):
        s1= dot(C(i,dc_matrix),gamma_h(dx,v)+Eps)
        return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])]),C
    
    def sigma_gl(v, i, Eps):
        s1= dot(C(i,dc_matrix),gamma_l(v)[1]+Eps)
        return as_tensor([(s1[0],s1[5],s1[4]),(s1[5],s1[1],s1[3]),(s1[4],s1[3],s1[2])])
    
    
    #******************TAPER INITIALIZE***************************************
    
    e,  V, dv,  v_,  x, dx=local_boun(mesh,frame,subdomains)  
    F2 = sum([dot(dot(C(i, dc_matrix),gamma_h(dx,dv)),gamma_h(dx,v_))*dx(i) for i in range(nphases)])   
    a= form(F2+ep1*dot(dv,v_)*dx)
    
    # bc applied
    boundary_dofs = locate_dofs_topological(V, fdim, np.concatenate((entity_mapr,entity_mapl), axis=0))
    v2a=Function(V)
    bc = dolfinx.fem.dirichletbc(v2a, boundary_dofs)
    A=assemble_matrix(a,[bc])  # Obtain coefficient matrix with BC applied: AA
    A.assemble()
    
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
    v0=[]
    for p in range(4): # 4 load cases meaning 
       # Boundary 
       v2a=Function(V) 
       v2a,dofs_l=dof_mapping_quad(v2a,V_l,V0_l[:,p],boundary_facets_left,entity_mapl)
       v2a,dofs_r=dof_mapping_quad(v2a,V_r,V0_r[:,p],boundary_facets_right,entity_mapr)  
       
       F2=sum([dot(dot(C(i,dc_matrix),gamma_e(x)[:,p]),gamma_h(dx,v_))*dx(i) for i in range(nphases)]) 
       bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
       F = petsc.assemble_vector(form(F2))
       
       Dhe[:,p]=F
       apply_lifting(F, [a], [bc]) # apply bc to rhs vector (Dhe) based on known fluc solutions
       set_bc(F, bc)         
       V0[:,p],v=ksp_solve(A,F,V)
       v0.append(v)
    
    V0_csr=csr_matrix(V0)    
    D1=V0_csr.T.dot(csr_matrix(-Dhe)).astype(np.float64)
    for s in range(4):
       for k in range(4): 
           f=dolfinx.fem.form(sum([Dee_(i)[s,k]*dx(i) for i in range(nphases)]))
           Dee[s,k]=dolfinx.fem.assemble_scalar(f)
           
    D_eff= (Dee.astype(np.float64) + D1)/(x_max-x_min)          
    D_eff=0.5*(D_eff+D_eff.T)
    
    #*****************TAPER TIMOSHENKO**********************************
    
    Eps=gamma_e(x)[:,0] 
    F1=sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx(i) for i in range(nphases)])
    a1=form(lhs(F1))
    Dll=assemble_matrix(a1)
    Dll.assemble()
    ai, aj, av=Dll.getValuesCSR()
    Dll=csr_matrix((av, aj, ai))    
    
    for p in range(4):
            Eps=gamma_e(x)[:,p] 
            F1=sum([inner(sigma_gl(dv,i,Eps),gamma_l(v_)[0])*dx(i) for i in range(nphases)])
            Dle[:,p]= petsc.assemble_vector(form(rhs(F1)))[:]
        
    F_dhl=sum([inner(sigma(dv,i,Eps,dx)[0],gamma_l(v_)[0])*dx(i) for i in range(nphases)])
    a3=form(lhs(F_dhl))
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
    
    for p in range(4): # 4 load cases meaning 
        # Boundary 
        v2a=Function(V) 
        v2a,dofs_l=dof_mapping_quad(v2a,V_l,V1_l[:,p],boundary_facets_left, entity_mapl)
        v2a,dofs_r=dof_mapping_quad(v2a,V_r,V1_r[:,p],boundary_facets_right,entity_mapr)  
        bc = [dolfinx.fem.dirichletbc(v2a, boundary_dofs)]
        
        F=petsc4py.PETSc.Vec().createWithArray(b[:,p],comm=MPI.COMM_WORLD)
        F.ghostUpdate(addv=petsc4py.PETSc.InsertMode.ADD, mode=petsc4py.PETSc.ScatterMode.REVERSE)
    
        apply_lifting(F, [a], [bc]) 
        set_bc(F, bc)          
        V1s[:,p]=ksp_solve(A,F,V)[0] 
        
    V1s_csr=csr_matrix(V1s)
    L=(x_max-x_min)
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
    
    end_time = time.perf_counter()
    print('Computed time for Taper:', str(end_time-t1),'\n')
    print('Total Time for Segment',str(int(segment)),'is:', str(end_time-start_time),'\n')    
    
    taper_timo.append(Deff_srt)
    print('\n Taper Timo Stiffness ')
    
    np.set_printoptions(precision=4)
    print(np.around(Deff_srt),'\n')  
    # left_timo.append(np.zeros([6,6]))
    # left_mass.append(np.zeros([6,6]))
    
    # right_timo.append(np.zeros([6,6]))
    # right_mass.append(np.zeros([6,6]))

    # taper_timo.append(np.zeros([6,6]))
    # taper_mass.append(np.zeros([6,6]))
print('\nleft_origin')
print(left_origin)

print('\nright_origin')
print(right_origin)

print('\ntaper_origin')
print(taper_origin)
#Append tip to left boundary values
left_timo.append(right_timo[-1])
left_origin.append(right_origin[-1])
left_mass.append(right_mass[-1])
print('\n\nleft_origin ###############')
print(left_origin)

#Prepend root to right boundary values
right_timo.insert(0,left_timo[0])
right_origin.insert(0,left_origin[0])
right_mass.insert(0,left_mass[0])
print('\n\nright_origin ###############')
print(right_origin)

#Prepend root to segment values
taper_timo.insert(0,left_timo[0])
taper_origin.insert(0,left_origin[0])
taper_mass.insert(0,left_mass[0])
print('\n\ntaper_origin ###############')
print(taper_origin)

#Append tip to segment values
taper_timo.append(right_timo[-1])
taper_origin.append(right_origin[-1])
taper_mass.append(right_mass[-1])
print('\n\ntaper_origin ###############')
print(taper_origin)

left_timo=np.array(left_timo)
right_timo=np.array(right_timo)
taper_timo=np.array(taper_timo)

left_mass=np.array(left_mass)
right_mass=np.array(right_mass)
taper_mass=np.array(taper_mass)

print(taper_origin)
write_beamdyn_files(taper_timo, taper_mass, taper_origin,'bar_urc_segment')
write_beamdyn_files(right_timo, right_mass, right_origin,'bar_urc_right')
write_beamdyn_files(left_timo, left_mass, left_origin,'bar_urc_left')






