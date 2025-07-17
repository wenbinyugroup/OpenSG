# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:04:59 2025

@author: bagla0
"""

import numpy as np
import dolfinx
import ufl
import opensg
import basix
from mpi4py import MPI
from slepc4py import SLEPc
    
def beam_reaction(file_name):

    data=np.loadtxt(file_name +'.out', delimiter=',', skiprows=0, dtype=str)
    index=data[1].split()
    last_data=data[-1].split()
    pp=7
    #beam_f=[[[index[pp+k],float(last_data[pp+k])] for k in range(6)]] ;if root also needed
    beam_force=[]
    num_segment=int((len(index)-13)/15)
    pp=13

    for seg in range(num_segment):
        beam_seg_reac=[]
        for f in range(6):
            sc=pp+30*(f)+seg
            if f>2:
                sc=pp+num_segment*(3+f)+seg
            beam_seg_reac.append([index[sc],float(last_data[sc])])
          #  print(f,index[sc])
        beam_force.append(beam_seg_reac)
          
    return beam_force


def recover_local_strain(timo,beam_force,segment,meshdata):
    Deff_srt,V0,V1=timo[:]
    V=dolfinx.fem.functionspace(meshdata["mesh"], basix.ufl.element(
        "CG", meshdata["mesh"].topology.cell_name(), 1, shape = (3, )))
    VV=dolfinx.fem.functionspace(meshdata["mesh"], basix.ufl.element(
        "CG", meshdata["mesh"].topology.cell_name(), 1, shape = (6, )))
    
    x,dx = ufl.SpatialCoordinate(meshdata["mesh"]), ufl.Measure('dx')(domain=meshdata["mesh"], subdomain_data=meshdata["subdomains"])
    
    rf=[beam_force[int(segment)][k][1] for k in range(6)]
    print('Beam reaction Force')
    for k in range(6):
        print(beam_force[int(segment)][k][0],'     ',beam_force[int(segment)][k][1]  )          
    FF=np.array((rf[2],rf[0],rf[1],rf[5],rf[3],rf[4])) # BeamDyn --> VABS convention 

    conv=['F1','F2','F3', 'M1','M2','M3'] 
    #print('VABS Input Reaction Force')
    #for k in range(6):
     #   print(conv[k], FF[k])

    Comp_srt=np.linalg.inv(Deff_srt)
    st=np.matmul(Comp_srt,FF) 
    
    st_m=np.array((st[0],st[3],st[4],st[5]), dtype=np.float64)  
    
    # First Derivative
    F_1d=np.matmul(Deff_srt,st)
    R1=opensg.recov(st)
    F1= np.matmul(R1,F_1d)
    st_Tim1=np.matmul(Comp_srt,F1)
    st_cl1=np.array([st_Tim1[0],st_Tim1[3],st_Tim1[4],st_Tim1[5]])
 #   gamma1=np.array([st_Tim1[1],st_Tim1[2]])

    # Second Derivative
    R2=opensg.recov(st_Tim1)
    F2=np.matmul(R1,F1)+np.matmul(R2,F_1d)
    st_Tim2=np.matmul(Comp_srt,F2)    
    st_cl2=np.array([st_Tim2[0],st_Tim2[3],st_Tim2[4],st_Tim2[5]])
  #  gamma2=np.array([st_Tim2[1],st_Tim2[2]])

    # Using Warping Function data (displacements)
    a1=np.matmul(V0,st_m) # disp_fluctuations 
    a2=np.matmul(V1,st_cl1) # disp_fluctuation from srt
    a3=np.matmul(V1,st_cl2) 
    a4=np.matmul(V0,st_cl1)
    
    w_1=dolfinx.fem.Function(V) # disp_fluctuations
    w1s_1=dolfinx.fem.Function(V) # disp_fluctuation from srt
    w1s_2=dolfinx.fem.Function(V)
    w_2=dolfinx.fem.Function(V)     
    
    for i in range(len(V0[:,0])):
        w_1.x.array[i],w1s_1.x.array[i],w1s_2.x.array[i],w_2.x.array[i]=a1[i],a2[i],a3[i],a4[i] 
    
    # 3D strain recovery
    st_Eb=opensg.gamma_h(dx,w_1,dim=3)+ufl.dot(opensg.gamma_e(x),ufl.as_vector((st_m))) # EB Contribution
    st_Timo=opensg.gamma_h(dx,w1s_1,dim=3)+opensg. gamma_l(w_2)+opensg.gamma_l(w1s_2)  # Timo Contribution
    st_3D=st_Eb+st_Timo     
    
    strain_3D=dolfinx.fem.Function(VV)

    fexpr1=dolfinx.fem.Expression(st_3D,VV.element.interpolation_points(), comm=MPI.COMM_WORLD)
    strain_3D.interpolate(fexpr1) 
    
    return strain_3D    
    
def local_stress(mat_param,segment_mesh,strain_3D,points):
     mesh=segment_mesh.meshdata["mesh"]
     CC_=opensg.CC(mat_param)  
     
     V_stiff=dolfinx.fem.functionspace(mesh, basix.ufl.element(
                 "DG", mesh.topology.cell_name(), 0, shape = (6,6 )))    
     # UFL Stiffness
     stiffness=dolfinx.fem.Function(V_stiff)
     for i,sub in enumerate(segment_mesh.meshdata["subdomains"].values):
         stiffness.x.array[36*i:36*i+36]=CC_[sub].flatten()  
     V_stress = dolfinx.fem.functionspace(mesh, basix.ufl.element(
        "CG", mesh.topology.cell_name(), 2, shape=(6,))) 
     
     stress_3D=dolfinx.fem.Function(V_stress)
     fexpr1 = dolfinx.fem.Expression(stiffness*strain_3D,V_stress.element.interpolation_points(), comm = MPI.COMM_WORLD)
     stress_3D.interpolate(fexpr1) 
     stress_eval=opensg.stress_output(mat_param,mesh,stress_3D,points)
     return stress_eval
 
def eigen_stiffness_matrix(mat_param,segment_mesh,strain_3D, N_eig):
    mesh=segment_mesh.meshdata["mesh"]
    
    dx = ufl.Measure('dx')(domain=mesh, subdomain_data=segment_mesh.meshdata["subdomains"])

    CC=opensg.CC(mat_param)  
    nphases=len(mat_param)
    V=dolfinx.fem.functionspace(mesh, basix.ufl.element(
            "CG", mesh.topology.cell_name(), 1, shape = (3, )))
    du,u_=ufl.TrialFunction(V), ufl.TestFunction(V)
 
    u_L = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
    bcs = [
        dolfinx.fem.dirichletbc(u_L, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, segment_mesh.left_submesh["entity_map"]), V),
        dolfinx.fem.dirichletbc(u_L, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, segment_mesh.right_submesh["entity_map"]), V)] # Constrained both ends
    # Taking no contribution from end boundary
    
    # Linear Elasticity Bilinear Form
    a = sum([ufl.dot(opensg.sigma(du,i,CC)[1],opensg.epsilon(u_)[1])*dx(i) for i in range(nphases)])
    # Stiffness matrix
    K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=bcs)
    K.assemble()     

    kgform = -sum([ufl.inner(opensg.sigma_prestress(i,CC,strain_3D)[0],ufl.grad(du).T*ufl.grad(u_))*dx(i) for i in range(nphases)])
    KG = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(kgform), bcs=bcs, diagonal=0)
    KG.assemble()    # epsilon(du) and grad(du) both are same      
    
    eigensolver = opensg.solve_GEP_shiftinvert(
    KG,   # Optimization Problem: Find maximum eigenvalue of this problem
    K,
    problem_type=SLEPc.EPS.ProblemType.GHIEP,
    solver=SLEPc.EPS.Type.KRYLOVSCHUR, 
    nev=N_eig,
    tol=1e-8,
  #  shift=1,
    )
    # Extract eigenpairs
    (eigval, eigvec_r, eigvec_i) = opensg.EPS_get_spectrum(eigensolver, V) 
    print('Critical Eigen val:', 1/(np.max(eigval)).real)

  #  pyvista.start_xvfb()
  #  pyvista.set_jupyter_backend("static")
    
    # Grid for the mesh
  #  tdim = mesh.topology.dim
  #  mesh_topology, mesh_cell_types, mesh_geometry = dolfinx.plot.vtk_mesh(mesh, tdim)
  #  mesh_grid = pyvista.UnstructuredGrid(mesh_topology, mesh_cell_types, mesh_geometry)
    
    # Grid for functions (2nd order elements)
  #  u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
   # u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    
    # Plot the first 3 eigenmodes
    #pl = pyvista.Plotter(shape=(1, 1))
    
   # for i in range(2):
        #    pl.subplot(1 , 0)
         #   pl = pyvista.Plotter(shape=(1, 1),off_screen=True)
         #   eigenmode = f"eigenmode_{i:02}"
         #   pl.add_text(
         #       f"Eigenmode {i+1}",
         #       font_size=12,
         #   )
        #    eigen_vector = eigvec_r[i]
        #    u_grid[eigenmode] = eigen_vector.x.array.reshape(
        #        u_geometry.shape[0], V.dofmap.index_map_bs
       #     )
            #pl.add_mesh(mesh_grid, style="wireframe")
      #      pl.add_mesh(u_grid.warp_by_vector(eigenmode, factor=10), show_scalar_bar=False)
      #      pl.view_isometric()
    
        
            # Save the plot as a PNG
     #       pl.screenshot(f"eigenmode_{i+1:02}.png")
    #        pl.show() 
            # Close the plotter to free memory
    #        pl.close()
    return eigval    

    
