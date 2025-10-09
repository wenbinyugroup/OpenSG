# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:04:59 2025

@author: bagla0
"""

import numpy as np
import dolfinx
import ufl
import basix
from mpi4py import MPI
from slepc4py import SLEPc
from dolfinx.fem import Function
import opensg.utils.solid as utils
#from petsc4py import PETSc
#import pyvista

def local_strain(timo, beam_out, segment, meshdata):
    """Recover local strain field from beam analysis results.

    This function recovers the local 3D strain field within a blade segment
    using the fluctuating functions from the homogenized beam analysis and
    the applied beam forces.

    Parameters
    ----------
    timo : list
        Timoshenko analysis results containing [Deff_srt, V0, V1] where:
        - Deff_srt: effective stiffness matrix
        - V0: boundary fluctuating functions
        - V1: volume fluctuating functions
    beam_force : list
        Beam force components [Fx, Fy, Fz, Mx, My, Mz] applied to the segment
    segment : int
        Segment index for identification
    meshdata : dict
        Dictionary containing mesh information including the 3D mesh object

    Returns
    -------
    dolfinx.fem.Function
        Function containing the recovered 3D strain field in the segment
    """
    Deff_srt,V0, V1 = timo[0],timo[3],timo[4]
    beam_force,beam_disp=beam_out
    V = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(3,)),
    )
    VV = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(6,)),
    )

    x, dx = (
        ufl.SpatialCoordinate(meshdata["mesh"]),
        ufl.Measure("dx")(
            domain=meshdata["mesh"], subdomain_data=meshdata["subdomains"]
        ),
    )
    segment=segment+1
    rf = [beam_force[int(segment)][k][1] for k in range(6)]
  #  print("Beam reaction Force")
  #  for k in range(6):
     #   print(beam_force[int(segment)][k][0], "     ", beam_force[int(segment)][k][1])
    FF = np.array(
        (rf[2], -rf[1], rf[0], rf[5], -rf[4], rf[3])
    )  # BeamDyn --> VABS convention
  #  print('VABS Reaction Force:',FF)

    Comp_srt = np.linalg.inv(Deff_srt)
    st=np.matmul(Comp_srt,FF) 
    st_m=np.array((st[0],st[3],st[4],st[5]), dtype=np.float64)  
    
    # First Derivative
    F_1d=np.matmul(Deff_srt,st)
    R1=utils.recov(np.array((st[0]+1,st[1],st[2],st[3],st[4],st[5]), dtype=np.float64) )
    F1= np.matmul(R1,F_1d)
    st_Tim1=np.matmul(Comp_srt,F1)
    st_cl1=np.array([st_Tim1[0],st_Tim1[3],st_Tim1[4],st_Tim1[5]])
    gamma1=np.array([st_Tim1[1],st_Tim1[2]])
    
    # Second Derivative
    R2=utils.recov(st_Tim1)
    F2=np.matmul(R1,F1)+np.matmul(R2,F_1d)
    st_Tim2=np.matmul(Comp_srt,F2)    
    st_cl2=np.array([st_Tim2[0],st_Tim2[3],st_Tim2[4],st_Tim2[5]])
    gamma2=np.array([st_Tim2[1],st_Tim2[2]])
    
    # Third Derivative
    R3=utils.recov(st_Tim2)
    F3=2*np.matmul(R2,F1)+np.matmul(R3,F_1d)+np.matmul(R1,F2)
    st_Tim3=np.matmul(Comp_srt,F3)    
    gamma3=np.array([st_Tim3[1],st_Tim3[2]])
    
    # Using Warping Function data (displacements)
        
    Q=np.array([(0,0),(0,0),(0,-1),(1,0)])
    
    st_m=st_m+np.matmul(Q,gamma1)
    st_cl1=st_cl1+np.matmul(Q,gamma2)
    st_cl2=st_cl2+np.matmul(Q,gamma3)

    # Using Warping Function data (displacements)
    a1 = np.matmul(V0, st_m)  # disp_fluctuations
    a2 = np.matmul(V1, st_cl1)  # disp_fluctuation from srt
    a3 = np.matmul(V1, st_cl2)
    a4 = np.matmul(V0, st_cl1)

    w_1 = dolfinx.fem.Function(V)  # disp_fluctuations
    w1s_1 = dolfinx.fem.Function(V)  # disp_fluctuation from srt
    w1s_2 = dolfinx.fem.Function(V)
    w_2 = dolfinx.fem.Function(V)

    for i in range(len(V0[:, 0])):
        w_1.x.array[i], w1s_1.x.array[i], w1s_2.x.array[i], w_2.x.array[i] = (
            a1[i],
            a2[i],
            a3[i],
            a4[i],
        )

    # 3D strain recovery
    st_Eb = utils.gamma_h(dx, w_1, dim=3) + ufl.dot(
        utils.gamma_e(x), ufl.as_vector((st_m))
    )  
    st_Timo = (
        utils.gamma_h(dx, w1s_1, dim=3) + utils.gamma_l(w_2) + utils.gamma_l(w1s_2)
    )  
    st_3D_b = st_Eb + st_Timo   # Beam Reference Frame

    strain_mm,strain_bb=Function(VV), Function(VV)

    fexpr1 = dolfinx.fem.Expression(
        st_3D_b, VV.element.interpolation_points(), comm=MPI.COMM_WORLD
    )
    strain_bb.interpolate(fexpr1)
    
    st_3D_m=(utils.Rsig(meshdata["frame"]).T)*st_3D_b # Material Reference Frame
    fexpr1=dolfinx.fem.Expression(
        st_3D_m,VV.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    strain_mm.interpolate(fexpr1) 

    # Local Displacement
    rf=[beam_disp[int(segment)][k][1] for k in range(6)]

   # for k in range(6):
   #     print(beam_disp[int(segment)][k][0])          
    FF=np.array((rf[2],-rf[1],rf[0],rf[5],-rf[4],rf[3])) # BeamDyn --> VABS convention 
   # print('VABS Displacement:',FF)
    
    
    u_global=ufl.as_vector([FF[i] for i in range(3)])
    th1,th2,th3=[FF[i] for i in range(3,6)]
    
    cs_coord=ufl.as_vector([0,x[1],x[2]])
    
    frame_g=ufl.as_tensor([(1,    -th3,    th2),
                      (th3,   1,     -th1), 
                     (-th2,  th1,      1)]) # transpose(C_Bb) is used in frame
    
    u_local=u_global +frame_g*(w_1+w1s_1+cs_coord)-cs_coord
    u_loc=Function(V)
    fexpr1=dolfinx.fem.Expression(
        u_local,V.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    u_loc.interpolate(fexpr1) 

    return strain_mm,strain_bb,u_local


def stress_eval(mat_param, meshdata, strain_mm, points):

    CC_ = utils.CC(mat_param)
    
    VV = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(6,)),
    )
    stress_mm=Function(VV)
    for idx,el in enumerate(meshdata["subdomains"].values):
        stress_mm.x.array[6*idx:6*idx+6]=np.matmul(CC_[el],strain_mm.x.array[6*idx:6*idx+6])
        
    stress_eval = utils.stress_output(mat_param, meshdata["mesh"], stress_mm, points)

    return stress_eval   
    
    
def eigen_solve(mat_param, meshdata, l_mesh, r_mesh, strain_mm, u_local):
    mesh = meshdata["mesh"]
    dx = ufl.Measure("dx")(
        domain=mesh, subdomain_data=meshdata["subdomains"]
    )

    CC = utils.CC(mat_param)
    nphases = len(mat_param)
    
    V = dolfinx.fem.functionspace(
        mesh, basix.ufl.element("CG", mesh.topology.cell_name(), 1, shape=(3,))
    )
    VV = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(6,)),
    )
    stress_mm=Function(VV)
    for idx,el in enumerate(meshdata["subdomains"].values):
        stress_mm.x.array[6*idx:6*idx+6]=np.matmul(CC[el],strain_mm.x.array[6*idx:6*idx+6])
        
    du, u_ = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_loc=Function(V)
    fexpr1=dolfinx.fem.Expression(
        u_local,V.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    u_loc.interpolate(fexpr1) 
  #  u_L = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, np.concatenate((r_mesh["entity_map"], l_mesh["entity_map"]), axis=0)
    )
    
    # Linear Elasticity Bilinear Form
    a = sum(
        [
            ufl.dot
            (
                ufl.dot(
                    ufl.as_tensor(CC[i]),utils.eigen_eps(du)
                       ),
                    utils.eigen_eps(u_)
            ) * dx(i)
            for i in range(nphases)
        ]
    )
   
    bcs = [dolfinx.fem.dirichletbc(u_loc, boundary_dofs)]

    # Stiffness matrix
    K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a), bcs=bcs)
    K.assemble()
    
    st_KG= ufl.as_tensor([(stress_mm[0],stress_mm[5],stress_mm[4]),
                      (stress_mm[5],stress_mm[1],stress_mm[3]),
                      (stress_mm[4],stress_mm[3],stress_mm[2])]) 
    kgform = -ufl.inner(
                st_KG,
                ufl.grad(du).T * ufl.grad(u_)
            )* dx
        
    KG = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(kgform), bcs=bcs, diagonal=0
    )
    
    KG.assemble()  # epsilon(du) and grad(du) both are same --> (3,3) tensor
    
    eigensolver = utils.solve_GEP_shiftinvert(
        KG,                      # Optimization Problem: Find maximum eigenvalue of this problem
        K,
        problem_type=SLEPc.EPS.ProblemType.GHEP,
        solver=SLEPc.EPS.Type.KRYLOVSCHUR,
        nev=2,
        tol=1e-8,
        #  shift=1,
    )
    
    # Extract eigenpairs
    (eigval, eigvec_r, eigvec_i) = utils.EPS_get_spectrum(eigensolver, V)
    print("Critical Eigen val:", 1 / (np.max(eigval)).real)

  #  pyvista.start_xvfb()
    
 #   pyvista.set_jupyter_backend("static")
    
    # Grid for the mesh
  #  tdim = mesh.topology.dim
  #  mesh_topology, mesh_cell_types, mesh_geometry = dolfinx.plot.vtk_mesh(mesh, tdim)
   # mesh_grid = pyvista.UnstructuredGrid(mesh_topology, mesh_cell_types, mesh_geometry)
    
    # Grid for functions (2nd order elements)
   # u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
  #  u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    
    # Plot the first 3 eigenmodes
   # pl = pyvista.Plotter(shape=(1, 1))
    
   # for i in range(1):
      #  pl.subplot(1 , 0)
      #  pl = pyvista.Plotter(off_screen=True)
       # eigenmode = f"eigenmode_{i:02}"
       # pl.add_text(
      #     f"Eigenmode {i+1}",
        #   font_size=12,
        #   )
      #  eigen_vector = eigvec_r[i]
      #  u_grid[eigenmode] = eigen_vector.x.array.reshape(
     #   u_geometry.shape[0], V.dofmap.index_map_bs
     ##    )
     #   pl.add_mesh(mesh_grid, style="wireframe")
     #   pl.add_mesh(u_grid.warp_by_vector(eigenmode, factor=10), show_scalar_bar=False)
     #   pl.add_text(f"Eigenmode {i+1}", font_size=12)
     #   pl.view_isometric()
        
        # Save the plot as a PNG
      #  pl.screenshot(f"eigenmode_{i+1:02}.png")
      #  pl.show()
        # Close the plotter to free memory
      #  pl.close()
    return 1 / (np.max(eigval)).real

def local_strain2D(timo, rf, segment, meshdata):
    """Recover local strain field from beam analysis results.

    This function recovers the local 3D strain field within a blade segment
    using the fluctuating functions from the homogenized beam analysis and
    the applied beam forces.

    Parameters
    ----------
    timo : list
        Timoshenko analysis results containing [Deff_srt, V0, V1] where:
        - Deff_srt: effective stiffness matrix
        - V0: boundary fluctuating functions
        - V1: volume fluctuating functions
    beam_force : list
        Beam force components [Fx, Fy, Fz, Mx, My, Mz] applied to the segment
    segment : int
        Segment index for identification
    meshdata : dict
        Dictionary containing mesh information including the 3D mesh object

    Returns
    -------
    dolfinx.fem.Function
        Function containing the recovered 3D strain field in the segment
    """
    Deff_srt, V0, V1 = timo
 #   beam_force,beam_disp=beam_out
    V = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(3,)),
    )
    VV = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("DG", meshdata["mesh"].topology.cell_name(), 1, shape=(6,)),
    )

    x, dx = (
        ufl.SpatialCoordinate(meshdata["mesh"]),
        ufl.Measure("dx")(
            domain=meshdata["mesh"], subdomain_data=meshdata["subdomains"]
        ),
    )

  #  print("Beam reaction Force")
  #  for k in range(6):
     #   print(beam_force[int(segment)][k][0], "     ", beam_force[int(segment)][k][1])
    FF = np.array(
        (rf[0], rf[1], rf[2], rf[3], rf[4], rf[5])
    )  # BeamDyn --> VABS convention
    print('VABS Reaction Force:',FF)

    Comp_srt = np.linalg.inv(Deff_srt)
    st=np.matmul(Comp_srt,FF) 
    st_m=np.array((st[0],st[3],st[4],st[5]), dtype=np.float64)  
    
    # First Derivative
    F_1d=np.matmul(Deff_srt,st)
    R1=utils.recov(np.array((st[0]+1,st[1],st[2],st[3],st[4],st[5]), dtype=np.float64) )
    F1= np.matmul(R1,F_1d)
    st_Tim1=np.matmul(Comp_srt,F1)
    st_cl1=np.array([st_Tim1[0],st_Tim1[3],st_Tim1[4],st_Tim1[5]])
    gamma1=np.array([st_Tim1[1],st_Tim1[2]])
    
    # Second Derivative
    R2=utils.recov(st_Tim1)
    F2=np.matmul(R1,F1)+np.matmul(R2,F_1d)
    st_Tim2=np.matmul(Comp_srt,F2)    
    st_cl2=np.array([st_Tim2[0],st_Tim2[3],st_Tim2[4],st_Tim2[5]])
    gamma2=np.array([st_Tim2[1],st_Tim2[2]])
    
    # Third Derivative
    R3=utils.recov(st_Tim2)
    F3=2*np.matmul(R2,F1)+np.matmul(R3,F_1d)+np.matmul(R1,F2)
    st_Tim3=np.matmul(Comp_srt,F3)    
    gamma3=np.array([st_Tim3[1],st_Tim3[2]])
    
    # Using Warping Function data (displacements)
        
    Q=np.array([(0,0),(0,0),(0,-1),(1,0)])
    
    st_m=st_m+np.matmul(Q,gamma1)
    st_cl1=st_cl1+np.matmul(Q,gamma2)
    st_cl2=st_cl2+np.matmul(Q,gamma3)

    # Using Warping Function data (displacements)
    a1 = np.matmul(V0, st_m)  # disp_fluctuations
    a2 = np.matmul(V1, st_cl1)  # disp_fluctuation from srt
    a3 = np.matmul(V1, st_cl2)
    a4 = np.matmul(V0, st_cl1)

    w_1 = dolfinx.fem.Function(V)  # disp_fluctuations
    w1s_1 = dolfinx.fem.Function(V)  # disp_fluctuation from srt
    w1s_2 = dolfinx.fem.Function(V)
    w_2 = dolfinx.fem.Function(V)

    for i in range(len(V0[:, 0])):
        w_1.x.array[i], w1s_1.x.array[i], w1s_2.x.array[i], w_2.x.array[i] = (
            a1[i],
            a2[i],
            a3[i],
            a4[i],
        )

    # 3D strain recovery
    st_Eb = utils.gamma_h(dx, w_1, dim=3) + ufl.dot(
        utils.gamma_e(x), ufl.as_vector((st_m))
    )  
    st_Timo = (
        utils.gamma_h(dx, w1s_1, dim=3) + utils.gamma_l(w_2) + utils.gamma_l(w1s_2)
    )  
    st_3D_b = st_Eb + st_Timo   # Beam Reference Frame

    strain_mm,strain_bb=Function(VV), Function(VV)

    fexpr1 = dolfinx.fem.Expression(
        st_3D_b, VV.element.interpolation_points(), comm=MPI.COMM_WORLD
    )
    strain_bb.interpolate(fexpr1)
    
    st_3D_m=(utils.Rsig(meshdata["frame"]).T)*st_3D_b # Material Reference Frame
    fexpr1=dolfinx.fem.Expression(
        st_3D_m,VV.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    strain_mm.interpolate(fexpr1) 
 

    return strain_mm,strain_bb, None

def stressrecov2D(mat_param, meshdata, strain_mm,points):

    CC_ = utils.CC(mat_param)
    
    VV = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(6,)),
    )
    
    stress_mm=Function(VV)
    for idx,el in enumerate(meshdata["subdomains"].values):
        stress_mm.x.array[6*idx:6*idx+6]=np.matmul(CC_[el],strain_mm.x.array[6*idx:6*idx+6])
  
    stress_eval = utils.stress_output(mat_param, meshdata["mesh"], stress_mm, points)

    return stress_eval 
   # else:
      #  return stress_mm