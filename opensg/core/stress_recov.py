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
from dolfinx import fem, default_scalar_type
#from petsc4py import PETSc
#import pyvista

def local_strain(timo, beam_out, segment, meshdata, mat_param, FF=None):
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
    
    Deff_srt,V0, V1 = timo[0],timo[1],timo[2]
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
    if FF is None:
      FF = np.array(
          (rf[2], -rf[1], rf[0], rf[5], -rf[4], rf[3])
      )  # BeamDyn --> VABS convention
   # FF = np.array(
   #     (32230.4005595904, -7663.907852209771, 251712.81004955297, -55608.54410550957, -4170203.8641732424, -123224.93244239496)
   #     )
   # FF = np.array(( 0, 0, 100, 0, 0, 0))
    Comp_srt = np.linalg.inv(Deff_srt)
    st=np.matmul(Comp_srt,FF) 
    st_m=np.array((st[0],st[3],st[4],st[5]), dtype=np.float64) 
   # st_m=np.array([0.01,0,0,0]) 
    # Improved way to print for engineering clarity
    labels = ['F1 (Axial)', 'F2 (Shear Y)', 'F3 (Shear Z)', 'M1 (Twist)', 'M2 (Bend Y)', 'M3 (Bend Z)']
    print("--- Beam Resultant Forces (VABS Convention) ---")
    for label, value in zip(labels, FF):
        print(f"{label:12}: {value:>12.4e}")
    
    # First Derivative
    F_1d=np.matmul(Deff_srt,st)
    st_linear = np.zeros(6, dtype=np.float64)
    st_linear[0] = 1.0
    R1 = utils.recov(st_linear)
  #  R1=utils.recov(np.array((st[0]+1,st[1],st[2],st[3],st[4],st[5]), dtype=np.float64) )
    F1= np.matmul(R1,F_1d)
    st_Tim1=np.matmul(Comp_srt,F1)
    st_cl1=np.array([st_Tim1[0],st_Tim1[3],st_Tim1[4],st_Tim1[5]])

    gamma1=np.array([st_Tim1[1],st_Tim1[2]])
    
    
    # Second Derivative
  #  R2=utils.recov(st_Tim1)
    F2=np.matmul(R1,F1) #+np.matmul(R2,F_1d)
    st_Tim2=np.matmul(Comp_srt,F2)    
    st_cl2=np.array([st_Tim2[0],st_Tim2[3],st_Tim2[4],st_Tim2[5]])
    gamma2=np.array([st_Tim2[1],st_Tim2[2]])
    
    
    # Third Derivative
    #R3=utils.recov(st_Tim2)
    F3=np.matmul(R1,F2)
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
    st_3D_m=ufl.dot((utils.Rsig(meshdata["frame"]).T),st_3D_b) # Material Reference Frame

   # th1,th2,th3=0,0,0
   # cs_coord=ufl.as_vector([0,x[1],x[2]])
    
   # frame_g=ufl.as_tensor([(1,    -th3,    th2),
  #                    (th3,   1,     -th1), 
  #                   (-th2,  th1,      1)]) # transpose(C_Bb) is used in frame
    
   # u_local=frame_g*(w_1+w1s_1+cs_coord)-cs_coord
    u_local=w_1+w1s_1
    u_loc=Function(V)
    fexpr1=dolfinx.fem.Expression(
        u_local,V.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    u_loc.interpolate(fexpr1)
    mesh=meshdata["mesh"]
    # strain_m at guass points
    CC_ = utils.CC(mat_param)
    q_el = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(6,), degree=2, scheme="default")
    Q = dolfinx.fem.functionspace(mesh, q_el)
    strain_quad=Function(Q)
    fexpr1=dolfinx.fem.Expression(
        st_3D_m,Q.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    strain_quad.interpolate(fexpr1) 
    coord_quad=Q.tabulate_dof_coordinates()

    V_stiff=dolfinx.fem.functionspace(mesh, basix.ufl.element(
            "DG", mesh.topology.cell_name(), 0, shape = (6,6 )))    

    stiff=Function(V_stiff)
    for i,subb in enumerate(meshdata["subdomains"].values):
        stiff.x.array[36*i:36*i+36]=CC_[subb].flatten()  
    stress_mm = ufl.dot(stiff, st_3D_m)

    stress_quad = dolfinx.fem.Function(Q)
    u_expr = dolfinx.fem.Expression(stress_mm, Q.element.interpolation_points())
    stress_quad.interpolate(u_expr)

    return st_3D_m,u_loc, strain_quad, stress_quad, coord_quad


    
     
def eigen_solve(mat_param, meshdata, l_mesh, r_mesh, st_3D_m, u_local):
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
    strain_mm=Function(VV)
    fexpr1=dolfinx.fem.Expression(
        st_3D_m,VV.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    strain_mm.interpolate(fexpr1)
    stress_mm=Function(VV)
    for idx,el in enumerate(meshdata["subdomains"].values):
        stress_mm.x.array[6*idx:6*idx+6]=np.matmul(CC[el],strain_mm.x.array[6*idx:6*idx+6])
        
    du, u_ = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_loc=Function(V)
    fexpr1=dolfinx.fem.Expression(
        u_local,V.element.interpolation_points(), comm=MPI.COMM_WORLD
        )
    u_loc.interpolate(fexpr1) 
    #u_L = np.array([0, 0, 0], dtype=dolfinx.default_scalar_type)
    #boundary_dofs = dolfinx.fem.locate_dofs_topological(
    #    V, mesh.topology.dim - 1, np.concatenate((r_mesh["entity_map"], l_mesh["entity_map"]), axis=0)
    #)
    #bcs = [dolfinx.fem.dirichletbc(u_loc, boundary_dofs)]

    # ==========================================
    # 1. LEFT BOUNDARY: Fully Fixed
    # ==========================================
    # Locate DOFs for the entire vector space V on the left facets
    dofs_left = fem.locate_dofs_topological(V, mesh.topology.dim - 1, l_mesh["entity_map"])

    # Option A: Fix to absolute zero [0, 0, 0]
    u_L = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
    bc_left = fem.dirichletbc(u_L, dofs_left, V)

    # Option B: If you intended to use your interpolated `u_loc` function as the fixed value:
    #bc_left = fem.dirichletbc(u_loc, dofs_left)

    # ==========================================
    # 2. RIGHT BOUNDARY: Only u2 (y) and u3 (z) fixed
    # ==========================================
    # Locate DOFs specifically for the sub-spaces (index 1 for u2, index 2 for u3)
    dofs_right_u2 = fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, r_mesh["entity_map"])
    dofs_right_u3 = fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, r_mesh["entity_map"])

    # Define a scalar zero for the individual components
    zero_scalar = default_scalar_type(0.0)

    # Create the boundary conditions for the specific sub-spaces
    bc_right_u2 = fem.dirichletbc(zero_scalar, dofs_right_u2, V.sub(1))
    bc_right_u3 = fem.dirichletbc(zero_scalar, dofs_right_u3, V.sub(2))

    # ==========================================
    # 3. MERGE BOUNDARY CONDITIONS
    # ==========================================
    # Combine all individual BCs into a single list to pass to your solver
    bcs = [bc_left, bc_right_u2, bc_right_u3]
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
   # KG = dolfinx.fem.petsc.assemble_matrix(
    #     dolfinx.fem.form(kgform), bcs=bcs
   #  )
    KG.assemble()  # epsilon(du) and grad(du) both are same --> (3,3) tensor
  #  bc_dofs = np.unique(boundary_dofs).astype(np.int32)
  #  if bc_dofs.size > 0:
   #     KG.zeroRowsColumns(bc_dofs, diag=0.0)
 #   KG.assemble()
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
  #  print("Eigen value:", 1 / (np.max(eigval)).real)

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


def stress_eval(mat_param, meshdata, strain_m):
    """Evaluate the material-frame 3D stress field from a recovered strain.

    ``strain_m`` is the (UFL-expression) material-frame strain returned first by
    :func:`local_strain`.  Returns the stress at quadrature points and at CG1
    nodes (with coordinates) for visualisation / XDMF export.
    """
    CC_ = utils.CC(mat_param)
    mesh = meshdata["mesh"]
    VV = dolfinx.fem.functionspace(
        meshdata["mesh"],
        basix.ufl.element("DG", meshdata["mesh"].topology.cell_name(), 0, shape=(6,)),
    )
    stress_mm, strain_mm = Function(VV), Function(VV)
    fexpr1 = dolfinx.fem.Expression(
        strain_m, VV.element.interpolation_points(), comm=MPI.COMM_WORLD
    )
    strain_mm.interpolate(fexpr1)
    for idx, el in enumerate(meshdata["subdomains"].values):
        stress_mm.x.array[6 * idx:6 * idx + 6] = np.matmul(
            CC_[el], strain_mm.x.array[6 * idx:6 * idx + 6])

    Q = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element("CG", meshdata["mesh"].topology.cell_name(), 1, shape=(6,)),
    )
    q_elem = dolfinx.fem.Function(Q)
    u_expr = dolfinx.fem.Expression(stress_mm, Q.element.interpolation_points())
    q_elem.interpolate(u_expr)
    coord_elem = Q.tabulate_dof_coordinates()

    q_el = basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(6,), degree=2, scheme="default")
    Q = dolfinx.fem.functionspace(mesh, q_el)
    q_quad = dolfinx.fem.Function(Q)
    u_expr = dolfinx.fem.Expression(stress_mm, Q.element.interpolation_points())
    q_quad.interpolate(u_expr)
    coord_quad = Q.tabulate_dof_coordinates()
    return q_quad, coord_quad, q_elem, coord_elem
