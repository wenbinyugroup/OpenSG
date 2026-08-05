import meshio
import numpy as np
from helper import *
import jax.numpy as jnp
import jax
import time
import jax.lax as lax
import os
np.set_printoptions(precision=5)
jax.clear_caches()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
np.set_printoptions(precision=5) 

#%load_ext line_profiler
jax.config.update('jax_default_matmul_precision', 'highest') 
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

cache_path = os.path.abspath("./jax_cache")
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
cache_path = os.path.abspath("./jax_cache")
if not os.path.exists(cache_path):
    os.makedirs(cache_path)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_path
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

start = time.time()
#print(f"  Start Compute: {start:.4f}s")
############### User Input #################################
n_model=2 # 1: Beam; 2: Plate; 3: 3D elastic

# Test: (3D SG)
#name="BCC_SW_2UC_45"
# Test: (2D SG)
#name='Plate_RHC_refined_45' 
name='Plate_RHC_SW_2UC_45'
#name='Solid_RHC_SW_2UC_45'
#name="CoreUC"
# Test: (1D SG)
#name='Plate_1D_SG_2UC_45'
n_sg, elem_rotation, mat_angles, mat_seq, material_param = generate_msh_from_sc(name+'.sc', 'sgmesh.msh')

material_param=jnp.array([(108e3,8e3,8e3,4e3,4e3,3e3,0.32,0.32,0.30),
                          (108e3,8e3,8e3,4e3,4e3,3e3,0.32,0.32,0.30),
                         (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])

angles = jnp.array([45, -45, 0.0]) # Put 0.0 if no angle used
#####################################################

model_map = {
    1: "Beam",
    2: "Plate",
    3: "3D elastic"
}
print('n_model:    ', model_map.get(n_model, 1))
mesh = meshio.read('sgmesh.msh') 
points = jnp.array(mesh.points, dtype=np.float32)[:,0:n_sg]
cells = jnp.array(mesh.cells[0].data, dtype=np.uint64) 

meshio_type = mesh.cells[0].type 


cell_type_map = {
    # 1D Elements (Intervals)
    "interval": CellType.interval,
    "line": CellType.interval,       # 2-node line
    "line3": CellType.interval,      # 3-node quadratic line
    "line4": CellType.interval,      # 4-node cubic line
    "line5": CellType.interval,      # 5-node quartic line 
    
    # 2D Elements
    "triangle": CellType.triangle,
    "quad": CellType.quadrilateral,
    
    # 3D Elements
    "tetra": CellType.tetrahedron,
    "hexahedron": CellType.hexahedron,
    "tetra10":CellType.tetrahedron
}
basis_degree_map = {
    "line3": 2,
    "line4": 3,
    "line5": 4,
    "tetra10": 2,
}
b_degree = basis_degree_map.get(meshio_type, 1)
q_degree = 6 if b_degree > 1 else 2


fe_type = FiniteElementType(
    cell_type=cell_type_map.get(meshio_type),
    family=ElementFamily.P,
    basis_degree=1,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=2,
)

def build_single_C_matrix(params):
    """Helper function to build a 6x6 C matrix from a 1D array of 9 properties."""
    E1, E2, E3, G12, G13, G23, v12, v13, v23 = params
    
    # Initialize empty 6x6 matrix
    S = jnp.zeros((6, 6))
    
    S = S.at[0, 0].set(1/E1)
    S = S.at[1, 1].set(1/E2)
    S = S.at[2, 2].set(1/E3)
    S = S.at[3, 3].set(1/G23)
    S = S.at[4, 4].set(1/G13)
    S = S.at[5, 5].set(1/G12)
    
    S = S.at[0, 1].set(-v12/E1).at[1, 0].set(-v12/E1)
    S = S.at[0, 2].set(-v13/E1).at[2, 0].set(-v13/E1)
    S = S.at[1, 2].set(-v23/E2).at[2, 1].set(-v23/E2)
    C= jnp.linalg.inv(S)
    return C

def rotate_C_matrix(C, t):
    """Rotates a 6x6 stiffness matrix C by angle t. Skips math if t=0."""

    def do_rotation(operand):
        C_mat, angle = operand
        th = jnp.deg2rad(angle)
        c, s = jnp.cos(th), jnp.sin(th)
        cs = c * s
        # theta ccw from from 123 direction (fiber/material)
        #Exam 3.1 of MSM book with theta ccw.
        R_Sig = jnp.array([
            [c**2, s**2, 0, 0, 0, 2*cs],
            [s**2, c**2, 0, 0, 0, -2*cs],
            [0,    0,    1, 0, 0, 0    ],
            [0,    0,    0, c, -s, 0    ],
            [0,    0,    0, s, c, 0   ],
            [-cs,  cs,   0, 0, 0, c**2 - s**2]
        ])
        return R_Sig @ C_mat @ R_Sig.T

    def skip_rotation(operand):
        C_mat, _ = operand
        return C_mat

    # jax.lax.cond(condition, true_function, false_function, arguments)
    return jax.lax.cond(
        t == 0.0,         # Condition
        skip_rotation,    # Run this if True
        do_rotation,      # Run this if False
        (C, t)            # Pack the arguments to pass into the functions
    )

cell_domain_ids = jnp.array(mesh.cell_data["gmsh:physical"][0]-1)   
@partial(jax.jit, static_argnames=['num_quad_points'])
def get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points, material_param, domain_angles):

    C_matrices_base = jax.vmap(build_single_C_matrix)(material_param)
    
    C_matrices_rotated = jax.vmap(rotate_C_matrix)(C_matrices_base, domain_angles)
    
    return C_matrices_rotated[cell_domain_ids]

@jax.jit    
def _element_residual_single_case(
    u_nd: jnp.ndarray, x_nd: jnp.ndarray, dphi_dxi_qnp: jnp.ndarray, phi_qn: jnp.ndarray,
    W_q: jnp.ndarray, C_ss: jnp.ndarray, 
    epsilon_bar: jnp.ndarray = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
):
    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    G_qpd = jnp.linalg.inv(J_qdp)
    det_J_q = jnp.linalg.det(J_qdp)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    
    Q, N, _ = dphi_dx_qnd.shape
    zeros = jnp.zeros((Q, N), dtype=dphi_dx_qnd.dtype)
    if x_nd.shape[1]==1: # 1D SG
        dphi_dx_3D = jnp.stack([zeros, zeros, dphi_dx_qnd[..., 0]], axis=-1)
    elif x_nd.shape[1]==2: # 2D SG
        dphi_dx_3D = jnp.stack([zeros, dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1]], axis=-1)
    else:
        dphi_dx_3D = dphi_dx_qnd
    dx, dy, dz = dphi_dx_3D[..., 0], dphi_dx_3D[..., 1], dphi_dx_3D[..., 2]
    u_x, u_y, u_z = u_nd[:, 0], u_nd[:, 1], u_nd[:, 2]
    eps_xx, eps_yy, eps_zz = dx @ u_x, dy @ u_y, dz @ u_z
    eps_yz = (dy @ u_z) + (dz @ u_y)
    eps_xz = (dx @ u_z) + (dz @ u_x)
    eps_xy = (dx @ u_y) + (dy @ u_x)

    eps_qs = jnp.stack([eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy], axis=-1) + epsilon_bar
    stress = jnp.einsum("ij, qj, q, q -> qi", C_ss, eps_qs, det_J_q, W_q) 

    R_x = jnp.sum(stress[:, 0, None] * dx + stress[:, 5, None] * dy + stress[:, 4, None] * dz, axis=0)
    R_y = jnp.sum(stress[:, 5, None] * dx + stress[:, 1, None] * dy + stress[:, 3, None] * dz, axis=0)
    R_z = jnp.sum(stress[:, 4, None] * dx + stress[:, 3, None] * dy + stress[:, 2, None] * dz, axis=0)
    return jnp.stack([R_x, R_y, R_z], axis=-1)
    
@jax.jit(static_argnames=['n_model', 'n_sg'])
def calculate_residual_batch_element_kernel_mixed_periodic(
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    C_ess: jnp.ndarray,
    periodic_cells, 
    unique_dofs_full: jnp.ndarray, 
    u_g_flat: jnp.ndarray,  
    n_model: int, 
    n_sg: int
):
    # ==========================================
    # 1. Compute Omega (Macroscopic Scaling)
    # ==========================================
    if n_model == 3:
        def _get_vol(x_nd):
            J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
            return jnp.sum(jnp.linalg.det(J_qdp) * W_q)

        elem_vols = jax.vmap(_get_vol)(x_end)
        omega = jnp.sum(elem_vols)

    elif n_model == 2:
        if n_sg == 3:
            omega = jnp.ptp(x_end[..., 0]) * jnp.ptp(x_end[..., 1])
        elif n_sg == 2:
            omega = jnp.ptp(x_end[..., 0])
        else:
            omega = 1.0
            
    elif n_model == 1:
        if n_sg ==3:
            omega = jnp.ptp(x_end[..., 0])
        else:
            omega = 1.0
    else:
        omega = 1.0
    # ==========================================
    # 2. Pre-define Masks for Ge
    # ==========================================    
    if n_model == 1:
            mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
            mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
            mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
    elif n_model == 2:
            mask_ones_2d = jnp.zeros((6, 6)).at[0,0].set(1.0).at[1,1].set(1.0).at[5,2].set(1.0)
            mask_y3_2d   = jnp.zeros((6, 6)).at[0,3].set(1.0).at[1,4].set(1.0).at[5,5].set(1.0)

    u_end = transform_global_unraveled_to_element_node(periodic_cells, u_g_flat, U=3)
    # ==========================================
    # 3. Element Processing Logic
    # ==========================================
    def element_process_all_cases(u_nd, x_nd, C_ss):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        
        if n_sg > 1:  # Covers n_sg == 2 and n_sg == 3
            detJ_q = jnp.abs(jnp.linalg.det(J_qdp))
        else:         # Covers n_sg == 1
            detJ_q = jnp.abs(J_qdp[..., 0, 0])

        # Spatial coordinates at quadrature points
        dV_q = detJ_q * W_q
        x_q = jnp.dot(phi_qn, x_nd) 
        
        # 2. Construct Ge based on the macroscopic model
        if n_model == 3:
            Ge = jnp.eye(6)
            # Einsum DOES NOT have 'q' on Ge since Ge is constant
            D_elem = jnp.einsum('ji,jk,kl,q->il', Ge, C_ss, Ge, dV_q)
            
            # Map over axis 1 (columns of the 6x6 matrix)
            run_cases = jax.vmap( 
                _element_residual_single_case, 
                in_axes=(None, None, None, None, None, None, 1) 
            )
            R_nodes = run_cases(u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, Ge)
            
        elif n_model == 2:
            # 2D Plate/Shell
            y3_q = x_q[:, n_sg-1]
            Ge = mask_ones_2d[None, :, :] + mask_y3_2d[None, :, :] * y3_q[:, None, None]
            
            # Einsum HAS 'q' on Ge
            D_elem = jnp.einsum('qji,jk,qkl,q->il', Ge, C_ss, Ge, dV_q)
            
            # Map over axis 2 (cases dimension)
            run_cases = jax.vmap( 
                _element_residual_single_case, 
                in_axes=(None, None, None, None, None, None, 2) 
            )
            R_nodes = run_cases(u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, Ge)
            
        elif n_model == 1:
            # 1D Beam
            y3_q = x_q[:, n_sg-1]
            y2_q = jnp.zeros_like(x_q[:, 0]) if n_sg == 1 else x_q[:, n_sg-2]
            Ge = (
                mask_ones_1d[None, :, :] + 
                mask_y2[None, :, :] * y2_q[:, None, None] + 
                mask_y3_1d[None, :, :] * y3_q[:, None, None]
            )

            # Einsum HAS 'q' on Ge
            D_elem = jnp.einsum('qji,jk,qkl,q->il', Ge, C_ss, Ge, dV_q)
            
            # Map over axis 2 (cases dimension)
            run_cases = jax.vmap( 
                _element_residual_single_case, 
                in_axes=(None, None, None, None, None, None, 2) 
            )
            R_nodes = run_cases(u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, Ge)

        return R_nodes, D_elem
        
    # Outer VMAP: Iterate over the batch of elements
    batch_processor = jax.vmap(element_process_all_cases, in_axes=(0, 0, 0))
    
    # R_end_cases shape: (E, num_cases, N, 3) where num_cases is 6 or 4
    R_end_cases, D_elem_batch = batch_processor(u_end, x_end, C_ess) 
    D_bar = jnp.sum(D_elem_batch, axis=0)
    R_cases_first = jnp.transpose(R_end_cases, (1, 0, 2, 3))

    def assemble_one_case(R_one_case_EN3):  #
        return transform_element_node_to_global_unraveled_sum(#(GlobalNDOF_unraveled, 3) 
            periodic_cells=periodic_cells, 
            v_en=R_one_case_EN3,
            num_nodes=u_g_flat.shape[0]//3,
        )
    R_global_cases = jax.vmap(assemble_one_case)(R_cases_first)
    num_cases = num_cases = R_global_cases.shape[0]
    R_elastic_matrix = R_global_cases.reshape(num_cases, -1).T
    R_elastic_matrix = R_elastic_matrix.at[0:3, :].set(0.0)
    return -R_elastic_matrix, D_bar, omega


@jax.jit
def _calculate_jacobian_batch_element_kernel_periodic(
    x_end: jnp.ndarray, u_f, dphi_dxi_qnp: jnp.ndarray, phi_qn: jnp.ndarray, 
    W_q: jnp.ndarray, C_ess: jnp.ndarray, periodic_cells: object,
):
    E = x_end.shape[0]
    u_enu = transform_global_unraveled_to_element_node(periodic_cells, u_f)

    N, D, U = x_end.shape[1], x_end.shape[2], u_enu.shape[2]
    u_et = u_enu.reshape(E, N * U)
    
    @jax.jit
    def residual_kernel(u_t, x_nd, C_ss):
      u_nd = u_t.reshape(N, U)
      R_elastic = _element_residual_single_case(
          u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, 
          epsilon_bar=jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      )
      return R_elastic.ravel() 

    J_uu = jax.vmap(jax.jacfwd(residual_kernel, argnums=0))(
        u_et, x_end, C_ess
    )
    return J_uu

@jax.jit(static_argnames=['n_model', 'n_sg'])
def calculate_RHS_and_Ke_batch_periodic(
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    C_ess: jnp.ndarray,
    periodic_cells, 
    u_g_flat: jnp.ndarray,  
    n_model: int, 
    n_sg: int
):
    # ==========================================
    # 1. Pre-define Masks for Ge
    # ==========================================    
    if n_model == 1:
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
    elif n_model == 2:
        mask_ones_2d = jnp.zeros((6, 6)).at[0,0].set(1.0).at[1,1].set(1.0).at[5,2].set(1.0)
        mask_y3_2d   = jnp.zeros((6, 6)).at[0,3].set(1.0).at[1,4].set(1.0).at[5,5].set(1.0)

    u_end = transform_global_unraveled_to_element_node(periodic_cells, u_g_flat, U=3)
    N = x_end.shape[1]
    U = 3

    # ==========================================
    # 2. The Merged Element Processing Kernel
    # ==========================================
    def element_process_all_cases_and_Ke(u_nd, x_nd, C_ss):
        x_q = jnp.dot(phi_qn, x_nd) 
        
        # -----------------------------------------
        # PART A: Get Residuals for all Macro Cases
        # -----------------------------------------
        if n_model == 3:
            Ge = jnp.eye(6)
            run_cases = jax.vmap(_element_residual_single_case, in_axes=(None, None, None, None, None, None, 1))
        elif n_model == 2:
            y3_q = x_q[:, n_sg-1]
            Ge = mask_ones_2d[None, :, :] + mask_y3_2d[None, :, :] * y3_q[:, None, None]
            run_cases = jax.vmap(_element_residual_single_case, in_axes=(None, None, None, None, None, None, 2))
        elif n_model == 1:
            y3_q = x_q[:, n_sg-1]
            y2_q = jnp.zeros_like(x_q[:, 0]) if n_sg == 1 else x_q[:, n_sg-2]
            Ge = (mask_ones_1d[None, :, :] + mask_y2[None, :, :] * y2_q[:, None, None] + mask_y3_1d[None, :, :] * y3_q[:, None, None])
            run_cases = jax.vmap(_element_residual_single_case, in_axes=(None, None, None, None, None, None, 2))

        # Output shape: (num_cases, N, U)
        R_nodes_cases = run_cases(u_nd, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, Ge)
        
        # -----------------------------------------
        # PART B: Get Element Stiffness Matrix (K_e)
        # -----------------------------------------
        def single_case_for_jac(u_flat):
            # Reshape 1D vector back to 2D nodes for the physics function
            u_nd_t = u_flat.reshape(N, U)
            # Use ZERO macro strain to perfectly isolate K * u
            dummy_eps = jnp.zeros(6) 
            R_single = _element_residual_single_case(u_nd_t, x_nd, dphi_dxi_qnp, phi_qn, W_q, C_ss, dummy_eps)
            return R_single.ravel()

        # Extract local K_e matrix. Shape: (N*U, N*U)
        K_elem = jax.jacfwd(single_case_for_jac)(u_nd.ravel())

        return R_nodes_cases, K_elem
        
    # ==========================================
    # 3. Outer VMAP and Assembly
    # ==========================================
    # Iterate over the batch of elements exactly once!
    batch_processor = jax.vmap(element_process_all_cases_and_Ke, in_axes=(0, 0, 0))
    
    # Run the batch!
    R_end_cases, J_uu_batch = batch_processor(u_end, x_end, C_ess) 
    
    # Assemble the RHS load vectors
    R_cases_first = jnp.transpose(R_end_cases, (1, 0, 2, 3))
    def assemble_one_case(R_one_case_EN3):  
        return transform_element_node_to_global_unraveled_sum(
            periodic_cells=periodic_cells, 
            v_en=R_one_case_EN3,
            num_nodes=u_g_flat.shape[0]//3,
        )
        
    R_global_cases = jax.vmap(assemble_one_case)(R_cases_first)
    num_cases = R_global_cases.shape[0]
    
    # Finalize global RHS matrix. Shape: (Ndof, num_cases)
    rhs_matrix = R_global_cases.reshape(num_cases, -1).T
    rhs_matrix = rhs_matrix.at[0:3, :].set(0.0)
    
    # We return the RHS vectors AND the batch of K matrices!
    return rhs_matrix, J_uu_batch

def ebe_jacobian_product_periodic(
    J_uu, periodic_cells: jnp.ndarray, n_unique_u: int, z_u: jnp.ndarray
):
    z_u_enu = transform_global_unraveled_to_element_node(
        periodic_cells, z_u
    )
    _,N,U=z_u_enu.shape
    z_u_local = z_u_enu.reshape(-1, N*U)

    f_u_local_flat = jnp.einsum('eij,ej->ei', J_uu, z_u_local) 
    f_u_local = f_u_local_flat.reshape(-1, N, U)
                     
    f_u_global_full = transform_element_node_to_global_unraveled_sum(
        periodic_cells=periodic_cells, v_en=f_u_local, num_nodes=n_unique_u // U,
    )
    f_u_global_full = f_u_global_full.at[0:3].set(z_u[0:3])
    return f_u_global_full

# --- Define the Block Preconditioner Apply Function ---
@partial(jax.jit, static_argnames=["n_unique_u"])
def apply_block_precond(inv_blocks, n_unique_u, x_reduced):
    
    # Apply 3x3 block matrix multiplication to each node
    x_nodes = x_reduced.reshape(-1, 3)
    precond_nodes = jnp.einsum('nij,nj->ni', inv_blocks, x_nodes)
    
    return precond_nodes.ravel()
    
@partial(jax.jit, static_argnames=["n_unique"])
def compute_block_inv_diag(J_uu, periodic_cells, n_unique):
    E, n_u, _ = J_uu.shape
    N = n_u // 3
    
    J_uu_reshaped = J_uu.reshape(E, N, 3, N, 3)
    
    # Extract blocks where the node index matches. Shape becomes (E, 3, 3, N)
    local_3x3_blocks = jnp.diagonal(J_uu_reshaped, axis1=1, axis2=3) 
    
    # Rearrange to (E, N, 3, 3)
    local_3x3_blocks = jnp.moveaxis(local_3x3_blocks, -1, 1) 
    
    # 2. Scatter-add to form the global nodal 3x3 matrices
    num_unique = n_unique // 3
    global_blocks = jnp.zeros((num_unique, 3, 3))
    global_blocks = global_blocks.at[periodic_cells].add(local_3x3_blocks)
    
    # 3. Invert the 3x3 blocks directly on the GPU
    I_3x3 = jnp.eye(3)[None, :, :]
    global_blocks_safe = global_blocks + I_3x3 * 1e-8
    
    # jnp.linalg.inv automatically batches over the first dimension!
    inv_global_blocks = jnp.linalg.inv(global_blocks_safe) # Shape: (num_nodes, 3, 3)
    
    # 4. Calculate mean scalar stiffness for the Lagrange multipliers
   # mean_stiffness = jnp.mean(jnp.abs(global_blocks_safe))
    
    return inv_global_blocks

# 2. Power Method for Eigenvalue Estimation      
# 1. Remove @jax.jit here! lax.map already handles the compilation.
def estimate_max_eigenvalue(A_op, M_op, b_col, num_iters=15):
    """
    Uses the Power Method to find the maximum eigenvalue of M^-1 * A.
    """
    # Use ones_like to completely bypass shape tracing errors
    v = jnp.ones_like(b_col)
    v = v / jnp.linalg.norm(v)
    
    def power_step(i, state):
        v_current, current_lam = state
        
        w = M_op(A_op(v_current)) 
        lam_new = jnp.vdot(v_current, w)
        
        # Added a tiny epsilon (1e-12) to prevent division by zero just in case
        v_new = w / (jnp.linalg.norm(w) + 1e-12) 
        
        return (v_new, lam_new)
        
    _, lambda_max = jax.lax.fori_loop(0, num_iters, power_step, (v, 0.0))
    
    return lambda_max * 1.05

# 3. The Chebyshev Wrapper
def apply_chebyshev_precond(inv_blocks, n_unique_u, eig_max, eig_min, A_op, degree, x_reduced):
    """
    Notice x_reduced is the LAST argument. 
    BiCGSTAB will pass the current residual into this slot.
    """
    d = (eig_max + eig_min) / 2.0
    c = (eig_max - eig_min) / 2.0
    
    i = jnp.arange(1, degree + 1)
    theta = d + c * jnp.cos(jnp.pi * (2 * i - 1) / (2 * degree))
    
    def step(z, theta_i):
        # A_op(z) expects a vector
        r = x_reduced - A_op(z) 
        
        # Here we manually pass 'r' into the x_reduced slot of your base preconditioner
        z_update = apply_block_precond(inv_blocks, n_unique_u, r)
        
        z_new = z + z_update / theta_i
        return z_new, None

    z0 = jnp.zeros_like(x_reduced)
    z_final, _ = jax.lax.scan(step, z0, theta)
    return z_final

# 4. Compute D1
def compute_effective_properties(
    delta_u_matrix: jnp.ndarray,
    R_f_reduced: jnp.ndarray,
    D_bar: jnp.ndarray, 
    omega,
    n_model: int,    
    n_sg: int,     
):
    D1 = jnp.einsum('ni,nj->ij', delta_u_matrix, R_f_reduced)

    # 4. Effective Stiffness
    C_eff = (D_bar + D1) / omega

    # 5. Invert for Compliance (jnp.linalg.inv is fine for 6x6)
  #  I = jnp.eye(6, dtype=C_eff.dtype)
   # Com = jnp.linalg.solve(C_eff, I)
  
    # 6. Extract Engineering Constants on the GPU
  #  E1, E2, E3 = 1/Com[0,0], 1/Com[1,1], 1/Com[2,2]
  #  v12, v13, v23 = -Com[0,1]/Com[0,0], -Com[0,2]/Com[0,0], -Com[1,2]/Com[1,1]
  #  G23, G13, G12 = 1/Com[3,3], 1/Com[4,4], 1/Com[5,5]

    return C_eff, D1, D_bar, omega #, jnp.array([E1, E2, E3, G12, G13, G23,v12, v13, v23]), D1
@jax.jit(static_argnames=['n_model', 'n_sg'])
def compute_homogenized_constants(
    x_end: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray,
    phi_qn: jnp.ndarray,          
    W_q: jnp.ndarray,
    C_ess: jnp.ndarray,
    n_model: int, 
    n_sg: int
):
    # ==========================================
    # 1. Compute Omega (Macroscopic Scaling)
    # ==========================================
    if n_model == 3:
        def _get_vol(x_nd):
            J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
            return jnp.sum(jnp.linalg.det(J_qdp) * W_q)
        elem_vols = jax.vmap(_get_vol)(x_end)
        omega = jnp.sum(elem_vols)
    elif n_model == 2:
        if n_sg == 3:
            omega = jnp.ptp(x_end[..., 0]) * jnp.ptp(x_end[..., 1])
        elif n_sg == 2:
            omega = jnp.ptp(x_end[..., 0])
        else:
            omega = 1.0
    elif n_model == 1:
        if n_sg == 3:
            omega = jnp.ptp(x_end[..., 0])
        else:
            omega = 1.0
    else:
        omega = 1.0

    # ==========================================
    # 2. Pre-define Masks for Ge
    # ==========================================    
    if n_model == 1:
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
    elif n_model == 2:
        mask_ones_2d = jnp.zeros((6, 6)).at[0,0].set(1.0).at[1,1].set(1.0).at[5,2].set(1.0)
        mask_y3_2d   = jnp.zeros((6, 6)).at[0,3].set(1.0).at[1,4].set(1.0).at[5,5].set(1.0)

    # ==========================================
    # 3. Compute D_bar (Element processing)
    # ==========================================
    def compute_D_elem(x_nd, C_ss):
        J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
        
        if n_sg > 1: 
            detJ_q = jnp.abs(jnp.linalg.det(J_qdp))
        else:        
            detJ_q = jnp.abs(J_qdp[..., 0, 0])

        dV_q = detJ_q * W_q
        x_q = jnp.dot(phi_qn, x_nd) 
        
        if n_model == 3:
            Ge = jnp.eye(6)
            D_elem = jnp.einsum('ji,jk,kl,q->il', Ge, C_ss, Ge, dV_q)
        elif n_model == 2:
            y3_q = x_q[:, n_sg-1]
            Ge = mask_ones_2d[None, :, :] + mask_y3_2d[None, :, :] * y3_q[:, None, None]
            D_elem = jnp.einsum('qji,jk,qkl,q->il', Ge, C_ss, Ge, dV_q)
        elif n_model == 1:
            y3_q = x_q[:, n_sg-1]
            y2_q = jnp.zeros_like(x_q[:, 0]) if n_sg == 1 else x_q[:, n_sg-2]
            Ge = (
                mask_ones_1d[None, :, :] + 
                mask_y2[None, :, :] * y2_q[:, None, None] + 
                mask_y3_1d[None, :, :] * y3_q[:, None, None]
            )
            D_elem = jnp.einsum('qji,jk,qkl,q->il', Ge, C_ss, Ge, dV_q)

        return D_elem
        
    # VMAP across all elements and sum
    D_elem_batch = jax.vmap(compute_D_elem)(x_end, C_ess) 
    D_bar = jnp.sum(D_elem_batch, axis=0)

    return D_bar, omega
#########################################################################
@partial(jax.jit, static_argnames=['n_unique', 'n_model', 'n_sg'])
def full_homogenization_pipeline(
    x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, 
    periodic_cells, unique_dofs, n_unique, n_model, n_sg
):
    C_ess = get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points=Q, 
    material_param=material_param, domain_angles=angles
    )

    #Dhe, D_bar, omega = calculate_residual_batch_element_kernel_mixed_periodic(
    #    x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells, 
   #     unique_dofs, u_0_g_full[unique_dofs], n_model, n_sg
   # )

   # J_uu = _calculate_jacobian_batch_element_kernel_periodic(
    #    x_end, u_0_g_full[unique_dofs], dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells
   # )
    Dhe, J_euu = calculate_RHS_and_Ke_batch_periodic(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells, 
        u_0_g[unique_dofs], n_model, n_sg
    ) 
    # 3. Solver Prep
    #t3 = time.time()
    inv_blocks = compute_block_inv_diag(J_euu, periodic_cells, n_unique)
   # print(f"  Preconditioner time: {time.time()-start:.4f}s")
 
    def solve_inner(b_col):
        # A_op expects ONE argument: the vector to multiply
        A_op = jax.tree_util.Partial(
            ebe_jacobian_product_periodic, 
            J_euu, periodic_cells, n_unique
        )
        
        # M_op freezes the first 3 args, waiting for 'x_reduced'
        M_op = jax.tree_util.Partial(
            apply_block_precond, 
            inv_blocks, n_unique
        )
        # Estimate the max eigenvalue dynamically using our M_op and A_op
        estimated_eig_max = estimate_max_eigenvalue(A_op, M_op, b_col, num_iters=15)
        estimated_eig_min = estimated_eig_max / 25.0 
        
        # Freeze all arguments EXCEPT x_reduced
        cheb_M = jax.tree_util.Partial(
            apply_chebyshev_precond,
            inv_blocks,     # arg 
            n_unique,       # arg 
            estimated_eig_max, # arg 
            estimated_eig_min, # arg
            A_op,           # arg 
            4           
        )
        res, _ = jax.scipy.sparse.linalg.cg(A_op, b_col,M=cheb_M, tol=1e-6)
        return res

    V0_matrix = jax.lax.map(solve_inner, -Dhe.T).T
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, Dhe)

    D_bar, omega=compute_homogenized_constants(x_end,
    dphi_dxi_qnp, phi_qn, W_q, C_ess, n_model, n_sg)

    C_eff = (D_bar + D1) / omega
    # 5. Invert for Compliance (jnp.linalg.inv is fine for 6x6)
    #  I = jnp.eye(6, dtype=C_eff.dtype)
    # Com = jnp.linalg.solve(C_eff, I)
    
    # 6. Extract Engineering Constants on the GPU
    #  E1, E2, E3 = 1/Com[0,0], 1/Com[1,1], 1/Com[2,2]
    #  v12, v13, v23 = -Com[0,1]/Com[0,0], -Com[0,2]/Com[0,0], -Com[1,2]/Com[1,1]
    #  G23, G13, G12 = 1/Com[3,3], 1/Com[4,4], 1/Com[5,5]
    print(f"  GPU processing starts ...: {time.time()-start:.4f}s")
    return C_eff, V0_matrix, C_ess

##########################################################################

xi_qp, W_q = get_quadrature(fe_type=fe_type)
phi_qn, dphi_dxi_qnp = eval_basis_and_derivatives(fe_type=fe_type, xi_qp=xi_qp)
Q = xi_qp.shape[0] 
V=points.shape[0]
U=3 # num of solution components
x_end = mesh_to_jax(vertices=points, cells=cells)

E = x_end.shape[0] # num elements
#print(f"  Quadrature i/p: {time.time()-start:.4f}s")    

# Periodic assembly map input

periodic_cells_en, dof_map_np = mesh_to_periodic_sparse_assembly_map(V, cells, points, n_model, atol=1e-6)
unique_dofs = jnp.unique(dof_map_np)
n_unique=len(unique_dofs)

u_0_g_full = jnp.zeros(shape=(V * U)) 
C_eff, V0_matrix, C_ess= full_homogenization_pipeline(
    x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q,  
    periodic_cells_en, unique_dofs, n_unique, n_model, n_sg
)    
print(C_eff)
print('Dehomogenization begins at time:', time.time()-start, 'sec \n')
# ==========================================
# 1. Core Element Dehomogenization Kernel
# ==========================================

def _element_dehomo_kernel(
    V0_ndH: jnp.ndarray,     # Shape: (N, 3, 6) - Element fluctuations for 6 cases
    x_nd: jnp.ndarray,      # Shape: (N, 3) - Element coordinates
    C_ss: jnp.ndarray,      # Shape: (6, 6) - Material stiffness for this element
    dphi_dxi_qnp: jnp.ndarray, 
    phi_qn: jnp.ndarray,
    epsilon_bar_H: jnp.ndarray, # Shape: (6,) - Macroscopic applied strain
    n_model: int, 
    n_sg: int
):
    # --- A. Kinematics ---
    J_qdp = jnp.einsum("nd,qnp->qdp", x_nd, dphi_dxi_qnp)
    print('dphi_dxi_qnp',dphi_dxi_qnp.shape)
    G_qpd = jnp.linalg.inv(J_qdp)
    dphi_dx_qnd = jnp.einsum("qpd,qnp->qnd", G_qpd, dphi_dxi_qnp)
    
    Q, N, _ = dphi_dx_qnd.shape
    zeros_qn = jnp.zeros((Q, N), dtype=dphi_dx_qnd.dtype)
    
    if x_nd.shape[1] == 1: 
        dphi_dx_qnd = jnp.stack([zeros_qn, zeros_qn, dphi_dx_qnd[..., 0]], axis=-1)
    elif x_nd.shape[1] == 2: 
        dphi_dx_qnd = jnp.stack([zeros_qn, dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1]], axis=-1)
  #  else:
  #      dphi_dx_qnd = dphi_dx_qnd
        
    dx_qn, dy_qn, dz_qn = dphi_dx_qnd[..., 0], dphi_dx_qnd[..., 1], dphi_dx_qnd[..., 2]

    # --- B. Fluctuation Strains (Gamma_h * S * V0) ---

    V0_x_nH, V0_y_nH, V0_z_nH = V0_ndH[:, 0, :], V0_ndH[:, 1, :], V0_ndH[:, 2, :] #(N,n_Hom)

    eps_xx_fluct_qH = dx_qn @ V0_x_nH
    eps_yy_fluct_qH = dy_qn @ V0_y_nH
    eps_zz_fluct_qH = dz_qn @ V0_z_nH
    eps_yz_fluct_qH = (dy_qn @ V0_z_nH) + (dz_qn @ V0_y_nH)
    eps_xz_fluct_qH = (dx_qn @ V0_z_nH) + (dz_qn @ V0_x_nH)
    eps_xy_fluct_qH = (dx_qn @ V0_y_nH) + (dy_qn @ V0_x_nH)

    # Shape: (Q, 6, 6)
    Gamma_h_S_V0_qsH = jnp.stack([
        eps_xx_fluct_qH, eps_yy_fluct_qH, eps_zz_fluct_qH, 
        eps_yz_fluct_qH, eps_xz_fluct_qH, eps_xy_fluct_qH
    ], axis=1)

    # --- C. Macroscopic Strain Mapping (Ge / Gamma_epsilon) ---
    x_qd = jnp.dot(phi_qn, x_nd) 

    if n_model == 3:
        Ge_sH = jnp.eye(6)
        Strain_Concentration_qsH = Gamma_h_S_V0_qsH  + Ge_sH[None, :, :]
        
    elif n_model == 2:
        mask_ones_2d = jnp.zeros((6, 6)).at[0,0].set(1.0).at[1,1].set(1.0).at[5,2].set(1.0)
        mask_y3_2d   = jnp.zeros((6, 6)).at[0,3].set(1.0).at[1,4].set(1.0).at[5,5].set(1.0)
        y3_q = x_qd[:, n_sg-1]
        Ge_sH = mask_ones_2d[None, :, :] + mask_y3_2d[None, :, :] * y3_q[:, None, None]
        Strain_Concentration_qsH = Gamma_h_S_V0_qsH + Ge_sH
        
    else: # n_model == 1
        mask_ones_1d = jnp.zeros((6, 4)).at[0, 0].set(1.0)
        mask_y2 = jnp.zeros((6, 4)).at[0, 3].set(-1.0).at[4, 1].set(1.0)
        mask_y3_1d = jnp.zeros((6, 4)).at[0, 2].set(1.0).at[5, 1].set(-1.0)
        y3_q = x_q[:, n_sg-1]
        y2_q = jnp.zeros_like(x_q[:, 0]) if n_sg == 1 else x_q[:, n_sg-2]
        Ge_sH = mask_ones_1d[None, :, :] + mask_y2[None, :, :] * y2_q[:, None, None] + mask_y3_1d[None, :, :] * y3_q[:, None, None]
        Strain_Concentration_qsH = Gamma_h_S_V0_qsH + Ge_sH
    
    # 1. Local Strains (Shape: Q, 6)
    Gamma_qs = jnp.einsum("qij,j->qi", Strain_Concentration_qsH, epsilon_bar_H)
    
    # 2. Local Stresses (Shape: Q, 6)
    # Note: If your C_ss is defined per quadrature point (Q, 6, 6), change this to "qij, qj -> qi"
    Sigma_qs = jnp.einsum("ij, qj -> qi", C_ss, Gamma_qs) 
    
    return Gamma_qs, Sigma_qs

# ==========================================
# 2. Global Batched Execution Wrapper
# ==========================================
@partial(jax.jit, static_argnames=['n_model', 'n_sg'])
def run_dehomogenization_pipeline(
    V0_matrix_gH: jnp.ndarray,     # Shape: (ndof, 6) - Output from your CG solver
    epsilon_bar_H: jnp.ndarray,   # Shape: (6,) - The applied macro strain
    x_end: jnp.ndarray,         # Shape: (num_elems, N, 3)
    C_ess: jnp.ndarray,         # Shape: (num_elems, 6, 6) - Material properties per element
    periodic_cells_en: jnp.ndarray,
    dphi_dxi_qnp: jnp.ndarray, 
    phi_qn: jnp.ndarray, 
    n_model: int, 
    n_sg: int
):
    # --- A. Map Global DOFs to Element DOFs ---
    # We vmap over the 6 columns (axis=1) of V0_matrix, stacking the result on the last axis
    get_element_V0 = jax.vmap(
        transform_global_unraveled_to_element_node, 
        in_axes=(None, 1, None), # 1 makes V-_matrix_gH to slice based on each H data
        out_axes=-1 # stack these 6 H arrays at the very end
    )
    # Resulting shape: (num_elems, N, 3, 6)
    V0_endH = get_element_V0(periodic_cells_en, V0_matrix_gH, 3)

    # --- B. Batch the Element Kernel ---
    # We map over elements (axis 0). dphi_dxi_qnp and phi_qn are same for all elements (None)
    batched_dehomo_calc = jax.vmap(
        _element_dehomo_kernel,
        in_axes=(0, 0, 0, None, None, None, None, None) 
    )

    # --- C. Execute Global Calculation ---
    # Gamma_global and Sigma_global will both have shape: (num_elems, Q, 6)
    Gamma_global_eqs, Sigma_global_eqs = batched_dehomo_calc(
        V0_endH, x_end, C_ess, dphi_dxi_qnp, phi_qn, epsilon_bar_H, n_model, n_sg
    )
    
    return Gamma_global_eqs, Sigma_global_eqs

# 1. Define your macroscopic strain state (e.g., pure tension in X)
macro_strain_H = jnp.array([0, 0.1, 0, 0.1, 0, 0])

# 2. Run the pipeline
local_strains_eqs, local_stresses_eqs = run_dehomogenization_pipeline(
    V0_matrix_gH=V0_matrix, 
    epsilon_bar_H=macro_strain_H, 
    x_end=x_end, 
    C_ess=C_ess, 
    periodic_cells_en=periodic_cells_en, 
    dphi_dxi_qnp=dphi_dxi_qnp, 
    phi_qn=phi_qn, 
    n_model=n_model, 
    n_sg=n_sg
)


def export_full_gauss_data(
    local_strains_eqs, 
    local_stresses_eqs, 
    node_coords_end, 
    phi_qn, 
    filename_prefix="gauss_results"
):
    # ==========================================
    # Step 1: Calculate Global Gauss Coordinates
    # ==========================================
    gauss_coords_eqd = jnp.einsum('qn, end -> eqd', phi_qn, node_coords_end)
    
    strains_flat_g = np.asarray(local_strains_eqs).reshape(-1, 6)
    stresses_flat_g = np.asarray(local_stresses_eqs).reshape(-1, 6)
    
    d = node_coords_end.shape[2]
    coords_flat_d = np.asarray(gauss_coords_eqd).reshape(-1, d)
    num_pts = strains_flat_g.shape[0]
    
    # ==========================================
    # Step 2: TXT File Export (Keeps 6 Components)
    # ==========================================
    gp_indices = np.arange(1, num_pts + 1).reshape(-1, 1)
    combined_data = np.hstack((gp_indices, coords_flat_d, strains_flat_g, stresses_flat_g))
    
    coord_headers = "Coord_X\tCoord_Y\tCoord_Z\t" if d == 3 else "Coord_X\tCoord_Y\t"
    header = (
        "GP_Index\t" + coord_headers +
        "Eps_xx\tEps_yy\tEps_zz\tEps_yz\tEps_xz\tEps_xy\t"
        "Sig_xx\tSig_yy\tSig_zz\tSig_yz\tSig_xz\tSig_xy"
    )
    
    total_floats = d + 12
    fmt_string = '\t'.join(['%d'] + ['%.6e'] * total_floats)
    
    txt_filename = f"{filename_prefix}.txt"
    np.savetxt(txt_filename, combined_data, fmt=fmt_string, header=header, comments='', delimiter='\t')
    
    # ==========================================
    # Step 3: VTK File Export (ParaView)
    # ==========================================
    if d == 2:
        coords_3d = np.hstack((coords_flat_d, np.zeros((num_pts, 1))))
    else:
        coords_3d = coords_flat_d
    coords_flat = np.asarray(gauss_coords_eqd).reshape(-1, d)
# Names for the 6 Voigt components in your specific order
    # Order: [0:xx, 1:yy, 2:zz, 3:yz, 4:xz, 5:xy]
    voigt_names = ["xx", "yy", "zz", "yz", "xz", "xy"]

    vtk_filename = f"{filename_prefix}.vtk"
    with open(vtk_filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Gauss Point Voigt Data\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Write Coordinates (Padded to 3D for VTK compatibility)
        f.write(f"POINTS {num_pts} float\n")
        coords_3d = np.hstack((coords_flat, np.zeros((num_pts, 1)))) if d == 2 else coords_flat
        np.savetxt(f, coords_3d, fmt='%.6e')
        
        # Write Connectivity (Every point is a Vertex)
        f.write(f"\nCELLS {num_pts} {num_pts * 2}\n")
        cell_data = np.column_stack((np.ones(num_pts, dtype=int), np.arange(num_pts, dtype=int)))
        np.savetxt(f, cell_data, fmt='%d')
        
        f.write(f"\nCELL_TYPES {num_pts}\n")
        np.savetxt(f, np.ones(num_pts, dtype=int), fmt='%d') # Type 1 = VTK_VERTEX
        
        # Write Point Data Header
        f.write(f"\nPOINT_DATA {num_pts}\n")

        # 2. Write 6 Strain Components
        for i, name in enumerate(voigt_names):
            f.write(f"SCALARS Eps_{name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, strains_flat_g[:, i], fmt='%.6e')

        # 3. Write 6 Stress Components
        for i, name in enumerate(voigt_names):
            f.write(f"SCALARS Sig_{name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            np.savetxt(f, stresses_flat_g[:, i], fmt='%.6e')

    print(f"Exported {num_pts} points to {vtk_filename}.")
    print(f" -> Saved Text Data: {txt_filename}")


# Now run the export!
#export_full_gauss_data(
#    local_strains_eqs,
#    local_stresses_eqs,
 #   x_end,
 #   phi_qn
#)

#####################################

import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def plot_stress_comparison(local_stresses_eqs, x_end, phi_qn, rhc_filename="RHC_Solid_refined_45.sc.sg"):
    # --- 1. Calculate and Flatten JAX Gauss Points ---
    # Shape: (q,n) * (e,n,d) -> (e,q,d)
    gauss_coords_qed = jnp.einsum('qn, end -> qed', phi_qn, x_end)
    gauss_coords_eqd = jnp.transpose(gauss_coords_qed, (1, 0, 2))
    
    jax_coords = np.asarray(gauss_coords_eqd.reshape(-1, 2))
    jax_sig_xx = np.asarray(local_stresses_eqs[:, :, 0]).flatten()

    # --- 2. Load RHC Data ---
    try:
        rhc_data = np.loadtxt(rhc_filename)
    except FileNotFoundError:
        print(f"Error: {rhc_filename} not found in path.")
        return

    rhc_y1, rhc_y2 = rhc_data[:, 0], rhc_data[:, 1]
    rhc_val_xx = rhc_data[:, 8]

    # --- 3. Robust Path Selection ---
    # Instead of hardcoding 0, find the median y2 value in the RHC file
    # This ensures we always find a "slice" of data.
    unique_rhc_y2 = np.unique(rhc_y2)
    target_y2 = unique_rhc_y2[len(unique_rhc_y2) // 2] # Pick the middle row
    
    print(f"RHC y2 range: {rhc_y2.min():.4f} to {rhc_y2.max():.4f}")
    print(f"Selecting path at y2 = {target_y2:.6f}")

    tol = 1e-3
    mask_rhc = np.abs(rhc_y2 - target_y2) < tol
    filtered_rhc_coords = np.column_stack((rhc_y1[mask_rhc], rhc_y2[mask_rhc]))
    filtered_rhc_stress = rhc_val_xx[mask_rhc]

    if len(filtered_rhc_coords) == 0:
        print("Still no points found. Try increasing 'tol' or check RHC format.")
        return

    # --- 4. Nearest Neighbor Match ---
    tree = KDTree(jax_coords)
    distances, jax_indices = tree.query(filtered_rhc_coords)
    matched_jax_stress = jax_sig_xx[jax_indices]

    # --- 5. Sort and Plot ---
    sort_idx = np.argsort(filtered_rhc_coords[:, 0])
    x_plot = filtered_rhc_coords[sort_idx, 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, filtered_rhc_stress[sort_idx], 'k--', label='RHC Reference', linewidth=2)
    plt.plot(x_plot, matched_jax_stress[sort_idx], 'r-', label='JAX Computation', alpha=0.8)

    plt.xlabel('Coordinate $y_1$')
    plt.ylabel('Stress $\sigma_{xx}$')
    plt.title(f'Stress Comparison along path $y_2 = {target_y2:.4f}$')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_png = "sigx_comparison_final.png"
    plt.savefig(output_png, dpi=300)
    plt.show()
    
    # Display in Colab

# Call with your JAX variables
#plot_stress_comparison(local_stresses_eqs, x_end, phi_qn)

def export_all_stress_comparisons(local_stresses_eqs, x_end, phi_qn, rhc_filename=name+".sc.sg"):
    # --- 1. Geometry Prep (JAX) ---
    # (q,n) * (e,n,d) -> (e,q,d)
    gauss_coords_qed = jnp.einsum('qn, end -> qed', phi_qn, x_end)
    gauss_coords_eqd = jnp.transpose(gauss_coords_qed, (1, 0, 2))
    jax_coords = np.asarray(gauss_coords_eqd.reshape(-1, 2))
    
    # --- 2. Load Reference Data (RHC) ---
    rhc_data = np.loadtxt(rhc_filename)
    rhc_y1, rhc_y2 = rhc_data[:, 0], rhc_data[:, 1]
    
    # --- 3. Robust Path Selection (Median y2) ---
    target_y1 = 8.9
    
    print(f"RHC y1 range: {rhc_y1.min():.4f} to {rhc_y1.max():.4f}")
    print(f"Generating 6 plots along path y1 = {target_y1:.6f}")

    mask_rhc = np.abs(rhc_y1 - target_y1) < 1e-2
    path_coords_rhc = np.column_stack((rhc_y1[mask_rhc], rhc_y2[mask_rhc]))
    sort_idx = np.argsort(path_coords_rhc[:, 1])
    x_plot = path_coords_rhc[sort_idx, 1]

    # --- 4. Spatial Matching (KDTree) ---
    # We only need to build the tree and query once for the coordinates
    tree = KDTree(jax_coords)
    _, jax_indices = tree.query(path_coords_rhc)
    print(jax_indices[0:5])
    print(path_coords_rhc[0:5])
    
    # --- 4b. Gap Detection & Sorting ---
    # Sort the coordinates to find sequential gaps
    sort_idx = np.argsort(path_coords_rhc[:, 1])
    x_plot_sorted = path_coords_rhc[sort_idx, 1]
    
    # Identify indices where the gap between consecutive points is > 4.0
    gaps = np.diff(x_plot_sorted)
    gap_indices = np.where(gaps > 4.0)[0]

    # --- 5. Loop through all 6 Stress Components ---
    # Order: sig_xx, sig_yy, sig_zz, sig_yz, sig_xz, sig_xy
    labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    
    for i, label in enumerate(labels):
        # Extract JAX and RHC components
        jax_comp = np.asarray(local_stresses_eqs[:, :, i]).flatten()[jax_indices]
        rhc_comp = rhc_data[mask_rhc, 8 + i]
        
        # Apply the sorting index to the components
        jax_comp_sorted = jax_comp[sort_idx]
        rhc_comp_sorted = rhc_comp[sort_idx]

        # Inject np.nan between points where the gap > 4.0
        # Adding 1 to gap_indices places the NaN exactly between the disconnected points
        x_plot_nans = np.insert(x_plot_sorted.astype(float), gap_indices + 1, np.nan)
        jax_comp_nans = np.insert(jax_comp_sorted.astype(float), gap_indices + 1, np.nan)
        rhc_comp_nans = np.insert(rhc_comp_sorted.astype(float), gap_indices + 1, np.nan)

        # Create Plot
        plt.figure(figsize=(8, 5))
        
        # 1. Plot the continuous lines (broken by NaNs at the gaps)
        plt.plot(x_plot_nans, rhc_comp_nans, 'r--', label=f'SwiftComp', linewidth=1.5)
        plt.plot(x_plot_nans, jax_comp_nans, 'b-', label=f'OpenSG 2.0', alpha=0.7)

        # 2. Scatter plot the actual data points
       # plt.scatter(x_plot_sorted, rhc_comp_sorted, color='black', s=20, marker='o', label=f'SwiftComp $\sigma_{{{label}}}$ Data', zorder=3)
        #plt.scatter(x_plot_sorted, jax_comp_sorted, color='red', s=20, marker='x', label=f'OpenSG-2.0 $\sigma_{{{label}}}$ Data', zorder=3)

        plt.xlabel('Coordinate $y_3$')
        plt.ylabel(f'Stress $\sigma_{{{label}}}$')
        plt.title(f'Comparison: $\sigma_{{{label}}}$ at $y_2 \\approx {target_y1:.4f}$')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Save unique file
        save_name = f"stress_comp_{label}.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close() # Close to save memory in Colab
        print(f" -> Saved: {save_name} (Broke line at {len(gap_indices)} gaps)")


import numpy as np
import jax.numpy as jnp
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os

def export_all_stress_comparisons(local_stresses_eqs, x_end, phi_qn, rhc_filename=name+".sc.sg", out_coords="matched_top_path.coords"):
    # --- 1. Geometry Prep (JAX) ---
    gauss_coords_qed = jnp.einsum('qn, end -> qed', phi_qn, x_end)
    gauss_coords_eqd = jnp.transpose(gauss_coords_qed, (1, 0, 2))
    jax_coords = np.asarray(gauss_coords_eqd.reshape(-1, 2))
    
    # --- 2. Load Reference Data (RHC) ---
    rhc_data = np.loadtxt(rhc_filename)
    rhc_y1, rhc_y2 = rhc_data[:, 0], rhc_data[:, 1] # rhc_y1 is y2, rhc_y2 is y3
    
    # --- 3. Top Edge Path Selection ---
    # Find the absolute maximum y3 coordinate in the mesh
    max_y3 = rhc_y2.max()
    print(f"Targeting top edge path: y3 ≈ {max_y3:.6f} (search tol=1e-4)")

    # Isolate all nodes that sit on this top edge
    top_edge_mask = np.abs(rhc_y2 - max_y3) <= 1e-4
    rhc_row_indices = np.where(top_edge_mask)[0] 
    
    path_coords_rhc = np.column_stack((rhc_y1[top_edge_mask], rhc_y2[top_edge_mask]))
    
    if len(path_coords_rhc) < 2:
        print(f"Error: Found {len(path_coords_rhc)} nodes on top edge. Need at least 2.")
        return
        
    print(f" -> Identified horizontal path with {len(path_coords_rhc)} nodes.")

    # --- 4. Spatial Matching (KDTree) & Distance Filtering ---
    tree = KDTree(jax_coords)
    distances, jax_indices = tree.query(path_coords_rhc)
    
    # STRICT distance tolerance (1e-5) ensuring OpenSG and SwiftComp meshes perfectly align
    dist_mask = distances <= 1e-5
    
    path_coords_rhc = path_coords_rhc[dist_mask]
    jax_indices = jax_indices[dist_mask]
    rhc_row_indices = rhc_row_indices[dist_mask]
    
    if len(path_coords_rhc) == 0:
        print("Error: KDTree matching failed. Meshes do not perfectly align within 1e-5 on top edge.")
        return
        
    print(f"Found {len(path_coords_rhc)} strictly matched points (tol=1e-5).")

    # =========================================================================
    # --- 4b. Extract, Sort (by y2), Print, and Save Coordinates ---
    # =========================================================================
    # CHANGED: We now sort by y2 (index 0) because the path is horizontal
    sort_idx = np.argsort(path_coords_rhc[:, 0]) 
    
    # SwiftComp matched and sorted coords
    rhc_y2_matched = path_coords_rhc[sort_idx, 0]
    rhc_y3_matched = path_coords_rhc[sort_idx, 1]
    
    # OpenSG matched and sorted coords
    jax_matched_coords = jax_coords[jax_indices][sort_idx]
    jax_y2_matched = jax_matched_coords[:, 0]
    jax_y3_matched = jax_matched_coords[:, 1]

    # 1. Print Coordinates to Console
    print("\n--- Matched Coordinates (y2, y3) ---")
    print(f"{'OpenSG':^22} | {'SwiftComp':^22}")
    print("-" * 47)
    for i in range(len(rhc_y3_matched)):
        print(f"({jax_y2_matched[i]:8.4f}, {jax_y3_matched[i]:8.4f}) | ({rhc_y2_matched[i]:8.4f}, {rhc_y3_matched[i]:8.4f})")
    print("-" * 47 + "\n")

    # 2. Write Coordinates to .coords file
    if os.path.dirname(out_coords):
        os.makedirs(os.path.dirname(out_coords), exist_ok=True)
        
    with open(out_coords, "w") as f:
        f.write("# OpenSG_y2 OpenSG_y3 SwiftComp_y2 SwiftComp_y3\n")
        for i in range(len(rhc_y3_matched)):
            f.write(f"{jax_y2_matched[i]:.6f} {jax_y3_matched[i]:.6f} {rhc_y2_matched[i]:.6f} {rhc_y3_matched[i]:.6f}\n")
    print(f" -> Coordinates successfully saved to '{out_coords}'\n")
    # =========================================================================

    # CHANGED: The baseline for plotting/gaps/non-dim is now y2
    x_plot_sorted = rhc_y2_matched 

    # --- 5. Loop through all 6 Stress Components ---
    labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    
    for i, label in enumerate(labels):
        jax_comp = np.asarray(local_stresses_eqs[:, :, i]).flatten()[jax_indices]
        rhc_comp = rhc_data[rhc_row_indices, 8 + i]
        
        jax_comp_sorted = jax_comp[sort_idx]
        rhc_comp_sorted = rhc_comp[sort_idx]

        # Consolidate overlapping points mathematically (now consolidating along y2)
        x_rounded = np.round(x_plot_sorted, decimals=4)
        unique_x = np.unique(x_rounded)
        
        single_x = []
        single_rhc = []
        single_jax = []
        
        for x_val in unique_x:
            idx = np.where(x_rounded == x_val)[0]
            if len(idx) > 0:
                single_x.append(x_val)
                single_rhc.append(np.max(rhc_comp_sorted[idx]))
                single_jax.append(np.max(jax_comp_sorted[idx]))

        single_x = np.array(single_x)
        single_rhc = np.array(single_rhc)
        single_jax = np.array(single_jax)

        # --- Filter out negative stress values ---
        pos_mask = (single_rhc >= 0) & (single_jax >= 0)
        pos_x = single_x[pos_mask]
        pos_rhc = single_rhc[pos_mask]
        pos_jax = single_jax[pos_mask]

        if len(pos_x) == 0:
            print(f" -> Skipping {label}: No positive stress values found.")
            continue

        # Gap Detection based on actual y2 spacing
        gaps = np.diff(pos_x)
        gap_indices = np.where(gaps > 4.0)[0]

        # --- CHANGED: Non-dimensionalize pos_x from 0 to 1 based on y2 bounds ---
        if len(pos_x) > 1:
            x_min = pos_x[0]
            x_max = pos_x[-1]
            if x_max > x_min:
                pos_x_nd = (pos_x - x_min) / (x_max - x_min)
            else:
                pos_x_nd = np.zeros_like(pos_x)
        else:
            pos_x_nd = pos_x

        # Insert NaNs to break the line at large spatial gaps
        plot_x_nd = np.insert(pos_x_nd.astype(float), gap_indices + 1, np.nan)
        plot_rhc = np.insert(pos_rhc.astype(float), gap_indices + 1, np.nan)
        plot_jax = np.insert(pos_jax.astype(float), gap_indices + 1, np.nan)

        # Create Plot
        plt.figure(figsize=(8, 5))
        
        # Line plots
        plt.plot(plot_x_nd, plot_rhc, 'r--', label='SwiftComp', linewidth=1.5)
        plt.plot(plot_x_nd, plot_jax, 'b-', label='OpenSG', alpha=0.7, linewidth=1.5)

        # CHANGED: Labels and titles updated to reflect the y2 non-dimensional parameter
        plt.xlabel('Non-dim parameter (y2)')
        plt.ylabel(f'$\sigma_{{{label}}}$')
        plt.title(f'Comparison: $\sigma_{{{label}}}$ (Top Edge, y3 = max)')
        
        plt.ylim(bottom=0) 
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        save_name = f"stress_comp_{label}_top_edge.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f" -> Saved: {save_name}")

export_all_stress_comparisons(local_stresses_eqs, x_end, phi_qn)

print('Dehomogenization completed at time:', time.time()-start, 'sec \n')