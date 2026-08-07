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
n_model=3 # 1: Beam; 2: Plate; 3: 3D elastic

name = os.environ.get('CASE', 'thin_iso')   # reads <name>.sc (cross lattice cell)
CELL_AREA = float(os.environ.get('CELL_AREA', '1.0'))   # L^2, the normalizer

# ISOTROPIC wall, SI (Pa): E1 E2 E3 G12 G13 G23 v12 v13 v23  (E=70 GPa, nu=0.3)
material_param=jnp.array([(70.0e9, 70.0e9, 70.0e9,
                           70.0e9/2.6, 70.0e9/2.6, 70.0e9/2.6,
                           0.30, 0.30, 0.30)])

angles = jnp.array([0.0]) # one angle per material domain; the .sc cannot carry
                          # per-element frames, so ONLY the isotropic cases can
                          # be verified through this route (see README.md)
#####################################################

n_sg=generate_msh_from_sc(name+'.sc', 'sgmesh.msh')

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
q_degree = 6 if b_degree > 1 else 3

# For line 5 , change basis degree=4
fe_type = FiniteElementType(
    cell_type=cell_type_map.get(meshio_type),
    family=ElementFamily.P,
    basis_degree=b_degree,
    lagrange_variant=LagrangeVariant.equispaced,
    quadrature_type=QuadratureType.default,
    quadrature_degree=q_degree,
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


# In[20]:


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

#########################################################################
@partial(jax.jit, static_argnames=['n_unique', 'n_model', 'n_sg'])
def full_homogenization_pipeline(
    x_end, u_0_g, dphi_dxi_qnp, phi_qn, W_q, 
    periodic_cells, unique_dofs, n_unique, n_model, n_sg
):
    C_ess = get_heterogeneous_C_matrix(cell_domain_ids, num_quad_points=Q, 
    material_param=material_param, domain_angles=angles
    )

    Dhe, D_bar, omega = calculate_residual_batch_element_kernel_mixed_periodic(
        x_end, dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells, 
        unique_dofs, u_0_g_full[unique_dofs], n_model, n_sg
    )

    J_uu = _calculate_jacobian_batch_element_kernel_periodic(
        x_end, u_0_g_full[unique_dofs], dphi_dxi_qnp, phi_qn, W_q, C_ess, periodic_cells
    )
     
    # 3. Solver Prep
    #t3 = time.time()
    inv_blocks = compute_block_inv_diag(J_uu, periodic_cells, n_unique)
   # print(f"  Preconditioner time: {time.time()-start:.4f}s")
 
    def solve_inner(b_col):
        # A_op expects ONE argument: the vector to multiply
        A_op = jax.tree_util.Partial(
            ebe_jacobian_product_periodic, 
            J_uu, periodic_cells, n_unique
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
        res, _ = jax.scipy.sparse.linalg.cg(A_op, b_col,M=cheb_M, tol=1e-4)
        return res

    V0_matrix = jax.lax.map(solve_inner, Dhe.T).T
    D1 = jnp.einsum('ni,nj->ij', V0_matrix, -Dhe)

    # C_eff, D1, D_bar, omega= compute_effective_properties(
    #     V0_matrix, -Dhe, omega,  D_bar, n_model, n_sg
    # )
    C_eff = (D_bar + D1) / omega
    # 5. Invert for Compliance (jnp.linalg.inv is fine for 6x6)
    #  I = jnp.eye(6, dtype=C_eff.dtype)
    # Com = jnp.linalg.solve(C_eff, I)
    
    # 6. Extract Engineering Constants on the GPU
    #  E1, E2, E3 = 1/Com[0,0], 1/Com[1,1], 1/Com[2,2]
    #  v12, v13, v23 = -Com[0,1]/Com[0,0], -Com[0,2]/Com[0,0], -Com[1,2]/Com[1,1]
    #  G23, G13, G12 = 1/Com[3,3], 1/Com[4,4], 1/Com[5,5]
    print(f"  GPU processing starts ...: {time.time()-start:.4f}s")
    return C_eff

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

periodic_cells, dof_map_np = mesh_to_periodic_sparse_assembly_map(V, cells, points, n_model, atol=1e-6)
unique_dofs = jnp.unique(dof_map_np)
n_unique=len(unique_dofs)
u_0_g_full = jnp.zeros(shape=(V * U)) 
C_eff= full_homogenization_pipeline(
    x_end, u_0_g_full, dphi_dxi_qnp, phi_qn, W_q,  
    periodic_cells, unique_dofs, n_unique, n_model, n_sg
)    
########################################################################
#labels = ["E1", "E2", "E3", "G12", "G13", "G23", "v12", "v13", "v23"];
#print("--- Effective Material Properties ---")
#for label, val in zip(labels, props):
#    print(f"{label}: {val}")

print('\n Effective Stiffness matrix \n')
print(C_eff)
print(f" \n Total Time taken: {time.time()-start:.4f}s")  


############### output ######################################
# The driver normalizes by the SG measure omega = the meshed WALL MATERIAL area.
# This study reports every route per CELL area L^2, so rescale by omega/L^2.
# omega is recomputed here from the mesh (the solver body is untouched).
import numpy as _np
_p = _np.asarray(mesh.points, float)[:, :2]
_t = _np.asarray(mesh.cells[0].data, int)
_a = 0.5*_np.abs((_p[_t[:, 1], 0]-_p[_t[:, 0], 0])*(_p[_t[:, 2], 1]-_p[_t[:, 0], 1])
                 - (_p[_t[:, 2], 0]-_p[_t[:, 0], 0])*(_p[_t[:, 1], 1]-_p[_t[:, 0], 1]))
_omega = float(_a.sum())
_C = _np.asarray(C_eff)
_C = 0.5*(_C + _C.T)*_omega/CELL_AREA
_S = _np.linalg.inv(_C)
_rows = [("E1", 1/_S[0, 0]), ("E2", 1/_S[1, 1]), ("E3", 1/_S[2, 2]),
         ("G23", 1/_S[3, 3]), ("G13", 1/_S[4, 4]), ("G12", 1/_S[5, 5]),
         ("nu12", -_S[0, 1]/_S[0, 0]), ("nu13", -_S[0, 2]/_S[0, 0]),
         ("nu23", -_S[1, 2]/_S[1, 1])]
print("\nomega (meshed wall material area) = %.10f ; cell area = %.6f"
      % (_omega, CELL_AREA))
print("9 effective constants:")
for _k, _v in _rows:
    print("  %-5s %14.6e" % (_k, _v))
_out = "%s_solidold.out" % name
with open(_out, "w") as _f:
    _f.write("# ROUTE C -- msg_solid OLD architecture, the single-script"
             " JAX_BICGoptimize driver (n_model=3)\n")
    _f.write("# source: %s.sc (cross lattice cell, L=1.0, ISOTROPIC wall"
             " E=70 GPa nu=0.3)\n" % name)
    _f.write("# a .sc carries ONE material id per element and this driver"
             " applies ONE fibre angle\n")
    _f.write("# per material, so per-element material FRAMES cannot be"
             " represented -- this route\n")
    _f.write("# can verify the isotropic cases only.\n")
    _f.write("# order [e11 e22 e33 2e23 2e13 2e12]\n")
    _f.write("# omega (meshed wall material area) = %.10f\n" % _omega)
    _f.write("# normalized by the CELL area L^2 = %.8f (C_eff*omega/L^2)\n"
             % CELL_AREA)
    _f.write("# solve %.2f s\n" % (time.time()-start))
    _f.write("# ---- C_eff (6x6, Pa) ----\n")
    for _i in range(6):
        _f.write(" ".join("%16.8e" % _C[_i, _j] for _j in range(6)) + "\n")
    _f.write("# ---- 9 effective constants ----\n")
    for _k, _v in _rows:
        _f.write("%-5s %16.8e\n" % (_k, _v))
print("wrote %s" % _out)
