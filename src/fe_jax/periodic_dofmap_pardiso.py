import numpy as np 
import jax.numpy as jnp


def dof_map_full(points, ndof_per_node=3):
    dof_map_disp = periodic_map(points)
    lambda_indices = jnp.arange(len(dof_map_disp) , 3*points.shape[0]+3 ) 
    return jnp.concatenate([dof_map_disp, lambda_indices])

def periodic_map(points, tol=1e-8, ndof_per_node=3):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0) 
    num_nodes = len(points)
    dof_map = np.arange(num_nodes)
    p=points.shape[1]
    if p==2:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-8).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-8).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-8).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-8).nonzero()[0]

        # Calculate Box Dimensions
        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]
        def map_boundary(slaves, masters, shift_vec):
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            dists = np.linalg.norm(target_pos[:, None, :] - master_pts[None, :, :], axis=2)

            nearest_idx = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)
            
            if not np.all(min_dists < tol):
                raise ValueError("Geometric mismatch on periodic boundary")
                
            # Update the global map
            dof_map[slaves] = masters[nearest_idx]
    
        # 1. Map Right -> Left (Shift x by -Lx)
        map_boundary(right_points, left_points, np.array([Lx, 0.0]))
        
        # 2. Map Top -> Bottom (Shift y by -Ly)
        map_boundary(top_points, bottom_points, np.array([0.0, Ly]))

    elif p==3:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-8).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-8).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-8).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-8).nonzero()[0]
        back_points = np.isclose(points[:, 2], min_xyz[2], atol=1e-8).nonzero()[0]
        front_points = np.isclose(points[:, 2], max_xyz[2], atol=1e-8).nonzero()[0]
        num_nodes = len(points)
        dof_map = np.arange(num_nodes)
    
        # Calculate Box Dimensions for 3D
        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]
        Lz = max_xyz[2] - min_xyz[2]
        
        def map_boundary(slaves, masters, shift_vec):
            if len(slaves) == 0: return # Handle empty sets if boundary is empty
            
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec

            dists = np.linalg.norm(target_pos[:, None, :] - master_pts[None, :, :], axis=2)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)
            
            if not np.all(min_dists < tol):
                raise ValueError(f"Geometric mismatch on periodic boundary with shift {shift_vec}")
                
            dof_map[slaves] = masters[nearest_idx]
    
        # 1. Map Right -> Left (Shift X)
        map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
        
        # 2. Map Top -> Bottom (Shift Y)
        map_boundary(top_points, bottom_points, np.array([0.0, Ly, 0.0]))
        
        # 3. Map Front -> Back (Shift Z)
        map_boundary(front_points, back_points, np.array([0.0, 0.0, Lz]))

    for _ in range(p):
            dof_map = dof_map[dof_map]
        
    node_periodic_map= jnp.array(dof_map)
    master_nodes = node_periodic_map  
    dof_offsets = jnp.arange(ndof_per_node)
    master_dof_indices = master_nodes[:, None] * ndof_per_node + dof_offsets[None, :]    
    return master_dof_indices.flatten()

def apply_pbc_reduction(K_full, R_full, dof_map_full):
    """
    Condenses K (N, N) -> K_red (M, M)
    Condenses R (N,)   -> R_red (M,)
    Where M is the number of unique master DOFs.
    """
    
    # 1. Identify Unique Master DOFs
    unique_dofs = jnp.unique(dof_map_full)
    num_reduced = len(unique_dofs)
    
    # 2. Reduce Residual (R)

    R_accum = jnp.zeros_like(R_full).at[dof_map_full].add(R_full)
    R_reduced = R_accum[unique_dofs]
    
    # 3. Reduce Stiffness Matrix (K)
    # K_red = L.T @ K_full @ L
    # Logic: K_reduced[I, J] = Sum(K_full[i, j]) where map[i]=I and map[j]=J
    
    # Step A: Sum Columns (Condense j -> J)
    # We want to add column j into column map[j]
    K_col_reduced = jnp.zeros_like(K_full).at[:, dof_map_full].add(K_full)
    
    # Step B: Sum Rows (Condense i -> I)
    # We want to add row i into row map[i]
    # Note: We operate on K_col_reduced from Step A
    K_total_reduced_fullsize = jnp.zeros_like(K_col_reduced).at[dof_map_full, :].add(K_col_reduced)
    
    # Step C: Extract only the unique master rows/cols
    K_reduced = K_total_reduced_fullsize[jnp.ix_(unique_dofs, unique_dofs)]
    
    return K_reduced, R_reduced, unique_dofs