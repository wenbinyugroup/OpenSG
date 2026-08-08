def mesh_to_periodic_sparse_assembly_map(
    V: int,
    cells: jnp.ndarray,
    points: jnp.ndarray,
    n_model: int,
    ndof_per_node: int = 3,
    atol=1e-6,
):
    dof_map_np = np.array(periodic_map(points, n_model, atol))
    master_nodes = dof_map_np[:][::ndof_per_node] // ndof_per_node
    master_nodes = master_nodes.astype(np.uint64) 

    # --- THE NEW COMPRESSION STEP ---
    # 1. Find the strictly unique master nodes (This is exactly your reduced set)
    unique_masters = np.unique(master_nodes)
    
    # 2. Create a translation array: Full Node ID -> Reduced Node ID
    full_to_reduced = np.full(V, -1, dtype=np.int32)
    full_to_reduced[unique_masters] = np.arange(len(unique_masters), dtype=np.int32)
    
    # 3. Map the original cells: Slave -> Master -> Reduced
    # master_nodes[cells] gets the master ID. 
    # full_to_reduced[...] converts it to the 0-indexed reduced space.
    reduced_periodic_cells = full_to_reduced[master_nodes[cells]]
    # --------------------------------

    return jnp.array(reduced_periodic_cells, dtype=jnp.int32), dof_map_np
        
def periodic_map(points, n_model, atol=1e-6, ndof_per_node=3):
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0) 
    num_nodes = len(points)
    dof_map = np.arange(num_nodes)
    n_sg=points.shape[1]
    if n_sg == 1 and n_model==3:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]

        # Calculate Box Dimensions for 1D
        Lx = max_xyz[0] - min_xyz[0]
        
        def map_boundary(slaves, masters, shift_vec):
            if len(slaves) == 0: return # Handle empty sets if boundary is empty
            
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            
            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            
            if not np.all(min_dists < atol):
                raise ValueError("Geometric mismatch on periodic boundary")
                
            # Update the global map
            dof_map[slaves] = masters[nearest_idx]

        # 1. Map Right -> Left (Shift x by Lx)
        map_boundary(right_points, left_points, np.array([Lx]))
        
    elif n_sg==2:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-6).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-6).nonzero()[0]

        # Calculate Box Dimensions
        Lx = max_xyz[0] - min_xyz[0]
        Ly = max_xyz[1] - min_xyz[1]
        def map_boundary(slaves, masters, shift_vec):
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            
            if not np.all(min_dists < atol):
                raise ValueError("Geometric mismatch on periodic boundary")
                
            # Update the global map
            dof_map[slaves] = masters[nearest_idx]
        if n_model==3: 
            # 1. Map Right -> Left (Shift x by -Lx)
            map_boundary(right_points, left_points, np.array([Lx, 0.0]))
            # 2. Map Top -> Bottom (Shift y by -Ly)
            map_boundary(top_points, bottom_points, np.array([0.0, Ly]))
        elif n_model==2:
            # 2. Map Top -> Bottom (Shift y by -Ly)
            map_boundary(right_points, left_points, np.array([Lx, 0.0]))
            
        #  while True:
        #      new_map = dof_map[dof_map]
        #      if np.array_equal(new_map, dof_map):
        #          break # Everything is fully resolved!
        #      dof_map = new_map    
        for _ in range(n_sg):
            dof_map = dof_map[dof_map]
    elif n_sg==3:
        left_points = np.isclose(points[:, 0], min_xyz[0], atol=1e-6).nonzero()[0]
        right_points = np.isclose(points[:, 0], max_xyz[0], atol=1e-6).nonzero()[0]
        bottom_points = np.isclose(points[:, 1], min_xyz[1], atol=1e-6).nonzero()[0]
        top_points = np.isclose(points[:, 1], max_xyz[1], atol=1e-6).nonzero()[0]
        back_points = np.isclose(points[:, 2], min_xyz[2], atol=1e-6).nonzero()[0]
        front_points = np.isclose(points[:, 2], max_xyz[2], atol=1e-6).nonzero()[0]
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

            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            mask = min_dists >= atol
            if np.any(mask):
                num_fail = np.sum(mask)
                max_err = np.max(min_dists)
                print(f"FAILED: {num_fail} nodes unmatched. Max distance: {max_err}")
                # Print the first few coordinates that failed to help you find them in your mesher
                print(f"Sample failed target pos: {target_pos[mask][0]}")
                raise ValueError(f"Geometric mismatch on periodic boundary with shift {shift_vec}")
            if not np.all(min_dists < atol):
                raise ValueError(f"Geometric mismatch on periodic boundary with shift {shift_vec}")
                
            dof_map[slaves] = masters[nearest_idx]
            
        if n_model==3:
             # 1. Map Right -> Left (Shift X)
             map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
             # 2. Map Top -> Bottom (Shift Y)
             map_boundary(top_points, bottom_points, np.array([0.0, Ly, 0.0]))
             # 3. Map Front -> Back (Shift Z)
             map_boundary(front_points, back_points, np.array([0.0, 0.0, Lz]))
          
        elif n_model==2:
             # 2. Map Right -> Left (Shift X)
             map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
             # 3. Map Top -> Bottom (Shift Y)
             map_boundary(top_points, bottom_points, np.array([0.0, Ly, 0.0]))
            
        else:
             # 2. Map Right -> Left (Shift X)
             map_boundary(right_points, left_points, np.array([Lx, 0.0, 0.0]))
             
        for _ in range(n_sg):
            dof_map = dof_map[dof_map]
    
    node_periodic_map= jnp.array(dof_map)
    master_nodes = node_periodic_map  
    dof_offsets = jnp.arange(ndof_per_node)
    master_dof_indices = master_nodes[:, None] * ndof_per_node + dof_offsets[None, :]    
    return master_dof_indices.flatten()