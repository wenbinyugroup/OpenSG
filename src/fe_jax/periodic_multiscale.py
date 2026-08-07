"""periodic_multiscale.py -- periodic local->global sparse assembly map.

Verbatim from the working single-file MSG driver (JAX_BICGoptimize_current.py +
fe_jax/periodic_multiscale.py): the map pairs opposite bounding-box faces and
re-points the element connectivity at the MASTER nodes, so periodicity is
carried by the assembly map itself -- every scatter/gather lands on the master
DOF and no constraint rows are needed.

Which faces get paired depends on BOTH the SG dimension (n_sg) and the macro
model (n_model = 1 beam / 2 plate / 3 solid):

    n_sg = 1, n_model = 3   pair x
    n_sg = 2, n_model = 3   pair x and y          (2-D SG -> 3-D solid)
    n_sg = 2, n_model = 2   pair x only           (y = through-thickness, free)
    n_sg = 3, n_model = 3   pair x, y and z
    n_sg = 3, n_model = 2   pair x and y          (z = thickness, free)
    n_sg = 3, otherwise     pair x only           (beam: axial only)

A free (unpaired) direction is what makes an isolated cross-section a FREE SG.
"""
import numpy as np
import jax.numpy as jnp
from scipy.spatial.distance import cdist


def mesh_to_periodic_sparse_assembly_map(
    V: int,
    cells,
    points,
    n_model: int,
    ndof_per_node: int = 3,
    atol=1e-6,
):
    dof_map_np = np.array(periodic_map(points, n_model, atol))
    master_nodes = dof_map_np[:][::ndof_per_node] // ndof_per_node
    master_nodes = master_nodes.astype(np.uint64)

    # --- THE COMPRESSION STEP ---
    # 1. Find the strictly unique master nodes (the reduced set)
    unique_masters = np.unique(master_nodes)

    # 2. Translation array: Full Node ID -> Reduced Node ID
    full_to_reduced = np.full(V, -1, dtype=np.int32)
    full_to_reduced[unique_masters] = np.arange(len(unique_masters),
                                                dtype=np.int32)

    # 3. Map the original cells: Slave -> Master -> Reduced
    reduced_periodic_cells = full_to_reduced[master_nodes[cells]]

    return jnp.array(reduced_periodic_cells, dtype=jnp.int32), dof_map_np


def periodic_map(points, n_model, atol=1e-6, ndof_per_node=3):
    points = np.asarray(points)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    num_nodes = len(points)
    dof_map = np.arange(num_nodes)
    n_sg = points.shape[1]

    def make_map_boundary(atol_):
        def map_boundary(slaves, masters, shift_vec):
            if len(slaves) == 0:
                return
            slave_pts = points[slaves]
            master_pts = points[masters]
            target_pos = slave_pts - shift_vec
            dists = cdist(target_pos, master_pts)
            nearest_idx = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(dists)), nearest_idx]
            if not np.all(min_dists < atol_):
                raise ValueError("Geometric mismatch on periodic boundary "
                                 "with shift %s (max %.2e)"
                                 % (shift_vec, float(np.max(min_dists))))
            dof_map[slaves] = masters[nearest_idx]
        return map_boundary

    map_boundary = make_map_boundary(atol)
    lo = [np.isclose(points[:, d], min_xyz[d], atol=1e-6).nonzero()[0]
          for d in range(n_sg)]
    hi = [np.isclose(points[:, d], max_xyz[d], atol=1e-6).nonzero()[0]
          for d in range(n_sg)]
    L = max_xyz - min_xyz

    def shift(d):
        s = np.zeros(n_sg); s[d] = L[d]
        return s

    if n_sg == 1:
        if n_model == 3:
            map_boundary(hi[0], lo[0], shift(0))
    elif n_sg == 2:
        if n_model == 3:                      # 2-D SG -> 3-D solid: both
            map_boundary(hi[0], lo[0], shift(0))
            map_boundary(hi[1], lo[1], shift(1))
        elif n_model == 2:                    # plate: in-plane only
            map_boundary(hi[0], lo[0], shift(0))
    elif n_sg == 3:
        if n_model == 3:
            map_boundary(hi[0], lo[0], shift(0))
            map_boundary(hi[1], lo[1], shift(1))
            map_boundary(hi[2], lo[2], shift(2))
        elif n_model == 2:
            map_boundary(hi[0], lo[0], shift(0))
            map_boundary(hi[1], lo[1], shift(1))
        else:
            map_boundary(hi[0], lo[0], shift(0))

    for _ in range(n_sg):
        dof_map = dof_map[dof_map]

    node_periodic_map = jnp.array(dof_map)
    master_nodes = node_periodic_map
    dof_offsets = jnp.arange(ndof_per_node)
    master_dof_indices = (master_nodes[:, None] * ndof_per_node
                          + dof_offsets[None, :])
    return master_dof_indices.flatten()


def apply_pbc_reduction(K_full, R_full, dof_map_full):
    """Condense K (N,N) -> (M,M) and R (N,) -> (M,) onto the unique master DOFs
    (K_red = L^T K L by scatter-add through the dof map)."""
    unique_dofs = jnp.unique(dof_map_full)

    R_accum = jnp.zeros_like(R_full).at[dof_map_full].add(R_full)
    R_reduced = R_accum[unique_dofs]

    K_col_reduced = jnp.zeros_like(K_full).at[:, dof_map_full].add(K_full)
    K_total = jnp.zeros_like(K_col_reduced).at[dof_map_full, :].add(K_col_reduced)
    K_reduced = K_total[jnp.ix_(unique_dofs, unique_dofs)]

    return K_reduced, R_reduced, unique_dofs
