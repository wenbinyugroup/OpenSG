"""periodic_multiscale.py (msg_shell) -- periodic local->global sparse assembly
map for a SHELL mesh, 2-D SG / solid macro model (n_model = 3).

Replication of the solid-side fe_jax/periodic_multiscale.py: same two
functions, same signatures, same body; the only change is ndof_per_node = 6,
because a shell contour node carries (w1,w2,w3,om1,om2,om3) instead of the
solid's three translations.  The DOFs are global components (the strain
operators contract them with the global direction cosines), so the tie is a
plain index map here too -- periodicity rides in the assembly map and no
transformation is formed.
"""
import numpy as np
import jax.numpy as jnp
from scipy.spatial.distance import cdist


def mesh_to_periodic_sparse_assembly_map(
    V: int,
    cells,
    points,
    n_model: int,
    ndof_per_node: int = 6,
    atol=1e-6,
):
    dof_map_np = np.array(periodic_map(points, n_model, atol, ndof_per_node))
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


def periodic_map(points, n_model, atol=1e-6, ndof_per_node=6):
    points = np.asarray(points)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    num_nodes = len(points)
    dof_map = np.arange(num_nodes)
    n_sg = points.shape[1]

    def map_boundary(slaves, masters, shift_vec):
        if len(slaves) == 0:
            return
        slave_pts = points[slaves]
        master_pts = points[masters]
        target_pos = slave_pts - shift_vec
        dists = cdist(target_pos, master_pts)
        nearest_idx = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(len(dists)), nearest_idx]

        if not np.all(min_dists < atol):
            raise ValueError("Geometric mismatch on periodic boundary")

        dof_map[slaves] = masters[nearest_idx]

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
        if n_model == 3:
            # 1. Map Right -> Left (Shift x by Lx)
            map_boundary(hi[0], lo[0], shift(0))
            # 2. Map Top -> Bottom (Shift y by Ly)
            map_boundary(hi[1], lo[1], shift(1))
        elif n_model == 2:
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
