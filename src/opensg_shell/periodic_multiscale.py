"""periodic_multiscale.py (msg_shell) -- periodic local->global sparse assembly
map for SHELL structure genes.

The shell counterpart of `fe_jax/periodic_multiscale.py`, kept deliberately
API-identical so both routes are driven the same way: pair opposite
bounding-box faces, resolve corner/edge chains by repeating dof_map[dof_map],
compress to the unique masters, and re-point the element connectivity at them.
Periodicity then rides in the local<->global assembly map -- no master-slave
transformation matrix, no constraint rows.

Only two things differ from the solid version:
  * a shell node carries 6 DOFs (w1,w2,w3,om1,om2,om3) instead of 3;
  * the elements are 2-node lines in the (y2,y3) plane.
The DOFs themselves are GLOBAL components (the strain operators contract them
with the global direction cosines x_{i;alpha} and C_{3i}), so a periodic image
needs no rotation -- an index map is exactly right, same as for the solid.

Which faces are paired follows the same (n_sg, n_model) switch as the core:
n_model = 3 (solid macro) ties both in-plane directions, 2 (plate) ties the
first only, 1 (beam) ties none.
"""
import numpy as np


def _load_core_periodic_map():
    """The core map from src/fe_jax/periodic_multiscale.py, loaded by path so
    the heavy fe_jax package __init__ (flax/cupy) is not required."""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "fe_jax", "periodic_multiscale.py")
    try:
        spec = importlib.util.spec_from_file_location("_fe_jax_periodic", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.periodic_map
    except Exception:
        return None


_CORE_MAP = _load_core_periodic_map()


def periodic_map(points, n_model=3, atol=1e-6, ndof_per_node=6):
    """Flattened master-DOF index array, exactly as the core map returns it,
    but with 6 DOFs per shell node."""
    pts = np.asarray(points, float)
    if _CORE_MAP is not None:
        return np.asarray(_CORE_MAP(pts, n_model, atol, ndof_per_node),
                          dtype=np.int64)
    return _periodic_map_fallback(pts, n_model, atol, ndof_per_node)


def _periodic_map_fallback(points, n_model, atol, ndof_per_node):
    """Same algorithm, numpy/scipy only (used when fe_jax cannot be loaded)."""
    from scipy.spatial.distance import cdist

    min_xyz, max_xyz = np.min(points, axis=0), np.max(points, axis=0)
    dof_map = np.arange(len(points))
    n_sg = points.shape[1]
    L = max_xyz - min_xyz
    ndir = {3: n_sg, 2: 1, 1: 0}.get(n_model, n_sg)

    for d in range(ndir):
        slaves = np.isclose(points[:, d], max_xyz[d], atol=1e-6).nonzero()[0]
        masters = np.isclose(points[:, d], min_xyz[d], atol=1e-6).nonzero()[0]
        if len(slaves) == 0 or len(masters) == 0:
            continue
        shift = np.zeros(n_sg); shift[d] = L[d]
        dists = cdist(points[slaves] - shift, points[masters])
        nearest = np.argmin(dists, axis=1)
        if not np.all(dists[np.arange(len(dists)), nearest] < atol):
            raise ValueError("Geometric mismatch on periodic boundary")
        dof_map[slaves] = masters[nearest]
    for _ in range(n_sg):
        dof_map = dof_map[dof_map]

    offs = np.arange(ndof_per_node)
    return (dof_map[:, None] * ndof_per_node + offs[None, :]).ravel()


def mesh_to_periodic_sparse_assembly_map(V, cells, points, n_model=3,
                                         ndof_per_node=6, atol=1e-6):
    """Shell analogue of the solid-side call.

    Returns (reduced_periodic_cells, dof_map_np, n_unique):
      reduced_periodic_cells (E, 2)  line connectivity re-pointed at the
                                     COMPRESSED master nodes;
      dof_map_np             (V,)    node -> compressed master, the array the
                                     assemblers take as `dof_map`;
      n_unique               int     independent node count.
    """
    dof_map_np = np.asarray(periodic_map(points, n_model, atol, ndof_per_node),
                            dtype=np.int64)
    master_nodes = dof_map_np[::ndof_per_node] // ndof_per_node

    unique_masters = np.unique(master_nodes)
    full_to_reduced = np.full(V, -1, dtype=np.int32)
    full_to_reduced[unique_masters] = np.arange(len(unique_masters),
                                                dtype=np.int32)
    node_master = full_to_reduced[master_nodes]
    reduced_periodic_cells = node_master[np.asarray(cells, int)]
    return reduced_periodic_cells, node_master.astype(int), len(unique_masters)


def periodic_node_map(points, n_model=3, tol=1e-6):
    """Convenience wrapper: (node -> compressed master, n_unique)."""
    pts = np.asarray(points, float)
    _, node_master, n_unique = mesh_to_periodic_sparse_assembly_map(
        len(pts), np.zeros((0, 2), int), pts, n_model, 6, tol)
    return node_master, n_unique


def ring_strip_dof_map(node_master, m):
    """dof_map for the one-quad-deep prismatic strip: the top row of nodes
    repeats the bottom row, so the master map is tiled twice."""
    nm = np.asarray(node_master, int)
    assert len(nm) == m, "node_master must have one entry per contour node"
    return np.concatenate([nm, nm])


def rigid_modes(periodic):
    """Rigid-body modes the solve must pin: 3 translations once opposite edges
    are tied (periodicity kills the in-plane rotation), 4 for a free SG."""
    return 3 if periodic else 4
