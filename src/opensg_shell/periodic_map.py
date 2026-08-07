"""periodic_map.py -- periodic local->global sparse assembly map for SHELL SGs.

Shell-mesh counterpart of the solid-side core map
(`fe_jax.setup.mesh_to_periodic_sparse_assembly_map`, the one
`opensg_solid.sg_homo` feeds to every macro model -- n_model 1 beam / 2 plate /
3 solid -- exactly as in the original single-file JAX_BICGoptimize driver).

Same rule, same architecture:
  * opposite bounding-box faces are paired (right -> left shifted by Lx,
    top -> bottom by Ly, front -> back by Lz),
  * corner/edge chains are resolved by repeating dof_map = dof_map[dof_map],
  * the ELEMENT CONNECTIVITY is then re-pointed at the master nodes.
Periodicity therefore needs no extra constraint rows: it is carried by the
assembly map, so every scatter/gather (local -> global and global -> local)
lands on the master DOF automatically.

Shell difference: a contour node carries 6 DOFs (w1,w2,w3,om1,om2,om3) rather
than 3, and the elements are 2-node lines in the (y2,y3) plane.  One further
consequence: pairing the two ends of a wall removes the rigid in-plane rotation
from the kernel (the free SG has 4 rigid modes, the periodic SG only the 3
translations), so `rigid_modes()` reports what the KKT kernel should carry.
"""
import numpy as np

def _load_core():
    """The core model-aware map (src/fe_jax/periodic_multiscale.py), loaded by
    file path so the heavy fe_jax package __init__ (flax/cupy) is not needed."""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "fe_jax", "periodic_multiscale.py")
    try:
        spec = importlib.util.spec_from_file_location("_periodic_multiscale", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.periodic_map, True
    except Exception:
        return None, False


_core_periodic_map, _CORE = _load_core()


def _periodic_map_local(points, tol=1e-6, ndof_per_node=3):
    """Verbatim mirror of fe_jax.setup.periodic_map, for environments where the
    fe_jax package cannot be imported (it pulls flax/cupy).  Kept identical so
    the shell and solid routes pair nodes by exactly the same rule."""
    from scipy.spatial.distance import cdist

    points = np.asarray(points, float)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    dof_map = np.arange(len(points))
    p = points.shape[1]
    L = max_xyz - min_xyz

    def map_boundary(slaves, masters, shift_vec):
        if len(slaves) == 0:
            return
        target_pos = points[slaves] - shift_vec
        dists = cdist(target_pos, points[masters])
        nearest_idx = np.argmin(dists, axis=1)
        if not np.all(dists[np.arange(len(dists)), nearest_idx] < tol):
            raise ValueError("Geometric mismatch on periodic boundary")
        dof_map[slaves] = masters[nearest_idx]

    for d in range(p):
        lo = np.isclose(points[:, d], min_xyz[d], atol=tol).nonzero()[0]
        hi = np.isclose(points[:, d], max_xyz[d], atol=tol).nonzero()[0]
        shift = np.zeros(p); shift[d] = L[d]
        map_boundary(hi, lo, shift)
    for _ in range(p):
        dof_map = dof_map[dof_map]

    offsets = np.arange(ndof_per_node)
    return (dof_map[:, None] * ndof_per_node + offsets[None, :]).ravel()


def periodic_node_map(points, n_model=3, tol=1e-6):
    """Node-level master map for a shell contour.

    points  : (m, 2) node coordinates in the cross-section plane.
    n_model : macro model, passed straight through to the core map -- 3 (solid)
              ties BOTH in-plane directions, 2 (plate) ties only the first,
              1 (beam) ties none.  Same switch as the solid side.
    Returns (node_master, n_unique): node_master[i] is the COMPRESSED index
    (0 .. n_unique-1) of the master node that node i is tied to, ready to be
    handed to the assemblers as `dof_map`."""
    pts = np.asarray(points, float)
    if _CORE:
        raw = np.asarray(_core_periodic_map(pts, n_model, tol, 1), dtype=np.int64)
    else:
        raw = np.asarray(_periodic_map_local(pts, tol, 1), dtype=np.int64)
    _, compressed = np.unique(raw, return_inverse=True)
    return compressed.astype(int), int(compressed.max()) + 1




def shell_periodic_assembly_map(points, cells, n_model=3, ndof_per_node=6,
                                tol=1e-6):
    """The shell analogue of mesh_to_periodic_sparse_assembly_map.

    points : (m, 2) contour nodes;  cells : (E, 2) line connectivity.
    Returns (periodic_cells, dof_map, n_unique):
      periodic_cells (E, 2)  element connectivity re-pointed at master nodes
                             -- scatter/gather through this and periodicity is
                             automatic, no constraint rows;
      dof_map        (m,)    node -> master (compressed);
      n_unique       int     number of independent nodes.
    The DOF-level map is periodic_cells*ndof_per_node + arange(ndof_per_node),
    the same construction the solid side uses with ndof_per_node = 3."""
    node_master, n_unique = periodic_node_map(points, n_model, tol)
    periodic_cells = node_master[np.asarray(cells, int)]
    return periodic_cells, node_master, n_unique


def ring_strip_dof_map(node_master, m):
    """dof_map for the one-quad-deep prismatic strip used by ring_solid: the
    top row of nodes repeats the bottom row, so the master map is simply
    tiled twice."""
    nm = np.asarray(node_master, int)
    assert len(nm) == m, "node_master must have one entry per contour node"
    return np.concatenate([nm, nm])


def check_frame_consistency(points, cells, node_master, tol=1e-6):
    """A shell DOF block (w1,w2,w3,om1,om2,om3) is expressed in the wall's own
    frame, so two nodes may only be tied when their local frames AGREE.  A
    periodic image of a wall end is a pure translation, hence its tangent must
    be PARALLEL (not antiparallel) to the master's.

    This catches the classic wrong cell: a closed ring (square tube) whose left
    and right walls sit on opposite bounding-box faces but are traversed in
    opposite directions -- tying them mixes +t with -t and yields an indefinite
    (negative-diagonal) stiffness.  The valid cell puts the paired faces at the
    two ends of the SAME wall (e.g. a cross / L cell of the lattice).

    Raises ValueError listing the offending pairs; returns the min dot product."""
    pts = np.asarray(points, float)
    cl = np.asarray(cells, int)
    nm = np.asarray(node_master, int)
    tan = np.zeros_like(pts)
    for a, b in cl:
        d = pts[b] - pts[a]
        n = np.linalg.norm(d)
        if n > 0:
            tan[a] += d / n
            tan[b] += d / n
    nrm = np.linalg.norm(tan, axis=1)
    ok = nrm > tol
    tan[ok] /= nrm[ok][:, None]

    bad, worst = [], 1.0
    for s in range(len(pts)):
        m = np.where(nm == nm[s])[0]
        for mm in m:
            if mm == s or not (ok[s] and ok[mm]):
                continue
            dot = float(tan[s] @ tan[mm])
            worst = min(worst, dot)
            if dot < 0.0:
                bad.append((s, mm, dot))
    if bad:
        head = "; ".join("nodes %d~%d (tangent dot %.3f)" % b for b in bad[:5])
        raise ValueError(
            "periodic tie joins nodes with opposing local shell frames: %s%s. "
            "Opposite walls of a closed ring are traversed in opposite "
            "directions -- use a cell whose paired faces are the two ends of "
            "the same wall (cross/L cell) instead."
            % (head, " (+%d more)" % (len(bad) - 5) if len(bad) > 5 else ""))
    return worst


def rigid_modes(periodic):
    """Number of rigid-body modes the KKT kernel must carry: 4 for a free SG
    (3 translations + the in-plane rotation), 3 once opposite edges are tied
    (periodicity removes the rotation)."""
    return 3 if periodic else 4
