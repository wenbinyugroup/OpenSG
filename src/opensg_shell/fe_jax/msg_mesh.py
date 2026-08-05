"""
MSG Shell — Cross-Section Mesh Utilities and YAML Loader

Provides:
  - load_yaml         : parse OpenSG-format YAML cross-section files
  - order_mesh        : chain disordered line elements into CCW closed loop
                        and insert midside nodes for 3-node quadratic elements
  - compute_curvature : per-element k22 via 3-point circumscribed circle
"""

import numpy as np
import yaml


# =============================================================================
# YAML Loader (OpenSG format)
# =============================================================================

def load_yaml(yaml_path):
    """Load an OpenSG-format YAML cross-section file.

    OpenSG uses space-separated YAML lists without commas, e.g. ``[1 2]``
    rather than ``[1, 2]``.  ``yaml.safe_load`` returns these as a single
    string inside a list, so a ``_parse_row`` helper is applied.

    Parameters
    ----------
    yaml_path : str — path to .yaml file

    Returns
    -------
    nodes_3d     : (N, 3) ndarray — node coordinates (z=0 for cross-sections)
    elements     : list of [n1, n2] — 1-based line element connectivity
    material_db  : dict {name: {E, G, nu, rho}}
    layup_db     : dict {layup_name: {mat_names, thick, angles}}
    elem_to_layup: dict {elem_id_1based: layup_name}
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    def _parse_row(row):
        if isinstance(row, str):
            return row.strip('[]').split()
        if isinstance(row, (list, tuple)):
            if len(row) == 1 and isinstance(row[0], str):
                return row[0].strip('[]').split()
            return [str(v) for v in row]
        return [str(row)]

    nodes_3d = np.array([[float(v) for v in _parse_row(nd)]
                         for nd in data['nodes']])
    elements = [[int(v) for v in _parse_row(el)]
                for el in data['elements']]

    material_db = {}
    for mat in data['materials']:
        material_db[mat['name']] = {
            'E':   mat['elastic']['E'],
            'G':   mat['elastic']['G'],
            'nu':  mat['elastic']['nu'],
            'rho': mat.get('density', 0.0),
        }

    elem_sets = {s['name']: s['labels'] for s in data['sets']['element']}

    layup_db = {}
    elem_to_layup = {}
    for sec in data['sections']:
        ln = sec['elementSet']
        layers = sec['layup']
        layup_db[ln] = {
            'mat_names': [ly[0] for ly in layers],
            'thick':     [ly[1] for ly in layers],
            'angles':    [ly[2] for ly in layers],
        }
        for eid in elem_sets[ln]:
            elem_to_layup[eid] = ln

    return nodes_3d, elements, material_db, layup_db, elem_to_layup


# =============================================================================
# Direct mesh (use the YAML connectivity verbatim, like FEniCS reads the .msh)
# =============================================================================

def read_mesh(nodes_3d, elements_1b, elem_to_layup):
    """Build the mesh straight from the YAML connectivity — NO chaining.

    This mirrors how OpenSG/FEniCS turns the YAML into a .msh: every node is a
    mesh vertex and every ``[n1 n2]`` (or ``[n1 nmid n2]``) is a line element,
    used exactly as given.  Because no single-loop walk is performed, ALL
    elements are kept, so multi-component cross-sections (airfoils with shear
    webs) are represented correctly.

    Parameters
    ----------
    nodes_3d     : (N, 3) YAML node coords; cross-section = columns (0, 1)
    elements_1b  : list of [n1, n2] (flat) or [n1, nmid, n2] (curved), 1-based
    elem_to_layup: dict {elem_id_1based: layup_name}

    Returns
    -------
    nodes : (N, 2) cross-section coordinates (y1, y2)
    cells : (E, k) int64 0-based connectivity, k = 2 (flat) or 3 (curved)
    layup_per_elem : list[str]  layup name per element (YAML order)
    """
    nodes = np.asarray(nodes_3d, dtype=float)[:, :2]
    cells = np.array([[int(v) - 1 for v in e] for e in elements_1b], dtype=np.int64)
    layup_per_elem = [elem_to_layup[i + 1] for i in range(len(elements_1b))]
    return nodes, cells, layup_per_elem


def element_e3_from_yaml(yaml_path):
    """(E,2) cross-section components of the material e3 (through-thickness, ply
    stacking normal) per element, from the YAML ``elementOrientations``.

    Each orientation row is 9 numbers; the OpenSG convention stores e3 such that
    its cross-section (y2, y3) components are ``(o[6], o[7])`` (o[8] ~ beam ~ 0).
    This e3 points OML->IML (inward) — the SAME direction the MSG plate dehom
    uses, so it is the ground-truth inward normal for the IML offset.
    """
    import yaml as _yaml
    with open(yaml_path) as f:
        d = _yaml.safe_load(f)

    def _row(r):
        if isinstance(r, str):
            return r.strip('[]').split()
        if isinstance(r, (list, tuple)) and len(r) == 1 and isinstance(r[0], str):
            return r[0].strip('[]').split()
        return [str(v) for v in r]
    ori = np.array([[float(v) for v in _row(o)] for o in d['elementOrientations']])
    e3 = ori[:, [6, 7]]
    return e3 / (np.linalg.norm(e3, axis=1, keepdims=True) + 1e-30)


def offset_oml_to_iml(nodes_2d, cells, layup_per_elem, layup_db, elem_e3=None,
                      frac=1.0):
    """Offset an OML reference mesh inward by ``frac`` x the laminate thickness.

    ``frac`` = 1.0 reaches the IML, 0.5 the mid-surface (material centroid — the
    most accurate reference, OML and IML bracket it), 0.0 stays at the OML.
    Each node is moved along the inward normal by ``frac`` x the local thickness
    (averaged over the elements that share it).  The connectivity is unchanged,
    so it stays a single light mesh (no new nodes/elements -> no memory blow-up).
    Pair this with :func:`msg_materials.shift_abd_reference` (z0 = frac x
    thickness, NOT a layup reversal) so the plate reference matches with e3 kept
    inward.

    Rationale: with an OML reference the thick spar-cap / web laminates extend
    inward and overlap at junctions; referencing the IML moves the material
    outward and reduces that double-counted overlap region.

    Parameters
    ----------
    nodes_2d : (N,2) OML cross-section coordinates
    cells    : (E,k) connectivity (uses end columns)
    layup_per_elem : list[str]  layup name per element
    layup_db : dict {name: {thick: [...], ...}}
    elem_e3  : (E,2) material e3 per element (from :func:`element_e3_from_yaml`).
               When given it is the inward direction (robust for webs/concave
               regions); otherwise a geometric centroid heuristic is used.

    Returns
    -------
    iml_nodes : (N,2) inward-offset coordinates
    """
    nodes = np.asarray(nodes_2d, dtype=float)
    n_node = nodes.shape[0]
    cen = nodes.mean(axis=0)
    acc_in = np.zeros((n_node, 2)); per_node = [[] for _ in range(n_node)]
    for e in range(cells.shape[0]):
        c0, c1 = int(cells[e, 0]), int(cells[e, -1])
        seg = nodes[c1] - nodes[c0]; L = np.hypot(seg[0], seg[1])
        if L < 1e-30:
            continue
        if elem_e3 is not None:
            inward = np.asarray(elem_e3[e], dtype=float)   # material e3 = inward
        else:
            t = seg / L
            inward = np.array([-t[1], t[0]])               # geometric normal
            mid = 0.5 * (nodes[c0] + nodes[c1])
            if (cen - mid) @ inward < 0.0:                 # orient toward centroid
                inward = -inward
        h = float(sum(layup_db[layup_per_elem[e]]['thick']))
        for c in (c0, c1):
            acc_in[c] += inward; per_node[c].append((inward, h))
    in_nrm = acc_in / (np.linalg.norm(acc_in, axis=1, keepdims=True) + 1e-30)
    # Thickness from the element whose inward is MOST ALIGNED with the consensus direction (the smooth
    # skin contour), NOT the average over all touching elements.  At a web/skin junction this picks the
    # skin thickness instead of the thick perpendicular web, so the node is not over-offset and the
    # thick-wall IML geometry does not fold (the old average folded it -> EI2/GA3 blew up at frac=1).
    thick = np.zeros(n_node)
    for c in range(n_node):
        if per_node[c]:
            thick[c] = max(per_node[c], key=lambda ih: float(ih[0] @ in_nrm[c]))[1]
    return nodes + frac * thick[:, None] * in_nrm          # frac x (skin) thickness inward


# =============================================================================
# Mesh Ordering (legacy single-loop chain — kept only for reference)
# =============================================================================

def order_mesh(nodes_3d, elements_1b, elem_to_layup):
    """Chain disordered line elements into a continuous CCW closed loop.

    Walks the adjacency graph from the first element's start node.
    If the resulting polygon has CW orientation (area < 0), the order is
    reversed.

    Two input element types are accepted:
      * 2-node ``[n1, n2]``  -> FLAT segment; the midside node is inserted at
        the chord midpoint, so the element carries zero curvature (k22 = 0).
      * 3-node ``[n1, nmid, n2]`` -> CURVED segment; the supplied (off-chord)
        midside node is kept, so ``compute_curvature`` recovers its real k22.

    Parameters
    ----------
    nodes_3d     : (N, 3) array — YAML coords (col0=y1, col1=y2 cross-section,
                   col2=z beam axis); only the cross-section (col0,col1) used
    elements_1b  : list of [n1, n2] (flat) or [n1, nmid, n2] (curved), 1-based
    elem_to_layup: dict {elem_id_1based: layup_name}

    Returns
    -------
    nodes_2d      : (2*M+1, 2) — ordered [y1, y2] coords with midside nodes
    cells         : (M, 3) int64 — element connectivity (0-based, 3-node)
    layup_per_elem: list[str] — layup name for each element (traversal order)
    is_closed     : bool
    """
    three_node = len(elements_1b[0]) >= 3
    if three_node:
        elems = [(e[0] - 1, e[2] - 1) for e in elements_1b]   # corner endpoints
        mids_given = [e[1] - 1 for e in elements_1b]          # supplied midside
    else:
        elems = [(e[0] - 1, e[1] - 1) for e in elements_1b]
        mids_given = None
    n_elems = len(elems)

    adj = {}
    for idx, (n1, n2) in enumerate(elems):
        adj.setdefault(n1, []).append((n2, idx))
        adj.setdefault(n2, []).append((n1, idx))

    start = elems[0][0]
    visited = set()
    nids = [start]
    eids = []
    cur = start

    while len(visited) < n_elems:
        found = False
        for nb, eidx in adj[cur]:
            if eidx not in visited:
                visited.add(eidx)
                nids.append(nb)
                eids.append(eidx)
                cur = nb
                found = True
                break
        if not found:
            break

    is_closed = (nids[-1] == nids[0])

    if len(visited) < n_elems:
        print(f"  WARNING: only {len(visited)}/{n_elems} elements chained "
              f"(mesh may be disconnected)")

    corner_coords = nodes_3d[nids, :2]

    # Signed area — ensure CCW
    area = 0.0
    for i in range(len(corner_coords) - 1):
        area += (corner_coords[i, 0] * corner_coords[i + 1, 1]
                 - corner_coords[i + 1, 0] * corner_coords[i, 1])
    area *= 0.5

    if area < 0:
        nids = nids[::-1]
        eids = eids[::-1]
        corner_coords = nodes_3d[nids, :2]

    n_elem = len(nids) - 1
    if three_node:
        # keep the supplied (possibly off-chord) midside for each element
        mid_coords = np.array([nodes_3d[mids_given[eidx], :2] for eidx in eids])
    else:
        # flat segments: midside at chord midpoint (collinear -> k22 = 0)
        mid_coords = 0.5 * (corner_coords[:-1] + corner_coords[1:])

    nodes_2d = np.zeros((2 * n_elem + 1, 2))
    nodes_2d[0::2] = corner_coords
    nodes_2d[1::2] = mid_coords

    cells = np.column_stack([
        np.arange(0, 2 * n_elem, 2),
        np.arange(1, 2 * n_elem + 1, 2),
        np.arange(2, 2 * n_elem + 2, 2),
    ]).astype(np.int64)

    layup_per_elem = [elem_to_layup[eidx + 1] for eidx in eids]

    return nodes_2d, cells, layup_per_elem, is_closed


# =============================================================================
# Curvature Computation
# =============================================================================

def compute_curvature(nodes_2d, cells, is_closed=True):
    """Per-element curvature k22 from the element's OWN three nodes.

    The circumscribed circle through (corner0, midside, corner1):
      * FLAT element  -> the three nodes are collinear (a 2-node segment has
        its midside on the chord), so the cross product vanishes and k22 = 0.
      * CURVED element -> an off-chord 3-node element gives k22 = -1/R.

    This is the geometrically honest curvature: curvature only appears when the
    element itself is curved, never as a smoothed average across flat segments.

    Sign convention: k22 = -(signed curvature) = -1/R for a CCW arc.

    Parameters
    ----------
    nodes_2d  : (2*N+1, 2) node coordinates including midside nodes
    cells     : (N, 3) element connectivity [corner0, midside, corner1]
    is_closed : bool — unused (kept for signature compatibility)

    Returns
    -------
    k22 : (N,) ndarray  (zero for flat elements)
    """
    p0 = nodes_2d[cells[:, 0]]
    p1 = nodes_2d[cells[:, 1]]
    p2 = nodes_2d[cells[:, 2]]
    n_elem = cells.shape[0]

    k22 = np.zeros(n_elem)
    for i in range(n_elem):
        d01 = p1[i] - p0[i]
        d12 = p2[i] - p1[i]

        cross = d01[0] * d12[1] - d01[1] * d12[0]
        l01   = np.linalg.norm(d01)
        l12   = np.linalg.norm(d12)
        l02   = np.linalg.norm(p2[i] - p0[i])
        denom = l01 * l12 * l02

        # collinear (flat) -> cross ~ 0 -> k22 = 0
        k22[i] = -2.0 * cross / denom if denom > 1e-30 else 0.0

    return k22


def mesh_curvature(nodes_2d, cells, elements_1b, is_closed=True):
    """Per-element curvature k22, gated on the YAML element type.

    If the YAML supplies **flat 2-node** line elements, every element is
    straight, so ``k22 = 0`` is returned immediately and the circumscribed-
    circle computation is skipped entirely.  Curvature is computed (via
    :func:`compute_curvature`) only when **curved 3-node** elements are given.

    Parameters
    ----------
    nodes_2d    : (2*N+1, 2) ordered node coordinates
    cells       : (N, 3) element connectivity
    elements_1b : the raw YAML element list (len 2 = flat, >=3 = curved)
    is_closed   : bool

    Returns
    -------
    k22 : (N,) ndarray  (all zeros for a flat 2-node mesh)
    """
    if len(elements_1b[0]) >= 3:
        return compute_curvature(nodes_2d, cells, is_closed)
    # Flat 2-node elements carry NO within-element curvature -> k22 = 0 for ALL of them (user directive).
    # A curved wall is captured by MESH REFINEMENT (more flat facets), not by a per-element curvature;
    # the old adjacency recovery (_curvature_from_corners) produced spurious k22 at webs / T-junctions /
    # sharp TE.  NOTE: this zeroes curvature for closed sections too (tube / st15) -> those now rely on
    # refinement for accuracy, matching how FEniCS-shell treats linear facets.
    return np.zeros(cells.shape[0])


def _curvature_from_corners(nodes_2d, cells):
    """Per-element wall curvature from the actual MESH ADJACENCY, not the element LIST order.

    For each element the curvature is the inter-element turning at its start corner, using the
    neighbour element that most smoothly CONTINUES this wall (max tangent alignment).  This is
    robust to multi-component meshes (airfoils with shear webs, where elements are not one ordered
    loop): a straight web -> the in-line web neighbour -> k22 = 0; a web/skin T-junction or a sharp
    TE wrap -> no smoothly-aligned neighbour (turn > ~78 deg) -> k22 = 0; a smooth skin -> its real
    -1/R.  (The old version used V[i-1],V[i],V[i+1] from the list order, which gave huge spurious
    curvature to appended web elements and to the TE.)"""
    N = cells.shape[0]
    ends = cells[:, [0, -1]].astype(np.int64)                 # (N,2) corner node ids
    touch = {}
    for ei in range(N):
        touch.setdefault(int(ends[ei, 0]), []).append(ei)
        touch.setdefault(int(ends[ei, 1]), []).append(ei)
    ALIGN_MIN = 0.2                                            # require turn < ~78 deg to be the same wall
    k22 = np.zeros(N)
    for i in range(N):
        a, b = int(ends[i, 0]), int(ends[i, 1])
        t_fwd = nodes_2d[b] - nodes_2d[a]; nf = np.linalg.norm(t_fwd)
        if nf < 1e-30:
            continue
        best_far, best_align = -1, ALIGN_MIN                  # wall-predecessor at corner a
        for e in touch[a]:
            if e == i:
                continue
            ea, eb = int(ends[e, 0]), int(ends[e, 1])
            far = eb if ea == a else ea
            t_prev = nodes_2d[a] - nodes_2d[far]; npv = np.linalg.norm(t_prev)
            if npv < 1e-30:
                continue
            align = float(np.dot(t_prev, t_fwd) / (npv * nf))
            if align > best_align:
                best_align, best_far = align, far
        if best_far < 0:                                      # web end / TE wrap / disconnected -> straight
            continue
        c = nodes_2d[best_far]
        d01, d12 = nodes_2d[a] - c, t_fwd
        cross = d01[0] * d12[1] - d01[1] * d12[0]
        denom = np.linalg.norm(d01) * nf * np.linalg.norm(nodes_2d[b] - c)
        k22[i] = -2.0 * cross / denom if denom > 1e-30 else 0.0
    return k22
