"""sg_mesh.py -- SG input parsing and mesh utilities of the general SG
engine, split out of rm_plate_2D.py (SSDM provenance, see sg_homo.py):
load_sg_input (.sc via opensg_solid.helper, or the helper's yaml),
_cell_basis (SG dim + nodes/elem -> basix cell type/degree), and
plot_sg_mesh (the ACTUAL mesh, elements colored by material, no title).

The legacy fe_jax core import is resolved by opensg_solid/__init__
($FE_JAX_CORE on sys.path) -- import this module through the package.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS MODULE
# ----------------------------------------------------------------------------
# inputs
#   path, out_base      the .sc/.yaml input; basename for the .sc conversion
#   dim, nodes_per_elem SG dimension (1/2/3) + nodes per element
#   thick, angles, mat_names, material_db, n_per_layer, elem_order,
#   fraction            laminate spec of laminate_to_sg (rm_plate_1D input)
# working
#   raw                 the yaml dict as loaded
#   table               {dim: {nodes/elem: (CellType, basis_degree)}}
#   nodes               (V, 3) node coordinates; mats (E,) material ids
#   uniq, color_of      material ids -> tab10 colors
#   polys, cols         2-D patch collection (corner nodes of quadratics)
#   _HEX_FACES/_TET_FACES, bfaces   exterior-face extraction (3-D render)
#   msh_path            optional .msh for the gmsh 3-D render (display
#                       needed; headless falls back to matplotlib)
#   node_x, bot, seq    laminate_to_sg through-thickness grid (ends-first
#                       gmsh/basix node order per element)
# outputs
#   sc                  {dim, nodes, cells, mat_id, materials, scale}
#   (CellType, degree)  the basix element for the SG
#   <path>.png          the mesh figure (elements by material, no title)
# ----------------------------------------------------------------------------
"""
import os

import numpy as np
import yaml

from fe_jax.basis_quadrature import CellType

from opensg_solid.helper.sc_to_yaml import convert as sc_convert


def _cell_basis(dim, nodes_per_elem):
    """(CellType, basis_degree) from the SG dimension + nodes/element."""
    table = {
        1: {2: (CellType.interval, 1), 3: (CellType.interval, 2),
            4: (CellType.interval, 3), 5: (CellType.interval, 4)},
        2: {3: (CellType.triangle, 1), 4: (CellType.quadrilateral, 1)},
        3: {4: (CellType.tetrahedron, 1), 8: (CellType.hexahedron, 1),
            10: (CellType.tetrahedron, 2)},
    }
    try:
        return table[dim][nodes_per_elem]
    except KeyError:
        raise ValueError("unsupported %dD element with %d nodes"
                         % (dim, nodes_per_elem))


def _numify(v):
    """Recursively convert CBaseLoader string scalars to numbers (leave true
    text -- names, types -- as strings)."""
    if isinstance(v, dict):
        return {k: _numify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_numify(x) for x in v]
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f.is_integer() and ("." not in v and "e" not in v.lower()) else f
        except ValueError:
            return v
    return v


_HEADER_STOP = ("nodes", "cells", "mat_id", "materials", "elements",
                "sets", "sections", "elementOrientations")


def read_yaml_header(path):
    """The leading scalar keys of an SG yaml -- the ANALYSIS REQUEST the
    file carries about itself, read without parsing the mesh blocks.

    The header sits at the top of the yaml, above the first big block
    key (nodes/cells/mat_id/materials/...), and may carry `msg`
    (shell | solid -- which ENGINE owns the file, see resolve_msg),
    `n_model` (1 beam / 2 plate / 3 3-D solid), `refined` (0 classical /
    1 shear-refined), `analysis` (H = homogenization, D =
    dehomogenization) and, for D, `epsilon_bar` -- the (6,) macro
    state.  Every key has a default; a headerless mesh is a classical
    plate homogenization whose engine is inferred from the dialect.

    In:  path str -- the SG yaml (anything else returns {})
    Out: dict of the header keys, {} when there is no header."""
    if not (isinstance(path, str)
            and path.lower().endswith((".yaml", ".yml"))
            and os.path.exists(path)):
        return {}
    head = []
    with open(path) as f:
        for ln in f:
            key = ln.split(":", 1)[0].strip() if ":" in ln else None
            if key in _HEADER_STOP and not ln[:1].isspace():
                break
            head.append(ln)
            if len(head) > 200:      # headers are a few lines; never scan
                break                # into a mesh that lacks the stop key
    try:
        d = yaml.safe_load("".join(head)) or {}
    except yaml.YAMLError:
        return {}
    return d if isinstance(d, dict) else {}


# ---------------------------------------------------------------------------
# WHICH ENGINE, AND WHICH SG -- the two facts every command prints, read from
# the file alone.  ONE implementation: opensg_solid.cli, opensg_shell.cli and
# the unified `opensg` dispatcher all call these.
# ---------------------------------------------------------------------------
# nodes-per-element -> the SG dimensions that element can belong to, the SAME
# table _cell_basis accepts (1-D interval p1..p4, 2-D tri/quad, 3-D tet/hex)
_CELL_NODES = {1: (2, 3, 4, 5), 2: (3, 4), 3: (4, 8, 10)}


def block_keys(path):
    """The mesh-block keys an SG yaml actually carries, by a line scan.

    A top-level key is a line that starts in column 0 and is not a list
    item -- the same test read_yaml_header stops on.

    In:  path str -- an SG yaml
    Out: set[str] -- the members of _HEADER_STOP present as top-level keys."""
    seen = set()
    with open(path) as f:
        for ln in f:
            c = ln[:1]
            if not c or c.isspace() or c == "-" or c == "#":
                continue                      # the ~1e6 mesh rows, skipped fast
            if ":" in ln:
                k = ln.split(":", 1)[0].strip()
                if k in _HEADER_STOP:
                    seen.add(k)
    return seen


def sg_dialect(path):
    """Which SG-yaml DIALECT the mesh blocks are written in -- i.e. which
    engine's loader can read this file, read from the file itself.

    Two dialects share the name "SG yaml":

      shell   nodes / elements / sets / sections / elementOrientations /
              materials -- an msg-shell contour or surface mesh: its
              `sections` carry a LAYUP and every element carries a material
              frame (opensg_shell.sg_mesh indexes both unconditionally)
      solid   nodes / cells / mat_id / materials -- the general SG engine's
              mesh (opensg_solid.load_sg_input), plus the two older
              opensg_solid spellings that use `elements` instead of `cells`:
              the 1-D through-thickness plate SG (per-ply `sections`, NO
              element frames) and the 2-D solid SG with per-element
              `elementOrientations` (NO `sections`)

    In:  path str -- an SG yaml
    Out: "shell" | "solid" | None (no recognizable SG mesh block: an ABD
         cache, a layup database, a macro-state or windIO file)."""
    b = block_keys(path)
    if "cells" in b or "mat_id" in b:
        return "solid"
    if {"elements", "sections", "elementOrientations"} <= b:
        return "shell"
    if "elements" in b:
        return "solid"
    return None


def resolve_msg(path):
    """The MSG engine that owns this SG yaml: "shell" or "solid".

    The `msg:` header key is the file's own explicit statement; when it is
    absent the mesh DIALECT decides (sg_dialect), so every yaml written
    before the key existed keeps working.  A `msg:` that contradicts the
    dialect is an error, not a silent override -- it means the file was
    copied or edited into the wrong engine's tree.

    In:  path str -- an SG yaml
    Out: "shell" | "solid" (raises SystemExit when neither can be had)."""
    stated = read_yaml_header(path).get("msg")
    dialect = sg_dialect(path)
    if stated is None:
        if dialect is None:
            raise SystemExit(
                "%s carries no `msg:` key and no recognizable SG mesh block,"
                " so the\nengine cannot be resolved: name it with `msg: shell`"
                " or `msg: solid` in the\nheader, or give a yaml whose mesh is"
                " the shell dialect (nodes / elements / sets /\nsections /"
                " elementOrientations / materials) or the solid dialect"
                " (nodes / cells /\nmat_id / materials)." % path)
        return dialect
    msg = str(stated).strip().lower()
    if msg not in ("shell", "solid"):
        raise SystemExit("%s: `msg:` must be shell or solid, got %r"
                         % (path, stated))
    if dialect is not None and dialect != msg:
        raise SystemExit(
            "%s: the header says `msg: %s` but the mesh blocks are the %s"
            " dialect\n(%s).\nThe %s engine cannot read a %s SG -- fix the"
            " `msg:` key or the mesh."
            % (path, msg, dialect, ", ".join(sorted(block_keys(path))),
               msg, dialect))
    return msg


def elem_node_count(path):
    """Nodes per element of the FIRST element, from either dialect's
    connectivity block (`elements:` or `cells:`).

    `- [1 2]` is one yaml scalar "1 2" (no commas) while `- [1, 2]` is a
    two-item list: both spellings are counted as tokens, the way _row does,
    so the node count never depends on the flow style.

    In:  path str -- an SG yaml
    Out: int -- nodes per element."""
    head, on = [], False
    with open(path) as f:
        for ln in f:
            top = (":" in ln) and not ln[:1].isspace() and not ln.startswith("-")
            if top:
                if on:
                    break                       # the block ended: next key
                on = ln.split(":", 1)[0].strip() in ("elements", "cells")
                continue
            if on:
                head.append(ln)
                if len(head) >= 8:              # the first item is enough
                    break
    for k in range(len(head), 0, -1):           # trim back to a parsable item
        try:
            e = yaml.safe_load("elements:\n" + "".join(head[:k]))["elements"][0]
        except Exception:
            continue
        return len(" ".join(str(x) for x in
                            (e if isinstance(e, list) else [e])).split())
    raise SystemExit("no `elements:`/`cells:` connectivity block in %s -- is it"
                     " an SG yaml ?" % path)


def node_span_dim(path):
    """The number of LEADING node-coordinate columns that actually vary.

    load_sg_input's own rule for a mesh that declares no `dim:` -- the SG
    coordinates always occupy the leading columns (points = nodes[:, 0:n_sg]).
    Used here only to break the element-count ties of the solid dialect.

    In:  path str -- an SG yaml
    Out: int 1 | 2 | 3."""
    lo = hi = None
    on = False
    with open(path) as f:
        for ln in f:
            if (":" in ln) and not ln[:1].isspace() and not ln.startswith("-"):
                if on:
                    break
                on = ln.split(":", 1)[0].strip() == "nodes"
                continue
            if not on:
                continue
            s = ln.strip()
            if s.startswith("- "):              # the list dash only -- never a
                s = s[2:]                       # coordinate's minus sign
            elif s.startswith("-["):
                s = s[1:]
            try:
                v = [float(t) for t in s.replace("[", " ").replace("]", " ")
                                        .replace(",", " ").split()]
            except ValueError:
                continue
            if not v:
                continue
            if lo is None:
                lo, hi = list(v), list(v)
                continue
            for i in range(min(len(lo), len(v))):
                lo[i] = min(lo[i], v[i])
                hi[i] = max(hi[i], v[i])
    if lo is None:
        return 1
    span = [h - l for l, h in zip(lo, hi)]
    tol = 1e-9 * max(max(span), 1.0)
    used = [i for i, s in enumerate(span) if s > tol]
    return (used[-1] + 1) if used else 1


def sg_dim(path):
    """The INTRINSIC (topological) dimension of the SG mesh: 1, 2 or 3.

    The dimension of the SG MANIFOLD, never of the space it is embedded in.
    A 1-D shell SG -- a cross-section contour of 2-node line elements -- is
    1; a 3-D shell SG -- a surface in space, tri/quad facets -- is 2, its
    nodes carrying three coordinates but the SG being two-dimensional.  It
    is therefore read from the ELEMENT NODE COUNT, not from the coordinate
    columns; only where a count is genuinely ambiguous in the solid dialect
    (4 nodes = a cubic line, a quad or a tet) does node_span_dim break the
    tie, which is load_sg_input's own rule.

    In:  path str -- an SG yaml
    Out: int 1 | 2 | 3."""
    n = elem_node_count(path)
    if sg_dialect(path) == "shell":
        return 1 if n <= 2 else 2      # a shell SG is a contour or a surface
    cand = [d for d in (1, 2, 3) if n in _CELL_NODES[d]]
    if not cand:
        raise SystemExit("unsupported element with %d nodes in %s" % (n, path))
    if len(cand) == 1:
        return cand[0]
    s = node_span_dim(path)
    if s in cand:
        return s
    below = [d for d in cand if d <= s]
    return max(below) if below else min(cand)


def load_sg_input(path, out_base=None):
    """The SG from EITHER a SwiftComp .sc (converted via the helper,
    which also writes <base>.yaml/.msh), a yaml directly (the NATIVE
    user-authored SG input -- the .sc helper emits the same dialect),
    or an ALREADY-PARSED sc dict (passed through unchanged -- the
    laminate_to_sg route).

    Speed: the yaml is parsed with CBaseLoader (libyaml parser, NO Python
    resolver/constructor typing -- the pure/typed loaders spend ~30 s building
    the ~1e6 typed objects of a multi-MB 3-D SG) and the big blocks are bulk
    numpy-converted; the result is cached in a <base>_sg.npz sidecar that later
    runs load in ~0.2 s (invalidated by the yaml mtime).

    Out: the parsed dict {dim, nodes, cells, mat_id, materials, scale}."""
    if isinstance(path, dict):
        return path
    ext = os.path.splitext(path)[1].lower()
    if ext == ".sc":
        return sc_convert(path, out_base)

    npz = os.path.splitext(path)[0] + "_sg.npz"
    if os.path.exists(npz) and os.path.getmtime(npz) >= os.path.getmtime(path):
        z = np.load(npz, allow_pickle=True)
        return {"dim": int(z["dim"]), "nodes": np.asarray(z["nodes"], float),
                "cells": [list(c) for c in z["cells"]],
                "mat_id": np.asarray(z["mat_id"], int),
                "materials": z["materials"].item(),
                "scale": float(z["scale"])}

    try:
        from yaml import CBaseLoader as _YL      # libyaml, untyped (all strings)
    except ImportError:
        from yaml import SafeLoader as _YL
    raw = yaml.load(open(path), Loader=_YL)
    nodes = np.asarray(raw["nodes"], float)      # numpy bulk str -> float
    try:                                          # rectangular fast path
        cells_arr = np.asarray(raw["cells"], np.int64)
        cells = [list(c) for c in cells_arr]
    except (ValueError, TypeError):               # mixed element sizes
        cells_arr = None
        cells = [[int(v) for v in c] for c in raw["cells"]]
    mat_id = np.asarray(raw["mat_id"], np.int64)
    materials = {int(k): _numify(v) for k, v in raw["materials"].items()}
    scale = float(raw.get("scale", 1.0))
    dim = raw.get("dim")
    if dim is None:
        # no user input needed: the SG dimension IS the mesh -- the
        # coordinates always occupy the LEADING columns (points =
        # nodes[:, 0:n_sg]), so dim = highest column that actually varies
        span = nodes.max(axis=0) - nodes.min(axis=0)
        used = span > 1e-9 * max(float(span.max()), 1.0)
        dim = int(np.nonzero(used)[0].max() + 1) if used.any() else 1
    dim = int(dim)
    sc = {"dim": dim, "nodes": nodes, "cells": cells,
          "mat_id": mat_id, "materials": materials, "scale": scale}
    try:                                          # best-effort sidecar cache
        np.savez_compressed(
            npz, dim=dim, nodes=nodes,
            cells=(cells_arr if cells_arr is not None
                   else np.array([np.array(c) for c in cells], dtype=object)),
            mat_id=mat_id, materials=np.array(materials, dtype=object),
            scale=scale)
    except Exception:
        pass
    return sc


def laminate_to_sg(thick, angles, mat_names, material_db,
                   n_per_layer=1, elem_order=4, fraction=0.0):
    """1-D through-thickness SG dict from a laminate spec -- the
    sg-engine equivalent of the rm_plate_1D layup input (the 1-D
    unification parity route).

    Materials are emitted as PRE-ROTATED type-2 ply C blocks built with
    rm_plate_1D.msg_materials.rotated_stiffness_6x6 -- the rm_plate_1D
    stack stays the source of truth for the ply stiffness, and its
    rotation convention (rotation_6x6(t) == sg_materials.rotate_C_matrix(-t))
    cannot confound a parity comparison.

    Node ordering per element is gmsh/basix line style: the two END
    nodes first, then the interior nodes left-to-right (what the
    engine's equispaced Lagrange basis expects).

    In:  thick (nlay,), angles (nlay,) deg, mat_names (nlay,),
         material_db {name: {E:[3], G:[3], nu:[3]}}; n_per_layer int;
         elem_order p (nodes/elem = p+1, engine supports 1..4);
         fraction -- reference plane, 0=OML .. 1=IML (z_ref = fraction*h)
    Out: sc dict {dim=1, nodes (V, 3), cells, mat_id, materials, scale}."""
    from opensg_solid.rm_plate_1D.msg_materials import rotated_stiffness_6x6

    p = int(elem_order)
    nlay = len(thick)
    h = float(sum(thick))
    z_ref = float(fraction) * h
    n_elem = nlay * n_per_layer
    n_node = p * n_elem + 1

    node_x = np.empty(n_node)
    idx = 0
    bot = np.concatenate([[0.0], np.cumsum(np.asarray(thick, float))])
    for k in range(nlay):
        for s in range(n_per_layer):
            xl = bot[k] + thick[k] * s / n_per_layer
            xr = bot[k] + thick[k] * (s + 1) / n_per_layer
            for j in range(p):
                node_x[p * idx + j] = xl + (xr - xl) * j / p
            idx += 1
    node_x[p * n_elem] = bot[-1]
    node_x = node_x - z_ref

    cells = []
    for e in range(n_elem):
        seq = list(range(p * e, p * e + p + 1))
        cells.append([seq[0], seq[-1]] + seq[1:-1])   # ends first, then interior

    mat_id = np.repeat(np.arange(nlay) + 1, n_per_layer)
    materials = {k + 1: {"type": 2,
                         "C": np.asarray(rotated_stiffness_6x6(
                             material_db[mat_names[k]]["E"],
                             material_db[mat_names[k]]["G"],
                             material_db[mat_names[k]]["nu"],
                             angles[k]), float)}
                 for k in range(nlay)}
    nodes = np.zeros((n_node, 3))
    nodes[:, 0] = node_x
    return {"dim": 1, "nodes": nodes, "cells": cells,
            "mat_id": mat_id, "materials": materials, "scale": 1.0}


_HEX_FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _gmsh_render(msh_path, png_path):
    """3-D mesh PNG via the gmsh viewer (needs a display -- headless
    boxes raise and the caller falls back to matplotlib)."""
    import gmsh
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(msh_path)
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.option.setNumber("Mesh.ColorCarousel", 2)   # color by physical
        gmsh.fltk.initialize()
        gmsh.write(png_path)
    finally:
        gmsh.finalize()


def _boundary_faces(cells, mats):
    """Exterior faces of a tet4/tet10/hex8 batch: faces referenced by
    exactly one element, each tagged with its owner's material id.
    Vectorized (sorted-face np.unique) for uniform batches -- the python
    dict pass costs ~7 s on a 546k-tet SG; mixed batches fall back."""
    try:
        carr = np.asarray(cells, np.int64)
    except (ValueError, TypeError):
        carr = None
    if carr is not None and carr.ndim == 2 and carr.shape[1] in (4, 8, 10):
        nn = carr.shape[1]
        if nn in (4, 10):
            fidx = np.asarray(_TET_FACES, int)
            corner = carr[:, :4]
        else:
            fidx = np.asarray(_HEX_FACES, int)
            corner = carr
        F = corner[:, fidx]                       # (E, nf, k), owner face order
        E, nf, k = F.shape
        flat = F.reshape(E * nf, k)
        key = np.sort(flat, axis=1)
        _, first, cnt = np.unique(key, axis=0, return_index=True,
                                  return_counts=True)
        ext = first[cnt == 1]                     # faces referenced exactly once
        emat = np.repeat(np.asarray(mats, int), nf)
        return [(tuple(int(v) for v in flat[i]), int(emat[i])) for i in ext]

    count = {}
    for ci, (c, m) in enumerate(zip(cells, mats)):
        nn = len(c)
        if nn in (4, 10):
            faces = _TET_FACES
            corner = c[:4]
        elif nn == 8:
            faces = _HEX_FACES
            corner = c
        else:
            return None
        for f in faces:
            nodes = tuple(corner[i] for i in f)
            key = tuple(sorted(nodes))
            if key in count:
                count[key] = None
            else:
                count[key] = (nodes, m)
    return [v for v in count.values() if v is not None]


def plot_sg_mesh(sc, path, msh_path=None):
    """PNG of the ACTUAL SG mesh, elements colored by material id (no
    title -- captions live in the document).  1-D: segments on a line;
    2-D: filled tri/quad patches (corner nodes of quadratic elements);
    3-D: gmsh render of msh_path when a display is available, else the
    exterior boundary faces (Poly3DCollection) -- a real mesh render
    either way, never a sketch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    nodes = np.asarray(sc["nodes"], float)
    mats = np.asarray(sc["mat_id"], int)
    uniq = np.unique(mats)
    cmap = plt.get_cmap("tab10")
    color_of = {m: cmap(i % 10) for i, m in enumerate(uniq)}
    dim = sc["dim"]
    fig = plt.figure(figsize=(7.5, 6.0))
    if dim == 1:
        ax = fig.add_subplot(111)
        for c, m in zip(sc["cells"], mats):
            xs = nodes[[c[0], c[-1]], 0]
            ax.plot(xs, [0, 0], "-", color=color_of[m], lw=6,
                    solid_capstyle="butt")
        ax.plot(nodes[:, 0], np.zeros(len(nodes)), "k.", ms=4)
        ax.set_yticks([])
        ax.set_xlabel("$y_3$")
    elif dim == 2:
        ax = fig.add_subplot(111)
        ncorner = {3: 3, 4: 4, 6: 3, 8: 4, 9: 4}
        polys, cols = [], []
        for c, m in zip(sc["cells"], mats):
            k = ncorner.get(len(c), len(c))
            polys.append(nodes[np.asarray(c[:k], int), :2])
            cols.append(color_of[m])
        ax.add_collection(PolyCollection(polys, facecolors=cols,
                                         edgecolors="k", linewidths=0.15))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("$y_1$")
        ax.set_ylabel("$y_2$")
    else:
        if msh_path is not None and os.path.exists(str(msh_path)):
            try:
                _gmsh_render(str(msh_path), path)
                plt.close(fig)
                return
            except Exception:
                pass                    # headless / no gmsh -> matplotlib
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax = fig.add_subplot(111, projection="3d")
        bfaces = _boundary_faces(sc["cells"], mats)
        if bfaces:
            polys = [nodes[np.asarray(f, int), :3] for f, _ in bfaces]
            cols = [color_of[m] for _, m in bfaces]
            coll = Poly3DCollection(polys, facecolors=cols,
                                    edgecolors="k", linewidths=0.1)
            ax.add_collection3d(coll)
            lo = nodes[:, :3].min(axis=0)
            hi = nodes[:, :3].max(axis=0)
            c0 = 0.5 * (lo + hi)
            r = 0.5 * float((hi - lo).max())
            ax.set_xlim(c0[0] - r, c0[0] + r)
            ax.set_ylim(c0[1] - r, c0[1] + r)
            ax.set_zlim(c0[2] - r, c0[2] + r)
        else:                           # unknown 3-D element: centroid cloud
            cent = np.stack([nodes[np.asarray(c, int)].mean(axis=0)
                             for c in sc["cells"]])
            for m in uniq:
                s = mats == m
                ax.scatter(cent[s, 0], cent[s, 1], cent[s, 2], s=2,
                           color=color_of[m])
        ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$"); ax.set_zlabel("$y_3$")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=color_of[m],
                          label="mat %d" % m) for m in uniq]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.86, 0.5),
               frameon=False)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
