"""h_refine_msh.py -- UNIFORM conforming h-refinement of a gmsh mesh.

Every cell splits into 2^dim children, so h HALVES per level, while
geometry, physical tags and conformity are untouched:

    line2  ->  2 line2      tri3  ->  4 tri3      quad4 ->  4 quad4
    tet4   ->  8 tet4       hex8  ->  8 hex8
    tri6   ->  4 tri6       quad9 ->  4 quad9     tet10 -> 8 tet10

    from opensg import helper
    helper.msh_h_refine("mesh.msh")                 # -> mesh_h2.msh
    helper.msh_h_refine("mesh_h2.msh")              # -> mesh_h4.msh
    helper.msh_h_refine("mesh.msh", levels=2)       # -> mesh_h4.msh
    helper.msh_h_refine("mesh.msh", target_h=0.05)  # levels from h_max
    helper.verify_refinement("mesh.msh", "mesh_h2.msh")   # the gate

On the CLI the mode is ONE flag whose number is always LEVELS
(`opensg msh_to_yaml <mesh>.msh --h_refine N`, bare flag = 1).

THE CONFORMITY RULE, and why this is safe for MIXED meshes.  Every new
node is identified by the SET OF PARENT CORNERS that defines it -- two
for an edge midpoint, four for a quad-face centre, eight for a hex
body centre -- keyed by those corners' sorted GLOBAL ids.  Any two
cells that share an entity therefore compute the SAME key and land on
the SAME node, with no coordinate-tolerance guessing:
  * neighbouring tets agree on the 6 midpoints of a shared face,
  * a boundary tri3/quad4 marker surface agrees with the 3-D cells it
    covers (its edge nodes are the same keys), and
  * a boundary quad's centre node IS the hex's face centre, because
    both are keyed by the same four corner ids -- the one case where
    coordinate-hash dedup silently produces two coincident nodes.
New coordinates are the MEAN of the defining corners, exact for the
straight-sided / bi- and tri-linear cells this refines.

TETS pick their internal diagonal per element (the octahedron left
after the 4 corner tets admits three), taking the SHORTEST -- the
standard quality rule.  The choice is interior, so it never affects
conformity: a shared face always splits into the same 4 triangles.
Child tets are orientation-fixed (signed volume checked, two nodes
swapped when negative), so a positively-oriented parent stays that way.

QUADRATIC meshes are refined through their linear skeleton: the mesh
is demoted to corners, split, then re-promoted by
helper.linear_msh_to_quad.  That is EXACT for straight-sided cells and
wrong for genuinely curved ones, so the curvature is MEASURED first
(every midside against the mean of its corners) and a curved mesh is
refused rather than silently flattened.  The tolerance is 1e-4 of the
local edge -- decimal .msh text puts a straight midside ~1e-6 off the
chord in relative terms, while real curvature is O(1e-2).

3-D SGs are the expensive case: one level is 8x the cells, so the
6-level cap (4096x in 2-D, 262144x in 3-D) is a runaway guard, not a
target.

In:  msh_path str -- gmsh ASCII 2.2; out str | None (None -> the
     `_h<k>` naming rule, which reads a trailing `_h<k>` on the input
     so reruns chain); levels int | None (None with no target_h -> 1);
     target_h float | None -- subdivide until max edge <= target_h
     (PYTHON-API ONLY, mutually exclusive with levels); verbose bool
Out: dict {msh, levels, n_nodes, n_elems, h_min, h_mean, h_max} -- and
     the written file (levels 0: nothing written, msh = the input).
"""
import itertools
import math
import os
import re
import time

import numpy as np

# ---------------------------------------------------------------- topology
# Per gmsh type: nc = corner count, new = the defining corner-index sets
# of the nodes the split ADDS (slot order after the corners), children =
# templates over the local slot vector [corners..., new...].
_SIMPLEX = {
    1: {"nc": 2, "new": [(0, 1)],
        "children": [(0, 2), (2, 1)]},
    2: {"nc": 3, "new": [(0, 1), (1, 2), (0, 2)],
        "children": [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)]},
    # tet4 midside slots follow linear_msh_to_quad's edge order:
    # 4=m01 5=m12 6=m02 7=m03 8=m23 9=m13
    4: {"nc": 4,
        "new": [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)],
        "children": None},          # per-element: shortest diagonal
}

# the 4 corner tets, then the octahedron under each of its 3 diagonals
_TET_CORNERS = [(0, 4, 6, 7), (1, 4, 5, 9), (2, 6, 5, 8), (3, 7, 9, 8)]
_TET_OCTA = {
    (4, 8): [(4, 8, 6, 7), (4, 8, 7, 9), (4, 8, 9, 5), (4, 8, 5, 6)],
    (6, 9): [(6, 9, 4, 7), (6, 9, 7, 8), (6, 9, 8, 5), (6, 9, 5, 4)],
    (7, 5): [(7, 5, 4, 6), (7, 5, 6, 8), (7, 5, 8, 9), (7, 5, 9, 4)],
}
_TET_DIAGS = [(4, 8), (6, 9), (7, 5)]


def _tensor_topo(dim):
    """quad4 (dim 2) / hex8 (dim 3) split, generated from the 3^dim
    lattice so the tables cannot drift from the gmsh corner order.

    In:  dim int -- 2 or 3
    Out: {nc, new, children} in the _SIMPLEX layout."""
    off = [(0, 0), (1, 0), (1, 1), (0, 1)] if dim == 2 else \
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    corners = [tuple(2 * v for v in o) for o in off]
    slot, new = {}, []
    for i, c in enumerate(corners):
        slot[c] = i
    # a lattice point is defined by the corners matching it on every
    # EVEN coordinate (the odd ones are the directions it spans)
    pts = sorted(itertools.product(range(3), repeat=dim),
                 key=lambda p: (sum(v % 2 for v in p), p))
    for p in pts:
        if not any(v % 2 for v in p):
            continue                       # an existing corner
        d = tuple(i for i, c in enumerate(corners)
                  if all(c[k] == p[k] for k in range(dim) if p[k] % 2 == 0))
        slot[p] = len(corners) + len(new)
        new.append(d)
    children = []
    for base in itertools.product(range(2), repeat=dim):
        children.append(tuple(
            slot[tuple(base[k] + o[k] for k in range(dim))] for o in off))
    return {"nc": len(corners), "new": new, "children": children}


_TOPO = dict(_SIMPLEX)
_TOPO[3] = _tensor_topo(2)                 # quad4
_TOPO[5] = _tensor_topo(3)                 # hex8

_NAME = {1: "line2", 2: "tri3", 3: "quad4", 4: "tet4", 5: "hex8",
         8: "line3", 9: "tri6", 10: "quad9", 11: "tet10"}
# quadratic -> its linear skeleton (refined through the demote/promote
# route); line3 rides along with whatever cell type dominates
_QUAD_TO_LIN = {9: 2, 10: 3, 11: 4, 8: 1}
_QUAD_NC = {8: 2, 9: 3, 10: 4, 11: 4}
# the edges whose midside each quadratic type stores, in file order
_QUAD_EDGES = {8: [(0, 1)],
               9: [(0, 1), (1, 2), (0, 2)],
               10: [(0, 1), (1, 2), (2, 3), (3, 0)],
               11: [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]}
_CELL_DIM = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 8: 1, 9: 2, 10: 2, 11: 3}


# ------------------------------------------------------------------- gmsh io
def read_msh(path):
    """Minimal gmsh 2.2 ASCII reader.

    In:  path str
    Out: {ids (N,), xyz (N, 3), phys [str block lines], blocks [{etype,
         ntags, tags (E, nt), conn (E, nc) 0-based rows into xyz}]}."""
    with open(path) as f:
        lines = f.read().split("\n")
    ver = lines[lines.index("$MeshFormat") + 1].split()
    if not ver[0].startswith("2") or (len(ver) > 1 and ver[1] != "0"):
        raise SystemExit(
            "%s is not gmsh ASCII 2.2 -- re-export first:"
            "  gmsh %s -format msh2 -save" % (path, path))
    phys = []
    if "$PhysicalNames" in lines:
        phys = lines[lines.index("$PhysicalNames"):
                     lines.index("$EndPhysicalNames") + 1]
    i_n = lines.index("$Nodes")
    nn = int(lines[i_n + 1])
    nd = np.loadtxt(lines[i_n + 2:i_n + 2 + nn], ndmin=2)
    ids = nd[:, 0].astype(np.int64)
    order = np.argsort(ids)
    ids_s, xyz = ids[order], nd[order, 1:4]
    i_e = lines.index("$Elements")
    ne = int(lines[i_e + 1])
    groups = {}
    for v in (ln.split() for ln in lines[i_e + 2:i_e + 2 + ne]):
        groups.setdefault((int(v[1]), int(v[2])), []).append(v)
    blocks = []
    for (et, nt), rows in groups.items():
        a = np.array(rows, dtype=np.int64)
        conn = np.searchsorted(ids_s, a[:, 3 + nt:])
        if not np.array_equal(ids_s[conn], a[:, 3 + nt:]):
            raise SystemExit("element references a node id missing from"
                             " $Nodes -- corrupt mesh?")
        blocks.append({"etype": et, "ntags": nt, "tags": a[:, 3:3 + nt],
                       "conn": conn})
    return {"ids": ids_s, "xyz": xyz, "phys": phys, "blocks": blocks}


def write_msh(path, xyz, blocks, phys=()):
    """Write gmsh 2.2 ASCII with nodes renumbered 1..N.

    In:  path str; xyz (N, 3); blocks (read_msh layout); phys block lines
    Out: None (writes path)."""
    n_el = sum(len(b["conn"]) for b in blocks)
    with open(path, "w", buffering=1 << 22) as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        if len(phys):
            f.write("\n".join(phys) + "\n")
        f.write("$Nodes\n%d\n" % len(xyz))
        np.savetxt(f, np.column_stack([np.arange(1, len(xyz) + 1), xyz]),
                   fmt=["%d", "%.10f", "%.10f", "%.10f"])
        f.write("$EndNodes\n$Elements\n%d\n" % n_el)
        eid = 1
        for b in blocks:
            rows = np.column_stack([
                np.arange(eid, eid + len(b["conn"])),
                np.full(len(b["conn"]), b["etype"], np.int64),
                np.full(len(b["conn"]), b["tags"].shape[1], np.int64),
                b["tags"], b["conn"] + 1])
            np.savetxt(f, rows, fmt="%d")
            eid += len(b["conn"])
        f.write("$EndElements\n")


# ------------------------------------------------------------------ measures
def _tri_area(p):
    """(E,) areas of (E, 3, 3) triangles."""
    return 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def _tri_area_vec(p):
    """(E, 3) AREA VECTORS of (E, 3, 3) triangles (norm = the area,
    direction = the right-hand normal, so a flipped cell reverses)."""
    return 0.5 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def _tet_vol(p):
    """(E,) SIGNED volumes of (E, 4, 3) tets."""
    return np.einsum("ei,ei->e", p[:, 1] - p[:, 0],
                     np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6.0


_G2 = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))


def _tensor_measure(p, dim):
    """(E,) SIGNED volume (dim 3) or (E, 3) AREA VECTORS (dim 2) of
    (E, 4|8, 3) bi-/tri-linear cells, by 2^dim Gauss -- EXACT for these
    maps (det J is at most cubic per direction).  Signed on purpose: an
    inverted cell must be detectable, which an abs() would hide."""
    off = [(0, 0), (1, 0), (1, 1), (0, 1)] if dim == 2 else \
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    tot = np.zeros(len(p)) if dim == 3 else np.zeros((len(p), 3))
    for g in itertools.product(_G2, repeat=dim):
        J = np.zeros((len(p), 3, dim))
        for n, o in enumerate(off):
            s = [(2 * o[k] - 1) for k in range(dim)]
            for k in range(dim):
                w = 0.5 * s[k]
                for m in range(dim):
                    if m != k:
                        w = w * 0.5 * (1.0 + s[m] * g[m])
                J[:, :, k] += w * p[:, n, :]
        if dim == 2:
            tot += np.cross(J[:, :, 0], J[:, :, 1])
        else:
            tot += np.einsum("ei,ei->e", J[:, :, 0],
                             np.cross(J[:, :, 1], J[:, :, 2]))
    return tot


def cell_measures(xyz, block, normal=None):
    """Per-cell measure of ONE block (length / area / volume), SIGNED so
    an inverted cell is detectable.

    Volume cells sign by det J.  SURFACE cells have no intrinsic sign in
    3-D space, so they are signed along a reference normal -- `normal`
    when given, else the block's own summed area vector -- which makes
    the sign mean "consistently wound with the rest of the block".

    In:  xyz (N, 3); block dict; normal (3,) | None
    Out: (E,) float."""
    et, c = block["etype"], block["conn"]
    nc = _QUAD_NC.get(et, _TOPO.get(et, {}).get("nc"))
    p = xyz[c[:, :nc]]
    if _CELL_DIM[et] == 1:
        return np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    if _CELL_DIM[et] == 2:
        a = _tri_area_vec(p) if et in (2, 9) else _tensor_measure(p, 2)
        n = np.asarray(normal, float) if normal is not None \
            else a.sum(axis=0)
        nn = np.linalg.norm(n)
        if nn <= 0.0:                  # a closed/folded surface: no
            return np.linalg.norm(a, axis=1)   # reference -> magnitudes
        return a @ (n / nn)
    if et in (4, 11):
        return _tet_vol(p)
    if et == 5:
        return _tensor_measure(p, 3)
    raise ValueError("no measure for gmsh type %d" % et)


def mesh_measure(xyz, blocks):
    """Total measure of the HIGHEST-dimension blocks (the cells; lower-
    dimension blocks are boundary markers and are excluded).

    In:  xyz (N, 3); blocks list
    Out: float."""
    dim = max(_CELL_DIM[b["etype"]] for b in blocks)
    return float(sum(np.abs(cell_measures(xyz, b)).sum()
                     for b in blocks if _CELL_DIM[b["etype"]] == dim))


# ---------------------------------------------------------------- refinement
def _new_nodes(xyz, blocks):
    """Allocate every node the split adds, keyed by the sorted GLOBAL
    ids of its defining corners (see the module docstring).

    In:  xyz (N, 3); blocks list
    Out: (xyz2 (N+M, 3), {(block index, slot): (E,) new node ids})."""
    want = {}                              # arity -> list of (key, where)
    for bi, b in enumerate(blocks):
        topo = _TOPO[b["etype"]]
        for s, d in enumerate(topo["new"]):
            k = np.sort(b["conn"][:, list(d)], axis=1)
            want.setdefault(len(d), []).append((k, (bi, s)))
    grown, out, nxt = [xyz], {}, len(xyz)
    for arity in sorted(want):
        keys = np.vstack([k for k, _ in want[arity]])
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        grown.append(xyz[uniq].mean(axis=1))
        off = 0
        for k, where in want[arity]:
            out[where] = nxt + inv[off:off + len(k)]
            off += len(k)
        nxt += len(uniq)
    return np.vstack(grown), out


def _fix_tet_orientation(xyz, conn):
    """Swap the last two nodes of any NEGATIVE-volume tet.

    In:  xyz (N, 3); conn (E, 4)
    Out: (E, 4) with every signed volume > 0."""
    bad = _tet_vol(xyz[conn]) < 0.0
    if bad.any():
        conn = conn.copy()
        conn[bad] = conn[bad][:, [0, 1, 3, 2]]
    return conn


def _refine_once(xyz, blocks):
    """One uniform split of every block.

    In:  xyz (N, 3); blocks list
    Out: (xyz2, blocks2)."""
    xyz2, new = _new_nodes(xyz, blocks)
    out = []
    for bi, b in enumerate(blocks):
        et = b["etype"]
        topo = _TOPO[et]
        loc = [b["conn"][:, k] for k in range(topo["nc"])]
        loc += [new[(bi, s)] for s in range(len(topo["new"]))]
        L = np.column_stack(loc)
        if et == 4:                        # tet: shortest diagonal per cell
            d2 = np.column_stack([
                np.sum((xyz2[L[:, a]] - xyz2[L[:, c]]) ** 2, axis=1)
                for a, c in _TET_DIAGS])
            pick = np.argmin(d2, axis=1)
            kids = np.empty((len(L), 8, 4), np.int64)
            for t, tmpl in enumerate(_TET_CORNERS):
                kids[:, t] = L[:, list(tmpl)]
            for j, diag in enumerate(_TET_DIAGS):
                m = pick == j
                if not m.any():
                    continue
                for t, tmpl in enumerate(_TET_OCTA[diag]):
                    kids[m, 4 + t] = L[np.ix_(m, list(tmpl))]
            conn = kids.reshape(-1, 4)
            conn = _fix_tet_orientation(xyz2, conn)
            nkid = 8
        else:
            tmpls = topo["children"]
            conn = np.stack([L[:, list(t)] for t in tmpls],
                            axis=1).reshape(-1, topo["nc"])
            nkid = len(tmpls)
        out.append({"etype": et, "ntags": b["ntags"],
                    "tags": np.repeat(b["tags"], nkid, axis=0),
                    "conn": conn})
    return xyz2, out


def _edge_stats(xyz, blocks):
    """(h_min, h_mean, h_max) over the mesh's unique cell edges."""
    pcs = []
    for b in blocks:
        for d in _TOPO[b["etype"]]["new"]:
            if len(d) == 2:
                pcs.append(np.sort(b["conn"][:, list(d)], axis=1))
    uniq = np.unique(np.vstack(pcs), axis=0)
    L = np.linalg.norm(xyz[uniq[:, 0]] - xyz[uniq[:, 1]], axis=1)
    return float(L.min()), float(L.mean()), float(L.max())


def _straightness(xyz, blocks):
    """Worst midside deviation of a QUADRATIC mesh, relative to the
    local cell size -- 0 for a straight-sided mesh.

    In:  xyz (N, 3); blocks list (quadratic types only are inspected)
    Out: float."""
    worst = 0.0
    for b in blocks:
        et = b["etype"]
        if et not in _QUAD_EDGES:
            continue
        nc, c = _QUAD_NC[et], b["conn"]
        h = np.linalg.norm(xyz[c[:, 1]] - xyz[c[:, 0]], axis=1)
        h = np.maximum(h, 1e-300)
        for s, (a, bb) in enumerate(_QUAD_EDGES[et]):
            mid = 0.5 * (xyz[c[:, a]] + xyz[c[:, bb]])
            dev = np.linalg.norm(xyz[c[:, nc + s]] - mid, axis=1)
            worst = max(worst, float((dev / h).max()))
        if et == 10:                       # quad9 also stores a centre
            ctr = xyz[c[:, :4]].mean(axis=1)
            dev = np.linalg.norm(xyz[c[:, 8]] - ctr, axis=1)
            worst = max(worst, float((dev / h).max()))
    return worst


def msh_h_refine(msh_path, out=None, levels=None, target_h=None,
                 verbose=True):
    """Uniformly h-refine a gmsh 2.2 mesh (see the module docstring).

    In:  msh_path str; out str | None; levels int | None;
         target_h float | None (python API only); verbose bool
    Out: dict {msh, levels, n_nodes, n_elems, h_min, h_mean, h_max}."""
    t0 = time.perf_counter()
    m = read_msh(msh_path)
    xyz, blocks, phys = m["xyz"], m["blocks"], m["phys"]

    kinds = sorted({b["etype"] for b in blocks})
    unknown = [k for k in kinds if k not in _TOPO and k not in _QUAD_TO_LIN]
    if unknown:
        raise SystemExit(
            "gmsh type(s) %s unsupported -- h_refine splits line2, tri3,"
            " quad4, tet4, hex8 and their quadratic grades (tri6, quad9,"
            " tet10)" % ", ".join(_NAME.get(k, str(k)) for k in unknown))

    quad_kinds = [k for k in kinds if k in _QUAD_TO_LIN and k != 8]
    lin_kinds = [k for k in kinds if k in _TOPO and k != 1]
    if quad_kinds and lin_kinds:
        raise SystemExit(
            "%s mixes quadratic (%s) and linear (%s) cells -- refine one"
            " grade at a time"
            % (os.path.basename(msh_path),
               ", ".join(_NAME[k] for k in quad_kinds),
               ", ".join(_NAME[k] for k in lin_kinds)))
    quadratic = bool(quad_kinds)

    if quadratic:
        # THRESHOLD 1e-4, not machine epsilon: gmsh 2.2 files store
        # coordinates as decimal text (this project's writers use
        # %.8f-%.10f), so a perfectly straight midside still lands
        # ~1e-8 absolute off the chord -- on a small edge that is ~1e-6
        # RELATIVE, which a tighter gate rejects as "curved".  Genuine
        # curvature is O(1e-2) of the edge, two orders clear of this.
        dev = _straightness(xyz, blocks)
        if dev > 1e-4:
            raise SystemExit(
                "%s is a CURVED quadratic mesh (worst midside deviation"
                " %.3g of the local edge length).  This refiner splits"
                " the linear skeleton and re-inserts midsides, which"
                " would flatten that curvature -- refine the LINEAR mesh"
                " first, then --p_refine."
                % (os.path.basename(msh_path), dev))
        for b in blocks:                   # demote to corners
            b["conn"] = b["conn"][:, :_QUAD_NC[b["etype"]]]
            b["etype"] = _QUAD_TO_LIN[b["etype"]]
        xyz, blocks = _drop_orphans(xyz, blocks)

    hmn, hme, hmx = _edge_stats(xyz, blocks)
    ne = sum(len(b["conn"]) for b in blocks)
    if verbose:
        print("h_refine: %s -- %d nodes, %d elements (%s), edge h"
              " min/mean/max = %.6g / %.6g / %.6g"
              % (os.path.basename(msh_path), len(xyz), ne,
                 ", ".join(_NAME.get(k, str(k)) for k in kinds),
                 hmn, hme, hmx))

    if levels is not None and target_h is not None:
        raise ValueError("levels and target_h are mutually exclusive")
    if target_h is not None:
        levels = max(0, math.ceil(math.log2(hmx / float(target_h))))
        if verbose:
            print("h_refine: target max edge %.6g -> %d level%s"
                  % (float(target_h), levels, "" if levels == 1 else "s"))
        if levels == 0:
            if verbose:
                print("h_refine: already at/below the target -- nothing"
                      " written")
            return {"msh": msh_path, "levels": 0, "n_nodes": len(xyz),
                    "n_elems": ne, "h_min": hmn, "h_mean": hme,
                    "h_max": hmx}
    if levels is None:
        levels = 1
    if levels > 6:
        raise SystemExit(
            "h_refine: %d levels = %gx the cells -- past the 6-level cap;"
            " check the level count (a typo is the usual cause)"
            % (levels, float(2 ** (levels * max(_CELL_DIM[k]
                                                for k in kinds)))))

    for lv in range(levels):
        xyz, blocks = _refine_once(xyz, blocks)
        hmn, hme, hmx = _edge_stats(xyz, blocks)
        if verbose:
            print("h_refine: level %d -> %d nodes, %d elements, h ="
                  " %.6g / %.6g / %.6g"
                  % (lv + 1, len(xyz),
                     sum(len(b["conn"]) for b in blocks), hmn, hme, hmx))

    stem = os.path.splitext(msh_path)[0]
    mt = re.search(r"_h(\d+)$", stem)
    k0 = int(mt.group(1)) if mt else 1
    base = stem[:mt.start()] if mt else stem
    out = out or "%s_h%d.msh" % (base, k0 * 2 ** levels)

    if quadratic:
        lin = out.replace(".msh", "_lin_tmp.msh")
        write_msh(lin, xyz, blocks, phys)
        from opensg_solid.helper.make_linear_msh_to_quad import (
            linear_msh_to_quad)
        linear_msh_to_quad(lin, out=out, verbose=False)
        os.remove(lin)
        m2 = read_msh(out)
        xyz, blocks = m2["xyz"], m2["blocks"]
        if verbose:
            print("h_refine: re-promoted to %s"
                  % ", ".join(_NAME.get(b["etype"], "?") for b in blocks))
    else:
        write_msh(out, xyz, blocks, phys)

    n_el = sum(len(b["conn"]) for b in blocks)
    if verbose:
        print("h_refine: wrote %s  (%d nodes / %d elements, %.1f s)"
              % (out, len(xyz), n_el, time.perf_counter() - t0))
    return {"msh": out, "levels": levels, "n_nodes": len(xyz),
            "n_elems": n_el, "h_min": hmn, "h_mean": hme, "h_max": hmx}


def _drop_orphans(xyz, blocks):
    """Compact nodes no block references any more (the demote step).

    In:  xyz (N, 3); blocks list
    Out: (xyz2, blocks) with conn remapped."""
    used = np.unique(np.concatenate([b["conn"].ravel() for b in blocks]))
    remap = np.full(len(xyz), -1, np.int64)
    remap[used] = np.arange(len(used))
    for b in blocks:
        b["conn"] = remap[b["conn"]]
    return xyz[used], blocks


# ------------------------------------------------------------------ verifier
_FACETS = {2: [(0, 1), (1, 2), (2, 0)],
           3: [(0, 1), (1, 2), (2, 3), (3, 0)],
           4: [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
           5: [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
               (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]}


def verify_refinement(parent, child, rtol=1e-9, verbose=True):
    """Gate a refined mesh against the mesh it came from.

    Four independent checks, each of which a real refinement bug breaks:
      measure     total area/volume is CONSERVED (a wrong child template
                  or a dropped octahedron tet changes it)
      duplicates  no two nodes share a position (the key-vs-coordinate
                  dedup failure: coincident nodes = a split seam)
      orientation every cell has positive measure (a bad tet diagonal or
                  a mis-ordered hex template flips det J)
      conformity  every interior facet is shared by exactly 2 cells and
                  the BOUNDARY facet measure matches the parent's (a
                  hanging node shows up as an unmatched facet)

    In:  parent str | dict, child str | dict -- .msh paths (or read_msh
         dicts); rtol float; verbose bool
    Out: dict {ok, measure_parent, measure_child, n_dup, n_negative,
         n_unmatched, boundary_parent, boundary_child, problems [str]}."""
    P = read_msh(parent) if isinstance(parent, str) else parent
    C = read_msh(child) if isinstance(child, str) else child
    probs = []

    mp = mesh_measure(P["xyz"], P["blocks"])
    mc = mesh_measure(C["xyz"], C["blocks"])
    if abs(mc - mp) > rtol * max(abs(mp), 1e-300):
        probs.append("measure %.12g -> %.12g (rel %.3g)"
                     % (mp, mc, abs(mc - mp) / abs(mp)))

    q = np.round(C["xyz"] / max(1e-12, 1e-9 * np.ptp(C["xyz"])))
    n_dup = len(C["xyz"]) - len(np.unique(q, axis=0))
    if n_dup:
        probs.append("%d duplicate node position(s)" % n_dup)

    n_neg = 0
    dim = max(_CELL_DIM[b["etype"]] for b in C["blocks"])
    # surface cells are signed against the PARENT's normal, never the
    # child's own -- a wholesale flip would otherwise rebase the
    # reference and report itself as fine
    ref = None
    if dim == 2:
        av = [(_tri_area_vec(P["xyz"][b["conn"][:, :3]])
               if b["etype"] in (2, 9)
               else _tensor_measure(P["xyz"][b["conn"][:, :4]], 2))
              for b in P["blocks"] if _CELL_DIM[b["etype"]] == 2]
        if av:
            ref = np.vstack(av).sum(axis=0)
    for b in C["blocks"]:
        if _CELL_DIM[b["etype"]] != dim:
            continue
        n_neg += int((cell_measures(C["xyz"], b, normal=ref) <= 0).sum())
    if n_neg:
        probs.append("%d cell(s) with non-positive measure" % n_neg)

    def facets(M):
        f = []
        for b in M["blocks"]:
            et = b["etype"]
            if _CELL_DIM[et] != dim or et not in _FACETS:
                continue
            nc = _QUAD_NC.get(et, _TOPO[et]["nc"])
            for fc in _FACETS[et]:
                f.append(np.sort(b["conn"][:, list(fc)], axis=1))
        return np.vstack(f) if f else np.zeros((0, 2), np.int64)

    fc = facets(C)
    n_unmatched = 0
    if len(fc):
        uniq, cnt = np.unique(fc, axis=0, return_counts=True)
        n_unmatched = int((cnt > 2).sum())
        if n_unmatched:
            probs.append("%d facet(s) shared by MORE than 2 cells"
                         % n_unmatched)

    def bnd_measure(M):
        f = facets(M)
        if not len(f):
            return 0.0
        uniq, cnt = np.unique(f, axis=0, return_counts=True)
        b = uniq[cnt == 1]
        if b.shape[1] == 2:
            return float(np.linalg.norm(
                M["xyz"][b[:, 0]] - M["xyz"][b[:, 1]], axis=1).sum())
        if b.shape[1] == 3:
            return float(_tri_area(M["xyz"][b]).sum())
        p = M["xyz"][b]
        return float(_tri_area(p[:, [0, 1, 2]]).sum()
                     + _tri_area(p[:, [0, 2, 3]]).sum())

    bp, bc = bnd_measure(P), bnd_measure(C)
    if abs(bc - bp) > 1e-6 * max(abs(bp), 1e-300):
        probs.append("boundary measure %.12g -> %.12g" % (bp, bc))

    ok = not probs
    if verbose:
        print("verify_refinement: %s" % ("PASS" if ok else "FAIL"))
        print("  measure   %.12g -> %.12g" % (mp, mc))
        print("  boundary  %.12g -> %.12g" % (bp, bc))
        print("  duplicates %d | non-positive %d | over-shared facets %d"
              % (n_dup, n_neg, n_unmatched))
        for p in probs:
            print("  PROBLEM: %s" % p)
    return {"ok": ok, "measure_parent": mp, "measure_child": mc,
            "n_dup": n_dup, "n_negative": n_neg,
            "n_unmatched": n_unmatched, "boundary_parent": bp,
            "boundary_child": bc, "problems": probs}
