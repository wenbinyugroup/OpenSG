"""make_linear_msh_to_quad.py -- a LINEAR gmsh mesh -> its QUADRATIC
twin: every tet4 becomes a tet10 (and any boundary tri3 a tri6, line2
a line3) by CONFORMING midside insertion -- one shared node per unique
edge, at the exact midpoint, so shared faces (and matching PERIODIC
faces: midpoints of matching edges coincide) stay conforming.

    from opensg import helper
    helper.linear_msh_to_quad("core_L2.msh")     # -> core_L2_quad.msh

Layout guarantees (the exact-pairing doctrine): corner node ids,
element ids, element ORDER and every physical/elementary tag are
UNCHANGED -- element k of the quadratic mesh is the same parent
element k of the linear mesh; midside nodes are appended after the
existing ids (max id + 1 onward).  Midsides are exact straight-edge
midpoints (the parent is a linear mesh -- no curved geometry exists to
honor).

gmsh node-ordering (version 2.2 convention, the one the engine's
gmsh-to-basix permutation expects):
  tet10 (type 11): 4 corners + midsides on edges
                   (1,2) (2,3) (1,3) (1,4) (3,4) (2,4)
  tri6  (type 9):  3 corners + midsides (1,2) (2,3) (1,3)
  line3 (type 8):  2 corners + the midpoint

In:  msh_path str -- gmsh ASCII version 2.2 (the project writers'
     dialect; binary or 4.x files are refused with the re-export hint);
     out str | None -- output path (None -> <stem>_quad.msh)
Out: dict {msh, n_nodes, n_corner, n_midside, n_elems} -- and the
     written file.
"""
import os
import time

import numpy as np

# per gmsh type: (quadratic type, 0-based corner pairs in midside order)
_EDGE_SLOTS = {
    4: (11, ((0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3))),
    2: (9, ((0, 1), (1, 2), (0, 2))),
    1: (8, ((0, 1),)),
}
_NCORNER = {4: 4, 2: 3, 1: 2}


def linear_msh_to_quad(msh_path, out=None):
    """Convert a linear gmsh 2.2 mesh to its conforming quadratic twin.

    In:  msh_path str; out str | None (None -> <stem>_quad.msh)
    Out: dict {msh, n_nodes, n_corner, n_midside, n_elems}."""
    t0 = time.perf_counter()
    with open(msh_path) as f:
        lines = f.read().split("\n")
    i_fmt = lines.index("$MeshFormat")
    ver = lines[i_fmt + 1].split()
    if not ver[0].startswith("2"):
        raise SystemExit(
            "%s is gmsh format %s -- this converter reads the ASCII 2.2"
            " dialect the project writers emit.  Re-export first:"
            "  gmsh %s -format msh2 -save" % (msh_path, ver[0], msh_path))
    if len(ver) > 1 and ver[1] != "0":
        raise SystemExit("binary .msh not supported -- re-export ASCII"
                         " (gmsh ... -format msh2 -save)")
    phys = []                                     # $PhysicalNames verbatim
    if "$PhysicalNames" in lines:
        i_p = lines.index("$PhysicalNames")
        phys = lines[i_p:lines.index("$EndPhysicalNames") + 1]

    i_n = lines.index("$Nodes")
    nn = int(lines[i_n + 1])
    nd = np.loadtxt(lines[i_n + 2:i_n + 2 + nn], ndmin=2)
    ids, xyz = nd[:, 0].astype(np.int64), nd[:, 1:4]
    order = np.argsort(ids)
    ids_s, xyz_s = ids[order], xyz[order]

    def rows_of(idv):
        """Node ids (any order/gaps) -> rows into ids_s/xyz_s."""
        r = np.searchsorted(ids_s, idv)
        if not np.array_equal(ids_s[r], idv):
            raise SystemExit("element references a node id missing from"
                             " $Nodes -- corrupt mesh?")
        return r

    i_e = lines.index("$Elements")
    ne = int(lines[i_e + 1])
    raw = [ln.split() for ln in lines[i_e + 2:i_e + 2 + ne]]
    # group by (type, ntags) so each block vectorizes; keep global order
    blocks = {}
    for k, v in enumerate(raw):
        et, nt = int(v[1]), int(v[2])
        if et not in _EDGE_SLOTS:
            raise SystemExit(
                "element type %d is not linear tet/tri/line -- this"
                " converter upgrades tet4/tri3/line2 meshes only" % et)
        blocks.setdefault((et, nt), []).append(k)
    A = {key: np.array([raw[k] for k in kk], dtype=np.int64)
         for key, kk in blocks.items()}

    # one global unique-edge pool (tets + tris + lines share midsides)
    pair_chunks, chunk_of = [], {}
    for key, a in A.items():
        et, nt = key
        conn = a[:, 3 + nt:]
        for sl, (p, q) in enumerate(_EDGE_SLOTS[et][1]):
            pair_chunks.append(np.sort(
                np.column_stack([conn[:, p], conn[:, q]]), axis=1))
            chunk_of[(key, sl)] = len(pair_chunks) - 1
    pairs = np.vstack(pair_chunks)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    mid_id0 = int(ids.max()) + 1
    mid_ids = mid_id0 + np.arange(len(uniq), dtype=np.int64)
    mid_xyz = 0.5 * (xyz_s[rows_of(uniq[:, 0])]
                     + xyz_s[rows_of(uniq[:, 1])])
    # slice inv back per chunk
    offs = np.cumsum([0] + [len(c) for c in pair_chunks])
    inv_of = {k: inv[offs[j]:offs[j + 1]]
              for k, j in chunk_of.items()}

    print("linear_msh_to_quad: %s -- %d nodes, %d elements, %d unique"
          " edges -> +%d midside nodes"
          % (os.path.basename(msh_path), nn, ne, len(uniq), len(uniq)))

    out = out or os.path.splitext(msh_path)[0] + "_quad.msh"
    with open(out, "w", buffering=1 << 22) as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        if phys:
            f.write("\n".join(phys) + "\n")
        f.write("$Nodes\n%d\n" % (nn + len(uniq)))
        np.savetxt(f, np.column_stack([ids, xyz]),
                   fmt=["%d", "%.8f", "%.8f", "%.8f"])
        np.savetxt(f, np.column_stack([mid_ids, mid_xyz]),
                   fmt=["%d", "%.8f", "%.8f", "%.8f"])
        f.write("$EndNodes\n$Elements\n%d\n" % ne)
        # rebuild every element row (quadratic type + appended midsides)
        # and emit in the ORIGINAL element order; the common one-block
        # case (all tet4, same tag count) streams through savetxt
        fulls = {}
        for key, a in A.items():
            et, nt = key
            qt, slots = _EDGE_SLOTS[et]
            head = a[:, :3 + nt].copy()
            head[:, 1] = qt
            mids = np.column_stack(
                [mid_ids[inv_of[(key, sl)]] for sl in range(len(slots))])
            fulls[key] = np.hstack([head, a[:, 3 + nt:], mids])
        if len(fulls) == 1:
            np.savetxt(f, next(iter(fulls.values())), fmt="%d")
        else:
            out_rows = [None] * ne
            for key, full in fulls.items():
                for r, k in enumerate(blocks[key]):
                    out_rows[k] = full[r]
            for row in out_rows:
                f.write(" ".join(str(int(x)) for x in row) + "\n")
        f.write("$EndElements\n")
    print("linear_msh_to_quad: wrote %s  (%d nodes / %d elements,"
          " %.1f s)" % (out, nn + len(uniq), ne,
                        time.perf_counter() - t0))
    return {"msh": out, "n_nodes": nn + len(uniq), "n_corner": nn,
            "n_midside": len(uniq), "n_elems": ne}
