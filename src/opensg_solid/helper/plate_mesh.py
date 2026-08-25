"""plate_mesh.py -- a periodic SG -> the full 3-D plate mesh (.msh).

The plate the macroscopic model homogenizes is rebuilt EXPLICITLY, the
way the HC_pm45 benchmark's 3-D deck relates to its 2-D SG:

  3-D SG   the unit cell is TILED nx x ny times in the plate plane,
           nx = round(a / Px), ny = round(b / Py) (the achieved a, b are
           printed -- whole cells only).  GATE: the node sets of each
           opposite in-plane face pair must match (a periodic,
           tile-conforming mesh); refused otherwise, because the tiles
           would not weld into one conforming solid.
  2-D SG   (width x thickness quads, the HC_pm45 kind) is tiled across
           the width, ny = round(b / P), and EXTRUDED along the span
           into hex8 layers, n_span = round(a / dx) with dx defaulting
           to the SG's median in-plane edge length.

Welding is vectorized: all tile nodes are keyed on coordinates rounded
to `tol` and np.unique(return_inverse) renumbers everything in one
pass.  Output is gmsh 2.2 with the material id as the physical AND
elementary tag -- the same dialect io.sc_to_yaml.write_msh emits and
plate_inp reads back.

Plate axes: x = span (0..a), y = width (0..b), z = thickness, centred
at z = 0.

In:  the canonical SG yaml (nodes / cells / mat_id / materials)
Out: <yaml stem>_plate.msh (or `out`)
"""
import os
import time

import numpy as np


def _weld(nodes, cells, tol):
    """Merge coincident nodes.  In: nodes (N, 3), cells (E, k) int,
    tol float.  Out: (nodes2, cells2) with first-occurrence coords."""
    key = np.round(nodes / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    return nodes[first], inv[cells]


def _face_match(nd, ax, tol):
    """Do the two faces normal to axis `ax` carry the same node set?
    In: nd (N, 3); ax int; tol float.  Out: bool."""
    lo = nd[np.abs(nd[:, ax] - nd[:, ax].min()) < tol]
    hi = nd[np.abs(nd[:, ax] - nd[:, ax].max()) < tol]
    if len(lo) != len(hi):
        return False
    oth = [i for i in range(3) if i != ax]
    A = np.round(lo[:, oth] / tol).astype(np.int64)
    B = np.round(hi[:, oth] / tol).astype(np.int64)
    return np.array_equal(A[np.lexsort(A.T)], B[np.lexsort(B.T)])


def _write_msh(path, nodes, cells, mats):
    """gmsh 2.2, material id as physical/elementary tag; tet4 -> type 4,
    hex8 -> type 5.  Buffered, so multi-million-element meshes stream.

    In: path str; nodes (N, 3); cells (E, 4|8) 0-based; mats (E,) int
    Out: None (writes path)."""
    et = {4: 4, 8: 5}[cells.shape[1]]
    with open(path, "w", buffering=1 << 22) as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n"
                % len(nodes))
        ids = np.arange(1, len(nodes) + 1)
        np.savetxt(f, np.column_stack([ids, nodes]),
                   fmt=["%d", "%.8f", "%.8f", "%.8f"])
        f.write("$EndNodes\n$Elements\n%d\n" % len(cells))
        head = np.column_stack([np.arange(1, len(cells) + 1),
                                np.full(len(cells), et),
                                np.full(len(cells), 2), mats, mats])
        np.savetxt(f, np.hstack([head, cells + 1]), fmt="%d")
        f.write("$EndElements\n")


def plate_mesh(yaml_path, a, b, out=None, dx=None, tol=1e-6):
    """The tiled/extruded plate mesh of a periodic SG.

    In:  yaml_path str -- the canonical SG yaml; a, b float -- the
         requested plate span (x) and width (y); rounded to WHOLE cells
         (2-D SG: the span direction uses dx layers instead); out str |
         None -- the .msh (None -> <stem>_plate.msh); dx float | None --
         2-D SG only, the spanwise element length (None -> the SG's
         median in-plane edge); tol float -- weld tolerance [mm]
    Out: dict {msh, nx, ny, a, b, n_nodes, n_elems, thickness} -- what
         was actually built."""
    t0 = time.perf_counter()
    # load_sg_input reads BOTH solid yaml spellings (the canonical
    # nodes/cells/mat_id and the mesh dialect msh_to_yaml emits, whose
    # rows are space-separated strings) and caches the parse in the
    # <base>_sg.npz sidecar
    from opensg_solid.sg_mesh import load_sg_input
    d = load_sg_input(yaml_path)
    nd = np.asarray(d["nodes"], float)[:, :3]
    cells = np.asarray(d["cells"], int)
    mats = np.asarray(d["mat_id"], int)
    span = nd.max(axis=0) - nd.min(axis=0)
    dim = 3 if span[2] > tol else 2
    base = os.path.splitext(yaml_path)[0]
    out = out or base + "_plate.msh"

    if dim == 3:
        if cells.shape[1] not in (4, 8):
            raise SystemExit("3-D tiling supports tet4/hex8 cells, got"
                             " %d-node" % cells.shape[1])
        for ax, nm in ((0, "x"), (1, "y")):
            if not _face_match(nd, ax, tol):
                raise SystemExit(
                    "the SG mesh is NOT tile-conforming in %s: the two"
                    " opposite faces carry different node sets, so the"
                    " tiles cannot weld into one conforming plate."
                    "  Re-mesh the SG periodic (matching face meshes)"
                    " or use a tie-based assembly instead." % nm)
        Px, Py, h = span
        nx, ny = max(1, round(a / Px)), max(1, round(b / Py))
        print("plate_mesh: 3-D SG %.4g x %.4g x %.4g -> tiling %d x %d"
              " cells (achieved a = %.4g, b = %.4g, requested %.4g,"
              " %.4g)" % (Px, Py, h, nx, ny, nx * Px, ny * Py, a, b))
        org = nd - nd.min(axis=0)                    # cell at the origin
        org[:, 2] -= h / 2.0                         # z centred
        NT = nx * ny
        offs = np.array([[i * Px, j * Py, 0.0]
                         for i in range(nx) for j in range(ny)])
        nodes_all = (org[None, :, :] + offs[:, None, :]).reshape(-1, 3)
        cells_all = (cells[None, :, :]
                     + (np.arange(NT) * len(nd))[:, None, None]
                     ).reshape(-1, cells.shape[1])
        mats_all = np.tile(mats, NT)
        a_out, b_out = nx * Px, ny * Py
    else:
        if cells.shape[1] != 4:
            raise SystemExit("2-D extrusion supports quad4 cells")
        P, h = span[0], span[1]
        ny = max(1, round(b / P))
        if dx is None:
            e2 = np.linalg.norm(nd[cells] - nd[np.roll(cells, -1, 1)],
                                axis=2)
            dx = float(np.median(e2))
        nsp = max(1, round(a / dx))
        print("plate_mesh: 2-D SG %.4g wide x %.4g thick -> %d cells"
              " across, %d spanwise layers of %.4g (achieved a = %.4g,"
              " b = %.4g)" % (P, h, ny, nsp, a / nsp, a, ny * P))
        # a positive-Jacobian hex needs its bottom quad CCW as seen from
        # +x (the extrusion direction): flip any CW quad first
        xq, yq = nd[cells][:, :, 0], nd[cells][:, :, 1]
        area2 = (xq * np.roll(yq, -1, 1)
                 - np.roll(xq, -1, 1) * yq).sum(axis=1)
        flip = area2 < 0
        if flip.any():
            cells = cells.copy()
            cells[flip] = cells[flip][:, ::-1]
            print("plate_mesh: flipped %d CW quads for a positive"
                  " extrusion Jacobian" % int(flip.sum()))
        w0 = nd[:, 0] - nd[:, 0].min()               # width from 0
        z0 = nd[:, 1] - (nd[:, 1].min() + nd[:, 1].max()) / 2.0
        n1 = len(nd)
        plane = np.vstack([np.column_stack(
            [np.zeros(n1), w0 + j * P, z0]) for j in range(ny)])
        npl = len(plane)                             # one spanwise plane
        xs = np.linspace(0.0, a, nsp + 1)
        nodes_all = np.vstack([plane + np.array([x, 0, 0])
                               for x in xs])
        # hex8 = the quad extruded between planes k and k+1: nodes 1-4 =
        # the quad at x_k, 5-8 = the same quad at x_k+1
        q = np.vstack([cells + j * n1 for j in range(ny)])
        mats_q = np.tile(mats, ny)
        cells_all = np.vstack([
            np.hstack([q + k * npl, q + (k + 1) * npl])
            for k in range(nsp)])
        mats_all = np.tile(mats_q, nsp)
        nx, a_out, b_out = nsp, a, ny * P

    nodes2, cells2 = _weld(nodes_all, cells_all, tol)
    print("plate_mesh: welded %d -> %d nodes; %d elements"
          % (len(nodes_all), len(nodes2), len(cells2)))
    _write_msh(out, nodes2, cells2, mats_all)
    print("plate_mesh: wrote %s  (%.1f s)"
          % (out, time.perf_counter() - t0))
    return {"msh": out, "nx": nx, "ny": ny, "a": a_out, "b": b_out,
            "n_nodes": len(nodes2), "n_elems": len(cells2),
            "thickness": float(span[2] if dim == 3 else span[1])}
