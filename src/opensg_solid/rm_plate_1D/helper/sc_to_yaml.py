"""sc_to_yaml.py -- SwiftComp .sc -> yaml (+ gmsh .msh) for ANY structure
gene, so the 2D-SG plate examples never depend on a private parser.

The .sc anatomy this reads (SwiftComp MSG-native input):

    <model header lines>            1-3 short lines (submodel flags etc.)
    dim  n_nodes  n_elems  n_mats  ...   <- the META line (>= 5 integers)
    <n_nodes>  id  x [y [z]]
    <n_elems>  id  mat_id  conn... (zero-padded; the zeros are NOT nodes)
    per material:
        mat_id  mat_type  n
        <aux line, e.g. "0 0">
        mat_type 0: E  nu                     (isotropic)
        mat_type 1: E1 E2 E3 / G12 G13 G23 / v12 v13 v23  (orthotropic)
        mat_type 2: 21 constants, upper-triangular 6x6 C over SIX lines
                    (these are typically PRE-ROTATED ply stiffnesses)
    <trailing scale line>           (volume/thickness normalization)

Outputs:
    <base>.yaml   dim, nodes, cells (0-based), mat_id per cell, materials,
                  scale -- the interchange the rm_plate stack can read
    <base>.msh    gmsh v2.2 (the element-type mapping of fe_jax/sc_to_msh:
                  1D 2/3/4/5-node intervals, 2D tri/quad, 3D tet4/tet10)

Run:  python sc_to_yaml.py input.sc [out_base]
API:  sc = convert("input.sc", "out_base"); sc["dim"], sc["mat_id"], ...
"""
import os
import sys

import numpy as np
import yaml


def read_sc(path):
    """Parse a .sc file.  Returns dict: dim, nodes (n,3) float, cells
    (list of 0-based connectivity lists), mat_id (n_elems,) int (1-based
    as in the file), materials {id: {type, aux, props, C?}}, scale."""
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    meta_idx = -1
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) >= 5:
            meta_idx = i
            dim, n_nodes, n_elems, n_mats = (int(parts[0]), int(parts[1]),
                                             int(parts[2]), int(parts[3]))
            break
    if meta_idx == -1:
        raise ValueError("no META line (>=5 integers) in %s" % path)

    nodes = np.zeros((n_nodes, 3))
    for ln in lines[meta_idx + 1: meta_idx + 1 + n_nodes]:
        p = ln.split()
        i = int(p[0]) - 1
        for d in range(dim):
            nodes[i, d] = float(p[1 + d])

    cells, mat_id = [], []
    e0 = meta_idx + 1 + n_nodes
    for ln in lines[e0: e0 + n_elems]:
        p = ln.split()
        mat_id.append(int(p[1]))
        conn = [int(n) - 1 for n in p[2:] if n != "0"]
        if dim == 3 and len(conn) == 10:
            # the tet10 slot convention of fe_jax/sc_to_msh: 4 corners,
            # slot 5 = 0, then the 6 midsides in slots 7-12
            raw = p[2:]
            conn = [int(n) - 1 for n in (raw[:4] + raw[6:12])]
        elif dim == 3 and len(conn) not in (4, 8):
            # e.g. SW_2UC_45.sc: every record has 9 nonzero slots -- not a
            # tet4/hex8/tet10 under any known SwiftComp slicing.  Refuse
            # loudly instead of emitting phantom node 0 / -1 entries.
            raise ValueError(
                "%s: 3-D element %s has %d nonzero connectivity slots -- "
                "not tet4/hex8/tet10; confirm this file's SwiftComp slot "
                "convention before converting" % (path, p[0], len(conn)))
        cells.append(conn)

    materials, k = {}, e0 + n_elems
    for _ in range(n_mats):
        head = lines[k].split()
        mid, mtype = int(head[0]), int(head[1])
        aux = lines[k + 1].split()
        m = {"type": mtype, "aux": [float(v) for v in aux]}
        if mtype == 0:
            p = lines[k + 2].replace("E+", "e+").replace("E-", "e-").split()
            m["E"], m["nu"] = float(p[0]), float(p[1])
            k += 3
        elif mtype == 1:
            vals = []
            j = k + 2
            while len(vals) < 9:
                vals += [float(v) for v in lines[j].split()]
                j += 1
            m["engineering"] = vals            # E1 E2 E3 G12 G13 G23 v12 v13 v23
            k = j
        else:                                   # type 2: upper-tri 6x6 C
            C = np.zeros((6, 6))
            j = k + 2
            for r in range(6):
                row = [float(v) for v in lines[j].split()]
                C[r, r:r + len(row)] = row
                C[r:r + len(row), r] = row
                j += 1
            m["C"] = C.tolist()
            k = j
        materials[mid] = m
    scale = float(lines[k].split()[0]) if k < len(lines) else 1.0

    return {"dim": dim, "nodes": nodes, "cells": cells,
            "mat_id": np.array(mat_id, int), "materials": materials,
            "scale": scale}


_GMSH_1D = {2: 1, 3: 8, 4: 26, 5: 27}
_GMSH_2D = {3: 2, 4: 3}
_GMSH_3D = {4: 4, 10: 11}


def write_msh(sc, path):
    """gmsh v2.2 with the material id as both physical/elementary tags
    (what the plate engine reads back as cell_domain_ids)."""
    dim, nodes = sc["dim"], sc["nodes"]
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n%d\n"
                % len(nodes))
        for i, x in enumerate(nodes):
            f.write("%d %.8f %.8f %.8f\n" % (i + 1, x[0], x[1], x[2]))
        f.write("$EndNodes\n$Elements\n%d\n" % len(sc["cells"]))
        tmap = _GMSH_1D if dim == 1 else _GMSH_2D if dim == 2 else _GMSH_3D
        for e, (conn, mid) in enumerate(zip(sc["cells"], sc["mat_id"])):
            et = tmap.get(len(conn))
            if et is None:
                raise ValueError("element %d: %d nodes unsupported for %dD"
                                 % (e + 1, len(conn), dim))
            f.write("%d %d 2 %d %d %s\n"
                    % (e + 1, et, mid, mid,
                       " ".join(str(n + 1) for n in conn)))
        f.write("$EndElements\n")


def write_yaml_file(sc, path):
    out = {"dim": int(sc["dim"]),
           "scale": float(sc["scale"]),
           "nodes": [[float(v) for v in row] for row in sc["nodes"]],
           "cells": [[int(n) for n in c] for c in sc["cells"]],
           "mat_id": [int(m) for m in sc["mat_id"]],
           "materials": {int(k): {kk: vv for kk, vv in m.items()}
                         for k, m in sc["materials"].items()}}
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)


def convert(sc_path, out_base=None):
    """Read .sc, write <base>.yaml + <base>.msh, return the parsed dict."""
    if out_base is None:
        out_base = os.path.splitext(sc_path)[0]
    sc = read_sc(sc_path)
    write_yaml_file(sc, out_base + ".yaml")
    write_msh(sc, out_base + ".msh")
    print("sc_to_yaml: %dD SG, %d nodes, %d cells, %d materials -> %s.yaml/.msh"
          % (sc["dim"], len(sc["nodes"]), len(sc["cells"]),
             len(sc["materials"]), out_base))
    return sc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python sc_to_yaml.py input.sc [out_base]")
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
