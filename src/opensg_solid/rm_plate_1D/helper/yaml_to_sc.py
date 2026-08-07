"""yaml_to_sc.py -- OpenSG 2-D/3-D SG yaml -> SwiftComp .sc, the inverse of
sc_to_yaml.py.

Needed because the original single-script MSG driver (JAX_BICGoptimize) reads a
.sc, while the mesh pipelines (PreVABS -> OpenSG_io, windIO -> OpenSG_io) emit
the OpenSG yaml dialect.  This lets the same mesh feed both architectures.

The .sc anatomy written here is the one sc_to_yaml.py / generate_msh_from_sc
read back:

    <3 short header lines, each with < 5 fields>
    dim  n_nodes  n_elems  n_mats  0  0        <- the META line (>= 5 integers)
    id  x  y [z]                               x n_nodes
    id  mat_id  conn...  0                     x n_elems (zeros = padding)
    per material:
        mat_id  mat_type  n_temp
        0  density
        E1 E2 E3 / G12 G13 G23 / nu12 nu13 nu23

LIMITATION worth knowing before you use it: a .sc carries ONE material id per
element, and the drivers that read it apply ONE fibre angle per material.  A
yaml's per-element `elementOrientations` therefore CANNOT be represented -- if
the SG relies on per-element material frames, the .sc route will not reproduce
it.  Emit distinct material ids per orientation group, or stay in the yaml
route and pass the frames as `elem_rotation`.
"""
import os

import numpy as np
import yaml as _yaml


def _row(r):
    return [float(v) for v in
            " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]


def read_opensg_yaml(path):
    """Parse the OpenSG SG yaml -> (nodes (N,3), cells (E,k) 1-based,
    mat_id (E,) 0-based, materials list)."""
    d = _yaml.safe_load(open(path))
    nodes = np.array([_row(r) for r in d["nodes"]], float)
    cells = [[int(v) for v in _row(r)] for r in d["elements"]]
    ne = len(cells)
    mat_id = np.zeros(ne, int)
    for k, s in enumerate(d.get("sets", {}).get("element", [])):
        mat_id[np.array(s["labels"], int) - 1] = k
    return nodes, cells, mat_id, d["materials"]


def convert(yaml_path, out_path=None, dim=2):
    """Write <yaml_path> as a SwiftComp .sc.  Returns the .sc path.

    dim = 2 keeps the in-plane coordinates (a 2-D SG in the y2-y3 plane),
    dim = 3 keeps all three."""
    if out_path is None:
        out_path = os.path.splitext(yaml_path)[0] + ".sc"
    nodes, cells, mat_id, materials = read_opensg_yaml(yaml_path)
    xyz = nodes[:, :dim]
    nn, ne = len(xyz), len(cells)
    nmat = len(materials)

    lines = ["1", "0 0", "0 0 0 0", "",
             "%d %d %d %d 0 0" % (dim, nn, ne, nmat), ""]
    for i, p in enumerate(xyz):
        lines.append("%d " % (i+1) + " ".join("%.12f" % v for v in p))
    lines.append("")
    for e, c in enumerate(cells):
        lines.append("%d %d " % (e+1, mat_id[e]+1) + " ".join(str(n) for n in c)
                     + " 0")
    lines.append("")
    for k, m in enumerate(materials):
        E = m["E"] if "E" in m else m["elastic"]["E"]
        G = m["G"] if "G" in m else m["elastic"]["G"]
        nu = m["nu"] if "nu" in m else m["elastic"]["nu"]
        rho = m.get("rho", m.get("density", 0.0))
        lines.append("%d 0 1" % (k+1))
        lines.append("0 %g" % float(rho))
        lines.append(" ".join("%g" % float(v) for v in E))
        lines.append(" ".join("%g" % float(v) for v in G))
        lines.append(" ".join("%g" % float(v) for v in nu))
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("yaml_to_sc: %dD SG, %d nodes, %d cells, %d materials -> %s"
          % (dim, nn, ne, nmat, out_path))
    return out_path
