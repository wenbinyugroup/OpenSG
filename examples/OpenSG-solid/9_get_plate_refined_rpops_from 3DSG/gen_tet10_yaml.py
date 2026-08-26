"""gen_tet10_yaml.py -- the CMC sandwich SG upgraded to tet10: conforming
midside insertion on the tet4 SG, periodicity gates verified, canonical
solid yaml written.

In : preovios_try.yaml (via load_sg_input -> the fresh preovios_try_sg.npz
     sidecar, i.e. EXACTLY the sc dict the tet4 .out was computed from:
     53,152 nodes / 196,739 tet4, materials 1 core iso, 2/3 ply -+45).
Out: preovios_try_tet10.yaml (canonical dialect: n_model 2 / refined 1 /
     msg solid header, nodes/cells/mat_id/materials) -- one shared midside
     node per unique edge at the exact midpoint, midsides appended after
     the 53,152 parent nodes, parent tet ids preserved in order, midside
     slots in the GMSH edge order (12, 23, 13, 14, 34, 24) that
     sg_homo._to_basix_order expects; console log with the periodicity-gate
     measurements (x and y face pairs, corner and midside populations,
     max mismatch vs the engine's atol=1e-6).
"""
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from opensg_solid.sg_mesh import load_sg_input
from opensg_solid.io.sc_to_yaml import write_yaml_file

print("start wall-clock:", time.strftime("%Y-%m-%d %H:%M:%S"))

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "preovios_try.yaml")
DST = os.path.join(HERE, "preovios_try_tet10.yaml")

sc = load_sg_input(SRC)
nodes = np.asarray(sc["nodes"], float)
cells4 = np.asarray(sc["cells"], np.int64)
assert nodes.shape[0] == 53152, nodes.shape
assert cells4.shape == (196739, 4), cells4.shape
print("tet4 SG loaded: %d nodes, %d cells, dim %d, scale %g"
      % (nodes.shape[0], cells4.shape[0], sc["dim"], sc["scale"]))
for mid in sorted(sc["materials"]):
    print("  material %d: %s" % (mid, sc["materials"][mid]))
print("  mat_id counts:", np.bincount(np.asarray(sc["mat_id"], int))[1:])

# ---- conforming midside insertion (one shared node per unique edge,
# exact midpoints; gmsh tet10 midside slot order)
GMSH_EDGES = np.array([(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)])
V = nodes.shape[0]
ep = cells4[:, GMSH_EDGES]                       # (E, 6, 2) node pairs
ep_sorted = np.sort(ep.reshape(-1, 2), axis=1)   # canonical edge keys
uniq, inv = np.unique(ep_sorted, axis=0, return_inverse=True)
mid_pts = 0.5 * (nodes[uniq[:, 0]] + nodes[uniq[:, 1]])
cells10 = np.hstack([cells4, (V + inv).reshape(-1, 6)])
nodes10 = np.vstack([nodes, mid_pts])
print("unique edges: %d -> tet10 SG: %d nodes, %d cells"
      % (len(uniq), nodes10.shape[0], cells10.shape[0]))

# ---- periodicity gates: the engine ties x-max -> x-min and y-max ->
# y-min node-to-node (n_model=2, n_sg=3; fe_jax.setup.periodic_map,
# HARD ValueError beyond atol) -- every slave INCLUDING the new
# midsides must find a master within 1e-6 after the box shift
ATOL = 1e-6
lo, hi = nodes10.min(axis=0), nodes10.max(axis=0)
span = hi - lo
print("bbox spans: x %.6g  y %.6g  z %.6g" % tuple(span))
gate_ok = True
for ax, ax_name in ((0, "x"), (1, "y")):
    masters = np.where(np.isclose(nodes10[:, ax], lo[ax], atol=ATOL))[0]
    slaves = np.where(np.isclose(nodes10[:, ax], hi[ax], atol=ATOL))[0]
    shift = np.zeros(3)
    shift[ax] = span[ax]
    d, _ = cKDTree(nodes10[masters]).query(nodes10[slaves] - shift)
    is_mid = slaves >= V
    d_c = d[~is_mid].max() if (~is_mid).any() else 0.0
    d_m = d[is_mid].max() if is_mid.any() else 0.0
    ok = d.max() < ATOL
    gate_ok &= ok
    print("%s gate: %d masters / %d slaves (%d corner + %d midside)  "
          "max mismatch corner %.3e midside %.3e  -> %s"
          % (ax_name, len(masters), len(slaves), int((~is_mid).sum()),
             int(is_mid.sum()), d_c, d_m, "PASS" if ok else "FAIL"))
if not gate_ok:
    raise SystemExit("periodicity gate FAILED -- tet10 yaml NOT written"
                     " (opposite-face triangulations do not mirror)")

# ---- write the canonical solid yaml with the engine's own writer
sc10 = {"dim": 3, "nodes": nodes10, "cells": [list(c) for c in cells10],
        "mat_id": np.asarray(sc["mat_id"], int),
        "materials": sc["materials"], "scale": sc["scale"]}
t0 = time.perf_counter()
write_yaml_file(sc10, DST, n_model=2, refined=1)
print("wrote %s (%.1f MB) in %.1f s"
      % (DST, os.path.getsize(DST) / 1e6, time.perf_counter() - t0))
print("end wall-clock:", time.strftime("%Y-%m-%d %H:%M:%S"))
