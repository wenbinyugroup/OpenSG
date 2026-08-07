"""Timoshenko 6x6 from the 2-D SOLID SG (OpenSG-solid Beam_solid KKT engine).

Input: square_tube_2Dsolid_t1only.yaml -- the OpenSG-conforming 2-D solid yaml
(see Rules/orientation_e1_out_of_plane.md): e1 = beam axis OUT OF PLANE,
e2 = wall tangent, e3 = ply normal, exactly as in the 1-D shell yaml.  The -45
deg fibre rotation is supplied through `angles`, not baked into e1.

The solid engine (sg_homo.plate_homo_2d) reads its own yaml dialect
{dim, nodes, cells, mat_id, materials}, so the OpenSG_io yaml is translated
here, and the per-element frame is handed over as `elem_rotation` (E, 9).

Run (from this folder):  python beam_solid_2dyaml.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import yaml as _yaml

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

############### User Input #################################
n_model = 1                    # 1: Beam; 2: Plate; 3: 3D elastic
YAML = "../square_tube_2Dsolid_t1only.yaml"
# m45 ply, SI (Pa): E1 E2 E3 G12 G13 G23 nu12 nu13 nu23
material_param = jnp.array([
    (142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42)])
angles = jnp.array([-45.0])    # fibre angle per material (deg)
OUT = "square_tube_beam_solid.out"
############################################################

d = _yaml.safe_load(open(YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)          # x y z(axial)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
xy = nd[:, :2]
ne = len(tri)

mat_id = np.ones(ne, int)          # sc["mat_id"] is 1-BASED
for k, s in enumerate(d["sets"]["element"]):
    mat_id[np.array(s["labels"], int) - 1] = k + 1

assert np.allclose(np.abs(ori[:, 2]), 1.0), \
    "e1 must be out of plane (Rules/orientation_e1_out_of_plane.md)"
assert np.allclose(ori[:, 8], 0.0), "e3 must be in plane"
print("%s: %d nodes, %d elements, e1 out of plane (|e1_z|=%.6f)"
      % (YAML, len(nd), ne, np.abs(ori[:, 2]).mean()))

# --- the engine's own dialect ------------------------------------------------
sc = {"dim": 2,
      "nodes": xy,
      "cells": [list(c) for c in tri],
      "mat_id": mat_id,
      "materials": {0: {"name": d["materials"][0]["name"]}},
      "scale": 1.0}

t0 = time.perf_counter()
r = plate_homo_2d(sc, material_param=material_param, angles=angles,
                  n_model=n_model, elem_rotation=elem_rotation_from_yaml(ori), plot=False)
C6 = np.asarray(r["C_eff"])
dt = time.perf_counter() - t0

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("solid 2-D SG (omega=%.6g)  [%.1f s]" % (r["omega"], dt))
print("beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):")
for i in range(6):
    print("  " + " ".join("%14.6e" % C6[i, j] for j in range(6)))
print("diagonal: " + "  ".join("%s=%.6g" % (LBL[i], C6[i, i]) for i in range(6)))
np.savetxt(OUT, C6, fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2 kappa3]"
                  " (Beam_solid KKT engine) from the 2-D solid SG %s,"
                  " fibre %+.0f deg via angles" % (YAML, float(angles[0])))
print("wrote %s" % OUT)

# --- orientation figure: arrows inside the solid, same style as the 1-D one --
cen = np.stack([xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]]).mean(0)
step = max(1, ne//160)
s = slice(None, None, step)
sc_a = 0.05
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.triplot(xy[:, 0], xy[:, 1], tri, color="0.75", lw=0.25)
ax.quiver(cen[s, 0], cen[s, 1], ori[s, 3], ori[s, 4], color="g",
          scale=1/sc_a, scale_units="xy", width=0.005, label="e2 (tangent)")
ax.quiver(cen[s, 0], cen[s, 1], ori[s, 6], ori[s, 7], color="b",
          scale=1/sc_a, scale_units="xy", width=0.005, label="e3 (normal)")
ax.set_aspect("equal"); ax.set_xlabel("y2"); ax.set_ylabel("y3")
ax.legend(loc="center", fontsize=9)
fig.tight_layout(); fig.savefig("square_tube_2Dsolid_orient.png", dpi=200)
print("wrote square_tube_2Dsolid_orient.png  (e1 is out of plane, not drawn)")
