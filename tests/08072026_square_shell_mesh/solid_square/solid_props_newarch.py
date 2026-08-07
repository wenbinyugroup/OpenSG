"""Equivalent 3-D solid properties of the square-tube 2-D SG -- OpenSG new
architecture (opensg_solid.sg_homo, n_model = 3).

Reads ../square_tube_2Dsolid_t1only.yaml (e1 = beam axis out of plane per
Rules/orientation_e1_out_of_plane.md; the -45 deg fibre is applied through
`angles`) and writes the 6x6 C_eff plus the 9 engineering constants.

Voigt order [e11 e22 e33 2e23 2e13 2e12], axis 1 = prismatic direction.

Run (from this folder):  python solid_props_newarch.py
"""
import time

import numpy as np
import jax.numpy as jnp
import yaml as _yaml

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

############### User Input #################################
n_model = 3                    # 1: Beam; 2: Plate; 3: 3D elastic
YAML = "../square_tube_2Dsolid_t1only.yaml"
# m45 ply, SI (Pa): E1 E2 E3 G12 G13 G23 nu12 nu13 nu23
material_param = jnp.array([
    (142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42)])
angles = jnp.array([-45.0])    # fibre angle per material (deg)
OUT = "square_solid_newarch.out"
############################################################

d = _yaml.safe_load(open(YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
ne = len(tri)
mat_id = np.ones(ne, int)          # sc["mat_id"] is 1-BASED
for k, s in enumerate(d["sets"]["element"]):
    mat_id[np.array(s["labels"], int) - 1] = k + 1

sc = {"dim": 2, "nodes": nd[:, :2], "cells": [list(c) for c in tri],
      "mat_id": mat_id, "materials": {0: {"name": d["materials"][0]["name"]}},
      "scale": 1.0}

t0 = time.perf_counter()
r = plate_homo_2d(sc, material_param=material_param, angles=angles,
                  n_model=n_model, elem_rotation=elem_rotation_from_yaml(ori), plot=False)
C = np.asarray(r["C_eff"])
C = 0.5*(C + C.T)
dt = time.perf_counter() - t0

S = np.linalg.inv(C)
E1, E2, E3 = 1/S[0, 0], 1/S[1, 1], 1/S[2, 2]
G23, G13, G12 = 1/S[3, 3], 1/S[4, 4], 1/S[5, 5]
n12, n13, n23 = -S[0, 1]/S[0, 0], -S[0, 2]/S[0, 0], -S[1, 2]/S[1, 1]

print("OpenSG new architecture, n_model=3, omega=%.6g  [%.1f s]" % (r["omega"], dt))
print("C_eff (6x6, Pa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))
rows = [("E1", E1), ("E2", E2), ("E3", E3), ("G23", G23), ("G13", G13),
        ("G12", G12), ("nu12", n12), ("nu13", n13), ("nu23", n23)]
print("9 effective constants:")
for k, v in rows:
    print("  %-5s %14.6e" % (k, v))

with open(OUT, "w") as f:
    f.write("# equivalent 3-D solid properties, OpenSG new architecture"
            " (opensg_solid.sg_homo, n_model=3)\n")
    f.write("# source: %s  (square tube, midline a=1.0, t=0.03, m45 ply -45 deg)\n"
            % YAML)
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]; omega=%.8g; solve %.1f s\n"
            % (r["omega"], dt))
    f.write("# ---- C_eff (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k, v in rows:
        f.write("%-5s %16.8e\n" % (k, v))
print("wrote %s" % OUT)
