"""Equivalent 3-D solid properties -- OpenSG new architecture, UNROTATED ply.

The head-to-head against the old JAX_BICGoptimize driver: same mesh, same
material, no per-element frames and no ply rotation (elem_rotation = None,
angles = 0), which is the only setting the .sc/old-driver route can express.

Reads ../square_tube_2Dsolid_t1only.yaml (mesh only) and writes the 6x6 C_eff
plus the 9 engineering constants.  Voigt [e11 e22 e33 2e23 2e13 2e12].

Run (from this folder):  python solid_props_newarch_plain.py
"""
import time

import numpy as np
import jax.numpy as jnp
import yaml as _yaml

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

############### User Input #################################
n_model = 3
YAML = "../square_tube_2Dsolid_t1only.yaml"
material_param = jnp.array([
    (142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42)])
angles = jnp.array([0.0])
OUT = "square_solid_newarch_plain.out"
############################################################

d = _yaml.safe_load(open(YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ne = len(tri)
mat_id = np.ones(ne, int)          # sc["mat_id"] is 1-BASED
for k, s in enumerate(d["sets"]["element"]):
    mat_id[np.array(s["labels"], int) - 1] = k + 1

sc = {"dim": 2, "nodes": nd[:, :2], "cells": [list(c) for c in tri],
      "mat_id": mat_id, "materials": {0: {"name": d["materials"][0]["name"]}},
      "scale": 1.0}

t0 = time.perf_counter()
r = plate_homo_2d(sc, material_param=material_param, angles=angles,
                  n_model=n_model, plot=False)
C = np.asarray(r["C_eff"])
C = 0.5*(C + C.T)
dt = time.perf_counter() - t0

S = np.linalg.inv(C)
rows = [("E1", 1/S[0, 0]), ("E2", 1/S[1, 1]), ("E3", 1/S[2, 2]),
        ("G23", 1/S[3, 3]), ("G13", 1/S[4, 4]), ("G12", 1/S[5, 5]),
        ("nu12", -S[0, 1]/S[0, 0]), ("nu13", -S[0, 2]/S[0, 0]),
        ("nu23", -S[1, 2]/S[1, 1])]

print("OpenSG new architecture, n_model=3, unrotated, omega=%.6g  [%.1f s]"
      % (r["omega"], dt))
print("C_eff (6x6, Pa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))
print("9 effective constants:")
for k, v in rows:
    print("  %-5s %14.6e" % (k, v))

with open(OUT, "w") as f:
    f.write("# equivalent 3-D solid properties, OpenSG new architecture"
            " (opensg_solid.sg_homo, n_model=3), UNROTATED ply\n")
    f.write("# source: %s ; angles=0, no elem_rotation -- matches what the\n"
            "# old .sc driver can express, for the code-to-code check\n" % YAML)
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]; omega=%.8g; solve %.1f s\n"
            % (r["omega"], dt))
    f.write("# ---- C_eff (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k, v in rows:
        f.write("%-5s %16.8e\n" % (k, v))
print("wrote %s" % OUT)
