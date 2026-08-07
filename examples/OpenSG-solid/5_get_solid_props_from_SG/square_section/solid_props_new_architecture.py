"""Equivalent 3-D solid properties of the square thin-walled section --
OpenSG NEW architecture (opensg_solid.sg_homo, n_model = 3).

Reads square_tube_2Dsolid.yaml (3228 nodes, 5384 linear triangles, one
orthotropic ply; e1 is out of plane per Rules/orientation_e1_out_of_plane.md)
and writes the 6x6 C_eff plus the 9 engineering constants.

USE_ORIENTATIONS = False reproduces exactly what the old .sc-based script can
express (unrotated ply, no per-element frames) -- that is the head-to-head
comparison in README.md.  Set it True to use the yaml's per-element frames with
the real -45 deg ply, which the old route cannot represent.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, YAML       3 = 3D elastic; reads the 2-D SG yaml
#   material_param      (1, 9) engineering constants (E1 E2 E3 G12 G13 G23
#                       nu12 nu13 nu23), SI (Pa)
#   USE_ORIENTATIONS    per-element frames + -45 deg ply, or unrotated
#   sc                  the engine's own dict {dim, nodes, cells, mat_id, ...}
#   r["C_eff"]          (6, 6) equivalent solid stiffness, order
#                       [e11 e22 e33 2e23 2e13 2e12]
#   square_solid_new_architecture.out    the output
# ----------------------------------------------------------------------------
"""
import time

import numpy as np
import jax.numpy as jnp
import yaml as _yaml

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

############### User Input #################################
n_model = 3                    # 1: Beam; 2: Plate; 3: 3D elastic
YAML = "square_tube_2Dsolid.yaml"
# orthotropic ply, SI (Pa): E1 E2 E3 G12 G13 G23 nu12 nu13 nu23
material_param = jnp.array([
    (142.0e9, 9.8e9, 9.8e9, 6.0e9, 6.0e9, 4.8e9, 0.30, 0.30, 0.42)])
USE_ORIENTATIONS = False       # True: per-element frames + the -45 deg ply
OUT = "square_solid_new_architecture.out"
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
angles = jnp.array([-45.0]) if USE_ORIENTATIONS else jnp.array([0.0])
elem_rotation = (elem_rotation_from_yaml(ori)
                 if USE_ORIENTATIONS else None)

t0 = time.perf_counter()
r = plate_homo_2d(sc, material_param=material_param, angles=angles,
                  n_model=n_model, elem_rotation=elem_rotation, plot=False)
C = np.asarray(r["C_eff"])
C = 0.5*(C + C.T)
dt = time.perf_counter() - t0

S = np.linalg.inv(C)
rows = [("E1", 1/S[0, 0]), ("E2", 1/S[1, 1]), ("E3", 1/S[2, 2]),
        ("G23", 1/S[3, 3]), ("G13", 1/S[4, 4]), ("G12", 1/S[5, 5]),
        ("nu12", -S[0, 1]/S[0, 0]), ("nu13", -S[0, 2]/S[0, 0]),
        ("nu23", -S[1, 2]/S[1, 1])]

print("OpenSG new architecture, n_model=%d, orientations=%s, omega=%.6g  [%.1f s]"
      % (n_model, USE_ORIENTATIONS, r["omega"], dt))
print("C_eff (6x6, Pa), order [e11 e22 e33 2e23 2e13 2e12]:")
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))
print("9 effective constants:")
for k, v in rows:
    print("  %-5s %14.6e" % (k, v))

with open(OUT, "w") as f:
    f.write("# equivalent 3-D solid properties -- OpenSG NEW architecture"
            " (opensg_solid.sg_homo, n_model=%d)\n" % n_model)
    f.write("# source: %s ; orientations=%s\n" % (YAML, USE_ORIENTATIONS))
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]; omega=%.8g; solve %.1f s\n"
            % (r["omega"], dt))
    f.write("# ---- C_eff (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k, v in rows:
        f.write("%-5s %16.8e\n" % (k, v))
print("wrote %s" % OUT)
