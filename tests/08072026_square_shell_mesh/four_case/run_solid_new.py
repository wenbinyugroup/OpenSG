"""ROUTE B -- equivalent 3-D solid stiffness from the 2-D SOLID SG, OpenSG NEW
architecture (opensg_solid.sg_homo.plate_homo_2d, n_model = 3).

plate_homo_2d normalizes by the SG measure omega = the meshed WALL MATERIAL
area (2Lt - t^2).  Every route in this study is reported per CELL area L^2, so
the returned C_eff is rescaled by omega/L^2.

For the anisotropic cases the ply angle rides in `angles` and the per-element
material FRAMES ride in `elem_rotation`, built from the 2-D solid yaml's
elementOrientations through elem_rotation_from_yaml (which cycles the yaml
component order (y2, y3, axial) into the solver order (axial, y2, y3)).

Run:  ~/miniconda3/envs/opensg_2_0/bin/python run_solid_new.py <case>
"""
import json
import sys
import time

import numpy as np
import jax.numpy as jnp
import yaml as _yaml

from crosscell import ISO, M45, material_param_row
from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_materials import elem_rotation_from_yaml

case = sys.argv[1]
# optional 2nd arg: override the fibre angle (the +45 sign-convention probe,
# see README.md -- the shell layup angle and this driver's rotate_C_matrix use
# OPPOSITE senses, so the same number is a different physical laminate)
ANG = float(sys.argv[2]) if len(sys.argv) > 2 else None
meta = json.load(open("inputs_meta.json"))[case]
L = 1.0
CELL_AREA = L*L
YAML = "%s_solid2d.yaml" % case
mat = ISO if meta["material"] == "iso" else M45
ply = float(mat["angle"]) if ANG is None else ANG
OUT = ("%s_solidnew.out" % case if ANG is None
       else "%s_solidnew_ang%+g.out" % (case, ANG))
material_param = jnp.array([material_param_row(mat)])
angles = jnp.array([ply])

d = _yaml.safe_load(open(YAML))
row = lambda r: [float(v) for v in
                 " ".join(str(x) for x in (r if isinstance(r, list) else [r])).split()]
nd = np.array([row(r) for r in d["nodes"]], float)
tri = np.array([[int(v) for v in row(r)] for r in d["elements"]], int) - 1
ori = np.array(d["elementOrientations"], float)
ne = len(tri)
mat_id = np.ones(ne, int)                      # sc["mat_id"] is 1-BASED
for k, s in enumerate(d["sets"]["element"]):
    mat_id[np.array(s["labels"], int) - 1] = k + 1

sc = {"dim": 2, "nodes": nd[:, :2], "cells": [list(c) for c in tri],
      "mat_id": mat_id, "materials": {0: {"name": d["materials"][0]["name"]}},
      "scale": 1.0}

# ISO is frame-independent -> pass None, exactly as the reference workflow does
elem_rot = None if ply == 0.0 else elem_rotation_from_yaml(ori)

t0 = time.perf_counter()
r = plate_homo_2d(sc, material_param=material_param, angles=angles, n_model=3,
                  elem_rotation=elem_rot, plot=False)
dt = time.perf_counter() - t0
omega = float(r["omega"])
C = np.asarray(r["C_eff"])
C = 0.5*(C + C.T)*omega/CELL_AREA              # per-omega -> per CELL area

S = np.linalg.inv(C)
rows = [("E1", 1/S[0, 0]), ("E2", 1/S[1, 1]), ("E3", 1/S[2, 2]),
        ("G23", 1/S[3, 3]), ("G13", 1/S[4, 4]), ("G12", 1/S[5, 5]),
        ("nu12", -S[0, 1]/S[0, 0]), ("nu13", -S[0, 2]/S[0, 0]),
        ("nu23", -S[1, 2]/S[1, 1])]

print("[B msg_solid new] %s  t/L=%.4f  omega=%.10f (exact 2Lt-t^2=%.10f)  [%.1f s]"
      % (case, meta["t"]/L, omega, meta["area_exact"], dt))
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))

with open(OUT, "w") as f:
    f.write("# ROUTE B -- msg_solid NEW architecture, opensg_solid.sg_homo.plate_homo_2d,"
            " n_model=3\n")
    f.write("# case %s : cross lattice cell, L=%.3f, t=%.4f (t/L=%.4f), %s ply %.1f deg\n"
            % (case, L, meta["t"], meta["t"]/L, meta["material"], ply))
    f.write("# mesh %d nodes / %d linear triangles, nt=%d through t, na=%d per arm\n"
            % (meta["solid_nodes"], meta["solid_elems"], meta["nt"], meta["na"]))
    f.write("# elem_rotation = %s\n"
            % ("elem_rotation_from_yaml(elementOrientations)"
               if elem_rot is not None else "None (isotropic)"))
    f.write("# order [e11 e22 e33 2e23 2e13 2e12]\n")
    f.write("# omega (meshed wall material area) = %.10f ; exact 2Lt-t^2 = %.10f\n"
            % (omega, meta["area_exact"]))
    f.write("# normalized by the CELL area L^2 = %.8f (C_eff*omega/L^2)\n" % CELL_AREA)
    f.write("# solve %.2f s\n" % dt)
    f.write("# ---- C_eff (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k, v in rows:
        f.write("%-5s %16.8e\n" % (k, v))
print("wrote %s" % OUT)
