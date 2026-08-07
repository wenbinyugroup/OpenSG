"""C44 mesh-convergence study for the SOLID routes (B and C share the mesh).

WHY.  C44 (the 2e23 macro shear) is the only BENDING-dominated term of the
cross lattice: the walls bend.  Linear CST triangles lock in bending, and the
parasitic shear energy scales with the ALONG-WALL element size measured in wall
thicknesses.  A mesh whose along-wall size is fixed in L therefore becomes
relatively coarser -- and the solid spuriously stiffer -- as t shrinks.  The
along-wall size must scale with t.

Reported against the slender-limit exact value 0.5 * E/(1-nu^2) * (t/L)^3 per
cell area, which the 1-D shell route reproduces to 1.5e-4 at t/L = 0.01.

Run:  ~/miniconda3/envs/opensg_2_0/bin/python convergence.py [thin_iso|thick_iso]
"""
import sys
import time

import numpy as np
import jax.numpy as jnp

from crosscell import ISO, cross_solid_mesh, material_param_row
from opensg_solid.sg_homo import plate_homo_2d

case = sys.argv[1] if len(sys.argv) > 1 else "thin_iso"
L = 1.0
t = 0.01 if case.startswith("thin") else 0.10
CELL = L*L
Ep = 70.0e9/(1-0.30**2)
exact = 0.5*Ep*(t/L)**3
mp = jnp.array([material_param_row(ISO)])

MODE = sys.argv[2] if len(sys.argv) > 2 else "coarse"
_G = {("thin", "coarse"): [(6, 99), (6, 149), (6, 198), (9, 149), (6, 396),
                           (12, 396)],
      ("thin", "fine"): [(12, 396), (18, 594), (24, 792)],
      # PRODUCTION mesh (mesh_sizing) + its 1.5x refinement
      ("thin", "prod"): [(18, 594), (27, 891)],
      ("thick", "coarse"): [(6, 20), (9, 30), (6, 40), (12, 40), (6, 80),
                            (12, 80)],
      ("thick", "fine"): [(12, 40), (18, 60), (24, 80)],
      ("thick", "prod"): [(18, 54), (27, 81)]}
GRID = _G[("thin" if case.startswith("thin") else "thick", MODE)]

print("%s : t/L=%.4f   exact slender C44 = %.6e Pa (per cell area L^2)"
      % (case, t/L, exact))
print("%-4s %-5s %8s %8s %9s %9s %16s %9s %9s"
      % ("nt", "na", "h_a/t", "aspect", "nodes", "tris", "C44", "vs exact",
         "vs prev"))
prev = None
rows = []
for nt, na in GRID:
    nd, tri, frame, area = cross_solid_mesh(L, t, nt, na)
    sc = {"dim": 2, "nodes": nd, "cells": [list(c) for c in tri],
          "mat_id": np.ones(len(tri), int), "materials": {0: {"name": "iso"}},
          "scale": 1.0}
    t0 = time.perf_counter()
    r = plate_homo_2d(sc, material_param=mp, angles=jnp.array([0.0]),
                      n_model=3, elem_rotation=None, plot=False)
    om = float(r["omega"])
    C = np.asarray(r["C_eff"])*om/CELL
    c44 = float(C[3, 3])
    h_a = (0.5*L - 0.5*t)/na
    print("%-4d %-5d %8.3f %8.3f %9d %9d %16.6e %+8.2f%% %8s   [%.0f s]"
          % (nt, na, h_a/t, h_a/(t/nt), len(nd), len(tri), c44,
             100*(c44-exact)/exact,
             ("%+.2f%%" % (100*(c44-prev)/prev)) if prev else "--",
             time.perf_counter()-t0))
    rows.append((nt, na, h_a/t, h_a/(t/nt), len(nd), len(tri), c44,
                 100*(c44-exact)/exact,
                 100*(c44-prev)/prev if prev else float("nan"),
                 float(C[0, 0])))
    prev = c44

with open("convergence_%s_%s.dat" % (case, MODE), "w") as f:
    f.write("# C44 mesh convergence, SOLID routes, %s (t/L=%.4f), per cell"
            " area L^2=%.4f\n" % (case, t/L, CELL))
    f.write("# exact slender limit 0.5*E/(1-nu^2)*(t/L)^3 = %.8e Pa\n" % exact)
    f.write("# nt = elements through the wall thickness; na = elements per arm\n")
    f.write("# %-4s %-5s %8s %8s %9s %9s %16s %10s %10s %16s\n"
            % ("nt", "na", "h_a/t", "aspect", "nodes", "tris", "C44",
               "err_exact", "d_prev", "C11"))
    for r_ in rows:
        f.write("  %-4d %-5d %8.3f %8.3f %9d %9d %16.8e %+9.3f%% %+9.3f%% %16.8e\n"
                % r_)
print("wrote convergence_%s_%s.dat" % (case, MODE))
