"""Equivalent 3-D solid properties from the 1-D SHELL SG -- msg_shell new
architecture (opensg_shell.build_solid_bundle, the Cab Gamma_e/Gamma_h route).

Reads ../square_tube_1Dshell.yaml (square tube wall midline, center reference,
single ply) and writes the 6x6 C3D plus the 9 engineering constants.

PERIODIC by default, per Rules/periodicity_in_solid_props.md: opposite face
edges and the corners of the 2-D SG are tied through the shell periodic
assembly map (opensg_shell.sg_periodicity), the mirror of the solid side's
fe_jax/periodic_multiscale.py.  Periodicity rides in the local->global assembly
map, so no constraint rows are added.

Normalized by the WALL MATERIAL area (0.12), the same measure the solid engines
use for n_model = 3, so C11 is directly the effective axial modulus.

Cell note: this ring has its walls ON the cell boundary, so each is shared with
the neighbour and integrated twice -- the numbers are those of the doubled-wall
lattice.  A cell with the walls running THROUGH it owns each wall once.

Run (from this folder):  python solid_props_msg_shell_1dyaml.py
"""
import time

import numpy as np

from opensg_shell import build_solid_bundle, GBAR_ORDER
from opensg_shell.sg_homo import elastic_constants

############### User Input #################################
YAML = "../square_tube_1Dshell.yaml"
a, t_w = 1.0, 0.03             # midline side, wall thickness
CELL_AREA = 4*a*t_w            # wall material area = 0.12 (matches n_model=3)
OUT = "square_solid_msg_shell.out"
############################################################

KEYS = ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23")
t0 = time.perf_counter()
B = build_solid_bundle(YAML, cell_area=CELL_AREA)      # periodic by default
C = np.asarray(B["C3D"])
cons, _ = elastic_constants(C)
dt = time.perf_counter() - t0

thr = 1e-6*np.max(np.abs(C))
terms = [(i+1, j+1) for i in range(6) for j in range(i, 6) if abs(C[i, j]) > thr]
print("msg_shell solid props, PERIODIC (reference=%s, wall G=%s, cell area=%.4f)"
      "  [%.1f s]" % (B["ref"], B["g_source"], CELL_AREA, dt))
print("resolved terms: %s" % terms)
print("C3D (6x6, Pa), %s:" % GBAR_ORDER)
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))
print("9 effective constants:")
for k in KEYS:
    print("  %-5s %14.6e" % (k, cons[k]))

with open(OUT, "w") as f:
    f.write("# equivalent 3-D solid properties from the 1-D SHELL SG\n")
    f.write("# msg_shell new architecture: opensg_shell.build_solid_bundle\n")
    f.write("# PERIODIC 2-D SG (opposite face edges + corners tied through the\n")
    f.write("# shell periodic assembly map) -- Rules/periodicity_in_solid_props.md\n")
    f.write("# source: %s (square tube, midline a=%.2f, t=%.3f, center ref)\n"
            % (YAML, a, t_w))
    f.write("# %s ; normalized by the wall material area %.6f; solve %.1f s\n"
            % (GBAR_ORDER, CELL_AREA, dt))
    f.write("# ---- C3D (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k in KEYS:
        f.write("%-5s %16.8e\n" % (k, cons[k]))
print("wrote %s" % OUT)
