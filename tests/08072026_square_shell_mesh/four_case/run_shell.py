"""ROUTE A -- equivalent 3-D solid stiffness from the 1-D SHELL SG.

opensg_shell.build_solid_bundle (the Cab Gamma_e/Gamma_h route), PERIODIC by
default: opposite face edges and corners of the 2-D SG are tied through the
shell periodic assembly map, so no constraint rows are added.

Normalized by the CELL area L^2 -- the natural measure for a lattice, it
includes the void, and it is the SAME measure routes B and C are rescaled to.

Run:  ~/miniconda3/envs/opensg_2_0/bin/python run_shell.py <case>
"""
import json
import sys
import time

import numpy as np

from opensg_shell import build_solid_bundle, GBAR_ORDER
from opensg_shell.solid_props import elastic_constants

case = sys.argv[1]
meta = json.load(open("inputs_meta.json"))[case]
L = 1.0
CELL_AREA = L*L
YAML = "%s_shell.yaml" % case
OUT = "%s_shell.out" % case
KEYS = ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23")

t0 = time.perf_counter()
B = build_solid_bundle(YAML, cell_area=CELL_AREA)      # periodic by default
C = np.asarray(B["C3D"])
C = 0.5*(C + C.T)
cons, _ = elastic_constants(C)
dt = time.perf_counter() - t0

thr = 1e-6*np.max(np.abs(C))
terms = [(i+1, j+1) for i in range(6) for j in range(i, 6) if abs(C[i, j]) > thr]
print("[A msg_shell] %s  t/L=%.4f  ref=%s  wall G=%s  cell area=%.6f  [%.1f s]"
      % (case, meta["t"]/L, B["ref"], B["g_source"], CELL_AREA, dt))
print("resolved terms: %s" % (terms,))
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))

with open(OUT, "w") as f:
    f.write("# ROUTE A -- msg_shell 1-D shell SG, opensg_shell.build_solid_bundle\n")
    f.write("# case %s : cross lattice cell, L=%.3f, t=%.4f (t/L=%.4f), %s ply %.1f deg\n"
            % (case, L, meta["t"], meta["t"]/L, meta["material"], meta["angle"]))
    f.write("# PERIODIC (default); reference=%s; wall G source=%s; %d elements per wall\n"
            % (B["ref"], B["g_source"], meta["nseg"]))
    f.write("# %s\n" % GBAR_ORDER)
    f.write("# normalized by the CELL area L^2 = %.8f (includes the void)\n" % CELL_AREA)
    f.write("# solve %.2f s\n" % dt)
    f.write("# ---- C3D (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k in KEYS:
        f.write("%-5s %16.8e\n" % (k, cons[k]))
print("wrote %s" % OUT)
