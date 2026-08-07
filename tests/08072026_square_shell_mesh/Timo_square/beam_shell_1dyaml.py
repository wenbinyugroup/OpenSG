"""Timoshenko 6x6 from the 1-D SHELL SG (msg_shell RM ring).

Reads square_tube_1Dshell.yaml (midline square, center reference, single m45
ply at -45 deg; e1 = beam axis out of plane, e2 = wall tangent, e3 = inward
normal) and runs the production RM 6-DOF ring homogenization.

Run (from this folder):  python beam_shell_1dyaml.py
"""
import time

import numpy as np

from opensg_shell import build_rm_bundle

############### User Input #################################
YAML = "../square_tube_1Dshell.yaml"
OUT = "square_tube_beam_shell.out"
############################################################

t0 = time.perf_counter()
B = build_rm_bundle(YAML)
C6 = np.asarray(B["Timo"])
dt = time.perf_counter() - t0

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
np.set_printoptions(precision=5, suppress=False)
print("shell 1-D SG: %s (reference=%s, wall G=%s)  [%.1f s]"
      % (YAML, B["ref"], B["g_source"], dt))
print("beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):")
for i in range(6):
    print("  " + " ".join("%14.6e" % C6[i, j] for j in range(6)))
print("diagonal: " + "  ".join("%s=%.6g" % (LBL[i], C6[i, i]) for i in range(6)))
np.savetxt(OUT, C6, fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2 kappa3]"
                  " (RM 6-DOF shell ring, MITC g23, MSG wall G) from the 1-D"
                  " shell SG %s, reference=%s" % (YAML, B["ref"]))
print("wrote %s" % OUT)
