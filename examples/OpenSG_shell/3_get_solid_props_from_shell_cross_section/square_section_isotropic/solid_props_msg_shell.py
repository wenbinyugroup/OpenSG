"""Equivalent 3-D solid properties of an ISOTROPIC square tube -- MSG_SHELL.

Reads the 1-D shell SG (square_tube_1Dshell.yaml: wall midline, center
reference, isotropic wall) and writes the 6x6 C3D plus the 9 engineering
constants.  PERIODIC by default per Rules/periodicity_in_solid_props.md: for a
2-D SG the opposite face edges and the corners are tied through the shell
periodic assembly map.

Normalized by the WALL MATERIAL area 4*a*t, the same measure the solid engines
use for n_model = 3, so the two routes are directly comparable.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   YAML                the 1-D shell SG
#   a, t_w              wall midline side, wall thickness
#   CELL_AREA           4*a*t = the wall material area
#   B["C3D"]            (6, 6) equivalent solid stiffness,
#                       order [e11 e22 e33 2e23 2e13 2e12]
#   square_solid_msg_shell.out      the output
# ----------------------------------------------------------------------------
"""
import numpy as np

from opensg_shell import build_solid_bundle, GBAR_ORDER
from opensg_shell.sg_homo import elastic_constants

############### User Input #################################
YAML = "square_tube_1Dshell.yaml"
a, t_w = 1.0, 0.03
CELL_AREA = 4*a*t_w
OUT = "square_solid_msg_shell.out"
############################################################

KEYS = ("E1", "E2", "E3", "G23", "G13", "G12", "nu12", "nu13", "nu23")
B = build_solid_bundle(YAML, cell_area=CELL_AREA)
C = np.asarray(B["C3D"])
cons, _ = elastic_constants(C)

print("msg_shell solid props (reference=%s, wall G=%s, cell area=%.4f)"
      % (B["ref"], B["g_source"], CELL_AREA))
print("C3D (6x6, Pa), %s:" % GBAR_ORDER)
for i in range(6):
    print("  " + " ".join("%14.6e" % C[i, j] for j in range(6)))
print("9 effective constants:")
for k in KEYS:
    print("  %-5s %14.6e" % (k, cons[k]))

with open(OUT, "w") as f:
    f.write("# equivalent 3-D solid properties from the 1-D SHELL SG\n")
    f.write("# msg_shell: opensg_shell.build_solid_bundle, PERIODIC 2-D SG\n")
    f.write("# source: %s (isotropic square tube, a=%.2f, t=%.3f, center ref)\n"
            % (YAML, a, t_w))
    f.write("# %s ; normalized by the wall material area %.4f\n"
            % (GBAR_ORDER, CELL_AREA))
    f.write("# ---- C3D (6x6, Pa) ----\n")
    for i in range(6):
        f.write(" ".join("%16.8e" % C[i, j] for j in range(6)) + "\n")
    f.write("# ---- 9 effective constants ----\n")
    for k in KEYS:
        f.write("%-5s %16.8e\n" % (k, cons[k]))
print("wrote %s" % OUT)
