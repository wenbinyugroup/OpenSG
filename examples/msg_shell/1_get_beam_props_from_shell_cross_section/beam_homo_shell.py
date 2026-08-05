"""Timoshenko beam properties of a thin-walled SHELL cross-section (1-D shell SG).

RM 6-DOF shell ring homogenization (independent-omega3 element with an element-wise
drilling Lagrange multiplier, gamma_23 tied by MITC) with the MSG (Yu-2002 LS) wall
transverse-shear G -- opensg_shell.build_rm_bundle, the production OpenSG-TW route.
The station is IEA-22 r/R = 0.2 (iea_s10), mid-surface (center) reference recorded
in the yaml's `reference` field.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   name                reads <name>_shell.yaml (1-D shell SG: nodes, elements,
#                       elementOrientations, sections/layups, materials, sets)
#   B                   build_rm_bundle result: B["Timo"] (6,6) + the RM warping
#                       V0/V1, strip geometry, layup dbs (consumed by example 2)
#   C6                  (6, 6) Timoshenko [eps11 gam12 gam13 kap1 kap2 kap3]
#                       = VABS diagonal order [EA GA2 GA3 GJ EI2 EI3]
#   <name>_beam_Timo.out, <name>_mesh.png, <name>_orient.png   the outputs
# ----------------------------------------------------------------------------
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell import build_rm_bundle, auto_emit

############### User Input #################################
name = "iea_s10"               # reads <name>_shell.yaml
############################################################

t0 = time.perf_counter()
B = build_rm_bundle(name + "_shell.yaml")     # RM 6-DOF ring + MSG wall G (g_source="msg")
C6 = np.asarray(B["Timo"])
dt = time.perf_counter() - t0

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
np.set_printoptions(precision=5)
print("shell 1-D SG: %s_shell.yaml  (reference=%s, wall G=%s)  [%.1f s]"
      % (name, B["ref"], B["g_source"], dt))
print("beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):")
print(C6)
print("diagonal: " + "  ".join("%s=%.5g" % (LBL[i], C6[i, i]) for i in range(6)))
np.savetxt(name + "_beam_Timo.out", C6, fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2 kappa3]"
                  " (RM 6-DOF shell ring, MITC g23, MSG wall G) from the 1-D shell"
                  " SG %s_shell.yaml, reference=%s" % (name, B["ref"]))

# ---- section mesh PNG: ring contour colored by layup (shell convention) ----
corners = np.asarray(B["corners"]); cells = np.asarray(B["red_cells"])
sets = sorted(set(B["layup_per_elem"]))
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(9.0, 5.0))
for k, ln in enumerate(sets):
    first = True
    for e in range(len(cells)):
        if B["layup_per_elem"][e] != ln:
            continue
        seg = corners[cells[e]]
        ax.plot(seg[:, 0], seg[:, 1], "-", color=cmap(k % 20), lw=2.2,
                label=ln if first else None)
        first = False
ax.plot(corners[:, 0], corners[:, 1], ".", color="0.25", ms=2.0)
ax.set_aspect("equal"); ax.set_xlabel("y2 (m)"); ax.set_ylabel("y3 (m)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(name + "_mesh.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- compulsory e1/e2/e3 orientation PNG ----
auto_emit(name + "_shell.yaml", out_png=name + "_orient.png")
print("wrote %s_beam_Timo.out (+ %s_mesh.png, %s_orient.png)" % (name, name, name))
