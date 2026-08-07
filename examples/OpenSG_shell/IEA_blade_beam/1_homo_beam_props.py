"""IEA-22 blade, r/R = 0.2 -- STEP 1: beam properties from the RM shell
cross-section (the benchmark's homogenization half).

RM 6-DOF shell ring homogenization (independent-omega3 element, element-wise
drilling Lagrange multiplier, gamma_23 tied by MITC) with the MSG (Yu-2002
least-squares) wall transverse-shear G -- opensg_shell.build_rm_bundle, the
production OpenSG-TW route, now running on the OpenSG-2.0 stack.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   name            reads <name>_shell.yaml (1-D shell SG: nodes, elements,
#                   elementOrientations, sections/layups, materials, sets)
#   B               build_rm_bundle result: Timo 6x6 + the RM warping V0/V1,
#                   strip geometry, layup/material dbs (consumed by step 2)
#   C6              (6,6) Timoshenko [eps11 gam12 gam13 kap1 kap2 kap3]
#                   = VABS diagonal order [EA GA2 GA3 GJ EI2 EI3]
#   corners, rc     (n_nd,2) ring node coords / (n_el,2) element connectivity
#   layups          per-element layup name; ldb/mdb the layup + material dbs
#   OUT             <name>_beam_Timo.out, <name>_ring_mesh.png
# ----------------------------------------------------------------------------
"""
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell import build_rm_bundle

############### User Input #################################
name = "iea_s10"                  # IEA-22 station at r/R = 0.20
############################################################

t0 = time.perf_counter()
B = build_rm_bundle(name + "_shell.yaml")
C6 = np.asarray(B["Timo"])
print("ring homogenized in %.1f s  (reference=%s, wall G=%s)"
      % (time.perf_counter() - t0, B["ref"], B["g_source"]))

LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
print("\nTimoshenko 6x6 diagonal:")
for i, l in enumerate(LBL):
    print("   %-4s = %12.5e" % (l, C6[i, i]))

np.savetxt(name + "_beam_Timo.out", C6, fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2 kappa3]"
                  " (RM 6-DOF shell ring, MITC g23, MSG wall G) from the 1-D"
                  " shell SG %s_shell.yaml, reference=%s" % (name, B["ref"]))

# ---- the ring mesh, elements colored by layup (the real computed ring) ----
corners = np.asarray(B["corners"])
rc = np.asarray(B["red_cells"])
layups = B["layup_per_elem"]
uniq = sorted(set(layups))
cmap = plt.get_cmap("tab20")
col = {ln: cmap(i % 20) for i, ln in enumerate(uniq)}

fig, ax = plt.subplots(figsize=(11.0, 5.0))
for e in range(rc.shape[0]):
    p = corners[[int(rc[e, 0]), int(rc[e, 1])]]
    ax.plot(p[:, 0], p[:, 1], "-", color=col[layups[e]], lw=2.2,
            solid_capstyle="butt")
ax.plot(corners[:, 0], corners[:, 1], "k.", ms=1.6)
ax.set_aspect("equal")
ax.set_xlabel("$y_2$ [m]")
ax.set_ylabel("$y_3$ [m]")
ax.grid(alpha=0.25)
handles = [plt.Line2D([0], [0], color=col[ln], lw=3, label=ln) for ln in uniq]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5),
           frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(name + "_ring_mesh.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nwrote %s_beam_Timo.out, %s_ring_mesh.png  (%d elements, %d layups)"
      % (name, name, rc.shape[0], len(uniq)))
