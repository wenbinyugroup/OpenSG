"""Step 6 -- two-step RM dehomogenization of every station: .SM / .EM / .U files.

Each station's .ff (u, theta, DCM, section resultants from BeamDyn) drives the
two-step recovery over the FULL cross-section including the thickness (every ring
element x n_depth ply-interior points, 1-D RM plate SG): step 1 the RM section
strains + gradients, step 2 the MSG-RM through-thickness stress/strain, plus the
total displacement (beam motion + RM warping).  VABS-layout outputs per station:
<tag>.SM (stress, material frame), <tag>.EM (strain), <tag>.U (displacement).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   prefix              station tag prefix; reads cross_sections/ + ff/
#   n_depth             through-thickness recovery points per element
#   stress_frame        "material" (ply axes) | "plate" (wall axes)
#   png_station         r of the station to plot (nearest match), e.g. 0.2
#   recovery/           output folder: <tag>.SM/.EM/.U + <tag>_dehom_stress.png
# ----------------------------------------------------------------------------
"""
import glob
import os
import re
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensg_shell.windio import dehom_to_files

############### User Input #################################
prefix = "iea22"                   # station tag prefix
n_depth = 9                        # through-thickness recovery points per element
stress_frame = "material"          # "material" (ply axes) | "plate" (wall axes)
png_station = 0.2                  # r of the station to plot (nearest .ff)
############################################################

ffs = sorted(glob.glob(os.path.join("ff", prefix + "_r*.ff")))
os.makedirs("recovery", exist_ok=True)
rs = np.array([int(re.search(r"_r(\d{4})", os.path.basename(f)).group(1)) / 1000.0
               for f in ffs])
png_r = rs[int(np.argmin(np.abs(rs - png_station)))]

for f, r in zip(ffs, rs):
    tag = os.path.basename(f)[:-3]
    y = os.path.join("cross_sections", tag + "_shell.yaml")
    t0 = time.perf_counter()
    D = dehom_to_files(y, f, out_base=os.path.join("recovery", tag),
                       n_depth=n_depth, frame=stress_frame)
    sig = D["sig"] / 1e6                                  # MPa for reporting
    imax = int(np.argmax(np.abs(sig[:, 0])))
    print("%s  eta=%.3f  [%5.1f s]  peak |s11| = %8.2f MPa at (%.3f, %.3f)  "
          "|u|max = %.4g m"
          % (tag, r, time.perf_counter() - t0, abs(sig[imax, 0]),
             D["pts"][imax, 0], D["pts"][imax, 1], np.abs(D["u"]).max()))

    if abs(r - png_r) < 1e-9:                             # stress cloud for one station
        pts = D["pts"]
        fig, ax = plt.subplots(2, 1, figsize=(9.5, 8.6))
        for a, (ci, lab) in zip(ax, [(0, r"$\sigma_{11}$"), (5, r"$\sigma_{12}$")]):
            mlim = np.nanpercentile(np.abs(sig[:, ci]), 99) or 1e-9
            cs = a.scatter(pts[:, 0], pts[:, 1], c=np.clip(sig[:, ci], -mlim, mlim),
                           s=4, cmap="rainbow", vmin=-mlim, vmax=mlim)
            a.set_aspect("equal"); a.set_xlabel("y2 (m)"); a.set_ylabel("y3 (m)")
            cb = fig.colorbar(cs, ax=a, shrink=0.9, pad=0.02)
            cb.set_label(lab + " (MPa)")
        fig.tight_layout()
        fig.savefig(os.path.join("recovery", tag + "_dehom_stress.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

print("wrote recovery/ (%d stations x .SM/.EM/.U, PNG at r=%.3f)" % (len(ffs), png_r))
