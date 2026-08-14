"""Step 2 -- beam properties of every station: RM shell homogenization + mass matrix.

Each 1-D shell SG yaml from step 1 is homogenized with the production RM 6-DOF ring
(build_rm_bundle: MSG wall transverse-shear G, MITC-tied gamma_23) and the 6x6 mass
matrix is integrated over the same ring (thickness density moments, wall-normal
projected).  Every station writes a VABS .K-layout file <tag>.K holding the mass
blocks, the Timoshenko stiffness/compliance, and the center/principal summaries.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   prefix              station tag prefix from step 1
#   cross_sections/     input folder of <prefix>_rXXXX_shell.yaml
#   props/              output folder of <tag>.K (VABS .K layout)
#   <prefix>_timo_by_r.dat   r + flattened Timoshenko 6x6 per station
#   <prefix>_mass_by_r.dat   r + mass-per-span + flattened mass 6x6 per station
#   <prefix>_beam_props.png  spanwise diagonal stiffness + mass distributions
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

from opensg_shell.windio import beam_props

############### User Input #################################
prefix = "iea22"                   # station tag prefix from step 1
############################################################

yams = sorted(glob.glob(os.path.join("cross_sections", prefix + "_r*_shell.yaml")))
os.makedirs("props", exist_ok=True)
LBL = ["EA", "GA2", "GA3", "GJ", "EI2", "EI3"]
rows_k = []; rows_m = []
for y in yams:
    tag = os.path.basename(y)[:-len("_shell.yaml")]
    r = int(re.search(r"_r(\d{4})", tag).group(1)) / 1000.0
    t0 = time.perf_counter()
    P = beam_props(y, out_k=os.path.join("props", tag + ".K"))
    dt = time.perf_counter() - t0
    C6, M6 = P["Timo"], P["Mass"]
    rows_k.append(np.r_[r, C6.flatten()])
    rows_m.append(np.r_[r, P["info"]["mpus"], M6.flatten()])
    print("%s  r=%.3f  [%5.1f s]  " % (tag, r, dt) +
          "  ".join("%s=%.4g" % (LBL[i], C6[i, i]) for i in range(6)) +
          "  mu=%.4g kg/m" % P["info"]["mpus"])

np.savetxt(prefix + "_timo_by_r.dat", np.array(rows_k), fmt="%.8e",
           header="r then Timoshenko 6x6 row-major (VABS order)")
np.savetxt(prefix + "_mass_by_r.dat", np.array(rows_m), fmt="%.8e",
           header="r mass_per_span then mass 6x6 row-major (VABS frame)")


