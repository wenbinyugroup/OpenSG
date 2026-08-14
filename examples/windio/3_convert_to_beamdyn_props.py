"""Step 3 -- station .K files -> BeamDyn blade (distributed properties) input.

Reads the Timoshenko stiffness and mass 6x6 from every station .K (VABS .K layout --
the reader also accepts native VABS .K files), rotates both to the BeamDyn blade frame
(beam-axis swap B = [[0,0,1],[0,-1,0],[1,0,0]], block-wise similarity), and writes the
BeamDyn blade properties file.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   prefix              station tag prefix; props/<prefix>_rXXXX.K from step 2
#   beamdyn/            output folder; <prefix>_bd_props.inp
# ----------------------------------------------------------------------------
"""
import glob
import os
import re

import numpy as np

from opensg_shell.windio import read_k_file, vabs_to_beamdyn, write_bd_props

############### User Input #################################
prefix = "iea22"                   # station tag prefix from step 2
############################################################

ks = sorted(glob.glob(os.path.join("props", prefix + "_r*.K")))
os.makedirs("beamdyn", exist_ok=True)
etas = []; K = []; M = []
for kf in ks:
    r = int(re.search(r"_r(\d{4})", os.path.basename(kf)).group(1)) / 1000.0
    Ki, Mi = read_k_file(kf)
    etas.append(r); K.append(Ki); M.append(Mi)
etas = np.array(etas); K = np.stack(K); M = np.stack(M)
Kb, Mb = vabs_to_beamdyn(K, M)
pf = write_bd_props(os.path.join("beamdyn", prefix + "_bd_props.inp"), etas, Kb, Mb)
print("BeamDyn props: %s  (%d stations, r = %.3f .. %.3f)"
      % (pf, len(etas), etas[0], etas[-1]))
