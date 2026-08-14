"""Step 5 -- run BeamDyn and extract the per-station .ff recovery inputs.

beamdyn_driver solves the static beam; the converged BldNd channels at each output
node (= each property station, trapezoidal quadrature) are mapped to the VABS frame
and written one .ff per station: u, theta (Wiener-Milenkovic), the exact DCM, and
the section-LOCAL force/moment resultants -- the glb-type input the dehom consumes.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   prefix              station tag prefix; beamdyn/<prefix>_bd_driver.inp from step 4
#   beamdyn_exe         beamdyn_driver executable (PATH name or full path)
#   ld_library          extra LD_LIBRARY_PATH for conda-built OpenFAST (or None)
#   ff/                 output folder: <prefix>_rXXXX.ff + <prefix>_ff.dat table
# ----------------------------------------------------------------------------
"""
import glob
import os
import re

import numpy as np

from opensg_shell.pynumad import run_beamdyn, extract_ff

############### User Input #################################
prefix = "iea22"                   # station tag prefix
refine = 3                         # trapezoidal refinement (MUST match step 4)
beamdyn_exe = os.path.expanduser("~/miniconda3/bin/beamdyn_driver")
ld_library = os.path.expanduser("~/miniconda3/lib")   # None if beamdyn_driver is self-contained
############################################################

ks = sorted(glob.glob(os.path.join("props", prefix + "_r*.K")))
etas = np.array([int(re.search(r"_r(\d{4})", os.path.basename(k)).group(1)) / 1000.0
                 for k in ks])
driver = os.path.join("beamdyn", prefix + "_bd_driver.inp")
out = run_beamdyn(driver, exe=beamdyn_exe, ld_library_path=ld_library)
print("BeamDyn done: %s" % out)

files, tab = extract_ff(out, etas, "ff", prefix=prefix, refine=refine,
                        table=os.path.join("ff", prefix + "_ff.dat"))
from opensg_shell.pynumad import read_ff
tip = read_ff(files[-1]); root = read_ff(files[0])
print("extracted %d .ff -> ff/   (tip u = [%.4g %.4g %.4g] m)"
      % (len(files), *tip["u"]))
print("root resultants: F = [%.4g %.4g %.4g] N   M = [%.4g %.4g %.4g] N-m"
      % (*root["F"], *root["M"]))
