"""Step 4 -- BeamDyn primary + driver inputs with the per-station nodal loads.

The primary input carries a straight, untwisted reference line and TRAPEZOIDAL
quadrature so the BeamDyn output nodes coincide with the blade property stations.
The driver applies the surface-traction force model as nodal loads about the x1
reference axis: Fx_i = p * P_i * ds_i (flap) and Mz_i = p * int(x2 ds)_i * ds_i
(torsion), sharing the origin of the sectional 6x6.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   windio, prefix      windIO blade + station tag prefix
#   p_traction          uniform flapwise OML surface traction [Pa]
#   beamdyn/            <prefix>_bd_primary.inp + <prefix>_bd_driver.inp
#                       (props file from step 3 referenced by name)
# ----------------------------------------------------------------------------
"""
import glob
import os
import re

import numpy as np

from opensg_shell.windio import load_blade, blade_length, station_loads, \
    write_bd_primary, write_bd_driver

############### User Input #################################
windio = "IEA-22-280-RWT.yaml"     # windIO blade file (same as step 1)
prefix = "iea22"                   # station tag prefix
p_traction = 1500.0                # flapwise OML surface traction [Pa]
refine = 3                         # trapezoidal quadrature refinement (sparse-station
                                   # sets need >1 to converge; must match step 5)
############################################################

ks = sorted(glob.glob(os.path.join("props", prefix + "_r*.K")))
etas = np.array([int(re.search(r"_r(\d{4})", os.path.basename(k)).group(1)) / 1000.0
                 for k in ks])
blade = load_blade(windio)
L = blade_length(blade)
loads = station_loads(blade, etas, p_traction)
print("%s: L=%.3f m, p=%.0f Pa, %d stations" % (windio, L, p_traction, len(etas)))
print("  sum F_flap = %.4f MN   sum M_z = %.4e N-m"
      % (loads[:, 0].sum() / 1e6, loads[:, 5].sum()))

primary = write_bd_primary(os.path.join("beamdyn", prefix + "_bd_primary.inp"), L,
                           prefix + "_bd_props.inp", refine=refine,
                           title="%s blade (OpenSG msg-shell sections)" % prefix)
driver = write_bd_driver(os.path.join("beamdyn", prefix + "_bd_driver.inp"),
                         prefix + "_bd_primary.inp", etas, loads,
                         title="Static analysis of %s under %.0f Pa flapwise surface "
                               "traction" % (prefix, p_traction))
print("wrote %s + %s" % (primary, driver))
