"""Effective 3-D elastic 6x6 C of a 3-D SG (xx yy zz yz xz xy order),
using the .sc material blocks as-is (no override -> the pre-rotated ply
C's stored in the file).

OPEN ITEM: the bundled SW_2UC_45.sc carries 9 nonzero connectivity
slots per element (not tet4/hex8/tet10 under any known SwiftComp
slicing) -- the converter refuses it until that convention is
confirmed.  Point `name` at a valid 3-D SG yaml to run.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       3 = 3-D elastic; reads <name>.yaml (materials as-is)
#   r                   sg_homo.plate_homo_2d result dict
#   r["C_eff"]          (6, 6) effective C (xx yy zz yz xz xy)
#   <name>_solid_C.out, <name>_mesh.png   the outputs
# ----------------------------------------------------------------------------
"""
import numpy as np

from opensg_solid.sg_homo import plate_homo_2d

############### User Input #################################
n_model = 3                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "SW_2UC_45"             # reads <name>.yaml
############################################################

r = plate_homo_2d(name + ".yaml", n_model=n_model)
np.set_printoptions(precision=5)
print("3-D effective 6x6 C  (xx yy zz yz xz xy):")
print(r["C_eff"])
np.savetxt(name + "_solid_C.out", r["C_eff"], fmt="%16.8e",
           header="3-D effective 6x6 C (xx yy zz yz xz xy) from the %dD"
                  " SG %s" % (r["n_sg"], name))
print("wrote %s_solid_C.out (+ %s_mesh.png)" % (name, name))
