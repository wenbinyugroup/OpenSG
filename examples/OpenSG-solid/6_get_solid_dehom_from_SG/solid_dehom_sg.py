"""Local strain/stress inside a 3-D SG under a MACRO 3-D strain ->
_dehom.txt/.vtk + the OpenSG dehom files _dehom.SM/.EM/.U + the
effective 6x6.  (Same OPEN ITEM as example 5: the bundled
SW_2UC_45.sc's element convention is unconfirmed.)

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       3 = 3-D elastic; reads <name>.yaml (materials as-is)
#   epsilon_bar         (6,) MACRO 3-D strain [exx eyy ezz 2eyz 2exz 2exy]
#   r                   sg_homo.plate_homo_2d result dict; C_eff (6, 6)
#   Gam, Sig, U         (E, Q, 6/6/3) Gauss strain/stress/fluct. disp
#   <name>_solid_C.out, <name>_dehom.txt/.vtk/.SM/.EM/.U,
#   <name>_mesh.png     outputs
# ----------------------------------------------------------------------------
"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields, export_gauss

############### User Input #################################
n_model = 3                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "SW_2UC_45"             # reads <name>.yaml

# the MACRO 3-D strain [exx, eyy, ezz, 2eyz, 2exz, 2exy] -- YOURS:
epsilon_bar = jnp.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
############################################################

r = plate_homo_2d(name + ".yaml", n_model=n_model)
np.savetxt(name + "_solid_C.out", r["C_eff"], fmt="%16.8e",
           header="3-D effective 6x6 C (xx yy zz yz xz xy) from the %dD"
                  " SG %s" % (r["n_sg"], name))
print("6x6 diag:", np.array2string(np.diag(r["C_eff"]), precision=4))

Gam, Sig, U = dehom_fields(r, epsilon_bar)
export_gauss(r, Gam, Sig, name + "_dehom", U_eqd=U)
for i, nm in enumerate(("xx", "yy", "zz", "yz", "xz", "xy")):
    print("  max|Sig_%s| = %.5e" % (nm, np.abs(Sig[..., i]).max()))
