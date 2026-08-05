"""Local 3-D strain/stress inside the 2-D honeycomb SG under a MACRO
plate strain -> per-Gauss-point _dehom.txt/.vtk + the plate 6x6.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       2 = plate; reads <name>.yaml
#   material_param      (3, 9) engineering constants;  angles (3,) deg
#   epsilon_bar         (6,) MACRO plate strain [e11 e22 2e12 k11 k22 2k12]
#   r                   sg_homo.plate_homo_2d result dict; C_eff (6, 6)
#   Gam, Sig            (E, Q, 6) Gauss strain/stress, SwiftComp order
#   <name>_plate_ABD.out, <name>_dehom.txt/.vtk, <name>_mesh.png  outputs
# ----------------------------------------------------------------------------
"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import plate_dehom_2d, export_gauss

############### User Input #################################
n_model = 2                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])

# the MACRO plate strain [e11, e22, 2e12, k11, k22, 2k12] -- YOURS, from
# the plate solution at the point of interest:
epsilon_bar = jnp.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.0])
############################################################

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                    angles=angles, n_model=n_model)
np.savetxt(name + "_plate_ABD.out", r["C_eff"], fmt="%16.8e",
           header="plate 6x6 [N11 N22 N12 M11 M22 M12] from the %dD SG %s"
                  % (r["n_sg"], name))
print("plate 6x6 diag:", np.array2string(np.diag(r["C_eff"]), precision=4))

Gam, Sig = plate_dehom_2d(r, epsilon_bar)
export_gauss(r, Gam, Sig, name + "_dehom")
for i, nm in enumerate(("xx", "yy", "zz", "yz", "xz", "xy")):
    print("  max|Sig_%s| = %.5e" % (nm, np.abs(Sig[..., i]).max()))
