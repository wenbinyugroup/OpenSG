"""Local strain/stress inside the 2-D honeycomb SG under a MACRO beam
state (Timoshenko recovery chain of the merged Beam_solid KKT engine)
-> _dehom.txt/.vtk + the OpenSG dehom files _dehom.SM/.EM/.U + the
beam Timoshenko 6x6.  Valid drivers = classical-channel states
(ext/twist/bend + combinations; see sg_dehom.compute_fluctuations_gpu).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       1 = beam (Timoshenko/KKT); reads <name>.yaml
#   material_param      (3, 9) engineering constants;  angles (3,) deg
#   epsilon_bar         (6,) MACRO beam state [eps11 gam12 gam13 kap1 kap2
#                       kap3] (old EB 4-vector maps to slots 0, 3, 4, 5)
#   r                   sg_homo.plate_homo_2d result dict; C_eff (6, 6) Timo
#   Gam, Sig, U         (E, Q, 6/6/3) Gauss strain/stress/fluct. disp
#   <name>_beam_Timo.out, <name>_dehom.txt/.vtk/.SM/.EM/.U,
#   <name>_mesh.png     outputs
# ----------------------------------------------------------------------------
"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d
from opensg_solid.sg_dehom import dehom_fields, export_gauss

############### User Input #################################
n_model = 1                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])

# the MACRO beam state [eps11, gam12, gam13, kappa1, kappa2, kappa3]
# (Timoshenko rows; the old EB 4-vector [eps11 kappa1 kappa2 kappa3]
# maps to slots 0, 3, 4, 5) -- YOURS:
epsilon_bar = jnp.array([0.001, 0.0, 0.0, 0.0, 0.01, 0.0])
############################################################

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                    angles=angles, n_model=n_model)
np.savetxt(name + "_beam_Timo.out", r["C_eff"], fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2"
                  " kappa3] (Beam_solid KKT engine) from the %dD SG %s"
                  % (r["n_sg"], name))
print("beam 6x6 diag:", np.array2string(np.diag(r["C_eff"]), precision=4))

Gam, Sig, U = dehom_fields(r, epsilon_bar)
export_gauss(r, Gam, Sig, name + "_dehom", U_eqd=U)
for i, nm in enumerate(("xx", "yy", "zz", "yz", "xz", "xy")):
    print("  max|Sig_%s| = %.5e" % (nm, np.abs(Sig[..., i]).max()))
