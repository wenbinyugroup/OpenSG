"""Local strain/stress inside the 2-D honeycomb SG under a MACRO beam
state (the generalized-Timoshenko recovery chain of the msg-shell route,
sg_dehom.compute_fluctuations_gpu: zeroth-order V0 + first-order V1
warping) -> _dehom.txt/.vtk + the OpenSG dehom files _dehom_global.SM/.EM
(SG axes) and _dehom.U + the beam Timoshenko 6x6.  Every beam state is a
valid driver, the transverse shear included; a FORCE [F1 F2 F3 M1 M2 M3]
goes in as dehom_fields(r, beam_force=...) instead of epsilon_bar.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       1 = beam (Timoshenko/KKT); reads <name>.yaml
#   material_param      (3, 9) engineering constants;  angles (3,) deg
#   epsilon_bar         (6,) MACRO beam state [eps11 gam12 gam13 kap1 kap2
#                       kap3] (old EB 4-vector maps to slots 0, 3, 4, 5)
#   r                   sg_homo.plate_homo_2d result dict; C_eff (6, 6) Timo
#   Gam, Sig, U         (E, Q, 6/6/3) Gauss strain/stress/fluct. disp
#   <name>_beam_Timo.out, <name>_dehom.txt/.vtk, <name>_dehom_global.SM/.EM,
#   <name>_dehom.U, <name>_mesh.png     outputs
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
density = [1.6e-9, 1.6e-9, 2.7e-9]   # tonne/mm^3 per material (MPa-mm system)

# the MACRO beam state [eps11, gam12, gam13, kappa1, kappa2, kappa3]
# (Timoshenko rows; the old EB 4-vector [eps11 kappa1 kappa2 kappa3]
# maps to slots 0, 3, 4, 5) -- YOURS:
epsilon_bar = jnp.array([0.001, 0.0, 0.0, 0.0, 0.01, 0.0])
############################################################

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                    angles=angles, n_model=n_model, density=density)
np.savetxt(name + "_beam_Timo.out", r["C_eff"], fmt="%16.8e",
           header="beam Timoshenko 6x6 [eps11 gam12 gam13 kappa1 kappa2"
                  " kappa3] (Beam_solid KKT engine) from the %dD SG %s"
                  % (r["n_sg"], name))
print("beam 6x6 diag:", np.array2string(np.diag(r["C_eff"]), precision=4))

Gam, Sig, U = dehom_fields(r, epsilon_bar)
# the clouds are SG-global: <name>_dehom_global.SM/.EM (the bare .SM/.EM
# names are the ply-axes files of `opensg <name>.yaml D --material`)
export_gauss(r, Gam, Sig, name + "_dehom", U_eqd=U, frame_kind="global")
for i, nm in enumerate(("xx", "yy", "zz", "yz", "xz", "xy")):
    print("  max|Sig_%s| = %.5e" % (nm, np.abs(Sig[..., i]).max()))
