"""Beam properties of the 2-D honeycomb SG cross-section.  TIMOSHENKO
6x6 [eps11 gam12 gam13 kappa1 kappa2 kappa3] -- the Beam_solid KKT
engine merged into sg_homo/sg_assembly (V0 EB solve under the 4
rigid-body Lagrange constraints + the l-chain V1s solve); the classical
EB 4x4 [eps11 kappa1 kappa2 kappa3] rides along in r["C_eff_EB"].

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       1 = beam (Timoshenko/KKT); reads <name>.yaml
#   material_param      (3, 9) engineering constants per material
#   angles              (3,) deg per material (0.0 = none)
#   r                   sg_homo.plate_homo_2d result dict
#   r["C_eff"]          (6, 6) Timo [eps11 gam12 gam13 kap1 kap2 kap3]
#   r["C_eff_EB"]       (4, 4) classical EB [eps11 kap1 kap2 kap3]
#   <name>.out          Timoshenko stiffness + compliance (SwiftComp .K
#                       layout, timed) -- written by the solver
#   <name>_mesh.png     the mesh figure
# ----------------------------------------------------------------------------
"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

############### User Input #################################
n_model = 1                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])
density = [1.6e-9, 1.6e-9, 2.7e-9]   # tonne/mm^3 per material (MPa-mm system):
                                     # CFRP plies, aluminum core -- edit to yours
############################################################

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                    angles=angles, n_model=n_model, density=density)
np.set_printoptions(precision=5)
print("beam 6x6  [eps11 gam12 gam13 kappa1 kappa2 kappa3]  (Timoshenko):")
print(r["C_eff"])
print("beam 4x4  [eps11 kappa1 kappa2 kappa3]  (EB, same run):")
print(r["C_eff_EB"])
print("mass 6x6  (VABS frame; mass/span %.5g, center (%.4g, %.4g)):"
      % (r["mass_info"]["mpus"], *r["mass_info"]["mass_center"]))
print(r["Mass"])
print("wrote %s.out (Timoshenko stiffness + compliance + 6x6 mass, .K layout)"
      " + %s_mesh.png" % (name, name))
