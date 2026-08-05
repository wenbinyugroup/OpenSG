"""Plate properties (6x6 ABD) of the 2-D honeycomb SG.  Input is the
YAML; the one-time SwiftComp conversion is the separate helper:
    python -m opensg_solid.rm_plate_1D.helper.sc_to_yaml RHC_SW_2UC_45.sc

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   n_model, name       2 = plate; reads <name>.yaml
#   material_param      (3, 9) engineering constants per material
#   angles              (3,) deg per material (0.0 = none)
#   r                   sg_homo.plate_homo_2d result dict
#   r["C_eff"]          (6, 6) plate ABD [N11 N22 N12 M11 M22 M12]
#   <name>_plate_ABD.out, <name>_mesh.png   the outputs
# ----------------------------------------------------------------------------
"""
import numpy as np
import jax.numpy as jnp

from opensg_solid.sg_homo import plate_homo_2d

############### User Input #################################
n_model = 2                    # 1: Beam; 2: Plate; 3: 3D elastic
name = "RHC_SW_2UC_45"         # reads <name>.yaml

# engineering constants per material (the .sc blocks carry PRE-ROTATED
# ply C's; this override rebuilds from E,G,nu + the angle instead):
material_param = jnp.array([
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),      # ply +45
    (108e3, 8e3, 8e3, 4e3, 4e3, 3e3, 0.32, 0.32, 0.30),      # ply -45
    (69e3, 69e3, 69e3, 26.54e3, 26.54e3, 26.54e3, 0.30, 0.30, 0.30)])
angles = jnp.array([45.0, -45.0, 0.0])   # deg per material; 0.0 = none
############################################################

r = plate_homo_2d(name + ".yaml", material_param=material_param,
                    angles=angles, n_model=n_model)
np.set_printoptions(precision=5)
print("plate 6x6  [N11 N22 N12 M11 M22 M12]:")
print(r["C_eff"])
np.savetxt(name + "_plate_ABD.out", r["C_eff"], fmt="%16.8e",
           header="plate 6x6 [N11 N22 N12 M11 M22 M12] from the %dD SG %s"
                  % (r["n_sg"], name))
print("wrote %s_plate_ABD.out (+ %s_mesh.png)" % (name, name))
