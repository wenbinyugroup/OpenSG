"""opensg_blade example -- the shortest Blade -> Timoshenko interface.

One call gives the Timoshenko stiffness K and the mass matrix M of a station
from the CURRENT state of the editable Blade object (pyNuMAD-style workflow:
edit the definition, update_blade, recompute -- no yaml files touched).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   blade_yaml   pyNuMAD blade file (windIO-v1 dialect)
#   station      st-id: 0-based station index or span r in [0, 1]
#   layer        design variable: layer whose thickness is scaled
#   scale        thickness multiplier for the edited design
# ----------------------------------------------------------------------------
"""
from opensg_shell import Blade, opensg_blade

############### User Input #################################
blade_yaml = "IEA-15-240-RWT.yaml"
station = 4                       # r = 0.3288
layer = "Spar_Cap_SS"             # design variable
scale = 1.2                       # thickness multiplier
############################################################

b = Blade(blade_yaml)

K, M = opensg_blade(b, station)              # baseline design
print("baseline  K:")
print(K)
print("baseline  M:")
print(M)

b.scale_layer_thickness(layer, scale)        # edit the definition ...
b.scale_layer_thickness("Spar_Cap_PS", scale)
b.update_blade()                             # ... regenerate ...
K2, M2 = opensg_blade(b, station)            # ... recompute

print("edited (spar caps x %.2f)  EA %.5g -> %.5g   EI2 %.5g -> %.5g"
      % (scale, K[0, 0], K2[0, 0], K[4, 4], K2[4, 4]))
