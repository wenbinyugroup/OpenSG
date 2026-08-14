"""Blade-class optimization demo -- edit the layup ON THE OBJECT, re-homogenize.

The pyNuMAD-style workflow: one Blade object holds the whole editable
definition; each design point mutates it (here: spar-cap thickness scale)
and asks for the station Timoshenko matrix -- no yaml editing anywhere.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   blade_yaml   pyNuMAD blade file (windIO-v1 dialect)
#   station      st-id: 0-based station index or span r in [0, 1]
#   layers       the design variables: layers whose thickness is scaled
#   scales       thickness multipliers to sweep
# ----------------------------------------------------------------------------
"""
from opensg_shell import Blade

############### User Input #################################
blade_yaml = "IEA-15-240-RWT.yaml"
station = 4                              # r = 0.3288
layers = ["Spar_Cap_SS", "Spar_Cap_PS"]  # design variables
scales = [0.8, 1.0, 1.2]                 # thickness multipliers
############################################################

print("stations:", " ".join("%.4f" % r for r in Blade(blade_yaml).stations))
print("%-6s  %12s  %12s  %12s  %10s" % ("scale", "EA", "EI2(flap)",
                                        "EI3(edge)", "mass/span"))
for s in scales:
    b = Blade(blade_yaml)                # fresh definition per design point
    for nm in layers:
        b.scale_layer_thickness(nm, s)
    b.update_blade()
    R = b.timo(station)
    K = R["Timo"]
    print("%-6.2f  %12.5g  %12.5g  %12.5g  %10.5g"
          % (s, K[0, 0], K[4, 4], K[5, 5], R["info"]["mpus"]))
