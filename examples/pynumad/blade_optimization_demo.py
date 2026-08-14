"""Blade-class optimization demo -- every editable property, one table.

The pyNuMAD-style workflow: one Blade object holds the whole editable
definition; each design move mutates it (layup, materials, geometry, webs,
or one station's Section) and asks for the station Timoshenko matrix -- no
yaml editing anywhere.  Each move runs on a FRESH Blade so the table rows
are independent changes against the same baseline.

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   blade_yaml   pyNuMAD blade file (windIO-v1 dialect)
#   station      st-id: 0-based station index (4 -> r = 0.3288)
#   MOVES        the design moves: (label, edit function on a fresh Blade)
#   K0, m0       baseline Timoshenko 6x6 and mass/span at the station
#   K, mp        each move's 6x6 and mass/span; the table prints % vs base
# ----------------------------------------------------------------------------
"""
import numpy as np

import opensg

############### User Input #################################
blade_yaml = "IEA-15-240-RWT.yaml"
station = 4                              # r = 0.3288
############################################################


def caps_thickness(b):
    """LAYUP, spanwise: spar-cap thickness distribution x1.2 (both sides)."""
    b.scale_layer_thickness("Spar_Cap_SS", 1.2)
    b.scale_layer_thickness("Spar_Cap_PS", 1.2)
    b.update_blade()


def skin_angle(b):
    """LAYUP, ply angle: rotate the full-surface triax skin to +20 deg."""
    b.layers["Shell_skin"]["fiber_orientation"] = {
        "grid": [0.0, 1.0], "values": [20.0, 20.0]}
    b.update_blade()


def soften_triax(b):
    """MATERIAL: glass_triax stiffness halved (every E component)."""
    E = b.materials["glass_triax"]["E"]
    b.set_material("glass_triax", E=[0.5 * float(v) for v in E])
    b.update_blade()


def chord_up(b):
    """GEOMETRY: chord distribution x1.1 (section grows self-similarly)."""
    b.scale_chord(1.1)
    b.update_blade()


def twist_plus5(b):
    """GEOMETRY: +5 deg twist everywhere (file unit is radians).  The 6x6
    is computed in the LOCAL chord frame, so its diagonal is twist-
    invariant by design -- twist rides to BeamDyn as frame metadata."""
    b.set_twist([float(v) + np.deg2rad(5.0) for v in b.twist["values"]])
    b.update_blade()


def axis_shift(b):
    """GEOMETRY: reference (pitch) axis moved 5% chord toward the TE."""
    b.offset["values"] = [float(v) + 0.05 for v in b.offset["values"]]
    b.update_blade()


def web_filler(b):
    """WEBS: shear-web filler thickness x2 (a spanwise layer like any other)."""
    b.scale_layer_thickness("web0_filler", 2.0)
    b.scale_layer_thickness("web1_filler", 2.0)
    b.update_blade()


def section_caps(b):
    """SECTION: caps x1.2 at THIS station only (live override, no
    definition change -- blade.section(st) + scale_thickness)."""
    S = b.section(station)
    S.scale_thickness(1.2, region="LP_SPAR")
    S.scale_thickness(1.2, region="HP_SPAR")


MOVES = [
    ("baseline", None),
    ("caps t x1.2 (layup, spanwise)", caps_thickness),
    ("skin plies -> +20 deg (layup)", skin_angle),
    ("glass_triax E x0.5 (material)", soften_triax),
    ("chord x1.1 (geometry)", chord_up),
    ("twist +5 deg (geometry)", twist_plus5),
    ("pitch axis +5% chord (geometry)", axis_shift),
    ("web fillers t x2 (webs)", web_filler),
    ("caps t x1.2 THIS station (Section)", section_caps),
]

print("stations:", " ".join("%.4f" % r
                            for r in opensg.read(blade_yaml).stations))
K0 = m0 = None
print("%-36s %7s %7s %7s %7s %7s %7s %7s"
      % ("design move", "EA%", "GA2%", "GA3%", "GJ%", "EI2%", "EI3%",
         "mass%"))
for label, move in MOVES:
    b = opensg.read(blade_yaml)          # fresh definition per design point
    if move is not None:
        move(b)
    R = b.timo(station)
    K, mp = np.asarray(R["Timo"]), float(R["info"]["mpus"])
    if K0 is None:
        K0, m0 = K, mp
        print("%-36s %7s %7s %7s %7s %7s %7s %7s" % ((label,) + ("--",) * 7))
        print("  baseline: EA %.4g  GA2 %.4g  GA3 %.4g  GJ %.4g  EI2 %.4g"
              "  EI3 %.4g  m %.4g" % (K[0, 0], K[1, 1], K[2, 2], K[3, 3],
                                      K[4, 4], K[5, 5], mp))
        continue
    pc = lambda i: 100.0 * (K[i, i] - K0[i, i]) / K0[i, i]
    print("%-36s %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f"
          % (label, pc(0), pc(1), pc(2), pc(3), pc(4), pc(5),
             100.0 * (mp - m0) / m0))


