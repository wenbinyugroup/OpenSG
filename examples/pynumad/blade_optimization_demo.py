"""Blade-class optimization demo -- every editable property, one move at a time.

The pyNuMAD-style workflow: one Blade object holds the whole editable
definition; each design move mutates it (layup, materials, geometry, webs,
or one station's Section) and asks for the station Timoshenko matrix -- no
yaml editing anywhere.  Every move runs on a FRESH blade and its result
block prints IMMEDIATELY after that move's solve, so the terminal reads as
a running optimization log.  The solves go through the in-memory pynumad
homogenizer (no files written).

# ----------------------------------------------------------------------------
# ALL VARIABLES USED IN THIS SCRIPT
# ----------------------------------------------------------------------------
#   blade_yaml           pyNuMAD blade file (windIO-v1 dialect)
#   station_id           st-id: 0-based station index (4 -> r = 0.3288)
#   DIAG_LABELS/UNITS    the VABS diagonal names and units
#   DESIGN_MOVES         (label, edit function on a fresh blade) per move
#   blade                the fresh editable Blade of the current move
#   stiffness            (6,6) Timoshenko matrix of the current move
#   mass_per_span        mass/span [kg/m] of the current move
#   stiffness_baseline,  the un-edited design the % columns compare to
#   mass_baseline
# ----------------------------------------------------------------------------
"""
import numpy as np

import opensg
from opensg_shell.pynumad.sg_homo import timo

############### User Input #################################
blade_yaml = "IEA-15-240-RWT.yaml"
station_id = 4                           # r = 0.3288
############################################################

DIAG_LABELS = ("EA", "GA2", "GA3", "GJ", "EI2", "EI3")
DIAG_UNITS = ("N", "N", "N", "N.m2", "N.m2", "N.m2")


def scale_spar_caps(blade):
    """LAYUP, spanwise: spar-cap thickness distribution x1.2 (both sides)."""
    blade.scale_layer_thickness("Spar_Cap_SS", 1.2)
    blade.scale_layer_thickness("Spar_Cap_PS", 1.2)
    blade.update_blade()


def rotate_skin_plies(blade):
    """LAYUP, ply angle: rotate the full-surface triax skin to +20 deg."""
    blade.layers["Shell_skin"]["fiber_orientation"] = {
        "grid": [0.0, 1.0], "values": [20.0, 20.0]}
    blade.update_blade()


def soften_glass_triax(blade):
    """MATERIAL: glass_triax stiffness halved (every E component)."""
    E_components = blade.materials["glass_triax"]["E"]
    blade.set_material("glass_triax",
                       E=[0.5 * float(E) for E in E_components])
    blade.update_blade()


def scale_chord_up(blade):
    """GEOMETRY: chord distribution x1.1 (section grows self-similarly)."""
    blade.scale_chord(1.1)
    blade.update_blade()


def add_twist(blade):
    """GEOMETRY: +5 deg twist everywhere (file unit is radians).  The 6x6
    is computed in the LOCAL chord frame, so its diagonal is twist-
    invariant by design -- twist rides to BeamDyn as frame metadata."""
    blade.set_twist([float(v) + np.deg2rad(5.0)
                     for v in blade.twist["values"]])
    blade.update_blade()


def shift_pitch_axis(blade):
    """GEOMETRY: reference (pitch) axis moved 5% chord toward the TE."""
    blade.offset["values"] = [float(v) + 0.05
                              for v in blade.offset["values"]]
    blade.update_blade()


def thicken_web_fillers(blade):
    """WEBS: shear-web filler thickness x2 (spanwise layers like any other)."""
    blade.scale_layer_thickness("web0_filler", 2.0)
    blade.scale_layer_thickness("web1_filler", 2.0)
    blade.update_blade()


def scale_caps_this_station(blade):
    """SECTION: caps x1.2 at THIS station only (live override via
    blade.section -- no definition change, thickness applied verbatim)."""
    section = blade.section(station_id)
    section.scale_thickness(1.2, region="LP_SPAR")
    section.scale_thickness(1.2, region="HP_SPAR")


DESIGN_MOVES = [
    ("spar-cap thickness x1.2 (layup, spanwise)", scale_spar_caps),
    ("skin plies -> +20 deg (layup, ply angle)", rotate_skin_plies),
    ("glass_triax E x0.5 (material)", soften_glass_triax),
    ("chord x1.1 (geometry)", scale_chord_up),
    ("twist +5 deg (geometry, chord-frame invariant)", add_twist),
    ("pitch axis +5% chord (geometry)", shift_pitch_axis),
    ("web fillers x2 (webs)", thicken_web_fillers),
    ("spar caps x1.2 THIS station only (Section)", scale_caps_this_station),
]


def print_move_block(move_label, stiffness, mass_per_span,
                     stiffness_baseline=None, mass_baseline=None):
    """One immediately-printed result block for a design move.

    In:  move_label str; stiffness (6,6); mass_per_span float [kg/m];
         stiffness_baseline/mass_baseline -- None for the baseline block
         (absolute values only), else the reference for the % column.
    Out: prints to the terminal; returns None."""
    print("-" * 72, flush=True)
    print(move_label, flush=True)
    for i, (label, unit) in enumerate(zip(DIAG_LABELS, DIAG_UNITS)):
        line = "   %-4s = %12.5e %-5s" % (label, stiffness[i, i], unit)
        if stiffness_baseline is not None:
            change = 100.0 * (stiffness[i, i] - stiffness_baseline[i, i]) \
                / stiffness_baseline[i, i]
            line += "  (%+7.2f %%)" % change
        print(line, flush=True)
    line = "   mass/span = %10.4f kg/m" % mass_per_span
    if mass_baseline is not None:
        line += "       (%+7.2f %%)" % (100.0 * (mass_per_span
                                                 - mass_baseline)
                                        / mass_baseline)
    print(line, flush=True)


blade = opensg.read(blade_yaml)
print("stations:", " ".join("%.4f" % r for r in blade.stations), flush=True)
stiffness_baseline, mass_matrix, section_info = timo(blade, station_id,
                                                     full=True)
mass_baseline = float(section_info["mpus"])
print_move_block("baseline (station %d, r = %.4f)"
                 % (station_id, blade.resolve(station_id)),
                 stiffness_baseline, mass_baseline)

for move_label, apply_move in DESIGN_MOVES:
    blade = opensg.read(blade_yaml)      # fresh definition per design point
    apply_move(blade)
    stiffness, mass_matrix, section_info = timo(blade, station_id,
                                                full=True)
    print_move_block(move_label, stiffness, float(section_info["mpus"]),
                     stiffness_baseline, mass_baseline)
