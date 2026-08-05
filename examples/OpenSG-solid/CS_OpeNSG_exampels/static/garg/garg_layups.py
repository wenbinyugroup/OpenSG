"""garg_layups.py -- STEP 1 of the Garg (2023) application: the three benchmark
laminates as layup dictionaries, each turned into its own through-thickness 1-D SG
by the ``segment_plate`` mesh helper.

Garg, Aggarwal, Aitharaju et al., "Machine-learning-enabled recovery of 3D stresses
in laminated composites" (2023), sec. 3 configurations -- three laminate FAMILIES:

  caseA  [0/90/0]     Pagano graphite/epoxy, equal thirds        (their figs 3-5)
  caseB  [0/90/90/0]  AS4-type, equal quarters                   (their fig 6)
  caseC  [0/core/0]   sandwich: stiff faces 0.1h, soft core 0.8h (their fig 7)

Total thickness h = 0.1 m here (aspect ratio S = a/h = 10 with a = 1 m); the
homogenization check re-scales h for the other aspect ratios, since the layup
FRACTIONS are what define each family.

Writes, per case, into its own subfolder:
    examples/garg/case<X>/garg_<X>_sg.yaml     the 1-D SG mesh (5-noded quartic)
    examples/garg/case<X>/garg_<X>_sg.png      its mesh figure

Run:
    python examples/garg/garg_layups.py

Module variables (no functions here -- this is the shared data module)
----------------------------------------------------------------------
MATERIAL_DB   {name: {"E": [E1, E2, E3], "G": [G12, G13, G23],
              "nu": [nu12, nu13, nu23], "rho"}} in Pa; "pagano" is the classic
              25:1 graphite/epoxy ratio set, "as4"/"face" are the Garg sec.-3
              lamina, "core" the soft sandwich core
H             total laminate thickness [m] the LAYUPS entries are written at;
              only the ply FRACTIONS thick_i/H matter downstream (the benchmark
              re-scales to h = a/S)
LAYUPS        {case: {"mat_names", "thick" [m], "angles" [deg]}}, ply 0 at the
              BOTTOM face -- the single source every garg script reads
name, layup   the __main__ loop pair; out/png = the per-case SG yaml + figure
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = the first ancestor holding opensg_jax/ (location-robust: also
# runs from the examples/RM_OpenSG_pagano copy)
CC = HERE
while not os.path.isdir(os.path.join(CC, "src", "opensg_solid", "rm_plate_1D")):
    _up = os.path.dirname(CC)
    if _up == CC:
        raise RuntimeError("rm_plate repo root not found above " + __file__)
    CC = _up
sys.path.insert(0, os.path.join(CC, "src"))

from opensg_solid.rm_plate_1D.segment_plate import plate_sg_yaml, plot_plate_sg

# Garg sec. 3 material sets (Pagano's classic ratios; E_T = 1 GPa scale)
MATERIAL_DB = {
    "pagano": {"E": [25.0e9, 1.0e9, 1.0e9], "G": [0.5e9, 0.5e9, 0.2e9],
               "nu": [0.25, 0.25, 0.25], "rho": 1.0},
    "as4": {"E": [181.0e9, 10.3e9, 10.3e9], "G": [7.17e9, 7.17e9, 3.5e9],
            "nu": [0.28, 0.28, 0.28], "rho": 1.0},
    "face": {"E": [131.0e9, 10.34e9, 10.34e9], "G": [6.205e9, 6.205e9, 3.0e9],
             "nu": [0.22, 0.22, 0.22], "rho": 1.0},
    "core": {"E": [0.5776e9, 0.5776e9, 0.5776e9], "G": [0.1079e9, 0.1079e9, 0.1079e9],
             "nu": [0.0025, 0.0025, 0.0025], "rho": 1.0},
}

H = 0.1                     # total thickness [m]  ->  S = a/h = 10 with a = 1 m

LAYUPS = {
    "caseA": {"mat_names": ["pagano", "pagano", "pagano"],
              "thick": [H / 3, H / 3, H / 3],
              "angles": [0.0, 90.0, 0.0]},
    "caseB": {"mat_names": ["as4"] * 4,
              "thick": [H / 4] * 4,
              "angles": [0.0, 90.0, 90.0, 0.0]},
    "caseC": {"mat_names": ["face", "core", "face"],
              "thick": [0.1 * H, 0.8 * H, 0.1 * H],
              "angles": [0.0, 0.0, 0.0]},
}

if __name__ == "__main__":
    for name, layup in LAYUPS.items():
        out = os.path.join(HERE, name, "garg_%s_sg.yaml" % name[-1])
        plate_sg_yaml(out, layup, MATERIAL_DB, fraction=0.5)
        png = plot_plate_sg(out)
        print("wrote %s + %s" % (os.path.relpath(out, CC), os.path.basename(png)))
