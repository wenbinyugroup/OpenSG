"""yu_layups.py -- the three cylindrical-bending cases of Yu-Hodges-Volovoi,
"Asymptotically accurate 3-D recovery from Reissner-like composite plate finite
elements", Computers and Structures 81 (2003) 439-454, section 6.1.

Paper problem statement (their Fig. 2 + Eq. 67): plate of width L = 4 in along x1,
INFINITE in x2 (cylindrical bending), thickness h = 1 in -> aspect ratio L/h = 4
(the deliberately THICK validation case), simply supported, loaded on BOTH faces:

    s3 = b3 = (p0/2) sin(pi x1 / L),   s_alpha = b_alpha = 0

i.e. sigma_33(x, h) = +p0/2 sin(px) on the top face and sigma_33(x, 0) =
-p0/2 sin(px) on the bottom face (the bottom traction b3 acts on the -z normal).
Results are normalized as sigma_bar = sigma / p0 (their Eq. 68) -- p0 = 1 here so
raw = normalized.

Material (all plies; psi units, the classic Pagano ratio set):
    EL = 25e6, ET = 1e6, GLT = 0.5e6, GTT = 0.2e6, nu_LT = nu_TT = 0.25

The three layups (STACKING FROM BOTTOM TO TOP, equal-thickness plies):
    case1  [15/-15]            antisymmetric angle ply
    case2  [30/-30/-30/30]     symmetric angle ply
    case3  [0.5/90.5/90.5/0.5] symmetric nearly cross ply (Yu perturbs the angles
                               by 0.5 deg only because Sutyrin's Mathematica exact
                               code could not do cross-ply; our state-space exact
                               solver has no such restriction, but we keep HIS
                               angles to reproduce HIS curves)

All three are SHEAR-COUPLED (angle plies): the exact solution has nonzero v,
sigma_12, sigma_23 -- this is Pagano 1970 (JCM 4:330, shear coupling), the
generalization the state-space ExactCyl was built for.

Writes, per case, into its own subfolder:
    examples/yu2003/case<N>/yu_<N>_sg.yaml    the through-thickness 1-D SG mesh
    examples/yu2003/case<N>/yu_<N>_sg.png     its mesh figure

Run:
    python examples/yu2003/yu_layups.py

Module variables (no functions here -- this is the shared data module)
----------------------------------------------------------------------
MATERIAL_DB   {"yu": {"E": [EL, ET, ET], "G": [GLT, GLT, GTT],
              "nu": [0.25]*3, "rho"}} in psi -- the single Pagano-ratio material
              every ply uses
L_SPAN        plate width along x1: 4.0 [in]
H             total thickness: 1.0 [in]  ->  S = L/h = 4 (FIXED in this paper)
P0            load amplitude p0 [psi]; 1.0 so all stresses are already the
              paper's normalized sigma_bar = sigma/p0
LAYUPS        {case: {"mat_names", "thick" [in], "angles" [deg]}}, ply 0 at the
              BOTTOM face -- the single source every yu2003 script reads
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

MATERIAL_DB = {
    "yu": {"E": [25.0e6, 1.0e6, 1.0e6], "G": [0.5e6, 0.5e6, 0.2e6],
           "nu": [0.25, 0.25, 0.25], "rho": 1.0},
}

L_SPAN = 4.0                # plate width along x1 [in]
H = 1.0                     # total thickness [in]  ->  L/h = 4
P0 = 1.0                    # load amplitude [psi]; results = sigma_bar directly

LAYUPS = {
    "case1": {"mat_names": ["yu", "yu"],
              "thick": [H / 2, H / 2],
              "angles": [15.0, -15.0]},
    "case2": {"mat_names": ["yu"] * 4,
              "thick": [H / 4] * 4,
              "angles": [30.0, -30.0, -30.0, 30.0]},
    "case3": {"mat_names": ["yu"] * 4,
              "thick": [H / 4] * 4,
              "angles": [0.5, 90.5, 90.5, 0.5]},
}

if __name__ == "__main__":
    for name, layup in LAYUPS.items():
        out = os.path.join(HERE, name, "yu_%s_sg.yaml" % name[-1])
        plate_sg_yaml(out, layup, MATERIAL_DB, fraction=0.5)
        png = plot_plate_sg(out)
        print("wrote %s + %s" % (os.path.relpath(out, CC), os.path.basename(png)))
