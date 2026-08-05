"""make_abaqus_inp.py -- generate the submit-ready Abaqus .inp files for the Garg
cylindrical-bending strip, with the MSG-RM 8x8 ABDG installed as the shell section.

The mesh/section/load/BC/output machinery lives in the CORE helper
``rm_plate.helper_inp_plate.write_plate_strip_inp``; this script supplies the
Garg cases: homogenize each laminate at the requested aspect ratio, compute the
closed-form and exact deflection predictions for the file header, and write
examples/garg/case<X>/garg_case<X>_S<S>.inp.

Submit manually (from a SPACE-FREE directory -- see the helper's gotcha note):
    abaqus job=garg_caseA_S10 interactive

Run:
    python examples/garg/make_abaqus_inp.py            # all three cases, S = 10
    python examples/garg/make_abaqus_inp.py --case caseC --S 4

Script variables (no functions here -- straight-line deck generation)
---------------------------------------------------------------------
a_          parsed CLI arguments: a_.case (laminate family or None = all), a_.S
            (aspect ratio a/h), a_.nel (span elements), a_.fsdt (deck flavour)
q0, a, p    load amplitude [Pa], span [m], wavenumber pi/a [1/m]
name, lay0  the LAYUPS key and its layup dict for the current case
h           laminate thickness for this S: h = a/S [m]
fr          per-ply thickness FRACTIONS of the family (thick_i / H)
thk         per-ply thicknesses re-scaled to this h: fr_i * h [m]
ang, mats   ply angles [deg] and material names of the case
r           rm_plate_msg result dict; r["ABDG"] is the mid-surface 8x8
D11, G11    bending and transverse-shear diagonals of the 8x8 (rows k11, 2g13)
w_msg       closed-form plate mid-span deflection q0/(p^4 D11) + q0/(p^2 G11) [m]
ex, zc, uvw exact solver instance and its profile (w_ex check value)
w_ex        exact 3-D mid-surface deflection amplitude [m]
hdr         header lines written into the .inp (case, S, both w predictions)
out         the deck path examples/garg/case<X>/garg_<X>_S<S>[_FSDT].inp
"""
import argparse
import os
import sys

import numpy as np

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
sys.path.insert(0, HERE)

from pagano_exact import ExactCyl
from opensg_solid.rm_plate_1D.msg_rm_plate import rm_plate_msg
from opensg_solid.rm_plate_1D.helper_inp_plate import (write_plate_strip_inp,
                                                write_plate_strip_inp_fsdt)
from garg_layups import MATERIAL_DB, LAYUPS, H

ap = argparse.ArgumentParser()
ap.add_argument("--case", default=None, help="caseA / caseB / caseC (default: all)")
ap.add_argument("--S", type=float, default=10.0, help="aspect ratio a/h (default 10)")
ap.add_argument("--nel", type=int, default=100, help="elements along the span")
ap.add_argument("--fsdt", action="store_true",
                help="write the community-FSDT composite-section variant instead "
                     "(_FSDT.inp: ply stack + lamina materials, Abaqus builds ABD "
                     "and its own transverse shear -- no MSG content)")
a_ = ap.parse_args()

q0 = 1.0e4
a = 1.0
p = np.pi / a

for name, lay0 in LAYUPS.items():
    if a_.case and name != a_.case:
        continue
    h = a / a_.S
    fr = [t / H for t in lay0["thick"]]
    thk = [f * h for f in fr]
    ang = lay0["angles"]; mats = lay0["mat_names"]

    r = rm_plate_msg(thk, ang, mats, MATERIAL_DB, fraction=0.5)
    D11 = float(r["ABDG"][3, 3]); G11 = float(r["ABDG"][6, 6])
    w_msg = q0 / (p ** 4 * D11) + q0 / (p ** 2 * G11)

    ex = ExactCyl(thk, ang, mats, MATERIAL_DB, a, q0=q0)
    zc, _, _, uvw = ex.profile(n_per_layer=41)
    w_ex = float(uvw[np.argmin(np.abs(zc)), 2])

    hdr = ["case %s: plies %s" % (name, ", ".join(
               "%s(%.4gmm/%g)" % (m, 1e3 * t, x) for m, t, x in zip(mats, thk, ang))),
           "S = a/h = %g,  h = %g m" % (a_.S, h),
           "PREDICTED mid-span |w|: MSG closed form %.6e m ; exact 3-D %.6e m"
           % (w_msg, w_ex),
           "  (check the .dat U3 at node NMID against these after the job)"]
    if a_.fsdt:
        out = os.path.join(HERE, name, "garg_%s_S%g_FSDT.inp" % (name, a_.S))
        write_plate_strip_inp_fsdt(out, thk, ang, mats, MATERIAL_DB, a=a, q0=q0,
                                   nel=a_.nel, header_lines=hdr + [
                                       "COMMUNITY FSDT variant: composite section, "
                                       "Abaqus ABD + its own shear"])
    else:
        out = os.path.join(HERE, name, "garg_%s_S%g.inp" % (name, a_.S))
        write_plate_strip_inp(out, r["ABDG"], a=a, q0=q0, nel=a_.nel, header_lines=hdr)
    print("wrote %s  (D11 %.4e, G11 %.4e; predicted |w| MSG %.5e, exact %.5e)"
          % (os.path.relpath(out, CC), D11, G11, w_msg, w_ex))
