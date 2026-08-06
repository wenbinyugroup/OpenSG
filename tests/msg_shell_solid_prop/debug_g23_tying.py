"""Tying-scheme sensitivity of the 2G23 finding: reading A/B x (mitc4_g23 | full).

Cross-section convention (as in the Timoshenko ring): only gamma_23 is MITC-tied,
gamma_13 stays untied -- 'mitc4_g23', the production default here too.  'full'
integrates both rows untied.  (For SURFACE 2-D SG elements both rows would be tied;
that case does not arise on the 1-D contour SG.)
Run (from this folder):  python debug_g23_tying.py
"""
import numpy as np

import opensg_shell.solid_props as SP
from opensg_shell.segment_indep import _surf_frame_batch
from opensg_shell import build_solid_bundle

_orig = SP.solid_macro_ops_batch


def _patched(Xq, e3q, xi, eta, cr, axx):
    BDe6, BGe6, dA = _orig(Xq, e3q, xi, eta, cr, axx)
    N, D1, D2, dA2, c = _surf_frame_batch(Xq, e3q, xi, eta, cr, axx)
    BGe6 = BGe6.copy()
    BGe6[:, 0, 3] = c["x31"] * c["y2"] + c["x21"] * c["y3"]
    BGe6[:, 1, 3] = c["x32"] * c["y2"] + c["x22"] * c["y3"]
    return BDe6, BGe6, dA


A_cell = np.pi
print("iso circle: C11 and C[2G23] vs tying scheme and 2G23-bucket reading")
print("%-24s %14s %14s" % ("", "C11", "C[2G23]"))
for tag, fn in (("A", _orig), ("B", _patched)):
    SP.solid_macro_ops_batch = fn
    for sch in ("mitc4_g23", "full"):
        B = build_solid_bundle("circle_iso_shell.yaml", cell_area=A_cell, shear=sch)
        C = np.asarray(B["C3D"])
        print("reading %s, %-12s %14.6e %14.6e" % (tag, sch, C[0, 0], C[3, 3]))
SP.solid_macro_ops_batch = _orig
