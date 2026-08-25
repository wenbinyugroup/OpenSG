"""gen_c3d10_deck.py -- the C3D10 (quadratic tet) AR5 reference deck.

The AR5 C3D4 reference (preovios_try_plate_AR5.inp, job AR5_3D) is a
constant-strain mesh, suspected of SMEARING the junction stress
concentration that the d2eps-driven recovery terms resolve.  This deck
is the SAME mesh -- same corner nodes, SAME ELEMENT IDS (element k =
the same parent tet) -- promoted to C3D10 by conforming midside nodes,
so the element-for-element pairing against the dehom fields carries
over unchanged while the element gains the linear-strain content a
junction needs.  Same clamped BC, same uniform q = 1 MPa down, same
position=CENTROIDAL S output (one value per element).

In:  preovios_try_plate_AR5.msh (the AR5 tiled plate mesh),
     preovios_try.yaml (materials)
Out: AR5_3D10.inp
"""
from datetime import datetime

from opensg import helper

print("gen_c3d10_deck: start %s"
      % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
info = helper.plate_inp(
    "preovios_try_plate_AR5.msh", yaml_path="preovios_try.yaml",
    q=1.0, order=2, out="AR5_3D10.inp",
    job_note="C3D10 junction-resolution reference: same tets, same"
             " element ids as the C3D4 AR5 deck, conforming midside"
             " nodes")
print("gen_c3d10_deck: %s" % info)
print("gen_c3d10_deck: end %s"
      % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
