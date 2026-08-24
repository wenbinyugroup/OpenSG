"""opensg.helper -- the user-facing alias of opensg_solid.helper, so

    from opensg import helper
    helper.plate_mesh("my_sg.yaml", a=5.0, b=5.0)
    helper.plate_inp("my_sg_plate.msh", q=1.0)

works exactly as documented there (SG yaml -> plate .msh -> Abaqus
.inp).  One implementation only: everything lives in
opensg_solid.helper; this module just re-exports it under the short
import the workflows use.
"""
from opensg_solid.helper import plate_inp, plate_mesh   # noqa: F401
