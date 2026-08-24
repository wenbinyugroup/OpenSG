"""opensg_solid.helper -- the plate-benchmark builders: turn a periodic
SG into the full 3-D Abaqus plate model the HC_pm45 study used, as two
importable steps the user drives directly:

    from opensg import helper
    helper.plate_mesh("my_sg.yaml", a=5.0, b=5.0)   # -> <stem>_plate.msh
    helper.plate_inp("my_sg_plate.msh", q=1.0)      # -> <stem>_plate.inp

plate_mesh   SG yaml -> the tiled/extruded plate mesh (.msh, gmsh 2.2).
             A 3-D SG is TILED nx x ny cells to reach the requested
             (a, b) -- the opposite-face node sets must match (gated) so
             the tiles weld into one conforming mesh.  A 2-D SG (the
             HC_pm45 kind: width x thickness) is tiled across the width
             and EXTRUDED along the span.
plate_inp    .msh (+ the SG yaml's materials) -> the Abaqus deck:
             linear *Static step (nlgeom=NO -- the apples-to-apples
             lesson), clamped boundary, q = 1 MPa top-face pressure by
             default, element-face *Dload so nonuniform loads (where
             q,1 / q,2 drive the refined recovery) need no subroutine.

Both are plain functions -- no CLI, no main(); every input is an
argument and every output path is printed.
"""
from .plate_inp import plate_inp                      # noqa: F401
from .plate_mesh import plate_mesh                    # noqa: F401
