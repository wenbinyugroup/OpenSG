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

And the COMPARISON side (the exact-pairing doctrine: one shell element
= one SG cell; the tiled 3-D mesh's tets ARE the SG's tets):

sg_shear_refined_gp_sampling   sg_centroids / elem_mean_sm /
             elem_mean_u / load_3d_csv / pair_cell / write_path_dat --
             element-by-element pairing of the 3-D FEA against the
             dehom Gauss fields (C3D4 single IP vs the same tet's
             Gauss mean).
sg_read_abaqus_rpt_for_elemental_stress_output   abaqus_to_sm /
             abaqus_to_u -- the 3-D reference's elemental output
             rewritten in the dehom .SM / .U layout so every
             .SM-consuming tool reads it unchanged.

All are plain functions -- no CLI, no main(); every input is an
argument and every output path is printed.
"""
from .plate_inp import plate_inp                      # noqa: F401
from .plate_mesh import plate_mesh                    # noqa: F401
from .sg_read_abaqus_rpt_for_elemental_stress_output import (  # noqa: F401
    abaqus_to_sm, abaqus_to_u)
from .sg_shear_refined_gp_sampling import (           # noqa: F401
    elem_mean_sm, elem_mean_u, load_3d_csv, pair_cell, ply_region,
    sg_centroids, write_path_dat)
