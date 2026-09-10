"""step04_solid_inp.py -- the 3-D FEA reference deck: the SG tiled
nx x ny, p-refined to C3D10, same SS-1 edges and pressure as the plate.

    python -m opensg_solid.helper.Plate_deck_analysis.step04_solid_inp \
           --sg <sg.yaml> --law <law_8x8.out> --q 7946.1 \
           [--nx 5 --ny 5] [--order 2] [--bc ss1] [--out decks]

WHY PRESSURE AND NOT GRAVITY.  OpenSG's dehom load columns are FACE
TRACTION columns (Yu Eqs. 29/45/64) -- there is no body-force column --
so with pressure the load treatment in the recovery is exact rather
than approximated.  The q here is the RAW face pressure on the TOP
MATERIAL SURFACE (write_solid finds those faces set-wise; void regions
carry nothing) -- the area transfer to the plate deck lives in step01,
never here.  Mirrors the validated pressure_case twin
gen_3d_pressure.py: read_sg, gate, tile, to_c3d10, write_solid.

In:  the tet4 (or hex8, --order 1) SG yaml; the 8x8 .out feeds only the
     gate diagnostic line
Out: <out>/<stem>_3d_<nx>x<ny>_<etype>.inp (path printed)"""
import argparse
import os
import sys

from opensg_solid.helper.Plate_deck_analysis.deck_core import (
    gate, read_sg, tile, to_c3d10, write_solid)


def main(argv=None):
    """Build the tiled 3-D reference deck for one SG.

    In:  argv list | None -- None means sys.argv[1:]
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sg", required=True,
                   help="the OpenSG solid SG yaml")
    p.add_argument("--law", default=None,
                   help="the 8x8 .out (gate diagnostic line only)")
    p.add_argument("--nx", type=int, default=5, help="cells along 1")
    p.add_argument("--ny", type=int, default=5, help="cells along 2")
    p.add_argument("--order", type=int, default=2, choices=[1, 2],
                   help="2 = C3D10 p-refinement (tet4 SG only,"
                        " default); 1 = C3D4/C3D8I")
    p.add_argument("--bc", default="ss1", choices=["ss1", "clamped"])
    p.add_argument("--load", default="pressure",
                   choices=["gravity", "pressure"])
    p.add_argument("--q", type=float, default=None,
                   help="[Pa] on the top MATERIAL faces, --load"
                        " pressure")
    p.add_argument("--grav", type=float, default=9.81)
    p.add_argument("--out", default=".", help="output directory")
    a = p.parse_args(argv)
    if a.load == "pressure" and a.q is None:
        p.error("--q is required with --load pressure")

    sg = read_sg(a.sg)
    gate(sg, a.law)
    if sg.get("dim2"):
        sys.exit("this SG is a 2-D cross-section -- build its 3-D deck"
                 " with deck_core's `solid2d` subcommand (tile across"
                 " the width + extrude the span), not step04")
    k = sg["cells"].shape[1]
    if a.order == 2 and k != 4:
        sys.exit("--order 2 (C3D10) needs a tet4 SG, got %d-node" % k)
    etype = {1: {4: "C3D4", 8: "C3D8I"}[k], 2: "C3D10"}[a.order]

    nodes, cells, mats = tile(sg, a.nx, a.ny)
    if a.order == 2:
        nodes, cells = to_c3d10(nodes, cells)
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.sg))[0]
    inp = os.path.join(a.out, "%s_3d_%dx%d_%s.inp"
                       % (stem, a.nx, a.ny, etype.lower()))
    write_solid(sg, nodes, cells, mats, inp, a.bc, a.load,
                a.q if a.q is not None else 0.0, a.grav, etype)
    print("deck    %s" % inp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
