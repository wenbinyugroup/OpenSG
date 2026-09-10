"""step01_make_decks.py -- the equivalent-plate deck (level 2) of an SG:
S8R, subdiv = 3, SS-1, pressure, plus its two-level cellmap.

    python -m opensg_solid.helper.Plate_deck_analysis.step01_make_decks \
           --sg <sg.yaml> --law <law_8x8.out> --q 7946.1 [--nx 5 --ny 5] \
           [--subdiv 3] [--etype S8R] [--bc ss1] [--out decks]

THE AREA TRANSFER LIVES HERE, NOT IN deck_core.  write_plate applies
its q verbatim as the plate *Dload (`ALL, P, -|q|`), so the caller owns
the transfer.  The 3-D deck's face pressure q acts only on the MATERIAL
part of the top plane (for a lattice/TPMS, a fraction of the cell), so
per cell it carries q * A_top, while a plate pressure acts on the full
footprint Px * Py.  Equal total force requires

    q_plate = q * A_top / (Px * Py)

and A_top is MEASURED from the SG mesh (boundary faces on z = zmax) --
never assumed -- exactly as the validated pressure_case twin
gen_plate_pressure.py does.  No moment enters the transfer: vertical
force, vertical offset to the reference surface, r x F = 0.

WHY subdiv = 3.  3 x 3 sub-elements per cell give every cell a real
central stencil for the first, second and mixed strain derivatives
(step07); 2 has no pure second derivatives.

In:  the SG yaml and the mesh-matched 8x8 refined .out
Out: <out>/<stem>_plate_<nx>x<ny>_<etype>.inp + _cellmap.csv (paths
     printed)"""
import argparse
import os
import sys

import numpy as np

from opensg_solid.helper.Plate_deck_analysis.deck_core import (
    _FACES, gate, read_sg, write_plate)


def a_top_of(sg):
    """Material area of one cell's top (z = zmax) surface, from the mesh.

    tet4: boundary triangles (faces used once) with all three corners
    on z = zmax -- gen_plate_pressure.py's measurement verbatim.
    hex8: the same once-used test on the boundary quads, each ring-
    ordered quad split into two triangles.  2-D section SG: the
    boundary edges of the quad section on z = zmax give the material
    top WIDTH, and the section is prismatic along the span, so times
    the span period Px that is one cell's top area.

    In:  sg dict -- deck_core.read_sg output
    Out: A_top float [length^2] per unit cell."""
    nd = np.asarray(sg["nodes"], float)
    cl = np.asarray(sg["cells"], np.int64)
    zmax = float(nd[:, 2].max())
    tol = 1e-6
    if sg.get("dim2"):
        ed = np.sort(cl[:, [[0, 1], [1, 2], [2, 3], [3, 0]]]
                     .reshape(-1, 2), axis=1)
        _, first, cnt = np.unique(ed, axis=0, return_index=True,
                                  return_counts=True)
        bnd = ed[first[cnt == 1]]
        top = bnd[np.all(np.abs(nd[bnd][:, :, 2] - zmax) < tol, axis=1)]
        w = float(np.abs(nd[top[:, 1], 1] - nd[top[:, 0], 1]).sum())
        return w * float(sg["span"][0])
    # ANY 3-D SG: one rule, shared with the load column.  Until 2026-09-04
    # this function had a tet4 branch and a hex8 branch and nothing for
    # tet10 -- a 10-column connectivity fell into the hex8 branch, was read
    # as hex corner indices, and returned 0.315363 for a cell whose real
    # top area is 0.234698: the plate deck was loaded 34 % too heavily and
    # every recovered field scaled by 1.34 (RF3 62,648 N against the 3-D
    # deck's 46,623 N).  _plate_face_loads_3d finds facets geometrically
    # for tet4/tet10/hex8/hex20/hex27 and its unit-pressure weights sum to
    # the facet area, so the smeared pressure and the load column now come
    # from the same facets by construction.
    from opensg_solid.sg_homo import _plate_face_loads_3d
    _, w = _plate_face_loads_3d(nd, cl, zmax, tol * max(
        1.0, float(np.ptp(nd[:, 2]))))
    return float(w.sum())

def main(argv=None):
    """Build the plate deck + cellmap for one SG and law.

    In:  argv list | None -- None means sys.argv[1:]
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sg", required=True,
                   help="the OpenSG solid SG yaml")
    p.add_argument("--law", required=True,
                   help="the homogenized 8x8 refined .out of that SG")
    p.add_argument("--nx", type=int, default=5, help="cells along 1")
    p.add_argument("--ny", type=int, default=5, help="cells along 2")
    p.add_argument("--subdiv", type=int, default=3,
                   help="elements per cell edge; subdiv = 3 is the"
                        " pipeline default -- 2 has no pure second"
                        " derivatives")
    p.add_argument("--etype", default="S8R", choices=["S4R", "S8R"],
                   help="S8R = quadratic THICK shell (default); never"
                        " S8R5/S9R5")
    p.add_argument("--bc", default="ss1", choices=["ss1", "clamped"])
    p.add_argument("--load", default="pressure",
                   choices=["gravity", "pressure"])
    p.add_argument("--q", type=float, default=None,
                   help="[Pa] the 3-D TOP-FACE pressure; the area"
                        " transfer to q_plate happens here")
    p.add_argument("--grav", type=float, default=9.81)
    p.add_argument("--out", default=".", help="output directory")
    a = p.parse_args(argv)
    if a.load == "pressure" and a.q is None:
        p.error("--q is required with --load pressure")

    sg = read_sg(a.sg)
    gate(sg, a.law)
    q_plate = a.q if a.q is not None else 0.0
    if a.load == "pressure":
        A_top = a_top_of(sg)
        Px, Py = float(sg["span"][0]), float(sg["span"][1])
        q_plate = a.q * A_top / (Px * Py)
        print("A_top %.6f  q_plate = q*A_top/(Px*Py) = %.6g * %.4f ="
              " %.4f Pa" % (A_top, a.q, A_top / (Px * Py), q_plate))
        print("total per cell: 3-D q*A_top = %.4f N   plate"
              " q_plate*Px*Py = %.4f N"
              % (a.q * A_top, q_plate * Px * Py))

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.sg))[0]
    inp = os.path.join(a.out, "%s_plate_%dx%d_%s.inp"
                       % (stem, a.nx, a.ny, a.etype.lower()))
    r = write_plate(sg, a.law, a.nx, a.ny, a.subdiv, inp, a.bc,
                    a.load, q_plate, a.grav, a.etype)
    print("deck    %s" % r["inp"])
    print("cellmap %s" % r["cellmap"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
