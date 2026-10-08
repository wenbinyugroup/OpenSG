"""fea_cell.py -- the 3-D FEA fields of ANY plate cell, chosen by the
SAME (di, dj) offset the ff-side uses, saved as .SM/.U suffixed with
the CELL TAG.

    python -m opensg_solid.helper.plate_rm.fea_cell <job> \
           --cell 1 2 --paths p1.dat p2.dat [--pitch 1.0] [--out DIR]

THE CELL TAG.  Offsets encode into filenames as i<|di|>[N]j<|dj|>[N],
N marking a negative (toward -x / -y) offset, plate centre = reference:

    (+1, +2) -> _i1j2        (-2, +3) -> _i2Nj3
    (0, 0)   -> _i0j0        (+1, -1) -> _i1j1N

Two stages: the odb -> rpt dump runs under `abaqus python`
(abq_cell_dump.py, subprocess-launched, cached -- the odb is touched
once per cell); the rpt -> path .SM/.U sampling runs here through
opensg_solid.io.abq_rpt.  Paths are given in CELL-LOCAL coordinates
(the same .dat files the OpenSG side samples) and are shifted by the
cell offset internally.

Out: <out>/fea_<path stem>_<tag>.SM / .U / _elements.csv
     <out>/fea_cell_<tag>.rpt   (the cached odb extraction)
"""
import argparse
import os
import subprocess
import sys

import numpy as np

from opensg_solid.io.abq_rpt import read_rpt, write_path_fields


def cell_tag(di, dj):
    """The filename token of a cell offset.  In: di, dj int.
    Out: str, e.g. (+1, -2) -> 'i1j2N'."""
    return "i%d%sj%d%s" % (abs(di), "N" if di < 0 else "",
                           abs(dj), "N" if dj < 0 else "")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("job", help="odb path WITHOUT .odb")
    p.add_argument("--cell", nargs=2, type=int, required=True,
                   metavar=("DI", "DJ"))
    p.add_argument("--paths", nargs="+", required=True)
    p.add_argument("--pitch", type=float, default=1.0,
                   help="cell pitch [m]; offset*pitch = cell centre")
    p.add_argument("--out", default=".")
    A = p.parse_args()

    di, dj = A.cell
    tag = cell_tag(di, dj)
    cx, cy = di * A.pitch, dj * A.pitch
    os.makedirs(A.out, exist_ok=True)
    rpt = os.path.join(A.out, "fea_cell_%s.rpt" % tag)

    if not os.path.exists(rpt):
        dump = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "abq_cell_dump.py")
        cmd = ["abaqus", "python", dump, A.job, rpt,
               str(cx), str(cy), str(A.pitch / 2.0)]
        print("odb -> rpt:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0 or not os.path.exists(rpt):
            sys.exit("abq_cell_dump failed (rc %d)" % rc)
    else:
        print("using cached %s" % rpt)

    tabs = read_rpt(rpt)
    print("rpt: %d elements, %d nodes"
          % (len(tabs["elem"][0]), len(tabs["node"][0])))
    for pth in A.paths:
        P = np.loadtxt(pth)
        P2 = P.copy()
        P2[:, 1] += cx                    # cell-local -> plate coords
        P2[:, 2] += cy
        stem = os.path.splitext(os.path.basename(pth))[0]
        pref = os.path.join(A.out, "fea_%s_%s" % (stem, tag))
        write_path_fields(tabs, P2, pref,
                          "3-D FEA cell %s (offset %+d, %+d; centred"
                          " %+.3f, %+.3f) along %s"
                          % (tag, di, dj, cx, cy,
                             os.path.basename(pth)))


if __name__ == "__main__":
    main()
