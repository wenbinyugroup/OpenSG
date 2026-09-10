"""step06_fea_path_dat.py -- sample the 3-D FEA cell .rpt along the
comparison paths and write ONE unified .dat per path.

    python -m opensg_solid.helper.Plate_deck_analysis.step06_fea_path_dat \
           <cell rpt> --paths p1.dat p2.dat [--offset DX DY] --out-dir fea

THE UNIFIED LAYOUT (steps 06, 10 and 12 all speak it): whitespace table,
one row per path point,

    s  x  y  z  S11 S22 S33 S12 S13 S23  U1 U2 U3

stress [Pa], displacement [m], s in [0, 1]; comment lines start with #.
x y z are the CELL-LOCAL path coordinates as given, so fea_<stem>.dat
and step10's opensg_<stem>.dat compare row by row.

WHY --offset.  The path .dat files (rows s y1 y2 y3) are CELL-LOCAL.  A
rpt whose coordinates are cell-local too (a cell dumped at the model
origin) needs no shift -- the default 0 0.  A rpt in the whole-plate
frame (a fea_cell_<tag>.rpt of a cell at nonzero (cx, cy), or a
whole-model dump) needs the paths shifted INTO the cell: the offset is
ADDED to y1/y2 before sampling, (DX, DY) = (DI*pitch, DJ*pitch) -- the
same shift plate_rm's fea_cell.py applies internally.  The written x y z
stay cell-local either way.

RELATION TO plate_rm/fea_cell.py.  That module is the whole odb ->
.SM/.U chain: it subprocess-dumps the cell rpt AND samples it into the
plate_rm .SM/.U/_elements.csv trio.  Here the chain is split at the rpt:
step05 dumps, step06 is only the rpt -> dat half, reusing the same
opensg_solid.io.abq_rpt sampling (nearest element centroid for the
piecewise-constant centroidal stress, nearest node for U) but emitting
the single unified table step12 consumes.  The distinct-element count is
printed as the audit that the path actually walks through the mesh.

In:  the step05 cell .rpt + path .dat files (rows s y1 y2 y3,
     cell-local coords)
Out: <out-dir>/fea_<path stem>.dat per path, unified layout; per-path
     point + distinct-element counts printed"""
import argparse
import os
import sys

import numpy as np

from opensg_solid.io.abq_rpt import read_rpt, sample_path

COLS = ("s", "x", "y", "z", "S11", "S22", "S33", "S12", "S13", "S23",
        "U1", "U2", "U3")


def path_table(tabs, P, dx=0.0, dy=0.0):
    """Sample one path and assemble the unified 13-column table.

    In:  tabs -- read_rpt output; P (n, 4) rows [s y1 y2 y3] CELL-LOCAL;
         dx, dy float -- added to y1/y2 before sampling (locates the
         cell inside the rpt frame)
    Out: (T (n, 13) [s x y z S*6 U*3], x y z CELL-LOCAL;
         n_distinct int)."""
    P = np.asarray(P, float)
    if P.ndim == 1:
        P = P[None, :]
    if P.shape[1] < 4:
        raise SystemExit("path needs rows [s y1 y2 y3], got %d columns"
                         % P.shape[1])
    Q = P[:, :4].copy()
    Q[:, 1] += dx
    Q[:, 2] += dy
    check_frame(tabs, Q, dx, dy)
    r = sample_path(tabs, Q)
    T = np.column_stack([P[:, 0], P[:, 1:4], r["S"], r["U"]])
    return T, r["n_distinct"]


def check_frame(tabs, Q, dx, dy):
    """Refuse a path that does not lie inside the rpt's own coordinate box.

    THE FAILURE THIS CATCHES (2026-09-04, cost a full set of wrong figures):
    a fea_cell_<tag>.rpt of an OFF-CENTRE cell keeps JOB coordinates -- cell
    (1,1) spans x[0.503, 1.497] -- while the path .dat is CELL-LOCAL,
    x[-0.446, -0.001].  Sampling without --offset then matches every path
    point to whichever element is nearest the wrong place: the run SUCCEEDS,
    writes a plausible-looking file, and the only visible symptom is the
    distinct-element count collapsing (3 for a 400-point path instead of
    ~140).  Silent, and downstream it reads as a model error rather than a
    frame error.  So this is a hard refusal with the offset to use.

    In:  tabs -- read_rpt output; Q (n, 4) path rows AFTER the offset;
         dx, dy float -- the offset applied, for the message
    Out: None; raises SystemExit when the path is outside the box."""
    exyz = tabs["elem"][1]
    lo, hi = exyz.min(axis=0), exyz.max(axis=0)
    pmin, pmax = Q[:, 1:4].min(axis=0), Q[:, 1:4].max(axis=0)
    span = np.maximum(hi - lo, 1e-30)
    out = (pmin > hi + 0.02 * span) | (pmax < lo - 0.02 * span)
    if not out.any():
        return
    ax = "xyz"[int(np.argmax(out))]
    ctr = 0.5 * (lo + hi)
    raise SystemExit(
        "path lies OUTSIDE the rpt in %s: rpt box x[%.3f, %.3f] y[%.3f,"
        " %.3f] z[%.3f, %.3f], shifted path x[%.3f, %.3f] y[%.3f, %.3f]"
        " z[%.3f, %.3f] (offset applied: %+g, %+g).\n"
        "  The rpt is in the WHOLE-PLATE frame and the paths are"
        " CELL-LOCAL -- pass --offset %.4g %.4g (the cell centre)."
        % (ax, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
           pmin[0], pmax[0], pmin[1], pmax[1], pmin[2], pmax[2],
           dx, dy, ctr[0], ctr[1]))


def write_unified(path, T, src=""):
    """Write one unified path .dat ('#' header + 13 columns).

    In:  path str; T (n, 13); src str -- provenance for the header
    Out: path str (the file written)."""
    hdr = ("3-D FEA path fields -- unified layout%s\n"
           % ((" -- " + src) if src else "")
           + "x y z are the CELL-LOCAL path coordinates;"
           " stress [Pa], displacement [m]\n"
           + " ".join("%14s" % c for c in COLS))
    np.savetxt(path, T, fmt="%14.6e", header=hdr)
    return path


def main(argv=None):
    """CLI -- see the module docstring.

    In:  argv list | None
    Out: int exit status."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("rpt", help="step05 cell .rpt")
    p.add_argument("--paths", nargs="+", required=True,
                   help="path .dat files, rows s y1 y2 y3 (cell-local)")
    p.add_argument("--offset", nargs=2, type=float,
                   default=[0.0, 0.0], metavar=("DX", "DY"),
                   help="ADDED to y1/y2: shifts the paths into the cell"
                        " when the rpt is in the whole-plate frame")
    p.add_argument("--out-dir", default=".")
    a = p.parse_args(argv)

    tabs = read_rpt(a.rpt)
    ne, nn = len(tabs["elem"][0]), len(tabs["node"][0])
    print("%s: %d elements, %d nodes"
          % (os.path.basename(a.rpt), ne, nn))
    if not ne or not nn:
        print("empty table in the rpt")
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    for pth in a.paths:
        P = np.loadtxt(pth)
        T, nd = path_table(tabs, P, a.offset[0], a.offset[1])
        stem = os.path.splitext(os.path.basename(pth))[0]
        out = os.path.join(a.out_dir, "fea_%s.dat" % stem)
        write_unified(out, T, src="%s @ offset (%+g, %+g)"
                      % (os.path.basename(a.rpt), a.offset[0],
                         a.offset[1]))
        print("wrote %s  (%d points, %d distinct elements)"
              % (out, len(T), nd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
